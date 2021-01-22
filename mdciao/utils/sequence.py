r"""

Functions for and around sequence alignment

The alignment takes only in one place (:obj:`my_bioalign`),
the rest of functions either prepare the alignment or produce
other objects derived from it (DataFrames, dictionaries,
maps between topologies etc)

.. currentmodule:: mdciao.utils.sequence


Functions
=========

.. autosummary::
   :toctree: generated/

"""
import numpy as _np
from pandas import DataFrame as _DF
from Bio.pairwise2 import align as _Bioalign
from .lists import contiguous_ranges as _cranges

def print_verbose_dataframe(df):
    r"""
    Print the full dataframe no matter how big

    Parameters
    ----------
    df

    Returns
    -------

    """
    import pandas as _pd
    from IPython.display import display as _display
    with _pd.option_context('display.max_rows', None,
                            'display.max_columns', None,
                            'display.width', 1000):
        _display(df)

def top2seq(top, replacement_letter="X"):
    r"""
    Return the AA sequence of :obj:`top` as a string

    Parameters
    ----------
    top : :obj:`mdtraj.Topology`
    replacement_letter : str, default is "X"
        If the AA has no one-letter-code,
        return this letter instead has to be a str of len(1)

    Returns
    -------
    seq : str of len top.n_residues
    """
    assert len(replacement_letter)==1

    return ''.join([str(rr.code).replace("None",replacement_letter) for rr in top.residues])

def my_bioalign(seq1, seq2,
                method="globalxs",
                argstuple=(-1,0)):
    r"""
    Align two sequences using a method of :obj:`Bioalign`

    Note
    ----
    This is a one-liner wrapper around whatever method
    of :obj:`Bioalign` has been chosen, typically
    pairwise2.align.globalxs

    The intention is to only use *this* method throughout
    mdciao, and change *here* any alignment parameters s.t.
    alignment is done using *always* the same parameters.

    The exposed arguments :obj:`method` and :obj:`argstuple`
    are there for future development but will raise
    NotImplementedErrors if changed.

    See https://biopython.org/DIST/docs/api/Bio.pairwise2-module.html
    for more info

    Parameters
    ----------
    seq1 : str, any length
    seq2 : str, any length
    method : str, default is "globalxs"
    argstuple : tuple, default is (-1,0)
        The

    Returns
    -------
    alignments : list
        A list of tuples, each containing seq1,seq2,score.
        See https://biopython.org/DIST/docs/api/Bio.pairwise2-module.html
        for more info


    """
    allowed_method="globalxs"
    allowed_tuple = (-1,0)
    if method!=allowed_method:
        raise (NotImplementedError("At the moment only %s is "
                                   "allowed as alignment method"%method))
    if argstuple[0]!=-1 or argstuple[1]!=0:
        raise NotImplementedError("At the moment only %s is "
                                   "allowed as argument tuple, got"
                                   "instead %s"%(str(allowed_tuple),
                                                    str(argstuple)))

    return getattr(_Bioalign, method)(seq1, seq2, *argstuple)

def alignment_result_to_list_of_dicts(ialg,
                                      seq_0_res_idxs,
                                      seq_1_res_idxs,
                                      topology_0=None,
                                      topology_1=None,
                                      key_AA_code_seq_0="AA_0",
                                      key_AA_code_seq_1="AA_1",
                                      key_resSeq_seq_0="resSeq_0",
                                      key_idx_seq_0="idx_0",
                                      key_idx_seq_1="idx_1",
                                      key_full_resname_seq_0='fullname_0',
                                      key_full_resname_seq_1='fullname_1',
                                      verbose=False,
                                      ):
    r"""
    Input an alignment result (:obj:`ialg`) and return it as
    a list of per-residue dictionaries with other complementary keys.

    This list of dictionaries is very suitable for further operations
    with :obj:`pandas.DataFrame`.

    TODO
    ----
    Decide whether we need key_resSeq_1 or not

    Parameters
    ----------
    ialg: list
        list with four entries, see obj:`my_bioalign`
        and https://biopython.org/DIST/docs/api/Bio.pairwise2-module.html
        for more info
    topology_0: :obj:`mdtraj.Topology` object
    seq_0_res_idxs:
        Zero-indexed residue indices of whatever was in seq_0
    seq_1_res_idxs:
        Zero-indexed residue indices of whatever was in seq_1
    key_AA_code_seq_0 : str, default is AA_0
        The key under which the residues one-letter code will
        be shown (=the column title in a :obj:`DataFrame`
    key_AA_code_seq_1 : str, default is AA_1
    key_resSeq_seq_0 : str, default is resSeq_0
    key_idx_seq_1
    key_full_resname_seq_0
    seq_0_res_idxs
    verbose: bool, default is False

    Returns
    -------
    alignment_dict : dictionary
        A dictionary containing the aligned sequences with annotated with different information

    """
    # Unpack the alignment
    top_0_seq, top_1_seq = ialg[0], ialg[1]
    # Some sanity checks
    assert len(top_0_seq) == len(top_1_seq)

    # Do we have the right indices?
    assert len(seq_1_res_idxs)==len(''.join([ii for ii in top_1_seq if ii.isalpha()]))
    assert len(seq_0_res_idxs)==len(''.join([ii for ii in top_0_seq if ii.isalpha()]))

    # Create needed iterators
    seq_1_res_idxs_iterator = iter(seq_1_res_idxs)
    idx_seq_0_iterator = iter(seq_0_res_idxs)

    top_0_resSeq_iterator = None
    resname_top_0_iterator = None
    if topology_0 is not None:
        top_0_resSeq_iterator = iter([topology_0.residue(ii).resSeq for ii in seq_0_res_idxs])
        resname_top_0_iterator = iter([str(topology_0.residue(ii)) for ii in seq_0_res_idxs])

    resname_top_1_iterator = None
    if topology_1 is not None:
        resname_top_1_iterator = iter([str(topology_1.residue(ii)) for ii in seq_1_res_idxs])

    alignment_dict = []
    for rt, rr in zip(top_0_seq, top_1_seq):
        alignment_dict.append({key_AA_code_seq_0: rt,
                               key_AA_code_seq_1: rr,
                               key_resSeq_seq_0: '~',
                               key_full_resname_seq_0: '~',
                               key_full_resname_seq_1: '~',
                               key_idx_seq_1: '~',
                               key_idx_seq_0: '~'})

        if rt.isalpha():
            if topology_0 is not None:
                alignment_dict[-1][key_full_resname_seq_0] = next(resname_top_0_iterator)
                alignment_dict[-1][key_resSeq_seq_0] = next(top_0_resSeq_iterator)
            alignment_dict[-1][key_idx_seq_0] = next(idx_seq_0_iterator)

        if rr.isalpha():
            alignment_dict[-1][key_idx_seq_1] = next(seq_1_res_idxs_iterator)
            if topology_1 is not None:
                alignment_dict[-1][key_full_resname_seq_1] = next(resname_top_1_iterator)

    # Add a field for matching vs nonmatching AAs
    for idict in alignment_dict:
        idict["match"] = False
        if idict[key_AA_code_seq_0]==idict[key_AA_code_seq_1]:
            idict["match"]=True

    if verbose:
        print("\nAlignment:")
        order = [key_idx_seq_1, key_AA_code_seq_1, key_full_resname_seq_1,
                                key_AA_code_seq_0, key_resSeq_seq_0, key_full_resname_seq_0, key_idx_seq_0, "match"]
        print_verbose_dataframe(_DF(alignment_dict)[order])

    return alignment_dict


def align_tops_or_seqs(top0, top1, substitutions=None,
                       seq_0_res_idxs=None,
                       seq_1_res_idxs=None,
                       return_DF=True,
                       verbose=False,
                       ):
    r""" Align two sequence-containing objects, i.e. strings and/or
    :obj:`mdtraj.Topology` objects

    Returns a :obj:`pandas.DataFrame`

    Relevant methods used under the hood are :obj:`my_bioalign` and
    :obj:`alignment_result_to_list_of_dicts`, see their docs
    for more info

    Parameters
    ----------
    top0 : :str or obj:`mdtraj.Topology`
    top1 : :str or obj:`mdtraj.Topology`
    substitutions : dictionary
        dictionary of patterns and replacements,
        in case some AAs of the topologies
    seq_0_res_idxs : iterable of integers, default is None
        only use these idxs for alignment in :obj:`top0`
    seq_1_res_idxs : iterable of integers, default is None
        only use these idxs for alignment in :obj:`top1`
    return_DF : bool, default is True
        If false, a list of alignment dictionaries instead
        of a dataframe will be returned
    verbose : bool, default is False

    Returns
    -------
    align : :obj:`pandas.DataFrame`
        See :obj:`alignment_result_to_list_of_dicts` for more info


    """
    if isinstance(top0, str):
        top0_seq = top0
        n_res_0 = len(top0_seq)
        top04a = None
    else:
        top0_seq = top2seq(top0)
        n_res_0 = top0.n_residues
        top04a = top0

    if isinstance(top1, str):
        top1_seq = top1
        n_res_1 = len(top1_seq)
        top14a = None
    else:
        n_res_1 = top1.n_residues
        top1_seq = top2seq(top1)
        top14a = top1

    if substitutions is not None:
        for key, val in substitutions.items():
            top0_seq = top0_seq.replace(key,val)
            top1_seq = top1_seq.replace(key,val)
            #print(key,val)

    if seq_0_res_idxs is None:
        seq_0_res_idxs=_np.arange(n_res_0, dtype=int)
    if seq_1_res_idxs is None:
        seq_1_res_idxs=_np.arange(n_res_1, dtype=int)

    top0_seq = "".join([top0_seq[ii] for ii in seq_0_res_idxs])
    top1_seq = "".join([top1_seq[ii] for ii in seq_1_res_idxs])

    align_list = alignment_result_to_list_of_dicts(my_bioalign(top0_seq, top1_seq)[0],
                                                   topology_0=top04a,
                                                   seq_0_res_idxs=seq_0_res_idxs,
                                                   seq_1_res_idxs=seq_1_res_idxs,
                                                   topology_1=top14a,
                                                   verbose=verbose,
                                                   )

    if return_DF:
        return _DF(align_list)
    else:
        return align_list



def maptops(top0,
            top1,
            allow_nonmatch=False,
            ):
    r""" map residues between topologies or sequences
    via their serial indices a sequence alignment

    Parameters
    ----------
    top0 : :obj:`~mdtraj.Topology` or str
    top1:  :obj:`~mdtraj.Topology` or str
    allow_nonmatch : bool, default is False
        If true, non-matches of
        equal length will be
        considered matches

    Returns
    -------
    top0_to_top1 : dict
        top0_to_top1[10] = 20
    top1_to_top0 : dict
        top1_to_top0[20] = 10

    """
    df = align_tops_or_seqs(top0, top1,
                            return_DF=True)

    return df2maps(df,allow_nonmatch=allow_nonmatch)

def df2maps(df, allow_nonmatch=True):
    r"""Map the columns "idx_0" and "idx_1" of an alignment
    (a :obj:`pandas.DataFrame`)

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Typically comes from  :obj:`align_tops_or_seqs`
    allow_nonmatch : bool, default is True
        Allow to map between ranges of residues that
        don't match, as long as nonmatching
        ranges are equal in length, s.t.
        A A
        A A
        B D
        B D
        C C
        C C
        maps BB to DD

    Non-matching first or last ranges will never be
    mapped

    Returns
    -------
    top0_to_top1 : dict
        top0_to_top1[10] = 20
    top1_to_top0 : dict
        top1_to_top0[20] = 10

    """
    if allow_nonmatch:
        _df = re_match_df(df)
    else:
        _df = df

    top0_to_top1 = {key: val for key, val in zip(_df[_df["match"] == True]["idx_0"].to_list(),
                                                 _df[_df["match"] == True]["idx_1"].to_list())}

    top1_to_top0 = {val:key for key, val in top0_to_top1.items()}

    return top0_to_top1, top1_to_top0

def re_match_df(df):
    r"""
    Return a copy of an alignment :obj:`pandas.Dataframe` with True 'match'-values
    for non-matching blocks that have equal length.

    For instance,
        A A True
        A A True
        B D False
        B D False
        C C True
        C C True
    gets re_matched to:
        A A True
        A A True
        B D True
        B D True
        C C True
        C C True

    The input :obj:`DataFrame` is left untouched and only a copy is returned


    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Typically comes from  :obj:`align_tops_or_seqs`

    Returns
    -------
    _df : :obj:`pandas.DataFrame`
        A re_matched copy of :obj:`df`

    """

    match_ranges = _cranges(df["match"].values)
    _df = df.copy()
    if False in match_ranges.keys():
        for rr in match_ranges[False]:
            try:
                if all(_df.loc[[rr[0] - 1, rr[-1] + 1]]["match"]) and \
                        all(["-" not in df[key].values[rr] for key in ["AA_0",
                                                                   "AA_1"]]):  # this checks for no insertions in the alignment ("=equal length ranges")
                    _df.at[rr, "match"] = True
            except KeyError:
                continue

    return _df

def superpose_w_CA_align(geom, ref,
                         res_indices=None,
                         ref_res_indices=None,
                         verbose=False,
                         allow_nonmatch=False):
    r"""
    Pre align on CA-atoms before calling :obj:`mdtraj.Trajectory.superpose`

    Changes :obj:`geom` in place and returns it as well

    Parameters
    ----------
    geom : :obj:`~mdtraj.Trajectory`
    ref : :obj:`~mdtraj.Trajectory`
    res_indices : iterable of ints, default is None
        Use only these indices for the sequence alignment
    ref_res_indices : iterable of ints, default is None
        Use only these indices for the sequence alignment
    allow_nonmatch : bool, default is True
        Allow to map between ranges of residues that
        don't match, as long as nonmatching
        ranges are equal in length, s.t.
        A A
        A A
        B D
        B D
        C C
        C C
        maps BB to DD

    Non-matching first or last ranges will never be
    mapped

    Returns
    -------
    geom : :obj:`~mdtraj.Trajectory`

    """
    df = align_tops_or_seqs(geom.top, ref.top,
                            seq_0_res_idxs=res_indices,
                            seq_1_res_idxs=ref_res_indices)

    g2rmap, _  = df2maps(df, allow_nonmatch=allow_nonmatch)
    if verbose:
        print_verbose_dataframe(df)
    g_ats, r_ats = [],[]
    for key, val in g2rmap.items():
        g, r = None, None
        try:
            g = geom.top.residue(key).atom("CA").index
            r = ref.top.residue(val).atom("CA").index
        except KeyError:
            pass
        if None not in [g,r]:
            g_ats.append(g)
            r_ats.append(r)
    geom.superpose(ref,atom_indices=g_ats, ref_atom_indices=r_ats)
    return geom
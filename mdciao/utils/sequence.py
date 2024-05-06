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
from .lists import contiguous_ranges as _cranges
import pandas as _pd
from IPython.display import display as _display
from collections import namedtuple as _namedtuple
from Bio import Align as _BioAlign


# See "Define original properties" https://pandas.pydata.org/pandas-docs/stable/development/extending.html#define-original-properties
class _ADF(_DF):
    r"""
    Sub-class of an :obj:`~pandas.DataFrame` to include the alignment_score as metadata.

    It can be then accessed via self.alignment_score and is preserved downstream

    Check https://pandas.pydata.org/pandas-docs/stable/development/extending.html#define-original-properties
    for more info
    """

    # normal properties
    _metadata = ["alignment_score"]

    @property
    def _constructor(self):
        return _ADF


class AlignmentDataFrame(_ADF):
    r"""
    Sub-class of an :obj:`~pandas.DataFrame` to include the alignment_score as metadata.

    Simply pass it as argument ' alignment_score=1' and it:
     * can be then accessed via self.alignment_score and
     * it is preserved downstream after operating on the df

    Check https://pandas.pydata.org/pandas-docs/stable/development/extending.html#define-original-properties
    for more info
    """

    def __init__(self,*args,**kwargs):
        alignment_score = kwargs.get("alignment_score")
        if alignment_score is not None:
            kwargs.pop("alignment_score")
        super().__init__(*args,**kwargs)
        self.alignment_score = alignment_score

def print_verbose_dataframe(df):
    r"""
    Print the full dataframe no matter how big

    Parameters
    ----------
    df

    Returns
    -------

    """
    rows, columns = df.shape
    with _pd.option_context('display.max_rows', rows,
                            'display.max_columns', columns,
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
                method="global",
                match=1, mismatch=0, open_gap_score=-1, extend_gap_score=-0.05,
                n_max=1000
                ):
    r"""
    Align two sequences using :obj:`Bio.Align.PairwiseAligner`

    Note
    ----
    The intention is to only use *this* method throughout
    mdciao, and change *here* any alignment parameters s.t.
    alignment is done using *always* the same parameters.

    See https://biopython.org/docs/1.75/api/Bio.Align.html?#Bio.Align.PairwiseAligner
    for more info

    Parameters
    ----------
    seq1 : str, any length
    seq2 : str, any length
    method : str, default is "global"
        Gets passed as argument "mode" to
        the underlying :obj:~`Bio.Align.PairwiseAligner`
        At the moment, any other value will raise NotImplementedError.
    match : int or float, default is 1
        Score value for a match.
        At the moment, any other value will raise NotImplementedError.
    mismatch : int or float, default is 0
        Penalty value (non-positve score) for a mismatch.
        At the moment, any other value will raise NotImplementedError.
    open_gap_score : int or float, default is -1
        Penalty value (non-positve score) for opening a gap.
        At the moment, any other value will raise NotImplementedError.
    extend_gap_score : int or float, default is 0.05
        Penalty value (non-positve score) for extending a gap.
        At the moment, any other value will raise NotImplementedError.
    n_max : int, default is 1000
        The maximum number of returned alignments.


    Returns
    -------
    alignments : list
        A list of namedtuples, each containing seq1,seq2,score.

    """
    _my_alg = _namedtuple("NamedTuplePairwiseAlignments", ["seq1", "seq2", "score"])

    # This is to be able to raise the NotImplemented but also to hard-code the only allowed method here
    allowed_method="global"
    end_gap_score = 0 # equivalent to the old "penalize_end_gaps": False in pairwise2
    if method!=allowed_method:
        raise (NotImplementedError("At the moment only %s is "
                                   "allowed as alignment method"%method))

    allowed_kwargs = {"match": 1, "mismatch": 0, "open_gap_score": -1, "extend_gap_score": -0.05}

    provided_kwargs = {key : val for key, val in locals().items() if key in allowed_kwargs}
    if allowed_kwargs == provided_kwargs:
        a = _BioAlign.PairwiseAligner(mode=method,
                                      match=match, mismatch=mismatch, open_gap_score=open_gap_score,
                                      extend_gap_score=extend_gap_score, end_gap_score=end_gap_score)
        alignments = [a for __, a in zip(range(n_max), a.align(seq1,seq2))]
        scores = _np.array([a.score for a in alignments])

        # Some edge cases in the tests produce alignments with no overlap at all, i.e. with an empty a.aligned attribute,
        # that are equally scored with other alignments, e.g.
        # ---X       --X
        # ----  vs   ---
        # IWN-       IWN
        # have the same scores b.c. mismatches are valued with 0

        # These alignments are badly scored alignments in "guess" mode that can be discarded safely
        # Hence, we mark the start of the alignment as _np.inf s.t. they sink
        # to the bottom the list of equally (badly) scored alignments
        starting_alignment_idxs = [[a.aligned[1][0][0] if len(a.aligned[1])>0 else _np.inf][0] for a in alignments]

        # TODO
        # Then comes this very interim solution, to not touch the test-suite ATM, checkout
        # this issue to find out why this reordering https://github.com/biopython/biopython/issues/4360

        # In each block of equally scored alignments, sort by ascending order
        # of first match-index for sequence 2
        order = _np.lexsort((starting_alignment_idxs, -scores))
        # reorder
        alignments = [alignments[ii] for ii in order[:n_max]]
        scores = [a.score for a in alignments] #reorder scores
        if hasattr(alignments[0],"sequences"): # some more backward compatibility
            seqs = [a._format_generalized().replace(" ","").splitlines() for a in alignments]
        else:
            seqs = [a.format().splitlines() for a in alignments]

        algs = [_my_alg(s[0],s[-1],score) for s,score in zip(seqs,scores)]
        return algs
    else:
        raise NotImplementedError(f"At the moment, the keyword arguments {list(allowed_kwargs.keys()) }are exposed"
                                  f"to make them highly visible, but their values can't be changed from {allowed_kwargs}."
                                  f"The input was instead {provided_kwargs}")

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
    Input an alignment result `ialg` and return it as
    a list of per-residue dictionaries with other complementary keys.


    This list of dictionaries is very suitable for further operations
    with :obj:`pandas.DataFrame`.

    TODO
    ----
    Decide whether we need key_resSeq_1 or not

    Parameters
    ----------
    ialg: namedtuple
        See return value of obj:`my_bioalign`
        for more info
        seq_0_res_idxs:
        Zero-indexed residue indices of whatever was in seq_0
    seq_1_res_idxs:
        Zero-indexed residue indices of whatever was in seq_1
    topology_0: :obj:`~mdtraj.Topology` object, default is None
    topology_1: :obj:`~mdtraj.Topology` object, default is None
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
    assert len(seq_1_res_idxs)==len(''.join([ii for ii in top_1_seq if ii!="-"]))
    assert len(seq_0_res_idxs)==len(''.join([ii for ii in top_0_seq if ii!="-"]))

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

        if rt!="-":
            if topology_0 is not None:
                alignment_dict[-1][key_full_resname_seq_0] = next(resname_top_0_iterator)
                alignment_dict[-1][key_resSeq_seq_0] = next(top_0_resSeq_iterator)
            alignment_dict[-1][key_idx_seq_0] = next(idx_seq_0_iterator)

        if rr!="-":
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
    :obj:`~mdtraj.Topology` objects

    Returns a list of :obj:`n_best` :obj:`AlignmentDataFrame` s,
    an mdciao sub-class of a :obj:`~pandas.DataFrame`

    A list is returned because sometimes there's more than
    one alignment with the best possible score (currently it's
    limited to 10 alignments)

    Relevant methods used under the hood are :obj:`my_bioalign` and
    :obj:`alignment_result_to_list_of_dicts`, see their docs
    for more info.

    Parameters
    ----------
    top0 : :str or obj:`~mdtraj.Topology`
    top1 : :str or obj:`~mdtraj.Topology`
    substitutions : dictionary
        dictionary of patterns and replacements,
        in case some AAs of the topologies
    seq_0_res_idxs : iterable of integers, default is None
        only use these idxs for alignment in :obj:`top0`
    seq_1_res_idxs : iterable of integers, default is None
        only use these idxs for alignment in :obj:`top1`
    return_DF : bool, default is True
        If False, a list of alignment dictionaries instead
        of :obj:`AlignmentDataFrame` s will be returned
    verbose : bool, default is False

    Returns
    -------
    alignments : list of :obj:`n_best` :obj:`AlignmentDataFrame` s
        These are just normal :obj:`~pandas.DataFrames` with an extra
        attribute .alignment_score to be used downstream.
        If :obj:`return_DF` is False, it's a list of lists of dicts,
        see :obj:`alignment_result_to_list_of_dicts` for more info


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

    alignments = my_bioalign(top0_seq, top1_seq)[:10]
    alignments = [aa for aa in alignments if aa.score == alignments[0].score]
    scores = [aa.score for aa in alignments]
    lists_of_lists_of_align_dicts = [alignment_result_to_list_of_dicts(aa,
                                                   topology_0=top04a,
                                                   seq_0_res_idxs=seq_0_res_idxs,
                                                   seq_1_res_idxs=seq_1_res_idxs,
                                                   topology_1=top14a,
                                                   verbose=verbose,
                                                   ) for aa in alignments]

    if return_DF:
        return [AlignmentDataFrame(aa, alignment_score=score) for aa, score in zip(lists_of_lists_of_align_dicts,
                                                                                  scores)]
    else:
        return lists_of_lists_of_align_dicts



def maptops(top0,
            top1,
            allow_nonmatch=False,
            ):
    r""" Use pairwise sequence alignment to produce maps of residue indices between topologies or sequences

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
                            return_DF=True)[0]

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
                    _df.loc[rr, "match"] = True
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
                            seq_1_res_idxs=ref_res_indices)[0]

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

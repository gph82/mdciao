import numpy as _np
from pandas import DataFrame as _DF
from Bio.pairwise2 import align as _Bioalign

from collections import defaultdict as _defdict

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
    Return the AA sequence of :obj:`top `as a string
    Parameters
    ----------
    top : :obj:`mdtraj.Topology`
    replacement_letter : str, default is "X"
        Has to be a str of len(1)

    Returns
    -------
    seq : str of len top.n_residues
    """
    assert len(replacement_letter)==1

    return ''.join([str(rr.code).replace("None",replacement_letter) for rr in top.residues])

def _my_bioalign(seq1,seq2,
                 method="globalxs",
                 argstuple=(-1,0)):
    r"""
    Align two sequences using a method of :obj:`Bioalign`

    Note
    ----
    This is a one-liner wrapper around whatever method
    of :obj:`Bioalign` has been chosen.

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

    Returns
    -------
    An alignment dictionary

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

def alignment_result_to_list_of_dicts(ialg, topology_0,
                                      seq_0_res_idxs,
                                      seq_1_res_idxs,
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

    TODO : decide whether we need key_resSeq_1 or not

    Parameters
    ----------
    ialg: list
        list with four entries, see obj:`_my_bioalign`
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
    top_0_resSeq_iterator = iter([topology_0.residue(ii).resSeq for ii in seq_0_res_idxs])
    seq_1_res_idxs_iterator = iter(seq_1_res_idxs)
    idx_seq_0_iterator = iter(seq_0_res_idxs)
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
            alignment_dict[-1][key_resSeq_seq_0] = next(top_0_resSeq_iterator)
            alignment_dict[-1][key_full_resname_seq_0] = next(resname_top_0_iterator)
            alignment_dict[-1][key_idx_seq_0] = next(idx_seq_0_iterator)

        if rr.isalpha():
            alignment_dict[-1][key_idx_seq_1] = next(seq_1_res_idxs_iterator)
            if resname_top_1_iterator is not None:
                alignment_dict[-1][key_full_resname_seq_1] = next(resname_top_1_iterator)

    # Add a field for matching vs nonmatching AAs
    for idict in alignment_dict:
        idict["match"] = False
        if idict[key_AA_code_seq_0]==idict[key_AA_code_seq_1]:
            idict["match"]=True

    if verbose:
        print("\nAlignment")
        order = [key_idx_seq_1, key_AA_code_seq_1, key_full_resname_seq_1,
                                key_AA_code_seq_0, key_resSeq_seq_0, key_full_resname_seq_0, key_idx_seq_0, "match"]
        print_verbose_dataframe(_DF(alignment_dict)[order])

    return alignment_dict

'''
def _align_tops(top0, top1, substitutions=None,
                seq_0_res_idxs=None,
                seq_1_res_idxs=None,
                return_DF=True):
    r"""
    Provided two :obj:`mdtraj.Topology` objects,
    return their alignment as a :obj:`pandas.DataFrame`.

    Relevant methods used under the hood are :obj:`_my_bioalign` and
    :obj:`alignment_result_to_list_of_dicts`

    Parameters
    ----------
    top0 : :obj:`mdtraj.Topology`
    top1 : :obj:`mdtraj.Topology`
    substitutions : dictionary
        dictionary of patterns and replacements,
        in case some AAs of the topologies
    seq_0_res_idxs : iterable of integers, default is None
    seq_1_res_idxs : iterable of integers, default is None
    Returns
    -------
    align : :obj:`pandas.DataFrame`
        See :obj:`alignment_result_to_list_of_dicts` for more info


    """
    top0_seq = ''.join([str(rr.code).upper() for rr in top0.residues])
    top1_seq = ''.join([str(rr.code).upper() for rr in top1.residues])

    my_subs = {"NONE":"X"}
    if substitutions is not None:
        my_subs.update(substitutions)
    for key, val in my_subs.items():
        top0_seq = top0_seq.replace(key,val)
        top1_seq = top1_seq.replace(key,val)
        #print(key,val)

    if seq_0_res_idxs is None:
        seq_0_res_idxs=_np.arange(top0.n_residues, dtype=int)
    if seq_1_res_idxs is None:
        seq_1_res_idxs=_np.arange(top1.n_residues, dtype=int)

    print(seq_0_res_idxs)
    top0_seq = "".join([top0_seq[ii] for ii in seq_0_res_idxs])
    top1_seq = "".join([top1_seq[ii] for ii in seq_1_res_idxs])

    align_list = alignment_result_to_list_of_dicts(_my_bioalign(top0_seq, top1_seq)[0],
                                                   top0,
                                                   seq_0_res_idxs=seq_0_res_idxs,
                                                   seq_1_res_idxs=seq_1_res_idxs,
                                                   topology_1=top1,
                                                   )

    if return_DF:
        return _DF(align_list)
    else:
        return align_list
'''

'''
def _align_tops_2_dicts(top0,top1,
                        fail_on_key_redundancies=False):
    alignm_list = _align_tops(top0, top1,
                              return_DF=False)

    matches = [al for al in alignm_list if al["match"]]
    # print(matches)
    key_0 = [al["fullname_0"] for al in matches]
    key_1 = [al["fullname_1"] for al in matches]
    if fail_on_key_redundancies:
        assert len(key_0) == len(_np.unique(key_0)), (len(key_0), len(_np.unique(key_0) ))
        assert len(key_1) == len(_np.unique(key_1)), (len(key_1), len(_np.unique(key_1) ))

    AAtop0toAAtop1 = _defdict(list)
    AAtop1toAAtop0 = _defdict(list)
    for idict in matches:
        AA0 = idict["fullname_0"]
        AA1 = idict["fullname_1"]
        AAtop0toAAtop1[AA0].append(idict)
        AAtop1toAAtop0[AA1].append(idict)

    return [{key: val for key, val in idict.items()} for idict in [AAtop0toAAtop1, AAtop1toAAtop0]]
'''

'''
# todo this is a bit of overkill, one alignment per residue
def residx_in_seq_by_alignment(ridx1, top1,
                               top2,
                               subset_1=None,
                               verbose=False,
                               fail_if_no_match=True):
    r"""
    For a given residue index in a given topology, return the equivalent
    residue index in a second topology
    Parameters
    ----------
    ridx1: int
    top1: mdtraj.Topology
    top2: mdtraj.Topology
    subset_1: iterable of integers, default is None
        Restrict the alignment to these indices of top1

    Returns
    -------
    ridx2: int
        Index so that top2.residue(ridx2) will return the residue equivalent
        to top1.residue(ridx1)

    """
    if subset_1 is None:
        subset_1 = _np.arange(top1.n_residues)

    assert ridx1 in subset_1

    top1_seq, top2_seq = [''.join([str(rr.code).replace("None","X") for rr in itop.residues])
                        for itop in [top1, top2]]

    top1_seq = ''.join([top1_seq[ii] for ii in subset_1])
    ires = alignment_result_to_list_of_dicts(
        _my_bioalign(top1_seq, top2_seq)[0],
        top1, subset_1,
        _np.arange(top2.n_residues),
        )

    idf = _DF(ires)
    if verbose:
        print_verbose_dataframe(idf)
    jdf = idf[idf["idx_0"] == _np.argwhere(subset_1==ridx1)[0,0]]
    #print(jdf)
    assert len(jdf)==1
    if not list(jdf["match"])[0]==True:
        #print("Sorry, no match after trying to align %s!"%top1.residue(ridx1))
        print("Sorry, no match after trying to align %s!"%top1.residue(ridx1))
        if fail_if_no_match:
            print_verbose_dataframe(idf)
            raise Exception
        else:
            print("returning other stuff")
            print(jdf)

    return list(jdf["idx_1"])[0]
'''
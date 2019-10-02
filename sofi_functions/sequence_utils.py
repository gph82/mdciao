from pandas import DataFrame as _DF

from Bio.pairwise2 import align as _Bioalign

def _print_verbose_dataframe(idf):
    import pandas as _pd
    with _pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(idf)

def _my_bioalign(seq1,seq2):
    return _Bioalign.globalxs(seq1, seq2, -1,0)

def alignment_result_to_list_of_dicts(ialg, topology_0,
                                      seq_0_res_idxs,
                                      seq_1_res_idxs,
                                      AA_code_seq_0_key="AA_0",
                                      AA_code_seq_1_key="AA_1",
                                      resSeq_seq_0_key="resSeq_0",
                                      idx_seq_0_key="idx_0",
                                      idx_seq_1_key="idx_1",
                                      full_resname_seq_0_key='fullname_0',
                                      verbose=False,
                                      ):
    r"""
    Provide an alignment result and return the result as a list of dictionaries
    suitable for operations with :obj:`pandas.DataFrame`

    Parameters
    ----------
    ialg: list
        list with four entries, see the return value of :obj:`Bio.pairwise.align.globalxx`
        It is assumed that some form of a call like ```globalxx(seq_0,seq_1)``` was issued.
        See their doc for more details
    topology_0: :obj:`mdtraj.Topology` object
        In this context, target means "not reference"
    seq_0_res_idxs:
        Zero-indexed residue indices of seq_0
    seq_1_res_idxs:
        Zero-indexed residue indices of seq_1
    AA_code_seq_0_key
    AA_code_seq_1_key
    resSeq_seq_0_key
    idx_seq_1_key
    full_resname_seq_0_key
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
    idx_seq_0_iterator = iter(_np.arange(len(seq_0_res_idxs)))
    resname_top_0_iterator = iter([str(topology_0.residue(ii)) for ii in seq_0_res_idxs])

    alignment_dict = []
    for rt, rr in zip(top_0_seq, top_1_seq):
        alignment_dict.append({AA_code_seq_0_key: rt,
                               AA_code_seq_1_key: rr,
                               resSeq_seq_0_key: '~',
                               full_resname_seq_0_key: '~',
                               idx_seq_1_key: '~',
                               idx_seq_0_key: '~'})

        if rt.isalpha():
            alignment_dict[-1][resSeq_seq_0_key] = next(top_0_resSeq_iterator)
            alignment_dict[-1][full_resname_seq_0_key] = next(resname_top_0_iterator)
            alignment_dict[-1][idx_seq_0_key] = next(idx_seq_0_iterator)

        if rr.isalpha():
            alignment_dict[-1][idx_seq_1_key] = next(seq_1_res_idxs_iterator)

    # Add a field for matching vs nonmatching AAs
    for idict in alignment_dict:
        idict["match"] = False
        if idict[AA_code_seq_0_key]==idict[AA_code_seq_1_key]:
            idict["match"]=True

    if verbose:
        print("\nAlignment dicts:")
        order = [idx_seq_1_key, AA_code_seq_1_key, AA_code_seq_0_key, resSeq_seq_0_key, full_resname_seq_0_key, "match"]
        print(_DF(alignment_dict)[order].to_string())

    return alignment_dict
import numpy as _np
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
        _print_verbose_dataframe(idf)
    jdf = idf[idf["idx_0"] == _np.argwhere(subset_1==ridx1)[0,0]]
    #print(jdf)
    assert len(jdf)==1
    if not list(jdf["match"])[0]==True:
        #print("Sorry, no match after trying to align %s!"%top1.residue(ridx1))
        print("Sorry, no match after trying to align %s!"%top1.residue(ridx1))
        if fail_if_no_match:
            _print_verbose_dataframe(idf)
            raise Exception
        else:
            print("returning other stuff")
            print(jdf)

    return list(jdf["idx_1"])[0]
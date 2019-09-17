import numpy as _np
from pandas import DataFrame as _DF
def alignment_result_to_list_of_dicts(ialg, target_top, refseq_idxs,
                                      res_top_key="res_top",
                                      res_ref_key="res_ref",
                                      resSeq_key="resSeq",
                                      idx_key="idx",
                                      resname_key='fullname',
                                      subset_of_residxs=None,
                                      re_merge_skipped_entries=True,
                                      verbose=True
                                      ):
    r"""
    Provide an alignment result and return the result as a list of dictionaries
    suitable for operations with :obj:`pandas.DataFrame`

    Parameters
    ----------
    ialg: list
        list with four entries, see the return value of :obj:`Bio.pairwise.align.globalxx`
        for mor details
    target_top: :obj:`mdtraj.Topology` object
        In this context, target means "not reference"
    refseq_idxs:
        Zero-indexed residue indices (in the reference topology) to be used
    res_top_key
    res_ref_key
    resSeq_key
    idx_key
    resname_key
    subset_of_residxs
    re_merge_skipped_entries: boolean, default is True
        #TODO is this really a good strategy? Do the method's argument allow to avoid this?

        :obj:`Bio.pairwise.align.globalxx` returns the aligned sequences in such way
        that even discrepancies of one AA, are "skipped" and end up shifting the
        aligned sequences by one position:
        AAABAAA vs. AAACAAA yields:

        * AAAB~AAA
        * AAA~CAAA
        Setting this boolean to True will try to "merge" this paddings ("~") so that the
        alignement results are:

        * AAABAAA
        * AAACAAA



    verbose

    Returns
    -------
    alignment_dict : dictionary
        A dictionary containing the aligned sequences with annotated with different information

    """

    # TODO refactor this so that the alignment is somehow clear
    if subset_of_residxs is None:
        subset_of_residxs = _np.arange(target_top.n_residues)
        #subset_of_residxs = _np.arange(_np.max((ialg[0], ialg[1])))

    top_resSeq_iterator = iter([target_top.residue(ii).resSeq for ii in subset_of_residxs])
    refseq_idxs_iterator = iter(refseq_idxs)
    resname_iterator = iter([str(target_top.residue(ii)) for ii in subset_of_residxs])
    itop_seq, iref_seq = ialg[0], ialg[1]
    assert len(itop_seq) == len(iref_seq)
    alignment_dict = []
    for rt, rr in zip(itop_seq, iref_seq):
        alignment_dict.append({res_top_key: rt,
                               res_ref_key: rr,
                               resSeq_key: '~',
                               resname_key:'~',
                               idx_key:    '~'})

        if rt.isalpha():
            alignment_dict[-1][resSeq_key] = next(top_resSeq_iterator)
            alignment_dict[-1][resname_key] = next(resname_iterator)

        if rr.isalpha():
            alignment_dict[-1][idx_key] = next(refseq_idxs_iterator)

    if re_merge_skipped_entries:
        pairs_to_merge, list_of_new_merged_dicts = find_mergeable_positions(alignment_dict,
                                                                            res_ref_key=res_ref_key,
                                                                            res_top_key=res_top_key,
                                                                            resSeq_key=resSeq_key,
                                                                            idx_key=idx_key,
                                                                            resname_key=resname_key,
                                                                            verbose=verbose,
                                                                            context=5,
                                                                            )
        alignment_dict = insert_mergeable_positions(alignment_dict, pairs_to_merge, list_of_new_merged_dicts)
        if verbose:
            print("\nFinal alignment after merging")
            order = [idx_key, res_ref_key, res_top_key, resSeq_key, resname_key]
            print(_DF(alignment_dict)[order].to_string())


    for idict in alignment_dict:
        idict["match"] = False
        if idict[res_top_key]==idict[res_ref_key]:
           idict["match"]=True

    return alignment_dict

def find_mergeable_positions(list_of_alignment_dicts,
                             context=2,
                             idx_key="idx",
                             resSeq_key="resSeq",
                             res_top_key="res_top",
                             res_ref_key="res_ref",
                             resname_key='fullname',
                             verbose=False,
                             ):
    r"""
    Takes the out dictionary of :obj:`alignment_result_to_list_of_dicts` and
    finds "mergeable" positions.

    These are positions in the alignment where it is desirable that:

    * 12345678
    * AAAB~AAA
    * AAA~CAAA

    be corrected to:

    * 1234567
    * AAABAAA
    * AAACAAA


    Parameters
    ----------
    list_of_alignment_dicts: list
        list containing alignment dictionaries as they are outputed by
        :obj:`alignment_result_to_list_of_dicts`
    context: int, default 2
        The maximum length of the mergeable ('~') position,
        e.g. 1 in the above alignment
    idx_key
    resSeq_key
    res_top_key
    res_ref_key
    resname_key
    verbose


    Returns
    -------
    pairs_to_merge, list_of_merged_dicts_for_insertion
    pairs_to_merge: list
        pairs of positions of the alignment that have
        been identified as "mergeable" positions, e.g.
        [[4,5]] in the above example
    list_of_merged_dicts_for_insertion: list
        list of alignment dictionaries containing only
        the "merged" position, to be inserted
        in the original alignment, e.g.:
        {'fullname': 'Bee',
        'res_top': 'B'
        'idx': 4,
        'resSeq': 300, #this idx is meaningless here
        'res_ref': 'C',
        }
        # TODO define the alignment dict somewhere


    """
    #TODO test
    dict_pairs_to_merge = []
    pairs_to_merge = []
    for ii, idict in enumerate(list_of_alignment_dicts):
        if idict[idx_key] == '~' and idict[res_ref_key] == '-':
            idict_next = list_of_alignment_dicts[ii + 1]
            if idict_next[resname_key] == '~' and idict_next[resSeq_key] == '~' and idict_next[res_top_key] == '-':
                pairs_to_merge.append([ii, ii + 1])
                dict_pairs_to_merge.append([idict, idict_next])


    # TODO: creating the DF structure two times here just for visualitzaiton purposes
    dataframevis = _DF.from_dict(list_of_alignment_dicts)
    order = [res_ref_key, res_top_key, resSeq_key, resname_key]
    #print(dataframevis)
    if verbose:
        print("The following merges will take place:")
    list_of_merged_dicts_for_insertion = []
    for pair_idxs, (idict, idict_next) in zip(pairs_to_merge, dict_pairs_to_merge):
        new_dict = {resname_key: idict[resname_key],
                    idx_key: idict_next[idx_key],
                    resSeq_key: idict[resSeq_key],
                    res_ref_key: idict_next[res_ref_key],
                    res_top_key: idict[res_top_key],
                    }
        df2print = _DF.from_dict([new_dict])[order].to_string()
        list_of_merged_dicts_for_insertion.append(new_dict)
        if verbose:
            print("old:")
            print(dataframevis[pair_idxs[0]-context:pair_idxs[1]+1+context][order].to_string())
            print("new:")
            print(df2print)
        print()

    return pairs_to_merge, list_of_merged_dicts_for_insertion

def insert_mergeable_positions(list_of_alignment_dicts,
                               pairs_to_merge,
                               list_of_merged_dicts_for_insertion):
    r"""
    Takes the result of :obj:`find_mergeable_positions` and inserts it in
    :obj:`alignment_result_to_list_of_dicts`

    Parameters
    ----------
    list_of_alignment_dicts: list
        What :obj:`alignment_result_to_list_of_dicts` returns.
    pairs_to_merge: list
        List of pairs to be merged. Typically the result of :obj:`find_mergeable_positions`
    list_of_merged_dicts_for_insertion: list
        List of dictionaries ready to be inserted in :obj:`list_of_alignment_dicts`.
        Typicall the result of :obj:`find_mergeable_positions`

    Returns
    -------
    list_of_alignment_dicts_new: list
        The input list :obj:`list_of_alignment_dicts` with the "mergeable" positions
        already merged

    """
    list_of_alignment_dicts_new = [ii for ii in list_of_alignment_dicts]

    for counter, ((ii, ii_next), new_dict) in enumerate(zip(pairs_to_merge, list_of_merged_dicts_for_insertion)):
        ii -= counter
        ii_next -= counter
        list_of_alignment_dicts_new = list_of_alignment_dicts_new[:ii] + [new_dict] + list_of_alignment_dicts_new[
                                                                                      ii_next + 1:]

    return list_of_alignment_dicts_new
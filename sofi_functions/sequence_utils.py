import numpy as _np
from pandas import DataFrame as _DF
def alignment_result_to_list_of_dicts(ialg, topology_0,
                                      seq_0_res_idxs,
                                      seq_1_res_idxs,
                                      AA_code_seq_0_key="AA_0",
                                      AA_code_seq_1_key="AA_1",
                                      resSeq_seq_0_key="resSeq_0",
                                      idx_seq_1_key="idx",
                                      full_resname_seq_0_key='fullname_0',
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

        If and when possible, this re-merging allows for a conservation
        of the sequential indices in both sequences

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

    resname_top_0_iterator = iter([str(topology_0.residue(ii)) for ii in seq_0_res_idxs])

    alignment_dict = []
    for rt, rr in zip(top_0_seq, top_1_seq):
        alignment_dict.append({AA_code_seq_0_key: rt,
                               AA_code_seq_1_key: rr,
                               resSeq_seq_0_key: '~',
                               full_resname_seq_0_key: '~',
                               idx_seq_1_key: '~'})

        if rt.isalpha():
            alignment_dict[-1][resSeq_seq_0_key] = next(top_0_resSeq_iterator)
            alignment_dict[-1][full_resname_seq_0_key] = next(resname_top_0_iterator)

        if rr.isalpha():
            alignment_dict[-1][idx_seq_1_key] = next(seq_1_res_idxs_iterator)

    if re_merge_skipped_entries:
        pairs_to_merge, list_of_new_merged_dicts = find_mergeable_positions(alignment_dict,
                                                                            AA_code_seq_1_key=AA_code_seq_1_key,
                                                                            AA_code_seq_0_key=AA_code_seq_0_key,
                                                                            resSeq_seq_0_key=resSeq_seq_0_key,
                                                                            idx_seq_1_key=idx_seq_1_key,
                                                                            full_resname_seq_0_key=full_resname_seq_0_key,
                                                                            verbose=verbose,
                                                                            context=5,
                                                                            )
        alignment_dict = insert_mergeable_positions(alignment_dict, pairs_to_merge, list_of_new_merged_dicts)
        if verbose:
            print("\nFinal alignment after merging")
            order = [idx_seq_1_key, AA_code_seq_1_key, AA_code_seq_0_key, resSeq_seq_0_key, full_resname_seq_0_key]
            print(_DF(alignment_dict)[order].to_string())

    # Add a field for matching vs nonmatching AAs
    for idict in alignment_dict:
        idict["match"] = False
        if idict[AA_code_seq_0_key]==idict[AA_code_seq_1_key]:
            idict["match"]=True

    return alignment_dict

def find_mergeable_positions(list_of_alignment_dicts,
                             context=2,
                             idx_seq_1_key="idx",
                             resSeq_seq_0_key="resSeq",
                             AA_code_seq_0_key="res_top",
                             AA_code_seq_1_key="res_ref",
                             full_resname_seq_0_key='fullname',
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
    idx_seq_1_key
    resSeq_seq_0_key
    AA_code_seq_0_key
    AA_code_seq_1_key
    full_resname_seq_0_key
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
        if idict[idx_seq_1_key] == '~' and idict[AA_code_seq_1_key] == '-':
            """
               AA_0 AA_1 fullname_0 idx resSeq_0
            65    L    L      LEU95  65       95
            66    M    -      MET96   ~       96
            67    -    T          ~  66        ~
            68    K    K      LYS97  67       97
            
            should be: 
                AA_0 AA_1 fullname_0 idx resSeq_0
            65    L    L      LEU95  65       95
            66    M    T      MET96  66       96
            67    K    K      LYS97  67       97
            
            AND 
                AA_0 AA_1 fullname_0  idx resSeq_0
            157    N    -     ASN187    ~      187
            158    E    E     GLU188  157      188
            159    -    E          ~  158        ~
            160    T    T     THR189  159      189

            should be: 
                AA_0 AA_1 fullname_0  idx resSeq_0
            157    N    E     ASN187  157      187
            158    E    E     GLU188  158      188
            159    T    T     THR189  159      189
            """
            for jj in [1,
                2
                       ]:
                try:
                    idict_next = list_of_alignment_dicts[ii + jj]
                except IndexError:
                    print("Exiting at", ii,jj)
                    print("Still missing")
                    print(_DF(list_of_alignment_dicts[ii:]))
                    break
                if idict_next[full_resname_seq_0_key] == '~' and\
                   idict_next[resSeq_seq_0_key] == '~' and \
                   idict_next[AA_code_seq_0_key] == '-':
                    # Ok this is mergeable in principle, but whe have
                    # to check that it leads to a re-alignment down
                    # the sequence
                    """
                           RR_H AA_H AA_N  match
                    0     CYSP2    X    -  False
                    1     CYSP3    X    -  False
                    2         ~    -    G  False
                    3         ~    -    G  False
                    4         ~    -    C  False
                    5      LEU4    L    L   True
                    """
                    pairs_to_merge.append([ii, ii + jj])
                    dict_pairs_to_merge.append([idict, idict_next])

                    dicts_in_between=[list_of_alignment_dicts[pp] for pp in range(ii+1,ii+jj)]
                    #print("LEN",len(dicts_in_between))
                    #print("Appending at %u (d=%u)"%(ii,jj))
                    #print(_DF.from_dict([idict]+dicts_in_between+[idict_next]))
                    #return None, None


    # TODO: creating the DF structure two times here just for visualitzaiton purposes
    dataframevis = _DF.from_dict(list_of_alignment_dicts)
    #order = [AA_code_seq_1_key, AA_code_seq_0_key, resSeq_seq_0_key, full_resname_seq_0_key]
    #print(dataframevis)
    if verbose:
     print("The following merges will take place:")
    list_of_merged_dicts_for_insertion = []
    for pair_idxs, (idict, idict_next) in zip(pairs_to_merge, dict_pairs_to_merge):
     new_dict = {full_resname_seq_0_key: idict[full_resname_seq_0_key],
                 idx_seq_1_key: idict_next[idx_seq_1_key],
                 resSeq_seq_0_key: idict[resSeq_seq_0_key],
                 AA_code_seq_1_key: idict_next[AA_code_seq_1_key],
                 AA_code_seq_0_key: idict[AA_code_seq_0_key],
                 }
     #df2print = _DF.from_dict([new_dict])[order].to_string()
     df2print = _DF.from_dict([new_dict]).to_string()
     list_of_merged_dicts_for_insertion.append(new_dict)
     if verbose:
         print("old:")
         print(dataframevis[pair_idxs[0]-context:pair_idxs[1]+1+context].to_string())
         #print(dataframevis[pair_idxs[0]-context:pair_idxs[1]+1+context][order].to_string())

         print("new:")
         print(df2print)
         print()

    return pairs_to_merge, list_of_merged_dicts_for_insertion

def _insert_mergeable_positions(list_of_alignment_dicts,
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

    Returns
    -------
    list_of_alignment_dicts_new: list
        The input list :obj:`list_of_alignment_dicts` with the "mergeable" positions
        already merged

    """
    # Create a dataframe for list extraction
    idf = _DF.from_dict(list_of_alignment_dicts)

    # Auto detect-which columns have the AA codes
    col1, col2 =detect_one_letter_columns_in_dataframe(idf)
    #print(idf)
    # Keep track of who many deletions have taken place already
    offsets = {key:0 for key in idf.keys()}
    for pair in pairs_to_merge:
        old_score = _np.sum([ii == jj for ii, jj in zip(idf[col1], idf[col2])])
        #print("old_score", old_score)
        candidate_df = idf.copy(deep=True)
        for ii in pair:
            for key, val in list_of_alignment_dicts[ii].items():
                if val in ['~','-']:
                    new_column_for_df = list(_np.delete(list(candidate_df[key]), ii-offsets[key]))+[val]
                    offsets[key]+=1
                    candidate_df[key] = new_column_for_df
        new_score = _np.sum([ii == jj for ii, jj in zip(candidate_df[col1], candidate_df[col2])])
        #print("new_score",new_score)
        if new_score>old_score:
            idf=candidate_df.copy(deep=True)
        else:
            print("This re-merge is not helping the alignment, skipping it %s"%pair)

    list_of_alignment_dicts_new = [{key: val for key, val in idf.loc[ii].items()} for ii in idf.T]
    list_of_alignment_dicts_new = [idict for idict in list_of_alignment_dicts_new if not all([val in ['~','-'] for val in idict.values()])]

    return list_of_alignment_dicts_new

def detect_one_letter_columns_in_dataframe(idf):
    one_letter_columns=[]
    for key, val in idf.items():
        if all([isinstance(ii,str) and len(ii)==1 for ii in val]):
            one_letter_columns.append(key)
    return one_letter_columns

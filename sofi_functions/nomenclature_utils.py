import mdtraj as _md
import numpy as _np
from .aa_utils import int_from_AA_code

def table2BW_by_AAcode(tablefile="GPCRmd_B2AR_nomenclature.xlsx",
                       modifications={"S262":"F264"},
                       keep_AA_code=True,
                       return_defs=False,
                       ):
    """

    :param tablefile: GPCRmd_B2AR nomenclature file in excel format
    :param modifications: Dictionary to store the modifications required in amino acid name
                        Parameter should be passed as a dictionary of the form {old name:new name}
    :param keep_AA_code: True if amino acid letter code is required else False. Default is True
                        If True then output dictionary will have key of the form "Q26" else "26"
    :param return_defs: if defs are required then True else False. Default is true
    :return: Dictionary if return_defs=false else dictionary and a list
    """
    out_dict = {}
    import pandas
    df = pandas.read_excel(tablefile, header=None)

    # Locate definition lines and use their indices
    defs = []
    for ii, row in df.iterrows():
        if row[0].startswith("TM") or row[0].startswith("H8"):
            defs.append(row[0])

        else:
            out_dict[row[2]] = row[1]

    # Replace some keys
    __ = {}
    for key, val in out_dict.items():
        for patt, sub in modifications.items():
            key = key.replace(patt,sub)
        __[key] = str(val)
    out_dict = __

    # Make proper BW notation as string with trailing zeros
    out_dict = {key:'%1.2f'%float(val) for key, val in out_dict.items()}

    if keep_AA_code:
        pass
    else:
        out_dict =  {int(key[1:]):val for key, val in out_dict.items()}

    if return_defs:
        return out_dict, defs
    else:
        return out_dict


def guess_missing_BWs(input_BW_dict,top, restrict_to_residxs=None):

    guessed_BWs = {}
    if restrict_to_residxs is None:
        restrict_to_residxs = [residue.index for residue in top.residues]

    """
    seq = ''.join([top._residues    [ii].code for ii in restrict_to_residxs])
    seq_BW =  ''.join([key[0] for key in input_BW_dict.keys()])
    ref_seq_idxs = [int_from_AA_code(key) for key in input_BW_dict.keys()]
    for alignmt in pairwise2.align.globalxx(seq, seq_BW)[:1]:
        alignment_dict = alignment_result_to_list_of_dicts(alignmt, top,
                                                            ref_seq_idxs,
                                                            #res_top_key="target_code",
                                                           #resname_key='target_resname',
                                                           #resSeq_key="target_resSeq",
                                                           #idx_key='ref_resSeq',
                                                           #re_merge_skipped_entries=False
                                                            )
        print(alignment_dict)
    return
    """
    out_dict = {ii:None for ii in range(top.n_residues)}
    for rr in restrict_to_residxs:
        residue = top.residue(rr)
        key = '%s%s'%(residue.code,residue.resSeq)
        try:
            (key, input_BW_dict[key])
            #print(key, input_BW_dict[key])
            out_dict[residue.index] = input_BW_dict[key]
        except KeyError:
            resSeq = int_from_AA_code(key)
            try:
                key_above = [key for key in input_BW_dict.keys() if int_from_AA_code(key)>resSeq][0]
                resSeq_above = int_from_AA_code(key_above)
                delta_above = int(_np.abs([resSeq - resSeq_above]))
            except IndexError:
                delta_above = 0
            try:
                key_below = [key for key in input_BW_dict.keys() if int_from_AA_code(key)<resSeq][-1]
                resSeq_below = int_from_AA_code(key_below)
                delta_below = int(_np.abs([resSeq-resSeq_below]))
            except IndexError:
                delta_below = 0

            if delta_above<=delta_below:
                closest_BW_key = key_above
                delta = -delta_above
            elif delta_above>delta_below:
                closest_BW_key = key_below
                delta = delta_below
            else:
                print(delta_above, delta_below)
                raise Exception

            if residue.index in restrict_to_residxs:
                closest_BW=input_BW_dict[closest_BW_key]
                base, exp = [int(ii) for ii in closest_BW.split('.')]
                new_guessed_val = '%s.%u*'%(base,exp+delta)
                #guessed_BWs[key] = new_guessed_val
                out_dict[residue.index] = new_guessed_val
                #print(key, new_guessed_val, residue.index, residue.index in restrict_to_residxs)
            else:
                pass
                #new_guessed_val = None

            # print("closest",closest_BW_key,closest_BW, key, new_guessed_val )

    #input_BW_dict.update(guessed_BWs)

    return out_dict

class CGN_transformer(object):
    def __init__(self, ref_PDB='3SN6', ref_path='.'):
        # Create dataframe with the alignment
        from pandas import read_table as _read_table
        from os import path as _path
        self._ref_PDB = ref_PDB

        self._DF = _read_table(_path.join(ref_path, 'CGN_%s.txt'%ref_PDB))
        #TODO find out how to properly do this with pandas

        self._dict = {key: self._DF[self._DF[ref_PDB] == key]["CGN"].to_list()[0] for key in self._DF[ref_PDB].to_list()}

        self._top =_md.load(_path.join(ref_path, ref_PDB+'.pdb')).top
        seq_ref = ''.join([str(rr.code).replace("None","X") for rr in self._top.residues])[:len(self._dict)]
        seq_idxs = _np.hstack([rr.resSeq for rr in self._top.residues])[:len(self._dict)]
        keyval = [{key:val} for key,val in self._dict.items()]
        #for ii, (iseq_ref, iseq_idx) in enumerate(zip(seq_ref, seq_idxs)):
        #print(ii, iseq_ref, iseq_idx )

        self._seq_ref  = seq_ref
        self._seq_idxs = seq_idxs

    @property
    def seq(self):
        return self._seq_ref

    @property
    def seq_idxs(self):
        return self._seq_idxs

    @property
    def AA2CGN(self):
        return self._dict

        #return seq_ref, seq_idxs, self._dict


def top2CGN_by_AAcode(top, ref_CGN_tf, keep_AA_code=True,
                      restrict_to_residxs=None):

    # TODO this lazy import will bite back
    from .Gunnar_utils import alignment_result_to_list_of_dicts
    from Bio import pairwise2


    if restrict_to_residxs is None:
        restrict_to_residxs = [residue.index for residue in top.residues]

    #out_dict = {ii:None for ii in range(top.n_residues)}
    #for ii in restrict_to_residxs:
    #    residue = top.residue(ii)
    #    AAcode = '%s%s'%(residue.code,residue.resSeq)
    #    try:
    #        out_dict[ii]=ref_CGN_tf.AA2CGN[AAcode]
    #    except KeyError:
    #        pass
    #return out_dict
    seq = ''.join([str(top.residue(ii).code).replace("None", "X") for ii in restrict_to_residxs])
    #
    res_idx2_PDB_resSeq = {}
    for alignmt in pairwise2.align.globalxx(seq, ref_CGN_tf.seq)[:1]:
        list_of_alignment_dicts = alignment_result_to_list_of_dicts(alignmt, top,
                                                            ref_CGN_tf.seq_idxs,
                                                            res_top_key="Nour_code",
                                                            resname_key='Nour_resname',
                                                            resSeq_key="Nour_resSeq",
                                                            res_ref_key='3SN6_code',
                                                            idx_key='3SN6_resSeq',
                                                            subset_of_residxs=restrict_to_residxs,
                                                            #re_merge_skipped_entries=False
                                                           )

        #import pandas as pd
        #with pd.option_context('display.max_rows', None, 'display.max_columns',
        #                       None):  # more options can be specified also
        #    for idict in list_of_alignment_dicts:
        #        idict["match"] = False
        #        if idict["Nour_code"]==idict["3SN6_code"]:
        #            idict["match"]=True
        #    print(DataFrame.from_dict(list_of_alignment_dicts))

        res_idx_array=iter(restrict_to_residxs)
        for idict in list_of_alignment_dicts:
             if '~' not in idict["Nour_resname"]:
                 idict["target_residx"]=\
                 res_idx2_PDB_resSeq[next(res_idx_array)]='%s%s'%(idict["3SN6_code"],idict["3SN6_resSeq"])
    out_dict = {}
    for ii in range(top.n_residues):
        try:
            out_dict[ii] = ref_CGN_tf.AA2CGN[res_idx2_PDB_resSeq[ii]]
        except KeyError:
            out_dict[ii] = None

    return out_dict
    # for key, equiv_at_ref_PDB in res_idx2_PDB_resSeq.items():
    #     if equiv_at_ref_PDB in ref_CGN_tf.AA2CGN.keys():
    #         iCGN = ref_CGN_tf.AA2CGN[equiv_at_ref_PDB]
    #     else:
    #         iCGN = None
    #     #print(key, top.residue(key), iCGN)
    #     out_dict[key]=iCGN
    # if keep_AA_code:
    #     return out_dict
    # else:
    #     return {int(key[1:]):val for key, val in out_dict.items()}
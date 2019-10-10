import mdtraj as _md
import numpy as _np
from .aa_utils import int_from_AA_code as _int_from_AA_code, shorten_AA as _shorten_AA
from .sequence_utils import alignment_result_to_list_of_dicts as _alignment_result_to_list_of_dicts, _my_bioalign
from pandas import DataFrame as _DF
def table2BW_by_AAcode(tablefile="GPCRmd_B2AR_nomenclature.xlsx",
                       modifications={"S262":"F264"},
                       keep_AA_code=True,
                       return_defs=False,
                       ):
    """
    Reads an excel table and returns a dictionary AAcodes so that e.g. '3.50' = self.AAcode2BW[R131]

    Parameters
    ----------
    tablefile : xlsx file
        GPCRmd_B2AR nomenclature file in excel format, optional
    modifications : dictionary
        Pass the modifications required in the residue name.
        Parameter should be passed as a dictionary of the form {old name:new name}.
    keep_AA_code : boolean
        'True' if amino acid letter code is required. (Default is True).
        If True then output dictionary will have key of the form "Q26" else "26".
    return_defs : boolean
        'True' if definition lines from the file are required then. (Default is True).

    Returns
    -------

    AAcode2BW : dictionary
        Dictionary with residues as key and their corresponding BW notation.

    defs : list (optional)
        if return_defs=false, a list containing the name of the fragments according to the excel file

    """
    AAcode2BW = {}
    import pandas
    df = pandas.read_excel(tablefile, header=None)

    # Locate definition lines and use their indices
    defs = []
    for ii, row in df.iterrows():
        if row[0].startswith("TM") or row[0].startswith("H8"):
            defs.append(row[0])

        else:
            AAcode2BW[row[2]] = row[1]

    # Replace some keys
    __ = {}
    for key, val in AAcode2BW.items():
        for patt, sub in modifications.items():
            key = key.replace(patt,sub)
        __[key] = str(val)
    AAcode2BW = __

    # Make proper BW notation as string with trailing zeros
    AAcode2BW = {key:'%1.2f'%float(val) for key, val in AAcode2BW.items()}

    if keep_AA_code:
        pass
    else:
        AAcode2BW =  {int(key[1:]):val for key, val in AAcode2BW.items()}

    if return_defs:
        return AAcode2BW, defs
    else:
        return AAcode2BW


def guess_missing_BWs(input_BW_dict,top, restrict_to_residxs=None, keep_keys=False):
    """
    Interpolates the BW for residues which are not present in the nomenclature file.

    Parameters
    ----------
    input_BW_dict : dictionary
        BW dictionary with residue names as the key and their corresponding BW notation
    top : :py:class:`mdtraj.Topology`
    restrict_to_residxs: list, optional
        residue indexes for which the BW needs to be estimated. (Default value is None, which means all).

    Returns
    -------
    BW : list
        list of len=top.n_residues including estimated missing BW-names, it also retains all the values from the input dictionary.

    """

    if restrict_to_residxs is None:
        restrict_to_residxs = [residue.index for residue in top.residues]

    #TODO keep this until we are sure there are no consquences
    out_list = [None for __ in top.residues]
    for rr in restrict_to_residxs:
        residue = top.residue(rr)
        key = '%s%s'%(residue.code,residue.resSeq)
        try:
            (key, input_BW_dict[key])
            #print(key, input_BW_dict[key])
            out_list[residue.index] = input_BW_dict[key]
        except KeyError:
            resSeq = _int_from_AA_code(key)
            try:
                key_above = [key for key in input_BW_dict.keys() if _int_from_AA_code(key)>resSeq][0]
                resSeq_above = _int_from_AA_code(key_above)
                delta_above = int(_np.abs([resSeq - resSeq_above]))
            except IndexError:
                delta_above = 0
            try:
                key_below = [key for key in input_BW_dict.keys() if _int_from_AA_code(key)<resSeq][-1]
                resSeq_below = _int_from_AA_code(key_below)
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
                out_list[residue.index] = new_guessed_val
                #print(key, new_guessed_val, residue.index, residue.index in restrict_to_residxs)
            else:
                pass
                #new_guessed_val = None

            # print("closest",closest_BW_key,closest_BW, key, new_guessed_val )

    #input_BW_dict.update(guessed_BWs)

    if keep_keys:
        guessed_BWs = {}
        used_keys = []
        for res_idx, val in enumerate(out_list):
            new_key = _shorten_AA(top.residue(res_idx))
            if new_key in input_BW_dict.keys():
                assert val==input_BW_dict[new_key],"This should not have happened %s %s %s"%(val, new_key, input_BW_dict[new_key])
            assert new_key not in used_keys
            guessed_BWs[new_key]=val
            used_keys.append(new_key)
        return guessed_BWs
    else:
        return out_list

class CGN_transformer(object):
    """
    Class to abstract, handle, and use common-Gprotein-nomenclature.
    See here_ for more info.
     .. _here: https://www.mrc-lmb.cam.ac.uk/CGN/faq.html
    """
    def __init__(self, ref_PDB='3SN6', ref_path='.', try_web_lookup=True):
        r"""

        Parameters
        ----------
        ref_PDB: str, default is '3SN6'
            The PDB four letter code that will be used for CGN purposes
        ref_path: str, default is '.'
            The local path where these files exist
             * 3SN6_CGN.txt
             * 3SN6.pdb

        try_web_lookup: bool, default is True
            If the local files are not found, try automatically a web lookup at
             * www.mrc-lmb.cam.ac.uk (for CGN)
             * rcsb.org (for the PDB)
        """
        # Create dataframe with the alignment
        from pandas import read_csv as _read_csv
        from os import path as _path
        self._ref_PDB = ref_PDB
        self._CGN_file = 'CGN_%s.txt' % ref_PDB
        try:

            self._DF = _read_csv(_path.join(ref_path, self._CGN_file), delimiter='\t')
        except FileNotFoundError:
            print("No local file %s numbering found"%self._CGN_file,end="")
            if try_web_lookup:
                web_address="www.mrc-lmb.cam.ac.uk"
                url = "https://%s/CGN/lookup_results/%s.txt" % (web_address, ref_PDB)
                print(", checking online in\n%s ..."%url,end="")
                import urllib
                try:
                    self._DF = _read_csv(url, delimiter='\t')
                    print("found! Continuing normally")
                except urllib.error.HTTPError as e:
                    print(e)
                    print("Aborting.")
                    return
            else:
                raise

        self._dict = {key: self._DF[self._DF[ref_PDB] == key]["CGN"].to_list()[0] for key in self._DF[ref_PDB].to_list()}

        try:
            pdbfile = ref_PDB+'.pdb'
            self._top =_md.load(_path.join(ref_path, pdbfile)).top
        except (OSError,FileNotFoundError):
            try:
                pdbfile = ref_PDB + '.pdb.gz'
                self._top = _md.load(_path.join(ref_path, pdbfile)).top
            except (OSError, FileNotFoundError):
                print("No local PDB file for %s found"%ref_PDB, end="")
                if try_web_lookup:
                    web_address="https://files.rcsb.org/download"
                    print(", checking online in %s ..."%web_address, end="")
                    self._top = _md.load_pdb("%s/%s" %(web_address, pdbfile)).top
                    print("found! Continuing normally")
                else:
                    raise

        seq_ref = ''.join([str(rr.code).replace("None","X") for rr in self._top.residues])[:len(self._dict)]
        seq_idxs = _np.hstack([rr.resSeq for rr in self._top.residues])[:len(self._dict)]
        keyval = [{key:val} for key,val in self._dict.items()]
        #for ii, (iseq_ref, iseq_idx) in enumerate(zip(seq_ref, seq_idxs)):
        #print(ii, iseq_ref, iseq_idx )

        self._seq_ref  = seq_ref
        self._seq_idxs = seq_idxs

        self._ref_PDB = ref_PDB

        self._fragment_names = []
        for key in self.AA2CGN.values():
            new_key = '.'.join(key.split(".")[:-1])
            #print(new_key)
            if new_key not in self._fragment_names:
                self._fragment_names.append(new_key)
                #print("yes")

    @property
    def fragment_names(self):
        r"""Name of the fragments according to the CGN numbering"""
        return self._fragment_names

    @property
    def seq(self):
        r""" Sequence of AAs (one-letter codes) in the reference pdb file.
        If an AA has no one-letter, the letter X is used"""
        return self._seq_ref

    @property
    def seq_idxs(self):
        r""" Indices contained in the original PDB as sequence indices.
        In an :obj:`mdtraj.Topology.Residue`, this index is called 'ResSeq'"""
        return self._seq_idxs

    @property
    def AA2CGN(self):
        r"""Dictionary with AA-codes as keys, so that AA2CGN["K25"] -> G.HN.42"""
        #If an AA does not have a CGN-name, it is not present in the keys. """
        return self._dict

    @property
    def ref_PDB(self):
        r""" Return the PDB code used for instantiation"""

        return self._ref_PDB

        #return seq_ref, seq_idxs, self._dict

    def top2map(self, top, restrict_to_residxs=None):
        r""" Align the sequence of :obj:`top` to the transformers sequence
        and return a list of CGN numbering for each residue in :obj:`top`.
        If no CGN numbering is found after the alignment, the entry will be None

        Parameters
        ----------
        top : obj:`mdtraj.Topology` object
        restrict_to_residxs: iterable of integers, default is None
            You can select a segment of the top that aligns best to the CGN numbering
            to improve the quality of the alignment. The return list will still
            be of length= top.n_residues

        Returns
        -------
        map : list
        """
        return _top2map(self.AA2CGN, top, restrict_to_residxs=restrict_to_residxs)

    def top2defs(self, top, return_defs=False):
        map = self.top2map(top)
        defs = _map2defs(map,'CGN')
        from sofi_functions.fragments import _print_frag
        for ii, (key, val) in enumerate(defs.items()):
            istr = _print_frag('%s' % key, top, val, fragment_desc='', return_string=True)
            print(istr, '%s-%s' % (map[val[0]], map[val[-1]]))

        if return_defs:
            return defs

#TODO CONSIDER CHANGING THE NAME "TRANSFORMER"
class BW_transformer(object):
    """
    Class to manage Ballesteros-Weinstein notation

    """
    def __init__(self, tablefile="GPCRmd_B2AR_nomenclature.xlsx", ref_path='.'):
        r"""

        Parameters
        ----------
        tablefile: str, default is 'GPCRmd_B2AR_nomenclature'
            The PDB four letter code that will be used for CGN purposes
        ref_path: str,default is '.'
            The local path where the needed files are

        """

        self._tablefile = tablefile

        self._AAcode2BW, self._defs = table2BW_by_AAcode(tablefile,return_defs=True)

        self._fragments = {key: [] for key in self._defs}
        for AArS, iBW in self._AAcode2BW.items():
            intBW = int(iBW[0])
            if intBW<8:
                key= 'TM%u'%intBW
            else:
                key = 'H%u' % intBW
            self._fragments[key].append(AArS)
        from . aa_utils import name_from_AA as _name_from_AA
        self._seq_fragments = {key:''.join([_name_from_AA(AA) for AA in val]) for key, val in self._fragments.items()}

    @property
    def tablefile(self):
        r""" The file used to instantiate this transformer"""
        return self._tablefile

    @property
    def AAcode2BW(self):
        r""" Dictionary AA-codes as keys, so that e.g. '3.50' = self.AAcode2BW[R131]"""
        return self._AAcode2BW

    @property
    def fragment_names(self):
        r""" List of the available fragments, e.g. ["TM1","TM2"]. Can also contain loops if the user asked for it"""
        return self._defs

    @property
    def fragments(self):
        r""" Dictionary of BW fragment definitions, keys are self.fragment_names"""
        return self._fragments

    @property
    def seq_fragments(self):
        r""" Dictionary of sequences for each fragment in self.fragments"""
        return self._seq_fragments

    @property
    def seq(self):
        r""" Sequence of the AA found in this BW transformer"""
        return ''.join(self._seq_fragments.values())

    def top2map(self, top, restrict_to_residxs=None):
        return _top2map(self.AAcode2BW, top, restrict_to_residxs=restrict_to_residxs)

    def top2defs(self, top, return_defs=False):
        map = self.top2map(top)
        defs = _map2defs(map,'BW')
        defs = {('TM%s'%key).replace('TM8','H8'):val for key, val in defs.items()}
        from sofi_functions.fragments import _print_frag
        for ii, (key, val) in enumerate(defs.items()):
            istr = _print_frag(key, top, val, fragment_desc='', return_string=True)
            print(istr, '%s-%s' % (map[val[0]], map[val[-1]]))
        if return_defs:
            return defs

# TODO TEST
def _top2map(consensus_dict, top, restrict_to_residxs=None):
    r"""
    Align the sequence of :obj:`top` to dictionary's sequence and return a
    list of consensus numbering for each residue in :obj:`top`. If no consensus numbering
    is found after the alignment, the entry will be None

    Parameters
    ----------
    consensus_dict : dictionary
        AA-codes as keys and nomenclature as values, e.g. AA2CGN["K25"] -> G.HN.42
    top : obj:`mdtraj.Topology` object
    restrict_to_residxs: iterable of integers, default is None
        You can select a segment of the top that aligns best to the CGN numbering
        to improve the quality of the alignment.

    Returns
    -------
    map : list
        list of length top.n_residues containing CGN numberings
    """

    if restrict_to_residxs is None:
        restrict_to_residxs = [residue.index for residue in top.residues]
    seq = ''.join([_shorten_AA(top.residue(ii), keep_index=False, substitute_fail='X') for ii in restrict_to_residxs])
    from sofi_functions.aa_utils import name_from_AA as _name_from_AA
    seqBW= ''.join([_name_from_AA(key) for key in consensus_dict.keys()])
    alignment = _alignment_result_to_list_of_dicts(_my_bioalign(seq, seqBW)[0],
                                                   top,
                                                   restrict_to_residxs,
                                                   [_int_from_AA_code(key) for key in consensus_dict],
                                                   #verbose=True
                                                   )
    alignment = _DF(alignment)
    alignment = alignment[alignment["match"] == True]
    out_list = [None for __ in top.residues]
    for idx, resSeq, AA in alignment[["idx_0","idx_1", "AA_1"]].values:
        out_list[int(idx)]=consensus_dict[AA + str(resSeq)]
    return out_list

def top2CGN_by_AAcode(top, ref_CGN_tf,
                      restrict_to_residxs=None,
                      verbose=False):
    """
    Returns a dictionary of CGN (Common G-protein Nomenclature) labelling for each residue.
    The keys are zero-indexed residue indices

    TODO if the length of the dictionary is always top.n_residues, consider simply returning a list

    Parameters
    ----------
    top : :py:class:`mdtraj.Topology`
    ref_CGN_tf : :obj: 'nomenclature_utils.CGN_transformer' object
    restrict_to_residxs: list, optional, default is None
        residue indexes for which the CGN needs to be found out. Default behaviour is for all
        residues in the :obj:`top`.

    Returns
    -------
    CGN_list : list
        list of length obj:`top.n_residues` containing the CGN numbering (if found), None otherwise

    """

    if restrict_to_residxs is None:
        restrict_to_residxs = [residue.index for residue in top.residues]


    seq = ''.join([str(top.residue(ii).code).replace("None", "X") for ii in restrict_to_residxs])


    # As complicated as this seems, it's just cosmetics for the alignment dictionaries
    AA_code_seq_0_key = "AA_input"
    full_resname_seq_0_key = 'resname_input'
    resSeq_seq_0_key = "resSeq_input"
    AA_code_seq_1_key = 'AA_ref(%s)' % ref_CGN_tf.ref_PDB
    idx_seq_1_key = 'resSeq_ref(%s)' % ref_CGN_tf.ref_PDB
    idx_seq_0_key = 'idx_input'
    for alignmt in _my_bioalign(seq, ref_CGN_tf.seq)[:1]:
        #TODO this fucntion has been changed and this transformer will not work anymore
        # major bug (still?)

        list_of_alignment_dicts = _alignment_result_to_list_of_dicts(alignmt, top,
                                                                     restrict_to_residxs,
                                                                     ref_CGN_tf.seq_idxs,
                                                                     AA_code_seq_0_key=AA_code_seq_0_key,
                                                                     full_resname_seq_0_key=full_resname_seq_0_key,
                                                                     resSeq_seq_0_key=resSeq_seq_0_key,
                                                                     AA_code_seq_1_key=AA_code_seq_1_key,
                                                                     idx_seq_1_key=idx_seq_1_key,
                                                                     idx_seq_0_key=idx_seq_0_key,
                                                                     )

        if verbose:
            import pandas as pd
            from .sequence_utils import _print_verbose_dataframe
            _print_verbose_dataframe(pd.DataFrame.from_dict(list_of_alignment_dicts))
            input("This is the alignment. Hit enter to continue")

        # TODO learn to tho dis with pandas
        list_out = [None for __ in top.residues]
        for idict in list_of_alignment_dicts:
            if idict["match"]==True:
                res_idx_input = restrict_to_residxs[idict[idx_seq_0_key]]
                match_name = '%s%s'%(idict[AA_code_seq_1_key],idict[idx_seq_1_key])
                iCGN = ref_CGN_tf.AA2CGN[match_name]
                if verbose:
                    print(res_idx_input,"res_idx_input",match_name, iCGN)
                list_out[res_idx_input]=iCGN

        if verbose:
            for idx, iCGN in enumerate(list_out):
                print(idx, iCGN, top.residue(idx))
            input("This is the actual return value. Hit enter to continue")
    return list_out

def _relabel_consensus(idx, input_dicts, no_key="NA"):
    labels  = [idict[idx] for idict in input_dicts]
    good_label = [ilab for ilab in labels if str(ilab).lower()!="none"]
    assert len(good_label)<=1, "There can only be one good label, but for residue %u found %s"%(idx, good_label)
    try:
        return good_label[0]
    except IndexError:
        return no_key

def csv_table2TMdefs_res_idxs(itop, keep_first_resSeq=True, complete_loops=True,
                              tablefile=None,
                              reorder_by_AA_names=False):
    # TODO pass this directly as kwargs?
    kwargs = {}
    if tablefile is not None:
        kwargs = {"tablefile": tablefile}
    segment_resSeq_dict = table2TMdefs_resSeq(**kwargs)
    resseq_list = [rr.resSeq for rr in itop.residues]
    if not keep_first_resSeq:
        raise NotImplementedError

    segment_dict = {}
    first_one=True
    for key,val in segment_resSeq_dict.items():
        print(key,val)
        res_idxs = [[ii for ii, iresSeq in enumerate(resseq_list) if iresSeq==ival][0] for ival in val]
        if not res_idxs[0]<res_idxs[1] and first_one:
            res_idxs[0]=0
            first_one = False
        segment_dict[key]=res_idxs

    if complete_loops:
        segment_dict = add_loop_definitions_to_TM_residx_dict(segment_dict)
        #for key, val in segment_dict.items():
            #print('%4s %s'%(key, val))

    if reorder_by_AA_names:
        _segment_dict = {}
        for iseg_key, (ilim, jlim) in segment_dict.items():
            for ii in _np.arange(ilim, jlim+1):
                _segment_dict[_shorten_AA(itop.residue(ii))]=iseg_key
        segment_dict = _segment_dict
    return segment_dict



def add_loop_definitions_to_TM_residx_dict(segment_dict, not_present=["ICL3"], start_with='ICL'):
    loop_idxs={"ICL":1,"ECL":1}
    loop_type=start_with
    keys_out = []
    for ii in range(1,7):
        key1, key2 = 'TM%u'%  (ii + 0), 'TM%u'%  (ii + 1)
        loop_key = '%s%s'%(loop_type, loop_idxs[loop_type])
        if loop_key in not_present:
            keys_to_append = [key1, key2]
        else:
            segment_dict[loop_key] = [segment_dict[key1][-1] + 1,
                                       segment_dict[key2][0] - 1]
            #print(loop_key, segment_dict[loop_key])
            keys_to_append = [key1, loop_key, key2]
        keys_out = keys_out+[key for key in keys_to_append if key not in keys_out]

        loop_idxs[loop_type] +=1
        if loop_type=='ICL':
            loop_type='ECL'
        elif loop_type=='ECL':
            loop_type='ICL'
        else:
            raise Exception
    out_dict = {key : segment_dict[key] for key in keys_out}
    if 'H8' in segment_dict:
        out_dict.update({'H8':segment_dict["H8"]})
    return out_dict

def table2TMdefs_resSeq(tablefile="GPCRmd_B2AR_nomenclature.xlsx",
                        #modifications={"S262":"F264"},
                        reduce_to_resSeq=True):

    all_defs, names = table2BW_by_AAcode(tablefile,
                                         #modifications=modifications,
                                         return_defs=True)

    # First pass
    curr_key = 'None'
    keyvals = list(all_defs.items())
    breaks = []
    for ii, (key, val) in enumerate(keyvals):
        if int(val[0]) != curr_key:
            #print(key, val)
            curr_key = int(val[0])
            breaks.append(ii)
            # print(curr_key)
            # input()

    AA_dict = {}
    for idef, ii, ff in zip(names, breaks[:-1], breaks[1:]):
        #print(idef, keyvals[ii], keyvals[pattern - 1])
        AA_dict[idef]=[keyvals[ii][0], keyvals[ff-1][0]]
    #print(names[-1], keyvals[pattern], keyvals[-1])
    AA_dict[names[-1]] = [keyvals[ff][0], keyvals[-1][0]]
    #print(AA_dict)

    # On dictionary: just-keep the resSeq
    if reduce_to_resSeq:
        AA_dict = {key: [''.join([ii for ii in ival if ii.isnumeric()]) for ival in val] for key, val in
                   AA_dict.items()}
        AA_dict = {key: [int(ival) for ival in val] for key, val in
                   AA_dict.items()}
    return AA_dict

def _guess_nomenclature_fragments(BWtf, top, fragments, cutoff=.75, verbose=False):
    aligned_BWs = BWtf.top2map(top)
    guess = []
    for ii, ifrag in enumerate(fragments):
        hits = [aligned_BWs[jj] for jj in ifrag if aligned_BWs[jj] is  not None]
        if len(hits)/len(ifrag)>=cutoff:
            guess.append(ii)
        if verbose:
            print(ii, len(hits)/len(ifrag))

    return guess

def _map2defs(indict,conv_type):
    from collections import defaultdict
    defs = defaultdict(list)
    for ii, key in enumerate(indict):
        if key is not None:
            if conv_type=='BW':
                new_key =key.split(".")[0]
            elif conv_type=='CGN':
                new_key = '.'.join(key.split(".")[:-1])
            else:
                raise Exception
            defs[new_key].append(ii)

    # TODO there has to be a better way for defdict->dict
    return {key: val for key, val in defs.items()}
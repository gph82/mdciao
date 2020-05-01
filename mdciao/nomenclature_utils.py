import mdtraj as _md
import numpy as _np
from .residue_and_atom_utils import int_from_AA_code as _int_from_AA_code, shorten_AA as _shorten_AA
from .sequence_utils import alignment_result_to_list_of_dicts as _alignment_result_to_list_of_dicts, _my_bioalign
from pandas import DataFrame as _DF, read_json as _read_json, read_excel as _read_excel
from collections import defaultdict as _defdict
from os import path as _path
import requests as _requests
from .list_utils import rangeexpand as _rangeexpand

def table2BW_by_AAcode(tablefile,
                       keep_AA_code=True,
                       return_fragments=False,
                       ):
    """
    Reads an excel table and returns a dictionary AAcodes so that e.g. self.AA2BW[R131] -> '3.50'

    Parameters
    ----------
    tablefile : xlsx file or pandas dataframe
        Ballesteros-Weinstein nomenclature file in excel format, optional
    keep_AA_code : boolean
        'True' if amino acid letter code is required. (Default is True).
        If True then output dictionary will have key of the form "Q26" else "26".
    return_defs : boolean
        'True' if definition lines from the file are required then. (Default is True).

    Returns
    -------

    AA2BW : dictionary
        Dictionary with residues as key and their corresponding BW notation.

    fragments : dict (optional)
        if return_fragments=True, a dictionary containing the fragments according to the excel file
    """

    if isinstance(tablefile,str):
        df = _read_excel(tablefile, header=None)
    else:
        df = tablefile

    # This is the most important
    AAcode2BW = {key: val for key, val in df[["AAresSeq", "BW"]].values}
    # Locate definition lines and use their indices
    fragments = _defdict(list)
    for key, AArS in df[["protein_segment", "AAresSeq"]].values:
        fragments[key].append(AArS)
    fragments = {key:val for key, val in fragments.items()}

    if keep_AA_code:
        pass
    else:
        AAcode2BW =  {int(key[1:]):val for key, val in AAcode2BW.items()}

    if return_fragments:
        return AAcode2BW, fragments
    else:
        return AAcode2BW


def guess_missing_BWs(input_BW_dict,top, restrict_to_residxs=None, keep_keys=False):
    """
    Estimates the BW for residues which are not present in the nomenclature file.

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
        list of len=top.n_residues including estimated missing BW-names,
        it also retains all the values from the input dictionary.

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

class consensus_labeler(object):
    """
    Class to manage consensus notations like
    * Ballesteros-Weinstein (BW)
    * Common-Gprotein-nomenclature

    The consensus labels are abbreviated to 'conlab' throughout

    """
    def __init__(self, ref_PDB=None, **PDB_finder_kwargs):
        r"""

        Parameters
        ----------
        tablefile: str, default is 'GPCRmd_B2AR_nomenclature'
            The PDB four letter code that will be used for CGN purposes
        ref_path: str,default is '.'
            The local path where the needed files are

        try_web_lookup: bool, default is True
            If the local files are not found, try automatically a web lookup at
             * www.mrc-lmb.cam.ac.uk (for CGN)
             * rcsb.org (for the PDB)

        """
        self._geom = None
        self._ref_top = None
        self._ref_PDB = ref_PDB
        if ref_PDB is not None:
            self._geom, self._PDB_file = PDB_finder(ref_PDB,
                                                    **PDB_finder_kwargs,
                                                    )

        self._conlab2AA = {val: key for key, val in self.AA2conlab.items()}
        self._fragment_names = list(self.fragments.keys())
        self._fragments_as_conlabs = {key: [self.AA2conlab[AA] for AA in val]
                                      for key, val in self.fragments.items()}

    @property
    def ref_PDB(self):
        r""" PDB code used for instantiation"""
        return self._ref_PDB

    @property
    def geom(self):
        return self._geom

    @property
    def top(self):
        return self._geom.top

    @property
    def conlab2AA(self):
        r""" Dictionary with consensus labels as keys, so that e.g. self.AA2BW["3.50"] -> 'R131' """
        return self._conlab2AA

    @property
    def AA2conlab(self):
        r""" Dictionary with AA-codes as keys, so that e.g. self.AA2BW[R131] -> '3.50' """
        return self._AA2conlab

    @property
    def fragment_names(self):
        r"""Name of the fragments according to the CGN numbering"""
        return self._fragment_names

    @property
    def fragments(self):
        return self._fragments

    @property
    def fragments_as_conlabs(self):
        return self._fragments_as_conlabs

    @property
    def dataframe(self):
        return self._dataframe

    @property
    def tablefile(self):
        r""" The file used to instantiate this transformer"""
        return self._tablefile

    def conlab2residx(self,top,
                      restrict_to_residxs=None,
                      map=None,
                      keep_consensus=False):
        if map is None:
            map = _top2consensus_map(self.AA2conlab, top,
                                     restrict_to_residxs=restrict_to_residxs,
                                     keep_consensus=keep_consensus)
        out_dict = {}
        for ii,imap in enumerate(map):
            if imap is not None:
                out_dict[imap]=ii
        return out_dict

    def top2map(self, top, restrict_to_residxs=None, fill_gaps=False,
                verbose=False):
        r""" Align the sequence of :obj:`top` to the transformer's sequence
        and return a list of BW numbering for each residue in :obj:`top`.
        If no BW numbering is found after the alignment, the entry will be None

        Parameters
        ----------
        top :
            :py:class:`mdtraj.Topology` object
        restrict_to_residxs: iterable of integers, default is None
            Use only these residues for alignment and labelling options.
            The return list will still be of length=top.n_residues
        fill_gaps: boolean, default is False
            Try to fill gaps in the consensus nomenclature by calling
            :obj:`_fill_CGN_gaps`

        Returns
        -------
        map : list of len = top.n_residues with the BW numbering entries
        """
        return _top2consensus_map(self.AA2conlab, top,
                                  restrict_to_residxs=restrict_to_residxs,
                                  keep_consensus=fill_gaps,
                                  verbose=verbose,
                                  )

    def top2defs(self, top, map=None,
                 return_defs=False,
                 fragments=None,
                 fill_gaps=False,
                 ):
        r"""
        Print the BW transformer's definitions for the subdomains,
        in terms of residue indices of the input :obj:`top`

        Parameters
        ----------
        top:
            :py:class:`mdtraj.Topology` object
        map:  list, default is None
            The user can parse an exisiting "top2map" map, otherwise this
            method will generate one on the fly. It is recommended (but
            not needed) to pre-compute and pass such a map in critical
            cases when naming errors are likely.
        return_defs: boolean, default is False
            If True, apart from printing the definitions,
            they are returned as a dictionary
        fragments: iterable of integers, default is None
            The user can parse an existing list of fragment-definitions,
            to check whether these definitions are broken in the new ones.
            An interactive prompt will appear in this case
        fill_gaps: boolean, default is False
            Try to fill gaps in the consensus nomenclature by calling
            :obj:`_fill_CGN_gaps`. It has no effect if the user inputs
            the map

        Returns
        -------
        defs : dictionary (if return_defs is True)
            Dictionary with subdomain names as keys and lists of indices as values
        """
        from mdciao.list_utils import in_what_fragment
        from mdciao.fragments import _print_frag

        if map is None:
            print("creating a temporary map, this is dangerous")
            map = self.top2map(top, fill_gaps=fill_gaps, verbose=False)

        conlab2residx = self.conlab2residx(top, map=map)
        defs = _defdict(list)
        for key, ifrag in self.fragments_as_conlabs.items():
            for iBW in ifrag:
                if iBW in conlab2residx.keys():
                    defs[key].append(conlab2residx[iBW])

        new_defs = {}
        for ii, (key, val) in enumerate(defs.items()):
            istr = _print_frag(key, top, val, fragment_desc='', return_string=True)
            if fragments is not None:
                ifrags = [in_what_fragment(idx, fragments) for idx in val]
                if ifrags[0]!=ifrags[-1]:
                    #todo AVOID ASKING THE USER
                    print(istr, '%s-%s' % (map[val[0]], map[val[-1]]))
                    answr = input("more than 1 fragments present. Input the ones to keep %s" % (_np.unique(ifrags)))
                    answr = _rangeexpand(answr)
                    tokeep = [idx for ii, idx in enumerate(val) if ifrags[ii] in answr]
                    new_defs[key] = tokeep

        for key, val in new_defs.items():
            defs[key]=val

        for ii, (key, val) in enumerate(defs.items()):
            istr = _print_frag(key, top, val, fragment_desc='', return_string=True)
            print(istr)

        if return_defs:
            return {key:val for key, val in defs.items()}

def CGN_finder(identifier,
               format='CGN_%s.txt',
               ref_path='.',
               try_web_lookup=True,
               verbose=True,
               dont_fail=False):
    from pandas import read_csv as _read_csv
    import urllib

    file2read = format%identifier
    _path.join(ref_path, file2read)
    try:
        _DF = _read_csv(file2read, delimiter='\t')
        return_name = file2read
    except FileNotFoundError:
        if verbose:
            print("No local file %s found" % file2read, end="")
        if try_web_lookup:
            web_address = "www.mrc-lmb.cam.ac.uk"
            url = "https://%s/CGN/lookup_results/%s.txt" % (web_address, identifier)
            if verbose:
                print(", checking online in\n%s ..." % url, end="")
            try:
                _DF = _read_csv(url, delimiter='\t')
                if verbose:
                    print("found! Continuing normally")
                return_name = url
            except urllib.error.HTTPError as e:
                print('CGN online db:',e)
                if dont_fail:
                    pass
                else:
                    raise
        else:
            raise

    if len(_DF)<=1:
        if dont_fail:
            pass
        else:
            print('CGN lookup returned empty')
            raise ValueError(identifier)
    return _DF, return_name

def PDB_finder(ref_PDB, ref_path='.',
               try_web_lookup=True,
               verbose=True):
    try:
        file2read = _path.join(ref_path, ref_PDB + '.pdb')
        _geom = _md.load(file2read)
        return_file = file2read
    except (OSError, FileNotFoundError):
        try:
            file2read = _path.join(ref_path, ref_PDB + '.pdb.gz')
            _geom = _md.load(file2read)
            return_file = file2read
        except (OSError, FileNotFoundError):
            if verbose:
                print("No local PDB file for %s found" % ref_PDB, end="")
            if try_web_lookup:
                _geom, return_file = md_load_rscb(ref_PDB,
                                                  verbose=verbose,
                                                  return_url=True)
                if verbose:
                    print("found! Continuing normally")

            else:
                raise

    return _geom, return_file

#todo document and refactor to better place?
def md_load_rscb(PDB,
                 web_address = "https://files.rcsb.org/download",
                 verbose=False,
                 return_url=False):
    url = '%s/%s.pdb' % (web_address, PDB)
    if verbose:
        print(", checking online in \n%s ..." % url, end="")
    igeom = _md.load_pdb(url)
    if return_url:
        igeom, url
    else:
        return igeom
class CGN_transformer(consensus_labeler):
    """
    Class to abstract, handle, and use common-Gprotein-nomenclature.
    See here_ for more info.
     .. _here: https://www.mrc-lmb.cam.ac.uk/CGN/faq.html
    """

    def __init__(self, ref_PDB,
                 ref_path='.',
                 try_web_lookup=True,
                 verbose=True):
        r"""

        Parameters
        ----------
        ref_PDB: str
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

        self._dataframe, self._CGN_file = CGN_finder(ref_PDB,
                                                     ref_path=ref_path,
                                                     try_web_lookup=try_web_lookup,
                                                     verbose=verbose)

        self._AA2conlab = {key: self._dataframe[self._dataframe[ref_PDB] == key]["CGN"].to_list()[0]
                           for key in self._dataframe[ref_PDB].to_list()}

        self._fragments = _defdict(list)
        for ires, key in self.AA2conlab.items():
            try:
                new_key = '.'.join(key.split(".")[:-1])
            except:
                print(key)
            #print(key,new_key)
            self._fragments[new_key].append(ires)
                #print("yes")
        #print(self.fragments)
        consensus_labeler.__init__(self, ref_PDB=ref_PDB,
                                   ref_path=ref_path,
                                   try_web_lookup=try_web_lookup,
                                   verbose=verbose)

    @property
    def CGN_file(self):
        r""" CGN_file used for instantiation"""
        return self._CGN_file

class BW_transformer(consensus_labeler):
    """
    Class to manage Ballesteros-Weinstein notation

    """
    def __init__(self, uniprot_name,
                 ref_PDB=None,
                 ref_path=".",
                 verbose=True,
                 try_web_lookup=True,
                 #todo write to disk should be moved to the superclass at some point
                 write_to_disk=False):

        xlsxname = '%s.xlsx'%uniprot_name
        if _path.exists(xlsxname):
            self._dataframe = _read_excel(xlsxname, converters={"BW":str}).replace({_np.nan:None})
            print("read %s locally."%xlsxname)
        else:
            self._dataframe = _uniprot_name_2_BWdf_from_gpcrdb(uniprot_name, verbose=verbose)
            if write_to_disk:
                self._dataframe.to_excel(xlsxname)
                print("wrote %s for future use"%xlsxname)
        self._AA2conlab, self._fragments = table2BW_by_AAcode(self.dataframe, return_fragments=True)
        # TODO can we do this using super?
        consensus_labeler.__init__(self,ref_PDB,
                                   ref_path=ref_path,
                                   try_web_lookup=try_web_lookup,
                                   verbose=verbose)

def _top2consensus_map(consensus_dict, top,
                       restrict_to_residxs=None,
                       keep_consensus=False,
                       verbose=False,
                       ):
    r"""
    Align the sequence of :obj:`top` to consensus
    dictionary's sequence (typically CGN_tf.AA2CGN))
    and return a list of consensus numbering for each residue
     in :obj:`top`. If no consensus numbering
    is found after the alignment, the entry will be None

    Parameters
    ----------
    consensus_dict : dictionary
        AA-codes as keys and nomenclature as values, e.g. AA2CGN["K25"] -> G.HN.42
    top :
        :py:class:`mdtraj.Topology` object
    restrict_to_residxs: iterable of integers, default is None
        Use only these residues for alignment and labelling purposes
    keep_consensus : boolean default is False
        Even if there is a consensus mismatch with the sequence of the input
        :obj:`consensus_dict`, try to relabel automagically, s.t.
        * ['G.H5.25', 'G.H5.26', None, 'G.H.28']
        will be grouped relabeled as
        * ['G.H5.25', 'G.H5.26', 'G.H.27', 'G.H.28']

    verbose: boolean, default is False
        be verbose

    Returns
    -------
    map : list
        list of length top.n_residues containing CGN numberings
    """

    if restrict_to_residxs is None:
        restrict_to_residxs = [residue.index for residue in top.residues]
    seq = ''.join([_shorten_AA(top.residue(ii), keep_index=False, substitute_fail='X') for ii in restrict_to_residxs])
    from mdciao.residue_and_atom_utils import name_from_AA as _name_from_AA
    seq_consensus= ''.join([_name_from_AA(key) for key in consensus_dict.keys()])
    alignment = _alignment_result_to_list_of_dicts(_my_bioalign(seq, seq_consensus)[0],
                                                   top,
                                                   restrict_to_residxs,
                                                   [_int_from_AA_code(key) for key in consensus_dict],
                                                   verbose=verbose
                                                   )
    alignment = _DF(alignment)
    alignment = alignment[alignment["match"] == True]
    out_list = [None for __ in top.residues]
    for idx, resSeq, AA in alignment[["idx_0","idx_1", "AA_1"]].values:
        out_list[int(idx)]=consensus_dict[AA + str(resSeq)]

    if keep_consensus:
        out_list = _fill_CGN_gaps(out_list, top, verbose=True)
    return out_list

def _uniprot_name_2_BWdf_from_gpcrdb(uniprot_name,
                                     GPCRmd="https://gpcrdb.org/services/residues/extended",
                                     verbose=True):
    url = "%s/%s"%(GPCRmd,uniprot_name)
    if verbose:
        print("requesting %s ..."%url,end="",flush=True)
    a = _requests.get(url)
    if a.text=='[]':
        raise ValueError('Uniprot name %s yields nothing'%uniprot_name)
    if verbose:
        print("done!")
    df =_read_json(a.text)
    mydict = df.T.to_dict()
    for key, val in mydict.items():
        try:
            for idict in val["alternative_generic_numbers"]:
                #print(key, idict["scheme"], idict["label"])
                val[idict["scheme"]]=idict["label"]
            val.pop("alternative_generic_numbers")
            val["AAresSeq"]='%s%s'%(val["amino_acid"],val["sequence_number"])
        except IndexError:
            pass
    DFout = _DF.from_dict(mydict, orient="index").replace({_np.nan:None})
    return DFout[["protein_segment", "AAresSeq","BW", "GPCRdb(A)", "display_generic_number"]]


def _fill_CGN_gaps(consensus_list, top, verbose=False):
    r""" Try to fill CGN consensus nomenclature gaps based on adjacent labels

    The idea is to fill gaps of the sort:
     * ['G.H5.25', 'G.H5.26', None, 'G.H.28']
      to
     * ['G.H5.25', 'G.H5.26', 'G.H.27', 'G.H.28']

    The size of the gap is variable, it just has to match the length of
    the consensus labels, i.e. 28-26=1 which is the number of "None" the
    input list had

    Parameters
    ----------
    consensus_list: list
        List of length top.n_residues with the original consensus labels
        Supossedly, it contains some "None" entries inside sub-domains
    top :
        :py:class:`mdtraj.Topology` object
    verbose : boolean, default is False

    Returns
    -------
    consensus_list: list
        The same as the input :obj:`consensus_list` with guessed missing entries
    """

    defs = _map2defs(consensus_list)
    #todo decrease verbosity
    for key, val in defs.items():

        # Identify problem cases
        if len(val)!=val[-1]-val[0]+1:
            if verbose:
                print(key)

            # Initialize residue_idxs_wo_consensus_labels control variables
            offset = int(consensus_list[val[0]].split(".")[-1])
            consensus_kept=True
            suggestions = []
            residue_idxs_wo_consensus_labels=[]

            # Check whether we can predict the consensus labels correctly
            for ii in _np.arange(val[0],val[-1]+1):
                suggestions.append('%s.%u'%(key,offset))
                if consensus_list[ii] is None:
                    residue_idxs_wo_consensus_labels.append(ii)
                else: # meaning, we have a consensus label, check it against suggestion
                    consensus_kept *= suggestions[-1]==consensus_list[ii]
                if verbose:
                    print('%6u %8s %10s %10s %s'%(ii, top.residue(ii),consensus_list[ii], suggestions[-1], consensus_kept))
                offset += 1
            if verbose:
                print()
            if consensus_kept:
                if verbose:
                    print("The consensus was kept, I am relabelling these:")
                for idx, res_idx in enumerate(_np.arange(val[0],val[-1]+1)):
                    if res_idx in residue_idxs_wo_consensus_labels:
                        consensus_list[res_idx] = suggestions[idx]
                        if verbose:
                            print(suggestions[idx])
            else:
                if verbose:
                    print("Consensus wasn't kept. Nothing done!")
            if verbose:
                print()
    return consensus_list


def _fill_BW_gaps(consensus_list, top, verbose=False):
    r""" Try to fill BW consensus nomenclature gaps based on adjacent labels

    The idea is to fill gaps of the sort:
     * ['1.25', '1.26', None, '1.28']
      to
     * ['1.25', '1.26', '1.27, '1.28']

    The size of the gap is variable, it just has to match the length of
    the consensus labels, i.e. 28-26=1 which is the number of "None" the
    input list had

    Parameters
    ----------
    consensus_list: list
        List of length top.n_residues with the original consensus labels
        Supossedly, it contains some "None" entries inside sub-domains
    top :
        :py:class:`mdtraj.Topology` object
    verbose : boolean, default is False

    Returns
    -------
    consensus_list: list
        The same as the input :obj:`consensus_list` with guessed missing entries
    """

    defs = _map2defs(consensus_list)
    for key, val in defs.items():

        # Identify problem cases
        if len(val)!=val[-1]-val[0]+1:
            if verbose:
                print(key)

            # Initialize residue_idxs_wo_consensus_labels control variables
            offset = int(consensus_list[val[0]].split(".")[-1])
            consensus_kept=True
            suggestions = []
            residue_idxs_wo_consensus_labels=[]

            # Check whether we can predict the consensus labels correctly
            for ii in _np.arange(val[0],val[-1]+1):
                suggestions.append('%s.%u'%(key,offset))
                if consensus_list[ii] is None:
                    residue_idxs_wo_consensus_labels.append(ii)
                else: # meaning, we have a consensus label, check it against suggestion
                    consensus_kept *= suggestions[-1]==consensus_list[ii]
                if verbose:
                    print(ii, top.residue(ii),consensus_list[ii], suggestions[-1], consensus_kept)
                offset += 1
            if verbose:
                print()
            if consensus_kept:
                if verbose:
                    print("The consensus was kept, I am relabelling these:")
                for idx, res_idx in enumerate(_np.arange(val[0],val[-1]+1)):
                    if res_idx in residue_idxs_wo_consensus_labels:
                        consensus_list[res_idx] = suggestions[idx]
                        if verbose:
                            print(suggestions[idx])
            if verbose:
                print()
    return consensus_list


def top2CGN_by_AAcode(top, ref_CGN_tf,
                      restrict_to_residxs=None,
                      verbose=False):
    """
    Returns a dictionary of CGN (Common G-protein Nomenclature) labelling for each residue.
    The keys are zero-indexed residue indices

    TODO if the length of the dictionary is always top.n_residues, consider simply returning a list

    Parameters
    ----------
    top :
        :py:class:`mdtraj.Topology` object
    ref_CGN_tf :
        :class:`CGN_transformer` object
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

def _relabel_consensus(idx, consensus_dicts, no_key="NA"):
    """
    Assigns labels based on the residue index
    Parameters
    ----------
    idx : int
        index for which the relabeling is needed
    consensus_dicts : list
        each item in the list should be a dictionary. The keys of each dictionary should be the residue idxs,
        and the corresponding value should be the label.
    no_key : str
        output message if there is no label for the residue idx in any of the dictionaries.

    Returns
    -------
    string
        label of the residue idx if present else "NA"

    """
    labels  = [idict[idx] for idict in consensus_dicts]
    good_label = [ilab for ilab in labels if str(ilab).lower()!="none"]
    assert len(good_label)<=1, "There can only be one good label, but for residue %u found %s"%(idx, good_label)
    try:
        return good_label[0]
    except IndexError:
        return no_key

def csv_table2TMdefs_res_idxs(itop, keep_first_resSeq=True, complete_loops=True,
                              tablefile=None,
                              reorder_by_AA_names=False):
    """

    Parameters
    ----------
    itop
    keep_first_resSeq
    complete_loops
    tablefile
    reorder_by_AA_names

    Returns
    -------

    """
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
    """
    Adds the intra- and extracellular loop definitions on the existing TM residue index dictionary.
    Example - If there are TM1-TM7 definitions with there corresponding indexes then the output will be-
        *ICL1 is added between TM1 and TM2
        *ECL1 is added between TM2 and TM3
        *ICL2 is added between TM3 and TM4
        *ECL2 is added between TM4 and TM5
        *ECL3 is added between TM6 and TM7

    Note- "ICL3" is not being explicitly added

    Parameters
    ----------
    segment_dict : dict
        TM definition as the keys and the residue idx list as values of the dictionary
    not_present : list
        definitions which should not be added to the existing TM definitions
    start_with : str
        only the string part the first definition that should be added to the existing TM definitions

    Returns
    -------
    dict
    updated dictionary with the newly added loop definitions as keys and the corresponding residue index list,
    as values. The original key-value pairs of the TM definition remains intact.

    """
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
    """
    Returns a dictionary with the TM number as key and their corresponding amino acid resSeq range as values,
    based on the BW nomenclature excel file

    Parameters
    ----------
    tablefile : xlsx file
        GPCRmd_B2AR nomenclature file in excel format, optional
    reduce_to_resSeq

    Returns
    -------
    dictionary
    with the TM definitions as the keys and the first and the last amino acid resSeq number as values.
    example- if amino acid Q26 corresponds to TM1, the output will be {'TM1' : [26, 26]}

    """

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

def _guess_nomenclature_fragments(CLtf, top, fragments,
                                  min_hit_rate=.6,
                                  verbose=False):
    """

    Parameters
    ----------
    CLtf:
        :class:`consensus_labeler` object
    top:
        :py:class:`mdtraj.Topology` object
    fragments :
    min_hit_rate: float, default is .75
        return only fragments with hit rates higher than this
    verbose: boolean
        be verbose

    Returns
    -------
    guess: list
        indices of the fragments with higher hit-rate than :obj:`cutoff`

    """
    aligned_BWs = CLtf.top2map(top, fill_gaps=False)
    #for ii, iBW in enumerate(aligned_BWs):
    #    print(ii, iBW, top.residue(ii))
    hits, guess = [], []
    for ii, ifrag in enumerate(fragments):
        hit = [aligned_BWs[jj] for jj in ifrag if aligned_BWs[jj] is not None]
        if len(hit)/len(ifrag)>=min_hit_rate:
            guess.append(ii)
        if verbose:
            print(ii, len(hit)/len(ifrag))
        hits.append(hit)
    return guess

def _map2defs(cons_list):
    r"""
    Regroup a list of consensus labels into their subdomains. The indices of the list
    are interpreted as residue indices in the topology used to generate :obj:`cons_list`
    in the first place, e.g. by using :obj:`nomenclature_utils._top2consensus_map`

    Note:
    -----
     The method will guess automagically whether this is a CGN or BW label by
     checking the type of the first character (numeric is BW, 3.50, alpha is CGN, G.H5.1)

    Parameters
    ----------
    cons_list: list
        Contains consensus labels for a given topology, s.t. indices of
        the list map to residue indices of a given topology, s.t.
        cons_list[10] has the consensus label of top.residue(10)

    Returns
    -------
    map : dictionary
        dictionary with subdomains as keys and lists of consesus labels as values
    """
    defs = _defdict(list)
    for ii, key in enumerate(cons_list):
        if key is not None:
            if key[0].isnumeric(): # it means it is BW
                new_key =key.split(".")[0]
            elif key[0].isalpha(): # it means it CGN
                new_key = '.'.join(key.split(".")[:-1])
            else:
                raise Exception(new_key)
            defs[new_key].append(ii)

    return {key: _np.array(val) for key, val in defs.items()}

def order_frags(fragment_names, consensus_labels):
    from natsort import natsorted
    labs_out = []
    for ifrag in fragment_names:
        if 'CL' in ifrag:
            toappend = natsorted([ilab for ilab in consensus_labels if ilab.endswith(ifrag)])
        else:
            toappend = natsorted([ilab for ilab in consensus_labels if ilab.startswith(ifrag)])
        if len(toappend) > 0:
            labs_out.extend(toappend)
    for ilab in consensus_labels:
        if ilab not in labs_out:
            labs_out.append(ilab)
    return labs_out

def order_BW(labels):
    return order_frags("1 12 2 23 3 34 ICL2 4 45 5 56 ICL3 6 67 7 78 8".split(), labels)
def order_CGN(labels):
    CGN_fragments = ['G.HN',
                     'G.hns1',
                     'G.S1',
                     'G.s1h1',
                     'G.H1',
                     'H.HA',
                     'H.hahb',
                     'H.HB',
                     'H.hbhc',
                     'H.HC',
                     'H.hchd',
                     'H.HD',
                     'H.hdhe',
                     'H.HE',
                     'H.hehf',
                     'H.HF',
                     'G.hfs2',
                     'G.S2',
                     'G.s2s3',
                     'G.S3',
                     'G.s3h2',
                     'G.H2',
                     'G.h2s4',
                     'G.S4',
                     'G.s4h3',
                     'G.H3',
                     'G.h3s5',
                     'G.S5',
                     'G.s5hg',
                     'G.HG',
                     'G.hgh4',
                     'G.H4',
                     'G.h4s6',
                     'G.S6',
                     'G.s6h5',
                     'G.H5']
    return order_frags(CGN_fragments,labels)

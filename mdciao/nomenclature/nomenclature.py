##############################################################################
#    This file is part of mdciao.
#    
#    Copyright 2020 Charité Universitätsmedizin Berlin and the Authors
#
#    Authors: Guillermo Pérez-Hernandez
#    Contributors:
#
#    mdciao is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    mdciao is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with mdciao.  If not, see <https://www.gnu.org/licenses/>.
##############################################################################

import mdtraj as _md
import numpy as _np

import mdciao.fragments as _mdcfrg
import mdciao.utils as _mdcu

from pandas import \
    read_json as _read_json, \
    read_excel as _read_excel, \
    read_csv as _read_csv, \
    DataFrame as _DataFrame

from collections import defaultdict as _defdict

from os import path as _path

import requests as _requests

from natsort import natsorted as _natsorted

def table2BW_by_AAcode(tablefile,
                       keep_AA_code=True,
                       return_fragments=False,
                       ):
    r"""
    Reads an excel table and returns a dictionary AAcodes so that e.g. self.AA2BW[R131] -> '3.50'

    Parameters
    ----------
    tablefile : xlsx file or pandas dataframe
        Ballesteros-Weinstein nomenclature file in excel format, optional
    keep_AA_code : boolean, default is True
        If True then output dictionary will have key of the form "Q26" else "26".
    return_fragments : boolean, default is True
        return a dictionary of fragments keyed by BW-fragment, e.g. "TM1"

    Returns
    -------

    AA2BW : dictionary
        Dictionary with residues as key and their corresponding BW notation.

    fragments : dict (optional)
        if return_fragments=True, a dictionary containing the fragments according to the excel file
    """

    if isinstance(tablefile,str):
        df = _read_excel(tablefile, header=0, engine="openpyxl")
    else:
        df = tablefile

    # TODO some overlap here with with _BW_web_lookup of BW_finder
    # figure out best practice to avoid code-repetition
    # This is the most important
    AAcode2BW = {key: str(val) for key, val in df[["AAresSeq", "BW"]].values}
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

def PDB_finder(PDB_code, local_path='.',
               try_web_lookup=True,
               verbose=True):
    r"""Return an :obj:`~mdtraj.Trajectory` by loading a local
    file or optionally looking up online, see :obj:`md_load_rscb`

    Note
    ----
    Since filenames are case-sensitive, e.g. 3CAP will not
    find `3cap.pdb` locally, but will be successfully found
    online (urls are not case-sensitive), returning the
    online file instead of the local one, which can lead
    to "successful" but wrong behaviour if the local
    file had already some modifications (strip non protein etc)

    Parameters
    ----------
    PDB_code : str
        4-letter PDB code
    local_path : str, default is "."
        What directory to look into
    try_web_lookup : bool, default is True
        If the file :obj:`ref_PDB` cannot be found locally
        as .pdb or .pdb.gz, a web lookup will be tried
        using :obj:`md_load_rscb`
    verbose : boolean, default is True

    Returns
    -------
    geom : :obj:`~mdtraj.Trajectory`
    return_file : str with filename,
        Will contain an url if web_lookup was necessary
    """
    try:
        file2read = _path.join(local_path, PDB_code + '.pdb')
        _geom = _md.load(file2read)
        return_file = file2read
    except (OSError, FileNotFoundError):
        try:
            file2read = _path.join(local_path, PDB_code + '.pdb.gz')
            _geom = _md.load(file2read)
            return_file = file2read
        except (OSError, FileNotFoundError):
            if verbose:
                print("No local PDB file for %s found in directory '%s'" % (PDB_code, local_path), end="")
            if try_web_lookup:
                _geom, return_file = md_load_rscb(PDB_code,
                                                  verbose=verbose,
                                                  return_url=True)
                if verbose:
                    print("found! Continuing normally")

            else:
                raise

    return _geom, return_file

def CGN_finder(identifier,
               format='CGN_%s.txt',
               local_path='.',
               try_web_lookup=True,
               verbose=True,
               dont_fail=False,
               write_to_disk=False):
    r"""Provide a four-letter PDB code and look up (first locally, then online)
    for a file that contains the Common-Gprotein-Nomenclature (CGN)
    consesus labels and return them as a :obj:`DataFrame`. See
    https://www.mrc-lmb.cam.ac.uk/CGN/ for more info on this nomenclature
    and :obj:`_finder_writer` for what's happening under the hood


    Parameters
    ----------
    identifier : str
        Typically, a PDB code
    format : str
        A format string that turns the :obj:`identifier`
        into a filename for local lookup, in case the
        user has custom filenames, e.g. 3SN6_consensus.txt
    local_path : str
        The local path to the local consensus file
    try_web_lookup : bool, default is True
        If the local lookup fails, go online
    verbose : bool, default is True
    dont_fail : bool, default is False
        Do not raise any errors that would interrupt
        a workflow and simply return None

    Returns
    -------
    DF : :obj:`DataFrame` with the consensus nomenclature

    References
    ----------
    * [2] Flock, T., Ravarani, C., Sun, D. et al.,
     *Universal allosteric mechanism for Gα activation by GPCRs.*
     Nature 524, 173–179 (2015)
     `<https://doi.org/10.1038/nature14663>`
    """
    file2read = format%identifier
    file2read = _path.join(local_path, file2read)
    rep = lambda istr : [istr.replace(" ","") if isinstance(istr,str) else istr][0]
    # using  delim_whitespace=True splits "Sort Number" in two keys, not feasible to generalize ATM
    local_lookup_lambda = lambda file2read :_read_csv(file2read, delimiter='\t').applymap(rep)

    web_address = "www.mrc-lmb.cam.ac.uk"
    url = "https://%s/CGN/lookup_results/%s.txt" % (web_address, identifier)
    web_lookup_lambda = local_lookup_lambda

    print("Using CGN-nomenclature, please cite the following 3rd party publications:\n"
          " * https://doi.org/10.1038/nature14663 (Babu et al 2015)")
    return _finder_writer(file2read, local_lookup_lambda,
                          url, web_lookup_lambda,
                          try_web_lookup=try_web_lookup,
                          verbose=verbose,
                          dont_fail=dont_fail,
                          write_to_disk=write_to_disk)

def _finder_writer(full_local_path,
                   local2DF_lambda,
                   full_web_address,
                   web2DF_lambda,
                   try_web_lookup=True,
                   verbose=True,
                   dont_fail=False,
                   write_to_disk=False):
    r"""
    Try local lookup with a local lambda, then web lookup with a
    web labmda and try to return a :obj:`DataFrame`
    Parameters
    ----------
    full_local_path
    full_web_address
    local2DF_lambda
    web2DF_lambda
    try_web_lookup
    verbose
    dont_fail

    Returns
    -------
    df : DataFrame or None

    """
    try:
        return_name = full_local_path
        _DF = local2DF_lambda(full_local_path)
        print("%s found locally."%full_local_path)
    except FileNotFoundError as e:
        _DF = e
        if verbose:
            print("No local file %s found" % full_local_path, end="")
        if try_web_lookup:
            return_name = full_web_address
            if verbose:
                print(", checking online in\n%s ..." % full_web_address, end="")
            try:
                _DF = web2DF_lambda(full_web_address)
                if verbose:
                    print("done without 404, continuing.")
            except Exception as e:
                print('Error getting or processing the web lookup:', e)
                _DF = e

    if isinstance(_DF, _DataFrame):
        if write_to_disk:
            if _path.exists(full_local_path):
                raise FileExistsError("Cannot overwrite exisiting file %s" % full_local_path)
            if _path.splitext(full_local_path)[-1]==".xlsx":
                _DF.to_excel(full_local_path)
            else:
                # see https://github.com/pandas-dev/pandas/issues/10415
                # This saves all values tab-separated s.t.
                # the resulting file can be re-read by pandas.read_csv
                _np.savetxt(full_local_path, _DF.to_numpy(str),
                fmt='%10s',
                delimiter="\t", header='\t'.join(_DF.keys()), comments='')


            print("wrote %s for future use" % full_local_path)
        return _DF, return_name
    else:
        if dont_fail:
            return None, return_name
        else:
            raise _DF


# TODO consider making private?
def BW_finder(BW_descriptor,
              format = "%s.xlsx",
              local_path=".",
              try_web_lookup=True,
              verbose=True,
              dont_fail=False,
              write_to_disk=False):
    r"""
    Return a :obj:`~pandas.DataFrame` containing
    a `Ballesteros-Weinstein-Numbering (BW)
    <https://doi.org/10.1016/S1043-9471(05)80049-7>`_ [1].

    The lookup is first local and then online
    on the `GPCRdb <https://gpcrdb.org/>`_ [3].

    This method wraps (with some python lambdas) around
    :obj:`_finder_writer`.

    Parameters
    ----------
    BW_descriptor : str
        Anything that can be used to find the needed
        BW information, locally or online:
         * a uniprot descriptor, e.g. `adrb2_human`
         * a full local filename, e.g. `my_BW.txt` or
          `path/to/my_BW.txt`
         * the "basename" filename, `adrb2_human` if
          `adrb2_human.xlsx` exists on :obj:`local_path`
          (see below :obj:`format`)
        All these ways of doing the same thing (descriptor, basename, fullname,
        localpath, fullpath) are for compatibility with other methods.
    format : str, default is "%s.xlsx".
        If :obj:`BW_descriptor` is not readable directly,
        try to find "BW_descriptor.xlsx" locally on :obj:`local_path`
    local_path : str, default is "."
        If :obj:`BW_descriptor` doesn't find the file locally,
        then try "local_path/BW_descriptor" before trying online
    try_web_lookup : boolean, default is True.
        If local lookup variants fail, go online, else Fail
    verbose : bool, default is False
        Be verbose.
    dont_fail : bool, default is False
        If True, when the lookup fails None will
        be returned. By default the method raises
        an exception if it could not find the info.
    write_to_disk : boolean, default is False
        Save the found BW info locally.

    Returns
    -------
    df : DataFrame or None
        The BW information as :obj:`~pandas.DataFrame`

    References
    ----------
    * [1] Juan A. Ballesteros, Harel Weinstein,
     *[19] Integrated methods for the construction of three-dimensional models and computational probing
     of structure-function relations in G protein-coupled receptors*,
     Editor(s): Stuart C. Sealfon, Methods in Neurosciences, Academic Press, Volume 25, 1995
     `<https://doi.org/10.1016/S1043-9471(05)80049-7>`_

    * [3] Gáspár Pándy-Szekeres, Christian Munk, Tsonko M Tsonkov, Stefan Mordalski,
     Kasper Harpsøe, Alexander S Hauser, Andrzej J Bojarski, David E Gloriam,
     *GPCRdb in 2018: adding GPCR structure models and ligands*,
     Nucleic Acids Research, Volume 46, Issue D1, 4 January 2018, Pages D440–D446,
     `<https://doi.org/10.1093/nar/gkx1109>`_

    """

    if _path.exists(BW_descriptor):
        fullpath = BW_descriptor
        try_web_lookup=False
    else:
        xlsxname = format % BW_descriptor
        fullpath = _path.join(local_path, xlsxname)
    GPCRmd = "https://gpcrdb.org/services/residues/extended"
    url = "%s/%s" % (GPCRmd, BW_descriptor)

    local_lookup_lambda = lambda fullpath : _read_excel(fullpath,
                                                        engine="openpyxl",
                                                        usecols=lambda x : x.lower()!="unnamed: 0",
                                                        converters={"BW": str}).replace({_np.nan: None},)
    web_looukup_lambda = lambda url : _BW_web_lookup(url, verbose=verbose)
    print("Using BW-nomenclature, please cite the following 3rd party publications:\n"
          " * https://doi.org/10.1016/S1043-9471(05)80049-7 (Weinstein et al 1995)\n"
          " * https://doi.org/10.1093/nar/gkx1109 (Gloriam et al 2018)")
    return _finder_writer(fullpath, local_lookup_lambda,
                          url, web_looukup_lambda,
                          try_web_lookup=try_web_lookup,
                          verbose=verbose,
                          dont_fail=dont_fail,
                          write_to_disk=write_to_disk)

def _BW_web_lookup(url, verbose=True,
                   timeout=5):
    r"""
    Lookup this url for a BW-notation
    return a ValueError if the lookup retuns an empty json
    Parameters
    ----------
    url
    verbose
    timeout : float, default is 1
        Timout in seconds for :obj:`_requests.get`
        https://requests.readthedocs.io/en/master/user/quickstart/#timeouts
    Returns
    -------

    """
    uniprot_name = url.split("/")[-1]
    a = _requests.get(url, timeout=timeout)
    #TODO use _url2json here
    if verbose:
        print("done!")
    if a.text == '[]':
        DFout = ValueError('Contacted %s url sucessfully (no 404),\n'
                           'but Uniprot name %s yields nothing' % (url, uniprot_name))
    else:
        df = _read_json(a.text)
        mydict = df.T.to_dict()
        fmt = 'old'
        for key, val in mydict.items():
            try:
                val["AAresSeq"] = '%s%s' % (val["amino_acid"], val["sequence_number"])
                if "alternative_generic_numbers" in val.keys():
                    for idict in val["alternative_generic_numbers"]:
                        # print(key, idict["scheme"], idict["label"])
                        val[idict["scheme"]] = idict["label"]
                    val.pop("alternative_generic_numbers")
                else:
                    fmt = 'new'
                    pass

            except IndexError:
                pass

        return_fields = \
            {"old": [
                "protein_segment",
                "AAresSeq",
                "BW",
                "GPCRdb(A)",
                "display_generic_number"],
            "new": [
                    "protein_segment",
                    "AAresSeq",
                    "display_generic_number"]}
        DFout = _DataFrame.from_dict(mydict, orient="index").replace({_np.nan: None})
        DFout = DFout[return_fields[fmt]]

    return DFout

#todo document and refactor to better place?
def md_load_rscb(PDB,
                 web_address = "https://files.rcsb.org/download",
                 verbose=False,
                 return_url=False):
    r"""
    Input a PDB code get an :obj:`~mdtraj.Trajectory` object.

    Thinly wraps around :obj:`~mdtraj.load_pdb` by constructing
    the url for the user.

    Parameters
    ----------
    PDB : str
        4-letter PDB code
    web_address: str, default is "https://files.rcsb.org/download"
    verbose : bool, default is False
        Be versose
    return_url : bool, default is False
        also return the actual url that was checked

    Returns
    -------
    traj : :obj:`~mdtraj.Trajectory`
    url  : str, optional
    """
    url = '%s/%s.pdb' % (web_address, PDB)
    if verbose:
        print(", checking online in \n%s ..." % url, end="")
    igeom = _md.load_pdb(url)
    if return_url:
        return igeom, url
    else:
        return igeom

class LabelerConsensus(object):
    """Parent class to manage consensus notations

    At the moment child classe are
     * :obj:`LabelerBW` for Ballesteros-Weinstein (BW)
     * :obj:`LabelerCGN` for Common-Gprotein-nomenclature (CGN)

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
        self._geom_PDB = None
        self._ref_top = None
        self._ref_PDB = ref_PDB
        if ref_PDB is not None:
            self._geom_PDB, self._PDB_file = PDB_finder(ref_PDB,
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
        r""" :obj:`~mdtraj.Trajectory` with with what was found
        (locally or online) using :obj:`ref_PDB`"""
        return self._geom_PDB

    @property
    def top(self):
        r""" :obj:`~mdtraj.Topology` with with what was found
                (locally or online) using :obj:`ref_PDB`"""
        return self._geom_PDB.top

    @property
    def seq(self):
        r""" The reference sequence in :obj:`dataframe`"""
        return ''.join([_mdcu.residue_and_atom.name_from_AA(val) for val in self.dataframe[self._AAresSeq_key].values.squeeze()])

    @property
    def conlab2AA(self):
        r""" Dictionary with consensus labels as keys, so that e.g.
            * self.conlab2AA["3.50"] -> 'R131' or
            * self.conlab2AA["G.hfs2.2"] -> 'R201' """
        return self._conlab2AA

    @property
    def AA2conlab(self):
        r""" Dictionary with short AA-codes as keys, so that e.g.
            * self.AA2BW["R131"] -> '3.50'
            * self.conlab2AA["R201"] -> "G.hfs2.2" """


        return self._AA2conlab

    @property
    def idx2conlab(self):
        r""" List of consensus labels in the order (=idx) they appear in the original dataframe

        This index is the row-index of the table, don't count on it being aligned with anything"""
        return self._idx2conlab
    @property
    def fragment_names(self):
        r"""Name of the fragments according to the consensus labels

        TODO OR NOT? Check!"""
        return self._fragment_names

    @property
    def fragments(self):
        r""" Dictionary of fragments keyed with fragment names
        and valued with the residue names (AAresSeq) in that fragment.
        """

        return self._fragments

    @property
    def fragments_as_conlabs(self):
        r""" Dictionary of fragments keyed with fragment names
        and valued with the consensus labels in that fragment

        Returns
        -------
        """
        return self._fragments_as_conlabs

    @property
    def dataframe(self):
        r"""
        :obj:`~pandas.DataFrame` summarizing this object's information

        Returns
        -------
        df : :obj:`~pandas.DataFrame`
        """
        return self._dataframe

    @property
    def tablefile(self):
        r""" The file used to instantiate this transformer"""
        return self._tablefile

    def conlab2residx(self, top,
                      map=None,
                      **top2labels_kwargs,
                      ):
        r"""
        Returns a dictionary keyed by consensus labels and valued
        by residue indices of the input topology in :obj:`top`.

        The default behaviour is to internally align :obj:`top`
        with the object's available consensus dictionary
        on the fly using :obj:`self.top2labels`. See the docs
        there for **top2labels_kwargs, in particular
        restrict_to_residxs, keep_consensus, and min_hit_rate

        Note
        ----
        This method is able to work with a new topology every
        time, performing a sequence alignment every call.
        The intention is to instantiate a
        :obj:`LabelerConsensus` just one time and use it with as
        many topologies as you like without changing any attribute
        of :obj:`self`.

        HOWEVER, if you know what you are doing, you can provide a
        list of consensus labels yourself using :obj:`map`. Then,
        this method is nothing but a table lookup (almost)

        Warning
        -------
        No checks are performed to see if the input of :obj:`map`
        actually matches the residues of :obj:`top` in any way,
        so that the output can be rubbish and go unnoticed.

        Parameters
        ----------
        top : :obj:`~mdtraj.Topology`
        map : list, default is None
            A pre-computed residx2consensuslabel map, i.e. the
            output of a previous, external call to :obj:`_top2consensus_map`
            If it contains duplicates, it is a malformed list.
            See the note above for more info

        Returns
        -------
        dict : keyed by consensus labels and valued with residue idxs
        """
        if map is None:
            map = self.top2labels(top,
                                  **top2labels_kwargs,
                                  )
        out_dict = {}
        for ii,imap in enumerate(map):
            if imap is not None and str(imap).lower()!="none":
                if imap in out_dict.keys():
                    raise ValueError("Entries %u and %u of the map, "
                                     "i.e. residues %s and %s of the input topology "
                                     "both have the same label %s.\n"
                                     "This method cannot work with a map like this!"%(out_dict[imap], ii,
                                                                                     top.residue(out_dict[imap]),
                                                                                     top.residue(ii),
                                                                                     imap))
                else:
                    out_dict[imap]=ii
        return out_dict

    def top2labels(self, top,
                   autofill_consensus=True,
                   **aligntop_kwargs):

        r""" Align the sequence of :obj:`top` to the sequence used
        to initialize this :obj:`LabelerConsensus` and return a
        list of consensus labels for each residue in :obj:`top`.

        Populates the attributes :obj:`most_recent_top2labels` and
        :obj:`most_recent_alignment`

        If a consensus label is returned as None it means one
        of two things:
         * this position was successfully aligned with a
           match but the data used to initialize this
           :obj:`ConsensusLabeler` did not contain a label
         * this position has a label in the original data
           but the sequence alignment is not matched (e.g.,
           bc of a point mutation)

        A heuristic to "autofill" the second case can be
        turned on using :obj:`autofill_consensus`,
        see :obj:`_fill_consensus_gaps`
        for more info

        Note
        ----
        This method uses :obj:`aligntop` internally,
        see the doc on that method for more info.

        Parameters
        ----------
        top :
            :obj:`~mdtraj.Topology` object
        autofill_consensus : boolean default is False
            Even if there is a consensus mismatch with the sequence of the input
            :obj:`AA2conlab_dict`, try to relabel automagically, s.t.
             * ['G.H5.25', 'G.H5.26', None, 'G.H.28']
             will be grouped relabeled as
             * ['G.H5.25', 'G.H5.26', 'G.H.27', 'G.H.28']
        min_hit_rate : float, default is .5
            With big topologies and many fragments,
            the alignment method (:obj:`mdciao.sequence.my_bioalign`)
            sometimes yields sub-optimal results. A value
            :obj:`min_hit_rate` >0, e.g. .5 means that a pre-alignment
            takes place to populate :obj:`restrict_to_residxs`
            with indices of those the fragments
            (:obj:`mdciao.fragments.get_fragments` defaults)
            with more than 50% alignment in the pre-alignment.
            If :obj:`min_hit_rate`>0, :obj`restrict_to_residx`
            has to be None.
        restrict_to_residxs: iterable of integers, default is None
            Instead of guessing using :obj:`min_hit_rate`,
            use only these residues for alignment and labelling options.
            The return list will still be of length=top.n_residues

        Returns
        -------
        map : list of len = top.n_residues with the consensus labels
        """

        self.aligntop(top, **aligntop_kwargs)
        out_list = alignment_df2_conslist(self.most_recent_alignment)
        out_list = out_list + [None for __ in range(top.n_residues - len(out_list))]
        # TODO we could do this padding in the alignment_df2_conslist method itself
        # with an n_residues optarg, IDK about best design choice
        if autofill_consensus:
            out_list = _fill_consensus_gaps(out_list, top, verbose=False)

        self._last_top2labels = out_list
        return out_list

    def top2frags(self, top,
                  fragments=None,
                  min_hit_rate=.5,
                  input_dataframe=None,
                  show_alignment=False,
                  verbose=True,
                  ):
        r"""
        Return the subdomains derived from the
        consensus nomenclature and map it out
        in terms of residue indices of the input :obj:`top`

        Note
        ----
        This method uses :obj:`aligntop` internally,
        see the doc on that method for more info.

        Parameters
        ----------
        top:
            :py:class:`~mdtraj.Topology` object
        fragments: iterable of integers, default is None
            The user can parse an existing list of fragment-definitions
            (via residue idxs) to check if the newly found, consensus
            definitions (:obj:`defs`) clash with the input in :obj:`fragments`.
            *Clash* means that the consensus definitions span over more
            than one of the fragments in defined in :obj:`fragments`.

            An interactive prompt will ask the user which fragments to
            keep in case of clashes.

            Check :obj:`check_if_subfragment` for more info
        min_hit_rate : float, default is .5
            With big topologies, like a receptor-Gprotein system,
            the "brute-force" alignment method
            (check :obj:`mdciao.sequence.my_bioalign`)
            sometimes yields sub-optimal results, e.g.
            finding short snippets of reference sequence
            that align in a completely wrong part of the topology.
            To avoid this, an initial, exploratory alignment
            is carried out. :obj:`min_hit_rate` = .5 means that
            only the fragments (:obj:`mdciao.fragments.get_fragments` defaults)
            with more than 50% alignment in this exploration
            are used to improve the second alignment
        input_dataframe : :obj:`~pandas.DataFrame`, default is None
            Expert option, use at your own risk.
            Instead of aligning :obj:`top` to
            the object's sequence to derive
            fragment definitions, input
            an existing alignment here, e.g.
            the self.most_recent_alignment
        show_alignment : bool, default is False,
            Show the entire alignment as :obj:`~pandas.DataFrame`
        verbose : bool, default is True
            Also print the definitions

        Returns
        -------
        defs : dictionary
            Dictionary with subdomain names as keys
            and lists of indices as values
        """

        if input_dataframe is None:
            top2self, self2top = self.aligntop(top, min_hit_rate=min_hit_rate, verbose=show_alignment)
        else:
            top2self, self2top = _mdcu.sequence.df2maps(input_dataframe)

        # TODO "topmaps" are more straighfdw than dataframes
        # but in pple we could make a lot of the next bookeeping as df2xxx functions

        frags =  self.fragments_as_idxs
        defs = {key:[self2top[idx] for idx in val if idx in self2top.keys()] for key,val in frags.items()}
        defs = {key:val for key, val in defs.items() if len(val)>0}

        new_defs = {}
        map_conlab = [self.idx2conlab[top2self[topidx]] if topidx in top2self.keys() else None for topidx in range(top.n_residues)]

        for ii, (key, res_idxs) in enumerate(defs.items()):
            if fragments is not None:
                new_defs[key] = _mdcfrg.check_if_subfragment(res_idxs, key, fragments, top, map_conlab)

        for key, res_idxs in new_defs.items():
            defs[key]=res_idxs

        for ii, (key, res_idxs) in enumerate(defs.items()):
            istr = _mdcfrg.print_frag(key, top, res_idxs, fragment_desc='',
                               idx2label=map_conlab,
                               return_string=True)
            if verbose:
                print(istr)

        return dict(defs)

    def aligntop(self, top,
                 restrict_to_residxs=None,
                 min_hit_rate=0,
                 verbose=False):
        r""" Align a topology with the object's sequence.
        Returns two maps (top2self, self2top) and
        populates the attribute self.most_recent_alignment

        Wraps around :obj:`mdciao.utils.sequence.align_tops_or_seqs`

        The indices of self are indices (row-indices)
        of the original :obj:`LabelerConsensus.dataframe`,
        which are the the ones in :obj:`ConsensusLabelers.seq`

        top : :obj:`~mdtraj.Topology` object
        restrict_to_residxs: iterable of integers, default is None
            Use only these residues for alignment and labelling purposes
            Helps "guide" the alignment method. E.g., one might be
            passing an Ballesteros-Weinstein in :obj:`AA2conlab_dict` but
            the topology also contains the whole G-protein. If available,
            one can pass here the indices of residues of the receptor
        min_hit_rate : float, default .5
            With big topologies and many fragments,
            the alignment method (:obj:`mdciao.sequence.my_bioalign`)
            sometimes yields sub-optimal results. A value
            :obj:`min_hit_rate` >0, e.g. .5 means that a pre-alignment
            takes place to populate :obj:`restrict_to_residxs`
            with indices of those the fragments
            (:obj:`mdciao.fragments.get_fragments` defaults)
            with more than 50% alignment in the pre-alignment.
            If :obj:`min_hit_rate`>0, :obj`restrict_to_residx`
            has to be None.
        verbose: boolean, default is False
            be verbose
        """

        if min_hit_rate > 0:
            assert restrict_to_residxs is None
            restrict_to_residxs = guess_nomenclature_fragments(self.seq,
                                                               top,
                                                               _mdcfrg.get_fragments(top, verbose=False),
                                                               min_hit_rate=min_hit_rate,
                                                               return_residue_idxs=True, empty=None)

        df = _mdcu.sequence.align_tops_or_seqs(top,
                                               self.seq,
                                               seq_0_res_idxs=restrict_to_residxs,
                                               return_DF=True,
                                               verbose=verbose)
        top2self, self2top = _mdcu.sequence.df2maps(df)

        conlab = _np.full(len(df),None)
        conlab[_np.flatnonzero(df["idx_0"].isin(top2self.keys()))]=self.dataframe.iloc[list(top2self.values())][self._nomenclature_key]
        df = df.join(_DataFrame({"conlab": conlab}))

        self._last_alignment_df = df

        return top2self, self2top

    @property
    def most_recent_alignment(self):
        r"""A :obj:`~pandas.DataFrame` with the most recent alignment

        Expert use only

        Returns
        -------
        df : :obj:`~pandas.DataFrame`

        """
        try:
            return self._last_alignment_df
        except AttributeError:
            print("No alignment has been carried out with this object yet")
            return None

    @property
    def most_recent_top2labels(self):
        r"""The most recent :obj:`self.top2labels`-result

        Expert use only

        Returns
        -------
        df : list

        """
        try:
            return self._last_top2labels
        except AttributeError:
            print("No alignment has been carried out with this object yet")
            return None

class LabelerCGN(LabelerConsensus):
    """
    Class to abstract, handle, and use common-Gprotein-nomenclature.
    See https://www.mrc-lmb.cam.ac.uk/CGN/faq.html for more info.
    """

    def __init__(self, PDB_input,
                 local_path='.',
                 try_web_lookup=True,
                 verbose=True,
                 write_to_disk=None):
        r"""

        Parameters
        ----------
        PDB_input: str
            The PDB-file to be used. For compatibility reasons, there's different use-cases.

            * Full path to an existing file containing the CGN nomenclature,
            e.g. '/abs/path/to/some/dir/CGN_ABCD.txt' (or ABCD.txt). Then this happens:
                * :obj:`local_path` gets overridden with '/abs/path/to/some/dir/'
                * a PDB four-letter-code is inferred from the filename, e.g. 'ABCD'
                * a file '/abs/path/to/some/dir/ABCD.pdb(.gz)' is looked for
                * if not found and :obj:`try_web_lookup` is True, then
                  'ABCD' is looked up online in the PDB rscb database

            * Full path to an existing PDB-file, e.g.
              '/abs/path/to/some/dir/ABCD.pdb(.gz)'. Then this happens:
                * :obj:`local_path` gets overridden with '/abs/path/to/some/dir/'
                * a file '/abs/path/to/some/dir/CGN_ABCD.txt is looked for
                * if not found and :obj:`try_web_lookup` is True, then
                  'ABCD' is looked up online in the CGN database

            * Four letter code, e.g. 'ABCD'. Then this happens:
                * look for the files '3SN6.pdb' and 'CGN_3SN6.txt' in :obj:`local_path`
                * if one or both of these files cannot be found there,
                  look up in their respective online databases (if
                  :obj:`try_web_lookup` is True)

        Note
        ----
            The intention behind this flexibility (which is hard to document and
            maintain) is to keep the signature of consensus labelers somewhat
            consistent for compatibility with other command line methods

        local_path: str, default is '.'
            The local path where these files exist, if they exist

        try_web_lookup: bool, default is True
            If the local files are not found, try automatically a web lookup at
            * www.mrc-lmb.cam.ac.uk (for CGN)
            * rcsb.org (for the PDB)
        """

        self._nomenclature_key="CGN"

        # TODO see fragment_overview...are there clashes
        if _path.exists(PDB_input):
            local_path, basename = _path.split(PDB_input)
            PDB_input = _path.splitext(basename)[0].replace("CGN_", "")
            # TODO does the check need to have the .txt extension?
            # TODO do we even need this check?
            #assert len(PDB_input) == 4 and "CGN_%s.txt" % PDB_input == basename
        self._dataframe, self._tablefile = CGN_finder(PDB_input,
                                                      local_path=local_path,
                                                      try_web_lookup=try_web_lookup,
                                                      verbose=verbose,
                                                      write_to_disk=write_to_disk)
        # The title of the column with this field varies between CGN and BW
        AAresSeq_key = [key for key in list(self.dataframe.keys()) if key.lower() not in [self._nomenclature_key.lower(), "Sort number".lower()]]
        assert len(AAresSeq_key)==1
        self._AAresSeq_key = AAresSeq_key

        self._AA2conlab = {key: self._dataframe[self._dataframe[PDB_input] == key][self._nomenclature_key].to_list()[0]
                           for key in self._dataframe[PDB_input].to_list()}

        self._idx2conlab = self.dataframe[self._nomenclature_key].values.tolist()


        self._fragments = _defdict(list)
        for ires, key in self.AA2conlab.items():
            try:
                new_key = '.'.join(key.split(".")[:-1])
            except:
                print(key)
            #print(key,new_key)
            self._fragments[new_key].append(ires)
        LabelerConsensus.__init__(self, ref_PDB=PDB_input,
                                  local_path=local_path,
                                  try_web_lookup=try_web_lookup,
                                  verbose=verbose)

    @property
    def fragments_as_idxs(self):
        r""" Dictionary of fragments keyed with fragment names
        and valued with idxs of the first column of self.dataframe,
        regardless of these residues having a consensus label or not

        Returns
        -------
        """
        AAresSeq_list = self.dataframe[self._AAresSeq_key].values.squeeze()
        assert len(_np.unique(AAresSeq_list))==len(AAresSeq_list),"Redundant residue names in the dataframe? Somethings wrong"
        AAresSeq2idx = {key:idx for idx,key in enumerate(AAresSeq_list)}
        defs =  {key: [AAresSeq2idx[AAresSeq] for AAresSeq in val] for key, val in self.fragments.items()}
        return defs


class LabelerBW(LabelerConsensus):
    """Manipulate Ballesteros-Weinstein notation

    """
    def __init__(self, uniprot_name,
                 ref_PDB=None,
                 local_path=".",
                 format="%s.xlsx",
                 verbose=True,
                 try_web_lookup=True,
                 #todo write to disk should be moved to the superclass at some point
                 write_to_disk=False):
        r"""

        Parameters
        ----------
        uniprot_name : str
            Descriptor by which to find the nomenclature,
            it gets directly passed to :obj:`BW_finder`
            Can be several anything that can be used to try and find,
            the needed information, locally or online:
            * a uniprot descriptor, e.g. `adrb2_human`
            * a full local filename
            * a part of a local filename
        ref_PDB
        local_path
        format
        verbose
        try_web_lookup
        write_to_disk
        """

        self._nomenclature_key = "BW"
        # TODO now that the finder call is the same we could
        # avoid cde repetition here
        self._dataframe, self._tablefile = BW_finder(uniprot_name,
                                            format=format,
                                            local_path=local_path,
                                            try_web_lookup=try_web_lookup,
                                            verbose=verbose,
                                            write_to_disk=write_to_disk
                                       )
        # The title of the column with this field varies between CGN and BW
        self._AAresSeq_key = "AAresSeq"
        self._AA2conlab, self._fragments = table2BW_by_AAcode(self.dataframe, return_fragments=True)
        self._idx2conlab = self.dataframe[self._nomenclature_key].values.tolist()
        # TODO can we do this using super?
        LabelerConsensus.__init__(self, ref_PDB,
                                  local_path=local_path,
                                  try_web_lookup=try_web_lookup,
                                  verbose=verbose)

        self._uniprot_name = uniprot_name

    @property
    def uniprot_name(self):
        return self._uniprot_name

    @property
    def fragments_as_idxs(self):
        r""" Dictionary of fragments keyed with fragment names
        and valued with idxs of the first column of self.dataframe,
        regardless of these residues having a consensus label or not

        Returns
        -------
        """

        return {key: list(self.dataframe[self.dataframe["protein_segment"] == key].index) for key in
                self.dataframe["protein_segment"].unique()}


def alignment_df2_conslist(alignment_as_df,
                           allow_nonmatch=False):
    r"""
    Build a list with consensus labels out of an alignment and a consensus dictionary.

    Parameters
    ----------
    alignment_as_df : :obj:`~pandas.DataFrame`
        The alignment of the target sequence
        to the reference sequence
    allow_nonmatch : bool, default is False
        If True, the consensus labels of
        non-matching residues will be used
        if there's alignment. E.g. if a
        position is mutated there's no
        identity match, but still want to
        use that consensus label.

    Returns
    -------
    consensus_labels : list
        List of consensus labels (when available, else None)
         up to the highest residue idx in "idx_0"
         of the alignment DF
    """

    n_residues = _np.max([int(ival) for ival in alignment_as_df["idx_0"].values if str(ival).isdigit()])
    out_list = _np.full(n_residues+1, None)

    if allow_nonmatch:
        _df = _mdcu.sequence.re_match_df(alignment_as_df)
    else:
        _df = alignment_as_df
    _df = _df[_df["match"]]

    out_list[_df["idx_0"].values.astype(int)] = _df["conlab"].values
    return out_list.tolist()

def _fill_consensus_gaps(consensus_list, top, verbose=False):
    r""" Try to fill CGN consensus nomenclature gaps based on adjacent labels

    The idea is to fill gaps of the sort:
     * ['G.H5.25', 'G.H5.26', None, 'G.H.28']
      to
     * ['G.H5.25', 'G.H5.26', 'G.H.27', 'G.H.28']

    or equivalently:
     * ['3.48', '3.49', None, '3.51']
      to
     * ['3.48', '3.49', "3.50", '3.51']

    The size of the gap is variable, it just has to match the length of
    the consensus labels, i.e. 28-26=1 which is the number of "None" the
    input list had

    Parameters
    ----------
    consensus_list: list
        List of length top.n_residues with the original consensus labels
        Supossedly, it contains some "None" entries inside sub-domains
    top :
        :py:class:`~mdtraj.Topology` object
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

'''
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
        :py:class:`~mdtraj.Topology` object
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
'''

'''
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
        :py:class:`~mdtraj.Topology` object
    ref_CGN_tf :
        :class:`LabelerCGN` object
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
    for alignmt in my_bioalign(seq, ref_CGN_tf.seq)[:1]:
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
            from .sequence_utils import print_verbose_dataframe
            print_verbose_dataframe(pd.DataFrame.from_dict(list_of_alignment_dicts))
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
'''

def choose_between_consensus_dicts(idx, consensus_maps, no_key="NA"):
    """
    Choose the best consensus label for a given :obj:`idx` in case
    there are more than one consensus(es) at play (e.g. BW and CGN).

    Wil raise error if both dictionaries have a consensus label for
    the same index (unsual case)

    ----------
    idx : int
        index for which the relabeling is needed
    consensus_maps : list
        The items in the list should be "gettable" by using :obj:`idx`,
        either by being lists, arrays, or dicts, s.t.,
        the corresponding value should be the label.
    no_key : str
        output message if there is no label for the residue idx in any of the dictionaries.

    Returns
    -------
    string
        label of the residue idx if present else :obj:`no_key`

    """
    labels  = [idict[idx] for idict in consensus_maps]
    good_label = _np.unique([ilab for ilab in labels if str(ilab).lower()!="none"]).tolist()
    assert len(good_label)<=1, "There can only be one good label, but for residue %u found %s"%(idx, good_label)
    try:
        return good_label[0]
    except IndexError:
        return no_key

'''
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
'''

'''
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
'''

'''
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
'''

def guess_nomenclature_fragments(refseq, top,
                                 fragments=None,
                                 min_hit_rate=.6,
                                 verbose=False,
                                 return_residue_idxs=False,
                                 empty=list):
    """Guess what fragments in the topology best match
    the consensus labels in a :obj:`LabelerConsensus` object

    The guess uses a cutoff for the quality of
    each segment's alignment to the sequence in :obj:`CLin`

    You can use the method to identify the receptor
    in topology where other molecules (e.g. the Gprot)
    are present (or the other way around)

    Parameters
    ----------
    refseq: str or :class:`LabelerConsensus`
        If not str, the sequence will
        be gotten from obj:`LabelerConsensus.seq` method

    top:
        :py:class:`~mdtraj.Topology` object
    fragments : iterable of iterables of idxs
        How :obj:`top` is split into fragments
        If None, will be generated using get_fragments defaults
    min_hit_rate: float, default is .6
        Only fragments with hit rates higher than this
        will be returned as a guess
    verbose: boolean
        be verbose
    return_residue_idxs : bool, default is False
        Return the list residue indices directly,
        instead of returning a list of fragment idxs.
    empty : class, list or None, default is list
        What to return in case of an emtpy guess,
        an empty list or a None
    Returns
    -------
    guess: list
        indices of the fragments (or residues) with higher hit-rate than :obj:`cutoff`


    """

    if fragments is None:
        fragments = _mdcfrg.get_fragments(top, verbose=False)

    try:
        seq_consensus = refseq.seq
    except AttributeError:
        assert isinstance(refseq,str),"refseq has to be either a %s with a 'seq' method or a string" \
                                      "but not a %s."%(LabelerConsensus,type(str))
        seq_consensus = refseq

    # TODO create a method out of this
    df = _mdcu.sequence.align_tops_or_seqs(top, seq_consensus)
    hit_idxs = df[df["match"]]["idx_0"].values
    hits, guess = [], []
    for ii, ifrag in enumerate(fragments):
        hit = _np.intersect1d(ifrag,hit_idxs)
        if len(hit) / len(ifrag) >= min_hit_rate:
            guess.append(ii)
        if verbose:
            print(ii, len(hit) / len(ifrag))
        hits.append(hit)

    guessed_res_idxs=[]
    if len(guess)>0:
        guessed_res_idxs = _np.hstack([fragments[ii] for ii in guess])

    if return_residue_idxs:
        guess = guessed_res_idxs

    if empty is None and len(guess)==0:
        guess = None
    return guess


def guess_by_nomenclature(CLin, top, fragments, nomenclature_name,
                          return_str=True, accept_guess=False,
                          **guess_kwargs):
    r"""
    Wrapper around :obj:`guess_nomenclature_fragments`to interpret
    its answer

    Parameters
    ----------
    CLin : :obj:`LabelerConsensus`
    top
    fragments
    nomenclature_name
    return_str
    accept_guess
    guess_kwargs

    Returns
    -------

    """
    guess = guess_nomenclature_fragments(CLin, top, fragments, **guess_kwargs)
    guess_as_string = ','.join(['%s' % ii for ii in guess])

    if len(guess) > 0:
        print("%s-labels align best with fragments: %s (first-last: %s-%s)."
              % (nomenclature_name, str(guess),
                 top.residue(fragments[guess[0]][0]),
                 top.residue(fragments[guess[-1]][-1])))
    else:
        print("No guessed fragment")
        return None

    if accept_guess:
        answer = guess_as_string
    else:
        answer = input("Input alternative in a format 1,2-6,10,20-25 or\nhit enter to accept the guess %s\n"%guess_as_string)

    if answer == '':
        answer = guess_as_string
    else:
        answer = ','.join(['%s' % ii for ii in _mdcu.lists.rangeexpand(answer)])

    if return_str:
        pass
    else:
        answer = [int(ii) for ii in answer.split(",")]
    return answer

def _map2defs(cons_list, splitchar="."):
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
    splitchar : str, default is "."
        The character to use to get the subdomain labels from the
        consensus labels, e.g. "3" from "3.50" or "G.H5" from "G.H5.1"
    Returns
    -------
    map : dictionary
        dictionary with subdomains as keys and lists of consensus labels as values
    """
    defs = _defdict(list)
    for ii, key in enumerate(cons_list):
        if str(key).lower()!= "none":
            assert splitchar in key, "Consensus keys have to have a '%s'-character" \
                                     " in them, but '%s' hasn't"%(splitchar, key)
            if key[0].isnumeric(): # it means it is BW
                new_key =key.split(splitchar)[0]
            elif key[0].isalpha(): # it means it CGN
                new_key = '.'.join(key.split(splitchar)[:-1])
            else:
                raise Exception([ii, splitchar])
            defs[new_key].append(ii)
    return {key: _np.array(val) for key, val in defs.items()}

def sort_consensus_labels(subset, sorted_superset,
                          append_diffset=True):
    r"""
    Sort consensus labels (BW or CGN)


    Parameters
    ----------
    subset : iterable
        list with the names (type str) to be ordered
    sorted_superset : iterable
        list with names in the desired order. Is a superset
        of :obj:`subset`
    append_diffset : bool, default is True
        Append the difference set subset-sorted_superset, i.e.
        all the elements in subset that are not in sorted_superset.

    Returns
    -------
    fragnames_out
    """

    by_frags = _defdict(dict)
    for item in subset:
        try:
            frag, idx = item.rsplit(".",maxsplit=1)
            by_frags[frag][idx] = item
        except:
            pass

    labs_out = []
    for frag in sorted_superset:
        if frag in by_frags.keys():
            for ifk in _natsorted(by_frags[frag].keys()):
                labs_out.append(by_frags[frag][ifk])

    if append_diffset:
        labs_out += [item for item in subset if item not in labs_out]

    return labs_out

def sort_BW_consensus_labels(labels, **kwargs):
    return sort_consensus_labels(labels, _GPCR_fragments, **kwargs)
def sort_CGN_consensus_labels(labels, **kwargs):
    return sort_consensus_labels(labels, _CGN_fragments, **kwargs)

def conslabel2fraglabel(labelres,defrag="@",prefix_GCPR=True):
    r"""
    Return a fragment label from a full consensus following some norms
    Parameters
    ----------
    labelres

    Returns
    -------
    labelfrag
    """

    label = labelres.split(defrag)[-1]
    label = label.rsplit(".",maxsplit=1)[0]
    if prefix_GCPR and str(label) in _GPCR_num2lett.keys():
        label = _GPCR_num2lett[label]
    return label

_GPCR_num2lett = {
    "1": "TM1 ",
    "12": "ICL1",
    "2": "TM2",
    "23": "ECL1",
    "3": "TM3",
    "34": "ICL2",
    "4": "TM4",
    "45": "ECL2",
    "5": "TM5",
    "56": "ICL3",
    "6": "TM6",
    "67": "ECL3",
    "7": "TM7",
    "8": "H8",
}

_GPCR_fragments=["NT",
                 "1", "TM1 ",
                 "12","ICL1",
                 "2", "TM2",
                 "23","ECL1",
                 "3", "TM3",
                 "34","ICL2",
                 "4", "TM4",
                 "45","ECL2",
                 "5", "TM5",
                 "56","ICL3",
                 "6", "TM6",
                 "67","ECL3",
                 "7", "TM7",
                 "78",
                 "8", "H8",
                 "CT"]

_CGN_fragments = ['G.HN',
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

# TODO this method is not used anywhere anymore, consider deleting
def compatible_consensus_fragments(top,
                                   existing_consensus_maps,
                                   CLs,
                                   autofill_consensus=True):
    r"""
    Expand (if possible) a list existing consensus maps
    using :obj:`LabelerConsensus` objects

    Note
    ----

    The origin of this plot is that :obj:`mdciao.cli.interface` needs
    all consensus labels it can get to prettify flareplots.

    #TODO this is no longer the case
    However, in the case of direct-selection by residue index (and possibly
    other cases), these consensus maps don't carry information about indices
    that were excluded when aligning the topology to a reference sequence
    (for better alignment).

    Hence, to be able to label these omitted residues, we
     * generate new consensus maps from all objects in :obj:`CLs`
     * aggregate them into one single map (if possible)
     * add them to :obj:`existing_consensus_maps`

    The last step will only work if the newly generated maps
    do not have differing definitions to those already in
    :obj:`existing` maps. Otherwise, an Exception is thrown.

    Parameters
    ----------
    top : :obj:`~mdtraj.Topology`
    existing_consensus_maps : list
        List of individual consensus maps, typically BW
        or CGN maps. These list are maps in this sense:
        cons_map[res_idx] = "3.50"
    CLs : list
        List of :obj:`mdciao.nomenclature.LabelerConsensus`-objects
        that will generate new consensus maps for all residues
        in :obj:`top`

    Returns
    -------
    new_frags : dict
        A new fragment definition, keyed with the consensus labels present
        in the input consensus labelers and compatible with the
        existing consensus labels
    """
    # If this doesn't work, nothing else will
    unified_existing_consensus_map = [choose_between_consensus_dicts(idx, existing_consensus_maps, no_key=None)
                                      for idx in range(top.n_residues)]

    # Same here
    new_maps = [iCL.top2labels(top, autofill_consensus=autofill_consensus, verbose=False) for iCL in CLs]
    unified_new_consensus_map = [choose_between_consensus_dicts(idx,new_maps,no_key=None) for idx in range(top.n_residues)]

    # Now incorporate new labels while checking with clashes with old ones
    for ii in range(top.n_residues):
        existing_val = unified_existing_consensus_map[ii]
        new_val = unified_new_consensus_map[ii]
        # take the new val, even if it's also None
        if existing_val is None:
            unified_existing_consensus_map[ii] = new_val
        # otherwise check no clashes with the existing map
        else:
            #print(existing_val, "ex not None")
            #print(new_val, "new val")
            assert existing_val==new_val, (new_val, existing_val, ii, top.residue(ii))

    new_frags = {}
    for iCL in CLs:
        new_frags.update(iCL.top2frags(top,
                                       #map_conlab=unified_new_consensus_map,
                                       verbose=False))

    # This should hold anyway bc of top2frags calling conlab2residx
    _mdcu.lists.assert_no_intersection(new_frags.values())

    return new_frags

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
from mdciao.utils.str_and_dict import _kwargs_subs

from pandas import \
    read_json as _read_json, \
    read_excel as _read_excel, \
    read_csv as _read_csv, \
    DataFrame as _DataFrame, \
    ExcelWriter as _ExcelWriter

from contextlib import contextmanager

from collections import defaultdict as _defdict, namedtuple as _namedtuple

from textwrap import wrap as _twrap

from mdciao.filenames import FileNames as _FN

from os import path as _path

import requests as _requests

from natsort import natsorted as _natsorted

from string import ascii_uppercase as _ascii_uppercase

from inspect import signature as _signature

_filenames = _FN()
_AA_chars_no_X = [char for char in _md.core.residue_names._AMINO_ACID_CODES.values() if char not in ["X", None]]


def _table2GPCR_by_AAcode(tablefile,
                          scheme="BW",
                          keep_AA_code=True,
                          return_fragments=False,
                          ):
    r"""
    Dictionary AAcodes so that e.g. AAcode2GPCR["R131"] -> '3.50' from a Excel file or an :obj:`pandas.DataFrame`

    Parameters
    ----------
    tablefile : xlsx file or pandas dataframe
        GPCR generic residue numbering in excel format
    scheme : str, default is "BW"
        The numbering scheme to choose. The available
        schemes depend on what was in the original
        :obj:`tablefile`. The options may include
        * "generic_display_number" (the one chosen by the GPCRdb)
        * "GPCRdb(A)", "GPCRdb(B)", ...
        * "BW"
        * "Wootten"
        * "Pin"
        * "Wang"
    keep_AA_code : boolean, default is True
        If True then output dictionary will have key of the form "Q26" else "26".
    return_fragments : boolean, default is True
        return a dictionary of fragments keyed by GPCR-fragment, e.g. "TM1"

    Returns
    -------
    AAcode2GPCR : dictionary
        Dictionary with residues as key and their corresponding GPCR notation.

    fragments : dict (optional)
        if return_fragments=True, a dictionary containing the fragments according to the excel file
    """

    if isinstance(tablefile, str):
        df = _read_excel(tablefile, header=0, engine="openpyxl")
    else:
        df = tablefile

    # TODO some overlap here with with _GPCR_web_lookup of GPCR_finder
    # figure out best practice to avoid code-repetition
    # This is the most important
    assert scheme in df.keys(), ValueError("'%s' isn't an available scheme.\nAvailable schemes are %s" % (
        scheme, [key for key in df.keys() if key in _GPCR_available_schemes + ["display_generic_number"]]))
    AAcode2GPCR = {key: str(val) for key, val in df[["AAresSeq", scheme]].values}
    # Locate definition lines and use their indices
    fragments = _defdict(list)
    for key, AArS in df[["protein_segment", "AAresSeq"]].values:
        fragments[key].append(AArS)
    fragments = {key: val for key, val in fragments.items()}

    if keep_AA_code:
        pass
    else:
        AAcode2GPCR = {int(key[1:]): val for key, val in AAcode2GPCR.items()}

    if return_fragments:
        return AAcode2GPCR, fragments
    else:
        return AAcode2GPCR


def _PDB_finder(PDB_code, local_path='.',
                try_web_lookup=True,
                verbose=True):
    r"""Return an :obj:`~mdtraj.Trajectory` by loading a local
    file or optionally looking up online, see :obj:`md_load_RCSB`

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
        using :obj:`md_load_RCSB`
    verbose : boolean, default is True
        Be verbose

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
                _geom, return_file = _md_load_rcsb(PDB_code,
                                                   verbose=verbose,
                                                   return_url=True)
                if verbose:
                    print("found! Continuing normally")

            else:
                raise

    return _geom, return_file


def _CGN_finder(uniprot_name,
                format='%s.txt',
                local_path='.',
                try_web_lookup=True,
                verbose=True,
                dont_fail=False,
                write_to_disk=False):
    r"""Provide a Uniprot name and look up (first locally, then online)
    for a file that contains the Common-Gprotein-Nomenclature (CGN)
    consensus labels and return them as a :obj:`~pandas.DataFrame`. See
    https://www.mrc-lmb.cam.ac.uk/CGN/ for more info on this nomenclature
    and `_finder_writer` for what's happening under the hood


    Parameters
    ----------
    uniprot_name : str
        Uniprot Name, e.g "GNAS2_HUMAN"
    format : str
        A format string that turns the `uniprot_name`
        into a filename for local lookup, in case the
        user has custom filenames, e.g. GNAS2_HUMAN_consensus.txt
    local_path : str
        The local path to the local consensus file
    try_web_lookup : bool, default is True
        If the local lookup fails, go online
    verbose : bool, default is True
        Be verbose
    dont_fail : bool, default is False
        Do not raise any errors that would interrupt
        a workflow and simply return None
    write_to_disk : bool, default is False
        Save the CGN data to disk

    Returns
    -------
    DF : :obj:`~pandas.DataFrame` with the consensus nomenclature

    References
    ----------
    * [2] Flock, T., Ravarani, C., Sun, D. et al.,
     * Universal allosteric mechanism for Gα activation by GPCRs.*
       Nature 524, 173–179 (2015)
      `<https://doi.org/10.1038/nature14663>`
    """
    file2read = format % uniprot_name
    file2read = _path.join(local_path, file2read)
    rep = lambda istr: [istr.replace(" ", "") if isinstance(istr, str) else istr][0]
    # using  delim_whitespace=True splits "Sort Number" in two keys, not feasible to generalize ATM
    local_lookup_lambda = lambda file2read: _read_csv(file2read, delimiter='\t').applymap(rep)

    web_address = "www.mrc-lmb.cam.ac.uk"
    url = "https://%s/CGN/lookup_results/%s.txt" % (web_address, uniprot_name)
    web_lookup_lambda = local_lookup_lambda

    print("Using CGN-nomenclature, please cite")
    lit = Literature()
    print(_format_cite(lit.scheme_CGN))
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
    web lambda and try to return a :obj:`DataFrame`
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
    return_name : str
        The URL or local path to
        the file that was used
    """
    try:
        return_name = full_local_path
        _DF = local2DF_lambda(full_local_path)
        print("%s found locally." % full_local_path)
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
                raise FileExistsError("Cannot overwrite existing file %s" % full_local_path)
            if _path.splitext(full_local_path)[-1] == ".xlsx":
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


def _GPCR_finder(GPCR_descriptor,
                 format="%s.xlsx",
                 local_path=".",
                 try_web_lookup=True,
                 verbose=True,
                 dont_fail=False,
                 write_to_disk=False):
    r"""
    Return a :obj:`~pandas.DataFrame` containing
    generic GPCR-numbering.

    The lookup is first local and then online
    on the `GPCRdb <https://gpcrdb.org/>`

    This method wraps (with some python lambdas) around
    :obj:`_finder_writer`.

    Please see the relevant references in :obj:`LabelerGPCR`.

    Parameters
    ----------
    GPCR_descriptor : str
        Anything that can be used to find the needed
        GPCR information, locally or online:
         * a uniprot descriptor, e.g. `adrb2_human`
         * a full local filename, e.g. `my_GPCR_consensus.txt` or
          `path/to/my_GPCR_consensus.txt`
         * the "basename" filename, `adrb2_human` if
          `adrb2_human.xlsx` exists on :obj:`local_path`
          (see below :obj:`format`)
        All these ways of doing the same thing (descriptor, basename, fullname,
        localpath, fullpath) are for compatibility with other methods.
    format : str, default is "%s.xlsx".
        If :obj:`GPCR_descriptor` is not readable directly,
        try to find "GPCR_descriptor.xlsx" locally on :obj:`local_path`
    local_path : str, default is "."
        If :obj:`GPCR_descriptor` doesn't find the file locally,
        then try "local_path/GPCR_descriptor" before trying online
    try_web_lookup : boolean, default is True.
        If local lookup variants fail, go online, else Fail
    verbose : bool, default is False
        Be verbose.
    dont_fail : bool, default is False
        If True, when the lookup fails None will
        be returned. By default the method raises
        an exception if it could not find the info.
    write_to_disk : boolean, default is False
        Save the found GPCR consensus nomenclature info locally.

    Returns
    -------
    df : DataFrame or None
        The GPCR consensus nomenclature information as :obj:`~pandas.DataFrame`
    """

    if _path.exists(GPCR_descriptor):
        fullpath = GPCR_descriptor
        try_web_lookup = False
    else:
        xlsxname = format % GPCR_descriptor
        fullpath = _path.join(local_path, xlsxname)
    GPCRmd = "https://gpcrdb.org/services/residues/extended"
    url = "%s/%s" % (GPCRmd, GPCR_descriptor.lower())

    local_lookup_lambda = lambda fullpath: _read_excel(fullpath,
                                                       engine="openpyxl",
                                                       usecols=lambda x: x.lower() != "unnamed: 0",
                                                       converters={key: str for key in _GPCR_available_schemes},
                                                       ).replace({_np.nan: None})
    web_looukup_lambda = lambda url: _GPCR_web_lookup(url, verbose=verbose)
    return _finder_writer(fullpath, local_lookup_lambda,
                          url, web_looukup_lambda,
                          try_web_lookup=try_web_lookup,
                          verbose=verbose,
                          dont_fail=dont_fail,
                          write_to_disk=write_to_disk)


def _GPCR_web_lookup(url, verbose=True,
                     timeout=5):
    r"""
    Lookup this url for a GPCR-notation
    return a ValueError if the lookup returns an empty json
    Parameters
    ----------
    url : str
    verbose : bool
    timeout : float, default is 1
        Timeout in seconds for :obj:`_requests.get`
        https://requests.readthedocs.io/en/master/user/quickstart/#timeouts
    Returns
    -------
    DF : :obj:`~pandas.DataFrame`
    """
    uniprot_name = url.split("/")[-1]
    a = _requests.get(url, timeout=timeout)

    return_fields = ["protein_segment",
                     "AAresSeq",
                     "display_generic_number"]
    pop_fields = ["sequence_number", "amino_acid", "alternative_generic_numbers"]
    # TODO use _url2json here
    if verbose:
        print("done!")
    if a.text == '[]':
        DFout = ValueError('Contacted %s url successfully (no 404),\n'
                           'but Uniprot name %s yields nothing' % (url, uniprot_name))
    else:
        df = _read_json(a.text)
        mydict = df.T.to_dict()
        for key, val in mydict.items():
            try:
                val["AAresSeq"] = '%s%s' % (val["amino_acid"], val["sequence_number"])
                if "alternative_generic_numbers" in val.keys():
                    for idict in val["alternative_generic_numbers"]:
                        # print(key, idict["scheme"], idict["label"])
                        val[idict["scheme"]] = idict["label"]
            except IndexError:
                pass

        DFout = _DataFrame.from_dict(mydict, orient="index").replace({_np.nan: None})
        return_fields += [key for key in DFout.keys() if key not in return_fields + pop_fields]
        DFout = DFout[return_fields]
        print("Please cite the following reference to the GPCRdb:")
        lit = Literature()
        print(_format_cite(lit.site_GPCRdb))
        print("For more information, call mdciao.nomenclature.references()")

    return DFout


def _md_load_rcsb(PDB,
                  web_address="https://files.rcsb.org/download",
                  verbose=False,
                  return_url=False):
    r"""
    Input a PDB code get an :obj:`~mdtraj.Trajectory` object.

    Thinly wraps around :obj:`~mdtraj.load_pdb` by constructing
    the url for the user.

    The difference with mdciao.pdb.pdb2traj is that pdb2traj
    actually downloads the full PDB file with annotations etc,
    which would be different from simply doing traj.save_pdb

    Parameters
    ----------
    PDB : str
        4-letter PDB code
    web_address: str, default is "https://files.rcsb.org/download"
        The web address of the RCSB PDB database
    verbose : bool, default is False
        Be verbose
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

    At the moment child classes are
     * :obj:`LabelerGPCR` for GPCR-notation, this can be:
       * structure based schemes (Gloriam et al)
       * sequence based schemes
         * Class-A: Ballesteros-Weinstein
         * Class-B: Wootten
         * Class-C: Pin
         * Class-F: Wang
     * :obj:`LabelerCGN` for Common-Gprotein-nomenclature (CGN)
     * :obj:`LabelerKLIFS` for Kinase-Ligand Interaction notation of the 85 pocket-residues of kinases

    The consensus labels are abbreviated to 'conlab' throughout

    """

    def __init__(self, **PDB_finder_kwargs):
        r"""

        Parameters
        ----------
        ref_PDB : str
            4-letter PDB code
        tablefile: str, default is 'GPCRmd_B2AR_nomenclature'
            The PDB four letter code that will be used for CGN purposes
        ref_path: str,default is '.'
            The local path where the needed files are

        try_web_lookup: bool, default is True
            If the local files are not found, try automatically a web lookup at
             * www.mrc-lmb.cam.ac.uk (for CGN)

        """
        self._conlab2AA = {val: key for key, val in self.AA2conlab.items()}

        self._fragment_names = list(self.fragments.keys())
        self._fragments_as_conlabs = {key: [self.AA2conlab[AA] for AA in val]
                                      for key, val in self.fragments.items()}
        self._fragments_as_resSeqs = {key : [_mdcu.residue_and_atom.int_from_AA_code(ival) for ival in val]
                                      for key, val in self.fragments.items()}

        self._idx2conlab = self.dataframe[self._nomenclature_key].values.tolist()
        self._conlab2idx = {lab: idx for idx, lab in enumerate(self.idx2conlab) if lab is not None}

    @property
    def seq(self):
        r""" The reference sequence in :obj:`dataframe`"""
        return ''.join(
            [_mdcu.residue_and_atom.name_from_AA(val) for val in self.dataframe[self._AAresSeq_key].values.squeeze()])

    @property
    def conlab2AA(self):
        r""" Dictionary with consensus labels as keys, so that e.g.
            * self.conlab2AA["3.50"] -> 'R131' or
            * self.conlab2AA["G.hfs2.2"] -> 'R201' """
        return self._conlab2AA

    @property
    def conlab2idx(self):
        r""" Dictionary with consensus labels as keys and zero-indexed row-indices of self.dataframe, as values so that e.g.
                   * self.conlab2AA["3.50"] -> 'R131' or
                   * self.conlab2AA["G.hfs2.2"] -> 'R201' """

        return self._conlab2idx

    @property
    def AA2conlab(self):
        r""" Dictionary with short AA-codes as keys, so that e.g.
            * self.AA2conlab["R131"] -> '3.50'
            * self.AA2conlab["R201"] -> "G.hfs2.2" """

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
    def fragments_as_resSeqs(self) -> dict:
        r""" Dictionary of fragments keyed with fragment names
        and valued with the residue sequence indices (resSeq) in that fragment

        Returns
        -------
        fragments_as_resSeqs : dict
        """
        return self._fragments_as_resSeqs

    @property
    def dataframe(self) -> _DataFrame:
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

    def aligntop(self, top,
                 restrict_to_residxs=None,
                 min_hit_rate=.5,
                 fragments='resSeq',
                 verbose=False):
        r""" Align a topology with the object's sequence.
        Returns two maps (`top2self`, `self2top`) and
        populates the attribute self.most_recent_alignment

        Wraps around :obj:`mdciao.utils.sequence.align_tops_or_seqs`

        The indices of self are indices (row-indices)
        of the original :obj:`~nomenclature.mdciao.LabelerConsensus.dataframe`,
        which are the ones in :obj:`~mdciao.nomenclature.LabelerConsensus.seq`

        Parameters
        ----------
        top : :obj:`~mdtraj.Topology` object or string
        restrict_to_residxs : iterable of integers, default is None
            Use only these residues for alignment and labelling purposes.
            Helps "guide" the alignment method. E.g., for big topologies
            the the alignment might find some small matches somewhere
            and, in some corner cases, match those instead of the
            desired ones. Here, one can pass residues indices
            defining the topology segment wherein the match should
            be contained to.
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
        fragments : str, iterable, None, or bool, default is 'resSeq'
            Fragment definitions to resolve situations where
            two (or more) alignments share the optimal alignment score.
            Consider aligning an input sequence 'XXLXX' to the object's
            sequence 'XXLLXX'. There are two equally scored alignments::

                XXL XX      XX LXX
                ||| ||  vs  || |||
                XXLLXX      XXLLXX

            In order to choose between these two alignments, it's
            checked which alignment observes the
            fragment definition passed here. This definition
            can be passed explicitly as iterable of integers
            or implicitly as a fragmentation heuristic, which
            will be used by :obj:`mdciao.fragments.get_fragments`
            on the `top`. So, if e.g. the input 'XXLXX' sequence
            is fragmented (explicitly or implicitly)
            into [XX],[LXX], then the second alignment
            will be chosen, given that it respects that fragmentation.

            Note
            ----
            `fragments` only has an effect if both
             * the `top` is an actual :obj:`~mdtraj.Topology` carrying the sequence
              indices, since if `top` is a sequence
              string, then there's no fragmentation heuristic possible.
             * two or more alignments share the optimal alignment score
            The method avoids breaking the consensus definitions
            across the input `fragments`, while also providing
            consensus definitions for those other residues not
            present in `fragments`. This is done by using 'resSeq' to infer
            the missing fragmentation. This keeps the functionality of
            respecting the original `fragments` while also providing
            consensus fragmentation other parts of the topology.
            For compatibility with other methods,
            passing `fragments=None` will still use the fragmentation
            heuristic (this might change in the future).
            **To explicitly circumvent this forced fragmentation
            and subsequent check, use `fragments=False`.
            This will simply use the first alignment that comes out of
            :obj:`mdciao.utils.sequence.my_bioalign`, regardless
            of there being other, equally scored, alignments and potential
            clashes with sensitive fragmentations.**
        verbose: boolean, default is False
            be verbose
        Returns
        ------
        top2self : dict
            Maps indices of `top` to indices
            of this objects self.seq
        self2top : dict
            Maps indices of this object's seq.seq
            to indices of this self.seq
        """
        debug = False
        #debug = True
        n_residues = [len(top) if isinstance(top, str) else top.n_residues][0]
        # Define fragments even if it turns out we will not need them
        # The code is easier to follow this way
        if isinstance(fragments, str):
            _frag_str = fragments
            if isinstance(top, _md.Topology):
                _fragments = _mdcfrg.fragments.get_fragments(top, _frag_str, verbose=False)
            else:
                _fragments = [_np.arange(n_residues)]
        elif fragments is None:
            _frag_str = "resSeq"
            if isinstance(top, _md.Topology):
                _fragments = _mdcfrg.fragments.get_fragments(top, _frag_str, verbose=False)
            else:
                _fragments = [_np.arange(n_residues)]
        elif fragments is False:
            _frag_str = "False"
            _fragments = [_np.arange(n_residues)]
        elif fragments is not None:
            _frag_str = "explict definition"
            _fragments = _mdcfrg.fragments.get_fragments(top, "resSeq", verbose=False)
            _fragments = _mdcfrg.mix_fragments(n_residues - 1,
                                               {"input %u" % ii: fr for ii, fr in enumerate(fragments)},
                                               _fragments, None)[0]

        if (min_hit_rate > 0):
            assert restrict_to_residxs is None
            restrict_to_residxs = guess_nomenclature_fragments(self.seq,
                                                               top,
                                                               _fragments,
                                                               verbose=verbose or debug,
                                                               min_hit_rate=min_hit_rate,
                                                               return_residue_idxs=True, empty=None)

        # In principle I'm introducing this only for KLIFS, could be for all nomenclatures
        if self._nomenclature_key == "KLIFS":
            chain_id = self.dataframe.chain_index[_np.hstack(list(self.fragments_as_idxs.values()))].unique()
            assert len(chain_id) == 1
            seq_1_res_idxs = self.dataframe[self.dataframe.chain_index == chain_id[0]].index
        else:
            seq_1_res_idxs = None
        df = _mdcu.sequence.align_tops_or_seqs(top,
                                               self.seq,
                                               seq_0_res_idxs=restrict_to_residxs,
                                               seq_1_res_idxs=seq_1_res_idxs,
                                               return_DF=True,
                                               verbose=verbose,
                                               )
        consfrags = []
        # For clever alternatives https://stackoverflow.com/a/3844832
        for ii, idf in enumerate(df):
            top2self, self2top = _mdcu.sequence.df2maps(idf)
            consfrags.append(self._selfmap2frags(self2top))
        all_fragments_equal = all(
            [cf == consfrags[0] for cf in consfrags[:1]])  # vacously True https://stackoverflow.com/a/19602868

        frags_already_printed = False
        for ii, idf in enumerate(df):
            top2self, self2top = _mdcu.sequence.df2maps(idf)
            alignment_column2conlab = _np.full(len(idf), None)
            alignment_column2conlab[_np.flatnonzero(idf["idx_0"].isin(top2self.keys()))] = self.dataframe.iloc[list(top2self.values())][
                self._nomenclature_key]
            topidx2conlab = {row.idx_0 : alignment_column2conlab[row_index] for row_index, row in idf[idf.match].iterrows()}
            if isinstance(top, str) or str(_frag_str).lower() in ["none", "false"] or len(df) == 1:
                confrags_compatible_with_frags = True
                if debug:
                    print("I'm not checking fragment compatibility because at least one of these statements is True")
                    print(
                        " * the input topology is a sequence string, s.t. no fragments can be extracted: %s" % isinstance(
                            top, str))
                    print(" * There is only one pairwise alignment with the maximum score: ", len(df) == 1, len(df))
                    print(" * The fragmentation heuristics were False or None: ",
                          str(_frag_str).lower() in ["none", "false"], _frag_str)
                    #      _frag_str, len(df))
                break
            else:
                fragments = _fragments  # This will have been defined already
                confrags_compatible_with_frags = True
                consfrags = self._selfmap2frags(self2top)
                if debug:
                    print("Iteration ", ii)
                    _mdcfrg.print_fragments(consfrags, top)
                for frag_idx, (confraglab, confragidxs) in enumerate(consfrags.items()):
                    confrag_is_subfragment = _mdcfrg.check_if_subfragment(confragidxs, confraglab, fragments, top,
                                                                          map_conlab=topidx2conlab,
                                                                          prompt=False)
                    if debug:
                        print(ii, confraglab, confrag_is_subfragment)
                    if not confrag_is_subfragment:
                        if not frags_already_printed:
                            print("fragments derived from '%s':"%_frag_str)
                            _mdcfrg.print_fragments(_fragments, top)
                            frags_already_printed = True
                        print(
                            "The consensus-sequence alignment nr. %u  (score = %u, %u other alignments also have this score),\n"
                            "defines the consensus fragment '%s' having clashes with the fragment definitions derived from '%s':" % (
                                ii, idf.alignment_score , len(df)-1, confraglab, _frag_str))
                        _mdcfrg.print_frag(frag_idx, top, confragidxs, resSeq_jumps=True, idx2label=topidx2conlab,
                                           fragment_desc="%s" % confraglab)
                        confragAAs = [_mdcu.residue_and_atom.shorten_AA(top.residue(idx), keep_index=True) for idx in
                                      confragidxs]

                        if not set(self.fragments[confraglab]).issuperset(confragAAs):
                            confrags_compatible_with_frags = False
                            break
                        else:
                            print("However all residues assigned to '%s' in the `top` are contained in reference consensus\n"
                                  "fragment (in name and index), so this alignment is considered compatible."%(confraglab))

                if confrags_compatible_with_frags:
                    if ii>0:
                        print("Picking alignment nr. %u with no aparent breaks."%ii)
                    break
        assert confrags_compatible_with_frags, ("None of the %u best pairwise alignments yield consensus fragments "
                                                "compatible with the '%s' fragmentation-heuristic, which yields\n%s\n"
                                                "Try increasing the `min_hit_rate` "
                                                "or using `restrict_to_residxs` to restrict the alignment only to\nthose "
                                                "residues of top` most likely to belong to this object's sequence as stored in `self.seq`." % (
                                                len(df), str(_frag_str),
                                                "\n".join(_mdcfrg.print_fragments(fragments, top))))

        df = idf

        df = df.join(_DataFrame({"conlab": alignment_column2conlab}))

        self._last_alignment_df = df

        return top2self, self2top

    @_kwargs_subs(aligntop)
    def top2labels(self, top,
                   allow_nonmatch=True,
                   autofill_consensus=True,
                   min_hit_rate=.5,
                   **aligntop_kwargs) -> list:

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

        To remedy the second case a-posteriori two things
        can be done:
         * recover the original label even though residues
           did not match, using :obj:`allow_nonmatch`.
           See :obj:`alignment_df2_conslist` for more info
         * reconstruct what the label could be using a heuristic
           to "autofill" the consensus labels, using
           :obj:`autofill_consensus`.
           See :obj:`_fill_consensus_gaps` for more info.

        Note
        ----
        This method uses :obj:`aligntop` internally,
        see the doc on that method for more info.

        Parameters
        ----------
        top :
            :obj:`~mdtraj.Topology` object
        allow_nonmatch : bool, default is True
            Use consensus labels for non-matching positions
            in case the non-matches have equal lengths
        autofill_consensus : boolean default is False
            Even if there is a consensus mismatch with the sequence of the input
            :obj:`AA2conlab_dict`, try to relabel automagically, s.t.
             * ['G.H5.25', 'G.H5.26', None, 'G.H.28']
            will be relabeled as
             * ['G.H5.25', 'G.H5.26', 'G.H.27', 'G.H.28']
        min_hit_rate : float, default is .5
            With big topologies and many fragments,
            the alignment method (:obj:`mdciao.sequence.my_bioalign`)
            sometimes yields sub-optimal results. A value
            :obj:`min_hit_rate` >0, e.g. .5 means that a pre-alignment
            takes place to populate :obj:`restrict_to_residxs`
            with indices of those the fragments
            (:obj:`mdciao.fragments.get_fragments` defaults)
            with more than 50%% alignment in the pre-alignment.
        aligntop_kwargs : dict
            Optional parameters for :obj:`~mdciao.nomenclature.LabelerConsensus.aligntop`,
            which are listed below

        Other Parameters
        ----------------
        %(substitute_kwargs)s

        Returns
        -------
        map : list
            List of len = top.n_residues with the consensus labels
        """
        self.aligntop(top, min_hit_rate=min_hit_rate, **aligntop_kwargs)
        out_list = _alignment_df2_conslist(self.most_recent_alignment, allow_nonmatch=allow_nonmatch)
        out_list = out_list + [None for __ in range(top.n_residues - len(out_list))]
        # TODO we could do this padding in the alignment_df2_conslist method itself
        # with an n_residues optarg, IDK about best design choice
        if autofill_consensus:
            out_list = _fill_consensus_gaps(out_list, top, verbose=False)

        self._last_top2labels = out_list
        return out_list

    @_kwargs_subs(top2labels)
    def conlab2residx(self, top,
                      map=None,
                      **top2labels_kwargs,
                      ):
        r"""
        Returns a dictionary keyed by consensus labels and valued
        by residue indices of the input topology in `top`.

        The default behaviour is to internally align `top`
        with the object's available consensus dictionary
        on the fly using :obj:`~mdciao.nomenclature.LabelerConsensus.top2labels`.
        See the docs there for **top2labels_kwargs, in particular
        `restrict_to_residxs`, `keep_consensus`, and `min_hit_rate

        Note
        ----
        This method is able to work with a new topology every
        time, performing a sequence alignment every call.
        The intention is to instantiate a
        :obj:`LabelerConsensus` just one time and use it with as
        many topologies as you like without changing any attribute
        of :obj:`self`.

        HOWEVER, if you know what you are doing, you can provide a
        list of consensus labels yourself using `map`. Then,
        this method is nothing but a table lookup (almost)

        Warning
        -------
        No checks are performed to see if the input of `map`
        actually matches the residues of `top` in any way,
        so that the output can be rubbish and go unnoticed.

        Parameters
        ----------
        top : :obj:`~mdtraj.Topology`
        map : list, default is None
            A pre-computed residx2consensuslabel map, i.e. the
            output of a previous, external call to :obj:`_top2consensus_map`
            If it contains duplicates, it is a malformed list.
            See the note above for more info
        top2labels_kwargs : dict
            Optional parameters for
            :obj:`~mdciao.nomenclature.LabelerConsensus.top2labels`,
            which are listed below

        Other Parameters
        ----------------
        %(substitute_kwargs)s

        Returns
        -------
        dict : keyed by consensus labels and valued with residue idxs
        """
        if map is None:
            map = self.top2labels(top,
                                  **top2labels_kwargs,
                                  )
        out_dict = {}
        for ii, imap in enumerate(map):
            if imap is not None and str(imap).lower() != "none":
                if imap in out_dict.keys():
                    raise ValueError("Entries %u and %u of the map, "
                                     "i.e. residues %s and %s of the input topology "
                                     "both have the same label %s.\n"
                                     "This method cannot work with a map like this!" % (out_dict[imap], ii,
                                                                                        top.residue(out_dict[imap]),
                                                                                        top.residue(ii),
                                                                                        imap))
                else:
                    out_dict[imap] = ii
        return out_dict

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
            :py:class:`~mdtraj.Topology` or path to topology file (e.g. a pdb)
        fragments: iterable of integers, default is None
            The user can parse an existing list of fragment-definitions
            (via residue idxs) to check if the newly found, consensus
            definitions (`defs`) clash with the input in `fragments`.
            *Clash* means that the `defs` would span over more
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

        if isinstance(top, str):
            top = _md.load(top).top

        if input_dataframe is None:
            top2self, self2top = self.aligntop(top, min_hit_rate=min_hit_rate, verbose=show_alignment,
                                               fragments=fragments)
        else:
            top2self, self2top = _mdcu.sequence.df2maps(input_dataframe)

        # TODO "topmaps" are more straighfdw than dataframes
        # but in pple we could make a lot of the next bookkeeping as df2xxx functions

        defs = self._selfmap2frags(self2top)

        new_defs = {}
        map_conlab = [self.idx2conlab[top2self[topidx]] if topidx in top2self.keys() else None for topidx in
                      range(top.n_residues)]

        for ii, (key, res_idxs) in enumerate(defs.items()):
            if fragments is not None:
                new_defs[key] = _mdcfrg.check_if_subfragment(res_idxs, key, fragments, top, map_conlab=map_conlab)

        for key, res_idxs in new_defs.items():
            defs[key] = res_idxs

        for ii, (key, res_idxs) in enumerate(defs.items()):
            istr = _mdcfrg.print_frag(key, top, res_idxs, fragment_desc='',
                                      idx2label=map_conlab,
                                      return_string=True)
            if verbose:
                print(istr)

        return dict(defs)

    def _selfmap2frags(self, self2top):
        r""" Take a self2top-mapping (coming from self.aligntop) and turn it into consensus fragment definitions """
        defs = {key: [self2top[idx] for idx in val if idx in self2top.keys()] for key, val in
                self.fragments_as_idxs.items()}
        defs = {key: val for key, val in defs.items() if len(val) > 0}
        return defs

    @property
    def most_recent_alignment(self) -> _DataFrame:
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
    Obtain and manipulate common-Gprotein-nomenclature.
    See https://www.mrc-lmb.cam.ac.uk/CGN/faq.html for more info.
    """

    def __init__(self, uniprot_name,
                 local_path='.',
                 format="%s.txt",
                 try_web_lookup=True,
                 verbose=True,
                 write_to_disk=None):
        r"""

        Parameters
        ----------
        uniprot_name: str
            UniProt name, e.g. 'GNAS2_HUMAN'. Please note the difference between UniProt Accession Code
            and UniProt entry name as explained `here <https://www.uniprot.org/help/difference%5Faccession%5Fentryname>`_.
            For compatibility reasons, there's different use-cases.

            * Full path to an existing file containing the CGN nomenclature,
            e.g. '/abs/path/to/some/dir/GNAS2_HUMAN.txt' (or GNAS2_HUMAN.xlsx). Then this happens:
                * :obj:`local_path` gets overridden with '/abs/path/to/some/dir/'
                * if none of these files can be found and :obj:`try_web_lookup` is True, then
                  'GNAS2_HUMAN' is looked up online in the CGN database

            * Uniprot Accession Code, e.g. 'GNAS2_HUMAN'. Then this happens:
                * look for the files 'GNAS2_HUMAN.txt' or 'GNAS2_HUMAN.xlsx' in `local_path`
                * if none of these files can be found and :obj:`try_web_lookup` is True, then
                  'GNAS2_HUMAN' is looked up online in the CGN database

        Note
        ----
            The intention behind this flexibility (which is hard to document and
            maintain) is to keep the signature of consensus labelers somewhat
            consistent for compatibility with other command line methods

        local_path: str, default is '.'
            The local path where these files exist, if they exist
        format : str, default is "%s.txt"
            How to construct a filename out of
            `uniprot_name`
        try_web_lookup: bool, default is True
            If the local files are not found, try automatically a web lookup at
            * www.mrc-lmb.cam.ac.uk
        """

        self._nomenclature_key = "CGN"

        # TODO see fragment_overview...are there clashes
        if _path.exists(uniprot_name):
            local_path, basename = _path.split(uniprot_name)
            uniprot_name = _path.splitext(basename)[0]#.replace("CGN_", "")
            # TODO does the check need to have the .txt extension?
            # TODO do we even need this check?
        self._dataframe, self._tablefile = _CGN_finder(uniprot_name,
                                                       local_path=local_path,
                                                       try_web_lookup=try_web_lookup,
                                                       verbose=verbose,
                                                       write_to_disk=write_to_disk,
                                                       format=format)
        # The title of the column with this field varies between CGN and GPCR
        AAresSeq_key = [key for key in list(self.dataframe.keys()) if
                        key.lower() not in [self._nomenclature_key.lower(), "Sort number".lower()]]
        assert len(AAresSeq_key) == 1
        self._AAresSeq_key = AAresSeq_key

        self._AA2conlab = {key: self._dataframe[self._dataframe[uniprot_name] == key][self._nomenclature_key].to_list()[0]
                           for key in self._dataframe[uniprot_name].to_list()}

        self._fragments = _defdict(list)
        for ires, key in self.AA2conlab.items():
            try:
                new_key = '.'.join(key.split(".")[:-1])
            except:
                print(key)
            # print(key,new_key)
            self._fragments[new_key].append(ires)
        LabelerConsensus.__init__(self,
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
        assert len(_np.unique(AAresSeq_list)) == len(
            AAresSeq_list), "Redundant residue names in the dataframe? Somethings wrong"
        AAresSeq2idx = {key: idx for idx, key in enumerate(AAresSeq_list)}
        defs = {key: [AAresSeq2idx[AAresSeq] for AAresSeq in val] for key, val in self.fragments.items()}
        return defs


class LabelerGPCR(LabelerConsensus):
    """Obtain and manipulate GPCR notation.

    This is based on the awesome GPCRdb REST-API,
    and follows the different schemes provided
    there. These schemes can be:

    * structure based schemes, available for all GPCR classes.
      You can use them by instantiating with "GPCRdb(A)", "GPCRdb(B)" etc.
      The relevant references are:
     * Isberg et al, (2015) Generic GPCR residue numbers - Aligning topology maps while minding the gaps,
       Trends in Pharmacological Sciences 36, 22--31,
       https://doi.org/10.1016/j.tips.2014.11.001
     * Isberg et al, (2016) GPCRdb: An information system for G protein-coupled receptors,
       Nucleic Acids Research 44, D356--D364,
       https://doi.org/10.1093/nar/gkv1178

    * sequence based schemes, with different names depending on the GPCR class
     * Class-A:
       Ballesteros et al, (1995) Integrated methods for the construction of three-dimensional models
       and computational probing of structure-function relations in G protein-coupled receptors,
       Methods in Neurosciences 25, 366--428,
       https://doi.org/10.1016/S1043-9471(05)80049-7
     * Class-B:
       Wootten et al, (2013) Polar transmembrane interactions drive formation of ligand-specific
       and signal pathway-biased family B G protein-coupled receptor conformations,
       Proceedings of the National Academy of Sciences of the United States of America 110, 5211--5216,
       https://doi.org/10.1073/pnas.1221585110
     * Class-C:
       Pin et al, (2003) Evolution, structure, and activation mechanism of family 3/C G-protein-coupled receptors
       Pharmacology and Therapeutics 98, 325--354
       https://doi.org/10.1016/S0163-7258(03)00038-X
     * Class-F:
       Wu et al, (2014) Structure of a class C GPCR metabotropic glutamate receptor 1 bound to an allosteric modulator
       Science 344, 58--64
       https://doi.org/10.1126/science.1249489

    Note
    ----
    Not all schemes might work work for all methods
    of this class. In particular, fragmentation
    heuristics are derived from 3.50(x50)-type
    formats. Other class A schemes
    like Oliveira and Baldwin-Schwarz
    can be used for residue mapping, labeling,
    but not for fragmentation. They are still
    partially usable but we have decided
    to omit them from the docs. Please
    see the full reference page for their citation.

    """

    def __init__(self, uniprot_name,
                 GPCR_scheme="display_generic_number",
                 local_path=".",
                 format="%s.xlsx",
                 verbose=True,
                 try_web_lookup=True,
                 # todo write to disk should be moved to the superclass at some point
                 write_to_disk=False):
        r"""

        Parameters
        ----------
        uniprot_name : str
            Descriptor by which to find the nomenclature,
            it gets directly passed to :obj:`GPCR_finder`
            Can be anything that can be used to try and find
            the needed information, locally or online:
            * a uniprot descriptor, e.g. `adrb2_human`
            * a full local filename
            * a part of a local filename
            Please note the difference between UniProt Accession Code
            and UniProt entry name as explained `here <https://www.uniprot.org/help/difference%5Faccession%5Fentryname>`_.
        GPCR_scheme : str, default is 'display_generic_number'
            The GPCR nomenclature scheme to use.
            The default is to use what the GPCRdb
            itself has chosen for this particular
            uniprot code. Not all schemes will be
            available for all choices of
            :obj:`uniprot_name`. You can
            choose from: 'BW', 'Wootten', 'Pin',
            'Wang', 'Fungal', 'GPCRdb(A)', 'GPCRdb(B)',
            'GPCRdb(C)', 'GPCRdb(F)', 'GPCRdb(D)',
            'Oliveira', 'BS', but not all are guaranteed
            to work
        local_path : str, default is "."
            Since the :obj:`uniprot_name` is turned into
            a filename in case it's a descriptor,
            this is the local path where to (potentially) look for files.
            In case :obj:`uniprot_name` is just a filename,
            we can turn it into a full path to
            a local file using this parameter, which
            is passed to :obj:`GPCR_finder`
            and :obj:`LabelerConsensus`. Note that this
            optional parameter is here for compatibility
            reasons with other methods and might disappear
            in the future.
        format : str, default is "%s.xlsx"
            How to construct a filename out of
            :obj:`uniprot_name`
        verbose : bool, default is True
            Be verbose. Gets passed to :obj:`GPCR_finder`
            :obj:`LabelerConsensus`
        try_web_lookup : bool, default is True
            Try a web lookup on the GPCRdb of the :obj:`uniprot_name`.
            If :obj:`uniprot_name` is e.g. "adrb2_human.xlsx",
            including the extension "xslx", then the lookup will
            fail. This what the :obj:`format` parameter is for
        write_to_disk : bool, default is False
            Save an excel file with the nomenclature
            information
        """

        self._nomenclature_key = GPCR_scheme
        # TODO now that the finder call is the same we could
        # avoid cde repetition here
        self._dataframe, self._tablefile = _GPCR_finder(uniprot_name,
                                                        format=format,
                                                        local_path=local_path,
                                                        try_web_lookup=try_web_lookup,
                                                        verbose=verbose,
                                                        write_to_disk=write_to_disk
                                                        )
        # The title of the column with this field varies between CGN and GPCR
        self._AAresSeq_key = "AAresSeq"
        self._AA2conlab, self._fragments = _table2GPCR_by_AAcode(self.dataframe, scheme=self._nomenclature_key,
                                                                 return_fragments=True)
        # TODO can we do this using super?
        LabelerConsensus.__init__(self,
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


class AlignerConsensus(object):
    """Use consensus labels for multiple sequence alignment.

    Instead of doing an *actual* multiple sequence alignment,
    we can exploit the existing consensus labels to align residues
    across very different (=low sequence identity) sequences and,
    optionally, topologies.

    Without topologies, the alignment via the consensus labels is limited
    to the reference UniProt residue sequence indices:

    >>> import mdciao
    >>> # Get the consensus labels from the GPCRdb and store them in a dict
    >>> maps = { "OPS": mdciao.nomenclature.LabelerGPCR("opsd_bovin"),
    >>>         "B2AR": mdciao.nomenclature.LabelerGPCR("adrb2_human"),
    >>>         "MUOR": mdciao.nomenclature.LabelerGPCR("oprm_mouse")}
    >>> # Pass the maps to AlignerConsensus
    >>> AC = mdciao.nomenclature.AlignerConsensus(maps)
    >>> AC.AAresSeq
        consensus   OPS  B2AR  MUOR
    0     1.25x25   NaN   Q26   NaN
    1     1.26x26   NaN   E27   NaN
    2     1.27x27   NaN   R28   NaN
    3     1.28x28   E33   D29   NaN
    4     1.29x29   P34   E30   M65
    ..        ...   ...   ...   ...
    285   8.55x55  V318  Q337  R348
    286   8.56x56  T319  E338  E349
    287   8.57x57  T320  L339  F350
    288   8.58x58  L321  L340  C351
    289   8.59x59  C322  C341  I352

    You can also filter the aligment using the `_match` methods

    >>> AC.AAresSeq_match("3.5*")
        consensus   OPS  B2AR  MUOR
    117   3.50x50  R131  R135  R165
    118   3.51x51  Y132  Y136  Y166
    119   3.52x52  F133  V137  I167
    120   3.53x53  A134  V138  A168
    ..        ...   ...   ...   ...

    With topologies, e.g. coming from specific pdbs (or
    from your own MD-setups), the alignment can be expressed
    in terms of residue indices or Cɑ-atom indices of those
    specific topologies. In this example, we are loading
    directly from the PDB, but you could load your own files
    with your own setups:

    >>> pdb3CAP = mdciao.cli.pdb("3CAP")
    >>> pdb3SN6 = mdciao.cli.pdb("3SN6")
    >>> pdbMUOR = mdciao.cli.pdb("6DDF")
    >>> AC = mdciao.nomenclature.AlignerConsensus(maps,
    >>>                                           tops={ "OPS": pdb3CAP.top,
    >>>                                                 "B2AR": pdb3SN6.top,
    >>>                                                 "MUOR": pdbMUOR.top})

    Zero-indexed, per-topology Cɑ atom indices:

    >>> AC.CAidxs_match("3.5*")
        consensus   OPS  B2AR  MUOR
    114   3.50x50  1065  7835  5370
    115   3.51x51  1076  7846  5381
    116   3.52x52  1088  7858  5393
    117   3.53x53  1095  7869  5401
    [...]

    Zero-indexed, per-topology residue serial indices:

    >>> AC.residxs_match("3.5*")
        consensus  OPS  B2AR  MUOR
    114   3.50x50  134  1007   706
    115   3.51x51  135  1008   707
    116   3.52x52  136  1009   708
    117   3.53x53  137  1010   709
    ..        ...   ...   ...   ...

    By default, the `_match` methods return only rows were all
    present systems ("OPS","B2AR", "MUOR" in the example)
    have consensus labels. E.g. here we ask for TM5 residues
    present in all three systems:

    >>> AC.AAresSeq_match("5.*")
        consensus   OPS  B2AR  MUOR
    167   5.35x36  N200  N196  E229
    168   5.36x37  E201  Q197  N230
    169   5.37x38  S202  A198  L231
    ..        ...   ...   ...   ...
    198   5.66x66  K231  K227  K260
    199   5.67x67  E232  R228  S261
    200   5.68x68  A233  Q229  V262

    But you can get relax the match and get an overview of missing
    residues using `omit_missing=False`:

    >>> AC.AAresSeq_match("5.*", omit_missing=False)
        consensus   OPS  B2AR  MUOR
    162   5.30x31   NaN   NaN  P224
    163   5.31x32   NaN   NaN  T225
    164   5.32x33   NaN   NaN  W226
    165   5.33x34   NaN   NaN  Y227
    166   5.34x35  N199   NaN  W228
    167   5.35x36  N200  N196  E229
    ..        ...   ...   ...   ...
    200   5.68x68  A233  Q229  V262
    201   5.69x69  A234  L230   NaN
    202   5.70x70  A235  Q231   NaN
    203   5.71x71  Q236  K232   NaN
    204   5.72x72  Q237  I233   NaN
    205   5.73x73   NaN  D234   NaN
    206   5.74x74   NaN  K235   NaN
    207   5.75x75   NaN  S236   NaN
    208   5.76x76   NaN  E237   NaN

    Here, we see e.g. that "MUOR" has more residues present at
    the beginning of TM5 (first row, from P224@5.30x31 on) and also that
    e.g. "B2AR" has the longest TM5 (last row, until E237@5.76x76).

    Finally, instead of selecting for labels,
    you can also select for systems, i.e. "Show me the systems that
    have my selection labels". Here, we ask what systems have '5.70...5.79' residues:

    >>> AC.AAresSeq_match("5.7*", select_keys=True)
        consensus   OPS  B2AR
    202   5.70x70  A235  Q231
    203   5.71x71  Q236  K232
    204   5.72x72  Q237  I233

    You notice the "MUOR"-column is missing, because it doesn't have '5.7*' residues

    """

    def __init__(self, maps, tops=None):
        r"""

        Parameters
        ----------
        maps : dict
            Dictionary of "maps", each one mapping residues
            to consensus labels. The keys of the dict
            can be arbitrary identifiers, to distinguish among
            the different systems, like different UniProt Codes,
            PDB IDs, or user specific for system-setups
            (WT vs MUT etc). The values in `maps` can be:

            * Type :obj:`~mdciao.nomenclature.LabelerGPCR`, :obj:`~mdciao.nomenclature.LabelerCGN`,
              or :obj:`~mdciao.nomenclature.LabelerKLIFS`
             Recommended option, the most succint and versatile.
             Pass this object and the maps will get created
             internally on-the-fly either by
             calling :obj:`~mdciao.nomenclature.LabelerGPCR.AA2conlab`
             (if no `tops` passed) or by calling :obj:`~mdciao.nomenclature.LabelerGPCR.top2labels`
             (if `tops` were passed).

            * Type dict.
             Only works if `tops` is None. Keyed with residue
             names (AAresSeq) and valued with consensus labels.
             Useful if for some reason you want to modify the dicts
             that are created by :obj:`~mdciao.nomenclature.LabelerGPCR.AA2conlab`
             before using them here.

            * Type list
             Only works if `tops` is not None. Zero-indexed with
             residue indices of the `tops` and valued with
             consensus labels. Useful if for some reason
             :obj:`~mdciao.nomenclature.LabelerGPCR.top2labels`
             doesn't work automatically on the `tops`, since
             :obj:`~mdciao.nomenclature.LabelerGPCR.top2labels`
             sometimes fails if it can't cleanly align the
             consensus sequences to the residues in the `tops`.

        tops : dict or None, default is None
            A dictionary of :obj:`~mdtraj.Topology` objects,
            which will allow to express the consensus alignment
            also in terms of residue indices and of atom indices,
            using :obj:`~mdciao.nomenclature.AlignerConsensus.CAidxs`
            and :obj:`~mdciao.nomenclature.AlignerConsensus.residxs`,
            respectively (otherwise these methods return None).
            If `tops` is present, self.keys will be in
            the same order as they appear in `tops`.
        """
        self._tops = tops
        # we keep the order of keys if top is present
        self._maps = {key : maps[key] for key in [maps.keys() if tops is None else tops.keys()][0]}
        self._keys = list(self._maps.keys())
        for key, imap in self.maps.items():
            if isinstance(imap, dict):
                assert self.tops is None, ValueError("If `maps` contains dictionaries, then `tops` has to be None")
                self._maps[key] = imap #nothing to do, already a dict
            elif isinstance(imap, list):
                assert self.tops is not None, ("If `maps` contains lists, then `tops` can't be None")
                self._maps[key] = {ii: lab for ii, lab in enumerate(imap)} # turn into dict
            elif isinstance(imap, LabelerConsensus):
                if self.tops is None:
                    self._maps[key]=imap.AA2conlab # nothing to do, already a dict
                else:
                    self._maps[key]={ii : lab for ii, lab in enumerate(imap.top2labels(self.tops[key]))}
            else:
                raise ValueError("`maps` should contain either list, dicts, or %s objects, but found %s for key %s"%(LabelerConsensus, type(imap), key))
        self._maps = {key : {ii : lab for ii, lab in imap.items() if str(lab).lower()!="none"} for key, imap in self.maps.items()}

        self._residxs = None
        for key, imap in self.maps.items():
            idf = _DataFrame([imap.values(), imap.keys()],
                             index=["consensus", key]).T

            if self._residxs is None:
                self._residxs = idf
            else:
                self._residxs = self._residxs.merge(idf, how="outer")

        sorted_keys = _sort_all_consensus_labels(self._residxs["consensus"], append_diffset=False)
        assert len(sorted_keys)==len(self._residxs["consensus"]),  (len(sorted_keys), len(self._residxs["consensus"]))
        self._residxs = self._residxs.sort_values("consensus", key=lambda col: col.map(lambda x: sorted_keys.index(x)))
        self._residxs = self._residxs.reset_index(drop=True)

        if self.tops is not None:
            self._AAresSeq, self._CAidxs = self.residxs.copy(), self.residxs.copy()
            self._residxs = self._residxs.astype({key: "Int64" for key in self.keys})
            for key in self.keys:
                not_nulls = ~self.residxs[key].isnull()
                self._AAresSeq[key] = [[_mdcu.residue_and_atom.shorten_AA(self.tops[key].residue(ii),keep_index=True) if not_null else ii][0]
                                       for not_null, ii in zip(not_nulls, self.residxs[key])]
                self._CAidxs[key] = [[self.tops[key].residue(ii).atom("CA").index if not_null else ii][0]
                                     for not_null, ii in zip(not_nulls, self.residxs[key])]
                self._maps[key] = {val : key for ii, (not_null, (key, val)) in enumerate(zip(not_nulls, self.AAresSeq[["consensus", key]].values)) if
                                                not_null}

            self._CAidxs = self._CAidxs.astype({key: "Int64" for key in self.keys})
        else:
            self._AAresSeq = self.residxs
            self._residxs, self._CAidxs = None, None

    @property
    def tops(self) -> dict:
        r"""
        The topologies given at input
        """
        return self._tops

    @property
    def maps(self) -> dict:
        r"""
        The dictionaries mapping residue names to consensus labels.
        """
        return self._maps

    @property
    def residxs(self) -> _DataFrame:
        r"""
        Consensus-label alignment expressed as zero-indexed residue indices of the respective `tops`

        Will have NaNs where residues weren't found,
        i.e. a given `map` didn't contain that consensus label

        Returns None if no `tops` were given at input.

        Returns
        -------
        df : :obj:`~pandas.DataFrame`
        """
        return self._residxs

    @property
    def keys(self) -> list:
        r"""
        The keys with which the `maps` and the `tops` were given at input
        """
        return self._keys

    @property
    def CAidxs(self) -> _DataFrame:
        r"""
        Consensus-label alignment expressed as atom indices of the Cɑ atoms of respective `tops`

        Will have NaNs where atoms weren't found,
        i.e. a given `map` didn't contain that consensus label

        Returns None if no `tops` were given at input.

        Returns
        -------
        df : :obj:`~pandas.DataFrame`
        """
        return self._CAidxs

    @property
    def AAresSeq(self) -> _DataFrame:
        r"""
        Consensus-label alignment expressed as residues

        Will have NaNs where residues weren't found,
        i.e. a given `map` didn't contain that consensus label

        Returns
        -------
        df : :obj:`~pandas.DataFrame`
        """
        return self._AAresSeq

    def residxs_match(self, patterns=None, keys=None, omit_missing=True, select_keys=False) -> _DataFrame:
        r"""
        Filter the `self.residxs` by rows and columns.

        You can filter by consensus label using `patterns` and by system using `keys`.

        By default, rows where None, or NaNs are present are excluded.

        Parameters
        ----------
        patterns : str, default is None
            A list in CSV-format of patterns to be matched
            by the consensus labels. Matches are done using
            Unix filename pattern matching, and are allows
            for exclusion, e.g. "3.*,-3.5*." will include all
            residues in TM3 except those in the segment 3.50...3.59
        keys : list, default is None
            If only a sub-set of columns need to match,
            provide them here as list of strings. If
            None, all columns will be used.
        select_keys : bool, default is False
            Use the `patterns` not only to select
            for rows but also to select for columns, i.e.
            for keys. Keys (=columns) not featuring
            any `patterns` will be dropped.
        omit_missing : bool, default is True
            Omit rows with missing values.

        Returns
        -------
        df : :obj:`~pandas.DataFrame`
        """
        return _only_matches(self.residxs, patterns=patterns, keys=keys, select_keys=select_keys, dropna=omit_missing,
                             filter_on="consensus")

    def AAresSeq_match(self, patterns=None, keys=None, omit_missing=True, select_keys=False) -> _DataFrame:
        r"""
        Filter the `self.AAresSeq` by rows and columns.

        You can filter by consensus label using `patterns` and by system using `keys`.

        By default, rows where None, or NaNs are present are excluded.

        Parameters
        ----------
        patterns : str, default is None
            A list in CSV-format of patterns to be matched
            by the consensus labels. Matches are done using
            Unix filename pattern matching, and are allows
            for exclusion, e.g. "3.*,-3.5*." will include all
            residues in TM3 except those in the segment 3.50...3.59
        keys : list, default is None
            If only a sub-set of columns need to match,
            provide them here as list of strings. If
            None, all columns will be used.
        select_keys : bool, default is False
            Use the `patterns` not only to select
            for rows but also to select for columns, i.e.
            for keys. Keys (=columns) not featuring
            any `patterns` will be dropped.
        omit_missing : bool, default is True
            Omit rows with missing values,

        Returns
        -------
        df : :obj:`~pandas.DataFrame`
        """
        return _only_matches(self.AAresSeq, patterns=patterns, keys=keys, select_keys=select_keys, dropna=omit_missing,
                             filter_on="consensus")

    def CAidxs_match(self, patterns=None, keys=None, omit_missing=True, select_keys=False) -> _DataFrame:
        r"""
        Filter the `self.CAidxs` by rows and columns.

        You can filter by consensus label using `patterns` and by system using `keys`.

        By default, rows where None, or NaNs are present are excluded.

        Parameters
        ----------
        patterns : str, default is None
            A list in CSV-format of patterns to be matched
            by the consensus labels. Matches are done using
            Unix filename pattern matching, and are allows
            for exclusion, e.g. "3.*,-3.5*." will include all
            residues in TM3 except those in the segment 3.50...3.59H8
             * "G.S*" will include all beta-sheets
        keys : list, default is None
            If only a sub-set of columns need to match,
            provide them here as list of strings. If
            None, all columns (except `filter_on`) will be used.
        select_keys : bool, default is False
            Use the `patterns` not only to select
            for rows but also to select for columns, i.e.
            for keys. Keys (=columns) not featuring
            any `patterns` will be dropped.
        omit_missing : bool, default is True
            Omit rows with missing values

        Returns
        -------
        df : :obj:`~pandas.DataFrame`
        """
        return _only_matches(self.CAidxs, patterns=patterns, keys=keys, select_keys=select_keys, dropna=omit_missing,
                             filter_on="consensus")

    def sequence_match(self,patterns=None, absolute=False)-> _DataFrame:
        r"""Matrix with the percentage of sequence identity within the set of the residues sharing consensus labels

        The comparison is done between the reference consensus sequences
        in self.AAresSeq, i.e., independently of any `tops` that the user
        has provided.

        Example:
        >>> AC.sequence_match(patterns="3.5*")
               OPS  B2AR  MUOR
        OPS   100%   29%   57%
        B2AR   29%  100%   43%
        MUOR   57%   43%  100%

        Meaning, for the residues having consensus labels 3.50 to 3.59,
        B2AR and OPS have 29% identity, or OPS and MUOR 57%. You can
        express this as absolute nubmer of residues:
        >>> AC.match_percentage("3.*", absolute=True)
              OPS  B2AR  MUOR
        OPS     7     2     4
        B2AR    2     7     3
        MUOR    4     3     7

        You can check which residues these are:
        >>> AC.AAresSeq_match("3.5*")
            consensus   OPS  B2AR  MUOR
        117   3.50x50  R135  R131  R165
        118   3.51x51  Y136  Y132  Y166
        119   3.52x52  V137  F133  I167
        120   3.53x53  V138  A134  A168
        121   3.54x54  V139  I135  V169
        122   3.55x55  C140  T136  C170
        123   3.56x56  K141  S137  H171

        You can see the two OPS/B2AR matches in 3.50x50 and 3.51x51

        Parameters
        ----------
        patterns : str, default is None
            A list in CSV-format of patterns to be matched
            by the consensus labels. Matches are done using
            Unix filename pattern matching, and are allows
            for exclusion, e.g. "3.*,-3.5*." will include all
            residues in TM3 except those in the segment 3.50...3.59
        absolute : bool, default is False
            Instead of returning a percentage, return
            the nubmer of matching residues as integers

        Returns
        -------
        match :  :obj:`~pandas.DataFrame`
            The matrix of sequence identity, for residues sharing consensus labels across
            the different systems.
        """
        match = _defdict(dict)
        for key1 in self.keys:
            for key2 in self.keys:
                if key1!=key2:
                    res = self.AAresSeq_match(patterns=patterns, keys=[key1,key2])
                    ident = (res[key1].map(lambda x : x[0])==res[key2].map(lambda x : x[0])).sum()
                else:
                    res = self.AAresSeq_match(patterns=patterns, keys=[key1])
                    ident = len(res)
                if not absolute:
                    ident = _np.round(ident / len(res) * 100)
                match[key1][key2] = ident  # .round()

        df = _DataFrame(match, dtype=int)

        if not absolute:
            df = df.applymap(lambda x: "%u%%" % x)

        return df

def _only_matches(df: _DataFrame, patterns=None, keys=None, select_keys=False, dropna=True, filter_on="index") -> _DataFrame:
    r"""
    Row-filter an :obj:`~pandas.DataFrame` by patterns in the values and column-filter by keys in the column names

    Filtering means:
     * don't include rows where None or Nans appear (except if relax=True)
     * include  anything that matches `patterns`

    Parameters
    ----------
    df : :obj:`~pandas.DataFrame` or None
        The dataframe to be filter by matching `patterns` and `keys`.
        If None, the method simply returns None.
    patterns : str, default is None
        A list in CSV-format of patterns to be matched
        by the consensus labels. Matches are done using
        Unix filename pattern matching, and are allows
        for exclusion, e.g. "3.*,-3.5*." will include all
        residues in TM3 except those in the segment 3.50...3.59
    keys : list, default is None
        If only a sub-set of columns need to match,
        provide them here as list of strings. If
        None, all columns (except `filter_on`) will be used.
    select_keys : bool, default is False
        Use the `patterns` not only to select
        for rows but also to select for columns, i.e.
        for keys. Keys (=columns) not featuring
        any `patterns` will be dropped.
    dropna : bool, default is True
        Use :obj:`~pandas.Dataframe.dropna` row-wise
        before returning
     filter_on : str, default is 'index'
        The column of `df` on which the `patterns`
        will be used for a match

    Returns
    -------
    df : :obj:`~pandas.DataFrame`
        Will only have `filter_on`+`keys` as columns
    """
    if df is None:
        return None
    if keys is None:
        keys = [key for key in df.keys() if key != filter_on]

    if patterns is not None:
        matching_keys = _mdcu.str_and_dict.fnmatch_ex(patterns, df[filter_on])
        matches = df[filter_on].map(lambda x: x in matching_keys)
        df = df[matches]
    df = df[[filter_on]+keys]

    if select_keys:
        df = df[[key for key in df.keys() if not all(df[key].isna())]]
    if dropna:
        df = df.dropna()

    # Try to return integers when possible
    try:
        df = df.astype({key:int for key in df.keys() if key!=filter_on})
    except (ValueError, TypeError) as e:
        pass
    return df


def _alignment_df2_conslist(alignment_as_df,
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
    out_list = _np.full(n_residues + 1, None)

    if allow_nonmatch:
        _df = _mdcu.sequence.re_match_df(alignment_as_df)
    else:
        _df = alignment_as_df
    _df = _df[_df["match"]]

    out_list[_df["idx_0"].values.astype(int)] = _df["conlab"].values
    return out_list.tolist()


def _fill_consensus_gaps(consensus_list, top, verbose=False):
    r""" Try to fill consensus-nomenclature gaps based on adjacent labels

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

    Note
    ----
    Currently, only Ballesteros-Weinstein (Class A GPCR-nomenclature scheme)
    is supported by this method.

    Parameters
    ----------
    consensus_list: list
        List of length top.n_residues with the original consensus labels
        In principle, it could contain some "None" entries inside sub-domains
    top :
        :obj:`~mdtraj.Topology` object
    verbose : boolean, default is False

    Returns
    -------
    consensus_list: list
        The same as the input :obj:`consensus_list` with guessed missing entries
    """

    defs = _map2defs(consensus_list)
    # todo decrease verbosity
    # Iterate over fragments
    for frag_key, conlabs in defs.items():
        # Identify problem cases
        if len(conlabs) != conlabs[-1] - conlabs[0] + 1:
            if verbose:
                print(frag_key)
            if "x" in consensus_list[conlabs[0]]:
                raise ValueError(
                    "Can't fill gaps in non 'BW' GPCR-nomenclature, like the provided '%s'" % consensus_list[
                        conlabs[0]])

            # Initialize residue_idxs_wo_consensus_labels control variables
            offset = int(consensus_list[conlabs[0]].split(".")[-1])
            consensus_kept = True
            suggestions = []
            residue_idxs_wo_consensus_labels = []

            # Check whether we can predict the consensus labels correctly
            for ii in _np.arange(conlabs[0], conlabs[-1] + 1):
                suggestions.append('%s.%u' % (frag_key, offset))
                if consensus_list[ii] is None:
                    residue_idxs_wo_consensus_labels.append(ii)
                else:  # meaning, we have a consensus label, check it against suggestion
                    consensus_kept *= suggestions[-1] == consensus_list[ii]
                if verbose:
                    print('%6u %8s %10s %10s %s' % (
                        ii, top.residue(ii), consensus_list[ii], suggestions[-1], consensus_kept))
                offset += 1
            if verbose:
                print()
            if consensus_kept:
                if verbose:
                    print("The consensus was kept, I am relabelling these:")
                for idx, res_idx in enumerate(_np.arange(conlabs[0], conlabs[-1] + 1)):
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


def choose_between_consensus_dicts(idx, consensus_maps, no_key="NA"):
    """
    Choose the best consensus label for a given :obj:`idx` in case
    there are more than one consensus(es) at play (e.g. GPCR and CGN).

    Wil raise error if both dictionaries have a consensus label for
    the same index (unusual case)

    Parameters
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
    labels = [idict[idx] for idict in consensus_maps]
    good_label = _np.unique([ilab for ilab in labels if str(ilab).lower() != "none"]).tolist()
    assert len(good_label) <= 1, "There can only be one good label, but for residue %u found %s" % (idx, good_label)
    try:
        return good_label[0]
    except IndexError:
        return no_key


def guess_nomenclature_fragments(refseq, top,
                                 fragments=None,
                                 min_hit_rate=.6,
                                 verbose=False,
                                 return_residue_idxs=False,
                                 empty=list):
    """Guess what fragments in the topology best match
    the consensus labels in a :obj:`LabelerConsensus` object

    The guess uses a cutoff, `min_hit_rate`, for the quality of
    each segment's alignment to the sequence in `CLin`.

    It only counts the matches for the protein residues of
    `top`, i.e. waters, ions, non-peptidic polymers are
    note taken into account.

    You can use the method to identify the receptor
    in topology where other molecules (e.g. the Gprot)
    are present (or the other way around)

    Parameters
    ----------
    refseq: str or :class:`LabelerConsensus`
        If not str, the sequence will
        be gotten from `LabelerConsensus.seq` method
    top:
        :py:class:`~mdtraj.Topology` object or string
        containing the sequence.
    fragments : iterable of iterables of idxs
        How `top` is split into fragments
        If None, will be generated using get_fragments defaults
    min_hit_rate: float, default is .6
        Only fragments with hit rates higher than this
        will be returned as a guess
    verbose: boolean
        be verbose
    return_residue_idxs : bool, default is False
        Return the list of residue indices directly,
        instead of returning a list of fragment idxs.
    empty : class, list or None, default is list
        What to return in case of an emtpy guess,
        an empty list or a None
    Returns
    -------
    guess: list
        indices of the fragments (or residues) with higher hit-rate than `min_hit_rate`


    """

    if fragments is None:
        fragments = _mdcfrg.get_fragments(top, verbose=False)

    try:
        seq_consensus = refseq.seq
    except AttributeError:
        assert isinstance(refseq, str), "refseq has to be either a %s with a 'seq' method or a string" \
                                        "but not a %s." % (LabelerConsensus, type(str))
        seq_consensus = refseq

    if isinstance(top, _md.Topology):
        protein_df = _DataFrame({"idx_0": _np.arange(top.n_residues),
                                 "is_protein": [rr.is_protein for rr in top.residues]})
    else:
        protein_df = _DataFrame({"idx_0": _np.arange(len(top)),
                                 "is_protein": [char in _AA_chars_no_X for char in top]})
    # TODO create a method out of this
    df = _mdcu.sequence.align_tops_or_seqs(top, seq_consensus)[0] #picking the first one here w/o asking if there's equivalent aligns might be problematic
    df = df.merge(protein_df, how="inner")
    hit_idxs = df[df.match & df.is_protein].idx_0.values
    hits, guess = [], []
    for ii, ifrag in enumerate(fragments):
        hit = _np.intersect1d(ifrag, hit_idxs)
        if len(hit) / len(ifrag) >= min_hit_rate:
            guess.append(ii)
        if verbose:
            print(ii, len(hit) / len(ifrag), len(hit), len(ifrag))
        hits.append(hit)

    guessed_res_idxs = []
    if len(guess) > 0:
        guessed_res_idxs = _np.hstack([fragments[ii] for ii in guess])

    if return_residue_idxs:
        guess = guessed_res_idxs

    if empty is None and len(guess) == 0:
        guess = None
    return guess

@_kwargs_subs(guess_nomenclature_fragments, exclude=["fragments", "return_residue_idxs"])
def guess_by_nomenclature(CLin, top, fragments=None, nomenclature_name=None,
                          return_str=False, accept_guess=False,
                          return_residue_idxs=False,
                          **guess_kwargs):
    r"""

    Guess which fragments of a topology best align with a consensus nomenclature.

    Wraps around :obj:`guess_nomenclature_fragments` to interpret its answer.

    Parameters
    ----------
    CLin : :obj:`LabelerConsensus`
        The nomenclature object to use
        for guessing
    top : :obj:`~mdtraj.Topology`
        The topology whose fragments
        are being matched with those
        in :obj:`CLin`
    fragments : iterable of iterables of idxs or str, default is None
        How `top` is split into fragments. If list of lists of indices,
        these are the fragments expressed as residue indices
        of `top`. If None or str, the `fragments` will be generated
        using :obj:`mdciao.fragments.get_fragments`,
        either with the default scheme (if `fragments` is None)
        or with the scheme-name provided by `fragments` (e.g.
        `fragments`="chains" or `fragments`="resSeq+".
        See the note at the bottom for how `fragments`
        relates to `return_residue_idxs`.
    nomenclature_name : str, default is None
        A string identifying the nomenclature
        type, e.g. "CGN" or "GPCR". For logging
        purposes. If None, it will the derived
        from the class name of `CLin`.
    return_str : bool, default is False
        Return the answer the user provided
        in case an alternative guess.
    accept_guess : bool, default is False
        Accept the guess of :obj:`guess_nomenclature_fragments`
        without asking for confirmation.
    return_residue_idxs : bool, default is False
        Return the list of residue indices directly,
        instead of returning a list of fragment idxs.
        Only has an effect if
    guess_kwargs : dict, optional
        Keyword arguments for some of the keyword arguments
        of :obj:`guess_nomenclature_fragments`, which
        are listed below.

    Other Parameters
    ----------------
    %(substitute_kwargs)s

    Returns
    -------
    answer : list or None
        The indices of the fragments of `top` that best
        match the reference sequence in `CLin`. If none match,
        then None will be returned. If `return_residue_idxs`
        is True, then the indices will be actual residue indices,
        and not fragment indices (see the note below for more).
        If return_str is True, the the answer is returned
        as a string of comma-separated-values.
    Note
    ----
    If the user doesn't provide any `fragments`, this method will generate
    them on-the-fly and subsequently guess which one of them matches
    `CLin` the best. Hence, `fragments`=None only makes sense if
    `return_residue_idxs` is True, because then the actual
    residue indices of `top` that best match `CLin` are returned.
    Otherwise, the method would say something like "The labels align best
    with fragments 0,1", and return the indices
    [0,1], which are useless since we don't know what
    the fragments are. In this case, an Exception is thrown.

    """
    if nomenclature_name is None:
        nomenclature_name = type(CLin).__name__.replace("Labeler", "")
    if fragments is None or isinstance(fragments,str):
        fragments = _mdcfrg.get_fragments(top, [_signature(_mdcfrg.get_fragments).parameters["method"].default if fragments is None else fragments][0],
                                          verbose=True)
        if return_residue_idxs is False:
            raise ValueError("`fragments` can't be None if `return_residue_idxs` is False.\n"
                             "Please see the note at the end of this method's documentation")
    guess = guess_nomenclature_fragments(CLin, top, fragments=fragments, **guess_kwargs)
    guess_as_string = ','.join(['%s' % ii for ii in guess])

    if len(guess) > 0:
        print("The %s-labels align best with fragments: %s (first-last: %s-%s)."
              % (nomenclature_name, str(guess),
                 top.residue(fragments[guess[0]][0]),
                 top.residue(fragments[guess[-1]][-1])))
    else:
        print("None of the above fragments appear to have sequences matching "
              "the consensus\nlabeler (type '%s')."%type(CLin).__name__)
        return None

    if accept_guess:
        answer = guess_as_string
    else:
        answer = input(
            "Input alternative in a format 1,2-6,10,20-25 or\nhit enter to accept the guess %s\n" % guess_as_string)

    if answer == '':
        answer = guess_as_string
    else:
        answer = ','.join(['%s' % ii for ii in _mdcu.lists.rangeexpand(answer)])

    if return_str:
        pass
    else:
        answer = [int(ii) for ii in answer.split(",")]
        if return_residue_idxs:
            answer = _np.hstack([fragments[ii] for ii in answer]).tolist()
    return answer


def _map2defs(cons_list, splitchar="."):
    r"""
    Subdomain definitions form a list of consensus labels.

    The indices of the list are interpreted as residue indices
    in the topology used to generate :obj:`cons_list`
    in the first place, e.g. by using :obj:`nomenclature_utils._top2consensus_map`

    Note:
    -----
    The method will guess automagically whether this is a CGN or GPCR label by
    checking the type of the first character (numeric is GPCR, 3.50, alpha is CGN, G.H5.1)

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
    defs : dictionary
        dictionary keyed with subdomain-names and valued with arrays of residue indices
    """
    defs = _defdict(list)
    for ii, key in enumerate(cons_list):
        if str(key).lower() != "none":
            assert splitchar in _mdcu.lists.force_iterable(key), "Consensus keys have to have a '%s'-character" \
                                                                 " in them, but '%s' (type %s) hasn't" % (
                                                                     splitchar, str(key), type(key))
            if key[0].isnumeric():  # it means it is GPCR
                new_key = key.split(splitchar)[0]
            elif key[0].isalpha():  # it means it CGN
                new_key = '.'.join(key.split(splitchar)[:-1])
            else:
                raise Exception([ii, splitchar])
            defs[new_key].append(ii)
    return {key: _np.array(val) for key, val in defs.items()}


def _sort_consensus_labels(subset, sorted_superset,
                           append_diffset=True):
    r"""
    Sort consensus labels (GPCR or CGN)

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
            frag, idx = item.rsplit(".", maxsplit=1)
            by_frags[frag][idx] = item
        except:
            pass

    # In case any negative integers are in the keys (=labels) convert them to ints s.t. natsort can work with them
    by_frags = {key : {[int(key2) if any([val2.startswith("-") for val2 in val]) else key2][0]: val2
                       for key2, val2 in val.items()
                       } for key, val in by_frags.items()}

    labs_out = []
    for frag in sorted_superset:
        if frag in by_frags.keys():
            for ifk in _natsorted(by_frags[frag].keys()):
                labs_out.append(by_frags[frag][ifk])

    if append_diffset:
        labs_out += [item for item in subset if item not in labs_out]

    return labs_out


def _sort_GPCR_consensus_labels(labels, **sort_consensus_labels_kwargs):
    r"""
    Sort consensus labels in order of appearance in the canonical GPCR scheme

    Parameters
    ----------
    labels : iterable
        The input consensus labels
    sort_consensus_labels_kwargs : dict, optional
        Optional arguments for :obj:`sort_consensus_labels`

    Returns
    -------

    """
    return _sort_consensus_labels(labels, _GPCR_fragments, **sort_consensus_labels_kwargs)

def _sort_KLIFS_consensus_labels(labels, **sort_consensus_labels_kwargs):
    r"""
    Sort consensus labels in order of appearance in the canonical KLIFS scheme

    Parameters
    ----------
    labels : iterable
        The input consensus labels
    sort_consensus_labels_kwargs : dict, optional
        Optional arguments for :obj:`sort_consensus_labels`

    Returns
    -------

    """
    return _sort_consensus_labels(labels, _KLIFS_fragments, **sort_consensus_labels_kwargs)


def _sort_CGN_consensus_labels(labels, **kwargs):
    r"""
    Sort consensus labels in order of appearance in the canonical GPCR scheme

    Parameters
    ----------
    labels : iterable
        The input consensus labels
    kwargs : dict, optional
        Optional arguments for :obj:`sort_consensus_labels`

    Returns
    -------

    """
    return _sort_consensus_labels(labels, _CGN_fragments, **kwargs)


def _conslabel2fraglabel(labelres, defrag="@", prefix_GPCR=True):
    r"""
    Return a fragment label from a full consensus following some norms

    Parameters
    ----------
    labelres : str
        The residue label, e.g.
        "GLU30@3.50"
    defrag : char, default is "@"
        The character separating
        residue and consensus label
    prefix_GPCR : bool, default is True
        If True, things like "3" (from "3.50")
        will be turned into "TM3"

    Returns
    -------
    labelfrag
    """

    label = labelres.split(defrag)[-1]
    label = label.rsplit(".", maxsplit=1)[0]
    if prefix_GPCR and str(label) in _GPCR_num2lett.keys():
        label = _GPCR_num2lett[label]
    return label

def _sort_all_consensus_labels(labels, append_diffset=True, order=["GPCR","CGN","KLIFS"], ):
    r"""
    Sort a mix of consensus labels GPCR, CGN, KLIFS

    Parameters
    ----------
    labels : list
        List of consensus labels. If
        it contains other labels like
        "None", None, "bogus", these
        labels will be put at the
        end of `sorted_labels` unless
        explicitly deactivated with
        `append_diffset`.
        append_diffset : bool, default is True
        Append the non-consensus labels
        at the end of `sorted_labels`
    order : list
        The order in which consensus labels
        will be sorted. Keys missing here
        won't be sorted in the output, i.e.
        if order=["GPCR","KLIFS"] then
        all CGN labels (if any) will either
        be at the end of the list (`append_diffset` = True)
        or entirely missing (`append_diffset` = False)

    Returns
    -------
    sorted_labels : list
        Sorted consensus labels
    """

    lambdas = {"GPCR":  lambda labels: _sort_GPCR_consensus_labels(labels, append_diffset=False),
               "CGN":   lambda labels: _sort_CGN_consensus_labels(labels, append_diffset=False),
               "KLIFS": lambda labels: _sort_KLIFS_consensus_labels(labels, append_diffset=False)}

    sorted_labels = []
    for key in order:
        sorted_labels += lambdas[key](labels)
    if append_diffset:
        sorted_labels += [lab for lab in labels if lab not in sorted_labels]

    return sorted_labels

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

_GPCR_fragments = ["NT",
                   "1", "TM1 ",
                   "12", "ICL1",
                   "2", "TM2",
                   "23", "ECL1",
                   "3", "TM3",
                   "34", "ICL2",
                   "4", "TM4",
                   "45", "ECL2",
                   "5", "TM5",
                   "56", "ICL3",
                   "6", "TM6",
                   "67", "ECL3",
                   "7", "TM7",
                   "78",
                   "8", "H8",
                   "CT"]

_CGN_fragments = ['G.HN',
                  'G.hns1',
                  'G.S1',
                  'G.s1h1',
                  'G.H1',
                  'G.h1ha',
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

_KLIFS_fragments = ['I',
                    'g.l',
                    'II',
                    'III',
                    'αC',
                    'b.l',
                    'IV',
                    'V',
                    'GK',
                    'hinge',
                    'linker',
                    'αD',
                    'αE',
                    'VI',
                    'c.l',
                    'VII',
                    'VIII',
                    'xDFG',
                    'a.l']


_GPCR_mandatory_fields = ["protein_segment",
                          "AAresSeq",
                          "display_generic_number",
                          ]

_GPCR_available_schemes = ["BW",
                           "Wootten", "Pin", "Wang", "Fungal",
                           "GPCRdb(A)", "GPCRdb(B)", "GPCRdb(C)", "GPCRdb(F)", "GPCRdb(D)",
                           "Oliveira", "BS",
                           ]


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
        List of individual consensus maps, typically GPCR
        or CGN maps. These list are maps in this sense:
        cons_map[res_idx] = "3.50"
    CLs : list
        List of :obj:`mdciao.nomenclature.LabelerConsensus`-objects
        that will generate new consensus maps for all residues
        in :obj:`top`
    autofill_consensus : boolean default is False
        Even if there is a consensus mismatch with the sequence of the input
        :obj:`AA2conlab_dict`, try to relabel automagically, s.t.
         * ['G.H5.25', 'G.H5.26', None, 'G.H.28']
         will be grouped relabeled as
         * ['G.H5.25', 'G.H5.26', 'G.H.27', 'G.H.28']

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
    unified_new_consensus_map = [choose_between_consensus_dicts(idx, new_maps, no_key=None) for idx in
                                 range(top.n_residues)]

    # Now incorporate new labels while checking with clashes with old ones
    for ii in range(top.n_residues):
        existing_val = unified_existing_consensus_map[ii]
        new_val = unified_new_consensus_map[ii]
        # take the new val, even if it's also None
        if existing_val is None:
            unified_existing_consensus_map[ii] = new_val
        # otherwise check no clashes with the existing map
        else:
            # print(existing_val, "ex not None")
            # print(new_val, "new val")
            assert existing_val == new_val, (new_val, existing_val, ii, top.residue(ii))

    new_frags = {}
    for iCL in CLs:
        new_frags.update(iCL.top2frags(top,
                                       # map_conlab=unified_new_consensus_map,
                                       verbose=False))

    # This should hold anyway bc of top2frags calling conlab2residx
    _mdcu.lists.assert_no_intersection(list(new_frags.values()))

    return new_frags


def _consensus_maps2consensus_frags(top, consensus_info, verbose=True, fragments=None):
    r"""
    Consensus fragments (like TM6 or G.H5) and maps from different input types

    Note
    ----
    This is a low-level, ad-hoc method not intended
    for API use It's very similar to
    cli._parse_consensus_options_and_return_fragment_defs,
    with which it might be merged in the future

    Parameters
    ----------
    top
    consensus_info : list
        The items of this list can be a mix of:
         * indexables containing the consensus
            labels (strings) themselves. They
            need to be "gettable" by residue index, i.e.
            dict, list or array. Typically, one
            generates these maps by using the top2labels
            method of the LabelerConsensus object.
            These will be returned "untouched" in
            :obj:`consensus_maps`
         * :obj:`LabelerConsensus`-objects
            Where the fragments are obtained from.
            Additionally, their
            top2labels and top2fragments methods are
            called on-the-fly, generating lists
            like the ones described above.
    verbose : bool, default is True
    fragments : iterable of ints, default is None
        The purpose of passing other fragment definitions
         here is that they **might** be used to
         to check whether obtained consensus fragments clash
         with these definitions or not. This is done by calling
         :obj:`~mdciao.fragments.check_if_subfragment`, check
         the docs there to find out what 'clash' means).
         The check will be carried out by
         :obj:`mdciao.nomenclature.LabelerConsensus.aligntop`
         only in cases when there's more than one optimal alignment.


    Returns
    -------
    consensus_maps : list
        Per-residue maps, one per each object in
        the input :obj:`consensus_maps`. If it
        already was a map, it is returned untouched,
        if it was an :obj:`LabelerConsensus`, a map
        was created out of it
    consensus_frags : dict
        One flat dict with all the consensus fragments from
        all (if any) the LabelerConsensus
        (not the maps) in the :obj:`consensus info`
    """

    consensus_frags = [cmap.top2frags(top, verbose=verbose, fragments=fragments) for cmap in consensus_info if
                       isinstance(cmap, LabelerConsensus)]
    _mdcu.lists.assert_no_intersection([item for d in consensus_frags for item in d.values()], "consensus fragment")
    consensus_frags = {key: val for d in consensus_frags for key, val in d.items()}
    consensus_maps = [cmap if not isinstance(cmap, LabelerConsensus) else cmap.top2labels(top, fragments=fragments) for
                      cmap
                      in consensus_info]
    return consensus_maps, consensus_frags


class Literature():
    r"""Quick access to the some of the references used by :obj:`nomenclature`"""

    # TODO this could be fine tuned but ATM its better to have all top-level attrs
    def __init__(self):
        self._keymap = {
            "site_GPCRdb": "Kooistra2021",
            "site_PDB": "Berman2000",
            "scheme_GPCR_struct1": "Isberg2015",
            "scheme_GPCR_struct2": "Isberg2016",
            "scheme_GPCR_A": "Ballesteros1995",
            "scheme_GPCR_B": "Wootten2013",
            "scheme_GPCR_C": "Pin2003",
            "scheme_GPCR_F": "Wu2014",
            "scheme_GPCR_A_O": "Oliveira1993",
            "scheme_GPCR_A_BS_1": "Baldwin1993",
            "scheme_GPCR_A_BS_2": "Baldwin1997",
            "scheme_GPCR_A_BS_3": "Schwartz1994",
            "scheme_GPCR_A_BS_4": "Schwartz1995",
            "scheme_CGN": "Flock2015",
            "site_UniProt": "Bateman2021",
            "site_KLIFS": "Kanev2021",
            "scheme_KLIFS1": "VanLinden2014",
            "scheme_KLIFS2": "Kooistra2016"
        }

        arts = _parse_bib()

        assert all([key in self._keymap.values() for key in arts.keys()]), (self._keymap.values(), arts.keys())
        for key, val in self._keymap.items():
            setattr(self, key, _art2cite(arts[val]))
            setattr(self, key + '_json', arts[val])


def _format_cite(cite, bullet="*", indent=1):
    star = " " * indent + bullet + " "
    othr = " " * indent + " " + " "
    lines = cite.splitlines()
    lines = _twrap(lines[0], 100) + lines[1:]
    lines = "\n".join([star + lines[0]] + [othr + line for line in lines[1:]])
    return lines


def _art2cite(art):
    first = True
    # first = False
    authors = art.author.split(" and ")
    if first:
        lname = authors[0].split(",")[0].strip()
    else:
        lname = authors[-1].split(",")[0].strip()
    lines = []
    lines.append("%s et al, (%s) %s" % (lname, art.year, art.title))
    lines.append("%s %s, %s" % (art.journal, art.volume, art.pages))
    try:
        lines.append("https://doi.org/%s" % art.doi)
    except AttributeError:
        pass
    lines = "\n".join(lines)
    return lines


def _parse_bib(bibfile=None):
    r"""
    Parse a latex .bib bibliography file and return entries as dictionaries

    The only recognized key is "@article", other entries will break this method

    Parameters
    ----------
    bib : str or None
        Path to a bibfile
        The default is to use
        :obj:`mdciao.filenames.FileNames.nomenclature_bib`

    Returns
    -------
    articles : dict
        Dictionary of dictionaries
        containing whatever was
        available in :obj:`bib`
    """
    if bibfile is None:
        bibfile = _filenames.nomenclature_bib
    with open(bibfile) as f:
        bibstr = f.read()

    articles = [[line.rstrip(",") for line in art.rstrip("\n").strip("{}").splitlines()] for art in
                bibstr.split("@article")]
    articles = {art[0]: art[1:] for art in articles if len(art) > 0}
    articles = {key: _art2dict(art) for key, art in articles.items()}
    articles = {key: _dict2namedtupple(val, key) for key, val in articles.items()}
    return articles


def _art2dict(lines):
    art = {}
    for line in lines:
        key, val = line.split("=", 1)
        art[key.strip()] = val.strip().strip("{}")
    return art


def _dict2namedtupple(idict, key):
    nt = _namedtuple("article", list(idict.keys()) + ["key"])
    return nt(*idict.values(), key)


def references():
    r"""
    Print out references relevant to this module
    """

    lit: Literature = Literature()
    print("mdciao.nomenclature functions thanks to these online databases. "
          "Please cite them if you use this module:")
    for attr in ["site_GPCRdb", "site_PDB", "scheme_CGN", "site_KLIFS"]:
        print(_format_cite(getattr(lit, attr)))
    print()

    print("Additionally, depending on the chosen nomenclature type, you should cite:")
    print(" * Structure based GPCR notation")
    for attr in ["scheme_GPCR_struct1", "scheme_GPCR_struct2"]:
        print(_format_cite(getattr(lit, attr), bullet="-", indent=3))
    print(" * Sequence based GPCR schemes:")
    for attr in ["scheme_GPCR_B",
                 "scheme_GPCR_C",
                 "scheme_GPCR_F",
                 "scheme_GPCR_A_O",
                 "scheme_GPCR_A_BS_1",
                 "scheme_GPCR_A_BS_2",
                 "scheme_GPCR_A_BS_3",
                 "scheme_GPCR_A_BS_4"]:
        print(_format_cite(getattr(lit, attr), bullet="-", indent=3))
    print(" * KLIFS 85 ligand binding site residues of kinases")
    for attr in ["scheme_KLIFS1", "scheme_KLIFS2", "site_KLIFS"]:
        print(_format_cite(getattr(lit, attr), bullet="-", indent=3))
    print()
    print("You can find all these references distributed as BibTex file distributed with mdciao here")
    print(" * %s" % _filenames.nomenclature_bib)


def _UniProtACtoPDBs(UniProtAC,
                     UniProtKB_API="https://rest.uniprot.org/uniprotkb"):
    r"""
    Retrieve PDB entries (and some metadata) associated with a UniprotAccession code
    by contacting the UniProt Knowledgebase (https://www.uniprot.org/help/uniprotkb)
    and looking up the 'uniProtKBCrossReferences' entry of the response

    Importantly, this metadata contains what chains and residue (sequence) indices
    are associated to the :obj:`UniProtAC`

    Since the metadata itself contains the PDB_id in uppercase under
    'id', the return dictionary is keyed with uppercase PDB_ids

    One such entry is returned as
    >>>     {'3E88': {'database': 'PDB',
    >>>               'id': '3E88',
    >>>               'properties': [{'key': 'Method', 'value': 'X-ray'},
    >>>                              {'key': 'Resolution', 'value': '2.50 A'},
    >>>                              {'key': 'Chains', 'value': 'A/B=146-480'}]}}

    Parameters
    ----------
    UniProtAC : str
        UniProt Accession code, e.g. 'P31751'
        Check for more info check https://www.uniprot.org/help/accession_numbers
    UniProtKB_API : str
        The url for programmatic access

    Returns
    -------
    PDBs_UPKB : dict
        PDB metadata associated with the
        :obj:`UniProtAC`, keyed with the
        PDB-ids (four letter codes) themselves

    """

    url = "%s/%s.json" % (UniProtKB_API, UniProtAC)
    PDBs_UPKB = {}
    with _requests.get(url) as resp:
        print("Please cite the following reference to the UniProt Knowledgebase:")
        lit = Literature()
        print(_format_cite(lit.site_UniProt))
        print("For more information, call mdciao.nomenclature.references()")
        data = resp.json()
        for entry in data["uniProtKBCrossReferences"]:
            # print(entry)
            if entry["database"].lower() == "pdb":
                PDBs_UPKB[entry["id"].upper()] = entry
    return PDBs_UPKB


def _mdTopology2residueDF(top) -> _DataFrame:
    r"""
    Return an :obj:`~mdtraj.Topology` as a :obj:`~pandas.DataFrame`

    The residue attributes are mapped to columns follows:
    "serial_index": rr.index
    "residue" : rr.name
    "code" : rr.code
    "Sequence_Index" : rr.resSeq
    "AAresSeq": shorten_AA(rr, substitute_fail="X", keep_index=True)
    "chain_index ": rr.chain.index

    Parameters
    ----------
    top : :obj:`~mdtraj.Topology`

    Returns
    -------
    df : :obj:`~pandas.DataFrame`
    """
    for_DF = []
    for rr in top.residues:
        for_DF.append({"serial_index": rr.index,
                       "residue": rr.name,
                       "code": rr.code,
                       "Sequence_Index": rr.resSeq,
                       "AAresSeq": _mdcu.residue_and_atom.shorten_AA(rr, substitute_fail="X", keep_index=True),
                       "chain_index": rr.chain.index})
    return _DataFrame(for_DF)


def _mdTrajectory2spreadsheets(traj, dest, **kwargs_to_excel):
    r"""

    Parameters
    ----------
    traj : obj`:~mdtraj.Trajectory`
    dest : :obj:`~pandas.ExcelWriter` or str
        Either the open ExcelWriter object or,
        if a string is passed, a the ExcelWriter
        object will be created on the fly
    kwargs_to_excel : keyword args for :obj:`pandas.DataFrame.to_excel`
        Can't contain sheet_name or index, will raise Exception

    """
    topdf, bonds = traj.top.to_dataframe()
    bondsdf, xyzdf = _DataFrame(bonds), _DataFrame(traj.xyz[0])
    unitcelldf = _DataFrame({"lengths": traj.unitcell_lengths[0],
                             "angles": traj.unitcell_angles[0]})

    # When dropping py36 support, use directly the contextlib.nullcontext for py37 and beyond
    # slack FTW :https://stackoverflow.com/a/55902915
    @contextmanager
    def nullcontext(enter_result=None):
        yield enter_result

    if isinstance(dest, str):
        cm = _ExcelWriter(dest)
    else:
        cm = nullcontext(dest)

    with cm as f:
        _DataFrame.to_excel(topdf, f, sheet_name="topology", index=False, **kwargs_to_excel)
        _DataFrame.to_excel(bondsdf, f, sheet_name="bonds", index=False, **kwargs_to_excel)
        _DataFrame.to_excel(unitcelldf, f, sheet_name="unitcell", index=False, **kwargs_to_excel)
        _DataFrame.to_excel(xyzdf, f, sheet_name="xyz", index=False, **kwargs_to_excel)


def _Spreadsheets2mdTrajectory(source):
    r""" Generate a :obj:`~mdtraj.Trajectory` from an Excel file or a dict with DataFrames

    Parameters
    ----------
    source : dict or str
        A dictionary of :obj:`~pandas.Dataframe`
        or a filename of an ExcelFile. It
        has to contain the keys or sheet names
        "topology", "bonds", "xyz" and "unitcell",
        and will have been typically generated
        with :obj:`_mdTrajectory2spreadsheets`

    Returns
    -------
    traj : :obj:`~mdtraj.Trajectory`

    """
    from ._md_from_dataframe import from_dataframe as _from_dataframe
    if isinstance(source, dict):
        idict = source
    else:
        idict = _read_excel(source,
                            None,
                            engine="openpyxl")
    idict["topology"].segmentID.fillna("", inplace=True)
    topology = _from_dataframe(idict["topology"], bonds=idict["bonds"].values)
    xyz = idict["xyz"].values
    geom = _md.Trajectory(xyz, topology,
                          unitcell_angles=idict["unitcell"].angles,
                          unitcell_lengths=idict["unitcell"].lengths)
    return geom


def _residx_from_UniProtPDBEntry_and_top(PDBentry, top, target_resSeqs=None):
    r"""
    Return the indices of residues of a :obj:`top` matching the 'Chains' entry of :obj:`PDBentry`

    The method picks the first chain of `top` that matches any of the entries in 'Chains',
    but it can be forced to keep trying with other chains of `top` via the parameter `target_resSeq`

    Parameters
    ----------
    PDBentry : dict
        One of the dictionaries coming from
        :obj:`_UniProtACtoPDBs` containing
        metadata associated with a PDB entry
    top : obj:`~mdtraj.Topology`
        It has to be the topology that
        results from reading the PDB entry,
        s.t. chain entries are the canonical
        ones, though no checks are carried out
    target_resSeqs : list, default is None
        Only accept chains of `top` that contain
        at least all these residue sequence
        indices, i.e. the 30 of GLU30. Helps
        in cases where there are two
        equivalent chains but one of them
        is missing some key residues


    Returns
    -------
    residxs : list
        The list of residue indexes (zero indexed)
        of :obj:`top` that match the 'Chains' entry
        of the :obj:`PDBentry`

    """

    property = [prop for prop in PDBentry["properties"] if prop["key"] == "Chains"]
    assert len(property) == 1
    chains, residues = property[0]["value"].split("=")

    chain_ids = chains.split("=")[0].split("/")
    for ii, chain in enumerate(top.chains):
        try:
            try_chain_id = chain.chain_id
        except AttributeError:
            try_chain_id = _ascii_uppercase[ii]
        if try_chain_id in chain_ids:
            residx2resSeq = {rr.index: rr.resSeq for rr in chain.residues}
            # Since chain_ids can be duplicated, stop after the first one
            if target_resSeqs is None:
                break
            else:
                # Don't break until all needed residues have been found
                if set(target_resSeqs).issubset(residx2resSeq.values()):
                    break

    # Check that the resSeqs in these segment of top are unique
    assert len(list(residx2resSeq.values())) == len(_np.unique(list(residx2resSeq.values())))

    # Invert the dict since values are unique and can serve as keys
    resSeq2residx = {val: key for key, val in residx2resSeq.items()}

    resSeqs = [int(rr) for rr in residues.split("-")]
    assert len(resSeqs) == 2
    resSeqs = _np.arange(resSeqs[0], resSeqs[1] + 1)
    residxs = [resSeq2residx[ii] for ii in resSeqs if ii in resSeq2residx.keys()]

    # There can be gaps in the crystallized resSeqs, but not int he indices
    assert _np.unique(_np.diff(residxs)) == [1]

    return residxs


class _KDF(_DataFrame):
    r"""
    Sub-class of an :obj:`~pandas.DataFrame` to include KLIFS-associated metadata.

    Check https://pandas.pydata.org/pandas-docs/stable/development/extending.html#define-original-properties
    for more info
    """

    # normal properties
    _metadata = ["UniProtAC", "PDB_id", "PDB_geom"]

    @property
    def _constructor(self):
        return _KDF


class _KLIFSDataFrame(_KDF):
    r"""
    Sub-class of an :obj:`~pandas.DataFrame` to include KLIFS-associated metadata as attributes.

    Simply pass arguments e.g. 'UniProtAC=P31751', PDB_id="3e8d",
    and PDB_geom=geom (an :obj:`~mdtraj.Trajectory`) and then will
    be accessible via self.UniProtAC, self.PDB_id and self.PDB_geom

    Note that no checks are done to see if these arguments are of the expected class.

    Implements its own :obj:`~pandas.DataFrame.to_excel`
    s.t. the attributes self.UniProtAC, self.PDB_id are saved
    into the sheet name, e.g. as "P31751_3e8d" and can be
    recovered upon reading an Excel file from disk.
    self.PDB_geom is also stored as extra sheets in the
    same file, s.t. when reading that Excel File using
    :obj:`_read_excel_as_KDF`, self.PDB_geom is re-instantiated
    as well as a :obj:`~mdtraj.Trajectory`.

    Check https://pandas.pydata.org/pandas-docs/stable/development/extending.html#define-original-properties
    for more info
    """

    def __init__(self, *args, **kwargs):
        argdict = {"UniProtAC": None,
                   "PDB_id": None,
                   "PDB_geom": None}

        for key in argdict.keys():
            val = kwargs.get(key)
            if val is not None:
                argdict[key] = kwargs.pop(key)

        super().__init__(*args, **kwargs)

        for key, val in argdict.items():
            setattr(self, key, val)

    def to_excel(self, excel_writer, **kwargs):
        r""" Like :obj:`~pandas.DataFrame.to_excel` but saves also the PDB topology and coordinates

        Also, the attributes self.UniProtAC and self.PDB_id are saved into
        the first sheet's name, e.g. as "P31751_3e8d" and can be recovered
        upon reading an Excel file from disk.

        The other sheets are called "topology", "bonds", "unitcell", and "xyz"

        If :obj:kwargs contains arguments "sheet_name", "index",
        an Exception will be thrown. """

        with _ExcelWriter(excel_writer) as writer:
            _DataFrame.to_excel(self, writer,
                                index=False,
                                sheet_name="%s_%s" % (self.UniProtAC, self.PDB_id),
                                **kwargs)
            _mdTrajectory2spreadsheets(self.PDB_geom, writer, **kwargs)


def _KLIFS_web_lookup(UniProtAC,
                      KLIFS_API="https://klifs.net/api",
                      timeout=5,
                      verbose=True,
                      url=None):
    r"""
    Lookup the best PDB-match on `KLIFS <https://klifs.net/>`_ for a given UniProt accession code
    and return a :obj:`~pandas.DataFrame` with the PDB-sequence, the residue-pocket nomenclature and
    the PDB-residues indices matching the :obj:`UniprotAC`

    Note that:
        * Lookup will  fail if there's more than one kinase_ID for the UniProtAC
        * The PDB with best KLIFS quality score will be picked
        * Residues belonging to that UniProt are queried on of UniProtKB
          using _UniProtACtoPDBs
        * The returned object is a subclass of :obj:`~pandas.DataFrame`
          with three extra attributes df.UniProtAC, df.PDB_id, df.PDB_geom
          an wrapper around df.write_excel that stores df.UniProtAC and df.PDB_id
          into the spreadsheet's sheet name. Check :obj:`_KLIFSDataFrame` for
          more info.
        *

    Parameters
    ----------
    UniProtAC : string
        UniProt accession code
    KLIFS_API : str, default is "https://klifs.net/api"
        The database API, check https://klifs.net/swagger/ for documentation
    verbose : bool, default is True
        Currently unused
    url : None, default is None
        Just for interface compatibility with _finder_writer
        (watch technical debt be born right here)
    Returns
    -------
    PDB_DF : :obj:`~pandas.DataFrame`
        A subclass of :obj:`~pandas.DataFrame` with
        three extra attributes df.UniProtAC, df.PDB_id, df.PDB_geom
        Will be a ValueError if the lookup was unsuccessful


    """

    url = "%s/kinase_ID?kinase_name=%s" % (KLIFS_API, UniProtAC)

    with _requests.get(url, timeout=timeout) as resp1:
        if resp1.ok:
            ACjson = resp1.json()
            if len(ACjson) > 1:
                print()
                df = _DataFrame(ACjson)
                # df.pop("")
                print(df)
            assert len(ACjson) == 1, ValueError("More than one 'kinase_ID's were found to match %s: %s ",
                                                (UniProtAC, [entry["kinase_ID"] for entry in ACjson]))
            ACjson = ACjson[0]
            if verbose:
                print("done!")
            print("Please cite the following reference to the KLIF structural database:")
            lit = Literature()
            print(_format_cite(lit.site_KLIFS))
            print("For more information, call mdciao.nomenclature.references()")
            kinase_ID = ACjson["kinase_ID"]
            url = "%s/structures_list?kinase_ID=%s" % (KLIFS_API, kinase_ID)
            with _requests.get(url, timeout=timeout) as resp2:
                PDBs = _DataFrame(resp2.json())

                # Sort with quality score
                PDBs.sort_values("quality_score", ascending=False, inplace=True)
                best_PDB, structure_ID = PDBs[["pdb", "structure_ID"]].values[0]
                # print(structure_ID,"structure ID")
                best_PDB = best_PDB.upper()
                # print("best PDB", best_PDB)
                # Get the residues
                url = "%s/interactions_match_residues?structure_ID=%s" % (KLIFS_API, structure_ID)
                with _requests.get(url, timeout=timeout) as resp3:
                    nomencl = _DataFrame(resp3.json())
                    nomencl.rename(columns={"index": "KLIFS_pocket_index",
                                            "KLIFS_position": "KLIFS"}, inplace=True)
                    nomencl.Xray_position = nomencl.Xray_position.astype(int)

                # Get the PDB as DF
                geom = _md_load_rcsb(best_PDB, verbose=False)
                PDB_DF = _mdTopology2residueDF(geom.top)
                # Get the PDB positions from UniProtKB
                PDBs_UPKB = _UniProtACtoPDBs(UniProtAC)
                residue_idxs = _residx_from_UniProtPDBEntry_and_top(PDBs_UPKB[best_PDB], geom.top,
                                                                    target_resSeqs=nomencl.Xray_position.values)
                uniprot_res = _np.full(geom.n_residues, None)
                uniprot_res[residue_idxs] = True
                PDB_DF["UniProtAC_res"] = uniprot_res
                PDB_DF.replace({_np.nan: None}, inplace=True)

                double_condidtion = PDB_DF[
                    (PDB_DF.UniProtAC_res) & (PDB_DF.Sequence_Index.map(lambda x: x in nomencl.Xray_position.values))]
                KLIFS_labels = _np.full(geom.n_residues, None)
                KLIFS_labels[double_condidtion.index.values] = nomencl.KLIFS.values
                PDB_DF["KLIFS"] = KLIFS_labels
                PDB_DF = _KLIFSDataFrame(PDB_DF, UniProtAC=UniProtAC, PDB_id=best_PDB, PDB_geom=geom)
        else:
            PDB_DF = ValueError('url : "%s", uniprot : "%s" error : "%s"' % (url, UniProtAC, resp1.text))

    return PDB_DF


def _KLIFS_finder(UniProtAC,
                  format='KLIFS_%s.xlsx',
                  local_path='.',
                  try_web_lookup=True,
                  verbose=True,
                  dont_fail=False,
                  write_to_disk=False):
    r"""Look up, first locally, then online for the 85-pocket-residues numbering scheme as found in
    the `Kinase–Ligand Interaction Fingerprints and Structure <https://klifs.net/>`_
    return them as a :obj:`~pandas.DataFrame`.

    Please see the relevant references in :obj:`LabelerKLIFS`.

    When reading from disk, the method _read_excel_as_KDF is used,
    which automatically populates attributes DF.UniProtAC, DF.PDB_code
    and DF.PDB_geom.

    Internally, this method wraps around :obj:`_finder_writer`

    Parameters
    ----------
    UniProtAC : str
       UniProt Accession Code, e.g. P31751 or
       filename e.g. 'KLIFS_P31751.xlsx'
    format : str
        A format string that turns the :obj:`identifier`
        into a filename for local lookup, in case the
        user has custom filenames, e.g. KLIFS_P31751.xlsx
    local_path : str
        The local path to the local KLIFS data file
    try_web_lookup : bool, default is True
        If the local lookup fails, go online
    verbose : bool, default is True
        Be verbose
    dont_fail : bool, default is False
        Do not raise any errors that would interrupt
        a workflow and simply return None
    write_to_disk : bool, default is False
        Save the data to disk

    Returns
    -------
    DF : :obj:`~pandas.DataFrame`
        Contains the KLIFS consensus nomenclature, and
        some extra attributes like DF.UniProtAC, DF.PDB_code, DF.PDB_geom
        If the lookup wasn't successful this will be a ValueError
    return_name : str
        The URL or local path to
        the file that was used
    """

    if _path.exists(UniProtAC):
        fullpath = UniProtAC
        try_web_lookup = False
    else:
        xlsxname = format % UniProtAC
        fullpath = _path.join(local_path, xlsxname)
    KLIFS_API = "https://klifs.net/api"
    url = "%s/kinase_ID?kinase_name=%s" % (KLIFS_API, UniProtAC)

    local_lookup_lambda = lambda fullpath: _read_excel_as_KDF(fullpath)

    web_looukup_lambda = lambda url: _KLIFS_web_lookup(UniProtAC, verbose=verbose, url=url, timeout=15)
    return _finder_writer(fullpath, local_lookup_lambda,
                          url, web_looukup_lambda,
                          try_web_lookup=try_web_lookup,
                          verbose=verbose,
                          dont_fail=dont_fail,
                          write_to_disk=write_to_disk)


def _read_excel_as_KDF(fullpath):
    r"""
    Read a KLIFS Excel file and return a :obj:`_KLIFSDataFrame`

    The PDB geom is instantiated from the extra-sheets of the Excel file

    Parameters
    ----------
    fullpath : str
        Path to the Excel file

    Returns
    -------
    df : :obj:`_KLIFSDataFrame`

    """
    idict = _read_excel(fullpath,
                        None,
                        engine="openpyxl")
    assert len(idict) == 5
    keys = list(idict.keys())
    UniProtAC, PDB_id = keys[0].split("_")
    geom = _Spreadsheets2mdTrajectory(idict)
    df = _KLIFSDataFrame(idict[keys[0]].replace({_np.nan: None}),
                         UniProtAC=UniProtAC, PDB_id=PDB_id, PDB_geom=geom)
    return df


class LabelerKLIFS(LabelerConsensus):
    """Obtain and manipulate Kinase-Ligand Interaction notation of the 85 pocket-residues of kinases.

    The residue notation is obtained from the
    `Kinase–Ligand Interaction Fingerprints and Structure database, KLIFS <https://klifs.net/>`_.

    The online lookup logic, implemented by the low-level method :obj:`_KLIFS_web_lookup`, is:

     * Query `KLIFS <https://klifs.net/>`_ with the UniProt accession code and
       get best structure match (highest KLIFS score) and its associated PDB.
     * Query `KLIFS <https://klifs.net/>`_ again for that structure/PDB
       and get their 85 pocket residue indices (in that specific PDB file)
       and their consensus names.
     * Query `UniProtKB <https://www.uniprot.org/>`_ on that PDB for
       the chainID and residue info associated with that UniProt accession code.
     * Query `RCSB PDB <https://rcsb.org/>`_ and get the geometry.
    All the above information is stored in this object and accessible via
    its attributes, check their individual documentation for more info.

    The local lookup logic, implemented by the low-level method :obj:`_KLIFS_finder`, is:

     * Use the :obj:`UniProtAC` directly or in combination with :obj:`format` ="KLIFS_%s.xlsx"
       and :obj:`local_path` to locate a local excel file. That excel file has been
       generated previously by calling :obj:`LabelerKLIFS` with :obj:`write_to_disk=True`
       or by using the :obj:`LabelerKLIFS.dataframe.to_excel` method of an
       already instantiated :obj:`LabelerKLIFS` object. That Excel file will
       contain, apart from the nomenclature, all other attributes, including
       the PDB geometry, needed to re-generate the  :obj:`LabelerKLIFS` locally.
       An example Excel file has been distributed with mdciao and you can find it with:

       >>> import mdciao
       >>> mdciao.examples.filenames.KLIFS_P31751_xlsx

    References
    ----------
    These are the most relevant references on the nomenclature itself,
    but please check `how to cite KLIFS <https://klifs.net/faq.php>`_ in case of doubt:

    * Van Linden, O. P. J., Kooistra, A. J., Leurs, R., De Esch, I. J. P., & De Graaf, C. (2014).
      KLIFS: A knowledge-based structural database to navigate kinase-ligand interaction space.
      Journal of Medicinal Chemistry, 57(2), 249–277. https://doi.org/10.1021/JM400378W
    * Kooistra, A. J., Kanev, G. K., Van Linden, O. P. J., Leurs, R., De Esch, I. J. P., & De Graaf, C. (2016).
      KLIFS: a structural kinase-ligand interaction database.
      Nucleic Acids Research, 44(D1), D365–D371. https://doi.org/10.1093/NAR/GKV1082
    * Kanev, G. K., de Graaf, C., Westerman, B. A., de Esch, I. J. P., & Kooistra, A. J. (2021).
      KLIFS: an overhaul after the first 5 years of supporting kinase research.
      Nucleic Acids Research, 49(D1), D562–D569. https://doi.org/10.1093/NAR/GKAA895

    """

    def __init__(self, UniProtAC,
                 local_path=".",
                 format="KLIFS_%s.xlsx",
                 verbose=True,
                 try_web_lookup=True,
                 write_to_disk=False):

        r"""

        Parameters
        ----------
        UniProtAC : str
            UniProt Accession Code, e.g. P31751
            it gets directly passed to :obj:`_KLIFS_finder`
            Can be anything that can be used to try and find
            the needed information, locally or online:
             * a UniProt Accession Code, e.g. 'P31751'
             * a full local filename, e.g. 'KLIFS_P31751.xlsx'
            Please note the difference between UniProt Accession Code
            and UniProt entry name as explained `here <https://www.uniprot.org/help/difference%5Faccession%5Fentryname>`_.
        local_path : str, default is "."
            Since the :obj:`UniProtAC` is turned into
            a filename in case it's a descriptor,
            this is the local path where to (potentially) look for files.
            In case :obj:`UniProtAC` is just a filename,
            we can turn it into a full path to
            a local file using this parameter, which
            is passed to :obj:`_KLIFS_finder`
            and :obj:`LabelerConsensus`. Note that this
            optional parameter is here for compatibility
            reasons with other methods and might disappear
            in the future.
        format : str, default is "KLIFS_%s.xlsx"
            How to construct a filename out of
            :obj:`UniProtAC`
        verbose : bool, default is True
            Be verbose. Gets passed to :obj:`_KLIFS_finder`
        try_web_lookup : bool, default is True
            Try a web lookup on the KLIFS of the :obj:`UniProtAC`.
            If :obj:`UniProtAC` is e.g. "KLIFS_P31751.xlsx",
            including the extension "xslx", then the lookup will
            fail. This what the :obj:`format` parameter is for
        write_to_disk : bool, default is False
            Save an excel file with the nomenclature
            information
        """

        self._nomenclature_key = "KLIFS"
        self._AAresSeq_key = "AAresSeq"
        self._dataframe, self._tablefile = _KLIFS_finder(UniProtAC,
                                                         format=format,
                                                         local_path=local_path,
                                                         try_web_lookup=try_web_lookup,
                                                         verbose=verbose,
                                                         write_to_disk=write_to_disk
                                                         )

        # TODO this works also for CGN, we could make a method out of this
        self._AA2conlab = {}
        for __, row in self.dataframe[self.dataframe.UniProtAC_res.astype(bool)].iterrows():
            key = "%s%u" % (row.residue, row.Sequence_Index)
            assert key not in self._AA2conlab
            self._AA2conlab[key] = row[self._nomenclature_key]

        self._fragments = _defdict(list)
        for ires, key in self.AA2conlab.items():
            if "." in str(key):
                frag_key = ".".join(key.split(".")[:-1])
                self._fragments[frag_key].append(ires)

        if self._dataframe.PDB_geom is None:
            super().__init__(ref_PDB=self._dataframe.PDB_id,
                             local_path=local_path,
                             try_web_lookup=try_web_lookup,
                             verbose=verbose)
        else:
            super().__init__(ref_PDB=None,
                             local_path=local_path,
                             try_web_lookup=try_web_lookup,
                             verbose=verbose)
            self._geom_PDB = self.dataframe.PDB_geom
            self._ref_PDB = self.dataframe.PDB_id

        # todo unify across Labelers
        self._fragments_as_idxs = {
            fragkey: self.dataframe[self.dataframe.KLIFS.map(
                lambda x: ".".join(str(x).split(".")[:-1]) == fragkey)].index.values.tolist()
            for fragkey in self.fragment_names}

    @property
    def fragments_as_idxs(self):
        r""" Dictionary of fragments keyed with fragment names
        and valued with idxs of the first column of self.dataframe,
        regardless of these residues having a consensus label or not

        Returns
        -------
        fragments_as_idxs : dict
        """

        return self._fragments_as_idxs

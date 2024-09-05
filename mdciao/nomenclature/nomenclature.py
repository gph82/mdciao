##############################################################################
#    This file is part of mdciao.
#    
#    Copyright 2024 Charité Universitätsmedizin Berlin and the Authors
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
from pandas import read_pickle as _read_pickle

import mdciao.fragments as _mdcfrg
import mdciao.utils as _mdcu
from mdciao.utils.str_and_dict import _kwargs_subs

from tempfile import NamedTemporaryFile as _NamedTemporaryFile

from pandas import \
    read_excel as _read_excel, \
    read_csv as _read_csv, \
    DataFrame as _DataFrame, \
    ExcelWriter as _ExcelWriter, \
    ExcelFile as _ExcelFile

from contextlib import nullcontext as _nullcontext

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


def _GPCRdbDataFrame2conlabs(tablefile,
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

    # TODO some overlap here with with _GPCR_web_lookup of GPCRdb_finder
    # figure out best practice to avoid code-repetition
    # This is the most important
    assert scheme in df.keys(), ValueError("'%s' isn't an available scheme.\nAvailable schemes are %s" % (
        scheme, [key for key in df.keys() if key in _GPCR_available_schemes + ["display_generic_number"]]))
    AA2conlab = {key: str(val) for key, val in df[["AAresSeq", scheme]].values}
    # Locate definition lines and use their indices
    fragments = _defdict(list)

    if all([len(conlab.split("."))==3 for conlab in AA2conlab.values()]): #ad-hoc inferring CGN or GPCR
        present_frags = df.display_generic_number.map(lambda x : (".".join(x.split(".")[:-1]))).unique() #ad-hoc for present fragment names
        assert all(["." in fragname for fragname in present_frags])
        name2fullname = {key.split(".")[1] : key for key in present_frags}
        name2fullname.update({val : val for val in name2fullname.values()}) #allows for re-runs
        df.protein_segment = df.protein_segment.map(lambda x : name2fullname[x])
    for key, AArS in df[["protein_segment", "AAresSeq"]].values:
        fragments[key].append(AArS)
    fragments = {key: val for key, val in fragments.items()}

    if not keep_AA_code:
        AA2conlab = {int(key[1:]): val for key, val in AA2conlab.items()}

    if return_fragments:
        return AA2conlab, fragments
    else:
        return AA2conlab

#TODO deprecate?
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
        If the file "local_path/PDB_code".pdb cannot be found locally
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
    found_locally = False
    try:
        return_name = full_local_path
        _DF = local2DF_lambda(full_local_path)
        print("%s found locally." % full_local_path)
        found_locally=True
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
                # web2DF_lambda wraps around _GPCRdb_web_lookup or _KLIFS_web_lookup, both of which
                # have their own printout of "done w/o 404", controlled
                # by the same "verbose" value used to call finder_writer, so we don't need
                # this here (might change if the methods ever become public)
                #if verbose:
                #    print("done without 404, continuing.")
            except Exception as e:
                print('Error getting or processing the web lookup:', e)
                _DF = e

    if isinstance(_DF, _DataFrame):
        if write_to_disk and not found_locally:
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


def _GPCRdb_finder(descriptor,
                   format="%s.xlsx",
                   local_path=".",
                   try_web_lookup=True,
                   verbose=True,
                   dont_fail=False,
                   write_to_disk=False):
    r"""
    Return a :obj:`~pandas.DataFrame` containing
    generic GPCR or CGN generic residue numbering.

    The lookup is first local and then online
    on the `GPCRdb <https://gpcrdb.org/>`

    This method wraps (with some python lambdas) around
    :obj:`_finder_writer`.

    Please see the relevant references in :obj:`LabelerGPCR`.

    Parameters
    ----------
    descriptor : str
        Anything that can be used to find the needed
        information, locally or online:
         * a UniProt name, e.g. 'adrb2_human', 'gnas2_human'
         * a full local filename, e.g. `my_consensus.xlsx` or
          `path/to/my_consensus.xlsx`
         * the "basename" filename, e.g. 'adrb2_human' if
          'adrb2_human.xlsx' exists on `local_path`
          (see below `format`)
        All these ways of doing the same thing (descriptor, basename, fullname,
        localpath, fullpath) are for compatibility with other methods.
    format : str, default is "%s.xlsx", alternative can be "%s.pkl"
        If `descriptor` is not readable directly,
        try to find "descriptor.xlsx" locally on :obj:`local_path`.
        Please note that loading pickled data from untrusted sources can be
        unsafe. See `here <https://docs.python.org/3/library/pickle.html>`_.
    local_path : str, default is "."
        If `descriptor` doesn't find the file locally,
        then try "local_path/descriptor" before trying online
    try_web_lookup : boolean, default is True.
        If local lookup variants fail, go online, else Fail
    verbose : bool, default is False
        Be verbose.
    dont_fail : bool, default is False
        If True, when the lookup fails None will
        be returned. By default, the method raises
        an exception if it could not find the info.
    write_to_disk : boolean, default is False
        Save the found consensus nomenclature info locally.

    Returns
    -------
    df : DataFrame or None
        The consensus nomenclature information as :obj:`~pandas.DataFrame`
    """

    if _path.exists(descriptor):
        fullpath = descriptor
        try_web_lookup = False
    else:
        xlsxname = format % descriptor
        fullpath = _path.join(local_path, xlsxname)
    GPCRdb = "https://gpcrdb.org/services/residues/extended"
    url = "%s/%s" % (GPCRdb, descriptor.lower())

    if fullpath.endswith(".xlsx"):
        local_lookup_lambda = lambda fullpath: _read_excel(fullpath,
                                                           engine="openpyxl",
                                                           usecols=lambda x: x.lower() != "unnamed: 0",
                                                           converters={key: str for key in _GPCR_available_schemes},
                                                           ).replace({_np.nan: None})
    elif fullpath.endswith(".pkl"):
        def local_lookup_lambda(fullpath): # not really a lambda anymore
            idf = _read_pickle(fullpath).replace({_np.nan: None})
            for key in list(idf.keys()):
                if key.lower() == "unnamed: 0":
                    idf.pop(key)
            return idf

    web_lookup_lambda = lambda url: _GPCRdb_web_lookup(url, verbose=verbose)
    return _finder_writer(fullpath, local_lookup_lambda,
                          url, web_lookup_lambda,
                          try_web_lookup=try_web_lookup,
                          verbose=verbose,
                          dont_fail=dont_fail,
                          write_to_disk=write_to_disk)


def _GPCRdb_web_lookup(url, verbose=True,
                       timeout=5):
    r"""
    Lookup this url for a GPCR- or CGN notation
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
    UniProt_name = url.split("/")[-1]
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
                           'but Uniprot name %s yields nothing' % (url, UniProt_name))
    else:
        df = _DataFrame(a.json())
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
        if "BW" not in DFout.keys(): #then we're in CGN territory
            print("Please cite the following reference to the CGN nomenclature:")
            print(_format_cite(lit.scheme_CGN))
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
    """Parent class to manage consensus notations.

    This object should not be invoked by the user directly, it is visible
    here for documentation purposes.

    At the moment child classes are

     * :obj:`LabelerGPCR` for GPCR-notation, this can be:
      * structure based schemes, by Gloriam et al
      * sequence based schemes
       * Class-A: Ballesteros-Weinstein
       * Class-B: Wootten
       * Class-C: Pin
       * Class-F: Wang
     * :obj:`LabelerCGN` for Common-Gprotein-nomenclature (CGN)
     * :obj:`LabelerKLIFS` for Kinase-Ligand Interaction notation of the 85 pocket-residues of kinases

    The consensus labels are abbreviated to 'conlab' throughout.

    """

    def __init__(self, **kwargs):
        self._conlab2AA = {val: key for key, val in self.AA2conlab.items()}

        self._fragment_names = list(self.fragments.keys())
        self._fragments_as_conlabs = {key: [self.AA2conlab[AA] for AA in val]
                                      for key, val in self.fragments.items()}
        self._fragments_as_resSeqs = {key : [_mdcu.residue_and_atom.int_from_AA_code(ival) for ival in val]
                                      for key, val in self.fragments.items()}

        self._idx2conlab = self.dataframe[self._conlab_column].values.tolist()
        self._conlab2idx = {lab: idx for idx, lab in enumerate(self.idx2conlab) if lab is not None}

    @property
    def seq(self):
        r""" The reference sequence in :obj:`dataframe`"""
        return ''.join(
            [_mdcu.residue_and_atom.name_from_AA(val) for val in self.dataframe.AAresSeq.values.squeeze()])

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
                 min_seqID_rate=.5,
                 fragments='resSeq',
                 verbose=False):
        r""" Align a topology with the object's sequence.
        Returns two maps (`top2self`, `self2top`) and
        populates the attribute self.most_recent_alignment

        Wraps around :obj:`mdciao.utils.sequence.align_tops_or_seqs`

        The indices of self are indices (row-indices)
        of the original :obj:`~mdciao.nomenclature.LabelerConsensus.dataframe`,
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
        min_seqID_rate : float, default .5
            With big topologies and many fragments,
            the alignment method (:obj:`mdciao.sequence.my_bioalign`)
            sometimes yields sub-optimal results. A value
            :obj:`min_seqID_rate` >0, e.g. .5 means that a pre-alignment
            takes place to populate :obj:`restrict_to_residxs`
            with indices of those the fragments
            (:obj:`mdciao.fragments.get_fragments` defaults)
            with more than 50% alignment in the pre-alignment.
            If :obj:`min_seqID_rate`>0, :obj`restrict_to_residx`
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
            This will simply use the first alignment that comes out of**
            :obj:`mdciao.utils.sequence.my_bioalign` **, regardless
            of there being other, equally scored, alignments and potential
            clashes with sensitive fragmentations.**
        verbose: boolean, default is False
            be verbose

        Returns
        -------
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

        if (min_seqID_rate > 0):
            assert restrict_to_residxs is None
            restrict_to_residxs = matching_fragments(self.seq,
                                                     top,
                                                     _fragments,
                                                     verbose=verbose or debug,
                                                     min_seqID_rate=min_seqID_rate,
                                                     return_residue_idxs=True, empty=None)

        # In principle I'm introducing this only for KLIFS, could be for all nomenclatures
        if self._conlab_column == "KLIFS":
            chain_id = self.dataframe.chain_id[_np.hstack(list(self.fragments_as_idxs.values()))].unique()
            assert len(chain_id) == 1
            seq_1_res_idxs = self.dataframe[self.dataframe.chain_id == chain_id[0]].index
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
        n_alignments=len(df)
        frags_already_printed = False
        for ii, idf in enumerate(df):
            top2self, self2top = _mdcu.sequence.df2maps(idf)
            topidx2conlab = _np.full([len(top) if isinstance(top,str) else top.n_residues][0], None)
            topidx2conlab[list(top2self.keys())] = self.dataframe.iloc[list(top2self.values())][self._conlab_column]


            idf = _mdcu.sequence.AlignmentDataFrame(idf.merge(_DataFrame(
                {"idx_0": _np.arange(len(topidx2conlab)),
                 "conlab": topidx2conlab}), how="left", on="idx_0"), #.replace(_np.nan, None)
                alignment_score=idf.alignment_score)
            # .replace has ffil problem with pandas < 1.5, not with > 2. allowing
            idf.conlab = [[val if val is not _np.nan else None][0] for val in idf.conlab.values]

            if isinstance(top, str) or str(_frag_str).lower() in ["none", "false"] or n_alignments == 1:
                confrags_compatible_with_frags = True
                if debug:
                    print("I'm not checking fragment compatibility because at least one of these statements is True")
                    print(
                        " * the input topology is a sequence string, s.t. no fragments can be extracted: %s" % isinstance(
                            top, str))
                    print(" * There is only one pairwise alignment with the maximum score: ", n_alignments == 1, n_alignments)
                    print(" * The fragmentation heuristics were False or None: ",
                          str(_frag_str).lower() in ["none", "false"], _frag_str)
                break
            else:
                fragments = _fragments  # This will have been defined already
                confrags_compatible_with_frags = True
                consfrags = self._selfmap2frags(self2top)
                if debug:
                    print("Iteration ", ii)
                    _mdcfrg.print_fragments(consfrags, top)
                alignment_already_printed = False
                for frag_idx, (confraglab, confragidxs) in enumerate(consfrags.items()):
                    confrag_is_subfragment, _, cand_frags = _mdcfrg.check_if_fragment_clashes(confragidxs, confraglab,
                                                                                              fragments, top,
                                                                                              map_conlab=topidx2conlab,
                                                                                              prompt=False)
                    if debug:
                        print(ii, confraglab, confrag_is_subfragment)
                    if not confrag_is_subfragment:
                        if not frags_already_printed:
                            print("Fragments derived from '%s':"%_frag_str)
                            printed_fragments = _mdcfrg.print_fragments(_fragments, top)
                            frags_already_printed = True

                        if not alignment_already_printed:
                            print("* Alignment %2u"% (ii))
                            _mdcu.str_and_dict.print_wrap(f"There are clashes between the above '{_frag_str}' definitions and the consensus fragments' definitions:")
                            alignment_already_printed=True

                        istr = _mdcfrg.print_frag(confraglab, top, confragidxs, resSeq_jumps=True, idx2label=topidx2conlab,
                                                  #fragment_desc="clash:",
                                                  just_return_string=True)
                        print(istr)
                        current_clashing_fragments = "\n".join([_mdcfrg.print_frag(frag_idx, top, fragments[frag_idx],
                                                                                   resSeq_jumps=True,
                                                                                   idx2label=topidx2conlab,
                                                                                   just_return_string=True) for frag_idx in
                                                                cand_frags])
                        print(f"clashes/spreads across these '{_frag_str}' fragments:")
                        print(current_clashing_fragments)

                        if verbose:
                            _mdcu.sequence.print_verbose_dataframe(
                                idf[idf[idf.idx_0.values == confragidxs[0]].index[0] - 1 - 5:
                                    idf[idf.idx_0.values == confragidxs[-1]].index[0] + 2 +5]
                            )


                        confragAAs = [_mdcu.residue_and_atom.shorten_AA(top.residue(idx), keep_index=True) for idx in
                                      confragidxs]
                        if not set(self.fragments[confraglab]).issuperset(confragAAs):
                            confrags_compatible_with_frags = False
                            only_in_top = set(confragAAs).difference(self.fragments[confraglab])
                            current_clashing_confrag = f"=> {confraglab} is not compatible with alignment '{ii}'. "
                            current_clashing_residues = f"The following residues of your input `top` seem to be the problem: {only_in_top}. "
                            istr = current_clashing_confrag+current_clashing_residues

                            if ii<len(df)-1:
                                    istr += f"Moving to the next equally scored alignment" \
                                            f" (score={idf.alignment_score:.2f}) to check if these clashes disappear."
                            elif ii==len(df)-1:
                                istr += f"This was the last available alignment."
                            if not verbose:
                                istr += f" Re-rerun with `verbose=True` to show the problematic part of this alignment here."

                            _mdcu.str_and_dict.print_wrap(istr)
                            break

                        else:
                            istr = f"but all the residues of `top` that have been assigned to {confraglab} using this " \
                                   f"alignment are contained (= same name, same index) in the reference {confraglab}, " \
                                   f"meaning, there's probably just a hole in {confraglab} in your `top`."
                            if not verbose:
                                istr += " Rerun with `verbose=True` to show that part of this alignment here."
                            _mdcu.str_and_dict.print_wrap(istr)
                            print(f"=> {confraglab} is hence considered compatible with alignment '{ii}'.")

                if confrags_compatible_with_frags:
                    if ii>0:
                        print("Picking alignment nr. %u with no apparent breaks."%ii)
                    break
            print()

            assertion_error_str = _mdcu.str_and_dict.print_wrap(
                f"\nNone of the {n_alignments} best pairwise alignments yield consensus fragments " \
                f"compatible with the \'{_frag_str}\' fragmentation-heuristic:\n",
                just_return_string=True)
            assertion_error_str += "\n%s\n"
            assertion_error_str += f"Current clashes are\n{current_clashing_confrag}\n{current_clashing_residues}See above for more details."
            assertion_error_str += "\n" + _mdcu.str_and_dict.print_wrap(
                f"Use `verbose=True` to see the (consensus incompatible) alignments. "
                f"Also, you can try increasing the `min_seqID_rate` or using `restrict_to_residxs` to restrict the alignment " \
                "only to those residues of `top` that most likely belong to the reference sequence as stored in " \
                "`self.seq`. Finally, if you _really_ know what you are doing, set the `fragments` yourself to avoid clashes.",
                just_return_string=True)

        assert confrags_compatible_with_frags, (assertion_error_str % "\n".join(printed_fragments))

        self._last_alignment_df = idf

        return top2self, self2top

    @_kwargs_subs(aligntop)
    def top2labels(self, top,
                   allow_nonmatch=True,
                   autofill_consensus=True,
                   min_seqID_rate=.5,
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
        top : :obj:`~mdtraj.Topology` object or str
            The topology as an object or a path
            to a filename, e.g. a pdb file.
        allow_nonmatch : bool, default is True
            Use consensus labels for non-matching positions
            in case the non-matches have equal lengths
        autofill_consensus : boolean default is False
            Even if there is a consensus mismatch with the sequence of the input
            :obj:`AA2conlab_dict`, try to relabel automagically, s.t.
             * ['G.H5.25', 'G.H5.26', None, 'G.H.28']
            will be relabeled as
             * ['G.H5.25', 'G.H5.26', 'G.H.27', 'G.H.28']
        min_seqID_rate : float, default is .5
            With big topologies and many fragments,
            the alignment method (:obj:`mdciao.sequence.my_bioalign`)
            sometimes yields sub-optimal results. A value
            :obj:`min_seqID_rate` >0, e.g. .5 means that a pre-alignment
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
        if isinstance(top, str):
            top = _md.load(top).top
        self.aligntop(top, min_seqID_rate=min_seqID_rate, **aligntop_kwargs)
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
        `restrict_to_residxs`, `keep_consensus`, and `min_seqID_rate`

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

    # TODO return atoms
    def top2frags(self, top,
                  fragments=None,
                  min_seqID_rate=.5,
                  input_dataframe=None,
                  show_alignment=False,
                  atoms=False,
                  verbose=True,
                  ) -> dict:
        r"""
        Return the subdomains derived from the
        consensus nomenclature and map it out
        in terms of residue indices of the input `top`

        Note
        ----
        This method uses `aligntop` internally,
        see the doc on that method for more info.

        Parameters
        ----------
        top:
            :obj:`~mdtraj.Topology` or path to topology file (e.g. a pdb)
        fragments: iterable of integers, default is None
            Any useful fragment definition as lists of residue indices.
            Useful means:

            * Help with the alignment needed for consensus fragment definition.
              Look at :obj:`LabelerConsensus.aligntop` and its `fragments`
              and `min_seqID_rate` parameters.
            * Check if the newly found, consensus fragment definitions (`defs`)
              clash with the input in `fragments`. Clash* means that
              the `defs` would span over more than
              one of the fragments in defined in `fragments`.

            An interactive prompt will ask the user which fragments to
            keep in case of clashes.

            Check the method :obj:`~mdciao.fragments.check_if_fragment_clashes`
            for more info.
        min_seqID_rate : float, default is .5
            With big topologies, like a receptor-Gprotein system,
            the "brute-force" alignment method
            (check :obj:`mdciao.sequence.my_bioalign`)
            sometimes yields sub-optimal results, e.g.
            finding short snippets of reference sequence
            that align in a completely wrong part of the topology.
            To avoid this, an initial, exploratory alignment
            is carried out. :obj:`min_seqID_rate` = .5 means that
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
        atoms : bool, default is False
            Instead of returning residue indices, return atom indices
        verbose : bool, default is True
            Also print the definitions

        Returns
        -------
        defs : dictionary
            Dictionary with subdomain names as keys
            and arrays of indices (residue or atom) as values
        """

        if isinstance(top, str):
            top = _md.load(top).top

        if input_dataframe is None:
            top2self, self2top = self.aligntop(top, min_seqID_rate=min_seqID_rate, verbose=show_alignment,
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
                _, new_defs[key], _ = _mdcfrg.check_if_fragment_clashes(res_idxs, key, fragments, top, map_conlab=map_conlab)

        for key, res_idxs in new_defs.items():
            defs[key] = res_idxs

        for ii, (key, res_idxs) in enumerate(defs.items()):
            istr = _mdcfrg.print_frag(key, top, res_idxs, fragment_desc='',
                                      idx2label=map_conlab,
                                      just_return_string=True)
            if verbose:
                print(istr)
        if atoms:
            defs = {key : _np.hstack([[aa.index for aa in top.residue(ii).atoms] for ii in frag]) for key, frag in defs.items()}
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

class LabelerGPCRdb(LabelerConsensus):

    def __init__(self, UniProt_name,
                 scheme="display_generic_number",
                 local_path=".",
                 format="%s.xlsx",
                 verbose=True,
                 try_web_lookup=True,
                 # todo write to disk should be moved to the superclass at some point
                 write_to_disk=False):
        r"""

        Parameters
        ----------
        UniProt_name : str
            Descriptor by which to find the generic residue labels,
            it gets directly passed to :obj:`GPCRdb_finder`, which
            operates in the following order:

            * First, try `UniProt_name`, as path to an existing file
              on disk. This is done ignoring the values of
              `local_path` and `format`. Supported formats are
              are spreadsheet ('adrb2_human.xlsx'), or `pandas binary ('adrb2_human.pkl')
              <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_pickle.html>`_.
              Please note that loading pickled data from untrusted sources can be
              unsafe. See `here <https://docs.python.org/3/library/pickle.html>`_.

            * If the above step doesn't find anything locally, then merge the
              `local_path` and the `UniProt_name` using the `format`. E.g.
              if `UniProt_name='adrb2_human'` and `local_path='my_nomenclature_files`
              then the method looks for 'my_nomenclature_files/adrb2_human.xlsx'

            * If none of these files can be found then 'adrb2_human'
              is looked up online in the GPCRdb, unless `try_web_lookup` is False.
            Note
            ----
            Please note the difference between UniProt Accession Code
            and UniProt entry name as explained `here <https://www.uniprot.org/help/difference%5Faccession%5Fentryname>`_.
        scheme : str, default is 'display_generic_number'
            The nomenclature scheme to use. Has no effect
            for G-proteins. It is only important for GPCRs,
            where the default is to use what the GPCRdb
            itself has chosen for this particular
            GPCR. Not all schemes will be
            available for all choices of
            GPCRs. You can
            choose from: 'BW', 'Wootten', 'Pin',
            'Wang', 'Fungal', 'GPCRdb(A)', 'GPCRdb(B)',
            'GPCRdb(C)', 'GPCRdb(F)', 'GPCRdb(D)',
            'Oliveira', 'BS', but not all are guaranteed
            to work
        local_path : str, default is "."
            Since the `UniProt_name` is turned into
            a filename, this is the local path where
            to (potentially) look for files.
            In case `UniProt_name` is just a filename,
            we can turn it into a full path to
            a local file using this parameter, which
            is passed to :obj:`GPCRdb_finder`.
        format : str, default is "%s.xlsx"
            How to construct a filename out of
            `UniProt_name`. Alternative is "%s.pkl"
            for pandas pickle format. Please note
            that loading pickled data from untrusted sources can be
            unsafe. See `here <https://docs.python.org/3/library/pickle.html>`_.
        verbose : bool, default is True
            Be verbose. Gets passed to :obj:`GPCRdb_finder`
        try_web_lookup : bool, default is True
            Try a web lookup on the GPCRdb of the `UniProt_name`.
            If `UniProt_name` is e.g. "adrb2_human.xlsx",
            including the extension "xslx", then the lookup will
            fail. This what the `format` parameter is for
        write_to_disk : bool, default is False
            Save an excel file with the nomenclature
            information
        """

        self._dataframe, self._tablefile = _GPCRdb_finder(UniProt_name,
                                                          format=format,
                                                          local_path=local_path,
                                                          try_web_lookup=try_web_lookup,
                                                          verbose=verbose,
                                                          write_to_disk=write_to_disk
                                                          )
        self._AA2conlab, self._fragments = _GPCRdbDataFrame2conlabs(self.dataframe, scheme=scheme,
                                                                    return_fragments=True)

        self._conlab_column = scheme
        # TODO can we do this using super?
        LabelerConsensus.__init__(self,
                                  local_path=local_path,
                                  try_web_lookup=try_web_lookup,
                                  verbose=verbose)

        self._UniProt_name = UniProt_name

    @property
    def UniProt_name(self):
        return self._UniProt_name

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


class LabelerGPCR(LabelerGPCRdb):
    r"""
    Obtain and manipulate GPCR generic residue numbering
    provided by the `GPCRdb <https://gpcrdb.org/>`_.

    This is based on the awesome GPCRdb REST-API.

    The generic residue labels are called indistinctly "consensus labels"
    or simply "nomenclature" throughout mdciao.

    The `GPCRdb <https://gpcrdb.org/>`_ offers different schemes for GPCR labels:

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
   Not all schemes might work for all methods
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class LabelerCGN(LabelerGPCRdb):
    r"""
    Obtain and manipulate G-protein generic residue numbering, i.e. 'Common Gα Numbering', CGN[1]
    provided by the `GPCRdb <https://gpcrdb.org/>`_.

    This is based on the awesome GPCRdb REST-API.

    The generic residue labels are called indistinctly "consensus labels"
    or simply "nomenclature" throughout mdciao.

    For an overview on CGN, see
    `the original website <https://www.mrc-lmb.cam.ac.uk/CGN/faq.html>`_.

    * Flock et al, (2015) Universal allosteric mechanism for Gα activation by GPCRs
      Nature 2015 524:7564 524, 173--179
      https://doi.org/10.1038/nature14663
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, scheme="display_generic_number", **kwargs)

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
             Recommended option, the most succinct and versatile.
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

        #TODO consider using the "consensus" column directly as index
        self._residxs = _DataFrame([{val: key for key, val in val.items()} for val in self.maps.values()],
                                   index=self.maps.keys())
        if None in self._residxs.keys():
            self._residxs.pop(None)
        self._residxs = self._residxs.T
        self._residxs["consensus"] = self._residxs.index.values
        self._residxs=self._residxs[["consensus"]+[key for key in self._residxs.keys() if key !="consensus"]]

        sorted_keys = _sort_all_consensus_labels(self._residxs["consensus"].values, append_diffset=False)[0]
        assert len(sorted_keys)==len(self._residxs["consensus"]),  (len(sorted_keys), len(self._residxs["consensus"]))
        self._residxs = self._residxs.sort_values("consensus", key=lambda col: col.map(lambda x: sorted_keys.index(x)))
        self._residxs.index = _np.arange(len(self._residxs))

        if self.tops is not None:
            self._AAresSeq, self._CAidxs = self.residxs.copy(), self.residxs.copy()
            self._residxs = self._residxs.astype({key: "Int64" for key in self.keys})
            for key in self.keys:
                not_nulls = self.residxs[key].notnull()
                #TODO check alternative for speedups
                #self._AAresSeq.loc[not_nulls, key]=self.residxs[key][not_nulls].map(lambda ii : _mdcu.residue_and_atom.shorten_AA(self.tops[key].residue(ii),
                #                                                          keep_index=True))
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
            try:
                df = df.map(lambda x: "%u%%" % x)
            except AttributeError:
                # Needed for py37 and py38
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
        The dataframe to be filtered by matching `patterns` and `keys`.
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

    df = df[[filter_on] + keys]
    if patterns is not None:
        matching_keys = _mdcu.str_and_dict.fnmatch_ex(patterns, df[filter_on])
        matches = df[filter_on].map(lambda x: x in matching_keys)
        df = df[matches]

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

    defs = conlabs2confrags(consensus_list)
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
    Choose the best consensus label for a given `idx` in case
    there are more than one consensus(es) at play (e.g. GPCR, CGN, KLIFS).

    Wil raise error if more than one dictionary has a consensus label for
    the same index (unusual case)

    Parameters
    ----------
    idx : int
        index for which the relabeling is needed
    consensus_maps : list
        The items in the list should be "gettable" by using `idx`,
        either by being lists, arrays, or dicts, s.t.,
        the corresponding value should be the label.
    no_key : str
        Output string if there is no label for the
        residue `idx` in any of the dictionaries.
        Mighg be removed in the future, since currently
        all calls to this method use no_key=None,
        since no method uses "NA" anymore
    Returns
    -------
    string: str
        label of the residue idx if present else :obj:`no_key`

    """
    labels = [idict[idx] for idict in consensus_maps]
    good_label = _np.unique([ilab for ilab in labels if str(ilab).lower() != "none"]).tolist()
    assert len(good_label) <= 1, "There can only be one good label, but for residue %u found %s" % (idx, good_label)
    try:
        return good_label[0]
    except IndexError:
        return no_key


def matching_fragments(refseq, top,
                       fragments=None,
                       min_seqID_rate=.6,
                       verbose=False,
                       return_residue_idxs=False,
                       empty=list):
    """Return fragments of the topology that match the reference sequence of :obj:`LabelerConsensus` object

    Matches are defined using `min_seqID_rate` is used as a cutoff
    each segment's alignment to the sequence in `CLin`.

    It only counts the matches for the protein residues of
    `top`, i.e. waters, ions, non-peptidic polymers are
    note taken into account.

    You can use the method to identify the receptor
    in topology where other molecules (e.g. the Gprot)
    are present (or the other way around)

    Parameters
    ----------
    refseq: str or :obj:`LabelerConsensus`
        If not str, the sequence will
        be gotten from :obj:`LabelerConsensus.seq` method
    top:
        :obj:`~mdtraj.Topology` object or string
        containing the sequence.
    fragments : iterable of iterables of idxs, str, or None
        How `top` is split into fragments. If str, use this
        heuristic when calling :obj:`~mdciao.fragments.get_fragments`.
        If None, will be generated using tje default option for
        :obj:`~mdciao.fragments.get_fragments`
    min_seqID_rate: float, default is .6
        Only fragments with sequence identity higher
        than this rate [0,1] will be returned as a guess
    verbose: boolean
        be verbose
    return_residue_idxs : bool, default is False
        Return the list of residue indices directly,
        instead of returning a list of fragment idxs.
    empty : class, list or None, default is list
        What to return in case of an empty guess,
        an empty list or a None

    Returns
    -------
    matches : list
        indices of the fragments (or residues) with higher hit-rate than `min_seqIDhit_rate`


    """

    if fragments is None:
        fragments = _mdcfrg.get_fragments(top, verbose=False)
    elif isinstance(fragments,str):
        fragments = _mdcfrg.fragments.get_fragments(top,method=fragments,verbose=False)

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
    hits, matching_frag_idxs = [], []
    summary = []
    for ii, ifrag in enumerate(fragments):
        hit = _np.intersect1d(ifrag, hit_idxs)
        if len(hit) / len(ifrag) >= min_seqID_rate:
            matching_frag_idxs.append(ii)
        hits.append(hit)
        summary.append([len(hit) / len(ifrag), len(hit), len(ifrag)])

    if verbose:
        print(_DataFrame(summary, columns=["hit rate", "# hits", "len(frg)"],
                         ).to_string(formatters={"hit rate" : "{:,.2f}".format}))

    matching_res_idxs = []
    if len(matching_frag_idxs) > 0:
        matching_res_idxs = _np.hstack([fragments[ii] for ii in matching_frag_idxs])

    if return_residue_idxs:
        matching_frag_idxs = matching_res_idxs

    if empty is None and len(matching_frag_idxs) == 0:
        matching_frag_idxs = None
    return matching_frag_idxs

@_kwargs_subs(matching_fragments, exclude=["fragments", "return_residue_idxs"])
def guess_by_nomenclature(CLin, top, fragments=None, nomenclature_name=None,
                          return_str=False, accept_guess=False,
                          return_residue_idxs=False,
                          **guess_kwargs):
    r"""

    Guess which fragments of a topology best align with a consensus nomenclature.

    Wraps around :obj:`matching_fragments` to interpret its answer.

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
        Accept the guess of :obj:`matching_fragments`
        without asking for confirmation.
    return_residue_idxs : bool, default is False
        Return the list of residue indices directly,
        instead of returning a list of fragment idxs.
        Only has an effect if
    guess_kwargs : dict, optional
        Keyword arguments for some of the keyword arguments
        of :obj:`matching_fragments`, which
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
    guess = matching_fragments(CLin, top, fragments=fragments, **guess_kwargs)
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


def conlabs2confrags(conlabs, splitchar=".", replace_GPCR_frags=False):
    r"""
    Subdomain definitions form a list of consensus labels.

    The indices of the list are interpreted as residue indices
    in the topology used to generate `cons_list`
    in the first place, e.g. by using :obj:`mdciao.nomenclature.LabelerConsensus.top2labels`


    Parameters
    ----------
    conlabs: list
        Consensus labels for a given topology, s.t. indices of
        the list map to residue indices of a given topology, s.t.
        cons_list[10] has the consensus label of top.residue(10)
    splitchar : str, default is "."
        The character to use to get the subdomain labels from the
        consensus labels, e.g. "3" from "3.50" or "G.H5" from "G.H5.1"
        If a label of `cons_list` doesn't have a `splitchar` in
        it, an Exception is thrown (this is a suspicious case)
    replace_GPCR_frags : bool, default is False
        If True, will replace the fragment labels coming from GPCR
        conlabs like "34.50" or "7.49" to "ICL2" or "TM7", respectively.
    Returns
    -------
    defs : dictionary
        dictionary keyed with subdomain-names and valued with arrays of residue indices
    """
    bad_labels = [val for val in conlabs if splitchar not in str(val) and str(val).lower() != "none"]
    if len(bad_labels)>0:
        raise ValueError(f"Some labels of 'cons_list' don't have '{splitchar}' in them. "
                         f"Are you sure these are valid consensus labels(e.g. '3.50' or 'G.H5.26'?:\n{bad_labels}")
    df = _DataFrame(conlabs, columns=["conlab"])
    conlab2confrag = lambda x: str(x)[::-1].split(splitchar, 1)[-1][::-1]
    df["frag"] = df.conlab.map(conlab2confrag)
    consensus_frags = {key: val.index.values for key, val in df.groupby("frag") if str(key).lower() != "none"}
    if replace_GPCR_frags:
        consensus_frags = {_GPCR_num2lett.get(key, key): val for key, val in consensus_frags.items()}

    return {key: _np.array(val) for key, val in consensus_frags.items()}


def _sort_consensus_labels(subset, sorted_superset,
                           append_diffset=True):
    r"""
    Sort consensus labels (GPCR, CGN, KLIFS)

    Parameters
    ----------
    subset : iterable
        list with the names (type str) to be ordered.
        If duplicates are present, they will also appear
        as duplicates in `fragnames_out`
    sorted_superset : iterable
        list with names in the desired order. Is a superset
        of :obj:`subset`
    append_diffset : bool, default is True
        Append the difference set subset-sorted_superset, i.e.
        all the elements in subset that are not in sorted_superset.

    Returns
    -------
    fragnames_out: list
        List with the labels of `subset` sorted according
        to some `sorted_superset` and potentially other
        labels not contained in the `sorted_superset` appended
        at the end.
    """

    by_frags = _defdict(dict)
    for item in subset:
        try:
            frag, idx = item.rsplit(".", maxsplit=1)
            by_frags[frag][idx] = item
        except ValueError:
            frag, idx = item, str(0)
        by_frags[frag][idx] = item
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

    # Recover the duplicates
    order = []
    for lab in labs_out:
        order.extend(_np.flatnonzero(_np.array(subset)==lab))
    labs_out = [subset[oo] for oo in order]
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

def _sort_all_consensus_labels(labels, append_diffset=True, order=["GPCR","CGN","KLIFS"]):
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
    sorted_indices : 1D _np.ndarray
        The indices of `labels` that return
        the sorted `sorted_labels`. Depending
        on `append_diffset` it will contain
        (or not) all indices of `labels`
    """

    lambdas = {"GPCR":  lambda labels: _sort_GPCR_consensus_labels(labels, append_diffset=False),
               "CGN":   lambda labels: _sort_CGN_consensus_labels(labels, append_diffset=False),
               "KLIFS": lambda labels: _sort_KLIFS_consensus_labels(labels, append_diffset=False)}

    sorted_labels = []
    for key in order:
        sorted_labels += lambdas[key](labels)
    if append_diffset:
        sorted_labels += [lab for lab in labels if lab not in sorted_labels]

    # Handle duplicates by including their indices only once (else _np.flatnonzero returns all indices all the time)
    sorted_indices = []
    for lab in sorted_labels:
        for ii in _np.flatnonzero(lab == _np.array(labels)):
            if ii not in sorted_indices:
                sorted_indices.append(ii)
    sorted_indices = _np.array(sorted_indices, ndmin=1)
    assert sorted_indices.ndim==1

    return sorted_labels, sorted_indices

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

_GPCR_fragments = ("NT",
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
                   "CT")

_CGN_fragments = ('G.HN',
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
                  'G.H5')

_KLIFS_fragments = ('I',
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
                    'a.l')


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
         here is that they **might** be used
         to check whether obtained consensus fragments clash
         with these definitions or not. This is done by calling
         :obj:`~mdciao.fragments.check_if_fragment_clashes`, check
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
    "chain_id" : rr.chain.chain_id

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
                       "chain_index": rr.chain.index,
                       "chain_id" : rr.chain.chain_id})
    return _DataFrame(for_DF)


def _mdTrajectory2spreadsheets(traj, dest, **kwargs_to_excel):
    r"""
    Transform the first frame (and only the first) of `traj` into a spreadsheet

    Parameters
    ----------
    traj : obj`:~mdtraj.Trajectory`
    dest : :obj:`~pandas.ExcelWriter` or str
        Either the open ExcelWriter object or,
        if a string is passed, an ExcelWriter
        object will be created on the fly
    kwargs_to_excel : keyword args for :obj:`pandas.DataFrame.to_excel`
        Can't contain sheet_name or index, will raise Exception

    """
    topdf, bonds = traj.top.to_dataframe()
    topdf["chain_id"] = [aa.residue.chain.chain_id for aa in traj.top.atoms] #Maybe open a PR on this at mdtraj?
    bondsdf, xyzdf = _DataFrame(bonds), _DataFrame(traj.xyz[0])
    if traj.unitcell_lengths is not None:
        unitcelldf = _DataFrame({"lengths": traj.unitcell_lengths[0],
                                 "angles": traj.unitcell_angles[0]})
    else:
        unitcelldf = _DataFrame({"lengths": [None, None, None],
                                 "angles": [None, None, None]})

    if isinstance(dest, str):
        cm = _ExcelWriter(dest)
    else:
        cm = _nullcontext(dest)

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
    unitcell = {}
    for key in ["angles", "lengths"]:
        unitcell[key] = idict["unitcell"][key]
        if len(idict["unitcell"][key])==0:
            unitcell[key]=None
    geom = _md.Trajectory(xyz, topology,
                          unitcell_angles=unitcell["angles"],
                          unitcell_lengths=unitcell["lengths"])
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
    _metadata = ["UniProtAC", "PDB_id", "PDB_geom", "kinase_ID", "structure_ID"]

    @property
    def _constructor(self):
        return _KDF


class _KLIFSDataFrame(_KDF):
    r"""
    Sub-class of an :obj:`~pandas.DataFrame` to include KLIFS-associated metadata as attributes.

    Passing named arguments, e.g.
     * kinase_ID=2
     * UniProtAC='P31751'
     * PDB_id="3e8d"
     * structure_ID=1904
     * PDB_geom=geom (an :obj:`~mdtraj.Trajectory`)

    will make them be accessible via self.kinase_ID, self.UniProtAC, self.PDB_id, self.structure_ID, and self.PDB_geom,

    Note that no checks are done to see if these arguments are of the expected class.

    Implements its own :obj:`~pandas.DataFrame.to_excel`
    s.t. the attributes are not lost when writing an Excel file.

    The first sheet's name is kinase_ID_UniProtAC_PDB_id_structure_ID (e.g.
    "2_P31751_3e8d_1904"), s.t. the attributes can be recovered upon reading from disk.
    The attributes are written in that order regardless of how you passed them initially.
    Since self.PDB_geom is also stored as extra sheets in the
    same file, reading that Excel File using :obj:`_read_excel_as_KDF`, re-instantiates
    self.PDB_geom an :obj:`~mdtraj.Trajectory`.

    Check https://pandas.pydata.org/pandas-docs/stable/development/extending.html#define-original-properties
    for more info
    """

    def __init__(self, *args, **kwargs):
        argdict = {
            "kinase_ID": None,
            "UniProtAC": None,
            "PDB_id": None,
            "structure_ID": None,
            "PDB_geom": None
        }

        for key in argdict.keys():
            if key in kwargs.keys():
                argdict[key] = kwargs.pop(key)

        super().__init__(*args, **kwargs)

        for key, val in argdict.items():
            setattr(self, key, val)

    def to_excel(self, excel_writer, save_PDB_geom=True, **kwargs):
        r""" Like :obj:`~pandas.DataFrame.to_excel`, but can save also the PDB topology and coordinates if present.

        Also, the attributes self.kinase_ID, self.UniProtAC, self.PDB_id, self.structure_ID
        are saved into the first sheet's name, e.g. as "2_P31751_3e8d_1904" and can be recovered
        upon reading an Excel file from disk.

        The other sheets are called "topology", "bonds", "unitcell", and "xyz"

        If :obj:kwargs contains arguments "sheet_name", "index",
        an Exception will be thrown. 
        
        Parameters
        ----------
        save_PDB_geom : bool, default is True
            In case self.PDB_geom is not None, by default 
            the method will write the PDB topology and coordinates 
            into extra sheets of the spreadsheets. 
            You can turn this off with this parameter.
        """


        with _ExcelWriter(excel_writer) as writer:
            _DataFrame.to_excel(self, writer,
                                index=False,
                                sheet_name="%s_%s_%s_%s" % (self.kinase_ID, self.UniProtAC, self.PDB_id, self.structure_ID),
                                **kwargs)
            if self.PDB_geom is not None and save_PDB_geom:
                _mdTrajectory2spreadsheets(self.PDB_geom, writer, **kwargs)


def _KLIFS_web_lookup(KLIFS_string,
                      KLIFS_API="https://klifs.net/api",
                      timeout=5,
                      verbose=True,
                      keep_PDB_geom=True) -> _KLIFSDataFrame:
    r"""
    Lookup the best PDB-match on `KLIFS <https://klifs.net/>`_
    and return a :obj:`~pandas.DataFrame` with the PDB-sequence,
    the residue-pocket nomenclature and the PDB-residues
    indices matching the `KLIFS_string`.

    Because the labels are provided on a per-structure (PDB)
    basis, but there's many structures (PDBs) per kinase,
    there's several ways to specify what set of labels
    the user wants.

    To familiarize yourself with KLIFS and ways to get information
    from KLIFS, check https://klifs.net/swagger

    Note that:
        * Lookup will fail if there's KLIFS_string is ambiguous,
          e.g. one UniProt Accession Code yields more than one kinase ID
        * If a kinase ID is provided, the PDB with best KLIFS
          quality score will be picked
        * The returned object is a subclass of :obj:`~pandas.DataFrame`
          with the extra attributes df.kinase_ID, df.UniProtAC,
          df.structure_ID, df.PDB_geom and df.PDB_id. It also has
          a wrapper around df.write_excel that stores that information
          into the spreadsheet's sheet name. Check :obj:`_KLIFSDataFrame` for
          more info.
        * Since KLIFS works with specific structure_IDs, we store the associated
        geometry (only one chain, containing only the kinase) in the self.PDB_geom
        attribute.

    Parameters
    ----------
    KLIFS_string : str
       A string formatted "key:value" which
       ultimately leads to a given KLIFS entry
       Acceptable keys and values for `KLIFS_string` are:
        * "UniProtAC", e.g. "UniProtAC:P31751"
        * "kinase_ID", e.g. "kinase_ID:2"
        * "structure_ID", e.g. "structure_ID:1904"
       Any of the above keys will yield the same `KLIFS_DF`, since
       the UniProtAC can be used to lookup the kinase_ID, and
       the kinase_ID automatically picks the best structure_ID (PDB),
       but the user can specify directly the kinase_ID or the structure_ID.
    KLIFS_API : str, default is "https://klifs.net/api"
        The database API, check https://klifs.net/swagger/ for documentation
    verbose : bool, default is True
        Currently unused
    keep_PDB_geom : bool, default is True
        Do not append the PDB geometry of the kinase
        and other chains associated to the kinase
        in the best PDB-match to the returned DataFrame.

    Returns
    -------
    KLIFS_DF : :obj:`~pandas.DataFrame`
        A subclass of :obj:`~pandas.DataFrame` with
        three extra attributes df.UniProtAC, df.PDB_id, df.PDB_geom
        Will be a ValueError if the lookup was unsuccessful. The
        "Sequence_Index" column refers to the resSeq indices
        of the kinase in the associated PDB_ID, which are the
        same as the ones in the associated structure_ID and
        are the same as the ones used by the object in PDB_DF.PDB_geom.
        They correspond to KLIFS's "Xray_position" in the KLIFS
        https://klifs.net/swagger/#/Interactions/get_interactions_match_residues
        API.

    """
    if ":" not in KLIFS_string or KLIFS_string.split(":")[0] not in ["UniProtAC", "kinase_ID", "structure_ID"]:
        return ValueError("Malformed KLIFS_string, it should be 'key:value', e.g.: 'UniProtAC:P31751', 'kinase_ID:2', or 'structure_ID:1904'")

    lookup_key, lookup_value = KLIFS_string.split(":")

    IDs = {"UniProtAC": None,
           "kinase_ID": None,
           "structure_ID": None}

    # Most upstream is lookup by UniProtAC
    if lookup_key=="UniProtAC":
        IDs["UniProtAC"]=lookup_value
        url = "%s/kinase_ID?kinase_name=%s" % (KLIFS_API, IDs["UniProtAC"])
        #print(url)
        with _requests.get(url, timeout=timeout) as resp1:
            if resp1.ok:
                ACjson = resp1.json()
            else:
                return ValueError('url : "%s", UniProt Accession Code: "%s" error : "%s"' % (url, lookup_value, resp1.text))

        if len(ACjson) > 1:
            raise(ValueError("More than one 'kinase_ID's were found to match %s: %s ",
                             (lookup_value, [entry["kinase_ID"] for entry in ACjson])))

        ACjson = ACjson[0]
        kinase_ID = ACjson["kinase_ID"]

        # We update these to force the next block (lookup by kinase_ID)
        lookup_key, lookup_value = "kinase_ID", kinase_ID

    # We enter here if the UniProtAC yielded a kinase_ID or if the user provided one directly
    if lookup_key=="kinase_ID":
        IDs["kinase_ID"] = int(lookup_value)
        url1 = "%s/structures_list?kinase_ID=%s" % (KLIFS_API, IDs["kinase_ID"])
        print(url1)
        with _requests.get(url1, timeout=timeout) as resp2:
            if not resp2.ok:
                return ValueError(f"'{lookup_value}' doesn't seem like a valid 'kinase_ID'.")
            PDBs = _DataFrame(resp2.json())

        # Sort via quality score
            PDBs.sort_values("quality_score", ascending=False, inplace=True)
            best_PDB, IDs["structure_ID"], chain = PDBs[["pdb", "structure_ID", "chain"]].values[0]
            best_PDB = best_PDB.upper()

    elif lookup_key=="structure_ID":
        IDs["structure_ID"] = int(lookup_value)
        if IDs["kinase_ID"] is None: #means also no UniProtAC
            with _requests.get("https://klifs.net/api/structure_list?structure_ID=%u"%IDs["structure_ID"]) as resp2:
                infos = resp2.json()
                assert len(infos) == 1
                IDs["kinase_ID"], chain, best_PDB = infos[0]["kinase_ID"], infos[0]["chain"], infos[0]['pdb'].upper()

    # Retrive the UniProtAC in case so far we haven't (we can't get this far w/o kinase_ID
    if IDs["UniProtAC"] is None:
        IDs["UniProtAC"] = _KLIFS_kinase_ID2UniProt(IDs["kinase_ID"])
    if verbose:
        print("done!")

    print("Please cite the following reference to the KLIF structural database:")
    lit = Literature()
    print(_format_cite(lit.site_KLIFS))
    print("For more information, call mdciao.nomenclature.references()")

    # Get the residues
    nomencl = _KLIFS_structure_ID2nomenclDF(IDs["structure_ID"], timeout=timeout)

    # Get the geometry
    geom = _KLIFS_structure_ID2Trajectory(IDs["structure_ID"], timeout=timeout)

    # Get the PDB as DF
    PDB_DF = _mdTopology2residueDF(geom.top)
    assert chain in PDB_DF.chain_id.unique()

    # Merge the topology DF and the nomenclDF
    PDB_DF = PDB_DF.merge(nomencl, left_on="Sequence_Index", right_on="Xray_position", how="outer").replace(_np.nan,
                                                                                                            None)
    PDB_DF = PDB_DF[["residue", "code", "Sequence_Index", "AAresSeq", "chain_id", "KLIFS_pocket_index", "KLIFS"]]

    PDB_DF = _KLIFSDataFrame(PDB_DF,
                             kinase_ID=int(IDs["kinase_ID"]), structure_ID=int(IDs["structure_ID"]),
                             UniProtAC= IDs["UniProtAC"], PDB_id=best_PDB,
                             PDB_geom={True: geom,
                                       False: None}[keep_PDB_geom]
                             )

    return PDB_DF

def _KLIFS_kinase_ID2UniProt(kinase_ID, KLIFS_API="https://klifs.net/api", timeout=5, ):
    url2 = "%s/kinase_information?kinase_ID=%s" % (KLIFS_API, kinase_ID)
    with _requests.get(url2, timeout=timeout) as resp2:
        infos = resp2.json()
        assert len(infos) == 1
        return infos[0]["uniprot"]

def _KLIFS_structure_ID2nomenclDF(structure_ID, KLIFS_API="https://klifs.net/api", timeout=5) -> _DataFrame:
    """

    Get the KLIFS nomenclature associated to a KLIFS structure_ID

    https://klifs.net/swagger/#/Interactions/get_interactions_match_residues

    Parameters
    ----------
    structure_ID : int or str
      The KLIFS structure ID
    KLIFS_API : str, default is "https://klifs.net/api"
    timeout : int , default is 5

    Returns
    -------
    nomencl : :obj:`~pandas.DataFrame`
        Columns are:
        * KLIFS_pocket_index (1,85)
        * KLIFS (the GRN labels)
        * Xray_position (the resSeq in this structure ID)
    """
    url = "%s/interactions_match_residues?structure_ID=%s" % (KLIFS_API, structure_ID)
    #print(url)
    with _requests.get(url, timeout=timeout) as resp:
        nomencl = _DataFrame(resp.json())
    nomencl.rename(columns={"index": "KLIFS_pocket_index",
                            "KLIFS_position": "KLIFS"}, inplace=True)
    nomencl = nomencl.replace({"Xray_position": {"_": None}})
    nomencl.Xray_position = nomencl.Xray_position.astype(float)
    return nomencl

def _KLIFS_structure_ID2Trajectory(structure_ID, KLIFS_API="https://klifs.net/api",
                                   timeout=5) -> _md.Trajectory:
    r"""

    Get an :obj:`~mdtraj.Trajectory` from a KLIFS structure_ID via web lookup.

    The trajectory contains the kinase and other chains associated to
    the kinase in the original PDB, according to https://klifs.net/swagger/#/Structures/get_structure_get_pdb_complex
    that means "the full structure (including solvent, cofactors, ligands, etc.) in PDB format"


    Parameters
    ----------
    structure_ID : int or str
        The KLIFS structure ID
    KLIFS_API : str, default is "https://klifs.net/api"
    timeout : int , default is 5

    Returns
    -------
    traj : :obj:`mdtraj.Trajectory`
        Consists of one single chain containing the kinase

    """

    url = "%s/structure_get_pdb_complex?structure_ID=%s" % (KLIFS_API, structure_ID)
    #print(url)
    with _requests.get(url, timeout=timeout) as resp:
        with _NamedTemporaryFile(suffix=".pdb") as f:
            with open(f.name, "w") as f2:
                f2.writelines(resp.text)
            traj = _md.load(f.name)

    return traj

def _KLIFS_finder(KLIFS_string,
                  format='KLIFS_%s.xlsx',
                  local_path='.',
                  try_web_lookup=True,
                  verbose=True,
                  dont_fail=False,
                  write_to_disk=False,
                  keep_PDB_geom=True):
    r"""Look up, first locally, then online for the 85-pocket-residues numbering scheme as found in
    the `Kinase–Ligand Interaction Fingerprints and Structure <https://klifs.net/>`_
    return them as a :obj:`~pandas.DataFrame`.

    Please see the relevant references in :obj:`LabelerKLIFS`.

    When reading from disk, the method _read_excel_as_KDF is used,
    which automatically populates attributes DF.UniProtAC, DF.PDB_code
    and, optionally, DF.PDB_geom.

    Internally, this method wraps around :obj:`_KLIFS_web_lookup` and :obj:`_read_excel_as_KDF`

    Parameters
    ----------
    KLIFS_string : str
        A string with a KLIFS identifier to be processed,
        or a filename for local lookup.
        If string, it has to be formatted "key:value" which
        ultimately leads to a given KLIFS entry
        Acceptable keys and values for `KLIFS_string` are:
         * "UniProtAC", e.g. "UniProtAC:P31751"
         * "kinase_ID", e.g. "kinase_ID:2"
         * "structure_ID", e.g. "structure_ID:1904"
        Any of the above keys will yield the same `KLIFS_DF`, since
        the UniProtAC can be used to lookup the kinase_ID, and
        the kinase_ID automatically picks the best structure_ID (PDB),
        but the user can specify directly the kinase_ID or the structure_ID.
        If filename, anthing pointint to valid file works,
        e.g. 'KLIFS_P31751.xlsx' for local lookup.
    format : str, default is 'KLIFS_%s.xlsx'
        A format string that turns the
        `KLIFS_string` directly into a filename
        for local lookup, in case the
        user has custom filenames, e.g. if
        the `KLIFS_string="P31751"` then this
         format specifier will turn it into
         `KLIFS_P31751.xlsx.`
    local_path : str, default is '.'
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
    keep_PDB_geom : bool, default is True
        If False, don't store the PDB geom in returned DataFrame
        when looking online or locally. For online lookups, the geom will have
        been downloaded and used though, it's
        just not stored as extra sheets in the returned DataFrame,
        making the method faster and lighter when the PDB geoms
        are not really needed. For local lookups, if the local filed
        was stored w/o the geom in the extra sheets and you are reading
        from the with this parameter set to True, you'll get an error.

    Returns
    -------
    DF : :obj:`~pandas.DataFrame`
        Contains the KLIFS consensus nomenclature, and
        some extra attributes like DF.UniProtAC, DF.PDB_code, DF.structure_ID,
        DF.kinase_ID, and optionally DF.PDB_geom.
        If the lookup wasn't successful this will be a ValueError
    return_name : str
        The URL or local path to
        the file that was used
    """

    if _path.exists(KLIFS_string):
        fullpath = KLIFS_string
        try_web_lookup = False
    else:
        xlsxname = format % KLIFS_string
        fullpath = _path.join(local_path, xlsxname)
    KLIFS_API = "https://klifs.net/api"
    url = "%s/kinase_ID?kinase_name=%s" % (KLIFS_API, KLIFS_string)

    local_lookup_lambda = lambda fullpath: _read_excel_as_KDF(fullpath, keep_PDB_geom=keep_PDB_geom)

    web_looukup_lambda = lambda url: _KLIFS_web_lookup(KLIFS_string, verbose=verbose, timeout=15, keep_PDB_geom=keep_PDB_geom)
    return _finder_writer(fullpath, local_lookup_lambda,
                          url, web_looukup_lambda,
                          try_web_lookup=try_web_lookup,
                          verbose=verbose,
                          dont_fail=dont_fail,
                          write_to_disk=write_to_disk)


def _read_excel_as_KDF(fullpath, keep_PDB_geom=True):
    r"""
    Instantiate a :obj:`_KLIFSDataFrame` from an :obj:`_KLIFSDataFrame` Excel

    The PDB geom is instantiated from the extra-sheets of the Excel file

    Parameters
    ----------
    fullpath : str
        Path to the Excel file
    keep_PDB_geom : bool, default is True
        If False, don't read the PDB geometry from the
        extra-sheets of the Excel file. The PDB_id will
        still be stored in the returned DataFrame.PDB_id attribute.
        This makes the method more faster and lighter
        when the PDB geoms are not really needed. If `read_PDB_geom`
        is True but the `fullpath` doesn't contain the necessary
        extra sheets to instantiate the PDB geometry, the
        method will fail. (This forces the user to be aware
        of which nomenclature files have been stored w/ and w/o
        the PDB geoms).
    Returns
    -------
    df : :obj:`_KLIFSDataFrame`

    """
    if keep_PDB_geom:
        idict = _read_excel(fullpath,
                            None,
                            engine="openpyxl")
        if len(idict) < 5:
            raise ValueError(f"Not enough sheets in {fullpath} to instantiate a PDB geometry.\n"
                             "Re-run with 'keep_PDB_geom=False' or re-generate the file with\n"
                             "the 'keep_PDB_geom=True' option.")
        geom = _Spreadsheets2mdTrajectory(idict)
        keys = list(idict.keys())
        kinase_ID, UniProtAC, PDB_id, structure_ID = keys[0].split("_")
        df = _KLIFSDataFrame(idict[keys[0]].replace({_np.nan: None}),
                             UniProtAC=UniProtAC,
                             PDB_id=PDB_id, PDB_geom=geom,
                             kinase_ID=int(kinase_ID), structure_ID=int(structure_ID))
    else:
        idict = _read_excel(fullpath,
                            0,
                            engine="openpyxl")
        with _ExcelFile(fullpath) as f:
            key = f.sheet_names[0]
        idict = {key : idict}
        kinase_ID, UniProtAC, PDB_id, structure_ID = key.split("_")
        df = _KLIFSDataFrame(idict[key].replace({_np.nan: None}),
                             UniProtAC=UniProtAC,
                             PDB_id=PDB_id, PDB_geom=None,
                             kinase_ID=int(kinase_ID), structure_ID=int(structure_ID))

    return df


class LabelerKLIFS(LabelerConsensus):
    """Obtain and manipulate Kinase-Ligand Interaction notation of the 85 pocket-residues of kinases.

    The residue notation is obtained from the
    `Kinase–Ligand Interaction Fingerprints and Structure database, KLIFS <https://klifs.net/>`_.

    Since the KLIFS database serves residue labels associated with specific PDBs,
    and there is more than one PDB per kinase, the lookup logic, implemented
    by the low-level method :obj:`_KLIFS_web_lookup`, allows for some flexibility
    in the input:
     * Query via a UniProt Accession Code, which yields a kinase ID (which are
       internal to KLIFS), which has quality-scored PDBs associated to it
       and then get labels from the highest-scored PDB
       (which has a unique structure ID internal to KLIFS).
       Please note the difference between UniProt Accession Code
       and UniProt entry name as explained `here <https://www.uniprot.org/help/difference%5Faccession%5Fentryname>`_.
     * Skip the first step and query directly via a KLIFS kinase ID. From
       that kinase ID, follow the same logic as above to get the associated
       highest-scored PDB (and its KLIFS structure ID) and then get the labels.
     * Skip the first and second step and query directly via KLIFS structure ID,
       and get the labels from the PDB associted to it, regardless of its score.
    Please see the docstring on `KLIFS_string` on how to choose between the
    above strategies.

    Once the PDB and structure ID have been determined, then this method also

     * gets the 85 pocket residue indices (in that specific PDB file)
       and their consensus names.
     * gets a geometry containing the kinase and other chains associated to
       the kinase in the original PDB, according to `the docs <https://klifs.net/swagger/#/Structures/get_structure_get_pdb_complex>`_ ,
       that means "the full structure (including solvent, cofactors, ligands, etc.) in PDB format"

    All the above information is stored in this object and accessible via
    its attributes, check their individual documentation for more info.

    The local lookup logic, implemented by the low-level method :obj:`_KLIFS_finder`, is:

     * Use the `KLIFS_string` directly or in combination with `format`="KLIFS_%s.xlsx"
       and `local_path` to locate a local excel file. That excel file has been
       generated previously by calling `LabelerKLIFS` with `write_to_disk=True`
       or by using the `LabelerKLIFS.dataframe.to_excel` method of an
       already instantiated `LabelerKLIFS` object. That Excel file will
       contain, apart from the nomenclature, all other attributes, including
       the PDB geometry, needed to re-generate the  `LabelerKLIFS` locally.
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

    def __init__(self, KLIFS_string,
                 local_path=".",
                 format="KLIFS_%s.xlsx",
                 verbose=True,
                 try_web_lookup=True,
                 write_to_disk=False,
                 keep_PDB_geom=True):

        r"""

        Parameters
        ----------
        KLIFS_string : str
            A string with a KLIFS identifier to be processed,
            or a filename for local lookup.
            If string, it has to be formatted "key:value" which
            ultimately leads to a given KLIFS entry (see above)
            Acceptable keys and values for `KLIFS_string` are:

            * "UniProtAC", e.g. "UniProtAC:P31751"
            * "kinase_ID", e.g. "kinase_ID:2"
            * "structure_ID", e.g. "structure_ID:1904"

            Any of the above keys will yield the same labels, since
            the UniProtAC can be used to look up the kinase_ID, and
            the kinase_ID automatically picks the best structure_ID (PDB),
            but the user can choose to specify directly the kinase_ID or the structure_ID.

            If a local file is to be used instead of online lookup,
            anything pointing to a valid file works,
            e.g. 'KLIFS_P31751.xlsx'.

            Finally, when the above fails, it will try to construct
            a file by combining `KLIFS_string` with the `format`
            and `local_path` arguments

        local_path : str, default is "."
            Since the `KLIFS_string` can be a filename
            (or turned into one, see above), this is the local path
            where to (potentially) look for files.
            Note that this optional parameter
            is here for compatibility
            reasons with other methods and might disappear
            in the future.
        format : str, default is 'KLIFS_%s.xlsx'
            A format string that turns the
            `KLIFS_string` directly into a filename
            for local lookup, in case the
            user has custom filenames, e.g. if
            the `KLIFS_string="P31751"` then this
            format specifier will turn it into
            `KLIFS_P31751.xlsx.`
        verbose : bool, default is True
            Be verbose. Gets passed to :obj:`_KLIFS_finder`
        try_web_lookup : bool, default is True
            Try a web lookup on the KLIFS via `KLIFS_string`.
            If `KLIFS_string` is e.g. "KLIFS_P31751.xlsx",
            including the extension "xslx", then the lookup will
            fail. This what the `format` parameter is for
        write_to_disk : bool, default is False
            Save an excel file with the nomenclature
            information.
        keep_PDB_geom : bool, default is True
            If False, don't store the PDB geom in returned DataFrame
            when looking online or locally. For online lookups, the
            geom will have been downloaded and used though, it's
            just not stored as extra sheets in the returned DataFrame,
            making the method faster and lighter when the PDB geoms
            are not really needed. For local lookups, if the local file
            was stored w/o the geom in the extra sheets and you are reading
            from the file with this parameter set to True, you'll get an error.
        """

        self._conlab_column = "KLIFS"
        self._dataframe, self._tablefile = _KLIFS_finder(KLIFS_string,
                                                         format=format,
                                                         local_path=local_path,
                                                         try_web_lookup=try_web_lookup,
                                                         verbose=verbose,
                                                         write_to_disk=write_to_disk,
                                                         keep_PDB_geom=keep_PDB_geom

                                                         )

        # TODO this works also for CGN, we could make a method out of this
        self._AA2conlab = {}
        for __, row in self.dataframe[self.dataframe.Sequence_Index.astype(bool)].iterrows():
            key = "%s%u" % (row.residue, row.Sequence_Index)
            assert key not in self._AA2conlab
            self._AA2conlab[key] = row[self._conlab_column]

        self._fragments = _defdict(list)
        for ires, key in self.AA2conlab.items():
            if "." in str(key):
                frag_key = ".".join(key.split(".")[:-1])
                self._fragments[frag_key].append(ires)

        if self._dataframe.PDB_geom is None:
            super().__init__(local_path=local_path,
                             try_web_lookup=try_web_lookup,
                             verbose=verbose)
        else:
            super().__init__(local_path=local_path,
                             try_web_lookup=try_web_lookup,
                             verbose=verbose)

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

def _lexsort_consensus_ctc_labels(labels, reverse = False, columns = [0, 1], sep = "-") -> tuple:
    r"""
    Sort contact-labels in ascending order of resSeq using both columns

    Wraps around :obj:`_sort_all_consensus_labels` with some string handling.

    It will also work with contact-labels consisting of only one residue,
    e.g. in the cases where the "anchor" has been deleted or the frequencies
    have been aggregated to per-residue frequencies

    >>> labels = ['3.50-G.H5.23',
    >>>           '3.50-7.53',
    >>>           '3.50-2.39',
    >>>           '4.50-6.60',
    >>>           '3.50-5.58']
    >>> sorted_labels, order = _lexsort_consensus_ctc_labels(labels)
    >>> sorted_labels
    >>> labels = ['3.50-2.39',
    >>>           '3.50-5.58',
    >>>           '3.50-7.53',
    >>>           '3.50-G.H5.23',
    >>>           '4.50-6.60']


    Parameters
    ----------
    labels : list or np.ndarray
        Strings describing the contact
        residues using consensus labels only.
        Labels can be just one residue "3.50" or
        both "3.50-2.50", but not 'mixed', as in
        >>> labels = ["3.50", "3.50-2.50"]
        Full labels, e.g. "GLU30@3.50", or non-consensus
        labels, e.g. "frag1", will be sorted last.
    reverse : bool, default is False
        If True, sort in descending
        order, instead of ascending
    columns : list
        The order of the columns,
        e.g. [0,1] means sort first
        by first column (idx 0),
        then by second column (idx 1).
    sep : char, default is "-"
        The character to use
        when separating the
        contact label into both residues

    Returns
    -------
    sorted_labels : list
        The sorted contact labels
    order : 1D np.ndarray
        The indices of `ctc_labels` that
        sort it into `sorted_labels`
    """
    split_labels = [_mdcu.str_and_dict.splitlabel(lab, sep=sep) for lab in labels]

    if not any([all([len(lab) == 1 for lab in split_labels]),
                all([len(lab) == 2 for lab in split_labels])]):
        raise ValueError(f"Labels have to be all single ('3.50') or double ('3.50-2.50'), but not mixed {labels}")
    split_labels = _np.vstack(split_labels).squeeze()
    if split_labels.ndim == 1:
        order = _sort_all_consensus_labels(split_labels)[1]
    elif split_labels.ndim ==2:
        #There's other ways but this was easy to set up
        lexsort = {key : val[columns[1]].to_dict() for key, val in _DataFrame(split_labels).groupby(by=columns[0])}
        order = []
        for key1 in _sort_all_consensus_labels(list(lexsort.keys()))[0]:
            for key2 in _sort_all_consensus_labels(list(lexsort[key1].values()))[0]:
                order.extend(_np.flatnonzero(_np.array(labels) == sep.join(_np.array([key1, key2])[columns])))
    else:
        raise ValueError

    if reverse:
        order = order[::-1]
    return [labels[ii] for ii in order], order

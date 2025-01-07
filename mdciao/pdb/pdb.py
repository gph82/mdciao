##############################################################################
#    This file is part of mdciao.
#    
#    Copyright 2025 Charité Universitätsmedizin Berlin and the Authors
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

import requests as _requests

from mdtraj import load_pdb as _mdloadpdb
from Bio import PDB as _PDB
from tempfile import NamedTemporaryFile as _NamedTemporaryFile
from shutil import copy as _copy, copyfileobj as _copyfileobj
try:
    from shutil import COPY_BUFSIZE as _COPY_BUFSIZE
except ImportError:
    _COPY_BUFSIZE =  1024 * 64 #py37 shutil has this buffersize hard-coded 256 * 64, i'm setting 1024 * 64 across the board

from urllib.error import HTTPError as _HTTPError, URLError as _URLError
from urllib.request import urlopen as _urlopen

def _url2json(url,timeout,verbose):
    r""" Wraps around :obj:`requests.get` and tries to do :obj:`requests.Response.json`

    Parameters
    ----------
    url : str
    timeout : int
        seconds
    verbose : bool

    Returns
    -------
    dict

    """
    if verbose:
        print("Calling %s..." % url, end="")
    a = _requests.get(url, timeout=timeout)
    if verbose:
        print("done!")
    try:
        json = a.json()
    except:
        json = ValueError('Could not create a json out of  %s\n'
                          'Please check (e.g. via browser) if that is a valid url'%url)
    return json

def pdb2ref(pdb, url="https://data.rcsb.org/rest/v1/core/entry/",
            timeout=5):
    r"""
    Print the primary citation of a pdb code via web lookup

    The citation object is returned as a dict,
    check these links for more info
     * https://data.rcsb.org/index.html#data-api
     * https://data.rcsb.org/data-attributes.html

    Parameters
    ----------
    pdb : str
        four-leter pdb-code
    url : str
        the base URL for the look-ups, default is 'https://data.rcsb.org/rest/v1/core/entry/'
    timeout : int
        passed to the :obj:`requests.get` API,
        'How many seconds to wait for the server to send data
        before giving up'

    Returns
    -------
    ref : dict
        Whatever is contained in "rcsb_primary_citation"
        of https://data.rcsb.org/data-attributes.html
    """
    url = url.strip("/")+"/" + pdb.strip("/")
    ref = _url2json(url,timeout,verbose=False)
    try:
        ref = ref["rcsb_primary_citation"]
    except KeyError:
        try:
            print(url,":", ref["message"])
        except KeyError:
            print(url, ":", "no primary citation")
        return

    print("Please cite the following 3rd party publication:")
    print( " * "+ref["title"])
    line=f'   {ref["rcsb_authors"][0]} et al., {ref["rcsb_journal_abbrev"]}'
    try:
        line += f' {ref["year"]}'
    except KeyError:
        assert ref["rcsb_journal_abbrev"].lower() in ["To be published".lower()]
    print(line)
    try:
        print("   https://doi.org/"+ref["pdbx_database_id_doi"])
    except KeyError:
        pass #what would be a good control here?

    return ref

def pdb2traj(code,
             filename=None,
             verbose=True,
             url="https://files.rcsb.org/download/",
             cif_first=False
             ):
    r""" Return a :obj:`~mdtraj.Trajectory` from a four-letter PDB code via RCSB PDB lookup

    Prints the corresponding citation.

    Will look up .pdb and .cif files (in that order) and return None if the lookup fails.

    Note
    ----
    Since mdtraj does not read mmCIFs natively (uses openmm to read them),
    mdciao opts to stay inside the existing dependencies and use BioPython's
    `PDB.MMCIFParser <https://biopython.org/docs/dev/api/Bio.PDB.MMCIFParser.html>`_ . and
    `PDB.MMCIF2Dict <https://biopython.org/docs/dev/api/Bio.PDB.MMCIF2Dict.html>`_ modules
    to read .cif-files. The conversion to .pdb is done by writing to a temporary .pdb-file
    and reading that file with mdtraj. In particular, mdtraj uses alternative location
    "A" regardless of occupancy factors, s.t. mdciao enforces this behavior in Biopython
    when saving to the intermediate .pdb-file.

    Please note that there is no single 'canonical' way to convert a PDBx/mmCIF file to a pdb.
    (RCSB lists a number of tools `here <http://mmcif.rcsb.org/docs/software-resources.html>`_)
    and both file formats have different philosophies and format specifications, which can
    affect things like chain definition and residue- and atom-order.


    Parameters
    ----------
    code : str
        four-letter code, e.g. 3SN6
    filename : str, or True, default is None
        * If str, save to this filename. The
          filename's extension can be used to
          convert into another format,
          i.e. the PDB ID might be accessible
          only as a CIF-file, but can be converted
          to .pdb on the fly.
        * If True, keep the original RCSB filename.

        Files are overwritten without checking
        if file exists.
    verbose : bool, default is False
    url : str, default is 'https://files.rcsb.org/download'
        base URL for lookups
    cif_first : boolean, default is False
        Try to get the .cif-file first instead of the .pdb-file
    Returns
    -------
    traj : :obj:`~mdtraj.Trajectory` or None

    """

    #TODO use print_v elsewhere
    print_v = lambda s, **kwargs: [print(s, **kwargs) if verbose else None][0]
    for ext in ["pdb","cif"][::{True:-1,
                                False:+1}[cif_first]]:
        geom = None
        url1 = f"{url.strip('/')}/{code}.{ext}"
        print_v("Checking %s" % url1, end=" ...", flush=True)
        # From https://docs.python.org/3/howto/urllib2.html#fetching-urls
        try:
            with _urlopen(url1) as response:
                if response.status == 200:
                    with _NamedTemporaryFile(delete=True, suffix=f".{ext}") as tmp_file:
                        _copyfileobj(response, tmp_file, length=_COPY_BUFSIZE*2)
                        if ext=="pdb":
                            geom = _mdloadpdb(tmp_file.name)
                        elif ext=="cif":
                            structure = _PDB.MMCIFParser().get_structure(tmp_file.name, tmp_file.name)
                            cif_dict = _PDB.MMCIF2Dict.MMCIF2Dict(tmp_file.name)
                            geom = _BIOStructure2MDTrajectory(structure,cif_dict=cif_dict)
                        if filename is not None:
                            if isinstance(filename, bool) and filename:
                                filename = f"{code}.{ext}"
                            print_v("Saving to %s..." % filename, end="", flush=True)
                            if filename.lower().endswith(f".{ext}"): #simply copy/rename dowloaded file
                                _copy(tmp_file.name, filename)
                            else: # convert to other formats
                                    geom.save(filename)
                            print_v(filename)
                else:
                    raise _URLError(response.reason,filename=url1)


        except (_HTTPError, _URLError) as e:
            print(url1, ":", e)

        if geom is not None:
            pdb2ref(code)
            break

    return geom

def _BIOStructure2pdbfile(structure, filename, cif_dict=None, disordered_select_A=False):
    r"""
    Wraps around BioPython's PDB writer, doing two things

    1. Adds a header with crystallographic information
    2. Truncates residue names to 3 characters only
    3. Set location "A" as default alternative locations for disordered atoms (optional)

    2. is because there are some padding/formatting issues with long
    residue names and high b-factor values, check. e.g. 8E0G.cif

    3. is because mdtraj chooses alternative location "A" regardless of
    the occupancy, s.t. reading/writing would lead to inconsistencies unless
    this flag is set to True.

    Parameters
    ----------
    structure : :obj:`Bio.PDB.Structure.Structure`
    filename : str
        Will overwrite existing `filename`
    cif_dict : dict, default is None
        Whatever :obj:`PDB.MMCIF2Dict.MMCIF2Dict`
        yielded on the same filename used
        to instantiate `structure`
    disordered_select_A : bool, default is True
        Force alternative location "A" for disordered
        atoms regardless of occupancy factor. Forces
        compatibility with mdtraj, which always
        keeps "A" regardless of occupancy.
    """

    _structure = structure.copy()
    [setattr(residue, "resname", residue.resname[:3]) for residue in _structure.get_residues()]
    if disordered_select_A:
        [atom.disordered_select("A") for atom in _structure.get_atoms() if atom.is_disordered()]
    if cif_dict is not None:
        cell_parameters = {
            "a": float(cif_dict.get("_cell.length_a", 0.0)[0]),
            "b": float(cif_dict.get("_cell.length_b", 0.0)[0]),
            "c": float(cif_dict.get("_cell.length_c", 0.0)[0]),
            "alpha": float(cif_dict.get("_cell.angle_alpha", 0.0)[0]),
            "beta": float(cif_dict.get("_cell.angle_beta", 0.0)[0]),
            "gamma": float(cif_dict.get("_cell.angle_gamma", 0.0)[0]),
            "space_group": cif_dict.get("_symmetry.space_group_name_H-M", "Unknown")[0],
            "z_value": cif_dict.get("_cell.Z_PDB", 0)[0]
        }
        cryst1_line = (
            f"CRYST1{cell_parameters['a']:9.3f}{cell_parameters['b']:9.3f}{cell_parameters['c']:9.3f}"
            f"{cell_parameters['alpha']:7.2f}{cell_parameters['beta']:7.2f}{cell_parameters['gamma']:7.2f} {cell_parameters['space_group']}{cell_parameters['z_value']:4}"
        )

    with open(filename, "w") as f:
        if cif_dict is not None:
            f.write(cryst1_line+"\n")
        remark_line = f"REMARK   1 CREATED USING Bio.PDB.PDBIO from {_structure.id}. CRYST1 record created via Bio.PDB.MMCIF2Dict.MMCIF2Dict and mdciao"
        f.write("REMARK   1\n"+remark_line+"\n")
        io = _PDB.PDBIO(use_model_flag=1)
        io.set_structure(_structure)
        io.save(f)

def _BIOStructure2MDTrajectory(structure, cif_dict=None):
    r"""
    Uses _BIOStructure2pdbfile to write a temporary pdb and read it with mdtraj

    The writing is done choosing location "A" for disordered atoms by default, b.c.
    this is what mdtraj would keep from the .pdb regardless of occupancy factors.

    Parameters
    ----------
    structure : :obj:`Bio.PDB.Structure.Structure`
    cif_dict : dict, default is None
        Whatever :obj:`PDB.MMCIF2Dict.MMCIF2Dict`
        yielded on the same filename used
        to instantiate `structure`
    """
    with _NamedTemporaryFile(delete=True, suffix=".pdb") as tmp_file:
        _BIOStructure2pdbfile(structure, tmp_file.name, cif_dict=cif_dict, disordered_select_A=True)
        return _mdloadpdb(tmp_file.name)

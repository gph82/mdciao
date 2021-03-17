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

from json import load as _jsonload, JSONDecodeError as _JSONDecodeError
from os.path import splitext as _psplitext, split as _psplit
import numpy as _np
from copy import deepcopy as _dcopy

import mdciao.fragments as _mdcfrg
import mdciao.utils as _mdcu

def x2site(site, fmt="AAresSeq"):
    """
    Return a site dictionary from a dict or an ascii file

    Parameters
    ----------
    site : str or dict
        Path to the ascii file. If the file isn't
        a json file, or doesn't contain the 'name'-field,
        the filename itself will be used as 'name' in the output
        If a dict is passed, it's checked that the dictionary
        has the needed keys to function as a site.
        See :obj:`mdciao.sites` for more info on how sites
        are defined.
    fmt: str, default is "AAresSeq"
        The expected format in case of reading a file.
        Only used when reading a file.

    Returns
    -------
    site : dict
        Keys are:
         * bonds
         * nbonds
         * name
        site["bonds"] is itself a dictionary
        valued with a list of pairs,
        e.g ["L394","K270"] or [[10,20],[300-4]]],
        depending on the type of bond specified

    """
    if isinstance(site, dict):
        idict = _dcopy(site)
    else:
        try:
            with open(site, "r") as f:
                idict = _jsonload(f)
        except _JSONDecodeError as e:
            idict = dat2site(site,fmt=fmt)

    try:
        bondtype = list(idict["bonds"].keys())
        assert len(bondtype)==1 and bondtype[0] in ["AAresSeq","residx"]
        bondtype = bondtype[0]
        bonds = idict["bonds"][bondtype]
        idict["n_bonds"] = len(bonds)
        if isinstance(bonds[0][0],str):  # can only be str spearated by "-"
            idict["bonds"][bondtype] = [item.split("-") for item in bonds if item[0] != '#' and "-" in item]
            if bondtype=="residx":
                idict["bonds"][bondtype] = [[int(pp) for pp in pair] for pair in  idict["bonds"][bondtype]]
        else:
            assert all([len(bond)==2 for bond in bonds]),bonds
    except KeyError:
        print("Malformed file for the site %s:\n%s" % (site,idict))
        raise

    if "name" not in idict.keys():
        if isinstance(site,str):
            idict["name"] = _psplitext(_psplit(site)[-1])[0]
        else:
            raise ValueError("A 'name'-key is mandatory when passing a site dictionary")

    return idict

def sites_to_res_pairs(site_dicts, top,
                       fragments=None,
                       **get_fragments_kwargs,
                       ):
    r"""Return the pairs of res_idxs needed to compute
    all the contacts contained in all the input sites.

    The idea is to join all needed pairs of res_idxs
    in one list regardless of where they come from.

    Parameters
    ----------
    site_dicts : list of dicts
        Check :obj:`x2site` for how these dicts look like
    top : :obj:`~mdtraj.Topology`
    fragments : list, default is None
        You can pass along fragment definitions so that
        it's easier to de-duplicate any AA in your input.
        Otherwise, these will be created on-the-fly
        by :obj:`mdciao.fragments.get_fragments`
    get_fragments_kwargs :
        see :obj:`fragments.get_fragments`

    Returns
    -------
    res_idxs_pairs : 2D np.ndarray
        Unique pairs contained in the :obj:`site_dicts`,
        expressed as residue indices of :obj:`top`
        [0,1] is considered != [0,1]
    site_maps : list
        For each site, a list with the indices of :obj:`res_idxs_pairs`
        that match the site's bonds.
    """
    if fragments is None:
        fragments = _mdcfrg.get_fragments(top, **get_fragments_kwargs)

    get_pair_lambda = {"AAresSeq":lambda bond :_mdcu.residue_and_atom.residues_from_descriptors(bond, fragments, top)[0],
                       "residx" : lambda bond : bond}
    res_idxs_pairs = []
    pair2idx = {}
    site_maps = []
    for ii, site in enumerate(site_dicts):
        imap=[]
        for bond_type, bonds in site["bonds"].items():
            for bond in bonds:
                pair = tuple(get_pair_lambda[bond_type](bond))
                if pair not in res_idxs_pairs:
                    res_idxs_pairs.append(pair)
                    pair2idx[pair]=len(res_idxs_pairs)-1
                imap.append(pair2idx[pair])
        site_maps.append(imap)
    #print(site_maps)
    return _np.vstack(res_idxs_pairs), site_maps

def site2str(site):
    r""" Produce a printable str for sitefile (json) or site-dict"""
    if isinstance(site,str):
        return site
    elif isinstance(site,dict):
        assert "name" in site.keys()
        return 'site dict with name %s'%site["name"]
    else:
        raise ValueError("What is this site %s? Only dicts or files are accepted"%site)

def dat2site(dat,comment="#",
             fmt="AAresSeq"):
    r""" Read a non-json (.dat, .txt...) file and turn it into a site dictionary

    The expected format is something like
    # interesting-contacts
    R38-H387
    GDP-R199
    GDP-R201

    The title line is optional, if present,
    will be used as name

    Parameters
    ----------
    dat : str,
        path to a file
    comment : str, default is "#"
        Ignore lines starting with
        the characters in this string
    fmt: str, default is "AAresSeq"
        The expected format of the file.
        'AAresSeq' means that the pairs
        are understood as AA-names
        followed by a sequence index:
        "GLU30-ARG131".

    -------
    site : a dictionary
     Same format as return of :obj:`x2site`
     "name" will be whatever :obj:`dat` was,
     without the extension
    """
    with open(dat,"r") as f:
        lines = f.read().splitlines()
    offset = 0
    name = _psplitext(_psplit(dat)[-1])[0]
    if lines[0].strip(" ").startswith("#"):
        name = lines[0].split("#")[1].strip(" ")
        offset +=1
    assert fmt in ["AAresSeq", "residx"], NotImplementedError("Only [AAresSeq, residx] are implemented for 'fmt' at the moment, can't do '%s'"%(fmt))
    site={"bonds":{fmt:[]}}
    for ii, line in enumerate(lines[offset:]):
        if line.strip(" ")[0] not in comment:
            assert line.count("-")==1, ValueError("The contact descriptor has to contain one (and just one) '-', got %s instead (%u-th line)"%(line, ii+1))
            site["bonds"][fmt].append(line)
            if fmt=="AAresSeq" and not any([chr.isalpha() for chr in line.replace("-","")]):
                raise ValueError("This can't be a %s line %s, are you sure you don't mean fmt=%s"%(fmt,line,"residx"))
            elif fmt=="residx" and not all([chr.isdigit() for chr in line.replace("-","")]):
                raise ValueError("This can't be a %s line %s are you sure you don't mean fmt=%s"%(fmt,line,"AAresSeq"))
    site["name"]=name
    return site


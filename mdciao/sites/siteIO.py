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

def load(site):
    r"""
    Return a site object from a dat or json file file or a dictionary.

    Simplest format is for the datfile to look  like this::
    >>> cat site.dat
    # contacts to look at :
    L394-K270
    D381-Q229
    Q384-Q229
    R385-Q229
    D381-K232
    Q384-I135

    But you can also annotate it in json format if you want
    >>> cat site.json
    {"name":"interesting contacts",
    "bonds": {"AAresSeq": [
            "L394-K270",
            "D381-Q229",
            "Q384-Q229",
            "R385-Q229",
            "D381-K232",
            "Q384-I135"
            ]}}

    Parameters
    ----------
    sitefile : str or dict
        Path to the ascii file. If the file isn't
        a json file, or doesn't contain the 'name'-field,
        the filename itself will be used as name.
        If a dict is passed, it's checked that the dictionary
        has the needed keys to function as a site.
    Returns
    -------
    site : dict
        Keys are:
         * bonds
         * nbonds
         * name
        site["bonds"] is itself a dictionary
        with only one key ATM, "AAresSeq",
        valued with a list of pairs, e.g ["L394","K270"]

    """
    if isinstance(site, dict):
        idict = _dcopy(site)
    else:
        try:
            with open(site, "r") as f:
                idict = _jsonload(f)
        except _JSONDecodeError as e:
            idict = dat2site(site)
    try:
        idict["bonds"]["AAresSeq"] = [item.split("-") for item in idict["bonds"]["AAresSeq"] if item[0] != '#']
        idict["n_bonds"] = len(idict["bonds"]["AAresSeq"])
    except:
        print("Malformed file for the site %s:\n%s" % (site,idict))

    if "name" not in idict.keys():
        if isinstance(site,str):
            idict["name"] = _psplitext(_psplit(site)[-1])[0]
        else:
            raise ValueError("A 'name'-key is mandatory when passing a dictionary")

    return idict

def sites_to_res_pairs(site_dicts, top,
                       fragments=None,
                       **get_fragments_kwargs,
                       ):
    r"""Take a list of dictionaries representing sites
    and return the pairs of res_idxs needed to compute
    all the contacts contained in them

    The idea is to join all needed pairs of res_idxs
    in one list regardless of where they come from.

    Parameters
    ----------
    site_dicts : list of dicts
        Check :obj:`load` for how these dics look like
    top : :obj:`mdtraj.Topology`
    fragments : list, default is None
        You can pass along a fragment definition so that
        :obj:`sites_to_AAresSeqdict` has an easier time
        understanding your input. Otherwise, one will
        be created on-the-fly by :obj:`mdciao.fragments.get_fragments`
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
        Anything else will raise
        a not implemented error

    -------
    site : a dictionary
     Same format as return of :obj:`load`
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
    assert fmt=="AAresSeq", NotImplementedError("Only 'AAresSeq is implemented for 'fmt' at the moment, can't do '%s'"%(fmt))
    site={"bonds":{"AAresSeq":[]}}
    for ii, line in enumerate(lines[offset:]):
        if line.strip(" ")[0] not in comment:
            assert line.count("-")==1, ValueError("The contact descriptor has to contain one (and just one) '-', got %s instead (%u-th line)"%(line, ii+1))
            site["bonds"]["AAresSeq"].append(line)
    site["name"]=name
    return site


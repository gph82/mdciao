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
         * pairs
         * n_pairs
         * name
        site["pairs"] is itself a dictionary
        valued with a list of pairs,
        e.g ["L394","K270"] or [[10,20],[300-4]]],
        depending on the type input

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
        bondtype = list(idict["pairs"].keys())
        assert len(bondtype)==1 and bondtype[0] in ["AAresSeq","residx"]
        bondtype = bondtype[0]
        pairs = idict["pairs"][bondtype]
        if isinstance(pairs[0][0],str):  # can only be str separated by "-"
            _pairs = []
            for item in pairs:
                if item[0].strip() != '#':
                    if "-" in item and isinstance(item,str):
                        _pairs.append(item.split("-"))
                    elif not isinstance(item, str) and len(item)==2:
                        _pairs.append(item)
                    else:
                        raise ValueError("Can't understand %s"%item)
                idict["pairs"][bondtype] = _pairs

            if bondtype=="residx":
                idict["pairs"][bondtype] = [[int(pp) for pp in pair] for pair in  idict["pairs"][bondtype]]
        else:
            assert all([len(bond)==2 for bond in pairs]),pairs
        idict["n_pairs"] = len(idict["pairs"][bondtype])

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
    r"""Return the pairs of res_idxs needed to compute the contacts contained in the input sites.

    The idea is to join all needed pairs of res_idxs
    in one list regardless of what site they come from.

    Note
    ----
    Any residue not found in :obj:`top` is assigned
    a 'None' in the returned :obj:`res_idx_pairs`.

    Parameters
    ----------
    site_dicts : list of dicts
        Anything that :obj:`mdciao.sites.x2site` understands
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
        Unique residue pairs contained in the :obj:`site_dicts`,
        expressed as residue indices of :obj:`top`
        [0,1] is considered != [0,1]. Any residues that
        couldn't be found will appear as 'None'
    site_maps : list
        For each site, a list with the indices of :obj:`res_idxs_pairs`
        that matches the site's pairs in :obj:`res_idxs_pairs`
    """
    res_idxs_pairs = []
    pair2idx = {}
    site_maps = []
    for ii, site in enumerate(site_dicts):
        imap=[]
        for bond_type, bonds in x2site(site)["pairs"].items():
            if bond_type=="AAresSeq":
                if fragments is None:
                    fragments = _mdcfrg.get_fragments(top, **get_fragments_kwargs)
                get_pair_lambda =  lambda bond: _mdcu.residue_and_atom.residues_from_descriptors(bond, fragments, top)[0]
            elif bond_type=="residx":
                get_pair_lambda = lambda bond: bond
            for bond in bonds:
                pair = tuple(list(get_pair_lambda(bond))+list(bond))
                if pair not in res_idxs_pairs:
                    res_idxs_pairs.append(pair)
                    pair2idx[pair]=len(res_idxs_pairs)-1
                imap.append(pair2idx[pair])
        site_maps.append(imap)
    #print(site_maps)
    return _np.vstack([pair[:2] for pair in res_idxs_pairs]), site_maps

def discard_empty_sites(ctc_idxs, site_maps, site_list, allow_partial_sites=True):
    r"""
    Helper method to reconfigure the output of :obj:`sites_to_res_pairs`
    discarding 'None'-entries

    Parameters
    ----------
    ctc_idxs : list
        List of residue index pairs
    site_maps : list
        List of len :obj:`sites`. Each item
        is itself a list, s.t. :obj:`site_maps[ii]`
        contains the list of indices of pairs
        in :obj:`ctc_idxs` that belong to the ii-th site
    site_list : list
        List containing site definitions
    allow_partial_sites : bool, default is True
        If False, a single 'None' is enough to
        eliminate the entire site. Default
        is to prune the site definitions
        of the 'None' values but keep
        the ones that work

    Returns
    -------
    out_ctc_idxs : list
        Like :obj:`ctc_idxs` but with the
        non-None entries only
    out_site_maps : list
        Like :obj:`site_maps` but with
        updated indices, both of
        the residue pairs in :obj:`site_maps`
        and of the sites in :obj:`out_sites`
        non-None entries
    out_sites : list
        The sites that weren't eliminated.
    discarded : dict
        Keys are "partial" and "full" indicating
        the indices of the sites in :obj:`site_list`
        where either partial deletion took place
        or the full site was discarded completely
    """
    discarded = {}
    if allow_partial_sites:
        gets_eliminated = lambda site : all([None in pair for pair in site["pairs"]["residx"]])
    else:
        gets_eliminated = lambda site : any([None in pair for pair in site["pairs"]["residx"]])

    out_sites = [{"name": isite["name"],
                  "pairs": {"residx": [list(ctc_idxs[ii]) for ii in site_maps[ii]]}}
                 for ii, isite in enumerate(site_list)]

    out_sites = {key: isite for key, isite in enumerate(out_sites) if not gets_eliminated(isite)}
    [isite["pairs"].__setitem__("residx", [pair for pair in isite["pairs"]["residx"] if None not in pair]) for isite in out_sites.values()]

    discarded["partial"] = [ii for ii, isite in out_sites.items() if len(isite["pairs"]["residx"])<len(site_maps[ii])]
    discarded["full"] = [ii for ii in range(len(site_list)) if ii not in out_sites.keys()]

    out_sites = {key: isite for key, isite in out_sites.items() if len(isite["pairs"]["residx"])>0}
    [isite.__setitem__("n_pairs", len(isite["pairs"]["residx"])) for isite in out_sites.values()]
    out_ctc_idxs, out_site_maps = sites_to_res_pairs(out_sites.values(), None)  # we know we won't need top or frags, can just parse None
    return out_ctc_idxs, out_site_maps, list(out_sites.values()), discarded

def site2str(site):
    r""" Produce a printable str for sitefile (json) or site-dict"""
    if isinstance(site,str):
        return site
    elif isinstance(site,dict):
        assert "name" in site.keys()
        return "site dict with name '%s'"%site["name"]
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
    site={"pairs":{fmt:[]}}
    for ii, line in enumerate(lines[offset:]):
        if line.strip(" ")[0] not in comment:
            assert line.count("-")==1, ValueError("The contact descriptor has to contain one (and just one) '-', got %s instead (%u-th line)"%(line, ii+1))
            site["pairs"][fmt].append(line)
            if fmt=="AAresSeq" and not any([chr.isalpha() for chr in line.replace("-","")]):
                raise ValueError("This can't be a %s line %s, are you sure you don't mean fmt=%s"%(fmt,line,"residx"))
            elif fmt=="residx" and not all([chr.isdigit() for chr in line.replace("-","")]):
                raise ValueError("This can't be a %s line %s are you sure you don't mean fmt=%s"%(fmt,line,"AAresSeq"))
    site["name"]=name
    return site


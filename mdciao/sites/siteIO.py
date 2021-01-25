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

def sites_to_AAresSeqdict(list_of_site_dicts, top, fragments,
                          raise_if_not_found=True,
                          **residues_from_descriptors_kwargs):

    r"""
    For a list of site dictionaries (see :obj:`load`), return
    a dictionary with keyed by all needed residue names and valued
    with their residue indices in :obj:`top`

    Note
    ----
    ATM it is not possible to have a residue name, e.g. ARG201 have different
    meanings (=res_idxs) in different sitefiles. All sitefiles are defined
    using AAresSeq names and all AAresSeq names need to mean the same
    residue across all sitefiles.

    TODO: The next (easy) step is to define another json entry with residue
    indices

    Parameters
    ----------
    list_of_site_dicts
    top
    fragments
    raise_if_not_found : boolean, default is True
        Fail if some of the residues are not found in the topology
    residues_from_descriptors_kwargs :
        see :obj:`mdciao.utils.residue_and_atom.residues_from_descriptors`

    Returns
    -------
    AAresSeq2residxs : dict
    """

    # Unfold all the resseqs
    AAresSeqs = [ss["bonds"]["AAresSeq"] for ss in list_of_site_dicts]
    AAresSeqs = [item for sublist in AAresSeqs for item in sublist]
    AAresSeqs = [item for sublist in AAresSeqs for item in sublist]
    AAresSeqs = [key for key in _np.unique(AAresSeqs)]
    residxs, _ = _mdcu.residue_and_atom.residues_from_descriptors(AAresSeqs, fragments, top,
                                              **residues_from_descriptors_kwargs)
    if None in residxs and raise_if_not_found:
        raise ValueError("These residues of your input have not been found. Please revise it:\n%s" %
                         ('\n'.join(["input %u"%ii for ii,__ in enumerate(residxs) if ii is None])))
    long2input = {str(top.residue(idx)):AA for idx,AA in zip(residxs,AAresSeqs)if idx is not None}

    AAresSeq2residxs = {}#key:None for key in AAresSeqs}
    for idx in residxs:
        if idx is not None:
            key = long2input[str(top.residue(idx))]
            #assert key in AAresSeqs  #otherwise _per_residue ... did not work
            AAresSeq2residxs[key] = idx

    return AAresSeq2residxs

def sites_to_res_pairs(site_dicts, top,
                       fragments=None,
                       **get_fragments_kwargs):
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
        The residue indices in :obj:`top` of the contacts in :obj:`site_dicts`,
        stacked together (might contain duplicates if the sites contain duplicates)
    AAdict : dict
        dictionary keyed by residue name and valued with the residue's index
        in :obj:`top`. Please see the note in :obj:`sites_to_AAresSeqdict`
        regarding duplicate residue names
    """
    if fragments is None:
        fragments = _mdcfrg.get_fragments(top, **get_fragments_kwargs)
    AAresSeq2residxs = sites_to_AAresSeqdict(site_dicts, top, fragments)
    res_idxs_pairs = _np.vstack(([[[AAresSeq2residxs[pp] for pp in pair] for pair in ss["bonds"]["AAresSeq"]] for ss in site_dicts]))
    return res_idxs_pairs, AAresSeq2residxs

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
             bonds="AAresSeq"):
    r""" Read a non-json (.dat, .txt...) file and turn it into a site dictionary

    Parameters
    ----------
    dat : string
        Filename
    comment : str, default is "#"
        Ignore lines starting with
        the characters in this string
    bonds : str, default "AAresSeq"
        How to interpret the file.
        Default is to interpret them
        in the AAresSeq format, e.g.
        "GLU30-ARG131"
    Returns
    -------
    site : a dictionary
     Same format as return of :obj:`load`
     "name" will be whatever :obj:`dat` was,
     without the extension



    """
    with open(dat,"r") as f:
        lines = f.read().splitlines()
    if bonds!="AAresSeq":
        raise NotImplementedError
    return {"name": _psplitext(_psplit(dat)[-1])[0],"bonds":{bonds:[line.replace(" ","") for line in lines if line.strip(" ")[0] not in comment]}}

def txt2site(txtfile,fmt="AAresSeq"):
    r""" Create a site-dict from a plain text file
    TODO merge with dat2site

    The expected format is something like
    # interesting-contacts
    R38-H387
    GDP-R199
    GDP-R201

    The title line is optional, if present,
    will be used as name

    Parameters
    ----------
    txtfile : path to a file
    fmt : str, default is 'AAresSeq'
        The expected format of the file.
        'AAresSeq' means that the pairs
        are understood as AA-names
        followed by a sequence index
        Anything else will raise
        a not implemented error


    Returns
    -------
    site : dict
        A site-dictionary with the usual format
        If no title-line was given, the filename
        witouth extension will be used as 'name'

    """
    with open(txtfile,"r") as f:
        lines = f.read().splitlines()
    offset=0
    name = _psplitext(_psplit(txtfile)[-1])[0]
    if lines[0].strip(" ").startswith("#"):
        name = lines[0].split("#")[1].strip(" ")
        offset +=1
    assert fmt=="AAresSeq", NotImplementedError("Only 'AAresSeq is implmented for 'fmt' at the moment, can't do '%s'"%(fmt))
    site={"bonds":{"AAresSeq":[]}}
    for ii, line in enumerate(lines[offset:]):
        assert line.count("-")==1, ValueError("The contact descriptor has to contain one (and just one) '-', got %s instead (%u-th line)"%(line, ii+1))
        site["bonds"]["AAresSeq"].append(line)
    site["name"]=name
    return site

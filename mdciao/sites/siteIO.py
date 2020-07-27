##############################################################################
#    This file is part of mdciao.
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

from json import load as _jsonload
from os.path import splitext as _psplitext, split as _psplit
import numpy as _np

import mdciao.fragments as _mdcfrg
import mdciao.utils as _mdcu

def sitefile2sitedict(sitefile):
    r"""
    Open a json file defining a site and turn into a site dictionary

    Examples
    --------
    >>> This could be inside a json file named site.json
    {"sitename":"interesting contacts",
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
    sitefile : str

    Returns
    -------
    site : dict
        Keys are:
         * bonds
         * nbonds
         * name
        And site["bonds"] is itself a dictionary with only one key ATM, "AAresSeq"

    """
    with open(sitefile, "r") as f:
        idict = _jsonload(f)
    try:
        idict["bonds"]["AAresSeq"] = [item.split("-") for item in idict["bonds"]["AAresSeq"] if item[0] != '#']
        idict["n_bonds"] = len(idict["bonds"]["AAresSeq"])
    except:
        print("Malformed .json file for the site %s" % sitefile)
    if "sitename" not in idict.keys():
        idict["name"] = _psplitext(_psplit(sitefile)[-1])[0]
    else:
        idict["name"] = _psplit(idict["sitename"])[-1]
    return idict

def sites_to_AAresSeqdict(list_of_site_dicts, top, fragments,
                          raise_if_not_found=True,
                          **residues_from_descriptors_kwargs):

    r"""
    For a list of site dictionaries (see :obj:`sitefile2sitedict`), return
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
        Check :obj:`sitefile2sitedict` for how these dics look like
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
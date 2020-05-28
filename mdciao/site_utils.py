from json import load as jsonload
from os.path import splitext, split as psplit
import numpy as _np

from mdciao.fragments import \
    get_fragments as _get_fragments, \
    per_residue_fragment_picker as _per_residue_fragment_picker

def sitefile2sitedict(sitefile):
    r"""
    Open a json file defining a site and turn into a site dictionary
    Parameters
    ----------
    sitefile : str

    Returns
    -------
    site : dict with the keys
    """
    with open(sitefile, "r") as f:
        idict = jsonload(f)
    try:
        idict["bonds"]["AAresSeq"] = [item.split("-") for item in idict["bonds"]["AAresSeq"] if item[0] != '#']
        idict["n_bonds"] = len(idict["bonds"]["AAresSeq"])
    except:
        print("Malformed .json file for the site %s" % sitefile)
    if "sitename" not in idict.keys():
        idict["name"] = splitext(psplit(sitefile)[-1])[0]
    else:
        idict["name"] = psplit(idict["sitename"])[-1]
    return idict

def sites_to_AAresSeqdict(list_of_site_dicts, top, fragments,
                          raise_if_not_found=True,
                          **_per_residue_fragment_picker_kwargs):

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
    _per_residue_fragment_picker_kwargs :
        see :obj:`_per_residue_fragment_picker`

    Returns
    -------
    AAresSeq2residxs : dict
    """

    # Unfold all the resseqs
    AAresSeqs = [ss["bonds"]["AAresSeq"] for ss in list_of_site_dicts]
    AAresSeqs = [item for sublist in AAresSeqs for item in sublist]
    AAresSeqs = [item for sublist in AAresSeqs for item in sublist]
    AAresSeqs = [key for key in _np.unique(AAresSeqs)]
    residxs, _ = _per_residue_fragment_picker(AAresSeqs, fragments, top,
                                              **_per_residue_fragment_picker_kwargs)
    long2input = {str(top.residue(idx)):AA for idx,AA in zip(residxs,AAresSeqs)}

    AAresSeq2residxs = {}#key:None for key in AAresSeqs}
    for idx in residxs:
        if idx is not None:
            key = long2input[str(top.residue(idx))]
            #assert key in AAresSeqs  #otherwise _per_residue ... did not work
            AAresSeq2residxs[key] = idx
    if None in AAresSeq2residxs.values() and raise_if_not_found:
        raise ValueError("These residues of your input have not been found. Please revise it:\n%s" % (
            '\n'.join([key for key, val in AAresSeq2residxs.items() if val is None])))

    return AAresSeq2residxs

def sites_to_ctc_idxs(site_dicts, top,
                      fragments=None,
                      **get_fragments_kwargs):
    r"""

    Parameters
    ----------
    site_dicts : list of dicts
    top : :obj:`mdtraj.Topology`
    get_fragments_kwargs :
        see :obj:`fragments.get_fragments`

    Returns
    -------
    ctc_idxs : 2D np.ndarray
        The residue indices in :obj:`top` of the contacts in :obj:`site_dict`,
        stacked together (might contain duplicates if the sites contain duplicates)
    AAdict : dict
        dictionary keyed by residue name and valued with the residue's index
        in :obj:`top`
    """
    if fragments is None:
        fragments = _get_fragments(top, **get_fragments_kwargs)
    AAresSeq2residxs = sites_to_AAresSeqdict(site_dicts, top, fragments)
    ctc_idxs = _np.vstack(([[[AAresSeq2residxs[pp] for pp in pair] for pair in ss["bonds"]["AAresSeq"]] for ss in site_dicts]))
    return ctc_idxs, AAresSeq2residxs
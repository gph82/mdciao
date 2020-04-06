from glob import glob as _glob
import numpy as _np
import mdtraj as _md
from .list_utils import re_warp

def get_sorted_trajectories(trajectories):
    r"""
    Common parser for something that can be interpreted as a trajectory

    Parameters
    ----------
    trajectories: can be one of these things:
        - pattern, e.g. "*.ext"
        - one string containing a filename
        - list of filenames
        - one :obj:`mdtraj.Trajectory` object
        - list of :obj:`mdtraj.Trajectory` objects

    Returns
    -------
        - for an input pattern, sorted trajectory filenames that match that pattern
        - for filename, one list containing that filename
        - for a list of filenames, a sorted list of filenames
        - one :obj:`mdtraj.Trajectory` object (i.e: does nothing)
        - list of :obj:`mdtraj.Trajectory` objects (i.e. does nothing)


    """
    if isinstance(trajectories,str):
        trajectories = _glob(trajectories)

    if isinstance(trajectories[0],str):
        xtcs = sorted(trajectories)
    elif isinstance(trajectories, _md.Trajectory):
        xtcs = [trajectories]
    else:
        assert all([isinstance(itraj, _md.Trajectory) for itraj in trajectories])
        xtcs = trajectories

    return xtcs

def _inform_about_trajectories(trajectories):
    r"""

    Parameters
    ----------
    trajectories: list of strings or :obj:`mdtraj.Trajectory` objects

    Returns
    -------
    nothing, just prints them as newline

    """
    assert isinstance(trajectories, list), "input has to be a list"
    return "\n".join([str(itraj) for itraj in trajectories])

# TODO consider dict_utils??
def _replace_w_dict(key, exp_rep_dict):
    r"""

    Parameters
    ----------
    key: str
    exp_rep_dict: dictionary
        keys are expressions that will be replaced with values, i.e.
        key = key.replace(key1, val1) for key1, val1 etc

    Returns
    -------
    key

    """
    for pat, exp in exp_rep_dict.items():
        key = key.replace(pat,exp)
    return key

def _delete_exp_in_keys(idict, exp, sep="-"):
    r"""
    Assuming the keys in the dictionary are formed by two segments
    joined by a separator, e.g. "GLU30-ARG40", deletes the segment
    containing the input expression, :obj:`exp`

    Will fail if not all keys have the expression to be deleted

    Parameters
    ----------
    idict: dictionary
    exp: str
    sep: str, default is "-",

    Returns
    -------
    dict:
        dictionary with the same values but the keys lack the
        segment containing :obj:`exp`
    """

    out_dict = {}
    for names, val in idict.items():
        name = [name for name in names.split(sep) if exp not in name]
        assert len(name) == 1, name
        out_dict[name[0]]=val
    return out_dict

def unify_freq_dicts(freqs,
                     exclude=None,
                     key_separator="-",
                     replacement_dict={},
                     reorder_keys=True):

    def order_key(key, sep):
        split_key = key.split(sep)
        return sep.join([split_key[ii] for ii in _np.argsort(split_key)])

    freqs_work = {}
    for key, idict in freqs.items():
        if reorder_keys:
            freqs_work[key] = {order_key(key, key_separator):val for key, val in idict.items()}
        else:
            freqs_work[key] = {key:val for key, val in idict.items()}

    if replacement_dict is not None:
        freqs_work = {key:{_replace_w_dict(key2, replacement_dict):val2 for key2, val2 in val.items()} for key, val in freqs_work.items()}

    not_shared = []
    shared = []
    for idict1 in freqs_work.values():
        for idict2 in freqs_work.values():
            if not idict1 is idict2:
                not_shared += list(set(idict1.keys()).difference(idict2.keys()))
                shared += list(set(idict1.keys()).intersection(idict2.keys()))

    shared = list(_np.unique(shared))
    not_shared = list(_np.unique(not_shared))
    all_keys = shared + not_shared

    if exclude is not None:
        print("Excluding")
        for ikey, ifreq in freqs_work.items():
            for key in shared:
                for pat in exclude:
                    if pat in key:
                        ifreq.pop(key)
                        print("%s from %s" % (key, ikey))
                        all_keys = [ak for ak in all_keys if ak != key]

    for ikey, ifreq in freqs_work.items():
        for key in not_shared:
            if key not in ifreq.keys():
                ifreq[key] = 0

    if len(not_shared)>0:
        print("These interactions are not shared:\n%s" % (', '.join(not_shared)))
        print("Their cummulative ctc freq is %f. " % _np.sum(
            [[ifreq[key] for ifreq in freqs_work.values()] for key in not_shared]))

    return freqs_work



def _replace4latex(istr):
    for gl in ['alpha','beta','gamma', 'mu']:
        istr = istr.replace(gl,'$\\'+gl+'$')

    if '$' not in istr and any([char in istr for char in ["_"]]):
        istr = '$%s$'%istr
    return istr

def iterate_and_inform_lambdas(ixtc,stride,chunksize, top=None):
    if isinstance(ixtc, _md.Trajectory):
        iterate = lambda ixtc: [ixtc[idxs] for idxs in re_warp(_np.arange(ixtc.n_frames)[::stride], chunksize)]
        inform = lambda ixtc, traj_idx, chunk_idx, running_f: print("Analysing trajectory object nr. %3u (%7u frames) in chunks of "
                                                   "%3u frames. chunknr %4u frames %8u" %
                                                   (traj_idx, ixtc.n_frames, chunksize, chunk_idx, running_f), end="\r", flush=True)
    else:
        iterate = lambda ixtc: _md.iterload(ixtc, top=top, stride=stride, chunk=_np.round(chunksize / stride))
        inform = lambda ixtc, traj_idx, chunk_idx, running_f: print("Analysing %20s (nr. %3u) in chunks of "
                                                   "%3u frames. chunknr %4u frames %8u" %
                                                   (ixtc, traj_idx, chunksize, chunk_idx, running_f), end="\r", flush=True)
    return iterate, inform
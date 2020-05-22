from glob import glob as _glob
import numpy as _np
import mdtraj as _md
from .list_utils import re_warp
from fnmatch import fnmatch as _fnmatch

_tunit2tunit = {"ps":  {"ps": 1,   "ns": 1e-3, "mus": 1e-6, "ms":1e-9},
                "ns":  {"ps": 1e3, "ns": 1,    "mus": 1e-3, "ms":1e-6},
                "mus": {"ps": 1e6, "ns": 1e3,  "mus": 1,    "ms":1e-3},
                "ms":  {"ps": 1e9, "ns": 1e6,  "mus": 1e3,  "ms":1},
                }

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

def _replace_w_dict(input_str, exp_rep_dict):
    r"""
    Sequentially perform string replacements on a string using a dictionary

    Parameters
    ----------
    input_str: str
    exp_rep_dict: dictionary
        keys are expressions that will be replaced with values, i.e.
        key = key.replace(key1, val1) for key1, val1 etc

    Returns
    -------
    key

    """
    for pat, exp in exp_rep_dict.items():
        input_str = input_str.replace(pat, exp)
    return input_str

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
                     replacement_dict=None,
                     ):
    r"""
    Provided with a dictionary of dictionaries, returns an equivalent,
    key-unified dictionary where all sub-dictionaries share their keys,
    putting zeroes where keys where absent originally.

    Use :obj:`key_separator` for "GLU30-LY40" == "LYS40-GLU30" to be True

    Parameters
    ----------
    freqs:  dictionary of dictionaries, e.g.:
        {A:{key1:valA1, key2:valA2, key3:valA3},
         B:{            key2:valB2, key3:valB3}}

    key_separator: str, default is "-"
        If keys are made up like "GLU30-LYS40", you can specify a separator s.t.
        "GLU30-LYS40" is considered equal to "LYS40-GLU30".
        Use "", "none" or None to differentiate

    exclude: list, default is None
         keys containing these strings will be excluded.
         NOTE: This is not implemented yet, will raise an error

    replacement_dict: dict, default is {}
        all keys/strings will be subjected to replacements following this
        dictionary, st. "GLH30" is "GLU30" if replacement_dict is {"GLH":"GLU"}
        This way mutations and or indexing can be accounted for in different setups

    Returns
    -------
    unified_dict: dictionary
        A dictionary  of dictionaries sharing keys:
       {A:{key1:valA1, key2:valA2, key3:valA3},
        B:{key1:0,     key2:valB2, key3:valB3}}
    """

    # Order key alphabetically using the separator_key
    def order_key(key, sep):
        split_key = key.split(sep)
        return sep.join([split_key[ii] for ii in _np.argsort(split_key)])

    # Create a copy, with re-ordered keys if needed
    freqs_work = {}
    for key, idict in freqs.items():
        if str(key_separator).lower()=="none" or len(key_separator)==0:
            freqs_work[key] = {key:val for key, val in idict.items()}
        else:
            freqs_work[key] = {order_key(key, key_separator):val for key, val in idict.items()}

    # Implement replacements
    if replacement_dict is not None:
        freqs_work = {key:{_replace_w_dict(key2, replacement_dict):val2 for key2, val2 in val.items()} for key, val in freqs_work.items()}

    # Perform the difference operations
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

    # Prune keys we're not interested in
    excluded = []
    if exclude is not None:
        raise NotImplementedError("This feature not yet implemented")
        """
        assert isinstance(exclude,list)
        print("Excluding")
        for ikey, ifreq in freqs_work.items():
            # IDK I had this condition here, i think it is more intuitive if
            # they are removed regardless if shared or not
            #for key in shared:
            for key in list(ifreq.keys()):
                for pat in exclude:
                    if pat in key:
                        ifreq.pop(key)
                        print("%s from %s" % (key, ikey))
                        print(ifreq.keys())
                        excluded.append(key)
                        #all_keys = [ak for ak in all_keys if ak != key]
        """

    # Set the non shared keys to zero
    for ikey, ifreq in freqs_work.items():
        for key in all_keys:
            if key not in ifreq.keys():
                ifreq[key] = 0

    if len(not_shared)>0:
        print("These interactions are not shared:\n%s" % (', '.join(not_shared)))
        print("Their cummulative ctc freq is %f. " % _np.sum(
            [[ifreq[key] for ifreq in freqs_work.values()] for key in not_shared]))

    return freqs_work

def freq_datfile2freqdict(ifile, comment=["#"]):
    r"""
    Reads an ascii file that contains contact frequencies (1st) column and
    contact labels . Columns are separeted by tabs or spaces.

    Contact labels have to come after the frequency in the
    form of "res1 res2, "res1-res2" or "res1 - res2",

    Columns other than the frequencies and the residue labels are ignored


    TODO use pandas to allow more flex, not needed for the moment

    Parameters
    ----------
    ifile : str
        The filename to be read

    comment : list of chars
        Any line starting with any of these
        charecters will be ignored
    Returns
    -------
    freqdict : dictionary

    """
    #TODO consider using pandas
    outdict = {}
    with open(ifile) as f:
        for iline in f.read().splitlines():
            if iline.strip()[0] not in comment:
                try:
                    iline = iline.replace("-"," ").split()
                    freq, names = float(iline[0]),"%s-%s"%(iline[1],iline[2])
                    outdict[names]=float(freq)
                except ValueError:
                    print(iline)
                    raise
    return outdict

def _replace4latex(istr):
    r"""
    One of two things:
        Prepares the input for latex rendering (in matplotlib, e.g.)
        For strings with greek letters or underscores.
        "alpha = 7"-> "$\alpha$ = 7"
        "C_2"      -> "$C_2$"
        "C^2"      -> "$C^2$"

    Note
    -----
        A combination of both ("alpha = C_2"->"$\alpha = C_2$") is not
        yet implemented

    Parameters
    ----------
    istr: str

    Returns
    -------
    alpha:$\alpha$

    """
    for gl in ['alpha','beta','gamma', 'mu', "Sigma"]:
        istr = istr.replace(gl,'$\\'+gl+'$')

    for syms in ["AA","Ang"]:
        istr = istr.replace(syms, '$\\' + syms + '$')

    # This mode of comparison will
    if any([cc in istr for cc in ["_", "^"]]):
        if '$' not in istr:
            istr = '$%s$'%istr
        else:
            raise NotImplementedError("The str already contains a dollar symbol, this is not implemented yet")
    return istr

def iterate_and_inform_lambdas(ixtc,chunksize, stride=1, top=None):
    r"""
    Given a trajectory (as object or file), returns
    a strided, chunked iterator and function for progress report

    Parameters
    ----------
    ixtc: str (filename) or :obj:`mdtraj.Trajectory` object
    chunksize: int
        The trajectory will be iterated over in chunks of this many frames
    stride: int, default is 1
        The stride with which to iterate over the trajectory
    top:  str (filename) or :obj:`mdtraj.Topology`
        If :obj:`ixtc` is a filename, the topology needed to read it

    Returns
    -------

    iterate, inform

    iterate: lambda(ixtc)
        strided, chunked iterator over :obj:`ixtc`

    inform: lambda(ixtc, traj_idx, chunk_idx, running_f)
        iterator that prints out streaming progress for every iteration

    Note
    ----

    The lambdas returned differ depending on the type of input, but signature
    is the same, s.t. the user does not have to care in posterior use

    """
    if isinstance(ixtc, _md.Trajectory):
        iterate = lambda ixtc: [ixtc[idxs] for idxs in re_warp(_np.arange(ixtc.n_frames)[::stride], chunksize)]
        inform = lambda ixtc, traj_idx, chunk_idx, running_f: \
            print("Streaming over trajectory object nr. %3u (%6u frames, %6u with stride %2u) in chunks of "
                  "%3u frames. Now at chunk nr %4u, frames so far %6u" %
                  (traj_idx, ixtc.n_frames, _np.ceil(ixtc.n_frames/stride), stride, chunksize, chunk_idx, running_f), end="\r", flush=True)
    elif ixtc.endswith(".pdb") or ixtc.endswith(".pdb.gz") or ixtc.endswith(".gro"):
        iterate =  lambda ixtc: [_md.load(ixtc)[::stride]]
        inform  =  lambda ixtc, traj_idx, chunk_idx, running_f: \
            print("Loaded %20s (nr. %3u) in full, using stride %2u but ignoring chunksize of "
                  "%6u frames. This number should always be 0 : %4u. Total frames loaded %6u" %
                  (ixtc, traj_idx, stride, chunksize, chunk_idx, running_f), end="\r", flush=True)
    else:
        iterate = lambda ixtc: _md.iterload(ixtc, top=top, stride=stride, chunk=_np.round(chunksize / stride))
        inform = lambda ixtc, traj_idx, chunk_idx, running_f: \
            print("Streaming %20s (nr. %3u) with stride %2u in chunks of "
                  "%6u frames. Now at chunk nr %4u, frames so far %6u" %
                  (ixtc, traj_idx, stride, chunksize, chunk_idx, running_f), end="\r", flush=True)
    return iterate, inform

def choose_between_good_and_better_strings(good_option, better_option,
                                           fmt="%s",
                                           never_use=[None, "None", "NA", "na"]):
    if good_option in never_use:
        if better_option in never_use:
            return ""
        else:
            return fmt % better_option
    elif good_option not in never_use:
        if better_option in never_use:
            return fmt % good_option
        else:
            return fmt % better_option

def fnmatch_ex(patterns_as_csv, list_of_keys):
    r"""
    Match the keys in :obj:`list_of_keys` against some naming patterns
    using Unix filename pattern matching
    TODO include link:  https://docs.python.org/3/library/fnmatch.html

    This method also allows for exclusions (grep -e)

    TODO: find out if regular expression re.findall() is better

    Uses fnmatch under the hood

    Parameters
    ----------
    patterns_as_csv : str
        Patterns to include or exclude, separated by commas, e.g.
        * "H*,-H8" will include all TMs but not H8
        * "G.S*" will include all beta-sheets
    list_of_keys : list
        Keys against which to match the patterns, e.g.
        * ["H1","ICL1", "H2"..."ICL3","H6", "H7", "H8"]

    Returns
    -------
    matching_keys : list

    """
    include_patterns = [pattern for pattern in patterns_as_csv.split(",") if not pattern.startswith("-")]
    exclude_patterns = [pattern[1:] for pattern in patterns_as_csv.split(",") if pattern.startswith("-")]
    #print(include_patterns)
    #print(exclude_patterns)
    # Define the match using a lambda
    matches_include = lambda key : any([_fnmatch(str(key), patt) for patt in include_patterns])
    matches_exclude = lambda key : any([_fnmatch(str(key), patt) for patt in exclude_patterns])
    passes_filter = lambda key : matches_include(key) and not matches_exclude(key)
    outgroup = []
    for key in list_of_keys:
        #print(key, matches_include(key),matches_exclude(key),include_patterns, exclude_patterns)
        if passes_filter(key):
            outgroup.append(key)
    return outgroup

def match_dict_by_patterns(patterns_as_csv, index_dict, verbose=False):
    r"""
    Joins all the values in an input dictionary if their key matches
    some patterns. This method also allows for exclusions (grep -e)

    TODO: find out if regular expression re.findall() is better

    Parameters
    ----------
    patterns_as_csv : str
        Comma-separated patterns to include or exclude, separated by commas, e.g.
        * "H*,-H8" will include all TMs but not H8
        * "G.S*" will include all beta-sheets
    index_dict : dictionary
        It is expected to contain iterable of ints or floats or anything that
        is "joinable" via np.hstack. Typically, something like:
        * {"H1":[0,1,...30], "ICL1":[31,32,...40],...}

    Returns
    -------
    matching_keys, matching_values : list, array of joined values

    """
    matching_keys =   fnmatch_ex(patterns_as_csv, index_dict.keys())
    if verbose:
        print(', '.join(matching_keys))

    if len(matching_keys)==0:
        matching_values = []
    else:
        matching_values = _np.hstack([index_dict[key] for key in matching_keys])

    return matching_keys, matching_values
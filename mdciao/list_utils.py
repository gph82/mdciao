import numpy as _np
import mdtraj as _md
from glob import glob as _glob
#taken from molpx
def re_warp(array_in, lengths):
    """Return iterable ::py:obj:array_in as a list of arrays, each
     one with the length specified in lengths

    Parameters
    ----------

    array_in: any iterable
        Iterable to be re_warped

    lengths : int or iterable of integers
        Lengths of the individual elements of the returned array. If only one int is parsed, all lengths will
        be that int. Special cases:
            * more lengths than needed are parsed: the last elements of the returned value are empty
            until all lengths have been used
            * less lengths than array_in could take: only the lenghts specified are returned in the
            warped list, the rest is unreturned
    Returns
    -------
    warped: list
    """

    if _np.ndim(lengths)==0:
        lengths = [lengths] * int(_np.ceil(len(array_in) / lengths))

    warped = []
    idxi = 0
    for ii, ll in enumerate(lengths):
        warped.append(array_in[idxi:idxi+ll])
        idxi += ll
    return warped

def is_iterable(var):
    """
    Checks if the input is an iterable or not

    Parameters
    ----------
    var : integer, float, string, list

    Returns
    -------
    boolean
        Returns 'True' if var is iterable else False

    """

    try:
        for ii in var:
            return True
    except:
        return False
    return True


def force_iterable(var):
    """
    Forces var to be iterable, if not already

    Parameters
    ----------
    var : integer, float, string , list

    Returns
    -------
    iterable
        var as iterable

    """
    if not is_iterable(var):
        return [var]
    else:
        return var


def does_not_contain_strings(iterable):
    """
    Checks if iterable has any string element, returns False if it contains atleast one string

    Parameters
    ----------
    iterable : integer, float, string or any combination thereof

    Returns
    -------
    boolean
        True if iterable does not contain any string, else False

    """

    return all([not isinstance(ii, str) for ii in iterable])

def unique_list_of_iterables_by_tuple_hashing(ilist, return_idxs=False):
    """
    Returns the unique entries(if there are duplicates) from a list of iterables.
    Order matters, i.e. [[0,1],[1,0]] are considered different iterables
    If ilist contains non-iterables, they will be considered as iterables for comparison purposes, s.t.
    1==[1]==np.array(1) and 'A'==['A']

    Parameters
    ----------
    ilist : list of iterables
        list of iterables with redundant entries (redundant in the list, not in entries)
    return_idxs : boolean
        'True' if required to return indices instead of unique list. (Default is False).

    Returns
    -------
    list
        list of unique iterables or indices of 'ilist' where the unique entries are

    """

    # Check for stupid input
    if len(ilist) == 1:
        # TODO: avoid using print functions and use loggers
        print("Input is an iterable of len = 1. Doing nothing")
        if not return_idxs:
            return ilist
        else:
            return [0]

    # Now for the actual work
    idxs_out = []
    ilist_out = []
    seen = []
    for ii, sublist in enumerate(ilist):
        if isinstance(sublist, _np.ndarray):
            sublist = sublist.flatten()
        this_objects_id = hash(tuple(force_iterable(sublist)))

        # print(sublist, this_objects_id)
        if this_objects_id not in seen:
            ilist_out.append(force_iterable(sublist))
            idxs_out.append(force_iterable(ii))
            seen.append(this_objects_id)
    if not return_idxs:
        return ilist_out
    else:
        return idxs_out
def window_average_fast(input_array_y, half_window_size=2):
    """
    Returns the moving average using np.convolve
    Parameters
    ----------
    input_array_y : array
                    numpy array for which moving average should be calculated
    half_window_size : int
                        the actual window size will be 2 * half_window_size + 1.
                        Example- when half window size = 2, moving average calculation will use window=5
    Returns
    -------
    array

    """
    input_array_y = (input_array_y).astype(float)
    window = _np.ones(2*half_window_size+1)
    return _np.convolve(input_array_y, window, mode="valid")/len(window)

#Lifted from my own aGPCR utils
def window_average(input_array_y, half_window_size=2):
    """
    Returns average and standard deviation inside the window
    Parameters
    ----------
    input_array_y : array
                    numpy array for which average and standard deviation should be calculated
    half_window_size : int
                            the actual window size will be half_window_size*2 + 1.
                        Example- when half window size = 2, moving average calculation will use window=5
    Returns
    -------
    array_out_mean, array_out_std
    two arrays corresponding to mean and standard deviation
    """
    array_out_mean = []
    array_out_std = []
    assert (half_window_size*2 + 1 <= len(input_array_y)),"In window average, input array should be >= half_window_size*2 + 1"
    for ii in range(half_window_size, len(input_array_y) - half_window_size):
        idxs = _np.hstack([_np.arange(ii - half_window_size, ii),
                           ii,
                           _np.arange(ii + 1, ii + half_window_size + 1)])
        #print(idxs.shape)
        array_out_mean.append(_np.average(input_array_y[idxs]))
        array_out_std.append(_np.std(input_array_y[idxs]))
    array_out_mean = _np.array(array_out_mean)
    array_out_std = _np.array(array_out_std)

    return array_out_mean, array_out_std

def window_average_vec(input_array_y, half_window_size=2):
    r"""
    like a convolution but returns also the std inside the window
    :param input_array_y:
    :param window_size:
    :param input_array_x:
    :return:
    """
    array_out_mean = []
    array_out_std = []
    for ii in range(half_window_size, len(input_array_y) - half_window_size):
        idxs = _np.hstack([_np.arange(ii - half_window_size, ii),
                           ii,
                           _np.arange(ii + 1, ii + half_window_size + 1)])
        print(idxs.shape)
        array_out_mean.append(_np.average(input_array_y[idxs, :], axis=0))
        array_out_std.append(_np.std(input_array_y[idxs, :],axis=0))
    array_out_mean = _np.array(array_out_mean)
    array_out_std = _np.array(array_out_std)

    return array_out_mean, array_out_std

# from https://www.rosettacode.org/wiki/Range_expansion#Python
def rangeexpand(txt):
    """
    For a given integer range or multiple integer ranges, returns a list of individual integers.
    Example- "1-2,3-4" will return [1,2,3,4]

    Parameters
    ----------
    txt : string
        string of integers or integer range separated by ","

    Returns
    -------
    list
        list of integers

    """
    lst = []
    for r in txt.split(','):
        if '-' in r[1:]:
            r0, r1 = r[1:].split('-', 1)
            lst += range(int(r[0] + r0), int(r1) + 1)
        else:
            lst.append(int(r))
    return lst

# TODO CONSISIDER MERGING in_what_N_fragments w. in_what_fragment
def in_what_N_fragments(idxs, fragment_list):
    """
    For each element of idxs, return the index of "fragments" in which it appears

    Parameters
    ----------
    idxs : integer, float, or iterable thereof
    fragment_list : iterable of iterables
        iterable of iterables containing integers or floats

    Returns
    -------
    list
        list of length len(idxs) containing an iterable with the indices of 'fragments' in which that index appears

    """

    # Sanity checking
    idxs = force_iterable(idxs)
    assert does_not_contain_strings(idxs), ("Input 'idxs' cannot contain strings")
    # TODO: consider refactoring the loop into the function call (avoid iterating over function calls)
    assert all([does_not_contain_strings(ifrag) for ifrag in fragment_list]), ("Input 'fragment_list'  cannot contain strings")

    result = (_np.vstack([_np.in1d(idxs, iseg) for ii, iseg in enumerate(fragment_list)])).T
    return [_np.argwhere(row).squeeze() for row in result]

def in_what_fragment(residx,
                     list_of_nonoverlapping_lists_of_residxs,
                     fragment_names=None):
    """
    For the residue id, returns the name(if provided) or the index of the "fragment"
    in which it appears

    Parameters
    ----------
    residx : int
        residue index
    list_of_nonoverlapping_lists_of_residxs : list
        list of integer list of non overlapping ids
    fragment_names : (optional) list of strings
        fragment names for each list in list_of_nonoverlapping_lists_of_residxs

    Returns
    -------
    integer or string
        returns the name(if fragment_names is provided) otherwise returns index of the "fragment"
        in which the residue index appears

    """

    # TODO deal with np.int64 etc
    assert type(residx) in (int, _np.int64), "Incorrect input: residx should be int, and not %s" % type(residx)
    assert does_not_contain_strings(list_of_nonoverlapping_lists_of_residxs)

    if fragment_names is not None:
        assert len(fragment_names) == len(list_of_nonoverlapping_lists_of_residxs)
    for ii, ilist in enumerate(list_of_nonoverlapping_lists_of_residxs):
        if residx in ilist:
            if fragment_names is None:
                return ii
            else:
                return fragment_names[ii]

def exclude_same_fragments_from_residx_pairlist(pairlist,
                                                fragments,
                                                return_excluded_idxs=False):
    """
    If the members of the pair belong to the same fragment, exclude them from pairlist.

    Parameters
    ----------
    pairlist : list of iterables
        each iterable within the list should be a pair.
    fragments : list of iterables
        each inner list should have residue indexes that form a fragment
    return_excluded_idxs : boolean
        True if index of excluded pair is needed as an output. (Default is False).

    Returns
    -------
    list
        pairs that don't belong to the same fragment,
        or index of the excluded pairs if return_excluded_idxs is True

    """

    idxs2exclude = [idx for idx, pair in enumerate(pairlist) if
                    _np.diff([in_what_fragment(ii, fragments) for ii in pair]) == 0]

    if not return_excluded_idxs:
        return [pair for ii, pair in enumerate(pairlist) if ii not in idxs2exclude]
    else:
        return idxs2exclude

def assert_min_len(input_iterable, min_len=2):
    """
    Checks if an iterable satisfies the criteria of minimum length. (Default minimum length is 2).
    Parameters
    ----------
    input_iterable : numpy array, list of list
                    example np.zeros((2,1,1) or [[1,2],[3,4]] when min_len = 2
    min_len : minimum length which the iterable should satisfy (Default is 2)

    Returns
    -------
    Prints error if each item within the iterable has lesser number of elements than min_len

    """
    for ii, element in enumerate(input_iterable):
        if _np.ndim(element)==0 or len(element) < min_len:
            aerror = 'The %s-th element has too few elements (min %s): %s' % (ii, min_len, element)
            raise AssertionError(aerror)

def join_lists(lists, idxs_of_lists_to_join):
    #todo document and test
    assert_min_len(idxs_of_lists_to_join)

    # Removing the redundant entries in each list and sorting them
    idxs_of_lists_to_join = [_np.unique(jo) for jo in idxs_of_lists_to_join]




    # Assert the join_fragments do not overlap
    assert_no_intersection(idxs_of_lists_to_join)
    joined_lists = []
    lists_idxs_used_for_joining = []
    for ii, jo in enumerate(idxs_of_lists_to_join):
        # print(ii,jo)
        this_new_frag = []
        lists_idxs_used_for_joining.extend(jo)
        for frag_idx in jo:
            # print(frag_idx)
            this_new_frag.extend(lists[frag_idx])
        # print(this_new_frag)
        joined_lists.append(_np.array(this_new_frag))

    # TODO: THIS is only good programming bc the lists are very very small, otherwise np.delete is the way to go
    surviving_initial_fragments = [ifrag for ii, ifrag in enumerate(lists)
                                   if ii not in lists_idxs_used_for_joining]
    lists = joined_lists + surviving_initial_fragments

    # Order wrt to the first index in each fragment
    order = _np.argsort([ifrag[0] for ifrag in lists])
    lists = [lists[oo] for oo in order]

    return lists

def assert_no_intersection(list_of_lists_of_integers):
    """
    Checks if two or more lists contain the same integer
    Parameters
    ----------
    list_of_lists_of_integers : list of lists

    Returns
    -------
    Prints assertion message if inner lists have the same integer, else no output

    """
    # Nested loops feasible here because the number of fragments will never become too large
    for ii, l1 in enumerate(list_of_lists_of_integers):
        for jj, l2 in enumerate(list_of_lists_of_integers):
            if (ii != jj):
                assert (l1 + l2), "Both lists are empty! See https://www.coopertoons.com/education/emptyclass_intersection/emptyclass_union_intersection.html"
                assert len(_np.intersect1d(l1, l2)) == 0, 'join fragment id overlaps!'

# TODO consider using np.delete in the code originally?
def pull_one_up_at_this_pos(iterable_in, idx, padding='~',
                            verbose=False):
    """

    Parameters
    ----------
    iterable_in : iterable
    idx : int or None
        index which needs to be removed (zero-based index), or None if nothing needs to be replaced
    padding : replacement to be appended to the end of iterable. Could be string, int, iterable and so on
    verbose : boolean
            be verbose

    Returns
    -------
    iterable without the element at the position idx on iterable_in

    """

    iterable_out = [ii for ii in iterable_in[:idx]]
    iterable_out += iterable_in[idx + 1:]
    iterable_out.append(padding)
    if verbose:
        for ii, (old, new) in enumerate(zip(iterable_in, iterable_out)):
            print(ii, old, new)
    return iterable_out

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

def put_this_idx_first_in_pair(idx, pair):
    """
    Returns the original pair if the value already appears first, else returns reversed pair
    Parameters
    ----------
    idx : value which needs to be brought in the first place (not the index but value itself)
    pair : list
            pair of values as a list
    Returns
    -------
    pair

    """
    if pair[0] != idx and pair[1] == idx:
        pair = pair[::-1]
    elif pair[0] == idx and pair[1] != idx:
        pass
    else:
        print(pair)
        raise Exception
    return pair

def get_sorted_trajectories(trajectories):
    if isinstance(trajectories,str):
        trajectories = _glob(trajectories)

    if isinstance(trajectories[0],str):
        xtcs = sorted(trajectories)
    else:
        xtcs = trajectories

    return xtcs

def _inform_about_trajectories(trajectories):
    return "\n".join([str(itraj) for itraj in trajectories])

# TODO consider dict_utils??
def _replace_w_dict(key, pat_exp_dict):
    for pat, exp in pat_exp_dict.items():
        key = key.replace(pat,exp)
    return key

def _delete_exp_in_keys(idict, exp, sep="-"):
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
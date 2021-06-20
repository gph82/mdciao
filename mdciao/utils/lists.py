r"""
Miscellaneous operations on list or list-like objects
.. autosummary::
   :nosignatures:
   :toctree: generated/


"""

import numpy as _np

from itertools import groupby as _groupby
from collections import defaultdict as _defdict
def contiguous_ranges(list_in):
    r"""
    For every unique entry in :obj:`list_in` return the contiguous ranges in list

    Parameters
    ----------
    list_in : list

    Returns
    -------
    ranges : dict
        The keys are with unique entries of list_in, values are the ranges
        in which the entry appears

    """
    offset = 0
    _ranges = _defdict(list)
    for key, grpr in _groupby(list_in):
        l = len(list(grpr))
        irange = _np.arange(offset, offset + l)
        _ranges[key].append(irange)
        offset+=l
    return dict(_ranges)

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
    Order matters, i.e. [[0,1],[1,0]] are considered different iterables (unlike np.unique does)
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

#TODO consider list and str utils for this?
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
    return [_np.flatnonzero(row) for row in result]

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
        returns the name(if names is provided) otherwise returns index of the "fragment"
        in which the residue index appears

    """

    # TODO deal with np.int64 etc
    assert type(residx) in (int, _np.int64), "Incorrect input: residx should be int, and not %s" % type(residx)
    assert does_not_contain_strings(list_of_nonoverlapping_lists_of_residxs), list_of_nonoverlapping_lists_of_residxs

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
    r"""
    Provided a list of lists, join them following idxs_of_lists_to_join

    Parameters
    ----------
    lists: iterable of iterables
        The lists to be joined

    idxs_of_lists_to_join: iterable of iterables containing integers
        The lists to join. These  3 things will be done before using this array
            - remove duplicate entries in each iterable
            - sort the entries in each iterable by ascending order
            - assert there is no overlap between iterables

    Returns
    -------
    joined_lists: iterable of iterables
        :obj:`lists` joined following the criterion of :obj:`idxs_of_lists_to_join`
        Once the new iterables have been created by joining the initial interables,
        they will be re-ordered by ascending first element

    """
    # Check that the input is an iterable of iterables
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

def assert_no_intersection(list_of_lists_of_integers, word='iterables'):
    """
    Checks if two or more lists contain the same integer
    Parameters
    ----------
    list_of_lists_of_integers : list of lists

    Returns
    -------
    Prints assertion message if inner lists have the same integer, else no output

    """
    # Looping here because the number of fragments will never become too large
    # Also of interest
    # len(f1),len(f2)
    # (196, 709)
    # %timeit np.intersect1d(f1,f2);
    # 24.4 µs ± 1.47 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    # %timeit set(f1).intersection(f2);
    # 84.9 µs ± 1.95 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    for ii, jj in _np.vstack(_np.triu_indices(len(list_of_lists_of_integers), k=1)).T:
        l1, l2 = list_of_lists_of_integers[ii], list_of_lists_of_integers[jj]
        assert len(l1) != 0 or len(l2) != 0, (l1,l2, "Both lists are empty! See https://www.coopertoons.com/education/emptyclass_intersection/emptyclass_union_intersection.html")
        assert len(_np.intersect1d(l1, l2)) == 0, 'input %s %u and %u have these elements ' \
                                                          'in common: %s:\n%s\nvs\n%s'%(word, ii, jj, set(l1).intersection(l2), l1, l2)

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

def hash_list(ilist):
    r"""Try to hash all the objects of a list (regardless of type) into one hash

    Parameters
    ----------
    iobj : anthing

    Returns : hashed object
    -------

    """
    list_of_hashes = []
    for iobj in ilist:
        try:
            res = hash(iobj)
        except:
            try:
                res = hash(tuple(iobj))
            except:
                try:
                    res = hash(tuple([hash(tuple(subitem)) for subitem in iobj]))
                except:
                    try:
                        res = hash(tuple([hash(tuple(subitem.flatten())) for subitem in iobj]))
                    except Exception as e:
                        print("Cannot hash type", type(iobj))
                        raise e
        list_of_hashes.append(res)
    return hash(tuple(list_of_hashes))

def idx_at_fraction(val_desc_order, frac):
    r"""
    Index of :obj:`val_desc_order` where np.cumsum(val)/np.sum(val)>= frac for the first time

    Parameters
    ----------
    val_desc_order : array like of floats
        The values that the determine the sum of which a fraction will be taken
        The have to be in descending order
    frac : float
        The target fraction of sum(val) that is needed

    Returns
    -------
    n : int
        Index of val where the fraction is attained for the first time.
        For the number of entries of :obj:`val`, just use n+1
    """

    assert all(_np.diff(val_desc_order)<=0), "Values must be in descending order!"
    assert 0<=frac<=1, "Fraction has to be in [0,1] ,not %s"%frac
    normalized_cumsum = (_np.cumsum(val_desc_order) / _np.sum(val_desc_order)).round(2) >= frac
    return _np.flatnonzero(normalized_cumsum>=frac)[0]


def _get_n_ctcs_from_freqs(ctc_control, ctc_freqs):
    r"""
    Helper method to understand what :obj:`ctc_control` is meant to do with :obj:`ctc_freqs`

    Provided with a set of frequencies :obj:`ctc_freqs`, the user decides how
    many them are kept by using the parameter `ctc_control`

    Parameters
    ----------
    ctc_control : int or float
        * If int:
          Keep these many contacts, i.e.
          return ctc_control directly
        * If float:
          Interpret ctcs_freq control as
          a fraction [0,1] and return
          cumsum(ctc_freqs)/sum(ctc_freqs)>= ctc_control
          That gives the number of contacts needed
          to "keep" the fraction of frequencies.
          Check :obj:`idx_at_fraction` for more info
    ctc_freqs : iterable
        Floats in descending order
    Returns
    -------
    n_ctcs : int
    or_fraction_needed : bool
        Whether an orienting-fraction value
        will be needed downstream. Is
        True if ctc_control was an integer
        and False if it was a float (if
        it was a float, we were interested in a fraction
        anyways)
    """
    or_fraction_needed = True
    total_n_ctcs = _np.array(ctc_freqs).sum()
    if isinstance(ctc_control, int):
        n_ctcs = int(ctc_control)
    else:
        if total_n_ctcs > 0:
            n_ctcs = idx_at_fraction(ctc_freqs, ctc_control) + 1
            or_fraction_needed = False
        else:
            n_ctcs = 0
    return n_ctcs, or_fraction_needed

def remove_from_lists(list_of_lists, remove_these):
    r"""
    Wraps safely around :obj:`numpy.setdiff1d` not returning empty lists

    Parameters
    ----------
    list_of_lists : iterable of iterables
    remove_these : iterable

    Returns
    -------
    clean_list : list
    """
    _fragments = [_np.setdiff1d(fr, remove_these, assume_unique=True) for fr in list_of_lists]
    return [fr.tolist() for fr in _fragments if len(fr)>0]
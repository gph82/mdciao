import numpy as _np

def is_iterable(var):
    """
    This function checks if the input is an iterable or not

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
    This function forces var to be iterable, if not already

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

# from https://www.rosettacode.org/wiki/Range_expansion#Python
def rangeexpand(txt):
    """
    This function takes in integer range or multiple integer ranges and returns a list of individual integers.
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
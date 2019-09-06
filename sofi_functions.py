import numpy as _np
import mdtraj as _md

def in_what_N_fragments(idxs, fragments):
    r"""
    For each element of idxs, return the index of "fragments" in which it appears

    :param idxs : integer, float, or iterable thereof

    :param fragments : iterable of iterables containing integers or floats

    :return : list of length len(idxs) containing an iterable with the indices of 'fragments' in which that index appears
    """

    # Sanity checking
    idxs = force_iterable(idxs)
    assert does_not_contain_strings(idxs), ("Input 'idxs' cannot contain strings")
    # TODO: consider refactoring the loop into the function call (avoid iterating over function calls)
    assert all([does_not_contain_strings(ifrag) for ifrag in fragments]), ("Input 'fragments'  cannot contain strings")

    result = (_np.vstack([_np.in1d(idxs, iseg) for ii, iseg in enumerate(fragments)])).T
    return [_np.argwhere(row).squeeze() for row in result]

def is_iterable(var):
    r"""
    Checks if var is iterable, returns True if iterable else False

    :param var : integer, float, string , list

    :return : True if var is iterable else False

    """

    try:
        for ii in var:
            return True
    except:
        return False
    return True


def force_iterable(var):
    r"""
    Forces var to be iterable, if not already

    :param var : integer, float, string , list
    :return : var as iterable

    """
    if not is_iterable(var):
        return [var]
    else:
        return var


def does_not_contain_strings(iterable):
    r"""Checks if iterable has any string element, returns False if it contains atleast one string

   :param iterable: integer, float, string or any combination thereof
   :return: True if iterable does not contain any string, else False

   """

    return all([not isinstance(ii, str) for ii in iterable])


def in_what_fragment(residx,
                     list_of_nonoverlapping_lists_of_residxs,
                     fragment_names=None):
    r"""
    For the residue id, returns the name(if provided) or the index of the "fragment"
    in which it appears

    :param residx: integer
    :param list_of_nonoverlapping_lists_of_residxs: list of integer list of non overlapping ids
    :param fragment_names: (optional) fragment names for each list in
        list_of_nonoverlapping_lists_of_residxs
    :return: integer or string if fragment name is given
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


def unique_list_of_iterables_by_tuple_hashing(ilist, return_idxs=False):
    r"""
    Returns the unique entries(if there are duplicates) from a list of iterables
    If ilist contains non-iterables, they will be considered as iterables for comparison purposes, s.t.
    1==[1]==np.array(1) and
    'A'==['A']

    :param ilist: list of iterables with redundant entries (redundant in the list, not in entries)
    :param return_idxs: boolean whether to return indices instead of unique list
    :return: list of unique iterables or indices of 'ilist' where the unique entries are
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


def exclude_same_fragments_from_residx_pairlist(pairlist,
                                                fragments,
                                                return_excluded_idxs=False):
    r"""If the members of the pair belong to the same fragment, exclude them from pairlist

    :param pairlist: list of iterables(each iterable within the list should be a pair)
    :param fragments: list of iterables
    :param return_excluded_idxs: True if index of excluded pair is needed as an output
    :return: pairs that don't belong to the same fragment, or index of the excluded pairs if return_excluded_idxs
            is True
    """

    idxs2exclude = [idx for idx, pair in enumerate(pairlist) if
                    _np.diff([in_what_fragment(ii, fragments) for ii in pair]) == 0]

    if not return_excluded_idxs:
        return [pair for ii, pair in enumerate(pairlist) if ii not in idxs2exclude]
    else:
        return idxs2exclude


# This is lifted from mdas, the original source shall remain there
def top2residue_bond_matrix(top, create_standard_bonds=True,
                            force_resSeq_breaks=False,
                            verbose=True):
    r"""
    :param top: md.Topology object
    :param create_standard_bonds: boolean. Force the method to create bonds if there are not upon reading (e.g.
    because the topology comes from a .gro-file instead of a .pdb-file.

    :return: symmetric adjacency matrix with entries ij=1 and ji=1 if there is a bond between atom i and atom j
    """
    if len(top._bonds) == 0:
        if create_standard_bonds:
            top.create_standard_bonds()
        else:
            raise ValueError("The parsed topology does not contain bonds! Aborting...")
    residue_bond_matrix = _np.zeros((top.n_residues, top.n_residues), dtype=int)
    for ibond in top._bonds:
        r1, r2 = ibond.atom1.residue.index, ibond.atom2.residue.index
        rSeq1, rSeq2 = ibond.atom1.residue.resSeq, ibond.atom2.residue.resSeq
        residue_bond_matrix[r1, r2] = 1
        residue_bond_matrix[r2, r1] = 1
        if force_resSeq_breaks and _np.abs(rSeq1 - rSeq2) > 1:  # mdtrajs bond-making routine does not check for resSeq
            residue_bond_matrix[r1, r2] = 0
            residue_bond_matrix[r2, r1] = 0
    for ii, row in enumerate(residue_bond_matrix):
        if row.sum()==0 and verbose:
            print("Residue with index %u (%s) has no bonds whatsoever"%(ii,top.residue(ii)))

    return residue_bond_matrix

def _print_frag(ii, top, iseg, **print_kwargs):
    try:
        istr = "fragment %u with %3u AAs %s(%u)-%s(%u)" % (ii, len(iseg),
                                                           top.residue(iseg[0]),

                                                           top.residue(iseg[0]).index,

                                                           top.residue(iseg[-1]),

                                                           top.residue(iseg[-1]).index)
    except:
        print(iseg)
        raise
    print(istr, **print_kwargs)


def get_fragments(top,
                  fragment_breaker_fullresname=None,
                  atoms=False,
                  join_fragments=None,
                  verbose=True,
                  auto_fragment_names=True,
                  frag_breaker_to_pick_idx=None,
                  method='resSeq', #Whatever comes after this(**) will be passed as named argument to interactive_segment_picker
                  **kwargs_interactive_segment_picker):
    r"""
    Returns the list of arrays containing the residues that are contained in a fragment

    :param top: Topology object obtained using the mdTraj module
    :param fragment_breaker_fullresname: list of full residue names, e.g. GLU30 that will be used to
            break fragments, so that [R1, R2, ... GLU30,...R10, R11] will be broken into [R1, R2, ...], [GLU30,...,R10,R11],
    :param atoms:
    :param join_fragments: List of lists of integer fragment idxs which are to be joined together.
                        Duplicate entries in any inner list will be removed
                        One fragment idx cannot appear in more than one inner list, otherwise program throws an error)

    :param verbose: Default is True
    :param method: either "resSeq" or "bonds", or "both" which will be the basis for creating fragments
    :return: List of integer array. Each array within the list has the residue ids that combine to form a fragment
    """

    # fragnames=None
    # if auto_fragment_names:
    #    fragnames=fragment_names
    old = top.residue(0).resSeq
    if method == 'resSeq':
        fragments = [[]]
        for ii, rr in enumerate(top.residues):
            delta = _np.abs(rr.resSeq - old)
            # print(delta, ii, rr, end=" ")
            if delta <= 1:
                # print("appending")
                fragments[-1].append(ii)
            else:
                # print("new")
                fragments.append([ii])
            old = rr.resSeq
    elif method in ['bonds','both']:
        from msmtools.estimation import connected_sets as _connected_sets
        if method=='bonds':
            residue_bond_matrix = top2residue_bond_matrix(top, verbose=False, force_resSeq_breaks=False)
        elif method=='both':
            residue_bond_matrix = top2residue_bond_matrix(top, verbose=False, force_resSeq_breaks=True)
        fragments = _connected_sets(residue_bond_matrix)
        fragments = [fragments[ii] for ii in _np.argsort([fr[0] for fr in fragments])]

    else:
        raise ValueError("Don't know what method '%s' is"%method)

    if verbose:
        print("Auto-detected fragments")
        for ii, iseg in enumerate(fragments):
            _print_frag(ii, top, iseg)

    if join_fragments is not None:
        # Removing the redundant entries in each list
        join_fragments = [_np.unique(jo) for jo in join_fragments]
        # Nested loops feasible here because the number of fragments will never become too large
        for ii, jo in enumerate(join_fragments):
            for jj, id in enumerate(join_fragments):
                if (ii != jj):
                    assert (len(
                        _np.intersect1d(join_fragments[ii], join_fragments[jj]))) == 0, 'join fragment id overlaps!'

        new_fragments = []
        fragment_idxs_that_where_used_for_joining = []

        for ii, jo in enumerate(join_fragments):
            # print(ii,jo)
            this_new_frag = []
            fragment_idxs_that_where_used_for_joining.extend(jo)
            for frag_idx in jo:
                # print(frag_idx)
                this_new_frag.extend(fragments[frag_idx])
            # print(this_new_frag)
            new_fragments.append(_np.array(this_new_frag))

        # fragment_idxs_that_where_used_for_joining = _np.hstack(join_orders)
        # TODO: THIS is only good programming bc the lists are very very small, otherwise np.delete is the way to go
        surviving_initial_fragments = [ifrag for ii, ifrag in enumerate(fragments)
                                       if ii not in fragment_idxs_that_where_used_for_joining]
        fragments = new_fragments + surviving_initial_fragments

        # Order wrt to the first index in each fragment
        order = _np.argsort([ifrag[0] for ifrag in fragments])
        fragments = [fragments[oo] for oo in order]

        print("Joined Fragments")
        for ii, iseg in enumerate(fragments):
            _print_frag(ii, top, iseg)

    if fragment_breaker_fullresname is not None:
        if isinstance(fragment_breaker_fullresname,str):
            fragment_breaker_fullresname=[fragment_breaker_fullresname]
        for breaker in fragment_breaker_fullresname:
            resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(breaker,fragments, top,
                                                                                      **kwargs_interactive_segment_picker)
            idx = resname2residx[breaker]
            ifrag = resname2fragidx[breaker]
            if idx is None:
                print("The fragment breaker %s appears nowhere" % breaker)
                # raise ValueError
            else:
                idx_split = _np.argwhere(idx ==_np.array(fragments[ifrag])).squeeze()
                #print('%s (index %s) found in position %s of frag %s%s' % (breaker, idx, idx_split, ifrag, fragments[ifrag]))
                subfrags = [fragments[ifrag][:idx_split], fragments[ifrag][idx_split:]]
                print("New fragments after breaker %s:" % breaker)
                if idx_split>0:
                    #print("now breaking up into", subfrags)
                    fragments = fragments[:ifrag] + subfrags + fragments[ifrag + 1:]
                    for ii, ifrag in enumerate(fragments):
                        _print_frag(ii, top, ifrag)
                else:
                    print("Not using it since it already was a fragment breaker in frag %s"%ifrag)
            print()
            # raise ValueError(idx)

    if not atoms:
        return fragments
    else:
        return [_np.hstack([[aa.index for aa in top.residue(ii).atoms]])]


def find_AA(top, AA):
    r"""
    This function is used to query the index of residue based on residue name

    :param top: :obj:`mdtraj.Topology` object
    :param AA: Valid residue name to be passed as a string, example "GLU30" or "E30"
    :return: list of res_idxs where the residue is present, so that top.residue(idx) would return the wanted AA
    """
    code = ''.join([ii for ii in AA if ii.isalpha()])
    if len(code)==1:
        return [rr.index for rr in top.residues if AA == '%s%u' % (rr.code, rr.resSeq)]
    elif len(code)==3:
        return [rr.index for rr in top.residues if AA == '%s%u' % (rr.name, rr.resSeq)]
    else:
        raise ValueError("The input AA %s must have an alphabetic code of either 3 or 1 letters"%AA)


def interactive_fragment_picker_by_AAresSeq(AAresSeq_idxs, fragments, top,
                                            default_fragment_idx=None,
                                            fragment_names=None, extra_string_info=''):
    r"""
    This function returns the fragment idxs and the residue idxs based on residue name.
    If a residue is present in multiple fragments, the function asks the user to choose the fragment, for which
    the residue idxs is reporte

    :param AAresSeq_idxs: string or list of of strings
           AAs of the form of "GLU30" or "E30", can be mixed
    :param fragments: iterable of iterables of integers
            The integers in the iterables of 'fragments' represent residue indices of that fragment
    :param top: mdtraj.Topology object
    :param default_fragment_idx: None or integer.
            Pick this fragment withouth asking in case of ambiguity. If None, the user will we prompted
    :param fragment_names: list of strings providing informative names for the input fragments
    :param extra_string_info: string with any additional info to be printed in case of ambiguity
    :return: two dictionaries, residuenames2residxs and residuenames2fragidxs. If the AA is not found then the
                dictionaries for that key contain None, e.g. residuenames2residxs[notfoundAA]=None
    """
    residuenames2residxs = {}
    residuenames2fragidxs = {}
    last_answer = 0

    #TODO break the iteration in this method into a separate method. Same AAcode in different fragments will overwrite
    # each other
    if isinstance(AAresSeq_idxs, str):
        AAresSeq_idxs = [AAresSeq_idxs]

    for key in AAresSeq_idxs:
        cands = find_AA(top, key)
        cand_fragments = in_what_N_fragments(cands, fragments)
        # TODO OUTSOURCE THIS?
        if len(cands) == 0:
            print("No residue found with resSeq %s"%key)
            residuenames2residxs[key] = None
            residuenames2fragidxs[key] = None
        else:
            if len(cands) == 1:
                cands = cands[0]
                answer = cand_fragments[0]
                # print(key,refgeom.top.residue(cands[0]), cand_fragments)
            elif len(cands) > 1:
                istr = "ambigous definition for AA %s" % key
                istr += extra_string_info
                print(istr)
                for cc, ss in zip(cands, cand_fragments):
                    istr = '%6s in fragment %2u with residue index %2u'%(top.residue(cc), ss, cc)
                    if fragment_names is not None:
                        istr += ' (%s)'%fragment_names[ss]
                    print(istr)
                if default_fragment_idx is None:
                    answer = input(
                        "input one fragment idx (out of %s) and press enter. Leave empty and hit enter to repeat last option [%s]\n" % ([int(cf) for cf in cand_fragments], last_answer))
                    if len(answer) == 0:
                        answer = last_answer
                    try:
                        answer = int(answer)
                        assert answer in cand_fragments, "The answer '%s' is not in the candidate fragments %s"%(answer,cand_fragments)
                    except (ValueError, AssertionError):
                        print( "Your answer has to be an integer "
                                "in the of the fragment list %s, but you gave %s" % ([int(cf) for cf in cand_fragments],answer))
                        raise
                    cands = cands[_np.argwhere([answer == ii for ii in cand_fragments]).squeeze()]
                    last_answer = answer

                elif isinstance(default_fragment_idx,int):
                    try:
                        assert default_fragment_idx in cand_fragments, "The answer '%s' is not in the candidate fragments %s"%(default_fragment_idx,cand_fragments)
                    except AssertionError:
                        print( "Your answer has to be an integer "
                                "in the of the fragment list %s, but you gave %s" % ([int(cf) for cf in cand_fragments], default_fragment_idx))
                        raise
                    # cands = default_fragment_idx
                    # answer = cand_fragments[_np.argwhere(cands==default_fragment_idx).squeeze()]
                    answer = default_fragment_idx
                    cands = cands[_np.argwhere([answer == ii for ii in cand_fragments]).squeeze()]

                    print("Automatically picked fragment %u"%default_fragment_idx)
                # print(refgeom.top.residue(cands))
                print()

            residuenames2residxs[key] = int(cands)  # this should be an integer
            residuenames2fragidxs[key] = int(answer) #ditto

    return residuenames2residxs, residuenames2fragidxs

def int_from_AA_code(key):
    return int(''.join([ii for ii in key if ii.isnumeric()]))

#FIXME Trying to integrate both AAresseq and resseq functions in one, if Guillermo agrees, add the test also
def interactive_fragment_picker_wip(AAresSeq_idxs, fragments, top,
                                            default_fragment_idx=None,
                                            fragment_names=None, extra_string_info=''):
    r"""
    This function returns the fragment idxs and the residue idxs based on residue name.
    If a residue is present in multiple fragments, the function asks the user to choose the fragment, for which
    the residue idxs is reporte

    :param AAresSeq_idxs: string or list of of strings
           AAs of the form of "GLU30" or "E30", can be mixed
    :param fragments: iterable of iterables of integers
            The integers in the iterables of 'fragments' represent residue indices of that fragment
    :param top: mdtraj.Topology object
    :param default_fragment_idx: None or integer.
            Pick this fragment withouth asking in case of ambiguity. If None, the user will we prompted
    :param fragment_names: list of strings providing informative names for the input fragments
    :param extra_string_info: string with any additional info to be printed in case of ambiguity
    :return: two dictionaries, residuenames2residxs and residuenames2fragidxs. If the AA is not found then the
                dictionaries for that key contain None, e.g. residuenames2residxs[notfoundAA]=None
    """
    residuenames2residxs = {}
    residuenames2fragidxs = {}
    last_answer = 0

    #TODO break the iteration in this method into a separate method. Same AAcode in different fragments will overwrite
    # each other
    if isinstance(AAresSeq_idxs, (str, int)):
        AAresSeq_idxs = [AAresSeq_idxs]

    for key in AAresSeq_idxs:

        if isinstance(key, str):
            cands = find_AA(top, key)
        elif isinstance(key, int):
            cands = [rr.index for rr in top.residues if rr.resSeq == key]

        cand_fragments = in_what_N_fragments(cands, fragments)
        # TODO OUTSOURCE THIS?
        if len(cands) == 0:
            print("No residue found with resSeq %s"%key)
            residuenames2residxs[key] = None
            residuenames2fragidxs[key] = None
        else:
            if len(cands) == 1:
                cands = cands[0]
                answer = cand_fragments[0]
                # print(key,refgeom.top.residue(cands[0]), cand_fragments)
            elif len(cands) > 1:
                istr = "ambigous definition for AA %s" % key
                istr += extra_string_info
                print(istr)
                for cc, ss in zip(cands, cand_fragments):
                    istr = '%6s in fragment %2u with residue index %2u'%(top.residue(cc), ss, cc)
                    if fragment_names is not None:
                        istr += ' (%s)'%fragment_names[ss]
                    print(istr)
                if default_fragment_idx is None:
                    answer = input(
                        "input one fragment idx (out of %s) and press enter. Leave empty and hit enter to repeat last option [%s]\n" % ([int(cf) for cf in cand_fragments], last_answer))
                    if len(answer) == 0:
                        answer = last_answer
                    try:
                        answer = int(answer)
                        assert answer in cand_fragments, "The answer '%s' is not in the candidate fragments %s"%(answer,cand_fragments)
                    except (ValueError, AssertionError):
                        print( "Your answer has to be an integer "
                                "in the of the fragment list %s, but you gave %s" % ([int(cf) for cf in cand_fragments],answer))
                        raise
                    cands = cands[_np.argwhere([answer == ii for ii in cand_fragments]).squeeze()]
                    last_answer = answer

                elif isinstance(default_fragment_idx,int):
                    try:
                        assert default_fragment_idx in cand_fragments, "The answer '%s' is not in the candidate fragments %s"%(default_fragment_idx,cand_fragments)
                    except AssertionError:
                        print( "Your answer has to be an integer "
                                "in the of the fragment list %s, but you gave %s" % ([int(cf) for cf in cand_fragments], default_fragment_idx))
                        raise
                    # cands = default_fragment_idx
                    # answer = cand_fragments[_np.argwhere(cands==default_fragment_idx).squeeze()]
                    answer = default_fragment_idx
                    cands = cands[_np.argwhere([answer == ii for ii in cand_fragments]).squeeze()]

                    print("Automatically picked fragment %u"%default_fragment_idx)
                # print(refgeom.top.residue(cands))
                print()

            residuenames2residxs[key] = int(cands)  # this should be an integer
            residuenames2fragidxs[key] = int(answer) #ditto

    return residuenames2residxs, residuenames2fragidxs

def bonded_neighborlist_from_top(top, n=1):
    """TODO: description of method in one line

    :param top: mdTraj Topology object
    :param n: number of bonds between bonded neighbors
    :return: neighbor of each residue as a list of list
            Each residue will have a corresponding neighbor list(if neighbors exists), or an empty list(if no neighbor exists)
            A neighbor exists between i and j residues if residue_bond_matrix has a 1 at position ij
    """
    residue_bond_matrix = top2residue_bond_matrix(top)
    neighbor_list = [[ii] for ii in range(residue_bond_matrix.shape[0])]
    for kk in range(n):
        for ridx, ilist in enumerate(neighbor_list):
            new_neighborlist = [ii for ii in ilist]
            #print("Iteration %u in residue %u"%(kk, ridx))
            for rn in ilist:
                row = residue_bond_matrix[rn]
                bonded = _np.argwhere(row == 1).squeeze()
                if _np.ndim(bonded)==0:
                    bonded=[bonded]
                toadd = [nn for nn in bonded if nn not in ilist and nn!=ridx]
                if len(toadd):
                    #print("neighbor %u adds new neighbor %s:"%(rn, toadd))
                    new_neighborlist += toadd
                    #print("so that the new neighborlist is: %s"%new_neighborlist)

            neighbor_list[ridx] = [ii for ii in _np.unique(new_neighborlist) if ii!=ridx]
            #break

    # Check that the neighborlist works both ways
    for ii, ilist in enumerate(neighbor_list):
        for nn in ilist:
            assert ii in neighbor_list[nn]

    return neighbor_list

# from https://www.rosettacode.org/wiki/Range_expansion#Python
def rangeexpand(txt):
    """
    This function takes in integer range or multiple integer ranges and returns a list of individual integers
    Example- "1-2,3-4" will return [1,2,3,4]

    :param txt: string of integers or integer range separated by ","
    :return: list of integers
    """
    lst = []
    for r in txt.split(','):
        if '-' in r[1:]:
            r0, r1 = r[1:].split('-', 1)
            lst += range(int(r[0] + r0), int(r1) + 1)
        else:
            lst.append(int(r))
    return lst


def ctc_freq_reporter_by_residue_neighborhood(ctcs_mean, resSeq2residxs, fragments, ctc_residxs_pairs, top,
                                              n_ctcs=5, select_by_resSeq=None,
                                              silent=False,
                                              ):
    """TODO one line description of method

    Parameters
    ----------
    ctcs_mean
    resSeq2residxs
    fragments
    ctc_residxs_pairs
    top
    n_ctcs
    select_by_resSeq
    silent

    Returns
    -------

    """
    order = _np.argsort(ctcs_mean)[::-1]
    assert len(ctcs_mean)==len(ctc_residxs_pairs)
    final_look = {}
    if select_by_resSeq is None:
        select_by_resSeq=list(resSeq2residxs.keys())
    elif isinstance(select_by_resSeq, int):
        select_by_resSeq=[select_by_resSeq]
    for key, val in resSeq2residxs.items():
        if key in select_by_resSeq:
            order_mask = _np.array([ii for ii in order if val in ctc_residxs_pairs[ii]])
            print("#idx    Freq  contact             segA-segB residxA   residxB   ctc_idx")

            isum=0
            seen_ctcs = []
            for ii, oo in enumerate(order_mask[:n_ctcs]):
                pair = ctc_residxs_pairs[oo]
                if pair[0]!=val and pair[1]==val:
                    pair=pair[::-1]
                elif pair[0]==val and pair[1]!=val:
                    pass
                else:
                    print(pair)
                    raise Exception
                idx1 = pair[0]
                idx2 = pair[1]
                s1 = in_what_fragment(idx1, fragments)
                s2 = in_what_fragment(idx2, fragments)
                imean = ctcs_mean[oo]
                isum += imean
                seen_ctcs.append(imean)
                print("%-6s %3.2f %8s-%-8s    %5u-%-5u %7u %7u %7u %3.2f" % ('%u:'%(ii+1), imean, top.residue(idx1), top.residue(idx2), s1, s2, idx1, idx2, oo, isum))
            if not silent:
                try:
                    answer = input("How many do you want to keep (Hit enter for None)?\n")
                except KeyboardInterrupt:
                    break
                if len(answer) == 0:
                    pass
                else:
                    answer = _np.arange(_np.min((int(answer),n_ctcs)))
                    final_look[val] = order_mask[answer]
            else:
                seen_ctcs = _np.array(seen_ctcs)
                n_nonzeroes = (seen_ctcs>0).astype(int).sum()
                answer=_np.arange(_np.min((n_nonzeroes,n_ctcs)))
                final_look[val]= order_mask[answer]

    # TODO think about what's best to return here
    return final_look
    # These were moved from the method to the API
    final_look = _np.unique(_np.hstack(final_look))
    final_look = final_look[_np.argsort(ctcs_mean[final_look])][::-1]
    return final_look


def table2BW_by_AAcode(tablefile="GPCRmd_B2AR_nomenclature.xlsx",
                       modifications={"S262":"F264"},
                       keep_AA_code=True,
                       return_defs=False,
                       ):
    """

    :param tablefile: GPCRmd_B2AR nomenclature file in excel format
    :param modifications: Dictionary to store the modifications required in amino acid name
                        Parameter should be passed as a dictionary of the form {old name:new name}
    :param keep_AA_code: True if amino acid letter code is required else False. Default is True
                        If True then output dictionary will have key of the form "Q26" else "26"
    :param return_defs: if defs are required then True else False. Default is true
    :return: Dictionary if return_defs=false else dictionary and a list
    """
    out_dict = {}
    import pandas
    df = pandas.read_excel(tablefile, header=None)

    # Locate definition lines and use their indices
    defs = []
    for ii, row in df.iterrows():
        if row[0].startswith("TM") or row[0].startswith("H8"):
            defs.append(row[0])

        else:
            out_dict[row[2]] = row[1]

    # Replace some keys
    __ = {}
    for key, val in out_dict.items():
        for patt, sub in modifications.items():
            key = key.replace(patt,sub)
        __[key] = str(val)
    out_dict = __

    # Make proper BW notation as string with trailing zeros
    out_dict = {key:'%1.2f'%float(val) for key, val in out_dict.items()}

    if keep_AA_code:
        pass
    else:
        out_dict =  {int(key[1:]):val for key, val in out_dict.items()}

    if return_defs:
        return out_dict, defs
    else:
        return out_dict


def guess_missing_BWs(input_BW_dict,top, restrict_to_residxs=None):

    guessed_BWs = {}
    if restrict_to_residxs is None:
        restrict_to_residxs = [residue.index for residue in top.residues]

    """
    seq = ''.join([top._residues    [ii].code for ii in restrict_to_residxs])
    seq_BW =  ''.join([key[0] for key in input_BW_dict.keys()])
    ref_seq_idxs = [int_from_AA_code(key) for key in input_BW_dict.keys()]
    for alignmt in pairwise2.align.globalxx(seq, seq_BW)[:1]:
        alignment_dict = alignment_result_to_list_of_dicts(alignmt, top,
                                                            ref_seq_idxs,
                                                            #res_top_key="target_code",
                                                           #resname_key='target_resname',
                                                           #resSeq_key="target_resSeq",
                                                           #idx_key='ref_resSeq',
                                                           #re_merge_skipped_entries=False
                                                            )
        print(alignment_dict)
    return
    """
    out_dict = {ii:None for ii in range(top.n_residues)}
    for rr in restrict_to_residxs:
        residue = top.residue(rr)
        key = '%s%s'%(residue.code,residue.resSeq)
        try:
            (key, input_BW_dict[key])
            #print(key, input_BW_dict[key])
            out_dict[residue.index] = input_BW_dict[key]
        except KeyError:
            resSeq = int_from_AA_code(key)
            try:
                key_above = [key for key in input_BW_dict.keys() if int_from_AA_code(key)>resSeq][0]
                resSeq_above = int_from_AA_code(key_above)
                delta_above = int(_np.abs([resSeq - resSeq_above]))
            except IndexError:
                delta_above = 0
            try:
                key_below = [key for key in input_BW_dict.keys() if int_from_AA_code(key)<resSeq][-1]
                resSeq_below = int_from_AA_code(key_below)
                delta_below = int(_np.abs([resSeq-resSeq_below]))
            except IndexError:
                delta_below = 0

            if delta_above<=delta_below:
                closest_BW_key = key_above
                delta = -delta_above
            elif delta_above>delta_below:
                closest_BW_key = key_below
                delta = delta_below
            else:
                print(delta_above, delta_below)
                raise Exception

            if residue.index in restrict_to_residxs:
                closest_BW=input_BW_dict[closest_BW_key]
                base, exp = [int(ii) for ii in closest_BW.split('.')]
                new_guessed_val = '%s.%u*'%(base,exp+delta)
                #guessed_BWs[key] = new_guessed_val
                out_dict[residue.index] = new_guessed_val
                #print(key, new_guessed_val, residue.index, residue.index in restrict_to_residxs)
            else:
                pass
                #new_guessed_val = None

            # print("closest",closest_BW_key,closest_BW, key, new_guessed_val )

    #input_BW_dict.update(guessed_BWs)

    return out_dict

class CGN_transformer(object):
    def __init__(self, ref_PDB='3SN6'):
        # Create dataframe with the alignment
        from pandas import read_table as _read_table
        self._ref_PDB = ref_PDB

        self._DF = _read_table('CGN_%s.txt'%ref_PDB)
        #TODO find out how to properly do this with pandas

        self._dict = {key: self._DF[self._DF[ref_PDB] == key]["CGN"].to_list()[0] for key in self._DF[ref_PDB].to_list()}

        self._top =_md.load(ref_PDB+'.pdb').top
        seq_ref = ''.join([str(rr.code).replace("None","X") for rr in self._top.residues])[:len(self._dict)]
        seq_idxs = _np.hstack([rr.resSeq for rr in self._top.residues])[:len(self._dict)]
        keyval = [{key:val} for key,val in self._dict.items()]
        #for ii, (iseq_ref, iseq_idx) in enumerate(zip(seq_ref, seq_idxs)):
        #print(ii, iseq_ref, iseq_idx )

        self._seq_ref  = seq_ref
        self._seq_idxs = seq_idxs

    @property
    def seq(self):
        return self._seq_ref

    @property
    def seq_idxs(self):
        return self._seq_idxs

    @property
    def AA2CGN(self):
        return self._dict

        #return seq_ref, seq_idxs, self._dict


def top2CGN_by_AAcode(top, ref_CGN_tf, keep_AA_code=True,
                      restrict_to_residxs=None):

    # TODO this lazy import will bite back
    from Gunnar_utils import alignment_result_to_list_of_dicts
    from Bio import pairwise2


    if restrict_to_residxs is None:
        restrict_to_residxs = [residue.index for residue in top.residues]

    #out_dict = {ii:None for ii in range(top.n_residues)}
    #for ii in restrict_to_residxs:
    #    residue = top.residue(ii)
    #    AAcode = '%s%s'%(residue.code,residue.resSeq)
    #    try:
    #        out_dict[ii]=ref_CGN_tf.AA2CGN[AAcode]
    #    except KeyError:
    #        pass
    #return out_dict
    seq = ''.join([str(top.residue(ii).code).replace("None", "X") for ii in restrict_to_residxs])
    #
    res_idx2_PDB_resSeq = {}
    for alignmt in pairwise2.align.globalxx(seq, ref_CGN_tf.seq)[:1]:
        list_of_alignment_dicts = alignment_result_to_list_of_dicts(alignmt, top,
                                                            ref_CGN_tf.seq_idxs,
                                                            res_top_key="Nour_code",
                                                            resname_key='Nour_resname',
                                                            resSeq_key="Nour_resSeq",
                                                            res_ref_key='3SN6_code',
                                                            idx_key='3SN6_resSeq',
                                                            subset_of_residxs=restrict_to_residxs,
                                                            #re_merge_skipped_entries=False
                                                           )

        #import pandas as pd
        #with pd.option_context('display.max_rows', None, 'display.max_columns',
        #                       None):  # more options can be specified also
        #    for idict in list_of_alignment_dicts:
        #        idict["match"] = False
        #        if idict["Nour_code"]==idict["3SN6_code"]:
        #            idict["match"]=True
        #    print(DataFrame.from_dict(list_of_alignment_dicts))

        res_idx_array=iter(restrict_to_residxs)
        for idict in list_of_alignment_dicts:
             if '~' not in idict["Nour_resname"]:
                 idict["target_residx"]=\
                 res_idx2_PDB_resSeq[next(res_idx_array)]='%s%s'%(idict["3SN6_code"],idict["3SN6_resSeq"])
    out_dict = {}
    for ii in range(top.n_residues):
        try:
            out_dict[ii] = ref_CGN_tf.AA2CGN[res_idx2_PDB_resSeq[ii]]
        except KeyError:
            out_dict[ii] = None

    return out_dict
    # for key, equiv_at_ref_PDB in res_idx2_PDB_resSeq.items():
    #     if equiv_at_ref_PDB in ref_CGN_tf.AA2CGN.keys():
    #         iCGN = ref_CGN_tf.AA2CGN[equiv_at_ref_PDB]
    #     else:
    #         iCGN = None
    #     #print(key, top.residue(key), iCGN)
    #     out_dict[key]=iCGN
    # if keep_AA_code:
    #     return out_dict
    # else:
    #     return {int(key[1:]):val for key, val in out_dict.items()}


def xtcs2ctcs(xtcs, top, ctc_residxs_pairs, stride=1,consolidate=True,
              chunksize=1000, return_time=False, c=True):
    ctcs = []
    print()
    times = []
    inform = lambda ixtc, ii, running_f : print("Analysing %20s in chunks of "
                                                "%3u frames. chunks %4u frames %8u" %
                                                (ixtc, chunksize, ii, running_f), end="\r", flush=True)
    for ii, ixtc in enumerate(xtcs):
        ictcs = []
        running_f = 0
        inform(ixtc, 0, running_f)
        itime = []
        for jj, igeom in enumerate(_md.iterload(ixtc, top=top, stride=stride,
                                                chunk=_np.round(chunksize/stride)
                                   )):
            running_f += igeom.n_frames
            inform(ixtc, jj, running_f)
            itime.append(igeom.time)
            ictcs.append(_md.compute_contacts(igeom, ctc_residxs_pairs)[0])
            #if jj==10:
            #    break

        times.append(_np.hstack(itime))
        ictcs = _np.vstack(ictcs)
        #print("\n", ii, ictcs.shape, "shape ictcs")
        ctcs.append(ictcs)
        print()

    if consolidate:
        try:
            actcs = _np.vstack(ctcs)
            times = _np.hstack(times)
        except ValueError as e:
            print(e)
            print([_np.shape(ic) for ic in ctcs])
            raise
    else:
        actcs = ctcs
        times = times

    if not return_time:
        return actcs
    else:
        return actcs, times









import numpy as _np


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
    r"""
   Checks if iterable has any string element, returns False if it contains atleast one string

   :param iterable : integer, float, string or any combination thereof
   :return : True if iterable does not contain any string, else False

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
    r"""
    If the members of the pair belong to the same fragment, exclude them from pairlist
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


def bonded_neighborlist_from_top(top, n=1):
    residue_bond_matrix = top2residue_bond_matrix(top)
    neighbor_list = [[ii] for ii in range(residue_bond_matrix.shape[0])]
    for kk in range(n):
        for ridx, ilist in enumerate(neighbor_list):
            new_neighborlist = [ii for ii in ilist]
            # print("Iteration %u in residue %u"%(kk, ridx))
            for rn in ilist:
                row = residue_bond_matrix[rn]
                bonded = _np.argwhere(row == 1).squeeze()
                if _np.ndim(bonded) == 0:
                    bonded = [bonded]
                toadd = [nn for nn in bonded if nn not in ilist and nn != ridx]
                if len(toadd):
                    # print("neighbor %u adds new neighbor %s:"%(rn, toadd))
                    new_neighborlist += toadd
                    # print("so that the new neighborlist is: %s"%new_neighborlist)

            neighbor_list[ridx] = [ii for ii in _np.unique(new_neighborlist) if ii != ridx]
            # break

    # Check that the neighborlist works both ways
    for ii, ilist in enumerate(neighbor_list):
        for nn in ilist:
            assert ii in neighbor_list[nn]

    return neighbor_list


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

    :param top: Topology object obtained using the mdTraj module
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





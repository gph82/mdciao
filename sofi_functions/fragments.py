import numpy as _np
from .aa_utils import find_AA
from .bond_utils import top2residue_bond_matrix
from .list_utils import in_what_N_fragments as _in_what_N_fragments, join_lists as _join_lists

def _print_frag(frag_idx, top, fragment, fragment_desc='fragment',
                return_string=False, **print_kwargs):
    # TODO document
    """

    Parameters
    ----------
    frag_idx
    top
    fragment
    print_kwargs

    Returns
    -------

    """
    try:
        istr = "%s %6s with %3u AAs %7s(%4u)-%-7s(%-4u)" % (fragment_desc, str(frag_idx), len(fragment),
                                                           top.residue(fragment[0]),
                                                           top.residue(fragment[0]).index,
                                                           top.residue(fragment[-1]),
                                                           top.residue(fragment[-1]).index)
    except:
        print(fragment)
        raise
    if return_string:
        return istr
    else:
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
    """
    Returns the list of arrays containing the residues that are contained in a fragment

    Parameters
    ----------
    top : :py:class:`mdtraj.Topology`
    fragment_breaker_fullresname : list
        list of full residue names. Example - GLU30 that will be used to break fragments,
        so that [R1, R2, ... GLU30,...R10, R11] will be broken into [R1, R2, ...], [GLU30,...,R10,R11]
    atoms : boolean, optional
    join_fragments : list of lists
        List of lists of integer fragment idxs which are to be joined together.
        Duplicate entries in any inner list will be removed.
        One fragment idx cannot appear in more than one inner list, otherwise program throws an error.
    verbose : boolean, optional
    auto_fragment_names : not used in the function
    frag_breaker_to_pick_idx : not used in the function
    method : 'resSeq' or or 'bonds' (or 'both') or 'chains'
        The method passed will be the basis for creating fragments
    kwargs_interactive_segment_picker : optional
        additional arguments

    Returns
    -------
    List of integer array
        Each array within the list has the residue ids that combine to form a fragment.

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
    elif method in ["chains"]:
        fragments = [[rr.index for rr in ichain.residues] for ichain in top.chains]
    else:
        raise ValueError("Don't know what method '%s' is"%method)

    # Inform of the first result
    if verbose:
        print("Auto-detected fragments")
        for ii, iseg in enumerate(fragments):
            _print_frag(ii, top, iseg)
    # Join if necessary
    if join_fragments is not None:
        fragments = _join_lists(fragments, join_fragments)
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
    :param top: :py:class:`mdtraj.Topology`
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
        cand_fragments = _in_what_N_fragments(cands, fragments)
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
                    istr = '%6s (res_idx %2u) in %s'%(top.residue(cc), cc, _print_frag(ss, top, fragments[ss], return_string=True))

                    if fragment_names is not None:
                        istr += ' (%s)'%fragment_names[ss]
                    print(istr)
                if default_fragment_idx is None:
                    answer = input(
                        "input one fragment idx (out of %s) and press enter.\nLeave empty and hit enter to repeat last option [%s]\n" % ([int(cf) for cf in cand_fragments], last_answer))
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

#FIXME Trying to integrate both AAresseq and resseq functions in one, if Guillermo agrees, add the test also
def interactive_fragment_picker_wip(AAresSeq_idxs, fragments, top,
                                            default_fragment_idx=None,
                                            fragment_names=None, extra_string_info=''):
    r"""
    This function returns the fragment idxs and the residue idxs based on residue name/residue index.
    If a residue is present in multiple fragments, the function asks the user to choose the fragment, for which
    the residue idxs is reporte

    :param AAresSeq_idxs: string or list of of strings
           AAs of the form of "GLU30" or "E30" or 30, can be mixed
    :param fragments: iterable of iterables of integers
            The integers in the iterables of 'fragments' represent residue indices of that fragment
    :param top: :py:class:`mdtraj.Topology`
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

        cand_fragments = _in_what_N_fragments(cands, fragments)
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





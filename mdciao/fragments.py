import numpy as _np
from .residue_and_atom_utils import find_AA
from .bond_utils import top2residue_bond_matrix
from .list_utils import in_what_N_fragments as _in_what_N_fragments, join_lists as _join_lists

def _print_frag(frag_idx, top, fragment, fragment_desc='fragment',
                return_string=False, **print_kwargs):
    """
    For pretty-printing of fragments of an :obj:`mtraj.topology`

    Parameters
    ----------
    frag_idx: int
        Index of the fragment to be printed
    top: :obj:`mdtraj.Topology`
        Topology in which the fragment appears
    fragment: iterable of indices
        The fragment in question, with zero-indexed residue indices
    fragment_desc: str, default is "fragment"
        Who to call the fragments, e.g. segment, block, monomer, chain
    return_string: bool, default is False
        Instead of printing, return the string
    print_kwargs:
        Optional keyword arguments to pass to the print function, e.g. "end=","" and such

    Returns
    -------
    None or str, see return_string option

    """
    try:
        istr = "%s %6s with %3u AAs %7s(%4u)-%-7s(%-4u) (%s) " % (fragment_desc, str(frag_idx), len(fragment),
                                                                   top.residue(fragment[0]),
                                                                   top.residue(fragment[0]).index,
                                                                   top.residue(fragment[-1]),
                                                                   top.residue(fragment[-1]).index,
                                                                   str(frag_idx))
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
                  verbose=True,
                  method='resSeq', #Whatever comes after this(**) will be passed as named argument to interactive_segment_picker
                  join_fragments=None,
                  **kwargs_interactive_segment_picker):
    """
    Given an :obj:`mdtraj.Topology` return its residues grouped into fragments using different methods.

    Parameters
    ----------
    top : :py:class:`mdtraj.Topology`
    fragment_breaker_fullresname : list
        list of full residue names. Example - GLU30 that will be used to break fragments,
        so that [R1, R2, ... GLU30,...R10, R11] will be broken into [R1, R2, ...], [GLU30,...,R10,R11]
    atoms : boolean, optional
        Instead of returning residue indices, retun atom indices
    join_fragments : list of lists
        List of lists of integer fragment idxs which are to be joined together.
        (After splitting them according to "methods")
        Duplicate entries in any inner list will be removed.
        One fragment idx cannot appear in more than one inner list, otherwise program throws an error.
    verbose : boolean, optional
    method : str, default is 'resSeq'
        The method passed will be the basis for creating fragments. Check the following options
        with the example sequence "…-A27,Lig28,K29-…-W40,D45-…-W50,GDP1"
        - 'resSeq'
            breaks at jumps in resSeq entry: […A27,Lig28,K29,…,W40],[D45,…,W50],[GDP1]
        - 'resSeq+'
            breaks only at negative jumps in resSeq: […A27,Lig28,K29,…,W40,D45,…,W50],[GDP1]
        - ‘bonds’
            breaks when AAs are not connected by bonds, ignores resSeq: […A27][Lig28],[K29,…,W40],[D45,…,W50],[GDP1]
        - 'resSeq_bonds'
            breaks at resSeq jumps and at missing bonds
        - 'chains'
            breaks into chains of the PDB file/entry

    kwargs_interactive_segment_picker : optional
        additional arguments

    Returns
    -------
    List of integer array
        Each array within the list has the residue indices of each fragment.
        These fragments do not have overlap. Their union contains all indices

    """

    # Auto detect fragments by resSeq
    old = top.residue(0).resSeq
    fragments_resSeq = [[]]
    for ii, rr in enumerate(top.residues):
        delta = _np.abs(rr.resSeq - old)
        # print(delta, ii, rr, end=" ")
        if delta <= 1:
            # print("appending")
            fragments_resSeq[-1].append(ii)
        else:
            # print("new")
            fragments_resSeq.append([ii])
        old = rr.resSeq

    from msmtools.estimation import connected_sets as _connected_sets
    if method=="resSeq":
        fragments = fragments_resSeq
    elif method=='resSeq_bonds':
        residue_bond_matrix = top2residue_bond_matrix(top, verbose=False, force_resSeq_breaks=True)
        fragments = _connected_sets(residue_bond_matrix)
    elif method=='bonds':
        residue_bond_matrix = top2residue_bond_matrix(top, verbose=False, force_resSeq_breaks=False)
        fragments = _connected_sets(residue_bond_matrix)
        fragments = [fragments[ii] for ii in _np.argsort([fr[0] for fr in fragments])]
    elif method == "chains":
        fragments = [[rr.index for rr in ichain.residues] for ichain in top.chains]
    elif method == "resSeq+":
        to_join = [[0]]
        for ii, ifrag in enumerate(fragments_resSeq[:-1]):
            r1 = top.residue(ifrag[-1])
            r2 = top.residue(fragments_resSeq[ii + 1][0])
            if r1.resSeq < r2.resSeq:
                to_join[-1].append(ii + 1)
            else:
                to_join.append([ii + 1])

        if False:
            print("Fragments by ascending resSeq")
            for idx, tj in enumerate(to_join):
                for ii in tj:
                    istr = _print_frag(idx, top, fragments_resSeq[ii],
                                       return_string=True)
                    print(istr)
                print(''.join(["-" for __ in range(len(istr))]))
        fragments = _join_lists(fragments_resSeq, [tj for tj in to_join if len(tj) > 1])

    # TODO check why this is not equivalent to "bonds" in the test_file
    elif method == 'molecules':
        raise NotImplementedError("method 'molecules' is not implemented yet")
        #fragments = [_np.unique([aa.residue.index for aa in iset]) for iset in top.find_molecules()]
    elif method == 'molecules_resSeq+':
        raise NotImplementedError("method 'molecules_resSeq+' is not implemented yet")
        """
        _molecules = top.find_molecules()
        _fragments = [_np.unique([aa.residue.index for aa in iset]) for iset in _molecules]
        _resSeqs =   [[top.residue(idx).resSeq for idx in ifrag] for ifrag in _fragments]
        _negjumps =  [_np.diff(_iresSeq) for _iresSeq in _resSeqs]
        fragments = []
        for ii, (ifrag, ijumps) in enumerate(zip(_fragments, _negjumps)):
            if all(ijumps>0):
                fragments.append(ifrag)
            else:
                offset = 0
                for kk, jj in enumerate(_np.argwhere(ijumps<0)):
                    print(ii, jj)
                    jj = int(jj)
                    #print(top.residue(ifrag[jj]), top.residue(ifrag[jj+1]))
                    offset+=jj
                    fragments.append(ifrag[:jj+1])
                    fragments.append(ifrag[jj+1:])
                    if kk>1:
                        raise Exception
        #print(_negjumps)
        """

    else:
        raise ValueError("Don't know what method '%s' is"%method)

    # Inform of the first result
    if verbose:
        print("Auto-detected fragments with method %s"%str(method))
        for ii, iseg in enumerate(fragments):
            end='\n'
            ri, rj = [top.residue(ii) for ii in [iseg[0],iseg[-1]]]
            if rj.resSeq-ri.resSeq!=len(iseg)-1:
                #print(ii, rj.resSeq-ri.resSeq, len(iseg)-1)
                end='resSeq jumps\n'
            _print_frag(ii, top, iseg, end=end)
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
        return [_np.hstack([[aa.index for aa in top.residue(ii).atoms] for ii in frag]) for frag in fragments]

_allowed_fragment_methods = ['resSeq',
                            'resSeq+',
                            'bonds',
                         #   'molecules',
                            'resSeq_bonds',
                            'chains']
def overview(topology,
             methods=['all']):

    """
    Prints the fragments created and their corresponding methods

    Parameters
    ----------
    topology :  :py:class:`mdtraj.Topology`
    methods : str or list of strings
                method(s) to be used for obtaining fragments


    Returns
    -------
    None
    prints the output from the get_fragments(), using the specified method(s)

    """

    if isinstance(methods,str):
        methods = [methods]

    if methods[0].lower() == 'all':
        try_methods = _allowed_fragment_methods
    else:
        for imethd in methods:
            assert imethd in _allowed_fragment_methods, ('input method %s is not known. ' \
                                           'Know methods are %s ' % (imethd, _allowed_fragment_methods))
        try_methods = methods

    for method in try_methods:
        get_fragments(topology,
                      method=method)
        print()

def interactive_fragment_picker_by_resSeq(resSeq_idxs, fragments, top,
                                          pick_first_fragment_by_default=False,
                                          additional_naming_dicts=None):

    # TODO command line tools residue_neighborhoods 255
    resSeq2residxs = {}
    resSeq2segidxs = {}
    last_answer = 0
    auto_last_answer_flag=False
    for key in resSeq_idxs:
        cands = [rr.index for rr in top.residues if rr.resSeq == key]
        # print(key)
        cand_fragments = _in_what_N_fragments(cands, fragments)
        if len(cands) == 0:
            print("No residue found with resSeq %s"%key)
        else:
            if len(cands) == 1:
                cands = cands[0]
                answer = cand_fragments
                # print(key,refgeom.top.residue(cands[0]), cand_fragments)
            elif len(cands) > 1:
                print("ambigous definition for resSeq %s" % key)
                #assert len(cand_fragments)==len(_np.unique(cand_fragments))
                for cc, ss, char in zip(cands, cand_fragments,"abcdefg"):
                    istr = '%s) %10s in fragment %2u with index %6u'%(char, top.residue(cc),ss, cc)
                    if additional_naming_dicts is not None:
                        extra=''
                        for key1, val1 in additional_naming_dicts.items():
                            if val1[cc] is not None:
                                extra +='%s: %s '%(key1,val1[cc])
                        if len(extra)>0:
                            istr = istr + ' (%s)'%extra.rstrip(" ")
                    print(istr)
                if not pick_first_fragment_by_default:
                    prompt =  "input one fragment idx (out of %s) and press enter.\n" \
                              "Leave empty and hit enter to repeat last option [%s]\n" \
                              "Use letters in case of repeated fragment index\n" % ([int(ii) for ii in cand_fragments], last_answer)

                    answer = input(prompt)

                    if len(answer) == 0:
                        answer = last_answer
                        cands = cands[_np.argwhere([answer == ii for ii in cand_fragments]).squeeze()]

                    elif answer.isdigit():
                        answer = int(answer)
                        cands = cands[_np.argwhere([answer == ii for ii in cand_fragments]).squeeze()]
                    elif answer.isalpha():
                        idx = "abcdefg".find(answer)
                        answer = cand_fragments[idx]
                        cands  = cands[idx]

                    #TODO implent k for keeping this answer from now on
                    if isinstance(answer,str) and answer=='k':
                        pass
                        print("Your answer has to be an integer in the of the fragment list %s" % cand_fragments)
                        raise Exception

                    assert answer in cand_fragments, (
                                "Your answer has to be an integer in the of the fragment list %s" % cand_fragments)
                    last_answer = answer
                else:
                    cands = cands[0]
                    answer = cand_fragments[0]
                    print("Automatically picked fragment %u"%answer)
                # print(refgeom.top.residue(cands))
                print()

            resSeq2residxs[key] = cands
            resSeq2segidxs[key] = answer

    return resSeq2residxs, resSeq2segidxs

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

    #TODO one usage in get_fragments 195
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


my_frag_colors=[
         'magenta',
         'yellow',
         'lime',
         'maroon',
         'navy',
         'olive',
         'orange',
         'purple',
         'teal',
]
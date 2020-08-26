##############################################################################
#    This file is part of mdciao.
#    
#    Copyright 2020 Charité Universitätsmedizin Berlin and the Authors
#
#    Authors: Guillermo Pérez-Hernandez
#    Contributors:
#
#    mdciao is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    mdciao is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with mdciao.  If not, see <https://www.gnu.org/licenses/>.
##############################################################################

import numpy as _np
from mdtraj.core.residue_names import _AMINO_ACID_CODES
import mdciao.utils as _mdcu
from  pandas import unique as _pandas_unique
from msmtools.estimation import connected_sets as _connected_sets

_allowed_fragment_methods = ['chains',
                             'resSeq',
                             'resSeq+',
                             'lig_resSeq+',
                             'bonds',
                             'resSeq_bonds',
                             "None",
                             ]


def print_fragments(fragments, top, **print_frag_kwargs):
    """Inform about fragments, very thinly wrapping around :obj:`print_frag`

    Parameters
    ----------
    fragments : dict or list
        Iterable with the sets of residue indexes
    top : :obj:`md.Topology`
    print_frag_kwargs : opt, keyword args for :obj:`print_frag`

    Returns
    -------

    """
    if isinstance(fragments,list):
        _fragments = {ii:val for ii, val in enumerate(fragments)}
    else:
        _fragments = {key:val for key, val in fragments.items()}

    for ii, iseg in _fragments.items():
        print_frag(ii, top, iseg, **print_frag_kwargs)

def print_frag(frag_idx, top, fragment, fragment_desc='fragment',
               idx2label=None,
               return_string=False,
               resSeq_jumps=True,
               **print_kwargs):
    """Pretty-printing of fragments of an :obj:`mtraj.topology`

    Parameters
    ----------
    frag_idx: int or str
        Index or name of the fragment to be printed
    top: :obj:`mdtraj.Topology`
        Topology in which the fragment appears
    fragment: iterable of indices
        The fragment in question, with zero-indexed residue indices
    fragment_desc: str, default is "fragment"
        Who to call the fragments, e.g. segment, block, monomer, chain
    idx2label : iterable or dictionary
        Pass along any consensus labels here
    resSeq_jumps : bool, default is True
        Inform whether the fragment contains jumps in the resSeq

    return_string: bool, default is False
        Instead of printing, return the string
    print_kwargs:
        Optional keyword arguments to pass to the print function, e.g. "end=","" and such

    Returns
    -------
    None or str, see return_string option

    """
    maplabel_first, maplabel_last = "", ""
    try:
        if idx2label is not None:
            maplabel_first = _mdcu.str_and_dict.choose_between_good_and_better_strings(None, idx2label[fragment[0]],
                                                                                       fmt="@%s")
            maplabel_last = _mdcu.str_and_dict.choose_between_good_and_better_strings(None, idx2label[fragment[-1]],
                                                                                      fmt="@%s")

        rfirst, rlast = [top.residue(ii) for ii in [fragment[0], fragment[-1]]]
        labfirst = "%8s%-10s" % (rfirst, maplabel_first)
        lablast = "%8s%-10s" % (rlast, maplabel_last)
        istr = "%s %6s with %4u AAs %8s%-10s (%4u) - %8s%-10s (%-4u) (%s) " % \
               (fragment_desc, str(frag_idx), len(fragment),
                #labfirst,
                rfirst, maplabel_first,
                rfirst.index,
                #lablast,
                rlast, maplabel_last,
                rlast.index,
                str(frag_idx))

        if rlast.resSeq - rfirst.resSeq != len(fragment) - 1:
            # print(ii, rj.resSeq-ri.resSeq, len(iseg)-1)
            istr += ' resSeq jumps'
    except:
        print(fragment)
        raise
    if return_string:
        return istr
    else:
        print(istr, **print_kwargs)

def get_fragments(top,
                  method='lig_resSeq+',
                  fragment_breaker_fullresname=None,
                  atoms=False,
                  verbose=True,
                  join_fragments=None,
                  **kwargs_residues_from_descriptors):
    """
    Given an :obj:`mdtraj.Topology` return its residues grouped into fragments using different methods.

    Parameters
    ----------
    top : :py:class:`mdtraj.Topology`
    method : str, default is 'lig_resSeq+'
        The method passed will be the basis for creating fragments. Check the following options
        with the example sequence

            "…-A27,Lig28,K29-…-W40,D45-…-W50,CYSP51,GDP52"

        - 'resSeq'
            breaks at jumps in resSeq entry:

            […A27,Lig28,K29,…,W40],[D45,…,W50,CYSP51,GDP52]

        - 'resSeq+'
            breaks only at negative jumps in resSeq:

            […A27,Lig28,K29,…,W40,D45,…,W50,CYSP51,GDP52]

        - 'bonds'
            breaks when residues are not connected by bonds, ignores resSeq:

            […A27][Lig28],[K29,…,W40],[D45,…,W50],[CYSP51],[GDP52]

            notice that because phosphorylated CYSP51 didn't get a
            bond in the topology, it's considered a ligand

        - 'resSeq_bonds'
            breaks at resSeq jumps and at missing bonds
        - 'lig_resSeq+'
            Like resSeq+ but put's any non-AA residue into it's own fragment.
            […A27][Lig28],[K29,…,W40],[D45,…,W50,CYSP51],[GDP52]
        - 'chains'
            breaks into chains of the PDB file/entry
        - None or 'None'
            all residues are in one fragment, fragment 0
    fragment_breaker_fullresname : list
        list of full residue names. Example [GLU30] will be used to break fragments,
        so that [R1, R2, ... GLU30,...R10, R11] will be broken into [R1, R2, ...], [GLU30,...,R10,R11]
    atoms : boolean, optional
        Instead of returning residue indices, return atom indices
    join_fragments : list of lists
        After getting the fragments with :obj:`method`,
        join these fragments again. The use case are hard
        cases where no method gets it right and some post-processing
        is needed.
        Duplicate entries in any inner list will be removed.
        One fragment idx cannot appear in more than one inner list,
        otherwise an exception is thrown
    verbose : boolean, optional
        Be verbose
    kwargs_residues_from_descriptors : optional
        additional arguments, see :obj:`residues_from_descriptors`

    Returns
    -------
    List of integer arrays
        Each array within the list has the residue indices of each fragment.
        These fragments do not have overlap. Their union contains all indices

    """

    _assert_method_allowed(method)

    # Auto detect fragments by resSeq
    fragments_resSeq = _get_fragments_by_jumps_in_sequence([rr.resSeq for rr in top.residues])[0]

    if method=="resSeq":
        fragments = fragments_resSeq
    elif method=='resSeq_bonds':
        residue_bond_matrix = _mdcu.bonds.top2residue_bond_matrix(top, verbose=False,
                                                      force_resSeq_breaks=True)
        fragments = _connected_sets(residue_bond_matrix)
    elif method=='bonds':
        residue_bond_matrix = _mdcu.bonds.top2residue_bond_matrix(top, verbose=False,
                                                      force_resSeq_breaks=False)
        fragments = _connected_sets(residue_bond_matrix)
        fragments = [fragments[ii] for ii in _np.argsort([fr[0] for fr in fragments])]
    elif method == "chains":
        fragments = [[rr.index for rr in ichain.residues] for ichain in top.chains]
    elif method == "resSeq+":
        fragments = _get_fragments_resSeq_plus(top, fragments_resSeq)
    elif method == "lig_resSeq+":
        fragments = _get_fragments_resSeq_plus(top, fragments_resSeq)
        for rr in top.residues:
            if rr.name[:3] not in _AMINO_ACID_CODES.keys():
                frag_idx = _mdcu.lists.in_what_fragment(rr.index,fragments)
                if len(fragments[frag_idx])>1:
                    list_for_removing=list(fragments[frag_idx])
                    list_for_removing.remove(rr.index)
                    fragments[frag_idx]=_np.array(list_for_removing)
                    fragments.append([rr.index])


    # TODO check why this is not equivalent to "bonds" in the test_file
    elif method == 'molecules':
        raise NotImplementedError("method 'molecules' is not fully implemented yet")
        #fragments = [_np.unique([aa.residue.index for aa in iset]) for iset in top.find_molecules()]
    elif method == 'molecules_resSeq+':
        raise NotImplementedError("method 'molecules_resSeq+' is not fully implemented yet")
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
    elif str(method).lower() == "none":
        fragments = [_np.arange(top.n_residues)]

    fragments = [fragments[ii] for ii in _np.argsort([ifrag[0] for ifrag in fragments])]
    # Inform of the first result
    if verbose:
        print("Auto-detected fragments with method '%s'"%str(method))
        print_fragments(fragments,top)
    # Join if necessary
    if join_fragments is not None:
        fragments = _mdcu.lists.join_lists(fragments, join_fragments)
        print("Joined Fragments")
        for ii, iseg in enumerate(fragments):
            print_frag(ii, top, iseg)

    if fragment_breaker_fullresname is not None:
        if isinstance(fragment_breaker_fullresname,str):
            fragment_breaker_fullresname=[fragment_breaker_fullresname]
        for breaker in fragment_breaker_fullresname:
            residxs, fragidx = _mdcu.residue_and_atom.residues_from_descriptors(breaker, fragments, top,
                                                           **kwargs_residues_from_descriptors)
            idx = residxs[0]
            ifrag = fragidx[0]
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
                        print_frag(ii, top, ifrag)
                else:
                    print("Not using it since it already was a fragment breaker in frag %s"%ifrag)
            print()
            # raise ValueError(idx)

    if not atoms:
        return fragments
    else:
        return [_np.hstack([[aa.index for aa in top.residue(ii).atoms] for ii in frag]) for frag in fragments]

def _get_fragments_by_jumps_in_sequence(sequence,jump=1):
    r"""
    Return the array in :obj:`sequence` chopped into sub-arrays (=fragments=
    where there's a jump in the sequence

    Parameters
    ----------
    sequence : iterable of ints
    jump : int, default is 1
        What is considered a jump in the sequence

    Returns
    -------
    idxs : list
        The idxs of :obj:`sequence` in each fragment
    fragments : list
        :obj:`sequence` chopped into the fragments
    """
    old = sequence[0]
    frag_idxs = [[]]
    frag_elements = [[]]
    for ii, rr in enumerate(sequence):
        delta = _np.abs(rr - old)
        # print(delta, ii, rr, end=" ")
        if delta <= jump:
            # print(delta, ii, rr, "appending")
            frag_idxs[-1].append(ii)
            frag_elements[-1].append(rr)
        else:
            # print(delta, ii, rr, "new")
            frag_idxs.append([ii])
            frag_elements.append([rr])
        # print(fragments[-1])
        old = rr
    return frag_idxs, frag_elements

def _get_fragments_resSeq_plus(top, fragments_resSeq):
    r"""
    Get fragments using the 'resSeq+' method
    Parameters
    ----------
    top
    fragments_resSeq

    Returns
    -------

    """
    to_join = [[0]]
    for ii, ifrag in enumerate(fragments_resSeq[:-1]):
        r1 = top.residue(ifrag[-1])
        r2 = top.residue(fragments_resSeq[ii + 1][0])
        if r1.resSeq < r2.resSeq:
            to_join[-1].append(ii + 1)
        else:
            to_join.append([ii + 1])

    return _mdcu.lists.join_lists(fragments_resSeq, [tj for tj in to_join if len(tj) > 1])

def overview(topology,
             methods=['all'],
             AAs=None,
             ):

    """
    Prints the fragments obtained by :obj:`get_fragments` for the available methods.

    Optionally, you can pass along a list of residue
    descriptors to be printed after the fragments have
    been shown.

    Parameters
    ----------
    topology :  :obj:`mdtraj.Topology`
    methods : str or list of strings
        method(s) to be used for obtaining fragments
    AAs : list, default is None
        Anything that :obj:`find_AA` can understand

    Returns
    -------
    fragments_out : dict
        The result of the fragmentation schemes keyed
        by their method name

    """

    if isinstance(methods,str):
        methods = [methods]

    if methods[0].lower() == 'all':
        try_methods = _allowed_fragment_methods
    else:
        for method in methods:
            _assert_method_allowed(method)
        try_methods = methods

    fragments_out = {}
    for method in try_methods:
        try:
            fragments_out[method] = get_fragments(topology,
                                                  method=method)
        except Exception as e:
            print("The method %s did not work:"%method)
            print(e)
            print()
        print()

    _mdcu.residue_and_atom.parse_and_list_AAs_input(AAs, topology)

    return fragments_out

def _assert_method_allowed(method):
    assert str(method) in _allowed_fragment_methods, ('input method %s is not known. ' \
                                                 'Know methods are %s ' %
                                                 (method, "\n".join(_allowed_fragment_methods)))

def check_if_subfragment(sub_frag, fragname, fragments, top,
                         map_conlab=None,
                         keep_all=False):
    r"""
    Input an iterable of integers representing a fragment and check if
    it clashes with other fragment definitions.

    Prompt for a choice in case it is necessary

    Example
    -------
    Let's assume the BW-nomenclature tells us that TM6 is [0,1,2,3]
    and we have already divided the topology into fragments
    using :obj:`get_fragments`, with method "resSeq+", meaning
    we have fragments for the receptor, Ga,Gb,Gg

    The purpusose is to check whether the BW-fragmentation is
    contained in the previous fragmentation:
    * [0,1,2,3] and :obj:`fragments`=[[0,1,2,3,4,6], [7,8,9]]
    is not a clash, bc TM6 is contained in fragments[0]
    * [0,1,2,3] and :obj:`fragments`=[[0,1],[2,3],[4,5,6,7,8]]
    is a clash. In this case the user will be prompted to choose
    which subset of "TM6" to keep:
     * "0": [0,1]
     * "1": [2,3]
     * "0-1" [0,1,2,3]


    Parameters
    ----------
    sub_frag : iterable of integers
    fragname : str
    fragments : iterable of iterables of integers
    top : :obj:`mdtraj.Trajectory`object
    map_conlab : list or dict, default is None
        maps residue idxs to consensus labels

    Returns
    -------
    tokeep = 1D numpy array
        If no clashes were found, this will be contain the same residues as
        :obj:`sub_frag` without prompting the user.
        Otherwise, the user has to input whether to leave the definition intact
        or pick a sub-set
    """
    # Get the fragment idxs of all residues in this fragment
    ifrags = [_mdcu.lists.in_what_fragment(idx, fragments) for idx in sub_frag]

    frag_cands = [ifrag for ifrag in _pandas_unique(ifrags) if ifrag is not None]
    if len(frag_cands) > 1 and not keep_all:
        # This only happens if more than one fragment is present
        print_frag(fragname, top, sub_frag, fragment_desc='',
                   idx2label=map_conlab)
        print("  %s clashes with other fragment definitions"%fragname)
        for jj in frag_cands:
            istr = print_frag(jj, top, fragments[jj],
                              fragment_desc="   input fragment",
                              return_string=True)
            n_in_fragment = len(_np.intersect1d(sub_frag, fragments[jj]))
            if n_in_fragment < len(fragments[jj]):
                istr += "%u residues outside %s" % (len(fragments[jj]) - n_in_fragment, fragname)
            print(istr)
        answr = input("Input what fragment idxs to include into %s  (fmt = 1 or 1-4, or 1,3):" % fragname)
        answr = _mdcu.lists.rangeexpand(answr)
        assert all([idx in ifrags for idx in answr])
        tokeep = _np.hstack([idx for ii, idx in enumerate(sub_frag) if ifrags[ii] in answr]).tolist()
        if len(tokeep) >= len(ifrags):
            raise ValueError("Cannot keep these fragments %s!" % (str(answr)))
        return tokeep
    else:
        return sub_frag

def _fragments_strings_to_fragments(fragment_input, top, verbose=False):
    r"""
    Method to help implement the input options wrt
    to fragments of :obj:`parsers.parser_for_interface`

    Check the documentation of interface (-h) for more details

    Check also :obj:`rangeexpand` to understand the expressions


    Parameters
    ----------
    fragment_input : list of strings
        Many cases are possible
        * ["consensus"] : fragment using "resSeq+"
        and return user_wants_consensus as True
        * [method] : fragment using "method"
        (see :obj:`get_fragments`) and return
        user_wants_consensus as False
        * [['exp1']] : this str represents the
        residues in one fragment (eg. "0-3,5" : 0,1,2,3,5).
        Assume that the missing residues the other fragment.
        Return the two fragments and user_wants_consensus as False
        * [["exp1"],
           ["exp2"],
           [...]]
        These strs are the fragments expressed as residue
        indices. Evaluate them and return them. Return
        user_wants_consensus as False
        * None or "None"

    top : :obj:`mdtraj.Topology`

    Returns
    -------
    fragments_as_residue_idxs, user_wants_consensus

    """
    user_wants_consensus = False
    assert isinstance(fragment_input,list)
    if ("".join(fragment_input).replace("-","").replace(",","")).isnumeric():
        method = "user input by residue index"
        # What we have is list residue idxs as strings like 0-100, 101-200, 201-300
        fragments_as_residue_idxs =[_mdcu.lists.rangeexpand(ifrag.strip(",")) for ifrag in fragment_input]
        for ii, ifrag in enumerate(fragments_as_residue_idxs):
            _mdcu.lists.assert_no_intersection(fragments_as_residue_idxs, word="fragments")
            if not all([aa in range(top.n_residues) for aa in ifrag]):
                print("Fragment %u has idxs outside of the geometry (total n_residues %u): %s"%(ii, top.n_residues,
                                                                         set(ifrag).difference(range(top.n_residues))))

        if len(fragment_input)==1:
            assert isinstance(fragment_input[0],str)
            method += "(only one fragment provided, assuming the rest of residues are fragment 2)"
            fragments_as_residue_idxs.append(_np.delete(_np.arange(top.n_residues), fragments_as_residue_idxs[0]))
    elif len(fragment_input)==1:
        if fragment_input[0].lower()=="consensus":
            user_wants_consensus = True
            method = 'resSeq+ (for later user_wants_consensus labelling)'
            fragments_as_residue_idxs = get_fragments(top, method='resSeq+',
                                                      verbose=False)
        else:
            method = fragment_input[0]
            fragments_as_residue_idxs = get_fragments(top, method=method,
                                                      verbose=False)
    if verbose:
        print("Using method '%s' these fragments were found"%method)
        for ii, ifrag in enumerate(fragments_as_residue_idxs):
            print_frag(ii, top, ifrag)

    return fragments_as_residue_idxs, user_wants_consensus

def frag_list_2_frag_groups(frag_list,
                            frag_idxs_group_1=None,
                            frag_idxs_group_2=None,
                            verbose=False):
    r"""
    Automagically find out the user wants to define
    two fragments out of list of fragments. This is used
    by CLTs interface

    Promt the user when disambguation is needed.

    Parameters
    ----------
    frag_list : list
        list of fragments, defined as residue indices
    frag_idxs_group_1 : iterable of ints, or str default is None
        When str, it has to be a rangeexpand exprs, e.g. 2-5,10
    frag_idxs_group_2 : iterable of ints, or str default is None


    Returns
    -------
    groups_as_residxs, groups_as_fragidxs

    """

    if verbose:
        for ii, ifrag in enumerate(frag_list):
            print("frag %u: %u-%u"%(ii, ifrag[0],ifrag[-1]))

    if len(frag_list) == 2 and \
            frag_idxs_group_1 is None and \
            frag_idxs_group_2 is None:
        print("Only two fragments detected with no values for frag_idxs_group_1 and frag_idxs_group_2.\n"
              "Setting frag_idxs_group_1=0 and frag_idxs_group_2=1")
        groups_as_fragidxs = [[0], [1]]
        #TODO I don't think i need the check for None in the frag_idxs_groups, right?
    else:
        groups_as_fragidxs = [frag_idxs_group_1, frag_idxs_group_2]
        for ii, ifrag_idxs in enumerate(groups_as_fragidxs):
            if ifrag_idxs is None:
                groups_as_fragidxs[ii] = _mdcu.lists.rangeexpand(input('Input group of fragments '
                                                                     '(e.g. 0,3 or 2-4,6) for group %u: ' % (ii + 1)))
            elif isinstance(ifrag_idxs, str):
                groups_as_fragidxs[ii] = _mdcu.lists.rangeexpand(ifrag_idxs)
    groups_as_residxs = [sorted(_np.hstack([frag_list[ii] for ii in iint])) for iint in
                            groups_as_fragidxs]

    return groups_as_residxs, groups_as_fragidxs

def frag_dict_2_frag_groups(frag_defs_dict, ng=2,
                            verbose=False):
    r"""
    Input a dictionary of fragment definitions, keyed by
    whatever and valued with residue idxs and prompt
    the user how to re-group them

    It wraps around :obj:`_match_dict_by_patterns` to
    under the hood

    TODO: refactor into str_and_dict_utils
    TODO: It will be mostly used with fragments so it's better here for the API? IDK

    Parameters
    ----------
    frag_defs_dict : dict
        Fragment definitions in residue idxs
    ng : int, default is 2
        wanted number of groups

    Returns
    -------
    groups_as_residue_idxs, groups_as_keys

    groups_as_residue_idxs : list of len ng
        Contains ng arrays with the concatenated
        and sorted residues in each group
    groups_as_keys : list of len ng
        Contains ng lists with the keys
        of :obj:`frag_defs_dict` in each of groups
    """

    groups_as_keys = []
    groups_as_residue_idxs = []
    if verbose:
        for key, val in frag_defs_dict.items():
            print("%s: %u-%u"%(key, val[0],val[-1]))
    for ii in range(1, ng + 1):
        print("group %u: " % ii, end='')
        answer = input(
            "Input a list of comma-separated posix-expressions.\n"
            "Prepend with '-' to exclude, e.g. 'TM*,-TM2,H8' to grab all TMs and H8, but exclude TM2)\n").replace(
            " ", "").strip("'").strip('"')
        igroup, res_idxs_in_group = _mdcu.str_and_dict.match_dict_by_patterns(answer, frag_defs_dict)
        groups_as_keys.append([ilab for ilab in frag_defs_dict.keys() if ilab in igroup])
        groups_as_residue_idxs.append(sorted(res_idxs_in_group))
        print(', '.join(groups_as_keys[-1]))

    return groups_as_residue_idxs, groups_as_keys

def splice_orphan_fragments(fragments, fragnames, highest_res_idx=None,
                            orphan_name="?"):
    r"""
    Return a fragment list where residues not present in :obj:`fragments` are
    now new interstitial fragments.

    "not-present" means outside of the ranges of each fragment, s.t.
    an existing fragment like [0,1,5,6] is actually considered [0,1,2,3,4,5,6]

    Parameters
    ----------
    fragments : list
        The initial fragments potentially missing some residues
    fragnames : list
        The names of :obj:`fragments`
    highest_res_idx : int, default is None
        The highest possible res_idx. The returned fragments
        will have this idx as last idx of the last fragment (input or orphan)
        If None, the highest value (max) of fragments is
        used
    orphan_name : str, default is ?


    Returns
    -------
    new_fragments : list
        The new fragments including those missing in the input
    new_names : list
        The new names, where the "orphans" have gotten the name orphan_char

    """
    if highest_res_idx is None:
        highest_res_idx = _np.max(_np.hstack(fragments))

    full_frags = []
    for ifrag, iname in zip(fragments,fragnames):
        if len(ifrag)==ifrag[-1]-ifrag[0]:
            print("Fragment %s has holes in it but will be considered from %u to %u regardless"%(iname,ifrag[0],ifrag[-1]))
        full_frags.append(_np.arange(ifrag[0],ifrag[-1]+1))
    orphans = _np.delete(_np.arange(highest_res_idx + 1), _np.hstack(full_frags))
    orphans = _get_fragments_by_jumps_in_sequence(orphans)[1]
    orphans_labels = [orphan_name for __ in orphans]
    new_frags = orphans + full_frags
    new_labels = orphans_labels + fragnames
    idxs = _np.argsort([ifrag[0] for ifrag in new_frags])
    return [list(new_frags[ii]) for ii in idxs], [new_labels[ii] for ii in idxs]

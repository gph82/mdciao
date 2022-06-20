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
import mdtraj as _md
from mdtraj.core.residue_names import _PROTEIN_RESIDUES
import mdciao.utils as _mdcu
from pandas import unique as _pandas_unique

_allowed_fragment_methods = ['chains',
                             'resSeq',
                             'resSeq+',
                             'lig_resSeq+',
                             'bonds',
                             'resSeq_bonds',
                             "None",
                             ]


def print_fragments(fragments, top, max_lines=40, **print_frag_kwargs):
    """Inform about fragments, very thinly wrapping around :obj:`print_frag`

    Parameters
    ----------
    fragments : dict or list
        Iterable with the sets of residue indexes
    top : :obj:`~mdtraj.Topology`
    max_lines : int, default is 40
        Maximum number of lines to print. E.g. if 40,
        first 20 and last 20 will be printed
    print_frag_kwargs : opt, keyword args for :obj:`print_frag`

    Returns
    -------
    frag_list : list
        List of the strings that
        are being printed, for further use

    """
    if isinstance(fragments,list):
        _fragments = {ii:val for ii, val in enumerate(fragments)}
    else:
        _fragments = {key:val for key, val in fragments.items()}
    frag_list = []
    for ii, iseg in _fragments.items():
        frag_list.append(print_frag(ii, top, iseg, return_string=True, **print_frag_kwargs))
    n = _np.round(max_lines/2).astype(int)
    if n * 2 < len(frag_list):
        frag_list = frag_list[:n] + ["...[long list: omitted %u items]..." % (len(frag_list) - 2 * n)] + frag_list[-n:]
    print("\n".join(frag_list))
    return frag_list

def print_frag(frag_idx, top, fragment, fragment_desc='fragment',
               idx2label=None,
               return_string=False,
               resSeq_jumps=True,
               label_width=10,
               **print_kwargs):
    """Pretty-printing of fragments of an :obj:`mtraj.topology`

    Parameters
    ----------
    frag_idx: int or str
        Index or name of the fragment to be printed
    top: :obj:`~mdtraj.Topology` or string
        Topology or string (=AA sequence) to "grab"
        the residue-names from when informing
    fragment: iterable of indices
        The fragment in question, with zero-indexed residue indices
    fragment_desc: str, default is "fragment"
        How to call the fragments, e.g. segment, block, monomer, chain
    idx2label : iterable or dictionary
        Pass along any consensus labels here
    resSeq_jumps : bool, default is True
        Inform whether the fragment contains jumps in the resSeq
    label_width : int, default is 10
        The width in characters given
        to the label descriptor. You can set this to zero
        if you know :obj:`idx2label` is None for all
        printed lines
    return_string: bool, default is False
        Instead of printing, return the string
    print_kwargs:
        Optional keyword arguments to pass to the print function, e.g. "end=","" and such

    Returns
    -------
    None or str, see return_string option

    """
    maplabel_first, maplabel_last = "", ""
    labfmt = "%-"+"%us"%label_width
    try:
        if idx2label is not None:
            maplabel_first = _mdcu.str_and_dict.choose_options_descencing([idx2label[fragment[0]]],
                                                                                       fmt="@%s")
            maplabel_last = _mdcu.str_and_dict.choose_options_descencing([idx2label[fragment[-1]]],
                                                                                      fmt="@%s")
        if isinstance(top,_md.Topology):
            rfirst, rlast = [top.residue(ii) for ii in [fragment[0], fragment[-1]]]

        elif isinstance(top,str):
            rfirst, rlast = [top[ii] for ii in [fragment[0], fragment[-1]]]
        rfirst_index, rlast_index = [fragment[0],fragment[-1]]

        labfirst = "%8s%-10s" % (rfirst, maplabel_first)
        lablast = "%8s%-10s" % (rlast, maplabel_last)
        istr = "%s %6s with %6u AAs %8s%s (%6u) - %8s%s (%-6u) (%s) " % \
               (fragment_desc, str(frag_idx), len(fragment),
                #labfirst,
                rfirst, labfmt%maplabel_first,
                rfirst_index,
                #lablast,
                rlast, labfmt%maplabel_last,
                rlast_index,
                str(frag_idx))

        if isinstance(top,_md.Topology) and  rlast.resSeq - rfirst.resSeq != len(fragment) - 1:
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
                  maxjump = 500,
                  salt=["Na+","Cl-","Na","Cl"],
                  water=True,
                  **kwargs_residues_from_descriptors):
    """
    Group residues of a molecular topology into fragments using different methods.

    Water and ions get their own fragment by default except for the methods
    None, chains, and any method involving bonds

    Parameters
    ----------
    top : :obj:`~mdtraj.Topology` or str
        When str, path to filename

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
            Also check :obj:`maxjump`
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
    salt : list, default is ["Na+","Cl+", "NA","CL"]
        Residues that match these residue names and
        have only one atom will be put together
        in the last fragment. Use salt = []
        to deactivate. Doesn't apply for methods
        involving bonds or None and chains
    water : bool, default is True
        Put water on its own fragment.
        Doesn't apply for methods
        involving bonds or None and chains
    maxjump : int or None, default is 500
        The maximum allowed positive sequence-jump
        in the 'resSeq+' methods, i.e. don't
        join ALA500 with GLU551 even though
        the jump in sequence is positive
        None means no limit for positive jumps
    kwargs_residues_from_descriptors : optional
        additional arguments, see :obj:`~mdciao.residue_and_atom.residues_from_descriptors`

    Returns
    -------
    List of integer arrays
        Each array within the list has the residue indices of each fragment.
        These fragments do not have overlap. Their union contains all indices

    """

    _assert_method_allowed(method)
    salt = [ss.lower() for ss in salt]
    if isinstance(top, str):
        top = _md.load(top).top

    # Auto detect fragments by resSeq
    fragments_resSeq = _get_fragments_by_jumps_in_sequence([rr.resSeq for rr in top.residues])[0]

    if method=="resSeq":
        fragments = fragments_resSeq
    elif method=='resSeq_bonds':
        residue_bond_matrix = _mdcu.bonds.top2residue_bond_matrix(top, verbose=False,
                                                      force_resSeq_breaks=True)
        fragments = _mdcu.bonds.connected_sets(residue_bond_matrix)
    elif method=='bonds':
        residue_bond_matrix = _mdcu.bonds.top2residue_bond_matrix(top, verbose=False,
                                                      force_resSeq_breaks=False)
        fragments = _mdcu.bonds.connected_sets(residue_bond_matrix)
        fragments = [fragments[ii] for ii in _np.argsort([fr[0] for fr in fragments])]
    elif method == "chains":
        fragments = [[rr.index for rr in ichain.residues] for ichain in top.chains]
    elif method == "resSeq+":
        fragments = _get_fragments_resSeq_plus(top, fragments_resSeq,maxjump=maxjump)
    elif method == "lig_resSeq+":
        fragments = _get_fragments_resSeq_plus(top, fragments_resSeq,maxjump=maxjump)
        lig_cands = _np.unique([top.atom(aa).residue.index for aa in top.select("not protein and not water")])
        for ii in lig_cands:
            rr = top.residue(ii)
            if rr.name[:3] in _PROTEIN_RESIDUES or rr.name.lower() in salt:
                continue
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

    if "resSeq" in str(method) and "bond" not in str(method):
        if water:
            fragments = _dry_fragments(fragments, top)
        fragments = _bland_fragments(fragments, top, salt)

    fragments = [fragments[ii] for ii in _np.argsort([ifrag[0] for ifrag in fragments])]

    # Inform of the first result
    if verbose:
        print("Auto-detected fragments with method '%s'"%str(method))
        print_fragments(fragments,top,label_width=0)
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
            #todo use _break_fragments
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

def _break_fragments(breakers, fragments):
    r"""
    Given a list of fragment breakers, break existing fragments further into sub-fragments

    The break is so that j appearing in fragment [a, b, ...,i,j,...z] will
    generate [a, b, ...,i],[j,...z]

    Parameters
    ----------
    breakers : iterable of idxs
        These indices will force
        a break in the :obj:`fragments`,
        generating sub-fragments.
    fragments : iterable of iterables
        The fragment definitions

    Returns
    -------
    fragments : list
    """
    _mdcu.lists.assert_no_intersection(fragments)
    for idx in _np.unique(breakers):
        ifrag = _mdcu.lists.in_what_fragment(idx, fragments)
        if ifrag is not None:
            idx_split = _np.flatnonzero(idx == _np.array(fragments[ifrag]))[0]
            subfrags = [fragments[ifrag][:idx_split], fragments[ifrag][idx_split:]]
            if idx_split > 0:
                # print("now breaking up into", subfrags)
                fragments = fragments[:ifrag] + subfrags + fragments[ifrag + 1:]
    return fragments

#TODO use this throught the code instead of the python explicit way
def fragment_slice(traj : _md.Trajectory, fragments, keys_or_idxs=None):
    r"""

    Slice a geometry using arbitrary fragment definitions, a la :obj:`mdtraj.Trajectory.atom_slice`

    Note
    ----
    Regardless of the order in which the selection
    is done (e.g. keys_or_idxs=[1,0]) the returned
    :obj:`sliced_traj` will always be in ascending
    order atoms as they appear in :obj:`traj`
    Parameters
    ----------
    traj : :obj:`mdtraj.Trajectory`
        The trajectory to slice
    fragments : list or dict
        The fragment definitions as residue indices.
        Can be as a list or a as a dict, e.g. the output of
        :obj:`mdciao.fragments.get_fragments` (list) or
        :obj:`mdciao.nomenclature.LabelerGPCR.top2frags` (dict)
    keys_or_idxs : iterable or None
        The keys or indices of the
        fragments to slice to, i.e.
        to keep. If None, all
        all fragments are used
        as a selection

    Returns
    -------
    sliced_traj : :obj:`mdtraj.Trajectory`
        A copy of :obj:`traj` only with the
        atoms present in the selected fragments

    """

    if keys_or_idxs is None:
        if isinstance(fragments,dict):
            keys_or_idxs = list(fragments.keys())
        else:
            keys_or_idxs = _np.arange(len(fragments))

    _fragments = _np.hstack([fragments[idx] for idx in keys_or_idxs])

    return traj.atom_slice([aa.index for aa in traj.top.atoms if aa.residue.index in _fragments])

def _dry_fragments(fragments, top):
    r"""
    Remove water molecules from :obj:`fragments` and append them at the end as their own fragment(s)

    Water is selected with top.select("water") (check https://www.mdtraj.org/1.9.5/atom_selection.html)

    Parameters
    ----------
    fragments : list of ints
    top : :obj:`~mdtraj.Topology`

    Returns
    -------
    dry_fragments : list
        The original :obj:`fragments` except
        the water molecules have been put into
        their own fragment(s) at the end of the list.
        If the water molecules are not contiguous in
        their serial residue-indices (not resSeq),
        they get split into contiguous fragments
    """
    waters = _np.unique([top.atom(aa).residue.index for aa in top.select("water")])
    if len(waters)>0:
        return _mdcu.lists.remove_from_lists(fragments, waters)+_get_fragments_by_jumps_in_sequence(waters)[1]
    else:
        return fragments

def _bland_fragments(fragments, top, salt):
    r"""
    Remove salt from :obj:`fragments` and append them at the end as their own fragment(s)

    Parameters
    ----------
    fragments : list of ints
    top : :obj:`~mdtraj.Topology`
    salt : list
        Ions will be matched using
        residue.name in :obj:`salt`
        and ensuring residue.n_atoms == 1

    Returns
    -------
    bland_fragments : list
        The original :obj:`fragments` except
        the ions have been put into
        their own fragment(s) at the end of the list.
        If the ions are not contiguous in
        their serial residue-indices (not resSeq),
        they get split into contiguous fragments
    """
    salt = [ss.lower() for ss in salt]
    ion_cands = _np.unique([top.atom(aa).residue.index for aa in top.select("not protein and not water")])
    ion_cands = [ii for ii in ion_cands if top.residue(ii).name.lower() in salt and top.residue(ii).n_atoms==1]
    if len(ion_cands)>0:
        return _mdcu.lists.remove_from_lists(fragments, ion_cands) + _get_fragments_by_jumps_in_sequence(ion_cands)[1]
    else:
        return fragments


def match_fragments(seq0, seq1,
                    frags0=None,
                    frags1=None,
                    probe=None,
                    verbose=False,
                    shortest=3):
    r"""
    Align fragments of seq0 and seq1 pairwise and return a matrix of scores.

    The score is the absolute number matches between
    two fragments. Depending on how informed
    the user is about the topologies, their fragments,
    their similarities, and what the user is trying
    to do, this absolute measure can be just right or
    be highly misleading, e.g:
     * two fragments of ~500 AAs each can score
       20 matches "easily", without this being meaningful
     * two fragments of 11 AAs each having 10 matches
       between them are almost identical
     * however, in absolute terms, the first case has a
       higher score

    If you know what you're doing, you can specify
    which one of the sequences is the :obj:`probe`, s.t.
    the score is divided by the length the fragment of
    the probe. E.g., if :obj:`probe` =1, it means
    that you are interested in finding out if
    fragments of :obj:`seq1` appear in fragments of :obj:`seq0`,
    (the 'target' sequence), regardless of how long
    the target fragments are. The score is
    then normalized to 1, where 1 means you
    found the entire probe fragment in the target fragment,
    no matter how long the probe or the target were.


    Parameters
    ----------
    seq0 : str or :obj:`~mdtraj.Topology`
    seq1 : str or :obj:`~mdtraj.Topology`
    frags0 : list or None, default is None
        If None, :obj:`get_fragments` will
        be called with the default options
        to generate a fragment list.
    frags1 : list or None, default is None
        If None, :obj:`get_fragments` will
        be called with the default options
        to generate a fragment list.
    probe : int, default is None
        If None, scores are absolute numbers.
        If 0, the scores are divided by
        the seq0's fragment length. If 1,
        by seq1's fragment length. In
        these cases, the score is
        always between 0 and 1,
        regardless how long the probe
        and the target fragments are.
    shortest : int, default is 3
        Fragments of len < :obj:`shortest`
        won't produce a score but a np.NaN,
        s.t. the :obj:`score` doesn't get
        highjacked by very small :obj:`probe`
        fragments, which will always yield
        relative good scores.
        Absolute scores (:obj:`probe` = None)
        are not affected by this.
    verbose : bool, default is False
        Be verbose, affects all methods
        called by the this method as well.

    Returns
    -------
    score : 2D np.ndarray of shape(len(frags0),len(frags1))
        Will be between 0 and 1 if a :obj:`probe`
        is specified
    frags0 : list
        The fragments that were either provided
        or generated on the fly. Their indices
        are the row-indices of :obj:`score`
    frags1 : list
        The fragments that were either provided
        or generated on the fly. Their indices
        are the row-indices of :obj:`score`
    """

    frags = []
    for iseq,ifrg in zip([seq0, seq1],
                         [frags0, frags1]):
        if ifrg is None:
            if isinstance(iseq,_md.Topology):
                ifrg = get_fragments(iseq,verbose=verbose)
            else:
                assert isinstance(iseq,str)
                ifrg = [_np.arange(len(iseq))]
        frags.append(ifrg)

    score = _np.zeros((len(frags[0]),len(frags[1])),dtype=float)
    for ii, ifrag in enumerate(frags[0]):
        for jj,jfrag in enumerate(frags[1]):
            df = _mdcu.sequence.align_tops_or_seqs(seq0, seq1,
                                                   seq_0_res_idxs=frags[0][ii],
                                                   seq_1_res_idxs=frags[1][jj],verbose=verbose)[0]
            score[ii,jj] = df["match"].sum()
            if probe is not None:
                if len([ifrag,jfrag][probe])<=shortest:
                    score[ii,jj] = _np.nan
                else:
                    score[ii,jj]/=len([ifrag,jfrag][probe])

            if verbose:
                for tt, itop in enumerate([seq0,seq1]):
                    print_frag(ii, seq0, ifrag, "seq%u fragment"%tt)
                print(score[ii,jj].round(2))
    return score, frags[0], frags[1]

def _get_fragments_by_jumps_in_sequence(sequence,jump=1):
    r"""
    Return the array in :obj:`sequence` chopped into sub-arrays (=fragments)
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

#TODO combine with the above method, they are pretty redundant
def _get_fragments_resSeq_plus(top, fragments_resSeq,maxjump=None):
    r"""
    Get fragments using the 'resSeq+' method
    Parameters
    ----------
    top
    fragments_resSeq
    maxjump

    Returns
    -------

    """
    to_join = [[0]]
    jump_is_short    = lambda idx1,idx2 :  [True if maxjump is None else idx2-idx1 <= maxjump][0]
    for ii, ifrag in enumerate(fragments_resSeq[:-1]):
        r1 = top.residue(ifrag[-1])
        r2 = top.residue(fragments_resSeq[ii + 1][0])
        if r1.resSeq < r2.resSeq and jump_is_short(r1.resSeq, r2.resSeq):
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
                                                      'Know methods are\n%s ' %
                                                      (method, "\n".join(_allowed_fragment_methods)))

def check_if_subfragment(sub_frag, fragname, fragments, top,
                         map_conlab=None,
                         keep_all=False,
                         prompt=True):
    r"""
    Input an iterable of integers representing a fragment and check if
    it clashes with other fragment definitions.

    Return False/True or prompt for a choice in case it is necessary

    Example
    -------
    Let's assume the GPCR-nomenclature tells us that TM6 is [0,1,2,3]
    and we have already divided the topology into fragments
    using :obj:`get_fragments`, with method "resSeq+", meaning
    we have fragments for the receptor,Ga,Gb,Gg

    The purpose is to check whether the GPCR-fragmentation is
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
        The fragment to be checked if
        subfragment of `fragments`
    fragname : str
    fragments : iterable of iterables of integers
    top : :obj:`~mdtraj.Trajectory` object
    map_conlab : list or dict, default is None
        maps residue idxs to consensus labels
    prompt : bool, default is True
        When False, no prompt is issued,
        and the returned value is a boolean
        whether or not `sub_frag` is actually a sub-fragment (=subset)
        of any one fragment of `fragments`
    Returns
    -------
    answer : 1D numpy array or boolean
        If `prompt` is True and no clashes were found,
        this will contain the same residues as
        :obj:`sub_frag` without prompting the user.
        Otherwise, the user has to input whether
        to leave the definition intact or pick a sub-set
        If `prompt` is False, this is a boolean
        saying whether `sub_frag` is a sub-fragment (subset)
        of any one fragment of `fragments`
    """
    # Get the fragment idxs of all residues in this fragment
    ifrags = [_mdcu.lists.in_what_fragment(idx, fragments) for idx in sub_frag]

    frag_cands = [ifrag for ifrag in _pandas_unique(ifrags) if ifrag is not None]
    if not prompt:
        return len(frag_cands) <= 1

    if len(frag_cands) > 1 and not keep_all:
        # This only happens if more than one fragment is present
        print_frag(fragname, top, sub_frag, fragment_desc='',
                   idx2label=map_conlab)
        print(" Subfragment %s clashes with other fragment definitions,\n"
              " because the residues of %s span over more than 1 fragment:"%(fragname, fragname))
        for jj in frag_cands:
            istr = print_frag(jj, top, fragments[jj],
                              fragment_desc="   input fragment",
                              return_string=True,
                              label_width=0)
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

    #TODO rename the strings to something else
    Try to understand how the user wants to fragment the topology
    Pretty flexible

    Check also :obj:`rangeexpand` to understand the expressions


    Parameters
    ----------
    fragment_input : list of strings
        Many cases are possible
        * ["consensus"] : user wants to use
        consensus labels such as "TM6" etc.
        Using "resSeq+" heuristic to fragment, and
        returns :obj:`user_wants_consensus` as True,
        this will trigger further prompts
        * [method] : fragment using "method"
        (see :obj:`get_fragments`) and return
        user_wants_consensus as False
        * [['exp1']] : this str represents the
        residues in one fragment (eg. "0-3,5" : 0,1,2,3,5).
        Assume that any missing residues belong the other fragment.
        Return the two fragments and user_wants_consensus as False
        * [["exp1"],
           ["exp2"],
           [...]]
        These strs are the fragments expressed as residue
        indices. Evaluate them and return them. Return
        user_wants_consensus as False
        * exp can also b ["LEU30-GLU40"]
        * None or "None"

    top : :obj:`~mdtraj.Topology`

    Returns
    -------
    fragments_as_residue_idxs, user_wants_consensus

    """
    user_wants_consensus = False
    assert isinstance(fragment_input,list)
    #TODO the following line is untested, the usecase not mentioned in the docs...?
    if len(fragment_input)==1 and isinstance(fragment_input[0],str) and " " in fragment_input[0]:
        fragment_input = fragment_input[0].split(" ")
    #if len(fragment_input)==1 and isinstance(fragment_input[0],str) and "," in fragment_input[0]:
    #    fragment_input=fragment_input[0].split(",")
    if str(fragment_input[0]).lower()=="consensus":
        user_wants_consensus = True
        method = 'resSeq+'
        fragments_as_residue_idxs = get_fragments(top, method='resSeq+',
                                                  verbose=False)
    elif str(fragment_input[0]) in _allowed_fragment_methods:
        method = fragment_input[0]
        fragments_as_residue_idxs = get_fragments(top, method=method,
                                                  verbose=False)
    else:
        method = "user input by residue array or range"
        fragments_as_residue_idxs = []
        temp_fragments=get_fragments(top,verbose=False)
        for fri in fragment_input:
            if not isinstance(fri,str):
                fragments_as_residue_idxs.append(fri)
            else:
                fragments_as_residue_idxs.append(
                    _mdcu.residue_and_atom.rangeexpand_residues2residxs(fri,
                                                                        temp_fragments,
                                                                        top,
                                                                        interpret_as_res_idxs=fri.replace("-","").replace(",","").isnumeric(),
                                                                        extra_string_info="\nThis fragmentation is only for disambiguation purposes:"
                                                                        ))
        if len(fragment_input)==1:
            assert isinstance(fragment_input[0],str)
            method += " with only one fragment provided (all other residues are fragment 2)"
            fragments_as_residue_idxs.append(_np.delete(_np.arange(top.n_residues), fragments_as_residue_idxs[0]))

    _mdcu.lists.assert_no_intersection(fragments_as_residue_idxs, word="fragments")
    for ii, ifrag in enumerate(fragments_as_residue_idxs):
        if not all([aa in range(top.n_residues) for aa in ifrag]):
            print("Fragment %u's definition had idxs outside of "
                  "the geometry (total n_residues %u) that have been deleted %s " % (
                  ii, top.n_residues, set(ifrag).difference(range(top.n_residues))))
            fragments_as_residue_idxs[ii]=[jj for jj in ifrag if jj<top.n_residues]

    if verbose:
        print("Using method '%s' these fragments were found" % method)
        for ii, ifrag in enumerate(fragments_as_residue_idxs):
            print_frag(ii, top, ifrag, label_width=0)

    return fragments_as_residue_idxs, user_wants_consensus

def frag_list_2_frag_groups(frag_list,
                            frag_idxs_group_1=None,
                            frag_idxs_group_2=None,
                            verbose=False):
    r"""
    Automagically find out the user wants to define
    two fragments out of list of fragments. This is used
    by CLTs interface

    Prompt the user when disambiguation is needed.

    Parameters
    ----------
    frag_list : list
        list of fragments, defined as residue indices
    frag_idxs_group_1 : iterable of ints, or str default is None
        When str, it has to be a rangeexpand exprs, e.g. 2-5,10
    frag_idxs_group_2 : iterable of ints, or str default is None
        When str, it has to be a rangeexpand exprs, e.g. 2-5,10

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
                groups_as_fragidxs[ii] = _mdcu.lists.rangeexpand(ifrag_idxs.strip(","))
    groups_as_residxs = [sorted(_np.hstack([frag_list[ii] for ii in iint])) for iint in
                            groups_as_fragidxs]

    return groups_as_residxs, groups_as_fragidxs

def frag_dict_2_frag_groups(frag_defs_dict, ng=2,
                            verbose=False,
                            answers=None):
    r"""
    Input a dictionary of fragment definitions, keyed by
    whatever and valued with residue idxs and prompt
    the user how to re-group them

    It wraps around :obj:`_match_dict_by_patterns`
    under the hood

    TODO: refactor into str_and_dict_utils
    TODO: It will be mostly used with fragments so it's better here for the API? IDK

    Parameters
    ----------
    frag_defs_dict : dict
        Fragment definitions in residue idxs
    ng : int, default is 2
        wanted number of groups
    answers : list, default is None
        List of strings. If provided,
        the items of this list will
        be passed as answers to the prompt
        asking for fragment choice. None and

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
    _answers = [None]*ng
    if answers is not None:
        for ii, ians in enumerate(answers):
            if isinstance(ians,str):
                _answers[ii]=ians
    answers = _answers

    if verbose:
        for key, val in frag_defs_dict.items():
            print("%s: %u-%u"%(key, val[0],val[-1]))
    for ii in range(1, ng + 1):
        print("group %u: " % ii, end='')
        if answers[ii-1] is None:
            answer = input(
                "Input a list of comma-separated posix-expressions.\n"
                "Prepend with '-' to exclude, e.g. 'TM*,-TM2,H8' to grab all TMs and H8, but exclude TM2)\n").replace(
                " ", "").strip("'").strip('"')
        else:
            answer = answers[ii-1]
        igroup, res_idxs_in_group = _mdcu.str_and_dict.match_dict_by_patterns(answer, frag_defs_dict)
        groups_as_keys.append([ilab for ilab in frag_defs_dict.keys() if ilab in igroup])
        groups_as_residue_idxs.append(sorted(res_idxs_in_group))
        print(', '.join(groups_as_keys[-1]))

    return groups_as_residue_idxs, groups_as_keys

def splice_orphan_fragments(fragments, fragnames, highest_res_idx=None,
                            orphan_name="?",
                            other_fragments=None):
    r"""
    Return a fragment list where residues not present in :obj:`fragments` are
    now new interstitial ('orphan') fragments.

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
        If the str contains a '%' character
        it will be used as a format identifier
        to use as orphan_name%ii
    other_fragments : dict, default is None
        A second set of fragment-definitions.
        If these other fragments are contained
        in the newly found orphans, then the
        orphans are re-shaped and renamed
        using this info. Typical usecase
        is for :obj:`fragments` to be
        consensus fragments (that don't
        necessarily cover the whole topology)
        and :obj:`other_fragments` to
        come from :obj:`fragments.get_fragments`
        and cover the whole topology


    Returns
    -------
    new_fragments : list
        The new fragments including those missing in the input
    new_names : list
        The new names, where the "orphans" have gotten the name orphan_char

    """
    assert len(fragments)==len(fragnames)
    if highest_res_idx is None:
        highest_res_idx = _np.max(_np.hstack(fragments))

    full_frags = []
    for ifrag, iname in zip(fragments,fragnames):
        if len(ifrag)==ifrag[-1]-ifrag[0]:
            print("Fragment %s has holes in it but will be considered from %u to %u regardless"%(iname,ifrag[0],ifrag[-1]))
        full_frags.append(_np.arange(ifrag[0],ifrag[-1]+1))
    orphans = _np.delete(_np.arange(highest_res_idx + 1), _np.hstack(full_frags))
    if len(orphans)>0:
        orphans = _get_fragments_by_jumps_in_sequence(orphans)[1]
        if other_fragments is not None:
            # The orphans have to be split using the limits of the other_fragments
            orphans = _break_fragments([frag[0] for frag in other_fragments.values()], orphans)
        if '%' in orphan_name:
            orphans_labels = [orphan_name%ii for ii, __ in enumerate(orphans)]
        else:
            orphans_labels = [orphan_name for __ in orphans]
        # The idea is that now orphans could be supersets of existing
        # fragments, s.t.
        if other_fragments is not None:
            popped = []
            for ii in range(len(orphans)):
                for xname,xfrag in other_fragments.items():
                    if xname not in popped and set(xfrag).issubset(orphans[ii]):
                        orphans[ii] = sorted(set(orphans[ii]).difference(xfrag))
                        popped.append(xname)
                        orphans.append(xfrag)
                        orphans_labels.append(xname)
        still_orphans = [ii for ii, oo in enumerate(orphans) if len(oo)>0]
        new_frags = [orphans[oo] for oo in still_orphans] + full_frags
        new_labels =[orphans_labels[oo] for oo in still_orphans] + fragnames
        idxs = _np.argsort([ifrag[0] for ifrag in new_frags])
        new_frags, new_names = [list(new_frags[ii]) for ii in idxs], [new_labels[ii] for ii in idxs]

        return new_frags, new_names
    else:
        return fragments, fragnames

def mix_fragments(highest_res_idx, consensus_frags, fragments, fragment_names):
    r"""Mix consensus frags with user-provided fragment definitions

    Wrapper around :obj:`mdciao.fragments.splice_orphan_fragments`,
    will possibly be merged with it in the future.
    The wrapper does several pre and post processing things.
    pre-processing:
     * makes the `consensus_frags` the main frags
     and the `fragments` the `other_fragments`.
     * It allows for `fragments` and/or `fragment_names`
     to be None and creates names on the fly if needed
    post-processing:
     * renames orphan fragments as sub-fragments of original
      fragments if possible
    In theory, if `fragments` covers all residues, no
    orphan's should be left

    Note
    ----
    This is an internal ad-hoc method for
    developing the best integration of
    the logic behind the fragmentation
    for flareplots, and might disappear in
    in the future. Not intended for API
    use

    Parameters:
    ----------
    highest_res_idx : int
        Typically top.n_residues - 1
    consensus_frags : dict
    fragments : list or None
        User provided fragments
    fragment_names : list or None
        User provided fragment names

    Returns:
    -------
    new_frags : list
        The new fragments
    new_names : list
        The new names
    """

    if fragments is not None:
        if fragment_names is not None:
            assert len(fragments)==len(fragment_names)
            other_frags = {fn: fr for fn, fr in zip(fragment_names, fragments)}
        else:
            other_frags = {"frag %u" % ii: fn for ii, fn in enumerate(fragments)}
    else:
        other_frags = None

    new_frags, new_names = splice_orphan_fragments(list(consensus_frags.values()),
                                          list(consensus_frags.keys()),
                                          highest_res_idx=highest_res_idx,
                                          other_fragments=other_frags,
                                          orphan_name="orphan %u")
    # todo put this into splice_orphan_fragments if it makes sense
    # reassign the orphans as sub-fragments of the parents, in case there's parents
    orphan_idxs = [ii for ii, nn in enumerate(new_names) if nn.startswith("orphan ")]
    if len(orphan_idxs) > 0 and other_frags is not None:
        parent_names = list(other_frags.keys())
        __, child = _mdcu.lists.find_parent_list([new_frags[oo] for oo in orphan_idxs],
                                                 list(other_frags.values()))
        for parent_idx, list_of_children in child.items():
            for idx, oo in enumerate(list_of_children):
                new_names[orphan_idxs[oo]] = "subfrag %u of %s"%(idx, parent_names[parent_idx])

    return new_frags, new_names


def flarekwargs_preparer(fragments, fragment_names, kwargs_freqs2flare, fixed_color_list, to_intersect_with):
    r"""
    Prepare the kwargs dictionary to call :obj:`mdciao.flare.freqs2flare` from a :obj:`mdciao.contacts.ContactGroup`


    Note
    ----
    This is a helper method that might disappear or get refactored somewhere.
    The motivation for this method is to keep freqs2flare as agnostic
    as possible with respect to things like contact-groups, interfaces,
    consensus labelers, sub-domains etc, so that the method's signature
    (already pretty long) is clear in cases where none of the above is
    needed and can be generally used to plot any set of values that have
    pair-relations associated with them.

    However, this decision somehow blurries the logic behind "fragment and residue selection"
    spreading it across two methods, because there's a partial selection
    before calling freqs2flare and a further selection inside freqs2flare.
    The upside (apart from the above mentioned conservation of generality) is
    that it allows for consistency in fragment colors and names when
    repeatedly calling the plot_freqs_as_flareplot when changing the 'scheme'
    parameter.

    Parameters
    ----------
    fragments : None or list of lists
    fragment_names : None or list of strings
    kwargs_freqs2flare : dict
    fixed_color_list : list of matplotlib colors
    to_intersect_with : the group of residues that need to be present

    Returns
    -------
    kwargs_freqs2flare : dict
    good_frags : list
    """
    good_frags = []
    if fragments is not None:
        good_frags = [ii for ii, fr in enumerate(fragments) if
                      len(_np.intersect1d(fr, to_intersect_with)) > 0]
        kwargs_freqs2flare["fragments"] = [fragments[ii] for ii in good_frags]
        kwargs_freqs2flare["colors"] = fixed_color_list[good_frags]
        if fragment_names is not None:
            assert len(fragment_names) <= len(fragment_names), \
                ValueError("Less fragment names (%u) than fragments (%u)?" % (len(fragment_names), len(fragment_names)))
            kwargs_freqs2flare["fragment_names"] = [fragment_names[ii] for ii in good_frags]
    return kwargs_freqs2flare, good_frags

def assign_fragments(res_idxs, fragments, raise_on_missing=True):
    r"""Assign a parent fragment to each residue in a list of res_idxs

    Note
    ----
    Simply wraps around :obj:`~mdciao.utils.lists.in_what_N_fragments`,
    first asserting that there's no intersection between the fragments
    and then checking whether some residxs are missing

    Parameters
    ----------
    res_idxs : iterable of ints
        The residue indices
    fragments : iterable of iterable if ints
        The fragment definitions.
        It will be checked that these definitions
        don't have common residues
    raise_on_missing : bool, default is True
        Whether to raise an Exception if
        any residue can't be found
        in the :obj:`fragments`

    Returns
    -------
    frag_idxs : np.array
        Array with the indices of
        :obj:`fragments` where each
        residue of :obj:`res_isxs`
        appears
    res_idxs : np.array
        If all residues were
        found in :obj:`fragments`,
        then this is a copy of the
        input array. If some didn't
        appear anywhere, but
        :obj:`raise_on_missing` was
        False, the missing residues
        have been deleted
    """
    #TODO very related to utils.lists.find_parent_list, perhaps merge?
    _mdcu.lists.assert_no_intersection(fragments, word="fragments")
    frag_idxs = _mdcu.lists.in_what_N_fragments(res_idxs, fragments)
    orphans = _np.flatnonzero([len(par) == 0 for par in frag_idxs])
    frag_idxs = _np.squeeze(frag_idxs)
    if len(orphans)>0:
        if raise_on_missing:
            raise ValueError("residues %s don't appear in any 'fragments'. "
                             "If you're OK with this, set 'check_if_subset=False'" % (_np.array(res_idxs)[orphans]))
        else:
            res_idxs = _np.delete(res_idxs, orphans)
            frag_idxs = _np.delete(frag_idxs, orphans)

    return frag_idxs, res_idxs
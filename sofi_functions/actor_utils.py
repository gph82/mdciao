import numpy as _np
import mdtraj as _md
from scipy.spatial.distance import pdist
from matplotlib import pyplot as _plt

from sofi_functions.fragments import get_fragments as _get_fragments

from sofi_functions.list_utils import in_what_N_fragments, in_what_fragment, exclude_same_fragments_from_residx_pairlist

#Import needed for the command line scripts
#from sofi_functions.tested_utils import unique_list_of_iterables_by_tuple_hashing

"""
fragment 0 with 203 AAs GLU30-LYS232 NT-TM5
fragment 1 with  78 AAs PHE264-CYSP341 TM6-CT
fragment 2 with  69 AAs CYSP2-GLY70 Gprot1 NT(Gα)-S3(Gα)
fragment 3 with 310 AAs ASP85-LEU394 Gprot1 α-helical domain(Gα)-CT(Gα)
fragment 4 with 339 AAs SER2-ASN340 Gprot3 NT(Gβ)-CT(Gβ)
fragment 5 with  67 AAs ALA2-CYSG68 Gprot4 NT(Gγ)-CT(Gγ)
fragment 6 with   2 AAs P0G395-GDP396 ligands
"""

fragment_names =  [
"NT-TM5",
"TM6-CT",
"NT(Gα)-S3(Gα)",
"domain(Gα)-CT(Gα)",
"NT(Gβ)-CT(Gβ)",
"NT(Gγ)-CT(Gγ)",
"ligands",
]
fragment_names_short =  [
"B2AR",
"B2AR",
"Gα-S3",
"Gα-αhlx",
"Gβ",
"Gγ",
"LIG",
]

fragment_names_B2AR = [
    'TM1', 'ICL1',
    "TM2", 'ECL1',
    'TM3', 'ICL2',
    'TM4', 'ECL2',
    'TM5', 'ICL3',
    'TM6', 'ECL3',
    'TM7', 'TM78',
    'H8'
]


def identify_boolean_blocks_in_sequence(binary_vector,
                                        minimum_block_length=1,
                                        max_block_interruption=0,
                                        verbose=False,
                                        names=["block","break"],
                                        ):
    blocks = []
    icounter = []
    ibreak = []

    for ii, ihx in enumerate(binary_vector):
        if ihx:
            icounter.append(ii)
            ibreak = []
        else:
            ibreak.append(ii)
        if verbose:
            print(ii, ihx, end=' ')
            if len(icounter) != 0:
                print('%u-%u' % (icounter[0], icounter[-1]), end=' ')
            else:
                print(icounter, end=' ')
            if len(ibreak) != 0:
                print('%u-%u' % (ibreak[0], ibreak[-1]))
            else:
                print(ibreak)

        if len(ibreak) > max_block_interruption:
            if len(icounter) > (minimum_block_length):
                if verbose:
                     print(names[0], icounter[0], icounter[-1])
                     print(names[1], ibreak[0], ibreak[-1])
                blocks.append(icounter)
                ibreak = []
            icounter = []

        # Catch the last break:
    if len(icounter) > minimum_block_length:
        if verbose:
            print(names[0], icounter[0], icounter[-1])
            try:
                print(names[1], ibreak[0], ibreak[-1])
            except IndexError:
                pass

        blocks.append(icounter)

    blocks = [_np.arange(ihx[0],ihx[-1]+1) for ihx in blocks]

    if _np.sum(binary_vector[:_np.ceil(max_block_interruption).astype(int)])>0:
        blocks[0]=_np.arange(0, blocks[0][-1]+1)

    return blocks

def identify_long_helices(geom, min_turns=5, aa_per_turn=3.6,
                          verbose=False, plot=False):



    ss_str = _md.compute_dssp(geom)[0]
    ss_vec = _np.zeros(len(ss_str), dtype=int)
    ss_vec[ss_str == 'H'] = 1

    helices = identify_boolean_blocks_in_sequence(ss_vec,
                                                  min_turns*aa_per_turn,
                                                  aa_per_turn,
                                                  verbose=verbose,
                                                  names=["hlx","brk"])
    if plot:
        _plt.figure()
        _plt.figure(figsize=(20, 5))
        _plt.plot(ss_vec, marker='.')
        iax = _plt.gca()
        xticks = _np.arange(geom.n_residues, step=15)
        iax.set_xticks(xticks)
        iax.set_xticklabels([geom.top.residue(ii).resSeq for ii in xticks])
        #iax.set_xticklabels([ii for ii in xticks])
        for ihx in helices:
            iax.axvspan(ihx[0]-.5, ihx[-1]+.5, alpha=.25)

        return helices, _plt.gca()
    else:
        return helices


def find_B_residues_in_A_using_fragment_info(
        fragmentsB,
        topB,
        fragmentsA,
        topA,
        fragA_to_fragB_dict):
    residxB_to_residxA_dict = {}

    for ii, ifragA in enumerate(fragmentsA):
        idx_fragB = fragA_to_fragB_dict[ii]
        ifragB = fragmentsB[idx_fragB]
        ifragB_resSeqs = _np.array([topB.residue(ii).resSeq for ii in ifragB])
        for residx in ifragA:
            residue_A = topA.residue(residx)
            match = _np.argwhere(ifragB_resSeqs == residue_A.resSeq).squeeze()
            # print(match)
            try:
                equivB_residue = topB.residue(ifragB[match])
            except TypeError:
                pass

            if str(equivB_residue) == str(residue_A):
                residx_B = equivB_residue.index
                residxB_to_residxA_dict[residx_B] = residue_A.index
                """
                if residue_A.n_atoms!=equivB_residue.n_atoms:
                    for aa in residue_A.atoms:
                        print(aa)
                    print()
                    for aa in equivB_residue.atoms:
                        print(aa)
                    raise
                """

        #if len(fr) == len(ft) != 0:
            #common_frags_ref_residxs.append(_np.array(fr))
            #common_frags_traj_residxs.append(_np.array(ft))
        # print()
    return residxB_to_residxA_dict

def re_warp_idxs(lengths):
    """Return iterable with the indexes to reshape a vector
    in the shapes specified in in lengths

    Parameters
    ----------

    lengths : int or iterable of integers
        Lengths of the individual elements of the returned array. If only one int is parsed, all lengths will
        be that int

    Returns
    -------
    warped: list
    """

    idxs_out = []
    idxi = 0
    for ll in lengths:
        idxs_out.append(_np.arange(idxi, idxi + ll))
        idxi += ll
    return idxs_out

def interactive_fragment_picker_by_resSeq(resSeq_idxs, fragments, top,
                                          pick_first_fragment_by_default=False,
                                          additional_naming_dicts=None):
    resSeq2residxs = {}
    resSeq2segidxs = {}
    last_answer = 0
    auto_last_answer_flag=False
    for key in resSeq_idxs:
        cands = [rr.index for rr in top.residues if rr.resSeq == key]
        # print(key)
        cand_fragments = in_what_N_fragments(cands, fragments)
        if len(cands) == 0:
            print("No residue found with resSeq %s"%key)
        else:
            if len(cands) == 1:
                cands = cands[0]
                answer = cand_fragments
                # print(key,refgeom.top.residue(cands[0]), cand_fragments)
            elif len(cands) > 1:
                print("ambigous definition for resSeq %s" % key)
                for cc, ss in zip(cands, cand_fragments):
                    istr = '%10s in fragment %2u with index %6u'%(top.residue(cc), ss,cc)
                    if additional_naming_dicts is not None:
                        extra=''
                        for key1, val1 in additional_naming_dicts.items():
                            if val1[cc] is not None:
                                extra +='%s: %s '%(key1,val1[cc])
                        if len(extra)>0:
                            istr = istr + ' (%s)'%extra.rstrip(" ")
                    print(istr)
                if not pick_first_fragment_by_default:
                    answer = input(
                        "input one fragment idx (out of %s) and press enter.\nLeave empty and hit enter to repeat last option [%s]\n" % ([int(ii) for ii in cand_fragments], last_answer))
                    if len(answer) == 0:
                        answer = last_answer
                    try:
                        answer = int(answer)
                    except:
                        #TODO implent k for keeping this answer from now on
                        if isinstance(answer,str) and answer=='k':
                            pass
                        print("Your answer has to be an integer in the of the fragment list %s" % cand_fragments)
                        raise Exception
                    assert answer in cand_fragments, (
                                "Your answer has to be an integer in the of the fragment list %s" % cand_fragments)
                    cands = cands[_np.argwhere([answer == ii for ii in cand_fragments]).squeeze()]
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

mycolors=[
         'lightblue',
         'lightgreen',
         'salmon',
         'lightgray',
    ]
mycolors=[
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

def ctc_freq_reporter_by_residue_neighborhood(ctcs_mean, resSeq2residxs, fragments, ctc_residxs_pairs, top,
                                              n_ctcs=5, select_by_resSeq=None,
                                              silent=False,
                                              ):
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

def exclude_neighbors_from_residx_pairlist(pairlist, top, n_exclude,
                                           return_excluded_idxs=False,
                                           ):
    fragments = _get_fragments(top, verbose=False)
    fragment_list = [in_what_fragment(ii, fragments) for ii in range(top.n_residues)]
    idx2keep_anyway = _np.array([idx for idx, pair in enumerate(pairlist) if
                                 fragment_list[pair[0]] != fragment_list[pair[1]]])

    # Exclude nearest neighbors by resSeq allowing for same resSeq in different fragments
    idxs2exclude = _np.array([idx for idx, pair in enumerate(pairlist) if
                              _np.abs(_np.diff([top.residue(ii).resSeq for ii in pair])) <= n_exclude and
                              idx not in idx2keep_anyway])

    if not return_excluded_idxs:
        return [pair for ii, pair in enumerate(pairlist) if ii not in idxs2exclude]
    else:
        return idxs2exclude

def mean_geometry(list_of_trajs, reference=None,
                  atom_indices=None,
                  ref_atom_indices=None,
                  image_molecule=False):
    r"""
    Return the mean geometry for a list of :obj:`mdtraj.Trajectory` objects.
    The unitcell info will be either the one in the first frame of the
    trajectories or in the reference trajectory if it is provided


    Parameters
    ----------
    list_of_trajs : list of :obj:`mdtraj.Trajectory` objects
    reference : :obj:`mdtraj.Trajectory`, default is None
        If provided, the trajectories in :obj:`list_of_trajs` will be
        oriented to this reference before taking the mean.
        Warning
        -------
        This will alter the input trajectoies!
    atom_indices : iterable of integers, default is None
        If provided, these atoms will be used for the orienting
        the trajectories to the reference
    ref_atom_indices : iterable of integers, default is None
        In case the reference trajectory needs has different
        atom indexing
    image_molecule : boolean, default is False
        Make molecules whole in the PBC-box before orienting and averaging
        Can be very time consuming, it is recommended to perform this
        operation in a pre-processing step

    Returns
    -------
    mean_traj : :obj:`mdtraj.Trajectory`
        An trajectory with the mean xyz coordinates

    """

    if atom_indices is not None:
        assert reference is not None, ValueError("This input is impossible")
    if image_molecule:
        raise NotImplementedError("This is not yet implemented")

    mean_xyz = []
    for ii, itraj in enumerate(list_of_trajs):
        if reference is not None:
            itraj.superpose(reference,atom_indices=atom_indices,
                            ref_atom_indices=ref_atom_indices)
        if ii ==0 and reference is None:
            unitcell_angles = itraj._unitcell_angles[0]
            unitcell_lenghts = itraj._unitcell_lenghts[0]
        mean_xyz.append(itraj.xyz)

    mean_xyz = _np.vstack(mean_xyz).mean(0)

    if reference is not None:
        unitcell_angles =  reference._unitcell_angles[0]
        unitcell_lenghts = reference._unitcell_lengths[0]

    return _md.Trajectory(mean_xyz, topology=itraj.topology,
                          unitcell_lengths=unitcell_lenghts,
                          unitcell_angles=unitcell_angles)

def my_RMSD(geom, ref, atom_indices=None,
            ref_atom_indices=None,
            weights='masses',
            check_same_atoms=False,
            per_residue=True):
    r"""
    Compute the RMSD of a given :obj:`geom` wrt to a given :obj:`ref`.
    There is no pre-orienting of the geometries for maximum overlap

    Parameters
    ----------
    geom : obj:`mdtraj.Trajectory`
    ref : obj:`mdtraj.Trajectory`
    atom_indices : iterable of integers, default is None
        The group of atom indices for which to compute the RMSD, default is all
    ref_atom_indices: iterable of integers, default is None
        If :obj:`geom` and :obj:`ref` don't share atom indices, use this parameter
        to pass a map of :obj:`atom_indices` in :obj:`ref`. The default behaviour
        is to use the same indexing.
    weights : str, default is "masses
        How the average is weighted
    check_same_atoms : boolean, default is False :
        Check that the atoms chosen by :obj:`atom_indices` in :obj:`geom` and
        obj:`ref_atom_indices` in :obj:`ref` have the same name.
    per_residue: boolean, default is True

    Returns
    -------

    """
    if atom_indices is None:
        atom_indices = _np.arange(geom.n_atoms)

    if ref_atom_indices is None:
        ref_atom_indices=atom_indices

    if check_same_atoms:
        for aa_idx_geom, aa_idx_ref in zip(atom_indices,
                                           ref_atom_indices):
            geom_res, ref_res =     geom.top.atom(aa_idx_geom).residue, ref.top.atom(aa_idx_ref).residue
            geom_aname, ref_aname = geom.top.atom(aa_idx_geom).name,    ref.top.atom(aa_idx_ref).name

            err_msg = 'geom %s is not equal ref %s (idxs %u vs %u)'%\
                      (geom_res, geom_res, aa_idx_geom, aa_idx_ref)

            assert '%s-%s'%(geom_res.name, geom_aname) == '%s-%s'%(ref_res.name, ref_aname), err_msg
    if weights != 'masses':
        raise NotImplementedError
    masses = _np.zeros(geom.n_atoms)
    masses[atom_indices] = [geom.top.atom(ii).element.mass for ii in atom_indices]

    delta_per_frame_per_atom = _np.zeros((geom.n_frames, geom.n_atoms))
    delta_per_frame_per_atom[:,atom_indices] =_np.linalg.norm(geom.xyz[:,atom_indices,:]-ref.xyz[0,ref_atom_indices,:], axis=2)

    # I am summing and averaging over a lot of zeroes here,
    # but it makes for code readability
    rmsd = _np.sqrt(_np.average(delta_per_frame_per_atom** 2,
                                axis=1, weights=masses))

    if per_residue:
        average_delta2_per_residue = _np.zeros((geom.n_frames, geom.n_residues))
        weight_per_residue = _np.zeros(geom.n_residues)
        residue_idxs = _np.unique([geom.top.atom(ii).residue.index for ii in atom_indices])
        for res_idx in residue_idxs:
            rr = geom.top.residue(res_idx)
            rr_aa_idxs = [aa.index for aa in rr.atoms]
            average_delta2_per_residue[:,res_idx] = _np.average(delta_per_frame_per_atom[:,rr_aa_idxs]**2,
                                                weights=masses[rr_aa_idxs],
                                                axis=1)
            weight_per_residue[res_idx] = masses[rr_aa_idxs].sum()


    if not per_residue:
        return rmsd
    else:
        return rmsd, average_delta2_per_residue, weight_per_residue, residue_idxs

def xtcs2contactpairs_auto(xtcs, top, cutoff_in_Ang,
                           stride=1,
                           chunksize=500,
                           exclude_nearest_neighbors=2,
                           fragments=True):

    if isinstance(top,str):
        top = _md.load(top).top


    ctc_mins, ctc_pairs = xtcs2mindists(xtcs, top, chunksize=chunksize, stride=stride)

    ctc_pairs_cutoff = select_ctcmins_by_cutoff(ctc_mins, ctc_pairs,
                                                cutoff_in_Ang)
    #print("Before neighbors")
    #for pair in ctc_pairs_cutoff:
    #    print(pair)
    #print()
    ctc_pairs_cutoff = exclude_neighbors_from_residx_pairlist(ctc_pairs_cutoff, top, exclude_nearest_neighbors,
                                                              )

    #print("After neighbors before fragments")
    #for pair in ctc_pairs_cutoff:
    #    print(pair)
    #print()

    if fragments:
        ctc_pairs_cutoff = exclude_same_fragments_from_residx_pairlist(ctc_pairs_cutoff, top)

    #print("After fragments")
    #for pair in ctc_pairs_cutoff:
    #    print(pair)
    #print()

    ctc_pairs_cutoff = _np.array(ctc_pairs_cutoff)[_np.argsort([_np.ravel_multi_index(rr, (top.n_residues, top.n_residues)) for rr in ctc_pairs_cutoff])]

    return ctc_pairs_cutoff

def xtcs2mindists(xtcs, top,
                  stride=1,
                  chunksize=1000, **COM_kwargs):

    #TODO avoid code repetition with xtcs2ctcs
    inform = lambda ixtc, ii, running_f: print(
        "Analysing %20s with stride %u in chunks of %3u frames. chunks read %4u. frames read %8u" % (ixtc, stride, chunksize, ii, running_f),
        end="\r", flush=True)

    ctc_mins, ctc_pairs = [],[]
    for ii, ixtc in enumerate(xtcs):
        running_f = 0
        inform(ixtc, 0, running_f)
        ires = {}
        for jj, igeom in enumerate(_md.iterload(ixtc, top=top, stride=stride, chunk=_np.round(chunksize/stride))):
            running_f += igeom.n_frames
            inform(ixtc, jj, running_f)
            mins, pairs, pair_idxs = igeom2mindist_COMdist_truncation(igeom, **COM_kwargs)
            for imin, ipair, idx in zip(mins, pairs, pair_idxs):
                try:
                    ires[idx]["val"] = _np.min((ires[idx]["val"], imin))
                except:
                    ires[idx] = {"val":imin,
                                 "pair":ipair}

            #if jj==5:
            #   break

        pair_idxs = sorted(ires.keys())
        ctc_mins.append( _np.array([ires[idx]["val"] for idx in pair_idxs]))
        ctc_pairs.append(_np.array([ires[idx]["pair"] for idx in pair_idxs]))
    print()
    return ctc_mins, ctc_pairs

def select_ctcmins_by_cutoff(ctc_mins, ctc_idxs, cutoff_in_Ang):
    r"""
    Provided a list of values (ctc_mins) and a list of idx_pairs (ctc_idxs),
    return the
    :param ctc_mins:
    :param ctc_idxs:
    :param cutoff_in_Ang:
    :return:
    """
    ctc_pairs = []
    highest_res_idx = _np.max([_np.max(ilist) for ilist in ctc_idxs])
    matrix_of_already_appended = _np.zeros((highest_res_idx+1, highest_res_idx+1))
    for ii, (ictc, iidxs) in enumerate(zip(ctc_mins, ctc_idxs)):
        tokeep_bool = ictc <= cutoff_in_Ang / 10
        tokeep = _np.argwhere(tokeep_bool).squeeze()
        pairs2keep = iidxs[tokeep]
        for pair in pairs2keep:
            ii,jj = pair
            # Append pair only if it hasn't been appended yet
            if not matrix_of_already_appended[ii,jj]:
                ctc_pairs.append(pair)
                matrix_of_already_appended[ii,jj]=1
                matrix_of_already_appended[jj,ii]=1

    ctc_pairs = _np.array(ctc_pairs)
    return ctc_pairs

def igeom2mindist_COMdist_truncation(igeom,
                                     res_COM_cutoff_Ang=25,
                                     ):


    COMs_xyz = geom2COMxyz(igeom)

    COMs_dist_triu = _np.array([pdist(ixyz) for ixyz in COMs_xyz])


    COMs_under_cutoff = COM_n_from_COM_dist_triu(COMs_dist_triu,
                                                 cutoff_nm=res_COM_cutoff_Ang/10)

    COMs_under_cutoff_pair_idxs = _np.argwhere(COMs_under_cutoff.sum(0) >= 1).squeeze()
    pairs = _np.vstack(_np.triu_indices(igeom.n_residues, 1)).T[COMs_under_cutoff_pair_idxs]
    try:
        ctcs, ctc_idxs_dummy = _md.compute_contacts(igeom, pairs)
    except MemoryError:
        print("\nCould not fit %u contacts for %u frames into memory"%(len(pairs), igeom.n_frames))
        raise


    assert _np.allclose(pairs, ctc_idxs_dummy)

    return ctcs.min(0), pairs, COMs_under_cutoff_pair_idxs


def COM_n_from_COM_dist_triu(COM_dist_triu,
                             cutoff_nm=2.5,
                             ):
    assert _np.ndim(COM_dist_triu)==2
    COM_dist_bool_int = (COM_dist_triu<=cutoff_nm).astype("int")
    return COM_dist_bool_int


def geom2COMxyz(igeom):
    masses = [_np.hstack([aa.element.mass for aa in rr.atoms]) for rr in igeom.top.residues]
    COMs_res_time_coords = [_np.average(igeom.xyz[:, [aa.index for aa in rr.atoms], :], axis=1, weights=rmass) for rr, rmass in
            zip(igeom.top.residues, masses)]
    COMs_time_res_coords = _np.swapaxes(_np.array(COMs_res_time_coords),0,1)
    return COMs_time_res_coords

def __find_by_AAresSeq(top, key):
    return [rr.index for rr in top.residues if key == '%s%u' % (rr.code, rr.resSeq)]

from .nomenclature_utils import table2BW_by_AAcode
# TODO check if I can deleted this already
def _table2BW_by_AAcode(tablefile="GPCRmd_B2AR_nomenclature.xlsx",
                       modifications={"S262":"F264"},
                       keep_AA_code=True,
                       return_defs=False,
                       ):
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




def __csv_table2TMdefs_resSeq(csvfile="GPCR_MD_TMdefs.csv",
                             modifications={"S262":"F264"},
                             reduce_to_resSeq=True):
    out_dict = {}
    lines = open(csvfile).read().splitlines()

    # 1st pass, locate definition lines
    for ii, line in enumerate(lines):
        if line.startswith("TM") or line.startswith("H8"):
            current_TM = line.split(",")[0]
            out_dict[current_TM] = ii

    # 2nd pass, extract intervals as AAresSeq
    AA_dict = {}
    for key1, key2 in zip(list(out_dict.keys()), list(out_dict.keys())[1:]):
        AA_dict[key1] = [lines[ii] for ii in [out_dict[key1] + 1, out_dict[key2] - 1]]
    AA_dict = {key: [ival.split(",")[1] for ival in val] for key, val in AA_dict.items()}

    # modifications, mutations etc
    __ = {}
    for key, val in AA_dict.items():
        for patt, sub in modifications.items():
            val = [ival.replace(patt,sub) for ival in val]
        __[key] = val
    AA_dict = __
    # On dictionary: just-keep the resSeq
    if reduce_to_resSeq:
        AA_dict = {key: [''.join([ii for ii in ival if ii.isnumeric()]) for ival in val] for key, val in
                   AA_dict.items()}
        AA_dict = {key: [int(ival) for ival in val] for key, val in
                   AA_dict.items()}
    return AA_dict




def dangerously_auto_fragments(itop, with_receptor=True, **get_fragments_kwargs):
    if not with_receptor:
        raise NotImplementedError

    GPCRmd_definition_dict = csv_table2TMdefs_res_idxs(itop)
    extra_breakers = [str(itop.residue(val[0])) for val in GPCRmd_definition_dict.values()][1:]

    G_fragnames = _G_fragnames
    R_fragnames = list(GPCRmd_definition_dict.keys())

    breakers = [
            'ALA39',
            "PHE68",
            "THR190",
            "ALA352"
        ]
    fragments = _get_fragments(itop,
                              #verbose=True,
                              fragment_breaker_fullresname=extra_breakers+breakers,
                              auto_fragment_names=False,
                              **get_fragments_kwargs
    )

    return fragments, R_fragnames+G_fragnames

_G_fragnames = [
        "alphaN",
        "RAS_1", "AHD", "RAS_2", "alpha5",
        "beta","gamma"
    ]

def _replace4latex(istr):
    for gl in ['alpha','beta','gamma']:
        istr = istr.replace(gl,'$\\'+gl+'$')

    if '$' not in istr and any([char in istr for char in ["_"]]):
        istr = '$%s$'%istr
    return istr

def _int_from_AA_code(key):
    return int(''.join([ii for ii in key if ii.isnumeric()]))



def _guess_missing_BWs(input_BW_dict,top, restrict_to_residxs=None):

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

def reorder_geometry(itraj, new_order_of_res_idxs):
    r"""

    Parameters
    ----------
    itraj : :obj:`mdtraj.Trajectory` object
        Trajectory to reordered
    new_order_of_res_idxs : iterable of integers
        New order of residues

    Warning
    -------
        The PBC elements are lost in this re-ordering

    Returns
    -------
    jtraj : :obj:`mdtraj.Trajectory`
        Trajectory with the residues ordered according to
        :obj:`new_order_of_res_idxs`

    """
    itop = _md.Topology()
    ixyz = []
    ichain = itop.add_chain()
    for ii, idx in enumerate(new_order_of_res_idxs):
        rr = itraj.top.residue(idx)
        rr.index = ii
        itop.add_residue(rr.name, rr.chain, resSeq=rr.resSeq)
        for aa in rr.atoms:
            itop.add_atom(aa.name, aa.element, itop.residue(ii))

        ixyz.append([itraj.xyz[:, aa.index, :] for aa in rr.atoms])
        ichain._residues.append(itop.residue(ii))
        # break

    ixyz = _np.vstack(ixyz).squeeze()
    #print(itop.n_residues)
    #print(itop.n_atoms)
    #print(ixyz.shape)
    igeom = _md.Trajectory([ixyz], topology=itop)
    return igeom
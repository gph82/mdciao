import numpy as _np
import mdtraj as _md
from .list_utils import in_what_fragment, re_warp

def ctc_freq_reporter_by_residue_neighborhood(ctc_freqs, resSeq2residxs, fragments, ctc_residxs_pairs, top,
                                              n_ctcs=5, restrict_to_resSeq=None,
                                              interactive=False,
                                              ):
    """Prints a formatted summary of contact frequencies AND
       returns a dictionary of neighborhoods

    Parameters
    ----------
    ctc_freqs: iterable of floats
        Contact frequencies between 0 and 1
    resSeq2residxs: dictionary
        Dictionary mapping residue sequence numbers (resSeq) to residue idxs
    fragments: iterable of integers
        Fragments of the topology defined as list of non-overlapping residue indices
    ctc_residxs_pairs: iterable of integer pairs
        The residue pairs for which the contact frequencies in :obj:`ctc_freqs`
        were computed.
    top : :py:class:`mdtraj.Topology`
    n_ctcs : integer, default is 5
        Number of contacts to report per residue.
    restrict_to_resSeq: int, default is None
        Produce the report only for the residue with this resSeq index. Default
        behaviour is to produce for all residues in :obj:`resSeq2residxs`
    interactive : boolean, default is False
        After reporting each neighborhood up to :obj:`n_ctcs` partners,
        ask the user how many should be kept

    Returns
    -------
    neighborhood : dictionary
       neighborhood[300] = [100,101,102,200,201]
       means that residue 300 has residues [100,101,102,200,201]
       as most frequent neighbors (up to n_ctcs or less, see option 'interactive')
    """
    order = _np.argsort(ctc_freqs)[::-1]
    assert len(ctc_freqs) == len(ctc_residxs_pairs)
    neighborhood = {}
    if restrict_to_resSeq is None:
        restrict_to_resSeq = list(resSeq2residxs.keys())
    elif isinstance(restrict_to_resSeq, int):
        restrict_to_resSeq = [restrict_to_resSeq]
    for key, val in resSeq2residxs.items():
        if key in restrict_to_resSeq:
            order_mask = _np.array([ii for ii in order if val in ctc_residxs_pairs[ii]])
            print("#idx    Freq  contact             segA-segB residxA   residxB   ctc_idx")

            isum = 0
            seen_ctcs = []
            for ii, oo in enumerate(order_mask[:n_ctcs]):
                pair = ctc_residxs_pairs[oo]
                if pair[0] != val and pair[1] == val:
                    pair = pair[::-1]
                elif pair[0] == val and pair[1] != val:
                    pass
                else:
                    print(pair)
                    raise Exception
                idx1 = pair[0]
                idx2 = pair[1]
                s1 = in_what_fragment(idx1, fragments)
                s2 = in_what_fragment(idx2, fragments)
                imean = ctc_freqs[oo]
                isum += imean
                seen_ctcs.append(imean)
                print("%-6s %3.2f %8s-%-8s    %5u-%-5u %7u %7u %7u %3.2f" % (
                 '%u:' % (ii + 1), imean, top.residue(idx1), top.residue(idx2), s1, s2, idx1, idx2, oo, isum))
            if interactive:
                try:
                    answer = input("How many do you want to keep (Hit enter for None)?\n")
                except KeyboardInterrupt:
                    break
                if len(answer) == 0:
                    pass
                else:
                    answer = _np.arange(_np.min((int(answer), n_ctcs)))
                    neighborhood[val] = order_mask[answer]
            else:
                seen_ctcs = _np.array(seen_ctcs)
                n_nonzeroes = (seen_ctcs > 0).astype(int).sum()
                answer = _np.arange(_np.min((n_nonzeroes, n_ctcs)))
                neighborhood[val] = order_mask[answer]
    # TODO think about what's best to return here
    # TODO think about making a pandas dataframe with all the above info
    return neighborhood



def xtcs2ctcs(xtcs, top, ctc_residxs_pairs, stride=1, consolidate=True,
              chunksize=1000, return_time=False,**mdcontacts_kwargs):
    """Returns the time-dependent traces of residue-residue contacts from a list of trajectory files

    Parameters
    ----------
    xtcs : list of strings
        list of filenames with trajectory data. Typically xtcs,
        but can be any type of file readable by :obj:mdtraj
    top : str or :py:class:`mdtraj.Topology`
        Topology that matches :obj:xtcs
    ctc_residxs_pairs : iterable
        List of (zero-indexed) residue pairs
    stride : int, default is 1
        Stride the trajectory data down by this value
    consolidate : boolean, default is True
        Return the time-traces consolidated
        into one array by using np.vstack
    chunksize : integer, default is 1000
        How many frames will be read into memory for
        computation of the contact time-traces. The higher the number,
        the higher the memory requirements
    return_time : boolean, default is False
        Return also the time array in ps

    Returns
    -------
    ctcs, or
    ctcs, times if return_time=True

    """
    ctcs = []
    times = []

    if isinstance(xtcs[0],_md.Trajectory):
        iterate = lambda ixtc : [ixtc[idxs] for idxs in re_warp(_np.arange(ixtc.n_frames)[::stride],chunksize)]
        inform = lambda ixtc, ii, running_f: print("Analysing a trajectory object in chunks of "
                                                   "%3u frames. chunks %4u frames %8u"%
                                                   (chunksize, ii, running_f), end="\r", flush=True)
    else:
        iterate = lambda ixtc: _md.iterload(ixtc, top=top, stride=stride, chunk=_np.round(chunksize / stride))
        inform = lambda ixtc, ii, running_f: print("Analysing %20s in chunks of "
                                                   "%3u frames. chunks %4u frames %8u" %
                                                   (ixtc, chunksize, ii, running_f), end="\r", flush=True)

    for ii, ixtc in enumerate(xtcs):
        ictcs = []
        running_f = 0
        inform(ixtc, 0, running_f)
        itime = []
        for jj, igeom in enumerate(iterate(ixtc)):
            running_f += igeom.n_frames
            inform(ixtc, jj, running_f)
            itime.append(igeom.time)
            jctcs, jidx_pairs = _md.compute_contacts(igeom, ctc_residxs_pairs,**mdcontacts_kwargs)
            # TODO do proper list comparison and do it only once
            assert len(jidx_pairs)==len(ctc_residxs_pairs)
            ictcs.append(jctcs)
            # if jj==10:
            #    break

        times.append(_np.hstack(itime))
        ictcs = _np.vstack(ictcs)
        # print("\n", ii, ictcs.shape, "shape ictcs")
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


def contact_matrix(trajectories, cutoff_Ang=3,
                      n_frames_per_traj=20, **mdcontacts_kwargs):
    r"""
    Return a matrix with the contact frequency for **all** possible contacts
    over all available frames
    Parameters
    ----------
    trajectories: list of obj:`mdtraj.Trajectory`
    n_frames_per_traj: int, default is 20
        Stride the trajectories so that, on average, this number of frames
        is used to compute the contacts
    mdcontacts_kwargs

    Returns
    -------
    ctc_freq : square 2D np.ndarray

    """

    top = trajectories[0].top
    n_res = top.n_residues
    mat = _np.zeros((n_res, n_res))
    ctc_idxs = _np.vstack(_np.triu_indices_from(mat, k=0)).T

    stride=_np.ceil(_np.sum([itraj.n_frames for itraj in trajectories])/n_frames_per_traj).astype(int)

    actcs = xtcs2ctcs(trajectories, top, ctc_idxs, stride=stride, chunksize=50,
                      consolidate=True, ignore_nonprotein=False, **mdcontacts_kwargs)

    actcs = (actcs <= cutoff_Ang/10).mean(0)
    assert len(actcs)==len(ctc_idxs)
    non_zero_idxs = _np.argwhere(actcs>0).squeeze()

    for idx in non_zero_idxs:
        ii, jj = ctc_idxs[idx]

        mat[ii][jj]=actcs[idx]
        if ii!=jj:
            mat[jj][ii] = actcs[idx]

    return mat
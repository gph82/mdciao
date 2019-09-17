import numpy as _np
import mdtraj as _md
from .list_utils import in_what_fragment, re_warp

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
    top : :py:class:`mdtraj.Topology`
    n_ctcs : integer
        Default is 5.
    select_by_resSeq
    silent : boolean, optional

    Returns
    -------

    """
    order = _np.argsort(ctcs_mean)[::-1]
    assert len(ctcs_mean) == len(ctc_residxs_pairs)
    final_look = {}
    if select_by_resSeq is None:
        select_by_resSeq = list(resSeq2residxs.keys())
    elif isinstance(select_by_resSeq, int):
        select_by_resSeq = [select_by_resSeq]
    for key, val in resSeq2residxs.items():
        if key in select_by_resSeq:
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
                imean = ctcs_mean[oo]
                isum += imean
                seen_ctcs.append(imean)
                print("%-6s %3.2f %8s-%-8s    %5u-%-5u %7u %7u %7u %3.2f" % (
                 '%u:' % (ii + 1), imean, top.residue(idx1), top.residue(idx2), s1, s2, idx1, idx2, oo, isum))
            if not silent:
                try:
                    answer = input("How many do you want to keep (Hit enter for None)?\n")
                except KeyboardInterrupt:
                    break
                if len(answer) == 0:
                    pass
                else:
                    answer = _np.arange(_np.min((int(answer), n_ctcs)))
                    final_look[val] = order_mask[answer]
            else:
                seen_ctcs = _np.array(seen_ctcs)
                n_nonzeroes = (seen_ctcs > 0).astype(int).sum()
                answer = _np.arange(_np.min((n_nonzeroes, n_ctcs)))
                final_look[val] = order_mask[answer]

    # TODO think about what's best to return here
    return final_look

    # These were moved from the method to the API
    final_look = _np.unique(_np.hstack(final_look))
    final_look = final_look[_np.argsort(ctcs_mean[final_look])][::-1]
    return final_look


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

r"""
Functions related to contact-map operations (WIP)

.. autosummary::
   :nosignatures:
   :toctree: generated/


"""

import numpy as _np
import mdtraj as _md
import os.path as _path
from subprocess import run as _run
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

    actcs = trajs2ctcs(trajectories, top, ctc_idxs, stride=stride, chunksize=50,
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

def contact_matrix_slim(trajectories, cutoff_Ang=3,
                       **mdcontacts_kwargs):
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

    actcs = trajs2ctcs(trajectories, top, ctc_idxs, stride=stride, chunksize=50,
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

def contact_map_to_dict(imat, top,
                        res_idxs=None,
                        consensus_labels_map=None,
                        ctc_freq_cutoff=0.01):

    if res_idxs is None:
        res_idxs = _np.arange(top.n_residues)

    if consensus_labels_map is None:
        consensus_labels_map = {key:None for key in res_idxs}
    assert imat.shape[0] == imat.shape[1] == len(res_idxs), (imat.shape, len(res_idxs))
    dict_out = {}
    for ii, jj in _np.array(_np.triu_indices_from(imat, k=1)).T:
        # print(ii,jj)
        val = imat[ii, jj]
        ii, jj = [res_idxs[kk] for kk in [ii, jj]]
        key = '%s@%s-%s@%s' % (top.residue(ii), consensus_labels_map[ii],
                               top.residue(jj), consensus_labels_map[jj])
        key = key.replace("@None","").replace("@none","")
        if val > ctc_freq_cutoff:
            dict_out[key] = val
    return dict_out

def xtcs2ctc_mat_dict(xtcs, top, list_ctc_cutoff_Ang,
                      stride=1,
                      return_time=False,
                      res_COM_cutoff_Ang=25,
                      chunksize=100,
                      n_jobs=1,
                      progressbar=False,
                      **mdcontacts_kwargs):
    """Returns the full contact map of residue-residue contacts from a list of trajectory files

    Parameters
    ----------
    xtcs : list of strings
        list of filenames with trajectory data. Typically xtcs,
        but can be any type of file readable by :obj:mdtraj
    top : str or :py:class:`mdtraj.Topology`
        Topology that matches :obj:xtcs
    stride : int, default is 1
        Stride the trajectory data down by this value
    chunksize : integer, default is 100
        How many frames will be read into memory for
        computation of the contact time-traces. The higher the number,
        the higher the memory requirements
    n_jobs : int, default is 1
        to how many processors to parallellize


    Returns
    -------
    ctc_mat

    """

    from tqdm import tqdm

    if progressbar:
        iterfunct = lambda a : tqdm(a)
    else:
        iterfunct = lambda a : a

    ictc_mat_dicts_itimes = _Parallel(n_jobs=n_jobs)(_delayed(per_xtc_ctc_mat_dict)(top, itraj, list_ctc_cutoff_Ang, chunksize, stride, ii, res_COM_cutoff_Ang,
                                                                               **mdcontacts_kwargs)
                                            for ii, itraj in enumerate(iterfunct(xtcs)))

    ctc_maps = {key:[] for key in list_ctc_cutoff_Ang}
    times = []

    for ictc_map, itimes in ictc_mat_dicts_itimes:
        for key in list_ctc_cutoff_Ang:
            ctc_maps[key].append(ictc_map[key])
        times.append(itimes)
    actcs = ctc_maps
    times = times

    if not return_time:
        return actcs
    else:
        return actcs, times

# TODO many of these could in principle be named tuples but IDK if
# its worth the effort and documentation-sphinx headeach

def per_xtc_ctc_mat_dict(top, itraj, list_ctc_cutoff_Ang, chunksize, stride,
                         traj_idx, res_COM_cutoff_Ang,
                         **mdcontacts_kwargs):

    from .actor_utils import igeom2mindist_COMdist_truncation
    iterate, inform = iterate_and_inform_lambdas(itraj, chunksize, stride=stride, top=top)
    ictcs, itime, iaps = [],[],[]
    running_f = 0
    inform(itraj, traj_idx, 0, running_f)
    ctc_sum = {icoff: _np.zeros((top.n_residues, top.n_residues), dtype=int) for icoff in list_ctc_cutoff_Ang}

    for jj, igeom in enumerate(iterate(itraj)):
        running_f += igeom.n_frames
        inform(itraj, traj_idx, jj, running_f)
        itime.append(igeom.time)

        ctcs_mins, ctc_residxs_pairs, COMs_under_cutoff_pair_idxs  = igeom2mindist_COMdist_truncation(igeom,
                                                                                                      res_COM_cutoff_Ang,
                                                                                                      CA_switch=True)
        jctcs, jidx_pairs, j_atompairs = compute_contacts(igeom, ctc_residxs_pairs, **mdcontacts_kwargs)
        # TODO do proper list comparison and do it only once
        assert len(jidx_pairs) == len(ctc_residxs_pairs)
        for icoff in list_ctc_cutoff_Ang:
            counts = (jctcs<=icoff/10).sum(0)
            positive_idxs = _np.argwhere(counts>0).squeeze()
            for idx in positive_idxs:
                ii, jj  = sorted(jidx_pairs[idx])
                ctc_sum[icoff][ii,jj] += counts[idx]
                ctc_sum[icoff][jj,ii] += counts[idx]
    return ctc_sum, itime

def contact_map(
        topology=None,
        trajectories=None,
        chunksize_in_frames=100,
        list_ctc_cutoff_Ang=[3],
        graphic_dpi=150,
        graphic_ext=".pdf",
        interface_cutoff_Ang=35,
        output_desc="contact",
        output_dir=".",
        stride=1,
        n_jobs=1,
        scheme="closest-heavy",
):
    output_desc = output_desc.strip(".")
    _offer_to_create_dir(output_dir)

    xtcs = _get_sorted_trajectories(trajectories)
    print("Will compute contact maps for the files:\n  %s\n with a stride of %u frames.\n" % (
    "\n  ".join([str(ixtc) for ixtc in xtcs]), stride))

    refgeom = _load_any_top(topology)




    print()
    from .contacts import xtcs2ctc_mat_dict
    from .lists import force_iterable as _force_iterable
    ctc_map_dict, times = xtcs2ctc_mat_dict(xtcs, refgeom.top, _force_iterable(list_ctc_cutoff_Ang),
                                            stride=stride,
                                            return_time=True,
                                            chunksize=chunksize_in_frames,
                                            n_jobs=n_jobs,
                                            progressbar=True,
                                            scheme=scheme
                                            )
    print()

    panelheight = 3
    n_cols = 3
    n_rows = _np.ceil((len(xtcs)+1)/n_cols).astype(int)
    panelsize2font = 3.5
    print("The following files have been created:")
    for key, val in ctc_map_dict.items():
        matfig, matax = plt.subplots(n_rows, n_cols,
                                         sharex=True,
                                         sharey=True,
                                         figsize=(n_cols * panelheight,
                                                  n_rows * panelheight),

                                         )
        axiter = iter(matax.flatten())
        n_frames = 0
        for imat, itime, ixtc in zip(val, times, xtcs):
            iax = next(axiter)
            i_nframes = len(itime[0])
            iax.imshow(imat/i_nframes)
            iax.set_title("%s\n(%s frames)"%(splitext(str(ixtc))[0], str(i_nframes)))
            n_frames += i_nframes

        iax = next(axiter)
        av = _np.sum(val,0)/n_frames
        iax.imshow(av)
        iax.set_title("overall@%2.1f$\\AA$"%key)
        for ii in range(n_cols):
            try:
                iax = next(axiter)
                iax.set_frame_on(False)
                iax.set_xticks([])
                iax.set_yticks([])
                iax.set_xticklabels([])
            except StopIteration:
                break

        fname = "%s@%2.1f.overall.%s" % (output_desc, key, graphic_ext.strip("."))
        fname = path.join(output_dir, fname)
        matfig.savefig(fname, dpi=graphic_dpi, bbox_inches="tight")
        print(fname)
    return ctc_map_dict
    fname_excel = fname.replace(graphic_ext.strip("."),"xlsx")
    #neighborhood.frequency_spreadsheet(ctc_cutoff_Ang, fname_excel, sort=sort_by_av_ctcs)
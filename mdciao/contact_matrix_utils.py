import numpy as _np

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

def atom_idxs2indexfile(atom_idxs,
                        group_name,
                        indexfile="index.ndx",
                        offset=1,
                        overwrite=False):
    r"""
    Write the equivalent of a GROMACS index file

    Parameters
    ----------
    atom_idxs : iterable of integers
    group_name : str
    indexfile : str, default is index.ndx
    offset : int, default is 1
        GROMACS uses serial indices starting at 1
    overwrite : boolean, default is False
        Overwrite :obj:`indexfile` if it exists

    Returns
    -------
    None

    """
    n = 15
    from .list_utils import re_warp
    import os.path as _path
    if _path.exists(indexfile):
        if not overwrite:
            raise FileExistsError("%s exists. Use overwrite=True if needed"%indexfile)

    with open(indexfile,"w") as f:
        f.write("[ %s ]\n"%group_name)
        for iline in re_warp(atom_idxs,n):
            f.write((" ".join(["%4u"%(ii+offset) for ii in iline]))+"\n")


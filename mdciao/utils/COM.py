r"""
Functions related to Center-Of-Mass (COM) computations

.. autosummary::
   :nosignatures:
   :toctree: generated/


"""
import numpy as _np
from scipy.spatial.distance import pdist as _pdist
import mdtraj as _md
from tqdm import tqdm as _tqdm


def geom2COMdist(geom, residue_pairs, subtract_max_radii=False, low_mem=True, periodic=False):
    r"""
    Returns the distances between pairs of residues' center-of-mass (COM)

    The option `subtract_max_radii` can be used to produce a time-dependent lower bound
    on the distance between any atoms of each residue pair, i.e. a lower bound
    on the pairwise-residue "mindist". This lower bound can be used to discard
    any contact between some pairs residues in any frame of `geom`
    without having to compute all pairwise atom distances between the `residue_pairs`.

    Parameters
    ----------
    geom: :obj:`mdtraj.Trajectory`
    residue_pairs: iterable of integer pairs
        pairs of residues by their zero-indexed serial indexes
    subtract_max_radii : bool, default is False
        Subtract the sum of maximum radii (as computed by
        :obj:`~mdciao.utils.COM.geom2max_residue_radius`)
        of both residues of each residue pair from
        the residue-residue COM distance, effectively producing
        a lower bound for the residue-residue mindist.
        Please note that this option can produce negative values,
        meaning the (very normal situation) of two residue COMs
        being closer to each than the sum their max-radii.
    low_mem : bool, default is True
        Try to use less memory by using each residue's
        maximum radius for all frames (instead of a frame-dependent value)
        when subtracting pairs of radii from the COM distances.
        This results in an even lower value for lower bound
        of the COM-distances, which is itself still a lower-bond.
        For the porpuses of cutoff-thresholding, this "overshooting"
        has little consequence, see the note below for a benchmark case.

    Note
    ----
        In a benchmark case of 6K frames and ~100K `residue_pairs`,
        after using `subtract_max_radii=True`, we checked for `residue_pairs`
        with a lower-bound for their residue-residue distance
        smaller than a given cutoff, deeming them potential contacts.
        As expected, thresholding the values obtained with
        `low_mem=True` resulted in longer lists of potential
        contacts (between 20%-30% longer depending on the cutoff),
        and accordingly longer computation times when calling
        :obj:`mdtraj.compute_contacts` on the kept `residue_pairs`. However,
        those "longer" lists (and times) are not significant at all when expressed as a
        percentage of the initial `residue_pairs`:

        * Using a cutoff of 1 Angstrom when thresholding the mindist
          values obtained with `low_mem=False` discarded 99.46% of the original `residue_pairs`,
          whereas using the `low_mem=True` values meant discarding "only" 99.29%.
        * Using a cutoff of 6 Angstrom when thresholding the mindist
          values obtained with `low_mem=False` discarded 98.3% of the original `residue_pairs`,
          whereas using the `low_mem=True` values meant discarding "only" 98%.

        So there is still a huge reduction in the number of potential contact-pairs
        when thresholding, from ca 100K residue pairs to between 500 and 2K.
        For this benchmark system, `low_mem=True` used ~10GBs of memory
        and `low_mem=False` used ~15GB. There's a table included
        as a comment in the source code of the method showing the benchmark numbers.

    Returns
    -------
    COMs_array : np.ndarray of shape(geom.n_frames, len(res_pairs))
        contains the time-traces of the distance between residue COMs,
        optionally minus the sum of each pair of residue radii.
        Please note that this option can produce negative values,
        meaning the residue-spheres actually intersect at some point.
    """

    residue_pairs = _np.array(residue_pairs)
    residue_pairs.sort(axis=1)
    assert all(residue_pairs[:, 0] < residue_pairs[:, 1]) #For scipy.pdist to work
    residue_idxs_unique, pair_map = _np.unique(residue_pairs, return_inverse=True)
    n_unique_residues = len(residue_idxs_unique)
    pair_map = pair_map.reshape(len(residue_pairs),2)

    # From the _pdist doc
    # For each i and j (where i<j<m),where m is the number of original observations.
    # The metric dist(u=X[i], v=X[j]) is computed and stored in entry
    #  m * i + j - ((i + 2) * (i + 1)) // 2.
    _pdist_ravel = lambda i, j : n_unique_residues * i + j - ((i + 2) * (i + 1)) // 2
    _pdist_idxs = _pdist_ravel(pair_map[:,0], pair_map[:,1])

    COMs_xyz = geom2COMxyz(geom, residue_idxs=residue_idxs_unique)[:, residue_idxs_unique]

    # Only do pdist of the needed residues
    if not periodic:
        # Grab only the _pdist_idxs
        COM_dists_t =  _np.array([_pdist(ixyz)[_pdist_idxs] for ixyz in COMs_xyz])
    else:
        sum_over_comps2 = None
        for ii in range(3):
            comp_dist = _np.array([_pdist(_np.array(ixyz,ndmin=2).T)[_pdist_idxs] for ixyz in COMs_xyz[:,:,ii]])
            comp_len = _np.vstack([geom.unitcell_lengths[:,ii] for __ in range(comp_dist.shape[1])]).T
            bool_mask = (comp_dist > (geom.unitcell_lengths[:, ii, _np.newaxis] * .5))
            comp_dist[bool_mask] -= comp_len[bool_mask]
            if sum_over_comps2 is None:
                sum_over_comps2 = comp_dist**2
            else:
                sum_over_comps2 += comp_dist**2
        COM_dists_t = _np.sqrt(sum_over_comps2)

    if subtract_max_radii:
        if low_mem:
            res_max_radius = geom2max_residue_radius(geom, residue_idxs_unique, res_COMs=COMs_xyz).max(0)
            max_radius_pairs = _np.array([res_max_radius[ii] + res_max_radius[jj] for ii, jj in pair_map])
        else:
            res_max_radius = geom2max_residue_radius(geom, residue_idxs_unique, res_COMs=COMs_xyz)
            max_radius_pairs = _np.array([res_max_radius[:, ii] + res_max_radius[:, jj] for ii, jj in pair_map]).T

        # Low mem vs high mem. 6000 frames, 106499 atoms
        # peak memory: 22178.34 MiB, increment: 10344.77 MiB
        # peak memory: 27577.62 MiB, increment: 15739.57 MiB <-high mem
        #|    |   cutoff / nm |   #pairs lo-mem |   #pairs hi-mem |   t lo-mem / s |   t hi-mem / s |    #pairs lo-mem / |    #pairs hi-mem / |    #pairs lo-mem / |
        #|    |               |                 |                 |                |                |    % residue_pairs |    % residue_pairs |    % #pairs hi-mem |
        #|---:|--------------:|----------------:|----------------:|---------------:|---------------:|-------------------:|-------------------:|-------------------:|
        #|  1 |           0.1 |             751 |             573 |           4.27 |           3.41 |               0.71 |               0.54 |             131.06 |
        #|  2 |           0.2 |             963 |             770 |           5.21 |           4.31 |               0.9  |               0.72 |             125.06 |
        #|  3 |           0.3 |            1198 |             979 |           6.33 |           5.21 |               1.12 |               0.92 |             122.37 |
        #|  4 |           0.4 |            1479 |            1219 |           7.59 |           6.3  |               1.39 |               1.14 |             121.33 |
        #|  5 |           0.5 |            1780 |            1496 |           9.05 |           7.43 |               1.67 |               1.4  |             118.98 |
        #|  6 |           0.6 |            2117 |            1796 |          10.4  |           8.98 |               1.99 |               1.69 |             117.87 |
        COM_dists_t -= max_radius_pairs

    return COM_dists_t

def geom2COMxyz(igeom, residue_idxs=None):
    r"""
    Returns the time-trace of per-residue center-of-masses (COMs)

    Warning
    -------
    In cases where molecules are not "whole", some residues might be
    split across the periodic boundaries, i.e. while all atoms are
    within the periodic unit cell, a split-residue's fragments are scattered
    in different places inside the cell. Then, the COM might
    fall between these fragments, far away from any
    of the residues atoms. In that case, the COM is meaningless.
    Checking for split residues (any intra-residue atom-atom distance
    larger than half the box size would be an indication)
    would make the computation too slow.

    Parameters
    ----------
    igeom : :obj:`mdtraj.Trajectory`

    residue_idxs : iterable, default is None
        Residues for which the center of mass will be computed. Default
        is to compute all residues. The excluded residues will appear
        as np.nans in the returned value, which has the same shape
        regardless of the input

    Returns
    -------
    COMs : numpy.ndarray of shape (igeom.n_frames, igeom.n_residues,3)

    """

    if residue_idxs is None:
        residue_idxs=_np.arange(igeom.top.n_residues)
    masses = [_np.hstack([aa.element.mass for aa in rr.atoms]) for rr in
              igeom.top.residues]
    COMs_res_time_coords = _np.zeros((igeom.n_residues,igeom.n_frames,3))
    COMs_res_time_coords[:,:,:] = _np.nan
    COMs_res_time_coords[residue_idxs] = [_np.average(igeom.xyz[:, [aa.index for aa in igeom.top.residue(index).atoms], :], axis=1, weights=masses[index])
                            for index in residue_idxs]
    COMs_time_res_coords = _np.swapaxes(_np.array(COMs_res_time_coords),0,1)
    return COMs_time_res_coords

def geom2max_residue_radius(geom, residue_idxs=None, res_COMs=None) -> _np.ndarray:
    r"""
    Per-residue maximum radius, i.e. the maximum distance between any atom of the residue and the residue's center of mass

    Warning
    -------
    No periodic-boundary-conditions are taken into account, i.e.
    residues are assumed "whole" or "unwrapped", and then the mass-weighted
    average of a residues's atoms' cartesian coordinates are computed. If your residues
    are not "whole", i.e. atoms o the same residue, then the residue COM
    might be meaningless (check the warning in :obj:`~mdciao.utils.COM.geom2COMxyz`)
    and so will be the maximum residue radius.

    Parameters
    ----------
    geom : :obj:`mdtraj.Trajectory`
    residue_idxs : iterable of ints, default is None
        The indices of the residues for which
        the residue radius will be computed.
        If None, all residues will be considered.
        If `residue_idxs` is None, then `res_COM` has to
        be None as well, else you would be
        providing `res_COMs` any information
        on what residues they belong to.
    res_COMs : np.ndarray, default is None
        The time-traces of the xyz coordinates of
        the residue COMs. Has to have
        shape(geom.n_frames, len(residue_idxs), 3).
        It will be computed on the fly if None is provided.

    Returns
    -------
    r : numpy.ndarray
        Shape (igeom.n_frames, len(residue_idxs))

    """
    if residue_idxs is None:
        residue_idxs = _np.arange(geom.n_residues)
        if res_COMs is not None:
            raise ValueError("If 'residue_idxs' is None, then 'res_COMs' has to be None as well.")

    if res_COMs is None:
        res_COMs = geom2COMxyz(geom, residue_idxs=residue_idxs)[:, residue_idxs]
    else:
        assert res_COMs.shape[0] == geom.n_frames
        assert res_COMs.shape[1] == len(residue_idxs)
    atom_idxs = [[aa.index for aa in geom.top.residue(rr).atoms] for rr in residue_idxs]

    r = [_np.linalg.norm(geom.xyz[:, idxs, :] - res_COMs[:, _np.newaxis, ii], axis=-1).max(axis=1) for ii, idxs in
         enumerate(atom_idxs)]

    r = _np.vstack(r).T

    return r


def _per_residue_unwrapping(traj,
                            max_res_radii_t=None, progressbar=False, inplace=False,
                            unwrap_when_smaller_than=.4) -> _md.Trajectory:
    r"""

    Translate (if needed) atoms of a residue to a different periodic image to make the residue "whole"

    Sometimes this is referred to as "unwrapping".
    To decide whether a given residue needs unwrapping
    in a given frame, the method checks if any of the residue's
    atoms are further away from the residue COM than a half the
    box dimensions. In theory, one would check the distance
    per-dimension (distance in x > .5 * box_x,  in y > .5 box_y etc),
    as in the minimum image convention, also taking into account
    frame-dependent box dimensions. Practically, it's faster to
    compare euclidean distance in 3D to the COM (i.e. the maximum
    residue radius) against the smallest box length for all frames.
    This might "overshoot" and try to unwrap a residue that
    doesn't need unwrapping, with no other effect than
    leaving that residues intact.

    Ensuring unbroken (sometimes called "whole") residues,
    even if that means having atoms of the same residue spread
    across adjacent periodic images allows for the
    computation of per-residue center-of-mass (COM) that
    wont place COMs "in the middle the box" when a residue
    has been split across the PBCs and "wrapped" into the box.

    Note
    ----
    This method only makes **residues** whole, but not necessarily the
    entire molecule itself, i.e. it is not :obj:`mdtraj.Trajectory.image_molecule`.
    The molecule itself, i.e. the bonds between residues
    might still be split across PBCs after running this method.
    What this method guarantees is that residue COMs
    have meaning and are placed at the center of the group of atoms that form
    the "whole" residue. This is very useful as a preprocessing step to compute
    lower bounds for residue distances under PBCs, because residue positions
    and maximum residue radii are meaningful even if the COMs are spread across
    different periodic images.

    Parameters
    ----------
    traj : :obj:`~mdtraj.Trajectory`
    max_res_radii_t : 2D np.ndarray, default is None
        Optionally, pass here the value of
        the residue radii before any unwrapping takes place,
        else it will be computed on the fly.
        Shape is (traj.n_frames, traj.n_residues).
    progressbar : bool, default is False
        Show progressbar.
    inplace : bool, default is False
        Change the coordinates of `traj`
        inplace. Default (False) is to
        return a new object and leave the
        original `traj`  untouched.
    unwrap_when_smaller_than : float, default is .4
        Unwrap when the maximum residue radius
        is larger than the minimum box length times
        this factor. Flagging residues
        that don't need unwrapping for unwrapping
        doesn't have any effect

    Returns
    -------
    out : :obj:`~mdtraj aj.Trajectory`
        A new trajectory with unwrapped
        residues or, if `inplace` was True,
        an updated `traj` with the unwrapped residues.
    """

    if max_res_radii_t is None:
        max_res_radii_t = geom2max_residue_radius(traj)
    else:
        assert max_res_radii_t.shape ==(traj.n_frames, traj.n_residues)
    broken_res_bool = max_res_radii_t > (traj.unitcell_lengths.min() * unwrap_when_smaller_than)

    if inplace:
        outtraj = traj
    else:
        outtraj = _md.Trajectory(_np.copy(traj._xyz), traj.top, time=traj.time, unitcell_angles=traj.unitcell_angles, unitcell_lengths=traj.unitcell_lengths)

    if broken_res_bool.any():
        PB = _tqdm(total = broken_res_bool.sum(), disable=not progressbar)
        for res_idx, res_bool in enumerate(broken_res_bool.T):
            time_frames = _np.flatnonzero(res_bool)
            if len(time_frames) > 0:
                atoms = [aa.index for aa in traj.top.residue(res_idx).atoms]
                whole_res_xyz_t = _unwrap(traj.xyz[time_frames][:, atoms, :], traj.unitcell_lengths[time_frames, :])
                #print(res_idx, time_frames.shape)
                for ff, ixyz in zip(time_frames, whole_res_xyz_t):
                    outtraj._xyz[ff, atoms, :] = ixyz
                PB.update(len(time_frames))
    return outtraj

_translations = _np.vstack(_np.vstack([[[(kk, jj, ii) for ii in range(-1,2)] for jj in range(-1,2)] for kk in range(-1,2)]))

def _unwrap(xyz_t, unitcell_lengths):
    r"""

    For a trajectory of xyz coordinates, typically from atoms
    belonging to the same residue, apply, for each atom,
    the periodic translation that places that atom in the
    same periodic image as the rest of the atoms.

    In 1D, a residue with 5 atoms is split across the
    boundary:
        |cde-----------ab|
    This method picks the atom closest to the center of the box (e)
    and finds, for the other atoms, which one of the periodic
    translations, -1 * box_x, 0 * box_x, or +1 * box_x,
    places each atom a,b,c,d the closet to e, regardless
    of splitting the residues across the PBC
    In this case, For c,d, the translation is "no translation",
    i.e. 0 * box_x, whereas for a and b
    it will be -1 * box_x, yielding
      ab|cde-------------|
    Once the translation has been decided, it is applied
    on the coordinates and the coordinates returned.

    For each frame, the frame-dependent `unitcell_lengths`
    PBCs are taken into account.

    Parameters
    ----------
    xyz_t : 3D np.ndarray
        (n_frames, n_atoms, 3)
    unitcell_lengths : 2D np.ndarray
        (n_frames, 3)

    Returns
    -------
    res_xyz_t : 3D np.ndarray
        (n_frames, n_atoms, 3)
    """
    translations_vec = _translations * unitcell_lengths[:, _np.newaxis, :]
    d2center = _np.linalg.norm(xyz_t - unitcell_lengths[:, _np.newaxis, :] / 2, axis=2)
    most_centered_atom = d2center.argmin(axis=1)
    xyz_best_centered_atom = _np.vstack([xyz_t[ii][jj] for ii, jj in enumerate(most_centered_atom)])
    xyz_t -= xyz_best_centered_atom[:, _np.newaxis, :]
    # Doing these two operations with integers hardly speeds up the computation
    translated_coords = xyz_t[:, :, _np.newaxis, :] + translations_vec[:, _np.newaxis, :, :]
    d2tran = _np.linalg.norm(translated_coords, axis=-1)
    # Since we're only interested in the argmin we could've just computed
    # the sum of the abs(components), which respects the order of values,
    # but there's hardly any speedup
    best_translation_idx_per_frame_per_atom = d2tran.argmin(axis=-1)

    final_trans = []
    # This loop could/should be optimized to vector operations but currently
    # it's only %5 of compute time (cf the translated_coords + d2tran which is 70%)
    for ii, row in enumerate(best_translation_idx_per_frame_per_atom):
        final_trans.append(translations_vec[ii,row])
    final_trans = _np.array(final_trans)
    xyz_t += final_trans + xyz_best_centered_atom[:, _np.newaxis, :]
    return xyz_t


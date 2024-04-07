r"""
Functions related to Center-Of-Mass (COM) computations

.. autosummary::
   :nosignatures:
   :toctree: generated/


"""
import numpy as _np
import mdtraj as _md
from tqdm import tqdm as _tqdm
from mdtraj.geometry.distance import compute_distances_core as _compute_distances_core

def geom2COMdist(geom, residue_pairs, subtract_max_radii=False, low_mem=True,
                 periodic=True, per_residue_unwrap=True) -> _np.ndarray:
    r"""
    Returns the time-trace of the distances between pairs of residues' centers-of-mass (COM)

    The option `subtract_max_radii` can be used to produce a time-dependent lower bound
    on the distance between any atoms of each residue pair, i.e. a lower bound
    on the pairwise-residue distance. This lower bound can then be used to discard
    any contact between some pairs residues in any frame of `geom`,
    without having to compute all pairwise atom distances between the `residue_pairs`.

    Please see below the `periodic` and `per_residue_unwrap` for how periodic-boundary-conditions
    affect this calculation.

    Parameters
    ----------
    geom: :obj:`mdtraj.Trajectory`
    residue_pairs: iterable of pairs of integers
        zero-indexed pairs of residues
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
        For the purposes of cutoff-thresholding, this "overshooting"
        has little consequence, see the note below for a benchmark case.
    periodic : bool, default is True
        Compute COM distances under the minimum image convention.
        Please see the warning in :obj:`~mdciao.utils.COM.COMs_xyz`
        wrt residues being split across periodic boundaries into
        adjacent periodic images and the impact on COM computation.
    per_residue_unwrap : bool, default is True
        Unwrap residues individually for the purpose of
        computing the residue COM. This is to deal with
        the warning in :obj:`~mdciao.utils.COM.geom2COMxyz`.
        The original coordinates in `geom` remain unchanged.
        Since periodic boundary conditions are necessarily
        used when unwrapping, and the unwrapping doesn't make
        the entire `geom` whole, but only residues, the method
        will fail if `per_residue_unwrap` is True but `periodic` is False,
        alerting the user of what they are trying to do. See the
        note below for information on the overhead induced
        by the `per_residue_unwrap`.

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

    Note
    ----
        Just how much overhead the per-residue unwrapping adds is hard to know a-priori.
        In our benchmark system, it can be between 5% or 25% of the overall computation time.
        It simply depends on how many residues are split across PBCs in how many frames,
        which depends highly on what post-processing steps the trajectory has undergone.
        * The 5% overhead is when no residues have to be unwrapped because the pre-processing
         has taken care of this, e.g. with `gmx trjconv -pbc whole`. The 5% is just the
         time used to check whether the residues need to be unwrapped or not.
        * The 25% overhead is for a plausible worst-case, when the split across PBCs
         affects a high number of residues. The system is effectively halved at
         the plane where most residues are affected simultaneously by the split.
        * For the highly-constructed, and physically inplausible
         situation where **all** are residues need to be unwrapped, the overhead is 70%.
         This is very unlikely, because it implies that **all** residues have some atoms
         just a few Angstrom away from the boundary

        In essence, you can always leave `per_residue_unwrap=True` on unless significant slowdown
        is noticed. Even better, if you notice a significant slowdown, why not pre-process
        your trajectory to be whole and centered in the box (which is sometimes called
        unwrapping), and then there's no need to use `per_residue_unwrap`. Note, even in that
        case a decision needs to be made whether to use PBCs when computing distances
        between residue COMs


    Returns
    -------
    COMs_array : np.ndarray of shape(geom.n_frames, len(res_pairs))
        contains the time-traces of the distance between residue COMs,
        optionally minus the sum of each pair of residue radii.
        Please note that this option can produce negative values,
        meaning the residue-spheres actually intersect at some point.
    """

    residue_pairs = _np.array(residue_pairs)
    residue_idxs_unique, pair_map = _np.unique(residue_pairs, return_inverse=True)
    pair_map = pair_map.reshape(len(residue_pairs),2)

    if per_residue_unwrap:
        assert periodic, ValueError("Cannot unwrap residues if 'periodic' is set to False.")
        # Per-residue per-frame unwraping
        unwrapped_residue_geom = _per_residue_unwrapping(geom,residue_idxs=residue_idxs_unique)

    else:
        unwrapped_residue_geom = geom
    # This would be worth migrating to mdanalysis
    # https://docs.mdanalysis.org/1.0.1/documentation_pages/core/groups.html#MDAnalysis.core.groups.ResidueGroup.center
    COMs_xyz = geom2COMxyz(unwrapped_residue_geom, residue_idxs=residue_idxs_unique)

    COM_dists_t = _compute_distances_core(COMs_xyz,
                                          pair_map,
                                          unitcell_vectors=geom.unitcell_vectors,
                                          periodic=periodic,
                                          )

    if subtract_max_radii:
        if low_mem:
            res_max_radius = geom2max_residue_radius(unwrapped_residue_geom, residue_idxs_unique, res_COMs=COMs_xyz).max(0)
            max_radius_pairs = res_max_radius[pair_map].sum(1)
        else:
            res_max_radius = geom2max_residue_radius(unwrapped_residue_geom, residue_idxs_unique, res_COMs=COMs_xyz)
            max_radius_pairs = res_max_radius[:, pair_map].sum(axis=-1)

        # TODO update memory numbers
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
    In some cases, molecules may have residues split across the periodic boundaries,
    i.e. all atoms are within the periodic unit cell, but atoms of the same
    residue are scattered in different places inside the cell, near the walls.
    Then, the residue COM might fall between these fragments, far away from any
    of the residues' atoms. In that case, the residue COM is not a useful measure for
    the residue's approximate position.

    Parameters
    ----------
    igeom : :obj:`mdtraj.Trajectory`

    residue_idxs : iterable, default is None
        Residues for which the center of mass will be computed. Default
        is to compute all residues.

    Returns
    -------
    COMs : numpy.ndarray of shape (igeom.n_frames, len(residue_idxs),3)

    """

    if residue_idxs is None:
        residue_idxs=_np.arange(igeom.top.n_residues)
    masses = [_np.hstack([aa.element.mass for aa in rr.atoms]) for rr in
              igeom.top.residues]
    COMs_res_time_coords = _np.array([_np.average(igeom.xyz[:, [aa.index for aa in igeom.top.residue(index).atoms], :],
                                                  axis=1, weights=masses[index])
                                      for index in residue_idxs])
    COMs_time_res_coords = _np.swapaxes(_np.array(COMs_res_time_coords), 0, 1)
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
        res_COMs = geom2COMxyz(geom, residue_idxs=residue_idxs)
    else:
        assert res_COMs.shape[0] == geom.n_frames
        assert res_COMs.shape[1] == len(residue_idxs)
    atom_idxs = [[aa.index for aa in geom.top.residue(rr).atoms] for rr in residue_idxs]

    r = [_np.linalg.norm(geom.xyz[:, idxs, :] - res_COMs[:, _np.newaxis, ii], axis=-1).max(axis=1) for ii, idxs in
         enumerate(atom_idxs)]

    r = _np.vstack(r).T

    return r


def _per_residue_unwrapping(traj,
                            residue_idxs=None,
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
    residue_idxs : iterable of ints, default is None
        Apply the method only to these residues. Default
        is to apply it to all.
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
    if residue_idxs is None:
        residue_idxs = _np.arange(traj.n_residues)
    if max_res_radii_t is None:
        max_res_radii_t = geom2max_residue_radius(traj, residue_idxs=residue_idxs)
    else:
        assert max_res_radii_t.shape ==(traj.n_frames, len(residue_idxs))
    broken_res_bool = max_res_radii_t > (traj.unitcell_lengths.min() * unwrap_when_smaller_than)
    assert broken_res_bool.shape == (traj.n_frames, len(residue_idxs)) #making extra sure nothing goes wrong here
    if inplace:
        outtraj = traj
    else:
        outtraj = _md.Trajectory(_np.copy(traj._xyz), traj.top, time=traj.time, unitcell_angles=traj.unitcell_angles, unitcell_lengths=traj.unitcell_lengths)

    if broken_res_bool.any():
        PB = _tqdm(total = broken_res_bool.sum(), disable=not progressbar)
        for res_idx, res_bool in zip(residue_idxs, broken_res_bool.T):
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


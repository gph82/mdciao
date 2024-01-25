r"""
Functions related to Center-Of-Mass (COM) computations

.. autosummary::
   :nosignatures:
   :toctree: generated/


"""
import numpy as _np
from scipy.spatial.distance import pdist as _pdist


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
    Per-residue radius, i.e. the maximum distance between any atom of the residue and the residue's center of mass

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
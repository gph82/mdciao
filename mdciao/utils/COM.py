r"""
Functions related to Center-Of-Mass (COM) computations

.. autosummary::
   :nosignatures:
   :toctree: generated/


"""
import numpy as _np
from scipy.spatial.distance import pdist as _pdist


def geom2COMdist(geom, residue_pairs, subtract_max_radii=False, low_mem=True):
    r"""
    Returns the distances between pairs of residues' center-of-mass (COM)

    The option `subtract_max_radii` can be used to produce a time-dependent lower bound
    on the distance between any atoms of each residue pair, i.e. a lower bound
    on the pairwise-residue "mindist". This lower bound can be used by to discard
    any contact between some pairs residues in any frame of `geom`
    without having to compute all pairwise atom distances between the `residue_pairs`.

    Parameters
    ----------
    geom: :obj:`mdtraj.Trajectory`
    residue_pairs: iterable of integer pairs
        pairs of residues by their zero-indexed serial indexes
    subtract_max_radii : bool, default is False
        If True, the maximum radius of each residue
        in residue pairs (as computed by
        :obj:`~mdciao.utils.COM.geom2max_residue_radius`.
        Please note that this option can produce negative values,
        meaning the residue-spheres actually intersect at some point.
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
        In a benchmark case of 6K frames and ~100K `residues_pairs`,
        with `subtract_max_radii=True`, we kept only those `residue_pairs`
        with a lower bound smaller than a a given cutoff.
        As expected, thresholding the residue-mindist values
        obtained with `low_mem=True` resulted in longer
        lists of potential contacts (between 20%-30% longer depending
        on the cutoff), and accordingly longer computation times when calling
        :obj:`mdtraj.compute_contacts` on the kept `residue_pairs`. However,
        those "longer" lists (and times) are not significant at all when expressed as a
        percentage of the initial `residue_pairs`:

        * Using a cutoff of 1 Angstrom when thresholding the mindist
          values obtained with `low_mem=False` discarded 99.5% of the original `residue_pairs`,
          whereas using the `low_mem=True` values meant discarding "only" 99.3%.
        * Using a cutoff of 6 Angstrom when thresholding the mindist
          values obtained with `low_mem=False` discarded 98.3% of the original `residue_pairs`,
          whereas using the `low_mem=True` values meant discarding "only" 98%.

        So there is still a huge reduction in the number of potential contact-pairs
        when thresholding, from ca 100K residue pairs to to between 500 and 2K.
        For this benchmark system, `low_mem=True` used
        ~20GBs of memory and `low_mem=False` used ~25GB. There's a table included
        as a comment in the source code of the method showing the benchmark numbers.

    Returns
    -------
    COMs_array : np.ndarray of shape(geom.n_frames, len(res_pairs))
        contains the time-traces of the distance between residue COMs,
        optionally minus the sum of each pair of residue radii.
        Please note that this option can produce negative values,
        meaning the residue-spheres actually intersect at some point.
    """


    residue_idxs_unique, pair_map = _np.unique(residue_pairs, return_inverse=True)
    n_unique_residues = len(residue_idxs_unique)
    pair_map = pair_map.reshape(len(residue_pairs),2)

    COMs_xyz = geom2COMxyz(geom, residue_idxs=residue_idxs_unique)[:, residue_idxs_unique]

    # Only do pdist of the needed residues
    COMs_dist_triu = _np.array([_pdist(ixyz) for ixyz in COMs_xyz])

    # From the _pdist doc
    # The metric dist(u=X[i], v=X[j]) is computed and stored in entry m * i + j - ((i + 2) * (i + 1)) // 2.
    _pdist_ravel = lambda i, j : n_unique_residues * i + j - ((i + 2) * (i + 1)) // 2
    _pdist_idxs = _pdist_ravel(pair_map[:,0], pair_map[:,1])

    # Grab only the _pdist_idxs
    COM_dists_t = COMs_dist_triu[:,_pdist_idxs]

    if subtract_max_radii:
        if low_mem:
            res_max_radius = geom2max_residue_radius(geom, residue_idxs_unique, res_COMs=COMs_xyz).max(0)
            max_radius_pairs = _np.array([res_max_radius[ii] + res_max_radius[jj] for ii, jj in pair_map])
        else:
            res_max_radius = geom2max_residue_radius(geom, residue_idxs_unique, res_COMs=COMs_xyz)
            max_radius_pairs = _np.array([res_max_radius[:, ii] + res_max_radius[:, jj] for ii, jj in pair_map]).T

        # Low mem vs high mem. 6000 frames, 106499 atoms
        # peak memory: 22681.03 MiB, increment: 20894.39 MiB <-low mem
        # peak memory: 32551.02 MiB, increment: 25883.46 MiB <-high mem
        #|   cutoff |  #pairs lo-mem |  #pairs hi-mem |   t low-mem / s |   t hi-mem / s |   #pairs lo-mem as   |   #pairs hi-mem as   |   #pairs lo-mem as |
        #|    nm    |                |                |                 |                |   % residue_pairs    |   % residue_pairs    |    % #pairs hi-mem |
        #|---------:|---------------:|---------------:|----------------:|---------------:|---------------------:|---------------------:|------------------:|
        #|      0.1 |            751 |            573 |            4.27 |           3.4  |                 0.71 |                 0.54 |             31.06 |
        #|      0.2 |            963 |            770 |            5.32 |           4.22 |                 0.9  |                 0.72 |             25.06 |
        #|      0.3 |           1198 |            979 |            6.24 |           5.18 |                 1.12 |                 0.92 |             22.37 |
        #|      0.4 |           1479 |           1219 |            7.59 |           6.45 |                 1.39 |                 1.14 |             21.33 |
        #|      0.5 |           1780 |           1496 |            9.41 |           7.54 |                 1.67 |                 1.4  |             18.98 |
        #|      0.6 |           2117 |           1796 |           10.48 |           8.94 |                 1.99 |                 1.69 |             17.87 |
        COM_dists_t -= max_radius_pairs

    return COM_dists_t

def geom2COMxyz(igeom, residue_idxs=None):
    r"""
    Returns the time trace of per-residue
    center-of-masses (COMs) in cartesian coordinates

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

    #TODO check the behaviour of COMs always having the same shape
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
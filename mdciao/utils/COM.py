r"""
Functions related to Center-Of-Mass (COM) computations

.. autosummary::
   :nosignatures:
   :toctree: generated/


"""
import numpy as _np
def geom2COMdist(igeom, residue_pairs):
    r"""
    Returns the distances between center-of-mass (COM)
    Parameters
    ----------
    igeom: :obj:`mdtraj.Trajectory`
    residue_pairs: iterable of integer pairs
        pairs of residues by their zero-indexed serial indexes

    Returns
    -------
    COMs_array : np.ndarray of shape(igeom.n_frames, len(res_pairs))
        contains the time-traces of the residue COMs
    """
    from scipy.spatial.distance import pdist, squareform
    residue_idxs_unique, pair_map = _np.unique(residue_pairs, return_inverse=True)
    pair_map = pair_map.reshape(len(residue_pairs),2)


    COMs_xyz = geom2COMxyz(igeom, residue_idxs=residue_idxs_unique)[:,residue_idxs_unique]

    # Only do pdist of the needed residues
    COMs_dist_triu = _np.array([pdist(ixyz) for ixyz in COMs_xyz])

    # Get square matrices
    COMs_square = _np.array([squareform(trupper) for trupper in COMs_dist_triu])

    # Use the pair map to get the right indices
    COMs_array = _np.array([COMs_square[:,ii,jj] for ii,jj in pair_map]).T

    return COMs_array

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
r"""
Simple operations with bonds between the
residues of an :obj:`mdtraj.Topology`

.. autosummary::
   :nosignatures:
   :toctree: generated/


"""
import numpy as _np
from scipy.sparse.csgraph import connected_components as _concom

def connected_sets(mat):
    r"""
    Return the connected components/sets of a an adjacency matrix

    Uses :obj:`~scipy.sparse.csgraph.connected_components`
    under the hood with directed=False

    Parameters
    ----------
    mat : 2D _np.array, square matrix (M,M)
        Adjacency matrix, can be symmetric or not.
        Nodes are always self-adjacent, i.e.
        the diagonal of :obj:`mat` is ignored

    Returns
    -------
    sets: list
        The connected components as 1D _np.ndarrays
    """
    sets = []
    nsets, labels = _concom(mat, directed=False)
    for ii in range(nsets):
        sets.append(_np.flatnonzero(labels==ii))
    return sets

def top2residue_bond_matrix(top,
                            force_resSeq_breaks=False,
                            verbose=True,
                            create_standard_bonds=False):
    r"""Return a symmetric residue-residue bond matrix from a :obj:`~mdtraj.Topology`.

    The bonds used are those found in :obj:`~mdtraj.Topology.bonds`

    Parameters
    ----------
    top : :obj:`mdtraj.Topology`
    force_resSeq_breaks : boolean, default is False
        Delete bonds if there is a resSeq jump between residues.
    verbose : boolean, default is True
        Print a statement if residue index has no bonds
    create_standard_bonds : boolean, default is False
        Advanced users only, can easily lead to wrong
        results in case of .gro files, because
        :obj:`~mdtraj.Topology.create_standard_bonds`
        needs chain information to avoid creating
        bonds between residues that follow one another
    Returns
    -------
    residue_bond_matrix : 2D np.ndarray
        Returns a symmetric adjacency matrix with entries ij=1 and ji=1,
        if there is a bond between residue i and residue j.
    """

    if len(top._bonds) == 0:
        if create_standard_bonds:
            top.create_standard_bonds()
        else:
            raise ValueError("\nThe parsed topology does not contain any bonds in the top._bonds attribute.\n"
                             " * If you are using low-level methods, like mdciao.utils.bonds.top2residue_bond_matrix,\n"
                             "   try using 'create_standard_bonds=True'. This will 'create bonds based on the atom\n"
                             "   and residue names for all standard residue types'.\n"
                             " * If you are using CLI methods like mdciao.cli.residue_neighborhoods or mdc_neighborhoods.py,\n"
                             "   try using 'naive_bonds=True'. This will create bonds between adjacent residues regardless of\n"
                             "   atom-types.\n"
                             "Please read the docs of the different options for potential pitfalls and risks!")

    residue_bond_matrix = _np.zeros((top.n_residues, top.n_residues), dtype=int)
    for ibond in top._bonds:
        r1, r2 = ibond.atom1.residue.index, ibond.atom2.residue.index
        rSeq1, rSeq2 = ibond.atom1.residue.resSeq, ibond.atom2.residue.resSeq
        residue_bond_matrix[r1, r2] = 1
        residue_bond_matrix[r2, r1] = 1
        if force_resSeq_breaks and _np.abs(rSeq1 - rSeq2) > 1:  # mdtrajs bond-making routine does not check for resSeq
            residue_bond_matrix[r1, r2] = 0
            residue_bond_matrix[r2, r1] = 0
    for ii, row in enumerate(residue_bond_matrix):
        if row.sum()==0 and verbose:
            print("Residue with index %u (%s) has no bonds to other residues"%(ii,top.residue(ii)))

    return residue_bond_matrix

#TODO do we need this anymore?
def top2residuebonds(top,**top2residue_bond_matrix_kwargs):
    return _residue_bond_matrix_to_triu_bonds(top2residue_bond_matrix(top, **top2residue_bond_matrix_kwargs))

#TODO do we need this anymore?
def _residue_bond_matrix_to_triu_bonds(residue_bond_matrix):
    _np.testing.assert_array_equal(residue_bond_matrix, residue_bond_matrix.T), \
        ("This is not a symmetric residue bond matrix\n", residue_bond_matrix)
    bonds = []
    for ii, jj in _np.vstack(_np.triu_indices_from(residue_bond_matrix, k=1)).T:
        if residue_bond_matrix[ii, jj] == 1:
            bonds.append([ii, jj])
    return bonds

def bonded_neighborlist_from_top(top, n=1, verbose=False):
    """
    Bonded neighbors of all the residues in the topology file.

    Parameters
    ----------
    top : :obj:`~mdtraj.Topology`
    n : int, default is 1
        Number of bonded neighbors considered, i.e.
        A-B-C has only [B] as neighbors if :obj:`n` = 1,
        but [B,C] if :obj:`n` = 2

    Returns
    -------
    neighbor_list : list of lists
        Lisf of len top.n_residues. The i-th  list
        contains the :obj:`n` -bonded neighbors of
        the i-th residue, in ascending order
    """

    residue_bond_matrix = top2residue_bond_matrix(top,verbose=verbose)
    return neighborlists_from_adjacency_matrix(residue_bond_matrix, n)

#todo this is very slow and a bit of overkill if one is only interested in the
#neighbors of one particular residue
# Do it graph-thetically, ideally
def neighborlists_from_adjacency_matrix(mat, n):
    r"""
    Return neighborlists from an adjacency matrix.

    The diagonal of :obj:`mat` is ignored, i.e. it can be 0 or 1

    Parameters
    ----------
    mat : 2D _np.array, square matrix (M,M)
        Adjacency matrix, can be symmetric or not
    n : int
        Connectedness.

    Returns
    -------
    neighbors : list
        A list of len M where the i-th entry
        contains the indices of the nodes
        separated from the i-ith node by
        a maximum of :obj:`n` jumps. The indices
        are always in ascending order
    """
    neighbor_list = [[ii] for ii in range(mat.shape[0])]
    for kk in range(n):
        for ridx, ilist in enumerate(neighbor_list):
            new_neighborlist = [ii for ii in ilist]
            for rn in ilist:
                row = mat[rn]
                bonded = _np.flatnonzero(row)
                toadd = [nn for nn in bonded if nn not in ilist and nn != ridx]
                if len(toadd):
                    # print("neighbor %u adds new neighbor %s:"%(rn, toadd))
                    new_neighborlist += toadd
                    # print("so that the new neighborlist is: %s"%new_neighborlist)

            neighbor_list[ridx] = [ii for ii in _np.unique(new_neighborlist) if ii != ridx]
            # break

    # Check that the neighborlist works both ways
    for ii, ilist in enumerate(neighbor_list):
        for nn in ilist:
            assert ii in neighbor_list[nn]

    return neighbor_list

def top2residue_bond_matrix_naive(top, only_protein=True, fragments=None):
    r""" Creates a naive (=linear) residue-residue bond-matrix,
    where a bond is assumed between the n-th and the n+1-th residue

    Usually, one would resort to this method if the topology's
    own create_standard_bonds() method does not work, e.g.
    because there's only alpha Carbons

    Parameters
    ----------
    top : :obj:`~mdtraj.Topology`
    only_protein : bool, default is True
        Only create bonds when both residues
        satisfy are protein residues. The
        check is done using the attribute
        :obj:`mdtraj.core.topology.Residue.is_protein`
    fragments : iterable of ints, default is None
        Use fragment/chain definition to avoid
        bonds between fragments/chains. These
        definitions need to cover the entire :obj:`~mdtraj.Topology`,
        i.e. each residue index must appear once (an and only once)
        in the :obj`fragments`. Note that breaks
        introduced via the :obj:`only_protein argument`
        will always be present, whether these residues
        are in the same fragment or not

    Returns
    -------
    mat : 2D np.ndarray
        Symmetric residue-residue bond-matrix.
        The diagonal is filled with ones as well.
    """
    bonds = _np.zeros((top.n_residues, top.n_residues), dtype=int)

    if fragments is None:
        fragments = [_np.arange(top.n_residues)]

    #TODO this fragment checking might defined elsewhere as a function
    # but I don't want to introduce any other dep here
    stacked = _np.concatenate(fragments)
    unique = _np.unique(stacked)
    missing = [ii for ii in range(top.n_residues) if ii not in stacked]
    if len(missing)>0:
        raise ValueError("Bad fragment definition. Residue idxs missing: %s"%missing)
    if len(unique)<len(stacked):
        raise ValueError("Bad fragment definition. Residue idxs repeated: %s"%_np.flatnonzero(_np.bincount(stacked)!=1))

    if only_protein is True:
        protein = [rr.index for rr in top.residues if rr.is_protein]
    else:
        protein = [rr.index for rr in top.residues]

    for ifrag in fragments:
        prot_frag = _np.intersect1d(ifrag, protein)
        for ii in prot_frag:
            if ii+1 in prot_frag:
                bonds[ii,ii+1] = 1
                bonds[ii+1,ii] = 1

    _np.fill_diagonal(bonds,1)
    return bonds

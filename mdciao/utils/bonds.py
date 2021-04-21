r"""
Simple operations with bonds between the
residues of an :obj:`mdtraj.Topology`

.. autosummary::
   :nosignatures:
   :toctree: generated/


"""
import numpy as _np

# This is lifted from mdas, the original source shall remain there
def top2residue_bond_matrix(top,
                            force_resSeq_breaks=False,
                            verbose=True,
                            create_standard_bonds=False):
    '''
    Returns a residue-residue bond matrix from the topology file.
    The entries in the bond matrix will have either 1 or 0, where 1 signifies a bond is present.

    Parameters
    ----------
    top : :py:class:`mdtraj.Topology`
    force_resSeq_breaks : boolean, default is False
        Delete bonds if there is a resSeq jump between residues.
    verbose : boolean, default is True
        Print a statement if residue index has no bonds
    create_standard_bonds : boolean, default is False
        Advanced users only, can easily lead to wrong
        results in case of .gro files, because
        :obj:`mdtraj.Topology.create_standard_bonds`
        needs chain information to avoid creating
        bonds between residues that follow one another


        
    Returns
    -------
    numpy matrix
        Returns a symmetric adjacency matrix with entries ij=1 and ji=1,
        if there is a bond between residue i and residue j.

    '''

    if len(top._bonds) == 0:
        if create_standard_bonds:
            top.create_standard_bonds()
        else:
            raise ValueError("\nThe parsed topology does not contain bonds!\n"
                             "If your input is a .gro file, you are advised\n"
                             "to generate a .pdb with chain information \n"
                             "file before continuing.")

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

def top2residuebonds(top,**top2residue_bond_matrix_kwargs):
    return _residue_bond_matrix_to_triu_bonds(top2residue_bond_matrix(top, **top2residue_bond_matrix_kwargs))


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
        the i-th residue.
    """

    #todo this is very slow and a bit of overkill if one is only interested in the
    #neighbors of one particular residue
    residue_bond_matrix = top2residue_bond_matrix(top,verbose=verbose)
    neighbor_list = [[ii] for ii in range(residue_bond_matrix.shape[0])]
    for kk in range(n):
        for ridx, ilist in enumerate(neighbor_list):
            new_neighborlist = [ii for ii in ilist]
            for rn in ilist:
                row = residue_bond_matrix[rn]
                bonded = _np.flatnonzero(row)
                toadd = [nn for nn in bonded if nn not in ilist and nn!=ridx]
                if len(toadd):
                    #print("neighbor %u adds new neighbor %s:"%(rn, toadd))
                    new_neighborlist += toadd
                    #print("so that the new neighborlist is: %s"%new_neighborlist)

            neighbor_list[ridx] = [ii for ii in _np.unique(new_neighborlist) if ii!=ridx]
            #break

    # Check that the neighborlist works both ways
    for ii, ilist in enumerate(neighbor_list):
        for nn in ilist:
            assert ii in neighbor_list[nn]

    return neighbor_list

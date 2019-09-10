import numpy as _np

# This is lifted from mdas, the original source shall remain there
def top2residue_bond_matrix(top, create_standard_bonds=True,
                            force_resSeq_breaks=False,
                            verbose=True):
    r"""
    :param top: md.Topology object
    :param create_standard_bonds: boolean. Force the method to create bonds if there are not upon reading (e.g.
    because the topology comes from a .gro-file instead of a .pdb-file.

    :return: symmetric adjacency matrix with entries ij=1 and ji=1 if there is a bond between atom i and atom j
    """
    if len(top._bonds) == 0:
        if create_standard_bonds:
            top.create_standard_bonds()
        else:
            raise ValueError("The parsed topology does not contain bonds! Aborting...")
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
            print("Residue with index %u (%s) has no bonds whatsoever"%(ii,top.residue(ii)))

    return residue_bond_matrix

def bonded_neighborlist_from_top(top, n=1):
    """TODO: description of method in one line

    :param top: mdTraj Topology object
    :param n: number of bonds between bonded neighbors
    :return: neighbor of each residue as a list of list
            Each residue will have a corresponding neighbor list(if neighbors exists), or an empty list(if no neighbor exists)
            A neighbor exists between i and j residues if residue_bond_matrix has a 1 at position ij
    """
    residue_bond_matrix = top2residue_bond_matrix(top)
    neighbor_list = [[ii] for ii in range(residue_bond_matrix.shape[0])]
    for kk in range(n):
        for ridx, ilist in enumerate(neighbor_list):
            new_neighborlist = [ii for ii in ilist]
            #print("Iteration %u in residue %u"%(kk, ridx))
            for rn in ilist:
                row = residue_bond_matrix[rn]
                bonded = _np.argwhere(row == 1).squeeze()
                if _np.ndim(bonded)==0:
                    bonded=[bonded]
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

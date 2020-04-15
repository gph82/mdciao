import mdtraj as md
import unittest
import numpy as _np
from filenames import filenames
from mdciao.bond_utils import top2residue_bond_matrix, bonded_neighborlist_from_top
import pytest

test_filenames = filenames()

class Test_top2residue_bond_matrix(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)
        self.geom_force_resSeq_breaks = md.load(test_filenames.file_for_test_force_resSeq_breaks_is_true_pdb)
        self.geom_no_bonds = md.load(test_filenames.file_for_no_bonds_pdb)
        self.geom_no_bonds.top._bonds=[]

    def test_bond_matrix(self):
        res_bond_matrix = _np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                                     [1, 1, 1, 0, 0, 0, 0, 0],
                                     [0, 1, 1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 1, 0, 0, 0],
                                     [0, 0, 0, 1, 1, 1, 0, 0],
                                     [0, 0, 0, 0, 1, 1, 0, 0], #LYS28 has a bond with GLU27
                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0]])
        assert (top2residue_bond_matrix(self.geom.top) == res_bond_matrix).all()

    def test_force_resSeq_breaks_is_true(self):
        res_bond_matrix = _np.array([ [1, 1, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 1, 0, 0, 0, 0, 0],
                                      [0, 1, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0], #LYS28 has been changed to LYS99 in the test file, so no bond with GLU27
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]])

        assert (top2residue_bond_matrix(self.geom_force_resSeq_breaks.top, force_resSeq_breaks=True) == res_bond_matrix).all()

    def test_no_bonds_fails(self):
        with pytest.raises(ValueError):
            top2residue_bond_matrix(self.geom_no_bonds.top, create_standard_bonds=False)

    def test_no_bonds_creates(self):
        mat = top2residue_bond_matrix(self.geom_no_bonds.top)
        bonds = [bond for bond in _np.argwhere(mat!=0).squeeze() if bond[0]!=bond[1]]
        assert _np.allclose(bonds[0], [1,2])
        assert _np.allclose(bonds[1], [2,1])


class Test_bonded_neighborlist_from_top(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)

    def test_neighbors(self):
       neighbors_from_function =  bonded_neighborlist_from_top(self.geom.top)
       actual_neighbors = [[1], [0, 2], [1], [4], [3, 5], [4], [], []]
       assert neighbors_from_function == actual_neighbors

if __name__ == '__main__':
   unittest.main()
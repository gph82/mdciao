import mdtraj as md
import unittest
import numpy as _np
from mdciao.filenames import filenames
from mdciao import bond_utils
import pytest

test_filenames = filenames()

class Test_top2residue_bond_matrix(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.small_monomer)
        self.geom_force_resSeq_breaks = md.load(test_filenames.small_monomer_LYS99)
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
        assert (bond_utils.top2residue_bond_matrix(self.geom.top) == res_bond_matrix).all()

    def test_force_resSeq_breaks_is_true(self):
        res_bond_matrix = _np.array([ [1, 1, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 1, 0, 0, 0, 0, 0],
                                      [0, 1, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0], #LYS28 has been changed to LYS99 in the test file, so no bond with GLU27
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]])

        assert (bond_utils.top2residue_bond_matrix(self.geom_force_resSeq_breaks.top, force_resSeq_breaks=True) == res_bond_matrix).all()

    def test_no_bonds_fails(self):
        with pytest.raises(ValueError):
            bond_utils.top2residue_bond_matrix(self.geom_no_bonds.top)

    def test_no_bonds_creates(self):
        mat = bond_utils.top2residue_bond_matrix(self.geom_no_bonds.top,
                                                 create_standard_bonds=True)
        #because the .gro filed does not contain chains, this will have
        #created all bonds except the non AAs, i.e. 6 and 7
        _np.testing.assert_array_equal(_np.argwhere(mat != 0),
                                       [[0, 0],
                                        [0, 1], [1, 0],
                                        [1, 1],
                                        [1, 2], [2, 1],
                                        [2, 2],
                                        [2, 3], [3, 2],
                                        [3, 3],
                                        [3, 4], [4, 3],
                                        [4, 4],
                                        [4, 5], [5, 4],
                                        [5, 5]])

class Test_top2residuebonds(unittest.TestCase):

    def test_works(self):
        _np.testing.assert_array_equal([[0,1],[1,2],[3,4],[4,5]],
                                       bond_utils.top2residuebonds(md.load(test_filenames.small_monomer).top))

class Test_residue_bond_matrix_to_triu_bonds(unittest.TestCase):

    def test_works(self):
        mat = _np.array([[1,1,0,0],
                         [1,1,0,0],
                         [0,0,1,1],
                         [0,0,1,1]])
        _np.testing.assert_array_equal([[0,1],
                                        [2,3]],
                                       bond_utils._residue_bond_matrix_to_triu_bonds(mat))

class Test_bonded_neighborlist_from_top(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.small_monomer)

    def test_neighbors(self):
       neighbors_from_function =  bond_utils.bonded_neighborlist_from_top(self.geom.top)
       actual_neighbors = [[1], [0, 2], [1], [4], [3, 5], [4], [], []]
       assert neighbors_from_function == actual_neighbors

if __name__ == '__main__':
   unittest.main()
import mdtraj as md
import unittest
import numpy as _np
from filenames import filenames
from sofi_functions.bond_utils import top2residue_bond_matrix, bonded_neighborlist_from_top

test_filenames = filenames()

class Test_top2residue_bond_matrix(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)
        self.geom_force_resSeq_breaks = md.load(test_filenames.file_for_test_force_resSeq_breaks_is_true_pdb)

    def test_it_just_works_with_top2residue_bond_matrix(self):
        res_bond_matrix = _np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 0, 0], #LYS28 has a bond with GLU27
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0]])
        assert (top2residue_bond_matrix(self.geom.top) == res_bond_matrix).all()

    def test_works_with_force_resSeq_breaks_is_true(self):
        res_bond_matrix = _np.array([ [1, 1, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 1, 0, 0, 0, 0, 0],
                                      [0, 1, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0], #LYS28 has been changed to LYS99 in the test file, so no bond with GLU27
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]])

        assert (top2residue_bond_matrix(self.geom_force_resSeq_breaks.top, force_resSeq_breaks=True) == res_bond_matrix).all()

class Test_bonded_neighborlist_from_top(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)

    def test_bonded_neighborlist_from_top_just_works(self):
       neighbors_from_function =  bonded_neighborlist_from_top(self.geom.top)
       actual_neighbors = [[1], [0, 2], [1], [4], [3, 5], [4], [], []]
       assert neighbors_from_function == actual_neighbors

if __name__ == '__main__':
   unittest.main()
import mdtraj as md
import unittest
import numpy as _np
from mdciao.examples import filenames as test_filenames
from mdciao.utils import bonds

class Test_top2residue_bond_matrix(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.small_monomer)
        self.geom_force_resSeq_breaks = md.load(test_filenames.small_monomer_LYS99)
        self.geom_no_bonds = md.load(test_filenames.file_for_no_bonds_gro)
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
        assert (bonds.top2residue_bond_matrix(self.geom.top) == res_bond_matrix).all()

    def test_force_resSeq_breaks_is_true(self):
        res_bond_matrix = _np.array([ [1, 1, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 1, 0, 0, 0, 0, 0],
                                      [0, 1, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0], #LYS28 has been changed to LYS99 in the test file, so no bond with GLU27
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]])

        assert (bonds.top2residue_bond_matrix(self.geom_force_resSeq_breaks.top, force_resSeq_breaks=True) == res_bond_matrix).all()

    def test_no_bonds_fails(self):
        with self.assertRaises(ValueError):
            bonds.top2residue_bond_matrix(self.geom_no_bonds.top)

    def test_no_bonds_creates(self):
        mat = bonds.top2residue_bond_matrix(self.geom_no_bonds.top,
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
                                       bonds.top2residuebonds(md.load(test_filenames.small_monomer).top))

class Test_residue_bond_matrix_to_triu_bonds(unittest.TestCase):

    def test_works(self):
        mat = _np.array([[1,1,0,0],
                         [1,1,0,0],
                         [0,0,1,1],
                         [0,0,1,1]])
        _np.testing.assert_array_equal([[0,1],
                                        [2,3]],
                                       bonds._residue_bond_matrix_to_triu_bonds(mat))

class Test_bonded_neighborlist_from_top(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.small_monomer)

    def test_neighbors(self):
       neighbors_from_function =  bonds.bonded_neighborlist_from_top(self.geom.top)
       actual_neighbors = [[1], [0, 2], [1], [4], [3, 5], [4], [], []]
       assert neighbors_from_function == actual_neighbors

class Test_neighborlists_from_adjacency_matrix(unittest.TestCase):

    def test_works(self):
        mat = _np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1]])

        nl_1 = [
            [1],        # 0
            [0, 2],     # 1
            [1],        # 2
            [4],        # 3
            [3, 5],     # 4
            [4, 6],     # 5
            [5],        # 6
            []          # 7
        ]

        nl_2 = [
            [1, 2],     # 0
            [0, 2],     # 1
            [0, 1],     # 2
            [4, 5],     # 3
            [3, 5, 6],  # 4
            [3, 4, 6,], # 5
            [4, 5],     # 6
            []          # 7
        ]

        self.assertListEqual(nl_1, bonds.neighborlists_from_adjacency_matrix(mat,1))
        self.assertListEqual(nl_2, bonds.neighborlists_from_adjacency_matrix(mat,2))

class Test_connected_sets(unittest.TestCase):

    def test_just_works(self):
        mat = _np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1]])
        sets = bonds.connected_sets(mat)
        self.assertListEqual([ss.tolist() for ss in sets], [[0,1,2],[3,4,5,6],[7]])

    def test_asymmetric(self):
        mat = _np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1]])
        sets = bonds.connected_sets(mat)
        self.assertListEqual([ss.tolist() for ss in sets], [[0,1,2],[3,4,5,6],[7]])

    def test_asymmetric_no_diagonal(self):
        mat = _np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]])
        sets = bonds.connected_sets(mat)
        self.assertListEqual([ss.tolist() for ss in sets], [[0,1,2],[3,4,5,6],[7]])

class Test_top2residue_bond_matrix_naive(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.small_monomer)
        self.fragments = [[0,1,2],[3,4,5],[6,7]]
        """
          Auto-detected fragments with method 'chains'
          fragment      0 with      3 AAs    GLU30 (     0) -    TRP32 (2     ) (0) 
          fragment      1 with      3 AAs    ILE26 (     3) -    LYS29 (5     ) (1)  resSeq jumps
          fragment      2 with      2 AAs   P0G381 (     6) -   GDP382 (7     ) (2) 
    
        """

    def test_just_works(self):
        mat = bonds.top2residue_bond_matrix_naive(self.geom.top)
        _np.testing.assert_array_equal(mat,
                                       _np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                                                  [1, 1, 1, 0, 0, 0, 0, 0],
                                                  [0, 1, 1, 1, 0, 0, 0, 0],
                                                  [0, 0, 1, 1, 1, 0, 0, 0],
                                                  [0, 0, 0, 1, 1, 1, 0, 0],
                                                  [0, 0, 0, 0, 1, 1, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 0, 0, 0, 1]])
                                       )

    def test_all_protein(self):
        mat = bonds.top2residue_bond_matrix_naive(self.geom.top, only_protein=False)
        _np.testing.assert_array_equal(mat,
                                       _np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                                                  [1, 1, 1, 0, 0, 0, 0, 0],
                                                  [0, 1, 1, 1, 0, 0, 0, 0],
                                                  [0, 0, 1, 1, 1, 0, 0, 0],
                                                  [0, 0, 0, 1, 1, 1, 0, 0],
                                                  [0, 0, 0, 0, 1, 1, 1, 0],
                                                  [0, 0, 0, 0, 0, 1, 1, 1],
                                                  [0, 0, 0, 0, 0, 0, 1, 1]])
                                       )
    def test_protein_chains(self):
        mat = bonds.top2residue_bond_matrix_naive(self.geom.top, fragments=self.fragments)
        _np.testing.assert_array_equal(mat,
                                       _np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                                                  [1, 1, 1, 0, 0, 0, 0, 0],
                                                  [0, 1, 1, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 1, 1, 0, 0, 0],
                                                  [0, 0, 0, 1, 1, 1, 0, 0],
                                                  [0, 0, 0, 0, 1, 1, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 0, 0, 0, 1]])
                                       )
if __name__ == '__main__':
   unittest.main()
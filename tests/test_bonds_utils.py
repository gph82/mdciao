import mdtraj as md
import unittest
import numpy as _np
from mdciao.examples import filenames as test_filenames
from mdciao.utils import bonds
from tempfile import NamedTemporaryFile as _NamedTemporaryFile

class Test_top2residue_bond_matrix(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.small_monomer)
        self.geom_force_resSeq_breaks = md.load(test_filenames.small_monomer_LYS99)
        self.geom_no_bonds = md.load(test_filenames.file_for_no_bonds_gro)
        self.geom_no_bonds.top._bonds=[]
        f = ['Generated with MDTraj, t= 0.0',
             ' 21',
             '  198TYR      N 3256   5.902   5.969   4.571',
             '  198TYR     CA 3258   5.854   6.105   4.548',
             '  198TYR     CB 3260   5.770   6.114   4.421',
             '  198TYR     CG 3263   5.718   6.247   4.382',
             '  198TYR    CD1 3264   5.772   6.313   4.273',
             '  198TYR    CE1 3266   5.727   6.444   4.233',
             '  198TYR     CZ 3268   5.628   6.497   4.305',
             '  198TYR     OH 3269   5.577   6.626   4.276',
             '  198TYR    CD2 3271   5.615   6.310   4.455',
             '  198TYR    CE2 3273   5.570   6.439   4.417',
             '  198TYR      C 3275   5.785   6.160   4.671',
             '  198TYR      O 3276   5.806   6.274   4.712',
             '  199GLUT     N 3277   5.701   6.072   4.723',
             '  199GLUT    CA 3279   5.608   6.111   4.837',
             '  199GLUT    CB 3281   5.500   6.007   4.849',
             '  199GLUT    CG 3284   5.431   5.990   4.987',
             '  199GLUT    CD 3287   5.338   6.097   5.022',
             '  199GLUT   OE1 3288   5.228   6.130   4.973',
             '  199GLUT   OE2 3289   5.388   6.160   5.134',
             '  199GLUT     C 3291   5.683   6.146   4.964',
             '  199GLUT     O 3292   5.660   6.244   5.033',
             '   9.60680   9.60680   9.60680   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000']
        with _NamedTemporaryFile(suffix=".gro") as tf:
            with open(tf.name, "w") as gro:
                gro.writelines("\n".join(f) + "\n")
            self.top_w_trir_res = md.load(tf.name).top

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

    def test_bond_matrix_titratable_False(self):
        _np.testing.assert_array_equal(bonds.top2residue_bond_matrix(self.top_w_trir_res, bond_titrable_residues=False), [[1,0],
                                                                                                                          [0,0]])
    def test_bond_matrix_titratable_True(self):
        _np.testing.assert_array_equal(bonds.top2residue_bond_matrix(self.top_w_trir_res, bond_titrable_residues=True), [[1,1],
                                                                                                                          [1,1]])

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

        nl_2_0 = [
            [1, 2],     # 0
            [], # 1
            [], # 2
            [], # 3
            [], # 4
            [], # 5
            [], # 6
            []  # 7
        ]

        self.assertListEqual(nl_1, bonds.neighborlists_from_adjacency_matrix(mat,1))
        self.assertListEqual(nl_2, bonds.neighborlists_from_adjacency_matrix(mat,2))
        self.assertListEqual(nl_2_0,
                              bonds.neighborlists_from_adjacency_matrix(mat, 2, indices=[0]))

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
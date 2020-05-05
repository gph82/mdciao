import unittest
from filenames import filenames
import mdtraj as md
import numpy as _np
from tempfile import TemporaryDirectory as _TDir
from os import path
from filecmp import cmp
test_filenames = filenames()

from mdciao.contact_matrix_utils import atom_idxs2GROMACSndx, \
    per_xtc_ctc_mat_dict

class Test_atom_idxs2indexfile(unittest.TestCase):

    def test_against_reference(self):
        geom = md.load(test_filenames.prot1_pdb)
        atom_idxs = _np.arange(geom.n_atoms)
        with _TDir(suffix="_mdciao_test") as tmpdir:
            indexfile = path.join(tmpdir,"test.ndx")
            atom_idxs2GROMACSndx(atom_idxs, "System",
                                 indexfile=indexfile,
                                 )
            assert cmp(indexfile,test_filenames.index_file)

class Test_per_xtc_ctc_mat_dict(unittest.TestCase):

    def setUp(self):
        self.pdb = test_filenames.prot1_pdb
        self.xtc = test_filenames.run1_stride_100_xtc
    def test_works(self):
        per_xtc_ctc_mat_dict(self.pdb,
                             self.xtc,
                             0)


if __name__ == '__main__':
    unittest.main()

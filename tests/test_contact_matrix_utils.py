import unittest
from filenames import filenames
import mdtraj as md
import numpy as _np
from tempfile import TemporaryDirectory as _TDir
from os import path
from filecmp import cmp
test_filenames = filenames()

from mdciao.contact_matrix_utils import atom_idxs2indexfile

class Test_atom_idxs2indexfile(unittest.TestCase):

    def test_against_reference(self):
        geom = md.load(test_filenames.prot1_pdb)
        atom_idxs = _np.arange(geom.n_atoms)
        with _TDir(suffix="_mdciao_test") as tmpdir:
            indexfile = path.join(tmpdir,"test.ndx")
            atom_idxs2indexfile(atom_idxs,"System",
                                indexfile=indexfile,
                                )
            assert cmp(indexfile,test_filenames.index_file)



if __name__ == '__main__':
    unittest.main()

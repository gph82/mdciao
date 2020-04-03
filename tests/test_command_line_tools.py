import mdtraj as md
import unittest
from filenames import filenames
from mdciao.command_line_tools import residue_neighborhoods, sites, interface
from tempfile import TemporaryDirectory
test_filenames = filenames()

class test_minimal_runs(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.prot1_pdb)
        self.run1_stride_100_xtc = md.load(test_filenames.run1_stride_100_xtc, top=self.geom.top)
        self.run1_stride_100_xtc_reverse = md.load(test_filenames.run1_stride_100_xtc, top=self.geom.top)[::-1]

    def test_neighborhoods(self):
        with TemporaryDirectory(suffix='_test_molpx_notebook') as tmpdir:
             residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                   "396",
                                   output_dir=tmpdir)

    def test_sites(self):
        with TemporaryDirectory(suffix='_test_molpx_notebook') as tmpdir:
             sites(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                   [test_filenames.GDP_json],
                                   output_dir=tmpdir)

    def test_interface(self):
        with TemporaryDirectory(suffix='_test_molpx_notebook') as tmpdir:
            interface(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                      frag_idxs_group_1=[0],
                      frag_idxs_group_2=[1],
                      output_dir=tmpdir)

if __name__ == '__main__':
    unittest.main()
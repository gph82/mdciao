import mdtraj as md
import unittest
from filenames import filenames
import pytest
from mdciao.command_line_tools import residue_neighborhoods, \
    sites, \
    interface, \
    contact_map
from tempfile import TemporaryDirectory
test_filenames = filenames()

class test_just_runs_all_fewest_options(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.prot1_pdb)
        self.run1_stride_100_xtc = md.load(test_filenames.run1_stride_100_xtc, top=self.geom.top)
        self.run1_stride_100_xtc_reverse = md.load(test_filenames.run1_stride_100_xtc, top=self.geom.top)[::-1]

    def test_neighborhoods(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                   "396",
                                   output_dir=tmpdir)

    def test_sites(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             sites(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                   [test_filenames.GDP_json],
                                   output_dir=tmpdir)

    def test_interface(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            interface(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                      frag_idxs_group_1=[0],
                      frag_idxs_group_2=[1],
                      output_dir=tmpdir)

    def test_contact_map(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            contact_map(self.geom, [self.run1_stride_100_xtc,
                                    self.run1_stride_100_xtc_reverse],
                      output_dir=tmpdir)


class test_residue_neighbrhoods_options_except_nomenclature_just_runs(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.prot1_pdb)
        self.run1_stride_100_xtc = md.load(test_filenames.run1_stride_100_xtc, top=self.geom.top)
        self.run1_stride_100_xtc_reverse = md.load(test_filenames.run1_stride_100_xtc, top=self.geom.top)[::-1]

    def test_res_idxs(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                   "1067",
                                   color_by_fragment=True,
                                   allow_same_fragment_ctcs=False,
                                   short_AA_names=True,
                                   res_idxs=True,
                                   table_ext=".dat",
                                   output_dir=tmpdir)

    def test_excel(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                   "396",
                                   table_ext=".xlsx",
                                   output_dir=tmpdir)

    def test_distro(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                   "396",
                                   distro=True,
                                   output_dir=tmpdir)

    def test_AAs(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                   "396",
                                   short_AA_names=True,
                                   output_dir=tmpdir)

    def test_colors(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                  "396",
                                  short_AA_names=True,
                                  color_by_fragment=False,
                                  output_dir=tmpdir)

        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                  "396",
                                  short_AA_names=True,
                                  color_by_fragment='c',
                                  output_dir=tmpdir)

    def test_just_N_ctcs(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                   "396",
                                   plot_timedep=False,
                                   separate_N_ctcs=True,
                                   short_AA_names=True,
                                   output_dir=tmpdir)

    def test_no_timedep_yes_N_ctcs(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                   "396",
                                   plot_timedep=False,
                                   separate_N_ctcs=True,
                                   short_AA_names=True,
                                   output_dir=tmpdir)

    def test_no_residues_returns_None(self):
        # This is more of an argparse fail
        # TODO consider throwing exception
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             assert None == residue_neighborhoods(self.geom, [self.run1_stride_100_xtc,
                                                              self.run1_stride_100_xtc_reverse],
                                   None,
                                   distro=True,
                                   output_dir=tmpdir)


    def test_separate_N_ctcs(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             residue_neighborhoods(self.geom, [self.run1_stride_100_xtc,
                                               self.run1_stride_100_xtc_reverse],
                                   "396",
                                   separate_N_ctcs=True,
                                   output_dir=tmpdir)


    def test_separate_N_ctcs_no_time_trace(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             residue_neighborhoods(self.geom, [self.run1_stride_100_xtc,
                                               self.run1_stride_100_xtc_reverse],
                                   "396",
                                   plot_timedep=False,
                                   separate_N_ctcs=True,
                                   output_dir=tmpdir)

    def test_wrong_input_resSeq_idxs(self):
        with pytest.raises(ValueError):
            with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
                 residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                       "300-200",
                                       distro=True,
                                       output_dir=tmpdir)

    def test_fragmentify_not_implemented(self):
        with pytest.raises(NotImplementedError):
            with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
                 residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                       "396",
                                       distro=True,
                                       fragmentify=False,
                                       output_dir=tmpdir)



class test_maximal_runs_no_nomenclature(unittest.TestCase):
    #this is a WIP

    def __test_sites(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             sites(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                   [test_filenames.GDP_json],
                                   output_dir=tmpdir)

    def __test_interface(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            interface(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                      frag_idxs_group_1=[0],
                      frag_idxs_group_2=[1],
                      output_dir=tmpdir)



if __name__ == '__main__':
    unittest.main()
import mdtraj as md
import unittest
from filenames import filenames
import pytest

from mdciao import command_line_tools

from mdciao.command_line_tools import \
    residue_neighborhoods, \
    sites, \
    interface

from mdciao.nomenclature_utils import \
    LabelerBW

from mdciao.contact_matrix_utils import \
    contact_map

from tempfile import TemporaryDirectory
test_filenames = filenames()

from unittest.mock import patch
from mock import mock

from pandas import \
    unique as _pandasunique

from os import \
    path as _path

from mdciao.fragments import \
    get_fragments

class TestJustRunsAllFewestOptions(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.prot1_pdb)
        self.run1_stride_100_xtc = md.load(test_filenames.run1_stride_100_xtc, top=self.geom.top)
        self.run1_stride_100_xtc_reverse = md.load(test_filenames.run1_stride_100_xtc, top=self.geom.top)[::-1]

    @patch('builtins.input', lambda *args: '4')
    def test_neighborhoods(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                   "200,396",
                                  output_dir=tmpdir
                                   )

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
    @unittest.skip("contact map is not being exposed anywhere ATM")
    def test_contact_map(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            contact_map(self.geom, [self.run1_stride_100_xtc,
                                    self.run1_stride_100_xtc_reverse],
                      output_dir=tmpdir)

class Test_residue_neighbrhoodsOptionsJustRuns(unittest.TestCase):
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

    def test_nomenclature_BW(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            input_values = (val for val in ["","a"])
            with mock.patch('builtins.input', lambda *x: next(input_values)):
                residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                      "131",
                                      BW_uniprot=_path.join(test_filenames.test_data_path,"adrb2_human_full"),
                                      # TODO include this in filenames
                                      output_dir=tmpdir
                                      )
    def test_nomenclature_CGN(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            input_values = (val for val in ["", "a"])
            with mock.patch('builtins.input', lambda *x: next(input_values)):
                residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                      "131",
                                      CGN_PDB="3SN6",
                                      output_dir=tmpdir
                                      )
    def test_nomenclature_CGN_and_BW(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            input_values = (val for val in ["", "", "a"])
            with mock.patch('builtins.input', lambda *x: next(input_values)):
                residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                      "131",
                                      CGN_PDB="3SN6",
                                      BW_uniprot=_path.join(test_filenames.test_data_path, "adrb2_human_full"),
                                      output_dir=tmpdir
                                      )

class Test_parse_consensus_option(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.prot1_pdb)

    def test_empty(self):
        residx2conlab= command_line_tools._parse_consensus_option(None,None,self.geom.top,
                                                   None,
                                                  )
        assert _pandasunique(residx2conlab)[0] is None

    def test_empty_w_return(self):
        residx2conlab, lblr  = command_line_tools._parse_consensus_option(None,None,self.geom.top,
                                                   None,
                                                   return_Labeler=True)
        assert lblr is None
        assert _pandasunique(residx2conlab)[0] is None

    def test_with_BW(self):
        option = _path.join(test_filenames.test_data_path,
                                "adrb2_human_full")
        fragments = get_fragments(self.geom.top)
        input_values = (val for val in [""])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residx2conlab, lblr = command_line_tools._parse_consensus_option(option, "BW",
                                                   self.geom.top,
                                                   fragments,
                                                   return_Labeler=True)
            self.assertIsInstance(lblr, LabelerBW)
            self.assertIsInstance(residx2conlab,list)

class Test_offer_to_create_dir(unittest.TestCase):

    def test_creates_dir(self):
       with TemporaryDirectory() as tmpdir:
           newdir = _path.join(tmpdir,"non_existent")
           input_values = (val for val in [""])
           with mock.patch('builtins.input', lambda *x: next(input_values)):
               command_line_tools._offer_to_create_dir(newdir)
           assert _path.exists(newdir)

    def test_does_nothing_bc_dir_exists(self):
       with TemporaryDirectory() as tmpdir:
           command_line_tools._offer_to_create_dir(tmpdir)

    def test_raises(self):
        input_values = (val for val in ["n"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            res = command_line_tools._offer_to_create_dir("testdir")
            assert res is None

if __name__ == '__main__':
    unittest.main()
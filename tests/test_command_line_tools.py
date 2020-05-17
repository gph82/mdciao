import mdtraj as md
import unittest
from filenames import filenames
import pytest

from tempfile import TemporaryDirectory as _TDir

from matplotlib import \
    pyplot as _plt

from mdciao import command_line_tools

from mdciao.contacts import ContactGroup, ContactPair

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

class TestCLTBaseClass(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.prot1_pdb)
        self.run1_stride_100_xtc = md.load(test_filenames.run1_stride_100_xtc, top=self.geom.top)
        self.run1_stride_100_xtc_reverse = md.load(test_filenames.run1_stride_100_xtc, top=self.geom.top)[::-1]

class Test_manage_timdep_plot_options(TestCLTBaseClass):

    def setUp(self):
        super(Test_manage_timdep_plot_options, self).setUp()
        ctc_idxs = [[1067,1068],
                    [1067,334],
                    [1067,488]]
        CPs = [ContactPair(pair,
                           [md.compute_contacts(itraj, [pair])[0].squeeze() for itraj in
                            [self.run1_stride_100_xtc,
                             self.run1_stride_100_xtc_reverse[:10]]],
                           [self.run1_stride_100_xtc.time,
                            self.run1_stride_100_xtc.time[:10]],
                           top=self.geom.top,
                           anchor_residue_idx=1067)
               for pair in ctc_idxs]

        self.ctc_grp = ContactGroup(CPs,
                                    top=self.geom.top,
                                    )

    """
    1:     1.00   GDP396-MG397           9-10       1067    1068     162 1.00
    2:     1.00   GDP396-THR55           9-4        1067     334      12 2.00
    3:     1.00   GDP396-ASP223          9-4        1067     488      82 3.00
    4:     1.00   GDP396-ARG201          9-4        1067     466      67 4.00
    5:     1.00   GDP396-LYS53           9-4        1067     332      10 5.00    
    """

    def test_works(self):
        with _TDir(suffix="_test_mdciao") as tmpdir:
            myfig = self.ctc_grp.plot_timedep_ctcs(3,
                                                   ctc_cutoff_Ang=3,
                                                   plot_N_ctcs=True,
                                                   pop_N_ctcs=False,
                                                   skip_timedep=False,
                                                   )

            command_line_tools._manage_timedep_ploting_and_saving_options(self.ctc_grp,
                                                                          myfig, 3,
                                                                   "test_neigh", "png",
                                                                          output_dir=tmpdir,
                                                                          )
        _plt.close("all")

    def test_separate_N_ctcs(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            myfig = self.ctc_grp.plot_timedep_ctcs(3,
                                                   ctc_cutoff_Ang=3,
                                                   plot_N_ctcs=True,
                                                   pop_N_ctcs=True,
                                                   skip_timedep=False,
                                                   )
            command_line_tools._manage_timedep_ploting_and_saving_options(self.ctc_grp,
                                                                          myfig, 3,
                                                                   "test_neigh", "png",
                                                                          output_dir=tmpdir,
                                                                          separate_N_ctcs=True,
                                                                          )
            _plt.close("all")
    def test_just_N_ctcs(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            myfig = self.ctc_grp.plot_timedep_ctcs(3,
                                                   ctc_cutoff_Ang=3,
                                                   plot_N_ctcs=True,
                                                   pop_N_ctcs=True,
                                                   skip_timedep=True,

                                                   )
            command_line_tools._manage_timedep_ploting_and_saving_options(self.ctc_grp,
                                                                          myfig, 3,
                                                                   "test_neigh", "png",
                                                                          output_dir=tmpdir,
                                                                          separate_N_ctcs=True,
                                                                          plot_timedep=False,
                                                                          )
            _plt.close("all")

    """
    def test_no_timedep_yes_N_ctcs(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            residue_neighborhoods(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                                  "396",
                                  plot_timedep=False,
                                  separate_N_ctcs=True,
                                  short_AA_names=True,
                                  output_dir=tmpdir)

    def test_separate_N_ctcs_no_time_trace(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            residue_neighborhoods(self.geom, [self.run1_stride_100_xtc,
                                              self.run1_stride_100_xtc_reverse],
                                  "396",
                                  plot_timedep=False,
                                  separate_N_ctcs=True,
                                  output_dir=tmpdir)
    """


class TestJustRunsAllFewestOptions(TestCLTBaseClass):

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

    @unittest.skip("contact map is not being exposed anywhere ATM")
    def test_contact_map(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            contact_map(self.geom, [self.run1_stride_100_xtc,
                                    self.run1_stride_100_xtc_reverse],
                      output_dir=tmpdir)

class Test_residue_neighbrhoodsOptionsJustRuns(TestCLTBaseClass):

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

    def test_no_residues_returns_None(self):
        # This is more of an argparse fail
        # TODO consider throwing exception
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             assert None == residue_neighborhoods(self.geom, [self.run1_stride_100_xtc,
                                                              self.run1_stride_100_xtc_reverse],
                                   None,
                                   distro=True,
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

class Test_sites_missing_options(TestCLTBaseClass):

    def test_scheme_CA(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            sites(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                  [test_filenames.GDP_json],
                  output_dir=tmpdir,
                  scheme="COM")

    def test_fragmentify_raises(self):
        with pytest.raises(NotImplementedError):
            with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
                sites(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                      [test_filenames.GDP_json],
                      output_dir=tmpdir,
                      scheme="COM",
                      fragmentify=False)

class Test_interface(TestCLTBaseClass):

    def test_interface(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            interface(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                      frag_idxs_group_1=[0],
                      frag_idxs_group_2=[1],
                      output_dir=tmpdir)

    def test_interface_wo_frag_idxs_groups(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            input_values = (val for val in ["0-1","2,3"])
            with mock.patch('builtins.input', lambda *x: next(input_values)):
                interface(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                          output_dir=tmpdir)

    def test_w_just_one_fragment_by_user(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            interface(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                      output_dir=tmpdir,
                      fragments=["0-5"])

    def test_w_just_one_fragment_by_user_and_n_ctcs(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            interface(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                      output_dir=tmpdir,
                      fragments=["0-5"],
                      n_nearest=1)

    def test_w_just_two_fragments_by_user(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            interface(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                      frag_idxs_group_1=[0],
                      frag_idxs_group_2=[1],
                      output_dir=tmpdir,
                      fragments=["0-5",
                                 "6-10"])

    def test_w_nomenclature_CGN_BW(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            input_values = (val for val in ["","", "", ""])
            with mock.patch('builtins.input', lambda *x: next(input_values)):
                interface(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                          frag_idxs_group_1=[0],
                          frag_idxs_group_2=[1],
                          output_dir=tmpdir,
                          fragments=["202-238",
                                     "634-659"],
                          CGN_PDB="3SN6",
                          BW_uniprot=_path.join(test_filenames.test_data_path, "adrb2_human_full"),
                          )

    def test_w_nomenclature_CGN_BW_fragments_are_consensus(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            input_values = (val for val in ["","", "", "", "TM6","*H5"])
            with mock.patch('builtins.input', lambda *x: next(input_values)):
                interface(self.geom, [self.run1_stride_100_xtc, self.run1_stride_100_xtc_reverse],
                          frag_idxs_group_1=[0],
                          frag_idxs_group_2=[1],
                          output_dir=tmpdir,
                          fragments=["consensus"],
                          CGN_PDB="3SN6",
                          BW_uniprot=_path.join(test_filenames.test_data_path, "adrb2_human_full"),
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

    def test_with_BW_already_instantiated(self):
        fragments = get_fragments(self.geom.top)
        BW = LabelerBW("adrb2_human_full",
                       local_path=test_filenames.test_data_path)
        input_values = (val for val in [""])
        with mock.patch('builtins.input', lambda *x: next(input_values)):

            residx2conlab, lblr = command_line_tools._parse_consensus_option(BW, "BW",
                                                   self.geom.top,
                                                   fragments,
                                                   return_Labeler=True)
            self.assertIsInstance(lblr, LabelerBW)
            self.assertEqual(lblr,BW)
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
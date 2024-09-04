import mdtraj as md
import unittest

import mdciao.contacts
from mdciao.examples import filenames as test_filenames
from mdciao.utils import str_and_dict

#see https://stackoverflow.com/questions/169070/how-do-i-write-a-decorator-that-restores-the-cwd
import contextlib
@contextlib.contextmanager
def remember_cwd():
    curdir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(curdir)

import shutil

from tempfile import TemporaryDirectory as _TDir

import os

from matplotlib import \
    pyplot as _plt

from mdciao import cli
from mdciao import fragments as mdcfragments
from mdciao import contacts

from mdciao.cli import *
#    residue_neighborhoods, \
#    sites, \
#    interface

from mdciao.nomenclature import \
    LabelerGPCR

from mdciao.parsers import \
    parser_for_CGN_overview, \
    parser_for_GPCR_overview

#from mdciao.contact_matrix_utils import \
#    contact_map

from tempfile import TemporaryDirectory

from unittest import mock
from pandas import \
    unique as _pandasunique

from os import \
    path as _path

import numpy as _np

class TestCLTBaseClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.geom = md.load(test_filenames.top_pdb)
        cls.traj = md.load(test_filenames.traj_xtc_stride_20, top=cls.geom.top)
        cls.traj_reverse = md.load(test_filenames.traj_xtc_stride_20, top=cls.geom.top)[::-1]

class Test_manage_timdep_plot_options(TestCLTBaseClass):

    @classmethod
    def setUpClass(cls):
        super(Test_manage_timdep_plot_options, cls).setUpClass()
        ctc_idxs = [[353, 348],
                    [353, 972],
                    [353, 347]]
        CPs = [contacts.ContactPair(pair,
                                    [md.compute_contacts(itraj, [pair])[0].squeeze() for itraj in
                                     [cls.traj,
                                      cls.traj_reverse[:10]]],
                                    [cls.traj.time,
                                     cls.traj_reverse.time[:10]],
                                    top=cls.geom.top,
                                    anchor_residue_idx=353)
               for pair in ctc_idxs]

        cls.ctc_grp = contacts.ContactGroup(CPs,
                                            top=cls.geom.top,
                                            neighbors_excluded=0,
                                            )

    """
    1:     0.55   LEU394-ARG389       0-0         353-348        33     0.55
    2:     0.47   LEU394-LYS270       0-3         353-972        71     1.02
    3:     0.38   LEU394-LEU388       0-0         353-347        32     1.39
    4:     0.23   LEU394-LEU230       0-3         353-957        56     1.62
    5:     0.10   LEU394-ARG385       0-0         353-344        29     1.73

    """

    def test_works(self):
        with _TDir(suffix="_test_mdciao") as tmpdir:
            myfig = self.ctc_grp.plot_timedep_ctcs(ctc_cutoff_Ang=3)
            fn= str_and_dict.FilenameGenerator("test_neig",3,tmpdir,"png","dat",150,"ps")
            cli._manage_timedep_ploting_and_saving_options(self.ctc_grp, fn, myfig,
                                                           savetrajs=True)
        _plt.close("all")

    def test_separate_N_ctcs(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            myfig = self.ctc_grp.plot_timedep_ctcs(pop_N_ctcs=True, ctc_cutoff_Ang=3)
            fn= str_and_dict.FilenameGenerator("test_neig",3,tmpdir,"png","dat",150,"ps")
            cli._manage_timedep_ploting_and_saving_options(self.ctc_grp,fn,myfig,
                                                           separate_N_ctcs=True,
                                                           savetrajs=True,
                                                           )
            _plt.close("all")
    def test_just_N_ctcs(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            myfig = self.ctc_grp.plot_timedep_ctcs(pop_N_ctcs=True, skip_timedep=True, ctc_cutoff_Ang=3)
            fn= str_and_dict.FilenameGenerator("test_neig",3,tmpdir,"png","dat",150,"ps")
            cli._manage_timedep_ploting_and_saving_options(self.ctc_grp, fn, myfig,
                                                           separate_N_ctcs=True,
                                                           plot_timedep=False,
                                                           )
            _plt.close("all")

    def test_no_files(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            fn = str_and_dict.FilenameGenerator("test_neig", 3, tmpdir, "png", "dat", 150, "ps")
            cli._manage_timedep_ploting_and_saving_options(self.ctc_grp, fn, [],
                                                           separate_N_ctcs=True,
                                                           plot_timedep=False,
                                                           )
            _plt.close("all")


class Test_residue_neighborhood(TestCLTBaseClass):

    @classmethod
    def setUpClass(cls):
        super(Test_residue_neighborhood, cls).setUpClass()
        cls.no_disk = True  # doesn't seem to speed things up much

    @mock.patch('builtins.input', lambda *args: '4')
    def test_neighborhoods_no_disk_works(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            with remember_cwd():
                os.chdir(tmpdir)
                input_values = (val for val in ["1.0"])
                with mock.patch('builtins.input', lambda *x: next(input_values)):
                    cli.residue_neighborhoods("200,395",
                                              self.traj,
                                              self.geom,
                                              no_disk=True
                                              )
                assert len(os.listdir(".")) == 0

    @mock.patch('builtins.input', lambda *args: '4')
    def test_neighborhoods(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            input_values = (val for val in ["1.0"])
            with mock.patch('builtins.input', lambda *x: next(input_values)):
                cli.residue_neighborhoods("200,395",
                                          [self.traj, self.traj_reverse],
                                          self.geom,
                                          output_dir=tmpdir,
                                          no_disk=self.no_disk)

    def test_no_top(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            input_values = (val for val in ["1.0"])
            with mock.patch('builtins.input', lambda *x: next(input_values)):
                cli.residue_neighborhoods("200,395",
                                          self.geom,
                                          output_dir=tmpdir,
                                          no_disk=self.no_disk)

    def test_res_idxs(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            cli.residue_neighborhoods("1043",
                                      [self.traj, self.traj_reverse],
                                      self.geom,
                                      fragment_colors=True,
                                      allow_same_fragment_ctcs=False,
                                      short_AA_names=True,
                                      res_idxs=True,
                                      output_dir=tmpdir,
                                      no_disk=self.no_disk)

    def test_excel(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            cli.residue_neighborhoods("395",
                                      [self.traj, self.traj_reverse],
                                      self.geom,
                                      table_ext=".xlsx",
                                      output_dir=tmpdir)

    def test_distro(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            cli.residue_neighborhoods("395",
                                      [self.traj, self.traj_reverse],
                                      self.geom,
                                      distro=True,
                                      output_dir=tmpdir,
                                      no_disk=self.no_disk)

    def test_AAs(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            cli.residue_neighborhoods("395",
                                      [self.traj, self.traj_reverse],
                                      self.geom,
                                      short_AA_names=True,
                                      output_dir=tmpdir,
                                      no_disk=self.no_disk)

    def test_colors(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            cli.residue_neighborhoods("395",
                                      [self.traj, self.traj_reverse],
                                      self.geom,
                                      short_AA_names=True,
                                      fragment_colors=True,
                                      output_dir=tmpdir,
                                      no_disk=self.no_disk)

        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            cli.residue_neighborhoods("395",
                                      [self.traj, self.traj_reverse],
                                      self.geom,
                                      short_AA_names=True,
                                      fragment_colors='c',
                                      output_dir=tmpdir,
                                      no_disk=self.no_disk)

    def test_no_residues_returns_None(self):
        # This is more of an argparse fail
        # TODO consider throwing exception
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            assert None == cli.residue_neighborhoods(None,
                                                     [self.traj, self.traj_reverse],
                                                     self.geom,
                                                     None,
                                                     distro=True,
                                                     output_dir=tmpdir,
                                                     no_disk=self.no_disk)

    def test_wrong_input_resSeq_idxs(self):
        with self.assertRaises(ValueError):
            with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
                cli.residue_neighborhoods("AX*",
                                          [self.traj, self.traj_reverse],
                                          self.geom,
                                          distro=True,
                                          output_dir=tmpdir,
                                          no_disk=self.no_disk)

    def test_nomenclature_GPCR(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            cli.residue_neighborhoods("R131",
                                      [self.traj, self.traj_reverse],
                                      self.geom,
                                      GPCR_UniProt=test_filenames.adrb2_human_xlsx,
                                      output_dir=tmpdir,
                                      accept_guess=True,
                                      no_disk=self.no_disk
                                      )

    def test_nomenclature_CGN(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            shutil.copy(test_filenames.gnas2_human_xlsx, tmpdir)
            with remember_cwd():
                os.chdir(tmpdir)
                cli.residue_neighborhoods("R131",
                                          [self.traj, self.traj_reverse],
                                          self.geom,
                                          CGN_UniProt="gnas2_human",
                                          output_dir=tmpdir,
                                          accept_guess=True,
                                          no_disk=self.no_disk
                                          )

    def test_nomenclature_CGN_and_GPCR(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            shutil.copy(test_filenames.gnas2_human_xlsx, tmpdir)
            with remember_cwd():
                os.chdir(tmpdir)
                cli.residue_neighborhoods("R131",
                                          [self.traj, self.traj_reverse],
                                          self.geom,
                                          CGN_UniProt="gnas2_human",
                                          GPCR_UniProt=test_filenames.adrb2_human_xlsx,
                                          output_dir=tmpdir,
                                          accept_guess=True,
                                          no_disk=self.no_disk
                                          )

    def test_no_contacts_at_allp(self):
        cli.residue_neighborhoods("R131",
                                  [self.traj, self.traj_reverse],
                                  self.geom,
                                  ctc_cutoff_Ang=.1,
                                  )

    def test_some_CG_have_no_contacts(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            cli.residue_neighborhoods("0-3",
                                      [self.traj, self.traj_reverse],
                                      self.geom,
                                      ctc_cutoff_Ang=3.2,
                                      res_idxs=True,
                                      output_dir=tmpdir,
                                      no_disk=self.no_disk
                                      )

    def test_no_bonds_fail(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            top = self.geom[0]
            top.top._bonds = []
            with self.assertRaises(ValueError):
                cli.residue_neighborhoods("R131",
                                          [self.traj, self.traj_reverse],
                                          top,
                                          output_dir=tmpdir)
    def test_naive_bonds_CAs(self):
        geom_CAs = self.geom[0].atom_slice(self.geom[0].top.select("name CA"))
        print(geom_CAs)
        with self.assertRaises(ValueError):
            cli.residue_neighborhoods("R131",
                                      [geom_CAs],
                                      no_disk=True,
                                      figures=False
                                      )
        n = cli.residue_neighborhoods("R131",
                                      [geom_CAs],
                                      no_disk=True,
                                      figures=False,
                                      naive_bonds=True,
                                      ctc_cutoff_Ang=1000,
                                      ctc_control=1.,
                                      )[861]
        assert n.n_ctcs==geom_CAs.top.n_residues-4-4-1# all residues minus 4 neighbors (x2 directions) minus itself

class Test_sites(TestCLTBaseClass):

    def test_sites(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             cli.sites([test_filenames.tip_json],[self.traj, self.traj_reverse], self.geom,
                  output_dir=tmpdir)

    def test_sites_no_distk(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             cli.sites([test_filenames.tip_json],[self.traj, self.traj_reverse], self.geom,
                  no_disk=True)

    def test_scheme_CA(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             cli.sites([test_filenames.tip_json], [self.traj, self.traj_reverse],self.geom,
                  output_dir=tmpdir,
                  scheme="COM")

    def test_w_table_xlsx(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             cli.sites([test_filenames.tip_json],[self.traj, self.traj_reverse], self.geom,
                  output_dir=tmpdir,
                  scheme="COM",
                       table_ext=".xlsx")

    def test_w_table_dat(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             cli.sites([test_filenames.tip_json], [self.traj, self.traj_reverse],self.geom,
                  output_dir=tmpdir,
                  scheme="COM",
                       table_ext=".dat")
    def test_sites_no_top(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             cli.sites([test_filenames.tip_json], [self.traj, self.traj_reverse],
                       output_dir=tmpdir)

    def test_sites_distro(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
             cli.sites([test_filenames.tip_json],
                       [self.traj, self.traj_reverse],
                       self.geom,
                       distro=True,
                       output_dir=tmpdir)

    def test_sites_w_incomplete_sites(self):
        # Only the last site will survive
        site_list = [
            {'name': 'site0', 'pairs': {'AAresSeq': ['ALA20-ALA21',  # ALA20 doesn't exist
                                                     'GLU101-GLU122']}},  # both exist

            {'name': 'site1', 'pairs': {'AAresSeq': ['GLU31-ALA20',  # GLU31 and ALA20 don't exist
                                                     'GLU17-GLU12']}},  # both exist

            {'name': 'site2', 'pairs': {'AAresSeq': ['GLN101-ALA122']}},  # GLN101 doesn't exist

            {'name': 'site3', 'pairs': {'AAresSeq': ['GLU101-GLU122']}}  # both exist, but was seen before
        ]
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            output_sites = cli.sites(site_list,
                      [self.traj, self.traj_reverse],
                      self.geom,
                      distro=True,
                      figures=False,
                      no_disk=True,
                      output_dir=tmpdir)
            assert len(output_sites)==1
            assert "site3" in output_sites.keys()
            assert output_sites["site3"]
            assert output_sites["site3"].n_ctcs == 1
            _np.testing.assert_array_equal(output_sites["site3"].res_idxs_pairs[0], [ 69, 852])

    def test_sites_w_incomplete_sites_allow_partial_sites(self):
        # Only the last site will survive
        site_list = [
            {'name': 'site0', 'pairs': {'AAresSeq': ['ALA20-ALA21',  # ALA20 doesn't exist
                                                     'GLU101-GLU122']}},  # both exist

            {'name': 'site1', 'pairs': {'AAresSeq': ['GLU31-ALA20',  # GLU31 and ALA20 don't exist
                                                     'GLU17-GLU12']}},  # both exist

            {'name': 'site2', 'pairs': {'AAresSeq': ['GLN101-ALA122']}},  # GLN101 doesn't exist

            {'name': 'site3', 'pairs': {'AAresSeq': ['GLU101-GLU122']}}  # both exist, but was seen before
        ]
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            output_sites = cli.sites(site_list,
                                     [self.traj, self.traj_reverse],
                                     self.geom,
                                     distro=True,
                                     figures=False,
                                     no_disk=True,
                                     output_dir=tmpdir,
                                     allow_partial_sites=True)
            assert len(output_sites)==3
            assert output_sites["site0"]
            assert output_sites["site0"].n_ctcs == 1
            _np.testing.assert_array_equal(output_sites["site0"].res_idxs_pairs[0], [69, 852])

            assert output_sites["site1"]
            assert output_sites["site1"].n_ctcs == 1
            _np.testing.assert_array_equal(output_sites["site1"].res_idxs_pairs[0], [709, 365])

            assert "site3" in output_sites.keys()
            assert output_sites["site3"]
            assert output_sites["site3"].n_ctcs == 1
            _np.testing.assert_array_equal(output_sites["site3"].res_idxs_pairs[0], [ 69, 852])

    def test_sites_consensus(self):
        res = cli.sites([test_filenames.tip_consensus_json],
                        [self.traj],
                        self.geom,
                        GPCR_UniProt=test_filenames.adrb2_human_xlsx,
                        CGN_UniProt=test_filenames.gnas2_human_xlsx,
                        no_disk=True, figures=False, accept_guess=True)
        res : mdciao.contacts.ContactGroup = res["interesting contacts"]
        self.assertSequenceEqual(res.ctc_labels_w_fragments_short_AA,
                                 ['L394@G.H5.26-K270@6.32x32',
                                  'D381@G.H5.13-Q229@5.68x68',
                                  'Q384@G.H5.16-Q229@5.68x68',
                                  'R385@G.H5.17-Q229@5.68x68',
                                  'D381@G.H5.13-K232@5.71x71',
                                  'Q384@G.H5.16-I135@3.54x54']
                                 )


class Test_interface(TestCLTBaseClass):

    @classmethod
    def setUpClass(cls):
        super(Test_interface, cls).setUpClass()
        cls.no_disk = True  # doesn't seem to speed things up much

    def test_interface(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            cli.interface([self.traj, self.traj_reverse],
                          self.geom,
                          interface_selection_1=[0],
                          interface_selection_2=[1],
                          output_dir=tmpdir,
                          flareplot=True,
                          no_disk=self.no_disk
                          )

    def test_no_top(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            cli.interface([self.traj, self.traj_reverse],
                          interface_selection_1=[0],
                          interface_selection_2=[1],
                          output_dir=tmpdir,
                          flareplot=False,
                          plot_timedep=False,
                          no_disk = self.no_disk
                          )

    def test_interface_wo_frag_idxs_groups(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            input_values = (val for val in ["0-1", "2,3"])
            with mock.patch('builtins.input', lambda *x: next(input_values)):
                cli.interface([self.traj, self.traj_reverse],
                              self.geom,
                              output_dir=tmpdir,
                              flareplot=False,
                              no_disk=self.no_disk
                              )

    def test_w_just_one_fragment_by_user(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            cli.interface([self.traj, self.traj_reverse],
                          self.geom,
                          output_dir=tmpdir,
                          fragments=["0-5"],
                          flareplot=False,
                          no_disk=self.no_disk
                          )

    def test_w_just_one_fragment_by_user_and_n_ctcs(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            cli.interface([self.traj, self.traj_reverse],
                          self.geom,
                          output_dir=tmpdir,
                          fragments=["0-5"],
                          n_nearest=1,
                          flareplot=False,
                          no_disk=self.no_disk
                          )

    def test_w_just_two_fragments_by_user(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            cli.interface([self.traj, self.traj_reverse],
                          self.geom,
                          interface_selection_1=[0],
                          interface_selection_2=[1],
                          output_dir=tmpdir,
                          fragments=["0-5",
                                     "6-10"],
                          flareplot=False,
                          no_disk=self.no_disk
                          )

    def test_w_nomenclature_CGN_GPCR(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            shutil.copy(test_filenames.gnas2_human_xlsx, tmpdir)
            shutil.copy(test_filenames.adrb2_human_xlsx, tmpdir)
            with remember_cwd():
                os.chdir(tmpdir)
                cli.interface([self.traj, self.traj_reverse],
                              self.geom,
                              output_dir=tmpdir,
                              fragments=["967-1001",  # TM6
                                         "328-353"],  # a5
                              CGN_UniProt="gnas2_human",
                              GPCR_UniProt="adrb2_human",
                              accept_guess=True,
                              flareplot=False,
                              no_disk=self.no_disk
                              )

    def test_w_nomenclature_CGN_GPCR_fragments_are_consensus(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            input_values = (val for val in ["TM6", "*H5"])
            shutil.copy(test_filenames.gnas2_human_xlsx, tmpdir)
            shutil.copy(test_filenames.adrb2_human_xlsx, tmpdir)
            with remember_cwd():
                os.chdir(tmpdir)
                with mock.patch('builtins.input', lambda *x: next(input_values)):
                    cli.interface([self.traj, self.traj_reverse],
                                  self.geom,
                                  output_dir=tmpdir,
                                  fragments=["consensus"],
                                  CGN_UniProt="gnas2_human",
                                  GPCR_UniProt="adrb2_human",
                                  accept_guess=True,
                                  flareplot=False,
                                  no_disk=self.no_disk
                                  )

    def test_w_nomenclature_CGN_GPCR_fragments_are_consensus_and_flareplot(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            input_values = (val for val in ["TM6", "*H5"])
            shutil.copy(test_filenames.gnas2_human_xlsx, tmpdir)
            shutil.copy(test_filenames.adrb2_human_xlsx, tmpdir)
            with remember_cwd():
                os.chdir(tmpdir)
                with mock.patch('builtins.input', lambda *x: next(input_values)):
                    cli.interface([self.traj, self.traj_reverse],
                                  self.geom,
                                  output_dir=tmpdir,
                                  fragments=["consensus"],
                                  CGN_UniProt="gnas2_human",
                                  GPCR_UniProt="adrb2_human",
                                  accept_guess=True,
                                  #no_disk=self.no_disk
                                  )


    def test_w_nomenclature_CGN_GPCR_fragments_are_consensus_and_flareplot_and_self_interface(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            shutil.copy(test_filenames.gnas2_human_xlsx, tmpdir)
            shutil.copy(test_filenames.adrb2_human_xlsx, tmpdir)
            with remember_cwd():
                os.chdir(tmpdir)
                cli.interface([self.traj, self.traj_reverse],
                              self.geom,
                              ctc_cutoff_Ang=5,
                              n_nearest=4,
                              output_dir=tmpdir,
                              fragments=["consensus"],
                              CGN_UniProt="gnas2_human",
                              GPCR_UniProt="adrb2_human",
                              accept_guess=True,
                              interface_selection_1='TM6',
                              interface_selection_2='TM5,TM6',
                              self_interface=True,
                              )

    def test_w_nomenclature_CGN_GPCR_fragments_are_consensus_and_flareplot_and_AA_selection_OR(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            shutil.copy(test_filenames.gnas2_human_xlsx, tmpdir)
            shutil.copy(test_filenames.adrb2_human_xlsx, tmpdir)
            with remember_cwd():
                os.chdir(tmpdir)
                intf = cli.interface([self.traj, self.traj_reverse],
                                     self.geom,
                                     ctc_cutoff_Ang=5,
                                     n_nearest=4,
                                     output_dir=tmpdir,
                                     fragments=["consensus"],
                                     CGN_UniProt="gnas2_human",
                                     GPCR_UniProt="adrb2_human",
                                     accept_guess=True,
                                     interface_selection_1='TM6',
                                     interface_selection_2='TM5',
                                     self_interface=True,
                                     AA_selection="5.50x50-5.55x55"
                                     )
            TM5 = ["5.50x50", "5.51x51", "5.52x52", "5.53x53", "5.54x54", "5.55x55"]
            assert all ([lab[1] in TM5 for lab in  intf.consensus_labels]), intf.consensus_labels
            _plt.close("all")

    def test_w_nomenclature_CGN_GPCR_fragments_are_consensus_and_flareplot_and_AA_selection_AND(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            shutil.copy(test_filenames.gnas2_human_xlsx, tmpdir)
            shutil.copy(test_filenames.adrb2_human_xlsx, tmpdir)
            with remember_cwd():
                intf = cli.interface([self.traj, self.traj_reverse],
                                     self.geom,
                                     ctc_cutoff_Ang=5,
                                     n_nearest=4,
                                     output_dir=tmpdir,
                                     fragments=["consensus"],
                                     CGN_UniProt="gnas2_human",
                                     GPCR_UniProt="adrb2_human",
                                     accept_guess=True,
                                     interface_selection_1='TM6',
                                     interface_selection_2='TM5',
                                     self_interface=True,
                                     AA_selection=["5.50x50-5.55x55", "6.45x45,6.49x49"]
                                     )
            TM5 = ["5.50x50", "5.51x51", "5.52x52", "5.53x53", "5.54x54", "5.55x55"]
            assert all ([lab[1] in TM5 for lab in  intf.consensus_labels]), intf.consensus_labels
            assert all ([lab[0] in ["6.45x45","6.49x49"] for lab in  intf.consensus_labels]), intf.consensus_labels
            _plt.close("all")

    def test_w_nomenclature_CGN_GPCR_fragments_are_consensus_and_flareplot_and_AA_selection_raises(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            shutil.copy(test_filenames.gnas2_human_xlsx, tmpdir)
            shutil.copy(test_filenames.adrb2_human_xlsx, tmpdir)
            with remember_cwd():
                with self.assertRaises(ValueError):
                    intf = cli.interface([self.traj, self.traj_reverse],
                                         self.geom,
                                         ctc_cutoff_Ang=5,
                                         n_nearest=4,
                                         output_dir=tmpdir,
                                         fragments=["consensus"],
                                         CGN_UniProt="gnas2_human",
                                         GPCR_UniProt="adrb2_human",
                                         accept_guess=True,
                                         interface_selection_1='TM6',
                                         interface_selection_2='TM5',
                                         self_interface=True,
                                         AA_selection=["5.50x50-5.55x55,6.45x45,6.49x49"]
                                         )




class Test_pdb(TestCLTBaseClass):

    def test_works(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            geom = cli.pdb("3SN6", filename=None)
            isinstance(geom,md.Trajectory)

class Test_parse_consensus_option(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.geom = md.load(test_filenames.top_pdb)

    def test_empty(self):
        residx2conlab= cli._parse_consensus_option(None, None, self.geom.top,
                                                   None,
                                                   )
        assert _pandasunique(_np.array(residx2conlab))[0] is None

    def test_empty_w_return(self):
        residx2conlab, lblr  = cli._parse_consensus_option(None, None, self.geom.top,
                                                           None,
                                                           return_Labeler=True)
        assert lblr is None
        assert _pandasunique(_np.array(residx2conlab))[0] is None

    def test_with_GPCR(self):
        fragments = mdcfragments.get_fragments(self.geom.top)
        input_values = (val for val in [""])
        option = test_filenames.adrb2_human_xlsx
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residx2conlab, lblr = cli._parse_consensus_option(option, "GPCR",
                                                              self.geom.top,
                                                              fragments,
                                                              return_Labeler=True,
                                                              try_web_lookup=False)
            self.assertIsInstance(lblr, LabelerGPCR)
            self.assertIsInstance(residx2conlab,list)

    def test_with_GPCR_already_instantiated(self):
        fragments = mdcfragments.get_fragments(self.geom.top)
        GPCR = LabelerGPCR(test_filenames.adrb2_human_xlsx)
        input_values = (val for val in [""])
        with mock.patch('builtins.input', lambda *x: next(input_values)):

            residx2conlab, lblr = cli._parse_consensus_option(GPCR, "GPCR",
                                                              self.geom.top,
                                                              fragments,
                                                              return_Labeler=True)
            self.assertIsInstance(lblr, LabelerGPCR)
            self.assertEqual(lblr,GPCR)
            self.assertIsInstance(residx2conlab,list)

    def test_with_GPCR_labels(self):
        fragments = mdcfragments.get_fragments(self.geom.top)
        GPCR = LabelerGPCR(test_filenames.adrb2_human_xlsx)
        GPCR_labels = GPCR.top2labels(self.geom.top)
        residx2conlab, lblr = cli._parse_consensus_option(GPCR_labels, "GPCR",
                                                              self.geom.top,
                                                              fragments,
                                                              return_Labeler=True)
        assert lblr is None
        assert residx2conlab is GPCR_labels
    def test_with_GPCR_labels_raises(self):
        fragments = mdcfragments.get_fragments(self.geom.top)
        GPCR = LabelerGPCR(test_filenames.adrb2_human_xlsx)
        GPCR_labels = GPCR.top2labels(self.geom.top)
        with self.assertRaises(AssertionError):
            residx2conlab, lblr = cli._parse_consensus_option(GPCR_labels[:20], "GPCR",
                                                              self.geom.top,
                                                              fragments,
                                                              return_Labeler=True)

    def test_no_answer(self):
        GPCR = LabelerGPCR(test_filenames.adrb2_human_xlsx)
        residx2conlab, lblr = cli._parse_consensus_option(GPCR, "GPCR",
                                                          self.geom.top,
                                                          [_np.arange(10)],
                                                          return_Labeler=True)

class Test_offer_to_create_dir(unittest.TestCase):

    def test_creates_dir(self):
       with TemporaryDirectory() as tmpdir:
           newdir = _path.join(tmpdir,"non_existent")
           input_values = (val for val in [""])
           with mock.patch('builtins.input', lambda *x: next(input_values)):
               cli._offer_to_create_dir(newdir)
           assert _path.exists(newdir)

    def test_does_nothing_bc_dir_exists(self):
       with TemporaryDirectory() as tmpdir:
           cli._offer_to_create_dir(tmpdir)

    def test_raises(self):
        input_values = (val for val in ["n"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            res = cli._offer_to_create_dir("testdir")
            assert res is None

class Test_load_any_geom(unittest.TestCase):

    def test_loads_file(self):
        geom = cli._load_any_geom(test_filenames.top_pdb)
        self.assertIsInstance(geom, md.Trajectory)

    def test_loads_geom(self):
        geom = md.load(test_filenames.top_pdb)
        geomout = cli._load_any_geom(geom)
        self.assertIs(geom, geomout)

class Test_parse_fragment_naming_options(unittest.TestCase):
    def setUp(self):
        self.fragments = [[0,1],
                          [2,3],
                          [4,5],
                          [6,7]]
    def test_empty_str(self):
        fragnames = cli._parse_fragment_naming_options("", self.fragments)
        self.assertSequenceEqual(["frag0","frag1","frag2","frag3"],
                                 fragnames)

    def test_None(self):
        fragnames = cli._parse_fragment_naming_options("None", self.fragments)
        self.assertSequenceEqual([None,None,None,None],
                                 fragnames)

    def test_list(self):
        fragnames = cli._parse_fragment_naming_options(["A","B","C","D"], self.fragments)
        self.assertSequenceEqual(["A","B","C","D"],
                                 fragnames)
    def test_csv(self):
        fragnames = cli._parse_fragment_naming_options("TM1,TM2,ICL3,H8", self.fragments)
        self.assertSequenceEqual(["TM1","TM2","ICL3","H8"],
                                 fragnames)

    def test_csv_wrong_nr_raises(self):
        with self.assertRaises(AssertionError):
            fragnames = cli._parse_fragment_naming_options("TM1,TM2", self.fragments)
            self.assertSequenceEqual(["TM1", "TM2", "ICL3", "H8"],
                                     fragnames)
    def test_danger_raises(self):
        with self.assertRaises(NotImplementedError):
            cli._parse_fragment_naming_options("TM1,danger", self.fragments)

class Test_parse_parse_coloring_options(unittest.TestCase):

    def test_none(self):
        color = cli._parse_coloring_options(None,2)
        self.assertSequenceEqual(["tab:blue","tab:blue"], color)

    def test_True(self):
        color = cli._parse_coloring_options(True,2)
        self.assertSequenceEqual(['magenta', 'yellow'], color)

    def test_string(self):
        color = cli._parse_coloring_options("salmon",2)
        self.assertSequenceEqual(['salmon', 'salmon'], color)

    def test_list(self):
        color = cli._parse_coloring_options(["pink", "salmon"],2)
        self.assertSequenceEqual(["pink", "salmon"], color)

    def test_raises(self):
        with self.assertRaises(ValueError):
            color = cli._parse_coloring_options(["pink", "salmon"],3)

class Test_color_schemes(unittest.TestCase):

    def test_scheme_P(self):
        self.assertSequenceEqual(cli._color_schemes("P"),["red", "purple", "gold", "darkorange"])

    def test_auto(self):
        self.assertEqual(len(cli._color_schemes("auto")),10)

    def test_color(self):
        self.assertEqual(cli._color_schemes("blue"),["blue"])

    def test_csv(self):
        self.assertEqual(cli._color_schemes("blue,red"),["blue","red"])


class Test_fragment_overview(unittest.TestCase):

    def test_runs(self):
        res = cli.fragment_overview(test_filenames.top_pdb)
        self.assertSequenceEqual(list(res.keys()),
                                 mdcfragments.fragments._allowed_fragment_methods)

class Test_fragment_overview_Nomenclature(unittest.TestCase):

    def test_CGN_paths(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            path_to_gnas2_human = _path.join(tmpdir, _path.basename(test_filenames.gnas2_human_xlsx))
            shutil.copy(test_filenames.gnas2_human_xlsx, tmpdir)
            with remember_cwd():
                os.chdir(tmpdir)
                a = parser_for_CGN_overview()
                a = a.parse_args([path_to_gnas2_human])
                a.__setattr__("topology",test_filenames.top_pdb)
                cli._fragment_overview(a, "CGN")

    def test_GPCR_paths_and_verbose(self):
        a = parser_for_GPCR_overview()
        a = a.parse_args([test_filenames.adrb2_human_xlsx])
        a.__setattr__("print_conlab",True)
        a.__setattr__("topology",test_filenames.top_pdb)
        cli._fragment_overview(a, "GPCR")

    def test_GPCR_url(self):
        a = parser_for_GPCR_overview()
        a = a.parse_args(["adrb2_human"])
        a.__setattr__("topology",test_filenames.pdb_3SN6)
        cli._fragment_overview(a, "GPCR")

    def test_raises(self):
        with self.assertRaises(ValueError):
            cli._fragment_overview(None, "BWx")

    def test_AAs(self):
        a = parser_for_CGN_overview()
        a = a.parse_args([test_filenames.gnas2_human_xlsx,
                          ])
        a.__setattr__("AAs","LEU394,LEU395")
        a.__setattr__("topology",test_filenames.top_pdb)
        cli._fragment_overview(a, "CGN")

    def test_labels(self):
        a = parser_for_GPCR_overview()
        a = a.parse_args([test_filenames.adrb2_human_xlsx])
        a.__setattr__("topology",test_filenames.top_pdb)
        a.__setattr__("labels","3.50")
        cli._fragment_overview(a, "GPCR")

    def test_no_top(self):
        a = parser_for_GPCR_overview()
        a = a.parse_args([test_filenames.adrb2_human_xlsx])
        a.__setattr__("labels","3.50")
        cli._fragment_overview(a, "GPCR")

class Test_compare(unittest.TestCase):

    def setUp(self):
        self.CG1 = contacts.ContactGroup([contacts.ContactPair([0,1],[[.1, .1]], [[0., 1.]]),
                                 contacts.ContactPair([0,2],[[.1, .2]], [[0., 1.]])])

        self.CG2 = contacts.ContactGroup([contacts.ContactPair([0,1],[[.1, .1, .1]], [[0., 1., 2.]]),
                                 contacts.ContactPair([0,3],[[.1, .2, .2]], [[0., 1., 2.]])])

    def test_just_works(self):
        with TemporaryDirectory(suffix='_test_mdciao') as tmpdir:
            with remember_cwd():
                os.chdir(tmpdir)
                myfig, freqs, plotted_freqs = cli.compare({"CG1": self.CG1, "CG2": self.CG2},
                                       ctc_cutoff_Ang=1.5, anchor="0")
                myfig.tight_layout()
                assert isinstance(freqs, dict)
                assert isinstance(plotted_freqs, dict)
                _plt.close("all")

if __name__ == '__main__':
    unittest.main()

class Test_residue_selection(unittest.TestCase):

    def test_works(self):
        residue_idxs, __, maps = cli.residue_selection("GLU30",test_filenames.small_monomer)

        _np.testing.assert_array_equal(residue_idxs,[0])

        self.assertDictEqual(maps, {})

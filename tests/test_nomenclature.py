import unittest
import mdtraj as md
import numpy as _np
from os import path
from tempfile import TemporaryDirectory as _TDir, mkdtemp, NamedTemporaryFile as _NamedTemporaryFile

import shutil
from urllib.error import HTTPError
from shutil import copy

import pytest
from mdciao import nomenclature
# It's a sign of bad design to have to import these private methods here
# for testing, they should be tested by the methods using them or
# made public
# Now tha API is more stable I have decided to hide most of these
# methods but keep the test
from mdciao.nomenclature import nomenclature
from mdciao.examples import filenames as test_filenames
from mdciao.utils.lists import assert_no_intersection
from mdciao import examples
from mdciao.utils.sequence import top2seq
from mdciao.utils.residue_and_atom import shorten_AA
from mdciao.fragments import get_fragments, fragment_slice

import mock

from pandas import DataFrame, read_excel


class Test_md_load_rcsb(unittest.TestCase):

    def test_works(self):
        geom = nomenclature._md_load_rcsb("3CAP",
                                          verbose=True,
                                          )
        assert isinstance(geom, md.Trajectory)

    def test_works_return_url(self):
        geom, url = nomenclature._md_load_rcsb("3CAP",
                                               # verbose=True,
                                               return_url=True
                                               )
        assert isinstance(geom, md.Trajectory)
        assert isinstance(url, str)
        assert "http" in url


class Test_PDB_finder(unittest.TestCase):

    def test_works_locally(self):
        geom, filename = nomenclature._PDB_finder(path.splitext(test_filenames.top_pdb)[0],
                                                  local_path=test_filenames.example_path,
                                                  try_web_lookup=False)
        assert isinstance(geom, md.Trajectory)
        assert isinstance(filename, str)

    def test_works_locally_pdbgz(self):
        geom, filename = nomenclature._PDB_finder("3SN6",
                                                  local_path=test_filenames.RCSB_pdb_path,
                                                  try_web_lookup=False)
        assert isinstance(geom, md.Trajectory)
        assert isinstance(filename, str)

    def test_works_online(self):
        geom, filename = nomenclature._PDB_finder("3SN6")

        assert isinstance(geom, md.Trajectory)
        assert isinstance(filename, str)
        assert "http" in filename

    def test_fails_bc_no_online_access(self):
        with pytest.raises((OSError, FileNotFoundError)):
            nomenclature._PDB_finder("3SN6",
                                     try_web_lookup=False)


class Test_CGN_finder(unittest.TestCase):

    def test_works_locally(self):
        df, filename = nomenclature._CGN_finder("3SN6",
                                                try_web_lookup=False,
                                                local_path=test_filenames.nomenclature_path)

        assert isinstance(df, DataFrame)
        assert isinstance(filename, str)
        _np.testing.assert_array_equal(list(df.keys()), ["CGN", "Sort number", "3SN6"])

    def test_works_online(self):
        df, filename = nomenclature._CGN_finder("3SN6",
                                                )

        assert isinstance(df, DataFrame)
        assert isinstance(filename, str)
        assert "http" in filename
        _np.testing.assert_array_equal(list(df.keys()), ["CGN", "Sort number", "3SN6"])

    def test_works_online_and_writes_to_disk_excel(self):
        with _TDir(suffix="_mdciao_test") as tdir:
            df, filename = nomenclature._CGN_finder("3SN6",
                                                    format="%s.xlsx",
                                                    local_path=tdir,
                                                    write_to_disk=True
                                                    )

            assert isinstance(df, DataFrame)
            assert isinstance(filename, str)
            assert "http" in filename
            _np.testing.assert_array_equal(list(df.keys()), ["CGN", "Sort number", "3SN6"])
            assert path.exists(path.join(tdir, "3SN6.xlsx"))

    def test_works_online_and_writes_to_disk_ascii(self):
        with _TDir(suffix="_mdciao_test") as tdir:
            df, filename = nomenclature._CGN_finder("3SN6",
                                                    local_path=tdir,
                                                    format="%s.txt",
                                                    write_to_disk=True
                                                    )

            assert isinstance(df, DataFrame)
            assert isinstance(filename, str)
            assert "http" in filename
            _np.testing.assert_array_equal(list(df.keys()), ["CGN", "Sort number", "3SN6"])
            assert path.exists(path.join(tdir, "3SN6.txt"))

    def test_works_local_does_not_overwrite(self):
        with _TDir(suffix="_mdciao_test") as tdir:
            infile = test_filenames.CGN_3SN6
            copy(infile, tdir)
            with pytest.raises(FileExistsError):
                nomenclature._CGN_finder("3SN6",
                                         try_web_lookup=False,
                                         local_path=tdir,
                                         write_to_disk=True
                                         )

    def test_raises_not_find_locally(self):
        with pytest.raises(FileNotFoundError):
            nomenclature._CGN_finder("3SN6",
                                     try_web_lookup=False
                                     )

    def test_not_find_locally_but_no_fail(self):
        DF, filename = nomenclature._CGN_finder("3SN6",
                                                try_web_lookup=False,
                                                dont_fail=True
                                                )
        assert DF is None
        assert isinstance(filename, str)

    def test_raises_not_find_online(self):
        with pytest.raises(HTTPError):
            nomenclature._CGN_finder("3SNw",
                                     )

    def test_not_find_online_but_no_raise(self):
        df, filename = nomenclature._CGN_finder("3SNw",
                                                dont_fail=True
                                                )
        assert df is None
        assert isinstance(filename, str)
        assert "www" in filename


class Test_GPCRmd_lookup_GPCR(unittest.TestCase):

    def test_works(self):
        DF = nomenclature._GPCR_web_lookup("https://gpcrdb.org/services/residues/extended/adrb2_human")
        assert isinstance(DF, DataFrame)

    def test_wrong_code(self):
        with pytest.raises(ValueError):
            raise nomenclature._GPCR_web_lookup("https://gpcrdb.org/services/residues/extended/adrb_beta2")


class Test_GPCR_finder(unittest.TestCase):

    def test_works_locally(self):
        df, filename = nomenclature._GPCR_finder(test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx,
                                                 try_web_lookup=False,
                                                 )

        assert isinstance(df, DataFrame)
        assert isinstance(filename, str)
        _np.testing.assert_array_equal(list(df.keys())[:3], nomenclature._GPCR_mandatory_fields)
        assert any([key in df.keys() for key in nomenclature._GPCR_mandatory_fields])  # at least one scheme

    def test_works_online(self):
        df, filename = nomenclature._GPCR_finder("adrb2_human",
                                                 )

        assert isinstance(df, DataFrame)
        assert isinstance(filename, str)
        assert "http" in filename
        _np.testing.assert_array_equal(list(df.keys())[:3], nomenclature._GPCR_mandatory_fields)
        assert any([key in df.keys() for key in nomenclature._GPCR_mandatory_fields])

    def test_raises_not_find_locally(self):
        with pytest.raises(FileNotFoundError):
            nomenclature._GPCR_finder("B2AR",
                                      try_web_lookup=False
                                      )

    def test_not_find_locally_but_no_fail(self):
        DF, filename = nomenclature._GPCR_finder("B2AR",
                                                 try_web_lookup=False,
                                                 dont_fail=True
                                                 )
        assert DF is None
        assert isinstance(filename, str)

    def test_raises_not_find_online(self):
        with pytest.raises(ValueError):
            nomenclature._GPCR_finder("B2AR",
                                      )

    def test_not_find_online_but_no_raise(self):
        df, filename = nomenclature._GPCR_finder("3SNw",
                                                 dont_fail=True
                                                 )
        assert df is None
        assert isinstance(filename, str)


class Test_table2GPCR_by_AAcode(unittest.TestCase):
    def setUp(self):
        self.file = test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx

    def test_just_works(self):
        table2GPCR = nomenclature._table2GPCR_by_AAcode(tablefile=self.file)
        self.assertDictEqual(table2GPCR,
                             {'Q26': '1.25',
                              'E27': '1.26',
                              'E62': '12.48',
                              'R63': '12.49',
                              'T66': '2.37',
                              'V67': '2.38'
                              })

    def test_keep_AA_code_test(self):  # dictionary keys will only have AA id
        table2GPCR = nomenclature._table2GPCR_by_AAcode(tablefile=self.file, keep_AA_code=False)
        self.assertDictEqual(table2GPCR,
                             {26: '1.25',
                              27: '1.26',
                              62: '12.48',
                              63: '12.49',
                              66: '2.37',
                              67: '2.38',
                              })

    def test_table2GPCR_by_AAcode_return_fragments(self):
        table2GPCR, defs = nomenclature._table2GPCR_by_AAcode(tablefile=self.file,
                                                              return_fragments=True)

        self.assertDictEqual(defs, {'TM1': ["Q26", "E27"],
                                    "ICL1": ["E62", "R63"],
                                    "TM2": ["T66", "V67"]})

    def test_table2B_by_AAcode_already_DF(self):
        from pandas import read_excel
        df = read_excel(self.file, header=0, engine="openpyxl")

        table2GPCR = nomenclature._table2GPCR_by_AAcode(tablefile=df)
        self.assertDictEqual(table2GPCR,
                             {'Q26': '1.25',
                              'E27': '1.26',
                              'E62': '12.48',
                              'R63': '12.49',
                              'T66': '2.37',
                              'V67': '2.38'
                              })


class TestClassSetUpTearDown_CGN_local(unittest.TestCase):
    # The setup is in itself a test
    def setUp(self):
        self.tmpdir = mkdtemp("_test_mdciao_CGN_local")
        self._CGN_3SN6_file = path.join(self.tmpdir, path.basename(test_filenames.CGN_3SN6))
        self._PDB_3SN6_file = path.join(self.tmpdir, path.basename(test_filenames.pdb_3SN6))
        shutil.copy(test_filenames.CGN_3SN6, self._CGN_3SN6_file)
        shutil.copy(test_filenames.pdb_3SN6, self._PDB_3SN6_file)
        self.cgn_local = nomenclature.LabelerCGN("3SN6",
                                                 try_web_lookup=False,
                                                 local_path=self.tmpdir,
                                                 )

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.tmpdir)


class TestLabelerCGN_local(TestClassSetUpTearDown_CGN_local):

    # The setup is in itself a test
    def setUp(self):
        self.tmpdir = mkdtemp("_test_mdciao_CGN_local")
        self._CGN_3SN6_file = path.join(self.tmpdir, path.basename(test_filenames.CGN_3SN6))
        self._PDB_3SN6_file = path.join(self.tmpdir, path.basename(test_filenames.pdb_3SN6))
        shutil.copy(test_filenames.CGN_3SN6, self._CGN_3SN6_file)
        shutil.copy(test_filenames.pdb_3SN6, self._PDB_3SN6_file)
        self.cgn_local = nomenclature.LabelerCGN("3SN6",
                                                 try_web_lookup=False,
                                                 local_path=self.tmpdir,
                                                 )

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.tmpdir)

    def test_correct_files(self):
        _np.testing.assert_equal(self.cgn_local.tablefile,
                                 self._CGN_3SN6_file)
        _np.testing.assert_equal(self.cgn_local.ref_PDB,
                                 "3SN6")

    def test_mdtraj_attributes(self):
        pass
        # _np.testing.assert_equal(cgn_local.geom,
        #                         self._geom_3SN6)

        # _np.testing.assert_equal(cgn_local.top,
        #                         self._geom_3SN6.top)

    def test_dataframe(self):
        self.assertIsInstance(self.cgn_local.dataframe, DataFrame)
        self.assertSequenceEqual(list(self.cgn_local.dataframe.keys()),
                                 ["CGN", "Sort number", "3SN6"])

    def test_correct_residue_dicts(self):
        _np.testing.assert_equal(self.cgn_local.conlab2AA["G.hfs2.2"], "R201")
        _np.testing.assert_equal(self.cgn_local.AA2conlab["R201"], "G.hfs2.2")

    def test_correct_fragments_dict(self):
        # Test "fragments" dictionary SMH
        self.assertIsInstance(self.cgn_local.fragments, dict)
        assert all([len(ii) > 0 for ii in self.cgn_local.fragments.values()])
        self.assertEqual(self.cgn_local.fragments["G.HN"][0], "T9")
        self.assertSequenceEqual(list(self.cgn_local.fragments.keys()),
                                 nomenclature._CGN_fragments)

    def test_correct_fragments_as_conlabs_dict(self):
        # Test "fragments_as_conslabs" dictionary SMH
        self.assertIsInstance(self.cgn_local.fragments_as_conlabs, dict)
        assert all([len(ii) > 0 for ii in self.cgn_local.fragments_as_conlabs.values()])
        self.assertEqual(self.cgn_local.fragments_as_conlabs["G.HN"][0], "G.HN.26")
        self.assertSequenceEqual(list(self.cgn_local.fragments_as_conlabs.keys()),
                                 nomenclature._CGN_fragments)

    def test_correct_fragment_names(self):
        self.assertSequenceEqual(self.cgn_local.fragment_names,
                                 list(self.cgn_local.fragments.keys()))

    def test_conlab2residx_wo_input_map(self):
        # More than anthing, this is testing _top2consensus_map
        # I know this a priori using find_AA
        out_dict = self.cgn_local.conlab2residx(self.cgn_local.top)
        self.assertEqual(out_dict["G.hfs2.2"], 164)

    def test_conlab2residx_w_input_map(self):
        # This should find R201 no problem

        map = [None for ii in range(200)]
        map[164] = "G.hfs2.2"
        out_dict = self.cgn_local.conlab2residx(self.cgn_local.top, map=map)
        self.assertEqual(out_dict["G.hfs2.2"], 164)

    def test_conlab2residx_w_input_map_duplicates(self):
        map = [None for ii in range(200)]
        map[164] = "G.hfs2.2"  # I know this a priori using find_AA
        map[165] = "G.hfs2.2"
        with pytest.raises(ValueError):
            self.cgn_local.conlab2residx(self.cgn_local.top, map=map)

    def test_top2frags_just_passes(self):
        defs = self.cgn_local.top2frags(self.cgn_local.top)
        self.assertSequenceEqual(list(defs.keys()),
                                 nomenclature._CGN_fragments)

    def test_top2frags_gets_dataframe(self):
        self.cgn_local.aligntop(self.cgn_local.top)
        defs = self.cgn_local.top2frags(self.cgn_local.top,
                                        input_dataframe=self.cgn_local.most_recent_alignment)
        self.assertSequenceEqual(list(defs.keys()),
                                 nomenclature._CGN_fragments)

    def test_top2frags_defs_are_broken_in_frags(self):
        input_values = (val for val in ["0-1"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            defs = self.cgn_local.top2frags(self.cgn_local.top,
                                            fragments=[_np.arange(0, 10),
                                                       _np.arange(10, 15),
                                                       _np.arange(15, 20)
                                                       ]
                                            )
            self.assertSequenceEqual(list(defs.keys()),
                                     nomenclature._CGN_fragments)
            _np.testing.assert_array_equal(defs["G.HN"], _np.arange(0, 15))

    def test_top2frags_defs_are_broken_in_frags_bad_input(self):
        input_values = (val for val in ["0-2"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):  # Checking against the input 1 and 1
            with pytest.raises(ValueError):
                self.cgn_local.top2frags(self.cgn_local.top,
                                         fragments=[_np.arange(0, 10),
                                                    _np.arange(10, 15),
                                                    _np.arange(15, 40)]
                                         )

    def test_fragments_as_idxs(self):
        frags_as_idsx = self.cgn_local.fragments_as_idxs
        _np.testing.assert_array_equal([len(ifrag) for ifrag in frags_as_idsx],
                                       [len(ifrag) for ifrag in self.cgn_local.fragments])
        _np.testing.assert_array_equal(list(frags_as_idsx.keys()),
                                       self.cgn_local.fragment_names)

    def test_PDB_full_path_exists(self):
        nomenclature.LabelerCGN(self._CGN_3SN6_file,
                                try_web_lookup=False,
                                local_path=self.tmpdir,
                                )

    # These tests only test it runs, not that the alignment is correct
    #  those checks are done in sequence tests
    def test_aligntop_with_self(self):
        top2self, self2top = self.cgn_local.aligntop(self.cgn_local.seq)
        self.assertDictEqual(top2self, self2top)

    def test_aligntop_with_self_residxs(self):
        top2self, self2top = self.cgn_local.aligntop(self.cgn_local.seq, restrict_to_residxs=[2, 3], min_hit_rate=0)
        self.assertDictEqual(top2self, self2top)
        self.assertTrue(all([key in [2, 3] for key in top2self.keys()]))
        self.assertTrue(all([val in [2, 3] for val in top2self.values()]))

    def test_most_recent_labels_None(self):
        assert self.cgn_local.most_recent_top2labels is None

    def test_most_recent_labels_works(self):
        labels = self.cgn_local.top2labels(self.cgn_local.top)
        self.assertListEqual(labels, self.cgn_local.most_recent_top2labels)


class TestLabelerGPCR_local_woPDB(unittest.TestCase):

    # The setup is in itself a test
    def setUp(self):
        self._geom_3SN6 = md.load(test_filenames.pdb_3SN6)
        self.tmpdir = mkdtemp("_test_mdciao_GPCR_local_no_pdb")
        self._GPCRmd_B2AR_nomenclature_test_xlsx = path.join(self.tmpdir,
                                                             path.basename(
                                                                 test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx))
        shutil.copy(test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx, self._GPCRmd_B2AR_nomenclature_test_xlsx)

        self.GPCR_local_no_pdb = nomenclature.LabelerGPCR(self._GPCRmd_B2AR_nomenclature_test_xlsx,
                                                          try_web_lookup=False)

    def test_correct_files(self):
        _np.testing.assert_equal(self.GPCR_local_no_pdb.ref_PDB,
                                 None)


class TestLabelerGPCR_local(unittest.TestCase):

    # The setup is in itself a test
    def setUp(self):
        self._geom_3SN6 = md.load(test_filenames.pdb_3SN6)
        self.tmpdir = mkdtemp("_test_mdciao_GPCR_local")
        self._PDB_3SN6_file = path.join(self.tmpdir, path.basename(test_filenames.pdb_3SN6))
        self._GPCRmd_B2AR_nomenclature_test_xlsx = path.join(self.tmpdir, path.basename(
            test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx))
        shutil.copy(test_filenames.pdb_3SN6, self._PDB_3SN6_file)
        shutil.copy(test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx, self._GPCRmd_B2AR_nomenclature_test_xlsx)
        self.GPCR_local_w_pdb = nomenclature.LabelerGPCR(self._GPCRmd_B2AR_nomenclature_test_xlsx,
                                                         ref_PDB="3SN6",
                                                         try_web_lookup=False,
                                                         local_path=self.tmpdir,
                                                         )
        # Check the excel and construct this
        self.conlab_frag_dicts = {"BW":
                                      {'TM1': ['1.25', '1.26'],
                                       'ICL1': ['12.48', '12.49'],
                                       'TM2': ['2.37', '2.38']},
                                  "display_generic_number":
                                      {'TM1': ['1.25x25', '1.26x26'],
                                       'ICL1': ['12.48x48', '12.49x49'],
                                       'TM2': ['2.37x37', '2.38x38']}
                                  }

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.tmpdir)

    def test_correct_files(self):
        _np.testing.assert_equal(self.GPCR_local_w_pdb.tablefile,
                                 self._GPCRmd_B2AR_nomenclature_test_xlsx)
        _np.testing.assert_equal(self.GPCR_local_w_pdb.ref_PDB,
                                 "3SN6")

    def test_mdtraj_attributes(self):
        pass
        # _np.testing.assert_equal(cgn_local.geom,
        #                         self._geom_3SN6)

        # _np.testing.assert_equal(cgn_local.top,
        #                         self._geom_3SN6.top)

    def test_dataframe(self):
        self.assertIsInstance(self.GPCR_local_w_pdb.dataframe, DataFrame)
        self.assertSequenceEqual(list(self.GPCR_local_w_pdb.dataframe.keys()),
                                 nomenclature._GPCR_mandatory_fields + nomenclature._GPCR_available_schemes)

    def test_correct_residue_dicts(self):
        if self.GPCR_local_w_pdb._nomenclature_key == "BW":
            _np.testing.assert_equal(self.GPCR_local_w_pdb.conlab2AA["1.25"], "Q26")
            _np.testing.assert_equal(self.GPCR_local_w_pdb.AA2conlab["Q26"], "1.25")
        elif self.GPCR_local_w_pdb._nomenclature_key == "display_generic_number":
            _np.testing.assert_equal(self.GPCR_local_w_pdb.conlab2AA["1.25x25"], "Q26")
            _np.testing.assert_equal(self.GPCR_local_w_pdb.AA2conlab["Q26"], "1.25x25")
        else:
            raise ValueError("no tests written for %s yet" % (self.GPCR_local_w_pdb._nomenclature_key))

    def test_correct_fragments_dict(self):
        # Test "fragments" dictionary SMH
        self.assertIsInstance(self.GPCR_local_w_pdb.fragments, dict)
        assert all([len(ii) > 0 for ii in self.GPCR_local_w_pdb.fragments.values()])
        self.assertEqual(self.GPCR_local_w_pdb.fragments["ICL1"][0], "E62")
        self.assertSequenceEqual(list(self.GPCR_local_w_pdb.fragments.keys()),
                                 ["TM1", "ICL1", "TM2"])

    def test_correct_fragments_as_conlabs_dict(self):
        # Test "fragments_as_conslabs" dictionary SMH
        self.assertIsInstance(self.GPCR_local_w_pdb.fragments_as_conlabs, dict)
        assert all([len(ii) > 0 for ii in self.GPCR_local_w_pdb.fragments_as_conlabs.values()])
        self.assertSequenceEqual(list(self.GPCR_local_w_pdb.fragments_as_conlabs.keys()),
                                 ["TM1", "ICL1", "TM2"])
        self.assertDictEqual(self.GPCR_local_w_pdb.fragments_as_conlabs,
                             self.conlab_frag_dicts[self.GPCR_local_w_pdb._nomenclature_key])

    def test_correct_fragment_names(self):
        self.assertSequenceEqual(self.GPCR_local_w_pdb.fragment_names,
                                 list(self.GPCR_local_w_pdb.fragments.keys()))

    def test_fragments_as_idxs(self):
        frags_as_idsx = self.GPCR_local_w_pdb.fragments_as_idxs
        self.assertSequenceEqual([len(ifrag) for ifrag in frags_as_idsx],
                                 [len(ifrag) for ifrag in self.GPCR_local_w_pdb.fragments])
        self.assertSequenceEqual(list(frags_as_idsx.keys()),
                                 self.GPCR_local_w_pdb.fragment_names)

    # These tests only test it runs, not that the alignment is correct
    #  those checks are done in sequence tests
    def test_aligntop_with_self(self):
        top2self, self2top = self.GPCR_local_w_pdb.aligntop(self.GPCR_local_w_pdb.seq)
        self.assertDictEqual(top2self, self2top)
        self.assertIsInstance(self.GPCR_local_w_pdb.most_recent_alignment, DataFrame)

    def test_aligntop_with_self_residxs(self):
        top2self, self2top = self.GPCR_local_w_pdb.aligntop(self.GPCR_local_w_pdb.seq, restrict_to_residxs=[2, 3], min_hit_rate=0)
        self.assertDictEqual(top2self, self2top)
        self.assertTrue(all([key in [2, 3] for key in top2self.keys()]))
        self.assertTrue(all([val in [2, 3] for val in top2self.values()]))

    def test_uniprot_name(self):
        self.assertEqual(self.GPCR_local_w_pdb.uniprot_name, self._GPCRmd_B2AR_nomenclature_test_xlsx)

class Test_aligntop_full(unittest.TestCase):
    # Has to be done with full GPCR nomencl, not with small one

    def setUp(self):
        self.GPCR = examples.GPCRLabeler_ardb2_human()
        self.geom = md.load(examples.filenames.pdb_3SN6)
        self.frags = get_fragments(self.geom.top) # receptor is idx 4
        self.anchors = {"1.50x50": 'N51',  # Anchor residues
                        "2.50x50": 'D79',
                        "3.50x50": 'R131',
                        "4.50x50": 'W158',
                        "5.50x50": 'P211',
                        "6.50x50": 'P288',
                        "7.50x50": 'P323',
                        "8.50x50": 'F332'}
        self.right_confrags = self.GPCR.top2frags(self.geom.top)
    def check_anchors(self, self2top):
        for clab, aa in self.anchors.items():
            assert self.GPCR.conlab2AA[clab] == aa # Assert the above anchor-dict holds before doing anything
            # Grab the dataframe row-indices which are used by top2self and self2top
            row_idx = self.GPCR.conlab2idx[clab]
            # Make the move from the dataframe to the top via the map
            top_idx = self2top[row_idx]
            # Grab the residue from the top
            top_res = self.geom.top.residue(top_idx)
            # Check that it's actually the anchor residue
            assert aa == shorten_AA(top_res, keep_index=True)



    def test_default_str_w_top(self):
        top2self, self2top =self.GPCR.aligntop(self.geom.top)
        _np.testing.assert_array_equal(self.frags[4],list(top2self.keys()))
        # The right fragment has been identified when the topology can be segmented

    def test_default_str_w_seq(self):
        top2self, self2top =self.GPCR.aligntop(top2seq(self.geom.top))
        # Necessarily, some mismatches will appear in this alignment, because
        # the receptor in 3SN6 starts at GLU30, all other residues of the N-term
        # before that have been aligned in small chunks in the ca. 1000 residues
        # of the G-protein in 3SN6 up to the GLU30 of the B2AR
        # The check here is to check that the most conserved regions are there
        self.check_anchors(self2top)

    def test_default_None_w_top(self):
        top2self, self2top =self.GPCR.aligntop(self.geom.top, fragments=None)
        _np.testing.assert_array_equal(self.frags[4],list(top2self.keys()))
        # The right fragment has been identified when the topology can be segmented

    def test_default_None_w_seq(self):
        top2self, self2top =self.GPCR.aligntop(top2seq(self.geom.top), fragments=None)
        self.check_anchors(self2top)

    def test_default_False(self):
        top2self, self2top =self.GPCR.aligntop(self.geom.top, fragments=False)
        self.check_anchors(self2top)

    def test_explict_definition_missing_frags(self):
        # This is the hardest test, we input an existing fragment definition
        # which doesn't cover all consensus parts, but only the tip of TM6
        # we still recover them via the mix-fragments
        fragments = self.right_confrags["TM6"][-10:]
        top2self, self2top =self.GPCR.aligntop(self.geom.top, fragments=[fragments])
        _np.testing.assert_array_equal(self.frags[4],list(top2self.keys()))

class Test_aligntop_fragment_clashes(unittest.TestCase):
    """
    We're testing some aligntop cases through the top2frags results, which are easier to check
    """

    def setUp(self):
        self.CGN = examples.CGNLabeler_3SN6()
        self.GPCR = examples.GPCRLabeler_ardb2_human()
        self.b2ar = md.load(examples.filenames.actor_pdb)

    def test_first_aligment_collides_with_resSeq(self):
        """

        There's two alignments sharing the same score of 276

        In the first alignment, TM5 and TM6 are defined as
        fragment    TM5 with     36 AAs   ASN196           (   166) -   GLN231           (201   ) (TM5)
        fragment    TM6 with     37 AAs   LYS232           (   202) -   GLN299           (238   ) (TM6)  resSeq jumps


        in which TM6 is broken, because the receptor resSeqs -fragments are

        fragment      0 with    203 AAs    GLU30           (     0) -   LYS232           (202   ) (0)
        fragment      1 with     78 AAs   PHE264           (   203) -  CYSP341           (280   ) (1)

        In the second alignment, TM5 and TM6 are defined as

        fragment    TM5 with     37 AAs   ASN196           (   166) -   LYS232           (202   ) (TM5)
        fragment    TM6 with     36 AAs   PHE264           (   203) -   GLN299           (238   ) (TM6)

        Which is correct with LYS232 being the last TM5 res (second alignment) and not the first of TM6 (first alignment)


        """
        frags = self.GPCR.top2frags(self.b2ar.top)

        self.assertListEqual(frags["TM5"], _np.arange(166, 202 + 1).tolist())
        self.assertListEqual(frags["TM6"], _np.arange(203, 238+1).tolist())

    def test_alignment_collides_with_but_frag_is_compatible(self):

        frags = get_fragments(self.b2ar.top)
        just_gprot = fragment_slice(self.b2ar, frags, [1])
        broken_gprot = just_gprot.atom_slice(just_gprot.top.select("residue != 220"))

        """
        
        The first alignment yields:
        
        G.s2s3 with      2 AAs   ASP215@G.s2s3.1  (   199) -   LYS216@G.s2s3.2  (200   ) (G.s2s3) 
          G.S3 with      7 AAs   VAL217@G.S3.1    (   201) -   VAL224@G.S3.8    (207   ) (G.S3)  resSeq jumps
    
        in which G.S3 is broken, because the broken_gprot fragments are like this as resSeq
        
        fragment      0 with     69 AAs    CYSP2           (     0) -    GLY70           (68    ) (0) 
        fragment      1 with    135 AAs    ASP85           (    69) -   PHE219           (203   ) (1) 
        fragment      2 with    174 AAs   MET221           (   204) -   LEU394           (377   ) (2)
        
        [Note, this forced situation is more common, it's just we hadn't encountered it yet] 

        However, even if G.S3 is broken across resSeq, the found G.S3 fits with the G.S3 defined in 
        the reference sequence, since:
        
        top residues ['V217', 'N218', 'F219',         'M221', 'F222', 'D223', 'V224']
        ref residues ['V217', 'N218', 'F219', 'H220', 'M221', 'F222', 'D223', 'V224']

        The missing H220 has broken the Gprot (in the resSeq) heuristic s.t. G.S3 will always be 
        "broken" across fragments, _in all optimal aligments_, but that is just because the
        check_if_subfragment check, with the resSeq heuristic, is really hard to pass.
        
        However, resSeq seems to be the right heuristic to pick here, because it's better to 
        
        * break all things as much as possible first
        * patch some of them together later via seq-align if we have that information 

        """
        frags = self.CGN.top2frags(broken_gprot.top)

        self.assertListEqual(frags["G.S3"], _np.arange(201, 207 + 1).tolist())

class Test_choose_between_consensus_dicts(unittest.TestCase):

    def test_works(self):
        str = nomenclature.choose_between_consensus_dicts(1,
                                                          [{1: "BW1"},
                                                           {1: None}])
        assert str == "BW1"

    def test_not_found(self):
        str = nomenclature.choose_between_consensus_dicts(1,
                                                          [{1: None},
                                                           {1: None}],
                                                          no_key="NAtest")
        assert str == "NAtest"

    def test_raises(self):
        with pytest.raises(AssertionError):
            nomenclature.choose_between_consensus_dicts(1,
                                                        [{1: "BW1"},
                                                         {1: "CGN1"}],
                                                        )


class Test_map2defs(unittest.TestCase):
    def setUp(self):
        self.cons_list = ['3.67', 'G.H5.1', 'G.H5.6', '5.69']
        self.cons_list_w_Nones = ['3.67', None, None, 'G.H5.1', 'G.H5.6', '5.69']
        self.cons_list_wo_dots = ['367', None, None, 'G.H5.1', 'G.H5.6', '5.69']

    def test_works(self):
        map2defs = nomenclature._map2defs(self.cons_list)
        assert _np.array_equal(map2defs['3'], [0])
        assert _np.array_equal(map2defs['G.H5'], [1, 2])
        assert _np.array_equal(map2defs['5'], [3])
        _np.testing.assert_equal(len(map2defs), 3)

    def test_works_w_Nones(self):
        map2defs = nomenclature._map2defs(self.cons_list_w_Nones)
        assert _np.array_equal(map2defs['3'], [0])
        assert _np.array_equal(map2defs['G.H5'], [3, 4])
        assert _np.array_equal(map2defs['5'], [5])
        _np.testing.assert_equal(len(map2defs), 3)

    def test_works_wo_dot_raises(self):
        with pytest.raises(AssertionError):
            nomenclature._map2defs(self.cons_list_wo_dots)


class Test_fill_CGN_gaps(unittest.TestCase):
    def setUp(self):
        self.top_3SN6 = md.load(test_filenames.pdb_3SN6).top
        self.top_mut = md.load(test_filenames.pdb_3SN6_mut).top
        self.cons_list_out = ['G.HN.26', 'G.HN.27', 'G.HN.28', 'G.HN.29', 'G.HN.30']
        self.cons_list_in = ['G.HN.26', None, 'G.HN.28', 'G.HN.29', 'G.HN.30']

    def test_fill_CGN_gaps_just_works_with_CGN(self):
        fill_cgn = nomenclature._fill_consensus_gaps(self.cons_list_in, self.top_mut, verbose=True)
        self.assertEqual(fill_cgn, self.cons_list_out)


class Test_fill_consensus_gaps(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.top_pdb)
        self.cons_list_in = ['3.46', '3.47', "3.48", None,
                             None, '3.51', '3.52']
        self.cons_list_out = ['3.46', '3.47', "3.48", "3.49",
                              "3.50", '3.51', '3.52']

    def test_fill_CGN_gaps_just_works_with_GPCR(self):
        fill_cgn = nomenclature._fill_consensus_gaps(self.cons_list_in, self.geom.top, verbose=True)
        self.assertEqual(fill_cgn, self.cons_list_out)


class Test_guess_by_nomenclature(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        GPCRlabs_file = path.relpath(test_filenames.adrb2_human_xlsx, test_filenames.RCSB_pdb_path)

        cls.GPCR_local_w_pdb = nomenclature.LabelerGPCR(GPCRlabs_file,
                                                        ref_PDB="3SN6",
                                                        local_path=test_filenames.RCSB_pdb_path,
                                                        format="%s",
                                                        )
        cls.fragments = get_fragments(cls.GPCR_local_w_pdb.top)

    def test_works_on_enter(self):
        import mock
        input_values = (val for val in [""])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            answer = nomenclature.guess_by_nomenclature(self.GPCR_local_w_pdb,
                                                        self.GPCR_local_w_pdb.top,
                                                        self.fragments,
                                                        "GPCR")
            self.assertEqual(answer, "4")

    def test_works_return_answer_as_list(self):
        import mock
        input_values = (val for val in [""])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            answer = nomenclature.guess_by_nomenclature(self.GPCR_local_w_pdb,
                                                        self.GPCR_local_w_pdb.top,
                                                        self.fragments,
                                                        "GPCR",
                                                        return_str=False,
                                                        )
            self.assertSequenceEqual(answer, [4])

    def test_works_return_guess(self):
        answer = nomenclature.guess_by_nomenclature(self.GPCR_local_w_pdb,
                                                    self.GPCR_local_w_pdb.top,
                                                    self.fragments,
                                                    "GPCR",
                                                    accept_guess=True
                                                    )
        self.assertEqual(answer, "4")

    def test_works_return_None(self):
        answer = nomenclature.guess_by_nomenclature(self.GPCR_local_w_pdb,
                                                    self.GPCR_local_w_pdb.top,
                                                    self.fragments,
                                                    "GPCR",
                                                    accept_guess=True,
                                                    min_hit_rate=2,  # impossible rate
                                                    )
        self.assertEqual(answer, None)


class Test_guess_nomenclature_fragments(unittest.TestCase):
    # The setup is in itself a test
    @classmethod
    def setUpClass(cls):
        GPCRlabs_file = path.relpath(test_filenames.adrb2_human_xlsx, test_filenames.RCSB_pdb_path)

        cls.GPCR_local_w_pdb = nomenclature.LabelerGPCR(GPCRlabs_file,
                                                        ref_PDB="3SN6",
                                                        local_path=test_filenames.RCSB_pdb_path,
                                                        format="%s",
                                                        )
        cls.fragments = get_fragments(cls.GPCR_local_w_pdb.top, verbose=False)

    def test_finds_frags(self):
        guessed_frags = nomenclature.guess_nomenclature_fragments(self.GPCR_local_w_pdb,
                                                                  self.GPCR_local_w_pdb.top,
                                                                  fragments=self.fragments,
                                                                  verbose=True,
                                                                  )
        _np.testing.assert_array_equal([4], guessed_frags)

    def test_finds_frags_res(self):
        guessed_res = nomenclature.guess_nomenclature_fragments(self.GPCR_local_w_pdb,
                                                                self.GPCR_local_w_pdb.top,
                                                                fragments=self.fragments,
                                                                return_residue_idxs=True
                                                                )
        _np.testing.assert_array_equal(self.fragments[4], guessed_res)

    def test_finds_frags_no_frags(self):
        guessed_frags = nomenclature.guess_nomenclature_fragments(self.GPCR_local_w_pdb,
                                                                  self.GPCR_local_w_pdb.top,
                                                                  )
        _np.testing.assert_array_equal([4], guessed_frags)

    def test_finds_frags_seq_as_str(self):
        guessed_frags = nomenclature.guess_nomenclature_fragments(self.GPCR_local_w_pdb.seq,
                                                                  self.GPCR_local_w_pdb.top,
                                                                  fragments=self.fragments,
                                                                  )
        _np.testing.assert_array_equal([4], guessed_frags)

    def test_finds_frags_nothing_None(self):
        seq = "THISSENTENCEWILLNEVERALIGN"
        guessed_frags = nomenclature.guess_nomenclature_fragments(seq,
                                                                  self.GPCR_local_w_pdb.top,
                                                                  fragments=self.fragments,
                                                                  empty=None
                                                                  )
        print(guessed_frags)
        assert guessed_frags is None


if __name__ == '__main__':
    unittest.main()


class Test_sort_consensus_labels(unittest.TestCase):

    def setUp(self):
        self.tosort = ["G.H1.10", "H.HA.20", "H8.10", "V34", "H8.1", "3.50", "2.50", "G.H1.1", "H.HA.10"]

    def test_GPCR(self):
        sorted = nomenclature._sort_GPCR_consensus_labels(self.tosort)
        _np.testing.assert_array_equal(
            ["2.50", "3.50", "H8.1", "H8.10", "G.H1.10", "H.HA.20", "V34", "G.H1.1", "H.HA.10"],
            sorted)

    def test_GPCR_dont_append(self):
        sorted = nomenclature._sort_GPCR_consensus_labels(self.tosort, append_diffset=False)
        _np.testing.assert_array_equal(["2.50", "3.50", "H8.1", "H8.10"],
                                       sorted)

    def test_CGN(self):
        sorted = nomenclature._sort_CGN_consensus_labels(self.tosort)
        _np.testing.assert_array_equal(
            ["G.H1.1", "G.H1.10", "H.HA.10", "H.HA.20", "H8.10", "V34", "H8.1", "3.50", "2.50"],
            sorted)

    def test_CGN_dont_append(self):
        sorted = nomenclature._sort_CGN_consensus_labels(self.tosort, append_diffset=False)
        _np.testing.assert_array_equal(["G.H1.1", "G.H1.10", "H.HA.10", "H.HA.20"],
                                       sorted)


class Test_compatible_consensus_fragments(TestClassSetUpTearDown_CGN_local):

    def setUp(self):
        super(Test_compatible_consensus_fragments, self).setUp()
        self.top = md.load(test_filenames.actor_pdb).top

    def test_works(self):
        # Obtain the full objects first
        full_map = self.cgn_local.top2labels(self.top,
                                             autofill_consensus=True,
                                             # verbose=True
                                             )

        frag_defs = self.cgn_local.top2frags(self.top,
                                             verbose=False,
                                             # show_alignment=True,
                                             # map_conlab=full_map
                                             )
        idxs_to_restrict_to = frag_defs["G.H5"]
        incomplete_map = [full_map[idx] if idx in idxs_to_restrict_to else None for idx in range(self.top.n_residues)]

        # Reconstruct the full definitions from the cgn_local object
        reconstructed_defs = nomenclature.compatible_consensus_fragments(self.top,
                                                                         [incomplete_map],
                                                                         [self.cgn_local],
                                                                         autofill_consensus=False)

        self.assertDictEqual(frag_defs, reconstructed_defs)


class Test_conslabel2fraglabel(unittest.TestCase):

    def test_just_works(self):
        self.assertEqual("TM3", nomenclature._conslabel2fraglabel("GLU30@3.50"))


class Test_alignment_df2_conslist(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.list_of_dicts = [
            {"idx_0": 0,
             "idx_1": 0,
             "match": True,
             "AA_0": "GLU",
             "AA_1": "GLU",
             "conlab": "3.50",
             },
            {"idx_0": 1,
             "idx_1": 1,
             "match": False,
             "AA_0": "LYS",
             "AA_1": "ARG",
             "conlab": "3.51",
             },
            {"idx_0": 2,
             "idx_1": 2,
             "match": True,
             "AA_0": "PHE",
             "AA_1": "PHE",
             "conlab": "3.52",
             },
        ]
        cls.df = DataFrame(cls.list_of_dicts)
        # cls.consensus_dict = {"GLU0": "3.50",
        #                      "ARG1": "3.51",
        #                      "PHE2": "3.52"}

    def test_works(self):
        out_list = nomenclature._alignment_df2_conslist(self.df)
        self.assertListEqual(out_list, ["3.50", None, "3.52"])

    def test_works_nonmatch(self):
        out_list = nomenclature._alignment_df2_conslist(self.df, allow_nonmatch=True)
        self.assertListEqual(out_list, ["3.50", "3.51", "3.52"])


class Test_consensus_maps2consensus_frag(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.CGN = nomenclature.LabelerCGN(examples.filenames.CGN_3SN6)
        cls.geom = md.load(examples.filenames.actor_pdb)
        cls.GPCR, cls.maps, cls.frags = {}, {}, {}
        for GPCR_scheme in ["BW", "GPCRdb(A)", "GPCRdb(B)"]:
            cls.GPCR[GPCR_scheme] = nomenclature.LabelerGPCR(examples.filenames.adrb2_human_xlsx,
                                                             GPCR_scheme=GPCR_scheme
                                                             )

            cls.maps[GPCR_scheme] = [lab.top2labels(cls.geom.top) for lab in [cls.CGN, cls.GPCR[GPCR_scheme]]]
            cls.frags[GPCR_scheme] = [lab.top2frags(cls.geom.top) for lab in [cls.CGN, cls.GPCR[
                GPCR_scheme]]]  # This method doesn't rely on   nomenclature._consensus_maps2consensus_frags

    def test_works_on_empty(self):
        maps, frags = nomenclature._consensus_maps2consensus_frags(self.geom.top, [], verbose=True)
        assert maps == []
        assert frags == {}

    def test_works_on_maps(self):
        for imaps in self.maps.values():
            maps, frags = nomenclature._consensus_maps2consensus_frags(self.geom.top, imaps, verbose=True)
            self.assertListEqual(maps, imaps)
            assert frags == {}

    def test_works_on_Labelers(self):
        for GPCR_scheme, GPCR in self.GPCR.items():
            maps, frags = nomenclature._consensus_maps2consensus_frags(self.geom.top, [self.CGN, GPCR], verbose=True)
            self.assertListEqual(maps, self.maps[GPCR_scheme])
            for ifrags in self.frags[GPCR_scheme]:
                for key, val in ifrags.items():
                    self.assertListEqual(frags[key], val)

    def test_works_on_mix(self):
        for GPCR_scheme, GPCR in self.GPCR.items():
            maps, frags = nomenclature._consensus_maps2consensus_frags(self.geom.top, [self.maps[GPCR_scheme][0], GPCR],
                                                                       verbose=True)
            self.assertListEqual(maps, self.maps[GPCR_scheme])
            self.assertDictEqual(frags, self.frags[GPCR_scheme][1])


class Test_UniProtACtoPDBs(unittest.TestCase):

    def test_just_works(self):
        result = nomenclature._UniProtACtoPDBs("P31751")
        assert isinstance(result, dict)
        assert "3e88".upper() in result.keys()
        self.assertDictEqual(result["3E88"],
                             {'database': 'PDB',
                              'id': '3E88',
                              'properties': [{'key': 'Method', 'value': 'X-ray'},
                                             {'key': 'Resolution', 'value': '2.50 A'},
                                             {'key': 'Chains', 'value': 'A/B=146-480'}]},
                             )


class Test_mdTopology2DF(unittest.TestCase):
    def test_just_works(self):
        top = md.load(examples.filenames.small_monomer).top
        self.assertListEqual(['   serial_index residue  code  Sequence_Index AAresSeq  chain_index',
                              "0             0     GLU     E              30      E30            0",
                              "1             1     VAL     V              31      V31            0",
                              "2             2     TRP     W              32      W32            0",
                              "3             3     ILE     I              26      I26            1",
                              "4             4     GLU     E              27      E27            1",
                              "5             5     LYS     K              29      K29            1",
                              "6             6     P0G  None             381     X381            2",
                              "7             7     GDP  None             382     X382            2"],
                             nomenclature._mdTopology2residueDF(top).to_string().splitlines())


class Test_residx_from_UniProtPDBEntry_and_top(unittest.TestCase):
    def test_just_works(self):
        # nomenclature._UniProtACtoPDBs("P54311")["3SN6"]
        # Beta sub-unit in 3SN6
        PDBentry = {'database': 'PDB',
                    'id': '3SN6',
                    'properties': [{'key': 'Method', 'value': 'X-ray'},
                                   {'key': 'Resolution', 'value': '3.20 A'},
                                   {'key': 'Chains',
                                    'value': 'B=2-340'}]}  # <---- For some reason GLN1 is not considered in this entry as belonging to the beta sub-unit
        top = md.load(examples.filenames.pdb_3SN6).top
        res_idxs = nomenclature._residx_from_UniProtPDBEntry_and_top(PDBentry, top)

        """
        mdciao.fragments.get_fragments(mdciao.examples.filenames.pdb_3SN6);
        Auto-detected fragments with method 'lig_resSeq+'
        fragment      0 with    349 AAs     THR9 (     0) -   LEU394 (348   ) (0)  resSeq jumps
        fragment      1 with    340 AAs     GLN1 (   349) -   ASN340 (688   ) (1) 
        fragment      2 with     58 AAs     ASN5 (   689) -    ARG62 (746   ) (2) 
        fragment      3 with    159 AAs  ASN1002 (   747) -  ALA1160 (905   ) (3) 
        fragment      4 with    284 AAs    GLU30 (   906) -   CYS341 (1189  ) (4)  resSeq jumps
        fragment      5 with    128 AAs     GLN1 (  1190) -   SER128 (1317  ) (5) 
        fragment      6 with      1 AAs  P0G1601 (  1318) -  P0G1601 (1318  ) (6) 
        """

        self.assertListEqual(res_idxs, _np.arange(349 + 1, 688 + 1).tolist())


class Test_KLIFSDataFrame(unittest.TestCase):

    def setUp(self):
        self.df = nomenclature._read_excel_as_KDF(test_filenames.KLIFS_P31751_xlsx)
        self.geom = md.load(test_filenames.pdb_3E8D)

    def test_just_works(self):
        assert self.df.PDB_id == "3E8D"
        assert self.df.UniProtAC == "P31751"
        assert self.df.PDB_geom == self.geom

    def test_write_to_excel(self):
        with _NamedTemporaryFile(suffix=".xlsx") as f:
            self.df.to_excel(f.name)
            df_dict = read_excel(f.name, None)
            assert len(df_dict) == 5
            assert isinstance(df_dict["P31751_3E8D"], DataFrame)
            assert nomenclature._Spreadsheets2mdTrajectory(df_dict)==self.geom


class Test_KLIFS_web_lookup(unittest.TestCase):

    def test_just_works(self):
        KLIFS_df = nomenclature._KLIFS_web_lookup("P31751")
        assert isinstance(KLIFS_df, nomenclature._KLIFSDataFrame)

    def test_wrong(self):
        KLIFS_df = nomenclature._KLIFS_web_lookup("P3175111")
        assert isinstance(KLIFS_df, ValueError)

class Test_read_excel_as_KDF(unittest.TestCase):

    def test_just_works(self):
        df = nomenclature._read_excel_as_KDF(test_filenames.KLIFS_P31751_xlsx)
        assert isinstance(df, nomenclature._KLIFSDataFrame)

        geom = md.load(test_filenames.pdb_3E8D)
        assert geom == df.PDB_geom


class Test_KLIFS_finder(unittest.TestCase):
    def setUp(self):
        # This acts as test_find_online
        self.UniProtAC = "P31751"
        self.KLIFS_df = nomenclature._KLIFS_finder(self.UniProtAC)[0]
        self.geom = md.load(test_filenames.pdb_3E8D)

    def test_finds_online(self):
        assert isinstance(self.KLIFS_df, nomenclature._KLIFSDataFrame)
        assert self.KLIFS_df.PDB_id == "3E8D"
        assert self.KLIFS_df.UniProtAC == "P31751"
        assert self.KLIFS_df.PDB_geom == self.geom

    def test_finds_local_with_uniprot(self):
        df, filename = nomenclature._KLIFS_finder(self.UniProtAC, try_web_lookup=False,
                                                  local_path=test_filenames.nomenclature_path)
        assert df.PDB_id == self.KLIFS_df.PDB_id
        assert df.UniProtAC == self.KLIFS_df.UniProtAC
        assert df.PDB_geom == self.geom

    def test_finds_local_with_explicit_filename(self):
        with _NamedTemporaryFile(suffix=".xslxs") as f:
        #with _TDir(suffix="_mdciao_test") as tdir:
            copy(test_filenames.KLIFS_P31751_xlsx, f.name)
            #full_path_local_filename = path.join(tdir, "very_specific.xlsx")
            #self.KLIFS_df.to_excel(full_path_local_filename)
            df, filename = nomenclature._KLIFS_finder(f.name)
            assert df.PDB_id == self.KLIFS_df.PDB_id
            assert df.PDB_id == self.KLIFS_df.PDB_id
            assert df.UniProtAC == self.KLIFS_df.UniProtAC
            assert df.PDB_geom == self.geom


class TestLabelerKLIFS(unittest.TestCase):

    # The setup is in itself a test
    def setUp(self):
        self.KLIFS = nomenclature.LabelerKLIFS(test_filenames.KLIFS_P31751_xlsx,
                                               local_path=test_filenames.RCSB_pdb_path,
                                               try_web_lookup=False)


    def test_setup_localKLIFS_webPDB(self):
        KLIFS = nomenclature.LabelerKLIFS(test_filenames.KLIFS_P31751_xlsx,
                                               )
    # Only code specific to this object
    def test_fragments_as_idxs(self):
        assert_no_intersection(list(self.KLIFS.fragments_as_idxs.values()))
        self.assertDictEqual(self.KLIFS.fragments_as_idxs,
                             {'I': [10, 11, 12],
                              'g.l': [13, 14, 15, 16, 17, 18],
                              'II': [19, 20, 21, 22],
                              'III': [32, 33, 34, 35, 36, 37],
                              'αC': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
                              'b.l': [61, 62, 64, 65, 66, 67, 68],
                              'IV': [69, 70, 71, 72],
                              'V': [80, 81, 82],
                              'GK': [83],
                              'hinge': [84, 85, 86],
                              'linker': [87, 88, 89, 90],
                              'αD': [91, 92, 93, 94, 95, 96, 97],
                              'αE': [119, 120, 121, 122, 123],
                              'VI': [124, 125, 126],
                              'c.l': [127, 128, 129, 130, 131, 132, 133, 134],
                              'VII': [135, 136, 137],
                              'VIII': [145],
                              'xDFG': [146, 147, 148, 149],
                              'a.l': [150, 151]})

    def test_KLIFs_fragmentation(self):
        self.geom = md.load(test_filenames.pdb_3E8D)

        frags = self.KLIFS.top2frags(self.geom.top)

        # The pdb of geom and the pdb of self.KLIFS is the same pdb
        self.assertDictEqual(frags, self.KLIFS.fragments_as_idxs)

class Test_mdTrajectory_and_spreadsheets(unittest.TestCase):

    def test_reading_and_writing_works(self):
        for pdb in [examples.filenames.pdb_3SN6,
                    examples.filenames.pdb_1U19,
                    examples.filenames.pdb_3CAP,
                    examples.filenames.pdb_3E8D]:
            geom = md.load(pdb)
            with _NamedTemporaryFile(suffix="_pdb_as.xlsx") as f:
                nomenclature._mdTrajectory2spreadsheets(geom,f.name)
                read_pdb = nomenclature._Spreadsheets2mdTrajectory(f.name)
                assert geom == read_pdb


class Test_AlignerConsensus(unittest.TestCase):

    def setUp(self):

        self.geom_3SN6 = md.load(test_filenames.pdb_3SN6)
        self.geom_3CAP = md.load(test_filenames.pdb_3CAP)
        self.geom_1U19 = md.load(test_filenames.pdb_1U19)

        self.CL_adrb2_human = nomenclature.LabelerGPCR(test_filenames.adrb2_human_xlsx)
        self.CL_opsd_bovin = nomenclature.LabelerGPCR("opsd_bovin")

        self.maps_3SN6 = self.CL_adrb2_human.top2labels(self.geom_3SN6.top)
        self.maps_3CAP = self.CL_opsd_bovin.top2labels(self.geom_3CAP.top)
        self.maps_1U19 = self.CL_opsd_bovin.top2labels(self.geom_1U19.top)

    def test_with_maps(self):
        AC = nomenclature.AlignerConsensus(tops={"3CAP": self.geom_3CAP.top,
                                                 "3SN6": self.geom_3SN6.top
                                                 },
                                           maps={"3SN6": self.maps_3SN6,
                                                 "3CAP": self.maps_3CAP
                                                 })
        self.assertListEqual(AC.keys, ["3CAP", "3SN6"])
        self.assertIsInstance(AC.residxs, DataFrame)
        self.assertIsInstance(AC.AAresSeq, DataFrame)
        self.assertIsInstance(AC.CAidxs, DataFrame)
        self.assertListEqual(list(AC.residxs.keys()), ["consensus"]+AC.keys)
        self.assertListEqual(list(AC.AAresSeq.keys()), ["consensus"]+AC.keys)
        self.assertListEqual(list(AC.CAidxs.keys()), ["consensus"]+AC.keys)


        matches = AC.residxs_match(patterns="3.5*")
        self.assertEqual(matches.to_string(),
                        "    consensus  3CAP  3SN6\n"
                        "102   3.50x50   134  1007\n"
                        "103   3.51x51   135  1008\n"
                        "104   3.52x52   136  1009\n"
                        "105   3.53x53   137  1010\n"
                        "106   3.54x54   138  1011\n"
                        "107   3.55x55   139  1012\n"
                        "108   3.56x56   140  1013"
                         )
        matches = AC.AAresSeq_match(patterns="3.5*")
        self.assertEqual(matches.to_string(),
                         "    consensus    3CAP    3SN6\n"
                         "102   3.50x50  ARG135  ARG131\n"
                         "103   3.51x51  TYR136  TYR132\n"
                         "104   3.52x52  VAL137  PHE133\n"
                         "105   3.53x53  VAL138  ALA134\n"
                         "106   3.54x54  VAL139  ILE135\n"
                         "107   3.55x55  CYS140  THR136\n"
                         "108   3.56x56  LYS141  SER137"
                         )
        matches = AC.CAidxs_match(patterns="3.5*")
        self.assertEqual(matches.to_string(),
                         "    consensus  3CAP  3SN6\n"
                         "102   3.50x50  1065  7835\n"
                         "103   3.51x51  1076  7846\n"
                         "104   3.52x52  1088  7858\n"
                         "105   3.53x53  1095  7869\n"
                         "106   3.54x54  1102  7874\n"
                         "107   3.55x55  1109  7882\n"
                         "108   3.56x56  1115  7889"
                         )

    def test_with_CL(self):
        AC = nomenclature.AlignerConsensus(tops={"3CAP": self.geom_3CAP.top,
                                                 "1U19": self.geom_1U19.top
                                                 },
                                           CL=self.CL_opsd_bovin)
        self.assertListEqual(AC.keys, ["3CAP", "1U19"])
        self.assertIsInstance(AC.residxs, DataFrame)
        self.assertIsInstance(AC.AAresSeq, DataFrame)
        self.assertIsInstance(AC.CAidxs, DataFrame)
        self.assertListEqual(list(AC.residxs.keys()), ["consensus"] + AC.keys)
        self.assertListEqual(list(AC.AAresSeq.keys()), ["consensus"] + AC.keys)
        self.assertListEqual(list(AC.CAidxs.keys()), ["consensus"] + AC.keys)

        matches = AC.residxs_match(patterns="3.5*")
        self.assertEqual(matches.to_string(),
                         "    consensus  3CAP  1U19\n"
                         "102   3.50x50   134   484\n"
                         "103   3.51x51   135   485\n"
                         "104   3.52x52   136   486\n"
                         "105   3.53x53   137   487\n"
                         "106   3.54x54   138   488\n"
                         "107   3.55x55   139   489\n"
                         "108   3.56x56   140   490"
                         )
        matches = AC.AAresSeq_match(patterns="3.5*")
        self.assertEqual(matches.to_string(),
                         "    consensus    3CAP    1U19\n"
                         "102   3.50x50  ARG135  ARG135\n"
                         "103   3.51x51  TYR136  TYR136\n"
                         "104   3.52x52  VAL137  VAL137\n"
                         "105   3.53x53  VAL138  VAL138\n"
                         "106   3.54x54  VAL139  VAL139\n"
                         "107   3.55x55  CYS140  CYS140\n"
                         "108   3.56x56  LYS141  LYS141"
                         )
        matches = AC.CAidxs_match(patterns="3.5*")
        self.assertEqual(matches.to_string(),
                         "    consensus  3CAP  1U19\n"
                         "102   3.50x50  1065  3817\n"
                         "103   3.51x51  1076  3828\n"
                         "104   3.52x52  1088  3840\n"
                         "105   3.53x53  1095  3847\n"
                         "106   3.54x54  1102  3854\n"
                         "107   3.55x55  1109  3861\n"
                         "108   3.56x56  1115  3867"
                         )

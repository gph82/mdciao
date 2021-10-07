import unittest
import mdtraj as md
import numpy as _np
from os import path
from tempfile import TemporaryDirectory as _TDir, mkdtemp
import shutil
from urllib.error import HTTPError
from shutil import copy

import pytest
from mdciao import nomenclature
# It's a sign of bad design to have to import these private methods here
# for testing, they should be tested by the methods using them or
# made public
# When API design is more stable will TODO
from mdciao.nomenclature.nomenclature import \
    _CGN_fragments, \
    _GPCR_web_lookup, \
    _fill_consensus_gaps, \
    _map2defs, \
    _consensus_maps2consensus_frags
    #_top2consensus_map
#TODO make these imports cleaner
from mdciao.examples import filenames as test_filenames
from mdciao import examples

from mdciao.fragments import get_fragments

import mock

from pandas import DataFrame



class Test_md_load_rscb(unittest.TestCase):

    def test_works(self):
        geom = nomenclature.md_load_rscb("3CAP",
                                         verbose=True,
                                         )
        assert isinstance(geom, md.Trajectory)
    def test_works_return_url(self):
        geom, url = nomenclature.md_load_rscb("3CAP",
                                              #verbose=True,
                                              return_url=True
                                              )
        assert isinstance(geom, md.Trajectory)
        assert isinstance(url, str)
        assert "http" in url

class Test_PDB_finder(unittest.TestCase):

    def test_works_locally(self):
        geom, filename = nomenclature.PDB_finder(path.splitext(test_filenames.top_pdb)[0],
                                                 local_path=test_filenames.example_path,
                                                 try_web_lookup=False)
        assert isinstance(geom, md.Trajectory)
        assert isinstance(filename, str)

    def test_works_locally_pdbgz(self):
        geom, filename = nomenclature.PDB_finder("3SN6",
                                                 local_path=test_filenames.RSCB_pdb_path,
                                                 try_web_lookup=False)
        assert isinstance(geom, md.Trajectory)
        assert isinstance(filename, str)

    def test_works_online(self):
        geom, filename = nomenclature.PDB_finder("3SN6")

        assert isinstance(geom, md.Trajectory)
        assert isinstance(filename, str)
        assert "http" in filename

    def test_fails_bc_no_online_access(self):
        with pytest.raises((OSError,FileNotFoundError)):
            nomenclature.PDB_finder("3SN6",
                                    try_web_lookup=False)

class Test_CGN_finder(unittest.TestCase):

    def test_works_locally(self):
        df, filename = nomenclature.CGN_finder("3SN6",
                                               try_web_lookup=False,
                                               local_path=test_filenames.nomenclature_path)

        assert isinstance(df, DataFrame)
        assert isinstance(filename,str)
        _np.testing.assert_array_equal(list(df.keys()),["CGN","Sort number","3SN6"])

    def test_works_online(self):
        df, filename = nomenclature.CGN_finder("3SN6",
                                               )

        assert isinstance(df, DataFrame)
        assert isinstance(filename, str)
        assert "http" in filename
        _np.testing.assert_array_equal(list(df.keys()), ["CGN", "Sort number", "3SN6"])

    def test_works_online_and_writes_to_disk_excel(self):
        with _TDir(suffix="_mdciao_test") as tdir:
            df, filename = nomenclature.CGN_finder("3SN6",
                                                   format="%s.xlsx",
                                                   local_path=tdir,
                                                   write_to_disk=True
                                                   )

            assert isinstance(df, DataFrame)
            assert isinstance(filename, str)
            assert "http" in filename
            _np.testing.assert_array_equal(list(df.keys()), ["CGN", "Sort number", "3SN6"])
            assert path.exists(path.join(tdir,"3SN6.xlsx"))

    def test_works_online_and_writes_to_disk_ascii(self):
        with _TDir(suffix="_mdciao_test") as tdir:
            df, filename = nomenclature.CGN_finder("3SN6",
                                                   local_path=tdir,
                                                   format="%s.txt",
                                                   write_to_disk=True
                                                   )

            assert isinstance(df, DataFrame)
            assert isinstance(filename, str)
            assert "http" in filename
            _np.testing.assert_array_equal(list(df.keys()), ["CGN", "Sort number", "3SN6"])
            assert path.exists(path.join(tdir,"3SN6.txt"))

    def test_works_local_does_not_overwrite(self):
        with _TDir(suffix="_mdciao_test") as tdir:
            infile = test_filenames.CGN_3SN6
            copy(infile,tdir)
            with pytest.raises(FileExistsError):
                nomenclature.CGN_finder("3SN6",
                                        try_web_lookup=False,
                                        local_path=tdir,
                                        write_to_disk=True
                                        )



    def test_raises_not_find_locally(self):
        with pytest.raises(FileNotFoundError):
            nomenclature.CGN_finder("3SN6",
                                    try_web_lookup=False
                                    )

    def test_not_find_locally_but_no_fail(self):
        DF, filename = nomenclature.CGN_finder("3SN6",
                                               try_web_lookup=False,
                                               dont_fail=True
                                               )
        assert DF is None
        assert isinstance(filename,str)

    def test_raises_not_find_online(self):
        with pytest.raises(HTTPError):
            nomenclature.CGN_finder("3SNw",
                                    )

    def test_not_find_online_but_no_raise(self):
        df, filename =    nomenclature.CGN_finder("3SNw",
                                                  dont_fail=True
                                                  )
        assert df is None
        assert isinstance(filename,str)
        assert "www" in filename

class Test_GPCRmd_lookup_GPCR(unittest.TestCase):

    def test_works(self):
        DF = _GPCR_web_lookup("https://gpcrdb.org/services/residues/extended/adrb2_human")
        assert isinstance(DF, DataFrame)

    def test_wrong_code(self):
        with pytest.raises(ValueError):
            raise _GPCR_web_lookup("https://gpcrdb.org/services/residues/extended/adrb_beta2")

class Test_GPCR_finder(unittest.TestCase):

    def test_works_locally(self):
        df, filename = nomenclature.GPCR_finder(test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx,
                                                try_web_lookup=False,
                                                )

        assert isinstance(df, DataFrame)
        assert isinstance(filename,str)
        _np.testing.assert_array_equal(list(df.keys())[:3], nomenclature.nomenclature._GPCR_mandatory_fields)
        assert any([key in df.keys() for key in nomenclature.nomenclature._GPCR_mandatory_fields]) #at least one scheme


    def test_works_online(self):
        df, filename = nomenclature.GPCR_finder("adrb2_human",
                                                )

        assert isinstance(df, DataFrame)
        assert isinstance(filename, str)
        assert "http" in filename
        _np.testing.assert_array_equal(list(df.keys())[:3],nomenclature.nomenclature._GPCR_mandatory_fields)
        assert any([key in df.keys() for key in nomenclature.nomenclature._GPCR_mandatory_fields])

    def test_raises_not_find_locally(self):
        with pytest.raises(FileNotFoundError):
            nomenclature.GPCR_finder("B2AR",
                                     try_web_lookup=False
                                     )

    def test_not_find_locally_but_no_fail(self):
        DF, filename = nomenclature.GPCR_finder("B2AR",
                                                try_web_lookup=False,
                                                dont_fail=True
                                                )
        assert DF is None
        assert isinstance(filename,str)


    def test_raises_not_find_online(self):
        with pytest.raises(ValueError):
            nomenclature.GPCR_finder("B2AR",
                                     )

    def test_not_find_online_but_no_raise(self):
        df, filename =    nomenclature.GPCR_finder("3SNw",
                                                   dont_fail=True
                                                   )
        assert df is None
        assert isinstance(filename,str)

class Test_table2GPCR_by_AAcode(unittest.TestCase):
    def setUp(self):
        self.file = test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx

    def test_just_works(self):
        table2GPCR = nomenclature.table2GPCR_by_AAcode(tablefile = self.file)
        self.assertDictEqual(table2GPCR,
                             {'Q26': '1.25',
                              'E27': '1.26',
                              'E62': '12.48',
                              'R63': '12.49',
                              'T66': '2.37',
                              'V67': '2.38'
                              })

    def test_keep_AA_code_test(self): #dictionary keys will only have AA id
        table2GPCR = nomenclature.table2GPCR_by_AAcode(tablefile = self.file, keep_AA_code=False)
        self.assertDictEqual(table2GPCR,
                             {26: '1.25',
                              27: '1.26',
                              62: '12.48',
                              63: '12.49',
                              66: '2.37',
                              67: '2.38',
                           })

    def test_table2GPCR_by_AAcode_return_fragments(self):
        table2GPCR, defs = nomenclature.table2GPCR_by_AAcode(tablefile=self.file,
                                                           return_fragments=True)

        self.assertDictEqual(defs,{'TM1':  ["Q26","E27"],
                                   "ICL1": ["E62","R63"],
                                   "TM2" : ["T66","V67"]})
    def test_table2B_by_AAcode_already_DF(self):
        from pandas import read_excel
        df = read_excel(self.file, header=0, engine="openpyxl")

        table2GPCR = nomenclature.table2GPCR_by_AAcode(tablefile=df)
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
        self._CGN_3SN6_file = path.join(self.tmpdir,path.basename(test_filenames.CGN_3SN6))
        self._PDB_3SN6_file = path.join(self.tmpdir,path.basename(test_filenames.pdb_3SN6))
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
        #_np.testing.assert_equal(cgn_local.geom,
        #                         self._geom_3SN6)

        #_np.testing.assert_equal(cgn_local.top,
        #                         self._geom_3SN6.top)

    def test_dataframe(self):
        self.assertIsInstance(self.cgn_local.dataframe, DataFrame)
        self.assertSequenceEqual(list(self.cgn_local.dataframe.keys()),
                                 ["CGN","Sort number","3SN6"])

    def test_correct_residue_dicts(self):
        _np.testing.assert_equal(self.cgn_local.conlab2AA["G.hfs2.2"],"R201")
        _np.testing.assert_equal(self.cgn_local.AA2conlab["R201"],"G.hfs2.2")

    def test_correct_fragments_dict(self):
        # Test "fragments" dictionary SMH
        self.assertIsInstance(self.cgn_local.fragments,dict)
        assert all([len(ii)>0 for ii in self.cgn_local.fragments.values()])
        self.assertEqual(self.cgn_local.fragments["G.HN"][0],"T9")
        self.assertSequenceEqual(list(self.cgn_local.fragments.keys()),
                                 _CGN_fragments)

    def test_correct_fragments_as_conlabs_dict(self):
        # Test "fragments_as_conslabs" dictionary SMH
        self.assertIsInstance(self.cgn_local.fragments_as_conlabs, dict)
        assert all([len(ii) > 0 for ii in self.cgn_local.fragments_as_conlabs.values()])
        self.assertEqual(self.cgn_local.fragments_as_conlabs["G.HN"][0], "G.HN.26")
        self.assertSequenceEqual(list(self.cgn_local.fragments_as_conlabs.keys()),
                                 _CGN_fragments)

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
        out_dict = self.cgn_local.conlab2residx(self.cgn_local.top,map=map)
        self.assertEqual(out_dict["G.hfs2.2"],164)

    def test_conlab2residx_w_input_map_duplicates(self):
        map = [None for ii in range(200)]
        map[164] = "G.hfs2.2"  # I know this a priori using find_AA
        map[165] = "G.hfs2.2"
        with pytest.raises(ValueError):
            self.cgn_local.conlab2residx(self.cgn_local.top, map=map)

    def test_top2frags_just_passes(self):
        defs = self.cgn_local.top2frags(self.cgn_local.top)
        self.assertSequenceEqual(list(defs.keys()),
                                 _CGN_fragments)

    def test_top2frags_gets_dataframe(self):
        self.cgn_local.aligntop(self.cgn_local.top)
        defs = self.cgn_local.top2frags(self.cgn_local.top,
                                        input_dataframe=self.cgn_local.most_recent_alignment)
        self.assertSequenceEqual(list(defs.keys()),
                                 _CGN_fragments)

    def test_top2frags_defs_are_broken_in_frags(self):

        input_values = (val for val in ["0-1"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            defs = self.cgn_local.top2frags(self.cgn_local.top,
                                            fragments=[_np.arange(0,10),
                                                      _np.arange(10,15),
                                                      _np.arange(15,20)
                                                      ]
                                            )
            self.assertSequenceEqual(list(defs.keys()),
                                     _CGN_fragments)
            _np.testing.assert_array_equal(defs["G.HN"],_np.arange(0,15))

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
        _np.testing.assert_array_equal([len(ifrag) for ifrag in frags_as_idsx], [len(ifrag) for ifrag in self.cgn_local.fragments])
        _np.testing.assert_array_equal(list(frags_as_idsx.keys()),
                                       self.cgn_local.fragment_names)

    def test_PDB_full_path_exists(self):
        self.cgn_local = nomenclature.LabelerCGN(self._CGN_3SN6_file,
                                    try_web_lookup=False,
                                    local_path=self.tmpdir,
                                    )

    # These tests only test it runs, not that the alignment is correct
    #  those checks are done in sequence tests
    def test_aligntop_with_self(self):
        top2self, self2top = self.cgn_local.aligntop(self.cgn_local.seq)
        self.assertDictEqual(top2self,self2top)
    def test_aligntop_with_self_residxs(self):
        top2self, self2top = self.cgn_local.aligntop(self.cgn_local.seq, restrict_to_residxs=[2, 3])
        self.assertDictEqual(top2self,self2top)
        self.assertTrue(all([key in [2,3] for key in top2self.keys()]))
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
                                                             path.basename(test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx))
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
        self._GPCRmd_B2AR_nomenclature_test_xlsx = path.join(self.tmpdir, path.basename(test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx))
        shutil.copy(test_filenames.pdb_3SN6, self._PDB_3SN6_file)
        shutil.copy(test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx,self._GPCRmd_B2AR_nomenclature_test_xlsx)
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
                                 nomenclature.nomenclature._GPCR_mandatory_fields+nomenclature.nomenclature._GPCR_available_schemes)

    def test_correct_residue_dicts(self):
        if self.GPCR_local_w_pdb._nomenclature_key=="BW":
            _np.testing.assert_equal(self.GPCR_local_w_pdb.conlab2AA["1.25"],"Q26")
            _np.testing.assert_equal(self.GPCR_local_w_pdb.AA2conlab["Q26"],"1.25")
        elif self.GPCR_local_w_pdb._nomenclature_key=="display_generic_number":
            _np.testing.assert_equal(self.GPCR_local_w_pdb.conlab2AA["1.25x25"],"Q26")
            _np.testing.assert_equal(self.GPCR_local_w_pdb.AA2conlab["Q26"],"1.25x25")
        else:
            raise ValueError("no tests written for %s yet"%(self.GPCR_local_w_pdb._nomenclature_key))
    def test_correct_fragments_dict(self):
        # Test "fragments" dictionary SMH
        self.assertIsInstance(self.GPCR_local_w_pdb.fragments,dict)
        assert all([len(ii)>0 for ii in self.GPCR_local_w_pdb.fragments.values()])
        self.assertEqual(self.GPCR_local_w_pdb.fragments["ICL1"][0],"E62")
        self.assertSequenceEqual(list(self.GPCR_local_w_pdb.fragments.keys()),
                                 ["TM1","ICL1","TM2"])

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
        self.assertSequenceEqual([len(ifrag) for ifrag in frags_as_idsx], [len(ifrag) for ifrag in self.GPCR_local_w_pdb.fragments])
        self.assertSequenceEqual(list(frags_as_idsx.keys()),
                                 self.GPCR_local_w_pdb.fragment_names)

    # These tests only test it runs, not that the alignment is correct
    #  those checks are done in sequence tests
    def test_aligntop_with_self(self):
        top2self, self2top = self.GPCR_local_w_pdb.aligntop(self.GPCR_local_w_pdb.seq)
        self.assertDictEqual(top2self,self2top)
        self.assertIsInstance(self.GPCR_local_w_pdb.most_recent_alignment,DataFrame)
    def test_aligntop_with_self_residxs(self):
        top2self, self2top = self.GPCR_local_w_pdb.aligntop(self.GPCR_local_w_pdb.seq, restrict_to_residxs=[2, 3])
        self.assertDictEqual(top2self,self2top)
        self.assertTrue(all([key in [2,3] for key in top2self.keys()]))
        self.assertTrue(all([val in [2, 3] for val in top2self.values()]))

    def test_uniprot_name(self):
        self.assertEqual(self.GPCR_local_w_pdb.uniprot_name, self._GPCRmd_B2AR_nomenclature_test_xlsx)

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
        self.cons_list =  ['3.67','G.H5.1','G.H5.6','5.69']
        self.cons_list_w_Nones = ['3.67', None, None, 'G.H5.1','G.H5.6','5.69']
        self.cons_list_wo_dots = ['367', None, None, 'G.H5.1','G.H5.6','5.69']


    def test_works(self):
        map2defs = _map2defs(self.cons_list)
        assert _np.array_equal(map2defs['3'], [0])
        assert _np.array_equal(map2defs['G.H5'], [1, 2])
        assert _np.array_equal(map2defs['5'], [3])
        _np.testing.assert_equal(len(map2defs),3)

    def test_works_w_Nones(self):
        map2defs = _map2defs(self.cons_list_w_Nones)
        assert _np.array_equal(map2defs['3'], [0])
        assert _np.array_equal(map2defs['G.H5'], [3,4])
        assert _np.array_equal(map2defs['5'], [5])
        _np.testing.assert_equal(len(map2defs), 3)

    def test_works_wo_dot_raises(self):
        with pytest.raises(AssertionError):
            _map2defs(self.cons_list_wo_dots)

class Test_fill_CGN_gaps(unittest.TestCase):
    def setUp(self):
        self.top_3SN6 = md.load(test_filenames.pdb_3SN6).top
        self.top_mut = md.load(test_filenames.pdb_3SN6_mut).top
        self.cons_list_out = ['G.HN.26', 'G.HN.27', 'G.HN.28', 'G.HN.29', 'G.HN.30']
        self.cons_list_in = ['G.HN.26', None, 'G.HN.28', 'G.HN.29', 'G.HN.30']

    def test_fill_CGN_gaps_just_works_with_CGN(self):
        fill_cgn = _fill_consensus_gaps(self.cons_list_in, self.top_mut, verbose=True)
        self.assertEqual(fill_cgn,self.cons_list_out)


class Test_fill_consensus_gaps(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.top_pdb)
        self.cons_list_in = ['3.46', '3.47', "3.48", None,
                             None, '3.51', '3.52']
        self.cons_list_out = ['3.46', '3.47', "3.48", "3.49",
                             "3.50", '3.51', '3.52']

    def test_fill_CGN_gaps_just_works_with_GPCR(self):
        fill_cgn = _fill_consensus_gaps(self.cons_list_in, self.geom.top, verbose=True)
        self.assertEqual(fill_cgn, self.cons_list_out)

class Test_guess_by_nomenclature(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        GPCRlabs_file = path.relpath(test_filenames.adrb2_human_xlsx, test_filenames.RSCB_pdb_path)

        cls.GPCR_local_w_pdb = nomenclature.LabelerGPCR(GPCRlabs_file,
                                                      ref_PDB="3SN6",
                                                      local_path=test_filenames.RSCB_pdb_path,
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
            self.assertSequenceEqual(answer,[4])

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
                                       min_hit_rate=2,  #impossible rate
                                       )
        self.assertEqual(answer, None)

class Test_guess_nomenclature_fragments(unittest.TestCase):
    # The setup is in itself a test
    @classmethod
    def setUpClass(cls):
        GPCRlabs_file = path.relpath(test_filenames.adrb2_human_xlsx, test_filenames.RSCB_pdb_path)

        cls.GPCR_local_w_pdb = nomenclature.LabelerGPCR(GPCRlabs_file,
                                                      ref_PDB="3SN6",
                                                      local_path=test_filenames.RSCB_pdb_path,
                                                      format="%s",
                                                      )
        cls.fragments = get_fragments(cls.GPCR_local_w_pdb.top,verbose=False)

    def test_finds_frags(self):
        guessed_frags = nomenclature.guess_nomenclature_fragments(self.GPCR_local_w_pdb,
                                                                  self.GPCR_local_w_pdb.top,
                                                                  fragments=self.fragments,
                                                                  verbose=True,
                                                                  )
        _np.testing.assert_array_equal([4],guessed_frags)

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
        _np.testing.assert_array_equal([4],guessed_frags)

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
        assert  guessed_frags is None

if __name__ == '__main__':
    unittest.main()

class Test_sort_consensus_labels(unittest.TestCase):

    def setUp(self):
        self.tosort = ["G.H1.10", "H.HA.20", "H8.10", "V34", "H8.1", "3.50", "2.50", "G.H1.1", "H.HA.10"]


    def test_GPCR(self):
        sorted = nomenclature.sort_GPCR_consensus_labels(self.tosort)
        _np.testing.assert_array_equal(["2.50", "3.50","H8.1", "H8.10", "G.H1.10", "H.HA.20", "V34", "G.H1.1", "H.HA.10"],
                                       sorted)

    def test_GPCR_dont_append(self):
        sorted = nomenclature.sort_GPCR_consensus_labels(self.tosort, append_diffset=False)
        _np.testing.assert_array_equal(["2.50", "3.50","H8.1", "H8.10"],
                                       sorted)

    def test_CGN(self):
        sorted = nomenclature.sort_CGN_consensus_labels(self.tosort)
        _np.testing.assert_array_equal(["G.H1.1", "G.H1.10", "H.HA.10", "H.HA.20", "H8.10", "V34", "H8.1", "3.50", "2.50"],
                                       sorted)

    def test_CGN_dont_append(self):
        sorted = nomenclature.sort_CGN_consensus_labels(self.tosort, append_diffset=False)
        _np.testing.assert_array_equal(["G.H1.1", "G.H1.10", "H.HA.10", "H.HA.20"],
                                       sorted)


class Test_compatible_consensus_fragments(TestClassSetUpTearDown_CGN_local):

    def setUp(self):
        super(Test_compatible_consensus_fragments,self).setUp()
        self.top = md.load(test_filenames.actor_pdb).top

    def test_works(self):
        # Obtain the full objects first
        full_map = self.cgn_local.top2labels(self.top,
                                             autofill_consensus=True,
                                             #verbose=True
                                             )

        frag_defs = self.cgn_local.top2frags(self.top,
                                             verbose=False,
                                             #show_alignment=True,
                                             #map_conlab=full_map
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
        self.assertEqual("TM3",nomenclature.conslabel2fraglabel("GLU30@3.50"))

class Test_alignment_df2_conslist(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.list_of_dicts = [
            {"idx_0": 0,
             "idx_1": 0,
             "match": True,
             "AA_0": "GLU",
             "AA_1": "GLU",
             "conlab":"3.50",
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
             "conlab":"3.52",
             },
        ]
        cls.df  = DataFrame(cls.list_of_dicts)
        #cls.consensus_dict = {"GLU0": "3.50",
        #                      "ARG1": "3.51",
        #                      "PHE2": "3.52"}

    def test_works(self):
        out_list = nomenclature.alignment_df2_conslist(self.df)
        self.assertListEqual(out_list, ["3.50", None, "3.52"])

    def test_works_nonmatch(self):
        out_list = nomenclature.alignment_df2_conslist(self.df, allow_nonmatch=True)
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
            cls.frags[GPCR_scheme] = [lab.top2frags(cls.geom.top) for lab in [cls.CGN, cls.GPCR[GPCR_scheme]]] # This method doesn't rely on  _consensus_maps2consensus_frags

    def test_works_on_empty(self):
        maps, frags = _consensus_maps2consensus_frags(self.geom.top, [], verbose=True)
        assert maps == []
        assert frags == {}

    def test_works_on_maps(self):
        for imaps in self.maps.values():
            maps, frags = _consensus_maps2consensus_frags(self.geom.top, imaps, verbose=True)
            self.assertListEqual(maps, imaps)
            assert frags == {}

    def test_works_on_Labelers(self):
        for GPCR_scheme, GPCR in self.GPCR.items():
            maps, frags = _consensus_maps2consensus_frags(self.geom.top, [self.CGN, GPCR], verbose=True)
            self.assertListEqual(maps, self.maps[GPCR_scheme])
            for ifrags in self.frags[GPCR_scheme]:
                for key, val in ifrags.items():
                    self.assertListEqual(frags[key],val)

    def test_works_on_mix(self):
        for GPCR_scheme, GPCR in self.GPCR.items():
            maps, frags = _consensus_maps2consensus_frags(self.geom.top, [self.maps[GPCR_scheme][0], GPCR], verbose=True)
            self.assertListEqual(maps, self.maps[GPCR_scheme])
            self.assertDictEqual(frags, self.frags[GPCR_scheme][1])

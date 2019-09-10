import unittest
import mdtraj as md
from os import path
from sofi_functions.nomenclature_utils import *
from filenames import filenames

test_filenames = filenames()
class Test_table2BW_by_AAcode(unittest.TestCase):
    def setUp(self):
        self.file = path.join(test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx)

    def test_table2BW_by_AAcode_just_works(self):
        table2BW = table2BW_by_AAcode(tablefile = self.file)
        self.assertDictEqual(table2BW,
                             {'Q26': '1.25',
                              'E27': '1.26',
                              'R28': '1.27',
                              'F264': '1.28',
                              'M40': '1.39',
                              'S41': '1.40',
                              'L42': '1.41',
                              'I43': '1.42',
                              'V44': '1.43',
                              'L45': '1.44',
                              'A46': '1.45',
                              'I47': '1.46',
                              'V48': '1.47'})

    def test_table2BW_by_AAcode_keep_AA_code_test(self): #dictionary keys will only have AA id
        table2BW = table2BW_by_AAcode(tablefile = self.file, keep_AA_code=False)
        self.assertDictEqual(table2BW,
                             {26: '1.25',
                              27: '1.26',
                              28: '1.27',
                              264: '1.28',
                              40: '1.39',
                              41: '1.40',
                              42: '1.41',
                              43: '1.42',
                              44: '1.43',
                              45: '1.44',
                              46: '1.45',
                              47: '1.46',
                              48: '1.47'})

    def test_table2BW_by_AAcode_return_defs_test(self):
        table2BW = table2BW_by_AAcode(tablefile=self.file, return_defs=True)
        self.assertEqual(table2BW,
                         ({'Q26': '1.25',
                           'E27': '1.26',
                           'R28': '1.27',
                           'F264': '1.28',
                           'M40': '1.39',
                           'S41': '1.40',
                           'L42': '1.41',
                           'I43': '1.42',
                           'V44': '1.43',
                           'L45': '1.44',
                           'A46': '1.45',
                           'I47': '1.46',
                           'V48': '1.47'},
                          ['TM1']))

class Test_guess_missing_BWs(unittest.TestCase):
    #TODO need to attain 100% coverage
    def setUp(self):
        self.file = path.join(test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx)
        self.geom = md.load(test_filenames.file_for_test_pdb)

    def test_guess_missing_BWs_just_works(self):
        table2BW = table2BW_by_AAcode(tablefile=self.file)
        guess_BW = guess_missing_BWs(table2BW, self.geom.top, restrict_to_residxs=None)
        self.assertDictEqual(guess_BW,
                             {0: '1.29*',
                              1: '1.30*',
                              2: '1.31*',
                              3: '1.27*',
                              4: '1.26',
                              5: '1.27*',
                              6: '1.28*',
                              7: '1.28*'})

class Test_CGN_transformer(unittest.TestCase):
    def setUp(self):
        self.cgn = CGN_transformer(ref_path=test_filenames.examples_path)

    def test_CGN_transformer_just_works(self):
        self.assertEqual(len(self.cgn.seq), len(self.cgn.seq_idxs))
        self.assertEqual(len(self.cgn.seq), len(self.cgn.AA2CGN))

class Test_top2CGN_by_AAcode(unittest.TestCase):
    def setUp(self):
        self.cgn = CGN_transformer(ref_path=test_filenames.examples_path)
        self.geom = md.load(test_filenames.file_for_test_pdb)

    def test_top2CGN_by_AAcode_just_works(self):
        top2CGN = top2CGN_by_AAcode(self.geom.top, self.cgn)
        self.assertDictEqual(top2CGN,
                             {0: 'G.HN.27',
                              1: 'G.HN.53',
                              2: 'H.HC.11',
                              3: 'H.hdhe.4',
                              4: 'G.S2.3',
                              5: 'G.S2.5',
                              6: None,
                              7: 'G.S2.6'})

if __name__ == '__main__':
    unittest.main()
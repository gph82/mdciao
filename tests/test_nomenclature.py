import unittest
import mdtraj as md
import numpy as _np
from os import path
from sofi_functions.nomenclature_utils import *
from sofi_functions.nomenclature_utils import _map2defs, _top2consensus_map, _fill_CGN_gaps
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
        table2BW, defs = table2BW_by_AAcode(tablefile=self.file, return_defs=True)
        self.assertEqual(table2BW,
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

        self.assertEqual(defs,['TM1'])

class Test_guess_missing_BWs(unittest.TestCase):
    #TODO change this test to reflect the new changes Guillermo recently added
    def setUp(self):
        self.file = path.join(test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx)
        self.geom = md.load(test_filenames.file_for_test_pdb)

    def _test_guess_missing_BWs_just_works(self):
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
    #TODO change this test to reflect the new changes Guillermo recently added
    def setUp(self):
        self.cgn = CGN_transformer(ref_path=test_filenames.examples_path)
        self.geom = md.load(test_filenames.file_for_test_pdb)

    def _test_top2CGN_by_AAcode_just_works(self):
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

class Test_map2defs(unittest.TestCase):
    def setUp(self):
        self.cons_list =  ['3.67','G.H5.1','G.H5.6','5.69']

    def test_map2defs_just_works(self):
        map2defs = _map2defs(self.cons_list)
        assert (_np.array_equal(map2defs['3'], [0]))
        assert (_np.array_equal(map2defs['G.H5'], [1, 2]))
        assert (_np.array_equal(map2defs['5'], [3]))

class Test_add_loop_definitions_to_TM_residx_dict(unittest.TestCase):
    def setUp(self):
        self.segment_dict = {'TM1': [20, 21, 22], 'TM2': [30, 33, 34], 'TM3': [40, 48], 'TM4': [50, 56],
                             'TM5': [60, 61],'TM6': [70], 'TM7': [80, 81, 82, 83, 89], 'H8': [90, 91, 92, 93, 94, 95]}

    def test_add_loop_definitions_to_TM_residx_dict_just_works(self):
        add_defs = add_loop_definitions_to_TM_residx_dict(self.segment_dict)
        self.assertEqual(add_defs['ICL1'],[23, 29])
        self.assertEqual(add_defs['ECL1'], [35, 39])
        self.assertEqual(add_defs['ICL2'], [49, 49])
        self.assertEqual(add_defs['ECL2'], [57, 59])
        self.assertEqual(add_defs['ECL3'], [71, 79])

#
# class Test_table2TMdefs_resSeq(unittest.TestCase):
#TODO test to be completed after clarifying from Guillermo

#     def setUp(self):
#         self.file = path.join(test_filenames.GPCRmd_B2AR_nomenclature_test_xlsx)
#
#     def table2TMdefs_resSeq_just_works(self):
#         table2TMdefs = table2TMdefs_resSeq(tablefile=self.file, return_defs=True)

# class Test_csv_table2TMdefs_res_idxs(unittest.TestCase):
# #TODO test to be completed after clarifying from Guillermo


class Test_top2consensus_map(unittest.TestCase):
    def setUp(self):
        self.cgn = CGN_transformer(ref_path=test_filenames.examples_path)
        self.geom = md.load(test_filenames.file_for_top2consensus_map)
        self.cons_list_test = ['G.HN.26','G.HN.27','G.HN.28','G.HN.29','G.HN.30']
        self.cons_list_keep_consensus = ['G.hfs2.1', 'G.hfs2.2', 'G.hfs2.3', 'G.hfs2.4',
                                         'G.hfs2.5', 'G.hfs2.6', 'G.hfs2.7']

    def test_top2consensus_map_just_works(self): #generally works
        cons_list = _top2consensus_map(consensus_dict=self.cgn.AA2CGN, top=self.geom.top)

        count = 1
        cons_list_out = []
        for ii, val in enumerate(cons_list):
            if val is not None:
                cons_list_out.append(val)
                count += 1
            if count > 5: #testing for the first 5 entries in the pdb file which have a valid CGN name
                break
        self.assertEqual(cons_list_out, self.cons_list_test)

    def test_top2consensus_map_keep_consensus_is_true(self):
        #In the output below, instead of None, None, it will be 'G.hfs2.4' and 'G.hfs2.5'
        # ['G.hfs2.1', 'G.hfs2.2', 'G.hfs2.3', None, None, 'G.hfs2.6', 'G.hfs2.7']
        cons_list = _top2consensus_map(consensus_dict=self.cgn.AA2CGN, top=self.geom.top, keep_consensus=True)
        cons_list_out = []

        for ii, val in enumerate(cons_list):
            if (ii > 434 and ii < 442):
                cons_list_out.append(val)
        self.assertEqual(cons_list_out, self.cons_list_keep_consensus)

class Test_fill_CGN_gaps(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.file_for_top2consensus_map)
        self.cons_list_in = ['G.hfs2.1', 'G.hfs2.2', 'G.hfs2.3', None,
                          None, 'G.hfs2.6', 'G.hfs2.7']
        self.cons_list_out = ['G.hfs2.1', 'G.hfs2.2', 'G.hfs2.3', 'G.hfs2.4',
                                         'G.hfs2.5', 'G.hfs2.6', 'G.hfs2.7']

    def test_fill_CGN_gaps_just_works(self):
        fill_cgn = _fill_CGN_gaps(self.cons_list_in, self.geom.top)
        self.assertEqual(fill_cgn,self.cons_list_out)

if __name__ == '__main__':
    unittest.main()
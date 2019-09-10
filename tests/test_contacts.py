
import mdtraj as md
import unittest
import numpy as _np
import mock
from filenames import filenames
from sofi_functions.contacts import ctc_freq_reporter_by_residue_neighborhood, xtcs2ctcs

test_filenames = filenames()
class Test_ctc_freq_reporter_by_residue_neighborhood(unittest.TestCase):
    def setUp(self):
        from sofi_functions.fragments import get_fragments, interactive_fragment_picker_by_AAresSeq
        self.geom = md.load(test_filenames.file_for_test_pdb)
        self.by_bonds_geom = get_fragments(self.geom.top,
                                                     verbose=True,
                                                     auto_fragment_names=True,
                                                     method='bonds')
        self.residues = ["GLU30", "VAL31"]
        self.resname2residx, self.resname2fragidx = interactive_fragment_picker_by_AAresSeq(self.residues,
                                                                                                 self.by_bonds_geom,
                                                                                                 self.geom.top)

    def test_ctc_freq_reporter_by_residue_neighborhood_just_works(self):
        ctcs_mean = [30, 5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]

        input_values = (val for val in ["1", "1"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):#Checking against the input 1 and 1
            ctc_freq = ctc_freq_reporter_by_residue_neighborhood(ctcs_mean, self.resname2residx, self.by_bonds_geom, ctc_residxs_pairs,
                                                             self.geom.top,
                                                             n_ctcs=5, select_by_resSeq=None,
                                                             silent=False)
            assert ctc_freq[0] == 0
            assert ctc_freq[1] == 0


        input_values = (val for val in ["1", "2"])
        with mock.patch('builtins.input', lambda *x: next(input_values)): #Checking against the input 1 and 2
            ctc_freq = ctc_freq_reporter_by_residue_neighborhood(ctcs_mean, self.resname2residx, self.by_bonds_geom, ctc_residxs_pairs,
                                                            self.geom.top,
                                                             n_ctcs=5, select_by_resSeq=None,
                                                             silent=False)
            assert ctc_freq[0] == 0
            assert (_np.array_equal(ctc_freq[1],[0, 1]))

    def test_ctc_freq_reporter_by_residue_neighborhood_select_by_resSeq_is_int(self):
        ctcs_mean = [30, 5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]

        input_values = (val for val in ["1", "1"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):#Checking against the input 1 and 1
            ctc_freq = ctc_freq_reporter_by_residue_neighborhood(ctcs_mean, self.resname2residx, self.by_bonds_geom, ctc_residxs_pairs,
                                                             self.geom.top,
                                                             n_ctcs=5, select_by_resSeq=1,
                                                             silent=False)
            assert ctc_freq == {}

    def test_ctc_freq_reporter_by_residue_neighborhood_silent_is_true(self):
        ctcs_mean = [30, 5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]

        input_values = (val for val in ["1", "1"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):#Checking against the input 1 and 1
            ctc_freq = ctc_freq_reporter_by_residue_neighborhood(ctcs_mean, self.resname2residx, self.by_bonds_geom, ctc_residxs_pairs,
                                                             self.geom.top,
                                                             n_ctcs=5, select_by_resSeq=None,
                                                             silent=True)
            assert (_np.array_equal(ctc_freq[0], [0]))
            assert (_np.array_equal(ctc_freq[1], [0, 1]))

class Test_xtcs2ctcs(unittest.TestCase):
    def setUp(self):
        file_geom = test_filenames.prot1_pdb
        file_xtc =  test_filenames.run1_stride_100_xtc
        self.geom = md.load(file_geom)
        self.xtcs = [file_xtc]

    def test_xtcs2ctcs_just_works(self):
        ctcs_trajs, time_array = xtcs2ctcs(self.xtcs, self.geom.top, [[1, 6]],  #stride=a.stride,
                                           # chunksize=a.chunksize_in_frames,
                                           return_time=True,
                                           consolidate=False)
        test_ctcs_trajs = _np.array([[0.4707409],
                                     [0.44984677],
                                     [0.49991393],
                                     [0.53672814],
                                     [0.53126746],
                                     [0.49817562],
                                     [0.46696925],
                                     [0.4860978],
                                     [0.4558717],
                                     [0.4896131]])
        test_time_array = _np.array([    0., 10000., 20000., 30000., 40000., 50000., 60000., 70000.,
       80000., 90000.])

        _np.testing.assert_array_almost_equal(ctcs_trajs[0][:10], test_ctcs_trajs, 4)
        assert(_np.array_equal(time_array[0][:10],test_time_array))

    def test_xtcs2ctcs_return_time_is_false(self):
        ctcs_trajs = xtcs2ctcs(self.xtcs, self.geom.top, [[1, 6]],  #stride=a.stride,
                                           # chunksize=a.chunksize_in_frames,
                                           return_time=False,
                                           consolidate=False)
        test_ctcs_trajs = _np.array([[0.4707409],
                                     [0.44984677],
                                     [0.49991393],
                                     [0.53672814],
                                     [0.53126746],
                                     [0.49817562],
                                     [0.46696925],
                                     [0.4860978],
                                     [0.4558717],
                                     [0.4896131]])
        _np.testing.assert_array_almost_equal(ctcs_trajs[0][:10], test_ctcs_trajs, 4)

    def test_xtcs2ctcs_consolidate_is_true(self):
        ctcs_trajs, time_array = xtcs2ctcs(self.xtcs, self.geom.top, [[1, 6]],  # stride=a.stride,
                                               # chunksize=a.chunksize_in_frames,
                                               return_time=True,
                                               consolidate=True)
        test_ctcs_trajs = _np.array([[0.4707409],
                                         [0.44984677],
                                         [0.49991393],
                                         [0.53672814],
                                         [0.53126746],
                                         [0.49817562],
                                         [0.46696925],
                                         [0.4860978],
                                         [0.4558717],
                                         [0.4896131]])
        test_time_array = _np.array([0., 10000., 20000., 30000., 40000., 50000., 60000., 70000.,
                                         80000., 90000.])

        _np.testing.assert_array_almost_equal(ctcs_trajs[:10], test_ctcs_trajs, 4)
        assert (_np.array_equal(time_array[:10], test_time_array))

if __name__ == '__main__':
    unittest.main()

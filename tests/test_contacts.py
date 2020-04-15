
import mdtraj as md
import unittest
from unittest.mock import Mock
import numpy as _np
import mock
from scipy.spatial.distance import cdist
from filenames import filenames
from mdciao.contacts import select_and_report_residue_neighborhood_idxs, \
    trajs2ctcs, \
    pick_best_label, \
    contact_pair, \
    per_traj_ctc

class Test_for_contacs(unittest.TestCase):
    def setUp(self):
        self.pdb_file = test_filenames.prot1_pdb
        self.file_xtc = test_filenames.run1_stride_100_xtc
        self.top = md.load(self.pdb_file).top
        self.traj = md.load(self.file_xtc, top = self.top)
        self.ctc_idxs = [[10, 20], [20, 30]]
        self.ctcs = md.compute_contacts(self.traj, self.ctc_idxs)[0]

        atoms_10 = [aa.index for aa in self.top.residue(10).atoms if aa.element.symbol != "H"]
        atoms_20 = [aa.index for aa in self.top.residue(20).atoms if aa.element.symbol != "H"]
        atoms_30 = [aa.index for aa in self.top.residue(30).atoms if aa.element.symbol != "H"]

        xyz_10 = self.traj.xyz[:, atoms_10]
        xyz_20 = self.traj.xyz[:, atoms_20]
        xyz_30 = self.traj.xyz[:, atoms_30]

        my_idxs = []
        for frame_10, frame_20, frame_30 in zip(xyz_10, xyz_20, xyz_30):
            D = cdist(frame_10, frame_20)
            idxs_1020 = _np.unravel_index(D.argmin(), D.shape)

            D = cdist(frame_20, frame_30)
            idxs_2030 = _np.unravel_index(D.argmin(), D.shape)

            my_idxs.append([atoms_10[idxs_1020[0]], atoms_20[idxs_1020[1]],
                            atoms_20[idxs_2030[0]], atoms_30[idxs_2030[1]]])
        self.my_idxs = _np.vstack(my_idxs)

class Test_per_traj_ctc(Test_for_contacs):
    def test_contacts_file(self):
        ctcs, time, __ = per_traj_ctc(self.top, self.file_xtc, self.ctc_idxs, 1000, 1, 0)
        _np.testing.assert_allclose(ctcs,self.ctcs)
        _np.testing.assert_allclose(time, self.traj.time)

    def test_contacts_geom(self):
        ctcs, time, __ = per_traj_ctc(self.top, self.traj, self.ctc_idxs, 1000, 1, 0)
        _np.testing.assert_allclose(ctcs,self.ctcs)
        _np.testing.assert_allclose(time, self.traj.time)

    def test_contacts_geom_stride(self):
        ctcs, time, __ = per_traj_ctc(self.top, self.traj, self.ctc_idxs, 1000, 2, 0)
        _np.testing.assert_allclose(ctcs,self.ctcs[::2])
        _np.testing.assert_allclose(time, self.traj.time[::2])

    def test_contacts_geom_chunk(self):
        ctcs, time, __ = per_traj_ctc(self.top, self.traj, self.ctc_idxs, 5, 1, 0)
        _np.testing.assert_allclose(ctcs,self.ctcs)
        _np.testing.assert_allclose(time, self.traj.time)

    def test_atoms(self):
        __, __, iatoms = per_traj_ctc(self.top, self.file_xtc, [[10, 20], [20, 30]], 1000, 1, 0)
        _np.testing.assert_allclose(iatoms, self.my_idxs)

test_filenames = filenames()
class Test_ctc_freq_reporter_by_residue_neighborhood(unittest.TestCase):
    def setUp(self):
        from mdciao.fragments import get_fragments, interactive_fragment_picker_by_AAresSeq
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
            ctc_freq = select_and_report_residue_neighborhood_idxs(ctcs_mean, self.resname2residx, self.by_bonds_geom, ctc_residxs_pairs,
                                                                   self.geom.top,
                                                                   n_ctcs=5, restrict_to_resSeq=None,
                                                                   interactive=True)
            assert ctc_freq[0] == 0
            assert ctc_freq[1] == 0


        input_values = (val for val in ["1", "2"])
        with mock.patch('builtins.input', lambda *x: next(input_values)): #Checking against the input 1 and 2
            ctc_freq = select_and_report_residue_neighborhood_idxs(ctcs_mean, self.resname2residx, self.by_bonds_geom, ctc_residxs_pairs,
                                                                   self.geom.top,
                                                                   n_ctcs=5, restrict_to_resSeq=None,
                                                                   interactive=True)
            assert ctc_freq[0] == 0
            assert (_np.array_equal(ctc_freq[1],[0, 1]))

    def test_ctc_freq_reporter_by_residue_neighborhood_select_by_resSeq_is_int(self):
        ctcs_mean = [30, 5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]

        input_values = (val for val in ["1", "1"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):#Checking against the input 1 and 1
            ctc_freq = select_and_report_residue_neighborhood_idxs(ctcs_mean, self.resname2residx, self.by_bonds_geom, ctc_residxs_pairs,
                                                                   self.geom.top,
                                                                   n_ctcs=5, restrict_to_resSeq=1,
                                                                   interactive=True)
            assert ctc_freq == {}

    def test_ctc_freq_reporter_by_residue_neighborhood_hit_enter(self):
        ctcs_mean = [30, 5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]

        input_values = (val for val in ["", ""])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            ctc_freq = select_and_report_residue_neighborhood_idxs(ctcs_mean, self.resname2residx, self.by_bonds_geom, ctc_residxs_pairs,
                                                                   self.geom.top,
                                                                   n_ctcs=5, restrict_to_resSeq=None,
                                                                   interactive=True)
            assert ctc_freq == {}

    def test_ctc_freq_reporter_by_residue_neighborhood_silent_is_true(self):
        ctcs_mean = [30, 5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]

        ctc_freq = select_and_report_residue_neighborhood_idxs(ctcs_mean, self.resname2residx, self.by_bonds_geom, ctc_residxs_pairs,
                                                               self.geom.top,
                                                               n_ctcs=5, restrict_to_resSeq=None,
                                                               interactive=False)
        assert (_np.array_equal(ctc_freq[0], [0]))
        assert (_np.array_equal(ctc_freq[1], [0, 1]))


    def test_ctc_freq_reporter_by_residue_neighborhood_keyboard_interrupt(self):
        from mdciao.fragments import interactive_fragment_picker_by_AAresSeq
        ctcs_mean = [30, 5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]
        with unittest.mock.patch('builtins.input', side_effect=KeyboardInterrupt):
            resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq("GLU30",self.by_bonds_geom,
                                                                                  self.geom.top)


            ctc_freq = select_and_report_residue_neighborhood_idxs(ctcs_mean, resname2residx, self.by_bonds_geom, ctc_residxs_pairs,
                                                                   self.geom.top,
                                                                   n_ctcs=5, restrict_to_resSeq=None,
                                                                   interactive=True)
            assert ctc_freq == {}

class Test_trajs2ctcs(Test_for_contacs):

    def setUp(self):
        #TODO read why I shouldn't be doing this...https://nedbatchelder.com/blog/201210/multiple_inheritance_is_hard.html
        super(Test_trajs2ctcs, self).setUp()
        self.xtcs = [self.file_xtc, self.file_xtc]
        self.ctcs_stacked = _np.vstack([self.ctcs, self.ctcs])
        self.times_stacked = _np.hstack([self.traj.time, self.traj.time])
        self.atoms_stacked = _np.vstack([self.my_idxs, self.my_idxs])

    def test_works(self):
        ctcs_trajs_consolidated = trajs2ctcs(self.xtcs, self.top, self.ctc_idxs)
        _np.testing.assert_allclose(ctcs_trajs_consolidated, self.ctcs_stacked)


    def test_return_time_and_atoms(self):
        ctcs_trajs_consolidated, times_consolidated, atoms_consolidated = trajs2ctcs(self.xtcs, self.top, self.ctc_idxs,
                                                                                     return_times_and_atoms=True
                                                                                     )
        _np.testing.assert_allclose(self.times_stacked, times_consolidated)
        _np.testing.assert_allclose(self.ctcs_stacked, ctcs_trajs_consolidated)
        _np.testing.assert_allclose(self.atoms_stacked, atoms_consolidated)

    def test_consolidate_is_false(self):
        ctcs, times, atoms = trajs2ctcs(self.xtcs, self.top, self.ctc_idxs,
                                        return_times_and_atoms=True,
                                        consolidate=False)


        [_np.testing.assert_equal(itraj, jtraj) for (itraj, jtraj) in zip([self.ctcs, self.ctcs], ctcs)]
        [_np.testing.assert_equal(itraj, jtraj) for (itraj, jtraj) in zip([self.traj.time, self.traj.time], times)]
        [_np.testing.assert_equal(itraj, jtraj) for (itraj, jtraj) in zip([self.my_idxs, self.my_idxs], atoms)]

    def test_progressbar(self):
        ctcs_trajs_consolidated = trajs2ctcs(self.xtcs, self.top, self.ctc_idxs, progressbar=True)



class Test_pick_best_label(unittest.TestCase):
    def test_pick_best_label_just_works(self):
        assert (pick_best_label(fallback = "Print this instead",test = "Print this" ) == "Print this")

    def test_pick_best_label_exclude_works(self):
        assert(pick_best_label(fallback = "Print this instead",test = None) == "Print this instead")
        assert(pick_best_label(fallback = "Print this instead",test = "None" ) == "Print this instead")
        assert(pick_best_label(fallback = "Print this instead",test = "NA" ) == "Print this instead")
        assert(pick_best_label(fallback = "Print this instead",test = "na" ) == "Print this instead")

class Test_contact_pair(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)

    def test_contact_pair_just_works(self):
        contact_pair_test = contact_pair(res_idxs_pair = [0, 1],
                                                  ctc_trajs = [[1.0, 1.1, 1.3]],
                                                  time_arrays = [[0, 1, 2]],
                                                  # top=None,
                                                  # trajs=None,
                                                  # fragment_idxs=None,
                                                  # fragment_names=None,
                                                  # fragment_colors=None,
                                                  # anchor_residue_idx=None,
                                                  # consensus_labels=None
                                                  )
        #Checking all the passed arguments
        assert (contact_pair_test.res_idxs_pair == [0, 1])
        assert _np.allclose(contact_pair_test.ctc_trajs, [1.0, 1.1, 1.3])
        assert (contact_pair_test.time_arrays == [[0, 1, 2]])
        assert (contact_pair_test.n_frames == [3])
        assert (contact_pair_test.n_trajs == 1)
        assert (contact_pair_test.time_max == 2)
        assert (contact_pair_test.trajlabels == ['traj 0'])
        assert _np.allclose(contact_pair_test.binarize_trajs(12), [True, True, False])
        assert(round(contact_pair_test.frequency_overall_trajs(12),2) == 0.67)
        assert list(_np.around(_np.array(contact_pair_test.frequency_per_traj(12)),2) == [0.67])

        assert (contact_pair_test.anchor_fragment_name == None)
        assert (contact_pair_test.anchor_fragment_name_best == None)
        assert (contact_pair_test.anchor_fragment_name_consensus == None)
        assert (contact_pair_test.anchor_index == None)
        assert (contact_pair_test.anchor_res_and_fragment_str == None)
        assert (contact_pair_test.anchor_residue == None)
        assert (contact_pair_test.anchor_residue_index == None)
        assert (contact_pair_test.consensus_labels == None)
        assert (contact_pair_test.fragment_colors == None)
        assert (contact_pair_test.fragment_idxs == None)
        assert (contact_pair_test.fragment_names == None)
        assert (contact_pair_test.partner_fragment_name==None)
        assert (contact_pair_test.partner_fragment_name_best == None)
        assert (contact_pair_test.partner_fragment_name_consensus == None)
        assert (contact_pair_test.partner_index == None)
        assert (contact_pair_test.partner_res_and_fragment_str == 'None@None')
        assert (contact_pair_test.partner_residue == None)
        assert (contact_pair_test.partner_residue_index == None)
        assert (contact_pair_test.top == None)
        assert (contact_pair_test.topology == None)
        assert (contact_pair_test.trajs == None)

    def test_contact_pair_with_top(self):
        contact_pair_test = contact_pair(res_idxs_pair = [0, 1],
                                                  ctc_trajs = [[1.0, 1.1, 1.3]],
                                                  time_arrays = [[0, 1, 2]],
                                                  top = self.geom.top,
                                                  # trajs=None,
                                                  # fragment_idxs=None,
                                                  # fragment_names=None,
                                                  # fragment_colors=None,
                                                  # anchor_residue_idx=None,
                                                  # consensus_labels=None
                                                  )
        assert (contact_pair_test.residue_names == ['GLU30', 'VAL31'])
        assert (contact_pair_test.residue_names_short == ['E30', 'V31'])



if __name__ == '__main__':
    unittest.main()

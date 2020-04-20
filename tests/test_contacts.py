
import mdtraj as md
import unittest
from unittest.mock import Mock
import numpy as _np
import mock
from scipy.spatial.distance import cdist
from filenames import filenames
import pytest
from mdciao.contacts import select_and_report_residue_neighborhood_idxs, \
    trajs2ctcs, \
    auto_format_fragment_string, \
    ContactPair, \
    per_traj_ctc, \
    geom2COMxyz, \
    geom2COMdist, \
    _atom_type, \
    _sum_ctc_freqs_by_atom_type

from mdciao.contacts import _Fragments, \
    _Residues, _NeighborhoodNames, \
    _TimeTraces, _ContactLabels

test_filenames = filenames()

class TestBaseClassContacs(unittest.TestCase):
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

class Test_per_traj_ctc(TestBaseClassContacs):
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
        __, __, iatoms = per_traj_ctc(self.top, self.file_xtc, self.ctc_idxs, 1000, 1, 0)
        _np.testing.assert_allclose(iatoms, self.my_idxs)

    def test_scheme_COM(self):
        test_COM = geom2COMdist(self.traj[:10], residue_pairs=self.ctc_idxs)
        ctcs, times, iatoms = per_traj_ctc(self.top, self.traj[:10], self.ctc_idxs, 1000, 1, 0, scheme="COM")
        # Im using my own routine here bc we're not testing that it gets it right (which is tested in test_geomcomdist
        # I am just testing that it wraps the method correctly
        _np.testing.assert_allclose(test_COM, ctcs)
        _np.testing.assert_allclose(times, self.traj.time[:10])
        assert iatoms.shape[0]==10
        assert iatoms.shape[1]==2*2
        assert all([_np.isnan(ii) for ii in iatoms.flatten()]  )

class Test_COM_utils(TestBaseClassContacs):

    def setUp(self):
        super(Test_COM_utils,self).setUp()
        self.traj_5_frames = self.traj[:5]
        # Very slow, but what other way of directly computing the COM is there?
        COMS_mdtraj = [md.compute_center_of_mass(self.traj_5_frames.atom_slice([aa.index for aa in rr.atoms])) for rr in
                       self.top.residues]
        # re order along the time axis
        self.COMS_mdtraj = _np.zeros((5, self.top.n_residues, 3))
        for ii in range(5):
            self.COMS_mdtraj[ii,:, :] = [jcom[ii] for jcom in COMS_mdtraj]

    def test_COMxyz_works(self):
        COMSs_mine = geom2COMxyz(self.traj_5_frames)
        _np.testing.assert_allclose(self.COMS_mdtraj, COMSs_mine)

    def test_COMxyz_works_some_residues(self):
        residue_idxs = [1, 3, 5, 7]
        COMSs_mine = geom2COMxyz(self.traj_5_frames, residue_idxs=residue_idxs)

        _np.testing.assert_allclose(COMSs_mine[:, [residue_idxs]],
                                    self.COMS_mdtraj[:,[residue_idxs]])

    def test_COMdist_works(self):
        res_pairs = [[0,10], [10,20]]
        Dref = _np.vstack((_np.linalg.norm(self.COMS_mdtraj[:,0]-self.COMS_mdtraj[:,10], axis=1),
                           _np.linalg.norm(self.COMS_mdtraj[:,10]- self.COMS_mdtraj[:,20], axis=1))).T
        COMdist =  geom2COMdist(self.traj_5_frames, residue_pairs=res_pairs)
        _np.testing.assert_allclose(Dref, COMdist)

class Test_trajs2ctcs(TestBaseClassContacs):

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

class TestBaseClassForTestingAttributes(unittest.TestCase):
    def setUp(self):
        self.trajs = md.load(test_filenames.run1_stride_100_xtc, top=test_filenames.prot1_pdb)[:3]
        self.trajs = [self.trajs[:2],
                      self.trajs[:3]]
        self.ctc_trajs = [[1, 2], [10, 11, 12]]
        self.atom_pair_trajs = [[[100, 200],[100, 201]],
                                [[101, 200],[100, 200], [100, 201]]]

class Test_TimeTraces(TestBaseClassForTestingAttributes):

    def test_works(self):
        cott = _TimeTraces(self.ctc_trajs, [self.trajs[0].time,
                                            self.trajs[1].time],
                           self.trajs, self.atom_pair_trajs)
        [_np.testing.assert_allclose(itraj, jtraj) for itraj, jtraj in zip(cott.ctc_trajs, self.ctc_trajs)]
        [_np.testing.assert_allclose(itraj, jtraj) for itraj, jtraj in zip(cott.feat_trajs, self.ctc_trajs)]
        [_np.testing.assert_allclose(itraj, jtraj) for itraj, jtraj in zip(cott.time_trajs, [self.trajs[0].time,
                                                                                             self.trajs[1].time])]

        [_np.testing.assert_allclose(itraj, jtraj) for itraj, jtraj in zip(cott.atom_pair_trajs, self.atom_pair_trajs)]
        assert all([itraj==jtraj for itraj, jtraj in zip(cott.trajs, self.trajs)])

    def test_fails_because_wrong_atom_trajs(self):
        with pytest.raises(AssertionError):
            _TimeTraces(self.ctc_trajs, [self.trajs[0].time,
                                         self.trajs[1].time],
                           self.trajs, [[[0,1]],[0,1]])

        with pytest.raises(AssertionError):
            _TimeTraces(self.ctc_trajs, [self.trajs[0].time,
                                         self.trajs[1].time],
                           self.trajs, [[[0,1,1]],[0,1]])

class Test_NumberOfThings(TestBaseClassForTestingAttributes):

    def test_works(self):
        from mdciao.contacts import _NumberOfthings
        cont = _NumberOfthings(len(self.ctc_trajs),
                               [len(itraj) for itraj in self.ctc_trajs]
                               )
        _np.testing.assert_equal(len(self.ctc_trajs), cont.n_trajs)
        _np.testing.assert_allclose([len(itraj) for itraj in self.ctc_trajs], cont.n_frames)
        _np.testing.assert_equal(_np.sum([len(itraj) for itraj in self.ctc_trajs]), cont.n_frames_total)

class Test_Residues(unittest.TestCase):


    def test_just_works(self):
        cors = _Residues([0, 1], ["GLU25", "ALA35"])

        _np.testing.assert_allclose([0,1], cors.idxs_pair)
        assert cors.names[0]=="GLU25" and cors.names[1]=="ALA35"
        assert cors.names_short[0]=="E25" and cors.names_short[1]=="A35", cors.names_short

        _np.testing.assert_equal(None, cors.anchor_index)
        _np.testing.assert_equal(None, cors.anchor_residue)
        _np.testing.assert_equal(None, cors.anchor_residue_index)

        _np.testing.assert_equal(None, cors.partner_index)
        _np.testing.assert_equal(None, cors.partner_residue)
        _np.testing.assert_equal(None, cors.partner_residue_index)

    def test_anchor_and_partner(self):
        cors = _Residues([10, 20],
                         ["GLU25", "ALA35"],
                         anchor_residue_idx=10
                         )

        _np.testing.assert_equal(10, cors.anchor_residue_index)
        _np.testing.assert_equal(0, cors.anchor_index)

        _np.testing.assert_equal(1, cors.partner_index)
        _np.testing.assert_equal(20, cors.partner_residue_index)

    def test_anchor_and_partner_top(self):
        top = md.load(test_filenames.prot1_pdb).top
        cors = _Residues([10, 20],
                         ["GLU25", "ALA35"],
                         anchor_residue_idx=10,
                         top = top
                         )

        _np.testing.assert_equal(10, cors.anchor_residue_index)
        _np.testing.assert_equal(0, cors.anchor_index)
        assert top.residue(10) is cors.anchor_residue

        _np.testing.assert_equal(1, cors.partner_index)
        _np.testing.assert_equal(20, cors.partner_residue_index)
        assert top.residue(20) is cors.partner_residue

    def test_names(self):
        cors = _Residues([10, 20],
                         ["GLU25", "ALA35"]
                         )
        assert cors.names[0] == "GLU25"
        assert cors.names[1] == "ALA35"

        assert cors.names_short[0] == "E25"
        assert cors.names_short[1] == "A35"

    def test_consensus_labels(self):
        cors = _Residues([10, 20],
                         ["GLU25", "ALA35"],
                         consensus_labels=["3.50","4.50"])
        assert cors.consensus_labels[0]=="3.50"
        assert cors.consensus_labels[1]=="4.50"


class Test_Fragments(unittest.TestCase):


    def test_just_works_empty(self):
        cof = _Fragments()
        assert cof.idxs is None
        assert cof.colors is None
        assert cof.names[0]==cof.names[1] == None

    def test_works(self):
        cof = _Fragments(fragment_idxs=[0, 1],
                         fragment_colors=["r","b"],
                         fragment_names=["fragA", "fragB"])
        _np.testing.assert_allclose([0,1], cof.idxs)

        assert cof.colors[0]=="r" and cof.colors[1]=="b"
        assert cof.names[0]=="fragA" and cof.names[1]=="fragB"

    def test_auto_fragnaming(self):
        cof = _Fragments(fragment_idxs=[0, 1],
                         )

        assert cof.names[0] == str(0) and cof.names[1] == str(1)


    def _test_anchor_and_partner(self):
        cors = _Residues([10, 20],
                         anchor_residue_idx=10
                         )

        _np.testing.assert_equal(10, cors.anchor_residue_index)
        _np.testing.assert_equal(0, cors.anchor_index)

        _np.testing.assert_equal(1, cors.partner_index)
        _np.testing.assert_equal(20, cors.partner_residue_index)

    def _test_anchor_and_partner_top(self):
        from mdciao.contacts import _Residues
        top = md.load(test_filenames.prot1_pdb).top
        cors = _Residues([10, 20],
                         anchor_residue_idx=10,
                         top = top
                         )

        _np.testing.assert_equal(10, cors.anchor_residue_index)
        _np.testing.assert_equal(0, cors.anchor_index)
        assert top.residue(10) is cors.anchor_residue

        _np.testing.assert_equal(1, cors.partner_index)
        _np.testing.assert_equal(20, cors.partner_residue_index)
        assert top.residue(20) is cors.partner_residue

    def _test_names(self):
        from mdciao.contacts import _Residues
        cors = _Residues([10, 20],
                         ["GLU25", "ALA35"]
                         )
        assert cors.names[0] == "GLU25"
        assert cors.names[1] == "ALA35"

        assert cors.names_short[0] == "E10"
        assert cors.names_short[1] == "A20"

class Test_NeighborhoodNames(unittest.TestCase):

    def test_works(self):
        cnns = _NeighborhoodNames(
            _Residues([10, 20],
                      ["GLU25", "ALA35"],
                      anchor_residue_idx=20
                      ),
            _Fragments([0, 1],
                       ["fragA", "fragB"],
                       ["r", "b"]
                       )
        )


        assert cnns.anchor_fragment == "fragB"
        assert cnns.partner_fragment == "fragA"

    def test_raises(self):
        with pytest.raises(AssertionError):
            _NeighborhoodNames(_Residues([10, 20],["GLU25", "ALA35"]),
                               _Fragments([0, 1],
                                          ["fragA", "fragB"],
                                          ["r", "b"]
                                          ))

    def test_fragments_consensus_name_None(self):
        cnns = _NeighborhoodNames(
            _Residues([10, 20],
                      ["GLU25", "ALA35"],
                      anchor_residue_idx=20
                      ),
            _Fragments([0, 1],
                       )
        )
        assert cnns.partner_fragment_consensus is None
        assert cnns.anchor_fragment_consensus is None

    def test_fragments_consensus_name(self):
        cnns = _NeighborhoodNames(
            _Residues([10, 20],
                      ["GLU25", "ALA35"],
                      anchor_residue_idx=20,
                      consensus_labels=["3.50","4.50"]
                      ),
            _Fragments([0, 1],
                       ["fragA", "fragB"],
                       )
        )
        assert cnns.partner_fragment_consensus == "3.50"
        assert cnns.anchor_fragment_consensus == "4.50"

    def test_fragments_names_best_no_consensus(self):
        cnns = _NeighborhoodNames(
            _Residues([10, 20],
                      ["GLU25", "ALA35"],
                      anchor_residue_idx=20
                      ),
            _Fragments([0, 1],
                       fragment_names=["fragA", "fragB"]
                       )
        )
        assert cnns.partner_fragment_best == "fragA", cnns.partner_fragment_best
        assert cnns.anchor_fragment_best == "fragB"

    def test_fragments_names_best_w_consensus(self):
        cnns = _NeighborhoodNames(
            _Residues([10, 20],
                      ["GLU25", "ALA35"],
                      anchor_residue_idx=20,
                      consensus_labels=["3.50","4.50"]
                      ),
            _Fragments([0, 1],
                       fragment_names=["fragA", "fragB"]
                       )
        )
        assert cnns.partner_fragment_best == "3.50"
        assert cnns.anchor_fragment_best == "4.50"

    def test_res_and_fragment_strs_no_consensus(self):
        cnns = _NeighborhoodNames(
            _Residues([10, 20],
                      ["GLU25", "ALA35"],
                      anchor_residue_idx=20,
                      #consensus_labels=["3.50","4.50"]
                      ),
            _Fragments([0, 1],
                       fragment_names=["fragA", "fragB"]
                       )
        )
        assert cnns.anchor_res_and_fragment_str == "ALA35@fragB"
        assert cnns.partner_res_and_fragment_str == "GLU25@fragA"

        assert cnns.anchor_res_and_fragment_str_short == "A35@fragB"
        assert cnns.partner_res_and_fragment_str_short == "E25@fragA"

    def test_res_and_fragment_strs_w_consensus(self):
        cnns = _NeighborhoodNames(
            _Residues([10, 20],
                      ["GLU25", "ALA35"],
                      anchor_residue_idx=20,
                       consensus_labels=["3.50","4.50"]
                      ),
            _Fragments([0, 1],
                       fragment_names=["fragA", "fragB"]
                       )
        )
        assert cnns.anchor_res_and_fragment_str == "ALA35@4.50"
        assert cnns.partner_res_and_fragment_str == "GLU25@3.50"

        assert cnns.anchor_res_and_fragment_str_short == "A35@4.50"
        assert cnns.partner_res_and_fragment_str_short == "E25@3.50"

class Test_auto_fragment_string(unittest.TestCase):

    def test_both_bad(self):
        assert auto_format_fragment_string(None,None)==""

    def test_both_good(self):
        assert auto_format_fragment_string("fragA","3.50")=="3.50"

    def test_only_option(self):
        assert auto_format_fragment_string("fragA",None)=="fragA", auto_format_fragment_string("fragA",None)

    def test_only_better_option(self):
        assert auto_format_fragment_string(None, "3.50")=="3.50"



class Test_ContactLabels(unittest.TestCase):

    def test_trajlabels_no_trajs(self):
        cls = _ContactLabels(2,
                             _Residues([10, 20],["GLU25", "ALA35"])
                             )

        assert cls.trajlabels[0]=="traj 0" and cls.trajlabels[1]=="traj 1", cls.trajlabels

    def test_trajlabels_w_mdtrajs(self):
        mdtrajs = md.load(test_filenames.run1_stride_100_xtc,
                               top=test_filenames.prot1_pdb)[:5]
        mdtrajs = [mdtrajs, mdtrajs]
        cls = _ContactLabels(2,
                             _Residues([10, 20], ["GLU25", "ALA35"]),
                             trajs = mdtrajs
                             )

        assert cls.trajlabels[0] == "mdtraj.00" and cls.trajlabels[1] == "mdtraj.01", cls.trajlabels

    def test_trajlabels_wo_mdtrajs(self):
        cls = _ContactLabels(2,
                             _Residues([10, 20], ["GLU25", "ALA35"]),
                             trajs=["file0.xtc", "file1.xtc"]
                             )

        assert cls.trajlabels[0] == "file0" and cls.trajlabels[1] == "file1", cls.trajlabels

    def test_ctc_labels(self):
        cls = _ContactLabels(2,
                             _Residues([10, 20], ["GLU25", "ALA35"]),
                             _Fragments(fragment_names=["fragA","fragB"])
                             )
        assert cls.ctc_label_w_fragments=="GLU25@fragA-ALA35@fragB", cls.ctc_label_w_fragments
        assert cls.ctc_label_short_AA_w_fragments=="E25@fragA-A35@fragB", cls.ctc_label_short_AA_w_fragments

    def test_ctc_label_no_fragments(self):
        cls = _ContactLabels(2,
                             _Residues([10, 20], ["GLU25", "ALA35"]),
                             )

        assert cls.ctc_label_no_fragments == "GLU25-ALA35", cls.ctc_label_no_fragments
        assert cls.ctc_label_no_fragments_short_AA == "E25-A35"


    def test_ctc_label_missing_frags_and_consensus(self):
        cls = _ContactLabels(2,
                             _Residues([10, 20], ["GLU25", "ALA35"]),
                             )

        assert cls.ctc_label_w_fragments == "GLU25-ALA35", cls.ctc_label_w_fragment
        assert cls.ctc_label_short_AA_w_fragments == "E25-A35", cls.ctc_label_short_AA_w_fragments

    def test_just_prints(self):
        cls = _ContactLabels(2,
                             _Residues([10, 20], ["GLU25", "ALA35"]),
                             )
        print(cls)



class Test_contact_pair(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)

    def test_works_minimal(self):
        ContactPair([0, 1],
                    [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                    [[0, 1, 2], [0, 1, 2, 3]],
                    )

    def test_with_top(self):
        contact_pair_test =  ContactPair([0, 1],
                    [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                    [[0, 1, 2], [0, 1, 2, 3]],
                                         top=self.geom.top,
                    )
        contact_pair_test.top == self.geom.top == contact_pair_test.topology

    def test_with_anchor(self):
        contact_pair_test = ContactPair([0, 1],
                                        [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                        [[0, 1, 2], [0, 1, 2, 3]],
                                        anchor_residue_idx=1,
                                        )
        contact_pair_test.top == self.geom.top
        assert contact_pair_test.neighborhood is not None

    def test_all_properties_w_empty_ones_just_runs(self):
        cpt = ContactPair([0, 1],
                                        [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                        [[0, 1, 2], [0, 1, 2, 3]]
                                        )
        cpt.time_traces
        cpt.n
        cpt.residues
        cpt.fragments
        cpt.labels
        cpt.label
        assert cpt.time_max==3
        assert cpt.neighborhood is None

    def test_binarize_trajs(self):
        cpt = ContactPair([0, 1],
                          [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                          [[0, 1, 2], [0, 1, 2, 3]]
                          )
        bintrajs = cpt.binarize_trajs(21)
        _np.testing.assert_array_equal([1, 1, 1],bintrajs[0])
        _np.testing.assert_array_equal([1, 1, 0, 0], bintrajs[1])

    def test_freq_per_traj(self):
        cpt = ContactPair([0, 1],
                          [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                          [[0, 1, 2], [0, 1, 2, 3]]
                          )
        freqs = cpt.frequency_per_traj(21)
        _np.testing.assert_equal(freqs[0],1)
        _np.testing.assert_equal(freqs[1],.5)

    def test_freq_overall_trajs(self):
        cpt = ContactPair([0, 1],
                          [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                          [[0, 1, 2], [0, 1, 2, 3]]
                          )
        _np.testing.assert_equal(cpt.frequency_overall_trajs(21),
                                 _np.mean([1, 1, 1]+[1, 1, 0, 0]))

    def test_freqency_dict_no_labels(self):
        cpt = ContactPair([0, 1],
                          [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                          [[0, 1, 2], [0, 1, 2, 3]]
                          )

        idict = cpt.frequency_dict(21)
        assert idict["freq"] == _np.mean([1, 1, 1]+[1, 1, 0, 0])
        assert idict["residue idxs"] == "0 1"
        assert idict["label"] == '%-15s - %-15s'%(0,1),idict["label"]

        idict = cpt.frequency_dict(21,AA_format="long")
        assert idict["label"] == '%-15s - %-15s' %(0,1), idict["label"]

        idict = cpt.frequency_dict(21, lb_format="join")
        assert idict["label"] == "0-1"

    def test_frequency_dict_w_labels(self):
        cpt = ContactPair([0, 1],
                          [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                          [[0, 1, 2], [0, 1, 2, 3]],
                          fragment_names=["fragA", "fragB"]
                          )
        idict = cpt.frequency_dict(21)
        assert idict["label"] == '%-15s - %-15s'%("0@fragA","1@fragB")

        idict = cpt.frequency_dict(21,AA_format="long")
        assert idict["label"] == '%-15s - %-15s'%("0@fragA","1@fragB")
        idict = cpt.frequency_dict(21, lb_format="join")
        assert idict["label"] == '0@fragA-1@fragB'

    def test_distro_overall_trajs(self):
        cpt = ContactPair([0, 1],
                          [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                          [[0, 1, 2], [0, 1, 2, 3]],
                          )
        x, h = cpt.distro_overall_trajs(bins=10)
        xt, ht = _np.histogram([1.0, 1.1, 1.3]+[2.0, 2.1, 2.3, 2.4],10)
        _np.testing.assert_array_equal(x,xt)
        _np.testing.assert_array_equal(h, ht)

    def test_formed_atom_pairs_fails(self):
        cpt = ContactPair([0, 1],
                          [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                          [[0, 1, 2], [0, 1, 2, 3]],
                          )
        with pytest.raises((AssertionError,ValueError)):
            cpt.count_formed_atom_pairs(21)

    def test_overall_stacked_formed_atoms(self):
        cpt = ContactPair([0, 1],
                          [[1.0, 2.5, 1.3], [2.0, 2.1, 2.3, 2.4]],
                          [[0, 1, 2], [0, 1, 2, 3]],
                          atom_pair_trajs=[
                              [[10,20],[11,20],[10,21]],
                              [[11,21],[10,21],[10,20],[11,20]]
                          ]
                          )

        fas = cpt._overall_stacked_formed_atoms(21)
        _np.testing.assert_array_equal([[10,20],
                                        #[11,20], this frame is not formed
                                        [10,21],
                                        [11,21],
                                        [10,21]], fas)

    def test_count_formed_atom_pairs(self):
        cpt = ContactPair([0, 1],
                          [[1.0, 2.5, 1.3], [2.0, 2.1, 2.3, 2.4]],
                          [[0, 1, 2], [0, 1, 2, 3]],
                          atom_pair_trajs=[
                              [[10,20],[11,20],[10,21]],
                              [[10,21],[10,21],[10,20],[11,20]]
                          ]
                          )
        # The count should be [10,20]:1, [10,21]:3,
        pairs, counts = cpt.count_formed_atom_pairs(21, sort=False)
        _np.testing.assert_array_equal(pairs,[[10,20],[10,21]])
        _np.testing.assert_array_equal(counts,[1,3])

        pairs, counts = cpt.count_formed_atom_pairs(21, sort=True)
        _np.testing.assert_array_equal(pairs,[[10,21],[10,20]])
        _np.testing.assert_array_equal(counts,[3,1])

    def test_frequency_dict_formed_atom_pairs_overall_trajs_fails(self):
        cpt = ContactPair([0, 1],
                          [[1.0, 2.5, 1.3], [2.0, 2.1, 2.3, 2.4]],
                          [[0, 1, 2], [0, 1, 2, 3]],
                          atom_pair_trajs=[
                              [[10, 20], [11, 20], [10, 21]],
                              [[10, 21], [10, 21], [10, 20], [11, 20]]
                          ]
                          )
        with pytest.raises(AssertionError):
            cpt.relative_frequency_of_formed_atom_pairs_overall_trajs(21)

    def test_frequency_dict_formed_atom_pairs_overall_trajs(self):
        # Completely bogus contact but testable
        atom_BB = list(self.geom.top.residue(0).atoms_by_name("CA"))[0].index
        atom_SC = list(self.geom.top.residue(1).atoms_by_name("CB"))[0].index
        cpt = ContactPair([0, 1],
                          [[1.0, 2.5, 1.3, 1.0]],
                          [[0, 1, 2, 3]],
                          atom_pair_trajs=[
                              [[atom_BB, atom_SC],
                               [atom_SC, atom_SC],
                               [atom_BB, atom_SC],
                               [atom_BB, atom_BB]
                               ],
                            ],
                          top=self.geom.top
                          )
        out_dict = cpt.relative_frequency_of_formed_atom_pairs_overall_trajs(21)
        _np.testing.assert_equal(out_dict["BB-SC"],2/3)
        _np.testing.assert_equal(out_dict["BB-BB"],1/3)
        assert len(out_dict)==2

        out_dict = cpt.relative_frequency_of_formed_atom_pairs_overall_trajs(21, min_freq=.5)
        _np.testing.assert_equal(out_dict["BB-SC"],2/3)
        assert len(out_dict)==1

    def test_frequency_dict_formed_atom_pairs_overall_trajs_aggregate_by_atomtype_False(self):
        # Completely bogus contact but testable
        atom_BB_1 = list(self.geom.top.residue(0).atoms_by_name("CA"))[0].index
        atom_BB_2 = list(self.geom.top.residue(0).atoms_by_name("N"))[0].index
        atom_SC = list(self.geom.top.residue(1).atoms_by_name("CB"))[0].index
        cpt = ContactPair([0, 1],
                          [[1.0, 2.5, 1.3, 1.0]],
                          [[0, 1, 2, 3]],
                          atom_pair_trajs=[
                              [[atom_BB_1, atom_SC],
                               [atom_SC, atom_SC],
                               [atom_BB_2, atom_SC],
                               [atom_BB_1, atom_BB_2]
                               ],
                            ],
                          top=self.geom.top
                          )
        out_dict = cpt.relative_frequency_of_formed_atom_pairs_overall_trajs(21,
                                                                             aggregate_by_atomtype=False,
                                                                             keep_resname=True)
        _np.testing.assert_equal(out_dict["%s-%s"%(self.geom.top.atom(atom_BB_1),
                                                   self.geom.top.atom(atom_SC))],1/3)
        _np.testing.assert_equal(out_dict["%s-%s"%(self.geom.top.atom(atom_BB_2),
                                                   self.geom.top.atom(atom_SC))],1/3)
        _np.testing.assert_equal(out_dict["%s-%s" % (self.geom.top.atom(atom_BB_1),
                                                     self.geom.top.atom(atom_BB_2))], 1 / 3)

        out_dict = cpt.relative_frequency_of_formed_atom_pairs_overall_trajs(21,
                                                                             aggregate_by_atomtype=False,
                                                                             keep_resname=False)
        _np.testing.assert_equal(out_dict["%s-%s" % (self.geom.top.atom(atom_BB_1).name,
                                                     self.geom.top.atom(atom_SC).name)], 1 / 3)
        _np.testing.assert_equal(out_dict["%s-%s" % (self.geom.top.atom(atom_BB_2).name,
                                                     self.geom.top.atom(atom_SC).name)], 1 / 3)
        _np.testing.assert_equal(out_dict["%s-%s" % (self.geom.top.atom(atom_BB_1).name,
                                                     self.geom.top.atom(atom_BB_2).name)], 1 / 3)
        assert len(out_dict)==3

    def test_prints(self):
        cpt = ContactPair([0, 1],
                          [[1.0, 2.5, 1.3], [2.0, 2.1, 2.3, 2.4]],
                          [[0, 1, 2], [0, 1, 2, 3]],
                          atom_pair_trajs=[
                              [[10, 20], [11, 20], [10, 21]],
                              [[10, 21], [10, 21], [10, 20], [11, 20]]
                          ]
                          )
        print(cpt)

class Test_atom_type(unittest.TestCase):
    def test_works(self):
        top = md.load(test_filenames.prot1_pdb).top
        atoms_BB = [aa for aa in top.residue(0).atoms if aa.is_backbone]
        atoms_SC = [aa for aa in top.residue(0).atoms if aa.is_sidechain]
        atoms_X = [aa for aa in top.atoms if not aa.is_backbone and not aa.is_sidechain]
        assert len(atoms_BB)>0
        assert len(atoms_SC)>0
        assert len(atoms_X)>0
        assert all([_atom_type(aa)=="BB" for aa in atoms_BB])
        assert all([_atom_type(aa)=="SC" for aa in atoms_SC])
        assert all([_atom_type(aa)=="X" for aa in atoms_X])

class Test_sum_ctc_freqs_by_atom_type(unittest.TestCase):
    def test_works(self):
        top = md.load(test_filenames.prot1_pdb).top
        atoms_BB = [aa for aa in top.residue(0).atoms if aa.is_backbone]
        atoms_SC = [aa for aa in top.residue(0).atoms if aa.is_sidechain]
        atom_pairs, counts = [],[]
        for trip in     [[atoms_BB[0], atoms_BB[1], 5],
                         [atoms_BB[0], atoms_BB[2], 4],
                         [atoms_BB[0], atoms_SC[1], 10],
                         [atoms_BB[1], atoms_SC[1], 20],
                         [atoms_SC[0], atoms_BB[2], 3],
                         [atoms_SC[1], atoms_SC[0], 1],
                         [atoms_SC[2], atoms_SC[0], 1]]:
            atom_pairs.append(trip[:2])
            counts.append(trip[2])

        dict_out = _sum_ctc_freqs_by_atom_type(atom_pairs, counts)

        assert dict_out["BB-BB"] == 9
        assert dict_out["BB-SC"] == 30
        assert dict_out["SC-BB"] == 3
        assert dict_out["SC-SC"] == 2
        assert len(dict_out)==4

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

class Test_pick_best_label(unittest.TestCase):
    def test_pick_best_label_just_works(self):
        assert (auto_format_fragment_string(option="Print this instead", better_option="Print this") == "Print this")

    def test_pick_best_label_exclude_works(self):
        assert(auto_format_fragment_string(option="Print this instead", better_option= None) == "Print this instead")
        assert(auto_format_fragment_string(option="Print this instead", better_option="None") == "Print this instead")
        assert(auto_format_fragment_string(option="Print this instead", better_option="NA") == "Print this instead")
        assert(auto_format_fragment_string(option="Print this instead", better_option="na") == "Print this instead")

class Test_contact_group(unittest.TestCase):
    pass




if __name__ == '__main__':
    unittest.main()

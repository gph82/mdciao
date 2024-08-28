# Note: I am testing a bunch of private methods here, and
# that is not best practice, see:
# https://softwareengineering.stackexchange.com/questions/380287/why-is-unit-testing-private-methods-considered-as-bad-practice
# https://softwareengineering.stackexchange.com/questions/375860/how-to-avoid-the-need-to-unit-test-private-methods
# https://stackoverflow.com/questions/105007/should-i-test-private-methods-or-only-public-ones/47401015#47401015
# ATM the "best practice" is to get the code tested and re-design when the API is more stable
from mdciao.contacts.contacts import \
    _sum_ctc_freqs_by_atom_type, \
    _NeighborhoodNames, \
    _NumberOfthings, \
    Residues, \
    _ContactStrings, \
    _TimeTraces, \
    _Fragments, \
    _linear_switchoff, \
    _delta_freq_pairs

from itertools import combinations as _combinations

import mdtraj as md
import unittest
from unittest.mock import Mock
import numpy as _np
from unittest import mock
from os import path
from scipy.spatial.distance import cdist
from mdciao.examples import filenames as test_filenames
from mdciao import contacts
from mdciao import examples
from mdciao import nomenclature
from mdciao.cli import sites as _mdcsites
from pandas import DataFrame as _DF
import pickle
import sys as _sys, platform as _platform

import io as _io, contextlib as _contextlib

from mdciao.fragments import get_fragments
from mdciao.utils.residue_and_atom import residues_from_descriptors

from matplotlib import pyplot as _plt

from tempfile import TemporaryDirectory as _TDir, NamedTemporaryFile as _NamedTfile, TemporaryFile as _TFil

import mdciao.utils.COM as mdcCOM

from mdciao import cli as _mdcli
from mdciao import utils as _mdcu



#TODO break this up by object type? Testfile is huge
class TestBaseClassContacts(unittest.TestCase):
    def setUp(self):
        self.pdb_file = test_filenames.top_pdb
        self.file_xtc = test_filenames.traj_xtc_stride_20
        self.top = md.load(self.pdb_file).top
        self.traj = md.load(self.file_xtc, top=self.top)
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

class Test_per_traj_ctc(TestBaseClassContacts):
    def test_contacts_file(self):
        ctcs, time, __ = contacts.per_traj_ctc(self.top, self.file_xtc, self.ctc_idxs, 1000, 1, 0)
        _np.testing.assert_allclose(ctcs, self.ctcs)
        _np.testing.assert_allclose(time, self.traj.time)

    def test_contacts_geom(self):
        ctcs, time, __ = contacts.per_traj_ctc(self.top, self.traj, self.ctc_idxs, 1000, 1, 0)
        _np.testing.assert_allclose(ctcs, self.ctcs)
        _np.testing.assert_allclose(time, self.traj.time)

    def test_contacts_geom_stride(self):
        ctcs, time, __ = contacts.per_traj_ctc(self.top, self.traj, self.ctc_idxs, 1000, 2, 0)
        _np.testing.assert_allclose(ctcs, self.ctcs[::2])
        _np.testing.assert_allclose(time, self.traj.time[::2])

    def test_contacts_geom_chunk(self):
        ctcs, time, __ = contacts.per_traj_ctc(self.top, self.traj, self.ctc_idxs, 5, 1, 0)
        _np.testing.assert_allclose(ctcs, self.ctcs)
        _np.testing.assert_allclose(time, self.traj.time)

    def test_atoms(self):
        __, __, iatoms = contacts.per_traj_ctc(self.top, self.file_xtc, self.ctc_idxs, 1000, 1, 0)
        _np.testing.assert_allclose(iatoms, self.my_idxs)

    def test_scheme_COM(self):
        test_COM = mdcCOM.geom2COMdist(self.traj[:10], residue_pairs=self.ctc_idxs)
        ctcs, times, iatoms = contacts.per_traj_ctc(self.top, self.traj[:10], self.ctc_idxs, 1000, 1, 0, scheme="COM")
        # Im using my own routine here bc we're not testing that it gets it right (which is tested in test_geomcomdist
        # I am just testing that it wraps the method correctly
        _np.testing.assert_allclose(test_COM, ctcs)
        _np.testing.assert_allclose(times, self.traj.time[:10])
        assert iatoms.shape[0] == 10
        assert iatoms.shape[1] == 2 * 2
        assert all([_np.isnan(ii) for ii in iatoms.flatten()])

    def test_contacts_geom_1_frame(self):
        ctcs, time, __ = contacts.per_traj_ctc(self.top, self.traj[0], self.ctc_idxs, 1000, 1, 0)
        _np.testing.assert_allclose(ctcs, self.ctcs[:1, :])
        _np.testing.assert_allclose(time, self.traj.time[:1])


class Test_trajs2ctcs(TestBaseClassContacts):

    def setUp(self):
        # TODO read why I shouldn't be doing this...https://nedbatchelder.com/blog/201210/multiple_inheritance_is_hard.html
        super(Test_trajs2ctcs, self).setUp()
        self.xtcs = [self.file_xtc, self.file_xtc]
        self.ctcs_stacked = _np.vstack([self.ctcs, self.ctcs])
        self.times_stacked = _np.hstack([self.traj.time, self.traj.time])
        self.atoms_stacked = _np.vstack([self.my_idxs, self.my_idxs])

    def test_works(self):
        ctcs_trajs_consolidated = contacts.trajs2ctcs(self.xtcs, self.top, self.ctc_idxs)
        _np.testing.assert_allclose(ctcs_trajs_consolidated, self.ctcs_stacked)

    def test_return_time_and_atoms(self):
        ctcs_trajs_consolidated, times_consolidated, atoms_consolidated = contacts.trajs2ctcs(self.xtcs, self.top,
                                                                                              self.ctc_idxs,
                                                                                              return_times_and_atoms=True
                                                                                              )
        _np.testing.assert_allclose(self.times_stacked, times_consolidated)
        _np.testing.assert_allclose(self.ctcs_stacked, ctcs_trajs_consolidated)
        _np.testing.assert_allclose(self.atoms_stacked, atoms_consolidated)

    def test_consolidate_is_false(self):
        ctcs, times, atoms = contacts.trajs2ctcs(self.xtcs, self.top, self.ctc_idxs,
                                                 return_times_and_atoms=True,
                                                 consolidate=False)

        [_np.testing.assert_equal(itraj, jtraj) for (itraj, jtraj) in zip([self.ctcs, self.ctcs], ctcs)]
        [_np.testing.assert_equal(itraj, jtraj) for (itraj, jtraj) in zip([self.traj.time, self.traj.time], times)]
        [_np.testing.assert_equal(itraj, jtraj) for (itraj, jtraj) in zip([self.my_idxs, self.my_idxs], atoms)]

    def test_progressbar(self):
        ctcs_trajs_consolidated = contacts.trajs2ctcs(self.xtcs, self.top, self.ctc_idxs, progressbar=True)

    def test_one_traj_one_frame_pdb_just_runs(self):
        contacts.trajs2ctcs([self.pdb_file], self.top, self.ctc_idxs)


class Test_per_traj_mindist_lower_bound_wo_periodic(unittest.TestCase):

    # We're repeating the same tests as Test_geom2max_residue_radius wrapped
    # wrapped in the functionality of per_traj_mindist_lower_bound
    def setUp(self) -> None:
        pdb = "CRYST1   89.862   89.862  142.612  90.00  90.00  90.00 P 1           1 \n" \
              "MODEL        0 \n" \
              "ATOM      1  CA  GLU A  30      0.0000  0.0000  0.0000  1.00  0.00           C \n" \
              "ATOM      2  CB  GLU A  30      2.0000  0.0000  0.0000  1.00  0.00           C \n" \
              "ATOM      3  C   GLU A  30      10.000  0.0000  0.0000  1.00  0.00           C \n" \
              "ATOM      4  CA  VAL A  31      0.0000  0.0000  0.0000  1.00  0.00           C \n" \
              "ATOM      5  CB  VAL A  31      0.0000  5.0000  0.0000  1.00  0.00           C \n" \
              "ATOM      6  C   VAL A  31      0.0000  10.000  0.0000  1.00  0.00           C \n" \
              "ATOM      7  CA  TRP A  32      0.0000  0.0000  0.0000  1.00  0.00           C \n" \
              "ATOM      8  CB  TRP A  32      0.0000  0.0000  8.0000  1.00  0.00           C \n" \
              "ATOM      9  C   TRP A  32      0.0000  0.0000  10.000  1.00  0.00           C \n" \
              "TER      10      TRP A  32 "
        with _NamedTfile(suffix=".pdb") as tf:
            with open(tf.name, "w") as f:
                f.write(pdb)
            self.geom = md.load(tf.name)
            self.geom = self.geom.join(self.geom)
            self.geom._xyz[1, :, :] *= 10
        # With the above geometry, it's easy to see that the COMs are, for x,y,z respectively
        # GLU30 (0+2+10)/3 = 4
        # VAL31 (0+5+10)/3 = 5
        # TRP31 (0+8+10)/3 = 6
        # And then the radii are, respectively
        # GLU30 10 - 4 = 6
        # VAL31 10 - 5 = 5 or 0 - 5 = 5
        # TRP31  0 - 6 = -6 in abs 6
        # Note that we have to multiply x .1 to be back in nm (pdb strg is in Ang)
        # and that the second frame is the first multiplied by 10

        # Check Test_geom2max_residue_radius on why these values
        self.COMs = _np.array([[[0.4, 0., 0., ],
                                [0., 0.5, 0.],
                                [0., 0., 0.6]],
                               [[4., 0., 0.],
                                [0., 5., 0.],
                                [0., 0., 6.]]])
        self.maxRr = _np.array([6., 5., 6.])
        self.COMd = _np.array([[0.64031242, 0.72111026, 0.78102497],
                               [6.40312424, 7.21110255, 7.81024968]])
        self.sum_maxR = _np.array([[6+5],[6+6],[5+6]]).squeeze()
        self.lower_bound_t = self.COMd - self.sum_maxR
        #print()
        #print(self.lower_bound_t.round(2))
    def test_works(self):
        lower_bounds = contacts.per_traj_mindist_lower_bound(self.geom.top, self.geom, [[0,1],[0,2],[1,2]], 1000, 1, 0, periodic=False)
        _np.testing.assert_array_almost_equal(lower_bounds, self.lower_bound_t.min(axis=0))

    def test_works_timetrace(self):
        lower_bounds_t = contacts.per_traj_mindist_lower_bound(self.geom.top, self.geom, [[0, 1], [0, 2], [1, 2]],
                                                               1000, 1, 0,
                                                               timetrace=True, periodic=False)

        _np.testing.assert_array_almost_equal(lower_bounds_t, self.lower_bound_t)

    def test_works_timetrace_lb_cutoff_Ang(self):
        lower_bounds_t_bool = contacts.per_traj_mindist_lower_bound(self.geom.top, self.geom, [[0, 1], [0, 2], [1, 2]],
                                                                    1000, 1, 0,
                                                                    timetrace=True,
                                                                    lb_cutoff_Ang=-110 , #it's weird it's negative but it's okay for tests,
                                                                    periodic = False
                                                                    )
        # We can evaluate the expressions above to these numbers
        ref_lower_bounds_t = _np.array([[-10.36 - 11.28 - 10.22],
                                        [-4.6 - 4.79 - 3.19]]
                                       )
        _np.testing.assert_array_almost_equal(lower_bounds_t_bool, [1])

    # Since trajs2lower_bounds is mainly a wrapper on per_traj_mindist_lower_bound
    #  i'd rather test it here than have its own class
    def test_trajs2lower_bounds(self):
        list_of_lbs = contacts.trajs2lower_bounds([self.geom, self.geom[::-1]],
                                                  self.geom.top, [[0, 1], [0, 2], [1, 2]],
                                                  periodic=False)

        _np.testing.assert_array_almost_equal(self.lower_bound_t.min(axis=0), list_of_lbs[0])
        _np.testing.assert_array_almost_equal(self.lower_bound_t.min(axis=0), list_of_lbs[1])

    def test_trajs2lower_bounds_timetrace(self):
        list_of_lbs = contacts.trajs2lower_bounds([self.geom, self.geom[::-1]],
                                                  self.geom.top, [[0, 1], [0, 2], [1, 2]],
                                                  periodic=False, timetrace=True)

        _np.testing.assert_array_almost_equal(self.lower_bound_t, list_of_lbs[0])
        _np.testing.assert_array_almost_equal(self.lower_bound_t[::-1], list_of_lbs[1])

    def test_trajs2lower_bounds_cutoff(self):
        list_of_lbs = contacts.trajs2lower_bounds([self.geom, self.geom[::-1]],
                                                  self.geom.top, [[0, 1], [0, 2], [1, 2]],
                                                  periodic=False,
                                                  lb_cutoff_Ang=-103) #it's weird it's negative but it's okay for tests

        _np.testing.assert_array_almost_equal([0,1], list_of_lbs[0])
        _np.testing.assert_array_almost_equal([0,1], list_of_lbs[1])

class Test_per_traj_mindist_lower_bound_actual_lb(unittest.TestCase):

    def test_all_lbs_are_smaller_than_actual_distances(self):
        geom=md.load(test_filenames.traj_xtc_stride_20, top=test_filenames.top_pdb)
        all_pairs = list(_combinations(range(geom.n_residues), 2))
        d = md.compute_contacts(geom, all_pairs, ignore_nonprotein=False)[0]
        lower_bounds = contacts.per_traj_mindist_lower_bound(geom.top, geom, all_pairs, 100, 1, 0)
        assert (lower_bounds<d).all()

class Test_per_traj_mindist_lower_bound_w_periodic(unittest.TestCase):

    # We're repeating the same tests as Test_geom2max_residue_radius wrapped
    # wrapped in the functionality of per_traj_mindist_lower_bound

    #
    def setUp(self) -> None:
        pdb = "CRYST1   101.00   101.00  101.000  90.00  90.00  90.00 P 1           1 \n" \
              "MODEL        0 \n" \
              "ATOM      1  CA  GLU A  30      0.0000  0.0000  0.0000  1.00  0.00           C \n" \
              "ATOM      2  CB  GLU A  30      2.0000  0.0000  0.0000  1.00  0.00           C \n" \
              "ATOM      3  C   GLU A  30      10.000  0.0000  0.0000  1.00  0.00           C \n" \
              "ATOM      4  CA  VAL A  31      0.0000  0.0000  0.0000  1.00  0.00           C \n" \
              "ATOM      5  CB  VAL A  31      0.0000  5.0000  0.0000  1.00  0.00           C \n" \
              "ATOM      6  C   VAL A  31      0.0000  10.000  0.0000  1.00  0.00           C \n" \
              "ATOM      7  CA  TRP A  32      0.0000  0.0000  0.0000  1.00  0.00           C \n" \
              "ATOM      8  CB  TRP A  32      0.0000  0.0000  8.0000  1.00  0.00           C \n" \
              "ATOM      9  C   TRP A  32      0.0000  0.0000  10.000  1.00  0.00           C \n" \
              "TER      10      TRP A  32 "
        with _NamedTfile(suffix=".pdb") as tf:
            with open(tf.name, "w") as f:
                f.write(pdb)
            self.geom = md.load(tf.name)
            self.geom = self.geom.join(self.geom)
            self.geom._xyz[1, :, :] *= 10

        # With periodic cells above (101,101,101), the unwrapping of residues produces
        # has an effec on the second frame, where coords have been *= 10. There,
        # some atoms approach the boundary and hence are considered "split across PBCs"
        # or "wrapped", s.t. the unwrapping changes their position:
        self.geom_unwrapped =_mdcu.COM._per_residue_unwrapping(self.geom)
        """
        ATOM      1  CA  GLU A  30       0.000   0.000   0.000  1.00  0.00           C
        ATOM      2  CB  GLU A  30       2.000   0.000   0.000  1.00  0.00           C
        ATOM      3  C   GLU A  30      10.000   0.000   0.000  1.00  0.00           C
        ATOM      4  CA  VAL A  31       0.000   0.000   0.000  1.00  0.00           C
        ATOM      5  CB  VAL A  31       0.000   5.000   0.000  1.00  0.00           C
        ATOM      6  C   VAL A  31       0.000  10.000   0.000  1.00  0.00           C
        ATOM      7  CA  TRP A  32       0.000   0.000   0.000  1.00  0.00           C
        ATOM      8  CB  TRP A  32       0.000   0.000   8.000  1.00  0.00           C
        ATOM      9  C   TRP A  32       0.000   0.000  10.000  1.00  0.00           C
        TER      10      TRP A  32
        ENDMDL
        MODEL        1
        ATOM      1  CA  GLU A  30       0.000   0.000   0.000  1.00  0.00           C
        ATOM      2  CB  GLU A  30      20.000   0.000   0.000  1.00  0.00           C
        ATOM      3  C   GLU A  30      -1.000   0.000   0.000  1.00  0.00           C #100 was changed to -1 (100 - 101 = 1)
        ATOM      4  CA  VAL A  31       0.000   0.000   0.000  1.00  0.00           C
        ATOM      5  CB  VAL A  31       0.000  50.000   0.000  1.00  0.00           C
        ATOM      6  C   VAL A  31       0.000 100.000   0.000  1.00  0.00           C
        ATOM      7  CA  TRP A  32       0.000   0.000 101.000  1.00  0.00           C #0 was changed to 101 (0 + 101 = 101)
        ATOM      8  CB  TRP A  32       0.000   0.000  80.000  1.00  0.00           C
        ATOM      9  C   TRP A  32       0.000   0.000 100.000  1.00  0.00           C
        TER   
        """

        # With the above geometry, it's easy to see that the COMs are, for x,y,z respectively
        # Frame 1
            # GLU30 (0+2+10)/3 = 4
            # VAL31 (0+5+10)/3 = 5
            # TRP31 (0+8+10)/3 = 6
        # Frame 2
            # GLU30 (0+20-1)/3     = 19/3  = 6.333333333333333
            # VAL31 (0+50+100)/3   = 50
            # TRP31 (101+80+100)/3 = 281/3 = 93.66666666666667
        # And then the max radii are, respectively
        # Frame 1
            # GLU30 10 - 4 = 6
            # VAL31 10 - 5 = 5 or 0 - 5 = 5
            # TRP31  0 - 6 = 6
        # Frame 2
            # GLU30 6.333 - 20 = 13.666
            # VAL31 100 - 50 = 50 or 0 - 50 = 50
            # TRP31 93.666 - 80 = 13.666

        # Note that we have to multiply x .1 to be back in nm (pdb strg is in Ang)
        self.COMs = _np.array([[[4, 0, 0, ],
                                [0, 5, 0],
                                [0, 0., 6]],
                               [[19 / 3, 0, 0],
                                [0, 50, 0],
                                [0, 0, 281 / 3]]] # [0, 0, 93.66666]
                              )/10

        # This essentially re-tests the setup
        _np.testing.assert_array_almost_equal(self.COMs, _mdcu.COM.geom2COMxyz(self.geom_unwrapped))

        self.max_radii = _np.abs(_np.array([[6, 5, 6],                       # [[0.6,       0.5, 0.6]
                                            [19 / 3 - 20, 50, 281 / 3 - 80]] # [1.36666667, 5.,  1.36666667]]
                                           )) / 10

        # This essentially re-tests the setup
        _np.testing.assert_array_almost_equal(self.max_radii, _mdcu.COM.geom2max_residue_radius(self.geom_unwrapped))



        # Note the last z-coordinate is 93.666 which is closer to the other z-coordinates (0)
        # via the minimum image convention at 93.6666 - 101 = -7.33333
        self.COMd = _np.array([[0.64031242, 0.72111026, 0.78102497],  # # pdist(self.COMs[0])
                               [5.0399515 , 0.96896279, 5.05349164]])  # pdist(self.COMs[1], where the z coordinate of the last residue has been changed to -7.3)

        _np.testing.assert_array_almost_equal(self.COMd, _mdcu.COM.geom2COMdist(self.geom, [[0,1], [0,2], [1,2]]))

        # We take the max of the max_radii for all frames
        self.sum_maxR = _np.array([abs(19 / 3 - 20) + 50,                   # radii res0 + res1
                                   abs(19 / 3 - 20) + abs(281 / 3 - 80),    # radii res0 + res2
                                   50 + abs(281 / 3 - 80)                   # radii res1 + res2
                                   ])/10
        self.lower_bound_t = self.COMd - self.sum_maxR
        # [[-5.72635425 -2.01222307 -5.5856417 ]
        # [-1.32671517 -1.76437054 -1.31317503]]
    def test_works(self):
        lower_bounds = contacts.per_traj_mindist_lower_bound(self.geom.top, self.geom, [[0,1],[0,2],[1,2]], 1000, 1, 0)
        _np.testing.assert_array_almost_equal(lower_bounds, self.lower_bound_t.min(axis=0))

    def test_works_timetrace(self):
        lower_bounds_t = contacts.per_traj_mindist_lower_bound(self.geom.top, self.geom, [[0, 1], [0, 2], [1, 2]],
                                                               1000, 1, 0, timetrace=True)

        _np.testing.assert_array_almost_equal(lower_bounds_t, self.lower_bound_t)

    def test_works_timetrace_lb_cutoff_Ang(self):
        lower_bounds_t_bool = contacts.per_traj_mindist_lower_bound(self.geom.top, self.geom, [[0, 1], [0, 2], [1, 2]],
                                                                    1000, 1, 0,
                                                                    timetrace=True,
                                                                    lb_cutoff_Ang=-57 , #it's weird it's negative but it's okay for tests,
                                                                    )
        # These are time-dep lower bounds
        # [[-5.72635425 -2.01222307 -5.5856417 ]
        # [-1.32671517 -1.76437054 -1.31317503]]
        _np.testing.assert_array_almost_equal(lower_bounds_t_bool, [0])

    # Since trajs2lower_bounds is mainly a wrapper on per_traj_mindist_lower_bound
    #  i'd rather test it here than have its own class
    def test_trajs2lower_bounds(self):
        list_of_lbs = contacts.trajs2lower_bounds([self.geom, self.geom[::-1]],
                                                  self.geom.top, [[0, 1], [0, 2], [1, 2]])

        _np.testing.assert_array_almost_equal(self.lower_bound_t.min(axis=0), list_of_lbs[0])
        _np.testing.assert_array_almost_equal(self.lower_bound_t.min(axis=0), list_of_lbs[1])

    def test_trajs2lower_bounds_timetrace(self):
        list_of_lbs = contacts.trajs2lower_bounds([self.geom, self.geom[::-1]],
                                                  self.geom.top, [[0, 1], [0, 2], [1, 2]], timetrace=True)

        _np.testing.assert_array_almost_equal(self.lower_bound_t, list_of_lbs[0])
        _np.testing.assert_array_almost_equal(self.lower_bound_t[::-1], list_of_lbs[1])

    def test_trajs2lower_bounds_cutoff(self):
        list_of_lbs = contacts.trajs2lower_bounds([self.geom, self.geom[::-1]],
                                                  self.geom.top, [[0, 1], [0, 2], [1, 2]],
                                                  lb_cutoff_Ang=-57) #it's weird it's negative but it's okay for tests

        _np.testing.assert_array_almost_equal([0], list_of_lbs[0])
        _np.testing.assert_array_almost_equal([0], list_of_lbs[1])


class BaseClassForTestingAttributes(unittest.TestCase):
    def setUp(self):
        self.trajs = md.load(test_filenames.traj_xtc_stride_20, top=test_filenames.top_pdb)[:3]
        self.trajs = [self.trajs[:2],
                      self.trajs[:3]]
        self.ctc_trajs = [[1, 2], [10, 11, 12]]
        self.atom_pair_trajs = [[[100, 200], [100, 201]],
                                [[101, 200], [100, 200], [100, 201]]]


class TestTimeTraces(BaseClassForTestingAttributes):

    def test_works(self):
        cott = _TimeTraces(self.ctc_trajs, [self.trajs[0].time,
                                                     self.trajs[1].time],
                                    self.trajs, self.atom_pair_trajs)
        [_np.testing.assert_allclose(itraj, jtraj) for itraj, jtraj in zip(cott.ctc_trajs, self.ctc_trajs)]
        [_np.testing.assert_allclose(itraj, jtraj) for itraj, jtraj in zip(cott.feat_trajs, self.ctc_trajs)]
        [_np.testing.assert_allclose(itraj, jtraj) for itraj, jtraj in zip(cott.time_trajs, [self.trajs[0].time,
                                                                                             self.trajs[1].time])]

        [_np.testing.assert_allclose(itraj, jtraj) for itraj, jtraj in zip(cott.atom_pair_trajs, self.atom_pair_trajs)]
        assert all([itraj == jtraj for itraj, jtraj in zip(cott.trajs, self.trajs)])

    def test_fails_because_wrong_atom_trajs(self):
        with self.assertRaises(AssertionError):
            _TimeTraces(self.ctc_trajs, [self.trajs[0].time,
                                                  self.trajs[1].time],
                                 self.trajs, [[[0, 1]], [0, 1]])

        with self.assertRaises(AssertionError):
            _TimeTraces(self.ctc_trajs, [self.trajs[0].time,
                                                  self.trajs[1].time],
                                 self.trajs, [[[0, 1, 1]], [0, 1]])


class TestNumberOfThings(BaseClassForTestingAttributes):

    def test_works(self):
        cont = _NumberOfthings(len(self.ctc_trajs),
                                        [len(itraj) for itraj in self.ctc_trajs]
                                        )
        _np.testing.assert_equal(len(self.ctc_trajs), cont.n_trajs)
        _np.testing.assert_allclose([len(itraj) for itraj in self.ctc_trajs], cont.n_frames)
        _np.testing.assert_equal(_np.sum([len(itraj) for itraj in self.ctc_trajs]), cont.n_frames_total)


class TestResidues(unittest.TestCase):

    def test_just_works(self):
        cors = Residues([0, 1], ["GLU25", "ALA35"])

        _np.testing.assert_allclose([0, 1], cors.idxs_pair)
        assert cors.names[0] == "GLU25" and cors.names[1] == "ALA35"
        assert cors.names_short[0] == "E25" and cors.names_short[1] == "A35", cors.names_short

        _np.testing.assert_equal(None, cors.anchor_index)
        _np.testing.assert_equal(None, cors.anchor_residue)
        _np.testing.assert_equal(None, cors.anchor_residue_index)

        _np.testing.assert_equal(None, cors.partner_index)
        _np.testing.assert_equal(None, cors.partner_residue)
        _np.testing.assert_equal(None, cors.partner_residue_index)

    def test_anchor_and_partner(self):
        cors = Residues([10, 20],
                        ["GLU25", "ALA35"],
                        anchor_residue_idx=10
                        )

        _np.testing.assert_equal(10, cors.anchor_residue_index)
        _np.testing.assert_equal(0, cors.anchor_index)

        _np.testing.assert_equal(1, cors.partner_index)
        _np.testing.assert_equal(20, cors.partner_residue_index)

    def test_anchor_and_partner_top(self):
        top = md.load(test_filenames.top_pdb).top
        cors = Residues([10, 20],
                        ["GLU25", "ALA35"],
                        anchor_residue_idx=10,
                        top=top
                        )

        _np.testing.assert_equal(10, cors.anchor_residue_index)
        _np.testing.assert_equal(0, cors.anchor_index)
        assert top.residue(10) is cors.anchor_residue

        _np.testing.assert_equal(1, cors.partner_index)
        _np.testing.assert_equal(20, cors.partner_residue_index)
        assert top.residue(20) is cors.partner_residue

    def test_names(self):
        cors = Residues([10, 20],
                        ["GLU25", "ALA35"]
                        )
        assert cors.names[0] == "GLU25"
        assert cors.names[1] == "ALA35"

        assert cors.names_short[0] == "E25"
        assert cors.names_short[1] == "A35"

    def test_consensus_labels(self):
        cors = Residues([10, 20],
                        ["GLU25", "ALA35"],
                        consensus_labels=["3.50", "4.50"])
        assert cors.consensus_labels[0] == "3.50"
        assert cors.consensus_labels[1] == "4.50"


class TestFragments(unittest.TestCase):

    def test_just_works_empty(self):
        cof = _Fragments()
        assert cof.idxs is None
        assert cof.colors[0] is cof.colors[1] is None
        assert cof.names[0] is cof.names[1] is None

    def test_works(self):
        cof = _Fragments(fragment_idxs=[0, 1],
                                  fragment_colors=["r", "b"],
                                  fragment_names=["fragA", "fragB"])
        _np.testing.assert_allclose([0, 1], cof.idxs)

        assert cof.colors[0] == "r" and cof.colors[1] == "b"
        assert cof.names[0] == "fragA" and cof.names[1] == "fragB"

    def test_auto_fragnaming(self):
        cof = _Fragments(fragment_idxs=[0, 1],
                                  )

        assert cof.names[0] == str(0) and cof.names[1] == str(1)

    def test_anchor_and_partner(self):
        cors = Residues([10, 20],
                        [None, None],
                        anchor_residue_idx=10
                        )

        _np.testing.assert_equal(10, cors.anchor_residue_index)
        _np.testing.assert_equal(0, cors.anchor_index)

        _np.testing.assert_equal(1, cors.partner_index)
        _np.testing.assert_equal(20, cors.partner_residue_index)

    def test_anchor_and_partner_top(self):
        top = md.load(test_filenames.top_pdb).top
        cors = Residues([10, 20],
                        [None, None],
                        anchor_residue_idx=10,
                        top=top
                        )

        _np.testing.assert_equal(10, cors.anchor_residue_index)
        _np.testing.assert_equal(0, cors.anchor_index)
        assert top.residue(10) is cors.anchor_residue

        _np.testing.assert_equal(1, cors.partner_index)
        _np.testing.assert_equal(20, cors.partner_residue_index)
        assert top.residue(20) is cors.partner_residue

    def test_names(self):
        cors = Residues([10, 20],
                        ["GLU25", "ALA35"]
                        )
        self.assertEqual(cors.names[0],"GLU25")
        self.assertEqual(cors.names[1],"ALA35")

        self.assertEqual(cors.names_short[0],"E25")
        self.assertEqual(cors.names_short[1],"A35")


class TestNeighborhoodNames(unittest.TestCase):

    def test_works(self):
        cnns = _NeighborhoodNames(
            Residues([10, 20],
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
        with self.assertRaises(AssertionError):
            _NeighborhoodNames(Residues([10, 20], ["GLU25", "ALA35"]),
                               _Fragments([0, 1],
                                                            ["fragA", "fragB"],
                                                            ["r", "b"]
                                                            ))

    def test_fragments_consensus_name_None(self):
        cnns = _NeighborhoodNames(
            Residues([10, 20],
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
            Residues([10, 20],
                     ["GLU25", "ALA35"],
                     anchor_residue_idx=20,
                     consensus_labels=["3.50", "4.50"]
                     ),
            _Fragments([0, 1],
                                ["fragA", "fragB"],
                                )
        )
        assert cnns.partner_fragment_consensus == "3.50"
        assert cnns.anchor_fragment_consensus == "4.50"

    def test_fragments_names_best_no_consensus(self):
        cnns = _NeighborhoodNames(
            Residues([10, 20],
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
            Residues([10, 20],
                     ["GLU25", "ALA35"],
                     anchor_residue_idx=20,
                     consensus_labels=["3.50", "4.50"]
                     ),
            _Fragments([0, 1],
                                fragment_names=["fragA", "fragB"]
                                )
        )
        assert cnns.partner_fragment_best == "3.50"
        assert cnns.anchor_fragment_best == "4.50"

    def test_res_and_fragment_strs_no_consensus(self):
        cnns = _NeighborhoodNames(
            Residues([10, 20],
                     ["GLU25", "ALA35"],
                     anchor_residue_idx=20,
                     # consensus_labels=["3.50","4.50"]
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
            Residues([10, 20],
                     ["GLU25", "ALA35"],
                     anchor_residue_idx=20,
                     consensus_labels=["3.50", "4.50"]
                     ),
            _Fragments([0, 1],
                                fragment_names=["fragA", "fragB"]
                                )
        )
        assert cnns.anchor_res_and_fragment_str == "ALA35@4.50"
        assert cnns.partner_res_and_fragment_str == "GLU25@3.50"

        assert cnns.anchor_res_and_fragment_str_short == "A35@4.50"
        assert cnns.partner_res_and_fragment_str_short == "E25@3.50"


class TestContactStrings(unittest.TestCase):

    def test_trajlabels_no_trajs(self):
        cls = _ContactStrings(2,
                              Residues([10, 20], ["GLU25", "ALA35"])
                              )

        assert cls.trajstrs[0] == "traj 0" and cls.trajstrs[1] == "traj 1", cls.trajstrs

    def test_trajlabels_w_mdtrajs(self):
        mdtrajs = md.load(test_filenames.traj_xtc_stride_20,
                          top=test_filenames.top_pdb)[:5]
        mdtrajs = [mdtrajs, mdtrajs]
        cls = _ContactStrings(2,
                              Residues([10, 20], ["GLU25", "ALA35"]),
                              trajs=mdtrajs
                              )

        assert cls.trajstrs[0] == "mdtraj.00" and cls.trajstrs[1] == "mdtraj.01", cls.trajstrs

    def test_trajlabels_wo_mdtrajs(self):
        cls = _ContactStrings(2,
                              Residues([10, 20], ["GLU25", "ALA35"]),
                              trajs=["file0.xtc", "file1.xtc"]
                              )

        assert cls.trajstrs[0] == "file0" and cls.trajstrs[1] == "file1", cls.trajstrs

    def test_ctc_labels(self):
        cls = _ContactStrings(2,
                              Residues([10, 20], ["GLU25", "ALA35"]),
                              _Fragments(fragment_names=["fragA", "fragB"])
                              )
        assert cls.w_fragments == "GLU25@fragA-ALA35@fragB", cls.w_fragments
        assert cls.w_fragments_short_AA == "E25@fragA-A35@fragB", cls.w_fragments_short_AA

    def test_ctc_label_nocontacts_Fragments(self):
        cls = _ContactStrings(2,
                              Residues([10, 20], ["GLU25", "ALA35"]),
                              )

        assert cls.no_fragments == "GLU25-ALA35", cls.no_fragments
        assert cls.no_fragments_short_AA == "E25-A35"

    def test_ctc_label_missing_frags_and_consensus(self):
        cls = _ContactStrings(2,
                              Residues([10, 20], ["GLU25", "ALA35"]),
                              )

        assert cls.w_fragments == "GLU25-ALA35", cls.ctc_label_w_fragment
        assert cls.w_fragments_short_AA == "E25-A35", cls.w_fragments_short_AA

    def test_just_prints(self):
        cls = _ContactStrings(2,
                              Residues([10, 20], ["GLU25", "ALA35"]),
                              )
        print(cls)


class TestContactPair(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.small_monomer)

    def test_works_minimal(self):
        contacts.ContactPair([0, 1],
                             [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                             [[0, 1, 2], [0, 1, 2, 3]],
                             )

    def test_stacked_contact_trace(self):
        cpt = contacts.ContactPair([0, 1],
                             [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                             [[0, 1, 2], [0, 1, 2, 3]],
                             )

        self.assertListEqual([1.0, 1.1, 1.3] + [2.0, 2.1, 2.3, 2.4], cpt.stacked_time_traces.tolist())


    def test_with_top(self):
        contact_pair_test = contacts.ContactPair([0, 1],
                                                 [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                                 [[0, 1, 2], [0, 1, 2, 3]],
                                                 top=self.geom.top,
                                                 )
        contact_pair_test.top == self.geom.top == contact_pair_test.topology

    def test_with_anchor(self):
        contact_pair_test = contacts.ContactPair([0, 1],
                                                 [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                                 [[0, 1, 2], [0, 1, 2, 3]],
                                                 anchor_residue_idx=1,
                                                 )
        contact_pair_test.top == self.geom.top
        assert contact_pair_test.neighborhood is not None

    def test_all_properties_w_empty_ones_just_runs(self):
        cpt = contacts.ContactPair([0, 1],
                                   [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                   [[0, 1, 2], [0, 1, 2, 3]]
                                   )
        cpt.time_traces
        cpt.n
        cpt.residues
        cpt.fragments
        cpt.labels
        cpt.label
        assert cpt.time_max == 3
        assert cpt.neighborhood is None

    def test_binarize_trajs(self):
        cpt = contacts.ContactPair([0, 1],
                                   [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                   [[0, 1, 2], [0, 1, 2, 3]]
                                   )
        bintrajs = cpt.binarize_trajs(21)
        _np.testing.assert_array_equal([1, 1, 1], bintrajs[0])
        _np.testing.assert_array_equal([1, 1, 0, 0], bintrajs[1])

    def test_freq_per_traj(self):
        cpt = contacts.ContactPair([0, 1],
                                   [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                   [[0, 1, 2], [0, 1, 2, 3]]
                                   )
        freqs = cpt.frequency_per_traj(21)
        _np.testing.assert_equal(freqs[0], 1)
        _np.testing.assert_equal(freqs[1], .5)

    def test_freq_overall_trajs(self):
        cpt = contacts.ContactPair([0, 1],
                                   [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                   [[0, 1, 2], [0, 1, 2, 3]]
                                   )
        _np.testing.assert_equal(cpt.frequency_overall_trajs(21),
                                 _np.mean([1, 1, 1] + [1, 1, 0, 0]))

    def test_frequency_dict_no_labels(self):
        cpt = contacts.ContactPair([0, 1],
                                   [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                   [[0, 1, 2], [0, 1, 2, 3]]
                                   )

        idict = cpt.frequency_dict(21)
        assert idict["freq"] == _np.mean([1, 1, 1] + [1, 1, 0, 0])
        assert idict["residues"] == "0 - 1"
        assert idict["label"] == ('%-15s - %-15s' % (0, 1)), idict["label"]

        idict = cpt.frequency_dict(21, AA_format="long")
        assert idict["label"] == ('%-15s - %-15s' % (0, 1)), idict["label"]

        idict = cpt.frequency_dict(21, pad_label=False)
        assert idict["label"] == "0-1"

    def test_frequency_dict_w_labels(self):
        cpt = contacts.ContactPair([0, 1],
                                   [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                   [[0, 1, 2], [0, 1, 2, 3]],
                                   fragment_names=["fragA", "fragB"]
                                   )
        idict = cpt.frequency_dict(21)
        self.assertEqual(idict["label"],
                         ('%-15s - %-15s' % ("0@fragA", "1@fragB")))

        idict = cpt.frequency_dict(21, AA_format="long")
        assert idict["label"] == ('%-15s - %-15s' % ("0@fragA", "1@fragB"))
        idict = cpt.frequency_dict(21, pad_label=False)
        assert idict["label"] == '0@fragA-1@fragB'

    def test_frequency_dict_w_labels_just_consensus(self):
        cpt = contacts.ContactPair([0, 1],
                                   [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                   [[0, 1, 2], [0, 1, 2, 3]],
                                   consensus_labels=["3.50","4.50"]
                                   )
        idict = cpt.frequency_dict(21,
                                   pad_label=False,
                                   AA_format="just_consensus")
        self.assertEqual(idict["label"],
                         "3.50-4.50")

    def test_frequency_dict_w_labels_just_consensus_raises(self):
        cpt = contacts.ContactPair([0, 1],
                                   [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                   [[0, 1, 2], [0, 1, 2, 3]],
                                   )
        with self.assertRaises(ValueError):
            idict = cpt.frequency_dict(21,
                                       pad_label=False,
                                       AA_format="just_consensus")


    def test_frequency_dict_by_atom_types(self):
        # From test_frequency_dict_formed_atom_pairs_overall_trajs_aggregate_by_atomtype_False(self):
        # Completely bogus contact but testable
        atom_BB_1 = list(self.geom.top.residue(0).atoms_by_name("CA"))[0].index
        atom_BB_2 = list(self.geom.top.residue(0).atoms_by_name("N"))[0].index
        atom_SC = list(self.geom.top.residue(1).atoms_by_name("CB"))[0].index
        cpt = contacts.ContactPair([0, 1],
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
        res = cpt.frequency_dict(30,atom_types=True)
        self.assertDictEqual(res["by_atomtypes"],{'BB-BB': 0.25, 'BB-SC': 0.5, 'SC-SC': 0.25})

    def test_distro_overall_trajs(self):
        cpt = contacts.ContactPair([0, 1],
                                   [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                   [[0, 1, 2], [0, 1, 2, 3]],
                                   )
        x, h = cpt.distro_overall_trajs(bins=10)
        xt, ht = _np.histogram([1.0, 1.1, 1.3] + [2.0, 2.1, 2.3, 2.4], 10)
        _np.testing.assert_array_equal(x, xt)
        _np.testing.assert_array_equal(h, ht)

    def test_formed_atom_pairs_fails(self):
        cpt = contacts.ContactPair([0, 1],
                                   [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                   [[0, 1, 2], [0, 1, 2, 3]],
                                   )
        with self.assertRaises((AssertionError, ValueError)):
            cpt.count_formed_atom_pairs(21)

    def test_overall_stacked_formed_atoms(self):
        cpt = contacts.ContactPair([0, 1],
                                   [[1.0, 2.5, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                   [[0, 1, 2], [0, 1, 2, 3]],
                                   atom_pair_trajs=[
                                       [[10, 20], [11, 20], [10, 21]],
                                       [[11, 21], [10, 21], [10, 20], [11, 20]]
                                   ]
                                   )

        fas = cpt._overall_stacked_formed_atoms(21)
        _np.testing.assert_array_equal([[10, 20],
                                        # [11,20], this frame is not formed
                                        [10, 21],
                                        [11, 21],
                                        [10, 21]], fas)

    def test_count_formed_atom_pairs(self):
        cpt = contacts.ContactPair([0, 1],
                                   [[1.0, 2.5, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                   [[0, 1, 2], [0, 1, 2, 3]],
                                   atom_pair_trajs=[
                                       [[10, 20], [11, 20], [10, 21]],
                                       [[10, 21], [10, 21], [10, 20], [11, 20]]
                                   ]
                                   )
        # The count should be [10,20]:1, [10,21]:3,
        pairs, counts = cpt.count_formed_atom_pairs(21, sort=False)
        _np.testing.assert_array_equal(pairs, [[10, 20], [10, 21]])
        _np.testing.assert_array_equal(counts, [1, 3])

        pairs, counts = cpt.count_formed_atom_pairs(21, sort=True)
        _np.testing.assert_array_equal(pairs, [[10, 21], [10, 20]])
        _np.testing.assert_array_equal(counts, [3, 1])

    def test_frequency_dict_formed_atom_pairs_overall_trajs_fails(self):
        cpt = contacts.ContactPair([0, 1],
                                   [[1.0, 2.5, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                   [[0, 1, 2], [0, 1, 2, 3]],
                                   atom_pair_trajs=[
                                       [[10, 20], [11, 20], [10, 21]],
                                       [[10, 21], [10, 21], [10, 20], [11, 20]]
                                   ]
                                   )
        with self.assertRaises(AssertionError):
            cpt.relative_frequency_of_formed_atom_pairs_overall_trajs(21)

    def test_frequency_dict_formed_atom_pairs_overall_trajs(self):
        # Completely bogus contact but testable
        atom_BB = list(self.geom.top.residue(0).atoms_by_name("CA"))[0].index
        atom_SC = list(self.geom.top.residue(1).atoms_by_name("CB"))[0].index
        cpt = contacts.ContactPair([0, 1],
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
        _np.testing.assert_equal(out_dict["BB-SC"], 2 / 3)
        _np.testing.assert_equal(out_dict["BB-BB"], 1 / 3)
        assert len(out_dict) == 2

        out_dict = cpt.relative_frequency_of_formed_atom_pairs_overall_trajs(21, min_freq=.5)
        _np.testing.assert_equal(out_dict["BB-SC"], 2 / 3)
        assert len(out_dict) == 1

    def test_frequency_dict_formed_atom_pairs_overall_trajs_aggregate_by_atomtype_False(self):
        # Completely bogus contact but testable
        atom_BB_1 = list(self.geom.top.residue(0).atoms_by_name("CA"))[0].index
        atom_BB_2 = list(self.geom.top.residue(0).atoms_by_name("N"))[0].index
        atom_SC = list(self.geom.top.residue(1).atoms_by_name("CB"))[0].index
        cpt = contacts.ContactPair([0, 1],
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
        _np.testing.assert_equal(out_dict["%s-%s" % (self.geom.top.atom(atom_BB_1),
                                                     self.geom.top.atom(atom_SC))], 1 / 3)
        _np.testing.assert_equal(out_dict["%s-%s" % (self.geom.top.atom(atom_BB_2),
                                                     self.geom.top.atom(atom_SC))], 1 / 3)
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
        assert len(out_dict) == 3

    def test_prints(self):
        cpt = contacts.ContactPair([0, 1],
                                   [[1.0, 2.5, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                   [[0, 1, 2], [0, 1, 2, 3]],
                                   atom_pair_trajs=[
                                       [[10, 20], [11, 20], [10, 21]],
                                       [[10, 21], [10, 21], [10, 20], [11, 20]]
                                   ]
                                   )
        print(cpt)

    def test_plot_timetrace_just_works(self):
        CP = contacts.ContactPair([0, 1],
                                  [[.10, .25, .13], [.20, .21, .23, .24]],
                                  [[0, 1, 2], [0, 1, 2, 3]]
                                  )
        _plt.figure()
        iax = _plt.gca()
        CP.plot_timetrace(iax)

    def test_plot_timetrace_just_options(self):
        CP = contacts.ContactPair([0, 1],
                                  [[.10, .25, .13], [.20, .21, .23, .24]],
                                  [[0, 1, 2], [0, 1, 2, 3]],
                                  top=self.geom.top
                                  )

        CP.plot_timetrace(ctc_cutoff_Ang=2, shorten_AAs=True, ylim_Ang="auto")

        with self.assertRaises(ValueError):
            CP.plot_timetrace(ylim_Ang="max")

    def test_plot_distance_distribution(self):
        CP = contacts.ContactPair([0, 1],
                                  [[.10, .25, .13], [.20, .21, .23, .24]],
                                  [[0, 1, 2], [0, 1, 2, 3]],
                                  )

        ax = CP.plot_distance_distribution(bins=4, shorten_AAs=True, ctc_cutoff_Ang=2, xlim=[.5, 3])

        #ax.figure.savefig("test.png")
        _plt.close("all")

    def test_plot_distance_distribution_smoothing(self):
        CP = contacts.ContactPair([0, 1],
                                  [[.10, .25, .13], [.20, .21, .23, .24]],
                                  [[0, 1, 2], [0, 1, 2, 3]],
                                  )

        ax = CP.plot_distance_distribution(bins=4, shorten_AAs=True, ctc_cutoff_Ang=2, label="True", xlim=[.5, 3],
                                           smooth_bw=True, background=False)
        ax = CP.plot_distance_distribution(ax=ax, bins=4, shorten_AAs=True, ctc_cutoff_Ang=2, label=".5", smooth_bw=.5,
                                           background=True)
        ax = CP.plot_distance_distribution(ax=ax, bins=4, shorten_AAs=True, ctc_cutoff_Ang=2, label=".5", smooth_bw=.5,
                                           background="red")
        #ax.figure.savefig("test.png")
        _plt.close("all")

    def test_retop(self):
        CG = examples.ContactGroupL394()

        CP : contacts.ContactPair = CG.contact_pairs[0]

        top = md.load(test_filenames.pdb_3SN6).top
        #print(CP.top, CP.residues.idxs_pair)
        #print(CP.residues.names_short)
        #print([utils.residue_and_atom.find_AA(AA,top) for AA in CP.residues.names_short])
        imap = {347:342,
                353:348}
        nCP : contacts.ContactPair = CP.retop(top,imap)
        # Test the residx
        _np.testing.assert_array_equal(nCP.residues.idxs_pair,[348, 342])

        # Test the non-nested attributes
        for attr in [
            "time_traces.trajs",
            "fragments.idxs",
            "fragments.names",
            "fragments.colors",
            "residues.consensus_labels"
        ]:
            attr1, attr2 = attr.split(".")
            assert getattr(getattr(CP,attr1),attr2) is getattr(getattr(nCP,attr1),attr2), attr
            assert getattr(getattr(CP, attr1), attr2)==getattr(getattr(nCP, attr1), attr2)
        assert nCP.residues.anchor_residue_index == 348

        # Thest the nested attributes
        for attr in [
            "time_traces.ctc_trajs",
            "time_traces.time_trajs"
        ]:
            attr1, attr2 = attr.split(".")
            l1, l2 = getattr(getattr(CP, attr1), attr2), getattr(getattr(nCP, attr1), attr2)
            assert l1 is not l2
            for traj, ntraj in zip(l1,l2):
                _np.testing.assert_array_equal(traj,ntraj)

        # Test the pair indices
        pair_freq = (CP.relative_frequency_of_formed_atom_pairs_overall_trajs(4, aggregate_by_atomtype=False))
        _np.testing.assert_array_equal(pair_freq, nCP.relative_frequency_of_formed_atom_pairs_overall_trajs(4,aggregate_by_atomtype=False))

    def test_retop_deepcopy(self):
        CG = examples.ContactGroupL394()

        CP: contacts.ContactPair = CG.contact_pairs[0]

        top = md.load(test_filenames.pdb_3SN6).top
        imap = {347: 342,
                353: 348}
        nCP: contacts.ContactPair = CP.retop(top, imap, deepcopy=True)
        for attr in [
            "time_traces.trajs",
            "fragments.idxs",
            "fragments.names",
            "fragments.colors",
            "residues.consensus_labels"
        ]:
            attr1, attr2 = attr.split(".")
            assert getattr(getattr(CP, attr1), attr2) is not getattr(getattr(nCP, attr1), attr2), attr
            assert getattr(getattr(CP, attr1), attr2)==getattr(getattr(nCP, attr1), attr2)
        assert nCP.residues.anchor_residue_index == 348
        for attr in [
            "time_traces.ctc_trajs",
            "time_traces.time_trajs"
        ]:
            attr1, attr2 = attr.split(".")
            l1, l2 = getattr(getattr(CP, attr1), attr2), getattr(getattr(nCP, attr1), attr2)
            assert l1 is not l2
            for traj, ntraj in zip(l1, l2):
                _np.testing.assert_array_equal(traj, ntraj)

    def test_serialize_as_dict(self):
        CG = examples.ContactGroupL394()
        CP: contacts.ContactPair = CG.contact_pairs[0]
        sCP = CP._serialized_as_dict()

        assert sCP["residues.idxs_pair"] is CP.residues.idxs_pair
        assert sCP["time_traces.ctc_trajs"] is CP.time_traces.ctc_trajs
        assert sCP["time_traces.time_trajs"] is CP.time_traces.time_trajs
        assert sCP.get("topology") is None
        assert sCP["time_traces.trajs"] is CP.time_traces.trajs
        assert sCP["time_traces.atom_pair_trajs"] is CP.time_traces.atom_pair_trajs
        assert sCP["fragments.idxs"] is CP.fragments.idxs
        assert sCP["fragments.names"] is CP.fragments.names
        assert sCP["fragments.colors"] is CP.fragments.colors
        assert sCP["residues.anchor_residue_index"] is CP.residues.anchor_residue_index
        assert sCP["residues.consensus_labels"] is CP.residues.consensus_labels

    def test_gen_labels(self):
        CG = examples.ContactGroupL394()
        CP: contacts.ContactPair = CG.contact_pairs[0]

        self.assertEqual(CP.gen_label("short"),"L394-L388")
        self.assertEqual(CP.gen_label("long") ,"LEU394-LEU388")
        self.assertEqual(CP.gen_label("short",fragments=True), "L394@G.H5.26-L388@G.H5.20")
        self.assertEqual(CP.gen_label("long",fragments=True) ,"LEU394@G.H5.26-LEU388@G.H5.20")
        self.assertEqual(CP.gen_label("just_consensus") ,"G.H5.26-G.H5.20")

        self.assertEqual(CP.gen_label("short", delete_anchor=True), "L388")
        self.assertEqual(CP.gen_label("long", delete_anchor=True), "LEU388")
        self.assertEqual(CP.gen_label("short", fragments=True, delete_anchor=True), "L388@G.H5.20")
        self.assertEqual(CP.gen_label("long", fragments=True, delete_anchor=True), "LEU388@G.H5.20")
        self.assertEqual(CP.gen_label("just_consensus", delete_anchor=True) ,"G.H5.26")

        with self.assertRaises(ValueError):
            CP.gen_label("wrong")

    def test_gen_labels_no_neighborhood(self):
        CP: contacts.ContactPair = contacts.ContactPair([0, 1],
                                                        [[1.0, 1.1, 1.3], [2.0, 2.1, 2.3, 2.4]],
                                                        [[0, 1, 2], [0, 1, 2, 3]],
                                                        fragment_names=["A", "B"]
                                                        )

        self.assertEqual(CP.gen_label("short"),"0-1")
        self.assertEqual(CP.gen_label("long") ,"0-1")
        self.assertEqual(CP.gen_label("short",fragments=True), "0@A-1@B")
        self.assertEqual(CP.gen_label("long",fragments=True) ,"0@A-1@B")

        #No neighbor
        self.assertEqual(CP.gen_label("short", delete_anchor=True), "0-1")
        self.assertEqual(CP.gen_label("long", delete_anchor=True), "0-1")
        self.assertEqual(CP.gen_label("short", fragments=True, delete_anchor=True), "0@A-1@B")
        self.assertEqual(CP.gen_label("long", fragments=True, delete_anchor=True), "0@A-1@B")
        with self.assertRaises(ValueError):
            CP.gen_label("wrong")


class Test_sum_ctc_freqs_by_atom_type(unittest.TestCase):
    def test_works(self):
        top = md.load(test_filenames.top_pdb).top
        atoms_BB = [aa for aa in top.residue(0).atoms if aa.is_backbone]
        atoms_SC = [aa for aa in top.residue(0).atoms if aa.is_sidechain]
        atom_pairs, counts = [], []
        for trip in [[atoms_BB[0], atoms_BB[1], 5],
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
        assert len(dict_out) == 4


class Test_select_and_report_residue_neighborhood_idxs(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.small_monomer)
        self.fragments = get_fragments(self.geom.top,
                                       verbose=True,
                                       auto_fragment_names=True,
                                       method='bonds')
        self.residues = ["GLU30", "VAL31"] #ie idxs 0 and 1
        self.residxs, self.fragidxs = residues_from_descriptors(self.residues,
                                                                self.fragments,
                                                                self.geom.top)
        self.ctc_cutoff_Ang = 3

    def test_select_and_report_residue_neighborhood_idxs_just_works(self):
        ctc_freqs = [1, .5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]

        input_values = (val for val in ["1", "2"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):  # Checking against the input 1 and 1
            per_residx_ctc_idxs = contacts.select_and_report_residue_neighborhood_idxs(ctc_freqs, self.ctc_cutoff_Ang, self.residxs,
                                                                            self.fragments, ctc_residxs_pairs,
                                                                            self.geom.top,
                                                                            interactive=True)
        _np.testing.assert_array_equal(per_residx_ctc_idxs[0],[0])
        _np.testing.assert_array_equal(per_residx_ctc_idxs[1],[0,1])

    def test_select_and_report_residue_neighborhood_idxs_select_by_resSeq(self):
        ctc_freqs = [1., .5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]

        input_values = (val for val in ["2"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):  # Checking against the input 1 and 1
            per_residx_ctc_idxs = contacts.select_and_report_residue_neighborhood_idxs(ctc_freqs, self.ctc_cutoff_Ang, self.residxs,
                                                                            self.fragments, ctc_residxs_pairs,
                                                                            self.geom.top,
                                                                            restrict_to_resSeq=[31],
                                                                            interactive=True)
        assert len(per_residx_ctc_idxs) == 1
        _np.testing.assert_array_equal(per_residx_ctc_idxs[1],[0,1])

    def test_select_and_report_residue_neighborhood_idxs_hit_enter(self):
        ctc_freq = [1.,.5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]

        input_values = (val for val in ["", ""])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            per_residx_ctc_idxs = contacts.select_and_report_residue_neighborhood_idxs(ctc_freq, self.ctc_cutoff_Ang, self.residxs,
                                                                            self.fragments, ctc_residxs_pairs,
                                                                            self.geom.top,
                                                                            interactive=True)
            assert per_residx_ctc_idxs == {}

    def test_select_and_report_residue_neighborhood_idxs_no_interactive(self):
        ctc_freq = [1.,.5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]
        per_residx_ctc_idxs = contacts.select_and_report_residue_neighborhood_idxs(ctc_freq, self.ctc_cutoff_Ang, self.residxs,
                                                                        self.fragments, ctc_residxs_pairs,
                                                                        self.geom.top,
                                                                        interactive=False)
        assert (_np.array_equal(per_residx_ctc_idxs[0], [0]))
        assert (_np.array_equal(per_residx_ctc_idxs[1], [0, 1]))

    def test_select_and_report_residue_neighborhood_idxs_no_interactive_true_ctc_percentage(self):
        ctc_freq = [1.,.5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]

        per_residx_ctc_idxs = contacts.select_and_report_residue_neighborhood_idxs(ctc_freq, self.ctc_cutoff_Ang, self.residxs,
                                                                        self.fragments, ctc_residxs_pairs,
                                                                        self.geom.top,
                                                                        ctcs_kept=.5, restrict_to_resSeq=None,
                                                                        interactive=False)
        assert (_np.array_equal(per_residx_ctc_idxs[0], [0]))


    def test_select_and_report_residue_neighborhood_idxs_no_interactive_ctc_percentage_no_ctcs(self):
        ctc_freq = [0, 0]
        ctc_residxs_pairs = [[0, 1], [2, 1]]

        per_residx_ctc_idxs = contacts.select_and_report_residue_neighborhood_idxs(ctc_freq, self.ctc_cutoff_Ang, self.residxs,
                                                                        self.fragments, ctc_residxs_pairs,
                                                                        self.geom.top,
                                                                        ctcs_kept=.5, restrict_to_resSeq=None,
                                                                        interactive=False)
        _np.testing.assert_array_equal(per_residx_ctc_idxs[0],[])
        _np.testing.assert_array_equal(per_residx_ctc_idxs[1],[])

    def test_select_and_report_residue_neighborhood_idxs_keyboard_interrupt(self):
        ctc_freq = [1.,.5]
        per_residx_ctc_idxs = [[0, 1], [2, 1]]
        with unittest.mock.patch('builtins.input', side_effect=KeyboardInterrupt):
            resname2residx, resname2fragidx = residues_from_descriptors("GLU30", self.fragments,
                                                                        self.geom.top)

            ctc_freq = contacts.select_and_report_residue_neighborhood_idxs(ctc_freq, self.ctc_cutoff_Ang, resname2residx,
                                                                            self.fragments, per_residx_ctc_idxs,
                                                                            self.geom.top,
                                                                            ctcs_kept=5, restrict_to_resSeq=None,
                                                                            interactive=True)
            assert ctc_freq == {}


class TestBaseClassContactGroup(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.actor_pdb)
        self.top = self.geom.top

        self.cp1 = contacts.ContactPair([0, 1], [[.1, .2, .3], [.4]], [[1, 2, 3], [1]])
        self.cp2 = contacts.ContactPair([0, 2], [[.15, .35, .25], [.16]], [[1, 2, 3], [1]])
        self.cp3 = contacts.ContactPair([1, 2], [[.15, .30, .35], [.45]], [[1, 2, 3], [1]])

        self.cp1_wtop = contacts.ContactPair([0, 1], [[.1, .2, .3]], [[1, 2, 3]], top=self.top)
        self.cp2_wtop = contacts.ContactPair([0, 2], [[.15, .25, .35]], [[1, 2, 3]], top=self.top)
        self.cp3_wtop_other = contacts.ContactPair([0, 2], [[.15, .25, .35]], [[1, 2, 3]],
                                                   top=md.load(test_filenames.small_monomer).top)

        self.cp1_w_anchor_and_frags = contacts.ContactPair([0, 1], [[.1, .2, .3], [.4, .5]], [[1, 2, 3], [1, 2]],
                                                           fragment_names=["fragA", "fragB"],
                                                           fragment_colors=["r", "b"],
                                                           anchor_residue_idx=0)

        self.cp2_w_anchor_and_frags = contacts.ContactPair([0, 2], [[.15, .25, .35], [.45, .45]], [[1, 2, 3], [1, 2]],
                                                           fragment_names=["fragA", "fragC"],
                                                           fragment_colors=["r", "g"],
                                                           anchor_residue_idx=0)

        self.cp3_w_anchor_and_frags_wrong_anchor_color = contacts.ContactPair([0, 3], [[.15, .25, .35], [.45, .45]],
                                                                              [[1, 2, 3], [1, 2]],
                                                                              fragment_names=["fragA", "fragC"],
                                                                              fragment_colors=["y", "g"],
                                                                              anchor_residue_idx=0)

        self.cp1_w_anchor_and_frags_and_top = contacts.ContactPair([0, 1], [[.1, .2, .3], [.4, .5]],
                                                                   [[1, 2, 3], [1, 2]],
                                                                   fragment_names=["fragA", "fragB"],
                                                                   anchor_residue_idx=0,
                                                                   top=self.top)
        self.cp2_w_anchor_and_frags_and_top = contacts.ContactPair([0, 2], [[.15, .25, .35], [.45, .45]],
                                                                   [[1, 2, 3], [1, 2]],
                                                                   fragment_names=["fragA", "fragC"],
                                                                   anchor_residue_idx=0,
                                                                   top=self.top)

        self.cp1_wtop_and_conslabs = contacts.ContactPair([0, 1], [[.1, .2, .3]], [[1, 2, 3]],
                                                          consensus_labels=["3.50", "4.50"],
                                                          top=self.top)
        self.cp2_wtop_and_conslabs = contacts.ContactPair([0, 2], [[.15, .25, .35]], [[1, 2, 3]],
                                                          consensus_labels=["3.50", "5.50"],
                                                          top=self.top)
        self.cp3_wtop_and_conslabs = contacts.ContactPair([1, 2], [[.25, .15, .35]], [[1, 2, 3]],
                                                          consensus_labels=["4.50", "5.50"],
                                                          top=self.top)

        self.cp3_wtop_and_wrong_conslabs = contacts.ContactPair([1, 2], [[.1, .2, 3]], [[1, 2, 3]],
                                                                consensus_labels=["4.50", "550"],
                                                                top=self.top)

        self.cp4_wtop_and_conslabs = contacts.ContactPair([3, 2], [[.25, .25, .35]], [[1, 2, 3]],
                                                          consensus_labels=["3.51", "5.50"],
                                                          top=self.top)

        self.cp5_wtop_and_wo_conslabs = contacts.ContactPair([4, 5], [[.2, .25, .35]], [[1, 2, 3]],
                                                             # consensus_labels=["3.51", "5.50"],
                                                             top=self.top)

        # Completely bogus contacts but testable
        #print(self.top.residue(1))
        #print([aa for aa in self.top.residue(1).atoms])
        self.atom_BB0 = list(self.top.residue(0).atoms_by_name("CA"))[0].index
        self.atom_SC0 = list(self.top.residue(0).atoms_by_name("CB"))[0].index
        self.atom_BB1 = list(self.top.residue(1).atoms_by_name("CA"))[0].index
        self.atom_SC1 = list(self.top.residue(1).atoms_by_name("CB"))[0].index
        self.atom_BB2 = list(self.top.residue(2).atoms_by_name("CA"))[0].index
        self.atom_SC2 = list(self.top.residue(2).atoms_by_name("CB"))[0].index

        self.cp1_w_atom_types = contacts.ContactPair([0, 1],
                                                     [[.15, .25, .35, .45]],
                                                     [[0, 1, 2, 3]],
                                                     atom_pair_trajs=[
                                                         [[self.atom_BB0, self.atom_SC1],
                                                          [self.atom_BB0, self.atom_BB1],
                                                          [self.atom_BB0, self.atom_BB1],
                                                          [self.atom_BB0, self.atom_BB1]
                                                          ],
                                                     ],
                                                     top=self.top
                                                     )
        #At 3.5, the above cp1 gives BB-SC, BB-BB, BB-BB:
        # 2/3 BB-BB, 1/3 BB-SC
        self.cp2_w_atom_types = contacts.ContactPair([0, 2],
                                                     [[.15, .25, .35, .45]],
                                                     [[0, 1, 2, 3]],
                                                     atom_pair_trajs=[
                                                         [[self.atom_BB0, self.atom_SC2],
                                                          [self.atom_BB0, self.atom_SC2],
                                                          [self.atom_SC0, self.atom_BB2],
                                                          [self.atom_SC0, self.atom_BB2]
                                                          ],
                                                     ],
                                                     top=self.top
                                                     )
        self.cp1_w_atom_types_0_1_switched = contacts.ContactPair([1, 0],
                                                     [[.15, .25, .35, .45]],
                                                     [[0, 1, 2, 3]],
                                                     atom_pair_trajs=[
                                                        [[self.atom_SC1, self.atom_BB0],
                                                         [self.atom_BB1, self.atom_BB0],
                                                         [self.atom_BB1, self.atom_BB0],
                                                         [self.atom_BB1, self.atom_BB0]]
                                                     ],
                                                     top=self.top,
                                                    anchor_residue_idx=0,
                                                     )
        #At 3.5, the above cp1 gives SC-BB, BB-BB, BB-BB:
        # 2/3 BB-BB, 1/3 SC-BB

class TestContactGroup(TestBaseClassContactGroup):


    def setUp(self):
        super(TestContactGroup, self).setUp()
        # test_works_minimal no top
        self.CG_cp1_cp2 = contacts.ContactGroup([self.cp1, self.cp2])
        assert self.CG_cp1_cp2.topology is self.CG_cp1_cp2.top is None

        # test_works_minimal_w top
        self.CG_cp1_wtop_cp2_wtop = contacts.ContactGroup([self.cp1_wtop, self.cp2_wtop], top=self.top)
        assert self.CG_cp1_wtop_cp2_wtop.topology is self.CG_cp1_wtop_cp2_wtop.top is self.top

    def test_works_minimal_top_raises(self):
        with self.assertRaises(AssertionError):
            contacts.ContactGroup([self.cp1, self.cp2], top=self.top)

    def test_n_properties(self):
        CG = self.CG_cp1_cp2
        _np.testing.assert_equal(CG.n_ctcs, 2)
        _np.testing.assert_equal(CG.n_trajs, 2)
        _np.testing.assert_array_equal(CG.n_frames, [3, 1])

    def test_time_properties(self):
        CG = self.CG_cp1_cp2
        _np.testing.assert_equal(3, CG.time_max)
        _np.testing.assert_array_equal([1, 2, 3], CG.time_arrays[0])
        _np.testing.assert_array_equal([1], CG.time_arrays[1])

    def test_wrong_top_raises(self):
        with self.assertRaises(ValueError):
            contacts.ContactGroup([self.cp1_wtop, self.cp2_wtop,
                                   self.cp3_wtop_other])

    def test_Residues(self):
        CG = self.CG_cp1_wtop_cp2_wtop
        _np.testing.assert_array_equal([[0, 1],
                                        [0, 2]],
                                       CG.res_idxs_pairs)

        _np.testing.assert_equal(CG.residue_names_short[0][0], "E30")
        _np.testing.assert_equal(CG.residue_names_short[0][1], "V31")
        _np.testing.assert_equal(CG.residue_names_short[1][0], "E30")
        _np.testing.assert_equal(CG.residue_names_short[1][1], "W32")

    def test_labels(self):
        CG = self.CG_cp1_wtop_cp2_wtop
        _np.testing.assert_equal(CG.ctc_labels[0], "GLU30-VAL31")
        _np.testing.assert_equal(CG.ctc_labels[1], "GLU30-TRP32")
        _np.testing.assert_equal(CG.ctc_labels_short[0], "E30-V31")
        _np.testing.assert_equal(CG.ctc_labels_short[1], "E30-W32")
        _np.testing.assert_equal(CG.trajlabels[0], "traj 0")
        _np.testing.assert_equal(CG.consensus_labels[0][0], None)
        _np.testing.assert_equal(CG.consensus_labels[0][1], None)
        _np.testing.assert_equal(CG.consensus_labels[1][0], None)
        _np.testing.assert_equal(CG.consensus_labels[1][1], None)

        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs])

        _np.testing.assert_equal(CG.consensus_labels[0][0], "3.50")
        _np.testing.assert_equal(CG.consensus_labels[0][1], "4.50")
        _np.testing.assert_equal(CG.consensus_labels[1][0], "3.50")
        _np.testing.assert_equal(CG.consensus_labels[1][1], "5.50")

        _np.testing.assert_equal(CG.ctc_labels_w_fragments_short_AA[0], "E30@3.50-V31@4.50")
        _np.testing.assert_equal(CG.ctc_labels_w_fragments_short_AA[1], "E30@3.50-W32@5.50")

    def test_consensuslabel2resname(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs])
        _np.testing.assert_equal(CG.consensuslabel2resname["3.50"], "E30")
        _np.testing.assert_equal(CG.consensuslabel2resname["4.50"], "V31")
        _np.testing.assert_equal(CG.consensuslabel2resname["5.50"], "W32")

    def test_relabel_consensus(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    contacts.ContactPair([0, 2], [[.15, .25, .35]], [[1, 2, 3]],
                                                         consensus_labels=["3.50", None],
                                                         top=self.top)])

        CG.relabel_consensus()
        _np.testing.assert_array_equal(CG.consensus_labels,[["3.50","4.50"],["3.50","W32"]])

    def test_relabel_consensus_w_extra(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    contacts.ContactPair([0, 2], [[.15, .25, .35]], [[1, 2, 3]],
                                                         consensus_labels=["3.50", None],
                                                         top=self.top)])

        CG.relabel_consensus(new_labels={"E30":"mut"})
        _np.testing.assert_array_equal(CG.consensus_labels,[["mut","4.50"],["mut","W32"]])

    def test_residx2resnameshort(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs])
        _np.testing.assert_equal(CG.residx2resnameshort[0], "E30")
        _np.testing.assert_equal(CG.residx2resnameshort[1], "V31")
        _np.testing.assert_equal(CG.residx2resnameshort[2], "W32")

    def test_residx2consensuslabel(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs,
                                    self.cp5_wtop_and_wo_conslabs,
                                    ])
        assert len(CG.residx2consensuslabel) == len(_np.unique(CG.res_idxs_pairs))
        _np.testing.assert_equal(CG.residx2consensuslabel[0], "3.50")
        _np.testing.assert_equal(CG.residx2consensuslabel[1], "4.50")
        _np.testing.assert_equal(CG.residx2consensuslabel[2], "5.50")
        _np.testing.assert_equal(CG.residx2consensuslabel[4], None)
        _np.testing.assert_equal(CG.residx2consensuslabel[5], None)

    def test_residx2resnamefragnamebest(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs,
                                    self.cp5_wtop_and_wo_conslabs,
                                    ])
        residx2resnamefragnamebest = CG.residx2resnamefragnamebest("@")
        assert len(residx2resnamefragnamebest) == len(_np.unique(CG.res_idxs_pairs))
        _np.testing.assert_equal(residx2resnamefragnamebest[0], "E30@3.50")
        _np.testing.assert_equal(residx2resnamefragnamebest[1], "V31@4.50")
        _np.testing.assert_equal(residx2resnamefragnamebest[2], "W32@5.50")
        _np.testing.assert_equal(residx2resnamefragnamebest[4], "V34")
        _np.testing.assert_equal(residx2resnamefragnamebest[5], "G35")

    def test_fragment_names_best_fragnames(self):
        CG = contacts.ContactGroup([self.cp1_w_anchor_and_frags,
                                    self.cp2_w_anchor_and_frags],
                                   neighbors_excluded=0
                                   )

        _np.testing.assert_array_equal(CG.fragment_names_best[0], ["fragA", "fragB"])
        _np.testing.assert_array_equal(CG.fragment_names_best[1], ["fragA", "fragC"])

    def test_fragment_names_best_consensus(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs])

        _np.testing.assert_array_equal(CG.fragment_names_best[0], ["3.50", "4.50"])
        _np.testing.assert_array_equal(CG.fragment_names_best[1], ["3.50", "5.50"])

    def test_consensus_labels_wrong_raises(self):
        with self.assertRaises(AssertionError):
            contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp3_wtop_and_wrong_conslabs])

    def test_neighborhoods_raises(self):
        CG = self.CG_cp1_cp2
        _np.testing.assert_equal(CG.shared_anchor_residue_index, None)
        with self.assertRaises(AssertionError):
            CG.anchor_res_and_fragment_str
        with self.assertRaises(AssertionError):
            CG.anchor_res_and_fragment_str_short
        with self.assertRaises(AssertionError):
            CG.partner_res_and_fragment_labels
        with self.assertRaises(AssertionError):
            CG.anchor_fragment_color
        with self.assertRaises(AssertionError):
            CG.partner_res_and_fragment_labels
        with self.assertRaises(AssertionError):
            CG.partner_res_and_fragment_labels_short

    def test_neighborhoods(self):
        CG = contacts.ContactGroup([self.cp1_w_anchor_and_frags,
                                    self.cp2_w_anchor_and_frags],
                                   neighbors_excluded=0
                                   )
        assert CG.is_neighborhood
        _np.testing.assert_equal(CG.shared_anchor_residue_index, 0)
        _np.testing.assert_equal(CG.anchor_res_and_fragment_str, "0@fragA")
        _np.testing.assert_equal(CG.anchor_res_and_fragment_str_short, "0@fragA")
        _np.testing.assert_equal(CG.partner_res_and_fragment_labels[0], "1@fragB")
        _np.testing.assert_equal(CG.partner_res_and_fragment_labels[1], "2@fragC")
        _np.testing.assert_equal(CG.partner_res_and_fragment_labels_short[0], "1@fragB")
        _np.testing.assert_equal(CG.partner_res_and_fragment_labels_short[1], "2@fragC")
        _np.testing.assert_equal(CG.anchor_fragment_color, "r")

    def test_neighborhood_w_partner_color(self):
        CG = contacts.ContactGroup([self.cp1_w_anchor_and_frags,
                                    self.cp2_w_anchor_and_frags],
                                   neighbors_excluded=0
                                   )
        _np.testing.assert_array_equal(["b", "g"], CG.partner_fragment_colors)

    def test_neighborhoods_wrong_anchor_color(self):
        CG = contacts.ContactGroup([self.cp1_w_anchor_and_frags,
                                    self.cp2_w_anchor_and_frags,
                                    self.cp3_w_anchor_and_frags_wrong_anchor_color],
                                   neighbors_excluded=0
                                   )
        assert CG.anchor_fragment_color is None

    def test_neighborhoods_w_top(self):
        CG = contacts.ContactGroup([self.cp1_w_anchor_and_frags_and_top,
                                    self.cp2_w_anchor_and_frags_and_top],
                                   neighbors_excluded=0
                                   )
        _np.testing.assert_equal(CG.shared_anchor_residue_index, 0)
        _np.testing.assert_equal(CG.anchor_res_and_fragment_str, "GLU30@fragA")
        _np.testing.assert_equal(CG.anchor_res_and_fragment_str_short, "E30@fragA")
        _np.testing.assert_equal(CG.partner_res_and_fragment_labels[0], "VAL31@fragB")
        _np.testing.assert_equal(CG.partner_res_and_fragment_labels[1], "TRP32@fragC")
        _np.testing.assert_equal(CG.partner_res_and_fragment_labels_short[0], "V31@fragB")
        _np.testing.assert_equal(CG.partner_res_and_fragment_labels_short[1], "W32@fragC")

    def test_residx2ctcidx(self):
        CG = self.CG_cp1_cp2
        _np.testing.assert_array_equal(CG.residx2ctcidx(1), [[0, 1]])
        _np.testing.assert_array_equal(CG.residx2ctcidx(2), [[1, 1]])
        _np.testing.assert_array_equal(CG.residx2ctcidx(0), [[0, 0],
                                                             [1, 0]])

    def test_binarize_trajs(self):
        CG = self.CG_cp1_cp2
        bintrajs = CG.binarize_trajs(2.5)
        _np.testing.assert_array_equal([1, 1, 0], bintrajs[0][0])
        _np.testing.assert_array_equal([0], bintrajs[0][1])
        _np.testing.assert_array_equal([1, 0, 1], bintrajs[1][0])
        _np.testing.assert_array_equal([1], bintrajs[1][1])

    def test_binarize_trajs_order(self):
        CG = contacts.ContactGroup([self.cp1, self.cp2, self.cp3])
        bintrajs = CG.binarize_trajs(2.5, order="traj")

        traj_1_ctc_1 = bintrajs[0][:, 0]
        traj_1_ctc_2 = bintrajs[0][:, 1]
        traj_1_ctc_3 = bintrajs[0][:, 2]

        traj_2_ctc_1 = bintrajs[1][:, 0]
        traj_2_ctc_2 = bintrajs[1][:, 1]
        traj_2_ctc_3 = bintrajs[1][:, 2]

        _np.testing.assert_array_equal([1, 1, 0], traj_1_ctc_1)
        _np.testing.assert_array_equal([1, 0, 1], traj_1_ctc_2)
        _np.testing.assert_array_equal([1, 0, 0], traj_1_ctc_3)

        _np.testing.assert_array_equal([0], traj_2_ctc_1)
        _np.testing.assert_array_equal([1], traj_2_ctc_2)
        _np.testing.assert_array_equal([0], traj_2_ctc_3)

    def test_distance_distributions(self):
        CG = self.CG_cp1_cp2
        h1, x1 = self.cp1.distro_overall_trajs(bins=10)
        h2, x2 = self.cp2.distro_overall_trajs(bins=10)
        distros = CG._distributions_of_distances(bins=10)
        _np.testing.assert_array_equal(distros[0][0], h1)
        _np.testing.assert_array_equal(distros[0][1], x1)
        _np.testing.assert_array_equal(distros[1][0], h2)
        _np.testing.assert_array_equal(distros[1][1], x2)

    def test_distirbution_dicts(self):
        CG = self.CG_cp1_cp2
        dicts = CG.distribution_dicts(bins=10,pad_label=False)
        _np.testing.assert_array_equal(list(dicts.keys()), ["0-1", "0-2"])
        for a, b in zip(dicts.values(), CG._distributions_of_distances(bins=10)):
            _np.testing.assert_array_equal(a[0],b[0])
            _np.testing.assert_array_equal(a[1],b[1])


    def test_time_traces_n_ctcs(self):
        CG = contacts.ContactGroup([self.cp1, self.cp2, self.cp3])
        ncts_tt = CG.n_ctcs_timetraces(2.5)
        _np.testing.assert_array_equal([3, 1, 1], ncts_tt[0])
        _np.testing.assert_array_equal([1], ncts_tt[1])

    def test_no_interface(self):
        CG = contacts.ContactGroup([self.cp1, self.cp2, self.cp3])
        assert CG.is_interface is False
        _np.testing.assert_array_equal(CG.interface_residxs, [[], []])
        _np.testing.assert_array_equal(CG.interface_residue_names_w_best_fragments_short, [[], []])
        _np.testing.assert_array_equal(CG.interface_reslabels_short, [[], []])
        _np.testing.assert_array_equal(CG.interface_labels_consensus, [[], []])
        with self.assertRaises(AssertionError):
            CG.plot_interface_frequency_matrix(None)
        assert CG.interface_frequency_matrix(None) is None

    def test_repframe(self):
        CG = contacts.ContactGroup([contacts.ContactPair([0, 1],
                                                         [[1],
                                                          [10],
                                                          [1, 15, 10, 15, 15]],
                                                         [[0],
                                                          [0],
                                                          [0, 1, 2, 3, 4]])])
        repframes, RMSDd, values = CG.repframes()
        traj_idx, frame_idx = repframes[0]
        assert traj_idx  == 2
        assert frame_idx == 1
        _np.testing.assert_array_equal(values[0], [15]) #traj 2, frame 1
        assert RMSDd[0] == 0 # since we're in 1D, the mode is one frame for all dimensions, and that frame has zero distance to the mode

    def test_repframe_mean(self):
        CG = contacts.ContactGroup([contacts.ContactPair([0, 1],
                                                         [[1],
                                                          [10],
                                                          [1, 15, 10, 15, 15]],
                                                         [[0],
                                                          [0],
                                                          [0, 1, 2, 3, 4]])])
        repframes, RMSDd, values = CG.repframes(scheme="mean")
        traj_idx, frame_idx = repframes[0]
        assert traj_idx  == 1
        assert frame_idx == 0
        ref_mean = _np.mean([[1]+[10]+[1, 15, 10, 15, 15]])
        _np.testing.assert_array_equal(values[0], [10])  # traj 1, frame 0
        assert RMSDd[0] == _np.abs(ref_mean-10)

    def test_repframe_minima(self):
        CG = contacts.ContactGroup([contacts.ContactPair([0, 1],
                                                         [[2],
                                                          [10],
                                                          [15, 15, 10, 15, 1]],
                                                         [[0],
                                                          [0],
                                                          [0, 1, 2, 3, 4]])])
        repframes, RMSDd, values = CG.repframes(scheme="min")
        traj_idx, frame_idx = repframes[0]
        assert traj_idx  == 2
        assert frame_idx == 4
        ref_min = 1
        _np.testing.assert_array_equal(values[0], ref_min)
        assert RMSDd[0] == 0

    def test_repframe_maxima(self):
        CG = contacts.ContactGroup([contacts.ContactPair([0, 1],
                                                         [[2],
                                                          [16],
                                                          [15, 15, 10, 15, 1]],
                                                         [[0],
                                                          [0],
                                                          [0, 1, 2, 3, 4]])])
        repframes, RMSDd, values = CG.repframes(scheme="max")
        traj_idx, frame_idx = repframes[0]
        assert traj_idx  == 1
        assert frame_idx == 0
        ref_max = 16
        _np.testing.assert_array_equal(values[0], ref_max)
        assert RMSDd[0] == 0

    def test_repframe_w_traj_violines_many_frames_just_runs(self):
        CG = examples.ContactGroupL394()
        with _TDir(suffix="_mdciao_example_CG") as t:
            try:
                examples.examples._link(test_filenames.traj_xtc,
                                        examples.examples._path.join(t, examples.examples._path.basename(test_filenames.traj_xtc)))
            except OSError:
                examples.examples._shcopy(test_filenames.traj_xtc,
                                          examples.examples._path.join(t, examples.examples._path.basename(test_filenames.traj_xtc)))
            with examples.examples.remember_cwd():
                examples.examples._chdir(t)
                repframes, RMSDd, values, trajs = CG.repframes(show_violins=True, return_traj=True, n_frames=10)
                assert len(repframes)==len(RMSDd)==len(values)==len(trajs)==10
                assert isinstance(trajs[0], md.Trajectory)

    def test_select_by_residues_CSV(self):
        CG = _mdcsites([{"name": "test_random",
                         "pairs": {"residx": [[100, 200],
                                              [100, 300],
                                              [10, 40],
                                              [200, 50],
                                              [20, 40],
                                              [10, 20]]
                                   }}],
                       test_filenames.traj_xtc,
                       test_filenames.top_pdb,
                       no_disk=True,
                       figures=False)["test_random"]
        keys = [str(CG.top.residue(ii)) for ii in [200,10, 1]]
        CSV = ','.join(keys)

        new_CG : contacts.ContactGroup = CG.select_by_residues(CSVexpression=CSV)
        assert new_CG.n_ctcs == 4
        assert new_CG.contact_pairs[0] is CG.contact_pairs[0]
        assert new_CG.contact_pairs[1] is CG.contact_pairs[2]
        assert new_CG.contact_pairs[2] is CG.contact_pairs[3]
        assert new_CG.contact_pairs[3] is CG.contact_pairs[5]
        assert isinstance(new_CG,contacts.ContactGroup)

        new_CG_dict = CG.select_by_residues(CSVexpression=CSV, merge=False)

        self.assertSequenceEqual(list(new_CG_dict.keys()),CSV.split(","))

        assert new_CG_dict[keys[0]].n_ctcs == 2
        assert new_CG_dict[keys[0]].contact_pairs[0] is CG.contact_pairs[0]
        assert new_CG_dict[keys[0]].contact_pairs[1] is CG.contact_pairs[3]

        assert new_CG_dict[keys[1]].n_ctcs == 2
        assert new_CG_dict[keys[1]].contact_pairs[0] is CG.contact_pairs[2]
        assert new_CG_dict[keys[1]].contact_pairs[1] is CG.contact_pairs[5]

        assert new_CG_dict[keys[2]] is None


    def test_select_by_residues_residue_indices(self):
        CG = _mdcsites([{"name": "test_random",
                         "pairs": {"residx": [[100, 200],
                                              [100, 300],
                                              [10, 40],
                                              [200, 50],
                                              [20, 40],
                                              [10, 20]]
                                   }}],
                       test_filenames.traj_xtc,
                       test_filenames.top_pdb,
                       no_disk=True,
                       figures=False)["test_random"]

        residue_indices = [10, 40, 1]
        new_CG : contacts.ContactGroup = CG.select_by_residues(residue_indices=residue_indices)
        assert new_CG.n_ctcs == 3
        assert new_CG.contact_pairs[0] is CG.contact_pairs[2]
        assert new_CG.contact_pairs[1] is CG.contact_pairs[4]
        assert new_CG.contact_pairs[2] is CG.contact_pairs[5]
        assert isinstance(new_CG,contacts.ContactGroup)

        new_CG_dict = CG.select_by_residues(residue_indices=residue_indices, merge=False)
        self.assertSequenceEqual(list(new_CG_dict.keys()),residue_indices)
        assert new_CG_dict[residue_indices[0]].n_ctcs == 2
        assert new_CG_dict[residue_indices[0]].contact_pairs[0] is CG.contact_pairs[2]
        assert new_CG_dict[residue_indices[0]].contact_pairs[1] is CG.contact_pairs[5]

        assert new_CG_dict[residue_indices[1]].n_ctcs == 2
        assert new_CG_dict[residue_indices[1]].contact_pairs[0] is CG.contact_pairs[2]
        assert new_CG_dict[residue_indices[1]].contact_pairs[1] is CG.contact_pairs[4]

        assert new_CG_dict[residue_indices[2]] is None

    def test_to_select_by_residues_residue_indices_n_residues_is_2(self):
        CG = _mdcsites([{"name": "test_random",
                         "pairs": {"residx": [[100, 200],
                                              [100, 300],
                                              [10, 40], #2
                                              [200, 50],
                                              [20, 40], #4
                                              [50, 20], #5
                                              [70, 20]] # this last one woud've been pikced up if n_residues=1
                                   }}],
                       test_filenames.traj_xtc,
                       test_filenames.top_pdb,
                       no_disk=True,
                       figures=False)["test_random"]

        residue_indices = [10, 40, 20, 50]
        new_CG : contacts.ContactGroup = CG.select_by_residues(residue_indices=residue_indices, n_residues=2)

        assert isinstance(new_CG, contacts.ContactGroup)
        assert new_CG.n_ctcs == 3
        assert new_CG.contact_pairs[0] is CG.contact_pairs[2]
        assert new_CG.contact_pairs[1] is CG.contact_pairs[4]
        assert new_CG.contact_pairs[2] is CG.contact_pairs[5]


        new_CG_dict = CG.select_by_residues(residue_indices=residue_indices, merge=False, n_residues=2)
        self.assertSequenceEqual(list(new_CG_dict.keys()),residue_indices)
        assert new_CG_dict[residue_indices[0]].n_ctcs == 1
        assert new_CG_dict[residue_indices[0]].contact_pairs[0] is CG.contact_pairs[2]

        assert new_CG_dict[residue_indices[1]].n_ctcs == 2
        assert new_CG_dict[residue_indices[1]].contact_pairs[0] is CG.contact_pairs[2]
        assert new_CG_dict[residue_indices[1]].contact_pairs[1] is CG.contact_pairs[4]

        assert new_CG_dict[residue_indices[2]].n_ctcs == 2
        assert new_CG_dict[residue_indices[2]].contact_pairs[0] is CG.contact_pairs[4]
        assert new_CG_dict[residue_indices[2]].contact_pairs[1] is CG.contact_pairs[5]

        assert new_CG_dict[residue_indices[3]].n_ctcs == 1
        assert new_CG_dict[residue_indices[3]].contact_pairs[0] is CG.contact_pairs[5]

    def test_to_select_by_residues_residue_pairs(self):
        CG = _mdcsites([{"name": "test_random",
                         "pairs": {"residx": [[100, 200],
                                              [100, 300],
                                              [10, 40], #2
                                              [200, 50],
                                              [20, 40], #4
                                              [50, 20], #5
                                              [70, 20]] # this last one woud've been pikced up if n_residues=1
                                   }}],
                       test_filenames.traj_xtc,
                       test_filenames.top_pdb,
                       no_disk=True,
                       figures=False)["test_random"]

        residue_pairs = [[10, 40], [200, 100], [10, 30]]

        new_CG : contacts.ContactGroup = CG.select_by_residues(residue_pairs=residue_pairs)
        assert isinstance(new_CG, contacts.ContactGroup)
        assert new_CG.n_ctcs == 2
        assert new_CG.contact_pairs[0] is CG.contact_pairs[2]
        assert new_CG.contact_pairs[1] is CG.contact_pairs[0]

        def test_to_select_by_residues_consensus(self):
            CG = examples.ContactGroupL394()

            new_CG : contacts.ContactGroup = CG.select_by_residues("G.H.21")
            assert isinstance(new_CG, contacts.ContactGroup)
            assert new_CG.n_ctcs == 1
            assert new_CG.contact_pairs[0] is CG.contact_pairs[1]


    def test_to_ContactGroups_per_traj(self):
        traj = md.load(test_filenames.traj_xtc_stride_20, top=test_filenames.top_pdb)
        CG : contacts.ContactGroup = _mdcli.residue_neighborhoods("L394",[traj, traj[:-1]],figures=False, no_disk=True)[353]
        new_CGs = CG.to_ContactGroups_per_traj()
        assert len(new_CGs)==2
        self.assertListEqual(list(new_CGs.keys()), CG.trajlabels)
        for ii, (key, iCG) in enumerate(new_CGs.items()):
            iCG : contacts.ContactGroup
            # Identical stuff
            _np.testing.assert_array_equal(CG.res_idxs_pairs, iCG.res_idxs_pairs)
            _np.testing.assert_array_equal(CG.consensus_labels, iCG.consensus_labels)
            _np.testing.assert_array_equal(CG.ctc_labels, iCG.ctc_labels)
            _np.testing.assert_array_equal(CG.neighbors_excluded, iCG.neighbors_excluded)
            _np.testing.assert_array_equal(CG.max_cutoff_Ang, iCG.max_cutoff_Ang)
            _np.testing.assert_array_equal(CG.interface_fragments, iCG.interface_fragments)
            _np.testing.assert_array_equal(CG.max_cutoff_Ang, iCG.max_cutoff_Ang)
            _np.testing.assert_array_equal(CG.n_ctcs, iCG.n_ctcs)
            _np.testing.assert_array_equal(CG.fragment_names_best, iCG.fragment_names_best)
            _np.testing.assert_array_equal(CG.is_neighborhood, iCG.is_neighborhood)
            _np.testing.assert_array_equal(CG.name, iCG.name)

            # Index dependent stuff
            _np.testing.assert_array_equal(CG.time_arrays[ii], iCG.time_arrays[0])
            _np.testing.assert_array_equal(CG.n_frames[ii], iCG.n_frames_total)
            _np.testing.assert_array_equal(iCG.trajlabels[0], 'mdtraj.00')

            for jj in range(CG.n_ctcs):
                _np.testing.assert_array_equal( CG.contact_pairs[jj].time_traces.ctc_trajs[ii],
                                               iCG.contact_pairs[jj].time_traces.ctc_trajs[0])
                _np.testing.assert_array_equal( CG.contact_pairs[jj].time_traces.atom_pair_trajs[ii],
                                               iCG.contact_pairs[jj].time_traces.atom_pair_trajs[0])
                assert CG.contact_pairs[jj].time_traces.trajs[ii] is iCG.contact_pairs[jj].time_traces.trajs[0]

class TestContactGroup_select_by_frames(TestBaseClassContacts):

    @classmethod
    def setUpClass(cls):
        super(TestContactGroup_select_by_frames, cls).setUp(cls)
        cls.second_traj = cls.traj[::-1][:20]
        cls.trajs = [cls.traj, cls.second_traj]
        b = _io.StringIO()
        with _contextlib.redirect_stdout(b):
            cls.CG : contacts.ContactGroup = _mdcli.sites([{"name": "test", "pairs":{"residx":[[100,200], [100,300]]}}],
                                                          [cls.file_xtc, cls.second_traj],
                                                          topology=cls.pdb_file,
                                   no_disk=True, figures=False)["test"]
        b.close()

        cls.unchanged_propeties = ['consensus_labels',
                                   'ctc_labels',
                                   'ctc_labels_short',
                                   'ctc_labels_w_fragments_short_AA',
                                   'fragment_names_best',
                                   'interface_fragments',
                                   'interface_labels_consensus',
                                   'interface_residue_names_w_best_fragments_short',
                                   'interface_residxs',
                                   'interface_reslabels_short',
                                   'is_interface',
                                   'is_neighborhood',
                                   'max_cutoff_Ang',
                                   'n_ctcs',
                                   'n_trajs',
                                   # 'name', unsure whether the name should be conserved
                                   'neighbors_excluded',
                                   'res_idxs_pairs',
                                   'residue_names_long',
                                   'residue_names_short',
                                   'shared_anchor_residue_index',
                                   #'trajlabels'
                                   ]

    def test_first_5(cls):
        newCG = cls.CG.select_by_frames(5)
        assert (newCG.n_ctcs == cls.CG.n_ctcs)
        assert (newCG.n_trajs==cls.CG.n_trajs)
        assert (newCG.top is cls.CG.top)
        _np.testing.assert_equal(newCG.trajlabels, ["mdtraj.00","mdtraj.01"])
        _np.testing.assert_equal(newCG.n_frames,[5,5])

        #Blanket check for most array-like properties
        for iattr in cls.unchanged_propeties:
            _np.testing.assert_array_equal(getattr(newCG, iattr), getattr(cls.CG, iattr), iattr)

        # Tests of the time array
        _np.testing.assert_array_equal(newCG.time_arrays, [time[:5] for time in cls.CG.time_arrays])

        # Test of the time-traces
        for ii in range(newCG.n_trajs):
            _np.testing.assert_array_equal([CP.time_traces.ctc_trajs[ii]       for CP in newCG.contact_pairs],
                                           [CP.time_traces.ctc_trajs[ii][:5] for CP in cls.CG.contact_pairs])
            _np.testing.assert_array_equal([CP.time_traces.atom_pair_trajs[ii]       for CP in newCG.contact_pairs],
                                           [CP.time_traces.atom_pair_trajs[ii][:5] for CP in cls.CG.contact_pairs])
            _np.testing.assert_array_equal([CP.time_traces.trajs[ii].xyz for CP in newCG.contact_pairs],
                                           [cls.trajs[ii].xyz[:5] for _ in cls.CG.contact_pairs])


    def test_last_5(cls):
        newCG: contacts.ContactGroup = cls.CG.select_by_frames(-5)
        assert (newCG.n_ctcs == cls.CG.n_ctcs)
        assert (newCG.n_trajs == cls.CG.n_trajs)
        assert (newCG.top is cls.CG.top)
        _np.testing.assert_equal(newCG.trajlabels, ["mdtraj.00", "mdtraj.01"])
        _np.testing.assert_equal(newCG.n_frames, [5, 5])

        # Blanket check for most array-like properties
        for iattr in cls.unchanged_propeties:
            _np.testing.assert_array_equal(getattr(newCG, iattr), getattr(cls.CG, iattr), iattr)

        # Tests of the time array
        _np.testing.assert_array_equal(newCG.time_arrays, [time[-5:] for time in cls.CG.time_arrays])

        # Test of the time-traces
        for ii in range(newCG.n_trajs):
            _np.testing.assert_array_equal([CP.time_traces.ctc_trajs[ii] for CP in newCG.contact_pairs],
                                           [CP.time_traces.ctc_trajs[ii][-5:] for CP in cls.CG.contact_pairs])
            _np.testing.assert_array_equal([CP.time_traces.atom_pair_trajs[ii] for CP in newCG.contact_pairs],
                                           [CP.time_traces.atom_pair_trajs[ii][-5:] for CP in
                                            cls.CG.contact_pairs])
            _np.testing.assert_array_equal([CP.time_traces.trajs[ii].xyz for CP in newCG.contact_pairs],
                                           [cls.trajs[ii].xyz[-5:] for _ in cls.CG.contact_pairs])



    def test_frames_dictionary(cls):
        newCG: contacts.ContactGroup = cls.CG.select_by_frames({1: [1, 2], 0: [4, 3]})
        assert (newCG.n_ctcs == cls.CG.n_ctcs)
        assert (newCG.n_trajs == cls.CG.n_trajs)
        assert (newCG.top is cls.CG.top)
        _np.testing.assert_equal(newCG.trajlabels, ["mdtraj.00", "mdtraj.01"])
        _np.testing.assert_equal(newCG.n_frames, [2, 2])

        # Blanket check for most array-like properties
        for iattr in cls.unchanged_propeties:
           _np.testing.assert_array_equal(getattr(newCG, iattr), getattr(cls.CG, iattr), iattr)

        # Tests of the time array
        _np.testing.assert_array_equal(newCG.time_arrays, [cls.CG.time_arrays[1][[1, 2]],
                                                           cls.CG.time_arrays[0][[4, 3]]]
                                       )

        # Test of the time dependent stuff
        # We can still iterate through CPs, but the inner implicit iteration has to go,
        # since we've changed the order of the trajs, so we have to write it out explicitly
        for ii in range(newCG.n_ctcs):
            _np.testing.assert_array_equal(newCG.contact_pairs[ii].time_traces.ctc_trajs[0],
                                           cls.CG.contact_pairs[ii].time_traces.ctc_trajs[1][[1, 2]])
            _np.testing.assert_array_equal(newCG.contact_pairs[ii].time_traces.ctc_trajs[1],
                                           cls.CG.contact_pairs[ii].time_traces.ctc_trajs[0][[4, 3]])

            _np.testing.assert_array_equal(newCG.contact_pairs[ii].time_traces.atom_pair_trajs[0],
                                           cls.CG.contact_pairs[ii].time_traces.atom_pair_trajs[1][[1, 2]])
            _np.testing.assert_array_equal(newCG.contact_pairs[ii].time_traces.atom_pair_trajs[1],
                                           cls.CG.contact_pairs[ii].time_traces.atom_pair_trajs[0][[4, 3]])

            _np.testing.assert_array_equal(newCG.contact_pairs[ii].time_traces.trajs[0].xyz,
                                           cls.trajs[1].xyz[[1, 2], :, :])
            _np.testing.assert_array_equal(newCG.contact_pairs[ii].time_traces.trajs[1].xyz,
                                           cls.trajs[0].xyz[[4, 3], :, :])


    def test_frames_list_of_pairs(cls):

        newCG: contacts.ContactGroup = cls.CG.select_by_frames([
            [1, 2],
            [0, 4],
            [1, 1],
            [0, 3],
            [1, 0]
        ])
        test_traj = md.Trajectory(xyz=_np.array([cls.trajs[1][2].xyz.squeeze(),
                                                 cls.trajs[0][4].xyz.squeeze(),
                                                 cls.trajs[1][1].xyz.squeeze(),
                                                 cls.trajs[0][3].xyz.squeeze(),
                                                 cls.trajs[1][0].xyz.squeeze()]), topology=cls.CG.top,
                                  time=_np.hstack([cls.trajs[1][2].time,
                                                   cls.trajs[0][4].time,
                                                   cls.trajs[1][1].time,
                                                   cls.trajs[0][3].time,
                                                   cls.trajs[1][0].time]),
                                  unitcell_lengths=_np.array([cls.trajs[1][2].unitcell_lengths.squeeze(),
                                                              cls.trajs[0][4].unitcell_lengths.squeeze(),
                                                              cls.trajs[1][1].unitcell_lengths.squeeze(),
                                                              cls.trajs[0][3].unitcell_lengths.squeeze(),
                                                              cls.trajs[1][0].unitcell_lengths.squeeze()]),
                                  unitcell_angles=_np.array([cls.trajs[1][2].unitcell_angles.squeeze(),
                                                             cls.trajs[0][4].unitcell_angles.squeeze(),
                                                             cls.trajs[1][1].unitcell_angles.squeeze(),
                                                             cls.trajs[0][3].unitcell_angles.squeeze(),
                                                             cls.trajs[1][0].unitcell_angles.squeeze()])
                                   )
        b = _io.StringIO()
        with _contextlib.redirect_stdout(b):
            ref_CG : contacts.ContactGroup = _mdcli.sites([{"name": "test", "pairs":{"residx":[[100,200], [100,300]]}}], test_traj,
                                   no_disk=True, figures=False)["test"]
        b.close()

        assert (newCG.n_ctcs == cls.CG.n_ctcs)
        assert (newCG.n_trajs == 1)
        assert (newCG.trajlabels[0]=="mdtraj.00")
        assert (newCG.top is cls.CG.top)

        # Blanket check for most array-like properties
        for iattr in cls.unchanged_propeties:
            if iattr in ["n_trajs", "trajlabels"]: #trajlabels is checked couple of lines up
                continue
            _np.testing.assert_array_equal(getattr(newCG, iattr), getattr(cls.CG, iattr), iattr)

        # Tests of the time array
        _np.testing.assert_array_equal(newCG.time_arrays, ref_CG.time_arrays)
        _np.testing.assert_array_equal(newCG.time_arrays, [test_traj.time])

        # Tests of the unitcell
        _np.testing.assert_array_equal(newCG.contact_pairs[0].time_traces.trajs[0].unitcell_lengths,
                                       test_traj.unitcell_lengths)
        _np.testing.assert_array_equal(newCG.contact_pairs[0].time_traces.trajs[0].unitcell_lengths,
                                       ref_CG.contact_pairs[0].time_traces.trajs[0].unitcell_lengths)

        _np.testing.assert_array_equal(newCG.contact_pairs[0].time_traces.trajs[0].unitcell_angles,
                                       test_traj.unitcell_angles)
        _np.testing.assert_array_equal(newCG.contact_pairs[0].time_traces.trajs[0].unitcell_angles,
                                       ref_CG.contact_pairs[0].time_traces.trajs[0].unitcell_angles)


        # Test of the time dependent stuff
        for ii in range(newCG.n_trajs):
            _np.testing.assert_array_equal([CP.time_traces.ctc_trajs[ii] for CP in newCG.contact_pairs],
                                           [CP.time_traces.ctc_trajs[ii] for CP in ref_CG.contact_pairs])
            _np.testing.assert_array_equal([CP.time_traces.atom_pair_trajs[ii] for CP in newCG.contact_pairs],
                                           [CP.time_traces.atom_pair_trajs[ii] for CP in ref_CG.contact_pairs])
            _np.testing.assert_array_equal([CP.time_traces.trajs[ii].xyz for CP in newCG.contact_pairs],
                                           [CP.time_traces.trajs[ii].xyz for CP in ref_CG.contact_pairs])

class TestContactGroupFrequencies(TestBaseClassContactGroup):

    @classmethod
    def setUpClass(cls):
        super(TestContactGroupFrequencies,cls).setUp(cls)
        cls.CG = contacts.ContactGroup(
            [cls.cp1_w_anchor_and_frags_and_top,
             cls.cp2_w_anchor_and_frags_and_top],
            neighbors_excluded=0
        )
        cls.GPCR = examples.GPCRLabeler_ardb2_human()
        cls.CGN = examples.CGNLabeler_gnas2_human()
        cls.intf = examples.examples.Interface_B2AR_Gas(GPCR_UniProt = cls.GPCR,
                                                         CGN_UniProt = cls.CGN)
        cls.total_intf_freq_at_3 = cls.intf.frequency_per_contact(3.0).sum()
        cls.L394 = examples.ContactGroupL394(GPCR_UniProt=None)
        assert cls.total_intf_freq_at_3 > 0
    def test_frequency_dicts(self):
        CG = self.CG
        freqdcit = CG.frequency_dicts(2, pad_label=False)
        self.assertDictEqual(freqdcit, {"E30@fragA-V31@fragB" : 2 / 5,
                                        "E30@fragA-W32@fragC" : 1 / 5})

    def test_frequency_dicts_sort(self):
        CG = contacts.ContactGroup(
            [self.cp2_w_anchor_and_frags_and_top,
             self.cp1_w_anchor_and_frags_and_top],
            neighbors_excluded=0
        )
        self.assertDictEqual(CG.frequency_dicts(2, pad_label=False, sort_by_freq=True),
                             {"E30@fragA-W32@fragC": 1 / 5,
                              "E30@fragA-V31@fragB": 2 / 5})

    def test_frequency_per_contact(self):
        CG = self.CG
        freqs = CG.frequency_per_contact(2)
        _np.testing.assert_array_equal([2 / 5, 1 / 5], freqs)

    def test_frequency_per_traj(self):
        CG = self.CG
        freqs = CG.frequency_per_traj(2)
        _np.testing.assert_array_equal(freqs[0],[2/3, 1/3])
        _np.testing.assert_array_equal(freqs[1],[0, 0])


    def test_frequency_per_residue_idx(self):
        CG = self.CG
        freq_dict = CG.frequency_sum_per_residue_idx_dict(2)
        self.assertDictEqual(freq_dict, {0: 2 / 5 + 1 / 5,
                                         1: 2 / 5,
                                         2: 1/ 5})

    #TODO This test is smh superflous with this CG but I wont write another one
    def test_frequency_per_residue_idx_reverse(self):
        CG = self.CG
        freq_dict = CG.frequency_sum_per_residue_idx_dict(2, sort_by_freq=False)
        self.assertDictEqual(freq_dict, {0: 2 / 5 + 1 / 5,
                                         1: 2 / 5,
                                         2: 1/ 5})

    def test_frequency_per_residue_idx_return_array(self):
        CG = self.CG
        freq_dict = CG.frequency_sum_per_residue_idx_dict(2)
        freq_array = CG.frequency_sum_per_residue_idx_dict(2, return_array=True)
        _np.testing.assert_array_equal(list(freq_dict.values()),freq_array[freq_array>0])


    def test_frequency_per_residue_name(self):
        CG = self.CG
        freq_dict = CG.frequency_sum_per_residue_names(2)[0]
        assert len(freq_dict) == 3
        _np.testing.assert_equal(freq_dict["E30@fragA"], 2 / 5 + 1 / 5)
        _np.testing.assert_equal(freq_dict["V31@fragB"], 2 / 5)
        _np.testing.assert_equal(freq_dict["W32@fragC"], 1 / 5)

    def test_frequency_per_residue_name_consensus(self):
        CG = self.L394
        """
        ['L394@G.H5.26, 
         'L388@G.H5.20',
         'R389@G.H5.21', 
         'L230@frag3',
         'R385@G.H5.17',
         'K270@frag3']
        """
        freq_dict = CG.frequency_sum_per_residue_names(4, AA_format="try_consensus")[0]
        assert len(freq_dict) == 6
        _np.testing.assert_equal(freq_dict["G.H5.26"], CG.select_by_residues("L394").frequency_per_contact(4).sum())
        _np.testing.assert_equal(freq_dict["G.H5.20"], CG.select_by_residues("L388").frequency_per_contact(4).sum())
        _np.testing.assert_equal(freq_dict["L230@frag3"], CG.select_by_residues("L230").frequency_per_contact(4).sum())
        _np.testing.assert_equal(freq_dict["G.H5.17"], CG.select_by_residues("R385").frequency_per_contact(4).sum())
        _np.testing.assert_equal(freq_dict["K270@frag3"], CG.select_by_residues("K270").frequency_per_contact(4).sum())



    def test_frequency_per_residue_name_no_sort(self):
        CG = self.CG
        freq_dict = CG.frequency_sum_per_residue_names(2, sort_by_freq=False)[0]
        assert len(freq_dict) == 3
        _np.testing.assert_equal(freq_dict["E30@fragA"], 2 / 5 + 1 / 5)
        _np.testing.assert_equal(freq_dict["V31@fragB"], 2 / 5)
        _np.testing.assert_equal(freq_dict["W32@fragC"], 1 / 5)

    def test_frequency_per_residue_name_dataframe(self):
        CG = self.CG
        freq_dict = CG.frequency_sum_per_residue_names(2,
                                                       return_as_dataframe=True)[0]
        assert len(freq_dict) == 3
        assert isinstance(freq_dict,_DF)
        _np.testing.assert_array_equal(freq_dict["label"].array, ["E30@fragA", "V31@fragB", "W32@fragC"])
        _np.testing.assert_array_equal(freq_dict["freq"].array, [2 / 5 + 1 / 5,
                                                                 2 / 5,
                                                                 1 / 5])

    def test_frequency_per_residue_name_consensus(self):
        CG = self.CG
        freq_dict = CG.frequency_sum_per_residue_names(2,
                                                       return_as_dataframe=True)[0]
        assert len(freq_dict) == 3

    def test_frequency_dict_by_consensus_labels_fails(self):
        CG = self.CG

        with self.assertRaises(AssertionError):
            CG.frequency_dict_by_consensus_labels(2)

    def test_frequency_dict_by_consensus_labels(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs,
                                    self.cp3_wtop_and_conslabs])

        freq_dict = CG.frequency_dict_by_consensus_labels(2)
        _np.testing.assert_equal(freq_dict["3.50"]["4.50"], 2 / 3)
        _np.testing.assert_equal(freq_dict["3.50"]["5.50"], 1 / 3)
        _np.testing.assert_equal(freq_dict["4.50"]["5.50"], 1 / 3)
        assert len(freq_dict) == 2

    def test_frequency_dict_by_consensus_labels_include_trilow(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs,
                                    self.cp3_wtop_and_conslabs])

        freq_dict = CG.frequency_dict_by_consensus_labels(2, include_trilower=True)
        _np.testing.assert_equal(freq_dict["3.50"]["4.50"], 2 / 3)
        _np.testing.assert_equal(freq_dict["3.50"]["5.50"], 1 / 3)
        _np.testing.assert_equal(freq_dict["4.50"]["5.50"], 1 / 3)

        _np.testing.assert_equal(freq_dict["4.50"]["3.50"], freq_dict["4.50"]["3.50"])
        _np.testing.assert_equal(freq_dict["5.50"]["3.50"], freq_dict["5.50"]["3.50"])
        _np.testing.assert_equal(freq_dict["5.50"]["4.50"], freq_dict["5.50"]["4.50"])
        assert len(freq_dict) == 3

    def test_frequency_dict_by_consensus_labels_return_triplets(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs,
                                    self.cp3_wtop_and_conslabs])

        freq_dict = CG.frequency_dict_by_consensus_labels(2, return_as_triplets=True)
        _np.testing.assert_array_equal(freq_dict[0], _np.array([["3.50", "4.50", 2 / 3]]).squeeze())
        _np.testing.assert_array_equal(freq_dict[1], _np.array([["3.50", "5.50", 1 / 3]]).squeeze())
        _np.testing.assert_array_equal(freq_dict[2], _np.array([["4.50", "5.50", 1 / 3]]).squeeze())

        assert len(freq_dict) == 3

    def test_frequency_dict_by_consensus_labels_interface_raises(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs,
                                    self.cp3_wtop_and_conslabs])
        with self.assertRaises(NotImplementedError):
            CG.frequency_dict_by_consensus_labels(2, sort_by_interface=True)

    def test_frequency_as_contact_matrix(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs,
                                    ])

        mat = CG.frequency_as_contact_matrix(2)
        mat_ref = _np.zeros((CG.top.n_residues, CG.top.n_residues))
        mat_ref[:, :] = _np.nan
        mat_ref[0, 1] = mat_ref[1, 0] = 2 / 3
        mat_ref[0, 2] = mat_ref[2, 0] = 1 / 3

        _np.testing.assert_array_equal(mat, mat_ref)

    def test_frequency_as_contact_matrix_CG(self):
        fragments = get_fragments(self.intf.top, method="resSeq+")
        mat, frags = self.intf.frequency_as_contact_matrix_CG(3.0, fragments=fragments, return_fragments=True)
        mat_ref = _np.zeros((4, 4))
        mat_ref[0,3] = mat_ref[3,0] = self.total_intf_freq_at_3
        _np.testing.assert_array_equal(mat_ref.round(3), mat)
        assert len(fragments)==len(frags)
        for ifrag, jfrag in zip(fragments, frags.values()):
            _np.testing.assert_array_equal(ifrag, jfrag)

    def test_frequency_as_contact_matrix_CG_consensus_labelers(self):
        fragments = get_fragments(self.intf.top, method="resSeq+")
        mat = self.intf.frequency_as_contact_matrix_CG(3.0, fragments=fragments,
                                                       consensus_labelers=[self.GPCR,
                                                                           self.CGN])
        assert isinstance(mat, _DF)
        _np.testing.assert_almost_equal(mat.values.sum(),
                                        self.total_intf_freq_at_3*2,
                                        decimal=2)


    def test_frequency_as_contact_matrix_CG_sparse(self):
        fragments = get_fragments(self.intf.top, method="resSeq+")
        mat = self.intf.frequency_as_contact_matrix_CG(3.0, fragments=fragments, sparse=True)
        mat_ref = _np.zeros((2,2))
        mat_ref[0,1] = mat_ref[1,0] = self.total_intf_freq_at_3
        _np.testing.assert_equal(mat_ref.round(3), mat)

    def test_frequency_as_contact_matrix_CG_interface(self):
        fragments = get_fragments(self.intf.top, method="resSeq+")
        mat = self.intf.frequency_as_contact_matrix_CG(3.0,
                                                       interface=True,
                                                       fragments=fragments)
        assert isinstance(mat, _DF)
        self.assertListEqual(list(mat.index), ["frag 0"])
        self.assertListEqual(list(mat.keys()),["frag 3"])
        _np.testing.assert_array_equal(self.total_intf_freq_at_3.round(3), mat.values[0,0

        ])

        # Test with a different set of fragments
        fragments = get_fragments(self.intf.top, method="resSeq")
        mat = self.intf.frequency_as_contact_matrix_CG(3.0,
                                                       interface=True,
                                                       fragments=fragments)
        assert isinstance(mat, _DF)
        self.assertListEqual(list(mat.index), ["frag 0", "frag 1", "frag 2", "frag 3"]) #these are frag 0
        self.assertListEqual(list(mat.keys()), ["frag 6", "frag 7", "frag 8"]) # these are frag 1

    def test_frequency_to_bfactor_just_runs(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs,
                                    ],
                                   interface_fragments = [[0],[1,2]])
        with _TDir() as tmpdir:
            pdb=path.join(tmpdir,"as_betas.pdb")
            betas = CG.frequency_to_bfactor(3.5,pdb, self.geom, interface_sign=True)
        assert len(betas) == self.geom.n_atoms

    def test_interface_frequency_matrix(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp3_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs,
                                   self.cp5_wtop_and_wo_conslabs,
                                   ],
                                  interface_fragments = [[0, 3, 4], [1, 2, 20]])
        refmat = _np.zeros((3, 2))
        refmat[:, :] = _np.nan
        refmat[0, 0] = 2 / 3  # [0,1]
        refmat[0, 1] = 1 / 3  # [0,2]
        refmat[1, 1] = 0  # [3,2]
        _np.testing.assert_array_equal(refmat, I.interface_frequency_matrix(2))

    def test_frequencies_of_atom_pairs_formed(self):
        CG = contacts.ContactGroup([self.cp1_w_atom_types, self.cp2_w_atom_types])
        list_of_dicts = CG.relative_frequency_formed_atom_pairs_overall_trajs(4)
        _np.testing.assert_equal(list_of_dicts[0]["BB-SC"], 1 / 3)
        _np.testing.assert_equal(list_of_dicts[0]["BB-BB"], 2 / 3)
        _np.testing.assert_equal(list_of_dicts[1]["BB-SC"], 2 / 3)
        _np.testing.assert_equal(list_of_dicts[1]["SC-BB"], 1 / 3)

    def test_frequency_table(self):
        CG = contacts.ContactGroup([contacts.ContactPair([0, 1], [[.4, .3, .25]], [[0, 1, 2]]),
                                    contacts.ContactPair([0, 2], [[.1, .2, .3]], [[0, 1, 2]])])

        table = CG.frequency_dataframe(2.5, pad_label=False)
        _np.testing.assert_array_equal(table["freq"].array, [1 / 3, 2 / 3])
        _np.testing.assert_array_equal(table["label"].array, ["0-1", "0-2"])
        _np.testing.assert_array_equal(table["sum"].array, [1 / 3, 1 / 3 + 2 / 3])

    def test_frequency_table_w_atom_types_and_names(self):
        CG = contacts.ContactGroup([self.cp1_w_atom_types, self.cp2_w_atom_types])

        table = CG.frequency_dataframe(3.5, pad_label=False, atom_types=True)
        _np.testing.assert_array_equal(table["freq"].array, [3 / 4, 3 / 4])
        _np.testing.assert_array_equal(table["label"].array, ["E30-V31", "E30-W32"])
        _np.testing.assert_array_equal(table["sum"].array, [3 / 4, 3 / 4 + 3 / 4])
        _np.testing.assert_equal(table["by_atomtypes"][0], " 66% BB-BB,  33% BB-SC")
        _np.testing.assert_equal(table["by_atomtypes"][1], " 66% BB-SC,  33% SC-BB")

    def test_frequency_delta_low_level(self):
        from mdciao.contacts.contacts import _delta_freq_pairs
        delta, pairs = _delta_freq_pairs([1., .60, 1], [[0, 1],
                                                      [2, 3],
                                                      [0, 2]],
                                         [.25, .50, 1], [[0, 1],
                                                         [2, 0],
                                                         [4, 2]])
        assert len(delta)==len(pairs)==4
        delta = {tuple(pp):dd for dd, pp in zip(delta,pairs)}
        assert delta[tuple([0,1])]==.25-1
        assert delta[tuple([0,2])]==.50-1
        assert delta[tuple([2,3])]==-.60
        assert delta[tuple([2,4])]==1

    def test_frequency_delta(self):
        CG1 = examples.ContactGroupL394(ctc_cutoff_Ang=3.5)
        CG2 = examples.ContactGroupL394(ctc_cutoff_Ang=5)
        CG1 : contacts.ContactGroup
        from mdciao.contacts.contacts import _delta_freq_pairs
        delta_ref, pairs_ref = _delta_freq_pairs(CG1.frequency_per_contact(3.5), CG1.res_idxs_pairs,
                                         CG2.frequency_per_contact(3.5), CG2.res_idxs_pairs)
        delta, pairs = CG1.frequency_delta(CG2, 3.5)
        _np.testing.assert_array_equal(delta_ref, delta)
        _np.testing.assert_array_equal(pairs_ref, pairs)

    def test_frequency_dataframe_just_runs(self):
        CG = contacts.ContactGroup([self.cp1_w_atom_types, self.cp2_w_atom_types])
        df = CG.frequency_dataframe(3.5, sort_by_freq=True, atom_types=True)
        assert isinstance(df, _DF)

class TestContactGroupFrequencies_max_cutoff(TestBaseClassContactGroup):

    @classmethod
    def setUpClass(cls):
        super(TestContactGroupFrequencies_max_cutoff,cls).setUp(cls)
        cls.CG = contacts.ContactGroup(
            [cls.cp1_w_anchor_and_frags_and_top,
             cls.cp2_w_anchor_and_frags_and_top],
            neighbors_excluded=0,
            max_cutoff_Ang=3
        )

    def test_cutoff(self):
        self.CG._check_cutoff_ok(3) # passes
        with self.assertRaises(ValueError) as cm:
            self.CG._check_cutoff_ok(6) #doesn't pass

    def test_frequency_dicts(self):
        with self.assertRaises(ValueError) as cm:
            self.CG.frequency_dicts(6, pad_label=False)

    def test_frequency_per_contact(self):
        with self.assertRaises(ValueError) as cm:
            self.CG.frequency_per_contact(6)

    def test_frequency_per_residue_idx(self):
        with self.assertRaises(ValueError) as cm:
            self.CG.frequency_sum_per_residue_idx_dict(6)

    def test_frequency_per_residue_name(self):
        with self.assertRaises(ValueError) as cm:
            self.CG.frequency_sum_per_residue_names(6)

    def test_frequency_dict_by_consensus_labels(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs,
                                    self.cp3_wtop_and_conslabs],
                                   max_cutoff_Ang=3)
        with self.assertRaises(ValueError) as cm:
            CG.frequency_dict_by_consensus_labels(6)

    def test_frequency_as_contact_matrix(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs,
                                    ],
                                   max_cutoff_Ang=3)
        with self.assertRaises(ValueError):
            CG.frequency_as_contact_matrix(6)

    def test_frequency_to_bfactor_just_runs(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                    self.cp2_wtop_and_conslabs,
                                    ],
                                   interface_fragments = [[0],[1,2]],
                                   max_cutoff_Ang=3)
        with self.assertRaises(ValueError):
            CG.frequency_to_bfactor(6,None, self.geom, interface_sign=True)

    def test_interface_frequency_matrix(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp3_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs,
                                   self.cp5_wtop_and_wo_conslabs,
                                   ],
                                  interface_fragments = [[0, 3, 4], [1, 2, 20]],
                                  max_cutoff_Ang=3)
        with self.assertRaises(ValueError):
            I.interface_frequency_matrix(6)

    def test_frequencies_of_atom_pairs_formed(self):
        CG = contacts.ContactGroup([self.cp1_w_atom_types, self.cp2_w_atom_types],
        max_cutoff_Ang=3,
                                   )
        with self.assertRaises(ValueError):
            CG.relative_frequency_formed_atom_pairs_overall_trajs(6)

    def test_frequency_table(self):
        CG = contacts.ContactGroup([contacts.ContactPair([0, 1], [[.4, .3, .25]], [[0, 1, 2]]),
                                    contacts.ContactPair([0, 2], [[.1, .2, .3]], [[0, 1, 2]])],
                                   max_cutoff_Ang=3)
        with self.assertRaises(ValueError):
            CG.frequency_dataframe(6, pad_label=False)

class TestContactGroupPlots(TestBaseClassContactGroup):

    @classmethod
    def setUpClass(cls):
        super(TestContactGroupPlots,cls).setUp(cls)
        cls.CG_cp1_cp2 = contacts.ContactGroup([cls.cp1, cls.cp2])
        cls.CG_cp1_cp2_both_w_anchor_and_frags = contacts.ContactGroup(
            [cls.cp1_w_anchor_and_frags,
             cls.cp2_w_anchor_and_frags],
            neighbors_excluded=0
        )

    def test_plot_freqs_as_bars_just_runs(self):
        CG = self.CG_cp1_cp2
        jax = CG.plot_freqs_as_bars(2, "test_site", cumsum=True)
        assert isinstance(jax, _plt.Axes)
        _plt.close("all")

    def test_plot_freqs_as_bars_just_runs_labels_short(self):
        CG = self.CG_cp1_cp2
        jax = CG.plot_freqs_as_bars(2, "test_site", shorten_AAs=True)
        assert isinstance(jax, _plt.Axes)
        _plt.close("all")

    def test_plot_freqs_as_bars_just_runs_labels_xlim(self):
        CG = self.CG_cp1_cp2
        jax = CG.plot_freqs_as_bars(2, "test_site", xlim=20)
        assert isinstance(jax, _plt.Axes)
        _plt.close("all")

    def test_plot_freqs_as_bars_display_sort(self):
        CG = self.CG_cp1_cp2
        CG.plot_freqs_as_bars(2, "test_site", sort_by_freq=True)


    def test_plot_freqs_as_bars_total_freq(self):
        CG = self.CG_cp1_cp2
        CG.plot_freqs_as_bars(2, "test_site", total_freq=CG.frequency_per_contact(2).sum())

    def test_plot_freqs_as_bars_no_neighborhood(self):
        CG = contacts.ContactGroup([self.cp1, self.cp2],name="test site")
        jax = CG.plot_freqs_as_bars(2)
        assert isinstance(jax, _plt.Axes)
        _plt.close("all")

    def test_plot_freqs_as_bars_no_neighborhood_fails(self):
        with self.assertRaises(AssertionError):
            CG = self.CG_cp1_cp2
            CG.plot_freqs_as_bars(2,)

    def test_plot_add_hatching_by_atomtypes(self):
        CG = contacts.ContactGroup([self.cp1_w_atom_types,
                                    self.cp1_w_atom_types_0_1_switched,
                                    self.cp2_w_atom_types])
        # Minimal plot
        iax = CG.plot_freqs_as_bars(3.5, title_label="test", plot_atomtypes=True)
        _plt.close("all")

    def test_plot_violins_no_neighborhood(self):
        CG = self.CG_cp1_cp2
        iax, order = CG.plot_violins(ctc_cutoff_Ang=3.5, title_label="CG_cp1_cp2",
                                     defrag="@",
                                     truncate_at_mean=5
                                     )
        assert isinstance(iax, _plt.Axes)
        assert isinstance(order, _np.ndarray)
        # ax.figure.savefig("test.png")
        _plt.close("all")

    def test_plot_violins_options1(self):
        CG = examples.ContactGroupL394()
        iax, order = CG.plot_violins(sort_by=True,
                                     shorten_AAs=True,
                                     ctc_cutoff_Ang=3.5,
                                     # xlim=2
                                     )
        assert isinstance(iax, _plt.Axes)
        assert isinstance(order, _np.ndarray)
        # ax.figure.savefig("test.png")
        _plt.close("all")

    def test_plot_violins_options2(self):
        CG = examples.ContactGroupL394()
        iax, order = CG.plot_violins(sort_by=True,
                                     shorten_AAs=True,
                                     truncate_at_mean=3.7
                                     )
        # ax.figure.savefig("test.png")
        assert isinstance(iax, _plt.Axes)
        assert isinstance(order, _np.ndarray)
        _plt.close("all")

    def test_plot_violins_options3(self):
        CG = examples.ContactGroupL394()
        iax, order = CG.plot_violins(sort_by=[0, 4],
                                     shorten_AAs=True,
                                     truncate_at_mean=3.7
                                     )
        # ax.figure.savefig("test.png")
        assert isinstance(iax, _plt.Axes)
        assert isinstance(order, _np.ndarray)
        _np.testing.assert_array_equal(order, [0, 4])
        _plt.close("all")

    def test_plot_violins_options4(self):
        CG = examples.ContactGroupL394()
        iax, order = CG.plot_violins(sort_by=2,
                                     shorten_AAs=True,
                                     stride=2,
                                     )
        #ax.figure.savefig("test.png")
        assert isinstance(iax, _plt.Axes)
        assert isinstance(order, _np.ndarray)
        _np.testing.assert_array_equal(order, [1, 0])
        _np.testing.assert_array_equal([1, 0],CG.means.argsort()[:2])
        #_plt.savefig("test.png")
        _plt.close("all")

    def test_plot_violins_raises_on_title(self):
        CG = self.CG_cp1_cp2
        with self.assertRaises(AssertionError):
            iax = CG.plot_violins(3.5)


    def test_plot_freqs_as_flareplot_just_runs(self):
        # This is just to test that it runs without error
        # the minimal examples here cannot test the full flareplot
        # TODO add full-fledged example here?
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,self.cp2_wtop_and_conslabs, self.cp3_wtop_and_conslabs])
        ifig, iax, flareplot_attrs = CG.plot_freqs_as_flareplot(10,)

    @unittest.skipIf(_sys.version.startswith("3.7") and _platform.system().lower()=="darwin", "Random segfaults when using md.compute_dssp on Python 3.7 on MacOs. See https://github.com/mdtraj/mdtraj/issues/1574")
    def test_plot_freqs_as_flareplot_just_runs_w_options(self):
        # This is just to test that it runs without error
        # the minimal examples here cannot test the full flareplot
        # TODO add full-fledged example here?
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,self.cp2_wtop_and_conslabs, self.cp3_wtop_and_conslabs],
                                   top=self.top)
        ifig, iax, flareplot_attrs = CG.plot_freqs_as_flareplot(10, SS=self.geom)
        ifig.tight_layout()
        _plt.close("all")
        #ifig.savefig("test.pdf")

    def test_plot_freqs_as_flareplot_just_runs_w_consensus_maps(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,self.cp2_wtop_and_conslabs, self.cp3_wtop_and_conslabs],
                                   top=self.top)
        ifig, iax, flareplot_attrs = CG.plot_freqs_as_flareplot(10, SS=self.geom,
                                                                consensus_maps=[["GPH"] * self.top.n_residues])
        ifig.tight_layout()
        _plt.close("all")

    def test_plot_freqs_as_flareplot_just_runs_w_SS_array(self):
        CG = contacts.ContactGroup([self.cp1_wtop_and_conslabs,self.cp2_wtop_and_conslabs, self.cp3_wtop_and_conslabs],
                                   top=self.top)
        ifig, iax, flareplot_attrs = CG.plot_freqs_as_flareplot(10,
                                                                SS=_np.array(["H"] * self.top.n_residues))
        ifig.tight_layout()
        _plt.close("all")


    def test_plot_neighborhood_raises(self):
        CG = self.CG_cp1_cp2
        with self.assertRaises(AssertionError):
            CG.plot_neighborhood_freqs(2)

    def test_plot_neighborhood_works_minimal(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags
        jax = CG.plot_neighborhood_freqs(2)
        assert isinstance(jax, _plt.Axes)
        _plt.close("all")

    def test_plot_neighborhood_works_options(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags
        jax = CG.plot_neighborhood_freqs(2, shorten_AAs=True)
        assert isinstance(jax, _plt.Axes)
        jax = CG.plot_neighborhood_freqs(2,xmax=10)
        assert isinstance(jax, _plt.Axes)
        _plt.plot()
        jax = _plt.gca()
        assert jax is CG.plot_neighborhood_freqs(2, ax=jax)
        _plt.close("all")

    def test_plot_get_hatches_for_plotting_atomtypes(self):
        CG = contacts.ContactGroup([self.cp1_w_atom_types, self.cp1_w_atom_types_0_1_switched])
        df = CG._get_hatches_for_plotting(3.5)
        # This checks that the inversion ["BB-SC"]-["SC-BB"] takes place when needed
        _np.testing.assert_array_equal(df.values[0, :], [2 / 3, 0, 1 / 3, 0, 0, 0, 0, 0])
        _np.testing.assert_array_equal(df.values[1, :], [2 / 3, 0, 1 / 3, 0, 0, 0, 0, 0])


    def test_plot_timedep_ctcs(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags
        figs = CG.plot_timedep_ctcs()
        _np.testing.assert_equal(len(figs), 1)
        assert isinstance(figs[0], _plt.Figure)
        ifig = figs[0]
        #ifig.savefig("test.png")
        _np.testing.assert_equal(len(ifig.axes), 2 + 2 )  # 2 + 2 x twiny, no N_ctcs bc there's no cutoff
        _plt.close("all")

    def test_plot_timedep_ctcs_sort_by_freq(self):
        # We can test the correct order using the labels, which get genarated internally and
        # are uniquelly associated with the ContactPair

        # reverse the ContactPairs s.t. the frequences are in ascending (unusual) order
        CG =  contacts.ContactGroup(self.CG_cp1_cp2_both_w_anchor_and_frags.contact_pairs[::-1],
                                    neighbors_excluded=0)

        figs = CG.plot_timedep_ctcs(ctc_cutoff_Ang=2)
        # text labels look like this, we can grab the freqs from there
        # '0$^{\\mathrm{fragA}}$-2$^{\\mathrm{fragC}}$ (20%)',
        # '0$^{\\mathrm{fragA}}$-1$^{\\mathrm{fragB}}$ (40%)'
        ifig = figs[0]
        _np.testing.assert_equal(len(ifig.axes), 2 + 2 + 1 + 1)  # 2 + 2 x twiny + N_ctcs + N_ctcs_twiny

        freqs = []
        for ax in ifig.axes:
            for txt in ax.texts:
                freqs.append(txt.get_text().split()[-1].strip("()%"))
        self.assertListEqual(freqs,['20','40']) #<- the freqs didn't get sorted

        figs = CG.plot_timedep_ctcs(ctc_cutoff_Ang=2, sort_by_freq=True)
        ifig = figs[0]
        freqs = []
        for ax in ifig.axes:
            for txt in ax.texts:
                freqs.append(txt.get_text().split()[-1].strip("()%"))
        self.assertListEqual(freqs, ['40', '20'])  # <- the freqs got sorted in descending order

        #ifig.savefig("test.png")
        _np.testing.assert_equal(len(ifig.axes), 2 + 2 + 1 + 1 )  # 2 + 2 x twiny + N_ctcs + N_ctcs_twiny
        _plt.close("all")

    def test_plot_timedep_ctcs_with_valid_cutoff_no_pop(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags
        figs = CG.plot_timedep_ctcs(ctc_cutoff_Ang=1)
        ifig = figs[0]
        _np.testing.assert_equal(len(figs), 1)
        #ifig.savefig("test.png")
        _np.testing.assert_equal(len(ifig.axes), 2 + 2 + 1 + 1)  # 2 + 2 x twiny + N_ctcs + N_ctcs_twiny
        _plt.close("all")

    def test_plot_timedep_ctcs_with_valid_cutoff_w_pop(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags
        figs = CG.plot_timedep_ctcs(pop_N_ctcs=True, ctc_cutoff_Ang=1)
        _np.testing.assert_equal(len(figs), 2)
        _np.testing.assert_equal(len(figs[0].axes), 2 + 2)  # 2 + 2 x twiny
        _np.testing.assert_equal(len(figs[1].axes), 1 + 1) # N_ctcs + N_ctcs_twiny
        #figs[1].savefig("test.png")
        _plt.close("all")

    def test_plot_timedep_ctcs_skip_empty(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags
        figs = CG.plot_timedep_ctcs(skip_timedep=True)
        _np.testing.assert_equal(len(figs), 0)
        _plt.close("all")

    def test_plot_timedep_ctcs_skip_w_valid_cutoff(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags
        figs = CG.plot_timedep_ctcs(skip_timedep=True, ctc_cutoff_Ang=1)
        _np.testing.assert_equal(len(figs), 1)
        _np.testing.assert_equal(len(figs[0].axes), 1 + 1)  # N_ctcs + N_ctcs_twiny
        #figs[0].savefig("test.png")
        _plt.close("all")

    def test_plot_timedep_ctcs_matrix(self):
        traj1 = md.load(test_filenames.traj_xtc, top=test_filenames.top_pdb)
        traj2 = traj1[:]
        traj2.time += 100
        r = _mdcli.residue_neighborhoods("L394", [traj1, traj1[:5], traj2],
                                         ctc_control=1.0, no_disk=True,
                                         figures=False)
        r: contacts.ContactGroup = r[353]
        fig, plotted_freqs, plotted_trajs = r.plot_timedep_ctcs_matrix(3,
                                                                       dt=1e-3, t_unit="ns",
                                                                       )
        freqs_by_freq = r.frequency_dicts(3.0, sort_by_freq=True, pad_label=False)
        self.assertDictEqual(plotted_freqs, freqs_by_freq)
        self.assertEqual(len(plotted_trajs),r.n_trajs)
        for ii, (traj, iax) in enumerate(zip(plotted_trajs, fig.axes)):
            img_array = list(iax.get_images())[0].get_array()
            self.assertEqual(img_array.shape[0], len(plotted_freqs))
            self.assertEqual(img_array.shape[1],  r.n_frames[ii])
            _np.testing.assert_array_equal(img_array, traj)
        self.assertEqual(ii,r.n_trajs-1)
        #fig.savefig("test.png")
        _plt.close("all")

    def test_plot_timedep_ctcs_matrix_anchor_1_traj_ctc_control_2(self):
        traj1 = md.load(test_filenames.traj_xtc, top=test_filenames.top_pdb)
        r = _mdcli.residue_neighborhoods("L394", traj1,
                                         ctc_control=1.0, no_disk=True,
                                         figures=False)
        r: contacts.ContactGroup = r[353]
        fig, plotted_freqs, plotted_trajs = r.plot_timedep_ctcs_matrix(3,
                                                                       dt=1e-3, t_unit="ns",
                                                                       anchor="LEU394",
                                                                       shorten_AAs=False,
                                                                       defrag=None,
                                                                       ctc_control=2,
                                                                       )
        freqs_by_freq = r.frequency_dicts(3.0, sort_by_freq=True, pad_label=False, defrag=None, AA_format="long")
        freqs_by_freq = _mdcu.str_and_dict.delete_exp_in_keys(freqs_by_freq, "LEU394")[0]
        freqs_by_freq = {key : val for ii, (key,val) in enumerate(freqs_by_freq.items()) if ii<2}
        self.assertDictEqual(plotted_freqs, freqs_by_freq)
        self.assertEqual(len(plotted_trajs),r.n_trajs)
        for ii, (traj, iax) in enumerate(zip(plotted_trajs, fig.axes)):
            img_array = list(iax.get_images())[0].get_array()
            self.assertEqual(img_array.shape[0], len(plotted_freqs))
            self.assertEqual(img_array.shape[1],  r.n_frames[ii])
            _np.testing.assert_array_equal(img_array, traj)
        self.assertEqual(ii,r.n_trajs-1)
        #fig.savefig("test.png")
        _plt.close("all")

    def test_plot_distance_distributions_just_works(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags
        jax = CG.plot_distance_distributions()
        assert isinstance(jax, _plt.Axes)
        _plt.close("all")

    def test_plot_plot_distance_distributions_just_works_w_options(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags
        jax = CG.plot_distance_distributions(xlim=[-1, 5], shorten_AAs=True, ctc_cutoff_Ang=3)
        assert isinstance(jax, _plt.Axes)
        _plt.close("all")

    def test_plot_plot_distance_distributions_no_neighborhood(self):
            CG = self.CG_cp1_cp2
            jax = CG.plot_distance_distributions(xlim=[-1, 5], shorten_AAs=True, ctc_cutoff_Ang=3)
            assert isinstance(jax, _plt.Axes)
            _plt.close("all")

    def test_plot_plot_distance_distributions_no_neighborhood_defrag(self):
            CG = self.CG_cp1_cp2
            jax = CG.plot_distance_distributions(xlim=[-1, 5], shorten_AAs=True, ctc_cutoff_Ang=3, defrag="@")
            assert isinstance(jax, _plt.Axes)
            _plt.close("all")

    def test_plot_frequency_sums_as_bars_just_works(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags
        jax = CG.plot_frequency_sums_as_bars(2.0, "test", xmax=4)
        assert isinstance(jax, _plt.Axes)
        _plt.close("all")

    def test_plot_interface_frequency_matrix(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   # self.cp3_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs,
                                   self.cp5_wtop_and_wo_conslabs],
                                  interface_fragments = [[0, 3, 5], [1, 2, 4]])
        print(I.frequency_dataframe(2))
        print(I.frequency_sum_per_residue_names(2))
        print(I.interface_labels_consensus)
        ifig, iax = I.plot_interface_frequency_matrix(2,
                                                      label_type="best")
        # ifig.tight_layout()
        # ifig.savefig("test.png", bbox_inches="tight")
        _plt.close("all")

    def test_plot_interface_frequency_matrix_other_labels(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   # self.cp3_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs,
                                   self.cp5_wtop_and_wo_conslabs],
                                  interface_fragments = [[0, 3, 5], [1, 2, 4]])
        print(I.frequency_dataframe(2))
        print(I.frequency_sum_per_residue_names(2))
        print(I.interface_labels_consensus)
        ifig, iax = I.plot_interface_frequency_matrix(2,
                                                      label_type="consensus")
        ifig, iax = I.plot_interface_frequency_matrix(2,
                                                      label_type="residue")
        with self.assertRaises(ValueError):
            I.plot_interface_frequency_matrix(2,
                                              label_type="blergh")
        _plt.close("all")

class TestContactGroupTable(TestBaseClassContactGroup):

    def test_excel(self):
        CG = contacts.ContactGroup([self.cp1_w_anchor_and_frags_and_top,
                                    self.cp2_w_anchor_and_frags_and_top],
                                   neighbors_excluded=0)
        with _TDir(suffix='_test_mdciao') as tmpdir:
            CG.frequency_table(2.5, path.join(tmpdir, "test.xlsx"))

    def test_dat(self):
        CG = contacts.ContactGroup([self.cp1_w_anchor_and_frags_and_top,
                                    self.cp2_w_anchor_and_frags_and_top],
                                   neighbors_excluded=0)
        with _TDir(suffix='_test_mdciao') as tmpdir:
            CG.frequency_table(2.5, path.join(tmpdir, "test.dat"))


class TestContactGroupSpreadsheet(TestBaseClassContactGroup):

    def test_frequency_spreadsheet_just_works(self):
        CG = contacts.ContactGroup([self.cp1_w_atom_types,
                                    self.cp2_w_atom_types],
                                   interface_fragments = [[0], [1, 2]]
                                   )
        with _TDir(suffix='_test_mdciao') as tmpdir:
            CG.frequency_table(2.5, path.join(tmpdir, "test.xlsx"),
                               atom_types=True)


class TestContactGroupASCII(TestBaseClassContactGroup):
    def test_frequency_str_ASCII_file_str(self):
        CG = contacts.ContactGroup([self.cp1_w_anchor_and_frags_and_top,
                                    self.cp2_w_anchor_and_frags_and_top],
                                   neighbors_excluded=0)

        istr = CG.frequency_table(2.5,None)
        self.assertEqual(istr[0], "#")
        self.assertIsInstance(istr, str)

    def test_frequency_str_ASCII_file(self):
        CG = contacts.ContactGroup([self.cp1_w_anchor_and_frags_and_top,
                                    self.cp2_w_anchor_and_frags_and_top],
                                   neighbors_excluded=0)

        with _TDir() as tmpdir:
            tfile = path.join(tmpdir,'freqfile.dat')
            CG.frequency_table(2.5, tfile, atom_types=False)
            from mdciao.utils.str_and_dict import freq_file2dict
            newfreq = freq_file2dict(tfile)
            _np.testing.assert_array_equal(list(newfreq.values()),CG.frequency_per_contact(2.5))

class TestContactGroupTrajdicts(TestBaseClassContactGroup):

    @classmethod
    def setUpClass(cls):
        super(TestContactGroupTrajdicts,cls).setUp(cls)
        cls.CG = contacts.ContactGroup(
            [cls.cp1_w_anchor_and_frags_and_top,
             cls.cp2_w_anchor_and_frags_and_top],
            neighbors_excluded=0
        )

    def test_to_per_traj_dicts_for_saving(self):
        CG = self.CG
        list_of_dicts = CG._to_per_traj_dicts_for_saving()

        data_traj_0 = list_of_dicts[0]["data"]
        _np.testing.assert_array_equal(data_traj_0[:, 0], self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[0])
        _np.testing.assert_array_equal(data_traj_0[:, 1],
                                       self.cp1_w_anchor_and_frags_and_top.time_traces.ctc_trajs[0] * 10)
        _np.testing.assert_array_equal(data_traj_0[:, 2],
                                       self.cp2_w_anchor_and_frags_and_top.time_traces.ctc_trajs[0] * 10)

        data_traj_1 = list_of_dicts[1]["data"]
        _np.testing.assert_array_equal(data_traj_1[:, 0], self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[1])
        _np.testing.assert_array_equal(data_traj_1[:, 1],
                                       self.cp1_w_anchor_and_frags_and_top.time_traces.ctc_trajs[1] * 10)
        _np.testing.assert_array_equal(data_traj_1[:, 2],
                                       self.cp2_w_anchor_and_frags_and_top.time_traces.ctc_trajs[1] * 10)

    def test_to_per_traj_dicts_for_saving_with_tunits_ns(self):
        CG = self.CG
        list_of_dicts = CG._to_per_traj_dicts_for_saving(t_unit="ns")

        data_traj_0 = list_of_dicts[0]["data"]
        _np.testing.assert_array_equal(data_traj_0[:, 0],
                                       self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[0] * 1e-3)

        data_traj_1 = list_of_dicts[1]["data"]
        _np.testing.assert_array_equal(data_traj_1[:, 0],
                                       self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[1] * 1e-3)

    def test_to_per_traj_dicts_for_saving_with_tunits_mus(self):
        CG = self.CG
        list_of_dicts = CG._to_per_traj_dicts_for_saving(t_unit="mus")

        data_traj_0 = list_of_dicts[0]["data"]
        _np.testing.assert_array_equal(data_traj_0[:, 0],
                                       self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[0] * 1e-6)

        data_traj_1 = list_of_dicts[1]["data"]
        _np.testing.assert_array_equal(data_traj_1[:, 0],
                                       self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[1] * 1e-6)

    def test_to_per_traj_dicts_for_saving_with_tunits_ms(self):
        CG = self.CG
        list_of_dicts = CG._to_per_traj_dicts_for_saving(t_unit="ms")

        data_traj_0 = list_of_dicts[0]["data"]
        _np.testing.assert_array_equal(data_traj_0[:, 0],
                                       self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[0] * 1e-9)

        data_traj_1 = list_of_dicts[1]["data"]
        _np.testing.assert_array_equal(data_traj_1[:, 0],
                                       self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[1] * 1e-9)

    def test_to_per_traj_dicts_for_saving_with_tunits_raises(self):
        CG = self.CG
        with self.assertRaises(ValueError):
            list_of_dicts = CG._to_per_traj_dicts_for_saving(t_unit="fs")

class TestContactGroupBintrajdicts(TestBaseClassContactGroup):

    @classmethod
    def setUpClass(cls):
        super(TestContactGroupBintrajdicts,cls).setUp(cls)
        cls.CG = contacts.ContactGroup(
            [cls.cp1_w_anchor_and_frags_and_top,
             cls.cp2_w_anchor_and_frags_and_top],
            neighbors_excluded=0
        )

    def test_to_per_traj_dicts_for_saving_bintrajs(self):
        CG = self.CG
        list_of_dicts = CG._to_per_traj_dicts_for_saving_bintrajs(2.5)
        bintrajs_per_traj = CG.binarize_trajs(2.5, order="traj")

        data_traj_0 = list_of_dicts[0]["data"]
        _np.testing.assert_array_equal(data_traj_0[:, 0], self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[0])
        _np.testing.assert_array_equal(data_traj_0[:, 1],
                                       bintrajs_per_traj[0][:, 0])
        _np.testing.assert_array_equal(data_traj_0[:, 2],
                                       bintrajs_per_traj[0][:, 1])

        data_traj_1 = list_of_dicts[1]["data"]
        _np.testing.assert_array_equal(data_traj_1[:, 0], self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[1])
        _np.testing.assert_array_equal(data_traj_1[:, 1],
                                       bintrajs_per_traj[1][:, 0])
        _np.testing.assert_array_equal(data_traj_1[:, 2],
                                       bintrajs_per_traj[1][:, 1])

    def test_to_per_traj_dicts_for_saving_bintrajs_ns(self):
        CG = self.CG
        list_of_dicts = CG._to_per_traj_dicts_for_saving_bintrajs(2.5, t_unit="ns")

        data_traj_0 = list_of_dicts[0]["data"]
        _np.testing.assert_array_equal(data_traj_0[:, 0],
                                       self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[0] * 1e-3)

        data_traj_1 = list_of_dicts[1]["data"]
        _np.testing.assert_array_equal(data_traj_1[:, 0],
                                       self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[1] * 1e-3)

    def test_to_per_traj_dicts_for_saving_bintrajs_mus(self):
        CG = self.CG
        list_of_dicts = CG._to_per_traj_dicts_for_saving_bintrajs(2.5, t_unit="mus")

        data_traj_0 = list_of_dicts[0]["data"]
        _np.testing.assert_array_equal(data_traj_0[:, 0],
                                       self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[0] * 1e-6)

        data_traj_1 = list_of_dicts[1]["data"]
        _np.testing.assert_array_equal(data_traj_1[:, 0],
                                       self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[1] * 1e-6)

    def test_to_per_traj_dicts_for_saving_bintrajs_ms(self):
        CG = self.CG
        list_of_dicts = CG._to_per_traj_dicts_for_saving_bintrajs(2.5, t_unit="ms")

        data_traj_0 = list_of_dicts[0]["data"]
        _np.testing.assert_array_equal(data_traj_0[:, 0],
                                       self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[0] * 1e-9)

        data_traj_1 = list_of_dicts[1]["data"]
        _np.testing.assert_array_equal(data_traj_1[:, 0],
                                       self.cp1_w_anchor_and_frags_and_top.time_traces.time_trajs[1] * 1e-9)

    def test_to_per_traj_dicts_for_saving_bintrajs_tunits_raises(self):
        CG = self.CG
        with self.assertRaises(ValueError):
            list_of_dicts = CG._to_per_traj_dicts_for_saving_bintrajs(2.5, t_unit="fs")

class TestContactGroupSavetrajs(TestBaseClassContactGroup):

    @classmethod
    def setUpClass(cls):
        super(TestContactGroupSavetrajs,cls).setUp(cls)
        cls.CG_cp1_cp2_both_w_anchor_and_frags_and_top = contacts.ContactGroup(
            [cls.cp1_w_anchor_and_frags_and_top,
             cls.cp2_w_anchor_and_frags_and_top],
            neighbors_excluded=0
        )
        cls.CG_cp1_cp2_both_wtop_and_conslabs = contacts.ContactGroup(
            [cls.cp1_wtop_and_conslabs,
             cls.cp2_wtop_and_conslabs])
    def test_save_trajs_no_anchor(self):
        CG = self.CG_cp1_cp2_both_wtop_and_conslabs
        with _TDir(suffix='_test_mdciao') as tempdir:
            CG.save_trajs("test", "dat", verbose=True, t_unit="ns",
                          output_dir=tempdir)

    def test_save_trajs_no_anchor_npy(self):
        CG = self.CG_cp1_cp2_both_wtop_and_conslabs
        with _TDir(suffix='_test_mdciao') as tempdir:
            CG.save_trajs("test", "npy", verbose=True, t_unit="ns",
                          output_dir=tempdir)


    def test_save_trajs_w_anchor(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags_and_top
        with _TDir(suffix='_test_mdciao') as tempdir:
            CG.save_trajs("test", "dat", verbose=True, t_unit="ns",
                          output_dir=tempdir)

    def test_save_trajs_no_anchor_excel(self):
        CG = self.CG_cp1_cp2_both_wtop_and_conslabs
        with _TDir(suffix='_test_mdciao') as tempdir:
            CG.save_trajs("test", "xlsx", verbose=True, t_unit="ns",
                          output_dir=tempdir)

    def test_save_trajs_w_anchor_excel(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags_and_top
        with _TDir(suffix='_test_mdciao') as tempdir:
            CG.save_trajs("test", "xlsx", verbose=True, t_unit="ns",
                          output_dir=tempdir)

    def test_save_trajs_w_anchor_None(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags_and_top
        with _TDir(suffix='_test_mdciao') as tempdir:
            CG.save_trajs("test", None, verbose=True, t_unit="ns",
                          output_dir=tempdir)

    def test_save_trajs_w_anchor_basename(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags_and_top
        # Overwrite the trajlabels for the basename to work
        CG._trajlabels = ["../non/existing/dir/traj1.xtc",
                          "../non/existing/dir/traj2.xtc"]
        with _TDir(suffix='_test_mdciao') as tempdir:
            CG.save_trajs("test", "dat", verbose=True, t_unit="ns",
                          output_dir=tempdir)

    def test_save_trajs_w_anchor_cutoff(self):
        CG = self.CG_cp1_cp2_both_w_anchor_and_frags_and_top
        with _TDir(suffix='_test_mdciao') as tempdir:
            CG.save_trajs("test_None", "dat", verbose=True, t_unit="ns",
                          ctc_cutoff_Ang=2.5,
                          output_dir=tempdir)

class TestContactGroupMeansNModes(unittest.TestCase):

    def setUp(self):
        time = [_np.arange(7), _np.arange(1)]
        traj00, traj01 = [1, 1, 1, 2, 3, 4, 5], [500]
        traj10, traj11 = [5, 5, 5, 5, 3, 4, 1], [1500]
        self.CG = contacts.ContactGroup([
            contacts.ContactPair([0, 1], [traj00, traj01], time),
            contacts.ContactPair([0, 2], [traj10, traj11], time)])

    def test_means(self):
        _np.testing.assert_array_equal([_np.mean([1, 1, 1, 2, 3, 4, 5] + [500]),
                                        _np.mean([5, 5, 5, 5, 3, 4, 1] + [1500])],
                                       self.CG.means)

    def test_modes(self):
        _np.testing.assert_array_equal([1, 5], self.CG.modes
                                       )
class TestContactGroupInterface(TestBaseClassContactGroup):

    def setUp(self):
        super(TestContactGroupInterface, self).setUp()
        # TODO here in case i need extra stuff, otherwise

    def test_instantiates_raises_duplicates(self):
        with self.assertRaises(AssertionError) as e:
            I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                       self.cp2_wtop_and_conslabs,
                                       self.cp4_wtop_and_conslabs],
                                      interface_fragments = [[0, 0],
                                                         [1, 2]])

        with self.assertRaises(AssertionError) as e:
            I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                       self.cp2_wtop_and_conslabs,
                                       self.cp4_wtop_and_conslabs],
                                      interface_fragments = [[0, 3],
                                                         [1, 1, 2]])

    def test_instantiates_to_no_interface(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs],
                                  interface_fragments = [[10, 30],
                                                     [11, 20]])
        assert I.is_interface is False

    def test_instantiates_to_no_interface_even1res(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs],
                                  interface_fragments = [[0, 30],
                                                     [11, 20]])
        assert I.is_interface is False
        _np.testing.assert_array_equal(I.interface_residxs[0], [0])
        assert I.interface_residxs[1] == []

    def test_residxs(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs],
                                  interface_fragments = [[3, 0],
                                                     [2, 1]])
        assert I.is_interface
        _np.testing.assert_array_equal(I.interface_residxs[0], [0, 3])
        _np.testing.assert_array_equal(I.interface_residxs[1], [1, 2])

    def test_interface_reslabels_short(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs],
                                  interface_fragments = [[3, 0],
                                                     [2, 1]])
        assert I.is_interface
        _np.testing.assert_equal(I.interface_reslabels_short[0][0], "E30")
        _np.testing.assert_equal(I.interface_reslabels_short[0][1], "V33")

        _np.testing.assert_equal(I.interface_reslabels_short[1][0], "V31")
        _np.testing.assert_equal(I.interface_reslabels_short[1][1], "W32")

    def test_interface_labels_consensus(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs],
                                  interface_fragments = [[3, 0],
                                                     [2, 1]])

        _np.testing.assert_equal(I.interface_labels_consensus[0][0], "3.50")
        _np.testing.assert_equal(I.interface_labels_consensus[0][1], "3.51")

        _np.testing.assert_equal(I.interface_labels_consensus[1][0], "4.50")
        _np.testing.assert_equal(I.interface_labels_consensus[1][1], "5.50")

    def test_interface_labels_consensus_some_missing(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs,
                                   self.cp5_wtop_and_wo_conslabs],
                                  interface_fragments = [[3, 0, 4],
                                                     [2, 1, 5]])
        _np.testing.assert_equal(I.interface_labels_consensus[0][0], "3.50")
        _np.testing.assert_equal(I.interface_labels_consensus[0][1], "3.51")
        _np.testing.assert_equal(I.interface_labels_consensus[0][2], None)

        _np.testing.assert_equal(I.interface_labels_consensus[1][0], "4.50")
        _np.testing.assert_equal(I.interface_labels_consensus[1][1], "5.50")
        _np.testing.assert_equal(I.interface_labels_consensus[1][2], None)

    @unittest.skip("This attribute has been deprecated")
    def test_interface_orphaned_labels(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs,
                                   self.cp5_wtop_and_wo_conslabs],
                                  interface_fragments = [[3, 0, 4],
                                                     [2, 1, 5]])
        _np.testing.assert_equal(I.interface_shortAAs_missing_conslabels[0][0], "V34")
        _np.testing.assert_equal(I.interface_shortAAs_missing_conslabels[1][0], "G35")

    def test_interface_residue_names_w_best_fragments_short(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs],
                                  interface_fragments = [[3, 0],
                                                     [2, 1]])

        _np.testing.assert_equal(I.interface_residue_names_w_best_fragments_short[0][0], "E30@3.50")
        _np.testing.assert_equal(I.interface_residue_names_w_best_fragments_short[0][1], "V33@3.51")

        _np.testing.assert_equal(I.interface_residue_names_w_best_fragments_short[1][0], "V31@4.50")
        _np.testing.assert_equal(I.interface_residue_names_w_best_fragments_short[1][1], "W32@5.50")

    def test_frequency_sum_per_residue_names_dict(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs,
                                   self.cp5_wtop_and_wo_conslabs],
                                  interface_fragments = [[3, 0, 4],
                                                     [2, 1, 5]])
        print(I.frequency_dataframe(2))
        print(I.interface_residxs)
        idicts = I.frequency_sum_per_residue_names(2,
                                                   sort_by_freq=False,
                                                   list_by_interface=True)

        assert len(idicts) == 2
        items0, items1 = list(idicts[0].items()), list(idicts[1].items())
        _np.testing.assert_array_equal(items0[0], ["E30@3.50", 1.])
        _np.testing.assert_array_equal(items0[1], ["V33@3.51", 0.])
        _np.testing.assert_array_equal(items0[2], ["V34", 1 / 3])

        _np.testing.assert_array_equal(items1[0], ["V31@4.50", 2 / 3.])
        _np.testing.assert_array_equal(items1[1], ["W32@5.50", 1 / 3])
        _np.testing.assert_array_equal(items1[2], ["G35", 1 / 3])

    def test_frequency_sum_per_residue_names_dict(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs,
                                   self.cp5_wtop_and_wo_conslabs],
                                  interface_fragments = [[3, 0, 4],
                                                     [2, 1, 5]])

        idicts = I.frequency_sum_per_residue_names(2,
                                                   list_by_interface=True)

        assert len(idicts) == 2
        self.assertDictEqual(idicts[0],
                             {"E30@3.50": 1.,
                              "V33@3.51": 0.,
                              "V34" : 1 / 3})
        self.assertDictEqual(idicts[1],
                             {"V31@4.50" : 2 / 3,
                              "W32@5.50" : 1 / 3,
                              "G35" : 1 / 3})

    # smh repeated from testing ContactGroup itself,
    # leaving it here
    def test_frequency_spreadsheet_w_interface(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs,
                                   self.cp5_wtop_and_wo_conslabs],
                                  interface_fragments = [[3, 0, 4],
                                                     [2, 1, 5]])
        with _TDir(suffix='_test_mdciao') as tmpdir:
            I.frequency_table(2.5, path.join(tmpdir, "test.xlsx"))

    def test_plot_frequency_sums_as_bars(self):
        I = contacts.ContactGroup([self.cp1_wtop_and_conslabs,
                                   self.cp2_wtop_and_conslabs,
                                   self.cp4_wtop_and_conslabs,
                                   self.cp5_wtop_and_wo_conslabs],
                                  interface_fragments = [[3, 0, 4],
                                                     [2, 1, 5]])

        iax = I.plot_frequency_sums_as_bars(2, "interface",
                                            list_by_interface=True,
                                            interface_vline=True)

class Test_retop_CG(unittest.TestCase):

    def test_just_works(self):
        intf = _mdcli.interface(md.load(test_filenames.actor_pdb),
                                fragments=[_np.arange(868, 875 + 1),
                                            _np.arange(328, 353 + 1)],
                                ctc_cutoff_Ang=30,
                                no_disk=True,
                                figures=False
                                )
        top3SN6 = md.load(test_filenames.pdb_3SN6)
        df = _mdcu.sequence.align_tops_or_seqs(intf.top, top3SN6.top,
                                                      #verbose=True,
                                                      return_DF=True)[0]
        mapping, __ = _mdcu.sequence.df2maps(df,allow_nonmatch=False)

        intf_retop = intf.retop(top3SN6.top, mapping)
        for list1, list2 in zip(intf.interface_residxs,
                                intf_retop.interface_residxs):
            for r1, r2 in zip(list1, list2):
                assert str(intf.top.residue(r1))==str(intf_retop.top.residue(r2))

class Test_archive_CG(unittest.TestCase):

    def test_works(self):
        CG = examples.ContactGroupL394()
        arch = CG.archive()

        sCP = arch["serialized_CPs"][0]
        contacts.ContactPair(sCP["residues.idxs_pair"],
                             sCP["time_traces.ctc_trajs"],
                             sCP["time_traces.time_trajs"],
                             trajs=sCP["time_traces.trajs"],
                             atom_pair_trajs=sCP["time_traces.atom_pair_trajs"],
                             fragment_idxs=sCP["fragments.idxs"],
                             fragment_names=sCP["fragments.names"],
                             fragment_colors=sCP["fragments.colors"],
                             )

        assert CG.interface_residxs is arch["interface_residxs"]
        assert CG.name is arch["name"]
        assert CG.neighbors_excluded is arch["neighbors_excluded"]

    def test_saves_npy(self):
        with _TDir() as t:
            fname = path.join(t,"archive.npy")
            CG = examples.ContactGroupL394()
            CG.archive(fname)

            loaded_CG = _np.load(fname,allow_pickle=True)[()]
        for key in ["serialized_CPs", "interface_residxs", "name", "neighbors_excluded"]:
            assert key in loaded_CG.keys()


class Test_linear_switchoff(unittest.TestCase):

    def setUp(self):
        self.d_in_Ang =_np.array([3., 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.])
        self.linearized = _np.array([1., .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.])
    def test_just_works(self):
        # Switch in 1 ang
        _np.testing.assert_allclose(self.linearized,
                                    _linear_switchoff(self.d_in_Ang, 3, 1))
        # Switch in 2 Ang
        _np.testing.assert_allclose(_np.array([1., .95, .90, .85, .80, .75, .70, .65, .60, .55, .5]),
                                    _linear_switchoff(self.d_in_Ang, 3, 2))

    def test_from_ContactPair(self):
        CP = contacts.ContactPair([0,1], [self.d_in_Ang / 10], [_np.arange(len(self.d_in_Ang))])
        bintrajs = CP.binarize_trajs(3,switch_off_Ang=1)
        _np.testing.assert_allclose(bintrajs[0],self.linearized,atol=1e-7)

    def test_from_ContactGroup_bintrajs(self):
        CG = contacts.ContactGroup([contacts.ContactPair([0, 1],
                                                         [self.d_in_Ang / 10],
                                                         [_np.arange(len(self.d_in_Ang))]),
                                    contacts.ContactPair([0, 2],
                                                         [self.d_in_Ang[::-1] / 10],
                                                         [_np.arange(len(self.d_in_Ang))])])

        bintrajs = CG.binarize_trajs(3,switch_off_Ang=1)
        _np.testing.assert_allclose(bintrajs[0][0], self.linearized,atol=1e-7)
        _np.testing.assert_allclose(bintrajs[1][0], self.linearized[::-1], atol=1e-7)

    def test_from_ContactGroup_frequencies(self):
        CG = contacts.ContactGroup([contacts.ContactPair([0, 1],
                                                         [self.d_in_Ang / 10],
                                                         [_np.arange(len(self.d_in_Ang))]),
                                    contacts.ContactPair([0, 2],
                                                         [self.d_in_Ang[::-1] / 10],
                                                         [_np.arange(len(self.d_in_Ang))])])

        _np.testing.assert_almost_equal(CG.frequency_per_contact(3, switch_off_Ang=1),
                                        [_np.mean(self.linearized), _np.mean(self.linearized)],
                                        )


class TestContactPairHashingEqualitySaving(TestBaseClassContactGroup):

    def test_not_equal(self):
        assert self.cp1 != self.cp2

    def test_copy_is_equal(self):
        assert self.cp1 == self.cp1.copy()

    def test_hash_survives_pickle(self):

        with _TDir() as tdir:
            picklefile = path.join(tdir,"CP.pickle")
            self.cp1.save(picklefile)
            cp1 = contacts.load(picklefile)

        assert self.cp1 == cp1

class TestContactGroupHashingEqualitySaving(TestBaseClassContactGroup):

    def test_works(self):
        CG1 = contacts.ContactGroup([self.cp1,self.cp2])
        CG1c = contacts.ContactGroup([self.cp1,self.cp2])
        CG2 = contacts.ContactGroup([self.cp1,self.cp2, self.cp3])
        assert CG1 == CG1c == CG1.copy()
        assert CG1 != CG2

    def test_hash_survives_pickle(self):
        CG1 = contacts.ContactGroup([self.cp1,self.cp2])
        with _TDir() as tdir:
            picklefile = path.join(tdir, "CG.pickle")
            CG1.save(picklefile)
            CG1p = contacts.load(picklefile)

        assert CG1 == CG1p

if __name__ == '__main__':
    unittest.main()

class TestBaseClassGroupOfGroupsOfContacts(unittest.TestCase):

    def setUp(self):
        self.top = md.load(test_filenames.actor_pdb).top

        self.interface_fragments = [[0,3,4],[1,2]]
        self.cp1_wtop_and_conslabs = contacts.ContactPair([0, 1], [[.1, .2, .3]], [[1, 2, 3]],
                                                          consensus_labels=["3.50", "4.50"],
                                                          top=self.top)

        self.cp2_wtop_and_conslabs = contacts.ContactPair([0, 2], [[.15, .25, .35]], [[1, 2, 3]],
                                                          consensus_labels=["3.50", "5.50"],
                                                          top=self.top)

        self.cp3_wtop_and_conslabs = contacts.ContactPair([3, 2], [[.25, .25, .35]], [[1, 2, 3]],
                                                          consensus_labels=["3.51", "5.50"],
                                                          top=self.top)

        self.cp4_wtop_and_1_conslabs = contacts.ContactPair([4, 1], [[.25, .15, .35]], [[1, 2, 3]],
                                                          consensus_labels=[None, "4.50"],
                                                          top=self.top)



        self.CG1 = contacts.ContactGroup([self.cp1_wtop_and_conslabs, self.cp2_wtop_and_conslabs],
                                         interface_fragments = [[0],[2,1]],
                                         top=self.top)
        self.CG2 = contacts.ContactGroup([self.cp3_wtop_and_conslabs, self.cp4_wtop_and_1_conslabs],
                                         interface_fragments = [[4,3],[1,2]],
                                         top=self.top)

class TestGroupOfGroupsOfContactsBasic(TestBaseClassGroupOfGroupsOfContacts):

    def test_just_works(self):
        GGC = contacts.GroupOfInterfaces({"CG1":self.CG1, "CG2":self.CG2})

        _np.testing.assert_equal(2, GGC.n_groups)

        _np.testing.assert_array_equal(GGC.interface_names,["CG1","CG2"])

    def test_interface_labels_consensus_None(self):
        GGC = contacts.GroupOfInterfaces({"CG1":self.CG1, "CG2":self.CG2},
                                         relabel_consensus=False)
        labs = GGC.interface_labels_consensus
        _np.testing.assert_array_equal(["3.50","3.51",None],labs[0])
        _np.testing.assert_array_equal(["4.50","5.50"],labs[1])


    def test_relabel_consensus(self):
        GGC = contacts.GroupOfInterfaces({"CG1": self.CG1, "CG2": self.CG2})
        labs = GGC.interface_labels_consensus
        _np.testing.assert_array_equal(["3.50", "3.51", "V34"], labs[0])
        _np.testing.assert_array_equal(["4.50", "5.50"], labs[1])

    def test_conlab2matidx(self):
        GGC = contacts.GroupOfInterfaces({"CG1": self.CG1, "CG2": self.CG2})
        self.assertDictEqual(GGC.conlab2matidx , {"3.50":[0,0],
                                                  "3.51":[0,1],
                                                  "V34":[0,2],
                                                  "4.50":[1,0],
                                                  "5.50":[1,1]})

    def test_frequency_dict_by_consensus_labels(self):
        GGC = contacts.GroupOfInterfaces({"CG1": self.CG1, "CG2": self.CG2})
        odict = GGC.interface_frequency_dict_by_consensus_labels(3)
        ref_dict = {"3.50":{"4.50":1   /GGC.n_groups,
                            "5.50":2/3 /GGC.n_groups},
                    "3.51":{"5.50":2/3 /GGC.n_groups},
                    "V34": {"4.50":2/3 /GGC.n_groups}}

        self.assertDictEqual(ref_dict, odict)


    def test_interface_matrix(self):
        GGC = contacts.GroupOfInterfaces({"CG1": self.CG1, "CG2": self.CG2})
        mat = GGC.interface_matrix(3)
        mat_ref = _np.zeros_like(mat)
        mat_ref[0,0] = 1.0 # 3.50,4.50
        mat_ref[0,1] = 2/3 #3.50,5.50

        mat_ref[1,0] = 0   # 3.51,4.50
        mat_ref[1,1] = 2/3 # 3.51,5.50

        mat_ref[2,0] = 2/3 #V34,4.50
        mat_ref[2,1] = 0 #V34,5.50

        mat_ref /= GGC.n_groups

        _np.testing.assert_array_equal(mat, mat_ref)


class Test_modified_mdtraj_contacts(unittest.TestCase):

    def setUp(self):
        self.traj = md.load(test_filenames.traj_xtc_stride_20, top=test_filenames.top_pdb)

    def test_works(self):
        ctcs_ref, residxs_ref = md.compute_contacts(self.traj, [[10, 20], [100, 200]])
        ctcs_tst, residxs_tst, __ = contacts._md_compute_contacts.compute_contacts(self.traj, [[10, 20], [100, 200]])
        _np.testing.assert_array_equal(ctcs_ref, ctcs_tst)
        _np.testing.assert_array_equal(residxs_ref, residxs_tst)

    def test_schemes(self):
        for scheme in ['ca', 'closest', 'closest-heavy', 'sidechain', 'sidechain-heavy']:
            ctcs_ref, residxs_ref = md.compute_contacts(self.traj, [[10, 20], [100, 200]], scheme=scheme)
            ctcs_tst, residxs_tst, __ = contacts._md_compute_contacts.compute_contacts(self.traj, [[10, 20], [100, 200]],
                                                                          scheme=scheme)
            _np.testing.assert_array_equal(ctcs_ref, ctcs_tst)
            _np.testing.assert_array_equal(residxs_ref, residxs_tst)

    def test_ca_atom_pairs(self):
        ctcs_tst, residxs_tst, aa_pairs = contacts._md_compute_contacts.compute_contacts(self.traj, [[10, 20], [100, 200]],
                                                                                   scheme="ca")
        assert len(aa_pairs)==len(ctcs_tst)
        assert _np.shape(aa_pairs)==(self.traj.n_frames,4)
        assert tuple(_np.unique(aa_pairs,axis=0).squeeze())==tuple([self.traj.top.residue(rr).atom("CA").index for rr in [10,20,100,200]])

    def test_softmin(self):
        ctcs_ref, residxs_ref = md.compute_contacts(self.traj, [[10, 20], [100, 200]], soft_min=True)
        ctcs_tst, residxs_tst, __ = contacts._md_compute_contacts.compute_contacts(self.traj, [[10, 20], [100, 200]], soft_min=True)
        _np.testing.assert_array_equal(ctcs_ref, ctcs_tst)
        _np.testing.assert_array_equal(residxs_ref, residxs_tst)

    def test_all(self):
        small_traj = self.traj.atom_slice(self.traj.top.select("residue < 10"))
        ctcs_ref, residxs_ref = md.compute_contacts(small_traj, "all")
        ctcs_tst, residxs_tst, __ = contacts._md_compute_contacts.compute_contacts(small_traj, "all")
        _np.testing.assert_array_equal(ctcs_ref, ctcs_tst)
        _np.testing.assert_array_equal(residxs_ref, residxs_tst)

    def test_failures(self):
        with _np.testing.assert_raises(ValueError):
            contacts._md_compute_contacts.compute_contacts(self.traj, 'al')

        with _np.testing.assert_raises(ValueError):
            contacts._md_compute_contacts.compute_contacts(self.traj, [])

        with _np.testing.assert_raises(ValueError):
            no_top = self.traj[:]
            setattr(no_top, "_topology", None)
            contacts._md_compute_contacts.compute_contacts(no_top, [[0,1]])

        with _np.testing.assert_raises(ValueError):
            contacts._md_compute_contacts.compute_contacts(self.traj, [[0,1]], scheme='CB    ')

    def test_passes(self):
        contacts._md_compute_contacts.compute_contacts(self.traj, [[0, 1]], scheme='ca', soft_min=True)

import tempfile

import numpy as np

from mdciao.utils.COM import *
from mdciao.utils.COM import _unwrap, _per_residue_unwrapping
import mdtraj as md
import numpy as _np
import unittest
from mdciao.examples import filenames as test_filenames

class Test_COM_utils(unittest.TestCase):

    def setUp(self):
        super(Test_COM_utils,self).setUp()
        self.pdb_file = test_filenames.top_pdb
        self.file_xtc = test_filenames.traj_xtc_stride_20
        self.top = md.load(self.pdb_file).top
        self.traj = md.load(self.file_xtc, top=self.top)

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

        _np.testing.assert_allclose(COMSs_mine,
                                    self.COMS_mdtraj[:,residue_idxs])

    def test_COMdist_works(self):
        res_pairs = [[0,10], [10,20]]
        Dref = _np.vstack((_np.linalg.norm(self.COMS_mdtraj[:,0]-self.COMS_mdtraj[:,10], axis=1),
                           _np.linalg.norm(self.COMS_mdtraj[:,10]- self.COMS_mdtraj[:,20], axis=1))).T
        COMdist =  geom2COMdist(self.traj_5_frames, residue_pairs=res_pairs)
        _np.testing.assert_array_almost_equal(Dref, COMdist)

class Test_geom2max_residue_radius(unittest.TestCase):

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
        with tempfile.NamedTemporaryFile(suffix=".pdb") as tf:
            with open(tf.name, "w") as f:
                f.write(pdb)
            self.geom = md.load(tf.name)
            self.geom = self.geom.join(self.geom)
            self.geom._xyz[1,:,:] *= 10
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

    def test_works(self):
        rrm = geom2max_residue_radius(self.geom)
        _np.testing.assert_array_almost_equal(rrm, _np.array([[.6, .5, .6],
                                                              [6, 5, 6]]))

    def test_w_residxs(self):
        rrm = geom2max_residue_radius(self.geom, residue_idxs=[1, 2])
        _np.testing.assert_array_almost_equal(rrm, _np.array([[.5, .6],
                                                              [5, 6]]))

    def test_w_residxs_and_resCOM(self):
        rrm = geom2max_residue_radius(self.geom, residue_idxs=[1, 2], res_COMs=_np.array([[[0, .5, 0],
                                                                                           [0, 0, .6]],
                                                                                          [[0, 5, 0],
                                                                                           [0, 0, 6]]
                                                                                          ]))
        _np.testing.assert_array_almost_equal(rrm, _np.array([[.5, .6],
                                                              [5, 6]]))

    def test_raises(self):
        with self.assertRaises(ValueError):
            geom2max_residue_radius(self.geom, res_COMs=_np.array([[[0, .5, 0],
                                                                    [0, 0, .6]],
                                                                   [[0, 5, 0],
                                                                    [0, 0, 6]]
                                                                   ]))

    def test_COMdist_subtract_radii_low_and_high_mem(self):
        res_coms = _np.array([[4,0,0],
                              [0,5,0],
                              [0,0,6]], dtype=float) #<- we know this from the definition of the geom
        res_coms *= .1 #convert to nm
        res_coms = _np.array([res_coms, res_coms*10]) # add a second frame
        _np.testing.assert_array_almost_equal(res_coms, geom2COMxyz(self.geom, [0,1,2])) # testing geom2COM on-the-fly

        d_0_1 = _np.sqrt(4**2+5**2)*.1
        d_0_2 = _np.sqrt(4**2+6**2)*.1
        d_1_2 = _np.sqrt(5**2+6**2)*.1
        d_0_1 = _np.array((d_0_1, d_0_1 * 10))
        d_0_2 = _np.array((d_0_2, d_0_2 * 10))
        d_1_2 = _np.array((d_1_2, d_1_2 * 10))
        comD = geom2COMdist(self.geom, [[0, 1], [0, 2], [1, 2]], periodic=False, per_residue_unwrap=False )
        _np.testing.assert_array_almost_equal(comD, _np.array([d_0_1, d_0_2, d_1_2]).T) # testing geom2COMdist w/o subtract_radii

        # Test low-mem
        # We subtract the same max-radii by hand (which we know as rrm from test_works)
        # for all frames of comD
        comD[:, 0] -= (6 + 5)
        comD[:, 1] -= (6 + 6)
        comD[:, 2] -= (5 + 6)

        # This tests low_mem
        comDr = geom2COMdist(self.geom, [[0, 1], [0, 2], [1, 2]], subtract_max_radii=True, periodic=False, per_residue_unwrap=False)
        _np.testing.assert_array_almost_equal(comDr, comD)  # the actual test

        # Test high_mem
        comD = geom2COMdist(self.geom, [[0, 1], [0, 2], [1, 2]],  periodic=False, per_residue_unwrap=False)
        # We subtract the per-frame max-radii by hand (which we know as rrm from test_works)
        comD[0][0] -= (.6 + .5)
        comD[0][1] -= (.6 + .6)
        comD[0][2] -= (.5 + .6)
        comD[1][0] -= (6 + 5)
        comD[1][1] -= (6 + 6)
        comD[1][2] -= (5 + 6)

        # This test high_mem
        comDr = geom2COMdist(self.geom, [[0, 1], [0, 2], [1, 2]], subtract_max_radii=True, low_mem=False, periodic=False, per_residue_unwrap=False)
        _np.testing.assert_array_almost_equal(comDr, comD)  # the actual test

class TestUnwrapping(unittest.TestCase):

    def setUp(self) -> None:
        pdb = "CRYST1   100.00   100.00  100.000  90.00  90.00  90.00 P 1           1 \n" \
              "MODEL        0 \n" \
              "ATOM      1  CA  GLU A  30      5.0000  30.000  10.000  1.00  0.00           C \n" \
              "ATOM      2  CB  GLU A  30      90.000  95.000  10.000  1.00  0.00           C \n" \
              "ATOM      3  CA  VAL A  31      0.0000  0.0000  0.0000  1.00  0.00           C \n" \
              "ATOM      4  CB  VAL A  31      0.0000  5.0000  0.0000  1.00  0.00           C \n" \
              "TER       4      VAL A  31 "
        with tempfile.NamedTemporaryFile(suffix=".pdb") as tf:
            with open(tf.name, "w") as f:
                f.write(pdb)
            self.geom = md.load(tf.name)

    def test_unwrap_coords(self):
        unitcell_lengths = _np.array([[10, 10, 10]])


        wrapped_coords = _np.array([[[.5, 3, 1], # <- This atom is closest to the center of the box, its coords won't change
                                     [9, 9.5, 1]]])

        un_wrapped_coords_ref = _np.array([[[.5, 3, 1],
                                            [-1, -0.5, 1]]])

        _unwrapped_coords = _unwrap(wrapped_coords, unitcell_lengths)
        _np.testing.assert_array_equal(_unwrapped_coords,
                                       un_wrapped_coords_ref
                                       )

    def test_unwrap_residues(self):
        per_res_unwrapped = _per_residue_unwrapping(self.geom)

        _np.testing.assert_array_equal(per_res_unwrapped.xyz, _np.array([[[.5, 3, 1],
                                                                          [-1, -0.5, 1],
                                                                          [0, 0, 0],
                                                                          [0, .5, 0]]]))

        assert per_res_unwrapped is not self.geom

    def test_unwrap_residues_residue_idxs(self):
        # will leave geom untouched bc residue 1 does not need unwrapping
        per_res_unwrapped = _per_residue_unwrapping(self.geom, residue_idxs=[1])

        _np.testing.assert_array_equal(per_res_unwrapped.xyz, _np.array([[[.5, 3, 1],
                                                                          [9, 9.5, 1],
                                                                          [0, 0, 0],
                                                                          [0, .5, 0]]]))

        # will have only unwrapped residue 0
        per_res_unwrapped = _per_residue_unwrapping(self.geom, residue_idxs=[0])

        _np.testing.assert_array_equal(per_res_unwrapped.xyz, _np.array([[[.5, 3, 1],
                                                                          [-1, -0.5, 1],
                                                                          [0, 0, 0],
                                                                          [0, .5, 0]]]))

        assert per_res_unwrapped is not self.geom

    def test_inplace_w_maxradii(self):
        geom = md.Trajectory(_np.copy(self.geom._xyz), self.geom.top, time=self.geom.time, unitcell_angles=self.geom.unitcell_angles,
                       unitcell_lengths=self.geom.unitcell_lengths)
        per_res_unwrapped = _per_residue_unwrapping(geom, inplace=True,
                                                    max_res_radii_t=geom2max_residue_radius(geom))
        _np.testing.assert_array_equal(per_res_unwrapped.xyz, _np.array([[[.5, 3, 1],
                                                                          [-1, -0.5, 1],
                                                                          [0, 0, 0],
                                                                          [0, .5, 0]]]))

        assert per_res_unwrapped is geom

    def test_COM_dist_wo_unwrapping_wo_subtract_wo_periodic(self):
        COM1 = np.array([5 + 90, 30 + 95, 10 + 10]) / 2 / 10    # [4.75 6.25 1.  ]
        COM2 = np.array([0 + 0, 0 + 5, 0 + 0]) / 2 / 10         # [0.   0.25 0.  ]
        D12 = np.linalg.norm(COM1 - COM2)

        np.testing.assert_array_equal(D12,
                                      geom2COMdist(self.geom, [[0, 1]],
                                                   subtract_max_radii=False, low_mem=True, periodic=False,
                                                   per_residue_unwrap=False))

    def test_COM_dist_wo_unwrapping_wo_subtract_w_periodic(self):
        # PBCs
        dx = 4.75
        dy = 6  # This is larger than half the box, hence pick the periodic image
        dy -= self.geom.unitcell_lengths[0, 1]
        dz = 1
        np.testing.assert_array_equal(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2),
                                      geom2COMdist(self.geom, [[0, 1]],
                                                   subtract_max_radii=False, low_mem=True, periodic=True,
                                                   per_residue_unwrap=False))

    def test_COM_dist_w_unwrapping_wo_subtract(self):
        """
        # These we take from above
        unwrapped_coords = _np.array([[[.5, 3, 1],
                                     [-1, -0.5, 1],
                                     [0, 0, 0],
                                     [0, .5, 0]]])

        COM1 = np.array([.5 - 1, 3 - .5, 1 + 1]) / 2    # [-0.25  1.25  1.  ]
        COM2 = np.array([0 + 0, 0 + .5, 0 + 0]) / 2     # [0.  .25 0. ]
        """
        dx = .25
        dy = 1
        dz = 1

        # None of the deltas is larger than half the box, no correction needed
        np.testing.assert_almost_equal(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2),
                                       geom2COMdist(self.geom, [[0, 1]],
                                                    subtract_max_radii=False, low_mem=True, periodic=True,
                                                    per_residue_unwrap=True))

    def test_COM_dist_w_unwrapping_w_subtract_low_and_hi_mem(self):
        # The COMs we take from above.
        COM1, COM2 = _np.array([-0.25, 1.25, 1.]), _np.array([0., .25, 0.])
        # Since it's only 2 atoms per residue, any atom of the residue can be used for the max residue radius
        max_r_1 = _np.linalg.norm(COM1 - np.array([.5, 3, 1]))
        max_r_2 = _np.linalg.norm(COM2 - np.array([0, 0, 0]))
        # None of the deltas is larger than half the box, no correction needed
        D12 = np.linalg.norm(COM1 - COM2) - max_r_1 - max_r_2
        np.testing.assert_almost_equal(D12,
                                       geom2COMdist(self.geom, [[0, 1]],
                                                    subtract_max_radii=True, low_mem=True, periodic=True,
                                                    per_residue_unwrap=True))

        np.testing.assert_almost_equal(D12,
                                       geom2COMdist(self.geom, [[0, 1]],
                                                    subtract_max_radii=True, low_mem=False, periodic=True,
                                                    per_residue_unwrap=True))

if __name__ == '__main__':
    unittest.main()
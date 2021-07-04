from mdciao.utils.COM import *
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

        _np.testing.assert_allclose(COMSs_mine[:, [residue_idxs]],
                                    self.COMS_mdtraj[:,[residue_idxs]])

    def test_COMdist_works(self):
        res_pairs = [[0,10], [10,20]]
        Dref = _np.vstack((_np.linalg.norm(self.COMS_mdtraj[:,0]-self.COMS_mdtraj[:,10], axis=1),
                           _np.linalg.norm(self.COMS_mdtraj[:,10]- self.COMS_mdtraj[:,20], axis=1))).T
        COMdist =  geom2COMdist(self.traj_5_frames, residue_pairs=res_pairs)
        _np.testing.assert_allclose(Dref, COMdist)

if __name__ == '__main__':
    unittest.main()
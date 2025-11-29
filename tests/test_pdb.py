import unittest

from tempfile import TemporaryDirectory as _TD
import os
from mdciao.examples.examples import remember_cwd, filenames
import mdtraj as md
from Bio.PDB import MMCIFParser, MMCIF2Dict
import gzip, shutil
from scipy.spatial.distance import cdist
import sys
MINOR_PYTHON_VERSION = sys.version_info.minor
from mdciao import pdb
class Test_pdb2ref(unittest.TestCase):

    def test_works(self):
        cite = pdb.pdb2ref("3SN6")
        assert isinstance(cite,dict)
        assert all([key in cite.keys() for key in ["title", "rcsb_authors", "rcsb_journal_abbrev","year"]])

    def test_work_wrong_PDB(self):
        cite = pdb.pdb2ref("0SN6")
        assert cite is None

    def test_work_with_to_be_published(self):
        cite = pdb.pdb2ref("7dua")
        assert isinstance(cite,dict)
    def test_work_wo_doi(self):
        cite = pdb.pdb2ref("6vg3")
        assert isinstance(cite, dict)

class Test_url2json(unittest.TestCase):

    def test_works(self):
        idict = pdb.pdb._url2json("https://data.rcsb.org/rest/v1/core/entry/3SN6",
                                                    5,True)
        assert isinstance(idict,dict)

    def test_work_url_works_json_is_404(self):
        idict = pdb.pdb._url2json("https://data.rcsb.org/rest/v1/core/entry/13SN6",
                                                    5,True)
        assert idict["status"] == 404

    def test_work_url_works_cant_json(self):
        idict = pdb.pdb._url2json("https://rcsb.org",5, True)
        assert isinstance(idict,ValueError)


class Test_pdb2traj(unittest.TestCase):

    def test_works(self):
        with _TD(suffix="_test_mdciao_pdb2traj") as tmpdir:
            with remember_cwd():
                os.chdir(tmpdir)
                pdb.pdb2traj("3SN6")
                assert len(os.listdir())==0
    def test_saves_pdb(self):
        with _TD(suffix="_test_mdciao_pdb2traj") as tmpdir:
            with remember_cwd():
                os.chdir(tmpdir)
                fname = "3SN6.pdb"
                pdb.pdb2traj("3SN6",fname)
                assert os.path.exists(fname)

    def test_gets_cif(self):
        with _TD(suffix="_test_mdciao_pdb2traj") as tmpdir:
            with remember_cwd():
                os.chdir(tmpdir)
                fname = "3SN6.cif"
                pdb.pdb2traj("3SN6", fname, cif_first=True, cif_mdtraj=MINOR_PYTHON_VERSION>10)
                assert os.path.exists(fname)

    def test_auto_filename(self):
        with _TD(suffix="_test_mdciao_pdb2traj") as tmpdir:
            with remember_cwd():
                os.chdir(tmpdir)
                pdb.pdb2traj("3SN6", True)
                assert os.path.exists("3SN6.pdb")

    def test_convert_w_extension(self):
        with _TD(suffix="_test_mdciao_pdb2traj") as tmpdir:
            with remember_cwd():
                os.chdir(tmpdir)
                fname = "3SN6.pdb"
                pdb.pdb2traj("3SN6", fname, cif_first=True,  cif_mdtraj=MINOR_PYTHON_VERSION>10)
                assert os.path.exists(fname)


    def test_wrong_url(self):
        pdb.pdb2traj("3SNXX")

    def test_mdtraj_cif(self):
        if MINOR_PYTHON_VERSION<=10:
            with self.assertRaises(ValueError):
                pdb.pdb2traj("4V6X")
        else:
            traj = pdb.pdb2traj("4V6X")
            assert isinstance(traj, md.Trajectory)


class Test_BiopythonPDB(unittest.TestCase):

    def test_BIOStructure2MDTrajectory(self):
        #implicitly also test BIOStructure2pdbfile
        ungzipped = os.path.basename(filenames.rcsb_8E0G_cif).replace(".cif.gz",".cif")
        #https://docs.python.org/3/library/gzip.html#examples-of-usage
        with _TD(suffix="_test_mdciao_pdb2traj") as tmpdir:
            with remember_cwd():
                os.chdir(tmpdir)
                with gzip.open(filenames.rcsb_8E0G_cif, 'rb') as f_in:
                    with open(ungzipped, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                structure = MMCIFParser().get_structure(ungzipped, ungzipped)
                geom = pdb.pdb._BIOStructure2MDTrajectory(structure, cif_dict=MMCIF2Dict.MMCIF2Dict(ungzipped))

        # Replicate what _BIOStructure2pdbfile will do with disordered_select_A=True
        [atom.disordered_select("A") for atom in structure.get_atoms() if atom.is_disordered()]
        structure_atoms = list(structure.get_atoms())
        assert geom.n_atoms == len(structure_atoms)
        assert geom.n_residues == len(list(structure.get_residues()))
        assert geom.n_chains == len(list(structure.get_chains()))
        import numpy as np

        # Match atom order via coordinates
        bio_coords = np.array([aa.get_coord() / 10 for aa in structure_atoms])
        md_coords = geom.xyz.squeeze()
        d = cdist(md_coords, bio_coords)
        argmin = d.argmin(axis=1)
        dmin = d.min(axis=1)
        assert all(dmin<=1e-4)

        # Same coords
        np.testing.assert_array_almost_equal(bio_coords[argmin], md_coords)
        # Same atom name
        assert all([aa.name==bb.name for aa,bb in zip(geom.top.atoms, np.array(structure_atoms)[argmin])])

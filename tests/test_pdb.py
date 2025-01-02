import unittest

from tempfile import TemporaryDirectory as _TD
import os
from mdciao.examples.examples import remember_cwd



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

    def test_saves_cif(self):
        with _TD(suffix="_test_mdciao_pdb2traj") as tmpdir:
            with remember_cwd():
                os.chdir(tmpdir)
                fname = "7K00.cif"
                pdb.pdb2traj("7K00", fname)
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
                pdb.pdb2traj("7K00", "7K00.pdb")
                assert os.path.exists("7K00.pdb")


    def test_wrong_url(self):
        pdb.pdb2traj("3SNXX")

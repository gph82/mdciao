import unittest

from mdciao import pdb
class Test_pdb2ref(unittest.TestCase):

    def test_works(self):
        cite = pdb.pdb2ref("3SN6")
        assert isinstance(cite,dict)
        assert all([key in cite.keys() for key in ["title", "rcsb_authors", "rcsb_journal_abbrev","year"]])

    def test_work_wrong_PDB(self):
        cite = pdb.pdb2ref("0SN6")
        assert cite is None

class Test_url2json(unittest.TestCase):

    def test_works(self):
        idict = pdb.pdb._url2json("https://data.rcsb.org/rest/v1/core/entry/3SN6",
                                                    5,True)
        assert isinstance(idict,dict)

    def test_work_wrong_url(self):
        idict = pdb.pdb._url2json("https://data.rsbc.org/rest/v1/core/entry/13SN6",
                                                    5,True)
        assert isinstance(idict,ValueError)

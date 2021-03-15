import unittest
from mdciao.filenames import filenames
from mdciao.sites import siteIO
import pytest

import numpy as _np
import mdtraj as _md

test_filenames = filenames()
import mdciao

class Test_load(unittest.TestCase):

    def test_runs(self):
        site = mdciao.sites.load(test_filenames.GDP_json)
        _np.testing.assert_equal(site["name"],"GDP")
        _np.testing.assert_equal(site["n_bonds"],5)
        _np.testing.assert_equal(len(site["bonds"]["AAresSeq"]),5)
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][0], ['GDP396', 'ARG201'])
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][1], ['GDP396', 'THR204'])
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][2], ['GDP396', 'VAL202'])
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][3], ['THR204', 'SER54'])
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][4], ['MG397', 'THR204'])

    def test_runs_w_dict(self):
        site = mdciao.sites.load(test_filenames.GDP_json)
        _np.testing.assert_equal(site["name"], "GDP")
        _np.testing.assert_equal(site["n_bonds"], 5)
        _np.testing.assert_equal(len(site["bonds"]["AAresSeq"]), 5)
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][0], ['GDP396', 'ARG201'])
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][1], ['GDP396', 'THR204'])
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][2], ['GDP396', 'VAL202'])
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][3], ['THR204', 'SER54'])
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][4], ['MG397', 'THR204'])
        site = mdciao.sites.load(site)

    def test_raises_wo_name(self):
        site = mdciao.sites.load(test_filenames.GDP_name_json)
        site.pop("name")
        with _np.testing.assert_raises(ValueError):
            mdciao.sites.load(site)

    def test_runs_with_name(self):
        site = mdciao.sites.load(test_filenames.GDP_name_json)
        _np.testing.assert_equal(site["name"], "siteGDP")

    def test_runs_with_dat(self):
        site = mdciao.sites.load(test_filenames.tip_dat)
        site_test = mdciao.sites.load(test_filenames.tip_json)
        self.assertDictEqual(site["bonds"], site_test["bonds"])
        self.assertEqual(site["name"],"tip.json as plain ascii")
        self.assertEqual(site["n_bonds"],site_test["n_bonds"])

class Test_dat2site(unittest.TestCase):

    def test_works(self):
        site = siteIO.dat2site(test_filenames.tip_dat)
        self.assertDictEqual(site, {"bonds": {"AAresSeq": [
            "L394-K270",
            "D381-Q229",
            "Q384-Q229",
            "R385-Q229",
            "D381-K232",
            "Q384-I135"
        ]}, "name":"tip.json as plain ascii"}
                             )

class Test_site2str(unittest.TestCase):

    def test_sitefile(self):
        mdciao.sites.site2str(test_filenames.GDP_json)
    def test_sitedict(self):
        site = mdciao.sites.load(test_filenames.GDP_json)
        mdciao.sites.site2str(site)
    def test_what(self):
        with _np.testing.assert_raises(ValueError):
            mdciao.sites.site2str([1])

class Test_sites_to_ctc_idxs(unittest.TestCase):
    def setUp(self):
        self.GDP_json = test_filenames.GDP_json
        self.geom = _md.load(test_filenames.actor_pdb)
        self.fragments = mdciao.fragments.get_fragments(self.geom.top)

    def test_the_idxs_work_no_frags(self):
        site = mdciao.sites.load(self.GDP_json)
        ctc_idxs, __ = mdciao.sites.sites_to_res_pairs([site], self.geom.top)
        for (ii,jj), (resi,resj) in zip(ctc_idxs,site["bonds"]["AAresSeq"]):
            _np.testing.assert_equal(str(self.geom.top.residue(ii)),resi)
            _np.testing.assert_equal(str(self.geom.top.residue(jj)),resj)

    def test_the_idxs_work_w_frags(self):
        site = mdciao.sites.load(self.GDP_json)
        ctc_idxs, __ = mdciao.sites.sites_to_res_pairs([site], self.geom.top,
                                                       fragments=self.fragments)
        for (ii,jj), (resi,resj) in zip(ctc_idxs,site["bonds"]["AAresSeq"]):
            _np.testing.assert_equal(str(self.geom.top.residue(ii)),resi)
            _np.testing.assert_equal(str(self.geom.top.residue(jj)),resj)


if __name__ == '__main__':
    unittest.main()
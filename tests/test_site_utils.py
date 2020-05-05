import mdtraj as md
import unittest
from filenames import filenames
import pytest

import numpy as _np
import mdtraj as _md

test_filenames = filenames()
import mdciao.site_utils

class Test_sitefile2site(unittest.TestCase):

    def test_runs(self):
        site = mdciao.site_utils.sitefile2sitedict(test_filenames.GDP_json)
        _np.testing.assert_equal(site["name"],"GDP")
        _np.testing.assert_equal(site["n_bonds"],5)
        _np.testing.assert_equal(len(site["bonds"]["AAresSeq"]),5)
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][0], ['GDP396', 'ARG201'])
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][1], ['GDP396', 'THR204'])
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][2], ['GDP396', 'VAL202'])
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][3], ['THR204', 'SER54'])
        _np.testing.assert_array_equal(site["bonds"]["AAresSeq"][4], ['MG397', 'THR204'])

    def test_runs_with_name(self):
        site = mdciao.site_utils.sitefile2sitedict(test_filenames.GDP_name_json)
        _np.testing.assert_equal(site["name"], "siteGDP")

class Test__sites_to_AAresSeqdict(unittest.TestCase):
    def setUp(self):
        self.GDP_json = test_filenames.GDP_json
        self.geom = _md.load(test_filenames.prot1_pdb)
        self.fragments = mdciao.fragments.get_fragments(self.geom.top)
    def test_works(self):
        site = mdciao.site_utils.sitefile2sitedict(self.GDP_json)
        AAdict = mdciao.site_utils.sites_to_AAresSeqdict([site], self.geom.top, self.fragments)
        for ibond in site["bonds"]["AAresSeq"]:
            assert ibond[0] in AAdict.keys() and ibond[1] in AAdict.keys()

    def test_raises_if_not_found(self):
        site = mdciao.site_utils.sitefile2sitedict(test_filenames.GDP_name_json)
        with pytest.raises(ValueError):
            mdciao.site_utils.sites_to_AAresSeqdict([site], self.geom.top, self.fragments)

    def test_does_not_raise_if_not_found(self):
        site = mdciao.site_utils.sitefile2sitedict(test_filenames.GDP_name_json)
        mdciao.site_utils.sites_to_AAresSeqdict([site], self.geom.top, self.fragments,
                                                raise_if_not_found=False)

class Test_sites_to_ctc_idxs(unittest.TestCase):
    def setUp(self):
        self.GDP_json = test_filenames.GDP_json
        self.geom = _md.load(test_filenames.prot1_pdb)
        self.fragments = mdciao.fragments.get_fragments(self.geom.top)

    def test_the_idxs_work_no_frags(self):
        site = mdciao.site_utils.sitefile2sitedict(self.GDP_json)
        ctc_idxs, __ = mdciao.site_utils.sites_to_ctc_idxs([site], self.geom.top)
        for (ii,jj), (resi,resj) in zip(ctc_idxs,site["bonds"]["AAresSeq"]):
            _np.testing.assert_equal(str(self.geom.top.residue(ii)),resi)
            _np.testing.assert_equal(str(self.geom.top.residue(jj)),resj)

    def test_the_idxs_work_w_frags(self):
        site = mdciao.site_utils.sitefile2sitedict(self.GDP_json)
        ctc_idxs, __ = mdciao.site_utils.sites_to_ctc_idxs([site], self.geom.top,
                                                           fragments=self.fragments)
        for (ii,jj), (resi,resj) in zip(ctc_idxs,site["bonds"]["AAresSeq"]):
            _np.testing.assert_equal(str(self.geom.top.residue(ii)),resi)
            _np.testing.assert_equal(str(self.geom.top.residue(jj)),resj)


if __name__ == '__main__':
    unittest.main()
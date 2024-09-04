import unittest
from mdciao.examples import filenames as test_filenames
from mdciao.sites import siteIO

import numpy as _np
import mdtraj as _md


import mdciao

class Test_x2site(unittest.TestCase):

    def test_runs(self):
        site = mdciao.sites.x2site(test_filenames.GDP_json)
        _np.testing.assert_equal(site["name"],"GDP")
        _np.testing.assert_equal(site["n_pairs"],5)
        _np.testing.assert_equal(len(site["pairs"]["AAresSeq"]),5)
        _np.testing.assert_array_equal(site["pairs"]["AAresSeq"][0], ['GDP396', 'ARG201'])
        _np.testing.assert_array_equal(site["pairs"]["AAresSeq"][1], ['GDP396', 'THR204'])
        _np.testing.assert_array_equal(site["pairs"]["AAresSeq"][2], ['GDP396', 'VAL202'])
        _np.testing.assert_array_equal(site["pairs"]["AAresSeq"][3], ['THR204', 'SER54'])
        _np.testing.assert_array_equal(site["pairs"]["AAresSeq"][4], ['MG397', 'THR204'])

    def test_runs_w_residx(self):
        site = mdciao.sites.x2site({"name": "interesting contacts",
                                    "pairs": {"residx": [
                                        "353-972",
                                        "340-956",
                                        "343-956",
                                        "344-956",
                                        "340-959",
                                        "343-865"
                                    ]}})
        self.assertDictEqual(site, {"pairs": {"residx": [
            [353,972],
            [340,956],
            [343,956],
            [344,956],
            [340,959],
            [343,865]
        ]}, "name": "interesting contacts",
        "n_pairs":6}
                             )

    def test_runs_w_residx_no_str(self):
        site = mdciao.sites.x2site({"pairs":
            {"residx": [
                [353, 972],
                [340, 956],
                [343, 956],
                [344, 956],
                [340, 959],
                [343, 865]]},
            "name": "interesting contacts",
            "n_pairs": 6})
        #it's the same dict repeated, nothing should change

        self.assertDictEqual(site,{"pairs":
            {"residx": [
                [353, 972],
                [340, 956],
                [343, 956],
                [344, 956],
                [340, 959],
                [343, 865]]},
            "name": "interesting contacts",
            "n_pairs": 6})

    def test_runs_w_residx_str_list(self):
        site = mdciao.sites.x2site({"pairs":
            {"residx": [
                ["353", "972"],
                ["340", "956"],
                ["343", "956"],
                ["344", "956"],
                ["340", "959"],
                ["343", "865"]]},
            "name": "interesting contacts",
            "n_pairs": 6})

        print(site)

    def test_raises(self):
        with self.assertRaises(KeyError):
             mdciao.sites.x2site({"fonds":
                {"residx": [
                    [353, 972],
                    [340, 956],
                    [343, 956],
                    [344, 956],
                    [340, 959],
                    [343, 865]]},
                "name": "interesting contacts",
                "n_pairs": 6})


    def test_runs_w_dict(self):
        site = mdciao.sites.x2site(test_filenames.GDP_json)
        _np.testing.assert_equal(site["name"], "GDP")
        _np.testing.assert_equal(site["n_pairs"], 5)
        _np.testing.assert_equal(len(site["pairs"]["AAresSeq"]), 5)
        _np.testing.assert_array_equal(site["pairs"]["AAresSeq"][0], ['GDP396', 'ARG201'])
        _np.testing.assert_array_equal(site["pairs"]["AAresSeq"][1], ['GDP396', 'THR204'])
        _np.testing.assert_array_equal(site["pairs"]["AAresSeq"][2], ['GDP396', 'VAL202'])
        _np.testing.assert_array_equal(site["pairs"]["AAresSeq"][3], ['THR204', 'SER54'])
        _np.testing.assert_array_equal(site["pairs"]["AAresSeq"][4], ['MG397', 'THR204'])

    def test_runs_w_dict_of_lists_of_res(self):
        site = mdciao.sites.x2site(test_filenames.GDP_json)
        #Run it again, should yield the same dict
        self.assertDictEqual(site, mdciao.sites.x2site(site))


    def test_runs_w_residx_and_comment(self):
        site = mdciao.sites.x2site({"name": "interesting contacts",
                                    "pairs": {"residx": [
                                        "353-972",
                                        "340-956",
                                        "# 343-956",
                                        "344-956",
                                        "340-959",
                                        "343-865"
                                    ]}})
        self.assertDictEqual(site, {"pairs": {"residx": [
            [353, 972],
            [340, 956],
            # [343,956],
            [344, 956],
            [340, 959],
            [343, 865]
        ]}, "name": "interesting contacts",
            "n_pairs": 5})

    def test_raises_wo_name(self):
        site = mdciao.sites.x2site(test_filenames.GDP_name_json)
        site.pop("name")
        with _np.testing.assert_raises(ValueError):
            mdciao.sites.x2site(site)

    def test_runs_with_name(self):
        site = mdciao.sites.x2site(test_filenames.GDP_name_json)
        _np.testing.assert_equal(site["name"], "siteGDP")

    def test_runs_with_dat(self):
        site = mdciao.sites.x2site(test_filenames.tip_dat)
        site_test = mdciao.sites.x2site(test_filenames.tip_json)
        self.assertDictEqual(site["pairs"], site_test["pairs"])
        self.assertEqual(site["name"],"tip.json as plain ascii")
        self.assertEqual(site["n_pairs"],site_test["n_pairs"])

class Test_dat2site(unittest.TestCase):

    def test_works_AAresSeq(self):
        site = siteIO.dat2site(test_filenames.tip_dat)
        self.assertDictEqual(site, {"pairs": {"AAresSeq": [
            "L394-K270",
            "D381-Q229",
            "Q384-Q229",
            "R385-Q229",
            "D381-K232",
            "Q384-I135"
        ]}, "name":"tip.json as plain ascii"}
                             )
    def test_works_residx(self):
        site = siteIO.dat2site(test_filenames.tip_residx_dat,fmt="residx")
        print(site)
        self.assertDictEqual(site, {"pairs": {"residx": [
            "353-972",
            "340-956",
            "343-956",
            "344-956",
            "340-959",
            "343-865"
        ]}, "name": "tip.json as plain ascii with residx"}
                             )
    def test_works_consensus(self):
        site = siteIO.dat2site(test_filenames.tip_consensus_dat,fmt="consensus")
        print(site)
        self.assertDictEqual(site, {"pairs": {"consensus": [
            "G.H5.26-6.32x32",
            "G.H5.13-5.68x68",
            "G.H5.16-5.68x68",
            "G.H5.17-5.68x68",
            "G.H5.13-5.71x71",
            "G.H5.16-3.54x54"
        ]}, "name": "tip.json as plain ascii with consensus"}
                             )
    def test_Valuerror(self):
        with self.assertRaises(ValueError):
            siteIO.dat2site(test_filenames.tip_residx_dat, fmt="AAresSeq")
        with self.assertRaises(ValueError):
            siteIO.dat2site(test_filenames.tip_dat, fmt="residx")


class Test_site2str(unittest.TestCase):

    def test_sitefile(self):
        mdciao.sites.site2str(test_filenames.GDP_json)
    def test_sitedict(self):
        site = mdciao.sites.x2site(test_filenames.GDP_json)
        mdciao.sites.site2str(site)
    def test_what(self):
        with _np.testing.assert_raises(ValueError):
            mdciao.sites.site2str([1])

class Test_sites_to_res_pairs(unittest.TestCase):
    def setUp(self):
        self.GDP_json = test_filenames.GDP_json
        self.geom = _md.load(test_filenames.actor_pdb)
        self.fragments = mdciao.fragments.get_fragments(self.geom.top)
        self.GPCR_map = mdciao.examples.GPCRLabeler_ardb2_human().top2labels(self.geom.top)
        self.CGN_map = mdciao.examples.CGNLabeler_gnas2_human().top2labels(self.geom.top)
    def test_the_idxs_work_no_frags(self):
        site = mdciao.sites.x2site(self.GDP_json)
        ctc_idxs, __ = mdciao.sites.sites_to_res_pairs([site], self.geom.top)
        for (ii,jj), (resi,resj) in zip(ctc_idxs,site["pairs"]["AAresSeq"]):
            _np.testing.assert_equal(str(self.geom.top.residue(ii)),resi)
            _np.testing.assert_equal(str(self.geom.top.residue(jj)),resj)

    def test_the_idxs_work_w_frags(self):
        site = mdciao.sites.x2site(self.GDP_json)
        ctc_idxs, __ = mdciao.sites.sites_to_res_pairs([site], self.geom.top,
                                                       fragments=self.fragments)
        for (ii,jj), (resi,resj) in zip(ctc_idxs,site["pairs"]["AAresSeq"]):
            _np.testing.assert_equal(str(self.geom.top.residue(ii)),resi)
            _np.testing.assert_equal(str(self.geom.top.residue(jj)),resj)

    def test_Nones_get_differentiated_and_added(self):
        # Even though the first two ones yield both (None,None) pairs, they
        # will get added to the list. The third one will be recognized
        # as a seen None,None
        ctc_idxs, site_maps = mdciao.sites.sites_to_res_pairs([{"name": "bogus1",       "pairs": {"AAresSeq": ["AAA1-AAA2"]}},
                                                               {"name": "bogus2",       "pairs": {"AAresSeq": ["AAA3-AAA4"]}},
                                                               {"name": "bogus1_repeat","pairs": {"AAresSeq": ["AAA1-AAA2"]}},
                                                               ], self.geom.top)
        _np.testing.assert_array_equal(ctc_idxs, [[None,None],
                                                  [None,None]])
        self.assertListEqual(site_maps, [[0],[1],[0]])

    def test_consensus(self):
        ctc_idxs, site_maps = mdciao.sites.sites_to_res_pairs([{"name": "interesting contacts",
                                                                "pairs": {"consensus": [
                                                                    "G.H5.26-6.32x32"]}}], self.geom.top,
                                                              consensus_maps={"GPCR": self.GPCR_map,
                                                                              "CGN": self.CGN_map})
        _np.testing.assert_array_equal(ctc_idxs, [[659, 209]])
        _np.testing.assert_array_equal(site_maps, [[0]])
    def test_consensus_fails_no_consensus_passed(self):
        with self.assertRaises(ValueError):
            mdciao.sites.sites_to_res_pairs([{"name":"interesting contacts",
                                              "pairs": {"consensus": [
                                                  "G.H5.26-6.32x32"]}}], self.geom.top)

    def test_consensus_no_label_present(self):
        ctc_idxs, site_maps = mdciao.sites.sites_to_res_pairs([{"name":"interesting contacts",
                                              "pairs": {"consensus": [
                                                  "G.H5.26-6.32x32"]}}], self.geom.top,
                                            consensus_maps={"GPCR":self.GPCR_map})
        _np.testing.assert_array_equal(ctc_idxs, [[None, 209]])
        _np.testing.assert_array_equal(site_maps, [[0]])

class Test_discard_empty_sites(unittest.TestCase):

    def setUp(self):
        self.site_list = [
            {'name': 'site0', 'pairs': {'AAresSeq': ['ALA20-ALA21',  # ALA20 doesn't exist
                                                     'GLU101-GLU122']}},  # both exist

            {'name': 'site1', 'pairs': {'AAresSeq': ['GLU31-ALA20',  # GLU31 and ALA20 don't exist
                                                     'GLU17-GLU12']}},  # both exist

            {'name': 'site2', 'pairs': {'AAresSeq': ['GLN101-ALA122']}},  # GLN101 doesn't exist

            {'name': 'site3', 'pairs': {'AAresSeq': ['GLU101-GLU122']}}]  # both exist, but was seen before
        self.top = _md.load(test_filenames.top_pdb).top

        self.ctc_idxs, self.site_maps = mdciao.sites.sites_to_res_pairs(self.site_list, self.top)

        # This has to work otherwise the below tests don't make sense
        _np.testing.assert_array_equal(self.ctc_idxs, [[None, 374],
                                                  [69, 852],
                                                  [None, None],
                                                  [709, 365],
                                                  [None, None]])
        self.assertListEqual(self.site_maps, [[0, 1],
                                              [2, 3],
                                              [4],
                                              [1]])

    def test_just_works(self):

        new_ctc_idxs, new_site_maps, new_sites, discarded = mdciao.sites.discard_empty_sites(self.ctc_idxs, self.site_maps, self.site_list)
        _np.testing.assert_array_equal(new_ctc_idxs, [
                                                # [None, 374],
                                                [69, 852],
                                                # [None, None],
                                                [709, 365],
                                                # [None, None]
                                                ])
        self.assertListEqual(new_site_maps, [[0],
                                         [1],
                                         [0],
                                         ])
        self.assertListEqual(discarded["partial"], [0,1])
        self.assertListEqual(discarded["full"],[2])
        assert len(new_site_maps)==3
        self.assertDictEqual(new_sites[0], {"name": "site0", "pairs": {"residx": [[69, 852]]}, "n_pairs": 1})
        self.assertDictEqual(new_sites[1], {"name": "site1", "pairs": {"residx": [[709, 365]]}, "n_pairs": 1})
        self.assertDictEqual(new_sites[2], {"name": "site3", "pairs": {"residx": [[69, 852]]}, "n_pairs": 1})

    def test_just_works_full(self):
        new_ctc_idxs, new_site_maps, new_sites, discarded = mdciao.sites.discard_empty_sites(self.ctc_idxs, self.site_maps,
                                                                                             self.site_list, allow_partial_sites=False)
        _np.testing.assert_array_equal(new_ctc_idxs, [
            # [None, 374],
            [69, 852],
            # [None, None],
            #[709, 365],
            # [None, None]
        ])
        self.assertListEqual(new_site_maps, [[0]])
        self.assertListEqual(discarded["partial"], [])
        self.assertListEqual(discarded["full"], [0,1,2])
        assert len(new_site_maps)==1
        self.assertDictEqual(new_sites[0], {"name": "site3", "pairs": {"residx": [[69, 852]]}, "n_pairs": 1})

    def test_all_empty(self):
        site_list = self.site_list[:2]  # The first two sites have each one nonexistent pair
        ctc_idxs, site_maps = mdciao.sites.sites_to_res_pairs(site_list, self.top)
        new_ctc_idxs, new_site_maps, new_sites, discarded = mdciao.sites.discard_empty_sites(ctc_idxs,
                                               site_maps,
                                               site_list[:2],
                                               allow_partial_sites=False, #both sites will be discarded
                                               )
        assert len(new_ctc_idxs)==len(new_site_maps)==len(new_sites)==0
        assert len(discarded["partial"])==0
        assert discarded["full"]==[0,1]

if __name__ == '__main__':
    unittest.main()
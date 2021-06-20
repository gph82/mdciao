from mdciao.flare import _utils
import numpy as np
from unittest import TestCase
from itertools import combinations
from unittest import skip
from mdciao.plots.plots import _colorstring
from mdciao.examples._filenames import filenames
import mdtraj as md
from matplotlib import pyplot as plt
from mdciao.flare import circle_plot_residues

filenames = filenames()


class TestAngulateSegments(TestCase):

    def test_works(self):
        angles = _utils.regspace_angles(10, circle=360)

        np.testing.assert_array_equal(angles,
                                      [0., -36., -72., -108., -144., -180., -216., -252., -288., -324.])

    def test_anti_clockwise(self):
        angles = _utils.regspace_angles(10, circle=360, clockwise=False)

        np.testing.assert_array_equal(angles,
                                      [0, 36, 72, 108, 144, 180, 216, 252, 288, 324])


class TestFragmentSelectionParser(TestCase):

    def setUp(self):
        self.fragments = [[0, 1], [2, 3], [4, 5]]
        self.all_pairs = np.vstack(list(combinations(np.arange(6), 2)))

    def test_works(self):
        condition = _utils.fragment_selection_parser(self.fragments)
        assert all([condition(pair) for pair in self.all_pairs])

    def test_show_only(self):
        condition = _utils.fragment_selection_parser(self.fragments, only_show_fragments=[2])
        shown_pairs = [pair for pair in self.all_pairs if condition(pair)]
        np.testing.assert_array_equal([[0, 4],
                                       [0, 5],
                                       [1, 4],
                                       [1, 5],
                                       [2, 4],
                                       [2, 5],
                                       [3, 4],
                                       [3, 5],
                                       [4, 5]],
                                      shown_pairs)

    def test_hide(self):
        condition = _utils.fragment_selection_parser(self.fragments, hide_fragments=[0])
        shown_pairs = [pair for pair in self.all_pairs if condition(pair)]
        np.testing.assert_array_equal([[2, 3],
                                       [2, 4],
                                       [2, 5],
                                       [3, 4],
                                       [3, 5],
                                       [4, 5]],
                                      shown_pairs)

    def test_raises(self):
        with np.testing.assert_raises(ValueError):
            _utils.fragment_selection_parser(self.fragments,
                                             hide_fragments=[0], only_show_fragments=[1])


class TestCartify(TestCase):

    def test_works(self):
        XY, angles = _utils.cartify_fragments([[0, 1, 2, 4]],
                                              return_angles=True)

        np.testing.assert_array_almost_equal(angles / np.pi * 180, [0, -90, -180, -270])
        np.testing.assert_array_almost_equal(XY, [[1, 0],
                                                  [0, -1],
                                                  [-1, 0],
                                                  [0, 1],
                                                  ])

    def test_padding_between(self):
        XY = _utils.cartify_fragments([[0], [1]],
                                      padding=[0, 1, 0],  # The first and 3rd positions should be empty
                                      )
        np.testing.assert_array_almost_equal(XY,
                                             [
                                                 [0, -1],
                                                 [0, 1],
                                             ])

    def test_padding_initial(self):
        XY = _utils.cartify_fragments([[0, 1]],
                                      padding=[2, 0, 0],
                                      )
        np.testing.assert_array_almost_equal(XY, [[-1, 0],
                                                  [0, 1],
                                                  ])

    def test_padding_final(self):
        XY = _utils.cartify_fragments([[0, 1]],
                                      padding=[0, 0, 2],
                                      )
        np.testing.assert_array_almost_equal(XY, [[1, 0],
                                                  [0, -1],
                                                  ])


class TestColors(TestCase):

    def setUp(self):
        self.ref_colors = _colorstring.split(",")
        self.fragments = [[0, 1], [2, 3], [4, 5]]

    def test_works_False(self):
        colorlist = _utils.col_list_from_input_and_fragments(False, self.fragments)
        np.testing.assert_array_equal(["gray"] * 6, colorlist)

    def test_works_True(self):
        colorlist = _utils.col_list_from_input_and_fragments(True, self.fragments)
        np.testing.assert_array_equal([self.ref_colors[0], self.ref_colors[0],
                                       self.ref_colors[1], self.ref_colors[1],
                                       self.ref_colors[2], self.ref_colors[2]],
                                      colorlist)

    def test_works_string(self):
        colorlist = _utils.col_list_from_input_and_fragments("salmon", self.fragments)
        np.testing.assert_array_equal(['salmon'] * 6,
                                      colorlist)

    def test_works_iterable(self):
        colorlist = _utils.col_list_from_input_and_fragments(["r", "g", "b"], self.fragments)
        np.testing.assert_array_equal(["r", "g", "b"],
                                      colorlist)

    def test_works_dict(self):
        colorlist = _utils.col_list_from_input_and_fragments({"frag1": "r",
                                                              "frag2": "g",
                                                              "frag3": "b"},
                                                             self.fragments)
        np.testing.assert_array_equal(["r", "r",
                                       "g", "g",
                                       "b", "b"],
                                      colorlist)

    def test_works_one_frag(self):
        colorlist = _utils.col_list_from_input_and_fragments(False, [0, 1, 2])
        np.testing.assert_array_equal(["gray"] * 3, colorlist)


class TestLambaCurve(TestCase):

    def setUp(self):
        self.fragments = [[0, 1],
                          [2, 3],
                          [4, 5],
                          [6, 7]]
        self.pairs = np.vstack(list(combinations(range(8), 2)))

    def test_works_all_pairs(self):
        ilambda = _utils.should_this_residue_pair_get_a_curve(self.fragments)
        assert all([ilambda(pair) for pair in self.pairs])

    def test_res_selection(self):
        res_selection = [1, 4]
        ilambda = _utils.should_this_residue_pair_get_a_curve(self.fragments,
                                                              select_residxs=res_selection)
        selected_pairs = np.vstack([pair for pair in self.pairs if ilambda(pair)])
        np.testing.assert_array_equal(selected_pairs, [[0, 1],  # bc 1
                                                       [0, 4],  # bc 4
                                                       [1, 2],  # bc 1
                                                       [1, 3],  #
                                                       [1, 4],
                                                       [1, 5],
                                                       [1, 6],
                                                       [1, 7],  # bc 1
                                                       [2, 4],  # bc 4
                                                       [3, 4],
                                                       [4, 5],
                                                       [4, 6],
                                                       [4, 7]
                                                       ])

    def test_frag_selection(self):
        # This is tested anyway in its own lambda but testing again here
        ilambda = _utils.should_this_residue_pair_get_a_curve(self.fragments,
                                                              mute_fragments=[0, 1])
        selected_pairs = np.vstack([pair for pair in self.pairs if ilambda(pair)])
        np.testing.assert_array_equal(selected_pairs, [[4, 5],
                                                       [4, 6],
                                                       [4, 7],
                                                       [5, 6],
                                                       [5, 7],
                                                       [6, 7]
                                                       ])

    def test_neighbor_selection(self):
        ilambda = _utils.should_this_residue_pair_get_a_curve(self.fragments,
                                                              exclude_neighbors=3)
        selected_pairs = np.vstack([pair for pair in self.pairs if ilambda(pair)])
        np.testing.assert_array_equal(selected_pairs, [[0, 4],
                                                       [0, 5],
                                                       [0, 6],
                                                       [0, 7],
                                                       [1, 5],
                                                       [1, 6],
                                                       [1, 7],
                                                       [2, 6],
                                                       [2, 7],
                                                       [3, 7],
                                                       ]
                                      )

    def test_neighbor_selection_w_top(self):
        top = md.load(filenames.actor_pdb).top  # All is one chain here
        ilambda = _utils.should_this_residue_pair_get_a_curve(self.fragments,
                                                              top=top,
                                                              exclude_neighbors=3)
        selected_pairs = np.vstack([pair for pair in self.pairs if ilambda(pair)])
        np.testing.assert_array_equal(selected_pairs, [[0, 4],
                                                       [0, 5],
                                                       [0, 6],
                                                       [0, 7],
                                                       [1, 5],
                                                       [1, 6],
                                                       [1, 7],
                                                       [2, 6],
                                                       [2, 7],
                                                       [3, 7],
                                                       ]
                                      )


class TestResidueLabels(TestCase):

    def test_works(self):
        plt.figure(figsize=(5, 5))
        iax = plt.gca()
        iax.set_aspect("equal")
        iax.set_xlim([-1, 1])
        iax.set_ylim([-1, 1])
        _utils.add_residue_labels(plt.gca(), [[1, 0],
                                              [0, -1],
                                              [-1, 0],
                                              [0, 1]],
                                  [15, 18, 21, 12],
                                  10,
                                  center=[0, 0])
        ifig = plt.gcf()
        ifig.tight_layout()
        # plt.savefig('test.png',bbox_inches="tight")

    def test_works_options(self):
        top = md.load(filenames.actor_pdb).top
        ifig, myax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
        iax = plt.gca()
        iax.set_aspect("equal")
        iax.set_xlim([-1, 1])
        iax.set_ylim([-1, 1])
        _utils.add_residue_labels(myax[0], [[1, 0],
                                            [0, -1],
                                            [-1, 0],
                                            [0, 1]],
                                  [15, 18, 21, 12],
                                  10,
                                  center=[0, 0],
                                  top=top,
                                  colors=['b'] * 4)

        _utils.add_residue_labels(myax[1], [[1, 0],
                                            [0, -1],
                                            [-1, 0],
                                            [0, 1]],
                                  [15, 18, 21, 12],
                                  10,
                                  center=[0, 0],
                                  top=top,
                                  highlight_residxs=[15],
                                  replacement_labels={18: "6 o'clock"},
                                  shortenAAs=False)

        ifig.tight_layout()
        # plt.savefig('test.png',bbox_inches="tight")


class TestFragmentLabels(TestCase):

    def test_works(self):
        plt.figure(figsize=(5, 5))
        iax = plt.gca()
        iax.set_aspect("equal")
        iax.set_xlim([-1, 1])
        iax.set_ylim([-1, 1])
        fragments = [np.arange(10), np.arange(10, 20)]
        _, _, plattrb = circle_plot_residues(fragments,
                                             iax=iax,
                                             padding=[0, 0, 0])

        _utils.add_fragment_labels(fragments, ["frag_10_20", "frag_30_40"],
                                   iax,
                                   r=plattrb["r"])
        ifig = plt.gcf()
        ifig.tight_layout()
        # plt.savefig('test.png',bbox_inches="tight")
        plt.close("all")

    def _test_works_options(self):
        top = md.load(filenames.actor_pdb).top
        ifig, myax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
        iax = plt.gca()
        iax.set_aspect("equal")
        iax.set_xlim([-1, 1])
        iax.set_ylim([-1, 1])
        _utils.add_residue_labels(myax[0], [[1, 0],
                                            [0, -1],
                                            [-1, 0],
                                            [0, 1]],
                                  [0,
                                   -np.pi / 2,
                                   -np.pi,
                                   -np.pi * 1.5],
                                  [15, 18, 21, 12],
                                  10,
                                  top=top,
                                  colors=['b'] * 4)

        _utils.add_residue_labels(myax[1], [[1, 0],
                                            [0, -1],
                                            [-1, 0],
                                            [0, 1]],
                                  [0,
                                   -np.pi / 2,
                                   -np.pi,
                                   -np.pi * 1.5],
                                  [15, 18, 21, 12],
                                  10,
                                  top=top,
                                  highlight_residxs=[15],
                                  replacement_labels={18: "6 o'clock"},
                                  shortenAAs=False)

        ifig.tight_layout()
        # plt.savefig('test.png',bbox_inches="tight")


class Testvalue2pos(TestCase):
    def test_works(self):
        x = [10, 20, 3, 5, 7, 1000, 0]

        v2p = _utils.value2position_map(x)
        assert 0 == v2p[10]
        assert 1 == v2p[20]
        assert 2 == v2p[3]
        assert 3 == v2p[5]
        assert 4 == v2p[7]
        assert 5 == v2p[1000]
        assert 6 == v2p[0]


class TestMyBezier(TestCase):

    def test_create(self):
        mybz = _utils.create_flare_bezier_2([[0, -1], [1, 0]], [0, 0])
        assert isinstance(mybz, _utils.my_BZCURVE)

    def test_plot(self):
        mybz = _utils.create_flare_bezier_2(np.array([[0, -1], [1, 0]]), [0, 0])
        mybz.plot(50)


class Test_parse_residue_and_fragments(TestCase):

    def setUp(self):
        self.res_idx_pairs = np.array([[1, 2],
                                       [1, 3],
                                       [1, 4],
                                       [2, 4],
                                       [10, 11],
                                       [10, 12],
                                       [10, 15]])

    def test_works(self):
        res_idxs_as_fragments = _utils._parse_residue_and_fragments(self.res_idx_pairs)
        np.testing.assert_equal(len(res_idxs_as_fragments), 1)
        np.testing.assert_array_equal(res_idxs_as_fragments, [np.arange(0, 16)])

    def test_sparse_True(self):
        res_idxs_as_fragments = _utils._parse_residue_and_fragments(self.res_idx_pairs, sparse=True)
        np.testing.assert_equal(len(res_idxs_as_fragments), 1)
        np.testing.assert_array_equal(res_idxs_as_fragments, [[1, 2, 3, 4, 10, 11, 12, 15]])

    def test_sparse_value(self):
        res_idxs_as_fragments = _utils._parse_residue_and_fragments(self.res_idx_pairs, sparse=[1, 10, 20])
        np.testing.assert_equal(len(res_idxs_as_fragments), 1)
        np.testing.assert_array_equal(res_idxs_as_fragments, [[1, 10, 20]])

    def test_fragments(self):
        fragments = [[0],
                     [1, 2, 3, 4, 5],
                     [6, 7, 8, 9],
                     [10, 11, 12, 13, 14, 15],
                     [20, 21, 50]]
        res_idxs_as_fragments = _utils._parse_residue_and_fragments(self.res_idx_pairs, fragments=fragments)
        np.testing.assert_equal(len(res_idxs_as_fragments), len(fragments))
        np.testing.assert_array_equal(res_idxs_as_fragments, fragments)

    def test_fragments_sparse(self):
        fragments = [[0],
                     [1, 2, 3, 4, 5],
                     [6, 7, 8, 9],
                     [10, 11, 12, 13, 14, 15],
                     [20, 21, 50]]
        res_idxs_as_fragments = _utils._parse_residue_and_fragments(self.res_idx_pairs, sparse=True,
                                                                    fragments=fragments)
        np.testing.assert_equal(len(res_idxs_as_fragments), 2)
        np.testing.assert_array_equal(res_idxs_as_fragments, [[1, 2, 3, 4], [10, 11, 12, 15]])

    def test_fragments_sparse_value(self):
        fragments = [[0],
                     [1, 2, 3, 4, 5],
                     [6, 7, 8, 9],
                     [10, 11, 12, 13, 14, 15],
                     [20, 21, 50]]
        res_idxs_as_fragments = _utils._parse_residue_and_fragments(self.res_idx_pairs,
                                                                    sparse=[1, 10, 20],
                                                                    fragments=fragments)
        np.testing.assert_equal(len(res_idxs_as_fragments), 3)
        np.testing.assert_array_equal(res_idxs_as_fragments, [[1], [10], [20]])


class Test_Aura(TestCase):

    def test_works(self):
        iax, xy, cpr_dict = circle_plot_residues([np.arange(50),
                                                  np.arange(50, 100)],
                                                 )
        _utils.add_aura(xy, np.mod(np.arange(len(xy)), 3), iax, r=cpr_dict["r"])
        # iax.figure.savefig("test.png")

    def test_works_w_options(self):
        iax, xy, cpr_dict = circle_plot_residues([np.arange(50),
                                                  np.arange(50, 100)],
                                                 )
        _utils.add_aura(xy, np.mod(np.arange(len(xy)), 3), iax, r=cpr_dict["r"],
                        lines=False)
        # iax.figure.savefig("test.png")

    def test_raises(self):
        iax, xy, cpr_dict = circle_plot_residues([np.arange(50),
                                                  np.arange(50, 100)],
                                                 )
        with self.assertRaises(NotImplementedError):
            _utils.add_aura(xy, -np.mod(np.arange(len(xy)), 3), iax, r=cpr_dict["r"],
                            lines=True)


class TestGetSetFonts(TestCase):

    def test_get(self):
        iax, _, _ = circle_plot_residues([np.arange(20)])

        fs = _utils.fontsize_get(iax)
        self.assertSequenceEqual(["n_polygons", "other"], list(fs.keys()))
        assert len(fs["other"]) == 0
        assert len(fs["n_polygons"]) == 1
        print(np.unique([tt.get_fontsize() for tt in iax.texts]))
        self.assertSequenceEqual(fs["n_polygons"], np.unique([np.round(tt.get_fontsize(), 2) for tt in iax.texts]))

    def test_set(self):
        iax1, _, _ = circle_plot_residues([np.arange(20)])
        iax2, _, _ = circle_plot_residues([np.arange(40)])

        fs1, fs2 = [_utils.fontsize_get(iax)["n_polygons"][0] for iax in [iax1, iax2]]
        assert fs1 != fs2

        _utils.fontsize_apply(iax1, iax2)
        new_fs2 = _utils.fontsize_get(iax2)["n_polygons"][0]
        assert fs1 == new_fs2

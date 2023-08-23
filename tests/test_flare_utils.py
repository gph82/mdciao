from mdciao.flare import _utils
import numpy as np
from unittest import TestCase
from itertools import combinations
from unittest import skip
from mdciao.plots.plots import _colorstring
from mdciao.examples import filenames as test_filenames
import mdtraj as md
from matplotlib import pyplot as plt, lines as mpllines
from mdciao.flare import circle_plot_residues


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

    def test_works_iterable_len_fragments(self):
        colorlist = _utils.col_list_from_input_and_fragments(["r", "g", "b"], self.fragments)
        np.testing.assert_array_equal(["r", "r",
                                       "g", "g",
                                       "b", "b"],
                                      colorlist)

    def test_works_iterable_len_residues(self):
        colorlist = _utils.col_list_from_input_and_fragments(["r", "r", "g", "g", "b", "b"], self.fragments)
        np.testing.assert_array_equal(["r", "r", "g", "g", "b", "b"],
                                      colorlist)

    def test_raises_wrong_length(self):
        with self.assertRaises(ValueError):
            _utils.col_list_from_input_and_fragments(["r", "r", "g", "g", "b"], self.fragments)

    def test_raises_wrong_type(self):
        with self.assertRaises(Exception):
            _utils.col_list_from_input_and_fragments(1, self.fragments)


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
        top = md.load(test_filenames.actor_pdb).top # All is one chain here
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
        top = md.load(test_filenames.actor_pdb).top
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
                                             ax=iax,
                                             padding=[0, 0, 0])

        _utils.add_fragment_labels(fragments, ["frag_10_20", "frag_30_40"],
                                   iax,
                                   r=plattrb["r"])
        ifig = plt.gcf()
        ifig.tight_layout()
        # plt.savefig('test.png',bbox_inches="tight")
        plt.close("all")

    def _test_works_options(self):
        top = md.load(test_filenames.actor_pdb).top
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
        assert isinstance(mybz.Line2D, mpllines.Line2D)


class Test_parse_residue_and_fragments(TestCase):

    def setUp(self):
        self.res_idx_pairs = np.array([[1, 2],
                                       [1, 3],
                                       [1, 4],
                                       [2, 4],
                                       [10, 11],
                                       [10, 12],
                                       [10, 15]])
        self.top = md.load(test_filenames.actor_pdb).top

    def test_works(self):
        res_idxs_as_fragments, anchors, muted = _utils._parse_residue_and_fragments(self.res_idx_pairs)
        np.testing.assert_equal(len(res_idxs_as_fragments), 1)
        np.testing.assert_array_equal(res_idxs_as_fragments, [np.arange(0, 16)])
        assert anchors is None
        assert muted is None

    def test_sparse_residues_True(self):
        res_idxs_as_fragments, _, _  = _utils._parse_residue_and_fragments(self.res_idx_pairs, sparse_residues=True)
        np.testing.assert_equal(len(res_idxs_as_fragments), 1)
        np.testing.assert_array_equal(res_idxs_as_fragments, [[1, 2, 3, 4, 10, 11, 12, 15]])

    def test_sparse_residues_value(self):
        res_idxs_as_fragments, _, _  = _utils._parse_residue_and_fragments(self.res_idx_pairs, sparse_residues=[1, 10, 20])
        np.testing.assert_equal(len(res_idxs_as_fragments), 1)
        np.testing.assert_array_equal(res_idxs_as_fragments, [[1, 10, 20]])

    def test_sparse_residues_top(self):
        res_idxs_as_fragments, _, _ = _utils._parse_residue_and_fragments(self.res_idx_pairs, top=self.top)
        np.testing.assert_equal(len(res_idxs_as_fragments), 1)
        np.testing.assert_array_equal(res_idxs_as_fragments, [np.arange(self.top.n_residues)])

    def test_fragments(self):
        fragments = [[0],
                     [1, 2, 3, 4, 5],
                     [6, 7, 8, 9],
                     [10, 11, 12, 13, 14, 15],
                     [20, 21, 50]]
        res_idxs_as_fragments, _, _  = _utils._parse_residue_and_fragments(self.res_idx_pairs, fragments=fragments)
        np.testing.assert_equal(len(res_idxs_as_fragments), len(fragments))
        self.assertListEqual(res_idxs_as_fragments, fragments)

    def test_fragments_sparse_residues(self):
        fragments = [[0],
                     [1, 2, 3, 4, 5],
                     [6, 7, 8, 9],
                     [10, 11, 12, 13, 14, 15],
                     [20, 21, 50]]
        res_idxs_as_fragments, _, _  = _utils._parse_residue_and_fragments(self.res_idx_pairs, sparse_residues=True,
                                                                    fragments=fragments)
        np.testing.assert_equal(len(res_idxs_as_fragments), 2)
        self.assertListEqual(res_idxs_as_fragments, [[1, 2, 3, 4], [10, 11, 12, 15]])

    def test_fragments_sparse_residues_w_anchor_and_mute(self):
        fragments = [[0],
                     [1, 2, 3, 4, 5],
                     [6, 7, 8, 9],
                     [10, 11, 12, 13, 14, 15],
                     [20, 21, 50]]
        res_idxs_as_fragments, new_anchors, new_mutes  = _utils._parse_residue_and_fragments(self.res_idx_pairs,
                                                                           sparse_residues=True,
                                                                           fragments=fragments,
                                                                           anchor_fragments=[1],
                                                                           mute_fragments=[3])
        np.testing.assert_equal(len(res_idxs_as_fragments), 2)
        np.testing.assert_array_equal(res_idxs_as_fragments, [[1, 2, 3, 4], [10, 11, 12, 15]])
        np.testing.assert_equal(new_anchors,[0]) #old was 1
        np.testing.assert_equal(new_mutes,[1])# old was 3

    def test_fragments_sparse_residues_w_anchor_and_mute_None(self):
        fragments = [[0],
                     [1, 2, 3, 4, 5],
                     [6, 7, 8, 9],
                     [10, 11, 12, 13, 14, 15],
                     [20, 21, 50]]
        res_idxs_as_fragments, new_anchors, new_mutes  = _utils._parse_residue_and_fragments(self.res_idx_pairs,
                                                                           sparse_residues=True,
                                                                           fragments=fragments,
                                                                           anchor_fragments=[0],
                                                                           mute_fragments=[3])
        np.testing.assert_equal(len(res_idxs_as_fragments), 2)
        np.testing.assert_array_equal(res_idxs_as_fragments, [[1, 2, 3, 4], [10, 11, 12, 15]])
        np.testing.assert_equal(new_anchors,None) #old was 0, frag 0 got deleted
        np.testing.assert_equal(new_mutes,[1])# old was 3

    def test_fragments_sparse_residues_value(self):
        fragments = [[0],
                     [1, 2, 3, 4, 5],
                     [6, 7, 8, 9],
                     [10, 11, 12, 13, 14, 15],
                     [20, 21, 50]]
        res_idxs_as_fragments = _utils._parse_residue_and_fragments(self.res_idx_pairs,
                                                                    sparse_residues=[1, 10, 20],
                                                                    fragments=fragments)[0]
        np.testing.assert_equal(len(res_idxs_as_fragments), 3)
        np.testing.assert_array_equal(res_idxs_as_fragments, [[1], [10], [20]])

    def test_fragments_sparse_fragments(self):
        fragments = [[0],
                     [1, 2, 3, 4, 5],
                     [6, 7, 8, 9],
                     [10, 11, 12, 13, 14, 15],
                     [20, 21, 50]]
        self.res_idx_pairs = np.array([[1, 2],
                                       [1, 3],
                                       [1, 4],
                                       [2, 4],
                                       [10, 11],
                                       [10, 12],
                                       [10, 15]])
        res_idxs_as_fragments = _utils._parse_residue_and_fragments(self.res_idx_pairs,
                                                                    fragments=fragments,
                                                                    sparse_fragments=True)[0]
        np.testing.assert_equal(len(res_idxs_as_fragments), 2)
        self.assertListEqual(res_idxs_as_fragments, [[1, 2, 3, 4, 5],
                                                     [10, 11, 12, 13, 14, 15]]
                             )

class Test_Aura(TestCase):

    def test_works(self):
        iax, xy, cpr_dict = circle_plot_residues([np.arange(50),
                                                  np.arange(50, 100)],
                                                 )
        _utils.add_aura(xy, np.mod(np.arange(len(xy)), 3), iax, r=cpr_dict["r"])
        # ax.figure.savefig("test.png")
        plt.close("all")

    def test_works_w_options(self):
        iax, xy, cpr_dict = circle_plot_residues([np.arange(50),
                                                  np.arange(50, 100)],
                                                 )
        _utils.add_aura(xy, np.mod(np.arange(len(xy)), 3), iax, r=cpr_dict["r"],
                        lines=False)
        # ax.figure.savefig("test.png")
        plt.close("all")
    def test_negative(self):
        iax, xy, cpr_dict = circle_plot_residues([np.arange(50),
                                                  np.arange(50, 100)],
                                                 )
        neg = np.ones(len(xy))
        neg[int(len(neg)/2):]=-1
        _utils.add_aura(xy, neg*np.mod(np.arange(len(xy)), 3), iax, r=cpr_dict["r"],
                        lines=True)
        #ax.figure.savefig("test.png")
        plt.close("all")


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

class Test_coarse_grain_freqs_by_frag(TestCase):

    def setUp(self):
        pass

    def test_works(self):
        freqs = [1, 2,
                 4, 5,
                 6]

        frags = [[0, 1], [2, 3], [4, 5]]

        pairs = [[0, 2], [1, 3],  # frags 0-1
                 [0, 4], [1, 5],  # frags 0-2
                 [2, 4]]  # frags 1-2

        ref_mat = np.zeros((3,3))
        ref_mat[0,1]=ref_mat[1,0]=3
        ref_mat[0,2]=ref_mat[2,0]=9
        ref_mat[1,2]=ref_mat[2,1]=6

        mat = _utils.coarse_grain_freqs_by_frag(freqs, pairs, frags)
        np.testing.assert_array_equal(mat, ref_mat)

    def test_raises_orphan(self):
        freqs = [1, 2,
                 4, 5,
                 6]

        frags = [[0, 1], [2, 3], [4, 5]]

        pairs = [[0, 2], [1, 3],  # frags 0-1
                 [0, 4], [1, 6],  # frags 0-2 and 0-None
                 [2, 4]]  # frags 1-2

        with np.testing.assert_raises(ValueError):
            _utils.coarse_grain_freqs_by_frag(freqs, pairs, frags)

    def test_ignores_orphan(self):
        freqs = [1, 2,
                 4, 5,
                 6]

        frags = [[0, 1], [2, 3], [4, 5]]

        pairs = [[0, 2], [1, 3],  # frags 0-1
                 [0, 4], [1, 6],  # frags 0-2 and 0-None
                 [2, 4]]  # frags 1-2

        ref_mat = np.zeros((3,3))
        ref_mat[0,1]=ref_mat[1,0]=3
        ref_mat[0,2]=ref_mat[2,0]=4 #since the second pair didn't get summed
        ref_mat[1,2]=ref_mat[2,1]=6
        mat = _utils.coarse_grain_freqs_by_frag(freqs, pairs, frags, check_if_subset=False)
        np.testing.assert_array_equal(mat, ref_mat)

class Test_sparsify_sym_matrix(TestCase):

    def test_just_works(self):
        mat = np.array([[1, 0, 2],
                        [0, 0, 0],
                        [2, 0, 3]])
        nm, nz = _utils.sparsify_sym_matrix(mat)
        np.testing.assert_array_equal(nz, [0,2])
        np.testing.assert_array_equal(nm, [[1,2],
                                           [2,3]])

class Test_sparsify_sym_matrix_by_row_sum(TestCase):

    def test_just_works(self):
        #First row.sum() is below eps, will be eliminated
        # Because matrix is sym, first column also goes
        mat = np.array([[0, 0, 0.01],
                        [0, .02, .02],
                        [0.01, .02, 0.0]])
        sparse_mat, kept_idxs, lost_value = _utils.sparsify_sym_matrix_by_row_sum(np.array(mat), eps=1e-2)
        np.testing.assert_array_equal([[0.02, 0.02],
                                       [0.02, 0.0]], sparse_mat)
        np.testing.assert_array_equal([1,2], kept_idxs)
        np.testing.assert_almost_equal(0.01, lost_value)
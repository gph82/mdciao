from mdciao.flare import flare
from mdciao.flare import sparse_freqs2flare
import numpy as np
from unittest import TestCase
from itertools import combinations
from unittest import skip

class TestAngulateSegments(TestCase):

    def test_works(self):
        angles = flare.regspace_angles(10, circle=360)

        np.testing.assert_array_equal(angles,
                                      [0., -36., -72., -108., -144., -180., -216., -252., -288., -324.])

    def test_anti_clockwise(self):
        angles = flare.regspace_angles(10, circle=360, clockwise=False)

        np.testing.assert_array_equal(angles,
                                      [0, 36, 72, 108, 144, 180, 216, 252, 288, 324])

    @skip("Not sure offset means")
    def test_offset(self):
        angles = flare.regspace_angles(10, circle=360,
                                       offset=45)


        print(angles)
        raise NotImplementedError

class TestFragmentSelectionParser(TestCase):

    def setUp(self):
        self.fragments = [[0,1],[2,3],[4,5]]
        self.all_pairs = np.vstack(list(combinations(np.arange(6),2)))


    def test_works(self):
        condition = flare.fragment_selection_parser(self.fragments)
        assert all([condition(pair) for pair in self.all_pairs])

    def test_show_only(self):
        condition = flare.fragment_selection_parser(self.fragments, only_show_fragments=[2])
        shown_pairs = [pair for pair in self.all_pairs if condition(pair)]
        np.testing.assert_array_equal([[0,4],
                                       [0,5],
                                       [1,4],
                                       [1,5],
                                       [2,4],
                                       [2,5],
                                       [3,4],
                                       [3,5],
                                       [4,5]],
                                      shown_pairs)

    def test_hide(self):
        condition = flare.fragment_selection_parser(self.fragments, hide_fragments=[0])
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
            flare.fragment_selection_parser(self.fragments,
                                            hide_fragments=[0], only_show_fragments=[1])


class TestCartify(TestCase):

    def test_works(self):
        XY, angles = flare.cartify_fragments([[0, 1, 2, 4]],
                                             return_angles=True)

        np.testing.assert_array_almost_equal(angles/np.pi*180, [0,-90,-180,-270])
        np.testing.assert_array_almost_equal(XY, [[ 1, 0],
                                                  [ 0,-1],
                                                  [-1, 0],
                                                  [ 0, 1],
                                                  ])

    def test_padding_between(self):
        XY = flare.cartify_fragments([[0], [1]],
                                     padding_between_fragments=1,  # The first and 3rd positions should be empty
                                     )
        np.testing.assert_array_almost_equal(XY,
                                             [
                                                 [0, -1],
                                                 [0,  1],
                                             ])

    def test_padding_initial(self):
        XY = flare.cartify_fragments([[0, 1]],
                                     padding_initial=2
                                     )
        np.testing.assert_array_almost_equal(XY, [[-1, 0],
                                                  [0, 1],
                                                  ])

    def test_padding_final(self):
        XY = flare.cartify_fragments([[0, 1]],
                                     padding_final=2
                                     )
        np.testing.assert_array_almost_equal(XY, [[ 1, 0],
                                                  [ 0,-1],
                                                  ])

class TestColors(TestCase):
    def setUp(self):
        from mdciao.plots.plots import _colorstring
        self.ref_colors = _colorstring.split(",")
        self.fragments = [[0,1],[2,3],[4,5]]

    def test_works_False(self):
        colorlist = flare.col_list_from_input_and_fragments(False, self.fragments, np.arange(10))
        np.testing.assert_array_equal(["tab:blue"]*10,colorlist)

    def test_works_True(self):
        colorlist = flare.col_list_from_input_and_fragments(True, self.fragments, np.arange(10))
        np.testing.assert_array_equal([self.ref_colors[0], self.ref_colors[0],
                                       self.ref_colors[1],self.ref_colors[1],
                                       self.ref_colors[2],self.ref_colors[2]]
                                        ,colorlist)


    def test_works_string(self):
        colorlist = flare.col_list_from_input_and_fragments("salmon", self.fragments, np.arange(10))
        np.testing.assert_array_equal(['salmon']*10,
                                      colorlist)

    def test_works_iterable(self):
        colorlist = flare.col_list_from_input_and_fragments(["r", "g"], self.fragments, np.arange(2))
        np.testing.assert_array_equal(["r","g"],
                                      colorlist)

    def test_works_dict(self):
        colorlist = flare.col_list_from_input_and_fragments({"frag1": "r",
                                                              "frag2": "g",
                                                              "frag3": "b"}, self.fragments, np.arange(2))
        np.testing.assert_array_equal(["r", "r",
                                       "g", "g",
                                       "b", "b"],
                                      colorlist)

class TestFlare(TestCase):

    def test_works(self):
        myax, __ = sparse_freqs2flare(np.reshape([1, 1, 1], (1, -1)),
                                      np.array([[0,1],[1,2],[2,3]]),
                                      np.arange(10),
                                      colors=["r"]*10,
                                      exclude_neighbors=0,
                                      angle_offset=0,
                                      )

        myax.figure.tight_layout()
        myax.figure.savefig("test.pdf")
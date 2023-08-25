from mdciao.flare import _utils
import numpy as np
from unittest import TestCase
from mdciao import flare
from itertools import combinations
from unittest import skip
from mdciao.plots.plots import _colorstring
from mdciao.examples import filenames as test_filenames
import mdtraj as md
from matplotlib import pyplot as plt



class TestFlare(TestCase):

    def test_works(self):
        myax, _, _ = flare.freqs2flare([1, 1, 1],
                                    np.array([[0, 1], [1, 2], [2, 3]]),
                                    exclude_neighbors=0,
                                    )

        myax.figure.tight_layout()
        #myax.figure.savefig("test.png")

    def test_options_only_curves(self):
        myfig = plt.figure(figsize=(5,5))
        iax = plt.gca()
        iax , _, _ = flare.freqs2flare(np.array([[1, 1, 1]]),
                                       np.array([[0, 1], [1, 2], [2, 3]]),
                                       fragments=[[0, 1], [2, 3]],
                                       exclude_neighbors=0,
                                       plot_curves_only=True,
                                       ax=iax,
                                       )

        iax.figure.tight_layout()
        #ax.figure.savefig("test.png")

    def test_options_sparse(self):
        myfig, myax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
        iax, _, _ = flare.freqs2flare(np.array([[1, 1, 1]]),
                                      np.array([[0, 1], [1, 2], [2, 3]]),
                                      fragments=[[0, 1], [2, 3], np.arange(4, 15)],
                                      exclude_neighbors=0,
                                      ax=myax[0]

                                      )
        iax, _, _ = flare.freqs2flare(np.array([[1, 1, 1]]),
                                      np.array([[0, 1], [1, 2], [2, 3]]),
                                      fragments=[[0, 1], [2, 3], np.arange(4, 15)],
                                      exclude_neighbors=0,
                                      ax=myax[1],
                                      sparse_residues=True,
                                      subplot=True,

                                      )

        iax.figure.tight_layout()
        #ax.figure.savefig("test.png")

    def test_coarse_grain(self):
        myfig = plt.figure(figsize=(5,5))
        iax = plt.gca()
        iax , _, _ = flare.freqs2flare(np.array([[1, 1, 1]]),
                                       np.array([[0, 1], [1, 2], [2, 3]]),
                                       fragments=[[0, 1], [2, 3]],
                                       coarse_grain=True,
                                       ax=iax,
                                       )
class TestCirclePlotResidues(TestCase):

    def test_works(self):
        iax, _, cpr_dict =  flare.circle_plot_residues([np.arange(50),
                                                        np.arange(50, 100)],
                                                       ss_array=["H"] * 100,
                                                       fragment_names=["A", "B"],
                                                       textlabels={0:"first",99:"last"},
                                                       aura=np.arange(100),
                                                       )
        for key in ["fragment_labels",
                    "dot_labels",
                    "dots",
                    "SS_labels"
                    ]:
            assert key in cpr_dict.keys()

        #ax.figure.savefig("test.png")

class TestAddBezier(TestCase):

    def test_works(self):
        iax, xy, cpr_dict = flare.circle_plot_residues([np.arange(5),
                                             np.arange(5, 10)],
                                            #ss_array=["H"] * 10,
                                            #fragment_names=["A", "B"]
                                             )
        pairs = [[0,5],
                 [5,6],
                 [9,3]
                 ]
        node_pairs = [(xy[ii],xy[jj]) for (ii,jj) in pairs]
        bzcurves =  flare.add_bezier_curves(iax,node_pairs
                                          )
        plt.close("all")
        #ax.figure.savefig("test.png")

class TestChord(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pairs = np.reshape(np.arange(40), (2, -1)).T
        print(cls.pairs)
        cls.frags = np.reshape(np.arange(49), (7, 7))
        # pairs go up to 40, frags up to 49, that frag will disappear
        print(cls.frags)
        cls.freqs = np.arange(20)
        cls.fragment_names = ["A", "B", "C", "D", "E", "F", "G"]
        print(cls.freqs)
        print(cls.freqs.sum())

    def test_just_works(self):

        iax, nonzeros, plotattrs = flare.freqs2chord(self.freqs,
                                                     self.pairs,
                                                     self.frags,
                                                     fragment_names=self.fragment_names)
        assert isinstance(iax, plt.Axes)
        assert len(iax.texts) == 6 + 6
        print(plotattrs["sigmas"])
        np.testing.assert_array_equal(nonzeros, [0, 1, 2, 3, 4, 5])
        for key in ["fragment_names", "fragment_labels", "r", "dots", "sigmas"]:
            assert key in plotattrs.keys()
        np.testing.assert_array_equal(plotattrs["sigmas"], [21, 70, 99, 28, 77, 85])
        assert plotattrs["dots"][0].radius > 0
        #ax.figure.savefig("test.ref.pdf")
        plt.close("all")

    def test_one_less_fragment(self):

        iax, nonzeros, plotattrs = flare.freqs2chord(self.freqs,
                                                     self.pairs,
                                                     self.frags,
                                                     fragment_names=self.fragment_names,
                                                     min_sigma=25,
                                                     )
        """
        This is how the coarse grain matrix looks like, 
        row nr. 0 wil be dropped
        [[ 0.  0.  0. 21.  0.  0.  0.]
         [ 0.  0.  0.  7. 63.  0.  0.]
         [ 0.  0.  0.  0. 14. 85.  0.]
         [21.  7.  0.  0.  0.  0.  0.]
         [ 0. 63. 14.  0.  0.  0.  0.]
         [ 0.  0. 85.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]]

        """
        #ax.figure.savefig("test.one_less.pdf")
        print(plotattrs["sigmas"])
        assert isinstance(iax, plt.Axes)
        assert len(iax.texts) == 5 + 5
        np.testing.assert_array_equal(nonzeros, [1, 2, 3, 4, 5])
        for key in ["fragment_names", "fragment_labels", "r", "dots", "sigmas"]:
            assert key in plotattrs.keys()
        np.testing.assert_array_equal(plotattrs["sigmas"], [# A isn't there bc 21 < 25
                                                            70, 99,
                                                            7, # note that frag-D has lost 21 cts from A
                                                            77, 85])
        assert plotattrs["dots"][0].radius > 0
        plt.close("all")


    def test_half_circle(self):

        iax, nonzeros, plotattrs = flare.freqs2chord(self.freqs,
                                                     self.pairs,
                                                     self.frags,
                                                     fragment_names=self.fragment_names,
                                                     normalize_to_sigma=self.freqs.sum()*2,
                                                     )
        #ax.figure.savefig("test.half.pdf")
        assert isinstance(iax, plt.Axes)
        assert len(iax.texts) == 6 + 6
        np.testing.assert_array_equal(nonzeros, [0, 1, 2, 3, 4, 5])
        for key in ["fragment_names", "fragment_labels", "r", "dots", "sigmas"]:
            assert key in plotattrs.keys()
        np.testing.assert_array_equal(plotattrs["sigmas"], [21, 70, 99, 28, 77, 85])
        assert plotattrs["dots"][0].radius > 0
        plt.close("all")
        print(plotattrs["sigmas"])

    def test_counter_clockwise(self):

        iax, nonzeros, plotattrs = flare.freqs2chord(self.freqs,
                                                     self.pairs,
                                                     self.frags,
                                                     fragment_names=self.fragment_names,
                                                     clockwise=False,
                                                     )
        #iax.figure.savefig("test.ccwise.pdf")
        assert isinstance(iax, plt.Axes)
        assert len(iax.texts) == 6 + 6
        np.testing.assert_array_equal(nonzeros, [0, 1, 2, 3, 4, 5])
        for key in ["fragment_names", "fragment_labels", "r", "dots"]:
            assert key in plotattrs.keys()
        np.testing.assert_array_equal(plotattrs["sigmas"], [21, 70, 99, 28, 77, 85])
        assert plotattrs["dots"][0].radius > 0
        plt.close("all")
        print(plotattrs["sigmas"])
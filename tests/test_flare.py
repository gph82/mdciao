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
                                    iax=iax,
                                    )

        iax.figure.tight_layout()
        #iax.figure.savefig("test.png")

    def test_options_sparse(self):
        myfig, myax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
        iax, _, _ = flare.freqs2flare(np.array([[1, 1, 1]]),
                                      np.array([[0, 1], [1, 2], [2, 3]]),
                                      fragments=[[0, 1], [2, 3], np.arange(4, 15)],
                                      exclude_neighbors=0,
                                      iax=myax[0]

                                      )
        iax, _, _ = flare.freqs2flare(np.array([[1, 1, 1]]),
                                      np.array([[0, 1], [1, 2], [2, 3]]),
                                      fragments=[[0, 1], [2, 3], np.arange(4, 15)],
                                      exclude_neighbors=0,
                                      iax=myax[1],
                                      sparse_residues=True,
                                      subplot=True,

                                      )

        iax.figure.tight_layout()
        #iax.figure.savefig("test.png")

    def test_coarse_grain(self):
        myfig = plt.figure(figsize=(5,5))
        iax = plt.gca()
        iax , _, _ = flare.freqs2flare(np.array([[1, 1, 1]]),
                                    np.array([[0, 1], [1, 2], [2, 3]]),
                                    fragments=[[0, 1], [2, 3]],
                                    coarse_grain=True,
                                    iax=iax,
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

        #iax.figure.savefig("test.png")

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
        #iax.figure.savefig("test.png")

class TestChord(TestCase):

    def test_just_works(self):
        pairs = np.reshape(np.arange(40), (2, -1)).T
        frags=np.reshape(np.arange(49), (7, 7))
        #pairs go up to 40, frags up to 49, that frag will disappear
        freqs = np.arange(20)
        fragment_names = ["A", "B", "C", "D", "E", "F", "G"]
        iax, nonzeros, plotattrs = flare.freqs2chord(freqs,
                                      pairs,
                                      frags,
                                      fragment_names=fragment_names)
        assert isinstance(iax, plt.Axes)
        assert len(iax.texts)==6+6
        np.testing.assert_array_equal(nonzeros,[0,1,2,3,4,5])
        for key in ["fragment_names","fragment_labels","r","dots"]:
            assert key in plotattrs.keys()
        assert plotattrs["dots"][0].radius>0
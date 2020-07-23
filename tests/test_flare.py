from mdciao.flare import _utils
import numpy as np
from unittest import TestCase
from mdciao import flare
from itertools import combinations
from unittest import skip
from mdciao.plots.plots import _colorstring
from mdciao.filenames import filenames
import mdtraj as md
from matplotlib import pyplot as plt



class TestFlare(TestCase):

    def test_works(self):
        myax, _ = flare.sparse_freqs2flare([1, 1, 1],
                                           np.array([[0, 1], [1, 2], [2, 3]]),
                                           exclude_neighbors=0,
                                           )

        myax.figure.tight_layout()
        #myax.figure.savefig("test.png")

    def test_options_only_curves(self):
        myfig = plt.figure(figsize=(5,5))
        iax = plt.gca()
        iax , _ = flare.sparse_freqs2flare(np.array([[1, 1, 1]]),
                                       np.array([[0, 1], [1, 2], [2, 3]]),
                                       fragments=[[0, 1], [2, 3]],
                                       exclude_neighbors=0,
                                        plot_curves_only=True,
                                           iax=iax,
                                       )

        iax.figure.tight_layout()
        #iax.figure.savefig("test.png")

    def test_options_sparse(self):
        myfig, myax = plt.subplots(1,2,sharex=True, sharey=True, figsize=(10,5))
        iax , _ = flare.sparse_freqs2flare(np.array([[1, 1, 1]]),
                                       np.array([[0, 1], [1, 2], [2, 3]]),
                                       fragments=[[0, 1], [2, 3],np.arange(4,15)],
                                       exclude_neighbors=0,
                                        iax=myax[0]

                                       )
        iax, _ = flare.sparse_freqs2flare(np.array([[1, 1, 1]]),
                                          np.array([[0, 1], [1, 2], [2, 3]]),
                                          fragments=[[0, 1], [2, 3], np.arange(4, 15)],
                                          exclude_neighbors=0,
                                          iax=myax[1],
                                          sparse=True

                                          )

        iax.figure.tight_layout()
        iax.figure.savefig("test.png")

class TestCirclePlotResidues(TestCase):

    def test_works(self):
        iax, _ =  flare.circle_plot_residues([np.arange(50),
                                              np.arange(50,100)],
                                             ss_array=["H"]*100,
                                             fragment_names=["A","B"])

        #iax.figure.savefig("test.png")

class TestAddBezier(TestCase):

    def test_works(self):
        iax, xy = flare.circle_plot_residues([np.arange(5),
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



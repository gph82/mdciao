import unittest
import numpy as _np
import pytest

from matplotlib import pyplot as _plt

from mdciao.contacts import ContactGroup, ContactPair

from mdciao import plots

from tempfile import TemporaryDirectory as _TDir
import os

from mdciao.filenames import filenames
test_filenames = filenames()
class TestPlotContactMatrix(unittest.TestCase):

    def test_plot_contact_matrix_runs_w_options(self):

        mat = _np.linspace(0,1,6).reshape(2,3)
        labels = [[1,2],["A","B","C"]]

        iax, ipix = plots.plot_contact_matrix(mat,
                                        labels,
                                        grid=True,
                                        colorbar=True,
                                        transpose=True
                                        )
        #iax.figure.savefig("test.png",bbox_inches="tight")
        _plt.close("all")

    def test_plot_contact_matrix_raises_labels(self):

        mat = _np.linspace(0,1,6).reshape(2,3)
        labels = [[1,2],["A","B"]]
        with pytest.raises(AssertionError):
            plots.plot_contact_matrix(mat,labels)

    def test_plot_contact_matrix_raises_range(self):

        mat = _np.linspace(0,2,6).reshape(2,3)
        labels = [[1,2],["A","B","C"]]
        with pytest.raises(AssertionError):
            plots.plot_contact_matrix(mat,labels)

class Test_plot_unified_freq_dicts(unittest.TestCase):

    def setUp(self):
        self.CG1_freqdict = {"0-1":1,  "0-2":.75, "0-3":.50, "4-6":.25}
        self.CG1_freqdict_shuffled = {"0-2":.75, "0-1":1, "0-3":.50, "4-6":.25}

        self.CG2_freqdict = {"0-1":.95, "0-2":.65, "0-3":.35, "4-6":0}


    def test_plot_unified_freq_dicts_minimal(self):
        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1":self.CG1_freqdict, "CG1copy":self.CG1_freqdict},
                                                  {"CG1":"r", "CG1copy":"b"})

        #myfig.savefig("1.test_full.png", bbox_inches="tight")
        _plt.close("all")

    def test_plot_unified_freq_dicts_remove_identities(self):
        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1": self.CG1_freqdict, "CG1copy": self.CG1_freqdict},
                                                  {"CG1": "r", "CG1copy": "b"},
                                                  remove_identities=True)
        #Check that the 0-1 contact has been removed
        #myfig.savefig("2.test_wo_ident.png", bbox_inches="tight")
        _plt.close("all")

    def test_plot_unified_freq_dicts_remove_identities_cutoff(self):
        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1": self.CG1_freqdict, "CG2": self.CG2_freqdict},
                                                  {"CG1": "r", "CG2": "b"},
                                                  remove_identities=True)
        #Check that the 0-1 contact is there
        #myfig.savefig("3.test_wo_ident_cutoff.ref.png", bbox_inches="tight")
        _plt.close("all")

        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1": self.CG1_freqdict, "CG2": self.CG2_freqdict},
                                                  {"CG1": "r", "CG2": "b"},
                                                  remove_identities=True,
                                                  identity_cutoff=.95)
        #Check that the 0-1 contact has been removed
        #myfig.savefig("4.test_wo_ident_cutoff.png", bbox_inches="tight")
        _plt.close("all")

    def test_plot_unified_freq_dicts_lower_cutoff_val(self):
        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1": self.CG1_freqdict, "CG2": self.CG2_freqdict},
                                                  {"CG1": "r", "CG2": "b"},
                                                  remove_identities=True,
                                                  identity_cutoff=.9,
                                                  lower_cutoff_val=.4
                                                  )
        #myfig.savefig("5.test_above_below_thres.png", bbox_inches="tight")
        _plt.close("all")

        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1": self.CG1_freqdict, "CG2": self.CG2_freqdict},
                                                  {"CG1": "r", "CG2": "b"},
                                                  remove_identities=True,
                                                  identity_cutoff=.95)
        #myfig.savefig("6.test_wo_ident_cutoff.png", bbox_inches="tight")
        _plt.close("all")


    def test_plot_unified_freq_dicts_order_std(self):
        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1": self.CG1_freqdict, "CG2": self.CG2_freqdict},
                                                  {"CG1": "r", "CG2": "b"},
                                                  sort_by="std",
                                                  lower_cutoff_val=0.0
                                                  )
        # Order should be inverted
        #myfig.savefig("7.test_order.std.png", bbox_inches="tight")
        _plt.close("all")

    def test_plot_unified_freq_dicts_order_keep(self):
        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1s": self.CG1_freqdict_shuffled, "CG2": self.CG2_freqdict},
                                                  {"CG1s": "r", "CG2": "b"},
                                                  sort_by="keep",
                                                  )
        # myfig.savefig("8.test_order.keep.png", bbox_inches="tight")
        _plt.close("all")




    def test_plot_unified_freq_dicts_remove_identities_vert(self):
        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1": self.CG1_freqdict, "CG2": self.CG2_freqdict},
                                                  {"CG1": "r", "CG2": "b"},
                                                  remove_identities=True,
                                                  vertical_plot=True)
        #myfig.savefig("9.test_wo_ident_vert.png", bbox_inches="tight")
        _plt.close("all")

    def test_plot_unified_freq_dicts_remove_identities_vert_order_by_std(self):
        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1": self.CG1_freqdict, "CG2": self.CG2_freqdict},
                                                  {"CG1": "r", "CG2": "b"},
                                                  sort_by="std",
                                                  #remove_identities=True,
                                                  lower_cutoff_val=0.0,
                                                  vertical_plot=True)
        #myfig.savefig("10.test_vert.std.png", bbox_inches="tight")
        _plt.close("all")

    def test_plot_unified_freq_dicts_ylim(self):
        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1":self.CG1_freqdict, "CG1copy":self.CG1_freqdict},
                                                  {"CG1":"r", "CG1copy":"b"}, ylim=2.25)

        #myfig.savefig("11.test_ylim.png", bbox_inches="tight")
        _plt.close("all")

    def test_plot_unified_freq_dicts_ax(self):
        _plt.figure()
        ax = _plt.gca()
        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1":self.CG1_freqdict, "CG1copy":self.CG1_freqdict},
                                                  {"CG1":"r", "CG1copy":"b"}, ylim=2.25, ax=ax)
        assert myax is ax
        _plt.close("all")

    def test_plot_just_one_dict(self):
        _plt.figure()
        ax = _plt.gca()
        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1": self.CG1_freqdict},
                                                  {"CG1": "r"})
        #myfig.savefig("12.test_just_one.png",bbox_inches="tight")
        _plt.close("all")

class Test_compare_groups_of_contacts(unittest.TestCase):

    def setUp(self):
        self.CG1 = ContactGroup([ContactPair([0,1],[[.1, .1]], [[0., 1.]]),
                                 ContactPair([0,2],[[.1, .2]], [[0., 1.]])])

        self.CG2 = ContactGroup([ContactPair([0,1],[[.1, .1, .1]], [[0., 1., 2.]]),
                                 ContactPair([0,3],[[.1, .2, .2]], [[0., 1., 2.]])])

    def test_just_works(self):
        myfig, freqs, __ = plots.compare_groups_of_contacts({"CG1":self.CG1,"CG2":self.CG2},
                                                      {"CG1":"r", "CG2":"b"},
                                                      ctc_cutoff_Ang=1.5)
        myfig.tight_layout()
        #myfig.savefig("1.test.png",bbox_inches="tight")
        _plt.close("all")


    def test_mutation(self):
        myfig, freqs, __ = plots.compare_groups_of_contacts({"CG1":self.CG1,"CG2":self.CG2},
                                                      {"CG1":"r", "CG2":"b"},
                                                      ctc_cutoff_Ang=1.5,
                                                      mutations_dict={"3":"2"})
        myfig.tight_layout()
        #myfig.savefig("2.test.png",bbox_inches="tight")
        _plt.close("all")

    def test_anchor(self):
        myfig, freqs, __ = plots.compare_groups_of_contacts({"CG1":self.CG1,"CG2":self.CG2},
                                                      {"CG1":"r", "CG2":"b"},
                                                      ctc_cutoff_Ang=1.5,
                                                      anchor="0")
        myfig.tight_layout()
        #myfig.savefig("3.test.png",bbox_inches="tight")
        _plt.close("all")

    def test_plot_singles(self):
        myfig, freqs, __ = plots.compare_groups_of_contacts({"CG1":self.CG1,"CG2":self.CG2},
                                                      {"CG1":"r", "CG2":"b"},
                                                      ctc_cutoff_Ang=1.5,
                                                      plot_singles=True)
        myfig.tight_layout()
        #myfig.savefig("4.test.png",bbox_inches="tight")
        _plt.close("all")

    def test_plot_singles_w_anchor(self):
        myfig, freqs, __ = plots.compare_groups_of_contacts({"CG1":self.CG1,"CG2":self.CG2},
                                                      {"CG1":"r", "CG2":"b"},
                                                      ctc_cutoff_Ang=1.5,
                                                      anchor="0",
                                                      plot_singles=True)
        myfig.tight_layout()
        #myfig.savefig("5.test.png",bbox_inches="tight")
        _plt.close("all")

    def test_with_freqdat(self):
        with _TDir(suffix="_test_mdciao") as tmpdir:
            freqfile = os.path.join(tmpdir,"freqtest.dat")
            with open(freqfile,"w") as f:
                f.write(self.CG2.frequency_str_ASCII_file(1.5, by_atomtypes=False))
            myfig, __, __ = plots.compare_groups_of_contacts({"CG1":self.CG1,
                                                        "CG2":self.CG2,
                                                        "CG2f":freqfile},
                                                       {"CG1":"r", "CG2":"b", "CG2f":"g"},
                                                       ctc_cutoff_Ang=1.5)
            # myfig.savefig("6.test.png",bbox_inches="tight")
        _plt.close("all")

    def test_with_freqdict(self):
        myfig, __, __ = plots.compare_groups_of_contacts({"CG1":self.CG1,
                                                    "CG2":self.CG2,
                                                    "CG2d":{"0-1":1, "0-3":1/3}},
                                                   {"CG1":"r", "CG2":"b", "CG2d":"g"},
                                                   ctc_cutoff_Ang=1.5)
        # myfig.savefig("7.test.png",bbox_inches="tight")
        _plt.close("all")

class Test_plot_w_smoothing_auto(unittest.TestCase):

    def test_just_runs(self):
        x = [0,1,2,3,4,5]
        y = [0,1,2,3,4,5]
        _plt.figure()
        plots.plot_w_smoothing_auto(_plt.gca(), x,y,
                              "test","r",
                              n_smooth_hw=1,
                              gray_background=True)

if __name__ == '__main__':
    unittest.main()
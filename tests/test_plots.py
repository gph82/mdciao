import unittest
import numpy as _np
import pytest

from matplotlib import pyplot as _plt, cm as _cm
from matplotlib.colors import is_color_like, to_rgb

from mdciao.contacts import ContactGroup, ContactPair
from mdciao.examples import ContactGroupL394

from mdciao import plots
from mdciao.plots.plots import _plot_freqbars_baseplot, _offset_dict, _color_dict_guesser, _try_colormap_string

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
        self.CG1_freqdict =          {"0-1":1,  "0-2":.75, "0-3":.50, "4-6":.25}
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

    def test_plot_unified_freq_dictsfigsize_None(self):
        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1": self.CG1_freqdict, "CG1copy": self.CG1_freqdict},
                                                        {"CG1": "r", "CG1copy": "b"}, ylim=2.25,
                                                        figsize=None)
        _plt.close("all")

    def test_plot_unified_freq_dictswinner(self):
        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1": {"0-1":0, "0-2":1},
                                                         "CG12": {"0-1":1, "0-2":0}},
                                                        figsize=None,
                                                        assign_w_color=True)
        _plt.close("all")

    def test_plot_just_one_dict(self):
        _plt.figure()
        ax = _plt.gca()
        myfig, myax, __ = plots.plot_unified_freq_dicts({"CG1": self.CG1_freqdict},
                                                  {"CG1": "r"})
        #myfig.savefig("12.test_just_one.png",bbox_inches="tight")
        _plt.close("all")

class Test_plot_unified_distro_dicts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.CGL394 = ContactGroupL394()
        D1 = cls.CGL394.distribution_dicts()
        D2 = {key: [val[0],val[1]+.02] for key,val in D1.items()}
        cls.dicts = {"L394":D1,
                     "L394shifted":D2}

    def test_plot_unified_distro_dicts(self):
        plots.plot_unified_distro_dicts(self.dicts, ctc_cutoff_Ang=3.5, n_cols=2,legend_rows=1)
        _plt.close("all")

class Test_compare_groups_of_contacts(unittest.TestCase):

    def setUp(self):
        self.CG1 = ContactGroup([ContactPair([0,1],[[.1, .1]], [[0., 1.]]),
                                 ContactPair([0,2],[[.1, .2]], [[0., 1.]])])

        self.CG2 = ContactGroup([ContactPair([0,1],[[.1, .1, .1]], [[0., 1., 2.]]),
                                 ContactPair([0,3],[[.1, .2, .2]], [[0., 1., 2.]])])

    def test_just_works(self):
        myfig, freqs, __ = plots.compare_groups_of_contacts({"CG1":self.CG1,"CG2":self.CG2},
                                                      ctc_cutoff_Ang=1.5)
        myfig.tight_layout()
        #myfig.savefig("1.test.png",bbox_inches="tight")
        _plt.close("all")

    def test_just_works_list(self):
        myfig, freqs, __ = plots.compare_groups_of_contacts([self.CG1, self.CG2],
                                                            ctc_cutoff_Ang=1.5)
        myfig.tight_layout()
        #myfig.savefig("1.1.test.png",bbox_inches="tight")
        _plt.close("all")

    def test_per_residue(self):
        myfig, freqs, __ = plots.compare_groups_of_contacts([self.CG1, self.CG2],
                                                            ctc_cutoff_Ang=1.5,
                                                            per_residue=True)
        myfig.tight_layout()
        #myfig.savefig("1.per_residue.test.png",bbox_inches="tight")
        _plt.close("all")


    def test_mutation(self):
        myfig, freqs, __ = plots.compare_groups_of_contacts({"CG1":self.CG1,"CG2":self.CG2},
                                                      ctc_cutoff_Ang=1.5,
                                                      mutations_dict={"3":"2"})
        myfig.tight_layout()
        #myfig.savefig("2.test.png",bbox_inches="tight")
        _plt.close("all")

    def test_anchor(self):
        myfig, freqs, __ = plots.compare_groups_of_contacts({"CG1":self.CG1,"CG2":self.CG2},
                                                            ctc_cutoff_Ang=1.5,
                                                      anchor="0")
        myfig.tight_layout()
        #myfig.savefig("3.test.png",bbox_inches="tight")
        _plt.close("all")

    def test_plot_singles(self):
        myfig, freqs, __ = plots.compare_groups_of_contacts({"CG1":self.CG1,"CG2":self.CG2},
                                                      ctc_cutoff_Ang=1.5,
                                                      plot_singles=True)
        myfig.tight_layout()
        #myfig.savefig("4.test.png",bbox_inches="tight")
        _plt.close("all")

    def test_plot_singles_w_anchor(self):
        myfig, freqs, __ = plots.compare_groups_of_contacts({"CG1":self.CG1,"CG2":self.CG2},
                                                      ctc_cutoff_Ang=1.5,
                                                      anchor="0",
                                                      plot_singles=True)
        myfig.tight_layout()
        #myfig.savefig("5.test.png",bbox_inches="tight")
        _plt.close("all")

    def test_with_freqdat(self):
        with _TDir(suffix="_test_mdciao") as tmpdir:
            freqfile = os.path.join(tmpdir,"freqtest.dat")
            self.CG2.frequency_table(1.5, freqfile)
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

    def test_color_list(self):
        myfig, freqs, __ = plots.compare_groups_of_contacts({"CG1": self.CG1, "CG2": self.CG2},
                                                            ["r","g"],
                                                            ctc_cutoff_Ang=1.5)
        myfig.tight_layout()
        # myfig.savefig("1.test.png",bbox_inches="tight")
        _plt.close("all")

    def test_with_CG_freqdict_file_as_list(self):
        with _TDir(suffix="_test_mdciao") as tmpdir:
            freqfile = os.path.join(tmpdir, "freqtest.dat")
            self.CG2.frequency_table(1.5,freqfile)
            myfig, __, __ = plots.compare_groups_of_contacts([self.CG1,
                                                              self.CG2,
                                                              {"0-1":1, "0-3":1/3},
                                                              freqfile],
                                                             ctc_cutoff_Ang=1.5)
            #myfig.savefig("8.test.png")
            _plt.close("all")

    def test_distro(self):
        CG : ContactGroup = ContactGroupL394()
        plots.compare_groups_of_contacts([CG,CG],distro=True)

class Test_plot_w_smoothing_auto(unittest.TestCase):

    def test_just_runs(self):
        x = [0,1,2,3,4,5]
        y = [0,1,2,3,4,5]
        _plt.figure()
        plots.plot_w_smoothing_auto(_plt.gca(), x,y,
                              "test","r",
                              n_smooth_hw=1,
                              gray_background=True)

class Test_color_by_values(unittest.TestCase):
    def setUp(self):
        self.freqs_by_sys_by_ctc = {"WT":   {"A": 1, "B": .0,  "C": 1.0},
                                    "mut1": {"A": 0, "B": .0,  "C": .25},
                                    "mut2": {"A": 1, "B": .75, "C": .25}}
        self.colordict = {"WT": "r", "mut1": "b", "mut2": "g"}
        from mdciao.plots import plots as _plots
        self.plots = _plots


    def test_runs(self):
        info = self.plots._color_by_values(["A", "B", "C"], self.freqs_by_sys_by_ctc,
                                           self.colordict)
        _np.testing.assert_equal(info,
                                 {"A": ("-", self.colordict["mut1"]),
                                  # the label gets the color of mut1 (only one winner)
                                  "B": ("+", self.colordict["mut2"]),
                                  # the label gets the color of mut2 (only one loser)
                                  "C": ("", "k")})  # nothing happens, neither winners nor losers

    def test_runs_wo_action(self):
        info = self.plots._color_by_values(["A", "B", "C"], self.freqs_by_sys_by_ctc,
                                           self.colordict,
                                           assign_w_color=False,
                                           )

        # nothing happens
        _np.testing.assert_equal(info,
                                 {"A": ("", "k"),
                                  "B": ("", "k"),
                                  "C": ("", "k")})

class Test_freqs_baseplot(unittest.TestCase):

    def test_baseplot_minimal(self):
        jax = _plot_freqbars_baseplot([1,2,3])
        assert isinstance(jax, _plt.Axes)
        _plt.close("all")

    def test_baseplot_pass_ax(self):
        _plt.plot()
        jax = _plt.gca()
        assert jax is _plot_freqbars_baseplot([1,2,3], jax=jax)
        _plt.close("all")

    def test_baseplot_truncate(self):

        jax = _plot_freqbars_baseplot([1,2,3], truncate_at=.5)

        _plt.close("all")
if __name__ == '__main__':
    unittest.main()


class Test_offset_dict(unittest.TestCase):

    def test_one(self):
        # Only one bar, padding is .2, hence position is 0 and width is 1-.2 = .8
        res, width = _offset_dict(["A"])
        self.assertDictEqual(res,{"A":0})
        self.assertEqual(width,.8)

    def test_two(self):

        res, width = _offset_dict(["left","right"])

        # Twice as many bars, with same padding, i.e. width = .8/2 = .4
        self.assertEqual(width,.4)

        # If width is .4 and there's two of them, they have to be centered around -.2 and +.2
        self.assertDictEqual(res,{"left":  -.2,
                                  "right":  .2})

    def test_three(self):
        res, width = _offset_dict(["left", "middle", "right"], width=.2)


        # Three with fixed width .2 means -.2, 0, +.2
        self.assertDictEqual(res, {"left": -.2,
                                   "middle":0,
                                   "right": +.2})

        self.assertEqual(width,.2)


class Test_colormaps(unittest.TestCase):

    def test_exception_works(self):
        colors = _try_colormap_string("jet", 10)
        self.assertEqual(len(colors),10)
        assert all([is_color_like(col) for col in colors])

    def test_exception_raises(self):
        with self.assertRaises(AttributeError):
            _try_colormap_string("Chet",10)

    def test_color_dict_guesser_None(self):
        colors = _color_dict_guesser(None, ["sys1","sys2","sys3"])
        colors_rgb_array = _np.vstack([to_rgb(col) for key, col in colors.items()])
        ref_color = _np.array(list(_cm.get_cmap("tab10")([0,1,2])))[:,:-1]
        _np.testing.assert_array_equal(colors_rgb_array, ref_color)
        self.assertListEqual(list(colors.keys()), ["sys1","sys2","sys3"])

    def test_color_dict_guesser_cmap(self):
        colors = _color_dict_guesser("Set2", ["sys1", "sys2", "sys3"])
        colors_rgb_array = _np.vstack([to_rgb(col) for key, col in colors.items()])
        ref_color = _np.array(list(_cm.get_cmap("Set2")([0, 1, 2])))[:, :-1]
        _np.testing.assert_array_equal(colors_rgb_array, ref_color)
        self.assertListEqual(list(colors.keys()), ["sys1", "sys2", "sys3"])

    def test_color_dict_guesser_list(self):
        colors = _color_dict_guesser(["r","g","b"], ["sys1", "sys2"])
        self.assertDictEqual(colors, {"sys1":"r","sys2":"g"})

    def test_color_dict_guesser_dict(self):
        colors = _color_dict_guesser({"sys1": "r", "sys2": "g"}, ["sys1", "sys2"])
        self.assertDictEqual(colors, {"sys1": "r", "sys2": "g"})


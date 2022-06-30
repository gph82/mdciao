import unittest
import numpy as _np
import pytest

from matplotlib import pyplot as _plt, cm as _cm
from matplotlib.colors import is_color_like, to_rgb

from mdciao.contacts import ContactGroup, ContactPair
from mdciao.examples import ContactGroupL394
from mdciao.cli import interface as _cli_interface
from mdciao import plots
from mdciao.plots.plots import _plot_freqbars_baseplot, _offset_dict, color_dict_guesser, _try_colormap_string, _plot_violin_baseplot, _color_tiler

from tempfile import TemporaryDirectory as _TDir
import os
import mdtraj as _md

from mdciao.examples import filenames as test_filenames

class TestPlotContactMatrix(unittest.TestCase):

    def test_plot_contact_matrix_runs_w_options(self):

        mat = _np.linspace(0,1,6).reshape(2,3)
        labels = [[1,2],["A","B","C"]]

        iax, ipix = plots.plot_matrix(mat,
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
            plots.plot_matrix(mat, labels)


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

    def test_interface(self):
        intf1 = _cli_interface(_md.load(test_filenames.actor_pdb),
                                fragments=[_np.arange(868, 875 + 1),  # ICL2
                                            _np.arange(328, 353 + 1)],  # Ga5,
                                ctc_cutoff_Ang=30,
                                no_disk=True,
                                figures=False
                                )
        intf2 = _cli_interface(_md.load(test_filenames.actor_pdb),
                                fragments=[_np.arange(860, 875 + 1),  # ICL2
                                            _np.arange(320, 353 + 1)],  # Ga5,
                                ctc_cutoff_Ang=30,
                                no_disk=True,
                                figures=False
                                )
        ifig, __, __  = plots.compare_groups_of_contacts({"1": intf1, "2":intf2}, ctc_cutoff_Ang=30,interface=True)
        #ifig.savefig("test.png")


class Test_plot_w_smoothing_auto(unittest.TestCase):

    def test_just_runs(self):
        x = [0,1,2,3,4,5]
        y = [0,1,2,3,4,5]
        _plt.figure()
        line2D = plots.plot_w_smoothing_auto(_plt.gca(), y, "test", "r", x=x, background=True, n_smooth_hw=1)
        assert isinstance(line2D, _plt.Line2D)
        _plt.close(_plt.gcf())

    def test_creates_x(self):
        y = [0,1,2,3,4,5]
        _plt.figure()
        line2D = plots.plot_w_smoothing_auto(_plt.gca(), y, "test", "r", background=True, n_smooth_hw=1)
        assert isinstance(line2D, _plt.Line2D)
        _plt.close(_plt.gcf())

    def test_background_False(self):
        y = [0,1,2,3,4,5]
        _plt.figure()
        line2D = plots.plot_w_smoothing_auto(_plt.gca(), y, "test", "r", background=False, n_smooth_hw=1)
        assert isinstance(line2D, _plt.Line2D)
        _plt.close(_plt.gcf())

    def test_background_str(self):
        y = [0,1,2,3,4,5]
        _plt.figure()
        line2D = plots.plot_w_smoothing_auto(_plt.gca(), y, "test", "r", background="green", n_smooth_hw=1)
        assert isinstance(line2D, _plt.Line2D)
        _plt.close(_plt.gcf())

    def test_background_rgb(self):
        y = [0,1,2,3,4,5]
        _plt.figure()
        line2D = plots.plot_w_smoothing_auto(_plt.gca(), y, "test", "r", background=[1.,0.,0.], n_smooth_hw=1)
        assert isinstance(line2D, _plt.Line2D)
        _plt.close(_plt.gcf())

    def test_background_raises(self):
        y = [0,1,2,3,4,5]
        _plt.figure()
        with self.assertRaises(AssertionError):
            line2D = plots.plot_w_smoothing_auto(_plt.gca(), y, "test", "r", background=["A",0.,0.], n_smooth_hw=1)

    def test_no_smoothing(self):
        y = [0,1,2,3,4,5]
        _plt.figure()
        line2D = plots.plot_w_smoothing_auto(_plt.gca(), y, "test", "r", )
        assert isinstance(line2D, _plt.Line2D)
        _plt.close(_plt.gcf())



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

        jax = _plot_freqbars_baseplot([1,2,3], lower_cutoff_val=.5)

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

class Test_dict_sorter(unittest.TestCase):

    def setUp(self):
        self.indict = {'R131-E392': 4.1,
                       'P138-M386': 6.85,
                       'K270-L394': 4.68,
                       'T274-L393': 4.95,
                       'Q142-I382': 7.73,
                       'T68-R389': 7.77,
                       'Y141-R389': 8.02,
                       'T274-E392': 5.73,
                       'V222-L393': 6.92}

    def test_residue(self):
        sorted_keys = plots.plots._key_sorter("residue", self.indict)

        self.assertListEqual(sorted_keys, [
            'T68-R389',
            "R131-E392",
            "P138-M386",
            'Y141-R389',
            'Q142-I382',
            'V222-L393',
            'K270-L394',
            'T274-E392',
            'T274-L393',
        ])

    def test_mean(self):
        sorted_keys = plots.plots._key_sorter("mean", self.indict)

        self.assertListEqual(sorted_keys,
                             list({'R131-E392': 4.1,
                                   'K270-L394': 4.68,
                                   'T274-L393': 4.95,
                                   'T274-E392': 5.73,
                                   'P138-M386': 6.85,
                                   'V222-L393': 6.92,
                                   'Q142-I382': 7.73,
                                   'T68-R389': 7.77,
                                   'Y141-R389': 8.02,
                                   }.keys()))

    def test_list(self):
        sorted_keys = plots.plots._key_sorter(["Y141-R389", "bogus1", "R131-E392", "bogus"], self.indict)

        self.assertListEqual(sorted_keys, ["Y141-R389","R131-E392"])

    def test_raises(self):
        with self.assertRaises(ValueError):
            plots.plots._key_sorter("bogus", self.indict)


class Test_colormaps(unittest.TestCase):

    def test_works(self):
        colors = _try_colormap_string("jet", 10)
        self.assertEqual(len(colors),10)
        assert all([is_color_like(col) for col in colors])

    def test_works_large_N(self):
        colors = _try_colormap_string("tab10", 20)
        self.assertEqual(len(colors),20)
        assert all([is_color_like(col) for col in colors])
        _np.testing.assert_array_equal(colors[0:10], colors[10:20])

    def test_exception_raises(self):
        with self.assertRaises(AttributeError):
            _try_colormap_string("Chet",10)

    def test_color_dict_guesser_None(self):
        colors = color_dict_guesser(None, ["sys1", "sys2", "sys3"])
        colors_rgb_array = _np.vstack([to_rgb(col) for key, col in colors.items()])
        ref_color = _np.array(list(_cm.get_cmap("tab10")([0,1,2])))[:,:-1]
        _np.testing.assert_array_equal(colors_rgb_array, ref_color)
        self.assertListEqual(list(colors.keys()), ["sys1","sys2","sys3"])

    def test_color_dict_guesser_cmap(self):
        colors = color_dict_guesser("Set2", ["sys1", "sys2", "sys3"])
        colors_rgb_array = _np.vstack([to_rgb(col) for key, col in colors.items()])
        ref_color = _np.array(list(_cm.get_cmap("Set2")([0, 1, 2])))[:, :-1]
        _np.testing.assert_array_equal(colors_rgb_array, ref_color)
        self.assertListEqual(list(colors.keys()), ["sys1", "sys2", "sys3"])

    def test_color_dict_guesser_cmap_n(self):
        colors = color_dict_guesser("Set2", 3)
        colors_rgb_array = _np.vstack([to_rgb(col) for key, col in colors.items()])
        ref_color = _np.array(list(_cm.get_cmap("Set2")([0, 1, 2])))[:, :-1]
        _np.testing.assert_array_equal(colors_rgb_array, ref_color)
        self.assertListEqual(list(colors.keys()), [0, 1, 2])

    def test_color_dict_guesser_list(self):
        colors = color_dict_guesser(["r", "g", "b"], ["sys1", "sys2"])
        self.assertDictEqual(colors, {"sys1":"r","sys2":"g"})

    def test_color_dict_guesser_dict(self):
        colors = color_dict_guesser({"sys1": "r", "sys2": "g"}, ["sys1", "sys2"])
        self.assertDictEqual(colors, {"sys1": "r", "sys2": "g"})

class Test_color_tiler(unittest.TestCase):

    def test_str(self):
        self.assertListEqual(['red', 'red', 'red', 'red', 'red'], _color_tiler("red", 5))  # pure str

    def test_list(self):
        self.assertListEqual(['red', 'red', 'red', 'red', 'red'],
                             _color_tiler(["red"], 5))  # list of str with 1 item

        self.assertListEqual(['red', 'blue', 'red', 'blue', 'red'],
                             _color_tiler(["red", "blue"], 5))  # list of strs

    def test_array(self):
        self.assertListEqual([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]],
                             _color_tiler([[1., 0., 0.],  # red
                                           [0., 1., 0.],  # green
                                           [0., 0., 1.]], 5))  # blue

class Test_plot_violin_baseplot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.CGL394 : ContactGroup = ContactGroupL394()
        cp: ContactPair

        cls.time_trajs = [cp.time_traces.ctc_trajs for cp in cls.CGL394._contacts]
        cls.time_trajs_shifted = [[itraj+.25 for itraj in cp] for cp in cls.time_trajs]
    def test_minimal(self):
        iax, _ = _plot_violin_baseplot(self.time_trajs, labels=self.CGL394.ctc_labels)
        #iax.figure.savefig("test.pdf")
        _plt.close("all")

    def test_second_axis(self):
        iax, _ = _plot_violin_baseplot(self.time_trajs)
        iax, _ = _plot_violin_baseplot(self.time_trajs_shifted, labels=self.CGL394.ctc_labels, jax=iax)
        #iax.figure.savefig("test.pdf")
        _plt.close("all")

class Test_plot_compare_violins(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.CGL394 : ContactGroup = ContactGroupL394()
        cls.CGL394_larger  = ContactGroupL394(ctc_control=.99, ctc_cutoff_Ang=5)

    def test_works(self):
        fig, ax, sorted_keys = plots.compare_violins({"small":self.CGL394, "big":self.CGL394_larger},
                                        anchor="L394",
                                        ymax=10, ctc_cutoff_Ang=4)

        #fig.savefig("test.pdf")
        _plt.close("all")

    def test_works_no_defrag_and_list_zero_freq_remove_identities(self):
        fig, ax, sorted_keys = plots.compare_violins([self.CGL394, self.CGL394_larger],
                                                     anchor="L394",
                                                     ymax=10, ctc_cutoff_Ang=4,
                                                     zero_freq=.5,
                                                     remove_identities=True,
                                                     identity_cutoff=.90,
                                                     defrag=None)
        # This should hold
        """
        {'L394@G.H5.26-I233@5.72x72': 0.39285714285714285, # below zero_freq
        'R389@G.H5.21-L394@G.H5.26': 0.9464285714285714,  # above identity
        'L388@G.H5.20-L394@G.H5.26': 0.9464285714285714,  # above identity
        'R385@G.H5.17-L394@G.H5.26': 0.6964285714285714, 
        'L394@G.H5.26-L230@5.69x69': 0.8392857142857143, 
        'L394@G.H5.26-K270@6.32x32': 0.4642857142857143,  # below zero_freq
        'L394@G.H5.26-K267@6.29x29': 0.0, # below zero_freq
        'L394@G.H5.26-Q229@5.68x68': 0.0} # below zero_freq
        """
        self.assertListEqual(sorted_keys, ["L230@5.69x69", 'R385@G.H5.17'])
        _plt.close("all")

    def test_works_no_defrag_and_list(self):
        fig, ax, sorted_keys = plots.compare_violins([self.CGL394, self.CGL394_larger],
                                        anchor="L394",
                                        ymax=10, ctc_cutoff_Ang=4,
                                        defrag=None)

        #fig.savefig("test.pdf")
        _plt.close("all")

    def test_repframes_True(self):
        fig, ax, sorted_keys = plots.compare_violins({"small": self.CGL394, "big": self.CGL394_larger},
                                                     anchor="L394",
                                                     ymax=10, ctc_cutoff_Ang=4,
                                                     representatives=True)


        #fig.savefig("test.pdf")
        _plt.close("all")

    def test_repframes_int(self):
        fig, ax, sorted_keys = plots.compare_violins({"small": self.CGL394, "big": self.CGL394_larger},
                                                     anchor="L394",
                                                     ymax=10, ctc_cutoff_Ang=4,
                                                     representatives=2)
        # fig.savefig("test.pdf")
        _plt.close("all")

    def test_repframes_dict_kwargs(self):
        fig, ax, sorted_keys = plots.compare_violins({"small": self.CGL394, "big": self.CGL394_larger},
                                                     anchor="L394",
                                                     ymax=10, ctc_cutoff_Ang=4,
                                                     representatives={"n_frames":3, "scheme":"mean"})
        #fig.savefig("test.pdf")
        _plt.close("all")

    def test_repframes_dict_geoms(self):
        traj = _md.load(test_filenames.traj_xtc_stride_20, top=test_filenames.top_pdb)
        repframes = {"small" : traj[:3],
                     "big" : traj[:5]}
        fig, ax, sorted_keys = plots.compare_violins({"small": self.CGL394, "big": self.CGL394_larger},
                                                     anchor="L394",
                                                     ymax=10, ctc_cutoff_Ang=4,
                                                     representatives=repframes)
        #fig.savefig("test.pdf")
        _plt.close("all")
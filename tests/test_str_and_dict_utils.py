import mdtraj as md
from os import path
import numpy as np

import unittest
from mdciao.examples import filenames as test_filenames

from mdciao.utils import str_and_dict

from tempfile import TemporaryDirectory as _TDir

import io as _io, contextlib as _contextlib

from pandas import \
    DataFrame as _DF,\
    ExcelWriter as _XW

class Test_get_sorted_trajectories(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.top_pdb)
        self.traj = md.load(test_filenames.traj_xtc_stride_20, top=self.geom.top)
        self.traj_reverse = md.load(test_filenames.traj_xtc_stride_20, top=self.geom.top)[::-1]

    def test_glob_with_pattern(self):
        str_and_dict.get_trajectories_from_input(path.join(test_filenames.example_path, "*.xtc"))

    def test_glob_with_filename(self):
        str_and_dict.get_trajectories_from_input(test_filenames.traj_xtc_stride_20)

    def test_with_one_trajectory_object(self):
        list_out = str_and_dict.get_trajectories_from_input(self.traj)
        assert len(list_out)==1
        assert isinstance(list_out[0], md.Trajectory)

    def test_with_trajectory_objects(self):
        str_and_dict.get_trajectories_from_input([self.traj,
                                                  self.traj_reverse])


    def test_fails_if_not_traj_at_all(self):
        with self.assertRaises(FileNotFoundError):
            str_and_dict.get_trajectories_from_input("bogus.xtc")

    def test_fails_if_not_trajs(self):
        with self.assertRaises(AssertionError):
            str_and_dict.get_trajectories_from_input([self.traj,
                                                      1])


class Test_inform_about_trajectories(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.top_pdb)
        self.traj = md.load(test_filenames.traj_xtc_stride_20, top=self.geom.top)
        self.traj_reverse = md.load(test_filenames.traj_xtc_stride_20, top=self.geom.top)[::-1]

    def test_fails_no_list(self):
        with self.assertRaises(AssertionError):
            str_and_dict.inform_about_trajectories(test_filenames.traj_xtc_stride_20)

    def test_list_of_files(self):
        str_and_dict.inform_about_trajectories([test_filenames.traj_xtc_stride_20,
                                                test_filenames.traj_xtc_stride_20])

    def test_list_of_trajs(self):
        str_and_dict.inform_about_trajectories([self.traj,
                                                self.traj_reverse])


class Test_replace_w_dict(unittest.TestCase):
    outkey = str_and_dict.replace_w_dict("key I don't like", {"key": "word", "like": "love"})
    assert outkey=="word I don't love"

class Test_delete_exp_inkeys(unittest.TestCase):

    def test_runs(self):
        indict = {"GLU30-LYS40":True,
                  "ARG40-GLU30":False}

        outdict, deleted = str_and_dict.delete_exp_in_keys(indict, "GLU30")
        self.assertEqual(deleted,["GLU30","GLU30"])
        assert len(outdict)==2
        assert outdict["LYS40"]
        assert not outdict["ARG40"]


class Test_iterate_and_inform_lambdas(unittest.TestCase):

    def setUp(self):
        self.filename = test_filenames.traj_xtc_stride_20
        self.pdb = test_filenames.top_pdb
        self.top = md.load(self.pdb).top
        self.traj = md.load(self.filename, top=self.top)
        self.stride=3

    def _call_iterators_and_test_them(self, iterate, inform, ixtc, stride=1):
        print()
        times = []
        nf = 0
        for ii, chunk in enumerate(iterate(ixtc)):
            times.append(chunk.time)
            nf += chunk.n_frames
            inform(ixtc, 0, ii, nf)
            print()
        times = np.hstack(times)
        # This is indirectly checking the right frames were grabbed
        assert np.allclose(times, self.traj.time[::stride])
        print()

    def test_filename_no_stride(self):
        iterate, inform = str_and_dict.iterate_and_inform_lambdas(self.filename,
                                                     10,
                                                     top=self.top)
        self._call_iterators_and_test_them(iterate, inform, self.filename, 1)

    def test_filename_no_stride_filename_is_ascii_just_works(self):
        iterate, inform = str_and_dict.iterate_and_inform_lambdas(self.pdb,
                                                     10,
                                                     top=self.top)
        nf = 0
        for ii, chunk in enumerate(iterate(self.pdb)):
            nf += chunk.n_frames
            inform(self.pdb,0,ii,nf)
        assert nf==1
        assert ii==0


    def test_filename_w_stride(self):
        iterate, inform = str_and_dict.iterate_and_inform_lambdas(self.filename,
                                                     10,
                                                     stride=self.stride,
                                                     top=self.top)

        self._call_iterators_and_test_them(iterate, inform, self.filename,
                                           stride = self.stride)

    def test_traj_no_stride(self):
        iterate, inform = str_and_dict.iterate_and_inform_lambdas(self.traj,
                                                     10,
                                                     )

        self._call_iterators_and_test_them(iterate, inform, self.traj)

    def test_traj_w_stride(self):
        iterate, inform = str_and_dict.iterate_and_inform_lambdas(self.traj,
                                                     10,
                                                     stride=self.stride
                                                     )

        self._call_iterators_and_test_them(iterate, inform, self.traj,
                                           stride=self.stride)

class Test_unify_freq_dicts(unittest.TestCase):

    def setUp(self):
        self.freq_dicts = {"WT":   {"GLU30-LYS40":1.0,
                                    "GLU30-LYS50":.5,
                                    "LYS60-GLU30": .1},
                           "K40A": {"ALA40-GLU30":.7,
                                   #"LYS50-GLU30":.5,
                                    "GLU30-LYS60": .2},
                           "prot": {"LYS40-GLH30": .3
                                    }}
        self.freq_dicts_frags = {"WT": {"GLU30@frag1-LYS40@frag2": 1.0,
                                        "GLU30@frag1-LYS50@frag3": .5,
                                        "LYS60@frag4-GLU30@frag1": .1},
                                 "K40A": {"ALA40-GLU30@frag1": .7,
                                          # "LYS50-GLU30":.5,
                                          "GLU30@frag1-LYS60@frag4": .2},
                                 "prot": {"LYS40@frag2-GLH30@frag1": .3
                                          }}
    def test_basic(self):
        out_dict = str_and_dict.unify_freq_dicts(self.freq_dicts,
                                    key_separator=None)
        #original
        assert out_dict["WT"]["GLU30-LYS40"] == 1.0
        assert out_dict["WT"]["GLU30-LYS50"] == .5
        assert out_dict["WT"]["LYS60-GLU30"] == .1
        #external
        assert out_dict["WT"]["ALA40-GLU30"] == 0 #from K40A
        assert out_dict["WT"]["GLU30-LYS60"] == 0 #from K40A
        assert out_dict["WT"]["LYS40-GLH30"] == 0 #from prot

        # original
        assert out_dict["K40A"]["ALA40-GLU30"] == .7
        assert out_dict["K40A"]["GLU30-LYS60"] == .2
        # external
        assert out_dict["K40A"]["GLU30-LYS40"] == 0 #from WT
        assert out_dict["K40A"]["GLU30-LYS50"] == 0 #from WT
        assert out_dict["K40A"]["LYS40-GLH30"] == 0 #from prot

        # original
        assert out_dict["prot"]["LYS40-GLH30"] == .3
        # external
        assert out_dict["prot"]["GLU30-LYS40"] == 0  # from WT
        assert out_dict["prot"]["GLU30-LYS50"] == 0  # from WT
        assert out_dict["prot"]["LYS60-GLU30"] == 0  # from WT

        assert out_dict["prot"]["LYS60-GLU30"] == 0  # from K40A
        assert out_dict["prot"]["ALA40-GLU30"] == 0  # from K40A

    def test_basic_defrag(self):
        out_dict = str_and_dict.unify_freq_dicts(self.freq_dicts,
                                                 key_separator=None,
                                                 defrag="@")
        #original
        assert out_dict["WT"]["GLU30-LYS40"] == 1.0
        assert out_dict["WT"]["GLU30-LYS50"] == .5
        assert out_dict["WT"]["LYS60-GLU30"] == .1
        #external
        assert out_dict["WT"]["ALA40-GLU30"] == 0 #from K40A
        assert out_dict["WT"]["GLU30-LYS60"] == 0 #from K40A
        assert out_dict["WT"]["LYS40-GLH30"] == 0 #from prot

        # original
        assert out_dict["K40A"]["ALA40-GLU30"] == .7
        assert out_dict["K40A"]["GLU30-LYS60"] == .2
        # external
        assert out_dict["K40A"]["GLU30-LYS40"] == 0 #from WT
        assert out_dict["K40A"]["GLU30-LYS50"] == 0 #from WT
        assert out_dict["K40A"]["LYS40-GLH30"] == 0 #from prot

        # original
        assert out_dict["prot"]["LYS40-GLH30"] == .3
        # external
        assert out_dict["prot"]["GLU30-LYS40"] == 0  # from WT
        assert out_dict["prot"]["GLU30-LYS50"] == 0  # from WT
        assert out_dict["prot"]["LYS60-GLU30"] == 0  # from WT

        assert out_dict["prot"]["LYS60-GLU30"] == 0  # from K40A
        assert out_dict["prot"]["ALA40-GLU30"] == 0  # from K40A

    def test_basic_exclude(self):
        with self.assertRaises(NotImplementedError):
            out_dict = str_and_dict.unify_freq_dicts(self.freq_dicts,
                                        exclude=["LYS6"], #to check that is just a pattern matching, not full str match
                                    key_separator=None)

        """
        #original
        assert out_dict["WT"]["GLU30-LYS40"] == 1.0
        assert out_dict["WT"]["GLU30-LYS50"] == .5
        assert "LYS60-GLU30" not in out_dict["WT"].keys(), out_dict["WT"].keys()

        #external
        assert out_dict["WT"]["ALA40-GLU30"] == 0 #from K40A
        assert out_dict["WT"]["GLU30-LYS60"] == 0 #from K40A
        assert out_dict["WT"]["LYS40-GLH30"] == 0 #from prot

        # original
        assert out_dict["K40A"]["ALA40-GLU30"] == .7
        assert out_dict["K40A"]["GLU30-LYS60"] == .2
        # external
        assert out_dict["K40A"]["GLU30-LYS40"] == 0 #from WT
        assert out_dict["K40A"]["GLU30-LYS50"] == 0 #from WT
        assert out_dict["K40A"]["LYS40-GLH30"] == 0 #from prot

        # original
        assert out_dict["prot"]["LYS40-GLH30"] == .3
        # external
        assert out_dict["prot"]["GLU30-LYS40"] == 0  # from WT
        assert out_dict["prot"]["GLU30-LYS50"] == 0  # from WT
        assert out_dict["prot"]["LYS60-GLU30"] == 0  # from WT

        assert out_dict["prot"]["LYS60-GLU30"] == 0  # from K40A
        assert out_dict["prot"]["ALA40-GLU30"] == 0  # from K40A
        """

    def test_mutations(self):
        out_dict = str_and_dict.unify_freq_dicts(self.freq_dicts,
                                    key_separator=None,
                                    replacement_dict={"ALA40":"LYS40"})
        #original
        assert out_dict["WT"]["GLU30-LYS40"] == 1.0
        assert out_dict["WT"]["GLU30-LYS50"] == .5
        assert out_dict["WT"]["LYS60-GLU30"] == .1
        #external
        assert "ALA40-GLU30" not in out_dict["WT"].keys() #from K40A, will be now LYS40
        assert out_dict["WT"]["LYS40-GLU30"] == 0 #from K40A, was mutated back to LYS
        assert out_dict["WT"]["GLU30-LYS60"] == 0 #from K40A
        assert out_dict["WT"]["LYS40-GLH30"] == 0 #from prot

        # original
        assert out_dict["K40A"]["LYS40-GLU30"] == .7 #was mutated
        assert out_dict["K40A"]["GLU30-LYS60"] == .2
        # external
        assert out_dict["K40A"]["GLU30-LYS40"] == 0 #from WT
        assert out_dict["K40A"]["GLU30-LYS50"] == 0 #from WT
        assert out_dict["K40A"]["LYS40-GLH30"] == 0 #from prot

        # original
        assert out_dict["prot"]["LYS40-GLH30"] == .3
        # external
        assert out_dict["prot"]["GLU30-LYS40"] == 0  # from WT
        assert out_dict["prot"]["GLU30-LYS50"] == 0  # from WT
        assert out_dict["prot"]["LYS60-GLU30"] == 0  # from WT

        assert out_dict["prot"]["LYS60-GLU30"] == 0  # from K40A
        assert out_dict["prot"]["LYS40-GLU30"] == 0  # from K40A

    def test_reorder(self):
        out_dict = str_and_dict.unify_freq_dicts(self.freq_dicts,
                                    key_separator="-",
                                    )
        #original
        assert out_dict["WT"]["GLU30-LYS40"] == 1.0
        assert out_dict["WT"]["GLU30-LYS50"] == .5
        assert out_dict["WT"]["GLU30-LYS60"] == .1
        assert "LYS60-GLU30" not in out_dict["WT"].keys() #bc reordering
        #external
        assert out_dict["WT"]["ALA40-GLU30"] == 0 #from K40A
        assert out_dict["WT"]["GLH30-LYS40"] == 0 #from prot, reordered

        # original
        assert out_dict["K40A"]["ALA40-GLU30"] == .7
        assert out_dict["K40A"]["GLU30-LYS60"] == .2 #was reordered
        assert "LYS60-GLU30" not in out_dict["K40A"].keys()
        # external
        assert out_dict["K40A"]["GLU30-LYS40"] == 0 #from WT
        assert out_dict["K40A"]["GLU30-LYS50"] == 0 #from WT
        assert out_dict["K40A"]["GLH30-LYS40"] == 0 #from prot, reordered

        # original
        assert out_dict["prot"]["GLH30-LYS40"] == .3 #was reordered
        assert "LYS40-GLH30" not in out_dict["prot"].keys()

        # external
        assert out_dict["prot"]["GLU30-LYS40"] == 0  # from WT
        assert out_dict["prot"]["GLU30-LYS50"] == 0  # from WT
        assert out_dict["prot"]["GLU30-LYS60"] == 0  # from WT, reordered
        assert "LYS60-GLU30" not in out_dict["prot"].keys()

        #assert out_dict["prot"]["LYS60-GLU30"] == 0  # from K40A not anymore, was set to 0 already
        assert out_dict["prot"]["ALA40-GLU30"] == 0  # from K40A

    def test_per_residue(self):
        """
         self.freq_dicts = {"WT":   {"GLU30-LYS40":1.0,
                                    "GLU30-LYS50":.5,
                                    "LYS60-GLU30": .1},
                           "K40A": {"ALA40-GLU30":.7,
                                    "GLU30-LYS60": .2},
                           "prot": {"LYS40-GLH30": .3
                                    }}
        Returns
        -------

        """
        out_dict = str_and_dict.unify_freq_dicts(self.freq_dicts,
                                                 per_residue=True)

        test_dict = {"WT":   {"GLU30": 1.0 + .5 + .1,
                              "LYS40": 1.0 + 0,
                              "LYS50": .5,
                              "LYS60": .1,
                              "ALA40": 0,
                              "GLH30": 0},
                     "K40A": {"ALA40": .7,
                              "GLU30": .7 + .2,
                              "LYS60": .2,
                              "LYS40": 0,
                              "LYS50": 0,
                              "GLH30": 0},
                     "prot": {"GLU30":0,
                              "LYS40":.3,
                              "LYS50":0,
                              "LYS60":0,
                              "ALA40":0,
                              "GLH30":.3}
                     }


        self.assertDictEqual(out_dict["WT"], test_dict["WT"])
        self.assertDictEqual(out_dict["K40A"], test_dict["K40A"])
        self.assertDictEqual(out_dict["prot"], test_dict["prot"])

    def test_replacement_takes_place_before_reordering(self):
        ud = str_and_dict.unify_freq_dicts({"WT": {"A100-B200": 1.0},
                                            "MUT": {"C100-B200": 1.0}},
                                           replacement_dict={"C100": "A100"})

        self.assertDictEqual(ud, {'WT': {'A100-B200': 1.0}, 'MUT': {'A100-B200': 1.0}})

class Test_average_freq_dicts(unittest.TestCase):

    def setUp(self):
        self.freq_dicts = {"T300": {"GLU30-LYS40": 1.0,
                                    "LYS60-GLU30": .1},
                           "T310": {"GLU30-LYS40": .5,
                                    }
                           }



    def test_works(self):
        av_frqs = str_and_dict.average_freq_dict(self.freq_dicts)
        self.assertDictEqual(av_frqs,{"GLU30-LYS40":.75,"GLU30-LYS60":.05})

    def test_weights(self):
        av_frqs = str_and_dict.average_freq_dict(self.freq_dicts, weights={"T300":0, "T310":1})
        self.assertDictEqual(av_frqs, {"GLU30-LYS40": .5, "GLU30-LYS60": .0})

class Test_str_latex(unittest.TestCase):

    def setUp(self):
        self.greek_letters=["alpha", "beta", "gamma", "mu"]
        self.sub_supers = ["C_2", "C^2"]

    def test_works(self):
        np.testing.assert_array_equal(str_and_dict.replace4latex("There's an alpha and a beta here, also C_2"),
                                    "There's an $\\alpha$ and a $\\beta$ here, also $\\mathrm{{C}_{2}}$")

    def test_nothing_happens(self):
        istr = str_and_dict.replace4latex("There's an alpha and a beta here, also C_2")
        np.testing.assert_array_equal(str_and_dict.replace4latex(istr),
                                    istr)

class Test_auto_fragment_string(unittest.TestCase):

    def test_one_bad(self):
        assert str_and_dict.choose_options_descencing([None]) == ""

    def test_both_bad(self):
        assert str_and_dict.choose_options_descencing([None, None]) == ""

    def test_both_good(self):
        assert str_and_dict.choose_options_descencing(["3.50","fragA"]) == "3.50"

    def test_only_option(self):
        self.assertEqual(str_and_dict.choose_options_descencing([None, "fragA"]),"fragA")

    def test_only_better_option(self):
        assert str_and_dict.choose_options_descencing(["3.50",None]) == "3.50"

    def test_pick_best_label_just_works(self):
        assert (str_and_dict.choose_options_descencing(["Print this","Print this instead"]) == "Print this")

    def test_pick_best_label_exclude_works(self):
        assert(str_and_dict.choose_options_descencing([None,  "Print this instead"]) == "Print this instead")
        assert(str_and_dict.choose_options_descencing(["None","Print this instead"]) == "Print this instead")
        assert(str_and_dict.choose_options_descencing(["NA",  "Print this instead"]) == "Print this instead")
        assert(str_and_dict.choose_options_descencing(["na",  "Print this instead"]) == "Print this instead")

class Test_freq_file2dict(unittest.TestCase):

    def setUp(self):
        self.freqs = {"0-1":0.3, "0-2":.4}
        self.freqs_frags = {"0@TM1-1@TM2": 0.3, "0@TM1-2@TM2": .4}

    def test_works_w_ascii(self):
        with _TDir(suffix="_test_mdciao") as tempdir:
            tmpfile = path.join(tempdir, "frqfile.dat")
            with open(tmpfile,"w") as f:
                for key, val in self.freqs.items():
                    f.write("%f %s\n"%(val, key))

            freqsin = str_and_dict.freq_file2dict(tmpfile)
        assert freqsin["0-1"]==.3
        assert freqsin["0-2"]==.4
        assert len(freqsin)==2

    def test_works_w_excel_no_header(self):
        with _TDir(suffix="_test_mdciao") as tempdir:
            tmpfile = "freqfile.xlsx"
            tmpfile =  path.join(tempdir,tmpfile)
            with _XW(tmpfile) as writer:
                _DF.from_dict({"freq" : list(self.freqs.values()),
                               "label": list(self.freqs.keys())}, ).to_excel(writer,
                                                                             index=False,
                                                                             )

            freqsin = str_and_dict.freq_file2dict(tmpfile)
            self.assertEqual(freqsin["0-1"],.3)
            self.assertEqual(freqsin["0-2"],.4)
            assert len(freqsin) == 2

    def test_works_w_excel_w_header(self):
        with _TDir(suffix="_test_mdciao") as tempdir:
            tmpfile = "freqfile.xlsx"
            tmpfile =  path.join(tempdir,tmpfile)
            with _XW(tmpfile) as writer:
                _DF.from_dict({"freq" : list(self.freqs.values()),
                               "label": list(self.freqs.keys())}, ).to_excel(writer,
                                                                             index=False,
                                                                             startrow=1
                                                                             )

            freqsin = str_and_dict.freq_file2dict(tmpfile)
            self.assertEqual(freqsin["0-1"],.3)
            self.assertEqual(freqsin["0-2"],.4)
            assert len(freqsin) == 2

    def test_works_w_excel_w_header_defrag(self):
        with _TDir(suffix="_test_mdciao") as tempdir:
            tmpfile = "freqfile.xlsx"
            tmpfile =  path.join(tempdir,tmpfile)
            with _XW(tmpfile) as writer:
                _DF.from_dict({"freq" : list(self.freqs.values()),
                               "label": list(self.freqs.keys())}, ).to_excel(writer,
                                                                             index=False,
                                                                             startrow=1
                                                                             )

            freqsin = str_and_dict.freq_file2dict(tmpfile, defrag="@")
            self.assertEqual(freqsin["0-1"],.3)
            self.assertEqual(freqsin["0-2"],.4)
            assert len(freqsin) == 2


class Test_fnmatch_functions(unittest.TestCase):

    def setUp(self):
        self.dict = {"John Perez":[0,1],
                     "Johnathan Hernandez":[2,3],
                     "Guille Doe":[4,5],
                     "Johnny Doe":[6,7]}
        self.patterns = "John*,*Doe,-Johnny*"
        # pick all FN John, LN Doe but avoid Johnny

    def test_fnmatch_ex_works(self):

        names = self.dict.keys()
        filtered = str_and_dict.fnmatch_ex(self.patterns,names)
        self.assertEqual(filtered,["John Perez", "Johnathan Hernandez","Guille Doe"])

    def test_match_dict_by_patterns_works(self):
        matching_keys, matching_values = str_and_dict.match_dict_by_patterns(self.patterns,self.dict,
                                                      verbose=True)
        self.assertEqual(matching_keys,["John Perez", "Johnathan Hernandez","Guille Doe"])
        np.testing.assert_array_equal(matching_values,np.arange(6))
        # pick all FN John, LN Doe but avoid Johny

    def test_match_dict_by_patterns_empty(self):
        matching_keys, matching_values = str_and_dict.match_dict_by_patterns("Maria",
                                                                                   self.dict,
                                                      verbose=True)
        self.assertEqual(matching_keys,[])
        np.testing.assert_array_equal(matching_values,[])
        # pick all FN John, LN Doe but avoid Johny

class Test_aggregate_freq_dict_per_residue(unittest.TestCase):

    def test_works(self):
        indict = {"A-B":1.5, "B-C":10, "A-D":100, "D-C":1000}
        out_dict=str_and_dict.sum_dict_per_residue(indict,"-")
        np.testing.assert_equal(out_dict["A"], 1.5+100)
        np.testing.assert_equal(out_dict["B"], 1.5 + 10)
        np.testing.assert_equal(out_dict["C"], 10 + 1000)
        np.testing.assert_equal(out_dict["D"], 1000 + 100)

class Test_sort_dict(unittest.TestCase):

    def test_works(self):
        indict = {"A":0.5, "B":1.0, "C":.75}
        self.assertDictEqual(str_and_dict.sort_dict_by_asc_values(indict),               {"A": 0.5, "C": .75, "B": 1.0})
        self.assertDictEqual(str_and_dict.sort_dict_by_asc_values(indict, reverse=True), {"B": 1.0, "C": .75, "A": 0.5})


class Test_lexsort_ctc_labels(unittest.TestCase):

    def test_works(self):
        labels = ["ALA30@3.50-GLU50", "HIS28-GLU50", "ALA30-GLU20"]
        sorted_labels, order = str_and_dict.lexsort_ctc_labels(labels)
        self.assertListEqual(sorted_labels, ["HIS28-GLU50", "ALA30-GLU20", "ALA30@3.50-GLU50"])
        self.assertListEqual(order.tolist(), [1, 2, 0])

    def test_works_one_residue(self):
        labels = ["GLU50", "ALA30", "GLU20"]
        sorted_labels, order = str_and_dict.lexsort_ctc_labels(labels)
        self.assertListEqual(sorted_labels, ["GLU20", "ALA30", "GLU50"])
        self.assertListEqual(order.tolist(), [2, 1, 0])
    def test_works_reverse(self):
        labels = ["ALA30@3.50-GLU50", "HIS28-GLU50", "ALA30-GLU20"]
        sorted_labels, order = str_and_dict.lexsort_ctc_labels(labels, reverse=True)
        self.assertListEqual(sorted_labels, ["HIS28-GLU50", "ALA30-GLU20", "ALA30@3.50-GLU50"][::-1])
        self.assertListEqual(order.tolist(), [1, 2, 0][::-1])

    def test_works_second_column_first(self):
        labels = ["ALA30@3.50-GLU50", "HIS28-GLU50", "ALA30-GLU20"]
        sorted_labels, order = str_and_dict.lexsort_ctc_labels(labels, columns=[1, 0])
        self.assertListEqual(sorted_labels, ["ALA30-GLU20", "HIS28-GLU50", "ALA30@3.50-GLU50"])
        self.assertListEqual(order.tolist(), [2, 1, 0])

    def test_works_second_column_first_reverse(self):
        labels = ["ALA30@3.50-GLU50", "HIS28-GLU50", "ALA30-GLU20"]
        sorted_labels, order = str_and_dict.lexsort_ctc_labels(labels, columns=[1, 0], reverse=True)
        self.assertListEqual(sorted_labels, ["ALA30@3.50-GLU50", "HIS28-GLU50", "ALA30-GLU20"])
        self.assertListEqual(order.tolist(), [0, 1, 2])

class Test_label2componentsdict(unittest.TestCase):

    def setUp(self):
        self.w = {
            1: "r1",
            2: "r1@f1",
            3: "r1@f1-r2",
            4: "r1@f1-r2",
            5: "r1@f1-r2@f2",
            6: "r1@f1-1-r2@f2",
            7: "r1@f1-r2@f2-2",
            8: "r1@f1-1-r2@f2-2",
            9: "r1-r2",
           10: "r1-r2@f2",
           11: "r1-r2@f2-1",
           12: "r1@f1-1-1-r2@f2-2-2"
        }

    def test_1(self):
        self.assertDictEqual({"res1": "r1"},
                             str_and_dict._label2componentsdict(self.w[1]),
                             self.w[1])

    def test_2(self):
        self.assertDictEqual({"res1": "r1",
                              "frag1": "f1"},
                             str_and_dict._label2componentsdict(self.w[2]),
                             self.w[2])

    def test_3(self):
        self.assertDictEqual({"res1": "r1",
                              "frag1": "f1",
                              "res2": "r2"
                              },
                             str_and_dict._label2componentsdict(self.w[3]),
                             self.w[3])

    def test_4_assume_False(self):
        self.assertDictEqual({"res1": "r1",
                              "frag1": "f1-r2",
                              },
                             str_and_dict._label2componentsdict(self.w[4],
                                                                assume_ctc_label=False),
                             self.w[4])

    def test_5(self):
        self.assertDictEqual({"res1": "r1",
                              "frag1": "f1",
                              "res2": "r2",
                              "frag2": "f2"},
                             str_and_dict._label2componentsdict(self.w[5]),
                             self.w[5])


    def test_6(self):
        self.assertDictEqual({"res1": "r1",
                              "frag1": "f1-1",
                              "res2": "r2",
                              "frag2": "f2"},
                             str_and_dict._label2componentsdict(self.w[6]),
                             self.w[6])

    def test_7(self):
        self.assertDictEqual({"res1": "r1",
                              "frag1": "f1",
                              "res2": "r2",
                              "frag2": "f2-2"},
                             str_and_dict._label2componentsdict(self.w[7]),
                             self.w[7])

    def test_8(self):
        self.assertDictEqual({"res1": "r1",
                              "frag1": "f1-1",
                              "res2": "r2",
                              "frag2": "f2-2"},
                             str_and_dict._label2componentsdict(self.w[8]),
                             self.w[8])

    def test_9(self):
        self.assertDictEqual({"res1": "r1",
                              "res2": "r2",
                              },
                             str_and_dict._label2componentsdict(self.w[9]),
                             self.w[9])

    def test_10(self):
        self.assertDictEqual({"res1": "r1",
                              "res2": "r2",
                              "frag2":"f2",
                              },
                             str_and_dict._label2componentsdict(self.w[10]),
                             self.w[10])

    def test_11(self):
        self.assertDictEqual({"res1": "r1",
                              "res2": "r2",
                              "frag2": "f2-1",
                              },
                             str_and_dict._label2componentsdict(self.w[11]),
                             self.w[11])

    def test_12(self):
        self.assertDictEqual({"res1": "r1",
                              "res2": "r2",
                              "frag1": "f1-1-1",
                              "frag2": "f2-2-2",
                              },
                             str_and_dict._label2componentsdict(self.w[12]),
                             self.w[11])

    def test_dont_split(self):
        label = "K417@RBD-Cl-1115@ions"
        # wrong behavior
        self.assertDictEqual({'res1': 'K417',
                              'frag1': 'RBD-Cl',  # wrong frag1
                              'res2': '1115',  # wrong res2
                              'frag2': 'ions'
                              },
                             str_and_dict._label2componentsdict(label),
                             label)
        # right behavior
        self.assertDictEqual({'res1': 'K417',
                              'frag1': 'RBD',  # right frag1
                              'res2': 'Cl-1115',  # right res2
                              'frag2': 'ions'
                              },
                             str_and_dict._label2componentsdict(label, dont_split=["Cl-1115"]),
                             label)

class Test_splitlabel(unittest.TestCase):

    def setUp(self):
        self.w = {
            3: "r1@f1-r2",
            5: "r1@f1-r2@f2",
            6: "r1@f1-1-r2@f2",
            7: "r1@f1-r2@f2-2",
            8: "r1@f1-1-r2@f2-2",
            9: "r1-r2",
           10: "r1-r2@f2",
           11: "r1-r2@f2-1",
           12: "r1@f1-1-1-r2@f2-2-2"

        }

    def test_3(self):
        self.assertListEqual(["r1@f1","r2"], str_and_dict.splitlabel(self.w[3]))

    def test_5(self):
        self.assertListEqual(["r1@f1","r2@f2"], str_and_dict.splitlabel(self.w[5]))

    def test_6(self):
        self.assertListEqual(["r1@f1-1","r2@f2"], str_and_dict.splitlabel(self.w[6]))

    def test_7(self):
        self.assertListEqual(["r1@f1","r2@f2-2"], str_and_dict.splitlabel(self.w[7]))

    def test_8(self):
        self.assertListEqual(["r1@f1-1","r2@f2-2"], str_and_dict.splitlabel(self.w[8]))

    def test_9(self):
        self.assertListEqual(["r1","r2"], str_and_dict.splitlabel(self.w[9]))

    def test_10(self):
        self.assertListEqual(["r1","r2@f2"], str_and_dict.splitlabel(self.w[10]))

    def test_11(self):
        self.assertListEqual(["r1","r2@f2-1"], str_and_dict.splitlabel(self.w[11]))

    def test_12(self):
        self.assertListEqual(["r1@f1-1-1", "r2@f2-2-2"], str_and_dict.splitlabel(self.w[12]))

class Test_defrag(unittest.TestCase):

    def test_works(self):
        label = 'res1@frag1-res2@frag2'
        np.testing.assert_equal("res1-res2", str_and_dict.defrag_key(label))

    def test_works_one_missing_frag(self):
        label = 'res1@frag1-res2    '
        np.testing.assert_equal("res1-res2", str_and_dict.defrag_key(label))

    def test_works_no_frags(self):
        label = 'res1-res2    '
        np.testing.assert_equal("res1-res2", str_and_dict.defrag_key(label))

    def test_works_just_one(self):
        label = 'res1@frag1    '
        np.testing.assert_equal("res1", str_and_dict.defrag_key(label))

class Test_replace_regex_special_chars():
    def test_works(self):
        word ="[]()^"
        assert str_and_dict._replace_regex_special_chars(word,"!!!!!")

class Test_latex_mathmode(unittest.TestCase):

    def test_works(self):
        self.assertEqual("$\\mathrm{There's an }\\alpha\\mathrm{ and a }\\beta\\mathrm{ here, also C_200}$",
                         str_and_dict.latex_mathmode("There's an alpha and a beta here, also C_200"))

class Test_latex_superscript_one_fragment(unittest.TestCase):

    def test_works(self):
        self.assertEqual(str_and_dict._latex_superscript_one_fragment("GLU30@beta_2AR"),
                         "GLU30$^{\\beta\\mathrm{_2AR}}$")
    def test_no_fragment(self):
        self.assertEqual(str_and_dict._latex_superscript_one_fragment("GLU30"),"GLU30")
if __name__ == '__main__':
    unittest.main()

class Test_FilenameGenerator(unittest.TestCase):

    def test_just_runs(self):
        fn = str_and_dict.FilenameGenerator("beta2 Gs",3.5,"project","png", "dat",150,"ps")
        self.assertEqual(fn.fullpath_overall_fig, "project/beta2_Gs.overall@3.5_Ang.png")
        self.assertEqual(fn.fullpath_overall_excel, "project/beta2_Gs.overall@3.5_Ang.xlsx")
        self.assertEqual(fn.fullpath_overall_dat, "project/beta2_Gs.overall@3.5_Ang.dat")
        self.assertEqual(fn.fullpath_pdb, "project/beta2_Gs.overall@3.5_Ang.as_bfactors.pdb")
        self.assertEqual(fn.fullpath_matrix, "project/beta2_Gs.matrix@3.5_Ang.png")
        self.assertEqual(fn.fullpath_flare_vec, "project/beta2_Gs.flare@3.5_Ang.pdf")
        self.assertEqual(fn.table_ext,"dat")
        self.assertEqual(fn.graphic_dpi,150)
        self.assertEqual(fn.t_unit,"ps")
        self.assertEqual(fn.fname_per_site_table("NPY"),'project/beta2_Gs.NPY@3.5_Ang.dat')
        self.assertEqual(fn.fname_timetrace_fig("traj1"),'beta2_Gs.traj1.time_trace@3.5_Ang.png')

    def test_table_ext_None_raises(self):
        with self.assertRaises(ValueError):
            fn = str_and_dict.FilenameGenerator("beta2 Gs",3.5,"project","png",None,150,"ps")

    def test_keeps_svg(self):
        fn = str_and_dict.FilenameGenerator("beta2 Gs",3.5,"project","svg", "dat",150,"ps")
        self.assertEqual(fn.fullpath_flare_vec, "project/beta2_Gs.flare@3.5_Ang.svg")

class Test_print_wrap(unittest.TestCase):

    def test_just_prints(self):
        string = "AAAABBBB"
        b = _io.StringIO()
        with _contextlib.redirect_stdout(b):
            str_and_dict.print_wrap(string, 4)
        self.assertEqual(b.getvalue(), "AAAA\nBBBB\n")
    def test_just_returns(self):
        string = "AAAABBBB"
        wrapped = str_and_dict.print_wrap(string,4, just_return_string=True)
        self.assertEqual(wrapped, "AAAA\nBBBB")

class Test_intblocks_in_str(unittest.TestCase):

    def test_just_works(self):
        self.assertListEqual(str_and_dict.intblocks_in_str("GLU30@3.50-GDP396@frag1"), [30,3,50,396,1])
    def test_fails(self):
        with self.assertRaises(ValueError):
            str_and_dict.intblocks_in_str("GLU")









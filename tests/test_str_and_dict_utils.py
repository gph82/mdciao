import mdtraj as md
from os import path
import numpy as np

import unittest
from mdciao.filenames import filenames

from mdciao.utils import str_and_dict

from tempfile import TemporaryDirectory as _TDir

import pytest

from pandas import \
    DataFrame as _DF,\
    ExcelWriter as _XW

test_filenames = filenames()

class Test_get_sorted_trajectories(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.top_pdb)
        self.traj = md.load(test_filenames.traj_xtc, top=self.geom.top)
        self.traj_reverse = md.load(test_filenames.traj_xtc, top=self.geom.top)[::-1]

    def test_glob_with_pattern(self):
        str_and_dict.get_sorted_trajectories(path.join(test_filenames.example_path,"*.xtc"))

    def test_glob_with_filename(self):
        str_and_dict.get_sorted_trajectories(test_filenames.traj_xtc)

    def test_with_one_trajectory_object(self):
        list_out = str_and_dict.get_sorted_trajectories(self.traj)
        assert len(list_out)==1
        assert isinstance(list_out[0], md.Trajectory)

    def test_with_trajectory_objects(self):
        str_and_dict.get_sorted_trajectories([self.traj,
                                 self.traj_reverse])


    def test_fails_if_not_trajs(self):
        with pytest.raises(AssertionError):
            str_and_dict.get_sorted_trajectories([self.traj,
                                     1])


class Test_inform_about_trajectories(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.top_pdb)
        self.traj = md.load(test_filenames.traj_xtc, top=self.geom.top)
        self.traj_reverse = md.load(test_filenames.traj_xtc, top=self.geom.top)[::-1]

    def test_fails_no_list(self):
        with pytest.raises(AssertionError):
            str_and_dict.inform_about_trajectories(test_filenames.traj_xtc)

    def test_list_of_files(self):
        str_and_dict.inform_about_trajectories([test_filenames.traj_xtc,
                                                test_filenames.traj_xtc])

    def test_list_of_trajs(self):
        str_and_dict.inform_about_trajectories([self.traj,
                                                self.traj_reverse])


class Test_replace_w_dict(unittest.TestCase):
    outkey = str_and_dict.replace_w_dict("key I don't like", {"key": "word", "like": "love"})
    assert outkey=="word I don't love"

class Test_delete_exp_inkeys(unittest.TestCase):
    indict = {"GLU30-LYS40":True,
              "ARG40-GLU30":False}

    outdict = str_and_dict.delete_exp_in_keys(indict, "GLU30")
    assert len(outdict)==2
    assert outdict["LYS40"]
    assert not outdict["ARG40"]


class Test_iterate_and_inform_lambdas(unittest.TestCase):

    def setUp(self):
        self.filename = test_filenames.traj_xtc
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
        with pytest.raises(NotImplementedError):
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

class Test_str_latex(unittest.TestCase):

    def setUp(self):
        self.greek_letters=["alpha", "beta", "gamma", "mu"]
        self.sub_supers = ["C_2", "C^2"]

    def test_works(self):
        import numpy as _np
        _np.testing.assert_array_equal(str_and_dict.replace4latex("There's an alpha and a beta here, also C_2"),
                                    "There's an $\\alpha$ and a $\\beta$ here, also $C_2$")

class Test_auto_fragment_string(unittest.TestCase):

    def test_both_bad(self):
        assert str_and_dict.choose_between_good_and_better_strings(None, None) == ""

    def test_both_good(self):
        assert str_and_dict.choose_between_good_and_better_strings("fragA", "3.50") == "3.50"

    def test_only_option(self):
        self.assertEquals(str_and_dict.choose_between_good_and_better_strings("fragA", None),"fragA")

    def test_only_better_option(self):
        assert str_and_dict.choose_between_good_and_better_strings(None, "3.50") == "3.50"

    def test_pick_best_label_just_works(self):
        assert (str_and_dict.choose_between_good_and_better_strings("Print this instead", "Print this") == "Print this")

    def test_pick_best_label_exclude_works(self):
        assert(str_and_dict.choose_between_good_and_better_strings("Print this instead", None) == "Print this instead")
        assert(str_and_dict.choose_between_good_and_better_strings("Print this instead", "None") == "Print this instead")
        assert(str_and_dict.choose_between_good_and_better_strings("Print this instead", "NA") == "Print this instead")
        assert(str_and_dict.choose_between_good_and_better_strings("Print this instead", "na") == "Print this instead")

class Test_freq_file2dict(unittest.TestCase):

    def setUp(self):
        self.freqs = {"0-1":0.3, "0-2":.4}


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

if __name__ == '__main__':
    unittest.main()
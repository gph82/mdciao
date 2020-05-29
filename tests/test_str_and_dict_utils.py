import mdtraj as md
from os import path
import numpy as np

import unittest
from filenames import filenames

from mdciao.str_and_dict_utils import \
    get_sorted_trajectories, \
    _inform_about_trajectories, \
    _replace_w_dict,\
    _delete_exp_in_keys, \
    iterate_and_inform_lambdas, \
    unify_freq_dicts,\
    _replace4latex, \
    choose_between_good_and_better_strings, \
    freq_ascii2dict

from mdciao import str_and_dict_utils

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
        get_sorted_trajectories(path.join(test_filenames.example_path,"*.xtc"))

    def test_glob_with_filename(self):
        get_sorted_trajectories(test_filenames.traj_xtc)

    def test_with_one_trajectory_object(self):
        list_out = get_sorted_trajectories(self.traj)
        assert len(list_out)==1
        assert isinstance(list_out[0], md.Trajectory)

    def test_with_trajectory_objects(self):
        get_sorted_trajectories([self.traj,
                                 self.traj_reverse])


    def test_fails_if_not_trajs(self):
        with pytest.raises(AssertionError):
            get_sorted_trajectories([self.traj,
                                     1])


class Test_inform_about_trajectories(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.top_pdb)
        self.traj = md.load(test_filenames.traj_xtc, top=self.geom.top)
        self.traj_reverse = md.load(test_filenames.traj_xtc, top=self.geom.top)[::-1]

    def test_fails_no_list(self):
        with pytest.raises(AssertionError):
            _inform_about_trajectories(test_filenames.traj_xtc)

    def test_list_of_files(self):
        _inform_about_trajectories([test_filenames.traj_xtc,
                                    test_filenames.traj_xtc])

    def test_list_of_trajs(self):
        _inform_about_trajectories([self.traj,
                                    self.traj_reverse])


class Test_replace_w_dict(unittest.TestCase):
    outkey = _replace_w_dict("key I don't like",  {"key":"word", "like":"love"})
    assert outkey=="word I don't love"

class Test_delete_exp_inkeys(unittest.TestCase):
    indict = {"GLU30-LYS40":True,
              "ARG40-GLU30":False}

    outdict = _delete_exp_in_keys(indict, "GLU30")
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
        iterate, inform = iterate_and_inform_lambdas(self.filename,
                                                     10,
                                                     top=self.top)
        self._call_iterators_and_test_them(iterate, inform, self.filename, 1)

    def test_filename_no_stride_filename_is_ascii_just_works(self):
        iterate, inform = iterate_and_inform_lambdas(self.pdb,
                                                     10,
                                                     top=self.top)
        nf = 0
        for ii, chunk in enumerate(iterate(self.pdb)):
            nf += chunk.n_frames
            inform(self.pdb,0,ii,nf)
        assert nf==1
        assert ii==0


    def test_filename_w_stride(self):
        iterate, inform = iterate_and_inform_lambdas(self.filename,
                                                     10,
                                                     stride=self.stride,
                                                     top=self.top)

        self._call_iterators_and_test_them(iterate, inform, self.filename,
                                           stride = self.stride)

    def test_traj_no_stride(self):
        iterate, inform = iterate_and_inform_lambdas(self.traj,
                                                     10,
                                                     )

        self._call_iterators_and_test_them(iterate, inform, self.traj)

    def test_traj_w_stride(self):
        iterate, inform = iterate_and_inform_lambdas(self.traj,
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
    def test_basic(self):
        out_dict = unify_freq_dicts(self.freq_dicts,
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

    def test_basic_exclude(self):
        with pytest.raises(NotImplementedError):
            out_dict = unify_freq_dicts(self.freq_dicts,
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
        out_dict = unify_freq_dicts(self.freq_dicts,
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
        out_dict = unify_freq_dicts(self.freq_dicts,
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

class Test_str_latex(unittest.TestCase):

    def setUp(self):
        self.greek_letters=["alpha", "beta", "gamma", "mu"]
        self.sub_supers = ["C_2", "C^2"]

    def test_works(self):
        import numpy as _np
        _np.testing.assert_array_equal(_replace4latex("There's an alpha and a beta here, also C_2"),
                                    "There's an $\\alpha$ and a $\\beta$ here, also $C_2$")

class Test_auto_fragment_string(unittest.TestCase):

    def test_both_bad(self):
        assert choose_between_good_and_better_strings(None, None) == ""

    def test_both_good(self):
        assert choose_between_good_and_better_strings("fragA", "3.50") == "3.50"

    def test_only_option(self):
        assert choose_between_good_and_better_strings("fragA", None) == "fragA", choose_between_good_and_better_strings("fragA", None)

    def test_only_better_option(self):
        assert choose_between_good_and_better_strings(None, "3.50") == "3.50"

    def test_pick_best_label_just_works(self):
        assert (choose_between_good_and_better_strings("Print this instead", "Print this") == "Print this")

    def test_pick_best_label_exclude_works(self):
        assert(choose_between_good_and_better_strings("Print this instead", None) == "Print this instead")
        assert(choose_between_good_and_better_strings("Print this instead", "None") == "Print this instead")
        assert(choose_between_good_and_better_strings("Print this instead", "NA") == "Print this instead")
        assert(choose_between_good_and_better_strings("Print this instead", "na") == "Print this instead")

class Test_freq_file2dict(unittest.TestCase):

    def setUp(self):
        self.freqs = {"0-1":0.3, "0-2":.4}


    def test_works_w_ascii(self):
        with _TDir(suffix="_test_mdciao") as tempdir:
            tmpfile = path.join(tempdir, "frqfile.dat")
            with open(tmpfile,"w") as f:
                for key, val in self.freqs.items():
                    f.write("%f %s\n"%(val, key))

            freqsin = str_and_dict_utils.freq_file2dict(tmpfile)
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

            freqsin = str_and_dict_utils.freq_file2dict(tmpfile)
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

            freqsin = str_and_dict_utils.freq_file2dict(tmpfile)
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
        filtered = str_and_dict_utils.fnmatch_ex(self.patterns,names)
        self.assertEqual(filtered,["John Perez", "Johnathan Hernandez","Guille Doe"])

    def test_match_dict_by_patterns_works(self):
        matching_keys, matching_values = str_and_dict_utils.match_dict_by_patterns(self.patterns,self.dict,
                                                      verbose=True)
        self.assertEqual(matching_keys,["John Perez", "Johnathan Hernandez","Guille Doe"])
        np.testing.assert_array_equal(matching_values,np.arange(6))
        # pick all FN John, LN Doe but avoid Johny

    def test_match_dict_by_patterns_empty(self):
        matching_keys, matching_values = str_and_dict_utils.match_dict_by_patterns("Maria",
                                                                                   self.dict,
                                                      verbose=True)
        self.assertEqual(matching_keys,[])
        np.testing.assert_array_equal(matching_values,[])
        # pick all FN John, LN Doe but avoid Johny

if __name__ == '__main__':
    unittest.main()
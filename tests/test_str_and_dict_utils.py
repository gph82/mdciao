import unittest
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
    _replace4latex

import pytest

test_filenames = filenames()

class Test_get_sorted_trajectories(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.prot1_pdb)
        self.run1_stride_100_xtc = md.load(test_filenames.run1_stride_100_xtc, top=self.geom.top)
        self.run1_stride_100_xtc_reverse = md.load(test_filenames.run1_stride_100_xtc, top=self.geom.top)[::-1]

    def test_glob_with_pattern(self):
        get_sorted_trajectories(path.join(test_filenames.examples_path,"*.xtc"))

    def test_glob_with_filename(self):
        get_sorted_trajectories(test_filenames.run1_stride_100_xtc)

    def test_with_one_trajectory_object(self):
        list_out = get_sorted_trajectories(self.run1_stride_100_xtc)
        assert len(list_out)==1
        assert isinstance(list_out[0], md.Trajectory)

    def test_with_trajectory_objects(self):
        get_sorted_trajectories([self.run1_stride_100_xtc,
                                 self.run1_stride_100_xtc_reverse])


    def test_fails_if_not_trajs(self):
        with pytest.raises(AssertionError):
            get_sorted_trajectories([self.run1_stride_100_xtc,
                                     1])


class Test_inform_about_trajectories(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.prot1_pdb)
        self.run1_stride_100_xtc = md.load(test_filenames.run1_stride_100_xtc, top=self.geom.top)
        self.run1_stride_100_xtc_reverse = md.load(test_filenames.run1_stride_100_xtc, top=self.geom.top)[::-1]

    def test_fails_no_list(self):
        with pytest.raises(AssertionError):
            _inform_about_trajectories(test_filenames.run1_stride_100_xtc)

    def test_list_of_files(self):
        _inform_about_trajectories([test_filenames.run1_stride_100_xtc,
                                    test_filenames.run1_stride_100_xtc])

    def test_list_of_trajs(self):
        _inform_about_trajectories([self.run1_stride_100_xtc,
                                    self.run1_stride_100_xtc_reverse])


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
        self.filename = test_filenames.run1_stride_100_xtc
        self.top = md.load(test_filenames.prot1_pdb).top
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

    def test_letters(self):
        for istr in self.greek_letters:
            assert _replace4latex(istr)=="$\\%s$"%istr, (istr,_replace4latex(istr))

    def test_sub_supers(self):
        for istr in self.sub_supers:
            assert _replace4latex(istr)=="$%s$"%istr, (istr,_replace4latex(istr))

    def test_notimplemented(self):
        with pytest.raises(NotImplementedError):
            _replace4latex("alpha C_2")

if __name__ == '__main__':
    unittest.main()
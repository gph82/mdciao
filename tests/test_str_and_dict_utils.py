import unittest
import mdtraj as md
from os import path

import unittest
from filenames import filenames

from mdciao.str_and_dict_utils import \
    get_sorted_trajectories, \
    _inform_about_trajectories, \
    _replace_w_dict,\
    _delete_exp_in_keys

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
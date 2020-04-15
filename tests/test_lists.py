import unittest
import numpy as _np
import pytest
from unittest import mock
from unittest.mock import patch
import io

from mdciao.list_utils import exclude_same_fragments_from_residx_pairlist, \
    unique_list_of_iterables_by_tuple_hashing, in_what_fragment, \
    does_not_contain_strings, force_iterable, is_iterable, in_what_N_fragments, rangeexpand, \
    pull_one_up_at_this_pos, assert_min_len, assert_no_intersection, window_average_fast, \
    put_this_idx_first_in_pair, re_warp, join_lists

#TODO correct the "just works" nomenclature and the repetition of names of the funciton in class methods
class Test_exclude_same_fragments_from_residx_pairlist(unittest.TestCase):

    def test_exclude_same_fragments_from_residx_pairlist_just_works(self):
        assert (exclude_same_fragments_from_residx_pairlist([[0, 1], [2, 3]], [[0, 1, 2], [3, 4]]) == [[2, 3]])

    def test_exclude_same_fragments_from_residx_pairlist_return_excluded_id(self):
        assert (exclude_same_fragments_from_residx_pairlist([[1, 2], [0, 3], [5, 6]], [[0, 1, 2], [3, 4], [5, 6]],
                                                        return_excluded_idxs=True)
            == [0, 2])

class Test_unique_list_of_iterables_by_tuple_hashing(unittest.TestCase):

    def test_works(self):
        assert (unique_list_of_iterables_by_tuple_hashing([1])) == [1]
        assert (unique_list_of_iterables_by_tuple_hashing([["A"], ["B"]]) ==
                unique_list_of_iterables_by_tuple_hashing([["A"], ["B"]]))

    def test_returns_index(self):
        assert (unique_list_of_iterables_by_tuple_hashing([1], return_idxs=True)) == [0]
        assert (unique_list_of_iterables_by_tuple_hashing([[1], [1], [2], [2]], return_idxs=True)) == [[0], [2]]

    def test_works_for_non_iterables(self):
        assert (unique_list_of_iterables_by_tuple_hashing([[1, 2], [3, 4], 1]) ==
                unique_list_of_iterables_by_tuple_hashing([[1, 2], [3, 4], _np.array(1)]))

    def test_reverse_is_not_same(self):
        assert not (unique_list_of_iterables_by_tuple_hashing([[1, 2], [3, 4]]) ==
                    unique_list_of_iterables_by_tuple_hashing([[2, 1], [3, 4]]))
        assert not (unique_list_of_iterables_by_tuple_hashing([["ABC"], ["BCD"]]) ==
                    unique_list_of_iterables_by_tuple_hashing([["BCD"], ["ABC"]]))

class Test_in_what_fragment(unittest.TestCase):

    def test_works(self):
        # Easiest test
        assert in_what_fragment(1, [[1, 2], [3, 4, 5, 6.6]]) == 0
        # Check that it returns the right fragments
        assert in_what_fragment(1, [[1, 2], [3, 4, 5, 6.6]], ["A", "B"]) == 'A'

    def test_should_be_integer(self):
        # Check that it fails when input is not an index
        with pytest.raises(AssertionError):
            in_what_fragment([1],[[1, 2], [3, 4, 5, 6.6]])

        with pytest.raises(AssertionError):
            in_what_fragment([1], [[1, 2], [3, 4, 5, 6.6, "A"]])

class Test_does_not_contain_strings(unittest.TestCase):

    def test_works(self):
        assert does_not_contain_strings([])
        assert does_not_contain_strings([9, 99, 999])
        assert does_not_contain_strings([9])
        assert not does_not_contain_strings(["A"])
        assert not does_not_contain_strings(["a", "b", "c"])
        assert not does_not_contain_strings(["A", "b", "c"])
        assert not does_not_contain_strings([[1], "ABC"])

class Test_force_iterable(unittest.TestCase):

    def test_works(self):
        assert len(force_iterable("A")) != 0
        assert len(force_iterable("abc")) != 0
        assert len(force_iterable([9, 99, 999])) != 0
        assert len(force_iterable(999)) != 0

class Test_is_iterable(unittest.TestCase):

    def test_is_iterable(self):
        assert is_iterable([]), is_iterable([])
        assert is_iterable("A")
        assert is_iterable("abc")
        assert is_iterable([9, 99, 999])
        assert not is_iterable(999)

class Test_in_what_N_fragments(unittest.TestCase):

    def setUp(self):
        self.fragments = [[0, 20, 30, 40],
                    [2, 4, 31, 1000],
                    [10]]

    def test_works(self):
        for fragment_idx, idx_query in enumerate([20, 4, 10]):
            assert len(in_what_N_fragments(idx_query, self.fragments)) == 1

    def test_in_what_N_fragments_for_wrong_idx(self):
        wrong_idx = 11
        assert len(in_what_N_fragments(wrong_idx, self.fragments)) == 1

class Test_rangeexpand(unittest.TestCase):
    def test_rangeexpand_just_works(self):
        assert (rangeexpand("1-2, 3-4") == [1, 2, 3, 4])
        assert (rangeexpand("1-2, 3,4") == [1, 2, 3, 4])
        assert (rangeexpand("1-2, 03, 4") == [1, 2, 3, 4])

class Test_pull_one_up_at_this_pos(unittest.TestCase):
    def test_pull_one_up_at_this_pos_just_works(self):
        assert(pull_one_up_at_this_pos([1,2,3],1,"~") == [1, 3, '~'])
        assert(pull_one_up_at_this_pos("string",1,"~") == ['s', 'r', 'i', 'n', 'g', '~'])
        assert(pull_one_up_at_this_pos([1,2,3],1,10) == [1, 3, 10])
        assert(pull_one_up_at_this_pos([1,2,3],1,[99]) == [1, 3, [99]])

    def test_pull_one_up_at_this_pos_verbose_works(self):
        assert(pull_one_up_at_this_pos([1,2,3],1,"~",verbose=True) == [1, 3, '~'])

class Test_assert_min_len(unittest.TestCase):
    def test_assert_min_len_just_works(self):
        assert_min_len([['a', 'b'], ['c', 'd'],[1, 2]])

    def test_assert_min_len_failed_assertion_just_works(self):
       with pytest.raises(AssertionError):
           assert_min_len([[1]])

    def test_assert_min_len_failed_assertion_works_empty_list(self):
        with pytest.raises(AssertionError):
            assert_min_len([[1,2],[]])

    def test_assert_min_length_min_len_works(self):
        assert_min_len([['a']], min_len=1)

class Test_assert_no_intersection(unittest.TestCase):
    def test_assert_no_intersection_just_works(self):
        assert_no_intersection([[1, 2], [3, 3]])

    def test_assert_no_intersection_empty_list_just_works(self):
        assert_no_intersection([[], [3, 3]])

    def test_failed_assertion_just_works(self):
        with pytest.raises(AssertionError):
            assert_no_intersection([[1,2,3],[3,3]])

    def test_assert_no_intersection_empty_lists(self):
        with pytest.raises(AssertionError):
            assert_no_intersection([[], []])

class Test_window_average_fast(unittest.TestCase):
    def test_window_average_fast_just_works(self):
        assert _np.allclose(window_average_fast(_np.arange(5)), _np.array([2.0]))
        assert _np.allclose(window_average_fast(_np.arange(10)), _np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))

    def test_window_average_fast_half_window_size_works(self):
        assert _np.allclose(window_average_fast(_np.arange(7), half_window_size=3), _np.array([3.0]))
        assert _np.allclose(window_average_fast(_np.arange(5), half_window_size=3), _np.array([1.42857143, 1.42857143, 1.42857143]))

class Test_join_lists(unittest.TestCase):
    def test_simple_run(self):
        in_lists = [[0, 1], [2, 3], [4, 5], [6, 7]]
        joined_lists = join_lists(in_lists, [[1, 2]])
        assert _np.allclose(joined_lists[0], [0, 1])
        assert _np.allclose(joined_lists[1], [2, 3, 4, 5]), (joined_lists[1])
        assert _np.allclose(joined_lists[2], [6, 7]), (joined_lists[2])

    def test_re_ordered_run(self):
        in_lists = [[0, 1], [2, 3], [4, 5], [6, 7]]
        joined_lists = join_lists(in_lists, [[3, 0],
                                             [1, 2]])
        assert _np.allclose(joined_lists[0], [0, 1, 6, 7])
        assert _np.allclose(joined_lists[1], [2, 3, 4, 5])


    def test_fails_overalpping_idxs(self):
        with pytest.raises(AssertionError):
            in_lists = [[0, 1], [2, 3], [4, 5], [6, 7]]
            join_lists(in_lists, [[1, 2],
                                  [2, 3]])

    def test_fails_no_minlen_list2join(self):
        with pytest.raises(AssertionError):
            in_lists = [[0, 1], [2, 3], [4, 5], [6, 7]]
            join_lists(in_lists, [[0],[1,2]])


class Test_put_this_idx_first_in_pair(unittest.TestCase):
    def test_works(self):
        assert(put_this_idx_first_in_pair(20, [10,20]) == [20,10])
        assert (put_this_idx_first_in_pair(10, [10, 20]) == [10, 20])
        assert (put_this_idx_first_in_pair("first", ["first", "last"]) == ["first", "last"])
        assert (put_this_idx_first_in_pair("first", ["last", "first"]) == ["first", "last"])

    def test_put_this_idx_first_in_pair_exception(self):
        with pytest.raises(Exception):
            put_this_idx_first_in_pair(99, [10, 20])

class Test_rewarp(unittest.TestCase):
    def test_int_input(self):
        assert all([_np.allclose(ii,jj) for ii,jj in zip([[0,1],
                                                          [2,3]],
                                                         re_warp([0,1,2,3], 2))])

    def test_short_input(self):
        assert _np.allclose(re_warp([0,1,2,3],[2]),[0,1])

if __name__ == '__main__':
    unittest.main()


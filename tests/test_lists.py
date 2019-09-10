import unittest
import numpy as _np

from sofi_functions.list_utils import exclude_same_fragments_from_residx_pairlist, \
    unique_list_of_iterables_by_tuple_hashing, in_what_fragment, \
    does_not_contain_strings, force_iterable, is_iterable, in_what_N_fragments, rangeexpand

class Test_exclude_same_fragments_from_residx_pairlist(unittest.TestCase):

    def test_exclude_same_fragments_from_residx_pairlist_just_works(self):
        assert (exclude_same_fragments_from_residx_pairlist([[0, 1], [2, 3]], [[0, 1, 2], [3, 4]]) == [[2, 3]])

    def test_exclude_same_fragments_from_residx_pairlist_return_excluded_id(self):
        assert (exclude_same_fragments_from_residx_pairlist([[1, 2], [0, 3], [5, 6]], [[0, 1, 2], [3, 4], [5, 6]],
                                                        return_excluded_idxs=True)
            == [0, 2])

class Test_unique_list_of_iterables_by_tuple_hashing(unittest.TestCase):

    def test_unique_list_of_iterables_by_tuple_hashing_just_works(self):
        assert (unique_list_of_iterables_by_tuple_hashing([1])) == [1]
        assert (unique_list_of_iterables_by_tuple_hashing([["A"], ["B"]]) ==
                unique_list_of_iterables_by_tuple_hashing([["A"], ["B"]]))

    def test_unique_list_of_iterables_by_tuple_hashing_returns_index(self):
        assert (unique_list_of_iterables_by_tuple_hashing([1], return_idxs=True)) == [0]
        assert (unique_list_of_iterables_by_tuple_hashing([[1], [1], [2], [2]], return_idxs=True)) == [[0], [2]]

    def test_unique_list_of_iterables_by_tuple_hashing_works_for_non_iterables(self):
        assert (unique_list_of_iterables_by_tuple_hashing([[1, 2], [3, 4], 1]) ==
                unique_list_of_iterables_by_tuple_hashing([[1, 2], [3, 4], _np.array(1)]))

    def test_unique_list_of_iterables_by_tuple_hashing_reverse_is_not_same(self):
        assert not (unique_list_of_iterables_by_tuple_hashing([[1, 2], [3, 4]]) ==
                    unique_list_of_iterables_by_tuple_hashing([[2, 1], [3, 4]]))
        assert not (unique_list_of_iterables_by_tuple_hashing([["ABC"], ["BCD"]]) ==
                    unique_list_of_iterables_by_tuple_hashing([["BCD"], ["ABC"]]))

class Test_in_what_fragment(unittest.TestCase):

    def test_in_what_fragment_just_works(self):
        # Easiest test
        assert in_what_fragment(1, [[1, 2], [3, 4, 5, 6.6]]) == 0
        # Check that it returns the right fragments
        assert in_what_fragment(1, [[1, 2], [3, 4, 5, 6.6]], ["A", "B"]) == 'A'

    def test_in_what_fragment_idxs_should_be_integer(self):
        # Check that it fails when input is not an index
        failed_assertion = False
        try:
            in_what_fragment([1],[[1, 2], [3, 4, 5, 6.6]])
        except AssertionError as e:
            failed_assertion = True
        assert failed_assertion

        # Check that it fails when strings are there
        failed_assertion = False
        try:
            in_what_fragment([1], [[1, 2], [3, 4, 5, 6.6, "A"]])
        except AssertionError as __:
            failed_assertion = True
        assert failed_assertion

class Test_does_not_contain_strings(unittest.TestCase):

    def test_does_not_contain_strings_just_works(self):
        assert does_not_contain_strings([])
        assert does_not_contain_strings([9, 99, 999])
        assert does_not_contain_strings([9])
        assert not does_not_contain_strings(["A"])
        assert not does_not_contain_strings(["a", "b", "c"])
        assert not does_not_contain_strings(["A", "b", "c"])
        assert not does_not_contain_strings([[1], "ABC"])

class Test_force_iterable(unittest.TestCase):

    def test_force_iterable_just_works(self):
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

    def test_in_what_N_fragments_just_works(self):
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

if __name__ == '__main__':
    unittest.main()


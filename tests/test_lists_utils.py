import unittest
import numpy as _np
from mdciao.examples import filenames as test_filenames


from mdciao.utils import lists
import mdtraj as md

#TODO correct the "just works" nomenclature and the repetition of names of the funciton in class methods
class Test_exclude_same_fragments_from_residx_pairlist(unittest.TestCase):

    def test_exclude_same_fragments_from_residx_pairlist_just_works(self):
        assert (lists.exclude_same_fragments_from_residx_pairlist([[0, 1], [2, 3]], [[0, 1, 2], [3, 4]]) == [[2, 3]])

    def test_exclude_same_fragments_from_residx_pairlist_return_excluded_id(self):
        assert (lists.exclude_same_fragments_from_residx_pairlist([[1, 2], [0, 3], [5, 6]], [[0, 1, 2], [3, 4], [5, 6]],
                                                        return_excluded_idxs=True)
            == [0, 2])

class Test_unique_list_of_iterables_by_tuple_hashing(unittest.TestCase):

    def test_works(self):
        self.assertListEqual(lists.unique_list_of_iterables_by_tuple_hashing(
            [[0, 1], [1, 0], "ABC", [3, 4], [0, 1], "ABC", "DEF", [0, 1, 2]]),
            [[0, 1], [1, 0], "ABC", [3, 4],                "DEF", [0, 1, 2]])

    def test_works_array(self):
        [_np.testing.assert_array_equal(ii,jj) for ii, jj in zip(
            lists.unique_list_of_iterables_by_tuple_hashing(
            [[0, 1], [1, 0], "ABC", [3, 4], [0, 1], "ABC", "DEF", _np.array([0, 1, 2])]),
            [[0, 1], [1, 0], "ABC", [3, 4],                "DEF", [0, 1, 2]])]

    def test_returns_index(self):
        self.assertListEqual(lists.unique_list_of_iterables_by_tuple_hashing(
            [[0, 1], [1, 0], "ABC", [3, 4], [0, 1], "ABC", "DEF", [0, 1, 2]], return_idxs=True),
             [0, 1, 2, 3, 6,7])

    def test_works_for_non_iterables(self):
        self.assertListEqual(lists.unique_list_of_iterables_by_tuple_hashing([[1, 2], [3, 4], 1]),
                             [[1,2], [3,4],[1]])

    def test_bad_input(self):
        self.assertListEqual(["A"],lists.unique_list_of_iterables_by_tuple_hashing(["A"]))
        self.assertListEqual([0], lists.unique_list_of_iterables_by_tuple_hashing(["A"], return_idxs=True))

    def test_ignore_order_True(self):
        self.assertListEqual(lists.unique_list_of_iterables_by_tuple_hashing(
            [[0, 1], [1, 0], "ABC", [3, 4], [0, 1], "ABC", "DEF", [0, 1, 2]], ignore_order=True),
            [[0, 1],         "ABC", [3, 4],                "DEF", [0, 1, 2]])

class Test_in_what_fragment(unittest.TestCase):

    def test_works(self):
        # Easiest test
        assert lists.in_what_fragment(1, [[1, 2], [3, 4, 5, 6.6]]) == 0
        # Check that it returns the right fragments
        assert lists.in_what_fragment(1, [[1, 2], [3, 4, 5, 6.6]], ["A", "B"]) == 'A'

    def test_should_be_integer(self):
        # Check that it fails when input is not an index
        with self.assertRaises(AssertionError):
            lists.in_what_fragment([1],[[1, 2], [3, 4, 5, 6.6]])

        with self.assertRaises(AssertionError):
            lists.in_what_fragment([1], [[1, 2], [3, 4, 5, 6.6, "A"]])

class Test_does_not_contain_strings(unittest.TestCase):

    def test_works(self):
        assert lists.does_not_contain_strings([])
        assert lists.does_not_contain_strings([9, 99, 999])
        assert lists.does_not_contain_strings([9])
        assert not lists.does_not_contain_strings(["A"])
        assert not lists.does_not_contain_strings(["a", "b", "c"])
        assert not lists.does_not_contain_strings(["A", "b", "c"])
        assert not lists.does_not_contain_strings([[1], "ABC"])

class Test_force_iterable(unittest.TestCase):

    def test_works(self):
        assert len(lists.force_iterable("A")) != 0
        assert len(lists.force_iterable("_abc")) != 0
        assert len(lists.force_iterable([9, 99, 999])) != 0
        assert len(lists.force_iterable(999)) != 0

class Test_is_iterable(unittest.TestCase):

    def test_is_iterable(self):
        assert lists.is_iterable([]), lists.is_iterable([])
        assert lists.is_iterable("A")
        assert lists.is_iterable("_abc")
        assert lists.is_iterable([9, 99, 999])
        assert not lists.is_iterable(999)

class Test_in_what_N_fragments(unittest.TestCase):

    def setUp(self):
        self.fragments = [[0, 20, 30, 40],
                    [2, 4, 31, 1000],
                    [10]]

    def test_works(self):
        for fragment_idx, idx_query in enumerate([20, 4, 10]):
            assert len(lists.in_what_N_fragments(idx_query, self.fragments)) == 1

    def test_in_what_N_fragments_for_wrong_idx(self):
        wrong_idx = 11
        assert len(lists.in_what_N_fragments(wrong_idx, self.fragments)) == 1

class Test_rangeexpand(unittest.TestCase):
    def test_rangeexpand_just_works(self):
        assert (lists.rangeexpand("1-2, 3-4") == [1, 2, 3, 4])
        assert (lists.rangeexpand("1-2, 3,4") == [1, 2, 3, 4])
        assert (lists.rangeexpand("1-2, 03, 4") == [1, 2, 3, 4])

class Test_assert_min_len(unittest.TestCase):
    def test_assert_min_len_just_works(self):
        lists.assert_min_len([['a', 'b'], ['c', 'd'],[1, 2]])

    def test_assert_min_len_failed_assertion_just_works(self):
       with self.assertRaises(AssertionError):
           lists.assert_min_len([[1]])

    def test_assert_min_len_failed_assertion_works_empty_list(self):
        with self.assertRaises(AssertionError):
            lists.assert_min_len([[1,2],[]])

    def test_assert_min_length_min_len_works(self):
        lists.assert_min_len([['a']], min_len=1)

class Test_assert_no_intersection(unittest.TestCase):
    def test_assert_no_intersection_just_works(self):
        lists.assert_no_intersection([[1, 2], [3, 3]])

    def test_assert_no_intersection_empty_list_just_works(self):
        lists.assert_no_intersection([[], [3, 3]])

    def test_failed_assertion_just_works(self):
        with self.assertRaises(AssertionError):
            lists.assert_no_intersection([[1,2,3],[3,3]])

    def test_assert_no_intersection_empty_lists(self):
        with self.assertRaises(AssertionError):
            lists.assert_no_intersection([[], []])

class Test_window_average_fast(unittest.TestCase):
    def test_window_average_fast_just_works(self):
        assert _np.allclose(lists.window_average_fast(_np.arange(5)), _np.array([2.0]))
        assert _np.allclose(lists.window_average_fast(_np.arange(10)), _np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))

    def test_window_average_fast_half_window_size_works(self):
        assert _np.allclose(lists.window_average_fast(_np.arange(7), half_window_size=3), _np.array([3.0]))
        assert _np.allclose(lists.window_average_fast(_np.arange(5), half_window_size=3), _np.array([1.42857143, 1.42857143, 1.42857143]))

class Test_join_lists(unittest.TestCase):
    def test_simple_run(self):
        in_lists = [[0, 1], [2, 3], [4, 5], [6, 7]]
        joined_lists = lists.join_lists(in_lists, [[1, 2]])
        assert _np.allclose(joined_lists[0], [0, 1])
        assert _np.allclose(joined_lists[1], [2, 3, 4, 5]), (joined_lists[1])
        assert _np.allclose(joined_lists[2], [6, 7]), (joined_lists[2])

    def test_re_ordered_run(self):
        in_lists = [[0, 1], [2, 3], [4, 5], [6, 7]]
        joined_lists = lists.join_lists(in_lists, [[3, 0],
                                             [1, 2]])
        assert _np.allclose(joined_lists[0], [0, 1, 6, 7])
        assert _np.allclose(joined_lists[1], [2, 3, 4, 5])


    def test_fails_overalpping_idxs(self):
        with self.assertRaises(AssertionError):
            in_lists = [[0, 1], [2, 3], [4, 5], [6, 7]]
            lists.join_lists(in_lists, [[1, 2],
                                  [2, 3]])

    def test_fails_no_minlen_list2join(self):
        with self.assertRaises(AssertionError):
            in_lists = [[0, 1], [2, 3], [4, 5], [6, 7]]
            lists.join_lists(in_lists, [[0],[1,2]])


class Test_put_this_idx_first_in_pair(unittest.TestCase):
    def test_works(self):
        assert(lists.put_this_idx_first_in_pair(20, [10,20]) == [20,10])
        assert (lists.put_this_idx_first_in_pair(10, [10, 20]) == [10, 20])
        assert (lists.put_this_idx_first_in_pair("first", ["first", "last"]) == ["first", "last"])
        assert (lists.put_this_idx_first_in_pair("first", ["last", "first"]) == ["first", "last"])

    def test_put_this_idx_first_in_pair_exception(self):
        with self.assertRaises(Exception):
            lists.put_this_idx_first_in_pair(99, [10, 20])

class Test_rewarp(unittest.TestCase):
    def test_int_input(self):
        assert all([_np.allclose(ii,jj) for ii,jj in zip([[0,1],
                                                          [2,3]],
                                                         lists.re_warp([0,1,2,3], 2))])

    def test_short_input(self):
        assert _np.allclose(lists.re_warp([0,1,2,3],[2]),[0,1])

class test_hash_list(unittest.TestCase):

    def test_passes(self):
        geom = md.load(test_filenames.small_monomer)
        ilist = [_np.random.randn(5000,2), # ndarray
                 [_np.random.randn(5000,2),_np.random.randn(5000,2)], #list thereof
                 'this_is_a_string', # str
                 ['this_is_a_substring', 'this_is_another_one'], # list thereof
                 1, # int
                 [1,2], #list of ints
                 [[1,],[2]], # list of lists,
                 geom,
                 geom.top,
                 [geom, geom],
                 [geom.top],
                 [geom, geom.top],
                 ]

        lists.hash_list(ilist)

class Test_contiguous_ranges(unittest.TestCase):

    def test_works(self):
        input = ["A","A", "A","0","A","A","0","0"]
        out_dict = lists.contiguous_ranges(input)
        print(out_dict)
        _np.testing.assert_array_equal(out_dict["A"][0],[0,1,2])
        _np.testing.assert_array_equal(out_dict["A"][1],[4,5])
        _np.testing.assert_equal(2,len(out_dict["A"]))
        _np.testing.assert_array_equal(out_dict["0"][0],[3])
        _np.testing.assert_array_equal(out_dict["0"][1],[6,7])
        _np.testing.assert_equal(2,len(out_dict["0"]))


class Test_idx_at_fraction(unittest.TestCase):

    def test_works(self):
        ncf = lists.idx_at_fraction([1, 1, 1], frac=2 / 3)
        _np.testing.assert_equal(ncf,1)

    def test_one_value(self):
        ncf = lists.idx_at_fraction([1], frac=2 / 3)
        _np.testing.assert_equal(ncf,0)

class Test_remove_from_lists(unittest.TestCase):
    def test_works(self):
        l = [
            [1, 2, 400, 6, 400, 6, 8],
            [400, 600, 700],
            [1, 2, 3, 4]
        ]
        nl = lists.remove_from_lists(l, [400,600,700])
        _np.testing.assert_array_equal(nl[0],[1,2,6,6,8])
        _np.testing.assert_array_equal(nl[1],[1,2,3,4])
    def test_weird(self):
        self.assertEqual([], lists.remove_from_lists([], [400, 600, 700]))
        self.assertEqual([], lists.remove_from_lists([[]], []))
        self.assertEqual([[400,500]], lists.remove_from_lists([[400,500]], []))

class Test_find_parent_list(unittest.TestCase):

    def test_works(self):
        fragments = [_np.arange(10),
                      _np.arange(50,100),
                       _np.arange(1000,2000)]
        subfragments = [[0,1,3], [4, 5, 6],
                        [10, 11, 12],
                        [60],
                        [1500,1600,1700],
                        [0]]
        parents_by_kid, kid_by_parents = lists.find_parent_list(subfragments, fragments)
        self.assertListEqual(parents_by_kid,[0,0, None,1,2,0])
        self.assertListEqual(list(kid_by_parents.keys()),[0,1,2])
        _np.testing.assert_array_equal(kid_by_parents[0],[0, 1, 5])
        _np.testing.assert_array_equal(kid_by_parents[1],[3])
        _np.testing.assert_array_equal(kid_by_parents[2],[4])

class Test_unique_product_w_intersection(unittest.TestCase):

    def test_works(self):
        a1 = [0,1,2,3]
        a2 = [2,3,4,5]

        res = lists.unique_product_w_intersection(a1,a2)

        self.assertListEqual(res.tolist(),
                                 [[0,2], #self a1-a1
                                  [0,3], #self a1-a1
                                  [0,4],
                                  [0,5],
                                  [1,2], #self a1-a1
                                  [1,3], #self a1-a1
                                  [1,4],
                                  [1,5],
                                  [2,3],
                                  [2,4],
                                  [2,5],
                                  [3,4],
                                  [3,5],
                                  ])

if __name__ == '__main__':
    unittest.main()


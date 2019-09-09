
import mdtraj as md
import unittest
import numpy as _np
from unittest.mock import patch
import mock

from sofi_functions.tested_utils import find_AA, top2residue_bond_matrix, get_fragments, \
    interactive_fragment_picker_by_AAresSeq,exclude_same_fragments_from_residx_pairlist,\
    unique_list_of_iterables_by_tuple_hashing, in_what_fragment,does_not_contain_strings, force_iterable, \
    is_iterable, in_what_N_fragments, int_from_AA_code, bonded_neighborlist_from_top, rangeexpand,\
    ctc_freq_reporter_by_residue_neighborhood, table2BW_by_AAcode, guess_missing_BWs, CGN_transformer, \
    top2CGN_by_AAcode, xtcs2ctcs, interactive_fragment_picker_wip

#OR import sofi_functions

#and then you call evertying as sofi_functions.xxxx
class Test_find_by_AA(unittest.TestCase):


    def setUp(self):
        self.geom = md.load("PDB/file_for_test.pdb")
        self.geom2frags = md.load("PDB/file_for_test_repeated_fullresnames.pdb")

    def test_it_just_works_with_long_AA_code(self):
        assert (find_AA(self.geom.top, "GLU30")) == [0]
        assert (find_AA(self.geom.top, "LYS28")) == [5]

    def test_it_just_works_with_short_AA_code(self):
        assert (find_AA(self.geom.top, 'E30')) == [0]
        assert (find_AA(self.geom.top, 'W32')) == [2]

    def test_does_not_find_AA(self):
        assert (find_AA(self.geom.top, "lys20")) == []   # small case won't give any result
        assert (find_AA(self.geom.top, 'w32')) == []    # small case won't give any result
        assert (find_AA(self.geom.top, 'w 32')) == []   # spaces between characters won't work

    def test_malformed_input(self):
        failed_assertion = False
        try:
            find_AA(self.geom.top, "GLUTAMINE")
        except ValueError as __:
            failed_assertion = True
        assert failed_assertion
    def test_ambiguity(self):
        # AMBIGUOUS definition i.e. each residue is present in multiple fragments
        assert (find_AA(self.geom2frags.top, "LYS28")) == [5, 13] # getting multiple idxs,as expected
        assert (find_AA(self.geom2frags.top, "K28")) == [5, 13]


class Test_top2residue_bond_matrix(unittest.TestCase):

    def setUp(self):
        self.geom = md.load("PDB/file_for_test.pdb")
        self.geom_force_resSeq_breaks = md.load("PDB/file_for_test_force_resSeq_breaks_is_true.pdb")

    def test_it_just_works_with_top2residue_bond_matrix(self):
        res_bond_matrix = _np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 0, 0], #LYS28 has a bond with GLU27
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0]])
        assert (top2residue_bond_matrix(self.geom.top) == res_bond_matrix).all()

    def test_works_with_force_resSeq_breaks_is_true(self):
        res_bond_matrix = _np.array([ [1, 1, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 1, 0, 0, 0, 0, 0],
                                      [0, 1, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0], #LYS28 has been changed to LYS99 in the test file, so no bond with GLU27
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]])

        assert (top2residue_bond_matrix(self.geom_force_resSeq_breaks.top, force_resSeq_breaks=True) == res_bond_matrix).all()

class Test_interactive_fragment_picker_no_ambiguity(unittest.TestCase):

    def setUp(self):
        self.geom = md.load("PDB/file_for_test.pdb")
        self.by_bonds_geom = get_fragments(self.geom.top,
                                           verbose=True,
                                           auto_fragment_names=True,
                                           method='bonds')

    def test_interactive_fragment_picker_by_AAresSeq_no_ambiguous(self):
        residues = ["GLU30", "GDP382"]
        resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom,
                                                                                  self.geom.top)
        # Checking if residue names gives the correct corresponding residue id
        assert (resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
        assert (resname2residx["GDP382"]) == 7  # GDP382 is the 8th residue

        # Checking if the residue name give the correct corresponding fragment id
        assert (resname2fragidx["GLU30"]) == 0  # GLU30 is in the 1st fragment
        assert (resname2fragidx["GDP382"]) == 3  # GDP382 is in the 4th fragment


    def test_interactive_fragment_picker_by_AAresSeq_not_present(self):
        residues = ["Glu30"]
        resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom,
                                                                                  self.geom.top)
        assert (resname2residx["Glu30"] == None)
        assert (resname2fragidx["Glu30"] == None)

class Test_interactive_fragment_picker_with_ambiguity(unittest.TestCase):

    def setUp(self):
        self.geom2frags = md.load("PDB/file_for_test_repeated_fullresnames.pdb")

        self.by_bonds_geom2frags = get_fragments(self.geom2frags.top,
                                                 verbose=True,
                                                 auto_fragment_names=True,
                                                 method='bonds')


    @patch('builtins.input', lambda *args: '4')
    def test_interactive_fragment_picker_by_AAresSeq_default_fragment_idx_is_none(self):
        residues = ["GLU30"]
        resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                                  self.geom2frags.top)

        # Checking if residue names gives the correct corresponding residue id
        # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
        assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th residue
        assert resname2fragidx["GLU30"] == 4 # GLU30 is in the 4th fragment

    @patch('builtins.input', lambda *args: "\n")
    def _test_interactive_fragment_picker_by_AAresSeq_default_fragment_idx_is_none_last_answer(self):
        residues = ["GLU30"]
        resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                                  self.geom2frags.top)

        # Checking if residue names gives the correct corresponding residue id
        # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
        assert (resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
        assert resname2fragidx["GLU30"] == 0 # GLU30 is in the 1st fragment

    #TODO JUST TRYING

    # def test_interactive_fragment_picker_by_AAresSeq_default_fragment_idx_is_none_last_answer(self):
    #     with mock.patch('builtins.input',return_value = '4'):
    #         resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq("GLU30", self.by_bonds_geom2frags,
    #                                                                               self.geom2frags.top)
    #         assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th residue
    #         assert resname2fragidx["GLU30"] == 4 # GLU30 is in the 4th fragment
    #
    #     with mock.patch('builtins.input',return_value = ''):
    #         resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq("VAL31", self.by_bonds_geom2frags,
    #                                                                               self.geom2frags.top)
    #         assert (resname2residx["VAL31"]) == 9  # VAL31 is the 9th residue
    #         assert (resname2fragidx["VAL31"]) == 4  # VAL31 is in the 4th fragment



    @patch('builtins.input', lambda *args: "xyz")
    def test_interactive_fragment_picker_by_AAresSeq_default_fragment_idx_is_none_ans_should_be_int(self):
        residues = ["GLU30"]

        failed_assertion = False
        try:
            interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                                  self.geom2frags.top)
        except (ValueError,AssertionError):
            failed_assertion = True
        assert failed_assertion

    @patch('builtins.input', lambda *args: "123")
    def test_interactive_fragment_picker_by_AAresSeq_default_fragment_idx_is_none_ans_should_be_in_list(self):
        residues = ["GLU30"]

        failed_assertion = False
        try:
            interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                                  self.geom2frags.top)
        except (ValueError,AssertionError):
            failed_assertion = True
        assert failed_assertion


    def test_interactive_fragment_picker_by_AAresSeq_default_fragment_idx_is_passed(self):
        residues = ["GLU30"]
        resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                                  self.geom2frags.top,default_fragment_idx=4)

        # Checking if residue names gives the correct corresponding residue id
        # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
        assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th residue
        assert (resname2fragidx["GLU30"]) == 4 # GLU30 is in the 4th fragment

    def test_interactive_fragment_picker_by_AAresSeq_default_fragment_idx_is_passed_special_case(self):
        residues = ["GLU30"]

        failed_assertion = False
        try:
            resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                                  self.geom2frags.top,default_fragment_idx=99)
        except (ValueError,AssertionError):
            failed_assertion = True
        assert failed_assertion


    def test_interactive_fragment_picker_by_AAresSeq_ambiguous(self):
        with mock.patch('builtins.input', return_value ='4'):
            resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq("GLU30",
                                                                                      self.by_bonds_geom2frags,
                                                                                      self.geom2frags.top)

            assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th
            assert (resname2fragidx["GLU30"]) == 4  # GLU30 is in the 4th fragment
        with mock.patch('builtins.input', return_value ='3'):
            resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq("GDP382",
                                                                                      self.by_bonds_geom2frags,
                                                                                      self.geom2frags.top)

            assert (resname2residx["GDP382"]) == 7  # GDP382 is the 7th residue
            assert (resname2fragidx["GDP382"]) == 3  # GDP382 is the 3rd fragment


    # def _test_interactive_fragment_picker_by_AAresSeq_pick_last_answer(self):
    #     residues = ["GLU30", "VAL31"]
    #     resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues,
    #                                                                               self.by_bonds_geom2frags,
    #                                                                               self.geom2frags.top)
    #     # Checking if residue names gives the correct corresponding residue id
    #     # NOTE:Just press Return for GLU30 when asked "input one fragment idx"
    #     # NOTE:Just press Return for VAL31, when asked to "input one fragment idx"
    #
    #     assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th
    #     assert (resname2fragidx["GLU30"]) == 4  # GLU30 is in the 4th fragment
    #
    #     assert (resname2residx["VAL31"]) == 9  # VAL31 is the 9th residue
    #     assert (resname2fragidx["VAL31"]) == 4  # VAL31 is in the 4th fragment

    @patch('builtins.input', lambda *args: '0')
    def test_interactive_fragment_picker_by_AAresSeq_fragment_name(self):
        residues = ["GLU30"]
        resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                                  self.geom2frags.top,
                                                                                  fragment_names=["A", "B", "C", "D",
                                                                                                  "E", "F", "G", "H"])
        # Checking if residue names gives the correct corresponding residue id
        # NOTE:Enter 0 for GLU30 when asked "input one fragment idx"

        assert (resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
        assert resname2fragidx["GLU30"] == 0  # GLU30 is in the 1st fragment


class Test_get_fragments(unittest.TestCase):
    def setUp(self):
        self.geom = md.load('PDB/file_for_test.pdb')
        self.geom_force_resSeq_breaks = md.load("PDB/file_for_test_force_resSeq_breaks_is_true.pdb")

    # Checking for "method" argument (which are resSeq and Bonds
    def test_get_fragments_method(self):
        by_resSeq = get_fragments(self.geom.top,
                                  verbose=True,
                                  auto_fragment_names=True,
                                  method='resSeq')
        by_bonds = get_fragments(self.geom.top,
                                 verbose=True,
                                 auto_fragment_names=True,
                                 method='bonds')

        assert _np.allclose(by_resSeq[0], [0, 1, 2])
        assert _np.allclose(by_resSeq[1], [3, 4, 5])
        assert _np.allclose(by_resSeq[2], [6, 7])

        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5])
        assert _np.allclose(by_bonds[2], [6])
        assert _np.allclose(by_bonds[3], [7])


    def test_get_fragments_join_fragments_normal(self):
        by_bonds = get_fragments(self.geom.top,
                                 join_fragments=[[1, 2]],
                                 verbose=True,
                                 auto_fragment_names=True,
                                 method='bonds')
        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5, 6])
        assert _np.allclose(by_bonds[2], [7])

    def test_get_fragments_join_fragments_special_cases(self):
        # Checking if redundant fragment ids are removed from the inner list for the argument "join_fragments"
        by_bonds = get_fragments(self.geom.top,
                                 join_fragments=[[1, 2, 2]],
                                 verbose=True,
                                 auto_fragment_names=True,
                                 method='bonds')

        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5, 6])
        assert _np.allclose(by_bonds[2], [7])

        # Checking for error from the overlapping ids in the argument "join_fragments"
        failed_assertion = False
        try:
            get_fragments(self.geom.top,
                          join_fragments=[[1, 2], [2, 3]],
                          verbose=True,
                          auto_fragment_names=True,
                          method='bonds')
        except AssertionError:
            failed_assertion = True
        assert failed_assertion

    def test_get_fragments_break_fragments_just_works(self):
        # Checking if the fragments are breaking correctly for the argument "fragment_breaker_fullresname"
        by_bonds = get_fragments(self.geom.top,
                                 fragment_breaker_fullresname=["VAL31", "GLU27"],  # two fragment breakers passed
                                 verbose=True,
                                 # auto_fragment_names=True,
                                 method='bonds')
        assert _np.allclose(by_bonds[0], [0])
        assert _np.allclose(by_bonds[1], [1, 2])
        assert _np.allclose(by_bonds[2], [3])
        assert _np.allclose(by_bonds[3], [4, 5])
        assert _np.allclose(by_bonds[4], [6])
        assert _np.allclose(by_bonds[5], [7])

    def test_get_fragments_break_fragments_special_cases_already_breaker(self):
        # No new fragments are created if an existing fragment breaker is passed
        by_bonds = get_fragments(self.geom.top,
                                 fragment_breaker_fullresname=["GLU30"],  # GLU30 is already a fragment breaker
                                 verbose=True,
                                 # auto_fragment_names=True,
                                 method='bonds')
        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5])
        assert _np.allclose(by_bonds[2], [6])
        assert _np.allclose(by_bonds[3], [7])

    def test_get_fragments_break_fragments_special_cases_already_breaker_passed_as_string(self):
        # Also works if input is a string instead of an iterable of strings
        by_bonds = get_fragments(self.geom.top,
                                 fragment_breaker_fullresname="GLU30",  # GLU30 is already a fragment breaker
                                 verbose=True,
                                 # auto_fragment_names=True,
                                 method='bonds')
        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5])
        assert _np.allclose(by_bonds[2], [6])
        assert _np.allclose(by_bonds[3], [7])

    def test_get_fragments_break_fragments_special_cases_breaker_not_present(self):
        # No new fragments are created if residue id is not present anywhere
        by_bonds = get_fragments(self.geom.top,
                                 fragment_breaker_fullresname=["Glu30"],  # not a valid id
                                 verbose=True,
                                 # auto_fragment_names=True,
                                 method='bonds')
        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5])
        assert _np.allclose(by_bonds[2], [6])
        assert _np.allclose(by_bonds[3], [7])

    def test_get_fragments_method_is_both(self):
        by_both = get_fragments(self.geom_force_resSeq_breaks.top, verbose=True, #the file has GLU27 and then LYS99 instead of LYS28
                      auto_fragment_names=True,
                      method='both') #method is both

        assert _np.allclose(by_both[0], [0, 1, 2])
        assert _np.allclose(by_both[1], [3, 4])
        assert _np.allclose(by_both[2], [5])
        assert _np.allclose(by_both[3], [6])
        assert _np.allclose(by_both[4], [7])

    def test_get_fragments_dont_know_method(self):
        failed_assertion = False
        try:
            get_fragments(self.geom.top,verbose=True,
                                 auto_fragment_names=True,
                                 method='xyz')
        except ValueError:
            failed_assertion = True
        assert failed_assertion


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


class Test_int_from_AA_code(unittest.TestCase):
    def test_int_from_AA_code_just_works(self):
        assert (int_from_AA_code("GLU30") == 30)
        assert (int_from_AA_code("E30") == 30)
        assert (int_from_AA_code("glu30") == 30)
        assert (int_from_AA_code("30glu40") == 3040)

class Test_bonded_neighborlist_from_top(unittest.TestCase):
    def setUp(self):
        self.geom = md.load("PDB/file_for_test.pdb")

    def test_bonded_neighborlist_from_top_just_works(self):
       neighbors_from_function =  bonded_neighborlist_from_top(self.geom.top)
       actual_neighbors = [[1], [0, 2], [1], [4], [3, 5], [4], [], []]
       assert neighbors_from_function == actual_neighbors

class Test_rangeexpand(unittest.TestCase):
    def test_rangeexpand_just_works(self):
        assert (rangeexpand("1-2, 3-4") == [1, 2, 3, 4])
        assert (rangeexpand("1-2, 3,4") == [1, 2, 3, 4])
        assert (rangeexpand("1-2, 03, 4") == [1, 2, 3, 4])

class Test_ctc_freq_reporter_by_residue_neighborhood(unittest.TestCase):
    def setUp(self):
        self.geom = md.load("PDB/file_for_test.pdb")
        self.by_bonds_geom = get_fragments(self.geom.top,
                                                     verbose=True,
                                                     auto_fragment_names=True,
                                                     method='bonds')
        self.residues = ["GLU30", "VAL31"]
        self.resname2residx, self.resname2fragidx = interactive_fragment_picker_by_AAresSeq(self.residues,
                                                                                                 self.by_bonds_geom,
                                                                                                 self.geom.top)

    def test_ctc_freq_reporter_by_residue_neighborhood_just_works(self):
        ctcs_mean = [30, 5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]

        input_values = (val for val in ["1", "1"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):#Checking against the input 1 and 1
            ctc_freq = ctc_freq_reporter_by_residue_neighborhood(ctcs_mean, self.resname2residx, self.by_bonds_geom, ctc_residxs_pairs,
                                                             self.geom.top,
                                                             n_ctcs=5, select_by_resSeq=None,
                                                             silent=False)
            assert ctc_freq[0] == 0
            assert ctc_freq[1] == 0


        input_values = (val for val in ["1", "2"])
        with mock.patch('builtins.input', lambda *x: next(input_values)): #Checking against the input 1 and 2
            ctc_freq = ctc_freq_reporter_by_residue_neighborhood(ctcs_mean, self.resname2residx, self.by_bonds_geom, ctc_residxs_pairs,
                                                            self.geom.top,
                                                             n_ctcs=5, select_by_resSeq=None,
                                                             silent=False)
            assert ctc_freq[0] == 0
            assert (_np.array_equal(ctc_freq[1],[0, 1]))

    def test_ctc_freq_reporter_by_residue_neighborhood_select_by_resSeq_is_int(self):
        ctcs_mean = [30, 5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]

        input_values = (val for val in ["1", "1"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):#Checking against the input 1 and 1
            ctc_freq = ctc_freq_reporter_by_residue_neighborhood(ctcs_mean, self.resname2residx, self.by_bonds_geom, ctc_residxs_pairs,
                                                             self.geom.top,
                                                             n_ctcs=5, select_by_resSeq=1,
                                                             silent=False)
            assert ctc_freq == {}

    def test_ctc_freq_reporter_by_residue_neighborhood_silent_is_true(self):
        ctcs_mean = [30, 5]
        ctc_residxs_pairs = [[0, 1], [2, 1]]

        input_values = (val for val in ["1", "1"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):#Checking against the input 1 and 1
            ctc_freq = ctc_freq_reporter_by_residue_neighborhood(ctcs_mean, self.resname2residx, self.by_bonds_geom, ctc_residxs_pairs,
                                                             self.geom.top,
                                                             n_ctcs=5, select_by_resSeq=None,
                                                             silent=True)
            assert (_np.array_equal(ctc_freq[0], [0]))
            assert (_np.array_equal(ctc_freq[1], [0, 1]))


class Test_table2BW_by_AAcode(unittest.TestCase):
    def setUp(self):
        self.file = "GPCRmd_B2AR_nomenclature_test.xlsx"

    def test_table2BW_by_AAcode_just_works(self):
        table2BW = table2BW_by_AAcode(tablefile = self.file)
        self.assertDictEqual(table2BW,
                             {'Q26': '1.25',
                              'E27': '1.26',
                              'R28': '1.27',
                              'F264': '1.28',
                              'M40': '1.39',
                              'S41': '1.40',
                              'L42': '1.41',
                              'I43': '1.42',
                              'V44': '1.43',
                              'L45': '1.44',
                              'A46': '1.45',
                              'I47': '1.46',
                              'V48': '1.47'})

    def test_table2BW_by_AAcode_keep_AA_code_test(self): #dictionary keys will only have AA id
        table2BW = table2BW_by_AAcode(tablefile = self.file, keep_AA_code=False)
        self.assertDictEqual(table2BW,
                             {26: '1.25',
                              27: '1.26',
                              28: '1.27',
                              264: '1.28',
                              40: '1.39',
                              41: '1.40',
                              42: '1.41',
                              43: '1.42',
                              44: '1.43',
                              45: '1.44',
                              46: '1.45',
                              47: '1.46',
                              48: '1.47'})

    def test_table2BW_by_AAcode_return_defs_test(self):
        table2BW = table2BW_by_AAcode(tablefile=self.file, return_defs=True)
        self.assertEqual(table2BW,
                         ({'Q26': '1.25',
                           'E27': '1.26',
                           'R28': '1.27',
                           'F264': '1.28',
                           'M40': '1.39',
                           'S41': '1.40',
                           'L42': '1.41',
                           'I43': '1.42',
                           'V44': '1.43',
                           'L45': '1.44',
                           'A46': '1.45',
                           'I47': '1.46',
                           'V48': '1.47'},
                          ['TM1']))


class Test_guess_missing_BWs(unittest.TestCase):
    #TODO need to attain 100% coverage
    def setUp(self):
        self.file = "GPCRmd_B2AR_nomenclature_test.xlsx"
        self.geom = md.load("PDB/file_for_test.pdb")

    def test_guess_missing_BWs_just_works(self):
        table2BW = table2BW_by_AAcode(tablefile=self.file)
        guess_BW = guess_missing_BWs(table2BW, self.geom.top, restrict_to_residxs=None)
        self.assertDictEqual(guess_BW,
                             {0: '1.29*',
                              1: '1.30*',
                              2: '1.31*',
                              3: '1.27*',
                              4: '1.26',
                              5: '1.27*',
                              6: '1.28*',
                              7: '1.28*'})

class Test_CGN_transformer(unittest.TestCase):
    def setUp(self):
        self.cgn = CGN_transformer()

    def test_CGN_transformer_just_works(self):
        self.assertEqual(len(self.cgn.seq), len(self.cgn.seq_idxs))
        self.assertEqual(len(self.cgn.seq), len(self.cgn.AA2CGN))

class Test_top2CGN_by_AAcode(unittest.TestCase):
    def setUp(self):
        self.cgn = CGN_transformer()
        self.geom = md.load("PDB/file_for_test.pdb")

    def test_top2CGN_by_AAcode_just_works(self):
        top2CGN = top2CGN_by_AAcode(self.geom.top, self.cgn)
        self.assertDictEqual(top2CGN,
                             {0: 'G.HN.27',
                              1: 'G.HN.53',
                              2: 'H.HC.11',
                              3: 'H.hdhe.4',
                              4: 'G.S2.3',
                              5: 'G.S2.5',
                              6: None,
                              7: 'G.S2.6'})

class Test_xtcs2ctcs(unittest.TestCase):
    def setUp(self):
        self.geom = md.load("prot1.pdb.gz")
        self.xtcs = ['run1_stride_100.xtc']

    def test_xtcs2ctcs_just_works(self):
        ctcs_trajs, time_array = xtcs2ctcs(self.xtcs, self.geom.top, [[1, 6]],  #stride=a.stride,
                                           # chunksize=a.chunksize_in_frames,
                                           return_time=True,
                                           consolidate=False)
        test_ctcs_trajs = _np.array([[0.4707409],
                                     [0.44984677],
                                     [0.49991393],
                                     [0.53672814],
                                     [0.53126746],
                                     [0.49817562],
                                     [0.46696925],
                                     [0.4860978],
                                     [0.4558717],
                                     [0.4896131]])
        test_time_array = _np.array([    0., 10000., 20000., 30000., 40000., 50000., 60000., 70000.,
       80000., 90000.])

        _np.testing.assert_array_almost_equal(ctcs_trajs[0][:10], test_ctcs_trajs, 4)
        assert(_np.array_equal(time_array[0][:10],test_time_array))

    def test_xtcs2ctcs_return_time_is_false(self):
        ctcs_trajs = xtcs2ctcs(self.xtcs, self.geom.top, [[1, 6]],  #stride=a.stride,
                                           # chunksize=a.chunksize_in_frames,
                                           return_time=False,
                                           consolidate=False)
        test_ctcs_trajs = _np.array([[0.4707409],
                                     [0.44984677],
                                     [0.49991393],
                                     [0.53672814],
                                     [0.53126746],
                                     [0.49817562],
                                     [0.46696925],
                                     [0.4860978],
                                     [0.4558717],
                                     [0.4896131]])
        _np.testing.assert_array_almost_equal(ctcs_trajs[0][:10], test_ctcs_trajs, 4)

    def test_xtcs2ctcs_consolidate_is_true(self):
        ctcs_trajs, time_array = xtcs2ctcs(self.xtcs, self.geom.top, [[1, 6]],  # stride=a.stride,
                                               # chunksize=a.chunksize_in_frames,
                                               return_time=True,
                                               consolidate=True)
        test_ctcs_trajs = _np.array([[0.4707409],
                                         [0.44984677],
                                         [0.49991393],
                                         [0.53672814],
                                         [0.53126746],
                                         [0.49817562],
                                         [0.46696925],
                                         [0.4860978],
                                         [0.4558717],
                                         [0.4896131]])
        test_time_array = _np.array([0., 10000., 20000., 30000., 40000., 50000., 60000., 70000.,
                                         80000., 90000.])

        _np.testing.assert_array_almost_equal(ctcs_trajs[:10], test_ctcs_trajs, 4)
        assert (_np.array_equal(time_array[:10], test_time_array))


class Test_interactive_fragment_picker_no_ambiguity_wip(unittest.TestCase):

    def setUp(self):
        self.geom = md.load("PDB/file_for_test.pdb")
        self.by_bonds_geom = get_fragments(self.geom.top,
                                           verbose=True,
                                           auto_fragment_names=True,
                                           method='bonds')

    def test_interactive_fragment_picker_wip_no_ambiguous(self):
        residues = ["GLU30", "GDP382", 30, 382]
        resname2residx, resname2fragidx = interactive_fragment_picker_wip(residues, self.by_bonds_geom,
                                                                                  self.geom.top)
        # Checking if residue names gives the correct corresponding residue id
        assert (resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
        assert (resname2residx[30]) == 0
        assert (resname2residx["GDP382"]) == 7  # GDP382 is the 8th residue
        assert (resname2residx[382]) == 7  # GDP382 is the 8th residue

        # Checking if the residue name give the correct corresponding fragment id
        assert (resname2fragidx["GLU30"]) == 0  # GLU30 is in the 1st fragment
        assert (resname2fragidx[30]) == 0  # GLU30 is in the 1st fragment
        assert (resname2fragidx["GDP382"]) == 3  # GDP382 is in the 4th fragment
        assert (resname2fragidx[382]) == 3  # GDP382 is in the 4th fragment



class Test_interactive_fragment_picker_with_ambiguity_wip(unittest.TestCase):

    def setUp(self):
        self.geom2frags = md.load("PDB/file_for_test_repeated_fullresnames.pdb")

        self.by_bonds_geom2frags = get_fragments(self.geom2frags.top,
                                                 verbose=True,
                                                 auto_fragment_names=True,
                                                 method='bonds')

    def test_interactive_fragment_picker_default_fragment_idx_is_none(self):
        residues = ["GLU30",30]
        input_values = (val for val in ["4", "4"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            resname2residx, resname2fragidx = interactive_fragment_picker_wip(residues, self.by_bonds_geom2frags,
                                                                                      self.geom2frags.top)

            # Checking if residue names gives the correct corresponding residue id
            # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
            assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th residue
            assert resname2fragidx["GLU30"] == 4 # GLU30 is in the 4th fragment

            assert (resname2residx[30]) == 8  # GLU30 is the 8th residue
            assert resname2fragidx[30] == 4 # GLU30 is in the 4th fragment

    def _test_interactive_fragment_picker_default_fragment_idx_is_none_last_answer(self):
        residues = ["GLU30",30]
        input_values = (val for val in ["\n", "\n"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            resname2residx, resname2fragidx = interactive_fragment_picker_wip(residues, self.by_bonds_geom2frags,
                                                                                      self.geom2frags.top)

            # Checking if residue names gives the correct corresponding residue id
            # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
            assert (resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
            assert resname2fragidx["GLU30"] == 0 # GLU30 is in the 1st fragment
            assert (resname2residx[30]) == 0  # GLU30 is the 1st residue
            assert resname2fragidx[30] == 0 # GLU30 is in the 1st fragment

    def test_interactive_fragment_picker_default_fragment_idx_is_none_ans_should_be_int(self):
        input_values = (val for val in ["xyz"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            failed_assertion = False
            try:
                interactive_fragment_picker_wip("GLU30", self.by_bonds_geom2frags, self.geom2frags.top)
            except (ValueError,AssertionError):
                failed_assertion = True
            assert failed_assertion

        input_values = (val for val in ["xyz"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            failed_assertion = False
            try:
                interactive_fragment_picker_wip(30, self.by_bonds_geom2frags, self.geom2frags.top)
            except (ValueError, AssertionError):
                failed_assertion = True
            assert failed_assertion

    def test_interactive_fragment_picker_default_fragment_idx_is_none_ans_should_be_in_list(self):

        input_values = (val for val in ["123"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            failed_assertion = False
            try:
                interactive_fragment_picker_wip("GLU30", self.by_bonds_geom2frags,self.geom2frags.top)

            except (ValueError,AssertionError):
                failed_assertion = True
            assert failed_assertion

        input_values = (val for val in ["123"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            failed_assertion = False
            try:
                interactive_fragment_picker_wip("30", self.by_bonds_geom2frags,self.geom2frags.top)
            except (ValueError,AssertionError):
                failed_assertion = True
            assert failed_assertion


    def test_interactive_fragment_picker_default_fragment_idx_is_passed(self):
        residues = ["GLU30", 30]
        resname2residx, resname2fragidx = interactive_fragment_picker_wip(residues, self.by_bonds_geom2frags,
                                                                                  self.geom2frags.top,default_fragment_idx=4)

        # Checking if residue names gives the correct corresponding residue id
        # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
        assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th residue
        assert (resname2fragidx["GLU30"]) == 4 # GLU30 is in the 4th fragment
        assert (resname2residx[30]) == 8  # GLU30 is the 8th residue
        assert (resname2fragidx[30]) == 4 # GLU30 is in the 4th fragment

    def test_interactive_fragment_picker_default_fragment_idx_is_passed_special_case(self):

        failed_assertion = False
        try:
            resname2residx, resname2fragidx = interactive_fragment_picker_wip("GLU30", self.by_bonds_geom2frags,
                                                                                  self.geom2frags.top,default_fragment_idx=99)
        except (ValueError,AssertionError):
            failed_assertion = True
        assert failed_assertion

        failed_assertion = False
        try:
            resname2residx, resname2fragidx = interactive_fragment_picker_wip(30, self.by_bonds_geom2frags,
                                                                              self.geom2frags.top,
                                                                              default_fragment_idx=99)
        except (ValueError, AssertionError):
            failed_assertion = True
        assert failed_assertion


    def test_interactive_fragment_picker_ambiguous(self):

        input_values = (val for val in ["4", "4"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residues = ["GLU30", 30]
            resname2residx, resname2fragidx = interactive_fragment_picker_wip(residues,
                                                                                      self.by_bonds_geom2frags,
                                                                                      self.geom2frags.top)

            assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th
            assert (resname2fragidx["GLU30"]) == 4  # GLU30 is in the 4th fragment
            assert (resname2residx[30]) == 8  # GLU30 is the 8th
            assert (resname2fragidx[30]) == 4  # GLU30 is in the 4th fragment

        input_values = (val for val in ["3", "3"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residues = ["GDP382", 382]
            resname2residx, resname2fragidx = interactive_fragment_picker_wip(residues,
                                                                                      self.by_bonds_geom2frags,
                                                                                      self.geom2frags.top)

            assert (resname2residx["GDP382"]) == 7  # GDP382 is the 7th residue
            assert (resname2fragidx["GDP382"]) == 3  # GDP382 is the 3rd fragment
            assert (resname2residx[382]) == 7  # GDP382 is the 7th residue
            assert (resname2fragidx[382]) == 3  # GDP382 is the 3rd fragment

    def test_interactive_fragment_picker_fragment_name(self):
        residues = ["GLU30", 30]
        input_values = (val for val in ["0", "0"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            resname2residx, resname2fragidx = interactive_fragment_picker_wip(residues, self.by_bonds_geom2frags,
                                                                                      self.geom2frags.top,
                                                                                      fragment_names=["A", "B", "C", "D",
                                                                                                      "E", "F", "G", "H"])
            # Checking if residue names gives the correct corresponding residue id
            # NOTE:Enter 0 for GLU30 when asked "input one fragment idx"

            assert (resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
            assert resname2fragidx["GLU30"] == 0  # GLU30 is in the 1st fragment
            assert (resname2residx[30]) == 0  # GLU30 is the 1st residue
            assert resname2fragidx[30] == 0  # GLU30 is in the 1st fragment

    def test_interactive_fragment_picker_idx_not_present(self):

        resname2residx, resname2fragidx = interactive_fragment_picker_wip(["GLU99",99], self.by_bonds_geom2frags,
                                                                              self.geom2frags.top,
                                                                              default_fragment_idx=99)
        assert(resname2residx["GLU99"] == None)
        assert (resname2residx[99] == None)
        assert (resname2fragidx["GLU99"] == None)
        assert (resname2fragidx[99] == None)


if __name__ == '__main__':
    unittest.main()

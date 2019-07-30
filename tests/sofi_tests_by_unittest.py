
import mdtraj as md
import unittest
import numpy as _np
from unittest.mock import patch
import mock

from sofi_functions import find_AA, top2residue_bond_matrix, get_fragments, \
    interactive_fragment_picker_by_AAresSeq,exclude_same_fragments_from_residx_pairlist,\
    unique_list_of_iterables_by_tuple_hashing, in_what_fragment,does_not_contain_strings, force_iterable, \
    is_iterable, in_what_N_fragments

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
        self.res_bond_matrix = _np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0, 0, 0],
                                          [0, 1, 1, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 1, 0, 0, 0],
                                          [0, 0, 0, 1, 1, 1, 0, 0],
                                          [0, 0, 0, 0, 1, 1, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0]])

    def test_it_just_works_with_top2residue_bond_matrix(self):
        assert (top2residue_bond_matrix(self.geom.top) == self.res_bond_matrix).all()

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


    def test_interactive_fragment_picker_by_AAresSeq_pick_first_fragment(self):
        residues = ["GLU30"]
        resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                                  self.geom2frags.top,
                                                                                  pick_first_fragment_by_default=True)

        assert (resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
        assert resname2fragidx["GLU30"] == 0  # GLU30 is in the 1st fragment

    @patch('builtins.input', lambda *args: '4')
    def test_interactive_fragment_picker_by_AAresSeq_ambiguous(self):
        #TODO
        residues = ["GLU30"] #, "GDP382"] # until we figure out how to inject input, run code, and inject agein
        resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues,
                                                                                  self.by_bonds_geom2frags,
                                                                                  self.geom2frags.top)

        # Checking if residue names gives the correct corresponding residue id
        # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
        # NOTE:Enter 3 for GDP382, when asked to "input one fragment idx"

        assert (resname2residx["GLU30"]) == 8  # GLU30 is the 9th residue
        assert (resname2fragidx["GLU30"]) == 4  # Same as entered explicitly

        #assert (resname2residx["GDP382"]) == 7  # GDP382 is the 8th residue
        #assert (resname2fragidx["GDP382"]) == 3  # Same as entered explicitly


    def _test_interactive_fragment_picker_by_AAresSeq_pick_last_answer(self):
        residues = ["GLU30", "VAL31"]
        resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues,
                                                                                  self.by_bonds_geom2frags,
                                                                                  self.geom2frags.top)
        # Checking if residue names gives the correct corresponding residue id
        # NOTE:Just press Return for GLU30 when asked "input one fragment idx"
        # NOTE:Just press Return for VAL31, when asked to "input one fragment idx"

        assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th
        assert (resname2fragidx["GLU30"]) == 4  # GLU30 is in the 4th fragment

        assert (resname2residx["VAL31"]) == 9  # VAL31 is the 9th residue
        assert (resname2fragidx["VAL31"]) == 4  # VAL31 is in the 4th fragment

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


    def _test_interactive_fragment_picker_by_AAresSeq_bad_answer(self):
        residues = ["GLU30"]

        self.assertRaises((ValueError,AssertionError), interactive_fragment_picker_by_AAresSeq,
                          residues, self.by_bonds_geom2frags, self.geom2frags.top)

        self.assertRaises((ValueError,AssertionError), interactive_fragment_picker_by_AAresSeq,residues,
                          self.by_bonds_geom2frags, self.geom2frags.top)

class Test_get_fragments(unittest.TestCase):
    def setUp(self):
        self.geom = md.load('PDB/file_for_test.pdb')

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
        assert _np.allclose(by_bonds[0], [3, 4, 5, 6])
        assert _np.allclose(by_bonds[1], [0, 1, 2])
        assert _np.allclose(by_bonds[2], [7])

    def test_get_fragments_join_fragments_special_cases(self):
        # Checking if redundant fragment ids are removed from the inner list for the argument "join_fragments"
        by_bonds = get_fragments(self.geom.top,
                                 join_fragments=[[1, 2, 2]],
                                 verbose=True,
                                 auto_fragment_names=True,
                                 method='bonds')

        assert _np.allclose(by_bonds[0], [3, 4, 5, 6])
        assert _np.allclose(by_bonds[1], [0, 1, 2])
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

if __name__ == '__main__':
    unittest.main()

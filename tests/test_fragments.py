
import mdtraj as md
import unittest
import numpy as _np
from unittest.mock import patch
import mock
from sofi_functions.fragments import get_fragments, \
    interactive_fragment_picker_by_AAresSeq, interactive_fragment_picker_wip
from filenames import filenames

test_filenames = filenames()

class Test_get_fragments(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)
        self.geom_force_resSeq_breaks = md.load(test_filenames.file_for_test_force_resSeq_breaks_is_true_pdb)

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

class Test_interactive_fragment_picker_no_ambiguity(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)
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
        self.geom2frags = md.load(test_filenames.file_for_test_repeated_fullresnames_pdb)

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

class Test_interactive_fragment_picker_no_ambiguity_wip(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)
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
        self.geom2frags = md.load(test_filenames.file_for_test_repeated_fullresnames_pdb)

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
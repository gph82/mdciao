
import mdtraj as md
import unittest
import numpy as _np
from unittest.mock import patch
import mock
from mdciao.fragments import get_fragments, \
    per_residue_fragment_picker, \
    overview, _allowed_fragment_methods, _print_frag
from filenames import filenames

import pytest

test_filenames = filenames()

class Test_overview(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)

    def test_just_runs(self):
        overview(self.geom.top)

    def test_select_method(self):
        overview(self.geom.top, "resSeq")

class Test_print_fragments(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)
        self.fragments = get_fragments(self.geom.top, verbose=False)


    def test_just_runs(self):
        _print_frag(0, self.geom.top, self.fragments[0])

    def test_other_name(self):
        _print_frag(0, self.geom.top, self.fragments[0], fragment_desc="blob")

    def test_raises(self):
        with pytest.raises(Exception):
            _print_frag(0, "self.geom.top", self.fragments[0])

    def test_returns(self):
        assert isinstance(_print_frag(0, self.geom.top, self.fragments[0],
                                      return_string=True), str)

class Test_get_fragments_methods(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)

    def test_resSeq(self):
        by_resSeq = get_fragments(self.geom.top,
                                  method='resSeq')

        assert _np.allclose(by_resSeq[0], [0, 1, 2])
        assert _np.allclose(by_resSeq[1], [3, 4])
        assert _np.allclose(by_resSeq[2], [5])
        assert _np.allclose(by_resSeq[3], [6, 7])

    def test_bonds(self):
        by_bonds = get_fragments(self.geom.top,
                                 method='bonds')

        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5])
        assert _np.allclose(by_bonds[2], [6])
        assert _np.allclose(by_bonds[3], [7])

    def test_resSeqplus(self):
        by_resSeqplus = get_fragments(self.geom.top,
                                 verbose=True,
                                 method='resSeq+')

        assert _np.allclose(by_resSeqplus[0], [0,1,2])
        assert _np.allclose(by_resSeqplus[1], [3,4,5,6,7])

    def test_resSeq_bonds(self):
        by_both = get_fragments(self.geom.top,
                                verbose=True,
                      method='resSeq_bonds') #method is both

        assert _np.allclose(by_both[0], [0, 1, 2])
        assert _np.allclose(by_both[1], [3, 4])
        assert _np.allclose(by_both[2], [5])
        assert _np.allclose(by_both[3], [6])
        assert _np.allclose(by_both[4], [7])

    def test_chains(self):
        by_chains = get_fragments(self.geom.top,
                                  verbose=True,
                                  method="chains")

        assert _np.allclose(by_chains[0],[0,1,2])
        assert _np.allclose(by_chains[1],[3,4,5])
        assert _np.allclose(by_chains[2],[6,7]), by_chains


    def test_dont_know_method(self):
        with pytest.raises(ValueError):
            get_fragments(self.geom.top,verbose=True,
                                 method='xyz')

    def test_molecule_raises_not_imp(self):
        with pytest.raises(NotImplementedError):
            get_fragments(self.geom.top,
                          method="molecules")

    def test_molecule_resSeqplus_raises_not_imp(self):
        with pytest.raises(NotImplementedError):
            get_fragments(self.geom.top,
                          method="molecules_resSeq+")


    def test_completeness(self):
        for imethod in _allowed_fragment_methods:
            frags = get_fragments(self.geom.top, method=imethod)
            assert _np.allclose(_np.hstack(frags),_np.arange(self.geom.top.n_residues))

class Test_get_fragments_other_options(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)
        self.geom_force_resSeq_breaks = md.load(test_filenames.file_for_test_force_resSeq_breaks_is_true_pdb)

    def test_join_fragments_normal(self):
        by_bonds = get_fragments(self.geom.top,
                                 join_fragments=[[1, 2]],
                                 verbose=True,
                                 method='bonds')
        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5, 6])
        assert _np.allclose(by_bonds[2], [7])

    def test_join_fragments_special_cases(self):
        # Checking if redundant fragment ids are removed from the inner list for the argument "join_fragments"
        by_bonds = get_fragments(self.geom.top,
                                 join_fragments=[[1, 2, 2]],
                                 verbose=True,
                                 method='bonds')

        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5, 6])
        assert _np.allclose(by_bonds[2], [7])

        # Checking for error from the overlapping ids in the argument "join_fragments"
        with pytest.raises(AssertionError):
            get_fragments(self.geom.top,
                          join_fragments=[[1, 2], [2, 3]],
                          verbose=True,
                          method='bonds')

    def test_atoms(self):
        for imethod in _allowed_fragment_methods:
            fragments_residxs = get_fragments(self.geom.top, method=imethod)
            fragments_atom_idxs = get_fragments(self.geom.top, atoms=True, method=imethod)

            for rfrag, afrag in zip(fragments_residxs, fragments_atom_idxs):
                ridxs_by_atom = _np.unique([self.geom.top.atom(aa).residue.index for aa in afrag])
                assert _np.allclose(ridxs_by_atom,rfrag)


    def test_break_fragments(self):
        # Checking if the fragments are breaking correctly for the argument "fragment_breaker_fullresname"
        by_bonds = get_fragments(self.geom.top,
                                 fragment_breaker_fullresname=["VAL31", "GLU27"],  # two fragment breakers passed
                                 verbose=True,
                                 method='bonds')
        assert _np.allclose(by_bonds[0], [0])
        assert _np.allclose(by_bonds[1], [1, 2])
        assert _np.allclose(by_bonds[2], [3])
        assert _np.allclose(by_bonds[3], [4, 5])
        assert _np.allclose(by_bonds[4], [6])
        assert _np.allclose(by_bonds[5], [7])

    def test_break_fragments_special_cases_already_breaker(self):
        # No new fragments are created if an existing fragment breaker is passed
        by_bonds = get_fragments(self.geom.top,
                                 fragment_breaker_fullresname=["GLU30"],  # GLU30 is already a fragment breaker
                                 verbose=True,
                                 method='bonds')
        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5])
        assert _np.allclose(by_bonds[2], [6])
        assert _np.allclose(by_bonds[3], [7])

    def test_break_fragments_special_cases_already_breaker_passed_as_string(self):
        # Also works if input is a string instead of an iterable of strings
        by_bonds = get_fragments(self.geom.top,
                                 fragment_breaker_fullresname="GLU30",  # GLU30 is already a fragment breaker
                                 verbose=True,
                                 method='bonds')
        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5])
        assert _np.allclose(by_bonds[2], [6])
        assert _np.allclose(by_bonds[3], [7])

    def test_break_fragments_special_cases_breaker_not_present(self):
        # No new fragments are created if residue id is not present anywhere
        by_bonds = get_fragments(self.geom.top,
                                 fragment_breaker_fullresname=["Glu30"],  # not a valid id
                                 verbose=True,
                                 method='bonds')
        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5])
        assert _np.allclose(by_bonds[2], [6])
        assert _np.allclose(by_bonds[3], [7])

@unittest.skip("interactive_fragment_picker_by_AAresSeq will be deprecated soon")
class Test_interactive_fragment_picker_no_ambiguity(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)
        self.by_bonds_geom = get_fragments(self.geom.top,
                                           verbose=True,
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

@unittest.skip("interactive_fragment_picker_by_AAresSeq will be deprecated soon")
class Test_interactive_fragment_picker_with_ambiguity(unittest.TestCase):

    def setUp(self):
        self.geom2frags = md.load(test_filenames.file_for_test_repeated_fullresnames_pdb)

        self.by_bonds_geom2frags = get_fragments(self.geom2frags.top,
                                                 verbose=True,
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

    @patch('builtins.input', lambda *args: "")
    def test_interactive_fragment_picker_by_AAresSeq_default_fragment_idx_is_none_last_answer(self):
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

        with pytest.raises((ValueError,AssertionError)):
            interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                    self.geom2frags.top)

    @patch('builtins.input', lambda *args: "123")
    def test_interactive_fragment_picker_by_AAresSeq_default_fragment_idx_is_none_ans_should_be_in_list(self):
        residues = ["GLU30"]

        with pytest.raises((ValueError, AssertionError)):
            interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                                  self.geom2frags.top)

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

        with pytest.raises(((ValueError, AssertionError))):
            resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                                  self.geom2frags.top,default_fragment_idx=99)

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

class Test_per_residue_fragment_picker_no_ambiguity(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)
        self.by_bonds_geom = get_fragments(self.geom.top,
                                           verbose=True,
                                           method='bonds')

    def test_no_ambiguous(self):
        residues = ["GLU30", "GDP382", 30, 382]
        resname2residx, resname2fragidx = per_residue_fragment_picker(residues, self.by_bonds_geom,
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

class Test_per_residue_fragment_picker(unittest.TestCase):

    def setUp(self):
        self.geom2frags = md.load(test_filenames.file_for_test_repeated_fullresnames_pdb)

        self.by_bonds_geom2frags = get_fragments(self.geom2frags.top,
                                                 verbose=True,
                                                 method='bonds')

    def test_default_fragment_idx_is_none(self):
        residues = ["GLU30",30]
        input_values = (val for val in ["4", "4"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            resname2residx, resname2fragidx = per_residue_fragment_picker(residues, self.by_bonds_geom2frags,
                                                                          self.geom2frags.top)

            # Checking if residue names gives the correct corresponding residue id
            # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
            assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th residue
            assert resname2fragidx["GLU30"] == 4 # GLU30 is in the 4th fragment

            assert (resname2residx[30]) == 8  # GLU30 is the 8th residue
            assert resname2fragidx[30] == 4 # GLU30 is in the 4th fragment

    def test_fragment_idx_is_none_last_answer(self):
        residues = ["GLU30",30]
        input_values = (val for val in ["", ""])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            resname2residx, resname2fragidx = per_residue_fragment_picker(residues, self.by_bonds_geom2frags,
                                                                          self.geom2frags.top)

            # Checking if residue names gives the correct corresponding residue id
            # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
            assert (resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
            assert resname2fragidx["GLU30"] == 0 # GLU30 is in the 1st fragment
            assert (resname2residx[30]) == 0  # GLU30 is the 1st residue
            assert resname2fragidx[30] == 0 # GLU30 is in the 1st fragment

    def test_default_fragment_idx_is_none_ans_should_be_int(self):
        input_values = (val for val in ["A"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            with pytest.raises((ValueError, AssertionError)):
                per_residue_fragment_picker("GLU30", self.by_bonds_geom2frags, self.geom2frags.top)


        input_values = (val for val in ["xyz"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            with pytest.raises((ValueError, AssertionError)):
               per_residue_fragment_picker(30, self.by_bonds_geom2frags, self.geom2frags.top)

    def test_default_fragment_idx_is_none_ans_should_be_in_list(self):
        input_values = (val for val in ["123"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            with pytest.raises((ValueError, AssertionError)):
                per_residue_fragment_picker("GLU30", self.by_bonds_geom2frags, self.geom2frags.top)

        input_values = (val for val in ["123"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            with pytest.raises((ValueError, AssertionError)):
                per_residue_fragment_picker("30", self.by_bonds_geom2frags, self.geom2frags.top)

    def test_default_fragment_idx_is_passed(self):
        residues = ["GLU30", 30]
        resname2residx, resname2fragidx = per_residue_fragment_picker(residues, self.by_bonds_geom2frags,
                                                                      self.geom2frags.top, pick_this_fragment_by_default=4)

        # Checking if residue names gives the correct corresponding residue id
        # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
        assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th residue
        assert (resname2fragidx["GLU30"]) == 4 # GLU30 is in the 4th fragment
        assert (resname2residx[30]) == 8  # GLU30 is the 8th residue
        assert (resname2fragidx[30]) == 4 # GLU30 is in the 4th fragment

    def test_default_fragment_idx_is_passed_special_case(self):

        with pytest.raises((ValueError, AssertionError)):
            resname2residx, resname2fragidx = per_residue_fragment_picker("GLU30", self.by_bonds_geom2frags,
                                                                          self.geom2frags.top,
                                                                          pick_this_fragment_by_default=99)

        with pytest.raises((ValueError, AssertionError)):
            resname2residx, resname2fragidx = per_residue_fragment_picker(30, self.by_bonds_geom2frags,
                                                                          self.geom2frags.top,
                                                                          pick_this_fragment_by_default=99)

    def test_ambiguous(self):

        input_values = (val for val in ["4", "4"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residues = ["GLU30", 30]
            resname2residx, resname2fragidx = per_residue_fragment_picker(residues,
                                                                          self.by_bonds_geom2frags,
                                                                          self.geom2frags.top)

            assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th
            assert (resname2fragidx["GLU30"]) == 4  # GLU30 is in the 4th fragment
            assert (resname2residx[30]) == 8  # GLU30 is the 8th
            assert (resname2fragidx[30]) == 4  # GLU30 is in the 4th fragment

        input_values = (val for val in ["3", "3"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residues = ["GDP382", 382]
            resname2residx, resname2fragidx = per_residue_fragment_picker(residues,
                                                                          self.by_bonds_geom2frags,
                                                                          self.geom2frags.top)

            assert (resname2residx["GDP382"]) == 7  # GDP382 is the 7th residue
            assert (resname2fragidx["GDP382"]) == 3  # GDP382 is the 3rd fragment
            assert (resname2residx[382]) == 7  # GDP382 is the 7th residue
            assert (resname2fragidx[382]) == 3  # GDP382 is the 3rd fragment

    def test_fragment_name(self):
        residues = ["GLU30", 30]
        input_values = (val for val in ["0", "0"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            resname2residx, resname2fragidx = per_residue_fragment_picker(residues, self.by_bonds_geom2frags,
                                                                          self.geom2frags.top,
                                                                          fragment_names=["A", "B", "C", "D",
                                                                                          "E", "F", "G", "H"])
            # Checking if residue names gives the correct corresponding residue id
            # NOTE:Enter 0 for GLU30 when asked "input one fragment idx"

            assert (resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
            assert resname2fragidx["GLU30"] == 0  # GLU30 is in the 1st fragment
            assert (resname2residx[30]) == 0  # GLU30 is the 1st residue
            assert resname2fragidx[30] == 0  # GLU30 is in the 1st fragment

    def test_idx_not_present(self):

        resname2residx, resname2fragidx = per_residue_fragment_picker(["GLU99", 99], self.by_bonds_geom2frags,
                                                                      self.geom2frags.top,
                                                                      pick_this_fragment_by_default=99)
        assert(resname2residx["GLU99"] == None)
        assert (resname2residx[99] == None)
        assert (resname2fragidx["GLU99"] == None)
        assert (resname2fragidx[99] == None)

    def test_extra_dicts(self):
        residues = ["GLU30", "TRP32"]
        consensus_dicts={"BW":{0 : "3.50"},
                        "CGN":{2 : "CGNt"}}
        resname2residx, resname2fragidx = per_residue_fragment_picker(residues, self.by_bonds_geom2frags,
                                                                      self.geom2frags.top,
                                                                      additional_naming_dicts=consensus_dicts,
                                                                      pick_this_fragment_by_default=0
                                                                      )

    def test_answer_letters(self):
        residues = ["GLU30", "TRP32"]
        input_values = (val for val in ["a", "b"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            resname2residx, resname2fragidx = per_residue_fragment_picker(residues, self.by_bonds_geom2frags,
                                                                      self.geom2frags.top,
                                                                      )


if __name__ == '__main__':
    unittest.main()
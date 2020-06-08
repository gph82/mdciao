
import mdtraj as md
import unittest
import numpy as _np
from unittest.mock import patch
import mock

from mdciao import fragments as mdcfragments
# I'm importing the private variable for testing other stuff, not to test the variable itself
from mdciao.fragments.fragments import _allowed_fragment_methods

from mdciao.filenames import filenames

import pytest

test_filenames = filenames()

class Test_overview(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.top_pdb)

    def test_just_runs(self):
        mdcfragments.overview(self.geom.top)

    def test_select_method(self):
        mdcfragments.overview(self.geom.top, "resSeq")

class Test_print_fragments(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.top_pdb)
        self.fragments = mdcfragments.get_fragments(self.geom.top, verbose=False)


    def test_just_runs(self):
         mdcfragments.print_frag(0, self.geom.top, self.fragments[0])

    def test_other_name(self):
         mdcfragments.print_frag(0, self.geom.top, self.fragments[0], fragment_desc="blob")

    def test_raises(self):
        with pytest.raises(Exception):
             mdcfragments.print_frag(0, "self.geom.top", self.fragments[0])

    def test_returns(self):
        assert isinstance(mdcfragments.print_frag(0, self.geom.top, self.fragments[0],
                                                  return_string=True), str)

    def test_uses_labels(self):
        outstr =  mdcfragments.print_frag(0, self.geom.top, self.fragments[0],
                                          return_string=True,
                                          idx2label={self.fragments[0][0 ]:"labelfirst",
                                        self.fragments[0][-1]:"labellast"})
        assert "@labelfirst" in outstr
        assert "@labellast"  in outstr

class Test_get_fragments_methods(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.small_monomer)

    def test_resSeq(self):
        by_resSeq = mdcfragments.get_fragments(self.geom.top,
                                               method='resSeq')

        assert _np.allclose(by_resSeq[0], [0, 1, 2])
        assert _np.allclose(by_resSeq[1], [3, 4])
        assert _np.allclose(by_resSeq[2], [5])
        assert _np.allclose(by_resSeq[3], [6, 7])

    def test_bonds(self):
        by_bonds = mdcfragments.get_fragments(self.geom.top,
                                              method='bonds')

        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5])
        assert _np.allclose(by_bonds[2], [6])
        assert _np.allclose(by_bonds[3], [7])

    def test_resSeqplus(self):
        by_resSeqplus = mdcfragments.get_fragments(self.geom.top,
                                                   verbose=True,
                                                   method='resSeq+')

        assert _np.allclose(by_resSeqplus[0], [0,1,2])
        assert _np.allclose(by_resSeqplus[1], [3,4,5,6,7])

    def test_resSeq_bonds(self):
        by_both = mdcfragments.get_fragments(self.geom.top,
                                             method='resSeq_bonds') #method is both

        assert _np.allclose(by_both[0], [0, 1, 2])
        assert _np.allclose(by_both[1], [3, 4])
        assert _np.allclose(by_both[2], [5])
        assert _np.allclose(by_both[3], [6])
        assert _np.allclose(by_both[4], [7])

    def test_chains(self):
        by_chains = mdcfragments.get_fragments(self.geom.top,
                                               method="chains")

        assert _np.allclose(by_chains[0],[0,1,2])
        assert _np.allclose(by_chains[1],[3,4,5])
        assert _np.allclose(by_chains[2],[6,7]), by_chains

    def test_NoneType(self):
        nonefrags = mdcfragments.get_fragments(self.geom.top,
                                               method=None)
        _np.testing.assert_array_equal(nonefrags[0],
                                       _np.arange(self.geom.top.n_residues))
        self.assertEqual(len(nonefrags),1)
    def test_Nonestr(self):
        nonefrags = mdcfragments.get_fragments(self.geom.top,
                                               method="None")
        _np.testing.assert_array_equal(nonefrags[0],
                                       _np.arange(self.geom.top.n_residues))
        self.assertEqual(len(nonefrags),1)

    def test_dont_know_method(self):
        with pytest.raises(AssertionError):
            mdcfragments.get_fragments(self.geom.top,
                                       method='xyz')

    @unittest.skip("Undecided whether to never implement this or not")
    def test_molecule_raises_not_imp(self):
        with pytest.raises(NotImplementedError):
            mdcfragments.get_fragments(self.geom.top,
                                       method="molecules")

    @unittest.skip("Undecided whether to never implement this or not")
    def test_molecule_resSeqplus_raises_not_imp(self):
        with pytest.raises(NotImplementedError):
            mdcfragments.get_fragments(self.geom.top,
                                       method="molecules_resSeq+")

    @unittest.skip("Do I need this?")
    def test_completeness(self):
        for imethod in _allowed_fragment_methods:
            frags = mdcfragments.get_fragments(self.geom.top, method=imethod)
            assert _np.allclose(_np.hstack(frags),_np.arange(self.geom.top.n_residues))

class Test_get_fragments_other_options(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.small_monomer)
        self.geom_force_resSeq_breaks = md.load(test_filenames.small_monomer_LYS99)

    def test_join_fragments_normal(self):
        by_bonds = mdcfragments.get_fragments(self.geom.top,
                                              join_fragments=[[1, 2]],
                                              verbose=True,
                                              method='bonds')
        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5, 6])
        assert _np.allclose(by_bonds[2], [7])

    def test_join_fragments_special_cases(self):
        # Checking if redundant fragment ids are removed from the inner list for the argument "join_fragments"
        by_bonds = mdcfragments.get_fragments(self.geom.top,
                                              join_fragments=[[1, 2, 2]],
                                              verbose=True,
                                              method='bonds')

        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5, 6])
        assert _np.allclose(by_bonds[2], [7])

        # Checking for error from the overlapping ids in the argument "join_fragments"
        with pytest.raises(AssertionError):
            mdcfragments.get_fragments(self.geom.top,
                                       join_fragments=[[1, 2], [2, 3]],
                                       verbose=True,
                                       method='bonds')

    def test_atoms(self):
        for imethod in _allowed_fragment_methods:
            fragments_residxs = mdcfragments.get_fragments(self.geom.top, method=imethod)
            fragments_atom_idxs = mdcfragments.get_fragments(self.geom.top, atoms=True, method=imethod)

            for rfrag, afrag in zip(fragments_residxs, fragments_atom_idxs):
                ridxs_by_atom = _np.unique([self.geom.top.atom(aa).residue.index for aa in afrag])
                assert _np.allclose(ridxs_by_atom,rfrag)


    def test_break_fragments(self):
        # Checking if the fragments are breaking correctly for the argument "fragment_breaker_fullresname"
        by_bonds = mdcfragments.get_fragments(self.geom.top,
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
        by_bonds = mdcfragments.get_fragments(self.geom.top,
                                              fragment_breaker_fullresname=["GLU30"],  # GLU30 is already a fragment breaker
                                              verbose=True,
                                              method='bonds')
        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5])
        assert _np.allclose(by_bonds[2], [6])
        assert _np.allclose(by_bonds[3], [7])

    def test_break_fragments_special_cases_already_breaker_passed_as_string(self):
        # Also works if input is a string instead of an iterable of strings
        by_bonds = mdcfragments.get_fragments(self.geom.top,
                                              fragment_breaker_fullresname="GLU30",  # GLU30 is already a fragment breaker
                                              verbose=True,
                                              method='bonds')
        assert _np.allclose(by_bonds[0], [0, 1, 2])
        assert _np.allclose(by_bonds[1], [3, 4, 5])
        assert _np.allclose(by_bonds[2], [6])
        assert _np.allclose(by_bonds[3], [7])

    def test_break_fragments_special_cases_breaker_not_present(self):
        # No new fragments are created if residue id is not present anywhere
        by_bonds = mdcfragments.get_fragments(self.geom.top,
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
        self.geom = md.load(test_filenames.top_pdb)
        self.by_bonds_geom = mdcfragments.get_fragments(self.geom.top,
                                                        verbose=True,
                                                        method='bonds')

    def test_interactive_fragment_picker_by_AAresSeq_no_ambiguous(self):
        residues = ["GLU30", "GDP382"]
        resname2residx, resname2fragidx = mdcfragments.interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom,
                                                                                               self.geom.top)
        # Checking if residue names gives the correct corresponding residue id
        assert (resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
        assert (resname2residx["GDP382"]) == 7  # GDP382 is the 8th residue

        # Checking if the residue name give the correct corresponding fragment id
        assert (resname2fragidx["GLU30"]) == 0  # GLU30 is in the 1st fragment
        assert (resname2fragidx["GDP382"]) == 3  # GDP382 is in the 4th fragment


    def test_interactive_fragment_picker_by_AAresSeq_not_present(self):
        residues = ["Glu30"]
        resname2residx, resname2fragidx = mdcfragments.interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom,
                                                                                               self.geom.top)
        assert (resname2residx["Glu30"] == None)
        assert (resname2fragidx["Glu30"] == None)

@unittest.skip("interactive_fragment_picker_by_AAresSeq will be deprecated soon")
class Test_interactive_fragment_picker_with_ambiguity(unittest.TestCase):

    def setUp(self):
        self.geom2frags = md.load(test_filenames.small_dimer)

        self.by_bonds_geom2frags = mdcfragments.get_fragments(self.geom2frags.top,
                                                              verbose=True,
                                                              method='bonds')


    @patch('builtins.input', lambda *args: '4')
    def test_interactive_fragment_picker_by_AAresSeq_default_fragment_idx_is_none(self):
        residues = ["GLU30"]
        resname2residx, resname2fragidx = mdcfragments.interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                                               self.geom2frags.top)

        # Checking if residue names gives the correct corresponding residue id
        # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
        assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th residue
        assert resname2fragidx["GLU30"] == 4 # GLU30 is in the 4th fragment

    @patch('builtins.input', lambda *args: "")
    def test_interactive_fragment_picker_by_AAresSeq_default_fragment_idx_is_none_last_answer(self):
        residues = ["GLU30"]
        resname2residx, resname2fragidx = mdcfragments.interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
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
            mdcfragments.interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                 self.geom2frags.top)

    @patch('builtins.input', lambda *args: "123")
    def test_interactive_fragment_picker_by_AAresSeq_default_fragment_idx_is_none_ans_should_be_in_list(self):
        residues = ["GLU30"]

        with pytest.raises((ValueError, AssertionError)):
            mdcfragments.interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                 self.geom2frags.top)

    def test_interactive_fragment_picker_by_AAresSeq_default_fragment_idx_is_passed(self):
        residues = ["GLU30"]
        resname2residx, resname2fragidx = mdcfragments.interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                                               self.geom2frags.top, default_fragment_idx=4)

        # Checking if residue names gives the correct corresponding residue id
        # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
        assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th residue
        assert (resname2fragidx["GLU30"]) == 4 # GLU30 is in the 4th fragment

    def test_interactive_fragment_picker_by_AAresSeq_default_fragment_idx_is_passed_special_case(self):
        residues = ["GLU30"]

        with pytest.raises(((ValueError, AssertionError))):
            resname2residx, resname2fragidx = mdcfragments.interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                                                   self.geom2frags.top, default_fragment_idx=99)

    def test_interactive_fragment_picker_by_AAresSeq_ambiguous(self):
        with mock.patch('builtins.input', return_value ='4'):
            resname2residx, resname2fragidx = mdcfragments.interactive_fragment_picker_by_AAresSeq("GLU30",
                                                                                                   self.by_bonds_geom2frags,
                                                                                                   self.geom2frags.top)

            assert (resname2residx["GLU30"]) == 8  # GLU30 is the 8th
            assert (resname2fragidx["GLU30"]) == 4  # GLU30 is in the 4th fragment
        with mock.patch('builtins.input', return_value ='3'):
            resname2residx, resname2fragidx = mdcfragments.interactive_fragment_picker_by_AAresSeq("GDP382",
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
        resname2residx, resname2fragidx = mdcfragments.interactive_fragment_picker_by_AAresSeq(residues, self.by_bonds_geom2frags,
                                                                                               self.geom2frags.top,
                                                                                               fragment_names=["A", "B", "C", "D",
                                                                                                  "E", "F", "G", "H"])
        # Checking if residue names gives the correct corresponding residue id
        # NOTE:Enter 0 for GLU30 when asked "input one fragment idx"

        assert (resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
        assert resname2fragidx["GLU30"] == 0  # GLU30 is in the 1st fragment

class Test_per_residue_fragment_picker_no_ambiguity(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.small_monomer)
        self.by_bonds_geom = mdcfragments.get_fragments(self.geom.top,
                                                        verbose=True,
                                                        method='bonds')

    def test_no_ambiguous(self):
        residues = ["GLU30", "GDP382", 30, 382]
        residxs, fragidxs = mdcfragments.per_residue_fragment_picker(residues, self.by_bonds_geom,
                                                                     self.geom.top)
        self.assertSequenceEqual([0,7,0,7], residxs)
        # GLU30 is the 1st residue
        # GDP382 is the 8th residue

        self.assertSequenceEqual([0,3,0,3], fragidxs)
        # GLU30 is in the 1st fragment
        # GDP382 is in the 4th fragment


class Test_per_residue_fragment_picker(unittest.TestCase):

    def setUp(self):
        self.geom2frags = md.load(test_filenames.small_dimer)

        self.by_bonds_geom2frags = mdcfragments.get_fragments(self.geom2frags.top,
                                                              verbose=True,
                                                              method='bonds')

    def test_default_fragment_idx_is_none(self):
        residues = ["GLU30", 30]
        input_values = (val for val in ["4", "4"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residxs, fragidxs = mdcfragments.per_residue_fragment_picker(residues, self.by_bonds_geom2frags,
                                                                         self.geom2frags.top)

            # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
            self.assertSequenceEqual(residxs, [8, 8])
            self.assertSequenceEqual(fragidxs, [4, 4])

    def test_fragment_idx_is_none_last_answer(self):
        residues = ["GLU30", 30]
        input_values = (val for val in ["", ""])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residxs, fragidxs = mdcfragments.per_residue_fragment_picker(residues, self.by_bonds_geom2frags,
                                                                         self.geom2frags.top)

            # NOTE:Accepted default "" for GLU30 when asked "input one fragment idx"
            self.assertSequenceEqual(residxs, [0, 0])
            self.assertSequenceEqual(fragidxs, [0, 0])

    def test_default_fragment_idx_is_none_ans_should_be_int(self):
        input_values = (val for val in ["A"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            with pytest.raises((ValueError, AssertionError)):
                mdcfragments.per_residue_fragment_picker("GLU30", self.by_bonds_geom2frags, self.geom2frags.top)

        input_values = (val for val in ["xyz"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            with pytest.raises((ValueError, AssertionError)):
                mdcfragments.per_residue_fragment_picker(30, self.by_bonds_geom2frags, self.geom2frags.top)

    def test_default_fragment_idx_is_none_ans_should_be_in_list(self):
        input_values = (val for val in ["123"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            with pytest.raises((ValueError, AssertionError)):
                mdcfragments.per_residue_fragment_picker("GLU30", self.by_bonds_geom2frags, self.geom2frags.top)

        input_values = (val for val in ["123"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            with pytest.raises((ValueError, AssertionError)):
                mdcfragments.per_residue_fragment_picker("30", self.by_bonds_geom2frags, self.geom2frags.top)

    def test_default_fragment_idx_is_passed(self):
        residues = ["GLU30", 30]
        residxs, fragidx = mdcfragments.per_residue_fragment_picker(residues, self.by_bonds_geom2frags,
                                                                    self.geom2frags.top,
                                                                    pick_this_fragment_by_default=4)

        # Checking if residue names gives the correct corresponding residue id
        # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
        self.assertSequenceEqual([8, 8], residxs)
        self.assertSequenceEqual([4, 4], fragidx)

    def test_default_fragment_idx_is_passed_special_case(self):
        with pytest.raises((ValueError, AssertionError)):
            mdcfragments.per_residue_fragment_picker("GLU30", self.by_bonds_geom2frags,
                                                     self.geom2frags.top,
                                                     pick_this_fragment_by_default=99)

        with pytest.raises((ValueError, AssertionError)):
            mdcfragments.per_residue_fragment_picker(30, self.by_bonds_geom2frags,
                                                     self.geom2frags.top,
                                                     pick_this_fragment_by_default=99)

    def test_ambiguous(self):
        input_values = (val for val in ["4", "4"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residues = ["GLU30", 30]
            residxs, fragidx = mdcfragments.per_residue_fragment_picker(residues,
                                                                        self.by_bonds_geom2frags,
                                                                        self.geom2frags.top)

            self.assertSequenceEqual([8, 8], residxs)
            self.assertSequenceEqual([4, 4], fragidx)

        input_values = (val for val in ["3", "3"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residues = ["GDP382", 382]
            residxs, fragidx = mdcfragments.per_residue_fragment_picker(residues,
                                                                        self.by_bonds_geom2frags,
                                                                        self.geom2frags.top)
            self.assertSequenceEqual([7, 7], residxs)
            self.assertSequenceEqual([3, 3], fragidx)

    def test_fragment_name(self):
        residues = ["GLU30", 30]
        input_values = (val for val in ["0", "0"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residxs, fragidx = mdcfragments.per_residue_fragment_picker(residues, self.by_bonds_geom2frags,
                                                                        self.geom2frags.top,
                                                                        fragment_names=["A", "B", "C", "D",
                                                                           "E", "F", "G", "H"])
            # Checking if residue names gives the correct corresponding residue id
            # NOTE:Enter 0 for GLU30 when asked "input one fragment idx"
            self.assertSequenceEqual([0, 0], residxs)
            self.assertSequenceEqual([0, 0], fragidx)

    def test_idx_not_present(self):
        residxs, fragidx = mdcfragments.per_residue_fragment_picker(["GLU99", 99], self.by_bonds_geom2frags,
                                                                    self.geom2frags.top,
                                                                    pick_this_fragment_by_default=99)
        self.assertSequenceEqual([None, None], residxs)
        self.assertSequenceEqual([None, None], fragidx)

    def test_extra_dicts(self):
        residues = ["GLU30", "TRP32"]
        consensus_dicts = {"BW": {0: "3.50"},
                           "CGN": {2: "CGNt"}}
        mdcfragments.per_residue_fragment_picker(residues, self.by_bonds_geom2frags,
                                                 self.geom2frags.top,
                                                 additional_naming_dicts=consensus_dicts,
                                                 pick_this_fragment_by_default=0
                                                 )

    def test_answer_letters(self):
        residues = ["GLU30", "TRP32"]
        input_values = (val for val in ["a", "b"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            mdcfragments.per_residue_fragment_picker(residues, self.by_bonds_geom2frags,
                                                     self.geom2frags.top,
                                                     )

class Test_list_of_fragments_strings_to_fragments(unittest.TestCase):

    def setUp(self):
        self.top = md.load(test_filenames.top_pdb).top
        self.fragments_by_resSeqplus = mdcfragments.get_fragments(self.top,
                                                                  method="resSeq+",
                                                                  verbose=False)
        self.fragments_by_resSeq = mdcfragments.get_fragments(self.top,
                                                              method="resSeq",
                                                              verbose=False)
    def test_consensus(self):
        from mdciao import fragments
        fragments, conlab  =  fragments.fragments_strings_to_fragments(["consensus"],
                                                  self.top)
        [_np.testing.assert_array_equal(ii,jj) for ii, jj in zip(fragments,
                                                  self.fragments_by_resSeqplus)]
        assert conlab

    def test_other_method(self):
        from mdciao import fragments
        fragments, conlab  =  fragments.fragments_strings_to_fragments(["resSeq"],
                                                  self.top)
        [_np.testing.assert_array_equal(ii,jj) for ii, jj in zip(fragments,
                                                  self.fragments_by_resSeq)]
        assert not conlab

    def test_one_fragment(self):
        from mdciao import fragments
        fragments, conlab =  fragments.fragments_strings_to_fragments(["0-10"],
                                                                      self.top)
        other = _np.arange(11, self.top.n_residues)
        [_np.testing.assert_array_equal(ii, jj) for ii, jj in zip(fragments,
                                                                  [_np.arange(11),
                                                                   other])]
        assert not conlab

    def test_more_than_one_fragment(self):
        from mdciao import fragments
        fragments, conlab =  fragments.fragments_strings_to_fragments(["0-10",
                                                                       "11-100",
                                                                       "200-210"],
                                                                      self.top)
        [_np.testing.assert_array_equal(ii, jj) for ii, jj in zip(fragments,
                                                                  [_np.arange(11),
                                                                   _np.arange(11,101),
                                                                   _np.arange(200,211)])]
        assert not conlab

    def test_verbose(self):
        from mdciao import fragments
        fragments, conlab =  fragments.fragments_strings_to_fragments(["0-10",
                                                                       "11-100",
                                                                       "200-210"],
                                                                      self.top,
                                                                      verbose=True)


class Test_frag_dict_2_frag_groups(unittest.TestCase):

    def test_works(self):
        input_values = (val for val in ["TM*,-TM2", "H8"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            groups_as_residue_idxs, \
            groups_as_keys, \
             = mdcfragments.frag_dict_2_frag_groups(
                                                                        {"TM1":[0,1],
                                                                         "TM2":[2,3],
                                                                         "TM3":[4,5],
                                                                         "H8":[6,7]},
                                                                        verbose=True,
                                                                    )
            self.assertSequenceEqual(groups_as_keys[0],["TM1","TM3"])
            self.assertSequenceEqual(groups_as_keys[1],["H8"])

            self.assertSequenceEqual(groups_as_residue_idxs[0],[0,1,4,5])
            self.assertSequenceEqual(groups_as_residue_idxs[1],[6,7])


class Test_frag_list_2_frag_groups(unittest.TestCase):

    def test_works_automatically(self):
        frags = [[0,1],
                 [2,3]]
        frags_out, frag_names = mdcfragments.frag_list_2_frag_groups(frags,
                                                                     verbose=True)
        self.assertSequenceEqual(frags[0],frags_out[0])
        self.assertSequenceEqual(frags[1],frags_out[1])
        self.assertEqual(len(frags_out),2)

    def test_works_frag_list_gt_2(self):
        frags = [[0,1],
                 [2,3],
                 [4,5]]

        input_values = (val for val in ["0,2","1"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            frags_out, frag_names = mdcfragments.frag_list_2_frag_groups(frags,
                                                                         )
            self.assertSequenceEqual([0,1,4,5], frags_out[0])
            self.assertSequenceEqual([2,3], frags_out[1])
            self.assertEqual(len(frags_out), 2)

    def test_works_frag_w_strs_as_input(self):
        frags = [[0,1],
                 [2,3],
                 [4,5]]

        frags_out, frag_names = mdcfragments.frag_list_2_frag_groups(frags,
                                                                     frag_idxs_group_1="0,2",
                                                                     frag_idxs_group_2="1",
                                                                     )
        self.assertSequenceEqual([0,1,4,5], frags_out[0])
        self.assertSequenceEqual([2,3], frags_out[1])
        self.assertEqual(len(frags_out), 2)

    def test_works_frag_w_ints_as_input(self):
        frags = [[0,1],
                 [2,3],
                 [4,5]]

        frags_out, frag_names = mdcfragments.frag_list_2_frag_groups(frags,
                                                                     frag_idxs_group_1=[0,2],
                                                                     frag_idxs_group_2=[1],
                                                                     )
        self.assertSequenceEqual([0,1,4,5], frags_out[0])
        self.assertSequenceEqual([2,3], frags_out[1])
        self.assertEqual(len(frags_out), 2)

    def test_works_frag_w_mixed_input(self):
        frags = [[0,1],
                 [2,3],
                 [4,5]]

        frags_out, frag_names = mdcfragments.frag_list_2_frag_groups(frags,
                                                                     frag_idxs_group_1="0,2",
                                                                     frag_idxs_group_2=[1],
                                                                     )
        self.assertSequenceEqual([0,1,4,5], frags_out[0])
        self.assertSequenceEqual([2,3], frags_out[1])
        self.assertEqual(len(frags_out), 2)



class Test_rangeexpand_residues2residxs(unittest.TestCase):

    def setUp(self):
        self.top = md.load(test_filenames.small_monomer).top
        self.fragments = mdcfragments.get_fragments(self.top, method="resSeq+",
                                                    verbose=False)

    def test_wildcards(self):
        expanded_range =   mdcfragments.rangeexpand_residues2residxs("GLU*",
                                                                     self.fragments,
                                                                     self.top)
        _np.testing.assert_array_equal(expanded_range, [0,4])

    def test_rangeexpand_res_idxs(self):
        expanded_range =   mdcfragments.rangeexpand_residues2residxs("2-4,6",
                                                                     self.fragments,
                                                                     self.top,
                                                                     interpret_as_res_idxs=True)
        _np.testing.assert_array_equal(expanded_range,[2,3,4,6])

    def test_rangeexpand_resSeq_w_jumps(self):
        expanded_range =   mdcfragments.rangeexpand_residues2residxs("26-381",
                                                                     self.fragments,
                                                                     self.top,
                                                                     )
        _np.testing.assert_array_equal(expanded_range,[3,4,5,6])

    def test_rangeexpand_resSeq_sort(self):
        expanded_range =   mdcfragments.rangeexpand_residues2residxs("381,26",
                                                                     self.fragments,
                                                                     self.top,
                                                                     sort=True
                                                                     )
        _np.testing.assert_array_equal(expanded_range,[3,6])


    def test_rangeexpand_raises_on_empty_range(self):
        with pytest.raises(ValueError):
            expanded_range =   mdcfragments.rangeexpand_residues2residxs("50-60",
                                                                         self.fragments,
                                                                         self.top,
                                                                         )
    def test_rangeexpand_raises_on_empty_wildcard(self):
        with pytest.raises(ValueError):
            expanded_range =   mdcfragments.rangeexpand_residues2residxs("ARG*",
                                                                         self.fragments,
                                                                         self.top,
                                                                         )

class Test_intersecting_fragments(unittest.TestCase):

    def setUp(self):
        self.fragments = [_np.arange(0,5),
                          _np.arange(5,10),
                          _np.arange(10,15)
                          ]
        self.top = md.load(test_filenames.top_pdb).top

    def test_no_clashes(self):
        result =   mdcfragments.check_if_subfragment([6, 7, 8],
                                                            "test_frag",
                                                     self.fragments,
                                                     self.top,
                                                     )
        _np.testing.assert_array_equal(result, [6,7,8])

    def test_clashes(self):
        input_values = (val for val in ["0"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):  # Checking against the input 1 and 1
            result =   mdcfragments.check_if_subfragment(_np.arange(3, 9),
                                                            "test_frag",
                                                         self.fragments,
                                                         self.top,
                                                         )
            _np.testing.assert_array_equal(result,[3,4])

    def test_clashes_keeps_all(self):
        result =   mdcfragments.check_if_subfragment(_np.arange(3, 9),
                                                            "test_frag",
                                                     self.fragments,
                                                     self.top,
                                                     keep_all=True)
        _np.testing.assert_array_equal(_np.arange(3,9),result)

if __name__ == '__main__':
    unittest.main()
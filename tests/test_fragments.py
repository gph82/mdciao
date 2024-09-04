
import mdtraj as md
import unittest
import numpy as _np
from unittest import mock

from mdciao import fragments as mdcfragments
# I'm importing the private variable for testing other stuff, not to test the variable itself
from mdciao.fragments.fragments import _allowed_fragment_methods, _fragments_strings_to_fragments

from mdciao.examples import filenames as test_filenames
from mdciao.examples import CGNLabeler_gnas2_human, GPCRLabeler_ardb2_human
from mdciao.utils.sequence import top2seq
from mdciao.nomenclature.nomenclature import _consensus_maps2consensus_frags

class Test_overview(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.top_pdb)

    def test_just_runs(self):
        mdcfragments.overview(self.geom.top)

    def test_select_method(self):
        mdcfragments.overview(self.geom.top, "resSeq")

    def test_bonds_on_gro_does_not_fail(self):
        mdcfragments.overview(md.load(test_filenames.file_for_no_bonds_gro).top, "bonds")


class Test_print_frag(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.top_pdb)
        self.fragments = mdcfragments.get_fragments(self.geom.top, verbose=False)

    def test_just_runs(self):
         mdcfragments.print_frag(0, self.geom.top, self.fragments[0])

    def tes_just_runs_w_sequence(self):
        mdcfragments.print_frag(0, top2seq(self.geom.top,"X"), self.fragments[0])

    def test_other_name(self):
         mdcfragments.print_frag(0, self.geom.top, self.fragments[0], fragment_desc="blob")

    def test_raises(self):
        with self.assertRaises(Exception):
             mdcfragments.print_frag(0, "self.geom.top", self.fragments[0])

    def test_returns(self):
        assert isinstance(mdcfragments.print_frag(0, self.geom.top, self.fragments[0],
                                                  just_return_string=True), str)

    def test_uses_labels(self):
        outstr =  mdcfragments.print_frag(0, self.geom.top, self.fragments[0],
                                          just_return_string=True,
                                          idx2label={self.fragments[0][0 ]:"labelfirst",
                                        self.fragments[0][-1]:"labellast"})
        assert "@labelfirst" in outstr
        assert "@labellast"  in outstr

class Test_print_fragments(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.top_pdb)
        self.fragments = mdcfragments.get_fragments(self.geom.top, verbose=False)

    def test_lists(self):
        printed_list = mdcfragments.print_fragments(self.fragments, self.geom.top)
        assert all([isinstance(item,str) for item in printed_list])
        assert len(printed_list)==len(self.fragments)
    def test_dict(self):
        frags = {"A":self.fragments[0],
                 "B":self.fragments[1]}
        printed_list = mdcfragments.print_fragments(frags, self.geom.top)
        assert all([isinstance(item,str) for item in printed_list])
        assert len(printed_list)==2

class Test_get_fragments_methods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.geom = md.load(test_filenames.small_monomer)

    def test_resSeq(self):
        by_resSeq = mdcfragments.get_fragments(self.geom.top,
                                               method='resSeq')

        assert _np.allclose(by_resSeq[0], [0, 1, 2])
        assert _np.allclose(by_resSeq[1], [3, 4])
        assert _np.allclose(by_resSeq[2], [5])
        assert _np.allclose(by_resSeq[3], [6, 7])

    def test_resSeq_file(self):
        by_resSeq = mdcfragments.get_fragments(test_filenames.small_monomer,
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
        with self.assertRaises(AssertionError):
            mdcfragments.get_fragments(self.geom.top,
                                       method='xyz')

    @unittest.skip("Undecided whether to never implement this or not")
    def test_molecule_raises_not_imp(self):
        with self.assertRaises(NotImplementedError):
            mdcfragments.get_fragments(self.geom.top,
                                       method="molecules")

    @unittest.skip("Undecided whether to never implement this or not")
    def test_molecule_resSeqplus_raises_not_imp(self):
        with self.assertRaises(NotImplementedError):
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
        with self.assertRaises(AssertionError):
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

    def test_ions_and_water(self):
        geom = md.load(test_filenames.ions_and_water)
        frags = mdcfragments.get_fragments(geom.top,verbose=False)
        assert len(frags)==3
        assert len(frags[1])==5
        assert len(frags[2])==10

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
        fragments, conlab = _fragments_strings_to_fragments("consensus",
                                                            self.top)
        [_np.testing.assert_array_equal(ii, jj) for ii, jj in zip(fragments,
                                                                  self.fragments_by_resSeqplus)]
        assert conlab

    def test_other_method(self):
        fragments, conlab = _fragments_strings_to_fragments("resSeq",
                                                            self.top)
        [_np.testing.assert_array_equal(ii, jj) for ii, jj in zip(fragments,
                                                                  self.fragments_by_resSeq)]
        assert not conlab

    def test_one_fragment(self):
        fragments, conlab = _fragments_strings_to_fragments(["0-10"],
                                                            self.top)
        other = _np.arange(11, self.top.n_residues)
        [_np.testing.assert_array_equal(ii, jj) for ii, jj in zip(fragments,
                                                                  [_np.arange(11),
                                                                   other])]
        assert not conlab

    def test_more_than_one_fragment(self):
        fragments, conlab = _fragments_strings_to_fragments(["0-10",
                                                            "11-100",
                                                            "200-210"],
                                                            self.top)
        [_np.testing.assert_array_equal(ii, jj) for ii, jj in zip(fragments,
                                                                  [_np.arange(11),
                                                                   _np.arange(11, 101),
                                                                   _np.arange(200, 211)])]
        assert not conlab

    def test_verbose(self):
        fragments, conlab = _fragments_strings_to_fragments(["0-10",
                                                            "11-100",
                                                            "200-210"],
                                                            self.top,
                                                            verbose=True)

    def test_mixed(self):
        fragments, conlab = _fragments_strings_to_fragments([_np.arange(10),
                                                            "GLU15-LEU132",
                                                            "200-210"],
                                                            self.top,
                                                            verbose=True)

    def test_fragment_outside(self):
        fragments, conlab = _fragments_strings_to_fragments([_np.arange(10),
                                                            "11-100",
                                                            "200-2100"],
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

    def test_works_with_answers(self):
        input_values = ["TM*,-TM2", "H8"]
        groups_as_residue_idxs, \
        groups_as_keys, \
         = mdcfragments.frag_dict_2_frag_groups({"TM1": [0, 1],
                                                 "TM2": [2, 3],
                                                 "TM3": [4, 5],
                                                 "H8": [6, 7]},
                                                verbose=True,
                                                answers=input_values
                                                )
        self.assertSequenceEqual(groups_as_keys[0],["TM1","TM3"])
        self.assertSequenceEqual(groups_as_keys[1],["H8"])

        self.assertSequenceEqual(groups_as_residue_idxs[0],[0,1,4,5])
        self.assertSequenceEqual(groups_as_residue_idxs[1],[6,7])

    def test_works_with_non_string_answers(self):
        input_values = ["TM*,-TM2", [2]]
        groups_as_residue_idxs, \
            groups_as_keys, \
            = mdcfragments.frag_dict_2_frag_groups({"TM1": [0, 1],
                                                    "TM2": [2, 3],
                                                    "TM3": [4, 5],
                                                    "H8": [6, 7],
                                                    "2":[8, 9]},
                                                   verbose=True,
                                                   answers=input_values
                                                   )
        self.assertSequenceEqual(groups_as_keys[0], ["TM1", "TM3"])
        self.assertSequenceEqual(groups_as_keys[1], ["2"])

        self.assertSequenceEqual(groups_as_residue_idxs[0], [0, 1, 4, 5])
        self.assertSequenceEqual(groups_as_residue_idxs[1], [8,9])
    def test_works_with_string_ranges(self):
        input_values = ["TM*,-TM2", "2-3"]
        groups_as_residue_idxs, \
            groups_as_keys, \
            = mdcfragments.frag_dict_2_frag_groups({"TM1": [0, 1],
                                                    "TM2": [2, 3],
                                                    "TM3": [4, 5],
                                                    "H8": [6, 7],
                                                    "2":[8, 9],
                                                    "3": [10, 11],
                                                    "4": [12,13]},
                                                   verbose=True,
                                                   answers=input_values
                                                   )
        self.assertSequenceEqual(groups_as_keys[0], ["TM1", "TM3"])
        self.assertSequenceEqual(groups_as_keys[1], ["2","3"])

        self.assertSequenceEqual(groups_as_residue_idxs[0], [0, 1, 4, 5])
        self.assertSequenceEqual(groups_as_residue_idxs[1], [8,9, 10, 11])

    def test_fails_on_empty(self):
        with self.assertRaises(ValueError):
                mdcfragments.frag_dict_2_frag_groups({"TM1": [0, 1],
                                                        "TM2": [2, 3],
                                                        "TM3": [4, 5],
                                                        "H8": [6, 7],
                                                        "2": [8, 9],
                                                        "3": [10, 11],
                                                        "4": [12, 13]},
                                                       verbose=True,
                                                       answers=["TM10",[4]]
                                                       )
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


class Test_match_fragments(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.actor_pdb)
        self.frags =mdcfragments.get_fragments(self.geom.top,verbose=False)

    def test_works(self):
        score, frg1, frg2 = mdcfragments.match_fragments(self.geom.top, self.geom.top, verbose=True)
        diag = _np.diag(score)
        _np.testing.assert_array_equal(diag, [len(ifrag) for ifrag in self.frags])
        [_np.testing.assert_array_equal(ifrag,jfrag) for ifrag, jfrag in zip(self.frags,frg1)]
        [_np.testing.assert_array_equal(ifrag,jfrag) for ifrag, jfrag in zip(self.frags,frg2)]

    def test_works_seq(self):
        score, frg1, frg2 = mdcfragments.match_fragments(self.geom.top,
                                                         top2seq(self.geom.top), verbose=True)
        score = score[:,0]
        _np.testing.assert_array_equal(score, [len(ifrag) for ifrag in frg1])
    def test_works_probe_and_short(self):
        score, _,_ = mdcfragments.match_fragments(self.geom.top, self.geom.top,probe=1)
        diag = _np.diag(score)
        _np.testing.assert_array_equal(diag,[ 1.,  1.,  1.,  1., _np.nan, _np.nan, _np.nan])

    def test_works_off_diag(self):
        seq0 = "AAA"+"BBBBAABBB"+"C"
        frags0 = [[0, 1, 2],
                  [3, 4, 5, 6, 7, 8, 9, 10, 11],
                  [12]]

        seq1 = "AA"
        frags1 = [[0, 1]]

        score, _, _ = mdcfragments.match_fragments(seq0,seq1,frags0=frags0,frags1=frags1,shortest=0)
        myscore = _np.array([[2, 2, 0]],ndmin=2).T
        _np.testing.assert_array_equal(score,myscore)

        scorep0, _, _ = mdcfragments.match_fragments(seq0, seq1, frags0=frags0, frags1=frags1, shortest=0,
                                                     probe=0)
        myscore = _np.array([[2/3, 2/9, 0/12]], ndmin=2).T
        _np.testing.assert_array_equal(scorep0,myscore)

        scorep1, _, _ = mdcfragments.match_fragments(seq0, seq1, frags0=frags0, frags1=frags1, shortest=0,
                                                     probe=1)
        myscore = _np.array([[2/2, 2/2, 0/2]], ndmin=2).T
        _np.testing.assert_array_equal(scorep1,myscore)

class Test_intersecting_fragments(unittest.TestCase):

    def setUp(self):
        self.fragments = [_np.arange(0,5),
                          _np.arange(5,10),
                          _np.arange(10,15)
                          ]
        self.top = md.load(test_filenames.top_pdb).top

    def test_no_clashes(self):
        was_subfragment, subfrag_after_prompt, intersects_with = mdcfragments.check_if_fragment_clashes([6, 7, 8],
                                                                                                       "test_frag",
                                                                                                       self.fragments,
                                                                                                       self.top,
                                                                                                       )
        assert was_subfragment
        _np.testing.assert_array_equal(intersects_with, [1])
        _np.testing.assert_array_equal(subfrag_after_prompt, [6,7,8])

    def test_clashes(self):
        input_values = (val for val in ["0"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):  # Checking against the input 1 and 1
            was_subfragment, subfrag_after_prompt, intersects_with = mdcfragments.check_if_fragment_clashes(
                _np.arange(3, 9),
                "test_frag",
                self.fragments,
                self.top,
                )
            assert was_subfragment is False
            _np.testing.assert_array_equal(intersects_with, [0,1])
            _np.testing.assert_array_equal(subfrag_after_prompt, [3,4])

    def test_clashes_no_prompt(self):
        was_subfragment, subfrag_after_prompt, intersects_with = mdcfragments.check_if_fragment_clashes(
            _np.arange(3, 9),
            "test_frag",
            self.fragments,
            self.top,
            prompt=False,
            )
        assert was_subfragment is False
        _np.testing.assert_array_equal(intersects_with, [0, 1])
        _np.testing.assert_array_equal(subfrag_after_prompt, _np.arange(3, 9))

class Test__get_fragments_by_jumps_in_sequence(unittest.TestCase):

    def test_works(self):
        seq = [0, 1, 2, 3, 4, 80, 81, 82, 50, 51, 52]
        frags, elements = mdcfragments._get_fragments_by_jumps_in_sequence(seq)
        self.assertListEqual(frags, [[0, 1, 2, 3, 4],
                                     [5, 6, 7],
                                     [8, 9, 10]])
        self.assertListEqual(elements, [[0, 1, 2, 3, 4],
                                        [80, 81, 82],
                                        [50, 51, 52]])

class Test_splice_fragments(unittest.TestCase):

    def test_works(self):
        fragments = [[0, 1, 2, 3],
                     [6, 7, 8],
                     [12, 13, 14]]
        fragnames = ["A", "B", "C"]
        newfrags, newnames = mdcfragments.splice_orphan_fragments(fragments, fragnames)

        self.assertListEqual(newfrags, [[0, 1, 2, 3],
                                        [4, 5],
                                        [6, 7, 8],
                                        [9, 10, 11],
                                        [12, 13, 14]])
        self.assertListEqual(newnames, ["A", "?", "B", "?", "C"])

    def test_works_no_action_needed(self):
        fragments = [[0, 1, 2, 3],
                     [4,5,6],
                     [7,8,9]]
        fragnames = ["A", "B", "C"]
        newfrags, newnames = mdcfragments.splice_orphan_fragments(fragments, fragnames)

        self.assertListEqual(newfrags, fragments)
        self.assertListEqual(newnames, fragnames)

    def test_works_naming(self):
        fragments = [[0, 1, 2, 3],
                     [6, 7, 8],
                     [12, 13, 14]]
        fragnames = ["A", "B", "C"]
        newfrags, newnames = mdcfragments.splice_orphan_fragments(fragments, fragnames, orphan_name='orphan_%u')

        self.assertListEqual(newfrags, [[0, 1, 2, 3],
                                        [4, 5],
                                        [6, 7, 8],
                                        [9, 10, 11],
                                        [12, 13, 14]])
        self.assertListEqual(newnames, ["A", "orphan_0", "B", "orphan_1", "C"])

    def test_nmax(self):
        fragments = [[0, 1, 2, 3],
                     [6, 7, 8],
                     [12, 13, 14]]
        fragnames = ["A", "B", "C"]
        newfrags, newnames = mdcfragments.splice_orphan_fragments(fragments, fragnames, highest_res_idx=17)
        self.assertListEqual(newfrags, [[0, 1, 2, 3],
                                        [4, 5],
                                        [6, 7, 8],
                                        [9, 10, 11],
                                        [12, 13, 14],
                                        [15, 16, 17]])
        _np.testing.assert_array_equal(newnames, ["A", "?", "B", "?", "C", "?"])

    def test_fragment_w_holes(self):
        fragments = [[0, 1, 2, 3],
                     [6, 8],
                     [12, 13, 14]]
        fragnames = ["A", "B", "C"]
        newfrags, newnames = mdcfragments.splice_orphan_fragments(fragments, fragnames)
        self.assertListEqual(newfrags, [[0, 1, 2, 3],
                                        [4, 5],
                                        [6, 7, 8],
                                        [9, 10, 11],
                                        [12, 13, 14]])

    def test_use_existing_frags(self):
        fragments = [[0, 1, 2, 3],
                     #[4, 5, 6, 7],
                     [8, 9, 10]]
        fragnames = ["A", "C"]
        newfrags, newnames = mdcfragments.splice_orphan_fragments(fragments, fragnames,
                                                                  other_fragments={"ex1":[4, 5]})

        self.assertListEqual(newfrags, [[0, 1, 2, 3],
                                        [4, 5],
                                        [6, 7],
                                        [8, 9, 10],
                                        ])
        self.assertListEqual(newnames, ["A", "ex1", "?", "C"])

class Test_assign_fragments(unittest.TestCase):

    def test_just_works(self):
        frag_idxs, res_idxs = mdcfragments.fragments.assign_fragments([0,4,5,1],[[0,1],[2,3],[4,5]])

        _np.testing.assert_array_equal(frag_idxs,[0,2,2,0])
        _np.testing.assert_array_equal(res_idxs, [0,4,5,1])

    def test_raises_missing(self):
        with _np.testing.assert_raises(ValueError):
           mdcfragments.fragments.assign_fragments([0, 4, 6, 1], [[0, 1], [2, 3], [4, 5]])

    def test_passes_missing(self):
        frag_idxs, res_idxs = mdcfragments.fragments.assign_fragments([0, 4, 6, 1], [[0, 1], [2, 3], [4, 5]], raise_on_missing=False)
        _np.testing.assert_array_equal(frag_idxs, [0, 2, 0])
        _np.testing.assert_array_equal(res_idxs, [0, 4, 1])

class Test_consensus_mix_fragment_info(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.GPCR = GPCRLabeler_ardb2_human(scheme="BW")
        cls.CGN = CGNLabeler_gnas2_human()
        cls.geom = md.load(test_filenames.actor_pdb)
        cls.consensus_frags = _consensus_maps2consensus_frags(cls.geom.top, [cls.GPCR, cls.CGN])[1]
        cls.fragments = mdcfragments.get_fragments(cls.geom.top)

    def test_works(self):
        fragments, fragnames = mdcfragments.mix_fragments(self.geom.top.n_residues - 1, self.consensus_frags, None, None)
        mixed_frags = {key:val for key, val in zip(fragnames, fragments)}
        for key, val in self.consensus_frags.items():
            # _mix_fragment_info calls -splice_fragments
            # which makes fragments full ranges [0,1,3,4]->[0,1,2,3,4]
            _np.testing.assert_array_equal(_np.arange(val[0],val[-1]+1),
                                           mixed_frags[key], key)
            mixed_frags.pop(key)
        self.assertDictEqual(mixed_frags, {"orphan 0": [280, 281, 282], # last CYS of B2AR and initial two CYSP2, CYSP3 of the GaN
                                           "orphan 1": _np.hstack([frag for frag in self.fragments[2:]]).tolist()  # Gb, Gg, Mg, GDP, the rest
                                           })

    def test_works_w_other_frags(self):
        fragments, fragnames = mdcfragments.mix_fragments(self.geom.top.n_residues - 1, self.consensus_frags, self.fragments, None)
        mixed_frags = {key: val for key, val in zip(fragnames, fragments)}
        for key, val in self.consensus_frags.items():
            # _mix_fragment_info calls -splice_fragments
            # which makes fragments full ranges [0,1,3,4]->[0,1,2,3,4]
            _np.testing.assert_array_equal(_np.arange(val[0], val[-1] + 1),
                                           mixed_frags[key], key)
            mixed_frags.pop(key)
        for key, val in {"subfrag 0 of frag 0": [280], #last CYS of B2AR
                         "subfrag 0 of frag 1": [281, 282],  # Initial two CYSP2, CYSP3 of the GaN
                         "frag 2": self.fragments[2],
                         "frag 3": self.fragments[3],
                         "frag 4": self.fragments[4],
                         "frag 5": self.fragments[5],
                         "frag 6": self.fragments[6]
                         }.items():
            _np.testing.assert_array_equal(mixed_frags[key], val, key)
            mixed_frags.pop(key)
        assert mixed_frags == {}

    def test_works_w_other_frags_other_names(self):
        fragments, fragnames = mdcfragments.mix_fragments(self.geom.top.n_residues - 1, self.consensus_frags, self.fragments,
                                                          ["B2AR", "Ga", "Gb","Gg", "P0G", "GDP", "MG"])
        mixed_frags = {key: val for key, val in zip(fragnames, fragments)}
        for key, val in self.consensus_frags.items():
            # _mix_fragment_info calls -splice_fragments
            # which makes fragments full ranges [0,1,3,4]->[0,1,2,3,4]
            _np.testing.assert_array_equal(_np.arange(val[0], val[-1] + 1),
                                           mixed_frags[key], key)
            mixed_frags.pop(key)
        for key, val in {"subfrag 0 of B2AR": [280], #last CYS of B2AR
                         "subfrag 0 of Ga": [281, 282],  # Initial two CYSP2, CYSP3 of the GaN
                         "Gb": self.fragments[2],
                         "Gg": self.fragments[3],
                         "P0G": self.fragments[4],
                         "GDP": self.fragments[5],
                         "MG": self.fragments[6]
                         }.items():
            _np.testing.assert_array_equal(mixed_frags[key], val, key)
            mixed_frags.pop(key)
        assert mixed_frags == {}

class Test_fragment_slice(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.geom = md.load(test_filenames.actor_pdb)
        cls.fragments_list = mdcfragments.get_fragments(cls.geom.top)
        """
        Auto-detected fragments with method 'lig_resSeq+'
        fragment      0 with    281 AAs    GLU30 (     0) -  CYSP341 (280   ) (0)  resSeq jumps
        fragment      1 with    379 AAs    CYSP2 (   281) -   LEU394 (659   ) (1)  resSeq jumps
        fragment      2 with    339 AAs     SER2 (   660) -   ASN340 (998   ) (2) 
        fragment      3 with     67 AAs     ALA2 (   999) -   CYSG68 (1065  ) (3) 
        fragment      4 with      1 AAs   P0G395 (  1066) -   P0G395 (1066  ) (4) 
        fragment      5 with      1 AAs   GDP396 (  1067) -   GDP396 (1067  ) (5) 
        fragment      6 with      1 AAs    MG397 (  1068) -    MG397 (1068  ) (6)   
        """
        cls.fragments_dict = {"B2AR": cls.fragments_list[0],
                              "Ga":   cls.fragments_list[1],
                              "Gb":   cls.fragments_list[2],
                              "Gg":   cls.fragments_list[3],
                              "P0G":  cls.fragments_list[4],
                              "GDP":  cls.fragments_list[5],
                              "Mg":   cls.fragments_list[6]
                              }
    def test_just_works(self):
        new_geom = mdcfragments.fragment_slice(self.geom, self.fragments_list[:1])
        new_frags = mdcfragments.get_fragments(new_geom.top)
        assert new_geom.n_residues == 281
        assert len(new_frags) == 1
        assert len(new_frags[0]) == 281
        assert str(new_geom.top.residue(0)) == "GLU30"
        assert str(new_geom.top.residue(280)) == "CYSP341"

    def test_keys(self):
        new_geom = mdcfragments.fragment_slice(self.geom, self.fragments_dict, keys_or_idxs=["Gb"])
        new_frags = mdcfragments.get_fragments(new_geom.top)
        assert new_geom.n_residues == 339
        assert len(new_frags) == 1
        assert len(new_frags[0]) == 339
        assert str(new_geom.top.residue(0)) == "SER2"
        assert str(new_geom.top.residue(338)) == "ASN340"

    def test_dict_no_keys(self):
        new_geom = mdcfragments.fragment_slice(self.geom, {key: self.fragments_dict[key] for key in ["Gg", "P0G"]})
        new_frags = mdcfragments.get_fragments(new_geom.top)
        assert new_geom.n_residues == 68
        assert len(new_frags) == 2
        assert len(new_frags[0]) == 67
        assert len(new_frags[1]) == 1
        assert str(new_geom.top.residue(0)) == "ALA2"
        assert str(new_geom.top.residue(66)) == "CYSG68"
        assert str(new_geom.top.residue(67)) == "P0G395"
if __name__ == '__main__':
    unittest.main()
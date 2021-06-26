
import mdtraj as md
import unittest
import numpy as _np
from unittest.mock import patch
import mock

from mdciao import fragments as mdcfragments
# I'm importing the private variable for testing other stuff, not to test the variable itself
from mdciao.fragments.fragments import _allowed_fragment_methods, _fragments_strings_to_fragments

from mdciao.filenames import filenames
from mdciao.utils.sequence import top2seq
import pytest

test_filenames = filenames()

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

class Test_print_fragments(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.top_pdb)
        self.fragments = mdcfragments.get_fragments(self.geom.top, verbose=False)

    def test_lists(self):
        mdcfragments.print_fragments(self.fragments, self.geom.top)

    def test_dict(self):
        frags = {"A":self.fragments[0],
                 "B":self.fragments[1]}
        mdcfragments.print_fragments(frags, self.geom.top)

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
        fragments, conlab  =  _fragments_strings_to_fragments(["consensus"],
                                                                        self.top)
        [_np.testing.assert_array_equal(ii,jj) for ii, jj in zip(fragments,
                                                  self.fragments_by_resSeqplus)]
        assert conlab

    def test_other_method(self):
        fragments, conlab  =  _fragments_strings_to_fragments(["resSeq"],
                                                                        self.top)
        [_np.testing.assert_array_equal(ii,jj) for ii, jj in zip(fragments,
                                                  self.fragments_by_resSeq)]
        assert not conlab

    def test_one_fragment(self):
        fragments, conlab =  _fragments_strings_to_fragments(["0-10"],
                                                                       self.top)
        other = _np.arange(11, self.top.n_residues)
        [_np.testing.assert_array_equal(ii, jj) for ii, jj in zip(fragments,
                                                                  [_np.arange(11),
                                                                   other])]
        assert not conlab

    def test_more_than_one_fragment(self):
        fragments, conlab =  _fragments_strings_to_fragments(["0-10",
                                                                       "11-100",
                                                                       "200-210"],
                                                                       self.top)
        [_np.testing.assert_array_equal(ii, jj) for ii, jj in zip(fragments,
                                                                  [_np.arange(11),
                                                                   _np.arange(11,101),
                                                                   _np.arange(200,211)])]
        assert not conlab

    def test_verbose(self):
        fragments, conlab =  _fragments_strings_to_fragments(["0-10",
                                                                       "11-100",
                                                                       "200-210"],
                                                                       self.top,
                                                                       verbose=True)

    def test_arrays(self):
        fragments, conlab = _fragments_strings_to_fragments([_np.arange(10),
                                                             "11-100",
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
        score, frg1, frg2 = mdcfragments.match_fragments(self.geom.top, self.geom.top)
        diag = _np.diag(score)
        _np.testing.assert_array_equal(diag, [len(ifrag) for ifrag in self.frags])
        [_np.testing.assert_array_equal(ifrag,jfrag) for ifrag, jfrag in zip(self.frags,frg1)]
        [_np.testing.assert_array_equal(ifrag,jfrag) for ifrag, jfrag in zip(self.frags,frg2)]

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

class Test__get_fragments_by_jumps_in_sequence(unittest.TestCase):

    def test_works(self):
        seq = [0, 1, 2, 3, 4, 80, 81, 82, 50, 51, 52]
        frags, elements = mdcfragments._get_fragments_by_jumps_in_sequence(seq)
        _np.testing.assert_array_equal(frags, [[0, 1, 2, 3, 4],
                                               [5, 6, 7],
                                               [8, 9, 10]])
        _np.testing.assert_array_equal(elements, [[0, 1, 2, 3, 4],
                                                  [80, 81, 82],
                                                  [50, 51, 52]])

class Test_splice_fragments(unittest.TestCase):

    def test_works(self):
        fragments = [[0, 1, 2, 3],
                     [6, 7, 8],
                     [12, 13, 14]]
        fragnames = ["A", "B", "C"]
        newfrags, newnames = mdcfragments.splice_orphan_fragments(fragments, fragnames)

        _np.testing.assert_array_equal(newfrags, [[0, 1, 2, 3],
                                                  [4, 5],
                                                  [6, 7, 8],
                                                  [9, 10, 11],
                                                  [12, 13, 14]])
        _np.testing.assert_array_equal(newnames, ["A", "?", "B", "?", "C"])

    def test_works_no_action_needed(self):
        fragments = [[0, 1, 2, 3],
                     [4,5,6],
                     [7,8,9]]
        fragnames = ["A", "B", "C"]
        newfrags, newnames = mdcfragments.splice_orphan_fragments(fragments, fragnames)

        _np.testing.assert_array_equal(newfrags, fragments)
        _np.testing.assert_array_equal(newnames, fragnames)

    def test_works_naming(self):
        fragments = [[0, 1, 2, 3],
                     [6, 7, 8],
                     [12, 13, 14]]
        fragnames = ["A", "B", "C"]
        newfrags, newnames = mdcfragments.splice_orphan_fragments(fragments, fragnames, orphan_name='orphan_%u')

        _np.testing.assert_array_equal(newfrags, [[0, 1, 2, 3],
                                                  [4, 5],
                                                  [6, 7, 8],
                                                  [9, 10, 11],
                                                  [12, 13, 14]])
        _np.testing.assert_array_equal(newnames, ["A", "orphan_0", "B", "orphan_1", "C"])

    def test_nmax(self):
        fragments = [[0, 1, 2, 3],
                     [6, 7, 8],
                     [12, 13, 14]]
        fragnames = ["A", "B", "C"]
        newfrags, newnames = mdcfragments.splice_orphan_fragments(fragments, fragnames, highest_res_idx=17)

        _np.testing.assert_array_equal(newfrags, [[0, 1, 2, 3],
                                                  [4, 5],
                                                  [6, 7, 8],
                                                  [9, 10, 11],
                                                  [12, 13, 14],
                                                  [15,16,17]])
        _np.testing.assert_array_equal(newnames, ["A", "?", "B", "?", "C", "?"])

    def test_fragment_w_holes(self):
        fragments = [[0, 1, 2, 3],
                     [6, 8],
                     [12, 13, 14]]
        fragnames = ["A", "B", "C"]
        newfrags, newnames = mdcfragments.splice_orphan_fragments(fragments, fragnames)
        _np.testing.assert_array_equal(newfrags,[[0,1,2,3],
                                                  [4,5],
                                                  [6,7,8],
                                                  [9,10,11],
                                                  [12,13,14]])

    def test_use_existing_frags(self):
        fragments = [[0, 1, 2, 3],
                     #[4, 5, 6, 7],
                     [8, 9, 10]]
        fragnames = ["A", "C"]
        newfrags, newnames = mdcfragments.splice_orphan_fragments(fragments, fragnames,
                                                                  other_fragments={"ex1":[4, 5]})

        _np.testing.assert_array_equal(newfrags, [[0, 1, 2, 3],
                                                  [4, 5],
                                                  [6, 7],
                                                  [8, 9, 10],
                                                 ])
        _np.testing.assert_array_equal(newnames, ["A", "ex1", "?", "C"])


if __name__ == '__main__':
    unittest.main()
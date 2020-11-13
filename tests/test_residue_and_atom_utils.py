import mdtraj as md
import numpy as np
import unittest
from mdciao.filenames import filenames
from mdciao.utils import residue_and_atom 
import mdciao.fragments as _mdcfrg
import pytest
import io
from contextlib import redirect_stdout
from unittest.mock import patch
import mock

test_filenames = filenames()

class Test_find_by_AA(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.small_monomer)
        self.geom2frags = md.load(test_filenames.small_dimer)

    def test_full_long_AA_code(self):
        self.assertSequenceEqual(residue_and_atom.find_AA(self.geom.top, "GLU30"),[0])
        self.assertSequenceEqual(residue_and_atom.find_AA(self.geom.top, "LYS29"),[5])

    def test_full_short_AA_code(self):
        self.assertSequenceEqual(residue_and_atom.find_AA(self.geom.top, 'E30'), [0])
        self.assertSequenceEqual(residue_and_atom.find_AA(self.geom.top, 'W32'), [2])

    def test_short_AA_code(self):
        self.assertSequenceEqual(residue_and_atom.find_AA(self.geom.top, 'E'), [0,4])

    def test_short_long_AA_code(self):
        self.assertSequenceEqual(residue_and_atom.find_AA(self.geom.top, 'GLU'), [0, 4])

    def test_does_not_find_AA(self):
        assert (residue_and_atom.find_AA(self.geom.top, "lys20")) == []   # small case won't give any result
        assert (residue_and_atom.find_AA(self.geom.top, 'w32')) == []    # small case won't give any result
        assert (residue_and_atom.find_AA(self.geom.top, 'w 32')) == []   # spaces between characters won't work

    def test_malformed_input(self):
        with pytest.raises(ValueError):
            residue_and_atom.find_AA(self.geom.top, "GLUTAMINE")

    def test_malformed_code(self):
        with pytest.raises(ValueError):
            residue_and_atom.find_AA(self.geom.top, "ARGI200")

    def test_ambiguity(self):
        # AMBIGUOUS definition i.e. each residue is present in multiple fragments
        self.assertSequenceEqual(residue_and_atom.find_AA(self.geom2frags.top, "LYS28"), [5, 13]) # getting multiple idxs,as expected
        self.assertSequenceEqual(residue_and_atom.find_AA(self.geom2frags.top, "K28"), [5, 13])

    def test_just_numbers(self):
        np.testing.assert_array_equal(residue_and_atom.find_AA(self.geom2frags.top,"28"),[5,13])

class Test_int_from_AA_code(unittest.TestCase):
    def test_int_from_AA_code(self):
        assert (residue_and_atom.int_from_AA_code("GLU30") == 30)
        assert (residue_and_atom.int_from_AA_code("E30") == 30)
        assert (residue_and_atom.int_from_AA_code("glu30") == 30)
        assert (residue_and_atom.int_from_AA_code("30glu40") == 3040)

class Test_name_from_AA(unittest.TestCase):
    def test_name_from_AA(self):
        assert(residue_and_atom.name_from_AA("GLU30") == 'GLU')
        assert (residue_and_atom.name_from_AA("E30") == 'E')

class Test_shorten_AA(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.small_monomer)

    def test_shorten_AA(self):
        assert(residue_and_atom.shorten_AA("GLU30") == 'E')
        assert(residue_and_atom.shorten_AA(self.geom.top.residue(1)) == 'V')

    def test_shorten_AA_substitute_fail_is_none(self):
        failed_assertion = False
        try:
            residue_and_atom.shorten_AA("glu30")
        except KeyError:
            failed_assertion = True
        assert failed_assertion

    def test_shorten_AA_substitute_fail_is_long(self):
        assert(residue_and_atom.shorten_AA("glu30", substitute_fail='long') == 'glu')

    def test_shorten_AA_substitute_fail_is_letter(self):
        assert(residue_and_atom.shorten_AA("glu30", substitute_fail='g') == 'g')

    def test_shorten_AA_substitute_fail_is_string_of_length_greater_than_1(self):
        failed_assertion = False
        try:
            residue_and_atom.shorten_AA("glu30", substitute_fail='glutamine')
        except ValueError:
            failed_assertion = True
        assert failed_assertion

    def test_shorten_AA_substitute_fail_is_0(self):
        assert(residue_and_atom.shorten_AA("glu30", substitute_fail=0) == 'g')

    def test_shorten_AA_substitute_fail_is_int(self):
        failed_assertion = False
        try:
            residue_and_atom.shorten_AA("glu30", substitute_fail=1)
        except ValueError:
            failed_assertion = True
        assert failed_assertion

    def test_shorten_AA_keep_index_is_true(self):
        assert(residue_and_atom.shorten_AA("GLU30", keep_index=True) == 'E30')
        assert(residue_and_atom.shorten_AA("glu30",substitute_fail='E',keep_index=True) == 'E30')

class Test_atom_type(unittest.TestCase):
    def test_works(self):
        top = md.load(test_filenames.pdb_3CAP).top
        atoms_BB = [aa for aa in top.residue(0).atoms if aa.is_backbone]
        atoms_SC = [aa for aa in top.residue(0).atoms if aa.is_sidechain]
        atoms_X = [aa for aa in top.atoms if not aa.is_backbone and not aa.is_sidechain]
        assert len(atoms_BB)>0
        assert len(atoms_SC)>0
        assert len(atoms_X)>0
        assert all([residue_and_atom.atom_type(aa) == "BB" for aa in atoms_BB])
        assert all([residue_and_atom.atom_type(aa) == "SC" for aa in atoms_SC])
        assert all([residue_and_atom.atom_type(aa) == "X" for aa in atoms_X])

class Test_residues_from_descriptors_no_ambiguity(unittest.TestCase):

    def setUp(self):
        self.geom = md.load(test_filenames.small_monomer)
        self.by_bonds_geom = _mdcfrg.get_fragments(self.geom.top,
                                                        verbose=True,
                                                        method='bonds')

    def test_no_ambiguous(self):
        residues = ["GLU30", "GDP382", 30, 382]
        residxs, fragidxs = residue_and_atom.residues_from_descriptors(residues, self.by_bonds_geom,
                                                                     self.geom.top)
        self.assertSequenceEqual([0,7,0,7], residxs)
        # GLU30 is the 1st residue
        # GDP382 is the 8th residue

        self.assertSequenceEqual([0,3,0,3], fragidxs)
        # GLU30 is in the 1st fragment
        # GDP382 is in the 4th fragment

    def test_overlaping_frags(self):
        residues = ["GLU30"]
        with pytest.raises(ValueError):
            residue_and_atom.residues_from_descriptors(residues, [np.arange(self.geom.n_residues),
                                                                  [np.arange(self.geom.n_residues)]],
                                                       self.geom.top)

    def test_not_in_fragment(self):
        residues = ["GLU30"]
        with pytest.raises(ValueError):
            residue_and_atom.residues_from_descriptors(residues, [np.arange(3,5)],
                                                       self.geom.top)


class Test_residues_from_descriptors(unittest.TestCase):

    def setUp(self):
        self.geom2frags = md.load(test_filenames.small_dimer)

        self.by_bonds_geom2frags = _mdcfrg.get_fragments(self.geom2frags.top,
                                                              verbose=True,
                                                              method='bonds')

    def test_default_fragment_idx_is_none(self):
        residues = ["GLU30", 30]
        input_values = (val for val in ["4", "4"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residxs, fragidxs = residue_and_atom.residues_from_descriptors(residues, self.by_bonds_geom2frags,
                                                                         self.geom2frags.top)

            # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
            self.assertSequenceEqual(residxs, [8, 8])
            self.assertSequenceEqual(fragidxs, [4, 4])

    def test_fragment_idx_is_none_last_answer(self):
        residues = ["GLU30", 30]
        input_values = (val for val in ["", ""])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residxs, fragidxs = residue_and_atom.residues_from_descriptors(residues, self.by_bonds_geom2frags,
                                                                         self.geom2frags.top)

            # NOTE:Accepted default "" for GLU30 when asked "input one fragment idx"
            self.assertSequenceEqual(residxs, [0, 0])
            self.assertSequenceEqual(fragidxs, [0, 0])

    def test_default_fragment_idx_is_none_ans_should_be_int(self):
        input_values = (val for val in ["A"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            with pytest.raises((ValueError, AssertionError)):
                residue_and_atom.residues_from_descriptors("GLU30", self.by_bonds_geom2frags, self.geom2frags.top)

        input_values = (val for val in ["xyz"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            with pytest.raises((ValueError, AssertionError)):
                residue_and_atom.residues_from_descriptors(30, self.by_bonds_geom2frags, self.geom2frags.top)

    def test_default_fragment_idx_is_none_ans_should_be_in_list(self):
        input_values = (val for val in ["123"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            with pytest.raises((ValueError, AssertionError)):
                residue_and_atom.residues_from_descriptors("GLU30", self.by_bonds_geom2frags, self.geom2frags.top)

        input_values = (val for val in ["123"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            with pytest.raises((ValueError, AssertionError)):
                residue_and_atom.residues_from_descriptors("30", self.by_bonds_geom2frags, self.geom2frags.top)

    def test_default_fragment_idx_is_passed(self):
        residues = ["GLU30", 30]
        residxs, fragidx = residue_and_atom.residues_from_descriptors(residues, self.by_bonds_geom2frags,
                                                                    self.geom2frags.top,
                                                                    pick_this_fragment_by_default=4)

        # Checking if residue names gives the correct corresponding residue id
        # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
        self.assertSequenceEqual([8, 8], residxs)
        self.assertSequenceEqual([4, 4], fragidx)

    def test_default_fragment_idx_is_passed_special_case(self):
        with pytest.raises((ValueError, AssertionError)):
            residue_and_atom.residues_from_descriptors("GLU30", self.by_bonds_geom2frags,
                                                     self.geom2frags.top,
                                                     pick_this_fragment_by_default=99)

        with pytest.raises((ValueError, AssertionError)):
            residue_and_atom.residues_from_descriptors(30, self.by_bonds_geom2frags,
                                                     self.geom2frags.top,
                                                     pick_this_fragment_by_default=99)

    def test_ambiguous(self):
        input_values = (val for val in ["4", "4"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residues = ["GLU30", 30]
            residxs, fragidx = residue_and_atom.residues_from_descriptors(residues,
                                                                        self.by_bonds_geom2frags,
                                                                        self.geom2frags.top)

            self.assertSequenceEqual([8, 8], residxs)
            self.assertSequenceEqual([4, 4], fragidx)

        input_values = (val for val in ["3", "3"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residues = ["GDP382", 382]
            residxs, fragidx = residue_and_atom.residues_from_descriptors(residues,
                                                                        self.by_bonds_geom2frags,
                                                                        self.geom2frags.top)
            self.assertSequenceEqual([7, 7], residxs)
            self.assertSequenceEqual([3, 3], fragidx)

    def test_fragment_name(self):
        residues = ["GLU30", 30]
        input_values = (val for val in ["0", "0"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residxs, fragidx = residue_and_atom.residues_from_descriptors(residues, self.by_bonds_geom2frags,
                                                                        self.geom2frags.top,
                                                                        fragment_names=["A", "B", "C", "D",
                                                                           "E", "F", "G", "H"])
            # Checking if residue names gives the correct corresponding residue id
            # NOTE:Enter 0 for GLU30 when asked "input one fragment idx"
            self.assertSequenceEqual([0, 0], residxs)
            self.assertSequenceEqual([0, 0], fragidx)

    def test_idx_not_present(self):
        residxs, fragidx = residue_and_atom.residues_from_descriptors(["GLU99", 99], self.by_bonds_geom2frags,
                                                                    self.geom2frags.top,
                                                                    pick_this_fragment_by_default=99)
        self.assertSequenceEqual([None, None], residxs)
        self.assertSequenceEqual([None, None], fragidx)

    def test_extra_dicts(self):
        residues = ["GLU30", "TRP32"]
        consensus_dicts = {"BW": {0: "3.50"},
                           "CGN": {2: "CGNt"}}
        residue_and_atom.residues_from_descriptors(residues, self.by_bonds_geom2frags,
                                                   self.geom2frags.top,
                                                   additional_resnaming_dicts=consensus_dicts,
                                                   pick_this_fragment_by_default=0
                                                   )

    def test_answer_letters(self):
        residues = ["GLU30", "TRP32"]
        input_values = (val for val in ["0.0", "4.0"])
        with mock.patch('builtins.input', lambda *x: next(input_values)):
            residue_and_atom.residues_from_descriptors(residues, self.by_bonds_geom2frags,
                                                     self.geom2frags.top,
                                                     )
            

class Test_rangeexpand_residues2residxs(unittest.TestCase):

    def setUp(self):
        self.top = md.load(test_filenames.small_monomer).top
        self.fragments = _mdcfrg.get_fragments(self.top, method="resSeq+",
                                                    verbose=False)

    def test_wildcards(self):
        expanded_range =   residue_and_atom.rangeexpand_residues2residxs("GLU*",
                                                                     self.fragments,
                                                                     self.top)
        np.testing.assert_array_equal(expanded_range, [0,4])

    def test_rangeexpand_res_idxs(self):
        expanded_range =   residue_and_atom.rangeexpand_residues2residxs("2-4,6",
                                                                     self.fragments,
                                                                     self.top,
                                                                     interpret_as_res_idxs=True)
        np.testing.assert_array_equal(expanded_range,[2,3,4,6])

    def test_rangeexpand_resSeq_w_jumps(self):
        expanded_range =   residue_and_atom.rangeexpand_residues2residxs("26-381",
                                                                     self.fragments,
                                                                     self.top,
                                                                     )
        np.testing.assert_array_equal(expanded_range,[3,4,5,6])

    def test_rangeexpand_resSeq_sort(self):
        expanded_range =   residue_and_atom.rangeexpand_residues2residxs("381,26",
                                                                     self.fragments,
                                                                     self.top,
                                                                     sort=True
                                                                     )
        np.testing.assert_array_equal(expanded_range,[3,6])


    def test_rangeexpand_raises_on_empty_range(self):
        with pytest.raises(ValueError):
            expanded_range =   residue_and_atom.rangeexpand_residues2residxs("50-60",
                                                                         self.fragments,
                                                                         self.top,
                                                                         )
    def test_rangeexpand_raises_on_empty_wildcard(self):
        with pytest.raises(ValueError):
            expanded_range =   residue_and_atom.rangeexpand_residues2residxs("ARG*",
                                                                         self.fragments,
                                                                         self.top,
                                                                         )

    def test_rangeexpand_w_ints_fails_as_resSeq(self):
        with pytest.raises(ValueError):
            residue_and_atom.rangeexpand_residues2residxs([0, 10, 20],
                                                          self.fragments,
                                                          self.top,
                                                          )

    def test_rangeexpand_w_ints_fails(self):
        expanded_range = residue_and_atom.rangeexpand_residues2residxs([0, 10, 20],
                                                                       self.fragments,
                                                                       self.top,
                                                                       interpret_as_res_idxs=True
                                                                       )
        np.testing.assert_array_equal(expanded_range,[0,10,20])


class Test_parse_and_list_AAs_input(unittest.TestCase):
    def setUp(self):
        self.top = md.load(test_filenames.small_monomer).top

    def test_None(self):
        f = io.StringIO()
        with redirect_stdout(f):
            residue_and_atom.parse_and_list_AAs_input(None, self.top)
        out = f.getvalue()
        assert out==""

    def test_prints(self):
        f = io.StringIO()
        with redirect_stdout(f):
            residue_and_atom.parse_and_list_AAs_input('GLU30,GLU31', self.top)
        out = f.getvalue().splitlines()
        np.testing.assert_equal(out[0],"0 GLU30")
        np.testing.assert_equal(out[1], "No %s found in the input topology"%"GLU31")

    def test_map_conlab(self):
        f = io.StringIO()
        with redirect_stdout(f):
            residue_and_atom.parse_and_list_AAs_input('GLU30,GLU31', self.top, map_conlab={0:'3.50'})
        out = f.getvalue().splitlines()
        np.testing.assert_equal(out[0],"0 GLU30 3.50")
        np.testing.assert_equal(out[1], "No %s found in the input topology"%"GLU31")


class Test_find_CA(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.pdb_3SN6)
        self.top = self.geom.top

    def test_works(self):
        CA = residue_and_atom.find_CA(self.top.residue(10))
        assert CA.name=="CA"

    def test_rules(self):
        res = residue_and_atom.find_AA(self.top,"P0G")[0]
        res = self.top.residue(res)
        CA = residue_and_atom.find_CA(res, CA_dict={"P0G":"CAA"})
        assert CA.name=="CAA"

    def test_just_one(self):
        geom = self.geom.atom_slice(self.top.select("name O"))
        res = geom.top.residue(10)
        CA = residue_and_atom.find_CA(res)
        assert CA.name == "O"

    def test_raises(self):
        with pytest.raises(NotImplementedError):
            CA = residue_and_atom.find_CA(self.top.residue(10), CA_name="CX")


if __name__ == '__main__':
    unittest.main()
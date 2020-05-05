import mdtraj as md
import numpy as np
import unittest
from filenames import filenames
from mdciao.residue_and_atom_utils import \
    find_AA, int_from_AA_code, name_from_AA, shorten_AA, _atom_type
import pytest

test_filenames = filenames()

class Test_find_by_AA(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)
        self.geom2frags = md.load(test_filenames.file_for_test_repeated_fullresnames_pdb)

    def test_long_AA_code(self):
        assert (find_AA(self.geom.top, "GLU30")) == [0]
        np.testing.assert_array_equal(find_AA(self.geom.top, "LYS29"),[5])

    def test_short_AA_code(self):
        assert (find_AA(self.geom.top, 'E30')) == [0]
        assert (find_AA(self.geom.top, 'W32')) == [2]

    def test_short_AA_code_relax(self):
         assert np.allclose(find_AA(self.geom.top, 'E', relax=True), [0,4])

    def test_long_AA_code_relax(self):
        assert np.allclose(find_AA(self.geom.top, 'GLU', relax=True), [0, 4])

    def test_relax_fails(self):
        with pytest.raises(ValueError):
            print((find_AA(self.geom.top, 'GLU'))) == [0]

    def test_does_not_find_AA(self):
        assert (find_AA(self.geom.top, "lys20")) == []   # small case won't give any result
        assert (find_AA(self.geom.top, 'w32')) == []    # small case won't give any result
        assert (find_AA(self.geom.top, 'w 32')) == []   # spaces between characters won't work

    def test_malformed_input(self):
        with pytest.raises(ValueError):
            find_AA(self.geom.top, "GLUTAMINE")

    def test_ambiguity(self):
        # AMBIGUOUS definition i.e. each residue is present in multiple fragments
        assert (find_AA(self.geom2frags.top, "LYS28")) == [5, 13] # getting multiple idxs,as expected
        assert (find_AA(self.geom2frags.top, "K28")) == [5, 13]

    def test_just_numbers(self):
        np.testing.assert_array_equal(find_AA(self.geom2frags.top,"28"),[5,13])

class Test_int_from_AA_code(unittest.TestCase):
    def test_int_from_AA_code(self):
        assert (int_from_AA_code("GLU30") == 30)
        assert (int_from_AA_code("E30") == 30)
        assert (int_from_AA_code("glu30") == 30)
        assert (int_from_AA_code("30glu40") == 3040)

class Test_name_from_AA(unittest.TestCase):
    def test_name_from_AA(self):
        assert(name_from_AA("GLU30") == 'GLU')
        assert (name_from_AA("E30") == 'E')

class Test_shorten_AA(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)

    def test_shorten_AA(self):
        assert(shorten_AA("GLU30") == 'E')
        assert(shorten_AA(self.geom.top.residue(1)) == 'V')

    def test_shorten_AA_substitute_fail_is_none(self):
        failed_assertion = False
        try:
            shorten_AA("glu30")
        except KeyError:
            failed_assertion = True
        assert failed_assertion

    def test_shorten_AA_substitute_fail_is_long(self):
        assert(shorten_AA("glu30", substitute_fail='long') == 'glu')

    def test_shorten_AA_substitute_fail_is_letter(self):
        assert(shorten_AA("glu30", substitute_fail='g') == 'g')

    def test_shorten_AA_substitute_fail_is_string_of_length_greater_than_1(self):
        failed_assertion = False
        try:
            shorten_AA("glu30", substitute_fail='glutamine')
        except ValueError:
            failed_assertion = True
        assert failed_assertion

    def test_shorten_AA_substitute_fail_is_0(self):
        assert(shorten_AA("glu30", substitute_fail=0) == 'g')

    def test_shorten_AA_substitute_fail_is_int(self):
        failed_assertion = False
        try:
            shorten_AA("glu30", substitute_fail=1)
        except ValueError:
            failed_assertion = True
        assert failed_assertion

    def test_shorten_AA_keep_index_is_true(self):
        assert(shorten_AA("GLU30", keep_index=True) == 'E30')
        assert(shorten_AA("glu30",substitute_fail='E',keep_index=True) == 'E30')

class Test_atom_type(unittest.TestCase):
    def test_works(self):
        top = md.load(test_filenames.prot1_pdb).top
        atoms_BB = [aa for aa in top.residue(0).atoms if aa.is_backbone]
        atoms_SC = [aa for aa in top.residue(0).atoms if aa.is_sidechain]
        atoms_X = [aa for aa in top.atoms if not aa.is_backbone and not aa.is_sidechain]
        assert len(atoms_BB)>0
        assert len(atoms_SC)>0
        assert len(atoms_X)>0
        assert all([_atom_type(aa)=="BB" for aa in atoms_BB])
        assert all([_atom_type(aa)=="SC" for aa in atoms_SC])
        assert all([_atom_type(aa)=="X" for aa in atoms_X])


if __name__ == '__main__':
    unittest.main()
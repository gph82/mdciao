import mdtraj as md
import unittest
from filenames import filenames
from mdciao.aa_utils import find_AA, int_from_AA_code, name_from_AA, shorten_AA

test_filenames = filenames()

class Test_find_by_AA(unittest.TestCase):
    def setUp(self):
        self.geom = md.load(test_filenames.file_for_test_pdb)
        self.geom2frags = md.load(test_filenames.file_for_test_repeated_fullresnames_pdb)

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

class Test_int_from_AA_code(unittest.TestCase):
    def test_int_from_AA_code_just_works(self):
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

    def test_shorten_AA_just_works(self):
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


if __name__ == '__main__':
    unittest.main()
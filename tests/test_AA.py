import mdtraj as md
import unittest
from filenames import filenames
from sofi_functions.aa_utils import find_AA, int_from_AA_code

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

if __name__ == '__main__':
    unittest.main()
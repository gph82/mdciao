import unittest

from mdciao import parsers

class TestParsersRun(unittest.TestCase):
    # This parsers will be tested anyway when testing the scripts,
    # but anyway, here we are
    def test_all(self):
        parsers.parser_for_frag_overview()
        parsers.parser_for_GPCR_overview()
        parsers.parser_for_densities()
        parsers.parser_for_CGN_overview()
        parsers.parser_for_rn()
        parsers.parser_for_interface()
        parsers.parser_for_dih()
        parsers.parser_for_sites()
        parsers.parser_for_compare_neighborhoods()
        parsers.parser_for_examples()
        parsers.parser_for_residues()

class Test_inform_of_parser(unittest.TestCase):

    def test_just_runs(self):
        p = parsers.parser_for_rn()
        parsers._inform_of_parser(p, [None,None])


if __name__ == '__main__':
    unittest.main()
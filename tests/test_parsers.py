import unittest

from mdciao import parsers

class TestParsersRun(unittest.TestCase):
    # This parsers will be tested anyway when testing the scripts,
    # but anyway, here we are
    def test_all(self):
        parsers.parser_for_contact_map()
        parsers.parser_for_frag_overview()
        parsers.parser_for_BW_overview()
        parsers.parser_for_densities()
        parsers.parser_for_CGN_overview()
        parsers.parser_for_rn()
        parsers.parser_for_interface()
        parsers.parser_for_dih()
        parsers.parser_for_sites()


if __name__ == '__main__':
    unittest.main()
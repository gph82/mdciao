import os
from shutil import rmtree
import unittest
import mdciao.examples
from tempfile import mkdtemp
import contextlib
@contextlib.contextmanager
def remember_cwd():
    curdir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(curdir)

class Test_ExamplesCLTs(unittest.TestCase):

    # It's already a test!
    def setUp(self):
        self.xCLTs = mdciao.examples.ExamplesCLTs(
            test=True
        )
        self.tmpdir = mkdtemp(suffix="mdciao_tests")
        print(self.tmpdir)
    def tearDown(self):
        rmtree(self.tmpdir)

    def test_mdc_sites(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_sites",
                                #show=True
                                )
        assert CP.returncode==0

    def test_mdc_neighborhood(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_neighborhoods",
                                #show=True
                                )
        assert CP.returncode==0


    def test_mdc_interface(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_interface",
                                #show=True
                                )
        assert CP.returncode==0

    def test_mdc_fragments(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_fragments",
                                #show=True
                                )
        assert CP.returncode==0

    def test_mdc_BW(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_GPCR_overview",
                                #show=True
                                )
        assert CP.returncode==0

    def test_mdc_CGN(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_CGN_overview",
                                #show=True
                                )
        assert CP.returncode == 0

    def test_mdc_pdb(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_pdb",
                                #show=True
                                )
        assert CP.returncode == 0

    def test_mdc_compare(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            self.xCLTs.run("mdc_compare")


class Test_ContactGroupL394(unittest.TestCase):

    def test_works(self):
        CG = mdciao.examples.ContactGroupL394()
        assert isinstance(CG,mdciao.contacts.ContactGroup)

    def test_except(self):
        with self.assertRaises(Exception):
            CG = mdciao.examples.ContactGroupL394(bogus_arg="bogus")


if __name__ == '__main__':
    unittest.main()
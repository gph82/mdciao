import os
from shutil import rmtree
import unittest
import mdciao.examples
from tempfile import TemporaryDirectory as _TDir, mkdtemp
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
            CP = self.xCLTs.run("mdc_BW_overview",
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


if __name__ == '__main__':
    unittest.main()
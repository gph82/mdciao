import os
from shutil import rmtree
import unittest
from mdciao.examples import examples, filenames
from mdciao import contacts
from tempfile import mkdtemp
import contextlib
import numpy as _np


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
        self.xCLTs = examples.ExamplesCLTs(
            test=True
        )
        self.tmpdir = mkdtemp(suffix="mdciao_tests")
        print(self.tmpdir)

        for fn in [filenames.traj_xtc,
                   filenames.top_pdb,
                   filenames.adrb2_human_xlsx,
                   filenames.gnas2_human_xlsx,
                   filenames.tip_json
                   ]:
            os.symlink(fn, os.path.join(self.tmpdir, os.path.basename(fn)))

    def tearDown(self):
        rmtree(self.tmpdir)

    def test_mdc_sites(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_sites",
                                # show=True
                                )
        assert _np.unique([iCP.returncode for iCP in CP]) == 0

    def test_mdc_neighborhood(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_neighborhoods",
                                # show=True
                                )
        assert _np.unique([iCP.returncode for iCP in CP]) == 0

    def test_mdc_interface(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_interface",
                                # show=True
                                )
        assert _np.unique([iCP.returncode for iCP in CP]) == 0

    def test_mdc_fragments(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_fragments",
                                # show=True
                                )
        assert _np.unique([iCP.returncode for iCP in CP]) == 0

    def test_mdc_BW(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_GPCR_overview",
                                # show=True
                                )
        assert _np.unique([iCP.returncode for iCP in CP]) == 0, CP

    def test_mdc_CGN(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_CGN_overview",
                                # show=True
                                )
        assert _np.unique([iCP.returncode for iCP in CP]) == 0

    def test_mdc_pdb(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_pdb",
                                # show=True
                                )
        assert _np.unique([iCP.returncode for iCP in CP]) == 0

    def test_mdc_compare(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_compare")
            assert _np.unique([iCP.returncode for iCP in CP]) == 0

    def test_mdc_residues(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_residues")
            assert _np.unique([iCP.returncode for iCP in CP]) == 0

    def test_mdc_notebooks(self):
        with remember_cwd():
            os.chdir(self.tmpdir)
            CP = self.xCLTs.run("mdc_notebooks")
            assert _np.unique([iCP.returncode for iCP in CP]) == 0


if __name__ == '__main__':
    unittest.main()

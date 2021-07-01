import os
import io as _io
from contextlib import redirect_stdout
import unittest
from mdciao.examples import examples
from mdciao import contacts
from tempfile import mkdtemp, TemporaryDirectory
import contextlib
@contextlib.contextmanager
def remember_cwd():
    curdir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(curdir)

from mdciao.examples import filenames
class Test_ContactGroupL394(unittest.TestCase):

    def test_works(self):
        CG = examples.ContactGroupL394()
        assert isinstance(CG,contacts.ContactGroup)

    def test_except(self):
        with self.assertRaises(Exception):
            CG = examples.ContactGroupL394(bogus_arg="bogus")

class Test_Zip(unittest.TestCase):

    def test_works(self):
        with TemporaryDirectory(suffix="_mdciao_test_zip") as td:
            zipfile = os.path.join(td,os.path.basename(filenames.zipfile_two_empties))
            os.symlink(filenames.zipfile_two_empties, zipfile)
            unzipped = examples._unzip2dir(zipfile)
            assert unzipped == os.path.splitext(zipfile)[0]
            os.path.exists(os.path.join("A.txt"))
            os.path.exists(os.path.join("B.txt"))

    def test_doesn_overwrite(self):
        with TemporaryDirectory(suffix="_mdciao_test_zip") as td:
            zipfile = os.path.join(td,os.path.basename(filenames.zipfile_two_empties))
            os.symlink(filenames.zipfile_two_empties, zipfile)
            unzipped = examples._unzip2dir(zipfile)
            b = _io.StringIO()
            with redirect_stdout(b) as stdout:
                unzipped = examples._unzip2dir(zipfile)
            stdout : _io.StringIO
            output = stdout.getvalue().splitlines()
            assert len(output)==3
            assert output[0]=="Unzipping to %s/two_empty_files"%td
            assert output[1]=="No unzipping of A.txt: file already exists."
            assert output[2]=="No unzipping of B.dat: file already exists."


if __name__ == '__main__':
    unittest.main()
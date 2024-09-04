import os
import io as _io
from contextlib import redirect_stdout
import unittest
from mdciao.examples import examples
from mdciao import contacts
from tempfile import mkdtemp, TemporaryDirectory
from unittest import mock
from glob import glob

import contextlib
@contextlib.contextmanager
def remember_cwd():
    curdir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(curdir)

from mdciao.examples import filenames as test_filenames
class Test_ContactGroupL394(unittest.TestCase):

    def test_works(self):
        CG = examples.ContactGroupL394()
        assert isinstance(CG,contacts.ContactGroup)

    def test_except(self):
        with self.assertRaises(Exception):
            CG = examples.ContactGroupL394(bogus_arg="bogus")

class Test_Interface_B2AR_Gas(unittest.TestCase):
    def test_works(self):
        intf = examples.Interface_B2AR_Gas()
        assert intf.is_interface

    def test_except(self):
        with self.assertRaises(Exception):
            CG = examples.Interface_B2AR_Gas(bogus_arg="bogus")

class Test_KLIFSLabeler_P31751(unittest.TestCase):
    def test_works(self):
        KLIFS = examples.KLIFSLabeler_P31751()
        assert KLIFS._conlab_column == "KLIFS"

class Test_AlignerConsensus_B2AR_HUMAN_vs_OPSD_BOVIN(unittest.TestCase):
    def test_works(self):
        AC = examples.AlignerConsensus_B2AR_HUMAN_vs_OPSD_BOVIN()
        self.assertDictEqual({'B2AR': 'R131', 'OPS': 'R135'},
                             AC.AAresSeq_match("3.50x50")[["B2AR","OPS"]].squeeze().to_dict())

class Test_Zip(unittest.TestCase):

    def test_works(self):
        with TemporaryDirectory(suffix="_mdciao_test_zip") as td:
            zipfile = os.path.join(td,os.path.basename(test_filenames.zipfile_two_empties))
            os.symlink(test_filenames.zipfile_two_empties, zipfile)
            unzipped = examples._unzip2dir(zipfile)
            assert unzipped == os.path.splitext(zipfile)[0]
            os.path.exists(os.path.join("A.txt"))
            os.path.exists(os.path.join("B.txt"))

    def test_doesn_overwrite(self):
        with TemporaryDirectory(suffix="_mdciao_test_zip") as td:
            zipfile = os.path.join(td,os.path.basename(test_filenames.zipfile_two_empties))
            os.symlink(test_filenames.zipfile_two_empties, zipfile)
            unzipped = examples._unzip2dir(zipfile)
            b = _io.StringIO()
            with redirect_stdout(b) as stdout:
                unzipped = examples._unzip2dir(zipfile)
            stdout : _io.StringIO
            output = stdout.getvalue().splitlines()
            assert len(output)==3
            assert output[0]=="Unzipping to '%s/two_empty_files'"%td
            assert output[1]=="No unzipping of 'A.txt': file already exists."
            assert output[2]=="No unzipping of 'B.dat': file already exists."

class Test_recursive_funct(unittest.TestCase):
    def test_works_as_first(self):
        res = examples._recursive_prompt("this_file_doesnt_exist.txt","this_file_doesnt_exist")
        assert res =="this_file_doesnt_exist.txt"

    def test_has_to_iterate_once(self):
        with TemporaryDirectory(suffix="_mdciao_test_recursive") as td:
            with remember_cwd():
                os.chdir(td)
                open("this_file_exists.txt","w").close()
                input_values = (val for val in [""])
                with mock.patch('builtins.input', lambda *x: next(input_values)):
                    res = examples._recursive_prompt("this_file_exists.txt", "this_file_exists",is_file=True,verbose=True)
                    assert os.path.realpath(res)==os.path.join(os.path.realpath(td),"this_file_exists_01.txt")

    def test_has_to_enter_recursion(self):
        with TemporaryDirectory(suffix="_mdciao_test_recursive") as td:
            with remember_cwd():
                os.chdir(td)
                open("this_file_exists.txt","w").close()
                open("this_file_exists_01.txt","w").close()
                input_values = (val for val in ["this_file_exists_01.txt",""])
                with mock.patch('builtins.input', lambda *x: next(input_values)):
                    res = examples._recursive_prompt("this_file_exists.txt", "this_file_exists",is_file=True,verbose=True)
                    assert os.path.realpath(res)==os.path.join(os.path.realpath(td),"this_file_exists_03.txt")

    def test_new_file_is_good(self):
        with TemporaryDirectory(suffix="_mdciao_test_recursive") as td:
            with remember_cwd():
                os.chdir(td)
                open("this_file_exists.txt","w").close()
                open("this_file_exists_01.txt","w").close()
                input_values = (val for val in ["new_file_totally_different.txt",""])
                with mock.patch('builtins.input', lambda *x: next(input_values)):
                    res = examples._recursive_prompt("this_file_exists.txt", "this_file_exists",is_file=True,verbose=True)
                    assert os.path.realpath(res)==os.path.join(os.path.realpath(td),"new_file_totally_different.txt")

    def test_escapes_recursion(self):
        with TemporaryDirectory(suffix="_mdciao_test_recursive") as td:
            with remember_cwd():
                os.chdir(td)
                for ii in range(50):
                    open("test_%02u.dat" % ii, "w").close()
                with self.assertRaises(RecursionError):
                    examples._recursive_prompt("test_00.dat", "test", is_file=True)

    def test_skip_on_existing(self):
        with TemporaryDirectory(suffix="_mdciao_test_recursive") as td:
            with remember_cwd():
                os.chdir(td)
                open("test.00.dat", "w").close()
                examples._recursive_prompt("test_00.dat", "test", is_file=True, skip_on_existing=True)

class Test_down_safely(unittest.TestCase):

    def test_just_works(self):
        with TemporaryDirectory(suffix="_mdciao_test_down_safely") as td:
            with remember_cwd():
                os.chdir(td)
                local_path = examples._down_url_safely("https://proteinformatics.uni-leipzig.de/mdciao/mdciao_test_small.zip",verbose=True)
                assert os.path.exists(local_path)
    def test_different_name(self):
        with TemporaryDirectory(suffix="_mdciao_test_down_safely") as td:
            with remember_cwd():
                os.chdir(td)
                local_path = examples._down_url_safely("https://proteinformatics.uni-leipzig.de/mdciao/mdciao_test_small.zip",verbose=True,
                                                       rename_to="myfile.zip")
                assert local_path.endswith("myfile.zip")
                assert os.path.exists(local_path)

    def test_skip_on_existing(self):
        with TemporaryDirectory(suffix="_mdciao_test_down_safely") as td:
            with remember_cwd():
                os.chdir(td)
                with open("mdciao_test_small.zip","w") as f:
                    f.write("Won't be overwrriten")
                local_path = examples._down_url_safely("https://proteinformatics.uni-leipzig.de/mdciao/mdciao_test_small.zip",
                                                       verbose=True, skip_on_existing=True)
                assert open("mdciao_test_small.zip").read() == "Won't be overwrriten"
                assert local_path.endswith("mdciao_test_small.zip")
                assert os.path.exists(local_path)

class Test_fetch_example_data(unittest.TestCase):

    def test_just_works(self):
        with TemporaryDirectory(suffix="_mdciao_test_fetch") as td:
            with remember_cwd():
                os.chdir(td)
                local_path = examples.fetch_example_data("https://proteinformatics.uni-leipzig.de/mdciao/mdciao_test_small.zip",unzip=False)
                assert os.path.exists(local_path)
                #assert os.path.exists((os.path.splitext(local_path))[0])
                files =  os.listdir(td)
                assert len(files)==1
                assert files[0]=="mdciao_test_small.zip"

    def test_just_unzip(self):
        with TemporaryDirectory(suffix="_mdciao_test_fetch") as td:
            with remember_cwd():
                os.chdir(td)
                local_path = examples.fetch_example_data("https://proteinformatics.uni-leipzig.de/mdciao/mdciao_test_small.zip",unzip=True)
                assert os.path.exists(local_path)
                assert os.path.exists((os.path.splitext(local_path))[0])
                files =  sorted(os.listdir(td))
                assert len(files)==2
                assert files[0]=="mdciao_test_small"
                assert files[1]=="mdciao_test_small.zip"
                unzipped_files =sorted(os.listdir(files[0]))
                assert len(unzipped_files)==2
                assert unzipped_files[0]=="A.dat"
                assert unzipped_files[1]=="B.dat"

    def test_alias(self):
        with TemporaryDirectory(suffix="_mdciao_test_fetch") as td:
            with remember_cwd():
                os.chdir(td)
                local_path = examples.fetch_example_data("test",
                                                         unzip=False)
                assert os.path.exists(local_path)
                # assert os.path.exists((os.path.splitext(local_path))[0])
                files = os.listdir(td)
                assert len(files) == 1
                assert files[0] == "mdciao_test_small.zip"


    def test_alias_unzip_to_otherfile(self):
        with TemporaryDirectory(suffix="_mdciao_test_fetch") as td:
            with remember_cwd():
                os.chdir(td)
                local_path = examples.fetch_example_data("test",
                                                         unzip="unzip_here")
                assert os.path.exists(local_path)
                # assert os.path.exists((os.path.splitext(local_path))[0])
                files = os.listdir(td)
                assert len(files) == 2
                assert files[0] == "unzip_here.zip"
                assert files[1] == "unzip_here"
                extracted = sorted(os.listdir(files[1]))
                assert extracted[0] == "A.dat"
                assert extracted[1] == "B.dat"


    def test_skip_on_existing(self):
        with TemporaryDirectory(suffix="_mdciao_test_fetch") as td:
            with remember_cwd():
                os.chdir(td)
                local_path = examples.fetch_example_data("test",
                                                         unzip=False)
                assert os.path.exists(local_path)
                # assert os.path.exists((os.path.splitext(local_path))[0])
                files = os.listdir(td)
                assert len(files) == 1
                assert files[0] == "mdciao_test_small.zip"
                # Create a fake file to test it doesn't ovewrite
                with open("mdciao_test_small.zip", "w") as f:
                    f.write("Won't be overwrriten")
                local_path = examples.fetch_example_data("test",
                                                         unzip=False, skip_on_existing=True)
                assert open("mdciao_test_small.zip").read() == "Won't be overwrriten"


class Test_notebooks(unittest.TestCase):

    def test_just_works(self):
        with TemporaryDirectory(suffix="_mdciao_test_notebooks") as td:
            new_notebook_path = examples.notebooks(os.path.join(td,"mdciao_notebooks"))
            assert os.path.realpath(new_notebook_path) != os.path.realpath(test_filenames.notebooks_path)
            new_nbs = sorted([os.path.basename(ff) for ff in os.listdir(new_notebook_path)])
            old_nbs = sorted([os.path.basename(ff) for ff in glob(os.path.join(test_filenames.notebooks_path,"*ipynb"))])
            self.assertListEqual(new_nbs, old_nbs)


class Test_Filenames(unittest.TestCase):

    def test_works(self):
        for attr in dir(examples.filenames):
            if not attr.startswith("_"):
                assert os.path.exists(getattr(examples.filenames,attr))

if __name__ == '__main__':
    unittest.main()

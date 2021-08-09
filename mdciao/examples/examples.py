##############################################################################
#    This file is part of mdciao.
#    
#    Copyright 2020 Charité Universitätsmedizin Berlin and the Authors
#
#    Authors: Guillermo Pérez-Hernandez
#    Contributors:
#
#    mdciao is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    mdciao is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with mdciao.  If not, see <https://www.gnu.org/licenses/>.
##############################################################################

from os import path as _path, getcwd as _getcwd, chdir as _chdir, link as _link, mkdir as _mkdir
from shutil import copy as _shcopy
from glob import glob as _glob
from mdciao import __path__ as mdc_path
from subprocess import run as _run
from zipfile import ZipFile as _ZF
from sys import prefix as _env_prefix

mdc_path = _path.split(mdc_path[0])[0]

long2short = {"--residues" : "-r",
              "--n_smooth_hw" : "-ns",
              "--table_ext" : "-tx",
              "--GPCR_uniprot" : "--GPCR",
              "--CGN_PDB"   : "--CGN"
              }

long2long = {key:key for key in long2short.keys()}

from requests import get as _rget
from tqdm.auto import tqdm as _tqdma
import contextlib as _contextlib
from mdciao.cli import residue_neighborhoods as _residue_neighborhoods
from tempfile import TemporaryDirectory as _TDir, TemporaryFile as _TF
import io as _io
@_contextlib.contextmanager
def remember_cwd():
    curdir = _getcwd()
    try:
        yield
    finally:
        _chdir(curdir)

class ExamplesCLTs(object):
    def __init__(self, test=False, short=False):
        r"""
        API interface to show or call the mdc_* scripts from the command line directly

        For instance, self.run("mdc_interface") will run an mdc_interface.py example

        Parameters
        ----------
        test : bool, default is False
            If True, internally all references to input filenames
            are converted from full paths to basenames, because
            it is assumed that the test method is calling the
            self.run() method form a temporary directory
            where links to these filenames have been
            created already. Additionally, the flag
            "-ni" for non-interactive runs is added to the methods
            that would otherwise prompt the user for some confirmations
        short : bool, default is False
            Whether to use the short or the long versions
            of the flags
        """
        self.xtc = filenames.traj_xtc
        self.pdb = filenames.top_pdb
        self.GPCRlabs_file = filenames.adrb2_human_xlsx
        self.CGN_file = filenames.CGN_3SN6
        self.sitefile = filenames.tip_json
        self.pdb_3SN6 = filenames.pdb_3SN6

        self.test = test
        cwd = _getcwd()
        for fn in ["xtc", "pdb", "GPCRlabs_file", "CGN_file", "sitefile", "pdb_3SN6"]:
            attr_val = getattr(self, fn)
            if self.test:
                setattr(self, fn, _path.basename(attr_val))
            else:
                setattr(self, fn, _path.relpath(attr_val,cwd))

        if short:
            self.opt_dict=long2short
        else:
            self.opt_dict = long2long
    @property
    def mdc_neighborhoods(self):
        return ["mdc_neighborhoods.py",
                "%s %s" % (self.pdb, self.xtc),
                self.opt_dict["--residues"] + " L394",
                self.opt_dict["--n_smooth_hw"] + " 1",
                self.opt_dict["--table_ext"] + " xlsx",
                self.opt_dict["--GPCR_uniprot"] + " %s" % self.GPCRlabs_file,
                self.opt_dict["--CGN_PDB"] + " %s" % self.CGN_file,
                ]
    @property
    def mdc_sites(self):
        return ["mdc_sites.py ",
                "%s %s" % (self.pdb, self.xtc),
                " --site_files %s" % self.sitefile,
                " --GPCR_uniprot %s" % self.GPCRlabs_file,
                " --CGN_PDB %s" % self.CGN_file
                ]

    @property
    def mdc_interface(self):
        return ["mdc_interface.py ",
                "%s %s" % (self.pdb, self.xtc),
                " --frag_idxs_group_1 0-2",
                " --frag_idxs_group_2 3",
                " --ctc_control 20",
                " --GPCR_uniprot %s" % self.GPCRlabs_file,
                " --CGN_PDB %s" % self.CGN_file,
                ]
    @property
    def mdc_GPCR_overview(self):
        return ["mdc_GPCR_overview.py",
                "%s" % self.GPCRlabs_file,
                "-t %s" % self.pdb]

    @property
    def mdc_CGN_overview(self):
        # This is the only one that needs network access
        return ["mdc_CGN_overview.py",
                "%s" % '3SN6',
                "-t %s" % self.pdb,
                ]

    @property
    def mdc_compare(self):
        return ["mdc_neighborhoods.py",
                "%s %s" % (self.pdb, self.xtc),
                self.opt_dict["--residues"] + " L394",
                self.opt_dict["--n_smooth_hw"] + " 1",
                self.opt_dict["--table_ext"] + " xlsx",
                "--ctc_cutoff_Ang 3",
                "\n\n",
                "mdc_neighborhoods.py",
                "%s %s" % (self.pdb, self.xtc),
                self.opt_dict["--residues"] + " L394",
                self.opt_dict["--n_smooth_hw"] + " 1",
                self.opt_dict["--table_ext"] + " xlsx",
                "--ctc_cutoff_Ang 4",
                "\n\n"
                "mdc_compare.py",
                "neighborhood.LEU394@frag0@3.0_Ang.xlsx",
                "neighborhood.LEU394@frag0@4.0_Ang.xlsx"
                ]
        pass

    @property
    def mdc_fragments(self):
        return ["mdc_fragments.py ",
                "%s" % (self.pdb)
                ]
        pass

    @property
    def mdc_pdb(self):
        return ["mdc_pdb.py 3SN6"]

    @property
    def mdc_residues(self):
        return ["mdc_residues.py ",
                "P0G,380-394,3.5* "
                "%s"% (self.pdb),
                " --GPCR_uniprot %s" % self.GPCRlabs_file,
                "-ni"]

    @property
    def mdc_notebooks(self):
        return ["mdc_notebooks.py "]

    @property
    def clts(self):
        return [attr for attr in dir(self) if attr.startswith("mdc")]

    def _assert_clt_exists(self, clt):
        assert clt in self.clts, "Input method %s is not in existing methods %s" % (clt, self.clts)

    def _join_args(self,clt):
        oneline = self.__getattribute__(clt)
        if self.test and all([istr not in clt for istr in ["_overview", "mdc_fragments", "mdc_pdb"]]):
            oneline += ["-ni"]
        return " ".join(oneline)

    def show(self, clt):
        self._assert_clt_exists(clt)
        print("%s example call:" % clt)
        print("%s--------------"%("".join(["-" for _ in clt]))) # really?
        oneline = self._join_args(clt)
        print(oneline.replace(" -", " \n-"))
        print("\n\nYou can re-run 'mdc_examples %s.py' with the '-x' option to execute the command directly\n"
              "or you can paste the line below into your terminal, add/edit options and then execute:\n"%clt)
        print(oneline)

    def run(self, clt,show=True, output_dir="."):
        if show:
            self.show(clt)
        oneline = self._join_args(clt)
        #if self.test:
        #    oneline = oneline

        CP = _run(oneline.split())
        if self.test:
            return CP


def ContactGroupL394(**kwargs):
    r"""
    Create an example :obj:`mdciao.contacts.ContactGroup` quickly.

    Wraps around :obj:`mdciao.cli.residue_neighborhoods` and asks
    for the LEU394 neighborhood.

    The input data is one very short (80 frames) version
    of the MD trajectory shipped with mdciao, kindly provided by
    Dr. H. Batebi. See the online examples for more info.

    Parameters
    ----------
    kwargs : optional keyword arguments
        For :obj:`mdciao.cli.residue_neighborhoods`

    Returns
    -------
    CG : a :obj:`~mdciao.contacts.ContactGroup`

    """
    with _TDir(suffix="_mdciao_example_CG") as t:
        for fn in [filenames.pdb_3SN6, filenames.traj_xtc,
                   filenames.top_pdb,
                   filenames.adrb2_human_xlsx, filenames.CGN_3SN6]:
            _link(fn, _path.join(t, _path.basename(fn)))

        with remember_cwd():
            _chdir(t)
            b = _io.StringIO()
            try:
                with _contextlib.redirect_stdout(b):
                    example_kwargs = {"topology": _path.basename(filenames.top_pdb),
                                      "n_smooth_hw": 1,
                                      "figures": False,
                                      "GPCR_uniprot": _path.basename(filenames.adrb2_human_xlsx),
                                      "CGN_PDB": _path.basename(filenames.CGN_3SN6),
                                      "accept_guess": True}
                    for key, val in kwargs.items():
                        example_kwargs[key]=val
                    return _residue_neighborhoods("L394",
                                                  _path.basename(filenames.traj_xtc),
                                                  **example_kwargs,
                                                  )["neighborhoods"][353]

            except Exception as e:
                print(b.getvalue())
                b.close()
                raise e

def notebooks(folder ="mdciao_notebooks"):
    r"""
    Copy the example Jupyter notebooks distributed with mdciao to this folder.

    The method never overwrites an existing folder, but
    keeps either asking or producing new folder names.

    Parameters
    ----------
    folder : str

    Returns
    -------
    folder : str
        The folder to which the notebooks were copied
        Can be identical to the input or one generated
        by :obj:`_recursive_prompt`.

    """

    pwd = _getcwd()

    nbs = sorted(_glob(_path.join(filenames.notebooks_path,"*.ipynb")))

    dest = _recursive_prompt(_path.join(pwd, folder), folder, verbose=True)
    print("Copying file(s):"
          "\n %s"%"\n -".join(nbs))
    _mkdir(dest)
    print("Here::")
    for ff in nbs:
        _shcopy(ff,dest)
        print(" %s"%_path.join(dest,_path.basename(ff)))

    return dest

def _recursive_prompt(input_path, pattern, count=1, verbose=False, is_file=False):
    r"""
    Ensure input_path doesn't exist and keep generating/prompting for alternative filenames/dirnames


    Poorman's attempt at actually a good recursive function, but does its job.
    A maximum recursion depth of 50 is hard-coded

    Parameters
    ----------
    input_path : str
        Example, "path/to/mdicao_example.zip"
    pattern : str
        The part of input_path used to generate new filenames,
        example "mdciao_example"
    count : int, default is 0
        Where in the recursion we are
    verbose : bool, default is False
    is_file : book, default is False
        Name-generating is different for folders than
        from files:
        * mdciao_notebook -> mdciao_notebook_00
        * mdciao_example.zip -> mdciao_example_00.zip

    Returns
    -------
    nox_path : str
        A newly created, previosuly non-existent path

    """

    max_recurr = 50
    cwd = _getcwd()
    ext = _path.splitext(input_path)[1]
    while _path.exists(input_path):
        if verbose:
            print("%s exists" % input_path)
        input_path = _path.join(cwd, pattern) + "_%02u" % count
        if is_file:
            input_path += ext
        count += 1
        if count > max_recurr:
            raise RecursionError("Over %u files/folders exist with the '%s' pattern, aborting" % (max_recurr, pattern))

    if count>1:
        print(input_path, "will be created")
        print("Hit Enter to accept or provide another path from %s%s:\r"%(cwd,_path.sep))
        answer = input()
    else:
        answer = ""
    if len(answer)==0:
        return input_path
    else:
        input_path = _path.join(cwd, answer)
        print("OK, your suggestion is",input_path)
    if _path.exists(input_path):
        print("%s already exists. Next suggestion:" % input_path)
        return _recursive_prompt(input_path, pattern, count=count, verbose=verbose, is_file=is_file)
    else:
        return _path.join(cwd,answer)


def fetch_example_data(url="http://proteinformatics.org/mdciao/mdciao_example.zip",
                       unzip=True):
    r"""
    Download the example data from the url and unzip it to the working directory.

    This example data is kindly provided by Dr. H. Batebi.

    New filenames for the downloaded file, and the resulting folder
    will be generated to avoid overwriting.

    No files will be overwritten when extracting.

    Parameters
    ----------
    url : str
        The url to download from
    unzip : bool, default is True
        Try unzipping the file
        after downloading

    Returns
    -------
    downed_fold_full : str
        The full path to the downloaded data

    """
    downed_file_full = _down_url_safely(url)
    if unzip:
        return _unzip2dir(downed_file_full)
    else:
        return  downed_file_full

def _unzip2dir(full_path_zipfile):
    r"""
    Unzip to a folder with the same name as the file, regardless of the structure of the zipfile

    The folder's full path is kept, including zipfile's name minus the .zip extension

    Background: "mdciao_example.zip" was zipped in origin with this structure:
     * mdciao_example/prot.pdb
     * mdciao_example/traj.xtc

    However, it might have been renamed to "mdciao_example_05.zip" when auto-downloading.
    If you unzip it directly, it will be extracted to mdciao_example, which ideally
    should be avoided.

    If for whatever reason, mdciao_example_05 *does* exist as a folder, existing files
    will not be overwritten

    Parameters
    ----------
    full_path_zipfile : str
        The zipfile. E.g. "Downloads/mdciao_example_00.zip" will be
        extracted to "Downloads/mdciao_example_00/"

    Returns
    -------
    full_dir : str
        The directory into which the zipfile was extracted

    """
    full_dir = _path.splitext(full_path_zipfile)[0]
    # https://stackoverflow.com/a/56362289
    #local_dir = _path.basename(full_dir)
    print("Unzipping to '%s'" % full_dir)
    with _ZF(full_path_zipfile) as zipdata:
        for zipinfo in zipdata.infolist():
            new = _path.basename(zipinfo.filename)
            zipinfo.filename = new
            if _path.exists(_path.join(full_dir,new)):
                print("No unzipping of %s: file already exists." % new)
            else:
                zipdata.extract(zipinfo,
                                path=full_dir
                                )
            #assert _path.exists(_path.join(full_dir,new))

    return full_dir

def _down_url_safely(url, chunk_size = 128, verbose=False):
    r"""
    Downloads a file from a URL to a tmpfile and copies it to the current directory

    If the file/folder exists already, a new filename is generated using _recursive_prompt

    Parameters
    ----------
    url
    chunk_size

    Returns
    -------

    """
    filename_orig = _path.basename(url)
    filename_nonx = _recursive_prompt(filename_orig,
                                      _path.splitext(filename_orig)[0],
                                      is_file=True, verbose=True)
    r = _rget(url, stream=True)
    total_size_in_bytes = int(r.headers.get('content-length', 0))
    pb = _tqdma(total=total_size_in_bytes,
                desc="Downloading %s to %s" % (filename_orig, _path.basename(filename_nonx)))
    with _TDir(suffix="_mdciao_download") as t:
        _filename = _path.join(t,_path.basename(filename_nonx))
        with open(_filename, 'wb') as fd:
            r.iter_content()
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
                pb.update(128)
        pb.close()
        if verbose:
            print("Dowloaded file %s"%filename_nonx)
        _shcopy(_filename,filename_nonx)

    return filename_nonx


#Check https://docs.python.org/3/library/sys.html#sys.prefix
class Filenames(object):
    r"""
    A class that contains the paths to the files used by mdciao.

    Note
    ----
    Many of these files don't necessarily ship with mdciao unless
    you downloaded mdciao's source, e.g. by cloning the github repo.

    """
    def __init__(self):
        # Check
        # https://docs.python.org/3.7/tutorial/modules.html#packages-in-multiple-directories
        from mdciao import __path__ as sfpath
        assert len(sfpath) == 1
        sfpath = _path.split(sfpath[0])[0].rstrip("/")

        if sfpath.startswith(_env_prefix):
            if sfpath.endswith(".egg"):
                rootdir = sfpath # we're a python setup.py install
            else:
                rootdir = _env_prefix # we're a "normal" pip/conda installation
            self.test_data_path = _path.join(rootdir, "data_for_mdciao")
            self.notebooks_path = _path.join(self.test_data_path, "notebooks")
        else:
            self.test_data_path = _path.join(sfpath, "tests", "data") # we're a python setup.py develop
            self.notebooks_path = _path.join(sfpath,"mdciao","examples")

        self.bogus_pdb_path = _path.join(self.test_data_path, "bogus_pdb")
        self.RSCB_pdb_path =  _path.join(self.test_data_path,"RSCB_pdb" )
        self.example_path =   _path.join(self.test_data_path,"examples")
        self.nomenclature_path = _path.join(self.test_data_path,"nomenclature")
        self.json_path = _path.join(self.test_data_path,"json")

        # pdbs for testing
        self.small_monomer = _path.join(self.bogus_pdb_path,
                                       "2_3AA_chains_and_two_ligs_monomer.pdb")
        self.file_for_no_bonds_gro = _path.join(self.bogus_pdb_path,
                                       "2_3AA_chains_and_two_ligs_monomer.gro")
        self.small_dimer = _path.join(self.bogus_pdb_path,
                                     "2_3AA_chains_and_two_ligs_dimer.pdb")
        # Force a break in the resSeq
        self.small_monomer_LYS99 = _path.join(self.bogus_pdb_path,
                                             "2_3AA_chains_and_two_ligs_monomer.LYS29toLYS99.pdb")

        self.actor_pdb = _path.join(self.example_path,"prot1.pdb.gz")

        self.ions_and_water = _path.join(self.bogus_pdb_path, "water_and_ions.pdb.gz")

        # Pure-PDBs
        self.pdb_3CAP = _path.join(self.RSCB_pdb_path,"3cap.pdb.gz")
        self.pdb_1U19 = _path.join(self.RSCB_pdb_path,"1u19.pdb.gz")
        self.pdb_3SN6 = _path.join(self.RSCB_pdb_path,"3SN6.pdb.gz")

        # Traj
        self.top_pdb = _path.join(self.example_path,"gs-b2ar.noH.pdb")
        # TODO/NOTE the time-array of the stride_20 xtc does not start at zero,
        # this helps debug things with time
        # the unstrided xtc, used in examples and doc, DOES start at zero for clarity
        self.traj_xtc_stride_20 = _path.join(self.example_path, "gs-b2ar.noH.stride.20.xtc")
        self.traj_xtc = _path.join(self.example_path, "gs-b2ar.noH.stride.5.xtc")

        # nomenclature
        self.CGN_3SN6 = _path.join(self.nomenclature_path,"CGN_3SN6.txt")
        self.GPCRmd_B2AR_nomenclature_test_xlsx = _path.join(self.nomenclature_path,"GPCRmd_B2AR_nomenclature_test.xlsx")
        self.pdb_3SN6_mut = _path.join(self.nomenclature_path, "3SN6_GLU10GLX.pdb.gz")
        self.adrb2_human_xlsx = _path.join(self.nomenclature_path,"adrb2_human.xlsx")

        #json
        self.GDP_json = _path.join(self.json_path,"GDP.json")
        self.GDP_name_json = _path.join(self.json_path,"GDP_name_XXX.json")
        self.tip_json = _path.join(self.json_path,"tip.json")
        self.tip_dat= _path.join(self.json_path,"tip.dat")
        self.tip_residx_dat= _path.join(self.json_path,"tip_residx.dat")

        #zip
        self.zipfile_two_empties = _path.join(self.example_path,"two_empty_files.zip")

filenames = Filenames()

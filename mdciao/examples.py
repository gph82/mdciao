#!/home/perezheg/miniconda3/bin/python
from os import path as _path, getcwd as _getcwd
from mdciao import __path__ as mdc_path
from subprocess import run as _run
from tempfile import TemporaryDirectory as _TD
mdc_path = _path.split(mdc_path[0])[0]
cwd = _getcwd()
import subprocess


class ExamplesCLTs(object):
    def __init__(self, test=False):
        from mdciao.filenames import filenames
        filenames = filenames()
        self.xtc = filenames.traj_xtc
        self.pdb = filenames.top_pdb

        #pdb_full = _path.join(examples_path,"gs-b2ar.pdb")
        #xtc_full = _path.join(examples_path,"gs-b2ar.xtc")
        #xtc, pdb = xtc_full, pdb_full

        self.BW_file = filenames.adrb2_human_xlsx
        self.CGN_file = filenames.CGN_3SN6
        self.sitefile = filenames.tip_json

        if not test:
            self.xtc = _path.relpath(self.xtc, cwd)
            self.pdb = _path.relpath(self.pdb, cwd)
            self.BW_file = _path.relpath(self.BW_file, cwd)
            self.CGN_file = _path.relpath(self.CGN_file, cwd)
            self.sitefile = _path.relpath(self.sitefile, cwd)
        self.test = test
    @property
    def mdc_neighborhoods(self):
        return ["mdc_neighborhoods.py",
                "%s %s" % (self.pdb, self.xtc),
                "--residues L394",
                "--ctc_cutoff_Ang 4",
                "--n_smooth_hw 1",
                "--table xlsx",
                "--BW_uniprot %s" % self.BW_file,
                "--CGN_PDB %s" % self.CGN_file,
                ]
    @property
    def mdc_sites(self):
        return ["mdc_sites.py ",
                "%s %s" % (self.pdb, self.xtc),
                " --site_files %s" % self.sitefile,
                " --ctc_cutoff_Ang 4",
                " --BW_uniprot %s" % self.BW_file,
                " --CGN_PDB %s" % self.CGN_file
                ]

    @property
    def mdc_interface(self):
        return ["mdc_interface.py ",
                "%s %s" % (self.pdb, self.xtc),
                " --ctc_cutoff_Ang 4 ",
                " --frag_idxs_group_1 0",
                " --frag_idxs_group_2 3",
                " --BW_uniprot %s" % self.BW_file,
                " --CGN_PDB %s" % self.CGN_file
                ]

    @property
    def mdc_compare_neighborhoods(self):
        pass



    @property
    def clts(self):

        return [attr for attr in dir(self) if attr.startswith("mdc")]

    def _assert_clt_exists(self, clt):
        assert clt in self.clts, "Input method %s is not in existing methods %s" % (clt, self.clts)

    def show(self, clt):
        self._assert_clt_exists(clt)
        print("%s example call:" % clt)
        print("%s--------------"%("".join(["-" for _ in clt]))) # really?
        oneline = self.__getattribute__(clt)
        if self.test:
            oneline = oneline[:-2]
        oneline = " ".join(oneline)
        print(oneline.replace("--", "\n--"))
        print("\n\nYou can re-run '-x' option to execute the command directly\n"
              "or you can paste the line below into your terminal, add/edit options and then execute:\n")
        print(oneline)

    def run(self, clt,show=True, write_to_tmpdir=False):
        if show:
            self.show(clt)
        oneline = self.__getattribute__(clt)
        if self.test:
            oneline = oneline[:-2]
        oneline = " ".join(oneline)
        if write_to_tmpdir:
            with _TD(suffix="mdciao") as tmpdir:
                oneline +=" --output_dir %s"%tmpdir
                _run(oneline.split(),
                     #text=True,
                     #stdin = subprocess.PIPE,
                     #encoding="utf8"
                     )

        else:
            _run(oneline.split(),
        #         text=True,
        #         #shell=True,
        #         stdin = subprocess.PIPE
                 )


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

from os import path as _path, getcwd as _getcwd, chdir as _chdir
from mdciao import __path__ as mdc_path
from subprocess import run as _run
mdc_path = _path.split(mdc_path[0])[0]
cwd = _getcwd()



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
        self.pdb_3SN6 = filenames.pdb_3SN6

        if not test:
            self.xtc = _path.relpath(self.xtc, cwd)
            self.pdb = _path.relpath(self.pdb, cwd)
            self.BW_file = _path.relpath(self.BW_file, cwd)
            self.CGN_file = _path.relpath(self.CGN_file, cwd)
            self.sitefile = _path.relpath(self.sitefile, cwd)
            self.pdb_3SN6 = _path.relpath(filenames.pdb_3SN6,cwd)

        self.test = test
    @property
    def mdc_neighborhoods(self):
        return ["mdc_neighborhoods.py",
                "%s %s" % (self.pdb, self.xtc),
                "--residues L394",
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
                " --BW_uniprot %s" % self.BW_file,
                " --CGN_PDB %s" % self.CGN_file
                ]

    @property
    def mdc_interface(self):
        return ["mdc_interface.py ",
                "%s %s" % (self.pdb, self.xtc),
                " --frag_idxs_group_1 0-2",
                " --frag_idxs_group_2 3",
                " --ctc_control 20",
                " --BW_uniprot %s" % self.BW_file,
                " --CGN_PDB %s" % self.CGN_file,
                ]
    @property
    def mdc_BW_overview(self):
        return ["mdc_BW_overview.py",
                "%s" % self.BW_file,
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
        pass

    @property
    def mdc_fragments(self):
        return ["mdc_fragments.py ",
                "%s" % (self.pdb)
                ]
        pass


    @property
    def clts(self):

        return [attr for attr in dir(self) if attr.startswith("mdc")]

    def _assert_clt_exists(self, clt):
        assert clt in self.clts, "Input method %s is not in existing methods %s" % (clt, self.clts)

    def _join_args(self,clt):
        oneline = self.__getattribute__(clt)
        if self.test:
            oneline = [arg for arg in oneline if "-BW" not in arg and "-CGN" not in arg]
        return " ".join(oneline)

    def show(self, clt):
        self._assert_clt_exists(clt)
        print("%s example call:" % clt)
        print("%s--------------"%("".join(["-" for _ in clt]))) # really?
        oneline = self._join_args(clt)
        print(oneline.replace(" -", " \n-"))
        print("\n\nYou can re-run 'mdc_examples %s.py' with the  '-x' option to execute the command directly\n"
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



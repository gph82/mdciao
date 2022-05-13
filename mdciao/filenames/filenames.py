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

import os.path as _path
from sys import prefix as _env_prefix

#Check https://docs.python.org/3/library/sys.html#sys.prefix
class FileNames(object):
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
        self.RCSB_pdb_path =  _path.join(self.test_data_path, "RCSB_pdb")
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
        self.pdb_3CAP = _path.join(self.RCSB_pdb_path, "3cap.pdb.gz")
        self.pdb_1U19 = _path.join(self.RCSB_pdb_path, "1u19.pdb.gz")
        self.pdb_3SN6 = _path.join(self.RCSB_pdb_path, "3SN6.pdb.gz")
        self.pdb_3E8D = _path.join(self.RCSB_pdb_path, "3E8D.pdb.gz")

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
        self.nomenclature_bib = _path.join(self.nomenclature_path,"nomenclature.bib")
        self.KLIFS_P31751_xlsx = _path.join(self.nomenclature_path, "KLIFS_P31751.xlsx")

        #json
        self.GDP_json = _path.join(self.json_path,"GDP.json")
        self.GDP_name_json = _path.join(self.json_path,"GDP_name_XXX.json")
        self.tip_json = _path.join(self.json_path,"tip.json")
        self.tip_dat= _path.join(self.json_path,"tip.dat")
        self.tip_residx_dat= _path.join(self.json_path,"tip_residx.dat")

        #zip
        self.zipfile_two_empties = _path.join(self.example_path,"two_empty_files.zip")


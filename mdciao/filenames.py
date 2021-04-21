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

from os import path
class filenames(object):
    def __init__(self):
        # Check
        # https://docs.python.org/3.7/tutorial/modules.html#packages-in-multiple-directories
        from mdciao import __path__ as sfpath
        assert len(sfpath) == 1
        sfpath = path.split(sfpath[0])[0]
        self.test_data_path = path.join(sfpath, "tests", "data")
        self.bogus_pdb_path = path.join(self.test_data_path, "bogus_pdb")
        self.RSCB_pdb_path =  path.join(self.test_data_path,"RSCB_pdb" )
        self.example_path =   path.join(self.test_data_path,"examples")
        self.nomenclature_path = path.join(self.test_data_path,"nomenclature")
        self.json_path = path.join(self.test_data_path,"json")

        # pdbs for testing
        self.small_monomer = path.join(self.bogus_pdb_path,
                                       "2_3AA_chains_and_two_ligs_monomer.pdb")
        self.file_for_no_bonds_gro = path.join(self.bogus_pdb_path,
                                       "2_3AA_chains_and_two_ligs_monomer.gro")
        self.small_dimer = path.join(self.bogus_pdb_path,
                                     "2_3AA_chains_and_two_ligs_dimer.pdb")
        # Force a break in the resSeq
        self.small_monomer_LYS99 = path.join(self.bogus_pdb_path,
                                             "2_3AA_chains_and_two_ligs_monomer.LYS29toLYS99.pdb")

        self.actor_pdb = path.join(self.example_path,"prot1.pdb.gz")

        # Pure-PDBs
        self.pdb_3CAP = path.join(self.RSCB_pdb_path,"3cap.pdb.gz")
        self.pdb_1U19 = path.join(self.RSCB_pdb_path,"1u19.pdb.gz")
        self.pdb_3SN6 = path.join(self.RSCB_pdb_path,"3SN6.pdb.gz")

        # Traj
        self.top_pdb = path.join(self.example_path,"gs-b2ar.noH.pdb")
        # TODO/NOTE the time-array of the stride_20 xtc does not start at zero,
        # this helps debug things with time
        # the unstrided xtc, used in examples and doc, DOES start at zero for clarity
        self.traj_xtc_stride_20 = path.join(self.example_path, "gs-b2ar.noH.stride.20.xtc")
        self.traj_xtc = path.join(self.example_path, "gs-b2ar.noH.stride.5.xtc")

        # nomenclature
        self.CGN_3SN6 = path.join(self.nomenclature_path,"CGN_3SN6.txt")
        self.GPCRmd_B2AR_nomenclature_test_xlsx = path.join(self.nomenclature_path,"GPCRmd_B2AR_nomenclature_test.xlsx")
        self.pdb_3SN6_mut = path.join(self.nomenclature_path, "3SN6_GLU10GLX.pdb.gz")
        self.adrb2_human_xlsx = path.join(self.nomenclature_path,"adrb2_human.xlsx")

        #json
        self.GDP_json = path.join(self.json_path,"GDP.json")
        self.GDP_name_json = path.join(self.json_path,"GDP_name_XXX.json")
        self.tip_json = path.join(self.json_path,"tip.json")
        self.tip_dat= path.join(self.json_path,"tip.dat")
        self.tip_residx_dat= path.join(self.json_path,"tip_residx.dat")




if __name__ == '__main__':
    pass



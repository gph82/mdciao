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

        # pdbs for testing
        self.small_monomer = path.join(self.bogus_pdb_path,
                                       "2_3AA_chains_and_two_ligs_monomer.pdb")
        self.file_for_no_bonds_pdb = path.join(self.bogus_pdb_path,
                                       "2_3AA_chains_and_two_ligs_monomer.gro")
        self.small_dimer = path.join(self.bogus_pdb_path,
                                     "2_3AA_chains_and_two_ligs_dimer.pdb")
        self.small_monomer_LYS99 = path.join(self.bogus_pdb_path,
                                             "2_3AA_chains_and_two_ligs_monomer.LYS29toLYS99.pdb")

        self.actor_pdb = path.join(self.example_path,"prot1.pdb.gz")

        self.pdb_3CAP = path.join(self.RSCB_pdb_path,"3cap.pdb.gz")
        self.pdb_3SN6 = path.join(self.RSCB_pdb_path,"3SN6.pdb.gz")

        # Traj
        self.top_pdb = path.join(self.example_path,"p2.noH.pdb")
        self.traj_xtc = path.join(self.example_path,"run1.1-p.stride.5.noH.xtc")

        # nomenclature
        self.CGN_3SN6 = path.join(self.nomenclature_path,"CGN_3SN6.txt")

        """
        self.file_for_top2consensus_map = path.join(self.test_data_path,
                                                    "file_for_top2consensus_map.pdb")

        self.prot1_pdb = path.join(self.examples_path,"prot1.pdb.gz")
        self.run1_stride_100_xtc = path.join(self.examples_path,"run1_stride_200.xtc")

        self.GPCRmd_B2AR_nomenclature_test_xlsx = path.join(self.test_data_path,"GPCRmd_B2AR_nomenclature_test.xlsx")

        self.GDP_json = path.join(self.test_data_path,"GDP.json")
        self.GDP_name_json = path.join(self.test_data_path,"GDP_name_XXX.json")

        self.index_file = path.join(self.test_data_path,"index.ndx")

        self.pdb_1U19 = path.join(self.test_data_path,"1u19.pdb.gz")
        self.pdb_5D5A = path.join(self.test_data_path,"5d5a.pdb.gz")
        """

if __name__ == '__main__':
    pass
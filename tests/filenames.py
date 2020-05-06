from os import path
class filenames(object):
    def __init__(self):
        # Check
        # https://docs.python.org/3.7/tutorial/modules.html#packages-in-multiple-directories
        from mdciao import __path__ as sfpath
        assert len(sfpath) == 1
        sfpath = path.split(sfpath[0])[0]
        self.examples_path = path.join(sfpath, 'examples')
        test_data_path = path.join(sfpath,"tests","data")
        self.test_data_path = test_data_path

        self.file_for_test_pdb = path.join(test_data_path,
                                           "file_for_test.pdb")
        self.file_for_test_repeated_fullresnames_pdb = path.join(test_data_path,
                                                                 "file_for_test_repeated_fullresnames.pdb")

        self.file_for_test_force_resSeq_breaks_is_true_pdb = path.join(test_data_path,
                                                                       "file_for_test_force_resSeq_breaks_is_true.pdb")

        self.file_for_no_bonds_pdb = path.join(test_data_path, "file_no_bonds.pdb")
        self.file_for_top2consensus_map = path.join(test_data_path,
                                                    "file_for_top2consensus_map.pdb")

        self.prot1_pdb = path.join(self.examples_path,"prot1.pdb.gz")
        self.run1_stride_100_xtc = path.join(self.examples_path,"run1_stride_200.xtc")

        self.GPCRmd_B2AR_nomenclature_test_xlsx = path.join(test_data_path,"GPCRmd_B2AR_nomenclature_test.xlsx")

        self.GDP_json = path.join(test_data_path,"GDP.json")
        self.GDP_name_json = path.join(test_data_path,"GDP_name_XXX.json")

        self.index_file = path.join(test_data_path,"index.ndx")

        self.pdb_3CAP = path.join(test_data_path,"3cap.pdb.gz")
        self.pdb_1U19 = path.join(test_data_path,"1u19.pdb.gz")
        self.pdb_5D5A = path.join(test_data_path,"5d5a.pdb.gz")


test_filenames = filenames()
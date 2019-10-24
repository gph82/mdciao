from os import path
class filenames(object):
    def __init__(self):
        # Check
        # https://docs.python.org/3.7/tutorial/modules.html#packages-in-multiple-directories
        from sofi_functions import __path__ as sfpath
        assert len(sfpath) == 1
        sfpath = path.split(sfpath[0])[0]
        self.examples_path = path.join(sfpath, 'examples')
        test_data_path = path.join(sfpath,"tests","data")

        self.file_for_test_pdb = path.join(test_data_path,
                                           "file_for_test.pdb")
        self.file_for_test_repeated_fullresnames_pdb = path.join(test_data_path,
                                                                 "file_for_test_repeated_fullresnames.pdb")

        self.file_for_test_force_resSeq_breaks_is_true_pdb = path.join(test_data_path,
                                                                       "file_for_test_force_resSeq_breaks_is_true.pdb")


        self.file_for_top2consensus_map = path.join(test_data_path,
                                                    "file_for_top2consensus_map.pdb")

        self.prot1_pdb = path.join(self.examples_path,"prot1.pdb.gz")
        self.run1_stride_100_xtc = path.join(self.examples_path,"run1_stride_200.xtc")

        self.GPCRmd_B2AR_nomenclature_test_xlsx = path.join(test_data_path,"GPCRmd_B2AR_nomenclature_test.xlsx")

test_filenames = filenames()
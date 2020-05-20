#!/home/perezheg/miniconda3/bin/python
from os import path, getcwd
from mdciao import __path__ as mdc_path
from subprocess import run
mdc_path = path.split(mdc_path[0])[0]
cwd = getcwd()

class ExamplesCLTs(object):
    def __init__(self, test=False):

        examples_path = path.join(mdc_path, 'examples')
        test_data_path = path.join(mdc_path, "tests", "data")
        xtc = path.join(examples_path, "run1.1-p.stride.5.noH.xtc")
        pdb = path.join(examples_path, "p2.noH.pdb")

        #pdb_full = path.join(examples_path,"gs-b2ar.pdb")
        #xtc_full = path.join(examples_path,"gs-b2ar.xtc")
        #xtc, pdb = xtc_full, pdb_full

        BW_file = path.join(test_data_path, "adrb2_human_full.xlsx")
        CGN_file = path.join(test_data_path, "CGN_3SN6.txt")
        sitefile = path.join(examples_path, "site_201.json")
        if not test:
            xtc = path.relpath(xtc, cwd)
            pdb = path.relpath(pdb, cwd)
            BW_file = path.relpath(BW_file, cwd)
            CGN_file = path.relpath(CGN_file, cwd)
            sitefile = path.relpath(sitefile, cwd)

        self._cmd_dict = {
            "mdc_neighborhood": ["mdc_neighborhoods.py",
                                "%s %s" % (pdb, xtc),
                                "--resSeq_idxs 394",
                                "--ctc_cutoff_Ang 4",
                                "--BW_uniprot %s" % BW_file,
                                "--CGN_PDB %s" % CGN_file,
                                "--n_smooth_hw 1",
                                "--table xlsx"
                ],

            "mdc_interface": ["mdc_interface.py ",
                             "%s %s" % (pdb, xtc),
                             " --fragments resSeq+",
                             " --ctc_cutoff_Ang 4 ",
                             " --frag_idxs_group_1 0",
                             " --frag_idxs_group_2 3",
                             " --BW_uniprot %s" % BW_file,
                             " --CGN_PDB %s" % CGN_file
                              ],

            "mdc_sites": ["mdc_sites.py ",
                         "%s %s" % (pdb, xtc),
                         " --site_files %s" % sitefile,
                         " --ctc_cutoff_Ang 4",
                         " --BW_uniprot %s" % BW_file,
                         " --CGN_PDB %s" % CGN_file
            ]
        }

    @property
    def clts(self):
        return list(self._cmd_dict.keys())

    def _assert_clt_exists(self, clt):
        assert clt in self.clts, "Input method %s is not in existing methods %s" % (clt, self.clts)

    def show(self, clt):
        self._assert_clt_exists(clt)
        print("%s example call:" % clt)
        oneline = " ".join(self._cmd_dict[clt])
        print(oneline.replace("--", "\n--"))
        print(oneline)

    def run(self, clt,show=True):
        if show:
            self.show(clt)
        oneline = " ".join(self._cmd_dict[clt])
        run(oneline.split())

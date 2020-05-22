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
        self.xtc = path.join(examples_path, "run1.1-p.stride.5.noH.xtc")
        self.pdb = path.join(examples_path, "p2.noH.pdb")

        #pdb_full = path.join(examples_path,"gs-b2ar.pdb")
        #xtc_full = path.join(examples_path,"gs-b2ar.xtc")
        #xtc, pdb = xtc_full, pdb_full

        self.BW_file = path.join(test_data_path, "adrb2_human_full.xlsx")
        self.CGN_file = path.join(test_data_path, "CGN_3SN6.txt")
        self.sitefile = path.join(examples_path, "site_201.json")
        if not test:
            self.xtc = path.relpath(self.xtc, cwd)
            self.pdb = path.relpath(self.pdb, cwd)
            self.BW_file = path.relpath(self.BW_file, cwd)
            self.CGN_file = path.relpath(self.CGN_file, cwd)
            self.sitefile = path.relpath(self.sitefile, cwd)

    @property
    def mdc_neighborhood(self):
        return ["mdc_neighborhoods.py",
                "%s %s" % (self.pdb, self.xtc),
                "--resSeq_idxs 394",
                "--ctc_cutoff_Ang 4",
                "--BW_uniprot %s" % self.BW_file,
                "--CGN_PDB %s" % self.CGN_file,
                "--n_smooth_hw 1",
                "--table xlsx"
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
                " --fragments resSeq+",
                " --ctc_cutoff_Ang 4 ",
                " --frag_idxs_group_1 0",
                " --frag_idxs_group_2 3",
                " --BW_uniprot %s" % self.BW_file,
                " --CGN_PDB %s" % self.CGN_file
                ],

    @property
    def clts(self):

        return [attr for attr in dir(self) if attr.startswith("mdc")]

    def _assert_clt_exists(self, clt):
        assert clt in self.clts, "Input method %s is not in existing methods %s" % (clt, self.clts)

    def show(self, clt):
        self._assert_clt_exists(clt)
        print("%s example call:" % clt)
        oneline = " ".join(self.__getattribute__(clt))
        print(oneline.replace("--", "\n--"))
        print(oneline)

    def run(self, clt,show=True):
        if show:
            self.show(clt)
        oneline = " ".join(self.__getattribute__(clt))
        run(oneline.split())
from setuptools import setup, find_packages

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

test_deps = [
    'coverage',
    'pytest',
    'nose',
    "mock"
]

doc_deps = [
    "sphinx",
    "sphinx-argparse",
    "sphinx-rtd-theme",
    "npsphinx",
    "sphinx-copybutton"
]

setup(
    name="mdciao",
    version="0.0.2.beta2",
    author_email="guillermo.perez@charite.de",
    description="mdciao: Analysis of Molecular Dynamics Simulations Using Residue Neighborhoods",
    url="https://github.com/gph82/mdciao",
    project_urls={
        "docs": "http://proteinformatics.org/mdciao",
    },
    python_requires=">=3.6",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    classifiers = ["Development Status :: 4 - Beta",
                   "Programming Language :: Python :: 3",
                   "Operating System :: OS Independent",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
                   "Topic :: Scientific/Engineering :: Chemistry",
                   "Topic :: Scientific/Engineering :: Bio-Informatics",
                   "Topic :: Scientific/Engineering :: Visualization",
                   ],
    packages=find_packages(),
    install_requires=[
                    "cython",
                    "numpy>=1.18.1",
                    "mdtraj",
                    "astunparse; python_version!='3.8'",
                    "astunparse<1.6.3; python_version=='3.8'",
                    "pandas",
                    "matplotlib",
                    "msmtools",
                    "scipy",
                    "joblib",
                    "openpyxl",
                    "biopython",
                    "ipython",
                    "XlsxWriter",
                    "requests",
                    "tqdm",
                    "natsort",
                    "bezier; python_version!='3.6'",
                    "bezier<2020.2.3; python_version=='3.6'"
                     ]
                     +test_deps
                     #+doc_deps
    # tests_require=test_deps, # has been deprecated, gotta learn how to use tox
    ,
    data_files=[
        ('data_for_mdciao/examples', ['tests/data/examples/gs-b2ar.noH.stride.5.xtc',
                                      'tests/data/examples/gs-b2ar.noH.pdb']),
        ('data_for_mdciao/nomenclature', ['tests/data/nomenclature/adrb2_human.xlsx',
                                          'tests/data/nomenclature/CGN_3SN6.txt']),
        ('data_for_mdciao/json', ['tests/data/json/tip.json']),
        ('data_for_mdciao/RSCB_pdb', ['tests/data/RSCB_pdb/3SN6.pdb.gz']),
    ],
    scripts=['scripts/mdc_neighborhoods.py',
             'scripts/mdc_sites.py',
             'scripts/mdc_interface.py',
             'scripts/mdc_fragments.py',
             'scripts/mdc_BW_overview.py',
             'scripts/mdc_CGN_overview.py',
             'scripts/mdc_compare.py',
             'scripts/mdc_examples.py',
             'scripts/mdc_pdb.py',
             'scripts/mdc_residues.py',
             #'scripts/density_by_sites.py',
             #'scripts/site_figures.py',
             #'scripts/residue_dihedrals.py',
             #'scripts/contact_maps.py'
             #'scripts/CARDS.py'
             ],
)


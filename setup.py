from setuptools import setup, find_packages

test_deps = [
    'coverage',
    'pytest',
    'nose',
    "mock"
]

doc_deps = [
    "sphinx",
    "sphinx-argparse",
    "sphinx-rtd-theme"
]

setup(
    name="mdciao",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
                    "cython",
                    "numpy",
                    "mdtraj",
                    "pandas",
                    "matplotlib",
                    "msmtools",
                    "scipy",
                    "joblib",
                    "xlrd",
                    "biopython",
                    "ipython",
                    "XlsxWriter",
                    "requests"
                    ]
                     +test_deps
                     +doc_deps
    # tests_require=test_deps, # has been deprecated, gotta learn how to use tox
    ,
    scripts=['scripts/residue_neighborhoods.py',
             #'scripts/compare_groups_of_contacts.py',
             'scripts/sites.py',
             #'scripts/site_figures.py',
             'scripts/interface_ctc_analyzer.py',
             #'scripts/density_by_sites.py',
             'scripts/fragment_overview.py',
             'scripts/BW_overview.py',
             'scripts/CGN_overview.py',
             #'scripts/residue_dihedrals.py',
             #'scripts/contact_maps.py'
             #'scripts/CARDS.py'
             ],
)


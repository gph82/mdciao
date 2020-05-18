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
                     #+doc_deps
    # tests_require=test_deps, # has been deprecated, gotta learn how to use tox
    ,
    scripts=['scripts/mdc_neighborhoods.py',
             'scripts/mdc_sites.py',
             'scripts/mdc_interface.py',
             'scripts/mdc_fragment_overview.py',
             'scripts/mdc_BW_overview.py',
             'scripts/mdc_CGN_overview.py',
             #'scripts/density_by_sites.py',
             #'scripts/site_figures.py',
             #'scripts/residue_dihedrals.py',
             #'scripts/contact_maps.py'
             #'scripts/CARDS.py'
             ],
)


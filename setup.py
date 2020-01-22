from setuptools import setup, find_packages
setup(
    name="mdciao",
    version="0.1",
    packages=find_packages(),
    install_requires=[
    "pandas",
    "matplotlib",
    "mdtraj",
    "numpy",
    "msmtools",
    "scipy",
    "joblib",
        "xlrd",
    "biopython",
    "ipython",
    "XlsxWriter", # 
    ],
    tests_require=[
    "mock"
    ],
    scripts=['scripts/residue_neighborhoods.py',
             'scripts/sites.py',
             'scripts/site_figures.py',
             'scripts/interface_ctc_analyzer.py',
             'scripts/density_by_sites.py',
             'scripts/fragment_overview.py',
             'scripts/BW_overview.py',
             #'scripts/CARDS.py'
             ]
)


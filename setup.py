from setuptools import setup, find_packages
setup(
    name="sofi_functions",
    version="0.1",
    #packages="sofi_functions", # find_packages() is not really needed for now
    packages=find_packages(),
    install_requires=[
    "pandas",
    "matplotlib",
    "mdtraj",
    "numpy",
    "msmtools",
    "scipy",
    "Bio",
    "ipython", # These two are for one call to Gunnar_utils
    "XlsxWriter", # 
    ],
    tests_require=[
    "mock"
    ],
    scripts=['scripts/residue_neighborhoods.py']
)


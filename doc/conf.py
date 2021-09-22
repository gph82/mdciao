# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'mdciao'
copyright = '2019-2021, Guillermo Perez-Hernandez'
author = 'Guillermo Perez-Hernandez'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinxarg.ext",
    "sphinx.ext.intersphinx",
    'sphinx_copybutton',
    'nbsphinx',
#    "numpydoc",
]

# Napoleon Settings for showing class documentation's init doc
#napoleon_include_init_with_doc = True

intersphinx_mapping = {'mdtraj': ('http://mdtraj.org/1.9.4/', None),
                       'matplotlib': ('https://matplotlib.org/',None),
                       'pandas': ('https://pandas.pydata.org/docs/',None),
                       'requests': ('https://requests.readthedocs.io/en/master/',None)}

autodoc_mock_imports = ["mdtraj", 
#                        "matplotlib"
]

autosummary_generate = True
autodoc_default_flags = ['members', 'inherited-members']
#numpydoc_class_members_toctree = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Number figures
numfig = True
numfig_format = {'figure': 'Fig. %s'}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
#html_theme = 'nature'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

html_sidebars = { '**': ['localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

latex_elements = {
  "papersize":"a4paper",
    'extraclassoptions': 'openany,oneside'

}

html_last_updated_fmt=""
try:
    from importlib import metadata
    version = metadata.version("mdciao")
except ImportError:
    import pkg_resources
    version = pkg_resources.get_distribution("mdciao").version
copybutton_prompt_text = ">>> "

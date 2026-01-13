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
copyright = '2019-2025, Guillermo Perez-Hernandez'
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
    "sphinx.ext.todo",
    "sphinxarg.ext",
    "sphinx.ext.intersphinx",
    'sphinx_copybutton',
    'nbsphinx',
#    "numpydoc",
]

napoleon_use_param = False
# Napoleon Settings for showing class documentation's init doc
#napoleon_include_init_with_doc = True

intersphinx_mapping = {'mdtraj': ('http://mdtraj.org/1.9.4/', None),
                       'matplotlib': ('https://matplotlib.org/',None),
                       'pandas': ('https://pandas.pydata.org/docs/',None),
                       'requests': ('https://requests.readthedocs.io/en/master/',None)}

autodoc_mock_imports = ["mdtraj", 
#                        "matplotlib"
]

autodoc_default_options = {'members' : True,
                           'inherited-members': True}


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
html_logo = 'imgs/mdciao.logo.png'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

html_sidebars = { '**': ['localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

latex_elements = {
  "papersize":"a4paper",
    'extraclassoptions': 'openany,oneside'

}

nbsphinx_execute = 'never'

html_last_updated_fmt=""
try:
    from importlib import metadata
    version = metadata.version("mdciao")
except ImportError:
    import pkg_resources
    version = pkg_resources.get_distribution("mdciao").version
copybutton_prompt_text = ">>> "

def rename_thumbnails(*args):
    """
    # This re-reads the notebooks to extract cell count (code and markup)
    # takes a bit longer, but might come in handy for specifying thumbnails
    # using cell idx rather than figure index
    import json
    with open("02.Missing_Contacts.ipynb", 'r') as f:
        notebook_data = json.load(f)
    notebook_data.keys()
    idxs = {}
    for ii, cc in enumerate(notebook_data["cells"]):
        if cc["cell_type"] == "code":
            idxs[cc["execution_count"]] = ii
    """
    from glob import glob
    from natsort import natsorted
    from sphinx.util import logging
    import shutil
    logger = logging.getLogger(__name__)
    fig_idxs = {"02.Missing_Contacts": 1,
                "03.Comparing_CGs_Bars": -1,
                "04.Comparing_CGs_Flares": -1,
                "05.Flareplot_Schemes": -3,
                "07.EGFR_Kinase_Inhibitors": -1,
                "08.Manuscript": -1,
                "09.Consensus_Labels": 3}
    for nb_basename, fig_idx in fig_idxs.items():
        exp = f"_build/doctrees/nbsphinx/notebooks_{nb_basename}_*_*.png" #doctrees/nbsphinx seems to be created already after "html-page-context"
        cands = [ff for ff in natsorted(glob(exp)) if not ff.endswith("selected_thumbnail.png")]
        #logger.info(f"Picking nr {fig_idx} from available files:"+"\n"+"\n".join(cands))
        if not cands:
            logger.warning(f"No thumbnails found for {nb_basename}")
            continue
        source_name = cands[fig_idx]
        target_name = f"_build/doctrees/nbsphinx/notebooks_{nb_basename}_selected_thumbnail.png"
        #logger.info(f"Will copy {source_name} to {target_name}")
        shutil.copy(source_name, target_name)

def setup(app):
    # Connect to the 'build-finished' event
    app.connect('build-finished', rename_thumbnails) # needs to run twice until I find a new event, perhaps html-page-context




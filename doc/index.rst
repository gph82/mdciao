.. sofi_functions documentation master file, created by
   sphinx-quickstart on Fri Sep  6 11:54:24 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to sofi_functions's documentation!
==========================================

The main goal of this Python library is to provide the command-line
tools to analyze molecular simulation data using residue-residue contacts. It ows
most of its functionality to the molecular dynamics analysis library
`mdtraj <http://mdtraj.org/>`_.

At the moment, these command line tools are:

 * residue_neighborhoods.py (tested and documented)
      Analyzes most frequent interaction partners for any chosen residue via summaries
      of their frequencies and time-traces.
 * sites.py (experimental)
      Group sets of residue-residue contacts into "sites" and present a summary of
      their frequencies and time-trances.


These commandline tools work with methods contained in the modules

 * contacts
 * fragments
 * command_line_tools

which are also exposed by sofi_tools after import into interactive
IPython terminal sessions and JuPyter notebooks.

These submodules depend themselves on more specific methods are packaged into the *_utils files

   * list_utils.py
   * bond_utils.py
   * aa_utils.py
   * nomenclature_utils.py

Finally, there is the submodule actor_utils.py, which is still containing important **but untested**
methods. These methods will gradually be refactored into the any of the above submodules.

.. note::
   **This library is still under heavy development
   It is guaranteed to change in the future.**

.. toctree::
   :maxdepth: 2
   :caption: Modules:

   contacts
   fragments

.. toctree::
   :maxdepth: 2
   :caption: Command Line Tools:

   residue_neighborhoods

.. toctree::
   :maxdepth: 2
   :caption: Submodules:

   aa_utils
   bond_utils
   list_utils
   nomenclature_utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

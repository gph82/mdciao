.. sofi_functions documentation master file, created by
   sphinx-quickstart on Fri Sep  6 11:54:24 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to sofi_functions's documentation!
==========================================

The main goal of this Python library is to provide the command-line
tools to analyze molecular simulation data using residue-residue contacts.

The starting point for these tools are the files typically generated in
the context of molecular dynamics (MD) simulations, i.e.

 * topology files, like prot.gro or prot.pdb
 * trajectory files, like traj1.xtc, traj2.xtc

Most of its functionality to the molecular dynamics analysis library
`mdtraj <http://mdtraj.org/>`_.

At the moment, these command-line tools are:

 * residue_neighborhoods.py (tested and documented)
      Analyzes most frequent interaction partners for any chosen residue via summaries
      of their frequencies and time-traces.
 * sites.py (experimental)
      Group sets of residue-residue contacts into "sites" and present a summary of
      their frequencies and time-trances.


These command-line tools work with methods contained in the submodules:

 * contacts
      For the computation residue-residue contacts and their presentation as time-traces
      or summarized probabilities (=frequencies).
 * fragments
      For the identification and handling (=joining, splitting, naming) of fragments in
      the molecular topology.

 * command_line_tools
      The residue_neighborhoods.py and sites.py are just wrappers
      around the methods contained in the submodule command_line_tools.
      This way, the user can also use the command-line
      tools in any interactive session, e.g. a in
      the IPython terminal and/or the JuPyter notebooks.

The lowest level modules are more specicif and are packed into the *_utils files:

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

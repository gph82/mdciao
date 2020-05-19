.. mdciao documentation master file, created by
   sphinx-quickstart on Fri Sep  6 11:54:24 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mdciao's documentation!
==================================

The main goal of this Python library is to provide quick, "one-shot" command-line
tools to analyze molecular simulation data using residue-residue distances, tyring to automate as much as possible while remaining highly customizable.

The analysis is based on contact-frequencies, i.e. the percentage of simulation time
that two given residues find each other at a distances smaller or equal than a given
cut-off value.

Starting from the files typically generated in
the context of molecular dynamics (MD) simulations, i.e.

* topology files, like prot.gro or prot.pdb
* trajectory files, like traj1.xtc, traj2.xtc

mdciao will calculate contact frequencies, distance time-traces and overall number of interaction partners and produce quasi paper-ready tables and figures. Under the hood, the module `mdtraj <http://mdtraj.org/>`_ is doing most of the computation and handling of molecular information, whereas mdciao provides functionalities like:

* fragment definitions for quickly defining regions of interest
* *automagic* map and incorporate consensus nomenclature like the Ballesteros-Weinstein (BW) or Common-G-Protein (CGN) to the analysis
* site definitions for analysing and comparing equivalent moieties accross different setups
* comparison tools to automatically detect and present frequency differences accross systems, e.g. to look for the effect of mutations, pH-differences etc
* TODO expand

.. note::

 Lastly, a note of caution regarding the above definitions for *contact* and *frequency*:

 * the kinetic information is averaged out. Contacts quickly breaking and forming and contacts that break (or form) only once **will have the same frequency** as long as the **fraction of total time** they are formed is the same. For analysis taking kinetics into account, use. e.g. `pyemma <http://mdtraj.org>`_.
 * The sharp, "distance-only" cutoff can sometimes over- or under-represent some interaction types. Modules like `get_contacts <https://github.com/getcontacts/getcontacts>`_ capture these interactions better.

However, both these issues (if/when they arise) can be spotted easily by looking at the time-traces of said contacts and informed decisions can be made wrt to parameters like the cutt-off value, number of contacts displayed and many others.

Command line tools
==================

At the moment, the command-line tools that the user can invoke directly from the terminal after installing mdciao are

* mdc_neighborhoods
* mdc_interface
* mdc_sites
* mdc_fragment_overview
* mdc_BW_overview
* mdc_CGN_overview
* mdc_compare_neighborhoods

You can see their documentation by using the ``-h`` flag whe invoking them from the command line or by checking these pages.

API
===
mdciao ships not only with the above command line tools, but with a number of submodules (loosely referred to as API from now on). The objects and methods in the API allow the experienced user to create their own scripts or interactive workflows in IPython or even better, IPython JuPyTer notebooks.

These can be imported into the namespace by simply by using ``import mdciao``.

Whereas the command-line-tools from above tend to be more stable, the API functions and object calls might change future. Bugfixes, refactors and redesigns are in the pipeline and experienced users should know how to deal with this.

All API objects and functions are extensively documented, just not linked here (yet). Please use their docstring: double-tab in Jupyter Notebooks, or cmd?+Enter in the IPython terminal.

.. toctree::
   :maxdepth: 1
   :caption: Command Line Tools:

   mdc_neighborhoods
   mdc_sites
   mdc_interface
   mdc_fragment_overview
   mdc_BW_overview
   mdc_CGN_overview
   mdc_compare_neighborhoods

.. toctree::
   :maxdepth: 2
   :caption: Modules:

   contacts
   fragments

.. toctree::
   :maxdepth: 2
   :caption: Submodules:

   aa_utils
   bond_utils
   list_utils
   nomenclature_utils
   sequence_utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

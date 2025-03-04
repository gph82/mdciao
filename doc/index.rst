.. mdciao documentation master file, created by
   sphinx-quickstart on Fri Sep  6 11:54:24 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mdciao: Accessible Analysis and Visualization of Molecular Dynamics Simulation Data
===================================================================================

|Pip Package| |Python Package| |MacOs Package| |Coverage| |DOI| |License|

.. figure:: imgs/banner.png
   :scale: 33%

.. figure:: imgs/distro_and_violin.png
   :scale: 25%

.. figure:: imgs/timedep_ctc_matrix.png
   :scale: 55%

.. _my-reference-label:
.. figure:: imgs/interface.combined.png
   :scale: 33%

``mdciao`` is a Python module that provides quick, "one-shot" command-line tools to analyze molecular simulation data using residue-residue distances. ``mdciao`` tries to automate as much as possible for non-experienced users while remaining highly customizable for advanced users, by exposing an API to construct your own analysis workflow.

Under the hood, the module `mdtraj <https://mdtraj.org/>`_ is doing most of the computation and handling of molecular information, using `BioPython <https://biopython.org/>`_ for sequence alignment, `pandas <https://pandas.pydata.org/>`_ for many table and IO related operations, and `matplotlib <https://matplotlib.org>`_ for visualization. It tries to automatically use the consensus nomenclature for

* GPCRs
    * via `Ballesteros-Weinstein-Numbering <https://www.sciencedirect.com/science/article/pii/S1043947105800497>`_ or structure-based schemes by `Gloriam et al <https://doi.org/10.1016/j.tips.2014.11.001>`_ for the receptor's TM domain, or
    * via generic-residue-numbering for the GAIN domain of `adhesion GPCRs <https://doi.org/10.1038/s41467-024-55466-6>`_
* G-proteins
    * via `Common G-alpha Numbering (CGN) <https://www.mrc-lmb.cam.ac.uk/CGN/faq.html>`_
* Kinases
    * via their `85 pocket-residue numbering scheme <https://doi.org/10.1021/JM400378W>`_

using local files or on-the-fly lookups of the `GPCRdb <https://gpcrdb.org/>`_
and/or `KLIFS <https://klifs.net/>`_.

Basic Principle
---------------

``mdciao``  takes the files typically generated by a molecular dynamics (MD) simulation, i.e.

* topology files, like *prot.gro* or *top.pdb*
* trajectory files, like *traj1.xtc*, *traj2.xtc*

and calculates the  time-traces of residue-residue distances, and from there, **contact frequencies** and **distance distributions**. The most simple command line call would look approximately like this::

 mdc_neighborhoods.py top.pdb traj.xtc --residues L394
 [...]
 The following 6 contacts capture 5.26 (~97%) of the total frequency 5.43 (over 9 contacts with nonzero frequency at 4.50 Angstrom).
 As orientation value, the first 6 ctcs already capture 90.0% of 5.43.
 The 6-th contact has a frequency of 0.52.
    freq          label            residues  fragments   sum
 1  1.00  L394@frag0 - L388@frag0  353 - 347    0 - 0   1.00
 2  1.00  L394@frag0 - R389@frag0  353 - 348    0 - 0   2.00
 3  0.97  L394@frag0 - L230@frag3  353 - 957    0 - 3   2.97
 4  0.97  L394@frag0 - R385@frag0  353 - 344    0 - 0   3.94
 5  0.80  L394@frag0 - I233@frag3  353 - 960    0 - 3   4.74
 6  0.52  L394@frag0 - K270@frag3  353 - 972    0 - 3   5.26
 The following files have been created:
 ./neighborhood.overall@4.5_Ang.pdf
 ./neighborhood.LEU394@frag0@4.5_Ang.dat
 ./neighborhood.LEU394@frag0.time_trace@4.5_Ang.pdf


You can also invoke::

 mdc_examples.py

for a list of all the built-in command-line toy-examples or::

 mdc_notebooks.py

for live Jupyter notebooks play around with. These are shown in the :ref:`Jupyter Notebook Gallery` along with other real-life, more elaborated examples.


.. note::

 A note of caution regarding the above definitions for *contact* and *frequency*:

 * the kinetic information is averaged out. Contacts quickly breaking and forming and contacts that break (or form) only once **will have the same frequency** as long as the **fraction of total time** they are formed is the same. For analysis taking kinetics into account, use. e.g. `pyemma <http://mdtraj.org>`_.
 * The sharp, "distance-only" cutoff can sometimes over- or under-represent some interaction types. Modules like `get_contacts <https://github.com/getcontacts/getcontacts>`_ or `ProLIF <https://prolif.readthedocs.io/en/stable/>`_ and the `PLIP webserver <https://plip-tool.biotec.tu-dresden.de/plip-web/plip/index>`_ have individual geometric definitions for each interaction type.
 * Frequencies are just **averages** over the input data. In some cases, *simply* computing averages is a bad idea. The user is `responsible for deciding over what data to average <https://en.wikipedia.org/wiki/Garbage_in,_garbage_out>`_. For example, if your data is highly heterogenous you might want to `cluster <https://manual.gromacs.org/documentation/2018/onlinehelp/gmx-cluster.html>`_ your data into into ``cluster1.xtc``, ``cluster.2.xtc`` etc and then do a per-cluster analysis with ``mdciao``. Same applies to single frames i.e. PDB files, where the word "frequency" doesn't make any sense.

 These issues (if/when they arise) can be spotted easily by looking at the time-traces and informed decisions can be made wrt to parameters like the cutt-off value, number of contacts displayed and many others.

.. |Pip Package| image::
   https://badge.fury.io/py/mdciao.svg
   :target: https://badge.fury.io/py/mdciao

.. |Python Package| image::
   https://github.com/gph82/mdciao/actions/workflows/python-package.yml/badge.svg
   :target: https://github.com/gph82/mdciao/actions/workflows/python-package.yml

.. |MacOs Package| image::
   https://github.com/gph82/mdciao/actions/workflows/python-package.macos.yml/badge.svg
   :target: https://github.com/gph82/mdciao/actions/workflows/python-package.macos.yml

.. |Coverage| image::
   https://codecov.io/gh/gph82/mdciao/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/gph82/mdciao

.. |License| image::
    https://img.shields.io/github/license/gph82/mdciao

.. |DOI| image::
   https://zenodo.org/badge/DOI/10.5281/zenodo.5643177.svg
   :target: https://doi.org/10.5281/zenodo.5643177

.. there's this issue about the self-referencing TOC that I cannot solve
.. https://github.com/sphinx-doc/sphinx/issues/4602

.. toctree::
   :hidden:

   installation
   CLI Tutorial <overview>
   API Jupyter Notebook Tutorial <notebooks/Tutorial.ipynb>
   notebooks/Covid-19-Spike-Protein-Example.ipynb
   notebooks/Covid-19-Spike-Protein-Interface.ipynb
   cli_cli/cli_cli
   api/api
   gallery


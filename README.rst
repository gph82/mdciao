mdciao: Analysis of Molecular Dynamics Simulations Using Residue Neighborhoods
==============================================================================

|Python Package| |Coverage|

.. figure:: doc/imgs/banner.png
   :scale: 33%

.. figure:: doc/imgs/interface.combined.png
   :scale: 33%

``mdciao`` is a Python module that provides quick, "one-shot" command-line tools to analyze molecular simulation data using residue-residue distances. ``mdciao`` tries to automate as much as possible for non-experienced users while remaining highly customizable for advanced users, by exposing an API to construct your own analysis workflow.

Under the hood, the module `mdtraj <https://mdtraj.org/>`_ is doing most of the computation and handling of molecular information, using `BioPython <https://biopython.org/>`_ for sequence alignment, `pandas <pandas.pydata.org/>`_ for many table and IO related operations, and `matplotlib <https://matplotlib.org.org>`_ for visualizaton. It tries to automatically use the

* `Ballesteros-Weinstein-Numbering (BW) <https://www.sciencedirect.com/science/article/pii/S1043947105800497>`_
* `Common G-alpha Numbering (CGN) <https://www.mrc-lmb.cam.ac.uk/CGN/faq.html>`_

consensus-nomenclature schemes by either using local files or on-the-fly lookups of the `GPRC.db <https://gpcrdb.org/>`_
and/or `<https://www.mrc-lmb.cam.ac.uk/CGN/>`_

Licenses
========
* ``mdciao`` is licensed under the `GNU Lesser General Public License v3.0 or later <https://www.gnu.org/licenses/lgpl-3.0-standalone.html>`_ (``LGPL-3.0-or-later``, see the LICENSE.txt).

* ``mdciao`` uses a modified version of the method `mdtraj.compute_contacts <https://github.com/mdtraj/mdtraj/blob/70a94ff87a6c4223ca1be78c752ef3ef452d3d44/mdtraj/geometry/contact.py#L42>`_  of `mdtraj <https://mdtraj.org/>`_. This modified version is published along with ``mdciao`` and can be found in `contacts/_md_compute_contacts.py <mdciao/contacts/_md_compute_contacts.py>`_. Please see that file for details on the modifications.

* Modules used by ``mdciao`` have different licenses. You can check any module's license in your Python environment using `pip-licenses <https://github.com/raimon49/pip-licenses>`_:
>>> pip-licenses | grep module_name

Status
======
``mdciao`` is in its initial development, with versions 0.Y.Z. Anything MAY change at any time.
`The public API SHOULD NOT be considered stable <https://semver.org/#spec-item-4>`_.

TODOs
=====
This is an informal list of known issues and TODOs:
 * overhaul the "printing" system with proper warnings
 * progressbar not very informative for one chunked trajs or parallel runs
 * improve sequence alignment choices
 * heuristics for proper font-sizing of flareplots could be optimized
 * parallel execution with memory mdtraj.Trajectory objects should be better

System Requirements
===================
At the moment, ``mdciao`` is CI-tested only for GNU/Linux and MacOS and Python versions
3.6 and 3.7

Authors
=======
``mdciao`` is written and maintained by Guillermo Pérez-Hernández (`ORCID <http://orcid.org/0000-0002-9287-8704>`_) currently at the `Institute of Medical Physics and Biophysics <https://biophysik.charite.de/ueber_das_institut/team/>`_ in the
`Charité Universitäsmedizin Berlin <https://www.charite.de/>`_.

Please cite "mdciao, G. Pérez-Hernández and P.W. Hildebrand, 2020 (in preparation)"


.. |Python Package| image::
   https://github.com/gph82/mdciao/workflows/Python%20package/badge.svg
   :target: https://github.com/gph82/mdciao/actions?query=workflow%3A%22Python+package%22

.. |Coverage| image::
   https://codecov.io/gh/gph82/mdciao/branch/master/graph/badge.svg?
   :target: https://codecov.io/gh/gph82/mdciao

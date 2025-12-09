mdciao: Accessible Analysis and Visualization of Molecular Dynamics Simulation Data
===================================================================================

|Pip Package| |Python Package| |MacOs Package| |Coverage| |DOI| |License|

.. figure:: doc/imgs/banner.png
   :scale: 33%

.. figure:: doc/imgs/distro_and_violin.png
   :scale: 25%

.. figure:: doc/imgs/timedep_ctc_matrix.png
   :scale: 55%

.. figure:: doc/imgs/interface.combined.png
   :scale: 33%

``mdciao`` is a Python module that provides quick, "one-shot" command-line tools to analyze molecular simulation data using residue-residue distances. ``mdciao`` tries to automate as much as possible for non-experienced users while remaining highly customizable for advanced users, by exposing an API to construct your own analysis workflow.

Under the hood, the module `mdtraj <https://mdtraj.org/>`_ is doing most of the computation and handling of molecular information, using `BioPython <https://biopython.org/>`_ for sequence alignment, `pandas <pandas.pydata.org/>`_ for many table and IO related operations, and `matplotlib <https://matplotlib.org>`_ for visualization. It tries to automatically use the consensus nomenclature for

* GPCRs
    * via `Ballesteros-Weinstein-Numbering <https://www.sciencedirect.com/science/article/pii/S1043947105800497>`_ or structure-based schemes by `Gloriam et al <https://doi.org/10.1016/j.tips.2014.11.001>`_ for the receptor's TM domain, or
    * via generic-residue-numbering for the GAIN domain of `adhesion GPCRs <https://doi.org/10.1038/s41467-024-55466-6>`_
* G-proteins
    * via `Common G-alpha Numbering (CGN) <https://www.mrc-lmb.cam.ac.uk/CGN/faq.html>`_
* Kinases
    * via their `85 pocket-residue numbering scheme <https://doi.org/10.1021/JM400378W>`_

using local files or on-the-fly lookups of the `GPCRdb <https://gpcrdb.org/>`_
and/or `KLIFS <https://klifs.net/>`_.

Licenses
========
* ``mdciao`` is licensed under the `GNU Lesser General Public License v3.0 or later <https://www.gnu.org/licenses/lgpl-3.0-standalone.html>`_ (``LGPL-3.0-or-later``, see the LICENSE.txt).

* ``mdciao`` uses a modified version of the method `mdtraj.compute_contacts <https://github.com/mdtraj/mdtraj/blob/70a94ff87a6c4223ca1be78c752ef3ef452d3d44/mdtraj/geometry/contact.py#L42>`_  of `mdtraj <https://mdtraj.org/>`__. This modified version is published along with ``mdciao`` and can be found in `contacts/_md_compute_contacts.py <mdciao/contacts/_md_compute_contacts.py>`_. Please see that file for details on the modifications.

* Modules used by ``mdciao`` have different licenses. You can check any module's license in your Python environment using `pip-licenses <https://github.com/raimon49/pip-licenses>`_:

  >>> pip-licenses | grep module_name

Documentation
=============
Currently, docs are hosted at `<https://mdciao.org>`_.

System Requirements
===================
``mdciao`` is developed in GNU/Linux, and CI-tested via `github actions <https://github.com/gph82/mdciao/actions>`_ for GNU/Linux and MacOs. Tested Python versions are:

* GNU/Linux: 3.8, 3.9, 3.10, 3.11, 3.12
* MacOs: 3.8, 3.9, 3.10, 3.11, 3.12.

So everything should work *out of the box* in these conditions.

.. admonition:: Python 3.13 users

   Python 3.13 support is unofficial, because the module ``bezier`` `currently requires python <=3.12 <https://github.com/dhermes/bezier>`_.

   Still, you can install mdciao in Python 3.13 if you install ``bezier`` previously with these environment variables:

   >>> BEZIER_NO_EXTENSION="True" BEZIER_IGNORE_VERSION_CHECK="True" pip install bezier
   >>> pip install mdciao

   You can check what these variables do `here <https://bezier.readthedocs.io/en/stable/development.html#environment-variables>`__.

   Since ``mdciao`` installs and passes the CI-tests for Python 3.13 in such an environment, you can use it **at your own risk**. Please report on any issues you might find.

Authors
=======
``mdciao`` is written and maintained by Guillermo Pérez-Hernández (`ORCID <http://orcid.org/0000-0002-9287-8704>`_) currently at the `Institute of Medical Physics and Biophysics <https://biophysik.charite.de/ueber_das_institut/team/>`_ in the
`Charité Universitäsmedizin Berlin <https://www.charite.de/>`_.

Please cite:
 * mdciao: Accessible Analysis and Visualization of Molecular Dynamics Simulation Data
    | Guillermo Pérez-Hernández, Peter W. Hildebrand
    | PLoS Comput Biol 21(4): e1012837.
    | https://doi.org/10.1371/journal.pcbi.1012837

Scope
======
``mdciao`` originated as a loose collection of CLI scripts used in our lab to streamline contact-frequency analysis of MD simulations with `mdtraj <https://mdtraj.org/>`__,
which is doing a lot of the heavy work under the hood of ``mdciao``. The goal was to take the less scripting-affine
lab members from their raw data to informative graphs about the general vicinity of *their* residues
of interest without much hassle. From there, it grew to incorporate many of the things routinely done in the lab
(with a focus on GPCRs and G proteins) and ultimately a package available for third-party use was made.

The main publications which have driven the development of ``mdciao`` are:
 * Function and dynamics of the intrinsically disordered carboxyl terminus of β2 adrenergic receptor.
    | Heng, J., Hu, Y., Pérez-Hernández, G. et al.
    | Nat Commun 14, 2005 (2023).
    | https://doi.org/10.1038/s41467-023-37233-1
 * Time-resolved cryo-EM of G-protein activation by a GPCR.
    | Papasergi-Scott, M.M., Pérez-Hernández, G., Batebi, H. et al.
    | Nature 629, 1182–1191 (2024).
    | https://doi.org/10.1038/s41586-024-07153-1
 * Mechanistic insights into G-protein coupling with an agonist-bound G-protein-coupled receptor.
    | Batebi, H., Pérez-Hernández, G., Rahman, S.N. et al.
    | Nat Struct Mol Biol (2024).
    | https://doi.org/10.1038/s41594-024-01334-2
 * Generic residue numbering of the GAIN domain of adhesion GPCRs.
    | Seufert, F., Pérez-Hernández, G., Pándy-Szekeres, G. et al.
    | Nat Commun 16, 246 (2025).
    | https://doi.org/10.1038/s41467-024-55466-6

TODOs
=====
You can find an informal list of TODOs and known issues `here <https://github.com/gph82/mdciao/blob/master/doc/TODOs.rst>`__.


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
   https://codecov.io/gh/gph82/mdciao/branch/master/graph/badge.svg?
   :target: https://codecov.io/gh/gph82/mdciao

.. |License| image::
    https://img.shields.io/github/license/gph82/mdciao

.. |DOI| image::
   https://zenodo.org/badge/DOI/10.5281/zenodo.5643177.svg
   :target: https://doi.org/10.5281/zenodo.5643177



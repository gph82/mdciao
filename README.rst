
.. figure:: imgs/banner.png
   :scale: 33%

Welcome to mdciao's documentation!
==================================

``mdciao`` is a Python module that provides quick, "one-shot" command-line tools to analyze molecular simulation data using residue-residue distances. ``mdciao`` tries to automate as much as possible for non-experienced users while remaining highly customizable for advanced users, by exposing an :ref:`API` to construct your own analysis workflow. Here you can find our :ref:`minimal_example`.

Under the hood, the module `mdtraj <https://mdtraj.org/>`_ is doing most of the computation and handling of molecular information, using `BioPython <https://biopython.org/>`_ for sequence alignment, `pandas <pandas.pydata.org/>`_ for many table and IO related operations, and `matplotlib <https://matplotlib.org.org>`_ for visualizaton.
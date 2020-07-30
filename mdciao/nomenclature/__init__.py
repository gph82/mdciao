r"""
Get and manipulate consensus nomenclature:
 * `Ballesteros-Weinstein-Numbering <https://www.sciencedirect.com/science/article/pii/S1043947105800497>`_ (BW)
 * `Common G-alpha Numbering (CGN) <https://www.mrc-lmb.cam.ac.uk/CGN/faq.html>`_

It uses either local files or accesses the `GPRC.db <https://gpcrdb.org/>`_ and/or `<https://www.mrc-lmb.cam.ac.uk/CGN/>`_

.. currentmodule:: mdciao.nomenclature

Functions
=========

.. autosummary::
    :nosignatures:
    :toctree: generated/


   guess_nomenclature_fragments
   PDB_finder
   BW_finder
   md_load_rscb


Classes
=======

.. autosummary::
    :toctree: generated

    LabelerConsensus
    LabelerBW
    LabelerCGN

"""
from .nomenclature import *
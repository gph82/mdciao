r"""
Get and manipulate consensus nomenclature:

 * `Ballesteros-Weinstein-Numbering (BW) <https://doi.org/10.1016/S1043-9471(05)80049-7>`_ [1]


 * `Common G-alpha Numbering (CGN) <https://doi.org/10.1038/nature14663>`_ [2]


It uses either local files or accesses the following databases:

 * `GPCRdb <https://gpcrdb.org/>`_ [3]

 * `<https://www.mrc-lmb.cam.ac.uk/CGN/>`_ [2]

 * `http://www.rcsb.org/ <https://files.rcsb.org/download>`_ [4]

Please see the `references below`_.


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

.. _`references below`:

References
==========

 * [1] Juan A. Ballesteros, Harel Weinstein,
   *[19] Integrated methods for the construction of three-dimensional models and computational probing
   of structure-function relations in G protein-coupled receptors*,
   Editor(s): Stuart C. Sealfon, Methods in Neurosciences, Academic Press, Volume 25, 1995
   `<https://doi.org/10.1016/S1043-9471(05)80049-7>`_

 * [2] Flock, T., Ravarani, C., Sun, D. et al.,
   *Universal allosteric mechanism for Gα activation by GPCRs.*
   Nature 524, 173–179 (2015)
   `<https://doi.org/10.1038/nature14663>`_

 * [3] Gáspár Pándy-Szekeres, Christian Munk, Tsonko M Tsonkov, Stefan Mordalski,
   Kasper Harpsøe, Alexander S Hauser, Andrzej J Bojarski, David E Gloriam,
   *GPCRdb in 2018: adding GPCR structure models and ligands*,
   Nucleic Acids Research, Volume 46, Issue D1, 4 January 2018, Pages D440–D446,
   `<https://doi.org/10.1093/nar/gkx1109>`_

 * [4] Helen M. Berman, John Westbrook, Zukang Feng, Gary Gilliland, T. N. Bhat,
   Helge Weissig, Ilya N. Shindyalov, Philip E. Bourne,
   *The Protein Data Bank*,
   Nucleic Acids Research, Volume 28, Issue 1, 1 January 2000, Pages 235–242,
   `<https://doi.org/10.1093/nar/28.1.235>`_

 * [5] Vignir Isberg, Chris de Graaf, Andrea Bortolato, Vadim Cherezov, Vsevolod Katritch,
   Fiona H. Marshall, Stefan Mordalski, Jean-Philippe Pin, Raymond C. Stevens, Gerrit  Vriend, David E. Gloriam,
   *Generic GPCR residue numbers – aligning topology maps while minding the gaps*,
   Trends in Pharmacological Sciences, Volume 36, Issue 1, 2015, Pages 22-31,
   `<https://doi.org/10.1016/j.tips.2014.11.001.>`_
"""
from .nomenclature import *
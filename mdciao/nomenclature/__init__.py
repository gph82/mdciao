r"""
Get and manipulate consensus nomenclature for GPCRs, G-proteins, and Kinases.

Uses local files and/or accesses the following databases and their public APIs

 * `GPCRdb <https://gpcrdb.org/>`_
 * `Common Gα Numbering (CGN) scheme <https://www.mrc-lmb.cam.ac.uk/CGN/>`_
 * `KLIFS - The structural kinase database <https://klifs.net/>`_
 * `RCSB PDB rcsb.org <https://rcsb.org/>`_
 * `UniProt Knowledgebase <https://www.uniprot.org/>`_

Please see the individual documentation of the `Labeler Classes <Classes>`_ for further references and cite them
whenever you use these nomenclature schemes in your final publication.

Additionally, use :obj:`mdciao.nomenclature.references` anytime to get more info.

.. currentmodule:: mdciao.nomenclature

Classes
=======

.. autosummary::
    :toctree: generated

    LabelerGPCR
    LabelerCGN
    LabelerKLIFS
    AlignerConsensus
    Literature

Functions
=========

.. autosummary::
    :nosignatures:
    :toctree: generated/

    guess_by_nomenclature
    guess_nomenclature_fragments
    references

References
==========
These are some of the references most relevant to this module:

* GPCRdb and naming schemes therein

 * Kooistra, A. J., Mordalski, S., Pándy-Szekeres, G., Esguerra, M., Mamyrbekov, A., Munk, C., … Gloriam, D. E. (2021). GPCRdb in 2021: Integrating GPCR sequence, structure and function. Nucleic Acids Research, 49(D1), D335–D343. https://doi.org/10.1093/nar/gkaa1080

 * Isberg, V., De Graaf, C., Bortolato, A., Cherezov, V., Katritch, V., Marshall, F. H., … Gloriam, D. E. (2015). Generic GPCR residue numbers - Aligning topology maps while minding the gaps. Trends in Pharmacological Sciences, 36(1), 22–31. https://doi.org/10.1016/j.tips.2014.11.001

 * Isberg, V., Mordalski, S., Munk, C., Rataj, K., Harpsøe, K., Hauser, A. S., … Gloriam, D. E. (2016). GPCRdb: An information system for G protein-coupled receptors. Nucleic Acids Research, 44(D1), D356–D364. https://doi.org/10.1093/nar/gkv1178

* Further GPCR naming schemes

 * Ballesteros, J. A., & Weinstein, H. (1995). Integrated methods for the construction of three-dimensional models and computational probing of structure-function relations in G protein-coupled receptors. Methods in Neurosciences, 25(C), 366–428. https://doi.org/10.1016/S1043-9471(05)80049-7

 * Wu, H., Wang, C., Gregory, K. J., Han, G. W., Cho, H. P., Xia, Y., … Stevens, R. C. (2014). Structure of a class C GPCR metabotropic glutamate receptor 1 bound to an allosteric modulator. Science, 344(6179), 58–64. https://doi.org/10.1126/science.1249489

 * Pin, J. P., Galvez, T., & Prézeau, L. (2003). Evolution, structure, and activation mechanism of family 3/C G-protein-coupled receptors. Pharmacology and Therapeutics, 98(3), 325–354. https://doi.org/10.1016/S0163-7258(03)00038-X

 * Wootten, D., Simms, J., Miller, L. J., Christopoulos, A., & Sexton, P. M. (2013). Polar transmembrane interactions drive formation of ligand-specific and signal pathway-biased family B G protein-coupled receptor conformations. Proceedings of the National Academy of Sciences of the United States of America, 110(13), 5211–5216. https://doi.org/10.1073/pnas.1221585110

 * Oliveira, L., Paiva, A. C. M., & Vriend, G. (1993). A common motif in G-protein-coupled seven transmembrane helix receptors. Journal of Computer-Aided Molecular Design, 7(6), 649–658. https://doi.org/10.1007/BF00125323

 * Schwartz, T. W., Gether, U., Schambye, H. T., & Hjorth, S. A. (1995). Molecular mechanism of action of non-peptide ligands for peptide receptors. Current Pharmaceutical Design, 1, 325–342.

 * Schwartz, T. W. (1994). Locating ligand-binding sites in 7tm receptors by protein engineering. Current Opinion in Biotechnology, 5(4), 434–444. https://doi.org/10.1016/0958-1669(94)90054-X

 * Baldwin, J. M. (1993). The probable arrangement of the helices in G protein-coupled receptors. The EMBO Journal, 12(4), 1693–1703. https://doi.org/10.1002/J.1460-2075.1993.TB05814.X

 * Baldwin, J. M., Schertler, G. F. X., & Unger, V. M. (1997). An alpha-carbon template for the transmembrane helices in the rhodopsin family of G-protein-coupled receptors. Journal of Molecular Biology, 272(1), 144–164. https://doi.org/10.1006/jmbi.1997.1240

* CGN naming scheme

 * Flock, T., Ravarani, C. N. J., Sun, D., Venkatakrishnan, A. J., Kayikci, M., Tate, C. G., … Babu, M. M. (2015). Universal allosteric mechanism for Gα activation by GPCRs. Nature 2015 524:7564, 524(7564), 173–179. https://doi.org/10.1038/nature14663

* KLIFS 85 ligand binding site residues of kinases

 * Van Linden, O. P. J., Kooistra, A. J., Leurs, R., De Esch, I. J. P., & De Graaf, C. (2014). KLIFS: A knowledge-based structural database to navigate kinase-ligand interaction space. Journal of Medicinal Chemistry, 57(2), 249–277. https://doi.org/10.1021/JM400378W
 * Kooistra, A. J., Kanev, G. K., Van Linden, O. P. J., Leurs, R., De Esch, I. J. P., & De Graaf, C. (2016). KLIFS: a structural kinase-ligand interaction database. Nucleic Acids Research, 44(D1), D365–D371. https://doi.org/10.1093/NAR/GKV1082
 * Kanev, G. K., de Graaf, C., Westerman, B. A., de Esch, I. J. P., & Kooistra, A. J. (2021). KLIFS: an overhaul after the first 5 years of supporting kinase research. Nucleic Acids Research, 49(D1), D562–D569. https://doi.org/10.1093/NAR/GKAA895


* PDB

 * Berman, H. M., Westbrook, J., Feng, Z., Gilliland, G., Bhat, T. N., Weissig, H., … Bourne, P. E. (2000, January 1). The Protein Data Bank. Nucleic Acids Research. Oxford Academic. https://doi.org/10.1093/nar/28.1.235

* UniProt

 * Bateman, A., Martin, M. J., Orchard, S., Magrane, M., Agivetova, R., Ahmad, S., … Zhang, J. (2021). UniProt: the universal protein knowledgebase in 2021. Nucleic Acids Research, 49(D1), D480–D489. https://doi.org/10.1093/NAR/GKAA1100




"""
from .nomenclature import *
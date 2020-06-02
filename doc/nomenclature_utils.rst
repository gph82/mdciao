nomenclature_utils
==================

Utils for handling nomenclature related functionality. Currently handles:
 * Ballesteros-Weinstein (BW) (link to BW on the GPCR md TODO SOFI CHECK)
 * Common G-protein nomenclature (CGN) (https://www.mrc-lmb.cam.ac.uk/CGN/index.html)


.. currentmodule:: mdciao.nomenclature_utils

.. autosummary::
    add_loop_definitions_to_TM_residx_dict
    LabelerBW
    CGN_finder
    LabelerCGN
    LabelerConsensus
    csv_table2TMdefs_res_idxs
    guess_missing_BWs
    md_load_rscb
    order_BW
    order_CGN
    order_frags
    PDB_finder
    table2BW_by_AAcode
    table2TMdefs_resSeq
    top2CGN_by_AAcode


.. autoclass:: mdciao.nomenclature_utils.LabelerCGN
    :members:
    :inherited-members:

.. autoclass:: mdciao.nomenclature_utils.LabelerBW
    :members:
    :inherited-members:
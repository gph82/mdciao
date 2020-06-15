r"""


This module methods and classes related to the
computation of residue-residue contacts from
molecular dynamics trajectories.

It is recommened to use higher-level methods of the API,
like those exposed by :obj:`mdciao.cli` to create
:obj:`ContactPair` or :obj:`ContactGroup` objects,
but of course experienced users can access them directly.


.. currentmodule:: mdciao.contacts


Functions
=========

.. autosummary::
   :toctree: generated/

   per_traj_ctc
   trajs2ctcs
   select_and_report_residue_neighborhood_idxs


Classes
=======

.. autosummary::
    :toctree: generated

    ContactPair
    ContactGroup

"""
from .contacts import *

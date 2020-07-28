r"""

Computation, bookkeeping, and manipulation
of residue-residue contacts.

The methods wrap around a modified version
of `mdtraj.compute_contacts` to extract
residue-residue distances. The modifications
consist in including the indices of the closest
atom-pairs in the returned values.

The classes contain and abstract these distances into
sense-making groups, offering methods to operate directly
on all residue-residue distances.

It is recommended to use higher-level methods of the API,
like those exposed by :obj:`mdciao.cli` to create
:obj:`ContactPair` or :obj:`ContactGroup` objects and
then use the methods stated above, but of course experienced
users can instantiate them directly.

.. currentmodule:: mdciao.contacts


Functions
=========

.. autosummary::
    :nosignatures:
    :toctree: generated/

    per_traj_ctc
    trajs2ctcs
    select_and_report_residue_neighborhood_idxs


Classes
=======

.. autosummary::
    :nosignatures:
    :toctree: generated

    Residues
    ContactPair
    ContactGroup
    GroupOfInterfaces

"""
from .contacts import *
from .contacts import _linear_switchoff

r"""
Guess and manipulate fragments, i.e., sub-regions
of molecular topologies. A fragment is just an iterable
of integers representing the residue indices belonging to it.

Methods not documented here are either private or will be soon,
please note the API is not yet stable.

.. currentmodule:: mdciao.fragments

.. autosummary::
   :toctree: generated/
   :nosignatures:

    get_fragments
    print_frag
    overview
    match_fragments


"""
from .fragments import *
from .fragments import _get_fragments_by_jumps_in_sequence
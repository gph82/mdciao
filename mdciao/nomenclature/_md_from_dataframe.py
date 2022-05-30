##############################################################################
#    This file is part of mdciao.
#
#    Copyright 2020 Charité Universitätsmedizin Berlin and the Authors
#
#    Authors: Guillermo Pérez-Hernandez
#    Contributors:
#
#    mdciao is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    mdciao is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with mdciao.  If not, see <https://www.gnu.org/licenses/>.
##############################################################################

##############################################################################
#    Please note: The following method "from_dataframe" is a
#    modified version of the one found in the original MDTraj
#    Python Library, whose original authors and copyright
#    holders are listed below.

#    The modifications consist in a bugfix that adds a third condition
#    to force new residue creation. A pull request has been already
#    opened upstream and hopefully soon this file won't be needed
#
##############################################################################


##############################################################################
# MDTraj: A Python Library for Loading, Saving, and Manipulating
#         Molecular Dynamics Trajectories.
# Copyright 2012-2013 Stanford University and the Authors
#
# Authors: Christian Schwantes
# Contributors: Robert McGibbon
#
# MDTraj is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MDTraj. If not, see <http://www.gnu.org/licenses/>.
##############################################################################

# opened issues for this:
# https://github.com/mdtraj/mdtraj/issues/1569
# https://github.com/mdtraj/mdtraj/pull/1740
# Further reading
# https://www.oreilly.com/library/view/understanding-open-source/0596005814/ch03.html
# https://tldrlegal.com/license/gnu-lesser-general-public-license-v3-(lgpl-3)
# http://oss-watch.ac.uk/resources/lgpl
# https://www.gnu.org/licenses/gpl-faq.html#AllCompatibility

##############################################################################
# Imports
##############################################################################

from __future__ import print_function, division

import itertools
import numpy as np
import os
import xml.etree.ElementTree as etree
from collections import namedtuple

from mdtraj.core import element as elem
from mdtraj.core.residue_names import (_PROTEIN_RESIDUES, _WATER_RESIDUES,
                                       _AMINO_ACID_CODES)
from mdtraj.core.selection import parse_selection
from mdtraj.utils import ilen, import_, ensure_type
from mdtraj.utils.six import string_types
from mdtraj.utils.singleton import Singleton

from mdtraj.core.topology import Atom, float_to_bond_type

def from_dataframe(atoms, bonds=None):
    """Create a mdtraj topology from a pandas data frame

    Parameters
    ----------
    atoms : pandas.DataFrame
        The atoms in the topology, represented as a data frame. This data
        frame should have columns "serial" (atom index), "name" (atom name),
        "element" (atom's element), "resSeq" (index of the residue)
        "resName" (name of the residue), "chainID" (index of the chain),
        and optionally "segmentID", following the same conventions
        as wwPDB 3.0 format.
    bonds : np.ndarray, shape=(n_bonds, 4) or (n_bonds, 2), dtype=float, Optional
        The bonds in the topology, represented as a n_bonds x 4 or n_bonds x 2
        size array of the indices of the atoms involved, type, and order of each
        bond, represented as floats. Type and order are optional. Specifying bonds
        here is optional. To create standard protein bonds, you can use
        `create_standard_bonds` to "fill in" the bonds on your newly created
        Topology object, although type and order of bond are not computed if
        that method is used.

    See Also
    --------
    create_standard_bonds
    """
    pd = import_('pandas')

    if bonds is None:
        bonds = np.zeros([0, 4], dtype=float)

    for col in ["name", "element", "resSeq",
                "resName", "chainID", "serial"]:
        if col not in atoms.columns:
            raise ValueError('dataframe must have column %s' % col)

    if "segmentID" not in atoms.columns:
        atoms["segmentID"] = ""

    from mdtraj import Topology
    out = Topology()
    if not isinstance(atoms, pd.DataFrame):
        raise TypeError('atoms must be an instance of pandas.DataFrame. '
                        'You supplied a %s' % type(atoms))
    if not isinstance(bonds, np.ndarray):
        raise TypeError('bonds must be an instance of numpy.ndarray. '
                        'You supplied a %s' % type(bonds))

    if not np.all(np.arange(len(atoms)) == atoms.index):
        raise ValueError('atoms must be uniquely numbered '
                         'starting from zero.')
    out._atoms = [None for i in range(len(atoms))]

    c = None
    r = None
    previous_chainID = None
    previous_resName = None
    previous_resSeq = None
    for atom_index, atom in atoms.iterrows():

        int(atom_index)  # Fixes bizarre hashing issue on Py3K.  See #545

        if atom['chainID'] != previous_chainID:
            previous_chainID = atom['chainID']

            c = out.add_chain()

        if atom['resSeq'] != previous_resSeq or atom['resName'] != previous_resName or c.n_atoms == 0:
            previous_resSeq = atom['resSeq']
            previous_resName = atom['resName']

            r = out.add_residue(atom['resName'], c, atom['resSeq'], atom['segmentID'])

        a = Atom(atom['name'], elem.get_by_symbol(atom['element']),
                 atom_index, r, serial=atom['serial'])
        out._atoms[atom_index] = a
        r._atoms.append(a)

    for bond in bonds:
        ai1 = int(bond[0])
        ai2 = int(bond[1])
        try:
            bond_type = float_to_bond_type(bond[2])
            bond_order = int(bond[3])
            if bond_order == 0:
                bond_order = None
        except IndexError:
            # Does not exist
            bond_type = None
            bond_order = None
        out.add_bond(out.atom(ai1), out.atom(ai2), bond_type, bond_order)

    out._numAtoms = out.n_atoms
    return out

#!/usr/bin/env python3

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

from mdciao.cli import interface
from mdciao.parsers import parser_for_interface
parser = parser_for_interface()
a  = parser.parse_args()
#from mdciao.command_line_tools import _inform_of_parser
#_inform_of_parser(parser)

if not a.fragmentify:
    a.fragment_names="None"

# Make a dictionary out ot of it and pop the positional keywords
b = {key:getattr(a,key) for key in dir(a) if not key.startswith("_")}
for key in ["trajectories", "fragmentify"]:
    b.pop(key)
b["interface_cutoff_Ang"] = [None if b["interface_cutoff_Ang"]==0 else b["interface_cutoff_Ang"]][0]

neighborhood = interface(a.trajectories,**b)
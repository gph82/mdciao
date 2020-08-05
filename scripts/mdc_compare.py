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
from mdciao.parsers import parser_for_compare_neighborhoods
from mdciao.cli import compare
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from pandas import DataFrame
# Get and instantiate parser
parser = parser_for_compare_neighborhoods()
a  = parser.parse_args()
nf = len(a.files)
if a.keys is not None:
    assert len(a.keys.split(","))==nf, "Mismatch number of files vs number of keys %u vs %u"%(nf,len(a.keys.split()))
    keys = a.keys.split(",")
    file_dict = {key:val for key, val in zip(keys, a.files)}
else:
    file_dict = a.files

colors =  a.colors.split(",")

mut_dict = {}
if a.mutations is not None:
    for pair in a.mutations.split(","):
        key, val = pair.split(":")
        mut_dict[key.replace(" ","")]=val.replace(" ","")

# Call the method
myfig, freqs, posret = compare(file_dict,
                               colors=colors,
                               anchor=a.anchor,
                               mutations_dict=mut_dict,
                               figsize=None,
                               )
myfig.tight_layout()
fname = "%s.%s"%(a.output_desc,a.graphic_ext.strip("."))
myfig.savefig(fname)
fname = "%s.xlsx"%a.output_desc
DataFrame.from_dict(posret).to_excel(fname)
plt.show()


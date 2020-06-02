#!/usr/bin/env python3
from mdciao.parsers import parser_for_compare_neighborhoods
from mdciao.command_line_tools import compare
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Get and instantiate parser
parser = parser_for_compare_neighborhoods()
a  = parser.parse_args()
nf = len(a.files)
if a.keys is None:
   keys = ["file %u"%ii for ii in range(nf)]
else:
    assert len(a.keys.split(","))==nf
    keys = a.keys.split(",")
file_dict = {key: file for key, file in zip(keys, a.files)}
col_dict =  {key: col for key, col in zip(keys, a.colors.split(","))}
mut_dict = {}
if a.mutations is not None:
    for pair in a.mutations.split(","):
        key, val = pair.split(":")
        mut_dict[key.replace(" ","")]=val.replace(" ","")

# Call the method
myfig, freqs, posret = compare(file_dict, col_dict,
                               anchor=a.anchor,
                               mutations_dict=mut_dict)
myfig.tight_layout()
fname = "%s.%s"%(a.output_desc,a.graphic_ext.strip("."))
myfig.savefig(fname)
plt.show()


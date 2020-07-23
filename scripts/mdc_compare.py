#!/usr/bin/env python3
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

colordict =  a.colors.split(",")

mut_dict = {}
if a.mutations is not None:
    for pair in a.mutations.split(","):
        key, val = pair.split(":")
        mut_dict[key.replace(" ","")]=val.replace(" ","")

# Call the method
myfig, freqs, posret = compare(file_dict, colordict=colordict,
                               anchor=a.anchor,
                               mutations_dict=mut_dict,
                               figsize=None,
                               width=None)
myfig.tight_layout()
fname = "%s.%s"%(a.output_desc,a.graphic_ext.strip("."))
myfig.savefig(fname)
fname = "%s.xlsx"%a.output_desc
DataFrame.from_dict(posret).to_excel(fname)
plt.show()


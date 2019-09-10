#!/home/perezheg/miniconda3/bin/python
import numpy as np
from sofi_functions.command_line_tools import residue_neighborhoods, my_parser

# Get and instantiate parser
parser = my_parser()
a  = parser.parse_args()

# Make a dictionary out ot of it and pop the positional keywords
b = {key:getattr(a,key) for key in dir(a) if not key.startswith("_")}
for key in ["topology","trajectories","resSeq_idxs","output_npy"]:
    b.pop(key)

# Call the method
out_dict = residue_neighborhoods(a.topology, a.trajectories, a.resSeq_idxs, **b)

if out_dict is not None:
    fname = a.output_npy
    if not fname.endswith(".npy"):
        fname += ".npy"
    np.save(fname,out_dict)
    print(fname)
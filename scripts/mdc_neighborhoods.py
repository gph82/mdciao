#!/home/perezheg/miniconda3/bin/python
import numpy as np
from mdciao.parsers import parser_for_rn, _inform_of_parser
from mdciao.command_line_tools import residue_neighborhoods

# Get and instantiate parser
parser = parser_for_rn()
a  = parser.parse_args()

# Make a dictionary out ot of it and pop the positional keywords
b = {key:getattr(a,key) for key in dir(a) if not key.startswith("_")}
for key in ["topology","trajectories","residues"]:
    b.pop(key)

# Call the method
out_dict = residue_neighborhoods(a.topology, a.trajectories, a.residues, **b)
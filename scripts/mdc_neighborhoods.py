#!/usr/bin/env python3
from mdciao.parsers import parser_for_rn, _inform_of_parser
from mdciao.cli import residue_neighborhoods

# Get and instantiate parser
parser = parser_for_rn()
a  = parser.parse_args()

if not a.fragmentify:
    a.fragments=["None"]
    a.fragment_names="None"
# Make a dictionary out ot of it and pop the positional keywords
b = {key:getattr(a,key) for key in dir(a) if not key.startswith("_")}
for key in ["topology","trajectories","residues", "fragmentify"]:
    b.pop(key)

# Call the method
out_dict = residue_neighborhoods(a.topology, a.trajectories, a.residues, **b)
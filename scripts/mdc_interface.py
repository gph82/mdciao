#!/usr/bin/env python3
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
for key in ["topology","trajectories","fragmentify"]:
    b.pop(key)
neighborhood = interface(a.topology, a.trajectories, **b)

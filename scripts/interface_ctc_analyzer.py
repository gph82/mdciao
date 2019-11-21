#!/home/perezheg/miniconda3/bin/python
from sofi_functions.command_line_tools import interface
from sofi_functions.parsers import parser_for_interface
parser = parser_for_interface()
a  = parser.parse_args()
#from sofi_functions.command_line_tools import _inform_of_parser
#_inform_of_parser(parser)

# Make a dictionary out ot of it and pop the positional keywords
b = {key:getattr(a,key) for key in dir(a) if not key.startswith("_")}
for key in ["topology","trajectories"]:
    b.pop(key)
neighborhood = interface(a.topology, a.trajectories, **b)

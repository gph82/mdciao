#!/home/perezheg/miniconda3/bin/python

from mdciao.command_line_tools import sites
from mdciao.parsers import parser_for_sites

# Get and instantiate parser
parser = parser_for_sites()
a  = parser.parse_args()
#_inform_of_parser(parser)

# Make a dictionary out ot of it and pop the positional keywords
b = {key:getattr(a,key) for key in dir(a) if not key.startswith("_")}
for key in ["topology","trajectories","site_files"]:
    b.pop(key)

sites(a.topology, a.trajectories, a.site_files, **b)

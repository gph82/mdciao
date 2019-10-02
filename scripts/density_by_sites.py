#!/home/perezheg/miniconda3/bin/python

from sofi_functions.command_line_tools import density_by_sites
from sofi_functions.command_line_tools import _inform_of_parser, parser_for_densities

# Get and instantiate parser
parser = parser_for_densities()
a  = parser.parse_args()
#_inform_of_parser(parser)

# Make a dictionary out ot of it and pop the positional keywords
b = {key:getattr(a,key) for key in dir(a) if not key.startswith("_")}
for key in ["topology","trajectories","site_files"]:
    b.pop(key)

density_by_sites(a.topology, a.trajectories, a.site_files, **b)

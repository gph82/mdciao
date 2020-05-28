#!python
from mdciao.command_line_tools import contact_map

from mdciao.parsers import parser_for_contact_map
parser = parser_for_contact_map()
a = parser.parse_args()
#from mdciao.command_line_tools import _inform_of_parser
#_inform_of_parser(parser)

# Make a dictionary out ot of it and pop the positional keywords
b = {key:getattr(a,key) for key in dir(a) if not key.startswith("_")}
for key in ["topology","trajectories"]:
    b.pop(key)
b["list_ctc_cutoff_Ang"] = [float(ii) for ii in b["list_ctc_cutoff_Ang"].split(",")]

contact_map_dicts = contact_map(a.topology, a.trajectories, **b)

import numpy as _np
_np.save("ctc_map_dicts.npy",contact_map_dicts)

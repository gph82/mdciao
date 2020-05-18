#!/home/perezheg/miniconda3/bin/python
from mdciao.parsers import parser_for_CGN_overview
from mdciao.command_line_tools import _fragment_overview
parser = parser_for_CGN_overview()
a  = parser.parse_args()
_fragment_overview(a,"CGN")

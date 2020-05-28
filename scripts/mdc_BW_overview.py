#!python
from mdciao.parsers import parser_for_BW_overview
from mdciao.command_line_tools import _fragment_overview
parser = parser_for_BW_overview()
a  = parser.parse_args()
_fragment_overview(a,"BW")

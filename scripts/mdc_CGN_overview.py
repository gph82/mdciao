#!/usr/bin/env python3
from mdciao.parsers import parser_for_CGN_overview
from mdciao.cli.cli import _fragment_overview
parser = parser_for_CGN_overview()
a  = parser.parse_args()
_fragment_overview(a,"CGN")

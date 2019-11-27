#!/home/perezheg/miniconda3/bin/python

from mdciao.command_line_tools import site_figures
from mdciao.parsers import _parser_add_topology, _parser_add_sites, _parser_add_nomenclature
import argparse

# Get and instantiate parser
parser = argparse.ArgumentParser(description='Create replines from sitefiles')
_parser_add_topology(parser)
_parser_add_sites(parser)
_parser_add_nomenclature(parser)
a  = parser.parse_args()
print(a)
site_figures(a.topology,a.site_files,CGN_PDB=a.CGN_PDB)

#!/home/perezheg/miniconda3/bin/python
from mdciao.parsers import parser_for_CGN_overview
from mdciao.nomenclature_utils import LabelerCGN
import mdtraj as _md

parser = parser_for_CGN_overview()
a  = parser.parse_args()

CGN = LabelerCGN(a.CGN_PDB)
CGN.top2defs(_md.load(a.topology).top)

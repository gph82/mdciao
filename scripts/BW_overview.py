#!/home/perezheg/miniconda3/bin/python
from mdciao.parsers import parser_for_BW_overview
from mdciao.nomenclature_utils import BW_transformer
import mdtraj as _md

parser = parser_for_BW_overview()
a  = parser.parse_args()

BW = BW_transformer(a.BW_uniprot)
BW.top2defs(_md.load(a.topology).top)

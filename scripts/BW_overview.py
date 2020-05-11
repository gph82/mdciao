#!/home/perezheg/miniconda3/bin/python
from mdciao.parsers import parser_for_BW_overview
from mdciao.nomenclature_utils import LabelerBW
import mdtraj as _md

parser = parser_for_BW_overview()
a  = parser.parse_args()

BW = LabelerBW(a.BW_uniprot,
               write_to_disk=a.write_to_disk)
top = _md.load(a.topology).top
map_conlab = BW.top2map(top)
BW.top2defs(top, map_conlab=map_conlab)
if a.print_conlab:
    for ii, ilab in enumerate(map_conlab):
        print(ii, top.residue(ii),ilab)
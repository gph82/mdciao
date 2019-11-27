#!/home/perezheg/miniconda3/bin/python
from mdciao.parsers import parser_for_frag_overview
from mdciao.fragments import get_fragments
import mdtraj as _md
import argparse

all_methods = ['resSeq',
               'resSeq+',
               'bonds',
               'resSeq_bonds',
               'chains']


parser = parser_for_frag_overview()
a  = parser.parse_args()

if a.methods[0].lower()=='all':
    try_methods = all_methods
else:
    for imethd in a.methods:
        assert imethd in all_methods,('input method %s is not known. ' \
                                      'Know methods are %s '%(imethd,all_methods))
    try_methods = a.methods

for method in try_methods:
    get_fragments(_md.load(a.topology).top,
                  method=method)
    print()
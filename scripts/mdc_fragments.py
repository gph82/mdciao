#!/usr/bin/env python3
from mdciao.parsers import parser_for_frag_overview
from mdciao.fragments import overview
import mdtraj as _md

parser = parser_for_frag_overview()
a  = parser.parse_args()
overview(_md.load(a.topology).top, methods=a.methods,AAs=a.AAs)
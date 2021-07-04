#!/usr/bin/env python3

##############################################################################
#    This file is part of mdciao.
#    
#    Copyright 2020 Charité Universitätsmedizin Berlin and the Authors
#
#    Authors: Guillermo Pérez-Hernandez
#    Contributors:
#    
#    mdciao is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    mdciao is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with mdciao.  If not, see <https://www.gnu.org/licenses/>.
##############################################################################

from mdciao.examples.examples import ExamplesCLTs
from mdciao.parsers import parser_for_examples, _inform_of_parser
from os import path
from fnmatch import filter as _fnfilter
execute = True
list = False
import sys
parser = parser_for_examples()
if len(sys.argv)==1:
    args = parser.parse_args(args=["-h"])
else:
    args = parser.parse_args(args=None)

#print(_inform_of_parser(parser))
#args = parser.parse_args()

clt = args.clt
ex = ExamplesCLTs(short=args.short)

if args.clt.endswith(".py"):
    clt = path.splitext(args.clt)[0]
clts = _fnfilter(ex.clts,"*%s*"%clt)
if len(clts)==0:
    print("'%s' did not return any command-line-tool."%clt)
    print(parser.epilog)
for clt in clts:
    ex.show(clt)
    if args.execute:
        ex.run(clt, show=False)

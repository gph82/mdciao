#!/usr/bin/env python3
from mdciao.examples import ExamplesCLTs
import argparse
from os import path
from fnmatch import filter as _fnfilter
execute = True
list = False
ex = ExamplesCLTs()
desc1 = "Wrapper script to showcase and optionally run examples of the \n" \
        "command-line-tools that ship with mdciao."
parser = argparse.ArgumentParser(description=desc1+
                                             "\n To show available examples and nothing more\n"
                                             "type any of 'mdc_examples.py l/list/?'"
                                 )

parser.add_argument("clt",
                    default=None,
                    type=str,
                    help="The example script to run. Can be part of the name as well\n "
                         "(neigh) for neighborhoods ")

parser.add_argument("-x",
                    action="store_true",
                    dest="execute",
                    help="whether to execute the command after showing it. "
                         "Default is False.")
parser.set_defaults(execute=False)

parser.set_defaults(show_list=False)

args = parser.parse_args()
clt = args.clt
if args.clt in ["l","lists","?"]:
    print(desc1)
    print("Availble command line tools are")
    print("\n".join([" * %s.py"%key for key in ex.clts]))
    print("Issue:\n"
          " * 'mdc_command.py -h' to view the command's documentation or\n"
          " * 'mdc_examples.py mdc_command' to show and/or run an example of that command ")
else:
    if args.clt.endswith(".py"):
        clt = path.splitext(args.clt)[0]
    clts = _fnfilter(ex.clts,"*%s*"%clt)
    for clt in clts:
        ex.show(clt)
        if args.execute:
            ex.run(clt, show=False)

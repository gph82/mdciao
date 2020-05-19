#!/home/perezheg/miniconda3/bin/python
from mdciao.examples import ExamplesCLTs
import argparse
from os import path
execute = True
list = False
ex = ExamplesCLTs()
parser = argparse.ArgumentParser(description="Wrapper script to showcase and optionally"
                                             "run examples of the command-line-tools (clt)"
                                             "that ship with mdciao"
                                 )

parser.add_argument("clt",
                    default=None,
                    type=str,
                    help="The example script to run")

parser.add_argument("-x",
                    action="store_true",
                    dest="execute",
                    help="whether to execute the command after showing it. "
                         "Default is False.")
parser.set_defaults(execute=False)


parser.add_argument("-l",
                    action="store_true",
                    dest="show_list",
                    help="show available examples and nothing more")
parser.set_defaults(show_list=False)

args = parser.parse_args()
clt = args.clt
if args.show_list:
    print("Availble command line tools are")
    print("\n".join(["%s.py"%key for key in ex.clts]))
else:
    if args.clt.endswith(".py"):
        clt = path.splitext(args.clt)[0]
    ex.show(clt)
    if args.execute:
        ex.run(clt, show=False)

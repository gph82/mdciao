#!/home/perezheg/miniconda3/bin/python
import argparse
import numpy as np
from sofi_functions.command_line_tools import residue_neighborhoods

parser = argparse.ArgumentParser(description='Small residue-residue contact analysis tool, initially developed for the '
                                             'receptor-G-protein complex. The provides the residue indices')

parser.add_argument('topology',    type=str,help='Topology file')
parser.add_argument('trajectories',type=str,help='trajectory file(s)', nargs='+')
parser.add_argument('--resSeq_idxs',type=str,help='the resSeq idxs of interest (in VMD these are called "resid"). Can be in a format 1,2-6,10,20-25')
parser.add_argument("--ctc_cutoff_Ang",type=float, help="The cutoff distance between two residues for them to be considered in contact. Default is 3 Angstrom.", default=3)
parser.add_argument("--stride", type=int, help="Stride down the input trajectoy files by this factor. Default is 1.", default=1)
parser.add_argument("--n_ctcs", type=int, help="Only the first n_ctcs most-frequent contacts will be written to the ouput. Default is 5.", default=5)
parser.add_argument("--n_nearest", type=int, help="Ignore this many nearest neighbors when computing neighbor lists. 'Near' means 'connected by this many bonds'. Default is 4.", default=4)
parser.add_argument("--chunksize_in_frames", type=int, help="Trajectories are read in chunks of this size (helps with big files and memory problems). Default is 10000", default=10000)
parser.add_argument("--nlist_cutoff_Ang", type=float, help="Cutoff for the initial neighborlist. Only atoms that are within this distance in the original reference "
                                                          "(the topology file) are considered potential neighbors of the residues in resSeq_idxs, s.t. "
                                                          "non-necessary distances (e.g. between N-terminus and G-protein) are not even computed. "
                                                          "Default is 15 Angstrom.", default=15)


parser.add_argument('--fragments',    dest='fragmentify', action='store_true', help="Auto-detect fragments (i.e. breaks) in the peptide-chain. Default is true.")
parser.add_argument('--no-fragments', dest='fragmentify', action='store_false')
parser.set_defaults(fragmentify=True)

parser.add_argument('--sort',    dest='sort', action='store_true', help="Sort the resSeq_idxs list. Defaut is True")
parser.add_argument('--no-sort', dest='sort', action='store_false')
parser.set_defaults(sort=True)

parser.add_argument('--pbc',    dest='pbc', action='store_true', help="Consider periodic boundary conditions when computing distances."
                                                                      " Defaut is True")
parser.add_argument('--no-pbc', dest='pbc', action='store_false')
parser.set_defaults(pbc=True)

parser.add_argument('--ask_fragment',    dest='ask', action='store_true', help="Interactively ask for fragment assignemnt when input matches more than one resSeq")
parser.add_argument('--no-ask_fragment', dest='ask', action='store_false')
parser.set_defaults(ask=True)
parser.add_argument('--output_npy', type=str, help="Name of the output.npy file for storing this runs' results", default='output.npy')
parser.add_argument('--output_ext', type=str, help="Extension of the output graphics, default is .pdf", default='.pdf')
parser.add_argument('--output_dir', type=str, help="directory to which the results are written. Default is '.'", default='.')

parser.add_argument('--fragment_names', type=str,
                    help="Name of the fragments. Leave empty if you want them automatically named."
                         " Otherwise, give a quoted list of strings separated by commas, e.g. "
                         "'TM1, TM2, TM3,'",
                    default="")
parser.add_argument("--BW_file", type=str, help="Json file with info about the Ballesteros-Weinstein definitions as downloaded from the GPRCmd", default='None')
parser.add_argument("--CGN_PDB", type=str, help="PDB code for a consensus G-protein nomenclature", default='None')

a  = parser.parse_args()

b = {key:getattr(a,key) for key in dir(a) if not key.startswith("_")}

for key in ["topology","trajectories","resSeq_idxs","output_npy"]:
    b.pop(key)

out_dict = residue_neighborhoods(a.topology, a.trajectories, a.resSeq_idxs, **b)

if out_dict is not None:
    fname = a.output_npy
    if not fname.endswith(".npy"):
        fname += ".npy"
    np.save(fname,out_dict)
    print(fname)
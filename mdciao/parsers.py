##############################################################################
#    This file is part of mdciao.
#    
#    Copyright 2025 Charité Universitätsmedizin Berlin and the Authors
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

import argparse
from mdciao.plots.plots import _colorstring
from mdciao.examples.examples import ExamplesCLTs as _xCLT
from matplotlib.colors import is_color_like as _is_color_like

# https://stackoverflow.com/questions/3853722/python-argparse-how-to-insert-newline-in-the-help-text
class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)

def _inform_of_parser(parser,args=None):
    r"""
    Print all the valuesof the variables in a parser
    TODO find out the native way of doing this
    Parameters
    ----------
    parser

    Returns
    -------

    """
    # TODO is this too hacky, wouldn't *args suffice?
    # This is just to run tests
    if args is None:
        a = parser.parse_args()
    else:
        a = parser.parse_args(args)
    for key, __ in a._get_kwargs():
        dval = parser.get_default(key)
        fmt = '%s=%s,'
        if isinstance(dval, str):
            fmt = '%s="%s",'
        print(fmt % (key, dval))

class Populate_input_Action(argparse.Action):
    r"""
    Create an alias named "input_" for this argument.

    Helps create a common interface for cli._fragment_overview

    https://docs.python.org/3/library/argparse.html#action
    Ctr+F "FooAction"
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, 'input_', values)

def _parser_top_traj(description=None):
    r"""
    Instantiate the basic parsers which can take topology and trajectories

    Parameters
    ----------
    description: str
        The initial description of the parser

    Returns
    -------

    """
    parser = argparse.ArgumentParser(description=description, formatter_class=SmartFormatter)
    _parser_add_topology(parser)
    _parser_add_trajectories(parser)
    return parser

def _parser_add_trajectories(parser):
    parser.add_argument('trajectories', type=str, help='trajectory file(s)', nargs='+')

def _parser_add_topology(parser):
    parser.add_argument('topology', type=str, help='Topology file')

def _parser_add_cutoff(parser):
    parser.add_argument("--ctc_cutoff_Ang", "-co", type=float,
                        help="The cutoff distance between two residues for them to be considered in contact. "
                             "Default is 4.5 Angstrom.",
                        default=4.5)

def _parser_add_n_neighbors(parser, default=4):
    parser.add_argument("--n_nearest","-nn", type=int,
                        help="Ignore this many nearest neighbors when computing neighbor lists."
                             " 'Near' means 'connected by this many bonds'. Default is %u."%default,
                        default=default)

def _parser_add_stride(parser,
                       help="Stride down the input trajectoy files by this factor. Default is 1." ):
    parser.add_argument("--stride", type=int,
                        help=help, default=1)

def _parser_add_chunk(parser,help="Trajectories are read in chunks of this size. "
                                  "Helps with big files and/or large number of contacts"
                                  " when you run into memory problems. Default is %s",
                      default=2000):
    parser.add_argument("--chunksize_in_frames", type=int,
                        help=help%default,
                        default=default)

def _parser_add_time_traces(parser):
    parser.add_argument("-nt",'--no-time-trace', dest="plot_timedep", action='store_false',
                        help="Don't plot the time-traces of the contacts. Default is to plot them."
                       )
    parser.set_defaults(plot_timedep=True)

def _parser_add_savetrajs(parser):
    parser.add_argument("-st","--save-trajs", dest="savetrajs", action='store_true',
                         help="Save trajectory data, default is not to save it.")
    parser.set_defaults(savetrajs=False)
    pass


def _parser_add_distro(parser):
    parser.add_argument('-d', '--distribution', dest="distro", action='store_true',
                        help='Plot distance distributions instead of contact bar plots. Default is False.')

    parser.set_defaults(distro=False)

def _parser_add_smooth(parser):
    parser.add_argument("--n_smooth_hw", "-ns", type=int,
                        help="Number of frames one half of the averaging window for the time-traces. Default is 0, which means no averaging.",
                        default=0)

def _parser_add_scheme(parser):
    parser.add_argument("--scheme",type=str, default='closest-heavy',
                        help="Type of scheme for computing distance between residues. Choices are "
                             "{'ca', 'closest', 'closest-heavy', 'sidechain', 'sidechain-heavy'}. "
                             "See mdtraj documentation for more info")

def _parser_add_ylim_Ang(parser):
    parser.add_argument("--ylim_Ang",type=str, default="10",
                        help="Limit in Angstrom of the y-axis of the time-traces. Default is 10. "
                             "Switch to any other float or 'auto' for automatic scaling")

def _parser_add_t_unit(parser):
    parser.add_argument("--t_unit", type=str,
                        help="Unit used for the temporal axis, default is ns.",
                        default="ns")

def _parser_add_curve_color(parser):
    parser.add_argument("--curve_color", type=str,
                        help="Type of color used for the curves. Default is auto. Alternatives are 'P' or 'H'",
                        default="auto")

def _parser_add_background(parser):
    parser.add_argument('--background', "-bg",
                        type=_bool_or_color_like,
                        default=True,
                        help="Type of background when using smoothing windows."
                             " Default (True) is to use the unsmoothed curve's color. "
                             " A color string e.g. 'g' or 'red' or 'gray' also works, "
                             "as does an RGB string '0.5, 1., 0.5'.  "
                             "Use False for no color.")

def _parser_add_fragments(parser):
    parser.add_argument("-fr",'--fragments', default='lig_resSeq+', nargs='+',
                        help=("R|How to sub-divide the topology into fragments.\n"
                              "Several options possible. Taking the example sequence:\n"
                              "…-A27,Lig28,K29-…-W40,D45-…-W50,CYSP51,GDP52\n"
                              " - 'resSeq'\n"
                              "     breaks at jumps in resSeq entry:\n"
                              "     […A27,Lig28,K29,…,W40],[D45,…,W50,CYSP51,GDP52]\n"
                              " - 'resSeq+'\n"
                              "     breaks only at negative jumps in resSeq:\n"
                              "     […A27,Lig28,K29,…,W40,D45,…,W50,CYSP51,GDP52]\n"
                              " - 'bonds'\n"
                              "     breaks when AAs are not connected by bonds,\n"
                              "     ignores resSeq:\n"
                              "     […A27][Lig28],[K29,…,W40],[D45,…,W50],[CYSP51],[GDP52]\n"""
                              "     notice that because phosphorylated CYSP51 didn't get a\n"
                              "     bond in the topology, it's considered a ligand\n"
                              " - 'resSeq_bonds'\n"
                              "     breaks both at resSeq jumps or missing bond\n"
                              " - 'lig_resSeq+'\n"
                              "     Like resSeq+ but put's any non-AA residue into\n"
                              "     it's own fragment:\n"
                              "     […A27][Lig28],[K29,…,W40],[D45,…,W50,CYSP51],[GDP52]\n"
                              " - 'chains'\n"
                              "     breaks into chains of the PDB file/entry\n"
                              " -   None or 'None'\n"
                              "     all residues are in one fragment, fragment 0\n"                              
                              " - 'consensus'\n"
                              "     If any consensus nomenclature is provided,\n"
                              "     ask the user for definitions using\n"
                              "     consensus labels\n"
                              " - 0-10,15,14 20,21,30-50 51 (example, advanced users only)\n" 
                              "     Input arbitrary fragments via their\n"
                              "     residue serial indices (zero-indexed) using space as\n"
                              "     separator. Not recommended.\n"
                              " - 'None'\n"
                              "     All residues are in one fragment (fragment 0).\n"
                              "     Can be harmless or potentially dangerous if residue\n"
                              "     labels are repeated.\n"
                              "If you are unsure of any of these options, use \n"
                              "the command line tool mdc_fragments.py on \n"
                              "your topology file."))

def _parser_add_matrix(parser):
    parser.add_argument('--no-matrix', dest='contact_matrix', action='store_false',
                        help="Do not produce a plot of the interface contact matrix")
    parser.set_defaults(contact_matrix=True)

def _parser_add_flare(parser):
    parser.add_argument('--no-flare', dest='flareplot', action='store_false',
                        help="Do not produce a flare plot of the interface contact matrix. If produced, regardless of the '--graphic_ext', "
                             "the flareplot will always be in .pdf-format, unless '--graphic_ext' is 'svg'.")
    parser.set_defaults(flareplot=True)

def _parser_add_output_dir(parser):
    parser.add_argument('-od','--output_dir', type=str, help="directory to which the results are written. Default is '.'",
                        default='.')

def _parser_add_nomenclature(parser):
    parser.add_argument("-GPCR", "--GPCR_UniProt", type=str,
                        help="Look for GPCR consensus nomenclature, e.g Ballesteros-Weinstein, using this UniProt name, "
                             "e.g. adrb2_human. First, try locally with 'adrb2_human.xlsx' (or a full path to the file), "
                             "then do a web-lookup on the fly on the GPCRdb. See https://gpcrdb.org/services/ for more details."
                             "Default is None.",
                        default='None')
    parser.add_argument("-CGN","--CGN_UniProt", type=str,
                        help="Look for Common-G-protein-Nomenclature, CGN, using this UniProt name, "
                             "e.g. gnas2_human. First, try locally with 'gnas2_human.xlsx' (or a full path to the file), "
                             "then do a web-lookup on the fly on the GPCRdb. See https://gpcrdb.org/services/ for more details."
                             "Default is None.",
                        default='None')
    parser.add_argument("-KLIFS","--KLIFS_string", type=str,
                        help="Look for Kinase consensus nomenclature, KLIFS, using this string. "
                             "e.g. P31751. First, try locally with 'KLIFS_P31751.xlsx' (or a full path to the file), "
                             "then do a web-lookup on the fly on KLIFS. For web-lookups, the string has to be "
                             " formatted as key:value, eg. 'UniProtAC:P31751'. See the online documentation on "
                             "mdciao's LabelerKLIFS object and also https://klifs.net/ for more details.",
                        default="None",
                             )
    parser.add_argument("--save_nomenclature",dest='save_nomenclature_files',action="store_true",
                        help='Save available nomenclature definitions to disk so that they can '
                             'be accessed locally in later uses. '
                             'Default is False',
                        default=False)
    parser.set_defaults(save_nomenclature_files=False)

def _parser_add_graphic_ext(parser):
    parser.add_argument('-gx','--graphic_ext', type=str, help="Extension of the output graphics, default is .pdf",
                        default='.pdf')

def _parser_add_repframes(parser):
    parser.add_argument("--n_repframes", type=int, help="Number of representative frames to use when writing the"
                                                        "interface strength into a .pdb-file",
                        default=1)

def _parser_add_no_fragfrag(parser):
    parser.add_argument('-nsf', '--no-same_fragment',
                        help="Don't allow contact partners in the same fragment. Default is to allow it.",
                        dest='allow_same_fragment_ctcs', action='store_false')
    parser.set_defaults(allow_same_fragment_ctcs=True)

def _parser_add_pbc(parser):
    parser.add_argument('--no-pbc', dest='pbc',
                        help="Do not consider periodic boundary conditions when computing distances."
                             " Default is to consider them",
                        action='store_false')
    parser.set_defaults(pbc=True)

def _parser_add_short_AA_names(parser):
    parser.add_argument('-sa','--short_AAs', dest='short_AA_names', action='store_true',
                        help="Use one-letter aminoacid names when possible, e.g. K145 insted of Lys145."
                             " Default is False")
    parser.set_defaults(short_AA_names=False)

def _parser_add_output_desc(parser, default='output_sites'):
    parser.add_argument('-o','--output_desc', type=str, help="Descriptor for output files. Default is %s"%default,
                        default=default)
    return parser


def _parser_add_title(parser):
    parser.add_argument("-t", "--title", default=None, type=str,
                        help="Name of the system. Used for figure titles (not filenames)"
                             "Defaults to --output_desc if None is given")

def _parser_add_n_jobs(parser):
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of processors to use. "
                                                              "The parallelization is done over trajectories and "
                                                              "not over contacts, beyond n_jobs>n_trajs "
                                                              "parallelization will not have any effect.")

def _parser_add_sites(parser):
    parser.add_argument('--site_files', type=str, nargs='+',
                        help='site file(s) in json format containing site information. Check here for more info: http://proteinformatics.uni-leipzig.de/mdciao/api/generated/mdciao.sites.html')

def _parser_add_fragment_names(parser):
    parser.add_argument('--fragment_names', type=str,
                        help="Name of the fragments. Default is to name them automatically."
                             " Otherwise, give a quoted list of strings separated by commas, e.g. "
                             "'TM1, TM2, TM3,'"
                             "Use 'None' to avoid naming them altogether.",
                        default="auto")

def _int_or_float_type(val):
    r"""Do nothing or cast the str to int or float """
    if isinstance(val, str):
        if val.isdigit():
            val = int(val)
        else:
            val = float(val)

    return val

def _bool_or_color_like(val):
    r""" Turn strings into booleans, rgb floats keep color-like strings"""
    if str(val).lower()=="true":
        val = True
    elif str(val).lower()=="false":
        val = False
    elif len(str(val).strip(",").split(","))==3:
        val = [float(ii) for ii in str(val).strip(",").split(",")]
    if not isinstance(val,bool):
        assert _is_color_like(val), "The argument 'background' has to be boolean (True/False) or color-like, but '%s' (%s) is neither" % (
        val, type(val))
    return val

def _parser_add_ctc_control(parser, default=6):
    parser.add_argument("-cc", "--ctc_control", type=_int_or_float_type,
                        help="Control the number of reported contacts. "
                             "Can be an integer (keep the first n contacts) or "
                             "a float representing a fraction [0,1] of the total"
                             "number of contacts."
                             "Default is %u."%default,
                        default=default)

def _parser_add_pop(parser):
    parser.add_argument("--pop_N_ctcs", dest="separate_N_ctcs", action="store_true",
                        help="Separate the plot with the total number contacts from the time-trace plot. "
                             "Default is False")
    parser.set_defaults(separate_N_ctcs=False)

def _parser_add_n_cols(parser):
    parser.add_argument("--n_cols", type=int, help="number of columns of the overall plot. Default is 4",
                        default=4)

def _parser_add_graphic_dpi(parser):
    parser.add_argument('--graphic_dpi', type=int,
                        help="Dots per Inch (DPI) of the graphic output. Only has an effect for bitmap outputs. Default is 150.",
                        default=150)

def _parser_add_table_ext(parser):
    parser.add_argument('-tx','--table_ext', type=str,
                        help="Extension for tabled files (.dat, .txt, .xlsx, .ods). Default is '.dat'",
                        default="dat")

def _parser_add_write_to_disk(parser):
    parser.set_defaults(write_to_disk=False)
    parser.add_argument("--keep",
                        help="Save the consensus file locally for later use, default is False",
                        dest="write_to_disk", action="store_true"
                        )
def _parser_add_print_conlab(parser):
    parser.set_defaults(print_conlab=False)
    parser.add_argument("--verbose",
                        help="Print the consensus labels for all residues",
                        dest="print_conlab", action="store_true"
                        )
def _parser_add_fill_gaps(parser):
    parser.set_defaults(fill_gaps=False)
    parser.add_argument("--autofill",
                        help="Try to guess missing consensus labels",
                        dest="fill_gaps", action="store_true"
                        )
def _parser_add_AAs(parser):
    parser.add_argument("--AAs",type=str,
                        help="Print the idxs and labels of these AAs, e.g. R131,GLU30",
                        default=None)

def _parser_add_atomtypes(parser):
    parser.add_argument("-at", "--atomtypes",
                        dest="plot_atomtypes", action="store_true",
                        help="Add the atom-types to the frequency bars by 'hatching' them.\n"
                             " '--' is sidechain-sidechain\n"
                             " '|' is backbone-backbone\n"
                             " '\\' is backbone-sidechain\n"
                             " '/' is sidechain-backbone\n"
                             "Default is false")
    parser.set_defaults(accept_guess=False)

def _parser_add_conslabels(parser):
    parser.add_argument("--labels",type=str,
                        help="Print the idxs and resnames of these consensus labels, e.g. 3.50,2.63",
                        default=None)
def _parser_add_residues(parser):
    parser.add_argument('-r', '--residues', type=str,
                        help='The residues of interest, as coma-separated-values without spaces.\n'
                             'The input is very flexible and accepts\n'
                             'mixed descriptors and wildcards, eg: "GLU*,ARG*,GDP*,LEU394,380-385"\n'
                             "is a valid input. Numbers are interpeted as a residue's sequence number\n"
                             " (394 in LEU394), unless --serial_idxs is passed as an option.")

def _parser_add_no_frag(parser):
    parser.add_argument('-nf',"--no-fragments", dest='fragmentify',action='store_false',
                        help="Do not use fragments. Default is to use them")
    parser.set_defaults(fragmentify=True)

#TODO unify _ vs - in arguments
def _parser_add_frag_colors(parser):
    parser.add_argument("--fragment_colors", type=str,
                        help="comma-separated vales of the fragment colors.\n"
                             " If only one value, use that color for all fragments\n"
                             " Any matplotlib colors can be used. Default is 'tab:blue'\n"
                             " Why 'tab'? check https://matplotlib.org/3.1.1/tutorials/colors/colors.html !",
                        default="tab:blue")

def _parser_add_guess(parser):
    parser.add_argument("-ni", "-no-interactive",
                        dest="accept_guess", action="store_true",
                        help="Try not to be interactive. This can make wrong choices for the user, advanced only.")
    parser.set_defaults(accept_guess=False)

def _parser_add_naive_bonds(parser):
    parser.add_argument('--naive_bonds', "-nb", dest='naive_bonds', action='store_true',
                        help="Build naive, linear bonds between protein residues "
                             "of the same fragment if mdtraj can't build them automatically. "
                             "For more info check mdciao.utils.bonds.top2residue_bond_matrix_naive ")
    parser.set_defaults(naive_bonds=False)

# TODO group the parser better!
# TODO add short versions of the most frequent options
def parser_for_rn():
    parser = _parser_top_traj(description='Analyse residue neighborhoods using a distance cutoff. '
                                          'residue-residue contacts are reported in terms of their'
                                          'overall frequencies and time-traces. A number of files '
                                          'containing plots, tables and data will be generated.' )

    _parser_add_residues(parser)
    _parser_add_cutoff(parser)

    parser.add_argument('--serial_idxs', dest='res_idxs', action='store_true',
                        help='Interpret the indices of --residues '
                             'not as their sequence idxs (e.g. 30 for GLU30), but as '
                             'their serial order in the topology (e.g. 0 for GLU30 if '
                             'GLU30 is the first residue in the topology). Default is False')
    parser.set_defaults(res_idxs=False)

    _parser_add_stride(parser)
    _parser_add_ctc_control(parser)
    _parser_add_n_neighbors(parser)
    _parser_add_chunk(parser)
    _parser_add_smooth(parser)
    _parser_add_fragments(parser)
    _parser_add_fragment_names(parser)
    _parser_add_no_frag(parser)
    _parser_add_frag_colors(parser)
    parser.add_argument('--no-sort', dest='sort',
                        help="Don't sort the residues by their index. Default is to sort them.",
                        action='store_false')
    parser.set_defaults(sort=True)
    _parser_add_pbc(parser)
    _parser_add_table_ext(parser)
    _parser_add_graphic_ext(parser)
    _parser_add_nomenclature(parser)
    _parser_add_output_dir(parser)
    _parser_add_output_desc(parser, default='neighborhood')
    _parser_add_t_unit(parser)
    _parser_add_curve_color(parser)
    _parser_add_background(parser)
    _parser_add_graphic_dpi(parser)
    _parser_add_short_AA_names(parser)
    _parser_add_no_fragfrag(parser)
    _parser_add_time_traces(parser)
    _parser_add_savetrajs(parser)
    _parser_add_distro(parser)
    _parser_add_n_cols(parser)
    _parser_add_n_jobs(parser)
    _parser_add_pop(parser)
    _parser_add_ylim_Ang(parser)
    _parser_add_guess(parser)
    _parser_add_switch(parser)
    _parser_add_atomtypes(parser)
    _parser_add_naive_bonds(parser)
    _parser_add_scheme(parser)
    return parser

def _parser_add_switch(parser):
    parser.add_argument("-s","--switch_off_Ang",
                        default=None,
                        type=float,
                        help="Use a linear switchoff instead of a crisp one. Deafault is None")
def parser_for_dih():
    parser = _parser_top_traj(description='Small analysis tool for computation of residue dihedrals, backbone and sidechains.')


    parser.add_argument('--resSeq_idxs', type=str,
                        help='the resSeq idxs of interest (in VMD these are called "resid"). '
                             'Can be in a format 1,2-6,10,20-25. No spaces are allowed.')
    parser.add_argument('--types',type=str,default='all', help='Types of dihedral angles to be computed. It can be '
                                                               'an "all", "backbone", or "sidechain"')
    _parser_add_stride(parser)
    _parser_add_chunk(parser)
    _parser_add_smooth(parser)
    #_parser_add_fragments(parser)
    _parser_add_fragment_names(parser)

    parser.add_argument('--sort', dest='sort', action='store_true', help="Sort the resSeq_idxs list. Default is True")
    parser.add_argument('--no-sort', dest='sort', action='store_false')
    parser.set_defaults(sort=True)

    #parser.add_argument('--pbc', dest='pbc', action='store_true',
    #                    help="Consider periodic boundary conditions when computing distances."
    #                         " Default is True")
    #parser.add_argument('--no-pbc', dest='pbc', action='store_false')
    #parser.set_defaults(pbc=True)

    parser.add_argument('--ask_fragment', dest='ask', action='store_true',
                        help="Interactively ask for fragment assignemnt when input matches more than one resSeq")
    parser.add_argument('--no-ask_fragment', dest='ask', action='store_false')
    parser.set_defaults(ask=True)
    parser.add_argument('--output_npy', type=str, help="Name of the output.npy file for storing this runs' results",
                        default='output.npy')
    _parser_add_table_ext(parser)
    parser.add_argument('--graphic_ext', type=str, help="Extension of the output graphics, default is .pdf",
                        default='.pdf')

    parser.add_argument('--serial_idxs', dest='res_idxs', action='store_true',
                        help='Interpret the indices of --resSeq_idxs '
                             'not as sequence idxs (e.g. 30 for GLU30), but as '
                             'their order in the topology (e.g. 0 for GLU30 if '
                             'GLU30 is the first residue in the topology). Default is False')
    parser.set_defaults(res_idxs=False)

    _parser_add_nomenclature(parser)
    _parser_add_output_dir(parser)
    _parser_add_output_desc(parser, default='dih')
    _parser_add_t_unit(parser)
    _parser_add_curve_color(parser)
    _parser_add_background(parser)
    _parser_add_graphic_dpi(parser)
    _parser_add_short_AA_names(parser)
    _parser_add_time_traces(parser)
    _parser_add_n_cols(parser)
    _parser_add_n_jobs(parser)

    parser.add_argument('--degrees',   dest='use_deg', action="store_true", help='Use degrees (default) or radians')
    parser.add_argument('--no-degrees',dest='use_deg', action="store_false")
    parser.set_defaults(use_deg=True)

    parser.add_argument('--cos',   dest='use_cos', action="store_true", help="Use the cosine of the angle instead of the angle. Default is not to use cosine")
    parser.add_argument('--no-cos',dest='use_cos', action="store_false")
    parser.set_defaults(use_cos=True)

    return parser

def parser_for_sites():
    parser = _parser_top_traj(description='Analyse a specific set of residue-residue contacts using a distance cutoff. '
                                          'The user has to provide one or more "site" files in a .json format')

    _parser_add_sites(parser)
    parser.add_argument('--default_fragment_index', default=None, type=int,
                        help="In case a residue identified as, e.g, GLU30, appears more than\n"
                             " one time in the topology, e.g. in case of a dimer, the user can\n"
                             " pass which fragment/monomer should be chosen by default. The\n"
                             " default behaviour (None) will prompt the user when necessary")
    _parser_add_scheme(parser)
    _parser_add_nomenclature(parser)
    _parser_add_output_dir(parser)
    _parser_add_output_desc(parser,default="sites")
    _parser_add_stride(parser)
    _parser_add_smooth(parser)
    _parser_add_no_frag(parser)
    #_parser_add_fragment_names(parser)
    _parser_add_cutoff(parser)
    _parser_add_pbc(parser)
    _parser_add_t_unit(parser)
    _parser_add_graphic_ext(parser)
    _parser_add_curve_color(parser)
    _parser_add_background(parser)
    _parser_add_graphic_dpi(parser)
    _parser_add_ylim_Ang(parser)
    _parser_add_short_AA_names(parser)
    _parser_add_n_jobs(parser)
    _parser_add_table_ext(parser)
    _parser_add_atomtypes(parser)
    _parser_add_guess(parser)
    _parser_add_distro(parser)
    _parser_add_savetrajs(parser)
    return parser

def parser_for_densities():
    parser = _parser_top_traj("For densities")
    _parser_add_sites(parser)
    _parser_add_output_dir(parser)
    _parser_add_stride(parser,help='Stride down the data by this factor. Default is 1.')

    return parser

def parser_for_interface():
    parser = _parser_top_traj(description='Analyse interfaces between any two groups of residues using a distance cutoff. '           
                                          'To help in the identification of these two groups of residues, '
                                          'the peptide-chain in the input topology '
                                          'can be automatically broken down into fragments and use them as input. '
                                          'The number of shown contacts depends on the parameters "ctc_control" and '
                                          '"min_freq". ')

    _parser_add_fragments(parser)
    parser.add_argument("-isel1","--interface_selection_1", type=str,
                        help="Indices of the fragments that belong to the one side of the interface, as CSVs or range, e.g. '1,3-4'. "
                             "Defaults to None which will prompt the user of information, except when "
                             "only two fragments are present. Then it defaults to [0]", default=None)
    parser.add_argument("-isel2","--interface_selection_2", type=str,
                        help="Indices of the fragments that belong to the other side of the interface, as CSVs or range, e.g. '1,3-4'. "
                             "Defaults to None which will prompt the user of information, except when "
                             "only two fragments are present. Then it defaults to [1]", default=None)
    _parser_add_cutoff(parser)
    _parser_add_ctc_control(parser, default=50)
    parser.add_argument("-mf", "--min_freq", type=float, default=.05,
                        help="Do not show frequencies smaller than this. Default is 0.05. If you notice the output being "
                             "truncated at values much larger than this, but suspect that some contacts "
                             "are not being reported, increase the 'ctc_control' parameter")
    parser.add_argument('--cmap',type=str,help="The colormap for the contact matrix. Default is 'binary' which is "
                                                 "black and white, but you can choose anthing from here: "
                                                 "https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html",
                        default="binary")
    _parser_add_n_neighbors(parser, default=0)
    _parser_add_stride(parser)
    _parser_add_smooth(parser)
    _parser_add_time_traces(parser)
    _parser_add_savetrajs(parser)
    _parser_add_n_jobs(parser)
    _parser_add_fragment_names(parser)
    _parser_add_no_frag(parser)
    _parser_add_pbc(parser)
    _parser_add_nomenclature(parser)
    _parser_add_chunk(parser)
    _parser_add_output_desc(parser,'interface')
    _parser_add_output_dir(parser)
    _parser_add_graphic_ext(parser)
    _parser_add_graphic_dpi(parser)
    _parser_add_curve_color(parser)
    _parser_add_t_unit(parser)
    _parser_add_background(parser)
    _parser_add_short_AA_names(parser)
    parser.add_argument('--no-sort_by_av_ctcs', dest='sort_by_av_ctcs', action='store_false',
                        help="When presenting the results summarized by residue, "
                             " don't sort by sum of frequencies (~average number of contacts), "
                             "but by ascending order whithin each interface member."
                             " Default is to sort them by frequencies."
                        )
    parser.set_defaults(sort_by_av_ctcs=True)
    _parser_add_scheme(parser)
    _parser_add_flare(parser)
    _parser_add_matrix(parser)
    _parser_add_pop(parser)
    _parser_add_guess(parser)
    _parser_add_title(parser)
    _parser_add_repframes(parser)
    return parser

def parser_for_frag_overview():
    parser = argparse.ArgumentParser(description='Break a molecular topology into fragments using different heuristics.')
    parser.add_argument('-m','--methods', help='What methods to test. '
                                          'Default is all.',
                        nargs='+',
                        type=str,
                        default=['all']
                        )
    _parser_add_topology(parser)
    _parser_add_AAs(parser)
    return parser


def parser_for_GPCR_overview():
    parser = argparse.ArgumentParser(description='Provide an overview of GPCR consensus nomenclature,'
                                                 ' optionally mapping it on an input topology. '
                                                 'The nomenclature can be read locally or over the network.')

    parser.add_argument("GPCR_UniProt_or_file", type=str,
                        help="Get GPCR definitions from here.\n"
                             "If a file is not found locally, check the GPCRdb\n"
                             "using this string as uniprot code e.g. adrb2_human. "
                             "See https://gpcrdb.org/services/ for more details.",
                        action=Populate_input_Action
                        )
    parser.add_argument("-t", '--topology', type=str, help='Topology file', default=None)

    _parser_add_write_to_disk(parser)
    _parser_add_print_conlab(parser)
    _parser_add_fill_gaps(parser)
    _parser_add_AAs(parser)
    _parser_add_conslabels(parser)

    return parser

def parser_for_pdb():
    parser = argparse.ArgumentParser(description='Lookup a four-letter PDB-code in the RCSB PDB and save it locally.')

    parser.add_argument("code", type=str,
                        help="four-letter PDB code"
                        )
    parser.add_argument("--ext","-x", type=str, default=".pdb",
                        help="extension of file to be saved. Default is '.pdb'")
    parser.add_argument("--output","-o", type=str, default=None,
                        help="Filename (with extension) to save to. If parsed, "
                             "'--ext' will be ignored. Default is to save '$code.pdb'")
    return parser


def parser_for_CGN_overview():
    parser = argparse.ArgumentParser(description="Provide an overview of a G-alpha Numbering (CGN)-type nomenclature,"
                                                 " optionally mapping it on an input topology. "
                                                 "The nomenclature can be read locally or over the network")

    parser.add_argument("CGN_UniProt_or_file", type=str,
                        help="Get CGN nomenclature definitions from here. \n"
                             "If a file is not found locally, check the GPCRdb\n"
                             "using this string as uniprot code e.g. gnas2_human. "
                             "See https://gpcrdb.org/services/ for more details.",
                        action=Populate_input_Action
                        )
    parser.add_argument("-t", '--topology', type=str, help='Topology file', default=None)

    _parser_add_write_to_disk(parser)
    _parser_add_print_conlab(parser)
    _parser_add_fill_gaps(parser)
    _parser_add_AAs(parser)
    _parser_add_conslabels(parser)

    return parser

def parser_for_KLIFS_overview():
    parser = argparse.ArgumentParser(description="Produce an overview of a the Kinase 85 pocket-residues nomenclature,"
                                                 "optionally mapping it on an input topology. "
                                                 "This nomenclature can be read locally or over the network from the "
                                                 "KLIFS database." )

    parser.add_argument("KLIFS_string_or_excelfile", type=str,
                        help="Get KLIFS definitions from here. A string,  "
                             "e.g. P31751 to construct a path to a an Excel File 'KLIFS_P31751.xlsx', "
                             "or an Excel File directly. If nothing is found locally, there will be a web-lookup "
                             " For web-lookups, the string has to be formatted as key:value, eg. 'UniProtAC:P31751'. "
                             "See the online documentation on mdciao's LabelerKLIFS object "
                             "and also https://klifs.net/ for more details.",
                        action=Populate_input_Action,
                             )
    parser.add_argument("-t", '--topology', type=str, help='Topology file', default=None)

    _parser_add_write_to_disk(parser)
    _parser_add_print_conlab(parser)
    #_parser_add_fill_gaps(parser)
    _parser_add_AAs(parser)
    _parser_add_conslabels(parser)

    return parser

def parser_for_compare_neighborhoods():
    parser = argparse.ArgumentParser(description="Compare residue-residue contact frequencies "
                                                 "from different files by generating a comparison plot and table",
                                     formatter_class=SmartFormatter
                                     )
    parser.add_argument("files", type=str, nargs="+",
                        help="R|Files (ASCII or .xlsx) containing the frequencies \n"
                             "and labels in the first two or three columns. Minimal examples \n"
                             "could be :\n"
                             ">>> #freq     label\n"
                             ">>> 1.        GLU30-ALA50\n"
                             ">>> .75       ALA50-ASP40\n"
                             "or\n"
                             ">>> 1.        GLU30 ALA50\n"
                             ">>> .75       ALA50 ASP40\n"
                             "Lines startint with '#' are ignored")
    parser.add_argument("-a","--anchor",type=str,default=None,
                        help="A residue that appears in all contacts. "
                             "It will be eliminated from the labels for clarity.")
    parser.add_argument("-k","--keys", type=str,default=None,
                        help="The keys used to label the files, e.g. 'WT,MUT'")
    parser.add_argument("-c","--colors", type=str, default=_colorstring,
                        help='Colors to use for the dicts, defaults to "%s"'%", ".join(_colorstring.split(",")))
    parser.add_argument("-m","--mutations",type=str, default=None,
                        help='A replacement dictionary, to be able to re-label '
                             'residues across systems, e.g. "GLU:ARG,LYS:PHE" changes '
                             'all GLUs to ARGs and all LYS to PHEs')
    parser.add_argument("-t", "--title", type=str, default='comparison',
                        help='Title of the plot. Default is "comparison"')
    parser.add_argument("-p","--pop-up", dest="pop",
                        help="pop-up an interactive figure before closing. "
                             "Default is not to pop-up but directly save to file",
                        action="store_true",
                        )
    parser.set_defaults(pop=False)
    _parser_add_output_desc(parser,"freq_comparison")
    _parser_add_graphic_ext(parser)
    return parser

def parser_for_examples():
    desc1 = "Wrapper script to showcase and optionally run examples of the\n" \
            "command-line-tools that ship with mdciao.\n"
    ex = _xCLT()
    epilogue = "Available command line tools are:\n"
    epilogue += "\n".join([" * %s.py"%key for key in ex.clts])
    epilogue += "\n\nYou can type for example:\n > mdc_interface.py  -h\n" \
                " to view the command's documentation or\n" \
                " > mdc_examples.py interf\n" \
                " to show and/or run an example of that command "
    parser = argparse.ArgumentParser(description=desc1, epilog=epilogue,
                                     formatter_class=argparse.RawDescriptionHelpFormatter
                                     )
    parser.add_argument("clt",
                        default=None,
                        type=str,
                        help="The command-line-tool (also known as script) to run. Can be part of the name as well\n "
                             "e.g. 'neigh' for mdc_neighborhoods.py")

    parser.add_argument("-x",
                        action="store_true",
                        dest="execute",
                        help="whether to execute the command after showing it. "
                             "Default is False.")
    parser.add_argument("--short_options","-so",help="Use the short option flags rather than the long ones",
                        action="store_true",
                        dest="short")
    parser.set_defaults(execute=False)

    parser.set_defaults(show_list=False)
    parser.set_defaults(short=False)

    return parser

def parser_for_residues():
    parser = argparse.ArgumentParser(description="Find residues in an input topology using Unix filename pattern matching\n"
                                                 "like in an 'ls' Unix operation.",
                                     formatter_class=SmartFormatter)
    parser.add_argument("residues",type=str,help="Unix-like expressions and ranges are allowed, e.g.\n"
                                                 "'GLU,PH*,380-394,3.50,GH.5*.', as are consensus descriptors\n"
                                                 "if consensus labels are provided")
    _parser_add_nomenclature(parser)
    _parser_add_topology(parser)
    _parser_add_guess(parser)
    _parser_add_fragments(parser) #todo check why the formatting is off here but not in parser for rn
    return parser


def parser_for_notebooks():
    parser = argparse.ArgumentParser(
        description='Copy the example Jupyter notebooks distributed with mdciao into a directory "mdciao_notebooks" placed the current working directory.'
                    'The method never overwrites an existing folder, but keeps either asking or producing new folder names.')
    return parser

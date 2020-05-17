import argparse

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
    parser.add_argument("--ctc_cutoff_Ang", type=float,
                        help="The cutoff distance between two residues for them to be considered in contact. Default is 3 Angstrom.",
                        default=3)

def _parser_add_n_neighbors(parser, default=4):
    parser.add_argument("--n_nearest", type=int,
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
                      default=10000):
    parser.add_argument("--chunksize_in_frames", type=int,
                        help=help%default,
                        default=default)

def _parser_add_time_traces(parser):
    parser.add_argument('--time-trace', dest="plot_timedep", action='store_true',
                        help='Plot time-traces of the contacts, default is True')
    parser.add_argument('--no-time-trace', dest="plot_timedep", action='store_false',
                       )
    parser.set_defaults(plot_timedep=True)

def _parser_add_distro(parser):
    parser.add_argument('--distribution', dest="distro", action='store_true',
                        help='Plot distance distributions instead of contact bar plots. Default is False.')
    parser.add_argument('--no-distribution', dest="distro", action='store_false',
                       )
    parser.set_defaults(distro=False)

def _parser_add_smooth(parser):
    parser.add_argument("--n_smooth_hw", type=int,
                        help="Number of frames one half of the averaging window for the time-traces. Default is 0, which means no averaging.",
                        default=0)

def _parser_add_scheme(parser):
    parser.add_argument("--scheme",type=str, default='closest-heavy',
                        help="Type for scheme for computing distance between residues. Choices are "
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
                        help="Type of color used for the curves. Default is auto. Alternatives are 'Peter' or 'Hobat'",
                        default="auto")

def _parser_add_gray_backgroud(parser):
    parser.add_argument('--gray-background', dest='gray_background', action='store_true',
                        help="Use gray background when using smoothing windows"
                             " Defaut is False")
    parser.add_argument('--no-gray-background', dest='gray_background', action='store_false')
    parser.set_defaults(gray_background=False)

def _parser_add_fragments(parser):
    parser.add_argument('--fragments', dest='fragmentify', action='store_true',
                        help="Auto-detect fragments (i.e. breaks) in the peptide-chain. Default is true.")
    parser.add_argument('--no-fragments', dest='fragmentify', action='store_false')
    parser.set_defaults(fragmentify=True)

def _parser_add_output_dir(parser):
    parser.add_argument('--output_dir', type=str, help="directory to which the results are written. Default is '.'",
                        default='.')

def _parser_add_nomenclature(parser):
    parser.add_argument("--BW_uniprot", type=str,
                        help="Look for Ballesteros-Weinstein definitions in the GPRCmd using a uniprot code, "
                             "e.g. adrb2_human. See https://gpcrdb.org/services/ for more details."
                             "Default is None.",
                        default='None')
    parser.add_argument("--BW_write",dest='write_to_disk_BW',action="store_true",
                        help='Write the BW definitions to disk so that it can be acessed locally in later uses. '
                             'Default is False',
                        default=False)
    parser.add_argument("--no-BW_write", dest='write_to_disk_BW', action="store_false",
                        default=True)
    parser.set_defaults(write_to_disk_BW=False)
    parser.add_argument("--CGN_PDB", type=str, help="PDB code for a consensus G-protein nomenclature", default='None')

def _parser_add_graphic_ext(parser):
    parser.add_argument('--graphic_ext', type=str, help="Extension of the output graphics, default is .pdf",
                        default='.pdf')

def _parser_add_no_fragfrag(parser):
    parser.add_argument('--same_fragment', dest='same_fragment', action='store_true',
                        help="Allow contact partners in the same fragment, default is True"
                             " Defaut is True")
    parser.add_argument('--no-same_fragment', dest='same_fragment', action='store_false')
    parser.set_defaults(allow_same_fragment_ctcs=True)

def _parser_add_pbc(parser):
    parser.add_argument('--pbc', dest='pbc', action='store_true',
                        help="Consider periodic boundary conditions when computing distances."
                             " Defaut is True")
    parser.add_argument('--no-pbc', dest='pbc', action='store_false')
    parser.set_defaults(pbc=True)

def _parser_add_short_AA_names(parser):
    parser.add_argument('--short_AAs', dest='short_AA_names', action='store_true',
                        help="Use one-letter aminoacid names when possible, e.g. K145 insted of Lys145."
                             " Defaut is False")
    parser.add_argument('--no-short_AAs', dest='short_AA_names', action='store_false')
    parser.set_defaults(short_AA_names=False)

def _parser_add_output_desc(parser, default='output_sites'):
    parser.add_argument('--output_desc', type=str, help="Descriptor for output files. Default is %s"%default,
                        default=default)
    return parser

def _parser_add_n_jobs(parser):
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of processors to use. "
                                                              "The parallelization is done over trajectories and "
                                                              "not over contacts, beyond n_jobs>n_trajs "
                                                              "parallelization will not have any effect.")

def _parser_add_sites(parser):
    parser.add_argument('--site_files', type=str, nargs='+',
                        help='site file(s) in json format containing site information, i.e., which bonds correspond to each site')

def _parser_add_fragment_names(parser):
    parser.add_argument('--names', type=str,
                        help="Name of the fragments. Leave empty if you want them automatically named."
                             " Otherwise, give a quoted list of strings separated by commas, e.g. "
                             "'TM1, TM2, TM3,'",
                        default="")

def _parser_add_n_ctcs(parser, default=5):
    parser.add_argument("--n_ctcs", type=int,
                        help="Only the first n_ctcs most-frequent contacts "
                             "will be written to the ouput. Default is %u."%default,
                        default=default)

def _parser_add_pop(parser):
    parser.add_argument("--pop_N_ctcs", dest="separate_N_ctcs", action="store_true",
                        help="Separate the plot with the total number contacts from the time-trace plot. "
                             "Default is False")
    parser.add_argument("--no-pop_N_ctcs", dest="separate_N_ctcs", action="store_false")
    parser.set_defaults(separate_N_ctcs=False)

def _parser_add_n_cols(parser):
    parser.add_argument("--n_cols", type=int, help="number of columns of the overall plot. Default is 4",
                        default=4)

def _parser_add_graphic_dpi(parser):
    parser.add_argument('--graphic_dpi', type=int,
                        help="Dots per Inch (DPI) of the graphic output. Only has an effect for bitmap outputs. Default is 150.",
                        default=150)

def _parser_add_table_ext(parser):
    parser.add_argument('--table_ext', type=str,
                        help="Extension for tabled files (.dat, .txt, .xlsx). Default is 'none', which does not write anything.",
                        default=None)

def parser_for_rn():
    parser = _parser_top_traj(description='Small residue-residue contact analysis tool, initially developed for the '
                                      'receptor-G-protein complex.')


    parser.add_argument('--resSeq_idxs', type=str,
                        help='the resSeq idxs of interest (in VMD these are called "resid"). '
                             'Can be in a format 1,2-6,10,20-25. No spaces are allowed.')

    _parser_add_cutoff(parser)
    _parser_add_stride(parser)
    _parser_add_n_ctcs(parser)
    _parser_add_n_neighbors(parser)
    _parser_add_chunk(parser)
    _parser_add_smooth(parser)
    parser.add_argument("--nlist_cutoff_Ang", type=float,
                        help="Cutoff for the initial neighborlist. Only atoms that are within this distance in the original reference "
                             "(the topology file) are considered potential neighbors of the residues in resSeq_idxs, s.t. "
                             "non-necessary distances (e.g. between N-terminus and G-protein) are not even computed. "
                             "Default is 15 Angstrom.", default=15)
    _parser_add_fragments(parser)
    _parser_add_fragment_names(parser)

    parser.add_argument('--sort', dest='sort', action='store_true', help="Sort the resSeq_idxs list. Defaut is True")
    parser.add_argument('--no-sort', dest='sort', action='store_false')
    parser.set_defaults(sort=True)

    parser.add_argument('--pbc', dest='pbc', action='store_true',
                        help="Consider periodic boundary conditions when computing distances."
                             " Defaut is True")
    parser.add_argument('--no-pbc', dest='pbc', action='store_false')
    parser.set_defaults(pbc=True)

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
    _parser_add_output_desc(parser, default='neighborhoods')
    _parser_add_t_unit(parser)
    _parser_add_curve_color(parser)
    _parser_add_gray_backgroud(parser)
    _parser_add_graphic_dpi(parser)
    _parser_add_short_AA_names(parser)
    _parser_add_no_fragfrag(parser)
    _parser_add_time_traces(parser)
    _parser_add_distro(parser)
    _parser_add_n_cols(parser)
    _parser_add_n_jobs(parser)
    _parser_add_pop(parser)
    _parser_add_ylim_Ang(parser)
    return parser

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
    _parser_add_fragments(parser)
    _parser_add_fragment_names(parser)

    parser.add_argument('--sort', dest='sort', action='store_true', help="Sort the resSeq_idxs list. Defaut is True")
    parser.add_argument('--no-sort', dest='sort', action='store_false')
    parser.set_defaults(sort=True)

    #parser.add_argument('--pbc', dest='pbc', action='store_true',
    #                    help="Consider periodic boundary conditions when computing distances."
    #                         " Defaut is True")
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
    _parser_add_gray_backgroud(parser)
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
    #todo THIS WILL BREARK PARSER FOR SITES!!!!
    parser = _parser_top_traj(description='Small residue-residue contact analysis tool, initially developed for the '
                                      'receptor-G-protein complex. The user has to provide "site" files in .json format')

    _parser_add_sites(parser)
    parser.add_argument('--default_fragment_index', default=None, type=int,
                        help="In case a residue identified as, e.g, GLU30, appears more than\n"
                             " one time in the topology, e.g. in case of a dimer, the user can\n"
                             " pass which fragment/monomer should be chosen by default. The\n"
                             " default behaviour (None) will prompt the user when necessary")
    _parser_add_scheme(parser)
    _parser_add_nomenclature(parser)
    _parser_add_output_dir(parser)
    _parser_add_stride(parser)
    _parser_add_smooth(parser)
    _parser_add_fragment_names(parser)
    _parser_add_cutoff(parser)
    _parser_add_t_unit(parser)
    _parser_add_graphic_ext(parser)
    _parser_add_curve_color(parser)
    _parser_add_gray_backgroud(parser)
    _parser_add_graphic_dpi(parser)
    _parser_add_ylim_Ang(parser)
    _parser_add_n_jobs(parser)
    return parser

def parser_for_densities():
    parser = _parser_top_traj("For densities")
    _parser_add_sites(parser)
    _parser_add_output_dir(parser)
    _parser_add_stride(parser,help='Stride down the data by this factor. Default is 1.')

    return parser

def parser_for_interface():
    parser = _parser_top_traj(description='Residue-residue contact analysis-tool where contacts are computed '
                                          ' between two groups of residues specified by the user.'                                          
                                          ' To help in the identification of these two groups of residues, '
                                          'the peptide-chain in the input topology '
                                          'can be automatically broken down into fragments and use them directly.')

    parser.add_argument('--fragments', default=['resSeq'], nargs='+',
                        help=("R|How to sub-divide the topology into fragments.\n"
                              "Several options possible. Taking the example sequence:\n"
                              "...-A27,Lig28,K29-...-W40,D45-...-W50,GDP1\n"
                              " - 'resSeq'\n"
                              "     breaks at jumps in resSeq entry:\n"
                              "     [...A27,Lig28,K29,...,W40],[D45,...,W50],[GDP1]\n"
                              " - 'resSeq+'\n"
                              "     breaks only at negative jumps in resSeq:\n"
                              "     [...A27,Lig28,K29,...,W40,D45,...,W50],[GDP1]\n"
                              " - 'bonds'\n"
                              "     breaks when AAs are not connected by bonds,\n"
                              "     ignores resSeq:\n"
                              "     [...A27][Lig28],[K29,...,W40],[D45,...,W50],[GDP1]\n"
                              " - 'resSeq_bonds'\n"
                              "     breaks both at resSeq jumps or missing bond\n"
                              " - 'chains'\n"
                              "     breaks into chains of the PDB file/entry\n"
                              " - 'consensus'\n"
                              "     If any consensus nomenclature is provided,\n"
                              "     ask the user for definitions using\n"
                              "     consensus labels\n"
                              " - 0-10,15,14 20,21,30-50 51 (example, advanced users only)\n" 
                              "     Input arbitary fragments via their\n"
                              "     residue serial indices (zero-indexed) using space as\n"
                              "     separator. Not recommended\n."
                              "If you are unsure of any of these options, use \n"
                              "the command line tool fragment_overview.py on \n"
                              "your topology file."))

    parser.add_argument("--frag_idxs_group_1", type=str,
                        help="Indices of the fragments that belong to the group_1. "
                             "Defaults to None which will prompt the user of information, except when "
                             "only two fragments are present. Then it defaults to [0]", default=None)
    parser.add_argument("--frag_idxs_group_2", type=str,
                        help="Indices of the fragments that belong to the group_2. "
                             "Defaults to None which will prompt the user of information, except when "
                             "only two fragments are present. Then it defaults to [1]", default=None)
    _parser_add_cutoff(parser)
    _parser_add_n_ctcs(parser, default=10)
    parser.add_argument("--interface_cutoff_Ang", type=float,
                        help="The is_interface between both groups is defined as the set of group_1-group_2-"
                             "distances that are within this "
                             "cutoff in the reference topology. Otherwise, a large number of "
                             "non-necessary distances (e.g. between N-terminus and G-protein) are computed. Default is 35.",
                        default=35)
    _parser_add_n_neighbors(parser, default=0)
    _parser_add_stride(parser)
    _parser_add_smooth(parser)
    _parser_add_time_traces(parser)
    _parser_add_n_jobs(parser)
    #_parser_add_fragment_names(parser)

    #parser.add_argument('--consolidate', dest='consolidate_opt', action='store_true',
    #                    help="Treat all trajectories as fragments of one single trajectory. Default is True")
    #parser.add_argument('--dont_consolidate', dest='consolidate_opt', action='store_false')
    #parser.set_defaults(consolidate_opt=True)
    _parser_add_nomenclature(parser)
    _parser_add_chunk(parser)
    _parser_add_output_desc(parser,'is_interface')
    _parser_add_output_dir(parser)
    _parser_add_graphic_ext(parser)
    _parser_add_graphic_dpi(parser)
    #_parser_add_ascii(parser)
    _parser_add_curve_color(parser)
    _parser_add_t_unit(parser)
    _parser_add_gray_backgroud(parser)
    _parser_add_short_AA_names(parser)
    parser.add_argument('--sort_by_av_ctcs', dest='sort_by_av_ctcs', action='store_true',
                        help="When presenting the results summarized by residue, "
                             " sort by sum of frequencies (~average number of contacts)."
                             " Defaut is True.")
    parser.add_argument('--no-sort_by_av_ctcs', dest='sort_by_av_ctcs', action='store_false')
    parser.set_defaults(sort_by_av_ctcs=True)
    _parser_add_scheme(parser)
    _parser_add_pop(parser)

    return parser

def parser_for_contact_map():
    parser = _parser_top_traj(description='Residue-residue contact-maps')

    parser.add_argument("--list_ctc_cutoff_Ang", type=str,
                        help="The cutoff distances between two residues for them to be considered in contact.",
                        default='3')
    _parser_add_stride(parser)
    _parser_add_n_jobs(parser)
    _parser_add_chunk(parser, default=100)
    _parser_add_output_desc(parser,'is_interface')
    _parser_add_output_dir(parser)
    _parser_add_graphic_ext(parser)
    _parser_add_graphic_dpi(parser)
    _parser_add_scheme(parser)

    return parser

def parser_for_frag_overview():
    parser = argparse.ArgumentParser(description='Provides overview of '
                                                 'fragments by different methods')
    parser.add_argument('--methods', help='What methods to test. '
                                          'Default is all.',
                        nargs='+',
                        type=str,
                        default=['all']
                        )
    _parser_add_topology(parser)
    return parser

def parser_for_BW_overview():
    parser = argparse.ArgumentParser(description='Provides overview of '
                                                 'BW nomenclature for a given topology')

    _parser_add_topology(parser)
    parser.add_argument("BW_uniprot_or_file", type=str,
                        help="Get Ballesteros-Weinstein definitions from here.\n"
                             "If a file is not found locally, look for\n"
                             " Ballesteros-Weinstein definitions in the GPRCmd\n"
                             "using this string as uniprot code, "
                             "e.g. adrb2_human. See https://gpcrdb.org/services/ for more details."
                        )

    parser.add_argument("--keep",
                        help="Save the BW locally for later use, default is False",
                        dest="write_to_disk", action="store_true"
                        )

    parser.set_defaults(write_to_disk=False)

    parser.add_argument("--verbose",
                        help="Print the consensus labels for all residues",
                        dest="print_conlab", action="store_true"
                        )
    parser.set_defaults(print_conlab=False)


    parser.add_argument("--autofill",
                        help="Try to guess missing consensus labels",
                        dest="fill_gaps", action="store_true"
                        )
    parser.set_defaults(fill_gaps=False)


    return parser

def parser_for_CGN_overview():
    parser = argparse.ArgumentParser(description='Provides overview of '
                                                 'CGN nomenclature for a given topology')

    _parser_add_topology(parser)
    parser.add_argument("CGN_PDB", type=str,
                        help="Look for CGN definitions in a database using a PDB code, "
                             "e.g. 3SN6. see www.mrc-lmb.cam.ac.uk")
    return parser

def parser_for_compare_neighborhoods():
    """
    width=.2, figsize=(10, 5),
                          fontsize=16,
                          substitutions=["MG", "GDP"],
                          mutations = {},
                          plot_singles=False, stop_at=.1, scale_fig=False
    :return:
    """
    parser = argparse.ArgumentParser(description="compare")
    parser.add_argument("anchor",type=str)
    parser.add_argument("--keys", type=str)
    parser.add_argument("--files", type=str)
    parser.add_argument("--colors", type=str)
    parser.add_argument("--mutations",type=str)
    #parser.add_argument("-width",type=float,default=.2)
    return parser
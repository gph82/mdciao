import argparse

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
    parser = argparse.ArgumentParser(description=description)
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

def _parser_add_stride(parser,
                       help="Stride down the input trajectoy files by this factor. Default is 1." ):
    parser.add_argument("--stride", type=int,
                        help=help, default=1)

def _parser_add_chunk(parser,help="Trajectories are read in chunks of this size "
                                  "(helps with big files and memory problems). Default is 10000",
                      default=10000):
    parser.add_argument("--chunksize_in_frames", type=int,
                        help=help,
                        default=default)

def _parser_add_smooth(parser):
    parser.add_argument("--n_smooth_hw", type=int,
                        help="Number of frames one half of the averaging window for the time-traces. Default is 0, which means no averaging.",
                        default=0)

def _parser_add_t_unit(parser):
    parser.add_argument("--t_unit", type=str,
                        help="Unit used for the temporal axis, default is ns.",
                        default="ns")

def _parser_add_curve_colort(parser):
    parser.add_argument("--curve_color", type=str,
                        help="Type of color used for the curves. Default is auto. Alternative is 'Peter'.",
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
    parser.add_argument("--BW_file", type=str,
                        help="Json file with info about the Ballesteros-Weinstein definitions as downloaded from the GPRCmd",
                        default='None')
    parser.add_argument("--CGN_PDB", type=str, help="PDB code for a consensus G-protein nomenclature", default='None')

def _parser_add_graphic_ext(parser):
    parser.add_argument('--graphic_ext', type=str, help="Extension of the output graphics, default is .pdf",
                        default='.pdf')

def _parser_add_pbc(parser):
    parser.add_argument('--pbc', dest='pbc', action='store_true',
                        help="Consider periodic boundary conditions when computing distances."
                             " Defaut is True")
    parser.add_argument('--no-pbc', dest='pbc', action='store_false')
    parser.set_defaults(pbc=True)

def _parser_add_output_desc(parser, default='output_sites'):
    parser.add_argument('--output_desc', type=str, help="Descriptor for output files. Default is %s"%default,
                        default=default)
    return parser

def _parser_add_sites(parser):
    parser.add_argument('--site_files', type=str, nargs='+',
                        help='site file(s) in json format containing site information, i.e., which bonds correspond to each site')

def _parser_add_fragment_names(parser):
    parser.add_argument('--fragment_names', type=str,
                        help="Name of the fragments. Leave empty if you want them automatically named."
                             " Otherwise, give a quoted list of strings separated by commas, e.g. "
                             "'TM1, TM2, TM3,'",
                        default="")

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

    _parser_add_nomenclature(parser)
    _parser_add_output_dir(parser)
    _parser_add_stride(parser)
    _parser_add_smooth(parser)
    _parser_add_fragment_names(parser)
    _parser_add_cutoff(parser)
    _parser_add_t_unit(parser)
    _parser_add_graphic_ext(parser)
    _parser_add_curve_colort(parser)
    _parser_add_gray_backgroud(parser)
    return parser


def parser_for_densities():
    parser = _parser_top_traj("For densities")
    _parser_add_sites(parser)
    _parser_add_output_dir(parser)
    _parser_add_stride(parser,help='Stride down the data by this factor. Default is 1.')

    return parser

def _parser_add_n_ctcs(parser, default=5):
    parser.add_argument("--n_ctcs", type=int,
                        help="Only the first n_ctcs most-frequent contacts will be written to the ouput. Default is 5.",
                        default=default)

def parser_for_rn():
    parser = _parser_top_traj(description='Small residue-residue contact analysis tool, initially developed for the '
                                      'receptor-G-protein complex.')


    parser.add_argument('--resSeq_idxs', type=str,
                        help='the resSeq idxs of interest (in VMD these are called "resid"). Can be in a format 1,2-6,10,20-25')

    _parser_add_cutoff(parser)
    _parser_add_stride(parser)
    _parser_add_n_ctcs(parser)
    parser.add_argument("--n_nearest", type=int,
                        help="Ignore this many nearest neighbors when computing neighbor lists. 'Near' means 'connected by this many bonds'. Default is 4.",
                        default=4)
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
    _parser_add_ascii(parser)
    parser.add_argument('--graphic_ext', type=str, help="Extension of the output graphics, default is .pdf",
                        default='.pdf')

    _parser_add_nomenclature(parser)
    _parser_add_output_dir(parser)
    _parser_add_output_desc(parser, default='neighborhoods')
    _parser_add_t_unit(parser)
    _parser_add_curve_colort(parser)
    _parser_add_gray_backgroud(parser)

    return parser

def _parser_add_ascii(parser):
    parser.add_argument('--output_ascii', type=str,
                        help="Extension for ascii files (.dat, .txt etc). Default is 'none', which does not write anything.",
                        default=None)


def parser_for_interface():
    parser = _parser_top_traj(description='Small residue-residue contact analysis tool, initially developed for the '
                                      'receptor-G-protein complex.\nContacts are computed between the group_1 '
                                      '(e.g. the receptor) and group_2 (e.g. the G-protein).\n '
                                      'To automate the identifaction of the residue indices corresponding to '
                                      'receptor and G-protein, the peptide-chain in the input topology '
                                      'is automatically broken down '
                                      'into fragments wherever there are jumps in the residue resSeq-indexing')

    _parser_add_cutoff(parser)
    _parser_add_stride(parser)
    _parser_add_smooth(parser)
    _parser_add_n_ctcs(parser, default=10)

    parser.add_argument('--fragments', default='resSeq', nargs="+",
                        help="How to detect fragments (i.e. breaks) in the peptide-chain. "
                             "Default is 'resSeq' so that the topology is split at the jumps in the residue sequence. \n"
                             "Alternatively, you can input specific fragments via their resSeq using space as separator using the following format, e.g.:"
                             "--fragments 1-10,15,14 20,21,30-50")

    parser.add_argument("-frag_idxs_group_1", type=int, nargs="+",
                        help="Indices of the fragments that belong to the group_1. "
                             "Defaults to [0,1], i.e. N-term to TM5 and TM6 to C-term of the receptor in the actor complex, except when "
                             "only two fragments are present. Then it defaults to [0]", default=[0, 1])
    parser.add_argument("-frag_idxs_group_2", type=int, nargs="+",
                        help="Indices of the fragments that belong to the group_1. "
                             "Defaults to [2,3,4,5], i.e. the all sub-units of the G-protein in the actor complex, except when "
                             "only two fragments are present. Then it defaults to [1]", default=[2, 3, 4, 5])
    parser.add_argument("--interface_cutoff_Ang", type=float,
                        help="The interface between group_1 and group_2 is defined as the set of group_1-group_2-"
                             "distances that are within this "
                             "cutoff in the reference topology. Otherwise, a large number of "
                             "non-necessary distances (e.g. between N-terminus and G-protein) are computed. Default is 35.",
                        default=35)

    _parser_add_fragment_names(parser)

    #parser.add_argument('--consolidate', dest='consolidate_opt', action='store_true',
    #                    help="Treat all trajectories as fragments of one single trajectory. Default is True")
    #parser.add_argument('--dont_consolidate', dest='consolidate_opt', action='store_false')
    #parser.set_defaults(consolidate_opt=True)
    _parser_add_nomenclature(parser)
    _parser_add_chunk(parser)
    _parser_add_output_desc(parser,'interface')
    _parser_add_graphic_ext(parser)
    _parser_add_ascii(parser)
    return parser

def fnmatch_ex(patterns_as_csv, list_of_keys):
    r"""
    Match the keys of the input dictionary against some naming patterns
    using Unix filename pattern matching TODO include link:  https://docs.python.org/3/library/fnmatch.html

    This method also allows for exclusions (grep -e)

    TODO: find out if regular expression re.findall() is better

    Parameters
    ----------
    patterns_as_csv : str
        Patterns to include or exclude, separated by commas, e.g.
        * "H*,-H8" will include all TMs but not H8
        * "G.S*" will include all beta-sheets
    list_of_keys : list
        Keys against which to match the patterns, e.g.
        * ["H1","ICL1", "H2"..."ICL3","H6", "H7", "H8"]

    Returns
    -------
    matching_keys : list

    """
    from fnmatch import fnmatch
    include_patterns = [pattern for pattern in patterns_as_csv.split(",") if not pattern.startswith("-")]
    exclude_patterns = [pattern[1:] for pattern in patterns_as_csv.split(",") if pattern.startswith("-")]
    #print(include_patterns)
    #print(exclude_patterns)
    match = lambda key, pattern: fnmatch(str(key), pattern) and all([not fnmatch(str(key), negpat) for negpat in exclude_patterns])
    outgroup = []
    for pattern in include_patterns:
        for key in list_of_keys:
            #print(key, pattern, match(key,pattern))
            if match(key, pattern):
                outgroup.append(key)
    return outgroup


def match_dict_by_patterns(patterns_as_csv, index_dict, verbose=False):
    r"""
    Joins all the values in an input dictionary if their key matches
    some patterns. This method also allows for exclusions (grep -e)

    TODO: find out if regular expression re.findall() is better

    Parameters
    ----------
    patterns_as_csv : str
        Comma-separated patterns to include or exclude, separated by commas, e.g.
        * "H*,-H8" will include all TMs but not H8
        * "G.S*" will include all beta-sheets
    index_dict : dictionary
        It is expected to contain iterable of ints or floats or anything that
        is "joinable" via np.hstack. Typically, something like:
        * {"H1":[0,1,...30], "ICL1":[31,32,...40],...}

    Returns
    -------
    matching_keys, matching_values : list, array of joined values

    """
    matching_keys =   fnmatch_ex(patterns_as_csv, index_dict.keys())
    if verbose:
        print(matching_keys)
    import numpy as _np

    matching_values = _np.hstack([index_dict[key] for key in matching_keys])
    return matching_keys, matching_values
r"""

Deal with residues, atoms, and their names, mostly.
The function :obj:`residues_from_descriptors` is probably the
most elaborate and most higher-level.

.. autosummary::
   :nosignatures:
   :toctree: generated/


"""
from mdtraj.core.residue_names import _AMINO_ACID_CODES
import mdtraj as _md
from fnmatch import filter as _fn_filter
import numpy as _np
from  pandas import unique as _pandas_unique
from mdciao.utils.lists import in_what_N_fragments as _in_what_N_fragments, force_iterable as _force_iterable
from mdciao.utils.str_and_dict import _kwargs_subs, match_dict_by_patterns as _match_dict_by_patterns
from collections import Counter as _Counter
from pandas import DataFrame as _DF
from collections import defaultdict as _defdict

def residues_from_descriptors(residue_descriptors,
                              fragments, top,
                              pick_this_fragment_by_default=None,
                              fragment_names=None,
                              additional_resnaming_dicts=None,
                              extra_string_info='',
                              just_inform=False,
                              ):
    r"""
    Returns residue idxs based on a list of residue descriptors.

    Fragments are needed to better identify residues. If a residue
    is present in multiple fragments, the user can dis-ambiguate
    or pick all residue idxs matching the :obj:`residue_descriptor`

    Because of this (one descriptor can match more than one residue)
    the return values are not necessarily of the same length
    as :obj:`residue_descriptors`

    Parameters
    ----------
    residue_descriptors: string or list of of strings
        AAs of the form of "GLU30" or "E30" or 30, can be mixed
    fragments: iterable of iterables of integers
        The integers in the iterables of 'fragments'
        represent residue indices of that fragment
    top: :obj:`~mdtraj.Topology`
    pick_this_fragment_by_default: None or integer.
        Pick this fragment without asking in case of ambiguity.
        If None, the user will we prompted
    fragment_names:
        list of strings providing informative names for the input :obj:`fragments`
    additional_resnaming_dicts : dict of dicts, default is None
        Dictionary of dictionaries. Lower-level dicts are keyed
        with residue indices and valued with additional residue names.
        Higher-level keys can be whatever. Use case is e.g. if "R131"
        needs to be disambiguated bc. it pops up in many fragments.
        You can pass {"GPCR":{895:"3.50", ...} here and that label
        will be displayed next to the residue. :obj:`mdciao.cli`
        methods use this.
    just_inform : bool, default is False
        Just inform about the AAs, don't ask for a selection
    extra_string_info: str,
        string with any additional info to be printed in case of ambiguity

    Returns
    -------
    residxs : list
        lists of integers that have been selected
    fragidxs : list
        The list of fragments where the residues are
    """
    residxs = []
    fragidxs = []
    last_answer = None
    first_run = True

    if isinstance(residue_descriptors, (str, int)):
        residue_descriptors = [residue_descriptors]

    for key in residue_descriptors:
        cands = _np.array(find_AA(str(key), top,extra_columns=additional_resnaming_dicts))
        cand_fragments =   _force_iterable(_np.squeeze(_in_what_N_fragments(cands, fragments)))
        # TODO refactor into smaller methods
        if len(cands) == 0:
            print("No residue found with descriptor %s" % key)
            residxs.append(None)
            fragidxs.append(None)
        elif len(cands) == 1:
            if len(cand_fragments)>1:
                raise ValueError("Your fragment definitions overlap, "
                                 "res_idx %u (%s) is found in fragments  %s"%
                                 (cands[0], top.residue(cands[0]), ', '.join([str(fr) for fr in cand_fragments])))
            elif len(cand_fragments)==0:

                raise ValueError("Your fragment definitions do not contain "
                                 "the residue of interest %u (%s)."%
                                 (cands[0], top.residue(cands[0])))
            residxs.append(cands[0])
            fragidxs.append(cand_fragments[0])
            if just_inform:
                istr = residue_line("0.0", top.residue(residxs[-1]), fragidxs[-1], consensus_maps = additional_resnaming_dicts, fragment_names=fragment_names)
                print(istr)
        else:
            istr = "Ambiguous definition for residue %s" % key
            istr += extra_string_info
            if not just_inform:
                print(istr)
            cand_chars = _np.hstack([['%s.%u'%(key,ii) for ii in range(n)] for key, n in _Counter(cand_fragments).items()]).tolist()
            for cc, ss, char in zip(cands, cand_fragments, cand_chars):
                istr = residue_line(char, top.residue(cc), ss, additional_resnaming_dicts, fragment_names=fragment_names)
                print(istr)
            if just_inform:
                print()
                residxs.extend([ii for ii in cands if ii not in residxs])
                fragidxs.extend([ii for ii in cand_fragments if ii not in fragidxs])
                continue
            if pick_this_fragment_by_default is None:
                if first_run is True:
                    last_answer = [int(ii) for ii in _np.unique(cand_fragments)][0]
                    choice_str = "choose first candidate fragment [%s]" % last_answer
                    first_run = False
                else:
                    choice_str = "repeat last option [%s]" % last_answer
                prompt = "Input one fragment idx out of %s and press enter (selects all matching residues in that fragment).\n" \
                         "Use one x.y descriptor in case of repeated fragment index.\n" \
                         "Leave empty and hit enter %s" % (
                         [int(ii) for ii in _np.unique(cand_fragments)], choice_str)

                answer = input(prompt)
            else:
                answer = str(pick_this_fragment_by_default)
                print("Automatically picked fragment %u" % pick_this_fragment_by_default)

            if len(answer) == 0:
                answer = last_answer

            if str(answer).isdigit():
                answer = int(answer)
                assert answer in cand_fragments
                idxs_w_answer = _np.flatnonzero([answer == ii for ii in cand_fragments])
                cands = cands[idxs_w_answer]
            elif '.' in str(answer) and answer in cand_chars:
                idx_w_answer = _np.flatnonzero([answer == ii for ii in cand_chars])
                answer = cand_fragments[idx_w_answer]
                cands = cands[idx_w_answer]
            else:
                raise ValueError("%s is not a possible answer" % answer)
                # TODO implent k for keeping this answer from now on

            assert answer in cand_fragments, (
                    "Your answer has to be an integer in the of the fragment list %s" % cand_fragments)
            last_answer = answer

            residxs.extend([int(ii) for ii in _force_iterable(cands)])
            fragidxs.extend([int(answer) for __ in _force_iterable(cands)])

    return residxs, fragidxs

@_kwargs_subs(residues_from_descriptors)
def rangeexpand_residues2residxs(range_as_str, fragments, top,
                                 interpret_as_res_idxs=False,
                                 sort=False,
                                 **residues_from_descriptors_kwargs):
    r"""
    Generalized range-expander from residue descriptors.

    Residue descriptors can be anything that :obj:`find_AA` understands.
    Expanding a range means getting "2-5,7" as input and returning "2,3,4,5,7".

    To dis-ambiguate descriptors, a fragment definition and a topology are needed

    TODO
    ----
    Internally, if an int or an iterable of ints is passed, they
    will be string-ified on-the-fly to work with the method "as-is",
    because the method was initially developed to interpret CLI-strings

    Note
    ----
    The input (= compressed range) is very flexible and accepts
    mixed descriptors and wildcards, eg: GLU*,ARG*,GDP*,LEU394,380-385 is a valid range.

    Expressions starting with "-", e.g. are exclusions, s.t. "GLU*,-GLU30" will
    select all GLUs except GLU30.

    Wildcards use the full resnames, i.e. "E*" is NOT equivalent to "GLU*"

    Expressions leading to empty ranges raise ValueError.

    Be aware, though, that wildcards are very powerful and easily "grab" a lot of
    residues, leading to long calculations and large outputs.

    See :obj:`find_AA` for more on residue descriptors.

    Parameters
    ----------
    range_as_str : string, int or iterable of ints
    fragments : list of iterable of residue indices
    top : :obj:`~mdtraj.Topology` object
    interpret_as_res_idxs : bool, default is False
        If True, indices without residue names ("380-385") values will be interpreted as
        residue indices, not residue sequential indices
    sort : bool
        sort the expanded range on return
    residues_from_descriptors_kwargs:
        Optional parameters for :obj:`~mdciao.utils.residue_and_atom.residues_from_descriptors`,
        which are listed below

    Other Parameters
    ---------------
    %(substitute_kwargs)s

    Returns
    -------
    residxs_out = list of unique residue indices
    """
    residxs_out = []
    AA_dict_for_exclusion, exclude = _top2AAmap(top), []
    if not isinstance(range_as_str,str):
        range_as_str = _force_iterable(range_as_str)
        assert all([isinstance(ii,(int,_np.int64)) for ii in range_as_str]),(range_as_str,[type(ii)  for ii in range_as_str])
        range_as_str= ','.join([str(ii) for ii in range_as_str])
    for r in [r for r in range_as_str.split(',') if r!=""]:
        if "*" in r or "?" in r or r.startswith("-"):
            if r.startswith("-"):
                exclude.extend(_match_dict_by_patterns(r[1:], AA_dict_for_exclusion)[1])
            else:
                filtered = find_AA(r, top, extra_columns= residues_from_descriptors_kwargs.get("additional_resnaming_dicts"))
                if len(filtered)==0:
                    raise ValueError("The input range contains '%s' which "
                                     "returns no residues!"%r)
                residxs_out.extend(filtered)
        else:
            resnames = r.split('-')
            is_range = "-" in r
            if interpret_as_res_idxs:
                # TODO clean the double "-" condiditon 
                if is_range:
                    assert len(resnames) == 2
                    for_extending = _np.arange(int(resnames[0]),
                                               int(resnames[-1]) + 1)
                else:
                    for_extending = [int(rr) for rr in resnames]
            else:
                for_extending, __ = residues_from_descriptors(resnames, fragments, top,
                                                            **residues_from_descriptors_kwargs)
                if None in for_extending:
                    for idesc in [int_from_AA_code(str(r)), name_from_AA(r)]:
                        if idesc not in [None,""]:
                            print("Trying with '%s'"%idesc)
                            residues_from_descriptors(idesc, fragments,top, just_inform=True)
                    if "." in r and residues_from_descriptors_kwargs.get("additional_resnaming_dicts", None) is None:
                        raise ValueError(f"The input range of residues contains a consensus label '{r}' which "
                                         "can't be resolved if no consensus information (GPCR, CGN, KLIFS) is passed. ")
                    else:
                        raise ValueError("The input range of residues contains '%s' which "
                                         "returns an untreatable range %s!\nCheck the above list for help." % (
                                         r, for_extending))
                if is_range:  # it was a pair
                    assert len(for_extending)==2
                    for_extending = _np.arange(for_extending[0],
                                               for_extending[-1] + 1)
                else: # it was something else
                    pass

            residxs_out.extend(for_extending)

    # Exclude the exclusions
    exclude = _np.unique(exclude)
    residxs_out=[rr for rr in residxs_out if rr not in exclude]

    if sort:
        residxs_out = sorted(residxs_out)

    residxs_out = _pandas_unique(_np.array(residxs_out))
    return residxs_out

def _top2AAmap(top):
    r"""

    Return a dictionary mapping AA expresions (GLU30, E30) to topology indices, for easier grabbing

    Maybe use with find_AA at some point

    Parameters
    ----------
    top : :obj:`mdtraj.Topology`

    Returns
    -------
    AA_dict : dict
        Keys are residue short and long codes (GLU30, E30) or
        long codes for nonstandard AAs (GTP365). Values
        are lists, since one topology might have more
        than one residue labeled E30
    """
    AA_dict = _defdict(list)
    [AA_dict[str(rr)].append(rr.index) for rr in top.residues]
    [AA_dict[shorten_AA(rr, keep_index=True, substitute_fail='long')].append(rr.index) for rr
     in top.residues]
    return {key : _np.unique(val).tolist() for key, val in AA_dict.items()}

def int_from_AA_code(key):
    """
    Returns the integer part from a residue name, None if there isn't

    Parameters
    ----------
    key : string
        Residue name passed as a string, example "GLU30"

    Returns
    -------
    int
        Integer part of the residue id, example- 30 if the input is "GLU30"

    """
    try:
        return int(''.join([ii for ii in key if ii.isnumeric()]))
    except ValueError:
        return None

def name_from_AA(key) -> str:
    """
    Return the residue name from a string

    Parameters
    ----------
    key : string or obj:`mdtraj.Topology.Residue` object
        Residue name passed as a string, example "GLU30" or as residue object

    Returns
    -------
    name: str
        Name of the residue, like "GLU" for "GLU30" or "E" for "E30"

    """

    if isinstance(key, _md.core.topology.Residue):
        name = key.name
    # CPs w/o topology will have purely numeric residue names, bc. these
    # are derived from the residue indices
    elif key.isnumeric():
        name = key
    else:
        rev_key = key[::-1]
        # Iterate from tail to head and break at the first alphabetic char
        for ii, char in enumerate(rev_key):
            if char.isalpha():
                break
        name = rev_key[ii:][::-1]
    return name

def shorten_AA(AA, substitute_fail=None, keep_index=False):
    r"""
    Return the short name of an AA, e.g. TRP30 to W by trying to
    use either the :obj:`mdtraj.Topology.Residue.code` attribute
    or :obj:`mdtraj` internals AA dictionary.

    Also work for residue strings w/o number, i.e. TRP to W.

    Parameters
    ----------
    AA: :obj:`~mdtraj.Topology.Residue` or a str
        The residue in question

    substitute_fail: str, default is None
        If there is no .code  attribute, there are different options
        depending on the value of this parameter
        * None : throw an exception when no short code is found (default)
        * 'long' : keep the residue's long name, i.e. do nothing
        * 'c': any alphabetic character, as long as it is of len=1
        * 0 : the first alphabetic character in the residue's name

    keep_index : bool, default is False
        If True return "Y30" for "TRP30", instead of returning just "Y"

    Returns
    -------
    code: str
        A string representing this AA using the short code

    """

    if isinstance(AA,str):
        name = name_from_AA(AA)
        if name.isnumeric():
            res = name
        else:
            res = str([name if len(name) == 1 else _AMINO_ACID_CODES.get(name)][0])
            idx = int_from_AA_code(AA)
            if idx is not None:
                res += str(idx)
    else:
        res = '%s%u'%(AA.code,AA.resSeq)

    #print(res, AA, substitute_fail)
    if "none" in str(res).lower():
        if substitute_fail is None:
            raise KeyError("There is no short version for your input %s (%s)"%(AA, type(AA)))
        elif isinstance(substitute_fail,str):
            if substitute_fail.lower()=="long":
                res = str(AA)
            elif len(substitute_fail)==1:
                res = '%s%u'%(substitute_fail,int_from_AA_code(str(AA)))
            else:
                raise ValueError("Cannot understand your input. Please refer to the docs.")
        elif isinstance(substitute_fail,int) and substitute_fail==0:
            res = '%s%u' % (str(AA)[0], int_from_AA_code(str(AA)))
        else:
            raise ValueError("Cannot understand your input. Please refer to the docs.")

    if keep_index:
        return res
    else:
        #Weird edge case: what happens if the name is just a number "700" and the keep_index is False?
        return name_from_AA(res)

def atom_type(aa, no_BB_no_SC='X'):
    r"""
    Return a string BB or SC for backbone or sidechain atom.

    Parameters
    ----------
    aa : :obj:`mdtraj.core.topology.Atom` object
    no_BB_no_SC : str, default is X
        Return this string if :obj:`aa` isn't either BB or SC

    Returns
    -------
    aatype : str

    """
    if aa.is_backbone:
        return 'BB'
    elif aa.is_sidechain:
        return 'SC'
    else:
        return no_BB_no_SC

def parse_and_list_AAs_input(AAs, top, map_conlab=None):
    r"""Helper method to print information regarding AA descriptors

    Parameters
    ----------
    AAs : None or str
        CSVs of AA descriptors, e.g.
        'GLU30,GLU*,GDP', anything that :obj:`find_AA` can read
        How AAs are being described
    top : :obj:`~mdtraj.Topology`
        Topology where the AAs live
    map_conlab : dict, list or array, default is None
        maps residue indices to consensus labels

    Returns
    -------

    """
    if str(AAs).lower()!="none":
        AAs = [aa.strip(" ") for aa in AAs.split(",")]
        for aa in AAs:
            cands = find_AA(aa, top)
            if len(cands) == 0:
                print("No %s found in the input topology" % aa)
            else:
                for idx in cands :
                    rr = top.residue(idx)
                    if map_conlab is not None:
                        print(idx, rr, map_conlab[idx])
                    else:
                        print(idx,rr)
        print()

def find_CA(res, CA_name="CA", CA_dict=None):
    r""" Return the CA atom (or something equivalent) for this residue


    Parameters
    ----------
    res : :obj:`mdtraj.Residue` object

    CA_name : str, default is "CA"
        The name by which you identify the CA.
        This overrules anything that's parsed
        in the :obj:`CA_dict`, i.e. if the
        residue you are passing has both
        an atom "CA" and an entry
        in the CA_dict, the "CA" atom will
        be returned.

    CA_dict : dict, default is None
        You can provide a dictionary keyed
        with residue names and valued
        with strings that identify a "CA"-equivalent
        atom (e.g. in ligands)
        If None, the default :obj:`_CA_rules` are used:
        _CA_rules = {"GDP": "C1", "P0G":"C12"}

    """

    if CA_dict is None:
        CA_dict = _CA_rules

    CA_atom = list(res.atoms_by_name(CA_name))
    if len(CA_atom) == 1:
        pass
    elif len(CA_atom) == 0:
        if res.n_atoms == 1:
            return list(res.atoms)[0]
        else:
            try:
                CA_atom = list(res.atoms_by_name(CA_dict[res.name]))
                assert len(CA_atom) == 1
            except KeyError:
                raise NotImplementedError(
                    "This method does not know what the 'CA' of a %s is. Known 'CA'-equivalents are %s" % (res, CA_dict.keys()))
    else:
        raise ValueError("More than one CA atom for %s:%s"%(res,CA_atom))
    return CA_atom[0]


_CA_rules = {"GDP": "C1", "P0G":"C12"}

_AA_types = {"positive": "ARG HIS LYS",
             "negative": "ASP GLU",
             "polar": "SER THR ASN GLN",
             "special": "CYS GLY PRO",
             "hydrophobic": "ALA ILE LEU MET PHE TRP TYR VAL"}
for _key in _AA_types.keys():
    # the str().split(.) is an ugly hack for building the docs, smh the dict gets turned into a named tuple
    _AA_types[_key] += ' '+" ".join([str(_AMINO_ACID_CODES[_AA]).split(".")[-1] for _AA in _AA_types[_key].split()])
_res2restype = {aa:key for key, val in _AA_types.items() for aa in val.split()}

def _residue_sidechain_membership(scheme,residue):
    r"""
    Which atoms of a `residue` should be considered when using `schemes` involving the sidechain.

    Handles cases such as GLYs (no sidechain) or non-protein residues gracefully by always returning atom indices.

    In GLYs:
     * `sidechain` chooses the sidechain Hydrogens if they're present,
     else it falls back to all other available atoms, including hydrogen.

     * `sidechain-heavy` also chooses the sidechain Hydrogens if they're present,
     else it falls back to all other available atoms, excluding hydrogen.

    This means that, when sidechain Hydrogens are present, `sidechain` and
    `sidechain-heavy` yield the same result, as currently implemented by
    mdtraj.

    For non-protein residues, `sidechain` chooses all atoms, including
    hydrogens, and `sidechain-heavy` all except hydrogens.

    This way, a minimum distance can always be computed by
    w/o breaking the flow of _md_compute_contacts.compute_contacts

    Parameters
    ----------
    scheme : str
        Has to be either "sidechain" or "sidechain-heavy"
    residue : mdtraj.core.topology.Residue
        The residue for which atom indices should be
        returned for a given `scheme`

    Returns
    -------
    memb : list
        List of the atom indices in `residue` that
        should be considered when using a given
        `scheme`. Will fall back to available
        (heavy)atoms in special cases such as Glycines
        or residues for which `residue.is_protein` is False,
        like ligands and nucleotides.
    """
    if scheme == "sidechain":
        if residue.is_protein:
            if residue.name == "GLY":
                memb = [atom.index for atom in residue.atoms if atom.is_sidechain]
                if len(memb) == 0: #This is the case of GLY not having Hs in the topology
                    memb = [atom.index for atom in residue.atoms]
            else:
                memb = [atom.index for atom in residue.atoms if atom.is_sidechain]
        else:
            memb = [atom.index for atom in residue.atoms]
    elif scheme == "sidechain-heavy":
        if residue.is_protein:
            if residue.name == "GLY":
                memb = [atom.index for atom in residue.atoms if atom.is_sidechain]
                if len(memb) == 0: #This is the case of GLY not having Hs in the topology
                    memb = [atom.index for atom in residue.atoms if not atom.element == _md.core.element.hydrogen]
            else:
                memb = [
                    atom.index
                    for atom in residue.atoms
                    if atom.is_sidechain and not (atom.element == _md.core.element.hydrogen)
                ]
        else:
            memb = [
                atom.index
                for atom in residue.atoms
                if not (atom.element == _md.core.element.hydrogen)
            ]
    else:
        raise NotImplementedError("`scheme` has to be 'sidechain' or 'sidechain-heavy'")
    return memb

def AAtype(res,
           return_color=False,
           typecolors={"positive": "blue",
                       "negative": "red",
                       "polar": "green",
                       "special": "gray",
                       "hydrophobic": "gray",
                       "NA": "purple"}):
    r"""
    Residue types, optionally color coded

    The types are:
    * "positive": "ARG HIS LYS",
    * "negative": "ASP GLU",
    * "polar": "SER THR ASN GLN",
    * "special": "CYS GLY PRO",
    * "hydrophobic": "ALA ILE LEU MET PHE TRP TYR VAL"

    Parameters
    ----------
    res : str or :obj:`~mdtraj.core.topology.Residue`
    return_color : bool, default is False
        Return the color associated
        with the type (positive:blue, negative:red, etc)
        rather than type itself
    typecolors : dict
        The map of types to colors

    Returns
    -------

    rtype : str
        Either the type or the color
    """
    if isinstance(res,str):
        key = res
    else:
        key = res.name

    rtype = _res2restype.get(key,"NA")
    if return_color:
        return typecolors[rtype]
    else:
        return rtype

def residue_line(item_desc, residue, frag_idx,
                 consensus_maps=None,
                 fragment_names=None,
                 table=False):
    r"""Return a string that describes the residue

    Can be used just to to inform or to help dis-ambiguating:

    >>> 0.0)        LEU45 with residue index   41 in fragment 0  (  CGN: LEU45@G.S1.6)
    >>> 3.0)        LEU45 with residue index  775 in fragment 3  ( GPCR: LEU45@1.44x44)


    Parameters
    ----------
    item_desc : str
        Description for the item of the list,
        "1.0" or "3.2"
    residue : :obj:`~mdtraj.core.Residue`
    frag_idx : int
        Fragment index
    fragment_names : list, default is None
        Fragment names
    consensus_maps : dict of indexables, default is None
        Dictionary of dictionaries. Lower-level dicts are keyed
        with residue indices and valued with additional residue names.
        Higher-level keys can be whatever. Use case is e.g. if "R131"
        needs to be disambiguated bc. it pops up in many fragments.
        You can pass {"GPCR":{895:"3.50", ...} here and that label
        will be displayed next to the residue.
    table : bool, default is False
        Assume a header has been already printed
        out and print the line with the
        inline tags

    Returns
    -------
    istr : str
        An informative string about this residue, that
        can be used to dis-ambiguate via the unique
        item descriptor, e.g:
        3.1)       GLU122 in fragment 3 with residue index 852 (: 3.41)

    """
    res_idx = residue.index

    fragname = " "
    if fragment_names is not None:
        fragname = ' (%s) ' % fragment_names[frag_idx]

    if not table:
        istr = '%-6s %10s in fragment %u%swith residue index %2u' % (item_desc + ')', residue, frag_idx, fragname, res_idx)
        if consensus_maps is not None:
            extra = ''
            for key1, val1 in consensus_maps.items():
                try:
                    jstr = val1[res_idx]
                    if jstr is not None:
                        extra += '%5s: %s@%s ' % (key1, str(residue), val1[res_idx])
                except (KeyError,IndexError):
                    pass
            if len(extra) > 0:
                istr = istr + ' (%s)' % extra.rstrip(" ")
    else:
        add_dicts = []
        if consensus_maps is not None:
            for key in consensus_maps.keys():
                add_dicts.append(_try_double_indexing(consensus_maps, key, res_idx))

        istr = "  ".join(["%10s" % str(item) for item in [residue, res_idx,
                                                    frag_idx,
                                                    residue.resSeq,
                                                    ] + add_dicts])
    return istr

def _try_double_indexing(indexable, idx1, idx2):
    try:
        return indexable[idx1][idx2]
    except (KeyError, IndexError,TypeError):
        return None


def top2lsd(top, substitute_fail="X",
            extra_columns=None):
    r"""
    Return a list of per-residue attributes as dictionaries

    Use :obj:`~pandas.DataFrame` on the return value for a nice table

    Parameters
    ----------
    top : :obj:`~mdtraj.Topology`
    substitute_fail : str, None, int, default is "X"
        If there is no .code  attribute, there are different options
        depending on the value of this parameter
         * None : throw an exception when no short code is found (default)
         * 'long' : keep the residue's long name, i.e. do nothing
         * 'c': any alphabetic character, as long as it is of len=1
         * 0 : the first alphabetic character in the residue's name
    extra_columns : dictionary of indexables
        Any other columns you want to
        include in the :obj:`~pandas.DataFrame`, e.g.
        {"GPCR" : [None, None,...,3.50, 3.51...],
         "CGN"  : [G.H5.25, None, None, ...]}
         If the values are lists, they sould be
         len=top.n_residues, if dicts, the dicts
         don't need to cover all residues of `top`, e.g.
        {"GPCR" : {200 : "3.50", 201 : "3.51"},
         "CGN"  : {0 : "G.H5.25"}}

    Returns
    -------
    df : :obj:`~pandas.DataFrame`
    """
    list_of_dicts = []
    for rr in top.residues:
        rdict = {"residue": str(rr),
                 "index": rr.index,
                 "name": rr.name,
                 "resSeq": rr.resSeq,
                 "code": shorten_AA(rr, substitute_fail=substitute_fail, keep_index=False),
                 "short": shorten_AA(rr, substitute_fail=substitute_fail, keep_index=True)}
        if extra_columns is not None:
            for key, val in extra_columns.items():
                try:
                    rdict[key] = val[rr.index]
                except (KeyError, IndexError):
                    rdict[key] = None
        list_of_dicts.append(rdict)

    return list_of_dicts

def find_AA(AA_pattern, top,
            extra_columns=None,
            return_df=False):
    r"""

    Residue matching with UNIX-shell patterns

    Similar to the shell command "ls",
    using posix-style wildcards like
    shown in the examples or here:
    https://docs.python.org/3/library/fnmatch.html

    Any other attribute that's passed
    as :obj:`extra_columns` will be
    matched as explained below, e.g.
    "3.50" to get one residue in
    the GPCR-nomenclature or "3.*"
    to get the whole TM-helix 3

    The examples use '*' as wildcard,
    but '?' (as in 'ls') also works

    Examples
    --------
        * 'PRO' : returns all PROs, matching
          via the attribute "name"
        * 'P'   : returns all PROs, matching
          via the attribute "code"
        * 'P*'  : returns all PROs,PHEs and
          any other residue that starts with "P",
          either in "name" or in "code"
        * 'PRO39' : returns PRO39, matching
          via full residue name (long)
        * 'P39'  : returns PRO39, matching
          via full residue name (short)
        * 'PRO3*' : returns all PROs
          with sequence indices that start
          with 3, e.g. 'PRO39, PRO323, PRO330' etc
        * '3' : returns all residues with
          sequence indices 3
        * '3*' : returns all residues with
          sequence indices that start with 3


    Parameters
    ----------
    AA_patt : str or int
    top : :obj:`~mdtraj.Topology`
    return_df : bool, default is False
        Return the full :obj:`~pandas.DataFrame`
        of the matching residues

    Returns
    -------
    AAs : list or :obj:`~pandas.DataFrame`
        List of serial residue indices, s.t.
        top.residue(idx) would return the wanted residue.
        With :obj:`return_df`, you can get the
        full :obj:`~pandas.DataFrame` of the
        matching residues.
    """

    lsd = top2lsd(top, substitute_fail="X", extra_columns=extra_columns)

    idxs = [ii for ii, idict in enumerate(lsd) if _fn_filter([str(val) for key, val in idict.items() if key!="index"], str(AA_pattern))]

    if return_df:
        return _DF([lsd[ii] for ii in idxs])
    else:
        return idxs


def _ls_AA_in_df(AA_patt, df):
    r""" Same as find_AA but using dataframe syntax...between 10 and 100 times slower (200mus to 20ms)"""
    from fnmatch import fnmatch as _fnmatch
    _AA = str(AA_patt)
    match = lambda val : _fnmatch(val,_AA)
    try:
        idxs = _np.flatnonzero(df.map(lambda val: str(val)).map(match).values.any(1)).tolist()
    except AttributeError:
        # Needed for py37 and py38
        idxs = _np.flatnonzero(df.applymap(lambda val: str(val)).applymap(match).values.any(1)).tolist()
    return idxs

def get_SS(SS,top=None):
    r"""
    Try to guess what type of input for secondary-structure computation the user wants, and compute it

    Parameters
    ----------
    SS : secondary structure information
        Can be many things:
        * triple of ints (CP_idx, traj_idx, frame_idx)
          Nothing happens, the tuple is returned as
          is and handled externally by the :obj:`ContactGroup`
          that called this method.
          Tuple representing a ContactPair, trajectory
          See the docs there for more info
        * True
          same as [0,0,0]
        * None or False
          Do nothing
        * :obj:`mdtraj.Trajectory`
          Use this geometry to compute the SS
        * string
          Path to a filename, of which only
          the first frame will be read. The
          SS will be computed from there.
          The file will be tried to read
          first without topology information
          (e.g. .pdb, .gro, .h5 will work),
          and when this fails, the :obj:`top`
          will be passed (e.g. .xtc, .dcd)
        * array_like
          Use the SS from here, s.t.ss_inf[idx]
          gives the SS-info for the residue
          with that idx
    top : :obj:`~mdtraj.Topology`, default is None

    Returns
    -------
    from_tuple : bool
        Whether the infor should be gotten from
        a tuple or not
    ss_array : np.ndarray or None
    """
    from_tuple = False
    ss_array = None
    if SS is None or (isinstance(SS, bool) and not SS):
        pass
    elif isinstance(SS, _md.Trajectory):
        ss_array = _md.compute_dssp(SS[0], simplified=True)[0]
    elif isinstance(SS, str):
        try:
            ss_array = _md.compute_dssp(_md.load(SS, frame=0), simplified=True)[0]
        except (OSError, ValueError) as e:
            ss_array = _md.compute_dssp(_md.load(SS, top=top, frame=0), simplified=True)[0]
    elif SS is True:
        from_tuple = (0, 0, 0)
    elif len(SS) == 3:
        from_tuple = SS
    else:
        ss_array = SS

    return from_tuple, ss_array
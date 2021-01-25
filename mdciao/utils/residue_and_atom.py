r"""

Deal with residues, atoms, and their names, mostly.
The function :obj:`residues_from_descriptors` is probably the
most elaborate and most higher-level.

.. autosummary::
   :nosignatures:
   :toctree: generated/


"""
from mdtraj.core.residue_names import _AMINO_ACID_CODES
from fnmatch import filter as _fn_filter
import numpy as _np
from  pandas import unique as _pandas_unique
from mdciao.utils.lists import in_what_N_fragments as _in_what_N_fragments, force_iterable as _force_iterable
from collections import Counter as _Counter
from pandas import DataFrame as _DF

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
        list of strings providing informative names input :obj:`fragments`
    additional_resnaming_dicts : dict of dicts, default is None
        Dictionary of dictionaries. Lower-level dicts are keyed
        with residue indices and valued with additional residue names.
        Higher-level keys can be whatever. Use case is e.g. if "R131"
        needs to be disambiguated bc. it pops up in many fragments.
        You can pass {"BW":{895:"3.50", ...} here and that label
        will be displayed next to the residue. :obj:`mdciao.cli`
        methods use this.
    just_inform : bool, default is False
        Just inform about the AAs, don't ask for a selection
    extra_string_info: string with any additional info to be printed in case of ambiguity

    Returns
    -------
    residxs : list
        lists of integers that have been selected
    fragidxs : list
        The list of fragments where the residues are
    """
    residxs = []
    fragidxs = []
    last_answer = '0'


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
                istr = residue_line("0.0", top.residue(residxs[-1]), fragidxs[-1], additional_resnaming_dicts, fragment_names=fragment_names)
                print(istr)
        else:
            istr = "ambiguous definition for AA %s" % key
            istr += extra_string_info
            if not just_inform:
                print(istr)
            cand_chars = _np.hstack([['%s.%u'%(key,ii) for ii in range(n)] for key, n in _Counter(cand_fragments).items()]).tolist()
            for cc, ss, char in zip(cands, cand_fragments, cand_chars):
                istr = residue_line(char, top.residue(cc), ss, additional_resnaming_dicts)
                print(istr)
            if just_inform:
                print()
                residxs.extend([ii for ii in cands if ii not in residxs])
                fragidxs.extend([ii for ii in cand_fragments if ii not in fragidxs])
                continue
            if pick_this_fragment_by_default is None:
                prompt = "Input one fragment idx out of %s and press enter (selects all matching residues in that fragment).\n" \
                         "Use one x.y descriptor in case of repeated fragment index.\n" \
                         "Leave empty and hit enter to repeat last option [%s]" % (
                         [int(ii) for ii in _np.unique(cand_fragments)], last_answer)

                answer = input(prompt)
            else:
                answer = str(pick_this_fragment_by_default)
                print("Automatically picked fragment %u" % pick_this_fragment_by_default)

            if len(answer) == 0:
                answer = last_answer

            if str(answer).isdigit():
                answer = int(answer)
                assert answer in cand_fragments
                idxs_w_answer = _np.argwhere([answer == ii for ii in cand_fragments]).squeeze()
                cands = cands[idxs_w_answer]
            elif '.' in answer and answer in cand_chars:
                idx_w_answer = _np.argwhere([answer == ii for ii in cand_chars]).squeeze()
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

def rangeexpand_residues2residxs(range_as_str, fragments, top,
                                 interpret_as_res_idxs=False,
                                 sort=False,
                                 **residues_from_descriptors_kwargs):
    r"""
    Generalized range-expander from residue descriptors.

    Residue descriptors can be anything that :obj:`find_AA` understands.
    Expanding a range means getting "2-5,7" as input and returning "2,3,4,5,7"

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

    Wildcards use the full resnames, i.e. E* is NOT equivalent to GLU*

    Be aware, though, that wildcards are very powerful and easily "grab" a lot of
    residues, leading to long calculations and large outputs.

    See :obj:`find_AA` for more on residue descriptors

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
        Optional parameters for :obj:`residues_from_descriptors`

    Returns
    -------
    residxs_out = list of unique residue indices
    """
    residxs_out = []
    #print("For the range", range_as_str)
    if not isinstance(range_as_str,str):
        range_as_str = _force_iterable(range_as_str)
        assert all([isinstance(ii,(int,_np.int64)) for ii in range_as_str]),(range_as_str,[type(ii)  for ii in range_as_str])
        range_as_str= ','.join([str(ii) for ii in range_as_str])
    for r in [r for r in range_as_str.split(',') if r!=""]:
        assert not r.startswith("-")
        if "*" in r or "?" in r:
            assert "-" not in r
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
                    raise ValueError("The input range of residues contains '%s' which "
                                     "returns an untreatable range %s!\nCheck the above list for help." % (r, for_extending))
                if is_range:  # it was a pair
                    assert len(for_extending)==2
                    for_extending = _np.arange(for_extending[0],
                                               for_extending[-1] + 1)
                else: # it was something else
                    pass

            residxs_out.extend(for_extending)

    if sort:
        residxs_out = sorted(residxs_out)

    residxs_out = _pandas_unique(residxs_out)
    return residxs_out

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

def name_from_AA(key):
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


    return ''.join([ii for ii in str(key) if ii.isalpha()])

def shorten_AA(AA, substitute_fail=None, keep_index=False):
    r"""
    Return the short name of an AA, e.g. TRP30 to W by trying to
    use either the :obj:`mdtraj.Topology.Residue.code` attribute
    or :obj:`mdtraj` internals AA dictionary

    Parameters
    ----------
    AA: :obj:`~mdtraj.Topology.Residue` or a str
        The residue in question

    substitute_fail: str, default is None
        If there is no .code  attribute, different options are there
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
        try:
            res = '%s%u'%(_AMINO_ACID_CODES[name_from_AA(AA)], int_from_AA_code(AA))
        except KeyError:
            res = None
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

def residue_line(item_desc, residue, frag_idx,
                 consensus_maps=None,
                 fragment_names=None,
                 table=False):
    r"""Return a string that describes the residue

    Can be used justo to inform or to help dis-ambiguating:
    0.0)        GLU10 in fragment 0 with residue index  6 (CGN: G.HN.27)
    ...
    1.0)        GLU10 in fragment 1 with residue index 363


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
        You can pass {"BW":{895:"3.50", ...} here and that label
        will be displayed next to the residue.
    table : bool, default is False
        Assume a header has been aready printed
        out and print the line with the
        inline tags

    Returns
    -------
    istr : str
        An informative string about this residue, that
        can be used to dis-ambiguate via the unique
        item descriptor, e.g:
        3.1)       GLU122 in fragment 3 with residue index 852 (BW: 3.41)

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
                        extra += '%s: %s ' % (key1, val1[res_idx])
                except (KeyError,IndexError):
                    pass
            if len(extra) > 0:
                istr = istr + ' (%s)' % extra.rstrip(" ")
    else:
        add_dicts = []
        if consensus_maps is not None:
            for key in ["BW","CGN"]:
                add_dicts.append(_try_double_indexing(consensus_maps, key, res_idx))

        istr = '%10s  %10u  %10u %10u %10s %10s' % (residue, res_idx,
                                                    frag_idx,
                                                    residue.resSeq,
                                                    add_dicts[0], add_dicts[1])
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
        If there is no .code  attribute, different options are there
        depending on the value of this parameter
         * None : throw an exception when no short code is found (default)
         * 'long' : keep the residue's long name, i.e. do nothing
         * 'c': any alphabetic character, as long as it is of len=1
         * 0 : the first alphabetic character in the residue's name
    extra_columns : dictionary of indexables
        Any other column you want to
        include in the :obj:`~pandas.DataFrame`

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
    the BW-nomenclature or "3.*"
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
    idxs = _np.flatnonzero(df.applymap(lambda val: str(val)).applymap(match).values.any(1)).tolist()
    return idxs
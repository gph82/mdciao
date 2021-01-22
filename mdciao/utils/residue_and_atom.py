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

def find_AA(top, AA_pattern):
    """
    Query the index of a residue(s) using a pattern.

    Parameters
    ----------
    top : :py:class:`mdtraj.Topology`
    AA_pattern : string
        Exact patterns work like this
         * "GLU30" and "E30" are equivalent and return the index for GLU30
         * "GLU" and "E" return indices for all GLUs
         * "GL" will raise ValueError
         * "30" will return GLU30 and LYS30

        Wildcards are matched against full residue names
         * "GLU*" will return indices for all GLUs (equivalent to GLU)
         * "GLU3?" will only return indices all GLUs in the thirties
         * "E*" will NOT return any GLUs

    #TODO rewrite everything cleaner with fnmatch etc
    #TODO handle cases when no residue was found uniformly accross mdciao either with None or []

    Returns
    -------
    list
        list of res_idxs where the residue is present,
        so that top.residue(idx) would return the wanted AA

    """
    get_name = {1: lambda rr: rr.code,
                2: lambda rr: rr.name,
                3: lambda rr: rr.name}

    if AA_pattern[-1].isalpha():
        lenA = len(AA_pattern)
        if lenA not in [1,2,3]:
            raise ValueError("purely alphabetic patterns must have "
                             " either 1, 2, or 3 letters, not  %s" % (AA_pattern))

        return [rr.index for rr in top.residues if AA_pattern == '%s' % (get_name[lenA](rr))]
    elif AA_pattern.isdigit():
        return [rr.index for rr in top.residues if rr.resSeq == int(AA_pattern)]
    elif "*" in AA_pattern or "?" in AA_pattern:
        resnames = [str(rr) for rr in top.residues]
        filtered = _fn_filter(resnames, AA_pattern)
        filtered_idxs  = [ii for ii, resname in enumerate(resnames) if resname in filtered]
        return  _np.unique(filtered_idxs)
    else:
        code = ''.join([ii for ii in AA_pattern if ii.isalpha()])
        try:
            return [rr.index for rr in top.residues if AA_pattern == '%s%u' % (get_name[len(code)](rr), rr.resSeq)]
        except KeyError:
            raise ValueError(
                "The input AA %s must have an alphabetic code of either 3 or 1 letters, but not %s" % (AA_pattern, code))


#TODO what happens when residue are not found?G1G1G1G1
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
        The integers in the iterables of 'fragments' represent residue indices of that fragment
    top: :obj:`mdtraj.Topology`
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
        cands = _np.array(find_AA(top, str(key)))
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
        else:
            istr = "ambiguous definition for AA %s" % key
            istr += extra_string_info
            if not just_inform:
                print(istr)
            cand_chars = _np.hstack([['%s.%u'%(key,ii) for ii in range(n)] for key, n in _Counter(cand_fragments).items()]).tolist()
            for cc, ss, char in zip(cands, cand_fragments, cand_chars):
                fname = " "
                if fragment_names is not None:
                    fname = ' (%s) ' % fragment_names[ss]
                istr = '%-6s %10s in fragment %2u%swith residue index %2u' % (char+')', top.residue(cc), ss, fname, cc)
                if additional_resnaming_dicts is not None:
                    extra = ''
                    for key1, val1 in additional_resnaming_dicts.items():
                        if cc in val1.keys() and val1[cc] is not None:
                            extra += '%s: %s ' % (key1, val1[cc])
                    if len(extra) > 0:
                        istr = istr + ' (%s)' % extra.rstrip(" ")
                print(istr)
            if just_inform:
                return
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
    top : :obj:`mdtraj.Topology` object
    interpret_as_res_idxs : bool, default is False
        If True, indices without residue names ("380-385") values will be interpreted as
        residue indices, not resdiue sequential indices
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
            filtered = find_AA(top, r)
            if len(filtered)==0:
                raise ValueError("The input range contains '%s' which "
                                 "returns no residues!"%r)
            residxs_out.extend(find_AA(top,r))
        else:
            resnames = r.split('-')
            if interpret_as_res_idxs:
                # TODO clean the double "-" condiditon 
                if "-" in r:  # it was a pair
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
                if "-" in r:  # it was a pair
                    assert len(for_extending)==2
                    for_extending = _np.arange(for_extending[0],
                                               for_extending[-1] + 1)
                else: # it was something else
                    pass
                    #for_extending = for_extending[0]

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
    AA: :obj:`mdtraj.Topology.Residue` or a str
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
    top : :obj:`mdtraj.Topology`
        Topology where the AAs live
    map_conlab : dict, list or array, default is None
        maps residue indices to consensus labels

    Returns
    -------

    """
    if str(AAs).lower()!="none":
        AAs = [aa.strip(" ") for aa in AAs.split(",")]
        for aa in AAs:
            cands = find_AA(top,aa)
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
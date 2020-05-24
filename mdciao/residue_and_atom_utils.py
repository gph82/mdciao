from mdtraj.core.residue_names import _AMINO_ACID_CODES
from fnmatch import filter as _fn_filter
import numpy as np
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
    #TODO handle cases when no residue was found uniformly
    # accross mdciao either with None or []
    Returns
    -------
    list
        list of res_idxs where the residue is present, so that top.residue(idx) would return the wanted AA

    """
    get_name = {1: lambda rr: rr.code,
                2: lambda rr: rr.name,
                3: lambda rr: rr.name}

    if AA_pattern.isalpha():
        lenA = len(AA_pattern)
        assert lenA in [1,3], ValueError("purely alphabetic patterns must have " 
                                     " either 3 or 1 letters, not  %s" % (AA_pattern))


        return [rr.index for rr in top.residues if AA_pattern == '%s' % (get_name[lenA](rr))]
    elif AA_pattern.isdigit():
        return [rr.index for rr in top.residues if rr.resSeq == int(AA_pattern)]
    elif "*" in AA_pattern or "?" in AA_pattern:
        resnames = [str(rr) for rr in top.residues]
        filtered = _fn_filter(resnames, AA_pattern)
        filtered_idxs  = [ii for ii, resname in enumerate(resnames) if resname in filtered]
        return  np.unique(filtered_idxs)
    else:
        code = ''.join([ii for ii in AA_pattern if ii.isalpha()])
        try:
            return [rr.index for rr in top.residues if AA_pattern == '%s%u' % (get_name[len(code)](rr), rr.resSeq)]
        except KeyError:
            raise ValueError(
                "The input AA %s must have an alphabetic code of either 3 or 1 letters, but not %s" % (AA_pattern, code))

def int_from_AA_code(key):
    """
    Returns the integer part from a residue name.

    Parameters
    ----------
    key : string
        Residue name passed as a string, example "GLU30"

    Returns
    -------
    int
        Integer part of the residue id, example- 30 if the input is "GLU30"

    """
    return int(''.join([ii for ii in key if ii.isnumeric()]))

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
    use either the :obj:`mdtraj.Topology.Residue.code' attribute
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

def _atom_type(aa, no_BB_no_SC='X'):
    r"""
    Return a string BB or SC for backbone or sidechain atom.
    Parameters
    ----------
    aa : :obj:`mtraj.core.topology.Atom` object
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
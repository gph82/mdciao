from mdtraj.core.residue_names import _AMINO_ACID_CODES
def find_AA(top, AA, relax=False):
    """
    Query the index of residue based on a string: e.g. "GLU30", "E30", "GLU","E".
    If provided only with a numeric str, "30", it will be interpreted as the
    resSeq entry.

    Parameters
    ----------
    top : :py:class:`mdtraj.Topology`
    AA : string
        Anything that could be used to identify a residue "GLU30" or "E30"
    relax : boolean, default is True
        Relaxes match criteria to include just the name ("GLU")
    #TODO think about relax=True always

    Returns
    -------
    list
        list of res_idxs where the residue is present, so that top.residue(idx) would return the wanted AA

    """

    if AA.isalpha():
        lenA = len(AA)
        if relax:
            assert lenA <= 3, ValueError("The input AA %s must have an alphabetic code of" 
                                         " either 3 or 1 letters" % (AA))

            get_name = {1: lambda rr : rr.code,
                        2: lambda rr : rr.name,
                        3: lambda rr : rr.name}

            return [rr.index for rr in top.residues if AA == '%s' % (get_name[lenA](rr))]

        else:
            raise ValueError(
                "Missing the resSeq index, all I got was %s. Check out the relax option" % (AA))
    elif AA.isdigit():
        return [rr.index for rr in top.residues if rr.resSeq == int(AA)]
    else:
        code = ''.join([ii for ii in AA if ii.isalpha()])

        if len(code)==1:
            return [rr.index for rr in top.residues if AA == '%s%u' % (rr.code, rr.resSeq)]
        elif len(code) in [2,3]:
            return [rr.index for rr in top.residues if AA == '%s%u' % (rr.name, rr.resSeq)]
        else:
            raise ValueError(
                "The input AA %s must have an alphabetic code of either 3 or 1 letters, but not %s" % (AA, code))

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
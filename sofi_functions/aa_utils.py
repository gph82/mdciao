from mdtraj.core.residue_names import _AMINO_ACID_CODES
def find_AA(top, AA):
    """
    This function is used to query the index of residue based on residue name.

    Parameters
    ----------
    top : :py:class:`mdtraj.Topology`
    AA : string
        Valid residue name to be passed as a string, example- "GLU30" or "E30"

    Returns
    -------
    list
        list of res_idxs where the residue is present, so that top.residue(idx) would return the wanted AA

    """
    code = ''.join([ii for ii in AA if ii.isalpha()])
    if len(code)==1:
        return [rr.index for rr in top.residues if AA == '%s%u' % (rr.code, rr.resSeq)]
    elif len(code)==3:
        return [rr.index for rr in top.residues if AA == '%s%u' % (rr.name, rr.resSeq)]
    else:
        raise ValueError("The input AA %s must have an alphabetic code of either 3 or 1 letters"%AA)

def int_from_AA_code(key):
    """
    This function returns the integer part from a residue name.

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
    key : string
        Residue name passed as a string, example "GLU30"

    Returns
    -------
    name: str
        Name of the residue, like "GLU" or "E" for "GLU30" or "E30", respectively

    """
    return ''.join([ii for ii in key if ii.isalpha()])

def shorten_AA(AA, substitute_fail=None):
    r"""
    return the short name of an AA, e.g. TRP30 to Y30 by trying to
    use either the :obj:`mdtraj.Topology.Residue.code' attribute
    or :obj:`mdtraj` internals AA dictionary

    Parameters
    ----------
    AA: :obj:`mdtraj.Topology.Residue` or a str
        The residue in question

    substitute_fail: str, default is None
        If there is no .code  attribute, different options are there
        depending on the value of this parameter

        * 'long' : keep the residue's long name, i.e. do nothing
        * 'c': any alphabetic character, as long as it is of len=1
        * None: throw an exception when no short code is found

    Returns
    -------
    code: str
        A string representing this AA using the short code

    """
    key_not_found = False
    if isinstance(AA,str):
        try:
            return '%s%u'%(_AMINO_ACID_CODES[name_from_AA(AA)],int_from_AA_code(AA))
        except KeyError:
            key_not_found=True
    else:
        try:
            return '%s%u'%(AA.code,AA.resSeq)
        except:
            key_not_found=True
            #raise NotImplementedError
    if key_not_found:
        if substitute_fail is None:
            raise KeyError("There is no short version for your input %s (%s)"%(AA, type(AA)))
        elif isinstance(substitute_fail,str):
            if substitute_fail.lower()=="long":
                return str(AA)
            elif len(substitute_fail)==1:
                return '%s%u'%(substitute_fail,int_from_AA_code(str(AA)))
            else:
                raise ValueError("Cannot understand your input. Please refer to the docs.")


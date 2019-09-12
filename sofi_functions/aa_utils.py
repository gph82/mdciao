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
import numpy as _np

#TODO refactor function name
#TODO enter npoints directly
#TODO rename circle?
#TODO rename offset->start
def angulate_segments(array_in, circle=360, offset=0, clockwise=True):
    r"""Return the angular values for spreading len(array_in) points equidistantly in a circle

    Note
    ----
    :obj:`offset` =0 and :obj:`circle` =360 will **not** place return a
    position at 360, since 0 **is** 360

    Parameters
    ----------
    array_in : np.ndarray
    circle : float, default is 360
        size of the circle in degrees
    offset : float, default is 0
        Where the circle is supposed to start
    clockwise : boolean, default is True
        In which direction the angle array grows.

    Returns
    -------
    angles : np.ndarray of len(array_in)

    """
    npoints = len(array_in)
    dang = circle / npoints
    if clockwise:
        fac=-1
    else:
        fac=1
    angles = _np.hstack([fac*(ii * dang)+offset for ii in range(npoints)])
    return angles

def fragment_selection_parser(fragments,
                              hide_fragments=None,
                              only_show_fragments=None):
    r"""
    Return a lambda that will decide whether a residue pair should be shown or not based on a fragment selection.

    Note
    ----
    At least one of :obj:`mute_fragments` or :obj:`anchor_fragment` should
    be None, i.e. the user can either make no choice (==> all residue pairs
    should be plotted) or select a subset of residue pairs to show by
    either `muting` some or `showing' some fragments, not muting or showing
    simultaneously

    Parameters
    ----------
    fragments : iterable
        Fragment list, each item is itself an iterable of residue indices
    hide_fragments : iterable, default is None
        subset of :obj:`fragments` to mute, i.e. residues belonging
        to these fragments will be deleted from the output
    only_show_fragments : iterable, default is None
        show  subset of :obj:`fragments` to force-show, i.e. residues belonging
        to these fragments will be shown even if they are originally
        muted

    Returns
    -------
    condition : lambda : residx_pair -> bool
        Take a residx_pair and tell whether it should be plotted
        according to the fragment selection
    """

    if hide_fragments is None and only_show_fragments is None:
        condition = lambda res_pair : True
    elif hide_fragments is None and only_show_fragments is not None:
        residxs = _np.hstack([fragments[iseg] for iseg in only_show_fragments])
        condition = lambda res_pair: _np.any([ires in residxs for ires in res_pair])
    elif hide_fragments is not None and only_show_fragments is None:
        residxs = _np.hstack([fragments[iseg] for iseg in hide_fragments])
        condition = lambda res_pair: _np.all([ires not in residxs for ires in res_pair])
    else:
        raise Exception("This input is not possible")

    return condition

import numpy as _np

#TODO rename circle?
#TODO rename offset->start
def regspace_angles(npoints, circle=360, offset=0, clockwise=True):
    r"""Return the angular values for spreading n points equidistantly in a circle

    Note
    ----
    The goal of this is to work together with :obj:`cartify_segments`, s.t.
     * offset=0 puts the first "angle" at 0 s.t that the first cartesian
     pair will be (1,0) and then clockwise on (0,-1) etc (see :obj:`pol2cart`)
     * even if :obj:`circle`=360, the last entry of the returned :obj:`angles`
       will not be 360, regardless of the :obj:`offset`

    Parameters
    ----------
    n : integer
        number of points
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
    npoints
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
    At least one of :obj:`hide_fragments` or :obj:`only_show_fragments` should
    be None, i.e. the user can either make no choice (==> all residue pairs
    should be plotted) or select a subset of residue pairs to show by
    either `hiding` some or `only_showing' some fragments, but not muting or showing
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
        raise ValueError("'hide_fragments' and 'only_show_fragments' can't simultaneously be 'None'")

    return condition

def cartify_segments(fragments,
                     r=1.0,
                     return_angles=False,
                     angle_offset=0,
                     padding_initial=0,
                     padding_final=0,
                     padding_between_fragments=0):
    r"""
    Cartesian coordinates on a circle of radius :obj:`r` for each sub-item in the lists of :obj:`fragments`

    Note
    ----
    works internally in radians

    Parameters
    ----------
    fragments: iterable of iterables
        They represent the fragments
    r : float, radius
    return_angles : bool, default is False
    angle_offset : float
        Degrees. Where in the circle the first sub-item in of
        the first item of :obj:`fragments` should start. Default
        is 0 which places the first point at xy = r,0 (3 o'clock)
    padding_initial : int, default is 0
        Add these many empty positions at the
        beginning of the position array
    padding_final : int, default is 0
        Add these many empty positions at the end
        of the position array
    padding_between_fragments : int, default is 0
        Add these many empty positions between fragments
    Returns
    -------

    """

    padding_initial = [None for ii in range(padding_initial)]
    padding_final = [None for ii in range(padding_final)]

    tostack = [_np.hstack(([None] * padding_between_fragments, iseg)) for iseg in fragments]

    if len(padding_initial) > 0:
        tostack = padding_initial + tostack
    if len(padding_final) > 0:
        tostack += padding_final

    spaced_segments = _np.hstack(tostack)
    spaced_angles = regspace_angles(len(spaced_segments),
                                    circle=2 * _np.pi,
                                    offset=angle_offset * _np.pi / 180
                                    )
    angles = _np.hstack([spaced_angles[ii] for ii, idx in enumerate(spaced_segments) if idx is not None])
    xy = pol2cart(_np.ones_like(angles) * r, angles)

    if not return_angles:
        return _np.vstack(xy).T
    else:
        return _np.vstack(xy).T, angles

def pol2cart(rho, phi):
    r"""
    Polar to cartesian coordinates. Angles in radians

    Parameters
    ----------
    rho : float
    phi : float

    Returns
    -------
    x, y
    """
    x = rho * _np.cos(phi)
    y = rho * _np.sin(phi)

    return (x, y)
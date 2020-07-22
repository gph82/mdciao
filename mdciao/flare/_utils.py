import numpy as _np
from mdciao.plots.plots import _colorstring
from mdciao.utils.bonds import bonded_neighborlist_from_top

_mycolors = _colorstring.split(",")

from bezier import Curve as _BZCurve
class my_BZCURVE(_BZCurve):
    """
    Modified Bezier curve to plot with line-width
    """

    def plot(self, num_pts, color=None, alpha=None, ax=None,lw=1, zorder=None):
        """Plot the current curve.

        Args:
            num_pts (int): Number of points to plot.
            color (Optional[Tuple[float, float, float]]): Color as RGB profile.
            alpha (Optional[float]): The alpha channel for the color.
            ax (Optional[matplotlib.artist.Artist]): matplotlib axis object
                to add plot to.

        Returns:
            matplotlib.artist.Artist: The axis containing the plot. This
            may be a newly created axis.

        Raises:
            NotImplementedError: If the curve's dimension is not ``2``.
        """
        from bezier import _plot_helpers
        if self._dimension != 2:
            raise NotImplementedError(
                "2D is the only supported dimension",
                "Current dimension",
                self._dimension,
            )

        s_vals = _np.linspace(0.0, 1.0, num_pts)
        points = self.evaluate_multi(s_vals)
        if ax is None:
            ax = _plot_helpers.new_axis()
        ax.plot(points[0, :], points[1, :], color=color, alpha=alpha, lw=lw, zorder=zorder)
        return ax

def create_flare_bezier(nodes, center=None):
    middle = _np.floor(len(nodes) / 2).astype("int")
    if center is not None:
        nodes = _np.vstack((nodes[:middle], center, nodes[middle:]))

    return my_BZCURVE(nodes.T, degree=2)
    #return _bezier.Curve(nodes.T, degree=3)

#TODO rename circle?
#TODO rename offset->start
def regspace_angles(npoints, circle=360, offset=0, clockwise=True):
    r"""Return the angular values for spreading n points equidistantly in a circle

    Note
    ----
    The goal of this is to work together with :obj:`cartify_fragments`, s.t.
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

def cartify_fragments(fragments,
                      r=1.0,
                      return_angles=False,
                      angle_offset=0,
                      **padding_kwargs):
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
    padding_kwargs : optional keyword args for :obj:`pad_fragment_positions`

    Returns
    -------

    """

    padded_fragment_list = pad_fragment_positions(fragments,**padding_kwargs)
    spaced_segments = _np.hstack(padded_fragment_list)
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

def pad_fragment_positions(fragments,
                           padding_initial=0,
                           padding_final=0,
                           padding_between_fragments=0,
                           pad_value=None):
    r"""
    Provided a list of fragments, introduce a number of :obj:`pad_value` between them

    Parameters
    ----------
    fragments : list of iterables
    padding_initial : int, default is 0
        Number of padding values to put at the very beginning of the fragment list
    padding_final : int, default is 0
        Number of padding values to put at the very end of the fragment list
    padding_between_fragments : int, default is 0
        Number of padding values to put at the beginning of each fragment
    pad_value : arbitryry, default is None
        What value to pad with (None, np.NaN, 0, "X") etc

    Returns
    -------
    padded_fragments : list
    """
    padding_initial = [pad_value for ii in range(padding_initial)]
    padding_final = [pad_value for ii in range(padding_final)]
    padded_frags = [_np.hstack(([pad_value] * padding_between_fragments, iseg)) for iseg in fragments]

    if len(padding_initial) > 0:
        padded_frags = padding_initial + padded_frags
    if len(padding_final) > 0:
        padded_frags += padding_final

    return padded_frags

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


# todo CLEAN THIS COLOR MESS
def col_list_from_input_and_fragments(colors,
                                      res_idxs,
                                      fragments=None):
    r"""
    Build a usable color list, taking fragments into account

    Note:
    ATM the return behaviour is VERY!!!! inconsistent, i believe
    is for compatibility reasons
    Sometimes the return value is of len(list_of...) and sometimes
    of len(fragments)
    #todo

    Parameters
    ----------
    colors : can be of different types
        * False
            All returned colors will be "tab:blue"
        * True
            All returned colors will differ by fragment
        * string (anything matplotlib can understand as color)
            Use this color for all residues
        * iterable (not dict)
            Has to be of len(list_of_non_zero_residue_idxs)
            Nothing happens, the list is done already
        * iterable (dict)
            Has to be of len(fragments)

    fragments : iterable of iterable of indices, default is None
        Needs to contain the set of :obj:`res_idxs`, will
        only one fragment assumed is fragments is None
    res_idxs

    Returns
    -------
    list of colors....of variable length??? TODO

    """

    if fragments is None:
        fragments = [res_idxs]

    if isinstance(colors, bool):
        if colors:
            jcolors = _np.tile(_mycolors, _np.ceil(len(fragments) / len(_mycolors)).astype("int"))
            col_list = _np.hstack([[jcolors[ii]] * len(iseg) for ii, iseg in enumerate(fragments)])
        else:
            col_list = [_mycolors[0] for __ in range(len(res_idxs))]

    elif isinstance(colors, str):
        col_list = [colors for __ in range(len(res_idxs))]

    elif isinstance(colors, (list, _np.ndarray)):
        assert len(colors) == len(res_idxs), (len(colors), len(res_idxs))
        col_list = colors

    elif isinstance(colors, dict):
        assert len(colors) == len(fragments)
        col_list = _np.hstack([[val] * len(iseg) for val, iseg in zip(colors.values(), fragments)])
    else:
        raise Exception

    return col_list

def curvify_segments(segments, r=1.0, angle_offset=0, padding=0):

    total_n_points = len(_np.hstack(segments))
    #print(total_n_points)
    factor_to_720 = 720/total_n_points
    if factor_to_720<=1:
        factor_to_720=1
    else:
        factor_to_720 = _np.ceil(720/total_n_points).round().astype(int)
    #print(total_n_points, factor_to_720)
    spaced_segments = _np.hstack([_np.hstack(([None]*factor_to_720, _np.repeat(iseg, factor_to_720))) for iseg in segments] + [None for ii in range(padding*factor_to_720)])
    angles = regspace_angles(spaced_segments, circle=2 * _np.pi, offset=angle_offset * _np.pi / 180)

    xy = pol2cart(_np.ones_like(angles) * r, angles)
    xy = _np.vstack(xy).T

    frag_curves_as_list_of_xy_pairs = [[] for ii in range(len(segments))]
    last_val = None
    current_frag=-1
    for ii, (ival, ixy) in enumerate(zip(spaced_segments,xy)):
        if ival is not None:
            if last_val is None:
                current_frag+=1
            frag_curves_as_list_of_xy_pairs[current_frag].append(ixy)
        last_val = ival
        #print(ii, current_frag, ival)

    #TODO shamelessly off-by-oneing this one
    return [_np.vstack(ifrag[:-padding]) for ifrag in frag_curves_as_list_of_xy_pairs]

def draw_arcs(fragments, iax, colors=None,
              lw=1,
              center=0, r=1, angle_offset=0, **cartify_kwargs):

    if colors is None:
        colors = ["k"]*len(fragments)
    angles = cartify_fragments(fragments,r=r, return_angles=True, **cartify_kwargs)[1]*180/_np.pi
    from mdciao.utils.lists import re_warp
    angles = re_warp(angles,[len(ifrag) for ifrag in fragments])
    from matplotlib.patches import Arc as _Arc
    for ii, iang in enumerate(angles):
        iarc = _Arc(center,
                    width=r,
                    height=r,
                    angle=angle_offset,
                    theta1=iang[-1],
                    theta2=iang[0],
                    lw=lw,
                    color=colors[ii],
                    )
        iax.add_artist(iarc)

def should_this_residue_pair_get_a_curve(
                                         fragments,
                                         mute_fragments=None,
                                         anchor_fragments=None,
                                         top=None,
                                         pair_selection=None,
                                         exclude_neighbors=0,
                                         ):
    r"""
    Discriminate whether there should even be a bezier curve for
    this contact based on a handful of criteria

    Parameters
    ----------
    pair_selection : list of pairs
    fragments
    mute_fragments
    anchor_fragments
    top

    Returns
    -------
    lambda : pair -> boolean
    """
    # Condition vicinities
    if pair_selection is not None:
        is_pair_not_muted_bc_direct_selection = lambda pair: len(_np.intersect1d(pair, pair_selection)) > 0
    else:
        is_pair_not_muted_bc_direct_selection = lambda pair: True

    # Condition in anchor segment or in muted segment
    is_pair_not_muted_bc_anchor_and_mute_segments = \
        fragment_selection_parser(fragments,
                                  hide_fragments=mute_fragments,
                                  only_show_fragments=anchor_fragments)

    # Condition not nearest neighbors
    if top is None:
        is_pair_not_muted_bc_nearest_neighbors = lambda pair: _np.abs(pair[0] - pair[1]) > exclude_neighbors
    else:
        nearest_n_neighbor_list = bonded_neighborlist_from_top(top, exclude_neighbors)
        is_pair_not_muted_bc_nearest_neighbors = lambda pair: pair[1] not in nearest_n_neighbor_list[pair[0]]

    lambda_out = lambda res_pair : is_pair_not_muted_bc_nearest_neighbors(res_pair) and \
                                   is_pair_not_muted_bc_anchor_and_mute_segments(res_pair) and \
                                   is_pair_not_muted_bc_direct_selection(res_pair)

    return lambda_out

def add_fragment_names(iax, xy,
                       fragments,
                       fragment_names,
                       residx2xyidx,
                       fontsize=5,
                       center=0,
                       r=1.0):
    r"""
    Add fragment names to a flareplot

    Parameters
    ----------
    iax : :obj:`matplotlib.Axes`
    xy : iterable of (x,y) tuples
    fragments : iterable if iterables of ints
    fragment_names  :iterable of strs, len(fragments)
    residx2xyidx : np.ndarray
        map to use idxs of :obj:`fragments` on :obj:`xy`,
        since almost always these will never coincide

    Returns
    -------
    fragment_labels : list

    """

    fragment_labels = []
    for seg_idxs, iname in zip(fragments, fragment_names):
        xseg, yseg = xy[residx2xyidx[seg_idxs]].mean(0) - center
        rho, phi = cart2pol(xseg, yseg)
        xseg, yseg = pol2cart(r , phi) + center

        iang = phi + _np.pi / 2
        if _np.cos(iang) < 0:
            iang = iang + _np.pi

        fragment_labels.append(iax.text(xseg, yseg, iname, ha="center", va="center",
                                        fontsize=fontsize,
                                        rotation=_np.rad2deg(iang)))

    return fragment_labels
def cart2pol(x, y):
    rho = _np.sqrt(x**2 + y**2)
    phi = _np.arctan2(y, x)
    return(rho, phi)

def add_residue_labels(iax,
                       xy_labels,
                       xy_angles,
                       res_idxs,
                       fontsize,
                       highlight_residxs=None,
                       top=None,
                       shortenAAs=True,
                       aa_offset=0,
                       colors=None,
                       forced_labels=None,
                       **text_kwargs
                       ):
    r"""
    Add residue names to a flareplot

    Parameters
    ----------
    iax : :obj:`matplotlib.Axes`
    res_idxs : np.ndarray
        The residue indices to use as labels
    xy_labels : 2D-np.ndarray
    xy_angles : 2D-np.narray
    fontsize : int
    highlight_residxs: iterable, default is None
        Highlight these labels in red
    top : :obj:`mtdtraj.Topology`, default is None
        Use the idxs in :obj:`res_idxs` on this
        topology to generate better labels
    shortenAAs: bool, default is True
        Use the short-code for the AAS (GLU30->E30)
    aa_offset : int, default is 0
        In case the resSeq idxs of the topologies
        don't match the desired sequence, provide
        and offset here
    colors : list of len(res_idxs), default is None,
        Individual residue colors, default is black
    forced_labels : list of len(res_idxs), defaut is None
        Individual residue labels, default is None
        Overwrite everything and use these labels

    Returns
    -------
    labels : list of text objects

    """
    assert len(res_idxs) == len(xy_labels) == len(xy_angles)

    if colors is None:
        colors = ['k']*len(res_idxs)
    else:
        assert len(colors)==len(res_idxs)

    if forced_labels is not None:
        assert len(forced_labels)==len(res_idxs)

    labels = []
    for ii, (res_idx, ixy, iang) in enumerate(zip(res_idxs, xy_labels, xy_angles)):
        if _np.cos(iang) < 0:
            iang = iang + _np.pi
        ilabel = res_idx
        txtclr = colors[ii]

        if top is not None:
            if not shortenAAs:
                ilabel = top.residue(res_idx)
            else:
                idxs = top.residue(res_idx).resSeq + aa_offset
                ilabel = ("%s%u" % (top.residue(res_idx).code, idxs)).replace("None", top.residue(res_idx).name)
            if highlight_residxs is not None and res_idx in highlight_residxs:
                txtclr = "red"
        if forced_labels is not None:
            ilabel = forced_labels[ii]

        itxt = iax.text(ixy[0], ixy[1], '%s' % ilabel,
                        color=txtclr,
                        va="center",
                        ha="center",
                        rotation=_np.rad2deg(iang),
                        fontsize=fontsize,
                        zorder=20,
                        **text_kwargs)
        labels.append(itxt)

    return labels


_SS2vmdcol = {'H':"purple", "E":"yellow","C":"cyan", "NA":"gray"}


def residx2dotidx(residues_to_plot_as_dots):
    residx2markeridx = _np.zeros(_np.max(residues_to_plot_as_dots) + 1, dtype=int)
    residx2markeridx[:] = _np.nan
    residx2markeridx[residues_to_plot_as_dots] = _np.arange(len(residues_to_plot_as_dots))

    return residx2markeridx


# DIrectly lifted from my molpx, coudl go with _dataunits2points prolly
def pts_per_axis_unit(iax, pt_per_inch=72):
    r"""
    Return how many pt per axis unit of a given maptplotlib axis a figure has

    Parameters
    ----------

    iax : :obj:`matplotlib.axes._subplots.AxesSubplot`

    pt_per_inch : how many points are in an inch (this number should not change)

    Returns
    --------

    pt_per_xunit, pt_per_yunit

    """

    # matplotlib voodoo
    # Get bounding box
    bbox = iax.get_window_extent().transformed(iax.get_figure().dpi_scale_trans.inverted())

    span_inch = _np.array([bbox.width, bbox.height], ndmin=2).T

    span_units = [iax.get_xlim(), iax.get_ylim()]
    span_units = _np.diff(span_units, axis=1)

    inch_per_unit = span_inch / span_units
    return inch_per_unit * pt_per_inch

def markersize_scatter(n,l, iax):
    r = l / (2 * n)
    ppau = _np.mean(pts_per_axis_unit(iax))
    rau = r * ppau
    s = (2 * rau) ** 2

    return s
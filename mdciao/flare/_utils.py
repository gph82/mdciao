##############################################################################
#    This file is part of mdciao.
#    
#    Copyright 2020 Charité Universitätsmedizin Berlin and the Authors
#
#    Authors: Guillermo Pérez-Hernandez
#    Contributors:
#
#    mdciao is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    mdciao is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with mdciao.  If not, see <https://www.gnu.org/licenses/>.
##############################################################################

import numpy as _np

from ._textutils import \
    outermost_text_corner as _outermost_corner_of_fancypatches, \
    any_overlap_via_FancyBoxPach

from mdciao.plots.plots import _colorstring
from mdciao.utils.bonds import bonded_neighborlist_from_top
from mdciao.utils.lists import \
    assert_no_intersection as _no_intersect, \
    re_warp as _re_warp, \
    find_parent_list as _find_parent_list
from mdciao.fragments import assign_fragments
from mdciao.utils.str_and_dict import replace4latex as _replace4latex
from matplotlib.colors import to_rgb as _to_rgb
from matplotlib.collections import LineCollection as _LCol

from collections import Counter as _Counter, defaultdict as _defdict
from matplotlib.patches import CirclePolygon as _CP, Polygon as _PG

#https://stackoverflow.com/a/33375738
def _make_color_transparent(color_fg, alpha, color_bg="white"):
    r"""Return a color that's a three-value RGB equivalent to a four valued (rgb+alpha) transparent color"""
    return [alpha * c1 + (1 - alpha) * c2 for (c1, c2) in zip(_to_rgb(color_fg), _to_rgb(color_bg))]

_mycolors = _colorstring.split(",")

from bezier import Curve as _BZCurve
class my_BZCURVE(_BZCurve):
    """
    Modified Bezier curve to plot with line-width and to have self.Line2D
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
        self._Line2D = ax.plot(points[0, :], points[1, :], color=color, alpha=alpha, lw=lw, zorder=zorder)[0]
        return ax

    @property
    def Line2D(self):
        return self._Line2D

def create_flare_bezier(nodes, center=None):
    middle = _np.floor(len(nodes) / 2).astype("int")
    if center is not None:
        nodes = _np.vstack((nodes[:middle], center, nodes[middle:]))
    return my_BZCURVE(nodes.T, degree=2)

def create_flare_bezier_2(nodes,
                        center):
    r"""

    Parameters
    ----------
    nodes : 2D np.ndarray
        The x-y positions of
        the nodes to be joined
        via a Bezier curve
    center : pair of floats
        x-y coordinate of the center

    Returns
    -------

    bzc : :obj:`my_BZCURVE`

    """
    assert len(nodes)==2
    nodes = _np.vstack((_np.array((nodes[0], center, nodes[1]),dtype="float")))
    return my_BZCURVE(nodes.T, degree=2)

def regspace_angles(npoints, circle=360, clockwise=True):
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
    angles = _np.hstack([fac*(ii * dang) for ii in range(npoints)])
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
                                    )+angle_offset * _np.pi / 180
    angles = _np.hstack([spaced_angles[ii] for ii, idx in enumerate(spaced_segments) if idx is not None])
    xy = pol2cart(_np.ones_like(angles) * r, angles)

    if not return_angles:
        return _np.vstack(xy).T
    else:
        return _np.vstack(xy).T, angles

def pad_fragment_positions(fragments,
                           padding=[0,0,0],
                           pad_value=None):
    r"""
    Provided a list of fragments, introduce a number of :obj:`pad_value` between them

    Parameters
    ----------
    fragments : list of iterables
    padding : list of ints, default is [0,0,0]
        Each integer controls the number of padding values used:
        * at the very beginning of the fragment list
        * at the beginning of each fragment
        * at the very end of the fragment list

    pad_value : arbritary, default is None
        What value to pad with (None, np.NaN, 0, "X") etc

    Returns
    -------
    padded_fragments : list
    """
    padding_initial = [pad_value for ii in range(padding[0])]
    padding_final = [pad_value for ii in range(padding[2])]
    padded_frags = [_np.hstack(([pad_value] * padding[1], iseg)) for iseg in fragments]

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
                                      residxs_as_fragments,
                                      default_color="gray",
                                      alpha=1,
                                      ):
    r"""
    per-residue color list taking possible fragmentation into account


    Parameters
    ----------
    colors : can be of different types
        * False or None
            All returned colors will be the default color
        * True
            All returned colors will differ by fragment
        * string (anything matplotlib can understand as color)
            Use this color for all residues
        * iterable (array or list, not dict)
            len(colors) has to be either len(fragments) or
            sum([len(frag) for frag in fragments)
            In the first case, it's expanded to assign
            each residue in fragment the same color
            In the second case, since each residue in
            the fragments (apparently) has already a color,
            nothing happens
        * iterable (dict)
            Has to be of len(residxs_as_fragments)
    res_idxs : iterable of ints or iterable thereof
        The residues for which to generate the
        colors. If an iterable of ints is passed,
        only one fragment is assumed

    Returns
    -------
    colors : list
        list of len(_np.hstack(fragments))

    """
    if colors is None:
        colors=False

    if isinstance(residxs_as_fragments[0], int):
        residxs_as_fragments = [residxs_as_fragments]

    if isinstance(colors, bool):
        if colors:
            to_tile = _mycolors
        else:
            to_tile = [default_color]
        jcolors = _np.tile(to_tile, _np.ceil(len(residxs_as_fragments) / len(to_tile)).astype("int"))
        col_list = _np.hstack([[jcolors[ii]] * len(iseg) for ii, iseg in enumerate(residxs_as_fragments)])
    elif isinstance(colors, str):
        to_tile = [colors]
        jcolors = _np.tile(to_tile, _np.ceil(len(residxs_as_fragments) / len(to_tile)).astype("int"))
        col_list = _np.hstack([[jcolors[ii]] * len(iseg) for ii, iseg in enumerate(residxs_as_fragments)])

    elif isinstance(colors, (list, _np.ndarray)):
        if len(colors) == len(residxs_as_fragments):
            col_list = _np.vstack([_np.vstack([colors[ii]]*len(fr)) for ii, fr in enumerate(residxs_as_fragments)]).squeeze().tolist()
        elif len(colors) == len(_np.hstack(residxs_as_fragments)):
            col_list = colors
        else:
            raise ValueError("The number of input colors (%u) "
                             "doesn't match the number of fragments (%u) "
                             "or the number of residues (%u) "%(len(colors),
                                                                len(residxs_as_fragments),
                                                                len(_np.hstack(residxs_as_fragments))))
    elif isinstance(colors, dict):
        assert len(colors) == len(residxs_as_fragments), (len(colors), len(residxs_as_fragments))
        col_list = [[val] * len(iseg) for val, iseg in zip(colors.values(), residxs_as_fragments)]
        col_list = _np.array([item for sublist in col_list for item in sublist])
        #col_list = _np.hstack([[val] * len(iseg) for val, iseg in zip(colors.values(), residxs_as_fragments)])
    else:
        raise Exception

    if alpha!=1:
        col_list = [_make_color_transparent(_to_rgb(col), alpha) for col in col_list]

    return col_list

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
                                         select_residxs=None,
                                         exclude_neighbors=0,
                                         ):
    r"""
    lambda for selecting and/or muting residue pairs

    A residue pair gets muted when at least one condition is met,
    even if another would "un-mute" it.

    Parameters
    ----------
    fragments : iterable if iterable of ints
        How residue idxs are divided into fragments.
        Any residue not contained here will
        not get a curve.
    mute_fragments : iterable of ints, default is None
        Idxs of fragments to be muted: any residue pair
        containing at least one residue in these fragments
        will be muted.
        Cannot be provided simultaneously with :obj:`anchor_fragments`
    anchor_fragments : iterable of ints, default is None
        Complementary of :obj:`mute_fragments`: any residue
        pair not containing at least one residue
        in these fragments will be muted.
        Cannot be provided simultaneously with :obj:`mute_fragments`
    select_residxs : list of idxs
        When provided, any pair *not* containing at least
        one of these residues will be muted
    exclude_neighbors : int, default is 0
        pairs with neighbors up to :obj:`exclude_neighbors`
        will be excluded. When no :obj:`top` is provided,
        the neighborhood condition is computed by
        substraction of the idxs themselves, i.e.
        [20-21] are 1-neighbors regardless of
        whether there is a bond or not connecting
        them
    top : :obj:`mdtraj.Topology`
        For implementing :obj:`exclude_neighbors` with
        topology (=bond) information, see :obj:`bonded_neighborlist_from_top`

    Returns
    -------
    lambda : pair -> boolean
        One can apply this lambda to any residue pair and it will return
        False if any of the muting conditions apply, otherwise True
    """

    is_pair_not_muted_bc_missing_in_fragment_def = lambda pair : len(_np.intersect1d(pair,_np.hstack(fragments)))==2

    # Condition residues
    if select_residxs is not None:
        is_pair_not_muted_bc_direct_selection = lambda pair: len(_np.intersect1d(pair, select_residxs)) > 0
    else:
        is_pair_not_muted_bc_direct_selection = lambda pair: True

    # Condition in anchor segment or in muted segment
    is_pair_not_muted_bc_anchor_and_mute_fragments = \
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
                                   is_pair_not_muted_bc_anchor_and_mute_fragments(res_pair) and \
                                   is_pair_not_muted_bc_direct_selection(res_pair) and \
                                   is_pair_not_muted_bc_missing_in_fragment_def(res_pair)

    return lambda_out

def add_fragment_labels(fragments,
                        fragment_names,
                        iax,
                        angle_offset=0,
                        padding=[0,0,0],
                        fontsize=5,
                        center=[0,0],
                        r=1.0,
                        xy=None):
    r"""
    Add fragment labels to a flareplot

    Very similar to :obj:`add_residue_labels` but does not
    "radiate" the labels, it puts them in the angular
    center of the residues contained in each fragment

    Parameters
    ----------
    fragments : iterable if iterables of ints
        The fragment definitions
    fragment_names  :iterable of strs, len(fragments)
        The fragment names
    iax : :obj:`~matplotlib.axes.Axes`
        The axis to write on
    angle_offset : scalar, scalar default is 0
        Where the circle starts, in degrees. 0 means 3 o'clock,
        90 12 o'clock etc. It's the phi of polar coordinates
    padding : list, default is [0,0,0]
        * first integer : Put this many empty positions before the first dot
        * second integer: Put this many empty positions between fragments
        * third integer : Put this many empty positions after the last dot
    fontsize : float, default is 5
        The fontsize in which the fragment names are
        written
    center : pair for floats, default is (0,0)
        The center of the flareplot
    r : scalar, default is 1
        The radius at which the labels will be put
    xy : list or 2D np.ndarray, None
        The positions of the members of each fragment,
        has to be of len==_np.hstack(fragments).
        If None, then positions will be computed by
        :obj:`cartify_fragments` using :obj:`r`,
        :obj:`angle_offset`, and :obj:`padding`.

    Returns
    -------
    fragment_labels : list of the :obj:`~matplotlib.text.Text` objects

    """
    if xy is None:
        _xy = cartify_fragments(fragments,
                                r=r,
                                angle_offset=angle_offset,
                                padding=padding)
        _xy += center
    else:
        assert len(xy)==len(_np.hstack(fragments))
        _xy = xy
    fragment_labels = []
    residx2xyidx = _re_warp(_np.arange(len(_np.hstack(fragments))), [len(ifrag) for ifrag in fragments])
    for seg_idxs, iname in zip(residx2xyidx, fragment_names):
        xseg, yseg = _xy[seg_idxs].mean(0) - center
        rho, phi = cart2pol(xseg, yseg)
        xseg, yseg = pol2cart(r , phi) + _np.array(center)

        ha="center"
        if yseg<0:
            va="top"
        else:
            va="bottom"
        iang = phi + _np.pi / 2
        if _np.cos(iang) < 0:
            iang = iang + _np.pi

        fragment_labels.append(iax.text(xseg, yseg, iname,
                                        ha=ha,
                                        va=va,
                                        bbox={"boxstyle": "square,pad=0.0", "fc": "none", "ec": "none", "alpha": .05},
                                        # we need this transparent Fancybox to check for overlaps
                                        fontsize=fontsize,
                                        rotation_mode="anchor",
                                        rotation=_np.rad2deg(iang)))

    return fragment_labels

def add_parent_labels(kwargs_freqs2flare, flareplot_attrs, kwargs_pffp):
    r"""
    Add the outer ring of "parent" fragment names to a flareplot that's using subdomains

    Note
    ----
    This is an ad-hoc **after** ContactGroup.plot_freqs_as_flareplot
    has called freqs2flare. Currently, I don't want to change
    that freqs2flare's signature. Hence, there's three different
    dictionaries organizing the input. This can/will change
    in the future

    Parameters
    ----------
    kwargs_freqs2flare : dict
        The dictionary of kwargs computed inside
        plot_freqs_as_flareplot  by using
        _dataframe2flarekwargs and
        then sent to freqs2flare
    flareplot_attrs : dict
        The dictionary of flareplot attributes
        produced by freqs2flare.
    kwargs_pffp : dict
        The optional arguments passed
        initially to plot_freqs_as_flareplot.
        It has to contain the keys "fragments"
        and "fragment_names"
    Returns
    -------
    flareplot_attrs : dict
        The same input dictionary with
        one new key "parent_labels"
    """
    if None in [kwargs_pffp.get("fragments"), kwargs_pffp["fragment_names"]]:
        pass
    else:
        not_passed_frags = list(set(kwargs_pffp["fragment_names"]).difference(kwargs_freqs2flare["fragment_names"]))
        # some original parents weren't passed to freqs2flare. This might be
        # 1) because the were excluded completely
        # 2) they were broken down into subfragments
        if not_passed_frags:
            # Let's find out if if we're in number 2)
            # First, we need to associate actual Text-objects
            # with actually passed kwargs_freqs2flare["fragments"]
            # since even some kwargs_freqs2flare["fragments"] might've
            # been excluded by freqs2flare because of sparse-type arguments
            plotted_fragTexts_by_input_idx = _defdict(list)
            for ff, fn in enumerate(kwargs_freqs2flare["fragment_names"]):
                for tt, txt in enumerate(flareplot_attrs["fragment_labels"]):
                    if txt.get_text() == _replace4latex(fn):  # The text object has been latexified already
                        plotted_fragTexts_by_input_idx[ff].append(tt)

            # Assert uniqueness
            assert all([len(val) == 1 for val in plotted_fragTexts_by_input_idx.values()])
            plotted_fragTexts_by_input_idx = dict(plotted_fragTexts_by_input_idx)
            plotted_fragTexts_by_input_idx = {key: val[0] for key, val in plotted_fragTexts_by_input_idx.items()}
            parent_labels, parent_xy = [], []

            #Store the axis before it gets removed
            iax = flareplot_attrs["fragment_labels"][0].axes
            _, kid_by_parents = _find_parent_list(kwargs_freqs2flare["fragments"], kwargs_pffp["fragments"])
            for parent_idx, kids in kid_by_parents.items():
                present_kids_by_TextObject_idx = [plotted_fragTexts_by_input_idx[kk] for kk in kids if kk in plotted_fragTexts_by_input_idx.keys()]
                if present_kids_by_TextObject_idx:
                    present_TextObjects = [flareplot_attrs["fragment_labels"][idx] for idx in present_kids_by_TextObject_idx]
                    present_weights =     [len(kwargs_freqs2flare["fragments"][kk]) for kk in kids if kk in plotted_fragTexts_by_input_idx.keys()]
                    assert len(present_weights)==len(present_TextObjects)
                    parent_xy.append(_np.average([tt.get_position() for tt in present_TextObjects],
                                                 axis=0, weights=present_weights))
                    parent_labels.append(kwargs_pffp["fragment_names"][parent_idx])
                    if len(present_TextObjects)==1 and present_TextObjects[0].get_text()==_replace4latex(parent_labels[-1]):
                        present_TextObjects[0].remove()

            if parent_labels:
                parent_labels = add_fragment_labels([[None]]*len(parent_labels),
                                                    [_replace4latex(str(ifrag)) for ifrag in
                                                     parent_labels],
                                                    iax,
                                                    fontsize=flareplot_attrs["fragment_labels"][0].get_fontsize() * 2,
                                                    r=flareplot_attrs["r"]+flareplot_attrs["dots"][0].radius*4,
                                                    xy=_np.array(parent_xy))
                flareplot_attrs["parent_labels"] = parent_labels

    return flareplot_attrs

def change_axlims_and_resize_Texts(iax, new_r, center=[0, 0]):
    r"""
    Change the x- and y-lims of a square plot and scale the Text objects to the new limits

    In matplotlib, fontsizes of Text-objects are constant in pts.
    Changing the axis limits doesn't affect the text size, but it
    affects their position. This means that on "zooming out", the
    text objects get closer to one another, potentially causing
    overlaps. This method scales the fontsize down by the amount
    of lost distance between text-object, effectively making them
    keep their size in axis data units

    Parameters
    ----------
    iax
    new_r
    center

    Returns
    -------

    """
    old_d_x = _np.abs(_np.diff(iax.get_xlim()))
    old_d_y = _np.abs(_np.diff(iax.get_ylim()))
    assert iax.get_aspect()==1
    assert old_d_x==old_d_y
    iax.set_xlim([center[0] - new_r, center[0] + new_r])
    iax.set_ylim([center[1] - new_r, center[1] + new_r])
    new_d = _np.abs(_np.diff(iax.get_xlim()))
    [lab.set_fontsize(lab.get_fontsize() * (old_d_x / new_d)) for lab in iax.texts]

def cart2pol(x, y):
    rho = _np.sqrt(x**2 + y**2)
    phi = _np.arctan2(y, x)
    return(rho, phi)

def add_residue_labels(iax,
                       xy_labels,
                       res_idxs,
                       fontsize,
                       center=None,
                       top=None,
                       shortenAAs=True,
                       aa_offset=0,
                       colors=None,
                       highlight_residxs=None,
                       replacement_labels=None,
                       **text_kwargs
                       ):
    r"""
    Add residue names to a flareplot

    Parameters
    ----------
    iax : :obj:`~matplotlib.axes.Axes`
    res_idxs : np.ndarray
        The residue indices to use as labels
    xy_labels : 2D-np.ndarray
        x-y position of the labels
    center: pair of floats, default is None
        When provided, this is the center
        of the circle that the :obj:`xy_labels`
        are assumed to lie on. The polar rho-angle
        will be computed and passed as :obj:`rotation`
        argument to the :obj:`matplotlib.text` call,
        making residue labels "radiate"
        out of the :obj:`center`.
    fontsize : int
    top : :obj:`mtdtraj.Topology`, default is None
        Use the idxs in :obj:`res_idxs` on this
        topology to generate better labels
    shortenAAs: bool, default is True
        Use the short-code for the AAS (GLU30->E30)
    aa_offset : int, default is 0
        Add this number to the resSeq value
    colors : list of len(res_idxs), default is None,
        Individual residue colors, default is black
    replacement_labels : dict, default is None
        Input individual replacements for residue labels here,
        keyed with residue idxs
        Typical cases could be a mutated residue that you want
        to show as R38A instead of just A38, or use
        e.g. GPCR or CGN consensus labels.
    highlight_residxs : iterable of ints, default is None
        In case you don't want to construct a whole
        color list for :obj:`colors`, you can simply
        input a subset of :obj:`res_idxs` here and
        they will be shown in red.


    Returns
    -------
    labels : list of :obj:`matplotlib.text.Text` objects

    """
    assert len(res_idxs) == len(xy_labels)

    if colors is None:
        colors = ['k']*len(res_idxs)
    else:
        assert len(colors)==len(res_idxs)

    if replacement_labels is None:
        replacement_labels = {}

    labels = []
    for ii, (res_idx, ixy) in enumerate(zip(res_idxs, xy_labels)):
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

        try:
            ilabel = replacement_labels[res_idx]
        except KeyError:
            pass # the residue does not have a replacement

        rotation = 0
        ha="left"
        if center is not None:
            rho, phi = cart2pol(ixy[0] - center[0], ixy[1] - center[1])
            if _np.cos(phi) < 0:
                phi = phi + _np.pi
                ha = "right"
            rotation = phi
        itxt = iax.text(ixy[0], ixy[1], '%s' % ilabel,
                        color=txtclr,
                        va="center",
                        ha=ha,
                        rotation=_np.rad2deg(rotation),
                        rotation_mode="anchor",
                        fontsize=fontsize,
                        zorder=20,
                        bbox={"boxstyle":"square,pad=0.0","fc":"none", "ec":"none", "alpha": .05},  # we need this transparent Fancybox to check for overlaps
                        **text_kwargs)
        labels.append(itxt)

    return labels


_SS2vmdcol = {'H':"purple", "E":"yellow","C":"cyan", "NA":"gray"}

def value2position_map(unique_idxs):
    r"""
    Return an dictionary v2p so that v2p[idx]=pos, where pos is the position of idx in unique_idxs

    Parameters
    ----------
    unique_idxs : iterable of ints
        No index can be repeated

    Returns
    -------
    value2pos : dict


    """
    values, positions = _np.unique(unique_idxs, return_index=True)
    assert len(values)==len(positions)
    value2pos = {key:val for key, val in zip(values, positions)}
    return value2pos

def overlappers(text_objects_1, text_objects_2):
    r"""
    Return a list of objects whose window_extents overlap in any way with any other object

    The self-overlap is ignored

    Parameters
    ----------
    text_objects_1 : list
        List of :obj:`matplotlib.text.Text` objects

    Returns
    -------
    overlappers : list
        The :obj:`matplotlib.text.Text` objects


    """

    boxes_1 = [t.get_window_extent(renderer=t.axes.figure.canvas.get_renderer()) for t in text_objects_1]
    # We could be computing the same boxes two times but makes for
    # better readability and more consistent return values
    boxes_2 = [t.get_window_extent(renderer=t.axes.figure.canvas.get_renderer()) for t in text_objects_2]
    return [t2 for t2, b2 in zip(text_objects_2,boxes_2) if any([(t1 is not t2 and b1.overlaps(b2)) for t1, b1 in zip(text_objects_1,boxes_1)])]

def overlappers_one_to_one(objects_1, objects_2,break_loop=True):
    r"""
    Check for overlaps of the i-th element of each list (no cross-pairs)


    Parameters
    ----------
    objects_1 : list
        List of objects with get_window_extent method

    objects_2 : list
        List of objects with get_window_extent method

    break_loop : bool, default is False
        Break the loop at the first overlap

    Returns
    -------
    overlappers : list
        List with the idxs of the pairs that overlap
        If break_loop is True, will contain only
        the first entry


    """

    assert len(objects_1)==len(objects_2)
    res = []
    for ii, (obj1,obj2) in enumerate(zip(objects_1,objects_2)):
        b1 = obj1.get_window_extent(renderer=obj1.axes.figure.canvas.get_renderer())
        b2 = obj2.get_window_extent(renderer=obj2.axes.figure.canvas.get_renderer())
        if b1.overlaps(b2):
            res.append(ii)
            if break_loop:
                break
    return res



def un_overlap_via_fontsize(text_objects, fac=.95, maxiter=50):
    r"""
    Iteratively (up to n_max) reduce the fontsize by :obj:`fac`
    until the text objects do not overlap anymore

    Parameters
    ----------
    text_objects : list
        List of :obj:`matplotlib.text.Text` objects
    fac : float, default is .95
        fontsize_ii_+1 = fontsize_ii * fac
    maxiter: int, default is 50
        Stop after these many iterations

    Returns
    -------

    """
    counter = 0
    while len(text_objects)>0 and any_overlap_via_FancyBoxPach(text_objects, text_objects) and counter < maxiter:
        [t1.set_size(t1.get_size() * fac) for t1 in text_objects]
        counter += 1

#TODO RENAME
def _parse_residue_and_fragments(res_idxs_pairs, sparse_residues=False,
                                 sparse_fragments=False, fragments=None, top=None,
                                 anchor_fragments=None, mute_fragments=None):
    r""" Decide what residues the user wants to get a dot.

    Typically, the output of this function will get parsed to :obj:`circle_plot_residues`

    Note
    ----
    I've outsourced this decision-making here because it's not clear to me yet
    what the best interplay in sparse_residues vs fragment could be

    Parameters
    ----------
    res_idxs_pairs : np.ndarray
        (N,2)-array with N residue pairs
    sparse_residues : boolean or iterable of ints, default is False
        * If False, return an array (0,np.max(res_idxs_pairs)+1)
        * If True,  return only np.unique(res_idxs_pairs)
        * If iterable of ints, override everything and return these indices
    sparse_fragments : bool, default is False
        If True, return all residues of the fragments where
        at least one residue is present in res_idxs_pairs
        An error will be raised if sparse_fragment is True
        and sparse_residues isn't False
    fragments : list, default is None
        List of integer lists representing how the residues
        are grouped together in fragments. Their union
        must always be a superset of whatever set resulted
        of the chosen :obj:`sparse_residues` option. If sparse_residues is False,
        residues present here will be added to the output even
        if they don't contain any relevant residue
    top : :obj:`~mdtraj.Topology`, default is None
        Only used if :obj:`fragments` is None
        and :obj:`sparse_residues` is False. Then, if
        top is None, dots will be assigned only up
        to max(res_idxs), and if top is passed, up to
         :obj:`top.n_residues`
    anchor_fragments : list, default is None
        In case this was passed as not None initially
        a re-indexing needs to happen here
    mute_fragments : list, default is None
        In case this was passed as not None initially
        a re-indexing needs to happen here

    Returns
    -------
    residues_as_fragments, anchor_fragments, mute_fragments
    """
    res_idxs = _np.unique(res_idxs_pairs)
    if sparse_fragments:
        assert sparse_residues is False
    if fragments is None:
        if isinstance(sparse_residues, bool):
            if sparse_residues:
                residues_as_fragments=[res_idxs]
            else:
                if top is None:
                    residues_as_fragments=[_np.arange(res_idxs[-1] + 1)]
                else:
                    residues_as_fragments=[_np.arange(top.n_residues)]
        else:
            residues_as_fragments=[sparse_residues]
    else:
        # Checks
        _no_intersect(fragments, word="fragments")
        if not set(res_idxs).issubset(_np.hstack(fragments)) and isinstance(sparse_residues, bool):
            raise ValueError("The input fragments do not contain all residues residx_array, " \
            "their set difference is %s" % sorted(set(res_idxs).difference(_np.hstack(fragments))))

        if isinstance(sparse_residues, bool):
            if sparse_residues:
                residues_as_fragments = [_np.intersect1d(ifrag,res_idxs) for ifrag in fragments]
            else:
                if sparse_fragments:
                    residues_as_fragments = [ifrag for ifrag in fragments if len(_np.intersect1d(ifrag,res_idxs))>0]
                else:
                    residues_as_fragments = fragments
        else:
            residues_as_fragments = [_np.intersect1d(ifrag, sparse_residues) for ifrag in fragments]

        residues_as_fragments = [ifrag for ifrag in residues_as_fragments if len(ifrag) > 0]

    re_indexed = []
    #TODO a lot of re-indexing or selections elswhere using some variation of "intersect_with"
    #try to refactor in one general method or many methods grouped together
    for am_frags in [anchor_fragments, mute_fragments]:
        if am_frags is not None:
            assert fragments is not None
            intersect_with = _np.hstack([fragments[ii] for ii in am_frags])
            am_frags = [ii for ii, ifrag in enumerate(residues_as_fragments) if len(_np.intersect1d(ifrag,intersect_with))>0]
            if len(am_frags)==0:
                am_frags = None
        else:
            pass
        re_indexed.append(am_frags)
    anchor_fragments, mute_fragments = re_indexed

    return residues_as_fragments, anchor_fragments, mute_fragments


def fontsize_get(iax):
    r"""
    Scan text elements and return a dictionary with fontsizes
    Parameters
    ----------
    iax : :obj:`~matplotlib.axes.Axes`

    Returns
    -------
    sizes : dict
        A dictionary with two lists,
        keyed with "n_polygons" and "other"
        * "n_polygons" contains a list in ascending
        order with the fontsizes of text elements
        of which there are as many as CirclePolygons
        This fact is a strong hint that those text
        elements correspond 1st to the residue-label texts
        and 2nd to the residue-SS-label texts
        * "other" contains all other fontsizes in
        :obj:`iax`, most likely belonging to the
        fragment-label texts
    """
    n_polygons = len([obj for obj in iax.findobj(_CP)])
    sizes = {"n_polygons": [],
             "other": []}

    c = _Counter([_np.round(txt.get_fontsize(), 2) for txt in iax.texts])

    for key, val in c.items():
        sizes[{True: "n_polygons",
               False: "other"}[val == n_polygons]].append(key)

    return {key: sorted(val) for key, val in sizes.items()}


def fontsize_apply(axA, axB):
    r"""
    Map the fontsizes of axA to the text-objects of axB

    Internally, :obj:`fontsize_get` is used
    Parameters
    ----------
    axA : :obj:`~matplotlib.axes.Axes`
    axB : :obj:`~matplotlib.axes.Axes`

    Returns
    -------

    """
    sizes1 = fontsize_get(axA)
    sizes2 = fontsize_get(axB)

    for key, val in sizes2.items():
        assert len(sizes1[key]) == len(val)

    fs2type = {}
    for key, val in sizes2.items():
        for ii, vv in enumerate(val):
            fs2type[vv] = sizes1[key][ii]

    for txt in axB.texts:
        fs = _np.round(txt.get_fontsize(), 2)
        txt.set_fontsize(fs2type[fs])

def add_aura(xy, aura, iax, r=1, fragment_lenghts=None, width=.10, subtract_baseline=True,
             colors=None,
             lines=False):
    r"""
    Create "auras", i.e. outer rings of the flareplot representing any scalar array

    Parameters
    ----------
    xy : np.ndarray of shape (N,2)
    aura : iterable of positive scalars
        The value to be plotted, e.g.
        conservation degree, RMSF,
        SASA, whatever.
        The indices are indices of
        :obj:`xy`, s.t. the have to
        be in the interval [0,len(xy)]
        If :obj:`xy` has been "sparsed"
        somehow, :obj:`aura` needs also
        to be sparsed
    r : float, default is 1
        Radius of the aura's baseline
    iax : :obj:`~matplotlib.axes.Axes`
        The axes containing the flareplot
    width : .10
        The width of the aura, in units of :obj:`r`
        The maximum aura value will always appear
        at r+r*width
    fragment_lenghts : iterable of ints, default is None
        How the :obj:`xy` is divided into fragments.
        if None, defaults to [len(xy)]
    subtract_baseline : bool, default is True
        Subtract the baseline of the :obj:`aura`.
        This highlights differences between
        high-values and low values,
        but makes the minimum values appear as zero.
        If False, then all values of the aura get compressed
        to the range [r,r+r*width], which can
        lead to poor resolution. For example, if
        you are plotting conservation degree in percent and
        all values are between 85 and 100%, not subtracting
        the baseline will make it hard to distinguish anything,
        all values will appear as "high".
    colors : anything
    lines : bool, default is False
        Use lines instead of polygons to represent the aura

    Returns
    -------
    artists : list
        Contains the  artist objects, either
        :obj:`~matplotlib.patches.Polygon` or
        :obj:`~matplotlib.collections.LineCollection`
    rmax : float
        The new maximum radius, i.e. (1+width)*r
    """

    if fragment_lenghts is None:
        fragment_lenghts = [len(xy)]
    assert len(aura) == len(xy) == sum(fragment_lenghts)
    aura = _np.array(aura,dtype=float)
    if any([a<0 for a in aura]):
        raise NotImplementedError("Negative aura-values not implemented yet")

    aura /= aura.max()
    if subtract_baseline:
        aura -= aura.min()
        aura /= aura.max()
    assert  aura.max()==1

    aura *= r * width

    col_list = col_list_from_input_and_fragments(colors, [_np.arange(l) for l in fragment_lenghts],alpha=.80)

    artists = []
    aura_iter = iter(aura)
    for jj, ixy in enumerate(_re_warp(xy, fragment_lenghts)):
        base, top, segments = [], [], []
        for ii, (x, y) in enumerate(ixy):
            theta = _np.arctan2(y, x)
            iaura = next(aura_iter)
            #theta_deg = theta / _np.pi * 180
            # print(theta)
            # plt.text(x,y,"%s %u"%(ii,theta_deg),fontsize=20)
            start = [_np.cos(theta) * r, _np.sin(theta) * r]
            end = [_np.cos(theta) * (r + iaura), _np.sin(theta) * (r + iaura)]
            base.append(start)
            top.append(end)
            segments.append([start, end])
        base = _np.array(base)
        top = _np.array(top)
        poly = _np.vstack((base, top[::-1]))
        # iax.scatter(base[:,0],base[:,1],color="r",zorder=100)
        # iax.scatter(top[:,0],top[:,1],color="g",zorder=100)
        if lines:
            artists.append(_LCol(segments, color=col_list[jj]))
        else:
            artists.append(_PG(poly, alpha=.80, color=col_list[jj],lw=0))

        iax.add_artist(artists[-1])
    return artists, r+aura.max()

def coarse_grain_freqs_by_frag(freqs, res_idxs_pairs, fragments,
                               check_if_subset=True):
    r"""
    Coarse-grain per-residue frequencies into a per-fragment contact-matrix

    The contact matrix gets symmetrized regardless of the input residue indices

    Parameters
    ----------
    freqs : iterable of floats
        The frequencies
    res_idxs_pairs : iterable of pairs
        The pairs
    fragments: iterable of iterables
        The fragments. Each individual
        residue of :obj:`res_index_pairs` can
        appear at most in one (and only one) fragment
        TODO this might not be true anymore,
        since self_interface can be True
    check_if_subset : bool, default is True
        Check whether all idxs in
        :obj:`res_idxs_pairs` belong
        to some fragment of :obj:`fragments`
        and raise an error if they dont.
        Bypassing this check  may lead
        to the residues missing from
        :obj:`fragments` not contributing
        to the freqs of the coarse-grained matrix

    Returns
    -------
    mat : np.ndarray
        Square, symmetric matrix of shape (len(fragments),len(fragments))
        TODO think about returning NaNs instead of zeros
        for the elements not present in :obj:`res_idxs_pairs`
    """

    parents, children = assign_fragments(_np.unique(res_idxs_pairs), fragments, raise_on_missing=check_if_subset)

    res_max = _np.max([_np.hstack(fragments).max(), _np.unique(res_idxs_pairs).max()])
    idx2frag = _np.zeros(res_max + 1)
    idx2frag[:] = _np.nan
    idx2frag[children] = parents
    frag_pairs = idx2frag[_np.array(res_idxs_pairs)].astype(int)
    mat_CG = _np.zeros((len(fragments), len(fragments)))
    for rp, fp, freq in zip(res_idxs_pairs, frag_pairs, freqs):
        if len(_np.intersect1d(rp, children)) == 2:
            mat_CG[fp[0], fp[1]] += freq
            mat_CG[fp[1], fp[0]] += freq

    mat_CG[_np.arange(len(fragments)), _np.arange(len(fragments))] /= 2 # the self-contacts need to be div 2

    return mat_CG

def sparsify_sym_matrix(mat, eps=1e-2):
    r"""
    Return a sparse matrix

    Checks of > eps are done element-wise,
    for entire row/column sum checks,
    see :obj: sparsify_sym_matrix_by_row_sum

    Parameters
    ----------
    mat : 2D np.ndarray of shape(N,N)
        The symmetric matrix
    eps : float, default is 1e-2
        Elements of :obj:`mat` are
        considered non-zero if > eps

    Returns
    -------
    mat, non_zeros
    """
    assert (mat==mat.T).all() #checks for squareness and symmetry
    non_zeros = _np.flatnonzero(_np.any(mat > eps, axis=1))
    mat = mat[non_zeros, :][:, non_zeros]
    return mat, non_zeros

def sparsify_sym_matrix_by_row_sum(mat, eps=1e-2):
    r"""
    Return a sparse matrix

    Parameters
    ----------
    mat : 2D np.ndarray of shape(N,N)
        The symmetric matrix
    eps : float, default is 1e-2
        Rows/Columns Elements of :obj:`mat` are
        considered non-zero row.sum() > eps

    Returns
    -------
    mat, non_zeros
    """
    assert (mat==mat.T).all() #checks for squareness and symmetry
    non_zeros = _np.flatnonzero(mat.sum(1)>eps)
    zeros = _np.flatnonzero(mat.sum(1)<=eps)
    discarded = mat.sum(1)[zeros].sum()
    mat = mat[non_zeros, :][:, non_zeros]
    return mat, non_zeros, discarded
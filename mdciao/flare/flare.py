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
from . import _utils as _futils
from mdciao.plots.plots import _points2dataunits
from mdciao.utils.str_and_dict import replace4latex
from mdciao.utils.residue_and_atom import get_SS as _get_SS

from matplotlib import pyplot as _plt
from matplotlib.patches import CirclePolygon as _CP
from inspect import signature as _signature

def freqs2flare(freqs, res_idxs_pairs,
                fragments=None,
                sparse_residues=False,
                sparse_fragments=False,
                exclude_neighbors=1,
                freq_cutoff=0,
                iax=None,
                fragment_names=None,
                center=_np.array([0,0]),
                r=1,
                mute_fragments=None,
                anchor_fragments=None,
                SS=None,
                panelsize=10,
                angle_offset=0,
                highlight_residxs=None,
                select_residxs=None,
                top=None,
                colors=True,
                fontsize=None,
                shortenAAs=True,
                aa_offset=0,
                markersize=None,
                bezier_linecolor='k',
                plot_curves_only=False,
                textlabels=True,
                padding=[1,1,1],
                lw=5,
                signed_colors=None,
                subplot=False,
                aura=None,
                ):
    r"""
    Plot contact frequencies as `flare plots`.

    The residues are plotted as dots lying on a circle of radius
    :obj:`r`, with Bezier curves of variable opacity connecting
    the dots. The curve opacity represents the contact frequency, :obj:`freq`,
    between two residues.

    For more info see the links here :obj:`mdciao.flare`.

    One can control separately what residues and what curves
    ultimately get shown, allowing for "contactless" residues
    to still appear as dots in the circle. This is very helpful
    to highlight the molecular topology. For example, it's useful to
    show an entire TM-helix even if only the residues in its middle
    have contacts. Furthermore, it allows for re-use of a "background"
    of residues on top of which different sets of curves (e.g. with
    different colors) can be plotted.

    * which/how residues get plotted:
        * :obj:`res_idxs_pairs` is the primary source of which
          residues will be plotted. All residues appearing
          in these pairs will always be plotted, no matter what.
        * :obj:`fragments` is used to
          a) expand the initial residue list and
          b) split the residues into fragments when placing the dots on the flareplot.
        * :obj:`sparse` is needed to accomplish b) without
          expanding the list
        * :obj:`highlight_residxs` show the labels of these residues
          in red

    * which/how residues get connected with bezier curves:
        * :obj:`exclude_neighbors` do not plot curves connecting neighbors
        * :obj:`freq_cutoff` do not plot curves with associated :obj:`freq`
          below this cutoff
        * :obj:`select_residxs` plot only the curves where these residues appear
        * :obj:`mute_fragments` do not plot curves from/to this fragment
        * :obj:`anchor_fragments` only plot curves from/to this fragment

    Parameters
    ----------
    freqs : numpy.ndarray
        The contact frequency to show in the flareplot. The
        opacity of the bezier curves connecting residue
        pairs will be proportional to this number. Can
        have different shapes:

        * (n)
          n is the number of residue pairs in :obj:`res_idxs_pairs`
        * (m,n)
          m is the number of frames, in this case,
          an average over m will be done automatically.

        If you already have an :obj:`mdciao.contacts.ContactGroup`-object,
        simply pass the result of calling
        :obj:`mdciao.contacts.ContactGroup.frequency_per_contact` (with
        a given cutoff)
    res_idxs_pairs : iterable of pairs
        pairs of residue indices for the above N contact frequencies.
        If you already have an :obj:`mdciao.contacts.ContactGroup`-object,
        simply pass the result of calling
        :obj:`mdciao.contacts.ContactGroup.res_idxs_pairs`
    fragments: list of lists of integers, default is None
        The residue indices to be drawn as a circle for the
        flareplot. These are the dots that will be plotted
        on that circle regardless of how many contacts they
        appear in. They can be any integers that could represent
        a residue. The only hard condition is that the set
        of np.unique(res_idxs_pairs) must be contained
        within :obj:`fragments`
    exclude_neighbors: int, default is 1
        Do not show contacts where the partners are separated by
        these many residues. If no :obj:`top` is passed, the
        neighborhood-condition is checked using residue
        serial-numbers, assuming the molecule only has one long peptidic-chain.
    freq_cutoff : float, default is 0
        Contact frequencies lower than this value will not be shown
    iax : :obj:`~matplotlib.axes.Axes`, default is None
        Parse an axis to draw on, otherwise one will be created
        using :obj:`panelsize`. In case you want to
        re-use the same cirlce of residues as a
        background to plot different sets
        of :obj:`freqs`, **YOU HAVE TO USE THE SAME**
        :obj:`fragments` and :obj:`sparse` values
         **on all calls**, else the
        bezier lines will be placed erroneously.
    fragment_names: iterable of strings, default is None
        The names of the fragments used in :obj:`fragments`
    panelsize: float, default is 10
        Size in inches of the panel (=figsize in matplotlib).
        Will be ignored if a pre-existing axis object is parsed
    center: np.ndarray, default is [0,0]
        In axis units, where the flareplot will be centered around
    r: float, default is 1
        In axis units, the radius of the flareplot
    textlabels : bool or array_like, default is True
        How to label the residue dots. Gets passed directly
        to :obj:`mdciao.flare.circle_plot_residues`.
        Options are:
         * True: the dots representing the residues
           will get a label automatically, either their
           serial index or the residue name, e.g. GLU30, if
           a :obj:`top` was passed.
         * False: no labeling
         * array_like : will be passed as :obj:`replacement_labels`
           to :obj:`mdciao.flare.add_fragmented_residue_labels`
    mute_fragments: iterable of integers, default is None
        Curves involving these fragments will be hidden. Fragments
        are expressed as indices of :obj:`fragments`
    anchor_fragments: iterable of integers, default is None
        Curves **not** involving these fragments will be
        **not** be shown, i.e. it is the complementary
         of :obj:`mute_fragments`. Both cannot be passed
         simulataneously.
    top: :obj:`~mdtraj.Topology` object, default is None
        If provided a top, residue names (e.g. GLU30) will be used
        instead of residue indices. Will fail if the residue indices
        in :obj:`res_idxs_pairs` can not be used to call :obj:`top.residue(ii)`
    SS : any, default is None
       Can be several things:
       * Array containing secondary structure (ss) information to
         be included in the flareplot. Indexed by residue index,
         i.e. it can also be a dictionary as long as
         SS[idx] returns the SS for residue with that residue idx
       * Path to filename, will be passed to
       :obj:`mdciao.utils.residue_and_atom.get_SS`, check
       the docs there
    angle_offset : float, default is 0
        In degrees, where the flareplot "begins". Zero is xy = [1,0]
    highlight_residxs : iterable of ints, default is None
        Show the labels for these residues in red
    select_residxs : iterable of ints, default is None
        Only the residues here can be connected with a Bezier curve
    colors: boolean, default is True
        Color control. Can take different inputs
         * True: use one different color per segment
         * False: defaults to gray.
         * str or char: use that color for all residues (e.g. "r" or "red")
         * A list of strings of len = number of drawn residues, which is
           equal to len(np.hstack(fragments)). Any other length will produce an error
    fontsize: int, default is None
    lw: float, default is None
        Line width of the contact lines
    shortenAAs: boolean, default is True
        Use short AA-codes, e.g. E30 for GLU30. Only has effect if a topology
        is parsed
    sparse_residues : boolean, default is False
        Show only those residues that appear in the initial :obj:`res_idxs_pairs`

        Note
        ----
        There is a development option for this argument where a residue
        list is passed, meaning, show these residues regardless of any other
        option that has been passed. Perhaps sparse changes in the future.
    sparse_fragments : boolean, default is False
        Same as :obj:`sparse_residues`, but with fragments. When
        :obj:`sparse_residues` isn't False, this option
        has no effect.
    bezier_linecolor : color-like, default is 'k'
        The color of the bezier curves connecting the residues.
        Can be a character, string or RGB value (not RGBA)
    padding : iterable of len 3, default is [1,1,1]
        The padding, expressed as empty dot positions. Each number is
        used for:
         * the beginning of the flareplot, before the first residue
         * between fragments
         * at the end of the plot, after the last residue
    signed_colors : dict, default is None
        Provide a color dictionary, e.g. {-1:"b", +1:"r"}
        to give different colors to positive and negative
        alpha values. If None, defaults to :obj:`bezier_linecolor`
    aa_offset : int, default is 0
        Add this number to the resSeq value
    plot_curves_only : bool, default is False
        Only plot the curves connecting the dots, but
        not the dots themselves or any other annotation.
        (labels, fragment names or SS information).
        The same caution as :obj:`iax` applies.
    subplot : bool, default is False
        If True, the method checks if
        :obj:`iax` is the last axis in a
        figure (=all other panels have
        been already drawn) and then
        transfers the last plot's
        fontsizes and linewidths
        to panels (if possible).
        It will help produce more homogeneous
        plots when heuristics about font-sizing
        fail
    aura : iterable, default is None
        Scalar array, indexed with residue indices,
        e.g. RMSF, SASA, conv. degree...
        It will be drawn as an *aura* around the
        flareplot.

    Returns
    -------
    iax : :obj:`~matplotlib.axes.Axes`
        You can do iax.figure.savefig("figure.png") to
        save the figure. Checkout
        :obj:`~matplotlib.figure.Figure.savefig` for more options

    plotted_pairs : 2D np.ndarray

    plot_attribs : dict
        Objects of the plot if the user wants
        to manipulate them further or re-use
        some attributes:
        "bezier_lw", "bezier_curves",
        "fragment_labels", "dot_labels",
        "dots", "SS_labels"
    """

    if _np.ndim(freqs)==1:
        freqs = _np.reshape(freqs,(1, -1))
    elif _np.ndim(freqs)==2:
        pass
    else:
        raise ValueError("Input array has to of shape either "
                         "(m) or (n, m) where n : n frames, and m: n_contacts")

    assert freqs.shape[1] == len(res_idxs_pairs), \
        "The size of the contact array and the " \
        "res_idxs_pairs array do not match %u vs %u"%(freqs.shape[1], len(res_idxs_pairs))

    # Figure out the combination of sparse_residues/fragments options
    residues_as_fragments = _futils._parse_residue_and_fragments(res_idxs_pairs,
                                                                 sparse_residues=sparse_residues,
                                                                 sparse_fragments=sparse_fragments,
                                                                 fragments=fragments,
                                                                 top=top)
    # Delete fragment names that won't be used
    if fragment_names is not None and fragments is not None:
        fragment_names = [fn for fr, fn in zip(fragments,fragment_names) if len(_np.intersect1d(fr,_np.hstack(residues_as_fragments)))>0]

    # Create a map
    residx2markeridx = _futils.value2position_map(_np.hstack(residues_as_fragments))

    if plot_curves_only:
        plot_attribs = {}
        assert iax is not None, ("You cannot use "
                                 "plot_curves_only=True and iax=None. Makes no sense")
        xy = _futils.cartify_fragments(residues_as_fragments,
                                       r=r,
                                       angle_offset=angle_offset,
                                       padding=padding,
                                       )
        xy += center
    else:
        iax, xy, plot_attribs = circle_plot_residues(residues_as_fragments,
                                                     fontsize=fontsize,
                                                     colors=colors,
                                                     panelsize=panelsize,
                                                     padding=padding,
                                                     center=center,
                                                     ss_array=_get_SS(SS, top=top)[1],
                                                     fragment_names=fragment_names,
                                                     iax=iax,
                                                     markersize=markersize,
                                                     textlabels=textlabels,
                                                     shortenAAs=shortenAAs,
                                                     highlight_residxs=highlight_residxs,
                                                     aa_offset=aa_offset,
                                                     top=top,
                                                     r=r,
                                                     angle_offset=angle_offset,
                                                     aura=aura)

        circle_radius_in_pts = iax.artists[0].radius * _points2dataunits(iax).mean()
        lw = circle_radius_in_pts # ??

    # All formed contacts
    ctcs_averaged = _np.average(freqs, axis=0)
    idxs_of_formed_contacts = _np.argwhere(_np.abs(ctcs_averaged)>freq_cutoff).squeeze()
    plot_this_pair_lambda = _futils.should_this_residue_pair_get_a_curve(residues_as_fragments,
                                                                         select_residxs=select_residxs,
                                                                         mute_fragments=mute_fragments,
                                                                         anchor_fragments=anchor_fragments,
                                                                         top=top, exclude_neighbors=exclude_neighbors)

    idxs_of_pairs2plot = _np.intersect1d(idxs_of_formed_contacts,
                                         [ii for ii, pair in enumerate(res_idxs_pairs) if plot_this_pair_lambda(pair)])

    if len(idxs_of_pairs2plot) > 0:
        pairs_of_nodes = [(xy[residx2markeridx[ii]],
                           xy[residx2markeridx[jj]]) for (ii, jj) in _np.array(res_idxs_pairs)[idxs_of_pairs2plot]]
        alphas = ctcs_averaged[idxs_of_pairs2plot]
        bezier_curves = add_bezier_curves(iax,
                                          pairs_of_nodes,
                                          alphas=alphas,
                                          center=center,
                                          lw=lw,
                                          bezier_linecolor=bezier_linecolor,
                                          signed_alphas=signed_colors
                                          )
        plot_attribs["bezier_lw"] = lw
        plot_attribs["bezier_curves"] = bezier_curves

    if subplot:
        if iax is iax.figure.axes[-1]:
            [_futils.fontsize_apply(iax, jax) for jax in iax.figure.axes]
            minwidth = min([_np.unique([line.get_linewidth() for line in jax.lines]) for jax in iax.figure.axes])
            [[line.set_linewidth(minwidth) for line in jax.lines] for jax in iax.figure.axes]
    return iax, idxs_of_pairs2plot, plot_attribs

def circle_plot_residues(fragments,
                         fontsize=None,
                         colors=True,
                         markersize=None,
                         r=1,
                         panelsize=4,
                         angle_offset=0,
                         padding=[1,1,1],
                         center=[0,0],
                         ss_array=None,
                         fragment_names=None,
                         iax=None,
                         textlabels=True,
                         shortenAAs=True,
                         highlight_residxs=None,
                         aa_offset=0,
                         top=None,
                         arc=False,
                         aura=None):
    r"""
    Circular background that serves as background for flare-plots. Is independent of
    the curves that will later be plotted onto it.

    Parameters
    ----------
    fragments : list
        List of iterables of residue idxs defining how the
        residues are split into fragments. If no
        :obj:`textlabels` are provided, the idxs
        themselves become the labels
    r : scalar
        The radius of the circle, in axis inuts
    angle_offset : scalar
        Where the circle starts, in degrees. 0 means 3 o'clock,
        90 12 o'clock etc. It's the phi of polar coordinates
    padding : list, default is [1,1,1]
        * first integer : Put this many empty positions before the first dot
        * second integer: Put this many empty positions between fragments
        * third integer : Put this many empty positions after the last dot
    center : pair of floats
        where the circle is centered, in axis units
    ss_array : dict, list or array
        One-letter codes (H,B,E,C) denoting secondary
        structure. Has to be indexable by whatever
        indices are on :obj:`fragments`
    fragment_names : list
        The names of the fragments
    iax : :obj:`~matplotlib.axes.Axes`, default is None
        An axis to draw the dots on. It's parent
        figure has to have a tight_layout=True
        attribute. If no axis is passed,
        one will be created.
    textlabels : bool or array_like, default is True
        How to label the residue dots
         * True: the dots representing the residues
           will get a label automatically, either their
           serial index or the residue name, e.g. GLU30, if
           a :obj:`top` was passed.
         * False: no labeling
         * array_like : will be passed as :obj:`replacement_labels`
           to :obj:`mdciao.flare.add_fragmented_residue_labels`.
           These labels act as replacement and can cover all or just
           some residue, e.g. a mutated residue that you want
           to show as R38A instead of just A38, or use
           e.g. GPCR or CGN consensus labels.
    fontsize
    colors
    markersize
    top : :obj:`mdtraj.Topology`, default is None
        If provided, residue labels wil be auto-generated from
        here
    shortenAAs : boolean, default is True
        If :obj:`top` is not None, use "E50" rather than "GLU50"
    aa_offset : int, default is 0
        Add this number to the resSeq value
    highlight_residxs : iterable of ints, default is None
        In case you don't want to construct a whole
        color list for :obj:`colors`, you can simply
        input a subset of :obj:`res_idxs` here and
        they will be shown in red.
    aura : iterable, default is None
        Scalar array, indexed with residue indices,
        e.g. RMSF, SASA, conv. degree...
        It will be drawn as an *aura* around the
        flareplot.

    Returns
    -------
    iax, xy, outdict

    outdict: dict
         Contains :obj:`matplotlib` objects
         like the dots and their labels:
         "fragment_labels", "dot_labels",
         "dots", "SS_labels", "r",
    """
    debug = False

    xy = _futils.cartify_fragments(fragments,
                                   r=r,
                                   angle_offset=angle_offset,
                                   padding=padding)
    xy += center

    n_positions = len(xy) + padding[0] + len(fragments)*padding[1] + padding[2]

    residues_to_plot_as_dots = _np.hstack(fragments)
    col_list = _futils.col_list_from_input_and_fragments(colors, fragments, alpha=.80)
    if iax is None:
        _plt.figure(figsize=(panelsize, panelsize), tight_layout=True)
        iax = _plt.gca()
    else:
        if not iax.figure.get_tight_layout():
            print("The passed figure was not instantiated with tight_layout=True\n"
                  "This may lead to some errors in the flareplot fontsizes.")
    # Do this first to have an idea of the points per axis unit necessary for the plot
    iax.set_xlim([center[0] - 3*r, center[0] + 3*r])
    iax.set_ylim([center[1] - 3*r, center[1] + 3*r])
    iax.set_aspect('equal')

    # Plot!
    if markersize is None:
        dot_radius = r * _np.sin(_np.pi/n_positions)
        dot_radius_in_pts = dot_radius*_points2dataunits(iax).mean()
        if dot_radius_in_pts < 1.5:
            print(ValueError("Drawing this many of residues (%u) in "
                             "a panel %3.1f inches wide/high "
                             "forces too small dotsizes and fontsizes.\n"
                             "If crowding effects "
                             "occur, either reduce the number of residues or increase "
                             "the panel size"%(n_positions, panelsize)))
        #TODO replace this with a call to RegularPolyCollection
        CPs = [_CP(ixy,
                   radius=dot_radius,
                   facecolor=col_list[ii],
                   edgecolor=None,
                   zorder=10) for ii, ixy in enumerate(xy)]
        [iax.add_artist(iCP) for iCP in CPs]
        outer_r_in_data_units = r + dot_radius
    else:
        raise NotImplementedError

    if debug:
        iax.add_artist(_plt.Circle(center,
                                   radius=outer_r_in_data_units,
                                   ec='r',
                                   fc=None,
                                   fill=False,
                                   zorder=10,
                                   lw=dot_radius_in_pts/10))
    # After the dots have been plotted,
    # we use their radius in points
    # as approximate fontsize
    if fontsize is None:
        fontsize = dot_radius_in_pts *1.75 # slightly smaller than the diameter
    else:
        raise NotImplementedError

    labels = []
    if textlabels:
        if isinstance(textlabels,bool):
            replacement_labels = None
        else:
            # Interpret the textlabels as replacements
            replacement_labels = textlabels
        outer_r_in_data_units += dot_radius
        labels = add_fragmented_residue_labels(fragments,
                                               iax,
                                               fontsize,
                                               center=center,
                                               r=outer_r_in_data_units,
                                               angle_offset=angle_offset,
                                               padding=padding,
                                               highlight_residxs=highlight_residxs,
                                               top=top,
                                               aa_offset=aa_offset,
                                               replacement_labels=replacement_labels)
        if debug:
            _futils._plot_fancypatches(labels, lw=dot_radius_in_pts / 10)
            iax.add_artist(_plt.Circle(center,
                                       radius=outer_r_in_data_units,
                                       ec='green',
                                       fc=None,
                                       fill=False,
                                       zorder=10,
                                       lw=dot_radius_in_pts / 10))

        delta_r = _futils._outermost_corner_of_fancypatches(labels)-outer_r_in_data_units
        outer_r_in_data_units += delta_r*.90 # fudge
        if debug:
            iax.add_artist(_plt.Circle(center,
                                       radius=outer_r_in_data_units,
                                       ec='b',
                                       fc=None,
                                       fill=False,
                                       zorder=10,
                                       lw=dot_radius_in_pts / 10))
    # Do we have SS dictionaries
    ss_labels = []
    if ss_array is not None:
        ss_colors = [_futils._SS2vmdcol[ss_array[res_idx]] for res_idx in residues_to_plot_as_dots]
        replacement_labels = {ii: ss_array[ii].replace("NA", " ").replace("E", "B") for ii in
                                 residues_to_plot_as_dots}
        ss_labels = add_fragmented_residue_labels(fragments,
                                                  iax,
                                                  fontsize * 2,
                                                  center=center,
                                                  r=outer_r_in_data_units,
                                                  angle_offset=angle_offset,
                                                  padding=padding,
                                                  shortenAAs=shortenAAs,
                                                  highlight_residxs=highlight_residxs, top=top,
                                                  aa_offset=aa_offset,
                                                  replacement_labels=replacement_labels,
                                                  colors=[_futils._make_color_transparent(col, .8, "w") for col in
                                                          ss_colors],
                                                  weight="bold")

        outer_r_in_data_units = _futils._outermost_corner_of_fancypatches(ss_labels)
        if debug:
            _futils._plot_fancypatches(ss_labels, lw=dot_radius_in_pts / 10)
            iax.add_artist(_plt.Circle(center,
                                       radius=outer_r_in_data_units,
                                       ec='purple',
                                       fc=None,
                                       fill=False,
                                       zorder=10,
                                       lw=dot_radius_in_pts / 10))

    if aura is not None:
        if debug:
            iax.add_artist(_plt.Circle([0, 0], outer_r_in_data_units,
                                       color=None,
                                       fill=False,
                                       ec="cyan",
                                       zorder=10,
                                       lw=dot_radius_in_pts / 10))
            iax.add_artist(_plt.Circle([0,0],outer_r_in_data_units, ec="k", alpha=.25, zorder=-100))

        auras, outer_r_in_data_units = _futils.add_aura(xy, aura[_np.hstack(fragments)], iax, outer_r_in_data_units+dot_radius,
                         [len(fr) for fr in fragments],
                         subtract_baseline=False
                         )

        if aura is not None:
            if debug:
                iax.add_artist(_plt.Circle([0, 0], outer_r_in_data_units,
                                           ec="g",
                                           facecolor=None,
                                           fill=False,
                                           zorder=10,
                                           lw=dot_radius_in_pts / 10))

    # Do we have names?
    frag_labels = []
    if fragment_names is not None:
        span = 2 * outer_r_in_data_units
        frag_fontsize_in_aus =  span/6 * 1/5 # (average_word_length, fraction of panel space)
        frag_fontsize_in_pts = _np.max((3*fontsize, frag_fontsize_in_aus * _points2dataunits(iax).mean()))
        outer_r_in_data_units += frag_fontsize_in_aus
        if debug:
            iax.add_artist(_plt.Circle(center,
                                       radius=outer_r_in_data_units,
                                       ec='pink',
                                       fc=None,
                                       fill=False,
                                       zorder=10,
                                       lw=dot_radius_in_pts / 10))
        frag_labels = _futils.add_fragment_labels(fragments,
                                                  [replace4latex(str(ifrag)) for ifrag in fragment_names],
                                                  iax,
                                                  angle_offset=angle_offset,
                                                  padding=padding,
                                                  fontsize=frag_fontsize_in_pts,
                                                  center=center,
                                                  r=outer_r_in_data_units
                                                  )

        outer_r_in_data_units = _futils._outermost_corner_of_fancypatches(frag_labels)

        if debug:
            iax.add_artist(_plt.Circle(center,
                                       radius=outer_r_in_data_units,
                                       ec='orange',
                                       fc=None,
                                       fill=False,
                                       zorder=10,
                                       lw=dot_radius_in_pts / 10))

    iax.set_yticks([])
    iax.set_xticks([])
    old_d = _np.abs(_np.diff(iax.get_xlim()))
    iax.set_xlim([center[0] - outer_r_in_data_units, center[0] + outer_r_in_data_units])
    iax.set_ylim([center[1] - outer_r_in_data_units, center[1] + outer_r_in_data_units])
    new_d = _np.abs(_np.diff(iax.get_xlim()))
    [lab.set_fontsize(lab.get_fontsize()*(old_d/new_d)) for lab in labels+ss_labels+frag_labels]
    _futils.un_overlap_via_fontsize(frag_labels, fac=.90)

    return iax, xy, {"fragment_labels":frag_labels,
                     "dot_labels":labels,
                     "dots":CPs,
                     "SS_labels":ss_labels,
                     "r": outer_r_in_data_units
                     }


def add_bezier_curves(iax,
                      nodepairs_xy,
                      alphas=None,
                      center=[0,0],
                      lw=1,
                      bezier_linecolor='k',
                      signed_alphas=None,
                      correct_adjacent_nodes=True
                      ):
    r"""
    Generate and plot bezier curves using :obj:`bezier.Curves` as a base class

    Parameters
    ----------
    iax : :obj:`~matplotlib.axes.Axes`
    nodepairs_xy : iterable of pairs of pairs of floats
        Each item is a pair of pairs [(x1,y1),(x2,y2)]
        to be connected with bezier curves 1<--->2
    alphas : iterable of floats, default is None
        The opacity of the curves connecting
        the pairs in :obj:`nodepairs_xy`.
        If None is provided, it will be set to 1
        for all curves. If provided,
        must be of len(nodepairs_xy).
    center : array-like with two floats, default is [0,0]
    lw : int, default is 1
    bezier_linecolor: :obj:`matplotlib` color, default is "k"
    signed_alphas : dict, default is None
        Provide a color dictionary, e.g. {-1:"b", +1:"r}
        to give different colors to positive and negative
        alpha values. Overwrites whatever is in :obj:`bezier_linecolor`.
        If None, defaults to :obj:`bezier_linecolor`
    correct_adjacent_nodes : boolean, default is True
        If two nodes are too close to each other,
        use a shifted center for the Bezier curve so
        that it's visually easier to find. Currently,
        adjacent is hard-coded to mean "nodes are ten
        times closer to each other than to the center"

    Returns
    -------
    bz_curves : list
        The :obj:`bezier.Curves` plotted

    """
    if alphas is None:
        alphas = _np.ones(len(nodepairs_xy)) * 1.0

    assert len(alphas)==len(nodepairs_xy)

    if signed_alphas in [None,{}]:
        signed_alphas={-1:bezier_linecolor,
                       +1:bezier_linecolor}

    bz_curves=[]
    for nodes, ialpha in zip(nodepairs_xy, alphas):

        d_nodes = _np.sqrt(((nodes[0] - nodes[1]) ** 2).sum())
        r = _np.mean([_np.linalg.norm(nodes[0] - center), _np.linalg.norm(nodes[1] - center)])
        pair_center = (nodes[0] + nodes[1]) / 2
        pc2c = center - pair_center
        if correct_adjacent_nodes and r / 10 > d_nodes:
            _center = pair_center + pc2c / 10
        else:
            _center = center
        bz_curves.append(_futils.create_flare_bezier_2(nodes, center=_center))
        bz_curves[-1].plot(50,
                           ax=iax,
                           alpha=_np.abs(ialpha),
                           color=signed_alphas[_np.sign(ialpha)],
                           lw=lw,  # _np.sqrt(markersize),
                           zorder=-1
                           )

    return bz_curves

def add_fragmented_residue_labels(fragments,
                                  iax,
                                  fontsize,
                                  center=[0,0],
                                  r=1,
                                  angle_offset=0,
                                  padding=[0,0,0],
                                  **add_res_labs_kwargs,
                                  ):
    r"""
    Like add_labels but starting from the fragments

    Parameters
    ----------
    fragments : iterable of iterables
        List of iterables of residue idxs defining how the
        residues are split into fragments. If no
        :obj:`textlabels` are provided, the idxs
        themselves become the labels
    iax : :obj:`~matplotlib.axes.Axes`
    fontsize : int
    r : scalar
        The radius of the circle, in axis inuts
    angle_offset : scalar
        Where the circle starts, in degrees. 0 means 3 o'clock,
        90 12 o'clock etc. It's the phi of polar coordinates
    padding : list, default is [1,1,1]
        * first integer : Put this many empty positions before the first dot
        * second integer: Put this many empty positions between fragments
        * third integer : Put this many empty positions after the last dot
    center : pair of floats
        where the circle is centered, in axis units
    add_res_labs_kwargs

    Returns
    -------
    labels : list of text objects

    """
    xy_labels, xy_angles = _futils.cartify_fragments(fragments,
                                                     r=r,
                                                     return_angles=True,
                                                     angle_offset=angle_offset,
                                                     padding=padding)
    xy_labels += center

    residues_to_plot_as_dots = _np.hstack(fragments)
    labels = _futils.add_residue_labels(iax,
                                        xy_labels,
                                        residues_to_plot_as_dots,
                                        fontsize,
                                        center=center,
                                        **add_res_labs_kwargs,
                                        )

    return labels

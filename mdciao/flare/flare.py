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

from matplotlib import pyplot as _plt
from matplotlib.patches import CirclePolygon as _CP

def freqs2flare(freqs, res_idxs_pairs,
                fragments=None,
                sparse=False,
                exclude_neighbors=1,
                freq_cutoff=0,
                iax=None,
                fragment_names=None,
                center=_np.array([0,0]),
                r=1,
                mute_fragments=None,
                anchor_fragments=None,
                ss_array=None,
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
        :obj:`fragments` **on all calls**, else the
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
    ss_array : 1D np.ndarray, default is None
        Array containing secondary structure (ss) information to
        be included in the flareplot. Indexed by residue index,
        i.e. it can also be a dictionary as long as
        ss_array[idx] returns the SS for residue with that residue idx
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
    sparse : boolean, default is False
        Show only those residues that appear in the initial :obj:`res_idxs_pairs`

        Note
        ----
        There is a hidden development option for this argument where a residue
        list is passed, meaning, show these residues regardless of any other
        option that has been passed. Perhaps sparse changes in the future.
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

    # Figure out the combination of sparse/fragments options
    residues_as_fragments = _futils._parse_residue_and_fragments(res_idxs_pairs,
                                                                 sparse=sparse,
                                                                 fragments=fragments)

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
                                                     ss_array=ss_array,
                                                     fragment_names=fragment_names,
                                                     iax=iax,
                                                     markersize=markersize,
                                                     textlabels=textlabels,
                                                     shortenAAs=shortenAAs,
                                                     highlight_residxs=highlight_residxs,
                                                     aa_offset=aa_offset,
                                                     top=top,
                                                     r=r,
                                                     angle_offset=angle_offset)

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
                         arc=False):
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
           e.g. BW or CGN consensus labels.
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

    # TODO review variable names
    # TODO this color mess needs to be cleaned up
    residues_to_plot_as_dots = _np.hstack(fragments)
    col_list = _futils.col_list_from_input_and_fragments(colors, fragments)
    if iax is None:
        _plt.figure(figsize=(panelsize, panelsize), tight_layout=True)
        iax = _plt.gca()
        # Do this first to have an idea of the points per axis unit necessary for the plot
        iax.set_xlim([center[0] - r, center[0] +  r])
        iax.set_ylim([center[1] - r, center[1] +  r])
    else:
        if not iax.figure.get_tight_layout():
            print("The passed figure was not instantiated with tight_layout=True\n"
                  "This may lead to some errors in the flareplot fontsizes.")

    iax.set_aspect('equal')

    # Create a map
    residx2markeridx = _futils.value2position_map(residues_to_plot_as_dots)

    # Plot!
    running_r_pad = 0
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
        if not arc:
            CPs = [_CP(ixy,
                       radius=dot_radius,
                       facecolor=col_list[ii],
                       edgecolor=None,
                       zorder=10) for ii, ixy in enumerate(xy)]
            [iax.add_artist(iCP) for iCP in CPs]
            running_r_pad += dot_radius
        else:
            lw = r*.05*_points2dataunits(iax).mean() #5% of the arc linewidth)
            _futils.draw_arcs(fragments, iax,
                              colors=[col_list[ifrag[0]] for ifrag in fragments],
                              center=center,
                              r=2*r,  # don't understand this 2 here
                              angle_offset=angle_offset,
                              padding=padding,
                              lw=2 * lw
                              )
            running_r_pad += 4 * lw
    else:
        raise NotImplementedError
        #iax.scatter(xy[:, 0], xy[:, 1], c=col_list, s=10, zorder=10)

    # After the dots have been plotted,
    # we use their radius as in points
    # as approximate fontsize
    if fontsize is None:
        opt_h_in_pts = dot_radius_in_pts
        fontsize = opt_h_in_pts
    else:
        raise NotImplementedError

    n_max = 100
    maxlen = 1
    labels = []
    if textlabels:
        if isinstance(textlabels,bool):
            replacement_labels = None
        else:
            # Interpret the textlabels as replacements
            replacement_labels = textlabels
        overlap = True
        counter = 0
        while overlap and counter < n_max:
            running_r_pad += dot_radius * maxlen  / 2 # This is a fudge.
            # The problem is that re-scaling at a later point might induce overlap
            # in this labels again, so I am anticipating that by adding some extra padding here
            [ilab.remove() for ilab in labels]
            labels = add_fragmented_residue_labels(fragments,
                                                   iax,
                                                   fontsize,
                                                   center=center,
                                                   r=r + running_r_pad,
                                                   angle_offset=angle_offset,
                                                   padding=padding,
                                                   highlight_residxs=highlight_residxs,
                                                   top=top,
                                                   aa_offset=aa_offset,
                                                   replacement_labels=replacement_labels)
            lab_lenths = [len(itext.get_text()) for itext in labels]
            idx_longest_label = _np.argmax(lab_lenths)
            maxlen=_np.max(lab_lenths)
            bad_txt_bb = labels[idx_longest_label].get_window_extent(renderer=iax.figure.canvas.get_renderer())
            bad_dot_bb =    CPs[idx_longest_label].get_window_extent(renderer=iax.figure.canvas.get_renderer())
            overlap = bad_txt_bb.overlaps(bad_dot_bb)
            counter+=1

        assert not overlap, ValueError("Tried to 'un'-overlap textlabels and residue markers %u times without success"%n_max)

        running_r_pad += dot_radius*maxlen/2
    if debug:
        iax.add_artist(_plt.Circle(center,
                                   radius=r + running_r_pad,
                                   ec='r',
                                   fc=None,
                                   fill=False,
                                   zorder=10))


    # Do we have SS dictionaries
    ss_labels = []
    if ss_array is not None:
        overlap = True
        counter = 0
        ss_colors = [_futils._SS2vmdcol[ss_array[res_idx]] for res_idx in residues_to_plot_as_dots]
        while overlap and counter < n_max:
            running_r_pad += dot_radius * 2
            [ilab.remove() for ilab in ss_labels]
            ss_labels = add_fragmented_residue_labels(fragments,
                                                      iax,
                                                      fontsize * 2,
                                                      center=center,
                                                      r=r + running_r_pad,
                                                      angle_offset=angle_offset,
                                                      padding=padding,
                                                      shortenAAs=shortenAAs,
                                                      highlight_residxs=highlight_residxs, top=top,
                                                      aa_offset=aa_offset,
                                                      replacement_labels={ii: ss_array[ii] for ii in
                                                                          residues_to_plot_as_dots},
                                                      colors=ss_colors,
                                                      weight="bold")
            idx_longest_label = _np.argmax([len(itext.get_text()) for itext in ss_labels])
            # We're doing this "by hand" here because it's just two or at most 3 offenders
            bad_ss_bb = ss_labels[idx_longest_label].get_window_extent(renderer=iax.figure.canvas.get_renderer())
            bad_dot_bb = CPs[idx_longest_label].get_window_extent(renderer=iax.figure.canvas.get_renderer())
            overlap = bad_ss_bb.overlaps(bad_dot_bb)
            if len(labels) >0 :
                overlap_w_text = bad_ss_bb.overlaps(labels[idx_longest_label].get_window_extent(renderer=iax.figure.canvas.get_renderer()))
                overlap = _np.any([overlap, overlap_w_text])
            counter += 1

        running_r_pad += dot_radius
    if debug:
        iax.add_artist(_plt.Circle(center,
                                   radius=r + running_r_pad,
                                   ec='b',
                                   fc=None,
                                   fill=False,
                                   zorder=10))

    # Do we have names?
    frag_labels = None
    if fragment_names is not None:
        span = (2*(r + running_r_pad))
        frag_fontsize_in_aus =  span/6 * 1/5 # (average_word_length, fraction of panel space)
        frag_fontsize_in_pts = frag_fontsize_in_aus * _points2dataunits(iax).mean()
        frag_labels = _futils.add_fragment_labels(fragments,
                                                  [replace4latex(ifrag) for ifrag in fragment_names],
                                                  iax,
                                                  angle_offset=angle_offset,
                                                  padding=padding,
                                                  #xy,
                                                  #residx2markeridx,
                                                  fontsize=frag_fontsize_in_pts,
                                                  center=center,
                                                  r=r + running_r_pad
                                                  )
        # First un-overlapp the labels themselves
        _futils.un_overlap_via_fontsize(frag_labels)
        frag_fontsize_in_pts = frag_labels[0].get_size()

        # Then find the overlappers among existing labels (to avoid using all labels unnecessarily)
        foverlappers = _futils.overlappers(frag_labels, CPs + labels + ss_labels)
        counter = 0
        while any(_futils.overlappers(frag_labels, foverlappers)) and counter < n_max:
            [fl.remove() for fl in frag_labels]
            running_r_pad += frag_fontsize_in_pts / _points2dataunits(iax).mean()
            frag_labels = _futils.add_fragment_labels(fragments,
                                                      [replace4latex(ifrag) for ifrag in fragment_names],
                                                      iax, #xy,
                                                      #residx2markeridx,
                                                      angle_offset=angle_offset,
                                                      padding=padding,
                                                      fontsize=frag_fontsize_in_pts,
                                                      center=center,
                                                      r=r + running_r_pad
                                                      )
            # print(counter, overlappers(frag_labels, foverlappers))
            counter += 1
            frag_fontsize_in_pts = frag_labels[0].get_size()

        running_r_pad += frag_fontsize_in_pts / _points2dataunits(iax).mean()
    iax.set_yticks([])
    iax.set_xticks([])
    iax.set_xlim([center[0] - r - running_r_pad, center[0] + r + running_r_pad])
    iax.set_ylim([center[1] - r - running_r_pad, center[1] + r + running_r_pad])
    return iax, xy, {"fragment_labels":frag_labels,
                     "dot_labels":labels,
                     "dots":CPs,
                     "SS_labels":ss_labels,
                     "r": r+running_r_pad}


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

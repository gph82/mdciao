import numpy as _np
from ._utils import \
    cartify_fragments,  \
    add_fragment_names, col_list_from_input_and_fragments, should_this_residue_pair_get_a_curve, add_residue_labels, create_flare_bezier, residx2dotidx, draw_arcs, \
    _SS2vmdcol, pts_per_axis_unit

from matplotlib import pyplot as _plt
from matplotlib.patches import CirclePolygon as _CP
def sparse_freqs2flare(ictcs, res_idxs_pairs,
                       fragments=None,
                       sparse=False,
                       exclude_neighbors=1,
                       freq_cutoff=0,
                       iax=None,
                       fragment_names=None,
                       center=_np.array([0,0]),
                       r=1,
                       mute_fragments=None,
                       anchor_segments=None,
                       ss_array=None,
                       panelsize=10,
                       angle_offset=0,
                       highlight_residxs=None,
                       pair_selection=None,
                       top=None,
                       colors=True,
                       fontsize=6,
                       shortenAAs=False,
                       aa_offset=0,
                       markersize=5,
                       bezier_linecolor='k',
                       plot_curves_only=False,
                       textlabels=True,
                       padding_beginning=0,
                       padding_end=0,
                       lw=5,
                       ):
    r"""

    The residues are plotted as dots lying on a circle of radius
    :obj:`r`, with bezier curves of variable :obj:`alpha` connecting
    the dots. The curve opacity represents the contact frequency
    between two residues.

    The control over what residues and what curves get ultimately shown
    is done separately(-ish) for the residues and the curves.

    The reason behind this is that in some/many cases, letting
    "contactless" residues appear in plot helps in showing
    the underlying molecular topology. For example, it's useful to
    show an entire TM-helix even if only the residues in its middle
    have contacts.

    In the simplest form, only the residues contained in :obj:`res_idxs_pairs`
    will be plotted (sparse=True).

    * Residue control:
        Those (and only those) residues contained in :obj:`fragments`
        will be plotted if this optional argument is parsed.
        If sparse=True, then the intersection of :obj:`res_idx_pairs`
        and :obj:`fragments` is plotted

    * Curve control:
        Many other optional parameters control whether a given
        curve is plotted or not
        TODO list


    Parameters
    ----------
    ictcs : numpy.ndarray
        Can have different shapes
        * (n)
            n is the number of residue pairs in :obj:`res_idxs_pairs`
        * (m,n)
            m is the number of frames
            In this case, an average over m will be done automatically
    res_idxs_pairs : iterable of pairs
        reside indices for which the above N contacts stand
    fragments: list of lists of integers, default is None
        The residue indices to be drawn as a circle for the
        flareplot. These *are* the dots that will be plotted
        on that circle regardless of how many contacts they
        appear in. They can be any integers that could represent
        a residue. The only hard condition is that the set
        of np.unique(res_idxs_pairs) must be contained
        within np.hstack(segments)
    exclude_neighbors: int, default is 1
        Do not show contacts where the partners are separated by
        these many residues.
        * Note: The "neighborhood-condition" is checked using
        residue serial-numbers, assuming the molecule only
        has one long peptidic-chain.
    average: boolean, default is True
        Average over T and represent the value with line transparency (alpha)
    alpha: float, defalut is 1.
        (Avanced use) fix the value of alpha regardless
        Will be ignored, however, if average is True
    freq_cutoff: float, default is 0
        Contact frequencies lower than this value will not be shown
    iax: Matplotlib axis object, default is None
        Parse an axis to draw on, otherwise one will be created
    fragment_names: iterable of strings, default is None
        The names of the segments used in :obj:`segments`
    panelsize: float, default is 4
        Size in inches of the panel (=figsize in Matplotlib).
        Will be ignored if a pre-existing axis object is parsed
    center: np.ndarray, default is [0,0]
        In axis units, where the flareplot will be centered around
    r: float, default is 1
        In axis units, the radius of the flareplot
    mute_fragments: iterable of integers, default is None
        Contacts involving these segments will be hidden. Segments
        are expressed as indices of :obj:`segments`
    anchor_segments: iterable of integers, default is None
        Only contacts involving these segments will be shown. Segments
        are expressed as indices of :obj:`segments`
    top: mdtraj.Topology object, default is None
        If provided a top, residue names (e.g. GLU30) will be used
        instead of residue indices. Will fail if the residue indices
        in :obj:`res_idxs_pairs` can not be used to call :obj:`top.residue(ii)`
    ss_dict
    angle_offset
    highlight_residxs
    pair_selection
    r2rfac: float, default is 1.1
        Fudge factor controlling the separation between labels. Still fudging
    colors: boolean, default is True
        Color control.
        * True uses one different color per segment (see visualize._mycolors)
        * False, defaults to blue. If a single string is given
        * One string uses that color for all residues (e.g. "r" or "red" for all red)
        * A list of strings of len = number of drawn residues, which is
        equal to len(np.hstack(segments)). Any other length will produce an error
        #todo perhaps this change in the future, not sure of the safest behaviour
    fontsize: int, default is 6
    return_descending_ctc_freqs#
    dotsize: float, default is 5
        Size of the dot used to represent a residue
    lw: float, default is 1
        Line width of the contact lines
    shortenAAs: boolean, default is False
        Use short AA-codes, e.g. E30 for GLU30. Only has effect if a topology
        is parsed
    Returns
    -------

    """

    padding_end += 5 #todo dangerous?

    if _np.ndim(ictcs)==1:
        ictcs = ictcs.reshape(1,-1)
    elif _np.ndim(ictcs)==2:
        pass

    else:
        raise ValueError("Input array has to of shape either "
                         "(m) or (n, m) where n : n frames, and m: n_contacts")

    assert ictcs.shape[1]==len(res_idxs_pairs), \
        "The size of the contact array and the " \
        "res_idxs_pairs array do not match %u vs %u"%(ictcs.shape[1], len(res_idxs_pairs))

    # Residue handling/bookkepping
    if sparse:
        residx_array = _np.unique(res_idxs_pairs)
    else:
        residx_array = _np.arange(_np.unique(res_idxs_pairs)[-1])+1

    if fragments is None:
        residues_as_fragments = [residx_array]
    else:
        if sparse:
            residues_as_fragments = [_np.intersect1d(ifrag,residx_array) for ifrag in fragments]
        else:
            residues_as_fragments = fragments
    residues_to_plot_as_dots = _np.hstack(residues_as_fragments)
    # Create a map
    residx2markeridx = residx2dotidx(residues_to_plot_as_dots)

    if plot_curves_only:
        assert iax is not None, ("You cannot use "
                                 "plot_curves_only=True and iax=None. Makes no sense")
    else:
        iax, xy = circle_plot_residues(residues_as_fragments,
                                       fontsize, colors,
                                       panelsize=panelsize,
                                       padding_beginning=padding_beginning,
                                       padding_end=padding_end,
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
        circle_radius_in_pts = iax.artists[0].radius * pts_per_axis_unit(iax).mean()
        lw = circle_radius_in_pts # ??

    # All formed contacts
    ctcs_averaged = _np.average(ictcs, axis=0)
    idxs_of_formed_contacts = _np.argwhere(ctcs_averaged>freq_cutoff).squeeze()

    plot_this_pair_lambda = should_this_residue_pair_get_a_curve(residues_as_fragments,
                                                                 pair_selection=pair_selection,
                                                                 mute_fragments=mute_fragments,
                                                                 anchor_fragments=anchor_segments,
                                                                 top=top, exclude_neighbors=exclude_neighbors)

    idxs_of_pairs2plot = _np.intersect1d(idxs_of_formed_contacts,
                                         [ii for ii, pair in enumerate(res_idxs_pairs) if plot_this_pair_lambda(pair)])

    alphas = ctcs_averaged[idxs_of_pairs2plot]
    pairs_of_nodes = [(xy[residx2markeridx[ii]],
                       xy[residx2markeridx[jj]]) for (ii,jj) in res_idxs_pairs[idxs_of_pairs2plot]]

    bezier_curves = add_bezier_curves(iax,
                                      pairs_of_nodes,
                                      alphas=alphas,
                                      center=center,
                                      lw=lw,
                                      bezier_linecolor=bezier_linecolor,
                                      )


def circle_plot_residues(fragments,
                         fontsize=None,
                         colors=True,
                         markersize=None,
                         r=1,
                         panelsize=4,
                         angle_offset=0,
                         padding_beginning=1,
                         padding_end=1,
                         padding_between_fragments=1,
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
    fragments
    r
    angle_offset
    padding_beginning
    padding_end
    center
    radial_padding_units
    ss_array
    plot_curves_only
    fragment_names
    iax
    residx2markeridx
    fontsize
    colors
    residues_to_plot_as_dots
    markersize
    textlabels
    shortenAAs
    highlight_residxs
    aa_offset
    top

    Returns
    -------

    """
    debug = False

    xy = cartify_fragments(fragments,
                           r=r,
                           angle_offset=angle_offset,
                           padding_initial=padding_beginning,
                           padding_final=padding_end,
                           padding_between_fragments=padding_between_fragments)
    xy += center

    n_positions = len(xy) + padding_beginning + padding_end + len(fragments)*padding_between_fragments

    # TODO review variable names
    # TODO this color mess needs to be cleaned up
    residues_to_plot_as_dots = _np.hstack(fragments)
    col_list = col_list_from_input_and_fragments(colors, residues_to_plot_as_dots, fragments=fragments)
    if iax is None:
        _plt.figure(figsize=(panelsize, panelsize), tight_layout=True)
        iax = _plt.gca()
        # Do this first to have an idea of the points per axis unit necessary for the plot
        iax.set_xlim([center[0] - r, center[0] +  r])
        iax.set_ylim([center[1] - r, center[1] +  r])

    iax.set_aspect('equal')

    # Create a map
    residx2markeridx = residx2dotidx(residues_to_plot_as_dots)

    # Plot!
    running_r_pad = 0
    if markersize is None:
        dot_radius = r * _np.sin(_np.pi/n_positions)
        dot_radius_in_pts = dot_radius*pts_per_axis_unit(iax).mean()
        if dot_radius_in_pts < 1.5:
            print(ValueError("This number of residues (%u) with this panelsize in inches (%3.1f) "
                             "forces too small dotsizes and fontsizes. Weird crowding effects "
                             "might occur. Either reduce the number of residues or increase "
                             "the panel size"%(n_positions, panelsize)))
        if not arc:
            CPs = [_CP(ixy,
                     radius=dot_radius,
                     color=col_list[ii],
                     zorder=10) for ii, ixy in enumerate(xy)]
            [iax.add_artist(iCP) for iCP in CPs]
            running_r_pad += dot_radius
        else:
            lw = r*.05*pts_per_axis_unit(iax).mean() #5% of the arc linewidth)
            draw_arcs(fragments, iax,
                      colors=[col_list[ifrag[0]] for ifrag in fragments],
                      center=center,
                      r=2*r, # don't understand this 2 here
                      angle_offset=angle_offset,
                      padding_initial=padding_beginning,
                      padding_final=padding_end,
                      padding_between_fragments=padding_between_fragments,
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
        overlap = True
        counter = 0
        while overlap and counter < n_max:
            running_r_pad += dot_radius * maxlen / 2
            [ilab.remove() for ilab in labels]
            labels = add_fragmented_residue_labels(fragments,
                                                   iax,
                                                   fontsize,
                                                   center=0,
                                                   r=r + running_r_pad,
                                                   angle_offset=angle_offset,
                                                   padding_beginning=padding_beginning,
                                                   padding_end=padding_end,
                                                   padding_between_fragments=padding_between_fragments,
                                                   shortenAAs=shortenAAs,
                                                   highlight_residxs=highlight_residxs, top=top,
                                                   aa_offset=aa_offset)
            lab_lenths = [len(itext.get_text()) for itext in labels]
            idx_longest_label = _np.argmax(lab_lenths)
            maxlen=_np.max(lab_lenths)
            bad_txt_bb = labels[idx_longest_label].get_window_extent(renderer=iax.figure.canvas.get_renderer())
            bad_dot_bb =    CPs[idx_longest_label].get_window_extent(renderer=iax.figure.canvas.get_renderer())
            overlap = bad_txt_bb.overlaps(bad_dot_bb)
            counter+=1
            #print(overlap, counter)

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
        ss_colors = [_SS2vmdcol[ss_array[res_idx]] for res_idx in residues_to_plot_as_dots]
        while overlap and counter < n_max:
            running_r_pad += dot_radius * 2
            [ilab.remove() for ilab in ss_labels]
            ss_labels = add_fragmented_residue_labels(fragments,
                                                   iax,
                                                   fontsize*2,
                                                   center=0,
                                                   r=r + running_r_pad,
                                                   angle_offset=angle_offset,
                                                   padding_beginning=padding_beginning,
                                                   padding_end=padding_end,
                                                   padding_between_fragments=padding_between_fragments,
                                                   shortenAAs=shortenAAs,
                                                   highlight_residxs=highlight_residxs, top=top,
                                                   aa_offset=aa_offset,
                                                   forced_labels=[ss_array[ii] for ii in residues_to_plot_as_dots],
                                                   colors=ss_colors,
                                                   weight="bold")
            idx_longest_label = _np.argmax([len(itext.get_text()) for itext in ss_labels])
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
    if fragment_names is not None:
        span = (2*(r + running_r_pad))
        frag_fontsize_in_aus =  span/6 * 1/5 # (average_word_length, fraction of space)
        frag_fontsize_in_pts = frag_fontsize_in_aus * pts_per_axis_unit(iax).mean()
        #frag_fontsize_in_pts = panelsize * 2
        frag_labels = add_fragment_names(iax, xy,
                           fragments,
                           fragment_names,
                           residx2markeridx,
                           fontsize=frag_fontsize_in_pts,
                           center=center,
                           r=r + running_r_pad
                           )

        frag_boxes = [fl.get_window_extent(renderer=iax.figure.canvas.get_renderer()) for fl in frag_labels]
        potential_clashing_artists = CPs+labels+ss_labels
        potential_clashing_boxes = [aa.get_window_extent(renderer=iax.figure.canvas.get_renderer()) for aa in potential_clashing_artists]
        overlapping_boxes = []
        overlapping_artists = []
        for fbox in frag_boxes:
            idxs = [ii for ii, bbox in enumerate(potential_clashing_boxes) if fbox.overlaps(bbox)]
            overlapping_artists.extend([potential_clashing_artists[ii] for ii in idxs])
            overlapping_boxes.extend([potential_clashing_boxes[ii] for ii in idxs])
        overlap = True
        counter = 0
        from mdciao.utils.str_and_dict import replace4latex
        while overlap and counter < n_max:
            [fl.remove() for fl in frag_labels]
            running_r_pad += frag_fontsize_in_aus
            frag_labels = add_fragment_names(iax, xy,
                                             fragments,
                                             [replace4latex(ifrag) for ifrag in fragment_names],
                                             residx2markeridx,
                                             fontsize=frag_fontsize_in_pts,
                                             center=center,
                                             r=r + running_r_pad
                                             )
            frag_boxes = [fl.get_window_extent(renderer=iax.figure.canvas.get_renderer()) for fl in frag_labels]
            _overlaps = [frag_boxes[0].overlaps(ibox) for ibox in potential_clashing_boxes]
            overlap = _np.any(_overlaps)
            counter +=1
        running_r_pad += frag_fontsize_in_pts / pts_per_axis_unit(iax).mean()


    iax.set_yticks([])
    iax.set_xticks([])
    iax.set_xlim([center[0] - r - running_r_pad, center[0] + r + running_r_pad])
    iax.set_ylim([center[1] - r - running_r_pad, center[1] + r + running_r_pad])
    return iax, xy

def add_bezier_curves(iax,
                      xy,
                      alphas=None,
                      center=0,
                      lw=1,
                      bezier_linecolor='k',
                      ):
    r"""
    Generate a bezier curves. Uses

    Parameters
    ----------
    iax : :obj:`matplotlib.Axes`
    xy : iterable of pairs of pairs of floats
        Each item is a pair of pairs [(x1,y1),(x2,y2)]
        to be connected with bezier curves 1<--->2
    alphas : iterable of floats, default is None
        The opacity of the curves connecting
        the pairs in :obj:`xy`.
        If None is provided, it will be set to 1
        for all curves. If provided,
        must be of len(xy)
    center
    lw
    bezier_linecolor

    Returns
    -------

    """
    if alphas is None:
        alphas = _np.ones(len(xy))

    assert len(alphas)==len(xy)

    bz_curves=[]
    for nodes, ialpha in zip(xy, alphas):
        bz_curves.append(create_flare_bezier(nodes, center=center))
        bz_curves[-1].plot(50,
                           ax=iax,
                           alpha=ialpha,
                           color=bezier_linecolor,
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
                                  padding_beginning=0,
                                  padding_end=0,
                                  padding_between_fragments=0,
                                  **add_res_labs_kwargs,
                                  ):
    r"""
    Like add_labels but starting from the fragments
    Parameters
    ----------
    fragments
    center
    r
    angle_offset
    padding_beginning
    padding_end
    iax
    residues_to_plot_as_dots
    fontsize
    shortenAAs
    highlight_residxs
    top
    aa_offset

    Returns
    -------
    labels : list of text objects

    """
    xy_labels, xy_angles = cartify_fragments(fragments,
                                             r=r,
                                             return_angles=True,
                                             angle_offset=angle_offset,
                                             padding_initial=padding_beginning,
                                             padding_final=padding_end,
                                             padding_between_fragments=padding_between_fragments)
    xy_labels += center

    residues_to_plot_as_dots = _np.hstack(fragments)
    labels = add_residue_labels(iax,
                                xy_labels, xy_angles,
                                residues_to_plot_as_dots,
                                fontsize,
                                **add_res_labs_kwargs,
                                )

    return labels
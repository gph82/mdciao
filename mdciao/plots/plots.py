import numpy as _np

from matplotlib import \
    rcParams as _rcParams, \
    pyplot as _plt

from mpl_toolkits.axes_grid1 import \
    make_axes_locatable as _make_axes_locatable

import mdciao.utils as _mdcu

def plot_w_smoothing_auto(ax, x, y,
                          label,
                          color,
                          gray_background=False,
                          n_smooth_hw=0):
    r"""
    A wrapper around :obj:`matplotlib.pyplot.plot` that allows
    to add a smoothing window (or not). See
    :obj:`window_average_fast` for more details

    Parameters
    ----------
    ax : :obj:`matplotlib.pyplot.Axes
    x : iterable of floats
    y : iterable of floats
    label : str
        Label for the legend
    color : str
        anything `matplotlib.pyplot.colors` understands
    gray_background : bool, default is False
        If True, instead of using a fainted version
        of :obj:`color`, the original :obj:`y`
        will be plotted in gray
        (useful to avoid over-coloring plots)
    n_smooth_hw : int, default is 0
        Half-size of the smoothing window.
        If 0, this method is identical to
        :obj:`matplotlib.pyplot.plot`

    Returns
    -------
    None

    """
    alpha = 1
    if n_smooth_hw > 0:
        alpha = .2
        x_smooth = _mdcu.lists.window_average_fast(_np.array(x), half_window_size=n_smooth_hw)
        y_smooth = _mdcu.lists.window_average_fast(_np.array(y), half_window_size=n_smooth_hw)
        ax.plot(x_smooth,
                y_smooth,
                label=label,
                color=color)
        label = None

        if gray_background:
            color = "gray"
    ax.plot(x, y,
            label=label,
            alpha=alpha,
            color=color)

def compare_groups_of_contacts(dictionary_of_groups,
                               colordict,
                               mutations_dict={},
                               width=.2,
                               ax=None,
                               figsize=(10, 5),
                               fontsize=16,
                               anchor=None,
                               plot_singles=False,
                               exclude=None,
                               ctc_cutoff_Ang=None,
                               **kwargs_plot_unified_freq_dicts,
                               ):
    r"""
    Compare contact groups accros different systems using different plots and strategies

    Parameters
    ----------
    dictionary_of_groups : dict
        The keys will be used as names for the contact groups, e.g. "WT", "MUT" etc
        The values can be:
          * :obj:`ContactGroup` objects
          * dictionaries where the keys are residue-pairs
          (one letter-codes, no fragment info, as in :obj:`ContactGroup.ctc_labels_short`)
          and the values are contact frequencies [0,1]
          * ascii-files (see :obj:`freq_datfile2freqdict`)
            with the contact labels in the second and frequencies in
            the third column
          * .xlsx files with the header in the second row,
            containing at least the column-names "label" and "freqs"

        Note
        ----
        If a :obj:`ContactGroup` is passed, then a :obj:`ctc_cutoff_Ang`
        needs to be passed along, otherwise frequencies cannot be computed
        on-the-fly

    colordict : dict
        Using the same keys as :obj:`dictionary_of_groups`,
        a color for each group
    anchor : str, default is None
        This string will be deleted from the contact labels,
        leaving only the partner-residue to identify the contact.
        The deletion takes place after the :obj:`mutations_dict`
        has been applied.
        No consistency-checks are carried out, i.e. use
        at your own risk
    width : float, default is .2
        The witdth of the bars
    ax : :obj:`matplotlib.pyplot.Axes`
        Axis to draw on
    figsize : tuple, default is (10,5)
        The figure size in inches, in case it is
        instantiated automatically by not passing an :obj:`ax`
    fontsize : float, default is 16
    mutations_dict : dictionary, default is {}
        A mutation dictionary that contains allows to plot together
        residues that would otherwise be identified as different
        contactacs. If there were two mutations, e.g A30K and D35A
        the mutation dictionary will be {"A30":"K30", "D35":"A35"}.
        You can also use this parameter for correcting indexing
        offsets, e.g {"GDP395":"GDP", "GDP396":"GDP"}
    plot_singles : bool, default is False
        Produce one extra figure with as many subplots as systems
        in :obj:`dictionary_of_groups`, where each system is
        plotted separately. The labels used will have been already
        "mutated" using :obj:`mutations_dict` and "anchored" using
        :obj:`anchor`. This plot is temporary and cannot be saved
    exclude
    ctc_cutoff_Ang : float, default is None
        Needed value to compute frequencies on-the-fly
        if the input was using :obj:`ContactGroup` objects
    kwargs_plot_unified_freq_dicts

    Returns
    -------
    myfig : :obj:`matplotlib.pyplot.Figure` object with the comparison plot

    freqs : dictionary of unified frequency dictionaries, including mutations and anchor

    posret : None (TODO find out why I am returning this!)

    """

    freqs = {key: {} for key in dictionary_of_groups.keys()}

    for key, ifile in dictionary_of_groups.items():
        if isinstance(ifile, str):
            idict = _mdcu.str_and_dict.freq_file2dict(ifile)
        elif all([istr in str(type(ifile)) for istr in ["mdciao", "contacts", "ContactGroup"]]):
            assert ctc_cutoff_Ang is not None, "Cannot provide a neighborhood object without a ctc_cutoff_Ang parameter"
            idict = {ilab:val for ilab, val in zip(ifile.ctc_labels_short,
                                                   ifile.frequency_per_contact(ctc_cutoff_Ang))}
        else:
            idict = {key:val for key, val in ifile.items()}

        idict = {_mdcu.str_and_dict.replace_w_dict(key, mutations_dict):val for key, val in idict.items()}
        if anchor is not None:
            idict = _mdcu.str_and_dict.delete_exp_in_keys(idict, anchor)
        freqs[key] = idict

    if plot_singles:
        nrows = len(freqs)
        myfig, myax = _plt.subplots(nrows, 1,
                                    sharey=True,
                                    sharex=True,
                                    figsize=(figsize[0], figsize[1]*nrows))
        for iax, (key, ifreq) in zip(myax, freqs.items()):
            plot_unified_freq_dicts({key: ifreq},
                                    colordict,
                                    ax=iax, width=width,
                                    fontsize=fontsize,
                                    **kwargs_plot_unified_freq_dicts)
            if anchor is not None:
                _plt.gca().text(0 - width * 2, 1.05, "%s and:" % anchor, ha="right", va="bottom")

        myfig.tight_layout()
        # _plt.show()

    freqs  = _mdcu.str_and_dict.unify_freq_dicts(freqs, exclude, defrag="@")
    myfig, iax, posret = plot_unified_freq_dicts(freqs,
                                                 colordict,
                                                 ax=ax,
                                                 width=width,
                                                 fontsize=fontsize,
                                                 figsize=figsize,
                                                 **kwargs_plot_unified_freq_dicts)
    if anchor is not None:
        _plt.text(0 - width * 2, 1.05, "%s and:" % anchor, ha="right", va="bottom")
    _plt.gcf().tight_layout()
    #_plt.show()

    return myfig, freqs, posret

"""
def add_hover_ctc_labels(iax, ctc_mat,
                         label_dict_by_index=None,
                         fmt='%3.1f',
                         hover=True,
                         cutoff=.01,
                        ):
    import mplcursors as _mplc
    assert ctc_mat.shape[0] == ctc_mat.shape[1]

    scatter_idxs_pairs = _np.argwhere(_np.abs(ctc_mat) > cutoff).squeeze()
    print("Adding %u labels" % len(scatter_idxs_pairs))
    if label_dict_by_index is None:
        fmt = '%s-%s' + '\n%s' % fmt
        labels = [fmt % (ii, jj, ctc_mat[ii, jj]) for ii, jj in scatter_idxs_pairs]
    else:
        labels = [label_dict_by_index[ii][jj] for ii, jj in scatter_idxs_pairs]

    artists = iax.scatter(*scatter_idxs_pairs.T,
                          s=.1,
                          # c="green"
                          alpha=0
                          )

    cursor = _mplc.cursor(
        pickables=artists,
        hover=hover,
    )
    cursor.connect("add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))
"""

def plot_unified_freq_dicts(freqs,
                            colordict,
                            width=.2,
                            ax=None,
                            figsize=(10, 5),
                            fontsize=16,
                            lower_cutoff_val=0,
                            sort_by='mean',
                            remove_identities=False,
                            vertical_plot=False,
                            identity_cutoff=1,
                            ylim=1
                            ):
    r"""
    Plot unified frequency dictionaries (= with identical keys) for different systems

    Parameters
    ----------
    freqs : dictionary of dictionaries
        The first-level dict is keyed by system names, e.g freqs.keys() = ["WT","D10A","D10R"]
        The second-level dict is keyed by contact names
    colordict : dict
        What color each system gets
    width : float, default is .2
        Bar width of the bar plot
    ax : :obj:`matplotlib.pyplot.Axes`, default is None

    figsize : iterable of len 2
        Figure size (x,y), in inches
        If you are transposing the figure (:obj:`vertical_plot` is True),
        you do not have to invert (y,x) this parameter here, it is
        done automatically.
    fontsize : int, default is 16
        Will be used in :obj:`matplotlib._rcParams["font.size"]
        # TODO be less invasive
    lower_cutoff_val : float, default is 0
        Do not plot values lower than this. The cutoff is applied
        to whatever property is used in :obj:`sort_by` (mean or std)
    sort_by : str, default is "mean"
        The sort_by by which to plot the contact. It is always descending
        and the property can be:
         * "mean" sort by mean frequency over all systems, making most
         frequent contacts appear on top
         * "std" sort by standard deviation over all frequencies, making
         the contacts with most different values appear on top. This
         highlights more "deviant" contacts and might hence be
         more informative than "mean" in cases where a lot of
         contacts have similar frequencies (high or low). If this option
         is activated, a faint dotted line is incorporated into the plot
         that marks the std for each contact group
         * "keep" keep the contacts in whatever order they have in the
         first dictionary
         TODO check this actually works
    remove_identities : bool, default is False
        If True, the contacts where freq[sys][ctc] = 1 across all systems
        will not be plotted nor considered in the sum over contacts
        TODO : the word identity might be confusing
    identity_cutoff : float, default is 1
        If :obj:`remove_identities`, use this value to define what
        is considered an identity, s.t. contacts with values e.g. .95
        can also be removed
        TODO conseder merging both identity paramters into one that is None or float
    vertical_plot : bool, default is False
        Plot the bars vertically in descending sort_by
        instead of horizontally (better for large number of frequencies)

    ylim

    Returns
    -------

    """
    _rcParams["font.size"] = fontsize

    #make copies of dicts
    freqs_work = {key:{key2:val2 for key2, val2 in val.items()} for key, val in freqs.items()}

    master_keys = list(freqs_work.keys())
    all_keys = list(freqs_work[master_keys[0]].keys())

    for mk in master_keys[1:]:
        assert len(all_keys)==len(list(freqs_work[mk].keys()))

    # Pop the keys for higher freqs
    keys_popped_above = []
    if remove_identities:
        for key in all_keys:
            if all([val[key]>=identity_cutoff for val in freqs_work.values()]):
                [idict.pop(key) for idict in freqs_work.values()]
                keys_popped_above.append(key)
        all_keys = [key for key in all_keys if key not in keys_popped_above]

    dicts_values_to_sort = {"mean":{},
                         "std": {},
                         "keep":{}}
    for ii, key in enumerate(all_keys):
        dicts_values_to_sort["std"][key] =  _np.std([idict[key] for idict in freqs_work.values()])*len(freqs_work)
        dicts_values_to_sort["mean"][key] = _np.mean([idict[key] for idict in freqs_work.values()])
        dicts_values_to_sort["keep"][key] = len(all_keys)-ii+lower_cutoff_val # trick to keep the same logic

    # Find the keys lower sort property
    keys_popped_below = []
    for ii, (key, val) in enumerate(dicts_values_to_sort[sort_by].items()):
        if val <= lower_cutoff_val:
            [idict.pop(key) for idict in freqs_work.values()]
            keys_popped_below.append(key)
            all_keys.remove(key)

    # Pop them form the needed dict
    final_ordered_dict = {key:val for key, val in dicts_values_to_sort[sort_by].items() if key not in keys_popped_below}

    # Sort the dict for plotting
    final_ordered_dict = {key:val for (key, val)  in
                          sorted(final_ordered_dict.items(),
                                 key = lambda item : item[1],
                                 reverse = True
                          )
                          }

    # Prepare the positions of the bars
    delta = {}
    for ii, key in enumerate(master_keys):
        delta[key] = width * ii

    if ax is None:
        if vertical_plot:
            figsize = figsize[::-1]
        myfig = _plt.figure(figsize=figsize)
    else:
        _plt.sca(ax)
        myfig = ax.figure

    for ii, (key, val) in enumerate(final_ordered_dict.items()):
        for jj, (skey, sfreq) in enumerate(freqs_work.items()):
            if ii == 0:
                label = '%s (Sigma= %2.1f)'%(skey, _np.sum(list(sfreq.values())))
                if len(keys_popped_above)>0:
                    label = label[:-1]+", +%2.1f above threshold)"%(_np.sum([freqs[skey][nskey] for nskey in keys_popped_above]))
                if len(keys_popped_below) > 0:
                    label = label[:-1] + ", +%2.1f below threshold)" % (
                        _np.sum([freqs[skey][nskey] for nskey in keys_popped_below]))
                label = _mdcu.str_and_dict.replace4latex(label)

            else:
                label = None

            if not vertical_plot:
                _plt.bar(ii + delta[skey], sfreq[key],
                         width=width,
                         color=colordict[skey],
                         label=label,
                         )
                if jj == 0:
                    _plt.text(ii, ylim+.05, key, rotation=45)
            else:
                _plt.barh(ii + delta[skey], sfreq[key],
                          height=width,
                          color=colordict[skey],
                          label=label,
                          )
                if jj == 0:
                    _plt.text(-.05, ii, key, ha='right', va="center") # TODO fix the vertical position here
    _plt.legend()
    if vertical_plot:
        _plt.yticks([])
        _plt.xlim(0, ylim)
        _plt.ylim(0 - width, ii + width * 2)
        _plt.xticks([0, .25, .50, .75, 1])
        [_plt.gca().axvline(ii, ls=":", color="k", zorder=-1) for ii in [.25, .5, .75]]
        _plt.gca().invert_yaxis()

        if sort_by == "std":
            _plt.plot(list(final_ordered_dict.values()), _np.arange(len(all_keys)), color='k', alpha=.25, ls=':')

    else:
        _plt.xticks([])
        _plt.xlim(0 - width, ii + width * len(final_ordered_dict))
        if ylim<=1:
            yticks = [0, .25, .50, .75, 1]

        else:
            yticks = _np.arange(0,_np.ceil(ylim),.50)
        _plt.yticks(yticks)
        [_plt.gca().axhline(ii, ls=":", color="k", zorder=-1) for ii in yticks[1:-1]]
        if sort_by == "std":
            _plt.plot(list(final_ordered_dict.values()),
                      color='k', alpha=.25, ls=':')

        _plt.ylim(0, ylim)


    return myfig, _plt.gca(), None #{all_keys[ii]:[idict[all_keys[ii]] for idict in freqs_work.values()] for ii in order4plot}

def add_tilted_labels_to_patches(jax, labels,
                                 label_fontsize_factor=1,
                                 trunc_y_labels_at=.65):
    r"""
    Iterate through :obj:`jax.patches` and place the text strings
    in :obj:`labels` on top of it.

    Parameters
    ----------
    jax
    labels
    label_fontsize_factor
    trunc_y_labels_at

    Returns
    -------

    """
    for ii, (ipatch, ilab) in enumerate(zip(jax.patches, labels)):
        ix = ii
        iy = ipatch.get_height()
        iy += .01
        if iy > trunc_y_labels_at:
            iy = trunc_y_labels_at
        jax.text(ix, iy, _mdcu.str_and_dict.replace4latex(ilab),
                 va='bottom',
                 ha='left',
                 rotation=45,
                 fontsize=_rcParams["font.size"]*label_fontsize_factor,
                 backgroundcolor="white"
                 )

def _get_highest_y_of_bbox_in_axes_units(txt_obj):
    r"""
    For an input text object, get the highest y-value of its bounding box in axis units

    Goal: Find out if a text box overlaps with the title. Useful for rotated texts of variable
    length.

    There are mpl methods (contains or overlaps) but they do not return the coordinate

    Parameters
    ----------
    txt_obj : :obj:`matplotlib.text.Text` object

    Returns
    -------
    y : float

    """
    jax  : _plt.Axes =  txt_obj.axes
    try:
        bbox = txt_obj.get_window_extent()
    except RuntimeError as e:
        jax.figure.tight_layout()
        bbox = txt_obj.get_window_extent()
    tf_inv_y = jax.transAxes.inverted()
    y = tf_inv_y.transform(bbox)[-1, 1]
    #print(bbox)
    #print(y)
    return y

def _dataunits2points(jax):
    r"""
    Return a conversion factor for points 2 dataunits
    Parameters
    ----------
    jax : obj:`matplotlib.Axes`

    Returns
    -------
    float : p2d
        Conversion factor so that points * p2d = points_in_dataunits

    """
    bbox = jax.get_window_extent()
    dy_in_points = bbox.bounds[3]
    dy_in_dataunits = _np.diff(jax.get_ylim())[0]
    return dy_in_points / dy_in_dataunits

def titlepadding_in_points_no_clashes_w_texts(jax, min_pts4correction=6):
    r"""
    Compute amount of upward padding need to avoid overlap between
    he axis title and any text object in the axis

    Parameters
    ----------
    jax : :obj:`matplotlib.Axis`
    min_pts4correction : int, default is 4
        Do not consider extensions smaller than this to
        need correction. Helps with multiple axis in the same fig

    Returns
    -------
    pad_id_points : float

    """

    max_y_texts = _np.max([_get_highest_y_of_bbox_in_axes_units(txt) for txt in jax.texts])
    dy = max_y_texts - jax.get_ylim()[1]
    data2pts = _dataunits2points(jax)
    pad_in_points = _np.max([0,dy])*data2pts
    if pad_in_points < min_pts4correction:
        pad_in_points = 0

    return pad_in_points

def plot_contact_matrix(mat, labels, pixelsize=1,
                        transpose=False, grid=False,
                        cmap="binary",
                        colorbar=False):
    r"""
    Plot a contact matrix. It is written to be able to
    plot rectangular matrices where rows and columns
    do not represent the same residues

    Parameters
    ----------
    mat : 2D numpy.ndarray of shape (N,M)
        The allowed values are in [0,1], else
        the method fails (NaNs are allowed)
    labels : list of len(2) with x and y labels
        The length of each list has to be N, M for
        x, y respectively, else this method fails
    pixelsize : int, default is 1
        The size in inches of the pixel representing
        the contact. Ultimately controls the size
        of the figure, because
        figsize = _np.array(mat.shape)*pixelsize
    transpose : boolean, default is False
    grid : boolean, default is False
        overlap a grid of dashed lines
    cmap : str, default is binary
        What :obj:`matplotlib.cmap` to use
    colorbar : boolean, default is False
        whether to use a colorbar

    Returns
    -------
    ax : :obj:`matplotlib.pyplot.Axes` object
    pixelsize : float, size of the pixel
        Helpful in cases where this method is called
        with the default value, in case the value
        changes in the future
    """
    _np.testing.assert_array_equal(mat.shape,[len(ll) for ll in labels])
    assert _np.nanmax(mat)<=1 and _np.nanmin(mat)>=0, (_np.nanmax(mat), _np.nanmin(mat))
    if transpose:
        mat = mat.T
        labels = labels[::-1]

    _plt.figure(figsize = _np.array(mat.shape)*pixelsize)
    im = _plt.imshow(mat,cmap=cmap)
    fig,iax = _plt.gcf(), _plt.gca()
    _plt.ylim([len(labels[0])-.5, -.5])
    _plt.xlim([-.5, len(labels[1])-.5])
    _plt.yticks(_np.arange(len(labels[0])),labels[0],fontsize=pixelsize*20)
    _plt.xticks(_np.arange(len(labels[1])), labels[1],fontsize=pixelsize*20,rotation=90)

    if grid:
        _plt.hlines(_np.arange(len(labels[0]))+.5,-.5,len(labels[1]),ls='--',lw=.5, color='gray', zorder=10)
        _plt.vlines(_np.arange(len(labels[1])) + .5, -.5, len(labels[0]), ls='--', lw=.5,  color='gray', zorder=10)

    if colorbar:
        # from https://stackoverflow.com/a/18195921
        divider = _make_axes_locatable(iax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        _plt.gcf().colorbar(im, cax=cax)
        im.set_clim(0.0, 1.0)
    fig.tight_layout()
    return iax, pixelsize
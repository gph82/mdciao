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

from matplotlib import \
    rcParams as _rcParams, \
    pyplot as _plt

from mpl_toolkits.axes_grid1 import \
    make_axes_locatable as _make_axes_locatable

import mdciao.utils as _mdcu

from os import path as _path

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
    ax : :obj:`~matplotlib.axes.Axes`
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

def compare_groups_of_contacts(groups,
                               colors=None,
                               mutations_dict={},
                               width=.2,
                               ax=None,
                               figsize=(10, 5),
                               fontsize=16,
                               anchor=None,
                               plot_singles=False,
                               exclude=None,
                               ctc_cutoff_Ang=None,
                               AA_format='short',
                               defrag='@',
                               per_residue=False,
                               title='comparison',
                               **kwargs_plot_unified_freq_dicts,
                               ):
    r"""
    Compare contact groups accros different systems using different plots and strategies

    Parameters
    ----------
    groups : iterable (list or dict)
        The contact groups. If dict, the keys will be used as names
        for the contact groups, e.g. "WT", "MUT" etc, if list the keys
        will be auto-generated.
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

    colors : iterable (list or dict), default is None
        Using the same keys as :obj:`dictionary_of_groups`,
        a color for each group. Defaults to some sane matplotlib choices
    anchor : str, default is None
        This string will be deleted from the contact labels,
        leaving only the partner-residue to identify the contact.
        The deletion takes place after the :obj:`mutations_dict`
        has been applied. The final anchor label will be that
        of the deleted keys (allows for keeping e.g. pre-existing
        consensus nomenclature).
        No consistency-checks are carried out, i.e. use
        at your own risk
    width : float, default is .2
        The witdth of the bars
    ax : :obj:`~matplotlib.axes.Axes`
        Axis to draw on
    figsize : tuple, default is (10,5)
        The figure size in inches, in case it is
        instantiated automatically by not passing an :obj:`ax`
    fontsize : float, default is 16
    mutations_dict : dictionary, default is {}
        A mutation dictionary that contains allows to plot together
        residues that would otherwise be identified as different
        contacts. If there were two mutations, e.g A30K and D35A
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
    AA_format : str, default is "short"
        see :obj:`ContactPair.frequency_dict` for more info
    defrag : str, default is "@"
        see :obj:`_mdcu.str_and_dict.unify_freq_dicts` for more info
    per_residue : bool, default is False
        Unify dictionaries by residue and not by pairs
    kwargs_plot_unified_freq_dicts : kwargs
        remove_identities

    Returns
    -------
    myfig : :obj:`matplotlib.pyplot.Figure` object with the comparison plot

    freqs : dictionary of unified frequency dictionaries, including mutations and anchor

    plotted_freqs : the unified freq dictionary sorted and purged like the one in the plot

    """
    if isinstance(groups, dict):
        pass
    else:
        _groups = {}
        for ii, item in enumerate(groups):
            if isinstance(item,str):
                key = _path.splitext(_path.basename(item))[0]
            elif isinstance(item,dict):
                key = "dict"
            else:
                key = "mdcCG"
            _groups["%s (%u)"%(key,ii)]=item
        groups = _groups

    freqs = {key: {} for key in groups.keys()}
    if isinstance(colors, list):
        assert len(colors) >= len(freqs)
        colors = {key:val for key, val in zip(freqs.keys(), colors)}

    for key, ifile in groups.items():
        if isinstance(ifile, str):
            idict = _mdcu.str_and_dict.freq_file2dict(ifile)
        elif all([istr in str(type(ifile)) for istr in ["mdciao", "contacts", "ContactGroup"]]):
            assert ctc_cutoff_Ang is not None, "Cannot provide a ContatGroup object without a ctc_cutoff_Ang parameter"
            idict = ifile.frequency_dicts(ctc_cutoff_Ang=ctc_cutoff_Ang,
                                         AA_format=AA_format,
                                         split_label=False)
        else:
            idict = {key:val for key, val in ifile.items()}

        idict = {_mdcu.str_and_dict.replace_w_dict(key, mutations_dict):val for key, val in idict.items()}

        if anchor is not None:
            idict, deleted_half_keys = _mdcu.str_and_dict.delete_exp_in_keys(idict, anchor)
            if len(_np.unique(deleted_half_keys))>1:
                raise ValueError("The anchor patterns differ by key, this is strange: %s"%deleted_half_keys)
            else:
                anchor=_mdcu.str_and_dict.defrag_key(deleted_half_keys[0],defrag=defrag,sep=" ")
        freqs[key] = idict
    if plot_singles:
        nrows = len(freqs)
        myfig, myax = _plt.subplots(nrows, 1,
                                    sharey=True,
                                    sharex=True,
                                    figsize=(figsize[0], figsize[1]*nrows))
        for iax, (key, ifreq) in zip(myax, freqs.items()):
            plot_unified_freq_dicts({key: ifreq},
                                    colordict=colors,
                                    ax=iax, width=width,
                                    fontsize=fontsize,
                                    **kwargs_plot_unified_freq_dicts)
            if anchor is not None:
                _plt.gca().text(0 - _np.abs(_np.diff(_plt.gca().get_xlim())) * .05, 1.05,
                                "%s and:" % _mdcu.str_and_dict.latex_superscript_fragments(anchor),
                                ha="right", va="bottom")

        myfig.tight_layout()
        # _plt.show()
    freqs  = _mdcu.str_and_dict.unify_freq_dicts(freqs, exclude, per_residue=per_residue, defrag=defrag)
    if per_residue:
        kwargs_plot_unified_freq_dicts["ylim"]= _np.max([_np.max(list(ifreqs.values())) for ifreqs in freqs.values()])
        kwargs_plot_unified_freq_dicts["remove_identities"] = False

    if ctc_cutoff_Ang is not None:
        title = title+'@%2.1f AA'%ctc_cutoff_Ang
    if anchor is not None:
        title+="\n%s and " % _mdcu.str_and_dict.latex_superscript_fragments(anchor)

    myfig, iax, plotted_freqs = plot_unified_freq_dicts(freqs,
                                                        colordict=colors,
                                                        ax=ax,
                                                        width=width,
                                                        fontsize=fontsize,
                                                        figsize=figsize,
                                                        title=title,
                                                        **kwargs_plot_unified_freq_dicts)

    _plt.gcf().tight_layout()
    #_plt.show()

    return myfig, freqs, plotted_freqs

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
                            colordict=None,
                            width=.2,
                            ax=None,
                            figsize=(10, 5),
                            panelheight_inches=5,
                            inch_per_contacts=1,
                            fontsize=16,
                            lower_cutoff_val=0,
                            sort_by='mean',
                            remove_identities=False,
                            vertical_plot=False,
                            identity_cutoff=1,
                            ylim=1,

                            assign_w_color=False,
                            title=None,
                            verbose_legend=True,
                            legend_rows=4,

                            ):
    r"""
    Plot unified frequency dictionaries (= with identical keys) for different systems

    Parameters
    ----------
    freqs : dictionary of dictionaries
        The first-level dict is keyed by system names,
        e.g freqs.keys() = ["WT","D10A","D10R"]
        The second-level dict is keyed by contact names
    colordict : dict, default is None.
        What color each system gets. Default is some sane matplotlib values
    width : None or float, default is .2
        Bar width each bar in the plot.
        If None, .8/len(freqs) will be used, leaving
        a .1 gap of free space between contacts.
    ax : :obj:`~matplotlib.axes.Axes`, default is None
        Plot into this axis, else create one using
        :obj:`figsize`.
    figsize : iterable of len 2
        Figure size (x,y), in inches. If None,
        one will be created using :obj:`panelheight_inches`
        and :obj:`inch_per_contacts`.
        If you are transposing the figure
        using :obj:`vertical_plot`, you do not
        have to invert (y,x) this parameter here, it is
        done automatically.
    panelheight_inches : int, default is 5
    inch_per_contacts : int, default is 1
    fontsize : int, default is 16
        Will be used in :obj:`matplotlib._rcParams["font.size"]
        # TODO be less invasive
    lower_cutoff_val : float, default is 0
        Do not plot values lower than this. The cutoff is applied
        to whatever property is used in :obj:`sort_by` (mean or std)
    sort_by : str, default is "mean"
        The property by which to sort the contacts.
        It is always descending and the property can be:
         * "mean" sort by mean frequency over all systems, making most
           frequent contacts appear on the left/top of the plot.
         * "std" sort by standard deviation over all frequencies, making
           the contacts with most different values appear on top. This
           highlights more "deviant" contacts and might hence be
           more informative than "mean" in cases where a lot of
           contacts have similar frequencies (high or low). If this option
           is activated, a faint dotted line is incorporated into the plot
           that marks the std for each contact group
         * "keep" keep the contacts in whatever order they have in the
           first dictionary
    remove_identities : bool, default is False
        If True, the contacts where freq[sys][ctc] = 1 across all systems
        will not be plotted nor considered in the sum over contacts
        TODO : the word identity might be confusing
    identity_cutoff : float, default is 1
        If :obj:`remove_identities`, use this value to define what
        is considered an identity, s.t. contacts with values e.g. .95
        can also be removed
        TODO consider merging both identity parameters into one that is None or float
    vertical_plot : bool, default is False
        Plot the bars vertically in descending sort_by
        instead of horizontally (better for large number of frequencies)
    ylim : float, default is 1
        The limit on the y-axis
    assign_w_color : boolean, default is False
        If there are contacts where only one system (as in keys, of :obj:`freqs`)
        appears, color the textlabel of that contact with the system's color
    legend_rows : int, default is 4
        The maximum number of rows per column of the legend.
        If you have 10 systems, :obj:`legend_rows`=5 means
        you'll get two columns, =2 means you'll get five.
    verbose_legend : bool, default is True
        Verbose legends inform about
        contacts that were in the input but
        have been left out of the plot. Contacts
        are left out if they are:
         * above the :obj:`identity_cutoff` or
         * below the :obj:`lower_cutoff_val`
        They will appear in the verbose legend
        as "+ A.a + B.b", respectively
        denoting the missing contacts that are
        "a(bove" and b(elow)" with their respective
        sums "A" and "B".

    Returns
    -------
    fig : :obj:`~matplotlib.figure.Figure`
    ax : :obj:`~matplotlib.axes.Axes`
    freqs : dict
        Dictionary of dictionaries with the plotted frequencies
        in the plotted order. It's keyed with first wity
        system-names first and contact-names second, like
        the input. It has the :obj:`sort_by` strategy
        as an extra key containing the value that resorted
        of that strategy for each contact-name.

    """
    _fontsize=_rcParams["font.size"]
    _rcParams["font.size"] = fontsize
    #make copies of dicts
    freqs_by_sys_by_ctc = {key:{key2:val2 for key2, val2 in val.items()} for key, val in freqs.items()}

    system_keys = list(freqs_by_sys_by_ctc.keys())
    all_ctc_keys = list(freqs_by_sys_by_ctc[system_keys[0]].keys())
    for sk in system_keys[1:]:
        assert len(all_ctc_keys)==len(list(freqs_by_sys_by_ctc[sk].keys())), "This is not a unified dictionary"

    # Pop the keys for higher freqs
    keys_popped_above = []
    if remove_identities:
        for key in all_ctc_keys:
            if all([val[key]>=identity_cutoff for val in freqs_by_sys_by_ctc.values()]):
                [idict.pop(key) for idict in freqs_by_sys_by_ctc.values()]
                keys_popped_above.append(key)
        all_ctc_keys = [key for key in all_ctc_keys if key not in keys_popped_above]

    dicts_values_to_sort = {"mean":{},
                         "std": {},
                         "keep":{}}
    for ii, key in enumerate(all_ctc_keys):
        dicts_values_to_sort["std"][key] =  _np.std([idict[key] for idict in freqs_by_sys_by_ctc.values()])*len(freqs_by_sys_by_ctc)
        dicts_values_to_sort["mean"][key] = _np.mean([idict[key] for idict in freqs_by_sys_by_ctc.values()])
        dicts_values_to_sort["keep"][key] = len(all_ctc_keys)-ii+lower_cutoff_val # trick to keep the same logic

    # Find the keys lower sort property
    ctc_keys_popped_below = []
    for ii, (ctc, val) in enumerate(dicts_values_to_sort[sort_by].items()):
        if val <= lower_cutoff_val:
            [idict.pop(ctc) for idict in freqs_by_sys_by_ctc.values()]
            ctc_keys_popped_below.append(ctc)
            all_ctc_keys.remove(ctc)

    # Pop them form the needed dict
    sorted_value_by_ctc_by_sys = {key:val for key, val in dicts_values_to_sort[sort_by].items() if key not in ctc_keys_popped_below}

    # Prepare the dict that stores the order for plotting
    # and the values used for that sorting
    sorted_value_by_ctc_by_sys = {key:val for (key, val)  in
                                 sorted(sorted_value_by_ctc_by_sys.items(),
                                 key = lambda item : item[1],
                                 reverse = True
                          )
                          }

    # Prepare the dict
    if colordict is None:
        colordict = {key:val for key,val in zip(system_keys, _colorstring.split(","))}
    winners = _color_by_values(all_ctc_keys, freqs_by_sys_by_ctc, colordict,
                               lower_cutoff_val=lower_cutoff_val, assign_w_color=assign_w_color)

    # Prepare the positions of the bars
    wpad=.2
    maxwidth = (1-wpad)/len(system_keys)
    if width is None:
        width=maxwidth
    else:
        width=_np.min([width,maxwidth])
    imax = (len(system_keys)-1)*width/2
    ls = _np.linspace(-imax, imax, len(system_keys))
    delta={key:val for key, val in zip(system_keys, ls)}

    if ax is None:
        if figsize is None:
            y_figsize=panelheight_inches
            x_figsize=inch_per_contacts*len(sorted_value_by_ctc_by_sys)
            figsize = [x_figsize,y_figsize]
        if vertical_plot:
            figsize = figsize[::-1]
        myfig = _plt.figure(figsize=figsize)
        ax = _plt.gca()
    else:
        myfig = ax.figure
        _plt.sca(ax)
    # Visual aides for debugging one-off errors in labelling, bar-position, and bar-width
    #_plt.axvline(.5-wpad/2,color="r")
    #_plt.axvline(-.5+wpad/2, color="r")
    for jj, (skey, sfreq) in enumerate(freqs_by_sys_by_ctc.items()):
        # Sanity check
        assert len(sfreq) == len(sorted_value_by_ctc_by_sys), "This shouldnt happen"

        bar_array = [sfreq[key] for key in sorted_value_by_ctc_by_sys.keys()]
        x_array = _np.arange(len(bar_array))

        # Label
        label = '%s (Sigma= %2.1f)'%(skey, _np.sum(list(sfreq.values())))
        if verbose_legend:
            if len(keys_popped_above)>0:
                extra = "above threshold"
                f = identity_cutoff
                label = label[:-1]+", +%2.1fa)"%\
                        (_np.sum([freqs[skey][nskey] for nskey in keys_popped_above]))
            if len(ctc_keys_popped_below) > 0:
                not_shown_sigma = _np.sum([freqs[skey][nskey] for nskey in ctc_keys_popped_below])
                if not_shown_sigma>0:
                    extra = "below threshold"
                    f = lower_cutoff_val
                    label = label[:-1] + ", +%2.1fb)" % (not_shown_sigma)
        label = _mdcu.str_and_dict.replace4latex(label)

        if len(bar_array)>0:
            if not vertical_plot:
                _plt.bar(x_array + delta[skey], bar_array,
                         width=width,
                         color=colordict[skey],
                         label=label,
                         )
            else:
                _plt.barh(x_array + delta[skey], bar_array,
                          height=width,
                          color=colordict[skey],
                          label=label,
                          )


            _plt.legend(ncol=_np.ceil(len(system_keys) / legend_rows).astype(int))

    if vertical_plot:
        for ii, key in enumerate(sorted_value_by_ctc_by_sys.keys()):
            # 1) centered in the middle of the bar, since plt.bar(align="center")
            # 2) displaced by one half width*nbars
            iix = ii \
                  - width / 2 \
                  + len(freqs_by_sys_by_ctc) * width / 2
            _plt.text(0 - .05, iix, key,
                      ha="right",
                      #rotation=45,
                      )
        _plt.yticks([])
        _plt.xlim(0, ylim)
        _plt.ylim(0 - width, ii + width * len(freqs_by_sys_by_ctc))
        _plt.xticks([0, .25, .50, .75, 1])
        [_plt.gca().axvline(ii, ls=":", color="k", zorder=-1) for ii in [.25, .5, .75]]
        _plt.gca().invert_yaxis()

        if sort_by == "std":
            _plt.plot(list(sorted_value_by_ctc_by_sys.values()), _np.arange(len(all_ctc_keys)), color='k', alpha=.25, ls=':')

    else:
        for ii, key in enumerate(sorted_value_by_ctc_by_sys.keys()):
            # 1) centered in the middle of the bar, since plt.bar(align="center")
            # 2) displaced by one half width*nbars
            iix = ii\
                  -width/2\
                  +len(freqs_by_sys_by_ctc)*width/2
            txt = _mdcu.str_and_dict.latex_superscript_fragments(key)
            txt = winners[key][0] + txt
            _plt.text(iix, ylim + .05, txt,
                      #ha="center",
                      ha='left',
                      rotation=45,
                      color=winners[key][1]
                      )
            #_plt.gca().axvline(iix) (visual aid)
        _plt.xticks(_np.arange(len(sorted_value_by_ctc_by_sys))-width, [])

        _plt.xlim(0 - width, ii + width * len(freqs_by_sys_by_ctc))
        _ax = ax.twiny()
        _ax.set_xlim(ax.get_xlim())
        _plt.xticks(ax.get_xticks(), [])
        _plt.sca(ax)
        if ylim<=1:
            yticks = [0, .25, .50, .75, 1]
        else:
            yticks = _np.arange(0,_np.ceil(ylim),.50)
        _plt.yticks(yticks)
        [_plt.gca().axhline(ii, ls=":", color="k", zorder=-1) for ii in yticks[1:-1]]
        if sort_by == "std":
            _plt.plot(list(sorted_value_by_ctc_by_sys.values()),
                      color='k', alpha=.25, ls=':')

        _plt.ylim(0, ylim)
        if title is not None:
            pad = _np.max([txt.get_window_extent(ax.figure.canvas.get_renderer()).height for txt in ax.texts])
            ax.set_title(_mdcu.str_and_dict.replace4latex(title),
                         pad=pad+_rcParams["axes.titlepad"]*2
                         )

    # Create a by-state dictionary explaining the plot
    out_dict = {key:{ss: val[ss] for ss in sorted_value_by_ctc_by_sys.keys()} for key, val in freqs_by_sys_by_ctc.items()}
    out_dict.update({sort_by: {key : _np.round(val,2) for key, val in sorted_value_by_ctc_by_sys.items()}})

    _rcParams["font.size"] = _fontsize
    return myfig, _plt.gca(),  out_dict

def add_tilted_labels_to_patches(jax, labels,
                                 label_fontsize_factor=1,
                                 trunc_y_labels_at=.65,
                                 single_label=False):
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
        if single_label:
            txt = _mdcu.str_and_dict._latex_superscript_one_fragment(ilab)
        else:
            txt = _mdcu.str_and_dict.latex_superscript_fragments(ilab)
        jax.text(ix, iy, txt,
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

def _points2dataunits(jax):
    r"""
    Return a conversion factor for points 2 dataunits
    Parameters
    ----------
    jax : obj:`~matplotlib.axes.Axes``

    Returns
    -------
    p2d : float
        Conversion factor so that points * p2d = points_in_dataunits

    """
    bbox = jax.get_window_extent()
    dx_pts, dy_pts = bbox.bounds[-2:]
    dx_in_dataunits, dy_in_dataunits = _np.diff(jax.get_xlim())[0], _np.diff(jax.get_ylim())[0]
    return _np.array((dx_pts/dx_in_dataunits, dy_pts / dy_in_dataunits)).T

def highest_y_textobjects_in_Axes_units(ax):
    r"""
    Return the highest y-value of the bounding boxes of the text objects, in Axes units

    Axes units are 0 at the left/bottom and 1 at the right/top of the axes (xlim, ylim)
    For more info
    https://matplotlib.org/3.1.1/tutorials/advanced/transforms_tutorial.html

    Parameters
    ----------
    ax : :obj:`~matplotlib.axes.Axes`

    Returns
    -------
    y : float
    """

    rend = ax.figure.canvas.get_renderer()
    return _np.max(
        [ax.transAxes.inverted().transform(txt.get_window_extent(rend).corners()[-1])[-1]
         for txt in ax.texts]
    )

#TODO deprecate
def titlepadding_in_points_no_clashes_w_texts(jax, min_pts4correction=6):
    r"""
    Compute amount of upward padding need to avoid overlap between
    the axis title and any text object in the axis

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

    max_y_texts = _np.max([_get_highest_y_of_bbox_in_axes_units(txt) for txt in jax.texts]+[0])
    dy = max_y_texts - jax.get_ylim()[1]
    data2pts = _points2dataunits(jax)[1]
    pad_in_points = _np.max([0,dy])*data2pts
    if pad_in_points < min_pts4correction:
        pad_in_points = 0

    return pad_in_points


def CG_panels(n_cols, CG_dict, ctc_cutoff_Ang,
               draw_empty=True,
               distro=False,
               short_AA_names=False,
               plot_atomtypes=False,
               switch_off_Ang=0,
               panelsize=4,
               panelsize2font=3.5,
               verbose=False):
    r"""
    One figure with each obj:`~mdciao.contacts.ContactGroup` as individual panel

    Wraps around plot_distance_distributions, plot_neighborhood_freqs

    Parameters
    ----------
    n_cols : int
        number of columns of the subplot
    CG_dict : dict
        dictionary of
        obj:`~mdciao.contacts.ContactGroup` objects
    ctc_cutoff_Ang : float
    draw_empty : bool, default is True
        To give a visual cue that some
        CGs are empty, the corresponding
        panels are drawn empty (it also
        ensures same panel position regardless
        of the result).
    distro : bool, default is False

    short_AA_names
    plot_atomtypes
    switch_off_Ang

    Returns
    -------
    fig : :obj:`~matplotlib.Figure`

    """
    if draw_empty:
        n_panels = len(CG_dict)
    else:
        n_panels = len([CG for CG in CG_dict.values() if CG is not None])

    n_cols = _np.min((n_cols, n_panels))
    n_rows = _np.ceil(n_panels / n_cols).astype(int)

    bar_fig, bar_ax = _plt.subplots(n_rows, n_cols,
                                    sharex=True,
                                    sharey=True,
                                    figsize=(n_cols * panelsize * 2, n_rows * panelsize), squeeze=False)

    # One loop for the histograms
    _rcParams["font.size"] = panelsize * panelsize2font
    for jax, (iname, ihood) in zip(bar_ax.flatten(),
                                   CG_dict.items()):
        if ihood is not None:
            if distro:
                ihood.plot_distance_distributions(nbins=20,
                                                  jax=jax,
                                                  label_fontsize_factor=panelsize2font / panelsize,
                                                  shorten_AAs=short_AA_names,
                                                  ctc_cutoff_Ang=ctc_cutoff_Ang,
                                                  )
            else:
                xmax = _np.max([ihood.n_ctcs for ihood in CG_dict.values() if
                                ihood is not None])
                if ihood.is_neighborhood:
                    ihood.plot_neighborhood_freqs(ctc_cutoff_Ang,
                                                  switch_off_Ang=switch_off_Ang,
                                                  ax=jax,
                                                  xmax=xmax,
                                                  label_fontsize_factor=panelsize2font / panelsize,
                                                  shorten_AAs=short_AA_names,
                                                  color=ihood.partner_fragment_colors,
                                                  plot_atomtypes=plot_atomtypes
                                                  )
                else:
                    ihood.plot_freqs_as_bars(ctc_cutoff_Ang, iname,
                                             ax=jax,
                                             xlim=xmax,
                                             label_fontsize_factor=panelsize2font / panelsize,
                                             shorten_AAs=short_AA_names,
                                             atom_types=plot_atomtypes,
                                             )
                    if verbose:
                        print()
                        print(ihood.frequency_dataframe(ctc_cutoff_Ang).round({"freq": 2, "sum": 2}))
                        print()

    if not distro:
        non_nan_rightermost_patches = [[p for p in jax.patches if not _np.isnan(p.get_x())][-1] for jax in
                                       bar_ax.flatten() if len(jax.patches) > 0]
        xmax = _np.nanmax([p.get_x() + p.get_width() / 2 for p in non_nan_rightermost_patches]) + .5
        [iax.set_xlim([-.5, xmax]) for iax in bar_ax.flatten()]
    bar_fig.tight_layout(h_pad=2, w_pad=0, pad=0)

    return bar_fig

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
    ax : :obj:`~matplotlib.axes.Axes` object
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

_colorstring = 'tab:blue,tab:orange,tab:green,tab:red,tab:purple,tab:brown,tab:pink,tab:gray,tab:olive,tab:cyan'


def _color_by_values(all_ctc_keys, freqs_by_sys_by_ctc, colordict,
                     lower_cutoff_val=0,
                     assign_w_color=True):
    r""" Helper method to decide how represent a contact text label:

    How the label is presented depends on:
        * If all freqs are zero except one: the text label gets
        assigned the "winner"'s color and the sign '+'
        * If all are non-zero except one, the text label gets
        assinged the "looser"'s color and the sign '-'

    For the method to work, :obj:`freqs_by_sys_by_ctc` has
    to be a unified frequency dictionary, i.e. all of the
    sub-dicts here have to have the same keys

    Parameters
    ----------
    all_ctc_keys : iterable of strings
        The contact keys that :obj:`freqs_by_sys_by_ctc` use
    freqs_by_sys_by_ctc : dict of dicts
        Unified frequency dictionary, keyed first by system
        and secondly with the keys in all_ctc_keys

    #TODO better variable naming, some things could be done outside the method
    """

    winners = {}
    for ctc_key in all_ctc_keys:
        system_keys = list(freqs_by_sys_by_ctc.keys())
        _vals = _np.array([val[ctc_key] for val in freqs_by_sys_by_ctc.values()])
        idx_loosers = _np.flatnonzero(_vals <= lower_cutoff_val)
        idx_winners = _np.flatnonzero(_vals > lower_cutoff_val)
        winners[ctc_key] = ("", "k")
        if assign_w_color:
            if  len(idx_winners) == 1:
                winners[ctc_key] = ('+', colordict[system_keys[_vals.argmax()]])
            else:
                if len(idx_loosers) == 1:
                    winners[ctc_key] = ('-', colordict[system_keys[_vals.argmin()]])

    return winners

def _plot_freqbars_baseplot(freqs,
                            jax=None,
                            truncate_at=None,
                            bar_width_in_inches=.75,
                            color=["tab:blue"],
                            ):
    r"""
    Base method for plotting the contact frequencies

    Parameters
    ----------
    freqs : iterable
        The values to plot
    jax : :obj:`~matplotlib.axes.Axes`, default is None
        If None is passed, one will be created
    truncate_at : float, default is None
        Only plot frequencies above this value (between 0 and 1)
    bar_width_in_inches : float, default is .75
        The width of the axis will vary with the number of plotted
        frequencies. This allows for plotting different :obj:`ContactGroup`
        objects each with different number of contacts and still appear
        uniform and have consistant bar_width across all barplots
    Returns
    -------
    jax : :obj:`~matplotlib.axes.Axes`
    """

    if truncate_at is not None:
        freqs = _np.array(freqs)[_np.array(freqs)>truncate_at]
    xvec = _np.arange(len(freqs))
    if jax is None:
        _plt.figure(figsize=(_np.max((7,bar_width_in_inches*len(freqs))),5))
        jax = _plt.gca()

    patches = jax.bar(xvec, freqs,
                      width=.25,
                      color=color
                      )
    jax.set_yticks([.25, .50, .75, 1])
    jax.set_ylim([0, 1])
    jax.set_xticks([])
    [jax.axhline(ii, color="lightgray", linestyle="--", zorder=-1) for ii in [.25, .50, .75]]
    return jax
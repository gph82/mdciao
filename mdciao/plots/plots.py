##############################################################################
#    This file is part of mdciao.
#    
#    Copyright 2024 Charité Universitätsmedizin Berlin and the Authors
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
    pyplot as _plt, \
    cm as _cm

from mdciao.utils.str_and_dict import _kwargs_subs

from matplotlib.colors import is_color_like as _is_color_like

from matplotlib.mlab import GaussianKDE as _GKDE

from mpl_toolkits.axes_grid1 import \
    make_axes_locatable as _make_axes_locatable

import mdciao.utils as _mdcu

from mdciao.nomenclature.nomenclature import _lexsort_consensus_ctc_labels
# The above line introduces a dependency of 'plots' on 'nomenclature', which were
# uncoupled so far. The alternative would be to put '_lexsort_consensus_ctc_labels'
# into 'utils.str_and_dict' (since it's essentially string operations).
# However, as plotting methods become increasing nomenclature-aware, such a
# plots -> nomenclature dependency will likely come in the future

from os import path as _path

from collections import defaultdict as _defdict

import mdtraj as _md

import matplotlib.transforms as _transforms

from pandas import DataFrame as _DataFrame

_schemes_for_sorting = frozenset(["mean", "std", "numeric", "residue", "keep", "consensus"])

def plot_w_smoothing_auto(y, ax=None, label=None, color=None, x=None, background=True, n_smooth_hw=0, ls="-"):
    r"""
    A wrapper around :obj:`matplotlib.pyplot.plot` that allows
    to add a smoothing window (or not). See
    :obj:`window_average_fast` for more details

    Parameters
    ----------
    y : iterable of floats
        The y array
    ax : :obj:`~matplotlib.axes.Axes` or None, default is None
        The axis where to draw onto. If None,
        the current axis will be used invoking
        :obj:`~matplotlib.pyplot.gca`. If there's no
        current axis, one will be created.
    label : str or None, default is None
        Label for the legend. If not None,
        it's understood the user wants a legend
        and a call to :obj:`~matplotlib.pyplot.legend`
        will be issued inside this method.
    color : None, or color-like value, default is None
        The default is to try to use the next color
        of the color-cyle of the axis. If no color cycle
        has been initialized, this is "tab:blue". Note
        that things may have been drawn already onto
        the axis w/o a color-cycle initialized, in that
        case None can't guarantee that colors will be
        repeat. For full control of this cases, pass
        a color value here.
    x : iterable of floats, default is None
        If not provided, will default to
        x = _np.arange(len(y))
    background : bool, or color-like, (str, hex, rgb), default is True
        When smoothing, the original curve can
        appear in the background in different colors
        * True:  use a fainted version of :obj:`color`
        * False: don't plot any background
        * color-like: use this color for the background,
          can be: str, hex, rgba, anything
          :obj:`matplotlib.pyplot.colors` understands
    n_smooth_hw : int, default is 0
        Half-size of the smoothing window.
        If 0, this method is identical to
        :obj:`matplotlib.pyplot.plot`
    ls : str, default is "-"
        The linestyle of the line, one of
        [-', '--', '-.', ':', ''], more info
        here for :obj:`matplotlib.lines.line2D`

    Returns
    -------
    Line2D : :obj:`matplotlib.pyplot.Line2D`
        The 2D smoothed line

    """

    alpha = 1
    if x is None:
        x = _np.arange(len(y))
    if ax is None:
        ax = _plt.gca()
    if label is None:
        call_legend=False
    else:
        call_legend=True

    if n_smooth_hw > 0:
        alpha = .2
        x_smooth = _mdcu.lists.window_average_fast(_np.array(x), half_window_size=n_smooth_hw)
        y_smooth = _mdcu.lists.window_average_fast(_np.array(y), half_window_size=n_smooth_hw)
        line2D = ax.plot(x_smooth,
                         y_smooth,
                         label=label,
                         color=color,
                         ls=ls)[0]
        label = None
        color = line2D.get_color()
        if background:
            if isinstance(background,bool):
                pass
            else:
                assert _is_color_like(background), "The argument 'background' has to be boolean (True/False) or color-like, but '%s' (%s) is neither"%(background, type(background))
                color = background

            _line2D = ax.plot(x, y,
                              label=label,
                              alpha=alpha,
                              color=color)[0]
    else:
        line2D = ax.plot(x, y,
                          label=label,
                          alpha=alpha,
                          color=color)[0]
    if call_legend:
        ax.legend()
    return line2D


def histogram_w_smoothing_auto(data, bins=10, ax=None,
                               smooth_bw=True, background=True, fill_below=True,
                               color=None, label=None,
                               alpha_below=.25, maxcount=False) -> _plt.Axes:
    r"""
    Plot a histogram of `data` with possibilities of smoothing and filling the area below

    Parameters
    ----------
    data : list or 1D _np.ndarray
        The data to be histogramed and plotted
    bins : int, default is 10
        Since this will be passed directly
        to :obj:`numpy.histogram` it can
        also take the same values that
        :obj:`numpy.histogram_bin_edges`
        can take.
    ax : :obj:`~matplotlib.axes.Axes` or None, default is None
        The axis to draw onto. If None,
        the current axis will be used invoking
        :obj:`~matplotlib.pyplot.gca`. If there's no
        current axis, one will be created.
    smooth_bw : bool or float, default is True
        If True, smooth the histogram using a
        Gaussian-kernel-density estimation with
        an estimator bandwidth of .5 (Angstrom).
        If float, use this value as estimator
        bandwidth, check :obj:`matplotlib.mlab.GaussianKDE`
        for more info.
    background : bool, or color-like, (str, hex, rgb), default is True
        When smoothing, the original curve can
        appear in the background in different colors

        * True:  use a fainted version of `color`
        * False: don't plot any background
        * color-like: use this color for the background,
          can be: str, hex, rgba, anything
          :obj:`matplotlib.pyplot.colors` understands
    fill_below : bool, default is True
        Fill the area underneath the histogram
        with a shade of `color`
    color : None or color-like, default is None
        Default behaviour is to take the next
        color of the color-cycle of the plot.
    label : str or None, default is None
        The label for the data, which will
        be shown in the legend
    alpha_below : float, default is .25
        The are below the curve will
        be filled with this alpha (transparency)
        value. Only has an effect if `fill_below`
        is True
    maxcount : bool or positive float, default is False
        Normalize when plotting the histogram,
        s.t. different datasets can be plotted
        together at the same height even with
        very different number of absolute
        counts. If True, counts will be normalized
        to the maximum number of counts, s.t.
        histograms will peak at 1. If any other
        positive value, that's where the peak
        will be.

    Returns
    -------
    ax : :obj:`~matplotlib.axes.Axes`
    """

    if ax is None:
        ax = _plt.gca()

    h, bin_edges = _np.histogram(data, bins=bins)
    if maxcount:
        h = h/h.max() * maxcount # multiply by boolean if True means multiply by one, if scalar by the value
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    if smooth_bw:
        if isinstance(smooth_bw, bool):
            smooth_bw = .5
        model = _GKDE(data, bw_method=smooth_bw)
        xs = _np.linspace(_np.min(data), _np.max(data), num=500)
        ys = model.evaluate(xs)
        ys /= ys.max()
        ys *= h.max()
        if maxcount:
            y = ys/ys.max() * maxcount
        line = ax.plot(xs, ys, label=label, color=color)[0]
        if fill_below:
            ax.fill_between(xs, ys, alpha=.1, color=line.get_color())
        if background:
            if isinstance(background, bool):
                ax.plot(x, h, alpha=.25, color=line.get_color(), zorder=-10)
            else:
                ax.plot(x, h, alpha=.25, color=background, zorder=-10)
    else:
        line = ax.plot(x, h, label=label, color=color)[0]
        if fill_below:
            ax.fill_between(x, h, alpha=alpha_below, color=line.get_color())
    if label is not None:
        ax.legend()

    return ax

def _pop_keys_by_scheme(sort_by, freqs_by_sys_by_ctc, mean_std_by_ctc,
                        lower_cutoff_val, identity_cutoff, remove_identities):
    r"""

    Helper method to drop some second-level keys off a dictionary of dictionaries

    The drop is done following different criteria for different sorting schemes.

    The first-level keys are typically system-keys (e.g. WT, MUT) and the second
    level keys are typically contact labels ("GLU30-ALA50").

    For dropping some keys, either the per-system values for each ctc-key is looked at
    (mean_std_by_ctc) or the full, disaggregated data (freqs_by_sys_by_ctc) is looked at,
    this is why both variables are needed.

    Parameters
    ----------
    sort_by
    freqs_by_sys_by_ctc
    mean_std_by_ctc
    lower_cutoff_val
    identity_cutoff
    remove_identities

    Returns
    -------
    all_ctc_keys
    freqs_by_sys_by_ctc
    keys_popped_above
    keys_popped_below
    """
    all_ctc_keys = list(list(freqs_by_sys_by_ctc.values())[0].keys())
    drop_below = {"std": lambda ctc: mean_std_by_ctc["std"][ctc] <= lower_cutoff_val,
                  "mean": lambda ctc: all(
                      [_np.abs(idict[ctc]) <= lower_cutoff_val for idict in freqs_by_sys_by_ctc.values()]),
                  }
    drop_below["keep"]    = drop_below["mean"]
    drop_below["numeric"] = drop_below["mean"]
    drop_below["residue"] = drop_below["mean"]
    drop_below["list"]    = drop_below["mean"]
    drop_below["consensus"] = drop_below["mean"]

    drop_above = lambda ctc: all([idict[ctc] >= identity_cutoff for idict in freqs_by_sys_by_ctc.values()]) \
                             and remove_identities
    keys_popped_above, keys_popped_below = [], []
    for ctc in all_ctc_keys:
        if drop_below[sort_by](ctc):
            keys_popped_below.append(ctc)
        if drop_above(ctc):
            keys_popped_above.append(ctc)
    for ctc in _np.unique(keys_popped_above + keys_popped_below):
        [idict.pop(ctc) for idict in freqs_by_sys_by_ctc.values()]
        all_ctc_keys.remove(ctc)

    return all_ctc_keys, freqs_by_sys_by_ctc, keys_popped_above, keys_popped_below

#TODO test
def _sorting_schemes(freqs_by_sys_by_ctc, sort_by='mean',
                     lower_cutoff_val=0,
                     remove_identities=False,
                     identity_cutoff=1,
                     ):
    r"""
    Sort a dictionary of dictionaries according to different schemes.

    It can also directly drop some keys according to some thresholds.

    Currently private since it might change in the future

    Goal is to make plot_unified_freq_dicts thinner and make use of this everywhere

    Parameters
    ----------
    freqs_by_sys_by_ctc
    sort_by
    lower_cutoff_val
    remove_identities
    identity_cutoff

    Returns
    -------
    kept_keys
    freqs_by_sys_by_ctc
    keys_popped_above
    ctc_keys_popped_below
    mean_std_per_ctc

    """

    system_keys = list(freqs_by_sys_by_ctc.keys())
    all_ctc_keys = list(freqs_by_sys_by_ctc[system_keys[0]].keys())
    for sk in system_keys[1:]:
        assert len(all_ctc_keys) == len(list(freqs_by_sys_by_ctc[sk].keys())), ValueError("This is not a unified dictionary")

    # 0. Compute means and stds for everybody
    dict_for_sorting = {key: {key : None for key in all_ctc_keys} for key in list(_schemes_for_sorting) + ["list"]}
    for key in all_ctc_keys:
        dict_for_sorting["std"][key] = _np.std([idict[key] for idict in freqs_by_sys_by_ctc.values()])
        dict_for_sorting["mean"][key] = _np.mean([idict[key] for idict in freqs_by_sys_by_ctc.values()])

    # First, get rid of non-list keys in case sort_by is a list (we do it here instead of its own method)
    # TODO consider moving this inside _sorter_by_key_or_val
    if isinstance(sort_by, list):
        if not set(sort_by).intersection(all_ctc_keys):
            raise ValueError(f"The 'sort_by' list '{sort_by}' doesn't contain "
                             f"any of the available contact pairs '{all_ctc_keys}'")
        kept_keys = [key for key in sort_by if key in all_ctc_keys] #setops don't conserve order
        excluded_ctc_keys = [key for key in all_ctc_keys if key not in kept_keys] #setops don't conserve order
        sort_by = "list"
    elif sort_by in _schemes_for_sorting:

        # Then sort, in case sort_by wasn't a list but an actual scheme (has its own method)
        kept_keys = _sorter_by_key_or_val(sort_by, dict_for_sorting[sort_by])
        if sort_by in ["mean", "std"]:
            kept_keys = kept_keys[::-1]

        excluded_ctc_keys = []
    else:
        raise ValueError(f"Argument 'sort_by' has to be one of {list(_schemes_for_sorting)}, but not '{sort_by}'")

    freqs_by_sys_by_ctc = {skey : {key : sval[key] for key in kept_keys} for skey, sval in freqs_by_sys_by_ctc.items()}

    # Finally, get rid of keys too high or too low (has its own method)
    kept_keys, freqs_by_sys_by_ctc, keys_popped_above, keys_popped_below = \
        _pop_keys_by_scheme(sort_by, freqs_by_sys_by_ctc, dict_for_sorting,
                            lower_cutoff_val, identity_cutoff, remove_identities)
    keys_popped_below+=excluded_ctc_keys

    return kept_keys, freqs_by_sys_by_ctc, keys_popped_above, keys_popped_below, {key : dict_for_sorting[key] for key in ["mean", "std"]}


def plot_unified_freq_dicts(freqs,
                            colordict=None,
                            width=.2,
                            ax=None,
                            figsize=(10, 5),
                            panelheight_inches=5,
                            inch_per_contacts=1,
                            fontsize=16,
                            sort_by='mean',
                            lower_cutoff_val=0,
                            remove_identities=False,
                            vertical_plot=False,
                            identity_cutoff=1,
                            assign_w_color=False,
                            title=None,
                            legend_rows=4,
                            verbose_legend=True,
                            half_sigma=False,
                            ):
    r"""
    Plot unified (= with identical keys) frequency dictionaries for different systems

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
        The height of the panel, in inches.
        Determines the figure size
        if :obj:`figsize` is None,
        else has no effect
    inch_per_contacts : int, default is 1
        How many inches each contact-pair
        is given in the panel. Determines
        the figure size if :obj:`figsize` is None,
        else has no effect
    fontsize : int, default is 16
        Will be used in :obj:`matplotlib._rcParams["font.size"]`
        # TODO be less invasive
    sort_by : str or list of strings, default is "mean"
        If str, the property by which to sort the contacts.
        If list, the list of contact labels in the order in
        which they will be shown.
        If str, the possibilities are
         * "mean" sort (descending) by mean frequency over all systems, making most
           frequent contacts appear on the left/top of the plot.
         * "std" sort (descending) by per-contact standard deviation over all systems, making
           the contacts with most different values appear on top. This
           highlights more "deviant" contacts and might hence be
           more informative than "mean" in cases where a lot of
           contacts have similar frequencies (high or low). If this option
           is activated, a faint dotted line is incorporated into the plot
           that marks the std for each contact group
         * "keep" keep the contacts in whatever order they have in the
           first dictionary
         * "numeric" sort (ascending) the contacts by the first number
          that appears in the contact labels, e.g. "30" if
          the label is "GLU30@3.50-GDP". You can use this
          to order by resSeq if the AA to sort by is the
          first one of the pair. Contact
          labels without numbers in them will be sorted
          alphabetically at the end of the labels with numbers.
         * "residue" alias for "numeric"
        * list of contact-labels : sort in the order established
          by this list. What will actually be plotted is the
          intersection of this list and the available
          contact labels of `freqs` *after* other parameters
          like `lower_cutoff_val` or `identity_cutoff`
          have taken effect, e.g. if a contact-label is
          discarded because of `lower_cutoff_val`,
          adding the label to this list won't have any effect.
    lower_cutoff_val : float, default is 0
        Hide contacts with small values. "values" changes
        meaning depending on `sort_by`. If `sort_by` is
        any of
         * "mean", "keep", "numeric", "residue" or a list,
           then the contacts where **all**
           systems have frequencies lower than this value
           are hidden.
         * "std", then the contacts where the standard
           deviation across systems *itself* is 
           lower than this value are hidden. 
           This hides contacts where all systems are
           similar, regardless of whether they're all
           around 1, around .5 or around 0
    remove_identities : bool, default is False
        If True, the contacts where
        freq[sys][ctc] >= :obj:`identity_cutoff`
        across all systems will not be plotted
        nor considered in the sum over contacts
        TODO : the word identity might be confusing
    vertical_plot : bool, default is False
        Plot the bars vertically in descending sort_by
        instead of horizontally (better for large number of frequencies)
    identity_cutoff : float, default is 1
        If :obj:`remove_identities`, use this value to define what
        is considered an identity, s.t. contacts with values e.g. .95
        can also be removed
        TODO consider merging both identity parameters into one that is None or float
    assign_w_color : boolean, default is False
        Color the text of the contact-labels according to
        the following criterion.
         * If all frequencies are below the `lower_cutoff_val`
           except for one system, then the label adopts
           the color of this system and gets prepended with a "+" sign.
         * If all frequencies are above the `lower_cutoff_val`
           except for one system, then the label adopts
           the color of this system and gets prepended with a "-" sign
        For more details see the paragraph "Visual Aides"
        `of this notebook <https://proteinformatics.uni-leipzig.de/mdciao/notebooks/Comparing_CGs_Bars.html#Visual-Aides>`_
    title : str, default is None
        The title of the plot, if any
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
    half_sigma : bool, default is False
        When True, instead of showing
        Sigma=20, Sigma = 2x10 will
        be shown. If a ContactGroup
        has a Sigma=10 normally, when showing
        per-residue values, that number
        doubles, because each contact is
        shown two times. Hence, showing
        half-sigma allows to "keep" the
        number 10 in the legend,
        even though the shown Sigma is 20

    Returns
    -------
    fig : :obj:`~matplotlib.figure.Figure`
    ax : :obj:`~matplotlib.axes.Axes`
    freqs : dict
        Dictionary of dictionaries with the plotted frequencies
        in the plotted order. It's keyed with
        system-names first and contact-names second, like
        the input. It has the `sort_by` strategy
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

    sorted_ctc_keys, freqs_by_sys_by_ctc, keys_popped_above, ctc_keys_popped_below, mean_std_by_ctc = \
        _sorting_schemes(freqs_by_sys_by_ctc, sort_by=sort_by,
                         lower_cutoff_val=lower_cutoff_val,
                         remove_identities=remove_identities,
                         identity_cutoff=identity_cutoff)
    _df = _DataFrame(freqs_by_sys_by_ctc)
    within_01 = _df.values.min()>=0 and _df.values.max() <= 1

    # Prepare the dict
    if colordict is None:
        colordict = {key:val for key,val in zip(system_keys, _colorstring.split(","))}
    winners = _color_by_values(sorted_ctc_keys, freqs_by_sys_by_ctc, colordict,
                               lower_cutoff_val=lower_cutoff_val, assign_w_color=assign_w_color)

    # Prepare the positions of the bars
    delta, width = _offset_dict(system_keys, width=width)
    if ax is None:
        if figsize is None:
            y_figsize=panelheight_inches
            x_figsize=inch_per_contacts*len(sorted_ctc_keys)
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
    hs=1
    two_times= ""
    if half_sigma:
        hs=.5
        two_times="2 x "

    for jj, (skey, sfreq) in enumerate(freqs_by_sys_by_ctc.items()):

        bar_array = [sfreq[key] for key in sorted_ctc_keys]
        x_array = _np.arange(len(bar_array))

        # Label
        label = '%s (Sigma= %s%2.1f)'%(skey, two_times, _np.sum(list(sfreq.values()))*hs)
        if verbose_legend:
            if len(keys_popped_above)>0:
                extra = "above threshold"
                f = identity_cutoff
                label = label[:-1]+", %s+%2.1fa)"%\
                        (two_times,(_np.sum([freqs[skey][nskey] for nskey in keys_popped_above]))*hs)
            if len(ctc_keys_popped_below) > 0:
                not_shown_sigma = _np.sum([freqs[skey][nskey] for nskey in ctc_keys_popped_below])
                if not_shown_sigma>0:
                    extra = "below threshold"
                    f = lower_cutoff_val
                    label = label[:-1] + ", %s+%2.1fb)" % (two_times, not_shown_sigma*hs)
        label = _mdcu.str_and_dict.replace4latex(label)

        if len(bar_array)>0:
            if not vertical_plot:
                _plt.bar(x_array + delta[skey], bar_array,
                         width=width,
                         color=colordict[skey],
                         label=label,
                         align="center",
                         )
            else:
                _plt.barh(x_array + delta[skey], bar_array,
                          height=width,
                          color=colordict[skey],
                          label=label,
                          )
            _plt.legend(ncol=_np.ceil(len(system_keys) / legend_rows).astype(int))

    if vertical_plot:
        for ii, key in enumerate(sorted_ctc_keys):
            # 1) centered in the middle of the bar, since plt.bar(align="center")
            # 2) displaced by one half width*nbars
            iix = ii
            txt = _mdcu.str_and_dict.latex_superscript_fragments(key)
            _plt.text(0 - .05, iix, txt,
                      ha="right",
                      va="center_baseline",
                      #rotation=45,
                      )
        _plt.yticks([])
        _plt.ylim(-.5, ii +.5)
        if within_01:
            _plt.xlim((0,1))
        ax.grid(axis="x", ls=":", color="k", zorder=-10)
        _add_grey_banded_bg(ax, len(sorted_ctc_keys), horizontal=True)
        ax.set_axisbelow(True)
        _plt.gca().invert_yaxis()

        if sort_by == "std":
            _plt.plot([mean_std_by_ctc["std"][key] for key in sorted_ctc_keys],
                      _np.arange(len(sorted_ctc_keys)), color='k', alpha=.25, ls=':', zorder=10)
        ax.axvline(0,color="k")
    else:
        trans = _transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for ii, key in enumerate(sorted_ctc_keys):
            # 1) centered (ha="left") in the middle of the bar, since plt.bar(align="center")
            # 2) slight correction of half-a-fontsize to the left
            xt = ii - _rcParams["font.size"] / _points2dataunits(ax)[0] / 2
            yt =  1.05
            txt = _mdcu.str_and_dict.latex_superscript_fragments(key)
            txt = winners[key][0] + txt
            _plt.text(xt, yt,
                      txt,
                      ha='left',
                      va="bottom",
                      rotation=45,
                      color=winners[key][1],
                      transform=trans,
                      )
            #_plt.gca().axvline(iix) (visual aid)
        _plt.xticks(_np.arange(len(sorted_ctc_keys)),[])

        _plt.xlim(-.5, ii +.5)
        if within_01:
            _plt.ylim(0, 1)
        _ax = ax.twiny()
        _ax.set_xlim(ax.get_xlim())
        _plt.xticks(ax.get_xticks(), [])
        _plt.sca(ax)
        ax.grid(axis="y", ls=":", color="k", zorder=-10)
        ax.set_axisbelow(True)
        if sort_by == "std":
            _plt.plot([mean_std_by_ctc["std"][key] for key in sorted_ctc_keys],
                      color='k', alpha=.25, ls=':', zorder=10)

        if title is not None:
            ax.set_title(_mdcu.str_and_dict.replace4latex(title),
                         pad=_titlepadding_in_points_no_clashes_w_texts(ax)
                         )
        _add_grey_banded_bg(ax, len(sorted_ctc_keys))
        ax.axhline(0, color="k")
    # Create a by-state dictionary explaining the plot
    out_dict = {key:{ss: val[ss] for ss in sorted_ctc_keys} for key, val in freqs_by_sys_by_ctc.items()}
    for key2 in ["mean", "std"]:
        out_dict[key2] = {key : _np.round(val,2) for key, val in mean_std_by_ctc[key2].items()}

    _rcParams["font.size"] = _fontsize
    return myfig, _plt.gca(),  out_dict

@_kwargs_subs(plot_unified_freq_dicts)
def compare_groups_of_contacts(groups,
                               colors=None,
                               mutations_dict=None,
                               width=.2,
                               ax=None,
                               figsize=(10, 5),
                               fontsize=16,
                               anchor=None,
                               plot_singles=False,
                               ctc_cutoff_Ang=None,
                               AA_format='short',
                               defrag='@',
                               per_residue=False,
                               title='comparison',
                               distro=False,
                               interface=False,
                               n_cols=1,
                               sharex = False,
                               **kwargs_plot_unified_freq_dicts,
                               ):
    r"""
    Compare contact groups across different systems using different plots and strategies

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
          * ascii-files with the contact the frequencies in the first
            column and labels in the second and/or third column,
            see :obj:`~mdciao.contacts.ContactGroup.frequency_str_ASCII_file`
            and :obj:`~mdciao.utils.str_and_dict.freq_ascii2dict`
          * .xlsx files with the header in the second row,
            containing at least the column-names "label" and "freqs"

        Note
        ----
        If a :obj:`ContactGroup` is passed, then a :obj:`ctc_cutoff_Ang`
        needs to be passed along, otherwise frequencies cannot be computed
        on-the-fly
    colors : iterable (list or dict), or str, default is None
        * If list, the colors will be assigned in the same
          order of :obj:`groups`.
        * If dict, has to have the
          same keys as :obj:`groups`.
        * If str, it has to be a case-sensitve colormap-name of matplotlib:
          https://matplotlib.org/stable/tutorials/colors/colormaps.html
        * If None, the 'tab10' colormap (tableau) is chosen
    mutations_dict : dictionary, default is None
        A mutation dictionary that allows to plot together
        residues that would otherwise be identified as different
        contacts. If there were two mutations, e.g A30K and D35A
        the mutation dictionary will be {"A30":"K30", "D35":"A35"}.
        You can also use this parameter for correcting indexing
        offsets, e.g {"GDP395":"GDP", "GDP396":"GDP"}.
    width : float, default is .2
        The witdth of the bars
    ax : :obj:`~matplotlib.axes.Axes` or array thereof, default is None
        The default is to let the method draw its own figure
        and axis, but you can pass pre-exisintg axis here.
        If `distro` is False, it means only one axis is
        needed, so you can pass the axis object direclty here.
        If `distro` is True, a subplot is needed, where
        each panel contains the distributions of each contact.
        Hence, pass an array of axis if `distro` is True.  See
        :obj:`mdciao.plots.plot_unified_distro_dicts` for more
        info (in particular `ax_array`).
    figsize : tuple, default is (10,5)
        The figure size in inches, in case it is
        instantiated automatically by not passing an :obj:`ax`
    fontsize : float, default is 16
        The fontsize to use
    anchor : str, default is None
        This string will be deleted from the contact labels,
        leaving only the partner-residue to identify the contact.
        The deletion takes place after the :obj:`mutations_dict`
        has been applied. The final anchor label will be that
        of the deleted keys (allows for keeping e.g. pre-existing
        consensus nomenclature).
        No consistency-checks are carried out, i.e. use
        at your own risk
    plot_singles : bool, default is False
        Produce one extra figure with as many subplots as systems
        in :obj:`dictionary_of_groups`, where each system is
        plotted separately. The labels used will have been already
        "mutated" using :obj:`mutations_dict` and "anchored" using
        :obj:`anchor`. This plot is temporary and cannot be saved
        Needed value to compute frequencies on-the-fly
        if the input was using :obj:`ContactGroup` objects
    AA_format : str, default is "short"
        see :obj:`~mdciao.contacts.ContactPair.frequency_dict` for more info
    defrag : str, default is "@"
        see :obj:`~mdciao.utils.str_and_dict.unify_freq_dicts` for more info
    per_residue : bool, default is False
        Unify dictionaries by residue and not by pairs.
        If True, :obj:`remove_identities` is set to False
        automatically when calling :obj:`plot_unified_freq_dicts`
    title : str, default is "comparison"
        The title for the plot
    distro : bool, default is False
        Instead of plotting contact frequencies,
        plot contact distributions
    interface : bool, default is False
        Sorts the residues into
        interface fragments. Will fail
        if the passed `groups`
        don't have self.is_interface==True
        It enforces a per-residue view,
        plotting a single bar per residue
        indicating in how many contacts that
        residue participates in. See below 'sort_by'
        for how these residues get sorted
        within their respective interface fragments.
    n_cols : int, default is 1
        Only has effect if `distro` is True. The number
        of columns in the multi-panel figure with the
        per-contact distributions.
    sharex : bool, or string, default is False
        Only has effect if `distro` is True.
        Can be True or "col", for sharing the x-axis
        across columns. See :obj:`~matplotlib.pyplot.subplots`
        for more info. Only has an effect if `ax` is None.
    kwargs_plot_unified_freq_dicts : dict
        Optional arguments for
        :obj:`~mdciao.plots.plot_unified_freq_dicts`.
        Some of them will be overwritten, e.g.
        if `interface` or `per_residue` are True,
        then `remove_identities` or `sort_by`
        get set internally for consistency.
        The optional parameters of are:

    Other Parameters
    ----------------
    %(substitute_kwargs)s


    Returns
    -------
    myfig : :obj:`~matplotlib.figure.Figure`
        Figure with the comparison plot
    freqs : dictionary
        Unified frequency dictionaries,
        including mutations and anchor
    plotted_freqs : dictionary
        Like `freqs` but sorted and purged
        according to the user-defined input options,
        s.t. it represents the plotted values
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

    if interface:
        assert all([igroup.is_interface for igroup in groups.values()])

    freqs = {key: {} for key in groups.keys()}
    colors = color_dict_guesser(colors, freqs.keys())

    for key, ifile in groups.items():
        if isinstance(ifile, str):
            idict = _mdcu.str_and_dict.freq_file2dict(ifile)
        elif all([istr in str(type(ifile)) for istr in ["mdciao", "contacts", "ContactGroup"]]):
            if distro:
                idict = ifile.distribution_dicts(AA_format=AA_format,
                                                 pad_label=False,
                                                 bins="auto")
            else:
                assert ctc_cutoff_Ang is not None, "Cannot provide a ContatGroup object without a `ctc_cutoff_Ang` parameter"
                if not interface:
                    idict = ifile.frequency_dicts(ctc_cutoff_Ang=ctc_cutoff_Ang,
                                              AA_format=AA_format,
                                              pad_label=False)
                else:
                    idict = ifile.frequency_sum_per_residue_names(ctc_cutoff_Ang=ctc_cutoff_Ang,
                                                                  AA_format=AA_format,
                                                                  list_by_interface=True)

        else:
            idict = {key:val for key, val in ifile.items()}

        if mutations_dict is not None:
            if interface:
                idict = [{_mdcu.str_and_dict.replace_w_dict(key, mutations_dict): val for key, val in jdict.items()}
                         for jdict in idict]
            else:
                idict = {_mdcu.str_and_dict.replace_w_dict(key, mutations_dict):val for key, val in idict.items()}

        if anchor is not None:
            if interface:
                res = [_mdcu.str_and_dict.delete_exp_in_keys(jdict, anchor) for jdict in idict]
                idict, deleted_half_keys = res[0][0], res[0][1], res[1][0] + res[1][1]
            else:
                idict, deleted_half_keys = _mdcu.str_and_dict.delete_exp_in_keys(idict, anchor)
            if len(_np.unique(deleted_half_keys))>1:
                raise ValueError("The anchor patterns differ by key, this is strange: %s"%deleted_half_keys)
            else:
                anchor=_mdcu.str_and_dict.defrag_key(deleted_half_keys[0],defrag=defrag,sep=" ")
        freqs[key] = idict

    if distro:
        freqs  = _mdcu.str_and_dict.unify_freq_dicts(freqs, defrag=defrag, is_freq=False)
        myfig, __ = plot_unified_distro_dicts(freqs, colors=colors,
                                              ctc_cutoff_Ang=ctc_cutoff_Ang,
                                              fontsize=fontsize,
                                              ax_array=ax,
                                              n_cols=n_cols,
                                              sharex=sharex,
                                              )
        if anchor is not None:
            title+="\n%s and " % _mdcu.str_and_dict.latex_superscript_fragments(anchor)
        myfig.suptitle(title, y=1, va="bottom", fontsize=_rcParams["font.size"]*2)
        plotted_freqs = None
    else:
        if plot_singles:
            nrows = len(freqs)
            myfig, myax = _plt.subplots(nrows, 1,
                                        sharey=True,
                                        sharex=True,
                                        figsize=(figsize[0], figsize[1]*nrows))
            for iax, (key, ifreq) in zip(myax, freqs.items()):
                if interface:
                    ifreq[0].update(ifreq[1])
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

        if interface:
            freqs = [_mdcu.str_and_dict.unify_freq_dicts({key : val[ii] for key, val in freqs.items()},
                                                          per_residue=False,
                                                          defrag=defrag) for ii in [0,1]]
            by_interface_sorted_keys = [_sorting_schemes(idict, sort_by=kwargs_plot_unified_freq_dicts.get("sort_by", "mean"),
                                                         remove_identities=False, lower_cutoff_val=0)[0] for idict in freqs]
            freqs = [{key : {key2 : val[key2] for key2 in by_interface_sorted_keys[ii]} for key, val in ifreq.items()} for ii, ifreq in enumerate(freqs)]
            kwargs_plot_unified_freq_dicts["sort_by"]="keep"

            for key in freqs[0].keys():
                freqs[0][key].update(freqs[1][key])
            freqs = freqs[0]
        else:
            freqs = _mdcu.str_and_dict.unify_freq_dicts(freqs,
                                                per_residue=per_residue,
                                                defrag=defrag)
        if per_residue or interface:
            ylim = _np.max([_np.max(list(ifreqs.values())) for ifreqs in freqs.values()])
            kwargs_plot_unified_freq_dicts["remove_identities"] = False
        else:
            ylim = None

        if ctc_cutoff_Ang is not None:
            title = _mdcu.str_and_dict.replace4latex(title+'@%2.1f AA'%ctc_cutoff_Ang)
        if anchor is not None:
            title+="\n%s and " % _mdcu.str_and_dict.latex_superscript_fragments(anchor)

        myfig, iax, plotted_freqs = plot_unified_freq_dicts(freqs,
                                                            colordict=colors,
                                                            ax=ax,
                                                            width=width,
                                                            fontsize=fontsize,
                                                            figsize=figsize,
                                                            title=title,
                                                            half_sigma=per_residue,
                                                            **kwargs_plot_unified_freq_dicts)

        _plt.gcf().tight_layout()
        #_plt.show()

    return myfig, freqs, plotted_freqs

def color_dict_guesser(colors, keys):
    r"""
    Helper function to construct a color dictionary from user input

    Parameters
    ----------
    colors : None, str, list or dict
        * None: use the first :obj:`n` "tab10" colors, where
          n is determined by :obj:`keys`
        * str: name of the matplotlib colormap to interpolate to :obj:`n` colors.
          Can be qualitative like "Set2" or "tab20" or
          quantitative like "viridis" or "Reds". For more info see:
          https://matplotlib.org/stable/tutorials/colors/colormaps.html
        * list: list of colors (["r","g","b"])
          Turn this list into a dictionary keyed with :obj:`keys`
    keys : int or list
        If int, create a list of keys = _np.arange(keys).
        If list, use that list directly as keys for the
        color dictionary that will be returned

    Returns
    -------
    colors : dict
    """
    if isinstance(keys, int):
        assert keys>0, ("If integer,'keys' has to be > 0 ")
        key_list = _np.arange(keys)
    else:
        key_list = keys

    if isinstance(colors,dict):
        assert all([key in colors.keys() for key in key_list])
    elif colors is None:
        return {key: val for key, val in zip(key_list, _colorstring.split(","))}
    elif isinstance(colors,str):
        if _is_colormapstring(colors):
            return {key: color for key, color in zip(key_list,_try_colormap_string(colors,len(key_list)))}
        else:
            assert any([_is_color_like(colors), all([_is_color_like(char) for char in colors])]),\
                ValueError("'%s' is neither a matplotlib colormap, nor a matplotlib color, nor exclusively made up from valid matplotlib one-letter color-codes 'bcgkmrwy'"%colors)
            return {key:colors for key in key_list}

    if isinstance(colors, list):
        assert len(colors) >= len(key_list)
        colors = {key:val for key, val in zip(key_list, colors)}

    return colors

def _try_colormap_string(colors, N):
    r"""
    This method does two things:
    * wrap the Exception thrown by matplotlib in a more informative error
    * cycle through color values in cases where the input :obj:`N` is larger than the colormaps's .N colors

    Parameters
    ----------
    colors : string
        Name of the matplotlib colormap, check
        here for more info:
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    N : int
        Number of colors

    Returns
    -------
    colors : list
        Len is :obj:`N`, each item is a np.ndarray of len(3)
    """
    try:
        cmap = getattr(_cm, colors)
    except AttributeError as e:
        print(
            "Your input colors string '%s' is not a matplotlib colormap.\n Check https://matplotlib.org/stable/tutorials/colors/colormaps.html"%colors)
        raise e
    if cmap.N >= N:
        if cmap.N > 20:
            interp = _np.linspace(0, 1, N)
        else:
            # These are the qualititave colormaps you  don't want to interpolate
            interp = _np.arange(N)
        return cmap(interp)[:, :-1].tolist()
    else:
        return _np.vstack([cmap(_np.arange(cmap.N)) for ii in range(_np.ceil(N/cmap.N).astype(int))])[:N,:-1].tolist()


def _is_colormapstring(istr):
    try:
        getattr(_cm, istr)
        return True
    except:
        return False

"""
def add_hover_ctc_labels(ax, ctc_mat,
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

    artists = ax.scatter(*scatter_idxs_pairs.T,
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

def _add_grey_banded_bg(ax, n, horizontal=False):
    r"""
    Add a backdrop of gray bands that alternate with the white background

    Helps visually assign vertical bar-plots to contact-groups when comparing, e..g
    in violin plots

    Parameters
    ----------
    ax : obj:`~matplotlib.axes.Axes``
    n : int
        Double the number of grey bands,
        i.e. if n=10 five alternating gray bands
        will be added to the :obj:`axis`,
        visually framing 10 ContactGroups
    horizontal : bool, default is False
        If True, the bands are added horizontally
        instead of vertically.

    Returns
    -------
    None
    """
    for ii in _np.arange(n)[::2]:
        if not horizontal:
            ax.axvspan(ii - .5, ii + .5, color="lightgray", alpha=.25, zorder=-10)
        else:
            ax.axhspan(ii - .5, ii + .5, color="lightgray", alpha=.25, zorder=-10)


def _offset_dict(keys, wpad=.2, width=None):
    r"""
    Compute the positive and negative offset-values
    needed to center :obj:`n` bars of a given :obj:`width` around zero

    The return value is a dict rather than a list
    to best interface with the other plotting methods

    Parameters
    ----------
    keys : list
        list of strings
    wpad : float, default is .2
        The minimum distance between
        adjacent groups of bars
    width : float, default is None
        The desired bar width. If
        None, it gets computed automatically
        to maximize space usage leaving
        :obj:`wpad` axis space between
        groups of bars. If you choose a
        width that would extend the space occupied
        by the bars beyond the wpad, it
        gets resized to stay within the padding
        area

    Returns
    -------
    offsets : dict
        keyed with :obj:`keys`, valued with the offsets
    width : float
        Either the input width
        or the width resulting
        from the (n,wpad) combination
    """
    n = len(keys)
    maxwidth = (1 - wpad) / n
    if width is None:
        width = maxwidth
    else:
        width = _np.min([width, maxwidth])
    imax = (n - 1) * width / 2
    ls = _np.linspace(-imax, imax, n)
    delta = {key: val for key, val in zip(keys, ls)}
    return delta, width

def plot_unified_distro_dicts(distros,
                              colors=None,
                              ctc_cutoff_Ang = None,
                              ax_array=None,
                              panelheight_inches=5,
                              fontsize=16,
                              n_cols=1,
                              legend_rows=4,
                              sharex=False,
                              ):
    r"""
    Plot unified (= with identical keys) distribution dictionaries for different systems

    Parameters
    ----------
    distros : dictionary of dictionaries
        The first-level dict is keyed by system names,
        e.g distros.keys() = ["WT","D10A","D10R"].
            The second-level dict is keyed by contact names
    colors : iterable (list or dict), or str, default is None
        * If list, the colors will be assigned in the same
          order of :obj:`groups`.
        * If dict, has to have the
          same keys as :obj:`groups`.
        * If str, it has to be a case-sensitve colormap-name of matplotlib:
          https://matplotlib.org/stable/tutorials/colors/colormaps.html
        * If None, the 'tab10' colormap (tableau) is chosen
        be hard coded in a lot places
    ctc_cutoff_Ang : float, default is None
        The cutoff to use
    ax_array : np.ndarray containing :obj:`~matplotlib.axes.Axes`, default is None
        By default, the method creates its
        own figure and subplots,  but you can
        pass a pre-existing array of subplots here,
        and the distributions will be plotted onto them.
        Be aware that no checks are done to see if `ax_array`
        has enough panels to accommodate all contacts, or the
        size of the figure is good for readability etc.
    panelheight_inches : int, default is 5
        The height of each panel. Currently
        the only control on figure size, which
        is instantiated as
        >>> figsize=(n_cols * panelheight_inches * 2, n_rows * panelheight_inches)
        Only has an effect if ax_array is None.
    fontsize : int, default is 16
        Will be used in :obj:`matplotlib._rcParams["font.size"]
        # TODO be less invasive
    n_cols : int, default is 1
        Number of columns of the plot, the rows
        will be computed automatically to accommodate enough
        panels for the number of contacts in the `distros`. Only
        has an effect if ax_array is None.
    legend_rows : int, default is 4
        The maximum number of rows per column of the legend.
        If you have 10 systems, :obj:`legend_rows`=5 means
        you'll get two columns, =2 means you'll get five.
    sharex : bool, default is False
        Whether the panels (subplots) will share their
        x-axis. Can be True or "col", for sharing
        across columns. See :obj:`~matplotlib.pyplot.subplots`
        for more info. Only has an effect if ax_array is None.

    Returns
    -------
    fig, axes : :obj:`~matplotlib.figure.Figure` and the axes array
    """
    _fontsize=_rcParams["font.size"]
    _rcParams["font.size"] = fontsize

    #make copies of dicts
    distros_by_sys_by_ctc = {key:{key2:val2 for key2, val2 in val.items()} for key, val in distros.items()}

    system_keys = list(distros_by_sys_by_ctc.keys())
    all_ctc_keys = list(distros_by_sys_by_ctc[system_keys[0]].keys())
    for sk in system_keys[1:]:
        assert len(all_ctc_keys)==len(list(distros_by_sys_by_ctc[sk].keys())), "This is not a unified dictionary"

    distros_by_ctc_by_sys = _defdict(dict)
    for sys_key, sys_dict in distros_by_sys_by_ctc.items():
        for ctc_key, val in sys_dict.items():
            distros_by_ctc_by_sys[ctc_key][sys_key]=val
    distros_by_ctc_by_sys = dict(distros_by_ctc_by_sys)

    # Prepare the dict
    colors = color_dict_guesser(colors, system_keys)

    n_cols = _np.min((n_cols, len(all_ctc_keys)))
    n_rows = _np.ceil(len(all_ctc_keys) / n_cols).astype(int)

    if ax_array is None:
        myfig, myax = _plt.subplots(n_rows, n_cols,
                                    sharey=True,
                                    sharex=sharex,
                                    figsize=(n_cols * panelheight_inches * 2, n_rows * panelheight_inches), squeeze=False)
    else:
        myax = ax_array
        myfig = myax.flatten()[0].figure


    for iax, (ctc_key, per_sys_distros) in zip(myax.flatten(),distros_by_ctc_by_sys.items()):
        iax.set_title(_mdcu.str_and_dict.replace4latex(ctc_key),
                     #pad=_titlepadding_in_points_no_clashes_w_texts(ax)
                     )

        for sys_key, data in per_sys_distros.items():
            label = sys_key
            if not data in [0,None]:
                h, bin_edges = data
                x = (bin_edges[:-1]+bin_edges[1:])/2
                if ctc_cutoff_Ang is not None:
                    freq = (h[_np.flatnonzero(x<=ctc_cutoff_Ang/10)]).sum()/h.sum()
                    label += " (%u%%)" % (freq * 100)
                h = h/h.max()
                iax.plot(x * 10, h, label=label, color=colors[sys_key])
                iax.fill_between(x * 10, h, alpha=.15, color=colors[sys_key])


        if ctc_cutoff_Ang is not None:
            iax.axvline(ctc_cutoff_Ang, zorder=10,alpha=.5, color="k",ls=":")
        iax.set_xlabel("D / $\AA$")
        iax.legend(ncol=_np.ceil(len(system_keys) / legend_rows).astype(int))
        iax.set_ylim([0,1.1])
    myfig.tight_layout()

    _rcParams["font.size"] = _fontsize
    return myfig, myax

def compare_violins(groups,
                    colors=None,
                    ctc_cutoff_Ang=None,
                    fontsize=16,
                    mutations_dict={},
                    legend_rows=4,
                    AA_format='short',
                    defrag='@',
                    anchor=None,
                    ymax=None,
                    key_separator="-",
                    sort_by="mean",
                    figsize=None,
                    panelheight_inches=5,
                    inch_per_contacts=1,
                    zero_freq = 1e-2,
                    remove_identities=False,
                    identity_cutoff=1,
                    representatives=None,
                    ):
    r"""
    Plot all distance-distributions of several :obj:`~mdciao.contacts.ContactGroup` s together using :obj:`~matplotlib.pyplot.violinplot` s

    Contacts across different :obj:`groups` are grouped together by
    matching their contact labels, since the residue indices
    might differ across :obj:`groups`. To achieve this:
        * "K30-D40" is considered equivalent to "D40-D30",
          use `key_separator` to change this.
        * "K30-D40" is considered equivalent to "K30-E40"
          if a :obj:`mutations_dict={"E40":"D40"}` is passed
        * "K30@3.50-D40" is considered equivalent to "K30-D40"
          if you defragment your labels using :obj:`defrag="@"`

    Parameters
    ----------
    groups : dictionary or list of :obj:`~mdciao.contacts.ContactGroup`-objects
        The keys are the system/setup descriptors, e.g. "WT", "MUT" etc.
        If list, keys will be generated on the fly "mdcCG 0, mdcCG 1..."
    colors : iterable (list or dict), or str, default is None
        * If list, the colors will be assigned in the same
          order of :obj:`groups`.
        * If dict, has to have the
          same keys as :obj:`groups`.
        * If str, it has to be a case-sensitive colormap-name of matplotlib:
          https://matplotlib.org/stable/tutorials/colors/colormaps.html
        * If None, the 'tab10' colormap (tableau) is chosen
        TODO: I could set the default to "tab10", but then it'd
        be hard coded in a lot places
    ctc_cutoff_Ang : float, default is None
        If provided, draw a horizontal line across the panel
        at this distance value.
    fontsize : int, default is 16
        Will be used in :obj:`~matplotlib.rcParams` ["font.size"]
    panelheight_inches : int, default is 5
        The height of the panel, in inches.
        Determines the figure size
        if :obj:`figsize` is None,
        else has no effect
    inch_per_contacts : int, default is 1
        How many inches each contact-pair
        is given in the panel. Determines
        the figure size if :obj:`figsize` is None,
        else has no effect
    figsize : None or iterable of len 2, default is None
        Figure size (x,y), in inches. If None,
        one will be created using :obj:`panelheight_inches`
        and :obj:`inch_per_contacts`.
        If you are transposing the figure
        using :obj:`vertical_plot`, you do not
        have to invert (y,x) this parameter here, it is
        done automatically.
    mutations_dict : dictionary, default is {}
        A mutation dictionary that contains allows to plot together
        residues that would otherwise be identified as different
        contacts. If there were two mutations, e.g A30K and D35A
        the mutation dictionary will be {"A30":"K30", "D35":"A35"}.
        You can also use this parameter for correcting indexing
        offsets, e.g {"GDP395":"GDP", "GDP396":"GDP"}
    legend_rows : int, default is 4
        The maximum number of rows per column of the legend.
        If you have 10 systems, :obj:`legend_rows`=5 means
        you'll get two columns, =2 means you'll get five.
    AA_format : str, default is "short"
        see :obj:`~mdciao.contacts.ContactPair.frequency_dict` for more info
    defrag : str, default is "@"
        see :obj:`~mdciao.utils.str_and_dict.unify_freq_dicts` for more info
    anchor : str, default is None
        When str, e.g. "L394", that residue is eliminated
        from the contact-labels. It is also checked
        that all :obj:`~mdciao.contacts.ContactGroup`-objects
        are indeed neighborhoods sharing this anchor, i.e.,
        *some* sanity checks are carried out
    ymax : float, default is None
        Maximum value of the y-axis,
        default is to set it automatically
    key_separator : str, default is "-"
        How each contact label separates
        the pair of residues, "ALA50-GLU30".
        If you set this to None, it means
        the label won't be separated before
        matching and "ALA50-GLU30" will be
        different from "GLU30-ALA50".
    sort_by : str or list, default is 'mean'
        By default, the violins are sorted
        by ascending order of mean distance, i.e.
        from most "formed" on the left of the plot
        to least "formed" on the right of the plot.
        However, for each residue pair, this mean is
        an average over the distance in all
        the different `groups`, so some
        heterogeneity is expected. Alternatively,
        you can sort using the contact labels,
        regardless of the distance values. Note
        that for this, string comparisons between
        contact-labels will take place. and that
        contact-labels are altered by `key_separator`
        to unify across different `groups`
        Try setting `key_separator` to None
        if you see unexpected behavior, although
        though this might have other side effects,
        (see :obj:`~mdciao.utils.str_and_dict.unify_freq_dicts`)
        `sort_by` can be a:
            * str : 'residue' or 'numeric'
               Sort by ascending residue sequence index (resSeq),
               which will be inferred from each contact label,
               e.g. 30 for "GLU30@3.50". See :obj:`~mdciao.contacts.ContactGroup.gen_ctc_labels`
               for more info on how they are generated.
               Internally, the order is generated via
               :obj:`~mdciao.utils.str_and_dict.lexsort_ctc_labels`.
               If you want to reverse or alter this
               ascending default order, we recommend using
               :obj:`~mdciao.utils.str_and_dict.lexsort_ctc_labels`
               **before** calling :obj:`compare_violins` and use
               its output (`labels`) as a list
               argument for `sort_by`. Also note that
               residue indices as contained in
               :obj:`~mdciao.contacts.ContactGroup.res_idx_pairs`
            * str : 'keep'
               Sort using the same order of the labels as in
               the first contact group
            * str : 'consensus'
               Sort following consensus nomenclature (GPCR, CGN or KLIFS)
            * list : a list of contact labels,
               eg. ["GLU30-ALA30", "ARG131@3.50-TYR20"].
               Only these residue pairs (in this order)
               will be shown, regardless of what other
               pairs are contained in the `groups`. It
               assumes the user knows what contacts
               are present and can come up with a meaningful
               list. Not all labels need to be in all
               `groups` nor do all `groups`
               have to contain all labels, but at least
               one label needs to match, otherwise the
               method will fail
    zero_freq : float, default is 1e-2
        Frequencies below this number will
        be considered zero and not shown it they are
        zero for the same residue pair across all :obj:`groups`
        For this parameter to have effect, you need a
        :obj:`ctc_cutoff_Ang`
    remove_identities : bool, default is False
        If True, the contacts where
        freq[sys][ctc] >= :obj:`identity_cutoff`
        across all systems will not be plotted
        nor considered in the sum over contacts.
        Only has an effect if `ctc_cutoff_Ang` is not None.
    identity_cutoff : float, default is 1
        If :obj:`remove_identities`, use this value to define what
        is considered an identity, s.t. contacts with values e.g. .95
        can also be removed.
        Only has an effect if `ctc_cutoff_Ang` is not None.
    representatives : anything (bool, int, dict, list) default is None
        Include information about representative values in the
        plot. This can be done in several ways. Easiest
        is to let this method call :obj:`mdciao.contacts.ContactGroup.repframes`
        internally. This will locate representative frames, extract
        their residue-residue distance values and plot them as small dots
        on top of the violins. When possible, also the geometries corresponding
        to these frames will be returned. Alternatively, the user
        can directly input a dictionary of  :obj:`~mdtraj.Trajectory` objects
        (representative or not) for which the residue-residue distance values
        will be computed and plotted. Check the docs of
        :obj:`mdciao.contacts.ContactGroup.repframes` to find out
        what is meant with "representative".
        This is what each type of input does:

        * boolean True:
           Calls :obj:`mdciao.ContactGroup.repframes` with the
           method's default parameters.
        * int > 0:
           Calls :obj:`mdciao.ContactGroup.repframes` with the
           parameter `n_frames` set to this integer. This parameter
           controls how many representatives are extracted and
           subsequently plotted.
        * dict of parameters:
           A dictionary with explict values for the optional
           parameters of :obj:`mdciao.contacts.ContactGroup.repframes`,
           usually `n_frames` (an int) and `scheme`, ("mean" or "mode"),
           depending what you mean with "representative". Check the method's
           documentation for more info.
        * dict of :obj:`~mdtraj.Trajectory` objects:
           Has to have the same keys as `groups`. No checks are done
           whether these objects match the actual molecular topologies
           of `groups`, so beware of potential mismatches here.
           Typically, these frames come from having used
           :obj:`mdciao.contacts.ContactGroup.repframes` with
           `return_traj`=True.
        * dict of dicts containing values
          #TODO not implemented yet

    Returns
    -------
    fig : :obj:`~matplotlib.figure.Figure`
    ax :  :obj:`~matplotlib.axes.Axes`
    labels : list
        The list of plotted labels,
        in the order they are plotted
    repframes : dict
        Will only be returned if
        `representatives` was not None.
        The representative frames for
        each `group` according to the
        parameters of `representatives`
    """
    _fontsize=_rcParams["font.size"]
    _rcParams["font.size"] = fontsize

    # Gather data
    data4violins_per_sys_per_ctc = {}
    freqs_per_sys_per_ctc = {}
    if isinstance(groups,list):
        _groups = {"mdcCG %u" % ii : item for ii, item in enumerate(groups)}
    else:
        _groups = groups
    repframes_per_sys_per_ctc = {}
    reptraj_per_sys_per_ctc = {}
    for syskey, group in _groups.items():
        labels = group.gen_ctc_labels(AA_format=AA_format,
                                      fragments=[True if defrag is None else False][0],
                                      )
        idict = {key : _np.hstack(cp.time_traces.ctc_trajs) * 10
                 for key, cp in zip(labels, group._contacts)}
        data4violins_per_sys_per_ctc[syskey] = idict
        if ctc_cutoff_Ang is not None:
            freqs_per_sys_per_ctc[syskey] = {key:freq for key, freq in zip(labels, group.frequency_per_contact(ctc_cutoff_Ang))}

        if bool(representatives):
            #Tune the kwargs on a per-case basis then call repframes only once,
            # wrapped in the try block for when there's no files
            repframes_kwargs = {"ctc_cutoff_Ang": ctc_cutoff_Ang,
                                "return_traj": True}
            # Do we have representatives?
            if isinstance(representatives, bool):
                pass
            if isinstance(representatives, int) and representatives>0:
                repframes_kwargs.update({"n_frames" : representatives,
                                         "verbose" : False})
            if isinstance(representatives, dict) and len(representatives)>0:
                if syskey not in representatives.keys() :
                    representatives.update(repframes_kwargs)
                    representatives.pop("ctc_cutoff_ang", None)
                    representatives.pop("show_violins", None)
                    representatives["return_traj"] = True
                else:
                    assert isinstance(representatives[syskey], _md.Trajectory)
                    d = _md.compute_contacts(representatives[syskey], contacts=group.res_idxs_pairs)[0].T
                    traj = representatives[syskey]
                    repframes_kwargs = None

            if repframes_kwargs is not None:
                try:
                    __, __, d, traj = group.repframes(**repframes_kwargs)
                except FileNotFoundError as e:
                    print(e)
                    repframes_kwargs["return_traj"] = False
                    __, __, d = group.repframes(**repframes_kwargs)
                    traj = None
                d = d.T.squeeze()

            repframes_per_sys_per_ctc[syskey] = {key: val * 10 for key, val in
                                                 zip(labels, d)}
            reptraj_per_sys_per_ctc[syskey]=traj
    representatives = bool(representatives)
    # Unify data
    data4violins_per_sys_per_ctc = _mdcu.str_and_dict.unify_freq_dicts(data4violins_per_sys_per_ctc,
                                                                       replacement_dict=mutations_dict,
                                                                       is_freq=False,
                                                                       val_missing=_np.nan,
                                                                       key_separator=key_separator) #todo use kwargs?
    if representatives:
        repframes_per_sys_per_ctc = _mdcu.str_and_dict.unify_freq_dicts(repframes_per_sys_per_ctc,
                                                                        replacement_dict=mutations_dict,
                                                                        is_freq=False,
                                                                        val_missing=_np.nan,
                                                                        key_separator=key_separator)

    # Delete some keys if all freqs are < zero_freq
    if ctc_cutoff_Ang is not None:
        # First unify
        freqs_per_sys_per_ctc = _mdcu.str_and_dict.unify_freq_dicts(freqs_per_sys_per_ctc,
                                                                    replacement_dict=mutations_dict,
                                                                    is_freq=True,
                                                                    val_missing=0,
                                                                    verbose=False,
                                                                    key_separator=key_separator)  # todo use kwargs?
        #Since the dict is unified, we can do this
        _freq_dict = _defdict(list)
        for idict in freqs_per_sys_per_ctc.values():
            for key, freq in idict.items():
                _freq_dict[key].append(freq)
        pop_from_below = [key for key, ifreqs in _freq_dict.items() if all([ifreq < zero_freq for ifreq in ifreqs])]
        pop_from_above = [key for key, ifreqs in _freq_dict.items() if all([ifreq >= identity_cutoff for ifreq in ifreqs])]
        [[idict.pop(key) for idict in data4violins_per_sys_per_ctc.values()] for key in pop_from_below]
        if remove_identities:
            [[idict.pop(key) for idict in data4violins_per_sys_per_ctc.values()] for key in pop_from_above]

    # TODO avoid repetition
    if anchor is not None:
        for key, idict in data4violins_per_sys_per_ctc.items():
            data4violins_per_sys_per_ctc[key], deleted_half_keys = _mdcu.str_and_dict.delete_exp_in_keys(idict, anchor)
        if len(_np.unique(deleted_half_keys)) > 1:
            raise ValueError("The anchor patterns differ by key, this is strange: %s" % deleted_half_keys)

        if representatives is not {}:
            for key, idict in repframes_per_sys_per_ctc.items():
                repframes_per_sys_per_ctc[key], deleted_half_keys = _mdcu.str_and_dict.delete_exp_in_keys(idict,
                                                                                                          anchor)
            if len(_np.unique(deleted_half_keys)) > 1:
                raise ValueError("The anchor patterns differ by key, this is strange: %s" % deleted_half_keys)

    # Gather keys
    all_sys_keys = list(_groups.keys())
    all_ctc_keys = list(data4violins_per_sys_per_ctc[all_sys_keys[0]])
    data4violins_per_ctc_per_sys = {key:{sk:data4violins_per_sys_per_ctc[sk][key] for sk in all_sys_keys} for key in all_ctc_keys}
    means_per_ctc_across_sys = {key:_np.nanmean(_np.hstack(list(val.values()))) for key, val in data4violins_per_ctc_per_sys.items()}

    # Prepare the dict
    colordict = color_dict_guesser(colors, all_sys_keys)
    sorted_keys = _sorter_by_key_or_val(sort_by, means_per_ctc_across_sys)
    key2ii = {key : ii for ii, key in enumerate(sorted_keys)}
    delta, width = _offset_dict(list(_groups.keys()))

    if figsize is None:
        figsize=(len(key2ii)*inch_per_contacts, panelheight_inches)
    myfig = _plt.figure(figsize=figsize)
    iax = _plt.gca()
    _add_grey_banded_bg(_plt.gca(), len(all_ctc_keys))

    for syskey in all_sys_keys:
        positions = [key2ii[key]+delta[syskey] for key, val in data4violins_per_sys_per_ctc[syskey].items() if val is not _np.nan and key in key2ii.keys()]
        idata = [val for key, val in data4violins_per_sys_per_ctc[syskey].items() if val is not _np.nan and key in key2ii.keys()]
        violins = _plt.violinplot(idata, positions=positions,
                                  widths=width,
                                  showmeans=True,
                                  showextrema=False,
                                  )
        violins["cmeans"].set_color(colordict[syskey])
        for vio in violins["bodies"]:
            vio.set_color(colordict[syskey])
        _plt.plot(_np.nan, _np.nan, "d",color=colordict[syskey],
                  #alpha=vio.get_alpha()*1.5,
                  label=_mdcu.str_and_dict.replace4latex(syskey))
        if representatives:
            irep = _np.vstack([val for key, val in repframes_per_sys_per_ctc[syskey].items() if val is not _np.nan and key in key2ii.keys()])
            _plt.plot(positions, irep, "o ",
                      ms=2.5,
                      color=colordict[syskey]
                      )

    # Cosmetics
    if defrag is None:
        prepare_str = lambda istr: _mdcu.str_and_dict.latex_superscript_fragments(istr)
    else:
        prepare_str = lambda istr: istr

    if anchor is not None:
        iax.set_title("%s neighborhood"%prepare_str(anchor))

    if ctc_cutoff_Ang is not None:
        iax.axhline(ctc_cutoff_Ang, color="gray",ls="--", zorder=-10)
    _plt.xticks(_np.arange(len(key2ii)),[prepare_str(key) for key in key2ii.keys()],
                rotation=45,
                va="top",
                ha="right",
                rotation_mode="anchor"
                )
    iax.set_xlim([0-.5,len(key2ii)-.5])
    iax.set_ylabel("D / $\AA$")
    if ymax is not None:
        iax.set_ylim([iax.get_ylim()[0], ymax])

    iax.legend(ncol=_np.ceil(len(all_sys_keys) / legend_rows).astype(int),
               #loc="upper left"
               )
    myfig.tight_layout()

    _rcParams["font.size"] = _fontsize
    if repframes_per_sys_per_ctc != {}:
        return myfig, iax, list(key2ii.keys()), reptraj_per_sys_per_ctc
    else:
        return myfig, iax, list(key2ii.keys())


def _sorter_by_key_or_val(sort_by, indict):
    r"""
    Helper method to sort the keys of a dictionary according to some rules

    The rules might use just the keys or just the values of the dict

    The `indict` is left unaltered.

    Parameters
    ----------
    sort_by : str, list
        Currently, can be
        * "residue" or "numeric", i.e.
          sort by ascending value(s) of the
          integers present in the labels,
          typically resSeq of the contact labels.
          Labels lacking any integer values
          will be sorted alphabetically at the end.
        * "mean" or "std"
          sort the dict ascending,
          by the values of the `indict`
        * "keep"
          keep the order of the keys
        * "consensus"
          Sort following consensus nomenclature
        * a list of contact labels.
          The returned `ordered_keys` will be the
          intersection of `sort_by` and `indict.keys()`,
          sorted wrt to `sort_by`
    indict : dict
        The dictionary to be
        sorted according to
        some `sort_by` criterion.
        It's assumed that the keys
        are contact labels with
        "-" as the separator
    Returns
    -------
    ordered_keys : list
        The list of sorted keys
    """
    all_ctc_keys= list(indict.keys())

    # First,
    if isinstance(sort_by, list):
        if not set(sort_by).intersection(all_ctc_keys):
            raise ValueError(f"The 'sort_by' list '{sort_by}' doesn't contain "
                             f"any of the available contact pairs '{all_ctc_keys}'")
        ordered_keys = [key for key in sort_by if key in all_ctc_keys]

    elif sort_by in ["residue", "numeric"]:
        numeric_keys, non_numeric_keys = [], []
        for key in all_ctc_keys:
            try:
                _mdcu.str_and_dict.intblocks_in_str(key)
                numeric_keys.append(key)
            except ValueError as e:
                non_numeric_keys.append(key)
        ordered_keys = _mdcu.str_and_dict.lexsort_ctc_labels(numeric_keys)[0]+sorted(non_numeric_keys)
        # we could go with natsorted almost directly but:
        # In[5]: natsorted(["0-20", "0-10", "ALA30-GLU50", "ALA30-GLU40", "ALA", "GLU5-ALA20"])
        # Out[5]: ['0-10', '0-20', 'ALA', 'ALA30-GLU40', 'ALA30-GLU50', 'GLU5-ALA20']
        # -> we would want ['0-10', '0-20', 'GLU5-ALA20', 'ALA30-GLU40', 'ALA30-GLU50', 'ALA']
    elif sort_by == "consensus":
        ordered_keys = _lexsort_consensus_ctc_labels(all_ctc_keys)[0]
    elif sort_by in ["mean", "std"]:
        ordered_keys = list(_mdcu.str_and_dict.sort_dict_by_asc_values(indict).keys())
    elif sort_by == "keep":
        ordered_keys = all_ctc_keys
    return ordered_keys

def add_tilted_labels_to_patches(ax, labels,
                                 label_fontsize_factor=1,
                                 trunc_y_labels_at=.65,
                                 single_label=False):
    r"""
    Iterate through :obj:`ax.patches` and place the text strings
    in :obj:`labels` on top of it.

    Fragment names are super-scripted and LaTex-words (alpha_2)
    are taken care of automatically, so there's no need to include
    dollar-signs.

    Parameters
    ----------
    ax : :obj:`~matplotlib.axes.Axes`
        The axes onto which the tilted labels
        will be added
    labels : list
        The strings with the labels,
    label_fontsize_factor, float, default is 1
        The labels will be plotted using the
        fontsize rcParams["font.size"]*label_fontsize_factor
    trunc_y_labels_at : float, default is .65
        The tilted labels are added at
        bar-height up to this value, as to remain
        more or less inside the panel
    single_label : bool, default is False
        Tells the method whether the label
        is R30@frag1-K50@frag2 or "R30@frag2".
        Helps the method put the fragments in
        super-script
    """
    for ii, (ipatch, ilab) in enumerate(zip(ax.patches, labels)):
        ix = ii
        iy = ipatch.get_height()
        iy += .05
        if iy > trunc_y_labels_at:
            iy = trunc_y_labels_at
        if single_label:
            txt = _mdcu.str_and_dict._latex_superscript_one_fragment(ilab)
        else:
            txt = _mdcu.str_and_dict.latex_superscript_fragments(ilab)
        ax.text(ix, iy, txt,
                va='bottom',
                ha='left',
                rotation=45,
                rotation_mode="anchor",
                fontsize=_rcParams["font.size"]*label_fontsize_factor,
                backgroundcolor="white",
                bbox={"boxstyle": "square,pad=0.2",
                       "fc": "white", "ec": "none", "alpha": .9},

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

def _points2dataunits(ax):
    r"""
    Return a conversion factor for points 2 dataunits
    Parameters
    ----------
    ax : :obj:`~matplotlib.axes.Axes`

    Returns
    -------
    p2d : float
        Conversion factor so that points * p2d = points_in_dataunits

    TODO revise that this isn't indeed d2p!!!!
    """
    bbox = ax.get_window_extent()
    dx_pts, dy_pts = bbox.bounds[-2:]
    dx_in_dataunits, dy_in_dataunits = _np.diff(ax.get_xlim())[0], _np.diff(ax.get_ylim())[0]
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
        The axes where the text objects
        are drawn onto

    Returns
    -------
    y : float
    """

    rend = ax.figure.canvas.get_renderer()
    return _np.max(
        [ax.transAxes.inverted().transform(txt.get_window_extent(rend).corners()[-1])[-1]
         for txt in ax.texts]
    )

def _titlepadding_in_points_no_clashes_w_texts(ax):
    r"""
    Compute amount of upward padding need to avoid overlap
    between the axis title and any text object in the axis.

    Returns None if no text objects are there

    Parameters
    ----------
    ax : :obj:`~matplotlib.axes.Axes`

    Returns
    -------
    pad_id_points : float or None

    """
    heights = [txt.get_window_extent(ax.figure.canvas.get_renderer()).height for txt in ax.texts]
    if len(heights)>0:
        pad_in_points = _np.max(heights)+_rcParams["axes.titlepad"]*2
    else:
        pad_in_points = None

    return pad_in_points


def CG_panels(n_cols, CG_dict, ctc_cutoff_Ang,
              draw_empty=True,
              distro=False,
              short_AA_names=False,
              plot_atomtypes=False,
              switch_off_Ang=0,
              panelsize=4,
              panelsize2font=3.5):
    r"""
    One figure with each obj:`~mdciao.contacts.ContactGroup` as individual panel

    Wraps around plot_distance_distributions, plot_neighborhood_freqs

    n_cols : int
        number of columns of the subplot
    CG_dict : dict
        dictionary of
        obj:`~mdciao.contacts.ContactGroup` objects
    ctc_cutoff_Ang : float
        The cutoff to use
    draw_empty : bool, default is True
        To give a visual cue that some
        CGs are empty, the corresponding
        panels are drawn empty (it also
        ensures same panel position regardless
        of the result).
    distro : bool, default is False
        Plot distance distributions instead
        of contact frequencies
    short_AA_names : bool, default is False
        Shorten residue names from "GLU30"->"E30"
    plot_atomtypes : bool, default is False
        Inform about the types of atoms that are
        interacting
    switch_off_Ang : float, default is None
        TODO
    panelsize : float, default is 4
        In inches, the size of the panels
        where the ContactGroups will be plotted
    panelsize2font : float, default is 3.5
        The default fontsize for the figure
        is panelsize*panelsize2font. 3.5
        seems to produce good spacing among labels

    Returns
    -------
    fig : :obj:`~matplotlib.figure.Figure`

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
                ihood.plot_distance_distributions(bins=20, ax=jax, shorten_AAs=short_AA_names,
                                                  ctc_cutoff_Ang=ctc_cutoff_Ang,
                                                  label_fontsize_factor=panelsize2font / panelsize)
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
                                             plot_atomtypes=plot_atomtypes,
                                             )

    if not distro:
        non_nan_rightermost_patches = [[p for p in jax.patches if not _np.isnan(p.get_x())][-1] for jax in
                                       bar_ax.flatten() if len(jax.patches) > 0]
        xmax = _np.nanmax([p.get_x() + p.get_width() / 2 for p in non_nan_rightermost_patches]) + .5
        [iax.set_xlim([-.5, xmax]) for iax in bar_ax.flatten()]
    bar_fig.tight_layout(h_pad=2, w_pad=0, pad=0)

    return bar_fig

def plot_matrix(mat, labels, pixelsize=1,
                transpose=False, grid=False,
                cmap="binary",
                colorbar=False):
    r"""
    Plot a matrix using :obj:`~matplotlib.pyplot.imshow`.

    Matrx can be non-symmetric and rectangular, rows
    and columns need not represent the same residues
    or groups of residues

    Parameters
    ----------
    mat : 2D numpy.ndarray of shape (N,M)
        The matrix to be plotted, NaNs are allowed
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
    cmap : str, default is 'binary'
        What :obj:`matplotlib.cmap` to use
    colorbar : boolean, default is False
        whether to use a colorbar
    transpose : bool, default is False
        Transpose the matrix when
        plotting

    Returns
    -------
    ax : :obj:`~matplotlib.axes.Axes` object
    pixelsize : float, size of the pixel
        Helpful in cases where this method is called
        with the default value, in case the value
        changes in the future
    """
    _np.testing.assert_array_equal(mat.shape,[len(ll) for ll in labels]), "Number of labels don't match number of rows/cols "
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
        im.set_clim(_np.nanmin([_np.nanmin(mat), 0]), _np.nanmax([_np.nanmax(mat), 1.0]))
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
                            ax=None,
                            lower_cutoff_val=None,
                            bar_width_in_inches=.75,
                            color="tab:blue",
                            ):
    r"""
    Base method for plotting the contact frequencies

    Parameters
    ----------
    freqs : iterable
        The values to plot
    ax : :obj:`~matplotlib.axes.Axes`, default is None
        If None is passed, one will be created
    lower_cutoff_val : float, default is None
        Only plot frequencies above this value (between 0 and 1)
    bar_width_in_inches : float, default is .75
        The width of the axis will vary with the number of plotted
        frequencies. This allows for plotting different :obj:`ContactGroup`
        objects each with different number of contacts and still appear
        uniform and have consistent bar_width across all barplots
    color : str or list, default is "tab:blue"
        The color or colors for the bar

    Returns
    -------
    ax : :obj:`~matplotlib.axes.Axes`
    """

    if lower_cutoff_val is not None:
        freqs = _np.array(freqs)[_np.array(freqs) > lower_cutoff_val]
    xvec = _np.arange(len(freqs))
    if ax is None:
        _plt.figure(figsize=(_np.max((7,bar_width_in_inches*len(freqs))),5))
        ax = _plt.gca()

    patches = ax.bar(xvec, freqs,
                     width=.25,
                     color=color
                     )
    ax.set_yticks([.25, .50, .75, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([])
    [ax.axhline(ii, color="lightgray", linestyle="--", zorder=-1) for ii in [.25, .50, .75]]
    return ax

def _plot_violin_baseplot(vdata,
                          ax=None,
                          violin_width_in_inches=.75,
                          colors="tab:blue",
                          labels=None,
                          offset=0
                          ):
    r"""
    Base method for plotting the residue-residue distances as :obj:~`matplotlib.pyplot.violinplot`

    Parameters
    ----------
    vdata : list
        Each entry represents a contact-pair and
        is itself a list of time-traces (one time-trace
        per trajectory)
    ax : :obj:`~matplotlib.axes.Axes`, default is None
        If None is passed, one will be created
    violin_width_in_inches : float, default is .75
        The width of the axis will vary with the number of plotted
        frequencies. This assigns this number of inches to each
        contact in the x-axis. This allows for plotting different
        :obj:`ContactGroup` objects each with different number of
        contacts and still appear uniform
    offset : float, default is 0
        The offset for horizontal positions of the violins
    colors : list or str, default is "tab:blue"
        The colors to be used. If list,
        each violin will get one color
        from the list. If string, depending on what string:
        * if is_color_like(colors)=True
         all violins get this color
        * if string is a matplotlib colormap
         interpolate N (=len(vdata)
         colors on that colormap and assign
         one to each violin

    Returns
    -------
    ax : :obj:`~matplotlib.axes.Axes`
    violins : dictionary
        See :obj:`~matplotlib.pyplot.violinplot` for more info
    """
    xvec = _np.arange(len(vdata)) + offset
    if ax is None:
        _plt.figure(figsize=(_np.max((7, violin_width_in_inches * len(vdata))), 5))
        ax = _plt.gca()
    else:
        _plt.sca(ax)
    vdata = [_np.hstack(dt) for dt in vdata]
    #means = [_np.mean(dt) for dt in per_CP_timetraces]
    vdata = _np.array(vdata, ndmin=2).T
    assert vdata.shape[0]==len(vdata)
    violins = _plt.violinplot(vdata,
                              positions=xvec,
                              showmeans=True,
                              #showmedians=True,
                              #widths=violin_width_in_inches,
                              widths=.25,
                              showextrema=False)
    #_plt.plot(xvec, means, " o", colors=colors)
    if labels is None:
        ax.set_xticks([])
    else:
        _plt.xticks(xvec, labels, rotation=45, ha="right", va="top",
                    rotation_mode="anchor")
    if isinstance(colors, list):
        assert len(colors)>=len(violins["bodies"]),"Not enough colors (%u) for the number of violins (%u)"%(len(colors),len(vdata))
    elif _is_colormapstring(colors):
        colors = _try_colormap_string(colors, len(vdata))
    elif isinstance(colors,str):
        colors = [colors]*len(vdata)

    violins["cmeans"].set_color(colors)
    for vio,col in zip(violins["bodies"],colors):
        vio.set_color(col)
    return ax, violins

def _color_tiler(colors, n):
    r"""
    Tile colors up to n individual colors, related flare._utils.tocol_list_from_input_and_fragments

    This is how it works

    >>> plots._color_tiler("red",5)             #pure str
    >>> ['red', 'red', 'red', 'red', 'red']

    >>> plots._color_tiler(["red"],5)           # list of str with 1 item
    >>> ['red', 'red', 'red', 'red', 'red']

    >>> plots._color_tiler(["red", "blue"], 5)  # list of strs
    >>> ['red', 'blue', 'red', 'blue', 'red']

    >>> plots._color_tiler([[1., 0., 0.],       # red
    >>>                     [0., 1., 0.],       # green
    >>>                     [0., 0., 1.]], 5)   # blue
    >>> [[1.0, 0.0, 0.0],
    >>>  [0.0, 1.0, 0.0],
    >>>  [0.0, 0.0, 1.0],
    >>>  [1.0, 0.0, 0.0],
    >>>  [0.0, 1.0, 0.0]]

    Parameters
    ----------
    colors : str, list, array
        Color input to be tiled up to :ob:`n` colors
        using :obj:`numpy.tile`, e.g.
    n : int
        The number of needed colors

    Returns
    -------
    tiled_colors : list

    """
    if isinstance(colors,str):
        tiled_colors = _np.tile(colors,  _np.ceil(n / 1).astype(int) )[:n]
    elif _np.array(colors).ndim in [0,1]:
        tiled_colors = _np.tile(colors,  _np.ceil(n / len(colors)).astype(int) )[:n]
    else:
        tiled_colors = _np.tile(colors, (_np.ceil(n / len(colors)).astype(int),1))[:n]

    return tiled_colors.tolist()

#TODO test & document
def _is_axvline(line, value=None):
    xvals = line.get_data()[0]
    res = False
    res = line.get_label().startswith("_child") and len(xvals)==2 and xvals[0]==xvals[1]
    if res and value is not None:
        res = xvals[0]==value
    return res

import numpy as _np
from matplotlib import rcParams as _rcParams
import matplotlib.pyplot as _plt
from .list_utils import   \
    window_average_fast as _wav
from .str_and_dict_utils import \
    _delete_exp_in_keys, \
    unify_freq_dicts, \
    _replace_w_dict, \
    _replace4latex

def plot_w_smoothing_auto(iax, x, y,
                          ilabel,
                          icolor,
                          gray_background=False,
                          n_smooth_hw=0):
    alpha = 1
    if n_smooth_hw > 0:
        alpha = .2
        x_smooth = _wav(x, half_window_size=n_smooth_hw)
        y_smooth = _wav(y, half_window_size=n_smooth_hw)
        iax.plot(x_smooth,
                 y_smooth,
                 label=ilabel,
                 color=icolor)
        ilabel = None

        if gray_background:
            icolor = "gray"
    iax.plot(x, y,
             label=ilabel,
             alpha=alpha,
             color=icolor)

def compare_neighborhoods(filedict,
                          colordict,
                          anchor=None,
                          width=.2,
                          ax=None,
                          figsize=(10, 5),
                          fontsize=16,
                          mutations = {},
                          plot_singles=False,
                          exclude=None,
                          ctc_cutoff_Ang=None,
                          reorder_keys=True,
                          **kwargs_plot_unified_freq_dicts,
                          ):

    def neighborhood_datfile2freqws(ifile):
        outdict = {}
        for iline in open(ifile).read().splitlines():
            try:
                iline = iline.split()
                freq, names = iline[0],iline[1]
                outdict[names]=float(freq)
            except ValueError:
                print(iline)
                raise
        return outdict

    freqs = {key: {} for key in filedict.keys()}

    for key, ifile in filedict.items():
        if isinstance(ifile, str):
            idict = neighborhood_datfile2freqws(ifile)
        elif "contact_group" in str(type(ifile)):
            assert ctc_cutoff_Ang is not None, "Cannot provide a neighborhood object without a ctc_cutoff_Ang parameter"
            idict = {ilab:val for ilab, val in zip(ifile.ctc_labels_short,
                                                   ifile.frequency_per_contact(ctc_cutoff_Ang))}
        else:
            idict = {key:val for key, val in ifile.items()}

        idict = {_replace_w_dict(key, mutations):val for key, val in idict.items()}
        if anchor is not None:
            idict = _delete_exp_in_keys(idict, anchor)
        freqs[key] = idict

    """
    for key, val in freqs.items():
        print(key)
        for key, val in val.items():
            print(key,val)
        print()
    """

    if plot_singles:
        myfig, myax = _plt.subplots(1, 2, sharey=True, figsize=(figsize[0] * 2, figsize[1]))
        for iax, (key, ifreq) in zip(myax, freqs.items()):
            for ii, (jkey, jfreq) in enumerate(ifreq.items()):
                # if ii==0:
                #    label=skey
                # else:
                #    label=None
                _plt.sca(iax)
                _plt.bar(ii, jfreq, width=width,
                        color=colordict[key],
                        #        label=label
                        )
                _plt.text(ii, 1.05, jkey, rotation=45)
            _plt.gca().text(0 - width * 2, 1.05, "%s and:" % anchor, ha="right", va="bottom")
            _plt.ylim(0, 1)
            _plt.xlim(0 - width, ii + width)
            _plt.yticks([0, .25, .50, .75, 1])
            [_plt.gca().axhline(ii, ls=":", color="k", zorder=-1) for ii in [.25, .5, .75]]
        myfig.tight_layout()

    freqs  = unify_freq_dicts(freqs, exclude, reorder_keys=reorder_keys)
    myfig, iax, posret = plot_unified_freq_dicts(freqs,
                                                 colordict,
                                                 ax=ax,
                                                 width=width,
                                                 fontsize=fontsize,
                                                 figsize=figsize,
                                                 **kwargs_plot_unified_freq_dicts)
    _plt.text(0 - width * 2, 1.05, "%s and:" % anchor, ha="right", va="bottom")
    _plt.gcf().tight_layout()
    _plt.show()

    return myfig, freqs, posret

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


def plot_unified_freq_dicts(freqs,
                            colordict,
                            width=.2,
                            ax=None,
                            figsize=(10, 5),
                            fontsize=16,
                            lower_cutoff_val=.1,
                            order='mean',
                            remove_identities=False,
                            orientation='horizontal',
                            identity_cutoff=1,
                            ylim=1
                            ):

    #make copies of dicts
    freqs_work = {key:{key2:val2 for key2, val2 in val.items()} for key, val in freqs.items()}

    master_keys = list(freqs_work.keys())
    all_keys = list(freqs_work[master_keys[0]].keys())

    for mk in master_keys[1:]:
        assert len(all_keys)==len(list(freqs_work[mk].keys()))

    if remove_identities:
        popped_keys = []
        for key in all_keys:
            if all([val[key]>=identity_cutoff for val in freqs_work.values()]):
                [idict.pop(key) for idict in freqs_work.values()]
                popped_keys.append(key)
        all_keys = [key for key in all_keys if key not in popped_keys]

    order_vals = {"mean":[],
                  "std": [],
                  "keep":[]}
    for ii, key in enumerate(all_keys):
        order_vals["std"].append(_np.std([idict[key] for idict in freqs_work.values()])*len(freqs_work))
        order_vals["mean"].append(_np.mean([idict[key] for idict in freqs_work.values()]))
        order_vals["keep"].append(len(all_keys)-ii)

    under_freq = []
    for ii, ival in enumerate(order_vals[order]):
        if ival <= lower_cutoff_val:
            under_freq.append(ii)

    order_vals = {key:[ival for ii, ival in enumerate(val) if ii not in under_freq] for key, val in order_vals.items()}
    all_keys = [key for ii,key in enumerate(all_keys) if ii not in under_freq]
    order4plot = _np.argsort(order_vals[order])[::-1]

    _rcParams["font.size"] = 16

    delta = {}
    for ii, key in enumerate(master_keys):
        delta[key] = width * ii

    _rcParams["font.size"] = fontsize
    if ax is None:
        if orientation.startswith("hor"):
            myfig = _plt.figure(figsize=figsize)
        elif orientation.startswith("vert"):
            myfig = _plt.figure(figsize=figsize[::-1])
    else:
        _plt.sca(ax)
        myfig = ax.figure

    for ii, idx in enumerate(order4plot):
        key = all_keys[idx]
        for jj, (skey, sfreq) in enumerate(freqs_work.items()):
            if ii == 0:
                label = '%s ($\\Sigma$= %2.1f)'%(skey, _np.sum(list(sfreq.values())))
            else:
                label = None
            if orientation=="horizontal":
                _plt.bar(ii + delta[skey], sfreq[key],
                         width=width,
                         color=colordict[skey],
                         label=label,
                         )
                if jj == 0:
                    _plt.text(ii, ylim+.05, key, rotation=45)
            if orientation=="vertical":
                _plt.barh(ii + delta[skey], sfreq[key],
                          height=width,
                          color=colordict[skey],
                          label=label,
                          )
                if jj == 0:
                    _plt.text(-.05, ii, key, ha='right')
    _plt.legend()
    if orientation=='vertical':
        _plt.yticks([])
        _plt.xlim(0, ylim)
        _plt.ylim(0 - width, ii + width * 2)
        _plt.xticks([0, .25, .50, .75, 1])
        [_plt.gca().axvline(ii, ls=":", color="k", zorder=-1) for ii in [.25, .5, .75]]
        _plt.gca().invert_yaxis()

        if order == "std":
            _plt.plot(_np.array(order_vals[order])[order4plot], _np.arange(len(all_keys)), [order4plot], color='k', alpha=.25, ls=':')

    elif orientation=="horizontal":
        _plt.xticks([])
        _plt.ylim(0, ylim)
        _plt.xlim(0 - width, ii + width * 2)
        if ylim<=1:
            yticks = [0, .25, .50, .75, 1]

        else:
            yticks = (_np.arange(_np.round(ylim)))
        _plt.yticks(yticks)
        [_plt.gca().axhline(ii, ls=":", color="k", zorder=-1) for ii in yticks[1:-1]]
        if order == "std":
            _plt.plot(_np.array(order_vals[order])[order4plot] ,color='k', alpha=.25, ls=':')



    return myfig, _plt.gca(),{all_keys[ii]:[idict[all_keys[ii]] for idict in freqs_work.values()] for ii in order4plot}

def RMSD_segments_vs_orientations(dict_RMSD,
                                  segments,
                                  orientations,
                                  defs_residxs,
                                  exclude_trajs=None,
                                  bins=50,
                                  myax=None,
                                  panelsize=5,
                                  colors = ["red",
                                            "blue"],
                                  legends=None,
                                  dt_ps=1,
                                  wrt="mean structure",
                                  sharex=False,
                                  n_half_window=5,
                                  ):

    rescale=False
    if isinstance(sharex,bool) or sharex=="col":
        sharex_ =sharex
    elif sharex=="row":
        rescale=True
        sharex_=False
    else:
        raise ValueError
    linestyles = ["-", "--", ":", "-."]
    if isinstance(colors, list):
        colors = {key:col for key, col in zip(dict_RMSD.keys(), colors)}
    if legends is None:
        legends = {key:key for key in dict_RMSD.keys()}
    if exclude_trajs is None:
        exclude_trajs = {key: [] for key in dict_RMSD.keys()}
    if isinstance(bins, int):
        bins = {key: bins for key in dict_RMSD.keys()}

    if myax is None:
        nrows = len(orientations)
        ncols = len(segments) * 2
        myfig, myax = _plt.subplots(nrows, ncols, figsize=(ncols * panelsize, nrows * panelsize),
                                   sharey=False,
                                   sharex=sharex_,
                                   squeeze=False,
                                   )

    def my_average(array, weights, axis=1):
        if _np.size(weights) == 1:
            return array
        return _np.average(array, weights=weights, axis=axis)

    to_return={}
    histo_axes, trajs_axes = [], []
    for ii, (key, iRMSF) in enumerate(dict_RMSD.items()):
        to_return[key]={}
        for jax, orientation in zip(myax, orientations):
            to_return[key][orientation]={}
            #print(orientation)
            for seg_key, iax, iax2 in zip(segments, jax[::2], jax[1:][::2]):
                tohisto = [my_average(jRMSF[1][:, defs_residxs[key][seg_key]],
                                      weights=jRMSF[3][defs_residxs[key][seg_key]],
                                      axis=1) for (jj, jRMSF) in enumerate(iRMSF[orientation])
                           if jj not in exclude_trajs[key]
                           ]
                to_return[key][orientation][seg_key]=tohisto
                trajlabels = [ii for ii,__ in enumerate(iRMSF[orientation]) if ii not in exclude_trajs[key]]

                h, x = _np.histogram(_np.hstack(tohisto) * 10, bins=bins[key])
                label = 'Gs-GDP %-10s (%u trajs, %2.1f $\mu$s total)' % (legends[key], len(tohisto), h.sum() * dt_ps * 1e-6)
                iax.plot(x[:-1], h / h.max(), color=colors[key], ls='--',
                         label=label
                         # alpha=.50
                         )
                iax.legend()
                iax.set_title("[RMSD %s]\nwrt: %s\n oriented on %s" % (seg_key, wrt, orientation))
                iax.set_ylim([0,1])
                [iax2.plot(_np.arange(len(th))*dt_ps*1e-6, th*10, color="gray", alpha=.25, zorder=-10) for (ii,th) in enumerate(tohisto)]
                [iax2.plot(_wav(_np.arange(len(th)), n_half_window)*dt_ps*1e-6  , _wav(th.squeeze(), n_half_window) * 10,
                           color=colors[key], ls=linestyles[ii], label=trajlabels[ii]) for (ii, th) in enumerate(tohisto)]
                iax2.legend()
                # iax2.set_title("[RMSD GDP(t)]")
                histo_axes.append(iax)
                trajs_axes.append(iax2)

    [iax.set_xlabel("RMSD / $\AA$") for iax in myax[-1][::2]]

    if rescale:
        imin =  _np.min([iax.get_ylim()[0] for iax in trajs_axes])
        imax =  _np.max([iax.get_ylim()[1] for iax in trajs_axes])
        [iax.set_ylim([imin, imax]) for iax in trajs_axes]

    iax.figure.tight_layout()
    return iax.figure, myax, to_return


def span_domains(iax, domain_dict,
                 span_color='b',
                 pattern=None,
                 alternate=False,
                 zorder=-1,
                 rotation=0,
                 y_bump=.8,
                 alpha=.25,
                 transpose=False,
                 invert=False,
                 axis_lims=None
                 ):
    if pattern is not None and isinstance(pattern, str):
        from .parsers import match_dict_by_patterns
        include, __ = match_dict_by_patterns(pattern, domain_dict,
                                                            # verbose=True
                                                            )

    if alternate:
        fac = -1
    else:
        fac = 1

    if isinstance(span_color,list):
        color_array = _np.tile(span_color, _np.ceil(len(include)/len(span_color)).astype(int)+1)
        print(include)
        print(len(color_array), len(include),"lenghtsss", len(domain_dict))

    # Prepare some lambdas
    get_y_pos = lambda idx, iax: {1: iax.get_ylim()[0] * y_bump, -1: iax.get_ylim()[-1]}[idx]
    if invert:
        get_y_pos = lambda idx, iax: {1: iax.get_ylim()[-1] * y_bump, -1: iax.get_ylim()[0]}[idx]
    get_x_pos = lambda idx, iax: {1: iax.get_xlim()[0] * y_bump, -1: iax.get_xlim()[-1]}[idx]

    if axis_lims is None:
        within_limits = lambda x,y : True
    else:
        assert len(axis_lims)==2
        within_limits = lambda x, y : axis_lims[0]<x<[1] and axis_lims[0]<y<[1]

    idx = 1
    for cc, key in enumerate(include):
        val = domain_dict[key]
        #print(idx, key, y)
        if not transpose:
            y = get_y_pos(idx, iax)
            x = _np.mean(val)
            if within_limits(x,y):
                iax.text(x, y, key, ha="center", rotation=rotation,
                         )
                iax.axvspan(_np.min(val)-.5, _np.max(val)+.5, alpha=alpha,
                            zorder=zorder, color=color_array[cc])
        else:
                x = get_x_pos(idx, iax)
                y = _np.mean(val)
                if within_limits(x, y):
                    iax.text(x, y, key,
                             ha="right", rotation=rotation,
                             va="center")
                    iax.axhspan(_np.min(val)-.5, _np.max(val)+.5, alpha=alpha,
                                zorder=zorder, color=color_array[cc])
        idx *= fac

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
        jax.text(ix, iy, _replace4latex(ilab),
                 va='bottom',
                 ha='left',
                 rotation=45,
                 fontsize=_rcParams["font.size"]*label_fontsize_factor,
                 backgroundcolor="white"
                 )

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
    labels : list of len(2) with the lenghts N, M
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
    if transpose:
        mat = mat.T
        labels = labels[::-1]

    _plt.figure(figsize = _np.array(mat.shape)*pixelsize)
    im = _plt.imshow(mat,cmap=cmap)
    _plt.ylim([len(labels[0])-.5, -.5])
    _plt.xlim([-.5, len(labels[1])-.5])
    _plt.yticks(_np.arange(len(labels[0])),labels[0],fontsize=pixelsize*20)
    _plt.xticks(_np.arange(len(labels[1])), labels[1],fontsize=pixelsize*20,rotation=90)

    if grid:
        _plt.hlines(_np.arange(len(labels[0]))+.5,-.5,len(labels[1]),ls='--',lw=.5, color='gray', zorder=10)
        _plt.vlines(_np.arange(len(labels[1])) + .5, -.5, len(labels[0]), ls='--', lw=.5,  color='gray', zorder=10)

    if colorbar:
        _plt.gcf().colorbar(im, ax=_plt.gcf())
        im.set_clim(0.0, 1.0)

    return _plt.gca(), pixelsize
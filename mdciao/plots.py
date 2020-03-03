import numpy as _np
from matplotlib import rcParams as _rcParams
import matplotlib.pyplot as _plt
from .list_utils import _replace4latex,  \
    window_average_fast as _wav

def plot_contact(ictc, iax,
                 color_scheme=None,
                 ctc_cutoff_Ang=0,
                 n_smooth_hw=0,
                 dt=1,
                 gray_background=False,
                 shorten_AAs=False,
                 t_unit='ps',
                 ylim_Ang=10,
                 max_handles_per_row=4,
                 ):
    if color_scheme is None:
        color_scheme = _rcParams['axes.prop_cycle'].by_key()["color"]
    color_scheme = _np.tile(color_scheme, _np.ceil(ictc.n_trajs/len(color_scheme)).astype(int)+1)
    iax.set_ylabel('D / $\\AA$', rotation=90)
    if isinstance(ylim_Ang, (int, float)):
        iax.set_ylim([0, ylim_Ang])
    elif isinstance(ylim_Ang, str) and ylim_Ang.lower()== 'auto':
        pass
    else:
        raise ValueError("Cannot understand your ylim value %s of type %s" % (ylim_Ang,type(ylim_Ang)))
    for traj_idx, (ictc_traj, itime, trjlabel) in enumerate(zip(ictc.feat_trajs,
                                                                ictc.time_arrays,
                                                                ictc.trajlabels)):

        ilabel = '%s'%trjlabel
        if ctc_cutoff_Ang > 0:
            ilabel += ' (%u%%)' % (ictc.frequency_per_traj(ctc_cutoff_Ang)[traj_idx] * 100)

        plot_w_smoothing_auto(iax, itime * dt, ictc_traj * 10,
                              ilabel,
                              color_scheme[traj_idx],
                              gray_background=gray_background,
                              n_smooth_hw=n_smooth_hw)
    iax.legend(loc=1, fontsize=_rcParams["font.size"]*.75,
               ncol=_np.ceil(ictc.n_trajs/max_handles_per_row).astype(int)
               )
    ctc_label = ictc.label
    if shorten_AAs:
        ctc_label = ictc.ctc_label_short
    ctc_label = ctc_label.replace("@None","")
    if ctc_cutoff_Ang>0:
        ctc_label += " (%u%%)"%(ictc.frequency_overall_trajs(ctc_cutoff_Ang) * 100)

    iax.text(_np.mean(iax.get_xlim()), 1*10/_np.max((10, iax.get_ylim()[1])), #fudge factor for labels
             ctc_label,
             ha='center')
    if ctc_cutoff_Ang>0:
        iax.axhline(ctc_cutoff_Ang, color='k', ls='--', zorder=10)

    iax.set_xlabel('t / %s' % _replace4latex(t_unit))
    iax.set_xlim([0, ictc.time_max * dt])
    iax.set_ylim([0,iax.get_ylim()[1]])

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
                          anchor,
                          colordict, width=.2, figsize=(10, 5),
                          fontsize=16,
                          mutations = {},
                          plot_singles=False,
                          freq_cutoff_val=.1,
                          exclude=None,
                          scale_fig=False):

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

    def un_anchor_keys(idict, anchor,sep="-"):
        out_dict = {}
        for names, val in idict.items():
            name = [name for name in names.split(sep) if anchor not in name]
            assert len(name) == 1, name
            out_dict[name[0]]=val
        return out_dict

    from matplotlib import rcParams as _rcParams

    freqs = {key: {} for key in filedict.keys()}
    #assert len(filedict) == 2

    for key, ifile in filedict.items():
        idict = neighborhood_datfile2freqws(ifile)
        idict = {replace_w_dict(key, mutations):val for key, val in idict.items()}
        idict = un_anchor_keys(idict, anchor)
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

    freqs  = unify_freq_dicts(freqs, exclude)
    myfig, __ = plot_unified_freq_dicts(freqs,
                                     colordict,
                                     width=width, fontsize=fontsize, figsize=figsize)
    _plt.text(0 - width * 2, 1.05, "%s and:" % anchor, ha="right", va="bottom")
    _plt.gcf().tight_layout()
    _plt.show()

    return myfig

def replace_w_dict(key, pat_exp_dict):
    for pat, exp in pat_exp_dict.items():
        key = key.replace(pat,exp)
    return key

def plot_unified_freq_dicts(freqs,
                            colordict,
                            width=.2,
                            figsize=(10, 5),
                            fontsize=16,
                            freq_cutoff_val=.1
                            ):
    
    master_keys = list(freqs.keys())
    all_keys = list(freqs[master_keys[0]].keys())
    for mk in master_keys[1:]:
        assert len(all_keys)==len(list(freqs[mk].keys()))

    mean = []
    for key in all_keys:
        imean = []
        for idict in freqs.values():
            imean.append(idict[key])
        mean.append(_np.mean(imean))

    _rcParams["font.size"] = 16

    delta = {}
    for ii, key in enumerate(master_keys):
        delta[key] = width * ii

    _rcParams["font.size"] = fontsize
    myfig = _plt.figure(figsize=figsize)
    for ii, idx in enumerate(_np.argsort(mean)[::-1]):
        key = all_keys[idx]
        # print(mean[idx], key)
        for jj, (skey, sfreq) in enumerate(freqs.items()):
            if ii == 0:
                label = '%s ($\\Sigma$= %2.1f)'%(skey, _np.sum(list(sfreq.values())))
            else:
                label = None
            _plt.bar(ii + delta[skey], sfreq[key], width=width,
                    color=colordict[skey],
                    label=label
                    )
            if jj == 0:
                _plt.text(ii, 1.05, key, rotation=45)
        if mean[idx] <= freq_cutoff_val:
            break
    _plt.legend()
    _plt.xticks([])
    _plt.ylim(0, 1)
    _plt.xlim(0 - width, ii + width * 2)
    _plt.yticks([0, .25, .50, .75, 1])
    [_plt.gca().axhline(ii, ls=":", color="k", zorder=-1) for ii in [.25, .5, .75]]
    return myfig, _plt.gca()

    # This will allow more than pair comparisons in the future
def unify_freq_dicts(freqs, exclude=None, replacement_dict={}):

    not_shared = []
    shared = []
    for idict1 in freqs.values():
        for idict2 in freqs.values():
            if not idict1 is idict2:
                not_shared += list(set(idict1.keys()).difference(idict2.keys()))
                shared += list(set(idict1.keys()).intersection(idict2.keys()))

    shared = list(_np.unique(shared))
    not_shared = list(_np.unique(not_shared))
    all_keys = shared + not_shared

    if exclude is not None:
        print("Excluding")
        for ikey, ifreq in freqs.items():
            for key in shared:
                for pat in exclude:
                    if pat in key:
                        ifreq.pop(key)
                        print("%s from %s" % (key, ikey))
                        all_keys = [ak for ak in all_keys if ak != key]

    for ikey, ifreq in freqs.items():
        for key in not_shared:
            if key not in ifreq.keys():
                ifreq[key] = 0

    if len(not_shared)>0:
        print("These interactions are not shared:\n%s" % (', '.join(not_shared)))
        print("Their cummulative ctc freq is %f. " % _np.sum(
            [[ifreq[key] for ifreq in freqs.values()] for key in not_shared]))

    return freqs

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
                                  ):
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
                                   sharex=sharex,
                                   squeeze=False,
                                   )

    def my_average(array, weights, axis=1):
        if _np.size(weights) == 1:
            return array
        return _np.average(array, weights=weights, axis=axis)

    for ii, (key, iRMSF) in enumerate(dict_RMSD.items()):
        for jax, orientation in zip(myax, orientations):
            #print(orientation)
            for seg_key, iax, iax2 in zip(segments, jax[::2], jax[1:][::2]):
                tohisto = [my_average(jRMSF[1][:, defs_residxs[key][seg_key]],
                                      weights=jRMSF[3][defs_residxs[key][seg_key]],
                                      axis=1) for (jj, jRMSF) in enumerate(iRMSF[orientation])
                           if jj not in exclude_trajs[key]
                           ]
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
                [iax2.plot(_wav(_np.arange(len(th)), 5)*dt_ps*1e-6  , _wav(th, 5) * 10,
                           color=colors[key], ls=linestyles[ii], label=trajlabels[ii]) for (ii, th) in enumerate(tohisto)]
                iax2.legend()
                # iax2.set_title("[RMSD GDP(t)]")

    [iax.set_xlabel("RMSD / $\AA$") for iax in myax[-1][::2]]

    iax.figure.tight_layout()
    return iax.figure, myax


def span_domains(iax, domain_dict, span_color='b', pattern=None, alternate=False):
    if pattern is not None and isinstance(pattern, str):
        from .parsers import match_dict_by_patterns
        include, __ = match_dict_by_patterns(pattern, domain_dict,
                                                            # verbose=True
                                                            )

    if alternate:
        fac = -1
    else:
        fac = 1
    idx = 1
    get_y_pos = lambda idx, iax: {1: iax.get_ylim()[0] * .80, -1: iax.get_ylim()[-1]}[idx]
    for cc, (key, val) in enumerate(domain_dict.items()):
        if key in include:
            y = get_y_pos(idx, iax)
            # print(idx, y)
            iax.text(_np.mean(val), y, key, ha="center")
            iax.axvspan(_np.min(val), _np.max(val), alpha=.25, zorder=-1, color=span_color)
            idx *= fac
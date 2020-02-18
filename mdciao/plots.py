import numpy as _np
from matplotlib import rcParams as _rcParams
import matplotlib.pyplot as _plt
from .list_utils import _replace4latex

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
        from .list_utils import window_average_fast as _wav
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
                          substitutions=["MG", "GDP"],
                          mutations = {},
                          plot_singles=False, stop_at=.1, scale_fig=False):

    from matplotlib import rcParams as _rcParams

    freqs = {key: {} for key in filedict.keys()}
    assert len(filedict) == 2
    for key, ifile in filedict.items():
        for iline in open(ifile).read().splitlines():
            try:
                iline = iline.split()
                freq, names = iline[0],iline[1]
                freq = float(freq)
                names = names.split("-")
                name = [name for name in names if anchor not in name]
                assert len(name)==1,name
                name = name[0]
                for isub in substitutions:
                    if name.startswith(isub):
                        name = isub
                        break
                for exp, pat in mutations.items():
                    name = name.replace(pat,exp)
                freqs[key][name] = freq
            except ValueError:
                print(iline)
                raise

    for key, val in freqs.items():
        print(key)
        for key, val in val.items():
            print(key,val)
        print()

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

    diffs = {}
    not_common = []
    common = []
    for idict1 in freqs.values():
        for idict2 in freqs.values():
            if not idict1 is idict2:
                not_common += list(set(idict1.keys()).difference(idict2.keys()))
                common += list(set(idict1.keys()).intersection(idict2.keys()))

    common = list(_np.unique(common))
    not_common = list(_np.unique(not_common))
    all_keys = common + not_common

    print("These interaction partners are not shared:", not_common)
    for ifreq in freqs.values():
        for key in not_common:
            if key not in ifreq.keys():
                ifreq[key] = 0

    _rcParams["font.size"] = 16

    delta = {}
    for ii, key in enumerate(filedict.keys()):
        delta[key] = width * ii

    mean = []
    for key in all_keys:
        imean = []
        for idict in freqs.values():
            imean.append(idict[key])
        mean.append(_np.mean(imean))

    _rcParams["font.size"] = fontsize
    _plt.figure(figsize=figsize)
    for ii, idx in enumerate(_np.argsort(mean)[::-1]):
        key = all_keys[idx]
        # print(mean[idx], key)
        for jj, (skey, sfreq) in enumerate(freqs.items()):
            if ii == 0:
                label = skey
            else:
                label = None
            _plt.bar(ii + delta[skey], sfreq[key], width=width,
                    color=colordict[skey],
                    label=label
                    )
            if jj == 0:
                _plt.text(ii, 1.05, key, rotation=45)
        if mean[idx] <= stop_at:
            break
    _plt.text(0 - width * 2, 1.05, "%s and:" % anchor, ha="right", va="bottom")
    _plt.legend()
    _plt.xticks([])
    _plt.ylim(0, 1)
    _plt.xlim(0 - width, ii + width * 2)
    _plt.yticks([0, .25, .50, .75, 1])
    [_plt.gca().axhline(ii, ls=":", color="k", zorder=-1) for ii in [.25, .5, .75]]
    _plt.gcf().tight_layout()
import numpy as _np
from matplotlib import rcParams as _rcParams
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
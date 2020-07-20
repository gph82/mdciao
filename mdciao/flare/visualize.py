
__author__ = 'gph82'
from .flare import fragment_selection_parser, cartify_segments, pol2cart, curvify_segments, col_list_from_input_and_fragments
from mdciao.utils.bonds import bonded_neighborlist_from_top

import numpy as _np

from matplotlib.widgets import AxesWidget
from ipywidgets import HBox as _HBox, VBox as _VBox
import molpx
#from .metrics import _res_pairs_2_CA_pairs, most_freq_SS as _most_freq_SS, bonded_neighborlist_from_top as _bonded_neighborlist_from_top
from myplots import empty_legend as _empty_legend
from scipy.spatial.distance import squareform as _sqf

#from . import generate as _generate
#from . import _bmutils
#from . import _linkutils

#from .analize import proj2simlist as _proj2simlist, \
#    simlist2datadicts as _simlist2datadicts

from matplotlib import rcParams as _rcParams
from matplotlib.patches import Circle as _Circle, Rectangle as _Rect

#import nglview as _nglview
#from ipywidgets import VBox as _VBox, Layout as _Layout, Button as _Button

import os as _os
#from . import _utils
from matplotlib import pyplot as _plt
#from myplots import mycolors

_mycolors=[
         'blue',
         'green',
         'red',
         'cyan',
         'magenta',
         'yellow',
         'lime',
         'maroon',
         'navy',
         'olive',
         'orange',
         'purple',
         'teal',
]
from bezier import Curve as _BZCurve
class my_BZCURVE(_BZCurve):
    """
    Modified Bezier curve to plot with line-width
    """

    def plot(self, num_pts, color=None, alpha=None, ax=None,lw=1):
        """Plot the current curve.

        Args:
            num_pts (int): Number of points to plot.
            color (Optional[Tuple[float, float, float]]): Color as RGB profile.
            alpha (Optional[float]): The alpha channel for the color.
            ax (Optional[matplotlib.artist.Artist]): matplotlib axis object
                to add plot to.

        Returns:
            matplotlib.artist.Artist: The axis containing the plot. This
            may be a newly created axis.

        Raises:
            NotImplementedError: If the curve's dimension is not ``2``.
        """
        from bezier import _plot_helpers
        if self._dimension != 2:
            raise NotImplementedError(
                "2D is the only supported dimension",
                "Current dimension",
                self._dimension,
            )

        s_vals = _np.linspace(0.0, 1.0, num_pts)
        points = self.evaluate_multi(s_vals)
        if ax is None:
            ax = _plot_helpers.new_axis()
        ax.plot(points[0, :], points[1, :], color=color, alpha=alpha, lw=lw)
        return ax


class _line_mixes(object):

    def __init__(self, cols = 'bgrcmy', lss = ['-','--','-.',':']):

        self._mixes = []
        self._cols = cols
        for ii in range(20):
            self._cols += cols
        for icol in cols:
            for ils in lss:
                self._mixes.append(_linestyle(icol, ils))

    @property
    def mixes(self):
        return self._mixes

class _linestyle(object):
    def __init__(self, color, linestyle):
        self.ls  = linestyle
        self.col = color


def segment_definition_table(top, names=['TM1','ICL1',
                                         'TM2','ECL1',
                                         'TM3','ICL2',
                                         'TM4','ECL2',
                                         'TM5','ICL3',
                                         'TM6','ECL3',
                                         'TM7-H8',
                                         'CT']):

    from .metrics import segment_residxs_ICS as _segment_residxs_ICS, segment_residxs_ECS as _segment_residxs_ECS

    last = 0
    segname = iter(names)
    for segment1, segment2 in zip(_segment_residxs_ICS(top), _segment_residxs_ECS(top)):
        for iseg in [segment1, segment2]:
            print('%-6s %8s %8s'%(next(segname), top.residue(last),   top.residue(iseg[0]-1)))
            print('%-6s %8s %8s'%(next(segname), top.residue(iseg[0]),top.residue(iseg[-1])))
            last = iseg[-1]+1

    for iseg in _segment_residxs_ICS(top)[-1:]:

        print('%-6s %8s %8s'%(next(segname), top.residue(last),   top.residue(iseg[0]-1)))
        print('%-6s %8s %8s'%(next(segname), top.residue(iseg[0]),top.residue(iseg[-1])))
        last = iseg[-1]+1
        #print(top.residue(last), top.residue(segment2[0] - 1))
        #print(top.residue(segment2[0]),top.residue(segment2[-1]))

def binary_ctcs2snake(ictcs, res_idxs_pairs, top, segments,
                      exclude_neighbors=1, alpha=1,
                      average=True,
                      iax=None,
                      segment_names=None,
                      mute_segments=[],
                      anchor_segments=[],
                      ss_dict=None,
                      panelsize=4,
                      colors=True,
                      **kwargs_svgsnake2matplotlib):
    """
    # Create a residue list
    list_of_non_zero_residue_indxs = _np.hstack(segments)

    # TODO review variable names
    col_list = _col_list_from_input_and_fragments(colors, segments, list_of_non_zero_residue_indxs)

    colors_by_resSeq={top.residue(ii).resSeq:cc for ii,cc in zip(list_of_non_zero_residue_indxs, col_list)}
    # TODO avoid code repetition with binary_ctc2flare
    # Create a dictionary
    residx2idx = _np.zeros(_np.max(list_of_non_zero_residue_indxs) + 1, dtype=int)
    residx2idx[:] = _np.nan
    residx2idx[list_of_non_zero_residue_indxs] = _np.arange(len(list_of_non_zero_residue_indxs))
    """


    ctcs_averaged = _np.average(ictcs, axis=0)
    res_idxs2mute, res_idxs2anchor, should_pair_be_plotted = residue_selection_by_mute_and_anchor_segments(segments, mute_segments,
                                                                                                    anchor_segments)

    # All formed contacts
    flat_idx_unique_formed_contacts = _np.argwhere(ctcs_averaged > 0).squeeze()

    # Snake plot!
    iax, dict_of_circles_by_residx_str = svgsnake2matplotlib(iax=iax,
                                                             **kwargs_svgsnake2matplotlib)

    # TODO avoid code repetition with binary_ctc2flare
    ctc_idxs_to_plot = []
    plotted_respairs = []
    array_for_sorting = []
    for ctc_idx in flat_idx_unique_formed_contacts:
        res_pair = res_idxs_pairs[ctc_idx].astype("int")
        if _np.abs(res_pair[0] - res_pair[1]) > exclude_neighbors and \
                should_pair_be_plotted(res_pair):
            ctc_idxs_to_plot.append(ctc_idx)
            plotted_respairs.append(res_pair)
            array_for_sorting.append(ctcs_averaged[ctc_idx])

    for ctc_idx in flat_idx_unique_formed_contacts:
        if ctc_idx in ctc_idxs_to_plot:
            if average:
                ialpha = ctcs_averaged[ctc_idx]
            else:
                ialpha = alpha
            res_pair = res_idxs_pairs[ctc_idx].astype("int")
            resSeq_pair = [top.residue(ii).resSeq for ii in res_pair]
            implcirc, jmplcirc = [dict_of_circles_by_residx_str[ip]["mpl"] for ip in resSeq_pair]
            # Find out which artist came first
            for ii in iax.artists:
                if implcirc is ii:
                    patchA, patchB = implcirc, jmplcirc
                    break
                elif jmplcirc is ii:
                    patchA, patchB = jmplcirc, implcirc
                    break

            # print(implcirc, jmplcirc)
            iax.annotate('',
                         xy=implcirc.get_center(), xycoords="data",
                         xytext=jmplcirc.get_center(), textcoords="data",
                         arrowprops=dict(arrowstyle="-",
                                         patchA=patchB, patchB=patchA,
                                         connectionstyle="arc3,rad=0.1",
                                         linewidth=5,
                                         alpha=ialpha
                                         ),
                         zorder=-10
                         )

    """
    if contact_dictionary is not None:
        descending_ctc_order = contact_dictionary["data"].argsort()[::-1][:20]
        for ii in descending_ctc_order:
            pair = contact_dictionary["resSeq_pairs"][ii]
            ctc = contact_dictionary["data"][ii]

    """

    return iax, iax.figure

def svgsnake2matplotlib(svgfile='snake_adrb2_human',
                        iax=None,
                        figsize=(20,20),
                        include_residx=False,
                        invert_y_axis=True,
                        colors_by_resSeq=None,
                        fontsize=10,
                        fadeout_by_resSeq=[]
                        ):
    import untangle

    try:
        doc = untangle.parse(svgfile)
    except OSError:
        doc = untangle.parse(svgfile,feature_external_ges=False)

    isres = lambda icirc: 'rcircle' in icirc['class']
    resSeq = lambda icirc: int(icirc['original_title'].split()[0][1:])

    if iax is None:
        _plt.figure(figsize=figsize)
        iax = _plt.gca()

    xmin = _np.inf
    xmax = -_np.inf
    ymin = _np.inf
    ymax = -_np.inf
    from collections import defaultdict as _defdict
    dict_of_circles_by_residx_str = _defdict(dict)
    for icircle in doc.svg.g.circle:
        x, y = float(icircle['cx']), float(icircle['cy'])
        radius = float(icircle["r"])
        facecolor = icircle["fill"]
        edgecolor = icircle["stroke"]
        if isres(icircle) and resSeq(icircle) in fadeout_by_resSeq:
            edgecolor='gray'
        lw = int(icircle['stroke-width'])
        #print(icircle)
        xmin, xmax = _np.min((xmin, x)), _np.max((xmax, x))
        ymin, ymax = _np.min((ymin, y)), _np.max((ymax, y))
        mplcirc = _Circle((x, y), radius=radius, facecolor=facecolor, edgecolor=edgecolor, lw=lw)
        iax.add_artist(mplcirc)
        if isres(icircle):
            iresSeq = resSeq(icircle)
            if iresSeq not in fadeout_by_resSeq:
                if colors_by_resSeq:
                    try:
                        facecolor = colors_by_resSeq[iresSeq]
                    except KeyError:
                        facecolor = 'white'

            assert iresSeq == int(icircle["id"])
            dict_of_circles_by_residx_str[iresSeq]["svg"] = icircle
            dict_of_circles_by_residx_str[iresSeq]["mpl"] = mplcirc
            mplcirc.set_facecolor(facecolor)

    for itxt in doc.svg.g.text:
        x, y = float(itxt['x']), float(itxt['y'])
        s = itxt.cdata
        textcolor='k'
        if itxt["onclick"] is None:
            if itxt["class"].startswith('rtext'): #we get the coords from the residue dit
                iresSeq = int(itxt["id"].replace('t', ""))
                if iresSeq in fadeout_by_resSeq:
                    textcolor='gray'
                mplcirc = dict_of_circles_by_residx_str[iresSeq]["svg"]
                x, y = float(mplcirc['cx']), float(mplcirc['cy'])
                if include_residx:
                    s = itxt["original_title"].split()[0]
            iax.text(x, y, s,
                     ha='center',
                     va='center',
                     fontsize=fontsize,
                     color=textcolor
                     )
        elif (itxt["onclick"].startswith('toggleLoop') and 'long' in itxt["onclick"]):
            iax.text(x, y, s,
                     ha='center',
                     va='center',
                     fontsize=fontsize
                     )

    for irect in doc.svg.g.rect:
        x, y = float(irect['x']), float(irect['y'])
        width, height = float(irect["width"]), float(irect["height"])
        facecolor = irect["fill"]
        edgecolor = irect["stroke"]
        if irect["onclick"] is None or (irect["onclick"].startswith('toggleLoop') and 'long' in irect["onclick"]):
            irect = _Rect((x, y), width, height,
                              joinstyle='round',
                              facecolor=facecolor, edgecolor=edgecolor,
                              )
            iax.add_artist(irect)
        # print(x,y)

    iax.set_xlim([xmin - 20, xmax + 20])
    iax.set_ylim([ymin - 20, ymax + 20])
    iax.set_xticks([])
    iax.set_yticks([])
    iax.set_xticklabels([])
    iax.set_yticklabels([])


    if invert_y_axis:
        iax.set_ylim(*iax.get_ylim()[::-1])

    iax.set_aspect('equal')

    return iax, dict_of_circles_by_residx_str


def adaptive_viewer(projdir, panelsize=2, ratio=1,
                    simdir = 'simdir',
                    ylim=None,
                    input_feats=None,
                    LL_modules=["gcc/7"],
                    feat_factor=1.0,
                    return_sims=False,
                    reg_exp=None,
                    force_featurization=False,
                    force_whole=False,
                    mdtraj_sel_str=None,
                    top=None,
                    n_jobs=1,
                    **dir2sim_kwargs,
                    ):


    simlist = _proj2simlist(projdir,
                           simdir=simdir,
                           LL_modules=LL_modules,
                           reg_exp=reg_exp,
                           force_whole=force_whole,
                           mdtraj_sel_str=mdtraj_sel_str,
                           top=top,
                           n_jobs=n_jobs,
                           **dir2sim_kwargs,
                           )

    epochs = _np.unique([isim.epoch for isim in simlist])
    n_epochs = len(_np.unique(epochs))
    print("\nn epochs", n_epochs, epochs)

    epoch_data, serial2epoch = _simlist2datadicts(simlist,
                                                 input_feats=input_feats,
                                                 force_featurization=force_featurization,
                                                 feat_factor=feat_factor,
                                                  n_jobs=n_jobs)
    seeds = []
    for isim in simlist:
        # Append seeds
        try:
            # print(_os.path.basename(isim.starting_pdb.replace('.pdb', "")))
            # print(_utils.starting_fname2seeds(_os.path.basename(isim.starting_pdb.replace('.pdb', ""))))
            seeds.append(_utils.starting_fname2seeds(_os.path.basename(isim.starting_pdb.replace('.pdb', ""))))
        except AttributeError:
            seeds.append([isim.epoch, isim.serial, 0])

    line_mixes = _line_mixes()
    n_epochs = len(epoch_data.keys())
    print("n epochs",n_epochs, epoch_data.keys())
    n_frames = _np.sum([_np.sum([len(ival) for ival in val]) for key, val in epoch_data.items()])


    if n_frames>0:
        n_sims = _np.max([len(val) for val in epoch_data.values()])
        myfig, myax = _plt.subplots(n_sims, n_epochs, sharex="col",
                                   figsize=(n_epochs*panelsize, n_sims*panelsize*ratio),
                                   sharey=True, squeeze=False)
        serial_count = 0
        serial2ax = {}
        for key, values in epoch_data.items():
            for ii, idata in enumerate(values):
                for jdata in idata.T:
                    myax[ii, key].plot(jdata, label='e%02u_s%03u'%(key,serial_count),
                                       color=line_mixes._cols[key],
                                       )
                myax[ii, key].legend(loc='best')
                serial2ax[serial_count] = myax[ii, key]
                serial_count += 1

        for ii, iseed in enumerate(seeds):
            epoch, serial, frame = iseed[0], iseed[1], iseed[2]
            if serial is None:
                serial = 0
            if frame is None:
                frame = 0
            if epoch is None:
                epoch = 0
            serial2ax[serial].axvline(frame,
                                      color=line_mixes._cols[serial2epoch[ii]]

                                      )
            t = serial2ax[serial].text(frame,
                                   _np.average(serial2ax[serial].get_ylim()),
                                   ii,
                                   color=line_mixes._cols[serial2epoch[ii]],
                                   backgroundcolor='w',
                                   ha='center')
            t.set_bbox(dict(facecolor='w',
                            alpha=0.75,
                            edgecolor='None'
                            )
                       )
        #myfig.tight_layout(w_pad=0, h_pad=0, pad=0)
        myfig.tight_layout()

        if ylim is not None:
            serial2ax[0].set_ylim(ylim)

        data = [[_np.hstack([jdata[:, ii] for jdata in epoch_data[key]]) for key in range(n_epochs)] for ii in
                range(idata.shape[1])]
        data = [_utils.list_of_lists_2_NaN_arrays(jdata) for jdata in data]

        nbins=25
        for idata in data:
            myfig2, myax2 = _plt.subplots(1, 2, sharey=True)  # _plt.figure()
            means, std = _np.vstack((_np.nanmean(idata, axis=(-1)),_np.nanstd(idata,axis=(-1))))
            ##_plt.boxplot(data)
            myax2[0].errorbar(_np.arange(n_epochs), means, std)
            myax2[0].axes.set_xlabel('adaptive epoch')

            for jj, jdata in enumerate(idata[:]):
                kdata = [kpoint for kpoint in jdata if not _np.isnan(kpoint)]
                h, x = _np.histogram(kdata, bins=nbins,
                                     #density=True,
                                     )
                myax2[1].plot(h, x[:-1], color=line_mixes._cols[jj])

            h, x = _np.histogram([kdata for kdata in _np.hstack([idata.reshape(-1) for idata in data])
                                  if not _np.isnan(kdata)], bins=nbins)
            myax2[1].plot(h, x[:-1], color='k')
        myfig2.tight_layout()

        if return_sims:
            return (myfig, myfig2), simlist
        else:
            return (myfig, myfig2)
    else:
        return simlist


def animate_epoch(Y, isims,
                  max_epoch=None,
                  myax=None,
                  panelsize=4,
                  alpha_past=.25,
                  color_list='bgrymk',
                  proj_idxs=[0,1],
                  dt=1,
                  t_unit="frames"):
    if max_epoch is None:
        max_epoch= _np.max([isim.epoch for isim in isims])

    if myax is None:
        myfig, myax = _plt.subplots(2, 2,
                                   sharex='col',
                                   sharey='row',
                                   figsize=(2 * panelsize,
                                            2 * panelsize))

    data = [iY[:, proj_idxs] for ii, iY in enumerate(Y) if isims[ii].epoch <= max_epoch or max_epoch<0]
    h, x, y = _np.histogram2d(*_np.vstack(data).T, bins=100)
    _plt.sca(myax[0, 0])
    _plt.contourf(y[:-1], x[:-1], -_np.log(h))

    _plt.ylabel("C %u" % proj_idxs[0])
    marks = 'odx.'
    for ii, iY in enumerate(Y):
        # if np.any(Y[ii][:,0]>9):
        ee = isims[ii].epoch
        color = color_list[ee]
        if ee <= max_epoch:
            if ee < max_epoch:
                alpha = alpha_past
            else:
                alpha = 1
            time_array = _np.arange(len(iY))*dt
            myax[0, 1].plot(time_array, iY[:, proj_idxs[0]], label=ii,
                            color=color,
                            # marker=marks[np.mod(ii,4)],
                            alpha=alpha,
                            )
            myax[1, 0].plot(iY[:, proj_idxs[1]], time_array, label=ii,
                            color=color,
                            # marker=marks[np.mod(ii,4)],
                            alpha=alpha,
                            )
            if ee == max_epoch:
                myax[0, 0].plot(iY[0, proj_idxs[1]], iY[0, proj_idxs[0]], '-o',
                                markerfacecolor=color, markeredgecolor=color)
    myax[1, 0].invert_yaxis()
    myax[1, 0].set_ylabel('t/ %s'%t_unit)
    myax[1, 0].set_xlabel("C %u" % proj_idxs[1])

    # plt.legend(fontsize=5)
    myfig.tight_layout()
    myax[0, 1].set_xlabel('t/ %s'%t_unit)
    return myfig, myax

def _blockify_showmat(iax, blocks, block_names=[], labels="lower"):
    r"""

    :param iax:
    :param blocks:
    :param block_names:
    :param labels: "lower", "upper", "left", "right"
    :return:
    """
    pass
    offset = 0
    if labels.lower()=="lower":
        y_pos = iax.dataLim.ymax
        y_offset = +20
        va = "top"
        ha = "left"
        rotation = -45
        clean_ax = iax.set_xticks
    elif labels.lower()=="upper":
        y_pos = iax.dataLim.ymin
        y_offset = -20
        va  = "bottom"
        ha = "left"
        clean_ax = iax.set_xticks
        rotation = +45
    elif labels.lower()=="left":
        y_pos = iax.dataLim.xmin
        y_offset = -5
        va = "center"
        ha = "right"
        rotation = 0
        clean_ax = iax.set_yticks

    #if len(block_names) == 0:
    #    block_names = [[""]*len(blocks)]

    for ii, iseg in enumerate(blocks):
        iax.axvline(len(iseg)+offset, color="k", alpha=.25)
        iax.axhline(len(iseg)+offset, color="k", alpha=.25)

        if len(block_names)==len(blocks):
            pos_pairs = [offset+len(iseg)/2, y_pos+y_offset]
            if labels.lower() in ["lower", "upper"]:
                pass
            else:
                pos_pairs = pos_pairs[::-1]
            iax.text(pos_pairs[0], pos_pairs[1], block_names[ii], va=va, ha=ha,
                 rotation=rotation,
                )
            clean_ax([])
        offset+=len(iseg)

def cart2pol(x, y):
    rho = _np.sqrt(x**2 + y**2)
    phi = _np.arctan2(y, x)
    return(rho, phi)


#Charged positive(BLUE) ARG, HIS, LYS
# Charged negative(RED) GLU, ASP
#Polar(GREEN) ASN, CYS, GLN, SER, THR
#Hydropobic + Other(WHITE) ALA, ILE, LEU, MET, VAL, PHE, TRP, TYR, GLY, PRO
color_restype_dict =      {rname: 'BLUE' for rname in 'ARG HIS LYS'.split()}
color_restype_dict.update({rname: 'RED' for rname in "GLU ASP".split()})
color_restype_dict.update({rname: 'GREEN' for rname in "ASN CYS GLN SER THR".split()})
color_restype_dict.update({rname: 'WHITE' for rname in "ALA ILE LEU MET VAL PHE TRP TYR GLY PRO".split()})

def color_by_restype(top, unknown='white',
                     replacement_dict=None):
    colors = []
    idict = {key:val for key, val in color_restype_dict.items()}
    for pat,new in replacement_dict.items():
        idict={key:val.replace(pat,new) for key, val in idict.items()}
    for rr in top.residues:
        try:
            colors.append(idict[rr.name])
        except KeyError:
            print("no color for %s. Using %s"%(rr,unknown))
            colors.append(unknown)
    return _np.hstack(colors)



def _list_of_list_idxs(ilist):
    r"""
    Provided a list of list, return the flat indices of the inner lists
    :param ilist:
    :return:
    """
    out_list = []
    offset = 0
    for jlist in ilist:
        out_list.append(_np.arange(offset, offset+len(jlist)))
        offset += len(jlist)
    return  out_list

_SS2vmdcol = {'H':"purple", "E":"yellow","C":"cyan", "NA":"gray"}

def _my_vmd_colormap(n, offset=.1,midpoint=.5):
    x = _np.linspace(0,1, num=n)
    nmid = int(round(n * midpoint))
    R = _np.zeros_like(x)+offset
    G = _np.zeros_like(x)+offset
    B = _np.zeros_like(x)+offset
    R[:nmid] = _np.linspace(1,0+offset, num=nmid)
    G[:nmid] = _np.linspace(0+offset,1, num=nmid)
    G[-nmid:] = _np.linspace(1,0+offset, num=nmid)
    B[-nmid:] = _np.linspace(0+offset,1, num=nmid)
    #print(x)
    #print(R)
    #print(G)
    #print(B)
    #_plt.figure()
    #_plt.plot(R,'-',color='red')
    #_plt.plot(G, '-', color='green')
    #_plt.plot(B, '-', color='blue')
    return _np.vstack((R,G,B)).T

def paint_by_segments(iwd, segments, top, colors, paint_complementary=True, reset_all_reps=True,n_comps=1):
    not_res = [ii for ii in range(top.n_residues) if ii not in _np.hstack(segments)]
    for ii in range(n_comps):
        if reset_all_reps:
            iwd.remove_cartoon(component=ii)
            iwd.remove_ball_and_stick(component=ii)

    for iseg, icol in zip(segments, colors):
        ires = _np.hstack([iseg[0] - 1, iseg, iseg[-1] + 1])
        ires = _np.array([ii for ii in ires if ii >= 0])
        iwd.add_cartoon(top.select("resid %s" % (' '.join(['%u' % ii for ii in ires]))), color=icol, component=ii)



def correlations(correlation_input, res_idxs_pairs, segments,
                 panelsize=4,
                 n_argmax=10,
                 extrema_geoms=None,
                 extrema_ctcs=None,
                 **binary_ctcs2flare_kwargs):

    bigbox = lambda tuple: _VBox(tuple)
    smallbox = lambda tuple: _HBox(tuple)
    from myplots  import empty_legend as _empty_legend

    if extrema_geoms is None and extrema_ctcs is None:
        static=True
        ncols = 2
        nrows = _np.ceil(correlation_input.shape[1] / 2.).astype(int)
        myfig, myax = _plt.subplots(nrows, ncols, squeeze=False,
                                    sharex=True, sharey=True,
                                    figsize=(ncols * panelsize, nrows * panelsize),
                                    )
    else:
        static=False

    iboxes = []
    for ii, icorr in enumerate(correlation_input.T):
        if not static:
            iax = None
            _plt.ioff()
        else:
            iax = myax.flatten()[ii]

        jcorr = _np.zeros_like(icorr)
        most_corr =_np.abs(icorr).argsort()[::-1][:n_argmax]
        jcorr[most_corr] = _np.abs(icorr)[most_corr]
        ilabel = 'proj %u\n%u most correlated\n$\in$[%2.1f,%2.1f]'%(ii, n_argmax,
                                                          _np.abs(icorr)[most_corr].min(),
                                                          jcorr.max())
        #print(label)
        iax, respairs_descending = binary_ctcs2flare(jcorr.reshape(1, -1),
                                                     res_idxs_pairs,
                                                     segments,
                                                     iax=iax,
                                                     panelsize=panelsize,
                                                     **binary_ctcs2flare_kwargs,
                                                     )
        _empty_legend(iax, [ilabel])
        iax.figure.tight_layout(h_pad=0,w_pad=0, pad=0)
        if not static:
            iax_wdg = AxesWidget(iax)
            toappend=[iax_wdg.canvas]

        if extrema_geoms is not None:
            [igeom.superpose(extrema_geoms[0]) for igeom in extrema_geoms]
            igeom = extrema_geoms[ii]
            iwd = molpx.visualize._nglwidget_wrapper(igeom)
            iwd._set_size(*['%fin' % inches for inches in iax.get_figure().get_size_inches()])

            most_freq_CAs = _res_pairs_2_CA_pairs(respairs_descending, igeom.top).T
            molpx._bmutils.add_atom_idxs_widget(most_freq_CAs[:n_argmax], iwd, color_list=['black'])

            toappend+=[iwd]
            paint_by_segments(iwd, segments, igeom.top, _mycolors)

            iwd.center()
            #axes.append(iax)
            #wdgs.append(iwd)

        if extrema_ctcs is not None:
            target_residx = _np.unique(res_idxs_pairs[respairs_descending[:n_argmax]])
            for jj, ictc in enumerate(extrema_ctcs[ii]):
                ctc_kwargs = {key:val for key, val in binary_ctcs2flare_kwargs.items()}
                jctc = _np.copy(ictc)
                for kk, ipair in enumerate(res_idxs_pairs):
                    if len(_np.intersect1d(ipair, target_residx))==0:
                        #jctc[kk]=0
                        pass
                if extrema_geoms is not None and 'ss_dict' not in ctc_kwargs.keys():
                    ctc_kwargs["ss_dict"] =_most_freq_SS(extrema_geoms[ii][jj])[_np.unique(res_idxs_pairs)]
                else:
                    pass

                iax, __ = binary_ctcs2flare(jctc.reshape(1,-1),
                                                         res_idxs_pairs,
                                                         segments,
                                                         iax=None,
                                                         panelsize=panelsize,
                                                         **ctc_kwargs,
                                                         )
                _empty_legend(iax, ['arg%s'%["min","max"][jj]])
                iax.figure.tight_layout(h_pad=0, w_pad=0, pad=0)

                iax_wdg = AxesWidget(iax)
                toappend+=[iax_wdg.canvas]

        if not static:
            iboxes.append(smallbox(toappend))

    _plt.ion()
    if not static:
        return bigbox(iboxes)
    else:
        return myfig

# from https://www.rosettacode.org/wiki/Range_expansion#Python
def _rangeexpand(txt):
    lst = []
    for r in txt.split(','):
        if '-' in r[1:]:
            r0, r1 = r[1:].split('-', 1)
            lst += range(int(r[0] + r0), int(r1) + 1)
        else:
            lst.append(int(r))
    return lst

def add_rep_vmdstyle_resid(iwd, top, reptype, resids, color='red'):
    if isinstance(resids, str):
        resids = _rangeexpand(resids)
    elif isinstance(resids,int):
        resids = [resids]

    selstr = 'residue %s'%(' '.join(['%u'%ii for ii in resids]))
    selection_atoms = top.select(selstr)
    if reptype.lower() == "licorice":
        iwd.add_licorice(selection_atoms)
    else:
        raise NotImplementedError


#def add_rep_vmdstyle(iwd,reptype, selection):


def binary_ctcs2flare(ictcs, res_idxs_pairs,
                      fragments=None,
                      exclude_neighbors=1,
                      alpha=1,
                      freq_cutoff=0,
                          iax=None,
                      fragment_names=None,
                      center=_np.array([0,0]),
                      r=1,
                      mute_segments=None,
                      anchor_segments=None,
                      ss_array=None,
                      panelsize=4,
                      angle_offset=0,
                      highlight_residxs=None,
                      vicinities=None,
                      top=None,
                      radial_padding_percent=10,
                      colors=True,
                      fontsize=6,
                      return_descending_ctc_freqs=False,
                      shortenAAs=False, aa_offset=0,
                      markersize=5,
                      bezier_linecolor='k',
                      plot_curves_only=False,
                      curves=False,
                      textlabels=True,
                      no_dots=False,
                      padding_beginning=0,
                      padding_end=0,
                      lw=1,
                      ):
    r"""
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
    fragments: list of lists of integers
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
    mute_segments: iterable of integers, default is None
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
    vicinities
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
    if not return_descending_ctc_freqs:
        return iax, res_pairs_descending
    else:
        return iax, res_pairs_descending, sorted(array_for_sorting)[::-1]
    """

    padding_end += 5 #todo dangerous?

    if _np.ndim(ictcs)==1:
        ictcs = ictcs.reshape(1,-1)
    elif _np.ndim(ictcs)==2:
        pass

    else:
        raise ValueError("Input array has to of shape either (m) or (n, m) where n : n frames, and m: n_contacts")

    assert ictcs.shape[1]==len(res_idxs_pairs), "The size of the contact array and the res_idxs_pairs array do not match %u vs %u"%(ictcs.shape[1], len(res_idxs_pairs))

    if iax is None:
        assert not plot_curves_only,("You cannot use plot_curves_only=True and iax=None. Makes no sense")
        _plt.figure(figsize=(panelsize, panelsize))
        iax = _plt.gca()

    radial_padding_units=1+radial_padding_percent/100


    # Define some useful lambdas for deciding if plotting or not plotting a pair
    # Create a residue list
    residue_idxs_in_input_segments = _np.hstack(fragments)
    # Condition did I select it with my fragments
    is_pair_not_muted_bc_one_residx_is_not_in_input_segments = lambda pair : all(_np.in1d(pair, residue_idxs_in_input_segments))

    # Condition vicinities
    if vicinities is not None:
        is_pair_not_muted_bc_vicinities = lambda pair : len(_np.intersect1d(pair,vicinities))>0
    else:
        is_pair_not_muted_bc_vicinities = lambda pair : True

    # Condition in anchor segment or in muted segment
    is_pair_not_muted_bc_anchor_and_mute_segments = \
        fragment_selection_parser(fragments, mute_segments, anchor_segments)

    # Condition not nearest neighbors
    if top is None:
        is_pair_not_muted_bc_nearest_neighbors = lambda pair : _np.abs(pair[0] - pair[1]) > exclude_neighbors
    else:
        nearest_n_neighbor_list = bonded_neighborlist_from_top(top, exclude_neighbors)
        is_pair_not_muted_bc_nearest_neighbors = lambda pair : pair[1] not in nearest_n_neighbor_list[pair[0]]

    # Angular/cartesian quantities
    xy = cartify_segments(fragments, r=r, angle_offset=angle_offset,
                          padding_initial=padding_beginning,
                          padding_final=padding_end,
                          padding_between_fragments=1)
    xy += center
    xy_labels, xy_angles = cartify_segments(fragments, r=r * radial_padding_units, return_angles=True, angle_offset=angle_offset,
                                            padding_initial=padding_beginning,
                                            padding_final=padding_end,
                                            padding_between_fragments=1)
    xy_labels += center

    # Do we have SS dictionaries
    if ss_array is not None:
        if plot_curves_only:
            print("Ignoring input %s because plot_curves_only is %s" % ("ss_dict", plot_curves_only))
        else:
            #assert len(ss_dict)==len(res_idxs_pairs)
            xy_labels_SS, xy_angles_SS = cartify_segments(fragments, r=r * radial_padding_units ** 1.7, return_angles=True, angle_offset=angle_offset,
                                                          padding_initial=padding_beginning,
                                                          padding_final=padding_end,
                                                          padding_between_fragments=1)
            xy_labels_SS += center
    # Do we have names?
    if fragment_names:
        if plot_curves_only:
            print("Ignoring input %s because plot_curves_only = %s" % ("fragment_names", plot_curves_only))
        else:
            for seg_idxs, iname in zip(_list_of_list_idxs(fragments), fragment_names):
                xseg, yseg = xy[seg_idxs].mean(0)-center
                rho, phi = cart2pol(xseg, yseg)
                xseg, yseg = pol2cart(r * radial_padding_units ** 2.5, phi) + center

                iang=phi+_np.pi/2
                if _np.cos(iang) < 0:
                    iang = iang + _np.pi
                iax.text(xseg, yseg, iname, ha="center", va="center",
                         fontsize=fontsize*2,
                         rotation=_np.rad2deg(iang))

    # TODO review variable names
    col_list = col_list_from_input_and_fragments(colors, fragments, residue_idxs_in_input_segments)

    # Plot!
    # Do this first to have an idea of the points per axis unit necessary for the plot
    iax.set_xlim([-1.5, 1.5])
    iax.set_ylim([-1.5, 1.5])
    iax.set_yticks([])
    iax.set_xticks([])
    iax.set_aspect("equal")

    """
    _plt.ion()
    iax.figure.canvas.draw()
    points_per_axis_unit = iax.get_window_extent().width / (_np.diff(iax.get_xlim()).squeeze())
    av_dist_between_centers = _np.linalg.norm(xy[:-1]-xy[1:], axis=1).mean()
    av_dist_between_centers /= iax.figure.get_size_inches().mean() # shamelessly some fudging here...
    s2 = (av_dist_between_centers * points_per_axis_unit) ** 2
    iax.scatter(xy[:, 0], xy[:, 1], c=col_list, s=s2)
    """
    if plot_curves_only:
        print("Not scattering, coloring, or labelling anything because plot_curves_only = %s" % (plot_curves_only))
    else:
        if not curves:
            iax.scatter(xy[:, 0], xy[:, 1], c=col_list, s=markersize, zorder=10)
        else:
            # todo CLEAN THIS curfify and colors thing up
            list_of_curves_for_fragments = curvify_segments(fragments, r=r, angle_offset=angle_offset, padding=padding_end)
            col_list_new = []
            for icol in col_list:
                if icol not in col_list_new:
                    col_list_new.append(icol)
            for ii, (icurve,icol) in enumerate(zip(list_of_curves_for_fragments, col_list_new)):
                iax.plot(icurve[:,0],icurve[:,1],'-',lw=5, color=icol)

        if textlabels:
            for shown_residue_index, (ires, ixy, iang) in enumerate(zip(residue_idxs_in_input_segments, xy_labels, xy_angles)):
                if _np.cos(iang) < 0:
                    iang = iang+_np.pi
                ilabel = ires
                txtclr = "k"
                if top is not None:
                    if not shortenAAs:
                        ilabel = top.residue(ires)
                    else:
                        idxs = top.residue(ires).resSeq + aa_offset
                        ilabel = ("%s%u"%(top.residue(ires).code,idxs)).replace("None",top.residue(ires).name)
                    if highlight_residxs is not None and ires in highlight_residxs:
                        txtclr="red"

                itxt = iax.text(ixy[0], ixy[1], '%s'%ilabel,
                                color=txtclr,
                                va="center",
                                ha="center",
                                rotation=_np.rad2deg(iang),
                                fontsize=fontsize)

    # TODO refactor to use less code
    if ss_array is not None and not plot_curves_only:
        for shown_residue_index, ixy, iang in zip(residue_idxs_in_input_segments, xy_labels_SS, xy_angles_SS):
            if _np.cos(iang) < 0:
                iang = iang+_np.pi
            ilabel = ss_array[shown_residue_index]
            itxt = iax.text(ixy[0], ixy[1], '%s'%ilabel,
                            ha="center", va="center", rotation=_np.rad2deg(iang),
                            fontsize=fontsize, color=_SS2vmdcol[ilabel], weight='heavy')

    # Create a dictionary
    residx2idx = _np.zeros(_np.max(residue_idxs_in_input_segments) + 1, dtype=int)
    residx2idx[:] = _np.nan
    residx2idx[residue_idxs_in_input_segments] = _np.arange(len(residue_idxs_in_input_segments))

    ctcs_averaged = _np.average(ictcs, axis=0)
    # All formed contacts
    flat_idx_unique_formed_contacts = _np.argwhere(ctcs_averaged>freq_cutoff).squeeze()

    # Create a dictionary of initialized bezier curves with the residxs as keys
    # TODO each curve will be used only once, but it is better to have it like this
    #  for per-frame operations later on (otherwise we could use the same loop)
    bz_curve = {}
    ctc_idxs_to_plot = []
    plotted_respairs = []
    array_for_sorting = []
    for shown_residue_index in flat_idx_unique_formed_contacts:
        res_pair = res_idxs_pairs[shown_residue_index].astype("int")
        if  is_pair_not_muted_bc_one_residx_is_not_in_input_segments(res_pair) and \
            is_pair_not_muted_bc_nearest_neighbors(res_pair) and \
            is_pair_not_muted_bc_anchor_and_mute_segments(res_pair) and\
            is_pair_not_muted_bc_vicinities(res_pair):
            node_idxs = residx2idx[res_pair].squeeze()
            nodes = xy[node_idxs]
            bz_curve[shown_residue_index] = create_flare_bezier(nodes, center=center)
            ctc_idxs_to_plot.append(shown_residue_index)
            plotted_respairs.append(res_pair)
            array_for_sorting.append(ctcs_averaged[shown_residue_index])
    for shown_residue_index in flat_idx_unique_formed_contacts:
        if shown_residue_index in ctc_idxs_to_plot:
            ialpha = ctcs_averaged[shown_residue_index]
            bz_curve[shown_residue_index].plot(50, ax=iax, alpha=ialpha,
                                   color=bezier_linecolor,
                              lw=lw,#_np.sqrt(markersize),
                              #zorder=-1
                              )

    # Cosmetics
    #itxt = _np.vstack([itxt.get_position() for itxt in iax.texts])
    #xlim, ylim = _np.vstack((itxt.min(0), itxt.max(0))).T
    #iax.set_xlim(xlim)
    #iax.set_ylim(ylim)

    res_pairs_descending = []
    if len(array_for_sorting)>0:
        res_pairs_descending = _np.vstack([plotted_respairs[ii] for ii in _np.argsort(array_for_sorting)[::-1]])
    if not return_descending_ctc_freqs:
        return iax, res_pairs_descending
    else:
        return iax, res_pairs_descending, sorted(array_for_sorting)[::-1]

#TODO refactor segments to fragments in all the module's methods
def topology2circle(top, segments=None, **binary_ctcs2flare_kwargs):
    r"""

    :param top:
    :return:
    """
    res_idxs_pairs = _np.vstack(_np.triu_indices(top.n_residues, k=1)).T

    ictcs = _np.zeros(res_idxs_pairs.shape[0])
    ictcs = ictcs.reshape(1,-1)
    if segments is None:
        segments = [[ii for ii in range(top.n_residues)]]

    return binary_ctcs2flare(ictcs, res_idxs_pairs, segments, top=top, **binary_ctcs2flare_kwargs)

def list2circle(ilist,fragments = None, **binary_ctcs2flare_kwargs):
    n_residues = len(ilist)
    res_idxs_pairs = _np.vstack(_np.triu_indices(n_residues, k=1)).T
    ictcs = _np.zeros(res_idxs_pairs.shape[0])
    ictcs = ictcs.reshape(1, -1)
    if fragments is None:
        fragments = [[ii for ii in range(n_residues)]]

    return binary_ctcs2flare(ictcs, res_idxs_pairs, fragments, **binary_ctcs2flare_kwargs)


def create_flare_bezier(nodes, center=None):
    middle = _np.floor(len(nodes) / 2).astype("int")
    if center is not None:
        nodes = _np.vstack((nodes[:middle], center, nodes[middle:]))

    return my_BZCURVE(nodes.T, degree=2)
    #return _bezier.Curve(nodes.T, degree=3)


def _hover(event):
    iax = event.inaxes
    try:
        annot = iax._annot
        iline = iax._iline
    except AttributeError:
        return

    vis = annot.get_visible()
    cont, __ = iline.contains(event)
    if cont:
        annot.set_visible(~vis)
        iax.figure.canvas.draw_idle()

def add_info_point(iax, istr,xy=None, color='gray'):
    if xy is None:
        xy = iax.get_xlim()[0], iax.get_ylim()[1]
    from matplotlib.patches import Circle as _Circle
    icircle = _Circle(xy, .1, color=color)
    iax.add_artist(icircle)#, markersize=20)
    annot = iax.annotate(istr, xy=(xy),
                        bbox=dict(boxstyle="round", fc="w"),
                         #verticalalignment='top',
                         #verticalalignment='bottom',
                         verticalalignment='center',
                         #horizontalalignment='center'
                         )

    annot.get_bbox_patch().set_alpha(0.75)
    annot.set_visible(False)
    iax._annot = annot
    iax._iline = icircle
    iax.figure.canvas.mpl_connect("button_release_event", _hover)


def _metric_to_label(metric, ii, igeom):
    ilabel = "state %u" % ii
    if metric is not None:
        m = metric["lambda"](igeom)
        # istr = $D_{FRET} = %2.1f \pm %2.1f\ \\AA$
        ilabel += "\n$%s = %2.1f \pm %2.1f\ %s$" % (metric["name"].strip("$"),
                                                    m.mean(), m.std(),
                                                    metric["units"].strip("$"))
    return ilabel
def represent(set_geoms, set_ctcs,
              selection=None,
              res_idxs_pairs=None,
              segments=None,
              remove_background=True,
              n_most_freq_ctcs = 1,
              form="vert",
              pi=None,
              metric=None,
              n_overlay=0,
              static=False,
              panelsize=5,
              infopoint=None,
              print_infopoint=False,
              segment_nglwdg_colors=None,
              basins=None,
              **binary_ctcs2flare_kwargs):
    r"""

    :param set_geoms:
    :param set_ctcs:
    :param res_idxs_pairs:
    :param remove_background:defaults to the upper triangular indices
    :return:
    """
    #reload(visualize)

    # Sanity-cchecks:
    #TODO avoid code repetitoon with represent_vicinities
    if res_idxs_pairs is None and segments is None:
        pass
    elif res_idxs_pairs is not None and segments is not None:
        pass
    else:
        raise ValueError("This is a suspicious input, aborting...")

    if selection is None:
        selection = _np.arange(len(set_geoms))

    from .metrics import most_freq_SS_weights
    from textwrap import wrap as _wrap

    if segment_nglwdg_colors is None:
        segment_nglwdg_colors=_mycolors

    if infopoint is not None:
        if isinstance(infopoint, str):
            if _os.path.exists(infopoint):
                state_dict = jsonfile2clusterdesc(infopoint)
            elif infopoint== 'contacts':
                pass
            else:
                raise Exception("Value of jsonfile %s is not accepted" % infopoint)



    toolbar = _rcParams['toolbar']
    _rcParams["toolbar"] = "None"
    iboxes = []
    wdgs = []
    axes = []

    dummy_ctc_mat = _sqf(set_ctcs[0])
    n_residues = dummy_ctc_mat.shape[0]


    if form.lower().startswith("vert"):
        bigbox   = lambda tuple : _VBox(tuple)
        smallbox = lambda tuple : _HBox(tuple)
        tupleorder = [0,1]
    else:
        bigbox   = lambda tuple : _HBox(tuple)
        smallbox = lambda tuple : _VBox(tuple)
        #tupleorder = [1,0] # why does this not work?
        tupleorder = [0,1]

    if res_idxs_pairs is None:
        res_idxs_pairs = _np.vstack(_np.triu_indices(n_residues, m=n_residues,k=1)).T

    if segments is None:
        segments = [[ii for ii in range(n_residues)]]

    if remove_background:
        background = _np.vstack([set_ctcs[ii] for ii in selection]).min(0)
    else:
        background = 0

    if pi is None:
        ipi = _np.ones(len(set_ctcs))
    else:
        ipi = pi

    # Sanity-checks:
    assert _np.allclose(_np.unique(res_idxs_pairs), _np.unique(_np.hstack(segments))), "This is a suspicious input, aborting..."


    for idx, (ii, igeom, ictcs) in enumerate(zip(selection, [set_geoms[ii] for ii in selection],
                                                            [set_ctcs[ii] for ii in selection])):
        _plt.ioff()
        if static:
            if idx==0:
                ncols = 2
                n_background = 0
                if remove_background:
                    n_background = 1
                nrows = _np.ceil((len(selection)+n_background) / 2.).astype(int)
                myfig, myax = _plt.subplots(nrows, ncols, squeeze=False,
                                            sharex=True, sharey=True,  #
                                            figsize=(ncols * panelsize, nrows * panelsize),
                                            )

            iax = myax[_np.unravel_index(idx+n_background, (nrows,ncols))]
        else:
            iax = None

        iax, respairs_descending, ctcs_freq_descending = binary_ctcs2flare(ictcs.reshape(1, -1) - background,
                                                                           res_idxs_pairs,
                                                                           segments,
                                                                           ss_array=_most_freq_SS(igeom)[_np.unique(res_idxs_pairs)],
                                                                           top=igeom.top,
                                                                           iax=iax,
                                                                           return_descending_ctc_freqs=True,
                                                                           **binary_ctcs2flare_kwargs,
                                                                           )

        if basins is not None:
            icol = color_by_basin(ii, basins)
        else:
            icol = "lightblue"
        if infopoint is not None:
            if "state_dict" in locals():
                info_str = ''
                for key, val in state_dict[ii].items():
                    if key != 'desc':
                        #       print(key)
                        info_str += '\n'.join(_wrap('*%5s : %s\n' % (key, val), 30))
                        info_str += '\n\n'
            else:
                info_str = 'ctc frequency:\n'
                for ifreq, ipair in zip(ctcs_freq_descending, respairs_descending[:n_most_freq_ctcs]):
                    r1,r2 = [igeom.top.residue(ip) for ip in ipair]
                    info_str+='%6s-%-6s (%3u%%)\n'%(r1,r2,ifreq*100)

            add_info_point(iax,info_str[:-1], color=icol)
            # check https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib
        else:
            pass
            #iax.plot(iax.get_xlim()[0], iax.get_ylim()[1], 'o',
            #         markersize=50, markerfacecolor=icol)


        iax_wdg = AxesWidget(iax)


        ilabel = _metric_to_label(metric, ii, igeom)

        if pi is not None:
            ilabel += '\n$\pi = %3u %s$' % (ipi[ii] * 100, '\%')

        #iax.text(0,(iax.get_ylim()[0]), ilabel, ha="center",va="top")
        _empty_legend(iax, [ilabel],
                      loc="lower left",
                      #loc='upper right'
                      )
        #iax.text(0, (iax.get_ylim()[0]), ilabel, ha="center", va="top")
        if not static:
            iax.set_frame_on(False)

            if n_overlay>0:
                iwd = molpx.visualize._nglwidget_wrapper(igeom[0])
                [molpx.visualize._nglwidget_wrapper(jgeom, iwd) for jgeom in igeom[1:n_overlay]]
            else:
                iwd = molpx.visualize._nglwidget_wrapper(igeom)
            iwd._set_size(*['%fin' % inches for inches in iax.get_figure().get_size_inches()])
            if len(respairs_descending)>0:
                most_freq_CAs = _res_pairs_2_CA_pairs(respairs_descending, igeom.top).T
                molpx._bmutils.add_atom_idxs_widget(most_freq_CAs[:n_most_freq_ctcs], iwd, color_list=['black'])
            # iwd.add_licorice(igeom.top.select("resid %s"%(' '.join(['%u'%ii for ii in np.hstack(respairs_descending[:n_most_freq])]))))
            toappend = [(iax_wdg.canvas, iwd)[ii] for ii in tupleorder]
            iboxes.append(smallbox(toappend))


            [iwd.remove_cartoon(component=ii) for ii in range(n_overlay+1)]
            [iwd.remove_ball_and_stick(component=ii) for ii in range(n_overlay + 1)]
            # iwd.add_cartoon(np.array(lower_atoms), color='gray')

            for iseg, icol in zip(segments, segment_nglwdg_colors):
                ires = _np.hstack([iseg[0] - 1, iseg, iseg[-1] + 1])
                ires = _np.array([ii for ii in ires if ii>=0])
                for ii in range(n_overlay+1):
                    iwd.add_cartoon(igeom.top.select("resid %s" % (' '.join(['%u' % ii for ii in ires]))), color=icol, component=ii)

            if "highlight_residxs" in binary_ctcs2flare_kwargs.keys():
                from .metrics import  _CA_of_residue_idx
                for iidx in binary_ctcs2flare_kwargs["highlight_residxs"]:
                    for ii in range(n_overlay + 1):
                        iwd.add_spacefill(_np.array(_CA_of_residue_idx(iidx,igeom.top),ndmin=1), color="red",component=ii)
            axes.append(iax)
            wdgs.append(iwd)

        if infopoint is not None and print_infopoint:
            print(ilabel)
            print(info_str)

        iax.set_yticks([])
        iax.set_xticks([])

    # _rcParams["toolbar"] = toolbar
    if remove_background:
        if static:
            iax = myax[_np.unravel_index(0, (nrows, ncols))]
        else:
            iax = None
        #  todo remove code
        iax, respairs_descending, ctcs_freq_descending = binary_ctcs2flare(background.reshape(1, -1),
                                                                           res_idxs_pairs,
                                                                           segments,
                                                                           iax=iax,
                                                                           ss_array=most_freq_SS_weights([set_geoms[ss] for ss in selection],
                                                                                                         ipi[selection])[_np.unique((res_idxs_pairs))],
                                                                           top=igeom.top,
                                                                           return_descending_ctc_freqs=True,
                                                                           **binary_ctcs2flare_kwargs,
                                                                           )
        _empty_legend(iax, ["background"])

        if infopoint is not None:
            if "state_dict" in locals():
                info_str = '\n'.join(_wrap(state_dict[ii]["desc"], 30))
            else:
                info_str = 'ctc frequency:\n'
                for ifreq, ipair in zip(ctcs_freq_descending, respairs_descending[:n_most_freq_ctcs]):
                    r1, r2 = [igeom.top.residue(ip) for ip in ipair]
                    info_str += '%6s-%-6s (%3u%%)\n' % (r1, r2, ifreq * 100)
            add_info_point(iax,info_str)
        iax_wdg = AxesWidget(iax)
        iboxes = [smallbox(tuple([iax_wdg.canvas]))]+iboxes
        # iax.set_yticks([])
        # iax.set_xticks([])
    _plt.ion()
    if static:
        return myfig
    else:
        iax.set_frame_on(False)
        return bigbox(iboxes)

def color_ax(iax, color):
    _plt.setp(iax.spines.values(), color=color)
    _plt.setp([iax.get_xticklines(), iax.get_yticklines()], color=color)

def represent_vicinities(set_geoms, set_ctcs,
                        resSeqs,
                        selection = None,
                        res_idxs_pairs = None,
                        segments = None,
                        n_most_freq_ctcs = 1,
                        metric = None,
                        panelsize = 5,
                         color_bar_by_segment=False,
                         color_ax_by_basin=None,
                        ** binary_ctcs2flare_kwargs):

    panelsize2font = 4
    if res_idxs_pairs is None and segments is None:
        pass
    elif res_idxs_pairs is not None and segments is not None:
        pass
    else:
        raise ValueError("This is a suspicious input, aborting...")

    if selection is None:
        selection = _np.arange(len(set_geoms))

    n_sets = len(selection)

    residxs_input = [[rr.index for rr in set_geoms[0].top.residues if rr.resSeq == jresSeq] for jresSeq in resSeqs]
    assert _np.all([len(rr)==1 for rr in residxs_input for rr in residxs_input])

    residxs_input=_np.array([rr[0] for rr in residxs_input])

    myfig, myax = _plt.subplots(n_sets, len(resSeqs)+1,
                                figsize=((len(resSeqs)+1)*panelsize,n_sets*panelsize)
                                #sharex='col',
                                #sharey=True
    )

    for idx, (ii, igeom, ictcs) in enumerate(zip(selection, [set_geoms[ii] for ii in selection],
                                                 [set_ctcs[ii] for ii in selection])):
        jctcs=_np.copy(ictcs)
        jctcs[~residxs_input]=0
        iax, respairs_descending, ctcs_freq_descending = binary_ctcs2flare(jctcs.reshape(1, -1),
                                                                           res_idxs_pairs,
                                                                           segments,
                                                                           ss_array=_most_freq_SS(igeom)[
                                                                               _np.unique(res_idxs_pairs)],
                                                                           top=igeom.top,
                                                                           iax=myax[idx,0],
                                                                           return_descending_ctc_freqs=True,
                                                                           **binary_ctcs2flare_kwargs,
                                                                           )

        ilabel = _metric_to_label(metric, ii, igeom)
        colortxt='k'
        iax.set_ylabel(ilabel.replace("=", "$\n$"),
                           ha="right", rotation=0, fontsize=panelsize * panelsize2font,
                       color=colortxt)

        if color_ax_by_basin is not None:
            icol = color_by_basin(ii, color_ax_by_basin)
            icircle = _Circle((iax.get_xlim()[0], iax.get_ylim()[1]), .3, color=icol)
            iax.add_artist(icircle)

        _empty_legend(iax, [ilabel])
        for jj, (jresSeq,jresidx) in enumerate(zip(resSeqs,residxs_input)):
            iax = myax[idx, 1+jj]
            only_pairs_with_this_residx =_np.array([ii for ii,ipair in enumerate(respairs_descending) if jresidx in ipair])
            #print(jresidx, jresSeq,only_pairs_with_this_residx)
            xlabels=[]
            for oo in only_pairs_with_this_residx[:n_most_freq_ctcs]:
                ipair = respairs_descending[oo]
                #print(igeom.top.residue(ipair[0]), igeom.top.residue(ipair[1]))
                if jresidx==ipair[0]:
                    xlabels.append('%s'%igeom.top.residue(ipair[1]))
                else:
                    xlabels.append('%s'%igeom.top.residue(ipair[0]))

            iax.set_title('%s'%igeom.top.residue(jresidx), fontsize=panelsize*panelsize2font)
            if len(only_pairs_with_this_residx)>0:
                xvec = _np.arange(len(only_pairs_with_this_residx[:n_most_freq_ctcs]))
                yvec = _np.array(ctcs_freq_descending)[only_pairs_with_this_residx[:n_most_freq_ctcs]]
                patches = iax.bar(xvec, yvec, width=.50,)
                for ix, iy, ilab in zip(xvec, yvec, xlabels):
                    iax.text(ix,iy,ilab,
                             va='bottom',
                             ha='left',
                             rotation=45,
                             fontsize=panelsize*panelsize2font,
                             backgroundcolor="white"
                    )
                isegs=[]
                if color_bar_by_segment is True:
                    assert "segment_names" in binary_ctcs2flare_kwargs.keys()
                    for oo, ipatch in zip(only_pairs_with_this_residx[:n_most_freq_ctcs],
                                           patches.get_children()):
                        ipair = respairs_descending[oo]
                        if jresidx == ipair[0]:
                            idx_col=ipair[1]
                        else:
                            idx_col = ipair[0]
                        iseg=[ii for ii, iseg in enumerate(segments) if idx_col in iseg]

                        assert len(iseg)==1
                        icol = _mycolors[iseg[0]]
                        isegs.append(iseg[0])
                        ipatch.set_color(icol)
                    isegs = _np.unique(isegs)
                    _empty_legend(iax,
                                  [binary_ctcs2flare_kwargs["segment_names"][ii] for ii in isegs],
                                  [_mycolors[ii] for ii in isegs],
                                  'o'*len(isegs),
                                  loc='upper right',
                                  fontsize=panelsize * panelsize2font,
                                  )
            iax.set_ylim([0,1])
            iax.set_xlim([-.5,n_most_freq_ctcs-.5])
            iax.set_xticks([])
            iax.set_yticks([.25,.50,.75,1])
            iax.set_yticklabels([])
            [iax.axhline(ii ,color="k",linestyle="--", zorder=-1) for ii in [.25, .50, .75]]
            if color_ax_by_basin is not None:
                icol = color_by_basin(ii, color_ax_by_basin)
                iax.plot(iax.get_xlim()[0], iax.get_ylim()[1], 'o',
                          markersize=50, markerfacecolor=icol)

        iax2=iax.twinx()
        iax2.set_yticks([.25, .50, .75, 1])
        iax2.set_yticklabels(["%3.2f"%ii for ii in [.25,.50,.75,1]], fontsize=panelsize*panelsize2font)
            #iax.set_xticklabels(xlabels, rotation=45, ha='right')

    myfig.tight_layout()

    return myfig

def color_by_basin(oo, basins, cols={"upper left":"blue",
                                     "bottom":"red",
                                     "middle left":"orange",
                                     'upper right':"green"}):
    for key in basins.keys():
        if oo in basins[key]:
            #print(oo,basins[key])
            return cols[key]


def jsonfile2clusterdesc(jsonfile="description.json"):
    from json import load as _jsonload
    state_dict = {}
    with open(jsonfile,'r') as opened_file:
        desc = _jsonload(opened_file)

    for set_key, set_val in desc.items():
        print(set_key, end=",")
        try:
            set_val["desc"]
        except KeyError:
            set_val["desc"] = ''
        for key, val in set_val.items():
            if key.lower() != 'desc':
                new_key = int(key.replace('_', " ").split()[-1])
                state_dict[new_key] = val
                state_dict[new_key]["desc"] = "IC-2D-plane %s : %s" % (set_key, set_val["desc"])

    return state_dict


def box2fig(ibox,
            length_in_px=2000, dpi=600,
            form='vert',
            fout="box.png",
            exclude_background=False,
            ):
    if exclude_background:
        start=1
    else:
        start=0
    from base64 import b64decode as _b64dcd
    from IPython.display import Image as _Image
    for ii, smallbox in enumerate(ibox.children[start:]):
        fnames = []
        for jj, iwd in enumerate(smallbox.children):
            fname = '%u.%u.png' % (ii, jj)
            #print(iwd)
            print(fname)
            fnames.append(fname)
            if hasattr(iwd, "_image_data"):
                iwd.render_image(trim=True)
                _Image(iwd._image_data)  # ?
                iwd._display_image()
                iwd.display(use_box=True)
                imgdata = _b64dcd(iwd._image_data)
                if len(imgdata)==0:
                    raise Exception("No image data for widget %u"%ii)

                with open(fname, 'wb') as f:
                    f.write(imgdata)

            else:
                ifig = iwd.figure
                ifig.savefig(fname, bbox_inches="tight", dpi=dpi)

        cmd = 'montage -geometry %ux%u %s %u.png' % (length_in_px, length_in_px, ' '.join(fnames), ii)
        print(cmd)
        _call_cmd_with_context_manager(cmd)

    cmd = 'montage -geometry %ux%u+200+200 -tile 1x%u -trim %s %s.tmp.png'%((ii+1) * length_in_px, length_in_px, ii+1, ' '.join(['%s.png' % jj for jj in range(ii + 1)]), fout)
    print(cmd)
    _call_cmd_with_context_manager(cmd)
    cmd = 'convert %s.tmp.png -trim %s'%(fout, fout)
    print(cmd)
    _call_cmd_with_context_manager(cmd)
    cmd = "rm %s.tmp.png"%(fout)
    print(cmd)
    _call_cmd_with_context_manager(cmd)

from subprocess import Popen as _Popen
def _call_cmd_with_context_manager(incmd):
    with _Popen(incmd.split()) as cmd:
        cmd

def stacked_bar_plot(indict,
                     fragments=None,
                     names=None,
                     iax=None,
                     padding=1,
                     alpha=.75,
                     w2hratio=2,
                     panelsize=5,
                     names_y_pos=1.05,
                     label_stride=2,
                     top=None,
                     xtick_rotation=45):

    n = _np.unique([len(val) for val in indict.values()])
    assert len(n) == 1
    n = n[0]

    padding += 1

    if iax is None:
        _plt.figure(figsize=(panelsize, panelsize/w2hratio))
        iax = _plt.gca()

    if fragments is None:
        fragments = [_np.arange(n)]

    colors = _plt.rcParams['axes.prop_cycle'].by_key()["color"]

    x_offset = 0
    label_vec = []
    label_pos_vec = []
    for ff, ifrag in enumerate(fragments):
        if ff==0:
            ipadding=0
        else:
            ipadding=padding

        ix = _np.arange(len(ifrag)) + x_offset + ipadding

        label_vec += ifrag[::label_stride]
        label_pos_vec += [ii for ii in ix[::label_stride]]

        offset = _np.zeros(n, dtype=_np.float)
        current_handles = []
        for ii, (key, val) in enumerate(indict.items()):
            ibar = iax.bar(ix, val[ifrag],
                    bottom=offset[ifrag],
                    width=1,
                    alpha=alpha,
                    color=colors[ii])
            x_offset = ix[-1]
            offset += val
            current_handles.append(ibar)

        if names is not None:
            iax.text(ix.mean(), names_y_pos, names[ff], ha="center")

    # Cosmetics
    [ibar.set_label(key) for key, ibar in zip(indict.keys(),current_handles)]
    iax.legend()
    _plt.gcf().tight_layout()
    if top is not None:
        label_vec = [str(top.residue(ii)) for ii in label_vec]
    _plt.xticks(label_pos_vec, label_vec, rotation=xtick_rotation,
                #ha="right"
                )
    return _plt.gcf(),_plt.gca()

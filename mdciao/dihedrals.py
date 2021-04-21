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
import mdtraj as _md
import matplotlib.pyplot as _plt
from matplotlib import rcParams as _rcParams

def xtcs2dihs(xtcs, top, dih_idxs, stride=1, consolidate=True,
              chunksize=1000, return_time=False,
              n_jobs=1,
              progressbar=False,
              ):
    """Returns the time-dependent traces of sidechain angles

    Parameters
    ----------
    xtcs : list of strings
        list of filenames with trajectory data. Typically xtcs,
        but can be any type of file readable by :obj:mdtraj
    top : str or :py:class:`mdtraj.Topology`
        Topology that matches :obj:xtcs
    dih_idxs : iterable
        List of quadroplets with atom idxs (zero-indexed) of the dihedrals
    stride : int, default is 1
        Stride the trajectory data down by this value
    consolidate : boolean, default is True
        Return the time-traces consolidated
        into one array by using np.vstack
    chunksize : integer, default is 1000
        How many frames will be read into memory for
        computation of the contact time-traces. The higher the number,
        the higher the memory requirements
    n_jobs : int, default is 1
        to how many processors to parallellize

    return_time : boolean, default is False
        Return also the time array in ps

    Returns
    -------
    chis, or
    chis, time_trajs if return_time=True

    """

    from tqdm import tqdm

    if progressbar:
        iterfunct = lambda a : tqdm(a)
    else:
        iterfunct = lambda a : a
    ictcs_itimes = _Parallel(n_jobs=n_jobs)(_delayed(per_xtc_dihs)(top, ixtc, dih_idxs,chunksize,stride,ii,
                                                                  )
                                            for ii, ixtc in enumerate(iterfunct(xtcs)))
    dihs = []
    times = []
    for idihs, itimes in ictcs_itimes:
        dihs.append(idihs)
        times.append(itimes)

    if consolidate:
        try:
            adihs = _np.vstack(dihs)
            times = _np.hstack(times)
        except ValueError as e:
            print(e)
            print([_np.shape(ic) for ic in dihs])
            raise
    else:
        adihs = dihs
        times = times

    if not return_time:
        return adihs
    else:
        return adihs, times

def per_xtc_dihs(top, ixtc, dih_idxs, chunksize, stride,
                traj_idx,
                **mddih_kwargs):

    iterate, inform = iterate_and_inform_lambdas(ixtc,stride, chunksize, top=top)
    idihs = []
    running_f = 0
    inform(ixtc, traj_idx, 0, running_f)
    itime = []
    for jj, igeom in enumerate(iterate(ixtc)):
        running_f += igeom.n_frames
        inform(ixtc, traj_idx, jj, running_f)
        itime.append(igeom.time)
        jdihs = _md.compute_dihedrals(igeom, dih_idxs, **mddih_kwargs)


        idihs.append(jdihs)

    itime = _np.hstack(itime)
    idihs = _np.vstack(idihs)

    return idihs, itime

def plot_dih(ictc, iax,
                 color_scheme=None,
                 ctc_cutoff_Ang=0,
                 n_smooth_hw=0,
                 dt=1,
                 gray_background=False,
                 shorten_AAs=False,
                 t_unit='ps',
                 max_handles_per_row=4,
               ylabel='dih / deg',
             lambda_ang=None
                 ):
    if color_scheme is None:
        color_scheme = _rcParams['axes.prop_cycle'].by_key()["color"]
    if lambda_ang is None:
        lambda_ang = lambda x : x
    color_scheme = _np.tile(color_scheme, _np.ceil(ictc.n_trajs/len(color_scheme)).astype(int)+1)
    iax.set_ylabel(ylabel, rotation=90)

    for traj_idx, (ictc_traj, itime, trjlabel) in enumerate(zip(ictc.feat_trajs,
                                                                ictc.time_trajs,
                                                                ictc.trajstrs)):

        ilabel = '%s'%trjlabel

        #print(ictc_traj*input2Ang)
        plot_w_smoothing_auto(iax, itime * dt, lambda_ang(ictc_traj),
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

    iax.text(_np.mean(iax.get_xlim()),
             iax.get_ylim()[0]+_np.abs(_np.diff(iax.get_ylim()))*.1,
             ctc_label,
             ha='center')

    iax.set_xlabel('t / %s' % _replace4latex(t_unit))
    iax.set_xlim([0, ictc.time_max * dt])
    #iax.set_ylim([0,iax.get_ylim()[1]])

class angle(object):
    r"""Class for storing everything related to a contact"""
    #todo consider packing some of this stuff in the site_obj class
    def __init__(self, atom_idx_quad,
                 dih_trajs,
                 time_arrays,
                 top=None,
                 trajs=None,
                 res_idx=None,
                 ang_type=None,
                 fragment_idx=None,
                 fragment_name=None,
                 fragment_color=None,
                 consensus_label=None):
        """

        Parameters
        ----------
        atom_idx_quad : iterable for 4 atom indices for this dihedral.
        dih_trajs : list of list, the code converts it into a list of array
        time_arrays : list of list,
        top : :py:class:`mdtraj.Topology`
        trajs:
        fragment_idx :
        fragment_name :
        fragment_color :
        anchor_residue_idx :
        consensus_label :
        """

        self.atom_idx_quad = atom_idx_quad
        self._res_idx = res_idx
        self._ang_type = ang_type
        self._dih_trajs = [_np.array(itraj) for itraj in dih_trajs]
        self._top = top
        self._trajs = trajs

        self._time_arrays = time_arrays
        self._n_trajs = len(dih_trajs)
        assert self._n_trajs == len(time_arrays)
        assert all([len(itraj) == len(itime) for itraj, itime in zip(dih_trajs, time_arrays)])
        self._time_max = _np.max(_np.hstack(time_arrays))

        self._consensus_label = consensus_label
        self._fragment_idx  = fragment_idx
        if fragment_name is None:
            # assert self.idxs is not None
            # self._fragment_names = self._fragment_idxs

            if self.fragment_idx is not None:
                self._fragment_name = self._fragment_idx
            else:
                self._fragment_name = None
        else:
            self._fragment_name = fragment_name
        self._fragment_color = fragment_color

    @property
    def fragment_name(self):
        """

        Returns
        -------
        list of list, Fragment names if passed, else fragment idxs. If both are not available then None(default)

        """
        return self._fragment_name

    @property
    def fragment_idx(self):
        """

        Returns
        -------
        list of list, Fragment idxs if passed, else None(default)

        """
        return self._fragment_idx

    @property
    def time_max(self):
        """

        Returns
        -------
        int or float, maximum time from list of list of time

        """
        return self._time_max

    @property
    def trajlabels(self):
        """

        Returns
        -------
        list, list of labels for each trajectory if passed.
        If labels are not passed then labels like 'traj 0','traj 1' and so on are assigned

        """
        if self.trajs is None:
            return ['traj %u'%ii for ii in range(self.n_trajs)]
        else:
            return self.trajs

    @property
    def label(self):
        return self.dih_label_short_latex

    @property
    def n_trajs(self):
        """

        Returns
        -------
        int, total number of trajectories that were passed.

        """
        return self._n_trajs

    @property
    def n_frames(self):
        """

        Returns
        -------
        list, list of frames in each trajectory.

        """
        return [len(itraj) for itraj in self.dih_trajs]



    @property
    def res_idx(self):
        """

        Returns
        -------
        list of residue index pair passed

        """
        return self._res_idx

    @property
    def residue_name(self):
        """

        Returns
        -------
        list, for each residue index in the residue contact pair, the corresponding residue name from the topology file.
        example : ['GLU30','VAL212']

        """
        if self.res_idx is not None:
            return str(self.topology.residue(self.res_idx))
        else:
            return None

    @property
    def residue_name_short(self):
        """

        Returns
        -------
        list, for each residue name in the residue contact pair, the corresponding short residue name from the topology file.
        example : ['E30', 'V212']

        """
        from .residue_and_atom_utils import shorten_AA as _shorten_AA
        if self.res_idx is not None:
            return _shorten_AA(self.residue_name, substitute_fail="long", keep_index=True)
        else:
            return None

    @property
    def feat_trajs(self):
        return self.dih_trajs

    @property
    def ang_type(self):
        return self._ang_type

    @property
    def dih_label(self):
        """

        Returns
        -------
        str,

        """
        dih_label = '[%s]'%('-'.join([str(self.top.atom(ii)) for ii in self.atom_idx_quad]))

        if self.ang_type is not None:
            dih_label = '%s %s'%(self.ang_type,dih_label)

        return dih_label


    @property
    def dih_label_short(self):
        """

        Returns
        -------
        str,

        """

        if self.ang_type is not None:
            return self.ang_type
        else:
            return self.dih_label



    @property
    def time_arrays(self):
        """

        Returns
        -------

        """
        return self._time_arrays

    @property
    def dih_trajs(self):
        """

        Returns
        -------

        """
        return self._dih_trajs

    @property
    def trajs(self):
        """

        Returns
        -------

        """
        return self._trajs

    @property
    def fragment_color(self):
        """

        Returns
        -------

        """
        return self._fragment_color

    @property
    def top(self):
        """

        Returns
        -------

        """
        return self._top

    @property
    def topology(self):
        """

        Returns
        -------

        """
        return self._top

    @property
    def consensus_label(self):
        """

        Returns
        -------

        """
        return self._consensus_label

    @property
    def dih_label_short_latex(self):
        if self.ang_type is not None:
            if self.ang_type.lower().startswith("chi"):
                return '$\\chi_{%s}$'%(self.ang_type[-1])
            else:
                return '$\\%s$'%(self.ang_type)
        else:
            return None

    def __str__(self):
        out = "Angle object for angle defined by atoms"
        out += '\n%s'%self.atom_idx_quad
        #out += "\n%s"%self.res_idx
        out += "\nFor %u trajectories"%self.n_trajs
        for var in dir(self):
            if not var.startswith("_"):
                out += '\n%s: %s'%(var, getattr(self,'%s'%var))
        return out

def plot_ramachandran(phi, psi, iax=None, panelsize=5, bins=72):
        if iax is None:
            _plt.figure(figsize=(panelsize,panelsize))
            iax=_plt.gca()
            h, x, y = _np.histogram2d(_np.hstack((phi,psi)))

def residue_dihedrals(topology, trajectories, resSeq_idxs,
                      res_idxs=False,
                      stride=1,
                      chunksize_in_frames=10000,
                      n_smooth_hw=0,
                      ask=True,
                      sort=True,
                      fragmentify=True,
                      fragment_names="",
                      graphic_ext=".pdf",
                      output_ascii=None,
                      BW_uniprot="None",
                      CGN_PDB="None",
                      output_dir='.',
                      color_by_fragment=True,
                      output_desc='dih',
                      t_unit='ns',
                      curve_color="auto",
                      gray_background=False,
                      graphic_dpi=150,
                      short_AA_names=False,
                      write_to_disk_BW=False,
                      plot_timedep=True,
                      n_cols=4,
                      types='all',
                      n_jobs=1,
                      use_deg=True,
                      use_cos=False,
                      ):
    ang2plotfac = 1
    xlabel, ang_u = 'dih', 'rad'
    bins = 72
    ang_lambda = lambda x : x
    xlim =_np.array([-_np.pi, +_np.pi])
    if use_deg:
        ang_lambda = lambda x : (180 / _np.pi) * x
        ang_u = 'deg'
        xlim = ang_lambda(xlim)
    if use_cos:
        ang_lambda = lambda x : _np.cos(x)
        ang_u = 'arb. u.'
        xlabel = '$cos(dih)$'
        bins = 100
        xlim = [-1,1]

    if resSeq_idxs is None:
        print("You have to provide some residue indices via the --resSeq_idxs option")
        return None

    dt = _t_unit2dt(t_unit)

    _offer_to_create_dir(output_dir)

    # String comparison to allow for command line argparse-use directly
    if str(output_ascii).lower() != 'none' and str(output_ascii).lower().strip(".") in ["dat", "txt", "xlsx"]:
        ascii_ext = str(output_ascii).lower().strip(".")
    else:
        ascii_ext = None


    _resSeq_idxs = rangeexpand(resSeq_idxs)
    if len(_resSeq_idxs) == 0:
        raise ValueError("Please check your input indices, "
                         "they do not make sense %s" % resSeq_idxs)
    else:
        resSeq_idxs = _resSeq_idxs
    if sort:
        resSeq_idxs = sorted(resSeq_idxs)

    xtcs = sorted(trajectories)
    try:
        print("Will compute contact frequencies for the files:\n"
              "%s\n with a stride of %u frames.\n" % (
            "\n  ".join(xtcs), stride))
    except:
        pass

    if isinstance(topology, str):
        refgeom = _md.load(topology)
    else:
        refgeom = topology
    if fragmentify:
        fragments = get_fragments(refgeom.top,method='bonds')
    else:
        raise NotImplementedError("This feature is not yet implemented")

    fragment_names = _parse_fragment_naming_options(fragment_names, fragments, refgeom.top)

    # Do we want BW definitions
    BW = _parse_consensus_option(BW_uniprot, 'BW', refgeom.top, fragments, write_to_disk=write_to_disk_BW)

    # Dow we want CGN definitions:
    CGN = _parse_consensus_option(CGN_PDB, 'CGN', refgeom.top, fragments)

    # TODO find a consistent way for coloring fragments
    fragcolors = [cc for cc in mycolors]
    fragcolors.extend(fragcolors)
    fragcolors.extend(fragcolors)
    if isinstance(color_by_fragment, bool) and not color_by_fragment:
        fragcolors = ['blue' for cc in fragcolors]
    elif isinstance(color_by_fragment, str):
        fragcolors = [color_by_fragment for cc in fragcolors]

    if res_idxs:
        resSeq2residxs = {refgeom.top.residue(ii).resSeq: ii for ii in resSeq_idxs}
        print("\nInterpreting input indices as zero-indexed residue indexes")
    else:
        resSeq2residxs, _ = _interactive_fragment_picker_by_resSeq(resSeq_idxs, fragments, refgeom.top,
                                                                  pick_first_fragment_by_default=not ask,
                                                                  additional_naming_dicts={"BW":BW,"CGN":CGN}
                                                                  )

    print('%10s  %10s  %10s  %10s %10s %10s' % tuple(("residue  residx fragment  resSeq BW  CGN".split())))
    for key, val in resSeq2residxs.items():
        print('%10s  %10u  %10u %10u %10s %10s' % (refgeom.top.residue(val), val, in_what_fragment(val, fragments),
                                                  key,
                                                  BW[val], CGN[val]))

    quad_dict_by_res_idxs = _dih_idxs_for_residue(resSeq2residxs.values(),refgeom)
    if types.lower()=='backbone':
        quad_dict_by_res_idxs={key:{key2:val2 for key2, val2 in val.items() if key2.lower() in ["phi","psi"]} for key, val in quad_dict_by_res_idxs.items()}
    elif types.lower()=='sidechain':
        quad_dict_by_res_idxs = {key: {key2: val2 for key2, val2 in val.items() if key2.lower() not in ["phi", "psi"]} for
                                 key, val in quad_dict_by_res_idxs.items()}
    elif types.lower()=='all':
        pass
    else:
        raise ValueError(types)
    dih_idxs =_np.vstack([[val2 for val2 in val.values()] for val in quad_dict_by_res_idxs.values()])
    dih_trajs, time_array = xtcs2dihs(xtcs, refgeom.top, dih_idxs , stride=stride,
                                       chunksize=chunksize_in_frames, return_time=True,
                                       consolidate=False,
                                       n_jobs=n_jobs,

                                       )
    print()
    # Create per-residue dicts with angle objects
    angles = {}
    idx_iter = iter(range(len(dih_idxs)))
    for res_idx, quad_dict in quad_dict_by_res_idxs.items():
        angles[res_idx] = []
        for ang_type, iquad in quad_dict.items():
            consensus_label = _choose_between_consensus_dicts(res_idx, [BW, CGN])
            fragment_idx =    in_what_fragment(res_idx, fragments)
            idx = next(idx_iter)
            angles[res_idx].append(angle(iquad,
                                         [itraj[:, idx] for itraj in dih_trajs],
                                         time_array,
                                        res_idx=res_idx,
                                         ang_type=ang_type,
                                         top=refgeom.top,
                                         consensus_label=consensus_label,
                                         trajs=xtcs,
                                         fragment_idx=fragment_idx,
                                         fragment_name=fragment_names[fragment_idx],
                                         fragment_color=fragcolors[fragment_idx],
                                         ))

    panelheight = 3
    n_cols =_np.min((n_cols, len(resSeq2residxs)))
    n_rows =_np.ceil(len(resSeq2residxs) / n_cols).astype(int)
    panelsize = 4
    panelsize2font = 3.5
    histofig, histoax = _plt.subplots(n_rows, n_cols,
                                      sharex=True,
                                      sharey=True,
                                      figsize=(n_cols * panelsize * 2, n_rows * panelsize), squeeze=False)

    # One loop for the histograms
    _rcParams["font.size"]=panelsize*panelsize2font
    for jax, res_idx in zip(histoax.flatten(),
                            angles.keys()):
        for idih in angles[res_idx]:
            h, x = _np.histogram(ang_lambda(_np.hstack(idih.dih_trajs)),
                                 bins=bins)
            jax.plot(x[:-1]*ang2plotfac,h,label=idih.dih_label_short_latex)
            jax.legend()
        jax.set_title(idih.residue_name)
    jax.set_xlim(xlim)
    [jax.set_xlabel('%s / %s'%(xlabel, ang_u)) for jax in histoax[-1]]
    histofig.tight_layout(h_pad=2, w_pad=0, pad=0)
    fname = "%s.overall.%s" % (output_desc.strip("."), graphic_ext.strip("."))
    fname = _path.join(output_dir, fname)
    histofig.savefig(fname, dpi=graphic_dpi)
    print("The following files have been created:")
    print(fname)

    # One loop for the time resolved neighborhoods
    if plot_timedep:
        for res_idx, res_dihs in angles.items():
            fname = '%s.%s.time_resolved.%s' % (output_desc.strip("."),
                                                idih.residue_name,
                                                graphic_ext.strip("."))
            fname = _path.join(output_dir, fname)
            n_rows=len(res_dihs)
            myfig, myax = _plt.subplots(n_rows, 1,
                                        figsize=(10, n_rows * panelheight),
                                        squeeze=False)
            myax = myax[:, 0]
            for iax, idih in zip(myax, res_dihs):

                # Plot individual angles
                from .contacts import plot_dih
                plot_dih(idih, iax,
                         color_scheme=_my_color_schemes(curve_color),
                         n_smooth_hw=n_smooth_hw,
                         dt=dt,
                         t_unit=t_unit,
                         gray_background=gray_background,
                         #shorten_AAs=short_AA_names,
                         lambda_ang=ang_lambda,
                             )

            # One title for all axes on top
            title = idih.residue_name
            if short_AA_names:
                title = idih.residue_name
            myfig.axes[0].set_title(title)

            myfig.tight_layout(h_pad=0,w_pad=0,pad=0)
            myfig.savefig(fname, bbox_inches="tight", dpi=graphic_dpi)
            _plt.close(myfig)
            print(fname)
            #ihood.save_trajs(output_desc,ascii_ext,output_dir, dt=dt,t_unit=t_unit, verbose=True)
            print()

    return None

def _get_dih_idxs(geom):
    methods = {
        "phi" : 'compute_phi',
        "psi":  'compute_psi',
        "chi1": 'compute_chi1',
        "chi2": 'compute_chi2',
        "chi3": 'compute_chi3',
        "chi4": 'compute_chi4',
        "chi5": 'compute_chi5',
    }
    idxs = {}
    for key, imeth in methods.items():
        idxs[key] = getattr(_md, imeth)(geom[0])[0]
    return idxs

def _angle_quadruplet2residx(quadruplets, top, max_residues_per_quad=1):
    r"""

    :param quadruplets:
    :param top:
    :param max_residues_per_quad:
    :return:
    """
    allow_only_complimentary = 4 - _np.arange(max_residues_per_quad)
    rowidx2residx = []
    for row in quadruplets:
        res_row = _np.array([top.atom(ii).residue.index for ii in row])

        # Because the bincount is done on integers representing the
        # true residue index as they appears in the topology,
        # these idxs are true idxs and robust towards omission
        # (e.g. the residue in question did not have computable angles)
        bc = _np.bincount(res_row)
        if _np.max(bc) in allow_only_complimentary:
            rowidx2residx.append(_np.argmax(bc))
        else:
            raise ValueError("Cannot assign quad %s (res_idxs %s) when only %u residues are allowed per quad"%(row,res_row,max_residues_per_quad))
            #rowidx2residx.append(None)

    return _np.array(rowidx2residx)

def _dih_idxs_for_residue(res_idxs, geom):
    quads = _get_dih_idxs(geom)
    quad_idxs_2_res_idxs = {key:_angle_quadruplet2residx(val, geom.top, max_residues_per_quad=2) for key, val in quads.items()}
    dict_out = {}
    for ii in res_idxs:
        quad_idx = {key:_np.argwhere(val==ii).squeeze() for key, val in quad_idxs_2_res_idxs.items()}
        dict_out[ii] = {key:quads[key][val] for key, val in quad_idx.items() if _np.size(val)>0}
    return dict_out


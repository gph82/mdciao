import numpy as _np
import mdtraj as _md
import matplotlib.pyplot as _plt
from matplotlib import rcParams as _rcParams

from joblib import Parallel as _Parallel, delayed as _delayed

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
    chis, time_arrays if return_time=True

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
                                                                ictc.time_arrays,
                                                                ictc.trajlabels)):

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
            # assert self.fragment_idxs is not None
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
        from .aa_utils import shorten_AA as _shorten_AA
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


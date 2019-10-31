import numpy as _np
import mdtraj as _md
from os import path as _path
from .list_utils import in_what_fragment, re_warp

import matplotlib.pyplot as _plt
from matplotlib import rcParams as _rcParams
from pandas import DataFrame as _DF


def ctc_freq_reporter_by_residue_neighborhood(ctc_freqs, resSeq2residxs, fragments,
                                              residxs_pairs, top,
                                              n_ctcs=5, restrict_to_resSeq=None,
                                              interactive=False,
                                              ):
    """Prints a formatted summary of contact frequencies
       Returns a residue-index keyed dictionary containing the indices
       of :obj:`residxs_pairs` relevant for this residue.



    Parameters
    ----------
    ctc_freqs: iterable of floats
        Contact frequencies between 0 and 1
    resSeq2residxs: dictionary
        Dictionary mapping residue sequence numbers (resSeq) to residue idxs
    fragments: iterable of integers
        Fragments of the topology defined as list of non-overlapping residue indices
    residxs_pairs: iterable of integer pairs
        The residue pairs for which the contact frequencies in :obj:`ctc_freqs`
        were computed.
    top : :py:class:`mdtraj.Topology`
    n_ctcs : integer, default is 5
        Number of contacts to report per residue.
    restrict_to_resSeq: int, default is None
        Produce the report only for the residue with this resSeq index. Default
        behaviour is to produce for all residues in :obj:`resSeq2residxs`
    interactive : boolean, default is False
        After reporting each neighborhood up to :obj:`n_ctcs` partners,
        ask the user how many should be kept

    Returns
    -------
    neighborhood : dictionary
       neighborhood[300] = [100,200,201,208,500,501]
       means that pairs :obj:`residxs_pairs[100]`,...
       are the most frequent formed contacts for residue 300
       (up to n_ctcs or less, see option 'interactive')
    """
    order = _np.argsort(ctc_freqs)[::-1]
    assert len(ctc_freqs) == len(residxs_pairs)
    neighborhood = {}
    if restrict_to_resSeq is None:
        restrict_to_resSeq = list(resSeq2residxs.keys())
    elif isinstance(restrict_to_resSeq, int):
        restrict_to_resSeq = [restrict_to_resSeq]
    for key, val in resSeq2residxs.items():
        if key in restrict_to_resSeq:
            order_mask = _np.array([ii for ii in order if val in residxs_pairs[ii]])
            print("#idx    Freq  contact             segA-segB residxA   residxB   ctc_idx")

            isum = 0
            seen_ctcs = []
            for ii, oo in enumerate(order_mask[:n_ctcs]):
                pair = residxs_pairs[oo]
                if pair[0] != val and pair[1] == val:
                    pair = pair[::-1]
                elif pair[0] == val and pair[1] != val:
                    pass
                else:
                    print(pair)
                    raise Exception
                idx1 = pair[0]
                idx2 = pair[1]
                s1 = in_what_fragment(idx1, fragments)
                s2 = in_what_fragment(idx2, fragments)
                imean = ctc_freqs[oo]
                isum += imean
                seen_ctcs.append(imean)
                print("%-6s %3.2f %8s-%-8s    %5u-%-5u %7u %7u %7u %3.2f" % (
                 '%u:' % (ii + 1), imean, top.residue(idx1), top.residue(idx2), s1, s2, idx1, idx2, oo, isum))
            if interactive:
                try:
                    answer = input("How many do you want to keep (Hit enter for None)?\n")
                except KeyboardInterrupt:
                    break
                if len(answer) == 0:
                    pass
                else:
                    answer = _np.arange(_np.min((int(answer), n_ctcs)))
                    neighborhood[val] = order_mask[answer]
            else:
                seen_ctcs = _np.array(seen_ctcs)
                n_nonzeroes = (seen_ctcs > 0).astype(int).sum()
                answer = _np.arange(_np.min((n_nonzeroes, n_ctcs)))
                neighborhood[val] = order_mask[answer]
    # TODO think about what's best to return here
    # TODO think about making a pandas dataframe with all the above info
    return neighborhood



def xtcs2ctcs(xtcs, top, ctc_residxs_pairs, stride=1, consolidate=True,
              chunksize=1000, return_time=False,**mdcontacts_kwargs):
    """Returns the time-dependent traces of residue-residue contacts from a list of trajectory files

    Parameters
    ----------
    xtcs : list of strings
        list of filenames with trajectory data. Typically xtcs,
        but can be any type of file readable by :obj:mdtraj
    top : str or :py:class:`mdtraj.Topology`
        Topology that matches :obj:xtcs
    ctc_residxs_pairs : iterable
        List of (zero-indexed) residue pairs
    stride : int, default is 1
        Stride the trajectory data down by this value
    consolidate : boolean, default is True
        Return the time-traces consolidated
        into one array by using np.vstack
    chunksize : integer, default is 1000
        How many frames will be read into memory for
        computation of the contact time-traces. The higher the number,
        the higher the memory requirements
    return_time : boolean, default is False
        Return also the time array in ps

    Returns
    -------
    ctcs, or
    ctcs, time_arrays if return_time=True

    """
    ctcs = []
    times = []

    if isinstance(xtcs[0],_md.Trajectory):
        iterate = lambda ixtc : [ixtc[idxs] for idxs in re_warp(_np.arange(ixtc.n_frames)[::stride],chunksize)]
        inform = lambda ixtc, ii, running_f: print("Analysing a trajectory object in chunks of "
                                                   "%3u frames. chunks %4u frames %8u"%
                                                   (chunksize, ii, running_f), end="\r", flush=True)
    else:
        iterate = lambda ixtc: _md.iterload(ixtc, top=top, stride=stride, chunk=_np.round(chunksize / stride))
        inform = lambda ixtc, ii, running_f: print("Analysing %20s in chunks of "
                                                   "%3u frames. chunks %4u frames %8u" %
                                                   (ixtc, chunksize, ii, running_f), end="\r", flush=True)

    for ii, ixtc in enumerate(xtcs):
        ictcs = []
        running_f = 0
        inform(ixtc, 0, running_f)
        itime = []
        for jj, igeom in enumerate(iterate(ixtc)):
            running_f += igeom.n_frames
            inform(ixtc, jj, running_f)
            itime.append(igeom.time)
            jctcs, jidx_pairs = _md.compute_contacts(igeom, ctc_residxs_pairs,**mdcontacts_kwargs)
            # TODO do proper list comparison and do it only once
            assert len(jidx_pairs)==len(ctc_residxs_pairs)
            ictcs.append(jctcs)
            # if jj==10:
            #    break

        times.append(_np.hstack(itime))
        ictcs = _np.vstack(ictcs)
        # print("\n", ii, ictcs.shape, "shape ictcs")
        ctcs.append(ictcs)
        print()

    if consolidate:
        try:
            actcs = _np.vstack(ctcs)
            times = _np.hstack(times)
        except ValueError as e:
            print(e)
            print([_np.shape(ic) for ic in ctcs])
            raise
    else:
        actcs = ctcs
        times = times

    if not return_time:
        return actcs
    else:
        return actcs, times


def contact_matrix(trajectories, cutoff_Ang=3,
                      n_frames_per_traj=20, **mdcontacts_kwargs):
    r"""
    Return a matrix with the contact frequency for **all** possible contacts
    over all available frames
    Parameters
    ----------
    trajectories: list of obj:`mdtraj.Trajectory`
    n_frames_per_traj: int, default is 20
        Stride the trajectories so that, on average, this number of frames
        is used to compute the contacts
    mdcontacts_kwargs

    Returns
    -------
    ctc_freq : square 2D np.ndarray

    """

    top = trajectories[0].top
    n_res = top.n_residues
    mat = _np.zeros((n_res, n_res))
    ctc_idxs = _np.vstack(_np.triu_indices_from(mat, k=0)).T

    stride=_np.ceil(_np.sum([itraj.n_frames for itraj in trajectories])/n_frames_per_traj).astype(int)

    actcs = xtcs2ctcs(trajectories, top, ctc_idxs, stride=stride, chunksize=50,
                      consolidate=True, ignore_nonprotein=False, **mdcontacts_kwargs)

    actcs = (actcs <= cutoff_Ang/10).mean(0)
    assert len(actcs)==len(ctc_idxs)
    non_zero_idxs = _np.argwhere(actcs>0).squeeze()

    for idx in non_zero_idxs:
        ii, jj = ctc_idxs[idx]

        mat[ii][jj]=actcs[idx]
        if ii!=jj:
            mat[jj][ii] = actcs[idx]

    return mat


from .actor_utils import _replace4latex


def pick_best_label(fallback, test, exclude=[None, "None", "NA", "na"]):
    if test not in exclude:
        return test
    else:
        return fallback

class contact_group(object):
    r"""Class for containing contact objects, ideally
    it can be used for vicinities, sites, interfaces etc"""

    def __init__(self, list_of_contact_objects,
                 top=None):

        self._contacts = list_of_contact_objects
        self._n_ctcs  = len(list_of_contact_objects)

        if top is None:
            self._top = self._unique_topology_from_ctcs()
        else:
            assert top is self._unique_topology_from_ctcs()
            self._top = top

        # Sanity checks about having grouped this contacts together

        # All contacts have the same number of trajs
        self._n_trajs =_np.unique([ictc.n_trajs for ictc in self._contacts])
        assert len(self._n_trajs)==1
        self._n_trajs=self._n_trajs[0]

        # All trajs have the same times
        ref_ctc = self._contacts[0]
        assert all([_np.allclose(ref_ctc.n_frames, ictc.n_frames) for ictc in self._contacts[1:]])
        self._time_arrays=ref_ctc.time_arrays
        self._time_max = ref_ctc.time_max
        self._n_frames = ref_ctc.n_frames

        # All contatcs have the same trajlabels
        for ictc in self._contacts[1:]:
            # Todo this will fail if tlab,rlab are not strings?
            assert all([rlab == tlab for rlab, tlab in zip(ref_ctc.trajlabels, ictc.trajlabels)])

        self._trajlabels = ref_ctc.trajlabels

    @property
    def n_frames(self):
        return self._n_frames

    @property
    def time_max(self):
        return self._time_max

    @property
    def trajlabels(self):
        return self._trajlabels

    @property
    def time_arrays(self):
        return self._time_arrays

    def _unique_topology_from_ctcs(self):
        if all([ictc.top is None for ictc in self._contacts]):
            return None

        top = _np.unique([ictc.top.__hash__() for ictc in self._contacts])
        if len(top)==1:
            return self._contacts[0].top
        else:
            raise ValueError("All contacts in a group of contacts"
                             " should have the same topology, but %s"%top)

    @property
    def n_trajs(self):
        return self._n_trajs

    @property
    def n_ctcs(self):
        return self._n_ctcs

    @property
    def shared_anchor_residue(self):
        r"""
        Returns none if no anchor residue is found
        """
        shared = _np.unique([ictc.anchor_residue_index for ictc in self._contacts])
        if len(shared)==1:
            return shared[0]
        else:
            return None

    # TODO many of these things could be re-factored into the contact object!!!
    @property
    def anchor_res_and_fragment_str(self):
        assert self.shared_anchor_residue is not None
        return self._contacts[0].anchor_res_and_fragment_str

    @property
    def anchor_res_and_fragment_str_short(self):
        assert self.shared_anchor_residue is not None
        return self._contacts[0].anchor_res_and_fragment_str_short


    @property
    def partner_res_and_fragment_labels(self):
        assert self.shared_anchor_residue is not None
        return [ictc.partner_res_and_fragment_str for ictc in self._contacts]

    @property
    def partner_res_and_fragment_labels_short(self):
        assert self.shared_anchor_residue is not None
        return [ictc.partner_res_and_fragment_str_short for ictc in self._contacts]

    @property
    def anchor_fragment_color(self):
        return self._contacts[0].fragment_colors[self._contacts[0].anchor_index]

    @property
    def top(self):
        return self._top

    @property
    def topology(self):
        return self._top

    # todo expose this?
    def _all_freqs(self, ctc_cutoff_Ang):
        return [ictc.frequency_overall(ctc_cutoff_Ang) for ictc in self._contacts]

    def timedep_n_ctcs(self, ctc_cutoff_Ang):
        bintrajs = [ictc.binarize_trajs(ctc_cutoff_Ang) for ictc in self._contacts]
        _n_ctcs_t = []
        for ii in range(self.n_trajs):
            _n_ctcs_t.append(_np.vstack([itraj[ii] for itraj in bintrajs]).T.sum(1))
        return _n_ctcs_t

    def histo(self, ctc_cutoff_Ang,
              jax=None):
        r"""
        Base method for histogramming contact frequencies of the contacts
        contained in this class
        Parameters
        ----------
        ctc_cutoff_Ang: float
        jax: if None is passed, one will be created

        Returns
        -------
        jax:
        """
        if jax is None:
            _plt.figure()
            jax = _plt.gca()
        freqs = self._all_freqs(ctc_cutoff_Ang)
        xvec = _np.arange(len(freqs))
        patches = jax.bar(xvec, freqs,
                          # label=res_and_fragment_str,
                          width=.25)
        jax.set_yticks([.25, .50, .75, 1])
        jax.set_ylim([0, 1])
        jax.set_xticks([])
        [jax.axhline(ii, color="k", linestyle="--", zorder=-1) for ii in [.25, .50, .75]]
        return jax

    def histo_neighborhood(self, ctc_cutoff_Ang,
                           n_nearest,
                           xlim=None,
                           jax=None,
                           shorten_AAs=False,
                           label_fontsize_factor=1):

        # Base plot
        jax = self.histo(ctc_cutoff_Ang,
                         jax=jax)
        # Cosmetics
        jax.set_title(
            "Contact frequency @%2.1f $\AA$\n"
            "%u nearest bonded neighbors excluded" % (ctc_cutoff_Ang, n_nearest))


        label_dotref = self.anchor_res_and_fragment_str
        label_bars = self.partner_res_and_fragment_labels
        if shorten_AAs:
            label_dotref = self.anchor_res_and_fragment_str_short
            label_bars = self.partner_res_and_fragment_labels_short

        jax.plot(-1, -1, 'o',
                 color=self.anchor_fragment_color,
                 label=_replace4latex(label_dotref))

        self.add_tilted_labels_to_patches(jax,
                                          label_bars,
                                          label_fontsize_factor=label_fontsize_factor)

        jax.legend()
        if xlim is not None:
            jax.set_xlim([-.5, xlim + 1 - .5])

        return jax

    def add_tilted_labels_to_patches(self, jax, labels, label_fontsize_factor=1):
        for ii, (ipatch, ilab) in enumerate(zip(jax.patches, labels)):
            ix = ii
            iy = ipatch.get_height()
            iy += .01
            if iy > .65:
                iy = .65
            jax.text(ix, iy, _replace4latex(ilab),
                     va='bottom',
                     ha='left',
                     rotation=45,
                     fontsize=_rcParams["font.size"]*label_fontsize_factor,
                     backgroundcolor="white"
                     )

    def plot_timedep_ctcs(self, panelheight, color_scheme, ctc_cutoff_Ang,
                          n_smooth_hw, dt, t_unit,
                          gray_background,
                          shorten_AAs=False,
                          myfig=None):
        if self.n_ctcs > 0:
            if myfig is None:
                myfig, myax = _plt.subplots(self.n_ctcs+1, 1,
                                            #sharex=True,
                                           #sharey=True,
                                           figsize=(10, (self.n_ctcs+1) * panelheight))
            else:
                # TODO test this
                myax = myfig.axes
            #myax = _np.array(myax, ndmin=1)

            # One title for all axes on top
            title=self.anchor_res_and_fragment_str
            if shorten_AAs:
                title = self.anchor_res_and_fragment_str_short
            myax[0].set_title(title)

            # Plot individual contacts
            for ictc, iax in zip(self._contacts, myax[:self.n_ctcs]):
                plot_contact(ictc,iax,
                             color_scheme,
                             ctc_cutoff_Ang,
                             n_smooth_hw,
                             dt,
                             gray_background,
                             shorten_AAs=shorten_AAs
                             )

            # Cosmetics
            [iax.set_xticklabels([]) for iax in myax[:self.n_ctcs-1]]
            [iax.set_xlim([0, self.time_max*dt]) for iax in myax[:self.n_ctcs]]
            myax[self.n_ctcs-1].set_xlabel('t / %s' % t_unit)

            # TODO figure out how to put xticklabels on top
            axtop, axbottom = myax[0], myax[self.n_ctcs-1]
            iax2 = axtop.twiny()
            iax2.set_xticks(axbottom.get_xticks())
            iax2.set_xticklabels(axbottom.get_xticklabels())
            iax2.set_xlim(axtop.get_xlim())
            iax2.set_xlabel(axbottom.get_xlabel())

        myfig.tight_layout(pad=0, h_pad=0, w_pad=0)
        return myfig

    def plot_timedep_Nctcs(self,
                           iax,
                           color_scheme, ctc_cutoff_Ang,
                           n_smooth_hw, dt, t_unit,
                           gray_background,
                           ):
        #Plot ncontacts in the last frame
        icol = iter(color_scheme)
        for n_ctcs_t, itime, traj_name in zip(self.timedep_n_ctcs(ctc_cutoff_Ang),
                                              self.time_arrays,
                                              self.trajlabels):
            plot_w_smoothing_auto(iax, itime*dt, n_ctcs_t,traj_name,next(icol),
                                  gray_background=gray_background,
                                  n_smooth_hw=n_smooth_hw)

        iax.set_ylabel('$\sum$ [ctcs < %s $\AA$]'%(ctc_cutoff_Ang))
        iax.set_xlabel('t / %s'%t_unit)
        iax.set_xlim([0,self.time_max*dt])
        iax.legend()

    def to_dicts_for_saving(self, dt=1, t_unit="ps"):
        dicts = []
        for ii in range(self.n_trajs):
            labels = ['time / %s'%t_unit]
            data = [self.time_arrays[ii]*dt]
            for ictc in self._contacts:
                labels.append('%s / Ang'%ictc.ctc_label)
                data.append(ictc.ctc_trajs[ii]*10)
            data= _np.vstack(data).T
            dicts.append({"header":labels,
                          "data":data
                          }
                         )
        return dicts

    def save_trajs(self, output_desc, ext,
                   output_dir='.',
                   dt=1,
                   t_unit="ps",
                   verbose=False):
        dicts = self.to_dicts_for_saving(dt=dt, t_unit=t_unit)
        for idict, ixtc in zip(dicts, self.trajlabels):
            traj_name = _path.splitext(ixtc)[0]
            savename = "%s.%s.%s.%s" % (
                output_desc, self.anchor_res_and_fragment_str.replace('*', ""), traj_name, ext)
            savename = _path.join(output_dir, savename)
            if ext == 'xlsx':
                _DF(idict["data"],
                    columns=idict["header"]).to_excel(savename,
                                                      float_format='%6.3f',
                                                      index=False)
            else:
                _np.savetxt(savename, idict["data"],
                            ' '.join(["%6.3f" for __ in idict["header"]]),
                            header=' '.join(["%6s" % key.replace(" ", "") for key in idict["header"]]))

            if verbose:
                print(savename)

def plot_contact(ictc, iax,
                 color_scheme,
                 ctc_cutoff_Ang,
                 n_smooth_hw,
                 dt,
                 gray_background,
                 shorten_AAs=False,
                 ):
    iax.set_ylabel('D / $\\AA$', rotation=90)
    iax.set_ylim([0, 10])
    icol = iter(color_scheme)
    for traj_idx, (ictc_traj, itime, trjlabel) in enumerate(zip(ictc.ctc_trajs,
                                                                ictc.time_arrays,
                                                                ictc.trajlabels)):

        ilabel = '%s (%u%%)' % (
            trjlabel, ictc.frequency_per_traj(ctc_cutoff_Ang)[traj_idx] * 100)

        plot_w_smoothing_auto(iax, itime * dt, ictc_traj * 10,
                              ilabel,
                              next(icol),
                              gray_background=gray_background,
                              n_smooth_hw=n_smooth_hw)

    iax.legend(loc=1)
    ctc_label = ictc.ctc_label
    if shorten_AAs:
        ctc_label = ictc.ctc_label_short
    iax.text(_np.mean(iax.get_xlim()), 1, ctc_label, ha='center')
    iax.axhline(ctc_cutoff_Ang, color='k', zorder=10)


def plot_w_smoothing_auto(iax, x, y,
                          ilabel,
                          icolor,
                          gray_background=False,
                          n_smooth_hw=0):
    alpha = 1
    if n_smooth_hw > 0:
        from .list_utils import window_average as _wav
        alpha = .2
        x_smooth, _ = _wav(x, half_window_size=n_smooth_hw)
        y_smooth, _ = _wav(y, half_window_size=n_smooth_hw)
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


class contact_pair(object):
    r"""Class for storing everything related to a contact"""
    #todo consider packing some of this stuff in the site_obj class
    def __init__(self, res_idx_pair,
                 ctc_trajs,
                 time_arrays,
                 top=None,
                 trajs=None,
                 fragment_idxs=None,
                 fragment_names=None,
                 fragment_colors=None,
                 anchor_residue_idx=None,
                 consensus_labels=None):

        self._res_idx_pair = res_idx_pair
        self._ctc_trajs = ctc_trajs
        self._top = top
        self._trajs = trajs

        self._time_arrays = time_arrays
        self._n_trajs = len(ctc_trajs)
        assert self._n_trajs == len(time_arrays)
        assert all([len(itraj)==len(itime) for itraj, itime in zip(ctc_trajs, time_arrays)])
        self._time_max = _np.max(time_arrays)

        self._anchor_residue_index = anchor_residue_idx
        self._partner_residue_index = None
        self._anchor_index = None
        self._partner_index = None
        self._anchor_residue = None
        self._partner_residue = None
        if self._anchor_residue_index is not None:
            assert self._anchor_residue_index in self.res_idx_pair
            self._anchor_index  = _np.argwhere(self.res_idx_pair == self.anchor_residue_index).squeeze()
            self._partner_index = _np.argwhere(self.res_idx_pair != self.anchor_residue_index).squeeze()
            self._partner_residue_index = self.res_idxs_pair[self.partner_index]
            if self.top is not None:
                self._anchor_residue  = self.top.residue(self.anchor_residue_index)
                self._partner_residue = self.top.residue(self.partner_residue_index)

        self._consensus_labels = consensus_labels
        self._fragment_idxs  = fragment_idxs
        self._fragment_names = fragment_names
        self._fragment_colors = fragment_colors

    #TODO many of these properties will fail if partner nor anchor are None
    # todo many of these properties could be simply methods with options
    # to reduce code

    @property
    def time_max(self):
        return self._time_max

    @property
    def trajlabels(self):
        if self.trajs is None:
            return ['traj %u'%ii for ii in range(self.n_trajs)]
        else:
            return self.trajs

    @property
    def n_trajs(self):
        return self._n_trajs

    @property
    def n_frames(self):
        return [len(itraj) for itraj in self.ctc_trajs]

    @property
    def anchor_residue(self):
        return self._anchor_residue

    @property
    def partner_residue(self):
        return self._partner_residue

    @property
    def res_idx_pair(self):
        return self._res_idx_pair

    @property
    def anchor_residue_index(self):
        return self._anchor_residue_index

    @property
    def partner_residue_index(self):
        return self._partner_residue_index

    @property
    def residue_names(self):
        return [str(self.topology.residue(ii)) for ii in self.res_idxs_pair]

    @property
    def residue_names_short(self):
        from .aa_utils import shorten_AA as _shorten_AA
        return [_shorten_AA(rr, substitute_fail="long", keep_index=True) for rr in self.residue_names]

    @property
    def ctc_label(self):
        ctc_label = '%s@%s-%s@%s' % (self.residue_names[0],
                                     pick_best_label(self.fragment_names[0], self.consensus_labels[0]),
                                     self.residue_names[1],
                                     pick_best_label(self.fragment_names[1], self.consensus_labels[1]))
        return ctc_label

    @property
    def ctc_label_short(self):
        ctc_label = '%s@%s-%s@%s' % (self.residue_names_short[0],
                                     pick_best_label(self.fragment_names[0], self.consensus_labels[0]),
                                     self.residue_names_short[1],
                                     pick_best_label(self.fragment_names[1], self.consensus_labels[1]))
        return ctc_label

    @property
    def anchor_fragment_name(self):
        r"""
        """
        return self.fragment_names[self.anchor_index]

    @property
    def partner_fragment_name(self):
        r"""
        """
        return self.fragment_names[self.partner_index]

    @property
    def partner_fragment_name_consensus(self):
        return self.consensus_labels[self.partner_index]

    @property
    def partner_fragment_name_best(self):
        return pick_best_label(self.partner_fragment_name,
                               self.partner_fragment_name_consensus)

    @property
    def anchor_fragment_name_consensus(self):
        return self.consensus_labels[self.anchor_index]

    @property
    def anchor_fragment_name_best(self):
        return pick_best_label(self.anchor_fragment_name,
                               self.anchor_fragment_name_consensus)

    @property
    def anchor_res_and_fragment_str(self):
        return '%s@%s' % (self.anchor_residue,
                          self.anchor_fragment_name_best)

    @property
    def anchor_res_and_fragment_str_short(self):
        return '%s@%s' % (self.residue_names_short[self.anchor_index],
                          self.anchor_fragment_name_best)

    @property
    def partner_res_and_fragment_str(self):
        return '%s@%s' % (self.partner_residue,
                          self.partner_fragment_name_best)

    @property
    def partner_res_and_fragment_str_short(self):
        return '%s@%s' % (self.residue_names_short[self.partner_index],
                          self.partner_fragment_name_best)

    @property
    def time_arrays(self):
        return self._time_arrays

    @property
    def ctc_trajs(self):
        return self._ctc_trajs

    @property
    def trajs(self):
        return self._trajs

    @property
    def fragment_colors(self):
        return self._fragment_colors

    @property
    def res_idxs_pair(self):
        return self._res_idx_pair

    @property
    def fragment_names(self):
        return self._fragment_names

    @property
    def anchor_index(self):
        return self._anchor_index

    @property
    def partner_index(self):
        return self._partner_index

    @property
    def top(self):
        return self._top

    @property
    def topology(self):
        return self._top

    @property
    def consensus_labels(self):
        return self._consensus_labels

    def binarize_trajs(self, ctc_cutoff_Ang):
        result = [itraj < ctc_cutoff_Ang / 10 for itraj in self._ctc_trajs]
        #print([ires.shape for ires in result])
        return result

    def frequency_overall(self, ctc_cutoff_Ang):
        return _np.mean(_np.hstack(self.binarize_trajs(ctc_cutoff_Ang)))

    def frequency_per_traj(self, ctc_cutoff_Ang):
        return [_np.mean(itraj) for itraj in self.binarize_trajs(ctc_cutoff_Ang)]

    def __str__(self):
        out = "Contact object for residue indices"
        out += "\n%s"%self.res_idxs_pair
        out += "\nanchor residue index: %s"%self.anchor_residue_index
        out += "\nFor %u trajectories"%self.n_trajs
        for var in dir(self):
            if not var.startswith("_"):
                out += '\n%s: %s'%(var, getattr(self,'%s'%var))
        return out

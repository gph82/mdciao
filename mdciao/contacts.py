import numpy as _np
import mdtraj as _md
from os import path as _path
from .list_utils import in_what_fragment, \
    put_this_idx_first_in_pair

from .COM_utils import geom2COMdist

from .residue_and_atom_utils import \
    shorten_AA as _shorten_AA, \
    _atom_type

from .str_and_dict_utils import \
    _replace_w_dict,\
    unify_freq_dicts,\
    _replace4latex, \
    iterate_and_inform_lambdas, \
    _tunit2tunit, \
    choose_between_good_and_better_strings

from .plots import plot_w_smoothing_auto, \
    add_tilted_labels_to_patches as _add_tilted_labels_to_patches, \
    plot_contact_matrix as _plot_contact_matrix

from collections import defaultdict, Counter as _col_Counter

import matplotlib.pyplot as _plt
from matplotlib import rcParams as _rcParams
from pandas import DataFrame as _DF, \
    ExcelWriter as _ExcelWriter

from joblib import Parallel as _Parallel, delayed as _delayed


def select_and_report_residue_neighborhood_idxs(ctc_freqs, resSeq2residxs, fragments,
                                                residxs_pairs, top,
                                                n_ctcs=5,
                                                restrict_to_resSeq=None,
                                                interactive=False,
                                                fac=.9
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
    selection : dictionary
       selection[300] = [100,200,201,208,500,501]
       means that pairs :obj:`residxs_pairs[100]`,...
       are the most frequent formed contacts for residue 300
       (up to n_ctcs or less, see option 'interactive')
    """
    assert len(ctc_freqs) == len(residxs_pairs)

    order = _np.argsort(ctc_freqs)[::-1]
    selection = {}
    if restrict_to_resSeq is None:
        restrict_to_resSeq = list(resSeq2residxs.keys())
    elif isinstance(restrict_to_resSeq, int):
        restrict_to_resSeq = [restrict_to_resSeq]
    for resSeq, residx in resSeq2residxs.items():
        if resSeq in restrict_to_resSeq:
            order_mask = _np.array([ii for ii in order if residx in residxs_pairs[ii]],dtype=int)
            print("#idx    Freq  contact             segA-segB residxA   residxB   ctc_idx  sum")
            isum = 0
            seen_ctcs = []
            for ii, oo in enumerate(order_mask[:n_ctcs]):
                pair = residxs_pairs[oo]
                idx1, idx2 = put_this_idx_first_in_pair(residx, pair)
                s1, s2 = [in_what_fragment(idx, fragments) for idx in [idx1,idx2]]
                imean = ctc_freqs[oo]
                isum += imean
                seen_ctcs.append(imean)
                print("%-6s %3.2f %8s-%-8s    %5u-%-5u %7u %7u %7u %3.2f" % (
                 '%u:' % (ii + 1), imean, top.residue(idx1), top.residue(idx2), s1, s2, idx1, idx2, oo, isum))
            total_n_ctcs = _np.array(ctc_freqs)[order_mask].sum()
            nc = _np.argwhere(_np.cumsum(_np.array(ctc_freqs)[order_mask])>=total_n_ctcs*fac)[0]+1
            print("These %u contacts capture %3.1f of the total %3.1f (over %u contacts)."
                  " %u ctcs already capture %3.1f%% of %3.1f."%(ii+1,isum,total_n_ctcs, len(order_mask), nc, fac*100, total_n_ctcs))

            if interactive:
                try:
                    answer = input("How many do you want to keep (Hit enter for None)?\n")
                except KeyboardInterrupt:
                    break
                if len(answer) == 0:
                    pass
                else:
                    answer = _np.arange(_np.min((int(answer), n_ctcs)))
                    selection[residx] = order_mask[answer]
            else:
                seen_ctcs = _np.array(seen_ctcs)
                n_nonzeroes = (seen_ctcs > 0).astype(int).sum()
                answer = _np.arange(_np.min((n_nonzeroes, n_ctcs)))
                selection[residx] = order_mask[answer]
    # TODO think about what's best to return here
    # TODO think about making a pandas dataframe with all the above info
    return selection

def trajs2ctcs(trajs, top, ctc_residxs_pairs, stride=1, consolidate=True,
               chunksize=1000, return_times_and_atoms=False,
               n_jobs=1,
               progressbar=False,
               **mdcontacts_kwargs):
    """Returns the time-dependent traces of residue-residue contacts from
    a list of trajectories

    Parameters
    ----------
    trajs : list of strings
        list of filenames with trajectory data. Typically xtcs,
        but can be any type of file readable by :obj:`mdtraj`
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
    return_times_and_atoms : boolean, default is False
        Return also the time array in ps and the indices of the the atoms
        behind the distanes in :obj:`ctcs`. See :obj:`per_traj_ctcs` for
        more info
    n_jobs : int, default is 1
        To how many processors to parallellize. The algorithm parallelizes
        over the trajectories themeselves, having 3 trajs and n_jobs=4
        is equal t n_jobs=3
    progressbar : bool, default is False
        Use a fancy :obj:`tqdm.tqdm` progressbar



    Returns
    -------
    ctcs, or
    ctcs, time_trajs, atom_idxs if return_time=True

    """

    from tqdm import tqdm

    if progressbar:
        iterfunct = lambda a : tqdm(a)
    else:
        iterfunct = lambda a : a
    print("AAAA",type(trajs),len(trajs),"AAAAA")
    ictcs_itimes_iaps = _Parallel(n_jobs=n_jobs)(_delayed(per_traj_ctc)(top, itraj, ctc_residxs_pairs, chunksize, stride, ii,
                                                                        **mdcontacts_kwargs)
                                            for ii, itraj in enumerate(iterfunct(trajs)))
    ctcs = []
    times = []
    aps = []
    for ictcs, itimes, iaps in ictcs_itimes_iaps:
        ctcs.append(ictcs)
        times.append(itimes)
        aps.append(iaps)

    if consolidate:
        actcs = _np.vstack(ctcs)
        times = _np.hstack(times)
        aps = _np.vstack(aps)
    else:
        actcs = ctcs

    if not return_times_and_atoms:
        return actcs
    else:
        return actcs, times, aps

def per_traj_ctc(top, itraj, ctc_residxs_pairs, chunksize, stride,
                 traj_idx,
                 **mdcontacts_kwargs):
    r"""
    Wrapper for :obj:`mdtraj.contacs` for strided, chunked computation
    of contacts of either :obj:`mdtraj.Trajectory` objects or
    trajectory files on disk (e.g. xtcs, dcs etc)

    You can fine-tune the computation itself using mdcontacts_kwargs

    Prints out progress report while working

    Parameters
    ----------
    top: `mdtraj.Topology`
    itraj: `mdtraj.Trajctory` or filename
    ctc_residxs_pairs: iterable of pairs of residue indices
        Distances to be computed
    chunksize: int
        Size (in frames) of the "chunks" in which the contacs will be computed.
        Decrease the chunksize if you run into memmory errors
    stride:int
        Stride with which the contacts will be streamed over
    traj_idx: int
        The index of the trajectory being computed. For completeness
        of the progress report
    mdcontacts_kwargs:
        Optional keyword arguments to pass to :obj:`mdtraj.contacs`

        Note:
        ----
        If "scheme" is contained in mdcontacts_kwargs and scheme==COM,
        the center of mass will be computed

    Returns
    -------
    ictcs, itime, iatps
    ictcs: 2D np.ndarray (Nframes, Nctcs), where Nctcs= len(ctc_residxs_pairs)
        time traces of the wanted contacts, in

    itime: 1D np.ndarray of len Nframes
        timestamps of the computed contacts

    iatps: 2D np.ndarray (Nframes, 2*Nctcs)
        atom-indices yielding distances in ictcs, helps dis-aggregate the
        residue interaction into backbone-backbone, backbone-sidechain, or sidechain-sidechain


    """
    iterate, inform = iterate_and_inform_lambdas(itraj, chunksize, stride=stride, top=top)
    ictcs, itime, iaps = [],[],[]
    running_f = 0
    inform(itraj, traj_idx, 0, running_f)
    for jj, igeom in enumerate(iterate(itraj)):
        running_f += igeom.n_frames
        inform(itraj, traj_idx, jj, running_f)
        itime.append(igeom.time)
            #TODO make lambda out of this if
        if 'scheme' in mdcontacts_kwargs.keys() and mdcontacts_kwargs["scheme"].upper()=='COM':
            jctcs = geom2COMdist(igeom, ctc_residxs_pairs)
            j_atompairs = _np.full((len(jctcs), 2*len(ctc_residxs_pairs)),_np.nan)
        else:
            jctcs, jidx_pairs, j_atompairs = compute_contacts(igeom, ctc_residxs_pairs, **mdcontacts_kwargs)
            # TODO do proper list comparison and do it only once
            assert len(jidx_pairs) == len(ctc_residxs_pairs)

        ictcs.append(jctcs)
        iaps.append(j_atompairs)

    itime = _np.hstack(itime)
    ictcs = _np.vstack(ictcs)
    iatps = _np.vstack(iaps)

    return ictcs, itime, iatps

class _TimeTraces(object):

    def __init__(self, ctc_trajs,
                 time_trajs,
                 trajs,
                 atom_pair_trajs):

        assert len(time_trajs)==len(ctc_trajs)
        self._ctc_trajs = [_np.array(itraj,dtype=float) for itraj in ctc_trajs]
        self._time_trajs =[_np.array(tt,dtype=float) for tt in time_trajs]
        self._trajs = trajs
        if trajs is not None:
            assert len(trajs)==len(ctc_trajs)
        self._atom_pair_trajs = atom_pair_trajs
        assert all([len(itraj) == len(itime) for itraj, itime in zip(ctc_trajs, time_trajs)])
        if atom_pair_trajs is not None:
            assert len(atom_pair_trajs)==len(ctc_trajs)
            assert all([len(itraj) == len(iatt) for itraj, iatt in zip(ctc_trajs, atom_pair_trajs)]), "atom_pair_trajs does not have the appropiate length"
            self._atom_pair_trajs = [_np.array(itraj) for itraj in self._atom_pair_trajs]
            assert all([itraj.shape[1]==2 for itraj in self._atom_pair_trajs])
    # Trajectories
    @property
    def ctc_trajs(self):
        """

        Returns
        -------

        """
        return self._ctc_trajs

    @property
    def atom_pair_trajs(self):
        return self._atom_pair_trajs

    @property
    def time_trajs(self):
        """

        Returns
        -------

        """
        return self._time_trajs

    @property
    def trajs(self):
        """

        Returns
        -------

        """
        return self._trajs

    @property
    def feat_trajs(self):
        return self.ctc_trajs

class _NumberOfthings(object):

    def __init__(self, n_trajs, n_frames):
        self._n_trajs = n_trajs
        self._n_frames = n_frames

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
        return self._n_frames

    @property
    def n_frames_total(self):
        return _np.sum(self._n_frames)

class _Residues(object):
    def __init__(self, res_idxs_pair,
                 residue_names,
                 anchor_residue_idx=None,
                 consensus_labels=None,
                 top=None):

        assert len(res_idxs_pair)==2
        assert all([isinstance(ii,(int,_np.int64)) for ii in res_idxs_pair])
        assert res_idxs_pair[0]!=res_idxs_pair[1]

        self._res_idxs_pair = _np.array(res_idxs_pair)
        self._residue_names = residue_names

        if anchor_residue_idx is not None:
            assert anchor_residue_idx in res_idxs_pair
        self._anchor_residue_index = anchor_residue_idx
        self._partner_residue_index = None
        self._anchor_index = None
        self._partner_index = None
        self._anchor_residue = None
        self._partner_residue = None
        if self._anchor_residue_index is not None:
            assert self._anchor_residue_index in self.idxs_pair
            self._anchor_index = _np.argwhere(self.idxs_pair == self.anchor_residue_index).squeeze()
            self._partner_index = _np.argwhere(self.idxs_pair != self.anchor_residue_index).squeeze()
            self._partner_residue_index = self.idxs_pair[self.partner_index]
            if top is not None:
                self._anchor_residue =  top.residue(self.anchor_residue_index)
                self._partner_residue = top.residue(self.partner_residue_index)
        if consensus_labels is None:
            consensus_labels = [None,None]
        assert len(consensus_labels)==2
        self._consensus_labels = consensus_labels

    @property
    def idxs_pair(self):
        """

        Returns
        -------
        list of residue index pair passed

        """
        return self._res_idxs_pair

    @property
    def names(self):
        """

        Returns
        -------
        list, for each residue index in the residue contact pair, the corresponding residue name from the topology file.
        example : ['GLU30','VAL212']

        """
        return self._residue_names

    @property
    def names_short(self):
        """

        Returns
        -------
        list, for each residue name in the residue contact pair, the corresponding short residue name from the topology file.
        example : ['E30', 'V212']

        """
        return [_shorten_AA(rr, substitute_fail="long", keep_index=True) for rr in self.names]

    @property
    def anchor_residue(self):
        """

        Returns
        -------
        str, anchor residue if anchor residue index is provided else None

        """
        return self._anchor_residue

    @property
    def partner_residue(self):
        """

        Returns
        -------
        str, partner residue if partner residue index is provided else None

        """
        return self._partner_residue

    @property
    def anchor_residue_index(self):
        """

        Returns
        -------
        int, anchor residue index if passed else None(default)

        """
        return self._anchor_residue_index

    @property
    def partner_residue_index(self):
        """

        Returns
        -------
        int, partner residue if passed else (default)

        """
        return self._partner_residue_index

    @property
    def anchor_index(self):
        """

        Returns
        -------

        """
        return self._anchor_index

    @property
    def partner_index(self):
        """

        Returns
        -------

        """
        return self._partner_index

    @property
    def consensus_labels(self):
        """

        Returns
        -------

        """
        return self._consensus_labels

class _NeighborhoodNames(object):

    def __init__(self, residue_container, fragment_container):
        # Eliminate a lot of assertions down the line
        assert residue_container.anchor_index is not None, ValueError("Cannot instantiate if the residue container does not have an anchor residue")
        # TODO not enough protection for the containers, but easier code
        # better forgiveness than permission?

        self._residues = residue_container
        self._fragments = fragment_container

    @property
    def residues(self):
        return self._residues

    @property
    def fragments(self):
        return self._fragments

    @property
    def anchor_fragment(self):
        """

        Returns
        -------
        str, fragment name in which the anchor residue is present.

        """
        return self.fragments.names[self.residues.anchor_index]

    @property
    def partner_fragment(self):
        """

        Returns
        -------
        str, fragment name in which the partner residue is present

        """
        return self.fragments.names[self.residues.partner_index]

    @property
    def partner_fragment_consensus(self):
        """

        Returns
        -------
        consensus label of the partner residue

        """
        if self.residues.consensus_labels is not None:
            return self.residues.consensus_labels[self.residues.partner_index]
        else:
            return None

    @property
    def anchor_fragment_consensus(self):
        """

        Returns
        -------
        consensus label of the anchor residue. If no anchor_index is present then returns None

        """
        if self.residues.consensus_labels is not None:
            return self.residues.consensus_labels[self.residues.anchor_index]
        else:
            return None

    @property
    def partner_fragment_best(self):
        """

        Returns
        -------

        """
        return choose_between_good_and_better_strings(self.partner_fragment,
                                                      self.partner_fragment_consensus)

    @property
    def anchor_fragment_best(self):
        """

        Returns
        -------

        """
        return choose_between_good_and_better_strings(self.anchor_fragment,
                                                      self.anchor_fragment_consensus)

    @property
    def anchor_res_and_fragment_str(self):
        """

        Returns
        -------

        """
        return '%s@%s' % (self.anchor_residue_name,
                          self.anchor_fragment_best)

    @property
    def anchor_residue_name(self):
        return self.residues.names[self.residues.anchor_index]

    @property
    def partner_residue_name(self):
        return self.residues.names[self.residues.partner_index]

    @property
    def anchor_residue_short(self):
        return self.residues.names_short[self.residues.anchor_index]

    @property
    def partner_residue_short(self):
        return self.residues.names_short[self.residues.partner_index]

    @property
    def anchor_res_and_fragment_str_short(self):
        """

        Returns
        -------

        """
        return '%s@%s' % (self.anchor_residue_short,
                          self.anchor_fragment_best)

    @property
    def partner_res_and_fragment_str(self):
        """

        Returns
        -------

        """
        return '%s@%s' % (self.partner_residue_name,
                          self.partner_fragment_best)

    @property
    def partner_res_and_fragment_str_short(self):
        """

        Returns
        -------

        """
        return '%s@%s' % (self.partner_residue_short,
                          self.partner_fragment_best)

class _ContactStrings(object):
    r"""
    Only Return Strings
    """
    def __init__(self,
                 n_trajs,
                 residue_container,
                 fragment_container=None,
                 trajs=None,
                 ):

        self._trajs = trajs
        self._n_trajs = n_trajs
        self._residues = residue_container
        self._fragments = fragment_container
        if self._fragments is None:
            self._fragnames = [None, None]
        else:
            self._fragnames = self._fragments.names

    @property
    def trajstrs(self):
        """

        Returns
        -------
        list, list of labels for each trajectory
        If labels were not passed, then labels like 'traj 0','traj 1' and so on are assigned
        If :obj:`mdtraj.Trajectory` objects were passed, then the "mdtraj" descriptor will be used
        If filenames were passed, then the extension will be cut-off
        """

        if self._trajs is None:
            trajlabels = ['traj %u' % ii for ii in range(self._n_trajs)]
        else:
            if isinstance(self._trajs[0], _md.Trajectory):
                trajlabels = ['mdtraj.%02u' % ii for ii in range(self._n_trajs)]
            else:
                trajlabels = [_path.splitext(ii)[0] for ii in self._trajs]

        return trajlabels

    @property
    def no_fragments(self):
        return "%s-%s"%(self._residues.names[0], self._residues.names[1])

    @property
    def no_fragments_short_AA(self):
        return "%s-%s" % (self._residues.names_short[0], self._residues.names_short[1])

    @property
    def w_fragments(self):
        """

        Returns
        -------
        str,

        """
        fmt = "@%s"
        ctc_label = '%s%s-%s%s' % (self._residues.names[0],
                                   self.fragment_labels_best(fmt)[0],
                                   self._residues.names[1],
                                   self.fragment_labels_best(fmt)[1])
        return ctc_label

    @property
    def w_fragments_short_AA(self):
        """
        A string of the form residue0@fragment0-residue1@fragment1

        Note
        ----

        """
        fmt = "@%s"
        ctc_label = '%s%s-%s%s' % (self._residues.names_short[0],
                                     self.fragment_labels_best(fmt)[0],
                                     self._residues.names_short[1],
                                     self.fragment_labels_best(fmt)[1])

        return ctc_label

    def fragment_labels_best(self,fmt):
        r"""
        The fragment name will try to pick the consensus nomenclature.
        If no consensus label for the residue exists, the actual fragment
        names are used as fallback (which themselves fallback to the fragment index)
        Only if no consensus label, no fragment name and no fragment indices are there,
        will this yeild "None" as a string.
        Returns
        -------
        list of two strings
        """
        return [choose_between_good_and_better_strings(self._fragnames[ii],
                                                       self._residues.consensus_labels[ii],
                                                       fmt=fmt)
                for ii in [0,1]]




    def __str__(self):
        istr = ["%s at %s with properties"%(type(self),id(self))]
        unprinted = []
        for iattr in [iattr for iattr in dir(self) if not iattr.startswith("__")]:
            if iattr[0]!="_":
                istr.append("%s:"%iattr)
                istr.append(" "+str(getattr(self,iattr)))
                istr.append(" ")

            else:
                unprinted.append(iattr)
        print(istr)
        return "\n".join(istr+["Unprinted"]+unprinted)

class _Fragments(object):

    def __init__(self,
                 fragment_idxs=None,
                 fragment_names=None,
                 fragment_colors=None,
                 ):
        self._fragment_idxs = fragment_idxs
        if fragment_colors is None:
            self._fragment_colors = [None,None]
        else:
            self._fragment_colors = fragment_colors

        if fragment_names is None:
            # assert self.idxs is not None
            # self._fragment_names = self._fragment_idxs
            if self.idxs is not None:
                self._fragment_names = [str(fidx) for fidx in self._fragment_idxs]
            else:
                self._fragment_names = [None, None]
        else:
            assert len(fragment_names)==2
            self._fragment_names = fragment_names

    @property
    def names(self):
        """

        Returns
        -------
        list of list, Fragment names if passed, else fragment idxs. If both are not available then None(default)

        """
        return self._fragment_names

    @property
    def idxs(self):
        """

        Returns
        -------
        list of list, Fragment idxs if passed, else None(default)

        """
        return self._fragment_idxs

    @property
    def colors(self):
        """

        Returns
        -------

        """
        return self._fragment_colors

class ContactPair(object):
    r"""Class for abstracting a single contact over many trajectory"""
    #todo consider packing some of this stuff in the site_obj class
    def __init__(self, res_idxs_pair,
                 ctc_trajs,
                 time_trajs,
                 top=None,
                 trajs=None,
                 atom_pair_trajs=None,
                 fragment_idxs=None,
                 fragment_names=None,
                 fragment_colors=None,
                 anchor_residue_idx=None,
                 consensus_labels=None):
        """

        Parameters
        ----------
        res_idxs_pair : iterable of two ints
            pair of residue indices, corresponding to the zero-indexed, serial number of the residues
        ctc_trajs : list of iterables of floats
            time traces of the contact. len(ctc_trajs) is N_trajs. Each traj can have different lenghts
            Will be cast into arrays
        time_trajs : list of iterables of floats
            time traces of the time-values, in ps. Not having the same shape as ctc_trajs will raise an error
        top : :py:class:`mdtraj.Topology`, default is None
            topology associated with the contact
        trajs: list of :obj:`mdtraj.Trajectory` objects, default is None
            The molecular trajectories responsible for which the contact has been evaluated.
            Not having the same shape as ctc_trajs will raise an error
        atom_pair_trajs: list of iterables of integers, default is None
            Time traces of the the pair of atom indices responsible for the distance in :obj:`ctc_trajs`
            Has to be of len(ctc_trajs) and each iterable of shape(Nframes, 2)
        fragment_idxs : iterable of two ints, default is None
            Indices of the fragments the residues of :obj:`res_idxs_pair`
        fragment_names : iterable of two strings, default is None
            Names of the fragments the residues of :obj:`res_idxs_pair`
        fragment_colors : iterable of matplotlib.colors, default is None
            Colors associated to the fragments of the residues of :obj:`res_idxs_pair`
        anchor_residue_idx : int, default is None
            Label this residue as the `anchor` of the contact, which will later allow
            for the implementation of a :obj:`neighborhood` (see docs).
            Has to be in :obj:`res_idxs_pair`.

            Note
            ----
            Using this argument will automatically populate other properties, like (this is not a complete list)
             - :obj:`anchor_index` will contain the [0,1] index of the anchor residue in :obj:`res_idxs_pair`
             - :obj:`partner_index` will contain the [0,1] index of the partner residue in :obj:`res_idxs_pair`
             - :obj:`partner_residue_index` will contain the other index of :obj:`res_idx_pair`
            and other properties which depend on having defined an anchor and a partner

            Furhtermore, if a topology is parsed as an argument:
             - :obj:`anchor_residue_name` will contain the anchor residue as an :obj:`mdtraj.core.Topology.Residue` object
             - :obj:`partner_residue_name` will contain the partner residue as an :obj:`mdtraj.core.Topology.Residue` object


        consensus_labels : iterable of strings, default is None
            Consensus nomenclature of the residues of :obj:`res_idxs_pair`
        """

        # Initialize the attribute holding classes
        self._attribute_trajs = _TimeTraces(ctc_trajs, time_trajs, trajs, atom_pair_trajs)
        self._attribute_n = _NumberOfthings(len(self._attribute_trajs.ctc_trajs),
                                            [len(itraj) for itraj in self._attribute_trajs.ctc_trajs])

        # Fail as early as possible
        _np.testing.assert_equal(self._attribute_n.n_trajs, len(self._attribute_trajs.time_trajs))

        residue_names = [str(ii) for ii in res_idxs_pair]
        if top is not None:
            residue_names = [str(top.residue(ii)) for ii in res_idxs_pair]

        self._attribute_residues = _Residues(res_idxs_pair,
                                             residue_names,
                                             anchor_residue_idx=anchor_residue_idx,
                                             consensus_labels=consensus_labels,
                                             top=top)

        self._attribute_fragments = _Fragments(fragment_idxs,
                                               fragment_names,
                                               fragment_colors)


        self._ctc_strings = _ContactStrings(self._attribute_n.n_trajs,
                                            self._attribute_residues,
                                            self._attribute_fragments,
                                            trajs=trajs)

        if self._attribute_residues.anchor_residue_index is not None:
            self._attribute_neighborhood_names = _NeighborhoodNames(self._attribute_residues,
                                                                    self._attribute_fragments)
        else:
            self._attribute_neighborhood_names = None

        self._top = top
        self._time_max = _np.max(_np.hstack(time_trajs))
        self._binarized_trajs = {}

    #Trajectories
    @property
    def time_traces(self):
        return self._attribute_trajs

    # Accounting
    @property
    def n(self):
        return self._attribute_n

    # Residues
    @property
    def residues(self):
        return self._attribute_residues

    # Fragments
    @property
    def fragments(self):
        return self._attribute_fragments

    # Neighborhood
    @property
    def neighborhood(self):
        return self._attribute_neighborhood_names

    # Labels (TODO rename to strings?)
    @property
    def labels(self):
        return self._ctc_strings

    @property
    def time_max(self):
        """

        Returns
        -------
        int or float, maximum time from list of list of time

        """
        return self._time_max

    @property
    def label(self):
        return self.labels.no_fragments

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

    def binarize_trajs(self, ctc_cutoff_Ang,
                       #switch_off_Ang=None
                       ):
        """
        Turn each distance-trajectory into a boolean using a cutoff.
        The comparison is done using "<=", s.t. d=ctc_cutoff yields True

        Whereas :obj:`ctc_cutoff_Ang` is in Angstrom, the trajectories are
        in nm, as produced by :obj:`mdtraj.compute_contacts`

        Note
        ----
        The method creates a dictionary in self._binarized_trajs keyed
        with the ctc_cutoff_Ang, to avoid re-computing already binarized
        trajs

        Parameters
        ----------
        ctc_cutoff_Ang: float
            Cutoff in Angstrom. The comparison operator is "<="


        Returns
        -------
        list of boolean arrays with the same shape as the trajectories

        """
        transform = lambda itraj: itraj <= ctc_cutoff_Ang / 10

        """
        if switch_off_Ang is not None:
            assert isinstance(switch_off_Ang, float) and switch_off_Ang>0
            m = -1 / (switch_off_Ang/10)
            b = 1 +  (ctc_cutoff_Ang / switch_off_Ang)
            def transform(d):
                res = m * d + b
                res[d<ctc_cutoff_Ang/10]=1
                res[d > (ctc_cutoff_Ang + switch_off_Ang)/10]=0

                return res
        """

        try:
            result = self._binarized_trajs[ctc_cutoff_Ang]
            #print("Grabbing already binarized %3.2f"%ctc_cutoff_Ang)
        except KeyError:
            #print("First time binarizing %3.2f. Storing them"%ctc_cutoff_Ang)
            result = [transform(itraj) for itraj in self.time_traces.ctc_trajs]
            self._binarized_trajs[ctc_cutoff_Ang] = result
        #print([ires.shape for ires in result])
        return result

    def frequency_per_traj(self, ctc_cutoff_Ang):
        """
        Contact frequencies for each trajectory

        Parameters
        ----------
        ctc_cutoff_Ang : float
            Cutoff in Angstrom. The comparison operator is "<="

        Returns
        -------
        freqs : array of len self.n.n_trajs with floats between [0,1]

        """

        return _np.array([_np.mean(itraj) for itraj in self.binarize_trajs(ctc_cutoff_Ang)])

    def frequency_overall_trajs(self, ctc_cutoff_Ang):
        """
        How many times this contact is formed overall frames. 
        Frequencies have values between 0 and 1
        
        Parameters
        ----------
        ctc_cutoff_Ang : float
            Cutoff in Angstrom. The comparison operator is "<="

        Returns
        -------
        freq: float
            Frequency of the contact over all trajectories

        """
        return _np.mean(_np.hstack(self.binarize_trajs(ctc_cutoff_Ang)))


    def frequency_dict(self, ctc_cutoff_Ang,
                       AA_format='short',
                       split_label=True):
        """
        Returns the :obj:`frequency_overall_trajs` as a more informative
        dictionary with keys "freq", "residue idxs", "label"

        Parameters
        ----------
        ctc_cutoff_Ang : float
            Cutoff in Angstrom. The comparison operator is "<="
        AA_format : str, default is "short"
            Amino-acid format ("E35" or "GLU25") for the value
            fdict["label"]. Can also be "long"
        split_label : bool, defaultis True
            Split the labels so that stacked contact labels
            become easier-to-read in plain ascii formats
            "E25@3.50    -    A35@4.50"
            "A30@longfrag-    A35@4.50

        Returns
        -------
        fdcit : dictionary

        """
        if AA_format== 'short':
            label = self.labels.w_fragments_short_AA
        elif AA_format== 'long':
            label = self.labels.w_fragments
        else:
            raise ValueError(AA_format)
        if split_label:
            label= '%-15s - %-15s'%tuple(label.split('-'))
        return {"freq":self.frequency_overall_trajs(ctc_cutoff_Ang),
                "residue idxs":'%u %u'%tuple(self.residues.idxs_pair),
                "label":label}

    def distro_overall_trajs(self, bins=10):
        """
        Wrapper around :obj:`numpy.histogram` to produce a distribution
        of the distance values (not the contact frequencies) this
        contact over all trajectories

        Parameters
        ----------
        bins : int or anything :obj:`numpy.histogram` accepts

        Returns
        -------
        h : _np.ndarray
            The counts (integer valued)
        x : _np.ndarray
            The bin edges ``(length(hist)+1)``.

        """
        return _np.histogram(_np.hstack(self.time_traces.ctc_trajs),
                             bins=bins)

    def _overall_stacked_formed_atoms(self, ctc_cutoff_Ang):
        r"""
        Returns the pairs of atom-indices responsible for the contact,
        only for the frames in which the contact was formed at the given cutoff

        Parameters
        ----------
        ctc_cutoff_Ang : float
            Cutoff in Angstrom. The comparison operator is "<="

        Returns
        -------
        formed_atom_pairs : _np.ndarray of len (N,2)

        """

        bintrajs = self.binarize_trajs(ctc_cutoff_Ang)
        formed_atom_pair_trajs = [atraj[itraj] for atraj, itraj in zip(self.time_traces.atom_pair_trajs, bintrajs)]

        return _np.vstack(formed_atom_pair_trajs)

    def count_formed_atom_pairs(self, ctc_cutoff_Ang,
                                sort=True):
        r"""
        Count how many times each atom-pair is considered formed overall trajectory

        Parameters
        ----------
        ctc_cutoff_Ang : float
            Cutoff in Angstrom. The comparison operator is "<="
        sort: boolean, default is True
            Return the counts by descending order

        Returns
        -------
        atom_pairs, counts : list of atom pairs, list of ints
        Note that no dictionary is returned bc atom_pairs is not hashable
        """

        assert self.time_traces.atom_pair_trajs is not None, ValueError("Cannot use this method if no atom_pair_trajs were parsed")
        counts = _col_Counter(["%u-%u"%tuple(fap) for fap in self._overall_stacked_formed_atoms(ctc_cutoff_Ang)])
        keys, counts = list(counts.keys()), list(counts.values())
        keys = [[int(ii) for ii in key.split("-")] for key in keys]
        if sort:
            keys = [keys[ii] for ii in _np.argsort(counts)[::-1]]
            counts=sorted(counts)[::-1]
        return keys, counts


    def relative_frequency_of_formed_atom_pairs_overall_trajs(self, ctc_cutoff_Ang,
                                                              keep_resname=False,
                                                              aggregate_by_atomtype=True,
                                                              min_freq=.05):
        r"""
        For those frames in which the contact is formed, group them by relative frequencies
        of individual atom pairs
        
        Parameters
        ----------
        ctc_cutoff_Ang: float
            Cutoff in Angstrom. The comparison operator is "<="
        keep_resname: bool, default is False
            Keep the atom's residue name in its descriptor. Only make
            sense if consolidate_by_atom_type is False
        aggregate_by_atomtype: bool, default is True
            Aggregate the frequencies of the contact by tye atom types involved.
            Atom types are backbone, sidechain or other (BB,SC, X)
        min_freq: float, default is .05
            Do not report relative frequencies below this cutoff, e.g.
            "BB-BB":.9, "BB-SC":0.03, "SC-SC":0.03, "SC-BB":0.03
            gets reported as "BB-BB":.9

        Returns
        -------
        out_dict : dictionary with the relative freqs
        """
        assert self.top is not None, "Missing a topolgy object"
        atom_pairs, counts = self.count_formed_atom_pairs(ctc_cutoff_Ang)
        atom_pairs_as_atoms = [[self.top.atom(ii) for ii in pair] for pair in atom_pairs]

        if aggregate_by_atomtype:
            dict_out = _sum_ctc_freqs_by_atom_type(atom_pairs_as_atoms, counts)
            return {key: val / _np.sum(counts) for key, val in dict_out.items() if
                    val / _np.sum(counts) > min_freq}
        else:
            if keep_resname:
                atom_pairs = ['-'.join([str(ii) for ii in key]) for key in atom_pairs_as_atoms]
            else:
                atom_pairs = ['-'.join([ii.name for ii in key]) for key in atom_pairs_as_atoms]
            return {key:count/_np.sum(counts) for key, count in zip(atom_pairs, counts)
                    if count/_np.sum(counts)>min_freq}

    def plot_timetrace(self,
                         iax,
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
        r"""
        Plot this contact's timetraces for all trajs onto :obj:`ax`
        Parameters
        ----------
        iax
        color_scheme : list, default is None
            Pass a list of colors understandable by matplotlib
        ctc_cutoff_Ang
        n_smooth_hw: int, default is 0
            Size, in frames, of half the window size of the
            smoothing window
        dt : float, default is 1
            The how units in :obj:`t_unit` one frame represents
        gray_background
        shorten_AAs
        t_unit
        ylim_Ang : float or "auto"
            The limit in Angstrom of the y-axis
        max_handles_per_row : int, default is
            legend control

        Returns
        -------

        """
        if color_scheme is None:
            color_scheme = _rcParams['axes.prop_cycle'].by_key()["color"]
        color_scheme = _np.tile(color_scheme, _np.ceil(self.n.n_trajs / len(color_scheme)).astype(int) + 1)
        iax.set_ylabel('D / $\\AA$', rotation=90)
        if isinstance(ylim_Ang, (int, float)):
            iax.set_ylim([0, ylim_Ang])
        elif isinstance(ylim_Ang, str) and ylim_Ang.lower() == 'auto':
            pass
        else:
            raise ValueError("Cannot understand your ylim value %s of type %s" % (ylim_Ang, type(ylim_Ang)))
        for traj_idx, (ictc_traj, itime, trjlabel) in enumerate(zip(self.time_traces.feat_trajs,
                                                                    self.time_traces.time_trajs,
                                                                    self.labels.trajstrs)):

            ilabel = '%s' % trjlabel
            if ctc_cutoff_Ang > 0:
                ilabel += ' (%u%%)' % (self.frequency_per_traj(ctc_cutoff_Ang)[traj_idx] * 100)

            plot_w_smoothing_auto(iax, itime * dt, ictc_traj * 10,
                                  ilabel,
                                  color_scheme[traj_idx],
                                  gray_background=gray_background,
                                  n_smooth_hw=n_smooth_hw)

        iax.legend(loc=1, fontsize=_rcParams["font.size"] * .75,
                   ncol=_np.ceil(self.n.n_trajs / max_handles_per_row).astype(int)
                   )
        ctc_label = self.label
        if shorten_AAs:
            ctc_label = self.labels.w_fragments_short_AA
        # TODO: I do not think this removals of "None" are necessary any more
        #ctc_label = ctc_label.replace("@None", "")
        if ctc_cutoff_Ang > 0:
            ctc_label += " (%u%%)" % (self.frequency_overall_trajs(ctc_cutoff_Ang) * 100)

        iax.text(_np.mean(iax.get_xlim()), 1 * 10 / _np.max((10, iax.get_ylim()[1])),  # fudge factor for labels
                 ctc_label,
                 ha='center')
        if ctc_cutoff_Ang > 0:
            iax.axhline(ctc_cutoff_Ang, color='k', ls='--', zorder=10)

        iax.set_xlabel('t / %s' % _replace4latex(t_unit))
        iax.set_xlim([0, self.time_max * dt])
        iax.set_ylim([0, iax.get_ylim()[1]])

    def __str__(self):

        istr = ["%s at %s with public stuff" % (type(self), id(self))]
        for iattr in dir(self):
            if iattr[0] != "_":
                istr.append("%s:" % iattr)
                istr.append(" " + str(getattr(self, iattr)))
                istr.append(" ")
        return "\n".join(istr)

class ContactGroup(object):
    r"""Class for containing contact objects, ideally
    it can be used for vicinities, sites, interfaces etc"""
    #TODO create an extra sub-class? Unsure
    def __init__(self,
                 list_of_contact_objects,
                 interface_residxs=None,
                 top=None,
                 use_AA_when_conslab_is_missing=True
                 ):
        r"""

        Parameters
        ----------
        list_of_contact_objects
        interface_residxs : list of two iterables of indexes
            An is_interface is defined by two, non-overlapping
            groups of residue indices.

            The only requirement is that the residues in
            each group do not overlap. The input `interface_residxs`
            need not have all or any of the residue indices in
            :obj:`res_idxs_pairs`

            The property :obj:`interface_residxs` groups
            the object's own residue idxs present in
            :obj:`residxs_pairs` into the two groups of the is_interface.

            #TODO document what happens if there is no overlap

        top
        """
        self._contacts = list_of_contact_objects
        self._n_ctcs  = len(list_of_contact_objects)
        self._interface_residxs = interface_residxs
        self._interface = False
        if top is None:
            self._top = self._unique_topology_from_ctcs()
        else:
            assert top is self._unique_topology_from_ctcs()
            self._top = top

        # Sanity checks about having grouped this contacts together
        if self._n_ctcs==0:
            raise NotImplementedError("This contact group has no contacts!")
        else:
            # All contacts have the same number of trajs
            self._n_trajs = _np.unique([ictc.n.n_trajs for ictc in self._contacts])
            assert len(self._n_trajs)==1, (self._n_trajs, [ictc.n.n_trajs for ictc in self._contacts])
            self._n_trajs=self._n_trajs[0]

            ref_ctc : ContactPair #TODO check if type-hinting is needed or it's just slow IDE over sshfs
            ref_ctc = self._contacts[0]

            # All trajs have the same length
            assert all([_np.allclose(ref_ctc.n.n_frames, ictc.n.n_frames) for ictc in self._contacts[1:]])
            self._time_arrays=ref_ctc.time_traces.time_trajs
            assert all([all([_np.array_equal(itime, jtime) for itime, jtime in zip(ref_ctc.time_traces.time_trajs,
                                                                                   ictc.time_traces.time_trajs)])
                        for ictc in self._contacts[1:]])
            self._time_max = ref_ctc.time_max
            self._n_frames = ref_ctc.n.n_frames

            # All contatcs have the same trajstrs
            already_printed = False
            for ictc in self._contacts[1:]:
                assert all([rlab.__hash__() == tlab.__hash__()
                            for rlab, tlab in zip(ref_ctc.labels.trajstrs, ictc.labels.trajstrs)])
                # todo why did I put this here in the first place
                #except AttributeError:
                #    if not already_printed:
                #        print("Trajectories unhashable, could not verify they are the same")
                #        already_printed = True
                #    else:
                #        pass

            #TODO
            # This is the part were short residue-names are used
            # instead of the consensus labels...rethink this perhaps?
            self._trajlabels = ref_ctc.labels.trajstrs
            self._cons2resname = {}
            self._residx2resname = {}
            self._residx2fragnamebest = {}
            self._residx2conslabels = {}
            self._residxs_missing_conslabels = []
            for conslab, val, ridx, fragname in zip(_np.hstack(self.consensus_labels),
                                                    _np.hstack(self.residue_names_short),
                                                    _np.hstack(self.res_idxs_pairs),
                                                    _np.hstack(self.fragment_names_best)):

                if ridx not in self._residx2resname.keys():
                    self._residx2resname[ridx] = val
                else:
                    assert self._residx2resname[ridx]==val, (self._residx2resname[ridx], val)

                if ridx not in self._residx2fragnamebest.keys():
                    self._residx2fragnamebest[ridx] = fragname
                else:
                    assert self._residx2fragnamebest[ridx] == fragname, (self._residx2fragnamebest[ridx], fragname)


                if ridx not in self._residx2conslabels.keys():
                    self._residx2conslabels[ridx] = conslab
                else:
                    assert self._residx2conslabels[ridx] == conslab, (self._residx2conslabels[ridx], conslab)

                if str(conslab).lower() in ["na","none"]:
                    self._residxs_missing_conslabels.append(ridx)
                else:
                    if conslab not in self._cons2resname.keys():
                        self._cons2resname[conslab]=val
                    else:
                        assert self._cons2resname[conslab]==val,(self._cons2resname[conslab],conslab,val)

            """
            # Finally do this dictionary
            self._resname2cons = {}
            for cl, kres in self._cons2resname.items():
                # Check the same residue always has the same consensus label
                assert kres not in self._resname2cons.keys(),"Consensus label '%s' " \
                                                              "would overwrite existing '%s' for " \
                                                              "residue with name '" \
                                                             "%s'. " \
                                                              "Check your consensus labels"%(cl,self._resname2cons[kres], val)
                self._resname2cons[kres]=cl

            # Append the missing ones
            for ii in self._residxs_missing_conslabels:
                self._resname2cons[self._residx2resname[ii]]=None
            """

            if self._interface_residxs is not None:
                # TODO prolly this is anti-pattern but I prefer these many sanity checks
                assert len(self._interface_residxs)==2
                intersect = list(set(self._interface_residxs[0]).intersection(self._interface_residxs[1]))
                assert len(intersect)==0, ("Some_residxs appear in both members of the is_interface %s, "
                                           "this is not possible"%intersect)
                _np.testing.assert_equal(len(self._interface_residxs[0]),len(_np.unique(self._interface_residxs[0])))
                _np.testing.assert_equal(len(self._interface_residxs[1]),len(_np.unique(self._interface_residxs[1])))

                res = []
                for ig in self._interface_residxs:
                    res.append(sorted(set(ig).intersection(_np.unique(self.res_idxs_pairs,
                                                                      ))))
                # TODO can I benefit from not sorting these idxs
                # later when using Group of Interfaces?

                # TODO would it be wise to keep all idxs of the initialisaton
                # to compare different interfaces?

                # TODO Is the comparison throuh residxs robust enought, would it be
                # better to compare consensus labels directly?

                self._interface_residxs = res
                if len(res[0])>0 and len(res[1])>0:
                    self._interface = True
            else:
                self._interface_residxs = [[],[]]
    #todo again the dicussion about named tuples vs a miriad of properties
    # I am opting for properties because of easyness of documenting i

    #TODO access to conctat labels with fragnames and/or consensus?
    @property
    def n_trajs(self):
        return self._n_trajs

    @property
    def n_ctcs(self):
        return self._n_ctcs

    @property
    def n_frames(self):
        return self._n_frames

    @property
    def time_max(self):
        return self._time_max

    @property
    def time_arrays(self):
        return self._time_arrays

    @property
    def res_idxs_pairs(self):
        return _np.vstack([ictc.residues.idxs_pair for ictc in self._contacts])

    @property
    def residue_names_short(self):
        return [ictc.residues.names_short for ictc in self._contacts]

    @property
    def fragment_names_best(self):
        return [ictc.labels.fragment_labels_best(fmt="%s") for ictc in self._contacts]

    @property
    def ctc_labels(self):
        return [ictc.labels.no_fragments for ictc in self._contacts]

    @property
    def ctc_labels_short(self):
        return [ictc.labels.no_fragments_short_AA
                for ictc in self._contacts]

    @property
    def ctc_labels_w_fragments_short_AA(self):
        return [ictc.labels.w_fragments_short_AA for ictc in self._contacts]

    @property
    def trajlabels(self):
        return self._trajlabels

    # The next objects can also be None
    @property
    def top(self):
        return self._top

    @property
    def topology(self):
        return self._top

    @property
    def consensus_labels(self):
        return [ictc.residues.consensus_labels for ictc in self._contacts]

    @property
    def consensuslabel2resname(self):
        return self._cons2resname

    @property
    def residx2consensuslabel(self):
        return self._residx2conslabels

    @property
    def residx2resnameshort(self):
        return self._residx2resname

    @property
    def residx2fragnamebest(self):
        return self._residx2fragnamebest

    def residx2resnamefragnamebest(self,fragsep="@"):
        idict = {}
        for key in _np.unique(self.res_idxs_pairs):
            val = self.residx2resnameshort[key]
            ifrag = self.residx2fragnamebest[key]
            if len(ifrag) > 0:
                val += "%s%s" % (fragsep, ifrag)
            idict[key] = val
        return idict

    @property
    def shared_anchor_residue_index(self):
        r"""
        Returns none if no anchor residue is found
        """
        if any([ictc.residues.anchor_residue_index is None for ictc in self._contacts]):
            #todo dont print so much
            #todo let it fail?
            print("Not all contact objects have an anchor_residue_index. Returning None")
        else:
            shared = _np.unique([ictc.residues.anchor_residue_index for ictc in self._contacts])
            if len(shared) == 1:
                return shared[0]
        return None

    @property
    def anchor_res_and_fragment_str(self):
        assert self.shared_anchor_residue_index is not None,"There is no anchor residue, This is not a neighborhood."
        return self._contacts[0].neighborhood.anchor_res_and_fragment_str

    @property
    def anchor_res_and_fragment_str_short(self):
        assert self.shared_anchor_residue_index is not None
        return self._contacts[0].neighborhood.anchor_res_and_fragment_str_short

    @property
    def partner_res_and_fragment_labels(self):
        assert self.shared_anchor_residue_index is not None
        return [ictc.neighborhood.partner_res_and_fragment_str for ictc in self._contacts]

    @property
    def partner_res_and_fragment_labels_short(self):
        assert self.shared_anchor_residue_index is not None
        return [ictc.neighborhood.partner_res_and_fragment_str_short for ictc in self._contacts]

    @property
    def anchor_fragment_color(self):
        assert self.shared_anchor_residue_index is not None
        _col = self._contacts[0].fragments.colors[self._contacts[0].residues.anchor_index]
        cond1 = not any([ictc.fragments.colors[ictc.residues.anchor_index] is None for ictc in self._contacts])
        cond2 = all([ictc.fragments.colors[ictc.residues.anchor_index] == _col for ictc in self._contacts[1:]])
        if cond1 and cond2:
            return _col
        else:
            print("Not all anchors have or share the same color, returning None")
            return None

    #todo there is redundant code for generating is_interface labels!
    # not sure we need it here, don't want to be testing now
    """
    @property
    def consensuslabel2resname(self):
        return self._cons2resname

    @property
    def resname2cons(self):
        return self._resname2cons
    """

    # Now the functions begin
    def _unique_topology_from_ctcs(self):
        if all([ictc.top is None for ictc in self._contacts]):
            return None

        top = _np.unique([ictc.top.__hash__() for ictc in self._contacts])
        if len(top)==1:
            return self._contacts[0].top
        else:
            raise ValueError("All contacts in a group of contacts"
                             " should have the same topology, but "
                             "I found these hashes %s"%top)

    def binarize_trajs(self, ctc_cutoff_Ang, order='contact'):
        r"""

        Parameters
        ----------
        ctc_cutoff_Ang
        order : str, default is "contact"
            Sort first by contact, then by traj index. Alternative is
            "traj", i.e. sort first by traj index, then by contact

        Returns
        -------
        bintrajs : list of boolean arrays
            if order==traj, each item of the list is a 2D np.ndarray
            with of shape(Nt,n_ctcs), where Nt is the number of frames
            of that trajectory

        """
        bintrajs = [ictc.binarize_trajs(ctc_cutoff_Ang) for ictc in self._contacts]
        if order=='contact':
            return bintrajs
        elif order=='traj':
            _bintrajs = []
            for ii in range(self.n_trajs):
                _bintrajs.append(_np.vstack([itraj[ii] for itraj in bintrajs]).T)
            bintrajs = _bintrajs
        else:
            raise ValueError(order)
        return bintrajs

    def residx2ctcidx(self,idx):
        r"""
        Indices of the contacts and the position (0 or 1) in which the residue with residue :obj:`idx` appears
        Parameters
        ----------
        idx: int

        Returns
        -------
        ctc_idxs : 2D np.ndarray of shape (N,2)
            The first index is the contact index, the second the the pair index (0 or 1)
        """
        ctc_idxs = []
        for ii, pair in enumerate(self.res_idxs_pairs):
            if idx in pair:
                ctc_idxs.append([ii,_np.argwhere(pair==idx).squeeze()])
        return _np.vstack(ctc_idxs)

    # TODO think about implementing a frequency class, but how
    # to do so without circular dependency to the ContactGroup object itself?
    def frequency_per_contact(self, ctc_cutoff_Ang):
        r"""
        Frequency per contact over all trajs
        Parameters
        ----------
        ctc_cutoff_Ang

        Returns
        -------
        freqs : 1D np.ndarray of len(n_ctcs)
        """
        return _np.array([ictc.frequency_overall_trajs(ctc_cutoff_Ang) for ictc in self._contacts])

    def frequency_sum_per_residue_idx_dict(self, ctc_cutoff_Ang):
        r"""
        Dictionary of aggregated :obj:`frequency_per_contact` per residue indices

        Parameters
        ----------
        ctc_cutoff_Ang

        Returns
        -------
        freqs_dict : dictionary
            keys are the residue indices present in :obj:`res_idxs_pairs`
            Values over 1 are possible, example if [0,1], [0,2] are always formed (=1)
            freqs_dict[0]=2

        """
        dict_sum = defaultdict(list)
        for (idx1, idx2), ifreq in zip(self.res_idxs_pairs,
                                       self.frequency_per_contact(ctc_cutoff_Ang)):
            dict_sum[idx1].append(ifreq)
            dict_sum[idx2].append(ifreq)
        dict_sum = {key: _np.sum(val) for key, val in dict_sum.items()}
        return dict_sum

    def frequency_sum_per_residue_names_dict(self, ctc_cutoff_Ang,
                                             sort=True,
                                             list_by_interface=False,
                                             return_as_dataframe=False,
                                             fragsep="@"):
        r"""
        Dictionary of aggregated :obj:`frequency_per_contact` keyed
        by residue names, using the most informative label possible
        (ATM it is residue@frag, see :obj:`ContactPair.labels` for more info on this)
        TODO add option the type of residue name we are using

        Parameters
        ----------
        ctc_cutoff_Ang
        sort : bool, default is True
            Sort by dictionary by descending order of frequencies
            TODO dicts have order since py 3.6 and it is useful for creating
            TODO a dataframe, then excel_table that's already sorted by descending frequencies
        list_by_interface : bool, default is False, NotImplemented
            group the freq_dict by is_interface residues
        return_as_dataframe : bool, default is False
            Return an :obj:`pandas.DataFrame` with the column names labels and freqs
        fragsep : str, default is @
            String to separate residue@fragname
        Returns
        -------

        """
        freqs = self.frequency_sum_per_residue_idx_dict(ctc_cutoff_Ang)

        if list_by_interface:
            assert self.is_interface
            freqs = [{idx:freqs[idx] for idx in iint} for iint in self.interface_residxs]
        else:
            freqs = [freqs] #this way it is a list either way

        if sort:
            freqs = [{key:val for key, val in sorted(idict.items(),
                                                     key=lambda item: item[1],
                                                     reverse=True)}
                     for idict in freqs]

        # Use the residue@frag representation but avoid empty fragments
        dict_out = []
        for ifreq in freqs:
            idict = {}
            for idx, val in ifreq.items():
                key = self.residx2resnamefragnamebest()[idx]
                idict[key] = val
            dict_out.append(idict)

        if return_as_dataframe:
            dict_out = [_DF({"label": list(idict.keys()),
                             "freq": list(idict.values())}) for idict in dict_out]

        if len(dict_out)==1:
            dict_out = dict_out[0]
        return dict_out

    """"
    # TODO this seems to be unused
    def frequency_table_by_residue(self, ctc_cutoff_Ang,
                                   list_by_interface=False):
        dict_list = self.frequency_sum_per_residue_names_dict(ctc_cutoff_Ang,
                                                     list_by_interface=list_by_interface)

        if list_by_interface:
            label_bars = list(dict_list[0].keys()) + list(dict_list[1].keys())
            freqs = _np.array(list(dict_list[0].values()) + list(dict_list[1].values()))
        else:
            label_bars, freqs = list(dict_list.keys()), list(dict_list.values())

        return _DF({"label": label_bars,
                    "freq": freqs})
    """

    def frequency_dict_by_consensus_labels(self, ctc_cutoff_Ang,
                                           return_as_triplets=False,
                                           sort_by_interface=False,
                                           include_trilower=False):
        r"""
        Return frequencies as a dictionary of dictionaries keyed by consensus labels

        Note
        ----
        Will fail if not all residues have consensus labels
        TODO this is very similar to :obj:`frequency_sum_per_residue_names_dict`,
        look at the usecase closesely and try to unify both methods

        Parameters
        ----------
        ctc_cutoff_Ang
        return_as_triplets: bool, default is False
            Return as the dictionary as a list of triplets, s.t.
            freq_dict[3.50][4.50]=.25 is returned as
            [[3.50,4.50,.25]]
            Makes it easier to iterate through in other methods
        sort_by_interface
        include_trilower : bool, default is False
            Include the transposed indexes in the returned dictionary. s.t.
            the contact pair [3.50][4.50]=.25 also generates [4.50][3.50]=.25
        Returns
        -------
        freqs : dictionary of dictionary or list of triplets (if return_as_triplets is True)

        """
        assert not any ([ilab[0] is None and ilab[1] is None for ilab in self.consensus_labels])
        dict_out = defaultdict(dict)
        for (key1, key2), ifreq in zip(self.consensus_labels,
                                       self.frequency_per_contact(ctc_cutoff_Ang)):
            dict_out[key1][key2] = ifreq
            if include_trilower:
                dict_out[key2][key1] = ifreq

        dict_out = {key:val for key,val in dict_out.items()}

        if sort_by_interface:
            raise NotImplementedError
            # TODO the usecase for this is not clear to me ATM
            """
            _dict_out = {key:dict_out[key] for key in self.interface_labels_consensus[0] if key in dict_out.keys()}
            assert len(_dict_out)==len(dict_out)
            dict_out = _dict_out
            _dict_out = {key:{key2:val[key2] for key2 in self.interface_labels_consensus[1] if key2 in val.keys()} for key,val in dict_out.items()}
            assert all([len(val1)==len(val2) for val1, val2 in zip(dict_out.values(), _dict_out.values())])
            dict_out = _dict_out
            """

        if return_as_triplets:
            _dict_out = []
            for key, val in dict_out.items():
                for key2, val2 in val.items():
                    _dict_out.append([key, key2, val2])
            dict_out = _dict_out
        return dict_out

    def frequency_dataframe(self, ctc_cutoff_Ang,
                            by_atomtypes=False,
                            **ctc_fd_kwargs):
        r"""
        Output a formatted dataframe with fields "label", "freq" and "sum", optionally
        dis-aggregated by type of contact in "by_atomtypes"

        Note
        ----
        The contacts in the table are sorted by their order in the instantiation

        Parameters
        ----------
        ctc_cutoff_Ang
        by_atomtypes: bool, default is False
            Add a column where the contact is dis-aggregated by the atom-types involved,
            sidechain or backbone (SC or BB)
        ctc_fd_kwargs: named optional arguments
            Check :obj:`ContactPair.frequency_dict` for more info on e.g
            AA_format='short' and or split_label


        Returns
        -------
        df : :obj:`pandas.DataFrame`
        """
        idf = _DF([ictc.frequency_dict(ctc_cutoff_Ang, **ctc_fd_kwargs) for ictc in self._contacts])
        df2return = idf.join(_DF(idf["freq"].values.cumsum(), columns=["sum"]))

        if by_atomtypes:
            idf = self.relative_frequency_formed_atom_pairs_overall_trajs(ctc_cutoff_Ang)
            idf = ['(%s)' % (', '.join(['%2u%% %s' % (val * 100, key) for key, val in idict.items()])) for idict in idf]
            df2return = df2return.join(_DF.from_dict({"by_atomtypes": idf}))

        return df2return

    def frequency_spreadsheet(self, ctc_cutoff_Ang,
                              fname_excel,
                              sort=False,
                              write_interface=True,
                              offset=0,
                              sheet1_name="pairs by frequency",
                              sheet2_name='residues by frequency',
                              **freq_dataframe_kwargs):
        r"""
        Write an Excel file with the :obj:`pandas.Dataframe` that is
        returned by :obj:`self.frequency_dataframe`. You can
        control that call with obj:`freq_dataframe_kwargs`

        Parameters
        ----------
        ctc_cutoff_Ang
        fname_excel
        sort : bool, default is True
            Sort by descing order of frequency
        write_interface: bool, default is True
            Treat contact group as is_interface
        offset : int, default is 0
            First line at which to start writing the table. For future devleopment
            TODO do not expose this, perhaps?
        freq_dataframe_kwargs: dict, default is {}
            Optional arguments to :obj:`self.frequency_dataframe`, like by_atomtypes (bool)

        Returns
        -------

        """

        main_DF = self.frequency_dataframe(ctc_cutoff_Ang, **freq_dataframe_kwargs)

        columns = ["label",
                   "freq",
                   "sum",
                   ]
        if "by_atomtypes" in freq_dataframe_kwargs.keys() and freq_dataframe_kwargs["by_atomtypes"]:
            columns += ["by_atomtypes"]

        writer = _ExcelWriter(fname_excel, engine='xlsxwriter')
        workbook = writer.book
        writer.sheets[sheet1_name] = workbook.add_worksheet(sheet1_name)
        writer.sheets[sheet1_name].write_string(0, offset,
                                      'pairs by contact frequency at %2.1f Angstrom' % ctc_cutoff_Ang)
        offset+=1
        main_DF.round({"freq": 2, "sum": 2}).to_excel(writer,
                                                      index=False,
                                                      sheet_name=sheet1_name,
                                                      startrow=offset,
                                                      startcol=0,
                                                      columns=columns,
                                                      )
        offset = 0
        writer.sheets[sheet2_name] = workbook.add_worksheet(sheet2_name)
        writer.sheets[sheet2_name].write_string(offset, 0, 'Av. # ctcs (<%2.1f Ang) by residue '%ctc_cutoff_Ang)

        offset += 1

        idfs = self.frequency_sum_per_residue_names_dict(ctc_cutoff_Ang,
                                                         sort=sort,
                                                         list_by_interface=write_interface,
                                                         return_as_dataframe=True)
        if not write_interface:
            idfs=[idfs]
        idfs[0].round({"freq": 2}).to_excel(writer,
                                            sheet_name=sheet2_name,
                                            startrow=offset,
                                            startcol=0,
                                            columns=[
                                                "label",
                                                "freq"],
                                            index=False
                                            )
        if write_interface:
            #Undecided about best placement for these
            idfs[1].round({"freq": 2}).to_excel(writer,
                                                     sheet_name=sheet2_name,
                                                     startrow=offset,
                                                     startcol=2+1,
                                                     columns=[
                                                         "label",
                                                         "freq"],
                                                     index=False
                                                     )

        writer.save()

    def frequency_as_contact_matrix(self,
                                    ctc_cutoff_Ang):
        r"""
        Returns a symmetrical, square matrix of
        size :obj:`top`.n_residues containing the
        frequencies of the pairs in :obj:`residxs_pairs`,
        and those pairs only, the rest will be NaNs

        If :obj:`top` is None the method will fail.

        Note
        ----
            This is NOT the full contact matrix unless
            all necessary residue pairs were used to
            construct this ContactGroup

        Parameters
        ----------
        ctc_cutoff_Ang

        Returns
        -------
        mat : numpy.ndarray

        """

        mat = _np.zeros((self.top.n_residues, self.top.n_residues))
        mat[:, :] = _np.nan
        for (ii, jj), freq in zip(self.res_idxs_pairs, self.frequency_per_contact(ctc_cutoff_Ang)):
            mat[ii, jj] = freq
            mat[jj, ii] = freq

        return mat

    def relative_frequency_formed_atom_pairs_overall_trajs(self, ctc_cutoff_Ang):
        r"""

        Parameters
        ----------
        ctc_cutoff_Ang

        Returns
        -------
        refreq_dicts : list of dicts
        """
        return [ictc.relative_frequency_of_formed_atom_pairs_overall_trajs(ctc_cutoff_Ang) for ictc in self._contacts]

    def distributions_of_distances(self, nbins=10):
        r"""
        Histograms each the distance values of each contact,
        returning a list with as many distributions as there
        are contacts.

        Parameters
        ----------
        nbins : int, default is 10

        Returns
        -------
        list_of_distros : list
            List of len self.n_ctcs, each entry contains
            the counts and edges of the bins
        """
        return [ictc.distro_overall_trajs(bins=nbins) for ictc in self._contacts]

    def n_ctcs_timetraces(self, ctc_cutoff_Ang):
        r"""
        time-traces of the number of contacts, by summing overall contacts for
        each frame

        Parameters
        ----------
        ctc_cutoff_Ang

        Returns
        -------
        nctc_trajs : list of 1D np.ndarrays

        """
        bintrajs = self.binarize_trajs(ctc_cutoff_Ang, order='traj')
        _n_ctcs_t = []
        for itraj in bintrajs:
            _n_ctcs_t.append(itraj  .sum(1))
        return _n_ctcs_t

    #TODO this is WIP, not in use
    #def add_ctc_type_to_histo(self, ctc_cutoff_Ang, jax):
    #    ctc_type_dict = self.relative_frequency_formed_atom_pairs_overall_trajs(ctc_cutoff_Ang)
    #    print(ctc_type_dict)
    #    pass
        #relative_frequency_of_formed_atom_pairs_overall_trajs

    def _plot_freqbars_baseplot(self, ctc_cutoff_Ang,
                                jax=None,
                                truncate_at=None,
                                bar_width_in_inches=.75,
                                ):
        r"""
        Base method for plotting the contact frequencies of the contacts
        contained in this object as bar plots

        Parameters
        ----------
        ctc_cutoff_Ang: float
        jax : :obj:`matplotlib.Axes`, default is None
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
        jax : :obj:`matplotlib.Axes`
        """

        freqs = self.frequency_per_contact(ctc_cutoff_Ang)
        if truncate_at is not None:
            freqs = freqs[freqs>truncate_at]
        xvec = _np.arange(len(freqs))
        if jax is None:
            _plt.figure(figsize=(_np.max((7,bar_width_in_inches*len(freqs))),5))
            jax = _plt.gca()

        patches = jax.bar(xvec, freqs,
                          # label=res_and_fragment_str,
                          width=.25)
        jax.set_yticks([.25, .50, .75, 1])
        jax.set_ylim([0, 1])
        jax.set_xticks([])
        [jax.axhline(ii, color="lightgray", linestyle="--", zorder=-1) for ii in [.25, .50, .75]]
        return jax

    def plot_freqs_as_bars(self,
                           ctc_cutoff_Ang,
                           title_label,
                           xlim=None,
                           jax=None,
                           shorten_AAs=False,
                           label_fontsize_factor=1,
                           truncate_at=None):
        r"""
        Plot a contact frequencies as a bar plot

        Parameters
        ----------
        ctc_cutoff_Ang : float
        title_label : str
        xlim : float, default is None
        jax : :obj:`matplotlib.pyplot.Axes`
        shorten_AAs : bool, default is None
        label_fontsize_factor : float
        truncate_at : float, default is None

        Returns
        -------
        jax : :obj:`matplotlib.pyplot.Axes`

        """
        # Base plot
        jax = self._plot_freqbars_baseplot(ctc_cutoff_Ang,
                                           jax=jax, truncate_at=truncate_at)
        # Cosmetics
        jax.set_title(
            "Contact frequency @%2.1f $\AA$ of site '%s'\n"
            % (ctc_cutoff_Ang, title_label))

        label_bars = [ictc.labels.w_fragments for ictc in self._contacts]
        if shorten_AAs:
            label_bars = [ictc.labels.w_fragments_short_AA for ictc in self._contacts]

        # TODO fragment names got changed (i thinkI to never return Nones,
        # this shouldn't be necessary anymore
        #label_bars = [ilab.replace("@None","") for ilab in label_bars]

        _add_tilted_labels_to_patches(jax,
                                      label_bars[:(jax.get_xlim()[1]).astype(int)+1],
                                      label_fontsize_factor=label_fontsize_factor
                                      )

        #jax.legend(fontsize=_rcParams["font.size"] * label_fontsize_factor)
        if xlim is not None:
            jax.set_xlim([-.5, xlim + 1 - .5])

        return jax

    def plot_neighborhood_freqs(self, ctc_cutoff_Ang,
                                n_nearest,
                                xmax=None,
                                jax=None,
                                shorten_AAs=False,
                                label_fontsize_factor=1,
                                sum_freqs=True):
        r"""
        Neighborhood-aware frequencies bar plot for this contact group
        
        Parameters
        ----------
        ctc_cutoff_Ang : float
        n_nearest : int
        xmax : int, default is None
            Default behaviour is to go to n_ctcs, use this
            parameter to homogenize different calls to this
            function over different contact groups, s.t.
            each subplot has equal xlimits
        jax
        shorten_AAs
        label_fontsize_factor
        sum_freqs: bool, default is True
            Add the sum of frequencies of the represented (and only those)
            frequencies

        Returns
        -------
        jax : :obj:`matplotlib.pyplot.Axes`
        """

        # Base plot
        jax = self._plot_freqbars_baseplot(ctc_cutoff_Ang,
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

        if sum_freqs:
            # HACK to avoid re-computing the frequencies
            label_dotref +='\n$\Sigma$ = %2.1f'%_np.sum([ipatch.get_height() for ipatch in jax.patches])

        jax.plot(-1, -1, 'o',
                 color=self.anchor_fragment_color,
                 label=_replace4latex(label_dotref))

        _add_tilted_labels_to_patches(jax,
                                      label_bars,
                                      label_fontsize_factor=label_fontsize_factor)

        jax.legend(fontsize=_rcParams["font.size"]*label_fontsize_factor)
        if xmax is not None:
            jax.set_xlim([-.5, xmax + 1 - .5])

        return jax

    def plot_neighborhood_distributions(self,
                                        nbins=10,
                                        xlim=None,
                                        jax=None,
                                        shorten_AAs=False,
                                        ctc_cutoff_Ang=None,
                                        n_nearest=None,
                                        label_fontsize_factor=1,
                                        max_handles_per_row=4):

        r"""
        Plot distance distributions for the distance trajectories
        of the contacts

        Parameters
        ----------
        nbins : int, default is 10
            How many bins to use for the distribution
        xlim : iterable of two floats, default is None
            Limits of the x-axis.
            Outlier can stretch the scale, this forces it
            to a given range
        jax : :obj:`matplotlib.pyplot.Axes`, default is None
            One will be created if None is passed
        shorten_AAs: bool, default is False
            Use amino-acid one-letter codes
        ctc_cutoff_Ang: float, default is None
            Include in the legend of the plot how much of the
            distribution is below this cutoff. A vertical line
            will be draw at this x-value
        n_nearest : int, default is None
            Add a line to the title specifying if any
            nearest bonded neighbors were excluded
        label_fontsize_factor
        max_handles_per_row: int, default is 4
            legend control

        Returns
        -------
        jax : :obj:`matplotlib.pyplot.Axes`

        """
        if jax is None:
            _plt.figure(figsize=(7, 5))
            jax = _plt.gca()

        label_dotref = self.anchor_res_and_fragment_str
        label_bars = self.partner_res_and_fragment_labels
        if shorten_AAs:
            label_dotref = self.anchor_res_and_fragment_str_short
            label_bars = self.partner_res_and_fragment_labels_short

        # Cosmetics
        title_str = "distribution for %s"%_replace4latex(label_dotref)
        if ctc_cutoff_Ang is not None:
            title_str += "\nclosest residues <= @%2.1f $\AA$"%(ctc_cutoff_Ang)
            jax.axvline(ctc_cutoff_Ang,color="k",ls="--",zorder=-1)
        if n_nearest is not None:
            title_str += "\n%u nearest bonded neighbors excluded" % (n_nearest)
        jax.set_title(title_str)

        # Base plot
        for ii, ((h, x), label) in enumerate(zip(self.distributions_of_distances(nbins=nbins), label_bars)):
            if ctc_cutoff_Ang is not None:
                if ii==0:
                    freqs = self.frequency_per_contact(ctc_cutoff_Ang)
                label+=" (%u%%)"%(freqs[ii]*100)
            jax.plot(x[:-1] * 10, h, label=label)
            jax.fill_between(x[:-1]*10, h, alpha=.15)
        if xlim is not None:
            jax.set_xlim(xlim)

        jax.set_xlabel("D / $\AA$")
        jax.set_ylabel("counts ")
        jax.set_ylim([0,jax.get_ylim()[1]])
        jax.legend(fontsize=_rcParams["font.size"]*label_fontsize_factor/self.n_ctcs**.25,
                   ncol=_np.ceil(self.n_ctcs / max_handles_per_row).astype(int),
                   loc=1,
                   )
        jax.figure.tight_layout()

        return jax

    def plot_timedep_ctcs(self, panelheight,
                          plot_N_ctcs=True,
                          pop_N_ctcs=False,
                          skip_timedep=False,
                          **plot_contact_kwargs,
                          ):
        r"""
        For each trajectory, plot the time-traces of the all the contacts
        and/or/ the timetrace of the overall number of contacts

        Parameters
        ----------
        panelheight : int
        plot_N_ctcs : bool, default is True
            Add an extra panel at the bottom of the figure containing
            the number of formed contacts for each frame for each trajecotry
            A valid cutoff has to be passed along in :obj:`plot_contact_kwargs`
            otherwise this has no effect
        pop_N_ctcs : bool, default is False
            Put the panel with the number of contacts in a separate figure
            A valid cutoff has to be passed along in :obj:`plot_contact_kwargs`
            otherwise this has no effect
        skip_timedep : bool, default is False
            Skip plotting the individual timetraces and plot only
            the time trace of overall formed contacts. This sets
            pop_N_ctcs to True internally
        plot_contact_kwargs

        Returns
        -------
        list_of_figs : list
            The wanted figure(s)

         Note
        ----
            The keywords :obj:`plot_N_ctcs`, :obj:`pop_N_ctcs`, and :obj:`skip_timedep`
            allow this method to both include or totally exclude the total
            number of contacts and/or the time-traces in the figure.
            This might change in the figure,
            it was coded this way to avoid breaking the command_line tools
            API. Also note that some combinations will produce an empty return!


        """
        valid_cutoff = "ctc_cutoff_Ang" in plot_contact_kwargs.keys() \
                       and plot_contact_kwargs["ctc_cutoff_Ang"] > 0

        figs_to_return = []
        if skip_timedep:
            pop_N_ctcs = True

        if pop_N_ctcs:
            assert plot_N_ctcs, "If just_N_ctcs is True, plot_N_ctcs has to be True also"

            fig_N_ctcs = _plt.figure(
                figsize=(10, panelheight),
            )
            ax_N_ctcs = _plt.gca()


        if self.n_ctcs > 0 and not skip_timedep:
            n_rows = self.n_ctcs
            if plot_N_ctcs and valid_cutoff and not pop_N_ctcs:
                n_rows +=1
            myfig, myax = _plt.subplots(n_rows, 1,
                                        figsize=(10, n_rows * panelheight),
                                        squeeze=False)
            figs_to_return.append(myfig)
            myax = myax[:,0]
            axes_iter = iter(myax)

            # Plot individual contacts
            for ictc in self._contacts:
                ictc.plot_timetrace(next(axes_iter),
                             **plot_contact_kwargs
                             )

            # Cosmetics
            [iax.set_xticklabels([]) for iax in myax[:self.n_ctcs-1]]
            [iax.set_xlabel('') for iax in myax[:self.n_ctcs - 1]]
            # TODO figure out how to put xticklabels on top
            axtop, axbottom = myax[0], myax[-1]
            iax2 = axtop.twiny()
            iax2.set_xticks(axbottom.get_xticks())
            iax2.set_xticklabels(axbottom.get_xticklabels())
            iax2.set_xlim(axtop.get_xlim())
            iax2.set_xlabel(axbottom.get_xlabel())

        if valid_cutoff:
            if plot_N_ctcs:
                if pop_N_ctcs:
                    iax = ax_N_ctcs
                    figs_to_return.append(fig_N_ctcs)
                else:
                    iax = next(axes_iter)
            ctc_cutoff_Ang = plot_contact_kwargs.pop("ctc_cutoff_Ang")
            for pkey in ["shorten_AAs", "ylim_Ang"]:
                try:
                    plot_contact_kwargs.pop(pkey)
                except KeyError:
                    pass
            self._plot_timedep_Nctcs(iax,
                                     ctc_cutoff_Ang,
                                     **plot_contact_kwargs,
                                     )
        [ifig.tight_layout(pad=0, h_pad=0, w_pad=0) for ifig in figs_to_return]
        return figs_to_return

    def _plot_timedep_Nctcs(self,
                            iax,
                            ctc_cutoff_Ang,
                            color_scheme=None,
                            dt=1, t_unit="ps",
                            n_smooth_hw=0,
                            gray_background=False,
                            max_handles_per_row=4,
                            ):
        #Plot ncontacts in the last frame
        if color_scheme is None:
            color_scheme = _rcParams['axes.prop_cycle'].by_key()["color"]
        color_scheme = _np.tile(color_scheme, _np.ceil(self.n_trajs / len(color_scheme)).astype(int) + 1)
        icol = iter(color_scheme)
        for n_ctcs_t, itime, traj_name in zip(self.n_ctcs_timetraces(ctc_cutoff_Ang),
                                              self.time_arrays,
                                              self.trajlabels):
            plot_w_smoothing_auto(iax, itime*dt, n_ctcs_t,traj_name,next(icol),
                                  gray_background=gray_background,
                                  n_smooth_hw=n_smooth_hw)

        iax.set_ylabel('$\sum$ [ctcs < %s $\AA$]'%(ctc_cutoff_Ang))
        iax.set_xlabel('t / %s'%t_unit)
        iax.set_xlim([0,self.time_max*dt])
        iax.set_ylim([0,iax.get_ylim()[1]])
        iax.legend(fontsize=_rcParams["font.size"]*.75,
                   ncol=_np.ceil(self.n_trajs / max_handles_per_row).astype(int),
                   loc=1,
                   )

    def plot_frequency_sums_as_bars(self,
                                    ctc_cutoff_Ang,
                                    title_str,
                                    xmax=None,
                                    jax=None,
                                    shorten_AAs=False,
                                    label_fontsize_factor=1,
                                    truncate_at=0,
                                    bar_width_in_inches=.75,
                                    list_by_interface=False,
                                    sort=True,
                                    interface_vline=False):
        r"""
        bar plot with per-residue sums of frequencies (=nr. of neighbors?, cumulative freq? SIP?)
        TODO think about introducing the term nr of neighbors
        Parameters
        ----------
        ctc_cutoff_Ang : float
        title_str : str
        xmax : float, default is None
            X-axis will extend from -.5 to xmax+.5
        jax : obj:`matplotlib.pyplot.Axes`, default is None
            If None, one will be created, else draw here
        shorten_AAs : boolean, default is False
            Unused ATM
        label_fontsize_factor : float, default is 1
            Some control over fontsizes when plotting a high
            number of bars
        truncate_at : float, default is 0
            Do not show sums of freqs lower than this value
        bar_width_in_inches : float, default is .75
            If no :obj:`jax` is parsed, this controls that the
            drawn figure always has a size proportional to the
            number of frequencies being shown. Allows for
            combining multiple subplots with different number of bars
            in one figure with all bars equally wide regardles of
            the subplot
        list_by_interface : boolean, default is True
            Separate residues by is_interface
        sort : boolean, default is True
            Sort sums of freqs in descending order
        interface_vline : bool, default is False
            Plot a vertical line visually separating both interfaces

        Returns
        -------
        ax : :obj:`matplotlib.pyplot.Axes`
        """

        # Base dict
        freqs_dict = self.frequency_sum_per_residue_names_dict(ctc_cutoff_Ang,
                                                               sort=sort,
                                                               list_by_interface=list_by_interface)

        # TODO the method plot_freqs_as_bars is very similar but
        # i think it's better to keep them separated

        # TODO this code is repeated in table_by_residue
        if list_by_interface:
            label_bars = list(freqs_dict[0].keys())+list(freqs_dict[1].keys())
            freqs = _np.array(list(freqs_dict[0].values())+list(freqs_dict[1].values()))
        else:
            label_bars, freqs = list(freqs_dict.keys()),_np.array(list(freqs_dict.values()))

        # Truncate
        label_bars = [label_bars[ii] for ii in _np.argwhere(freqs>truncate_at).squeeze()]
        freqs = freqs[freqs>truncate_at]

        xvec = _np.arange(len(freqs))
        if jax is None:
            _plt.figure(figsize=(_np.max((7, bar_width_in_inches * len(freqs))), 5))
            jax = _plt.gca()

        patches = jax.bar(xvec, freqs,
                          width=.25)
        yticks = _np.arange(.5,_np.max(freqs)+.25, .5)
        jax.set_yticks(yticks)
        jax.set_xticks([],[])
        [jax.axhline(ii, color="lightgray", linestyle="--", zorder=-1) for ii in yticks]

        # Cosmetics
        jax.set_title(
            "Average nr. contacts @%2.1f $\AA$ \nper residue of contact group '%s'"
            % (ctc_cutoff_Ang, title_str))

        #TODO AFAIK this has been taken care of in the label-producing properties
        #label_bars = [ilab.replace("@None", "") for ilab in label_bars]

        _add_tilted_labels_to_patches(jax,
                                      label_bars[:(jax.get_xlim()[1]).astype(int) + 1],
                                      label_fontsize_factor=label_fontsize_factor,
                                      trunc_y_labels_at=.65*_np.max(freqs)
                                      )

        if xmax is not None:
            jax.set_xlim([-.5, xmax + 1 - .5])

        if list_by_interface and interface_vline:
            xpos = len([ifreq for ifreq in freqs_dict[0].values() if ifreq >truncate_at])
            jax.axvline(xpos-.5,color="lightgray", linestyle="--",zorder=-1)
        return jax

    @property
    def is_interface(self):
        r""" Whether this ContactGroup can be interpreted as an is_interface.

        Note that if none of the residxs_pairs parsed at initialization,
        were found in self.residxs_pairs, this property will evaluate to False
        """

        return self._interface

    @property
    def interface_residxs(self):
        r"""
        The residues split into the is_interface
        to that is_interface, in ascending order within each member
        of the is_interface. Empty lists mean non residues were
        found in the is_interface defined at initialization

        Returns
        -------

        """
        return self._interface_residxs


    @property
    def interface_reslabels_short(self):
        r"""
        Residue labels of whatever  residues :obj:`interface_residxs` holds
        Returns
        -------

        """
        labs = [[], []]
        for ii, ig in enumerate(self.interface_residxs):
            for idx in ig:
                labs[ii].append(self.residx2resnameshort[idx])

        return labs

    @property
    def interface_labels_consensus(self):
        r"""
        Consensus labels of whatever  residues :obj:`interface_residxs` holds.

        If there is no consensus labels, the corresponding label is None

        Returns
        -------

        """

        # TODO do I have anything like this anymore (with self.is_interface=False)
        #if self._interface_residxs is not None \
        #        and not hasattr(self, "_interface_labels_consensus"):
        labs = [[], []]
        for ii, ig in enumerate(self.interface_residxs):
            for idx in ig:
                labs[ii].append(self.residx2consensuslabel[idx])

        return labs
        #elif hasattr(self, "_interface_labels_consensus"):
        #    return self._interface_labels_consensus
        #else:
        #    return None

    @property
    def interface_residue_names_w_best_fragments_short(self):
        r"""
        Best possible residue@fragment string for the residues in :obj:`interface_residxs`

        In case neigher a consensus label > fragment name > fragment index is found,
        nothing is return after the residue name

        Returns
        -------

        """
        labs_out = []
        for ints in self.interface_residxs:
            labs_out.append([self.residx2resnamefragnamebest()[jj] for jj in ints])


        return labs_out

    @property
    def interface_orphaned_labels(self):
        r"""
        Short residue names that lack consensus nomenclature, sorted by is_interface
        Returns
        -------
        olist : list of len 2
        """
        #TODO eliminate the "orphan" label string but wait until the group of is_interface objecs is tested
        return [[self.residx2resnameshort[ii] for ii in idxs if ii in self._residxs_missing_conslabels]
                for idxs in self.interface_residxs]

    """ I am commenting all this until the is_interface UI is better
    def interface_relabel_orphans(self):
        labs_out = [[], []]
        for ii, labels in enumerate(self.interface_labels_consensus):
            for jlab in labels:
                if jlab in self._orphaned_residues_new_label.keys():
                    new_lab = self._orphaned_residues_new_label[jlab]
                    print(jlab, new_lab)
                    labs_out[ii].append(new_lab)
                else:
                    labs_out[ii].append(jlab)

        self._interface_labels_consensus = labs_out
        # return labs_out
    """

    """
    IDT we need this anymore
    def _freq_name_dict2_interface_dict(self, freq_dict_in,
                                        sort=True):
        assert self.is_interface
        dict_out = [{}, {}]
        for ii, ilabs in enumerate(self.interface_residue_names_w_best_fragments_short):
            # print(ilabs)
            if sort:
                for idx in freq_dict_in.keys():
                    if idx in ilabs:
                        dict_out[ii][idx] = freq_dict_in[idx]
            else:
                for jlab in ilabs:
                    dict_out[ii][jlab] = freq_dict_in[jlab]
        return dict_out
    """

    def plot_interface_frequency_matrix(self, ctc_cutoff_Ang,
                                        transpose=False,
                                        label_type='best',
                                        **plot_mat_kwargs,
                                        ):
        r"""
        Plot the :obj:`interface_frequency_matrix`

        The first group of :obj:`interface_residxs` are the row indices,
        shown in the y-axis top-to-bottom (since imshow is used to plot)
        The second group of :obj:`interface_residxs` are the column indices,
        shown in the x-axis left-to-right


        Parameters
        ----------
        ctc_cutoff_Ang
        transpose : bool, default is False
        label_type : str, default is "best"
            Best tries resname@consensus(>fragname>fragidx)
            Alternatives are "residue" or "consensus", but"consensus" alone
            might lead to empty labels since it is not guaranteed
            that all residues of the interface have consensus labels
        plot_mat_kwargs

        Returns
        -------
        iax : :obj:`matplotlib.pyplot.Axes`
        fig : :obj:`matplotlib.pyplot.Figure`

        """
        assert self.is_interface
        mat = self.interface_frequency_matrix(ctc_cutoff_Ang)
        if label_type=='consensus':
            labels = self.interface_labels_consensus
        elif label_type=='residue':
            labels = self.interface_reslabels_short
        elif label_type=='best':
            labels = self.interface_residue_names_w_best_fragments_short
        else:
            raise ValueError(label_type)

        iax, __ = _plot_contact_matrix(mat,labels,
                                       transpose=transpose,
                                       **plot_mat_kwargs,
                                       )
        return iax.figure, iax

    # TODO would it be better to make use of self.frequency_dict_by_consensus_labels
    def interface_frequency_matrix(self, ctc_cutoff_Ang):
        r"""
        Rectangular matrix of size (N,M) where N is the length
        of :obj:`interface_residxs`[0] and M the length of
        :obj:`interface_residxs`[1].

        Note
        ----
        Pairs missing from :obj:`res_idxs_pairs` will be NaNs,
        to differentitte from those pairs that were present
        but have zero contact

        Parameters
        ----------
        ctc_cutoff_Ang

        Returns
        -------
        mat : 2D numpy.ndarray
        """
        mat = None
        if self.is_interface:
            mat = self.frequency_as_contact_matrix(ctc_cutoff_Ang)
            mat = mat[self.interface_residxs[0],:]
            mat = mat[:,self.interface_residxs[1]]
        return mat

    def _to_per_traj_dicts_for_saving(self, t_unit="ps"):
        r"""
        For every trajectory return an dictionary with
        the data ready to be put into an ascii file,
        with pretty headers etc.

        The value "data" for every dict contains the
        distance values and has the shape (Nti,Ncts),
        where Nti is the number of frames of the i-th trajectory

        TODO: choose the type of contact label
        TODO: Think about using this wrapper for the plots

        Parameters
        ----------

        t_unit : str, default is "ps"
            The time unit to use for the labels. Alternatives are
            "ps", "ns", "mus" and "ms"

        Returns
        -------
        list_of_dicts with keys "header" and "data"
        """

        if t_unit not in _tunit2tunit.keys():
            raise ValueError("I don't know the time unit %s, only %s"%(t_unit, _tunit2tunit.keys()))
        dicts = []
        for ii in range(self.n_trajs):
            labels = ['time / %s'%t_unit]
            data = [self.time_arrays[ii] * _tunit2tunit["ps"][t_unit]]
            for ictc in self._contacts:
                labels.append('%s / Ang'%ictc.labels.w_fragments_short_AA)
                data.append(ictc.time_traces.ctc_trajs[ii]*10)
            data= _np.vstack(data).T
            dicts.append({"header":labels,
                          "data":data
                          }
                         )
        return dicts

    def _to_per_traj_dicts_for_saving_bintrajs(self, ctc_cutoff_Ang, t_unit="ps"):
        r"""
        For every trajectory return an dictionary with
        the data ready to be put into an ascii file,
        with pretty headers etc.

        Parameters
        ----------
        ctc_cutoff_Ang : float
        t_unit : str, default is "ps"
            The time unit to use for the labels. Alternatives are
            "ps", "ns", "mus" and "ms"

        Returns
        -------
        list_of_dicts with keys "header" and "data"

        """

        if t_unit not in _tunit2tunit.keys():
            raise ValueError("I don't know the time unit %s, only %s"%(t_unit, _tunit2tunit.keys()))

        bintrajs = self.binarize_trajs(ctc_cutoff_Ang, order="traj")
        labels = ['time / %s' % t_unit]
        for ictc in self._contacts:
            labels.append('%s / Ang' % ictc.label)

        dicts = []
        for ii in range(self.n_trajs):
            data = [self.time_arrays[ii]*_tunit2tunit["ps"][t_unit]]+[bintrajs[ii].T.astype(int)]
            data= _np.vstack(data).T
            dicts.append({"header":labels,
                          "data":data
                          }
                         )
        return dicts

    def save_trajs(self, prepend_filename,
                   ext,
                   output_dir='.',
                   t_unit="ps",
                   verbose=False,
                   ctc_cutoff_Ang=None,
                   self_descriptor="mdciaoCG"
                   ):
        r"""
        Save time-traces to disc.

        Filenames will be created based on the property
        :obj:`self.trajlabels`, but using only the basenames and
        prepending with the string :obj:`prepend_filename`

        If there is an anchor residue (i.e. this :obj:`ContactGroup`
        is a neighborhood, the anchor will be included in the filename,
        otherwise the string "contact_group" will be used.
        You can control the output_directory using :obj:`output_dir`

        If a ctc_cutoff is given, the time-traces will be binarized
        (see :obj:`self.binarize_trajs`). Else, the distances themselves
        are stored.

        Parameters
        ----------
        prepend_filename: str
            Each filename will be prepended with this string
        ext : str
            Extension, can be "xlsx" or anything :obj:`numpy.savetext`
            can handle
        output_dir: str, default is "."
            The output directory
        t_unit : str, default is "ps"
            Other units are "ns", "mus", and "ms". The transformation
            happens internally
        verbose: boolean, default is False
            Prints filenames
        ctc_cutoff_Ang: float, default is None
            Use this cutoff and save bintrajs instead
        self_descriptor : str, default is "mdciaoCG"

        Returns
        -------
        None
        """

        if ctc_cutoff_Ang is None:
            dicts = self._to_per_traj_dicts_for_saving(t_unit=t_unit)
        else:
            dicts = self._to_per_traj_dicts_for_saving_bintrajs(ctc_cutoff_Ang,t_unit=t_unit)

        if str(ext).lower()=="none":
            ext='dat'

        for idict, ixtc  in zip(dicts, self.trajlabels):
            ixtc_path, ixtc_basename = _path.split(ixtc)

            if self.shared_anchor_residue_index is not None:
                self_descriptor = self.anchor_res_and_fragment_str.replace('*', "")

            if ctc_cutoff_Ang is None:
                savename_fmt = "%s.%s.%s.%s"
            else:
                savename_fmt = "%s.%s.%s.bintrajs.%s"

            savename = savename_fmt % (prepend_filename.strip("."), self_descriptor.strip("."), ixtc_basename, ext.strip("."))
            savename = savename.replace(" ","_")
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

class group_of_interfaces(object):
    def __init__(self, dict_of_interfaces):
        self._interfaces = dict_of_interfaces


    @property
    def n_interfaces(self):
        return len(self.interfaces)

    @property
    def conlab2matidx(self):
        return [{key:ii for ii, key in enumerate(conlabs)} for conlabs in self.interface_labels_consensus]

    def interface_matrix(self,ctc_cutoff_Ang):
        labels = self.interface_labels_consensus
        mat = _np.zeros((len(labels[0]),len(labels[1])))
        conlab2matidx = self.conlab2matidx
        for key, iint in self.interfaces.items():
            idict = iint.frequency_dict_by_consensus_labels(ctc_cutoff_Ang)
            print(key)
            for key1, val in idict.items():
                for key2, val2 in val.items():
                    if key1 not in conlab2matidx[0]:
                        key1 = iint._orphaned_residues_new_label[key1]
                    ii, jj = conlab2matidx[0][key1], conlab2matidx[1][key2]
                    #print(key1,key2,val2)
                    mat[ii,jj] += val2

        mat = mat / self.n_interfaces

        return mat

    def frequency_dict_by_consensus_labels(self, ctc_cutoff_Ang,
                                           return_as_triplets=False,
                                           ):
        mat = self.interface_matrix(ctc_cutoff_Ang)
        dict_out = defaultdict(dict)
        for ii, jj in _np.argwhere(mat > 0):
            key1, key2 = self.interface_labels_consensus[0][ii], self.interface_labels_consensus[1][jj]
            dict_out[key1][key2] = mat[ii, jj]

        # Make a normal dictionary
        dict_out = {key: val for key, val in dict_out.items()}

        dict_pairs = []
        for key, val in dict_out.items():
            for key2,val2 in val.items():
                #print(key,key2,val2.round(2))
                dict_pairs.append((key,key2,val2))
        if return_as_triplets:
            return dict_pairs
        else:
            return dict_out

    def compare(self, ctc_cutoff_Ang, colordict, replacement_dict={},
                per_residue=False,
                **plot_unified_freq_dicts_kwargs):
        from .plots import plot_unified_freq_dicts
        freqs = {key:val.frequency_dict_by_consensus_labels(ctc_cutoff_Ang,
                                                            return_as_triplets=True
                                                            )
                                for key, val in self.interfaces.items()
                                }
        # Create and new dicts and do the replacements
        _freqs = {}
        for fkey, fval in freqs.items():
            if not per_residue:
                idict = {"-".join(sorted(triplet[:2])):triplet[-1] for triplet in fval}
            else:
                idict = self.interfaces[fkey].frequency_sum_per_residue_names_dict(ctc_cutoff_Ang)

            _freqs[fkey]={_replace_w_dict(key, replacement_dict):val for key, val in idict.items()}

        freqs =  unify_freq_dicts(_freqs)
        plot_unified_freq_dicts(freqs, colordict, **plot_unified_freq_dicts_kwargs)

    def frequency_table(self,ctc_cutoff_Ang):
        return self.frequency_dict_by_consensus_labels(ctc_cutoff_Ang, return_as_triplets=True)

    @property
    def interfaces(self):
        return self._interfaces

    @property
    def interface_labels_consensus(self):
        _interface_labels_consensus = [[], []]
        for key, interface in self.interfaces.items():
            for ii, ilabs in enumerate(interface.interface_labels_consensus):
                for jlab in ilabs:
                    if jlab not in _interface_labels_consensus[ii]:
                        _interface_labels_consensus[ii].append(jlab)
        from .nomenclature_utils import order_BW, order_CGN
        _interface_labels_consensus[0] = order_BW(_interface_labels_consensus[0])
        _interface_labels_consensus[1] = order_CGN(_interface_labels_consensus[1])
        return _interface_labels_consensus

    def plot_interface_matrix(self,ctc_cutoff_Ang,
                              annotate=True,
                              **kwargs_plot_interface_matrix):
        mat = self.interface_matrix(ctc_cutoff_Ang)
        iax, pixelsize = _plot_contact_matrix(mat,
                                              self.interface_labels_consensus,
                                              **kwargs_plot_interface_matrix)
        offset=8*pixelsize
        padding=pixelsize*2

        if annotate:
            n_x = len(self.interface_labels_consensus[1])
            for ii, (pdb, iint) in enumerate(self.interfaces.items()):
                xlabels = []
                for key in self.interface_labels_consensus[0]:
                    if key in iint.consensuslabel2resname.keys():
                        xlabels.append(iint.consensuslabel2resname[key])
                    elif hasattr(iint,"_orphaned_residues_new_label") and key in iint._orphaned_residues_new_label.values():
                        xlabels.append({val:key for key, val in iint._orphaned_residues_new_label.items()}[key])
                    else:
                        xlabels.append("-")
                ylabels = []
                for key in self.interface_labels_consensus[1]:
                    if key in iint.consensuslabel2resname.keys():
                        ylabels.append(iint.consensuslabel2resname[key])
                    else:
                        ylabels.append("-")
                y =  offset+ii+n_x+padding*ii
                x = -offset-ii    -padding*ii
                # todo transpose x,y,xlabels ylabels when needed accoding to kwargs
                iax.text(0-1, y,
                         pdb,
                         fontsize=pixelsize*20,
                         ha="right", va="bottom")
                iax.text(x,0-1,pdb,
                         fontsize=pixelsize*20)
                for jj, ilab in enumerate(xlabels):
                    pass
                    iax.text(jj, y,ilab,
                             fontsize=pixelsize*20,
                             rotation=90,
                             ha="center")
                for jj, ilab in enumerate(ylabels):
                    iax.text(x, jj, ilab,
                             fontsize=pixelsize*20,
                             va="center"
                             )

        return iax.figure, iax

    @property
    def PDBs(self):
        return list(self.interfaces.keys())


    def rename_orphaned_residues_foreach_interface(self,
                                                   alignment_as_DF,
                                                   interface_idx=0):

        assert interface_idx==0,NotImplementedError(interface_idx)

        long2short = lambda istr: ['@'.join(ival.split('@')[:2]) for ival in istr.split("_") if ival != 'None'][0]
        orphan_residues_by_short_label = defaultdict(dict)
        for pdb, iint in self.interfaces.items():
            iint._orphaned_residues_new_label = {}
            for AA in iint.interface_orphaned_labels[interface_idx]:
                line = self.dict_same_orphan_labels_by_alignemntDF(AA,alignment_as_DF,pdb)
                line = {key:line[key] for key in self.PDBs}
                long_key  = '_'.join([str(ival) for ival in line.values()])
                short_key = long2short(long_key)
                #print(pdb, AA, short_key)
                #print(long_key)
                iint._orphaned_residues_new_label [AA] = short_key
                assert pdb not in orphan_residues_by_short_label[short_key].keys()
                orphan_residues_by_short_label[short_key][pdb]=line[pdb]
                iint.interface_relabel_orphans()
        self._orphans_renamed = {key:val for key,val in orphan_residues_by_short_label.items()}

    @property
    def orphans_renamed(self):
        return self._orphans_renamed

    #todo move this outside?
    def dict_same_orphan_labels_by_alignemntDF(self,AA,aDF,pdb):
        hit = [(val, ii) for ii, val in enumerate(aDF[pdb].values) if str(val).split('@')[0] == AA]
        assert len(hit) == 1, hit
        hit = hit[0]
        return aDF.iloc[hit[1]].to_dict()

def _sum_ctc_freqs_by_atom_type(atom_pairs, counts):
    r"""
    Starting from a list of atom pairs and an associated list of counts
    representing a contact frequency, aggregate the frequencies by
    type of contact into "BB-BB", "SC-SC", "SC-BB", "BB-SC" depending
    on the atom-types involved in the contact.
    "BB" is backbone, "SC" is sidechain and "X" is unknown
    Parameters
    ----------
    atom_pairs : iterable of pairs of :obj:`mdtraj.core.Atom`-objects
    counts : iterable of ints or floats

    Returns
    -------
    count_dict : dictionary

    """
    atom_pairs = ['-'.join([_atom_type(aa) for aa in pair]) for pair in atom_pairs]
    dict_out = {key: 0 for key in atom_pairs}
    for key, count in zip(atom_pairs, counts):
        dict_out[key] += count
    return dict_out

# todo check that mdtraj's license allows for this
from mdtraj.utils import ensure_type
import itertools
from mdtraj.utils.six import string_types
from mdtraj.utils.six.moves import xrange
from mdtraj.core import element
def compute_contacts(traj, contacts='all', scheme='closest-heavy', ignore_nonprotein=True, periodic=True,
                     soft_min=False, soft_min_beta=20):
    """Compute the distance between pairs of residues in a trajectory.

    Parameters
    ----------
    traj : md.Trajectory
        An mdtraj trajectory. It must contain topology information.
    contacts : array-like, ndim=2 or 'all'
        An array containing pairs of indices (0-indexed) of residues to
        compute the contacts between, or 'all'. The string 'all' will
        select all pairs of residues separated by two or more residues
        (i.e. the i to i+1 and i to i+2 pairs will be excluded).
    scheme : {'ca', 'closest', 'closest-heavy', 'sidechain', 'sidechain-heavy'}
        scheme to determine the distance between two residues:
            'ca' : distance between two residues is given by the distance
                between their alpha carbons
            'closest' : distance is the closest distance between any
                two atoms in the residues
            'closest-heavy' : distance is the closest distance between
                any two non-hydrogen atoms in the residues
            'sidechain' : distance is the closest distance between any
                two atoms in residue sidechains
            'sidechain-heavy' : distance is the closest distance between
                any two non-hydrogen atoms in residue sidechains
    ignore_nonprotein : bool
        When using `contact==all`, don't compute contacts between
        "residues" which are not protein (i.e. do not contain an alpha
        carbon).
    periodic : bool, default=True
        If periodic is True and the trajectory contains unitcell information,
        we will compute distances under the minimum image convention.
    soft_min : bool, default=False
        If soft_min is true, we will use a diffrentiable version of
        the scheme. The exact expression used
         is d = \frac{\beta}{log\sum_i{exp(\frac{\beta}{d_i}})} where
         beta is user parameter which defaults to 20nm. The expression
         we use is copied from the plumed mindist calculator.
         http://plumed.github.io/doc-v2.0/user-doc/html/mindist.html
    soft_min_beta : float, default=20nm
        The value of beta to use for the soft_min distance option.
        Very large values might cause small contact distances to go to 0.

    Returns
    -------
    distances : np.ndarray, shape=(n_frames, n_pairs), dtype=np.float32
        Distances for each residue-residue contact in each frame
        of the trajectory
    residue_pairs : np.ndarray, shape=(n_pairs, 2), dtype=int
        Each row of this return value gives the indices of the residues
        involved in the contact. This argument mirrors the `contacts` input
        parameter. When `all` is specified as input, this return value
        gives the actual residue pairs resolved from `all`. Furthermore,
        when scheme=='ca', any contact pair supplied as input corresponding
        to a residue without an alpha carbon (e.g. HOH) is ignored from the
        input contacts list, meanings that the indexing of the
        output `distances` may not match up with the indexing of the input
        `contacts`. But the indexing of `distances` *will* match up with
        the indexing of `residue_pairs`

    Examples
    --------
    >>> # To compute the contact distance between residue 0 and 10 and
    >>> # residues 0 and 11
    >>> _md.compute_contacts(t, [[0, 10], [0, 11]])

    >>> # the itertools library can be useful to generate the arrays of indices
    >>> group_1 = [0, 1, 2]
    >>> group_2 = [10, 11]
    >>> pairs = list(itertools.product(group_1, group_2))
    >>> print(pairs)
    [(0, 10), (0, 11), (1, 10), (1, 11), (2, 10), (2, 11)]
    >>> _md.compute_contacts(t, pairs)

    See Also
    --------
    mdtraj.geometry.squareform : turn the result from this function
        into a square "contact map"
    Topology.residue : Get residues from the topology by index
    """
    if traj.topology is None:
        raise ValueError('contact calculation requires a topology')

    if isinstance(contacts, string_types):
        if contacts.lower() != 'all':
            raise ValueError('(%s) is not a valid contacts specifier' % contacts.lower())

        residue_pairs = []
        for i in xrange(traj.n_residues):
            residue_i = traj.topology.residue(i)
            if ignore_nonprotein and not any(a for a in residue_i.atoms if a.name.lower() == 'ca'):
                continue
            for j in xrange(i+3, traj.n_residues):
                residue_j = traj.topology.residue(j)
                if ignore_nonprotein and not any(a for a in residue_j.atoms if a.name.lower() == 'ca'):
                    continue
                if residue_i.chain == residue_j.chain:
                    residue_pairs.append((i, j))

        residue_pairs = _np.array(residue_pairs)
        if len(residue_pairs) == 0:
            raise ValueError('No acceptable residue pairs found')

    else:
        residue_pairs = ensure_type(_np.asarray(contacts), dtype=_np.int, ndim=2, name='contacts',
                                    shape=(None, 2), warn_on_cast=False)
        if not _np.all((residue_pairs >= 0) * (residue_pairs < traj.n_residues)):
            raise ValueError('contacts requests a residue that is not in the permitted range')

    # now the bulk of the function. This will calculate atom distances and then
    # re-work them in the required scheme to get residue distances
    scheme = scheme.lower()
    if scheme not in ['ca', 'closest', 'closest-heavy', 'sidechain', 'sidechain-heavy']:
        raise ValueError('scheme must be one of [ca, closest, closest-heavy, sidechain, sidechain-heavy]')

    if scheme == 'ca':
        if soft_min:
            import warnings
            warnings.warn("The soft_min=True option with scheme=ca gives"
                          "the same results as soft_min=False")
        filtered_residue_pairs = []
        atom_pairs = []

        for r0, r1 in residue_pairs:
            ca_atoms_0 = [a.index for a in traj.top.residue(r0).atoms if a.name.lower() == 'ca']
            ca_atoms_1 = [a.index for a in traj.top.residue(r1).atoms if a.name.lower() == 'ca']
            if len(ca_atoms_0) == 1 and len(ca_atoms_1) == 1:
                atom_pairs.append((ca_atoms_0[0], ca_atoms_1[0]))
                filtered_residue_pairs.append((r0, r1))
            elif len(ca_atoms_0) == 0 or len(ca_atoms_1) == 0:
                # residue does not contain a CA atom, skip it
                if contacts != 'all':
                    # if the user manually asked for this residue, and didn't use "all"
                    import warnings
                    warnings.warn('Ignoring contacts pair %d-%d. No alpha carbon.' % (r0, r1))
            else:
                raise ValueError('More than 1 alpha carbon detected in residue %d or %d' % (r0, r1))

        residue_pairs = _np.array(filtered_residue_pairs)
        distances = _md.compute_distances(traj, atom_pairs, periodic=periodic)
        aa_pairs = atom_pairs

    elif scheme in ['closest', 'closest-heavy', 'sidechain', 'sidechain-heavy']:
        if scheme == 'closest':
            residue_membership = [[atom.index for atom in residue.atoms]
                                  for residue in traj.topology.residues]
        elif scheme == 'closest-heavy':
            # then remove the hydrogens from the above list
            residue_membership = [[atom.index for atom in residue.atoms if not (atom.element == element.hydrogen)]
                                  for residue in traj.topology.residues]
        elif scheme == 'sidechain':
            residue_membership = [[atom.index for atom in residue.atoms if atom.is_sidechain]
                                  for residue in traj.topology.residues]
        elif scheme == 'sidechain-heavy':
            # then remove the hydrogens from the above list
            residue_membership = [[atom.index for atom in residue.atoms if atom.is_sidechain and not (atom.element == element.hydrogen)]
                                  for residue in traj.topology.residues]

        residue_lens = [len(ainds) for ainds in residue_membership]

        atom_pairs = []
        n_atom_pairs_per_residue_pair = []
        for pair in residue_pairs:
            atom_pairs.extend(list(itertools.product(residue_membership[pair[0]], residue_membership[pair[1]])))
            n_atom_pairs_per_residue_pair.append(residue_lens[pair[0]] * residue_lens[pair[1]])

        atom_distances = _md.compute_distances(traj, atom_pairs, periodic=periodic)

        # now squash the results based on residue membership
        n_residue_pairs = len(residue_pairs)
        distances = _np.zeros((len(traj), n_residue_pairs), dtype=_np.float32)
        n_atom_pairs_per_residue_pair = _np.asarray(n_atom_pairs_per_residue_pair)

        aa_pairs = []
        for i in xrange(n_residue_pairs):
            index = int(_np.sum(n_atom_pairs_per_residue_pair[:i]))
            n = n_atom_pairs_per_residue_pair[i]
            if not soft_min:
                idx_min = atom_distances[:, index : index + n].argmin(axis=1)
                aa_pairs.append(_np.array(atom_pairs[index: index + n])[idx_min])
                # TODO do not call the min function again here, call idx_min
                distances[:, i] = atom_distances[:, index : index + n].min(axis=1)
            else:
                distances[:, i] = soft_min_beta / \
                                  _np.log(_np.sum(_np.exp(soft_min_beta /
                                                       atom_distances[:, index : index + n]), axis=1))

    else:
        raise ValueError('This is not supposed to happen!')

    return distances, residue_pairs, _np.hstack(aa_pairs)
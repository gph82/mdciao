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
from os import path as _path

import mdciao.plots as _mdcplots
import mdciao.utils as _mdcu
import mdciao.nomenclature as _mdcn

import mdciao.flare as _mdcflare

from ._md_compute_contacts import compute_contacts as _compute_contacts

from pickle import dump as _pdump,load as _pload

from copy import deepcopy as _deepcopy

from collections import \
    defaultdict as _defdict, \
    Counter as _col_Counter

from tqdm import tqdm as _tqdm

from matplotlib import \
    pyplot as _plt,\
    rcParams as _rcParams

from pandas import \
    DataFrame as _DF, \
    ExcelWriter as _ExcelWriter

from joblib import \
    Parallel as _Parallel, \
    delayed as _delayed

def select_and_report_residue_neighborhood_idxs(ctc_freqs, res_idxs, fragments,
                                                residxs_pairs, top,
                                                ctcs_kept=5,
                                                restrict_to_resSeq=None,
                                                interactive=False,
                                                fraction=.9
                                                ):
    """Group residue pairs into neighborhoods using pre-computed contact frequencies

    Returns a residue-index keyed dictionary containing the indices
    of :obj:`residxs_pairs` relevant for this residue.

    Can be used interactively to decide on-the-fly which residues to
    include in the neighborhood..

    Parameters
    ----------
    ctc_freqs: iterable of floats
        Contact frequencies between 0 and 1
    res_idxs: list of integers
        list of residue idxs for which one wants to extract the neighborhoods
    fragments: iterable of integers
        Fragments of the topology defined as list of non-overlapping residue indices
    residxs_pairs: iterable of integer pairs
        The residue pairs for which the contact frequencies in :obj:`ctc_freqs`
        were computed.
    top : :obj:`~mdtraj.Topology`
        The topology from which the residues come
    ctcs_kept : integer or float, default is 5
        Control how many contacts to report per residue. There's two
        types of behaviour:
        * If int, it means directly keep these many contacts
        * if float, it must be in in [0,1] and represents a fraction
          of the total number of contacts to keep
    restrict_to_resSeq: iterable, default is None
        Only cycle through the residues in :obj:`res_idxs` with these resSeq indices.
    interactive : boolean, default is False
        After reporting each neighborhood up to :obj:`ctcs_kept`,
        ask the user how many should be kept
    fraction : float, default is .9
        report how many contacts one needs to keep
        to arrive at this fraction of the overall contacts.

    Returns
    -------
    selection : dictionary
        Dictionary keyed with residue indices and valued with lists of
        indices for :obj:`residxs_pairs` s.t. for example:

           selection[30] = [100,200]

        means that for the residue with the index 300, the :obj:`residxs_pairs`
        on the 100-th and 200-th position, e.g. contain the pairs for its most
        frequent neighbors, e.g. [30-45] and [30-145].
        :obj:`ctcs_kept` controls the length of the output, see also option 'interactive')
        Each i-th list is sorted by descending
        frequency of the contacts of residue-i
        and is truncated at freq==0 regardless
        of :obj:`ctcs_kept`
    """
    assert len(ctc_freqs) == len(residxs_pairs)

    order = _np.argsort(ctc_freqs)[::-1]
    selection = {}
    ctc_freqs = _np.array(ctc_freqs)
    if restrict_to_resSeq is None:
        restrict_to_resSeq = [top.residue(ii).resSeq for ii in res_idxs]

    elif isinstance(restrict_to_resSeq, int):
        restrict_to_resSeq = [restrict_to_resSeq]
    for residx in res_idxs:
        resSeq = top.residue(residx).resSeq
        _fraction = fraction
        if resSeq in restrict_to_resSeq:
            order_mask = _np.array([ii for ii in order if residx in residxs_pairs[ii]],dtype=int)
            print("#idx   freq      contact       fragments     res_idxs      ctc_idx  Sum")
            isum = 0
            seen_ctcs = []
            total_n_ctcs = _np.array(ctc_freqs)[order_mask].sum()
            if float(ctcs_kept).is_integer():
                n_ctcs = int(ctcs_kept)
            else:
                if total_n_ctcs>0:
                    n_ctcs = _idx_at_fraction(ctc_freqs[order_mask], ctcs_kept)+1
                    _fraction = None
                else:
                    n_ctcs = 0
            for ii, oo in enumerate(order_mask[:n_ctcs]):
                ifreq = ctc_freqs[oo]
                if ifreq.round(2)==0:
                    break
                isum += ifreq
                pair = residxs_pairs[oo]
                idx1, idx2 = _mdcu.lists.put_this_idx_first_in_pair(residx, pair)
                frg1, frg2 = [_mdcu.lists.in_what_fragment(idx, fragments) for idx in [idx1,idx2]]
                seen_ctcs.append(ifreq)
                print("%-6s %3.2f %8s-%-8s %5u-%-5u %7u-%-7u %5u     %3.2f" % (
                 '%u:' % (ii + 1), ifreq, top.residue(idx1), top.residue(idx2), frg1, frg2, idx1, idx2, oo, isum))
            if n_ctcs>0:
                _contact_fraction_informer(_np.min([ii, len(order_mask)]), ctc_freqs[order_mask], or_frac=_fraction)
            else:
                print("No contacts here!")
            if interactive:
                try:
                    answer = input("How many do you want to keep (Hit enter for None)?\n")
                except KeyboardInterrupt:
                    break
                if len(answer) == 0:
                    pass
                else:
                    answer = _np.arange(_np.min((int(answer), ctcs_kept)))
                    selection[residx] = order_mask[answer]
            else:
                seen_ctcs = _np.array(seen_ctcs)
                n_nonzeroes = (seen_ctcs > 0).astype(int).sum()
                answer = _np.arange(_np.min((n_nonzeroes, n_ctcs)))
                selection[residx] = order_mask[answer]

    return selection

def _save_as_pickle(obj, filename,verbose=True):
    with open(filename, "wb") as f:
        _pdump(obj, f)
    if verbose:
        print("pickled %s to '%s'" % (obj, filename))

def load(filename,return_copy=True):
    r"""Load a pickled object

    Parameters
    ----------
    filaname : str
        path to pickled object
    return_copy : bool, default is True
        Issue obj.copy() before returning the pickled object,
        this forces a reinstantiation (the pickled object might
        not have all funcitons yet, although it could be that it does
    Returns
    -------

    """
    with open(filename,"rb") as f:
        obj = _pload(f)

    if return_copy:
        obj = obj.copy()

    return obj

def trajs2ctcs(trajs, top, ctc_residxs_pairs, stride=1, consolidate=True,
               chunksize=1000, return_times_and_atoms=False,
               n_jobs=1,
               progressbar=False,
               **mdcontacts_kwargs):
    """Time-traces of residue-residue distances from
    a list of trajectories

    Parameters
    ----------
    trajs : list
        list of trajectories. Each item can be a str
        with the path to a file or an
        :obj:`~mdtraj.Trajectory` object.
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
        Return also the time array in ps and the indices of the atoms
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
    ctcs :
    ctcs, time_trajs, atom_idxs if return_time=True

    """

    if progressbar:
        iterfunct = lambda a : _tqdm(a)
    else:
        iterfunct = lambda a : a
    assert isinstance(trajs,list) #otherwise we will iterate through the frames of a single traj
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
    Wrapper for :obj:`mdtraj.contacts` for strided, chunked computation
    of contacts

    Input can be directly :obj:`mdtraj.Trajectory` objects or
    trajectory files on disk (e.g. xtcs, dcs etc)

    You can fine-tune the computation itself using mdcontacts_kwargs

    Prints out progress report while working

    Parameters
    ----------
    top: `~mdtraj.Topology`
    itraj: `~mdtraj.Trajectory` or filename
    ctc_residxs_pairs: iterable of pairs of residue indices
        Distances to be computed
    chunksize: int
        Size (in frames) of the "chunks" in which the contacts will be computed.
        Decrease the chunksize if you run into memory errors
    stride:int
        Stride with which the contacts will be streamed over
    traj_idx: int
        The index of the trajectory being computed. For completeness
        of the progress report
    mdcontacts_kwargs:
        Optional keyword arguments to pass to :obj:`mdtraj.contacts`

        Note:
        -----
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
    # The creation of lambdas managing the file(xtc,pdb) vs traj case
    # elsewhere allows to keep the code here simple
    iterate, inform = _mdcu.str_and_dict.iterate_and_inform_lambdas(itraj, chunksize, stride=stride, top=top)
    ictcs, itime, iaps = [],[],[]
    running_f = 0
    inform(itraj, traj_idx, 0, running_f)
    for jj, igeom in enumerate(iterate(itraj)):
        running_f += igeom.n_frames
        inform(itraj, traj_idx, jj, running_f)
        itime.append(igeom.time)
            #TODO make lambda out of this if
        if 'scheme' in mdcontacts_kwargs.keys() and mdcontacts_kwargs["scheme"].upper()=='COM':
            jctcs = _mdcu.COM.geom2COMdist(igeom, ctc_residxs_pairs)
            j_atompairs = _np.full((len(jctcs), 2*len(ctc_residxs_pairs)),_np.nan)
        else:
            jctcs, jidx_pairs, j_atompairs = _compute_contacts(igeom, ctc_residxs_pairs, **mdcontacts_kwargs)
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

        _np.testing.assert_equal(len(time_trajs),len(ctc_trajs))
        self._ctc_trajs = [_np.array(itraj,dtype=float) for itraj in ctc_trajs]
        self._time_trajs =[_np.array(tt,dtype=float) for tt in time_trajs]
        self._trajs = trajs
        if trajs is not None:
            assert len(trajs)==len(ctc_trajs)
        self._atom_pair_trajs = atom_pair_trajs
        _np.testing.assert_array_equal([len(itraj) for itraj in ctc_trajs],
                                       [len(itime) for itime in time_trajs])
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

class Residues(object):
    r"""Container for :obj:`mdtraj.Topology.Residues` objects

    """

    def __init__(self, res_idxs_pair,
                 residue_names,
                 anchor_residue_idx=None,
                 consensus_labels=None,
                 top=None):
        r"""

        Parameters
        ----------
        res_idxs_pair : iterable of len two
            two integers representing the residue serial index (residxs)
        residue_names : iterable of len two
            two strings with the residue names
        anchor_residue_idx
        consensus_labels
        top
        """

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
            self._anchor_index = int(_np.argwhere(self.idxs_pair == self.anchor_residue_index))
            self._partner_index = int(_np.argwhere(self.idxs_pair != self.anchor_residue_index))
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
        """serial indices of the pair of residues

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
        return [_mdcu.residue_and_atom.shorten_AA(rr, substitute_fail="long", keep_index=True) for rr in self.names]

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
        The index [0,1] of the anchor residue

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
        """Labels derived from BW, CGN or other type
        of consensus nomenclature. They were parsed
        at initialization

        TODO

        Warning
        -------
        This property can be changed externally by
        the method :obj:`ContactGroup.relabel_consensus`.
        This is bad practice and probably anti-pattern
        but ATM there's no way around it

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
        return _mdcu.str_and_dict.choose_options_descencing([self.partner_fragment_consensus,
                                                             self.partner_fragment])

    @property
    def anchor_fragment_best(self):
        """

        Returns
        -------

        """
        return _mdcu.str_and_dict.choose_options_descencing([self.anchor_fragment_consensus,
                                                             self.anchor_fragment])

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
            self._fragnames_consensus = [None, None]
        else:
            self._fragnames = self._fragments.names
            self._fragnames_consensus = self._fragments.consensus

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

    @property
    def just_consensus(self):
        ctc_label = '%s-%s' % (self._residues.consensus_labels[0],
                                   self._residues.consensus_labels[1],
                                   )
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

        return [_mdcu.str_and_dict.choose_options_descencing([self._residues.consensus_labels[ii],
                                                              self._fragnames_consensus[ii],
                                                              self._fragnames[ii]],
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
                 consensus_fragnames=None
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

        if consensus_fragnames is None:
            self._consensus_fragnames = [None,None]
        else:
            assert len(consensus_fragnames)==2
            self._consensus_fragnames = consensus_fragnames


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

    @property
    def consensus(self):
        """
        name of fragments according to consensus nomenclature
        Returns
        -------

        """
        return self._consensus_fragnames

class ContactPair(object):
    r"""Container for a contacts between two residues

    This is the first level of abstraction of mdciao.
    It is the "closest" to the actual data, and its
    methods carry out most of the low-level operations on
    the data, e.g., the frequency calculations or the
    basic plotting. Other classes like :obj:`ContactGroup`
    usually just wrap around a collection of :obj:`ContactPair`-objects
    and use their methods.

    This class just needs the pair of residue (serial) indices,
    the time-traces of the distances between the residues
    (for all input trajectories), and the time-traces
    of the timestamps in those trajectories.

    Many other pieces of complementary information can be provided
    as optional parameters, allowing the class to produce
    better plots, labels, and tables.

    Some sanity checks are carried out upon instantiation to ensure
    things like same number of steps in the in the distance and timestamp
    time-traces.

    Note
    ----
    Higher-level methods in the API, like those exposed by :obj:`mdciao.cli`
    will return :obj:`ContactPair` or :obj:`ContactGroup` objects already
    instantiated and ready to use. It is recommened to use those instead
    of individually calling :obj:`ContactPair` or :obj:`ContactGroup`.

    """
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
                 consensus_labels=None,
                 consensus_fragnames=None):
        """

        Parameters
        ----------
        res_idxs_pair : iterable of two ints
            pair of residue indices, corresponding to the zero-indexed, serial number of the residues
        ctc_trajs : list of iterables of floats
            time traces of the contact in nm. len(ctc_trajs) is N_trajs. Each traj can have different lengths
            Will be cast into arrays.
        time_trajs : list of iterables of floats
            time traces of the time-values, in ps. Not having the same shape as ctc_trajs will raise an error
        top : :py:class:`mdtraj.Topology`, default is None
            topology associated with the contact
        trajs: list of :obj:`mdtraj.Trajectory` objects, default is None
            The molecular trajectories responsible for which the contact has been evaluated.
            Not having the same shape as ctc_trajs will raise an error
        atom_pair_trajs: list of iterables of integers, default is None
            Time traces of the pair of atom indices responsible for the distance in :obj:`ctc_trajs`
            Has to be of len(ctc_trajs) and each iterable of shape(Nframes, 2)
        fragment_idxs : iterable of two ints, default is None
            Indices of the fragments the residues of :obj:`res_idxs_pair`
        fragment_names : iterable of two strings, default is None
            Names of the fragments the residues of :obj:`res_idxs_pair`
        fragment_colors : iterable of len 2, default is None
            Colors associated to the fragments of the residues of :obj:`res_idxs_pair`. A color
            is anything that :obj:`matplotlib.colors` recognizes
        anchor_residue_idx : int, default is None
            Label this residue as the `anchor` of the contact, i.e. the residue
            that's shared across a number of contacts. Has to be in :obj:`res_idxs_pair`.

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
        consensus_fragnames : iterable of strings, default is None
            Consensus fragments names of the residues of :obj:`res_idxs_pair`

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

        self._attribute_residues = Residues(res_idxs_pair,
                                            residue_names,
                                            anchor_residue_idx=anchor_residue_idx,
                                            consensus_labels=consensus_labels,
                                            top=top)

        self._attribute_fragments = _Fragments(fragment_idxs,
                                               fragment_names,
                                               fragment_colors,
                                               consensus_fragnames)


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
        self._time_max = _np.nanmax(_np.hstack(time_trajs))
        self._time_min = _np.nanmin(_np.hstack(time_trajs))
        self._binarized_trajs = _defdict(dict)

    #Trajectories
    @property
    def time_traces(self):
        r"""
        Contains time-traces stored as a :obj:`_TimeTraces` objects
        Returns
        -------

        """
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
    def time_min(self):
        """

        Returns
        -------
        int or float, maximum time from list of list of time

        """
        return self._time_min
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

    def copy(self):
        r"""copy this object by re-instantiating another :obj:`ContactPair` object
        with the same attributes. In theory self == self.copy() should hold (but
        not self is self.copy()

        Returns
        -------
        CP : :obj:`ContactPair`

        """
        return self.retop(self.top, mapping={key:key for key in self.residues.idxs_pair})

    def retop(self,top, mapping, deepcopy=False, **CP_kwargs):
        r"""Return a copy of this object with a different topology.

        Uses the :obj:`mapping` to generate new residue- and
        and atom-indices where necessary, using the rest
        of the object's attributes (time-traces, labels, colors,
        fragments...) as they were.

        Note
        ----
        This method will (rightly) fail if:
         * the mapping doesn't contain the needed residues
         * the individual atoms of those residues cannot
           be uniquely mapped between topologies

        TODO
        ----
         * Generate mapping on-the-fly if mapping is None

        Parameters
        ----------
        top : :obj:`~mdtraj.Topology`
            The new topology
        mapping : indexable (array, dict, list)
            A mapping of old residue indices
            to new residue indices. Usually,
            comes from aligning the old and the
            new topology using :obj:`mdciao.utils.sequence.maptops`.
            These maps only contain (key,value) pairs
            whenever there's been a "match", s.t
            this method will fail if :obj:`maping`
            doesn't contain all the residues in
            this :obj:`ContactPair`.
        deepcopy : bool, default is False
            Use :obj:`copy.deepcopy` on the attributes
            when creating the new :obj:`ContactPair`.
            If False, the identity holds:

            >>> self.residues.consensus_labels is CP.residues.consensus_labels

            If True, only the equality holds:

            >>> self.residues.consensus_labels == CP.residues.consensus_labels

            Note that :obj:`time_traces` are always created
            new no matter what.
        CP_kwargs : dict
            Optional keyword arguments to instantiate the
            new :obj:`ContactPair`. Any key-value pairs
             inputted here will update the internal
             dictionary being used, which is:


            >>>  {
            "top": top,
            "trajs": self.time_traces.trajs,
            "fragment_idxs": self.fragments.idxs,
            "fragment_names": self.fragments.names,
            "fragment_colors": self.fragments.colors,
            "anchor_residue_idx": anchor_residue_index,
            "consensus_labels": self.residues.consensus_labels
            }

        Returns
        -------
        CP : :obj:`ContactPair`
            A new CP with updated top and indices
        """
        new_pairs = [mapping[ii] for ii in self.residues.idxs_pair]
        atom_pair_trajs = None
        if self.time_traces.atom_pair_trajs is not None:
            oldat2newat = _mapatoms(self.top, top, mapping, {ii: self.top.atom(ii).name for ii in _np.unique(self.time_traces.atom_pair_trajs)})
            atom_pair_trajs = [oldat2newat[itraj] for itraj in self.time_traces.atom_pair_trajs]

        anchor_residue_index = None

        if self.residues.anchor_residue_index is not None:
            anchor_residue_index = mapping[self.residues.anchor_residue_index]
        if deepcopy:
            _copy = lambda x: _deepcopy(x)
        else:
            _copy = lambda x: x

        mapping_kwargs = {
            "top": top,
            "trajs": _copy(self.time_traces.trajs),
            "atom_pair_trajs": atom_pair_trajs,
            "fragment_idxs": _copy(self.fragments.idxs),
            "fragment_names": _copy(self.fragments.names),
            "fragment_colors": _copy(self.fragments.colors),
            "anchor_residue_idx": anchor_residue_index,
            "consensus_labels": _copy(self.residues.consensus_labels)
        }
        for key, val in CP_kwargs.items():
            mapping_kwargs[key]=val

        return ContactPair(
            new_pairs,
            _copy(self.time_traces.ctc_trajs),
            _copy(self.time_traces.time_trajs),
            **mapping_kwargs,
            )

    def __hash__(self):
        tohash = []
        for attr in self._hashable_attrs:
            if "." in attr:
                attr1, attr2 = attr.split(".")
                gattr = getattr(getattr(self, attr1), attr2)
            else:
                gattr = getattr(self, attr)
            tohash.append(gattr)
        return _mdcu.lists.hash_list(tohash)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    @property
    def _hashable_attrs(self):
        return ["residues.idxs_pair",
                "time_traces.ctc_trajs",
                "time_traces.time_trajs",
                "topology",
                "time_traces.trajs",
                "time_traces.atom_pair_trajs",
                "fragments.idxs",
                "fragments.names",
                "fragments.colors",
                "residues.anchor_residue_index",
                "residues.consensus_labels"]

    def save(self,filename):
        r"""Save this :obj:`ContactPair` as a pickle

        Parameters
        ----------
        filename : str
            filename

        Returns
        -------

        """
        _save_as_pickle(self, filename,verbose=False) # Better not be verbose here

    def _serialized_as_dict(self,exclude=None):
        r"""
        Serialize light-weight attributes (everything except mdtraj and mdtops) into
        a dictionary

        Returns
        -------
        tosave : dict

        """
        if exclude is None:
            exclude=[]
        tosave = {}
        for attr in set(self._hashable_attrs).difference(exclude):
            if "." in attr:
                attr1, attr2 = attr.split(".")
                value = getattr(getattr(self, attr1), attr2)
            else:
                value = getattr(self, attr)
            # print(value)
            if attr=="time_traces.trajs":
                if isinstance(value[0], _md.Trajectory):
                    value = ['mdtraj.%02u' % ii for ii, __ in enumerate(value)]
            if not isinstance(value, _md.Topology):
                tosave[attr] = value
        return tosave

    def binarize_trajs(self, ctc_cutoff_Ang,
                       switch_off_Ang=None
                       ):
        """Turn each distance-trajectory into a boolean using a cutoff.
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
        bintrajs : list of boolean arrays with the same shape as the trajectories
        """
        transform = lambda itraj: itraj <= ctc_cutoff_Ang / 10
        _switchoff = 0
        if switch_off_Ang is not None:
            _switchoff = switch_off_Ang
            transform = lambda itraj : _linear_switchoff(itraj,
                                                         ctc_cutoff_Ang/10.,
                                                         switch_off_Ang/10.)


        try:
            result = self._binarized_trajs[ctc_cutoff_Ang][_switchoff]
            #print("Grabbing already binarized %3.2f w switchoff %3.2f"%(ctc_cutoff_Ang,_switchoff))
        except KeyError:
            #print("First time binarizing %3.2f. Storing them"%ctc_cutoff_Ang)
            result = [transform(itraj) for itraj in self.time_traces.ctc_trajs]
            self._binarized_trajs[ctc_cutoff_Ang][_switchoff] = result
        #print([ires.shape for ires in result])
        return result

    def frequency_per_traj(self, ctc_cutoff_Ang,switch_off_Ang=None):
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

        return _np.array([_np.mean(itraj) for itraj in self.binarize_trajs(ctc_cutoff_Ang,
                                                                           switch_off_Ang=switch_off_Ang)])

    def frequency_overall_trajs(self, ctc_cutoff_Ang,switch_off_Ang=None):
        """How many times this contact is formed overall frames.
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
        return _np.mean(_np.hstack(self.binarize_trajs(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang)))


    def frequency_dict(self, ctc_cutoff_Ang,
                       switch_off_Ang=None,
                       AA_format='short',
                       split_label=True,
                       atom_types=False,
                       ):
        """
        Returns the :obj:`frequency_overall_trajs` as a more informative
        dictionary with keys "freq", "residue idxs", "label"

        Parameters
        ----------
        ctc_cutoff_Ang : float
            Cutoff in Angstrom. The comparison operator is "<="
        AA_format : str, default is "short"
            Amino-acid format ("E35" or "GLU25") for the value
            fdict["label"]. Can also be "long" or "just_consensus"
        split_label : bool, default is True
            Split the labels so that stacked contact labels
            become easier-to-read in plain ascii formats
             - "E25@3.50____-    A35@4.50"
             - "A30@longfrag-    A35@4.50
        atom_types : bool, default is false
            Include the relative frequency of atom-type-pairs
            involved in the contact
        Returns
        -------
        fdcit : dictionary

        """

        label = self.label_flex(AA_format, split_label)

        fdict = {"freq":self.frequency_overall_trajs(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang),
                "label":label.rstrip(" "),
                "residue idxs": '%u %u' % tuple(self.residues.idxs_pair)
                }

        if atom_types:
            fdict.update({"by_atomtypes" :
                              self.relative_frequency_of_formed_atom_pairs_overall_trajs(ctc_cutoff_Ang,
                                                                                         switch_off_Ang=switch_off_Ang)})
        return fdict

    def label_flex(self, AA_format="short",split_label=True):
        r"""
        A more flexible method to produce the label of this :obj:`ContactPair`

        Parameters
        ----------
        AA_format : str, default is "short"
            Amino-acid format for the label, can
            be "short" (A35@BW4.50), "long" (ALA35@4.50),
            or "just_consensus" (4.50)
        split_label : bool, default is True
            Split the labels so that stacked contact labels
            become easier-to-read in plain ascii formats
             - "E25@3.50____-    A35@4.50"
             - "A30@longfrag-    A35@4.50

        Returns
        -------
        label : str
        """

        if AA_format== 'short':
            label = self.labels.w_fragments_short_AA
        elif AA_format== 'long':
            label = self.labels.w_fragments
        elif AA_format== 'just_consensus':
            #TODO where do we put this assertion?
            if None in self._attribute_residues.consensus_labels:
                raise ValueError("Residues %s don't have both consensus labels:%s" % (
                    self._attribute_residues.names_short,
                    self._attribute_residues.consensus_labels))
            label = self.labels.just_consensus
        else:
            raise ValueError(AA_format)
        if split_label:
            label= '%-15s - %-15s'%tuple(_mdcu.str_and_dict.splitlabel(label, '-'))

        return label

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
        Count how many times each atom-pair is considered in contact in the trajectories

        Ideally we would return a dictionary but atom pairs is not hashable

        Parameters
        ----------
        ctc_cutoff_Ang : float
            Cutoff in Angstrom. The comparison operator is "<="
        sort: boolean, default is True
            Return the counts by descending order

        Returns
        -------
        atom_pairs: list of atom pairs
        counts : list of ints

        """

        assert self.time_traces.atom_pair_trajs is not None, ValueError("Cannot use this method if no atom_pair_trajs were parsed")
        counts = _col_Counter(["%u-%u"%tuple(fap) for fap in self._overall_stacked_formed_atoms(ctc_cutoff_Ang)])
        keys, counts = list(counts.keys()), list(counts.values())
        keys = [[int(ii) for ii in key.split("-")] for key in keys]
        if sort:
            keys = [keys[ii] for ii in _np.argsort(counts)[::-1]]
            counts=sorted(counts)[::-1]
        return keys, counts

    def partial_counts_formed_atom_pairs(self, ctc_cutoff_Ang,
                                         switch_off_Ang=None,
                                         sort=True):
        r"""
        Count how many times each atom-pair is considered in contact in the trajectories

        Since the :obj:`switch_off_Ang` parameter introduces partial counts, the
        return value need not be integer counts

        Ideally we would return a dictionary but atom pairs is not hashable

        Parameters
        ----------
        ctc_cutoff_Ang : float
            Cutoff in Angstrom. The comparison operator is "<="
        sort: boolean, default is True
            Return the counts by descending order

        Returns
        -------
        atom_pairs: list of atom pairs
        counts : list of ints

        """
        from scipy.sparse import csr_matrix as _csr
        assert self.time_traces.atom_pair_trajs is not None, ValueError("Cannot use this method if no atom_pair_trajs were parsed")

        stacked_at_pair_trajs = _np.vstack(self.time_traces.atom_pair_trajs)
        stacked_counts = _np.hstack(self.binarize_trajs(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang))
        assert len(stacked_counts)==len(stacked_at_pair_trajs)==self._attribute_n.n_frames_total,\
            (len(stacked_counts) , len(stacked_at_pair_trajs) , self._attribute_n.n_frames_total)

        unique_at_pairs = _np.unique(stacked_at_pair_trajs,axis=0).squeeze()
        #Dictionary implementation 2x faster than sparse-matrix implementation!
        mat = {"%u-%u" % tuple(ap):0 for ap in unique_at_pairs}
        for ((ii, jj), cc) in zip(stacked_at_pair_trajs, stacked_counts):
            mat["%u-%u"%(ii,jj)] += cc
        keys, counts = list(mat.keys()),_np.array(list(mat.values()))
        keys = _np.array([[int(ii) for ii in key.split("-")] for key in keys])
        keys = keys[counts!=0]
        counts = counts[counts!=0]
        if sort:
            keys = _np.array([keys[ii] for ii in _np.argsort(counts)[::-1]])
            counts = sorted(counts)[::-1]
        return keys, counts


    def relative_frequency_of_formed_atom_pairs_overall_trajs(self, ctc_cutoff_Ang,
                                                              switch_off_Ang=None,
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
        out_dict : dictionary
            Relative freqs, keyed by atom-type (atoms) involved in the contact
            The order is the same as in :obj:`self.ctc_labels`
        """
        assert self.top is not None, "Missing a topolgy object"
        if switch_off_Ang is None:
            atom_pairs, counts = self.count_formed_atom_pairs(ctc_cutoff_Ang)
        else:
            atom_pairs, counts = self.partial_counts_formed_atom_pairs(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang)
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

    def _plot_timetrace(self,
                        iax,
                        color_scheme=None,
                        ctc_cutoff_Ang=0,
                        switch_off_Ang=None,
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
                ilabel += ' (%u%%)' % (self.frequency_per_traj(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang)[traj_idx] * 100)

            _mdcplots.plot_w_smoothing_auto(iax, itime * dt, ictc_traj * 10,
                                  ilabel,
                                  color_scheme[traj_idx],
                                  gray_background=gray_background,
                                  n_smooth_hw=n_smooth_hw)

        iax.legend(loc=1, fontsize=_rcParams["font.size"] * .75,
                   ncol=_np.ceil(self.n.n_trajs / max_handles_per_row).astype(int)
                   )
        #ctc_label = self.label
        ctc_label = self.labels.w_fragments
        if shorten_AAs:
            ctc_label = self.labels.w_fragments_short_AA
        ctc_label = _mdcu.str_and_dict.latex_superscript_fragments(ctc_label)
        if ctc_cutoff_Ang > 0:
            ctc_label += " (%u%%)" % (self.frequency_overall_trajs(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang) * 100)

        iax.text(_np.mean(iax.get_xlim()), 1 * 10 / _np.max((10, iax.get_ylim()[1])),  # fudge factor for labels
                 ctc_label,
                 ha='center')
        if ctc_cutoff_Ang > 0:
            iax.axhline(ctc_cutoff_Ang, color='k', ls='--', zorder=10)

        if switch_off_Ang is not None:
            iax.axhline(ctc_cutoff_Ang+switch_off_Ang, color='k', ls='--', zorder=10)

        iax.set_xlabel('t / %s' % _mdcu.str_and_dict.replace4latex(t_unit))
        iax.set_xlim([self.time_min*dt, self.time_max * dt])
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
    r"""Container for :obj:`ContactPair`-objects

    This class is the second level of abstraction after :obj:`ContactPair`
    and provides methods to
     * perform operations on all the contact-pairs simultaneously and
     * plot/show/save the result of these operations

    In many cases, the methods of :obj:`ContactGroup` thinly wrap and
    iterate around equally named methods of the :obj:`ContactPair`-objects.

    Note
    ----
    Higher-level methods in the API, like those exposed by :obj:`mdciao.cli`
    will return :obj:`ContactPair` or :obj:`ContactGroup` objects already
    instantiated and ready to use. It is recommened to use those instead
    of individually calling :obj:`ContactPair` or :obj:`ContactGroup`.

    """

    #TODO create an extra interface-class? Unsure
    def __init__(self,
                 list_of_contact_objects,
                 interface_residxs=None,
                 top=None,
                 name=None,
                 neighbors_excluded=None,
                 use_AA_when_conslab_is_missing=True,#TODO this is for the interfaces
                 ):
        r"""

        Parameters
        ----------
        list_of_contact_objects : list
            list of :obj:`ContactPair` objects
        interface_residxs : list of two iterables of indexes, default is None
            An interface is defined by two, non-overlapping
            groups of residue indices.

            That's the only requirement. The input `interface_residxs`
            need not have all or any of the residue indices in
            :obj:`res_idxs_pairs`

            The property :obj:`interface_residxs` groups
            the object's own residue idxs present in
            :obj:`residxs_pairs` into the two groups of the interface.

            #TODO document what happens if there is no overlap

        top : :obj:`~mdtraj.Topology`, default is None

        name : string, default is None
            Optional name you want to give this object,
            ATM it is only used for the title of the
            :obj:`ContactGroup.plot_distance_distributions`
            title when the object is not a neighborhood

        """
        self._contacts = list_of_contact_objects
        self._n_ctcs  = len(list_of_contact_objects)
        self._interface_residxs = interface_residxs
        self._neighbors_excluded = neighbors_excluded
        self._is_interface = False
        self._is_neighborhood = False
        self._name = name
        if top is None:
            self._top = self._unique_topology_from_ctcs()
        else:
            assert top is self._unique_topology_from_ctcs()
            self._top = top

        # Sanity checks about having grouped this contacts together
        if self._n_ctcs==0:
            raise NotImplementedError("A ContactGroup has been initialized with no contacts,\n"
                                      "probably because no residues were found within the cutoff."
                                      )
            # TODO imppelment an empty CG or a propety self.empty?
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
            self._time_min = ref_ctc.time_min
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
            # First update to this TODO, rely less and less on _type_of_attrs
            # and get them new every time from the underlying .residue objects)
            self._trajlabels = ref_ctc.labels.trajstrs
            self._residx2resnameshort = {}
            self._residx2resnamelong = {}
            self._residx2fragnamebest = {}
            for conslab, rnshort, rnlong, ridx, fragname in zip(_np.hstack(self.consensus_labels),
                                                                _np.hstack(self.residue_names_short),
                                                                _np.hstack(self.residue_names_long),
                                                                _np.hstack(self.res_idxs_pairs),
                                                                _np.hstack(self.fragment_names_best)):
                if ridx not in self._residx2resnameshort.keys():
                    self._residx2resnameshort[ridx] = rnshort
                    self._residx2resnamelong[ridx] = rnlong
                else:
                    assert self._residx2resnameshort[ridx] == rnshort, (self._residx2resnameshort[ridx], rnshort)
                    assert self._residx2resnamelong[ridx] == rnlong, (self._residx2resnamelong[ridx], rnlong)

                if ridx not in self._residx2fragnamebest.keys():
                    self._residx2fragnamebest[ridx] = fragname
                else:
                    assert self._residx2fragnamebest[ridx] == fragname, (self._residx2fragnamebest[ridx], fragname)


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
                self._resname2cons[self._residx2resnameshort[ii]]=None
            """

            if self._interface_residxs is not None:
                # TODO prolly this is anti-pattern but I prefer these many sanity checks
                assert len(self._interface_residxs)==2
                intersect = list(set(self._interface_residxs[0]).intersection(self._interface_residxs[1]))
                assert len(intersect)==0, ("Some_residxs appear in both members of the interface %s, "
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

                # TODO Is the comparison throuh residxs robust enough, would it be
                # better to compare consensus labels directly?

                self._interface_residxs = res
                if len(res[0])>0 and len(res[1])>0:
                    self._is_interface = True
            else:
                self._interface_residxs = [[],[]]

            if self.shared_anchor_residue_index is not None:
                self._is_neighborhood=True
                if self.neighbors_excluded is None:
                    raise ValueError("This ContactGroup looks like a neighborhood,\n"
                                     "(all contacts share the residue %s), "
                                     "but no 'neighbors_excluded' have been parsed!\n"
                                     "If you're trying to build a site object,\n"
                                     "use 'neighbors_excluded'=0', else input the right number of"
                                     "'neighbors_excluded'"%self.shared_anchor_residue_index)

    #todo again the dicussion about named tuples vs a miriad of properties
    # I am opting for properties because of easiness of documenting i

    @property
    def neighbors_excluded(self):
        return self._neighbors_excluded

    @property
    def name(self):
        return self._name

    #TODO access to conctat labels with fragnames and/or consensus?
    @property
    def n_trajs(self):
        return self._n_trajs

    @property
    def n_ctcs(self):
        r"""
        The number of contact pairs (:obj:`mdciao.contacts.ContactPair` -objects) stored in this object
        Returns
        -------

        """
        return self._n_ctcs

    @property
    def n_frames(self):
        r"""
        List of per-trajectory n_frames
        Returns
        -------
        n_frames : list
        """
        return self._n_frames

    @property
    def n_frames_total(self):
        r"""
        Total number of frames
        Returns
        -------
        n_frames : int
        """
        return _np.sum(self._n_frames)

    @property
    def time_max(self):
        return self._time_max

    @property
    def time_min(self):
        return self._time_min

    @property
    def time_arrays(self):
        return self._time_arrays

    @property
    def res_idxs_pairs(self):
        r"""
        List of pairs of residue indices of the contacts in this object

        Returns
        -------

        """
        return _np.vstack([ictc.residues.idxs_pair for ictc in self._contacts])

    @property
    def residue_names_short(self):
        return [ictc.residues.names_short for ictc in self._contacts]

    @property
    def residue_names_long(self):
        return [ictc.residues.names for ictc in self._contacts]

    @property
    def fragment_names_best(self):
        return [ictc.labels.fragment_labels_best(fmt="%s") for ictc in self._contacts]

    @property
    def ctc_labels(self):
        return [ictc.labels.no_fragments for ictc in self._contacts]

    @property
    def ctc_labels_short(self):
        r"""
        Short contact labels without fragment info, e.g. E30-R40

        Returns
        -------
        labels : list
        """
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
        _cons2resname = {}
        for conslab, resname, ridx, in zip(_np.hstack(self.consensus_labels),
                                           _np.hstack(self.residue_names_short),
                                           _np.hstack(self.res_idxs_pairs),
                                  ):
            if conslab not in _cons2resname.keys():
                _cons2resname[conslab] = resname
            else:
                assert _cons2resname[conslab] == resname, (_cons2resname[conslab], conslab, resname)

        return _cons2resname

    @property
    def residx2consensuslabel(self):
        _residx2conslabels = {}
        for conslab, ridx in zip(_np.hstack(self.consensus_labels),
                                 _np.hstack(self.res_idxs_pairs),
                                 ):
            if ridx not in _residx2conslabels.keys():
                _residx2conslabels[ridx] = conslab
            else:
                assert _residx2conslabels[ridx] == conslab, (_residx2conslabels[ridx], conslab)

        return _residx2conslabels

    @property
    def residx2resnameshort(self):
        return self._residx2resnameshort

    @property
    def residx2resnamelong(self):
        return self._residx2resnamelong

    @property
    def residx2fragnamebest(self):
        return self._residx2fragnamebest

    def residx2resnamefragnamebest(self,fragsep="@",shorten_AAs=True):
        idict   = {}
        for key in _np.unique(self.res_idxs_pairs):
            if shorten_AAs:
                val = self.residx2resnameshort[key]
            else:
                val = self.residx2resnamelong[key]
            ifrag = self.residx2fragnamebest[key]
            if len(ifrag) > 0:
                val += "%s%s" % (fragsep, ifrag)
            idict[key] = val
        return idict

    @property
    def is_neighborhood(self):
        return self._is_neighborhood

    #TODO make this a property at instantiation and build neighborhoods a posteriori?
    @property
    def shared_anchor_residue_index(self):
        r"""
        Returns none if no anchor residue is found or if the ContactGroup is empty
        """
        if any([ictc.residues.anchor_residue_index is None for ictc in self._contacts]):
            #todo dont print so much
            #todo let it fail?
            #print("Not all contact objects have an anchor_residue_index. Returning None")
            return None
        else:
            shared = _np.unique([ictc.residues.anchor_residue_index for ictc in self._contacts])
            if len(shared) == 1:
                return shared[0]

    @property
    def anchor_res_and_fragment_str(self):
        assert self.is_neighborhood,"There is no anchor residue, This is not a neighborhood."
        return self._contacts[0].neighborhood.anchor_res_and_fragment_str.rstrip("@")

    @property
    def anchor_res_and_fragment_str_short(self):
        assert self.is_neighborhood
        return self._contacts[0].neighborhood.anchor_res_and_fragment_str_short.rstrip("@")

    @property
    def partner_res_and_fragment_labels(self):
        assert self.is_neighborhood
        return [ictc.neighborhood.partner_res_and_fragment_str.rstrip("@") for ictc in self._contacts]

    @property
    def partner_res_and_fragment_labels_short(self):
        assert self.is_neighborhood
        return [ictc.neighborhood.partner_res_and_fragment_str_short.rstrip("@") for ictc in self._contacts]

    @property
    def anchor_fragment_color(self):
        assert self.is_neighborhood
        _col = self._contacts[0].fragments.colors[self._contacts[0].residues.anchor_index]
        cond1 = not any([ictc.fragments.colors[ictc.residues.anchor_index] is None for ictc in self._contacts])
        cond2 = all([ictc.fragments.colors[ictc.residues.anchor_index] == _col for ictc in self._contacts[1:]])
        if cond1 and cond2:
            return _col
        else:
            print("Not all anchors have or share the same color, returning None")
            return None

    @property
    def partner_fragment_colors(self):
        assert self.is_neighborhood
        _col = self._contacts[0].fragments.colors[self._contacts[0].residues.anchor_index]
        partner_fragment_colors = [ictc.fragments.colors[ictc.residues.partner_index] for ictc in self._contacts]
        not any([ictc.fragments.colors[ictc.residues.partner_index] is None for ictc in self._contacts])
        if not any([icol is None for icol in partner_fragment_colors]):
            return partner_fragment_colors
        else:
            print("Not all partners have a defined color, returning None")
            return None

    def relabel_consensus(self,
                          new_labels=None,
                          ):
        """Relabel any residue missing its consensus label to shortAA

        Alternative (or additional) labels can be given as a
        dictionary.

        Parameters
        ----------
        new_labels : dict
            keyed with shortAA-codes and valued
            with the new desired labels

        TODO
        ----
        Perhaps its better to key both with shortAAs and/or
        consensus labels also?

        Warning
        -------
        For expert use only. The changes
        in consensus labels propagates down to
        the attribute consensus labels of the
        the low-level attribute :obj:`Residues.consensus_labels`
        of the :obj:`Residues` objects
        underlying each of the :obj:`ContactPair`s
        in this :obj:`ContactGroup`

        """
        if new_labels is None:
           new_labels = {}

        for cp in self._contacts:
            consensus_labels = cp.residues.consensus_labels  # We need the attribute outside
            for ii in [0, 1]:
                if cp.residues.names_short[ii] in new_labels.keys():
                    consensus_labels[ii] = new_labels[cp.residues.names_short[ii]]
                elif str(consensus_labels[ii]).lower() == "none":
                    consensus_labels[ii] = cp.residues.names_short[ii]

    #todo there is redundant code for generatinginterface labels!
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

    def binarize_trajs(self, ctc_cutoff_Ang, switch_off_Ang=None, order='contact'):
        r""" Binarize trajs

        Parameters
        ----------
        ctc_cutoff_Ang
        order : str, default is "contact"
            Sort first by contact, then by traj index. Alternative is
            "traj", i.e. sort first by traj index, then by contact
        switch_off_Ang : float, default is None
            Implements a linear switchoff
            from :obj:`ctc_cutoff_Ang` to :obj:`ctc_cutoff_Ang`+`switch_off_Ang`.
            E.g. if the cutoff is 3 Ang and the switch is 1 Ang, then
             * 3.0 -> 1.0
             * 3.5 -> .5
             * 4.0 -> 0.0
            TODO: change the name "binarize"

        Returns
        -------
        bintrajs : list of boolean arrays
            if order==traj, each item of the list is a 2D np.ndarray
            with of shape(Nt,n_ctcs), where Nt is the number of frames
            of that trajectory

        """
        bintrajs = [ictc.binarize_trajs(ctc_cutoff_Ang,
                                        switch_off_Ang=switch_off_Ang
                                        ) for ictc in self._contacts]
        if order=='contact':
            return bintrajs
        elif order=='traj':
            _bintrajs = [_np.zeros((nf,self.n_ctcs), dtype=bool) for nf in self.n_frames]
            for ii in range(self.n_trajs):
                for jj in range(self.n_ctcs):
                    _bintrajs[ii][:,jj] = bintrajs[jj][ii]

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
            The first index is the contact index, the second the pair index (0 or 1)
        """
        ctc_idxs = []
        for ii, pair in enumerate(self.res_idxs_pairs):
            if idx in pair:
                ctc_idxs.append([ii,_np.argwhere(pair==idx).squeeze()])
        return _np.vstack(ctc_idxs)


    def frequency_dicts(self, ctc_cutoff_Ang,
                       sort=False,
                       **kwargs):
        """
        Wraps around the method :obj:`ContactPair.frequency_dict`
        of each of the underlying :obj:`ContactPair` s and
        returns one frequency dict keyed by contact label

        Parameters
        ----------
        ctc_cutoff_Ang : float
            Cutoff in Angstrom. The comparison operator is "<="
        sort : bool, default is False
            Sort by descending frequency. Default
            is to return in the same order
            as :obj:`ContactGroup._contacts`
        kwargs : optional keyword arguments
            Check :obj:`ContactPair.frequency_dict`

        Returns
        -------
        fdict : dictionary

        """
        frequency_dicts = [cp.frequency_dict(ctc_cutoff_Ang=ctc_cutoff_Ang, **kwargs) for cp in self._contacts]
        if sort:
            frequency_dicts = sorted(frequency_dicts,
                                     key=lambda value: value["freq"],
                                     reverse=True)
        return {idict["label"] : idict["freq"] for idict in frequency_dicts}

    # TODO think about implementing a frequency class, but how
    # to do so without circular dependency to the ContactGroup object itself?
    def frequency_per_contact(self, ctc_cutoff_Ang,
                              switch_off_Ang=None):
        r"""
        Frequency per contact over all trajs
        Parameters
        ----------
        ctc_cutoff_Ang

        Returns
        -------
        freqs : 1D np.ndarray of len(n_ctcs)
        """
        return _np.array([ictc.frequency_overall_trajs(ctc_cutoff_Ang,switch_off_Ang=switch_off_Ang) for ictc in self._contacts])

    def frequency_sum_per_residue_idx_dict(self, ctc_cutoff_Ang,
                                           switch_off_Ang=None,
                                           return_array=False):
        r"""
        Dictionary of aggregated :obj:`frequency_per_contact` per residue indices
        Values over 1 are possible, example if [0,1], [0,2]
        are always formed (=1) freqs_dict[0]=2

        Parameters
        ----------
        ctc_cutoff_Ang
        return_array : bool, default is False
            If True, the return value is not a dict
            but an array of len(self.top.n_residues)

        Returns
        -------
        freqs_dict : dictionary or array
            If dict, keys are the residue indices present in :obj:`res_idxs_pairs`
            If array, idxs are the residue indices of self.top


        """
        dict_sum = _defdict(list)
        for (idx1, idx2), ifreq in zip(self.res_idxs_pairs,
                                       self.frequency_per_contact(ctc_cutoff_Ang,
                                                                  switch_off_Ang=switch_off_Ang)):
            dict_sum[idx1].append(ifreq)
            dict_sum[idx2].append(ifreq)
        dict_sum = {key: _np.sum(val) for key, val in dict_sum.items()}
        if return_array:
            array_sum = _np.zeros(self.top.n_residues)
            array_sum[list(dict_sum.keys())] = list(dict_sum.values())
            return array_sum
        else:
            return dict_sum

    def frequency_sum_per_residue_names(self, ctc_cutoff_Ang,
                                        switch_off_Ang=None,
                                        sort=True,
                                        shorten_AAs=True,
                                        list_by_interface=False,
                                        return_as_dataframe=False,
                                        fragsep="@"):
        r"""
        Aggregate the frequencies of :obj:`frequency_per_contact` keyed
        by residue name, using the most informative names possible,
        see :obj:`self.residx2resnamefragnamebest` for more info on this

        Parameters
        ----------
        ctc_cutoff_Ang
        sort : bool, default is True
            Sort by dictionary by descending order of frequencies
            TODO dicts have order since py 3.6 and it is useful for creating
            TODO a dataframe, then excel_table that's already sorted by descending frequencies
        shorten_AAs : bool, default is True
            Use E30 instead of GLU30
        list_by_interface : bool, default is False
            group the freq_dict by interface residues.
            Only has an effect if self.is_interface
        return_as_dataframe : bool, default is False
            Return an :obj:`~pandas.DataFrame` with the column names labels and freqs
        fragsep : str, default is @
            String to separate residue@fragname
        Returns
        -------
        res : list
            list of dictionaries (or dataframes).
            If :obj:`list_by_interface` is True,
            then the list has two items, default
            (False) is to be of len=1

        """
        freqs = self.frequency_sum_per_residue_idx_dict(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang)

        if list_by_interface and self.is_interface:
                freqs = [{idx:freqs[idx] for idx in iint} for iint in self.interface_residxs]
        else:
            freqs = [freqs] #this way it is a list either way

        if sort:
            freqs = [{key:val for key, val in sorted(idict.items(),
                                                     key=lambda item: item[1],
                                                     reverse=True)}
                     for idict in freqs]

        # Use the residue@frag representation but avoid empty fragments
        list_out = []
        for ifreq in freqs:
            idict = {}
            for idx, val in ifreq.items():
                key = self.residx2resnamefragnamebest(shorten_AAs=shorten_AAs)[idx]
                idict[key] = val
            list_out.append(idict)

        if return_as_dataframe:
            list_out = [_DF({"label": list(idict.keys()),
                             "freq": list(idict.values())}) for idict in list_out]

        return list_out

    """"
    # TODO this seems to be unused
    def frequency_table_by_residue(self, ctc_cutoff_Ang,
                                   list_by_interface=False):
        dict_list = self.frequency_sum_per_residue_names(ctc_cutoff_Ang,
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
                                           switch_off_Ang=None,
                                           return_as_triplets=False,
                                           sort_by_interface=False,
                                           include_trilower=False):
        r"""
        Return frequencies as a dictionary of dictionaries keyed by consensus labels

        Note
        ----
        Will fail if not all residues have consensus labels
        TODO this is very similar to :obj:`frequency_sum_per_residue_names`,
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
        dict_out = _defdict(dict)
        for (key1, key2), ifreq in zip(self.consensus_labels,
                                       self.frequency_per_contact(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang)):
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
                            switch_off_Ang=None,
                            atom_types=False,
                            sort=False,
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
        atom_types : bool, default is false
            Include the relative frequency of atom-type-pairs
            involved in the contact
        sort : bool, default is False
            Sort by descending frequency value,
            default is to keep the order of
            :obj:`self._contacts`
        ctc_fd_kwargs: named optional arguments
            Check :obj:`ContactPair.frequency_dict` for more info on e.g
            AA_format='short' and or split_label


        Returns
        -------
        df : :obj:`pandas.DataFrame`
        """
        idicts = [ictc.frequency_dict(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang, atom_types=atom_types, **ctc_fd_kwargs) for ictc in self._contacts]
        if atom_types is True:
            for jdict in idicts:
                istr =  '%s' % (', '.join(['%3u%% %s' % (val * 100, key)
                                           for key, val in sorted(jdict["by_atomtypes"].items(),key=lambda item: item[1],reverse=True)]))
                jdict.pop("by_atomtypes")
                jdict["by_atomtypes"]=istr

        idf = _DF(idicts)
        if sort:
            idf.sort_values("freq",
                            ignore_index=True,
                            inplace=True,
                            ascending=False
                            )
        df2return = idf.join(_DF(idf["freq"].values.cumsum(), columns=["sum"]))
        return df2return

    def frequency_table(self, ctc_cutoff_Ang,
                        fname,
                        switch_off_Ang=None,
                        sort=False,
                        write_interface=True,
                        **freq_dataframe_kwargs):
        r"""
        Print and/or save frequencies as a formatted table

        Internally, it calls :obj:`frequency_spreadsheet` and/or
        :obj:`frequency_str_ASCII_file` depending on the
        extension of :obj:`fname`

        If you want a :obj:`~pandas.DataFrame` use
        :obj:`frequency_dataframe`

        Parameters
        ----------
        ctc_cutoff_Ang : float
        fname : str or None
            Full path to the desired filename
            Spreadsheet extensions are currently
            only '.xlsx', all other extensions
            save to formatted ascii. `None`
            returns the formatted ascii string.
        switch_off_Ang : float, default is None,
        sort : check frequency_sum_per_residue_names
        write_interface : check frequency_sum_per_residue_names
        freq_dataframe_kwargs

        Returns
        -------
        table : None or str
            If :obj:`fname` is none, then return
            the table as formatted string, using
        """

        if _path.splitext(str(fname))[1] in [".xlsx"]:
            freq_dataframe_kwargs["split_label"] = False
            main_DF = self.frequency_dataframe(ctc_cutoff_Ang,
                                               switch_off_Ang=switch_off_Ang,
                                               **freq_dataframe_kwargs)
            idfs = self.frequency_sum_per_residue_names(ctc_cutoff_Ang,
                                                        switch_off_Ang=switch_off_Ang,
                                                        sort=sort,
                                                        list_by_interface=write_interface,
                                                        return_as_dataframe=True)
            self.frequency_spreadsheet(main_DF,idfs,ctc_cutoff_Ang,fname)
        else:
            freq_dataframe_kwargs["split_label"] = True
            main_DF = self.frequency_dataframe(ctc_cutoff_Ang,
                                               switch_off_Ang=switch_off_Ang,
                                               **freq_dataframe_kwargs)
            return self.frequency_str_ASCII_file(main_DF,ascii_file=fname)

    def frequency_spreadsheet(self, sheet1_dataframe,
                              sheet2_dataframes,
                              ctc_cutoff_Ang,
                              fname_excel,
                              sheet1_name="pairs by frequency",
                              sheet2_name='residues by frequency',
                              ):
        r"""
        Write an Excel file with the :obj:`~pandas.Dataframe` that is
        returned by :obj:`self.frequency_dataframe`. You can
        control that call with obj:`freq_dataframe_kwargs`

        Parameters
        ----------
        ctc_cutoff_Ang
        fname_excel
        sort : bool, default is True
            Sort by descing order of frequency
        write_interface: bool, default is True
            Treat contact group as interface
        freq_dataframe_kwargs: dict, default is {}
            Optional arguments to :obj:`self.frequency_dataframe`, like by_atomtypes (bool)

        Returns
        -------

        """
        offset = 0
        columns = ["label",
                   "freq",
                   "sum",
                   ]
        if "by_atomtypes" in sheet1_dataframe.keys():
            columns += ["by_atomtypes"]

        writer = _ExcelWriter(fname_excel, engine='xlsxwriter')
        workbook = writer.book
        writer.sheets[sheet1_name] = workbook.add_worksheet(sheet1_name)
        writer.sheets[sheet1_name].write_string(0, offset,
                                      'pairs by contact frequency at %2.1f Angstrom' % ctc_cutoff_Ang)
        offset+=1
        sheet1_dataframe.round({"freq": 2, "sum": 2}).to_excel(writer,
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

        sheet2_dataframes[0].round({"freq": 2}).to_excel(writer,
                                                         sheet_name=sheet2_name,
                                                         startrow=offset,
                                                         startcol=0,
                                                         columns=[
                                                "label",
                                                "freq"],
                                                         index=False
                                                         )
        if len(sheet2_dataframes)>1:
            #Undecided about best placement for these
            sheet2_dataframes[1].round({"freq": 2}).to_excel(writer,
                                                             sheet_name=sheet2_name,
                                                             startrow=offset,
                                                             startcol=2+1,
                                                             columns=[
                                                         "label",
                                                         "freq"],
                                                             index=False
                                                             )

        writer.save()

    def frequency_str_ASCII_file(self, idf,
                                 ascii_file=None):
        r"""
        Return a string with the frequencies

        Parameters
        ----------
        ctc_cutoff_Ang : float
            The cutoff
        switch_off_Ang : bool, default is False
            Whether to use a switch or not
        by_atomtypes : bool, default is True
            Include the type of atoms involved in the contact
        ascii_file : str, default is None
            If provided a filename, write the frequencies directly to it

        Returns
        -------

        """

        idf = idf.round({"freq": 2, "sum": 2})
        istr = idf.to_string(index=False,
                             header=True,
                             #How to justify the column labels (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_string.html)
                             justify="center",
                             formatters=_mdcu.str_and_dict.df_str_formatters(idf[[key for key in ["by_atomtypes","label"] if key in idf.keys()]])
                             )
        istr = '#%s\n'%istr[1:]
        if ascii_file is None:
            return istr
        else:
            with open(ascii_file, "w") as f:
                f.write(istr)

    def frequency_as_contact_matrix(self,
                                    ctc_cutoff_Ang,
                                    switch_off_Ang=None):
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
        for (ii, jj), freq in zip(self.res_idxs_pairs, self.frequency_per_contact(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang)):
            mat[ii, jj] = freq
            mat[jj, ii] = freq

        return mat

    def relative_frequency_formed_atom_pairs_overall_trajs(self, ctc_cutoff_Ang, switch_off_Ang=None, **kwargs):
        r"""

        Parameters
        ----------
        ctc_cutoff_Ang

        Returns
        -------
        refreq_dicts : list of dicts
        """
        return [ictc.relative_frequency_of_formed_atom_pairs_overall_trajs(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang,**kwargs) for ictc in self._contacts]

    def distributions_of_distances(self, bins=10):
        r"""
        Histograms the distance values of each contact,
        returning a list with as many distributions as there
        are contacts.

        Parameters
        ----------
        bins : int, default is 10

        Returns
        -------
        list_of_distros : list
            List of len self.n_ctcs, each entry contains
            the counts and edges of the bins
        """
        return [ictc.distro_overall_trajs(bins=bins) for ictc in self._contacts]

    def distribution_dicts(self,
                           bins=10,
                           **kwargs):
        """
        Wraps around the method :obj:`ContactGroup.distributions_of_distances`
        and returns one distribution dict keyed by contact label (see kwargs and CP.label_flex

        Parameters
        ----------
        kwargs : optional keyword arguments
            Check :obj:`ContactPair.frequency_dict`

        Returns
        -------
        fdict : dictionary

        """
        distro_dicts = {ictc.label_flex(**kwargs) : data for ictc, data in zip(self._contacts, self.distributions_of_distances(
            bins=bins))}


        return distro_dicts

    def n_ctcs_timetraces(self, ctc_cutoff_Ang, switch_off_Ang=None):
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
        bintrajs = self.binarize_trajs(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang,order='traj')
        _n_ctcs_t = []
        for itraj in bintrajs:
            _n_ctcs_t.append(itraj  .sum(1))
        return _n_ctcs_t

    def plot_freqs_as_bars(self,
                           ctc_cutoff_Ang,
                           title_label=None,
                           switch_off_Ang=None,
                           xlim=None,
                           ax=None,
                           color=["tab:blue"],
                           shorten_AAs=False,
                           label_fontsize_factor=1,
                           truncate_at=None,
                           total_freq=None,
                           atom_types=False,
                           display_sort=False,
                           sum_freqs=True,
                           defrag=None,
                           ):
        r"""
        Plot a contact frequencies as a bar plot

        Parameters
        ----------
        ctc_cutoff_Ang : float
        title_label : str, default is None
            If None, the method will default to self.name
            If self.name is also None, the method will fail
        xlim : float, default is None
        ax : :obj:`~matplotlib.axes.Axes`
        shorten_AAs : bool, default is None
        label_fontsize_factor : float
        truncate_at : float, default is None
        display_sort : boolean, default is False
            The frequencies are by default plotted in the order
            in which the :obj:`ContactPair`-objects are stored
            in the :obj:`ContactGroup`-object's _contact_pairs
            This order depends on the ctc_cutoff_Ang originally
            used to instantiate this :obj:`ContactPair`
            If True, you can re-sort them with this cutoff for
            display purposes only (the original order is untouched)
        Returns
        -------
        ax : :obj:`~matplotlib.axes.Axes`

        """

        # Base plot
        if title_label is None and not self.is_neighborhood:
            assert self.name is not None, ("Cannot use a 'nameless' ContactGroup and 'title_label'=None.\n"
                                           "Either instantiate self.name or pass a 'title_label' ")
            title_label = self.name

        freqs = self.frequency_per_contact(ctc_cutoff_Ang,
                                           switch_off_Ang=switch_off_Ang,
                                           )
        if display_sort:
            order = _np.argsort(freqs)[::-1]
        else:
            order = _np.arange(len(freqs))

        ax = _mdcplots.plots._plot_freqbars_baseplot(freqs[order],
                                                     jax=ax,
                                                     color=color,
                                                     truncate_at=truncate_at)

        label_bars = [ictc.labels.w_fragments for ictc in self._contacts]
        if shorten_AAs:
            label_bars = [ictc.labels.w_fragments_short_AA for ictc in self._contacts]

        # Cosmetics
        sigma = _np.sum([ipatch.get_height() for ipatch in ax.patches])
        title = "Contact frequency @%2.1f AA"%ctc_cutoff_Ang
        if self.is_neighborhood:
            title+="\n%s nearest bonded neighbors excluded\n" % (str(self.neighbors_excluded).replace("None","no"))
            label_dotref = self.anchor_res_and_fragment_str
            label_bars = self.partner_res_and_fragment_labels
            if shorten_AAs:
                label_dotref = self.anchor_res_and_fragment_str_short
                label_bars = self.partner_res_and_fragment_labels_short
            if sum_freqs:
                label_dotref = "\n".join([_mdcu.str_and_dict.latex_superscript_fragments(label_dotref),
                                          _mdcu.str_and_dict.replace4latex('Sigma = %2.1f' % sigma)])  # sum over all bc we did not truncate
                ax.plot(_np.nan, _np.nan, 'o',
                        color=self.anchor_fragment_color,
                        label=_mdcu.str_and_dict.latex_superscript_fragments(label_dotref))
        else:
            if sum_freqs:
                title+= " of '%s' (Sigma = %2.1f)\n" % (title_label,sigma)
                if total_freq is not None:
                    title+="these %u most frequent contacts capture %4.2f %% of all contacts\n" % (self.n_ctcs,
                                                                                           sigma / total_freq * 100,
                                                                                                   )
        if defrag is not None:
            label_bars = [_mdcu.str_and_dict.defrag_key(ilab, defrag=defrag) for ilab in label_bars]
        _mdcplots.add_tilted_labels_to_patches(ax,
                                               [label_bars[ii] for ii in order][:(ax.get_xlim()[1]).astype(int) + 1],  #can't remember this
                                               label_fontsize_factor=label_fontsize_factor
                                               )

        ax.set_title(_mdcu.str_and_dict.replace4latex(title),
                     y = _np.max([1, _mdcplots.highest_y_textobjects_in_Axes_units(ax)])
                     )

        #ax.legend(fontsize=_rcParams["font.size"] * label_fontsize_factor)
        if xlim is not None:
            ax.set_xlim([-.5, xlim + 1 - .5])

        if self.is_neighborhood:
            ax.legend(fontsize=_rcParams["font.size"] * label_fontsize_factor)
        if atom_types:
            self._add_hatching_by_atomtypes(ax, ctc_cutoff_Ang, display_order=order, switch_off_Ang=switch_off_Ang)

        return ax

    def plot_neighborhood_freqs(self, ctc_cutoff_Ang,
                                switch_off_Ang=None,
                                color=["tab:blue"],
                                xmax=None,
                                ax=None,
                                shorten_AAs=False,
                                label_fontsize_factor=1,
                                sum_freqs=True,
                                plot_atomtypes=False,
                                display_sort=False):
        r"""
        Wrapper around :obj:`ContactGroup.plot_freqs_as_bars`
        for plotting neighborhoods

        #TODO perhaps get rid of the wrapper altogether. ATM it would break the API

        Parameters
        ----------
        ctc_cutoff_Ang : float
        xmax : int, default is None
            Default behaviour is to go to n_ctcs, use this
            parameter to homogenize different calls to this
            function over different contact groups, s.t.
            each subplot has equal xlimits
        ax : :obj:`~matplotlib.axes.Axes`
        shorten_AAs
        label_fontsize_factor
        sum_freqs: bool, default is True
            Add the sum of frequencies of the represented (and only those)
            frequencies
        display_sort : boolean, default is False
            The frequencies are by default plotted in the order
            in which the :obj:`ContactPair`-objects are stored
            in the :obj:`ContactGroup`-object's _contact_pairs
            This order depends on the ctc_cutoff_Ang originally
            used to instantiate this :obj:`ContactPair`
            If True, you can re-sort them with this cutoff for
            display purposes only (the original order is untouched)

        Returns
        -------
        ax : :obj:`~matplotlib.axes.Axes`
        """

        assert self.is_neighborhood, "This ContactGroup is not a neighborhood, use ContactGroup.plot_freqs_as_bars() instead"

        ax = self.plot_freqs_as_bars(ctc_cutoff_Ang,
                                     ax=ax,
                                     xlim=xmax,
                                     shorten_AAs=shorten_AAs,
                                     truncate_at=None,
                                     atom_types=plot_atomtypes,
                                     display_sort=display_sort,
                                     switch_off_Ang=switch_off_Ang,
                                     label_fontsize_factor=label_fontsize_factor,
                                     color=color,
                                     sum_freqs=sum_freqs
                                     )
        return ax

    def _get_hatches_for_plotting(self, ctc_cutoff_Ang, switch_off_Ang=None):
        r"""
        Wrapper around :obj:`self.relative_frequency_formed_atom_pairs_overall_trajs`
        to fill zeroes and invert labels ["SC-BB"] labels so that the anchor
        residue always comes first in case of this :obj:`ContactGroup` being a
        neighborhood

        Parameters
        ----------
        ctc_cutoff_Ang
        switch_off_Ang

        Returns
        -------
        df : :obj:`pandas.DataFrame`
            filled zeroes and swapped ["BB-SC"]["SC-BB"] columns when necessary

        """
        list_of_dicts = self.relative_frequency_formed_atom_pairs_overall_trajs(ctc_cutoff_Ang,
                                                                                switch_off_Ang=switch_off_Ang)

        df = _DF(list_of_dicts, columns=_hatchets).fillna(0) # Letting pandas work for us filling zeroes
        swap_order = [ii for ii, ictc in enumerate(self._contacts) if ictc.residues.anchor_index==1]
        if len(swap_order)>0:
            df.loc[swap_order,["SC-BB", "BB-SC"]] = df.loc[swap_order,["BB-SC", "SC-BB"]].values
        return df

    def _add_hatching_by_atomtypes(self, jax, ctc_cutoff_Ang, display_order=False, switch_off_Ang=None,
                                   ):
        r"""
        Add hatches representing contact-type to the frequency bars in :obj:`jax`

        A small legend will appear at the bottom of the plot

        Parameters
        ----------
        jax : :obj:`~matplotlib.axes.Axes`
            The axis where the frequency bars where plotted
        ctc_cutoff_Ang : float
            The cutoff that was used (otherwise we cannot compute atomtype freqs)
        order : iterable of ints, default is None
            If None, the hatches will be assigned to the bars in
            the same order they appear in
        display_order : iterable of ints, default is False
            The hatches are by default plotted in the order
            in which the :obj:`ContactPair`-objects are stored
            in the :obj:`ContactGroup`-object's _contact_pairs
            This order depends on the ctc_cutoff_Ang originally
            used to instantiate this :obj:`ContactPair`
            Pass an order here to re-sort them for a different cutoff for
            display purposes only (the original order is untouched)
        switch_off_Ang

        Returns
        -------
        Nothing

        """

        hatched_lists = self._get_hatches_for_plotting(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang).values
        if display_order is not None:
            hatched_lists = hatched_lists[display_order]
        heights = _np.array([ipatch.get_height() for ipatch in jax.patches])
        width = jax.patches[0].get_width()
        color = jax.patches[0].get_facecolor()

        w_hatched_lists = hatched_lists*heights[:,_np.newaxis]
        for ii, key in enumerate(_hatchets.keys()):
            jax.bar(_np.arange(len(w_hatched_lists)),
                    w_hatched_lists[:,ii],
                    color="r",
                    fill=False,
                    ec="w",
                    #ec="lightgray",
                    alpha=.5,
                    width=width*1.,
                    fc=None,
                    bottom = w_hatched_lists[:,:ii].sum(1),
                    hatch=_hatchets[key],
                    lw=0)

        # Add the hatchet_legend
        leg1 = jax.get_legend()
        # Empty plots
        _hatchets_to_plot = [key for ii, key in enumerate(_hatchets.keys()) if w_hatched_lists[:,ii].sum()>0]
        ebars = [
            jax.bar(_np.nan, _np.nan,
                    color="r",
                    #fill=True,
                    ec="w",
                    fc=color,
                    hatch=_hatchets[key],
                    #width=.01,
                    lw=0)[0]
            for key in _hatchets_to_plot]

        pd = _mdcplots.plots._points2dataunits(jax)[1]
        try:
            lowbar_fspts = jax.texts[0].get_fontsize() * .75
        except IndexError:
            lowbar_fspts = _rcParams["font.size"] * .75
        lowbar_fsaus = lowbar_fspts / pd
        y_leg = -2 * lowbar_fsaus  # fudged to "close enough"
        place_legend = lambda y: getattr(jax, "legend")(ebars, _hatchets_to_plot,
                                                        loc=[0, y],
                                                        ncol=4,
                                                        framealpha=0,
                                                        frameon=False,
                                                        fontsize=lowbar_fspts,
                                                        handletextpad=.1,
                                                        columnspacing=1,
                                                        handlelength=1.)
        leg2 = place_legend(y_leg)

        cc, rend = 0, jax.figure.canvas.get_renderer()
        while jax.bbox.overlaps(leg2.get_window_extent(renderer=rend)):
            leg2.remove()
            y_leg += y_leg*.05
            leg2 = place_legend(y_leg)
            #print(cc,y_leg)
            cc+=1
            if cc>5:
                break
        if leg1 is not None:
            jax.add_artist(leg1)

    def plot_distance_distributions(self,
                                    bins=10,
                                    xlim=None,
                                    jax=None,
                                    shorten_AAs=False,
                                    ctc_cutoff_Ang=None,
                                    label_fontsize_factor=1,
                                    max_handles_per_row=4,
                                    defrag=None):

        r"""
        Plot distance distributions for the distance trajectories
        of the contacts

        The title will get try to get the name from :obj:`self.name`

        Parameters
        ----------
        bins : int, default is 10
            How many bins to use for the distribution
        xlim : iterable of two floats, default is None
            Limits of the x-axis.
            Outlier can stretch the scale, this forces it
            to a given range
        jax : :obj:`~matplotlib.axes.Axes`, default is None
            One will be created if None is passed
        shorten_AAs: bool, default is False
            Use amino-acid one-letter codes
        ctc_cutoff_Ang: float, default is None
            Include in the legend of the plot how much of the
            distribution is below this cutoff. A vertical line
            will be draw at this x-value
            nearest bonded neighbors were excluded
        label_fontsize_factor
        max_handles_per_row: int, default is 4
            legend control

        Returns
        -------
        jax : :obj:`~matplotlib.axes.Axes`

        """
        if jax is None:
            _plt.figure(figsize=(7, 5))
            jax = _plt.gca()

        if self.is_neighborhood:
            title = self.anchor_res_and_fragment_str
            label_bars = self.partner_res_and_fragment_labels
            if shorten_AAs:
                title = self.anchor_res_and_fragment_str_short
                label_bars = self.partner_res_and_fragment_labels_short
        else:
            title = self.name
            if title is None:
                title = self.__class__.__name__
            label_bars = self.ctc_labels_w_fragments_short_AA

        if defrag is not None:
            title = _mdcu.str_and_dict.defrag_key(title,defrag=defrag)
            label_bars = [_mdcu.str_and_dict.defrag_key(ilab,defrag=defrag) for ilab in label_bars]

        # Cosmetics

        title_str = "distribution for %s"%_mdcu.str_and_dict.latex_superscript_fragments(title)
        if ctc_cutoff_Ang is not None:
            title_str += "\nresidues within %2.1f $\AA$"%(ctc_cutoff_Ang)
            jax.axvline(ctc_cutoff_Ang,color="k",ls="--",zorder=-1)
        if self.neighbors_excluded not in [None,0]:
            title_str += "\n%u nearest bonded neighbors excluded" % (self.neighbors_excluded)
        jax.set_title(title_str)

        # Base plot
        for ii, ((h, x), label) in enumerate(zip(self.distributions_of_distances(bins=bins), label_bars)):
            label = _mdcu.str_and_dict.latex_superscript_fragments(label)
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
                          **plot_timetrace_kwargs,
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
        plot_timetrace_kwargs

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
        valid_cutoff = "ctc_cutoff_Ang" in plot_timetrace_kwargs.keys() \
                       and plot_timetrace_kwargs["ctc_cutoff_Ang"] > 0


        figs_to_return = []
        if self.n_frames_total==1:
            #print("Just one frame, not plotting any time-traces")
            return figs_to_return

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
                ictc._plot_timetrace(next(axes_iter),
                                     **plot_timetrace_kwargs
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
            ctc_cutoff_Ang = plot_timetrace_kwargs.pop("ctc_cutoff_Ang")
            for pkey in ["shorten_AAs", "ylim_Ang"]:
                try:
                    plot_timetrace_kwargs.pop(pkey)
                except KeyError:
                    pass
            self._plot_timedep_Nctcs(iax,
                                     ctc_cutoff_Ang,
                                     **plot_timetrace_kwargs,
                                     )
        [ifig.tight_layout(pad=0, h_pad=0, w_pad=0) for ifig in figs_to_return]
        return figs_to_return

    def _plot_timedep_Nctcs(self,
                            iax,
                            ctc_cutoff_Ang,
                            switch_off_Ang=None,
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
        for n_ctcs_t, itime, traj_name in zip(self.n_ctcs_timetraces(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang),
                                              self.time_arrays,
                                              self.trajlabels):
            _mdcplots.plot_w_smoothing_auto(iax, itime*dt, n_ctcs_t,traj_name,next(icol),
                                  gray_background=gray_background,
                                  n_smooth_hw=n_smooth_hw)

        iax.set_ylabel('$\sum$ [ctcs < %s $\AA$]'%(ctc_cutoff_Ang))
        iax.set_xlabel('t / %s'%t_unit)
        iax.set_xlim([self.time_min*dt,self.time_max*dt])
        iax.set_ylim([0,iax.get_ylim()[1]])
        iax.legend(fontsize=_rcParams["font.size"]*.75,
                   ncol=_np.ceil(self.n_trajs / max_handles_per_row).astype(int),
                   loc=1,
                   )

    def plot_frequency_sums_as_bars(self,
                                    ctc_cutoff_Ang,
                                    title_str,
                                    switch_off_Ang=None,
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
        Bar plot with per-residue sums of frequencies (called \Sigma in mdciao)

        Parameters
        ----------
        ctc_cutoff_Ang : float
        title_str : str
        xmax : float, default is None
            X-axis will extend from -.5 to xmax+.5
        jax : obj:`~matplotlib.axes.Axes``, default is None
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
            Separate residues by interface
        sort : boolean, default is True
            Sort sums of freqs in descending order
        interface_vline : bool, default is False
            Plot a vertical line visually separating both interfaces

        Returns
        -------
        ax : :obj:`~matplotlib.axes.Axes`

        """

        # Base list of dicts
        frq_dict_list = self.frequency_sum_per_residue_names(ctc_cutoff_Ang,
                                                          switch_off_Ang=switch_off_Ang,
                                                          sort=sort,
                                                          shorten_AAs=shorten_AAs,
                                                          list_by_interface=list_by_interface)

        # TODO the method plot_freqs_as_bars is very similar but
        # i think it's better to keep them separated

        # [j for i in klist for j in i]
        label_bars = [j for idict in frq_dict_list for j in idict.keys()]
        freqs = _np.array([j for idict in frq_dict_list for j in idict.values()])

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
        jax.set_xticks([])
        [jax.axhline(ii, color="lightgray", linestyle="--", zorder=-1) for ii in yticks]

        # Cosmetics
        jax.set_title(
            "Average nr. contacts @%2.1f $\AA$ \nper residue of '%s'"
            % (ctc_cutoff_Ang, _mdcu.str_and_dict.replace4latex(title_str)))

        _mdcplots.add_tilted_labels_to_patches(jax,
                                               label_bars[:(jax.get_xlim()[1]).astype(int) + 1],
                                               label_fontsize_factor=label_fontsize_factor,
                                               trunc_y_labels_at=.65 * _np.max(freqs),
                                               single_label=True,
                                               )

        if xmax is not None:
            jax.set_xlim([-.5, xmax + 1 - .5])

        if list_by_interface and interface_vline:
            xpos = len([ifreq for ifreq in frq_dict_list[0].values() if ifreq >truncate_at])
            jax.axvline(xpos-.5,color="lightgray", linestyle="--",zorder=-1)
        return jax

    def plot_freqs_as_flareplot(self,ctc_cutoff_Ang,
                                consensus_maps=None,
                                SS=None,
                                **kwargs_freqs2flare,
                                ):
        r"""
        Produce contact flareplots by thinly wrapping around :obj:`mdciao.flare.freqs2flare`


        Parameters
        ----------
        ctc_cutoff_Ang : float
        consensus_maps : list, default is None
            List containing dictionaries of consensus labels.
            The items in the list should be "gettable" by residue index
            either by being lists, arrays, or dicts, s.t.,
            the corresponding value should be the label.
        SS : secondary structure information, default is None
            Whether and how to include information about
            secondary structure. Can be many things
            * triple of ints (CP_idx, traj_idx, frame_idx)
              Go to contact group CP_idx, trajectory traj_idx
              and grab this frame to compute the SS.
              Will read xtcs when necessary or otherwise
              directly grab it from a :obj:`mdtraj.Trajectory`
              in case it was passed. Ignores potential stride
              values.
              See :obj:`ContactPair.time_traces` for more info
            * True
              same as [0,0,0]
            * None or False
              Do nothing
            * :obj:`mdtraj.Trajectory`
              Use this geometry to compute the SS
            * array_like
              Use the SS from here, s.t.ss_inf[idx]
              gives the SS-info for the residue
              with that idx
        kwargs_freqs2flare: optargs
            Keyword arguments for :obj:`mdciao.flare.freqs2flare`
            except for :obj:`top` and :obj:`ss_array`

        Returns
        -------
        ifig, iax
        """

        if consensus_maps is None:
            pass
        else:
            textlabels = []
            for rr in self.top.residues:
                clab = _mdcn.choose_between_consensus_dicts(rr.index, consensus_maps,
                                                            no_key=None)
                rlab = '%s%s' % (_mdcu.residue_and_atom.shorten_AA(rr, keep_index=True, substitute_fail="long"),
                                 _mdcu.str_and_dict.choose_options_descencing([clab], fmt='@%s'))
                textlabels.append(rlab)
            kwargs_freqs2flare["textlabels"] = textlabels

        from_tuple = False
        if SS is None or isinstance(SS,bool) and not SS:
            kwargs_freqs2flare["ss_array"] = None
        elif isinstance(SS, _md.Trajectory):
            kwargs_freqs2flare["ss_array"] = _md.compute_dssp(SS[0],simplified=True)[0]
        elif SS is True:
            from_tuple = [0,0,0]
        elif len(SS)==3:
            from_tuple = SS
        else:
            kwargs_freqs2flare["ss_array"] = SS

        # Introspect?
        if from_tuple:
            idx_cp, idx_traj, idx_frame = from_tuple
            traj = self._contacts[idx_cp].time_traces.trajs[idx_traj]
            if isinstance(traj,str):
                traj = _md.load(traj,top=self.top,frame=idx_frame)
            else:
                traj = traj[idx_frame]
                assert isinstance(traj,_md.Trajectory)
            kwargs_freqs2flare["ss_array"] = \
            _md.compute_dssp(traj)[0]

        kwargs_freqs2flare["top"]=self.top
        iax, _, _ = _mdcflare.freqs2flare(self.frequency_per_contact(ctc_cutoff_Ang),
                                       self.res_idxs_pairs,
                                       **kwargs_freqs2flare,
                                       )
        ifig = iax.figure
        ifig.tight_layout()
        return ifig, iax

    def retop(self,top, mapping, deepcopy=False):
        r"""Return a copy of this object with a different topology.

        Uses the :obj:`mapping` to generate new residue-indices
        where necessary, using the rest of the attributes
        (time-traces, labels, colors, fragments...) as they were

        Wraps thinly around :obj:`mdciao.contacts.ContactPair.retop`

        top : :obj:`~mdtraj.Topology`
            The new topology
        mapping : indexable (array, dict, list)
            A mapping of old residue indices
            to new residue indices. Usually,
            comes from aligning the old and the
            new topology using :obj:`mdciao.utils.sequence.maptops`
        deepcopy : bool, default is False
            Use :obj:`copy.deepcopy` on the attributes
            when creating the new :obj:`ContactPair`.

        Returns
        -------
        CG : :obj:`ContactGroup`
        """
        CPs = [CP.retop(top, mapping, deepcopy=deepcopy) for CP in self._contacts]
        interface_residxs = None
        if self.interface_residxs is not None:
            interface_residxs = [[mapping[ii] for ii in iintf] for iintf in self.interface_residxs]

        return ContactGroup(CPs,
                            interface_residxs=interface_residxs,
                            top=top, name=self.name,
                            neighbors_excluded=self.neighbors_excluded,
                            )

    @property
    def is_interface(self):
        r""" Whether this ContactGroup can be interpreted as an interface.

        Note
        ----
        If none of the :obj:`residxs_pairs`
        were found in the :obj:`interface_residxs`
        (both provided at initialization),
        this property will evaluate to False even if
        some indeces were parsed
        """

        return self._is_interface

    @property
    def interface_residxs(self):
        r"""
        The residues split into the interface,
        in ascending order within each member
        of the interface. Empty lists mean non residues were
        found in the interface defined at initialization

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

        labs = [[], []]
        for ii, ig in enumerate(self.interface_residxs):
            for idx in ig:
                labs[ii].append(self.residx2consensuslabel[idx])

        return labs

    @property
    def interface_residue_names_w_best_fragments_short(self):
        r"""Best possible residue@fragment string for
        the residues in :obj:`interface_residxs`


        In case neither a consensus label > fragment name > fragment index is found,
        nothing is returned after the residue name

        Returns
        -------

        """
        labs_out = []
        for ints in self.interface_residxs:
            labs_out.append([self.residx2resnamefragnamebest()[jj] for jj in ints])


        return labs_out

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
                                        switch_off_Ang=None,
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
        plot_mat_kwargs: see :obj:`plot_mat`
            pixelsize, transpose, grid, cmap, colorbar

        Returns
        -------
        iax : :obj:`~matplotlib.axes.Axes`
        fig : :obj:`matplotlib.pyplot.Figure`

        """
        assert self.is_interface
        mat = self.interface_frequency_matrix(ctc_cutoff_Ang, switch_off_Ang=switch_off_Ang)
        if label_type=='consensus':
            labels = self.interface_labels_consensus
        elif label_type=='residue':
            labels = self.interface_reslabels_short
        elif label_type=='best':
            labels = self.interface_residue_names_w_best_fragments_short
        else:
            raise ValueError(label_type)

        iax, __ = _mdcplots.plot_contact_matrix(mat,labels,
                                       transpose=transpose,
                                       **plot_mat_kwargs,
                                       )
        return iax.figure, iax

    # TODO would it be better to make use of self.interface_frequency_dict_by_consensus_labels
    def interface_frequency_matrix(self, ctc_cutoff_Ang, switch_off_Ang=None):
        r"""
        Rectangular matrix of size (N,M) where N is the length
        of the first list of :obj:`interface_residxs` and M the
        length of the second list of :obj:`interface_residxs`.

        Note
        ----
        Pairs missing from :obj:`res_idxs_pairs` will be NaNs,
        to differentiate from those pairs that were present
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
            mat = self.frequency_as_contact_matrix(ctc_cutoff_Ang,switch_off_Ang=switch_off_Ang)
            mat = mat[self.interface_residxs[0],:]
            mat = mat[:,self.interface_residxs[1]]
        return mat

    def frequency_to_bfactor(self, ctc_cutoff_Ang,
                             pdbfile,
                             geom,
                             interface_sign=False):
        r"""Save the contact frequency aggregated by residue to a pdb file

        Parameters
        ----------
        ctc_cutoff_Ang : float
        pdbfile : str
        geom : :obj:`mdtraj.Trajectory`
            Has to have the same topology as :obj:`self.top`
        interface_sign : bool, default is False
            Give the bfactor values of the
            members of the interface different sign
            s.t. the appear with different colors
            in a visualizer
        Returns
        -------
        bfactors : 1D np.array of len(self.top.n_atoms)

        """

        bfactors = self.frequency_sum_per_residue_idx_dict(ctc_cutoff_Ang, return_array=True)
        bfactors = _np.array([bfactors[aa.residue.index] for aa in self.top.atoms])
        assert geom.top == self.top, "The parsed geometry has to have the same top as self.top"
        if interface_sign:
            assert self.is_interface
            interface_0_atoms = _np.hstack([[aa.index for aa in geom.top.residue(ii).atoms] for ii in self.interface_residxs[0]])
            bfactors[interface_0_atoms] *= -1
        geom.save(pdbfile,bfactors=bfactors)
        return bfactors

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

        if t_unit not in _mdcu.str_and_dict.tunit2tunit.keys():
            raise ValueError("I don't know the time unit %s, only %s" % (t_unit, _mdcu.str_and_dict.tunit2tunit.keys()))
        dicts = []
        for ii in range(self.n_trajs):
            labels = ['time / %s'%t_unit]
            data = [self.time_arrays[ii] * _mdcu.str_and_dict.tunit2tunit["ps"][t_unit]]
            for ictc in self._contacts:
                labels.append('%s / Ang'%ictc.labels.w_fragments_short_AA)
                data.append(ictc.time_traces.ctc_trajs[ii]*10)
            data= _np.vstack(data).T
            dicts.append({"header":labels,
                          "data":data
                          }
                         )
        return dicts

    def _to_per_traj_dicts_for_saving_bintrajs(self, ctc_cutoff_Ang, switch_off_Ang=None, t_unit="ps"):
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

        if t_unit not in _mdcu.str_and_dict.tunit2tunit.keys():
            raise ValueError("I don't know the time unit %s, only %s" % (t_unit, _mdcu.str_and_dict.tunit2tunit.keys()))

        bintrajs = self.binarize_trajs(ctc_cutoff_Ang, switch_off_Ang=None, order="traj")
        labels = ['time / %s' % t_unit]
        for ictc in self._contacts:
            labels.append('%s / Ang' % ictc.label)

        dicts = []
        for ii in range(self.n_trajs):
            data = [self.time_arrays[ii] * _mdcu.str_and_dict.tunit2tunit["ps"][t_unit]] + [bintrajs[ii].T.astype(int)]
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
        Save time-traces to disk.

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

            if self.is_neighborhood:
                self_descriptor = self.anchor_res_and_fragment_str.replace('*', "")

            if ctc_cutoff_Ang is None:
                savename_fmt = "%s.%s.%s.%s"
            else:
                savename_fmt = "%s.%s.%s.bintrajs.%s"

            savename = savename_fmt % (prepend_filename.strip("."), self_descriptor.strip("."), ixtc_basename, ext.strip("."))
            savename = savename.replace(" ","_")
            savename = _path.join(output_dir, savename)
            if ext.endswith('xlsx'):
                _DF(idict["data"],
                    columns=idict["header"]).to_excel(savename,
                                                      float_format='%6.3f',
                                                      index=False)
            elif ext.endswith("npy"):
                print("am I here?",savename)
                _np.save(savename,idict)
            else:
                _np.savetxt(savename, idict["data"],
                            ' '.join(["%6.3f" for __ in idict["header"]]),
                            header=' '.join(["%6s" % key.replace(" ", "") for key in idict["header"]]))

            if verbose:
                print(savename)

    def save(self,filename):
        r"""Save this :obj:`ContactGroup` as a pickle

        Parameters
        ----------
        filename : str
            filename

        Returns
        -------

        """
        _save_as_pickle(self, filename)

    def archive(self,filename=None, **kwargs):
        r""" Save this :obj:`ContactGroup`'s list of :obj:`ContactPairs` as a list of dictionaries that
        can be used to re-instantiate an equivalent :obj:`ContactGroup`

        The method :obj:`ContactGroup.save` creates a pickle that has a lot of redundant information

        Parameters
        ----------
        filename : str, default is None
            Has to end in "npy". Default is
            to return the dictionary

        Returns
        -------
        archive : dict

        """

        tosave = {"serialized_CPs": [cp._serialized_as_dict(**kwargs) for cp in self._contacts],
                  "interface_residxs": self.interface_residxs,
                  "name": self.name,
                  "neighbors_excluded":self.neighbors_excluded}
        if filename is not None:
            assert filename.endswith("npy")
            _np.save(filename,tosave)
        else:
            return tosave

    def copy(self):
        r"""copy this object by re-instantiating another :obj:`ContactGroup` object
        with the same attributes.

        In theory self == self.copy() should hold, but not self is self.copy()

        Returns
        -------
        CG : :obj:`ContactGroup`

        """
        return ContactGroup([CP.copy() for CP in self._contacts],
                            interface_residxs=self.interface_residxs,
                            top=self.top)

    def __hash__(self):
        return hash(tuple([hash(tuple([CP.__hash__() for CP in self._contacts])),
                           hash(tuple(self.interface_residxs[0])),
                           hash(tuple(self.interface_residxs[1])),
                           hash(self.top)]))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


_hatchets = {"BB-BB": "||",
             "SC-SC": "--",
             "BB-SC": "///",
             "SC-BB": '\\\\\\',
             "X-SC": "///",
             "SC-X": "///",
             "BB-X": "||",
             "X-BB": "||"
             }

class GroupOfInterfaces(object):
    r"""Container for :obj:`ContactGroup` objects

    Note
    ----
    * ATM the only :obj:`ContactGroup` s allowed are the ones
      that were initialized with some :obj:`interface_residxs`
      so that :obj:`ContactGroup.is_interface` will yield True

    * This object does not use residue indices at all and
      performs (and reports) all lookups with consensus labels
      The idea is that each contact group can have a different
      topologies. This is why relabel_consensus is True by
      default, because we cannot allow missing consensus labels
      beyond this point

    """
    def __init__(self, dict_of_CGs, relabel_consensus=True):
        r"""

        Parameters
        ----------
        dict_of_CGs : dict
        relabel_consensus : bool, default is True


        """

        self._contact_groups = dict_of_CGs

        assert all([iint.is_interface for iint in self._contact_groups.values()]), NotImplementedError

        if relabel_consensus:
            self.relabel_consensus()

    @property
    def interfaces(self):
        return self._contact_groups

    @property
    def n_groups(self):
        return len(self.interfaces)

    @property
    def interface_names(self):
        r"""The keys with which the object was initialized"""
        return list(self.interfaces.keys())

    # TODO rename interfaces to interface_dict?
    @property
    def interface_list(self):
        return list(self._contact_groups.values())

    def relabel_consensus(self,**kwargs):
        r"""Calls :obj:`ContactGroup.relabel_consensus`
        on all underlying contact groups

        Parameters
        ----------
        ** kwargs : optional named args, e.g. new_labels

        Returns
        -------

        """
        [iintf.relabel_consensus(**kwargs) for iintf in self.interface_list]

    @property
    def conlab2matidx(self):
        r"""Maps consensus labels (strings)
        to matrix indices of the :obj:`interface_matrix`"""

        out_dict= {}
        for jj, conlabs in enumerate(self.interface_labels_consensus):
            for ii, key in enumerate(conlabs):
                assert key not in out_dict.keys(), "Something went very wrong, %s is already used for %s, but found again" \
                                                   "also for %s"%(key,out_dict[key], [ii,jj])
                assert key is not None
                out_dict[key] = [jj, ii]

        return out_dict

    def interface_matrix(self,ctc_cutoff_Ang):
        r"""Average of all interface_matrices contained in the this object

        Parameters
        ----------
        ctc_cutoff_Ang : float
            The contact cutoff in Ang

        Returns
        -------
        mat : ndarray
            The rows contain the residues in :obj:`interface_residxs`[0]
            and the columns the ones in :obj:`interface_residxs`[1].
            Their consensus labels, in the same order, can
            be found in :obj:`interface_labels_consensus`


        """

        labels = self.interface_labels_consensus
        mat = _np.zeros((len(labels[0]),len(labels[1])))
        conlab2matidx = self.conlab2matidx
        for key, iint in self.interfaces.items():
            idict = iint.frequency_dict_by_consensus_labels(ctc_cutoff_Ang, return_as_triplets=True)
            for key1, key2, freq in idict:
                key1_i, ii = conlab2matidx[key1]
                key2_i, jj = conlab2matidx[key2]
                assert key1_i == 0 and key2_i == 1
                mat[ii,jj] += freq

        mat = mat / self.n_groups
        return mat

    def interface_frequency_dict_by_consensus_labels(self, ctc_cutoff_Ang,
                                                     return_as_triplets=False,
                                                     ):
        r"""Contact frequencies of the :obj:`interface_matrix` as a sparse dictionary
        of dicts double-keyed by consensus labels, e.g. res[key1][key2] : freq



        Parameters
        ----------
        ctc_cutoff_Ang : float
        return_as_triplets : bool
            instead of a dict [key1][key2]->freq res1-res2,
            return a list with triplets[[key1,key2,freq],...]

        Returns
        -------

        """
        mat = self.interface_matrix(ctc_cutoff_Ang)
        conlab2matidx = self.conlab2matidx
        dict_out = _defdict(dict)
        for ii, jj in _np.argwhere(mat > 0):
            key1, key2 = self.interface_labels_consensus[0][ii], self.interface_labels_consensus[1][jj]
            _np.testing.assert_array_equal(conlab2matidx[key1], [0, ii]) #sanity check
            _np.testing.assert_array_equal(conlab2matidx[key2], [1, jj]) #sanity check
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

    #TODO document
    def compare(self, **kwargs):
        r"""Very thin wrapper around :obj:`mdciao.plots.compare_groups_of_contacts`


        """
        return _mdcplots.compare_groups_of_contacts(self.interfaces,
                                                    **kwargs
                                                    )

    @property
    def interface_labels_consensus(self):
        r"""Union of all underlying consensus labels,
        split into the two interface members

        Note
        ----
        The underlying :obj:`ContactGroup.interface_labels_consensus` is used

        Returns
        -------

        """

        _interface_labels_consensus = [[], []]
        for __, interface in self.interfaces.items():
            for ii, ilabs in enumerate(interface.interface_labels_consensus):
                for jlab in ilabs:
                    if jlab not in _interface_labels_consensus[ii]:
                        _interface_labels_consensus[ii].append(jlab)
        #print(_interface_labels_consensus)
        # TODO this re-ordering for proper matching
        #_interface_labels_consensus[0] = _mdcn.sort_BW_consensus_labels(_interface_labels_consensus[0])
        #_interface_labels_consensus[1] = _mdcn.sort_CGN_consensus_labels(_interface_labels_consensus[1])
        return _interface_labels_consensus

    """
    def frequency_table(self,ctc_cutoff_Ang):
        return self.interface_frequency_dict_by_consensus_labels(ctc_cutoff_Ang, return_as_triplets=True)
    
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
    
    def rename_orphaned_residues_foreach_interface(self,
                                                   alignment_as_DF,
                                                   interface_idx=0):

        assert interface_idx==0,NotImplementedError(interface_idx)

        long2short = lambda istr: ['@'.join(ival.split('@')[:2]) for ival in istr.split("_") if ival != 'None'][0]
        orphan_residues_by_short_label = _defdict(dict)
        for pdb, iint in self.interfaces.items():
            iint._orphaned_residues_new_label = {}
            for AA in iint.interface_shortAAs_missing_conslabels[interface_idx]:
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
    """

def _linear_switchoff(d, cutoff, switch_off):
    r"""
    Returns 1 for d<=cutoff, 0 for d>cutoff+switch and a linear value [1,0[ between both

    d, cutoff, and switch_off have to be in the same units
    Parameters
    ----------
    d : 1d ndarray
    cutoff : float
    switch_off : float


    Returns
    -------
    res : iterable of floats e [1,0] of len(d)

    """
    m = -1 / switch_off
    b = 1 + (cutoff / switch_off)
    res = m * _np.array(d) + b
    res[d < cutoff] = 1
    res[d > (cutoff + switch_off)] = 0
    return _np.array(res,dtype=_np.float)

def _quadratic_switchoff(d, cutoff, switch_off):
    r"""
    Returns 1 for d<=cutoff, 0 for d>cutoff+switch and a quaratic value [1,0[ between both

    d, cutoff, and switch_off have to be in the same units
    Parameters
    ----------
    d : 1d ndarray
    cutoff : float
    switch_off : float


    Returns
    -------
    res : iterable of floats e [1,0] of len(d)

    """
    # y = k(d-cutoff)^2+c
    c = 1
    k= -1/(switch_off**2)
    res = k*(d-cutoff)**2+c
    res[d < cutoff] = 1
    res[d > (cutoff + switch_off)] = 0
    return _np.array(res,dtype=_np.float)

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
    atom_pairs = ['-'.join([_mdcu.residue_and_atom.atom_type(aa) for aa in pair]) for pair in atom_pairs]
    dict_out = {key: 0 for key in atom_pairs}
    for key, count in zip(atom_pairs, counts):
        dict_out[key] += count
    return dict_out

def _contact_fraction_informer(n_kept, ctc_freqs, or_frac=.9):
    r"""
    Return the fraction of the sum(ctc_freqs) kept by using the first :obj:`n_kept` contacts

    Parameters
    ----------
    n_kept : int
        The number of contacts kept
    ctc_freqs : array-like of floats
        The frequencies in descending order
    or_frac : float, default is .9
        Orientation fraction, i.e. print how many contacts
        would be needed to caputre this fraction of the
        neighborhood. Can be None and nothing will be printed

    Returns
    -------
    None

    """
    assert all(_np.diff(ctc_freqs)<=0), "Values must be in descending order!"
    captured_freq = ctc_freqs[:n_kept].sum()
    total_freq = ctc_freqs.sum()
    if total_freq==0:
        print("No contacts formed at this frequency")
    else:
        print("These %u contacts capture %4.2f (~%u%%) of the total frequency %4.2f (over %u contacts)" %
              (n_kept, captured_freq, captured_freq / total_freq * 100, total_freq, len(ctc_freqs)))
        if or_frac is not None:
            idx = _idx_at_fraction(ctc_freqs, or_frac)
            print("As orientation value, the first %u ctcs already capture %3.1f%% of %3.2f." % (idx+1, or_frac * 100, total_freq))
            print("The %u-th contact has a frequency of %4.2f"%(idx+1, ctc_freqs[idx]))
            print()

def _idx_at_fraction(val_desc_order, frac):
    r"""
    Index of :obj:`val` where np.cumsum(val)/np.sum(val)>= frac for the first time
    Parameters
    ----------
    val_desc_order : array like of floats
        The values that the determine the sum of which a fraction will be taken
        The have to be in descending order
    frac : float
        The target fraction of sum(val) that is needed

    Returns
    -------
    n : int
        Index of val where the fraction is attained for the first time.
        For the number of entries of :obj:`val`, just use n+1
    """

    assert all(_np.diff(val_desc_order)<=0), "Values must be in descending order!"
    assert 0<=frac<=1, "Fraction has to be in [0,1] ,not %s"%frac
    normalized_cumsum = _np.cumsum(val_desc_order) / _np.sum(val_desc_order) >= frac
    return _np.flatnonzero(normalized_cumsum>=frac)[0]

def _mapatoms(top0, top1,resmapping, atom0idx2atom_name):
    r"""
    Return a map (array) of atoms of top0 on top1

    Will fail when more than two atoms of the same
    residue have the same name

    Parameters
    ----------
    top0 : :obj:`~mdtraj.Topology`
    top1 : :obj:`~mdtraj.Topology`
    resmapping : indexable (dict, list, array)
        Indexed by residue indices in top0
        valued with new residue indices in top1
    atom0idx2atom_name : dict
        Indexed by atom indices in top0,
        valued with their name, e.g. "CA"

    Returns
    -------
    atom0idx2atom1idx : 1D np.array
        1D array of len top1.n_atoms, valued
        with np.nan everywhere except for
        the indices (=keys) of :obj:`atom0idx2atom_name`,
        where its valued with the indices of their
        equivalent top1 atoms.

    """
    atom0idx2atom1idx = _np.full(top0.n_atoms, _np.nan, dtype=int)
    # it's safe to assume that the -9223372036854775808
    # int-value for nan will break things (which we want) downstream

    for ii, iname in atom0idx2atom_name.items():
        old_atom = top0.atom(ii)
        old_res = top0.residue(old_atom.residue.index)
        new_res = top1.residue(resmapping[old_res.index])
        new_atom = list(new_res.atoms_by_name(iname))
        assert len(new_atom) == 1, "The old atom %s of old residue %s (idx %u) can't be uniquely identified " \
                                   "in the new residue %s (idx %u): %s" % (str(old_atom),
                                                                           str(old_res), old_res.index,
                                                                           str(new_res), new_res.index, new_atom)
        atom0idx2atom1idx[ii] = new_atom[0].index

    return atom0idx2atom1idx
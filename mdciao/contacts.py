import numpy as _np
import mdtraj as _md
from os import path as _path
from .list_utils import in_what_fragment, \
    put_this_idx_first_in_pair

from .str_and_dict_utils import \
    _replace_w_dict,\
    unify_freq_dicts,\
    _replace4latex, \
    iterate_and_inform_lambdas \

from .plots import plot_w_smoothing_auto, plot_contact
from collections import defaultdict

import matplotlib.pyplot as _plt
from matplotlib import rcParams as _rcParams
from pandas import DataFrame as _DF
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
            order_mask = _np.array([ii for ii in order if residx in residxs_pairs[ii]])
            print("#idx    Freq  contact             segA-segB residxA   residxB   ctc_idx")
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

            total_n_ctcs = ctc_freqs[order_mask].sum()
            nc = _np.argwhere(_np.cumsum(ctc_freqs[order_mask])>=total_n_ctcs*fac)[0]+1
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



def trajs2ctcs(xtcs, top, ctc_residxs_pairs, stride=1, consolidate=True,
               chunksize=1000, return_times_and_atoms=False,
               n_jobs=1,
               progressbar=False,
               **mdcontacts_kwargs):
    """Returns the time-dependent traces of residue-residue contacts from
    a list of trajectories

    Parameters
    ----------
    xtcs : list of strings
        list of filenames with trajectory data. Typically xtcs,
        but can be any type of file readable by
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
        behind the distanes in :obj:`ctcs`. See :obj:`per_xtc_ctcs` for
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
    ctcs, time_arrays, atom_idxs if return_time=True

    """

    from tqdm import tqdm

    if progressbar:
        iterfunct = lambda a : tqdm(a)
    else:
        iterfunct = lambda a : a
    ictcs_itimes_iaps = _Parallel(n_jobs=n_jobs)(_delayed(per_xtc_ctc)(top, ixtc, ctc_residxs_pairs,chunksize,stride,ii,
                                                                  **mdcontacts_kwargs)
                                            for ii, ixtc in enumerate(iterfunct(xtcs)))
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

def per_xtc_ctc(top, ixtc, ctc_residxs_pairs, chunksize, stride,
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
    ixtc: `mdtraj.Trajctory` or filename
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
    iterate, inform = iterate_and_inform_lambdas(ixtc, chunksize, stride=stride, top=top)
    ictcs, itime, iaps = [],[],[]
    running_f = 0
    inform(ixtc, traj_idx, 0, running_f)
    for jj, igeom in enumerate(iterate(ixtc)):
        running_f += igeom.n_frames
        inform(ixtc, traj_idx, jj, running_f)
        itime.append(igeom.time)
            #TODO make lambda out of this if
        if 'scheme' in mdcontacts_kwargs.keys() and mdcontacts_kwargs["scheme"].upper()=='COM':
            jctcs = geom2COMdist(igeom, ctc_residxs_pairs)
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

def xtcs2ctc_mat_dict(xtcs, top, list_ctc_cutoff_Ang,
                      stride=1,
                      return_time=False,
                      res_COM_cutoff_Ang=25,
                      chunksize=100,
                      n_jobs=1,
                      progressbar=False,
                      **mdcontacts_kwargs):
    """Returns the full contact map of residue-residue contacts from a list of trajectory files

    Parameters
    ----------
    xtcs : list of strings
        list of filenames with trajectory data. Typically xtcs,
        but can be any type of file readable by :obj:mdtraj
    top : str or :py:class:`mdtraj.Topology`
        Topology that matches :obj:xtcs
    stride : int, default is 1
        Stride the trajectory data down by this value
    chunksize : integer, default is 100
        How many frames will be read into memory for
        computation of the contact time-traces. The higher the number,
        the higher the memory requirements
    n_jobs : int, default is 1
        to how many processors to parallellize


    Returns
    -------
    ctc_mat

    """

    from tqdm import tqdm

    if progressbar:
        iterfunct = lambda a : tqdm(a)
    else:
        iterfunct = lambda a : a

    ictc_mat_dicts_itimes = _Parallel(n_jobs=n_jobs)(_delayed(per_xtc_ctc_mat_dict)(top, ixtc, list_ctc_cutoff_Ang, chunksize, stride, ii, res_COM_cutoff_Ang,
                                                                               **mdcontacts_kwargs)
                                            for ii, ixtc in enumerate(iterfunct(xtcs)))

    ctc_maps = {key:[] for key in list_ctc_cutoff_Ang}
    times = []

    for ictc_map, itimes in ictc_mat_dicts_itimes:
        for key in list_ctc_cutoff_Ang:
            ctc_maps[key].append(ictc_map[key])
        times.append(itimes)
    actcs = ctc_maps
    times = times

    if not return_time:
        return actcs
    else:
        return actcs, times

def per_xtc_ctc_mat_dict(top, ixtc, list_ctc_cutoff_Ang, chunksize, stride,
                         traj_idx, res_COM_cutoff_Ang,
                         **mdcontacts_kwargs):

    from .actor_utils import igeom2mindist_COMdist_truncation
    iterate, inform = iterate_and_inform_lambdas(ixtc,chunksize, stride=stride, top=top)
    ictcs, itime, iaps = [],[],[]
    running_f = 0
    inform(ixtc, traj_idx, 0, running_f)
    ctc_sum = {icoff: _np.zeros((top.n_residues, top.n_residues), dtype=int) for icoff in list_ctc_cutoff_Ang}

    for jj, igeom in enumerate(iterate(ixtc)):
        running_f += igeom.n_frames
        inform(ixtc, traj_idx, jj, running_f)
        itime.append(igeom.time)

        ctcs_mins, ctc_residxs_pairs, COMs_under_cutoff_pair_idxs  = igeom2mindist_COMdist_truncation(igeom,
                                                                                                      res_COM_cutoff_Ang,
                                                                                                      CA_switch=True)
        jctcs, jidx_pairs, j_atompairs = compute_contacts(igeom, ctc_residxs_pairs, **mdcontacts_kwargs)
        # TODO do proper list comparison and do it only once
        assert len(jidx_pairs) == len(ctc_residxs_pairs)
        for icoff in list_ctc_cutoff_Ang:
            counts = (jctcs<=icoff/10).sum(0)
            positive_idxs = _np.argwhere(counts>0).squeeze()
            for idx in positive_idxs:
                ii, jj  = sorted(jidx_pairs[idx])
                ctc_sum[icoff][ii,jj] += counts[idx]
                ctc_sum[icoff][jj,ii] += counts[idx]
    return ctc_sum, itime

def geom2COMdist(igeom, residue_pairs):
    r"""
    Returns the distances between center-of-mass (COM)
    Parameters
    ----------
    igeom: :obj:`mdtraj.Trajectory`
    residue_pairs: iterable of integer pairs
        pairs of residues by their zero-indexed serial indexes

    Returns
    -------
    COMs_array : np.ndarray of shape(igeom.n_frames, len(res_pairs))
        contains the time-traces of the residue COMs
    """
    from scipy.spatial.distance import pdist, squareform
    residue_idxs_unique, pair_map = _np.unique(residue_pairs, return_inverse=True)
    pair_map = pair_map.reshape(len(residue_pairs),2)


    COMs_xyz = geom2COMxyz(igeom, residue_idxs=residue_idxs_unique)[:,residue_idxs_unique]

    # Only do pdist of the needed residues
    COMs_dist_triu = _np.array([pdist(ixyz) for ixyz in COMs_xyz])

    # Get square matrices
    COMs_square = _np.array([squareform(trupper) for trupper in COMs_dist_triu])

    # Use the pair map to get the right indices
    COMs_array = _np.array([COMs_square[:,ii,jj] for ii,jj in pair_map]).T

    return COMs_array


def geom2COMxyz(igeom, residue_idxs=None):
    r"""
    Returns the time trace of per-residue
    center-of-masses (COMs) in cartesian coordinates

    Parameters
    ----------
    igeom : :obj:`mdtraj.Trajectory`

    residue_idxs : iterable, default is None
        Residues for which the center of mass will be computed. Default
        is to compute all residues. The excluded residues will appear
        as np.nans in the returned value, which has the same shape
        regardless of the input

    Returns
    -------
    rCOMs : numpy.ndarray of shape (igeom.n_frames, igeom.n_residues,3)

    """

    if residue_idxs is None:
        residue_idxs=_np.arange(igeom.top.n_residues)
    masses = [_np.hstack([aa.element.mass for aa in rr.atoms]) for rr in
              igeom.top.residues]
    COMs_res_time_coords = _np.zeros((igeom.n_residues,igeom.n_frames,3))
    COMs_res_time_coords[:,:,:] = _np.nan
    COMs_res_time_coords[residue_idxs] = [_np.average(igeom.xyz[:, [aa.index for aa in igeom.top.residue(index).atoms], :], axis=1, weights=masses[index])
                            for index in residue_idxs]
    COMs_time_res_coords = _np.swapaxes(_np.array(COMs_res_time_coords),0,1)
    return COMs_time_res_coords

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

    actcs = trajs2ctcs(trajectories, top, ctc_idxs, stride=stride, chunksize=50,
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

def contact_matrix_slim(trajectories, cutoff_Ang=3,
                       **mdcontacts_kwargs):
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

    actcs = trajs2ctcs(trajectories, top, ctc_idxs, stride=stride, chunksize=50,
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

def pick_best_label(fallback, test, exclude=[None, "None", "NA", "na"]):
    if test not in exclude:
        return test
    else:
        return fallback

class contact_group(object):
    r"""Class for containing contact objects, ideally
    it can be used for vicinities, sites, interfaces etc"""

    def __init__(self,
                 list_of_contact_objects,
                 interface_residxs=None,
                 top=None):
        self._contacts = list_of_contact_objects
        self._n_ctcs  = len(list_of_contact_objects)
        self._interface_residxs = interface_residxs
        if top is None:
            self._top = self._unique_topology_from_ctcs()
        else:
            assert top is self._unique_topology_from_ctcs()
            self._top = top

        # Sanity checks about having grouped this contacts together

        # All contacts have the same number of trajs
        self._n_trajs =_np.unique([ictc.n_trajs for ictc in self._contacts])
        if self.n_ctcs==0:
            raise NotImplementedError("This contact group has no contacts!")
        else:
            assert len(self._n_trajs)==1, (self.n_trajs, [ictc.n_trajs for ictc in self._contacts])

            self._n_trajs=self._n_trajs[0]

            # All trajs have the same times
            ref_ctc = self._contacts[0]
            assert all([_np.allclose(ref_ctc.n_frames, ictc.n_frames) for ictc in self._contacts[1:]])
            self._time_arrays=ref_ctc.time_arrays
            self._time_max = ref_ctc.time_max
            self._n_frames = ref_ctc.n_frames

            # All contatcs have the same trajlabels
            already_printed = False
            for ictc in self._contacts[1:]:
                try:
                    assert all([rlab.__hash__() == tlab.__hash__()
                                for rlab, tlab in zip(ref_ctc.trajlabels, ictc.trajlabels)])
                except AttributeError:
                    if not already_printed:
                        print("Trajectories unhashable, could not verify they are the same")
                        already_printed = True
                    else:
                        pass

            self._trajlabels = ref_ctc.trajlabels
            self._cons2resname = {}
            for key, val in zip(_np.hstack(self.consensus_labels),
                                _np.hstack(self.residue_names_short)):
                if str(key).lower() in ["na","none"]:
                    key = val
                if key not in self._cons2resname.keys():
                    self._cons2resname[key]=val
                else:
                    assert self._cons2resname[key]==val,(self._cons2resname[key],key,val)

            self._resname2cons = {val: key for key, val in self._cons2resname.items()}

    #def
    #import inspect
    #>> > inspect.getmembers(MyClass, lambda a: not (inspect.isroutine(a)))

    #todo there is redundant code for generating interface labels!
    @property
    def cons2resname(self):
        return self._cons2resname

    @property
    def resname2cons(self):
        return self._resname2cons

    def frequency_dict_by_consensus_labels(self, ctc_cutoff_Ang,
                                           return_as_triplets=False,
                                           sort_by_interface=False):
        dict_out = defaultdict(dict)
        for (key1, key2), ifreq in zip(self.consensus_labels,
                                       self.frequency_per_contact(ctc_cutoff_Ang)):

            dict_out[key1][key2] = ifreq
        dict_out = {key:val for key,val in dict_out.items()}

        if sort_by_interface:
            _dict_out = {key:dict_out[key] for key in self.interface_labels_consensus[0] if key in dict_out.keys()}
            assert len(_dict_out)==len(dict_out)
            dict_out = _dict_out
            _dict_out = {key:{key2:val[key2] for key2 in self.interface_labels_consensus[1] if key2 in val.keys()} for key,val in dict_out.items()}
            assert all([len(val1)==len(val2) for val1, val2 in zip(dict_out.values(), _dict_out.values())])
            dict_out = _dict_out

        if return_as_triplets:
            _dict_out = []
            for key, val in dict_out.items():
                for key2, val2 in val.items():
                    _dict_out.append([key, key2, val2])
            dict_out = _dict_out
        return dict_out

    @property
    def interface_residxs(self):
        res = None
        if self._interface_residxs is not None:
            res = []
            for ig in self._interface_residxs:
                res.append(sorted(set(ig).intersection(_np.unique(self.res_idxs_pairs,
                                 #return_index=True
                                 ))))
        return res

    @property
    def interface_labels(self):
        labs_out = [[],[]]
        for ii, ints in enumerate(self.interface_residxs):
            for idx in ints:
                ctc_idx, pair_idx = self.residx2ctcidx(idx)[0]
                labs_out[ii].append(self.ctc_labels[ctc_idx].split("-")[pair_idx])

        return labs_out

    @property
    def consensus_labels(self):
        return [ictc.consensus_labels for ictc in self._contacts]

    @property
    def residue_names_short(self):
        return [ictc.residue_names_short for ictc in self._contacts]

    @property
    def interface_labels_consensus(self):
        if self._interface_residxs is not None\
                and not hasattr(self,"_interface_labels_consensus"):
            labs = [[],[]]
            for ii, ig in enumerate(self.interface_residxs):
                for idx in ig:
                    ctc_idx, res_idx = self.residx2ctcidx(idx)[0]
                    labs[ii].append(self.consensus_labels[ctc_idx][res_idx])

            return labs
        elif hasattr(self,"_interface_labels_consensus"):
            return self._interface_labels_consensus

    @property
    def interface_orphaned_labels(self):
        return [[AA for AA in conlabs if "." not in AA] for conlabs in
                self.interface_labels_consensus]

    def interface_relabel_orphans(self):
        labs_out =[[],[]]
        for ii, labels in enumerate(self.interface_labels_consensus):
            for jlab in labels:
                if jlab in self._orphaned_residues_new_label.keys():
                    new_lab = self._orphaned_residues_new_label[jlab]
                    print(jlab, new_lab)
                    labs_out[ii].append(new_lab)
                else:
                    labs_out[ii].append(jlab)

        self._interface_labels_consensus=labs_out
        #return labs_out

    def residx2ctcidx(self,idx):
        res = []
        for ii, pair in enumerate(self.res_idxs_pairs):
            if idx in pair:
                res.append([ii,_np.argwhere(pair==idx).squeeze()])
        return _np.vstack(res)

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
    def ctc_labels(self):
        return [ictc.ctc_label for ictc in self._contacts]

    @property
    def ctc_labels_short(self):
        return [ictc.ctc_label_short for ictc in self._contacts]

    @property
    def res_idxs_pairs(self):
        return _np.vstack([ictc.res_idxs_pair for ictc in self._contacts])

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

    def frequency_per_contact(self, ctc_cutoff_Ang):
        return _np.array([ictc.frequency_overall_trajs(ctc_cutoff_Ang) for ictc in self._contacts])

    def distro_per_contact(self, nbins=10):
        return [ictc.distro_overall_trajs(bins=nbins) for ictc in self._contacts]

    def binarize_trajs(self, ctc_cutoff_Ang, order='contact'):
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

    def timedep_n_ctcs(self, ctc_cutoff_Ang):
        bintrajs = self.binarize_trajs(ctc_cutoff_Ang,order='traj')
        _n_ctcs_t = []
        for itraj in bintrajs:
            _n_ctcs_t.append(itraj  .sum(1))
        return _n_ctcs_t

    def add_ctc_type_to_histo(self, ctc_cutoff_Ang, jax):
        ctc_type_dict = self.frequency_dict_formed_atom_pairs_overall_trajs(ctc_cutoff_Ang)
        print(ctc_type_dict)
        pass
        #frequency_dict_formed_atom_pairs_overall_trajs

    def histo(self, ctc_cutoff_Ang,
              jax=None,
              truncate_at=None,
              bar_width_in_inches=.75,
              ):
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

    def histo_site(self,
                   ctc_cutoff_Ang,
                   site_name,
                   xlim=None,
                   jax=None,
                   shorten_AAs=False,
                   label_fontsize_factor=1,
                   truncate_at=0):

        # Base plot
        jax = self.histo(ctc_cutoff_Ang,
                         jax=jax, truncate_at=truncate_at)
        # Cosmetics
        jax.set_title(
            "Contact frequency @%2.1f $\AA$ of site '%s'\n"
            % (ctc_cutoff_Ang, site_name))

        label_bars = [ictc.ctc_label for ictc in self._contacts]
        if shorten_AAs:
            label_bars = [ictc.ctc_label_short for ictc in self._contacts]

        label_bars = [ilab.replace("@None","") for ilab in label_bars]

        self.add_tilted_labels_to_patches(jax,
                                          label_bars[:(jax.get_xlim()[1]).astype(int)+1],
                                          label_fontsize_factor=label_fontsize_factor
                                          )

        #jax.legend(fontsize=_rcParams["font.size"] * label_fontsize_factor)
        if xlim is not None:
            jax.set_xlim([-.5, xlim + 1 - .5])

        return jax

    def histo_neighborhood(self, ctc_cutoff_Ang,
                           n_nearest,
                           xlim=None,
                           jax=None,
                           shorten_AAs=False,
                           label_fontsize_factor=1,
                           sum_neighbors=True):

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

        if sum_neighbors:
            # HACK to avoid re-computing the frequencies
            label_dotref +='\n$\Sigma$ = %2.1f'%_np.sum([ipatch.get_height() for ipatch in jax.patches])

        jax.plot(-1, -1, 'o',
                 color=self.anchor_fragment_color,
                 label=_replace4latex(label_dotref))

        self.add_tilted_labels_to_patches(jax,
                                          label_bars,
                                          label_fontsize_factor=label_fontsize_factor)

        jax.legend(fontsize=_rcParams["font.size"]*label_fontsize_factor)
        if xlim is not None:
            jax.set_xlim([-.5, xlim + 1 - .5])

        return jax

    def distro_neighborhood(self,
                            nbins=10,
                           xlim=None,
                           jax=None,
                           shorten_AAs=False,
                            ctc_cutoff_Ang=None,
                            n_nearest=None,
                           label_fontsize_factor=1,
                            max_handles_per_row=4):


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
        for ii, ((h, x), label) in enumerate(zip(self.distro_per_contact(nbins=nbins), label_bars)):
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


        return jax

    def histo_summary(self,
                      ctc_cutoff_Ang,
                      site_name,
                      xlim=None,
                      jax=None,
                      shorten_AAs=False,
                      label_fontsize_factor=1,
                      truncate_at=0,
                      bar_width_in_inches=.75,
                      list_by_interface=False,
                      sort=True):

        # Base dict
        freqs_dict = self.frequency_per_residue(ctc_cutoff_Ang,
                                                sort=sort,
                                                list_by_interface=list_by_interface)
        # TODO this code is repeated in table_by_residue
        if list_by_interface:
            label_bars = list(freqs_dict[0].keys())+list(freqs_dict[1].keys())
            freqs = _np.array(list(freqs_dict[0].values())+list(freqs_dict[1].values()))
        else:
            label_bars, freqs = list(freqs_dict.keys()),list(freqs_dict.values())

        # Truncate
        label_bars = [label_bars[ii] for ii in _np.argwhere(freqs>truncate_at).squeeze()]
        freqs = freqs[freqs>truncate_at]

        xvec = _np.arange(len(freqs))
        if jax is None:
            _plt.figure(figsize=(_np.max((7, bar_width_in_inches * len(freqs))), 5))
            jax = _plt.gca()

        patches = jax.bar(xvec, freqs,
                          width=.25)
        yticks = _np.arange(.5,_np.max(freqs), .5)
        jax.set_yticks(yticks)
        #jax.set_xticks([])
        [jax.axhline(ii, color="lightgray", linestyle="--", zorder=-1) for ii in yticks]

        # Cosmetics
        jax.set_title(
            "Average nr. contacts @%2.1f $\AA$ \nper residue of site '%s'"
            % (ctc_cutoff_Ang, site_name))


        label_bars = [ilab.replace("@None", "") for ilab in label_bars]

        self.add_tilted_labels_to_patches(jax,
                                          label_bars[:(jax.get_xlim()[1]).astype(int) + 1],
                                          label_fontsize_factor=label_fontsize_factor,
                                          trunc_y_labels_at=.65*_np.max(freqs)
                                          )

        # jax.legend(fontsize=_rcParams["font.size"] * label_fontsize_factor)
        if xlim is not None:
            jax.set_xlim([-.5, xlim + 1 - .5])


        return jax

    def table_summary_to_excel(self, ctc_cutoff_Ang,
                               fname_excel,
                               sort=False,
                               write_interface=True,
                               offset=0,
                               **freq_tab_kwargs):

        columns = ["label",
                   "freq",
                   "sum",
                   ]
        if "breakdown" in freq_tab_kwargs.keys() and freq_tab_kwargs["breakdown"]:
            columns += ["breakdown"]
        from pandas import ExcelWriter as _ExcelWriter
        writer = _ExcelWriter(fname_excel, engine='xlsxwriter')
        workbook = writer.book
        writer.sheets["pairs by frequency"] = workbook.add_worksheet('pairs by frequency')
        writer.sheets["pairs by frequency"].write_string(0, offset,
                                      'pairs by contact frequency at %2.1f Angstrom'%ctc_cutoff_Ang)
        offset+=1
        self.frequency_table(ctc_cutoff_Ang, **freq_tab_kwargs).round({"freq": 2, "sum": 2}).to_excel(writer,
                                                                                   index=False,
                                                                                   sheet_name='pairs by frequency',
                                                                                   startrow=offset,
                                                                                   startcol=0,
                                                                                   columns=columns,
                                                                                     )
        offset = 0
        writer.sheets["residues by frequency"] = workbook.add_worksheet('residues by frequency')
        writer.sheets["residues by frequency"].write_string(offset, 0, 'Av. # ctcs (<%2.1f Ang) by residue '%ctc_cutoff_Ang)
        if write_interface:
            offset+=1

            idfs = self.frequency_per_residue(ctc_cutoff_Ang,
                                              sort=sort,
                                              list_by_interface=True,
                                              return_as_dataframe=True)


            idfs[0].round({"freq": 2}).to_excel(writer,
                                                  sheet_name='residues by frequency',
                                                  startrow=offset,
                                                  startcol=0,
                                                  columns=[
                                                      "label",
                                                      "freq"],
                                                  index=False
                                                  )
            #Undecided about best placement for these
            idfs[1].round({"freq": 2}).to_excel(writer,
                                                     sheet_name='residues by frequency',
                                                     startrow=offset,
                                                     startcol=2+1,
                                                     columns=[
                                                         "label",
                                                         "freq"],
                                                     index=False
                                                     )

        writer.save()

    def contact_map(self,
                    ctc_cutoff_Ang):
        mat = _np.zeros((self.top.n_residues, self.top.n_residues))
        mat[:,:] = _np.nan
        for (ii, jj), freq in zip(self.res_idxs_pairs, self.frequency_per_contact(ctc_cutoff_Ang)):
            mat[ii,jj] = freq
            mat[jj,ii] = freq

        return mat

    def add_tilted_labels_to_patches(self, jax, labels,
                                     label_fontsize_factor=1,
                                     trunc_y_labels_at=.65):
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

    def plot_timedep_ctcs(self, panelheight,
                          plot_N_ctcs=True,
                          pop_N_ctcs=False,
                          skip_timedep=False,
                          **plot_contact_kwargs,
                          ):

        figs_to_return = []
        if pop_N_ctcs:
            assert plot_N_ctcs, "If just_N_ctcs is True, plot_N_ctcs has to be True also"

            fig_N_ctcs = _plt.figure(
                figsize=(10, panelheight),
            )
            ax_N_ctcs = _plt.gca()

        if self.n_ctcs > 0 and not skip_timedep:
            n_rows = self.n_ctcs
            if plot_N_ctcs:
                n_rows +=1
            myfig, myax = _plt.subplots(n_rows, 1,
                                        figsize=(10, n_rows * panelheight),
                                        squeeze=False)
            figs_to_return.append(myfig)
            myax = myax[:,0]
            axes_iter = iter(myax)

            # Plot individual contacts
            for ictc in self._contacts:
                plot_contact(ictc,next(axes_iter),
                             **plot_contact_kwargs
                             )


            # Cosmetics
            [iax.set_xticklabels([]) for iax in myax[:self.n_ctcs-1]]
            [iax.set_xlabel('') for iax in myax[:self.n_ctcs - 1]]

            # TODO figure out how to put xticklabels on top
            axtop, axbottom = myax[0], myax[self.n_ctcs-1]
            iax2 = axtop.twiny()
            iax2.set_xticks(axbottom.get_xticks())
            iax2.set_xticklabels(axbottom.get_xticklabels())
            iax2.set_xlim(axtop.get_xlim())
            iax2.set_xlabel(axbottom.get_xlabel())

        if "ctc_cutoff_Ang" in plot_contact_kwargs.keys() \
                and plot_contact_kwargs["ctc_cutoff_Ang"] > 0:
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
            self.plot_timedep_Nctcs(iax,
                                    ctc_cutoff_Ang,
                                    **plot_contact_kwargs,
                                    )

        [ifig.tight_layout(pad=0, h_pad=0, w_pad=0) for ifig in figs_to_return]
        return figs_to_return

    def plot_timedep_Nctcs(self,
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
        for n_ctcs_t, itime, traj_name in zip(self.timedep_n_ctcs(ctc_cutoff_Ang),
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

    # TODO document this to say that these labels are already ordered bc
    # within one given contact_group object/interface, the
    # residues can be sorted according to their order
    @property
    def interface_reslabels_short(self):
        if self._interface_residxs is not None:
            labs = [[], []]
            for ii, ig in enumerate(self.interface_residxs):
                for idx in ig:
                    ctc_idx, res_idx = self.residx2ctcidx(idx)[0]
                    labs[ii].append(self.residue_names_short[ctc_idx][res_idx])

            return labs

    def plot_interface_matrix(self,ctc_cutoff_Ang,
                              transpose=False,
                              label_type='consensus',
                              **plot_mat_kwargs,
                              #label_type='residue',
                              #label_type='both'
                              ):
        mat = self.interface_matrix(ctc_cutoff_Ang)
        if label_type=='consensus':
            labels = self.interface_labels_consensus
        elif label_type=='residue':
            labels = self.interface_reslabels_short
        elif label_type=='both':
            labels = [['%8s %8s'%(ilab,jlab) for ilab, jlab in zip(conlabs,reslabs)]
                      for conlabs, reslabs in zip(
                    self.interface_reslabels_short,
                    self.interface_labels_consensus
                )
                      ]

        iax, __ = _plot_interface_matrix(mat,labels,
                               transpose=transpose,
                                         **plot_mat_kwargs,
                               )
        return iax.figure, iax

    # TODO would it be better to make use of self.frequency_dict_by_consensus_labels
    def interface_matrix(self,ctc_cutoff_Ang):
        mat = None
        if self._interface_residxs is not None:
            mat = _np.zeros((len(self.interface_residxs[0]),
                             len(self.interface_residxs[1])))
            freqs = self.frequency_per_contact(ctc_cutoff_Ang)
            for ii, idx1 in enumerate(self.interface_residxs[0]):
                for jj, idx2 in enumerate(self.interface_residxs[1]):
                    for kk, pair in enumerate(self.res_idxs_pairs):
                        if _np.allclose(sorted(pair),sorted([idx1,idx2])):
                            mat[ii,jj]=freqs[kk]

        return mat

    def to_per_traj_dicts_for_saving(self, dt=1, t_unit="ps"):
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

    def to_per_traj_dicts_for_saving_N_ctcs(self, ctc_cutoff_Ang, dt=1, t_unit="ps"):
        bintrajs = self.binarize_trajs(ctc_cutoff_Ang, order="traj")
        labels = ['time / %s' % t_unit]
        for ictc in self._contacts:
            labels.append('%s / Ang' % ictc.ctc_label)

        dicts = []
        for ii in range(self.n_trajs):
            data = [self.time_arrays[ii]*dt]+[bintrajs[ii].T.astype(int)]
            data= _np.vstack(data).T
            dicts.append({"header":labels,
                          "data":data
                          }
                         )
        return dicts

    def frequency_table_by_residue(self, ctc_cutoff_Ang,
                                   list_by_interface=False):
        dict_list = self.frequency_per_residue(ctc_cutoff_Ang,
                                               list_by_interface=list_by_interface)

        if list_by_interface:
            label_bars = list(dict_list[0].keys()) + list(dict_list[1].keys())
            freqs = _np.array(list(dict_list[0].values()) + list(dict_list[1].values()))
        else:
            label_bars, freqs = list(dict_list.keys()), list(dict_list.values())

        return _DF({"label":label_bars,
                    "freq":freqs})

    def frequency_table(self, ctc_cutoff_Ang,
                        breakdown=False,
                        **ctc_fd_kwargs):
        idf = _DF([ictc.frequency_dict(ctc_cutoff_Ang, **ctc_fd_kwargs) for ictc in self._contacts])
        df2return = idf.join(_DF(idf["freq"].values.cumsum(), columns=["sum"]))

        if breakdown:
            idf = self.frequency_dict_formed_atom_pairs_overall_trajs(ctc_cutoff_Ang)
            idf = ['(%s)'%(', '.join(['%2u%% %s'%(val*100,key) for key, val in idict.items()])) for idict in idf]
            df2return = df2return.join(_DF.from_dict({"breakdown": idf}))

        return df2return

    def frequency_dict_formed_atom_pairs_overall_trajs(self, ctc_cutoff_Ang):
        return [ictc.frequency_dict_formed_atom_pairs_overall_trajs(ctc_cutoff_Ang) for ictc in self._contacts]


    def frequency_per_residue_idx(self, ctc_cutoff_Ang):
        dict_sum = defaultdict(list)
        for (idx1,idx2), ifreq in zip(self.res_idxs_pairs,
                                      self.frequency_per_contact(ctc_cutoff_Ang)):
            dict_sum[idx1].append(ifreq)
            dict_sum[idx2].append(ifreq)
        dict_sum = {key:_np.sum(val) for key, val in dict_sum.items()}
        return dict_sum

    def frequency_per_residue(self, ctc_cutoff_Ang,
                              sort=True,
                              list_by_interface=False,
                              return_as_dataframe=False):
        freqs = self.frequency_per_residue_idx(ctc_cutoff_Ang)
        dict_out = {}
        keys = list(freqs.keys())
        if sort:
            keys = [keys[ii] for ii in _np.argsort(list(freqs.values()))[::-1]]
        for idx in keys:
            ifreq = freqs[idx]
            ctc_idx, pair_idx = self.residx2ctcidx(idx)[0]
            dict_out[self.ctc_labels[ctc_idx].split("-")[pair_idx]]=ifreq

        if list_by_interface:
            _dict_out = [{},{}]
            for ii, ilabs in enumerate(self.interface_labels):
                #print(ilabs)
                if sort:
                    for idx in dict_out.keys():
                        if idx in ilabs:
                            _dict_out[ii][idx]=dict_out[idx]
                else:
                    for jlab in ilabs:
                       _dict_out[ii][jlab]=dict_out[jlab]
            dict_out = _dict_out

            if return_as_dataframe:
                dict_out = [_DF({"label": list(dict_out[ii].keys()),
                                 "freq":  list(dict_out[ii].values())}) for ii in [0, 1]]
        else:
            if return_as_dataframe:
                _DF({"label": list(dict_out.keys()),
                     "freq": list(dict_out.values())})


        return dict_out

    def save_trajs(self, output_desc,
                   ext,
                   output_dir='.',
                   dt=1,
                   t_unit="ps",
                   basename=False,
                   verbose=False,
                   ctc_cutoff_Ang=None):

        if ctc_cutoff_Ang is None:
            dicts = self.to_per_traj_dicts_for_saving(dt=dt, t_unit=t_unit)
        else:
            dicts = self.to_per_traj_dicts_for_saving_N_ctcs(ctc_cutoff_Ang, dt=dt, t_unit=t_unit)

        if str(ext).lower()=="none":
            ext='dat'

        for idict, ixtc  in zip(dicts, self.trajlabels):
            jxtc = ixtc
            if basename:
                jxtc = _path.basename(ixtc)

            if ctc_cutoff_Ang is None:
                savename = "%s.%s.%s.%s" % (
                    output_desc, self.anchor_res_and_fragment_str.replace('*', ""), jxtc, ext.strip("."))
            else:
                savename = "%s.%s.%s.N_ctcs.%s" % (
                    output_desc, self.anchor_res_and_fragment_str.replace('*', ""), jxtc, ext.strip("."))
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

class contact_pair(object):
    r"""Class for storing everything related to a contact"""
    #todo consider packing some of this stuff in the site_obj class
    def __init__(self, res_idxs_pair,
                 ctc_trajs,
                     time_arrays,
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
        res_idxs_pair : list of residue index pair. The list will have only two values, each corresponding to the
                        serial number of the residue index.
        ctc_trajs : list of list, the code converts it into a list of array
        time_arrays : list of list,
        top : :py:class:`mdtraj.Topology`
        trajs:
        fragment_idxs :
        fragment_names :
        fragment_colors :
        anchor_residue_idx :
        consensus_labels :
        """

        self._res_idxs_pair = res_idxs_pair
        self._ctc_trajs = [_np.array(itraj) for itraj in ctc_trajs]
        self._top = top
        self._trajs = trajs
        self._atom_pair_trajs = atom_pair_trajs
        self._time_arrays = time_arrays
        self._n_trajs = len(ctc_trajs)
        assert self._n_trajs == len(time_arrays)
        assert all([len(itraj)==len(itime) for itraj, itime in zip(ctc_trajs, time_arrays)])
        self._time_max = _np.max(_np.hstack(time_arrays))
        self._binarized_trajs = {}

        self._anchor_residue_index = anchor_residue_idx
        self._partner_residue_index = None
        self._anchor_index = None
        self._partner_index = None
        self._anchor_residue = None
        self._partner_residue = None
        if self._anchor_residue_index is not None:
            assert self._anchor_residue_index in self.res_idxs_pair
            self._anchor_index  = _np.argwhere(self.res_idxs_pair == self.anchor_residue_index).squeeze()
            self._partner_index = _np.argwhere(self.res_idxs_pair != self.anchor_residue_index).squeeze()
            self._partner_residue_index = self.res_idxs_pair[self.partner_index]
            if self.top is not None:
                self._anchor_residue  = self.top.residue(self.anchor_residue_index)
                self._partner_residue = self.top.residue(self.partner_residue_index)

        self._consensus_labels = consensus_labels
        self._fragment_idxs  = fragment_idxs
        if fragment_names is None:
            # assert self.fragment_idxs is not None
            # self._fragment_names = self._fragment_idxs

            if self.fragment_idxs is not None:
                self._fragment_names = self._fragment_idxs
            else:
                self._fragment_names = None
        else:
            self._fragment_names = fragment_names
        self._fragment_colors = fragment_colors

    # TODO reorder so that anchor_residue always comes first??

    #TODO many of these properties will fail if partner nor anchor are None
    # todo many of these properties could be simply methods with options
    # to reduce code

    @property
    def atom_pair_trajs(self):
        return self._atom_pair_trajs

    def count_formed_atom_pairs(self, ctc_cutoff_Ang,
                                sort=True):
                                #use_atom_names=False):
        bintrajs = self.binarize_trajs(ctc_cutoff_Ang)
        formed_atom_pair_trajs = [atraj[itraj==1] for atraj, itraj in zip(self.atom_pair_trajs, bintrajs)]
        formed_atom_pairs = _np.vstack(formed_atom_pair_trajs)
        formed_atom_pairs_hashes = _np.zeros_like(formed_atom_pairs, dtype=_np.int64)

        hash2pair = {}
        for ii, fap in enumerate(formed_atom_pairs):
            ihash = hash(tuple(fap))
            formed_atom_pairs_hashes[ii]= ihash
            if ihash not in hash2pair.keys():
                hash2pair[ihash]=fap

        counts = []
        for ihash, ipair in hash2pair.items():
            counts.append((formed_atom_pairs_hashes == ihash).sum())
        keys = list(hash2pair.values())
        if sort:
            keys = [keys[ii] for ii in _np.argsort(counts)[::-1]]
            counts=sorted(counts)[::-1]

        return keys, counts


    def frequency_dict_formed_atom_pairs_overall_trajs(self, ctc_cutoff_Ang,
                                                keep_resname=False,
                                                consolidate=True,
                                                min_percentage=5):
        keys, counts = self.count_formed_atom_pairs(ctc_cutoff_Ang)

        if consolidate:
            keys = ['-'.join([_atom_type(self.top.atom(ii)) for ii in key]) for key in keys]
            dict_out = {key:0 for key in keys}
            for key, count in zip(keys, counts):
                dict_out[key] += count
            return  {key: val / _np.sum(counts) for key, val in dict_out.items() if val / _np.sum(counts) > min_percentage / 100}

        else:
            if keep_resname:
                keys = ['-'.join([str(self.top.atom(ii)) for ii in key]) for key in keys]
            else:
                keys = ['-'.join([str(self.top.atom(ii)).name for ii in key]) for key in keys]
            return {key:count for key, count in zip(keys, counts)}


    @property
    def fragment_names(self):
        """

        Returns
        -------
        list of list, Fragment names if passed, else fragment idxs. If both are not available then None(default)

        """
        return self._fragment_names

    @property
    def fragment_idxs(self):
        """

        Returns
        -------
        list of list, Fragment idxs if passed, else None(default)

        """
        return self._fragment_idxs

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
            trajlabels = ['traj %u'%ii for ii in range(self.n_trajs)]
        else:
            if isinstance(self.trajs[0], _md.Trajectory):
                trajlabels = ['mdtraj.%02u'%ii for ii in range(self.n_trajs)]
            else:
                trajlabels = [_path.splitext(ii)[0] for ii in self.trajs]

        return trajlabels

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
        return [len(itraj) for itraj in self.ctc_trajs]

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
    def res_idxs_pair(self):
        """

        Returns
        -------
        list of residue index pair passed

        """
        return self._res_idxs_pair

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
    def residue_names(self):
        """

        Returns
        -------
        list, for each residue index in the residue contact pair, the corresponding residue name from the topology file.
        example : ['GLU30','VAL212']

        """
        return [str(self.topology.residue(ii)) for ii in self.res_idxs_pair]

    @property
    def residue_names_short(self):
        """

        Returns
        -------
        list, for each residue name in the residue contact pair, the corresponding short residue name from the topology file.
        example : ['E30', 'V212']

        """
        from .aa_utils import shorten_AA as _shorten_AA
        return [_shorten_AA(rr, substitute_fail="long", keep_index=True) for rr in self.residue_names]

    @property
    def ctc_label(self):
        """

        Returns
        -------
        str,

        """
        ctc_label = '%s@%s-%s@%s' % (self.residue_names[0],
                                     pick_best_label(self.fragment_names[0], self.consensus_labels[0]),
                                     self.residue_names[1],
                                     pick_best_label(self.fragment_names[1], self.consensus_labels[1]))
        return ctc_label

    @property
    def ctc_label_short(self):
        """

        Returns
        -------
        str,

        """
        ctc_label = '%s@%s-%s@%s' % (self.residue_names_short[0],
                                     pick_best_label(self.fragment_names[0], self.consensus_labels[0]),
                                     self.residue_names_short[1],
                                     pick_best_label(self.fragment_names[1], self.consensus_labels[1]))
        return ctc_label

    @property
    def anchor_fragment_name(self):
        """

        Returns
        -------
        str, fragment name in which the anchor residue is present.
            If no anchor_index is provided then returns None(default)

        """
        if self.anchor_index is not None:
            return self.fragment_names[self.anchor_index]
        else:
            return None

    @property
    def partner_fragment_name(self):
        """

        Returns
        -------
        str, fragment name in which the partner residue is present

        """
        if self.partner_index is not None:
            return self.fragment_names[self.partner_index]
        else:
            return None

    @property
    def partner_fragment_name_consensus(self):
        """

        Returns
        -------
        consensus label of the partner residue

        """
        if self.partner_index is not None:
            return self.consensus_labels[self.partner_index]
        else:
            return None

    @property
    def partner_fragment_name_best(self):
        """

        Returns
        -------

        """
        if self.partner_index is not None:
            return pick_best_label(self.partner_fragment_name,
                                   self.partner_fragment_name_consensus)
        else:
            return None

    @property
    def anchor_fragment_name_consensus(self):
        """

        Returns
        -------
        consensus label of the anchor residue. If no anchor_index is present then returns None

        """
        if self.anchor_index is not None:
            return self.consensus_labels[self.anchor_index]
        else:
            return None

    @property
    def anchor_fragment_name_best(self):
        """

        Returns
        -------

        """
        if self.anchor_index is not None:
            return pick_best_label(self.anchor_fragment_name,
                                   self.anchor_fragment_name_consensus)
        else:
            return None

    @property
    def anchor_res_and_fragment_str(self):
        """

        Returns
        -------

        """
        if self.anchor_index is not None:
            return '%s@%s' % (self.anchor_residue,
                              self.anchor_fragment_name_best)
        else:
            return None

    @property
    def anchor_res_and_fragment_str_short(self):
        """

        Returns
        -------

        """
        return '%s@%s' % (self.residue_names_short[self.anchor_index],
                          self.anchor_fragment_name_best)

    @property
    def partner_res_and_fragment_str(self):
        """

        Returns
        -------

        """
        return '%s@%s' % (self.partner_residue,
                          self.partner_fragment_name_best)

    @property
    def partner_res_and_fragment_str_short(self):
        """

        Returns
        -------

        """
        return '%s@%s' % (self.residue_names_short[self.partner_index],
                          self.partner_fragment_name_best)

    @property
    def time_arrays(self):
        """

        Returns
        -------

        """
        return self._time_arrays

    @property
    def feat_trajs(self):
        return self.ctc_trajs

    @property
    def label(self):
        return self.ctc_label

    @property
    def ctc_trajs(self):
        """

        Returns
        -------

        """
        return self._ctc_trajs

    @property
    def trajs(self):
        """

        Returns
        -------

        """
        return self._trajs

    @property
    def fragment_colors(self):
        """

        Returns
        -------

        """
        return self._fragment_colors

    @property
    def fragment_names(self):
        """

        Returns
        -------

        """
        return self._fragment_names

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
    def consensus_labels(self):
        """

        Returns
        -------

        """
        return self._consensus_labels

    def binarize_trajs(self, ctc_cutoff_Ang,
                       #switch_off_Ang=None
                       ):
        """

        Parameters
        ----------
        ctc_cutoff_Ang

        Returns
        -------

        """
        transform = lambda itraj: itraj < ctc_cutoff_Ang / 10

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
            result = [transform(itraj) for itraj in self._ctc_trajs]
            self._binarized_trajs[ctc_cutoff_Ang] = result
        #print([ires.shape for ires in result])
        return result

    def frequency_dict(self, ctc_cutoff_Ang,
                       AA_format='short',
                       lb_format='split'):
        """

        Parameters
        ----------
        ctc_cutoff_Ang

        Returns
        -------

        """
        if AA_format== 'short':
            label = self.ctc_label_short
        elif AA_format== 'long':
            label = self.ctc_label
        else:
            raise ValueError(AA_format)

        if lb_format=='split':
            label= '%-15s - %-15s'%tuple(label.split('-'))
        elif lb_format=='join':
            pass
        else:
            raise ValueError(lb_format)

        return {"freq":self.frequency_overall_trajs(ctc_cutoff_Ang),
                "residue idxs":'%u %u'%tuple(self.res_idxs_pair),
                "label":label}

    def frequency_overall_trajs(self, ctc_cutoff_Ang):
        """

        Parameters
        ----------
        ctc_cutoff_Ang

        Returns
        -------

        """
        return _np.mean(_np.hstack(self.binarize_trajs(ctc_cutoff_Ang)))

    def distro_overall_trajs(self, bins=10):
        """

        Parameters
        ----------
        bins

        Returns
        -------
        x, h

        """
        return _np.histogram(_np.hstack(self._ctc_trajs),
                             bins=bins)


    def frequency_per_traj(self, ctc_cutoff_Ang):
        """

        Parameters
        ----------
        ctc_cutoff_Ang

        Returns
        -------

        """
        return [_np.mean(itraj) for itraj in self.binarize_trajs(ctc_cutoff_Ang)]

    def __str__(self):
        out = "Contact object for residue indices"
        out += "\n%s"%self.res_idxs_pair
        out += "\nanchor residue index: %s"%self.anchor_residue_index
        out += "\nFor %u trajectories"%self.n_trajs
        for var in dir(self):
            if not var.startswith("_"):
                try_print=True
                if var.startswith("anchor") and self.anchor_index is None:
                    try_print=False
                if var.startswith("partner") and self.partner_index is None:
                    try_print=False
                if try_print:
                    out += '\n%s: %s'%(var, getattr(self,'%s'%var))
        return out

def contact_map_to_dict(imat, top,
                        res_idxs=None,
                        consensus_labels_map=None,
                        ctc_freq_cutoff=0.01):

    if res_idxs is None:
        res_idxs = _np.arange(top.n_residues)

    if consensus_labels_map is None:
        consensus_labels_map = {key:None for key in res_idxs}
    assert imat.shape[0] == imat.shape[1] == len(res_idxs), (imat.shape, len(res_idxs))
    dict_out = {}
    for ii, jj in _np.array(_np.triu_indices_from(imat, k=1)).T:
        # print(ii,jj)
        val = imat[ii, jj]
        ii, jj = [res_idxs[kk] for kk in [ii, jj]]
        key = '%s@%s-%s@%s' % (top.residue(ii), consensus_labels_map[ii],
                               top.residue(jj), consensus_labels_map[jj])
        key = key.replace("@None","").replace("@none","")
        if val > ctc_freq_cutoff:
            dict_out[key] = val
    return dict_out

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
                idict = self.interfaces[fkey].frequency_per_residue(ctc_cutoff_Ang)

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
        iax, pixelsize = _plot_interface_matrix(mat,
                                                self.interface_labels_consensus,
                                     **kwargs_plot_interface_matrix)
        offset=8*pixelsize
        padding=pixelsize*2

        if annotate:
            n_x = len(self.interface_labels_consensus[1])
            for ii, (pdb, iint) in enumerate(self.interfaces.items()):
                xlabels = []
                for key in self.interface_labels_consensus[0]:
                    if key in iint.cons2resname.keys():
                        xlabels.append(iint.cons2resname[key])
                    elif hasattr(iint,"_orphaned_residues_new_label") and key in iint._orphaned_residues_new_label.values():
                        xlabels.append({val:key for key, val in iint._orphaned_residues_new_label.items()}[key])
                    else:
                        xlabels.append("-")
                ylabels = []
                for key in self.interface_labels_consensus[1]:
                    if key in iint.cons2resname.keys():
                        ylabels.append(iint.cons2resname[key])
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

def _plot_interface_matrix(mat,labels,pixelsize=1,
                           transpose=False, grid=False,
                           cmap="binary",
                           colorbar=False):
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

def _atom_type(aa, no_BB_no_SC='X'):
    if aa.is_backbone:
        return 'BB'
    elif aa.is_sidechain:
        return 'SC'
    else:
        return no_BB_no_SC

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
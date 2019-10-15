#!/home/perezheg/miniconda3/bin/python
import numpy as np
import mdtraj as md
from itertools import product

from sofi_functions.fragments import get_fragments
from sofi_functions.list_utils import in_what_fragment, rangeexpand
from sofi_functions.contacts import xtcs2ctcs
from sofi_functions.parsers import parser_for_interface
from sofi_functions.list_utils import window_average as _wav
from sofi_functions.command_line_tools import _parse_BW_option, _parse_CGN_option
from matplotlib import pyplot as plt

parser = parser_for_interface()
a  = parser.parse_args()

xtcs = sorted(a.trajectories)
print("Will compute contact frequencies for the files:\n  %s\n with a stride of %u frames.\n"%("\n  ".join(xtcs),a.stride))

group_1 = a.frag_idxs_group_1
group_2 = a.frag_idxs_group_2

refgeom = md.load(a.topology)

if isinstance(a.fragments[0],str):
    if a.fragments[0].lower() in ['resseq','ask']:
        fragments = get_fragments(refgeom.top)
        assert len(fragments)>=2, ("You need more than one chain in your topology if you are going to auto-dectect fragments! Aborting.")

"""
else:
    fragments = []
    # TODO did I loose this optarg at some point
    for ifrag in a.fragments_resSeq:
        if ifrag.endswith("-"):
            ifrag+=str(refgeom.top.residue(-1).resSeq)
        ifrag = rangeexpand(ifrag)
        fragments.append(np.unique([rr.index for rr in refgeom.topology.residues if rr.resSeq in ifrag]))
#else:
#    raise NotImplementedError("This feature is not yet implemented")
"""

BW, BWtf = _parse_BW_option(a.BW_file, refgeom.top, fragments,return_tf=True)
fragment_defs = {}
if str(a.BW_file).lower()!='none':
    print("INFO: these are the BW fragments mapped onto your topology")
    fragment_defs.update(BWtf.top2defs(refgeom.top, return_defs=True))
    input("Hit enter to continue!\n")
CGN, CGNtf = _parse_CGN_option(a.CGN_PDB, refgeom.top, fragments, return_tf=True)
if str(a.CGN_PDB).lower()!='none':
    print("INFO: these are the CGN fragments mapped onto your topology")
    fragment_defs.update(CGNtf.top2defs(refgeom.top, return_defs=True))
    input("Hit enter to continue!\n")
groups=[]
group_idxs=[]

from sofi_functions.parsers import match_dict_by_patterns as _match_dict_by_patterns
if a.fragments[0].lower()=="ask":
    for ii in [1,2]:
        answer = input("group%u:\nInput a list of posix-expressions like 'TM*,-TM2,H8' (TM2 will be avoided)\n"%ii).replace(" ","").strip("'").strip('"')
        igroup, igroup_idxs = _match_dict_by_patterns(answer, fragment_defs)
        groups.append(igroup)
        group_idxs.append(igroup_idxs)
        print(groups[-1])
if len(groups)==0:
    if len(fragments)==2:
        print("Only two fragments detected. Overriding inputs for -frag_idxs_group_1/2 with [0] and [1].")
        group_1 = [0]
        group_2 = [1]

    print("\nComputing distances in the interface between fragments %s and %s.\n"
          "The interface is defined by the residues within %3.1f Angstrom of each other in the reference topology.\n"
          "Computing interface..."
          % (group_1, group_2, a.interface_cutoff_Ang),end="")

    ctc_idxs = np.hstack([[list(product(fragments[ii], fragments[jj])) for ii in group_1] for jj in group_2])
else:
    ctc_idxs = np.vstack(list(product(group_idxs[0], group_idxs[1])))

ctcs, ctc_idxs = md.compute_contacts(refgeom, np.vstack(ctc_idxs))
print("done!")

ctc_idxs_receptor_Gprot = np.argwhere(ctcs[0]<a.interface_cutoff_Ang/10).squeeze()
_, ctc_idxs_receptor_Gprot = md.compute_contacts(refgeom, ctc_idxs[ctc_idxs_receptor_Gprot])

print()
print("From %u potential group_1-group_2 distances, the interface was reduced to only %u potential contacts.\nIf this "
      "number is still too high (i.e. the computation is too slow) consider using a smaller interface cutoff"%(len(ctc_idxs), len(ctc_idxs_receptor_Gprot)))


ctcs, times = xtcs2ctcs(xtcs, refgeom.top, ctc_idxs_receptor_Gprot, stride=a.stride, return_time=True,
                        consolidate=False,
                        chunksize=a.chunksize_in_frames)

# Stack all data
actcs = np.vstack(ctcs)
time_array = np.hstack(times)

# Get frequencies
ctcs_bin = (actcs<=a.ctc_cutoff_Ang/10).astype("int").sum(0)
ctc_frequency = ctcs_bin/actcs.shape[0]
ctcs_trajectory_std = np.vstack([np.mean(ictcs < a.ctc_cutoff_Ang/10, 0) for ictcs in ctcs]).std(0)


ctcs_alive = (np.argwhere(ctc_frequency>0)).squeeze()
ctc_freq_per_residues = np.zeros(refgeom.topology.n_residues)
for idx in ctcs_alive:
    ii, jj = ctc_idxs_receptor_Gprot[idx]
    ctc_freq_per_residues[ii] += ctcs_bin[idx]
    ctc_freq_per_residues[jj] += ctcs_bin[idx]
ctc_freq_per_residues /= actcs.shape[0]
"""
plt.figure()
#plt.plot(ctc_freq_per_residues*100)
iax = plt.gca()
iax.set_xlim([0,refgeom.top.n_residues])
xticks = [int(ii) for ii in iax.get_xticks() if 0<= ii < refgeom.top.n_residues]
xticks_fragments = []
cm = plt.get_cmap("Spectral",len(fragments))
for ii, ifrag in enumerate(fragments):
    plt.bar(ifrag, ctc_freq_per_residues[ifrag]*100,width=.5, color='k')
    iax.axvspan(ifrag[0]-.25,ifrag[-1]+.25, alpha=.5,
                color=cm(ii)
                )
    xticks_fragments.append(ifrag[-1])
    #[l.set_facecolor("k") for l in lines]
    #break

# Eliminate some xticks if they're too close to the fragments
xticks = [xtick for xtick in xticks if  all(np.abs(xtick-np.array(xticks_fragments))>20)]
xticks = np.unique(np.hstack((xticks, xticks_fragments)))
iax.set_xticks(xticks)
resSeq_array = [str(rr) for rr in refgeom.top.residues]
iax.set_xticklabels([resSeq_array[ii] for ii in xticks])

good_ones = []
resididxs_alive=np.argwhere(ctc_freq_per_residues!=0).squeeze()
for ii in range(refgeom.top.n_residues):
    d_left   = np.abs(ii-resididxs_alive[resididxs_alive<=ii])
    if len(d_left)>0:
        closest_left = d_left.min()
    else:
        closest_left = np.inf
    d_right  = np.abs(ii-resididxs_alive[resididxs_alive>=ii])
    if len(d_right)>0:
        closest_right = d_right.min()
    else:
        closest_right = np.inf
    limits = [closest_left, closest_right]
    isolated =  all([lim>10 for lim in limits])
    if not isolated:
       print(ii)
       good_ones.append(ii)
plt.figure()
#plt.plot(ctc_freq_per_residues*100)
iax = plt.gca()
#iax.set_xlim([0,refgeom.top.n_residues])
#xticks = [int(ii) for ii in iax.get_xticks() if 0<= ii < refgeom.top.n_residues]
#xticks_fragments = []
#cm = plt.get_cmap("Spectral",len(fragments))
offset = 0
for ii, ifrag in enumerate(fragments):
    #print(ifrag)
    jgood = [ii for ii in ifrag if ii in good_ones]
    xvec =  np.arange(offset, offset+len(jgood))
    plt.bar(xvec, ctc_freq_per_residues[jgood]*100,width=.5, color='k')
    offset+=len(xvec)
    iax.axvspan(xvec[0]-.25,xvec[-1]+.25, alpha=.5,
                color=cm(ii)
                )
    #xticks_fragments.append(ifrag[-1])
    #[l.set_facecolor("k") for l in lines]
    #break
"""


order = np.argsort(ctc_frequency)[::-1]
print("")
istr = "#Freq  resA  resB  fragA-fragB   residxA   residxB   ctc_idx"
if a.CGN_PDB is not None or a.BW_file is not None:
    istr += 'BW/CGN naming'
print()
from sofi_functions.nomenclature_utils import _relabel_consensus
# TODO compute labels just once and be done with it!
# todo: ideas create a contact class?
for oo in order[:a.n_ctcs]:
    pair = ctc_idxs_receptor_Gprot[oo]
    idx1, idx2 = pair
    s1, s2 = [in_what_fragment(idx, fragments) for idx in pair]
    imean = ctc_frequency[oo]
    print("%3.2f %s-%s  %3u-%-3u      %3u %3u %3u"%(imean, refgeom.top.residue(idx1), refgeom.top.residue(idx2), s1, s2, idx1, idx2,oo))

panelheight=5
panelheight2fontsize=a.n_ctcs*panelheight*6
myfig, myax = plt.subplots(a.n_ctcs, 1, sharex=True, sharey=True,
                           figsize=(5*panelheight, a.n_ctcs*panelheight ))
from matplotlib import rcParams
#rcParams["font.size"]=max(myfig.get_size_inches())/10  #fudging a bit
for ii, oo in enumerate(order[:a.n_ctcs]):
    pair = ctc_idxs_receptor_Gprot[oo]
    idx1, idx2 = pair
    r1,r2 = [refgeom.top.residue(ii) for ii in pair]
    s1, s2 = [in_what_fragment(idx, fragments) for idx in pair]
    lc1, lc2 = [_relabel_consensus(ii, [CGN, BW], no_key=None) for ii in pair]
    frag1, frag2 = [in_what_fragment(idx, fragments) for idx in pair]
    if lc1 is None:
        lc1 = frag1
    if lc2 is None:
        lc2 = frag2

    plt.sca(myax[ii])
    iax = myax[ii]
    icol = iter(plt.rcParams['axes.prop_cycle'].by_key()["color"])
    for ictc, itime, ixtc in zip(ctcs, times, xtcs):
        imean = np.mean(ictc < a.ctc_cutoff_Ang/10, 0)
        ilabel = '%s (%u %s)' % (ixtc, (ictc[:, oo] < a.ctc_cutoff_Ang/10).mean() * 100, '%')
        alpha = 1
        icolor = next(icol)
        if a.n_smooth_hw > 0:
            alpha = .2
            itime_smooth, _ = _wav(itime, half_window_size=a.n_smooth_hw)
            ictc_smooth, _ = _wav(ictc[:, oo], half_window_size=a.n_smooth_hw)
            plt.plot(itime_smooth / 1e3,
                     ictc_smooth*10,
                     label=ilabel,
                     color=icolor)
            ilabel = None

        plt.plot(itime / 1e3, ictc[:, oo]*10,
                 alpha=alpha,
                 label= ilabel,
                 color=icolor,
                 )

        plt.legend()
    xt, yt = np.mean(itime / 1e3), 1
    ilabel = '%s@%s-%s@%s (%u%% $\pm$ %u%% in all trajs)'%(r1, lc1, r2, lc2,
                                                           ctc_frequency[oo] * 100, ctcs_trajectory_std[oo]*100)
    plt.text(xt,yt,ilabel, ha='center')

    plt.ylabel('d / Ang')
    plt.ylim([0, 10])

    iax.axhline(a.ctc_cutoff_Ang, color='r')

iax.set_xlabel('t / ns')
myfig.tight_layout(h_pad=2, w_pad=0, pad=0)
myfig.tight_layout(h_pad=0,w_pad=0,pad=0)
fname = '%s_vs_time.%s'%(a.output_desc.strip("."),a.graphic_ext.strip("."))
myfig.savefig(fname,bbox_inches="tight")
print("Saved %s!"%fname)
plt.close()

plt.figure()
# np.savetxt()

# TODO make a method out of this, preferably in sofi_functions.contacts
max_t = np.max([len(itime) for itime in times])
n_ctcs_out = np.zeros((max_t, len(xtcs)))
n_ctcs_out[:,:] = np.nan
icol = iter(plt.rcParams['axes.prop_cycle'].by_key()["color"])
for ii, (itime, ixtc) in enumerate(zip(times, xtcs)):
    n_ctcs = (ctcs[ii] < a.ctc_cutoff_Ang / 10).sum(1)
    alpha = 1
    icolor = next(icol)
    ilabel=ixtc
    if a.n_smooth_hw > 0:
        itime_smooth, _ = _wav(itime, half_window_size=a.n_smooth_hw)
        n_ctcs_smooth, _ = _wav(n_ctcs, half_window_size=a.n_smooth_hw)
        plt.plot(itime_smooth / 1e3,
                 n_ctcs_smooth,
                 label=ilabel,
                 color=icolor)
        ilabel = None
        alpha = .2
    plt.plot(itime/1e3, n_ctcs, label=ilabel, alpha=alpha, color=icolor)
    n_ctcs_out[:len(n_ctcs), ii] = n_ctcs
plt.ylabel("# interface contacts")
plt.xlabel("t / ns")
plt.legend()
fname = "ncontacts_%s_vs_time.%s"%(a.output_desc.strip("."),a.graphic_ext.strip("."))
plt.savefig(fname)
print("Saved %s!"%fname)

if str(a.output_ascii).lower()!="none":
    fname = "ncontacts_%s_vs_time.%s" % (a.output_desc.strip("."), a.output_ascii.strip("."))
    np.savetxt(fname, n_ctcs_out,
               fmt='%4s',
               header=' '.join(xtcs)
               )
    print("Saved %s!" % fname)

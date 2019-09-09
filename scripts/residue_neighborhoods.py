#!/home/perezheg/miniconda3/bin/python
import argparse
import numpy as np
import mdtraj as md
from matplotlib import pyplot as plt
from json import load as jsonload

from sofi_functions.actor_utils import get_fragments, \
ctc_freq_reporter_by_residue_neighborhood, table2BW_by_AAcode,\
    CGN_transformer,top2CGN_by_AAcode,_relabel_consensus,guess_missing_BWs,\
    xtcs2ctcs, rangeexpand, \
    interactive_fragment_picker_by_resSeq, \
    in_what_fragment, mycolors, bonded_neighborlist_from_top, dangerously_auto_fragments, _print_frag, \
    unique_list_of_iterables_by_tuple_hashing \

parser = argparse.ArgumentParser(description='Small residue-residue contact analysis tool, initially developed for the '
                                             'receptor-G-protein complex. The provides the residue indices')

parser.add_argument('topology',    type=str,help='Topology file')
parser.add_argument('trajectories',type=str,help='trajectory file(s)', nargs='+')
parser.add_argument('--resSeq_idxs',type=str,help='the resSeq idxs of interest (in VMD these are called "resid"). Can be in a format 1,2-6,10,20-25')
parser.add_argument("--ctc_cutoff_Ang",type=float, help="The cutoff distance between two residues for them to be considered in contact. Default is 3 Angstrom.", default=3)
parser.add_argument("--stride", type=int, help="Stride down the input trajectoy files by this factor. Default is 1.", default=1)
parser.add_argument("--n_ctcs", type=int, help="Only the first n_ctcs most-frequent contacts will be written to the ouput. Default is 5.", default=5)
parser.add_argument("--n_nearest", type=int, help="Ignore this many nearest neighbors when computing neighbor lists. 'Near' means 'connected by this many bonds'. Default is 4.", default=4)
parser.add_argument("--chunksize_in_frames", type=int, help="Trajectories are read in chunks of this size (helps with big files and memory problems). Default is 10000", default=10000)
parser.add_argument("--nlist_cutoff_Ang", type=float, help="Cutoff for the initial neighborlist. Only atoms that are within this distance in the original reference "
                                                          "(the topology file) are considered potential neighbors of the residues in resSeq_idxs, s.t. "
                                                          "non-necessary distances (e.g. between N-terminus and G-protein) are not even computed. "
                                                          "Default is 15 Angstrom.", default=15)


parser.add_argument('--fragments',    dest='fragmentify', action='store_true', help="Auto-detect fragments (i.e. breaks) in the peptide-chain. Default is true.")
parser.add_argument('--no-fragments', dest='fragmentify', action='store_false')
parser.set_defaults(fragmentify=True)

parser.add_argument('--sort',    dest='sort', action='store_true', help="Sort the resSeq_idxs list. Defaut is True")
parser.add_argument('--no-sort', dest='sort', action='store_false')
parser.set_defaults(sort=True)

parser.add_argument('--pbc',    dest='pbc', action='store_true', help="Consider periodic boundary conditions when computing distances."
                                                                      " Defaut is True")
parser.add_argument('--no-pbc', dest='pbc', action='store_false')
parser.set_defaults(pbc=True)

parser.add_argument('--ask_fragment',    dest='ask', action='store_true', help="Interactively ask for fragment assignemnt when input matches more than one resSeq")
parser.add_argument('--no-ask_fragment', dest='ask', action='store_false')
parser.set_defaults(ask=True)
parser.add_argument('--output_npy', type=str, help="Name of the output.npy file for storing this runs' results", default='output.npy')
parser.add_argument('--output_ext', type=str, help="Extension of the output graphics, default is .pdf", default='.pdf')


parser.add_argument('--fragment_names', type=str,
                    help="Name of the fragments. Leave empty if you want them automatically named."
                         " Otherwise, give a quoted list of strings separated by commas, e.g. "
                         "'TM1, TM2, TM3,'",
                    default="")
parser.add_argument("--BW_file", type=str, help="Json file with info about the Ballesteros-Weinstein definitions as downloaded from the GPRCmd", default='None')
parser.add_argument("--CGN_PDB", type=str, help="PDB code for a consensus G-protein nomenclature", default='None')

a  = parser.parse_args()
a.ext = a.output_ext

xtcs = sorted(a.trajectories)
print("Will compute contact frequencies for the files:\n  %s\n with a stride of %u frames.\n"%("\n  ".join(xtcs),a.stride))

fragment_names = a.fragment_names

refgeom = md.load(a.topology)
if a.fragmentify:
    fragments = get_fragments(refgeom.top)
else:
    raise NotImplementedError("This feature is not yet implemented")

if fragment_names == '':
    fragment_names=['frag%u'%ii for ii in range(len(fragments))]
else:
    assert isinstance(fragment_names, str), "Argument --fragment_names invalid: %s"%fragment_names
    if 'danger' not in fragment_names.lower():
        fragment_names=[ff.strip(" ") for ff in fragment_names.split(",")]
        assert len(fragment_names)==len(fragments), "Mismatch between nr. fragments and fragment names %s vs %s (%s)"%(len(fragments), len(fragment_names), fragment_names)
    else:
        fragments, fragment_names = dangerously_auto_fragments(refgeom.top,
                                                               method="bonds",
                                                               verbose=False,
                                                               force_resSeq_breaks=True,
                                                               frag_breaker_to_pick_idx=0,
                                                               )
        fragment_names.extend(refgeom.top.residue(ifrag[0]).name for ifrag in fragments[len(fragment_names):])



        for ifrag_idx, (ifrag, frag_name) in enumerate(zip(fragments, fragment_names)):
            _print_frag(ifrag_idx, refgeom.top, ifrag, end='')
            print(" ", frag_name)
#todo this is code repetition from sites.py
if a.BW_file=='None':
    BW = [None for __ in range(refgeom.top.n_residues)]
else:
    with open(a.BW_file,"r") as f:
        idict = jsonload(f)
    BW = table2BW_by_AAcode(idict["file"])
    try:
        assert idict["guess_BW"]
    except:
        raise NotImplementedError("The BW json file has to contain the guess_BW=True")
    answer = input("Which fragments are succeptible of a BW-numbering?(Can be in a format 1,2-6,10,20-25)\n")
    restrict_to_residxs = np.hstack([fragments[ii] for ii in rangeexpand(answer)])
    BW = guess_missing_BWs(BW,refgeom.top, restrict_to_residxs=restrict_to_residxs)

if a.CGN_PDB=='None':
    CGN = [None for __ in range(refgeom.top.n_residues)]
else:
    CGN_tf = CGN_transformer(a.CGN_PDB)
    answer = input("Which fragments are succeptible of a CGN-numbering?(Can be in a format 1,2-6,10,20-25)\n")
    restrict_to_residxs = np.hstack([fragments[ii] for ii in rangeexpand(answer)])
    CGN    = top2CGN_by_AAcode(refgeom.top, CGN_tf, restrict_to_residxs=restrict_to_residxs)
mycolors.extend(mycolors)
mycolors.extend(mycolors)
#mycolors = {key:mycolors[ii] for ii, key in enumerate(fragment_names)}

resSeq_idxs = rangeexpand(a.resSeq_idxs)
if a.order:
    resSeq_idxs = sorted(resSeq_idxs)

print("\nWill compute neighborhoods for the residues with resid")
print("%s"%resSeq_idxs)
print("excluding %u nearest neighbors\n"%a.n_nearest)

resSeq2residxs, _ = interactive_fragment_picker_by_resSeq(resSeq_idxs, fragments,refgeom.top,
                                                         pick_first_fragment_by_default=not a.ask,
                                                         )

print('%10s  %6s  %7s  %10s'%tuple(("residue  residx    fragment  input_resSeq".split())))
for key, val in resSeq2residxs.items():
    print('%10s  %6u  %7u  %10u'%(refgeom.top.residue(val), val, in_what_fragment(val, fragments), key))

# Create a neighborlist
nl = bonded_neighborlist_from_top(refgeom.top, n=a.n_nearest)

# Use it to prune the contact indices
ctc_idxs = np.vstack([[np.sort([val,ii]) for ii in range(refgeom.top.n_residues) if ii not in nl[val] and ii!=val   ] for val in resSeq2residxs.values()])


print("\nPre-computing likely neighborhoods by reducing the neighbor-list to %u Angstrom in the reference geom %s..."%(a.nlist_cutoff_Ang,a.topology),end="",flush=True)
ctcs, ctc_idxs = md.compute_contacts(refgeom, np.vstack(ctc_idxs), periodic=a.pbc)
print("done!")

ctc_idxs_small = np.argwhere(ctcs[0]<a.nlist_cutoff_Ang/10).squeeze()
_, ctc_idxs_small = md.compute_contacts(refgeom, ctc_idxs[ctc_idxs_small])
ctc_idxs_small = unique_list_of_iterables_by_tuple_hashing(ctc_idxs_small)

print("From %u potential distances, the neighborhoods have been reduced to only %u potential contacts.\nIf this "
      "number is still too high (i.e. the computation is too slow), consider using a smaller nlist_cutoff_Ang "%(len(ctc_idxs), len(ctc_idxs_small)))

ctcs_trajs, time_array = xtcs2ctcs(xtcs, refgeom.top, ctc_idxs_small, stride=a.stride,
                              chunksize=a.chunksize_in_frames, return_time=True,
                              consolidate=False
                              )
actcs = np.vstack(ctcs_trajs)
ctcs_mean = np.mean(actcs < a.ctc_cutoff_Ang / 10, 0)

final_look = ctc_freq_reporter_by_residue_neighborhood(ctcs_mean, resSeq2residxs,
                                                       fragments, ctc_idxs_small,
                                                       refgeom.top,
                                                       silent=True,
                                                       n_ctcs=a.n_ctcs)

#print("Will take a look at:")
split_by_neighborhood = True
if not split_by_neighborhood:
    myfig, myax = plt.subplots(len(final_look), 1, sharex=True, sharey=True)

    for ii, oo in enumerate(final_look):
        pair = ctc_idxs_small[oo]
        print([refgeom.top.residue(jj) for jj in pair])
        plt.sca(myax[ii])
        plt.plot(time_array/1e3, actcs[:, oo],
                 label='%s-%s (%u)' % (refgeom.top.residue(pair[0]), refgeom.top.residue(pair[1]), ctcs_mean[oo] * 100))
        plt.legend()
        # plt.yscale('log')
        plt.ylabel('A / $\\AA$')
        plt.ylim([0, 1])
        iax = plt.gca()
        iax.axhline(a.ctc_cutoff_Ang / 10, color='r')
    iax.set_xlabel('t / ns')
    plt.show()
else:
    print("The following files have been created")
    panelheight=3
    xvec = np.arange(np.max([len(val) for val in final_look.values()]))
    n_cols=np.min((4, len(resSeq2residxs)))
    n_rows = np.ceil(len(resSeq2residxs)/n_cols).astype(int)
    panelsize=4
    panelsize2font=3.5
    histofig, histoax = plt.subplots(n_rows,n_cols, sharex=True, sharey=True, figsize=(n_cols*panelsize*2, n_rows*panelsize), squeeze=False)
    list_of_axes = list(histoax.flatten())
    for jax, residx in zip(list_of_axes, resSeq2residxs.values()):# in np.unique([ctc_idxs_small[ii] for ii in final_look]):
        toplot=final_look[residx]
        anchor_cons_label = _relabel_consensus(residx, [BW,CGN])
        anchor_frag_idx = in_what_fragment(residx, fragments)
        anchor_frag_label = fragment_names[anchor_frag_idx]
        if anchor_cons_label not in [None, "None", "NA"]:
            anchor_frag_label = anchor_cons_label
        res_and_fragment_str = '%s@%s' % (refgeom.top.residue(residx),anchor_frag_label)

        jax.set_title(
            "Contact frequency @%2.1f $\AA$\n%u nearest bonded neighbors excluded" % (a.ctc_cutoff_Ang, a.n_nearest))
        jax.set_ylim([0, 1])
        jax.set_xlim([-.5, len(toplot)+1 - .5])
        jax.set_xticks([])
        jax.set_yticks([.25, .50, .75, 1])
        # jax.set_yticklabels([])
        [jax.axhline(ii, color="k", linestyle="--", zorder=-1) for ii in [.25, .50, .75]]

        jax.plot(-1, -1, 'o', color=mycolors[anchor_frag_idx], label=res_and_fragment_str)
        jax.legend(fontsize=panelsize * panelsize2font)
        #toplot=[kk for kk in final_look if residx in ctc_idxs_small[kk]]

        if len(toplot)>0:
            myfig, myax = plt.subplots(len(toplot), 1, sharex=True, sharey=True,
                                       figsize=(10,len(toplot)*panelheight))
            myax=np.array(myax,ndmin=1)
            myax[0].set_title(res_and_fragment_str)
            partner_labels = []
            partner_colors = []
            for ii, oo in enumerate(toplot):
                idx1,idx2 = ctc_idxs_small[oo]
                s1 = in_what_fragment(idx1, fragments)
                s2 = in_what_fragment(idx2, fragments)
                labels_consensus = [_relabel_consensus(jj, [BW,CGN]) for jj in [idx1,idx2]]
                labels_frags = [fragment_names[ss] for ss in [s1,s2]]

                for jj, jcons in enumerate(labels_consensus):
                    if str(jcons).lower() != "na":
                        labels_frags[jj] = jcons

                #print([refgeom.top.residue(jj) for jj in pair])
                plt.sca(myax[ii])
                ctc_label = '%s@%s-%s@%s' % (
                             refgeom.top.residue(idx1),labels_frags[0],
                             refgeom.top.residue(idx2),labels_frags[1])
                for jj, jxtc in enumerate(xtcs):
                    plt.plot(time_array[jj] / 1e3, ctcs_trajs[jj][:, oo]*10,
                             label='%s (%u%%)'%(jxtc, np.mean(ctcs_trajs[jj][:, oo] < a.ctc_cutoff_Ang / 10) * 100))
                plt.legend(loc=1)
                # plt.yscale('log')
                plt.ylabel('A / $\\AA$')
                plt.ylim([0, 10])
                iax = plt.gca()
                iax.text(np.mean(iax.get_xlim()),1, ctc_label, ha='center')
                iax.axhline(a.ctc_cutoff_Ang, color='r')
                if residx == idx1:
                    partner, partnerseg, partnercons = idx2, s2, labels_consensus[1]
                else:
                    partner, partnerseg, partnercons = idx1, s1, labels_consensus[0]
                if partnercons not in [None, "NA"]:
                    ipartnerlab = partnercons
                else:
                    ipartnerlab = fragment_names[partnerseg]
                partner_labels.append('%s@%s' % (refgeom.top.residue(partner), ipartnerlab))
                partner_colors.append(mycolors[partnerseg])
            # TODO re-use code from mdas.visualize represent vicinities
            patches = jax.bar(xvec[:len(toplot)], ctcs_mean[toplot],
                    #label=res_and_fragment_str,
                             width=.25)


            for ix, iy, ilab, ipatch, icol in zip(xvec, ctcs_mean[toplot], partner_labels,
                                                  patches.get_children(), partner_colors):
                iy += .01
                if iy>.65:
                    iy=.65
                jax.text(ix, iy, ilab,
                         va='bottom',
                         ha='left',
                         rotation=45,
                         fontsize=panelsize * panelsize2font,
                         backgroundcolor="white"
                         )

                ipatch.set_color(icol)
                """
                isegs.append(iseg[0])
                    
                
                isegs = _np.unique(isegs)
                _empty_legend(iax,
                              [binary_ctcs2flare_kwargs["fragment_names"][ii] for ii in isegs],
                              [_mycolors[ii] for ii in isegs],
                              'o' * len(isegs),
                              loc='upper right',
                              fontsize=panelsize * panelsize2font,
                              )
                """
            #plt.show()
            #plt.close()


            iax.set_xlabel('t / ns')
            myfig.tight_layout(h_pad=0,w_pad=0,pad=0)

            axtop, axbottom = myax[0], myax[-1]
            iax2 = axtop.twiny()
            iax2.set_xticks(axbottom.get_xticks())
            iax2.set_xticklabels(axbottom.get_xticklabels())
            iax2.set_xlim(axbottom.get_xlim())
            iax2.set_xlabel(axbottom.get_xlabel())

            fname = 'neighborhood.%s.time_resolved.%s'%(res_and_fragment_str.replace('*',""), a.ext.strip("."))
            plt.savefig(fname,bbox_inches="tight")
            plt.close(myfig)
            print(fname)
            #plt.show()


    histofig.tight_layout(h_pad=2, w_pad=0, pad=0)



    fname = "neighborhoods_overall.%s"%a.ext.strip(".")
    histofig.savefig(fname)
    print(fname)

fname = a.output_npy
if not fname.endswith(".npy"):
    fname+=".npy"
np.save(fname,{"args":a,
                      "ctc_idxs":ctc_idxs_small, 
                      'ctcs':actcs, 
                      'time_array':time_array})
print(fname)
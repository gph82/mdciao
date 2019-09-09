#!/home/perezheg/miniconda3/bin/python
import argparse
import numpy as np
import mdtraj as md
from matplotlib import pyplot as plt
from os.path import splitext, split as psplit
from json import load as jsonload

# TODO check what's going on here
from actor_utils import get_fragments,\
    xtcs2ctcs, rangeexpand, \
    interactive_fragment_picker_by_AAresSeq, dangerously_auto_fragments, \
    in_what_fragment, _print_frag, _replace4latex, table2BW_by_AAcode, \
    guess_missing_BWs, _relabel_consensus, CGN_transformer, top2CGN_by_AAcode


parser = argparse.ArgumentParser(description='Small residue-residue contact analysis tool, initially developed for the '
                                             'receptor-G-protein complex. The user has to provide "site" files in .json format')

parser.add_argument('topology',    type=str,help='Topology file')
parser.add_argument('trajectories',type=str,help='trajectory file(s)', nargs='+')
parser.add_argument('--site_files',type=str,nargs='+', help='site file(s) in json format containing site information, i.e., which bonds correspond to each site')
parser.add_argument("--stride", type=int, help="Stride down the input trajectoy files by this factor. Default is 1.", default=1)
parser.add_argument("--ctc_cutoff_Ang",type=float, help="The cutoff distance between two residues for them to be considered in contact. Default is 3 Angstrom.", default=3)
parser.add_argument("--chunksize_in_frames", type=int, help="Trajectories are read in chunks of this size (helps with big files and memory problems). Default is 10000", default=10000)
parser.add_argument("--BW_file", type=str, help="Json file with info about the Ballesteros-Weinstein definitions as downloaded from the GPRCmd", default='None')
parser.add_argument("--CGN_PDB", type=str, help="PDB code for a consensus G-protein nomenclature", default='None')

parser.add_argument('--fragments',    dest='fragmentify', action='store_true', help="Auto-detect fragments (i.e. breaks) in the peptide-chain. Default is true.")
parser.add_argument('--no-fragments', dest='fragmentify', action='store_false')
parser.set_defaults(fragmentify=True)

parser.add_argument('--pbc',    dest='pbc', action='store_true', help="Consider periodic boundary conditions when computing distances."
                                                                      " Defaut is True")
parser.add_argument('--no-pbc', dest='pbc', action='store_false')
parser.set_defaults(pbc=True)

parser.add_argument('--default_fragment_index', default=None, type=int, help="In case a residue identified as, e.g, GLU30, appears more than\n"
                                                               " one time in the topology, e.g. in case of a dimer, the user can\n"
                                                               " pass which fragment/monomer should be chosen by default. The\n"
                                                               " default behaviour (None) will prompt the user when necessary")

parser.add_argument('--output_npy', type=str, help="Name of the output.npy file for storing this runs' results",
                    default='output_sites')

parser.add_argument('--fragment_names', type=str,
                    help="Name of the fragments. Leave empty if you want them automatically named."
                                                                    " Otherwise, give a quoted list of strings separated by commas, e.g. "
                         "'TM1, TM2, TM3,'",
                    default="")
a  = parser.parse_args()

# Prepare naming
desc_out = a.output_npy
desc_out = desc_out.rstrip(".")

# Inform about trajectories
xtcs = sorted(a.trajectories)
print(a.site_files)
print("Will compute the sites\n %s\nin the trajectories:\n  %s\n with a stride of %u frames.\n"%("\n ".join(a.site_files),
                                                                                                  "\n  ".join(xtcs),a.stride))
# Inform about fragments
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
                                                               pick_first_fragment_by_default=True,
                                                               )
        fragment_names.extend(refgeom.top.residue(ifrag[0]).name for ifrag in fragments[len(fragment_names):])



for ifrag_idx, (ifrag, frag_name) in enumerate(zip(fragments, fragment_names)):
    _print_frag(ifrag_idx, refgeom.top, ifrag, end='')
    print(" ", frag_name)

# TODO write functions for these two
# Do we want BW definitions
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

# Dow we want CGN definitions:
if a.CGN_PDB=='None':
    CGN = [None for __ in range(refgeom.top.n_residues)]
else:
    CGN_tf = CGN_transformer(a.CGN_PDB)
    answer = input("Which fragments are succeptible of a CGN-numbering?(Can be in a format 1,2-6,10,20-25)\n")
    restrict_to_residxs = np.hstack([fragments[ii] for ii in rangeexpand(answer)])
    CGN    = top2CGN_by_AAcode(refgeom.top, CGN_tf, restrict_to_residxs=restrict_to_residxs)    

def sitefile2site(sitefile):
    with open(sitefile,"r") as f:
        idict = jsonload(f)
    try:
        idict["bonds"]["AAresSeq"]=[item.split("-") for item in idict["bonds"]["AAresSeq"] if item[0]!='#']
        idict["n_bonds"]=len(idict["bonds"]["AAresSeq"])
    except:
        print("Malformed .json file for the site %s"%sitefile2site())
    if "sitename" not in idict.keys():
        idict["name"]=splitext(psplit(sitefile)[-1])[0]
    else:
        idict["name"]=psplit(idict["sitename"])[-1]
    return idict

sites = [sitefile2site(ff) for ff in a.site_files]

AAresSeqs = [ss["bonds"]["AAresSeq"] for ss in sites]
AAresSeqs = [item for sublist in AAresSeqs for item in sublist]
AAresSeqs = [item for sublist in AAresSeqs for item in sublist]

resSeq2residxs, _ = interactive_fragment_picker_by_AAresSeq(AAresSeqs, fragments,refgeom.top,
                                                            default_fragment_idx=a.default_fragment_index,
                                                         )
if None in resSeq2residxs.values():
    raise ValueError("These residues of your input have not been found. Please revise it:\n%s"%('\n'.join([key for key, val in resSeq2residxs.items() if val is None])))
print('%10s  %6s  %7s  %10s'%tuple(("residue  residx    fragment  input_resSeq".split())))
for key, val in resSeq2residxs.items():
    print('%10s  %6u  %7u  %s'%(refgeom.top.residue(val), val, in_what_fragment(val, fragments), key))

ctc_idxs_small = np.array([resSeq2residxs[key] for key in AAresSeqs]).reshape(-1,2)
ctcs, time_array = xtcs2ctcs(xtcs, refgeom.top, ctc_idxs_small, stride=a.stride,
                             chunksize=a.chunksize_in_frames,
                             return_time=True, consolidate=False)

# Slap the contact values, res_pairs and idxs onto the site objects, so that they're more standaloune
ctc_pairs_iterators = iter(ctc_idxs_small)
ctc_value_idx = iter(np.arange(len(ctc_idxs_small))) # there has to be a better way
for isite in sites:
    isite["res_idxs"]=[]
    isite["ctc_idxs"]=[]
    for __ in range(isite["n_bonds"]):
        isite["res_idxs"].append(next(ctc_pairs_iterators))
        isite["ctc_idxs"].append(next(ctc_value_idx))
    isite["res_idxs"] = np.vstack(isite["res_idxs"])
    isite["ctc_idxs"] = np.array(isite["ctc_idxs"])
    #print(isite["res_idxs"])
    #print(isite["ctc_idxs"])
    isite["ctcs"]=[]
    for ctc_traj in ctcs:
        isite["ctcs"].append(ctc_traj[:,isite["ctc_idxs"]])
    #print()

print("The following files have been created")
panelheight=3
xvec = np.arange(np.max([ss["n_bonds"] for ss in sites]))
n_cols=np.min((4, len(sites)))
n_rows = np.ceil(len(sites)/n_cols).astype(int)
panelsize=4
panelsize2font=3.5
histofig, histoax = plt.subplots(n_rows,n_cols, sharex=True, sharey=True, figsize=(n_cols*panelsize*2, n_rows*panelsize), squeeze=False)
list_of_axes = list(histoax.flatten())
for jax, isite in zip(list_of_axes, sites):# in np.unique([ctc_idxs_small[ii] for ii in final_look]):
    j_av_ctcs = np.vstack(isite["ctcs"])
    ctcs_mean = np.mean(j_av_ctcs < a.ctc_cutoff_Ang / 10, 0)
    patches = jax.bar(np.arange(isite["n_bonds"]), ctcs_mean,
                      # label=res_and_fragment_str,
                      width=.25)
    jax.set_title(
        "Contact frequency @%2.1f $\AA$\n of site '%s'" % (a.ctc_cutoff_Ang, isite["name"]))
    jax.set_ylim([0, 1])
    jax.set_xlim([-.5, xvec[-1]+1 - .5])
    jax.set_xticks([])
    jax.set_yticks([.25, .50, .75, 1])
    # jax.set_yticklabels([])
    [jax.axhline(ii, color="k", linestyle="--", zorder=-1) for ii in [.25, .50, .75]]
    #jax.legend(fontsize=panelsize * panelsize2font)


    # Also, prepare he timedep plot, since it'll also iterate throuth the pairs
    myfig, myax = plt.subplots(isite["n_bonds"], 1, sharex=True, sharey=True,
                               figsize=(10, isite["n_bonds"] * panelheight))

    myax=np.array(myax,ndmin=1)
    myax[0].set_title("site: %s" % (isite["name"]))

    # Loop over the pairs to attach labels to the bars
    for ii, (iy, ipair) in enumerate(zip(ctcs_mean, isite["res_idxs"])):
        labels_consensus = [_relabel_consensus(jj, [BW, CGN]) for jj in ipair]
        #TODO this .item() is to comply to
        labels_frags = [in_what_fragment(idx.item(), fragments, fragment_names=fragment_names) for idx in ipair]
        for jj, jcons in enumerate(labels_consensus):
            if str(jcons).lower() != "na":
                labels_frags[jj] = jcons

        ilab = '%s@%s-\n%s@%s' % (
            refgeom.top.residue(ipair[0]), labels_frags[0],
            refgeom.top.residue(ipair[1]), labels_frags[1]
        )

        ilab = _replace4latex(ilab)
        ix = ii
        iy += .01
        if iy>.65:
            iy=.65
        jax.text(ix-.05, iy, ilab,
                 va='bottom',
                 ha='left',
                 rotation=45,
                 fontsize=panelsize * panelsize2font,
                 backgroundcolor="white"
                 )

        # Now the trajectory loops
        # Loop over trajectories
        for itime, ixtc, ictc in zip(time_array, xtcs, isite["ctcs"]):
            plt.sca(myax[ii])
            plt.plot(itime / 1e3,
                     ictc[:, ii]*10,
                     label=ixtc
            #         ctcs_mean[oo] * 100
            )
            plt.legend(loc=1)

            plt.ylabel('d / Ang')
            plt.ylim([0, 10])
            iax = plt.gca()
            iax.axhline(a.ctc_cutoff_Ang, color='r')
        myax[ii].text(np.mean(iax.get_xlim()), .5, ilab.replace("\n", ""), ha='left')


    iax.set_xlabel('t / ns')
    myfig.tight_layout(h_pad=0,w_pad=0,pad=0)

    axtop, axbottom = myax[0], myax[-1]
    iax2 = axtop.twiny()
    iax2.set_xticks(axbottom.get_xticks())
    iax2.set_xticklabels(axbottom.get_xticklabels())
    iax2.set_xlim(axbottom.get_xlim())
    iax2.set_xlabel(axbottom.get_xlabel())

    fname = 'site.%s.%s.time_resolved.pdf'%(isite["name"].replace(" ","_"),desc_out.strip("."))
    plt.savefig(fname,bbox_inches="tight")
    plt.close(myfig)
    print(fname)
        #plt.show()

histofig.tight_layout(h_pad=2, w_pad=0, pad=0)
fname = "sites_overall.%s.pdf"%desc_out.rstrip(".")
histofig.savefig(fname)
print(fname)


#!/home/perezheg/miniconda3/bin/python
from sofi_functions.parsers import parser_for_interface
parser = parser_for_interface()
a  = parser.parse_args()
#from sofi_functions.command_line_tools import _inform_of_parser
#_inform_of_parser(parser)

from sofi_functions.command_line_tools import interface
# Make a dictionary out ot of it and pop the positional keywords
b = {key:getattr(a,key) for key in dir(a) if not key.startswith("_")}
for key in ["topology","trajectories"]:
    b.pop(key)
neighborhood = interface(a.topology, a.trajectories, **b)

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

from matplotlib import rcParams
#rcParams["font.size"]=max(myfig.get_size_inches())/10  #fudging a bit

    #ilabel = '%s@%s-%s@%s (%u%% $\pm$ %u%% in all trajs)'%(r1, lc1, r2, lc2,
    #                                                       ctc_frequency[oo] * 100, ctcs_trajectory_std[oo]*100)


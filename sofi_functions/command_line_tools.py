import numpy as np
import mdtraj as md
from matplotlib import pyplot as plt,rcParams as _rcParams


from json import load as jsonload
from os.path import splitext, split as psplit
from textwrap import wrap as _twrap
from itertools import product
from os import path, mkdir
from tempfile import TemporaryDirectory as _TD


from sofi_functions.fragments import \
    interactive_fragment_picker_by_AAresSeq as _interactive_fragment_picker_by_AAresSeq, \
    get_fragments, _print_frag

from sofi_functions.nomenclature_utils import \
    CGN_transformer, BW_transformer,\
    _relabel_consensus, _guess_nomenclature_fragments

from sofi_functions.contacts import \
    ctc_freq_reporter_by_residue_neighborhood, \
    xtcs2ctcs,contact_group, contact_pair

from sofi_functions.list_utils import \
    rangeexpand, \
    unique_list_of_iterables_by_tuple_hashing, \
    in_what_fragment

from sofi_functions.bond_utils import \
    bonded_neighborlist_from_top

from sofi_functions.actor_utils import \
    mycolors, dangerously_auto_fragments, \
    interactive_fragment_picker_by_resSeq

from sofi_functions.parsers import \
    match_dict_by_patterns as _match_dict_by_patterns

def _inform_of_parser(parser):
    # TODO find out where the keys are hiding in parser...
    a = parser.parse_args()
    print("Here I am")
    for key, __ in a._get_kwargs():
        dval = parser.get_default(key)
        fmt = '%s=%s,'
        if isinstance(dval, str):
            fmt = '%s="%s",'
        print(fmt % (key, dval))

def _offer_to_create_dir(output_dir):
    if not path.isdir(output_dir):
        answer = input("\nThe directory '%s' does not exist. Create it on the fly [y/n]?\nDefault [y]: " % output_dir)
        if len(answer) == 0 or answer.lower().startswith("y"):
            mkdir(output_dir)
        else:
            print("Stopping. Please check your variable 'output_dir' and try again")
            return


def _parse_consensus_option(identifier, type,
                            top, fragments,
                            return_tf=False,
                            **tf_kwargs):

    if str(identifier).lower() == 'none':
        map_out = [None for __ in range(top.n_residues)]
        tf_out = None
    else:
        tf_out = {"BW": BW_transformer,
                  "CGN":CGN_transformer}[type](identifier,**tf_kwargs)
        answer = _parse_fragment_answer(tf_out, top, fragments, type)
        restrict_to_residxs = np.hstack([fragments[ii] for ii in rangeexpand(answer)])
        map_out = tf_out.top2map(top,
                                 restrict_to_residxs=restrict_to_residxs
                                 )
    if not return_tf:
        return map_out
    else:
        return map_out, tf_out

def _parse_fragment_answer(Ntf, top, fragments, name,**guess_kwargs):
    guess = _guess_nomenclature_fragments(Ntf, top, fragments,**guess_kwargs)
    answer = input("Which fragments are succeptible of %s-numbering?"
                   "(Can be in a format 1,2-6,10,20-25)\n"
                   "Leave empty to accept our guess %s\n" % (name, guess))
    if answer is '':
        answer = ','.join(['%s' % ii for ii in guess])

    return answer

def _parse_fragment_naming_options(fragment_names, fragments, top):
    if fragment_names == '':
        fragment_names = ['frag%u' % ii for ii in range(len(fragments))]
    elif fragment_names.lower()=="none":
        fragment_names = [None for __ in fragments]
    else:
        assert isinstance(fragment_names, str), "Argument --fragment_names invalid: %s" % fragment_names
        if 'danger' not in fragment_names.lower():
            fragment_names = [ff.strip(" ") for ff in fragment_names.split(",")]
            assert len(fragment_names) == len(
                fragments), "Mismatch between nr. fragments and fragment names %s vs %s (%s)" % (
                len(fragments), len(fragment_names), fragment_names)

        elif 'danger' in fragment_names.lower():
            fragments, fragment_names = dangerously_auto_fragments(top,
                                                                   method="bonds",
                                                                   verbose=False,
                                                                   force_resSeq_breaks=True,
                                                                   frag_breaker_to_pick_idx=0,
                                                                   )
            fragment_names.extend(top.residue(ifrag[0]).name for ifrag in fragments[len(fragment_names):])

            for ifrag_idx, (ifrag, frag_name) in enumerate(zip(fragments, fragment_names)):
                _print_frag(ifrag_idx, top, ifrag, end='')
                print(" ", frag_name)

    return fragment_names, fragments


def residue_neighborhoods(topology, trajectories, resSeq_idxs,
                          res_idxs=False,
                          ctc_cutoff_Ang=3,
                          stride=1,
                          n_ctcs=5,
                          n_nearest=4,
                          chunksize_in_frames=10000,
                          nlist_cutoff_Ang=15,
                          n_smooth_hw=0,
                          ask=True,
                          sort=True,
                          pbc=True,
                          fragmentify=True,
                          fragment_names="",
                          graphic_ext=".pdf",
                          output_ascii=None,
                          BW_file="None",
                          CGN_PDB="None",
                          output_dir='.',
                          color_by_fragment=True,
                          output_desc='neighborhood',
                          t_unit='ns',
                          curve_color="auto",
                          gray_background=False,
                          graphic_dpi=150,
                          short_AA_names=False,
                          ):
    # todo use a proper unit module
    # like this https://pypi.org/project/units/
    if t_unit == 'ns':
        dt = 1e-3
    elif t_unit == 'mus':
        dt = 1e-6
    else:
        raise ValueError("Time unit not known ", t_unit)

    _offer_to_create_dir(output_dir)

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
        refgeom = md.load(topology)
    else:
        refgeom = topology
    if fragmentify:
        fragments = get_fragments(refgeom.top,method='bonds')
    else:
        raise NotImplementedError("This feature is not yet implemented")

    fragment_names, fragments = _parse_fragment_naming_options(fragment_names, fragments, refgeom.top)

    # Do we want BW definitions
    BW = _parse_consensus_option(BW_file, 'BW', refgeom.top, fragments)

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
        resSeq2residxs, _ = interactive_fragment_picker_by_resSeq(resSeq_idxs, fragments, refgeom.top,
                                                                  pick_first_fragment_by_default=not ask,
                                                                  additional_naming_dicts={"BW":BW,"CGN":CGN}
                                                                  )
    print("\nWill compute neighborhoods for the residues with resid")
    print("%s" % resSeq_idxs)
    print("excluding %u nearest neighbors\n" % n_nearest)

    print('%10s  %10s  %10s  %10s %10s %10s' % tuple(("residue  residx fragment  resSeq BW  CGN".split())))
    for key, val in resSeq2residxs.items():
        print('%10s  %10u  %10u %10u %10s %10s' % (refgeom.top.residue(val), val, in_what_fragment(val, fragments),
                                                  key,
                                                  BW[val], CGN[val]))

    # Create a neighborlist
    nl = bonded_neighborlist_from_top(refgeom.top, n=n_nearest)

    # Use it to prune the contact indices
    ctc_idxs = np.vstack(
        [[np.sort([val, ii]) for ii in range(refgeom.top.n_residues) if ii not in nl[val] and ii != val] for val in
         resSeq2residxs.values()])

    print(
        "\nPre-computing likely neighborhoods by reducing the neighbor-list to %u Angstrom in the reference geom %s..." % (
            nlist_cutoff_Ang, topology), end="", flush=True)
    ctcs, ctc_idxs = md.compute_contacts(refgeom, np.vstack(ctc_idxs), periodic=pbc)
    print("done!")

    ctc_idxs_small = np.argwhere(ctcs[0] < nlist_cutoff_Ang / 10).squeeze()
    _, ctc_idxs_small = md.compute_contacts(refgeom, ctc_idxs[ctc_idxs_small])
    ctc_idxs_small = unique_list_of_iterables_by_tuple_hashing(ctc_idxs_small)

    print("From %u potential distances, the neighborhoods have been reduced to only %u potential contacts.\nIf this "
          "number is still too high (i.e. the computation is too slow), consider using a smaller nlist_cutoff_Ang " % (
              len(ctc_idxs), len(ctc_idxs_small)))

    ctcs_trajs, time_array = xtcs2ctcs(xtcs, refgeom.top, ctc_idxs_small, stride=stride,
                                       chunksize=chunksize_in_frames, return_time=True,
                                       consolidate=False
                                       )
    actcs = np.vstack(ctcs_trajs)
    ctcs_mean = np.mean(actcs < ctc_cutoff_Ang / 10, 0)

    final_look = ctc_freq_reporter_by_residue_neighborhood(ctcs_mean, resSeq2residxs,
                                                           fragments, ctc_idxs_small,
                                                           refgeom.top,
                                                           interactive=False,
                                                           n_ctcs=n_ctcs)

    # Create the neighborhoods as groups of contact_pair objects
    neighborhoods = {}
    for key, val in final_look.items():
        neighborhoods[key] = []
        for idx in val:
            pair = ctc_idxs_small[idx]
            consensus_labels = [_relabel_consensus(idx, [BW, CGN]) for idx in pair]
            fragment_idxs = [in_what_fragment(idx, fragments) for idx in pair]
            neighborhoods[key].append(contact_pair(pair,
                                                   [itraj[:, idx] for itraj in ctcs_trajs],
                                                   time_array,
                                                   top=refgeom.top,
                                                   anchor_residue_idx=key,
                                                   consensus_labels=consensus_labels,
                                                   trajs=xtcs,
                                                   fragment_idxs=fragment_idxs,
                                                   fragment_names=[fragment_names[idx] for idx in fragment_idxs],
                                                   fragment_colors=[fragcolors[idx] for idx in fragment_idxs]
                                                   ))
        neighborhoods[key] = contact_group(neighborhoods[key])

    panelheight = 3
    n_cols = np.min((4, len(resSeq2residxs)))
    n_rows = np.ceil(len(resSeq2residxs) / n_cols).astype(int)
    panelsize = 4
    panelsize2font = 3.5
    histofig, histoax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True,
                                     figsize=(n_cols * panelsize * 2, n_rows * panelsize), squeeze=False)

    # One loop for the histograms
    _rcParams["font.size"]=panelsize*panelsize2font
    for jax, ihood in zip(histoax.flatten(),
                                   neighborhoods.values()):

        ihood.histo_neighborhood(ctc_cutoff_Ang, n_nearest,
                                 jax=jax,
                                 xlim=n_ctcs,
                                 label_fontsize_factor=panelsize2font/panelsize,
                                 shorten_AAs=short_AA_names
                                 )


    histofig.tight_layout(h_pad=2, w_pad=0, pad=0)
    fname = "%s.overall.%s" % (output_desc, graphic_ext.strip("."))
    fname = path.join(output_dir, fname)
    histofig.savefig(fname, dpi=graphic_dpi)
    print("The following files have been created")
    print(fname)

    # One loop for the time resolved neighborhoods
    for ihood in neighborhoods.values():
        fname = '%s.%s.time_resolved.%s' % (output_desc,
                                            ihood.anchor_res_and_fragment_str.replace('*', ""),
                                            graphic_ext.strip("."))
        fname = path.join(output_dir, fname)
        myfig = ihood.plot_timedep_ctcs(panelheight, _my_color_schemes(curve_color),
                                        ctc_cutoff_Ang=ctc_cutoff_Ang,
                                        n_smooth_hw=n_smooth_hw,
                                        dt=dt,
                                        t_unit=t_unit,
                                        gray_background=gray_background,
                                        shorten_AAs=short_AA_names,
                                        plot_N_ctcs=True)

        # One title for all axes on top
        title = ihood.anchor_res_and_fragment_str
        if short_AA_names:
            title = ihood.anchor_res_and_fragment_str_short
        myfig.axes[0].set_title(title)
        myfig.tight_layout(h_pad=0,w_pad=0,pad=0)
        myfig.savefig(fname, bbox_inches="tight", dpi=graphic_dpi)
        plt.close(myfig)
        print(fname)
        if str(output_ascii).lower() != 'none' and str(output_ascii).lower().strip(".") in ["dat", "txt", "xlsx"]:
            ext = str(output_ascii).lower().strip(".")
            ihood.save_trajs(output_desc,ext,output_dir, dt=dt,t_unit=t_unit, verbose=True)
        print()

    return {"ctc_idxs": ctc_idxs_small,
            'ctcs_trajs': ctcs_trajs,
            'time_array': time_array}


def _sitefile2site(sitefile):
    with open(sitefile, "r") as f:
        idict = jsonload(f)
    try:
        idict["bonds"]["AAresSeq"] = [item.split("-") for item in idict["bonds"]["AAresSeq"] if item[0] != '#']
        idict["n_bonds"] = len(idict["bonds"]["AAresSeq"])
    except:
        print("Malformed .json file for the site %s" % _sitefile2site())
    if "sitename" not in idict.keys():
        idict["name"] = splitext(psplit(sitefile)[-1])[0]
    else:
        idict["name"] = psplit(idict["sitename"])[-1]
    return idict


def sites(topology,
          trajectories,
          site_files,
          ctc_cutoff_Ang=3,
          stride=1,
          chunksize_in_frames=10000,
          n_smooth_hw=0,
          pbc=True,
          BW_file="None",
          CGN_PDB="None",
          default_fragment_index=None,
          fragment_names="",
          fragmentify=True,
          output_npy="output_sites",
          output_dir='.',
          graphic_ext=".pdf",
          t_unit='ns',
          curve_color="auto",
          gray_background=False,
          graphic_dpi=150,
          output_desc="sites",
          short_AA_names=False,
          ):
    # todo use a proper unit module
    # like this https://pypi.org/project/units/
    if t_unit=='ns':
        dt = 1e-3
    elif t_unit=='mus':
        dt = 1e-6
    else:
        raise ValueError("Time unit not known ",t_unit)

    # todo this is an ad-hoc for a powerpoint presentation
    from matplotlib import rcParams
    rcParams["xtick.labelsize"] = np.round(rcParams["font.size"]*2)
    rcParams["ytick.labelsize"] = np.round(rcParams["font.size"]*2)
    rcParams["font.size"] = np.round(rcParams["font.size"] * 2)
    _offer_to_create_dir(output_dir)

    # Prepare naming
    desc_out = output_npy
    desc_out = desc_out.rstrip(".")

    # Inform about trajectories
    xtcs = sorted(trajectories)

    print("Will compute the sites\n %s\nin the trajectories:\n  %s\n with a stride of %u frames.\n" % (
        "\n ".join(site_files),
        "\n  ".join(xtcs), stride))
    # Inform about fragments
    refgeom = md.load(topology)
    if fragmentify:
        fragments = get_fragments(refgeom.top, verbose=False)
    else:
        raise NotImplementedError("This feature is not yet implemented")

    fragment_names, fragments = _parse_fragment_naming_options(fragment_names, fragments, refgeom.top)


    for ifrag_idx, (ifrag, frag_name) in enumerate(zip(fragments, fragment_names)):
        _print_frag(ifrag_idx, refgeom.top, ifrag, end='')
        print(" ", frag_name)

    # Do we want BW definitions
    BW = _parse_consensus_option(BW_file, 'BW', refgeom.top, fragments)

    # Dow we want CGN definitions:
    CGN = _parse_consensus_option(CGN_PDB, 'CGN', refgeom.top, fragments)


    #TODO PACKAGE THIS SOMEHOW BETTER
    sites = [_sitefile2site(ff) for ff in site_files]
    AAresSeq2residxs = _sites_to_AAresSeqdict(sites, refgeom.top, fragments,
                                              default_fragment_idx=default_fragment_index,
                                              fragment_names=fragment_names)


    print('%10s  %10s  %10s  %10s %10s' % tuple(("residue  residx fragment fragment_name CGN ".split())))
    for key, val in AAresSeq2residxs.items():
        print('%10s  %10u  %10u  %10s %10s' % (refgeom.top.residue(val), val, in_what_fragment(val, fragments), key, CGN[val]))

    ctc_idxs_small = _sites_to_ctc_idxs_old(sites, AAresSeq2residxs)
    ctcs, time_array = xtcs2ctcs(xtcs, refgeom.top, ctc_idxs_small, stride=stride,
                                 chunksize=chunksize_in_frames,
                                 return_time=True, consolidate=False, periodic=pbc)

    # Abstract each site to a group of contacts
    site_as_gc = {}
    ctc_pairs_iterators = iter(ctc_idxs_small)
    ctc_value_idx = iter(np.arange(len(ctc_idxs_small)))  # there has to be a better way
    for isite in sites:
        key = isite["name"]
        site_as_gc[key] = []
        for __ in range(isite["n_bonds"]):
            pair = next(ctc_pairs_iterators)
            idx = next(ctc_value_idx)
            consensus_labels = [_relabel_consensus(idx, [BW, CGN]) for idx in pair]
            fragment_idxs = [in_what_fragment(idx, fragments) for idx in pair]
            site_as_gc[key].append(contact_pair(pair,
                                               [itraj[:, idx] for itraj in ctcs],
                                               time_array,
                                               top=refgeom.top,
                                               consensus_labels=consensus_labels,
                                               trajs=xtcs,
                                               fragment_idxs=fragment_idxs,
                                               fragment_names=[fragment_names[idx] for idx in fragment_idxs],
                                               #fragment_colors=[fragcolors[idx] for idx in fragment_idxs]
                                               ))
        site_as_gc[key] = contact_group(site_as_gc[key])

    print("The following files have been created")
    panelheight = 3
    n_cols = np.min((4, len(sites)))
    n_rows = np.ceil(len(sites) / n_cols).astype(int)
    panelsize = 4
    panelsize2font = 3.5
    histofig, histoax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True,
                                     figsize=(n_cols * panelsize * 2, n_rows * panelsize), squeeze=False)

    # One loop for the histograms
    _rcParams["font.size"] = panelsize * panelsize2font
    for jax, (site_name, isite_nh) in zip(histoax.flatten(),
                                       site_as_gc.items()):
        isite_nh.histo_site(ctc_cutoff_Ang, site_name,
                         jax=jax,
                         xlim=np.max([ss["n_bonds"] for ss in sites]),
                         label_fontsize_factor=panelsize2font / panelsize,
                         shorten_AAs=short_AA_names
                         )


    histofig.tight_layout(h_pad=2, w_pad=0, pad=0)
    fname = "%s.overall.%s" % (output_desc, graphic_ext.strip("."))
    fname = path.join(output_dir, fname)
    histofig.savefig(fname, dpi=graphic_dpi)
    plt.close(histofig)
    print("The following files have been created")
    print(fname)
    for site_name, isite_nh in site_as_gc.items():
        fname = 'site.%s.%s.time_resolved.%s' % (
            site_name.replace(" ", "_"), desc_out.strip("."), graphic_ext.strip("."))
        fname = path.join(output_dir, fname)
        myfig = isite_nh.plot_timedep_ctcs(panelheight,
                                           _my_color_schemes(curve_color),
                                           ctc_cutoff_Ang=ctc_cutoff_Ang,
                                           n_smooth_hw=n_smooth_hw,
                                           dt=dt,
                                           t_unit=t_unit,
                                           gray_background=gray_background,
                                           shorten_AAs=short_AA_names,
                                           plot_N_ctcs=False
                                           )
        # One title for all axes on top
        myfig.axes[0].set_title("site: %s" % (isite["name"]))
        plt.savefig(fname, bbox_inches="tight", dpi=graphic_dpi)
        plt.close(myfig)
        print(fname)

    return

def density_by_sites(topology,
          trajectories,
          site_files,
          default_fragment_index=None,
          desc_out="density",
          output_dir='.',
                     stride=1,
          ):

    from subprocess import check_output as _co, STDOUT as _STDOUT, CalledProcessError

    _offer_to_create_dir(output_dir)

    # Prepare naming
    desc_out = desc_out.rstrip(".")

    # Inform about trajectories
    xtcs = sorted(trajectories)

    print("Will compute the densities for sites\n %s\nin the trajectories:\n  %s" % (
        "\n ".join(site_files),
        "\n  ".join(xtcs)))
    # Inform about fragments
    refgeom = md.load(topology)
    fragments = get_fragments(refgeom.top)

    sites = [_sitefile2site(ff) for ff in site_files]

    with _TD() as tmpdirname:
        tocat=xtcs
        if stride > 1:
            tocat=[]
            for ii, ixtc in enumerate(xtcs):
                strided_file = path.join(tmpdirname,'%u.strided.%u.xtc'%(ii,stride))
                tocat.append(strided_file)
                cmd = "gmx trjconv -f %s -o %s -skip %u"%(ixtc, strided_file,stride)
                # OMG SUPER DANGEROUS
                try:
                    _co(cmd.split())
                except:
                    pass
        trjcatted_tmp='trjcatted.xtc'
        trjcatted_tmp=path.join(tmpdirname,trjcatted_tmp)
        cmd = "gmx trjcat -f %s -o %s -cat"%(' '.join(tocat),trjcatted_tmp)
        _co(cmd.split())

        for ss in sites:
            #print(ss["name"])
            AAresSeqs = ss["bonds"]["AAresSeq"]
            AAresSeqs = [item for sublist in AAresSeqs for item in sublist]
            #AAresSeqs = [item for sublist in AAresSeqs for item in sublist]

            resSeq2residxs, _ = _interactive_fragment_picker_by_AAresSeq(AAresSeqs, fragments, refgeom.top,
                                                                     default_fragment_idx=default_fragment_index,
                                                                     )

            aa_in_sites = np.unique(np.hstack([[aa.index for aa in refgeom.top.residue(ii).atoms] for ii in resSeq2residxs.values()]))
            aa_in_sites_gmx = aa_in_sites+1
            atom_in_sites_gmx_str=','.join([str(ii) for ii in aa_in_sites_gmx])
            if False:
                for ixtc in xtcs:
                    outfile = '%s.site.%s'%(splitext(ixtc)[0],ss["name"])
                    outfile = path.join(output_dir,outfile)

                    cmd = "gmx_gromaps maptide -spacing .05 -f %s -s %s -mo %s -select 'atomnr %s'"%(ixtc,topology,outfile,atom_in_sites_gmx_str)
                    print(cmd)
                    out = _co(_cmdstr2cmdtuple(cmd))#, stderr=_STDOUT)  # this should, in principle, work
                    #cmd = (gmxbin, "check", "-f", fname)

            outfile = '%s.site.%s.stride.%02u'%(desc_out.strip("."),ss["name"].strip("."),stride)
            outfile = path.join(output_dir,outfile)
            cmd = "gmx_gromaps maptide -spacing .1 -f %s -s %s -mo %s -select 'atomnr %s'"%(trjcatted_tmp,topology,outfile,atom_in_sites_gmx_str)
            print(cmd)
            out = _co(_cmdstr2cmdtuple(cmd))

def site_figures(topology,
          site_files,
                 BW_file="None",
                 CGN_PDB="None",
          ):


    print("Will print VMD lines for sites\n %s"%(
        "\n ".join(site_files)))
    # Inform about fragments
    refgeom = md.load(topology)
    fragments = get_fragments(refgeom.top)

    sites = [_sitefile2site(ff) for ff in site_files]

    # Dow we want CGN definitions:

    CGN = _parse_consensus_option(CGN_PDB, 'CGN', refgeom.top, fragments)
    #for key, val in CGN.items():
    #    print(key,val)
    for ss in sites:
        #print(ss["name"])
        AAresSeqs = ss["bonds"]["AAresSeq"]
        AAresSeqs = [item for sublist in AAresSeqs for item in sublist]

        resSeq2residxs, _ = _interactive_fragment_picker_by_AAresSeq(AAresSeqs, fragments, refgeom.top)
        ctc_idxs = _sites_to_ctc_idxs_old([ss], resSeq2residxs)
        title="\nsite: %s"%ss["name"]
        if "info" in ss.keys():
            title+= ' (%s)'%ss["info"]
        print(title)
        if "best run" in ss.keys():
            vmd_cmd =  "mol new %s\n"%topology
            vmd_cmd += "mol addfile %s waitfor all\n"%(ss["best run"])
            vmd_cmd += 'mol addfile figs/density.site.%s.ccp4\n'%ss["name"]
            print(vmd_cmd[:-1])
        from .aa_utils import shorten_AA as _shorten_AA
        for r1,r2 in ctc_idxs:
            r1 = refgeom.top.residue(r1)
            r2 = refgeom.top.residue(r2)
            print("(resname %s and resid %s or resname %s and resid %s) and noh and not name O N "%(r1.name,r1.resSeq, r2.name,r2.resSeq))
            print('%s-%s'%(_shorten_AA(r1), _shorten_AA(r2)))
            print('%s-%s'%(CGN[r1.index], CGN[r2.index]))


        input()

def interface(
    topology=None,
    trajectories=None,
    BW_file="None",
    CGN_PDB="None",
    chunksize_in_frames=10000,
    ctc_cutoff_Ang=3,
    curve_color="auto",
    frag_idxs_group_1=None,
    frag_idxs_group_2=None,
    fragments=['resSeq'],
    graphic_dpi=150,
    graphic_ext=".pdf",
    gray_background=False,
    interface_cutoff_Ang=35,
    n_ctcs=10,
    n_smooth_hw=0,
    output_desc="interface",
    output_dir=".",
    short_AA_names=False,
    stride=1,
    t_unit="ns",
):
    # todo use a proper unit module
    # like this https://pypi.org/project/units/
    if t_unit == 'ns':
        dt = 1e-3
    elif t_unit == 'mus':
        dt = 1e-6
    else:
        raise ValueError("Time unit not known ", t_unit)

    xtcs = sorted(trajectories)
    print("Will compute contact frequencies for the files:\n  %s\n with a stride of %u frames.\n" % (
    "\n  ".join(xtcs), stride))

    group_1 = frag_idxs_group_1
    group_2 = frag_idxs_group_2

    refgeom = md.load(topology)
    frag_cons = False
    auto_fragments = get_fragments(refgeom.top)
    if isinstance(fragments, list):
        if isinstance(fragments[0], str):
            if fragments[0] in ['resSeq']:
                assert len(auto_fragments) >= 2, ("You need more than one chain in "
                                             "your topology if you are going to "
                                             "auto-dectect fragments! Aborting.")
            elif fragments[0] == 'consensus':
                frag_cons = True
            else:
                raise ValueError("Cannot understand your --fragments option", fragments)
        else:
            auto_fragments = []
            assert len(fragments) >= 2, ("You need to input more than "
                                           "two fragments! Aborting.")
            print("User input")
            for ii, ifrag in enumerate(auto_fragments):
                ifrag = rangeexpand(ifrag)
                auto_fragments.append(ifrag)
                _print_frag(ii, refgeom.top, ifrag)
    else:
        raise Exception("Cannot understand your --fragments option %s" % fragments)

    # TODO THIS RENAMING HERE FOR CONSISTENCY WITH PARSERS WILL BE A PROBLEM LATER
    fragments = auto_fragments

    BW, BWtf = _parse_consensus_option(BW_file, 'BW', refgeom.top, fragments, return_tf=True)
    fragment_defs = {}
    if str(BW_file).lower() != 'none':
        print("INFO: these are the BW fragments mapped onto your topology")
        fragment_defs.update(BWtf.top2defs(refgeom.top, return_defs=True))
        input("Hit enter to continue!\n")
    CGN, CGNtf = _parse_consensus_option(CGN_PDB, 'CGN', refgeom.top, fragments, return_tf=True)
    if str(CGN_PDB).lower() != 'none':
        print("INFO: these are the CGN fragments mapped onto your topology")
        fragment_defs.update(CGNtf.top2defs(refgeom.top, return_defs=True))
        input("Hit enter to continue!\n")

    groups = []
    group_idxs = []
    if frag_cons:
        for ii in [1, 2]:
            print("group %u: " % ii, end='')
            # TODO mange this vars better
            istr = str([None,
                        frag_idxs_group_1,
                        frag_idxs_group_2][ii])
            if istr.lower() != 'none':
                # TODO we're calling this method twice but I don't see a way around it RN
                igroup, igroup_idxs = _match_dict_by_patterns(istr, fragment_defs)
                if len(igroup) == len(igroup_idxs) == 0:
                    raise ValueError("Your input '%s' does not match any fragment groups" % istr)
                else:
                    answer = istr
            else:
                answer = input(
                    "Input a list of posix-expressions like 'TM*,-TM2,H8' (TM2 will be avoided)\n").replace(
                    " ", "").strip("'").strip('"')
            igroup, igroup_idxs = _match_dict_by_patterns(answer, fragment_defs)
            groups.append(igroup)
            group_idxs.append(igroup_idxs)
            print(', '.join(groups[-1]))
        group_1, group_2 = groups
        fragments = list(fragment_defs.values())
        fragment_names = list(fragment_defs.keys())

    if len(groups) == 0:
        if len(fragments) == 2:
            print("Only two fragments detected. Overriding inputs for -frag_idxs_group_1/2 with [0] and [1].")
            group_1 = [0]
            group_2 = [1]
        else:
            group_1 = rangeexpand(group_1)
            group_2 = rangeexpand(group_2)
        ctc_idxs = np.hstack([[list(product(fragments[ii], fragments[jj])) for ii in group_1] for jj in group_2])
    else:
        ctc_idxs = np.vstack(list(product(group_idxs[0], group_idxs[1])))

    print("\nComputing distances in the interface between fragments\n%s\nand\n%s.\n"
          "The interface is defined by the residues within %3.1f "
          "Angstrom of each other in the reference topology.\n"
          "Computing interface..."
          % ('\n'.join(_twrap(', '.join(['%s' % gg for gg in group_1]))),
             '\n'.join(_twrap(', '.join(['%s' % gg for gg in group_2]))),
             interface_cutoff_Ang), end="")

    ctcs, ctc_idxs = md.compute_contacts(refgeom, np.vstack(ctc_idxs))
    print("done!")

    ctc_idxs_receptor_Gprot = np.argwhere(ctcs[0] < interface_cutoff_Ang / 10).squeeze()
    _, ctc_idxs_receptor_Gprot = md.compute_contacts(refgeom, ctc_idxs[ctc_idxs_receptor_Gprot])

    print()
    print(
        "From %u potential group_1-group_2 distances, the interface was reduced to only %u potential contacts.\nIf this "
        "number is still too high (i.e. the computation is too slow) consider using a smaller interface cutoff" % (
        len(ctc_idxs), len(ctc_idxs_receptor_Gprot)))

    ctcs, times = xtcs2ctcs(xtcs, refgeom.top, ctc_idxs_receptor_Gprot, stride=stride, return_time=True,
                            consolidate=False,
                            chunksize=chunksize_in_frames)

    # Stack all data
    actcs = np.vstack(ctcs)

    # Get frequencies so that we don't create unnecessary ctc objects
    ctcs_bin = (actcs <= ctc_cutoff_Ang / 10).astype("int").sum(0)
    ctc_frequency = ctcs_bin / actcs.shape[0]
    order = np.argsort(ctc_frequency)[::-1]
    #ctcs_trajectory_std = np.vstack([np.mean(ictcs < ctc_cutoff_Ang / 10, 0) for ictcs in ctcs]).std(0)

    ctc_objs = []
    for idx in order[:n_ctcs]:
        ifreq = ctc_frequency[idx]
        if ifreq > 0:
            pair = ctc_idxs_receptor_Gprot[idx]
            consensus_labels = [_relabel_consensus(idx, [BW, CGN]) for idx in pair]
            fragment_idxs = [in_what_fragment(idx, fragments) for idx in pair]
            ctc_objs.append(contact_pair(pair,
                                         [itraj[:, idx] for itraj in ctcs],
                                         times,
                                         top=refgeom.top,
                                         consensus_labels=consensus_labels,
                                         trajs=xtcs,
                                         fragment_idxs=fragment_idxs,
                                         # fragment_names=fragment_names,#[fragment_names[idx] for idx in fragment_idxs],
                                         # fragment_colors=[fragcolors[idx] for idx in fragment_idxs]
                                          ))

    neighborhood = contact_group(ctc_objs)
    print(neighborhood.frequency_report(ctc_cutoff_Ang))
    panelheight = 3
    n_cols = 1
    n_rows = 1
    panelsize = 4
    panelsize2font = 3.5
    fudge = 7
    histofig, histoax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True,
                                     figsize=(n_cols * panelsize * np.ceil(neighborhood.n_ctcs/fudge),
                                              n_rows * panelsize),
                                     )

    # One loop for the histograms
    _rcParams["font.size"] = panelsize * panelsize2font
    neighborhood.histo_site(ctc_cutoff_Ang,
                            'interface',
                            jax=histoax,
                            xlim=n_ctcs,
                            label_fontsize_factor=panelsize2font / panelsize,
                            shorten_AAs=short_AA_names
                            )
    histofig.tight_layout(h_pad=2, w_pad=0, pad=0)
    fname = "%s.overall.%s" % (output_desc, graphic_ext.strip("."))
    fname = path.join(output_dir, fname)
    histofig.savefig(fname, dpi=graphic_dpi)
    print("The following files have been created")
    print(fname)

    fname = '%s.time_resolved.%s' % (output_desc,
                                     graphic_ext.strip("."))
    fname = path.join(output_dir, fname)
    myfig = neighborhood.plot_timedep_ctcs(panelheight, _my_color_schemes(curve_color),
                                           ctc_cutoff_Ang=ctc_cutoff_Ang,
                                           n_smooth_hw=n_smooth_hw,
                                           dt=dt,
                                           t_unit=t_unit,
                                           gray_background=gray_background,
                                           shorten_AAs=short_AA_names,
                                           plot_N_ctcs=True)

    # One title for all axes on top
    myfig.axes[0].set_title("site: %s" % (output_desc))
    plt.savefig(fname, bbox_inches="tight", dpi=graphic_dpi)
    plt.close(myfig)
    print(fname)

    return neighborhood

def _cmdstr2cmdtuple(cmd):
    return [ii.replace("nr", "nr ") for ii in cmd.replace("atomnr ", "atomnr").replace("'", "").split()]

def _sites_to_AAresSeqdict(sites, top, fragments,
                           raise_if_not_found=True,
                           **interactive_fragment_picker_by_AAresSeq_kwargs):


    AAresSeqs = [ss["bonds"]["AAresSeq"] for ss in sites]
    AAresSeqs = [item for sublist in AAresSeqs for item in sublist]
    AAresSeqs = [item for sublist in AAresSeqs for item in sublist]

    AAresSeq2residxs, _ = _interactive_fragment_picker_by_AAresSeq(AAresSeqs, fragments, top,
                                                                 **interactive_fragment_picker_by_AAresSeq_kwargs)

    if None in AAresSeq2residxs.values() and raise_if_not_found:
        raise ValueError("These residues of your input have not been found. Please revise it:\n%s" % (
            '\n'.join([key for key, val in AAresSeq2residxs.items() if val is None])))

    return AAresSeq2residxs

def _sites_to_ctc_idxs(sites, top,**kwargs):
    fragments = get_fragments(top)
    AAresSeq2residxs = _sites_to_AAresSeqdict(sites,top, fragments)
    return _sites_to_ctc_idxs_old(sites, AAresSeq2residxs)


def _sites_to_ctc_idxs_old(sites, AAresSeq2residxs):
    return np.vstack(([[[AAresSeq2residxs[pp] for pp in pair] for pair in ss["bonds"]["AAresSeq"]] for ss in sites]))


def _my_color_schemes(istr):
    return {"peter": ["red", "purple", "gold", "darkorange"],
            "hobat": ["m", "darkgreen", "darkorange", "navy"],
            "auto":  plt.rcParams['axes.prop_cycle'].by_key()["color"]}[str(istr).lower()]
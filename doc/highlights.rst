.. https://stackoverflow.com/a/31332035 for forcing paragraph braks after figure captions
.. |nbspc| unicode:: U+00A0 .. non-breaking space


Highlights
----------

.. _`initial example`:
* paper-ready tables and figures from the command line::

   mdc_neighborhoods.py prot.pdb traj.xtc -r L394 --GPCR adrb2_human --CGN 3SN6 -ni -at #ni: not interactive, at: show atom-types

  .. figure:: imgs/bars_and_PDF.png
      :scale: 40%
      :align: left
      :name: highlights_1

      (click to enlarge) **a)** contact frequencies for LEU394, as in :numref:`freqs`, but annotated with consensus nomenclature and atom types (``--GPCR,--CGN,-at`` options, see below). **b)** associated distance distributions, obtained by adding the ``-d`` flag to the CLI call. **c)** Automatically generated table using the ``-tx xlsx`` option.

  |nbspc|
.. _consensus_HL:

* *automagically* map and incorporate consensus nomenclature to the analysis, either from local files or over the network in the `GPRC.db <https://gpcrdb.org/>`_ and from `<https://www.mrc-lmb.cam.ac.uk/CGN/>`_::

   ...
   No local file ./adrb2_human.xlsx found, checking online in
   https://gpcrdb.org/services/residues/extended/adrb2_human ...done!
   Please cite the following reference to the GPCRdb:
    * Kooistra et al, (2021) GPCRdb in 2021: Integrating GPCR sequence, structure and function
      Nucleic Acids Research 49, D335--D343
      https://doi.org/10.1093/nar/gkaa1080
   For more information, call mdciao.nomenclature.references()
   done without 404, continuing.
   GPCR-labels align best with fragments: [3] (first-last: GLU30-LEU340).
   These are the GPCR fragments mapped onto your topology:
       TM1 with     32 AAs    GLU30@1.29x29   (   760) -    PHE61@1.60x60   (791   ) (TM1)
      ICL1 with      4 AAs    GLU62@12.48x48  (   792) -    GLN65@12.51x51  (795   ) (ICL1)
       TM2 with     32 AAs    THR66@2.37x37   (   796) -    LYS97@2.68x67   (827   ) (TM2)
      ECL1 with      4 AAs    MET98@23.49x49  (   828) -   PHE101@23.52x52  (831   ) (ECL1)
       TM3 with     36 AAs   GLY102@3.21x21   (   832) -   SER137@3.56x56   (867   ) (TM3)
      ICL2 with      8 AAs   PRO138@34.50x50  (   868) -   LEU145@34.57x57  (875   ) (ICL2)
       TM4 with     27 AAs   THR146@4.38x38   (   876) -   HIS172@4.64x64   (902   ) (TM4)
      ECL2 with     20 AAs   TRP173           (   903) -   THR195           (922   ) (ECL2)  resSeq jumps
       TM5 with     42 AAs   ASN196@5.35x36   (   923) -   GLU237@5.76x76   (964   ) (TM5)
      ICL3 with      2 AAs   GLY238           (   965) -   ARG239           (966   ) (ICL3)
       TM6 with     35 AAs   CYS265@6.27x27   (   967) -   GLN299@6.61x61   (1001  ) (TM6)
      ECL3 with      4 AAs   ASP300           (  1002) -   ILE303           (1005  ) (ECL3)
       TM7 with     25 AAs   ARG304@7.31x30   (  1006) -   ARG328@7.55x55   (1030  ) (TM7)
        H8 with     12 AAs   SER329@8.47x47   (  1031) -   LEU340@8.58x58   (1042  ) (H8)
   ...
   ...
   Using CGN-nomenclature, please cite
    * Flock et al, (2015) Universal allosteric mechanism for G$\alpha$ activation by GPCRs
      Nature 2015 524:7564 524, 173--179
      https://doi.org/10.1038/nature14663
   No local file ./CGN_3SN6.txt found, checking online in
   https://www.mrc-lmb.cam.ac.uk/CGN/lookup_results/3SN6.txt ...done without 404, continuing.
   No local PDB file for 3SN6 found in directory '.', checking online in
   https://files.rcsb.org/download/3SN6.pdb ...found! Continuing normally
   CGN-labels align best with fragments: [0] (first-last: LEU4-LEU394).
   These are the CGN fragments mapped onto your topology:
      G.HN with   28 AAs     THR9@G.HN.26   (   5) -    VAL36@G.HN.53   (32  ) (G.HN)
    G.hns1 with    3 AAs    TYR37@G.hns1.1  (  33) -    ALA39@G.hns1.3  (35  ) (G.hns1)
      G.S1 with    7 AAs    THR40@G.S1.1    (  36) -    LEU46@G.S1.7    (42  ) (G.S1)
   ...
    G.hgh4 with   10 AAs   TYR311@G.hgh4.1  ( 270) -   THR320@G.hgh4.10 (279 ) (G.hgh4)
      G.H4 with   27 AAs   PRO321@G.H4.1    ( 280) -   ARG347@G.H4.27   (306 ) (G.H4)
    G.h4s6 with   11 AAs   ILE348@G.h4s6.1  ( 307) -   TYR358@G.h4s6.20 (317 ) (G.h4s6)
      G.S6 with    5 AAs   CYS359@G.S6.1    ( 318) -   PHE363@G.S6.5    (322 ) (G.S6)
    G.s6h5 with    5 AAs   THR364@G.s6h5.1  ( 323) -   ASP368@G.s6h5.5  (327 ) (G.s6h5)
      G.H5 with   26 AAs   THR369@G.H5.1    ( 328) -   LEU394@G.H5.26   (353 ) (G.H5)
   ...

  |nbspc|
.. _residues_HL:

* easy residue selection, with an extended `UNIX-like pattern matching <https://docs.python.org/3/library/fnmatch.html>`_. You can preview residue selections like ``GLU,P0G,3.50,380-394,G.HN.*``, which means:

  - *GLU**: all GLUs, equivalent to *GLU*
  - *P0G*: the B2AR ligand (agonist)
  - *3.50*: Ballesteros-Weinstein notation
  - *380-394*: range of residues with sequence indices 380 to 394 (both incl.). This is the :math:`G\alpha_5`-subunit.
  - *G.HN.** : CGN-nomenclature for the :math:`G\alpha_N`-subunit
 You can check your selection **before** running a computation by using ``mdc_residues.py``::

  >>> mdc_residues.py GLU*,P0G,380-394,G.HN.* prot.pdb --GPCR adrb2_human --CGN 3SN6 -ni
  Your selection 'GLU*,P0G,380-394,G.HN.*' yields:
    residue      residx    fragment      resSeq       GPCR        CGN
      GLU10           6           0         10       None    G.HN.27
      GLU15          11           0         15       None    G.HN.32
  ...
     GLU306        1008           3        306       7.33       None
     GLU338        1040           3        338       8.56       None
     P0G395        1043           4        395       None       None
     ARG131         861           3        131       3.50       None
     ARG380         339           0        380       None    G.H5.12
     ASP381         340           0        381       None    G.H5.13
  ...
     LEU393         352           0        393       None    G.H5.25
     LEU394         353           0        394       None    G.H5.26
       THR9           5           0          9       None    G.HN.26
      ASP11           7           0         11       None    G.HN.28
      GLN12           8           0         12       None    G.HN.29
  ...
      GLN35          31           0         35       None    G.HN.52
      VAL36          32           0         36       None    G.HN.53


 |nbspc|
.. _pdb_HL:

* easy grabbing structures from the RSC PDB::

   >>> mdc_pdb.py 3SN6 -o 3SN6.gro

   Checking https://files.rcsb.org/download/3SN6.pdb ...done
   Saving to 3SN6.gro...done
   Please cite the following 3rd party publication:
    * Crystal structure of the beta2 adrenergic receptor-Gs protein complex
     Rasmussen, S.G. et al., Nature 2011
     https://doi.org/10.1038/nature10361

  |nbspc|
.. _fragmentation_HL:

* fragmentation heuristics to easily identify molecules and/or molecular fragments. These heuristics will work on .pdf-files lacking `TER and CONNECT records <http://www.wwpdb .org/documentation/file-format-content/format33/v3.3.html>`_ or other file formats, like `.gro files <http://manual.gromacs.org/documentation/2020/reference-manual/file-formats.html#gro>`_, that simply don't include these records::

   Auto-detected fragments with method 'lig_resSeq+'
   fragment      0 with  349 AAs     THR9           (   0) -   LEU394           (348 ) (0)  resSeq jumps
   fragment      1 with  340 AAs     GLN1           ( 349) -   ASN340           (688 ) (1)
   fragment      2 with   58 AAs     ASN5           ( 689) -    ARG62           (746 ) (2)
   fragment      3 with  159 AAs  ASN1002           ( 747) -  ALA1160           (905 ) (3)
   fragment      4 with  284 AAs    GLU30           ( 906) -   CYS341           (1189) (4)  resSeq jumps
   fragment      5 with  128 AAs     GLN1           (1190) -   SER128           (1317) (5)
   fragment      6 with    1 AAs  P0G1601           (1318) -  P0G1601           (1318) (6)

  In this example, we saved the crystal structure `3SN6 <https://www.rcsb.org/structure/3SN6>`_ as a .gro-file (``mdc_pdb.py 3SN6 -o 3SN6.gro``). We are able to recover sensible fragments:

  * :math:`G\alpha`
  * :math:`G\beta`
  * :math:`G\gamma`
  * bacteriophage T4 lysozyme as N-terminus of the receptor (next)
  * :math:`\beta 2` adrenergic receptor
  * VHH antibody
  * ligand.

  For clarity, we omitted the fragmentation in our `initial example`_ with the option ``-nf``, but all CLI tools do this fragmentation by default. Alternatively, one can use::

   mdc_fragments.py 3SN6.gro

  to get an overview of all available fragmentation heuristics and their results without computing any contacts whatsoever.

.. _`mdc_interface.py example`:

* use fragment definitions --like the ones above, ``0`` for the :math:`G\alpha`-unit and ``3`` for the receptor-- to compute interfaces in an automated way, i.e. without having to specifying individual residues::

   >>> mdc_interface.py prot.pdb traj.xtc -fg1 0 -fg2 3 --GPCR adrb2_human --CGN 3SN6 -t "3SN6 beta2AR-Galpha interface" -ni
   ...
   These 50 contacts capture 15.40 (~99%) of the total frequency 15.52 (over 21177 contacts)
   As orientation value, 31 ctcs already capture 90.0% of 15.52.
   The 31-th contact has a frequency of 0.14
       freq                         label residue idxs    sum
   0   1.00   D381@G.H5.13    - Q229@5.68      340 956   1.00
   1   1.00   R385@G.H5.17    - Q229@5.68      344 956   2.00
   2   1.00   D381@G.H5.13    - K232@5.71      340 959   3.00
   3   0.98   Q384@G.H5.16    - I135@3.54      343 865   3.98
   4   0.96   T350@G.h4s6.3   - R239@ICL3      309 966   4.93
   5   0.85   E392@G.H5.24    - T274@6.36      351 976   5.79
   6   0.68   Q384@G.H5.16    - Q229@5.68      343 956   6.46
   ...
   The following files have been created:
   ./interface.overall@3.5_Ang.xlsx
   ./interface.overall@3.5_Ang.dat
   ./interface.overall@3.5_Ang.as_bfactors.pdb
   ./interface.overall@3.5_Ang.pdf
   ./interface.matrix@3.5_Ang.pdf
   ./interface.flare@3.5_Ang.pdf
   ./interface.time_trace@3.5_Ang.pdf
   ./interface.mdciaoCG.traj.dat

 .. figure:: imgs/interface.matrix@3.5_Ang.Fig.4.png
      :scale: 25%
      :align: left
      :name: interface_matrix

      [``interface.matrix@3.5_Ang.pdf``](click to enlarge). Interface contact matrix between the :math:`\beta`2AR receptor and the :math:`G\alpha`-unit protein, using a cutoff of 3.5 AA. The labelling incorporates consensus nomenclature to identify positions and domains of both receptor and G-protein. Please note: this is **not a symmetric** contact-matrix. The y-axis shows residues in the :math:`G\alpha`-unit and the x-axis in the receptor.

* Since :numref:`interface_matrix` is bound to incorporate a lot of blank pixels, ``mdciao`` will also produce sparse plots and figures that highlight the formed contacts only:

 .. figure:: imgs/interface.overall@3.5_Ang.Fig.5.png
      :scale: 15%
      :align: left
      :name: interface_bars


      [``interface.overall@3.5_Ang.pdf``](click to enlarge) **Upper panel**: most frequent contacts sorted by frequency, i.e. for each non-empty pixel of :numref:`interface_matrix`, there is a bar shown. **Lower panel**: per-residue aggregated contact-frequencies, showing each residue's average participation in the interface (same info will be written to `interface.overall@3.5_Ang.xlsx`). Also, the number of shown contacts/bars can be controlled either with the `--ctc_control` and/or `--min_freq` parameters of `mdc_interface.py`.

* A very convenient way to incorporate the molecular topology into the visualization of contact frequencies are the so-called `FlarePlots <https://github.com/GPCRviz/flareplot>`_ (cool live-demo `here <https://gpcrviz.github.io/flareplot/>`_). These show the molecular topology (residues, fragments) on a circle with curves connecting the residues for which a given frequency has been computed. The `mdc_interface.py example`_ above will also generate a flareplot:

 .. figure:: imgs/interface.flare@3.5_Ang.small.png
      :scale: 70%
      :align: left
      :name: fig_flare

      [``interface.flare@3.5_Ang.pdf``](click to enlarge) FlarePlot of the frequencies shown in the figures :numref:`interface_matrix` and :numref:`interface_bars`. Residues are shown as dots on a circumference, split into fragments following any available labelling information. The contact frequencies are represented as lines connecting these dots/residues, with the line-opacity proportional to the frequencie's value. The secondary stucture of each residue is also included as color-coded letters: H(elix), B(eta), C(oil). We can clearly see the :math:`G\alpha_5`-subunit in contact with the receptor's TM3, ICL2, and TM5-ICL3-TM6 regions. Note that this plot is always produced as .pdf to be able to zoom into it as much as needed.

* Similar to how the flareplot (:numref:`fig_flare`) is mapping contact-frequencies (:numref:`interface_bars`, upper panel) onto the molecular topology, the next figure maps the **lower** panel :numref:`interface_bars` on the molecular geometry. It simply puts the values shown there in the `temperature factor <http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM>`_  of a pdb file, representing the calculated interface as a *heatmap*, which can be visualized in VMD using the `Beta coloring <https://www.ks.uiuc.edu/Research/vmd/vmd-1.7.1/ug/node74.html>`_.

 .. figure:: imgs/interface_BRG.png
      :scale: 70%
      :align: left

      [``interface.overall@3.5_Ang.as_bfactors.pdb``](click to enlarge) 3D visualization of the interface as heatmap (blue-green-red) using `VMD <https://www.ks.uiuc.edu/Research/vmd/>`_. We clearly see the regions noted in :numref:`fig_flare` (TM5-ICL3-TM6 and :math:`G\alpha_5`-subunit) in particular the **residues** of :numref:`interface_bars` (lower panel) light up. Please note that for the homepage-banner (red-blue heatmap), the ``signed_colors`` argument has been used when calling the :obj:`mdciao.flare.freqs2flare` method of the API. At the moment this is not possible just by using ``mdc_interface.py``, sorry!

 You can use this snippet to generate a VMD `visualiazation state` file, ``view_mdciao_interface.vmd`` to view the heatmap::

   echo 'mol new ./interface.overall@3.5_Ang.as_bfactors.pdb
         mol modstyle 0 0 NewCartoon
         mol modcolor 0 0 Beta
         color scale method BGR ' > view_mdciao_interface.vmd
   vmd -e view_mdciao_interface.vmd


 ``view_mdciao_interface.vmd`` will work with any ``*.as_bfactors.pdb`` file that ``mdciao`` generates. For our example, you can also paste this viewpoint into your VMD console and generate a view equivalent to the above picture (results may vary with other files)::

   molinfo top set {center_matrix rotate_matrix scale_matrix global_matrix} {{{1 0 0 -66.7954} {0 1 0 -66.6322} {0 0 1 -45.2629} {0 0 0 1}} {{-0.688392 0.720507 0.0835694 0} {-0.0925729 0.0269995 -0.995339 0} {-0.719405 -0.692919 0.0481138 0} {0 0 0 1}} {{0.0348044 0 0 0} {0 0.0348044 0 0} {0 0 0.0348044 0} {0 0 0 1}} {{1 0 0 0.15} {0 1 0 0.12} {0 0 1 0} {0 0 0 1}}}


* A different approach is to look **only** for a particular set of pre-defined contacts. Simply writing this set into a human readable `JSON <https://www.json.org/>`_ file will allow `mdc_sites.py` to compute and present these (and only these) contacts, as in the example file `tip.json`::


   >>> echo '
   >>> {"name":"interface small",
   >>> "pairs": {"AAresSeq": [
   >>>         "L394-K270",
   >>>         "D381-Q229",
   >>>         "Q384-Q229",
   >>>         "R385-Q229",
   >>>         "D381-K232",
   >>>         "Q384-I135"
   >>>         ]}}' > tip.json

  One added bonus is that the same .json files can be used file across different setups as long as the specified residues are present.

  The command::

   >>> mdc_sites.py prot.pdb traj.xtc --site tip.json -at -nf -sa #sa: short AA-names
   ...
   The following files have been created:
   ./sites.overall@3.5_Ang.pdf
   ...

  generates the following figure (tables are generated but not shown). The option ``-at`` (``--atomtypes``) generates the patterns ("hatching") of the bars. They indicate what atom types (sidechain or backbone) are responsible for the contact:

 .. figure:: imgs/sites.overall@3.5_Ang.Fig.6.png
      :scale: 50%
      :align: left
      :name: sites_freq

      [``sites.overall@3.5_Ang.pdf``](click to enlarge) Contact frequencies of the residue pairs specified in the file `tip.json`, shown with the contact type indicated by the stripes on the bars. Use e.g. the `3D-visualisation <http://proteinformatics.uni-leipzig.de/mdsrv.html?load=file://base/mdciao/gs-b2ar.ngl>`_ to check how "L394-K270" switches between SC-SC and SC-BB.

 |nbspc|
.. _comparison_HL:

* compare contact frequencies coming from different calculations, to detect and show contact changes across different systems. For example, to look for the effect of different ligands, mutations, pH-values etc. In this case, we compare the neighborhood of R131 (3.50 on the receptor) between our MD simulations and the crystal structure straight from the PDB. First, we grab the file on the fly with ``mdc_pdb.py``::

   >>> mdc_pdb.py 3SN6
   Checking https://files.rcsb.org/download/3SN6.pdb ...done
   Saving to 3SN6.pdb...done
   Please cite the following 3rd party publication:
    * Crystal structure of the beta2 adrenergic receptor-Gs protein complex
      Rasmussen, S.G. et al., Nature 2011
      https://doi.org/10.1038/nature10361

  Now we use ``mdc_neighborhoods.py`` on it::

   >>> mdc_neighborhoods.py 3SN6.pdb 3SN6.pdb -r R131 -o 3SN6 -co 4 -nf -o 3SN6.X
   ...
   #idx   freq      contact       fragments     res_idxs      ctc_idx  Sum
   1:     1.00   ARG131-TYR391       0-0        1007-345        14     1.00
   2:     1.00   ARG131-TYR326       0-0        1007-1174      111     2.00
   3:     1.00   ARG131-ILE278       0-0        1007-1126       97     3.00
   These 3 contacts capture 3.00 (~100%) of the total frequency 3.00 (over 120 contacts)
   ...
   The following files have been created:
   ...
   ./3SN6.X.ARG131@4.0_Ang.dat

  Now we use ``mdc_neighborhoods.py`` on our data::

   >>> mdc_neighborhoods.py prot.pdb traj.xtc -r R131 -co 4 -nf -o 3SN6.MD
   ...
   #idx   freq      contact       fragments     res_idxs      ctc_idx  Sum
   1:     0.87   ARG131-TYR391       0-0         861-350        12     0.87
   2:     0.69   ARG131-TYR326       0-0         861-1028      104     1.55
   3:     0.44   ARG131-TYR219       0-0         861-946        70     1.99
   4:     0.12   ARG131-ILE278       0-0         861-980        92     2.11
   These 4 contacts capture 2.11 (~99%) of the total frequency 2.11 (over 115 contacts)
   ...
   The following files have been created:
   ...
   ./3SN6.MD.ARG131@4.0_Ang.dat

 Please note that we have omitted most of the terminal output, and that we have used the option ``-o`` to label output-files differently: ``3SN6.X`` and ``3SN6.MD``. Now we compare both these outputs::

   >>> mdc_compare.py 3SN6.X.ARG131@4.0_Ang.dat 3SN6.MD.ARG131@4.0_Ang.dat -k Xray,MD -t "3SN6 cutoff 4AA" -a R131
   These interactions are not shared:
   Y219
   Their cumulative ctc freq is 0.44.
   Created files
   freq_comparison.pdf
   freq_comparison.xlsx


 .. figure:: imgs/freq_comparison.png
      :scale: 50%
      :align: left
      :name: comparisonfig

      [``freq_comparison.pdf``]Neighborhood comparison for R131 between our MD simulations and the original 3SN6 crystal structure. We can see how the neighborhood *relaxes* and changes.  Some close residues, in particular I278, move further than 4 Ang away from R131. Analogously, we see how Y219 also enters the neighborhood. You can see these residues highlighted in the `3D visualization`_. We have used a custom title and custom keys for clarity of the figure (options ``-t`` and ``-k``). Also, since all contact labels share the 'R131'  label, we can remove it with the ``-a`` (anchor residue).


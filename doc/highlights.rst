Highlights
----------
* paper-ready tables and figures from the command line

  .. figure:: imgs/bars_and_PDF.png
      :scale: 40%
      :align: left


      Left panel: most frequent neighbors of LEU394, the C-terminal residue in the :math:`\alpha_5` helix of the Gs-protein. A cutoff of 3.5 AA between heavy-atoms has been used. Residue labels combine residue names and consenus nomenclature. Right panel: associated distance distributions.


* easy input of target residues, e.g. the following is valid and will evaluate and show all these residues together::

  -r GLU*,GDP,L394,380-394

* different fragmentation heuristics to easily define regions of interest, either because the file lacks TER and CONNECT records (like a `.pdb <http://www.wwpdb .org/documentation/file-format-content/format33/v3.3.html>`_) or because they are not useful::

   Auto-detected fragments with method lig_resSeq+
   fragment      0 with  349 AAs            THR9(   0)- LEU394        (348 ) (0) resSeq jumps
   fragment      1 with  340 AAs            GLN1( 349)- ASN340        (688 ) (1)
   fragment      2 with  217 AAs            ASN5( 689)-ALA1160        (905 ) (2) resSeq jumps
   fragment      3 with  284 AAs           GLU30( 906)- CYS341        (1189) (3) resSeq jumps
   fragment      4 with  128 AAs            GLN1(1190)- SER128        (1317) (4)
   fragment      5 with    1 AAs         P0G1601(1318)-P0G1601        (1318) (5)

  In this example of the crystal structure `3SN6 <https://www.rcsb.org/structure/3SN6>`_, the chains are recovered even if they are missing from the `.gro <http://manual.gromacs.org/documentation/2020/reference-manual/file-formats.html#gro>`_ file.

* *automagically* map and incorporate consensus nomenclature like the `Ballesteros-Weinstein-Numbering <https://www.sciencedirect.com/science/article/pii/S1043947105800497>`_ (BW) or `Common G-alpha Numbering (CGN) <https://www.mrc-lmb.cam.ac.uk/CGN/faq.html>`_  to the analysis, either from local files or over the network in the `GPRC.db <https://gpcrdb.org/>`_ and from `<https://www.mrc-lmb.cam.ac.uk/CGN/>`_ ::

   No local file ./adrb2_human.xlsx found, checking online in
   https://gpcrdb.org/services/residues/extended/adrb2_human ...done!
    TM1 with   32 AAs      GLU30@1.29( 906)-  PHE61@1.60   (937 ) (TM1)
   ICL1 with    4 AAs     GLU62@12.48( 938)-  GLN65@12.51  (941 ) (ICL1)
    ...
    TM5 with   42 AAs     ASN196@5.35(1069)- GLU237@5.76   (1110) (TM5)
    TM6 with   35 AAs     CYS265@6.27(1113)- GLN299@6.61   (1147) (TM6)
    ...
    ...
   No local file ./CGN_3SN6.txt found, checking online in
   https://www.mrc-lmb.cam.ac.uk/CGN/lookup_results/3SN6.txt...done
   No local PDB file for 3SN6 found in directory ., checking online in
   https://files.rcsb.org/download/3SN6.pdb ...found!./mdas/mdas/notebooks/data/B2_model_imp/runfiles/confout1.gro
     G.HN with   28 AAs    THR9@G.HN.26(   5)-  VAL36@G.HN.53(32  ) (G.HN)
   G.hns1 with    3 AAs   TYR37@G.hns1.1(  33)-  ALA39@G.hns1.3(35  ) (G.hns1)
   ...
   G.s6h5 with    5 AAs  THR364@G.s6h5.1( 323)- ASP368@G.s6h5.5(327 ) (G.s6h5)
     G.H5 with   26 AAs   THR369@G.H5.1( 328)- LEU394@G.H5.26(353 ) (G.H5)

.. _`mdc_interface.py example`:

* use fragment definitions --like the ones above-- to compute interfaces in an automated way, i.e. without having to specifying individual residues::

    mdc_interface.py gs-b2ar.pdb gs-b2ar.xtc  -fg1 0-2 -fg2 3 --BW_un adrb2_human --CGN 3SN6 -t "3SN6 beta_2Ar-G_s interface"

 .. figure:: imgs/interface.matrix@3.5_Ang.Fig.4.png
      :scale: 25%
      :align: left

      (click to enlarge). Interface contact matrix between the B2AR receptor and the Gs protein, using a cutoff of 3.5 AA. The labelling incorporates consensus nomenclature to identify positions and domains of both receptor (BW) and G-protein (CGN). Please note: this is not a **symmetric** contact-matrix. The y-axis shows residues in the Gs protein and the x-axis in the receptor.

* Since **Fig. 4** is bound to incorporate a lot of blank pixels, ``mdciao`` will also produce sparse plots and figures that highlight the formed contacts only:

 .. figure:: imgs/interface.overall@3.5_Ang.Fig.5.png
      :scale: 15%
      :align: left
      :name: interface_bars


      (click to enlarge) Upper panel: most frequent contacts sorted by frequency. The lower panel aggreates and sorts the upper panel into per-residue frequencies, showing their average participation in the interface (same info will be written to `interface.overall@3.5_Ang.xlsx`). Also, the number of shown contacts can be controllod either with the `--n_ctcs` and/or `--min_freq` parameters of `mdc_interface.py`.

* A very convenient way to incorporate the molecular topology into the visualization of contact frequencies are the so-called `FlarePlots <https://github.com/GPCRviz/flareplot>`_ (cool live-demo `here <https://gpcrviz.github.io/flareplot/>`_). These show the molecular topology (residues, fragments) on a circle with curves connecting the residues for which a given frequency has been computed. The `mdc_interface.py example`_ above will also generate a flareplot:

 .. figure:: imgs/interface.flare@3.5_Ang.small.png
      :scale: 75%
      :align: left
      :name: fig_flare

      (click to enlarge) FlarePlot of the frequencies shown in the upper pannel of :numfig:`interface_bars`. Residues are shown as dots on the outer circumference, split into fragments following any available labelling (BW or CGN) information. The contact frequencies are represented as lines connecting residue pairs, with an opacity proportional to the frequencie's value. The secondary stucture of each residue is also included. (Note: This plot is always produced as .pdf to be able to zoom into it as much as needed. Click the zoomed-in inset at the top of the page to read the labels clearly)

* Similar to how :numfig:`fig_flare` maps contact frequencies (:numfig:`interface_bars`, upper panel) onto the molecular topology, the next figure maps the **lower** panel :numfig:`interface_bars` on the molecular geometry. It simply puts the values shown there in the `temperature factor <http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM>`_  of a pdb file, representing the calculated interface as a *heatmap*

 .. figure:: imgs/interface_BRG.png
      :scale: 50%
      :align: left

      (click to enlarge) 3D visualization of the interface as heatmap at hand using `VMD <https://www.ks.uiuc.edu/Research/vmd/>`_. Note, for the figure at the top of the page the option signal_beta=True has been passed to the cli.interface method of the API. At the moment this is not possible just by using `mdc_interface.py`, sorry!


* A different approach is to look **only** for a particular set of pre-defined contacts. Simply writing this set into a human readable `JSON <https://www.json.org/>`_ file will allow `mdc_sites.py` to compute and present these (and only these) contacts, as in the example file `tip.json`::

   {"sitename":"interesting contacts",
   "bonds": {"AAresSeq": [
               "L394-K270",
               "D381-Q229",
               "Q384-Q229",
               "R385-Q229",
               "D381-K232",
               "Q384-I135"
               ]}}


  One added bonus is that the same .json files can be used file across different setups as long as the specified residues are present.

  The command::

   mdc_sites.py gs-b2ar.pdb gs-b2ar.xtc --site_files tip.json -at

  generates the following figure (tables are generated but not shown). The option `-at` (`--atomtypes`) generates the patterns ("hatching") of the bars. They indicate how the atom types (sidechain or backbone) responsible for the contact:

 | `-`   is sidechain-sidechain (SC-SC),
 | `|`   is backbone-backbone(BB-BB),
 | `/`   is SC-SC, and
 | `\\`  is SC-BB

.. figure:: imgs/sites.overall@3.5_Ang.Fig.6.png
      :scale: 50%
      :align: left

      **Fig. 5** (click to enlarge) Contact frequencies of the residue pairs specified in the file `tip.json`, shown with the contact type indicated by the stripes on the bars. Use e.g. the `3D-visualisation <http://proteinformatics.charite.de/html/mdsrvdev.html?load=file://_Guille/gs-b2ar.ngl>`_ to check how "L394-K270" switches between SC-SC and SC-BB.

* compare contact frequencies coming from different calculations

* compare, detect, and show frequency differences across different systems, e.g. to look for the effect of mutations, pH-differences etc
* TODO expand



Highlights
----------

.. _`initial example`:

* paper-ready tables and figures from the command line::

   mdc_neighborhoods.py prot.pdb traj.xtc -r L394 --BW adrb2_human --CGN 3SN6 -ni -at #ni: not interactive, at: show atom-types

  .. figure:: imgs/bars_and_PDF.png
      :scale: 40%
      :align: left
      :name: highlights_1

      (click to enlarge) **a)** most frequent neighbors of LEU394, the C-terminal residue in the :math:`\alpha_5` helix of the Gs-protein. A cutoff of 3.5 AA between heavy-atoms has been used. Residue labels combine residue names and consenus nomenclature. **b)** associated distance distributions, obtained by adding the ``-d`` flag to the CLI call. **c)** Automatically generated table using the ``-tx xlsx`` option.

* easy input of target residues, e.g. the following is valid and will evaluate and show all these residues together::

  -r GLU*,GDP,L394,380-394

* fragmentation heuristics to easily identify molecules and/or molecular fragments. These heuristics will work on .pdf-files lacking `TER and CONNECT records <http://www.wwpdb .org/documentation/file-format-content/format33/v3.3.html>`_ or other file formats, like `.gro files <http://manual.gromacs.org/documentation/2020/reference-manual/file-formats.html#gro>`_, that simply don't include these records::

   Auto-detected fragments with method lig_resSeq+
   fragment      0 with  349 AAs            THR9(   0)- LEU394        (348 ) (0) resSeq jumps
   fragment      1 with  340 AAs            GLN1( 349)- ASN340        (688 ) (1)
   fragment      2 with  217 AAs            ASN5( 689)-ALA1160        (905 ) (2) resSeq jumps
   fragment      3 with  284 AAs           GLU30( 906)- CYS341        (1189) (3) resSeq jumps
   fragment      4 with  128 AAs            GLN1(1190)- SER128        (1317) (4)
   fragment      5 with    1 AAs         P0G1601(1318)-P0G1601        (1318) (5)

  In this example, we saved the crystal structure `3SN6 <https://www.rcsb.org/structure/3SN6>`_ as a .gro-file, and were able to recover the chains: :math:`G\alpha`, :math:`G\beta`, :math:`G\gamma`, :math:`\beta 2AR`, antibody, and ligand.  For clarity, we omitted the fragmentation in our `initial example`_ with the option ``-nf``, but all CLI tools do this fragmentation by default. Alternatively, one can use::

   mdc_fragments.py prot.pdb

  to produce an overview of all available fragmentation heuristics and their results without computing any contacts whatsoever.

* *automagically* map and incorporate consensus nomenclature like the `Ballesteros-Weinstein-Numbering <https://www.sciencedirect.com/science/article/pii/S1043947105800497>`_ (BW) or `Common G-alpha Numbering (CGN) <https://www.mrc-lmb.cam.ac.uk/CGN/faq.html>`_  to the analysis, either from local files or over the network in the `GPRC.db <https://gpcrdb.org/>`_ and from `<https://www.mrc-lmb.cam.ac.uk/CGN/>`_::

   ...
   Using BW-nomenclature, please cite the following 3rd party publications:
    * https://doi.org/10.1016/S1043-9471(05)80049-7 (Weinstein et al 1995)
    * https://doi.org/10.1093/nar/gkx1109 (Gloriam et al 2018)
   No local file ./adrb2_human.xlsx found, checking online in
   https://gpcrdb.org/services/residues/extended/adrb2_human ...done!
   done without 404, continuing.
   BW-labels align best with fragments: [3] (first-last: GLU30-LEU340).
   These are the BW fragments mapped onto your topology:
    N-term with   19 AAs     GLY5           (   1) -   THR321           (674 ) (N-term)  resSeq jumps
       TM1 with   35 AAs    GLN11@1.25      ( 703) -    PHE61@1.60      (791 ) (TM1)  resSeq jumps
      ICL1 with    4 AAs    GLU62@12.48     ( 792) -    GLN65@12.51     (795 ) (ICL1)
       TM2 with   32 AAs    THR66@2.37      ( 796) -    LYS97@2.68      (827 ) (TM2)
      ECL1 with    4 AAs    MET98@23.49     ( 828) -   PHE101@23.52     (831 ) (ECL1)
       TM3 with   36 AAs   GLY102@3.21      ( 832) -   SER137@3.56      (867 ) (TM3)
      ICL2 with    8 AAs   PRO138@34.50     ( 868) -   LEU145@34.57     (875 ) (ICL2)
       TM4 with   27 AAs   THR146@4.38      ( 876) -   HIS172@4.64      (902 ) (TM4)
      ECL2 with   20 AAs   TRP173           ( 903) -   THR195           (922 ) (ECL2)  resSeq jumps
       TM5 with   42 AAs   ASN196@5.35      ( 923) -   GLU237@5.76      (964 ) (TM5)
      ICL3 with    2 AAs   GLY238           ( 965) -   ARG239           (966 ) (ICL3)
       TM6 with   35 AAs   CYS265@6.27      ( 967) -   GLN299@6.61      (1001) (TM6)
      ECL3 with    4 AAs   ASP300           (1002) -   ILE303           (1005) (ECL3)
       TM7 with   25 AAs   ARG304@7.31      (1006) -   ARG328@7.55      (1030) (TM7)
        H8 with   11 AAs   SER329@8.47      (1031) -   LEU339@8.57      (1041) (H8)
    C-term with    1 AAs   LEU340           (1042) -   LEU340           (1042) (C-term)
   ...
   ...
   Using CGN-nomenclature, please cite the following 3rd party publications:
    * https://doi.org/10.1038/nature14663 (Babu et al 2015)
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



.. _`mdc_interface.py example`:

* use fragment definitions --like the ones above, 0 for the :math:`G\alpha`-unit and 3 for the receptor-- to compute interfaces in an automated way, i.e. without having to specifying individual residues::

   mdc_interface.py prot.pdb traj.xtc -fg1 0 -fg2 3 --BW adrb2_human --CGN 3SN6 -t "3SN6 beta2AR-Galpha interface" -ni

  This gives an (edited) output::

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
   The following files have been created
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

      (click to enlarge). Interface contact matrix between the :math:`\beta`2AR receptor and the :math:`G\alpha`-unit protein, using a cutoff of 3.5 AA. The labelling incorporates consensus nomenclature to identify positions and domains of both receptor (BW) and G-protein (CGN). Please note: this is **not a symmetric** contact-matrix. The y-axis shows residues in the :math:`G\alpha`-unit and the x-axis in the receptor.

* Since :numref:`interface_matrix` is bound to incorporate a lot of blank pixels, ``mdciao`` will also produce sparse plots and figures that highlight the formed contacts only:

 .. figure:: imgs/interface.overall@3.5_Ang.Fig.5.png
      :scale: 15%
      :align: left
      :name: interface_bars


      (click to enlarge) **Upper panel**: most frequent contacts sorted by frequency, i.e. for each non-empty pixel of :numref:`interface_matrix`, there is a bar shown. **Lower panel**: per-residue aggregated contact-frequencies, showing each residue's average participation in the interface (same info will be written to `interface.overall@3.5_Ang.xlsx`). Also, the number of shown contacts/bars can be controlled either with the `--ctc_control` and/or `--min_freq` parameters of `mdc_interface.py`.

* A very convenient way to incorporate the molecular topology into the visualization of contact frequencies are the so-called `FlarePlots <https://github.com/GPCRviz/flareplot>`_ (cool live-demo `here <https://gpcrviz.github.io/flareplot/>`_). These show the molecular topology (residues, fragments) on a circle with curves connecting the residues for which a given frequency has been computed. The `mdc_interface.py example`_ above will also generate a flareplot:

 .. figure:: imgs/interface.flare@3.5_Ang.small.png
      :scale: 70%
      :align: left
      :name: fig_flare

      (click to enlarge) FlarePlot of the frequencies shown in the figures :numref:`interface_matrix` and :numref:`interface_bars`. Residues are shown as dots on a circumference, split into fragments following any available labelling (BW or CGN) information. The contact frequencies are represented as lines connecting these dots/residues, with the line-opacity proportional to the frequencie's value. The secondary stucture of each residue is also included as color-coded letters: H(elix), B(eta), C(oil). We can clearly see the :math:`G\alpha_5`-subunit in contact with the receptor's TM3, ICL2, and TM5-ICL3-TM6 regions. Note that this plot is always produced as .pdf to be able to zoom into it as much as needed.

* Similar to how the flareplot (:numref:`fig_flare`) is mapping contact-frequencies (:numref:`interface_bars`, upper panel) onto the molecular topology, the next figure maps the **lower** panel :numref:`interface_bars` on the molecular geometry. It simply puts the values shown there in the `temperature factor <http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM>`_  of a pdb file, representing the calculated interface as a *heatmap*

 .. figure:: imgs/interface_BRG.png
      :scale: 70%
      :align: left

      (click to enlarge) 3D visualization of the interface as heatmap (blue-green-red) using `VMD <https://www.ks.uiuc.edu/Research/vmd/>`_. We clearly see the regions noted in :numref:`fig_flare` (TM5-ICL3-TM6 and :math:`G\alpha_5`-subunit) in particular the **residues** of :numref:`interface_bars` (lower panel) light up. Please note that for the homepage-banner (red-blue heatmap), the ``signed_colors`` argument has been used when calling the :obj:`mdciao.flare.freqs2flare` method of the API. At the moment this is not possible just by using ``mdc_interface.py``, sorry!


* A different approach is to look **only** for a particular set of pre-defined contacts. Simply writing this set into a human readable `JSON <https://www.json.org/>`_ file will allow `mdc_sites.py` to compute and present these (and only these) contacts, as in the example file `tip.json`::


   cat tip.json
   {"name":"interface small",
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

   mdc_sites.py prot.pdb traj.xtc --site tip.json -at -nf -sa #sa: short AA-names

  generates the following figure (tables are generated but not shown). The option ``-at`` (``--atomtypes``) generates the patterns ("hatching") of the bars. They indicate what atom types (sidechain or backbone) are responsible for the contact:

.. figure:: imgs/sites.overall@3.5_Ang.Fig.6.png
      :scale: 50%
      :align: left

      (click to enlarge) Contact frequencies of the residue pairs specified in the file `tip.json`, shown with the contact type indicated by the stripes on the bars. Use e.g. the `3D-visualisation <http://proteinformatics.charite.de/html/mdsrvdev.html?load=file://_Guille/gs-b2ar.ngl>`_ to check how "L394-K270" switches between SC-SC and SC-BB.

* compare contact frequencies coming from different calculations, to detect and show contact changes across different systems, e.g. to look for the effect of different ligands, mutations, pH-values etc. In this case, we compare R131@3.50 in an active and inactive :math:`\beta2 AR`. First, we grab the necessary structrures on the fly with ``mdc_pdb.py``::

   >>> mdc_pdb.py 3SN6
   Checking https://files.rcsb.org/download//3SN6.pdb ...done
   Saving to 3SN6.pdb...done
   Please cite the following 3rd party publication:
    * Crystal structure of the beta2 adrenergic receptor-Gs protein complex
      Rasmussen, S.G. et al., Nature 2011
      https://doi.org/10.1038/nature10361

   >>> mdc_pdb.py 6OBA
   Checking https://files.rcsb.org/download//6OBA.pdb ...done
   Saving to 6OBA.pdb...done
   Please cite the following 3rd party publication:
    * An allosteric modulator binds to a conformational hub in the beta2adrenergic receptor.
      Liu, X. et al., Nat Chem Biol 2020
      https://doi.org/10.1038/s41589-020-0549-2

  Now we use ``mdc_neighborhoods.py`` on both downloaded files::

   >>> mdc_neighborhoods.py 3SN6.pdb 3SN6.pdb -r R131 -o 3SN6 -tx dat -nf
   ...
   #idx   freq      contact       fragments     res_idxs      ctc_idx  Sum
   1:     1.00   ARG131-ILE278       0-0        1007-1126       97     1.00
   ...
   The following files have been created
   ...
   ./3SN6.ARG131@3.5_Ang.dat

   >>> mdc_neighborhoods.py 6OBA.pdb 6OBA.pdb -r R131 -o 6OBA -tx dat -nf
   ...
   #idx   freq      contact       fragments     res_idxs      ctc_idx  Sum
   1:     1.00   ARG131-TYR141       0-0         100-110        28     1.00
   2:     1.00   ARG131-SER143       0-0         100-112        30     2.00
   3:     1.00   ARG131-THR68        0-0         100-37          7     3.00
   ...
   The following files have been created
   ...
   ./6OBA.ARG131@3.5_Ang.dat

  Please note that we have omitted most of the terminal output, and that we have used the option ``-o`` to label output-files differently. Now we can simply compare these output files:



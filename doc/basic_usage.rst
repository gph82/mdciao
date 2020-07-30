.. _minimal_example:

Basic Usage
-----------

.. toctree::
   :maxdepth: 2

.. _`3D visualization`:

.. note::
   The simulation data for generating these examples was kindly provided by Dr. H. Batebi. It can be 3D-visualized interactively `here <http://proteinformatics.charite.de/html/mdsrvdev.html?load=file://_Guille/gs-b2ar.ngl>`_ while checking out the examples.

This command::

 mdc_neighborhoods.py p2.noH.pdb run1.1-p.stride.5.noH.xtc --residues L394 -nf


will print the following to the terminal (some headers have been left out)::

 ...
 #idx   freq      contact       fragments     res_idxs      ctc_idx  Sum
 1:     0.55   LEU394-ARG389       0-0         353-348        33     0.55
 2:     0.47   LEU394-LYS270       0-0         353-972        71     1.02
 3:     0.38   LEU394-LEU388       0-0         353-347        32     1.39
 4:     0.23   LEU394-LEU230       0-0         353-957        56     1.62
 5:     0.10   LEU394-ARG385       0-0         353-344        29     1.73
 These 5 contacts capture 1.7 of the total frequency 1.8 (over 87 contacts). 4 ctcs already capture 90.0% of 1.8.
 The following files have been created
 ./neighborhoods.overall@3.5_Ang.pdf
 ./neighborhoods.LEU394.time_trace@3.5_Ang.pdf
 ./neighborhoods.LEU394.gs-b2ar.dat

And produce the following figures (not the captions):

.. figure:: imgs/neighborhoods.overall@3.5_Ang.Fig.1.png
   :scale: 50%

   Using 3.5 AA as distance cutoff, the most frequent neighbors of LEU394, the C-terminal residue in the alpha5 helix of the Gs-protein are shown. The stimulation started from the `3SN6` structure (including the B2AR receptor). The stimulation itself can be seen interactively in the link posted in the `3D visualization`_.

Annotated figures with the timetraces of the above distances are also produced automatically:

.. figure:: imgs/neighborhoods.LEU394.time_trace@3.5_Ang.Fig.2.png
   :scale: 33%
   :align: center

   Time-traces of the residue-residue distances behind the frequency barplots of Fig. 1. The last time-trace represents the total number of neighbors (distances below the given cutoff) at any given moment in the trajectory. On average, LEU394 has around 1.7 non-bonded neighbors below the cutoff (see legend of Fig.1)

Anything that gets shown in any way to the output can be saved for later use as human readable ASCII-files, Excel-tables or NumPy `.npy` files for later use.
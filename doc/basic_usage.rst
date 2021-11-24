.. _`Basic-Usage`:

Basic Usage
-----------

.. _`3D visualization`:

Below you will find a very simple example of how to use ``mdciao`` from the command-line. Keep scrolling to the :ref:`Highlights` for more elaborate CLI-examples or jump to the :ref:`Jupyter Notebook <Jupyter Notebook Tutorial>` for a minimal API walkthrough. More notebooks fill follow soon.

.. admonition:: Data and 3D visualization

    The simulation data for generating these examples was kindly provided by Dr. H. Batebi. It can be 3D-visualized interactively `here <http://proteinformatics.uni-leipzig.de/mdsrv.html?load=file://base/mdciao/gs-b2ar.ngl>`_ while checking out the examples. You can also download `mdciao_example.zip <http://proteinformatics.org/mdciao/mdciao_example.zip>`_ and follow along.

.. admonition:: Copying Terminal Commands

   All code snippets can be copied and pasted directly into your terminal using the `copy icon <https://sphinx-copybutton.readthedocs.io>`_ in the upper right of the snippet's frame:

   >>> Click the button at end of the frame to copy this text ;)

This basic command::

 mdc_neighborhoods.py prot.pdb traj.xtc --residues L394 -nf #nf: don't use fragments


will print the following to the terminal (some headers have been left out)::

 ...
 #idx   freq      contact       fragments     res_idxs      ctc_idx  Sum
 1:     0.55   LEU394-ARG389       0-0         353-348        33     0.55
 2:     0.47   LEU394-LYS270       0-0         353-972        71     1.02
 3:     0.38   LEU394-LEU388       0-0         353-347        32     1.39
 4:     0.23   LEU394-LEU230       0-0         353-957        56     1.62
 5:     0.10   LEU394-ARG385       0-0         353-344        29     1.73
 These 5 contacts capture 1.73 (~97%) of the total frequency 1.76 (over 87 contacts)
 As orientation value, 4 ctcs already capture 90.0% of 1.76.
 The 4-th contact has a frequency of 0.23

 The following files have been created:
 ./neighborhood.overall@3.5_Ang.pdf
 ./neighborhood.LEU394@3.5_Ang.dat
 ./neighborhood.LEU394.time_trace@3.5_Ang.pdf

produce the following figures (not the captions):

.. figure:: imgs/neighborhoods.overall@3.5_Ang.Fig.1.png
   :scale: 50%
   :name: freqs

   [``neighborhood.overall@3.5_Ang.png``] Using 3.5 AA as distance cutoff, the most frequent neighbors of LEU394, the C-terminal residue in the :math:`\alpha_5` helix of the Gs-protein, are shown. :math:`\Sigma` is the sum over frequencies and represents the average number of neighbors of LEU394. The simulation started from the `3SN6 structure <https://www.rcsb.org/structure/3SN6>`_ (beta2 adrenergic receptor-Gs protein complex, no antibody). The simulation itself can be seen interactively `in 3D here <http://proteinformatics.uni-leipzig.de/mdsrv.html?load=file://base/mdciao/gs-b2ar.ngl>`_.

Annotated figures with the timetraces of the above distances are also produced automatically:

.. figure:: imgs/neighborhoods.LEU394.time_trace@3.5_Ang.Fig.2.png
   :scale: 33%
   :align: center

   [``neighborhoods.LEU394.time_trace@3.5_Ang.png``] Time-traces of the residue-residue distances used for the frequencies in :numref:`freqs`. The last time-trace represents the total number of neighbors (:math:`\Sigma`) within the given cutoff at any given moment in the trajectory. On average, LEU394 has around 1.7 non-bonded neighbors below the cutoff (see legend of :numref:`freqs`).

Anything that gets shown in any way to the output can be saved for later use as human readable ASCII-files (``.dat,.txt``), spreadsheets (``.ods,.xlsx``) or NumPy (``.npy``) files.
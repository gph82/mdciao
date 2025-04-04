.. _`Basic-Usage`:

Basic Usage
-----------

.. _`3D visualization`:

Below you will find a very simple example of how to use ``mdciao`` from the command-line. Keep scrolling to the :ref:`Highlights` for more elaborate CLI-examples or jump to the :ref:`Jupyter Notebook <Jupyter Notebook Tutorial>` for a minimal API walkthrough. For real usecase and some context on definitions etc, go to the :ref:`Jupyter Notebook Gallery`.

.. admonition:: Data and 3D visualization

    The simulation data for generating these examples was kindly provided by Dr. H. Batebi. It can be 3D-visualized interactively `here <http://proteinformatics.uni-leipzig.de/mdsrv.html?load=file://base/mdciao/gs-b2ar.ngl>`_ while checking out the examples. You can also download `mdciao_example.zip <http://proteinformatics.org/mdciao/mdciao_example.zip>`_ and follow along.

.. admonition:: Copying Terminal Commands

   All code snippets can be copied and pasted directly into your terminal using the `copy icon <https://sphinx-copybutton.readthedocs.io>`_ in the upper right of the snippet's frame:

   >>> Click the button at end of the frame to copy this text ;)

This basic command::

 mdc_neighborhoods.py top.pdb traj.xtc --residues L394 -nf #nf: don't use fragments


will print the following to the terminal (some headers have been left out)::

 ...
 The following 6 contacts capture 5.26 (~97%) of the total frequency 5.43 (over 9 contacts with nonzero frequency at 4.50 Angstrom).
 [...]
    freq          label            residues  fragments   sum
 1  1.00  L394@frag0 - L388@frag0  353 - 347    0 - 0   1.00
 2  1.00  L394@frag0 - R389@frag0  353 - 348    0 - 0   2.00
 3  0.97  L394@frag0 - L230@frag3  353 - 957    0 - 3   2.97
 4  0.97  L394@frag0 - R385@frag0  353 - 344    0 - 0   3.94
 5  0.80  L394@frag0 - I233@frag3  353 - 960    0 - 3   4.74
 6  0.52  L394@frag0 - K270@frag3  353 - 972    0 - 3   5.26
 The following files have been created:
 ./neighborhood.overall@4.5_Ang.pdf
 ./neighborhood.LEU394@frag0@4.5_Ang.dat
 ./neighborhood.LEU394@frag0.time_trace@4.5_Ang.pdf

produce the following figures (not the captions):

.. figure:: imgs/neighborhood.overall@4.5_Ang.Fig.1.png
   :scale: 50%
   :name: freqs

   [``neighborhood.overall@4.5_Ang.pdf``] Using 4.5 Å as distance cutoff, the most frequent neighbors of LEU394, the C-terminal residue in the :math:`\alpha_5` helix of the Gs-protein, are shown. :math:`\Sigma` is the sum over frequencies and represents the average number of neighbors of LEU394. The simulation started from the `3SN6 structure <https://www.rcsb.org/structure/3SN6>`_ (beta2 adrenergic receptor-Gs protein complex, no antibody). The simulation itself can be seen interactively `in 3D here <http://proteinformatics.uni-leipzig.de/mdsrv.html?load=file://base/mdciao/gs-b2ar.ngl>`_.

Annotated figures with the timetraces of the above distances are also produced automatically:

.. figure:: imgs/neighborhood.LEU394.time_trace@4.5_Ang.Fig.2.png
   :scale: 33%
   :align: center

   [``neighborhood.LEU394.time_trace@4.5_Ang.pdf``] Time-traces of the residue-residue distances used for the frequencies in :numref:`freqs`. The last time-trace represents the total number of neighbors (:math:`\Sigma`) within the given cutoff at any given moment in the trajectory. On average, LEU394 has around 4 non-bonded neighbors below the cutoff (see legend of :numref:`freqs`).

Anything that gets shown in any way to the output can be saved for later use as human readable ASCII-files (``.dat,.txt``), spreadsheets (``.ods,.xlsx``) or NumPy (``.npy``) files.
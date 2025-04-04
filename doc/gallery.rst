.. _gallery:

Jupyter Notebook Gallery
========================

.. |br| raw:: html

   </br>

The notebooks can be accessed locally by issuing::

 mdc_notebooks.py

from the CLI. This will create a local "sandboxed" copy of the notebooks,
which you can modify and play around with without breaking
the original notebooks. Note: the Covid notebooks are not shipped with `mdciao`.

Tutorials
---------

.. list-table::

    * - .. figure:: _build/html/_images/interface.combined.png
           :target: notebooks/01.Tutorial.html
           :height: 100px

           ..

           |br| `One single notebook providing <notebooks/01.Tutorial.html>`_
           |br| `an overview of mdciao <notebooks/01.Tutorial.html>`_

FAQs
----
These are issue-specific notebooks. They are more verbose and explore
how particular optional parameters affect the output of some methods.

.. For the thumbnail image, It's not trivial to
.. predict how nbsphinx will name .png-files from
.. the notebooks, s.t. the hard-links to the notebook-generated
.. images are flaky and hard to maintain. We have opted for a generic
.. hard-link here (notebook_name_selected_thumbnail.png) and a method
.. in docs/conf.py that uses a mapping to a notebook's figures via
.. sequential zero indexing, called rename_thumbnails. See there
.. for more details

.. list-table::

    * - .. figure:: _build/doctrees/nbsphinx/notebooks_02.Missing_Contacts_selected_thumbnail.png
           :target: notebooks/02.Missing_Contacts.html
           :height: 100px

           ..

           |br| `Missing Contacts <notebooks/02.Missing_Contacts.html>`_

      - .. figure:: _build/doctrees/nbsphinx/notebooks_03.Comparing_CGs_Bars_selected_thumbnail.png
           :target: notebooks/03.Comparing_CGs_Bars.html
           :height: 100px

           ..

           |br| `Comparing Frequencies: <notebooks/03.Comparing_CGs_Bars.html>`_
           |br| `Bar Plots <notebooks/03.Comparing_CGs_Bars.html>`_

    * - .. figure:: _build/doctrees/nbsphinx/notebooks_05.Flareplot_Schemes_selected_thumbnail.png
           :target: notebooks/05.Flareplot_Schemes.html
           :height: 100px

           ..

           |br| `Controlling Flareplots: <notebooks/05.Flareplot_Schemes.html>`_
           |br| `Schemes <notebooks/05.Flareplot_Schemes.html>`_

      - .. figure:: _build/doctrees/nbsphinx/notebooks_04.Comparing_CGs_Flares_selected_thumbnail.png
           :target: notebooks/04.Comparing_CGs_Flares.html
           :height: 100px

           ..

           |br| `Comparing Frequencies: <notebooks/04.Comparing_CGs_Flares.html>`_
           |br| `Flareplots <notebooks/04.Comparing_CGs_Flares.html>`_

Examples
--------
These are worked out examples from the real world, from raw data to final results.
They are the best starting point to copy and modify with your own data.

.. list-table::

    * - .. figure:: _build/doctrees/nbsphinx/notebooks_08.Manuscript_selected_thumbnail.png
           :target: notebooks/08.Manuscript.html
           :height: 100px

           ..

           |br| `Interfaces: <notebooks/08.Manuscript.html>`_
           |br| `β2 Adrenergic Receptor in Complex with <notebooks/08.Manuscript.html>`_
           |br| `Empty Gs-Protein <notebooks/08.Manuscript.html>`_

      - .. figure:: _build/doctrees/nbsphinx/notebooks_07.EGFR_Kinase_Inhibitors_selected_thumbnail.png
           :target: notebooks/07.EGFR_Kinase_Inhibitors.html
           :height: 100px

           ..

           |br| `Binding-Pockets: <notebooks/07.EGFR_Kinase_Inhibitors.html>`_
           |br| `EGFR Kinase Inhibitors <notebooks/07.EGFR_Kinase_Inhibitors.html>`_

    * - .. figure:: _build/doctrees/nbsphinx/notebooks_Covid-19-Spike-Protein-Example_23_1.png
           :target: notebooks/Covid-19-Spike-Protein-Example.html
           :height: 100px

           ..

           |br| `Covid-19 Spike Protein <notebooks/Covid-19-Spike-Protein-Example.html>`_
           |br| `Example 1: Residue Neighborhoods <notebooks/Covid-19-Spike-Protein-Example.html>`_

      - .. figure:: imgs/spike_intf.small.png
           :target: notebooks/Covid-19-Spike-Protein-Interface.html
           :height: 100px

           ..

           |br| `Covid-19 Spike Protein <notebooks/Covid-19-Spike-Protein-Interface.html>`_
           |br| `Example 2: Molecular Interfaces <notebooks/Covid-19-Spike-Protein-Interface.html>`_

    * - .. figure:: imgs/MSA_via_Consensus_Labels.png
           :target: notebooks/06.MSA_via_Consensus_Labels.html
           :height: 100px

           ..

           |br| `3D Multiple Sequence Alignment via <notebooks/06.MSA_via_Consensus_Labels.html>`_
           |br| `Consensus Labels on μ-Opioid Receptor, <notebooks/06.MSA_via_Consensus_Labels.html>`_
           |br| `β2 Adregneric Receptor, Opsin, and <notebooks/06.MSA_via_Consensus_Labels.html>`_
           |br| `Dopamine D1 Receptor <notebooks/06.MSA_via_Consensus_Labels.html>`_

      - .. figure:: _build/doctrees/nbsphinx/notebooks_09.Consensus_Labels_selected_thumbnail.png
           :target: notebooks/09.Consensus_Labels.html
           :height: 100px

           ..

           |br| `Contact Frequencies  <notebooks/09.Consensus_Labels.html>`_
           |br| `for multiple systems <notebooks/09.Consensus_Labels.html>`_
           |br| `via consensus labels <notebooks/09.Consensus_Labels.html>`_

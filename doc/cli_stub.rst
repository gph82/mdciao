Command Line Tools
------------------

The best way to find out about all of ``mdciao``'s command-line tools is to use a command-line tool shipped with ``mdciao``::

 >>> mdc_examples.py
 usage: mdc_examples.py [-h] [-x] [--short_options] clt

 Wrapper script to showcase and optionally run examples of the
 command-line-tools that ship with mdciao.

 Available command line tools are:
  * mdc_GPCR_overview.py
  * mdc_CGN_overview.py
  * mdc_compare.py
  * mdc_fragments.py
  * mdc_interface.py
  * mdc_neighborhoods.py
  * mdc_pdb.py
  * mdc_residues.py
  * mdc_sites.py

 You can type for example:
  > mdc_interface.py  -h
  to view the command's documentation or
  > mdc_examples.py interf
  to show and/or run an example of that command


What these tools do is:

* mdc_neighborhoods
   Analyse residue neighborhoods using a distance cutoff. Example in :numref:`highlights_1`.
* mdc_interface
   Analyse interfaces between any two groups of residues using a distance cutoff. Example :ref:`here <mdc_interface.py example>` with :numref:`interface_matrix`, :numref:`interface_bars`, and :numref:`fig_flare`, among others.
* mdc_sites
   Analyse a specific set of residue-residue contacts using a distance cutoff. Example in :numref:`sites_freq`.
* mdc_fragments
   Break a molecular topology into fragments using different heuristics. Example :ref:`here <fragmentation_HL>`.
* mdc_GPCR_overview
   Map a consensus GPCR nomenclature (e.g. Ballesteros-Weinstein and others) nomenclature on an input topology. Example :ref:`here <consensus_HL>`.
* mdc_CGN_overview
   Map a Common G-alpha Numbering (CGN)-type nomenclature on an input topology Example :ref:`here <consensus_HL>`.
* mdc_compare
   Compare residue-residue contact frequencies from different files. Example :ref:`here <comparison_HL>`  with :numref:`comparisonfig`.
* mdc_pdb
   Lookup a four-letter PDB-code in the RCSB PDB and save it locally. Example :ref:`here <pdb_HL>`.
* mdc_residues
    Find residues in an input topology using Unix filename pattern matching. Example :ref:`here <residues_HL>`.

You can see their documentation by using the ``-h`` flag when invoking them from the command line, keep reading the ref:`Highlights` or the :ref:`CLI Reference`.
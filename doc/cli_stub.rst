Command line tools (quick intro)
--------------------------------

The best way to find out about ``mdciao``'s command-line tools is to use a command-line tool shipped with ``mdciao``::

 mdc_examples.py ?

 Wrapper script to showcase and optionally run examples of the
 command-line-tools that ship with mdciao.
 Availble command line tools are
  * mdc_BW_overview.py
  * mdc_CGN_overview.py
  * mdc_compare.py
  * mdc_fragments.py
  * mdc_interface.py
  * mdc_neighborhoods.py
  * mdc_sites.py
 Issue:
  * 'mdc_command.py -h' to view the command's documentation or
  * 'mdc_examples.py mdc_command' to show and/or run an example of that command


What these tools do is:

* mdc_neighborhoods
   Analyse residue neighborhoods using a distance cutoff
* mdc_interface
   Analyse interfaces between any two groups of residues using a distance cutoff
* mdc_sites
   Analyse a specific set of residue-residue contacts using a distance cutoff
* mdc_fragments
   Break a molecular topology into fragments using different heuristics.
* mdc_BW_overview
   Map a Ballesteros-Weinstein (BW)-type nomenclature on an input topology.
* mdc_CGN_overview
   Map a Common G-alpha Numbering (CGN)-type nomenclature on an input topology
* mdc_compare
   Compare residue-residue contact frequencies from different files

You can see their documentation by using the `-h` flag whe invoking them from the command line or by checking the help menu to your left.
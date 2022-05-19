.. _commandlineinterface:

CLI Reference
=============
We refer to the command-line-tools collectively as the command-line-interface, or CLI. Their options go well beyond those shown in the :ref:`Basic-Usage`, as we saw in the :ref:`Highlights`. Use these pages (menu to the left) or the ``mdc_command.py -h`` syntax to get help. For an inline overview of all options and ready-to-use examples, use::

 >>> mdc_examples.py
 usage: mdc_examples.py [-h] [-x] [--short_options] clt

 Wrapper script to showcase and optionally run examples of the
 command-line-tools that ship with mdciao.

 Available command line tools are:
  * mdc_GPCR_overview.py
  * mdc_CGN_overview.py
  * mdc_KLIFS_overview.py
  * mdc_compare.py
  * mdc_fragments.py
  * mdc_interface.py
  * mdc_neighborhoods.py
  * mdc_notebooks.py
  * mdc_pdb.py
  * mdc_residues.py
  * mdc_sites.py

 You can type for example:
  > mdc_interface.py  -h
  to view the command's documentation or
  > mdc_examples.py interf
  to show and/or run an example of that command


.. toctree::
   :maxdepth: 2
   :hidden:

   mdc_neighborhoods
   mdc_interface
   mdc_sites
   mdc_fragments
   mdc_compare
   mdc_GPCR_overview
   mdc_CGN_overview
   mdc_KLIFS_overview
   mdc_residues
   mdc_pdb
   mdc_notebooks
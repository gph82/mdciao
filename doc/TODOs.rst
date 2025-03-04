.. _TODOs:

TODOs
=====

This is an informal list of known issues and TODOs:
 * adopt this project structure https://github.com/MolSSI/cookiecutter-cms
 * keeping vs reporting contacts: a design choice has to be made wrt to the effect of ctc_cutoff_Ang on a ContactGroup:
   If a given cutoff makes a ContactPair have freq=0, should the CP be kept in the ConctactGroup, simply not reported? The `max_cutoff_Ang` is already in place s.t. you can have a buffer of some Angstrom, but then the ContactGroup.n_ctcs would be hard to interpret.
 * overhaul the "printing" system with proper logging and warnings (perhaps use `loguru <https://github.com/Delgan/loguru>`_)
 * the affiliation of a residue to a fragment is done as "res@frag" on the string output and res^frag in figures, this implementation is simply using replace("@","^"), could be better
 * harmonize documentation API cli methods (mdciao.cli) and the CLI scripts (mdc_*)
 * The interface between API methods and cli scripts could be better, using sth like `click <https://click.palletsprojects.com/en/7.x/>`_
 * The API-cli methods (interface, neighborhoods, sites, etc) have very similar flows, and although a lot of effort has been put into refactoring into smaller methods, there's still some repetition.
 * Most of the tests were written against a very rigid API that mimicked the CLI closely. Now the API is more flexible
   and many `tests could be re-written or deleted <https://en.wikipedia.org/wiki/Technical_debt>`_ , like those needing
   mock-input or writing to tempdirs because writing figures or files could not be avoided.
 * There's some inconsistencies in private vs public attributes of classes. An attribute might've "started" as private and is exceptionally used somewhere else until the number of exceptions is enough for it to make sense to be public, documented and well tested.
 * The labelling names should be harmonized (ctc_label, anchor_res...) and the logic of how/where it gets constructed (short_AA vs AA_format) is not obvious sometimes
 * The way uniprot or PDB codes are transformed to relative and/or absolute filenames to check if they exist locally should be unified across all lookup functions, like GPCR_finder, PDB_finder and/or the different LabelerConsensus objects, possibly by dropping optargs like 'local_path' or 'format'.
 * Some closely related methods could/should be integrated into each other by generalising a bit, but sometimes the generalisation is unnecessarily complicated to code (and test!) for a slightly different scenario (though we try to hard to avoid it). E.g. there's several methods for computing, reporting, and saving contact frequencies and contact-matrices, or different methods to assign residue idxs to fragments, `find_parent_list, `in_what_N_fragments`, or `assign_fragments`. Still, we opted for more smaller methods, which are individually easier to maintain, but that could simply be a `questionable choice <https://en.wikipedia.org/wiki/Technical_debt>`_.
 * The 'dictionary unifying' methods could be replaced with pandas.DataFrame.merge/join
 * Writing to files, file manipulation should be done with pathlib
 * There's many other TODOs spread throughout the code
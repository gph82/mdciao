API (quick intro)
-----------------

The API is what you get to your Python scope upon importing ``mdciao``, .e.g.::

 import mdciao

The API has one method exclusively dedicated to replicate the Command-Line-Interface, ``mdciao.cli`` and a lot of other methods that expose useful functions to create your own workflows. Check the :ref:`API-reference` page for more information.

.. _api_note:
.. note::
   Whereas the command-line-tools tend to be more stable, the API is guaranteed to change until the first major release . Bugfixes, refactors and redesigns are in the pipeline and experienced users should know how to deal with this.
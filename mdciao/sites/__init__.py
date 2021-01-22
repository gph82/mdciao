r"""
Reading and manipulating sites.

Sites are collection of contacts that can be either
constructed by hand or read from plain ascii files,
like annotated json or unformatted dat-files::
    >>> cat site.dat
    # contacts to look at :
    L394-K270
    D381-Q229
    Q384-Q229
    R385-Q229
    D381-K232
    Q384-I135

    >>> cat site.json
    {"name":"interesting contacts",
    "bonds": {"AAresSeq": [
            "L394-K270",
            "D381-Q229",
            "Q384-Q229",
            "R385-Q229",
            "D381-K232",
            "Q384-I135"
            ]}}

.. autosummary::
   :toctree: generated/

   load
   sites_to_AAresSeqdict
   sites_to_res_pairs


"""
from .siteIO import *
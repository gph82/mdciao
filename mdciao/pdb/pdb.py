##############################################################################
#    This file is part of mdciao.
#    
#    Copyright 2020 Charité Universitätsmedizin Berlin and the Authors
#
#    Authors: Guillermo Pérez-Hernandez
#    Contributors:
#
#    mdciao is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    mdciao is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with mdciao.  If not, see <https://www.gnu.org/licenses/>.
##############################################################################

import requests as _requests

from mdtraj import load_pdb as _load_pdb
from tempfile import NamedTemporaryFile as _NamedTemporaryFile
from shutil import copy as _copy

from urllib.error import HTTPError as _HTTPError, URLError as _URLError

def _url2json(url,timeout,verbose):
    r""" Wraps around :obj:`requests.get` and tries to do :obj:`requests.Response.json`

    Parameters
    ----------
    url : str
    timeout : int
        seconds
    verbose : bool

    Returns
    -------
    dict

    """
    if verbose:
        print("Calling %s..." % url, end="")
    a = _requests.get(url, timeout=timeout)
    if verbose:
        print("done!")
    try:
        json = a.json()
    except:
        json = ValueError('Could not create a json out of  %s\n'
                          'Please check (e.g. via browser) if that is a valid url'%url)
    return json

def pdb2ref(pdb, url="https://data.rcsb.org/rest/v1/core/entry/",
            timeout=5):
    r"""
    Print the primary citation of a pdb code via web lookup

    The citation object is returned as a dict,
    check these links for more info
     * https://data.rcsb.org/index.html#data-api
     * https://data.rcsb.org/data-attributes.html

    Parameters
    ----------
    pdb : str
        four-leter pdb-code
    url : str
        the base URL for the look-ups, default is 'https://data.rcsb.org/rest/v1/core/entry/'
    timeout : int
        passed to the :obj:`requests.get` API,
        'How many seconds to wait for the server to send data
        before giving up'

    Returns
    -------
    ref : dict
        Whatever is contained in "rcsb_primary_citation"
        of https://data.rcsb.org/data-attributes.html
    """
    url = url.strip("/")+"/" + pdb.strip("/")
    ref = _url2json(url,timeout,verbose=False)
    try:
        ref = ref["rcsb_primary_citation"]
    except KeyError:
        print(url,":", ref["message"])
        return

    print("Please cite the following 3rd party publication:")
    print(" * "+ref["title"])
    print("   "+ref["rcsb_authors"][0],"et al.,", ref["rcsb_journal_abbrev"], ref["year"])
    print("   https://doi.org/"+ref["pdbx_database_id_doi"])

    return ref

def pdb2traj(code,
             filename=None,
             verbose=True,
             url="https://files.rcsb.org/download/",
             ):
    r""" Return a :obj:`~mdtraj.Trajectory` from a four-letter PDB code via RSCB PBB lookup

    Thinly wraps around :obj:`mdtraj.load_pdb`, printing the corresponding citation.
    Will return None if lookup fails

    Parameters
    ----------
    code : str
        four-letter code, e.g. 3SN6
    filename : str, default is None
        if str, save to this file,
        eventually overwriting
    verbose : bool, default is False
    url : str, default is 'https://files.rcsb.org/download'
        base URL for lookups

    Returns
    -------
    traj : :obj:`~mdtraj.Trajectory` or None

    """
    url1 = "%s/%s.pdb" % (url.strip("/"),code.strip("/"))

    #TODO use print_v elsewhere
    print_v = lambda s, **kwargs: [print(s, **kwargs) if verbose else None][0]
    print_v("Checking %s" % url1, end=" ...", flush=True)
    geom = None
    try:
        a = _requests.get(url1)
        if a.ok:
            with _NamedTemporaryFile(mode="w", suffix=".pdb") as f:
                f.writelines(a.text)
                print_v("done")
                geom = _load_pdb(f.name) #TODO use string.IO
                if filename is not None:
                    print_v("Saving to %s..." % filename, end="", flush=True)
                    if filename.lower().endswith(".pdb"):
                        _copy(f.name, filename)
                    else:
                        geom.save(filename)
                    print_v(filename)
        else:
            raise _URLError(a.text,filename=url1)


    except (_HTTPError, _URLError) as e:
        print(url1, ":", e)

    if geom is not None:
        pdb2ref(code)

    return geom

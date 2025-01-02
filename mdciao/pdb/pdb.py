##############################################################################
#    This file is part of mdciao.
#    
#    Copyright 2025 Charité Universitätsmedizin Berlin and the Authors
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

from mdtraj import load as _mdload
from tempfile import NamedTemporaryFile as _NamedTemporaryFile
from shutil import copy as _copy, copyfileobj as _copyfileobj
try:
    from shutil import COPY_BUFSIZE as _COPY_BUFSIZE
except ImportError:
    _COPY_BUFSIZE =  1024 * 64 #py37 shutil has this buffersize hard-coded 256 * 64, i'm setting 1024 * 64 across the board

from urllib.error import HTTPError as _HTTPError, URLError as _URLError
from urllib.request import urlopen as _urlopen

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
        try:
            print(url,":", ref["message"])
        except KeyError:
            print(url, ":", "no primary citation")
        return

    print("Please cite the following 3rd party publication:")
    print( " * "+ref["title"])
    line=f'   {ref["rcsb_authors"][0]} et al., {ref["rcsb_journal_abbrev"]}'
    try:
        line += f' {ref["year"]}'
    except KeyError:
        assert ref["rcsb_journal_abbrev"].lower() in ["To be published".lower()]
    print(line)
    try:
        print("   https://doi.org/"+ref["pdbx_database_id_doi"])
    except KeyError:
        pass #what would be a good control here?

    return ref

def pdb2traj(code,
             filename=None,
             verbose=True,
             url="https://files.rcsb.org/download/",
             ):
    r""" Return a :obj:`~mdtraj.Trajectory` from a four-letter PDB code via RCSB PDB lookup

    Thinly wraps around :obj:`mdtraj.load_pdb`, printing the corresponding citation.

    Will look up .pdb and .cif files (in that order) and return None the lookup fails

    Parameters
    ----------
    code : str
        four-letter code, e.g. 3SN6
    filename : str, or True, default is None
        * If str, save to this filename. The
          filename's extension can be used to
          force conversion into another format,
          i.e. the PDB ID might be accessible
          only as a CIF-file, but can be converted
          to .pdb on the fly.
        * If True, keep the original RCSB filename.

        Files are overwritten without checking
        if file exists.
    verbose : bool, default is False
    url : str, default is 'https://files.rcsb.org/download'
        base URL for lookups

    Returns
    -------
    traj : :obj:`~mdtraj.Trajectory` or None

    """

    #TODO use print_v elsewhere
    print_v = lambda s, **kwargs: [print(s, **kwargs) if verbose else None][0]
    for ext in ["pdb","cif"]:
        geom = None
        url1 = f"{url.strip('/')}/{code}.{ext}"
        print_v("Checking %s" % url1, end=" ...", flush=True)
        # From https://docs.python.org/3/howto/urllib2.html#fetching-urls
        try:
            with _urlopen(url1) as response:
                if response.status == 200:
                    with _NamedTemporaryFile(delete=True, suffix=f".{ext}") as tmp_file:
                        _copyfileobj(response, tmp_file, length=_COPY_BUFSIZE*2)
                        geom = _mdload(tmp_file.name)
                        if filename is not None:
                            if isinstance(filename, bool) and filename:
                                filename = f"{code}.{ext}"

                            print_v("Saving to %s..." % filename, end="", flush=True)
                            if filename.lower().endswith(f".{ext}"):
                                _copy(tmp_file.name, filename)
                            else:
                                geom.save(filename)
                            print_v(filename)
                else:
                    raise _URLError(response.reason,filename=url1)


        except (_HTTPError, _URLError) as e:
            print(url1, ":", e)

        if geom is not None:
            pdb2ref(code)
            break

    return geom

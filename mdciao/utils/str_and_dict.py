r"""
Functions for manipulating strings and dictionaries, also a bit of IO.


.. currentmodule:: mdciao.utils.str_and_dict


Functions
=========

.. autosummary::
   :toctree: generated/

"""
from glob import glob as _glob
import numpy as _np
import mdtraj as _md
from .lists import re_warp, contiguous_ranges as _cranges
from fnmatch import fnmatch as _fnmatch
from pandas import read_excel as _read_excel
from os import path as _path, listdir as _ls
import re as _re
from collections import defaultdict as _defdict

tunit2tunit = {"ps":  {"ps": 1, "ns": 1e-3, "mus": 1e-6, "ms":1e-9},
                "ns":  {"ps": 1e3, "ns": 1,    "mus": 1e-3, "ms":1e-6},
                "mus": {"ps": 1e6, "ns": 1e3,  "mus": 1,    "ms":1e-3},
                "ms":  {"ps": 1e9, "ns": 1e6,  "mus": 1e3,  "ms":1},
               }

def get_sorted_trajectories(trajectories):
    r"""
    Common parser for something that can be interpreted as a trajectory

    Parameters
    ----------
    trajectories: can be one of these things:
        - pattern, e.g. "*.ext"
        - one string containing a filename
        - list of filenames
        - one :obj:`mdtraj.Trajectory` object
        - list of :obj:`mdtraj.Trajectory` objects

    Returns
    -------
        - for an input pattern, sorted trajectory filenames that match that pattern
        - for filename, one list containing that filename
        - for a list of filenames, a sorted list of filenames
        - for one :obj:`mdtraj.Trajectory` object, a list containing that object
        - list of :obj:`mdtraj.Trajectory` objects (i.e. does nothing)


    """
    if isinstance(trajectories,str):
        _trajectories = _glob(trajectories)
        if len(_trajectories)==0:
            raise FileNotFoundError("Couldn't find (or pattern-match) anything to '%s'.\n"
                                    "ls $CWD[%s]:\n%s:"%(trajectories,
                                                         _path.abspath(_path.curdir),
                                                         "\n".join(_ls(_path.curdir))))
        else:
            trajectories=_trajectories

    if isinstance(trajectories[0],str):
        xtcs = sorted(trajectories)
    elif isinstance(trajectories, _md.Trajectory):
        xtcs = [trajectories]
    else:
        assert all([isinstance(itraj, _md.Trajectory) for itraj in trajectories])
        xtcs = trajectories

    return xtcs

def inform_about_trajectories(trajectories):
    r"""
    Return a string that informs about the trajectories

    Parameters
    ----------
    trajectories: list of strings or :obj:`mdtraj.Trajectory` objects

    Returns
    -------
    nothing, just prints them as newline

    """
    assert isinstance(trajectories, list), "input has to be a list"
    return "\n".join([str(itraj) for itraj in trajectories])

def replace_w_dict(input_str, exp_rep_dict):
    r"""
    Sequentially perform string replacements on a string using a dictionary

    Parameters
    ----------
    input_str: str
    exp_rep_dict: dictionary
        keys are expressions that will be replaced with values, i.e.
        key = key.replace(key1, val1) for key1, val1 etc

    Returns
    -------
    key

    """
    for pat, exp in exp_rep_dict.items():
        input_str = input_str.replace(pat, exp)
    return input_str

def delete_exp_in_keys(idict, exp, sep="-"):
    r"""
    Assuming the keys in the dictionary are formed by two segments
    joined by a separator, e.g. "GLU30-ARG40", deletes the segment
    containing the input expression, :obj:`exp`

    Will fail if not all keys have the expression to be deleted

    Parameters
    ----------
    idict: dictionary
    exp: str
    sep: str, default is "-",

    Returns
    -------
    dict:
        dictionary with the same values but the keys lack the
        segment containing :obj:`exp`

    dhk : list
        List with the deleted half-keys
    """

    out_dict = {}
    deleted_half_keys=[]
    for names, val in idict.items():
        new_name, dhk = delete_pattern_in_ctc_label(exp,names,sep)
        deleted_half_keys.extend(dhk)
        out_dict[new_name]=val
    return out_dict,deleted_half_keys

def delete_pattern_in_ctc_label(pattern, label, sep):
    new_name = [name for name in splitlabel(label, sep) if pattern not in name]
    deleted_half_keys = [name for name in splitlabel(label,sep) if pattern in name]
    assert len(new_name) == 1, (new_name, pattern)
    return new_name[0], deleted_half_keys

def unify_freq_dicts(freqs,
                     exclude=None,
                     key_separator="-",
                     replacement_dict=None,
                     defrag=None,
                     per_residue=False,
                     distro=False,
                     ):
    r"""
    Provided with a dictionary of dictionaries, returns an equivalent,
    key-unified dictionary where all sub-dictionaries share their keys,
    putting zeroes where keys where absent originally.

    Use :obj:`key_separator` for "GLU30-LY40" == "LYS40-GLU30" to be True

    Parameters
    ----------
    freqs:  dictionary of dictionaries, e.g.:
        {A:{key1:valA1, key2:valA2, key3:valA3},
         B:{            key2:valB2, key3:valB3}}

    key_separator: str, default is "-"
        Specify how residues are separated in the contact
        label, eg. "GLU30-LYS40".
        With this knowledge, the method can split the label
        before comparison so that "GLU30-LYS40" is considered
        equal to "LYS40-GLU30". Use "", "none" or None to differentiate.
        It will also be passed to :obj:`defrag_key` in case
        :obj:`defrag` is not None.

    exclude: list, default is None
         keys containing these strings will be excluded.
         NOTE: This is not implemented yet, will raise an error

    replacement_dict: dict, default is {}
        all keys/strings will be subjected to replacements following this
        dictionary, st. "GLH30" is "GLU30" if replacement_dict is {"GLH":"GLU"}
        This way mutations and or indexing can be accounted for in different setups
    defrag : char, default is None
        If a char is given, "@", anything after that character in the labels
        will be consider fragment information and ignored. This is only recommended
        for advanced users, usually the fragment information helps keep track
        of residue names in complex topologies:
            R201@frag1 and R201@frag3 will both be "R201"
    per_residue : bool, default is False
        Aggregate interactions to their residues

    Returns
    -------
    unified_dict: dictionary
        A dictionary  of dictionaries sharing keys:
       {A:{key1:valA1, key2:valA2, key3:valA3},
        B:{key1:0,     key2:valB2, key3:valB3}}
    """

    # Order key alphabetically using the separator_key
    def order_key(key, sep):
        split_key = splitlabel(key,sep)
        return sep.join([split_key[ii] for ii in _np.argsort(split_key)])

    # Create a copy, with re-ordered keys if needed
    freqs_work = {}
    for key, idict in freqs.items():
        if str(key_separator).lower()=="none" or len(key_separator)==0:
            freqs_work[key] = {key:val for key, val in idict.items()}
        else:
            freqs_work[key] = {order_key(key, key_separator):val for key, val in idict.items()}

    # Implement replacements
    if replacement_dict is not None:
        freqs_work = {key:{replace_w_dict(key2, replacement_dict):val2 for key2, val2 in val.items()} for key, val in freqs_work.items()}

    if defrag is not None:
        freqs_work = {key:{defrag_key(key2, defrag):val2 for key2, val2 in val.items()} for key, val in freqs_work.items()}

    if per_residue:
        freqs_work = {key:sum_dict_per_residue(val, key_separator) for key, val in freqs_work.items()}

    # Perform the difference operations
    not_shared = []
    shared = []
    for idict1 in freqs_work.values():
        for idict2 in freqs_work.values():
            if not idict1 is idict2:
                not_shared += list(set(idict1.keys()).difference(idict2.keys()))
                shared += list(set(idict1.keys()).intersection(idict2.keys()))

    shared = list(_np.unique(shared))
    not_shared = list(_np.unique(not_shared))
    all_keys = shared + not_shared
    # Prune keys we're not interested in
    excluded = []
    if exclude is not None:
        raise NotImplementedError("This feature not yet implemented")
        """
        assert isinstance(exclude,list)
        print("Excluding")
        for ikey, ifreq in freqs_work.items():
            # IDK I had this condition here, i think it is more intuitive if
            # they are removed regardless if shared or not
            #for key in shared:
            for key in list(ifreq.keys()):
                for pat in exclude:
                    if pat in key:
                        ifreq.pop(key)
                        print("%s from %s" % (key, ikey))
                        print(ifreq.keys())
                        excluded.append(key)
                        #all_keys = [ak for ak in all_keys if ak != key]
        """

    # Set the non shared keys to zero
    for ikey, ifreq in freqs_work.items():
        for key in all_keys:
            if key not in ifreq.keys():
                ifreq[key] = 0

    if len(not_shared)>0 and not distro:
        print("These interactions are not shared:\n%s" % (', '.join(not_shared)))
        print("Their cumulative ctc freq is %3.2f. " % _np.sum(
            [[ifreq[key] for ifreq in freqs_work.values()] for key in not_shared]))

    return freqs_work

def average_freq_dict(freqs,
                      weights=None,
                      **unify_freq_dicts_kwargs
                     ):
    r"""
    Average frequencies (or anything) over dictionaries.

    Typically, the input :obj:`freqs` are keyed first by system,
    then by contact label, e.g. {"T300":{"GDP-R201":1.0},
                                "T320":{"GDP-R201":.25},
                                "MUT":{"GDP-L201":25}}

    The input data need not be unified, the method calls
    :obj:`unify_freq_dicts` internally. In the example above
    you have to call it with the arg replacement_dict={"L201:R201"}
    so tha it can understand that mutation when unifying


    Parameters
    ----------
    freqs : dict of dicts
        The dictionaries containing frequence dictionaries,

    weights : dict, default is None
        relative weights of each dictionary

    unify_freq_dicts_kwargs

    Returns
    -------
    averaged_dict : dict
        an averaged dictionary keyed only with the


    """
    freqs_work = unify_freq_dicts(freqs,**unify_freq_dicts_kwargs)

    sys_keys = list(freqs_work.keys())
    frq_keys = list(freqs_work[sys_keys[0]].keys())
    averaged_dict = {}
    if weights is None:
        weights = {key:1 for key in sys_keys}
    for fk in frq_keys:
        averaged_dict[fk] = _np.average([freqs_work[isys][fk] for isys in sys_keys],
                                        weights=[weights[isys] for isys in sys_keys])

    return averaged_dict

def sum_dict_per_residue(idict, sep):
    r"""Return a "per-residue" sum of values from a "per-residue-pair" keyed dictionary

    Note:
    There is a closely related method in :obj:`mdciao.contacts.ContactGroup`
    that allows to query the freqs from the object already aggregated
    by residue. This is for when the object is either not accessible, e.g.
    because the freqs were loaded from a file

    Parameters
    ----------
    idict : dict
        Keyed with contact labels like "res1@frag1-res2@3.50" etc
    sep : char
        Character that separates fragments in the label

    Returns
    -------
    aggr : dict
        keyed with "res1@frag1" etc

    """
    out_dict = _defdict(list)
    for key, freq in idict.items():
        key1, key2 = splitlabel(key, sep) #This will fail if sep is not in key or sep does not separate in two
        out_dict[key1].append(freq)
        out_dict[key2].append(freq)
    return {key:_np.sum(val) for key, val in out_dict.items()}

def freq_file2dict(ifile, defrag=None):
    r"""
    Read a file containing the frequencies ("freq") and labels ("label")
    of pre-computed contacts

    Parameters
    ----------
    ifile : str
        Path to file, can be a .xlsx, .dat, .txt

    defrag : str, default is None
        If passed a string, e.g "@", the fragment information
        of the contact label will be deleted upon reading,
        so that R131@frag1 becomes R131. This is done
        by calling :obj:`defrag_key` internally

    Returns
    -------
    dict : keyed by labels and valued with frequencies, e.g .{"0-1":.3, "0-2":.1}

    """
    ext = _path.splitext(ifile)[-1]
    if ext.lower() == ".xlsx":
        df = _read_excel(ifile, engine="openpyxl")
        if "freq" in df.keys() and "label" in df.keys():
            res = {key: val for key, val in zip(df["label"].values, df["freq"].values)}
        else:
            row_lab, col_lab = _np.argwhere(df.values == "label").squeeze()
            row_freq, col_freq = _np.argwhere(df.values == "freq").squeeze()
            assert row_lab == row_freq,"File %s yields a weird dataframe on read \n%s"%(ifile,df)
            res = {key: val for key, val in zip(df.values[row_freq + 1:, col_lab].tolist(),
                                                 df.values[row_freq + 1:, col_freq].tolist())}
    else:
        res = freq_ascii2dict(ifile)
    if defrag is not None:
        res = {defrag_key(key, defrag=defrag):val for key, val in res.items()}

    return res

def freq_ascii2dict(ifile, comment=["#"]):
    r"""
    Reads an ascii file that contains contact frequencies (1st) column and
    contact labels . Columns are separated by tabs or spaces.

    Contact labels have to come after the frequency in the
    form of "res1 res2, "res1-res2" or "res1 - res2",

    Columns other than the frequencies and the residue labels are ignored


    TODO use pandas to allow more flex, not needed for the moment

    Parameters
    ----------
    ifile : str
        The filename to be read

    comment : list of chars
        Any line starting with any of these
        characters will be ignored
    Returns
    -------
    freqdict : dictionary

    """
    #TODO consider using pandas
    outdict = {}
    with open(ifile) as f:
        for iline in f.read().splitlines():
            if iline.strip()[0] not in comment:
                try:
                    iline = iline.replace("-"," ").split()
                    freq, names = float(iline[0]),"%s-%s"%(iline[1],iline[2])
                    outdict[names]=float(freq)
                except ValueError:
                    print(iline)
                    raise
    return outdict

def _find_latex_chunks(istr, blockers=['$$','\$']):
    r"""
    Find if a string has latex chunks in it, chunks of
     the string that are within two dollar signs.
    "There is $\alpha$ in the middle of this sentence"
    returns [[range(10,17)]]
    Parameters
    ----------
    istr

    Returns
    -------
    list of ranges

    """
    for bb in blockers:
        assert bb not in istr, ("Cannot find latex chunks in string %s, it contains the blocker %s"%(istr,bb))
    matches = [m.start() for m in _re.finditer('\$',istr)]
    assert _np.mod(len(matches),2)==0, "The string %s has to contain an even number of dollar signs, " \
                                       "but it contains %u"%(istr,len(matches))
    ranges = [_np.arange(ii,jj+1) for ii,jj in _np.reshape(matches,(-1,2))]
    return ranges

def replace4latex(istr):
    r"""
    Prepares the input for latex rendering (in matplotlib, e.g.)

    For strings with greek letters or underscores.
     * "alpha = 7"-> "$\alpha$ = 7"
     * "C_2"      -> "$C_2$"
     * "C^2"      -> "$C^2$"

    Parameters
    ----------
    istr: str
        "There's an alpha and a beta here, also C_200"

    Returns
    -------
    alpha: str
        'There's an $\alpha$ and a $\beta$ here, also $C_{200}$'

    """

    for gl in _symbols:
        if gl in istr:
            istr = _latexify(gl,istr)

    for c in ["_", "^"]:
        for word in istr.split():
            if c in word:
                istr = _latexify(word, istr)

    return istr

_symbols =  ['alpha','beta','gamma', 'mu', "Sigma"]+["AA"]
_scripts =  ["^","_"]

def _replace_regex_special_chars(word,
                                 repl_char="!",
                                 special_chars=["^", "[", "]", "(", ")"]):
    r"""
    Ad-hoc method to replace special regexp-chars with something else before
    computing char positions using regexp.finditer

    Note:
    this method only makes sense because downstream from here, finditer is used
    to search a substring in a string and special chars break that search.

    Note:
    somewhere, a dev that knows how to use regex is crying

    Parameters
    ----------
    word : str
    repl_char : char, default is '!'
        The replacement character
    special_chars : list
        The characters that trigger replacement

    Returns
    -------
    word : str
        A string with all special characters repaced with :obj:`repl_char`

    """
    for sp in special_chars:
        word = word.replace(sp, repl_char)
    return word

def _latexify(word, istr):
    # Look for appearances of this word in the whole string
    _word = _replace_regex_special_chars(word).replace("\\","\\\\")
    spans = [m.span() for m in _re.finditer(_replace_regex_special_chars(_word), _replace_regex_special_chars(istr))]
    for ii in range(len(spans)):
        span = spans[ii]
        latex_ranges = _find_latex_chunks(istr)
        add_dollar_signs = lambda istr : "$"+istr+"$"
        if any([set(lr).issuperset(span) for lr in latex_ranges]):
            add_dollar_signs = lambda istr:  istr
            # This substring is already within a latex chunk, can't do anything
            # except check if it's been enclosed in dollars but no \symbol, e.g. $beta_2$
        if word in _symbols:
            if istr[span[0] - 1] != '\\':
                new = add_dollar_signs('\%s' % word)
                istr = istr[:span[0]] + new + istr[span[1]:]
        for char in _scripts:
            if char in word:
                try:
                    word1,word2 = word.split(char)
                    if word2[0] not in ["{","\\"]:
                        new = word1+add_dollar_signs("%s{\mathrm{%s}}"%(char,word2))
                        istr = istr[:span[0]] + new + istr[span[1]:]
                except (ValueError, IndexError) as e:
                    print("Cannot latexify word with more than one instance of %s in the same word: %s"%(char,word))
        spans = [m.span() for m in _re.finditer(word, istr)]
    istr = istr.replace("$$","")
    return istr

def latex_mathmode(istr, enclose=True):
    r"""
    Prepend *symbol* words with "\\ " and protect *non-symbol* words with '\\mathrm{}'

    * *symbol* words are things that can
      be interpreted by LaTeX in math mode, e.g.
      '\\alpha' or '\\AA'
    * *non-symbol* words are everything else

    Works "opposite" to :obj:`replace4latex` and for the moment
    it's my (very bad) solution for latexifying contact-labels' fragments
    as super indices where the the fragments themselves contain
    sub-indices (GLU30^$\beta_2AR}


    >>> replace4latex("There's an alpha and a beta here, also C_200")
    "There's an $\alpha$ and a $\beta$ here, also $C_{200}$"

    >>> latex_mathmode("There's an alpha and a beta here, also C_200")
    "$\\mathrm{There's an }\\alpha\\mathrm{ and a }\\beta\\mathrm{ here, also C_200}$"

    Parameters
    ----------
    istr : string
    enclose : bool, default is True
        Return string enclosed in
        dollar-signs: '$string$'
        Use False for cases where
        the LaTeX math-mode is already
        active

    Returns
    -------
    istr : string
    """
    output = []
    exp = "(%s)" % "|".join(["\%s" % ss if ss == "^" else "%s" % ss for ss in _symbols])
    for word in _re.split(exp, istr):
        if len(word) > 0:
            if word in _symbols:
                word = "\\%s" % word
            else:
                word = "\\mathrm{%s}" % word
            output.append(word)
    output = "".join(output)
    if enclose:
        output= "$%s$"%output
    return output

def latex_superscript_fragments(contact_label, defrag="@"):
    r"""
    Format fragment descriptors as Latex math-mode superscripts

    Thinly wrap around :obj:`_latex_superscript_one_fragment` with :obj:`splitlabel`

    Parameters
    ----------
    contact_label : str
        contact label of any form,
        as long as to AAs are joined
        with '-' character
    defrag : char, default is '@'
        The character to divide
        residue and fragment label
    Returns
    -------
    contact_label : str

    """
    return '-'.join(_latex_superscript_one_fragment(w, defrag=defrag) for w in splitlabel(contact_label, "-"))

def _latex_superscript_one_fragment(label, defrag="@"):
    r"""
    Format s.t. the fragment descriptor appears as superindex in LaTeX math-mode

    Parameters
    ----------
    label : str
        Contact label, "GLU30" and
        optionally "GLU30@beta_2AR"
    defrag : char, default is '@'
        The character to divide
        residue and fragment label

    Returns
    -------
    label : str
    """
    words = label.split(defrag,maxsplit=1)
    if len(words)==1:
        return label
    elif len(words)==2:
       return words[0] +"$^{%s}$" % latex_mathmode(words[1], enclose=False)

def _label2componentsdict(istr,sep="-",defrag="@",
                          assume_ctc_label=True):
    r"""
    Identify the components of label like 'residue1@frag1-residue2@frag2' and return them as dictionary

    Parameters
    ----------
    istr : str
        Can be of any of these forms:
        * res1
        * res1@frag1
        * res1@frag1-res2
        * res1@frag1-res2@frag2
        * res1-res2@frag2
        * res1-res2

        The fragment names can contain the separator, e.g.
        'res1@B2AR-CT-res2@Gprot' is possible, but residue
        names cannot.

        The special case 'res1@frag1-r2' is handled with
        the parameter :obj:`assume_ctc_label` (see below)

        Labels have to start with a residue.
    sep : char, default is "-"
        The character that separates pairs of labels
    defrag : char, default is "@"
        The character that separates residues form their host fragment
    assume_ctc_label : bool, default is True
        In special cases of the form 'res1@frag1-r2', assume
        this is a contact label, i.e. 'r2' does not
        belong to the name of the fragment of res1, but is
        the second residue.

    Returns
    -------
    label : dict
        A dictionary tuple with the components present in :obj:`istr`.
        Keys can be 'res1','frag1','res2','frag2'
    """
    assert len(sep)==len(defrag)==1, "The 'sep' and 'defrag' arguments have to have both len 1, have " \
                                     "instead %s (%u) %s (%u)"%(sep,len(sep),defrag,len(defrag))

    bits = {}

    if defrag not in istr:
        for ii, ires in enumerate(istr.split(sep),start=1):
            bits["res%u"%ii]=ires
    else:
        spans = [0] + _np.hstack([m.span() for m in _re.finditer(defrag, istr)]).tolist() + [len(istr)]

        # Counters
        r, f = 1, 1
        for ii, jj in _np.reshape(spans, (-1, 2)):
            iw = istr[ii:jj + 1]
            #print(iw, ii, jj)

            if sep in iw and ii == 0:
                ires, jres = iw.replace(defrag,"").split(sep)
                bits["res%u"%r]=ires
                r+=1
                bits["res%u"%r]=jres
                r+=1
                f+=1 # because we've already established res1 hasn't any fragment
            else:
                if defrag not in iw:
                    if sep not in iw or not assume_ctc_label:
                        bits["frag%u"%f]=iw
                        f+=1
                    elif sep in iw and assume_ctc_label:
                        ires, ifrag = [jw[::-1] for jw in iw[::-1].split(sep, 1)]
                        if "res1" in bits.keys():
                            if "frag1" in bits.keys():
                                bits["frag%u"%f]=iw
                                f+=1
                            else:
                                if "res2" in bits.keys():
                                    bits["frag%u" % f] = iw
                                else:
                                    bits["frag%u"%f]=ifrag
                                    f+=1
                                    bits["res%u"%r]=ires
                                    r+=1

                else:
                    assert iw.endswith(defrag)
                    if sep not in iw:
                        bits["res%u"%r]=iw.split(defrag)[0]
                        r+=1
                    else:
                        ires, ifrag = [jw[::-1] for jw in iw[::-1][1:].split(sep, 1)]
                        bits["frag%u"%f]=ifrag
                        bits["res%u"%r]=ires
                        f+=1
                        r+=1


    return bits

def splitlabel(label, sep="-", defrag="@"):
    r"""
    Split a contact label. Analogous to label.split(sep) but more robust
    because fragment names can contain the separator character.

    Parameters
    ----------
    label : str
        Can be of any of these forms:
         * res1
         * res1@frag1
         * res1@frag1-res2
         * res1@frag1-res2@frag2
         * res1-res2@frag2
         * res1-res2

        The fragment names can contain the separator, e.g.
        'res1@B2AR-CT-res2@Gprot' is possible. Residue
        names cannot contain the separator.

        The method assumes that labels start with a residue,
        (see above), else you'll get weird behaviour.
    sep : char, default is "-"
        The character that separates pairs of labels
    defrag : char, default is "@"
        The character that separates residues form their host fragment

    Returns
    -------
    split : list
        A list equivalent to having used label.split(sep)
        but the separator is ignored in the fragment labels.
    """

    bits = _label2componentsdict(label,sep=sep,defrag=defrag)

    split = [bits["res1"]]
    if "frag1" in bits.keys():
        split[0] += "%s%s" % (defrag,bits["frag1"])
    if "res2" in bits.keys():
        split.append(bits["res2"])
        if "frag2" in bits.keys():
            split[1] += "%s%s" % (defrag,bits["frag2"])
    return split

def intblocks_in_str(istr):
    r"""
    Return the integers that appear as contiguous blocks in strings

    E.g.  "GLU30@3.50-GDP396@frag1" returns [30,3,50,396,1]

    Parameters
    ----------
    istr : string

    Returns
    -------
    ints : list

    """
    intblocks = _cranges([char.isdigit() for char in istr])[True]
    return [int("".join([istr[idx] for idx in block])) for block in intblocks]

def iterate_and_inform_lambdas(ixtc,chunksize, stride=1, top=None):
    r"""
    Given a trajectory (as object or file), returns
    a strided, chunked iterator and function for progress report

    Parameters
    ----------
    ixtc: str (filename) or :obj:`mdtraj.Trajectory` object
    chunksize: int
        The trajectory will be iterated over in chunks of this many frames
    stride: int, default is 1
        The stride with which to iterate over the trajectory
    top:  str (filename) or :obj:`mdtraj.Topology`
        If :obj:`ixtc` is a filename, the topology needed to read it

    Returns
    -------

    iterate, inform

    iterate: lambda(ixtc)
        strided, chunked iterator over :obj:`ixtc`

    inform: lambda(ixtc, traj_idx, chunk_idx, running_f)
        iterator that prints out streaming progress for every iteration

    Note
    ----

    The lambdas returned differ depending on the type of input, but signature
    is the same, s.t. the user does not have to care in posterior use

    """
    if isinstance(ixtc, _md.Trajectory):
        # TODO it's time to use yield here, this is killing performance
        iterate = lambda ixtc: [ixtc[idxs] for idxs in re_warp(_np.arange(ixtc.n_frames)[::stride], chunksize)]
        inform = lambda ixtc, traj_idx, chunk_idx, running_f: \
            print("Streaming over trajectory object nr. %3u (%6u frames, %6u with stride %2u) in chunks of "
                  "%3u frames. Now at chunk nr %4u, frames so far %6u" %
                  (traj_idx, ixtc.n_frames, _np.ceil(ixtc.n_frames/stride), stride, chunksize, chunk_idx, running_f), end="\r", flush=True)
    elif ixtc.endswith(".pdb") or ixtc.endswith(".pdb.gz") or ixtc.endswith(".gro"):
        iterate =  lambda ixtc: [_md.load(ixtc)[::stride]]
        inform  =  lambda ixtc, traj_idx, chunk_idx, running_f: \
            print("Loaded %20s (nr. %3u) in full, using stride %2u but ignoring chunksize of "
                  "%6u frames. This number should always be 0 : %4u. Total frames loaded %6u" %
                  (ixtc, traj_idx, stride, chunksize, chunk_idx, running_f), end="\r", flush=True)
    else:
        iterate = lambda ixtc: _md.iterload(ixtc, top=top, stride=stride, chunk=_np.round(chunksize / stride))
        inform = lambda ixtc, traj_idx, chunk_idx, running_f: \
            print("Streaming %20s (nr. %3u) with stride %2u in chunks of "
                  "%6u frames. Now at chunk nr %4u, frames so far %6u" %
                  (ixtc, traj_idx, stride, chunksize, chunk_idx, running_f), end="\r", flush=True)
    return iterate, inform

def choose_options_descencing(options,
                              fmt="%s",
                              dont_accept=["none", "na"]):
    r"""
    Return the first entry that's acceptable according to some rule

    If no is found, "" is returned
    Parameters
    ----------
    options : list
    fmt : str, default is "%s"
        You can specify a different
        format here. Will only
        apply in case something
        is returned
    dont_accept : list
        Move down the list if
        current item is one
        of these

    Returns
    -------
    best : str
        Either the best entry in :obj:`options`
        or "" if no option was found
    """
    for option in options:
        if str(option).lower() not in dont_accept:
            return fmt%str(option)
    return ""


def fnmatch_ex(patterns_as_csv, list_of_keys):
    r"""
    Match the keys in :obj:`list_of_keys` against some naming patterns
    using Unix filename pattern matching
    TODO include link:  https://docs.python.org/3/library/fnmatch.html

    This method also allows for exclusions (grep -e)

    TODO: find out if regular expression re.findall() is better

    Uses fnmatch under the hood

    Parameters
    ----------
    patterns_as_csv : str
        Patterns to include or exclude, separated by commas, e.g.
        * "H*,-H8" will include all TMs but not H8
        * "G.S*" will include all beta-sheets
    list_of_keys : list
        Keys against which to match the patterns, e.g.
        * ["H1","ICL1", "H2"..."ICL3","H6", "H7", "H8"]

    Returns
    -------
    matching_keys : list

    """
    include_patterns = [pattern for pattern in patterns_as_csv.split(",") if not pattern.startswith("-")]
    exclude_patterns = [pattern[1:] for pattern in patterns_as_csv.split(",") if pattern.startswith("-")]
    #print(include_patterns)
    #print(exclude_patterns)
    # Define the match using a lambda
    matches_include = lambda key : any([_fnmatch(str(key), patt) for patt in include_patterns])
    matches_exclude = lambda key : any([_fnmatch(str(key), patt) for patt in exclude_patterns])
    passes_filter = lambda key : matches_include(key) and not matches_exclude(key)
    outgroup = []
    for key in list_of_keys:
        #print(key, matches_include(key),matches_exclude(key),include_patterns, exclude_patterns)
        if passes_filter(key):
            outgroup.append(key)
    return outgroup

def match_dict_by_patterns(patterns_as_csv, index_dict, verbose=False):
    r"""
    Joins all the values in an input dictionary if their key matches
    some patterns. This method also allows for exclusions (grep -e)

    TODO: find out if regular expression re.findall() is better

    Parameters
    ----------
    patterns_as_csv : str
        Comma-separated patterns to include or exclude, separated by commas, e.g.
        * "H*,-H8" will include all TMs but not H8
        * "G.S*" will include all beta-sheets
    index_dict : dictionary
        It is expected to contain iterable of ints or floats or anything that
        is "joinable" via np.hstack. Typically, something like:
        * {"H1":[0,1,...30], "ICL1":[31,32,...40],...}

    Returns
    -------
    matching_keys, matching_values : list, array of joined values

    """
    matching_keys =   fnmatch_ex(patterns_as_csv, index_dict.keys())
    if verbose:
        print(', '.join(matching_keys))

    if len(matching_keys)==0:
        matching_values = []
    else:
        matching_values = _np.hstack([index_dict[key] for key in matching_keys])

    return matching_keys, matching_values

def defrag_key(key, defrag="@", sep="-"):
    r"""Remove fragment information from a contact label

    Parameters
    ----------
    key : str
        Contact label with some sort of pair information
        e.g. e.g. R1@frag1-E2@frag2->R1-E2
    defrag: char, default is "@"
        Character that indicates the beginning of the
        fragment
    sep : char, default is "-"
        Character that indicates the separation
        between first and second residue of the pair

    Returns
    -------

    """
    return sep.join([kk.split(defrag,1)[0].strip(" ") for kk in splitlabel(key,sep)])

def df_str_formatters(df):
    r"""
    Return formatters for :obj:`~pandas.DataFrame.to_string'

    In principle, this should be solved by
    https://github.com/pandas-dev/pandas/issues/13032,
    but I cannot get it to work

    Parameters
    ----------
    df : :obj:`~pandas.DataFrame`

    Returns
    -------
    formatters : dict
        Keyed with :obj:`df`-keys
        and valued with lambdas
        s.t. formatters[key][istr]=formatted_istr

    """
    formatters = {}
    for key in df.keys():
        fmt = "%%-%us"%max([len(ii)+1 for ii in df[key]])
        formatters[key]=lambda istr : fmt%istr
    return formatters

class FilenameGenerator(object):
    r"""
    Generate per project filenames when you need them

    This is a WIP to consolidate all filenaming in one place,
    s.t. all sanitizing and project-specific naming operations happen
    here and not in the cli methods

    A named tuple would've been enough, but we need some
     methods for dynamic naming (e.g. per-residue or per-traj)

    """

    def __init__(self, output_desc, ctc_cutoff_Ang, output_dir, graphic_ext, table_ext, graphic_dpi, t_unit):

        self._graphic_ext = graphic_ext.strip(".")
        self._output_desc = output_desc.strip(".")
        self._ctc_cutoff_Ang = ctc_cutoff_Ang
        self._output_dir = output_dir
        self._graphic_dpi = graphic_dpi
        self._t_unit = t_unit
        self._allowed_table_exts = ["dat", "txt", "xlsx", "ods"] #TODO what about npy?
        assert str(table_ext).lower != "none"
        self._table_ext = str(table_ext).lower().strip(".")
        if self._table_ext not in self._allowed_table_exts:
            raise ValueError("The table extension, cant be '%s', "
                             "has be one of %s"%(table_ext,self._allowed_table_exts))


    @property
    def output_dir(self):
        return self._output_dir
    @property
    def basename_wo_ext(self):
        return "%s.overall@%2.1f_Ang" % (self.output_desc,
                                         self.ctc_cutoff_Ang)
    @property
    def ctc_cutoff_Ang(self):
        return self._ctc_cutoff_Ang

    @property
    def output_desc(self):
        return self._output_desc.replace(" ","_")

    @property
    def fullpath_overall_no_ext(self):
        return _path.join(self.output_dir, self.basename_wo_ext)

    @property
    def graphic_ext(self):
        return self._graphic_ext

    @property
    def graphic_dpi(self):
        return self._graphic_dpi
    @property
    def table_ext(self):
        return self._table_ext

    @property
    def t_unit(self):
        return self._t_unit
    @property
    def fullpath_overall_fig(self):
        return ".".join([self.fullpath_overall_no_ext, self.graphic_ext])

    def fname_per_residue_table(self,istr):
        assert self.table_ext is not None
        fname = '%s.%s@%2.1f_Ang.%s' % (self.output_desc,
                                        istr.replace('*', "").replace(" ","_"),
                                        self.ctc_cutoff_Ang,
                                        self.table_ext)
        return _path.join(self.output_dir, fname)

    def fname_per_site_table(self, istr):
        return self.fname_per_residue_table(istr)


    def fname_timetrace_fig(self, surname):
        return '%s.%s.time_trace@%2.1f_Ang.%s' % (self.output_desc,
                                                  surname.replace(" ", "_"),
                                                  self.ctc_cutoff_Ang,
                                                  self.graphic_ext)
    @property
    def fullpath_overall_excel(self):
        return ".".join([self.fullpath_overall_no_ext, "xlsx"])

    @property
    def fullpath_overall_dat(self):
        return ".".join([self.fullpath_overall_no_ext, "dat"])

    @property
    def fullpath_pdb(self):
        return ".".join([self.fullpath_overall_no_ext, "as_bfactors.pdb"])

    @property
    def fullpath_matrix(self):
        return self.fullpath_overall_fig.replace("overall@", "matrix@")

    @property
    def fullpath_flare_pdf(self):
        return '.'.join([self.fullpath_overall_no_ext.replace("overall@", "flare@"), 'pdf'])


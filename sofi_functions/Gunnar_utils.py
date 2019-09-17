
from collections import defaultdict as _df
import numpy as _np
from string import whitespace as _whitespace
from pandas import DataFrame as _DF, ExcelWriter as _EW
from xlsxwriter.utility import xl_col_to_name as _xl_col_to_name
from Bio import pairwise2 as _pairwise2
from tempfile import TemporaryDirectory as _TDir
import os as _os
from subprocess import check_output as _check_output
from IPython.display import Image as _Image, display as _display

class alignment(object):

    def __init__(self, textfile='exported.txt',
                 structure_keyword='structure',
                 replacements={"Homo_sap":"Homo Sap"},
                 homogenize_keys=True):
        pass

        self._initialized = False
        self.structure_keyword = structure_keyword
        self.textfile=textfile

        self._raw_input = open(textfile).read().splitlines()
        self._structure_lines = [ii for ii, iline in enumerate(self._raw_input) if iline.startswith(self.structure_keyword)]
        self._raw_data = {}
        for ii, iline in enumerate(self._raw_input):
            key, val = iline.split("\t")
            if key!=self.structure_keyword and len(key)!=0:
                self._raw_data[key]=val

        # Get group labels
        self.group_labels = self._identify_group_labels()
        print("Identified these groups:\n%s\n"%('\n'.join([ii for ii in self.group_labels])))

        # Get groups
        self.groups = self._identify_groups()

        # Pad sequences to the at the right
        self._max_seq_length = _np.max([len(iseq) for iseq in self._raw_data.values()])

        self.data = {key: {"seq": val + " " * (self._max_seq_length - len(val)),
                           "group": self.in_what_group(key)
                           }
                     for key, val in self._raw_data.items() if key not in self.group_labels}

        if homogenize_keys:
            print("Did some renaming:")
            self.data = {_homogenize_str(key, True, replacement_dict=replacements): val for key, val in
                         self.data.items()}
            self.groups = {key: [_homogenize_str(ival, replacement_dict=replacements) for ival in val] for key, val in
                           self.groups.items()}
            print()
        self._initialized = True

        if False:
            # TODO not much difference between __init__ and __initialize_data...consolidate

            """
            self._title_length = self._identify_title_length()


            print("Title length",self._title_length)
            self._first_line = self._structure_lines[0]
            self._space_lines = [ii for ii, iline in enumerate(self._raw_input) if len(iline)==0]

            # First pass at re-patching the data
            self._raw_data = self._raw_input_to_raw_data()

            # Pop structure entry
            self._raw_data.pop(self.structure_keyword)


            # Pop groups
            raw_keys = list(self._raw_data.keys())
            [self._raw_data.pop(key) for key in raw_keys if any([gl in key for gl in self.group_labels])]

            

            # The raw data has to have a consistent block length
            self.block_length = self._identify_block_length()

            # Since we have checked for a consistent block length,
            # we can simply eliminate spaces
            """

    def _identify_title_length(self):
        title_lengths=_np.unique([iline.find('.') for iline in self._raw_input if '|' in iline])
        assert len(title_lengths)==1, ValueError("There's inconsistencies in the title lengths",title_lengths)
        return title_lengths[0]

    def _raw_input_to_raw_data(self):
        if not self._initialized:
            raw_data = _df(list)
            for line in self._raw_input[self._first_line:]:
                if len(line)>0 and line[0] not in _whitespace:
                    key, val = line[:self._title_length], line[self._title_length:]
                    key, val =key.strip(_whitespace), val.lstrip(_whitespace)
                    fmt = '%%-%us %%s'%self._title_length
                    #print(fmt%(key,val[:10]))
                    raw_data[key].append(val)
            raw_data = {key: ''.join(val) for key, val in raw_data.items()}
            return raw_data
        else:
            print("Object already initialized, doing nothing")

    def _identify_groups(self):
        if not self._initialized:
            groups = {key:[] for key in self.group_labels}
            for key, val in self._raw_data.items():
                if key in self.group_labels:
                    current_key = key
                elif key != self.structure_keyword:
                    if key not in groups[current_key]:
                        groups[current_key].append(key)
                        #print(current_key, key)
            return groups
        else:
            print("Object already initialized, doing nothing")

    def _identify_group_labels(self):
        if not self._initialized:
            group_labels=[]
            for key, val in self._raw_data.items():
                uniquely = _np.unique([ii for ii in val])
                no_AAs = all([not ii.isalpha() for ii in uniquely])
                if no_AAs:
                    #print(key, len(uniquely), uniquely)
                    group_labels.append(key)

            # Unique-ify the group labels conserving the order:
            group_labels = [gl.split()[0] for gl in group_labels if len(gl) > 0]
            gl_out = []
            for gl in group_labels:
                if gl not in gl_out:
                    gl_out.append(gl)
            group_labels = gl_out
            return group_labels

        else:
            print("Object already initialized, doing nothing")

    def _identify_block_length(self):
        #from string import isspace
        if not self._initialized:
            for key, val in self._raw_data.items():
                spaces =_np.argwhere([ival.isspace() for ival in val]).squeeze()
                block_length=_np.unique(_np.diff(spaces))-1
                assert len(block_length)==1,TypeError("This sting does not have constant spacing, but spaces between blocks include ",block_length)
        else:
            print("Object already initialized, doing nothing")
        return block_length[0]

    def show_group(self,group):
        for key in self.groups[group]:
            print(key, self.data[key])

    def in_what_group(self,key, fail_if_more_than_one_group=True):
        groupout=[]
        for group_key, group_members in self.groups.items():
            if key in group_members:
                groupout.append(group_key)
        n_g = len(groupout)
        if fail_if_more_than_one_group and n_g>1:
            raise ValueError("Found %s in more than one group %s. Aborting"%(key,groupout))

        else:
            if len(groupout)==0:
                raise ValueError("No group found for '%s' (groups=%s"%(key,self.group_labels))
            else:
                return groupout[0]

    def show_groups(self,exclude=[], just_keys=False):
        for group_name, group_members in self.groups.items():
            if group_name not in exclude:
                print(group_name)
                for member in group_members:
                    if just_keys:
                        print(member)
                    else:
                        print(member, self.data[member])
                print()

    def get_group(self,group_name):
        return {key:self.data[key] for key in self.groups[group_name]}


    def export_to_txt(self,fileout='myexport.txt', maxlength=None):
        fmt = '%%-%us %%s\n' % self._title_length
        with open(fileout,'w') as f:
            for key, val in self.data.items():
                jval = val["seq"][:]
                if maxlength is not None:
                    jval=jval[:maxlength]
                f.write(fmt%(key,jval))

    def to_dataframe(self):
        out_dict = {}
        for ii, groupname in enumerate(self.group_labels):
            out_dict[groupname]=[]
            for member, val in self.get_group(groupname).items():
                out_dict[member]=val["seq"]
            out_dict[' '*ii]=[]

        return _DF.from_dict(out_dict, orient='index')

    def to_excel(self, fileout=None, engine='xlsxwriter',
                 input_dataframe=None, aa_col_width=5):

        if fileout is None:
            fileout = '%s.xlsx'%_os.path.splitext(self.textfile)[0]
            print("Exporting to %s"%fileout)
        if input_dataframe is None:
            mydf = self.to_dataframe()
        else:
            mydf = input_dataframe
        with _EW(fileout, engine=engine) as myexcel:
            mydf.to_excel(myexcel, header=_np.arange(1, mydf.shape[1]+1))
            ws = myexcel.sheets["Sheet1"]
            my_COL_NAMES = [_xl_col_to_name(ii) for ii in range(len(mydf.keys())+1)]
            first_col_fmt = myexcel.book.add_format({"align": "left"})
            first_col_width = _np.max([len(key) for key in mydf.transpose().keys()])
            ws.set_column("A:A", first_col_width, first_col_fmt)

            other_col_fmt = myexcel.book.add_format({"align": "center"})
            col_range_fmtd = '%s:%s' % (my_COL_NAMES[1], my_COL_NAMES[-1])
            ws.set_column(col_range_fmtd, aa_col_width, other_col_fmt)

            myexcel.save()
    def _squeeze_sequence(self, seq):
        seq_out, seq_idxs = [],[]
        for ii, ichar in enumerate(seq):
            if ichar.isalpha():
                seq_out.append(ichar)
                seq_idxs.append(ii)

        return ''.join(seq_out), seq_idxs

    @property
    def dataframe(self):
        out_DF = _DF.from_dict({key:[val["group"]]+[ichar for ichar in val["seq"]] for key, val in self.data.items()}, orient="index")
        out_DF.rename(mapper={0:"Group"}, axis=1, inplace=True)
        return out_DF

    def align_MDTopology(self, top,
                                subset_keys=None,
                                reference_key=None,
                                res_top_key='res_top',
                                res_ref_key="res_ref",
                                idx_key='idx',
                                resSeq_key="resSeq",
                                resname_key="fullname",
                                return_alignment_to_reference=False,
                         verbose=False,
                                ):
        if subset_keys is None:
            raise NotImplementedError
        if reference_key is None:
            raise NotImplementedError

        print("Aligning the input MD sequence to %s"%reference_key)

        # Prepare sequences
        refseq_squeezed, refseq_idxs=self._squeeze_sequence(self.data[reference_key]["seq"])
        top_seq = ''.join([rr.code for rr in top.residues if rr.is_protein])

        # Align
        alignment=_pairwise2.align.globalxx(top_seq, refseq_squeezed)

        # Mix each alignemnt with Gunnar's alignment
        subset = {key:val for key,val in self.data.items()}
        for ialg in alignment:
            #print(ialg)
            list_of_alignment_dicts = alignment_result_to_list_of_dicts(ialg, top, refseq_idxs,
                                                                        res_ref_key=res_ref_key,
                                                                        res_top_key=res_top_key,
                                                                        resSeq_key=resSeq_key,
                                                                        idx_key=idx_key)


            mixed_aligments = mix_alignments(subset, list_of_alignment_dicts,
                                             reference_key,
                                             res_ref_key=res_ref_key,
                                             res_top_key=res_top_key,
                                             resSeq_key=resSeq_key,
                                             idx_key=idx_key)
            new_order = [key for key in subset_keys if key != reference_key] + [reference_key] \
                        + [res_top_key] \
                        + [resSeq_key] \
                        + [resname_key]

            if not return_alignment_to_reference:
                return mixed_aligments[new_order]
            else:
                return mixed_aligments[new_order], _DF(list_of_alignment_dicts)

            break


def mix_alignments(subset,
                   list_of_alignment_dicts,
                   refseqkey,
                   res_top_key="res_top",
                   res_ref_key="res_ref",
                   resSeq_key="resSeq",
                   idx_key="idx",
                   resname_key='fullname'
                   ):

    alignment_iterator = iter([idict for idict in list_of_alignment_dicts if idict[res_ref_key].isalpha()])
    refseq_idxs =[idict[idx_key] for idict in list_of_alignment_dicts if idict[res_ref_key]]
    seqlenths = [len(ival["seq"]) for ival in subset.values()]
    assert len(_np.unique(seqlenths)) == 1, seqlenths
    max_idx = _np.max(seqlenths)
    row_dicts = []
    for ii in range(max_idx):
        row_dicts.append({key: val["seq"][ii] for key, val in subset.items()})
        row_dicts[-1][res_top_key] = '~'
        row_dicts[-1][resSeq_key] = '~'
        if ii in refseq_idxs:
            rrtt = next(alignment_iterator)
            rt, resSeq, rr, fullname = [rrtt[key] for key in [res_top_key, resSeq_key, res_ref_key, resname_key]]
            assert rr == row_dicts[-1][refseqkey], (rr, row_dicts[-1][refseqkey])
            row_dicts[-1][res_top_key] = rt
            row_dicts[-1][resSeq_key] = resSeq
            row_dicts[-1][resname_key] = fullname

    output = _DF(row_dicts)

    return output

def _homogenize_str(istr, verbose=False, replacement_dict=None):
    #new = istr.replace("_"," ")
    new = istr[:]
    if "[" in new:
        prev_char = new[new.find("[")-1]
        if not prev_char.isspace():
            new=new.replace(prev_char+"[",prev_char+" [")
    new = new.replace("_ ["," [")

    if replacement_dict is not None:
        for key, val in replacement_dict.items():
            new = new.replace(key,val)
    if verbose and new!=istr:
        print("%-30s -> %s"%(istr,new))
    return new

def weblogo2pdf_and_png(weblogo_formatter_output, filename_wo_ext="weblogo", show=True):
    with _TDir() as tmpdirname:
        f = open(_os.path.join(tmpdirname, "out.eps"), "w")
        f.write(weblogo_formatter_output
                .decode()
                )
        f.close()
        #print(f.name)
        _check_output(["ps2pdf", "-dEPSCrop", f.name, "%s.pdf"%filename_wo_ext])
        _check_output(["pdftocairo", "%s.pdf"%filename_wo_ext, "-png", "-r", "600"])

    if show:
        _display(_Image("%s-1.png"%filename_wo_ext))

def dataframe2seqlists(dataframe_in, exclude_chars=['~']):
    seqlist = []
    for icol in dataframe_in.iloc[:, 0:-3]:
        iseq = ''.join(dataframe_in[icol].tolist())
        print(iseq,end='')
        if not any([ichar in iseq for ichar in exclude_chars]):
            seqlist.append(iseq)
            print("")
        else:
            print("*")
    return seqlist

def myfasta
import unittest
import mdtraj as md
import numpy as _np
from mdciao.filenames import filenames
from pandas import DataFrame as _DF
from mdciao.fragments import get_fragments
from mdciao.utils import sequence
import pytest

test_filenames = filenames()

class Test_print_verbose_dataframe(unittest.TestCase):

    def test_just_prints(self):
        df = _DF.from_dict({"x":_np.arange(500),
                            "x²":_np.arange(500)**2}
                           )
        sequence.print_verbose_dataframe(df)

class Test_top2seq(unittest.TestCase):

    def test_works(self):
        top = md.load(test_filenames.small_monomer).top
        seq = sequence.top2seq(top)
        _np.testing.assert_array_equal("EVWIEKXX",seq)
    def test_other_letter(self):
        top = md.load(test_filenames.small_monomer).top
        seq = sequence.top2seq(top, replacement_letter="Y")
        _np.testing.assert_array_equal("EVWIEKYY", seq)


class Test_my_bioalign(unittest.TestCase):

    def test_works(self):
        seq1 = "EVWIEKXX"
        seq2 = seq1[::-1]+seq1.replace("I","A")+seq1[::-1]
        algnmt = sequence.my_bioalign(seq1, seq2)[0]
        res1 = "".join(["-" for __ in seq1])+seq1+"".join(["-" for __ in seq1])
        res2 = seq2
        _np.testing.assert_array_equal(algnmt[0],res1)
        _np.testing.assert_array_equal(algnmt[1],res2)

    def test_raises(self):
        with pytest.raises(NotImplementedError):
            sequence.my_bioalign(None, None, method="other")
        with pytest.raises(NotImplementedError):
            sequence.my_bioalign(None, None, argstuple=(-2, 1))

class Test_alignment_result_to_list_of_dicts(unittest.TestCase):

    def test_works_small(self):
        top = md.load(test_filenames.small_monomer).top
        seq0 = sequence.top2seq(top)
        seq1 = seq0[::-1]+seq0.replace("I","A")+seq0[::-1]
        seq0_idxs = _np.arange(len(seq0))
        seq1_idxs = _np.arange(len(seq1))
        # We know the result a prioriy
        res0 = "".join(["-" for __ in seq0])+seq0+"".join(["-" for __ in seq0])

        ialg = sequence.my_bioalign(seq0, seq1)[0]

        result = sequence.alignment_result_to_list_of_dicts(ialg,
                                                            top,
                                                            seq0_idxs,
                                                            seq1_idxs,
                                                            verbose=True
                                                            )
        df = _DF.from_dict(result)
        assert "resSeq_1" not in df.keys()

        _np.testing.assert_array_equal("".join(df["AA_0"].to_list()),
                                       res0)
        _np.testing.assert_array_equal("".join(df["AA_1"].to_list()),
                                       seq1)

        _np.testing.assert_array_equal([str(ii) for ii in df["resSeq_0"]],
                                       ["~" for __ in seq0]+[rr.resSeq for rr in top.residues]+["~" for __ in seq0])

        _np.testing.assert_array_equal(df["fullname_0"],
                                       ["~" for __ in seq0] + [str(rr) for rr in top.residues] + ["~" for __ in seq0])
        _np.testing.assert_equal(_np.unique(df["fullname_1"]),"~")

        _np.testing.assert_array_equal([str(ii) for ii in df["idx_0"]],
                                       ["~" for __ in seq0] + seq0_idxs.tolist() + ["~" for __ in seq0])
        _np.testing.assert_array_equal(df["idx_1"],
                                       seq1_idxs)

        match_seq0 = [True for __ in seq0]
        match_seq0[seq0.find("I")]=False
        match = [False for __ in seq0]+match_seq0+[False for __ in seq0]
        _np.testing.assert_array_equal(df["match"],
                                       match)
        _np.testing.assert_equal(_np.unique(df["fullname_1"]),"~")

    def test_works_3CAP_vs_1U19_just_runs(self):
        geom_3CAP = md.load(test_filenames.pdb_3CAP)
        geom_1U19 = md.load(test_filenames.pdb_1U19)

        frag_3CAP = get_fragments(geom_3CAP.top,
                                  verbose=True,
                                  atoms=True
                                  )[0]
        frag_1U19 = get_fragments(geom_1U19.top,
                                  verbose=False,
                                  atoms=True
                                  )[0]
        geom_3CAP = geom_3CAP.atom_slice(frag_3CAP)
        geom_1U19 = geom_1U19.atom_slice(frag_1U19)

        seq_3CAP = sequence.top2seq(geom_3CAP.top)
        seq_1U19 = sequence.top2seq(geom_1U19.top)

        ialg = sequence.my_bioalign(seq_3CAP,
                                     seq_1U19,
                                     )[0]

        sequence.alignment_result_to_list_of_dicts(ialg,
                                                   geom_3CAP.top,
                                                   _np.arange(geom_3CAP.n_residues),
                                                   _np.arange(geom_1U19.n_residues),
                                                   topology_1=geom_1U19.top,
                                                   #verbose=True
                                                   )

if __name__ == '__main__':
    unittest.main()
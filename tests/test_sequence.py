import unittest
import mdtraj as md
import numpy as _np
from mdciao.examples import filenames as test_filenames
from pandas import DataFrame as _DF
from mdciao.fragments import get_fragments
from mdciao.utils import sequence

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

class Test_align_tops(unittest.TestCase):

    # The actual alignment tests are happening at alignment_result_to_list_of_dicts,
    # see those tests for alignment

    def test_works_w_itself(self):
        top = md.load(test_filenames.small_monomer).top
        df = sequence.align_tops_or_seqs(top, top)[0]
        assert isinstance(df, _DF)
        assert isinstance(df, sequence.AlignmentDataFrame)
        assert df.alignment_score == top.n_residues
        #sequence.print_verbose_dataframe(df)

    def test_works_w_itself_list(self):
        top = md.load(test_filenames.small_monomer).top
        df = sequence.align_tops_or_seqs(top, top, return_DF=False)[0]
        assert isinstance(df, list)
        #print(df)

    def test_works_w_itself_strs(self):
        top = md.load(test_filenames.small_monomer).top
        str1 = sequence.top2seq(top)
        df = sequence.align_tops_or_seqs(str1, str1, return_DF=False)[0]
        assert isinstance(df, list)

    def test_works_w_dimer(self):
        top2 = md.load(test_filenames.small_dimer).top
        seq_1_res_idxs = [6,7,8,9,10,11]
        df = sequence.align_tops_or_seqs(top2, top2, seq_1_res_idxs=seq_1_res_idxs)[0]
        _np.testing.assert_array_equal(df[df["match"]==True]["idx_0"].to_list(),
                                       seq_1_res_idxs)
        _np.testing.assert_array_equal(df[df["match"]==True]["idx_1"].to_list(),
                                       seq_1_res_idxs)

    def test_substitions(self):
        top2 = md.load(test_filenames.small_monomer).top
        df = sequence.align_tops_or_seqs(top2, top2, substitutions={"E": "G", "X": "Y"})[0]
        _np.testing.assert_array_equal("GVWIGKYY", ''.join(df["AA_0"]))


class Test_maptops(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.top1 = md.load(test_filenames.pdb_3CAP).top
        cls.top2 = md.load(test_filenames.pdb_1U19).top

    def test_works(self):
        top12top2, top22top1 = sequence.maptops(self.top1, self.top2)
        for key, val in top12top2.items():
            self.top2.residue(val).code == self.top1.residue(key).code

        for key, val in top22top1.items():
            self.top1.residue(val).code == self.top2.residue(key).code
    def test_works_nonmatch(self):
        seq2 = list(sequence.top2seq(self.top2))
        seq2[10:13] = "LOL"
        seq2 = "".join(seq2)
        top12top2, top22top1 = sequence.maptops(self.top1,seq2)
        assert all([key not in top12top2.values() for key in [10,11,12]])
        # Now allow nonmatches
        top12top2, top22top1 = sequence.maptops(self.top1,seq2, allow_nonmatch=True)
        assert all([key in top12top2.keys() for key in [10,11,12]])

class Test_re_match_df(unittest.TestCase):

    def test_works(self):
        df = sequence.align_tops_or_seqs("AABBCC", "AADDCC")[0]
        df_re_matched = sequence.re_match_df(df)

        assert all(df_re_matched["match"].values)

    def test_detects_insertion(self):
        df = sequence.align_tops_or_seqs("AABBFFFFFCC", "AADDCC")[0]
        df_re_matched = sequence.re_match_df(df)
        print(df_re_matched)
        self.assertListEqual(df_re_matched["match"].values.tolist(),[True,True,False,False,False,False,False,False,False,True,True])

class Test_df2maps(unittest.TestCase):

    def test_works(self):
        df_good = sequence.align_tops_or_seqs("AABBCC", "AADDCC")[0]
        map1, map2 = sequence.df2maps(df_good)
        self.assertDictEqual(map1,map2)
        self.assertDictEqual(map1,{ii:ii for ii in range(len(map1))})

    def test_works_only_matches(self):
        df_good = sequence.align_tops_or_seqs("AABBCC", "AADDCC")[0]
        map1, map2 = sequence.df2maps(df_good, allow_nonmatch=False)
        self.assertDictEqual(map1,map2)
        self.assertDictEqual(map1,{0:0, 1:1, 4:4, 5:5})

    def test_detects_insertion(self):
        df_bad = sequence.align_tops_or_seqs("AABBFFFFFCC", "AADDCC")[0]
        map1, map2 = sequence.df2maps(df_bad)
        self.assertDictEqual(map1, {0: 0, 1: 1, 9: 4, 10: 5})
        self.assertDictEqual(map2, {0: 0, 1: 1, 4: 9, 5: 10})


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
        with self.assertRaises(NotImplementedError):
            sequence.my_bioalign(None, None, method="other")
        with self.assertRaises(NotImplementedError):
            sequence.my_bioalign(None, None, extend_gap_score=10)

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
                                                            seq0_idxs,
                                                            seq1_idxs,
                                                            topology_0=top,
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
                                                   _np.arange(geom_3CAP.n_residues),
                                                   _np.arange(geom_1U19.n_residues),
                                                   topology_0=geom_3CAP.top,
                                                   topology_1=geom_1U19.top,
                                                   #verbose=True
                                                   )

class Test_superpose_w_CA_align(unittest.TestCase):

    def test_works(self):
        geom_3CAP = md.load(test_filenames.pdb_3CAP)
        geom_1U19 = md.load(test_filenames.pdb_1U19)
        sequence.superpose_w_CA_align(geom_3CAP,geom_1U19,verbose=True)


class Test_AlignmentDataframe(unittest.TestCase):

    def test_just_works(self):
        #     Check https://pandas.pydata.org/pandas-docs/stable/development/extending.html#define-original-properties
        # for more info
        df = sequence.AlignmentDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]},alignment_score=1)
        assert df.alignment_score == 1
        assert df[["A", "B"]].alignment_score == 1

if __name__ == '__main__':
    unittest.main()
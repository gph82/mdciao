def test_in_what_N_fragments():
    from sofi_functions import in_what_N_fragments
    fragments = [[0, 20, 30, 40],
                 [2, 4, 31, 1000],
                 [10]]

    for fragment_idx, idx_query in enumerate([20, 4, 10]):
        assert len(in_what_N_fragments(idx_query, fragments)) == 1

    wrong_idx = 11
    assert len(in_what_N_fragments(wrong_idx, fragments)) == 1


def test_is_iterable():
    from sofi_functions import is_iterable
    assert is_iterable([]), is_iterable([])
    assert is_iterable("A")
    assert is_iterable("abc")
    assert is_iterable([9, 99, 999])
    assert not is_iterable(999)


def test_force_iterable():
    from sofi_functions import force_iterable
    assert len(force_iterable("A")) != 0
    assert len(force_iterable("abc")) != 0
    assert len(force_iterable([9, 99, 999])) != 0
    assert len(force_iterable(999)) != 0


def test_does_not_contain_strings():
    from sofi_functions import does_not_contain_strings
    assert does_not_contain_strings([])
    assert not does_not_contain_strings(["A"])
    assert not does_not_contain_strings(["a", "b", "c"])
    assert not does_not_contain_strings(["A", "b", "c"])
    assert not does_not_contain_strings([[1], "ABC"])
    assert does_not_contain_strings([9, 99, 999])
    assert does_not_contain_strings([9])


def test_in_what_fragment():
    from sofi_functions import in_what_fragment

    # Check that it returns the right fragments
    assert in_what_fragment(1, [[1, 2], [3, 4, 5, 6.6]], ["A", "B"]) == 'A'

    # Easiest test
    assert in_what_fragment(1, [[1, 2], [3, 4, 5, 6.6]]) == 0

    # Check that it fails when input is not an index
    failed_assertion = False
    try:
        in_what_fragment([1],[[1, 2], [3, 4, 5, 6.6]])
    except AssertionError as e:
        failed_assertion = True
    assert failed_assertion

    # Check that it fails when strings are there
    failed_assertion = False
    try:
        in_what_fragment([1], [[1, 2], [3, 4, 5, 6.6, "A"]])
    except AssertionError as __:
        failed_assertion = True
    assert failed_assertion


def test_unique_list_of_iterables_by_tuple_hashing():
    from sofi_functions import  unique_list_of_iterables_by_tuple_hashing
    import numpy as _np
    assert (unique_list_of_iterables_by_tuple_hashing([1])) == [1]
    assert (unique_list_of_iterables_by_tuple_hashing([1], return_idxs=True)) == [0]
    assert (unique_list_of_iterables_by_tuple_hashing([[1], [1], [2], [2]], return_idxs=True)) == [[0], [2]]

    assert (unique_list_of_iterables_by_tuple_hashing([[1, 2], [3, 4], 1]) ==
            unique_list_of_iterables_by_tuple_hashing([[1, 2], [3, 4], _np.array(1)]))

    assert (unique_list_of_iterables_by_tuple_hashing([["A"], ["B"]]) ==
            unique_list_of_iterables_by_tuple_hashing([["A"], ["B"]]))

    assert not (unique_list_of_iterables_by_tuple_hashing([[1, 2], [3, 4]]) ==
                unique_list_of_iterables_by_tuple_hashing([[2, 1], [3, 4]]))
    assert not (unique_list_of_iterables_by_tuple_hashing([["ABC"], ["BCD"]]) ==
                unique_list_of_iterables_by_tuple_hashing([["BCD"], ["ABC"]]))


def test_exclude_same_fragments_from_residx_pairlist():
    from sofi_functions import exclude_same_fragments_from_residx_pairlist
    assert (exclude_same_fragments_from_residx_pairlist([[0, 1], [2, 3]], [[0, 1, 2], [3, 4]]) == [[2, 3]])
    assert (exclude_same_fragments_from_residx_pairlist([[1, 2], [0, 3], [5, 6]], [[0, 1, 2], [3, 4], [5, 6]],
                                                        return_excluded_idxs=True)
            == [0, 2])


def test_get_fragments_method():
    import mdtraj as md
    from sofi_functions import get_fragments
    import numpy as _np
    geom = md.load('/Users/sofitiwari/work_Charite/PDB/file_for_test.pdb')

    # Checking for "method" argument
    by_resSeq = get_fragments(geom.top,
                              verbose=True,
                              auto_fragment_names=True,
                              method='resSeq')
    by_bonds = get_fragments(geom.top,
                             verbose=True,
                             auto_fragment_names=True,
                             method='bonds')

    assert _np.allclose(by_resSeq[0], [0, 1, 2])
    assert _np.allclose(by_resSeq[1], [3, 4, 5])
    assert _np.allclose(by_resSeq[2], [6, 7])

    assert _np.allclose(by_bonds[0], [0, 1, 2])
    assert _np.allclose(by_bonds[1], [3, 4, 5])
    assert _np.allclose(by_bonds[2], [6])
    assert _np.allclose(by_bonds[3], [7])

def test_get_fragments_join_fragments_normal():
    import mdtraj as md
    from sofi_functions import get_fragments
    import numpy as _np
    geom = md.load('/Users/sofitiwari/work_Charite/PDB/file_for_test.pdb')
    # Checking if redundant fragment ids are removed from the inner list for the argument "join_fragments"
    by_bonds = get_fragments(geom.top,
                             join_fragments=[[1, 2]],
                             verbose=True,
                             auto_fragment_names=True,
                             method='bonds')

    assert _np.allclose(by_bonds[0], [3, 4, 5, 6])
    assert _np.allclose(by_bonds[1], [0, 1, 2])
    assert _np.allclose(by_bonds[2], [7])

def test_get_fragments_join_fragments_special_cases():
    import mdtraj as md
    from sofi_functions import get_fragments
    import numpy as _np
    geom = md.load('/Users/sofitiwari/work_Charite/PDB/file_for_test.pdb')
    # Checking if redundant fragment ids are removed from the inner list for the argument "join_fragments"
    by_bonds = get_fragments(geom.top,
                             join_fragments=[[1, 2, 2]],
                             verbose=True,
                             auto_fragment_names=True,
                             method='bonds')

    assert _np.allclose(by_bonds[0], [3, 4, 5, 6])
    assert _np.allclose(by_bonds[1], [0, 1, 2])
    assert _np.allclose(by_bonds[2], [7])

    # Checking for error from the overlapping ids in the argument "join_fragments"
    failed_assertion = False
    try:
        get_fragments(geom.top,
                      join_fragments=[[1, 2], [2, 3]],
                      verbose=True,
                      auto_fragment_names=True,
                      method='bonds')
    except AssertionError:
        failed_assertion = True
    assert failed_assertion

def test_get_fragments_break_fragments_just_works():
    import mdtraj as md
    from sofi_functions import get_fragments
    import numpy as _np
    geom = md.load('/Users/sofitiwari/work_Charite/PDB/file_for_test.pdb')
    # Checking if the fragments are breaking correctly for the argument "fragment_breaker_fullresname"
    by_bonds = get_fragments(geom.top,
                             fragment_breaker_fullresname=["VAL31", "GLU27"],  # two fragment breakers passed
                             verbose=True,
                             # auto_fragment_names=True,
                             method='bonds')
    assert _np.allclose(by_bonds[0], [0])
    assert _np.allclose(by_bonds[1], [1, 2])
    assert _np.allclose(by_bonds[2], [3])
    assert _np.allclose(by_bonds[3], [4, 5])
    assert _np.allclose(by_bonds[4], [6])
    assert _np.allclose(by_bonds[5], [7])

def test_get_fragments_break_fragments_special_cases():
    import mdtraj as md
    from sofi_functions import get_fragments
    import numpy as _np
    geom = md.load('/Users/sofitiwari/work_Charite/PDB/file_for_test.pdb')
    # No new fragments are created if an existing fragment breaker is passed
    by_bonds = get_fragments(geom.top,
                             fragment_breaker_fullresname=["GLU30"],  # GLU30 is already a fragment breaker
                             verbose=True,
                             # auto_fragment_names=True,
                             method='bonds')
    assert _np.allclose(by_bonds[0], [0, 1, 2])
    assert _np.allclose(by_bonds[1], [3, 4, 5])
    assert _np.allclose(by_bonds[2], [6])
    assert _np.allclose(by_bonds[3], [7])

    # Also works if input is a string instead of an iterable of strings
    by_bonds = get_fragments(geom.top,
                             fragment_breaker_fullresname="GLU30",  # GLU30 is already a fragment breaker
                             verbose=True,
                             # auto_fragment_names=True,
                             method='bonds')
    assert _np.allclose(by_bonds[0], [0, 1, 2])
    assert _np.allclose(by_bonds[1], [3, 4, 5])
    assert _np.allclose(by_bonds[2], [6])
    assert _np.allclose(by_bonds[3], [7])

    # No new fragments are created if residue id is not present anywhere
    by_bonds = get_fragments(geom.top,
                             fragment_breaker_fullresname=["Glu30"],  # not a valid id
                             verbose=True,
                             # auto_fragment_names=True,
                             method='bonds')
    assert _np.allclose(by_bonds[0], [0, 1, 2])
    assert _np.allclose(by_bonds[1], [3, 4, 5])
    assert _np.allclose(by_bonds[2], [6])
    assert _np.allclose(by_bonds[3], [7])


def test_interactive_fragment_picker_by_AAresSeq_no_ambiguous():
    import mdtraj as md
    from sofi_functions import get_fragments, interactive_fragment_picker_by_AAresSeq
    residues = ["GLU30", "GDP382"]

    # NO AMBIGUOUS definition i.e. each residue is present in only one fragment
    geom = md.load("/Users/sofitiwari/work_Charite/PDB/file_for_test.pdb")
    by_bonds = get_fragments(geom.top,
                                            verbose=True,
                                            auto_fragment_names=True,
                                            method='bonds')
    resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, by_bonds,
                                                                                             geom.top)

    # Checking if residue names gives the correct corresponding residue id
    assert (resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
    assert (resname2residx["GDP382"]) == 7  # GDP382 is the 8th residue

    # Checking if the residue name give the correct corresponding fragment id
    assert resname2fragidx["GLU30"] == 0  # GLU30 is in the 1st fragment
    assert resname2fragidx["GDP382"] == 3  # GDP382 is in the 4th fragment

def test_interactive_fragment_picker_by_AAresSeq_not_present():
    import mdtraj as md
    from sofi_functions import get_fragments, interactive_fragment_picker_by_AAresSeq
    residues = ["Glu30"]

    # NO AMBIGUOUS definition i.e. each residue is present in only one fragment
    geom = md.load("/Users/sofitiwari/work_Charite/PDB/file_for_test.pdb")
    by_bonds = get_fragments(geom.top,
                             verbose=True,
                             auto_fragment_names=True,
                             method='bonds')
    resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, by_bonds,
                                                                              geom.top)

    assert (resname2residx["Glu30"] == None)
    assert (resname2fragidx["Glu30"] == None)


def test_interactive_fragment_picker_by_AAresSeq_pick_first_fragment():
    import mdtraj as md
    from sofi_functions import get_fragments, interactive_fragment_picker_by_AAresSeq
    residues = ["GLU30"]

    # AMBIGUOUS definition i.e. each residue is present in multiple fragments
    geom = md.load("/Users/sofitiwari/work_Charite/PDB/file_for_test_repeated_fullresnames.pdb")
    by_bonds = get_fragments(geom.top,
                             verbose=True,
                             auto_fragment_names=True,
                             method='bonds')
    resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, by_bonds,
                                                                              geom.top,pick_first_fragment_by_default=True)

    assert(resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
    assert resname2fragidx["GLU30"] == 0  # GLU30 is in the 1st fragment


def test_interactive_fragment_picker_by_AAresSeq_pick_last_answer():
    import mdtraj as md
    from sofi_functions import get_fragments, interactive_fragment_picker_by_AAresSeq
    residues = ["GLU30","VAL31"]

    # AMBIGUOUS definition i.e. each residue is present in multiple fragments
    geom = md.load("/Users/sofitiwari/work_Charite/PDB/file_for_test_repeated_fullresnames.pdb")
    by_bonds = get_fragments(geom.top,
                             verbose=True,
                             auto_fragment_names=True,
                             method='bonds')
    resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, by_bonds,
                                                                              geom.top)
    # Checking if residue names gives the correct corresponding residue id
    # NOTE:Just press Return for GLU30 when asked "input one fragment idx"
    # NOTE:Just press Return for VAL31, when asked to "input one fragment idx"

    assert(resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
    assert (resname2fragidx["GLU30"]) == 0  # GLU30 is in the 1st fragment

    assert(resname2residx["VAL31"]) == 1  # VAL31 is the 1st residue
    assert (resname2fragidx["VAL31"]) == 0  # VAL31 is in the 1st fragment

def test_interactive_fragment_picker_by_AAresSeq_bad_answer():
    import mdtraj as md
    from sofi_functions import get_fragments, interactive_fragment_picker_by_AAresSeq
    residues = ["GLU30"]

    # AMBIGUOUS definition i.e. each residue is present in multiple fragments
    geom = md.load("/Users/sofitiwari/work_Charite/PDB/file_for_test_repeated_fullresnames.pdb")
    by_bonds = get_fragments(geom.top,
                             verbose=True,
                             auto_fragment_names=True,
                             method='bonds')

    failed_assertion = False
    try:
        resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, by_bonds,
                                                                                  geom.top)
    except (ValueError, AssertionError) as e:
        failed_assertion = True
    assert failed_assertion

    failed_assertion = False
    try:
        resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, by_bonds,
                                                                                  geom.top)
    except (ValueError, AssertionError) as e:
        failed_assertion = True
    assert failed_assertion

def test_interactive_fragment_picker_by_AAresSeq_fragment_name():
    import mdtraj as md
    from sofi_functions import get_fragments, interactive_fragment_picker_by_AAresSeq
    residues = ["GLU30"]

    # AMBIGUOUS definition i.e. each residue is present in multiple fragments
    geom = md.load("/Users/sofitiwari/work_Charite/PDB/file_for_test_repeated_fullresnames.pdb")
    by_bonds = get_fragments(geom.top,
                             verbose=True,
                             auto_fragment_names=True,
                             method='bonds')
    resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, by_bonds,
                                                                              geom.top,
                                                                              fragment_names=["A","B","C","D","E","F","G","H"])
    # Checking if residue names gives the correct corresponding residue id
    # NOTE:Enter 0 for GLU30 when asked "input one fragment idx"

    assert(resname2residx["GLU30"]) == 0  # GLU30 is the 1st residue
    assert resname2fragidx["GLU30"] == 0  # GLU30 is in the 1st fragment

def test_interactive_fragment_picker_by_AAresSeq_ambiguous():
    import mdtraj as md
    import numpy as _np
    from sofi_functions import get_fragments, interactive_fragment_picker_by_AAresSeq
    residues = ["GLU30", "GDP382"]

    # AMBIGUOUS definition i.e. residues appear in multiple fragments
    geom = md.load("/Users/sofitiwari/work_Charite/PDB/file_for_test_repeated_fullresnames.pdb")
    by_bonds = get_fragments(geom.top,
                        verbose=True,
                        auto_fragment_names=True,
                        method='bonds')

    resname2residx, resname2fragidx = interactive_fragment_picker_by_AAresSeq(residues, by_bonds,geom.top)


    # Checking if residue names gives the correct corresponding residue id
    # NOTE:Enter 4 for GLU30 when asked "input one fragment idx"
    # NOTE:Enter 3 for GDP382, when asked to "input one fragment idx"

    assert (resname2residx["GLU30"]) == 8  # GLU30 is the 9th residue
    assert (resname2residx["GDP382"]) == 7  # GDP382 is the 8th residue

    assert (resname2fragidx["GLU30"]) == 4  # Same as entered explicitly
    assert (resname2fragidx["GDP382"]) == 3 # Same as entered explicitly


def test_find_by_AA():
    import mdtraj as md
    from sofi_functions import find_AA
    geom = md.load("/Users/sofitiwari/work_Charite/PDB/file_for_test.pdb")

    assert (find_AA(geom.top, "GLU30")) == [0]
    assert (find_AA(geom.top, "LYS28")) == [5]
    assert (find_AA(geom.top, "lys20")) == []   # small case won't give any result

    assert (find_AA(geom.top, 'E30')) == [0]
    assert (find_AA(geom.top, 'W32')) == [2]
    assert (find_AA(geom.top, 'w32')) == []    # small case won't give any result
    assert (find_AA(geom.top, 'w 32')) == []   # spaces between characters won't work

    failed_assertion = False
    try:
        find_AA(geom.top, "GLUTAMINE")
    except ValueError as __:
        failed_assertion = True
    assert failed_assertion

    # AMBIGUOUS definition i.e. each residue is present in multiple fragments
    geom = md.load("/Users/sofitiwari/work_Charite/PDB/file_for_test_repeated_fullresnames.pdb")
    assert (find_AA(geom.top, "LYS28")) == [5, 13] # getting multiple idxs,as expected
    assert (find_AA(geom.top, "K28")) == [5, 13]


def test_top2residue_bond_matrix():
    import mdtraj as md
    import numpy as _np
    from sofi_functions import top2residue_bond_matrix
    geom = md.load("/Users/sofitiwari/work_Charite/PDB/file_for_test.pdb")

    res_bond_matrix = _np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0],
                                [0, 1, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0, 0],
                                [0, 0, 0, 1, 1, 1, 0, 0],
                                [0, 0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]])

    assert (top2residue_bond_matrix(geom.top) == res_bond_matrix).all()










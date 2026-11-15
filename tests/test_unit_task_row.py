import pytest

from kernel_parallel_records import CheckRecipe, ParallelLimits, UnitContract, UnitSpec
from kernel_unit_task import build_public_row
from kernel_parallel_projection import render_unit_task
from kernel_public_evidence import declared_public_check_paths


def make_spec(owns, creates, check):
    contract = UnitContract('a', 'make result', owns, creates, (), (), (), (), check)
    visible = tuple(sorted((*owns, *creates, *check.source_paths)))
    return UnitSpec('call', 'a', 'execution', 'seed', contract, (), (), visible,
                    (), '', 0, 0, 'manifest', 'policy', ParallelLimits())


def test_owned_sources_and_check_paths_are_separate_from_selectors():
    check = CheckRecipe(('tests/test_a.py',), ('tests/test_a.py::test_value',), 'pytest')
    spec = make_spec(('pkg/a.py', 'pkg/b.py'), ('pkg/new.py',), check)
    row = build_public_row(spec, {
        'pkg/a.py': b'first\n', 'pkg/b.py': b'second\n',
        'tests/test_a.py': b'def test_value(): pass\n',
    })
    assert row['files'] == {'pkg/a.py': {'parent_src': 'first\n'},
                            'pkg/b.py': {'parent_src': 'second\n'}}
    assert row['entry'] == {'path': 'pkg/a.py', 'fn': ''}
    assert row['public_check_paths'] == ['tests/test_a.py']
    assert row['check_selectors'] == ['tests/test_a.py::test_value']
    assert row['public_tests'] == {'tests/test_a.py': 'def test_value(): pass\n'}
    assert row['new_file_roots'] == ['pkg/new.py']
    assert row['scoped_edits'] is True
    assert row['msg'] == render_unit_task(spec)


def test_create_only_row_has_authenticated_absence_and_no_fabricated_test():
    spec = make_spec((), ('pkg/new.py',), CheckRecipe((), (), 'none'))
    row = build_public_row(spec, {})
    assert row['files'] == {}
    assert row['entry'] == {'create_path': 'pkg/new.py', 'absent_paths': ['pkg/new.py']}
    assert row['public_tests'] == {}
    assert 'public_check_paths' not in row
    assert declared_public_check_paths(row) is None
    with pytest.raises(ValueError, match='requires nonempty paths'):
        declared_public_check_paths({**row, 'public_check_paths': []})
    assert row['check_selectors'] == []
    with pytest.raises(ValueError, match='already exists'):
        build_public_row(spec, {'pkg/new.py': b'existing'})


def test_no_check_owned_row_reaches_public_evidence_without_fabricated_check():
    spec = make_spec(('pkg/a.py',), (), CheckRecipe((), (), 'none'))
    row = build_public_row(spec, {'pkg/a.py': b'VALUE = 1\n'})
    assert row['files'] == {'pkg/a.py': {'parent_src': 'VALUE = 1\n'}}
    assert row['public_tests'] == {}
    assert row['check_selectors'] == []
    assert 'public_check_paths' not in row
    assert declared_public_check_paths(row) is None


def test_missing_owned_bytes_refuse_instead_of_silently_omitting_source():
    spec = make_spec(('pkg/a.py', 'pkg/b.py'), (), CheckRecipe((), (), 'none'))
    with pytest.raises(ValueError, match='missing pkg/b.py'):
        build_public_row(spec, {'pkg/a.py': b'one'})


def test_builder_rejects_normalized_non_test_check_source_path():
    check = CheckRecipe(('pkg/a.py',), ('pkg/a.py::test_value',), 'pytest')
    spec = make_spec(('pkg/a.py',), (), check)

    with pytest.raises(ValueError, match='INVALID_PUBLIC_TEST_PATH: check source is not a pytest test file'):
        build_public_row(spec, {'pkg/a.py': b'def test_value(): pass\n'})

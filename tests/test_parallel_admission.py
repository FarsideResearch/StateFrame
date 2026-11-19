import os
from pathlib import Path

import pytest

from kernel_parallel_admission import DirectorySnapshot, admit, rescan_proposal, scan_coupling
from kernel_parallel_contracts import canonical_signature
from kernel_parallel_records import (AdmissionResult, CheckRecipe, ConsumeSpec, ExportSpec, ParallelLimits,
                                     ParallelRequest, UnitContract, format_parallel_limit)


def snap(tmp_path, files):
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    return DirectorySnapshot(tmp_path, 3)


def unit(name, path, *, creates=(), reads=(), exports=(), consumes=(), tests=()):
    return UnitContract(name, 'result', (path,) if path else (), creates, exports, consumes, reads, (),
                        CheckRecipe((), tests, 'pytest' if tests else 'none'))


def request(*units):
    return ParallelRequest('call', units, '2')


def admitted(req, snapshot, limits=None, **kwargs):
    return admit(req, snapshot, (), limits or ParallelLimits(), protected=kwargs.get('protected', lambda _: False),
                 case_insensitive=kwargs.get('case_insensitive', False))


def limit_detail(name, value):
    return format_parallel_limit(name, value)


def test_admit_dependency_order_and_exact_cycle_members(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'def a(): pass\n', 'b.py': 'def b(): pass\n', 'c.py': 'def c(): pass\n'})
    sig = canonical_signature('def a(): pass', 'a')
    a = unit('a', 'a.py', exports=(ExportSpec('a', 'a.py', 'a', sig),))
    b = unit('b', 'b.py', consumes=(ConsumeSpec('a', None, None),), exports=(ExportSpec('b', 'b.py', 'b', sig),))
    c = unit('c', 'c.py', consumes=(ConsumeSpec('b', None, None),))
    result = admitted(request(c, b, a), snapshot)
    assert result.status == 'ADMITTED'
    assert result.order == ('a', 'b', 'c')
    assert [(edge.src_unit, edge.dst_unit) for edge in result.edges] == [('a', 'b'), ('b', 'c')]
    a_cycle = unit('a', 'a.py', consumes=(ConsumeSpec('b', None, None),),
                   exports=(ExportSpec('a', 'a.py', 'a', sig),))
    result = admitted(request(a_cycle, b, c), snapshot)
    assert result.refusal.reason == 'DEPENDENCY_CYCLE'
    assert result.refusal.members == ('a', 'b')


@pytest.mark.parametrize('expected,alter', [
    ('TOO_FEW_UNITS', lambda units: units[:1]),
    ('TOO_MANY_UNITS', lambda units: units * 4),
    ('OWNERSHIP_OVERLAP', lambda units: (units[0], unit('b', 'a.py'))),
    ('UNRESOLVED_CONSUME', lambda units: (unit('a', 'a.py', consumes=(ConsumeSpec('unknown', None, None),)), units[1])),
    ('EXPORT_OUTSIDE_OWNERSHIP', lambda units: (unit('a', 'a.py', exports=(ExportSpec('x', 'b.py', 'x', None),)), units[1])),
    ('DUPLICATE_EXPORT', lambda units: (unit('a', 'a.py', exports=(ExportSpec('x', 'a.py', 'f', None), ExportSpec('x', 'a.py', 'g', None))), units[1])),
    ('READ_NOT_ALLOWED', lambda units: (unit('a', 'a.py', reads=('b.py',)), units[1])),
    ('UNSUPPORTED_CONTRACT', lambda units: (unit('a', 'a.js', exports=(ExportSpec('x', 'a.js', 'f', None),)), unit('b', 'b.py', consumes=(ConsumeSpec('x', None, None),)))),
])
def test_refusal_reasons_have_exact_values(tmp_path, expected, alter):
    snapshot = snap(tmp_path, {'a.py': 'def f(): pass\n', 'b.py': 'def g(): pass\n', 'a.js': 'function f() {}\n'})
    base = (unit('a', 'a.py'), unit('b', 'b.py'))
    result = admitted(request(*alter(base)), snapshot)
    assert result.status == 'REFUSED'
    assert result.refusal.reason == expected
    assert result.edges == ()


def test_omitted_python_export_signatures_admit_existing_and_created_symbols(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'def f(): pass\n', 'b.py': 'def g(): pass\n'})
    existing = unit('a', 'a.py', exports=(ExportSpec('f', 'a.py', 'f', None),))
    created = unit('b', 'b.py', exports=(ExportSpec('new', 'b.py', 'new', None),))
    result = admitted(request(existing, created), snapshot)
    assert result.status == 'ADMITTED'
    assert result.refusal is None
    assert result.order == ('a', 'b')


def test_unsafe_paths_aliases_and_caps(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'a = 1\n', 'b.py': 'b = 1\n'})
    b = unit('b', 'b.py')
    for path in ('../a.py', '/tmp/a.py', '.git/config', 'venv/file.py', 'bad//path'):
        assert admitted(request(unit('a', path), b), snapshot).refusal.reason == 'UNSAFE_PATH'
    assert admitted(request(unit('a', 'a.py'), b), snapshot,
                    protected=lambda path: path == 'a.py').refusal.reason == 'UNSAFE_PATH'
    (tmp_path / 'link.py').symlink_to(tmp_path / 'a.py')
    assert admitted(request(unit('a', 'link.py'), b), snapshot).refusal.reason == 'UNSAFE_PATH'
    os.link(tmp_path / 'a.py', tmp_path / 'hard.py')
    assert admitted(request(unit('a', 'a.py'), b), snapshot).refusal.reason == 'UNSAFE_PATH'
    (tmp_path / 'hard.py').unlink()
    assert admitted(request(unit('a', 'A.py'), unit('b', 'a.py')), snapshot,
                    case_insensitive=True).refusal.reason in ('UNSAFE_PATH', 'OWNERSHIP_OVERLAP')
    assert admitted(request(unit('a', 'a.py'), b), snapshot,
                    ParallelLimits(max_paths_total=1)).refusal.reason == 'TOO_MANY_PATHS'
    assert admitted(request(unit('a', 'a.py'), b), snapshot,
                    ParallelLimits(max_paths_total=1)).refusal.detail == limit_detail('max_paths_total', 1)
    assert admitted(request(unit('a', 'a.py'), b), snapshot,
                    ParallelLimits(max_owned_source_bytes=1)).refusal.reason == 'UNIT_CONTEXT_TOO_LARGE'
    assert admitted(request(unit('a', 'a.py'), b), snapshot,
                    ParallelLimits(max_owned_source_bytes=1)).refusal.detail == (
                        f'owned_source_bytes={len(snapshot.read_bytes("a.py"))}; '
                        + limit_detail('max_owned_source_bytes', 1))


def test_create_with_isolated_missing_ancestors_is_admitted(tmp_path):
    snapshot = snap(tmp_path, {'b.py': 'pass\n'})
    create_path = 'missing/sub/file.py'
    result = admitted(request(unit('a', None, creates=(create_path,)), unit('b', 'b.py')), snapshot)
    assert result.status == 'ADMITTED'
    assert result.refusal is None
    assert not (tmp_path / 'missing').exists()


def test_existing_create_target_is_refused_but_existing_owned_file_is_admitted(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'VALUE = 1\n', 'b.py': 'VALUE = 2\n'})

    create_result = admitted(
        request(unit('a', None, creates=('a.py',)), unit('b', 'b.py')), snapshot)
    own_result = admitted(
        request(unit('a', 'a.py'), unit('b', 'b.py')), snapshot)

    assert (create_result.status, create_result.refusal.reason) == ('REFUSED', 'CREATE_EXISTS')
    assert create_result.refusal.paths == ('a.py',)
    assert own_result.status == 'ADMITTED'
    assert own_result.refusal is None


def test_existing_create_target_is_refused_and_must_be_declared_owned(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'def existing(): pass\n', 'b.py': 'def other(): pass\n'})
    req = request(unit('a', None, creates=('a.py',)), unit('b', 'b.py'))

    result = admitted(req, snapshot)

    assert result.status == 'REFUSED'
    assert result.refusal.reason == 'CREATE_EXISTS'
    assert result.refusal.paths == ('a.py',)
    assert result.adjustments == ()
    assert AdmissionResult.from_json(result.to_json()) == result


def test_protected_existing_create_target_is_refused_without_dropping_unit(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'VALUE = 1\n', 'b.py': 'VALUE = 2\n'})
    result = admitted(request(unit('a', None, creates=('a.py',)), unit('b', 'b.py')),
                      snapshot, protected=lambda path: path == 'a.py')

    assert result.status == 'REFUSED'
    assert result.refusal.reason == 'UNSAFE_PATH'
    assert result.refusal.paths == ('a.py',)
    assert result.adjustments == ()


def test_existing_create_target_is_refused_before_owned_source_budgeting(tmp_path):
    snapshot = snap(tmp_path, {
        'a1.py': 'x' * 10, 'a2.py': 'y' * 10, 'b.py': 'z',
    })
    create_both = unit('a', None, creates=('a1.py', 'a2.py'))

    result = admitted(request(create_both, unit('b', 'b.py')), snapshot,
                      ParallelLimits(max_owned_source_bytes=15))

    assert result.status == 'REFUSED'
    assert result.refusal.reason == 'CREATE_EXISTS'
    assert result.refusal.paths == ('a1.py',)
    assert result.adjustments == ()


def test_existing_create_symlink_is_refused_as_unsafe(tmp_path):
    snapshot = snap(tmp_path, {'b.py': 'VALUE = 2\n'})
    outside = tmp_path.parent / 'outside.py'
    outside.write_text('PRIVATE = True\n')
    (tmp_path / 'a.py').symlink_to(outside)

    result = admitted(request(unit('a', None, creates=('a.py',)), unit('b', 'b.py')), snapshot)

    assert result.status == 'REFUSED'
    assert result.refusal.reason == 'UNSAFE_PATH'
    assert result.refusal.paths == ('a.py',)
    assert 'symlink' in result.refusal.detail


def test_existing_create_with_same_unit_read_is_still_refused_as_create_collision(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'VALUE = 1\n', 'b.py': 'VALUE = 2\n'})

    result = admitted(
        request(unit('a', None, creates=('a.py',), reads=('a.py',)), unit('b', 'b.py')),
        snapshot)

    assert result.status == 'REFUSED'
    assert result.refusal.reason == 'CREATE_EXISTS'
    assert result.refusal.paths == ('a.py',)


def test_unsafe_create_path_does_not_probe_outside_the_workspace(tmp_path, monkeypatch):
    snapshot = snap(tmp_path, {'b.py': 'VALUE = 2\n'})
    calls = []
    original_exists = snapshot.exists

    def checked_exists(path):
        calls.append(path)
        assert not path.startswith('../')
        return original_exists(path)

    monkeypatch.setattr(snapshot, 'exists', checked_exists)
    result = admitted(
        request(unit('a', None, creates=('../outside.py',)), unit('b', 'b.py')),
        snapshot)

    assert result.status == 'REFUSED'
    assert result.refusal.reason == 'UNSAFE_PATH'
    assert calls == []


def test_create_degrade_does_not_bypass_an_already_unmet_minimum(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'a', 'b.py': 'b'})

    result = admitted(
        request(unit('a', None, creates=('a.py',)), unit('b', 'b.py')),
        snapshot, ParallelLimits(min_units=3), protected=lambda path: path == 'a.py')

    assert result.status == 'REFUSED'
    assert result.refusal.reason == 'TOO_FEW_UNITS'
    assert result.adjustments == ()


def test_existing_create_collision_refuses_the_whole_plan_before_dependencies(tmp_path):
    snapshot = snap(tmp_path, {
        'a.py': 'def existing(): pass\n', 'b.py': 'def consumer(): pass\n',
        'c.py': 'def independent(): pass\n',
    })
    producer = UnitContract('a', 'create an interface', (), ('a.py',),
                            (ExportSpec('created', 'a.py', 'new_api', None),), (), (), (),
                            CheckRecipe((), (), 'none'))
    consumer = UnitContract('b', 'consume the planned interface', ('b.py',), (), (),
                            (ConsumeSpec('created', None, None),), (), (),
                            CheckRecipe((), (), 'none'))
    independent = unit('c', 'c.py')
    result = admitted(request(producer, consumer, independent), snapshot,
                      protected=lambda path: path == 'a.py')

    assert result.status == 'REFUSED'
    assert result.refusal.reason == 'UNSAFE_PATH'
    assert result.refusal.paths == ('a.py',)


@pytest.mark.parametrize('other_path_kind', ['owns', 'creates', 'reads', 'check_source'])
def test_create_with_missing_ancestor_used_by_another_subframe_is_refused(tmp_path, other_path_kind):
    snapshot = snap(tmp_path, {'b.py': 'pass\n'})
    create_path = 'missing/sub/file.py'
    other_path = 'missing/other.py'
    if other_path_kind == 'owns':
        other = unit('b', other_path)
    elif other_path_kind == 'creates':
        other = unit('b', None, creates=(other_path,))
    elif other_path_kind == 'reads':
        other = unit('b', 'b.py', reads=(other_path,))
    else:
        other = UnitContract('b', 'result', ('b.py',), (), (), (), (), (),
                             CheckRecipe((other_path,), (), 'none'))

    result = admitted(request(unit('a', None, creates=(create_path,)), other), snapshot)

    assert result.status == 'REFUSED'
    assert result.refusal.reason == 'UNSAFE_PATH'
    assert result.refusal.paths == (create_path,)
    assert create_path in result.refusal.detail
    assert other_path in result.refusal.detail
    assert 'missing ancestor' in result.refusal.detail
    assert not (tmp_path / 'missing').exists()


def test_create_with_missing_ancestor_used_as_another_subframe_target_is_refused(tmp_path):
    snapshot = snap(tmp_path, {'b.py': 'pass\n'})
    create_path = 'missing/sub/file.py'
    ancestor_target = 'missing'
    result = admitted(request(
        unit('a', None, creates=(create_path,)),
        unit('b', None, creates=(ancestor_target,)),
    ), snapshot)

    assert result.status == 'REFUSED'
    assert result.refusal.reason == 'UNSAFE_PATH'
    assert result.refusal.paths == (create_path,)
    assert 'missing ancestor missing' in result.refusal.detail
    assert ancestor_target in result.refusal.detail
    assert not (tmp_path / 'missing').exists()


def test_create_with_regular_file_ancestor_is_refused_with_reason(tmp_path):
    snapshot = snap(tmp_path, {'blocked': 'not a directory\n', 'b.py': 'pass\n'})
    create_path = 'blocked/sub/file.py'
    result = admitted(request(unit('a', None, creates=(create_path,)), unit('b', 'b.py')), snapshot)

    assert result.status == 'REFUSED'
    assert result.refusal.reason == 'UNSAFE_PATH'
    assert result.refusal.paths == (create_path,)
    assert create_path in result.refusal.detail
    assert 'blocked' in result.refusal.detail
    assert 'not a directory' in result.refusal.detail


@pytest.mark.parametrize(('ancestor_kind', 'reason'), [
    ('symlink', 'symlink'),
    ('fifo', 'special file'),
])
def test_create_with_symlink_or_special_file_ancestor_is_refused(tmp_path, ancestor_kind, reason):
    snap(tmp_path, {'b.py': 'pass\n'})
    ancestor = tmp_path / 'blocked'
    if ancestor_kind == 'symlink':
        target = tmp_path / 'outside'
        target.mkdir()
        ancestor.symlink_to(target, target_is_directory=True)
    else:
        os.mkfifo(ancestor)
    snapshot = DirectorySnapshot(tmp_path, 3)
    create_path = 'blocked/sub/file.py'

    result = admitted(request(unit('a', None, creates=(create_path,)), unit('b', 'b.py')), snapshot)

    assert result.status == 'REFUSED'
    assert result.refusal.reason == 'UNSAFE_PATH'
    assert result.refusal.paths == (create_path,)
    assert create_path in result.refusal.detail
    assert 'blocked' in result.refusal.detail
    assert reason in result.refusal.detail


def test_every_admission_cap_refusal_reports_its_config_source(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'a = 1\n', 'b.py': 'b = 1\n', 'c.py': 'c = 1\n',
                               'read.py': 'value = 1\n'})
    cases = (
        (request(unit('a', 'a.py')), ParallelLimits(min_units=3),
         'TOO_FEW_UNITS', limit_detail('min_units', 3)),
        (request(unit('a', 'a.py'), unit('b', 'b.py'), unit('c', 'c.py')),
         ParallelLimits(max_units=2), 'TOO_MANY_UNITS', limit_detail('max_units', 2)),
        (request(unit('a', 'a.py', creates=('new_one.py', 'new_two.py')), unit('b', 'b.py')),
         ParallelLimits(max_paths_per_unit=2), 'TOO_MANY_PATHS', limit_detail('max_paths_per_unit', 2)),
        (request(unit('a', 'a.py', reads=('b.py', 'read.py')), unit('b', 'b.py')),
         ParallelLimits(max_paths_per_unit=1), 'TOO_MANY_PATHS', limit_detail('max_paths_per_unit', 1)),
        (request(unit('a', 'a.py'), unit('b', 'b.py')), ParallelLimits(max_paths_total=1),
         'TOO_MANY_PATHS', limit_detail('max_paths_total', 1)),
        (request(unit('a', 'a.py', reads=('read.py',)), unit('b', 'b.py')),
         ParallelLimits(max_paths_total=2), 'TOO_MANY_PATHS', limit_detail('max_paths_total', 2)),
        (request(unit('a', 'a.py'), unit('b', 'b.py')), ParallelLimits(max_owned_source_bytes=4),
         'UNIT_CONTEXT_TOO_LARGE', 'owned_source_bytes=6; ' + limit_detail('max_owned_source_bytes', 4)),
    )
    for parallel_request, limits, reason, detail in cases:
        refusal = admitted(parallel_request, snapshot, limits).refusal
        assert refusal.reason == reason
        assert refusal.detail == detail


def test_existing_consume_ambiguity_and_missing(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'pass\n', 'b.py': 'pass\n', 'lib.py': 'def f(): pass\ndef f(x): pass\n'})
    result = admitted(request(unit('a', 'a.py', consumes=(ConsumeSpec(None, 'lib.py', 'f'),)),
                              unit('b', 'b.py')), snapshot)
    assert result.refusal.reason == 'AMBIGUOUS_CONSUME'
    result = admitted(request(unit('a', 'a.py', consumes=(ConsumeSpec(None, 'lib.py', 'nope'),)),
                              unit('b', 'b.py')), snapshot)
    assert result.refusal.reason == 'UNRESOLVED_CONSUME'


@pytest.mark.parametrize('producer,consumer', [
    ('pkg/a.py', 'import pkg.a\n'),
    ('pkg/a.py', 'from . import a\n'),
    ('src/pkg/a.py', 'from pkg import a\n'),
])
def test_static_import_coupling_refuses_with_evidence(tmp_path, producer, consumer):
    consumer_path = 'src/pkg/b.py' if producer.startswith('src/') else 'pkg/b.py'
    snapshot = snap(tmp_path, {producer: 'def f(): pass\n', consumer_path: consumer,
                               str(Path(producer).parent / '__init__.py'): ''})
    result = admitted(request(unit('a', producer), unit('b', consumer_path)), snapshot)
    assert result.refusal.reason == 'UNDECLARED_COUPLING'
    assert result.refusal.evidence[0].path == consumer_path
    assert result.refusal.evidence[0].line == 1


def test_dynamic_import_couples_by_scan_and_limits_are_unknown(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'def f(): pass\n', 'b.py': 'module = __import__("a")\n'})
    units = (unit('a', 'a.py'), unit('b', 'b.py'))
    edges, pairs = scan_coupling(units, snapshot, ParallelLimits())
    assert edges != []
    assert [pair.status for pair in pairs] != ['INDEPENDENT_BY_SCAN']
    edges, pairs = scan_coupling(units, snapshot, ParallelLimits(scan_max_bytes=1))
    assert edges == []
    assert [(pair.status, pair.reason) for pair in pairs] == [('UNKNOWN', 'BYTE_LIMIT')]


def test_rescan_detects_new_import_in_created_file(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'def f(): pass\n', 'b.py': 'pass\n'})
    units = (unit('a', 'a.py'), unit('b', 'b.py', creates=('new.py',)))
    admission = admitted(request(*units), snapshot)
    assert admission.status == 'ADMITTED'
    composed_root = tmp_path / 'composed'
    composed = snap(composed_root, {'a.py': 'def f(): pass\n', 'b.py': 'pass\n', 'new.py': 'import a\n'})
    result = rescan_proposal(units, admission, composed, 'b')
    assert result.refusal.reason == 'UNDECLARED_COUPLING'
    assert result.refusal.evidence[0].path == 'new.py'


def test_rescan_retains_new_unknown_coverage(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'def f(): pass\n', 'b.py': 'pass\n'})
    units = (unit('a', 'a.py'), unit('b', 'b.py', creates=('new.py',)))
    admission = admitted(request(*units), snapshot, ParallelLimits(scan_max_files=2))
    assert [(pair.status, pair.reason) for pair in admission.pairs] == [('INDEPENDENT_BY_SCAN', '')]
    composed = snap(tmp_path / 'composed', {'a.py': 'def f(): pass\n', 'b.py': 'pass\n', 'new.py': 'pass\n'})
    result = rescan_proposal(units, admission, composed, 'b')
    assert result.refusal is None
    assert [(pair.a, pair.b, pair.status, pair.reason) for pair in result.pairs] == [
        ('a', 'b', 'UNKNOWN', 'FILE_LIMIT')]
    (tmp_path / 'composed/new.py').write_text('def broken(:\n')
    parsed = DirectorySnapshot(tmp_path / 'composed', 4)
    roomy = admitted(request(*units), snapshot)
    result = rescan_proposal(units, roomy, parsed, 'b')
    assert result.refusal is None
    assert [(pair.status, pair.reason) for pair in result.pairs] == [
        ('UNKNOWN', 'UNPARSEABLE_PYTHON')]


def test_check_source_cannot_expose_sibling_or_unsafe_path(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'secret = 1\n', 'b.py': 'value = 1\n', 'check.py': 'pass\n'})
    a = unit('a', 'a.py')
    for path in ('a.py', '../a.py', 'missing.py'):
        b = UnitContract('b', 'result', ('b.py',), (), (), (), (), (),
                         CheckRecipe((path,), ('check.py::test_value',), 'pytest'))
        result = admitted(request(a, b), snapshot)
        assert (result.status, result.refusal.reason, result.refusal.paths) == (
            'REFUSED', 'READ_NOT_ALLOWED', (path,))
    b = UnitContract('b', 'result', ('b.py',), (), (), (), (), (),
                     CheckRecipe(('check.py',), ('check.py::test_value',), 'pytest'))
    assert admitted(request(a, b), snapshot).status == 'ADMITTED'
    aliased = UnitContract('b', 'result', ('b.py',), (), (), (), (), (),
                           CheckRecipe(('A.py',), ('check.py::test_value',), 'pytest'))
    result = admitted(request(a, aliased), snapshot, case_insensitive=True)
    assert (result.refusal.reason, result.refusal.paths) == ('READ_NOT_ALLOWED', ('A.py',))
    protected = admitted(request(a, b), snapshot, protected=lambda path: path == 'check.py')
    assert (protected.status, protected.refusal) == ('ADMITTED', None)


def test_protected_test_can_be_read_and_checked_but_not_owned(tmp_path):
    test_path = 'tests/test_checkout.py'
    snapshot = snap(tmp_path, {
        'a.py': 'value = 1\n', 'b.py': 'value = 2\n',
        test_path: 'def test_checkout():\n    assert 1 == 1\n',
    })
    def protected(path):
        return path == test_path
    a = UnitContract('a', 'result', ('a.py',), (), (), (), (test_path,), (),
                     CheckRecipe((test_path,), (test_path + '::test_checkout',), 'pytest'))
    b = unit('b', 'b.py')
    allowed = admitted(request(a, b), snapshot, protected=protected)
    assert (allowed.status, allowed.refusal, allowed.order) == ('ADMITTED', None, ('a', 'b'))

    owned = unit('a', test_path)
    refused = admitted(request(owned, b), snapshot, protected=protected)
    assert refused.status == 'REFUSED'
    assert (refused.refusal.reason, refused.refusal.paths) == ('UNSAFE_PATH', (test_path,))
    assert refused.refusal.detail == f'{test_path}: path is protected'


def test_oversized_repository_marks_pair_unknown(tmp_path):
    files = {'a.py': 'def f(): pass\n', 'b.py': 'pass\n'}
    files.update({f'unrelated_{index}.txt': 'data' for index in range(4)})
    snapshot = snap(tmp_path, files)
    edges, pairs = scan_coupling((unit('a', 'a.py'), unit('b', 'b.py')), snapshot,
                                 ParallelLimits(scan_max_files=3))
    assert edges == []
    assert [(pair.a, pair.b, pair.status, pair.reason) for pair in pairs] == [
        ('a', 'b', 'UNKNOWN', 'FILE_LIMIT')]


def test_unrelated_symlink_marks_coupling_unknown_without_reading_target(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + '_secret')
    outside.write_text('secret')
    snapshot = snap(tmp_path, {'a.py': 'pass\n', 'b.py': 'pass\n'})
    (tmp_path / 'foreign.py').symlink_to(outside)
    snapshot = DirectorySnapshot(tmp_path, 3)
    edges, pairs = scan_coupling((unit('a', 'a.py'), unit('b', 'b.py')), snapshot, ParallelLimits())
    assert edges == []
    assert [(pair.status, pair.reason) for pair in pairs] == [('UNKNOWN', 'UNSAFE_REPOSITORY_ENTRY')]


def test_identical_path_overlap_names_the_path_once(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'pass\n', 'b.py': 'pass\n'})
    result = admitted(request(unit('a', 'a.py'), unit('b', 'a.py')), snapshot)
    assert result.refusal.reason == 'OWNERSHIP_OVERLAP'
    assert result.refusal.paths == ('a.py',)
    # The refusal names both subframes so the model can see whose work to merge.
    assert result.refusal.members == ('a', 'b')


def test_casefold_and_ancestor_ownership_overlap(tmp_path):
    snapshot = snap(tmp_path, {'A.py': 'pass\n', 'a.py': 'pass\n', 'pkg/a.py': 'pass\n'})
    result = admitted(request(unit('a', 'A.py'), unit('b', 'a.py')), snapshot, case_insensitive=True)
    assert result.refusal.reason == 'OWNERSHIP_OVERLAP'
    assert result.refusal.paths == ('A.py', 'a.py')
    result = admitted(request(unit('a', 'pkg/a.py'), unit('b', None, creates=('pkg/a.py/nested.py',))), snapshot)
    assert result.refusal.reason == 'OWNERSHIP_OVERLAP'
    assert result.refusal.paths == ('pkg/a.py', 'pkg/a.py/nested.py')


def test_non_python_duplicate_definition_is_ambiguous(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'pass\n', 'b.py': 'pass\n',
                               'lib.js': 'function f() {}\nfunction f(x) {}\n'})
    result = admitted(request(unit('a', 'a.py', consumes=(ConsumeSpec(None, 'lib.js', 'f'),)),
                              unit('b', 'b.py')), snapshot)
    assert result.refusal.reason == 'AMBIGUOUS_CONSUME'


def test_declared_edge_does_not_hide_reverse_undeclared_import(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'import b\ndef a(): pass\n', 'b.py': 'def b(): pass\n'})
    sig = canonical_signature('def b(): pass', 'b')
    a = unit('a', 'a.py', exports=(ExportSpec('a_export', 'a.py', 'a', sig),))
    b = unit('b', 'b.py', consumes=(ConsumeSpec('a_export', None, None),))
    result = admitted(request(a, b), snapshot)
    assert result.refusal.reason == 'UNDECLARED_COUPLING'
    assert result.refusal.evidence[0].path == 'a.py'


def test_package_reexport_coupling_and_ambiguous_module_roots(tmp_path):
    snapshot = snap(tmp_path, {'pkg/impl.py': 'def f(): pass\n',
                               'pkg/__init__.py': 'from .impl import f\n',
                               'consumer.py': 'from pkg import f\n'})
    result = admitted(request(unit('a', 'pkg/impl.py'), unit('b', 'consumer.py')), snapshot)
    assert result.refusal.reason == 'UNDECLARED_COUPLING'
    assert result.refusal.evidence[0].path == 'consumer.py'
    snapshot = snap(tmp_path, {'src/pkg/impl.py': 'def f(): pass\n',
                               'src/pkg/__init__.py': 'from .impl import f\n'})
    _, pairs = scan_coupling((unit('a', 'pkg/impl.py'), unit('b', 'consumer.py')), snapshot, ParallelLimits())
    assert pairs[0].status == 'UNKNOWN'


def test_non_python_consume_requires_executable_recipe(tmp_path):
    snapshot = snap(tmp_path, {'a.js': 'function f() {}\n', 'b.py': 'pass\n'})
    export = ExportSpec('f', 'a.js', 'f', None)
    malformed = UnitContract('b', 'result', ('b.py',), (), (), (ConsumeSpec('f', None, None),), (), (),
                             CheckRecipe((), ('tests/test_b.py',), 'none'))
    assert admitted(request(unit('a', 'a.js', exports=(export,)), malformed), snapshot).refusal.reason == 'UNSUPPORTED_CONTRACT'
    blank = UnitContract('b', 'result', ('b.py',), (), (), (ConsumeSpec('f', None, None),), (), (),
                         CheckRecipe((), ('  ',), 'pytest'))
    assert admitted(request(unit('a', 'a.js', exports=(export,)), blank), snapshot).refusal.reason == 'UNSUPPORTED_CONTRACT'


def test_casefolded_existing_consume_cannot_alias_ownership(tmp_path):
    snapshot = snap(tmp_path, {'Foo.py': 'def f(): pass\n', 'foo.py': 'def f(): pass\n', 'b.py': 'pass\n'})
    result = admitted(request(unit('a', 'Foo.py'),
                              unit('b', 'b.py', consumes=(ConsumeSpec(None, 'foo.py', 'f'),))),
                      snapshot, case_insensitive=True)
    assert result.refusal.reason == 'UNRESOLVED_CONSUME'


def test_non_python_comments_do_not_define_symbols(tmp_path):
    snapshot = snap(tmp_path, {'a.py': 'pass\n', 'b.py': 'pass\n',
                               'lib.js': '// function fake() {}\nfunction real() {}\n'})
    result = admitted(request(unit('a', 'a.py', consumes=(ConsumeSpec(None, 'lib.js', 'fake'),)),
                              unit('b', 'b.py')), snapshot)
    assert result.refusal.reason in ('UNRESOLVED_CONSUME', 'AMBIGUOUS_CONSUME')
    assert admitted(request(unit('a', 'a.py', consumes=(ConsumeSpec(None, 'lib.js', 'real'),)),
                            unit('b', 'b.py')), snapshot).status == 'ADMITTED'


def test_limit_pairs_are_sorted_and_scan_checks_size_before_read(tmp_path, monkeypatch):
    snapshot = snap(tmp_path, {'a.py': 'pass\n', 'b.py': 'pass\n', 'c.py': 'pass\n'})
    units = (unit('c', 'c.py'), unit('a', 'a.py'), unit('b', 'b.py'))
    _, pairs = scan_coupling(units, snapshot, ParallelLimits(scan_max_files=1))
    assert [(p.a, p.b) for p in pairs] == [('a', 'b'), ('a', 'c'), ('b', 'c')]
    original = snapshot.read_bytes
    def guarded(path):
        if path == 'c.py':
            raise AssertionError('oversized file was read')
        return original(path)
    monkeypatch.setattr(snapshot, 'read_bytes', guarded)
    _, pairs = scan_coupling(units, snapshot, ParallelLimits(scan_max_bytes=10))
    assert [(p.status, p.reason) for p in pairs] == [('UNKNOWN', 'BYTE_LIMIT')] * 3


def test_directory_snapshot_streams_large_file_hash(tmp_path, monkeypatch):
    (tmp_path / 'large.bin').write_bytes(b'x' * (2 << 20))
    original = Path.read_bytes
    def guarded(self):
        if self.name == 'large.bin':
            raise AssertionError('whole-file read')
        return original(self)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    snapshot = DirectorySnapshot(tmp_path, 0)
    assert len(snapshot.manifest_digest) == 64

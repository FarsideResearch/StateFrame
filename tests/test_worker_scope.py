"""Host scope survives public projection and feeds actual file/effect guards."""
import pytest

from stateframe.task_row import build_row
from stateframe.workcopy import create
from kernel_information_policy import InformationPolicy
from kernel_effects import effect_policy_for_run, scan, diff
from kernel_file_ops import FileOperationRefused, file_move
from probes.react_repair import FileSet, _is_product_path, _product_source_paths


@pytest.mark.parametrize('scoped', [False, True])
def test_projected_row_keeps_host_scope_for_effects_and_capture(tmp_path, monkeypatch, scoped):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'app.py').write_text('VALUE = 1\n')
    (repo / 'other.py').write_text('OTHER = 1\n')
    (repo / 'tests').mkdir()
    (repo / 'tests/test_app.py').write_text('def test_ok(): assert True\n')
    work = create(repo, tmp_path / 'workspace')
    row = build_row(work, task='Generate a product file',
                    edit_paths=['app.py'] if scoped else None)
    row['fixed_src'] = 'private grading material'
    monkeypatch.setenv('INITIAL_ORACLE_SYMPTOM', 'OFF')
    projected = InformationPolicy(row).worker_row(row)
    assert projected['tier'] == 'USER-REPO'
    assert projected['scoped_edits'] is scoped
    assert 'fixed_src' not in projected
    before = scan(work.copy, exclude=lambda _: False)
    policy = effect_policy_for_run(work.copy, projected, {'tests/test_app.py'}, baseline_manifest=before)
    (work.copy / 'generated.py').write_text('GENERATED = 1\n')
    (work.copy / 'build.ok').write_text('built\n')
    (work.copy / 'other.py').write_text('OTHER = 2\n')
    effects = diff(before, scan(work.copy, exclude=lambda _: False))
    assert {e.path for e in effects} == {'generated.py', 'other.py', 'build.ok'}
    for effect in effects:
        assert policy.authorize(effect)[0] is (not scoped)
    assert ('generated.py' in _product_source_paths(work.copy, FileSet.from_row(projected), projected)) is (not scoped)
    assert ('build.ok' in _product_source_paths(work.copy, FileSet.from_row(projected), projected)) is (not scoped)
    if scoped:
        with pytest.raises(FileOperationRefused, match='NEW_FILE_FORBIDDEN'):
            file_move(work.copy, 'app.py', 'renamed.py', protected=lambda _: False,
                      editable=lambda p: p in projected['files'],
                      new_file_allowed=lambda p: _is_product_path(p, projected))
    else:
        effect = file_move(work.copy, 'app.py', 'renamed.py', protected=lambda _: False,
                           editable=lambda p: p in projected['files'],
                           new_file_allowed=lambda p: _is_product_path(p, projected))
        assert effect.changed_paths == ('app.py', 'renamed.py')


def test_explicit_new_file_prohibition_survives_projection(monkeypatch):
    monkeypatch.setenv('INITIAL_ORACLE_SYMPTOM', 'OFF')
    row = {'allow_new_files': False, 'new_file_roots': ['src'], 'scoped_edits': True,
           'tier': 'private-dataset-label'}
    projected = InformationPolicy(row).worker_row(row)
    assert projected['allow_new_files'] is False
    assert projected['new_file_roots'] == ['src']
    assert 'tier' not in projected

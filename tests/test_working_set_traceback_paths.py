"""S8: check output only protects safe, editable workspace source paths."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kernel_working_set_snapshot import freeze_preparation


def snapshot_for(output, root=None, editable=()):
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=root) if root is not None else None,
        edit_contract={"edit_authority": {"editable_paths": list(editable)}},
        _check_output_receipts={"failed": {"exit_code": 1, "stderr": output}})
    return freeze_preparation({"messages": [{"role": "user", "content": "Task: fix"}]},
                              runtime, request_id="s8", byte_cap=100_000)


def test_r5_scout_traceback_paths_are_workspace_relative_and_editable(tmp_path):
    # Adapted from scout_repros/r5_regex.py: an external pytest frame,
    # absolute workspace frames, and relative path:line diagnostics.
    root = tmp_path / "repo"
    root.mkdir()
    output = f'''FAILED tests/test_mod.py::test_x
Traceback (most recent call last):
  File "{tmp_path}/.venv/lib/python3.12/site-packages/_pytest/runner.py", line 341, in from_call
  File "{root}/tests/test_mod.py", line 12, in test_x
  File "{root}/src/pkg/mod.py", line 88, in f
src/pkg/other.py:17: DeprecationWarning: x
tests/test_mod.py:12: AssertionError
'''
    snapshot = snapshot_for(output, root, ("src/pkg/mod.py", "src/pkg/other.py"))
    assert snapshot.protected_paths == ("src/pkg/mod.py", "src/pkg/other.py")


@pytest.mark.parametrize("diagnostic", [
    './src/mod.py:17: failure',
    'src\\mod.py:17: failure',
    '{root}/src/mod.py:17: failure',
    '  File "{root}/src/mod.py", line 17, in f',
    '  File "./src/mod.py", line 17, in f',
])
def test_diagnostic_aliases_share_canonical_protection(tmp_path, diagnostic):
    assert snapshot_for(diagnostic.format(root=tmp_path), tmp_path,
                        ("src/mod.py",)).protected_paths == ("src/mod.py",)


@pytest.mark.parametrize("diagnostic", [
    '../src/mod.py:17: failure',
    'src/../mod.py:17: failure',
    'backups/src/mod.py:17: failure',
    '/external/src/mod.py:17: failure',
    '{root}-other/src/mod.py:17: failure',
    'C:\\external\\src\\mod.py:17: failure',
    '  File "{root}/../src/mod.py", line 17, in f',
    '  File "{root}/backups/src/mod.py", line 17, in f',
    'readonly.py:17: failure',
])
def test_unsafe_external_and_noneditable_paths_do_not_protect(tmp_path, diagnostic):
    assert snapshot_for(diagnostic.format(root=tmp_path), tmp_path,
                        ("src/mod.py",)).protected_paths == ()


def test_relative_symlink_cannot_protect_external_source(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").symlink_to(tmp_path, target_is_directory=True)
    assert snapshot_for('src/mod.py:17: failure', root,
                        ("src/mod.py",)).protected_paths == ()


def test_quoted_traceback_preserves_spaces_in_workspace_path(tmp_path):
    path = 'src/my module.py'
    assert snapshot_for(f'  File "{tmp_path}/{path}", line 17', tmp_path,
                        (path,)).protected_paths == (path,)


def test_without_workspace_only_safe_relative_diagnostics_are_protected():
    output = './src/mod.py:1\n/external/src/other.py:2\n../src/bad.py:3'
    assert snapshot_for(output).protected_paths == ("src/mod.py",)


def test_noneditable_traceback_resident_can_defer_under_pressure(tmp_path):
    from kernel_working_set_allocator import FULL, DEFERRED
    from kernel_working_set_prepare import PreparedCandidate, prepare_candidate
    from kernel_working_set_render import domain, render_assignment
    from test_working_set_prepare import source

    target = source("target", "target content\n")
    dependency = source("readonly", "large dependency " * 1_000 + "\n", ordinal=2)
    snapshot = replace(snapshot_for(
        f'readonly.py:10: failure\n  File "{tmp_path}/target.py", line 12',
        tmp_path, ("target.py",)), sources=(target, dependency),
        recovery_allowed=True, recovery_owner_allowed=True)
    assert domain(snapshot, target) == (FULL,)
    assert domain(snapshot, dependency) == (FULL, DEFERRED)
    minimum = render_assignment(snapshot, {"target": FULL, "readonly": DEFERRED})
    snapshot = replace(snapshot, byte_cap=len(minimum.wire))
    prepared = prepare_candidate(snapshot, measure=lambda wire: (1, 0, "test"))
    assert isinstance(prepared, PreparedCandidate)
    assert dict(prepared.rendered.certificate.assignment) == {
        "target": FULL.name, "readonly": DEFERRED.name}
    assert prepared.total_bytes <= snapshot.byte_cap

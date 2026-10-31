"""Regressions for shell source anchors and replay's physical execution lifetime."""
import hashlib
import json
from types import SimpleNamespace
import uuid

import pytest

from kernel_exec_session import ExecSession
from kernel_stateframe_action_ledger import ActionLedgerError, build_action_context


def exchange(name, arguments, output, call_id):
    return [
        {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}]},
        {"role": "tool", "tool_call_id": call_id, "content": output},
    ]


def read_exchange(content, call_id):
    revision = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
    return exchange("read", {"path": "app.py"},
                    f"[read page] path=app.py revision={revision} lines=1-1/1 "
                    f"chars={len(content)} clip=none\ncurrent source\n{content}", call_id)


@pytest.mark.parametrize("content", ["VALUE = 2\n", "", None])
def test_shell_retires_unanchorable_read_site(tmp_path, content):
    path = tmp_path / "app.py"
    if content is not None:
        path.write_text(content)
    messages = [{"role": "user", "content": "Task: change the value"}]
    messages += read_exchange("VALUE = 1\n", "read-1")
    with pytest.raises(ActionLedgerError):
        build_action_context(messages, workspace_root=tmp_path)
    messages += exchange("shell", {"command": "python change.py"}, "exit code: 0", "shell-1")
    context = build_action_context(messages, workspace_root=tmp_path)
    assert context["open_edit_site"] is None
    assert context["site_context"]["reason"] == "shell_execution_requires_current_source_read"
    from kernel_stateframe_framefit5 import _next_sites
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    assert _next_sites(runtime, context) == []
    if content:
        from kernel_stateframe_framefit5 import build_typed_state
        context["journal_receipt_records"] = []
        state = build_typed_state(messages, context, runtime)
        assert state["next_site"] is None
        assert state["open_work"]["site_context_status"] == "UNVERIFIABLE"


def test_native_read_after_shell_establishes_current_site(tmp_path):
    (tmp_path / "app.py").write_text("VALUE = 2\n")
    messages = read_exchange("VALUE = 1\n", "read-1")
    messages += exchange("shell", {"command": "python change.py"}, "exit code: 0", "shell-1")
    messages += read_exchange("VALUE = 2\n", "read-2")
    context = build_action_context(messages, workspace_root=tmp_path)
    assert "site_context" not in context
    assert "VALUE = 2" in context["open_edit_site"]["current_bytes"]


def test_replay_uuid_does_not_reuse_execution_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("kernel_exec_session.uuid.uuid4", lambda: uuid.UUID(int=1))
    with ExecSession(tmp_path) as first:
        marker = first._directory / "supervisor.log"
        marker.write_text("previous attempt evidence")
        with ExecSession(tmp_path) as second:
            assert first._directory != second._directory
            assert second._directory.stat().st_mode & 0o777 == 0o700
    assert marker.read_text() == "previous attempt evidence"


def test_live_and_resume_workspace_identity_ignores_generated_caches(tmp_path):
    import py_compile
    import shutil
    from stateframe import workcopy
    from stateframe.resume import fresh_workcopy
    from stateframe._child import install_repo_context
    from kernel_effects import scan, diff
    from kernel_shell_tool import workspace_identity

    source = tmp_path / 'source'
    source.mkdir()
    (source / 'app.py').write_text('VALUE = 1\n')
    live = tmp_path / 'live'
    work = workcopy.create(source, live / 'workspace')
    shutil.copytree(work.copy, live / 'parent')
    resumed = tmp_path / 'resumed'
    shutil.copytree(live / 'parent', resumed / 'parent')
    fresh_workcopy(resumed, source)
    manifests = []
    for run in (live, resumed):
        # The child materializes the immutable parent, then checks/imports may
        # create bytecode whose embedded absolute source path varies on resume.
        child = SimpleNamespace(_materialize_public_tests=lambda *args: None)
        install_repo_context(child, run / 'parent')
        root = run / 'worker/repo'
        root.mkdir(parents=True)
        child._materialize_public_tests(root, None, {})
        py_compile.compile(str(root / 'app.py'), doraise=True)
        (root / '.pytest_cache').mkdir()
        (root / '.pytest_cache/state').write_text(str(run))
        manifests.append(scan(root, exclude=lambda _: False))
    changes = diff(*manifests)
    assert changes and all('__pycache__' in e.path or '.pytest_cache' in e.path for e in changes)
    assert workspace_identity(manifests[0]) == workspace_identity(manifests[1])


@pytest.mark.parametrize('change', ['bytes', 'mode', 'delete', 'type', 'create'])
def test_workspace_identity_keeps_real_product_changes(tmp_path, change):
    from kernel_effects import scan
    from kernel_shell_tool import workspace_identity
    path = tmp_path / 'app.py'
    path.write_text('VALUE = 1\n')
    before = workspace_identity(scan(tmp_path, exclude=lambda _: False))
    if change == 'bytes':
        path.write_text('VALUE = 2\n')
    elif change == 'mode':
        path.chmod(0o755)
    elif change == 'delete':
        path.unlink()
    elif change == 'type':
        path.unlink()
        path.mkdir()
    else:
        (tmp_path / 'new.py').write_text('VALUE = 3\n')
    assert workspace_identity(scan(tmp_path, exclude=lambda _: False)) != before


@pytest.mark.parametrize('name', ['__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache',
                                  '.venv', 'env', 'venv', 'build', 'node_modules'])
def test_workspace_identity_matches_workcopy_directory_exclusions(tmp_path, name):
    from kernel_effects import scan
    from kernel_shell_tool import workspace_identity
    (tmp_path / 'app.py').write_text('VALUE = 1\n')
    before = workspace_identity(scan(tmp_path, exclude=lambda _: False))
    (tmp_path / name).mkdir()
    (tmp_path / name / 'generated').write_text('run-specific')
    assert workspace_identity(scan(tmp_path, exclude=lambda _: False)) == before


def test_auto_verification_preserves_shell_exchange_in_both_frame_inputs():
    from kernel_stateframe_runtime import StateFrameRuntime
    from kernel_stateframe_f4 import latest_complete_tool_exchange
    observation = 'files changed by this command: modified: app.py'
    messages = exchange('shell', {'command': 'generate'}, observation, 'shell-1')
    uncapped = list(messages)  # Preparation can share message objects.
    events = []
    runtime = SimpleNamespace(
        _auto_verify_delivery={'content': 'pytest exit 0', 'tool_call_id': 'shell-1',
                               'target': 'tests/test_app.py'},
        journal=SimpleNamespace(emit=lambda *args, **kwargs: events.append((args, kwargs))))
    StateFrameRuntime._deliver_auto_verify(runtime, messages, uncapped, messages)
    for sequence in (messages, uncapped):
        latest = latest_complete_tool_exchange(sequence)
        assert latest is not None and len(latest) == 3
        assert [call['function']['name'] for call in latest[0]['tool_calls']] == ['shell', 'run_tests']
        assert latest[1]['content'] == observation
        assert latest[2]['content'] == 'pytest exit 0'
    assert runtime._auto_verify_delivery is None and len(events) == 1


def test_engine_public_checks_use_host_selected_run_python(tmp_path, monkeypatch):
    import subprocess
    import sys
    from probes import react_repair
    from stateframe.runner import RunOptions, _tests_python, use_run_env_python
    interpreter = tmp_path / 'env/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    options = RunOptions(repo=tmp_path, task='check', endpoint=None)
    options._capabilities = {}
    use_run_env_python(options, SimpleNamespace(python=interpreter))
    monkeypatch.setenv('STATEFRAME_TEST_PYTHON', _tests_python(options))
    (tmp_path / 'test_value.py').write_text('def test_value():\n    assert True\n')
    commands = []

    def sandbox_boundary(command, root, **kwargs):
        commands.append(command)
        return SimpleNamespace(completed=subprocess.CompletedProcess(command, 0, '1 passed\n', ''),
                               output_truncated=False, timed_out=False)

    monkeypatch.setattr('kernel_execution_policy.run_sandboxed', sandbox_boundary)
    result = react_repair.run_tests(tmp_path, 'value', 'test_value.py', row={'_skip_db': True})
    assert 'pytest exit 0' in result
    assert commands and all(command[0] == _tests_python(options) for command in commands)

"""Foreground orchestration with only the process boundary substituted."""

import pytest

from kernel_exec_session import CleanupReport, ExecResult
from kernel_shell_tool import ShellTool, request_confirmation
from kernel_tool_settings import ToolSettings
from kernel_tool_tape import ToolReplayRefused, configure_replay, tape_for_run


class ProcessBoundary:
    def __init__(self, action=None, *, cleanup=True):
        self.action, self.cleanup = action, cleanup
        self.calls = []

    def configure(self, cap):
        self.initial = cap

    def run(self, argv, cap, *, cwd):
        self.calls.append((argv, cap, cwd))
        if self.action:
            self.action(cwd)
        return ExecResult(0, False, b"hi\n", False, 3, 0.01,
                          CleanupReport(self.cleanup, 0, (), "test cleanup"))


@pytest.fixture
def setup_tool(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "app.py").write_text("VALUE = 1\n")
    (root / "test_app.py").write_text("assert True\n")
    run = tmp_path / "run"
    run.mkdir()
    row = {"files": {"app.py": "VALUE = 1\n"}, "scoped_edits": False,
           "allowed_product_paths": ["app.py"], "new_file_roots": []}

    def make(action=None, settings=None, cleanup=True):
        process = ProcessBoundary(action, cleanup=cleanup)
        tool = ShellTool(run, settings or ToolSettings(run_env=False), {}, process)
        tool.bind(root, row, {str(root / "test_app.py")})
        configure_replay(run, active=False)
        return tool, process
    return make, root, run


def test_roundtrip_and_durable_tape(setup_tool):
    make, root, _ = setup_tool
    tool, process = make()
    reply = tool.execute({"command": "echo hi"}, "call-1")
    assert "$ echo hi\nexit 0 | 0.01s | standard | network loopback\nhi\n" == reply.observation
    assert not reply.applied
    assert process.calls[0][0] == ["/bin/sh", "-c", "echo hi"]
    assert process.calls[0][2] == root
    assert tape_for_run().lookup("call-1").completed


def test_readonly_refuses_write_and_grants_only_scratch(setup_tool):
    make, root, run = setup_tool
    tool, process = make(settings=ToolSettings(shell="read-only", run_env=False))
    reply = tool.execute({"command": "touch x"}, "call-1")
    assert reply.observation.startswith("REFUSED READ_ONLY_POLICY:")
    assert not process.calls
    assert not tool.capability.workspace_writable
    assert tool.capability.write_roots == ((run / "shell-home").resolve(),)
    assert tool.capability.network == "none"
    assert not (root / "x").exists()


def test_shell_restores_protected_test_and_keeps_product_edit(setup_tool):
    make, root, _ = setup_tool

    def write(cwd):
        (cwd / "test_app.py").write_text("assert False\n")
        (cwd / "app.py").write_text("VALUE = 2\n")

    tool, _ = make(write)
    reply = tool.execute({"command": "python change.py"}, "call-1")
    assert (root / "test_app.py").read_text() == "assert True\n"
    assert (root / "app.py").read_text() == "VALUE = 2\n"
    assert [(e.path, e.change) for e in reply.applied] == [("app.py", "modified")]
    assert "restored (not allowed): test_app.py" in reply.observation


def test_replay_reexecutes_local_and_serves_taped_bytes(setup_tool):
    make, _, run = setup_tool
    tool, process = make()
    first = tool.execute({"command": "echo hi"}, "call-1")
    configure_replay(run, active=True)
    second = tool.execute({"command": "echo hi"}, "call-1")
    assert first.observation == second.observation
    assert len(process.calls) == 2


def test_replay_external_refuses_before_execution(setup_tool):
    make, _, run = setup_tool
    tool, process = make(settings=ToolSettings(shell_network="full", run_env=False))
    tool.execute({"command": "echo hi"}, "call-1")
    configure_replay(run, active=True)
    with pytest.raises(ToolReplayRefused, match="requires --replay-external"):
        tool.execute({"command": "echo hi"}, "call-1")
    assert len(process.calls) == 1


def test_replay_prestate_mismatch_refuses(setup_tool):
    make, root, run = setup_tool
    tool, process = make()
    tool.execute({"command": "echo hi"}, "call-1")
    (root / "app.py").write_text("changed\n")
    configure_replay(run, active=True)
    with pytest.raises(ToolReplayRefused, match="pre-state"):
        tool.execute({"command": "echo hi"}, "call-1")
    assert len(process.calls) == 1


def test_cleanup_failure_leaves_intent_incomplete(setup_tool):
    make, _, run = setup_tool
    tool, process = make(cleanup=False)
    reply = tool.execute({"command": "echo hi"}, "call-1")
    assert reply.fatal and reply.observation.startswith("REFUSED CLEANUP_FAILED:")
    assert not tape_for_run().lookup("call-1").completed
    configure_replay(run, active=True)
    with pytest.raises(ToolReplayRefused, match="may or may not have happened"):
        tool.execute({"command": "echo hi"}, "call-1")
    assert len(process.calls) == 1


def test_ask_decline_is_taped_and_not_asked_again(setup_tool, monkeypatch):
    make, _, run = setup_tool
    tool, process = make(settings=ToolSettings(shell_ask=True, run_env=False))
    asks = []

    def answer(*args):
        asks.append(args)
        return False

    monkeypatch.setattr("kernel_shell_tool.request_confirmation", answer)
    first = tool.execute({"command": "echo hi"}, "call-1")
    assert first.observation.startswith("REFUSED ASK_DECLINED:")
    configure_replay(run, active=True)
    assert tool.execute({"command": "echo hi"}, "call-1").observation == first.observation
    assert len(asks) == 1 and not process.calls


def test_confirmation_channel_noninteractive_declines(tmp_path):
    import threading
    from stateframe.shell_host import answer_pending

    result = []
    thread = threading.Thread(target=lambda: result.append(
        request_confirmation(tmp_path, "call-1", "echo hi", timeout=2)))
    thread.start()
    import time
    deadline = time.monotonic() + 1
    while not (tmp_path / "shell-ask.json").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    answer_pending(tmp_path)
    thread.join(3)
    assert not thread.is_alive() and result == [False]
    assert not (tmp_path / "shell-ask.json").exists()


def test_env_identity_never_runs_mutable_python(tmp_path):
    from kernel_run_env import RunEnv, env_identity
    env = tmp_path / "env"
    site = env / "lib/python3.11/site-packages"
    site.mkdir(parents=True)
    (env / "bin").mkdir()
    python = env / "bin/python"
    marker = tmp_path / "executed"
    python.write_text(f"#!/bin/sh\ntouch {marker}\nexit 99\n")
    python.chmod(0o755)
    info = site / "example-1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: example\nVersion: 1\n")
    run_env = RunEnv(env, python, python.parent,
                     {key: str(env / "cache" / key) for key in
                      ("npm_config_cache", "CARGO_HOME", "GOPATH", "GOMODCACHE")}, "clone")
    first = env_identity(run_env)
    (site / "bad.pth").write_text(f"import pathlib; pathlib.Path({str(marker)!r}).touch()")
    assert env_identity(run_env) == first
    (info / "METADATA").write_text("Name: example\nVersion: 2\n")
    assert env_identity(run_env) != first
    assert not marker.exists()


def test_shell_replay_detects_poststate_divergence(setup_tool):
    make, root, run = setup_tool
    tool, process = make()
    tool.execute({"command": "echo hi"}, "call-1")
    process.action = lambda cwd: (cwd / "app.py").write_text("VALUE = 9\n")
    configure_replay(run, active=True)
    with pytest.raises(ToolReplayRefused, match="workspace manifest"):
        tool.execute({"command": "echo hi"}, "call-1")
    assert (root / "app.py").read_text() == "VALUE = 9\n"


def test_shell_uses_immutable_baseline_for_protected_restoration(setup_tool):
    import shutil
    make, root, run = setup_tool
    parent = run / "parent"
    shutil.copytree(root, parent)
    (root / "test_app.py").write_text("already corrupted\n")
    process = ProcessBoundary(lambda cwd: (cwd / "test_app.py").write_text("corrupted again\n"))
    tool = ShellTool(run, ToolSettings(run_env=False), {"parent_snapshot": str(parent)}, process)
    tool.bind(root, {"files": {"app.py": ""}, "scoped_edits": False,
                     "allowed_product_paths": ["app.py"]}, {str(root / "test_app.py")})
    configure_replay(run, active=False)
    tool.execute({"command": "python change.py"}, "call-1")
    assert (root / "test_app.py").read_text() == "assert True\n"


def test_shell_off_never_launches(setup_tool):
    make, _, _ = setup_tool
    tool, process = make(settings=ToolSettings(shell="off", run_env=False))
    assert tool.execute({"command": "echo hi"}, "call-1").observation.startswith("REFUSED SHELL_OFF:")
    assert not process.calls


def test_shell_effect_receipt_retires_old_source_window(tmp_path):
    from types import SimpleNamespace
    from kernel_post_edit_refresh import refresh_after_applied_edit
    from kernel_request_source_window import source_revision

    path = tmp_path / "app.py"
    path.write_text("VALUE = 2\n")
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path),
                              _read_reuse_visibility_candidates=[{
                                  "path": "app.py", "revision": source_revision("VALUE = 1\n"),
                                  "start_line": 1, "end_line": 1, "content": "VALUE = 1\n"}])
    receipt = {"kind": "shell_effect", "result_class": "APPLIED",
               "changed_files": [{"path": "app.py"}], "turn": 2}
    current = source_revision(path.read_text())
    result = refresh_after_applied_edit(runtime, receipt,
                                       {"product_files": {"app.py": current}, "product_revision": "new"})
    assert result == {"app.py": current}
    assert runtime._read_reuse_visibility_candidates[0]["content"] == "VALUE = 2\n"
    assert runtime._post_edit_refresh["product_revision"] == "new"


def test_host_path_authority_resolves_executable_targets(tmp_path, monkeypatch):
    import os
    from stateframe.shell_host import path_authority
    bin_dir, real = tmp_path / "bin", tmp_path / "toolchain"
    bin_dir.mkdir()
    real.mkdir()
    binary = real / "compiler"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    (bin_dir / "cc").symlink_to(binary)
    monkeypatch.setenv("PATH", os.pathsep.join((str(bin_dir), str(tmp_path / "missing"), ".")))
    paths, roots = path_authority()
    assert paths == [str(bin_dir)]
    assert roots == sorted((str(bin_dir), str(real)))


def test_plain_shell_host_records_effect_without_check_credit(setup_tool, monkeypatch):
    from kernel_plain_contracts import EditAuthority, PlainJournal
    from kernel_plain_host import PlainHost
    import kernel_shell_tool

    make, root, run = setup_tool
    tool, _ = make(lambda cwd: (cwd / "app.py").write_text("VALUE = 2\n"))
    monkeypatch.setattr(kernel_shell_tool, "_active", tool)
    exported = []
    host = PlainHost(EditAuthority(root, ["app.py"]), check_runner=None,
                     journal=PlainJournal(run / "plain"), hooks=None,
                     on_product_edit=lambda: exported.append(True))
    host.bind_tool_call("call-1")
    output = host.execute("shell", {"command": "python change.py"})
    assert "files changed by this command: modified: app.py" in output
    assert exported == [True]
    assert host.book.product_edits == 1 and not host.book.checks and host.book.reads == 0
    assert host.book.completion(host.authority.product_revision())["status"] == "UNVERIFIED"


def test_ask_decline_external_is_safe_to_replay_without_optin(setup_tool, monkeypatch):
    make, _, run = setup_tool
    tool, process = make(settings=ToolSettings(shell_ask=True, shell_network="full", run_env=False))
    monkeypatch.setattr("kernel_shell_tool.request_confirmation", lambda *args: False)
    first = tool.execute({"command": "echo hi"}, "call-1")
    configure_replay(run, active=True)
    assert tool.execute({"command": "echo hi"}, "call-1").observation == first.observation
    assert not process.calls


def test_environment_only_change_invalidates_checks(setup_tool):
    from types import SimpleNamespace
    from kernel_run_env import RunEnv
    from kernel_shell_tool import publish_effects

    make, root, run = setup_tool
    tool, process = make()
    env = run / "env"
    site = env / "lib/python3.11/site-packages/example.dist-info"
    site.mkdir(parents=True)
    metadata = site / "METADATA"
    metadata.write_text("Name: example\nVersion: 1\n")
    tool.environment = RunEnv(env, env / "bin/python", env / "bin",
        {key: str(env / "cache" / key) for key in
         ("npm_config_cache", "CARGO_HOME", "GOPATH", "GOMODCACHE")}, "clone")
    process.action = lambda cwd: metadata.write_text("Name: example\nVersion: 2\n")
    reply = tool.execute({"command": "python install.py"}, "call-1")
    assert reply.environment_changed and not reply.applied
    assert "environment changed:" in reply.observation
    receipts = []
    runtime = SimpleNamespace(target=root / "app.py", _invalidate_binding_cache=lambda: None,
        journal=SimpleNamespace(manifest={}, emit=lambda *args, **kwargs: None))
    bolt = SimpleNamespace(root=root, product_edits=0, last_test_exit=0,
                           receipt=lambda *args, **kwargs: receipts.append((args, kwargs)))
    publish_effects(reply, runtime, bolt, set())
    assert bolt.last_test_exit is None and bolt.product_edits == 0
    assert runtime.journal.manifest["environment_revision"] == reply.environment_identity
    assert receipts[0][0][0] == "shell_effect"
    assert receipts[0][0][3] == "APPLIED"


def test_effect_summary_is_bounded_and_names_omission():
    from kernel_shell_tool import bounded_summary
    text = bounded_summary("a" * 10000)
    assert len(text.encode()) < 4200
    assert "5904 bytes omitted" in text


def test_failed_protected_restore_is_fatal_not_an_ordinary_tool_error(setup_tool):
    from kernel_shell_tool import ShellCleanupFailed, ShellExecutionFailed
    make, root, _ = setup_tool
    tool, _ = make(lambda cwd: (cwd / "test_app.py").write_text("changed\n"))
    digest = tool.baseline["test_app.py"].digest
    tool.baseline_store._path(digest).unlink()
    with pytest.raises(ShellExecutionFailed, match="EffectRestoreError"):
        tool.execute({"command": "python change.py"}, "call-1")
    assert not issubclass(ShellCleanupFailed, Exception)
    assert not tape_for_run().lookup("call-1").completed


def test_host_path_authority_preserves_search_precedence(tmp_path, monkeypatch):
    import os
    from stateframe.shell_host import path_authority
    first, second = tmp_path / "z-first", tmp_path / "a-second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("PATH", os.pathsep.join((str(first), str(second), str(first))))
    assert path_authority()[0] == [str(first), str(second)]

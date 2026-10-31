"""Host adapters exercise real tape, collectors, effects and schema contracts."""
import json
from types import SimpleNamespace

import pytest

from kernel_agent_tools import AgentTools
from kernel_custom_tools import CustomToolSet, load_tool_files
from kernel_exec_session import CleanupReport, ExecResult
from kernel_shell_tool import ShellTool
from kernel_tool_settings import ToolSettings
from kernel_tool_tape import configure_replay, tape_for_run


class Handle:
    """Named process boundary; collectors and orchestration remain real."""
    def __init__(self):
        self.data = b"ready"
        self.code = None
        self.lost_bytes = 0
        self.stops = 0

    def read_available(self):
        value, self.data = self.data, b""
        return value

    def poll(self):
        return self.code

    def stop(self):
        self.stops += 1
        self.code = 0
        return CleanupReport(True, 1, (), "stopped")


class ProcessBoundary:
    def __init__(self):
        self.calls = []
        self.handles = []
        self.action = None

    def configure(self, cap):
        self.initial = cap

    def spawn(self, argv, cap, *, cwd):
        self.calls.append((argv, cap))
        handle = Handle()
        self.handles.append(handle)
        return handle

    def run(self, argv, cap, *, cwd):
        self.calls.append((argv, cap))
        if self.action:
            self.action(cwd)
        return ExecResult(0, False, b"accepted\n", False, 9, .01, CleanupReport(True, 0, (), "ok"))

    def stop_all(self):
        for handle in self.handles:
            handle.stop()
        return CleanupReport(True, len(self.handles), (), "ok")


@pytest.fixture
def host(tmp_path, monkeypatch):
    import kernel_agent_tools
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "app.py").write_text("x=1\n")
    process = ProcessBoundary()
    shell = ShellTool(tmp_path, ToolSettings(run_env=False, web=True), {}, process)
    shell.bind(root, {"files": {"app.py": "x=1\n"}, "scoped_edits": False, "allowed_product_paths": ["app.py", "generated.py"]}, set())
    hub = AgentTools(shell, CustomToolSet((), "empty"))
    hub.bind()
    monkeypatch.setattr(kernel_agent_tools, "_active", hub)
    configure_replay(tmp_path, active=False)
    return hub, process, tmp_path


def test_jobs_readonly_own_scratch_limit_and_output(host):
    hub, process, run = host
    assert not hub.offered("job_wait")
    for index in range(4):
        reply = hub.shell.execute({"command": "echo ready", "background": True}, f"c{index}")
        assert f"started job j{index+1}:" in reply.observation
        cap = process.calls[-1][1]
        assert cap.purpose == "job" and not cap.workspace_writable
        assert cap.write_roots == (run / "jobs" / f"j{index+1}",)
    assert hub.offered("job_stop")
    assert hub.shell.execute({"command": "echo ready", "background": True}, "limit").observation.startswith("REFUSED JOB_LIMIT:")
    assert len(process.calls) == 4
    result = hub.execute("job_wait", {"job": "j1", "until": "output", "text": "ready"}, "wait")
    assert "met" in result.observation and "output matched" in result.observation
    assert "ready" in hub.execute("job_output", {"job": "j1"}, "out").observation
    assert "job j1 stopped" in hub.execute("job_stop", {"job": "j1"}, "stop").observation


def test_runtime_defers_owned_job_output_to_taped_adapter(host):
    from kernel_stateframe_runtime import StateFrameRuntime
    hub, _, _ = host
    hub.shell.execute({"command": "echo ready", "background": True}, "launch")

    class LegacyJobs:
        def execute(self, name, arguments):
            pytest.fail("legacy job organ intercepted a per-run tool")

    runtime = SimpleNamespace(job_watch=LegacyJobs())
    assert StateFrameRuntime.job_operation(runtime, "job_output", {"job": "j1"}) is None
    reply = hub.execute("job_output", {"job": "j1"}, "output")
    assert "ready" in reply.observation
    record = tape_for_run().lookup("output")
    assert record is not None and record.tool == "job_output"
    assert record.observation == reply.observation


def test_runtime_preserves_legacy_jobs_without_adapter(monkeypatch):
    from kernel_stateframe_runtime import StateFrameRuntime
    monkeypatch.setattr("kernel_agent_tools._active", None)
    calls = []
    runtime = SimpleNamespace(
        job_watch=SimpleNamespace(execute=lambda name, arguments: calls.append((name, arguments)) or "legacy"),
        journal=SimpleNamespace(emit=lambda *args, **kwargs: None),
        owner=SimpleNamespace(receipt=lambda *args: None))
    assert StateFrameRuntime.job_operation(runtime, "job_output", {"job": "old"}) == "legacy"
    assert calls == [("job_output", {"job": "old"})]


def test_snapshot_frozen_until_journal_and_next_request(host):
    hub, process, _ = host
    hub.shell.execute({"command": "echo ready", "background": True}, "launch")
    hub.jobs.next_request(1)
    first = hub.snapshot()
    process.handles[0].data = b"new bytes"
    assert hub.snapshot() == first
    assert hub.jobs.jobs["j1"].cursor == 0
    events = []
    journal = SimpleNamespace(emit=lambda *a, **k: events.append((a, k)))
    hub.jobs.commit(journal)
    assert events[0][0] == ("job_snapshot_committed",)
    hub.jobs.commit(journal)
    assert len(events) == 1
    hub.jobs.next_request(1)
    assert hub.snapshot() == first
    hub.jobs.next_request(2)
    second = hub.snapshot()
    assert second["sequence"] == 2 and second["jobs"][0]["tail"] == "new bytes"


def test_snapshot_caps_and_replay_never_spawn(host):
    hub, process, run = host
    args = {"command": "echo ready", "background": True}
    original = hub.shell.execute(args, "launch").observation
    process.handles[0].data = b"x" * 20000
    frame = hub.snapshot()
    assert len(frame["jobs"][0]["tail"].encode()) <= 1024
    configure_replay(run, active=True)
    hub.jobs.jobs.clear()
    hub.jobs.frozen = None
    hub.jobs.sequence = 0
    assert hub.shell.execute(args, "launch").observation == original
    assert len(process.calls) == 1
    assert hub.snapshot() == frame
    configure_replay(run, active=False)
    hub.jobs.frozen = None
    assert hub.snapshot()["jobs"][0]["state"] == "lost on restart"


def test_wait_matches_output_even_at_exit(host):
    hub, process, _ = host
    hub.shell.execute({"command": "echo ready", "background": True}, "launch")
    process.handles[0].code = 0
    result = hub.execute("job_wait", {"job": "j1", "until": "output", "text": "ready"}, "wait")
    assert "met" in result.observation and "output matched" in result.observation
    assert process.handles[0].stops == 1


def test_wait_deadline_and_bad_conditions(host):
    hub, process, _ = host
    hub.shell.execute({"command": "echo ready", "background": True}, "launch")
    hub.jobs.deadline = 0
    assert "timeout" in hub.execute("job_wait", {"job": "j1", "until": "exit"}, "wait").observation
    assert hub.execute("job_wait", {"job": "j1", "until": "endpoint", "port": True}, "bad").observation.startswith("REFUSED BAD_CONDITION:")
    assert hub.execute("job_stop", {"job": "missing"}, "missing").observation.startswith("REFUSED UNKNOWN_JOB:")


def test_endpoint_probe_uses_session_and_readonly_capability(host):
    hub, process, _ = host
    hub.shell.execute({"command": "echo ready", "background": True}, "launch")
    result = hub.execute("job_wait", {"job": "j1", "until": "endpoint", "port": 12345}, "wait")
    assert "endpoint reachable" in result.observation
    assert process.calls[-1][1].purpose == "probe"
    assert not process.calls[-1][1].workspace_writable
    assert "12345" == process.calls[-1][0][-1]


def test_web_private_refusal_taped_and_disabled_gating(host):
    hub, process, run = host
    args = {"url": "http://127.0.0.1/", "max_chars": 300}
    result = hub.execute("web_fetch", args, "web")
    assert result.observation.startswith("REFUSED WEB_REFUSED_ADDRESS:")
    assert tape_for_run().lookup("web").replay_class == "observation"
    configure_replay(run, active=True)
    assert hub.execute("web_fetch", args, "web").observation == result.observation
    assert not process.calls


@pytest.mark.parametrize("url,code", [("file:///etc/passwd", "WEB_REFUSED_SCHEME"),
                                      ("http://example.com:81", "WEB_REFUSED_PORT")])
def test_web_contract_codes(host, url, code):
    hub, _, _ = host
    assert hub.execute("web_fetch", {"url": url}, "web").observation.startswith(f"REFUSED {code}:")


def test_custom_real_schema_argv_effects_and_digest(host):
    hub, process, run = host
    executable = run / "custom"
    executable.write_text("#!/bin/sh\nprintf accepted\\n\n")
    executable.chmod(0o755)
    document = {"schema": "stateframe-tools/1", "tools": [{"name": "sample_tool", "description": "Sample",
        "parameters": {"type": "object", "properties": {"value": {"type": "string", "maxLength": 100}}, "required": ["value"]},
        "argv": [str(executable), "{value}"], "network": "none", "writes": "workspace", "replay": "local", "timeout_seconds": 10}]}
    file = run / "tools.json"
    file.write_text(json.dumps(document))
    custom = load_tool_files([file], builtin_names=frozenset(), shell_network="loopback")
    hub.custom = {tool.name: tool for tool in custom.tools}
    process.action = lambda root: (root / "generated.py").write_text("VALUE=1\n")
    result = hub.execute("sample_tool", {"value": "; touch forbidden", "why": "test"}, "custom")
    assert process.calls[-1][0] == [str(executable), "; touch forbidden"]
    assert process.calls[-1][1].purpose == "custom_tool"
    assert "created: generated.py" in result.observation
    assert result.applied[0].path == "generated.py"
    assert hub.execute("sample_tool", {"value": 3}, "invalid").observation.startswith("REFUSED BAD_ARGUMENTS:")
    executable.write_text("#!/bin/sh\nexit 2\n")
    assert hub.execute("sample_tool", {"value": "hi"}, "changed").observation.startswith("REFUSED TOOL_CHANGED:")
    assert len(process.calls) == 1


def test_git_disabled_refusal(host):
    hub, process, _ = host
    assert not hub.offered("git")
    assert hub.execute("git", {"action": "log"}, "git").observation.startswith("REFUSED NOT_A_GIT_REPO:")
    assert not process.calls


def test_git_contract_adapter_runs_sanitized_history(host):
    import subprocess
    from kernel_git_view import GitView, excluded_path
    hub, process, run = host
    source = run / "source"
    source.mkdir()
    def git(*args):
        return subprocess.run(["git", *args], cwd=source, check=True, capture_output=True).stdout
    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    for value in (1, 2):
        (source / "app.py").write_text(f"x={value}\n")
        (source / ".env").write_text("SECRET=must-not-leak\n")
        git("add", "app.py", ".env")
        git("commit", "-qm", f"revision {value}")
    view_root = run / "view"
    view_root.mkdir()
    hub.git = GitView.build(source, view_root, secret_filter=excluded_path)
    hub.bind()
    def run_git(argv, cap, *, cwd):
        process.calls.append((argv, cap))
        result = subprocess.run(argv, env=dict(cap.env), cwd=cwd, capture_output=True, timeout=5)
        return ExecResult(result.returncode, False, result.stdout + result.stderr, False,
                          len(result.stdout + result.stderr), .01, CleanupReport(True, 0, (), "ok"))
    process.run = run_git
    assert "revision 2" in hub.execute("git", {"action": "log", "max_count": 1}, "log").observation
    assert "x=1" in hub.execute("git", {"action": "show", "rev": "HEAD~1", "path": "app.py"}, "show").observation
    assert hub.execute("git", {"action": "show", "rev": "HEAD:.env"}, "secret").observation.startswith("REFUSED EXCLUDED_PATH:")
    assert hub.execute("git", {"action": "show", "rev": "missing"}, "missing").observation.startswith("REFUSED BAD_REVISION:")
    assert hub.execute("git", {"action": "diff", "path": ".env"}, "diff-secret").observation.startswith("REFUSED EXCLUDED_PATH:")
    for _, cap in process.calls:
        assert cap.purpose == "git" and cap.network == "none" and not cap.workspace_writable
        assert hub.git.git_dir in cap.read_roots
        assert set(hub.git.object_dirs) <= set(cap.read_roots)
    before = len(process.calls)
    configure_replay(run, active=True)
    assert "revision 2" in hub.execute("git", {"action": "log", "max_count": 1}, "log").observation
    assert len(process.calls) == before
    hub.shell.spec["repo"] = str(source)
    pinned = AgentTools(hub.shell, CustomToolSet((), "empty"))
    original_head = pinned.git.head_oid
    (source / "app.py").write_text("x=3\n")
    git("add", "app.py")
    git("commit", "-qm", "moved source head")
    resumed = AgentTools(hub.shell, CustomToolSet((), "empty"))
    assert resumed.git.head_oid == original_head
    assert resumed.git.git_dir == pinned.git.git_dir


def test_custom_readonly_and_external_replay(host):
    from dataclasses import replace
    from kernel_tool_tape import ToolReplayRefused
    hub, process, run = host
    file = __import__("pathlib").Path(__file__).parent / "acceptance/fixtures/tools.json"
    definitions = load_tool_files([file], builtin_names=frozenset(), shell_network="loopback")
    tool = replace(definitions.tools[0], replay="external")
    hub.custom[tool.name] = tool
    hub.execute(tool.name, {"value": "ok"}, "external")
    assert not process.calls[-1][1].workspace_writable
    assert process.calls[-1][1].network == "none"
    configure_replay(run, active=True)
    with pytest.raises(ToolReplayRefused, match="replay-external"):
        hub.execute(tool.name, {"value": "ok"}, "external")
    assert len(process.calls) == 1


def test_job_cleanup_failure_is_fatal_and_intent_stays_incomplete(host):
    hub, process, _ = host
    hub.shell.execute({"command": "echo ready", "background": True}, "launch")
    process.handles[0].stop = lambda: CleanupReport(False, 0, (1,), "survivor")
    reply = hub.execute("job_stop", {"job": "j1"}, "stop")
    assert reply.fatal and reply.observation.startswith("REFUSED CLEANUP_FAILED:")
    assert not tape_for_run().lookup("stop").completed


def test_wait_keyboard_interrupt_is_not_swallowed(host):
    hub, process, _ = host
    hub.shell.execute({"command": "echo ready", "background": True}, "launch")
    def interrupted():
        raise KeyboardInterrupt()
    process.handles[0].read_available = interrupted
    with pytest.raises(KeyboardInterrupt):
        hub.execute("job_wait", {"job": "j1", "until": "exit"}, "wait")
    assert not tape_for_run().lookup("wait").completed


def test_done_cleanup_is_taped_and_byte_stable(host):
    hub, process, run = host
    args = {"command": "echo ready", "background": True}
    hub.shell.execute(args, "launch")
    result = hub.stop_jobs("done")
    assert "stopped all jobs" in result
    configure_replay(run, active=True)
    hub.jobs.jobs.clear()
    hub.shell.execute(args, "launch")
    assert hub.stop_jobs("done") == result
    assert hub.jobs.jobs["j1"].state == "stopped"


def test_green_terminal_jobs_and_observations_are_dispatchable(host):
    from kernel_greenlock_menu_authority import decide
    hub, _, _ = host
    hub.shell.execute({"command": "echo ready", "background": True}, "launch")
    lock = SimpleNamespace(runtime=SimpleNamespace(), terminal_engaged=True)
    for name in ("job_wait", "job_output", "job_stop", "web_fetch"):
        assert decide(lock, name, {}, for_menu=True).allowed
        assert decide(lock, name, {}).allowed


def test_readonly_shell_is_observation_replay(host):
    from dataclasses import replace
    hub, process, run = host
    hub.shell.settings = replace(hub.shell.settings, shell="read-only")
    hub.shell.capability = replace(hub.shell.capability, workspace_writable=False, network="none")
    reply = hub.shell.execute({"command": "echo ready"}, "readonly")
    # echo is outside the restricted inspection grammar, use an inspection command.
    assert reply.observation.startswith("REFUSED READ_ONLY_POLICY:")
    first = hub.shell.execute({"command": "pwd"}, "pwd")
    configure_replay(run, active=True)
    assert hub.shell.execute({"command": "pwd"}, "pwd").observation == first.observation
    assert len(process.calls) == 1


def test_unscoped_user_codegen_admits_new_roots_but_not_tests_or_secrets(host):
    from kernel_effects import effect_policy_for_run
    from probes.react_repair import _is_product_path
    hub, process, _ = host
    row = {"tier": "USER-REPO", "scoped_edits": False,
           "files": {"app.py": "x=1\n"}, "allowed_product_paths": ["app.py"]}
    hub.shell.policy = effect_policy_for_run(hub.shell.root, row, set(), baseline_manifest=hub.shell.baseline)
    def generate(root):
        (root / "generated.py").write_text("GENERATED=1\n")
        (root / "newpkg").mkdir()
        (root / "newpkg/helper.py").write_text("VALUE=1\n")
        (root / "test_generated.py").write_text("assert False\n")
        (root / ".env").write_text("SECRET=value\n")
    process.action = generate
    reply = hub.shell.execute({"command": "python scripts/generate.py"}, "generate")
    assert {e.path for e in reply.applied} == {"generated.py", "newpkg", "newpkg/helper.py"}
    assert not (hub.shell.root / "test_generated.py").exists()
    assert not (hub.shell.root / ".env").exists()
    assert _is_product_path("generated.py", row)
    assert _is_product_path("newpkg/helper.py", row)
    assert not _is_product_path("generated.py", {**row, "scoped_edits": True})


def test_custom_registration_rejects_optional_builtin_collision(host):
    from dataclasses import replace
    from pathlib import Path
    from kernel_custom_tools import CustomToolError
    hub, _, _ = host
    source = Path(__file__).parent / "acceptance/fixtures/tools.json"
    custom = load_tool_files([source], builtin_names=frozenset(), shell_network="loopback")
    collision = replace(custom.tools[0], name="probe")
    with pytest.raises(CustomToolError, match="duplicate built-in"):
        AgentTools(hub.shell, CustomToolSet((collision,), custom.digest))

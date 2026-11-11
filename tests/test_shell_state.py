"""STATEFRAME_SHELL_STATE: cwd and exported variables persist between foreground shell calls.

The process boundary here really runs the argv (without a sandbox), so the wrapper script, its exit trap
and the state capture are exercised end to end; only sandboxing is substituted.
"""
import subprocess

import pytest

from _timeouts import scaled
import kernel_shell_state
from kernel_exec_session import CleanupReport, ExecResult
from kernel_shell_tool import ShellTool
from kernel_tool_settings import ToolSettings
from kernel_tool_tape import ToolReplayRefused, configure_replay, tape_for_run


class RealProcess:
    def __init__(self):
        self.calls = []

    def configure(self, cap):
        self.initial = cap

    def run(self, argv, cap, *, cwd):
        self.calls.append((argv, cap, cwd))
        done = subprocess.run(argv, cwd=cwd, env=dict(cap.env), capture_output=True, timeout=scaled(30))
        output = done.stdout + done.stderr
        return ExecResult(done.returncode, False, output, False, len(output), 0.01,
                          CleanupReport(True, 0, (), "test cleanup"))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "app.py").write_text("VALUE = 1\n")
    run = tmp_path / "run"
    run.mkdir()
    row = {"files": {"app.py": "VALUE = 1\n"}, "scoped_edits": False,
           "allowed_product_paths": ["app.py", "sub"], "new_file_roots": ["sub"]}

    def make(mode="on", settings=None):
        monkeypatch.setenv(kernel_shell_state.MODE_ENV, mode)
        process = RealProcess()
        tool = ShellTool(run, settings or ToolSettings(run_env=False), {}, process)
        tool.bind(root, row, set())
        configure_replay(run, active=False)
        return tool, process
    return make, root.resolve(), run


def _output(reply):
    return reply.observation


def test_off_keeps_plain_argv_and_tape_args(setup):
    make, root, run = setup
    tool, process = make("off")
    tool.execute({"command": "cd /tmp && export X=1"}, "call-1")
    reply = tool.execute({"command": "pwd"}, "call-2")
    assert process.calls[1][0] == ["/bin/sh", "-c", "pwd"]
    assert str(root) in _output(reply)
    assert "shell cwd" not in _output(reply)
    assert "_shell_state" not in tape_for_run().lookup("call-2").args
    assert not (run / kernel_shell_state.STATE_DIR).exists()


def test_cd_and_export_persist_to_the_next_call(setup):
    make, root, _ = setup
    tool, _ = make()
    first = tool.execute({"command": "mkdir -p sub && cd sub && export X=1"}, "call-1")
    assert "shell cwd: sub" in _output(first)
    second = tool.execute({"command": "pwd; echo \"X=$X\""}, "call-2")
    assert f"{root}/sub\n" in _output(second)
    assert "X=1\n" in _output(second)
    assert tape_for_run().lookup("call-2").args["_shell_state"] == {
        "cwd": "sub", "set": {"X": "1"}, "unset": []}


def test_state_is_kept_when_the_command_fails(setup):
    make, root, _ = setup
    tool, _ = make()
    tool.execute({"command": "mkdir -p sub && cd sub && export Y=2 && false"}, "call-1")
    reply = tool.execute({"command": "pwd; echo \"Y=$Y\""}, "call-2")
    assert f"{root}/sub\n" in _output(reply) and "Y=2\n" in _output(reply)


def test_unset_persists(setup):
    make, _, _ = setup
    tool, _ = make()
    tool.execute({"command": "export Z=3"}, "call-1")
    tool.execute({"command": "unset Z"}, "call-2")
    reply = tool.execute({"command": "echo \"Z=${Z-gone}\""}, "call-3")
    assert "Z=gone\n" in _output(reply)


def test_directory_outside_the_workspace_is_not_kept(setup):
    make, root, _ = setup
    tool, _ = make()
    first = tool.execute({"command": "cd /"}, "call-1")
    assert "directory outside the workspace was not kept" in _output(first)
    reply = tool.execute({"command": "pwd"}, "call-2")
    assert f"{root}\n" in _output(reply)


def test_engine_switches_and_wrapper_names_are_never_kept(setup):
    make, _, _ = setup
    tool, _ = make()
    tool.execute({"command": "export STATEFRAME_SHELL_STATE=off __sf_x=1 KEEP=1"}, "call-1")
    assert tool.shell_state.set_vars == {"KEEP": "1"}


def test_removed_directory_restarts_at_the_root(setup):
    make, root, _ = setup
    tool, _ = make()
    tool.execute({"command": "mkdir -p sub && cd sub"}, "call-1")
    tool.execute({"command": "cd .. && rm -rf sub && cd /tmp; cd \"$OLDPWD\" 2>/dev/null; true"}, "call-2")
    (root / "sub").mkdir(exist_ok=True)
    (root / "sub").rmdir()
    reply = tool.execute({"command": "pwd"}, "call-3")
    assert f"{root}\n" in _output(reply)


def test_replay_reexecution_restores_the_same_state(setup):
    make, root, run = setup
    (root / "sub").mkdir()
    tool, _ = make()
    tool.execute({"command": "cd sub && export X=1"}, "call-1")
    first = tool.execute({"command": "pwd; echo \"X=$X\""}, "call-2")
    replayed, process = make()
    configure_replay(run, active=True)
    replayed.execute({"command": "cd sub && export X=1"}, "call-1")
    second = replayed.execute({"command": "pwd; echo \"X=$X\""}, "call-2")
    assert second.observation == first.observation
    assert len(process.calls) == 2
    assert replayed.shell_state == tool.shell_state


def test_replay_refuses_a_call_started_from_a_different_state(setup):
    make, root, run = setup
    (root / "sub").mkdir()
    tool, _ = make()
    tool.execute({"command": "cd sub"}, "call-1")
    tool.execute({"command": "pwd"}, "call-2")
    replayed, _ = make()
    configure_replay(run, active=True)
    with pytest.raises(ToolReplayRefused):
        replayed.execute({"command": "pwd"}, "call-2")


def test_taped_replay_restores_state_without_running(setup):
    make, _, run = setup
    settings = ToolSettings(shell="read-only", run_env=False)
    tool, _ = make(settings=settings)
    tool.execute({"command": "pwd"}, "call-1")
    stored = kernel_shell_state.load(run, "call-1")
    kernel_shell_state.store(run, "call-1", kernel_shell_state.ShellState("sub", {"R": "1"}, ()))
    replayed, process = make(settings=settings)
    configure_replay(run, active=True)
    replayed.execute({"command": "pwd"}, "call-1")
    assert stored == kernel_shell_state.ShellState()
    assert process.calls == []
    assert replayed.shell_state == kernel_shell_state.ShellState("sub", {"R": "1"}, ())


def test_taped_replay_refuses_a_missing_state_record(setup):
    make, _, run = setup
    settings = ToolSettings(shell="read-only", run_env=False)
    tool, _ = make(settings=settings)
    tool.execute({"command": "pwd"}, "call-1")
    for path in (run / kernel_shell_state.STATE_DIR).iterdir():
        path.unlink()
    replayed, _ = make(settings=settings)
    configure_replay(run, active=True)
    with pytest.raises(ToolReplayRefused, match="shell state"):
        replayed.execute({"command": "pwd"}, "call-1")


@pytest.mark.host_sandbox
def test_state_persists_through_the_real_sandbox(tmp_path, monkeypatch):
    from kernel_exec_session import ExecSession
    monkeypatch.setenv(kernel_shell_state.MODE_ENV, "on")
    root = tmp_path / "workspace"
    (root / "sub").mkdir(parents=True)
    (root / "app.py").write_text("VALUE = 1\n")
    run = tmp_path / "run"
    run.mkdir()
    row = {"files": {"app.py": "VALUE = 1\n"}, "scoped_edits": False,
           "allowed_product_paths": ["app.py", "sub"], "new_file_roots": ["sub"]}
    with ExecSession(run) as session:
        tool = ShellTool(run, ToolSettings(run_env=False), {}, session)
        tool.bind(root, row, set())
        configure_replay(run, active=False)
        first = tool.execute({"command": "cd sub && export X=1"}, "call-1")
        second = tool.execute({"command": "pwd; echo \"X=$X\""}, "call-2")
    assert "shell cwd: sub" in first.observation, first.observation
    assert f"{root.resolve()}/sub\n" in second.observation, second.observation
    assert "X=1\n" in second.observation, second.observation


def test_invalid_mode_is_refused(monkeypatch):
    monkeypatch.setenv(kernel_shell_state.MODE_ENV, "maybe")
    with pytest.raises(ValueError, match="off or on"):
        kernel_shell_state.enabled()

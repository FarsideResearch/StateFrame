"""Checkpoint-save barriers for test-run interruptions."""

from __future__ import annotations

import ast
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fixtures.process_signals import signal_run_owned, wait_for_checkpoint_saved


TESTS = Path(__file__).resolve().parent

# These exercise process signalling, containment or a non-run helper, not a StateFrame run.
PRIMITIVE_TESTS = {
    "test_launch_unit_kill.py",
    "test_orphan_guard.py",  # plants and stops detached processes, never a StateFrame run
    "test_orphan_guard_portability.py",  # plants and stops a detached launcher, never a StateFrame run
    "orphan_guard_mutants.py",  # contains deliberately incorrect process-signal mutants, never a StateFrame run
    "test_orphan_guard_acceptance.py",  # stops its own test helpers, never a StateFrame run
    "test_process_signal_liveness.py",  # probes its process-signal fixture helpers, never a StateFrame run
    "test_exec_tracker.py",
    "test_launch_identity_signals.py",
    "test_cleanup_containment.py",
    "test_cleanup_escalation.py",
    "test_cleanup_verified_reuse.py",
    "test_descendant_signal_platforms.py",
    "test_outer_check_collect.py",
    "test_hook_driver.py",
    "test_process_signal_authority.py",
    "test_checkpoint_live_probe.py",
    "test_checkpoint_kill_barrier.py",
    "test_exec_session.py",
    "test_exec_supervisor_parent_death.py",  # launches the supervisor primitive directly, not a StateFrame run
    "test_frame_routes_engine.py",
    "test_powerups_python_in_sandbox.py",
    "test_check_sandbox_nproc.py",
}


def _raw_run_signals(source: str, relative: Path) -> list[tuple[int, str]]:
    """Find direct signals in run tests, including imported aliases."""
    tree = ast.parse(source)
    os_names = {"os"}
    signal_modules = set()
    raw_names = {"signal_verified", "signal_owned"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            os_names.update(alias.asname or alias.name for alias in node.names
                            if alias.name == "os")
            signal_modules.update(alias.asname or alias.name for alias in node.names
                                  if alias.name == "fixtures.process_signals")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "os":
                raw_names.update(alias.asname or alias.name for alias in node.names
                                 if alias.name in {"kill", "killpg"})
            elif node.module == "fixtures":
                signal_modules.update(alias.asname or alias.name for alias in node.names
                                      if alias.name == "process_signals")
            elif node.module and node.module.endswith("process_signals"):
                raw_names.update(alias.asname or alias.name for alias in node.names
                                 if alias.name in {"signal_verified", "signal_owned"})
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        raw = isinstance(func, ast.Name) and func.id in raw_names
        group_probe = (isinstance(func, ast.Attribute) and func.attr == "killpg"
                       and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
                       and node.args[1].value == 0)
        raw |= (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id in os_names and func.attr in {"kill", "killpg"}
                and not group_probe)
        raw |= (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id in signal_modules
                and func.attr in {"signal_verified", "signal_owned"})
        if isinstance(func, ast.Attribute) and func.attr in {
                "kill", "terminate", "send_signal"}:
            # Endpoint and server lifetimes are independent of the StateFrame run.
            endpoint = isinstance(func.value, ast.Name) and func.value.id in {
                "endpoint", "server"}
            endpoint |= (relative.as_posix() == "acceptance/_support.py"
                         and isinstance(func.value, ast.Attribute)
                         and isinstance(func.value.value, ast.Name)
                         and func.value.value.id == "self" and func.value.attr == "process")
            raw |= not endpoint
        if raw:
            violations.append((node.lineno, ast.unparse(node)))
    return violations


def _unbarriered_signals(root: Path) -> dict[str, list[tuple[int, str]]]:
    violations = {}
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        # The monitor owns and reaps its engine process group after the leader exits; run-kill decisions stay in tests.
        # The orphan guard only stops launchd-adopted leftovers, after their run has ended.
        if relative.as_posix() in {"fixtures/process_signals.py", "engine_run.py", "_orphan_guard.py"} or (
                relative.parts[0] != "acceptance" and path.name in PRIMITIVE_TESTS):
            continue
        found = _raw_run_signals(path.read_text(encoding="utf-8"), relative)
        if found:
            violations[relative.as_posix()] = found
    return violations


def test_run_kills_use_checkpoint_barrier() -> None:
    violations = _unbarriered_signals(TESTS)
    assert not violations, violations


def test_engine_run_primitive_exemption_is_path_exact(tmp_path: Path) -> None:
    source = "import os\nos.killpg(123, 9)\n"
    (tmp_path / "engine_run.py").write_text(source, encoding="utf-8")
    acceptance = tmp_path / "acceptance"
    acceptance.mkdir()
    (acceptance / "engine_run.py").write_text(source, encoding="utf-8")

    assert _unbarriered_signals(tmp_path) == {
        "acceptance/engine_run.py": [(2, "os.killpg(123, 9)")],
    }


def test_orphan_guard_primitive_exemptions_are_path_exact(tmp_path: Path) -> None:
    source = "import os\nos.killpg(123, 9)\n"
    acceptance = tmp_path / "acceptance"
    acceptance.mkdir()
    names = ("orphan_guard_mutants.py", "test_orphan_guard_acceptance.py")
    for name in names:
        (tmp_path / name).write_text(source, encoding="utf-8")
        (acceptance / name).write_text(source, encoding="utf-8")

    assert _unbarriered_signals(tmp_path) == {
        f"acceptance/{name}": [(2, "os.killpg(123, 9)")] for name in names
    }


def test_group_existence_probe_is_not_a_run_kill() -> None:
    assert not _raw_run_signals("import os\nos.killpg(pgid, 0)\n", Path("test_engine_run.py"))


@pytest.mark.parametrize("source", (
    "import os\nos.kill(123, 9)\n",
    "from os import killpg as stop\nstop(123, 9)\n",
    "from fixtures.process_signals import signal_verified as stop\nstop(123, ident, 9)\n",
    "from fixtures import process_signals as signals\nsignals.signal_owned(child, 9)\n",
    "process.kill()\n",
    "runner.send_signal(9)\n",
    "self.runner.kill()\n",
))
def test_guard_rejects_planted_raw_acceptance_kill(source: str) -> None:
    assert _raw_run_signals(source, Path("acceptance/test_new_run.py"))


def _child(tmp_path: Path, body: str) -> subprocess.Popen[bytes]:
    script = tmp_path / "checkpoint_control_writer.py"
    script.write_text(body)
    return subprocess.Popen([sys.executable, str(script)], start_new_session=True)


def test_delayed_complete_save_precedes_signal(tmp_path: Path) -> None:
    run = tmp_path / "run"
    events = run / "checkpoints" / "events.jsonl"
    events.parent.mkdir(parents=True)
    marker = tmp_path / "finished-writing"
    child = _child(tmp_path, (
        "import pathlib, time\n"
        f"events = pathlib.Path({str(events)!r})\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        "with events.open('ab', buffering=0) as stream:\n"
        "    stream.write(b'{\"event\": \"checkpoint_sa')\n"
        "    time.sleep(1.5)\n"
        "    marker.write_text('complete')\n"
        "    stream.write(b'ved\"}\\n')\n"
        "while True: time.sleep(1)\n"
    ))
    try:
        started = time.monotonic()
        signal_run_owned(run, child, signal.SIGKILL, deadline=started + 4,
                         what="delayed writer")
        assert child.wait(timeout=5) == -signal.SIGKILL
        assert marker.read_text() == "complete"
        assert time.monotonic() - started >= 1.4
        assert json.loads(events.read_text()) == {"event": "checkpoint_saved"}
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_missing_save_fails_without_signalling(tmp_path: Path) -> None:
    run = tmp_path / "run"
    child = _child(tmp_path, "import time\ntime.sleep(30)\n")
    try:
        started = time.monotonic()
        with pytest.raises(AssertionError) as error:
            signal_run_owned(run, child, signal.SIGKILL, deadline=started + 0.3,
                             what="never saved")
        assert time.monotonic() - started < 0.6
        assert str(run) in str(error.value)
        assert "never saved" in str(error.value)
        assert "saw 0/1 saves; last event: None" in str(error.value)
        assert child.poll() is None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_torn_save_line_is_not_counted(tmp_path: Path) -> None:
    run = tmp_path / "run"
    events = run / "checkpoints" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_bytes(b'{"event": "checkpoint_sa')
    with pytest.raises(AssertionError, match="saw 0/1 saves; last event: None"):
        wait_for_checkpoint_saved(run, deadline=time.monotonic() + 0.2,
                                  what="torn checkpoint")


def test_timeout_reports_complete_save_count_and_last_event(tmp_path: Path) -> None:
    run = tmp_path / "run"
    events = run / "checkpoints" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text('{"event":"checkpoint_saved"}\n'
                      '{"event":"checkpoint_started"}\n'
                      '{"event":"checkpoint_saved"')
    with pytest.raises(AssertionError, match="saw 1/2 saves; last event: 'checkpoint_started'"):
        wait_for_checkpoint_saved(run, deadline=time.monotonic() + 0.2, minimum=2,
                                  what="second checkpoint")


def test_explicit_opt_out_signals_before_first_save(tmp_path: Path) -> None:
    run = tmp_path / "run"
    child = _child(tmp_path, "import time\ntime.sleep(30)\n")
    try:
        signal_run_owned(run, child, signal.SIGKILL, deadline=time.monotonic(),
                         what="pre-checkpoint interruption", require_checkpoint_saved=False)
        assert child.wait(timeout=5) == -signal.SIGKILL
        assert not (run / "checkpoints" / "events.jsonl").exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)

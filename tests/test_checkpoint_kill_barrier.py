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
from _timeouts import scaled
from fixtures.process_signals import (
    record_identity,
    signal_run_owned,
    signal_run_verified,
    wait_for_checkpoint_saved,
)

TESTS = Path(__file__).resolve().parent

# These exercise process signalling, containment or a non-run helper, not a StateFrame run.
PRIMITIVE_TESTS = {
    "test_launch_unit_kill.py",
    "test_reason_log_kill_sites.py",  # exercises the kill primitives' log lines on its own sleepers, never a StateFrame run
    "test_orphan_guard.py",  # plants and stops detached processes, never a StateFrame run
    "test_orphan_guard_portability.py",  # plants and stops a detached launcher, never a StateFrame run
    "orphan_guard_mutants.py",  # contains deliberately incorrect process-signal mutants, never a StateFrame run
    "test_orphan_guard_acceptance.py",  # stops its own test helpers, never a StateFrame run
    "test_orphan_guard_unit.py",  # exercises the orphan-guard primitive, never a StateFrame run
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
    # Exercises a forked checkpoint writer directly; the only raw signal is timeout cleanup.
    "test_checkpoint_writer_reason_redaction.py",
    "test_checkpoint_kill_barrier.py",
    "test_exec_session.py",
    "test_exec_supervisor_parent_death.py",  # launches the supervisor primitive directly, not a StateFrame run
    "test_frame_routes_engine.py",
    "test_powerups_python_in_sandbox.py",
    "test_check_sandbox_nproc.py",
    "test_cgroup_leaf_pattern.py",
}

# Preserve the release's exact file/function/line/column authority for these
# mutation fixtures; the bridge's standalone fixtures use scoped counts below.
REVIEWED_RAW_SIGNALS = {
    ("test_mutation_linux_regressions.py", "_stop_process_group", 329, 8,
     "os.killpg(process.pid, signal.SIGTERM)"),
    ("test_mutation_linux_regressions.py", "_stop_process_group", 333, 12,
     "os.killpg(process.pid, signal.SIGKILL)"),
    ("test_mutation_linux_regressions.py", "_interrupt_and_resume", 380, 8,
     "os.kill(sender_pid, signal.SIGKILL)"),
}

APPROVED_TEST_SIGNALS: dict[tuple[str, str, str], tuple[int, str]] = {
    ('test_parallel_bridge_process.py', 'test_process_integration_sixteen_channels_thirtytwo_workers', 'process.terminate()'):
        (1, 'Stops only a command-verified standalone journal/seal helper after runtime shutdown.'),
    ('test_parallel_cursor_recovery.py', '_kill_recorded_child', 'os.kill(process.pid, signal.SIGKILL)'):
        (1, 'Stops the held cursor-write fixture child after checking its command marker.'),
    ('test_parallel_tick_table.py', 'test_pid_that_dies_between_ticks_leaves_the_family', 'os.kill(child, signal.SIGKILL)'):
        (1, 'Kills the child created by the held synthetic worker to test family eviction.'),
    ('test_parallel_tick_table.py', 'test_raw_stat_identity_handles_a_comm_with_parentheses_and_spaces', 'os.killpg(process.pid, signal.SIGKILL)'):
        (1, 'Reaps the held standalone stat-parser fixture process group.'),
    ('test_parallel_tick_table.py', 'procs', 'os.killpg(process.pid, signal.SIGKILL)'):
        (1, 'Reaps the fixture-owned synthetic worker groups, without model requests.'),
    ('test_parallel_bridge_recorded.py', '_kill_leader', 'process.kill()'):
        (1, 'Stops a held standalone sleeper after verifying its command on Linux.'),
    ('test_no_engine_wall_clock_stops.py', 'test_job_watch_dispatches_when_summed_job_wall_exceeds_old_cap', 'process.kill()'):
        (1, 'Stops the standalone sleep job this test launched through the job watch; no engine run.'),
    ('test_parallel_supervisor_scale.py', 'test_live_unit_reports_a_dead_unit_from_the_same_single_scan', 'os.killpg(handle.pgid, signal.SIGKILL)'):
        (1, 'Kills the held synthetic worker to test the shared process scan.'),
    ('test_parallel_supervisor_scale.py', 'test_siblings_are_polled_from_one_scan', 'os.killpg(handle.pgid, signal.SIGKILL)'):
        (1, 'Kills and reaps each held synthetic sibling before testing shared polling.'),
    ('test_parallel_supervisor_scale.py', 'procs', 'os.killpg(process.pid, signal.SIGKILL)'):
        (1, 'Reaps only the standalone sleeper groups created by this fixture.'),
    ("test_unit_checks.py", "process_boundary", "signal_owned(leader, 9)"):
        (1, "test-only process-boundary fixture cleanup"),
    ("test_unit_checks.py", "test_check_command_refuses_reaped_leader_with_live_child",
     "signal_verified(child_pid[0], child_identity[0], 0)"):
        (1, "identity-checked liveness probe in a process-boundary test"),
    ("test_unit_checks.py", "test_check_command_refuses_reaped_leader_with_live_child",
     "signal_verified(child_pid[0], child_identity[0], 9)"):
        (1, "identity-checked descendant cleanup in a process-boundary test"),
    ("test_unit_checks.py", "test_linux_check_ignores_unreadable_unrelated_environ",
     "unrelated.terminate()"):
        (1, "stop the unrelated sleeper created by this test"),
    ("test_unit_checks.py", "test_detached_check_child_cannot_publish_pass_or_write_later",
     "signal_verified(pid, identity, signal.SIGKILL)"):
        (1, "identity-checked cleanup of the detached test child"),
    ("test_parallel_signal_identity.py", "test_unreaped_worker_leader_with_wrong_birth_is_never_signalled",
     "process.kill()"):
        (1, "cleanup the held child after asserting wrong-birth rejection"),
    ("test_parallel_signal_identity.py", "test_macos_audit_token_signal_checks_birth_and_stops_owned_process",
     "process.kill()"):
        (1, "cleanup the held test child after signal identity assertions"),
    ("test_parallel_worker_prep.py", "_run_scripted_engine", "endpoint_process.terminate()"):
        (1, "stop the local scripted endpoint server"),
    ("test_parallel_worker_prep.py", "_run_scripted_engine", "endpoint_process.kill()"):
        (1, "bounded escalation for the local scripted endpoint server"),
    ("test_parallel_worker_prep.py", "test_spawn_to_process_record_crash_keeps_durable_launch_intent",
     "process.kill()"):
        (1, "stop the synthetic worker process used by the launch crash test"),
    ("test_parallel_resources.py",
     "test_slot_lease_is_exclusive_across_processes_and_released_when_holder_is_killed",
     "process.kill()"):
        (2, "stop the synthetic lease holders after the exclusivity assertions"),
    ("test_parallel_host.py", "test_real_scripted_worker_submission_seals_recorded_bytes_and_receipt",
     "endpoint_process.terminate()"):
        (1, "stop the local scripted endpoint server"),
    ("test_parallel_host.py",
     "test_interrupted_worker_resume_replays_exact_wire_without_authorizing_replayed_sends",
     "endpoint_process.terminate()"):
        (1, "stop the local scripted endpoint server"),
    ("test_parallel_host.py", "test_macos_process_argv_preserves_run_path_spaces",
     "signal_owned(process, signal.SIGTERM)"):
        (1, "stop the standalone argv fixture process"),
    ("test_parallel_host.py", "test_parent_exit_cancels_and_reaps_a_real_worker_process_tree",
     "signal_owned(process, 9)"):
        (1, "test teardown after verifying worker process-tree containment"),
    ("test_parallel_host.py", "test_exited_child_keeps_subframe_unsealable_until_process_group_is_gone",
     "signal_verified(pid, identity, signal.SIGKILL)"):
        (1, "identity-checked teardown of a synthetic worker descendant"),
    ("test_unit_visibility.py", "test_real_oversized_subframe_has_typed_terminal_and_zero_sends",
     "endpoint_process.terminate()"):
        (1, "stop the local scripted endpoint server"),
    ("test_check_permits.py", "test_two_processes_are_exclusive_and_kill_releases",
     "first.kill()"):
        (1, "stop the synthetic permit holder after the lock assertion"),
    ("test_check_permits.py", "test_two_permits_allow_two_holders_and_refuse_third",
     "child.kill()"):
        (1, "stop the synthetic permit holder after the lock assertion"),
    ("e2e/test_parallel_interrupt_safety.py", "test_signal_helpers_verify_launch_identity_before_delivery",
     "signal_verified(process.pid, type(identity)(process.pid, identity.started_epoch, 'different command', identity.kernel_identity), signal.SIGTERM)"):
        (1, "assert that a mismatched command cannot signal the standalone fixture"),
    ("e2e/test_parallel_interrupt_safety.py", "test_signal_helpers_verify_launch_identity_before_delivery",
     "signal_verified(process.pid, type(identity)(process.pid, identity.started_epoch, identity.command, 'darwin:0:0'), signal.SIGTERM)"):
        (1, "assert that a mismatched kernel birth cannot signal the standalone fixture"),
    ("e2e/test_parallel_interrupt_safety.py", "test_signal_helpers_verify_launch_identity_before_delivery",
     "signal_verified(os.getpid(), type(identity)(os.getpid(), identity.started_epoch, identity.command), signal.SIGTERM)"):
        (1, "assert that self-signalling is rejected"),
    ("e2e/test_parallel_interrupt_safety.py", "test_signal_helpers_verify_launch_identity_before_delivery",
     "signal_owned(process, signal.SIGTERM)"):
        (1, "stop the standalone fixture after identity assertions"),
    ("e2e/test_parallel_interrupt_safety.py", "test_signal_helpers_verify_launch_identity_before_delivery",
     "signal_owned(process, signal.SIGKILL)"):
        (1, "bounded cleanup if the standalone fixture ignores SIGTERM"),
    ("test_e2e_parallel_r7_slot_proof.py",
     "test_sigkilled_child_mid_handoff_keeps_slot_until_family_exit_proof",
     "os.kill(child.pid, signal.SIGKILL)"):
        (1, "SIGKILLs the held slot-handoff fixture child this test launched; no StateFrame run is signalled"),
    ("test_e2e_parallel_r7_slot_proof.py",
     "test_sigkilled_child_mid_handoff_keeps_slot_until_family_exit_proof", "child.kill()"):
        (1, "teardown of the same held fixture child"),
    ("test_probe_supervisor_loss.py", "_rpc", "os.kill(pid, signal.SIGKILL)"):
        (1, "host-side kill of the probe's own exec supervisor, the fault under test; not a StateFrame run"),
    ("test_parallel_bridge_r7_slot3.py",
     "test_sigkill_during_permanent_handoff_failure_keeps_slot_until_family_proof",
     "child.send_signal(signal.SIGTERM)"):
        (1, "stops the held slot-handoff fixture child this test launched"),
    ("test_parallel_bridge_r7_slot3.py",
     "test_sigkill_during_permanent_handoff_failure_keeps_slot_until_family_proof", "child.kill()"):
        (2, "SIGKILL and teardown of the held slot-handoff fixture child"),
    ("test_parallel_bridge_r6_slot2.py",
     "test_failed_handoff_keeps_slot_unavailable_until_claim_or_family_proof", "child.terminate()"):
        (1, "stops the held slot-handoff fixture child this test launched"),
    ("test_parallel_bridge_r6_slot2.py",
     "test_failed_handoff_keeps_slot_unavailable_until_claim_or_family_proof", "child.kill()"):
        (1, "teardown of the held slot-handoff fixture child"),
}


def _raw_run_signals(source: str, relative: Path) -> list[tuple[int, str, str]]:
    """Find direct signals in run tests, including imported aliases."""
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree)
               for child in ast.iter_child_nodes(parent)}
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
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def function_name(node: ast.AST) -> str:
        current = node
        while id(current) in parents:
            current = parents[id(current)]
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current.name
        return "<module>"

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
            owner, call = function_name(node), ast.unparse(node)
            site = (relative.as_posix(), owner, node.lineno, node.col_offset, call)
            if site not in REVIEWED_RAW_SIGNALS:
                violations.append((node.lineno, owner, call))
    return violations


def _unbarriered_signals(root: Path) -> dict[str, list[tuple[int, str]]]:
    violations: dict[str, list[tuple[int, str]]] = {}
    actual: dict[tuple[str, str, str], list[int]] = {}
    scanned_paths: set[str] = set()
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        scanned_paths.add(relative.as_posix())
        # The monitor owns and reaps its engine process group after the leader exits; run-kill decisions stay in tests.
        # The orphan guard only stops launchd-adopted leftovers, after their run has ended.
        if relative.as_posix() in {"fixtures/process_signals.py", "engine_run.py", "_orphan_guard.py"} or (
                relative.parts[0] != "acceptance" and path.name in PRIMITIVE_TESTS):
            continue
        found = _raw_run_signals(path.read_text(encoding="utf-8"), relative)
        for line, owner, call in found:
            actual.setdefault((relative.as_posix(), owner, call), []).append(line)
    for key, lines in actual.items():
        relative, _owner, call = key
        approved = APPROVED_TEST_SIGNALS.get(key)
        if approved is None:
            violations.setdefault(relative, []).extend((line, call) for line in lines)
        elif len(lines) != approved[0]:
            violations.setdefault(relative, []).append(
                (min(lines), f"reviewed signal count {len(lines)} != {approved[0]}: {call}"))
    for key, (expected, reason) in APPROVED_TEST_SIGNALS.items():
        relative, _owner, call = key
        if relative in scanned_paths and len(actual.get(key, ())) != expected:
            violations.setdefault(relative, []).append(
                (0, f"reviewed signal count {len(actual.get(key, ()))} != {expected}: {call} ({reason})"))
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
    names = ("orphan_guard_mutants.py", "test_orphan_guard_acceptance.py",
             "test_orphan_guard_unit.py")
    for name in names:
        (tmp_path / name).write_text(source, encoding="utf-8")
        (acceptance / name).write_text(source, encoding="utf-8")

    assert _unbarriered_signals(tmp_path) == {
        f"acceptance/{name}": [(2, "os.killpg(123, 9)")] for name in names
    }


@pytest.mark.parametrize("filename,function,call", [
    ("test_check_permits.py", "test_two_processes_are_exclusive_and_kill_releases", "first.kill()"),
    ("test_parallel_bridge_recorded.py", "_kill_leader", "process.kill()"),
])
def test_reviewed_test_signal_sites_reject_added_calls(tmp_path: Path, filename, function, call) -> None:
    source = tmp_path / filename
    source.write_text(f"def {function}():\n    {call}\n    {call}\n", encoding="utf-8")
    violations = _unbarriered_signals(tmp_path)
    assert any(f"reviewed signal count 2 != 1: {call}" in detail
               for _, detail in violations[filename])


def test_group_existence_probe_is_not_a_run_kill() -> None:
    assert not _raw_run_signals("import os\nos.killpg(pgid, 0)\n", Path("test_engine_run.py"))


@pytest.mark.parametrize("function,line,column,call", (
    ("_stop_process_group", 329, 8, "os.killpg(process.pid, signal.SIGTERM)"),
    ("_stop_process_group", 333, 12, "os.killpg(process.pid, signal.SIGKILL)"),
    ("_interrupt_and_resume", 380, 8, "os.kill(sender_pid, signal.SIGKILL)"),
))
def test_reviewed_raw_signals_are_call_site_exact(function: str, line: int,
                                                  column: int, call: str) -> None:
    source = (f"import os, signal\ndef {function}(process, sender_pid):\n"
              + "\n" * (line - 3) + f"{' ' * column}{call}\n")
    path = Path("test_mutation_linux_regressions.py")
    assert _raw_run_signals(source, path) == []
    assert _raw_run_signals(source, Path("other.py")) == [(line, function, call)]
    assert _raw_run_signals(source.replace(f"def {function}", "def other"), path) == [(line, "other", call)]
    assert _raw_run_signals("\n" + source, path) == [(line + 1, function, call)]
    duplicate = source.replace(f"{' ' * column}{call}\n",
                               f"{' ' * column}{call}; {call}\n")
    assert _raw_run_signals(duplicate, path) == [(line, function, call)]


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
        signal_run_owned(run, child, signal.SIGKILL, deadline=started + scaled(4),
                         what="delayed writer")
        assert child.wait(timeout=scaled(5)) == -signal.SIGKILL
        assert marker.read_text() == "complete"
        assert time.monotonic() - started >= 1.4
        assert json.loads(events.read_text()) == {"event": "checkpoint_saved"}
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=scaled(5))


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
            child.wait(timeout=scaled(5))


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
        assert child.wait(timeout=scaled(5)) == -signal.SIGKILL
        assert not (run / "checkpoints" / "events.jsonl").exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=scaled(5))


def test_verified_mid_request_interruption_needs_no_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    launched = time.time()
    child = _child(tmp_path, "import time\ntime.sleep(30)\n")
    try:
        identity = record_identity(child.pid, launched_after=launched,
                                   command_fragment=str(tmp_path / "checkpoint_control_writer.py"))
        signal_run_verified(run, child.pid, identity, signal.SIGKILL,
                            deadline=time.monotonic() + scaled(1),
                            what="held first provider request", require_checkpoint_saved=False)
        assert child.wait(timeout=scaled(5)) == -signal.SIGKILL
        assert not (run / "checkpoints" / "events.jsonl").exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=scaled(5))

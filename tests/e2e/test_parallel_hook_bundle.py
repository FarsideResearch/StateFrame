"""Inherited hook bundles survive real subframe startup and replay resume."""
from __future__ import annotations

import json
from pathlib import Path
import signal
import time

import pytest

from barrier_endpoint import BarrierEndpoint
from e2e_support import read_json, start_cli, subframe_directories, tool_step
from process_signals import record_identity, signal_run_owned, signal_run_verified
from stateframe.engine.kernel_hooks import verify_bundle
from stateframe.parallel_worker import launch_worker
from stateframe.runner import final_candidate_with_file_digests


pytestmark = pytest.mark.host_sandbox


def test_hook_enabled_parallel_worker_and_resume(
        tmp_path: Path, isolated_engine_python: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tests").mkdir()
    for unit in ("a", "b"):
        (repo / f"{unit}.py").write_text("VALUE = 1\n")
        (repo / f"tests/test_{unit}.py").write_text(
            f"from {unit} import VALUE\ndef test_value():\n    assert VALUE == 2\n")
    source = tmp_path / "hook-source"
    source.mkdir()
    (source / "guard.py").write_text("print('inherited worker guard')\n")
    config = source / "hooks.json"
    config.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [{
        "name": "guard", "event": "pre_tool", "tools": ["read", "str_replace"],
        "argv": [str(isolated_engine_python), "guard.py"], "files": ["guard.py"],
    }]}))
    # Complete a parent step before fan-out so interruption waits for a saved checkpoint.
    parent = [tool_step("read", {"path": "a.py"}), tool_step("parallel", {"subframes": [{
        "id": unit, "outcome": "Update VALUE", "owns": [f"{unit}.py"], "creates": [],
        "reads": [], "exports": [], "consumes": [], "requirements": [],
        "check": {"tests": [f"tests/test_{unit}.py"]},
    } for unit in ("a", "b")]})]
    workers = {unit: [tool_step("str_replace", {
        "path": f"{unit}.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2",
    })] for unit in ("a", "b")}
    # A read completes a hook and a checkpoint without rerunning time-varying failed checks.
    workers["a"].insert(0, tool_step("read", {"path": "a.py"}))
    run = tmp_path / "run"
    with BarrierEndpoint(parent, workers, hold_subframe_request={"a": 2},
                         barrier_timeout=1800) as endpoint:
        with start_cli(isolated_engine_python, endpoint.url, repo, run,
                       "Update VALUE using subframes.", arguments=("--hooks", str(config))) as process:
            assert endpoint.wait_for_requests("a", 2, timeout=1800), "worker never reached its second request"
            assert endpoint.wait_for_requests("b", 1, timeout=1800), "sibling worker never reached its provider"
            child = subframe_directories(run)["a"]
            child_spec = read_json(child / "spec.json")
            assert child_spec["hooks"] == read_json(run / "spec.json")["hooks"]
            verify_bundle(child_spec["hooks"]["bundle_manifest"], child / "hook-bundle")
            tape = (child / "hook-tape.jsonl").read_bytes()
            assert tape, "the read must complete its hook before interruption"
            launch = read_json(child / "proc/1/process.json")
            identity = record_identity(launch["pid"], launched_after=launch["started_epoch"],
                                       launched_before=launch["started_epoch"], command_fragment=str(child))
            signal_run_verified(child, launch["pid"], identity, signal.SIGKILL,
                                deadline=time.monotonic() + 180,
                                what="hook-enabled worker interruption")
            signal_run_owned(run, process, signal.SIGKILL, deadline=time.monotonic() + 180,
                             what="hook-enabled parent interruption", group=True)
            process.wait(timeout=180)

        # Restart through the worker launcher using the original CLI's home and pinned spec.
        monkeypatch.setenv("HOME", str(run.parent / "home"))
        handle = launch_worker(child, 2)
        try:
            handle.reap(timeout=1800)
            assert handle.exit_code == 0, handle.log_path.read_text()
        finally:
            if handle.exit_code is None:
                handle.cancel("test cleanup")
                handle.escalate(grace=0)
                handle.reap(timeout=180)
        endpoint.assert_consumed()

        assert read_json(child / "spec.json")["resume"]["attempt"] == 2
        verify_bundle(child_spec["hooks"]["bundle_manifest"], child / "hook-bundle")
        assert (child / "proc/2/process.json").is_file()
        assert (child / "hook-tape.jsonl").read_bytes().startswith(tape)
        record_paths = [child / "hooks/records.jsonl", child / "attempts/1/hooks/records.jsonl"]
        records = [json.loads(line) for path in record_paths for line in path.read_text().splitlines()]
        guards = [row for row in records if row["hook"] == "guard" and row["event"] == "pre_tool"]
        executed = [row for row in guards if row["executed"]]
        assert len(executed) == 2
        assert all(row["decision"] == "continue" and row["error_code"] is None for row in executed)
        assert any(row["replayed"] for row in guards)
        candidate, digests = final_candidate_with_file_digests(child)
        assert candidate is not None and candidate["a.py"] == b"VALUE = 2\n"
        assert digests is not None and "a.py" in digests

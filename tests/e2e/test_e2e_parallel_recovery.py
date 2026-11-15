"""Crash seams, replay and checkpoint proofs with real parent and subframe processes."""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import pytest
from _timeouts import scaled
from barrier_endpoint import BarrierEndpoint
from e2e_support import (
    assert_shipped_wire,
    assert_subframe_surface,
    crash_hook_arguments,
    interrupt_run,
    parallel_call_roots,
    parent_wire_files,
    read_json,
    result_records,
    start_cli,
    start_resume,
    subframe_directories,
    subframe_wire_files,
    tool_step,
    wait_for_parent_wire,
    wire_message_text,
)
from process_signals import record_identity, signal_run_verified

from stateframe import replay_tape
from stateframe.engine.kernel_parallel_txn import TxnStore

pytestmark = pytest.mark.host_sandbox


def _contract(unit_id: str, owns: list[str]) -> dict[str, object]:
    return {
        "id": unit_id,
        "outcome": "Update the owned source files.",
        "owns": owns,
        "creates": [],
        "exports": [],
        "consumes": [],
        "reads": [],
        "requirements": [],
        "check": {"tests": []},
    }


def _parallel(*contracts: dict[str, object]) -> dict[str, object]:
    return tool_step("parallel", {"subframes": list(contracts)})


def _assert_admitted(call_root: Path) -> None:
    intent = read_json(call_root / "intent.json")
    assert intent["admission"]["status"] == "ADMITTED"


def _two_file_fixture(root: Path) -> Path:
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg/a.py").write_text("VALUE = 1\n")
    (root / "pkg/b.py").write_text("OTHER = 1\n")
    (root / "tests/test_smoke.py").write_text("def test_smoke():\n    assert True\n")
    return root


def _two_file_scripts() -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    parent = [
        _parallel(_contract("pair", ["pkg/a.py", "pkg/b.py"])),
        tool_step("done", {"summary": "Both durable source changes are visible."}),
    ]
    subframes = {
        "pair": [
            tool_step("str_replace", {"path": "pkg/a.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"}),
            tool_step("str_replace", {"path": "pkg/b.py", "old_str": "OTHER = 1", "new_str": "OTHER = 2"}),
            tool_step("done", {"summary": "Both owned files are ready for publication."}),
        ],
    }
    return parent, subframes


def _dependent_crash_scripts() -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    parent, subframes = _two_file_scripts()
    pair = parent[0]["arguments"]["subframes"][0]
    pair["exports"] = [{"id": "pair.value", "path": "pkg/a.py", "symbol": "VALUE",
                        "signature": {"kind": "constant", "params": [],
                                      "returns": None, "decorators": []}}]
    dependent = _contract("dependent", ["pkg/c.py"])
    dependent["consumes"] = [{"export_id": "pair.value"}]
    parent[0]["arguments"]["subframes"].append(dependent)
    subframes["dependent"] = [
        tool_step("str_replace", {"path": "pkg/c.py", "old_str": "RESULT = 1",
                                   "new_str": "RESULT = 2"}),
        tool_step("done", {"summary": "The published interface was consumed."}),
    ]
    return parent, subframes


def _latest_state(store: TxnStore) -> tuple[int, str]:
    records = store.records()
    assert records
    latest: dict[int, tuple[object, str]] = {}
    for record, disposition in records:
        latest[record.txn_id] = (record, disposition)
    assert sorted(latest) == [1]
    return 1, latest[1][1]


def _wire_number(path: Path) -> int:
    return int(path.name.split(".", 1)[0])


def _request_for_number(paths: list[Path], number: int) -> Path:
    matches = [path for path in paths if _wire_number(path) == number]
    assert matches, f"request {number} was not recorded: {paths}"
    return matches[-1]


def _parallel_disposition(path: Path) -> dict[str, object]:
    found = [payload["state"]["framefit5"]["action_ledger"].get("latest_parallel_disposition")
             for payload in assert_shipped_wire([path])]
    dispositions = [value for value in found if isinstance(value, dict)]
    assert len(dispositions) == 1, f"expected one parallel disposition in {path}"
    assert dispositions[0].get("schema") == "stateframe-parallel-disposition/1"
    return dispositions[0]


@pytest.mark.parametrize(
    ("phase", "expected_state", "expected_changed"),
    [
        ("after_decided", "DECIDED", 0),
        ("between_materialized_files", "DECIDED", 1),
        ("after_materialized", "MATERIALIZED", 2),
        ("after_published", "PUBLISHED", 2),
        ("after_ack", "PUBLISHED", 2),
    ],
)
def test_crash_publication_recovers_one_commit_and_rebuilds_next_request_byte_identically(
        tmp_path: Path, isolated_engine_python: Path, phase: str,
        expected_state: str, expected_changed: int) -> None:
    from e2e_support import install_crash_hook

    repo = _two_file_fixture(tmp_path / "repo")
    (repo / "pkg/c.py").write_text("RESULT = 1\n")
    parent, subframes = _dependent_crash_scripts()
    run_dir = tmp_path / "run"
    marker = tmp_path / "crash-marker.txt"
    install_crash_hook(isolated_engine_python)

    with BarrierEndpoint(parent, subframes, hold_parent_request=2,
                         barrier_timeout=scaled(300)) as endpoint:
        publication_at_first_send: list[bool] = []
        original_arrival = endpoint._arrive_and_wait

        def observe_arrival(marker: str):
            if marker == "dependent" and not publication_at_first_send:
                records = TxnStore(run_dir / "parallel" / parallel_call_roots(run_dir)[0].name /
                                   "txn").records()
                publication_at_first_send.append(any(
                    record.txn_id == 1 and disposition == "PUBLISHED"
                    for record, disposition in records))
            return original_arrival(marker)

        endpoint._arrive_and_wait = observe_arrival
        process = start_cli(
            isolated_engine_python, endpoint.url, repo, run_dir,
            "Apply both declared source updates and publish one logical transaction.",
            parallel="2",
            arguments=("--parallel-min-subframes", "1", *crash_hook_arguments(phase, marker)),
        )
        process.wait(timeout=900)
        call_root, = parallel_call_roots(run_dir)
        _assert_admitted(call_root)
        assert marker.read_text() == phase
        store = TxnStore(call_root / "txn")
        assert _latest_state(store) == (1, expected_state)
        continuation_workspaces = list((run_dir / "worker" / "continuation-workspaces").glob("*/repo"))
        assert len(continuation_workspaces) == 1
        workspace = continuation_workspaces[0]
        changed = sum((workspace / path).read_text().endswith("= 2\n")
                      for path in ("pkg/a.py", "pkg/b.py"))
        assert changed == expected_changed
        assert not (call_root / "result.json").exists()
        assert len([row for row in endpoint.requests if row["marker"] == "pair"]) == 3

        first_resume = start_resume(isolated_engine_python, endpoint.url, run_dir)
        try:
            first_request = wait_for_parent_wire(run_dir, 2, first_resume,
                                                 timeout=scaled(240))
            first_bytes = first_request.read_bytes()
            assert "VALUE = 2" in wire_message_text(first_request)
            interrupt_run(first_resume, run_dir)
        finally:
            if first_resume.poll() is None:
                interrupt_run(first_resume, run_dir)

        endpoint.release_parent_request()
        second_resume = start_resume(isolated_engine_python, endpoint.url, run_dir)
        assert second_resume.wait(timeout=900) == 0
        endpoint.assert_consumed()

        second_request = _request_for_number(parent_wire_files(run_dir), 2)
        assert second_request.read_bytes() == first_bytes
        assert "VALUE = 2" in wire_message_text(second_request)
        assert_shipped_wire(parent_wire_files(run_dir))
        children = subframe_directories(run_dir)
        assert set(children) == {"pair", "dependent"}
        for path in subframe_wire_files(children["pair"]):
            assert_subframe_surface(path)
        assert_shipped_wire(subframe_wire_files(children["pair"]), task_marker="Subframe pair:")
        assert_shipped_wire(subframe_wire_files(children["dependent"]),
                            task_marker="Subframe dependent:")
        assert publication_at_first_send == [True]

    final_records = store.records()
    assert [record.txn_id for record, disposition in final_records
            if disposition == "PUBLISHED"] == [1, 2]
    result, = result_records(run_dir).values()
    assert result.txn_order == (1, 2)
    outcomes_by_txn = sorted(((item.unit_id, item.status, item.txn_id) for item in result.outcomes),
                             key=lambda item: item[2])
    assert outcomes_by_txn == [
        ("pair", "COMMITTED_UNVERIFIED", 1),
        ("dependent", "COMMITTED_UNVERIFIED", 2),
    ]
    assert (run_dir / "workspace/pkg/a.py").read_text() == "VALUE = 2\n"
    assert (run_dir / "workspace/pkg/b.py").read_text() == "OTHER = 2\n"
    assert (run_dir / "workspace/pkg/c.py").read_text() == "RESULT = 2\n"
    outcome_lines = (call_root / "outcomes.jsonl").read_text().splitlines()
    assert len(outcome_lines) == 2
    assert json.loads(outcome_lines[0])["txn_id"] == 1


def test_killed_subframe_mid_request_restarts_same_execution_and_commits_once(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _two_file_fixture(tmp_path / "repo")
    parent = [
        _parallel(_contract("pair", ["pkg/a.py", "pkg/b.py"])),
        tool_step("done", {"summary": "The restarted subframe committed both files once."}),
    ]
    subframes = {
        "pair": [
            tool_step("str_replace", {"path": "pkg/a.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"}),
            tool_step("str_replace", {"path": "pkg/b.py", "old_str": "OTHER = 1", "new_str": "OTHER = 2"}),
            tool_step("done", {"summary": "Both owned files are ready."}),
        ],
    }
    run_dir = tmp_path / "run"

    with BarrierEndpoint(parent, subframes, hold_subframe_request={"pair": 1},
                         barrier_timeout=scaled(300)) as endpoint:
        process = start_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Recover the active subframe after a process interruption.", parallel="2",
                            arguments=("--parallel-min-subframes", "1"))
        assert endpoint.wait_for_requests("pair", 1, timeout=scaled(240)), (
            "subframe never entered its provider request")
        call_root, = parallel_call_roots(run_dir)
        _assert_admitted(call_root)
        spec_path = call_root / "units/pair/unit-spec.json"
        spec_before = read_json(spec_path)
        first_launch = read_json(call_root / "host/units/pair/1.json")
        process_record = read_json(call_root / "units/pair/proc/1/process.json")
        pid, pgid = process_record.get("pid"), process_record.get("pgid")
        assert type(pid) is int and type(pgid) is int and pid == pgid and pid > 0
        assert (first_launch["pid"], first_launch["pgid"]) == (pid, pgid)
        assert pgid != os.getpgrp()
        identity = record_identity(pid, launched_after=process_record["started_epoch"],
                                   launched_before=process_record["started_epoch"],
                                   command_fragment=str(call_root / "units/pair"))
        # The provider is holding request 1, so this worker has no saved checkpoint yet.
        signal_run_verified(call_root / "units" / "pair", pid, identity, signal.SIGKILL,
                            deadline=time.monotonic() + 30,
                            what="held subframe worker interruption",
                            require_checkpoint_saved=False)
        interrupt_run(process, run_dir, require_checkpoint_saved=False)

        resumed = start_resume(isolated_engine_python, endpoint.url, run_dir)
        assert resumed.wait(timeout=900) == 0
        endpoint.assert_consumed()

        second_launch = read_json(call_root / "host/units/pair/2.json")
        assert second_launch["incarnation"] == 2
        assert read_json(spec_path) == spec_before
        assert second_launch["pid"] != pid
        assert [row["marker"] for row in endpoint.requests].count("pair") >= 3
        result, = result_records(run_dir).values()
        assert result.txn_order == (1,)
        assert [(item.unit_id, item.status, item.txn_id) for item in result.outcomes] == [
            ("pair", "COMMITTED_UNVERIFIED", 1),
        ]
        assert_shipped_wire(parent_wire_files(run_dir))
        children = subframe_directories(run_dir)
        assert set(children) == {"pair"}
        assert_shipped_wire(subframe_wire_files(children["pair"]), task_marker="Subframe pair:")

    states = {record.txn_id: disposition
              for record, disposition in TxnStore(call_root / "txn").records()}
    assert states == {1: "PUBLISHED"}
    assert (run_dir / "workspace/pkg/a.py").read_text() == "VALUE = 2\n"
    assert (run_dir / "workspace/pkg/b.py").read_text() == "OTHER = 2\n"


def test_completed_parallel_replay_keeps_observation_and_skips_subframe_launches(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _two_file_fixture(tmp_path / "repo")
    parent, subframes = _two_file_scripts()
    run_dir = tmp_path / "run"

    with BarrierEndpoint(parent, subframes, hold_parent_request=2,
                         barrier_timeout=scaled(300)) as endpoint:
        process = start_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Replay the completed parallel result without relaunching subframes.", parallel="2",
                            arguments=("--parallel-min-subframes", "1"))
        assert endpoint.wait_for_requests("parent", 2, timeout=scaled(240)), (
            "parallel result was not followed by a parent request")
        call_root, = parallel_call_roots(run_dir)
        _assert_admitted(call_root)
        result_before = (call_root / "result.json").read_bytes()
        next_parent_before = _request_for_number(parent_wire_files(run_dir), 2).read_bytes()
        receipt_before = _parallel_disposition(_request_for_number(parent_wire_files(run_dir), 2))
        assert receipt_before["call_id"] == call_root.name
        launch_files_before = sorted((call_root / "host/units/pair").glob("[0-9]*.json"))
        provider_requests_before = sum(row["marker"] == "pair" for row in endpoint.requests)
        worker_sends_before = len((call_root / "units/pair/sends.jsonl").read_text().splitlines())
        interrupt_run(process, run_dir)

        resumed = start_resume(isolated_engine_python, endpoint.url, run_dir)
        assert resumed.wait(timeout=900) == 0
        endpoint.assert_consumed()

        assert (call_root / "result.json").read_bytes() == result_before
        assert sorted((call_root / "host/units/pair").glob("[0-9]*.json")) == launch_files_before
        assert sum(row["marker"] == "pair" for row in endpoint.requests) == provider_requests_before
        assert len((call_root / "units/pair/sends.jsonl").read_text().splitlines()) == worker_sends_before
        next_parent_after = _request_for_number(parent_wire_files(run_dir), 2).read_bytes()
        assert replay_tape.normalized_digest(next_parent_after) == replay_tape.normalized_digest(next_parent_before)
        assert next_parent_after == next_parent_before
        receipt_after = _parallel_disposition(_request_for_number(parent_wire_files(run_dir), 2))
        assert receipt_after == receipt_before
        result, = result_records(run_dir).values()
        assert result.observation.encode("utf-8") == json.loads(result_before)["observation"].encode("utf-8")
        assert result.txn_order == (1,)
        assert_shipped_wire(parent_wire_files(run_dir))
        children = subframe_directories(run_dir)
        assert_shipped_wire(subframe_wire_files(children["pair"]), task_marker="Subframe pair:")


def test_checkpoint_after_parallel_step_rebuilds_the_next_request_byte_identically(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _two_file_fixture(tmp_path / "repo")
    parent, subframes = _two_file_scripts()
    run_dir = tmp_path / "run"

    with BarrierEndpoint(parent, subframes, hold_parent_request=2,
                         barrier_timeout=scaled(300)) as endpoint:
        process = start_cli(
            isolated_engine_python, endpoint.url, repo, run_dir,
            "Resume from the checkpoint captured after the parallel step.", parallel="2",
            arguments=("--parallel-min-subframes", "1"),
            extra_environment={"STATEFRAME_CHECKPOINT": "save",
                               "STATEFRAME_CHECKPOINT_EVERY": "1",
                               "STATEFRAME_CHECKPOINT_KEEP": "2"},
        )
        assert endpoint.wait_for_requests("parent", 2, timeout=scaled(300))
        call_root, = parallel_call_roots(run_dir)
        _assert_admitted(call_root)
        checkpoint_files = sorted((run_dir / "checkpoints").glob("*/checkpoint.json"))
        assert checkpoint_files, "checkpoint was not saved after the parallel step"
        checkpoint_meta = read_json(checkpoint_files[-1])
        assert checkpoint_meta["sends"] == 1
        assert (call_root / "result.json").is_file()
        request_before = _request_for_number(parent_wire_files(run_dir), 2).read_bytes()
        interrupt_run(process, run_dir)

        resumed = start_resume(isolated_engine_python, endpoint.url, run_dir)
        assert resumed.wait(timeout=900) == 0
        endpoint.assert_consumed()

        spec = read_json(run_dir / "spec.json")
        assert spec["resume"]["checkpoint"]["sends"] == 1
        request_after = _request_for_number(parent_wire_files(run_dir), 2)
        assert request_after.read_bytes() == request_before
        assert "VALUE = 2" in wire_message_text(request_after)
        assert_shipped_wire(parent_wire_files(run_dir))
        children = subframe_directories(run_dir)
        assert set(children) == {"pair"}
        assert_shipped_wire(subframe_wire_files(children["pair"]), task_marker="Subframe pair:")
        result, = result_records(run_dir).values()
        assert result.txn_order == (1,)

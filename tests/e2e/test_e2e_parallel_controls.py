"""Lifecycle, frozen-check, visibility, and contract proofs through real wires."""
from __future__ import annotations

from pathlib import Path
import shutil
import signal
import time

import pytest

from _journal import measurement_events, measurement_journal_paths
from barrier_endpoint import BarrierEndpoint
from process_signals import signal_run_owned
from e2e_support import (
    assert_shipped_wire,
    assert_subframe_surface,
    parallel_call_roots,
    parent_wire_files,
    read_json,
    result_records,
    run_cli,
    start_cli,
    subframe_directories,
    subframe_wire_files,
    tool_names,
    tool_step,
    wire_message_text,
)
from stateframe.engine.kernel_parallel_contracts import canonical_signature
from stateframe.engine.kernel_parallel_records import CheckReceipt, Proposal
from stateframe.engine.kernel_parallel_txn import TxnStore, manifest_digest


pytestmark = pytest.mark.host_sandbox


def _contract(unit_id: str, outcome: str, *, owns: list[str] | None = None,
              creates: list[str] | None = None, exports: list[dict] | None = None,
              consumes: list[dict] | None = None, reads: list[str] | None = None,
              checks: list[str] | None = None) -> dict:
    return {"id": unit_id, "outcome": outcome, "owns": owns or [], "creates": creates or [],
            "exports": exports or [], "consumes": consumes or [], "reads": reads or [],
            "requirements": [], "check": {"tests": checks or []}}


def _parallel(*contracts: dict) -> dict:
    return tool_step("parallel", {"subframes": list(contracts)})


def _simple_repo(root: Path) -> Path:
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg/__init__.py").write_text("")
    (root / "tests/test_smoke.py").write_text("def test_smoke():\n    assert True\n")
    return root


def _call_result(run_dir: Path):
    result, = result_records(run_dir).values()
    return result


def _worker_check_receipt(call_root: Path, unit_id: str) -> CheckReceipt:
    """The subframe's own check receipt from its terminal result.

    Sealed proposals carry only a PASS (or NOT_RUN) worker receipt; a FAIL or INVALID
    receipt is dropped from the proposal and the composition check decides, so the
    worker's raw verdict and evidence are read from its terminal envelope.
    """
    envelope = read_json(call_root / "units" / unit_id / "engine-result.json")
    return CheckReceipt.from_json(envelope["result"]["check_receipt"])


def _assert_all_subframe_wires(run_dir: Path, ids: set[str]) -> dict[str, Path]:
    children = subframe_directories(run_dir)
    assert set(children) == ids
    for unit_id, directory in children.items():
        wires = subframe_wire_files(directory)
        assert_shipped_wire(wires, task_marker=f"Subframe {unit_id}:")
        for wire in wires:
            assert_subframe_surface(wire)
    assert_shipped_wire(parent_wire_files(run_dir))
    return children


def test_red_check_continues_and_no_check_requires_model_done(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _simple_repo(tmp_path / "repo")
    (repo / "pkg/a.py").write_text("VALUE = 0\n")
    (repo / "pkg/b.py").write_text("OTHER = 0\n")
    (repo / "tests/test_a.py").write_text(
        "from pkg.a import VALUE\n\ndef test_value():\n    assert VALUE == 2\n")
    parent = [
        _parallel(
            _contract("a", "Set the tested value to two.", owns=["pkg/a.py"],
                      checks=["tests/test_a.py"]),
            _contract("b", "Update the independent value.", owns=["pkg/b.py"]),
        ),
        tool_step("done", {"summary": "Both subframes finished."}),
    ]
    children = {
        "a": [
            tool_step("str_replace", {"path": "pkg/a.py", "old_str": "VALUE = 0",
                                      "new_str": "VALUE = 1"}),
            tool_step("str_replace", {"path": "pkg/a.py", "old_str": "VALUE = 1",
                                      "new_str": "VALUE = 2"}),
        ],
        "b": [
            tool_step("str_replace", {"path": "pkg/b.py", "old_str": "OTHER = 0",
                                      "new_str": "OTHER = 1"}),
            tool_step("done", {"summary": "The no-check edit is complete."}),
        ],
    }
    run_dir = tmp_path / "run"
    with BarrierEndpoint(parent, children) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Set the tested value to two and update the independent value.",
                            parallel="2")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()
    wires = _assert_all_subframe_wires(run_dir, {"a", "b"})
    assert {name: len(subframe_wire_files(path)) for name, path in wires.items()} == {
        "a": 2, "b": 2}
    assert {item.unit_id: item.check for item in _call_result(run_dir).outcomes} == {
        "a": "PASS", "b": "NOT_RUN"}
    assert len(measurement_journal_paths(wires["a"])) == 1
    events = [event for event in measurement_events(wires["a"])
              if event.get("kind") == "unit_check"]
    assert [event["status"] for event in events] == ["FAIL", "PASS"]


def test_creation_lifecycle_and_no_check_subframe(tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _simple_repo(tmp_path / "repo")
    (repo / "pkg/other.py").write_text("VALUE = 0\n")
    (repo / "tests/test_create.py").write_text(
        "def test_created_module():\n"
        "    from pkg.new import value\n"
        "    assert value() == 1\n")
    parent = [
        _parallel(
            _contract("create", "Create and refine a new module.", creates=["pkg/new.py"],
                      checks=["tests/test_create.py"]),
            _contract("plain", "Update the existing value.", owns=["pkg/other.py"]),
        ),
        tool_step("done", {"summary": "The declared file is created, checked, and committed."}),
    ]
    children = {
        "create": [
            tool_step("file_create", {"path": "pkg/new.py", "content": "def value():\n    return 0\n"}),
            tool_step("str_replace", {"path": "pkg/new.py", "old_str": "return 0", "new_str": "return 1"}),
        ],
        "plain": [
            tool_step("str_replace", {"path": "pkg/other.py", "old_str": "VALUE = 0", "new_str": "VALUE = 1"}),
            tool_step("done", {"summary": "The no-check source update is ready."}),
        ],
    }
    run_dir = tmp_path / "run"
    with BarrierEndpoint(parent, children) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Add the new pricing module in a declared subframe.", parallel="2")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()

    result = _call_result(run_dir)
    outcomes = {item.unit_id: item for item in result.outcomes}
    assert (outcomes["create"].status, outcomes["create"].check) == ("COMMITTED", "PASS")
    assert (outcomes["plain"].status, outcomes["plain"].check) == ("COMMITTED_UNVERIFIED", "NOT_RUN")
    assert result.final_check == "PASS"
    assert result.integration == "VERIFIED"
    call_root, = parallel_call_roots(run_dir)
    create_proposal = Proposal.from_json(read_json(call_root / "proposals" / "create" / "proposal.json"))
    assert [(entry.path, entry.kind) for entry in create_proposal.entries] == [("pkg/new.py", "create")]
    assert create_proposal.check_receipt is not None
    assert create_proposal.check_receipt.status == "PASS"
    assert not (repo / "pkg/new.py").exists(), "subframe creation escaped the saved candidate"
    assert (run_dir / "workspace/pkg/new.py").read_text() == "def value():\n    return 1\n"
    child_wires = _assert_all_subframe_wires(run_dir, {"create", "plain"})
    assert len(subframe_wire_files(child_wires["create"])) == 2
    assert len(subframe_wire_files(child_wires["plain"])) == 2
    assert "file_create" in tool_names(subframe_wire_files(child_wires["create"])[0])
    assert "run_tests" in tool_names(subframe_wire_files(child_wires["create"])[0])
    assert "run_tests" not in tool_names(subframe_wire_files(child_wires["plain"])[0])


def test_created_file_with_undeclared_sibling_import_is_refused_after_seal(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _simple_repo(tmp_path / "repo")
    (repo / "pkg/a.py").write_text("VALUE = 1\n")
    (repo / "pkg/b.py").write_text("OTHER = 1\n")
    parent = [
        _parallel(
            _contract("a", "Update a value.", owns=["pkg/a.py"]),
            _contract("b", "Create a helper.", owns=["pkg/b.py"], creates=["pkg/new.py"]),
        ),
        tool_step("done", {"summary": "The undeclared import was refused."}),
    ]
    children = {
        "a": [tool_step("str_replace", {"path": "pkg/a.py", "old_str": "VALUE = 1",
                                        "new_str": "VALUE = 2"}),
              tool_step("done", {"summary": "The owned value is ready."})],
        "b": [tool_step("file_create", {"path": "pkg/new.py",
                                        "content": "from pkg.a import VALUE\nRESULT = VALUE\n"}),
              tool_step("done", {"summary": "The helper is ready."})],
    }
    run_dir = tmp_path / "run"
    with BarrierEndpoint(parent, children) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Reject a new subframe file that imports an undeclared sibling.", parallel="2")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()
    result = _call_result(run_dir)
    assert result.admission.status == "ADMITTED"
    outcomes = {item.unit_id: item for item in result.outcomes}
    assert outcomes["b"].status == "REFUSED"
    assert outcomes["b"].reason == "UNDECLARED_COUPLING"
    assert not (run_dir / "workspace/pkg/new.py").exists()
    assert_shipped_wire(parent_wire_files(run_dir))
    _assert_all_subframe_wires(run_dir, {"a", "b"})


def test_check_scope_refusal_and_writes_invalidate_receipt_without_candidate_effect(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _simple_repo(tmp_path / "repo")
    (repo / "pkg/a.py").write_text("VALUE = 1\n")
    (repo / "pkg/b.py").write_text("OTHER = 1\n")
    # The parent's own public-test run also collects this file (it must be test_*.py), so the
    # writes happen only inside the disposable frozen-check clone, never in the live workspace.
    (repo / "tests/test_selected.py").write_text(
        "from pathlib import Path\n"
        "def test_mutating_check():\n"
        "    if 'stateframe-unit-check-' not in str(Path.cwd()):\n"
        "        return\n"
        "    Path('pkg/a.py').write_text('VALUE = 99\\n')\n"
        "    Path('created_by_check.txt').write_text('not allowed\\n')\n")
    (repo / "tests/other.py").write_text("def test_not_selected():\n    assert False\n")
    parent = [
        _parallel(
            _contract("a", "Update a and verify only the frozen selector.", owns=["pkg/a.py"],
                      checks=["tests/test_selected.py::test_mutating_check"]),
            _contract("b", "Update b independently.", owns=["pkg/b.py"]),
        ),
        tool_step("done", {"summary": "The attempted check mutation was rejected."}),
    ]
    children = {
        "a": [
            tool_step("run_tests", {"target": "tests/other.py"}),
            tool_step("str_replace", {"path": "pkg/a.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"}),
            tool_step("run_tests", {"target": "tests/test_selected.py::test_mutating_check"}),
            tool_step("done", {"summary": "The selected check cannot certify writes."}),
        ],
        "b": [
            tool_step("str_replace", {"path": "pkg/b.py", "old_str": "OTHER = 1", "new_str": "OTHER = 2"}),
            tool_step("done", {"summary": "The independent update is ready."}),
        ],
    }
    run_dir = tmp_path / "run"
    with BarrierEndpoint(parent, children) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Update these modules and preserve each check's declared scope.", parallel="2")
        endpoint.assert_consumed()
    assert (run_dir / "summary.json").is_file(), completed.stdout + completed.stderr

    result = _call_result(run_dir)
    outcomes = {item.unit_id: item for item in result.outcomes}
    assert (outcomes["a"].status, outcomes["a"].reason) == ("REFUSED", "COMPOSITION_CHECK_FAILED")
    assert outcomes["b"].status == "COMMITTED_UNVERIFIED"
    call_root, = parallel_call_roots(run_dir)
    spec = read_json(call_root / "units/a/unit-spec.json")
    assert spec["contract"]["check"]["selectors"] == ["tests/test_selected.py::test_mutating_check"]
    proposal = Proposal.from_json(read_json(call_root / "proposals/a/proposal.json"))
    assert proposal.check_receipt is None
    receipt = _worker_check_receipt(call_root, "a")
    assert (receipt.status, receipt.invalid_reason) == ("INVALID", "CHECK_WROTE_FILES")
    assert receipt.wrote_paths == ("created_by_check.txt", "pkg/a.py")
    assert "pytest exit 0" in Path(receipt.raw_evidence_ref).read_text()
    assert (run_dir / "workspace/pkg/a.py").read_text() == "VALUE = 1\n"
    assert not (run_dir / "workspace/created_by_check.txt").exists()
    assert (repo / "pkg/a.py").read_text() == "VALUE = 1\n"
    children = _assert_all_subframe_wires(run_dir, {"a", "b"})
    assert len(subframe_wire_files(children["a"])) == 4
    a_text = "\n".join(wire_message_text(path) for path in subframe_wire_files(children["a"]))
    assert "SCOPE_VIOLATION" in a_text
    assert result.final_check in {"INVALID", "FAIL"}


def test_withheld_read_and_projected_pytest_diagnostics_keep_raw_evidence(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _simple_repo(tmp_path / "repo")
    (repo / "pkg/a.py").write_text("VALUE = 1\n")
    (repo / "pkg/hidden.py").write_text(
        "def explode():\n"
        "    raise RuntimeError('HIDDEN_EXCEPTION_SENTINEL')\n")
    (repo / "pkg/other.py").write_text("OTHER = 1\n")
    (repo / "tests/test_a.py").write_text(
        "def test_failure():\n"
        # Built at run time so the literal is in the captured output, never in this visible source.
        "    print('captured pkg/hidden.py ' + 'HIDDEN_' + 'CAPTURED_SENTINEL')\n"
        "    from pkg.hidden import explode\n"
        "    explode()\n")
    parent = [
        _parallel(
            _contract("a", "Run a scoped diagnostic check.", owns=["pkg/a.py"],
                      checks=["tests/test_a.py"]),
            _contract("other", "Update the independent file.", owns=["pkg/other.py"]),
        ),
        tool_step("done", {"summary": "The withheld-file diagnostic was projected."}),
    ]
    children = {
        "a": [
            tool_step("read", {"path": "pkg/hidden.py"}),
            tool_step("run_tests", {"target": "tests/test_a.py"}),
            tool_step("done", {"summary": "The diagnostic belongs to withheld source."}),
        ],
        "other": [
            tool_step("str_replace", {"path": "pkg/other.py", "old_str": "OTHER = 1", "new_str": "OTHER = 2"}),
            tool_step("done", {"summary": "The unrelated update is ready."}),
        ],
    }
    run_dir = tmp_path / "run"
    with BarrierEndpoint(parent, children) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Keep sibling source private while checking this subframe.", parallel="2")
        endpoint.assert_consumed()
    assert (run_dir / "summary.json").is_file(), completed.stdout + completed.stderr

    result = _call_result(run_dir)
    call_root, = parallel_call_roots(run_dir)
    proposal = Proposal.from_json(read_json(call_root / "proposals/a/proposal.json"))
    assert proposal.check_receipt is None
    receipt = _worker_check_receipt(call_root, "a")
    assert receipt.status == "FAIL"
    raw = Path(receipt.raw_evidence_ref).read_text()
    assert "HIDDEN_CAPTURED_SENTINEL" in raw
    assert "HIDDEN_EXCEPTION_SENTINEL" in raw
    assert "pkg/hidden.py" in raw
    child_dirs = _assert_all_subframe_wires(run_dir, {"a", "other"})
    a_wires = subframe_wire_files(child_dirs["a"])
    a_text = "\n".join(wire_message_text(path) for path in a_wires)
    assert "HIDDEN_PATH" in a_text
    assert "[frame in withheld file pkg/hidden.py:2]" in a_text
    assert "HIDDEN_CAPTURED_SENTINEL" not in a_text
    assert "HIDDEN_EXCEPTION_SENTINEL" not in a_text
    assert result.observation.find("HIDDEN_EXCEPTION_SENTINEL") == -1


def test_signature_mismatch_records_exact_canonical_shapes_and_keeps_manifest(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _simple_repo(tmp_path / "repo")
    before = "def read_value(value):\n    return value\n"
    (repo / "pkg/api.py").write_text(before)
    (repo / "pkg/other.py").write_text("VALUE = 0\n")
    expected = canonical_signature(before, "read_value").to_json()
    export = {"id": "api.read_value", "path": "pkg/api.py", "symbol": "read_value",
              "signature": expected}
    parent = [
        _parallel(
            _contract("api", "Change read_value's body without changing its signature.",
                      owns=["pkg/api.py"], exports=[export]),
            _contract("other", "Change the independent value.", owns=["pkg/other.py"]),
        ),
        tool_step("done", {"summary": "The public signature mismatch was rejected."}),
    ]
    children = {
        "api": [
            tool_step("str_replace", {"path": "pkg/api.py", "old_str": "def read_value(value):",
                                       "new_str": "def read_value(value, fallback):"}),
            tool_step("done", {"summary": "The incompatible signature is submitted for verification."}),
        ],
        "other": [
            tool_step("str_replace", {"path": "pkg/other.py", "old_str": "VALUE = 0", "new_str": "VALUE = 1"}),
            tool_step("done", {"summary": "The independent update is ready."}),
        ],
    }
    before_manifest = manifest_digest(repo)
    run_dir = tmp_path / "run"
    with BarrierEndpoint(parent, children) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Change the API implementation without changing its contract.", parallel="2")
        endpoint.assert_consumed()
    assert (run_dir / "summary.json").is_file(), completed.stdout + completed.stderr

    result = _call_result(run_dir)
    api = next(item for item in result.outcomes if item.unit_id == "api")
    assert (api.status, api.reason) == ("REFUSED", "SIGNATURE_MISMATCH")
    call_root, = parallel_call_roots(run_dir)
    report = read_json(call_root / "refused-contracts" / "api.json")
    verdict = report["verdicts"][0]
    assert verdict["status"] == "FAIL"
    assert verdict["reason"] == "SIGNATURE_MISMATCH"
    assert verdict["expected"] == {
        "kind": "function",
        "params": [{"name": "value", "kind": "pos", "has_default": False, "annotation": None}],
        "returns": None,
        "decorators": [],
    }
    assert verdict["actual"] == {
        "kind": "function",
        "params": [
            {"name": "value", "kind": "pos", "has_default": False, "annotation": None},
            {"name": "fallback", "kind": "pos", "has_default": False, "annotation": None},
        ],
        "returns": None,
        "decorators": [],
    }
    assert manifest_digest(repo) == before_manifest
    _assert_all_subframe_wires(run_dir, {"api", "other"})


def test_blocked_producer_cancels_declared_consumer_but_commits_independent_subframe(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _simple_repo(tmp_path / "repo")
    source = "def value():\n    return 1\n"
    (repo / "pkg/a.py").write_text(source)
    (repo / "pkg/b.py").write_text("VALUE = 0\n")
    (repo / "pkg/c.py").write_text("def consume():\n    return 0\n")
    export = {"id": "a.value", "path": "pkg/a.py", "symbol": "value",
              "signature": canonical_signature(source, "value").to_json()}
    parent = [
        _parallel(
            _contract("a", "Provide the value API.", owns=["pkg/a.py"], exports=[export]),
            _contract("b", "Update an independent constant.", owns=["pkg/b.py"]),
            _contract("c", "Consume the declared value API.", owns=["pkg/c.py"],
                      consumes=[{"export_id": "a.value"}]),
        ),
        tool_step("done", {"summary": "The blocked producer was isolated from independent work."}),
    ]
    children = {
        "a": [tool_step("subframe_blocked", {"reason_code": "INTERFACE_WRONG",
                                           "detail": "The declared API is unavailable."})],
        "b": [
            tool_step("str_replace", {"path": "pkg/b.py", "old_str": "VALUE = 0", "new_str": "VALUE = 1"}),
            tool_step("done", {"summary": "The independent change is ready."}),
        ],
    }
    run_dir = tmp_path / "run"
    with BarrierEndpoint(parent, children) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Implement the declared producer and consumer work.", parallel="2")
        assert (run_dir / "summary.json").is_file(), completed.stdout + completed.stderr
        endpoint.assert_consumed()

    result = _call_result(run_dir)
    outcomes = {item.unit_id: item for item in result.outcomes}
    assert (outcomes["a"].status, outcomes["a"].reason) == ("BLOCKED", "INTERFACE_WRONG")
    assert outcomes["a"].detail == "The declared API is unavailable."
    assert (outcomes["c"].status, outcomes["c"].reason, outcomes["c"].detail) == (
        "CANCELLED", "PRODUCER_FAILED", "a")
    assert outcomes["b"].status == "COMMITTED_UNVERIFIED"
    assert result.txn_order == (1,)
    call_root, = parallel_call_roots(run_dir)
    assert not (call_root / "units/c/unit-spec.json").exists()
    children_dirs = _assert_all_subframe_wires(run_dir, {"a", "b"})
    assert "subframe_blocked" in tool_names(subframe_wire_files(children_dirs["a"])[0])
    assert {request["marker"] for request in endpoint.requests} == {"parent", "a", "b"}


def test_composition_check_rejects_b_only_after_a_commits_to_parent_state(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _simple_repo(tmp_path / "repo")
    (repo / "pkg/a.py").write_text("VALUE = 1\n")
    (repo / "pkg/b.py").write_text("OTHER = 0\n")
    (repo / "tests/test_a.py").write_text(
        "from pkg.a import VALUE\n"
        "def test_a_value():\n    assert VALUE == 2\n")
    (repo / "tests/test_b.py").write_text(
        "from pkg.a import VALUE\n"
        "def test_b_snapshot_value():\n    assert VALUE == 1\n")
    parent = [
        _parallel(
            _contract("a", "Change A's value to two.", owns=["pkg/a.py"],
                      checks=["tests/test_a.py::test_a_value"]),
            _contract("b", "Change B while checking its original A snapshot.", owns=["pkg/b.py"],
                      checks=["tests/test_b.py::test_b_snapshot_value"]),
        ),
        tool_step("done", {"summary": "B's snapshot check was re-evaluated after A committed."}),
    ]
    children = {
        "a": [
            tool_step("str_replace", {"path": "pkg/a.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"}),
        ],
        "b": [
            tool_step("str_replace", {"path": "pkg/b.py", "old_str": "OTHER = 0", "new_str": "OTHER = 1"}),
        ],
    }
    run_dir = tmp_path / "run"
    process = None
    with BarrierEndpoint(parent, children, hold_subframe_request={"b": 1}) as endpoint:
        try:
            process = start_cli(
                isolated_engine_python, endpoint.url, repo, run_dir,
                "Apply both changes, then check them against the current parent state.", parallel="2")
            assert endpoint.wait_for_requests("b", 1, timeout=180), "B did not reach its held edit request"
            deadline = time.monotonic() + 180
            call_root = None
            while time.monotonic() < deadline:
                roots = parallel_call_roots(run_dir)
                if roots:
                    call_root = roots[0]
                    log = call_root / "txn" / "log.jsonl"
                    if log.is_file() and any(record.unit_id == "a" and state == "PUBLISHED"
                                             for record, state in TxnStore(log.parent).records()):
                        break
                time.sleep(0.05)
            else:
                raise AssertionError("A did not publish before B's held check was released")
            endpoint.release_subframe("b")
            process.wait(timeout=900)
            assert (run_dir / "summary.json").is_file()
            endpoint.assert_consumed()
        finally:
            if process is not None and process.poll() is None:
                signal_run_owned(run_dir, process, signal.SIGKILL,
                                 deadline=time.monotonic() + 30,
                                 what="composition-check cleanup", group=True)
                process.wait(timeout=10)

    result = _call_result(run_dir)
    outcomes = {item.unit_id: item for item in result.outcomes}
    assert outcomes["a"].status == "COMMITTED"
    assert (outcomes["b"].status, outcomes["b"].reason) == (
        "REFUSED", "COMPOSITION_CHECK_FAILED")
    assert outcomes["b"].detail.startswith("FAIL")
    assert result.final_check == "FAIL"
    assert result.integration == "FAILED"
    call_root, = parallel_call_roots(run_dir)
    proposal = Proposal.from_json(read_json(call_root / "proposals/b/proposal.json"))
    worker_receipt = proposal.check_receipt
    assert worker_receipt is not None and worker_receipt.status == "PASS"
    composed = tmp_path / "parent-plus-b"
    shutil.copytree(run_dir / "workspace", composed)
    for entry in proposal.entries:
        blob = call_root / "proposals/b/blobs" / entry.after_sha256
        (composed / entry.path).write_bytes(blob.read_bytes())
    from stateframe.engine.kernel_effects import scan
    from stateframe.engine.kernel_unit_policy import _manifest_digest

    def ignored(path: str) -> bool:
        return any(part in {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
                            ".git", ".venv", "venv"} for part in Path(path).parts)

    composition_digest = _manifest_digest(scan(composed, exclude=ignored))
    assert worker_receipt.product_manifest_digest != composition_digest
    assert (run_dir / "workspace/pkg/a.py").read_text() == "VALUE = 2\n"
    assert (run_dir / "workspace/pkg/b.py").read_text() == "OTHER = 0\n"
    _assert_all_subframe_wires(run_dir, {"a", "b"})


def test_unknown_config_coupling_is_recorded_and_cannot_be_certified_by_empty_checks(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _simple_repo(tmp_path / "repo")
    (repo / "pkg/settings.ini").write_text("rate=1\n")
    (repo / "pkg/consumer.py").write_text(
        "from pathlib import Path\n"
        "def rate():\n"
        "    return Path('pkg/settings.ini').read_text()\n")
    parent = [
        _parallel(
            _contract("config", "Update the local rate setting.", owns=["pkg/settings.ini"]),
            _contract("consumer", "Update how the setting is loaded.", owns=["pkg/consumer.py"]),
        ),
        tool_step("done", {"summary": "The unknown config relationship is recorded."}),
    ]
    children = {
        "config": [
            tool_step("str_replace", {"path": "pkg/settings.ini", "old_str": "rate=1", "new_str": "rate=2"}),
            tool_step("done", {"summary": "The config edit is ready."}),
        ],
        "consumer": [
            tool_step("str_replace", {"path": "pkg/consumer.py", "old_str": "read_text()",
                                       "new_str": "read_text().strip()"}),
            tool_step("done", {"summary": "The consumer edit is ready."}),
        ],
    }
    run_dir = tmp_path / "run"
    with BarrierEndpoint(parent, children) as endpoint:
        completed = run_cli(isolated_engine_python, endpoint.url, repo, run_dir,
                            "Update these modules despite an unscanned config relationship.", parallel="2")
        endpoint.assert_consumed()
    assert (run_dir / "summary.json").is_file(), completed.stdout + completed.stderr

    result = _call_result(run_dir)
    pair, = result.unknown_coupling
    assert (pair.a, pair.b, pair.status, pair.reason) == (
        "config", "consumer", "UNKNOWN", "UNSUPPORTED_LANGUAGE")
    assert result.final_check == "NOT_RUN"
    assert result.integration == "UNVERIFIED"
    _assert_all_subframe_wires(run_dir, {"config", "consumer"})

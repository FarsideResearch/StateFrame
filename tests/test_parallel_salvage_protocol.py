"""A salvaged subframe candidate reaches the parent through the ordinary commit path.

A subframe that stops without submitting (for example HARNESS_CONTRACT after an
edit) leaves its newest candidate bundle at parallel/<call>/salvage/<unit>.json.
With STATEFRAME_PARALLEL_SALVAGE_ADOPT=on the parent turns that bundle into a
proposal against the subframe's start snapshot and offers it to the same
acceptance gates a submitted proposal passes. Off, nothing reads the bundle.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

import stateframe.engine.kernel_parallel as parallel
from stateframe.engine.kernel_parallel import ParallelError, execute_parallel
from stateframe.engine.kernel_parallel_records import CheckReceipt
from stateframe.engine.kernel_parallel_txn import TxnStore

from test_parallel_dispatch import ScriptedHost, restore_snapshot_permissions, setup_context, unit  # noqa: F401

SWITCH = "STATEFRAME_PARALLEL_SALVAGE_ADOPT"
CALL = "parallel_1"


class StoppedHost(ScriptedHost):
    """A subframe in `stopped` edits its workspace, then ends without a submission."""

    def __init__(self, changes, *, stopped, terminal="HARNESS_CONTRACT", drift=None, extra=None,
                 delete=None, submitted=(), corrupt_before=None):
        super().__init__(changes, terminal={name: terminal for name in stopped},
                         corrupt_before=corrupt_before)
        self.stopped = set(stopped) | set(submitted)
        self.drift = drift or {}
        self.extra = extra or {}
        self.delete = delete or {}
        self.root = None

    def poll(self, result_id):
        op, scope, _ = self._pending[result_id]
        if op == "status" and scope in self.stopped:
            self._capture(scope)
        return super().poll(result_id)

    def _capture(self, scope):
        launch = self.launched[scope]
        # The compact capture: declared files plus every file that differs from the start.
        files = {path: content.encode()
                 for path, content in {**self.changes[scope], **self.extra.get(scope, {})}.items()}
        for path in self.delete.get(scope, ()):
            del files[path]
        record = {"proposals": 3, "target": next(iter(self.changes[scope])),
                  "capture": "stateframe-compact/1", "deleted_files": sorted(self.delete.get(scope, ())),
                  "files_base64": {name: base64.b64encode(data).decode("ascii")
                                   for name, data in files.items()},
                  "file_hashes": {name: hashlib.sha256(data).hexdigest()
                                  for name, data in files.items()},
                  "patch": "--- a\n+++ b\n@@ edit\n"}
        folder = Path(launch["run_dir"]) / "worker" / "candidates"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "0003.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        for path, content in self.drift.items():
            (self.root / path).write_text(content)


def _run(tmp_path, monkeypatch, host, *, switch, units=None, files=None):
    if switch is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, switch)
    run, root, context = setup_context(tmp_path, monkeypatch, files=files)
    host.root = root
    result = execute_parallel(context, {"subframes": units or [unit("a", "a.py"), unit("b", "b.py")]},
                              host=host, store_root=run / "parallel" / CALL)
    return run, root, context, result


def _salvaged(context):
    return [fields for name, fields in context.runtime.journal.events if name == "parallel_unit_salvaged"]


def _statuses(result):
    return {item.unit_id: item.status for item in result.outcomes}


def test_stopped_subframe_edit_is_adopted_through_the_commit_path(tmp_path, monkeypatch):
    host = StoppedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}}, stopped={"a"})
    run, root, context, result = _run(tmp_path, monkeypatch, host, switch="on")

    assert _statuses(result) == {"a": "COMMITTED_UNVERIFIED", "b": "COMMITTED_UNVERIFIED"}
    assert root.joinpath("a.py").read_text() == "VALUE = 2\n"
    assert not [call for call in host.calls if call[0] == "seal" and call[1] == "a"]
    published = [(record.unit_id, state) for record, state in
                 TxnStore(run / "parallel" / CALL / "txn").records() if state == "PUBLISHED"]
    assert sorted(published) == [("a", "PUBLISHED"), ("b", "PUBLISHED")]
    adopted = next(item for item in result.outcomes if item.unit_id == "a")
    assert adopted.txn_id in result.txn_order
    assert [(delta.path, delta.added_lines, delta.removed_lines) for delta in adopted.files] == [("a.py", 1, 1)]
    assert result.integration != "FAILED"

    bundle = run / "parallel" / CALL / "salvage" / "a.json"
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    line = next(row for row in result.observation.splitlines() if row.startswith("Subframe a salvage:"))
    assert "stopped HARNESS_CONTRACT" in line and "adopted" in line and "not adopted" not in line
    assert f"parallel/{CALL}/salvage/a.json" in line and f"sha256={digest}" in line
    assert "files: a.py" in line
    event, = _salvaged(context)
    assert (event["adoption"], event["adoption_reason"], event["sha256"]) == ("ADOPTED", None, digest)
    assert event["adopted_files"] == ["a.py"]
    record = json.loads((run / "parallel" / CALL / "salvage" / "a.adoption.json").read_text())
    assert (record["adoption"], record["txn_id"], record["terminal"]) == (
        "ADOPTED", adopted.txn_id, "HARNESS_CONTRACT")
    saved = json.loads((run / "parallel" / CALL / "result.json").read_text())
    assert saved["observation"] == result.observation


def test_stale_base_salvage_is_refused_and_kept(tmp_path, monkeypatch):
    host = StoppedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}}, stopped={"a"},
                       drift={"a.py": "VALUE = 7\n"})
    run, root, context, result = _run(tmp_path, monkeypatch, host, switch="on")

    assert _statuses(result) == {"a": "REFUSED", "b": "COMMITTED_UNVERIFIED"}
    refused = next(item for item in result.outcomes if item.unit_id == "a")
    assert (refused.reason, refused.txn_id, refused.files) == ("HARNESS_CONTRACT", None, ())
    assert root.joinpath("a.py").read_text() == "VALUE = 7\n"
    assert result.integration == "FAILED"
    line = next(row for row in result.observation.splitlines() if row.startswith("Subframe a salvage:"))
    assert "not adopted (STALE_BASE: a.py)" in line and "files: a.py" in line
    assert f"parallel/{CALL}/salvage/a.json" in line
    event, = _salvaged(context)
    assert (event["adoption"], event["adoption_reason"]) == ("REFUSED", "STALE_BASE: a.py")
    assert (run / "parallel" / CALL / "salvage" / "a.json").is_file()
    assert {record.unit_id for record, _ in TxnStore(run / "parallel" / CALL / "txn").records()} == {"b"}


@pytest.mark.parametrize("check_status,adopted", [("FAIL", False), ("PASS", True)])
def test_salvage_adoption_runs_the_composition_check(tmp_path, monkeypatch, check_status, adopted):
    checked = []

    def frozen_check(spec, candidate, evidence_dir, *, test_python):
        checked.append((spec.unit_id, candidate.joinpath("a.py").read_text()))
        return CheckReceipt(spec.contract.check.digest(), "product", "harness", "environment",
                            spec.policy_digest, check_status,
                            "planted failing control" if check_status == "FAIL" else None,
                            (), str(evidence_dir), check_status)

    monkeypatch.setattr(parallel, "_check", frozen_check)
    with_check = unit("a", "a.py")
    with_check["check"] = {"tests": ["tests/test_a.py"]}
    host = StoppedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}}, stopped={"a"})
    run, root, context, result = _run(
        tmp_path, monkeypatch, host, switch="on", units=[with_check, unit("b", "b.py")],
        files={"a.py": "VALUE = 1\n", "b.py": "OTHER = 1\n", "tests/test_a.py": "assert True\n"})

    assert ("a", "VALUE = 2\n") in checked
    assert _statuses(result)["a"] == ("COMMITTED" if adopted else "REFUSED")
    assert root.joinpath("a.py").read_text() == ("VALUE = 2\n" if adopted else "VALUE = 1\n")
    event, = _salvaged(context)
    assert event["adoption"] == ("ADOPTED" if adopted else "REFUSED")
    if not adopted:
        assert event["adoption_reason"].startswith("COMPOSITION_CHECK_FAILED")


def test_salvage_outside_ownership_or_with_deletes_is_refused(tmp_path, monkeypatch):
    host = StoppedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}}, stopped={"a", "b"},
                       extra={"a": {"b.py": "OTHER = 9\n"}}, delete={"b": ("b.py",)})
    run, root, context, result = _run(tmp_path, monkeypatch, host, switch="on")

    assert _statuses(result) == {"a": "REFUSED", "b": "REFUSED"}
    assert root.joinpath("a.py").read_text() == "VALUE = 1\n"
    assert root.joinpath("b.py").read_text() == "OTHER = 1\n"
    reasons = {event["unit"]: event["adoption_reason"] for event in _salvaged(context)}
    assert reasons["a"].startswith("SCOPE_VIOLATION")
    assert reasons["b"].startswith("SALVAGE_UNSUPPORTED")
    assert not TxnStore(run / "parallel" / CALL / "txn").records()


def test_tampered_salvage_bundle_is_refused(tmp_path, monkeypatch):
    host = StoppedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}}, stopped={"a"})
    preserve = parallel.preserve_unit_candidate

    def tamper(call_root, unit_id, **options):
        receipt = preserve(call_root, unit_id, **options)
        path = call_root / "salvage" / f"{unit_id}.json"
        path.write_bytes(path.read_bytes() + b" ")
        return receipt

    monkeypatch.setattr(parallel, "preserve_unit_candidate", tamper)
    run, root, context, result = _run(tmp_path, monkeypatch, host, switch="on")

    assert _statuses(result)["a"] == "REFUSED"
    assert root.joinpath("a.py").read_text() == "VALUE = 1\n"
    event, = _salvaged(context)
    assert event["adoption_reason"].startswith("SALVAGE_INVALID")


def test_subframe_without_edits_has_nothing_to_adopt(tmp_path, monkeypatch):
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}},
                        terminal={"a": "HARNESS_CONTRACT"})
    run, root, context, result = _run(tmp_path, monkeypatch, host, switch="on")

    assert _statuses(result) == {"a": "REFUSED", "b": "COMMITTED_UNVERIFIED"}
    assert "salvage" not in result.observation
    assert not _salvaged(context)
    assert not (run / "parallel" / CALL / "salvage").exists()


@pytest.mark.parametrize("switch", [None, "off"])
def test_switch_off_keeps_the_salvage_unread(tmp_path, monkeypatch, switch):
    host = StoppedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}}, stopped={"a"})
    run, root, context, result = _run(tmp_path, monkeypatch, host, switch=switch)

    assert _statuses(result) == {"a": "REFUSED", "b": "COMMITTED_UNVERIFIED"}
    assert root.joinpath("a.py").read_text() == "VALUE = 1\n"
    bundle = run / "parallel" / CALL / "salvage" / "a.json"
    assert sorted(path.name for path in bundle.parent.iterdir()) == ["a.json"]
    event, = _salvaged(context)
    assert event == {"unit": "a", "terminal": "HARNESS_CONTRACT", "status": "PRESERVED",
                     "path": "salvage/a.json", "source": "0003.json",
                     "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                     "files": ["a.py"], "files_total": 1}
    assert result.observation.splitlines() == [
        "Subframe results:",
        "subframe_request_cap=2",
        "Plan a: owns=a.py; check=NOT_RUN; consumes=-",
        "Plan b: owns=b.py; check=NOT_RUN; consumes=-",
        "Subframe a: REFUSED; no file changes; contract=None; check=NOT_RUN; requests=0; "
        "reason=HARNESS_CONTRACT",
        "Subframe b: COMMITTED_UNVERIFIED; b.py (+1/-1); contract=True; check=NOT_RUN; requests=0",
        "Final check: NOT_RUN; integration: FAILED",
    ]


def test_switch_value_is_validated_before_any_launch(tmp_path, monkeypatch):
    host = StoppedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}}, stopped={"a"})
    monkeypatch.setenv(SWITCH, "maybe")
    run, root, context = setup_context(tmp_path, monkeypatch)
    with pytest.raises(ParallelError, match=SWITCH):
        execute_parallel(context, {"subframes": [unit("a", "a.py"), unit("b", "b.py")]},
                         host=host, store_root=run / "parallel" / CALL)
    assert not host.launched


def test_rejected_submission_is_salvaged(tmp_path, monkeypatch):
    host = StoppedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}}, stopped=(),
                       submitted={"a"}, corrupt_before="a")
    run, root, context, result = _run(tmp_path, monkeypatch, host, switch="on")

    refused = next(item for item in result.outcomes if item.unit_id == "a")
    assert (refused.status, refused.reason) == ("REFUSED", "STALE_BASE")
    assert root.joinpath("a.py").read_text() == "VALUE = 1\n"
    bundle = run / "parallel" / CALL / "salvage" / "a.json"
    event, = _salvaged(context)
    assert (event["unit"], event["terminal"], event["status"], event["files"]) == (
        "a", "SUBMITTED", "PRESERVED", ["a.py"])
    assert event["sha256"] == hashlib.sha256(bundle.read_bytes()).hexdigest()
    # It already failed the gates once; it is kept, never offered to them again.
    assert "adoption" not in event and "salvage" not in result.observation
    assert not (bundle.parent / "a.adoption.json").exists()


@pytest.mark.parametrize("switch", ["on", "off"])
def test_salvage_failure_never_aborts_the_call(tmp_path, monkeypatch, switch):
    host = StoppedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}}, stopped={"a"})
    monkeypatch.setenv(SWITCH, switch)
    run, root, context = setup_context(tmp_path, monkeypatch)
    host.root = root
    emit = context.runtime.journal.emit

    def failing(name, **fields):
        if name == "parallel_unit_salvaged":
            raise OSError("journal device is full")
        return emit(name, **fields)

    context.runtime.journal.emit = failing
    result = execute_parallel(context, {"subframes": [unit("a", "a.py"), unit("b", "b.py")]},
                              host=host, store_root=run / "parallel" / CALL)
    assert _statuses(result) == {"a": "COMMITTED_UNVERIFIED" if switch == "on" else "REFUSED",
                                 "b": "COMMITTED_UNVERIFIED"}
    assert (run / "parallel" / CALL / "result.json").is_file()

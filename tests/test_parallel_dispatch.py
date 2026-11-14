"""Parent subframe admission and dependency scheduling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import time

import pytest
import stateframe.engine.kernel_parallel as kernel_parallel

from stateframe.engine.kernel_live_projection import workspace_binding
from stateframe.engine.kernel_parallel import (ParallelContext, ParallelError, RecoveryRequired,
                                               execute_parallel, install)
from stateframe.engine.kernel_parallel_records import (HostAnswer, ParallelLimits, ParallelMeasured,
                                                       Proposal, ProposalEntry, Refusal, UnitSpec, UnitUsage)
from stateframe.engine.kernel_parallel_txn import TxnStore
from stateframe.engine.kernel_tool_tape import configure_replay
from stateframe.parallel_resources import SendLedger


@pytest.fixture(autouse=True)
def restore_snapshot_permissions(tmp_path):
    yield
    for manifest in tmp_path.rglob("snapshot-manifest.json"):
        snapshot = manifest.parent
        for directory in sorted((path for path in snapshot.rglob("*") if path.is_dir()), reverse=True):
            directory.chmod(0o755)
        snapshot.chmod(0o755)


class Journal:
    def __init__(self):
        self.events = []

    def emit(self, name, **fields):
        self.events.append((name, fields))


class Runtime:
    def __init__(self, root: Path, call_id: str):
        self.root = root
        self.target = root / "a.py"
        self.current_tool_call_id = call_id
        self.current_step = 1
        self.journal = Journal()

    def binding(self):
        return workspace_binding(self.root, "disabled")

    def _invalidate_binding_cache(self):
        pass


class Bolt:
    def __init__(self, root: Path):
        self.root = root
        self.product_edits = 0
        self.last_test_exit = 0
        self._last_green_revision = "old-green"
        self.receipts = []

    def receipt(self, *args, **kwargs):
        self.receipts.append((args, kwargs))


def unit(unit_id: str, path: str, *, exports=None, consumes=None):
    return {"id": unit_id, "outcome": f"Update {path}", "owns": [path], "creates": [],
            "exports": exports or [], "consumes": consumes or [], "reads": [],
            "requirements": [], "check": {"tests": []}}


def test_parent_tool_schema_and_parser_use_subframes_only():
    import stateframe.engine.kernel_parallel as parallel

    function = parallel.PARALLEL_TOOL["function"]
    parameters = function["parameters"]
    assert function["name"] == "parallel"
    assert parameters["required"] == ["subframes"]
    assert set(parameters["properties"]) == {"subframes"}

    contracts = [unit("a", "a.py"), unit("b", "b.py")]
    request = parallel._request({"subframes": contracts}, "call", "2")
    assert tuple(item.id for item in request.units) == ("a", "b")
    with pytest.raises(ParallelError, match="subframes array"):
        parallel._request({"units": contracts}, "call", "2")


def test_parent_refusal_observation_translates_internal_unit_reason_and_limit():
    import stateframe.engine.kernel_parallel as parallel
    from stateframe.engine.kernel_parallel_records import AdmissionResult

    refusal = Refusal(
        "TOO_MANY_UNITS",
        "max_subframes=2 (--parallel-max-subframes / STATEFRAME_PARALLEL_MAX_SUBFRAMES)",
        (), (), (),
    )
    admission = AdmissionResult("REFUSED", refusal, (), (), (), 0, "manifest", ParallelLimits())
    observation = parallel._observation(
        admission, (), "NOT_RUN", "NOT_RUN", ParallelMeasured(0.0, 0.0, 0.0, 0, 0.0))
    assert observation == (
        "REFUSED TOO_MANY_SUBFRAMES: max_subframes=2 "
        "(--parallel-max-subframes / STATEFRAME_PARALLEL_MAX_SUBFRAMES)")
    assert "unit" not in observation.casefold()


def test_peak_uses_durable_intervals_not_sequential_status_poll_counts():
    import stateframe.engine.kernel_parallel as parallel

    # One status tick can see the first and third RUNNING despite their slots
    # being reused between sequential polls. Their intervals never overlap.
    intervals = [(100.0, 212.07), (150.0, 260.0), (212.37, 300.0)]
    assert parallel._peak_interval_overlap(intervals) == 2
    assert parallel._peak_interval_overlap([(1.0, 2.0), (2.0, 3.0)]) == 1
    with pytest.raises(ParallelError, match="timing interval"):
        parallel._peak_interval_overlap([(3.0, 2.0)])


def test_peak_recovers_unrecorded_interval_from_host_state(tmp_path):
    import stateframe.engine.kernel_parallel as parallel

    root = tmp_path / "parallel" / "call_0"
    intervals = root / "intervals"
    intervals.mkdir(parents=True)
    (intervals / "a-1.json").write_text(json.dumps({
        "running_since": 100.0, "finished_epoch": 212.07}))
    host = root / "host" / "units" / "b"
    host.mkdir(parents=True)
    (host / "1.json").write_text(json.dumps({
        "started_epoch": 150.0, "finished_epoch": 260.0}))
    third = root / "host" / "units" / "c"
    third.mkdir()
    (third / "1.json").write_text(json.dumps({
        "started_epoch": 212.37, "finished_epoch": 300.0}))
    assert parallel._peak_active_subframes(root) == 2


def test_blocked_terminal_retains_worker_reason_and_bounded_detail(tmp_path):
    import stateframe.engine.kernel_parallel as parallel

    call_root = tmp_path / "parallel" / "call_0"
    result_path = call_root / "units" / "a" / "engine-result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({"result": {
        "result": "SUBFRAME_BLOCKED", "reason_code": "INTERFACE_WRONG",
        "detail": "x" * 300}}))
    usage = UnitUsage(1, 1, 5, 3, 100)
    outcome = parallel._blocked_outcome(call_root, "a", usage, 1.5, 0.25)
    assert (outcome.status, outcome.reason, outcome.detail, outcome.check) == (
        "BLOCKED", "INTERFACE_WRONG", "x" * 300, "NOT_RUN")
    result_path.write_text(json.dumps({"result": {
        "result": "SUBFRAME_BLOCKED", "reason_code": "INTERFACE_WRONG",
        "detail": "x" * 301}}))
    with pytest.raises(ParallelError, match="invalid reason or detail"):
        parallel._blocked_outcome(call_root, "a", usage, 1.5, 0.25)


def setup_context(tmp_path: Path, monkeypatch, *, call_id="parallel_1", files=None):
    root = tmp_path / "workspace"
    root.mkdir()
    for path, content in (files or {"a.py": "VALUE = 1\n", "b.py": "OTHER = 1\n"}).items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parent").mkdir()
    limits = ParallelLimits(unit_max_requests=2, parent_reserve_requests=1)
    spec = {"parallel_mode": "2", "parallel_limits": limits.to_json(),
            "run_identity_seed": "parent-seed", "max_turns": 50, "test_cmd": None,
            "parent_snapshot": str(run / "parent")}
    import stateframe.engine.kernel_parallel as parallel
    monkeypatch.setattr(parallel, "_service", None)
    install(run, spec)
    configure_replay(run, active=False)
    runtime = Runtime(root, call_id)
    bolt = Bolt(root)
    context = ParallelContext(runtime, bolt, root, set(), "parent-seed", 1, (), limits,
                              "2", lambda _path: False)
    return run, root, context


class ScriptedHost:
    """Only the host channel is simulated; admission, projection and publication are real."""

    def __init__(self, changes: dict[str, dict[str, str]], *, corrupt_before: str | None = None,
                 terminal: dict[str, str] | None = None, exit_order: tuple[str, ...] = (),
                 omit_exit_sequence: bool = False, run_first: bool = False):
        self.changes = changes
        self.corrupt_before = corrupt_before
        self.terminal = terminal or {}
        self.exit_order = exit_order
        self.omit_exit_sequence = omit_exit_sequence
        self.run_first = run_first
        self.status_count = {}
        self.launched = {}
        self.launch_times = {}
        self.last_finished = 0.0
        self.calls = []
        self._pending = {}

    def submit(self, op, scope, payload):
        self.calls.append((op, scope, payload))
        ref = str(len(self.calls))
        self._pending[ref] = (op, scope, payload)
        if op == "launch":
            self.launched[scope] = payload
            self.launch_times[scope] = max(time.time(), self.last_finished)
        return ref

    def poll(self, result_id):
        op, scope, payload = self._pending[result_id]
        call_id = next(iter(self.launched.values()))["spec_path"].split("/units/")[0].split("/")[-1] \
            if self.launched else "unused"
        if op == "launch":
            answer = {"state": "STARTING"}
        elif op == "status":
            self.status_count[scope] = self.status_count.get(scope, 0) + 1
            order = self.exit_order or tuple(self.launched)
            state = "RUNNING" if self.run_first and self.status_count[scope] == 1 else "EXITED"
            started = self.launch_times[scope]
            if state == "EXITED":
                self.last_finished = max(self.last_finished, started + 1.5)
            answer = {"state": state, "terminal": self.terminal.get(scope, "SUBMITTED"),
                      "exit_code": 0, "running_since": started,
                      "finished_epoch": started + 1.5 if state == "EXITED" else None,
                      "slot": tuple(self.launched).index(scope) + 2,
                      "slot_wait_seconds": 0.25}
            if not self.omit_exit_sequence:
                answer["exited_seq"] = order.index(scope) + 1
        elif op == "seal":
            spec_path = Path(self.launched[scope]["spec_path"])
            spec = UnitSpec.from_json(json.loads(spec_path.read_text()))
            proposal_dir = spec_path.parents[2] / "proposals" / scope
            blobs = proposal_dir / "blobs"
            blobs.mkdir(parents=True, exist_ok=True)
            entries = []
            for path, content in self.changes[scope].items():
                snapshot_root = Path(self.launched[scope]["start_snapshot_path"])
                before_path = snapshot_root / path
                snapshot_manifest = json.loads((snapshot_root / "snapshot-manifest.json").read_text())
                original_mode = next(row["mode"] for row in snapshot_manifest["files"]
                                     if row["path"] == path)
                before = hashlib.sha256(before_path.read_bytes()).hexdigest()
                if scope == self.corrupt_before:
                    before = "0" * 64
                after = content.encode()
                digest = hashlib.sha256(after).hexdigest()
                (blobs / digest).write_bytes(after)
                entries.append(ProposalEntry(path, "modify", before, digest, original_mode))
            proposal = Proposal(spec.call_id, scope, spec.execution_id, payload["incarnation"],
                                tuple(entries), spec.contract.digest(), None)
            path = proposal_dir / "proposal.json"
            path.write_text(json.dumps(proposal.to_json()))
            answer = {"proposal_path": str(path), "proposal_digest": proposal.digest()}
        else:
            answer = {"state": "CANCEL_REQUESTED"}
        return HostAnswer(1, call_id, int(result_id), op, "OK", None, answer)

    def cancel(self, scope):
        self.calls.append(("cancel", scope, {}))


def test_two_parent_calls_share_cumulative_subframe_send_allowance(tmp_path, monkeypatch):
    import stateframe.engine.kernel_parallel as parallel

    run, root, context = setup_context(tmp_path, monkeypatch)
    context.limits = ParallelLimits(unit_max_requests=4, parent_reserve_requests=2)
    parent_spec = dict(parallel._service[1], max_turns=13,
                       parallel_limits=context.limits.to_json())
    monkeypatch.setattr(parallel, "_service", (run, parent_spec))

    class SendingHost(ScriptedHost):
        def poll(self, result_id):
            op, scope, _ = self._pending[result_id]
            answer = super().poll(result_id)
            if op == "status" and answer.payload["state"] == "EXITED":
                directory = Path(self.launched[scope]["run_dir"])
                ledger = SendLedger(directory / "sends-ledger.jsonl")
                for index in range(4):
                    send_id = f"{scope}:{index}"
                    assert ledger.authorize_with_cap(send_id, 4)
                    ledger.settle(send_id, {})
            return answer

    first_host = SendingHost({"a": {"a.py": "VALUE = 2\n"},
                              "b": {"b.py": "OTHER = 2\n"}})
    contracts = {"subframes": [unit("a", "a.py"), unit("b", "b.py")]}
    first = execute_parallel(context, contracts, host=first_host,
                             store_root=run / "parallel" / "parallel_1")
    assert first.admission.status == "ADMITTED"
    assert [item.usage.physical_sends for item in first.outcomes] == [4, 4]
    budget_path = run / "parallel/budget.json"
    after_first = json.loads(budget_path.read_bytes())
    assert (after_first["parent_steps_spent"], after_first["subframe_sends_authorized"],
            after_first["held_reservations"], after_first["free"], after_first["parent_reserve"]) == (
                1, 8, 0, 2, 2)
    assert root.joinpath("a.py").read_text() == "VALUE = 2\n"

    context.runtime.current_tool_call_id = "parallel_2"
    context.runtime.current_step = 2
    second_host = ScriptedHost({})
    second = execute_parallel(context, contracts, host=second_host,
                              store_root=run / "parallel" / "parallel_2")
    assert (second.admission.status, second.admission.refusal.reason) == (
        "REFUSED", "RESOURCES_UNAVAILABLE")
    assert second_host.calls == []
    after_second = json.loads(budget_path.read_bytes())
    assert (after_second["parent_steps_spent"], after_second["subframe_sends_authorized"],
            after_second["held_reservations"], after_second["free"], after_second["parent_reserve"]) == (
                2, 8, 0, 1, 2)
    assert sum(after_second[key] for key in ("parent_steps_spent", "subframe_sends_authorized",
                                              "held_reservations", "free", "parent_reserve")) == 13


def test_parent_allowance_below_completion_reserve_refuses_without_launch(tmp_path, monkeypatch):
    import stateframe.engine.kernel_parallel as parallel

    run, _, context = setup_context(tmp_path, monkeypatch)
    context.limits = ParallelLimits(unit_max_requests=2, parent_reserve_requests=2)
    parent_spec = dict(parallel._service[1], max_turns=2,
                       parallel_limits=context.limits.to_json())
    monkeypatch.setattr(parallel, "_service", (run, parent_spec))
    host = ScriptedHost({})
    result = execute_parallel(context,
                              {"subframes": [unit("a", "a.py"), unit("b", "b.py")]},
                              host=host, store_root=run / "parallel" / "parallel_1")
    assert (result.admission.status, result.admission.refusal.reason) == (
        "REFUSED", "RESOURCES_UNAVAILABLE")
    assert host.calls == []


@pytest.mark.host_sandbox
def test_tiny_checked_plan_is_advisory_in_journal_and_observation(tmp_path, monkeypatch):
    from stateframe.engine.kernel_parallel import TINY_PLAN_ADVISORY

    run, root, context = setup_context(tmp_path, monkeypatch)
    (root / "tests").mkdir()
    for name in ("a", "b"):
        (root / "tests" / f"test_{name}.py").write_text(
            f"def test_{name}():\n    assert True\n")
    units = []
    for name in ("a", "b"):
        item = unit(name, f"{name}.py")
        item["check"] = {"tests": [f"tests/test_{name}.py"]}
        units.append(item)
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"},
                         "b": {"b.py": "OTHER = 2\n"}})
    result = execute_parallel(context, {"subframes": units}, host=host,
                              store_root=run / "parallel" / "parallel_1")
    assert result.admission.status == "ADMITTED"
    assert TINY_PLAN_ADVISORY in result.observation
    assert ("parallel_plan_completed", {"action": "tiny_plan_advisory",
            "reason_code": "TINY_PLAN", "detail": TINY_PLAN_ADVISORY}) in context.runtime.journal.events


def test_tiny_plan_classification_has_large_and_no_check_controls(tmp_path, monkeypatch):
    from stateframe.engine.kernel_parallel import _request, _tiny_plan
    from stateframe.engine.kernel_parallel_admission import DirectorySnapshot

    _, root, _ = setup_context(tmp_path, monkeypatch)
    (root / "tests").mkdir()
    for name in ("a", "b"):
        (root / "tests" / f"test_{name}.py").write_text(
            f"def test_{name}():\n    assert True\n")
    units = []
    for name in ("a", "b"):
        item = unit(name, f"{name}.py")
        item["check"] = {"tests": [f"tests/test_{name}.py"]}
        units.append(item)
    request = _request({"subframes": units}, "parallel_1", "2")
    assert _tiny_plan(request, DirectorySnapshot(root, 0))
    (root / "a.py").write_text("#" + "x" * 4100)
    assert not _tiny_plan(request, DirectorySnapshot(root, 0))
    (root / "a.py").write_text("VALUE = 1\n")
    units[0]["check"] = {"tests": []}
    assert not _tiny_plan(_request({"subframes": units}, "parallel_1", "2"),
                          DirectorySnapshot(root, 0))


def test_overlap_refuses_before_any_launch(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    host = ScriptedHost({})
    result = execute_parallel(context,
        {"subframes": [unit("a", "a.py"), unit("b", "a.py")]},
                              host=host, store_root=run / "parallel" / "parallel_1")
    assert result.admission.status == "REFUSED"
    assert result.admission.refusal.reason == "OWNERSHIP_OVERLAP"
    assert set(result.admission.refusal.paths) == {"a.py"}
    assert host.calls == []
    assert root.joinpath("a.py").read_text() == "VALUE = 1\n"
    assert "REFUSED OWNERSHIP_OVERLAP" in result.observation


def test_parent_tool_surface_is_conditional_and_subframe_named(tmp_path, monkeypatch):
    import stateframe.engine.kernel_parallel as parallel
    from probes import react_repair as repair

    parallel._service = None
    baseline = repair._tools_for_probe_modes(probe_gate_on=False, observation_probe_on=False)
    assert "parallel" not in [item["function"]["name"] for item in baseline]
    run, _, _ = setup_context(tmp_path, monkeypatch)
    tools = repair._tools_for_probe_modes(probe_gate_on=False, observation_probe_on=False)
    offered = [item for item in tools if item["function"]["name"] == "parallel"]
    assert len(offered) == 1
    assert "subframe" in offered[0]["function"]["description"]
    assert run.joinpath("parent").is_dir()
    baseline_receipt, = [item for item in baseline if item["function"]["name"] == "read_check_receipt"]
    receipt, = [item for item in tools if item["function"]["name"] == "read_check_receipt"]
    pinned = (Path(__file__).parent / "fixtures/read_check_receipt_off.json").read_bytes()
    assert json.dumps(baseline_receipt, separators=(",", ":"), ensure_ascii=False).encode() + b"\n" == pinned
    assert "next_handle" not in baseline_receipt["function"]["description"]
    assert "subframe" in receipt["function"]["description"]
    assert "next_handle" in receipt["function"]["description"]
    assert [item for item in tools if item["function"]["name"] not in
            {"parallel", "read_check_receipt"}] == [
                item for item in baseline if item["function"]["name"] != "read_check_receipt"]


def test_parallel_closure_discards_queued_native_sibling():
    from kernel_native_tools import NativeToolBatch

    batch = NativeToolBatch()
    batch.journal = Journal()
    batch.origin = "request-1"
    batch.index = 1
    batch.size = 2
    batch.inflight = {"id": "parallel-call", "function": {"name": "parallel", "arguments": "{}"}}
    batch.pending = [{"id": "later-edit", "function": {"name": "str_replace", "arguments": "{}"}}]
    batch.action_closed({"step_record_id": "step-1", "result": "APPLIED"})
    assert batch.pending == []
    assert ("parallel_batch_discarded", {"count": 1}) in batch.journal.events
    discarded = next(fields for name, fields in batch.journal.events if name == "native_batch_discarded")
    assert discarded["reason"] == "parallel_batch_discarded"
    assert [item["id"] for item in discarded["calls"]] == ["later-edit"]


def test_dependency_starts_after_producer_commit_and_projects_committed_digest(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch,
                                       files={"a.py": "def source(x):\n    return x\n",
                                              "b.py": "OTHER = 1\n", "c.py": "def use():\n    return 1\n"})
    from stateframe.engine.kernel_parallel_contracts import canonical_signature
    signature = canonical_signature("def source(x):\n    return x\n", "source")
    host = ScriptedHost({"a": {"a.py": "def source(x):\n    return x + 1\n"},
                         "b": {"b.py": "OTHER = 2\n"}, "c": {"c.py": "def use():\n    return 2\n"}})
    result = execute_parallel(context, {"subframes": [
        unit("a", "a.py", exports=[{"id": "src", "path": "a.py", "symbol": "source",
                                      "signature": signature.to_json()}]),
        unit("b", "b.py"), unit("c", "c.py", consumes=[{"export_id": "src"}])]},
        host=host, store_root=run / "parallel" / "parallel_1")
    assert result.txn_order == (1, 2, 3), result.observation
    assert result.measured.peak_active_workers == 2
    spec_c = UnitSpec.from_json(json.loads(Path(host.launched["c"]["spec_path"]).read_text()))
    txn_a = next(item for item, _ in TxnStore(
        run / "parallel" / "parallel_1" / "txn").records() if item.unit_id == "a")
    assert spec_c.start_generation >= txn_a.post_generation
    assert spec_c.interfaces[0].source_digest == hashlib.sha256(root.joinpath("a.py").read_bytes()).hexdigest()


def test_dependent_has_no_launch_before_producer_is_durably_published(tmp_path, monkeypatch):
    from stateframe.engine.kernel_parallel_contracts import canonical_signature

    source = "def source(x):\n    return x\n"
    run, _root, context = setup_context(
        tmp_path, monkeypatch, files={"a.py": source, "c.py": "def use():\n    return 1\n"})
    signature = canonical_signature(source, "source")
    host = ScriptedHost({"a": {"a.py": "def source(x):\n    return x + 1\n"},
                         "c": {"c.py": "def use():\n    return 2\n"}})

    def refuse_publication(_store, _txn_id, _publication_id):
        raise OSError("planted publication failure")

    monkeypatch.setattr(TxnStore, "mark_published", refuse_publication)
    with pytest.raises(RecoveryRequired, match="RECOVERY_REQUIRED.*planted publication failure"):
        execute_parallel(context, {"subframes": [
            unit("a", "a.py", exports=[{"id": "src", "path": "a.py", "symbol": "source",
                                         "signature": signature.to_json()}]),
            unit("c", "c.py", consumes=[{"export_id": "src"}])]},
            host=host, store_root=run / "parallel" / "parallel_1")
    assert "a" in host.launched
    assert "c" not in host.launched
    dispositions = [state for record, state in TxnStore(run / "parallel" / "parallel_1" / "txn").records()
                    if record.unit_id == "a"]
    assert dispositions and dispositions[-1] == "MATERIALIZED"


def test_bad_before_digest_cancels_dependent_and_keeps_independent(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch,
                                       files={"a.py": "def source(x):\n    return x\n",
                                              "b.py": "OTHER = 1\n", "c.py": "def use():\n    return 1\n"})
    from stateframe.engine.kernel_parallel_contracts import canonical_signature
    signature = canonical_signature("def source(x):\n    return x\n", "source")
    host = ScriptedHost({"a": {"a.py": "def source(x):\n    return x + 1\n"},
                         "b": {"b.py": "OTHER = 2\n"}}, corrupt_before="a")
    result = execute_parallel(context, {"subframes": [
        unit("a", "a.py", exports=[{"id": "src", "path": "a.py", "symbol": "source",
                                      "signature": signature.to_json()}]),
        unit("b", "b.py"), unit("c", "c.py", consumes=[{"export_id": "src"}])]},
        host=host, store_root=run / "parallel" / "parallel_1")
    statuses = {item.unit_id: item for item in result.outcomes}
    assert statuses["a"].reason == "STALE_BASE"
    assert statuses["c"].status == "CANCELLED"
    assert statuses["c"].reason == "PRODUCER_FAILED"
    assert statuses["c"].detail == "a"
    assert statuses["b"].status == "COMMITTED_UNVERIFIED", result.observation
    assert "c" not in host.launched
    assert root.joinpath("a.py").read_text() == "def source(x):\n    return x\n"
    assert root.joinpath("b.py").read_text() == "OTHER = 2\n"


def test_context_carries_parent_requirement_clauses_and_structured_refs(tmp_path, monkeypatch):
    run, root, setup = setup_context(tmp_path, monkeypatch)
    runtime = setup.runtime
    runtime.requirements_ledger = {
        "requirements": [{"id": "R1", "exact_clause": "Keep a.py compatible",
                          "contract": {"subject_binding_ids": ["B1"]}},
                         {"id": "R2", "exact_clause": "Preserve b.py"}],
        "bindings": [{"id": "B1", "locator": {"path": "a.py", "symbol": "VALUE"}}],
    }
    from stateframe.engine.kernel_parallel import context as make_context

    actual = make_context(runtime, setup.bolt, root, setup.created, protected=setup.protected)
    assert [(item.id, item.text, item.refs) for item in actual.requirements] == [
        ("R1", "Keep a.py compatible", ("a.py::VALUE",)),
        ("R2", "Preserve b.py", ()),
    ]
    wanted = unit("a", "a.py")
    wanted["requirements"] = ["R1"]
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}})
    result = execute_parallel(actual, {"subframes": [wanted, unit("b", "b.py")]},
                              host=host, store_root=run / "parallel" / "parallel_1")
    assert result.admission.status == "ADMITTED"
    spec = UnitSpec.from_json(json.loads(Path(host.launched["a"]["spec_path"]).read_text()))
    assert [(item.id, item.text, item.refs) for item in spec.requirement_clauses] == [
        ("R1", "Keep a.py compatible", ("a.py::VALUE",))]
    intent = json.loads((run / "parallel" / "parallel_1" / "intent.json").read_text())
    assert intent["requirements_digest"] == hashlib.sha256(
        json.dumps([item.to_json() for item in actual.requirements],
                   sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_exit_sequence_controls_transaction_order_and_host_metrics(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}},
                        exit_order=("b", "a"))
    result = execute_parallel(context, {"subframes": [unit("a", "a.py"), unit("b", "b.py")]},
                              host=host, store_root=run / "parallel" / "parallel_1")
    assert [(record.txn_id, record.unit_id) for record, _ in TxnStore(
        run / "parallel" / "parallel_1" / "txn").records() if _ == "PUBLISHED"] == [
            (1, "b"), (2, "a")]
    assert result.measured.queue_wait_seconds == 0.5
    assert {item.unit_id: item.wall_seconds for item in result.outcomes} == {"a": 1.5, "b": 1.5}
    assert result.measured.estimated_serial_seconds == 3.0
    assert result.measured.parallel_wall_seconds > 0
    assert result.measured.speedup == round(3.0 / result.measured.parallel_wall_seconds, 3)
    assert [(fields["unit"], fields["slot"]) for kind, fields in context.runtime.journal.events
            if kind == "parallel_unit_started"] == [("a", 2), ("b", 3)]
    recorded = json.loads((run / "parallel" / "parallel_1" / "result.json").read_text())
    assert recorded["measured"]["estimated_serial_seconds"] == 3.0
    assert {item["unit_id"]: item["wall_seconds"] for item in recorded["outcomes"]} == {
        "a": 1.5, "b": 1.5}
    snapshot_manifest = json.loads((Path(host.launched["a"]["start_snapshot_path"]) /
                                    "snapshot-manifest.json").read_text())
    assert host.launched["a"]["snapshot_manifest_digest"] == hashlib.sha256(
        json.dumps(snapshot_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_host_finish_before_lease_start_cannot_create_vacuous_wall(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)

    class InvalidClockHost(ScriptedHost):
        def poll(self, result_id):
            answer = super().poll(result_id)
            if answer.op == "status" and answer.payload["state"] == "EXITED":
                answer.payload["finished_epoch"] = answer.payload["running_since"] - 1
            return answer

    host = InvalidClockHost({"a": {"a.py": "VALUE = 2\n"},
                             "b": {"b.py": "OTHER = 2\n"}})
    with pytest.raises(ParallelError, match="finish time"):
        execute_parallel(context, {"subframes": [unit("a", "a.py"), unit("b", "b.py")]},
                         host=host, store_root=run / "parallel" / "parallel_1")
    assert root.joinpath("a.py").read_text() == "VALUE = 1\n"
    assert TxnStore(run / "parallel" / "parallel_1" / "txn").records() == []


def test_peak_counts_host_running_workers_not_queued_launches(tmp_path, monkeypatch):
    run, _, context = setup_context(tmp_path, monkeypatch)
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}},
                        run_first=True)
    result = execute_parallel(context, {"subframes": [unit("a", "a.py"), unit("b", "b.py")]},
                              host=host, store_root=run / "parallel" / "parallel_1")
    assert result.measured.peak_active_workers == 2
    assert result.measured.queue_wait_seconds == 0.5


def test_missing_host_exit_sequence_refuses_before_seal(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}},
                        omit_exit_sequence=True)
    with pytest.raises(ParallelError, match="exited_seq"):
        execute_parallel(context, {"subframes": [unit("a", "a.py"), unit("b", "b.py")]},
                         host=host, store_root=run / "parallel" / "parallel_1")
    assert not any(op == "seal" for op, _, _ in host.calls)
    assert root.joinpath("a.py").read_text() == "VALUE = 1\n"


def test_start_snapshot_is_read_only_with_authenticated_original_modes(tmp_path):
    from stateframe.engine.kernel_parallel import _snapshot

    root = tmp_path / "workspace"
    (root / "pkg").mkdir(parents=True)
    source = root / "pkg" / "a.py"
    source.write_text("VALUE = 1\n")
    source.chmod(0o644)
    snapshot = _snapshot(root, tmp_path / "artifacts", 0)
    copied = snapshot.root / "pkg" / "a.py"
    manifest = json.loads((snapshot.root / "snapshot-manifest.json").read_text())
    assert stat.S_IMODE(copied.stat().st_mode) == 0o444
    assert stat.S_IMODE((snapshot.root / "pkg").stat().st_mode) == 0o555
    assert manifest["files"][0]["path"] == "pkg/a.py"
    assert manifest["files"][0]["mode"] == 0o644
    assert snapshot.manifest_digest == manifest["source_manifest_digest"]
    assert snapshot.list_files() == ["pkg/a.py"]


def test_parent_steps_read_the_bolt_stamp_when_the_runtime_lacks_it():
    from types import SimpleNamespace
    context = SimpleNamespace(bolt=SimpleNamespace(current_step=2), runtime=SimpleNamespace())
    assert kernel_parallel._parent_steps(context) == 3  # 0-based loop index + 1
    context = SimpleNamespace(bolt=SimpleNamespace(), runtime=SimpleNamespace(current_step=2))
    assert kernel_parallel._parent_steps(context) == 2
    context = SimpleNamespace(bolt=SimpleNamespace(), runtime=SimpleNamespace(owner=SimpleNamespace(current_step=4)))
    assert kernel_parallel._parent_steps(context) == 4
    assert kernel_parallel._parent_steps(SimpleNamespace(bolt=object(), runtime=object())) == 0

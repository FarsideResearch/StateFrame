import hashlib
import json

import pytest

from stateframe import profiles
from stateframe.cli import _run_options, build_parser
from stateframe.endpoint import Endpoint
from stateframe.engine.kernel_parallel import _snapshot
from stateframe.engine.kernel_parallel_admission import DirectorySnapshot
from stateframe.engine.kernel_parallel_records import (
    PARALLEL_LIMIT_OPTIONS,
    AdmissionResult,
    CheckRecipe,
    ExecutionState,
    ParallelLimits,
    ParallelMeasured,
    ParallelRequest,
    ParallelResult,
    Refusal,
    UnitContract,
    UnitOutcome,
    UnitSpec,
    UnitUsage,
    canonical_json,
)
from stateframe.engine.kernel_tool_tape import ToolTape
from stateframe.resume import ResumeRefused, archive_attempt, assess
from stateframe.runner import _parallel_summary


def _run_spec(run):
    parent = run / "parent"
    parent.mkdir(parents=True)
    source = profiles.engine_source_summary()
    return {
        "schema": "stateframe-run/1",
        "environment": {},
        "public_row": {},
        "parent_snapshot": str(parent),
        "chat_url": "http://127.0.0.1:8080/v1/chat/completions",
        "model": "local-model",
        "max_turns": 3,
        "repo": str(run),
        "profile": "default",
        "engine_source": source,
    }


def _pending(run, tool, args):
    ToolTape(run).intent(call_id="pending-call", tool=tool, args=args, replay_class="local",
                         capability_digest="cap", definition_digest=None,
                         env_identity="env", pre_manifest_digest="before")


def _pending_input_refusal(run, intent):
    ToolTape(run).intent(call_id="pending-call", tool="parallel", args=intent, replay_class="local",
                         capability_digest=hashlib.sha256(b"stateframe-subframe-input/1").hexdigest(),
                         definition_digest=hashlib.sha256(canonical_json(intent.get("arguments"))).hexdigest(),
                         env_identity="env", pre_manifest_digest="before")


def _recovery_intent():
    contract_a = UnitContract("a", "first", ("a.py",), (), (), (), (), (), CheckRecipe((), (), "none"))
    contract_b = UnitContract("b", "second", ("b.py",), (), (), (), (), (), CheckRecipe((), (), "none"))
    request = ParallelRequest("call", (contract_a, contract_b), "auto")
    admission = AdmissionResult("ADMITTED", None, (), (), ("a", "b"), 0, "a" * 64, ParallelLimits())
    return {"schema": "stateframe-parallel-recovery/1", "call_id": "call",
            "request": request.to_json(), "admission": admission.to_json(),
            "reservations": {"a": 12, "b": 12}, "generation": 0, "mode": "auto",
            "policy_digest": "b" * 64, "executions": []}


def _live_pending(run):
    """Build the tape and mutable durable intent written by a real parallel call."""
    work = run.parent / "source"
    work.mkdir()
    (work / "a.py").write_text("VALUE = 1\n")
    (work / "b.py").write_text("OTHER = 1\n")
    run.mkdir()
    (run / "spec.json").write_text(json.dumps(_run_spec(run)))
    call = run / "parallel" / "call"
    snapshot = _snapshot(work, call, 0)
    contract_a = UnitContract("a", "first", ("a.py",), (), (), (), (), (), CheckRecipe((), (), "none"))
    contract_b = UnitContract("b", "second", ("b.py",), (), (), (), (), (), CheckRecipe((), (), "none"))
    request = ParallelRequest("call", (contract_a, contract_b), "auto")
    limits = ParallelLimits()
    admission = AdmissionResult("ADMITTED", None, (), (), ("a", "b"), 0,
                                snapshot.manifest_digest, limits)
    intent = {
        "schema": "stateframe-parallel-intent/1", "call_id": "call",
        "request": request.to_json(), "admission": admission.to_json(),
        "reservations": {"a": 12, "b": 12}, "generation": 0, "mode": "auto",
        "policy_digest": "b" * 64, "executions": [],
        "meter_calibration": {}, "subframe_cap_calibration": {},
        "requirements_digest": "a" * 64,
        "snapshot_manifests": {"g0": snapshot.snapshot_manifest_digest},
    }
    ToolTape(run).intent(call_id="call", tool="parallel", args=intent, replay_class="local",
                         capability_digest=intent["policy_digest"],
                         definition_digest=request.digest(), env_identity="env",
                         pre_manifest_digest=snapshot.manifest_digest)
    worker = call / "units" / "a"
    worker.mkdir(parents=True)
    spec = UnitSpec("call", "a", "exec-a", "seed-a", contract_a, (), (), ("a.py",), (), "",
                    0, 0, snapshot.manifest_digest, intent["policy_digest"], limits)
    (worker / "unit-spec.json").write_bytes(canonical_json(spec.to_json()) + b"\n")
    durable = {**intent, "executions": [ExecutionState(
        "a", "exec-a", str(worker), spec.digest(), 1, None).to_json()]}
    (call / "intent.json").write_bytes(canonical_json(durable) + b"\n")
    return call, durable


def test_resume_assess_accepts_live_parallel_intent_and_durable_execution(tmp_path):
    run = tmp_path / "run"
    call, durable = _live_pending(run)
    tape_before = (run / "tool-tape.jsonl").read_bytes()
    intent_before = (call / "intent.json").read_bytes()

    state = assess(run)

    descriptor = state["parallel_recovery"]
    assert descriptor["schema"] == "stateframe-parallel-recovery/1"
    assert descriptor["intent_digest"] == hashlib.sha256(canonical_json(durable)).hexdigest()
    assert [item["unit_id"] for item in descriptor["executions"]] == ["a"]
    assert (run / "tool-tape.jsonl").read_bytes() == tape_before
    assert (call / "intent.json").read_bytes() == intent_before


@pytest.mark.parametrize("change", ["immutable_field", "snapshot_digest"])
def test_resume_assess_rejects_tampered_live_parallel_intent(tmp_path, change):
    run = tmp_path / "run"
    call, durable = _live_pending(run)
    if change == "immutable_field":
        durable["policy_digest"] = "c" * 64
    else:
        durable["snapshot_manifests"]["g0"] = "c" * 64
    (call / "intent.json").write_bytes(canonical_json(durable) + b"\n")

    with pytest.raises(ResumeRefused, match="parallel recovery live intent is damaged"):
        assess(run)


def test_resume_assess_rejects_live_parallel_tape_identity_mismatch(tmp_path):
    run = tmp_path / "run"
    _live_pending(run)
    tape = run / "tool-tape.jsonl"
    record = json.loads(tape.read_text())
    record["capability_digest"] = "c" * 64
    tape.write_text(json.dumps(record) + "\n")

    with pytest.raises(ResumeRefused, match="live intent identity differs from the tape"):
        assess(run)


def test_resume_assess_accepts_only_supported_parallel_pending_recovery_without_mutation(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    spec = _run_spec(run)
    (run / "spec.json").write_text(json.dumps(spec))
    artifact_root = run / "parallel" / "call"
    artifact_root.mkdir(parents=True)
    intent = _recovery_intent()
    _pending(run, "parallel", intent)
    before = sorted((path.relative_to(run).as_posix(), path.read_bytes())
                    for path in run.rglob("*") if path.is_file())

    state = assess(run)

    after = sorted((path.relative_to(run).as_posix(), path.read_bytes())
                   for path in run.rglob("*") if path.is_file())
    descriptor = state["parallel_recovery"]
    assert descriptor["schema"] == "stateframe-parallel-recovery/1"
    assert descriptor["call_id"] == "call"
    assert descriptor["intent_digest"] == hashlib.sha256(canonical_json(intent)).hexdigest()
    assert descriptor["txns"] == []
    assert descriptor["proposals"] == []
    assert descriptor["executions"] == []
    assert before == after
    assert not (artifact_root / "txn").exists()


def test_resume_assess_accepts_valid_pending_input_refusal_without_artifacts(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps(_run_spec(run)))
    arguments = {"subframes": [{"id": "broken", "owns": "not-an-array"}]}
    _pending_input_refusal(run, {"schema": "stateframe-parallel-input-refusal/1",
                                 "arguments": arguments})
    before = (run / "tool-tape.jsonl").read_bytes()
    state = assess(run)
    assert state["parallel_recovery"] == {
        "schema": "stateframe-parallel-input-refusal/1", "call_id": "pending-call",
        "arguments": arguments}
    assert (run / "tool-tape.jsonl").read_bytes() == before


def test_resume_assess_keeps_recorded_input_refusal_when_parser_now_accepts(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps(_run_spec(run)))
    arguments = {"subframes": [
        {"id": "a", "outcome": "First", "owns": ["a.py"], "creates": []},
        {"id": "b", "outcome": "Second", "owns": ["b.py"], "creates": []},
    ]}
    _pending_input_refusal(run, {"schema": "stateframe-parallel-input-refusal/1",
                                 "arguments": arguments})
    refusal = Refusal("INVALID_ARGUMENTS", "subframe a outcome: historical refusal",
                      (), (), (), "provide outcome as text")
    admission = AdmissionResult("REFUSED", refusal, (), (), (), 0, "before", ParallelLimits())
    result = ParallelResult("pending-call", admission, (), (), "NOT_RUN", (), "NOT_RUN",
                            "REFUSED INVALID_ARGUMENTS: subframe a outcome: historical refusal",
                            ParallelMeasured(0.0, 0.0, 0.0, 0, 0.0))
    output = run / "parallel/pending-call/result.json"
    output.parent.mkdir(parents=True)
    output.write_bytes(canonical_json(result.to_json()))

    state = assess(run)
    assert state["parallel_recovery"] == {
        "schema": "stateframe-parallel-input-refusal/1", "call_id": "pending-call",
        "arguments": arguments,
    }


def test_resume_assess_rejects_tampered_input_refusal_identity(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps(_run_spec(run)))
    arguments = {"subframes": [{"id": "broken", "owns": "not-an-array"}]}
    _pending_input_refusal(run, {"schema": "stateframe-parallel-input-refusal/1",
                                 "arguments": arguments})
    tape = run / "tool-tape.jsonl"
    record = json.loads(tape.read_text())
    record["definition_digest"] = "0" * 64
    tape.write_text(json.dumps(record) + "\n")

    with pytest.raises(ResumeRefused, match="input refusal intent is invalid"):
        assess(run)


def test_resume_assess_refuses_input_refusal_with_execution_artifacts(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps(_run_spec(run)))
    _pending_input_refusal(run, {"schema": "stateframe-parallel-input-refusal/1",
                                 "arguments": {"subframes": "invalid"}})
    (run / "parallel/pending-call/units").mkdir(parents=True)
    with pytest.raises(ResumeRefused, match="unexpected execution artifacts"):
        assess(run)


@pytest.mark.parametrize("intent", [
    {"schema": "stateframe-parallel-input-refusal/1", "arguments": []},
    {"schema": "stateframe-parallel-input-refusal/1", "arguments": {"subframes": []}, "extra": 1},
])
def test_resume_assess_rejects_malformed_pending_input_refusal(tmp_path, intent):
    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps(_run_spec(run)))
    _pending_input_refusal(run, intent)
    with pytest.raises(ResumeRefused, match="input refusal"):
        assess(run)


def test_resume_refuses_incomplete_or_conflicting_parallel_intent(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps(_run_spec(run)))
    (run / "parallel" / "call").mkdir(parents=True)
    intent = _recovery_intent()
    intent["generation"] = 1
    _pending(run, "parallel", intent)
    with pytest.raises(ResumeRefused, match="identities are inconsistent"):
        assess(run)


def test_resume_refuses_execution_spec_digest_that_differs_from_disk(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps(_run_spec(run)))
    call = run / "parallel" / "call"
    snapshot = call / "snapshots" / "g0"
    snapshot.mkdir(parents=True)
    (snapshot / "a.py").write_text("VALUE = 1\n")
    digest = DirectorySnapshot(snapshot, 0).manifest_digest
    intent = _recovery_intent()
    contract = ParallelRequest.from_json(intent["request"]).units[0]
    unit_spec = UnitSpec("call", "a", "exec-a", "seed-a", contract, (), (), ("a.py",), (), "",
                         0, 0, digest, "policy", ParallelLimits())
    worker = call / "units" / "a"
    worker.mkdir(parents=True)
    (worker / "unit-spec.json").write_bytes(canonical_json(unit_spec.to_json()) + b"\n")
    intent["executions"] = [ExecutionState("a", "exec-a", str(worker), "0" * 64,
                                             1, None).to_json()]
    _pending(run, "parallel", intent)

    with pytest.raises(ResumeRefused, match="spec or start snapshot identity"):
        assess(run)


def test_resume_refuses_execution_with_different_frozen_policy(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps(_run_spec(run)))
    call = run / "parallel" / "call"
    snapshot = call / "snapshots" / "g0"
    snapshot.mkdir(parents=True)
    (snapshot / "a.py").write_text("VALUE = 1\n")
    digest = DirectorySnapshot(snapshot, 0).manifest_digest
    intent = _recovery_intent()
    contract = ParallelRequest.from_json(intent["request"]).units[0]
    unit_spec = UnitSpec("call", "a", "exec-a", "seed-a", contract, (), (), ("a.py",), (), "",
                         0, 0, digest, "different-policy", ParallelLimits())
    worker = call / "units" / "a"
    worker.mkdir(parents=True)
    (worker / "unit-spec.json").write_bytes(canonical_json(unit_spec.to_json()) + b"\n")
    intent["executions"] = [ExecutionState("a", "exec-a", str(worker), unit_spec.digest(),
                                             1, None).to_json()]
    _pending(run, "parallel", intent)

    with pytest.raises(ResumeRefused, match="spec or start snapshot identity"):
        assess(run)


@pytest.mark.parametrize("tool,args", [
    ("deploy", {}),
    ("parallel", {"schema": "stateframe-parallel-request/1", "call_id": "call"}),
])
def test_every_other_pending_intent_keeps_the_existing_refusal_message(tmp_path, tool, args):
    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps({
        "schema": "stateframe-run/1", "environment": {}, "public_row": {},
        "parent_snapshot": str(run / "parent"), "chat_url": "http://127.0.0.1/v1/chat/completions",
        "model": "x", "max_turns": 1, "repo": str(run),
    }))
    _pending(run, tool, args)

    with pytest.raises(ResumeRefused) as error:
        assess(run)

    assert str(error.value) == "operation pending-call may or may not have happened; reconcile and start a new run"


def test_archive_attempt_leaves_run_scoped_parallel_artifacts_in_place(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    spec = {"schema": "stateframe-run/1", "environment": {}, "max_turns": 2,
            "parallel_mode": "auto"}
    (run / "spec.json").write_text(json.dumps(spec))
    parallel_result = run / "parallel" / "call" / "result.json"
    parallel_result.parent.mkdir(parents=True)
    parallel_result.write_text('{"call_id":"call","preserved":true}\n')
    original_result = parallel_result.read_bytes()
    state = {"run": run, "spec": spec, "attempt": 1, "max_turns": 2, "previous_status": "ENGINE_ERROR"}

    archived = archive_attempt(state, tmp_path / "engine", use_checkpoint=False)

    assert archived.is_dir()
    assert parallel_result.is_file()
    assert parallel_result.read_bytes() == original_result
    assert not (archived / "parallel").exists()


def test_parallel_summary_uses_serialized_result_usage_and_measured_wall_values(tmp_path):
    run = tmp_path / "run"
    call_root = run / "parallel" / "call"
    call_root.mkdir(parents=True)
    admission = AdmissionResult("ADMITTED", None, (), (), ("a", "b"), 7, "a" * 64, ParallelLimits())
    outcomes = (
        UnitOutcome("a", "COMMITTED", None, None, 1, (), True, "PASS", UnitUsage(2, 1, 10, 3, 42), 1.25, 0.0),
        UnitOutcome("b", "BLOCKED", "not admitted", None, None, (), None, "SKIPPED",
                    UnitUsage(1, 1, 5, 2, None), 2.75, 0.0),
    )
    result = ParallelResult("call", admission, outcomes, (1,), "PASS", (), "COMPLETE", "all done",
                            ParallelMeasured(2.5, 4.0, 1.6, 2, 0.0))
    (call_root / "result.json").write_bytes(canonical_json(result.to_json()) + b"\n")

    summary = _parallel_summary(run)

    assert [unit["status"] for unit in summary["subframes"]] == ["COMMITTED", "BLOCKED"]
    assert [unit["usage"] for unit in summary["subframes"]] == [
        {"requests": 2, "physical_sends": 1, "prompt_tokens": 10, "completion_tokens": 3,
         "initial_frame_tokens": 42},
        {"requests": 1, "physical_sends": 1, "prompt_tokens": 5, "completion_tokens": 2,
         "initial_frame_tokens": None},
    ]
    assert [unit["wall_seconds"] for unit in summary["subframes"]] == [1.25, 2.75]
    assert summary["totals"] == {
        "requests": 3, "physical_sends": 2, "prompt_tokens": 15, "completion_tokens": 5,
        "initial_frame_tokens": 42, "subframes": 2, "subframes_with_initial_frame_tokens": 1,
        "subframes_with_wall": 2,
    }
    assert summary["calls"][0]["measured"]["parallel_wall_seconds"] == 2.5
    assert summary["calls"][0]["measured"]["estimated_serial_seconds"] == 4.0
    assert summary["measured"]["parallel_wall_seconds"] == 2.5
    assert summary["measured"]["estimated_serial_seconds"] == 4.0
    assert summary["measured"]["speedup"] == 1.6
    assert summary["measured"]["peak_active_subframes"] == 2
    assert summary["calls"][0]["measured"]["peak_active_subframes"] == 2
    assert summary["errors"] == []


def test_cli_parallel_flag_defaults_and_round_trips_for_run_and_chat():
    from stateframe import profiles

    assert "STATEFRAME_PARALLEL" not in profiles.load(profiles.DEFAULT)
    parser = build_parser()
    for command, extra in (("run", ["--task", "fix the sample"]), ("chat", [])):
        base = [command, "--repo", ".", *extra, "--endpoint", "http://127.0.0.1:8080/v1", "--model", "local-model"]
        default = parser.parse_args(base)
        assert default.parallel == "off"
        assert _run_options(default, Endpoint.from_args(default.endpoint, default.model, None)).parallel_mode == "off"

        selected = parser.parse_args([*base, "--parallel", "3"])
        options = _run_options(selected, Endpoint.from_args(selected.endpoint, selected.model, None))
        assert selected.parallel == "3"
        assert options.parallel_mode == "3"

    nine = parser.parse_args(["chat", "--repo", ".", "--endpoint", "http://127.0.0.1/v1",
                              "--model", "x", "--parallel", "9", "--parallel-max", "16"])
    assert nine.parallel == "9"
    assert _run_options(nine, Endpoint.from_args(nine.endpoint, nine.model, None)).parallel_cli == {
        "max_concurrency": 16}
    help_text = parser._subparsers._group_actions[0].choices["run"].format_help()
    assert all(option["flag"] in help_text for option in PARALLEL_LIMIT_OPTIONS.values())
    assert all("unit" not in option["flag"].casefold() for option in PARALLEL_LIMIT_OPTIONS.values())
    assert "--parallel-max-units" not in help_text
    renamed = parser.parse_args(["chat", "--repo", ".", "--endpoint", "http://127.0.0.1/v1",
                                 "--model", "x", "--parallel-max-subframes", "9"])
    assert _run_options(renamed, Endpoint.from_args(renamed.endpoint, renamed.model, None)).parallel_cli == {
        "max_units": 9}
    with pytest.raises(SystemExit):
        parser.parse_args(["chat", "--repo", ".", "--endpoint", "http://127.0.0.1/v1",
                           "--model", "x", "--parallel-max-units", "9"])


def test_parallel_limits_are_validated_before_run_directory_creation(tmp_path):
    from stateframe.runner import RunError, RunOptions, prepare

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text("def test_sample(): assert True\n")
    endpoint = Endpoint.from_args("http://127.0.0.1:8080/v1", "local-model", None)
    options = RunOptions(repo, "check", endpoint, run_dir=tmp_path / "run", parallel_mode="9")
    with pytest.raises(RunError, match="--parallel-max") as raised:
        prepare(options)
    assert str(raised.value) == (
        "--parallel 9 exceeds --parallel-max=4; raise --parallel-max or "
        "STATEFRAME_PARALLEL_MAX_CONCURRENCY")
    assert not (tmp_path / "run").exists()

    options.parallel_cli = {"max_concurrency": 16, "unit_request_tokens": 40000}
    with pytest.raises(RunError, match="--parallel-subframe-tokens"):
        prepare(options)
    assert not (tmp_path / "run").exists()


def test_auto_concurrency_sums_configured_endpoint_slots_without_clamping_to_four(monkeypatch):
    from stateframe import parallel_resources
    from stateframe.runner import auto_parallel_concurrency

    primary = "http://127.0.0.1:8080/v1"
    secondary = "http://127.0.0.1:8081/v1"
    seen = []

    def slots(endpoint):
        seen.append(endpoint)
        return list(range(5)) if endpoint == primary else list(range(7))

    monkeypatch.setattr(parallel_resources, "idle_slots", slots)
    spec = {"chat_url": primary + "/chat/completions",
            "routes": {"unit": {"endpoint": secondary}}}
    assert auto_parallel_concurrency(spec, 16) == 12
    assert seen == [primary, secondary]

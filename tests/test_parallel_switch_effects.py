"""Behavioral evidence for the documented parallel settings."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateframe import runner
from stateframe.engine.kernel_check_permits import acquire
from stateframe.engine.kernel_parallel_admission import DirectorySnapshot, admit
from stateframe.engine.kernel_parallel_budget import BudgetStore
from stateframe.engine.kernel_parallel_records import (
    PARALLEL_LIMIT_OPTIONS,
    CheckRecipe,
    ParallelLimits,
    ParallelRequest,
    UnitContract,
    UnitSpec,
    format_parallel_limit,
)
from stateframe.engine.kernel_unit_policy import UnitPolicy


def _switch_values(monkeypatch, values: dict[str, str]) -> dict[str, str]:
    for name in runner.OPT_IN_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    for switch, value in values.items():
        monkeypatch.setenv(switch, value)
    resolved = runner.resolve_opt_in_switches()
    assert all(resolved[switch] == value for switch, value in values.items())
    return resolved


def _switches(monkeypatch, switch: str, value: str) -> dict[str, str]:
    return _switch_values(monkeypatch, {switch: value})


def _limits(monkeypatch, setting: str, switch: str, value: int, *,
            additional: dict[str, int] | None = None) -> ParallelLimits:
    assert PARALLEL_LIMIT_OPTIONS[setting]["switch"] == switch
    values = {switch: str(value)}
    values.update({name: str(item) for name, item in (additional or {}).items()})
    return ParallelLimits.from_settings(
        _switch_values(monkeypatch, values), {}, prompt_budget=33_344)


def _contract(unit_id: str, path: str, *, reads: tuple[str, ...] = ()) -> UnitContract:
    return UnitContract(
        unit_id, f"update {unit_id}", (path,), (), (), (), reads, (),
        CheckRecipe((), (), "none"),
    )


def _admit(tmp_path: Path, files: dict[str, str], units: tuple[UnitContract, ...],
           limits: ParallelLimits):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    snapshot = DirectorySnapshot(root, 1)
    request = ParallelRequest("call", units, "2")
    return admit(request, snapshot, (), limits, protected=lambda _path: False,
                 case_insensitive=False)


def test_max_subframes_switch_caps_admitted_units(tmp_path, monkeypatch):
    files = {f"src/{name}.py": f"VALUE = {index}\n"
             for index, name in enumerate(("a", "b", "c"))}
    units = tuple(_contract(name, f"src/{name}.py") for name in ("a", "b", "c"))

    limited = _limits(monkeypatch, "max_units", "STATEFRAME_PARALLEL_MAX_SUBFRAMES", 2)
    refused = _admit(tmp_path, files, units, limited)
    assert (refused.status, refused.refusal.reason) == ("REFUSED", "TOO_MANY_UNITS")
    assert refused.refusal.detail == format_parallel_limit("max_units", 2)

    allowed = _limits(monkeypatch, "max_units", "STATEFRAME_PARALLEL_MAX_SUBFRAMES", 3)
    assert _admit(tmp_path, files, units, allowed).status == "ADMITTED"


def test_min_subframes_switch_requires_configured_floor(tmp_path, monkeypatch):
    files = {f"src/{name}.py": "VALUE = 1\n" for name in ("a", "b")}
    units = tuple(_contract(name, f"src/{name}.py") for name in ("a", "b"))

    limited = _limits(monkeypatch, "min_units", "STATEFRAME_PARALLEL_MIN_SUBFRAMES", 3)
    refused = _admit(tmp_path, files, units, limited)
    assert (refused.status, refused.refusal.reason) == ("REFUSED", "TOO_FEW_UNITS")
    assert refused.refusal.detail == format_parallel_limit("min_units", 3)

    allowed = _limits(monkeypatch, "min_units", "STATEFRAME_PARALLEL_MIN_SUBFRAMES", 2)
    assert _admit(tmp_path, files, units, allowed).status == "ADMITTED"


def test_max_paths_per_subframe_switch_caps_declared_paths(tmp_path, monkeypatch):
    files = {"src/a.py": "VALUE = 1\n", "src/b.py": "VALUE = 2\n",
             "src/support.py": "VALUE = 3\n"}
    units = (
        _contract("a", "src/a.py", reads=("src/support.py",)),
        _contract("b", "src/b.py"),
    )

    limited = _limits(
        monkeypatch, "max_paths_per_unit", "STATEFRAME_PARALLEL_MAX_PATHS_PER_SUBFRAME", 1)
    refused = _admit(tmp_path, files, units, limited)
    assert (refused.status, refused.refusal.reason) == ("REFUSED", "TOO_MANY_PATHS")
    assert refused.refusal.detail == format_parallel_limit("max_paths_per_unit", 1)

    allowed = _limits(
        monkeypatch, "max_paths_per_unit", "STATEFRAME_PARALLEL_MAX_PATHS_PER_SUBFRAME", 2)
    assert _admit(tmp_path, files, units, allowed).status == "ADMITTED"


def test_max_paths_total_switch_caps_all_declared_paths(tmp_path, monkeypatch):
    files = {"src/a.py": "VALUE = 1\n", "src/b.py": "VALUE = 2\n"}
    units = (_contract("a", "src/a.py"), _contract("b", "src/b.py"))

    limited = _limits(
        monkeypatch, "max_paths_total", "STATEFRAME_PARALLEL_MAX_PATHS_TOTAL", 1,
        additional={"STATEFRAME_PARALLEL_MAX_PATHS_PER_SUBFRAME": 1})
    refused = _admit(tmp_path, files, units, limited)
    assert (refused.status, refused.refusal.reason) == ("REFUSED", "TOO_MANY_PATHS")
    assert refused.refusal.detail == format_parallel_limit("max_paths_total", 1)

    allowed = _limits(
        monkeypatch, "max_paths_total", "STATEFRAME_PARALLEL_MAX_PATHS_TOTAL", 2,
        additional={"STATEFRAME_PARALLEL_MAX_PATHS_PER_SUBFRAME": 1})
    assert _admit(tmp_path, files, units, allowed).status == "ADMITTED"


def test_subframe_request_tokens_switch_blocks_oversized_first_frame(tmp_path, monkeypatch):
    import kernel_request_budget
    from kernel_context import token_estimate
    from probes.react_repair import _UnitContextTooLarge, _enforce_unit_first_request

    source_path = "pkg/a.py"
    source = 'value = "' + "x" * 4000 + '"\n'
    (tmp_path / "pkg").mkdir()
    (tmp_path / source_path).write_text(source, encoding="utf-8")
    messages = [{"role": "user", "content": f"[Current source: {source_path}]\n{source}"}]
    tools = [{"type": "function", "function": {"name": "read"}}]
    measured = token_estimate({"messages": messages, "tools": tools})
    assert measured > 100
    monkeypatch.setattr(
        kernel_request_budget, "_request_token_meter",
        lambda *_args: SimpleNamespace(
            measure=lambda body: (token_estimate(body), "fixture_counter"),
            measure_subframe_cap=lambda body, **_kwargs: (token_estimate(body), "tokenize")),
    )
    events: list[tuple[str, dict[str, object]]] = []
    runtime = SimpleNamespace(journal=SimpleNamespace(
        emit=lambda kind, **fields: events.append((kind, fields))))
    prepared = SimpleNamespace(messages=messages, capture=SimpleNamespace(request_id="r1"))
    tools_body = tools

    def policy_for(limit: int) -> UnitPolicy:
        limits = _limits(
            monkeypatch, "unit_request_tokens",
            "STATEFRAME_PARALLEL_SUBFRAME_REQUEST_TOKENS", limit)
        contract = UnitContract(
            "a", "update a", (source_path,), (), (), (), (), (), CheckRecipe((), (), "none"))
        spec = UnitSpec(
            "call", "a", "execution", "seed", contract, (), (), (source_path,), (), "",
            0, 0, "manifest", "policy", limits,
        )
        return UnitPolicy(spec)

    low_policy = policy_for(100)
    low_run = tmp_path / "low-run"
    with pytest.raises(_UnitContextTooLarge, match="ceiling 100"):
        _enforce_unit_first_request(runtime, prepared, tools_body, low_policy, tmp_path, low_run)
    assert events[0][0] == "unit_frame_tokens"
    assert {key: events[0][1][key] for key in
            ("request", "tokens", "counter", "meter_kind", "confidence", "calibration_id", "cap")} == {
        "request": "r1", "tokens": measured, "counter": "tokenize", "meter_kind": "tokenize",
        "confidence": "exact", "calibration_id": "tokenize-exact", "cap": 100,
    }
    assert events[1][0] == "provider_request_not_dispatched"
    assert events[1][1]["reason_code"] == "unit_context_too_large"
    assert not (low_run / "wire").exists()

    events.clear()
    accepted = _enforce_unit_first_request(
        runtime, prepared, tools_body, policy_for(measured), tmp_path, tmp_path / "accepted-run")
    assert accepted is None
    assert len(events) == 1 and events[0][0] == "unit_frame_tokens"
    assert events[0][1]["tokens"] == measured
    assert events[0][1]["meter_kind"] == "tokenize"
    assert events[0][1]["confidence"] == "exact"
    assert events[0][1]["calibration_id"] == "tokenize-exact"


def test_max_owned_source_bytes_switch_rejects_large_owned_source(tmp_path, monkeypatch):
    source = "value = 1\n"
    files = {"src/a.py": source, "src/b.py": "value = 2\n"}
    units = (_contract("a", "src/a.py"), _contract("b", "src/b.py"))

    limited = _limits(
        monkeypatch, "max_owned_source_bytes", "STATEFRAME_PARALLEL_MAX_OWNED_SOURCE_BYTES",
        len(source.encode("utf-8")) - 1)
    refused = _admit(tmp_path, files, units, limited)
    assert (refused.status, refused.refusal.reason) == ("REFUSED", "UNIT_CONTEXT_TOO_LARGE")
    assert refused.refusal.detail == (
        f"owned_source_bytes={len(source.encode('utf-8'))}; "
        + format_parallel_limit("max_owned_source_bytes", len(source.encode("utf-8")) - 1))

    allowed = _limits(
        monkeypatch, "max_owned_source_bytes", "STATEFRAME_PARALLEL_MAX_OWNED_SOURCE_BYTES",
        len(source.encode("utf-8")))
    assert _admit(tmp_path, files, units, allowed).status == "ADMITTED"


def test_subframe_max_requests_switch_caps_physical_sends(tmp_path, monkeypatch):
    from stateframe.parallel_resources import SendLedger

    limits = _limits(
        monkeypatch, "unit_max_requests", "STATEFRAME_PARALLEL_SUBFRAME_MAX_REQUESTS", 2)
    limited_path = tmp_path / "limited/parallel/call/units/a/sends-ledger.jsonl"
    limited_path.parent.mkdir(parents=True)
    ledger = SendLedger(limited_path)
    assert ledger.authorize_with_cap("send-1", limits.unit_max_requests)
    assert ledger.authorize_with_cap("send-2", limits.unit_max_requests)
    assert not ledger.authorize_with_cap("send-3", limits.unit_max_requests)
    assert ledger.remaining(limits.unit_max_requests) == 0

    control = _limits(
        monkeypatch, "unit_max_requests", "STATEFRAME_PARALLEL_SUBFRAME_MAX_REQUESTS", 3)
    control_path = tmp_path / "control/parallel/call/units/a/sends-ledger.jsonl"
    control_path.parent.mkdir(parents=True)
    control_ledger = SendLedger(control_path)
    assert [control_ledger.authorize_with_cap(f"send-{index}", control.unit_max_requests)
            for index in range(1, 4)] == [True, True, True]


def test_parent_reserve_requests_switch_preserves_parent_allowance(tmp_path, monkeypatch):
    limited = _limits(
        monkeypatch, "parent_reserve_requests", "STATEFRAME_PARALLEL_PARENT_RESERVE_REQUESTS", 3)
    limited_run = tmp_path / "limited-run"
    limited_budget = BudgetStore(
        limited_run, limit=6, parent_reserve=limited.parent_reserve_requests)
    assert not limited_budget.reserve("call", {"a": 3}, parent_steps_spent=1)
    limited_state = json.loads((limited_run / "parallel/budget.json").read_text())
    assert limited_state["calls"] == {}
    assert limited_state["parent_reserve"] == 3

    control = _limits(
        monkeypatch, "parent_reserve_requests", "STATEFRAME_PARALLEL_PARENT_RESERVE_REQUESTS", 2)
    control_run = tmp_path / "control-run"
    control_budget = BudgetStore(
        control_run, limit=6, parent_reserve=control.parent_reserve_requests)
    assert control_budget.reserve("call", {"a": 3}, parent_steps_spent=1)
    control_state = json.loads((control_run / "parallel/budget.json").read_text())
    assert control_state["calls"]["call"]["reservations"] == {"a": 3}
    assert control_state["parent_reserve"] == 2


def test_check_permits_switch_limits_active_checks(tmp_path, monkeypatch):
    one = _limits(monkeypatch, "check_permits", "STATEFRAME_PARALLEL_CHECK_PERMITS", 1)
    root = tmp_path / "permits"
    with acquire(root, one.check_permits, timeout=0):
        with pytest.raises(TimeoutError, match="no check permit available"):
            with acquire(root, one.check_permits, timeout=0):
                pytest.fail("one configured permit admitted a concurrent check")

        two = _limits(monkeypatch, "check_permits", "STATEFRAME_PARALLEL_CHECK_PERMITS", 2)
        with acquire(root, two.check_permits, timeout=0) as permit:
            assert permit == 1


def test_max_concurrency_switch_caps_auto_dispatch(monkeypatch):
    import stateframe.parallel_resources as resources

    monkeypatch.setattr(resources, "idle_slots", lambda _endpoint: [0, 1, 2, 3])
    spec = {"chat_url": "http://127.0.0.1:8080/v1"}
    limited = _limits(
        monkeypatch, "max_concurrency", "STATEFRAME_PARALLEL_MAX_CONCURRENCY", 2)
    assert runner.auto_parallel_concurrency(spec, limited.max_concurrency) == 2

    control = _limits(
        monkeypatch, "max_concurrency", "STATEFRAME_PARALLEL_MAX_CONCURRENCY", 4)
    assert runner.auto_parallel_concurrency(spec, control.max_concurrency) == 4


def test_slots_switch_filters_lease_candidates(monkeypatch):
    from stateframe.parallel_resources import parse_slot_pool, slot_candidates

    switch = "STATEFRAME_PARALLEL_SLOTS"
    resolved = _switches(monkeypatch, switch, "5,2")
    pool = parse_slot_pool(resolved[switch], source=switch)
    idle = [0, 2, 5, 7]

    assert pool == (5, 2)
    assert slot_candidates(pool, idle, 0, fallback_capacity=8) == [5, 2]
    assert slot_candidates(None, idle, 0, fallback_capacity=8) == [0, 2, 5, 7]

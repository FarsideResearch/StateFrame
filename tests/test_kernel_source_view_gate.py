"""FU-070-ALLOC: deferred-first subframe gate and FILE floor check."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kernel_source_view import file_floor_messages


def _notice(path: str) -> dict[str, str]:
    return {"role": "user", "content":
            "[Deferred required source paths]\nRequired source paths are not attached; "
            f"read one only if your edit needs it:\n- {path}"}


def _gate_fixture(tmp_path, monkeypatch, *, cap: int):
    import kernel_request_budget
    from kernel_context import token_estimate
    from test_unit_visibility import policy
    from kernel_unit_policy import UnitPolicy

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/a.py").write_text("".join(f"V{n} = {n}\n" for n in range(400)))
    original = policy(("pkg/a.py",))
    unit = UnitPolicy(replace(original.spec, limits=replace(
        original.spec.limits, unit_request_tokens=cap)))
    monkeypatch.setattr(kernel_request_budget, "_request_token_meter",
                        lambda *_args: SimpleNamespace(
                            measure=lambda body: (token_estimate(body), "fixture"),
                            measure_subframe_cap=lambda body, **_kw: (token_estimate(body), "tokenize")))
    events: list[tuple[str, dict[str, object]]] = []
    runtime = SimpleNamespace(journal=SimpleNamespace(
        emit=lambda kind, **fields: events.append((kind, fields))))
    return unit, runtime, events


def _deferred_prepared(request_id: str):
    return SimpleNamespace(messages=[{"role": "user", "content": "Task: fix a"}, _notice("pkg/a.py")],
                           capture=SimpleNamespace(request_id=request_id))


def test_gate_passes_a_deferred_first_request_with_a_current_obligation(tmp_path, monkeypatch) -> None:
    from probes.react_repair import _enforce_unit_first_request

    unit, runtime, _events = _gate_fixture(tmp_path, monkeypatch, cap=100_000)
    source = (tmp_path / "pkg/a.py").read_bytes()
    runtime._request_source_closure = {
        "deferred_paths": ["pkg/a.py"],
        "revisions": {"pkg/a.py": "sha256:" + hashlib.sha256(source).hexdigest()}}
    _enforce_unit_first_request(runtime, _deferred_prepared("r1"), [], unit, tmp_path, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_gate_refuses_a_deferred_request_whose_obligation_is_stale(tmp_path, monkeypatch) -> None:
    from probes.react_repair import _UnitContextTooLarge, _enforce_unit_first_request

    unit, runtime, _events = _gate_fixture(tmp_path, monkeypatch, cap=100_000)
    runtime._request_source_closure = {"deferred_paths": ["pkg/a.py"],
                                       "revisions": {"pkg/a.py": "sha256:" + "0" * 64}}
    with pytest.raises(_UnitContextTooLarge, match="missing full source: pkg/a.py"):
        _enforce_unit_first_request(runtime, _deferred_prepared("r2"), [], unit, tmp_path,
                                    tmp_path / "run")


def test_gate_passes_three_pages_and_map_via_source_view(tmp_path, monkeypatch) -> None:
    from probes.react_repair import _enforce_unit_first_request

    unit, runtime, _events = _gate_fixture(tmp_path, monkeypatch, cap=100_000)
    source = (tmp_path / "pkg/a.py").read_text()
    messages = [{"role": "user", "content": "Task: fix a"}, *file_floor_messages("pkg/a.py", source)]
    prepared = SimpleNamespace(messages=messages, capture=SimpleNamespace(request_id="r3"))
    _enforce_unit_first_request(runtime, prepared, [], unit, tmp_path, tmp_path / "run")


def test_floor_above_band_is_needs_source_capacity_with_zero_transport(tmp_path, monkeypatch) -> None:
    from kernel_terminal_policy import terminal_category
    from probes.react_repair import _UnitContextTooLarge, _enforce_unit_first_request

    unit, runtime, events = _gate_fixture(tmp_path, monkeypatch, cap=300)
    source = (tmp_path / "pkg/a.py").read_bytes()
    runtime._request_source_closure = {
        "deferred_paths": ["pkg/a.py"],
        "revisions": {"pkg/a.py": "sha256:" + hashlib.sha256(source).hexdigest()}}
    run_dir = tmp_path / "run"
    with pytest.raises(_UnitContextTooLarge, match="needs_source_capacity: pkg/a.py"):
        _enforce_unit_first_request(runtime, _deferred_prepared("r4"), [], unit, tmp_path, run_dir)
    assert runtime.system_outcome == "needs_source_capacity"
    assert terminal_category(runtime.system_outcome, runtime) == "fatal"
    assert "floor" in runtime.unit_context_detail and "band 300" in runtime.unit_context_detail
    kinds = [kind for kind, _ in events]
    assert kinds == ["unit_frame_tokens", "provider_request_not_dispatched"]
    assert events[1][1]["reason_code"] == "needs_source_capacity"
    assert (run_dir / "unit-refused-request.json").is_file()
    assert not (run_dir / "wire").exists()
    json.loads((run_dir / "unit-refused-request.json").read_text())


def _value_file_prepared(tmp_path, request_id: str, max_request_bytes: int | None):
    """2,800 `VALUE_nnnn = n` lines: the FU-070 checkpoint shape."""
    (tmp_path / "pkg/a.py").write_text("".join(f"VALUE_{n:04d} = {n}\n" for n in range(2800)))
    source = (tmp_path / "pkg/a.py").read_bytes()
    capture = SimpleNamespace(request_id=request_id)
    if max_request_bytes is not None:
        capture._capacity_retry_limits = SimpleNamespace(max_request_bytes=max_request_bytes)
    return SimpleNamespace(
        messages=[{"role": "user", "content": "Task: fix a"}, _notice("pkg/a.py")], capture=capture
    ), {"deferred_paths": ["pkg/a.py"],
        "revisions": {"pkg/a.py": "sha256:" + hashlib.sha256(source).hexdigest()}}


def test_floor_over_serialized_byte_cap_is_needs_source_capacity(tmp_path, monkeypatch) -> None:
    from probes.react_repair import _UnitContextTooLarge, _enforce_unit_first_request

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    unit, runtime, events = _gate_fixture(tmp_path, monkeypatch, cap=10_000_000)
    prepared, closure = _value_file_prepared(tmp_path, "r5", max_request_bytes=30_000)
    runtime._request_source_closure = closure
    run_dir = tmp_path / "run"
    with pytest.raises(_UnitContextTooLarge, match="needs_source_capacity: pkg/a.py floor") as caught:
        _enforce_unit_first_request(runtime, prepared, [], unit, tmp_path, run_dir)
    detail = str(caught.value)
    assert "serialized bytes exceeds cap 30000" in detail
    assert runtime.system_outcome == "needs_source_capacity"
    assert [kind for kind, _ in events][-1] == "provider_request_not_dispatched"
    assert events[-1][1]["reason_code"] == "needs_source_capacity"
    assert (run_dir / "unit-refused-request.json").is_file()


def test_floor_within_serialized_byte_cap_passes(tmp_path, monkeypatch) -> None:
    from probes.react_repair import _enforce_unit_first_request

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    unit, runtime, _events = _gate_fixture(tmp_path, monkeypatch, cap=10_000_000)
    prepared, closure = _value_file_prepared(tmp_path, "r6", max_request_bytes=200_000)
    runtime._request_source_closure = closure
    _enforce_unit_first_request(runtime, prepared, [], unit, tmp_path, tmp_path / "run")


ALLOCATOR_ENV = "STATEFRAME_WORKING_SET_ALLOCATOR"


def _clean_floor_body(path: str, source: str) -> str:
    """The FILE floor as the transport sends it: task row plus every page and the map line."""
    body = {"messages": [{"role": "user", "content": "Task: fix a"},
                         *file_floor_messages(path, source)], "tools": []}
    return json.dumps(body)


def test_byte_floor_never_runs_with_allocator_off(tmp_path, monkeypatch) -> None:
    import kernel_source_view
    from probes.react_repair import _enforce_unit_first_request

    monkeypatch.delenv(ALLOCATOR_ENV, raising=False)

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("serialized byte floor must not run with the allocator off")

    monkeypatch.setattr(kernel_source_view, "file_floor_bytes", _must_not_run)
    unit, runtime, events = _gate_fixture(tmp_path, monkeypatch, cap=10_000_000)
    prepared, closure = _value_file_prepared(tmp_path, "r7", max_request_bytes=30_000)
    runtime._request_source_closure = closure
    _enforce_unit_first_request(runtime, prepared, [], unit, tmp_path, tmp_path / "run")
    assert [kind for kind, _ in events] == ["unit_frame_tokens"]


def test_byte_floor_still_enforced_when_token_meter_times_out(tmp_path, monkeypatch) -> None:
    import kernel_parallel_meter
    from probes.react_repair import _UnitContextTooLarge, _enforce_unit_first_request

    monkeypatch.setenv(ALLOCATOR_ENV, "on")

    def _timeout(*_args, **_kwargs):
        raise TimeoutError("token meter timed out")

    monkeypatch.setattr(kernel_parallel_meter, "measure_subframe_cap", _timeout)
    unit, runtime, _events = _gate_fixture(tmp_path, monkeypatch, cap=10_000_000)
    prepared, closure = _value_file_prepared(tmp_path, "r8", max_request_bytes=30_000)
    runtime._request_source_closure = closure
    with pytest.raises(_UnitContextTooLarge, match="needs_source_capacity: pkg/a.py floor") as caught:
        _enforce_unit_first_request(runtime, prepared, [], unit, tmp_path, tmp_path / "run")
    assert "serialized bytes exceeds cap 30000" in str(caught.value)
    assert runtime.system_outcome == "needs_source_capacity"


def test_byte_floor_is_measured_on_the_normalized_wire(tmp_path, monkeypatch) -> None:
    """A private `_` key on a fixed row is stripped before transport, so it must not count."""
    from probes.react_repair import _enforce_unit_first_request

    monkeypatch.setenv(ALLOCATOR_ENV, "on")
    unit, runtime, _events = _gate_fixture(tmp_path, monkeypatch, cap=10_000_000)
    prepared, closure = _value_file_prepared(tmp_path, "r9", max_request_bytes=None)
    prepared.messages[0]["_classification"] = "p" * 5000
    runtime._request_source_closure = closure
    source = (tmp_path / "pkg/a.py").read_text()
    wire = len(_clean_floor_body("pkg/a.py", source).encode())
    prepared.capture._capacity_retry_limits = SimpleNamespace(max_request_bytes=wire)
    _enforce_unit_first_request(runtime, prepared, [], unit, tmp_path, tmp_path / "run")


def test_byte_floor_just_under_cap_passes_and_just_over_is_refused(tmp_path, monkeypatch) -> None:
    from probes.react_repair import _UnitContextTooLarge, _enforce_unit_first_request

    monkeypatch.setenv(ALLOCATOR_ENV, "on")
    unit, runtime, _events = _gate_fixture(tmp_path, monkeypatch, cap=10_000_000)
    prepared, closure = _value_file_prepared(tmp_path, "r10", max_request_bytes=None)
    runtime._request_source_closure = closure
    source = (tmp_path / "pkg/a.py").read_text()
    floor = len(_clean_floor_body("pkg/a.py", source).encode())

    prepared.capture._capacity_retry_limits = SimpleNamespace(max_request_bytes=floor)
    _enforce_unit_first_request(runtime, prepared, [], unit, tmp_path, tmp_path / "run")

    prepared.capture._capacity_retry_limits = SimpleNamespace(max_request_bytes=floor - 1)
    with pytest.raises(_UnitContextTooLarge, match="needs_source_capacity: pkg/a.py floor"):
        _enforce_unit_first_request(runtime, prepared, [], unit, tmp_path, tmp_path / "run2")

"""Count real source fitting work through all passes of request preparation."""

from __future__ import annotations

import copy
import json
import os
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import capability_request_preflight as preflight
import pytest

import kernel_request_budget as budget
import kernel_request_evidence_closure as closure
import kernel_request_source_window as window
from kernel_stateframe_framefit5 import PRODUCT_MODE, _serialize_state_message
from kernel_stateframe_runtime import StateFrameRuntime


TURN_SWITCHES = (
    "WORKSPACE_OBJECTS", "TURN_ATOMIC_PUBLICATION", "TURN_BOUNDED_QUALIFICATION",
    "TURN_DELIVERY_PROOFS", "TURN_GENERATION_OVERLAP", "TURN_INDEXED_READS",
    "TURN_MINIMAL_LAUNCH", "TURN_PANELS", "TURN_REPO_INTEL", "TURN_TIMING",
)


def _switches(monkeypatch, mode):
    for name in TURN_SWITCHES:
        monkeypatch.setenv("STATEFRAME_" + name, mode)
    monkeypatch.setenv(window.SITE_FACTS_ENV, "on")
    monkeypatch.setenv(budget.MODE_ENV, "on")


def _field_fixture(root, *, wide=False):
    root.mkdir(parents=True, exist_ok=True)
    paths = tuple(f"module_{index:02}.py" for index in range(11))
    for index in range(4989 if wide else 0):
        (root / f"unrelated_{index:04}.py").write_text(f"VALUE = {index}\n")
    sources = {}
    for index, path in enumerate(paths):
        imports = f"import optional_provider_{index}\n" if index else ""
        sources[path] = ("# unrelated source padding\n" * 560 + imports
                         + f"def repair_case():\n    return {index}\n")
        (root / path).write_text(sources[path])
    ticket = "Fix repair_case() in " + ", ".join(paths) + "."
    receipts = []
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.owner = SimpleNamespace(root=root, fn="repair_case", task_row={
        "editable_paths": paths, "problem_statement": ticket,
    })
    runtime.target = root / paths[0]
    runtime.stateframe_framefit_mode = "on"
    runtime.stateframe_f4_mode = "off"
    runtime.frame_ceiling = 33_344
    runtime.workspace_scope = "source-fit-fixture"
    runtime.journal = SimpleNamespace(emit=lambda kind, **row: receipts.append((kind, row)))
    task = {"role": "user", "content": ticket}
    messages = [{"role": "system", "content": "Use exact source evidence."}, task]
    state = {"state": {"framefit5": {"mode": PRODUCT_MODE, "action_ledger": {}}}}
    messages += [
        {"role": "user", "content": _serialize_state_message(state)},
        {"role": "user", "content": f"[Current source: {paths[0]}]\n{sources[paths[0]]}"},
    ]
    read = {"type": "function", "function": {
        "name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }}
    return runtime, messages, [read], receipts


def _prepare(runtime, messages, tools):
    capture = SimpleNamespace(journal=runtime.journal, request_id="source-fit", step=2)
    wire_messages, wire_tools = budget.apply_runtime_budget(
        runtime, capture, copy.deepcopy(messages), copy.deepcopy(tools),
        tokenizer=lambda _text: [1],
    )
    assert not getattr(capture, "request_budget_unavailable", None)
    return {"messages": wire_messages, "tools": wire_tools}, capture


def _probe(monkeypatch, root):
    calls = Counter()
    reads = Counter()
    anchors = Counter()
    projections = Counter()
    reads_by_pass = []
    seconds_by_pass = []
    real_bind = preflight.bound_runtime_sources

    def source_reads():
        return sum(count for path, count in reads.items() if path.is_relative_to(root))

    def bind(*args, **kwargs):
        start = source_reads()
        started = time.perf_counter()
        result = real_bind(*args, **kwargs)
        seconds_by_pass.append(time.perf_counter() - started)
        reads_by_pass.append(source_reads() - start)
        return result

    monkeypatch.setattr(preflight, "bound_runtime_sources", bind)

    def wrap(module, name, counter, key):
        original = getattr(module, name)

        def counted(*args, **kwargs):
            counter[key(*args, **kwargs)] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)

    for name in ("read_text", "read_bytes"):
        wrap(Path, name, reads, lambda path, *_args, **_kwargs: path)
    wrap(window, "build_source_closure", calls, lambda *_a, **_kw: "acquire")
    wrap(window, "_cached_closure_is_fresh", calls, lambda *_a, **_kw: "validate")
    wrap(closure, "_dependency_candidates", calls, lambda *_a, **_kw: "dependency")
    wrap(window, "verify_final_source_delivery", calls, lambda *_a, **_kw: "final proof")

    def anchor_key(source, runtime, ticket, *, path="", is_target=True, **_options):
        units = closure.units_for_path(closure.closure_plan_for_runtime(runtime), path)
        return (source, ticket, path, is_target, tuple(unit.key() for unit in units
                                                     if unit.role != "target_reservation"))

    wrap(window, "_anchor_ranges", anchors, anchor_key)
    wrap(window, "_fit_ranges", projections,
         lambda _lines, anchors, **kwargs: (
             tuple(anchors), kwargs["path"], kwargs["revision"], kwargs["max_chars"],
             kwargs["context_lines"],
         ))
    return calls, source_reads, reads_by_pass, seconds_by_pass, anchors, projections


def test_three_budget_passes_share_source_work(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    runtime, messages, tools, receipts = _field_fixture(tmp_path, wide=True)
    calls, source_reads, reads_by_pass, seconds_by_pass, anchors, projections = _probe(monkeypatch, tmp_path)
    started = time.perf_counter()
    _wire, _capture = _prepare(runtime, messages, tools)
    elapsed = time.perf_counter() - started
    passes = [row for kind, row in receipts if kind == "request_source_closure"]
    work = {**calls, "source reads": source_reads(), "reads by pass": reads_by_pass, "anchors": sum(anchors.values()),
            "unique anchors": len(anchors), "windows": sum(projections.values()),
            "unique windows": len(projections)}
    print(
        f"prepare={elapsed:.3f}s, source fitting={sum(seconds_by_pass):.3f}s, "
        f"spans={[row['delivered_spans'] for row in passes]}, {work}"
    )
    assert len(passes) == 3
    assert passes[-1]["delivered_spans"] == 11
    assert calls["final proof"] > 0
    assert (calls["acquire"], calls["validate"], calls["dependency"]) == (1, 0, 11), work
    assert max(anchors.values()) == max(projections.values()) == 1, work
    assert source_reads() <= 34 and reads_by_pass[1:] == [0, 0], work
    assert sum(seconds_by_pass) < 10, seconds_by_pass

    # A new prepare validates the existing closure once, then shares it again.
    reads_by_pass.clear()
    calls.clear()
    _prepare(runtime, messages, tools)
    assert (calls["acquire"], calls["validate"], calls["dependency"]) == (0, 1, 11), calls
    assert reads_by_pass[1:] == [0, 0], reads_by_pass


def _wire_bytes(body):
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _acquisition(receipts):
    return [row for kind, row in receipts if kind == "request_source_closure"]


def _assert_prepare_parity(reference, actual):
    expected, expected_capture, expected_events = reference
    wire, capture, events = actual
    assert _wire_bytes(wire) == _wire_bytes(expected)
    assert _acquisition(events) == _acquisition(expected_events)
    assert capture.request_source_final_delivery == expected_capture.request_source_final_delivery
    assert capture.request_budget_final_wire == expected_capture.request_budget_final_wire


@pytest.mark.parametrize("fallback", [False, True])
def test_prepare_keeps_windows_and_delivery_receipts(tmp_path, monkeypatch, fallback):
    from kernel_request_diet import RequestDietDeferred
    import kernel_request_diet as diet

    real_apply = diet.apply_diet

    def prepare(mode):
        runtime, messages, tools, receipts = _field_fixture(tmp_path)
        with monkeypatch.context() as patch:
            _switches(patch, mode)
            called = False

            def apply(*args, **kwargs):
                nonlocal called
                if fallback and not called:
                    called = True
                    raise RequestDietDeferred({"reason": "fixture requests hard-budget recovery"})
                return real_apply(*args, **kwargs)

            patch.setattr(diet, "apply_diet", apply)
            wire, capture = _prepare(runtime, messages, tools)
        assert len(_acquisition(receipts)) == (4 if fallback else 3)
        return wire, capture, receipts

    expected = prepare("off")
    actual = prepare("on")
    _assert_prepare_parity(expected, actual)


@pytest.mark.parametrize("changed", ["target", "negative dependency", "resolver", "schema", "budget"])
def test_changed_inputs_match_fresh_fitting(tmp_path, monkeypatch, changed):
    def prepare(mode):
        runtime, messages, tools, events = _field_fixture(tmp_path / mode)
        real_bind = preflight.bound_runtime_sources
        pass_number = 0

        def bind(content, current, **options):
            nonlocal pass_number
            pass_number += 1
            if pass_number == 2:
                root = current.owner.root
                if changed == "target":
                    old = current.target.stat()
                    source = current.target.read_text()
                    current.target.write_text(source.replace("return 0", "return 9"))
                    os.utime(current.target, ns=(old.st_atime_ns, old.st_mtime_ns))
                if changed == "negative dependency":
                    (root / "optional_provider_1.py").write_text("VALUE = 1\n")
                if changed == "resolver":
                    (root / "optional_provider_1.local").write_text("VALUE = 1\n")
                    patch.setattr(closure, "_SOURCE_EXTENSIONS",
                                  (*closure._SOURCE_EXTENSIONS, ".local"))
                if changed == "schema":
                    # Grow the menu into the range where the target window has
                    # to fit a different space even at the same stage ceiling.
                    options["tools"][0]["function"]["description"] = "schema details " * 4500
            if pass_number >= 2 and changed == "budget":
                options["max_request_bytes"] = 9_000
            return real_bind(content, current, **options)

        with monkeypatch.context() as patch:
            _switches(patch, mode)
            patch.setattr(preflight, "bound_runtime_sources", bind)
            wire, capture = _prepare(runtime, messages, tools)
        return wire, capture, events

    expected = prepare("off")
    actual = prepare("on")
    _assert_prepare_parity(expected, actual)
    first, second, *_ = _acquisition(actual[2])
    old_plan, new_plan = first["closure"], second["closure"]
    if changed == "target":
        assert old_plan["revisions"]["module_00.py"] != new_plan["revisions"]["module_00.py"]
        old_receipt = window.verify_final_source_delivery(
            copy.deepcopy(old_plan), actual[0]["messages"], payload=actual[0],
            sources=new_plan["source_contents"],
        )
        assert not window.source_delivery_satisfied(old_receipt)
    if changed in {"negative dependency", "resolver"}:
        assert old_plan["dependency_paths"]["module_01.py"] != new_plan["dependency_paths"]["module_01.py"]
    if changed in {"schema", "budget"}:
        assert old_plan["revisions"] == new_plan["revisions"]
        assert first["source_reservation"] != second["source_reservation"]

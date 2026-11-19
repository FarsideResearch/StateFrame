"""Decision-panel work and byte parity at the live runtime boundaries."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import kernel_decision_cache as decisions
import pytest
from kernel_stateframe_runtime import StateFrameRuntime
from test_turn_panels_budget import Fixture


FILES = 4_500
RECEIPTS = 200
MENU = [{"type": "function", "function": {"name": "read"}}]
DRAFT = {"op": "ADOPT", "draft": {
    "key": {"subject": "file_0000.py::answer", "question": "BEHAVIOR_OWNER"},
    "selected": {"type": "CLAUSE_INTERPRETATION"},
    "basis_refs": ["read-0"],
}}


def _wire(frame, panel, menu):
    return json.dumps([frame, panel, menu], ensure_ascii=False).encode()


def _runtime(root, paths):
    # Reuse the minimal real-frame fixture, keeping unrelated organs inert.
    fixture = Fixture.__new__(Fixture)
    fixture.target = root / paths[0]
    fixture.counts = Counter()
    fixture.history = []
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.__dict__.update(vars(fixture.runtime()))
    runtime.owner.root = root
    runtime.owner.task_row = {"files": paths, "msg": "Keep the answer consistent."}
    runtime.owner.fn = "answer"
    runtime.owner.current_step = 1
    runtime.owner._receipts.append(_receipt(0))
    runtime._target_source = lambda: runtime.target.read_text(encoding="utf-8")
    runtime.projector_kind = "legacy"
    runtime.journal.task_id = "decision-panel"
    runtime.journal.path = root.parent / "journal.jsonl"
    runtime.record_timeline_event = lambda *args, **kwargs: None
    runtime._record_multi_definition_open_work = None
    runtime._action_before = None
    runtime.current_action = None
    runtime.binding = lambda: {"product_revision": "product"}
    runtime._ripple_preview_after_apply = lambda receipt: None
    runtime.actions = SimpleNamespace(append=lambda **kwargs: {}, diagnose=dict)
    runtime.native_batch = SimpleNamespace(pending=True)
    runtime.memory_isolation = runtime.self_plan_state = runtime.unified_lifecycle = None
    runtime._active_capture = runtime._last_capture = runtime.continuation = None
    runtime.latest_exchange_mode = "off"
    runtime.endpoint_profile = None
    runtime._record_passive_retention_capture = lambda *args: None
    runtime._failure_repair_session = runtime._auto_verify_delivery = None
    runtime.close_step = lambda *args: None
    runtime._prepare_continuation_boundary = lambda *args: None
    runtime._sync_work_carry = runtime._configure_probe_gate = lambda *args: None
    runtime._bind_long_horizon_capture = lambda *args, **kwargs: None
    runtime.decision_cache = decisions.DecisionCacheStore(**runtime._decision_cache_context())
    frame, _, _ = runtime._frame()
    assert runtime.decision_cache.ingest(
        DRAFT, input_receipt_refs={"read-0"})["accepted"]
    return runtime


def _receipt(index):
    return {"kind": "read", "receipt_id": f"read-{index}", "result_class": "OK",
            "detail": "Read résumé", "turn": index, "attempt_id": None}


def _project(runtime):
    frame, _, _ = runtime._frame()
    return _wire(frame, runtime._settled_state_message(frame), decisions.tool_surface(MENU))


def _run(runtime, count):
    for index in range(1, count):
        receipt = _receipt(index)
        runtime.owner._receipts.append(receipt)
        runtime.completed_receipt(receipt, receipt["detail"])
    queued = runtime.prepare_queued_native_action([], uncapped=[], tools=MENU)
    assert not queued.messages and list(queued.tools) == MENU
    return _project(runtime), _project(runtime)


def _switches(monkeypatch, mode):
    monkeypatch.setenv("STATEFRAME_FRAME_EDIT_INTEGRITY", "off")
    monkeypatch.setenv("STATEFRAME_DECISION_CACHE", "on")
    monkeypatch.setenv("STATEFRAME_CONCLUSION_CACHE", "off")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", mode)
    for name in ("PANELS", "INDEXED_READS", "DELIVERY_PROOFS", "BOUNDED_QUALIFICATION",
                 "GENERATION_OVERLAP", "MINIMAL_LAUNCH", "REPO_INTEL", "ATOMIC_PUBLICATION", "TIMING"):
        monkeypatch.setenv("STATEFRAME_TURN_" + name, mode)


def test_large_receipt_and_prepare_paths_reuse_live_decision_panel(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    paths = tuple(f"file_{index:04}.py" for index in range(FILES))
    for path in paths:
        (root / path).write_text("def answer():\n    return 'café'\n", encoding="utf-8")
    _switches(monkeypatch, "on")
    runtime = _runtime(root, paths)
    _project(runtime)  # Admit the first unchanged render.
    counts = Counter()
    read_text = Path.read_text
    render = decisions._render_settled_state_r43
    dependencies = decisions._dependency_rows

    def count_read(path, *args, **kwargs):
        if path.parent == root and path.name != paths[0]:
            counts["unrelated_source_reads"] += 1
        return read_text(path, *args, **kwargs)

    def count_render(*args, **kwargs):
        counts["full_panel_renders"] += 1
        return render(*args, **kwargs)

    def count_dependencies(*args, **kwargs):
        counts["dependency_projections"] += 1
        return dependencies(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", count_read)
        patch.setattr(decisions, "_render_settled_state_r43", count_render)
        patch.setattr(decisions, "_dependency_rows", count_dependencies)
        started = perf_counter()
        optimized = _run(runtime, RECEIPTS)
        elapsed = perf_counter() - started
    print(f"{FILES} files / {RECEIPTS} receipts: {elapsed:.6f}s {dict(counts)}")
    assert counts == Counter(), f"{dict(counts)} in {elapsed:.6f}s"
    assert elapsed < 10

    _switches(monkeypatch, "off")
    control = _runtime(root, paths[:1])
    _project(control)
    assert optimized == _run(control, RECEIPTS)


def test_edit_reprojects_selected_slots_and_stale_basis_cannot_render(tmp_path, monkeypatch):
    from kernel_turn_contracts import Stale
    from kernel_turn_panels import ProjectionFacts, slot_key

    _switches(monkeypatch, "on")
    first, other = "first.py", "other.py"
    (tmp_path / first).write_text("def answer():\n    return 1\n")
    (tmp_path / other).write_text("def spare():\n    return 1\n")
    runtime = _runtime(tmp_path, (first, other))
    _project(runtime)
    looked_up = []
    lookup = ProjectionFacts.lookup

    def record_lookup(facts, at, key, cursor, limit):
        looked_up.append(key)
        return lookup(facts, at, key, cursor, limit)

    monkeypatch.setattr(ProjectionFacts, "lookup", record_lookup)
    (tmp_path / other).write_text("def spare():\n    return 2\n")
    _project(runtime)
    assert looked_up == []

    (tmp_path / first).write_text("def answer():\n    return 2\n")
    queued = runtime.prepare_queued_native_action([], uncapped=[], tools=MENU)
    assert not queued.messages
    assert looked_up == [slot_key("frame", "live")]
    frame, _, _ = runtime._frame()
    assert frame["decision_cache"]["records"][0]["basis_binding"] == "STALE"
    runtime._settled_state_message(frame)
    assert looked_up == [slot_key("frame", "live"), slot_key("settled", "live")]

    runtime.context_scope = "changed policy"
    with pytest.raises(Stale, match="authority"):
        runtime._settled_state_message(frame)

    frame, _, _ = runtime._frame()
    runtime.owner._receipts[0]["detail"] = "Amended evidence"
    with pytest.raises(Stale, match="authority"):
        runtime._settled_state_message(frame)
    frame, _, _ = runtime._frame()
    runtime._settled_state_message(frame)


@pytest.mark.parametrize("extension", ["off", "on"])
def test_frozen_sidecar_rejects_changed_source_or_evidence(tmp_path, monkeypatch, extension):
    _switches(monkeypatch, "on")
    monkeypatch.setenv("STATEFRAME_CONCLUSION_CACHE", extension)
    first = tmp_path / "a.py"
    first.write_text("def answer():\n    return 1\n")
    runtime = _runtime(tmp_path, (first.name,))
    runtime.conclusion_cache_mode = extension

    frame, _, _ = runtime._frame()
    clean = runtime.decision_cache.ingest(
        DRAFT, input_receipt_refs={"read-0"}, context=runtime._decision_cache_input_context)
    assert clean["accepted"]
    assert runtime._settled_state_message(frame) is not None

    runtime._frame()
    runtime.owner._receipts[0]["detail"] = "Amended evidence"
    rejected = runtime.decision_cache.ingest(
        DRAFT, input_receipt_refs={"read-0"}, context=runtime._decision_cache_input_context)
    assert not rejected["accepted"] and "receipt basis changed" in rejected["reason"]

    runtime._frame()
    first.write_text("def answer():\n    return 2\n")
    rejected = runtime.decision_cache.ingest(
        DRAFT, input_receipt_refs={"read-0"}, context=runtime._decision_cache_input_context)
    assert not rejected["accepted"] and "source basis changed" in rejected["reason"]


def test_projection_rolls_back_if_authority_changes_after_fold(tmp_path, monkeypatch):
    from kernel_turn_contracts import Stale

    _switches(monkeypatch, "on")
    first = tmp_path / "a.py"
    first.write_text("def answer():\n    return 1\n")
    runtime = _runtime(tmp_path, (first.name,))
    _project(runtime)
    panels = runtime._live_decision_panels
    previous = panels.publication.head()
    original_render = panels.panels.render
    frame, _, _ = runtime._frame()
    # Change only the settled recipe. The frame root should remain published.
    frame["open_work_ids"] = ["check:finish"]

    def change_after_fold(*args):
        value = original_render(*args)
        runtime.owner._receipts[0]["detail"] = "Concurrent amendment"
        return value

    with monkeypatch.context() as patch:
        patch.setattr(panels.panels, "render", change_after_fold)
        with pytest.raises(Stale, match="during projection"):
            runtime._settled_state_message(frame)
    assert panels.publication.head() == previous
    frame, _, _ = runtime._frame()
    assert runtime._settled_state_message(frame) is not None


def test_provider_prepare_reaches_live_panel_projection(tmp_path, monkeypatch):
    """Exercise the real prepare prefix and stop at the measured projection boundary."""
    _switches(monkeypatch, "on")
    first = tmp_path / "a.py"
    first.write_text("def answer():\n    return 1\n")
    runtime = _runtime(tmp_path, (first.name,))
    runtime._history_state_tail_active = lambda provider_request: False

    class ReachedProjection(Exception):
        pass

    def project():
        frame, _, _ = StateFrameRuntime._frame(runtime)
        assert runtime._settled_state_message(frame) is not None
        raise ReachedProjection

    runtime._frame = project
    with pytest.raises(ReachedProjection):
        runtime.prepare([], uncapped=[], tools=MENU, provider_request=False)


def test_panel_checks_workspace_fence_even_when_point_evidence_is_unchanged(tmp_path, monkeypatch):
    from kernel_turn_contracts import Stale
    from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
    from kernel_workspace_feed import WorkspaceFeed
    from kernel_workspace_state import WorkspaceFence, WorkspaceState

    _switches(monkeypatch, "on")
    root = tmp_path / "workspace"
    root.mkdir()
    target, other = "target.py", "other.py"
    (root / target).write_text("def answer():\n    return 1\n")
    (root / other).write_text("def spare():\n    return 1\n")
    feed = WorkspaceFeed(root, enabled=True)
    state = WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence"), feed=feed)
    consumers = WorkspaceConsumers(state)
    bind_consumers(root, consumers)
    try:
        runtime = _runtime(root, (target, other))
        frame, _, _ = runtime._frame()
        assert runtime._settled_state_message(frame)
        with state.begin_write((other,)):
            (root / other).write_text("def spare():\n    return 2\n")
        with pytest.raises(Stale, match="authority"):
            runtime._settled_state_message(frame)
        assert _project(runtime)
    finally:
        feed.close()

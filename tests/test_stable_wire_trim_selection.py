"""Real allocator trims must select the same candidates with different replay meters."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

import kernel_cache_layout_candidate as layout
import kernel_frame_allocation_capture as allocation
import kernel_request_budget as budget
from kernel_request_diet import RequestTokenMeter, _tokenization_content
from kernel_stable_wire import is_stable_wire_projection, recorded_window_layout
from test_final_cache_layout import _body
from test_stable_wire import Journal, _window


@pytest.fixture
def allocator(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    monkeypatch.setenv("STATEFRAME_PAGE_LOG", "off")
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE_HEADROOM", "16000")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    for name in ("_REPLAY_LAYOUT_DECISION_PROVIDER", "_REPLAY_STABLE_WIRE_DECISION_PROVIDER",
                 "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER", "_REPLAY_STABLE_WIRE_RESERVES_PROVIDER"):
        monkeypatch.setattr(layout, name, None)
    # Supply host/source authority, but leave the allocator and optional trim real.
    patches = {
        "kernel_execution_reply_budget.fit_execution_replies": lambda *a, **k: [],
        "kernel_execution_reply_budget.project_execution_delivery": lambda semantic, *a, **k: semantic,
        "kernel_stateframe_capacity.fit_protected_frame": lambda *a, **k: {"changed": False},
        "kernel_stateframe_capacity.fit_protected_source_overhead": lambda *a, **k: None,
        "kernel_stateframe_capacity.within_protected_capacity": lambda *a, **k: True,
        "kernel_frame_optional_allocation.snapshot_optional_inputs": lambda *a: ({}, {}),
        "kernel_stateframe_framefit5._workspace_root": lambda *a: tmp_path,
        "kernel_frame_source_snapshot.recent_editable_read_windows": lambda *a: [],
        "kernel_frame_source_snapshot.public_check_source_windows": lambda *a: [],
        "kernel_frame_source_snapshot.editable_product_windows": lambda *a: ([], []),
        "kernel_frame_source_snapshot.frame_source_snapshot": lambda *a: {},
        "kernel_code_bookmarks.resolve": lambda *a: None,
        "kernel_code_bookmarks.windows": lambda *a: [],
        "kernel_code_bookmarks.enabled": lambda: False,
        "kernel_completion_owner._current_tool_admission": lambda *a: (False, None),
        "kernel_request_source_window.verify_final_source_delivery": lambda *a, **k: {"status": "COMPLETE"},
        "kernel_request_source_window.source_delivery_satisfied": lambda receipt: True,
        "kernel_step_contract.reconcile_final_source_access": lambda *a: False,
    }
    for target, fn in patches.items():
        monkeypatch.setattr(target, fn)
    monkeypatch.setattr(budget, "ENDPOINT_PROMPT_TOKEN_BUDGET", 33344)
    monkeypatch.setattr(budget, "_lean_request_limits", lambda *a, **k: (533504, 33344))
    body = _body()
    body["messages"][3] = _window()
    import kernel_stateframe_framefit5 as framefit5
    row = body["messages"][2]
    payload = json.loads(row["content"].split("\n", 1)[1])
    payload["state"]["framefit5"]["mode"] = framefit5.PRODUCT_MODE
    row["content"] = framefit5._serialize_state_message(payload)
    body["messages"].insert(2, {"role": "user", "_request_budget_section": "optional",
                                  "content": "[Optional context] "})
    body["messages"][2]["content"] += "x" * (5754 - len(json.dumps(body).encode()))
    assert len(json.dumps(body).encode()) == 5754
    length = len(_tokenization_content(body))

    def prepare(replay=False, *, token_delta=None, exchange=False, marker=None):
        request_body = copy.deepcopy(body)
        if marker is not None:
            request_body["messages"].insert(3, copy.deepcopy(marker))
        request_length = len(_tokenization_content(request_body)) if marker is not None else length
        if exchange:
            request_body["messages"][-1:-1] = [
                row for row in _body(exchange=True)["messages"] if row["role"] in {"assistant", "tool"}]
        meter = RequestTokenMeter(
            tokenizer=(lambda text: 33320 + token_delta + round((len(text) - request_length) / 10))
                      if token_delta is not None else (lambda text: 100) if replay else
                      (lambda text: 33320 + round((len(text) - request_length) / 10)),
            template_overhead_tokens=0)
        capture = SimpleNamespace(
            request_id="trim-selection", step=1, journal=Journal(),
            _framefit5_semantic_messages=copy.deepcopy(request_body["messages"]),
            _framefit5_task_message=request_body["messages"][1],
            _request_source_closure_override={"units": [], "paths": [], "revisions": {},
                                              "deferred_paths": [], "source_contents": {}},
            request_source_final_delivery=None)
        runtime = SimpleNamespace(frame_ceiling=8192, owner=SimpleNamespace(root=tmp_path),
                                  _code_bookmark_carry=(), _empirical_frame_sources={},
                                  _framefit_action_context={}, _stable_wire_state=None)
        meter.bind_request(capture.request_id, capture.request_id)
        monkeypatch.setattr(budget, "_request_token_meter", lambda *a, **k: meter)
        fitted = allocation.finalize_frame_candidate(runtime, capture, request_body, frame_counter=lambda b: 1)
        return fitted, capture, runtime, meter

    def render(prepared):
        fitted, capture, runtime, meter = prepared
        return layout.render_final_cache_candidate(capture, runtime, fitted, meter=meter,
                                                   inputs=capture._frame_candidate_inputs)

    return prepare, render


def test_initial_allocation_replays_with_thirty_more_tokens(monkeypatch, allocator):
    prepare, render = allocator
    prepared = prepare()
    live = render(prepared)
    record = prepared[1]._stable_wire_reserves_record
    assert record["reserves"] == [[70, 256]]
    _install_replay(monkeypatch, live, record)
    replay = prepare(token_delta=30)
    rebuilt = render(replay)
    assert json.dumps(rebuilt).encode() == json.dumps(live).encode()
    assert replay[1]._stable_wire_reserves_record == record


def test_initial_removal_replays_even_without_a_final_trim(monkeypatch, allocator):
    from kernel_frame_optional_allocation import identity

    prepare, render = allocator
    prepared = prepare(token_delta=30)
    live = render(prepared)
    assert is_stable_wire_projection(live["messages"])
    record = prepared[1]._stable_wire_reserves_record
    assert record["reserves"] == []
    assert record["initial_selection"]["removed"] == [
        [2, identity(prepared[1]._framefit5_semantic_messages[2])]]
    _install_replay(monkeypatch, live, record)
    replay = prepare(replay=True)
    rebuilt = render(replay)
    assert json.dumps(rebuilt).encode() == json.dumps(live).encode()
    assert replay[1]._stable_wire_reserves_record == record


def test_initial_selection_never_measures_during_replay(monkeypatch, allocator):
    prepare, render = allocator
    prepared = prepare()
    live = render(prepared)
    record = prepared[1]._stable_wire_reserves_record
    _install_replay(monkeypatch, live, record)

    def forbidden(*args, **kwargs):
        pytest.fail("initial selection replay consulted a current measurement")

    monkeypatch.setattr(allocation, "final_request_tokens", forbidden)
    monkeypatch.setattr(allocation, "_measure_source_candidate", forbidden)
    with monkeypatch.context() as initial_fit:
        initial_fit.setattr(RequestTokenMeter, "measure", forbidden)
        initial_fit.setattr(RequestTokenMeter, "_measure_with_wire_bytes", forbidden)
        replay = prepare(token_delta=30)
    rebuilt = render(replay)
    assert json.dumps(rebuilt).encode() == json.dumps(live).encode()


def test_initial_source_selection_replays_without_measuring(monkeypatch, allocator):
    prepare, render = allocator
    source = _window(path="pkg/b.py", text="other = 2\n")
    window = {"path": "pkg/b.py", "content": source["content"], "source_chars": 10,
              "start_line": 1, "end_line": 1}
    monkeypatch.setattr("kernel_frame_source_snapshot.recent_editable_read_windows",
                        lambda *a: [copy.deepcopy(window)])
    prepared = prepare(replay=True, exchange=True)
    assert any(row.get("content") == source["content"] for row in prepared[0]["messages"])
    live = render(prepared)
    assert is_stable_wire_projection(live["messages"])
    record = prepared[1]._stable_wire_reserves_record
    assert record["initial_selection"]["input_id"] != record["initial_selection"]["output_id"]
    assert any(row[0] == 3 for row in record["initial_selection"]["decisions"])
    _install_replay(monkeypatch, live, record)

    def forbidden(*args, **kwargs):
        pytest.fail("initial source selection consulted a current measurement")

    with monkeypatch.context() as initial_fit:
        initial_fit.setattr(allocation, "final_request_tokens", forbidden)
        initial_fit.setattr(allocation, "_measure_source_candidate", forbidden)
        initial_fit.setattr(RequestTokenMeter, "measure", forbidden)
        initial_fit.setattr(RequestTokenMeter, "_measure_with_wire_bytes", forbidden)
        replay = prepare(token_delta=30, exchange=True)
    rebuilt = render(replay)
    assert json.dumps(rebuilt).encode() == json.dumps(live).encode()
    assert replay[1]._stable_wire_reserves_record == record
    serialized = json.dumps(record["initial_selection"])
    assert "pkg/b.py" not in serialized
    assert "other = 2" not in serialized


@pytest.mark.parametrize("field", ["input_id", "output_id", "kept", "removed", "decisions", "counter_id"])
def test_unknown_initial_candidate_fails_open(monkeypatch, allocator, field):
    prepare, render = allocator
    prepared = prepare()
    live = render(prepared)
    record = copy.deepcopy(prepared[1]._stable_wire_reserves_record)
    initial = record["initial_selection"]
    unknown = "sha256:" + "0" * 64
    if field in {"input_id", "output_id"}:
        initial[field] = unknown
    elif field == "removed":
        initial[field] = [[999, unknown]]
    elif field == "counter_id":
        counter = next(row[3] for row in initial["decisions"] if row[3])
        counter[4] = unknown
    else:
        initial[field][0][1] = unknown
    _install_replay(monkeypatch, live, record)
    replay = prepare(token_delta=30)
    rebuilt = render(replay)
    assert not is_stable_wire_projection(rebuilt["messages"])
    assert replay[2]._stable_wire_state is None
    assert replay[1]._stable_wire_reserves_record is None
    assert len([event for event in replay[1].journal.events
                if event["kind"] == "stable_wire_replay_mismatch"]) == 1


def test_old_record_without_initial_selection_keeps_meter_driven_allocation(monkeypatch, allocator):
    prepare, render = allocator
    prepared = prepare()
    live = render(prepared)
    record = copy.deepcopy(prepared[1]._stable_wire_reserves_record)
    del record["initial_selection"]
    _install_replay(monkeypatch, live, record)
    replay = prepare(token_delta=30)
    assert not any("[Optional context]" in row.get("content", "") for row in replay[0]["messages"])
    rebuilt = render(replay)
    assert not is_stable_wire_projection(rebuilt["messages"])
    assert [event for event in replay[1].journal.events if event["kind"] == "stable_wire_replay_mismatch"]


def _install_replay(monkeypatch, live, record):
    monkeypatch.setattr(layout, "_REPLAY_LAYOUT_DECISION_PROVIDER", lambda: "layout")
    monkeypatch.setattr(layout, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", lambda: "applied")
    monkeypatch.setattr(layout, "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER", lambda: recorded_window_layout(live["messages"]))
    monkeypatch.setattr(layout, "_REPLAY_STABLE_WIRE_RESERVES_PROVIDER", lambda: copy.deepcopy(record))


def test_real_allocator_replays_selection_without_measuring(monkeypatch, allocator):
    prepare, render = allocator
    live_prepared = prepare()
    starting = json.dumps(live_prepared[0]).encode()
    assert len(starting) == 5754
    live = render(live_prepared)
    record = live_prepared[1]._stable_wire_reserves_record
    assert record["reserves"] == [[70, 256]]
    assert is_stable_wire_projection(live["messages"])
    assert not any("[Optional context]" in row.get("content", "") for row in live["messages"])
    _install_replay(monkeypatch, live, record)
    replay = prepare(replay=True)
    assert json.dumps(replay[0]).encode() == starting
    def forbidden_measurement(*args, **kwargs):
        pytest.fail("selection replay consulted allocator token measurements")

    monkeypatch.setattr(allocation, "final_request_tokens", forbidden_measurement)
    rebuilt = render(replay)
    assert json.dumps(rebuilt).encode() == json.dumps(live).encode()
    assert replay[1]._stable_wire_reserves_record == record
    from kernel_frame_optional_allocation import identity
    from kernel_request_measurement import digest
    assert record["selections"] == [[{"kind": "message", "index": 2,
                                      "candidate_id": identity(live_prepared[1]._framefit5_semantic_messages[2])}]]
    assert replay[1]._frame_candidate_measurement.body_sha256 == digest(json.dumps(replay[0]).encode())


@pytest.mark.parametrize("unknown_index", [2, 999999])
def test_unknown_recorded_candidate_fails_open(monkeypatch, allocator, unknown_index):
    prepare, render = allocator
    prepared = prepare()
    live = render(prepared)
    record = copy.deepcopy(prepared[1]._stable_wire_reserves_record)
    record["selections"] = [[{"kind": "message", "index": unknown_index, "candidate_id": "sha256:" + "0" * 64}]]
    _install_replay(monkeypatch, live, record)
    replay = prepare(replay=True)
    rebuilt = render(replay)
    assert not is_stable_wire_projection(rebuilt["messages"])
    assert replay[2]._stable_wire_state is None
    assert replay[1]._stable_wire_reserves_record is None
    assert [event for event in replay[1].journal.events if event["kind"] == "stable_wire_replay_mismatch"]
    assert any("[Optional context]" in row.get("content", "") for row in rebuilt["messages"])


def test_old_record_without_selections_uses_reserves_and_final_check(monkeypatch, allocator):
    prepare, render = allocator
    prepared = prepare()
    live = render(prepared)
    record = {key: value for key, value in prepared[1]._stable_wire_reserves_record.items()
              if key not in {"selections", "initial_selection"}}
    _install_replay(monkeypatch, live, record)
    replay = prepare(replay=True)
    rebuilt = render(replay)
    # Legacy authority does not know which optional row was removed. Its windows
    # and epoch still pass; the unchanged transport comparison detects the bytes.
    assert is_stable_wire_projection(rebuilt["messages"])
    assert any("[Optional context]" in row.get("content", "") for row in rebuilt["messages"])
    assert json.dumps(rebuilt).encode() != json.dumps(live).encode()


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("token_delta", [None, 30])
def test_real_selection_survives_durable_transport_and_pending_replay(
        monkeypatch, tmp_path, allocator, pending, token_delta):
    import kernel_bounded_http
    from kernel_measurement import ACTIVE_REQUEST
    from stateframe import _child
    from stateframe.replay_tape import Replayer, build, pending_request
    from test_stable_wire_record import ENDPOINT, RESPONSE

    prepare, render = allocator
    prepared = prepare()
    live = render(prepared)
    wire = json.dumps(live).encode()
    record = prepared[1]._stable_wire_reserves_record
    run = tmp_path / "recorded"
    run.mkdir()
    monkeypatch.setattr(_child, "HOST_STATE", {})
    monkeypatch.setattr(_child, "emit", lambda *a, **k: None)

    class Interrupted(BaseException):
        pass

    def transport(sent, *args, **kwargs):
        assert sent == wire
        assert json.loads((run / "wire/0001.stable-wire.json").read_bytes()) == record
        assert (run / "wire/0001.request.json").read_bytes() == wire
        assert not (run / "sends.jsonl").exists()
        if pending:
            raise Interrupted
        return RESPONSE

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", transport)
    _child.install_wire_recorder(run, ENDPOINT)
    active = ACTIVE_REQUEST.set(prepared[1])
    try:
        if pending:
            with pytest.raises(Interrupted):
                kernel_bounded_http.request_bytes(wire, ENDPOINT, "")
        else:
            assert kernel_bounded_http.request_bytes(wire, ENDPOINT, "") == RESPONSE
    finally:
        ACTIVE_REQUEST.reset(active)
    replayer = Replayer(build(run / "wire", run / "sends.jsonl"),
                        pending=pending_request(run / "wire"))
    assert replayer.reserves_for_layout_decision() == record
    detached = replayer.reserves_for_layout_decision()
    detached["selections"][0][0]["index"] = 999999
    assert replayer.reserves_for_layout_decision() == record
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", lambda *a, **k: RESPONSE)
    _child.install_wire_recorder(resumed, ENDPOINT, replayer)
    replay = prepare(replay=True, token_delta=token_delta)
    rebuilt = render(replay)
    active = ACTIVE_REQUEST.set(replay[1])
    try:
        assert kernel_bounded_http.request_bytes(json.dumps(rebuilt).encode(), ENDPOINT, "") == RESPONSE
    finally:
        ACTIVE_REQUEST.reset(active)
    assert replayer.diverged is None
    assert replayer.pending is None
    assert json.loads((resumed / "wire/0001.stable-wire.json").read_bytes()) == record
    assert json.loads((resumed / "sends.jsonl").read_bytes())["stable_wire_reserves"] == record


@pytest.mark.parametrize("bad_selection", [
    None, [], [None], [[{"kind": "message", "index": True, "candidate_id": "sha256:" + "0" * 64}]],
    [[{"kind": "message", "index": -1, "candidate_id": "sha256:" + "0" * 64}]],
    [[{"kind": "message", "index": 0, "candidate_id": "secret content"}]],
    [[{"kind": "unknown", "index": 0, "candidate_id": "sha256:" + "0" * 64}]],
    [[{"kind": "message", "index": 0, "candidate_id": "sha256:" + "0" * 64}] * 2],
])
def test_selection_record_rejects_malformed_or_content_fields(bad_selection):
    from kernel_stable_wire_record import validate_reserves_record

    with pytest.raises(ValueError):
        validate_reserves_record({"reserves": [[70, 256]], "trim_count": 1, "epoch": 1,
                                  "selections": bad_selection})


def test_selection_is_atomic_and_addresses_duplicate_rows_by_index():
    from kernel_frame_optional_allocation import identity
    from kernel_stable_wire_selection import apply_selection, removed_candidates

    row = {"role": "user", "content": "optional"}
    before = {"messages": [row, copy.deepcopy(row), {"role": "user", "content": "tail"}]}
    after = {"messages": [copy.deepcopy(row), {"role": "user", "content": "tail"}]}
    selected = removed_candidates(before, after)
    assert selected == [{"kind": "message", "index": 1, "candidate_id": identity(row)}]
    body = copy.deepcopy(before)
    bad = selected + [{"kind": "message", "index": 99, "candidate_id": identity(row)}]
    assert apply_selection(body, bad) is False
    assert body == before
    assert apply_selection(body, selected) is True
    assert body == after


def test_selection_never_advertises_unrecorded_rewrites():
    from kernel_stable_wire_selection import removed_candidates

    before = {"messages": [{"role": "user", "content": "optional"},
                           {"role": "user", "content": "protected history"}], "tools": []}
    after = {"messages": [{"role": "user", "content": "rewritten history"}], "tools": []}
    assert removed_candidates(before, after) is None
    assert removed_candidates(before, {**before, "tools": [{}]}) is None


def test_live_trim_fails_open_if_removals_do_not_capture_all_effects(monkeypatch, allocator):
    prepare, render = allocator
    prepared = prepare()
    original = copy.deepcopy(prepared[0])

    def rewrite_history(body, **kwargs):
        body["messages"][-1]["content"] += " changed history"

    monkeypatch.setattr("kernel_stateframe_capacity.fit_protected_source_overhead", rewrite_history)
    result = render(prepared)
    assert not is_stable_wire_projection(result["messages"])
    assert prepared[0] == original
    assert prepared[1]._stable_wire_reserves_record is None
    assert prepared[2]._stable_wire_state is None


def test_selection_refuses_serialization_order_changes():
    from kernel_stable_wire_selection import removed_candidates

    before = {"messages": [{"role": "user", "content": "kept"}]}
    after = {"messages": [{"content": "kept", "role": "user"}]}
    assert before == after
    assert json.dumps(before) != json.dumps(after)
    assert removed_candidates(before, after) is None


def test_allocator_without_trims_records_initial_authority(allocator):
    prepare, render = allocator
    prepared = prepare(replay=True)
    result = render(prepared)
    assert is_stable_wire_projection(result["messages"])
    assert prepared[1]._stable_wire_reserves_record == {
        "reserves": [], "trim_count": 0, "epoch": 1,
        "initial_selection": prepared[1]._stable_wire_initial_selection}

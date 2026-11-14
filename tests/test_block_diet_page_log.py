"""Read-fact wire projection composes with typed frame encodings."""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from _journal import measurement_events
from engine_run import run_monitored
from kernel_cache_layout_view import semantic_messages
from kernel_stateframe_envelope import serialize_stateframe_payload
from stateframe.scripted_endpoint import Script, serve

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src/stateframe/engine/kernel_block_diet.py").is_file():
    pytest.skip("block-diet module is absent from the rebuilt 0.6.34 tip", allow_module_level=True)


def _stable_projection(monkeypatch, body):
    from types import SimpleNamespace

    from test_stable_wire import _invoke, _window

    body = copy.deepcopy(body)
    for index, row in enumerate(body["messages"]):
        if row.get("content", "").startswith("[Current source:"):
            body["messages"][index] = _window()
    projected, capture, _journal = _invoke(
        monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    assert capture._final_cache_layout_record["stable_wire"] == "applied"
    # Exercise recorded ownership, including its positional status carrier.
    recorded = json.loads(json.dumps(projected))
    assert semantic_messages(recorded["messages"]) == semantic_messages(body["messages"])
    return recorded


@pytest.mark.host_sandbox
def test_page_log_multi_turn_qualification_and_preflight(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    document = json.loads((ROOT / "examples/discount_bug.script.json").read_text())
    # Retain a nonempty base across later read-fact updates, then edit and test.
    document["steps"][1:1] = [copy.deepcopy(document["steps"][0]) for _ in range(3)]
    profile = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text())
    profile["switches"].update(STATEFRAME_PAGE_LOG="on", STATEFRAME_BLOCK_DIET="on")
    profile_path = tmp_path / "page-log-profile.json"
    profile_path.write_text(json.dumps(profile))
    script = Script(document)
    server = serve(script)
    run = tmp_path / "run"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("STATEFRAME_", "REACT_"))}
    environment.update(PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                       STATEFRAME_HOME=str(tmp_path / "home"), TMPDIR=str(tmp_path))
    try:
        process = run_monitored([
            sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", "Fix the pricing code so the tests pass.",
            "--profile", str(profile_path),
            "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
            # Exercise sealed requests and sandboxed checks without cloning the
            # host dependency environment, which is unrelated to this projection.
            "--no-run-env",
            "--run-dir", str(run), "--test-python", sys.executable, "--max-turns", "16", "--quiet",
        ], cwd=ROOT, env=environment, capture_output=True, text=True,
            expected_seconds=240, run_dir=run, check=False)
    finally:
        server.shutdown()
        server.server_close()
    traceback = run / "child-traceback.txt"
    output = process.stdout + process.stderr
    if traceback.exists():
        output += traceback.read_text()
    assert process.returncode == 0, output
    effective = json.loads((run / "spec.json").read_text())["environment"]
    assert effective["STATEFRAME_BLOCK_DIET"] == "on"
    assert effective["STATEFRAME_PAGE_LOG"] == "on"
    script.assert_consumed()
    events = list(measurement_events(run))
    assert sum(row.get("kind") == "qualified_request_sealed" for row in events) >= 4
    assert not [row for row in events if row.get("kind") == "stateframe_page_log_failed_open"]
    assert not [row for row in events if row.get("kind") == "framefit5_request_preflight_failed"]
    chains = []
    for request in sorted((run / "wire").glob("*.request.json")):
        messages = json.loads(request.read_bytes())["messages"]
        base = next((json.loads(row["content"].split("\n", 1)[1]) for row in messages
                     if str(row.get("content", "")).startswith("[StateFrame]\n")), None)
        if base is None:
            continue
        deltas = [row for row in messages if row.get("name") == "stateframe_frame_delta"]
        if base["state"]["framefit5"]["read_facts"] and deltas:
            chains.append(messages)
        for row in semantic_messages(messages):
            if str(row.get("content", "")).startswith("[StateFrame]\n"):
                facts = json.loads(row["content"].split("\n", 1)[1])["state"]["framefit5"]["read_facts"]
                assert all(set(fact) <= {"path", "status", "receipts", "lines"} for fact in facts)
    assert chains, "must exercise deltas superseding nonempty base read facts"


@pytest.mark.parametrize("initial", [False, True])
def test_projection_rebuilds_each_delta_from_trimmed_prior_state(monkeypatch, initial):
    import kernel_block_diet as diet
    from kernel_stateframe_frame_append import _make_delta, materialize_payload
    from test_block_diet import _read_payload

    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "on")
    full = _read_payload()
    if not initial:
        del full["state"]["framefit5"]["read_facts"]
    states = [full]
    for operation in ("add", "append", "replace", "revision", "delete", "remove-frame", "parent"):
        full = copy.deepcopy(full)
        frame = full["state"]["framefit5"]
        if operation == "add":
            frame["read_facts"] = copy.deepcopy(_read_payload()["state"]["framefit5"]["read_facts"][:1])
        elif operation == "append":
            frame["read_facts"].append(copy.deepcopy(_read_payload()["state"]["framefit5"]["read_facts"][1]))
        elif operation == "replace":
            frame["read_facts"][0]["status"] = "CHANGED"
        elif operation == "revision":
            frame["read_facts"][0]["current_revision"] = "sha256:" + "f" * 64
        elif operation == "delete":
            del frame["read_facts"]
        elif operation == "remove-frame":
            full["state"]["framefit5"] = None
        else:
            full["state"]["framefit5"] = copy.deepcopy(_read_payload()["state"]["framefit5"])
        states.append(full)
    base = {"role": "user", "content": serialize_stateframe_payload(states[0])}
    deltas = [_make_delta(before, after, turn) for turn, (before, after)
              in enumerate(zip(states, states[1:]), 1)]
    page = {"role": "user", "content": "[page src:pkg/a.py v1]\nsource text"}
    body = {"messages": [base, page, *deltas]}
    before = copy.deepcopy(body)
    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "off")
    assert diet.render_read_facts(body) is body
    assert diet.snapshot_read_facts(body) is None
    assert json.dumps(body).encode() == json.dumps(before).encode()
    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "on")
    snapshot = diet.snapshot_read_facts(body)
    projected = diet.render_read_facts(body)
    assert body == before
    assert projected["messages"][1] == page
    assert diet.render_read_facts(projected) == projected
    assert diet.host_read_fact_messages(projected["messages"], snapshot) == body["messages"]
    for turn in range(len(deltas) + 1):
        actual = materialize_payload(projected["messages"][0], projected["messages"][2:2 + turn])
        expected = copy.deepcopy(states[turn])
        frame = expected["state"]["framefit5"]
        if isinstance(frame, dict) and "read_facts" in frame:
            frame["read_facts"] = diet.compact_read_facts(frame["read_facts"])
        assert actual == expected
        prefix = {"messages": body["messages"][:2 + turn]}
        assert diet.render_read_facts(prefix)["messages"] == projected["messages"][:2 + turn]



@pytest.mark.parametrize("encoding", ["page-log", "final-cache", "stable-wire"])
def test_read_fact_authority_overlay_skips_losslessly_and_journals(monkeypatch, encoding):
    from types import SimpleNamespace

    import kernel_block_diet as diet
    from kernel_cache_layout_candidate import _project_final_cache_layout
    from test_block_diet import _read_payload
    from test_final_cache_layout import _body

    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "on")
    body = _body()
    payload = json.loads(body["messages"][2]["content"].split("\n", 1)[1])
    facts = _read_payload()["state"]["framefit5"]["read_facts"]
    facts[0]["stop_authority"] = "current authority"
    payload["state"]["framefit5"]["read_facts"] = facts
    body["messages"][2]["content"] = serialize_stateframe_payload(payload)
    if encoding == "page-log":
        from kernel_stateframe_page_log import project_request

        body = project_request(body, token_meter=None, require_product=False,
                               measure_token_policy=False).body
    else:
        body = _project_final_cache_layout(body)
    if encoding == "stable-wire":
        body = _stable_projection(monkeypatch, body)
    before = copy.deepcopy(body)
    events = []
    journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))
    assert diet.snapshot_read_facts(body) is None
    assert diet.render_read_facts(body, journal=journal) is body
    assert body == before
    assert semantic_messages(body["messages"])[2]["content"] == serialize_stateframe_payload(payload)
    assert events == [("block_diet_skipped", {
        "producer": "kernel_block_diet", "cut": 2,
        "reason": "read_fact_authority_overlay", "dispatch": False})]


def test_preflight_validates_full_delta_carriers_then_returns_sealed_wire(monkeypatch):
    from types import SimpleNamespace

    import capability_request_preflight as preflight
    import kernel_block_diet as diet
    from kernel_stateframe_frame_append import _make_delta
    from test_block_diet import _read_payload

    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "on")
    base = _read_payload()
    current = copy.deepcopy(base)
    current["state"]["framefit5"]["read_facts"][0]["status"] = "CHANGED"
    messages = [{"role": "user", "content": serialize_stateframe_payload(base)},
                _make_delta(base, current, 1)]
    snapshot = diet.snapshot_read_facts({"messages": messages})
    wire = diet.render_read_facts({"messages": messages})["messages"]
    allocation = SimpleNamespace(digest="allocation")
    sealed = SimpleNamespace(
        allocation=allocation, body={"messages": wire, "tools": []}, body_digest="body",
        metadata=json.dumps({"block_diet_host_read_facts": snapshot}),
        validate=lambda *_args: None)
    runtime = SimpleNamespace(_active_capture=SimpleNamespace(qualified_request=sealed), current_request_id="r")
    observed = []

    def check(full, _spec, _runtime, **kwargs):
        assert full == messages
        assert kwargs["frame_request_allocation"] is allocation
        observed.append(copy.deepcopy(full))
        return {"checked": True}

    monkeypatch.setattr(preflight, "check_presentation", check)
    assert diet.allocation_message_views(messages, runtime, allocation) == (wire, messages)
    assert diet.allocation_message_views(wire, runtime, allocation) == (wire, messages)
    modified = copy.deepcopy(messages)
    modified[-1]["content"] += " "
    assert diet.allocation_message_views(modified, runtime, allocation) == (modified, modified)
    # Metadata projection stays bound to the seal even if configuration changes.
    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "off")
    prepared, receipt = preflight.preflight_presentation_or_degrade(
        wire, {}, runtime, tools=[], frame_request_allocation=allocation,
        precomposed_host_text={"schema": preflight.PRECOMPOSED_HOST_TEXT_SCHEMA,
                               "request_id": "r", "body_digest": "body",
                               "allocation_digest": "allocation", "receipt": {}})
    assert observed == [messages]
    assert prepared == wire
    assert receipt["checked"] is True


def test_projected_delta_tampering_and_invalid_full_chain_are_rejected(monkeypatch):
    import kernel_block_diet as diet
    from kernel_request_seal import RequestSealError
    from kernel_stateframe_frame_append import FrameAppendError, _make_delta
    from test_block_diet import _read_payload

    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "on")
    base = _read_payload()
    after = copy.deepcopy(base)
    after["state"]["framefit5"]["read_facts"][0]["status"] = "CHANGED"
    body = {"messages": [{"role": "user", "content": serialize_stateframe_payload(base)},
                         _make_delta(base, after, 1)]}
    snapshot = diet.snapshot_read_facts(body)
    projected = diet.render_read_facts(body)["messages"]
    forged = copy.deepcopy(projected)
    forged[-1]["content"] = forged[-1]["content"].replace("CHANGED", "FORGED")
    with pytest.raises(RequestSealError, match="read-fact projection"):
        diet.host_read_fact_messages(forged, snapshot)
    broken = copy.deepcopy(body)
    broken["messages"][-1]["content"] = broken["messages"][-1]["content"].replace("CHANGED", "FORGED")
    with pytest.raises(FrameAppendError, match="content digest"):
        diet.render_read_facts(broken)


@pytest.mark.parametrize("stable", [False, True])
def test_final_cache_and_stable_wire_host_roundtrip(monkeypatch, stable):
    import kernel_block_diet as diet
    from kernel_cache_layout_candidate import _project_final_cache_layout
    from test_block_diet import _read_payload
    from test_final_cache_layout import _body

    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "on")
    body = _body()
    full = json.loads(body["messages"][2]["content"].split("\n", 1)[1])
    full["state"]["framefit5"]["read_facts"] = _read_payload()["state"]["framefit5"]["read_facts"]
    body["messages"][2]["content"] = serialize_stateframe_payload(full)
    body = _project_final_cache_layout(body)
    if stable:
        body = _stable_projection(monkeypatch, body)
    before = copy.deepcopy(body)
    snapshot = diet.snapshot_read_facts(body)
    projected = diet.render_read_facts(body)
    view = semantic_messages(projected["messages"])
    payload = next(json.loads(row["content"].split("\n", 1)[1]) for row in view
                   if row.get("content", "").startswith("[StateFrame]\n"))
    assert payload["state"]["framefit5"]["read_facts"] == diet.compact_read_facts(
        full["state"]["framefit5"]["read_facts"])
    assert diet.host_read_fact_messages(projected["messages"], snapshot) == body["messages"]
    assert body == before


@pytest.mark.parametrize("opaque_kind", ["frame", "malformed-frame", "delta"])
def test_stable_wire_diet_preserves_owned_positions_and_opaque_rows(monkeypatch, opaque_kind):
    from types import SimpleNamespace

    import kernel_block_diet as diet
    import kernel_stable_wire as stable
    from kernel_stable_wire_ownership import projection_order
    from test_block_diet import _read_payload
    from test_newest_user_turn_survives_wire_projection import ADVISORY, _final_cache_shaped_body
    from test_stable_wire import Journal, Meter, _invoke

    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "on")
    body = _final_cache_shaped_body()
    frame_index = stable._frame_index(body["messages"])
    payload = json.loads(body["messages"][frame_index]["content"].split("\n", 1)[1])
    facts = _read_payload()["state"]["framefit5"]["read_facts"]
    payload["state"]["framefit5"]["read_facts"] = facts
    body["messages"][frame_index]["content"] = serialize_stateframe_payload(payload)
    opaque = next(row for row in body["messages"] if row.get("content") == ADVISORY)
    opaque["content"] = {
        "frame": serialize_stateframe_payload(_read_payload()),
        "malformed-frame": "[StateFrame]\nnot JSON",
        "delta": "[StateFrame delta]\nnot JSON",
    }[opaque_kind]
    if opaque_kind == "delta":
        opaque["name"] = "stateframe_frame_delta"
    capture = SimpleNamespace(journal=Journal(), request_id="read-facts",
                              _framefit5_newest_user_suffix=(copy.deepcopy(opaque),))
    # Install the existing deterministic source-delivery fixture, then project
    # with the real captured-user ownership rather than inferring it from text.
    _invoke(monkeypatch, _final_cache_shaped_body(), SimpleNamespace(_stable_wire_state=None))
    body = stable.project_stable_wire(
        capture, SimpleNamespace(_stable_wire_state=None), body, meter=Meter(),
        inputs={"token_ceiling": 100000, "byte_ceiling": 1000000,
                "source_plan": {}, "sources": {}, "receipt": None})
    assert capture._final_cache_layout_record["stable_wire"] == "applied"
    before = copy.deepcopy(body)
    index = stable._frame_index(body["messages"])
    order = projection_order(body["messages"], index)
    snapshot = diet.snapshot_read_facts(body)
    projected = diet.render_read_facts(body)
    recorded = json.loads(json.dumps(projected))["messages"]
    assert projection_order(recorded, index) == order
    assert order.newest_count == 1
    assert len(recorded) == len(body["messages"])
    for position, row in enumerate(body["messages"]):
        if position != index:
            assert recorded[position] == row
    assert opaque in stable.canonicalize_stable_wire(recorded)[0]
    materialized = json.loads(stable.materialize_stable_frame(recorded).split("\n", 1)[1])
    assert materialized["state"]["framefit5"]["read_facts"] == diet.compact_read_facts(facts)
    assert diet.host_read_fact_messages(recorded, snapshot) == body["messages"]
    assert snapshot is not None and set(snapshot) == {str(index)}
    assert body == before


@pytest.mark.parametrize("operation", ["render", "snapshot", "restore"])
def test_stable_wire_diet_rejects_missing_ownership(monkeypatch, operation):
    import kernel_block_diet as diet
    import kernel_stable_wire as stable
    from kernel_stable_wire_ownership import ORDER_MARKER
    from test_block_diet import _read_payload
    from test_stable_wire import _body

    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "on")
    body = _stable_projection(monkeypatch, _body(payload=_read_payload()))
    snapshot = diet.snapshot_read_facts(body)
    wire = diet.render_read_facts(body)
    index = stable._frame_index(wire["messages"])
    status = wire["messages"][index + 2]
    status["content"] = status["content"].split(ORDER_MARKER, 1)[0]
    with pytest.raises(stable.StableWireError, match="row ownership"):
        if operation == "render":
            diet.render_read_facts(wire)
        elif operation == "snapshot":
            diet.snapshot_read_facts(wire)
        else:
            diet.host_read_fact_messages(wire["messages"], snapshot)


@pytest.mark.parametrize("mode", [None, "off"])
@pytest.mark.parametrize("dialect", ["openai-strict", "llama.cpp"])
def test_diet_off_provider_bytes_and_journal_match_head(monkeypatch, mode, dialect):
    import subprocess
    from types import SimpleNamespace

    import kernel_block_diet as diet
    from kernel_final_wire_allocation import _finalize_provider_request_body
    from kernel_provider_dialect import ProviderCapabilities
    from test_block_diet import _read_payload
    from test_stable_wire import _body

    # Read the exact pre-diet release from local objects; never change a checkout.
    source = subprocess.check_output([
        "git", "show", "56af0dc6d6:"
        "src/stateframe/engine/kernel_final_wire_allocation.py"], cwd=ROOT, text=True)
    baseline = {"__name__": "baseline_final_wire_allocation"}
    exec(compile(source, "HEAD/kernel_final_wire_allocation.py", "exec"), baseline)
    body = _stable_projection(monkeypatch, _body(payload=_read_payload()))
    if dialect == "openai-strict":
        monkeypatch.setenv("REACT_REASONING_EFFORT", "none")
    if mode is None:
        monkeypatch.delenv("STATEFRAME_BLOCK_DIET", raising=False)
    else:
        monkeypatch.setenv("STATEFRAME_BLOCK_DIET", mode)
    capabilities = ProviderCapabilities(dialect=dialect)
    wires, journals = [], []
    for finalize in (baseline["_finalize_provider_request_body"], _finalize_provider_request_body):
        events = []
        journal = SimpleNamespace(manifest={"reasoning_profile": "off"},
                                  emit=lambda kind, **fields: events.append((kind, fields)))
        selected = finalize(SimpleNamespace(journal=journal), copy.deepcopy(body), capabilities)
        assert diet.snapshot_read_facts(selected) is None
        wires.append(json.dumps(selected).encode())
        journals.append(json.dumps(events).encode())
    assert wires[0] == wires[1]
    assert journals[0] == journals[1] == b"[]"


def test_invalid_wire_mode_falls_off_and_journals(monkeypatch):
    from types import SimpleNamespace

    from kernel_final_wire_allocation import _finalize_provider_request_body
    from kernel_provider_dialect import ProviderCapabilities
    from test_block_diet import _read_payload

    events = []
    capture = SimpleNamespace(journal=SimpleNamespace(
        manifest={"reasoning_profile": "off"}, emit=lambda kind, **fields: events.append((kind, fields))))
    body = {"model": "local", "messages": [{"role": "user", "content": serialize_stateframe_payload(_read_payload())}]}
    monkeypatch.setenv("REACT_REASONING_EFFORT", "none")
    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "off")
    off = _finalize_provider_request_body(capture, body, ProviderCapabilities())
    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "invalid-secret-value")
    import kernel_block_diet as diet
    assert diet.snapshot_read_facts(body) is None
    assert _finalize_provider_request_body(capture, body, ProviderCapabilities()) == off
    assert len(events) == 1
    assert events[0][0] == "block_diet_skipped"
    assert events[0][1]["reason"] == "invalid_mode"
    assert "invalid-secret-value" not in json.dumps(events)


def test_authority_marker_inside_source_data_does_not_parse_as_overlay(monkeypatch):
    import kernel_block_diet as diet
    from kernel_stateframe_final_message import AUTHORITY_OVERLAY_HEADER
    from test_block_diet import _read_payload

    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "on")
    source = {"role": "tool", "content": "source code: " + AUTHORITY_OVERLAY_HEADER + "\nnot JSON"}
    body = {"messages": [{"role": "user", "content": serialize_stateframe_payload(_read_payload())}, source]}
    projected = diet.render_read_facts(body)
    assert projected["messages"][1] == source
    assert len(projected["messages"][0]["content"]) < len(body["messages"][0]["content"])


@pytest.mark.parametrize("encoding", ["plain", "final-cache", "stable-wire"])
def test_allocation_capture_semantics_use_full_facts(monkeypatch, encoding):
    from types import SimpleNamespace

    import kernel_block_diet as diet
    import kernel_frame_allocation_capture as capture_module
    from kernel_cache_layout_candidate import _project_final_cache_layout
    from test_block_diet import _read_payload
    from test_final_cache_layout import _body

    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "on")
    body = _body()
    payload = json.loads(body["messages"][2]["content"].split("\n", 1)[1])
    facts = _read_payload()["state"]["framefit5"]["read_facts"]
    payload["state"]["framefit5"]["mode"] = "FRAME_ONLY"
    payload["state"]["framefit5"]["read_facts"] = facts
    body["messages"][2]["content"] = serialize_stateframe_payload(payload)
    if encoding != "plain":
        body = _project_final_cache_layout(body)
    if encoding == "stable-wire":
        body = _stable_projection(monkeypatch, body)
    snapshot = diet.snapshot_read_facts(body)
    wire = diet.render_read_facts(body)
    seen = []

    class Observed(Exception):
        pass

    def select(_optional, _inventory, frame):
        seen.append(frame["read_facts"])
        raise Observed

    monkeypatch.setattr(capture_module, "selection_from_frame", select)
    with pytest.raises(Observed):
        capture_module.capture_candidate(
            runtime=SimpleNamespace(), capture=SimpleNamespace(), body=wire,
            source_plan={"units": []}, receipt={"status": "COMPLETE", "proof_status": "VERIFIED"},
            sources={}, optional_inputs=({}, {}), byte_ceiling=100000,
            token_ceiling=33344, measurement_identity="tokenize", read_fact_snapshot=snapshot)
    assert seen == [facts]


def test_early_allocation_validation_uses_full_facts_and_sealed_placement(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import kernel_block_diet as diet
    import kernel_frame_optional_allocation as optional_module
    import kernel_frame_allocation_validation as validation
    import kernel_frame_source_snapshot as sources
    import kernel_request_source_window as windows
    from kernel_frame_optional_allocation import identity
    from kernel_request_budget import HARD_MAX_REQUEST_BYTES
    from test_block_diet import _read_payload

    monkeypatch.setenv("STATEFRAME_BLOCK_DIET", "on")
    payload = _read_payload()
    messages = [{"role": "user", "content": serialize_stateframe_payload(payload)}]
    snapshot = diet.snapshot_read_facts({"messages": messages})
    wire = diet.render_read_facts({"messages": messages})["messages"]
    authorization = {"scope": "workspace", "generation": "1"}
    plan = {"workspace_authorization": authorization}
    allocation = SimpleNamespace(
        validate=lambda: None, source_layout={"source_plan": plan}, optional_heads={}, site_inventory={},
        identity={"product_revision": "revision", "workspace_authorization": authorization,
                  "ledger_digest": identity(payload["state"]["framefit5"]["action_ledger"])},
        capacity={"frame_ceiling": 10000, "token_ceiling": 33344,
                  "host_byte_ceiling": HARD_MAX_REQUEST_BYTES,
                  "template_environment": validation.template_environment(), "system_messages_digest": identity([])})
    sealed = SimpleNamespace(allocation=allocation, body={"messages": wire},
                             metadata=json.dumps({"block_diet_host_read_facts": snapshot}))
    runtime = SimpleNamespace(binding=lambda: {"product_revision": "revision"}, frame_ceiling=10000,
                              owner=SimpleNamespace(root=tmp_path),
                              _request_source_closure=plan,
                              _active_capture=SimpleNamespace(qualified_request=sealed))
    seen = []

    def select(frame, *_args, **_kwargs):
        seen.append(frame["read_facts"])
        return frame

    def placement(actual, allocated):
        assert actual == wire
        assert allocated is allocation

    monkeypatch.setattr(sources, "frame_source_snapshot", lambda *_args: {})
    monkeypatch.setattr(windows, "_validate_frame_request_allocation_placement", placement)
    monkeypatch.setattr(optional_module, "render_optional_selection", select)
    validation.validate_allocation_authority(allocation, runtime, wire)
    validation.validate_allocation_authority(allocation, runtime, messages)
    assert seen == [payload["state"]["framefit5"]["read_facts"]] * 2

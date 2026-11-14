"""Allocator/stable-wire coexistence must be explicit before source placement."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import kernel_cache_layout_candidate as cache_candidate
import kernel_stable_wire as stable
import pytest
import test_read_admission_qualification as qualification
from kernel_final_wire_allocation import build_qualified_request
from kernel_request_seal import RequestSealError
from test_stable_wire import Journal
from tools.pinned_tree import pinned_tree

REASON = "working_set_allocator_incompatible"


@pytest.mark.parametrize("stable_mode,recorded", [("on", None), ("off", "applied")])
def test_on_stable_wire_refuses_two_requests_without_shifting_prefix(
        tmp_path, monkeypatch, stable_mode, recorded):
    """Changing ON source sets cannot publish a silently unstable stable prefix."""
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", stable_mode)
    monkeypatch.setattr(cache_candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", lambda: recorded)
    prefixes = []
    refused = []

    def qualify(capture, runtime, body, **kwargs):
        before = json.dumps(body).encode()
        state = copy.deepcopy(runtime.working_set_state.checkpoint())
        ledger = runtime.read_capture_ledger.all()
        prefix = json.dumps(body["messages"][:2]).encode()
        prefixes.append(prefix)
        try:
            return build_qualified_request(capture, runtime, body, **kwargs)
        except RequestSealError as error:
            assert error.outcome == "HARNESS_CONTRACT"
            assert error.reason_code == REASON
            assert json.dumps(body).encode() == before
            assert json.dumps(body["messages"][:2]).encode() == prefix
            assert runtime.working_set_state.checkpoint() == state
            assert runtime.read_capture_ledger.all() == ledger
            assert capture.qualified_request is None
            assert not getattr(capture, "_working_set_certificate", None)
            rows = [json.loads(line) for line in capture.journal.path.read_bytes().splitlines()]
            event, = [row for row in rows if row["kind"] == "stable_wire_refused"]
            assert event["reason_code"] == REASON
            assert event["outcome"] == "HARNESS_CONTRACT"
            assert event["request_id"] == capture.request_id
            assert not any(row["kind"] in {"working_set_prepared", "qualified_request_sealed"}
                           for row in rows)
            refused.append(capture.request_id)
            raise

    monkeypatch.setattr(qualification, "build_qualified_request", qualify)
    for scenario in ("K1", "K2"):
        root = tmp_path / scenario
        root.mkdir()
        with pytest.raises(RequestSealError, match=REASON):
            qualification.seal_case(root, monkeypatch, scenario)
    assert len(refused) == 2
    assert prefixes[0] == prefixes[1]


@pytest.mark.parametrize("recorded", [None, "plain"])
def test_allocator_without_stable_projection_has_no_conflict(monkeypatch, recorded):
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    monkeypatch.setattr(cache_candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", lambda: recorded)
    capture = SimpleNamespace(journal=Journal(), request_id="compatible")
    stable.require_allocator_compatibility(capture)
    assert capture.journal.events == []


def test_refusal_has_explicit_resume_policy():
    from kernel_continuation_restart import (
        DOCUMENTED_REJECT,
        resume_event_rejection_reason,
    )

    assert "stable_wire_refused" in DOCUMENTED_REJECT
    reason = resume_event_rejection_reason("stable_wire_refused")
    assert reason is not None and "unclassified" not in reason


def test_refusal_preserves_existing_stable_epoch_and_prefix(monkeypatch):
    from test_stable_wire import _body, _invoke, _serialized_stable_messages, _window

    runtime = SimpleNamespace(_stable_wire_state=None)
    projected, capture, _ = _invoke(monkeypatch, _body([_window()]), runtime)
    assert stable.is_stable_wire_projection(projected["messages"])
    assert runtime._stable_wire_state is not None
    previous_state = copy.deepcopy(runtime._stable_wire_state)
    previous_body = json.dumps(projected).encode()
    previous_prefix = _serialized_stable_messages(projected["messages"])
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    for request_id in ("next-1", "next-2"):
        capture.request_id = request_id
        with pytest.raises(stable.StableWireAllocatorConflict):
            stable.require_allocator_compatibility(capture)
        assert _serialized_stable_messages(projected["messages"]) == previous_prefix
        assert json.dumps(projected).encode() == previous_body
        assert runtime._stable_wire_state == previous_state
    assert [row["request_id"] for row in capture.journal.events
            if row["kind"] == "stable_wire_refused"] == ["next-1", "next-2"]


@pytest.mark.parametrize("stable_mode", ["off", "on"])
def test_allocator_off_provider_bytes_match_carried_3b(tmp_path, tmp_path_factory, stable_mode):
    """Exercise actual provider serialization with both stable switch settings."""
    root = Path(__file__).resolve().parents[1]
    base = pinned_tree("1f6b7e9bc1", tmp_path_factory)
    driver = '''
import json, sys
from pathlib import Path
import pytest
from test_request_determinism import _run_scripted_example, EXAMPLES
with pytest.MonkeyPatch.context() as patch:
    patch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    from stateframe import runner
    runner.test_interpreter.choose = lambda *args, **kwargs: (sys.executable, "parity test interpreter")
    patch.setenv("SF_LAB_DETFIX3", "on")
    hashes, _, _ = _run_scripted_example(
        patch, "discount_bug", EXAMPLES["discount_bug"], Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(hashes))
'''
    results = []
    for label, tree in (("base", base), ("current", root)):
        destination = tmp_path / f"{label}.json"
        environment = {k: v for k, v in os.environ.items() if not k.startswith("STATEFRAME_")}
        environment["STATEFRAME_STABLE_WIRE"] = stable_mode
        environment["PYTHONPATH"] = os.pathsep.join(str(tree / path) for path in
            ("src", "src/stateframe/engine", "tests"))
        result = subprocess.run(
            [sys.executable, "-c", driver, str(tmp_path / label), str(destination)],
            cwd=tree, env=environment, capture_output=True, text=True, timeout=600, check=False)
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        results.append(json.loads(destination.read_text()))
    assert results[0] and results[0] == results[1]

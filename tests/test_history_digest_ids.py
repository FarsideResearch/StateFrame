"""STATEFRAME_HISTORY_DIGEST_IDS: the model-facing history digest may omit its tool-call id frontier."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from _stateframe_home import isolated_stateframe_home

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import kernel_framefit_receipt_history as receipt_history
from kernel_stateframe_capacity import validate_read_ranges
from kernel_stateframe_framefit5 import FrameFit5Error
from tools.pinned_tree import pinned_tree

from stateframe import profiles, runner

SWITCH = "STATEFRAME_HISTORY_DIGEST_IDS"
FIXTURE = ROOT / "tests" / "fixtures" / "pin_bm10_b" / "0020.request.json"


def _recorded_digest() -> dict[str, object]:
    """The coverage-contract digest of a recorded shipped request."""
    body = json.loads(FIXTURE.read_text(encoding="utf-8"))
    messages = body.get("messages") or body["body"]["messages"]
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and content.startswith("[StateFrame]\n"):
            payload = json.loads(content.split("\n", 1)[1].split("\n[", 1)[0])
            digest = payload["state"]["framefit5"]["action_ledger"]["history_digest"]
            assert digest["coverage_contract"] == "journal-call-coverage/1"
            assert len(digest["call_ids"]) == 64
            return copy.deepcopy(digest)
    raise AssertionError("fixture has no StateFrame block")


def _legacy_digest() -> dict[str, object]:
    return {"kind": "action_history_digest", "call_ids": ["call_1", "call_2"], "row_count": 2,
            "kind_counts": {"read": 2}, "tool_counts": {"read": 2},
            "rows_sha256": "0" * 64, "source_text": "not_carried"}


def _attach(monkeypatch, digest, environ):
    """Run attach_history on a validated context stub and return the attached digest."""
    monkeypatch.setattr(receipt_history, "validate_context", lambda context: context)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    state = {"action_ledger": {"reads": [], "edits": []}}
    history = {"history_digest": digest, "receipt_count": 0, "receipts_sha256": "0" * 64,
               "historical_read_facts": {"count": 0}, "historical_applied_edits": {"count": 0},
               "journal_receipt_records": []}
    receipt_history.attach_history(state, history)
    return state["action_ledger"]["history_digest"]


def test_switch_values():
    assert receipt_history.history_digest_ids_enabled({}) is True
    assert receipt_history.history_digest_ids_enabled({SWITCH: "on"}) is True
    assert receipt_history.history_digest_ids_enabled({SWITCH: " OFF "}) is False
    with pytest.raises(ValueError, match=SWITCH):
        receipt_history.history_digest_ids_enabled({SWITCH: "maybe"})


@pytest.mark.parametrize("environ", [{}, {SWITCH: "on"}])
def test_ids_on_or_unset_keeps_the_legacy_digest_byte_identical(monkeypatch, environ):
    monkeypatch.delenv(SWITCH, raising=False)
    recorded = _recorded_digest()
    attached = _attach(monkeypatch, recorded, environ)
    assert json.dumps(attached, sort_keys=True) == json.dumps(recorded, sort_keys=True)
    assert attached is not recorded


def test_ids_off_omits_only_call_ids(monkeypatch):
    recorded = _recorded_digest()
    attached = _attach(monkeypatch, recorded, {SWITCH: "off"})
    assert "call_ids" not in attached
    expected = {key: value for key, value in recorded.items() if key != "call_ids"}
    assert json.dumps(attached, sort_keys=True) == json.dumps(expected, sort_keys=True)
    # The source digest (the host ledger) keeps its ids.
    assert len(recorded["call_ids"]) == 64
    # The cut digest still passes the typed frame validator; the ids never were its coverage authority.
    validate_read_ranges({"action_ledger": {"reads": [], "history_digest": attached}}, [])


def test_ids_off_never_cuts_a_legacy_digest(monkeypatch):
    legacy = _legacy_digest()
    attached = _attach(monkeypatch, legacy, {SWITCH: "off"})
    assert attached == legacy


def test_validator_still_rejects_a_legacy_digest_without_ids():
    legacy = _legacy_digest()
    legacy.pop("call_ids")
    with pytest.raises(FrameFit5Error, match="malformed"):
        validate_read_ranges({"action_ledger": {"reads": [], "history_digest": legacy}}, [])


def test_r2hc3_ships_with_ids_dropped():
    assert profiles.load("r2hc3")[SWITCH] == "off"


@pytest.mark.parametrize("flag", [None, "on", "off"])
def test_rendered_block_matches_base_except_opted_out_ids(tmp_path, tmp_path_factory, flag):
    baseline = pinned_tree("d714124478acc48f46090be1fa7bab439de5e380", tmp_path_factory)
    # Replay a recorded frame through the real attachment, context validator and serializer.
    # Each process imports its own checkout, so the control uses the released implementation.
    script = r'''
import copy, json, sys
from pathlib import Path
from kernel_framefit_receipt_history import attach_history, empty_summary
from kernel_stateframe_capacity import _sha, validate_read_ranges
from kernel_stateframe_framefit5 import _serialize_state_message
body = json.loads(Path(sys.argv[1]).read_text())
messages = body.get("messages") or body["body"]["messages"]
row = next(row for row in messages if str(row.get("content", "")).startswith("[StateFrame]\n"))
payload = json.loads(row["content"].split("\n", 1)[1])
state = payload["state"]["framefit5"]
digest = copy.deepcopy(state["action_ledger"].pop("history_digest"))
history = {"schema": "framefit-receipt-history/1", "receipt_count": digest["row_count"],
           "receipts_sha256": _sha([]), "history_digest": digest, "reads": {},
           "historical_read_facts": empty_summary(), "historical_applied_edits": empty_summary()}
context = {"journal_receipt_count": digest["row_count"], "journal_receipt_records": [],
           "journal_receipt_history": history}
history["projection_sha256"] = _sha(context)
before = copy.deepcopy(context)
attach_history(state, context)
assert context == before
assert len(digest["call_ids"]) == 64
validate_read_ranges({"action_ledger": {"reads": [], "history_digest": state["action_ledger"]["history_digest"]}}, [])
print(json.dumps({"block": _serialize_state_message(payload), "context": context}))
'''
    outputs = []
    for checkout, setting in ((baseline, None), (ROOT, flag)):
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("STATEFRAME_")}
        environment["STATEFRAME_HOME"] = isolated_stateframe_home()
        environment["PYTHONPATH"] = os.pathsep.join(
            str(path) for path in (checkout / "src/stateframe/engine", checkout / "src"))
        if setting is not None:
            environment[SWITCH] = setting
        result = subprocess.run([sys.executable, "-c", script, str(FIXTURE)],
                                env=environment, capture_output=True, text=True, timeout=60,
                                check=False)
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))
    assert outputs[0]["context"] == outputs[1]["context"]
    expected = outputs[0]["block"]
    if flag == "off":
        from kernel_stateframe_framefit5 import _serialize_state_message

        payload = json.loads(expected.split("\n", 1)[1])
        digest = payload["state"]["framefit5"]["action_ledger"]["history_digest"]
        assert len(digest.pop("call_ids")) == 64
        expected = _serialize_state_message(payload)
        actual = json.loads(outputs[1]["block"].split("\n", 1)[1])
        assert "call_ids" not in actual["state"]["framefit5"]["action_ledger"]["history_digest"]
    assert outputs[1]["block"].encode() == expected.encode()


@pytest.mark.parametrize("value", [None, "on", "off"])
def test_effective_profile_and_shell_override(monkeypatch, value):
    for name in runner.OPT_IN_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv(SWITCH, value)
    options = runner.RunOptions(repo=ROOT, task="Check history digest.", endpoint=None)
    assert runner._provider_environment(options)[SWITCH] == (value or "off")


@pytest.mark.parametrize("frontier", [None, "call_1", [""], [1], ["b", "a"], ["a", "a"]])
def test_coverage_digest_still_rejects_malformed_present_ids(frontier):
    digest = _recorded_digest()
    digest["call_ids"] = frontier
    with pytest.raises(FrameFit5Error, match="malformed"):
        validate_read_ranges({"action_ledger": {"reads": [], "history_digest": digest}}, [])


def test_projection_is_deeply_isolated(monkeypatch):
    recorded = _recorded_digest()
    original = copy.deepcopy(recorded)
    attached = _attach(monkeypatch, recorded, {SWITCH: "off"})
    attached["kind_counts"]["read"] = 9999
    assert recorded == original


@pytest.mark.parametrize("field,value", [
    ("coverage_contract", "unknown/1"), ("coverage_contract", None),
    ("coverage_run_id", ""), ("call_count", -1), ("call_count", True),
    ("call_ids_sha256", "g" * 64), ("call_ids_sha256", "0" * 63),
])
def test_omitted_frontier_does_not_bypass_coverage_validation(field, value):
    digest = _recorded_digest()
    digest.pop("call_ids")
    digest[field] = value
    with pytest.raises(FrameFit5Error, match="malformed"):
        validate_read_ranges({"action_ledger": {"reads": [], "history_digest": digest}}, [])


def _proven_history(ids=None):
    from kernel_stateframe_call_coverage import (
        CoverageSession,
        bind_index,
        context_index,
    )
    from kernel_stateframe_capacity import _fold
    from test_coverage_index_base_compat import _MemoryJournal

    journal = _MemoryJournal()
    context = bind_index({}, journal)
    session = CoverageSession(context_index(context), None)
    ledger = {}
    _fold(ledger, {"call_ids": ids if ids is not None else ["call_1", "call_2"]},
          kind="read", coverage=session)
    session.commit()
    return journal, context, ledger["history_digest"]


def test_omitted_ids_keep_exact_host_coverage_and_survive_refolding(monkeypatch):
    from kernel_stateframe_call_coverage import CoverageSession, context_index
    from kernel_stateframe_capacity import _fold, ledger_ids, validate_coverage
    from kernel_stateframe_framefit5 import _serialize_state_message

    journal, context, digest = _proven_history()
    monkeypatch.setenv(SWITCH, "off")
    original = copy.deepcopy(digest)
    ledger = {"reads": [], "history_digest": receipt_history.model_facing_digest(digest)}
    validate_coverage({"action_ledger": ledger}, context)
    assert ledger_ids(ledger, context) == {"call_1", "call_2"}
    session = CoverageSession(context_index(context), ledger["history_digest"])
    _fold(ledger, {"call_ids": ["call_3"]}, kind="read", coverage=session)
    session.commit()
    validate_coverage({"action_ledger": ledger}, context)
    assert ledger_ids(ledger, context) == {"call_1", "call_2", "call_3"}
    block = _serialize_state_message({"state": {"framefit5": {"action_ledger": ledger}}})
    rendered = json.loads(block.split("\n", 1)[1])
    assert "call_ids" not in rendered["state"]["framefit5"]["action_ledger"]["history_digest"]
    assert digest == original
    assert journal.records[-1]["coverage_proof"]["digest"]["call_ids"] == ["call_1", "call_2", "call_3"]


@pytest.mark.parametrize("field,value", [
    ("rows_sha256", "f" * 64), ("row_count", 2), ("call_count", 1),
    ("coverage_run_id", "another-run"), ("call_ids_sha256", "0" * 64),
    ("call_ids", []), ("call_ids", ["forged"]),
])
def test_omitted_ids_never_authorize_a_changed_host_proof(monkeypatch, field, value):
    from kernel_stateframe_call_coverage import context_index

    _journal, context, digest = _proven_history()
    monkeypatch.setenv(SWITCH, "off")
    projected = receipt_history.model_facing_digest(digest)
    projected[field] = value
    with pytest.raises(FrameFit5Error, match="host proof"):
        context_index(context).covered_ids(projected)


def test_capacity_refolding_never_reintroduces_rendered_ids(monkeypatch):
    from kernel_stateframe_capacity import fit_protected_frame, ledger_ids
    from kernel_stateframe_framefit5 import _serialize_state_message, _state_payload
    from test_source_window_order_state import _request_body

    journal, context, digest = _proven_history()
    monkeypatch.setenv(SWITCH, "off")
    context["covered_call_ids"] = ["call_1", "call_2", "call_3"]
    # Reuse a frame-only request fixture so source-truth refresh sees the same
    # product mode that build_typed_state always supplies.
    body = _request_body()
    state_message = next(row for row in body["messages"]
                         if row["content"].startswith("[StateFrame]\n"))
    payload, state = _state_payload(state_message)
    frame = state["framefit5"]
    frame.update({"action_ledger": {"reads": [], "edits": [], "unclassified_actions": [],
                                    "history_digest": digest},
                  "read_facts": [], "applied_edits": [], "open_sites": [], "next_site": None})
    state_message["content"] = _serialize_state_message(payload)
    before = copy.deepcopy(body)
    fit_protected_frame(body, action_context=context, max_request_bytes=1_000_000)
    assert body != before
    rendered = next(_state_payload(row)[1]["framefit5"] for row in body["messages"]
                    if row["content"].startswith("[StateFrame]\n"))
    assert "call_ids" not in rendered["action_ledger"]["history_digest"]
    assert ledger_ids(rendered["action_ledger"], context) == {"call_1", "call_2", "call_3"}
    assert journal.records[-1]["coverage_proof"]["digest"]["call_ids"] == ["call_1", "call_2", "call_3"]


def test_omitted_minimized_frontier_keeps_its_exact_host_proof(monkeypatch):
    from kernel_stateframe_call_coverage import CoverageSession, context_index

    _journal, context, digest = _proven_history()
    index = context_index(context)
    session = CoverageSession(index, digest)
    assert session.minimize_frontier(digest)
    session.commit()
    assert digest["call_ids"] == []
    monkeypatch.setenv(SWITCH, "off")
    projected = receipt_history.model_facing_digest(digest)
    assert "call_ids" not in projected
    assert index.covered_ids(projected) == {"call_1", "call_2"}


@pytest.mark.parametrize("ids", [
    [f"call_{index:03}" for index in range(100)],
    [f"call_{index:03}_" + "x" * 180 for index in range(80)],
    ["x" * 300, "雪" * 100, "call_1", "call_2"],
])
def test_omitted_frontier_restores_only_its_original_host_prefix(monkeypatch, ids):
    from kernel_stateframe_call_coverage import CoverageSession, context_index
    from kernel_stateframe_capacity import _fold

    _journal, context, digest = _proven_history(ids)
    index = context_index(context)
    original = copy.deepcopy(digest)
    later = {"history_digest": copy.deepcopy(digest)}
    session = CoverageSession(index, later["history_digest"])
    _fold(later, {"call_ids": ["newer_call"]}, kind="read", coverage=session)
    session.commit()
    monkeypatch.setenv(SWITCH, "off")
    projected = receipt_history.model_facing_digest(digest)
    assert index.covered_ids(projected) == set(ids)
    assert digest == original


def test_shell_switch_is_forwarded_and_declared(monkeypatch):
    assert runner.OPT_IN_SWITCHES[SWITCH] == ("on", "off")
    assert SWITCH in runner.engine_switches()
    monkeypatch.setenv(SWITCH, "on")
    assert runner.resolve_opt_in_switches()[SWITCH] == "on"
    monkeypatch.setenv(SWITCH, "sometimes")
    with pytest.raises(runner.RunError, match=SWITCH):
        runner.resolve_opt_in_switches()
    monkeypatch.delenv(SWITCH)
    assert SWITCH not in runner.resolve_opt_in_switches()

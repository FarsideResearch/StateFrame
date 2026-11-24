from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import kernel_fanout_continuation as continuation
import kernel_fanout_continuation_cli as continuation_cli


FIXTURES = Path(__file__).parent / "fixtures" / "fanout_continuation"
R6_WORKSPACE = FIXTURES / "r6" / "workspace"


def _unit(
    unit_id: str,
    *,
    status: str = "COMMITTED",
    check_result: str = "PASS",
    owns: tuple[str, ...] = (),
    checks: tuple[str, ...] = (),
    child_sends: int = 1,
    changed_paths: tuple[str, ...] = (),
    contract: bool | None = True,
    reason: str | None = None,
) -> dict[str, object]:
    return {
        "id": unit_id,
        "owns": list(owns),
        "check": list(checks),
        "status": status,
        "check_result": check_result,
        "child_sends": child_sends,
        "changed_paths": list(changed_paths),
        "contract": contract,
        "reason": reason,
    }


def _case(name: str) -> dict[str, object]:
    if name == "r6":
        return {
            "schema": "stateframe-protected-fanout-disposition/1",
            "kind": "protected_fanout_disposition",
            "call_id": "native_9ad5523efee214ee8bde_0",
            "turn_index": 3,
            "units": [
                _unit("invoicing", owns=("shop/invoicing.py",),
                      checks=("tests/test_invoicing.py",), changed_paths=("shop/invoicing.py",)),
                _unit("pricing", owns=("shop/pricing.py",),
                      checks=("tests/test_pricing.py",), changed_paths=("shop/pricing.py",)),
                _unit("shipping", status="REFUSED", check_result="NOT_RUN",
                      owns=("shop/shipping.py",), checks=("tests/test_shipping.py",),
                      child_sends=0, contract=None,
                      reason="SUBFRAME_CONTEXT_TOO_LARGE: first_frame_tokens=35136; cap=33344"),
            ],
            "final_check": "FAIL",
            "integration": "FAILED",
            "failure": {
                "unit_id": "shipping",
                "nodeid": "tests/test_shipping.py::test_bulky_light_carton_uses_carrier_dimensional_divisor",
                "location": "tests/test_shipping.py:14",
                "expected": "9",
                "observed": "11",
                "assertion": "assert quote.billable_pounds == 9",
            },
            # Deliberate hostile metadata must never be copied to the frame text.
            "recorded_run": "r6_p2_a1",
            "fanout_observation_sha256": "f" * 64,
            "recorded_reread_trace": "send 4 onward",
        }
    names = {
        "r7": ("invoicing_fix", "pricing_fix", "shipping_fix"),
        "r8": ("invoicing", "pricing", "shipping"),
        "repeat": ("invoicing_fix", "pricing_fix", "shipping_fix"),
    }[name]
    return {
        "schema": "stateframe-protected-fanout-disposition/1",
        "kind": "protected_fanout_disposition",
        "call_id": f"{name}-call",
        "turn_index": 6,
        "units": [
            _unit(unit_id, owns=(f"shop/{path}.py",), checks=(f"tests/test_{path}.py",),
                  changed_paths=(f"shop/{path}.py",))
            for unit_id, path in zip(names, ("invoicing", "pricing", "shipping"), strict=True)
        ],
        "final_check": "PASS",
        "integration": "VERIFIED",
        "recorded_run": f"{name}_p2_a1",
        "fanout_observation_sha256": "e" * 64,
    }


@pytest.mark.parametrize("name", ["r6", "r7", "r8", "repeat"])
def test_recorded_shapes_render_golden_sections_without_provenance(name: str):
    workspace = R6_WORKSPACE if name == "r6" else None
    section, pin, events = continuation.render(_case(name), workspace, budget=3072)
    expected = (FIXTURES / f"{name}.expected.txt").read_text().rstrip("\n")

    assert section == expected
    assert len(section.encode("utf-8")) <= 3072
    assert "recorded run=" not in section
    assert "fanout_observation_sha256" not in section
    assert "reread trace" not in section
    assert events == []
    if name == "r6":
        assert pin
        assert len(pin.encode("utf-8")) <= continuation.MAX_PIN_BYTES
    else:
        assert pin == ""


def test_lifecycle_partial_repaired_verified_then_retired(tmp_path: Path):
    workspace = tmp_path
    source = workspace / "shop/shipping.py"
    test = workspace / "tests/test_shipping.py"
    source.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    source.write_text("DIM = 100\ndef quote_weight(value):\n    return value / DIM\n")
    test.write_text(
        "from shop.shipping import quote_weight\n\n"
        "def test_weight():\n    assert quote_weight(12) == 9\n"
    )
    partial = _case("r6")
    partial["failure"] = {
        "unit_id": "shipping", "nodeid": "tests/test_shipping.py::test_weight",
        "location": "tests/test_shipping.py:4", "expected": "9", "observed": "12",
        "assertion": "assert quote_weight(12) == 9",
    }
    section, pin, _events = continuation.render(partial, workspace, budget=3072)
    assert "remaining=[shipping]" in section
    assert "next_action: repair only shipping" in section
    assert "shop/shipping.py" in pin and "tests/test_shipping.py" in pin

    repaired = copy.deepcopy(partial)
    repaired["lifecycle"] = {
        "current_turn_index": 5,
        "fanout_turn_index": 3,
        "passed_checks": {"tests/test_shipping.py": 5},
    }
    # The recorded parent result must itself pass before the continuation may say VERIFIED.
    # The r6 fixture's recorded final check is FAIL, so the repaired render reports FAILED here.
    not_verified, _pin, _events = continuation.render(repaired, workspace, budget=3072)
    assert "final_check=FAIL; final_integration=FAILED" in not_verified
    assert "final_check=PASS" not in not_verified
    assert "integration=VERIFIED" not in not_verified

    repaired["final_check"] = "PASS"
    repaired["integration"] = "VERIFIED"
    verified, released_pin, _events = continuation.render(repaired, workspace, budget=3072)
    assert "remaining=[]" in verified
    assert "final_check=PASS; final_integration=VERIFIED" in verified
    assert "next_action: all subframes committed and verified" in verified
    assert released_pin == ""

    retired = copy.deepcopy(repaired)
    retired["lifecycle"]["current_turn_index"] = 6
    retired_section, retired_pin, events = continuation.render(retired, workspace, budget=3072)
    assert retired_section == ""
    assert retired_pin == ""
    assert any(event.get("kind") == "fanout_continuation_retired" for event in events)


def test_verified_disposition_is_rendered_once_then_released():
    disposition = _case("r7")
    first, first_pin, _events = continuation.render(disposition, None, budget=3072)
    assert first
    assert first_pin == ""

    following = copy.deepcopy(disposition)
    following["lifecycle"] = {"current_turn_index": 7, "fanout_turn_index": 6}
    second, second_pin, _events = continuation.render(following, None, budget=3072)
    assert second == ""
    assert second_pin == ""


def test_windowed_pin_contains_only_remaining_owned_and_check_files(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/shipping.py").write_text(
        "def compute_weight(value):\n    return value * 2\n"
    )
    (tmp_path / "tests/test_shipping.py").write_text(
        "from src.shipping import compute_weight\n\n"
        "def test_weight():\n    assert compute_weight(2) == 3\n"
    )
    (tmp_path / "src/pricing.py").write_text("SECRET_FROM_COMMITTED_UNIT = True\n")
    (tmp_path / "tests/test_pricing.py").write_text("SECRET_TEST_FROM_COMMITTED_UNIT = True\n")
    disposition = {
        "kind": "protected_fanout_disposition", "call_id": "pin-case", "turn_index": 0,
        "units": [
            _unit("pricing", owns=("src/pricing.py",), checks=("tests/test_pricing.py",)),
            _unit("shipping", status="FAILED", check_result="FAIL", owns=("src/shipping.py",),
                  checks=("tests/test_shipping.py",), changed_paths=("src/shipping.py",)),
        ],
        "final_check": "FAIL", "integration": "FAILED", "remaining": ["shipping"],
        "failure": {"unit_id": "shipping", "nodeid": "tests/test_shipping.py::test_weight",
                    "location": "tests/test_shipping.py:4", "expected": "3", "observed": "4",
                    "assertion": "assert compute_weight(2) == 3"},
    }

    section, pin, _events = continuation.render(disposition, tmp_path, budget=3072)

    assert "[FANOUT CONTINUATION]" in section
    assert "src/shipping.py" in pin
    assert "tests/test_shipping.py" in pin
    assert "SECRET_FROM_COMMITTED_UNIT" not in pin
    assert "SECRET_TEST_FROM_COMMITTED_UNIT" not in pin
    assert len(pin.encode("utf-8")) <= continuation.MAX_PIN_BYTES
    assert "revision=sha256:" + hashlib.sha256((tmp_path / "src/shipping.py").read_bytes()).hexdigest() in pin


def test_pin_over_cap_degrades_to_section_and_returns_typed_reason(tmp_path: Path):
    units = []
    for index in range(5):
        source_path = f"src/u{index}.py"
        test_path = f"tests/test_u{index}.py"
        (tmp_path / source_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / test_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / source_path).write_text(
            f"def compute_{index}(value):\n" + "    value += 1\n" * 220 + "    return value\n"
        )
        (tmp_path / test_path).write_text(
            f"def test_u{index}():\n" + "    value = 1\n" * 220 + f"    assert value == {index}\n"
        )
        units.append(_unit(f"u{index}", status="FAILED", check_result="FAIL",
                           owns=(source_path,), checks=(test_path,), changed_paths=(source_path,)))
    disposition = {
        "kind": "protected_fanout_disposition", "call_id": "large-pin", "turn_index": 0,
        "units": units, "final_check": "FAIL", "integration": "FAILED",
        "remaining": [f"u{index}" for index in range(5)],
        "failure": {"unit_id": "u0", "nodeid": "tests/test_u0.py::test_u0",
                    "location": "tests/test_u0.py:1", "expected": "0", "observed": "1",
                    "assertion": "assert value == 0"},
    }

    section, pin, events = continuation.render(disposition, tmp_path, budget=3072)

    assert section
    assert pin == ""
    event = next(event for event in events if event.get("kind") == "fanout_pin_degraded")
    assert event.get("reason") == "pin_exceeds_budget"
    assert event.get("cap_bytes") == continuation.MAX_PIN_BYTES


def test_max_unit_section_is_deterministically_truncated_under_cap():
    disposition = {
        "kind": "protected_fanout_disposition", "call_id": "max-units", "turn_index": 0,
        "units": [
            _unit(f"unit_{index:04d}", owns=(f"src/pkg_{index}/module.py",),
                  checks=(f"tests/test_pkg_{index}.py",), changed_paths=(f"src/pkg_{index}/module.py",))
            for index in range(1024)
        ],
        "final_check": "PASS", "integration": "VERIFIED",
    }

    section, _pin, _events = continuation.render(disposition, None, budget=3072)

    assert len(section.encode("utf-8")) <= 3072
    assert "subframes_truncated=" in section
    assert "unit_0000" in section
    assert "unit_1023" not in section


def test_switch_off_and_absent_parallel_item_leave_frame_unchanged(monkeypatch, tmp_path: Path):
    frame = {"mode": "FRAME_ONLY", "action_ledger": {"reads": [], "edits": []}}
    before = json.dumps(frame, sort_keys=True, separators=(",", ":")).encode()
    monkeypatch.setenv(continuation.SWITCH_NAME, "off")
    events = continuation.attach_to_frame(frame, None, tmp_path, budget=3072)
    after = json.dumps(frame, sort_keys=True, separators=(",", ":")).encode()
    assert before == after
    assert events == []


def test_switch_off_with_parallel_disposition_preserves_request_bytes(monkeypatch, tmp_path: Path):
    frame = {"mode": "FRAME_ONLY", "action_ledger": {"reads": [], "edits": []},
             "protected_fanout_disposition": _case("r6")}
    before = json.dumps(frame, sort_keys=True, separators=(",", ":")).encode()
    monkeypatch.setenv(continuation.SWITCH_NAME, "off")

    events = continuation.attach_to_frame(frame, None, tmp_path, budget=3072)

    after = json.dumps(frame, sort_keys=True, separators=(",", ":")).encode()
    assert before == after
    assert "fanout_continuation" not in frame
    assert "fanout_continuation_pin" not in frame
    assert events == []


def _tool_group(call_id: str, name: str, arguments: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"role": "assistant", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]},
        {"role": "tool", "tool_call_id": call_id, "content": "complete"},
    ]


def test_lifecycle_uses_post_fanout_check_and_invalidates_it_after_owned_edit():
    disposition = {
        "kind": "protected_fanout_disposition", "call_id": "fanout-call", "turn_index": 0,
        "units": [_unit("shipping", status="FAILED", check_result="FAIL",
                         owns=("shop/shipping.py",), checks=("tests/test_shipping.py",))],
        "remaining": ["shipping"], "final_check": "FAIL", "integration": "FAILED",
    }
    messages = _tool_group("fanout-call", "parallel", {})
    messages.extend(_tool_group("check-call", "pytest", {"target": "tests/test_shipping.py"}))
    context = {"journal_receipt_records": [{
        "kind": "check_receipt", "status": "PASS", "selector": "tests/test_shipping.py",
        "execution_complete": True,
    }]}

    repaired = continuation._lifecycle_evidence(disposition, messages, context)
    assert repaired["passed_checks"] == {"tests/test_shipping.py": 2}
    verified, pin, _events = continuation.render(
        {**disposition, "lifecycle": repaired}, None, budget=3072,
    )
    assert "remaining=[]" in verified
    assert pin == ""

    messages.extend(_tool_group("edit-call", "edit", {"path": "shop/shipping.py"}))
    stale = continuation._lifecycle_evidence(disposition, messages, context)
    assert stale["passed_checks"] == {}
    partial, _pin, _events = continuation.render(
        {**disposition, "lifecycle": stale}, None, budget=3072,
    )
    assert "remaining=[shipping]" in partial


def test_same_turn_owned_edit_keeps_passed_check_unresolved():
    disposition = {
        "kind": "protected_fanout_disposition", "call_id": "fanout-call", "turn_index": 0,
        "units": [_unit("shipping", status="FAILED", check_result="FAIL",
                         owns=("shop/shipping.py",), checks=("tests/test_shipping.py",))],
        "remaining": ["shipping"], "final_check": "FAIL", "integration": "FAILED",
    }
    messages = _tool_group("fanout-call", "parallel", {})
    messages.extend([
        {"role": "assistant", "tool_calls": [
            {"id": "check-call", "type": "function", "function": {
                "name": "pytest", "arguments": json.dumps({"target": "tests/test_shipping.py"})}},
            {"id": "edit-call", "type": "function", "function": {
                "name": "edit", "arguments": json.dumps({"path": "shop/shipping.py"})}},
        ]},
        {"role": "tool", "tool_call_id": "check-call", "content": "complete"},
        {"role": "tool", "tool_call_id": "edit-call", "content": "complete"},
    ])
    context = {"journal_receipt_records": [{
        "kind": "check_receipt", "status": "PASS", "selector": "tests/test_shipping.py",
        "execution_complete": True,
    }]}

    lifecycle = continuation._lifecycle_evidence(disposition, messages, context)

    assert lifecycle["passed_checks"] == {}


def test_unscoped_passing_receipt_cannot_resolve_a_unit():
    disposition = {
        "kind": "protected_fanout_disposition", "call_id": "fanout-call", "turn_index": 0,
        "units": [_unit("shipping", status="FAILED", check_result="FAIL",
                         owns=("shop/shipping.py",), checks=("tests/test_shipping.py",))],
        "remaining": ["shipping"], "final_check": "FAIL", "integration": "FAILED",
    }
    messages = _tool_group("fanout-call", "parallel", {})
    messages.extend(_tool_group("check-call", "pytest", {"target": "tests/test_shipping.py"}))

    lifecycle = continuation._lifecycle_evidence(
        disposition, messages,
        {"journal_receipt_records": [{"kind": "check_receipt", "status": "PASS"}]},
    )

    assert lifecycle["passed_checks"] == {}


def test_similar_but_distinct_check_path_does_not_resolve_a_unit():
    disposition = {
        "kind": "protected_fanout_disposition", "call_id": "fanout-call", "turn_index": 0,
        "units": [_unit("shipping", status="FAILED", check_result="FAIL",
                         owns=("shop/shipping.py",), checks=("tests/test_shipping.py2",))],
        "remaining": ["shipping"], "final_check": "FAIL", "integration": "FAILED",
    }
    messages = _tool_group("fanout-call", "parallel", {})
    messages.extend(_tool_group("check-call", "pytest", {"target": "tests/test_shipping.py2"}))
    context = {"journal_receipt_records": [{
        "kind": "check_receipt", "status": "PASS", "selector": "tests/test_shipping.py",
        "execution_complete": True,
    }]}

    lifecycle = continuation._lifecycle_evidence(disposition, messages, context)

    assert lifecycle["passed_checks"] == {}


@pytest.mark.parametrize(("tool_name", "arguments", "receipt_selector"), [
    ("pytest", {"target": "tests/test_shipping.py"}, "tests/test_shipping.py::test_weight"),
    ("run_tests", {"selectors": ["tests/test_shipping.py"]}, "tests/test_shipping.py::test_weight"),
    ("run_tests", {"args": "-q tests/test_shipping.py"}, "tests/test_shipping.py"),
])
def test_check_selector_path_and_nodeid_aliases_resolve(
        tool_name: str, arguments: dict[str, object], receipt_selector: str,
):
    disposition = {
        "kind": "protected_fanout_disposition", "call_id": "fanout-call", "turn_index": 0,
        "units": [_unit("shipping", status="FAILED", check_result="FAIL",
                         owns=("shop/shipping.py",), checks=("tests/test_shipping.py::test_weight",))],
        "remaining": ["shipping"], "final_check": "FAIL", "integration": "FAILED",
    }
    messages = _tool_group("fanout-call", "parallel", {})
    messages.extend(_tool_group("check-call", tool_name, arguments))
    context = {"journal_receipt_records": [{
        "kind": "check_receipt", "status": "PASS", "selector": receipt_selector,
        "execution_complete": True,
    }]}

    lifecycle = continuation._lifecycle_evidence(disposition, messages, context)

    assert lifecycle["passed_checks"] == {"tests/test_shipping.py::test_weight": 2}


@pytest.mark.parametrize(("tool_name", "arguments"), [
    ("write", {"path": "shop/shipping.py"}),
    ("write_file", {"path": "shop/shipping.py"}),
    ("apply_patch", {"path": "shop/shipping.py"}),
    ("insert_before", {"path": "shop/shipping.py"}),
    ("replace_module", {"path": "shop/shipping.py"}),
    ("replace_module", {}),
    ("file_delete", {"target": {"path": "shop/shipping.py"}}),
    ("file_move", {"source": "shop/shipping.py", "destination": "shop/archive.py"}),
])
def test_supported_mutations_after_passing_check_keep_unit_unresolved(
        tool_name: str, arguments: dict[str, object],
):
    disposition = {
        "kind": "protected_fanout_disposition", "call_id": "fanout-call", "turn_index": 0,
        "units": [_unit("shipping", status="FAILED", check_result="FAIL",
                         owns=("shop/shipping.py",), checks=("tests/test_shipping.py",))],
        "remaining": ["shipping"], "final_check": "FAIL", "integration": "FAILED",
    }
    messages = _tool_group("fanout-call", "parallel", {})
    messages.extend(_tool_group("check-call", "pytest", {"target": "tests/test_shipping.py"}))
    messages.extend(_tool_group("mutation-call", tool_name, arguments))
    context = {"journal_receipt_records": [{
        "kind": "check_receipt", "status": "PASS", "selector": "tests/test_shipping.py",
        "execution_complete": True,
    }]}

    lifecycle = continuation._lifecycle_evidence(disposition, messages, context)

    assert lifecycle["passed_checks"] == {}


def test_unscoped_single_failure_is_not_attached_to_multiple_units():
    disposition = {
        "units": [
            _unit("shipping", status="FAILED", check_result="FAIL"),
            _unit("pricing", status="FAILED", check_result="FAIL"),
        ],
        "failure": {"nodeid": "tests/test_shipping.py::test_price"},
    }

    assert continuation._failure_for("pricing", disposition) is None


def test_display_text_removes_control_characters_and_unsafe_paths():
    assert continuation._clean("safe\x1b[31mred\x07") == "safe[31mred"
    assert continuation._relative_path("src/\x1bsecret.py") is None


def test_recorded_failure_matches_test_file_before_node_selector(tmp_path: Path):
    units = [
        _unit("shipping", status="FAILED", check_result="FAIL",
              checks=("tests/test_shipping.py::test_weight",)),
        _unit("pricing", status="FAILED", check_result="FAIL",
              checks=("tests/test_pricing.py::test_price",)),
    ]

    failure = continuation_cli._failure_from_text(
        "_____ test_weight _____\ntests/test_shipping.py:4: AssertionError\nE assert 11 == 9\n",
        tmp_path, continuation._unit_rows({"units": units}),
    )

    assert failure is not None
    assert failure["unit_id"] == "shipping"
    assert failure["observed"] == "11"
    assert failure["expected"] == "9"


def test_ambiguous_recorded_failure_is_not_assigned_to_first_unit(tmp_path: Path):
    units = [
        _unit("shipping", status="FAILED", check_result="FAIL"),
        _unit("pricing", status="FAILED", check_result="FAIL"),
    ]

    failure = continuation_cli._failure_from_text(
        "_____ test_unknown _____\ntests/test_unknown.py:4: AssertionError\nE assert 11 == 9\n",
        tmp_path, continuation._unit_rows({"units": units}),
    )

    assert failure is not None
    assert failure["unit_id"] == "unknown"


def test_malformed_protected_disposition_fails_open_with_typed_event(monkeypatch):
    monkeypatch.setenv(continuation.SWITCH_NAME, "on")
    frame = {"protected_fanout_disposition": {
        "kind": "protected_fanout_disposition", "call_id": "bad-call",
        "units": [_unit("duplicate"), _unit("duplicate")],
        "final_check": "FAIL", "integration": "FAILED",
    }}

    events = continuation.attach_to_frame(
        frame, None, None, semantic_messages=_tool_group("read-call", "read", {}),
    )

    assert "fanout_continuation" not in frame
    assert "fanout_continuation_pin" not in frame
    assert events == [{"kind": "fanout_continuation_unavailable",
                       "reason": "malformed_protected_disposition"}]


@pytest.mark.parametrize("close_tool", ["parallel", "done"])
def test_closed_fanout_releases_all_continuation_fields(close_tool: str, monkeypatch):
    monkeypatch.setenv(continuation.SWITCH_NAME, "on")
    disposition = {
        "kind": "protected_fanout_disposition", "call_id": "fanout-call", "turn_index": 0,
        "units": [_unit("shipping", status="FAILED", check_result="FAIL",
                         owns=("shop/shipping.py",), checks=("tests/test_shipping.py",))],
        "remaining": ["shipping"], "final_check": "FAIL", "integration": "FAILED",
    }
    messages = _tool_group("fanout-call", "parallel", {})
    messages.extend(_tool_group("next-call", close_tool, {}))
    frame = {"protected_fanout_disposition": disposition}

    events = continuation.attach_to_frame(frame, {}, None, semantic_messages=messages)

    assert "fanout_continuation" not in frame
    assert "fanout_continuation_pin" not in frame
    assert any(event.get("kind") == "fanout_continuation_retired" for event in events)


def test_frame_without_parallel_disposition_preserves_serialized_request_bytes(monkeypatch, tmp_path: Path):
    from kernel_stateframe_capacity import fit_protected_frame
    from kernel_stateframe_framefit5 import VOLATILE_TAIL_MARKER, _serialize_state_message

    monkeypatch.setenv(continuation.SWITCH_NAME, "on")
    frame = {"schema": "stateframe-framefit5/1", "mode": "FRAME_ONLY",
             "action_ledger": {"reads": [], "edits": [], "unclassified_actions": []}}
    body = {"messages": [{"role": "user", "content": _serialize_state_message(
        {"state": {"framefit5": frame}})},
        {"role": "user", "content": VOLATILE_TAIL_MARKER}], "tools": []}
    before = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()

    fit_protected_frame(body, workspace_root=tmp_path, fanout_enabled=True)

    after = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert after == before


@pytest.mark.parametrize("shape", ["ordinary", "disposition", "retained_continuation"])
@pytest.mark.parametrize("switch", [None, "on", "off"])
def test_fanout_off_preserves_exact_request_bytes(shape, switch, monkeypatch, tmp_path):
    from kernel_stateframe_capacity import fit_protected_frame
    from kernel_stateframe_framefit5 import VOLATILE_TAIL_MARKER, _serialize_state_message

    if switch is None:
        monkeypatch.delenv(continuation.SWITCH_NAME, raising=False)
    else:
        monkeypatch.setenv(continuation.SWITCH_NAME, switch)
    frame = {"schema": "stateframe-framefit5/1", "mode": "FRAME_ONLY",
             "action_ledger": {"reads": [], "edits": [], "unclassified_actions": []}}
    if shape != "ordinary":
        frame["protected_fanout_disposition"] = _case("r7")
    if shape == "retained_continuation":
        frame["fanout_continuation"] = "retained section"
        frame["fanout_continuation_pin"] = "retained pin"
    body = {"messages": [{"role": "user", "content": _serialize_state_message(
        {"state": {"framefit5": frame}})},
        {"role": "user", "content": VOLATILE_TAIL_MARKER}], "tools": []}
    before = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()

    audit = fit_protected_frame(body, workspace_root=tmp_path)

    assert json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() == before
    assert audit["changed"] is False
    assert audit.get("fanout_continuation_events", []) == []


def test_fanout_off_skips_renderer_import(monkeypatch, tmp_path):
    import builtins
    from kernel_stateframe_capacity import fit_protected_frame
    from kernel_stateframe_framefit5 import VOLATILE_TAIL_MARKER, _serialize_state_message

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "kernel_fanout_continuation", "OFF fitting imported the fanout renderer"
        return original_import(name, *args, **kwargs)

    frame = {"schema": "stateframe-framefit5/1", "mode": "FRAME_ONLY",
             "action_ledger": {"reads": [], "edits": [], "unclassified_actions": []},
             "protected_fanout_disposition": _case("r7")}
    body = {"messages": [{"role": "user", "content": _serialize_state_message(
        {"state": {"framefit5": frame}})},
        {"role": "user", "content": VOLATILE_TAIL_MARKER}], "tools": []}
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    fit_protected_frame(body, workspace_root=tmp_path, fanout_enabled=False)


@pytest.mark.parametrize("module_name", ["kernel_parallel", "stateframe.engine.kernel_parallel"])
@pytest.mark.parametrize("mode", ["absent", "uninstalled", "auto", "force"])
def test_finalizer_fanout_gate_uses_installed_parent_service(module_name, mode, monkeypatch, tmp_path):
    import importlib
    import sys
    from types import SimpleNamespace
    import kernel_request_budget as request_budget
    import kernel_stateframe_capacity as capacity
    from kernel_frame_allocation_capture import finalize_frame_candidate
    from kernel_request_diet import RequestTokenMeter
    from kernel_stateframe_framefit5 import VOLATILE_TAIL_MARKER, _serialize_state_message

    service = importlib.import_module(module_name)
    monkeypatch.setattr(service, "_service", None)
    monkeypatch.delitem(sys.modules, "kernel_parallel", raising=False)
    monkeypatch.delitem(sys.modules, "stateframe.engine.kernel_parallel", raising=False)
    if mode != "absent":
        monkeypatch.setitem(sys.modules, module_name, service)
    if mode in {"auto", "force"}:
        service.install(tmp_path, {"parallel_mode": mode})
    monkeypatch.setenv(continuation.SWITCH_NAME, "on")
    task = {"role": "user", "content": "Keep the request stable."}
    frame = {"schema": "stateframe-framefit5/1", "mode": "FRAME_ONLY", "action_ledger": {}}
    body = {"messages": [task, {"role": "user", "content": _serialize_state_message(
        {"state": {"framefit5": frame}})},
        {"role": "user", "content": VOLATILE_TAIL_MARKER}], "tools": []}
    runtime = SimpleNamespace(frame_ceiling=8192, owner=SimpleNamespace(root=tmp_path))
    capture = SimpleNamespace(
        request_id="fanout-gate", _framefit5_semantic_messages=[task],
        _framefit5_task_message=task,
        _request_source_closure_override={"units": [], "paths": [], "deferred_paths": []},
    )
    meter = RequestTokenMeter(tokenizer=lambda _text: [1], template_overhead_tokens=0)
    meter.bind_request(capture.request_id, capture.request_id)
    monkeypatch.setattr(request_budget, "_request_token_meter", lambda *_args: meter)

    class GateChecked(Exception):
        pass

    def check_gate(_body, **kwargs):
        assert kwargs.get("fanout_enabled") is (mode in {"auto", "force"})
        raise GateChecked

    monkeypatch.setattr(capacity, "fit_protected_frame", check_gate)
    with pytest.raises(GateChecked):
        finalize_frame_candidate(runtime, capture, body, frame_counter=lambda _body: 1)


def test_capacity_pressure_keeps_typed_continuation_and_pin(tmp_path: Path):
    from kernel_stateframe_capacity import fit_protected_frame
    from kernel_stateframe_framefit5 import VOLATILE_TAIL_MARKER, _serialize_state_message, _state_payload

    source = tmp_path / "shop/shipping.py"
    test = tmp_path / "tests/test_shipping.py"
    source.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    source.write_text("def quote_weight(value):\n    return value * 2\n")
    test.write_text("from shop.shipping import quote_weight\n\ndef test_weight():\n    assert quote_weight(2) == 3\n")
    frame = {
        "schema": "stateframe-framefit5/1", "mode": "FRAME_ONLY",
        "action_ledger": {
            "reads": [], "edits": [],
            "unclassified_actions": [
                {"kind": "assistant_note", "turn_index": index, "text": "old " * 400,
                 "text_sha256": hashlib.sha256(("old " * 400).encode()).hexdigest()}
                for index in range(8)
            ],
        },
        "protected_fanout_disposition": {
            "kind": "protected_fanout_disposition", "call_id": "capacity-case", "turn_index": 8,
            "units": [_unit("shipping", status="FAILED", check_result="FAIL",
                             owns=("shop/shipping.py",), checks=("tests/test_shipping.py",),
                             changed_paths=("shop/shipping.py",))],
            "final_check": "FAIL", "integration": "FAILED", "remaining": ["shipping"],
            "failure": {"unit_id": "shipping", "nodeid": "tests/test_shipping.py::test_weight",
                        "location": "tests/test_shipping.py:4", "expected": "3", "observed": "4",
                        "assertion": "assert quote_weight(2) == 3"},
        },
    }
    body = {"messages": [{"role": "user", "content": _serialize_state_message({"state": {"framefit5": frame}})}],
            "tools": []}
    from kernel_stateframe_capacity import protected_bytes
    ceiling = protected_bytes(body) - 1200

    fit_protected_frame(body, max_request_bytes=ceiling, workspace_root=tmp_path,
                        fanout_enabled=True)

    final = _state_payload(body["messages"][0])[1]["framefit5"]
    assert "[FANOUT CONTINUATION]" in final["fanout_continuation"]
    assert "shop/shipping.py" in final["fanout_continuation_pin"]
    assert len(final["fanout_continuation_pin"].encode()) <= continuation.MAX_PIN_BYTES
    assert len(final["action_ledger"]["unclassified_actions"]) < 8


def test_disposition_conversion_keeps_every_recorded_outcome():
    request = {"units": [
        {"id": "ok", "owns": ["src/ok.py"], "check": {"selectors": ["tests/test_ok.py"]}},
        {"id": "crash", "owns": ["src/crash.py"], "check": {"selectors": ["tests/test_crash.py"]}},
    ]}
    result = {
        "call_id": "convert-call",
        "admission": {"status": "ADMITTED", "refusal": None},
        "outcomes": [
            {"unit_id": "ok", "status": "COMMITTED", "check": "PASS", "contract_ok": True,
             "files": [{"path": "src/ok.py"}], "usage": {"requests": 2}, "reason": None, "detail": None},
            {"unit_id": "crash", "status": "ENGINE_ERROR", "check": "NOT_RUN", "contract_ok": None,
             "files": [], "usage": {"requests": 0}, "reason": "ENGINE_ERROR", "detail": "worker exited"},
        ],
        "final_check": "FAIL", "integration": "FAILED", "observation": "ignored",
    }

    disposition = continuation.disposition_from_records(
        "convert-call", request, result, turn_index=4,
    )

    assert [unit["status"] for unit in disposition["units"]] == ["COMMITTED", "ENGINE_ERROR"]
    assert disposition["units"][1]["reason"] == "ENGINE_ERROR: worker exited"
    assert disposition["remaining"] == ["crash"]


def test_nested_protected_disposition_retains_outer_call_identity():
    nested = {"units": [_unit("shipping", status="FAILED", check_result="FAIL")],
              "final_check": "FAIL", "integration": "FAILED"}
    frame = {"action_ledger": {"protected_actions": [{
        "kind": "protected_fanout_disposition", "call_id": "outer-call", "turn_index": 12,
        "disposition": nested,
    }]}}

    disposition, unavailable = continuation._find_disposition(frame)

    assert not unavailable
    assert disposition is not None
    assert disposition["call_id"] == "outer-call"
    assert disposition["turn_index"] == 12
    assert disposition["units"] == nested["units"]


def test_duplicate_disposition_unit_ids_are_refused():
    disposition = {"units": [_unit("same"), _unit("same")]}

    with pytest.raises(continuation.FanoutContinuationError, match="duplicate subframe ids"):
        continuation.render(disposition, None, budget=3072)

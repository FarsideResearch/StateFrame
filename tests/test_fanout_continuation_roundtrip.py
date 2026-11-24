"""Parallel writer rows must round-trip into the continuation reader, and final checks never infer PASS."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

import kernel_fanout_continuation as continuation
import kernel_parallel
from kernel_stateframe_framefit5 import FrameFit5Error, _validated_parallel_disposition
from test_parallel_disposition import _parallel_record


def _write_workspace(root: Path) -> Path:
    workspace = root / "parent"
    (workspace / "shop").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "shop" / "shipping.py").write_text("def quote():\n    return 11\n", encoding="utf-8")
    (workspace / "tests" / "test_shipping.py").write_text(
        "def test_quote():\n    assert quote() == 9\n", encoding="utf-8",
    )
    return workspace


def _writer_item(tmp_path: Path, scenario: str, *, final_check: str | None = None,
                 integration: str | None = None) -> dict[str, object]:
    """Build the frame item through the production parallel writer."""
    result, request, _turns = _parallel_record(tmp_path, scenario)
    replacements = {}
    if final_check is not None:
        replacements["final_check"] = final_check
    if integration is not None:
        replacements["integration"] = integration
    if replacements:
        result = dataclasses.replace(result, **replacements)
    return kernel_parallel.render_parallel_disposition(result, request, journal=None)


def _render_through_runtime(item: dict[str, object], workspace: Path) -> tuple[dict[str, object], list]:
    """Attach the disposition the way the runtime does, from a frame that holds the writer row."""
    frame: dict[str, object] = {"action_ledger": {"unclassified_actions": [item]}}
    events = continuation.attach_to_frame(frame, None, workspace)
    return frame, events


def test_writer_rows_round_trip_into_continuation_with_pins(tmp_path):
    workspace = _write_workspace(tmp_path)
    item = _writer_item(tmp_path, "partial")

    frame, events = _render_through_runtime(item, workspace)

    assert events == []
    section = frame.get("fanout_continuation", "")
    assert "[FANOUT CONTINUATION]" in section
    assert "remaining=[shipping]" in section
    assert "final_check=FAIL; final_integration=FAILED" in section
    pin = frame.get("fanout_continuation_pin", "")
    assert "shop/shipping.py" in pin
    assert "return 11" in pin


def test_all_children_finished_but_final_check_failed_renders_failed(tmp_path):
    workspace = _write_workspace(tmp_path)
    item = _writer_item(tmp_path, "verified", final_check="FAIL", integration="FAILED")

    frame, events = _render_through_runtime(item, workspace)

    assert events == []
    section = frame.get("fanout_continuation", "")
    assert "final_check=FAIL; final_integration=FAILED" in section
    assert "final_check=PASS" not in section
    assert "integration=VERIFIED" not in section
    assert "all subframes committed and verified" not in section


def test_legacy_records_without_unit_scopes_still_render_without_pins(tmp_path):
    """Records written before unit_scopes existed keep rendering the recorded facts; pins degrade."""
    workspace = _write_workspace(tmp_path)
    item = _writer_item(tmp_path, "partial")
    section = item["sections"]["disposition"]
    section.pop("unit_scopes")
    section.pop("unit_scope_fields")

    frame, events = _render_through_runtime(item, workspace)

    # The only event is the typed pin degradation for the unit that has no recorded owned paths.
    assert [event["kind"] for event in events] == ["fanout_pin_degraded"]
    assert events[0]["unit_ids"] == ["shipping"]
    rendered = frame.get("fanout_continuation", "")
    assert "remaining=[shipping]" in rendered
    assert "final_check=FAIL; final_integration=FAILED" in rendered
    assert "shop/shipping.py" not in frame.get("fanout_continuation_pin", "")


def test_writer_scope_rows_cover_every_unit_and_pass_the_frame_validator(tmp_path):
    item = _writer_item(tmp_path, "partial")
    section = item["sections"]["disposition"]

    # Scope rows follow the detail order (failures first), so compare as sets of unit ids.
    assert {row[0] for row in section["unit_scopes"]} == {row[0] for row in section["units"]}
    assert len(section["unit_scopes"]) == len(section["units"])
    shipping = next(row for row in section["unit_scopes"] if row[0] == "shipping")
    assert shipping[1] == ["shop/shipping.py"]
    assert shipping[2] == ["tests/test_shipping.py"]
    assert _validated_parallel_disposition(item)


@pytest.mark.parametrize("mutation", ["owns_not_list", "unknown_unit", "wrong_width"])
def test_malformed_unit_scope_rows_are_refused_by_the_frame_validator(tmp_path, mutation):
    item = _writer_item(tmp_path, "partial")
    scopes = item["sections"]["disposition"]["unit_scopes"]
    if mutation == "owns_not_list":
        scopes[0][1] = "shop/invoicing.py"
    elif mutation == "unknown_unit":
        scopes[0][0] = "ghost"
    else:
        scopes[0].append("extra")
    with pytest.raises(FrameFit5Error, match="unit scope"):
        _validated_parallel_disposition(item)

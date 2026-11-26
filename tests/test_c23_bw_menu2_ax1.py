"""BW-MENU2: replay through send 2 and inspect the provider-visible menu."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from kernel_native_tools import NativeToolBatch
from stateframe.runner import resolve_opt_in_switches
from test_allocation_replay_corpus import CASE_METADATA, _run_case, _source_root
from test_stable_tool_head_e2e import _raw_tool_arrays
from tools.pinned_tree import pinned_tree

SWITCH = "STATEFRAME_COMPACT_BATCH_CHECK_RECEIPTS"
BASE_COMMIT = "6e3b096cf3059f759138a88f7fdf46acd3d75baa"
LEGACY_TAPES = ("legacy-b8-off", "legacy-v037-slot0")


@pytest.mark.host_sandbox
@pytest.mark.parametrize("case_id", LEGACY_TAPES)
def test_recorded_profile_hits_send_two_byte_wall_on_release(
    tmp_path: Path, case_id: str, tmp_path_factory, monkeypatch,
) -> None:
    monkeypatch.setenv("PINNED_TREE_REQUIRED", "1")
    source = pinned_tree(BASE_COMMIT, tmp_path_factory) / "src"
    result = _run_case(tmp_path, case_id, stop_after_provider_requests=2, source_root=source)
    assert result.endpoint_requests == result.served_steps == 1, result.evidence[-4000:]
    assert result.engine_result == "BLOCKED", result.evidence[-4000:]
    suspension = result.summary["engine_terminal"]["suspension"]
    assert suspension["due_ordinal"] == 2
    assert suspension["reason_code"] == "BYTE_POLICY_DOMAIN_EXHAUSTED"
    assert suspension["cap"] == 82_026
    assert suspension["floor_bytes"] > suspension["cap"]


@pytest.mark.host_sandbox
@pytest.mark.parametrize("case_id", LEGACY_TAPES)
def test_send_two_preserves_the_launch_menu(tmp_path: Path, case_id: str, monkeypatch) -> None:
    source = _source_root()
    registry = source / "stateframe" / "engine" / "switch-registry.json"
    # Replaying this assertion against 0.10.2 must reach the recorded byte wall.
    if SWITCH in json.loads(registry.read_bytes())["switches"]:
        overrides = CASE_METADATA[case_id]["profile_settings"]["switch_overrides"]
        monkeypatch.setitem(overrides, SWITCH, "on")
    result = _run_case(tmp_path, case_id, stop_after_provider_requests=2, source_root=source)
    assert result.endpoint_requests == result.served_steps == 2, (
        result.summary.get("engine_terminal") if result.summary else result.evidence[-4000:]
    )

    first, second = _raw_tool_arrays(tmp_path / "run")
    names = {schema["function"]["name"] for schema in json.loads(first)}
    assert {"run_tests", "read", "edit"} <= names
    assert second == first
    assert result.request_bytes[1] <= 82_026
    wire = tmp_path / "run" / "wire" / "0002.request.json"
    messages = json.loads(wire.read_bytes())["messages"]
    assert any("characters of check output omitted]" in (row.get("content") or "")
               for row in messages if row["role"] == "user")
    assert any("multi-selector oracle" in row["content"] and "AssertionError" in row["content"]
               for row in messages if row["role"] == "tool")


def test_long_check_reasons_keep_legacy_bytes_unless_enabled(monkeypatch) -> None:
    summary = "RESULT_RED: pytest exit 1"
    output = "AssertionError\n" * 180
    reason = f"{summary}\n{output}"
    receipt = {"batch_index": 0, "tool": "run_tests", "status": "RED", "reason": reason}
    header = "[NATIVE BATCH RECEIPTS]\nEvery queued call has an explicit host receipt.\n"
    legacy = f"{header}call 0 run_tests: RED(reason={reason})"
    for mode in (None, "off"):
        if mode is None:
            monkeypatch.delenv(SWITCH, raising=False)
        else:
            monkeypatch.setenv(SWITCH, mode)
        assert resolve_opt_in_switches().get(SWITCH) == mode
        assert NativeToolBatch.render_model_receipts([receipt]) == legacy

    monkeypatch.setenv(SWITCH, "on")
    assert resolve_opt_in_switches()[SWITCH] == "on"
    rendered = NativeToolBatch.render_model_receipts([receipt])
    assert rendered == (f"{header}call 0 run_tests: RED(reason={summary} "
                        f"[{len(output)} characters of check output omitted])")
    assert receipt["reason"] == reason


@pytest.mark.parametrize(("tool", "status", "size"), (
    ("run_tests", "RED", 1024),
    ("run_tests", "ERROR", 1025),
    ("edit", "REJECTED", 1025),
))
def test_compact_mode_keeps_other_receipt_details(monkeypatch, tool, status, size) -> None:
    reason = "result\n" + "x" * size
    receipt = {"batch_index": 3, "tool": tool, "status": status, "reason": reason}
    monkeypatch.setenv(SWITCH, "on")
    assert NativeToolBatch.render_model_receipts([receipt]).endswith(
        f"call 3 {tool}: {status}(reason={reason})"
    )

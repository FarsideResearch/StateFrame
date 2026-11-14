"""Source presentation gaps must never masquerade as exact edit anchors."""
from __future__ import annotations

import json
import urllib.request

import pytest

from acceptance._support import Endpoint
from kernel_context import token_estimate
from kernel_real04_bugs import (
    BOUNDED_SOURCE_HEADER, OMISSION_MARKER, _window_text,
    bound_working_source_message, verify_bounded_window,
)
from probes.react_repair import str_replace_edit


def _large_source() -> str:
    return "def target():\n" + "".join(
        f"    value_{index} = {index}\n" for index in range(500)
    ) + "    return value_499\n"


def test_scripted_anchor_spanning_rendered_marker(tmp_path):
    source = _large_source()
    path = tmp_path / "large.py"
    path.write_text(source)
    # Frozen pre-fix transcript: the old renderer cut through source lines.
    anchor = source[:303] + "\n" + OMISSION_MARKER + "\n" + source.rstrip("\n")[-303:]
    rendered = ("[Working source: large.py]\n" + BOUNDED_SOURCE_HEADER
                + " path='large.py' lines=1-502 reference='source:large.py'\n" + anchor)
    assert OMISSION_MARKER in anchor
    assert anchor not in source
    script = tmp_path / "reply.json"
    script.write_text(json.dumps({"steps": [{"tool": "str_replace", "arguments": {
        "path": "large.py", "old_str": anchor, "new_str": "def target():\n    return 1\n",
    }}]}))
    with Endpoint(script) as endpoint:
        request = urllib.request.Request(endpoint.url + "/chat/completions", data=json.dumps({
            "model": "scripted", "messages": [{"role": "user", "content": rendered}],
            "tools": [{"type": "function", "function": {"name": "str_replace"}}],
        }).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            reply = json.load(response)
    call = reply["choices"][0]["message"]["tool_calls"][0]["function"]
    assert call["name"] == "str_replace"
    arguments = json.loads(call["arguments"])
    assert arguments.pop("path") == "large.py"
    observation = str_replace_edit(path, **arguments)
    assert "REJECTED" in observation
    assert "the marker is not file content; re-read lines 1-502" in observation
    assert path.read_text() == source


def test_oversized_target_is_explicitly_read_only():
    source = _large_source()
    rendered = _window_text("large.py", source, "target", "source:large.py", 300)
    assert "read-only/non-editable" in rendered
    assert "re-read lines 1-502" in rendered
    assert OMISSION_MARKER not in rendered
    assert "def target():" not in rendered
    assert token_estimate({"role": "user", "content": rendered}) <= 300
    assert verify_bounded_window(rendered, path="large.py", source=source)
    assert not verify_bounded_window(rendered + "\nforged source", path="large.py", source=source)


def test_read_only_notice_fits_a_previously_admitted_small_budget():
    rendered = _window_text("large.py", _large_source(), "target", "source:large.py", 62)
    assert "read-only/non-editable" in rendered
    assert token_estimate({"role": "user", "content": rendered}) <= 62


@pytest.mark.parametrize("budget", [1, 50, 300])
def test_oversized_window_never_emits_partial_source_even_at_tiny_budgets(budget):
    source = _large_source()
    rendered = _window_text("large.py", source, "target", "source:large.py", budget)
    assert "read-only/non-editable" in rendered
    assert verify_bounded_window(rendered, path="large.py", source=source)


@pytest.mark.parametrize("labels", ["off", "plain"])
def test_fitting_windows_keep_exact_previous_bytes(monkeypatch, labels):
    from kernel_read_fixes import source_header
    monkeypatch.setenv("STATEFRAME_SOURCE_LABELS", labels)
    source = "# before\ndef target():\n    return 1\n# after\n"
    expected = (source_header("small.py", role="working") + "\n"
                + BOUNDED_SOURCE_HEADER + " path='small.py' lines=2-3 reference='source:small.py'\n"
                + "# before\ndef target():\n    return 1\n# after")
    assert _window_text("small.py", source, "target", "source:small.py", 1000) == expected
    assert verify_bounded_window(expected, path="small.py", source=source)
    assert not verify_bounded_window(expected + "altered", path="small.py", source=source)


def test_unchanged_message_preserves_identity(monkeypatch):
    monkeypatch.setenv("STATEFRAME_REAL04_WATCHDOG_SPAN", "on")
    message = {"role": "user", "content": "[Working source: small.py]\ndef target():\n    return 1\n"}
    assert bound_working_source_message(message, path="small.py", target_fn="target",
                                       reference="source:small.py", max_tokens=1000) is message


def test_bounding_required_source_preserves_retrieval_and_item_accounting(monkeypatch):
    from kernel_context import SelectableItem, SelectionPriority
    from kernel_measurement import canonical_bytes
    from kernel_real04_bugs import bound_required_source_items

    monkeypatch.setenv("STATEFRAME_REAL04_WATCHDOG_SPAN", "on")
    source = _large_source()
    message = {"role": "user", "content": "[Working source: large.py]\n" + source}
    item = SelectableItem("source:large.py", SelectionPriority.IMMEDIATE_WORKING_SET,
                          "source", canonical_bytes(message).decode(), token_estimate(message),
                          retrievable_ref="source:large.py")
    control = SelectableItem("control", SelectionPriority.REQUIRED_CONTROL_CONTEXT,
                             "control", "control", 20)
    items, messages = bound_required_source_items(
        (item, control), {"source:large.py": message}, path="large.py",
        target_fn="target", available_tokens=200)
    bounded = messages["source:large.py"]
    assert "read-only/non-editable" in bounded["content"]
    assert verify_bounded_window(bounded["content"], path="large.py", source=source)
    assert items[0].text == canonical_bytes(bounded).decode()
    assert items[0].token_estimate == token_estimate(bounded) <= 180
    assert items[0].retrievable_ref == item.retrievable_ref
    assert items[1] is control
    assert message["content"].endswith(source)


def test_tiny_window_without_any_inserted_marker_keeps_previous_bytes(monkeypatch):
    monkeypatch.setenv("STATEFRAME_SOURCE_LABELS", "off")
    assert _window_text("tiny.py", "VALUE = 1\n", "", "source:tiny.py", 1) == (
        "[Working source: tiny.py]\n" + BOUNDED_SOURCE_HEADER
        + " path='tiny.py' lines=1-1 reference='source:tiny.py'\n")


def test_verifier_refuses_legacy_editable_gap():
    source = _large_source()
    body = source[:303] + "\n" + OMISSION_MARKER + "\n" + source.rstrip("\n")[-303:]
    rendered = ("[Working source: large.py]\n" + BOUNDED_SOURCE_HEADER
                + " path='large.py' lines=1-502 reference='source:large.py'\n" + body)
    assert not verify_bounded_window(rendered, path="large.py", source=source)


MARKERS = (
    OMISSION_MARKER, "[... middle omitted ...]",
    "[LEAN tool output bounded; 50000 UTF-8 bytes, middle omitted]",
    "[observation excerpt] omitted_lines=12 omitted_bytes=120 sha256=abc",
    "[panel excerpt] omitted_bytes=120 sha256=abc",
    "[... draft truncated; recall with read draft:1]",
)


@pytest.mark.parametrize("marker", MARKERS)
@pytest.mark.parametrize("suffix", [".py", ".txt"])
def test_marker_guard_is_always_on_for_scalar_anchors(tmp_path, monkeypatch, marker, suffix):
    monkeypatch.setenv("STATEFRAME_EDIT_GUARD", "off")
    monkeypatch.setenv("STATEFRAME_REAL04_WATCHDOG_SPAN", "off")
    path = tmp_path / ("source" + suffix)
    source = "first\nsecond\nthird\n"
    path.write_text(source)
    result = str_replace_edit(path, "first\n" + marker + "\nthird", "replacement")
    assert "the marker is not file content; re-read lines 1-3" in result
    assert path.read_text() == source


def test_marked_anchor_noop_still_explains_the_marker(tmp_path):
    path = tmp_path / "value.py"
    path.write_text("VALUE = 1\n")
    result = str_replace_edit(path, OMISSION_MARKER, OMISSION_MARKER)
    assert "REJECTED" in result
    assert "the marker is not file content; re-read lines 1-1" in result
    assert path.read_text() == "VALUE = 1\n"


def test_read_only_notice_cannot_be_used_as_an_edit_anchor(tmp_path):
    source = _large_source()
    path = tmp_path / "large.py"
    path.write_text(source)
    notice = _window_text("large.py", source, "target", "source:large.py", 62).split("\n", 2)[2]
    result = str_replace_edit(path, notice, "def target():\n    return 1\n")
    assert "the marker is not file content; re-read lines 1-502" in result
    assert path.read_text() == source


def test_batch_marker_anchor_rejected_before_any_writes(tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import apply_batch
    monkeypatch.setenv("STATEFRAME_TOKENLAB_ATOMIC_MULTI_FILE_APPLY", "on")
    path = tmp_path / "value.py"
    source = "VALUE = 1\n"
    path.write_text(source)
    result = apply_batch(tmp_path, [
        {"path": "value.py", "old": source, "new": "VALUE = 2\nOTHER = 0\n"},
        {"path": "value.py", "old": OMISSION_MARKER, "new": "VALUE = 3\n"},
    ])
    assert "the marker is not file content; re-read lines 1-1" in str(result)
    assert path.read_text() == source


def test_typed_plan_legacy_anchor_guard(tmp_path):
    from kernel_revision_bound_edit import EditAuthority, EditPlan
    source = b"VALUE = 1\n"
    authority = EditAuthority(root=tmp_path, sources={"value.py": source},
                              base_revision="base", authority_ref="authority",
                              allowed_paths=("value.py",))
    plan = EditPlan.from_mapping({"base_revision": "base", "authority_ref": "authority",
                                 "inventory_ref": "inventory",
                                 "edits": [{"path": "value.py", "old_str": OMISSION_MARKER,
                                            "new_str": "VALUE = 2\n"}]})
    result = authority.admit(plan)
    assert not result.accepted
    assert "the marker is not file content; re-read lines 1-1" in str(result)


PARTIAL_MARKERS = (
    OMISSION_MARKER[4:-4],
    "[... middle omitted",
    OMISSION_MARKER.replace("middle lines", "middle\nlines"),
    "[Source window read-only/non-editable;",
    "[LEAN tool output bounded; 50000 UTF-8 bytes,\nmiddle omitted]",
)


def _replace(route, path, old, new):
    if route == "scalar":
        return str_replace_edit(path, old, new)
    from kernel_jq_finish import validated_str_replace
    return validated_str_replace(path, old, new)


@pytest.mark.parametrize("route", ["scalar", "jq"])
@pytest.mark.parametrize("marker", (*MARKERS, *PARTIAL_MARKERS))
def test_partial_and_complete_markers_explain_recovery(tmp_path, monkeypatch, route, marker):
    monkeypatch.setenv("STATEFRAME_JQ_FINISH", "on" if route == "jq" else "off")
    monkeypatch.setenv("STATEFRAME_EMPIRICAL_FIXES", "on")
    path = tmp_path / "value.py"
    source = "VALUE = 1\nOTHER = 2\n"
    path.write_text(source)
    result = _replace(route, path, marker, "VALUE = 3\n")
    assert "the marker is not file content; re-read lines 1-2" in result
    assert "candidate_id=" not in result
    assert path.read_text() == source


@pytest.mark.parametrize("route", ["scalar", "jq", "batch", "typed"])
@pytest.mark.parametrize("marker", (*MARKERS, *PARTIAL_MARKERS))
def test_genuine_marker_literals_remain_editable(tmp_path, monkeypatch, route, marker):
    source = "MARKER = " + repr(marker) + "\nVALUE = 1\n"
    # The anchor includes literal source, as in kernel_real04_bugs. A split
    # marker also occurs in a real triple-quoted string and must stay editable.
    if "\n" in marker:
        source = 'MARKER = """' + marker + '"""\nVALUE = 1\n'
    path = tmp_path / "value.py"
    path.write_text(source)
    old = source.split("\nVALUE", 1)[0]
    new = old.replace("MARKER", "NOTICE", 1)
    if route in {"scalar", "jq"}:
        result = _replace(route, path, old, new)
        assert "applied" in result, result
        assert path.read_text() == source.replace(old, new, 1)
    elif route == "batch":
        from kernel_coordinated_change_ledger import apply_batch
        monkeypatch.setenv("STATEFRAME_TOKENLAB_ATOMIC_MULTI_FILE_APPLY", "on")
        result = apply_batch(tmp_path, [{"path": "value.py", "old": old, "new": new}])
        assert path.read_text() == source.replace(old, new, 1), result
    else:
        from kernel_revision_bound_edit import EditAuthority, EditPlan
        authority = EditAuthority(root=tmp_path, sources={"value.py": source.encode()},
                                  base_revision="base", authority_ref="authority",
                                  allowed_paths=("value.py",))
        plan = EditPlan.from_mapping({"base_revision": "base", "authority_ref": "authority",
                                     "inventory_ref": "inventory", "edits": [
                                         {"path": "value.py", "old_str": old, "new_str": new}]})
        assert authority.admit(plan).accepted


@pytest.mark.parametrize("marker", PARTIAL_MARKERS)
def test_partial_markers_rejected_by_typed_admission(tmp_path, marker):
    from kernel_revision_bound_edit import EditAuthority, EditPlan
    authority = EditAuthority(root=tmp_path, sources={"value.py": b"VALUE = 1\n"},
                              base_revision="base", authority_ref="authority",
                              allowed_paths=("value.py",))
    plan = EditPlan.from_mapping({"base_revision": "base", "authority_ref": "authority",
                                 "inventory_ref": "inventory", "edits": [
                                     {"path": "value.py", "old_str": marker, "new_str": "VALUE = 2\n"}]})
    result = authority.admit(plan)
    assert not result.accepted
    assert "the marker is not file content; re-read lines 1-1" in str(result)


@pytest.mark.parametrize("marker", PARTIAL_MARKERS)
def test_partial_markers_reject_atomic_batch(tmp_path, monkeypatch, marker):
    from kernel_coordinated_change_ledger import apply_batch
    monkeypatch.setenv("STATEFRAME_TOKENLAB_ATOMIC_MULTI_FILE_APPLY", "on")
    path = tmp_path / "value.py"
    path.write_text("VALUE = 1\n")
    result = apply_batch(tmp_path, [
        {"path": "value.py", "old": "VALUE = 1", "new": "VALUE = 2\nOTHER = 0"},
        {"path": "value.py", "old": marker, "new": "VALUE = 3"},
    ])
    assert "the marker is not file content; re-read lines 1-1" in str(result)
    assert path.read_text() == "VALUE = 1\n"


def test_watchdog_off_keeps_oversized_message_identity(monkeypatch):
    monkeypatch.setenv("STATEFRAME_REAL04_WATCHDOG_SPAN", "off")
    message = {"role": "user", "content": "[Working source: large.py]\n" + _large_source()}
    assert bound_working_source_message(message, path="large.py", target_fn="target",
                                       reference="source:large.py", max_tokens=1) is message


def test_batch_can_edit_a_literal_introduced_in_its_overlay(tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import apply_batch
    monkeypatch.setenv("STATEFRAME_TOKENLAB_ATOMIC_MULTI_FILE_APPLY", "on")
    path = tmp_path / "value.py"
    source = "VALUE = 1\n"
    introduced = "MARKER = " + repr(OMISSION_MARKER) + "\n"
    path.write_text(source)
    result = apply_batch(tmp_path, [
        {"path": "value.py", "old": source, "new": introduced},
        {"path": "value.py", "old": OMISSION_MARKER, "new": "literal updated"},
    ])
    assert path.read_text() == introduced.replace(OMISSION_MARKER, "literal updated"), result


def test_batch_removed_marker_uses_disk_recovery_range(tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import apply_batch
    monkeypatch.setenv("STATEFRAME_TOKENLAB_ATOMIC_MULTI_FILE_APPLY", "on")
    path = tmp_path / "value.py"
    source = "MARKER = " + repr(OMISSION_MARKER) + "\n"
    path.write_text(source)
    result = apply_batch(tmp_path, [
        {"path": "value.py", "old": source, "new": "VALUE = 1\nOTHER = 2\n"},
        {"path": "value.py", "old": OMISSION_MARKER, "new": "literal updated"},
    ])
    assert "the marker is not file content; re-read lines 1-1" in str(result)
    assert path.read_text() == source


def test_literal_marker_noop_retains_batch_noop_contract(tmp_path):
    from kernel_coordinated_change_ledger import validate_batch
    source = "MARKER = " + repr(OMISSION_MARKER) + "\n"
    (tmp_path / "value.py").write_text(source)
    result = validate_batch(tmp_path, [
        {"path": "value.py", "old": OMISSION_MARKER, "new": OMISSION_MARKER},
    ])
    assert not result.valid
    assert result.failures[0].code == "NO_OP"


def test_typed_legacy_matching_still_uses_exact_raw_bytes(tmp_path):
    from kernel_revision_bound_edit import EditAuthority, EditPlan
    authority = EditAuthority(root=tmp_path, sources={"value.py": b"VALUE = 1\r\nOTHER = 2\r\n"},
                              base_revision="base", authority_ref="authority",
                              allowed_paths=("value.py",))
    plan = EditPlan.from_mapping({"base_revision": "base", "authority_ref": "authority",
                                 "inventory_ref": "inventory", "edits": [
                                     {"path": "value.py", "old_str": "VALUE = 1\nOTHER = 2\n",
                                      "new_str": "VALUE = 3\n"}]})
    result = authority.admit(plan)
    assert not result.accepted
    assert "MATCH[0]" in str(result)

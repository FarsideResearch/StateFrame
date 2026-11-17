"""STATEFRAME_EDIT_PROTOCOL selects one edit interface per run; legacy bytes stay put."""
from __future__ import annotations

import copy
import json
import re

import pytest

from kernel_edit_protocol_switch import MODE_ENV, handles_enabled, mode
from kernel_edit_source_tool_surface import (
    EDIT_SOURCE_SCHEMA, HIDDEN_LEGACY_TOOLS, REWRITE_SOURCE_SCHEMA, tool_surface)
from kernel_edit_tool_names import EDIT_SOURCE_TOOL, REWRITE_SOURCE_TOOL

HANDLE_PATTERN = re.compile(r"\bq\d+h\d+\b")


def _names(tools):
    return [tool["function"]["name"] for tool in tools]


def _menus(react_repair):
    menus = []
    for gate in (False, True):
        for probe in (False, True):
            menus.append(react_repair._tools_for_probe_modes(
                probe_gate_on=gate, observation_probe_on=probe))
    return menus


def test_switch_defaults_to_legacy_safe_and_rejects_junk(monkeypatch):
    monkeypatch.delenv(MODE_ENV, raising=False)
    assert mode() == "legacy_safe" and not handles_enabled()
    monkeypatch.setenv(MODE_ENV, " Handles ")
    assert handles_enabled()
    monkeypatch.setenv(MODE_ENV, "on")
    with pytest.raises(ValueError, match=MODE_ENV):
        mode()


def test_legacy_safe_menu_is_byte_identical_to_base(monkeypatch):
    from probes import react_repair
    monkeypatch.delenv(MODE_ENV, raising=False)
    plain = [json.dumps(m, sort_keys=True) for m in _menus(react_repair)]
    monkeypatch.setenv(MODE_ENV, "legacy_safe")
    explicit = [json.dumps(m, sort_keys=True) for m in _menus(react_repair)]
    assert plain == explicit
    for serialized in plain:
        assert '"name": "str_replace"' in serialized
        assert EDIT_SOURCE_TOOL not in serialized and REWRITE_SOURCE_TOOL not in serialized


def test_tool_surface_is_identity_object_when_legacy(monkeypatch):
    monkeypatch.delenv(MODE_ENV, raising=False)
    tools = [{"type": "function", "function": {"name": "str_replace"}}]
    assert tool_surface(tools) is tools


def test_handles_menu_has_exactly_the_two_protocol_tools_and_no_legacy_editors(monkeypatch):
    from probes import react_repair
    monkeypatch.setenv(MODE_ENV, "handles")
    for menu in _menus(react_repair):
        names = _names(menu)
        assert names.count(EDIT_SOURCE_TOOL) == 1 and names.count(REWRITE_SOURCE_TOOL) == 1
        assert not HIDDEN_LEGACY_TOOLS & set(names), names
        assert "read" in names and "grep" in names  # observation tools survive
        serialized = json.dumps(menu, sort_keys=True)
        assert HANDLE_PATTERN.search(serialized) is None
        assert "[CURRENT SOURCE]" not in serialized


def test_protocol_schemas_are_static_and_well_formed():
    for schema in (EDIT_SOURCE_SCHEMA, REWRITE_SOURCE_SCHEMA):
        serialized = json.dumps(schema, sort_keys=True)
        assert HANDLE_PATTERN.search(serialized) is None
        assert schema["function"]["parameters"]["type"] == "object"
    edits = EDIT_SOURCE_SCHEMA["function"]["parameters"]["properties"]["edits"]
    assert edits["minItems"] == 1 and edits["maxItems"] == 4
    assert set(edits["items"]["required"]) == {"handle", "old_text", "new_text"}
    assert set(REWRITE_SOURCE_SCHEMA["function"]["parameters"]["required"]) == {"handle", "new_text"}


def test_tool_surface_never_mutates_its_input(monkeypatch):
    monkeypatch.setenv(MODE_ENV, "handles")
    tools = [{"type": "function", "function": {"name": n}} for n in ("read", "edit", "str_replace")]
    frozen = copy.deepcopy(tools)
    out = tool_surface(tools)
    assert tools == frozen and out is not tools
    assert _names(out) == ["read", EDIT_SOURCE_TOOL, REWRITE_SOURCE_TOOL]

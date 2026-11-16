"""Every classifier that keys on edit-tool names admits the protocol tools."""
from __future__ import annotations

from kernel_bounded_locate import _EDIT_TOOLS as LOCATE_TEXT_EDITS
from kernel_bounded_locate_code import EDIT_TOOLS as LOCATE_EDITS
from kernel_common_sense_governor import _EDIT_KINDS, _OBSERVATION_KINDS
from kernel_continuation_menu_authority import EDIT_TOOLS, EFFECT_TOOLS
from kernel_edit_tool_names import DESTRUCTIVE_EDIT_TOOLS, PROTOCOL_EDIT_TOOL_NAMES
from kernel_stale_edit_policy import STALE_REPAIR_TOOLS, allows_stale_repair
from probes.react_repair import SCOPED_WRITE_TOOLS


def test_protocol_tools_are_edits_everywhere_names_are_classified():
    for registry in (EDIT_TOOLS, EFFECT_TOOLS, STALE_REPAIR_TOOLS, _EDIT_KINDS,
                     _OBSERVATION_KINDS, LOCATE_EDITS, LOCATE_TEXT_EDITS, SCOPED_WRITE_TOOLS):
        assert PROTOCOL_EDIT_TOOL_NAMES <= registry, registry


def test_legacy_members_are_untouched():
    assert {"edit", "str_replace", "insert_before", "shell", "parallel"} <= EDIT_TOOLS
    assert {"edit", "str_replace", "file_create", "work"} <= _EDIT_KINDS
    assert {"edit", "str_replace", "insert_before", "replace_module",
            "file_create", "file_delete", "file_move"} <= SCOPED_WRITE_TOOLS
    assert "shell" not in SCOPED_WRITE_TOOLS


def test_stale_repair_policy_treats_protocol_tools_like_str_replace():
    for environ in ({"KERNEL_STALE_EDIT_REPAIR": "1"}, {}):
        assert (allows_stale_repair("edit_source", environ=environ)
                == allows_stale_repair("str_replace", environ=environ))


def test_rewrite_source_is_destructive_and_edit_source_is_not():
    assert "rewrite_source" in DESTRUCTIVE_EDIT_TOOLS
    assert "edit_source" not in DESTRUCTIVE_EDIT_TOOLS

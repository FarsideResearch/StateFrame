"""A file operation must remain reachable through every request gate."""
import hashlib
import json
from pathlib import Path

from kernel_bounded_locate_code import EDIT_TOOLS as LOCATE_EDITS
from kernel_capability_floor import CAPABILITY_NAMES
from kernel_completion_authority import ACTION_RESULT_EFFECTS
from kernel_completion_owner import _PRODUCT_EDIT_TOOLS, TOOL_SPECS
from kernel_continuation_evidence import EXECUTION_RECEIPT_CONTRACTS
from kernel_continuation_menu_authority import EDIT_TOOLS as CONTINUATION_MENU_EDITS
from kernel_continuation_runtime import EFFECT_TOOLS
from kernel_frame_budget import _EDIT_TOOLS as FRAME_EDITS
from kernel_greenlock_menu_authority import EDIT_ACTIONS
from kernel_lean import PLAIN_TOOL_NAMES, _KNOWN_TOOL_NAMES
from kernel_post_edit_refresh import _EDIT_KINDS as POST_EDIT_KINDS
from kernel_post_green_menu import WORK_ACTIONS
from kernel_request_tool_surface import _REQUEST_TOOL_FLOOR
from kernel_stateframe_f4 import _EDIT_TOOL_NAMES
from kernel_tool_surface import dispatch_admissible
from probes.react_repair import TOOLS


def test_file_operation_registries():
    names = {"file_delete", "file_move"}
    advertised = {schema["function"]["name"] for schema in TOOLS}
    for registry in (
        advertised, set(TOOL_SPECS), set(EXECUTION_RECEIPT_CONTRACTS),
        CONTINUATION_MENU_EDITS, EFFECT_TOOLS, FRAME_EDITS, EDIT_ACTIONS,
        LOCATE_EDITS, PLAIN_TOOL_NAMES, _KNOWN_TOOL_NAMES,
        set(_REQUEST_TOOL_FLOOR), CAPABILITY_NAMES, _PRODUCT_EDIT_TOOLS,
        _EDIT_TOOL_NAMES, POST_EDIT_KINDS, WORK_ACTIONS,
        set(ACTION_RESULT_EFFECTS),
    ):
        assert names <= registry, registry
    assert all(dispatch_admissible(name) for name in names)


def test_file_operation_is_in_release_manifest():
    engine = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
    manifest = json.loads((engine / "release-manifest.json").read_text())
    source = engine / "kernel_file_ops.py"
    assert manifest["files"][source.name] == hashlib.sha256(source.read_bytes()).hexdigest()

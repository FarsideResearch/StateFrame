"""The foreground tool must survive every mutation and request registry."""
from kernel_bounded_locate_code import EDIT_TOOLS as LOCATE_EDITS
from kernel_capability_floor import CAPABILITY_NAMES
from kernel_completion_authority import ACTION_RESULT_EFFECTS
from kernel_completion_owner import _PRODUCT_EDIT_TOOLS, TOOL_SPECS
from kernel_continuation_evidence import EXECUTION_RECEIPT_CONTRACTS, READ_ONLY
from kernel_continuation_menu_authority import EDIT_TOOLS as MENU_EDITS
from kernel_continuation_runtime import EFFECT_TOOLS
from kernel_frame_budget import _EDIT_TOOLS, _CHECK_TOOLS
from kernel_greenlock_menu_authority import EDIT_ACTIONS
from kernel_lean import PLAIN_TOOL_NAMES, _KNOWN_TOOL_NAMES
from kernel_post_edit_refresh import _EDIT_KINDS
from kernel_post_green_menu import WORK_ACTIONS
from kernel_request_tool_surface import _REQUEST_TOOL_FLOOR
from kernel_stateframe_f4 import _EDIT_TOOL_NAMES
from probes.react_repair import TOOLS


def test_shell_mutation_registries():
    for registry in (
        {s["function"]["name"] for s in TOOLS}, TOOL_SPECS, EXECUTION_RECEIPT_CONTRACTS,
        MENU_EDITS, EFFECT_TOOLS, _EDIT_TOOLS, EDIT_ACTIONS, LOCATE_EDITS,
        PLAIN_TOOL_NAMES, _KNOWN_TOOL_NAMES, _REQUEST_TOOL_FLOOR,
        CAPABILITY_NAMES, _PRODUCT_EDIT_TOOLS, _EDIT_TOOL_NAMES, WORK_ACTIONS,
        ACTION_RESULT_EFFECTS,
    ):
        assert "shell" in registry, registry
    assert "shell_effect" in _EDIT_KINDS
    assert "shell" not in _CHECK_TOOLS | READ_ONLY
    assert "tests_passed" not in ACTION_RESULT_EFFECTS["shell"]


def test_shell_schema_matches_completion_owner():
    from kernel_shell_tool import SHELL_SCHEMA
    from kernel_completion_owner import canonicalize_tool_surface
    canonical = canonicalize_tool_surface([SHELL_SCHEMA])
    assert canonical[0]["function"]["parameters"]["required"] == ["command"]
    props = canonical[0]["function"]["parameters"]["properties"]
    assert props["timeout_seconds"]["minimum"] == 1
    assert props["timeout_seconds"]["maximum"] == 600
    assert props["background"] == {"type": "boolean"}


def test_plain_shell_schema_and_host_agree():
    from kernel_plain_host import PlainHost
    from kernel_plain_loop import TOOLS as PLAIN_TOOLS
    assert "shell" in PlainHost.SUPPORTED
    assert "shell" in {s["function"]["name"] for s in PLAIN_TOOLS}


def test_shell_module_is_release_declared_and_hashed():
    import hashlib
    import json
    from pathlib import Path
    from kernel_release_switch_registry import _declared_python_files
    engine = Path(__file__).resolve().parents[1] / "src/stateframe/engine"
    source = engine / "kernel_shell_tool.py"
    assert source.name in {p.name for p in _declared_python_files(engine)}
    manifest = json.loads((engine / "release-manifest.json").read_text())
    assert manifest["files"][source.name] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_agent_observation_registries_do_not_grant_check_or_edit_credit():
    from kernel_agent_schemas import OBSERVATION_NAMES, BUILTIN_SCHEMAS
    from kernel_continuation_evidence import has_execution_receipt_contract
    for name in OBSERVATION_NAMES:
        assert name in TOOL_SPECS
        assert has_execution_receipt_contract(name)
        assert name in PLAIN_TOOL_NAMES and name in _KNOWN_TOOL_NAMES
        assert name in EFFECT_TOOLS
        assert name not in _EDIT_TOOLS | _CHECK_TOOLS | _PRODUCT_EDIT_TOOLS
    assert {s["function"]["name"] for s in BUILTIN_SCHEMAS} == OBSERVATION_NAMES


def test_per_run_registration_menu_and_admission(host):
    from kernel_agent_tools import project_menu
    from kernel_tool_surface import dispatch_admissible
    from kernel_frame_budget import compact_tool_surface
    hub, _, _ = host
    hub.shell.execute({"command": "echo ready", "background": True}, "launch")
    for gear in ("g0", "g1", "g2", "g3"):
        selected = compact_tool_surface(project_menu([]), gear, truth={"green": True})
        names = {s["function"]["name"] for s in selected}
        assert {"job_wait", "job_output", "job_stop", "web_fetch"} <= names
        assert all(dispatch_admissible(name) for name in names)

from test_agent_tools import host  # noqa: E402,F401

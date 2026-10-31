"""Product tools share intention-state permissions, including stale repair."""
from types import SimpleNamespace

import pytest

from kernel_continuation_menu_authority import EDIT_TOOLS, state_allow_set, per_step_allow_set
from kernel_continuation_runtime import EFFECT_TOOLS, LiveContinuation
from kernel_stale_edit_policy import STALE_REPAIR_TOOLS
from test_agent_tools import host as host


@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("repair", ["0", "1"])
def test_live_intention_menu_product_tools(host, monkeypatch, stale, repair):
    import kernel_shell_tool
    hub, _, _ = host
    monkeypatch.setattr(kernel_shell_tool, "_active", hub.shell)
    monkeypatch.setenv("KERNEL_STALE_EDIT_REPAIR", repair)
    state = SimpleNamespace(active_id="i1", effects={}, intentions={
        "i1": SimpleNamespace(phase="active", stale_dependencies=("source",) if stale else ())})
    runtime = SimpleNamespace()
    live = LiveContinuation.__new__(LiveContinuation)
    live.runtime = runtime
    live.source_capture = None
    live.engine = SimpleNamespace(state=state)
    runtime.continuation = live
    allowed = per_step_allow_set(runtime)
    if not stale or repair == "1":
        assert EDIT_TOOLS <= allowed
    else:
        assert not EDIT_TOOLS & allowed
    assert "web_fetch" in allowed


def test_every_continuation_edit_registry_tracks_product_tools():
    from kernel_continuation_journal import _EDIT_TOOL_NAMES
    # These are the state-dependent sets that admit product edits. Compare the
    # whole registry so adding the next editor cannot leave stale repair behind.
    for registry in (STALE_REPAIR_TOOLS, EFFECT_TOOLS, _EDIT_TOOL_NAMES):
        assert "edit" in registry
        assert EDIT_TOOLS <= registry


@pytest.mark.parametrize("phase", ["active", "completed"])
def test_jobs_and_observations_survive_intention_states(host, phase):
    hub, _, _ = host
    hub.shell.execute({"command": "echo ready", "background": True}, "launch")
    state = SimpleNamespace(active_id="i1", effects={}, intentions={
        "i1": SimpleNamespace(phase=phase)})
    assert {"job_wait", "job_output", "job_stop", "web_fetch"} <= state_allow_set(state)


def test_custom_tools_and_job_cleanup_are_available_during_stale_repair(host, monkeypatch):
    from kernel_custom_tools import load_tool_files
    from kernel_stale_edit_policy import allows_stale_repair
    from pathlib import Path
    hub, _, _ = host
    definitions = load_tool_files([Path(__file__).parent / "acceptance/fixtures/tools.json"],
                                 builtin_names=frozenset(), shell_network="loopback")
    import kernel_agent_tools
    hub = kernel_agent_tools.AgentTools(hub.shell, definitions)
    monkeypatch.setattr(kernel_agent_tools, "_active", hub)
    monkeypatch.setenv("KERNEL_STALE_EDIT_REPAIR", "1")
    for phase in ("active", "completed"):
        state = SimpleNamespace(active_id="i", intentions={"i": SimpleNamespace(phase=phase)}, effects={})
        assert set(hub.custom) <= state_allow_set(state)
    assert all(allows_stale_repair(name) for name in hub.custom)
    monkeypatch.setenv("KERNEL_STALE_EDIT_REPAIR", "0")
    assert allows_stale_repair("job_stop")
    assert not allows_stale_repair("generate_tool")

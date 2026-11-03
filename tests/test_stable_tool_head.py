"""Stable tool head: one append-only provider tool list per run, dispatch limited to the callable menu."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

import kernel_agent_tools as agent_tools
import kernel_stable_tool_head as head
from kernel_completion_owner import callable_tool_schemas, validate_tool_arguments


def tool(name: str, description: str = "d", **properties: dict) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {"why": {"type": "string"}},
                       "required": []}}}


def names(tools: list[dict]) -> list[str]:
    return [item["function"]["name"] for item in tools]


def test_mode_defaults_on_when_unset(monkeypatch):
    monkeypatch.delenv(head.MODE_ENV, raising=False)
    assert head.mode() == "on"
    assert head.enabled()


def test_mode_off_disables_stable_tool_head(monkeypatch):
    monkeypatch.setenv(head.MODE_ENV, "off")
    assert head.mode() == "off"
    assert not head.enabled()


def test_mode_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv(head.MODE_ENV, "yes")
    with pytest.raises(ValueError):
        head.mode()


def test_runner_forwards_shell_off_into_run_spec(tmp_path, monkeypatch):
    from stateframe.endpoint import Endpoint
    from stateframe.runner import RunOptions, prepare

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text("def test_sample():\n    assert True\n")
    monkeypatch.setenv(head.MODE_ENV, "off")
    endpoint = Endpoint.from_args("http://127.0.0.1:8080/v1", "scripted", None)
    _run_dir, _copy, spec = prepare(RunOptions(
        repo=repo, task="inspect sample", endpoint=endpoint, run_dir=tmp_path / "run"))
    assert spec["environment"][head.MODE_ENV] == "off"


def test_listed_tools_grow_append_only_and_never_reorder_across_phases():
    runtime = SimpleNamespace()
    first = head.listed_tools(runtime, [tool("read"), tool("edit"), tool("done")])
    # A narrower, reordered phase menu keeps the same listed bytes.
    narrowed = head.listed_tools(runtime, [tool("done"), tool("read")])
    assert json.dumps(narrowed) == json.dumps(first)
    # A new tool is appended, so everything before it keeps its bytes.
    grown = head.listed_tools(runtime, [tool("authored_check"), tool("edit")])
    assert names(grown) == ["read", "edit", "done", "authored_check"]
    assert json.dumps(grown[:3]) == json.dumps(first)


def test_seed_uses_only_offerable_tools_after_current_menu_in_deterministic_order():
    runtime = SimpleNamespace()
    current = [tool("read"), tool("edit")]
    offerable = [tool("skill_load"), tool("authored_check"), tool("read"), tool("edit")]

    seeded = head.listed_tools(runtime, current, offerable)

    assert names(seeded) == ["read", "edit", "authored_check", "skill_load"]
    assert seeded[:2] == current


def test_seed_uses_filtered_run_registry_and_does_not_reseed_later(monkeypatch):
    monkeypatch.setenv(head.MODE_ENV, "on")

    class RunTools:
        names = frozenset({"skill_load", "web_fetch", "git_commit", "mcp_disabled", "custom_disabled"})

        @staticmethod
        def schemas():
            return [tool("custom_enabled")]

    monkeypatch.setattr(agent_tools, "_active", RunTools())
    admitted = agent_tools.project_menu([
        tool("read"), tool("authored_check"), tool("skill_load"), tool("web_fetch"),
        tool("git_commit"), tool("mcp_disabled"), tool("custom_disabled"),
    ])
    runtime = SimpleNamespace()
    head.remember_offerable_tools(runtime, admitted)

    seeded = head.listed_tools(runtime, [tool("read")], head.run_offerable_tools(runtime))
    head.remember_offerable_tools(runtime, [tool("late_offer")])
    later = head.listed_tools(
        runtime, [tool("late_tool"), tool("read")], head.run_offerable_tools(runtime))

    assert names(seeded) == ["read", "authored_check", "custom_enabled"]
    assert not {"skill_load", "web_fetch", "git_commit", "mcp_disabled", "custom_disabled"} & set(names(seeded))
    assert "mcp_disabled" not in names(later)
    assert names(later) == ["read", "authored_check", "custom_enabled", "late_tool"]


def test_offerable_sources_merge_once_with_registered_schema_priority(monkeypatch):
    monkeypatch.setenv(head.MODE_ENV, "on")
    runtime = SimpleNamespace(_completion_owner_full_tools=[
        tool("shared", description="registered"), tool("owner_only"),
    ], _readiness_full_tools=[tool("readiness_only"), tool("shared", description="readiness")])

    head.remember_offerable_tools(runtime, [tool("shared", description="initial"), tool("initial_only")])
    head.remember_offerable_tools(runtime, [tool("ignored_late_surface")])

    assert head.run_offerable_tools(runtime) == [
        tool("shared", description="registered"), tool("owner_only"),
        tool("readiness_only"), tool("initial_only"),
    ]


def test_final_surface_seeds_request_one_without_changing_callable_names(monkeypatch):
    import kernel_completion_owner as completion_owner
    import kernel_done_is_submit_authority as done_authority
    import kernel_read_lockout
    import kernel_request_tool_surface as request_surface

    monkeypatch.setenv(head.MODE_ENV, "on")
    monkeypatch.setenv("STATEFRAME_TOOL_MENU_MEMO", "off")
    monkeypatch.setattr(done_authority, "enabled", lambda _runtime: False)
    monkeypatch.setattr(done_authority, "candidate_exists", lambda _runtime: False)
    monkeypatch.setattr(kernel_read_lockout, "filter_tools", lambda _runtime, tools: list(tools))
    monkeypatch.setattr(completion_owner, "bind_final_tool_surface", lambda *args, **kwargs: True)

    runtime = SimpleNamespace()
    head.remember_offerable_tools(runtime, [
        tool("read"), tool("edit"), tool("authored_check"),
    ])
    capture = SimpleNamespace()

    def author(request_id: str, current: list[dict]) -> tuple[dict, dict]:
        body = {"messages": [{"role": "user", "content": "task"}],
                "tools": copy.deepcopy(current)}
        surface = request_surface.author_final_tool_surface(
            capture, runtime, body, request_id=request_id, bind_current_surface=True)
        assert surface is not None
        return surface, body

    first, first_body = author("request-1", [tool("read"), tool("edit")])
    second, second_body = author(
        "request-2", [tool("edit"), tool("read"), tool("authored_check")])

    assert names(first_body["tools"]) == ["read", "edit", "authored_check"]
    assert first_body["tools"] == second_body["tools"]
    assert first["names"] == ["read", "edit"]
    assert second["names"] == ["edit", "read", "authored_check"]
    assert names(runtime._completion_owner_dispatch_tools) == ["read", "edit", "authored_check"]


def test_listed_tools_are_idempotent_within_a_request():
    runtime = SimpleNamespace()
    menu = [tool("read"), tool("edit")]
    once = head.listed_tools(runtime, menu)
    assert head.listed_tools(runtime, copy.deepcopy(menu)) == once


def test_description_churn_keeps_first_schema_but_a_parameter_change_replaces_it():
    runtime = SimpleNamespace()
    head.listed_tools(runtime, [tool("edit", "Edit a file in place.")])
    terse = head.listed_tools(runtime, [tool("edit", "Host edit action.")])
    assert terse[0]["function"]["description"] == "Edit a file in place."
    changed = head.listed_tools(runtime, [tool("edit", "x", path={"type": "string"})])
    assert "path" in changed[0]["function"]["parameters"]["properties"]


def test_callable_subset_keeps_listed_order_and_schemas():
    listed = [tool("read"), tool("edit"), tool("done")]
    assert names(head.callable_subset(listed, ["done", "read"])) == ["read", "done"]


def test_menu_note_names_callable_and_gated_tools_only_when_narrower():
    listed = [tool("read"), tool("edit"), tool("done")]
    assert head.menu_note(listed, ["read", "edit", "done"]) is None
    note = head.menu_note(listed, ["read", "done"])
    assert note.startswith(head.NOTE_HEADER)
    assert "callable now: read, done" in note and "not callable this turn (a call is refused): edit" in note


def test_apply_menu_note_keeps_one_current_note_before_a_prefill():
    listed = [tool("read"), tool("edit")]
    body = {"messages": [{"role": "user", "content": "frame"},
                         {"role": "assistant", "content": "prefill"}]}
    head.apply_menu_note(body, listed, ["read"])
    head.apply_menu_note(body, listed, ["edit"])
    notes = [m for m in body["messages"] if m["content"].startswith(head.NOTE_HEADER)]
    assert len(notes) == 1 and "callable now: edit" in notes[0]["content"]
    assert body["messages"][-1] == {"role": "assistant", "content": "prefill"}
    head.apply_menu_note(body, listed, ["read", "edit"])
    assert not any(m["content"].startswith(head.NOTE_HEADER) for m in body["messages"])


def test_contract_dispatch_refuses_a_listed_but_gated_tool():
    contract = {"admitted_tool_schemas": [tool("read"), tool("edit"), tool("done")],
                "advertised_actions": ["read", "done"]}
    callable_tools = callable_tool_schemas(contract)
    assert names(callable_tools) == ["read", "done"]
    errors = validate_tool_arguments(callable_tools, "edit", {})
    assert errors and errors[0].startswith("tool 'edit' is not offered for the current state; choose one of:")
    assert not any("not offered" in error
                   for error in validate_tool_arguments(callable_tools, "read", {"path": "a.py"}))


def test_contract_without_advertised_actions_keeps_every_admitted_schema():
    contract = {"admitted_tool_schemas": [tool("read"), tool("edit")]}
    assert names(callable_tool_schemas(contract)) == ["read", "edit"]
    assert callable_tool_schemas({}) == []

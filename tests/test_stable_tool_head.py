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


def test_constant_menu_switch_defaults_off_and_rejects_invalid_values(monkeypatch):
    monkeypatch.delenv(head.CONSTANT_MENU_ENV, raising=False)
    assert head.constant_menu_mode() == "off"
    assert not head.constant_menu_enabled()
    monkeypatch.setenv(head.CONSTANT_MENU_ENV, "on")
    assert head.constant_menu_enabled()
    monkeypatch.setenv(head.CONSTANT_MENU_ENV, "yes")
    with pytest.raises(ValueError):
        head.constant_menu_enabled()


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
    import kernel_request_tool_surface as request_surface

    monkeypatch.setenv(head.MODE_ENV, "on")
    monkeypatch.setenv("STATEFRAME_TOOL_MENU_MEMO", "off")
    monkeypatch.setattr(done_authority, "enabled", lambda _runtime: False)
    monkeypatch.setattr(done_authority, "candidate_exists", lambda _runtime: False)
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


def test_constant_menu_seeds_launch_with_full_schemas_and_navigation_dispatch(monkeypatch):
    import kernel_completion_owner as completion_owner
    import kernel_done_is_submit_authority as done_authority
    import kernel_request_tool_surface as request_surface

    monkeypatch.setenv(head.MODE_ENV, "on")
    monkeypatch.setenv(head.CONSTANT_MENU_ENV, "on")
    monkeypatch.setenv("STATEFRAME_TOOL_MENU_MEMO", "off")
    monkeypatch.setenv("STATEFRAME_REQUEST_TOOL_FLOOR", "off")
    monkeypatch.setattr(done_authority, "enabled", lambda _runtime: False)
    monkeypatch.setattr(done_authority, "candidate_exists", lambda _runtime: False)
    monkeypatch.setattr(completion_owner, "bind_final_tool_surface", lambda *args, **kwargs: True)

    full_names = [
        "authored_check", "done", "edit", "file_create", "file_delete", "file_move",
        "grep", "list_files", "probe", "read", "read_check_receipt", "recall",
        "recall_evidence", "recall_generated_artifact", "references", "run_tests", "shell",
        "str_replace", "where_defined", "who_calls", "decide", "insert_before", "predict", "work",
    ]
    full = [tool(name) for name in full_names]
    first_probe = next(item for item in full if item["function"]["name"] == "probe")
    first_probe["function"]["description"] = "First-phase probe schema."
    first_probe["function"]["parameters"] = {
        "type": "object", "properties": {"first_phase": {"type": "string"}},
        "required": ["first_phase"], "additionalProperties": False,
    }
    runtime = SimpleNamespace(
        _completion_owner_full_tools=copy.deepcopy(full),
        minimal_launch_pending=lambda: True,
    )
    capture = SimpleNamespace()
    first_body = {"messages": [{"role": "user", "content": "task"}],
                  "tools": copy.deepcopy(full)}
    first = request_surface.author_final_tool_surface(
        capture, runtime, first_body, request_id="request-1", bind_current_surface=True)

    assert first is not None
    assert names(first_body["tools"]) == full_names
    assert first["names"] == ["list_files", "read"]
    first_dispatch = names(runtime._completion_owner_dispatch_tools)
    assert first_dispatch == ["list_files", "read"]
    callable_schemas = completion_owner.callable_tool_schemas({
        "admitted_tool_schemas": first_body["tools"],
        "callable_actions": first["names"],
    })
    assert names(callable_schemas) == ["list_files", "read"]
    with pytest.raises(completion_owner.RequestActionContractError, match="not offered"):
        completion_owner.dispatch_tool_call(callable_schemas, "str_replace", {})
    assert any(
        str(message.get("content", "")).startswith(head.NOTE_HEADER)
        and "callable now: list_files, read" in str(message.get("content", ""))
        for message in first_body["messages"]
    )

    runtime.minimal_launch_pending = lambda: False
    second_full = copy.deepcopy(full)
    second_probe = next(item for item in second_full if item["function"]["name"] == "probe")
    second_probe["function"]["description"] = "Later-phase probe schema."
    second_full.remove(second_probe)
    second_body = {"messages": [{"role": "user", "content": "next"}],
                   "tools": second_full}
    second = request_surface.author_final_tool_surface(
        capture, runtime, second_body, request_id="request-2", bind_current_surface=True)

    assert second is not None
    assert second_body["tools"] == first_body["tools"]
    assert next(item for item in second_body["tools"]
                if item["function"]["name"] == "probe")["function"]["description"] == (
                    "First-phase probe schema.")
    callable_names = [name for name in full_names if name != "probe"]
    assert second["names"] == callable_names
    assert names(runtime._completion_owner_dispatch_tools) == callable_names
    rebound = request_surface.author_final_tool_surface(
        capture, runtime, second_body, request_id="request-2", bind_current_surface=True)
    assert rebound is not None
    assert rebound["names"] == callable_names
    assert second_body["tools"] == first_body["tools"]

    second_probe["function"]["parameters"] = {
        "type": "object", "properties": {"later_phase": {"type": "integer"}},
        "required": ["later_phase"], "additionalProperties": False,
    }
    late_full = [*second_full, second_probe, tool("late_tool")]
    runtime.available_tools = late_full
    third_body = {"messages": [{"role": "user", "content": "later"}],
                  "tools": copy.deepcopy(late_full)}
    third = request_surface.author_final_tool_surface(
        capture, runtime, third_body, request_id="request-3", bind_current_surface=True)
    assert third is not None
    assert "late_tool" in third["names"]
    assert next(item for item in third_body["tools"]
                if item["function"]["name"] == "probe") == second_probe


def test_constant_menu_preloads_generated_recall_schema_but_does_not_call_it(monkeypatch):
    import kernel_completion_owner as completion_owner
    import kernel_done_is_submit_authority as done_authority
    import kernel_request_tool_surface as request_surface

    monkeypatch.setenv(head.MODE_ENV, "on")
    monkeypatch.setenv(head.CONSTANT_MENU_ENV, "on")
    monkeypatch.setenv("KERNEL_GENERATED_ARTIFACTS", "1")
    monkeypatch.setenv("STATEFRAME_TOOL_MENU_MEMO", "off")
    monkeypatch.setenv("STATEFRAME_REQUEST_TOOL_FLOOR", "off")
    monkeypatch.setattr(done_authority, "enabled", lambda _runtime: False)
    monkeypatch.setattr(done_authority, "candidate_exists", lambda _runtime: False)
    monkeypatch.setattr(completion_owner, "bind_final_tool_surface", lambda *args, **kwargs: True)

    navigation = [tool("list_files"), tool("read")]
    runtime = SimpleNamespace(
        _completion_owner_full_tools=copy.deepcopy(navigation),
        minimal_launch_pending=lambda: True,
    )
    body = {"messages": [{"role": "user", "content": "task"}],
            "tools": copy.deepcopy(navigation)}
    surface = request_surface.author_final_tool_surface(
        SimpleNamespace(), runtime, body, request_id="request-1", bind_current_surface=True)

    assert surface is not None
    assert names(body["tools"]) == ["list_files", "read", "recall_generated_artifact"]
    assert surface["names"] == ["list_files", "read"]
    callable_schemas = completion_owner.callable_tool_schemas({
        "admitted_tool_schemas": body["tools"],
        "callable_actions": surface["names"],
    })
    with pytest.raises(completion_owner.RequestActionContractError, match="not offered"):
        completion_owner.dispatch_tool_call(callable_schemas, "recall_generated_artifact", {})


def test_menu_guidance_uses_a_message_without_changing_tool_schemas():
    listed = [tool("read"), tool("edit")]
    body = {"messages": [{"role": "user", "content": "task"}], "tools": copy.deepcopy(listed)}

    head.apply_menu_note(body, listed, ["read"], guidance="Use the admitted action after two refusals.")

    assert body["tools"] == listed
    assert any(
        message.get("content") == (
            f"{head.GUIDANCE_NOTE_HEADER} Use the admitted action after two refusals.")
        for message in body["messages"]
    )


def test_volatile_tail_replaces_notes_after_wire_normalization():
    listed = [tool("read"), tool("edit")]
    body = {"messages": [{"role": "user", "content": (
        f"{head.VOLATILE_TAIL_MARKER}\n\ncheck facts\n\n"
        f"{head.NOTE_HEADER} old menu\n{head.GUIDANCE_NOTE_HEADER} old guidance"
    )}]}
    head.apply_menu_note(body, listed, ["read"], guidance="Current limits.", volatile_tail=True)
    messages = copy.deepcopy(body["messages"])
    head.apply_menu_note(body, listed, ["read"], guidance="Current limits.", volatile_tail=True)

    assert messages == body["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    assert content.count(head.NOTE_HEADER) == 1
    assert content.count(head.GUIDANCE_NOTE_HEADER) == 1
    assert "check facts" in content
    assert "old menu" not in content and "old guidance" not in content


def test_stage_tool_description_can_stay_fixed_as_the_stage_changes():
    from kernel_idealpath import stage_tool_schema

    off_schemas = [stage_tool_schema(stage) for stage in ("S1", "S2", "S3")]
    assert [schema["function"]["description"] for schema in off_schemas] == [
        f"Submit the typed IDEALPATH1 {stage} state answer."
        for stage in ("S1", "S2", "S3")
    ]

    on_schemas = [stage_tool_schema(stage, stable_description=True) for stage in ("S1", "S2", "S3")]

    assert len({schema["function"]["description"] for schema in on_schemas}) == 1
    assert [schema["function"]["parameters"]["properties"]["stage"]["const"]
            for schema in on_schemas] == ["S1", "S2", "S3"]


def test_requirement_draft_can_move_from_the_schema_to_guidance():
    from kernel_idealpath import requirement_table_tool_guidance, requirement_table_tool_schema

    draft = {"rows": [{"requirement": "Keep the visible draft."}]}
    off_schema = requirement_table_tool_schema(draft)
    assert "Keep the visible draft." in off_schema["function"]["description"]

    on_schema = requirement_table_tool_schema(draft, include_draft=False)
    guidance = requirement_table_tool_guidance(draft)
    assert "Keep the visible draft." not in on_schema["function"]["description"]
    assert guidance is not None and "Keep the visible draft." in guidance


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

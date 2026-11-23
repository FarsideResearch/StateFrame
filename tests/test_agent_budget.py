"""Schema admission and populated request size use the engine's request meter."""
import builtins
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from kernel_agent_schemas import BUILTIN_SCHEMAS, measure_schemas
from kernel_custom_tools import load_tool_files
from kernel_request_budget import ENDPOINT_PROMPT_TOKEN_BUDGET
from kernel_request_diet import RequestTokenMeter
from kernel_shell_tool import SHELL_SCHEMA


def custom_schemas():
    file = Path(__file__).parent / "acceptance/fixtures/tools.json"
    return load_tool_files([file], builtin_names=frozenset(), shell_network="loopback").schemas()


def test_schema_fallback_admission_is_explicit():
    report = measure_schemas(custom_schemas())
    for name, row in report.items():
        assert row["tokens"] <= (1500 if name == "custom_total" else 1200)
        assert row["method"] == "calibrated_bytes"


def test_schema_tokenization_overlaps_preparation_and_publishes_exact_report(tmp_path, monkeypatch):
    import kernel_agent_tools
    import kernel_request_diet
    import kernel_skills
    from kernel_custom_tools import CustomToolSet
    from kernel_tool_settings import ToolSettings

    started = Event()
    release = Event()
    inputs = []

    def tokenize(content, *_args, **_kwargs):
        inputs.append(content)
        started.set()
        assert release.wait(10), "workspace preparation could not run during tokenization"
        return [0] * count(content)

    def count(content):
        return len(content) % 97 + 1

    expected = measure_schemas(tokenizer=count)
    monkeypatch.setattr(kernel_request_diet, "_http_tokenize", tokenize)
    monkeypatch.setattr(kernel_agent_tools, "_active", None)
    monkeypatch.setattr(kernel_skills, "_active", None)
    shell = SimpleNamespace(spec={"chat_url": "http://127.0.0.1:9"},
                            run_dir=tmp_path, settings=ToolSettings(web=True))
    try:
        kernel_agent_tools.configure(shell, CustomToolSet((), "empty"), defer_schema_budget=True)
        assert started.wait(10)
        assert len(inputs) == 1
        assert not list(tmp_path.glob("tool-schema-budget-*.json"))
        why = BUILTIN_SCHEMAS[-1]["function"]["parameters"]["properties"]["why"]
        monkeypatch.setitem(why, "description", "Added by later menu setup.")
        monkeypatch.setattr(RequestTokenMeter, "measure", lambda *_: (1, "tokenize"))
    finally:
        release.set()

    tools = kernel_agent_tools.service()
    tools.bind()
    assert len(inputs) == len(BUILTIN_SCHEMAS) + 1
    assert tools.schema_budget == expected
    assert json.loads(tools.schema_budget_path.read_text()) == expected
    assert kernel_agent_tools.project_menu([]) == [
        item for item in BUILTIN_SCHEMAS if item["function"]["name"] == "web_fetch"
    ]
    assert len(inputs) == len(BUILTIN_SCHEMAS) + 1
    assert list(tmp_path.glob("tool-schema-budget-*.json")) == [tools.schema_budget_path]


def test_deferred_schema_budget_keeps_estimates_advisory(tmp_path, monkeypatch):
    import kernel_agent_tools
    from kernel_custom_tools import CustomToolSet
    from kernel_tool_settings import ToolSettings

    monkeypatch.setattr(kernel_agent_tools, "_active", None)
    endpoint = "http://127.0.0.1:9"
    monkeypatch.setattr(RequestTokenMeter, "measure",
                        lambda *_: (1201, "calibrated_bytes"))
    monkeypatch.setattr(kernel_agent_tools.AgentTools, "_prepare_schema_budget",
                        lambda self, **_kwargs: lambda **_options: measure_schemas())
    shell = SimpleNamespace(spec={"chat_url": endpoint},
                            run_dir=tmp_path, settings=ToolSettings())
    kernel_agent_tools.configure(shell, CustomToolSet((), "empty"), defer_schema_budget=True)
    kernel_agent_tools.project_menu([])
    tools = kernel_agent_tools.service()
    assert tools.schema_budget["shell"] == {"tokens": 1201, "method": "calibrated_bytes"}
    assert json.loads(tools.schema_budget_path.read_text()) == tools.schema_budget
    assert list(tmp_path.glob("tool-schema-budget-*.json")) == [tools.schema_budget_path]


def test_deferred_schema_budget_still_refuses_an_oversized_schema(tmp_path, monkeypatch):
    import kernel_agent_tools
    import kernel_request_diet
    from kernel_custom_tools import CustomToolError, CustomToolSet
    from kernel_tool_settings import ToolSettings

    monkeypatch.setattr(kernel_request_diet, "_http_tokenize", lambda *_a, **_kw: [0] * 1201)
    monkeypatch.setattr(kernel_agent_tools, "_active", None)
    shell = SimpleNamespace(spec={"chat_url": "http://127.0.0.1:9"},
                            run_dir=tmp_path, settings=ToolSettings())
    kernel_agent_tools.configure(shell, CustomToolSet((), "empty"), defer_schema_budget=True)
    with pytest.raises(CustomToolError, match="schema exceeds 1200 tokens: 1201"):
        kernel_agent_tools.project_menu([])
    assert not list(tmp_path.glob("tool-schema-budget-*.json"))


def test_web_implementation_loads_on_first_dispatch(tmp_path, monkeypatch):
    from kernel_agent_tools import AgentTools
    from kernel_custom_tools import CustomToolSet
    from kernel_tool_settings import ToolSettings

    shell = SimpleNamespace(spec={}, run_dir=tmp_path, settings=ToolSettings(web=True))
    imports = []
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "kernel_web_fetch":
            imports.append(name)
            return SimpleNamespace(
                WebPolicy=lambda *_args, **_kwargs: None,
                fetch=lambda *_args: SimpleNamespace(ok=True, status=200, text="fetched text"),
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    tools = AgentTools(shell, CustomToolSet((), "empty"))
    tools.bind()
    assert any(item["function"]["name"] == "web_fetch" for item in tools.schemas())
    assert imports == []

    def observe(name, args, call_id, operation, **kwargs):
        return operation()

    monkeypatch.setattr(tools, "observation", observe)
    reply = tools.execute("web_fetch", {"url": "https://example.com"}, "fetch-1")
    assert reply.observation == "fetched text"
    assert imports == ["kernel_web_fetch"]


def populated_body():
    jobs = [{"id": f"j{i}", "command": "python server.py", "state": "running",
             "exit_code": None, "events": ["output matched"], "tail": "x" * 1024} for i in range(4)]
    frame = {"jobs": jobs, "effects": [{"path": f"module_{i}.py", "change": "modified"} for i in range(30)],
             "task": "Inspect generated files and verify the declared checks.",
             "product_revision": "a" * 64, "environment_revision": "b" * 64}
    return {"messages": [{"role": "system", "content": "Use tools; only host checks certify completion."},
                         {"role": "user", "content": json.dumps(frame)},
                         {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "shell", "arguments": '{"command":"make"}'}}]},
                         {"role": "tool", "tool_call_id": "c1", "content": "x" * 6144 + "\n[bytes omitted]\n" + "y" * 6144}],
            "tools": [SHELL_SCHEMA, *BUILTIN_SCHEMAS, *custom_schemas()]}


def test_populated_request_fits_default_budget_with_real_fallback_meter():
    count, method = RequestTokenMeter().measure(populated_body())
    assert method == "calibrated_bytes"
    assert count < ENDPOINT_PROMPT_TOKEN_BUDGET


@pytest.mark.host_sandbox
def test_schema_and_populated_request_through_engine_tokenizer_transport():
    from stateframe.scripted_endpoint import Script, serve
    script = Script({"steps": [], "tokenize": {"mode": "ok"}})
    server = serve(script)
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/tokenize"
        report = measure_schemas(custom_schemas(), endpoint=endpoint)
        assert all(row["method"] == "tokenize" for row in report.values())
        count, method = RequestTokenMeter(endpoint=endpoint).measure(populated_body())
        assert method == "tokenize" and count < ENDPOINT_PROMPT_TOKEN_BUDGET
    finally:
        server.shutdown()
        server.server_close()


def test_framefit_composes_populated_state_without_losing_jobs_or_exchange():
    from kernel_stateframe_framefit5 import compose_frame_only, STATE_HEADER, PRODUCT_MODE
    body = populated_body()
    source = body["messages"]
    frame = json.loads(source[1]["content"])
    frame["mode"] = PRODUCT_MODE
    block = {"role": "user", "content": STATE_HEADER + "\n" + json.dumps({"state": {"framefit5": frame}})}
    prepared = compose_frame_only([source[0], block], source)
    encoded = json.dumps(prepared)
    assert all(f"j{i}" in encoded for i in range(4))
    assert "module_29.py" in encoded
    assert next(m for m in prepared if m.get("role") == "tool")["content"] == source[-1]["content"]
    count, _ = RequestTokenMeter().measure({"messages": prepared, "tools": body["tools"]})
    assert count < ENDPOINT_PROMPT_TOKEN_BUDGET

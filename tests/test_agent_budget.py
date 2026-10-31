"""Schema admission and populated request size use the engine's request meter."""
import json
from pathlib import Path

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

"""The request seal binds the provider's serialized body, not private routing tags."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from _journal import measurement_events
from engine_run import run_monitored
from kernel_final_wire_allocation import _finalize_provider_request_body
from kernel_provider_dialect import ProviderCapabilities
from kernel_request_measurement import RequestMeasurement
from kernel_request_seal import RequestSealError
from stateframe.scripted_endpoint import Script, serve


ROOT = Path(__file__).resolve().parents[1]


def _strict_body(body: dict[str, object]) -> dict[str, object]:
    capture = SimpleNamespace(frame_route=None, journal=SimpleNamespace(manifest={}))
    return _finalize_provider_request_body(
        capture, body, ProviderCapabilities("openai-strict"),
    )


def _measurement(body: dict[str, object]) -> tuple[RequestMeasurement, dict[str, object]]:
    wire = json.dumps(_strict_body(body)).encode()
    measured = RequestMeasurement.bind(
        body, tokenizer_identity="strict-test", request_id="r", logical_request_id="r",
        tokens=1, method="tokenize", template_overhead_tokens=0,
        max_prompt_tokens=33_344, max_request_bytes=100_000, wire=wire,
    )
    receipt = {"body_sha256": measured.body_sha256, "request_bytes": measured.request_bytes,
               **measured.receipt_fields()}
    return measured, receipt


def _validate(measured: RequestMeasurement, receipt: dict[str, object], body: dict[str, object]) -> None:
    measured.validate(body, receipt, request_id="r", logical_request_id="r",
                      tokenizer_identity="strict-test")


def test_private_section_change_does_not_invalidate_wire_measurement() -> None:
    body = {"model": "strict-mock", "messages": [
        {"role": "user", "content": "[Model source window]", "_request_budget_section": "stage_evidence"}],
        "tools": [], "max_tokens": 100}
    measured, receipt = _measurement(body)
    changed = copy.deepcopy(body)
    changed["messages"][0]["_request_budget_section"] = "other"  # type: ignore[index]
    _validate(measured, receipt, _strict_body(changed))
    del changed["messages"][0]["_request_budget_section"]  # type: ignore[index]
    _validate(measured, receipt, _strict_body(changed))


@pytest.mark.parametrize("field,value", [("content", "changed"), ("role", "system"),
                                         ("name", "changed")])
def test_visible_message_change_still_breaks_seal(field: str, value: str) -> None:
    body = {"model": "strict-mock", "messages": [
        {"role": "user", "content": "source", "_request_budget_section": "stage_evidence"}],
        "tools": [], "max_tokens": 100}
    measured, receipt = _measurement(body)
    changed = _strict_body(body)
    changed["messages"][0][field] = value  # type: ignore[index]
    with pytest.raises(RequestSealError, match="measurement body/projection/provider identity changed"):
        _validate(measured, receipt, changed)


def _run_scripted(tmp_path: Path, *, strict: bool, steps: list[dict[str, object]],
                  inject_allocation_replan: bool = False) -> tuple[object, Path, Script]:
    source_root = Path(os.environ.get("SF_SEAL_TEST_SOURCE", str(ROOT / "src")))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 7\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests/test_value.py").write_text(
        "from app import VALUE\ndef test_value(): assert VALUE == 7\n", encoding="utf-8")
    script = Script({"model": "strict-mock" if strict else "scripted",
                     "strict_openai": strict, "steps": steps})
    server = serve(script)
    run = tmp_path / "run"
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("STATEFRAME_", "REACT_"))}
    environment.update({"PYTHONPATH": os.pathsep.join((str(source_root),
                                                         str(source_root / "stateframe/engine"))),
                        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                        "STATEFRAME_HOME": str(tmp_path / "state-home")})
    wrapper = tmp_path / "python-with-private-source-tag"
    bootstrap = "\n".join((
        "import json, sys",
        "from pathlib import Path",
        "child, run = map(Path, sys.argv[1:3])",
        "sys.path.insert(0, str(child.parent.parent))",
        "import stateframe._child as host",
        "host.install_environment(json.loads((run / 'spec.json').read_text()))",
        "sys.path.insert(0, str(child.parent / 'engine'))",
        "import kernel_final_wire_allocation as allocation",
        "original = allocation._prepare_qualified_hook_body",
        "def tagged(context, body, request_id):",
        "    result = original(context, body, request_id)",
        "    for message in result.get('messages', []):",
        "        content = str(message.get('content', ''))",
        "        if content.startswith(('[Current source:', '[Model source window]', '[read page]')):",
        "            message['_request_budget_section'] = 'stage_evidence'",
        "            (run / 'tagged-source-window').write_text(request_id)",
        "            break",
        "    return result",
        "allocation._prepare_qualified_hook_body = tagged",
        *((
            "import kernel_frame_allocation_validation as validation",
            "from kernel_frame_optional_allocation import FrameAllocationReplanNeeded",
            "validate = validation.validate_allocation_authority",
            "def replan_after_charge(allocation, runtime, messages, tools=None, **kwargs):",
            "    budget = getattr(runtime, 'continuation_limits', None)",
            "    if budget is not None and budget.continuation_model_decisions_used and not (run / 'injected-replan').exists():",
            "        (run / 'injected-replan').write_text(str(budget.continuation_model_decisions_used))",
            "        raise FrameAllocationReplanNeeded('allocation ledger authority changed')",
            "    return validate(allocation, runtime, messages, tools, **kwargs)",
            "validation.validate_allocation_authority = replan_after_charge",
        ) if inject_allocation_replan else ()),
        "raise SystemExit(host.main([str(child), str(run)]))",
    ))
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os,sys\n"
        f"real_python={sys.executable!r}\n"
        "args=sys.argv[1:]\n"
        "if len(args) >= 3 and args[0] == '-B' and os.path.basename(args[1]) == '_child.py':\n"
        f"    os.execv(real_python, [real_python, '-B', '-c', {bootstrap!r}, *args[1:]])\n"
        "os.execv(real_python, [real_python, *args])\n", encoding="utf-8")
    wrapper.chmod(0o755)
    try:
        process = run_monitored(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Read app.py and tests/test_value.py; run the test and report VALUE.",
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--model", script.model, "--run-dir", str(run), "--max-turns",
             "32" if inject_allocation_replan else "12",
             "--test-python", sys.executable, "--python", str(wrapper), "--quiet"],
            cwd=ROOT, env=environment, capture_output=True, text=True,
            expected_seconds=240, run_dir=run, check=False)
        return process, run, script
    finally:
        server.shutdown()
        server.server_close()


def assert_sealed_wire(run: Path, *, minimum: int) -> None:
    requests = sorted((run / "wire").glob("*.request.json"))
    assert len(requests) >= minimum
    rows = [row for row in measurement_events(run)
            if row.get("kind") == "request_budget_final_wire" and row.get("status") == "READY"]
    digests = [row["body_sha256"] for row in rows if "body_sha256" in row]
    for request in requests:
        wire = request.read_bytes()
        assert "sha256:" + hashlib.sha256(wire).hexdigest() in digests
        assert all(not key.startswith("_") for message in json.loads(wire)["messages"]
                   for key in message)


@pytest.mark.host_sandbox
def test_hosted_stage_evidence_sends_seal_exact_wire(tmp_path: Path) -> None:
    steps: list[dict[str, object]] = [
        {"tool": "read", "arguments": {"path": path}}
        for path in ("app.py", "tests/test_value.py", "app.py", "tests/test_value.py")
    ] + [{"tool": "run_tests", "arguments": {"target": "tests/test_value.py"}},
         {"tool": "done", "arguments": {"summary": "VALUE is seven and tests pass."}}]
    process, run, script = _run_scripted(tmp_path, strict=True, steps=steps)
    output = process.stdout + process.stderr
    traceback = run / "child-traceback.txt"
    if traceback.is_file():
        output += traceback.read_text(encoding="utf-8")
    assert len(script.requests) >= 4, output
    assert (run / "tagged-source-window").is_file(), output
    assert "measurement body/projection/provider identity changed" not in output
    assert process.returncode == 0, output
    script.assert_consumed()
    assert_sealed_wire(run, minimum=4)

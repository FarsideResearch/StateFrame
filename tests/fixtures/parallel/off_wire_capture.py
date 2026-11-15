"""Capture raw requests on main or the candidate, with deterministic external inputs.

Regenerate from an unmodified b520a637 tree, from this checkout's root:
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 .venv/bin/python \
        tests/fixtures/parallel/off_wire_capture.py MAIN_TREE OUTPUT_DIR omitted
Copy OUTPUT_DIR/run/wire/000{1,2,3}.request.json verbatim to off_wire_main_b520a637/.
Only provider tokenization/replies and check-process results are simulated. Request
construction, failed-check recording, edits, CLI option parsing and child wiring are real.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import secrets
import shutil
import subprocess
import sys
import time


def main() -> int:
    tree, output = (Path(value).resolve() for value in sys.argv[1:3])
    mode = sys.argv[3]
    # Do not inherit operator switches, user skills, routes or the invoking pytest profile.
    for name in tuple(os.environ):
        if name.startswith(("STATEFRAME_", "SF_", "DS_", "LLAMA_")):
            os.environ.pop(name)
    os.environ["STATEFRAME_HOME"] = str(output / "home")
    os.environ["PYTHONHASHSEED"] = "0"  # caller also sets it before interpreter startup
    sys.path[:0] = [str(tree / "src"), str(tree / "src/stateframe/engine")]
    random.seed(0)
    secrets.token_hex = lambda n=32: "0123456789abcdef" * (n // 8) + "0123456789abcdef"[:(n % 8) * 2]
    time.time = lambda: 1_700_000_000.0
    time.monotonic = lambda: 1000.0
    time.perf_counter = lambda: 1000.0
    time.time_ns = lambda: 1_700_000_000_000_000_000
    time.monotonic_ns = lambda: 1_000_000_000_000
    time.perf_counter_ns = lambda: 1_000_000_000_000

    from stateframe import _child, cli, runner
    from stateframe.endpoint import Endpoint
    from stateframe.scripted_endpoint import Script

    repo = output / "repo"
    shutil.copytree(tree / "examples/discount_bug", repo)
    script_data = json.loads((tree / "examples/discount_bug.script.json").read_text())
    edit = script_data["steps"][1]
    check = script_data["steps"][2]
    script = Script({"model": "scripted", "steps": [check, edit, script_data["steps"][3]]})
    arguments = ["run", "--repo", str(repo), "--run-dir", str(output / "run"),
                 "--task", "Fix the pricing code so the tests pass.",
                 "--endpoint", "http://127.0.0.1:8080/v1", "--model", "scripted",
                 "--profile", "r2hc3", "--max-turns", "8"]
    if mode != "omitted":
        arguments += ["--parallel", mode]
    args = cli.build_parser().parse_args(arguments)
    endpoint = Endpoint.from_args(args.endpoint, args.model, args.api_key_env)
    options = cli._run_options(args, endpoint)
    run, _, _spec = runner.prepare(options, printer=lambda *_: None)

    # The runner hands the child its home-identity pin through the process environment
    # (not the spec); the in-process harness plays that role before the child captures it.
    os.environ[_child.RUNNER_HOME_IDENTITY_ENV] = _spec["environment"][_child.RUNNER_HOME_IDENTITY_ENV]
    _child.install_environment(_spec)

    import kernel_bounded_http
    import kernel_execution_policy as execution
    import kernel_request_diet

    # Keep the production token-meter/cache path, replacing only its provider call.
    kernel_request_diet._http_tokenize = lambda content, *args, **kwargs: max(1, len(content.encode()) // 4)
    checks = []

    def run_check(command, root, **kwargs):
        fixed = edit["arguments"]["new_str"] in (root / "shop/pricing.py").read_text()
        checks.append("PASS" if fixed else "FAIL")
        text = ("tests/test_pricing.py::test_quarter_off PASSED\n" if fixed else
                "tests/test_pricing.py::test_quarter_off FAILED\nAssertionError: assert 20.0 == 60.0\n")
        return execution.PolicyResult(
            subprocess.CompletedProcess(command, 0 if fixed else 1, text, ""),
            False, False, False,
            execution.CapabilityReport("Linux", "bwrap", True, True, True, True, True, True),
        )

    execution.run_sandboxed = run_check
    dispatch = []

    def provider(wire, endpoint, key, **kwargs):
        body = json.loads(wire)
        offered = [tool["function"]["name"] for tool in body.get("tools", [])]
        if mode not in ("omitted", "off"):
            from kernel_completion_owner import dispatch_tool_call
            from kernel_tool_surface import dispatch_admissible
            assert dispatch_admissible("parallel")
            assert dispatch_tool_call(body["tools"], "parallel", {"subframes": []}).name == "parallel"
            dispatch.append(True)
        step = script.next_call(offered)
        index = len(script.requests) + 1
        script.requests.append(body)
        call = {"id": f"call_{index}", "type": "function",
                "function": {"name": step["tool"], "arguments": json.dumps(step["arguments"])}}
        return json.dumps({"id": f"chatcmpl-{index}", "object": "chat.completion", "model": "scripted",
                           "choices": [{"index": 0, "finish_reason": "tool_calls",
                                        "message": {"role": "assistant", "content": "", "tool_calls": [call]}}],
                           "usage": {"prompt_tokens": 10000, "completion_tokens": 100, "total_tokens": 10100}}).encode()

    kernel_bounded_http.request_bytes = provider
    _child.watch_parent = lambda: None  # this harness is already a monitored isolated process
    status = _child.main([str(tree / "src/stateframe/_child.py"), str(run)])
    if status:
        trace = run / "child-traceback.txt"
        raise AssertionError(trace.read_text() if trace.exists() else f"child status {status}")
    script.assert_consumed()
    assert checks[0] == "FAIL" and checks[-1] == "PASS", checks
    (output / "evidence.json").write_text(json.dumps({"checks": checks, "dispatch": dispatch}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

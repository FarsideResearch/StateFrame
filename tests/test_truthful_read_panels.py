"""Provider request coverage for optional source panels."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError
from urllib.parse import urlsplit
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "014bb2aded"
PROFILE_PATH = Path("src/stateframe/profiles/r2hc3.json")
TASK = "Change VALUE from one to two in pkg/value.py and keep the module importable."
PROMPT_BUDGET = 25000
SWITCH = "STATEFRAME_TRUTHFUL_READ_PANELS"
PANEL = "Required source paths are not attached; read one only if your edit needs it:"
PARTIAL_PANEL = "Source paths with deferred ranges (read only if needed):"
STALE_REASON = "Source ranges excluded because their evidence is stale:"


def _step(tool: str, **arguments: object) -> dict[str, object]:
    return {"tool": tool, "arguments": arguments}


def _cli_files() -> dict[str, str]:
    filler = "".join(f"# row {index:04d} {'x' * 58}\n" for index in range(1, 1000))
    return {
        "pkg/__init__.py": "",
        "pkg/value.py": "VALUE = 1\n" + filler,
        "pkg/context_a.py": "# " + ("a b c d e f g h i j k l m n o p q r s t u v w x y z " * 1000),
        "pkg/context_b.py": "# " + ("a b c d e f g h i j k l m n o p q r s t u v w x y z " * 1000),
        "pkg/context_c.py": "# " + ("a b c d e f g h i j k l m n o p q r s t u v w x y z " * 1000),
        "tests/test_value.py": (
            "from pkg.value import VALUE\n"
            "def test_value():\n"
            "    assert VALUE == 2\n"
        ),
    }


def _cli_steps() -> list[dict[str, object]]:
    return [
        _step("read", path="pkg/context_a.py", max_chars=64000),
        _step("read", path="pkg/context_b.py", max_chars=64000),
        _step("read", path="pkg/context_c.py", max_chars=64000),
        _step("read", path="pkg/value.py", start_line=1, end_line=1000, max_chars=64000),
        _step("str_replace", path="pkg/value.py", old_str="VALUE = 1", new_str="VALUE = 2"),
        _step("run_tests", target="tests/test_value.py"),
        _step("done", summary="VALUE is two and the test passes."),
    ]


def _run_inputs(path: Path, tree: Path, switch: str, port: int):
    # Use the scripted audit's fixture, profile overlay, and CLI arguments in
    # both transports so the runner derives the same public row and runtime.
    repo = path / "repo"
    for relative, content in _cli_files().items():
        source = repo / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(content, encoding="utf-8")
    profile = json.loads((tree / PROFILE_PATH).read_text(encoding="utf-8"))
    profile["switches"][SWITCH] = switch
    profile_path = path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    run = path / "run"
    args = [
        "run", "--repo", str(repo), "--task", TASK, "--target", "pkg/value.py",
        "--endpoint", f"http://127.0.0.1:{port}/v1", "--model", "scripted",
        "--run-dir", str(run), "--test-python", sys.executable, "--max-turns", "10",
        "--profile", str(profile_path), "--no-run-env", "--quiet",
    ]
    return run, args


def _run_environment(path: Path, tree: Path) -> dict[str, str]:
    return {
        **{key: value for key, value in os.environ.items()
           if not key.startswith(("STATEFRAME_", "SF_", "REACT_", "LLAMA_CPP_"))},
        "STATEFRAME_HOME": str(path / "home"),
        "STATEFRAME_PROMPT_BUDGET": str(PROMPT_BUDGET),
        "PYTHONPATH": os.pathsep.join(map(str, (tree / "src", tree / "src/stateframe/engine"))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TMPDIR": str(path),
    }


def _script():
    from stateframe.scripted_endpoint import Script

    return Script({"model": "scripted", "steps": _cli_steps(), "tokenize": {"mode": "ok"}})


def _wire_requests(run: Path) -> list[bytes]:
    return [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]


def _run_cli(path: Path, tree: Path, switch: str):
    from engine_run import run_monitored
    from stateframe.scripted_endpoint import serve

    script = _script()
    server = serve(script)
    try:
        run, args = _run_inputs(path, tree, switch, server.server_port)
        result = run_monitored(
            [sys.executable, "-m", "stateframe", *args],
            env=_run_environment(path, tree), capture_output=True, text=True, expected_seconds=240,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        script.assert_consumed()
        requests = _wire_requests(run)
        assert [json.loads(body) for body in requests] == script.requests
        return requests, json.loads((run / "spec.json").read_text()), script.tokenize_calls
    finally:
        server.shutdown()
        server.server_close()


def _request_metrics(requests: list[bytes]) -> dict[str, object]:
    sizes = [len(request) for request in requests]
    return {"calls": len(sizes), "bytes_per_call": sizes, "total_bytes": sum(sizes)}


def _deferred_panels(requests: list[bytes]) -> list[str]:
    return [
        message["content"]
        for request in requests
        for message in json.loads(request)["messages"]
        if isinstance(message.get("content"), str)
        and message["content"].startswith("[Deferred required source paths]\n")
    ]


def _assert_switch_on(requests: list[bytes]) -> None:
    panels = _deferred_panels(requests)
    assert any(
        f"\n{PARTIAL_PANEL}\n- pkg/value.py" in panel
        and "\n- pkg/value.py:1-" in panel
        and PANEL not in panel
        for panel in panels
    )


@pytest.mark.host_sandbox
def test_profile_switch_changes_provider_request_and_off_matches_untouched_base(
        tmp_path, tmp_path_factory, capsys):
    from tools.pinned_tree import pinned_tree

    base = pinned_tree(BASE_COMMIT, tmp_path_factory)
    base_requests, base_spec, base_tokenize = _run_cli(tmp_path / "base", base, "off")
    off_requests, off_spec, off_tokenize = _run_cli(tmp_path / "off", ROOT, "off")
    on_requests, on_spec, on_tokenize = _run_cli(tmp_path / "on", ROOT, "on")

    assert base_spec["public_row"] == off_spec["public_row"] == on_spec["public_row"]
    assert off_spec["environment"][SWITCH] == "off"
    assert on_spec["environment"][SWITCH] == "on"
    assert off_requests == base_requests
    assert any(PANEL in panel for panel in _deferred_panels(off_requests))
    assert all(PARTIAL_PANEL not in panel for panel in _deferred_panels(off_requests))
    _assert_switch_on(on_requests)
    assert len(on_requests) == len(off_requests)

    base_metrics = _request_metrics(base_requests)
    off_metrics = _request_metrics(off_requests)
    on_metrics = _request_metrics(on_requests)
    metrics = {
        "base": base_metrics,
        "off": off_metrics,
        "on": on_metrics,
        "tokenize_calls": {"base": base_tokenize, "off": off_tokenize, "on": on_tokenize},
        "switch_extra_calls": int(on_metrics["calls"]) - int(off_metrics["calls"]),
        "switch_extra_bytes": int(on_metrics["total_bytes"]) - int(off_metrics["total_bytes"]),
        "switch_per_call_byte_delta": [
            len(on_request) - len(off_request)
            for on_request, off_request in zip(on_requests, off_requests)
        ],
    }
    with capsys.disabled():
        print("TRUTHFUL_READ_PANEL_METRICS " + json.dumps(metrics, sort_keys=True), file=sys.stderr)


def test_cli_request_builder_labels_partially_delivered_source(tmp_path):
    run, args = _run_inputs(tmp_path, ROOT, "on", 8080)
    # The child installs process-wide state. Isolate that state while keeping
    # runner and child together inside the socket-free test process.
    result = subprocess.run(
        [sys.executable, "-B", __file__, *args], env=_run_environment(tmp_path, ROOT),
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    spec = json.loads((run / "spec.json").read_text())
    assert spec["environment"][SWITCH] == "on"
    assert "STATEFRAME_EXISTENCE_ORACLE" not in spec["environment"]
    assert spec["environment"]["STATEFRAME_WORKING_SET_ALLOCATOR"] == "on"
    _assert_switch_on(_wire_requests(run))


def _in_process_cli(args: list[str]) -> int:
    from stateframe import _child, cli, runner
    from stateframe.engine import kernel_test_isolation
    from stateframe.scripted_endpoint import make_handler
    import kernel_bounded_http
    import kernel_execution_policy as execution
    import kernel_provider_dialect
    from stateframe.engine import kernel_provider_dialect as runner_dialect

    def no_sockets(event, _arguments):
        if event in {"socket.__new__", "socket.connect", "socket.bind"}:
            raise PermissionError("unexpected socket in the in-process CLI test")

    sys.addaudithook(no_sockets)
    script = _script()
    handler_type = make_handler(script)
    received = []

    def exchange(url, body=None, headers=()):
        handler = object.__new__(handler_type)
        handler.path = urlsplit(url).path
        handler.headers = {"Content-Length": str(len(body or b"")), **dict(headers)}
        handler.rfile = BytesIO(body or b"")
        reply = []
        handler._send = lambda status, value: reply.extend((status, json.dumps(value).encode()))
        if body is None:
            handler.do_GET()
        else:
            handler.do_POST()
        return reply

    def open_in_memory(_opener, request, *args, **kwargs):
        url = request.full_url
        status, raw = exchange(url, request.data, request.header_items())
        if status != 200:
            raise HTTPError(url, status, "scripted reply", {}, BytesIO(raw))

        class Response(BytesIO):
            def geturl(self):
                return url

        response = Response(raw)
        response.status = status
        return response

    def probe(_endpoint, _key, path, **kwargs):
        status, raw = exchange(path)
        return status, json.loads(raw)

    def provider(wire, endpoint, _key, **kwargs):
        status, raw = exchange(endpoint, wire)
        assert status == 200, raw
        received.append(wire)
        return raw

    def drive_child(run, _options, _printer):
        spec = json.loads((run / "spec.json").read_text())
        name = _child.RUNNER_HOME_IDENTITY_ENV
        os.environ[name] = spec["environment"][name]
        os.chdir(run)
        return _child.main([_child.__file__, str(run)])

    def startup_probe(python, code, **kwargs):
        return subprocess.check_output([python, "-I", "-B", "-c", code])

    def host_check(command, cwd, **kwargs):
        passed = (cwd / "pkg/value.py").read_text().startswith("VALUE = 2\n")
        return {"command": command, "passed": passed, "exit_code": int(not passed), "tail": ""}

    def child_check(command, cwd, **kwargs):
        check = host_check(command, cwd)
        verdict = "PASSED" if check["passed"] else "FAILED"
        completed = subprocess.CompletedProcess(
            command, check["exit_code"], f"tests/test_value.py::test_value {verdict}\n", "",
        )
        capabilities = execution.CapabilityReport(
            "Linux", "bwrap", True, True, True, True, True, True,
        )
        return execution.PolicyResult(completed, False, False, False, capabilities)

    # Stub transport and host containment. The normal CLI still builds the
    # task row, creates the child runtime, qualifies the wire, and records it.
    urllib.request.OpenerDirector.open = open_in_memory
    kernel_provider_dialect.bounded_json_probe = runner_dialect.bounded_json_probe = probe
    kernel_bounded_http.request_bytes = provider
    runner._startup_probe = startup_probe
    kernel_test_isolation.check_sandbox_exec_denial = lambda _path: None
    runner.run_test_cmd = host_check
    execution.run_sandboxed = child_check
    runner.drive = drive_child
    _child.watch_parent = lambda: None
    run = Path(args[args.index("--run-dir") + 1])
    status = cli.main(args)
    if status:
        trace = run / "child-traceback.txt"
        if trace.exists():
            print(trace.read_text(), file=sys.stderr)
        return status
    script.assert_consumed()
    assert _wire_requests(run) == received
    return status


def test_oracle_only_lists_paths_with_no_receipt_as_unattached(tmp_path, monkeypatch):
    import kernel_request_source_window as source_window

    source = tmp_path / "pkg/value.py"
    source.parent.mkdir(parents=True)
    content = "VALUE = 1\n"
    source.write_text(content, encoding="utf-8")
    (source.parent / "unread.py").write_text("OTHER = 1\n", encoding="utf-8")
    revision = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
    clipped_read = (
        f"[read page] path=pkg/value.py revision={revision} lines=1-1/1 "
        "requested=1-1 effective=lines:1-1@1:1 clip=max_chars chars=8 bytes=8 "
        "truncated=true decoded_with_replacement=false"
    )
    messages = [
        {"role": "user", "content": "Task"},
        {"role": "tool", "content": clipped_read, "turn": 1},
    ]
    monkeypatch.setenv("STATEFRAME_EXISTENCE_ORACLE", "on")
    monkeypatch.setenv(SWITCH, "on")

    source_window.upsert_deferred_source_notice(
        messages, ("pkg/value.py", "pkg/unread.py"), workspace_root=tmp_path,
    )
    text = "\n".join(str(message.get("content", "")) for message in messages)

    assert "were delivered but clipped" in text
    assert text.split(PANEL + "\n")[1] == "- pkg/unread.py"
    assert source_window.deferred_source_notice_paths(messages) == (
        ("pkg/value.py", "pkg/unread.py"),
    )


def test_stale_source_evidence_is_not_labeled_as_budget_eviction(monkeypatch):
    import kernel_request_diet as request_diet

    monkeypatch.setenv(SWITCH, "on")
    monkeypatch.delenv("STATEFRAME_DIET_EVICT_DUPLICATED_ONLY", raising=False)
    messages = [{
        "role": "user",
        "content": "The current source is attached above.",
        "_framebuild_evidence_envelope": {
            "source_ref": "pkg/value.py",
            "source_revision": "sha256:current",
            "covered_intervals": [{"start_line": 1, "end_line": 2}],
        },
    }]
    request_diet._upsert_dropped_source_notice(messages, [
        {
            "path": "pkg/value.py", "start_line": 1, "end_line": 2,
            "revision": "sha256:budget", "reason": "budget_eviction",
        },
        {
            "path": "pkg/value.py", "start_line": 1, "end_line": 2,
            "revision": "sha256:stale", "reason": "stale_evidence",
        },
        {
            "path": "pkg/value.py", "start_line": 1, "end_line": 2,
            "revision": "sha256:old-revision", "reason": "stale_revision",
        },
    ])
    notice = "\n".join(str(message.get("content", "")) for message in messages)
    budget_header = "Ranges omitted by request budgeting:"
    stale_header = STALE_REASON
    stale_revision_header = "Source ranges excluded because their revision is stale:"
    budget_row = "pkg/value.py:1-2 revision=sha256:budget reason=budget_eviction"
    stale_row = "pkg/value.py:1-2 revision=sha256:stale reason=stale_evidence"
    stale_revision_row = (
        "pkg/value.py:1-2 revision=sha256:old-revision reason=stale_revision"
    )

    assert budget_header in notice
    assert stale_header in notice
    assert stale_revision_header in notice
    assert notice.index(budget_header) < notice.index(budget_row) < notice.index(stale_header)
    assert notice.index(stale_header) < notice.index(stale_row) < notice.index(stale_revision_header)
    assert notice.index(stale_revision_header) < notice.index(stale_revision_row)


if __name__ == "__main__":
    raise SystemExit(_in_process_cli(sys.argv[1:]))

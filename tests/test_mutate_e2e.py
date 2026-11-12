"""Integrated mutation admission and dispatch checks."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from _journal import measurement_events


def _measurement_failure_tail(run: Path) -> str:
    markers = ("framefit_fail_closed", "final_tool_surface_authoring_failed",
               "provider_boundary_contract_error")
    lines = []
    for event in measurement_events(run):
        encoded = json.dumps(event, sort_keys=True, ensure_ascii=False)
        if any(marker in encoded for marker in markers):
            lines.append(encoded[:1200])
    return "\n".join(lines)[-2200:]


def test_input_snapshots_refuse_secret_links_and_preserve_tail(tmp_path):
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_runtime import admit_inputs

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "log.txt").write_bytes(b"alpha\nbeta\ngamma\n")
    (repo / ".env.local").write_text("secret")
    (repo / "linked").symlink_to("log.txt")
    snapshot = tmp_path / "snap"
    result = admit_inputs(repo, [{"path": "log.txt", "tail_bytes": 6}],
                          MutationPolicy(level="prebuilt"), snapshot_root=snapshot)
    assert result.descriptors[0]["id"] == "in0"
    assert result.files["in0"].read_bytes() == b"gamma\n"
    for name in (".env.local", "linked"):
        rejected_root = tmp_path / ("snap-" + name)
        with pytest.raises((ValueError, OSError)):
            admit_inputs(repo, [name], MutationPolicy(level="prebuilt"),
                         snapshot_root=rejected_root)
        assert not rejected_root.exists()


def test_mutation_input_cannot_cross_run_read_scope(tmp_path):
    from kernel_mutate_inputs import admit_inputs
    from kernel_mutate_policy import MutationPolicy

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "allowed.txt").write_text("allowed")
    (repo / "private.txt").write_text("secret")
    policy = MutationPolicy(level="prebuilt", read_globs=("*.txt",))
    with pytest.raises(ValueError, match="read scope"):
        admit_inputs(repo, ["private.txt"], policy, snapshot_root=tmp_path / "rejected",
                     ordinary_read_globs=("allowed.txt",))
    admitted = admit_inputs(repo, ["allowed.txt"], policy, snapshot_root=tmp_path / "admitted",
                           ordinary_read_globs=("allowed.txt",))
    assert admitted.files["in0"].read_text() == "allowed"


def test_prebuilt_smoke_fixture_paths_are_normalized(tmp_path):
    from kernel_mutate_build import smoke, stage_unit

    root = Path(__file__).resolve().parents[1] / "src/stateframe/mutations_prebuilt/file_facts"
    manifest = json.loads((root / "unit.json").read_text())
    staged = stage_unit({"main.py": (root / "main.py").read_bytes()}, manifest, tmp_path,
                        fixture_root=root)
    calls = []

    class Executor:
        def run(self, spec):
            calls.append(spec)
            result = {"wire": 1, "call": spec.call, "kind": "result", "result": {
                "size_bytes": 11, "line_count": 2, "encoding": "utf-8", "newline": "lf",
                "bom": "none", "binary": False}}
            body = json.dumps(result, separators=(",", ":")).encode()
            return SimpleNamespace(cleanup=SimpleNamespace(ok=True), stdout_overflow=False,
                                   timed_out=False, exit_code=0,
                                   stdout=b"SFMW1 " + str(len(body)).encode() + b"\n" + body)

    assert smoke(staged, Executor()).ok
    assert len(calls) == 1
    assert calls[0].inputs["in0"].read_bytes() == (root / "fixtures/sample.txt").read_bytes()


def test_prebuilt_digest_is_checked_before_launch(tmp_path):
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    catalog = tmp_path / "prebuilt"
    unit = catalog / "file_facts"
    unit.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "src/stateframe/mutations_prebuilt/file_facts"
    for name in ("unit.json", "main.py"):
        (unit / name).write_bytes((source / name).read_bytes())
    (catalog / "manifest.json").write_text(json.dumps({
        "schema": "stateframe-mutation-prebuilt/1", "units": {"file_facts": "sha256:" + "0" * 64}}))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.txt").write_text("hello")
    run = tmp_path / "run"
    run.mkdir()
    runtime = MutationRuntime(level="prebuilt", journal=Journal(run), run_dir=None,
                              policy=MutationPolicy(level="prebuilt"), repo=repo,
                              runtime_root=tmp_path / "runtime", catalog_root=catalog,
                              executor_factory=lambda: pytest.fail("untrusted prebuilt launched"))
    reply = runtime.dispatch({"op": "use", "id": "file_facts", "args": json.dumps({
        "input": {"input_id": "in0"}, "inputs": ["sample.txt"]})}, "call-digest")
    assert reply.startswith("REFUSED MUTATION_DIGEST")


def test_prebuilt_digest_refusal_completes_tool_tape(tmp_path):
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime
    from kernel_tool_tape import ToolTape, configure_replay

    run = tmp_path / "run"
    run.mkdir()
    catalog = tmp_path / "prebuilt"
    unit = catalog / "file_facts"
    unit.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "src/stateframe/mutations_prebuilt/file_facts"
    for name in ("unit.json", "main.py"):
        (unit / name).write_bytes((source / name).read_bytes())
    (catalog / "manifest.json").write_text(json.dumps({
        "schema": "stateframe-mutation-prebuilt/1", "units": {"file_facts": "sha256:" + "0" * 64}}))
    repo = tmp_path / "repo"
    repo.mkdir()
    configure_replay(run, active=False)
    runtime = MutationRuntime(level="prebuilt", journal=Journal(run), run_dir=run,
                              policy=MutationPolicy(level="prebuilt"), repo=repo,
                              catalog_root=catalog)
    reply = runtime.dispatch({"op": "use", "id": "file_facts", "args": json.dumps({
        "input": {"input_id": "in0"}, "inputs": []})}, "call-taped")
    assert reply.startswith("REFUSED MUTATION_DIGEST")
    assert ToolTape(run).lookup("call-taped").completed


@pytest.mark.host_sandbox
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux bubblewrap/seccomp isolate mutation payloads",
)
def test_file_facts_use_runs_isolated_and_leaves_no_runtime_root(tmp_path):
    from kernel_mutate_exec import MutationExecutor
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime
    from kernel_mutate_sweep import sweep

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.txt").write_bytes(b"first\nsecond\n")
    run = tmp_path / "run"
    run.mkdir()
    parent = tmp_path / "rt"
    parent.mkdir()
    root = parent / ("a" * 32)
    root.mkdir()
    runtime = MutationRuntime(level="prebuilt", journal=Journal(run), run_dir=None,
                              policy=MutationPolicy(level="prebuilt"), repo=repo,
                              runtime_root=root,
                              executor_factory=lambda: MutationExecutor(root, network_envelope="none"))
    try:
        reply = runtime.dispatch({"op": "use", "id": "file_facts", "args": json.dumps({
            "input": {"input_id": "in0"}, "inputs": ["sample.txt"]})}, "call-real")
        assert "untrusted mutation observation" in reply
        assert '"line_count": 2' in reply
        ledger = [json.loads(row) for row in (root / "launches.jsonl").read_text().splitlines()]
        assert {row["call"] for row in ledger} == {"call-real"}
    finally:
        runtime.close()
    assert not root.exists()
    report = sweep(parent, run_locks=lambda _name: False)
    assert not report.unresolved
    assert not root.exists()


@pytest.mark.host_sandbox
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux bubblewrap/seccomp isolate authored payloads",
)
def test_authored_reader_submit_and_use(tmp_path):
    from kernel_mutate_exec import MutationExecutor
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "odd.dat").write_text("code|37\n")
    run = tmp_path / "run"
    run.mkdir()
    root = tmp_path / "rt" / ("b" * 32)
    root.mkdir(parents=True)
    runtime = MutationRuntime(level="authored", journal=Journal(run), run_dir=None,
                              policy=MutationPolicy(level="authored"), repo=repo,
                              executor_factory=lambda: MutationExecutor(root, network_envelope="none"))
    source = ("def handle(input, ctx):\n"
              "    text = input.get('text') or ctx.input_text(input['input_id'])\n"
              "    return {'value': int(text.split('|')[1])}\n")
    manifest = {"schema": "stateframe-mutation-unit/1", "name": "odd_reader",
                "revision": 1, "parent": None, "entry": "main.py", "summary": "Read odd records",
                "kind": "reader", "input_schema": {"type": "object"},
                "output_schema": {"type": "object", "properties": {"value": {"type": "integer"}},
                                  "required": ["value"], "additionalProperties": False},
                "requires": {"network": False, "writes": False},
                "smoke": [{"input": {"input_id": "in0"}, "inputs": ["odd.dat"],
                           "expect": {"value": 37}}],
                "limits": {"wall_s": 10, "cpu_s": 5, "mem_mb": 256}}
    try:
        assert "OPEN" in runtime.dispatch({"op": "enter", "args": json.dumps({
            "goal": "Read odd records"})}, "enter-odd")
        ready = runtime.dispatch({"op": "submit", "args": json.dumps({
            "manifest": manifest, "files": {"main.py": source}})}, "submit-odd")
        assert "READY" in ready, ready
        result = runtime.dispatch({"op": "use", "id": "odd_reader", "args": json.dumps({
            "input": {"input_id": "in0"}, "inputs": ["odd.dat"]})}, "use-odd")
        assert '"value": 37' in result, result
    finally:
        runtime.close()
    assert not root.exists()


@pytest.mark.host_sandbox
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux bubblewrap mounts the unit at /sf/unit for diagnostic path checks",
)
def test_raising_authored_handler_diagnostic_reaches_reply(tmp_path):
    from kernel_mutate_exec import MutationExecutor
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    repo = tmp_path / "repo"
    repo.mkdir()
    root = tmp_path / "rt" / ("e" * 32)
    root.mkdir(parents=True)
    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path / "run"),
                              policy=MutationPolicy(level="authored"), repo=repo,
                              executor_factory=lambda: MutationExecutor(root, network_envelope="none"))
    try:
        assert "OPEN" in runtime.dispatch({"op": "enter", "args": json.dumps({
            "goal": "Check exception diagnostics"})}, "enter-raise")
        reply = runtime.dispatch({"op": "submit", "args": json.dumps({
            "code": "def handle(input, ctx):\n    raise ValueError('bad row')\n"})},
            "submit-raise")
        assert "REFUSED MUTATION_ADMISSION EXECUTION:" in reply, reply
        assert "HandlerError" in reply and "ValueError: bad row" in reply
        assert "/sf/unit/main.py" in reply and "line 2" in reply
        assert "repair: fix the unit and submit again" in reply
    finally:
        runtime.close()
    assert not root.exists()


@pytest.mark.host_sandbox
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux bubblewrap/seccomp contain broker launches",
)
def test_broker_launch_uses_broker_request_document(tmp_path):
    from kernel_mutate_exec import LaunchSpec, MutationExecutor, MutationLimits

    root = tmp_path / ("c" * 32)
    root.mkdir()
    executor = MutationExecutor(root, network_envelope="full")
    request = {"url": "http://127.0.0.1/", "accept": None,
               "max_bytes": 1024, "timeout_s": 1, "domains": None}
    launch = "d" * 32
    spec = LaunchSpec("broker", "broker-call", launch, None, {}, {},
                      json.dumps({"call": "broker-call", "launch": launch,
                                  "role": "broker", "request": request}).encode(),
                      MutationLimits(wall_s=3, cpu_s=2))
    try:
        outcome = executor.run(spec)
        assert outcome.cleanup.ok
        assert outcome.stdout.startswith(b"SFMB1 "), outcome.stdout[:80]
        header, body = outcome.stdout.split(b"\n", 1)
        metadata_size = int(header.split()[1])
        metadata = json.loads(body[:metadata_size])
        assert metadata["error"] != "invalid_request", metadata
    finally:
        executor.close()


def test_broker_frame_requires_exact_bounded_bytes():
    from kernel_mutate_broker import parse_broker

    metadata = {"status": 200, "headers": {"content-type": "text/plain"},
                "final_url": "https://example.com/x", "redirects": [], "error": None}
    body = b"abc"
    encoded = json.dumps(metadata).encode()
    frame = b"SFMB1 " + str(len(encoded)).encode() + b" " + str(len(body)).encode() + b"\n" + encoded + body
    assert parse_broker(frame, max_bytes=3) == (metadata, body)
    with pytest.raises(ValueError):
        parse_broker(frame + b"!", max_bytes=3)
    with pytest.raises(ValueError):
        parse_broker(frame, max_bytes=2)


def test_phased_need_uses_broker_evidence_and_reinvokes(tmp_path):
    from kernel_mutate_build import source_digest
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    catalog = tmp_path / "catalog"
    unit = catalog / "network_reader"
    unit.mkdir(parents=True)
    manifest = {"schema": "stateframe-mutation-unit/1", "name": "network_reader",
                "revision": 1, "parent": None, "entry": "main.py", "summary": "read",
                "kind": "reader", "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "requires": {"network": True, "writes": False},
                "smoke": [{"input": {}, "inputs": [], "expect": {}}],
                "limits": {"wall_s": 3, "cpu_s": 2, "mem_mb": 128}}
    source = b"def handle(input, ctx): return {'text': ctx.fetch('https://example.com/x').body.decode()}\n"
    (unit / "main.py").write_bytes(source)
    (unit / "unit.json").write_text(json.dumps(manifest))
    (catalog / "manifest.json").write_text(json.dumps({
        "schema": "stateframe-mutation-prebuilt/1", "units": {
            "network_reader": source_digest({"main.py": source}, manifest)}}))
    repo = tmp_path / "repo"
    repo.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    root = tmp_path / "rt"
    root.mkdir()
    calls = []

    class Executor:
        def __init__(self):
            self.root = root

        def run(self, spec):
            calls.append(spec)
            if spec.role == "broker":
                metadata = {"status": 200, "headers": {"content-type": "text/plain"},
                            "final_url": "https://example.com/x", "redirects": [], "error": None}
                encoded = json.dumps(metadata).encode()
                stdout = b"SFMB1 " + str(len(encoded)).encode() + b" 4\n" + encoded + b"DATA"
            else:
                doc = json.loads(spec.stdin_doc)
                if spec.role == "validator":
                    result = {"kind": "result", "result": {"ok": True, "errors": [], "hints": []}}
                elif spec.role == "smoke":
                    result = {"kind": "result", "result": {}}
                elif doc["phase"] == 0:
                    result = {"kind": "need", "fetch": [{"method": "GET",
                        "url": "https://example.com/x", "accept": ""}]}
                else:
                    assert doc["responses"][0]["bytes"] == 4
                    assert spec.responses[next(iter(spec.responses))].read_bytes() == b"DATA"
                    result = {"kind": "result", "result": {"text": "DATA"}}
                body = json.dumps({"wire": 1, "call": spec.call, **result}).encode()
                stdout = b"SFMW1 " + str(len(body)).encode() + b"\n" + body
            return SimpleNamespace(cleanup=SimpleNamespace(ok=True), stdout_overflow=False,
                                   timed_out=False, exit_code=0, stdout=stdout, wall_s=.1, cpu_s=None)

    executor = Executor()
    runtime = MutationRuntime(level="prebuilt", journal=Journal(run), run_dir=None,
                              policy=MutationPolicy(level="prebuilt", network="public-web"),
                              repo=repo, catalog_root=catalog,
                              executor_factory=lambda: executor)
    reply = runtime.dispatch({"op": "use", "id": "network_reader", "args": '{"input":{}}'}, "phased-call")
    assert '"text": "DATA"' in reply, reply
    assert [spec.role for spec in calls] == ["validator", "smoke", "payload", "broker", "payload"]
    assert [row["kind"] for row in runtime.journal.events() if row["kind"].startswith("BROKER_")] == [
        "BROKER_INTENT", "BROKER_COMPLETE"]


@pytest.mark.host_sandbox
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux seccomp qualification and pidfd identity-safe orphan cleanup",
)
def test_scripted_run_catalog_file_facts_authored_reader_and_cleanup(tmp_path):
    import os
    import subprocess
    import sys
    from stateframe.scripted_endpoint import Script, serve

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "report.odd").write_text("name|41\n", encoding="utf-8")
    (repo / "README.txt").write_text("Odd report input\n", encoding="utf-8")
    manifest = {"schema": "stateframe-mutation-unit/1", "name": "odd_reader",
                "revision": 1, "parent": None, "entry": "main.py", "summary": "Read odd rows",
                "kind": "reader", "input_schema": {"type": "object"},
                "output_schema": {"type": "object", "properties": {"value": {"type": "integer"}},
                                  "required": ["value"], "additionalProperties": False},
                "requires": {"network": False, "writes": False},
                "smoke": [{"input": {"text": "sample|7"}, "expect": {"value": 7}}],
                "limits": {"wall_s": 10, "cpu_s": 5, "mem_mb": 256}}
    source = ("def handle(input, ctx):\n"
              "    text = input.get('text') or ctx.input_text(input['input_id'])\n"
              "    return {'value': int(text.split('|')[1])}\n")
    def step(op, ident=None, args=None):
        fields = {"op": op}
        if ident is not None:
            fields["id"] = ident
        if args is not None:
            fields["args"] = json.dumps(args)
        return {"tool": "mutate", "required": True, "arguments": fields}

    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        step("catalog"),
        step("use", "file_facts", {"input": {"input_id": "in0"},
                                    "inputs": ["report.odd"]}),
        step("enter", args={"goal": "Read the pipe delimited odd record"}),
        step("submit", args={"manifest": manifest, "files": {"main.py": source}}),
        step("use", "odd_reader", {"input": {"input_id": "in0"},
                                   "inputs": ["report.odd"]}),
        {"tool": "done", "required": True,
         "arguments": {"summary": "The odd report has value 41."}},
    ]})
    server = serve(script)
    run = tmp_path / "run"
    user_home = tmp_path / "home"
    user_home.mkdir()
    home = user_home / ".stateframe"
    from mutate_leftovers import assert_no_leftovers, snapshot_leftovers
    runtime_parent = home / "rt" / "mutations"
    before_leftovers = snapshot_leftovers(runtime_parent)
    command = [sys.executable, "-m", "stateframe", "run", "--mutations", "authored", "--repo", str(repo),
               "--task", "Read report.odd with a small authored reader and report its value.",
               "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
               "--model", "scripted", "--run-dir", str(run), "--quiet",
               "--max-turns", "20", "--test-python", sys.executable]
    try:
        result = subprocess.run(command, env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                                              "HOME": str(user_home)},
                                capture_output=True, text=True, timeout=420)
        assert result.returncode == 0, "\n".join(
            line for line in (result.stdout + result.stderr).splitlines()
            if "engine error:" in line or "engine result:" in line or "model requests:" in line) + "\n" + \
            _measurement_failure_tail(run)
        script.assert_consumed()
        assert (run / "spec.json").exists(), sorted(path.name for path in run.iterdir())
        assert json.loads((run / "spec.json").read_text()).get("mutations", {}).get("level") == "authored"
        assert (run / "mutations" / "journal.jsonl").exists(), {
            "run": sorted(path.name for path in run.iterdir()),
            "mutation_events": sorted(path.name for path in (run / "mutations").iterdir())
            if (run / "mutations").exists() else [],
            "tool_replies": [[str(message.get("content"))[:140] for message in request.get("messages", [])
                              if message.get("role") == "tool"][-2:]
                             for request in script.requests],
        }
        events = [json.loads(line) for line in
                  (run / "mutations" / "journal.jsonl").read_text().splitlines()]
        assert any(row.get("kind") == "ADMISSION" and row.get("name") == "odd_reader"
                   for row in events)
        terminal = [row for row in events if row.get("kind") == "TERMINAL"]
        import base64
        assert any(b'"value":41' in base64.b64decode(row["frame_b64"])
                   for row in terminal)
        assert not runtime_parent.exists() or not list(runtime_parent.iterdir())
        assert_no_leftovers(before_leftovers, snapshot_leftovers(runtime_parent))
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "linux", reason="mutation authoring launches require the Linux sandbox")
def test_scripted_failed_check_then_authoring_reentry_seals_and_finishes(tmp_path):
    import os
    import subprocess
    import sys
    from stateframe.scripted_endpoint import Script, serve
    from mutate_leftovers import assert_no_leftovers, snapshot_leftovers

    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "main.py").write_text("VALUE = 1\n")
    (repo / "tests/test_once.py").write_text(
        "from main import VALUE\n"
        "def test_once():\n"
        "    assert VALUE == 2\n")
    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "run_tests", "required": True, "arguments": {"target": "tests/test_once.py"}},
        {"tool": "str_replace", "required": True, "arguments": {
            "path": "main.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"}},
        {"tool": "mutate", "required": True, "arguments": {
            "op": "enter", "args": json.dumps({"goal": "Inspect failed check"})}},
        {"tool": "run_tests", "required": True, "arguments": {"target": "tests/test_once.py"}},
        {"tool": "mutate", "required": True, "arguments": {"op": "leave"}},
        {"tool": "done", "required": True, "arguments": {"summary": "The check now passes."}},
    ]})
    server = serve(script)
    run = tmp_path / "run"
    user_home = tmp_path / "home"
    user_home.mkdir()
    runtime_parent = user_home / ".stateframe" / "rt" / "mutations"
    before = snapshot_leftovers(runtime_parent)
    command = [sys.executable, "-m", "stateframe", "run", "--mutations", "authored",
               "--repo", str(repo), "--task", "Run checks for main.py and inspect the first failure.",
               "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
               "--run-dir", str(run), "--max-turns", "12", "--test-python", sys.executable,
               "--quiet"]
    try:
        result = subprocess.run(command, env={**os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "HOME": str(user_home)}, capture_output=True, text=True, timeout=420)
        assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-3000:]
        script.assert_consumed()
        assert len(script.requests) >= 5
        assert "HARNESS_CONTRACT" not in result.stdout + result.stderr
        authoring_request = script.requests[3]
        assert "run_tests" in [tool["function"]["name"]
                               for tool in authoring_request["tools"]]
        assert_no_leftovers(before, snapshot_leftovers(runtime_parent))
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "linux", reason="mutation authoring launches require the Linux sandbox")
def test_stale_parallel_tool_call_during_authoring_refused_through_work(tmp_path):
    import os
    import subprocess
    import sys
    import threading
    from http.server import ThreadingHTTPServer
    from stateframe.scripted_endpoint import Script, make_handler

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("value = 1\n")
    (repo / "test_app.py").write_text("def test_value(): assert True\n")
    marker = repo / "stale-shell-ran"
    script = Script({"model": "scripted", "steps": [
        {"tool": "mutate", "arguments": {"op": "enter", "args": json.dumps({
            "goal": "Inspect app.py", "inputs": [], "kind": "reader", "needs": {}})}},
        {"tool": "mutate", "arguments": {"op": "leave"}},
        {"tool": "done", "required": False, "arguments": {"summary": "Inspected."}},
    ]})
    base_handler = make_handler(script)

    class Handler(base_handler):
        def _send(self, status, payload):
            if status == 200 and len(script.requests) == 2 and "choices" in payload:
                payload["choices"][0]["message"]["tool_calls"].insert(0, {
                    "id": "stale-shell", "type": "function", "function": {"name": "shell",
                    "arguments": json.dumps({"command": f"touch {marker}"})}})
            return super()._send(status, payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    run = tmp_path / "run"
    command = [sys.executable, "-m", "stateframe", "run", "--mutations", "authored",
               "--repo", str(repo), "--task", "Inspect app.py with a mutation.",
               "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
               "--run-dir", str(run), "--max-turns", "6", "--test-python", sys.executable, "--quiet"]
    try:
        subprocess.run(command, env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                                     "HOME": str(tmp_path)}, capture_output=True, text=True, timeout=240)
        assert len(script.requests) >= 3
        replies = [str(message.get("content", "")) for request in script.requests[2:]
                   for message in request.get("messages", []) if message.get("role") == "tool"]
        assert any("REFUSED MUTATION_FRAME" in reply for reply in replies)
        assert not marker.exists()
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux /proc child discovery and identity-safe orphan cleanup",
)
def test_scripted_run_sqlite_input_survives_sigkill_resume_and_cleans_mutation_runtime(tmp_path):
    import os
    import signal
    import sqlite3
    import subprocess
    import sys
    import time
    from stateframe.scripted_endpoint import Script, serve
    from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
    from mutate_leftovers import assert_no_leftovers, snapshot_leftovers

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_report.py").write_text("def test_report(): assert True\n")
    (repo / "reader.py").write_text("def read_report(): return None\n")
    (repo / "report.odd").write_text("name|41\n")
    database = repo / "app.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE settings (key TEXT NOT NULL, value TEXT NOT NULL)")
    connection.execute("INSERT INTO settings VALUES ('theme', 'blue')")
    connection.commit()
    connection.close()
    database_digest = hashlib.sha256(database.read_bytes()).hexdigest()
    manifest = {"schema": "stateframe-mutation-unit/1", "name": "slow_reader",
                "revision": 1, "parent": None, "entry": "main.py", "summary": "Read odd rows",
                "kind": "reader", "input_schema": {"type": "object"},
                "output_schema": {"type": "object", "properties": {"value": {"type": "integer"}},
                                  "required": ["value"], "additionalProperties": False},
                "requires": {"network": False, "writes": False},
                "smoke": [{"input": {"mode": "smoke"}, "expect": {"value": 7}}],
                "limits": {"wall_s": 25, "cpu_s": 5, "mem_mb": 256}}
    source = ("import time\n"
              "def handle(input, ctx):\n"
              "    if input.get('mode') == 'smoke': return {'value': 7}\n"
              "    time.sleep(20)\n"
              "    return {'value': int(ctx.input_text(input['input_id']).split('|')[1])}\n")
    def step(op, ident=None, args=None):
        fields = {"op": op}
        if ident is not None:
            fields["id"] = ident
        if args is not None:
            fields["args"] = json.dumps(args)
        return {"tool": "mutate", "required": True, "arguments": fields}

    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        step("catalog"),
        step("use", "sqlite_peek", {"input": {"input_id": "in0"}, "inputs": ["app.db"]}),
        step("enter", args={"goal": "Read the odd report"}),
        step("submit", args={"manifest": manifest, "files": {"main.py": source}}),
        step("use", "slow_reader", {"input": {"mode": "real", "input_id": "in0"},
                                     "inputs": ["report.odd"]}),
        {"tool": "done", "required": True,
         "arguments": {"summary": "The interrupted reader was recorded."}},
    ]})
    server = serve(script)
    run = tmp_path / "run"
    home = tmp_path / "home"
    home.mkdir()
    runtime_parent = home / ".stateframe" / "rt" / "mutations"
    before = snapshot_leftovers(runtime_parent)
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                   "STATEFRAME_MUTATIONS": "authored", "HOME": str(home)}
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    command = [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
               "--task", "Read app.db with sqlite_peek, then report.odd with a slow authored reader.",
               "--target", "reader.py",
               "--endpoint", endpoint, "--model", "scripted", "--run-dir", str(run),
               "--quiet", "--max-turns", "20", "--test-python", sys.executable]
    launched_at = time.time()
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    child_pid = None
    slow_reader_call = None
    last_rows = []
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(f"run exited before payload launch: {process.stdout.read()}")
            tape_path = run / "tool-tape.jsonl"
            if slow_reader_call is None and tape_path.is_file():
                try:
                    tape_rows = [json.loads(line) for line in tape_path.read_text().splitlines()]
                except ValueError:
                    tape_rows = []
                calls = [row["call_id"] for row in tape_rows if row.get("kind") == "intent"
                         and row.get("tool") == "mutate" and row.get("args", {}).get("op") == "use"
                         and row["args"].get("id") == "slow_reader"]
                assert len(calls) <= 1, calls
                slow_reader_call = calls[0] if calls else None
            spec_path = run / "spec.json"
            if slow_reader_call is not None and spec_path.is_file():
                seed = json.loads(spec_path.read_text()).get("run_identity_seed")
                ledger = runtime_parent / str(seed) / "launches.jsonl"
                if ledger.is_file():
                    try:
                        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
                    except ValueError:
                        rows = []
                    last_rows = rows[-6:]
                    if any(row.get("call") == slow_reader_call and row.get("role") == "payload"
                           and row.get("state") == "STARTED"
                           for row in rows):
                        children = Path(f"/proc/{process.pid}/task/{process.pid}/children")
                        child_pids = [int(value) for value in children.read_text().split()]
                        matching = [pid for pid in child_pids if "_child.py" in
                                    Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()]
                        assert len(matching) == 1, matching
                        child_pid = matching[0]
                        break
            time.sleep(0.1)
        assert child_pid is not None, ("slow reader payload did not reach STARTED",
                                       script.position, slow_reader_call, last_rows)
        assert hashlib.sha256(database.read_bytes()).hexdigest() == database_digest
        child_identity = record_identity(child_pid, launched_after=launched_at,
                                         command_fragment="_child.py")
        signal_run_verified(run, child_pid, child_identity, signal.SIGKILL,
                            deadline=deadline, what="slow mutation payload")
        process.communicate(timeout=90)
        resumed = subprocess.run([sys.executable, "-m", "stateframe", "resume", str(run),
                                  "--endpoint", endpoint], env=environment,
                                 capture_output=True, text=True, timeout=420)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        script.assert_consumed()
        assert hashlib.sha256(database.read_bytes()).hexdigest() == database_digest
        tape = [json.loads(line) for line in (run / "tool-tape.jsonl").read_text().splitlines()]
        def use_call(name):
            calls = [row["call_id"] for row in tape if row.get("kind") == "intent"
                     and row.get("tool") == "mutate" and row.get("args", {}).get("op") == "use"
                     and row["args"].get("id") == name]
            assert len(calls) == 1, calls
            return calls[0]

        events = [json.loads(line) for line in
                  (run / "mutations" / "journal.jsonl").read_text().splitlines()]
        sqlite_call = use_call("sqlite_peek")
        sqlite_admissions = [row for row in events if row.get("kind") == "ADMISSION"
                             and row.get("call") == sqlite_call]
        assert len(sqlite_admissions) == 1
        sqlite_results = [row for row in events if row.get("call") == sqlite_call
                          and row.get("kind") in {"INTERRUPTED", "TERMINAL"}]
        assert len(sqlite_results) == 1
        assert sqlite_results[0]["kind"] == "TERMINAL"
        import base64
        frame = base64.b64decode(sqlite_results[0]["frame_b64"])
        assert b"settings" in frame and b"blue" in frame
        reader_call = use_call("slow_reader")
        assert reader_call == slow_reader_call
        reader_admissions = [row for row in events if row.get("kind") == "ADMISSION"
                             and row.get("call") == reader_call]
        assert len(reader_admissions) == 1
        reader_results = [row for row in events if row.get("call") == reader_call
                          and row.get("kind") in {"INTERRUPTED", "TERMINAL"}]
        assert len(reader_results) == 1
        assert reader_results[0]["kind"] == "INTERRUPTED"
        outcome, reason, *_ = reader_results[0]["observation"].split()
        assert outcome == "REFUSED"
        assert reason == "MUTATION_INTERRUPTED"
        assert not runtime_parent.exists() or not list(runtime_parent.iterdir())
        assert_no_leftovers(before, snapshot_leftovers(runtime_parent))
    finally:
        if process.poll() is None:
            signal_run_owned(run, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="test cleanup", require_checkpoint_saved=False)
            process.wait(timeout=10)
        if process.stdout is not None:
            process.stdout.close()
        server.shutdown()
        server.server_close()

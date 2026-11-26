"""Report chronology is evidence-bound and optional at the provider boundary."""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

from _timeouts import scaled
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
from stateframe.scripted_endpoint import Script, serve
from tools.pinned_tree import pinned_tree


ROOT = Path(__file__).resolve().parents[1]
BASE = "29765b39820fd39800fd7292f52fb5bcb81e6069"
SWITCH = "STATEFRAME_REPORT_CHRONOLOGY"
HEADER = "[REPORT CHRONOLOGY]"
REPORT = (
    "The résumé preview showed the wrong price. Before that, I applied the discount. "
    "Before that, the résumé preview showed the correct price."
)
DRIVER = """
import secrets, sys
from types import SimpleNamespace
from stateframe import cli, profiles, runner

if sys.argv[1] == 'ordinary':
    # Discovery has no starting file hint and uses the ordinary prompt.
    build_row = runner.build_row
    def without_starting_hint(*args, **kwargs):
        return {**build_row(*args, **kwargs), 'entry': {'path': '', 'fn': ''},
                'path': '', 'rel': '', 'files': {}, 'scoped_edits': True,
                'scope_source': 'discovery'}
    runner.build_row = without_starting_hint
    load = profiles.load
    profiles.load = lambda name: {**load(name), 'STATEFRAME_WORKING_SET_STABLE_FIRST': 'off',
                                  'STATEFRAME_CONSTANT_TOOL_MENU': 'off'}
runner.secrets = SimpleNamespace(token_hex=lambda n: '0123456789abcdef' * 2 if n == 16
                                else secrets.token_hex(n))
sys.exit(cli.main(sys.argv[2:]))
"""

REQUALIFICATION_DRIVER = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
import kernel_request_reprepare as reprepare
import kernel_request_source_window as source_window
from kernel_request_evidence_closure import source_revision

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=True)
source = "x = 1\\n"
(root / "price.py").write_text(source)
revision = source_revision(source)
content = ("[Current source: price.py]\\nx = 0\\n"
           "Coverage: lines 1-1, columns 0-0; current bytes are delivered for this range.")
body = {"messages": [{"role": "user", "content": content}], "tools": []}
wire = json.dumps(body).encode()
plan = {"units": [{"path": "price.py", "start_line": 1, "end_line": 1}],
        "revisions": {"price.py": revision}, "paths": ["price.py"],
        "target_path": "price.py"}
mismatch = {"kind": "source_range_bytes_mismatch", "path": "price.py", "message": 0}
delivery = {"integrity_omissions": [mismatch], "proof_failures": [mismatch]}
source_window.verify_final_source_delivery = lambda *args, **kwargs: {
    "status": "COMPLETE", "proof_status": "VERIFIED", "proof_failures": []}
result = reprepare._source_requalification_attempt(
    SimpleNamespace(), SimpleNamespace(owner=SimpleNamespace(root=root)), plan,
    {"price.py": source}, body, wire, delivery)
assert result is not None and result[4] == [("price.py", "rerendered_live_range")]
assert "x = 1\\n" in result[0]["messages"][0]["content"]
print(result[1].hex())
"""


def _source_requalification_wire(tree, work):
    repo = work / "repo"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((
        str(tree / "src/stateframe/engine"), str(tree / "src"),
    ))
    result = subprocess.run(
        [sys.executable, "-c", REQUALIFICATION_DRIVER, str(repo)],
        cwd=tree, env=env, text=True, capture_output=True, timeout=scaled(30),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return bytes.fromhex(result.stdout.strip())


def _exercise(tree, work, *, switch=None, report=REPORT, view="operational", resume=None,
              first_only=False, capacity_limit=None):
    repo, run = work / "repo", work / "run"
    if not repo.exists():
        shutil.copytree(ROOT / "examples/discount_bug", repo)
    document = json.loads((ROOT / "examples/discount_bug.script.json").read_text())
    if view == "ordinary":
        document["steps"] = [
            {"tool": "read", "arguments": {"path": path}}
            for path in ("shop/pricing.py", "tests/test_pricing.py", "shop/__init__.py")
        ]
    if first_only:
        document["steps"] = document["steps"][:1]
    script = Script(document)
    raw_requests = []
    interrupted, release = threading.Event(), threading.Event()
    server = serve(script)

    class RecordingHandler(server.RequestHandlerClass):
        def do_POST(self):
            if self.path == "/v1/chat/completions":
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                raw_requests.append(raw)
                if capacity_limit is not None and len(raw_requests) == 1:
                    payload = json.dumps({"error": {
                        "message": f"maximum context length is {capacity_limit} tokens",
                    }}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if resume and len(raw_requests) == 3:
                    interrupted.set()
                    release.wait(scaled(60))
                    return
                original = self.rfile
                try:
                    self.rfile = io.BytesIO(raw)
                    super().do_POST()
                finally:
                    self.rfile = original
            else:
                super().do_POST()

    server.RequestHandlerClass = RecordingHandler
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    env = {name: value for name, value in os.environ.items()
           if not name.startswith(("STATEFRAME_", "REACT_", "SF_LAB_"))}
    env.update(PYTHONPATH=str(tree / "src"), STATEFRAME_HOME=str(work / "home"),
               STATEFRAME_PROVIDER="llama.cpp", SF_LAB_DETFIX3="on")
    if switch is not None:
        env[SWITCH] = switch
    if view == "ordinary":
        env["STATEFRAME_WORKING_SET_ALLOCATOR"] = "off"
    if resume != "checkpoint":
        env["STATEFRAME_CHECKPOINT"] = "off"
    cli = [sys.executable, "-c", DRIVER, view]
    command = [*cli, "run", "--repo", str(repo), "--task", report, "--endpoint", endpoint,
               "--model", "scripted", "--run-dir", str(run), "--test-python", sys.executable,
               "--quiet"]
    if resume == "checkpoint":
        command.append("--keep-checkpoints")
    if first_only:
        command += ["--max-turns", "1"]
    elif view == "ordinary":
        command += ["--max-turns", "3"]
    process = None
    try:
        if resume:
            process = subprocess.Popen(command, cwd=tree, env=env, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            deadline = time.monotonic() + scaled(420)
            while not interrupted.wait(0.1):
                if process.poll() is not None:
                    pytest.fail("initial run exited before interruption: " + str(process.communicate()))
                assert time.monotonic() < deadline, "initial run did not reach its third request"
            child, = subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()
            pid = int(child)
            identity = record_identity(pid, launched_after=0, command_fragment="_child.py")
            signal_run_verified(run, pid, identity, signal.SIGKILL,
                                deadline=time.monotonic() + scaled(90), what="chronology parity",
                                require_checkpoint_saved=resume == "checkpoint")
            release.set()
            process.communicate(timeout=scaled(90))
            command = [*cli, "resume", str(run), "--endpoint", endpoint, "--keep-checkpoints"]
        result = subprocess.run(command, cwd=tree, env=env, text=True, capture_output=True,
                                timeout=scaled(420))
        if first_only:
            assert result.returncode == 1, result.stdout + result.stderr
            summary = json.loads((run / "summary.json").read_text())
            assert summary["engine_result"] == "RESPONSE_BUDGET_EXHAUSTED"
        else:
            allowed_codes = {0, 2} if view == "ordinary" else {0}
            assert result.returncode in allowed_codes, result.stdout + result.stderr
        script.assert_consumed()
        expected_sends = len(document["steps"]) + bool(resume) + bool(capacity_limit)
        assert len(raw_requests) == expected_sends, [len(raw) for raw in raw_requests]
        if resume:
            assert raw_requests[2] == raw_requests[3]
            spec = json.loads((run / "spec.json").read_text())
            assert bool(spec["resume"].get("checkpoint")) == (resume == "checkpoint")
        if view == "ordinary":
            task = json.loads(raw_requests[0])["messages"][1]["content"]
            assert task.startswith(f"Task: {report}\n\n") and "\nBegin." in task
        run.rename(work / f"completed-{len(list(work.glob('completed-*')))}")
        return raw_requests
    finally:
        release.set()
        if process is not None and process.poll() is None:
            with suppress(ProcessLookupError):
                signal_run_owned(
                    run, process, signal.SIGTERM, deadline=time.monotonic(),
                    what="chronology parity cleanup", require_checkpoint_saved=False,
                )
            process.communicate(timeout=scaled(30))
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(("view", "resume"), [("ordinary", "replay"), ("operational", "checkpoint")])
def test_r001_bug_told_out_off_request_bytes_match_pinned_099(tmp_path, tmp_path_factory, view, resume):
    base = pinned_tree(BASE, tmp_path_factory)
    expected = _exercise(base, tmp_path, view=view, resume=resume)
    assert _exercise(ROOT, tmp_path, view=view, resume=resume) == expected
    assert _exercise(ROOT, tmp_path, switch="false", view=view, resume=resume) == expected


def test_r001_bug_told_out_off_requalification_bytes_match_pinned_099(
        tmp_path, tmp_path_factory):
    base = pinned_tree(BASE, tmp_path_factory)
    requalified = _source_requalification_wire(base, tmp_path / "pinned-requalification")
    assert _source_requalification_wire(ROOT, tmp_path / "current-requalification") == requalified
    requalified_body = json.loads(requalified)
    assert requalified_body["messages"][0]["content"].endswith(
        "Coverage: lines 1-1, columns 0-0; current bytes are delivered for this range.")
    assert "x = 1\n" in requalified_body["messages"][0]["content"]

    settings = {"capacity_limit": 32768, "first_only": True}
    expected = _exercise(base, tmp_path / "pinned-capacity", **settings)
    assert len(expected) == 2 and expected[0] != expected[1]
    assert _exercise(ROOT, tmp_path / "current-capacity", **settings) == expected
    assert _exercise(ROOT, tmp_path / "current-capacity-false", switch="false",
                     **settings) == expected


def test_r001_bug_told_out_switch_defaults_off(monkeypatch):
    from stateframe import profiles, runner

    monkeypatch.delenv(SWITCH, raising=False)
    assert SWITCH in runner.OPT_IN_SWITCHES
    assert SWITCH in runner.engine_switches()
    assert profiles.load("r2hc3").get(SWITCH, "off") == "off"
    assert SWITCH not in runner.resolve_opt_in_switches()


@pytest.mark.parametrize("chronological", [False, True], ids=["reversal", "chronological"])
def test_r001_bug_told_out_on_path_is_bounded_and_idempotent(tmp_path, monkeypatch, chronological):
    from kernel_request_evidence_closure import report_chronology, ticket_digest
    from kernel_stateframe_framefit5 import _add_report_chronology

    if chronological:
        report = "The preview showed the correct price before I applied the discount."
        assert _exercise(ROOT, tmp_path, report=report, switch="on", first_only=True) == _exercise(
            ROOT, tmp_path, report=report, first_only=True)
        return

    row = {"task_id": "ticket", "msg": REPORT}
    chronology = report_chronology(row, REPORT)
    assert chronology is not None
    assert chronology.ticket.sha256 == ticket_digest(REPORT)
    assert report_chronology({**row, "_authorized_ticket": "Private alternate."}, REPORT) is None
    assert report_chronology(row, "Alternate public brief.") is None
    for report in (
        "The preview showed the wrong price after I applied the discount.",
        "Before the preview showed the wrong price, I applied the discount.",
    ):
        assert report_chronology({"msg": report}, report) is not None
    for report in (
        "The preview did not show an error before I applied the discount.",
        REPORT + " I retried with another discount.",
        REPORT + " The browser hung.",
        "The browser hung. Before I applied the discount, the preview showed the correct price.",
        "The preview worked before the app crashed. The preview worked after the app crashed.",
        "The preview showed the correct price before I applied the discount.",
        "The preview showed " + "é" * 1000 + " after I applied the discount.",
    ):
        assert report_chronology({"msg": report}, report) is None

    owner = SimpleNamespace(task_row=row, _report_chronology=chronology)
    runtime = SimpleNamespace(owner=owner, frame_ceiling=40000)
    monkeypatch.setattr("kernel_request_budget._request_token_meter", lambda *_:
                        SimpleNamespace(measure=lambda _: (10, "tokenize")))
    messages = [{"role": "system", "content": "system"},
                {"role": "user", "content": "Task: " + REPORT},
                {"role": "user", "content": "[StateFrame]\n{}"}]
    baseline = list(messages)
    _add_report_chronology(messages, runtime, {"price.py": "x" * runtime.frame_ceiling})
    assert messages == baseline
    _add_report_chronology(messages, runtime, {})
    _add_report_chronology(messages, runtime, {})
    assert messages == baseline + [{"role": "user", "content": chronology.panel}]

    for raw in _exercise(ROOT, tmp_path, switch="on", resume="checkpoint"):
        contents = [message.get("content") or "" for message in json.loads(raw)["messages"]]
        assert sum(content.count(HEADER) for content in contents) == 1
        assert any(REPORT in content for content in contents)
        panel_with_call, = [content for content in contents if HEADER in content]
        panel = panel_with_call[panel_with_call.index(HEADER):]
        assert panel == chronology.panel
        spans = re.findall(r'bytes \[(\d+), (\d+)\) ("[^"\n]+")', panel)
        assert [json.loads(quote) for _, _, quote in spans] == [
            "the résumé preview showed the correct price", "Before that, ",
            "I applied the discount", "Before that, ", "The résumé preview showed the wrong price",
        ]
        for start, end, quote in spans:
            assert REPORT.encode()[int(start):int(end)].decode() == json.loads(quote)
        assert panel.count("Intervening transition unknown.") == 2


def test_r001_report_chronology_rejects_cross_attempt_event_linkage():
    from kernel_request_evidence_closure import report_chronology

    report = (
        "The app crashed in Chrome after I clicked Save. "
        "The app worked in Firefox before I clicked Save."
    )
    assert report_chronology({"msg": report}, report) is None


def test_r001_rejects_order_cues_inside_unicode_quoted_logs():
    from kernel_request_evidence_closure import report_chronology

    reports = (
        "The app's log displayed «The connection was closed after the client's browser sent a request».",
        "The log displayed ‹The connection was closed after the client sent a request›.",
        "The log displayed «The connection was closed after the client sent a request”.",
        "The log displayed 「The connection was closed after the client sent a request」.",
        "The log displayed ❛The connection was closed after the client sent a request❜.",
    )
    for report in reports:
        assert report_chronology({"msg": report}, report) is None, report


def test_r001_preserves_outer_order_with_quoted_plural_possessive():
    from kernel_request_evidence_closure import report_chronology

    report = "The log displayed “The users’ session expired” after I reopened the project."
    chronology = report_chronology({"msg": report}, report)
    assert chronology is not None
    spans = re.findall(r'bytes \[(\d+), (\d+)\) ("[^"\n]+")', chronology.panel)
    assert [json.loads(quote) for _, _, quote in spans] == [
        "I reopened the project", "after", "The log displayed “The users’ session expired”",
    ]
    for start, end, quote in spans:
        assert report.encode()[int(start):int(end)].decode() == json.loads(quote)


def test_r001_report_chronology_rejects_order_cues_inside_quoted_logs():
    from kernel_request_evidence_closure import report_chronology

    reports = (
        'The log displayed "The connection was closed after the client sent a request".',
        "The log displayed “The connection was closed after the client sent a request”.",
        "The log displayed “The connection was closed after the client sent a request to the app’s port”.",
        "The log displayed ‘The connection was closed after the client sent a request’ on the app's port.",
        "The log displayed ‘The résumé was rejected after the café sent the app’s request’ on its port.",
        "The log displayed ‘The connection was closed after the client sent a request\".",
        "The logs' display showed 'The connection was closed after the client sent a request' on the logs' port.",
    )
    for report in reports:
        assert report_chronology({"msg": report}, report) is None

    report = "The résumé showed ‘session closed’ on the app's port after I opened the café."
    chronology = report_chronology({"msg": report}, report)
    assert chronology is not None
    spans = re.findall(r'bytes \[(\d+), (\d+)\) ("[^"\n]+")', chronology.panel)
    assert [json.loads(quote) for _, _, quote in spans] == [
        "I opened the café", "after", "The résumé showed ‘session closed’ on the app's port",
    ]
    for start, end, quote in spans:
        assert report.encode()[int(start):int(end)].decode() == json.loads(quote)


def test_r001_report_chronology_reserves_source_read_at_6000_token_limit(monkeypatch):
    import kernel_request_budget
    from kernel_request_evidence_closure import report_chronology
    from kernel_stateframe_framefit5 import _add_report_chronology

    report = REPORT
    chronology = report_chronology({"msg": report}, report)
    assert chronology is not None
    source = "\n".join(f"# source line {index}" for index in range(178))
    owner = SimpleNamespace(task_row={"msg": report}, _report_chronology=chronology)
    capture = SimpleNamespace(final_tool_surface={"tools": []})
    runtime = SimpleNamespace(owner=owner, frame_ceiling=6000, _active_capture=capture)

    def measure(body):
        tokens = len(json.dumps(body, ensure_ascii=False).encode("utf-8")) // 4
        return tokens, "tokenize"

    monkeypatch.setattr(kernel_request_budget, "ENDPOINT_PROMPT_TOKEN_BUDGET", 6000)
    monkeypatch.setattr(kernel_request_budget, "ENDPOINT_SAFETY_HEADROOM_TOKENS", 4096)
    monkeypatch.setattr(kernel_request_budget, "_request_token_meter",
                        lambda *_: SimpleNamespace(measure=measure))
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: " + report},
        {"role": "user", "content": "[StateFrame]\n" + "x" * 4000},
        {"role": "user", "content": "[Source read]\n" + source},
    ]
    projected = messages[:3] + [{"role": "user", "content": chronology.panel}] + messages[3:]
    tools = capture.final_tool_surface["tools"]
    baseline_tokens, _ = measure({"messages": messages, "tools": tools})
    projected_tokens, _ = measure({"messages": projected, "tools": tools})
    assert baseline_tokens + 4096 <= 6000
    assert projected_tokens + 4096 > 6000

    # A panel admitted on the prior request must not consume room reserved for
    # the next request's complete source read.
    prior_request = [dict(message) for message in projected]
    _add_report_chronology(prior_request, runtime, {})
    assert prior_request == messages

    # Final assembly measures the source read already present on the wire and
    # leaves the panel out when it would consume the reserved request room.
    _add_report_chronology(messages, runtime, {})
    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: " + report},
        {"role": "user", "content": "[StateFrame]\n" + "x" * 4000},
        {"role": "user", "content": "[Source read]\n" + source},
    ]


def test_r001_report_chronology_does_not_shorten_next_admitted_read(tmp_path, monkeypatch):
    import kernel_read_admission as admission
    import kernel_request_budget
    from kernel_cache_layout_candidate import _project_final_cache_layout, final_layout_mode
    from kernel_cache_layout_view import FINAL_LAYOUT_NAME, semantic_messages
    from kernel_request_evidence_closure import report_chronology
    from kernel_source_delivery import deliver_source, normalize_read_request
    from kernel_stateframe_framefit5 import (
        _add_report_chronology,
        prepare_report_chronology_read_admission,
        restore_report_chronology_read_admission,
    )
    from kernel_working_set_live import read_admission_floor
    from kernel_working_set_prepare import Measurement
    from kernel_working_set_proof import verify_certificate

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.delenv("STATEFRAME_FINAL_CACHE_LAYOUT", raising=False)
    assert final_layout_mode() == "on"
    monkeypatch.setattr(kernel_request_budget, "ENDPOINT_PROMPT_TOKEN_BUDGET", 6000)
    monkeypatch.setattr(kernel_request_budget, "ENDPOINT_SAFETY_HEADROOM_TOKENS", 4096)
    monkeypatch.setattr(
        kernel_request_budget, "_request_token_meter",
        lambda *_: SimpleNamespace(measure=lambda body: (
            len(json.dumps(body).encode()) // 4, "tokenize")),
    )

    def measure(wire):
        return Measurement(len(wire) // 4, len(wire) // 4, "tokenize")

    monkeypatch.setattr("kernel_working_set_live.read_admission_measure", lambda _: measure)
    report = REPORT
    chronology = report_chronology({"msg": report}, report)
    assert chronology is not None
    source_path = ROOT / "src/stateframe/engine/kernel_native_restart.py"
    raw = source_path.read_bytes()
    path = source_path.name
    (tmp_path / path).write_bytes(raw)
    args = {"path": path, "max_lines": 178}
    request = normalize_read_request(args)
    source = deliver_source(raw, path, 0, args=request)
    limits = "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2"

    def admitted_source(chronology_enabled):
        owner = SimpleNamespace(
            root=tmp_path,
            task_row={"msg": report},
            _report_chronology=chronology if chronology_enabled else None,
        )
        capture = SimpleNamespace(final_tool_surface={"tools": []}, qualified_request=None)
        runtime = SimpleNamespace(owner=owner, frame_ceiling=6000, _active_capture=capture)
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Task: " + report},
            {"role": "user", "content": limits},
            {"role": "user", "content": "[StateFrame]\n" + json.dumps(
                {"state": {"framefit5": {"owner_protocol": {
                    "schema": "stateframe-framefit5-owner/1",
                    "continuation_limits": limits,
                }}}},
                sort_keys=True,
                separators=(",", ":"))},
            {"role": "user", "content": "[WIRE VOLATILE TAIL]"},
        ]
        if chronology_enabled:
            _add_report_chronology(messages, runtime, {})
        messages[0]["content"] += "x" * 12401
        body = _project_final_cache_layout({"model": "local", "messages": messages, "tools": []})
        assert body["messages"][-1]["name"] == FINAL_LAYOUT_NAME
        assert any(chronology.panel in row["content"]
                   for row in body["messages"]) == chronology_enabled
        sealed_wire = json.dumps(body).encode()
        sealed_qualified = SimpleNamespace(body=body)
        capture.qualified_request = sealed_qualified
        admission.note_sealed_request(runtime, body["messages"], body["tools"])
        runtime.native_batch = SimpleNamespace(pending=[{
            "id": "read-source", "type": "function",
            "function": {"name": "read", "arguments": json.dumps(args)},
        }])
        runtime._read_attempt_scope = SimpleNamespace(call_id="read-source")
        read_messages, restore = prepare_report_chronology_read_admission(runtime, messages)
        try:
            admitted = admission.admit(
                runtime, raw=raw, display=path, start=0,
                request=request, source=source, visible_messages=read_messages,
            )
        finally:
            if restore is not None:
                restore_report_chronology_read_admission(runtime, restore)
        assert capture.qualified_request is sealed_qualified
        assert json.dumps(capture.qualified_request.body).encode() == sealed_wire
        assert json.dumps(runtime._read_admission_provider_body).encode() == sealed_wire
        assert admitted.result_class == "OK", admitted.result_class
        state = runtime._read_admission_state
        floor = read_admission_floor(state.snapshot, state.pages, require_feasible=True, measure=measure)
        assert floor is not None
        assert verify_certificate(floor.wire, floor.certificate)
        assert HEADER.encode() not in floor.wire
        rows = semantic_messages(json.loads(floor.wire)["messages"])
        assert any(admitted.text in (row.get("content") or "") for row in rows)
        _add_report_chronology(rows, runtime, {})
        assert all(HEADER not in (row.get("content") or "") for row in rows)
        return admitted.metadata["effective_range"], admitted.content

    baseline_range, baseline_content = admitted_source(False)
    assert 0 < baseline_range["end_line"] < args["max_lines"]
    assert admitted_source(True) == (baseline_range, baseline_content)

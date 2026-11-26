from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from _journal import measurement_events
from _stateframe_home import isolated_stateframe_home
from _timeouts import scaled
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, make_handler
from tools.pinned_tree import pinned_tree

ROOT = Path(__file__).resolve().parents[1]
BASE = "29765b39820fd39800fd7292f52fb5bcb81e6069"
SWITCH = "STATEFRAME_EXAMPLE_SLOTS"
BRIEF = (
    "Fix the pricing code so the tests pass.\n\n"
    "Example: `/teams/teams`\n"
    "The second `teams` is the caller-supplied team name."
)
ANNOTATION = "Annotated example (⟦…⟧ marks a runtime input): `/teams/⟦teams⟧`"
AMBIGUOUS = BRIEF.replace("second `teams`", "`teams`")
DISCOVERY = BRIEF.replace("Fix the pricing code so the tests pass.", "Make the public checks pass.")

_LAUNCH = """
import secrets, sys
from types import SimpleNamespace
from stateframe import cli, runner

runner.secrets = SimpleNamespace(
    token_hex=lambda n: "0123456789abcdef" * 2 if n == 16 else secrets.token_hex(n),
)
raise SystemExit(cli.main(sys.argv[1:]))
"""


def _requests(tree, work, switch, *, brief=BRIEF, entry=True, resume=None,
              route_escalation=False, forced_route_requalification=False):
    work.mkdir()
    repo = work / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    if not entry:
        # Use a large repo and a check with no static product import to steer scope.
        extra = repo / "extras"
        extra.mkdir()
        for index in range(201):
            (extra / f"module_{index:03d}.py").write_text("VALUE = 0\n")
        check = repo / "tests/test_pricing.py"
        check.write_text("from importlib import import_module\n\n" + check.read_text().replace(
            "from shop.pricing import apply_discount, order_total",
            "pricing = import_module('shop.pricing')\n"
            "apply_discount, order_total = pricing.apply_discount, pricing.order_total",
        ))
    run = work / "run"
    waiting, release = threading.Event(), threading.Event()
    raw_requests = []
    route_profile = None
    if route_escalation:
        profile = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text())
        profile["switches"]["STATEFRAME_ROUTING_SCHEMA"] = "2"
        route_profile = work / "routing-schema-2.json"
        route_profile.write_text(json.dumps(profile))

    class ScriptedRun(Script):
        def next_call(self, offered):
            if resume is not None and self.position == 1 and not release.is_set():
                waiting.set()
                if not release.wait(scaled(60)):
                    raise TimeoutError("test did not release the interrupted request")
                raise ScriptStepUnavailable("interrupted before a reply")
            return super().next_call(offered)

    script_document = json.loads((ROOT / "examples/discount_bug.script.json").read_bytes())
    if route_escalation:
        script_document["models"] = ["reader", "editor"]
        write = {"tool": "str_replace", "required": True, "arguments": {
            "path": "shop/pricing.py",
            "old_str": "    return round(price * percent / 100, 2)",
            "new_str": "    return round(price * (100 - percent) / 100, 2)",
        }}
        if forced_route_requalification:
            # An edit-only response forces react_repair.work through its
            # measurement.prepare(forced_route=..., frozen_route_capture=...)
            # requalification branch; no read is run first.
            script_document["steps"] = [
                write,
                write,
                {"tool": "run_tests", "required": True,
                 "arguments": {"target": "tests/test_pricing.py"}},
                {"tool": "done", "required": True,
                 "arguments": {"summary": "The pricing correction passes the test suite."}},
            ]
        else:
            script_document["steps"] = [
                {"tool": "read", "required": True, "arguments": {"path": "shop/pricing.py"},
                 "also": [write]},
                write,
                {"tool": "run_tests", "required": True,
                 "arguments": {"target": "tests/test_pricing.py"}},
                {"tool": "done", "required": True,
                 "arguments": {"summary": "The pricing correction passes the test suite."}},
            ]
    script = ScriptedRun(script_document)

    class CaptureRawChat(make_handler(script)):
        def do_POST(self):
            if self.path == "/v1/chat/completions":
                size = int(self.headers["Content-Length"])
                # Tee the handler's read to capture the actual HTTP entity.
                stream = self.rfile

                class CaptureBody:
                    def read(self, length):
                        payload = stream.read(length)
                        assert len(payload) == length == size
                        raw_requests.append(payload)
                        return payload

                self.rfile = CaptureBody()
                try:
                    super().do_POST()
                finally:
                    self.rfile = stream
            else:
                super().do_POST()

    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureRawChat)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STATEFRAME_")}
    environment.update(
        PYTHONPATH=str(tree / "src"), STATEFRAME_HOME=isolated_stateframe_home(),
        STATEFRAME_PROVIDER="llama.cpp",
        STATEFRAME_CHECKPOINT="on" if resume == "checkpoint" else "off",
    )
    if switch is not None:
        environment[SWITCH] = switch
    command = [sys.executable, "-c", _LAUNCH]
    options = dict(cwd=work, env=environment, text=True)
    run_args = [
        "run", "--repo", str(repo), "--task", brief, "--endpoint", endpoint,
        "--model", "scripted", "--run-dir", str(run), "--test-python", sys.executable,
        "--max-turns", "16", "--shell", "off", "--quiet", "--keep-checkpoints",
    ]
    if not entry:
        run_args += ["--edit-scope", "widen"]
    if route_escalation:
        run_args += ["--route", "plan.model=reader", "--route", "edit.model=editor",
                     "--profile", str(route_profile)]
    process = None
    try:
        if resume is None:
            finished = subprocess.run(command + run_args, **options, capture_output=True,
                                      timeout=scaled(600))
        else:
            process = subprocess.Popen(command + run_args, **options, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            deadline = time.monotonic() + scaled(600)
            while not waiting.wait(0.25):
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    pytest.fail(stdout + stderr)
                assert time.monotonic() < deadline, "provider did not reach the interruption point"
            assert len(raw_requests) == 2
            child, = subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()
            identity = record_identity(int(child), launched_after=0, command_fragment="_child.py")
            signal_run_verified(run, int(child), identity, signal.SIGKILL,
                                deadline=time.monotonic() + scaled(60), what="example slots parity",
                                require_checkpoint_saved=resume == "checkpoint")
            release.set()
            process.communicate(timeout=scaled(60))
            finished = subprocess.run(
                command + ["resume", str(run), "--endpoint", endpoint, "--quiet", "--keep-checkpoints"],
                **options, capture_output=True, timeout=scaled(600),
            )
        assert finished.returncode == 0, finished.stdout + finished.stderr
        if resume is not None:
            assert raw_requests[1] == raw_requests[2], "resume rebuilt different in-flight bytes"
        if resume == "checkpoint":
            events = (run / "checkpoints" / "events.jsonl").read_text().splitlines()
            assert any(json.loads(line)["event"] == "checkpoint_restored" for line in events)
        script.assert_consumed()
    finally:
        release.set()
        if process is not None and process.poll() is None:
            signal_run_owned(run, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="example slots cleanup", require_checkpoint_saved=False)
            process.communicate(timeout=scaled(10))
        server.shutdown()
        server.server_close()
        thread.join()
    sends = [json.loads(line) for line in (run / "sends.jsonl").read_text().splitlines()]
    sent_hashes = {record["request_sha256"] for record in sends}
    assert {hashlib.sha256(raw).hexdigest() for raw in raw_requests} <= sent_hashes
    assert len(raw_requests) == len(script.steps) + (resume is not None)
    task = next(row["content"] for row in json.loads(raw_requests[0])["messages"]
                if (row.get("content") or "").startswith("Task: " + brief))
    assert ("\nentry_point: " in task) == entry
    assert ("\nBegin." in task) != entry
    return raw_requests


def _assert_forced_route_annotation(requests):
    forced = json.loads(requests[1])
    assert forced["model"] == "editor"
    messages = forced["messages"]
    assert any(BRIEF in (row.get("content") or "") for row in messages)
    assert any("A previous step proposed these changes against revision" in
               (row.get("content") or "") for row in messages)
    annotation_rows = [row for row in messages if ANNOTATION in (row.get("content") or "")]
    assert len(annotation_rows) == 1
    assert annotation_rows[0]["content"].count(ANNOTATION) == 1


@pytest.mark.host_sandbox
@pytest.mark.parametrize(("entry", "resume", "brief", "switches",
                         "route_escalation", "forced_route_requalification"), [
    pytest.param(True, "checkpoint", BRIEF, (None, "off"), False, False,
                 id="operational-checkpoint"),
    pytest.param(False, "replay", DISCOVERY, (None, "off"), False, False,
                 id="discovery-replay"),
    pytest.param(True, "replay", AMBIGUOUS, ("on",), False, False,
                 id="ambiguous-replay"),
    pytest.param(True, None, BRIEF, (None, "off"), True, True,
                 id="forced-route-requalification"),
])
def test_r018_brief_verbatim_annotated_off_request_bytes_match_pinned_099(
    tmp_path, tmp_path_factory, entry, resume, brief, switches,
    route_escalation, forced_route_requalification,
):
    baseline = pinned_tree(BASE, tmp_path_factory)
    expected = _requests(
        baseline, tmp_path / "base", None, brief=brief, entry=entry, resume=resume,
        route_escalation=route_escalation,
        forced_route_requalification=forced_route_requalification,
    )
    if forced_route_requalification:
        assert json.loads(expected[0])["model"] == "reader"
        forced = json.loads(expected[1])
        assert forced["model"] == "editor"
        assert any("A previous step proposed these changes against revision" in
                   (row.get("content") or "") for row in forced["messages"])
    for switch in switches:
        actual = _requests(ROOT, tmp_path / (switch or "unset"), switch,
                           brief=brief, entry=entry, resume=resume,
                           route_escalation=route_escalation,
                           forced_route_requalification=forced_route_requalification)
        assert actual == expected


def test_r018_brief_verbatim_annotated_switch_defaults_off(monkeypatch):
    from stateframe import profiles, runner

    assert SWITCH not in profiles.load(profiles.DEFAULT)
    assert SWITCH not in runner.OPT_IN_RUNNER_DEFAULTS
    assert runner.OPT_IN_SWITCHES[SWITCH] == ("off", "on")
    assert SWITCH in runner.engine_switches(runner.ENGINE_DIR)
    monkeypatch.delenv(SWITCH, raising=False)
    assert SWITCH not in runner.resolve_opt_in_switches()
    monkeypatch.setenv(SWITCH, "off")
    assert runner.resolve_opt_in_switches()[SWITCH] == "off"


def test_r018_brief_verbatim_annotated_rejects_binding_scoped_to_another_version():
    from probes.react_repair import _render_example_slots

    brief = (
        "Example: `/v2/teams/teams`\n"
        "The second `teams` is the caller-supplied team name in v1 only."
    )
    assert _render_example_slots(brief) is None


def test_example_slots_ignores_html_comments():
    from probes.react_repair import _render_example_slots

    brief = (
        "Use the fixed path `/teams/teams` for this task.\n"
        "<!--\n"
        "Example: `/teams/teams`\n"
        "The second `teams` is the caller-supplied team name.\n"
        "-->"
    )
    assert _render_example_slots(brief) is None
    assert _render_example_slots("<!-- hidden note -->\n" + BRIEF) == ANNOTATION
    fenced_comment = "```text\n<!--\nExample: `/hidden/hidden`\n-->\n```\n" + BRIEF
    assert _render_example_slots(fenced_comment) == ANNOTATION


def test_example_slots_ignores_html_comment_after_escaped_backtick():
    from probes.react_repair import _render_example_slots

    brief = (
        "The escaped character is \\` <!-- comment starts with `\n"
        "Example: `/teams/teams`\n"
        "The second `teams` is the caller-supplied team name.\n"
        "-->"
    )
    assert _render_example_slots(brief) is None


def test_example_slots_preserves_html_comment_inside_backticked_example():
    from probes.react_repair import _render_example_slots

    brief = (
        "For example, `<!-- generated --><h1>Acme</h1>`\n"
        "The `Acme` is caller-supplied."
    )
    assert _render_example_slots(brief) == (
        "Annotated example (⟦…⟧ marks a runtime input): "
        "`<!-- generated --><h1>⟦Acme⟧</h1>`"
    )


def test_r018_brief_verbatim_annotated_accepts_binding_after_blank_markdown_line():
    from probes.react_repair import _render_example_slots

    brief = BRIEF.replace("\nThe second", "\n\nThe second")
    assert _render_example_slots(brief) == ANNOTATION


def test_r018_brief_verbatim_annotated_rejects_wrapped_binding_with_trailing_scope():
    from probes.react_repair import _render_example_slots

    brief = (
        "Example: `/v2/teams/teams`\n\n"
        "The second `teams` is caller-supplied team name\n"
        "in v1 only."
    )
    assert _render_example_slots(brief) is None


def test_r018_rejects_inline_binding_with_wrapped_version_qualifier():
    from probes.react_repair import _render_example_slots

    brief = (
        "Example: `/v2/teams/teams`, where the second `teams` is the caller-supplied team name\n"
        "in v1 only."
    )
    assert _render_example_slots(brief) is None


def test_r018_brief_verbatim_annotated_accepts_customer_slug_binding():
    from probes.react_repair import _render_example_slots

    brief = (
        "Example: `/customers/customers`\n"
        "The second `customers` is the caller-supplied customer slug."
    )
    assert _render_example_slots(brief) == (
        "Annotated example (⟦…⟧ marks a runtime input): `/customers/⟦customers⟧`"
    )


@pytest.mark.host_sandbox
def test_r018_brief_verbatim_annotated_routed_edit_requalification_matches_pinned_099(
    tmp_path, tmp_path_factory,
):
    baseline = pinned_tree(BASE, tmp_path_factory)
    expected = _requests(baseline, tmp_path / "base-routed", None, route_escalation=True)
    actual = _requests(ROOT, tmp_path / "off-routed", "off", route_escalation=True)
    assert json.loads(expected[0])["model"] == "reader"
    assert json.loads(expected[1])["model"] == "editor"
    assert any("A previous step proposed these changes against revision" in
               (row.get("content") or "") for row in json.loads(expected[1])["messages"])
    assert actual == expected
    for work in (tmp_path / "base-routed", tmp_path / "off-routed"):
        events = measurement_events(work / "run")
        assert sum(event.get("kind") == "route_proposal_preserved" for event in events) == 1


@pytest.mark.host_sandbox
def test_r018_brief_verbatim_annotated_on_forced_routed_requalification_is_annotated(tmp_path):
    requests = _requests(ROOT, tmp_path / "on-routed", "on", route_escalation=True,
                         forced_route_requalification=True)
    _assert_forced_route_annotation(requests)


@pytest.mark.host_sandbox
def test_r018_brief_verbatim_annotated_on_path_is_bounded_and_idempotent(tmp_path):
    from types import SimpleNamespace

    from kernel_cache_layout_view import semantic_messages
    from probes.react_repair import _example_slots_preparer, _render_example_slots

    assert _render_example_slots(BRIEF) == ANNOTATION
    multiple = (BRIEF.replace("/teams/teams", "/teams/teams/42")
                + "\nThe `42` comes from the caller.")
    assert _render_example_slots(multiple) == ANNOTATION.replace("⟦teams⟧", "⟦teams⟧/⟦42⟧")
    for ambiguous in (
        AMBIGUOUS,
        BRIEF.replace("second", "third"),
        BRIEF.replace("caller-supplied", "fixed"),
        BRIEF + "\nExample: `/teams/another`",
        BRIEF.replace("/teams/teams", "/teams/⟦teams⟧"),
        BRIEF.replace("/teams/teams", "/teams/teams/" + "a" * 256),
        "Example: `/x/x/x`, where `x/x` comes from the caller.",
        "```text\n" + BRIEF + "\n```",
    ):
        assert _render_example_slots(ambiguous) is None
    assert _render_example_slots(BRIEF + "\n" + ANNOTATION) is None

    task = {"role": "user", "content": "Task: " + BRIEF + "\n\nTarget orientation"}
    tail = {"role": "user", "content": "[WIRE VOLATILE TAIL]\n\nRequired evidence"}
    raw = [task, tail]
    snapshot = [task.copy(), tail.copy()]
    inputs = {"task_message": task, "source_messages": [task], "sources": {"app.py": "x" * 2000}}
    prepare = _example_slots_preparer(lambda rows, **_: (rows, {}), BRIEF, ANNOTATION)
    runtime = SimpleNamespace(frame_ceiling=512)
    assert prepare(raw, runtime=runtime, tools=[], framefit5_inputs=inputs) == (raw, {})
    runtime.frame_ceiling = 8000
    display, receipt = prepare(raw, runtime=runtime, tools=[], framefit5_inputs=inputs)
    assert display[-1]["content"].count(ANNOTATION) == 1
    assert prepare(display, runtime=runtime, tools=[], framefit5_inputs=inputs) == (display, receipt)
    assert raw == snapshot

    original = _requests(ROOT, tmp_path / "off", "off")
    annotated = _requests(ROOT, tmp_path / "on", "on", resume="replay")
    for source, display in zip(original, annotated[:2] + annotated[3:], strict=True):
        before, after = json.loads(source), json.loads(display)
        before["messages"] = semantic_messages(before["messages"])
        after["messages"] = semantic_messages(after["messages"])
        positions = [row for row in after["messages"] if ANNOTATION in (row.get("content") or "")]
        assert len(positions) == 1
        assert any(BRIEF in (row.get("content") or "") for row in after["messages"])
        rendered, = positions
        assert rendered["content"].count(ANNOTATION) == 1
        rendered["content"] = rendered["content"].replace("\n\n" + ANNOTATION, "")
        # Compare the rest of the presentation, including source and check evidence.
        assert after == before
    forced = _requests(ROOT, tmp_path / "on-forced-route", "on", route_escalation=True,
                       forced_route_requalification=True)
    _assert_forced_route_annotation(forced)

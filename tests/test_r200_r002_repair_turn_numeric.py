"""Repair-turn unit advice and its pinned provider-wire contract."""

import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from _timeouts import scaled
from _stateframe_home import isolated_stateframe_home
from engine_run import run_monitored, start_monitored
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, make_handler
from tools.pinned_tree import pinned_tree

ROOT = Path(__file__).resolve().parents[1]
SWITCH = "STATEFRAME_ARITHMETIC_UNITS"
PREFIX = "[Arithmetic units]"
PRODUCT = '''from typing import Annotated

def dollars(cents: Annotated[int, "cents"]) -> Annotated[float, "USD"]:
    return cents * 100
'''
CHECK = '''from pricing import dollars

def test_dollars():
    assert dollars(250) == 2.5
'''
DRIVER = """
import secrets
import sys
from types import SimpleNamespace

from stateframe import cli, runner

runner.secrets = SimpleNamespace(
    token_hex=lambda size: "0123456789abcdef" * 2 if size == 16 else secrets.token_hex(size))
raise SystemExit(cli.main(sys.argv[1:]))
"""


def _advisories(raw):
    found = []
    for message in json.loads(raw).get("messages", []):
        content = message.get("content")
        parts = [content] if isinstance(content, str) else (
            [part.get("text") for part in content
             if isinstance(part, dict) and isinstance(part.get("text"), str)]
            if isinstance(content, list) else [])
        found.extend(line for part in parts for line in part.splitlines()
                     if line.startswith(PREFIX))
    return found


def _run(checkout, work, switch=None, *, resume=False, profile_switch=None,
         tokenizer_unavailable_after_interrupt=False, tokenizer_recovers_before_resume=False,
         capacity_retry_on_repair=False, compare_resume=True):
    repo = work / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "pricing.py").write_text(PRODUCT)
    (repo / "tests/test_pricing.py").write_text(CHECK)
    run_dir = work / "run"
    pending = threading.Event()
    release = threading.Event()
    steps = [
        {"tool": "read", "arguments": {"path": "pricing.py"}},
        {"tool": "read", "arguments": {"path": "tests/test_pricing.py"}},
        {"tool": "run_tests", "arguments": {"target": "tests/test_pricing.py"}},
        {"tool": "str_replace", "arguments": {
            "path": "pricing.py", "old_str": "cents * 100", "new_str": "cents / 100"}},
        {"tool": "run_tests", "arguments": {"target": "tests/test_pricing.py"}},
        {"tool": "done", "arguments": {"summary": "Convert cents to dollars."}},
    ]

    class RepairScript(Script):
        def next_call(self, offered):
            if resume and self.position == 3 and not release.is_set():
                pending.set()
                release.wait(scaled(240))
                raise ScriptStepUnavailable("interrupted on the repair request")
            return super().next_call(offered)

    script = RepairScript({"steps": steps, "tokenize": {"mode": "ok"}})
    requests = []
    capacity_retry = {"rejected": False}

    class RecordingHandler(make_handler(script)):
        def do_POST(self):
            if (tokenizer_recovers_before_resume and not release.is_set()
                    and self.path.split("?", 1)[0].endswith("/tokenize")
                    and script.position == 3):
                script.tokenize["mode"] = "fail"
            if self.path.split("?", 1)[0].endswith("/chat/completions"):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                requests.append(raw)
                self.rfile = io.BytesIO(raw)
                if (capacity_retry_on_repair and script.position == 3
                        and not capacity_retry["rejected"]):
                    capacity_retry["rejected"] = True
                    self._send(400, {"error": {
                        "message": "request exceeds the available context size",
                        "type": "invalid_request_error", "code": "context_length_exceeded",
                    }})
                    return
            super().do_POST()

    env = {key: value for key, value in os.environ.items() if not key.startswith("STATEFRAME_")}
    env.update(PYTHONPATH=str(checkout / "src"), STATEFRAME_HOME=isolated_stateframe_home(),
               STATEFRAME_PROVIDER="llama.cpp", PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    if switch is not None:
        env[SWITCH] = switch
    server = ThreadingHTTPServer(("127.0.0.1", 0), RecordingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    command = [sys.executable, "-c", DRIVER]
    endpoint_args = ["--endpoint", endpoint, "--model", script.model, "--quiet", "--json"]
    arguments = [
        "run", "--repo", str(repo), "--task", "Fix the cents to dollars conversion in pricing.py.",
        "--target", "pricing.py", "--run-dir", str(run_dir),
        "--test-python", sys.executable, *endpoint_args,
    ]
    if resume:
        arguments += ["--keep-checkpoints"]
    if profile_switch is not None:
        profile = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text())
        profile["switches"][SWITCH] = profile_switch
        path = work / "profile.json"
        path.write_text(json.dumps(profile))
        arguments += ["--profile", str(path)]
    process = None
    try:
        if resume:
            launched_after = time.time()
            process = start_monitored(
                command + arguments, cwd=work, env=env, run_dir=run_dir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                expected_seconds=scaled(240), step="interrupt on the repair request",
            )
            deadline = time.monotonic() + scaled(240)
            while not pending.wait(0.25):
                assert process.poll() is None, process.communicate()
                assert time.monotonic() < deadline, "repair request did not reach the endpoint"
            (engine_pid,) = subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()
            engine_pid = int(engine_pid)
            identity = record_identity(engine_pid, launched_after=launched_after,
                                       command_fragment="_child.py")
            signal_run_verified(run_dir, engine_pid, identity, signal.SIGKILL,
                                deadline=deadline, what="repair-turn interruption")
            if tokenizer_unavailable_after_interrupt:
                script.tokenize["mode"] = "fail"
            elif tokenizer_recovers_before_resume:
                script.tokenize["mode"] = "ok"
            release.set()
            process.communicate(timeout=scaled(60))
            arguments = ["resume", str(run_dir), *endpoint_args]
        result = run_monitored(command + arguments, cwd=work, env=env, run_dir=run_dir,
                               expected_seconds=scaled(240), step="finish the scripted repair")
    finally:
        release.set()
        if process is not None and process.poll() is None:
            signal_run_owned(run_dir, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="repair-turn test cleanup",
                             group=True)
            process.communicate(timeout=scaled(10))
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.returncode == 0, result.stdout + result.stderr
    script.assert_consumed()
    assert len(requests) == (len(steps) + int(resume)
                             + int(capacity_retry_on_repair))
    if resume:
        if compare_resume and requests[3] != requests[4]:
            original, rebuilt = (json.loads(requests[index]) for index in (3, 4))
            def describe(body):
                contents = [message.get("content", "") for message in body.get("messages", [])]
                text = "\n".join(
                    part if isinstance(part, str) else str(part.get("text", ""))
                    for content in contents
                    for part in (content if isinstance(content, list) else [content])
                    if isinstance(part, (str, dict))
                )
                return {
                    "advisories": [line for line in text.splitlines()
                                   if line.startswith(PREFIX)],
                    "content_lengths": [len(str(content)) for content in contents],
                    "tool_names": [row.get("function", {}).get("name")
                                   for row in body.get("tools", [])],
                    "message_count": len(body.get("messages", [])),
                }
            assert requests[3] == requests[4], (
                "repair request bytes changed on resume: "
                + json.dumps({"original": describe(original), "rebuilt": describe(rebuilt)}))
        events = [json.loads(line.removeprefix("[sf] "))
                  for line in (run_dir / "engine.log").read_text().splitlines()
                  if line.startswith("[sf] ")]
        assert any(event["kind"] == "checkpoint_verified" and event["send"] == 4
                   for event in events)
    return requests, run_dir


@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resumed"])
def test_r002_repair_turn_numeric_off_request_bytes_match_pinned_099(tmp_path, tmp_path_factory, resume):
    base = pinned_tree("29765b39820fd39800fd7292f52fb5bcb81e6069", tmp_path_factory)
    expected, _ = _run(base, tmp_path / "base", resume=resume)
    for configuration in (None, "off"):
        actual, _ = _run(ROOT, tmp_path / str(configuration), configuration, resume=resume)
        assert actual == expected


def test_arithmetic_units_off_pinned_bytes_match_after_capacity_retry(tmp_path, tmp_path_factory):
    base = pinned_tree("29765b39820fd39800fd7292f52fb5bcb81e6069", tmp_path_factory)
    expected, _ = _run(base, tmp_path / "base", capacity_retry_on_repair=True)
    for configuration in (None, "off"):
        actual, _ = _run(
            ROOT, tmp_path / str(configuration), configuration,
            capacity_retry_on_repair=True,
        )
        assert actual == expected


def test_r002_repair_turn_numeric_switch_defaults_off(monkeypatch):
    from stateframe import profiles, runner

    monkeypatch.delenv(SWITCH, raising=False)
    assert SWITCH not in profiles.load(profiles.DEFAULT)
    assert SWITCH in runner.OPT_IN_SWITCHES
    assert SWITCH not in runner.resolve_opt_in_switches()
    monkeypatch.setenv(SWITCH, "off")
    assert runner.resolve_opt_in_switches()[SWITCH] == "off"
    monkeypatch.setenv(SWITCH, "false")
    assert runner.resolve_opt_in_switches()[SWITCH] == "off"
    monkeypatch.setenv(SWITCH, "on")
    assert runner.resolve_opt_in_switches()[SWITCH] == "on"


def test_r002_repair_turn_numeric_on_path_is_bounded_and_idempotent(tmp_path):
    from hashlib import sha256
    from tests._journal import measurement_events

    requests, run_dir = _run(ROOT, tmp_path, profile_switch="on")
    repaired = requests[3]
    messages = json.loads(repaired)["messages"]
    assert [wire.count(PREFIX.encode()) for wire in requests] == [0, 0, 0, 1, 0, 0]
    (line,) = [line for line in messages[-1]["content"].splitlines() if line.startswith(PREFIX)]
    assert len(line.encode()) <= 720
    assert "`cents * 100`" in line
    assert "cents=1/100 USD/stored unit" in line
    assert "100=scalar" in line
    assert "Implied scale and dimensions 1/100 USD/stored unit" in line
    assert "Required 1 USD/stored unit" in line
    assert sha256(PRODUCT.encode()).hexdigest() in line
    assert sha256(CHECK.encode()).hexdigest() in line
    assert any(PRODUCT in row["content"] for row in messages[:-1] if isinstance(row["content"], str))
    assert any(CHECK in row["content"] for row in messages[:-1] if isinstance(row["content"], str))
    events = measurement_events(run_dir)
    (failure,) = [event for event in events if event["kind"] == "check_receipt"
                  and event["status"] == "FAIL"]
    assert f'{failure["receipt_id"]}@{failure["revision"]}' in line
    body_hash = "sha256:" + sha256(repaired).hexdigest()
    (sealed,) = [event for event in events if event["kind"] == "qualified_request_sealed"
                 and event["body_sha256"] == body_hash]
    (submitted,) = [event for event in events if event["kind"] == "request_submitted"
                    and event["request_id"] == sealed["request_id"]]
    assert submitted["payload_hash"] == body_hash
    (delivery,) = [row for row in submitted["advice_delivery"] if row["component"] == "arithmetic_units"]
    assert delivery["present_in_submitted_context"] is True
    assert delivery["check_revision"] == failure["revision"]
    assert delivery["receipt_id"] == failure["receipt_id"]
    assert delivery["text"] == line


def test_arithmetic_units_resume_preserves_recorded_advisory_without_tokenizer(tmp_path):
    requests, _run_dir = _run(
        ROOT, tmp_path, resume=True, profile_switch="on",
        tokenizer_unavailable_after_interrupt=True, compare_resume=False,
    )
    assert len(_advisories(requests[3])) == 1
    assert requests[3] == requests[4]


def test_arithmetic_units_resume_does_not_add_unrecorded_advisory(tmp_path):
    requests, _run_dir = _run(
        ROOT, tmp_path, resume=True, profile_switch="on",
        tokenizer_recovers_before_resume=True,
    )
    assert not _advisories(requests[3])


def test_literal_count_multipliers_preserve_the_declared_storage_scale():
    import ast
    from fractions import Fraction

    from kernel_r200_r002_repair_turn_numeric import _propagate, _unit

    declared = {"monthly_cents": _unit(ast.parse(
        "Annotated[int, 'cents']", mode="eval").body)}
    for count in (3, 4, 100):
        actual, value, _operands = _propagate(
            ast.parse(f"monthly_cents * {count}", mode="eval").body,
            declared, {"monthly_cents": Fraction(250)},
        )
        assert actual == declared["monthly_cents"]
        assert value == 250 * count


@pytest.mark.parametrize("problem", ["stale-check", "stale-source", "ambiguous", "missing-span", "over-budget"])
def test_arithmetic_units_requires_current_unambiguous_visible_evidence(tmp_path, problem):
    from hashlib import sha256
    from types import SimpleNamespace

    from kernel_r200_r002_repair_turn_numeric import advice, admit
    from kernel_source_delivery import deliver_source

    revision = "sha256:" + "a" * 64
    receipt_id = "check-receipt:" + "b" * 24
    verdict = {"status": "FAIL", "tested_revision": revision, "current_revision": revision,
               "changed_since_check": False, "check_receipt_id": receipt_id}
    context = (">       assert dollars(250) == 2.5\nE       assert 25000 == 2.5\n"
               "tests/test_pricing.py:4: AssertionError")
    if problem == "ambiguous":
        context += "\ntests/test_pricing.py:5: AssertionError"
    receipt = {"outcome": "BEHAVIOR_FAIL", "reached_observation": True, "execution_complete": True,
               "receipt_id": receipt_id, "revision": revision, "capture_complete": True,
               "output_truncated": False,
               "selector": "tests/test_pricing.py", "assertion_context_omitted_bytes": 0,
               "assertion_context_preview": context}
    tool_text = f"assertion_context={context!r} full_output_handle={receipt_id}"
    pages = []
    for path, source in [("pricing.py", PRODUCT), ("tests/test_pricing.py", CHECK)]:
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(source)
        pages.append({"role": "user", "content": deliver_source(
            source.encode(), path, 0, args={"path": path}).text})
    if problem == "stale-source":
        (tmp_path / "pricing.py").write_text(PRODUCT.replace("* 100", "* 10"))
    elif problem == "stale-check":
        verdict["current_revision"] = "sha256:" + sha256(b"new candidate").hexdigest()
    elif problem == "missing-span":
        pages.pop()

    if problem != "over-budget":
        assert advice(tmp_path, pages, verdict, receipt, tool_text) is None
        return
    assert advice(tmp_path, pages, verdict, receipt, tool_text)
    meter = SimpleNamespace(measure=lambda _body: (100_000, "tokenize"))
    capture = SimpleNamespace(final_tool_surface={"tools": []}, _request_token_meter=meter,
                              _capacity_retry_original_body={"model": "scripted"})
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path), frame_ceiling=33_344,
                              _active_capture=capture)
    assert admit([*pages, {"content": ""}], runtime, verdict, receipt,
                 tool_text, None, False, lambda line: line) is None


def test_arithmetic_units_accepts_named_pytest_selector(tmp_path):
    from kernel_r200_r002_repair_turn_numeric import advice
    from kernel_source_delivery import deliver_source

    revision = "sha256:" + "a" * 64
    receipt_id = "check-receipt:" + "b" * 24
    verdict = {"status": "FAIL", "tested_revision": revision, "current_revision": revision,
               "changed_since_check": False, "check_receipt_id": receipt_id}
    context = (">       assert dollars(250) == 2.5\nE       assert 25000 == 2.5\n"
               "tests/test_pricing.py:4: AssertionError")
    receipt = {"outcome": "BEHAVIOR_FAIL", "reached_observation": True,
               "execution_complete": True, "receipt_id": receipt_id, "revision": revision,
               "capture_complete": True, "output_truncated": False, "selector": "test_dollars",
               "assertion_context_omitted_bytes": 0, "assertion_context_preview": context}
    tool_text = f"assertion_context={context!r} full_output_handle={receipt_id}"
    pages = []
    for path, source in [("pricing.py", PRODUCT), ("tests/test_pricing.py", CHECK)]:
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(source)
        pages.append({"role": "user", "content": deliver_source(
            source.encode(), path, 0, args={"path": path}).text})

    assert advice(tmp_path, pages, verdict, receipt, tool_text)


def test_storage_scale_through_multiplication_and_division():
    import ast
    from fractions import Fraction

    from kernel_r200_r002_repair_turn_numeric import Unit, _propagate, _unit

    def unit(label):
        return _unit(ast.parse(f"Annotated[float, {label!r}]", mode="eval").body)

    values = {"amount": Fraction(250), "rate": Fraction(50)}
    declared = {"amount": unit("cents"), "rate": unit("cents/s")}
    for expression, required in (
        ("amount / 100", unit("cents")),
        ("amount / rate", unit("s")),
        ("rate / amount", unit("ratio").combine(unit("s"), -1)),
        ("amount / 100 / 100", unit("cents")),
    ):
        actual, _value, _operands = _propagate(
            ast.parse(expression, mode="eval").body, declared, values)
        assert actual == required


@pytest.mark.parametrize("structured", [False, True])
def test_reassembly_replaces_only_the_advisory_in_the_final_tail(monkeypatch, structured):
    from kernel_stateframe_framefit5 import (
        GREENFACT_TEXT, _remove_arithmetic_units,
        _split_host_greenfact, _split_volatile_tail_content, _volatile_tail_content,
        fit_headcarry_to_request_budget,
    )

    notes = f"{PREFIX} The prior reasoning remains available."
    payload = [{"type": "text", "text": notes}] if structured else notes
    original = _volatile_tail_content(payload, (), 1)
    tail = {"role": "user", "content": original}
    for detail in ("previous", "recomputed", "final"):
        _remove_arithmetic_units(tail)
        evidence, green = _split_host_greenfact(_split_volatile_tail_content(tail["content"]))
        line = (f"{PREFIX} {detail}. Evidence from app.py and check "
                f"check-receipt:{'a' * 24}@sha256:{'b' * 64}.")
        tail["content"] = _volatile_tail_content(evidence, [line], green)
        assert str(tail["content"]).count("Evidence from app.py") == 1
        assert notes in str(tail["content"])
    monkeypatch.setenv(SWITCH, "on")
    fit_headcarry_to_request_budget(
        [tail], max_request_bytes=len(json.dumps(original).encode()),
        measure_request_bytes=lambda: len(json.dumps(tail["content"]).encode()),
    )
    assert tail["content"] == original
    assert GREENFACT_TEXT.strip() in json.dumps(tail["content"])

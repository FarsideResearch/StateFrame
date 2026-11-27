"""Requested-evidence checks and 0.9.9 wire parity."""

from __future__ import annotations

import ast
import hashlib
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
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from _journal import measurement_events
from _stateframe_home import isolated_stateframe_home
from _timeouts import scaled
from acceptance._support import assert_tree_imports, tree_python
from fixtures.process_signals import (
    record_identity,
    signal_run_owned,
    signal_run_verified,
)
from kernel_tool_call_loop_breaker import (
    UnboundedProgressWatch,
    requested_evidence_probe_enabled,
)
from tools.compare_no_mcp_requests import write_launcher
from tools.pinned_tree import pinned_tree

from stateframe import runner
from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, make_handler

_COMMIT = "29765b39820fd39800fd7292f52fb5bcb81e6069"
_CHAIN_TIP = "1886f91950f1bd0c7c9456359e04e39607ca094b"
_CURRENT = Path(__file__).resolve().parents[1]
_SWITCH = "STATEFRAME_REQUESTED_EVIDENCE_PROBE"
_MARKER = b"[Requested repository evidence]"
_CONFIG_PATH = "shop/return-mode.cfg"
_DECLARATION_LINE = 30
_TRUE = "DISCOUNT_RETURNS = 'remaining'"
_PLANNED_TRUE = "R = 7"
_PLANNED_ALTERNATIVE = "R = 0"
_PLANNED_CONFIG = "#\n" * (_DECLARATION_LINE - 1) + _PLANNED_TRUE + "\n" + "#\n" * 11
_BASELINE_LINE = "    return round(price * (100 + percent) / 100, 2)"
_PLAN_PATH = "planning/discount-adapter.md"
_PLAN_TASK = (
    f"Use {_PLAN_PATH} to fix the return expression in shop/pricing.py. Before editing, "
    "request a read of only the declaration line identified in the planning notes. "
    "The pricing test source has already been reviewed, and the tests run after the edit. "
    "Only edit shop/pricing.py."
)
_PLAN_TEXT = """# Pricing adapter planning notes

`apply_discount` returns the final amount charged to the customer. This shop supports
two provider contracts. In discount mode the percentage is the share of the list price
to deduct. In charge mode the percentage is the share of the list price to collect.
The integration owner's declaration on line 30 of `shop/return-mode.cfg` selects
charge mode for zero and discount mode for a positive value.

The migration is scoped to `shop/pricing.py`. Keep the existing validation and money
rounding, and leave the provider declaration and tests alone. Each shop deploys with
one fixed contract, so the pricing helper does not need to load configuration at runtime.
"""
_EXPECTED_CHARGES = {"positive": [60.0, 108.02, 0.0, 42.0], "zero": [20.0, 15.43, 42.0, 0.0]}


def _fixture(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(_CURRENT / "examples/discount_bug", repo)
    (repo / "shop/terms.py").write_text(_TRUE + "\n", encoding="utf-8")
    launcher = tmp_path / "launch.py"
    write_launcher(launcher)
    document = json.loads((_CURRENT / "examples/discount_bug.script.json").read_bytes())
    document["slots"] = [65536]
    return repo, launcher, document


def _planning_fixture(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(_CURRENT / "examples/discount_bug", repo)
    (repo / _CONFIG_PATH).write_text(_PLANNED_CONFIG, encoding="utf-8")
    planning = repo / _PLAN_PATH
    planning.parent.mkdir(parents=True, exist_ok=True)
    planning.write_text(_PLAN_TEXT, encoding="utf-8")

    pricing = repo / "shop/pricing.py"
    source = pricing.read_text(encoding="utf-8")
    assert source.count("    return round(price * percent / 100, 2)") == 1
    source = source.replace("    return round(price * percent / 100, 2)", _BASELINE_LINE, 1)
    source = source.replace(
        "Return the price after taking ``percent`` percent off, rounded to cents.",
        "Return the customer charge under the active pricing contract, rounded to cents.",
    ).replace("apply one order-wide discount", "apply the checkout pricing rule")
    pricing.write_text(source, encoding="utf-8")

    tests = repo / "tests/test_pricing.py"
    tests.write_text(
        "from pathlib import Path\n\n"
        "import pytest\n\n"
        "from shop.pricing import apply_discount\n\n\n"
        "def test_apply_discount_follows_contract_specific_expectation():\n"
        "    config = Path(__file__).parents[1] / 'shop/return-mode.cfg'\n"
        "    provider_mode = int(config.read_text().splitlines()[29].split('=')[1])\n"
        f"    expected = {_EXPECTED_CHARGES['positive']!r} if provider_mode > 0 "
        f"else {_EXPECTED_CHARGES['zero']!r}\n"
        "    cases = [(80.0, 25), (123.45, 12.5), (42.0, 100), (42.0, 0)]\n"
        "    if [apply_discount(*case) for case in cases] != expected:\n"
        "        pytest.fail('The customer charge disagrees with the provider contract')\n\n\n"
        "def test_rejects_out_of_range_percent():\n"
        "    with pytest.raises(ValueError):\n"
        "        apply_discount(10.0, 120)\n",
        encoding="utf-8",
    )

    return repo


@contextmanager
def _provider(script, *, transport_retry_at=None, intercept_chat=None):
    bodies = []
    script.request_paths = []

    class Capture(make_handler(script)):
        def do_GET(self):  # noqa: N802
            script.request_paths.append(f"GET {self.path}")
            super().do_GET()

        def do_POST(self):  # noqa: N802
            script.request_paths.append(f"POST {self.path}")
            if self.path == "/v1/chat/completions":
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                bodies.append(raw)
                if transport_retry_at == len(bodies):
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Retry-After", "0")
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"scripted temporary failure"}}')
                    return
                if intercept_chat is not None:
                    try:
                        reply = intercept_chat(raw)
                    except (AssertionError, ScriptStepUnavailable) as error:
                        self._send(409, {"error": {"message": str(error)}})
                        return
                    if reply is not None:
                        self._send(200, reply)
                        return
                stream, self.rfile = self.rfile, io.BytesIO(raw)
                try:
                    super().do_POST()
                finally:
                    self.rfile = stream
            else:
                super().do_POST()

    with ThreadingHTTPServer(("127.0.0.1", 0), Capture) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", bodies
        finally:
            server.shutdown()
            thread.join()


def _run_args(python, launcher, repo, run, endpoint, *,
              task="Fix the pricing code so the tests pass.", model="scripted"):
    args = [python, str(launcher), "run", "--repo", str(repo), "--task",
            task, "--model", model,
            "--endpoint", endpoint, "--run-dir", str(run), "--test-python", sys.executable,
            "--max-turns", "0", "--no-run-env", "--no-project-hooks", "--keep-checkpoints", "--quiet"]
    return args


def _environment(tmp_path, switch=None):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("STATEFRAME_") and key != "PYTHONPATH"}
    env.update(STATEFRAME_HOME=isolated_stateframe_home(), PYTHONDONTWRITEBYTECODE="1",
               STATEFRAME_CHECKPOINT_MIN_FREE_BYTES="0", PYTHONHASHSEED="0")
    if switch is not None:
        env[_SWITCH] = switch
    return env


def _run(python, launcher, repo, run, script, env, *, transport_retry_at=None):
    with _provider(script, transport_retry_at=transport_retry_at) as (endpoint, bodies):
        result = subprocess.run(_run_args(python, launcher, repo, run, endpoint), env=env,
                                capture_output=True, text=True, timeout=scaled(300))
    assert result.returncode == 0, result.stdout + result.stderr
    script.assert_consumed()
    synthetic_retries = int(transport_retry_at is not None and len(bodies) >= transport_retry_at)
    assert len(bodies) == len(script.requests) + synthetic_retries >= 4
    return bodies


def _only_action(payload):
    message = payload["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    assert len(calls) == 1, message
    function = calls[0]["function"]
    return function["name"], json.loads(function["arguments"])


def _assert_retry_and_checkpoint_requalification(run):
    events = []
    locations = [run, *sorted((run / "attempts").glob("*"))]
    for location in locations:
        if (location / "worker" / "measurement-runs").is_dir():
            events.extend(measurement_events(location))
    assert any(row["kind"] == "provider_transport_retry" for row in events)
    assert any(row["kind"] == "qualified_request_sealed" for row in events)
    engine_log = (run / "engine.log").read_text(encoding="utf-8")
    assert '"kind":"checkpoint_verified"' in engine_log


def _run_and_resume(python, launcher, repo, run, script, env, *, interrupt_position=2,
                    interrupt_when=None, interrupt_after_event=None, transport_retry_at=None,
                    interrupt_deadline=180, intercept_chat=None,
                    task="Fix the pricing code so the tests pass."):
    waiting, release = threading.Event(), threading.Event()
    original = script.next_call
    endpoint_errors = []

    def interrupt_if_requested(body):
        if (script.position == interrupt_position and not release.is_set()
                and (interrupt_when is None or interrupt_when(body))):
            waiting.set()
            if not release.wait(scaled(180)):
                raise TimeoutError("the test did not release the interrupted request")
            raise ScriptStepUnavailable("interrupted before returning a response")

    def intercept(raw):
        body = json.loads(raw)
        try:
            interrupt_if_requested(body)
        except ScriptStepUnavailable:
            script.requests.append(body)
            raise
        try:
            return intercept_chat(raw, body)
        except AssertionError as error:
            endpoint_errors.append(error)
            raise

    def interrupt_scripted_call(offered):
        interrupt_if_requested(script.requests[-1])
        return original(offered)

    if intercept_chat is None:
        script.next_call = interrupt_scripted_call
    with _provider(script, transport_retry_at=transport_retry_at,
                   intercept_chat=intercept if intercept_chat else None) as (endpoint, bodies):
        process = subprocess.Popen(_run_args(python, launcher, repo, run, endpoint,
                                             task=task, model=script.model), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + scaled(interrupt_deadline)
            while (not waiting.wait(0.1)
                   or (interrupt_after_event is not None and not any(
                       row["kind"] == interrupt_after_event for row in measurement_events(run)))):
                if process.poll() is not None:
                    raise AssertionError("run exited before the interruption: " + "\n".join(process.communicate()))
                if time.monotonic() >= deadline:
                    trigger_events = sum(
                        row.get("kind") == interrupt_after_event
                        for row in measurement_events(run)
                    ) if interrupt_after_event else 0
                    run_entries = sorted(path.name for path in run.iterdir()) if run.is_dir() else []
                    failure = (
                        "run did not reach the interruption: "
                        f"script_position={script.position}, scripted_requests={len(script.requests)}, "
                        f"provider_sends={len(bodies)}, "
                        f"probe_sends={sum(_MARKER in body for body in bodies)}, "
                        f"trigger_events={trigger_events}, slots_calls={script.slots_calls}, "
                        f"tokenize_calls={script.tokenize_calls}, run_entries={run_entries}, "
                        f"request_paths={script.request_paths[-20:]}"
                    )
                    signal_run_owned(run, process, signal.SIGKILL, deadline=time.monotonic(),
                                     what="requested evidence probe diagnostic cleanup",
                                     require_checkpoint_saved=False)
                    stdout, stderr = process.communicate(timeout=scaled(90))
                    raise AssertionError(failure + "\nstdout:\n" + stdout + "\nstderr:\n" + stderr)
            children = subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()
            assert len(children) == 1, children
            child = int(children[0])
            identity = record_identity(child, launched_after=0, command_fragment="_child.py")
            signal_run_verified(run, child, identity, signal.SIGKILL,
                                deadline=deadline, what="requested evidence probe parity")
            release.set()
            process.communicate(timeout=scaled(90))
            assert process.returncode != 0
            resumed = subprocess.run(
                [python, str(launcher), "resume", str(run), "--endpoint", endpoint, "--quiet"],
                env=env, capture_output=True, text=True,
                timeout=scaled(480 if intercept_chat is not None else 240),
            )
            if endpoint_errors:
                raise endpoint_errors[0]
            assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        finally:
            release.set()
            if process.poll() is None:
                signal_run_owned(run, process, signal.SIGKILL, deadline=time.monotonic(),
                                 what="requested evidence probe parity cleanup", require_checkpoint_saved=False)
                process.communicate(timeout=scaled(90))
    script.assert_consumed()
    synthetic_retries = int(transport_retry_at is not None and len(bodies) >= transport_retry_at)
    assert len(bodies) == len(script.requests) + synthetic_retries >= 5
    return bodies


@pytest.mark.host_sandbox
def test_r084_missing_repository_fact_off_request_bytes_match_pinned_099(tmp_path, tmp_path_factory):
    base = pinned_tree(_COMMIT, tmp_path_factory)
    chain_tip = pinned_tree(_CHAIN_TIP, tmp_path_factory)
    repo, launcher, document = _fixture(tmp_path)
    launcher.write_text(
        "import linecache, runpy, secrets, sys\n"
        "_sequence = 0\n"
        "def fixed_hex(size=32):\n"
        "    global _sequence\n"
        "    caller = sys._getframe(1)\n"
        "    if size == 16 and 'run_identity_seed' in linecache.getline("
        "caller.f_code.co_filename, caller.f_lineno):\n"
        "        return '0123456789abcdef0123456789abcdef'\n"
        "    _sequence += 1\n"
        "    return format(_sequence, 'x').zfill(size * 2)[-size * 2:]\n"
        "secrets.token_hex = fixed_hex\n"
        "runpy.run_module('stateframe', run_name='__main__')\n",
        encoding="utf-8",
    )
    stalled_document = {
        **document,
        "steps": [document["steps"][0],
                  {"tool": "grep", "arguments": {"pattern": "missing-r084-fact"}},
                  *document["steps"][1:]],
    }
    run = tmp_path / "run"
    released = tree_python(base / "src", tmp_path / "release-python")
    tip_python = tree_python(chain_tip / "src", tmp_path / "chain-tip-python")
    candidate = tree_python(_CURRENT / "src", tmp_path / "candidate-python")
    assert_tree_imports(released, base / "src", Path(base.anchor))
    assert_tree_imports(tip_python, chain_tip / "src", Path(chain_tip.anchor))
    assert_tree_imports(candidate, _CURRENT / "src", Path(_CURRENT.anchor))
    release_script = StallingScript(stalled_document)
    expected = _run_and_resume(
        released, launcher, repo, run, release_script, _environment(tmp_path),
        transport_retry_at=9)
    assert release_script.stall_replies == 8
    _assert_retry_and_checkpoint_requalification(run)
    assert len(expected) >= 10, "parity fixture must sustain a repeated no-progress stall"
    assert expected[8] == expected[9], "transport retry must resend identical provider bytes"
    run.rename(tmp_path / "released-run")
    tip_script = StallingScript(stalled_document)
    expected_tip = _run_and_resume(
        tip_python, launcher, repo, run, tip_script, _environment(tmp_path),
        transport_retry_at=9)
    assert tip_script.stall_replies == 8
    _assert_retry_and_checkpoint_requalification(run)
    assert expected_tip[8] == expected_tip[9]
    run.rename(tmp_path / "chain-tip-run")
    for switch in (None, "false"):
        candidate_script = StallingScript(stalled_document)
        actual = _run_and_resume(
            candidate, launcher, repo, run, candidate_script, _environment(tmp_path, switch),
            transport_retry_at=9,
        )
        assert candidate_script.stall_replies == 8
        _assert_retry_and_checkpoint_requalification(run)
        assert actual == expected
        assert actual == expected_tip
        assert all(_MARKER not in body for body in actual)
        run.rename(tmp_path / f"candidate-{switch}-run")


@pytest.mark.host_sandbox
def test_r084_off_request_bytes_match_pinned_099_after_capacity_refit(tmp_path, tmp_path_factory):
    from test_capacity_400_acceptance import CAPACITY_ERROR, CapacityEndpoint

    base = pinned_tree(_COMMIT, tmp_path_factory)
    chain_tip = pinned_tree(_CHAIN_TIP, tmp_path_factory)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 6\n", encoding="utf-8")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text(
        "from app import VALUE\n\ndef test_value():\n    assert VALUE == 7\n",
        encoding="utf-8",
    )
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        "import linecache, runpy, secrets, sys\n"
        "_sequence = 0\n"
        "def fixed_hex(size=32):\n"
        "    global _sequence\n"
        "    caller = sys._getframe(1)\n"
        "    if size == 16 and 'run_identity_seed' in linecache.getline("
        "caller.f_code.co_filename, caller.f_lineno):\n"
        "        return '0123456789abcdef0123456789abcdef'\n"
        "    _sequence += 1\n"
        "    return format(_sequence, 'x').zfill(size * 2)[-size * 2:]\n"
        "secrets.token_hex = fixed_hex\n"
        "runpy.run_module('stateframe', run_name='__main__')\n",
        encoding="utf-8",
    )
    released = tree_python(base / "src", tmp_path / "release-python")
    tip_python = tree_python(chain_tip / "src", tmp_path / "chain-tip-python")
    candidate = tree_python(_CURRENT / "src", tmp_path / "candidate-python")
    assert_tree_imports(released, base / "src", Path(base.anchor))
    assert_tree_imports(tip_python, chain_tip / "src", Path(chain_tip.anchor))
    assert_tree_imports(candidate, _CURRENT / "src", Path(_CURRENT.anchor))

    def run_with_refit(python, checkout, run, switch=None):
        home = tmp_path / f"home-{run.name}"
        home.mkdir()
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("STATEFRAME_", "REACT_")) and key != "PYTHONPATH"}
        env.update(HOME=str(home), STATEFRAME_HOME=isolated_stateframe_home(),
                   PYTHONPATH=str(checkout / "src"),
                   PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
        if switch is not None:
            env[_SWITCH] = switch
        with CapacityEndpoint((CAPACITY_ERROR,)) as endpoint:
            result = subprocess.run(
                [python, str(launcher), "run", "--repo", str(repo), "--task",
                 "Fix app.py so VALUE is 7.", "--model", "capacity-mock",
                 "--endpoint", endpoint.url, "--run-dir", str(run), "--max-turns", "1",
                 "--test-cmd", "python3 -m pytest -q tests/test_app.py",
                 "--runner-kind", "pytest", "--test-python", sys.executable,
                 "--target", "app.py", "--edit", "app.py", "--shell", "off",
                 "--no-project-hooks", "--no-instructions", "--timeout", "180",
                 "--quiet", "--json"],
                cwd=_CURRENT, env=env, capture_output=True, text=True, timeout=scaled(300),
            )
            bodies = tuple(endpoint.requests)
        assert result.returncode == 0, result.stdout + result.stderr
        events = measurement_events(run)
        assert any(row["kind"] == "capacity_context_limit_learned"
                   and row.get("provider_limit_tokens") == 32768 for row in events)
        assert any(row["kind"] == "capacity_retry" and row.get("retry_scheduled") is True
                   for row in events)
        assert len(bodies) == 2 and bodies[0] != bodies[1]
        return bodies

    expected = run_with_refit(released, base, tmp_path / "release-run")
    expected_tip = run_with_refit(tip_python, chain_tip, tmp_path / "chain-tip-run")
    for switch in (None, "false"):
        actual = run_with_refit(
            candidate, _CURRENT, tmp_path / f"candidate-{switch}-run", switch,
        )
        assert actual == expected
        assert actual == expected_tip


def test_r084_probe_admission_respects_learned_capacity_before_adding_advisory(
        monkeypatch, tmp_path):
    import importlib
    from types import SimpleNamespace

    from kernel_capacity_retry import learn_context_limit, learned_request_limits
    from kernel_measurement import AuditJournal
    from kernel_wire_stability import VOLATILE_TAIL_MARKER
    from kernel_working_set_prepare import Measurement
    from kernel_working_set_snapshot import PreparationSnapshot

    limits = learned_request_limits(8192, 1792)
    assert limits.max_prompt_tokens == 5376
    assert 5374 <= limits.max_prompt_tokens < 5441
    endpoint, key = "http://provider.invalid/v1", "fixture-key"
    journal = AuditJournal(tmp_path, "capacity-refit-probe", {})
    learn_context_limit(journal, endpoint, "scripted", key, 8192)
    capture = SimpleNamespace(
        request_id="capacity-refit-probe",
        journal=journal,
    )
    body = {
        "model": "scripted",
        "max_tokens": limits.max_output_tokens,
        "messages": [{"role": "user", "content": (
            VOLATILE_TAIL_MARKER + "\nUse the supplied evidence to choose the next edit."
        )}],
        "tools": [{"type": "function", "function": {"name": "read"}}],
    }

    import kernel_launch_identity_cost
    import kernel_provider_dialect
    import kernel_request_budget
    import kernel_working_set_render
    import kernel_working_set_shadow
    import kernel_working_set_live

    monkeypatch.setattr(kernel_launch_identity_cost, "reasoning_profile",
                        lambda **_kwargs: SimpleNamespace(mode="on"))
    monkeypatch.setattr(kernel_provider_dialect, "capabilities_for",
                        lambda *_args, **_kwargs: SimpleNamespace(dialect="llama.cpp"))
    monkeypatch.setattr(
        kernel_working_set_shadow, "freeze_shadow",
        lambda proposal, _runtime, **_kwargs: PreparationSnapshot(
            fixed_body=json.dumps(proposal).encode(), sources=(),
            request_id=capture.request_id, due_ordinal=1, byte_cap=100_000,
            prompt_limit=33_344,
        ),
    )
    monkeypatch.setattr(kernel_working_set_render, "domain", lambda *_args: ())
    monkeypatch.setattr(
        kernel_working_set_render, "render_assignment",
        lambda snapshot, _assignment: SimpleNamespace(wire=snapshot.fixed_body),
    )
    monkeypatch.setattr(
        kernel_request_budget, "_request_token_meter",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        kernel_working_set_live, "live_measure",
        lambda _meter: lambda wire: Measurement(
            5441 if _MARKER in wire else 5374, 0, "controlled-capacity-fixture"),
    )
    monkeypatch.syspath_prepend(str(_CURRENT / "src/stateframe/engine/probes"))

    candidate = importlib.import_module("react_repair")._requested_evidence_body(
        body, capture, object(), endpoint, key)

    assert candidate is body, (
        "probe admission must omit an advisory that exceeds the learned prompt cap"
    )


def test_r084_missing_repository_fact_switch_defaults_off(monkeypatch):
    monkeypatch.delenv(_SWITCH, raising=False)
    assert not requested_evidence_probe_enabled()
    assert _SWITCH not in runner.resolve_opt_in_switches()
    assert _SWITCH in runner.OPT_IN_SWITCHES
    assert _SWITCH in runner.engine_switches()
    monkeypatch.setenv(_SWITCH, "false")
    assert not requested_evidence_probe_enabled()
    assert runner.resolve_opt_in_switches()[_SWITCH] == "off"
    monkeypatch.setenv(_SWITCH, "on")
    assert requested_evidence_probe_enabled()
    assert runner.resolve_opt_in_switches()[_SWITCH] == "on"


def _request_text(body):
    messages = body.get("messages")
    assert isinstance(messages, list)
    return "\n".join(str(message.get("content") or "") for message in messages
                     if isinstance(message, dict))


def _delivered_fact_values(body):
    messages = body.get("messages")
    assert isinstance(messages, list)
    values = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or "")
        if (f"[Current source: {_CONFIG_PATH}]" in content
                or f"[read page] path={_CONFIG_PATH}" in content):
            values.update(re.findall(r"(?m)^\s*R\s*=\s*(\d+)\s*$", content))
    return values


def _offline_completion(body, response_number, tool_name, arguments):
    """Return a deterministic tool-call reply without invoking a model server."""
    return {
        "id": f"chatcmpl-offline-{response_number}",
        "object": "chat.completion",
        "created": 0,
        "model": body["model"],
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call-offline-{response_number}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _offline_decision(body, response_number):
    """Score only the repository fact in the provider-bound request."""
    facts = _delivered_fact_values(body)
    visible = _request_text(body)
    if _MARKER.decode() in visible:
        assert not facts, "the probe must ask for evidence before seeing the answer"
        return _offline_completion(
            body, response_number, "read",
            {"path": _CONFIG_PATH, "start_line": _DECLARATION_LINE, "end_line": _DECLARATION_LINE},
        )
    assert len(facts) == 1, f"expected one bound repository fact, got {facts!r}"
    fact, = facts
    assert fact in {"7", "0"}
    replacement = ("    return round(price * (100 - percent) / 100, 2)"
                   if fact == "7" else "    return round(price * percent / 100, 2)")
    return _offline_completion(
        body, response_number, "str_replace",
        {"path": "shop/pricing.py", "old_str": _BASELINE_LINE, "new_str": replacement},
    )


class StallingScript(Script):
    """Repeat one identical no-hit search before continuing the normal task."""

    def __init__(self, document, *, stall_replies=8):
        super().__init__(document)
        self.stall_replies_left = stall_replies
        self.stall_replies = 0
        self.steps[1]["repeat"] = True

    def next_call(self, offered):
        if self.position == 1 and self.stall_replies_left:
            self.stall_replies_left -= 1
            self.stall_replies += 1
            if not self.stall_replies_left:
                self.steps[1]["repeat"] = False
        return super().next_call(offered)


class OfflinePlanningEndpoint(StallingScript):
    """Keep ordinary actions scripted and score probe decisions deterministically."""

    stall_position = 1

    def __init__(self, document):
        super().__init__(document)
        self.elicitation = None
        self.decision = None
        self.decision_wire = None
        self.offline_response_count = 0

    def intercept(self, raw, body):
        is_probe = _MARKER in raw
        facts = _delivered_fact_values(body)
        if not is_probe:
            if self.elicitation is None:
                assert not facts, "the repository fact must remain hidden until its read"
                return None
            if self.decision is not None:
                return None
            assert facts == {"7"}, "the resumed decision must receive the truthful read"
        assert self.position == self.stall_position
        assert _PLAN_TEXT.strip() in _request_text(body)
        self.requests.append(body)
        completion = _offline_decision(body, self.offline_response_count)
        self.offline_response_count += 1
        if is_probe:
            assert self.elicitation is None, "the probe must be delivered only once"
            self.elicitation = completion
        else:
            self.decision_wire = raw
            self.decision = completion
            self.position += 1
        return completion


def _candidate_source(source, name, action):
    assert action["path"] == "shop/pricing.py"
    assert name in {"edit", "str_replace"}
    if name == "edit":
        candidate = action["function_source"].strip()
    else:
        old_text, new_text = action["old_str"], action["new_str"]
        assert old_text and source.count(old_text) == 1
        candidate = source.replace(old_text, new_text, 1)
    original_tree, candidate_tree = ast.parse(source), ast.parse(candidate)
    original_function, candidate_function = original_tree.body[1], candidate_tree.body[1]
    assert isinstance(original_function, ast.FunctionDef)
    assert isinstance(candidate_function, ast.FunctionDef)
    original_return, candidate_return = original_function.body[-1], candidate_function.body[-1]
    assert isinstance(original_return, ast.Return) and isinstance(candidate_return, ast.Return)
    expression = candidate_return.value
    assert ast.dump(expression) != ast.dump(original_return.value), "the return value must change"
    assert isinstance(expression, ast.Call)
    assert isinstance(expression.func, ast.Name) and expression.func.id == "round"
    assert len(expression.args) == 2 and not expression.keywords
    assert isinstance(expression.args[1], ast.Constant) and expression.args[1].value == 2
    # Only execute the model's candidate after checking it is a pure arithmetic edit.
    arithmetic = (ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub)
    for node in ast.walk(expression.args[0]):
        if isinstance(node, ast.Name):
            assert node.id in {"price", "percent"} and isinstance(node.ctx, ast.Load)
        elif isinstance(node, ast.Constant):
            assert type(node.value) in (int, float) and abs(node.value) <= 1000
        else:
            assert isinstance(node, (ast.Load, *arithmetic)), ast.dump(node)
    candidate_return.value = original_return.value
    assert ast.dump(candidate_tree) == ast.dump(original_tree)
    return candidate


def _assert_candidate_tests_pass(repo, source, fact, tmp_path):
    candidate_repo = tmp_path / f"candidate-{fact}"
    shutil.copytree(repo, candidate_repo)
    pricing = candidate_repo / "shop/pricing.py"
    pricing.write_text(source, encoding="utf-8")
    (candidate_repo / _CONFIG_PATH).write_text(
        _PLANNED_CONFIG.replace(
            _PLANNED_TRUE, _PLANNED_TRUE if fact == "positive" else _PLANNED_ALTERNATIVE, 1,
        ), encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_pricing.py"],
        cwd=candidate_repo, env=environment, capture_output=True, text=True,
        timeout=scaled(60),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_r084_requested_evidence_helpers_reject_unbound_or_clipped_reads(monkeypatch, tmp_path):
    import importlib
    from types import SimpleNamespace

    watch = UnboundedProgressWatch(threshold=4)
    watch.enable_evidence_probe()
    binding = {"artifact": "revision"}
    for boundary in range(3):
        assert not watch.observe(binding, [], boundary=boundary)
    assert watch.claim_evidence_probe("revision", boundary=2)
    snapshot = json.loads(json.dumps(watch.snapshot()))
    resumed = UnboundedProgressWatch.from_snapshot(snapshot)
    assert not resumed.claim_evidence_probe("revision", boundary=2)
    stop_watch = UnboundedProgressWatch.from_snapshot(snapshot)
    assert not stop_watch.observe(binding, [], boundary=3)
    assert not stop_watch.claim_evidence_probe("revision", boundary=3)
    assert stop_watch.observe(binding, [], boundary=4)  # The normal stop still fires.

    repo = _planning_fixture(tmp_path)
    original_pricing = (repo / "shop/pricing.py").read_text(encoding="utf-8")
    monkeypatch.syspath_prepend(str(_CURRENT / "src/stateframe/engine/probes"))
    repair = importlib.import_module("react_repair")
    from kernel_stateframe_framefit5 import VOLATILE_TAIL_MARKER
    from kernel_working_set_prepare import Measurement
    from kernel_working_set_snapshot import PreparationSnapshot

    class Journal:
        manifest = {}

        def __init__(self):
            self.events = []

        def emit(self, kind, **fields):
            event = {"kind": kind, **fields}
            self.events.append(event)
            return event

    journal = Journal()
    probe_watch = UnboundedProgressWatch.from_snapshot(snapshot)
    assert not probe_watch.claim_evidence_probe("revision", boundary=2)
    journal.events.append({
        "kind": "requested_evidence_probe_trigger",
        "request_id": "requested-evidence-probe",
        "revision": probe_watch._requested_evidence_revision,
    })
    capture = SimpleNamespace(
        request_id="requested-evidence-probe",
        journal=journal,
        _requested_evidence_revision=probe_watch._requested_evidence_revision,
        _capacity_retry_limits=SimpleNamespace(max_request_bytes=100_000, max_prompt_tokens=33_344),
    )
    body = {
        "model": "scripted",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content":
                      VOLATILE_TAIL_MARKER + "\nUse a bounded repository read if needed."}],
        "tools": [{"type": "function", "function": {"name": "read"}}],
    }

    import kernel_launch_identity_cost
    import kernel_provider_dialect
    import kernel_request_budget
    import kernel_working_set_live
    import kernel_working_set_render
    import kernel_working_set_shadow

    monkeypatch.setattr(kernel_launch_identity_cost, "reasoning_profile",
                        lambda **_kwargs: SimpleNamespace(mode="on"))
    monkeypatch.setattr(kernel_provider_dialect, "capabilities_for",
                        lambda *_args, **_kwargs: SimpleNamespace(dialect="llama.cpp"))
    monkeypatch.setattr(
        kernel_working_set_shadow, "freeze_shadow",
        lambda proposal, _runtime, **_kwargs: PreparationSnapshot(
            fixed_body=json.dumps(proposal).encode(), sources=(),
            request_id=capture.request_id, due_ordinal=1,
            byte_cap=100_000, prompt_limit=33_344,
        ),
    )
    monkeypatch.setattr(kernel_working_set_render, "domain", lambda *_args: ())
    monkeypatch.setattr(
        kernel_working_set_render, "render_assignment",
        lambda snapshot, _assignment: SimpleNamespace(wire=snapshot.fixed_body),
    )
    monkeypatch.setattr(kernel_request_budget, "_request_token_meter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        kernel_working_set_live, "live_measure",
        lambda _meter: lambda _wire: Measurement(1, 0, "offline-delivery-fixture"),
    )
    probe_body = repair._requested_evidence_body(
        body, capture, object(), "http://provider.invalid/v1", "")
    assert probe_body is not body and _request_text(probe_body).count(_MARKER.decode()) == 1

    provider_sends = []

    def local_provider_send(request_body, response_number, request_id=None):
        raw = json.dumps(request_body, separators=(",", ":")).encode()
        provider_sends.append(raw)
        send_id = f"send-{response_number}"
        journal.emit("request_submitted", event_id=send_id,
                     request_id=request_id or capture.request_id)
        return raw, send_id, _offline_decision(json.loads(raw), response_number)

    probe_wire, probe_send, elicitation = local_provider_send(probe_body, 1)
    assert _MARKER in probe_wire and probe_wire == provider_sends[0]
    read_name, read_args = _only_action(elicitation)
    assert read_name == "read" and read_args["path"] == _CONFIG_PATH
    first, last = read_args["start_line"], read_args["end_line"]
    assert type(first) is int and type(last) is int
    assert 1 <= first <= _DECLARATION_LINE <= last <= len(_PLANNED_CONFIG.splitlines())
    assert last - first + 1 <= 12

    capture.qualified_request = SimpleNamespace(body=probe_body)
    capture.selected_response_event_id = elicitation["id"]
    capture.transport_attempt_id = probe_send
    capture._requested_evidence_rendered = True
    prepared = SimpleNamespace(capture=capture)
    requested_read = repair._requested_evidence_reply(
        prepared, read_name, read_args, single_action=True)
    assert requested_read == {
        "request_id": capture.request_id,
        "response_id": elicitation["id"],
        "delivered_send": probe_send,
        "revision": capture._requested_evidence_revision,
        "path": _CONFIG_PATH,
        "start_line": first,
        "end_line": last,
    }

    factual_line = _PLANNED_CONFIG.splitlines()[_DECLARATION_LINE - 1]
    probe_capture = SimpleNamespace(
        origin_response_id=elicitation["id"],
        capture_id="read-capture-1",
        identity=SimpleNamespace(
            lossy_span=None,
            byte_start=0,
            byte_end=len(factual_line.encode()),
            revision="sha256:" + hashlib.sha256(_PLANNED_CONFIG.encode()).hexdigest(),
            payload_sha256=hashlib.sha256(factual_line.encode()).hexdigest(),
            path=_CONFIG_PATH,
            first_line=_DECLARATION_LINE,
            last_line=_DECLARATION_LINE,
        ),
    )
    wrong_response_capture = SimpleNamespace(
        origin_response_id="chatcmpl-unrelated",
        capture_id=probe_capture.capture_id,
        identity=probe_capture.identity,
    )
    assert not repair._requested_evidence_record_read(
        probe_watch, journal, requested_read, "OK", {"clip_reason": None}, wrong_response_capture)
    assert not repair._requested_evidence_record_read(
        probe_watch, journal, requested_read, "OK", {"clip_reason": "line_limit"}, probe_capture)
    assert not any(row["kind"] == "requested_evidence_probe_read" for row in journal.events)
    assert getattr(probe_watch, "requested_evidence_read", None) is None
    assert repair._requested_evidence_record_read(
        probe_watch, journal, requested_read, "OK", {"clip_reason": None}, probe_capture)
    fact_event, = [row for row in journal.events if row["kind"] == "requested_evidence_probe_read"]
    read_evidence = _PLANNED_CONFIG
    assert fact_event["source_revision"] == "sha256:" + hashlib.sha256(
        read_evidence.encode()).hexdigest()

    decision_body = {
        "model": "scripted",
        "messages": [{"role": "user", "content": f"[Current source: {_CONFIG_PATH}]\n"
                      + read_evidence}],
        "tools": [{"type": "function", "function": {"name": "str_replace"}}],
    }
    truth_wire, truth_send, truth_decision = local_provider_send(
        decision_body, 2, "truth-decision-request")
    counterfactual_wire = truth_wire.replace(_PLANNED_TRUE.encode(), _PLANNED_ALTERNATIVE.encode())
    assert _delivered_fact_values(json.loads(counterfactual_wire)) == {"0"}
    counterfactual_wire, counterfactual_send, counterfactual = local_provider_send(
        json.loads(counterfactual_wire), 3, "counterfactual-decision-request")
    assert len({elicitation["id"], truth_decision["id"], counterfactual["id"]}) == 3
    assert len({probe_send, truth_send, counterfactual_send}) == 3
    assert len(provider_sends) == 3
    submitted_requests = {
        row["request_id"] for row in journal.events if row["kind"] == "request_submitted"
    }
    assert submitted_requests == {
        capture.request_id, "truth-decision-request", "counterfactual-decision-request",
    }
    assert _delivered_fact_values(json.loads(truth_wire)) == {"7"}
    assert _PLANNED_ALTERNATIVE.encode() not in truth_wire
    assert _PLANNED_TRUE.encode() not in counterfactual_wire
    truth_name, truth_action = _only_action(truth_decision)
    counterfactual_name, counterfactual_action = _only_action(counterfactual)
    truthful_source = _candidate_source(original_pricing, truth_name, truth_action)
    alternative_source = _candidate_source(original_pricing, counterfactual_name, counterfactual_action)
    assert truthful_source != alternative_source

    certificate = SimpleNamespace(
        witnesses=[SimpleNamespace(slot_id="source-slot", mode="FULL")],
        snapshot=SimpleNamespace(slots=[SimpleNamespace(
            key="source-slot", capture_ids=(probe_capture.capture_id,))]),
    )
    decision_capture = SimpleNamespace(
        request_id="decision-request",
        selected_response_event_id=truth_decision["id"],
        transport_attempt_id=truth_send,
        _working_set_certificate=certificate,
        journal=journal,
    )
    assert probe_watch.requested_evidence_read == (capture.request_id, probe_capture.capture_id)
    assert certificate.witnesses[0].slot_id == certificate.snapshot.slots[0].key
    assert probe_capture.capture_id in certificate.snapshot.slots[0].capture_ids
    repair._requested_evidence_followup(probe_watch, decision_capture, truth_name)
    assert probe_watch.requested_evidence_read is None

    trigger, = [row for row in journal.events if row["kind"] == "requested_evidence_probe_trigger"]
    reply, = [row for row in journal.events if row["kind"] == "requested_evidence_probe_response"]
    fact, = [row for row in journal.events if row["kind"] == "requested_evidence_probe_read"]
    used, = [row for row in journal.events if row["kind"] == "requested_evidence_probe_used"]
    assert trigger["request_id"] == reply["request_id"] == fact["request_id"]
    assert reply["action"] == "read" and reply["requested_read"] == {
        "path": _CONFIG_PATH, "start_line": first, "end_line": last,
    }
    submitted = {row["event_id"] for row in journal.events if row["kind"] == "request_submitted"}
    assert reply["delivered_send"] in submitted
    assert fact["source_revision"] == probe_capture.identity.revision
    assert {key: fact[key] for key in ("path", "start_line", "end_line")} == {
        "path": _CONFIG_PATH, "start_line": first, "end_line": last,
    }
    assert used["probe_request_id"] == trigger["request_id"]
    assert used["capture_id"] == fact["capture_id"] == probe_capture.capture_id
    assert used["action"] == truth_name
    assert not probe_watch.claim_evidence_probe("revision", boundary=2)

    _assert_candidate_tests_pass(repo, truthful_source, "positive", tmp_path)
    _assert_candidate_tests_pass(repo, alternative_source, "zero", tmp_path)


@pytest.mark.host_sandbox
def test_r084_missing_repository_fact_on_path_is_bounded_and_idempotent(tmp_path):
    repo = _planning_fixture(tmp_path)
    launcher = tmp_path / "launch.py"
    write_launcher(launcher)
    original_pricing = (repo / "shop/pricing.py").read_text(encoding="utf-8")
    document = {
        "model": "scripted",
        "steps": [
            {"tool": "read", "arguments": {"path": _PLAN_PATH}},
            {"tool": "grep", "arguments": {"pattern": "missing-r084-fact"}, "repeat": True},
            {"tool": "run_tests", "arguments": {"target": "tests/test_pricing.py"}},
            {"tool": "done", "arguments": {"summary": "Verified the pricing adapter."}},
        ],
        "tokenize": {"mode": "ok"},
    }
    script = OfflinePlanningEndpoint(document)
    python = tree_python(_CURRENT / "src", tmp_path / "candidate-python")
    run = tmp_path / "run"
    bodies = _run_and_resume(
        python, launcher, repo, run, script, _environment(tmp_path, "on"),
        interrupt_position=script.stall_position,
        interrupt_when=lambda body: _MARKER.decode() in json.dumps(body, ensure_ascii=False),
        interrupt_after_event="requested_evidence_probe_trigger",
        interrupt_deadline=360,
        intercept_chat=script.intercept,
        task=_PLAN_TASK,
    )

    probe_sends = [body for body in bodies if _MARKER in body]
    assert len(probe_sends) == sum(body.count(_MARKER) for body in bodies) == 2
    assert all(_PLANNED_TRUE.encode() not in body for body in probe_sends)
    assert script.stall_replies > 0
    read_name, read_args = _only_action(script.elicitation)
    assert read_name == "read" and read_args["path"] == _CONFIG_PATH
    first, last = read_args["start_line"], read_args["end_line"]
    assert type(first) is int and type(last) is int
    assert 1 <= first <= _DECLARATION_LINE <= last <= len(_PLANNED_CONFIG.splitlines())
    assert last - first + 1 <= 12

    truth_wire = script.decision_wire
    assert isinstance(truth_wire, bytes)
    assert _delivered_fact_values(json.loads(truth_wire)) == {"7"}
    assert _PLANNED_TRUE.encode() in truth_wire
    assert _PLANNED_ALTERNATIVE.encode() not in truth_wire
    truth_name, truth_action = _only_action(script.decision)
    truthful_source = _candidate_source(original_pricing, truth_name, truth_action)

    counterfactual_wire = truth_wire.replace(
        _PLANNED_TRUE.encode(), _PLANNED_ALTERNATIVE.encode())
    assert _delivered_fact_values(json.loads(counterfactual_wire)) == {"0"}
    counterfactual = _offline_decision(
        json.loads(counterfactual_wire), script.offline_response_count)
    counterfactual_name, counterfactual_action = _only_action(counterfactual)
    alternative_source = _candidate_source(
        original_pricing, counterfactual_name, counterfactual_action)
    assert truthful_source != alternative_source
    assert len({script.elicitation["id"], script.decision["id"], counterfactual["id"]}) == 3

    events = measurement_events(run)
    trigger, = [row for row in events if row["kind"] == "requested_evidence_probe_trigger"]
    reply, = [row for row in events if row["kind"] == "requested_evidence_probe_response"]
    fact, = [row for row in events if row["kind"] == "requested_evidence_probe_read"]
    used, = [row for row in events if row["kind"] == "requested_evidence_probe_used"]
    assert trigger["request_id"] == reply["request_id"] == fact["request_id"]
    assert reply["action"] == "read" and reply["requested_read"] == {
        "path": _CONFIG_PATH, "start_line": first, "end_line": last,
    }
    delivered, = [row for row in events if row["kind"] == "request_submitted"
                  and row["event_id"] == reply["delivered_send"]]
    assert delivered["request_id"] == trigger["request_id"]
    assert fact["source_revision"] == "sha256:" + hashlib.sha256(
        _PLANNED_CONFIG.encode()).hexdigest()
    assert {key: fact[key] for key in ("path", "start_line", "end_line")} == {
        "path": _CONFIG_PATH, "start_line": first, "end_line": last,
    }
    assert used["probe_request_id"] == trigger["request_id"]
    assert used["capture_id"] == fact["capture_id"]
    assert used["action"] == truth_name
    assert sum(row["kind"] == "requested_evidence_probe_trigger" for row in events) == 1
    assert sum(row["kind"] == "requested_evidence_probe_response" for row in events) == 1
    assert sum(row["kind"] == "requested_evidence_probe_read" for row in events) == 1
    assert sum(row["kind"] == "requested_evidence_probe_used" for row in events) == 1

    summary = json.loads((run / "summary.json").read_bytes())
    assert summary["engine_result"] == "SUBMITTED"
    assert summary["candidate_captured"]
    assert summary["candidate_files_written"] == ["shop/pricing.py"]
    assert summary["check_outcome"] == "pass" and summary["test_execution_evidence"]
    assert (run / "workspace/shop/pricing.py").read_text(encoding="utf-8") == truthful_source
    assert any(
        row.get("kind") == "action_outcome_closed"
        and isinstance(row.get("record"), dict)
        and row["record"].get("tool") == truth_name
        and row["record"].get("result") == "APPLIED"
        and any(change.get("path") == "shop/pricing.py"
                for change in row["record"].get("changes", []))
        for row in events
    )
    _assert_candidate_tests_pass(repo, truthful_source, "positive", tmp_path)
    _assert_candidate_tests_pass(repo, alternative_source, "zero", tmp_path)
    assert (repo / "shop/pricing.py").read_text(encoding="utf-8") == original_pricing
    assert (repo / "shop/pricing.py").read_text(encoding="utf-8") == original_pricing

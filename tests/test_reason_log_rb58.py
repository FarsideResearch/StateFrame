"""RB5/RB8 reason-log sites: each reports one stderr line per cause per run, and the run end summarises once.

Two helper singletons exist in one process on purpose: bare `kernel_reason_log` (engine/child style imports)
and `stateframe.engine.kernel_reason_log` (runner and _child package imports). Each fixture resets and
configures both, and each test reads the singleton the code under test actually writes to.
"""
import io
import json
import sys
import types

import pytest

from stateframe import _child, chat, runner
from stateframe.engine import kernel_reason_log as pkg_rl
import kernel_plain_loop as plain
import kernel_reason_log as rl
import kernel_stop_reason_taxonomy as taxonomy


@pytest.fixture(autouse=True)
def stream(monkeypatch):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    for module in (rl, pkg_rl):
        module.reset_for_tests()
    out = io.StringIO()
    for module in (rl, pkg_rl):
        module.configure(level="WARN", stream=out)
    yield out
    for module in (rl, pkg_rl):
        module.reset_for_tests()


class RecordingJournal:
    def __init__(self):
        self.events = []

    def write(self, kind, **fields):
        self.events.append((kind, fields))
        return len(self.events)


def _provider(body, timeout):
    # One plain assistant turn with no tool calls: the loop appends CONTINUE and then hits request_cap.
    return {"choices": [{"message": {"content": "thinking"}, "finish_reason": "stop"}]}


def test_plain_loop_end_writes_one_problems_summary(stream):
    rl.emit_reason("fallback", "fixture_fallback", "fixture_component", "fixture fallback", "unit test")
    journal = RecordingJournal()
    limits = plain.Limits(model="unit-model", max_requests=1)
    plain.run_loop("task", provider=_provider, execute=lambda name, args: "", limits=limits,
                   journal=journal, sleep=lambda seconds: None)
    kinds = [kind for kind, _ in journal.events]
    assert kinds[-1] == "reason_log_summary"
    assert kinds.index("loop_ended") < kinds.index("reason_log_summary")
    summary_fields = journal.events[-1][1]
    assert summary_fields["warnings"] == 1 and summary_fields["errors"] == 0
    assert stream.getvalue().count("[stateframe] problems: 0 ERROR, 1 WARN") == 1
    # A second flush in the same process stays silent: the summary is idempotent per run.
    assert rl.flush_summary(journal) is None


def test_taxonomy_record_failure_is_debug_only(stream, monkeypatch):
    monkeypatch.setattr(taxonomy, "enabled", lambda runtime=None: True)

    def broken(*args, **kwargs):
        raise RuntimeError("fixture taxonomy failure")

    monkeypatch.setattr(taxonomy, "classify_run_end", broken)
    runtime = types.SimpleNamespace(journal=None)
    assert taxonomy.record(runtime, {}) is None
    # DEBUG is counted for the run but never printed at the default WARN level.
    assert stream.getvalue() == ""
    cause = rl.run_summary()["by_cause"]["degrade:stop_reason_taxonomy_failed"]
    assert cause["level"] == "DEBUG" and cause["count"] == 1


def test_tokenizer_fallback_reported_once_per_child(request, stream, monkeypatch):
    host_state = dict(_child.HOST_STATE)  # install_token_count_warning writes HOST_STATE; restore it after
    request.addfinalizer(lambda: (_child.HOST_STATE.clear(), _child.HOST_STATE.update(host_state)))
    class RequestTokenMeter:
        def measure(self, body):
            return (len(body), "calibrated_bytes")

        def unavailable_reason(self):
            return "endpoint_down"

    fake_diet = types.ModuleType("kernel_request_diet")
    fake_diet.RequestTokenMeter = RequestTokenMeter
    monkeypatch.setitem(sys.modules, "kernel_request_diet", fake_diet)
    _child.install_token_count_warning()
    meter = RequestTokenMeter()
    assert meter.measure("abc")[1] == "calibrated_bytes"
    meter.measure("abcd")
    lines = [line for line in stream.getvalue().splitlines() if ": " in line and "engine_child" in line]
    assert len(lines) == 1
    assert lines[0].startswith("[stateframe] WARN engine_child: ")
    assert "endpoint_down" in lines[0]


def test_capability_probe_failure_logged_as_fallback(stream, monkeypatch):
    def no_python(*args, **kwargs):
        raise OSError("fixture interpreter missing")

    monkeypatch.setattr(runner.subprocess, "run", no_python)
    options = types.SimpleNamespace(python="/nonexistent/python", _deadline=None, timeout_s=5)
    receipt = runner._probe_capabilities(options)
    assert receipt["yaml"]["enabled"] is None
    assert "[stateframe] WARN runner: capability probe unavailable" in stream.getvalue()
    assert runner._reason_log.run_summary()["by_cause"]["fallback:capability_probe_unavailable"]["count"] == 1


def test_slot_status_failure_logged_as_fallback(stream, monkeypatch):
    class BrokenEndpoint:
        timeout_s = 5

        def get_slots(self):
            raise OSError("fixture slots unreachable")

    monkeypatch.setattr(runner, "_provider_capabilities_for",
                        lambda endpoint, key, options: types.SimpleNamespace(slots=True))
    monkeypatch.setattr(runner, "replace", lambda endpoint, **changes: BrokenEndpoint())
    endpoint = types.SimpleNamespace(chat_url="http://127.0.0.1:1/v1", api_key=lambda: "", timeout_s=5)
    options = types.SimpleNamespace(endpoint=endpoint, _deadline=None, timeout_s=5, _context_tokens=65_536)
    runner.check_slot_context(options)
    assert "[stateframe] WARN runner: context slot check skipped" in stream.getvalue()


def test_chat_problems_line_only_when_problems_exist(tmp_path):
    lines = []
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"problems": {"errors": 1, "warnings": 2}}), encoding="utf-8")
    chat._print_problems_line(summary, lines.append)
    assert lines == ['[stateframe] problems: 1 ERROR, 2 WARN (see summary.json "problems")']
    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps({"problems": {"errors": 0, "warnings": 0}}), encoding="utf-8")
    chat._print_problems_line(clean, lines.append)
    chat._print_problems_line(tmp_path / "missing.json", lines.append)
    assert len(lines) == 1

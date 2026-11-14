"""RB6/RB7 legacy degrade prints route through the reason log: one stderr line per cause per run."""
import io
import re
import threading

import pytest

import expectation_ledger
import kernel_hypotheses
import kernel_nitpicker
import kernel_reason_log as rl
import kernel_silent_off_organs
from expectation_ledger import run_typed_suite
from kernel_liveness import _emit_receipt
import kernel_web_fetch as web_fetch
from kernel_web_fetch import WebPolicy, fetch

LINE = re.compile(r"^\[stateframe\] (WARN|ERROR) (?P<component>[a-z_]+): .+", re.M)


@pytest.fixture(autouse=True)
def stream(monkeypatch):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    out = io.StringIO()
    rl.configure(level="WARN", stream=out)
    yield out
    rl.reset_for_tests()


def _components(out):
    return [m.group("component") for m in LINE.finditer(out.getvalue())]


def test_hypothesis_tracker_degrade_is_one_line(stream):
    tracker = kernel_hypotheses.HypothesisTracker("abc12345", trace_cb=None)
    tracker._degrade(RuntimeError("graph broke"))
    tracker._degrade(RuntimeError("graph broke again"))
    assert _components(stream) == ["hypotheses"]
    assert "RuntimeError: graph broke" in stream.getvalue()
    # _degrade is permanent-silence after the first call, so the second never reaches the log.
    assert rl.run_summary()["by_cause"]["degrade:hypotheses:degraded"]["count"] == 1


def test_nitpicker_trace_failure_disables_trace_and_logs_once(stream):
    def broken_trace(*_args):
        raise ValueError("sink down")

    state = kernel_nitpicker.NitpickerState(task_id="t1", target_symbol="fn", trace_cb=broken_trace)
    # __post_init__ already fired one trace event; the callback is now disabled.
    assert state.trace_cb is None
    state.trace_cb = broken_trace
    state._record("NITPICKER_PROBE", "again")
    assert state.trace_cb is None
    assert _components(stream) == ["nitpicker"]
    assert "ValueError" in stream.getvalue()


def test_web_fetch_refusals_log_one_line_per_cause(stream):
    policy = WebPolicy(("example.com",), allowed_ports=(443,))
    assert not fetch("ftp://example.com/x", policy).ok
    assert not fetch("ftp://example.com/y", policy).ok
    assert not fetch("http://example.com:81/", policy).ok
    assert _components(stream) == ["web_fetch", "web_fetch"]
    causes = rl.run_summary()["by_cause"]
    assert causes["refusal:web_fetch:url_policy"]["count"] == 2
    assert causes["refusal:web_fetch:port_policy"]["count"] == 1


def test_web_fetch_deadline_logs_timeout_category(stream):
    release = threading.Event()

    def blocked_resolver(host, port, *, type):
        release.wait(5)
        return []

    try:
        result = fetch("http://example.com/", WebPolicy((), deadline=0.02), resolver=blocked_resolver)
    finally:
        release.set()
    assert not result.ok and "deadline" in result.refusal
    assert _components(stream) == ["web_fetch"]
    assert "timeout:web_fetch:deadline" in rl.run_summary()["by_cause"]


def test_liveness_receipt_sink_failure_logs_once(stream):
    def failing_sink(_receipt):
        raise RuntimeError("sink unavailable")

    for _ in range(3):
        assert _emit_receipt({}, receipt_path=None, receipt_sink=failing_sink) == (
            "receipt_sink_failed:RuntimeError",)
    assert _components(stream) == ["liveness"]
    assert rl.run_summary()["by_cause"]["degrade:liveness:census_sink"]["count"] == 3


def test_expectation_suite_runner_failure_logs_once(stream, tmp_path):
    def failing_runner(*_args, **_kwargs):
        raise RuntimeError("runner unavailable")

    for _ in range(2):
        result = run_typed_suite(tmp_path, run=failing_runner)
        assert result.status is expectation_ledger.SuiteStatus.INFRA_ERROR
    assert _components(stream) == ["expectation_ledger"]


def test_silent_off_organ_failure_logs_with_journal(stream, monkeypatch):
    class Journal:
        def __init__(self):
            self.events = []

        def emit(self, kind, **payload):
            self.events.append((kind, payload))

    def boom(*_args, **_kwargs):
        raise RuntimeError("ledger import broke")

    monkeypatch.setattr(kernel_silent_off_organs, "_requested", lambda organ: (True, None))
    monkeypatch.setattr(expectation_ledger, "ExpectationLedger", boom)
    journal = Journal()
    wiring = kernel_silent_off_organs.wire_expectation_ledger(journal)
    assert wiring.effective == "failed"
    assert _components(stream) == ["silent_off_organs"]
    reasons = [payload for kind, payload in journal.events if kind == "reason_logged"]
    assert [row["cause_key"] for row in reasons] == ["silent_off_organs:expectation_ledger"]


@pytest.mark.parametrize("reason,cause", [
    ("TimeoutError", "timeout"),
    ("web fetch total deadline exceeded during DNS resolution", "deadline"),
    ("DNS returned no addresses", "dns"),
    ("host is outside the web allow list: evil.test", "host_policy"),
    ("port is not allowed: 81", "port_policy"),
    ("response body exceeds 2097152 bytes", "size_limit"),
    ("something unexpected", "other"),
])
def test_web_fetch_refusal_cause_is_stable_and_coarse(reason, cause):
    assert web_fetch._refusal_cause(reason) == cause

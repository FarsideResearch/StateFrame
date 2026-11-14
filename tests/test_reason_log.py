"""kernel_reason_log: never raises, dedupes, formats, gates by level, summarizes, reaches the child."""
import copy
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"))
import kernel_reason_log as rl  # noqa: E402

LINE = re.compile(r"^\[stateframe\] (ERROR|WARN) [a-z_]+: .+ \(.+\)$")


class Journal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append((kind, fields))


class Raising:
    def emit(self, kind, **fields):
        raise RuntimeError("journal closed")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    yield
    rl.reset_for_tests()


def cfg(**kw):
    stream = io.StringIO()
    rl.configure(stream=stream, **kw)
    return stream


def test_never_raises_with_bad_journals_and_args():
    cfg()
    class NoEmit: pass
    for journal in (None, Raising(), NoEmit(), object()):
        assert isinstance(rl.emit_reason("kill", f"k{id(journal)}", "c", "w", "y", journal=journal), bool)
    assert rl.emit_kill("c", target=object(), why=None, pid="x", elapsed_s="bad") in (True, False)
    assert isinstance(rl.emit_refusal("c", what="w", why="y", dropped=object(), measured=lambda: 1), bool)
    assert isinstance(rl.emit_reason(None, None, None, None, None, level=5, journal=3, bad={1: 2}), bool)
    assert rl.flush_summary(Raising()) is not None or True


def test_unwritable_problems_file_is_swallowed(tmp_path):
    stream = cfg(problems_path=tmp_path / "no" / "such" / "p.jsonl")
    assert rl.emit_reason("kill", "a", "c", "w", "y") is True
    assert stream.getvalue().count("\n") == 1


def test_dedupe_one_event_one_line_counts_all():
    journal = Journal()
    stream = cfg(journal=journal)
    results = [rl.emit_kill("kernel_probe", target="probe", why="wall_deadline", pid=i) for i in range(1000)]
    assert results.count(True) == 1
    assert len(journal.events) == 1 and journal.events[0][0] == "reason_logged"
    assert stream.getvalue().count("\n") == 1
    assert rl.emit_kill("kernel_probe", target="probe", why="other") is True
    summary = rl.run_summary()
    assert summary["by_cause"]["kill:kernel_probe:wall_deadline"]["count"] == 1000
    assert summary["total"] == 1001


def test_stderr_format_and_newline_collapse():
    stream = cfg()
    rl.emit_kill("kernel_probe", target="probe", pid=4123, unit="probe-17", why="wall 30.0s exceeded\ninjected",
                 measured=31.2, limit=30.0, elapsed_s=31.2)
    rl.emit_refusal("budget", what="dropped\r\nfile", why="over cap", limit=10, measured=20, dropped="a/b.py:10-40",
                    level="WARN")
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    for line in lines:
        assert LINE.match(line) and len(line) <= 400
    assert lines[0].startswith("[stateframe] ERROR kernel_probe: killed probe pid 4123 unit 'probe-17' (wall 30.0s")
    assert "dropped a/b.py:10-40" in lines[1] and "measured 20 vs limit 10" in lines[1]


def test_level_gating_counts_but_hides(monkeypatch, tmp_path):
    monkeypatch.setenv(rl.LEVEL_ENV, "error")
    journal = Journal()
    path = tmp_path / "p.jsonl"
    stream = cfg(journal=journal, problems_path=path)
    rl.emit_reason("fallback", "w", "c", "what", "why", level="WARN")
    rl.emit_reason("kill", "e", "c", "what", "why", level="ERROR")
    assert stream.getvalue().count("ERROR") == 1 and "WARN" not in stream.getvalue()
    assert [e[1]["level"] for e in journal.events] == ["ERROR"]
    assert len(path.read_text().splitlines()) == 1
    assert rl.run_summary()["by_cause"]["fallback:w"]["level"] == "WARN"


def test_invalid_level_defaults_to_warn_with_one_warning(monkeypatch):
    monkeypatch.setenv(rl.LEVEL_ENV, "loud")
    stream = cfg()
    rl.configure(stream=stream)
    rl.emit_reason("fallback", "w", "c", "what", "why")
    out = stream.getvalue()
    assert out.count("invalid STATEFRAME_LOG_LEVEL") == 1 and "fallback" not in out and "what" in out


def test_info_level_shows_info(monkeypatch):
    monkeypatch.setenv(rl.LEVEL_ENV, "info")
    stream = cfg()
    rl.emit_reason("fallback", "i", "c", "recovered", "ok", level="INFO")
    assert "[stateframe] INFO c: recovered" in stream.getvalue()


def test_request_bytes_unchanged():
    body = {"messages": [{"role": "user", "content": "x"}], "tools": [{"name": "t"}]}
    before = copy.deepcopy(body)
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    ident = id(body)
    cfg(journal=Journal())
    rl.emit_reason("refusal", "b", "c", "w", "y", payload=body, dropped=body)
    rl.emit_refusal("c", what="w", why="z", dropped=body)
    assert body == before and id(body) == ident
    assert hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == digest


def test_summary_flush_once_and_problems_file(tmp_path):
    journal = Journal()
    path = tmp_path / "p.jsonl"
    stream = cfg(journal=journal, problems_path=path)
    rl.emit_kill("c", target="t", why="w1")
    rl.emit_kill("c", target="t", why="w1")
    rl.emit_timeout("c", what="rpc", why="slow", deadline_s=5, elapsed_s=6)
    assert rl.flush_summary() is not None and rl.flush_summary() is None
    kinds = [k for k, _ in journal.events]
    assert kinds.count("reason_log_summary") == 1
    assert "[stateframe] problems: 1 ERROR, 1 WARN" in stream.getvalue()
    merged = rl.problems_from_file(path)
    assert merged["errors"] == 1 and merged["warnings"] == 1
    assert merged["by_cause"]["kill:c:w1"]["count"] == 2
    assert merged["by_cause"]["kill:c:w1"]["first"]["target"] == "t"


def test_flush_silent_when_clean():
    stream = cfg()
    assert rl.flush_summary() is None and stream.getvalue() == ""


def test_plain_journal_write_fallback():
    class Plain:
        def __init__(self): self.rows = []
        def write(self, kind, **f): self.rows.append(kind)
    plain = Plain()
    cfg()
    rl.emit_reason("kill", "p", "c", "w", "y", journal=plain)
    assert plain.rows == ["reason_logged"]


def test_child_env_plumbing(tmp_path):
    import os
    from unittest import mock
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from stateframe import _child
    spec = {"environment": {"STATEFRAME_LOG_LEVEL": "error", "STATEFRAME_PROBLEMS_FILE": str(tmp_path / "p")}}
    with mock.patch.dict(os.environ, {"STATEFRAME_LOG_LEVEL": "debug"}):  # inherited: stripped
        _child.install_environment(spec)
        assert os.environ["STATEFRAME_LOG_LEVEL"] == "error"
        assert os.environ["STATEFRAME_PROBLEMS_FILE"] == str(tmp_path / "p")


def test_runner_problems_summary_and_printing(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from stateframe import runner
    (tmp_path / "problems.jsonl").write_text(
        json.dumps({"level": "ERROR", "category": "kill", "cause_key": "a", "component": "ca", "what": "w",
                    "why": "y", "fields": {"pid": 1}}) + "\n"
        + json.dumps({"level": "WARN", "category": "timeout", "cause_key": "b", "component": "cb",
                      "what": "w2", "why": "y2", "fields": {}}) + "\n")
    problems = runner._problems_summary(tmp_path)
    assert problems["errors"] == 1 and problems["warnings"] == 1 and problems["log_level"] == "warn"
    assert problems["by_cause"]["kill:a"]["first"] == {"pid": 1}
    out = []
    runner._print_problems(problems, out.append, "warn")
    assert out[0].startswith("[stateframe] problems: 1 ERROR, 1 WARN")
    assert len(out) == 3 and all(LINE.match(x) for x in out[1:])
    clean = []
    runner._print_problems(runner._problems_summary(tmp_path / "none"), clean.append, "warn")
    assert clean == []
    runner._print_problems(runner._problems_summary(tmp_path / "none"), clean.append, "info")
    assert clean == ["[stateframe] problems: none"]


def test_flush_ignores_causes_below_level(tmp_path):
    journal = Journal()
    stream = cfg(journal=journal, problems_path=tmp_path / "p.jsonl")
    rl.emit_reason("degrade", "a", "c", "w", "y", level="INFO")
    rl.emit_reason("degrade", "b", "c", "w", "y", level="DEBUG")
    assert rl.flush_summary() is None
    assert not [e for e in journal.events if e[0] == "reason_log_summary"]
    assert stream.getvalue() == "" and not (tmp_path / "p.jsonl").exists()


def test_flush_counts_only_causes_at_level():
    journal = Journal()
    cfg(journal=journal)
    rl.emit_reason("degrade", "a", "c", "w", "y", level="INFO")
    rl.emit_reason("kill", "b", "c", "w", "y", level="ERROR")
    summary = rl.flush_summary()
    assert list(summary["by_cause"]) == ["kill:b"] and summary["total"] == 1
    ev = [f for k, f in journal.events if k == "reason_log_summary"]
    assert len(ev) == 1 and ev[0]["errors"] == 1 and list(ev[0]["by_cause"]) == ["kill:b"]

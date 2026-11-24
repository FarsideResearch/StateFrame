"""An unreadable problems file must not be reported as a clean run."""
from __future__ import annotations

from pathlib import Path

from stateframe import runner


def test_unreadable_problems_file_is_unknown_not_zero(tmp_path):
    (tmp_path / runner.PROBLEMS_FILE).mkdir()  # exists but cannot be read as a file
    problems = runner._problems_summary(tmp_path)
    assert problems["errors"] is None and problems["warnings"] is None
    assert problems["unknown"] == "problems file unreadable: IsADirectoryError"
    lines: list[str] = []
    runner._print_problems(problems, lines.append, "warn")
    assert lines == ["[stateframe] problems: unknown (problems file unreadable: IsADirectoryError)"]


def test_problems_read_failure_is_unknown_after_successful_open(tmp_path, monkeypatch):
    path = tmp_path / runner.PROBLEMS_FILE
    path.write_text("{}\n")
    assert path.is_file()
    with path.open("rb") as stream:
        assert stream.read() == b"{}\n"

    original_read_text = Path.read_text
    attempted_reads: list[Path] = []

    def fail_problems_read(target, *args, **kwargs):
        if target == path:
            attempted_reads.append(target)
            raise PermissionError("private exception detail")
        return original_read_text(target, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_problems_read)
    problems = runner._problems_summary(tmp_path)
    lines: list[str] = []
    runner._print_problems(problems, lines.append, "warn")

    assert attempted_reads == [path]
    assert problems["errors"] is None and problems["warnings"] is None
    assert problems["unknown"] == "problems file unreadable: PermissionError"
    assert lines == ["[stateframe] problems: unknown (problems file unreadable: PermissionError)"]
    assert all("private exception detail" not in line for line in lines)


def test_problems_summary_retains_sink_self_errors(tmp_path):
    import json

    rows = [
        {"level": "ERROR", "category": "kill", "cause_key": "legacy-error", "component": "worker",
         "what": "worker stopped", "why": "limit", "fields": {}},
        {"level": "WARN", "category": "timeout", "cause_key": "legacy-warn", "component": "provider",
         "what": "request waited", "why": "slow", "fields": {}},
        {"summary": True, "pid": 101, "by_cause": {}, "self_errors": 3},
        {"summary": True, "pid": 101, "by_cause": {}, "self_errors": 3},
        {"summary": True, "pid": 303, "by_cause": {}, "self_errors": 2},
        {"summary": True, "pid": 202, "by_cause": {}},
        {"summary": True, "pid": 404, "by_cause": {
            "kill:legacy-error": {"count": 2, "level": "ERROR", "component": "worker",
                                  "what": "worker stopped", "why": "limit", "first": {}}},
         "self_errors": 0},
    ]
    (tmp_path / runner.PROBLEMS_FILE).write_text("".join(json.dumps(row) + "\n" for row in rows))

    problems = runner._problems_summary(tmp_path)
    lines: list[str] = []
    runner._print_problems(problems, lines.append, "warn")
    summary_diagnostic = problems.get("self_errors")
    diagnostic_line = (
        "[stateframe] problem diagnostics incomplete: reason log reported 5 self-errors; some causes may be missing")
    print_diagnostic = diagnostic_line in lines

    assert problems["errors"] == 1 and problems["warnings"] == 1
    assert problems["by_cause"]["kill:legacy-error"]["count"] == 2
    assert problems["by_cause"]["timeout:legacy-warn"]["count"] == 1
    assert (summary_diagnostic, print_diagnostic) == (5, True), (problems, lines)
    assert lines[0].startswith("[stateframe] problems: 1 ERROR, 1 WARN")


def test_missing_problems_file_is_still_clean(tmp_path):
    problems = runner._problems_summary(tmp_path)
    assert problems["errors"] == 0 and problems["warnings"] == 0
    assert "unknown" not in problems


def test_merge_failure_is_unknown(tmp_path, monkeypatch):
    (tmp_path / runner.PROBLEMS_FILE).write_text("")

    def boom(*_a, **_k):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(runner._reason_log, "problems_from_file", boom)
    problems = runner._problems_summary(tmp_path)
    assert problems["errors"] is None
    assert problems["unknown"] == "problems merge failed: RuntimeError"


def test_chat_and_inspect_show_unknown(tmp_path):
    import json
    from stateframe import chat, inspect_run
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"problems": {"errors": None, "warnings": None, "unknown": "x: OSError"}}))
    lines: list[str] = []
    chat._print_problems_line(summary, lines.append)
    assert lines == ["[stateframe] problems: unknown (x: OSError)"]
    (tmp_path / "wire").mkdir()
    assert "problems: unknown (x: OSError)" in inspect_run.overview(tmp_path)


def test_empty_and_non_utf8_problems_file_are_clean(tmp_path):
    path = tmp_path / runner.PROBLEMS_FILE
    for body in (b"", b"\xff\xfe\x00not json\r\n"):
        path.write_bytes(body)
        problems = runner._problems_summary(tmp_path)
        assert problems["errors"] == 0 and "unknown" not in problems

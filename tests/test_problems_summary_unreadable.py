"""An unreadable problems file must not be reported as a clean run."""
from __future__ import annotations

from stateframe import runner


def test_unreadable_problems_file_is_unknown_not_zero(tmp_path):
    (tmp_path / runner.PROBLEMS_FILE).mkdir()  # exists but cannot be read as a file
    problems = runner._problems_summary(tmp_path)
    assert problems["errors"] is None and problems["warnings"] is None
    assert problems["unknown"] == "problems file unreadable: IsADirectoryError"
    lines: list[str] = []
    runner._print_problems(problems, lines.append, "warn")
    assert lines == ["[stateframe] problems: unknown (problems file unreadable: IsADirectoryError)"]


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

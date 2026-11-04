"""Every journaled unavailable event must say why it could not proceed."""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from _class_guard_scanners import scan_unavailable_events


_UNAVAILABLE_SCAN_FLOOR = 12  # Current literal journal emissions in src/stateframe.


def test_unavailable_events_include_a_reason() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src/stateframe").rglob("*.py"))
    report = scan_unavailable_events(sources, root=root)
    print(report.receipt)

    assert re.fullmatch(
        r"unavailable-reason scan: scanned=\d+ with_reason=\d+ violations=\d+",
        report.receipt,
    )
    assert report.scanned == report.with_reason + len(report.violations)
    assert report.scanned >= _UNAVAILABLE_SCAN_FLOOR
    assert not report.violations, "unavailable events missing reason:\n" + "\n".join(report.violations)


def test_source_scanner_rejects_a_planted_unavailable_event(tmp_path: Path) -> None:
    planted = tmp_path / "planted_event.py"
    planted.write_text(
        'def call(emit):\n    emit("x_unavailable")\n',
        encoding="utf-8",
    )
    report = scan_unavailable_events([planted], root=tmp_path)
    assert report.scanned == 1
    assert report.with_reason == 0
    assert len(report.violations) == 1
    assert report.receipt == "unavailable-reason scan: scanned=1 with_reason=0 violations=1"


def test_source_scanner_rejects_a_blank_unavailable_reason(tmp_path: Path) -> None:
    planted = tmp_path / "blank_reason.py"
    planted.write_text(
        'def call(emit):\n    emit("x_unavailable", reason="  ")\n',
        encoding="utf-8",
    )
    report = scan_unavailable_events([planted], root=tmp_path)
    assert report.scanned == 1
    assert report.with_reason == 0
    assert len(report.violations) == 1


def test_source_scanner_resolves_an_event_name_variable(tmp_path: Path) -> None:
    planted = tmp_path / "variable_event.py"
    planted.write_text(
        'def call(emit):\n    event = "x_unavailable"\n    emit(event)\n',
        encoding="utf-8",
    )
    report = scan_unavailable_events([planted], root=tmp_path)
    assert report.receipt == "unavailable-reason scan: scanned=1 with_reason=0 violations=1"


def test_source_scanner_rejects_placeholder_reasons(tmp_path: Path) -> None:
    planted = tmp_path / "placeholder_reason.py"
    planted.write_text(
        'def call(emit):\n'
        '    emit("first_unavailable", reason="unavailable")\n'
        '    emit("second_unavailable", reason=None)\n'
        '    reason = None\n'
        '    emit("third_unavailable", reason=reason)\n',
        encoding="utf-8",
    )
    report = scan_unavailable_events([planted], root=tmp_path)
    assert report.receipt == "unavailable-reason scan: scanned=3 with_reason=0 violations=3"


def test_abandoned_readiness_decision_reports_the_consumer_failure() -> None:
    from kernel_done_readiness_authority import abandon_decision

    events = []
    journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))
    runtime = SimpleNamespace(journal=journal, current_step=1)
    decision = SimpleNamespace(key=("artifact", "evidence", "scope"))

    abandon_decision(runtime, decision, RuntimeError("private detail"))

    assert len(events) == 1
    kind, fields = events[0]
    assert kind == "done_decision_unavailable"
    # The existing event fields stay intact; the new reason names the failed stage.
    assert fields["producer"] == "kernel_done_readiness_authority"
    assert fields["decision"] == "FALLBACK_LEGACY"
    assert fields["reason_code"] == "NO_CALLABLE_RECOVERY"
    assert fields["recovery"] is None
    assert fields["dispatch"] is False
    assert fields["reason"] == "readiness decision consumer failed: RuntimeError"
    assert "private detail" not in fields["reason"]

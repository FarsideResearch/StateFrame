from __future__ import annotations

import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

import kernel_reason_log as reason_log
from kernel_run_vitals import FLAG_ENV, RunVitalsAccumulator, emit_run_vitals


class _TerminalJournal:
    def __init__(self, error: BaseException | None = None) -> None:
        self._run_vitals_accumulator = RunVitalsAccumulator()
        self.terminal_outcome = "COMPLETED"
        self.attempts: list[tuple[str, dict[str, object]]] = []
        self.error = error

    def emit(self, event: str, **payload: object) -> object:
        self.attempts.append((event, payload))
        if self.error is not None:
            raise self.error
        return {"event": event, **payload}


@pytest.fixture
def reason_output(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    monkeypatch.delenv(reason_log.PROBLEMS_ENV, raising=False)
    monkeypatch.delenv(reason_log.LEVEL_ENV, raising=False)
    monkeypatch.setattr(reason_log, "_STATE", reason_log._fresh())
    output = StringIO()
    reason_log.configure(level="WARN", stream=output)
    return output


def test_failed_terminal_vitals_emit_is_visible_without_a_second_journal_write(
    monkeypatch: pytest.MonkeyPatch,
    reason_output: StringIO,
) -> None:
    monkeypatch.setenv(FLAG_ENV, "on")
    journal = _TerminalJournal(RuntimeError("private terminal journal detail"))
    reason_log.bind_journal(journal)
    terminal_outcome = journal.terminal_outcome

    assert emit_run_vitals(journal) is None
    assert emit_run_vitals(journal) is None

    assert [event for event, _payload in journal.attempts] == ["run_vitals"]
    diagnostics = reason_output.getvalue().splitlines()
    assert len(diagnostics) == 1
    assert "run_vitals_emit_failed" in diagnostics[0]
    assert "RuntimeError" in diagnostics[0]
    assert "private terminal journal detail" not in diagnostics[0]
    assert journal.terminal_outcome == terminal_outcome


def test_successful_terminal_vitals_emit_returns_the_emitted_record(
    monkeypatch: pytest.MonkeyPatch,
    reason_output: StringIO,
) -> None:
    monkeypatch.setenv(FLAG_ENV, "on")
    journal = _TerminalJournal()

    result = emit_run_vitals(journal)

    assert isinstance(result, dict)
    assert result["event"] == "run_vitals"
    assert [event for event, _payload in journal.attempts] == ["run_vitals"]
    assert reason_output.getvalue() == ""


def test_off_mode_does_not_attempt_terminal_vitals_emit(
    monkeypatch: pytest.MonkeyPatch,
    reason_output: StringIO,
) -> None:
    monkeypatch.setenv(FLAG_ENV, "off")
    journal = _TerminalJournal(RuntimeError("should not be reached"))

    assert emit_run_vitals(journal) is None

    assert journal.attempts == []
    assert reason_output.getvalue() == ""


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_terminal_vitals_emit_propagates_interrupts(
    monkeypatch: pytest.MonkeyPatch,
    reason_output: StringIO,
    interrupt: type[BaseException],
) -> None:
    monkeypatch.setenv(FLAG_ENV, "on")
    journal = _TerminalJournal(interrupt())
    terminal_outcome = journal.terminal_outcome

    with pytest.raises(interrupt):
        emit_run_vitals(journal)

    assert [event for event, _payload in journal.attempts] == ["run_vitals"]
    assert journal.terminal_outcome == terminal_outcome
    assert reason_output.getvalue() == ""


def test_package_import_reports_the_failed_write(tmp_path: Path) -> None:
    """With only src on the path the engine modules exist under their package name alone."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from io import StringIO\n"
        "import stateframe.engine.kernel_reason_log as reason_log\n"
        "import stateframe.engine.kernel_run_vitals as vitals\n"
        "class Journal:\n"
        "    _run_vitals_accumulator = vitals.RunVitalsAccumulator()\n"
        "    def emit(self, *args, **kwargs):\n"
        "        raise RuntimeError('private journal message')\n"
        "output = StringIO()\n"
        "reason_log.configure(level='WARN', stream=output)\n"
        "vitals.emit_run_vitals(Journal())\n"
        "print(output.getvalue())\n",
        encoding="utf-8",
    )
    source = Path(__file__).resolve().parents[1] / "src"
    environment = {"PATH": os.environ["PATH"], "PYTHONPATH": str(source), FLAG_ENV: "on"}
    done = subprocess.run([sys.executable, str(probe)], env=environment, cwd=tmp_path,
                          capture_output=True, text=True, check=True)
    assert "run_vitals_emit_failed" in done.stdout
    assert "private journal message" not in done.stdout + done.stderr

"""Edit n-turn continuation has no engine clock budget; only a run deadline the user set limits it.

Clock values are injected (no sleeping, no os.environ mutation). Engine modules are imported
the way tests/test_reason_log_limits.py does; new names are referenced inside test bodies so
the RED run reports per-test failures instead of a collection error.
"""
import io
import math
from pathlib import Path

import pytest

import kernel_reason_log as rl
import kernel_step_contract as sc

ROOT = Path(__file__).resolve().parents[1]
REACT_REPAIR = ROOT / "src" / "stateframe" / "engine" / "probes" / "react_repair.py"
RETIRED_NAME = "STATEFRAME_EDIT_NTURN_RUN_BUDGET_SECONDS"


@pytest.fixture()
def out(monkeypatch):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    stream = io.StringIO()
    rl.configure(level="WARN", stream=stream)
    yield stream
    rl.reset_for_tests()


class _Journal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append((kind, fields))
        return fields


def test_no_engine_clock_budget_exists():
    from stateframe import runner

    assert not hasattr(sc, "EDIT_NTURN_RUN_BUDGET_SECONDS")
    assert not hasattr(sc, "EDIT_NTURN_RUN_BUDGET_ENV")
    assert not hasattr(sc, "edit_nturn_run_budget_seconds")
    assert not hasattr(runner, "resolve_edit_nturn_budget_env")
    for base in (ROOT / "src", ROOT / "docs"):
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert RETIRED_NAME not in text, f"retired override still named in {path.relative_to(ROOT)}"
    for doc in [*(ROOT / "docs" / "guide").glob("*.md"), ROOT / "README.md"]:
        assert RETIRED_NAME not in doc.read_text(encoding="utf-8"), doc.name


def test_long_run_without_user_deadline_still_admits_edit_turns():
    remaining = sc.edit_nturn_remaining_seconds(now=10_000_000.0, sandbox_deadline=None)
    assert math.isinf(remaining)
    assert sc.edit_nturn_budget_allows(remaining, 300.0) is True


def test_user_deadline_still_limits():
    remaining = sc.edit_nturn_remaining_seconds(now=100.0, sandbox_deadline=150.0)
    assert remaining == pytest.approx(50.0)
    assert sc.edit_nturn_budget_allows(remaining, 300.0) is False
    assert sc.edit_nturn_remaining_seconds(now=200.0, sandbox_deadline=150.0) == 0.0


def test_nan_remaining_fails_closed():
    assert sc.edit_nturn_budget_allows(math.nan, 300.0) is False


def test_react_repair_has_no_engine_clock_budget():
    text = REACT_REPAIR.read_text(encoding="utf-8")
    for retired in ("_edit_nturn_provider_wait", "_provider_t0", "EDIT_NTURN_RUN_BUDGET", '"wall_clock"'):
        assert retired not in text, retired
    assert "edit_nturn_remaining_seconds(" in text


def test_skip_log_names_the_run_deadline_not_an_engine_budget(out):
    journal = _Journal()
    sc.emit_edit_nturn_skipped_budget(journal, n_sites=2, turns_used=1, remaining_seconds=3.0,
                                      required_seconds=12.0)
    text = out.getvalue()
    assert "run deadline" in text
    assert "EDIT_NTURN_RUN_BUDGET_SECONDS" not in text
    assert [kind for kind, _ in journal.events] == ["edit_nturn_skipped_budget"]
    assert journal.events[0][1]["reason"] == "run_deadline"

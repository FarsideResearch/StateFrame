"""pytest plugin that records what a run actually collected and how each test ended (loaded with ``-p``).

tests/test_regfam_f8.py and tests/_gate_verdict.py run pytest in a child process with this plugin and
``SF_COLLECTION_CENSUS`` naming an output file. The child writes one JSON document: every collected item
(node id, file, marker names), every collect report that failed or was skipped, each test's final outcome
(passed / failed / skipped / xfailed / xpassed) and the session exit status. Without the variable the plugin
does nothing, so loading it by accident is harmless.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

CENSUS_ENV = "SF_COLLECTION_CENSUS"

_reports: list[dict[str, str]] = []
_items: list[dict[str, object]] = []
_outcomes: dict[str, str] = {}


def _relative(path: object, root: Path) -> str:
    candidate = Path(str(path)).resolve()
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return candidate.as_posix()


def _write(extra: dict[str, object]) -> None:
    target = os.environ.get(CENSUS_ENV)
    if target:
        document = {"items": _items, "reports": _reports, "outcomes": _outcomes, **extra}
        Path(target).write_text(json.dumps(document), encoding="utf-8")


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if not os.environ.get(CENSUS_ENV) or report.outcome == "passed":
        return
    _reports.append({
        "nodeid": report.nodeid,
        "outcome": report.outcome,
        "detail": str(report.longrepr)[-2000:],
    })


def pytest_collection_finish(session: pytest.Session) -> None:
    if not os.environ.get(CENSUS_ENV):
        return
    root = Path(str(session.config.rootpath)).resolve()
    _items[:] = [
        {
            "nodeid": item.nodeid,
            "file": _relative(item.path, root),
            "markers": sorted({mark.name for mark in item.iter_markers()}),
        }
        for item in session.items
    ]
    _write({"phase": "collected"})


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Fold setup/call/teardown into one final outcome: a failed phase wins, an xfail is never a pass."""
    if not os.environ.get(CENSUS_ENV):
        return
    if _outcomes.get(report.nodeid) == "failed":
        return
    expected_failure = hasattr(report, "wasxfail")
    if report.failed:
        outcome = "failed"
    elif report.skipped:
        outcome = "xfailed" if expected_failure else "skipped"
    elif report.when == "call":
        outcome = "xpassed" if expected_failure else "passed"
    else:
        return  # a passing setup or teardown does not decide the outcome
    _outcomes[report.nodeid] = outcome


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if os.environ.get(CENSUS_ENV):
        _write({"phase": "finished", "exitstatus": int(exitstatus)})

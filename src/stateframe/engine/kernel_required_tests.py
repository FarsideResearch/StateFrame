"""Deterministic comparison of complete host-observed pytest runs."""
from __future__ import annotations

import json


SCHEMA = "stateframe-required-tests/1"
MAX_PYTEST_REPORT_BYTES = 1536 * 1024
_OUTCOMES = frozenset({"passed", "failed", "skipped"})
PYTEST_REPORT_NOT_PRODUCED = "pytest report was not produced"
_PYTEST_REPORT_REASON_MESSAGES = {
    "pytest_import_failed": "pytest could not be imported",
    "session_incomplete": "pytest session evidence was incomplete",
    "collection_only": "pytest was run in collection-only mode",
    "no_collected_test_set": "pytest did not report a collected test set",
    "tests_unfinished": "pytest did not finish every collected test",
    "unsupported_exit_status": "pytest returned an unsupported exit status",
    "invalid_test_identity": "pytest test identity was outside the checked repository or duplicated",
    "report_too_large": "pytest report exceeded its size bound",
    "report_write_failed": "pytest report was not produced",
}


class RequiredTests:
    """Compare baseline passes with the latest complete current check."""

    def __init__(self) -> None:
        self._baseline: frozenset[str] | None = None
        self._current: dict[str, str] | None = None

    def observe(self, kind: str, complete: bool, outcomes: dict[str, str]) -> None:
        if kind not in {"baseline", "current"}:
            raise ValueError("kind must be 'baseline' or 'current'")
        if not isinstance(complete, bool):
            raise ValueError("complete must be a bool")
        if not complete:
            return

        normalized = _validated_outcomes(outcomes)
        if kind == "baseline":
            self._baseline = frozenset(
                test_id for test_id, outcome in normalized.items() if outcome == "passed")
        else:
            self._current = normalized

    def has_observation(self, kind: str) -> bool:
        if kind == "baseline":
            return self._baseline is not None
        if kind == "current":
            return self._current is not None
        raise ValueError("kind must be 'baseline' or 'current'")

    def report(self) -> dict[str, object]:
        if self._baseline is None or self._current is None:
            return {"missing": [], "failed": [], "passed": [], "state": "unknown"}

        missing: list[str] = []
        failed: list[str] = []
        passed: list[str] = []
        for test_id in sorted(self._baseline):
            outcome = self._current.get(test_id)
            if outcome == "passed":
                passed.append(test_id)
            elif outcome == "failed":
                failed.append(test_id)
            else:
                # Skips and absent ids are not evidence of a passing current check.
                missing.append(test_id)
        state = "regressed" if missing or failed else "ok"
        return {"missing": missing, "failed": failed, "passed": passed, "state": state}

    def to_json(self) -> str:
        document = {
            "schema": SCHEMA,
            "baseline": None if self._baseline is None else sorted(self._baseline),
            "current": None if self._current is None else {
                test_id: self._current[test_id] for test_id in sorted(self._current)
            },
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(cls, value: str) -> RequiredTests:
        try:
            document = json.loads(value)
        except (TypeError, ValueError) as error:
            raise ValueError("required-test state must be valid JSON") from error
        if not isinstance(document, dict) or set(document) != {"schema", "baseline", "current"}:
            raise ValueError("required-test state has an invalid shape")
        if document.get("schema") != SCHEMA:
            raise ValueError("required-test state has an unsupported schema")

        baseline = document.get("baseline")
        if baseline is not None and (
                not isinstance(baseline, list)
                or any(not isinstance(test_id, str) or not test_id for test_id in baseline)
                or len(set(baseline)) != len(baseline)):
            raise ValueError("required-test baseline ids are invalid")
        current = document.get("current")
        if current is not None:
            current = _validated_outcomes(current)

        restored = cls()
        restored._baseline = None if baseline is None else frozenset(baseline)
        restored._current = current
        return restored


def _validated_outcomes(outcomes: object) -> dict[str, str]:
    if not isinstance(outcomes, dict):
        raise ValueError("outcomes must be a dict of test ids to statuses")
    normalized: dict[str, str] = {}
    for test_id, outcome in outcomes.items():
        if not isinstance(test_id, str) or not test_id:
            raise ValueError("test ids must be non-empty strings")
        if not isinstance(outcome, str) or outcome not in _OUTCOMES:
            raise ValueError(f"unsupported test outcome for {test_id!r}")
        normalized[test_id] = outcome
    return normalized


def _incomplete_report() -> dict[str, object]:
    return {"complete": False, "outcomes": {}}


def parse_pytest_report_with_reason(
        report: bytes, exit_code: int | None) -> tuple[dict[str, object], str | None]:
    """Accept complete reports or a known bounded reason for an incomplete one."""
    if not isinstance(report, bytes) or len(report) > MAX_PYTEST_REPORT_BYTES:
        return _incomplete_report(), PYTEST_REPORT_NOT_PRODUCED
    try:
        document = json.loads(report)
    except (TypeError, ValueError, RecursionError):
        return _incomplete_report(), PYTEST_REPORT_NOT_PRODUCED
    if isinstance(document, dict) and set(document) == {"incomplete_reason"}:
        code = document.get("incomplete_reason")
        message = (_PYTEST_REPORT_REASON_MESSAGES.get(code, PYTEST_REPORT_NOT_PRODUCED)
                   if isinstance(code, str) else PYTEST_REPORT_NOT_PRODUCED)
        return _incomplete_report(), message
    if not isinstance(document, dict) or set(document) != {"exit_code", "outcomes"}:
        return _incomplete_report(), PYTEST_REPORT_NOT_PRODUCED
    try:
        if type(document["exit_code"]) is not int or document["exit_code"] != exit_code:
            return _incomplete_report(), PYTEST_REPORT_NOT_PRODUCED
        outcomes = _validated_outcomes(document["outcomes"])
    except (ValueError, RecursionError):
        return _incomplete_report(), PYTEST_REPORT_NOT_PRODUCED
    expected_exit = (int("failed" in outcomes.values()) if outcomes else 5)
    if exit_code != expected_exit:
        return _incomplete_report(), PYTEST_REPORT_NOT_PRODUCED
    return {"complete": True, "outcomes": outcomes}, None


def parse_pytest_report(report: bytes, exit_code: int | None) -> dict[str, object]:
    """Accept only a complete session whose outcome agrees with the process."""
    return parse_pytest_report_with_reason(report, exit_code)[0]


# Invoked by the existing sandbox with the selected pytest interpreter. The host
# passes launch details as Python literals, keeping the declared pytest argv intact.
PYTEST_LEDGER_SOURCE = r'''
import json
import os
import sys

REASON_CODES = frozenset({
    "pytest_import_failed", "session_incomplete", "collection_only",
    "no_collected_test_set", "tests_unfinished", "unsupported_exit_status",
    "invalid_test_identity", "report_too_large", "report_write_failed",
})

def write_bytes(data):
    try:
        with open(report_path, "wb") as stream:
            if stream.write(data) != len(data):
                return False
        return True
    except Exception:
        return False

def write_incomplete(reason):
    if reason not in REASON_CODES:
        reason = "session_incomplete"
    try:
        encoded = json.dumps({"incomplete_reason": reason},
                             separators=(",", ":")).encode()
        if len(encoded) > 128:
            return False
    except Exception:
        return False
    return write_bytes(encoded)

if sys.path[:1] == [""]:
    sys.path[0] = os.path.dirname(os.path.realpath(console_script)) if console_script else repo
try:
    import pytest
    main, hookimpl = pytest.main, pytest.hookimpl
    sys.argv[0] = console_script or os.path.join(os.path.dirname(pytest.__file__), "__main__.py")
except Exception:
    write_incomplete("pytest_import_failed")
    os.execv(sys.executable, [sys.executable, *original_args, *sys.argv[1:]])


class Ledger:
    def __init__(self):
        self.selected = None
        self.outcomes = {}
        self.finished = set()
        self.incomplete = False
        self.report_bytes = None
        self.report_reason = "session_incomplete"

    def collect(self, nodeids):
        try:
            selected = set(nodeids)
            if len(selected) != len(nodeids) or self.selected not in (None, selected):
                self.incomplete = True
            self.selected = selected
        except Exception:
            self.incomplete = True

    def pytest_collection_finish(self, session):
        try:
            self.collect([item.nodeid for item in session.items])
        except Exception:
            self.incomplete = True

    @hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):
        try:
            self.collect(ids)
        except Exception:
            self.incomplete = True

    def pytest_collectreport(self, report):
        try:
            if report.failed:
                self.incomplete = True
        except Exception:
            self.incomplete = True

    def pytest_runtest_logreport(self, report):
        try:
            nodeid = report.nodeid
            if nodeid in self.finished or report.outcome == "rerun":
                self.incomplete = True
            if report.failed:
                self.outcomes[nodeid] = "failed"
            elif report.skipped:
                self.outcomes.setdefault(nodeid, "skipped")
            elif report.when == "call" and report.passed:
                self.outcomes.setdefault(nodeid, "passed")
            if report.when == "teardown":
                self.finished.add(nodeid)
        except Exception:
            self.incomplete = True

    @hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        try:
            if self.incomplete:
                self.report_reason = "session_incomplete"
                return
            if session.config.option.collectonly:
                self.report_reason = "collection_only"
                return
            if self.selected is None:
                self.report_reason = "no_collected_test_set"
                return
            if self.selected != self.finished or self.selected != self.outcomes.keys():
                self.report_reason = "tests_unfinished"
                return
            if exitstatus not in (0, 1, 5):
                self.report_reason = "unsupported_exit_status"
                return

            outcomes = {}
            for nodeid, outcome in self.outcomes.items():
                path, separator, suffix = nodeid.partition("::")
                relative = os.path.relpath(session.config.rootpath / path, repo)
                identity = relative + separator + suffix
                if relative == ".." or relative.startswith("../") or identity in outcomes:
                    self.report_reason = "invalid_test_identity"
                    return
                outcomes[identity] = outcome
            encoded = json.dumps({"exit_code": int(exitstatus), "outcomes": outcomes},
                                 separators=(",", ":")).encode()
            if len(encoded) > report_limit:
                self.report_reason = "report_too_large"
            else:
                self.report_bytes = encoded
                self.report_reason = None
        except Exception:
            self.report_reason = "session_incomplete"

    def write_report(self):
        try:
            if self.report_reason is not None:
                write_incomplete(self.report_reason)
            elif not isinstance(self.report_bytes, bytes) or not write_bytes(self.report_bytes):
                write_incomplete("report_write_failed")
        except Exception:
            write_incomplete("session_incomplete")


def run_with_ledger(args=None):
    ledger = Ledger()
    try:
        return main(args, plugins=[ledger])
    except BaseException:
        ledger.report_bytes = None
        ledger.report_reason = "session_incomplete"
        raise
    finally:
        ledger.write_report()


if console_script:
    import runpy

    pytest.console_main = lambda: run_with_ledger()
    runpy.run_path(console_script, run_name="__main__")
else:
    sys.exit(run_with_ledger(sys.argv[1:]))
'''

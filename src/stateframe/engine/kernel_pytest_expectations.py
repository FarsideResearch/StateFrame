"""One optional, execution-scoped prediction on a native run_tests call."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path

from stateframe.outer_checks import _pytest_ledger_command, _read_required_test_report


_OUTCOMES = ("passed", "failed", "skipped")
_MAX_NODE = 1024
PREDICTION_SCHEMA = {
    "type": "object", "description": "Optional prediction for one exact pytest node in this check.",
    "properties": {
        "nodeid": {"type": "string", "maxLength": _MAX_NODE,
                   "description": "Full repository-relative node ID, including any parameter ID."},
        "outcome": {"type": "string", "enum": list(_OUTCOMES)},
    },
    "required": ["nodeid", "outcome"], "additionalProperties": False,
}


@dataclass
class Prediction:
    nodeid: str
    expected: str
    execution_id: str | None = None
    mismatch: dict[str, str] | None = None

    @classmethod
    def from_argument(cls, value):
        if not isinstance(value, dict) or value.keys() != {"nodeid", "outcome"}:
            return None
        node = value["nodeid"]
        if (not isinstance(node, str) or not 0 < len(node) <= _MAX_NODE
                or not node.isprintable() or value["outcome"] not in _OUTCOMES):
            return None
        return cls(node, value["outcome"])

    @contextmanager
    def active(self):
        token = ACTIVE.set(self)
        try:
            yield
        finally:
            ACTIVE.reset(token)

    def bind(self, execution_id, report):
        """Bind only when a complete report names this exact executed node."""
        if (self.execution_id is not None or not isinstance(execution_id, str)
                or not execution_id or not isinstance(report, dict)
                or report.get("complete") is not True):
            return False
        outcomes = report.get("outcomes")
        if not isinstance(outcomes, dict) or outcomes.get(self.nodeid) not in _OUTCOMES:
            return False
        self.execution_id = execution_id
        return True

    def observe(self, execution_id, report):
        if not isinstance(report, dict) or report.get("complete") is not True:
            return
        if self.execution_id is None and not self.bind(execution_id, report):
            return
        if self.execution_id != execution_id:
            return
        outcomes = report.get("outcomes")
        if not isinstance(outcomes, dict):
            return
        actual = outcomes.get(self.nodeid)
        if actual is not None and actual != self.expected:
            self.mismatch = {"execution_id": execution_id, "nodeid": self.nodeid,
                             "expected": self.expected, "observed": actual}


ACTIVE: ContextVar[Prediction | None] = ContextVar("native_pytest_prediction", default=None)


# Both released entry points supply a plugin to pytest.main. Let the profiling
# driver enter first, then run the ledger driver with both plugins on the same
# session. Restore pytest.main before invoking project tests or their plugins.
_PROFILE_LEDGER_BRIDGE = """
import sys
try:
    import pytest
except Exception:
    pass  # The original driver owns the import-failure fallback.
else:
    _native_main = pytest.main
    def _with_ledger(args, plugins):
        profile_plugins = plugins
        def _run_session(args, plugins):
            pytest.main = _native_main
            return _native_main(args, plugins=[*profile_plugins, *plugins])
        pytest.main = _run_session
        sys.argv[1:] = args
        try:
            exec(_ledger_source, {"__name__": "__main__"})
        except SystemExit as stopped:
            return stopped.code
        finally:
            pytest.main = _native_main
    pytest.main = _with_ledger
"""


def _reported_command(command, relative, env, profiled):
    argv = list(command)
    if not profiled:
        return _pytest_ledger_command(argv, Path("."), relative, env)
    index = argv.index("-c")
    # profiled_pytest_command preserves the original interpreter flags and
    # inserts a single "pytest" marker between its Python code and pytest args.
    native = [*argv[:index], "-m", "pytest", *argv[index + 3:]]
    wrapped = _pytest_ledger_command(native, Path("."), relative, env)
    if wrapped is None:
        return None
    argv[index + 1] = (f"_ledger_source = {wrapped[index + 1]!r}\n" +
                       _PROFILE_LEDGER_BRIDGE + argv[index + 1])
    return argv if len(argv) <= 128 and sum(map(len, argv)) <= 64 * 1024 else None


def run_reported(command, *, root, env, execution_id, profiled, run):
    """Run the existing check once. Only the sandboxed child writes the report."""
    # Replay compares argv. Use an execution-derived relative filename and reserve it
    # exclusively; an existing file (including a symlink) can never supply evidence.
    name = ".stateframe-pytest-" + hashlib.sha256(execution_id.encode()).hexdigest()
    relative = Path(".test-tmp") / name
    path = root / relative
    with ExitStack() as cleanup:
        try:
            path.parent.mkdir(exist_ok=True)
            directory = os.open(path.parent, os.O_DIRECTORY | os.O_NOFOLLOW)
            cleanup.callback(os.close, directory)
            fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=directory)
            os.close(fd)
        except OSError:
            return run(command), None

        def remove_report():
            # A test can move or remove its scratch directory. Never follow a
            # changed parent on cleanup or let missing telemetry fail the check.
            try:
                os.unlink(name, dir_fd=directory)
            except FileNotFoundError:
                pass
            except OSError as error:
                logging.getLogger(__name__).warning("pytest report cleanup failed: %s", type(error).__name__)

        cleanup.callback(remove_report)
        argv = _reported_command(command, relative, env, profiled)
        result = run(argv if argv is not None else command)
        report = None
        if argv is not None and getattr(result, "execution_stop", None) is None:
            try:
                report, _ = _read_required_test_report(path, result.returncode)
            except OSError:
                pass  # An absent or unreadable report has no prediction outcome.
        return result, report


def record(runtime, mismatch):
    runtime.journal.emit("pytest_expectation_mismatch", producer="kernel_pytest_expectations",
                         dispatch=False, **mismatch)
    return "[pytest expectation mismatch] " + json.dumps(mismatch, ensure_ascii=True, separators=(",", ":"))

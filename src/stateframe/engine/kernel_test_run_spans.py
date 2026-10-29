"""Opt-in monotonic observations for public pytest executions.

The profile is journal evidence only. Its child plugin observes pytest hooks and
never changes selection, outcomes, output streams, or the host's test policy.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
import time
from collections.abc import Mapping, Sequence
from typing import Iterator
import uuid


MODE_ENV = "STATEFRAME_TEST_RUN_SPANS"
PROFILE_SCHEMA = "stateframe-test-run-profile/1"
CHILD_PROFILE_SCHEMA = "stateframe-pytest-child-profile/1"
PROFILE_EVENT = "test_run_profile"
MISMATCH_EVENT = "test_run_spans_shadow_mismatch"
_CURRENT: ContextVar["TestRunProfile | None"] = ContextVar("test_run_profile", default=None)
_PARENT: ContextVar["TestRunProfile | None"] = ContextVar("test_run_profile_parent", default=None)
_MAX_SPANS = 128
_MAX_MARKERS = 128
_MAX_CHILD_RUNS = 8
_MAX_REPORT_BYTES = 16 * 1024


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Read a bounded mode; invalid telemetry settings fail open to OFF."""
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    return value if value in {"off", "shadow", "on"} else "off"


def parse_child_report(payload: object) -> dict[str, object] | None:
    """Validate and reduce the bounded child report before it reaches a journal."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        return None
    raw = bytes(payload)
    if not raw or len(raw) > _MAX_REPORT_BYTES:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, TypeError):
        return None
    if not isinstance(value, Mapping) or value.get("schema") != CHILD_PROFILE_SCHEMA:
        return None

    result: dict[str, object] = {"schema": CHILD_PROFILE_SCHEMA}
    execution_id = value.get("execution_id")
    if isinstance(execution_id, str) and 1 <= len(execution_id) <= 128:
        result["execution_id"] = execution_id
    for key in (
        "child_ready_ns", "pytest_session_start_ns", "collection_start_ns",
        "collection_finish_ns", "session_finish_ns", "child_finished_ns",
        "child_wall_ns", "child_cpu_ns", "pytest_exit_code", "selected_count",
        "finished_count", "deselected_count", "selected_nodeids_sha256",
        "outcomes_sha256",
    ):
        item = value.get(key)
        if type(item) is int and item >= 0:
            result[key] = item
        elif key.endswith("_sha256") and isinstance(item, str) and len(item) == 64 \
                and all(char in "0123456789abcdef" for char in item):
            result[key] = item

    counts = value.get("counts")
    allowed_counts = {"collected", "passed", "failed", "errored", "skipped", "xfailed",
                      "xpassed", "deselected"}
    if isinstance(counts, Mapping):
        result["counts"] = {
            key: item for key, item in counts.items()
            if key in allowed_counts and type(item) is int and item >= 0
        }
    phase_ns = value.get("phase_ns")
    if isinstance(phase_ns, Mapping):
        result["phase_ns"] = {
            key: item for key, item in phase_ns.items()
            if key in {"setup", "call", "teardown"} and type(item) is int and item >= 0
        }
    return result


class TestRunProfile:
    """One execution's monotonic span set and optional shadow comparisons."""

    def __init__(self, profile_mode: str, *, execution_id: str | None = None,
                 runtime: object | None = None) -> None:
        self.mode = profile_mode
        journal = (getattr(runtime, "journal", None) or
                   getattr(runtime, "audit_journal", None)) if runtime is not None else None
        if journal is None and runtime is not None and hasattr(runtime, "_sequence"):
            journal = runtime
        if execution_id:
            self.execution_id = execution_id
        elif journal is not None:
            from kernel_wire_identity import next_wire_id
            owner = getattr(runtime, "owner", None)
            step = getattr(owner, "current_step", None)
            self.execution_id = "check-execution:" + next_wire_id(
                journal, "check-execution", step, "test-run-profile")
        else:
            self.execution_id = f"check-execution:{uuid.uuid4().hex}"
        self.runtime = runtime
        self.started_ns = time.monotonic_ns()
        self.finished_ns: int | None = None
        self.spans: list[dict[str, object]] = []
        self.markers: list[dict[str, object]] = []
        self.child_runs: list[dict[str, object]] = []
        self.shadow_mismatches: list[dict[str, object]] = []
        self.children: list[TestRunProfile] = []
        self.parent: TestRunProfile | None = None
        self.suppress_event = False
        self._open_spans: dict[int, tuple[str, int, str | None]] = {}
        self._next_span_token = 0
        self.start_span("total_host_window")

    def bind_runtime(self, runtime: object | None) -> None:
        if runtime is not None:
            self.runtime = runtime

    def inherit_lock_observations(self) -> None:
        if self.parent is None:
            return
        for row in self.parent.spans:
            if str(row.get("name", "")).startswith("test_lock_") and len(self.spans) < _MAX_SPANS:
                self.spans.append(dict(row))
        for row in self.parent.markers:
            if str(row.get("name", "")).startswith("test_lock_") and len(self.markers) < _MAX_MARKERS:
                self.markers.append(dict(row))

    def mark(self, name: str, at_ns: int | None = None) -> None:
        if len(self.markers) >= _MAX_MARKERS:
            return
        instant = at_ns if type(at_ns) is int and at_ns >= 0 else time.monotonic_ns()
        self.markers.append({"name": str(name)[:64], "monotonic_ns": instant})

    def start_span(self, name: str, *, overlap_group: str | None = None) -> int:
        self._next_span_token += 1
        token = self._next_span_token
        self._open_spans[token] = (str(name)[:64], time.monotonic_ns(), overlap_group)
        return token

    def end_span(self, token: int | None, *, at_ns: int | None = None) -> None:
        if type(token) is not int:
            return
        opened = self._open_spans.pop(token, None)
        if opened is None or len(self.spans) >= _MAX_SPANS:
            return
        name, start_ns, overlap_group = opened
        end_ns = at_ns if type(at_ns) is int and at_ns >= start_ns else time.monotonic_ns()
        row: dict[str, object] = {
            "name": name,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": max(0, end_ns - start_ns),
        }
        if overlap_group:
            row["overlap_group"] = str(overlap_group)[:64]
        self.spans.append(row)

    @contextmanager
    def span(self, name: str, *, overlap_group: str | None = None) -> Iterator[None]:
        token = self.start_span(name, overlap_group=overlap_group)
        try:
            yield
        finally:
            self.end_span(token)

    def record_sandbox_result(self, trace: object, payload: object, *,
                              stdout: object, stderr: object,
                              returncode: object) -> None:
        if len(self.child_runs) >= _MAX_CHILD_RUNS:
            return
        child = parse_child_report(payload)
        if child is not None and child.get("execution_id") != self.execution_id:
            self.mark("child_profile_execution_id_mismatch")
            child = None
        trace_map = trace if isinstance(trace, Mapping) else {}
        prefix = f"attempt_{len(self.child_runs) + 1}_"
        for start_name, end_name, span_name, overlap in (
            ("sandbox_spawn_start_ns", "sandbox_spawn_complete_ns", "sandbox_spawn", None),
            ("sandbox_spawn_complete_ns", "child_exit_observed_ns", "child_lifetime", "child_io"),
            ("sandbox_spawn_complete_ns", "stdout_eof_ns", "stdout_drain", "child_io"),
            ("sandbox_spawn_complete_ns", "stderr_eof_ns", "stderr_drain", "child_io"),
        ):
            start_ns, end_ns = trace_map.get(start_name), trace_map.get(end_name)
            if type(start_ns) is int and type(end_ns) is int and end_ns >= start_ns \
                    and len(self.spans) < _MAX_SPANS:
                row: dict[str, object] = {
                    "name": prefix + span_name,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_ns": end_ns - start_ns,
                }
                if overlap:
                    row["overlap_group"] = overlap
                self.spans.append(row)
        self.mark(prefix + "child_exit_observed", trace_map.get("child_exit_observed_ns"))
        for key in ("stdout_eof_ns", "stderr_eof_ns"):
            self.mark(prefix + key, trace_map.get(key))

        normalized_returncode = returncode if type(returncode) is int else None
        self.child_runs.append({
            "attempt": len(self.child_runs) + 1,
            "returncode": normalized_returncode,
            "stdout_chars": len(stdout) if isinstance(stdout, str) else 0,
            "stderr_chars": len(stderr) if isinstance(stderr, str) else 0,
            "report": child,
        })
        if self.mode == "shadow":
            self._compare_shadow(normalized_returncode, stdout, stderr, child)

    def _compare_shadow(self, returncode: int | None, stdout: object, stderr: object,
                        child: Mapping[str, object] | None) -> None:
        def mismatch(field: str, expected: object, observed: object) -> None:
            if len(self.shadow_mismatches) < 16:
                self.shadow_mismatches.append({
                    "field": field[:64], "expected": expected, "observed": observed,
                })

        if child is None:
            mismatch("child_report", "valid_profile", "missing_or_invalid")
            return
        child_exit = child.get("pytest_exit_code")
        if type(returncode) is int and type(child_exit) is int and returncode != child_exit:
            mismatch("returncode", returncode, child_exit)

        output = "\n".join(value for value in (stdout, stderr) if isinstance(value, str))
        try:
            from kernel_vacuity_sentinel import structured_pytest_counts
            host_counts = structured_pytest_counts(output)
        except Exception:  # optional comparison must never affect the check result
            host_counts = None
        child_counts = child.get("counts")
        if host_counts is not None and isinstance(child_counts, Mapping):
            normalized = {
                "collected": child_counts.get("collected", 0),
                "passed": child_counts.get("passed", 0),
                "failed": child_counts.get("failed", 0),
                "errored": child_counts.get("errored", 0),
                "skipped": child_counts.get("skipped", 0),
                "xfailed": child_counts.get("xfailed", 0),
                "xpassed": child_counts.get("xpassed", 0),
                "deselected": child_counts.get("deselected", 0),
            }
            if dict(host_counts) != normalized:
                mismatch("outcome_counts", dict(host_counts), normalized)

    def _event(self) -> dict[str, object]:
        finished_ns = time.monotonic_ns()
        self.finished_ns = finished_ns
        for token in tuple(self._open_spans):
            self.end_span(token, at_ns=finished_ns)
        rows = sorted(self.spans, key=lambda item: (int(item["start_ns"]), str(item["name"])))
        return {
            "profile_schema": PROFILE_SCHEMA,
            "execution_id": self.execution_id,
            "profile_mode": self.mode,
            "monotonic_started_ns": self.started_ns,
            "monotonic_finished_ns": finished_ns,
            "total_host_window_ns": max(0, finished_ns - self.started_ns),
            "spans": rows,
            "markers": self.markers,
            "child_runs": self.child_runs,
            "shadow_mismatches": self.shadow_mismatches,
            "producer": "kernel_test_run_spans",
        }

    def finalize(self, runtime: object | None = None) -> None:
        if self.finished_ns is not None:
            return
        self.bind_runtime(runtime)
        try:
            event = self._event()
        except Exception:  # profiling must never replace a completed check
            return
        if self.suppress_event:
            for child in self.children:
                try:
                    child.inherit_lock_observations()
                    child.finalize(child.runtime)
                except Exception:  # each child profile is advisory
                    continue
            return
        journal = getattr(self.runtime, "journal", None)
        if journal is None and callable(getattr(self.runtime, "write", None)):
            journal = self.runtime
        emit = getattr(journal, "emit", None)
        write = getattr(journal, "write", None)
        publish = emit if callable(emit) else write if callable(write) else None
        if publish is None:
            return
        try:
            publish(PROFILE_EVENT, **event)
        except Exception:  # telemetry must never replace the test result
            return
        for mismatch in self.shadow_mismatches:
            try:
                publish(MISMATCH_EVENT, execution_id=self.execution_id,
                        field=mismatch["field"], expected=mismatch["expected"],
                        observed=mismatch["observed"], producer="kernel_test_run_spans")
            except Exception:  # each diagnostic is advisory and bounded
                break


def current_profile() -> TestRunProfile | None:
    return _CURRENT.get()


def _loaded_runtime() -> object | None:
    """Return an already-active runtime without importing or initializing one."""
    import sys
    module = sys.modules.get("kernel_pilot")
    active = getattr(module, "ACTIVE_RUNTIME", None)
    getter = getattr(active, "get", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None


@contextmanager
def profile_scope(*, runtime: object | None = None) -> Iterator[TestRunProfile | None]:
    """Create one profile per call tree and publish it after the outer boundary."""
    existing = _CURRENT.get()
    if existing is not None:
        existing.bind_runtime(runtime)
        yield existing
        return
    selected_mode = mode()
    if selected_mode == "off":
        yield None
        return
    parent = _PARENT.get()
    try:
        profile = TestRunProfile(
            selected_mode,
            runtime=(runtime if runtime is not None else
                     parent.runtime if parent is not None else _loaded_runtime()),
        )
    except Exception:  # fail open if a monotonic observer cannot start
        yield None
        return
    profile.parent = parent
    token = _CURRENT.set(profile)
    try:
        yield profile
    finally:
        _CURRENT.reset(token)
        if parent is not None:
            parent.children.append(profile)
        else:
            try:
                profile.finalize(runtime)
            except Exception:  # telemetry cleanup must not mask the check result
                pass


@contextmanager
def independent_profile_scope() -> Iterator[None]:
    """Let a multi-selector dispatcher create one profile for each child check."""
    parent = _CURRENT.get()
    current_token = _CURRENT.set(None)
    parent_token = _PARENT.set(parent)
    try:
        yield
    finally:
        _PARENT.reset(parent_token)
        _CURRENT.reset(current_token)


_PYTEST_RUNNER_SOURCE = r'''import hashlib as _hashlib
import json as _json
import os as _os
import sys as _sys
import time as _time
try:
    import resource as _resource
    _RUSAGE_CHILDREN = _resource.RUSAGE_CHILDREN
except (ImportError, AttributeError):
    _RUSAGE_CHILDREN = 1

_child_ready_ns = _time.monotonic_ns()
_execution_id = __PROFILE_EXECUTION_ID__
_fd_name = "STATEFRAME_TEST_RUN_SPANS_FD"
try:
    _report_fd = int(_os.environ.pop(_fd_name))
except (KeyError, TypeError, ValueError):
    _report_fd = None

def _usage_ns(who):
    try:
        usage = _resource.getrusage(who)
        return int((usage.ru_utime + usage.ru_stime) * 1000000000)
    except (ImportError, AttributeError, OSError, ValueError):
        return 0

_cpu_start_ns = _time.process_time_ns() + _usage_ns(_RUSAGE_CHILDREN)
_wall_start_ns = _time.monotonic_ns()
_plugin = None
_exit_code = None
_phase_ns = {"setup": 0, "call": 0, "teardown": 0}
_outcomes = {}
_selected_ids = []
_counts = {"collected": 0, "passed": 0, "failed": 0, "errored": 0,
           "skipped": 0, "xfailed": 0, "xpassed": 0, "deselected": 0}
_collection_start_ns = None
_collection_finish_ns = None
_session_start_ns = None
_session_finish_ns = None

try:
    import pytest as _pytest
except Exception:
    # Pytest's normal module import failure is preserved by executing the exact
    # legacy interpreter/pytest argv in this same sandboxed child.
    _runner_args = _sys.argv[1:]
    if _runner_args and _runner_args[0] == "pytest":
        _runner_args = _runner_args[1:]
    _fallback = [*__PROFILE_PYTHON_FLAGS__, "-m", "pytest", *_runner_args]
    _os.execv(_sys.executable, [_sys.executable, *_fallback])

class _ProfilePlugin:
    def pytest_sessionstart(self, session):
        global _session_start_ns
        _session_start_ns = _time.monotonic_ns()
        self.session = session

    def pytest_collectstart(self, collector):
        global _collection_start_ns
        if _collection_start_ns is None:
            _collection_start_ns = _time.monotonic_ns()

    def pytest_collection_finish(self, session):
        global _collection_finish_ns, _selected_ids, _counts
        _collection_finish_ns = _time.monotonic_ns()
        _selected_ids = [str(item.nodeid) for item in session.items]
        _counts["collected"] = len(_selected_ids) + _counts["deselected"]

    def pytest_deselected(self, items):
        _counts["deselected"] += len(items)

    def pytest_runtest_logreport(self, report):
        phase = str(getattr(report, "when", ""))
        if phase in _phase_ns:
            duration = getattr(report, "duration", 0.0)
            try:
                _phase_ns[phase] += max(0, int(float(duration) * 1000000000))
            except (TypeError, ValueError, OverflowError):
                pass
        nodeid = str(getattr(report, "nodeid", ""))
        if not nodeid:
            return
        wasxfail = getattr(report, "wasxfail", None)
        if phase == "setup":
            if getattr(report, "failed", False):
                _outcomes[nodeid] = "ERROR"
            elif getattr(report, "skipped", False):
                _outcomes[nodeid] = "XFAILED" if wasxfail else "SKIPPED"
        elif phase == "call":
            if wasxfail and getattr(report, "passed", False):
                _outcomes[nodeid] = "XPASSED"
            elif getattr(report, "failed", False):
                _outcomes[nodeid] = "FAILED"
            elif getattr(report, "skipped", False):
                _outcomes[nodeid] = "XFAILED" if wasxfail else "SKIPPED"
            elif getattr(report, "passed", False):
                _outcomes[nodeid] = "PASSED"
        elif phase == "teardown" and getattr(report, "failed", False):
            _outcomes[nodeid] = "ERROR"

    def pytest_sessionfinish(self, session, exitstatus):
        global _session_finish_ns, _exit_code
        _session_finish_ns = _time.monotonic_ns()
        try:
            _exit_code = int(exitstatus)
        except (TypeError, ValueError):
            _exit_code = None

def _build_report(exit_code):
    selected = sorted(_selected_ids)
    outcome_rows = sorted((nodeid, outcome) for nodeid, outcome in _outcomes.items())
    counts = dict(_counts)
    for outcome in _outcomes.values():
        key = {"PASSED": "passed", "FAILED": "failed", "ERROR": "errored",
               "SKIPPED": "skipped", "XFAILED": "xfailed", "XPASSED": "xpassed"}.get(outcome)
        if key is not None:
            counts[key] += 1
    try:
        session = getattr(_plugin, "session", None)
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        stats = getattr(reporter, "stats", None)
        if isinstance(stats, dict):
            for key, source in (("passed", "passed"), ("failed", "failed"),
                                ("errored", "error"), ("skipped", "skipped"),
                                ("xfailed", "xfailed"), ("xpassed", "xpassed"),
                                ("deselected", "deselected")):
                rows = stats.get(source, ())
                counts[key] = len(rows) if isinstance(rows, (list, tuple)) else 0
            counts["collected"] = sum(
                counts[key] for key in ("passed", "failed", "errored", "skipped",
                                        "xfailed", "xpassed", "deselected")
            )
    except Exception:  # custom pytest configurations may omit terminal reporting
        pass
    finished_ns = _time.monotonic_ns()
    return {
        "schema": "stateframe-pytest-child-profile/1",
        "execution_id": _execution_id,
        "child_ready_ns": _child_ready_ns,
        "pytest_session_start_ns": _session_start_ns,
        "collection_start_ns": _collection_start_ns,
        "collection_finish_ns": _collection_finish_ns,
        "session_finish_ns": _session_finish_ns,
        "child_finished_ns": finished_ns,
        "child_wall_ns": max(0, finished_ns - _wall_start_ns),
        "child_cpu_ns": max(0, _time.process_time_ns() + _usage_ns(_RUSAGE_CHILDREN) - _cpu_start_ns),
        "pytest_exit_code": exit_code,
        "selected_count": len(selected),
        "finished_count": len(_outcomes),
        "deselected_count": counts["deselected"],
        "selected_nodeids_sha256": _hashlib.sha256("\0".join(selected).encode()).hexdigest(),
        "outcomes_sha256": _hashlib.sha256(
            "\0".join(nodeid + "\0" + outcome for nodeid, outcome in outcome_rows).encode()
        ).hexdigest(),
        "counts": counts,
        "phase_ns": _phase_ns,
    }

try:
    _plugin = _ProfilePlugin()
    _runner_args = _sys.argv[1:]
    if _runner_args and _runner_args[0] == "pytest":
        _runner_args = _runner_args[1:]
    _exit_code = int(_pytest.main(_runner_args, plugins=[_plugin]))
finally:
    if _report_fd is not None:
        try:
            payload = _json.dumps(_build_report(_exit_code), separators=(",", ":")).encode()
            if len(payload) <= 16384:
                offset = 0
                while offset < len(payload):
                    offset += _os.write(_report_fd, payload[offset:])
        except BaseException:
            pass
        try:
            _os.close(_report_fd)
        except OSError:
            pass

_sys.exit(1 if _exit_code is None else _exit_code)
'''


def profiled_pytest_command(command: Sequence[object], execution_id: str) -> tuple[str, ...] | None:
    """Wrap a ``python -m pytest`` argv while forwarding pytest args verbatim."""
    argv = tuple(str(item) for item in command)
    if len(argv) < 4:
        return None
    module_index = next((index for index in range(1, len(argv) - 1)
                         if argv[index] == "-m" and argv[index + 1] == "pytest"), None)
    if module_index is None:
        return None
    python_flags = argv[1:module_index]
    pytest_args = argv[module_index + 2:]
    source = _PYTEST_RUNNER_SOURCE.replace("__PROFILE_EXECUTION_ID__", repr(execution_id))
    source = source.replace("__PROFILE_PYTHON_FLAGS__", repr(python_flags))
    # Keep the canonical runner name visible to host command policies that
    # authorize a pytest invocation from its argv, without passing the marker
    # through to pytest's own target selectors.
    return (argv[0], *python_flags, "-c", source, "pytest", *pytest_args)


__all__ = [
    "CHILD_PROFILE_SCHEMA", "MISMATCH_EVENT", "MODE_ENV", "PROFILE_EVENT",
    "PROFILE_SCHEMA", "TestRunProfile", "current_profile", "mode",
    "independent_profile_scope", "parse_child_report", "profile_scope",
    "profiled_pytest_command",
]

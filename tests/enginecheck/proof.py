"""Suite-level and red-proof integrity checks for enginecheck runs.

A suite only passes when every required node id was collected, executed and passed. A red proof only counts when the
named test fails through the assertion it was written for, not through an import, fixture or setup error. Outcomes
come from pytest's junit XML, never from human-readable output.
"""
from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_RUN_TIMEOUT_SECONDS = 900
_TAIL_CHARS = 2000
# Failures of these types mean the test never reached its own assertion, so they never count as a red proof.
_HARNESS_EXCEPTIONS = frozenset({"ImportError", "ModuleNotFoundError", "NameError", "SyntaxError", "IndentationError"})
_TRACEBACK_PREFIX = re.compile(r"^E\s+")


@dataclass(frozen=True)
class SuiteReport:
    """Outcome of one suite run. A non-strict XPASS (an xfail test that passes) is reported as passed, as pytest does."""

    required: tuple[str, ...]
    collected: tuple[str, ...]
    executed: tuple[str, ...]
    passed: tuple[str, ...]
    failed: dict[str, str]
    skipped: dict[str, str]
    errors: dict[str, str]
    exit_code: int

    @property
    def resolved_required(self) -> frozenset[str]:
        """Required ids with bare ids expanded to every collected parametrization."""
        return frozenset(_resolve(self.required, self.collected))

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(sorted(self.resolved_required - set(self.collected)))

    @property
    def ok(self) -> bool:
        required = self.resolved_required
        return (
            self.exit_code == 0
            and bool(required)
            and not (self.failed or self.errors or self.skipped)
            and required == set(self.collected) == set(self.executed) == set(self.passed)
        )

    def summary(self) -> str:
        counts = (
            f"required={len(self.resolved_required)} collected={len(self.collected)} "
            f"executed={len(self.executed)} passed={len(self.passed)}"
        )
        if self.ok:
            return f"SUITE OK {counts}"
        extra = sorted(set(self.collected) - self.resolved_required)
        parts = [
            f"missing=[{_join(self.missing)}]",
            f"failed=[{_join(self.failed)}]",
            f"errors=[{_join(self.errors)}]",
            f"skipped=[{_join(self.skipped)}]",
        ]
        if extra:
            parts.append(f"extra=[{_join(extra)}]")
        if self.exit_code != 0:
            parts.append(f"exit={self.exit_code}")
        return f"SUITE NOT_OK {counts} {' '.join(parts)}"


@dataclass(frozen=True)
class RedProof:
    node: str
    status: str  # passed | failed | errored | skipped | not_collected | unknown
    message: str
    intended: bool


def run_nodes(
    node_ids: Sequence[str],
    *,
    cwd: Path,
    extra_args: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> SuiteReport:
    """Run exactly these node ids in cwd. Bare ids match every parametrization. `env` merges over os.environ."""
    required = tuple(_normalize(node) for node in node_ids)
    if not required or any(not node or node.startswith("-") for node in required):
        raise ValueError("run_nodes needs at least one node id and no option-like entries")
    with tempfile.TemporaryDirectory(prefix="enginecheck-junit-") as tmp:
        junit = Path(tmp) / "junit.xml"
        # --rootdir pins junit file paths to cwd even when cwd sits inside another checkout. xunit1 keeps the `file`
        # attribute, which the default xunit2 drops.
        command = [
            sys.executable, "-m", "pytest", "-p", "no:cacheprovider", f"--rootdir={Path(cwd).resolve()}",
            "-o", "junit_family=xunit1", f"--junitxml={junit}", "-q", *required, *extra_args,
        ]
        # The child leads its own session so a timeout can kill everything it spawned, not just pytest.
        proc = subprocess.Popen(
            command, cwd=cwd, env=_child_env(env), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=_RUN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            return _empty(required, 124, {"<timeout>": f"pytest exceeded {_RUN_TIMEOUT_SECONDS}s"})
        output = (stdout + stderr)[-_TAIL_CHARS:].strip()
        try:
            root = ET.parse(junit).getroot()
        except (FileNotFoundError, ET.ParseError):
            return _empty(required, proc.returncode or 1, {"<pytest>": output or "no junit report written"})
        return _parse(root, required, proc.returncode, output)


def red_proof(
    node_id: str,
    *,
    cwd: Path,
    expect: str,
    env: Mapping[str, str] | None = None,
) -> RedProof:
    """Run one test. Intended only if it was collected, executed and failed with `expect` in the failure message."""
    node = _normalize(node_id)
    if not node or node.startswith("-") or not expect:
        raise ValueError("red_proof needs a node id and a non-empty expect")
    status, message = _classify(node, run_nodes([node], cwd=cwd, env=env))
    exc_type = _exception_type(message)
    # An exception-class expect must equal the failure's type; any other expect is a substring of the message.
    matched = exc_type == expect if _is_exception_name(expect) else expect in message
    intended = status == "failed" and matched and exc_type not in _HARNESS_EXCEPTIONS
    return RedProof(node=node, status=status, message=message, intended=intended)


def _resolve(required: Sequence[str], collected: Sequence[str]) -> set[str]:
    """Expand bare ids to matching parametrizations; an unmatched id stays itself so it reports as missing."""
    resolved: set[str] = set()
    for node in required:
        if "[" in node:
            resolved.add(node)
            continue
        # A class id (or a bracketless function id) covers its methods and parametrizations.
        matches = {case for case in collected if case == node or case.startswith((node + "[", node + "::"))}
        resolved |= matches or {node}
    return resolved


def _normalize(node: str) -> str:
    cleaned = node.strip()
    return cleaned[2:] if cleaned.startswith("./") else cleaned


def _child_env(env: Mapping[str, str] | None) -> dict[str, str]:
    merged = dict(os.environ)
    merged.pop("PYTEST_CURRENT_TEST", None)  # the child must not look like it runs inside another test
    if env is not None:
        merged.update(env)
    return merged


def _empty(required: tuple[str, ...], exit_code: int, errors: dict[str, str]) -> SuiteReport:
    return SuiteReport(required, (), (), (), {}, {}, errors, exit_code)


def _parse(root: ET.Element, required: tuple[str, ...], exit_code: int, output: str) -> SuiteReport:
    collected: list[str] = []
    executed: list[str] = []
    passed: list[str] = []
    failed: dict[str, str] = {}
    skipped: dict[str, str] = {}
    errors: dict[str, str] = {}
    for case in root.iter("testcase"):
        outcomes = [child for child in case if child.tag in ("failure", "error", "skipped")]
        if case.get("classname", "") == "":
            # Collection-level outcome (module failed to import or was skipped): no class or test name, only the module.
            key = case.get("file") or case.get("name", "").replace(".", "/") + ".py"
            for child in outcomes:
                if child.tag == "error":
                    errors[key] = _error_detail(child)
                elif child.tag == "skipped":
                    skipped[key] = child.get("message") or ""
            continue
        node = _node_id(case)
        collected.append(node)
        if not outcomes:
            passed.append(node)
        for child in outcomes:
            if child.tag == "failure":
                failed[node] = child.get("message") or ""
            elif child.tag == "error":
                errors[node] = _error_detail(child)
            else:
                skipped[node] = child.get("message") or ""
        if not any(child.tag in ("error", "skipped") for child in outcomes):
            executed.append(node)
    if exit_code != 0 and not (failed or errors or skipped):
        errors["<pytest>"] = output or f"pytest exited {exit_code} without a test outcome"
    return SuiteReport(
        required=required,
        collected=tuple(collected),
        executed=tuple(executed),
        passed=tuple(passed),
        failed=failed,
        skipped=skipped,
        errors=errors,
        exit_code=exit_code,
    )


def _node_id(case: ET.Element) -> str:
    """Rebuild pytest's path::Class::name[param] id from a junit testcase (classname is the dotted module path)."""
    file_attr = case.get("file", "")
    name = case.get("name", "")
    classname = case.get("classname", "")
    module = file_attr[:-3].replace("/", ".") if file_attr.endswith(".py") else file_attr
    if classname == module:
        return f"{file_attr}::{name}"
    if classname.startswith(module + "."):
        return f"{file_attr}::{classname[len(module) + 1:].replace('.', '::')}::{name}"
    # Unexpected mapping: keep the raw classname so the id cannot match a required id by accident.
    return f"{file_attr}::{classname}::{name}"


def _error_detail(child: ET.Element) -> str:
    """The exception line of a traceback (last non-empty 'E' line), falling back to the message attribute."""
    lines = [line.strip() for line in (child.text or "").splitlines() if line.strip()]
    if lines:
        return _TRACEBACK_PREFIX.sub("", lines[-1])
    return child.get("message") or ""


def _classify(node: str, report: SuiteReport) -> tuple[str, str]:
    if report.errors:
        return "errored", report.errors.get(node) or next(iter(report.errors.values()))
    if node in report.skipped:
        return "skipped", report.skipped[node]
    if node not in report.collected:
        return "not_collected", ""
    if node in report.failed:
        return "failed", report.failed[node]
    if node in report.passed:
        return "passed", ""
    return "unknown", ""


def _exception_type(message: str) -> str:
    """Exception class from a pytest crash line such as 'pkg.NameError: msg'.

    Plain rewritten asserts ('assert 1 == 2') carry no type prefix in junit, but pytest always raises AssertionError
    for them, so that mapping is explicit here.
    """
    head, sep, _ = message.partition(":")
    if sep and head.strip() and " " not in head.strip():
        return head.strip().rsplit(".", 1)[-1]
    return "AssertionError" if message.startswith("assert") else ""


def _is_exception_name(expect: str) -> bool:
    """Exception-class style expects (AssertionError) match the type exactly; other text is matched as a substring."""
    return expect.isidentifier() and expect[:1].isupper()


def _kill_group(proc: subprocess.Popen[str]) -> None:
    """Kill the child's whole process group and reap it, so a hung test cannot outlive the timeout."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.communicate()


def _join(items: Iterable[str]) -> str:
    return ", ".join(sorted(items))

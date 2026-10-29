"""Host-owned fingerprints for detecting weakened pre-existing tests.

The guard is deliberately independent of the model and of the test runner.
It records a structural baseline at environment-receipt time, compares that
baseline with the live workspace, and turns weakening findings into typed
obligations.  New test files and cases are reported separately: they are not
treated as a weakening, but their mutation-style pre-fix failure still needs a
host runner to verify.
"""
from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SWITCH = "STATEFRAME_TEST_INTEGRITY"
SCHEMA = "stateframe-test-integrity/1"
OBLIGATION_KIND = "TEST_WEAKENED"
_MODES = {"off", "on"}
_EXCLUDED_PARTS = frozenset({
    ".git", ".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__",
    ".tox", ".nox", ".venv", "venv", "node_modules", "private",
    "hidden_tests", "near_miss", "gold", "oracle",
})
_SKIP_NAMES = frozenset({
    "skip", "skipif", "xfail", "importorskip", "expectedfailure",
})
_JUSTIFICATION_TYPES = frozenset({
    "intentional_test_change", "test_rewrite", "fixture_contract_change",
    "false_positive",
})


def mode() -> str:
    """Return the strict two-state switch for this mechanism."""
    value = os.environ.get(SWITCH, "off").strip().lower()
    if value not in _MODES:
        raise ValueError(f"{SWITCH} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str).encode()


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _source_digest(source: bytes) -> str:
    return "sha256:" + hashlib.sha256(source).hexdigest()


def _ast_dump(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _ast_shape(value: object) -> object:
    """Produce a JSON-safe tree used for conservative assertion comparison."""
    if isinstance(value, ast.AST):
        return [value.__class__.__name__, [
            [field, _ast_shape(child)]
            for field, child in ast.iter_fields(value)
        ]]
    if isinstance(value, list):
        return [_ast_shape(item) for item in value]
    if isinstance(value, tuple):
        return [_ast_shape(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _qualified_name(node: object) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _qualified_name(node.func)
    if isinstance(node, ast.Subscript):
        return _qualified_name(node.value)
    return ""


def _marker_kind(name: str) -> str | None:
    lowered = name.lower()
    final = lowered.rsplit(".", 1)[-1]
    if final in {"xfail", "expectedfailure"} or final.endswith("xfail"):
        return "xfail"
    if final in {"skip", "skipif", "importorskip"} or final.endswith("skip"):
        return "skip"
    return None


def _marker_record(node: ast.AST) -> dict[str, str] | None:
    name = _qualified_name(node)
    kind = _marker_kind(name)
    if kind is None:
        return None
    return {"kind": kind, "expression": _ast_dump(node)}


def _decorator_records(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[list[dict], list[str]]:
    markers: list[dict] = []
    parametrization: list[str] = []
    for decorator in node.decorator_list:
        marker = _marker_record(decorator)
        if marker is not None:
            markers.append(marker)
        if "parametrize" in _qualified_name(decorator).lower():
            parametrization.append(_ast_dump(decorator))
    return markers, parametrization


def _call_markers(node: ast.AST) -> list[dict]:
    markers: list[dict] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        marker = _marker_record(child)
        if marker is not None:
            markers.append(marker)
    return markers


def _assertion_record(node: ast.Assert) -> dict[str, object]:
    stronger_terms: list[str] = []
    if isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.And):
        stronger_terms = [_ast_dump(value) for value in node.test.values]
    return {
        "expression": _ast_dump(node.test),
        "shape": _ast_shape(node.test),
        "and_terms": stronger_terms,
    }


def _exception_records(node: ast.AST) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Try):
            for handler in child.handlers:
                exception_type = _ast_dump(handler.type) if handler.type is not None else "bare"
                has_raise = any(isinstance(item, ast.Raise) for item in ast.walk(handler))
                records.append({
                    "type": exception_type,
                    "name": handler.name or "",
                    "swallows": not has_raise,
                })
        elif isinstance(child, ast.Call):
            call_name = _qualified_name(child.func).lower()
            if call_name.endswith(".raises") or call_name.rsplit(".", 1)[-1] == "suppress":
                records.append({
                    "type": call_name,
                    "name": "",
                    "swallows": True,
                })
    return records


class _TestCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.class_names: list[str] = []
        self.tests: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_names.append(node.name)
        for item in node.body:
            self.visit(item)
        self.class_names.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name.startswith("test"):
            prefix = "::".join([*self.class_names, node.name])
            self.tests[prefix] = node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name.startswith("test"):
            prefix = "::".join([*self.class_names, node.name])
            self.tests[prefix] = node


def _is_test_path(path: Path, relative: Path) -> bool:
    name = path.name
    if name == "conftest.py" or name.startswith("test_") or name.endswith("_test.py"):
        return True
    return any(part.lower() in {"test", "tests"} for part in relative.parts[:-1])


def _excluded(relative: Path) -> bool:
    return bool(_EXCLUDED_PARTS.intersection(relative.parts))


def _module_markers(tree: ast.Module) -> list[dict[str, str]]:
    markers: list[dict[str, str]] = []
    for statement in tree.body:
        targets: list[ast.AST] = []
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            names = []
            if isinstance(statement, ast.Assign):
                names = [target.id for target in statement.targets
                         if isinstance(target, ast.Name)]
            elif isinstance(statement.target, ast.Name):
                names = [statement.target.id]
            if "pytestmark" in names:
                targets.append(statement.value)
        for target in targets:
            for node in ast.walk(target):
                marker = _marker_record(node)
                if marker is not None:
                    markers.append(marker)
    return markers


def _fingerprint_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, object]:
    markers, parametrization = _decorator_records(node)
    markers.extend(_call_markers(node))
    assertions = [
        _assertion_record(child)
        for child in ast.walk(node)
        if isinstance(child, ast.Assert)
    ]
    return {
        "assertions": assertions,
        "skip_markers": [item for item in markers if item["kind"] == "skip"],
        "xfail_markers": [item for item in markers if item["kind"] == "xfail"],
        "parametrization": parametrization,
        "exceptions": _exception_records(node),
    }


def _fingerprint_source(relative: str, source: bytes) -> dict[str, object]:
    result: dict[str, object] = {
        "source": _source_digest(source),
        "tests": {},
        "module_markers": [],
    }
    try:
        tree = ast.parse(source.decode("utf-8"), filename=relative)
    except (SyntaxError, UnicodeDecodeError) as error:
        result["parse_error"] = type(error).__name__
        return result
    collector = _TestCollector()
    collector.visit(tree)
    result["tests"] = {
        name: _fingerprint_function(node)
        for name, node in sorted(collector.tests.items())
    }
    result["module_markers"] = _module_markers(tree)
    return result


@dataclass(frozen=True)
class TestSnapshot:
    root: str
    files: dict[str, dict[str, object]]
    fingerprint: str

    @property
    def test_count(self) -> int:
        return sum(len(value.get("tests", {}))
                   for value in self.files.values()
                   if isinstance(value.get("tests", {}), Mapping))

    def to_receipt(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "root": self.root,
            "fingerprint": self.fingerprint,
            "file_count": len(self.files),
            "test_count": self.test_count,
            "files": json.loads(json.dumps(self.files, sort_keys=True, default=str)),
        }

    @classmethod
    def from_receipt(cls, receipt: Mapping[str, object]) -> "TestSnapshot":
        files = receipt.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("test-integrity receipt has no files")
        normalized = {str(path): dict(value) for path, value in files.items()
                      if isinstance(value, Mapping)}
        fingerprint = str(receipt.get("fingerprint") or _digest(normalized))
        return cls(str(receipt.get("root") or ""), normalized, fingerprint)


def fingerprint_test_tree(root: str | Path) -> TestSnapshot:
    root_path = Path(root).resolve()
    files: dict[str, dict[str, object]] = {}
    if not root_path.is_dir():
        return TestSnapshot(str(root_path), files, _digest(files))
    for path in sorted(root_path.rglob("*.py")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            relative_path = path.relative_to(root_path)
        except ValueError:
            continue
        if _excluded(relative_path) or not _is_test_path(path, relative_path):
            continue
        try:
            source = path.read_bytes()
        except OSError:
            continue
        files[relative_path.as_posix()] = _fingerprint_source(
            relative_path.as_posix(), source)
    return TestSnapshot(str(root_path), files, _digest(files))


def _as_snapshot(value: TestSnapshot | str | Path) -> TestSnapshot:
    return value if isinstance(value, TestSnapshot) else fingerprint_test_tree(value)


@dataclass(frozen=True)
class TestIntegrityFinding:
    code: str
    path: str
    test_id: str | None
    detail: str
    before: object = None
    after: object = None

    @property
    def obligation_kind(self) -> str:
        return OBLIGATION_KIND

    @property
    def obligation_id(self) -> str:
        identity = {
            "code": self.code,
            "path": self.path,
            "test_id": self.test_id,
            "detail": self.detail,
            "before": self.before,
            "after": self.after,
            "obligation_kind": self.obligation_kind,
        }
        return "TEST_WEAKENED:" + hashlib.sha256(_canonical(identity)).hexdigest()[:20]

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "path": self.path,
            "test_id": self.test_id,
            "detail": self.detail,
            "before": self.before,
            "after": self.after,
            "obligation_kind": self.obligation_kind,
            "obligation_id": self.obligation_id,
        }


@dataclass(frozen=True)
class TestIntegrityReport:
    baseline_fingerprint: str
    current_fingerprint: str
    findings: tuple[TestIntegrityFinding, ...]
    new_files: tuple[str, ...]
    new_tests: tuple[str, ...]
    obligations: tuple[dict[str, object], ...]

    @property
    def weakened(self) -> bool:
        return bool(self.findings)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "baseline_fingerprint": self.baseline_fingerprint,
            "current_fingerprint": self.current_fingerprint,
            "weakened": self.weakened,
            "findings": [finding.to_dict() for finding in self.findings],
            "new_files": list(self.new_files),
            "new_tests": list(self.new_tests),
            "obligations": [dict(item) for item in self.obligations],
        }


def _finding(code: str, path: str, test_id: str | None, detail: str,
             before: object = None, after: object = None) -> TestIntegrityFinding:
    return TestIntegrityFinding(code, path, test_id, detail, before, after)


def _added_markers(before: Sequence[Mapping[str, object]],
                   after: Sequence[Mapping[str, object]],
                   kind: str) -> bool:
    before_values = Counter(json.dumps(dict(item), sort_keys=True, default=str)
                            for item in before if item.get("kind") == kind)
    after_values = Counter(json.dumps(dict(item), sort_keys=True, default=str)
                           for item in after if item.get("kind") == kind)
    return any(after_values[key] > before_values[key] for key in after_values)


def _assertions_weakened(before: Sequence[Mapping[str, object]],
                         after: Sequence[Mapping[str, object]]) -> tuple[str, str] | None:
    unmatched = list(range(len(after)))
    stronger_used: Counter[tuple[int, str]] = Counter()
    missing = []
    for old in before:
        exact = next((index for index in unmatched
                      if after[index].get("expression") == old.get("expression")), None)
        if exact is not None:
            unmatched.remove(exact)
            continue
        expression = str(old.get("expression", ""))
        stronger = next((index for index in range(len(after))
                         if expression in after[index].get("and_terms", ())
                         and stronger_used[(index, expression)] <
                         list(after[index].get("and_terms", ())).count(expression)), None)
        if stronger is not None:
            stronger_used[(stronger, expression)] += 1
            continue
        missing.append(old.get("expression", "assertion"))
    if not missing:
        return None
    code = "assertion_removed" if len(after) < len(before) else "assertion_relaxed"
    return code, str(missing[0])


def _exceptions_weakened(before: Sequence[Mapping[str, object]],
                         after: Sequence[Mapping[str, object]]) -> str | None:
    before_signatures = {(item.get("type"), item.get("name"), item.get("swallows"))
                         for item in before}
    after_signatures = {(item.get("type"), item.get("name"), item.get("swallows"))
                        for item in after}
    if not after_signatures - before_signatures:
        return None
    if any(item.get("swallows") for item in after
           if (item.get("type"), item.get("name"), item.get("swallows"))
           not in before_signatures):
        return "exception_caught"
    return "exception_handler_changed"


def compare_test_trees(before: TestSnapshot | str | Path,
                       current: TestSnapshot | str | Path) -> TestIntegrityReport:
    baseline = _as_snapshot(before)
    live = _as_snapshot(current)
    findings: list[TestIntegrityFinding] = []
    baseline_paths = set(baseline.files)
    live_paths = set(live.files)
    new_files = tuple(sorted(live_paths - baseline_paths))
    new_tests: list[str] = []
    for path in new_files:
        tests = live.files[path].get("tests", {})
        if isinstance(tests, Mapping):
            new_tests.extend(f"{path}::{name}" for name in sorted(tests))
    for path in sorted(baseline_paths - live_paths):
        findings.append(_finding("test_file_deleted", path, None,
                                 "a pre-existing test file is absent"))

    for path in sorted(baseline_paths & live_paths):
        old_file = baseline.files[path]
        new_file = live.files[path]
        old_tests = old_file.get("tests", {})
        new_file_tests = new_file.get("tests", {})
        old_tests = old_tests if isinstance(old_tests, Mapping) else {}
        new_file_tests = new_file_tests if isinstance(new_file_tests, Mapping) else {}
        if old_file.get("parse_error") is None and new_file.get("parse_error") is not None:
            findings.append(_finding(
                "test_source_unparseable", path, None,
                "a pre-existing test file no longer parses",
                old_file.get("source"), new_file.get("parse_error")))
        old_module = old_file.get("module_markers", ())
        new_module = new_file.get("module_markers", ())
        if isinstance(old_module, Sequence) and isinstance(new_module, Sequence):
            if _added_markers(old_module, new_module, "skip"):
                findings.append(_finding("skip_marker_added", path, None,
                                         "a module-level skip marker was added"))
            if _added_markers(old_module, new_module, "xfail"):
                findings.append(_finding("xfail_marker_added", path, None,
                                         "a module-level xfail marker was added"))
        for name in sorted(set(old_tests) - set(new_file_tests)):
            findings.append(_finding("test_deleted", path, name,
                                     "a pre-existing test case is absent"))
        for name in sorted(set(new_file_tests) - set(old_tests)):
            new_tests.append(f"{path}::{name}")
        for name in sorted(set(old_tests) & set(new_file_tests)):
            old_test = old_tests[name]
            new_test = new_file_tests[name]
            if not isinstance(old_test, Mapping) or not isinstance(new_test, Mapping):
                continue
            old_skip = old_test.get("skip_markers", ())
            new_skip = new_test.get("skip_markers", ())
            old_xfail = old_test.get("xfail_markers", ())
            new_xfail = new_test.get("xfail_markers", ())
            if isinstance(old_skip, Sequence) and isinstance(new_skip, Sequence) \
                    and _added_markers(old_skip, new_skip, "skip"):
                findings.append(_finding("skip_marker_added", path, name,
                                         "a test skip marker or skip call was added"))
            if isinstance(old_xfail, Sequence) and isinstance(new_xfail, Sequence) \
                    and _added_markers(old_xfail, new_xfail, "xfail"):
                findings.append(_finding("xfail_marker_added", path, name,
                                         "a test xfail marker or xfail call was added"))
            old_param = old_test.get("parametrization", ())
            new_param = new_test.get("parametrization", ())
            if isinstance(old_param, Sequence) and isinstance(new_param, Sequence):
                if list(old_param) != list(new_param):
                    code = ("parametrization_weakened"
                            if len(new_param) < len(old_param) or
                            any(old not in new_param for old in old_param)
                            else "parametrization_changed")
                    findings.append(_finding(code, path, name,
                                             "test parametrization changed",
                                             old_param, new_param))
            old_assertions = old_test.get("assertions", ())
            new_assertions = new_test.get("assertions", ())
            if isinstance(old_assertions, Sequence) and isinstance(new_assertions, Sequence):
                weakened = _assertions_weakened(old_assertions, new_assertions)
                if weakened is not None:
                    code, expression = weakened
                    findings.append(_finding(code, path, name,
                                             "an existing assertion was removed or relaxed",
                                             expression, new_assertions))
            old_exceptions = old_test.get("exceptions", ())
            new_exceptions = new_test.get("exceptions", ())
            if isinstance(old_exceptions, Sequence) and isinstance(new_exceptions, Sequence):
                exception_code = _exceptions_weakened(old_exceptions, new_exceptions)
                if exception_code is not None:
                    findings.append(_finding(exception_code, path, name,
                                             "the test now catches or swallows an exception",
                                             old_exceptions, new_exceptions))

    obligations = tuple({
        "id": finding.obligation_id,
        "kind": OBLIGATION_KIND,
        "status": "OPEN",
        "finding": finding.to_dict(),
    } for finding in findings)
    return TestIntegrityReport(
        baseline_fingerprint=baseline.fingerprint,
        current_fingerprint=live.fingerprint,
        findings=tuple(findings),
        new_files=new_files,
        new_tests=tuple(sorted(new_tests)),
        obligations=obligations,
    )


@dataclass(frozen=True)
class MutationReceipt:
    status: str
    paths: tuple[str, ...]
    baseline_failed: bool | None
    current_failed: bool | None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-test-mutation/1",
            "status": self.status,
            "paths": list(self.paths),
            "baseline_failed": self.baseline_failed,
            "current_failed": self.current_failed,
            "reason": self.reason,
        }


def _run_failed(result: object) -> bool:
    if isinstance(result, bool):
        return not result
    if isinstance(result, int):
        return result != 0
    if isinstance(result, Mapping):
        if "exit_code" in result:
            return result.get("exit_code") != 0
        if "returncode" in result:
            return result.get("returncode") != 0
        if "passed" in result:
            return not bool(result.get("passed"))
        status = str(result.get("status", "")).upper()
        return status in {"FAIL", "FAILED", "ERROR", "RED", "VIOLATED"}
    return getattr(result, "returncode", 1) != 0


def require_pre_fix_failure(paths: Sequence[str], *,
                            run_on_snapshot: Callable[[tuple[str, ...]], object] | None,
                            run_current: Callable[[tuple[str, ...]], object] | None = None
                            ) -> MutationReceipt:
    """Require a newly added test to fail when run against the pre-fix state."""
    selected = tuple(str(path) for path in paths)
    if not selected:
        return MutationReceipt("NO_NEW_TESTS", selected, None, None, "NO_NEW_TESTS")
    if not callable(run_on_snapshot):
        return MutationReceipt("LIVE_RUN_REQUIRED", selected, None, None,
                               "PREFIX_RUNNER_REQUIRED")
    baseline_failed = _run_failed(run_on_snapshot(selected))
    current_failed = None
    if callable(run_current):
        current_failed = _run_failed(run_current(selected))
    if not baseline_failed:
        return MutationReceipt("REFUSED", selected, False, current_failed,
                               "NEW_TEST_DID_NOT_FAIL_ON_PREFIX")
    return MutationReceipt("VERIFIED", selected, True, current_failed)


check_new_tests_fail_on_snapshot = require_pre_fix_failure


def _typed_justification(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    kind = value.get("type")
    reason = value.get("reason")
    author = value.get("author")
    evidence = value.get("evidence")
    return (isinstance(kind, str) and kind in _JUSTIFICATION_TYPES
            and isinstance(reason, str) and bool(reason.strip())
            and isinstance(author, str) and bool(author.strip())
            and isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes))
            and bool(evidence))


class TestIntegrityGuard:
    """Stateful host guard attached to one environment receipt and journal."""

    __test__ = False

    def __init__(self, root: str | Path, *, journal: object = None,
                 baseline: TestSnapshot | None = None,
                 enabled_override: bool | None = None) -> None:
        self.root = Path(root).resolve()
        self.enabled = enabled() if enabled_override is None else bool(enabled_override)
        self.baseline = baseline or fingerprint_test_tree(self.root)
        self.journal = journal
        self.events: list[dict[str, object]] = []
        self._obligations: dict[str, dict[str, object]] = {}
        self._last_report: TestIntegrityReport | None = None
        self._mutation_receipts: dict[str, MutationReceipt] = {}

    def env_receipt(self) -> dict[str, object]:
        receipt = self.baseline.to_receipt()
        receipt.update({"enabled": self.enabled, "switch": SWITCH})
        return receipt

    def bind_journal(self, journal: object) -> None:
        self.journal = journal

    def restore_baseline(self, receipt: Mapping[str, object]) -> None:
        self.baseline = TestSnapshot.from_receipt(receipt)

    def _emit(self, kind: str, **fields: object) -> dict[str, object] | None:
        if not self.enabled:
            return None
        payload = {"producer": "kernel_test_integrity", "fired": True, **fields}
        if self.journal is not None and hasattr(self.journal, "emit"):
            event = self.journal.emit(kind, **payload)
            if isinstance(event, Mapping):
                row = dict(event)
            else:
                row = {"kind": kind, **payload}
        elif isinstance(self.journal, list):
            row = {"kind": kind, **payload}
            self.journal.append(row)
        else:
            row = {"kind": kind, **payload}
        self.events.append(row)
        return row

    def baseline_event(self) -> dict[str, object] | None:
        return self._emit("test_integrity_baseline", snapshot=self.env_receipt(),
                          baseline_fingerprint=self.baseline.fingerprint,
                          test_count=self.baseline.test_count)

    def baseline_restored_event(self) -> dict[str, object] | None:
        return self._emit(
            "test_integrity_baseline_restored",
            baseline_fingerprint=self.baseline.fingerprint,
            test_count=self.baseline.test_count,
        )

    def audit(self) -> TestIntegrityReport:
        if not self.enabled:
            empty = TestIntegrityReport(
                self.baseline.fingerprint, self.baseline.fingerprint, (), (), (), ())
            self._last_report = empty
            return empty
        report = compare_test_trees(self.baseline, fingerprint_test_tree(self.root))
        for obligation in report.obligations:
            self._obligations.setdefault(obligation["id"], dict(obligation))
        active = tuple(item for item in self._obligations.values()
                       if item.get("status") == "OPEN")
        report = TestIntegrityReport(
            report.baseline_fingerprint, report.current_fingerprint, report.findings,
            report.new_files, report.new_tests, active)
        self._last_report = report
        self._emit(
            "test_integrity_checked",
            baseline_fingerprint=report.baseline_fingerprint,
            current_fingerprint=report.current_fingerprint,
            finding_count=len(report.findings),
            open_obligation_count=len(active),
            obligation_ids=[item["id"] for item in active],
            new_files=list(report.new_files),
            new_tests=list(report.new_tests),
        )
        for finding in report.findings:
            self._emit("test_integrity_obligation", obligation=finding.to_dict(),
                       obligation_id=finding.obligation_id, obligation_kind=OBLIGATION_KIND)
        return report

    @property
    def open_obligations(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(item) for item in self._obligations.values()
                     if item.get("status") == "OPEN")

    @property
    def last_report(self) -> TestIntegrityReport | None:
        return self._last_report

    def completion_blocker(self) -> str | None:
        if not self.enabled:
            return None
        if self._last_report is None:
            self.audit()
        if not self.open_obligations:
            return None
        ids = ", ".join(str(item["id"]) for item in self.open_obligations)
        return "TEST_WEAKENED obligation open: " + ids

    def mutation_receipt_for(self, paths: Sequence[str]) -> MutationReceipt | None:
        key = _digest([str(path) for path in paths])
        return self._mutation_receipts.get(key)

    def mutation_blocker(self) -> str | None:
        """Return a blocker while newly discovered tests lack a pre-fix receipt."""
        if not self.enabled:
            return None
        report = self._last_report or self.audit()
        if not report.new_tests:
            return None
        receipt = self.mutation_receipt_for(report.new_tests)
        if receipt is not None and receipt.status == "VERIFIED":
            return None
        return "TEST_MUTATION_UNVERIFIED: " + ", ".join(report.new_tests)

    def close_obligation(self, obligation_id: str, *, justification: object,
                         dispute: object,
                         host_acceptor: Callable[[object], bool] | None = None
                         ) -> dict[str, object]:
        obligation = self._obligations.get(str(obligation_id))
        if obligation is None:
            return {"status": "REFUSED", "reason": "UNKNOWN_OBLIGATION",
                    "obligation_id": str(obligation_id)}
        if not _typed_justification(justification):
            return {"status": "REFUSED", "reason": "INVALID_TYPED_JUSTIFICATION",
                    "obligation_id": str(obligation_id)}
        if not isinstance(dispute, Mapping):
            return {"status": "OPEN", "reason": "HOST_DISPUTE_REQUIRED",
                    "obligation_id": str(obligation_id)}
        if not callable(host_acceptor):
            return {"status": "OPEN", "reason": "HOST_ACCEPTANCE_REQUIRED",
                    "obligation_id": str(obligation_id)}
        try:
            accepted = bool(host_acceptor(dispute))
        except Exception:  # host adapters must fail closed
            accepted = False
        if not accepted:
            return {"status": "OPEN", "reason": "HOST_DISPUTE_NOT_ACCEPTED",
                    "obligation_id": str(obligation_id)}
        obligation["status"] = "CLOSED"
        obligation["closure"] = {
            "justification": dict(justification),
            "dispute": dict(dispute),
            "authority": "HOST_ACCEPTED_DISPUTE",
        }
        self._emit("test_integrity_obligation_closed", obligation_id=str(obligation_id),
                   justification=dict(justification), dispute=dict(dispute))
        return {"status": "CLOSED", "obligation_id": str(obligation_id),
                "authority": "HOST_ACCEPTED_DISPUTE"}

    def dispute(self, arguments: Mapping[str, object], *,
                host_acceptor: Callable[[object], bool] | None = None) -> dict[str, object]:
        if not isinstance(arguments, Mapping):
            return {"status": "REFUSED", "reason": "INVALID_ARGUMENTS"}
        check_id = str(arguments.get("check_id") or "")
        basis = arguments.get("basis")
        claim = arguments.get("claim")
        if not check_id.startswith("TEST_WEAKENED:"):
            return {"status": "REFUSED", "reason": "NOT_TEST_INTEGRITY_OBLIGATION"}
        if isinstance(basis, Mapping):
            justification = basis
        elif isinstance(basis, Sequence) and not isinstance(basis, (str, bytes)):
            fields: dict[str, object] = {"evidence": list(basis)}
            for item in basis:
                if isinstance(item, str) and "=" in item:
                    key, value = item.split("=", 1)
                    if key in {"type", "reason", "author"}:
                        fields[key] = value
            justification = fields
        else:
            justification = basis
        return self.close_obligation(
            check_id, justification=justification,
            dispute={"check_id": check_id, "claim": claim, "basis": basis},
            host_acceptor=host_acceptor,
        )

    def check_new_tests(self, paths: Sequence[str], *,
                        run_on_snapshot: Callable[[tuple[str, ...]], object] | None,
                        run_current: Callable[[tuple[str, ...]], object] | None = None
                        ) -> MutationReceipt:
        receipt = require_pre_fix_failure(
            paths, run_on_snapshot=run_on_snapshot, run_current=run_current)
        key = _digest(list(receipt.paths))
        self._mutation_receipts[key] = receipt
        self._emit("test_integrity_mutation_check", mutation=receipt.to_dict(),
                   paths=list(receipt.paths))
        return receipt


__all__ = [
    "MutationReceipt", "TestIntegrityFinding", "TestIntegrityGuard",
    "TestIntegrityReport", "TestSnapshot", "SCHEMA", "SWITCH",
    "check_new_tests_fail_on_snapshot", "compare_test_trees", "enabled",
    "fingerprint_test_tree", "mode", "require_pre_fix_failure",
]

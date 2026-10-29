"""Allowlisted pytest/Vitest runners with one normalized result contract.

The command surface intentionally follows ``react_repair.run_tests`` for pytest.
Vitest is invoked only through the fixed local binary path and never installs
dependencies or executes a task-provided command string.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from task_manifest import RunnerName, TaskManifest, load_task_manifest
from kernel_tool_paths import AmbiguousToolPath, unique_tool_match


RunnerStatus: TypeAlias = Literal[
    "PASS",
    "ASSERTION_FAIL",
    "NO_TESTS",
    "COLLECTION_ERROR",
    "HARNESS_ERROR",
    "INFRA_ERROR",
    "TIMEOUT",
    "UNKNOWN",
]
DbGuard: TypeAlias = Callable[[], bool]
SubprocessRun: TypeAlias = Callable[..., subprocess.CompletedProcess[str]]
MAX_OUTPUT_TAIL = 4_000
VITEST_BINARY = "node_modules/.bin/vitest"


@dataclass(frozen=True, slots=True)
class TestResult:
    """Normalized result consumed by preflight, agent feedback, and grading."""

    runner: RunnerName
    status: RunnerStatus
    exit_code: int | None
    collected: int
    selected: tuple[str, ...]
    failed_nodes: tuple[str, ...]
    primary_failure: str | None
    output_tail: str

    def __post_init__(self) -> None:
        if self.runner not in {"pytest", "vitest", "eslint", "static"}:
            raise ValueError(f"unsupported runner: {self.runner}")
        if self.status not in {
            "PASS",
            "ASSERTION_FAIL",
            "NO_TESTS",
            "COLLECTION_ERROR",
            "HARNESS_ERROR",
            "INFRA_ERROR",
            "TIMEOUT",
            "UNKNOWN",
        }:
            raise ValueError(f"unsupported test status: {self.status}")
        if self.collected < 0:
            raise ValueError("collected cannot be negative")
        if self.exit_code is not None and not isinstance(self.exit_code, int):
            raise ValueError("exit_code must be an integer or null")

    @property
    def passed(self) -> bool:
        """Whether this is an authoritative collected-test pass."""

        return self.status == "PASS" and self.collected > 0

    @property
    def legacy_output(self) -> str:
        """Render the old parser-compatible ``pytest exit N`` surface."""

        if self.exit_code is None:
            if self.status == "TIMEOUT":
                message = f"{self.runner} TIMED OUT."
            elif self.status == "INFRA_ERROR":
                message = f"{self.runner} NOT RUN — {self.primary_failure or 'environment failure.'}"
            else:
                message = f"{self.runner} produced no usable exit code."
            return _join_output(message, self.output_tail)

        meaning = {
            "PASS": "ALL TESTS PASSED.",
            "ASSERTION_FAIL": "TESTS FAILED — the selected assertion or behavior did not pass.",
            "NO_TESTS": "NO TESTS COLLECTED — this is not evidence about the patch.",
            "COLLECTION_ERROR": "TEST COLLECTION FAILED — the suite could not provide a test verdict.",
            "HARNESS_ERROR": "HARNESS ERROR — the runner invocation or configuration was invalid.",
            "INFRA_ERROR": "INFRASTRUCTURE ERROR — the environment could not run the suite.",
            "TIMEOUT": "TESTS TIMED OUT.",
            "UNKNOWN": "RUNNER OUTPUT WAS AMBIGUOUS — no verdict is claimed.",
        }[self.status]
        return _join_output(
            f"{self.runner} exit {self.exit_code} — {meaning}", self.output_tail
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "runner": self.runner,
            "status": self.status,
            "exit_code": self.exit_code,
            "collected": self.collected,
            "selected": list(self.selected),
            "failed_nodes": list(self.failed_nodes),
            "primary_failure": self.primary_failure,
            "output_tail": self.output_tail,
        }


def default_db_guard() -> bool:
    """Standalone-safe default; production callers inject ``react_repair._ensure_db``."""

    return True


def run_pytest(
    root: Path,
    fn: str,
    explicit: str | None = None,
    rel_hint: str | None = None,
    *,
    selectors: Sequence[str] = (),
    requires_db: bool = True,
    db_guard: DbGuard | None = None,
    run: SubprocessRun | None = None,
    timeout: float = 300,
    env: Mapping[str, str] | None = None,
) -> TestResult:
    """Run the current pytest contract and return a typed result.

    ``explicit`` and ``rel_hint`` retain the positional API used by the existing
    React loop.  ``selectors`` is the manifest-native equivalent and may contain
    paths or pytest node ids.
    """

    subprocess_run = run or subprocess.run
    requested = _selector_tuple(selectors) or _selector_tuple(
        (explicit,) if explicit else ()
    )
    gate = _database_gate("pytest", requires_db, db_guard, requested)
    if gate is not None:
        return gate

    targets, kexpr, resolve_error = _resolve_pytest_targets(
        root, fn, requested, subprocess_run
    )
    selected = tuple(requested or targets)
    if resolve_error:
        return _result(
            "pytest",
            "HARNESS_ERROR",
            None,
            selected,
            resolve_error,
        )
    if not targets:
        return _result(
            "pytest",
            "NO_TESTS",
            5,
            selected,
            "no test file in this repository calls that function",
        )

    from kernel_test_python import test_python
    command = [test_python(), "-m", "pytest", "-x", "-q", "--no-header"]
    if kexpr:
        command.extend(["-k", kexpr])
    command.extend(["-p", "no:cacheprovider", *targets])
    process_env = _process_env(env, requires_db)
    try:
        completed = subprocess_run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=process_env,
        )
    except subprocess.TimeoutExpired:
        return _result(
            "pytest", "TIMEOUT", None, selected, f"pytest timed out after {timeout}s"
        )
    except (FileNotFoundError, OSError) as exc:
        return _result(
            "pytest",
            "INFRA_ERROR",
            None,
            selected,
            f"pytest executable unavailable: {type(exc).__name__}",
        )
    except Exception as exc:  # noqa: BLE001 - runner boundary must fail closed
        return _result(
            "pytest",
            "INFRA_ERROR",
            None,
            selected,
            f"pytest subprocess failed: {type(exc).__name__}",
        )

    return parse_pytest_output(
        completed.stdout or "",
        completed.stderr or "",
        completed.returncode,
        selected=selected,
        rel_hint=rel_hint,
    )


def run_vitest(
    root: Path,
    selectors: Sequence[str] = (),
    *,
    requires_db: bool = True,
    db_guard: DbGuard | None = None,
    run: SubprocessRun | None = None,
    timeout: float = 300,
    env: Mapping[str, str] | None = None,
) -> TestResult:
    """Run the pinned local Vitest binary with the JSON reporter enabled."""

    selected = _selector_tuple(selectors)
    gate = _database_gate("vitest", requires_db, db_guard, selected)
    if gate is not None:
        return gate
    command = [VITEST_BINARY, "run", "--reporter=json", *selected]
    process_env = _process_env(env, requires_db)
    subprocess_run = run or subprocess.run
    try:
        completed = subprocess_run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=process_env,
        )
    except subprocess.TimeoutExpired:
        return _result(
            "vitest", "TIMEOUT", None, selected, f"vitest timed out after {timeout}s"
        )
    except (FileNotFoundError, OSError) as exc:
        return _result(
            "vitest",
            "INFRA_ERROR",
            None,
            selected,
            f"node or vitest is unavailable: {type(exc).__name__}",
        )
    except Exception as exc:  # noqa: BLE001 - runner boundary must fail closed
        return _result(
            "vitest",
            "INFRA_ERROR",
            None,
            selected,
            f"vitest subprocess failed: {type(exc).__name__}",
        )

    return parse_vitest_output(
        completed.stdout or "",
        completed.stderr or "",
        completed.returncode,
        selected=selected,
    )


def run_allowlisted(
    root: Path,
    *,
    runner: RunnerName = "pytest",
    fn: str = "",
    explicit: str | None = None,
    selectors: Sequence[str] = (),
    rel_hint: str | None = None,
    requires_db: bool = True,
    db_guard: DbGuard | None = None,
    run: SubprocessRun | None = None,
    timeout: float = 300,
    env: Mapping[str, str] | None = None,
) -> TestResult:
    """Dispatch only to a known runner enum; task data cannot supply a command."""

    if runner == "pytest":
        return run_pytest(
            root,
            fn,
            explicit,
            rel_hint,
            selectors=selectors,
            requires_db=requires_db,
            db_guard=db_guard,
            run=run,
            timeout=timeout,
            env=env,
        )
    if runner == "vitest":
        return run_vitest(
            root,
            selectors,
            requires_db=requires_db,
            db_guard=db_guard,
            run=run,
            timeout=timeout,
            env=env,
        )
    return _result(
        runner,
        "HARNESS_ERROR",
        None,
        _selector_tuple(selectors),
        f"runner {runner!r} is allowlisted but has no executor in slice 1",
    )


def run_task_tests(
    task: TaskManifest | Mapping[str, object],
    root: Path,
    *,
    fn: str = "",
    explicit: str | None = None,
    db_guard: DbGuard | None = None,
    run: SubprocessRun | None = None,
    timeout: float = 300,
    env: Mapping[str, str] | None = None,
) -> TestResult:
    """Run the runner and DB policy declared by one normalized task row."""

    manifest = task if isinstance(task, TaskManifest) else load_task_manifest(task)
    selectors = manifest.public.selectors
    selected_explicit = explicit or (selectors[0] if selectors else None)
    return run_allowlisted(
        root,
        runner=manifest.public.runner,
        fn=fn,
        explicit=selected_explicit,
        selectors=selectors,
        requires_db=manifest.public.requires_db,
        db_guard=db_guard,
        run=run,
        timeout=timeout,
        env=env,
    )


def run_tests(
    root: Path,
    fn: str,
    explicit: str | None,
    rel_hint: str | None = None,
    *,
    runner: RunnerName = "pytest",
    requires_db: bool = True,
    db_guard: DbGuard | None = None,
    run: SubprocessRun | None = None,
    timeout: float = 300,
    env: Mapping[str, str] | None = None,
) -> TestResult:
    """Compatibility name for callers replacing the old local ``run_tests``."""

    return run_allowlisted(
        root,
        runner=runner,
        fn=fn,
        explicit=explicit,
        rel_hint=rel_hint,
        requires_db=requires_db,
        db_guard=db_guard,
        run=run,
        timeout=timeout,
        env=env,
    )


def run_tests_legacy(*args: object, **kwargs: object) -> str:
    """Compatibility wrapper for code that still consumes the old string surface."""

    result = run_tests(
        *cast(tuple[Path, str, str | None], args), **cast(dict[str, object], kwargs)
    )
    return result.legacy_output


def parse_pytest_output(
    stdout: str,
    stderr: str,
    exit_code: int,
    *,
    selected: Sequence[str] = (),
    rel_hint: str | None = None,
) -> TestResult:
    """Normalize pytest without treating exit code alone as a collected pass."""

    text = _strip_ansi(stdout + ("\n" + stderr if stderr else ""))
    counts = _pytest_counts(text)
    collected = counts["collected"]
    failed_nodes = _pytest_failed_nodes(text)
    primary = _primary_failure(text, failed_nodes)
    if rel_hint and primary and rel_hint not in text and failed_nodes:
        primary = f"{primary} (selected target: {rel_hint})"

    if exit_code == 5 or re.search(r"no tests ran", text, re.IGNORECASE):
        status: RunnerStatus = "NO_TESTS"
    elif exit_code == 4:
        status = "HARNESS_ERROR"
    elif _pytest_collection_error(text, collected):
        status = "COLLECTION_ERROR"
    elif exit_code == 0:
        status = "PASS" if collected > 0 else "UNKNOWN"
    elif exit_code == 1:
        status = "ASSERTION_FAIL" if collected > 0 or failed_nodes else "UNKNOWN"
    elif exit_code in {2, 3}:
        status = "HARNESS_ERROR"
    else:
        status = "UNKNOWN"

    if status == "UNKNOWN" and not primary:
        primary = "pytest output did not contain a structural test verdict"
    return TestResult(
        runner="pytest",
        status=status,
        exit_code=exit_code,
        collected=collected,
        selected=_selector_tuple(selected),
        failed_nodes=failed_nodes,
        primary_failure=primary,
        output_tail=_tail(text),
    )


def parse_vitest_output(
    stdout: str,
    stderr: str = "",
    exit_code: int = 0,
    *,
    selected: Sequence[str] = (),
) -> TestResult:
    """Normalize Vitest JSON output, then a strict known-summary fallback.

    Arbitrary pretty output is never treated as a failure or a pass.  The fallback
    accepts only the documented ``Tests ... (N)`` summary grammar; ambiguous output
    remains ``UNKNOWN``.
    """

    text = _strip_ansi(stdout + ("\n" + stderr if stderr else ""))
    selected_tuple = _selector_tuple(selected)
    report = _find_vitest_report(text)
    if report is not None:
        parsed = _parse_vitest_json(report, exit_code, selected_tuple, text)
        if parsed is not None:
            return parsed

    pretty = _parse_vitest_summary(text, exit_code, selected_tuple)
    if pretty is not None:
        return pretty

    classified = _classify_vitest_failure(text, exit_code)
    if classified is not None:
        status, message = classified
        return TestResult(
            runner="vitest",
            status=status,
            exit_code=exit_code,
            collected=0,
            selected=selected_tuple,
            failed_nodes=(),
            primary_failure=message,
            output_tail=_tail(text),
        )
    return TestResult(
        runner="vitest",
        status="UNKNOWN",
        exit_code=exit_code,
        collected=0,
        selected=selected_tuple,
        failed_nodes=(),
        primary_failure="vitest output was not a recognized JSON report or summary",
        output_tail=_tail(text),
    )


def _database_gate(
    runner: RunnerName,
    requires_db: bool,
    db_guard: DbGuard | None,
    selected: Sequence[str] = (),
) -> TestResult | None:
    if not requires_db:
        return None
    guard = db_guard or default_db_guard
    try:
        alive = guard()
    except Exception as exc:  # noqa: BLE001 - infra must not become test failure
        return _result(
            runner,
            "INFRA_ERROR",
            None,
            selected,
            f"database guard failed: {type(exc).__name__}",
        )
    if not alive:
        return _result(
            runner,
            "INFRA_ERROR",
            None,
            selected,
            "database guard reported unavailable; tests were not run",
        )
    return None


def _process_env(env: Mapping[str, str] | None, requires_db: bool) -> dict[str, str]:
    result = dict(os.environ)
    if env:
        result.update(env)
    if requires_db:
        database_name = result.get("REACT_DB_NAME") or (
            f"mission_control_test_react_{uuid.uuid4().hex[:8]}"
        )
        pass  # product: no lab database default
    return result


def _resolve_pytest_targets(
    root: Path,
    fn: str,
    selectors: Sequence[str],
    run: SubprocessRun,
) -> tuple[list[str], str | None, str | None]:
    requested = _selector_tuple(selectors)
    if not requested:
        try:
            found = run(
                ["grep", "-rl", "--include=test_*.py", fn, str(root)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            targets = [
                line for line in (found.stdout or "").splitlines() if line.strip()
            ]
        except Exception:  # noqa: BLE001 - no target is a typed no-oracle outcome
            targets = []
        return targets, None, None

    targets: list[str] = []
    kexpr: str | None = None
    for selector in requested:
        base = selector.split("::", 1)[0].strip("`'\"")
        if (
            "/" not in base
            and "::" not in selector
            and base.startswith("test_")
            and not base.endswith(".py")
        ):
            kexpr = base
            try:
                found = run(
                    ["grep", "-rl", "--include=test_*.py", f"def {base}", str(root)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                matches = [
                    line for line in (found.stdout or "").splitlines() if line.strip()
                ]
            except Exception:  # noqa: BLE001
                matches = []
            if not matches:
                return (
                    [],
                    kexpr,
                    f"no test named `{base}` is defined in this repository",
                )
            try:
                selected = unique_tool_match(root, base, [Path(value) for value in matches])
            except AmbiguousToolPath as error:
                return [], kexpr, str(error)
            targets.append(str(selected))
            continue
        candidate = _safe_selector_path(root, base)
        if candidate is None:
            return [], kexpr, f"test selector escapes sandbox: `{base}`"
        if not candidate.exists():
            matches = list(root.rglob(Path(base).name))
            if len(matches) == 1:
                suffix = selector[len(selector.split("::", 1)[0]) :]
                targets.append(str(matches[0].relative_to(root.resolve())) + suffix)
            elif len(matches) > 1:
                return (
                    [],
                    kexpr,
                    f"ambiguous test path `{base}` ({len(matches)} matches)",
                )
            else:
                return [], kexpr, f"test path does not exist: `{base}`"
        else:
            suffix = selector[len(selector.split("::", 1)[0]) :]
            target_text = str(candidate.relative_to(root.resolve())) + suffix
            targets.append(target_text)
    return targets, kexpr, None


def _safe_selector_path(root: Path, base: str) -> Path | None:
    try:
        root_resolved = root.resolve()
        candidate = Path(base)
        resolved = (
            candidate if candidate.is_absolute() else root / candidate
        ).resolve()
        if not resolved.is_relative_to(root_resolved):
            return None
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None


def _pytest_counts(text: str) -> dict[str, int]:
    collected_match = re.search(r"collected\s+(\d+)\s+items?", text, re.IGNORECASE)
    collected = int(collected_match.group(1)) if collected_match else 0
    summary = text.split("\n")[-30:]
    counts = {
        "passed": 0,
        "failed": 0,
        "error": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    for line in summary:
        if not re.search(
            r"\b(?:passed|failed|error|errors|skipped|xfailed|xpassed)\b", line
        ):
            continue
        for count, label in re.findall(
            r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)",
            line,
            re.IGNORECASE,
        ):
            key = "error" if label.lower().startswith("error") else label.lower()
            counts[key] = counts.get(key, 0) + int(count)
    summary_total = sum(counts.values())
    return {"collected": max(collected, summary_total), **counts}


def _pytest_failed_nodes(text: str) -> tuple[str, ...]:
    nodes: list[str] = []
    for line in text.splitlines():
        match = re.match(r"\s*(?:FAILED|ERROR)\s+([^\s-]+)", line)
        if match:
            nodes.append(match.group(1))
    return tuple(dict.fromkeys(nodes))


def _pytest_collection_error(text: str, collected: int) -> bool:
    if re.search(r"ERROR collecting", text, re.IGNORECASE):
        return True
    if re.search(r"collected\s+0\s+items?", text, re.IGNORECASE) and re.search(
        r"(?:ImportError|ModuleNotFoundError|SyntaxError|conftest|collection)",
        text,
        re.IGNORECASE,
    ):
        return True
    return collected == 0 and bool(
        re.search(r"(?:ImportError|ModuleNotFoundError|SyntaxError)\s*:", text)
    )


def _find_vitest_report(text: str) -> Mapping[str, object] | None:
    decoder = json.JSONDecoder()
    documents: list[object] = []
    stripped = text.strip()
    if stripped:
        try:
            documents.append(json.loads(stripped))
        except json.JSONDecodeError:
            pass
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate or candidate[0] not in "[{":
            continue
        try:
            documents.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            document, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        documents.append(document)
    for document in documents:
        report = _locate_report(document)
        if report is not None:
            return report
    return None


def _locate_report(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        report_keys = {
            "numTotalTests",
            "numPassedTests",
            "numFailedTests",
            "testResults",
            "testFileResults",
        }
        if report_keys.intersection(value):
            return value
        if "assertionResults" in value:
            return {"testResults": [value]}
        for child in value.values():
            found = _locate_report(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        if value and all(isinstance(item, Mapping) for item in value):
            if any(
                "assertionResults" in cast(Mapping[str, object], item) for item in value
            ):
                return {"testResults": value}
        for child in value:
            found = _locate_report(child)
            if found is not None:
                return found
    return None


def _parse_vitest_json(
    report: Mapping[str, object],
    exit_code: int,
    selected: tuple[str, ...],
    text: str,
) -> TestResult | None:
    files = _mapping_list(
        report.get(
            "testResults", report.get("testFileResults", report.get("files", []))
        )
    )
    assertions: list[tuple[Mapping[str, object], str | None]] = []
    suite_errors: list[str] = []
    file_names: list[str] = []
    for file_result in files:
        file_name = _first_text(file_result, "name", "testFilePath", "filepath", "file")
        if file_name:
            file_names.append(file_name)
        nested = _mapping_list(
            file_result.get("assertionResults", file_result.get("testResults", []))
        )
        for assertion in nested:
            assertions.append((assertion, file_name))
        for key in ("message", "failureMessage", "testExecError", "errorMessage"):
            message = file_result.get(key)
            if isinstance(message, str) and message.strip():
                suite_errors.append(message.strip())

    passed = failed = pending = 0
    failed_nodes: list[str] = []
    primary: str | None = None
    for assertion, file_name in assertions:
        status = str(assertion.get("status", "")).lower()
        if status in {"passed", "pass", "success"}:
            passed += 1
        elif status in {"failed", "fail", "error"}:
            failed += 1
            node = _assertion_node(assertion, file_name)
            failed_nodes.append(node)
            if primary is None:
                primary = _assertion_failure(assertion)
        elif status in {"pending", "skipped", "todo", "disabled"}:
            pending += 1

    total_raw = _first_int(report, "numTotalTests", "totalTests")
    total = total_raw if total_raw is not None else len(assertions)
    report_failed = _first_int(report, "numFailedTests") or 0
    report_pending = _first_int(report, "numPendingTests", "numTodoTests") or 0
    failed = max(failed, report_failed)
    pending = max(pending, report_pending)
    if total == 0 and assertions:
        total = len(assertions)
    if not assertions and not files and total_raw is None:
        return None

    if not primary:
        primary = (
            suite_errors[0]
            if suite_errors
            else _primary_failure(text, tuple(failed_nodes))
        )
    if exit_code == 0 and total > 0 and failed == 0:
        status: RunnerStatus = "PASS"
    elif _vitest_collection_from_report(report, suite_errors, assertions):
        status = "COLLECTION_ERROR"
    elif total == 0 and failed == 0:
        status = "NO_TESTS"
    elif failed > 0 or exit_code == 1:
        status = (
            "ASSERTION_FAIL" if total > 0 and not suite_errors else "COLLECTION_ERROR"
        )
    elif exit_code == 5:
        status = "NO_TESTS"
    else:
        status = "UNKNOWN"
    if status == "UNKNOWN" and not primary:
        primary = "Vitest JSON report did not support a definitive verdict"
    actual_selected = selected or tuple(dict.fromkeys(file_names))
    return TestResult(
        runner="vitest",
        status=status,
        exit_code=exit_code,
        collected=max(total, passed + failed + pending),
        selected=actual_selected,
        failed_nodes=tuple(dict.fromkeys(failed_nodes)),
        primary_failure=primary,
        output_tail=_tail(text),
    )


def _vitest_collection_from_report(
    report: Mapping[str, object],
    suite_errors: Sequence[str],
    assertions: Sequence[tuple[Mapping[str, object], str | None]],
) -> bool:
    if suite_errors and not assertions:
        return True
    for key in ("message", "failureMessage", "testExecError", "errorMessage"):
        message = report.get(key)
        if isinstance(message, str) and message.strip() and not assertions:
            return True
    failed_suites = _first_int(report, "numFailedTestSuites") or 0
    return failed_suites > 0 and not assertions


def _parse_vitest_summary(
    text: str,
    exit_code: int,
    selected: tuple[str, ...],
) -> TestResult | None:
    total: int | None = None
    passed = failed = pending = 0
    for line in text.splitlines():
        stripped = line.strip()
        match = re.fullmatch(r"Tests\s+(.+?)\s+\((\d+)\)", stripped)
        if not match:
            continue
        total = int(match.group(2))
        for term in match.group(1).split("|"):
            count_match = re.fullmatch(
                r"\s*(\d+)\s+(passed|failed|skipped|pending|todo)\s*",
                term,
                re.IGNORECASE,
            )
            if not count_match:
                return None
            count = int(count_match.group(1))
            label = count_match.group(2).lower()
            if label == "passed":
                passed += count
            elif label == "failed":
                failed += count
            else:
                pending += count
        break
    if total is None:
        return None
    failed_nodes = _pretty_failed_nodes(text) if failed else ()
    primary = _primary_failure(text, failed_nodes)
    if exit_code == 0 and total > 0 and failed == 0:
        status: RunnerStatus = "PASS"
    elif total == 0:
        status = "NO_TESTS"
    elif failed > 0 or exit_code == 1:
        status = "ASSERTION_FAIL"
    else:
        status = "UNKNOWN"
    return TestResult(
        runner="vitest",
        status=status,
        exit_code=exit_code,
        collected=total,
        selected=selected,
        failed_nodes=failed_nodes,
        primary_failure=primary,
        output_tail=_tail(text),
    )


def _pretty_failed_nodes(text: str) -> tuple[str, ...]:
    nodes: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"[❯×✗✕]\s+(.+)", stripped)
        if match:
            nodes.append(match.group(1).strip())
    return tuple(dict.fromkeys(nodes))


def _classify_vitest_failure(
    text: str, exit_code: int
) -> tuple[RunnerStatus, str] | None:
    lowered = text.lower()
    if any(
        token in lowered for token in ("timed out", "timeout", "exceeded the timeout")
    ):
        return "TIMEOUT", "Vitest reported a timeout"
    if any(
        token in lowered for token in ("command not found", "enoent", "node: not found")
    ):
        return "INFRA_ERROR", "Node/Vitest environment is unavailable"
    if "cannot find module" in lowered:
        if "vitest" in lowered or "node_modules/.bin" in lowered:
            return "INFRA_ERROR", "Node/Vitest environment is unavailable"
        return "COLLECTION_ERROR", "Vitest could not load a test dependency"
    if any(
        token in lowered
        for token in ("unknown option", "unknown argument", "invalid option")
    ):
        return "HARNESS_ERROR", "Vitest rejected the runner invocation"
    if any(
        token in lowered for token in ("parseerror", "syntaxerror", "failed to load")
    ):
        return "COLLECTION_ERROR", "Vitest could not parse or load a test module"
    if any(
        token in lowered
        for token in ("no test files found", "no tests found", "no tests collected")
    ):
        return "NO_TESTS", "Vitest collected no tests"
    if exit_code == 5:
        return "NO_TESTS", "Vitest collected no tests"
    return None


def _assertion_node(assertion: Mapping[str, object], file_name: str | None) -> str:
    full_name = _first_text(assertion, "fullName", "name", "title") or "<unnamed test>"
    ancestors = assertion.get("ancestorTitles")
    if isinstance(ancestors, list):
        names = [item for item in ancestors if isinstance(item, str) and item]
        if names:
            full_name = " > ".join([*names, full_name])
    location = assertion.get("location")
    line: int | None = None
    if isinstance(location, Mapping):
        line = _first_int(location, "line")
    if line is None:
        line = _first_int(assertion, "line")
    prefix = f"{file_name}::" if file_name else ""
    suffix = f" (line {line})" if line is not None else ""
    return f"{prefix}{full_name}{suffix}"


def _assertion_failure(assertion: Mapping[str, object]) -> str | None:
    for key in ("failureMessages", "failureMessage", "failureDetails", "error"):
        value = assertion.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().splitlines()[0][:MAX_OUTPUT_TAIL]
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip().splitlines()[0][:MAX_OUTPUT_TAIL]
    return None


def _mapping_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _first_text(value: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item
    return None


def _first_int(value: Mapping[str, object], *keys: str) -> int | None:
    for key in keys:
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool):
            return item
    return None


def _result(
    runner: RunnerName,
    status: RunnerStatus,
    exit_code: int | None,
    selected: Sequence[str],
    primary_failure: str,
) -> TestResult:
    return TestResult(
        runner=runner,
        status=status,
        exit_code=exit_code,
        collected=0,
        selected=_selector_tuple(selected),
        failed_nodes=(),
        primary_failure=primary_failure,
        output_tail=_tail(primary_failure),
    )


def _primary_failure(text: str, failed_nodes: Sequence[str]) -> str | None:
    if failed_nodes:
        return failed_nodes[0]
    for line in text.splitlines():
        stripped = line.strip()
        if (
            stripped.startswith(("E ", "FAILED", "ERROR collecting", "> "))
            or "AssertionError" in stripped
            or "SyntaxError" in stripped
            or "ImportError" in stripped
        ):
            return stripped[:MAX_OUTPUT_TAIL]
    return None


def _selector_tuple(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(value for value in values if isinstance(value, str) and value)


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)


def _tail(value: str) -> str:
    clean = value.strip()
    return clean[-MAX_OUTPUT_TAIL:]


def _join_output(header: str, tail: str) -> str:
    return header if not tail else f"{header}\n{tail}"

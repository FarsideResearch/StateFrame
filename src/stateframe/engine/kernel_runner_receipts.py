"""Host-owned generic check descriptors, executors, and neutral receipts.

Task data may select a registered descriptor by identifier, but it cannot
provide an arbitrary shell command.  Commands are always argv vectors and are
executed with ``shell=False`` through the existing effect boundary.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from kernel_environment_types import EnvironmentInputVector, ResourcePredicate


CheckKind: TypeAlias = Literal["TEST", "BUILD", "LINT", "SCHEMA", "PROCESS"]
AdapterName: TypeAlias = Literal["plain", "junit", "tap", "pytest"]
ReceiptStatus: TypeAlias = Literal[
    "PASS", "FAIL", "NO_TESTS", "SKIPPED", "CRASH", "TIMEOUT", "INFRA_ERROR",
]
SkipState: TypeAlias = Literal["NOT_SKIPPED", "NO_TESTS", "SKIPPED"]
ApplicabilityState: TypeAlias = Literal["current", "stale", "unknown", "mixed_inputs"]
BindingMode: TypeAlias = Literal["historical_execution", "live_at_acceptance"]
CommandRunner: TypeAlias = Callable[..., subprocess.CompletedProcess[str]]
MAX_OUTPUT_TAIL = 4_000


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _tail(text: str) -> str:
    return text[-MAX_OUTPUT_TAIL:]


@dataclass(frozen=True, slots=True)
class CheckDescriptor:
    """An immutable check definition owned by the host installation."""

    check_id: str
    runner: str
    argv: tuple[str, ...]
    cwd: str = "."
    kind: CheckKind = "TEST"
    adapter: AdapterName = "plain"
    timeout: float = 300.0
    env: tuple[tuple[str, str], ...] = ()
    environment_predicates: tuple[ResourcePredicate, ...] = ()
    binding_mode: BindingMode = "historical_execution"

    def __post_init__(self) -> None:
        if not self.check_id.strip() or not self.runner.strip():
            raise ValueError("check_id and runner are required")
        if not self.argv or any(not item for item in self.argv):
            raise ValueError("descriptor argv must be a non-empty vector")
        if self.cwd.startswith(("/", "~")) or ".." in Path(self.cwd).parts:
            raise ValueError("descriptor cwd must be relative to its product root")
        if self.kind not in {"TEST", "BUILD", "LINT", "SCHEMA", "PROCESS"}:
            raise ValueError(f"unsupported check kind: {self.kind}")
        if self.adapter not in {"plain", "junit", "tap", "pytest"}:
            raise ValueError(f"unsupported check adapter: {self.adapter}")
        if self.timeout <= 0:
            raise ValueError("descriptor timeout must be positive")
        if any(not isinstance(item, ResourcePredicate) for item in self.environment_predicates):
            raise ValueError("environment_predicates must contain ResourcePredicate values")
        if self.binding_mode not in {"historical_execution", "live_at_acceptance"}:
            raise ValueError(f"unsupported binding mode: {self.binding_mode}")


@dataclass(frozen=True, slots=True)
class RunnerReceipt:
    """Runner-neutral result; pytest-specific meanings do not enter here."""

    check_id: str
    runner: str
    kind: CheckKind
    argv: tuple[str, ...]
    cwd: str
    status: ReceiptStatus
    exit_code: int | None
    signal: int | None
    timed_out: bool
    passed: int | None
    failed: int | None
    skipped: int | None
    total: int | None
    skip_state: SkipState
    output_reference: str
    output_tail: str
    product_revision: str | None = None
    environment_input_vector: EnvironmentInputVector | None = None
    applicability: ApplicabilityState = "current"

    @property
    def behavior_pass(self) -> bool:
        """Only a collected TEST pass discharges a behavior check."""
        return self.kind == "TEST" and self.status == "PASS" and (self.total or 0) > 0

    @property
    def currently_applicable(self) -> bool:
        """Current acceptance consults applicability separately from PASS."""
        return self.behavior_pass and self.applicability == "current"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "runner-neutral-receipt/1",
            "check_id": self.check_id,
            "runner": self.runner,
            "kind": self.kind,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "status": self.status,
            "exit_code": self.exit_code,
            "signal": self.signal,
            "timed_out": self.timed_out,
            "counts": {
                "passed": self.passed,
                "failed": self.failed,
                "skipped": self.skipped,
                "total": self.total,
            },
            "skip_state": self.skip_state,
            "output_reference": self.output_reference,
            "output_tail": self.output_tail,
            "product_revision": self.product_revision,
            "environment_input_vector": (
                self.environment_input_vector.to_dict()
                if self.environment_input_vector is not None else None
            ),
            "applicability": self.applicability,
        }


@dataclass(frozen=True, slots=True)
class CheckExecution:
    receipt: RunnerReceipt
    output: str
    observation: str


def _receipt(
    descriptor: CheckDescriptor,
    *,
    status: ReceiptStatus,
    exit_code: int | None,
    output: str,
    passed: int | None = None,
    failed: int | None = None,
    skipped: int | None = None,
    total: int | None = None,
    skip_state: SkipState = "NOT_SKIPPED",
    signal_number: int | None = None,
    timed_out: bool = False,
    product_revision: str | None = None,
    environment_input_vector: EnvironmentInputVector | None = None,
    applicability: ApplicabilityState = "current",
) -> RunnerReceipt:
    return RunnerReceipt(
        check_id=descriptor.check_id,
        runner=descriptor.runner,
        kind=descriptor.kind,
        argv=descriptor.argv,
        cwd=descriptor.cwd,
        status=status,
        exit_code=exit_code,
        signal=signal_number,
        timed_out=timed_out,
        passed=passed,
        failed=failed,
        skipped=skipped,
        total=total,
        skip_state=skip_state,
        output_reference=_digest(output),
        output_tail=_tail(output),
        product_revision=product_revision,
        environment_input_vector=environment_input_vector,
        applicability=applicability,
    )


def _status_from_counts(
    descriptor: CheckDescriptor,
    *,
    exit_code: int,
    passed: int,
    failed: int,
    skipped: int,
    total: int,
    output: str,
) -> RunnerReceipt:
    if descriptor.kind == "TEST" and total == 0:
        status: ReceiptStatus = "NO_TESTS" if exit_code in {0, 5} else "FAIL"
        return _receipt(descriptor, status=status, exit_code=exit_code, output=output,
                        passed=passed, failed=failed, skipped=skipped, total=total,
                        skip_state="NO_TESTS" if status == "NO_TESTS" else "NOT_SKIPPED")
    status: ReceiptStatus = "PASS" if exit_code == 0 and failed == 0 else "FAIL"
    return _receipt(descriptor, status=status, exit_code=exit_code, output=output,
                    passed=passed, failed=failed, skipped=skipped, total=total)


def adapt_plain(
    stdout: str, stderr: str, exit_code: int, descriptor: CheckDescriptor,
) -> RunnerReceipt:
    """Adapt a process exit without assigning pytest-specific meanings."""
    output = (stdout or "") + (("\n" + stderr) if stderr else "")
    if exit_code < 0:
        return _receipt(descriptor, status="CRASH", exit_code=exit_code, output=output,
                        signal_number=-exit_code)
    return _receipt(descriptor, status=("PASS" if exit_code == 0 else "FAIL"),
                    exit_code=exit_code, output=output,
                    total=(1 if descriptor.kind == "TEST" and exit_code == 0 else None),
                    passed=(1 if descriptor.kind == "TEST" and exit_code == 0 else None))


def adapt_junit(text: str, exit_code: int, descriptor: CheckDescriptor) -> RunnerReceipt:
    """Adapt JUnit XML while keeping no-tests distinct from a pass."""
    if exit_code < 0:
        return _receipt(descriptor, status="CRASH", exit_code=exit_code, output=text,
                        signal_number=-exit_code)
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return _receipt(descriptor, status="FAIL", exit_code=exit_code, output=text)
    cases = list(root.iter("testcase"))
    total = len(cases)
    if total:
        skipped = sum(1 for case in cases if case.find("skipped") is not None)
        failed = sum(1 for case in cases
                     if case.find("failure") is not None or case.find("error") is not None)
        passed = total - skipped - failed
    else:
        total = _attr_int(root, "tests")
        failed = _attr_int(root, "failures") + _attr_int(root, "errors")
        skipped = _attr_int(root, "skipped")
        passed = max(0, total - failed - skipped)
    return _status_from_counts(descriptor, exit_code=exit_code, passed=passed,
                               failed=failed, skipped=skipped, total=total, output=text)


def _attr_int(node: ElementTree.Element, name: str) -> int:
    try:
        return max(0, int(node.attrib.get(name, "0")))
    except (TypeError, ValueError):
        return 0


_TAP_PLAN = re.compile(r"^\s*1\.\.(\d+)\s*$")
_TAP_CASE = re.compile(r"^\s*(not )?ok\b", re.IGNORECASE)


def adapt_tap(text: str, exit_code: int, descriptor: CheckDescriptor) -> RunnerReceipt:
    """Adapt TAP 13 output, including plan-zero/no-test output."""
    if exit_code < 0:
        return _receipt(descriptor, status="CRASH", exit_code=exit_code, output=text,
                        signal_number=-exit_code)
    planned: int | None = None
    passed = failed = skipped = 0
    cases = 0
    for line in text.splitlines():
        plan = _TAP_PLAN.match(line)
        if plan:
            planned = int(plan.group(1))
            continue
        case = _TAP_CASE.match(line)
        if not case:
            continue
        cases += 1
        if case.group(1):
            failed += 1
        elif "# skip" in line.casefold() or "# todo" in line.casefold():
            skipped += 1
        else:
            passed += 1
    total = planned if planned is not None else cases
    if planned == 0 or (descriptor.kind == "TEST" and total == 0):
        status: ReceiptStatus = "NO_TESTS" if exit_code in {0, 5} else "FAIL"
        return _receipt(descriptor, status=status, exit_code=exit_code, output=text,
                        passed=passed, failed=failed, skipped=skipped, total=total,
                        skip_state="NO_TESTS" if status == "NO_TESTS" else "NOT_SKIPPED")
    return _status_from_counts(descriptor, exit_code=exit_code, passed=passed,
                               failed=failed, skipped=skipped, total=total, output=text)


def adapt_pytest(stdout: str, stderr: str, exit_code: int,
                 descriptor: CheckDescriptor) -> RunnerReceipt:
    """Keep pytest's exit meanings inside the pytest adapter only."""
    output = (stdout or "") + (("\n" + stderr) if stderr else "")
    passed = _count_word(output, "passed")
    failed = _count_word(output, "failed") + _count_word(output, "error")
    skipped = _count_word(output, "skipped")
    collected_match = re.search(r"collected\s+(\d+)\s+items?", output, re.IGNORECASE)
    total = (int(collected_match.group(1)) if collected_match else
             passed + failed + skipped)
    if exit_code == 5 or (exit_code == 0 and total == 0 and "no tests ran" in output.casefold()):
        return _receipt(descriptor, status="NO_TESTS", exit_code=exit_code, output=output,
                        passed=passed, failed=failed, skipped=skipped, total=total,
                        skip_state="NO_TESTS")
    return _status_from_counts(descriptor, exit_code=exit_code, passed=passed,
                               failed=failed, skipped=skipped, total=total, output=output)


def _count_word(text: str, word: str) -> int:
    match = re.search(rf"(\d+)\s+{re.escape(word)}\b", text, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def adapt_output(stdout: str, stderr: str, exit_code: int,
                 descriptor: CheckDescriptor) -> RunnerReceipt:
    output = (stdout or "") + (("\n" + stderr) if stderr else "")
    if descriptor.adapter == "junit":
        return adapt_junit(output, exit_code, descriptor)
    if descriptor.adapter == "tap":
        return adapt_tap(output, exit_code, descriptor)
    if descriptor.adapter == "pytest":
        return adapt_pytest(stdout, stderr, exit_code, descriptor)
    return adapt_plain(stdout, stderr, exit_code, descriptor)


def _product_revision(root: Path) -> str | None:
    try:
        from kernel_product_manifest import product_revision
        return product_revision(root)
    except (ImportError, OSError, ValueError):
        return None


def _run_subprocess(
    root: Path, descriptor: CheckDescriptor, *, run: CommandRunner | None,
    timeout: float | None, environment: Mapping[str, str],
) -> tuple[RunnerReceipt | None, str]:
    cwd_path = (root / descriptor.cwd).resolve()
    try:
        cwd_path.relative_to(root.resolve())
    except ValueError:
        receipt = _receipt(descriptor, status="INFRA_ERROR", exit_code=None,
                           output="descriptor cwd escapes product root")
        return receipt, receipt.output_tail
    execute = run or subprocess.run
    try:
        completed = execute(
            list(descriptor.argv), cwd=str(cwd_path), capture_output=True, text=True,
            timeout=timeout if timeout is not None else descriptor.timeout,
            env=dict(environment), shell=False,
        )
    except subprocess.TimeoutExpired as error:
        output = str(error.stdout or "") + str(error.stderr or "")
        receipt = _receipt(descriptor, status="TIMEOUT", exit_code=None, output=output,
                           timed_out=True)
        return receipt, receipt.output_tail
    except (FileNotFoundError, OSError) as error:
        receipt = _receipt(descriptor, status="INFRA_ERROR", exit_code=None,
                           output=f"runner unavailable: {type(error).__name__}")
        return receipt, receipt.output_tail
    except Exception as error:  # noqa: BLE001 - host runner boundary is fail-closed
        receipt = _receipt(descriptor, status="INFRA_ERROR", exit_code=None,
                           output=f"runner invocation failed: {type(error).__name__}")
        return receipt, receipt.output_tail
    output = (completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else "")
    return adapt_output(completed.stdout or "", completed.stderr or "", completed.returncode,
                        descriptor), output


def execute_check(root: Path, descriptor: CheckDescriptor, *, owner: object | None = None,
                  run: CommandRunner | None = None, timeout: float | None = None,
                  environment_input_vector: EnvironmentInputVector | None = None,
                  applicability: ApplicabilityState | None = None) -> CheckExecution:
    """Execute one host descriptor through the existing effect sandbox."""
    environment = dict(os.environ)
    environment.update(dict(descriptor.env))
    output_holder: dict[str, str] = {}
    revision = _product_revision(root)
    if applicability is not None:
        captured_applicability = applicability
    elif not descriptor.environment_predicates:
        captured_applicability = "current"
    elif environment_input_vector is None:
        captured_applicability = "unknown"
    elif environment_input_vector.input_coverage != "complete":
        captured_applicability = "mixed_inputs"
    else:
        # A vector is an input capture, not proof that its inputs still match
        # the current ledger.  The binding owner must explicitly authorize
        # current applicability after evaluating all predicates.
        captured_applicability = "unknown"

    def dispatch() -> RunnerReceipt:
        receipt, output = _run_subprocess(root, descriptor, run=run, timeout=timeout,
                                          environment=environment)
        assert receipt is not None
        output_holder["value"] = output
        return receipt

    try:
        from kernel_effect_application import apply_effect
        application = apply_effect(
            operation="run_check",
            snapshot={"check_id": descriptor.check_id, "kind": descriptor.kind,
                      "argv": descriptor.argv, "cwd": descriptor.cwd},
            dispatch=dispatch,
            owner=owner,
            shadow_value=_receipt(descriptor, status="SKIPPED", exit_code=None,
                                  output="effect boundary shadow mode",
                                  skip_state="SKIPPED", product_revision=revision,
                                  environment_input_vector=environment_input_vector,
                                  applicability=captured_applicability),
        )
        if not application.executed:
            receipt = application.value
            assert isinstance(receipt, RunnerReceipt)
            return CheckExecution(receipt=receipt, output=receipt.output_tail,
                                  observation=application.rendered or receipt.status)
        receipt = application.value
    except (ImportError, RuntimeError, ValueError) as error:
        # A boundary failure is a typed inability, never permission to retry
        # the effect outside its authorization path.
        receipt = _receipt(
            descriptor, status="INFRA_ERROR", exit_code=None,
            output="effect boundary unavailable: " + type(error).__name__,
            product_revision=revision,
            environment_input_vector=environment_input_vector,
            applicability=("unknown" if descriptor.environment_predicates else captured_applicability),
        )
    assert isinstance(receipt, RunnerReceipt)
    changes: dict[str, object] = {}
    if receipt.product_revision != revision:
        changes["product_revision"] = revision
    if receipt.environment_input_vector != environment_input_vector:
        changes["environment_input_vector"] = environment_input_vector
    if receipt.applicability != captured_applicability:
        changes["applicability"] = captured_applicability
    if changes:
        receipt = dataclass_replace(receipt, **changes)
    output = output_holder.get("value", receipt.output_tail)
    return CheckExecution(receipt=receipt, output=output,
                          observation=render_observation(receipt))


def dataclass_replace(receipt: RunnerReceipt, **changes: object) -> RunnerReceipt:
    """Typed local wrapper avoiding a broad ``dataclasses.replace`` return."""
    from dataclasses import replace
    return replace(receipt, **changes)


def render_observation(receipt: RunnerReceipt) -> str:
    return (f"{receipt.runner} {receipt.kind} {receipt.status} "
            f"exit={receipt.exit_code} output={receipt.output_reference}")


class CheckRegistry:
    """Host-owned descriptor registry."""

    def __init__(self, descriptors: Sequence[CheckDescriptor] = ()) -> None:
        self._descriptors = {item.check_id: item for item in descriptors}

    def register(self, descriptor: CheckDescriptor) -> None:
        self._descriptors[descriptor.check_id] = descriptor

    def get(self, check_id: str) -> CheckDescriptor | None:
        return self._descriptors.get(check_id)

    def descriptors(self) -> tuple[CheckDescriptor, ...]:
        return tuple(self._descriptors.values())


HOST_CHECKS = CheckRegistry()


def register_check(descriptor: CheckDescriptor) -> None:
    HOST_CHECKS.register(descriptor)


def register_task_checks(row: Mapping[str, object]) -> tuple[str, ...]:
    """Register declared argv checks for one scoped build row."""
    raw_checks = row.get("build_checks", row.get("checks", ()))
    entries: list[Mapping[str, object]] = []
    if isinstance(raw_checks, Mapping):
        entries = [raw_checks]
    elif isinstance(raw_checks, Sequence) and not isinstance(raw_checks, (str, bytes)):
        entries = [item for item in raw_checks if isinstance(item, Mapping)]
    build_command = row.get("build_command")
    if isinstance(build_command, Sequence) and not isinstance(build_command, (str, bytes)):
        entries.append({"check_id": "declared-build", "argv": build_command, "kind": "BUILD"})
    registered: list[str] = []
    for item in entries:
        check_id = str(item.get("check_id") or item.get("id") or "").strip()
        argv = item.get("argv", item.get("command"))
        if not check_id or not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)):
            continue
        try:
            env_value = item.get("env")
            raw_predicates = item.get("environment_predicates", ())
            predicates = tuple(
                value for value in raw_predicates
                if isinstance(value, ResourcePredicate)
            ) if isinstance(raw_predicates, Sequence) and not isinstance(raw_predicates, (str, bytes)) else ()
            descriptor = CheckDescriptor(
                check_id=check_id,
                runner=str(item.get("runner") or "subprocess"),
                argv=tuple(str(value) for value in argv),
                cwd=str(item.get("cwd") or "."),
                kind=str(item.get("kind") or
                         ("BUILD" if row.get("repo") == "build" else "TEST")),
                adapter=str(item.get("adapter") or "plain"),
                timeout=float(item.get("timeout", 300.0)),
                env=tuple(sorted((str(key), str(value)) for key, value in env_value.items()))
                if isinstance(env_value, Mapping) else (),
                environment_predicates=predicates,
                binding_mode=str(item.get("binding_mode") or "historical_execution"),
            )
        except (TypeError, ValueError):
            continue
        register_check(descriptor)
        registered.append(check_id)
    return tuple(dict.fromkeys(registered))


def run_registered_check(root: Path, check_id: str, *, owner: object | None = None,
                         run: CommandRunner | None = None,
                         environment_input_vector: EnvironmentInputVector | None = None,
                         applicability: ApplicabilityState | None = None) -> CheckExecution:
    descriptor = HOST_CHECKS.get(check_id)
    if descriptor is None:
        unknown = CheckDescriptor(check_id=check_id, runner="unknown", argv=("unknown",),
                                  kind="PROCESS")
        receipt = _receipt(unknown, status="INFRA_ERROR", exit_code=None,
                          output="unknown host-owned check descriptor")
        return CheckExecution(receipt, receipt.output_tail, render_observation(receipt))
    return execute_check(root, descriptor, owner=owner, run=run,
                         environment_input_vector=environment_input_vector,
                         applicability=applicability)


def description_for_descriptor(descriptor: CheckDescriptor) -> str:
    return (f"{descriptor.kind} check {descriptor.check_id!r} uses host runner "
            f"{descriptor.runner!r} with {descriptor.adapter} results")


def pytest_descriptor(argv: Sequence[str], *, check_id: str = "default:pytest",
                      cwd: str = ".", timeout: float = 300.0) -> CheckDescriptor:
    """Describe the legacy Python check without changing its command construction."""
    return CheckDescriptor(check_id=check_id, runner="pytest", argv=tuple(argv), cwd=cwd,
                           kind="TEST", adapter="pytest", timeout=timeout)


def capability_facts() -> dict[str, object]:
    return {
        "descriptors": [
            {"check_id": item.check_id, "runner": item.runner, "kind": item.kind,
             "adapter": item.adapter}
            for item in HOST_CHECKS.descriptors()
        ],
        "shell": False,
        "pytest_exit_semantics": "pytest adapter only",
        "default_python_descriptor": "pytest_descriptor",
        "receipt_schema": "runner-neutral-receipt/1",
        "environment_binding": "versioned-input-vector",
        "boundary_failure": "INFRA_ERROR; no fallback dispatch",
    }


__all__ = [
    "CheckDescriptor", "CheckExecution", "CheckKind", "CheckRegistry", "HOST_CHECKS",
    "RunnerReceipt", "adapt_junit", "adapt_output", "adapt_plain", "adapt_pytest", "adapt_tap",
    "capability_facts", "description_for_descriptor", "execute_check", "register_check",
    "register_task_checks",
    "pytest_descriptor", "render_observation", "run_registered_check",
]

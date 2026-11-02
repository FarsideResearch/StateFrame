"""Host-owned environment receipt and first-turn execution gate.

The preflight is deliberately a small, model-agnostic seam.  It measures the
checkout, runs the repository's declared setup recipe when one is available,
and proves that the selected test runner can collect at least one test.  The
result is an evidence receipt, not advice: a failed setup or collection blocks
the provider request and is classified as ``ENV_BROKEN``.

The module is opt-in because importing it must not change the legacy request
or journal shape.  Once enabled, a successful or broken preflight emits one
``env_preflight_fired`` journal event before the first model request.  A
missing event remains visible as ``NOT_FIRED`` instead of being mistaken for
an enabled mechanism.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from kernel_test_python import selected_test_python


MODE_ENV = "KERNEL_ENV_PREFLIGHT"
SCHEMA = "kernel-env-receipt/1"
EVENT_KIND = "env_preflight_fired"

ENV_READY = "ENV_READY"
ENV_BROKEN = "ENV_BROKEN"
NOT_FIRED = "NOT_FIRED"
FIRED = "FIRED"
ON = "ON"
OFF = "OFF"
UNSET = "UNSET"
INVALID = "INVALID"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", "disabled"})
_SWITCH_CONTAINER_KEYS = (
    "switches", "switch_names", "feature_switches", "harness_switches",
    "named_switches", "required_switches", "all_switches", "effective_switches",
    "switches_above",
)
_COMMAND_KEYS = (
    "dependency_commands", "dependency_install", "install_commands",
    "install_command", "ci_install", "ci_recipe",
)
_TEST_COMMAND_KEYS = (
    "collection_command", "collect_command", "test_command", "test_cmd",
    "pytest_command",
)
_COUNT_PATTERNS = (
    re.compile(r"\bcollected\s+(\d+)\s+(?:items?|tests?)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s+(?:items?|tests?)\s+collected\b", re.IGNORECASE),
    re.compile(r"\bcollected\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s+tests?\s+selected\b", re.IGNORECASE),
    re.compile(r"^\s*#?\s*tests?\s+(\d+)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bTests?:\s*(\d+)\b", re.IGNORECASE),
)


class EnvPreflightError(RuntimeError):
    """A caller attempted a model turn without a runnable environment."""

    def __init__(self, receipt: "EnvReceipt") -> None:
        self.receipt = receipt
        super().__init__(receipt.status)


@dataclass(frozen=True, slots=True)
class CommandEvidence:
    command: tuple[str, ...]
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr).strip()

    @property
    def output_digest(self) -> str:
        return _digest_text(self.output)

    def as_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "output_digest": self.output_digest,
        }


@dataclass(frozen=True, slots=True)
class CheckoutFacts:
    path: str
    valid: bool
    head_sha: str | None
    file_count: int
    evidence: CommandEvidence | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "valid": self.valid,
            "head_sha": self.head_sha,
            "file_count": self.file_count,
            "evidence": self.evidence.as_dict() if self.evidence else None,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class DependencyFacts:
    status: str
    recipe: tuple[tuple[str, ...], ...] = ()
    recipe_source: str = "none"
    evidence: tuple[CommandEvidence, ...] = ()
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "recipe": [list(command) for command in self.recipe],
            "recipe_source": self.recipe_source,
            "evidence": [item.as_dict() for item in self.evidence],
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class CollectionFacts:
    status: str
    command: tuple[str, ...]
    count: int
    exit_code: int | None
    output_digest: str
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "command": list(self.command),
            "count": self.count,
            "exit_code": self.exit_code,
            "output_digest": self.output_digest,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class SwitchFact:
    name: str
    raw_value: str | None
    effective: str

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_value": self.raw_value,
            "effective": self.effective,
        }


@dataclass(frozen=True, slots=True)
class EnvReceipt:
    """Bounded host facts for one pre-model-turn environment check."""

    checkout: CheckoutFacts
    dependencies: DependencyFacts
    collection: CollectionFacts
    switches: Mapping[str, SwitchFact]
    off_above: tuple[str, ...]
    classification: str
    failure_stage: str | None = None
    failure_reason: str | None = None
    event_status: str = NOT_FIRED
    event_id: str | None = None
    event_error: str | None = None
    mechanism: str = MODE_ENV
    schema: str = SCHEMA
    chargeable: bool = False

    @property
    def receipt_digest(self) -> str:
        return _digest(self._core_dict())

    @property
    def status(self) -> str:
        return self.event_status if self.event_status != FIRED else self.classification

    @property
    def result_class(self) -> str:
        return self.classification

    @property
    def can_start_model(self) -> bool:
        return self.classification == ENV_READY and self.event_status == FIRED

    @property
    def fired(self) -> bool:
        return self.event_status == FIRED

    def _core_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "mechanism": self.mechanism,
            "checkout": self.checkout.as_dict(),
            "dependencies": self.dependencies.as_dict(),
            "collection": self.collection.as_dict(),
            "switches": {
                name: value.as_dict() for name, value in sorted(self.switches.items())
            },
            "off_above": list(self.off_above),
            "classification": self.classification,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "chargeable": self.chargeable,
        }

    def as_dict(self) -> dict[str, object]:
        value = self._core_dict()
        value.update({
            "status": self.status,
            "event_status": self.event_status,
            "event_id": self.event_id,
            "event_error": self.event_error,
            "receipt_digest": self.receipt_digest,
            "can_start_model": self.can_start_model,
        })
        return value

    to_dict = as_dict


def mode() -> str:
    """Return the strict opt-in mode; absent configuration is legacy OFF."""
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False, default=str).encode()


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _digest_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _command(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    raise TypeError("command must be a string or sequence")


def _commands(value: object) -> tuple[tuple[str, ...], ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_command(value),)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = list(value)
        if not items:
            return ()
        if all(isinstance(item, str) for item in items):
            return (_command(items),)
        return tuple(_command(item) for item in items)
    raise TypeError("commands must be a string or sequence")


def _as_evidence(command: tuple[str, ...], result: object) -> CommandEvidence:
    if isinstance(result, CommandEvidence):
        return result
    if isinstance(result, subprocess.CompletedProcess):
        return CommandEvidence(command, result.returncode,
                               str(result.stdout or ""), str(result.stderr or ""))
    if isinstance(result, Mapping):
        code = result.get("returncode", result.get("exit_code"))
        return CommandEvidence(
            command,
            int(code) if isinstance(code, int) and not isinstance(code, bool) else None,
            str(result.get("stdout", "") or ""), str(result.get("stderr", "") or ""),
            bool(result.get("timed_out", False)),
        )
    if isinstance(result, tuple) and result:
        code = result[0]
        return CommandEvidence(
            command,
            int(code) if isinstance(code, int) and not isinstance(code, bool) else None,
            str(result[1] or "") if len(result) > 1 else "",
            str(result[2] or "") if len(result) > 2 else "",
        )
    if isinstance(result, int) and not isinstance(result, bool):
        return CommandEvidence(command, result)
    raise TypeError("command runner must return CompletedProcess, mapping, tuple, int, or CommandEvidence")


def _run(command: tuple[str, ...], root: Path, timeout: float,
         runner: Callable[..., object] | None) -> CommandEvidence:
    if not command:
        return CommandEvidence(command, 2, stderr="empty command")
    try:
        if runner is None:
            from kernel_execution_policy import run_sandboxed
            execution = run_sandboxed(
                command,
                root,
                timeout=timeout,
                output_limit=256 * 1024,
                cpu_seconds=min(900.0, max(1.0, timeout)),
                file_bytes=8 * 1024 * 1024,
            )
            result = execution.completed
            if execution.timed_out:
                return CommandEvidence(
                    command, 124, result.stdout or "", result.stderr or "", timed_out=True,
                )
        else:
            result = runner(command, root, timeout)
        return _as_evidence(command, result)
    except subprocess.TimeoutExpired as error:
        return CommandEvidence(command, 124, str(error.stdout or ""),
                               str(error.stderr or ""), timed_out=True)
    except (OSError, subprocess.SubprocessError) as error:
        return CommandEvidence(command, 127, stderr=f"{type(error).__name__}: {error}")
    except Exception as error:  # runner adapters must become host evidence, not model work
        return CommandEvidence(command, 127, stderr=f"{type(error).__name__}: {error}")


def _file_count(root: Path) -> int:
    if not root.is_dir():
        return 0
    count = 0
    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        try:
            if path.is_file() and not path.is_symlink():
                count += 1
        except OSError:
            continue
    return count


def _task_mapping(task: Mapping[str, object] | None) -> Mapping[str, object]:
    return task if isinstance(task, Mapping) else {}


def _checkout(root: Path, task: Mapping[str, object], timeout: float,
              runner: Callable[..., object] | None) -> CheckoutFacts:
    files = _file_count(root)
    if not root.is_dir():
        return CheckoutFacts(str(root), False, None, files, error="checkout directory missing")
    command = ("git", "-C", str(root), "rev-parse", "HEAD")
    evidence = _run(command, root, timeout, runner)
    head = evidence.stdout.strip().splitlines()[0] if evidence.stdout.strip() else None
    task_head = False
    if not head:
        for key in ("head_sha", "checkout_sha", "commit", "sha"):
            candidate = task.get(key)
            if isinstance(candidate, str) and candidate.strip():
                head = candidate.strip()
                task_head = True
                break
    # Build-task sandboxes are intentionally materialised without .git.  Their
    # host-owned row SHA is the checkout identity; a real git checkout still
    # requires a successful HEAD query.
    valid = bool(head) and files > 0 and (
        evidence.exit_code in (0, None) or task_head)
    error = None
    if not valid:
        error = "git HEAD unavailable" if not head else "checkout contains no files"
    return CheckoutFacts(str(root), valid, head, files, evidence=evidence, error=error)


def _workflow_recipe(root: Path) -> tuple[tuple[tuple[str, ...], ...], str]:
    """Find a narrow install command from a checked-in CI recipe."""
    workflows = sorted((root / ".github" / "workflows").glob("*.y*ml"))
    for workflow in workflows:
        try:
            for line in workflow.read_text(encoding="utf-8", errors="replace").splitlines():
                text = line.strip()
                if not text.startswith("run:"):
                    continue
                candidate = text[4:].strip().strip("\"'")
                if not any(token in candidate for token in ("pip install", "uv sync",
                                                            "poetry install", "npm ci",
                                                            "yarn install", "pnpm install")):
                    continue
                return (_command(candidate), workflow.relative_to(root).as_posix())
        except (OSError, ValueError):
            continue
    if (root / "package-lock.json").is_file():
        return (("npm", "ci"), "package-lock.json")
    if (root / "pnpm-lock.yaml").is_file():
        return (("pnpm", "install", "--frozen-lockfile"), "pnpm-lock.yaml")
    if (root / "yarn.lock").is_file():
        return (("yarn", "install", "--frozen-lockfile"), "yarn.lock")
    if (root / "requirements.txt").is_file():
        return ((selected_test_python(), "-m", "pip", "install", "-r", "requirements.txt"),
                "requirements.txt")
    if any((root / name).is_file() for name in ("pyproject.toml", "setup.py", "setup.cfg")):
        return ((selected_test_python(), "-m", "pip", "install", "-e", "."), "python-package")
    return (), "none"


def _task_commands(task: Mapping[str, object], keys: Sequence[str]) -> tuple[tuple[str, ...], ...] | None:
    for key in keys:
        if key in task:
            return _commands(task.get(key))
    return None


def _dependencies(root: Path, task: Mapping[str, object], timeout: float,
                  runner: Callable[..., object] | None,
                  explicit: Sequence[object] | str | None) -> DependencyFacts:
    recipe_source = "none"
    if explicit is not None:
        recipe = _commands(explicit)
        recipe_source = "caller"
    else:
        recipe = _task_commands(task, _COMMAND_KEYS)
        if recipe is not None:
            recipe_source = "task"
        else:
            recipe, recipe_source = _workflow_recipe(root)
    if not recipe:
        return DependencyFacts("NOT_CONFIGURED", recipe=(), recipe_source=recipe_source)
    evidence: list[CommandEvidence] = []
    for command in recipe:
        result = _run(command, root, timeout, runner)
        evidence.append(result)
        if result.exit_code != 0:
            detail = "timed out" if result.timed_out else (result.output.splitlines()[-1]
                                                             if result.output else "command failed")
            return DependencyFacts(ENV_BROKEN, recipe=recipe, recipe_source=recipe_source,
                                   evidence=tuple(evidence), error=detail[:240])
    return DependencyFacts(ENV_READY, recipe=recipe, recipe_source=recipe_source,
                           evidence=tuple(evidence))


def _collection_command(root: Path, task: Mapping[str, object], explicit: object) -> tuple[str, ...]:
    if explicit is not None:
        command = _command(explicit)
    else:
        configured = _task_commands(task, _TEST_COMMAND_KEYS)
        if configured:
            command = configured[0]
        elif task.get("language") == "node" or (root / "package.json").is_file():
            command = ("node", "--test")
        else:
            executable = task.get("python") or task.get("interpreter") or selected_test_python()
            command = (str(executable), "-m", "pytest", "--collect-only", "-q", "--no-header")
    paths = task.get("test_paths", task.get("tests"))
    if isinstance(paths, str):
        paths = (paths,)
    if isinstance(paths, Sequence) and not isinstance(paths, (bytes, bytearray)):
        command = (*command, *(str(path) for path in paths))
    return command


def _collection(root: Path, task: Mapping[str, object], timeout: float,
                runner: Callable[..., object] | None, explicit: object) -> CollectionFacts:
    command = _collection_command(root, task, explicit)
    evidence = _run(command, root, timeout, runner)
    output = evidence.output
    count = 0
    for pattern in _COUNT_PATTERNS:
        match = pattern.search(output)
        if match:
            count = int(match.group(1))
            break
    if evidence.exit_code != 0:
        error = "timed out" if evidence.timed_out else (output.splitlines()[-1]
                                                         if output else "collection failed")
        return CollectionFacts(ENV_BROKEN, command, count, evidence.exit_code,
                               evidence.output_digest, error[:240])
    if count <= 0:
        return CollectionFacts(ENV_BROKEN, command, count, evidence.exit_code,
                               evidence.output_digest, "collection count is zero or unavailable")
    return CollectionFacts(ENV_READY, command, count, evidence.exit_code,
                           evidence.output_digest)


def _looks_like_switch(name: str) -> bool:
    return (name.startswith(("KERNEL_", "STATEFRAME_", "REACT_"))
            or name.endswith(("_MODE", "_SWITCH", "_GUARD", "_ENABLED", "_INTERVENTIONS")))


def _switch_inputs(task: Mapping[str, object], explicit: Iterable[str] | None) -> tuple[list[str], dict[str, object]]:
    names: list[str] = []
    values: dict[str, object] = {}

    def add(value: object, *, take_values: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                text = str(key).strip()
                if text and text not in names:
                    names.append(text)
                if take_values and text:
                    values[text] = item
        elif isinstance(value, str):
            if value.strip() and value.strip() not in names:
                names.append(value.strip())
        elif isinstance(value, Iterable):
            for item in value:
                add(item, take_values=False)

    if explicit is not None:
        add((explicit,) if isinstance(explicit, str) else tuple(explicit))
    for key in _SWITCH_CONTAINER_KEYS:
        if key in task:
            add(task.get(key), take_values=True)
    for key, value in task.items():
        text = str(key).strip()
        if text.isupper() and _looks_like_switch(text):
            add(text, take_values=True)
            values[text] = value
    if MODE_ENV not in names:
        names.insert(0, MODE_ENV)
    return names, values


def _switch_fact(name: str, configured: Mapping[str, object]) -> SwitchFact:
    raw: object = os.environ.get(name)
    if raw is None and name in configured and isinstance(configured[name], (str, int, bool)):
        raw = configured[name]
    if name == MODE_ENV and raw is None:
        raw = "on"
    raw_text = None if raw is None else str(raw)
    if raw is None:
        effective = UNSET
    else:
        normalized = raw_text.strip().lower()
        effective = ON if normalized in _TRUE_VALUES else OFF if normalized in _FALSE_VALUES else INVALID
    return SwitchFact(name, raw_text, effective)


def _switches(task: Mapping[str, object], explicit: Iterable[str] | None,
              above: Iterable[str] | None, mechanism: str) -> tuple[dict[str, SwitchFact], tuple[str, ...]]:
    names, configured = _switch_inputs(task, explicit)
    facts = {name: _switch_fact(name, configured) for name in names}
    if above is None:
        candidate = task.get("switches_above")
        if isinstance(candidate, str):
            candidate = (candidate,)
        above = (candidate if isinstance(candidate, Iterable)
                 else tuple(name for name in facts if name != mechanism))
    elif isinstance(above, str):
        above = (above,)
    above_names = tuple(str(name) for name in above if str(name) in facts and str(name) != mechanism)
    return facts, tuple(name for name in above_names if facts[name].effective != ON)


def _failure(checkout: CheckoutFacts, dependencies: DependencyFacts,
             collection: CollectionFacts, stage: str | None = None,
             reason: str | None = None) -> tuple[str, str | None, str | None]:
    if not checkout.valid:
        return ENV_BROKEN, "checkout", checkout.error
    if dependencies.status == ENV_BROKEN:
        return ENV_BROKEN, "dependencies", dependencies.error
    if collection.status == ENV_BROKEN:
        return ENV_BROKEN, "collection", collection.error
    return ENV_READY, stage, reason


def _emit(receipt: EnvReceipt, journal: object | None) -> EnvReceipt:
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        return receipt
    try:
        fired = replace(receipt, event_status=FIRED, event_error=None)
        event = emitter(
            EVENT_KIND,
            producer="kernel_env_preflight",
            mechanism=receipt.mechanism,
            receipt=fired.as_dict(),
            receipt_digest=receipt.receipt_digest,
            event_fired=True,
            action="environment_preflight",
        )
        event_id = event.get("event_id") if isinstance(event, Mapping) else None
        return replace(fired, event_id=str(event_id) if event_id else None)
    except Exception as error:  # evidence must say NOT_FIRED; do not fake liveness
        return replace(receipt, event_status=NOT_FIRED,
                       event_error=f"{type(error).__name__}: {error}"[:240])


def _receipt_digest_from_mapping(value: Mapping[str, object]) -> str:
    excluded = {"status", "event_status", "event_id", "event_error",
                "receipt_digest", "can_start_model"}
    return _digest({key: item for key, item in value.items() if key not in excluded})


def verify_event_fired(receipt: EnvReceipt, events: Iterable[Mapping[str, object]]) -> EnvReceipt:
    """Reconcile a receipt with persisted events; absence is explicitly NOT_FIRED."""
    for event in events:
        if event.get("kind") != EVENT_KIND or event.get("event_fired") is not True:
            continue
        matching_digest = event.get("receipt_digest")
        nested = event.get("receipt")
        if matching_digest != receipt.receipt_digest and not (
                isinstance(nested, Mapping)
                and _receipt_digest_from_mapping(nested) == receipt.receipt_digest):
            continue
        event_id = event.get("event_id")
        return replace(receipt, event_status=FIRED,
                       event_id=str(event_id) if event_id else receipt.event_id,
                       event_error=None)
    return replace(receipt, event_status=NOT_FIRED, event_id=None)


def stamp_batch_record(record: object, receipt: EnvReceipt) -> None:
    """Stamp the exact receipt into a mutable batch row when one is supplied."""
    payload = receipt.as_dict()
    if isinstance(record, MutableMapping):
        record["env_receipt"] = payload
        previous = record.get("env_receipts")
        rows = list(previous) if isinstance(previous, list) else []
        if not any(isinstance(row, Mapping)
                   and row.get("receipt_digest") == receipt.receipt_digest for row in rows):
            rows.append(payload)
        record["env_receipts"] = rows
        return
    method = getattr(record, "stamp_env_receipt", None)
    if callable(method):
        method(payload)
        return
    try:
        setattr(record, "env_receipt", payload)
    except (AttributeError, TypeError):
        pass


def run_preflight(
    root: str | os.PathLike[str],
    *,
    task: Mapping[str, object] | None = None,
    journal: object | None = None,
    batch_record: object | None = None,
    command_runner: Callable[..., object] | None = None,
    dependency_commands: Sequence[object] | str | None = None,
    collection_command: object | None = None,
    switch_names: Iterable[str] | None = None,
    switches_above: Iterable[str] | None = None,
    mechanism: str = MODE_ENV,
    timeout: float = 120.0,
) -> EnvReceipt | None:
    """Run one opt-in preflight and return the host receipt.

    ``None`` is the intentional OFF result.  On the ON path a journal is part
    of the proof: without a fired event ``can_start_model`` is false and the
    receipt status is ``NOT_FIRED``.
    """
    if mode() == "off":
        return None
    if timeout <= 0:
        raise ValueError("preflight timeout must be positive")
    root_path = Path(root).resolve()
    task_map = _task_mapping(task)
    checkout = _checkout(root_path, task_map, timeout, command_runner)
    dependencies = _dependencies(root_path, task_map, timeout, command_runner,
                                 dependency_commands)
    if dependencies.status == ENV_BROKEN:
        collection = CollectionFacts("SKIPPED", (), 0, None, _digest_text(""),
                                      "dependency setup failed")
    else:
        collection = _collection(root_path, task_map, timeout, command_runner,
                                 collection_command)
    switches, off_above = _switches(task_map, switch_names, switches_above, mechanism)
    classification, failure_stage, failure_reason = _failure(
        checkout, dependencies, collection)
    receipt = EnvReceipt(
        checkout=checkout,
        dependencies=dependencies,
        collection=collection,
        switches=switches,
        off_above=off_above,
        classification=classification,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        mechanism=mechanism,
    )
    receipt = _emit(receipt, journal)
    if batch_record is not None:
        stamp_batch_record(batch_record, receipt)
    return receipt


def dispatch_if_ready(receipt: EnvReceipt | None, request: Callable[[], Any]) -> Any:
    """Call ``request`` only when the opt-in receipt proves a live environment."""
    if receipt is not None and not receipt.can_start_model:
        return receipt
    return request()


def render_host_facts(receipt: EnvReceipt | Mapping[str, object]) -> str:
    """Render bounded facts suitable for a model-facing host block."""
    value = receipt.as_dict() if isinstance(receipt, EnvReceipt) else dict(receipt)
    compact = {
        "status": value.get("status"),
        "classification": value.get("classification"),
        "checkout": {
            "valid": value.get("checkout", {}).get("valid")
            if isinstance(value.get("checkout"), Mapping) else None,
            "head_sha": value.get("checkout", {}).get("head_sha")
            if isinstance(value.get("checkout"), Mapping) else None,
            "file_count": value.get("checkout", {}).get("file_count")
            if isinstance(value.get("checkout"), Mapping) else None,
        },
        "dependencies": {
            "status": value.get("dependencies", {}).get("status")
            if isinstance(value.get("dependencies"), Mapping) else None,
            "recipe_source": value.get("dependencies", {}).get("recipe_source")
            if isinstance(value.get("dependencies"), Mapping) else None,
        },
        "collection": {
            "status": value.get("collection", {}).get("status")
            if isinstance(value.get("collection"), Mapping) else None,
            "count": value.get("collection", {}).get("count")
            if isinstance(value.get("collection"), Mapping) else None,
            "exit_code": value.get("collection", {}).get("exit_code")
            if isinstance(value.get("collection"), Mapping) else None,
        },
        "switches": {
            name: (item.get("effective") if isinstance(item, Mapping) else item)
            for name, item in (value.get("switches", {}).items()
                               if isinstance(value.get("switches"), Mapping) else ())
        },
        "off_above": value.get("off_above", []),
        "event": {"status": value.get("event_status"), "event_id": value.get("event_id")},
        "receipt_digest": value.get("receipt_digest"),
    }
    return "[ENV_RECEIPT]\n" + json.dumps(compact, sort_keys=True, separators=(",", ":"))


# Explicit aliases keep the seam discoverable to runner integrations without
# creating a second implementation or a second switch.
env_preflight_mode = mode
preflight = run_preflight
guard_model_turn = dispatch_if_ready


__all__ = [
    "EVENT_KIND", "ENV_BROKEN", "ENV_READY", "FIRED", "INVALID", "MODE_ENV",
    "NOT_FIRED", "OFF", "ON", "UNSET", "CollectionFacts", "CommandEvidence",
    "CheckoutFacts", "DependencyFacts", "EnvPreflightError", "EnvReceipt",
    "SwitchFact", "dispatch_if_ready", "enabled", "env_preflight_mode", "guard_model_turn",
    "mode", "preflight", "render_host_facts", "run_preflight", "stamp_batch_record",
    "verify_event_fired",
]

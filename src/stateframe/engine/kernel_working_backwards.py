"""R-140 working-backwards tooling with a strict public/private boundary.

The protocol has two separate jobs:

* admit only tasks for which every supplied arm ended in a real failure;
* let a controller validate the reference behaviour in a worker environment,
  then hand the model only a bounded, public ``what would it take`` packet.

Reference workspaces and worker command output are deliberately consumed behind
the validator boundary.  They are never copied into a packet, a render, or a
receipt.  The runtime hook is opt-in through ``STATEFRAME_WORKING_BACKWARDS``;
the default path returns its input list unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


MODE_ENV = "STATEFRAME_WORKING_BACKWARDS"
SCHEMA = "stateframe-working-backwards/1"
REPORT_SCHEMA = "stateframe-working-backwards-report/1"
HEADER = "[WORKING_BACKWARDS_PACKET]"
MAX_TEXT = 900
MAX_SECTION_ITEMS = 8
MAX_RENDER_CHARS = 6000

_WORKER_ENV = {
    "PATH": os.defpath,
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONIOENCODING": "utf-8",
}

_GREEN = frozenset({
    "GREEN", "PASS", "PASSED", "SUCCESS", "SUCCEEDED", "CERTIFIED", "OK",
})
_FAIL = frozenset({
    "FAIL", "FAILED", "RED", "ERROR", "EXCEPTION", "TIMEOUT", "TIMED_OUT",
    "BUDGET_EXHAUSTED", "UNRESOLVED_RED", "HARNESS_ERROR",
})
_UNKNOWN = frozenset({
    "UNKNOWN", "NOT_RUN", "NOT_EXECUTED", "SKIPPED", "INCONCLUSIVE", "ABORTED",
})
# Fixed public categories for GoldValidation.runner (finding 4); an
# unrecognized value collapses to "custom" rather than surviving verbatim.
_RUNNER_CATEGORIES = frozenset({"pytest", "unittest", "custom"})

# These are field names or path-shaped markers, not ordinary prose words such
# as "gold validation" or "oracle".  The packet builder uses an allowlist for
# fields and this second check catches accidental inclusion in public prose.
_PRIVATE_KEY_MARKERS = frozenset({
    "fixed_src", "fixed_source", "gold_src", "gold_source", "gold_hash",
    "hidden_tests", "hidden_test", "near_miss", "oracle_digest", "reference_patch",
    "reference_source", "solution_patch", "patch_text", "private_data", "secret",
})
_PRIVATE_TEXT_RE = re.compile(
    r"(?i)(?:^|[\s/\\.:_-])(?:fixed_(?:src|source)|gold_(?:src|source|hash)|"
    r"hidden_tests?|near_miss|oracle_digest|reference_patch|reference_source|"
    r"solution_patch|patch_text|private_data|secret)(?:$|[\s/\\.:_-])"
)


class WorkingBackwardsError(ValueError):
    """Raised when a packet or worker validation breaks its bounded contract."""


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict opt-in mode; absent configuration is legacy OFF."""

    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _private_key(value: object) -> bool:
    text = str(value).strip().lower()
    return text in _PRIVATE_KEY_MARKERS or any(
        marker in text for marker in _PRIVATE_KEY_MARKERS
        if marker.endswith(("_src", "_source", "_tests", "_test", "_patch", "_hash"))
    )


def _private_text(value: str) -> bool:
    return bool(_PRIVATE_TEXT_RE.search(value))


def _safe_text(value: object, field: str, *, limit: int = MAX_TEXT) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    if _private_text(value):
        raise WorkingBackwardsError(f"{field} contains a private-data marker")
    text = " ".join(value.replace("\r", " ").replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _safe_identifier(value: object, field: str) -> str:
    text = _safe_text(value, field, limit=180)
    if not text:
        return "task"
    return text


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        candidate = to_dict()
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _digest(value: object) -> str:
    encoded = json.dumps(
        _canonical(value), ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _normal_status(value: object) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if text in _GREEN or text.startswith("PASS_"):
        return "GREEN"
    if text in _FAIL or any(token in text for token in ("FAIL", "ERROR", "TIMEOUT")):
        return "FAIL"
    if text in _UNKNOWN or not text:
        return "UNKNOWN"
    return "UNKNOWN"


def _field(value: object, *names: str) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


@dataclass(frozen=True, slots=True)
class GoldValidation:
    """Public projection of a controller-only worker validation."""

    status: str
    worker_env: str = "UNKNOWN"
    runner: str = "pytest"
    exit_code: int | None = None
    timed_out: bool = False

    def __post_init__(self) -> None:
        status = str(self.status).strip().upper().replace("-", "_")
        if status in _GREEN:
            status = "GREEN"
        elif status in _FAIL:
            status = "NOT_GREEN"
        elif status not in {"NOT_GREEN", "UNKNOWN"}:
            status = "UNKNOWN"
        environment = str(self.worker_env).strip().upper().replace("-", "_")
        if environment in {"READY", "OK", "AVAILABLE", "GREEN"}:
            environment = "READY"
        elif environment in {"NOT_READY", "MISSING", "BROKEN", "UNAVAILABLE"}:
            environment = "NOT_READY"
        else:
            environment = "UNKNOWN"
        # Fix (post-review r44, finding 4): `runner` used to accept
        # arbitrary text (up to 80 chars) with no category restriction, so
        # a full filesystem path (e.g. a controller-side checkout path)
        # could survive straight into a report/render that otherwise claims
        # to be path-free. Restrict to a small fixed category; anything
        # else collapses to the generic "custom" category, never the raw
        # text.
        runner_text = _safe_text(self.runner, "worker runner", limit=80)
        runner = runner_text.strip().lower() if runner_text else "pytest"
        if runner not in _RUNNER_CATEGORIES:
            runner = "custom"
        if self.exit_code is not None and (
            not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool)
        ):
            raise WorkingBackwardsError("worker exit_code must be an integer or null")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "worker_env", environment)
        object.__setattr__(self, "runner", runner)
        object.__setattr__(self, "timed_out", bool(self.timed_out))

    @property
    def green(self) -> bool:
        # Fix (post-review r44, finding 2): a nonzero/known-failing
        # exit_code is authoritative CONTRADICTING evidence -- a receipt
        # claiming status="GREEN" (or a caller-asserted `passed=True`) while
        # exit_code=1 is a conflict, not admissible green evidence. The old
        # property ignored exit_code entirely.
        if self.exit_code is not None and self.exit_code != 0:
            return False
        if self.timed_out:
            return False
        return self.status == "GREEN" and self.worker_env == "READY"

    @property
    def reason_code(self) -> str:
        if self.green:
            return "GOLD_PUBLIC_CHECKS_GREEN"
        if self.worker_env != "READY":
            return "WORKER_ENV_NOT_READY"
        if self.timed_out:
            return "GOLD_CHECK_TIMEOUT"
        if self.status == "NOT_GREEN":
            return "GOLD_NOT_GREEN"
        return "GOLD_VALIDATION_UNKNOWN"

    def to_dict(self) -> dict[str, object]:
        """Return only safe facts; no command, path, stdout, stderr, or digest."""

        return {
            "status": self.status,
            "worker_env": self.worker_env,
            "runner": self.runner,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_value(cls, value: object) -> "GoldValidation":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            # Fix (post-review r44, finding 2): a bare boolean is an
            # underspecified receipt -- it asserts pass/fail with NO
            # environment or execution evidence at all, yet used to
            # synthesize worker_env="READY" outright. Reject it; admission
            # requires an explicit structured receipt.
            raise WorkingBackwardsError(
                "worker admission requires a structured receipt "
                "(status/worker_env/exit_code), not a bare boolean")
        passed = _field(value, "gold_green", "gold_passed")
        environment_ready = _field(value, "worker_env_ready", "environment_ready")
        worker_env = _field(value, "worker_env", "environment")
        status = _field(value, "status", "result_class", "result", "outcome")
        nested_gold = _field(value, "gold", "gold_result", "gold_receipt")
        if isinstance(nested_gold, Mapping) or nested_gold is not None:
            if status is None:
                status = _field(nested_gold, "status", "result_class", "result", "outcome")
            if passed is None:
                passed = _field(nested_gold, "passed", "green", "ok")
            if _field(value, "exit_code") is None:
                status_code = _field(nested_gold, "returncode", "exit_code")
            else:
                status_code = _field(value, "exit_code")
        else:
            status_code = _field(value, "returncode", "exit_code")
        object_passed = _field(value, "passed")
        if passed is None:
            passed = object_passed
        if passed is True:
            status = "GREEN"
        elif passed is False and status is None:
            status = "NOT_GREEN"
        if status is None and isinstance(status_code, int) and not isinstance(status_code, bool):
            status = "GREEN" if status_code == 0 else "NOT_GREEN"
        if status is None:
            status = "UNKNOWN"
        if isinstance(environment_ready, bool):
            worker_env = "READY" if environment_ready else "NOT_READY"
        if worker_env is None:
            # Fix (post-review r44, finding 2): environment readiness must
            # be its own EXPLICIT evidence (worker_env_ready/environment_ready
            # or a worker_env/environment field) -- it must never be inferred
            # from the status text alone. `{'status': 'GREEN'}` with no
            # environment evidence at all is UNKNOWN, not synthesized READY.
            worker_env = "UNKNOWN"
        timed_out = bool(_field(value, "timed_out", "timeout"))
        if nested_gold is not None:
            timed_out = timed_out or bool(_field(nested_gold, "timed_out", "timeout"))
        runner = _field(value, "runner", "runner_name") or "pytest"
        exit_code = status_code if isinstance(status_code, int) and not isinstance(status_code, bool) else None
        return cls(str(status), str(worker_env), str(runner), exit_code, timed_out)


def _worker_spec(value: object) -> tuple[Path, Path, tuple[str, ...], Mapping[str, str], float, str]:
    spec = _mapping(value)
    from kernel_test_python import selected_test_python
    python_value = spec.get("worker_python", spec.get("python", selected_test_python()))
    workspace_value = spec.get("gold_workspace", spec.get("workspace"))
    if workspace_value is None:
        raise WorkingBackwardsError("worker workspace is required")
    command_value = spec.get("command", spec.get("test_command"))
    if command_value is None:
        raise WorkingBackwardsError("worker command is required")
    if isinstance(command_value, str):
        command = tuple(shlex.split(command_value))
    elif isinstance(command_value, Sequence) and not isinstance(command_value, (bytes, bytearray)):
        command = tuple(str(item) for item in command_value)
    else:
        raise WorkingBackwardsError("worker command must be a sequence or shell-free string")
    if not command or any("\x00" in item for item in command):
        raise WorkingBackwardsError("worker command is empty or contains NUL")
    environment_value = spec.get("environment", {})
    if not isinstance(environment_value, Mapping):
        raise WorkingBackwardsError("worker environment must be an object")
    environment = {str(key): str(item) for key, item in environment_value.items()}
    timeout_value = spec.get("timeout", spec.get("timeout_seconds", 600))
    try:
        timeout = float(timeout_value)
    except (TypeError, ValueError) as exc:
        raise WorkingBackwardsError("worker timeout must be positive") from exc
    if timeout <= 0 or timeout > 3600:
        raise WorkingBackwardsError("worker timeout must be between 0 and 3600 seconds")
    runner = str(spec.get("runner", "pytest"))
    return Path(str(python_value)), Path(str(workspace_value)), command, environment, timeout, runner


def run_gold_in_worker(spec: object) -> GoldValidation:
    """Run only the declared public-check command in a controller-owned worker.

    The command may inspect the reference checkout supplied by the controller,
    but this function returns no process output or filesystem identity.  A
    non-zero result is intentionally not converted into a model-facing failure
    tail: it simply refuses oracle admission.
    """

    try:
        executable, workspace, command, environment, timeout, runner = _worker_spec(spec)
    except (OSError, WorkingBackwardsError):
        return GoldValidation("UNKNOWN", worker_env="NOT_READY")
    if not workspace.is_dir() or not executable.is_file():
        return GoldValidation("UNKNOWN", worker_env="NOT_READY", runner=runner)
    # Fix (post-review r44, finding 1): `executable` (the declared worker
    # interpreter) was checked for existence and then never used -- the
    # unrelated `command` executed with the CONTROLLER's inherited
    # environment/PATH, so a controller Python/public-test invocation could
    # certify GREEN although the declared worker interpreter/environment
    # would fail. Require the command's own first token to BE the declared
    # interpreter (resolved); anything else is a command/interpreter
    # mismatch and refuses admission rather than silently substituting the
    # controller's own resolution of that name.
    try:
        first_token_resolved = Path(command[0]).resolve()
    except (OSError, ValueError):
        first_token_resolved = None
    if first_token_resolved is None or first_token_resolved != executable.resolve():
        return GoldValidation("UNKNOWN", worker_env="NOT_READY", runner=runner)
    child_environment = dict(_WORKER_ENV)
    child_environment.update(environment)
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=child_environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return GoldValidation("UNKNOWN", worker_env="READY", runner=runner, exit_code=124, timed_out=True)
    except (OSError, subprocess.SubprocessError):
        return GoldValidation("UNKNOWN", worker_env="READY", runner=runner)
    return GoldValidation(
        "GREEN" if completed.returncode == 0 else "NOT_GREEN",
        worker_env="READY",
        runner=runner,
        exit_code=completed.returncode,
    )


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    name: str
    status: str
    failure_class: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _safe_identifier(self.name, "arm name"))
        status = _normal_status(self.status)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "failure_class", _safe_text(self.failure_class, "failure class", limit=120))

    def to_dict(self) -> dict[str, str]:
        value = {"name": self.name, "status": self.status}
        if self.failure_class:
            value["failure_class"] = self.failure_class
        return value


def _arm_items(task: Mapping[str, object]) -> list[tuple[str, object]]:
    raw = task.get("arms", task.get("arm_results", task.get("outcomes")))
    if isinstance(raw, Mapping):
        return [(str(name), value) for name, value in raw.items()]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return [(str(index), value) for index, value in enumerate(raw)]
    return []


def arm_outcomes(task: Mapping[str, object]) -> tuple[ArmOutcome, ...]:
    outcomes: list[ArmOutcome] = []
    for fallback_name, value in _arm_items(task):
        name = _field(value, "name", "arm", "id") or fallback_name
        status = _field(value, "status", "result", "outcome", "grade", "behaviour")
        terminal = _field(value, "terminal")
        if status is None:
            status = _field(terminal, "status", "result", "outcome", "grade")
        failure_class = _field(
            value, "failure_class", "autopsy_class", "failure_kind", "reason_code",
        ) or _field(terminal, "failure_class", "autopsy_class", "failure_kind") or ""
        outcomes.append(ArmOutcome(str(name), str(status or ""), str(failure_class)))
    return tuple(outcomes)


def all_arms_failed(task: object) -> bool:
    """Return true only for a non-empty arm set in which every status is FAIL."""

    if not isinstance(task, Mapping):
        return False
    outcomes = arm_outcomes(task)
    return bool(outcomes) and all(outcome.status == "FAIL" for outcome in outcomes)


def _sequence_values(value: object, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Mapping):
        values = tuple(f"{key}: {child}" for key, child in value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(str(item) for item in value)
    else:
        values = (str(value),)
    result: list[str] = []
    for item in values:
        text = _safe_text(item, field)
        if text and text not in result:
            result.append(text)
    return tuple(result[:MAX_SECTION_ITEMS])


_PUBLIC_TASK_FIELDS = frozenset({
    "task_id", "id", "fn",
    "brief", "msg", "task",
    "public_tests", "public_selectors", "selectors",
    "arms", "arm_results", "outcomes",
    "gap_classes",
    "analysis", "public_analysis",
})


def _public_task_projection(task: Mapping[str, object]) -> dict[str, object]:
    """The ONLY task view the analysis callback and packet builder may see.

    Fix (post-review r44, finding 3): the caller-supplied ``analysis``
    callback previously received the WHOLE raw task -- including any
    private fields it happens to carry (fixed_src, hidden_test_src,
    oracle_symptom, ...) -- and its returned prose was trusted as public,
    letting a buggy or malicious callback copy private source straight into
    a packet section while the packet still claimed ``private_inputs_used:
    False``. Building this explicit allowlisted projection BEFORE invoking
    analysis means a private field is never even reachable from analysis or
    packet construction, regardless of what the callback tries to do with
    it. Worker validation is a SEPARATE path and still receives the full
    raw task (the worker legitimately needs reference/checkout access).
    """
    return {key: task[key] for key in _PUBLIC_TASK_FIELDS if key in task}


def _analysis(task: Mapping[str, object], analysis: object) -> Mapping[str, object]:
    if isinstance(analysis, Mapping):
        return analysis
    candidate = task.get("analysis", task.get("public_analysis", {}))
    return candidate if isinstance(candidate, Mapping) else {}


def _section(
    task: Mapping[str, object], analysis: Mapping[str, object], key: str, default: str,
) -> tuple[str, ...]:
    value = analysis.get(key, task.get(key))
    result = _sequence_values(value, f"public analysis {key}")
    return result or (default,)


def _public_tests(task: Mapping[str, object]) -> tuple[str, ...]:
    value = task.get("public_tests", task.get("public_selectors", task.get("selectors", ())))
    if isinstance(value, Mapping):
        raw = []
        for name, item in value.items():
            if isinstance(item, Mapping):
                item_value = item.get("selector", item.get("id", item.get("path", name)))
            else:
                item_value = item or name
            raw.append(item_value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw = list(value)
    else:
        raw = [value] if value else []
    result: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            item = item.get("selector", item.get("id", item.get("path", "")))
        text = _safe_text(item, "public test selector", limit=220)
        if text and text not in result:
            result.append(text)
    return tuple(result[:MAX_SECTION_ITEMS])


@dataclass(frozen=True, slots=True)
class WorkingBackwardsPacket:
    """A deterministic, public-only packet suitable for the next model request."""

    task_id: str
    brief: str
    public_tests: tuple[str, ...]
    arms: tuple[ArmOutcome, ...]
    validation: GoldValidation
    solved_end_state: tuple[str, ...]
    know: tuple[str, ...]
    do: tuple[str, ...]
    observe: tuple[str, ...]
    verify: tuple[str, ...]
    ask: tuple[str, ...]
    gap_classes: tuple[str, ...]
    probe_input_fields: tuple[str, ...] = (
        "public_task.brief", "public_task.public_tests", "arm_outcomes", "public_analysis",
    )
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise WorkingBackwardsError(f"schema must be {SCHEMA}")
        if not self.arms or not all(arm.status == "FAIL" for arm in self.arms):
            raise WorkingBackwardsError("packet requires all arms to be failed")
        if not self.validation.green:
            raise WorkingBackwardsError("packet requires green gold validation")
        object.__setattr__(self, "task_id", _safe_identifier(self.task_id, "task id"))
        object.__setattr__(self, "brief", _safe_text(self.brief, "public brief"))
        # Fix (post-review r44, finding 4): __post_init__ previously
        # validated only task_id/brief. Every other text-bearing field
        # (public_tests, the seven what-would-it-take sections,
        # gap_classes, probe_input_fields) was trusted as-is -- a packet
        # instance built OUTSIDE build_packet (e.g. `dataclasses.replace`)
        # could carry a private-data marker straight past construction and
        # into `render()`, which -- unlike `to_dict()` -- never re-checked
        # it. Re-run the SAME `_safe_text` sanitizer (raises on a private
        # marker) on every field here, at the one place no caller can skip.
        for name in ("public_tests", "solved_end_state", "know", "do", "observe",
                     "verify", "ask", "gap_classes", "probe_input_fields"):
            object.__setattr__(self, name, tuple(
                _safe_text(item, f"packet {name}") for item in getattr(self, name)
                if _safe_text(item, f"packet {name}")
            ))

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "public_only": True,
            "public_task": {
                "brief": self.brief,
                "public_tests": list(self.public_tests),
            },
            "all_arms_failed": True,
            "arms": [arm.to_dict() for arm in self.arms],
            "worker_validation": self.validation.to_dict(),
            "what_would_it_take": {
                "solved_end_state": list(self.solved_end_state),
                "know": list(self.know),
                "do": list(self.do),
                "observe": list(self.observe),
                "verify": list(self.verify),
                "ask": list(self.ask),
                "gap_classes": list(self.gap_classes),
            },
            "probe_basis": {
                "public_fields": list(self.probe_input_fields),
                "private_inputs_used": False,
            },
        }

    @property
    def packet_id(self) -> str:
        return _digest(self._payload())

    def to_dict(self) -> dict[str, object]:
        value = {"packet_id": self.packet_id, **self._payload()}
        _assert_public(value)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "WorkingBackwardsPacket":
        if value.get("schema") != SCHEMA:
            raise WorkingBackwardsError(f"schema must be {SCHEMA}")
        public_task = value.get("public_task")
        if not isinstance(public_task, Mapping):
            raise WorkingBackwardsError("packet public_task is required")
        work = value.get("what_would_it_take")
        if not isinstance(work, Mapping):
            raise WorkingBackwardsError("packet what_would_it_take is required")
        raw_arms = value.get("arms")
        if not isinstance(raw_arms, Sequence) or isinstance(raw_arms, (str, bytes, bytearray)):
            raise WorkingBackwardsError("packet arms are required")
        arms = tuple(
            ArmOutcome(
                str(item.get("name", "arm")),
                str(item.get("status", "")),
                str(item.get("failure_class", "")),
            )
            for item in raw_arms if isinstance(item, Mapping)
        )
        validation = GoldValidation.from_value(value.get("worker_validation"))
        packet = cls(
            task_id=str(value.get("task_id", "task")),
            brief=str(public_task.get("brief", "")),
            public_tests=tuple(_sequence_values(public_task.get("public_tests", ()), "public tests")),
            arms=arms,
            validation=validation,
            solved_end_state=tuple(_sequence_values(work.get("solved_end_state"), "solved end state")),
            know=tuple(_sequence_values(work.get("know"), "public analysis know")),
            do=tuple(_sequence_values(work.get("do"), "public analysis do")),
            observe=tuple(_sequence_values(work.get("observe"), "public analysis observe")),
            verify=tuple(_sequence_values(work.get("verify"), "public analysis verify")),
            ask=tuple(_sequence_values(work.get("ask"), "public analysis ask")),
            gap_classes=tuple(_sequence_values(work.get("gap_classes"), "gap classes")),
            probe_input_fields=tuple(_sequence_values(
                _mapping(value.get("probe_basis")).get("public_fields"), "probe basis",
            )) or (
                "public_task.brief", "public_task.public_tests", "arm_outcomes", "public_analysis",
            ),
        )
        supplied = value.get("packet_id")
        if supplied not in (None, "") and str(supplied) != packet.packet_id:
            raise WorkingBackwardsError("packet_id does not match public packet content")
        return packet

    def render(self, *, max_chars: int = MAX_RENDER_CHARS) -> str:
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 256:
            raise WorkingBackwardsError("packet render bound must be at least 256 characters")
        # Fix (post-review r44, finding 4): render() used to build its text
        # straight from self.* without the same boundary check to_dict()
        # applies -- run the identical fail-closed scan here too, so no
        # rendering path can ever disagree with to_dict() about what is
        # actually public.
        _assert_public(self._payload())
        lines = [
            HEADER,
            "PUBLIC-ONLY: yes",
            f"TASK: {self.task_id}",
            f"BRIEF: {self.brief}",
            "WORKER VALIDATION: " + self.validation.status
            + f" (environment={self.validation.worker_env}; reason={self.validation.reason_code})",
            "PUBLIC TESTS: " + (", ".join(self.public_tests) if self.public_tests else "none declared"),
            "ALL ARMS FAILED: yes",
        ]
        for title, values in (
            ("SOLVED END STATE", self.solved_end_state),
            ("KNOW", self.know),
            ("DO", self.do),
            ("OBSERVE", self.observe),
            ("VERIFY", self.verify),
            ("ASK", self.ask),
            ("GAP CLASSES", self.gap_classes),
        ):
            lines.append(title + ":")
            lines.extend(f"- {value}" for value in values)
        lines.append("PROBE BASIS: " + ", ".join(self.probe_input_fields)
                     + "; private inputs used: no")
        rendered = "\n".join(lines)
        if len(rendered) <= max_chars:
            return rendered
        suffix = "\n[WORKING_BACKWARDS_PACKET_TRUNCATED]"
        return rendered[: max_chars - len(suffix)].rstrip() + suffix


def _assert_public(value: object) -> None:
    """Fail closed if any packet field accidentally crosses the public boundary."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if _private_key(key):
                raise WorkingBackwardsError(f"private field {key!r} crossed public boundary")
            _assert_public(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _assert_public(child)
    elif isinstance(value, str) and _private_text(value):
        raise WorkingBackwardsError("private marker crossed public boundary")


def build_packet(
    task: Mapping[str, object],
    validation: GoldValidation | Mapping[str, object] | bool,
    *,
    analysis: Mapping[str, object] | None = None,
) -> WorkingBackwardsPacket:
    """Compile one all-failed task into a public-only packet."""

    if not isinstance(task, Mapping) or not all_arms_failed(task):
        raise WorkingBackwardsError("task is not an all-arms-fail task")
    checked = GoldValidation.from_value(validation)
    if not checked.green:
        raise WorkingBackwardsError("gold validation is not green")
    # Fix (post-review r44, finding 3): project to the public-only field
    # allowlist HERE, at the actual packet-construction boundary -- this is
    # the documented public entry point and may be called directly (not
    # only through run_protocol), so the guarantee belongs here, not only
    # at one caller.
    task = _public_task_projection(task)
    arm_rows = arm_outcomes(task)
    public_analysis = _analysis(task, analysis)
    brief = _safe_text(
        task.get("brief", task.get("msg", task.get("task", ""))), "public brief",
    ) or "No public brief supplied."
    explicit_gaps = public_analysis.get("gap_classes", task.get("gap_classes"))
    gap_classes = _sequence_values(explicit_gaps, "gap classes")
    if not gap_classes:
        gap_classes = tuple(dict.fromkeys(
            arm.failure_class for arm in arm_rows if arm.failure_class
        )) or ("UNRESOLVED_PUBLIC_FAILURE",)
    return WorkingBackwardsPacket(
        task_id=_safe_identifier(task.get("task_id", task.get("id", task.get("fn", "task"))), "task id"),
        brief=brief,
        public_tests=_public_tests(task),
        arms=arm_rows,
        validation=checked,
        solved_end_state=_section(
            task, public_analysis, "solved_end_state",
            "The declared public checks pass in the worker environment and the public contract is evidenced.",
        ),
        know=_section(
            task, public_analysis, "know",
            "State the public contract and separate observed facts from unresolved causes.",
        ),
        do=_section(
            task, public_analysis, "do",
            "Reproduce the public failure, make the smallest contract-preserving change, and keep the evidence public.",
        ),
        observe=_section(
            task, public_analysis, "observe",
            "Compare each failed arm's public evidence with the source and record what is still unknown.",
        ),
        verify=_section(
            task, public_analysis, "verify",
            "Run every declared public check in the same worker environment and require a green result.",
        ),
        ask=_section(
            task, public_analysis, "ask",
            "Which public contract detail remains ambiguous after the available evidence is inspected?",
        ),
        gap_classes=gap_classes,
    )


def _validation_from_worker(worker: object, task: Mapping[str, object]) -> GoldValidation:
    if worker is None:
        return GoldValidation("UNKNOWN", worker_env="NOT_READY")
    if callable(worker):
        return GoldValidation.from_value(worker(task))
    validate = getattr(worker, "validate", None)
    if callable(validate):
        return GoldValidation.from_value(validate(task))
    return run_gold_in_worker(worker)


@dataclass(frozen=True, slots=True)
class TaskDecision:
    task_id: str
    status: str
    reason: str
    validation: GoldValidation | None = None
    packet: WorkingBackwardsPacket | None = None

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "task_id": _safe_identifier(self.task_id, "decision task id"),
            "status": self.status,
            "reason": self.reason,
            "candidate": self.status != "SKIPPED",
        }
        if self.validation is not None:
            value["worker_validation"] = self.validation.to_dict()
        if self.packet is not None:
            value["packet"] = self.packet.to_dict()
        _assert_public(value)
        return value


@dataclass(frozen=True, slots=True)
class ProtocolReport:
    results: tuple[TaskDecision, ...]
    schema: str = REPORT_SCHEMA

    @property
    def packets(self) -> tuple[WorkingBackwardsPacket, ...]:
        return tuple(row.packet for row in self.results if row.packet is not None)

    def to_dict(self) -> dict[str, object]:
        value = {
            "schema": self.schema,
            "results": [row.to_dict() for row in self.results],
            "candidate_count": sum(row.status != "SKIPPED" for row in self.results),
            "packet_count": len(self.packets),
            "refused_count": sum(row.status == "REFUSED" for row in self.results),
        }
        _assert_public(value)
        return value


def run_protocol(
    tasks: Sequence[Mapping[str, object]] | Mapping[str, object],
    worker: object,
    *,
    analysis: Mapping[str, object] | Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
) -> ProtocolReport:
    """Validate and packetize every all-arms-fail task in ``tasks``."""

    rows: Sequence[object]
    if isinstance(tasks, Mapping):
        rows = (tasks,)
    elif isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes, bytearray)):
        rows = tasks
    else:
        raise WorkingBackwardsError("tasks must be a mapping or sequence")
    decisions: list[TaskDecision] = []
    for raw_task in rows:
        if not isinstance(raw_task, Mapping):
            decisions.append(TaskDecision("task", "SKIPPED", "TASK_NOT_PUBLIC_MAPPING"))
            continue
        task_id = _safe_identifier(raw_task.get("task_id", raw_task.get("id", raw_task.get("fn", "task"))), "task id")
        if not all_arms_failed(raw_task):
            decisions.append(TaskDecision(task_id, "SKIPPED", "NOT_ALL_ARMS_FAILED"))
            continue
        try:
            checked = _validation_from_worker(worker, raw_task)
        except Exception:  # worker failures become a non-admitting host receipt
            checked = GoldValidation("UNKNOWN", worker_env="UNKNOWN")
        if not checked.green:
            decisions.append(TaskDecision(task_id, "REFUSED", checked.reason_code, checked))
            continue
        try:
            public_task = _public_task_projection(raw_task)
            if callable(analysis):
                selected_analysis = analysis(public_task)
            elif isinstance(analysis, Mapping) and task_id in analysis and isinstance(analysis[task_id], Mapping):
                selected_analysis = analysis[task_id]
            else:
                selected_analysis = analysis
            packet = build_packet(public_task, checked, analysis=selected_analysis)
        except WorkingBackwardsError:
            decisions.append(TaskDecision(task_id, "REFUSED", "PUBLIC_PACKET_REJECTED", checked))
            continue
        decisions.append(TaskDecision(task_id, "PACKET_READY", "PUBLIC_PACKET_READY", checked, packet))
    report = ProtocolReport(tuple(decisions))
    report.to_dict()
    return report


def append_packet(
    messages: list[dict], packet: WorkingBackwardsPacket | Mapping[str, object] | None,
    *,
    provider_request: bool = True,
) -> list[dict]:
    """Append a packet only in the explicit arm; OFF returns the same list."""

    if not provider_request or not enabled() or packet is None:
        return messages
    checked = packet if isinstance(packet, WorkingBackwardsPacket) else WorkingBackwardsPacket.from_mapping(packet)
    if any(isinstance(message, Mapping) and HEADER in str(message.get("content", ""))
           for message in messages):
        return messages
    return [*messages, {"role": "user", "content": checked.render()}]


def attach_packet(owner: object, packet: WorkingBackwardsPacket | Mapping[str, object]) -> bool:
    """Attach a validated packet to a runtime owner in the explicit arm."""

    if not enabled():
        return False
    checked = packet if isinstance(packet, WorkingBackwardsPacket) else WorkingBackwardsPacket.from_mapping(packet)
    setattr(owner, "working_backwards_packet", checked)
    return True


def run_and_attach_packet(
    owner: object,
    tasks: Sequence[Mapping[str, object]] | Mapping[str, object],
    worker: object,
    *,
    analysis: Mapping[str, object] | Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
) -> ProtocolReport:
    """Run the opt-in protocol at the controller seam and attach its first packet."""
    if not enabled():
        return ProtocolReport(())
    report = run_protocol(tasks, worker, analysis=analysis)
    if report.packets:
        attach_packet(owner, report.packets[0])
    return report


# Names used by protocol callers that prefer verb-oriented APIs.
compile_packet = build_packet
compile_packets = run_protocol
validate_gold = run_gold_in_worker


__all__ = [
    "ArmOutcome", "GoldValidation", "HEADER", "MODE_ENV", "MAX_RENDER_CHARS",
    "ProtocolReport", "REPORT_SCHEMA", "SCHEMA", "TaskDecision",
    "WorkingBackwardsError", "WorkingBackwardsPacket", "all_arms_failed",
    "append_packet", "arm_outcomes", "attach_packet", "build_packet", "compile_packet",
    "compile_packets", "enabled", "mode", "run_and_attach_packet", "run_gold_in_worker",
    "run_protocol", "validate_gold",
]

"""Host-owned, revision-bound reproduction evidence for bug work.

This is a kernel TRUTH mechanism: it only accepts observations made by the
host runner.  Prompt wording belongs to the coding adapter; this module owns
admission, evidence identity, and the partial-completion boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


MODE_ENV = "STATEFRAME_REPRO_FIRST_GATE"
# editing permit: the editing permit is separate from completion and strict test-first ordering is not a default.
EDIT_PERMIT_ENV = "STATEFRAME_REPRO_EDIT_PERMIT"
EXEMPTION_KINDS = frozenset({
    "feature_contract_test", "refactor", "docs", "env_unavailable", "flaky_ambiguous",
})
SCHEMA = "repro-first-gate/1"
DEFAULT_MAX_ATTEMPTS = 3
UNKNOWN_REVISION = "UNKNOWN"

_BUG_TICKET_VALUES = frozenset({
    "bug", "bugfix", "defect", "incident", "regression", "repair",
})
_RED_VALUES = frozenset({"RED", "FAIL", "FAILED", "FAILURE", "FAILING"})
_GREEN_VALUES = frozenset({
    "GREEN", "PASS", "PASSED", "SUCCESS", "CERTIFIED", "OK", "PASSED_TESTS",
})
_UNKNOWN_VALUES = frozenset({
    "UNKNOWN", "INFRA", "INFRASTRUCTURE", "NOT_RUN", "NO_ORACLE", "TIMEOUT",
})
_ERROR_LINE = re.compile(
    r"(?:^\s*E\s+|AssertionError|Traceback \(|(?:^|\s)(?:FAILED|ERROR)\b|\bassert\s+)",
    re.IGNORECASE,
)
_ENVIRONMENT_FAILURE = re.compile(
    r"(?:ImportError|ModuleNotFoundError|No module named|cannot import name|"
    r"collection error|test collection failed)",
    re.IGNORECASE,
)
_EXIT_CODE = re.compile(r"pytest\s+(?:exit|exited)\s+(\d+)", re.IGNORECASE)
_IGNORED_PARTS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".test-tmp", "htmlcov", ".ruff_cache",
    ".mypy_cache", "measurement-runs", "portfolio", "continuation-workspaces",
})
_IGNORED_FILE_NAMES = frozenset({".coverage"})
_IGNORED_FILE_PREFIXES = (".coverage.",)


def repro_first_gate_mode() -> str:
    """Return the strictly validated feature mode."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return repro_first_gate_mode() == "on"


def edit_permit_mode() -> str:
    """advisory (default): would-be edit refusals are journaled, not enforced. enforce: lane-original refusal."""
    value = os.environ.get(EDIT_PERMIT_ENV, "advisory").strip().lower()
    if value not in {"advisory", "enforce"}:
        raise ValueError(f"{EDIT_PERMIT_ENV} must be advisory or enforce")
    return value


def _ticket_field(ticket: object, *names: str) -> object:
    if isinstance(ticket, Mapping):
        for name in names:
            if name in ticket:
                return ticket[name]
    for name in names:
        value = getattr(ticket, name, None)
        if value is not None:
            return value
    return None


def _normalized_ticket_kind(ticket: object) -> str:
    if isinstance(ticket, str):
        return ticket.strip().lower()
    value = _ticket_field(
        ticket, "ticket_type", "issue_type", "task_type", "category", "kind", "type",
    )
    return str(value).strip().lower() if value is not None else ""


def requires_repro(ticket: object) -> bool:
    """Gate only explicitly typed bug/regression work.

    A symptom containing the word ``bug`` is not a ticket type.  That keeps
    the kernel domain-general and avoids silently changing feature-task
    semantics.
    """
    if _normalized_ticket_kind(ticket) in _BUG_TICKET_VALUES:
        return True
    return bool(
        _ticket_field(ticket, "is_bug") is True
        or _ticket_field(ticket, "is_regression") is True
    )


def product_edit_allowed(gate: object | None, *, code_hash: object = None,
                         is_test: bool = False) -> bool:
    """Return the side-effect-free product-edit permission used by both menu and write gates.

    ``admit_edit`` remains responsible for journaling and advancing gate state.  This
    predicate only answers whether a product-path edit could be admitted at the current
    witness, so a request menu cannot advertise an edit that the write boundary rejects.
    """
    if gate is None or not bool(getattr(gate, "active", False)) or is_test:
        return True
    if getattr(gate, "edit_permit", "advisory") != "enforce":
        return True
    if getattr(gate, "exemption", None) is not None:
        return True
    if getattr(gate, "_unreproduced_risk", None) is not None:
        return False
    receipt = getattr(gate, "receipt", None)
    if receipt is None:
        return False
    if bool(getattr(gate, "_edit_started", False)):
        return True
    current = _revision(code_hash if code_hash is not None else getattr(gate, "revision", None))
    return current == getattr(receipt, "code_hash", UNKNOWN_REVISION)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False, default=str).encode()


def _hash_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _tree_inventory(root: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (any(part in _IGNORED_PARTS for part in relative.parts)
                or relative.name in _IGNORED_FILE_NAMES
                or any(relative.name.startswith(prefix) for prefix in _IGNORED_FILE_PREFIXES)):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        rows.append((relative.as_posix(), _hash_bytes(path.read_bytes())))
    return rows


def code_hash(value: object) -> str:
    """Return a deterministic workspace/file identity without exposing source."""
    if isinstance(value, Path) or isinstance(value, str):
        candidate = Path(value)
        if candidate.is_dir():
            return _hash_bytes(_canonical({"files": _tree_inventory(candidate.resolve())}))
        if candidate.is_file():
            return _hash_bytes(candidate.read_bytes())
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _hash_bytes(_canonical(value))


def _revision(value: object) -> str:
    if value is None:
        return UNKNOWN_REVISION
    text = str(value).strip()
    return text or UNKNOWN_REVISION


def _first_error_line(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    fallback = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("pytest exit"):
            continue
        if _ERROR_LINE.search(line):
            if re.match(r"^E\s+", line, re.IGNORECASE) or \
                    "assertionerror" in line.lower() or "traceback (" in line.lower():
                return line[:500]
            fallback = fallback or line[:500]
    return fallback


def _exit_code(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = _EXIT_CODE.search(value)
        if match:
            return int(match.group(1))
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _status(value: object, exit_code: int | None) -> str:
    if isinstance(value, str):
        upper = value.strip().upper()
        if upper in _RED_VALUES or "RED" == upper:
            return "RED"
        if upper in _GREEN_VALUES or "GREEN" == upper:
            return "GREEN"
        if upper in _UNKNOWN_VALUES:
            return "UNKNOWN"
        if "pytest exit 1" in upper:
            return "RED"
        if "pytest exit 0" in upper:
            return "GREEN"
    if exit_code == 1:
        return "RED"
    if exit_code == 0:
        return "GREEN"
    return "UNKNOWN"


@dataclass(frozen=True)
class ProbeResult:
    status: str
    command: str
    probe_id: str
    code_hash: str
    first_error_line: str | None = None
    exit_code: int | None = None
    output: str = ""
    outcome: str = "UNKNOWN"
    reached_observation: bool | None = None
    failure_phase: str = "UNKNOWN"

    @property
    def fingerprint(self) -> str:
        return _hash_bytes(_canonical({
            "command": self.command,
            "probe_id": self.probe_id,
        }))

    @classmethod
    def from_value(cls, value: object, *, command: object = None,
                   probe_id: object = None, code_hash: object = None,
                   exit_code: object = None) -> "ProbeResult":
        if isinstance(value, cls):
            return cls(
                status=value.status,
                command=str(command or value.command),
                probe_id=str(probe_id or value.probe_id),
                code_hash=_revision(code_hash if code_hash is not None else value.code_hash),
                first_error_line=value.first_error_line,
                exit_code=_exit_code(exit_code if exit_code is not None else value.exit_code),
                output=value.output,
                outcome=value.outcome,
                reached_observation=value.reached_observation,
                failure_phase=value.failure_phase,
            )
        fields = value if isinstance(value, Mapping) else {}
        raw_output = value if isinstance(value, str) else fields.get(
            "output", fields.get("stdout", fields.get("observation", "")))
        stderr = fields.get("stderr", "") if isinstance(fields, Mapping) else ""
        output = "\n".join(part for part in (str(raw_output or ""), str(stderr or "")) if part)
        raw_exit = exit_code if exit_code is not None else (
            fields.get("exit_code", fields.get("exit")) if isinstance(fields, Mapping) else None)
        parsed_exit = _exit_code(raw_exit if raw_exit is not None else output)
        raw_status = fields.get("status", fields.get("result_class",
                             fields.get("result"))) if isinstance(fields, Mapping) else value
        raw_command = command if command is not None else (
            fields.get("command", fields.get("cmd", "pytest"))
            if isinstance(fields, Mapping) else "pytest")
        raw_probe = probe_id if probe_id is not None else (
            fields.get("probe_id", fields.get("selector", raw_command))
            if isinstance(fields, Mapping) else raw_command)
        raw_hash = code_hash if code_hash is not None else (
            fields.get("code_hash", fields.get("revision", UNKNOWN_REVISION))
            if isinstance(fields, Mapping) else UNKNOWN_REVISION)
        explicit_error = fields.get("first_error_line") if isinstance(fields, Mapping) else None
        error_line = (str(explicit_error).strip()[:500] if explicit_error else
                      _first_error_line(output))
        explicit_outcome = str(fields.get("outcome", "")).upper() if isinstance(fields, Mapping) else ""
        explicit_phase = str(fields.get("failure_phase", "")).upper() if isinstance(fields, Mapping) else ""
        if (
            _ENVIRONMENT_FAILURE.search(output)
            or explicit_outcome == "ENVIRONMENT"
            or explicit_phase in {"IMPORT", "COLLECTION"}
        ):
            probe_status = "ENVIRONMENT"
            probe_outcome = "ENVIRONMENT"
            probe_phase = (
                explicit_phase
                if explicit_phase in {"COLLECTION", "IMPORT", "EXECUTION", "UNKNOWN"}
                else "IMPORT" if "IMPORT" in output.upper() else "COLLECTION"
            )
            observation_reached = False
        else:
            probe_status = _status(raw_status, parsed_exit)
            probe_outcome = (
                explicit_outcome
                if explicit_outcome in {"PASS", "BEHAVIOR_FAIL", "INVALID_CHECK", "RESOURCE_LIMIT", "NOT_RUN", "UNKNOWN"}
                else ("PASS" if probe_status == "GREEN" else "BEHAVIOR_FAIL" if probe_status == "RED" else "UNKNOWN")
            )
            probe_phase = explicit_phase if explicit_phase in {
                "NONE", "EXECUTION", "ASSERTION", "OBSERVATION", "TIMEOUT", "UNKNOWN"
            } else ("NONE" if probe_outcome == "PASS" else "ASSERTION" if probe_outcome == "BEHAVIOR_FAIL" else "UNKNOWN")
            raw_reached = fields.get("reached_observation") if isinstance(fields, Mapping) else None
            observation_reached = raw_reached if isinstance(raw_reached, bool) else (
                True if probe_outcome == "PASS" else probe_phase in {"ASSERTION", "OBSERVATION"} if probe_outcome == "BEHAVIOR_FAIL" else None
            )
        return cls(
            status=probe_status,
            command=str(raw_command or "pytest"),
            probe_id=str(raw_probe or raw_command or "pytest"),
            code_hash=_revision(raw_hash),
            first_error_line=error_line or None,
            exit_code=parsed_exit,
            output=output[-12000:],
            outcome=probe_outcome,
            reached_observation=observation_reached,
            failure_phase=probe_phase,
        )


@dataclass(frozen=True)
class ReproReceipt:
    schema: str
    receipt_id: str
    probe_fingerprint: str
    command: str
    probe_id: str
    code_hash: str
    first_error_line: str
    attempt: int
    authority: str = "HOST_OBSERVED"
    outcome: str = "BEHAVIOR_FAIL"
    reached_observation: bool = True
    failure_phase: str = "ASSERTION"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    status: str
    code: str
    detail: str = ""
    receipt: ReproReceipt | None = None

    @property
    def allowed(self) -> bool:
        return self.accepted

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        if self.receipt is not None:
            value["receipt"] = self.receipt.to_dict()
        return value


class ReproFirstGate:
    """State machine for a single typed bug/regression ticket."""

    def __init__(self, ticket: object, *, revision: object = None,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS, mode: str | None = None,
                 journal: object = None, task_id: object = None):
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        selected_mode = repro_first_gate_mode() if mode is None else mode
        if selected_mode not in {"off", "on"}:
            raise ValueError("mode must be off or on")
        self.ticket = ticket
        self.ticket_kind = _normalized_ticket_kind(ticket) or "unspecified"
        self.mode = selected_mode
        self.required = requires_repro(ticket)
        self.revision = _revision(revision)
        self.max_attempts = max_attempts
        self.journal = journal
        self.task_id = task_id
        self.attempts = 0
        self.receipt: ReproReceipt | None = None
        self.events: list[dict[str, object]] = []
        self._latest_green: ProbeResult | None = None
        self._latest_probe: ProbeResult | None = None
        self._edit_started = False
        self._unreproduced_risk: str | None = None
        self.edit_permit = edit_permit_mode() if self.mode == "on" else "advisory"
        self.exemption: dict[str, str] | None = None
        self.witness_version = 0

    @property
    def active(self) -> bool:
        return self.mode == "on" and self.required

    @property
    def completion_cap(self) -> str | None:
        return "partial" if self._unreproduced_risk is not None else None

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "active": self.active,
            "ticket_kind": self.ticket_kind,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "receipt": self.receipt.to_dict() if self.receipt else None,
            "completion_cap": self.completion_cap,
            "unreproduced_risk": self._unreproduced_risk,
            "edit_permit": self.edit_permit,
            "exemption": dict(self.exemption) if self.exemption else None,
            "witness_version": self.witness_version,
        }

    def exempt(self, kind: object, reason: object) -> GateDecision:
        """Typed exemption: authorizes editing, never marks the ticket satisfied."""
        if not self.active:
            return GateDecision(True, "INACTIVE", "NOT_APPLICABLE")
        kind_text = str(kind or "").strip().lower()
        reason_text = str(reason or "").strip()
        if kind_text not in EXEMPTION_KINDS:
            return self._decision(
                False, "REJECTED", "REPRO_EXEMPTION_UNKNOWN",
                f"Exemption kind must be one of {sorted(EXEMPTION_KINDS)}.", action="exempt")
        if not reason_text:
            return self._decision(
                False, "REJECTED", "REPRO_EXEMPTION_REASON_REQUIRED",
                "A typed exemption requires a reason.", action="exempt")
        self.exemption = {"kind": kind_text, "reason": reason_text[:1000]}
        return self._decision(
            True, "EXEMPT", "REPRO_EXEMPTED",
            f"Editing authorized by typed exemption {kind_text}; completion stays unverified.",
            action="exempt")

    def _emit(self, action: str, decision: GateDecision, *,
              probe: ProbeResult | None = None, code_hash: object = None) -> None:
        if not self.active:
            return
        payload: dict[str, object] = {
            "schema_name": SCHEMA,
            "producer": "kernel_repro_first",
            "organ_id": "repro_first_gate",
            "event_name": "REPRO_FIRST_GATE_FIRED",
            "action": action,
            "decision": decision.code,
            "status": decision.status,
            "reason_code": decision.code,
            "ticket_kind": self.ticket_kind,
            "attempt": self.attempts,
            "code_hash": _revision(code_hash if code_hash is not None else self.revision),
            "probe_fingerprint": probe.fingerprint if probe is not None else None,
            "dispatch": False,
        }
        self.events.append(dict(payload))
        emitter = getattr(self.journal, "emit", None)
        if callable(emitter):
            emitter("repro_first_gate_fired", **payload)

    def _decision(self, accepted: bool, status: str, code: str, detail: str,
                  *, action: str, probe: ProbeResult | None = None,
                  code_hash: object = None,
                  receipt: ReproReceipt | None = None) -> GateDecision:
        decision = GateDecision(accepted, status, code, detail, receipt)
        self._emit(action, decision, probe=probe, code_hash=code_hash)
        return decision

    def observe_probe(self, value: object, *, command: object = None,
                      probe_id: object = None, code_hash: object = None,
                      exit_code: object = None) -> GateDecision:
        probe = ProbeResult.from_value(value, command=command, probe_id=probe_id,
                                       code_hash=code_hash, exit_code=exit_code)
        if not self.active:
            return GateDecision(True, "INACTIVE", "NOT_APPLICABLE")
        self.attempts += 1
        self._latest_probe = probe
        if probe.outcome == "ENVIRONMENT":
            return self._decision(
                False, "REJECTED", "REPRO_ENVIRONMENT",
                "The check failed during collection/import; environment failure is not a reproduction.",
                action="reproduce", probe=probe, code_hash=probe.code_hash)
        if probe.status == "RED":
            if not probe.first_error_line:
                return self._decision(
                    False, "REJECTED", "REPRO_ERROR_LINE_REQUIRED",
                    "A red run without its captured first-error line is not a ReproReceipt.",
                    action="reproduce", probe=probe)
            if probe.code_hash == UNKNOWN_REVISION:
                return self._decision(
                    False, "REJECTED", "REPRO_REVISION_REQUIRED",
                    "The red probe must be bound to a host-observed code hash.",
                    action="reproduce", probe=probe)
            if probe.outcome != "BEHAVIOR_FAIL" or probe.reached_observation is not True:
                return self._decision(
                    False, "REJECTED", "REPRO_TICKET_OBSERVATION_REQUIRED",
                    "A red check must reach and fail the admitted ticket observation.",
                    action="reproduce", probe=probe, code_hash=probe.code_hash)
            if self.receipt is None:
                receipt_id = _hash_bytes(_canonical({
                    "fingerprint": probe.fingerprint,
                    "code_hash": probe.code_hash,
                    "attempt": self.attempts,
                }))
                self.receipt = ReproReceipt(
                    schema=SCHEMA,
                    receipt_id=receipt_id,
                    probe_fingerprint=probe.fingerprint,
                    command=probe.command,
                    probe_id=probe.probe_id,
                    code_hash=probe.code_hash,
                    first_error_line=probe.first_error_line,
                    attempt=self.attempts,
                    outcome=probe.outcome,
                    reached_observation=probe.reached_observation is True,
                    failure_phase=probe.failure_phase,
                )
                self.witness_version += 1
                return self._decision(
                    True, "RED", "RED_CAPTURED",
                    "Host captured a discriminating red probe and its first-error line.",
                    action="reproduce", probe=probe, code_hash=probe.code_hash,
                    receipt=self.receipt)
            return self._decision(
                True, "RED", "RED_ALREADY_CAPTURED",
                "A ReproReceipt already exists; the original receipt remains authoritative.",
                action="reproduce", probe=probe, code_hash=probe.code_hash,
                receipt=self.receipt)
        if probe.status == "GREEN":
            self._latest_green = probe
            if self.receipt is None:
                return self._decision(
                    False, "REJECTED", "REPRO_NOT_DISCRIMINATING",
                    "A probe that is already green does not reproduce the reported bug.",
                    action="reproduce", probe=probe, code_hash=probe.code_hash)
            if probe.fingerprint != self.receipt.probe_fingerprint:
                return self._decision(
                    False, "REJECTED", "REPRO_PROBE_MISMATCH",
                    "Completion must rerun the same probe that produced the red receipt.",
                    action="verify", probe=probe, code_hash=probe.code_hash,
                    receipt=self.receipt)
            if probe.code_hash == self.receipt.code_hash:
                return self._decision(
                    False, "REJECTED", "REPRO_NOT_DISCRIMINATING",
                    "The same probe is still green on the receipt revision; it is not a red-to-green change.",
                    action="verify", probe=probe, code_hash=probe.code_hash,
                    receipt=self.receipt)
            return self._decision(
                True, "GREEN", "GREEN_CAPTURED",
                "The same host probe is green on a different revision.",
                action="verify", probe=probe, code_hash=probe.code_hash,
                receipt=self.receipt)
        code = "UNREPRODUCED_REQUIRED" if self.attempts >= self.max_attempts else "PROBE_NOT_RED"
        detail = ("The bounded reproduction budget is exhausted; file "
                  "UNREPRODUCED(risk) before partial completion." if code == "UNREPRODUCED_REQUIRED"
                  else "The host probe did not produce a discriminating red result.")
        return self._decision(False, "REJECTED", code, detail,
                              action="reproduce", probe=probe, code_hash=probe.code_hash)

    def admit_edit(self, path: object, *, code_hash: object = None,
                   is_test: bool = False) -> GateDecision:
        if not self.active or is_test:
            return GateDecision(True, "INACTIVE", "NOT_APPLICABLE")
        if self.edit_permit == "enforce" and self.exemption is None:
            return self._admit_edit_strict(path, code_hash=code_hash)
        # Advisory permit / typed exemption: evaluate the witness state for the journal only.
        emitter, self.journal = self.journal, None
        events_before = len(self.events)
        try:
            strict = self._admit_edit_strict(path, code_hash=code_hash)
        finally:
            self.journal = emitter
            del self.events[events_before:]
        if strict.accepted:
            return self._decision(True, strict.status, strict.code, strict.detail, action="edit",
                                  code_hash=code_hash, receipt=strict.receipt)
        self._edit_started = True
        suffix = "EXEMPT" if self.exemption is not None else "ADVISORY"
        return self._decision(
            True, "ADMITTED_UNWITNESSED", f"{strict.code}_{suffix}",
            f"Edit admitted without a current ReproReceipt ({strict.code}); "
            "a red-then-green reproduction still strengthens completion evidence.",
            action="edit", code_hash=code_hash, receipt=self.receipt)

    def _admit_edit_strict(self, path: object, *, code_hash: object = None) -> GateDecision:
        if not self.active:
            return GateDecision(True, "INACTIVE", "NOT_APPLICABLE")
        current = _revision(code_hash if code_hash is not None else self.revision)
        if self._unreproduced_risk is not None:
            return self._decision(
                False, "REJECTED", "UNREPRODUCED_REQUIRED",
                "No product edit is admitted after UNREPRODUCED(risk) without a new ticket.",
                action="edit", code_hash=current)
        if self.receipt is None:
            return self._decision(
                False, "REJECTED", "REPRO_REQUIRED",
                f"[repro-first-gate:REPRO_REQUIRED] Reproduce the typed {self.ticket_kind} "
                f"before editing product path {path}; capture a red first-error line.",
                action="edit", code_hash=current)
        if not self._edit_started and current != self.receipt.code_hash:
            return self._decision(
                False, "REJECTED", "REPRO_STALE",
                "The ReproReceipt is bound to a different current code hash; reproduce again before editing.",
                action="edit", code_hash=current, receipt=self.receipt)
        self._edit_started = True
        return self._decision(
            True, "ADMITTED", "REPRO_CONFIRMED",
            "The first product edit is admitted by the host ReproReceipt.",
            action="edit", code_hash=current, receipt=self.receipt)

    def refuse_edit(self, path: object, *, code_hash: object = None,
                    is_test: bool = False) -> str | None:
        decision = self.admit_edit(path, code_hash=code_hash, is_test=is_test)
        return None if decision.allowed else decision.detail

    def mark_unreproduced(self, risk: object) -> GateDecision:
        if not self.active:
            return GateDecision(True, "INACTIVE", "NOT_APPLICABLE")
        text = str(risk or "").strip()
        if self.receipt is not None:
            return self._decision(
                False, "REJECTED", "REPRO_ALREADY_CAPTURED",
                "UNREPRODUCED cannot replace an existing discriminating ReproReceipt.",
                action="unreproduced")
        if self.attempts < self.max_attempts:
            return self._decision(
                False, "REJECTED", "REPRO_ATTEMPTS_REMAIN",
                f"Run {self.max_attempts - self.attempts} more bounded reproduction attempt(s) before filing risk.",
                action="unreproduced")
        if not text:
            return self._decision(
                False, "REJECTED", "UNREPRODUCED_RISK_REQUIRED",
                "UNREPRODUCED(risk) requires a non-empty risk explanation.",
                action="unreproduced")
        self._unreproduced_risk = text[:1000]
        return self._decision(
            True, "PARTIAL", "UNREPRODUCED_FILED",
            "The host exhausted bounded attempts; completion is capped at partial.",
            action="unreproduced")

    def complete(self, probe: object = None, *, code_hash: object = None,
                 command: object = None, probe_id: object = None,
                 exit_code: object = None) -> GateDecision:
        if not self.active:
            return GateDecision(True, "COMPLETE", "NOT_APPLICABLE")
        if self.receipt is None and self.exemption is not None:
            return self._decision(
                True, "EXEMPT_UNVERIFIED", "REPRO_EXEMPT_UNVERIFIED",
                f"Typed exemption {self.exemption['kind']} authorized the work; the ticket is not verified.",
                action="complete")
        if self.receipt is None and self.edit_permit != "enforce":
            if self._unreproduced_risk is not None:
                return self._decision(
                    True, "PARTIAL", "UNREPRODUCED",
                    "UNREPRODUCED(risk) caps this ticket at partial completion.",
                    action="complete")
            return self._decision(
                True, "COMPLETE_UNWITNESSED", "REPRO_NOT_WITNESSED",
                "No reproduction witness was admitted; completion is accepted unwitnessed (advisory permit).",
                action="complete")
        if self._unreproduced_risk is not None:
            return self._decision(
                False, "PARTIAL", "UNREPRODUCED",
                "UNREPRODUCED(risk) caps this ticket at partial completion.",
                action="complete")
        if self.receipt is None:
            return self._decision(
                False, "REJECTED", "REPRO_REQUIRED",
                "Completion requires a discriminating red ReproReceipt first.",
                action="complete")
        candidate = self._latest_green if probe is None else ProbeResult.from_value(
            probe, command=command, probe_id=probe_id, code_hash=code_hash,
            exit_code=exit_code)
        if candidate is None or candidate.status != "GREEN":
            return self._decision(
                False, "REJECTED", "REPRO_GREEN_REQUIRED",
                "The same reproduction probe must be observed green at completion.",
                action="complete", code_hash=code_hash, receipt=self.receipt)
        current = _revision(code_hash if code_hash is not None else candidate.code_hash)
        if candidate.fingerprint != self.receipt.probe_fingerprint:
            return self._decision(
                False, "REJECTED", "REPRO_PROBE_MISMATCH",
                "Completion must use the exact command/probe identity from the red receipt.",
                action="complete", probe=candidate, code_hash=current,
                receipt=self.receipt)
        if candidate.code_hash != UNKNOWN_REVISION and candidate.code_hash != current:
            return self._decision(
                False, "REJECTED", "REPRO_CODE_HASH_MISMATCH",
                "The green result is not bound to the current completion code hash.",
                action="complete", probe=candidate, code_hash=current,
                receipt=self.receipt)
        if current == self.receipt.code_hash:
            return self._decision(
                False, "REJECTED", "REPRO_NOT_DISCRIMINATING",
                "The green result is bound to the same revision as the red receipt.",
                action="complete", probe=candidate, code_hash=current,
                receipt=self.receipt)
        return self._decision(
            True, "COMPLETE", "GREEN_CONFIRMED",
            "The same host reproduction probe is green at the current code hash.",
            action="complete", probe=candidate, code_hash=current,
            receipt=self.receipt)


__all__ = [
    "DEFAULT_MAX_ATTEMPTS", "GateDecision", "MODE_ENV", "ProbeResult",
    "ReproFirstGate", "ReproReceipt", "SCHEMA", "code_hash", "enabled",
    "product_edit_allowed", "requires_repro", "repro_first_gate_mode",
]

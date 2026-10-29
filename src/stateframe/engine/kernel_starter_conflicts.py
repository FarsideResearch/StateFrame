"""Host-owned detection of starter behavior that contradicts a public ticket.

The ordinary host-check path answers whether the current checkout satisfies a
frozen check.  This module preserves the useful *starting* observation when a
ticket-grounded check already fails in the starter checkout.  It is deliberately
small and data driven:

* only ``oracle_basis=TICKET`` checks are candidates;
* the clause span and every asserted expected literal must occur verbatim in
  the supplied public ticket;
* only an executed counterexample is a contradiction; setup, binding, and
  harness failures are not facts about the product;
* the baseline facts are immutable, while current facts are refreshed against
  the current checkout after edits.

The runtime owns journal and frame wiring.  Pure functions and the tracker are
kept here so synthetic ticket/starter tests never need an evaluation task.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path


MODE_ENV = "STATEFRAME_STARTER_CONFLICTS"
SCHEMA = "stateframe-starter-conflict/1"
FACT_TYPE = "STARTER_CONFLICT"
PREDICATE = "starter_contradicts_ticket"
CONFLICT = "CONFLICT"

_COMMON_EXPECTED_KEYS = frozenset({
    "status", "body", "data", "output", "result", "response", "value",
    "error", "errors", "message", "detail", "code", "type", "ok",
    "status_code", "http_status", "http_code",
    "contains", "error_contains", "preserved", "outputs", "path", "paths",
    "anchor", "delta", "total_path", "component_paths", "cases", "calls",
})
_STATUS_KEYS = frozenset({"status", "status_code", "http_status", "http_code"})
_IGNORED_PATH_KEYS = frozenset({
    "path", "paths", "anchor", "total_path", "component_paths", "outputs",
    "cases", "calls", "input_index", "args", "kwargs",
})


def mode() -> str:
    """Return the strictly validated starter-conflict switch."""
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _copy(value: object) -> object:
    try:
        return copy.deepcopy(value)
    except (TypeError, ValueError):
        return str(value)


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _text(value: object, limit: int = 800) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)] + "..."


def _ticket_text(ticket: object) -> str:
    if isinstance(ticket, str):
        return ticket
    if isinstance(ticket, Mapping):
        for key in ("text", "content", "brief", "ticket", "body", "source"):
            value = ticket.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _spec_for(check: object, specs: object) -> object | None:
    if specs is None:
        return None
    obligation = str(_field(check, "obligation_id", _field(check, "obligation", "")))
    if isinstance(specs, Mapping):
        candidate = specs.get(obligation)
        if candidate is not None:
            return candidate
        values = specs.values()
    elif isinstance(specs, Iterable) and not isinstance(specs, (str, bytes, bytearray)):
        values = specs
    else:
        return None
    for candidate in values:
        if str(_field(candidate, "obligation_id", _field(candidate, "id", ""))) == obligation:
            return candidate
    return None


def clause_text(check: object, specs: object = None) -> str:
    """Return the exact public clause span attached to a generated check."""
    provenance = _field(check, "provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    for candidate in (
        provenance.get("clause_span"), provenance.get("clause"),
        _field(_spec_for(check, specs), "clause_span", None),
        _field(_spec_for(check, specs), "clause", None),
        _field(check, "clause", None),
    ):
        if isinstance(candidate, Mapping):
            candidate = candidate.get("text", candidate.get("quote", ""))
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return ""


def _iter_expected_literals(value: object, path: tuple[str, ...] = ()) -> Iterable[tuple[str, object]]:
    """Yield assertion leaves and task-specific mapping keys.

    Structural JSON names are not product literals: a ticket saying "400
    response" need not contain the implementation vocabulary ``status`` or
    ``body``.  Product names, error text, units, and scalar expected values
    are retained and must be grounded verbatim.
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text not in _COMMON_EXPECTED_KEYS and key_text not in _IGNORED_PATH_KEYS:
                yield (".".join((*path, "<key>")), key_text)
            yield from _iter_expected_literals(child, (*path, key_text))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _iter_expected_literals(child, (*path, str(index)))
        return
    if path and path[-1] in _IGNORED_PATH_KEYS:
        return
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (str, int, float)):
        literal = str(value)
        if literal:
            yield (".".join(path), value)


def asserted_literals(expected: object) -> tuple[str, ...]:
    """Return unique scalar/key literals that a check asserts."""
    result: list[str] = []
    for _, value in _iter_expected_literals(expected):
        text = str(value)
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _literal_forms(value: str) -> tuple[str, ...]:
    # Ticket grounding is intentionally verbatim.  In particular, ``4`` is
    # not grounded by the larger token ``400``; do not normalize numeric text.
    return (value,)


def _literal_in_ticket(literal: str, public_ticket: str) -> bool:
    if re.fullmatch(r"-?(?:\d+(?:\.\d*)?|\.\d+)", literal):
        pattern = rf"(?<![A-Za-z0-9_.]){re.escape(literal)}(?![A-Za-z0-9_.])"
        return re.search(pattern, public_ticket) is not None
    return literal in public_ticket


def ticket_grounding(check: object, ticket: object, specs: object = None) -> "TicketGrounding":
    """Check the conservative ticket-only admission rule for one definition."""
    public_ticket = _ticket_text(ticket)
    clause = clause_text(check, specs)
    reasons: list[str] = []
    if str(_field(check, "oracle_basis", "")).upper() != "TICKET":
        reasons.append("ORACLE_BASIS_NOT_TICKET")
    if str(_field(check, "interpretation", "VALIDATED")).upper() != "VALIDATED":
        reasons.append("CHECK_NOT_VALIDATED")
    if not public_ticket:
        reasons.append("MISSING_PUBLIC_TICKET")
    if not clause:
        reasons.append("MISSING_CLAUSE_SPAN")
    elif clause not in public_ticket:
        reasons.append("CLAUSE_NOT_VERBATIM_IN_TICKET")
    missing = tuple(literal for literal in asserted_literals(_field(check, "expected"))
                    if not any(_literal_in_ticket(form, public_ticket)
                               for form in _literal_forms(literal)))
    if missing:
        reasons.append("ASSERTED_LITERAL_NOT_IN_TICKET")
    return TicketGrounding(
        eligible=not reasons,
        clause=clause,
        literals=asserted_literals(_field(check, "expected")),
        missing_literals=missing,
        reason="OK" if not reasons else "+".join(reasons),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class TicketGrounding:
    eligible: bool
    clause: str
    literals: tuple[str, ...]
    missing_literals: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "eligible": self.eligible,
            "clause": self.clause,
            "asserted_literals": list(self.literals),
            "missing_literals": list(self.missing_literals),
            "reason": self.reason,
        }


def _receipt_status(receipt: object) -> str:
    candidates = (
        _field(receipt, "underlying_status"), _field(receipt, "underlying_result"),
        _field(receipt, "status"), _field(receipt, "result"),
        _field(receipt, "result_class"),
    )
    for candidate in candidates:
        status = str(candidate or "").upper()
        if status and status != "REUSED":
            return status
    return ""


def _receipt_execution(receipt: object) -> str:
    value = _field(receipt, "execution", None)
    return str(value or "EXECUTED").upper()


def _executed_counterexample(receipt: object) -> bool:
    return (_receipt_status(receipt) == "COUNTEREXAMPLE"
            and _receipt_execution(receipt) == "EXECUTED")


def _receipt_value(receipt: object, key: str, fallback: object = None) -> object:
    value = _field(receipt, key, None)
    if value is not None:
        return value
    counterexample = _field(receipt, "counterexample", None)
    if isinstance(counterexample, Mapping):
        value = counterexample.get(key)
        if value is not None:
            return value
    return fallback


def _find_key(value: object, keys: frozenset[str]) -> object:
    if isinstance(value, Mapping):
        for key in keys:
            if key in value and not isinstance(value[key], Mapping) and not (
                    isinstance(value[key], Sequence)
                    and not isinstance(value[key], (str, bytes, bytearray))):
                return value[key]
        for child in value.values():
            found = _find_key(child, keys)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found = _find_key(child, keys)
            if found is not None:
                return found
    return None


def _status_pair(expected: object, observed: object) -> tuple[object, object]:
    return _find_key(expected, _STATUS_KEYS), _find_key(observed, _STATUS_KEYS)


def _render_value(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _subject_from_clause(clause: str) -> str:
    clean = clause.replace("`", "")
    lower = clean.lower()
    for phrase in ("validation errors", "validation error", "error handler", "unknown speeds",
                   "unknown speed"):
        if phrase in lower:
            return phrase
    # Keep the public wording's subject when it has a conventional status tail.
    match = re.match(
        r"(.+?)\s+(?:must|should)\s+(?:return|respond(?:\s+with)?|be)\b",
        clean, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.match(r"(.+?)\s+as\s+(?:an?\s+)?(?:HTTP\s+)?\d+\b",
                     clean, flags=re.IGNORECASE)
    if match:
        subject = match.group(1).strip()
        return subject[7:] if subject.lower().startswith("reject ") else subject
    return "the checked behavior"


def _behavior_text(clause: str, expected: object, observed: object) -> tuple[str, str]:
    expected_status, observed_status = _status_pair(expected, observed)
    if observed_status is not None:
        subject = _subject_from_clause(clause)
        return (f"{subject} return HTTP {observed_status}",
                f"HTTP {expected_status}" if expected_status is not None
                else _render_value(expected))
    return (f"the checked behavior returned {_render_value(observed)}",
            _render_value(expected))


def _provenance_location(check: object) -> tuple[str | None, int | None]:
    provenance = _field(check, "provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    for key in ("location", "source_location"):
        value = provenance.get(key)
        if isinstance(value, str) and ":" in value:
            path, _, line = value.rpartition(":")
            if line.isdigit():
                return path, int(line)
    starter = provenance.get("starter")
    if isinstance(starter, Mapping):
        path = starter.get("path", starter.get("file"))
        line = starter.get("line")
        if isinstance(line, Sequence) and not isinstance(line, (str, bytes)):
            line = line[0] if line else None
        return (str(path) if path else None,
                int(line) if str(line).isdigit() else None)
    return None, None


def _entry_source(root: Path, entry_point: object) -> tuple[Path | None, str | None]:
    if not isinstance(entry_point, str) or ":" not in entry_point:
        return None, None
    module, attribute = entry_point.split(":", 1)
    module_path = root.joinpath(*module.split("."))
    source = module_path.with_suffix(".py")
    if not source.is_file():
        source = module_path / "__init__.py"
    return (source if source.is_file() else None,
            attribute.rsplit(".", 1)[-1] if attribute else None)


def _source_location(root: Path | None, check: object, observed: object) -> str:
    explicit_path, explicit_line = _provenance_location(check)
    source: Path | None = None
    name: str | None = None
    if root is not None:
        source, name = _entry_source(root, _field(check, "entry_point"))
    if source is not None and name:
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            functions = [node for node in ast.walk(tree)
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and node.name == name]
            observed_status = _find_key(observed, _STATUS_KEYS)
            for function in functions:
                matching_lines: list[int] = []
                for node in ast.walk(function):
                    if not isinstance(node, ast.Return):
                        continue
                    status = _find_key(_ast_literal(node.value), _STATUS_KEYS)
                    if status is not None and str(status) == str(observed_status):
                        matching_lines.append(node.lineno)
                if matching_lines:
                    return f"{source.relative_to(root).as_posix()}:{min(matching_lines)}"
                if functions:
                    return f"{source.relative_to(root).as_posix()}:{function.lineno}"
        except (OSError, UnicodeError, SyntaxError, ValueError):
            pass
    if explicit_path and explicit_line is not None:
        path = explicit_path.replace("\\", "/")
        return f"{path}:{explicit_line}"
    if explicit_path:
        return f"{explicit_path}:1"
    if source is not None and root is not None:
        try:
            return f"{source.relative_to(root).as_posix()}:1"
        except ValueError:
            return f"{source.name}:1"
    entry = str(_field(check, "entry_point", "starter"))
    return entry.replace(":", "/") + ":1"


def _ast_literal(node: ast.AST | None) -> object:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        if isinstance(node, ast.Dict):
            result = {}
            for key, value in zip(node.keys, node.values):
                try:
                    key_value = ast.literal_eval(key)
                except (ValueError, TypeError, SyntaxError):
                    continue
                result[key_value] = _ast_literal(value)
            return result
        if isinstance(node, ast.List):
            return [_ast_literal(item) for item in node.elts]
    return None


@dataclasses.dataclass(frozen=True, slots=True)
class StarterConflictFact:
    check_id: str
    obligation_id: str
    clause: str
    location: str
    observed: object
    expected: object
    observed_behavior: str
    expected_behavior: str
    message: str
    starter_snapshot: str
    revision: str | None = None
    grounding: TicketGrounding | None = None
    evidence_id: str = ""
    fact_type: str = FACT_TYPE
    kind: str = "CONTRACT"
    status: str = CONFLICT
    predicate: str = PREDICATE

    @property
    def fact_id(self) -> str:
        return digest({
            "fact_type": self.fact_type, "check_id": self.check_id,
            "obligation_id": self.obligation_id, "clause": self.clause,
            "location": self.location, "observed": self.observed,
            "expected": self.expected,
        })

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "StarterConflictFact":
        """Restore a fact carried by a resumed journal event."""
        nested = value.get("value") if isinstance(value.get("value"), Mapping) else value
        grounding_raw = nested.get("grounding") if isinstance(nested, Mapping) else None
        grounding = (TicketGrounding(
            bool(grounding_raw.get("eligible", True)),
            str(grounding_raw.get("clause", value.get("clause", ""))),
            tuple(str(item) for item in grounding_raw.get("asserted_literals", ())),
            tuple(str(item) for item in grounding_raw.get("missing_literals", ())),
            str(grounding_raw.get("reason", "RESTORED")),
        ) if isinstance(grounding_raw, Mapping) else None)
        return cls(
            check_id=str(value.get("check_id", nested.get("check_id", ""))),
            obligation_id=str(value.get("obligation_id", nested.get("obligation_id", ""))),
            clause=str(value.get("clause", nested.get("ticket_clause", ""))),
            location=str(value.get("location", nested.get("location", "unknown:1"))),
            observed=_copy(nested.get("observed")),
            expected=_copy(nested.get("expected")),
            observed_behavior=str(nested.get("observed_behavior", "the checked behavior")),
            expected_behavior=str(nested.get("expected_behavior", "the ticket requirement")),
            message=str(value.get("message", nested.get("message", ""))),
            starter_snapshot=str(value.get("starter_snapshot", nested.get("starter_snapshot", ""))),
            revision=value.get("revision", nested.get("revision")),
            grounding=grounding,
            evidence_id=str(nested.get("evidence_id", value.get("evidence_id", ""))),
            fact_type=str(value.get("fact_type", FACT_TYPE)),
            kind=str(value.get("kind", "CONTRACT")),
            status=str(value.get("status", CONFLICT)),
            predicate=str(value.get("predicate", PREDICATE)),
        )

    def to_dict(self) -> dict[str, object]:
        value = {
            "fact_type": self.fact_type,
            "message": self.message,
            "check_id": self.check_id,
            "obligation_id": self.obligation_id,
            "observed_behavior": self.observed_behavior,
            "expected_behavior": self.expected_behavior,
            "observed": _copy(self.observed),
            "expected": _copy(self.expected),
            "ticket_clause": self.clause,
            "location": self.location,
            "starter_snapshot": self.starter_snapshot,
            "revision": self.revision,
            "evidence_id": self.evidence_id or "starter-conflict:" + self.fact_id,
            "grounding": self.grounding.to_dict() if self.grounding else None,
        }
        try:
            from kernel_host_facts import HostFact

            host = HostFact(
                kind=self.kind, predicate=self.predicate, subject=self.check_id,
                value=value,
                scope={"ticket_clause": self.clause, "location": self.location,
                       "starter_snapshot": self.starter_snapshot},
                evidence_ids=(value["evidence_id"],),
                dependency_fingerprint=digest({"starter_snapshot": self.starter_snapshot,
                                               "check_id": self.check_id,
                                               "clause": self.clause}),
                status=self.status, revision=self.revision,
            )
            result = host.to_dict()
        except (ImportError, TypeError, ValueError):
            result = {
                "schema": "stateframe-host-facts/1",
                "fact_id": self.fact_id,
                "kind": self.kind,
                "predicate": self.predicate,
                "subject": self.check_id,
                "value": value,
                "scope": {"ticket_clause": self.clause, "location": self.location},
                "evidence_ids": [value["evidence_id"]],
                "status": self.status,
                "revision": self.revision,
            }
        result.update({
            # Preserve the host-facts envelope when this is projected into
            # ``frame["host_facts"]``; the starter-specific schema is an
            # additive discriminator rather than a replacement envelope.
            "starter_conflict_schema": SCHEMA,
            "fact_type": self.fact_type,
            "check_id": self.check_id,
            "obligation_id": self.obligation_id,
            "clause": self.clause,
            "location": self.location,
            "message": self.message,
            "starter_snapshot": self.starter_snapshot,
            "revision": self.revision,
        })
        return result


def _fact_for(check: object, receipt: object, grounding: TicketGrounding, *, root: Path | None,
              starter_snapshot: str, revision: str | None) -> StarterConflictFact:
    observed = _receipt_value(receipt, "observed", _field(receipt, "observed_values", None))
    expected = _field(check, "expected", _receipt_value(receipt, "expected"))
    observed_behavior, expected_behavior = _behavior_text(grounding.clause, expected, observed)
    location = _source_location(root, check, observed)
    message = (
        f"Current code {observed_behavior} here ({location}); "
        f"ticket clause \"{grounding.clause}\" requires {expected_behavior}; "
        "this must change."
    )
    evidence_id = str(_field(receipt, "receipt_id", "") or "")
    if not evidence_id:
        evidence_id = "starter-conflict:" + digest({
            "check_id": _field(check, "check_id", ""),
            "snapshot": starter_snapshot, "observed": observed,
        })
    return StarterConflictFact(
        check_id=str(_field(check, "check_id", "")),
        obligation_id=str(_field(check, "obligation_id", _field(check, "obligation", ""))),
        clause=grounding.clause,
        location=location,
        observed=_copy(observed), expected=_copy(expected),
        observed_behavior=observed_behavior, expected_behavior=expected_behavior,
        message=message, starter_snapshot=starter_snapshot, revision=revision,
        grounding=grounding, evidence_id=evidence_id,
    )


def detect_starter_conflicts(
    ticket: object,
    checks: Sequence[object],
    receipts: Iterable[object],
    *,
    root: str | Path | None = None,
    snapshot: object = "starter",
    revision: object = None,
    specs: object = None,
) -> tuple[StarterConflictFact, ...]:
    """Return grounded contradictions proved by executed starter receipts."""
    root_path = Path(root).resolve() if root is not None else None
    starter_snapshot = str(snapshot)
    by_id: dict[str, object] = {}
    for receipt in receipts or ():
        check_id = str(_field(receipt, "check_id", _field(receipt, "check", "")))
        if check_id:
            by_id[check_id] = receipt
    facts: list[StarterConflictFact] = []
    for check in checks or ():
        check_id = str(_field(check, "check_id", _field(check, "id", "")))
        receipt = by_id.get(check_id)
        if receipt is None or not _executed_counterexample(receipt):
            continue
        grounding = ticket_grounding(check, ticket, specs)
        if not grounding.eligible:
            continue
        facts.append(_fact_for(
            check, receipt, grounding, root=root_path,
            starter_snapshot=starter_snapshot,
            revision=None if revision is None else str(revision),
        ))
    return tuple(facts)


def workspace_snapshot(root: str | Path, *, binding: object = None) -> str:
    """Hash the visible starter/current checkout without cache noise."""
    root = Path(root).resolve()
    rows: list[dict[str, object]] = []
    ignored = {".git", "__pycache__", ".pytest_cache", ".test-tmp"}
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if any(part in ignored for part in relative.parts):
                continue
            try:
                content = path.read_bytes()
            except OSError:
                continue
            rows.append({"path": relative.as_posix(),
                         "sha256": hashlib.sha256(content).hexdigest(),
                         "size": len(content)})
    return digest({"files": rows, "binding": binding})


def _call_runner(runner: Callable, root: Path, checks: Sequence[object], **kwargs) -> tuple[object, ...]:
    try:
        result = runner(root, checks, **kwargs)
    except TypeError:
        # Synthetic runners commonly expose only ``(root, checks)``.  This
        # adapter does not weaken the receipt rule; it only keeps the pure API
        # convenient for offline tests.
        result = runner(root, checks)
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes, bytearray, Mapping)):
        return tuple(result)
    return (result,) if result is not None else ()


class StarterConflictTracker:
    """Immutable starter evidence plus revision-bound current contradiction state."""

    def __init__(self, ticket: object, checks: Sequence[object], *,
                 starter_snapshot: object, starter_facts: Sequence[StarterConflictFact],
                 current_facts: Sequence[StarterConflictFact] | None = None,
                 specs: object = None, root: str | Path | None = None,
                 emit: Callable | object | None = None, task_ref: object = "") -> None:
        self.ticket = _ticket_text(ticket)
        self.checks = tuple(checks or ())
        self.specs = specs
        self.root = Path(root).resolve() if root is not None else None
        self.starter_snapshot = str(starter_snapshot)
        self.starter_facts = tuple(starter_facts)
        self.facts = tuple(current_facts if current_facts is not None else starter_facts)
        self.current_snapshot = self.starter_snapshot
        self.task_ref = str(task_ref or "")
        self._emit_callback = emit
        self._fired: set[str] = set()
        self._last_cleared: set[str] = set()
        self.starter_receipts: tuple[object, ...] = ()
        self.current_receipts: tuple[object, ...] = ()
        self._announce(self.facts, initial=True)

    @classmethod
    def from_receipts(cls, ticket: object, checks: Sequence[object], receipts: Iterable[object], *,
                      root: str | Path | None = None, snapshot: object = "starter",
                      revision: object = None, specs: object = None,
                      emit: Callable | object | None = None, task_ref: object = "") -> "StarterConflictTracker":
        materialized = tuple(receipts or ())
        facts = detect_starter_conflicts(
            ticket, checks, materialized, root=root, snapshot=snapshot,
            revision=revision, specs=specs)
        tracker = cls(ticket, checks, starter_snapshot=snapshot,
                      starter_facts=facts, specs=specs, root=root,
                      emit=emit, task_ref=task_ref)
        tracker.starter_receipts = materialized
        tracker.current_receipts = tracker.starter_receipts
        return tracker

    @classmethod
    def from_starter(cls, *args, **kwargs) -> "StarterConflictTracker":
        return cls.from_receipts(*args, **kwargs)

    def _emit(self, kind: str, **fields: object) -> None:
        target = self._emit_callback
        if target is None:
            return
        emitter = target if callable(target) else getattr(target, "emit", None)
        if callable(emitter):
            emitter(kind, **fields)

    def _announce(self, facts: Sequence[StarterConflictFact], *, initial: bool = False) -> None:
        current_ids = {fact.fact_id for fact in facts}
        for fact in facts:
            if fact.fact_id in self._fired:
                continue
            self._fired.add(fact.fact_id)
            self._emit(
                "starter_conflict_fired", producer="kernel_starter_conflicts",
                task_ref=self.task_ref, check_id=fact.check_id,
                obligation_id=fact.obligation_id, fact=fact.to_dict(),
                message=fact.message, location=fact.location,
                starter_snapshot=fact.starter_snapshot,
                current_revision=fact.revision,
                initial=initial, dispatch=False,
            )
        cleared = self._last_cleared - current_ids
        for fact_id in sorted(cleared):
            self._emit("starter_conflict_cleared", producer="kernel_starter_conflicts",
                       task_ref=self.task_ref, fact_id=fact_id,
                       starter_snapshot=self.starter_snapshot, dispatch=False)
        self._last_cleared = current_ids

    def refresh(self, receipts: Iterable[object], *, snapshot: object = None,
                revision: object = None) -> tuple[StarterConflictFact, ...]:
        """Reduce current receipts; the starter facts are never recomputed."""
        current_snapshot = self.current_snapshot if snapshot is None else str(snapshot)
        materialized = tuple(receipts or ())
        current = detect_starter_conflicts(
            self.ticket, self.checks, materialized, root=self.root,
            snapshot=self.starter_snapshot, revision=(current_snapshot if revision is None else revision),
            specs=self.specs)
        previous_ids = {fact.fact_id for fact in self.facts}
        self.current_receipts = materialized
        self.current_snapshot = current_snapshot
        self.facts = tuple(current)
        self._last_cleared = previous_ids
        self._announce(self.facts)
        return self.facts

    def refresh_from_root(self, root: str | Path | None = None, *, runner: Callable | None = None,
                          environment: object = None, cache: dict | None = None,
                          binding: object = None, force: bool = False) -> tuple[StarterConflictFact, ...]:
        """Run the frozen checks against the current checkout after an edit."""
        root_path = Path(root or self.root or ".").resolve()
        current_snapshot = workspace_snapshot(root_path, binding=binding)
        if not force and current_snapshot == self.current_snapshot:
            return self.facts
        # Missing ticket/bundle evidence cannot prove that an existing
        # contradiction was repaired.  Keep the immutable fact until the
        # same frozen check is available to produce a passing receipt.
        if not self.ticket or not self.checks:
            self.current_snapshot = current_snapshot
            return self.facts
        if runner is None:
            try:
                from kernel_obligation_assess import run_registered_checks
                runner = run_registered_checks
            except ImportError:
                return self.facts
        kwargs = {"snapshot": current_snapshot, "environment": environment, "cache": cache}
        receipts = _call_runner(runner, root_path, self.checks, **kwargs)
        return self.refresh(receipts, snapshot=current_snapshot, revision=current_snapshot)

    def frame_facts(self) -> tuple[dict[str, object], ...]:
        return tuple(fact.to_dict() for fact in self.facts)


def _fact_rows(facts: Sequence[StarterConflictFact]) -> list[dict[str, object]]:
    return [fact.to_dict() for fact in facts]


def project_frame(frame: Mapping[str, object], facts: Sequence[StarterConflictFact], *,
                  requirements_ledger: object = None) -> dict[str, object]:
    """Project current typed facts into a frame and any available ledger."""
    stale_keys = {"starter_conflicts", "starter_conflict_facts"}.intersection(frame)
    if not facts and requirements_ledger is None and not stale_keys:
        return frame if isinstance(frame, dict) else dict(frame)
    result = dict(frame)
    rows = _fact_rows(tuple(facts))
    result["starter_conflicts"] = rows
    result["starter_conflict_facts"] = rows
    existing = result.get("host_facts", ())
    existing_rows = list(existing) if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)) else []
    existing_rows = [row for row in existing_rows if not _is_starter_fact_row(row)]
    if rows:
        seen = {str(row.get("fact_id", "")) for row in existing_rows if isinstance(row, Mapping)}
        result["host_facts"] = [*existing_rows, *[row for row in rows
                                                    if str(row.get("fact_id", "")) not in seen]]
    elif "host_facts" in result:
        # A prior frame may be used as the projection input by a live adapter.
        # Do not let a repaired starter contradiction survive that projection.
        result["host_facts"] = existing_rows
    ledger = requirements_ledger if requirements_ledger is not None else result.get("requirements_ledger")
    if ledger is not None:
        updated = attach_to_requirements_ledger(ledger, tuple(facts))
        if updated is not None:
            result["requirements_ledger"] = updated
    return result


def _is_starter_fact_row(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("starter_conflict_schema") == SCHEMA:
        return True
    if value.get("fact_type") == FACT_TYPE or value.get("predicate") == PREDICATE:
        return True
    nested = value.get("value")
    return isinstance(nested, Mapping) and nested.get("fact_type") == FACT_TYPE


def attach_to_requirements_ledger(ledger: object, facts: Sequence[StarterConflictFact]) -> object:
    """Attach facts without inventing a ledger schema when another lane supplies it."""
    rows = _fact_rows(tuple(facts))
    if isinstance(ledger, Mapping):
        result = copy.deepcopy(dict(ledger))
        existing = result.get("starter_conflicts", ())
        prior = list(existing) if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)) else []
        prior = [row for row in prior if not _is_starter_fact_row(row)]
        # This ledger is the current projection for this task.  Replacing the
        # module-owned rows makes a repaired contradiction disappear instead
        # of leaving a stale host fact behind.
        result["starter_conflicts"] = [*prior, *rows]
        for key in ("requirements", "entries", "contracts"):
            items = result.get(key)
            if not isinstance(items, list):
                continue
            updated_items = []
            for item in items:
                if not isinstance(item, Mapping):
                    updated_items.append(item)
                    continue
                item_copy = copy.deepcopy(dict(item))
                obligation = str(item_copy.get("obligation_id", item_copy.get("id", "")))
                matching = [row for row in rows if str(row.get("obligation_id", "")) == obligation]
                if matching:
                    item_copy["starter_conflicts"] = matching
                elif "starter_conflicts" in item_copy:
                    prior_item = item_copy.get("starter_conflicts")
                    if (isinstance(prior_item, Sequence)
                            and not isinstance(prior_item, (str, bytes, bytearray))):
                        item_copy["starter_conflicts"] = [
                            row for row in prior_item
                            if not _is_starter_fact_row(row)
                        ]
                updated_items.append(item_copy)
            result[key] = updated_items
        return result
    for name in ("record_starter_conflicts", "attach_starter_conflicts", "record_host_facts"):
        method = getattr(ledger, name, None)
        if callable(method):
            try:
                returned = method(rows)
            except (TypeError, ValueError):
                continue
            return ledger if returned is None else returned
    return ledger


detect_conflicts = detect_starter_conflicts
compute_starter_conflicts = detect_starter_conflicts
ground_ticket_check = ticket_grounding


__all__ = [
    "CONFLICT", "FACT_TYPE", "MODE_ENV", "PREDICATE", "SCHEMA",
    "StarterConflictFact", "StarterConflictTracker", "TicketGrounding",
    "asserted_literals", "attach_to_requirements_ledger", "compute_starter_conflicts",
    "clause_text", "detect_conflicts", "detect_starter_conflicts", "digest", "enabled",
    "ground_ticket_check", "mode", "project_frame", "ticket_grounding", "workspace_snapshot",
]

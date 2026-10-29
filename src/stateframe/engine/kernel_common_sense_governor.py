"""Host-fact action governor for the host facts common-sense lane.

The governor is deliberately small at the decision boundary: it consumes
typed runtime facts, returns legal action options, and renders at most one
typed control line. It never certifies, completes, or waives an obligation.
The rule catalogue is data so every rule has an explicit owner, dependency,
effect, and unknown policy instead of becoming another prose advisor.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from kernel_done_acceptance import done_accepted


MODE_ENV = "STATEFRAME_COMMON_SENSE_GOVERNOR"
SCHEMA = "stateframe-common-sense-governor/1"
MAX_CLAUSE_CHARS = 360
MAX_SCOPE_ITEMS = 8

_TRUTHY = frozenset({"1", "on", "true", "yes"})
_SATISFIED = frozenset({"PASS", "PASSED", "SATISFIED", "CLOSED", "COMPLETE"})
_UNKNOWN = frozenset({
    "UNKNOWN", "MISSING_CHECKS", "NO_VERIFIER", "REQUIRED_EVIDENCE_GAP",
    "CLAUSE_COVERAGE", "COVERAGE_OPEN", "INCONCLUSIVE", "UNVERIFIED",
    "NOT_EXECUTED", "STALE", "UNBOUND_OBSERVABLE",
})
_EDIT_KINDS = frozenset({"edit", "str_replace", "file_create", "work"})
_OBSERVATION_KINDS = frozenset({"read", "edit", "str_replace", "file_create", "done"})
_NO_OP_RESULTS = frozenset({"NO_OP", "NOOP", "EDIT_NOOP", "IDENTICAL"})
_NO_OP_TEXT = re.compile(r"\b(?:NO[- _]?OP|IDENTICAL)\b", re.IGNORECASE)
_CONCRETE_BEHAVIOR = re.compile(
    r"\b(?:behavior|behaviour|calculate|compute|return|returns|raise|raises|valid|invalid|"
    r"must|should|preserve|thread|handle|handles|support|supports|implement|implemented|"
    r"status|http|charge|language|field|endpoint|call|accept|reject)\b",
    re.IGNORECASE,
)
_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:py|js|jsx|ts|tsx|json|yaml|yml|toml|md)(?::\d+(?:-\d+)?)?"
)
_SYMBOL_RE = re.compile(
    r"\b(?:function|method|symbol|class|entrypoint|endpoint)\s+([A-Za-z_][A-Za-z0-9_.]*)",
    re.IGNORECASE,
)


def _text(value: object, limit: int = 1000) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())[:limit]


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            result = converter()
        except (TypeError, ValueError):
            return None
        return result if isinstance(result, Mapping) else None
    return None


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def enabled(runtime: object | None = None) -> bool:
    value = getattr(runtime, "stateframe_common_sense_governor_mode", None)
    if value is None:
        value = os.environ.get(MODE_ENV, "off")
    return str(value).strip().lower() in _TRUTHY


def mode(environ: Mapping[str, str] | None = None) -> str:
    value = (os.environ if environ is None else environ).get(MODE_ENV, "off")
    normalized = str(value or "off").strip().lower()
    if normalized not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return normalized


@dataclass(frozen=True, slots=True)
class RuleSpec:
    """Typed rule metadata; effects are restricted to non-authoritative controls."""

    rule_id: str
    version: str
    family: str
    subscribed_events: tuple[str, ...]
    required_evidence: tuple[str, ...]
    dependency_paths: tuple[str, ...]
    effects: tuple[str, ...]
    unknown_policy: str = "ABSTAIN"
    deployment: str = "ENFORCE"

    def __post_init__(self) -> None:
        if not self.rule_id or not self.version or not self.family:
            raise ValueError("common-sense rule identity is required")
        if self.unknown_policy not in {"ABSTAIN", "HOST_BLOCK", "RELEASE"}:
            raise ValueError("unknown policy must be ABSTAIN, HOST_BLOCK, or RELEASE")
        if self.deployment not in {"SHADOW", "ENFORCE"}:
            raise ValueError("deployment must be SHADOW or ENFORCE")
        forbidden = {"SET_COMPLETE", "SET_TEST_PASSED", "WAIVE_OBLIGATION"}
        if forbidden.intersection(self.effects):
            raise ValueError("common-sense rules cannot certify or waive requirements")


def _rule(rule_id: str, family: str, *, evidence: tuple[str, ...], paths: tuple[str, ...],
          effects: tuple[str, ...], events: tuple[str, ...] = ("completion",),
          unknown: str = "ABSTAIN") -> RuleSpec:
    return RuleSpec(
        rule_id=rule_id,
        version="1",
        family=family,
        subscribed_events=events,
        required_evidence=evidence,
        dependency_paths=paths,
        effects=effects,
        unknown_policy=unknown,
    )


# The first four rule families address the museum failure.  The remaining
# entries are universal templates whose host inputs exist in v46 (receipts,
# bindings, action menus, budgets, and completion ownership).
RULE_SPECS: tuple[RuleSpec, ...] = (
    _rule("CSG-E01", "EVIDENCE", evidence=("current_public_verdict", "open_clause"), paths=("assessment", "test_receipt"), effects=("RequireActionInput", "RouteExecution")),
    _rule("CSG-E02", "EVIDENCE", evidence=("receipt_scope", "current_binding"), paths=("assessment", "binding", "receipt"), effects=("RequireActionInput",)),
    _rule("CSG-E03", "EVIDENCE", evidence=("check_status",), paths=("assessment", "test_receipt"), effects=("RequireActionInput",)),
    _rule("CSG-E04", "EVIDENCE", evidence=("host_fact",), paths=("assessment",), effects=("RecordIncident",)),
    _rule("CSG-S01", "SCOPE", evidence=("clause_text", "revision_diff"), paths=("assessment", "binding", "diff"), effects=("RequireActionInput", "OfferAction")),
    _rule("CSG-S02", "SCOPE", evidence=("named_surface", "revision_diff"), paths=("assessment", "diff"), effects=("OfferAction",)),
    _rule("CSG-S03", "SCOPE", evidence=("named_file", "current_file_set"), paths=("assessment", "binding"), effects=("OfferAction",)),
    _rule("CSG-S04", "SCOPE", evidence=("named_symbol", "revision_diff"), paths=("assessment", "diff"), effects=("OfferAction",)),
    _rule("CSG-T01", "STOPPING", evidence=("done_attempt", "open_clause"), paths=("completion", "assessment"), effects=("RestrictAction", "OfferAction")),
    _rule("CSG-T02", "STOPPING", evidence=("attempt_disposition",), paths=("action_receipt", "assessment"), effects=("RequireActionInput",)),
    _rule("CSG-T03", "STOPPING", evidence=("dependency_digest",), paths=("revision", "assessment"), effects=("RecordIncident",)),
    _rule("CSG-T04", "STOPPING", evidence=("host_impossibility",), paths=("assessment", "capability"), effects=("RequestHostProcedure",), unknown="RELEASE"),
    _rule("CSG-P01", "PROPORTION", evidence=("done_attempt", "open_clause"), paths=("completion", "assessment"), effects=("RestrictAction", "OfferAction")),
    _rule("CSG-P02", "PROPORTION", evidence=("unmet_scope",), paths=("diff", "assessment"), effects=("OfferAction",)),
    _rule("CSG-P03", "PROPORTION", evidence=("available_action",), paths=("menu", "capability"), effects=("RouteExecution",)),
    _rule("CSG-A01", "AUTHORITY", evidence=("completion_owner",), paths=("completion", "menu"), effects=("RouteExecution",)),
    _rule("CSG-A02", "AUTHORITY", evidence=("completion_disposition",), paths=("completion",), effects=("RecordIncident",)),
    _rule("CSG-C01", "CONTROL", evidence=("typed_action",), paths=("menu", "capability"), effects=("RequireActionInput",)),
    _rule("CSG-C02", "CONTROL", evidence=("delivery_key",), paths=("revision", "assessment"), effects=("RecordIncident",)),
    _rule("CSG-D01", "DEPENDENCY", evidence=("current_binding",), paths=("binding", "diff"), effects=("UpdateApplicability",)),
    _rule("CSG-H01", "HANDOFF", evidence=("host_capability",), paths=("capability", "assessment"), effects=("RequestHostProcedure",), unknown="HOST_BLOCK"),
    _rule("CSG-H02", "HANDOFF", evidence=("attempt_history",), paths=("action_receipt", "assessment"), effects=("RequestHostProcedure",), unknown="HOST_BLOCK"),
)


@dataclass(frozen=True, slots=True)
class ControlAction:
    tool: str
    reason: str
    clears: str
    arguments: Mapping[str, object] = ()
    rule_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "reason": self.reason,
            "clears": self.clears,
            "arguments": dict(self.arguments) if isinstance(self.arguments, Mapping) else {},
            "rule_ids": list(self.rule_ids),
        }


@dataclass(frozen=True, slots=True)
class GovernorDecision:
    dependency_key: str
    rule_ids: tuple[str, ...]
    clauses: tuple[Mapping[str, object], ...]
    unmet_scope: tuple[str, ...]
    actions: tuple[ControlAction, ...]
    control_line: str
    release_uncertified: bool = False
    host_blocked: bool = False
    advisory: bool = False
    frame_kind: str = ""

    @property
    def fired(self) -> bool:
        return bool(self.rule_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "dependency_key": self.dependency_key,
            "rule_ids": list(self.rule_ids),
            "clauses": [dict(clause) for clause in self.clauses],
            "unmet_scope": list(self.unmet_scope),
            "actions": [action.to_dict() for action in self.actions],
            "control_line": self.control_line,
            "release_uncertified": self.release_uncertified,
            "host_blocked": self.host_blocked,
            "advisory": self.advisory,
            "frame_kind": self.frame_kind,
        }


@dataclass(frozen=True, slots=True)
class _Facts:
    revision: object
    public_green: bool
    clauses: tuple[Mapping[str, object], ...]
    changed_paths: tuple[str, ...]
    available_paths: tuple[str, ...]
    done_attempts: int
    attempts: Mapping[str, int]
    impossibilities: frozenset[str]
    terminal_requested: bool
    inspected_paths: tuple[str, ...]
    no_op_paths: tuple[str, ...]
    scope_blockers: tuple[str, ...]

    @property
    def unmet_scope(self) -> tuple[str, ...]:
        missing: list[str] = []
        available = self.available_paths
        changed = set(self.changed_paths)
        for clause in self.clauses:
            named = _named_paths(str(clause.get("clause", "")), available)
            for path in named:
                if path not in changed and path not in missing:
                    missing.append(path)
        return tuple(missing[:MAX_SCOPE_ITEMS])

    def dependency_payload(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "public_green": self.public_green,
            "clauses": [
                {
                    "id": clause.get("id"),
                    "clause": clause.get("clause"),
                    "verdict": clause.get("verdict"),
                    "reason": clause.get("reason"),
                    "attempts": self.attempts.get(str(clause.get("id", "")), 0),
                    "impossible": str(clause.get("id", "")) in self.impossibilities,
                }
                for clause in self.clauses
            ],
            "changed_paths": self.changed_paths,
            "available_paths": self.available_paths,
            "terminal_requested": self.terminal_requested,
        }


def _records(runtime: object) -> tuple[Mapping[str, object], ...]:
    result: list[Mapping[str, object]] = []
    journal = getattr(runtime, "journal", None)
    source = getattr(journal, "records", None)
    try:
        source = source() if callable(source) else source
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        source = ()
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        result.extend(value for value in source if isinstance(value, Mapping))
    owner = getattr(runtime, "owner", None)
    for source in (
        getattr(owner, "_receipts", ()),
        getattr(runtime, "receipts", ()),
        (getattr(runtime, "last_completed_receipt", None),),
    ):
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            result.extend(value for value in source if isinstance(value, Mapping))
    return tuple(result)


def _binding(runtime: object) -> Mapping[str, object]:
    provider = getattr(runtime, "binding", None)
    try:
        value = provider() if callable(provider) else provider
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        value = None
    return value if isinstance(value, Mapping) else {}


def _paths(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Mapping):
        values = tuple(value)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(value)
    else:
        return ()
    result: list[str] = []
    for item in values:
        path = str(item).strip().replace("\\", "/")
        if path and path not in result and not path.startswith("<"):
            result.append(path.split(":", 1)[0])
    return tuple(sorted(result))


def _record_view(record: Mapping[str, object]) -> Mapping[str, object]:
    """Unwrap live action receipts without treating their event wrapper as facts."""
    value = record.get("record", record.get("payload", record))
    if isinstance(value, Mapping):
        nested = value.get("receipt")
        return nested if isinstance(nested, Mapping) else value
    return record


def _record_action_kind(record: Mapping[str, object]) -> str:
    view = _record_view(record)
    for key in ("action_kind", "tool", "kind"):
        value = view.get(key)
        if value:
            return str(value).strip().lower()
    return ""


def _record_paths(record: Mapping[str, object]) -> tuple[str, ...]:
    view = _record_view(record)
    values: list[str] = []
    for key in ("paths", "scope", "changed_paths", "changed_files", "path", "target",
                "target_path", "file", "file_path"):
        for path in _paths(view.get(key, ())):
            if path not in values:
                values.append(path)
    parameters = view.get("action_parameters")
    if isinstance(parameters, Mapping):
        for key in ("path", "target", "file", "file_path"):
            for path in _paths(parameters.get(key, ())):
                if path not in values:
                    values.append(path)
    return tuple(sorted(values))


def _record_result(record: Mapping[str, object]) -> str:
    view = _record_view(record)
    return str(view.get("result_class", view.get("result", view.get("status", "")))).upper()


def _record_observation(record: Mapping[str, object]) -> str:
    view = _record_view(record)
    return _text(view.get("observation", view.get("content", "")), 1200)


def _action_signatures(runtime: object) -> tuple[tuple[str, ...], ...]:
    signatures: list[tuple[str, ...]] = []
    for record in _records(runtime):
        kind = _record_action_kind(record)
        if kind not in _OBSERVATION_KINDS:
            continue
        view = _record_view(record)
        sequence = str(record.get("sequence", record.get("event_id", "")))
        signatures.append((sequence, kind, _record_result(record), ",".join(_record_paths(record)),
                           _text(view.get("observation", ""), 160)))
    return tuple(signatures)


def _inspected_paths(runtime: object) -> tuple[str, ...]:
    values: list[str] = []
    for name in ("inspected_paths", "read_paths", "_read_paths"):
        for path in _paths(getattr(runtime, name, ())):
            if path not in values:
                values.append(path)
    snapshots = getattr(runtime, "source_snapshots", ())
    if isinstance(snapshots, Sequence) and not isinstance(snapshots, (str, bytes, bytearray)):
        for snapshot in snapshots:
            if isinstance(snapshot, Mapping):
                path = str(snapshot.get("target", snapshot.get("path", ""))).strip()
                if path and path not in values:
                    values.append(path)
    for record in _records(runtime):
        if _record_action_kind(record) != "read":
            continue
        for path in _record_paths(record):
            if path not in values:
                values.append(path)
    return tuple(sorted(values))


def _no_op_paths(runtime: object) -> tuple[str, ...]:
    values: list[str] = []
    for name in ("no_op_paths", "edit_noop_paths", "_edit_noop_paths"):
        for path in _paths(getattr(runtime, name, ())):
            if path not in values:
                values.append(path)
    for record in _records(runtime):
        if _record_action_kind(record) not in _EDIT_KINDS:
            continue
        if _record_result(record) not in _NO_OP_RESULTS and not _NO_OP_TEXT.search(_record_observation(record)):
            continue
        for path in _record_paths(record):
            if path not in values:
                values.append(path)
    return tuple(sorted(values))


def _snapshot_text(runtime: object, target: str) -> str:
    snapshots = getattr(runtime, "source_snapshots", ())
    if not isinstance(snapshots, Sequence) or isinstance(snapshots, (str, bytes, bytearray)):
        return ""
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        path = str(snapshot.get("target", snapshot.get("path", ""))).strip()
        if path != target:
            continue
        content = snapshot.get("content", snapshot.get("value", ""))
        if isinstance(content, Mapping):
            content = content.get("value", "")
        return str(content or "")
    return ""


def _plausibly_changed(runtime: object, target: str, changed_paths: Sequence[str]) -> bool:
    """Accept an adjacent implementation edit only when host evidence links it."""
    if target in changed_paths:
        return True
    source = _snapshot_text(runtime, target).casefold()
    if not source:
        return False
    for path in changed_paths:
        normalized = str(path).replace("\\", "/").casefold()
        if normalized in source or Path(normalized).stem in source:
            return True
    return False


def _scope_blockers(
        runtime: object,
        clauses: Sequence[Mapping[str, object]],
        available: Sequence[str],
        changed: Sequence[str],
        inspected: Sequence[str],
        no_ops: Sequence[str],
) -> tuple[str, ...]:
    blockers: list[str] = []
    for clause in clauses:
        text = str(clause.get("clause", ""))
        if not _CONCRETE_BEHAVIOR.search(text):
            continue
        for path in _named_paths(text, available):
            if (path in changed or path in inspected
                    or _plausibly_changed(runtime, path, changed)
                    or _plausibly_changed(runtime, path, no_ops)):
                continue
            if path not in blockers:
                blockers.append(path)
    return tuple(blockers[:MAX_SCOPE_ITEMS])


def _binding_paths(binding: Mapping[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("product_files", "files", "source_files", "harness_files"):
        candidate = binding.get(key)
        for path in _paths(candidate):
            if path not in values:
                values.append(path)
    return tuple(sorted(values))


def _changed_from_bindings(record: Mapping[str, object]) -> tuple[str, ...]:
    before = record.get("before_binding")
    after = record.get("after_binding")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return ()
    before_files = before.get("product_files")
    after_files = after.get("product_files")
    if not isinstance(before_files, Mapping) or not isinstance(after_files, Mapping):
        return ()
    return tuple(sorted(
        str(path) for path in set(before_files) | set(after_files)
        if before_files.get(path) != after_files.get(path)
    ))


def _changed_paths(runtime: object) -> tuple[str, ...]:
    values: list[str] = []
    for name in ("changed_paths", "changed_files", "edited_paths", "_changed_paths"):
        for path in _paths(getattr(runtime, name, ())):
            if path not in values:
                values.append(path)
    for record in _records(runtime):
        payload = record.get("record")
        base = payload if isinstance(payload, Mapping) else record
        nested = base.get("receipt")
        view = nested if isinstance(nested, Mapping) else base
        result = str(view.get("result_class", view.get("result", ""))).upper()
        kind = str(view.get(
            "kind", view.get("action_kind", view.get("tool", "")))).lower()
        if kind not in _EDIT_KINDS and not bool(view.get("artifact_change")):
            continue
        if result not in {"APPLIED", "SUCCESS", "OK", "GREEN", "PASS", "PASSED", ""}:
            continue
        for key in (
            "paths", "scope", "changed_paths", "changed_files", "path", "target",
            "target_path", "file", "file_path",
        ):
            for path in _paths(view.get(key, ())):
                if path not in values:
                    values.append(path)
        for path in _changed_from_bindings(base):
            if path not in values:
                values.append(path)
    return tuple(sorted(values))


def _available_paths(runtime: object, binding: Mapping[str, object]) -> tuple[str, ...]:
    values = list(_binding_paths(binding))
    snapshots = getattr(runtime, "source_snapshots", ())
    if isinstance(snapshots, Sequence) and not isinstance(snapshots, (str, bytes, bytearray)):
        for snapshot in snapshots:
            if isinstance(snapshot, Mapping):
                path = str(snapshot.get("target", snapshot.get("path", ""))).strip()
                if path and path not in values:
                    values.append(path)
    owner = getattr(runtime, "owner", None)
    root_value = getattr(owner, "root", None)
    if not values and root_value is not None:
        try:
            root = Path(root_value).resolve()
            for path in root.rglob("*"):
                if len(values) >= 5000:
                    break
                if not path.is_file() or any(part.startswith(".") for part in path.relative_to(root).parts):
                    continue
                relative = path.relative_to(root).as_posix()
                if relative not in values and not relative.startswith("backups/"):
                    values.append(relative)
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
    return tuple(sorted(values))


def _named_paths(clause: str, available: Sequence[str]) -> tuple[str, ...]:
    text = clause.casefold()
    matches: list[str] = []
    for match in _PATH_RE.findall(clause):
        path = match.split(":", 1)[0]
        if path in available and path not in matches:
            matches.append(path)
    for path in available:
        name = Path(path).stem.casefold()
        if len(name) < 3:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", text) and path not in matches:
            matches.append(path)
    return tuple(sorted(matches))


def _assessment_rows(runtime: object) -> tuple[Mapping[str, object], ...]:
    source = getattr(runtime, "host_closure_assessments", ())
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes, bytearray)):
        return ()
    rows: list[Mapping[str, object]] = []
    for item in source:
        value = _as_mapping(item)
        if value is None:
            continue
        ident = str(value.get("obligation_id", value.get("id", ""))).strip()
        try:
            from kernel_premature_rule import enabled as premature_enabled
            from kernel_premature_rule import has_clause_closing_receipt
            if premature_enabled(runtime) and has_clause_closing_receipt(value, ident):
                continue
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
        verdict = str(value.get("verdict", value.get("status", ""))).upper()
        reason = str(value.get("reason", value.get("reason_code", ""))).upper()
        if not ident or verdict in _SATISFIED:
            continue
        if verdict not in {"UNKNOWN", "UNSATISFIED", "OPEN", ""} and reason not in _UNKNOWN:
            continue
        clause = _text(value.get("clause", value.get("clause_text", "")), MAX_CLAUSE_CHARS)
        rows.append({
            "id": ident,
            "clause": clause or ident,
            "verdict": verdict or "UNKNOWN",
            "reason": reason or "UNKNOWN",
            "check_ids": tuple(str(item) for item in value.get("check_ids", ()) or ()),
            "receipts": tuple(value.get("receipts", ()) or ()),
            "location_hint": str(value.get("location_hint", "") or ""),
            "targets": tuple(str(item) for item in value.get("targets", ()) or ()),
        })
    return tuple(rows)


def _attempts(runtime: object, rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    result = {str(row.get("id", "")): 0 for row in rows}
    untagged_edits = 0
    for name in ("governor_attempts_by_obligation", "attempts_by_obligation"):
        source = getattr(runtime, name, None)
        if isinstance(source, Mapping):
            for ident in result:
                value = source.get(ident, 0)
                if type(value) is int and value > 0:
                    result[ident] = value
    for record in _records(runtime):
        payload = record.get("record")
        base = payload if isinstance(payload, Mapping) else record
        nested = base.get("receipt")
        view = nested if isinstance(nested, Mapping) else base
        kind = str(base.get("kind", record.get("kind", ""))).lower()
        action_kind = str(view.get(
            "action_kind", view.get("tool", base.get("action_kind", base.get("tool", kind))))).lower()
        if kind in {"done", "test", "test_run", "observation"} or action_kind not in _EDIT_KINDS:
            continue
        result_class = view.get(
            "result_class", view.get("result", base.get("result_class", base.get("result", ""))))
        if str(result_class).upper() not in {"APPLIED", "SUCCESS", "OK", "PASS", "PASSED"}:
            continue
        ident = str(view.get(
            "obligation_id", view.get(
                "clause_id", base.get("obligation_id", base.get("clause_id", ""))))).strip()
        if ident in result:
            result[ident] += 1
        elif not ident:
            untagged_edits += 1
    # Some live owner receipts do not carry an obligation ID.  A single open
    # clause still gets credit for one admitted product edit, while multi-clause
    # runs remain conservative until each clause is explicitly attributed.
    if len(result) == 1 and untagged_edits:
        ident = next(iter(result))
        result[ident] = max(result[ident], untagged_edits)
    return result


def _impossibilities(runtime: object) -> frozenset[str]:
    values: set[str] = set()
    for name in ("host_classified_impossibilities", "governor_impossibilities", "host_impossibilities"):
        source = getattr(runtime, name, ())
        if isinstance(source, Mapping):
            values.update(str(key) for key, value in source.items() if value)
        elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            values.update(str(item) for item in source)
    return frozenset(value for value in values if value)


def _done_attempts(runtime: object) -> int:
    for name in ("done_attempts", "done_count", "_done_challenge_count"):
        value = getattr(runtime, name, None)
        if type(value) is int and value > 0:
            return value
    return sum(
        1 for record in _records(runtime)
        if str(record.get("kind", record.get("action_kind", ""))).lower() == "done"
    )


def _public_green(runtime: object, binding: Mapping[str, object]) -> bool:
    explicit = getattr(runtime, "public_green", None)
    if type(explicit) is bool:
        return explicit
    verdict_fn = getattr(runtime, "public_test_verdict", None)
    if callable(verdict_fn):
        try:
            verdict = verdict_fn()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False
        state = str(getattr(verdict, "state", _field(verdict, "state", ""))).upper()
        return state in _SATISFIED
    return bool(getattr(runtime, "green", False))


def _terminal_requested(runtime: object) -> bool:
    if bool(getattr(runtime, "done_requested", False)):
        return True
    state = getattr(runtime, "_completion_owner_state", None)
    if isinstance(state, Mapping) and bool(state.get("terminal_choice_offered")):
        return True
    if bool(done_accepted(runtime)):
        return True
    decision = getattr(runtime, "_last_completion_decision", None)
    return decision is not None and _field(decision, "accepted", None) is True


def _facts(runtime: object) -> _Facts:
    binding = _binding(runtime)
    rows = _assessment_rows(runtime)
    attempts = _attempts(runtime, rows)
    changed_paths = _changed_paths(runtime)
    available_paths = _available_paths(runtime, binding)
    inspected_paths = _inspected_paths(runtime)
    no_op_paths = _no_op_paths(runtime)
    return _Facts(
        revision=binding.get("product_revision", binding.get("revision", getattr(runtime, "workspace_revision", "unknown"))),
        public_green=_public_green(runtime, binding),
        clauses=rows,
        changed_paths=changed_paths,
        available_paths=available_paths,
        done_attempts=_done_attempts(runtime),
        attempts=attempts,
        impossibilities=_impossibilities(runtime),
        terminal_requested=_terminal_requested(runtime),
        inspected_paths=inspected_paths,
        no_op_paths=no_op_paths,
        scope_blockers=_scope_blockers(
            runtime, rows, available_paths, changed_paths, inspected_paths, no_op_paths),
    )


def _render_clause(clause: Mapping[str, object]) -> str:
    return f"{clause.get('id')}: {json.dumps(str(clause.get('clause', '')), ensure_ascii=False)}"


def _action_templates(rule_ids: tuple[str, ...], clauses: Sequence[Mapping[str, object]],
                      unmet_scope: Sequence[str]) -> tuple[ControlAction, ...]:
    clause_ids = tuple(str(clause.get("id", "")) for clause in clauses)
    label = ", ".join(clause_ids) or "open requirements"
    scope = ", ".join(unmet_scope) or "the named requirement surface"
    return (
        ControlAction("edit", f"repair the open requirement surface ({scope})", f"host evidence for {label}", rule_ids=rule_ids),
        ControlAction("str_replace", f"repair the open requirement surface ({scope})", f"host evidence for {label}", rule_ids=rule_ids),
        ControlAction("read", f"inspect the named clause and unchanged surface ({scope})", f"scope evidence for {label}", rule_ids=rule_ids),
    )


def _filter_actions(actions: Sequence[ControlAction], tools: object) -> tuple[ControlAction, ...]:
    available: set[str] = set()
    if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray)):
        for tool in tools:
            if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping):
                name = str(tool["function"].get("name", ""))
                if name:
                    available.add(name)
    return tuple(action for action in actions if action.tool in available)


def _line(clauses: Sequence[Mapping[str, object]], unmet_scope: Sequence[str],
          actions: Sequence[ControlAction], *, host_blocked: bool = False,
          advisory: bool = False) -> str:
    clause_text = "; ".join(_render_clause(clause) for clause in clauses)
    scope_text = ", ".join(unmet_scope) if unmet_scope else "none"
    action_text = ", ".join(f"{action.tool}()" for action in actions) or "host repair"
    if advisory:
        return (
            "[COMMON-SENSE GOVERNOR] scope advisory: named surfaces are unchanged, "
            f"but this is a candidate gap rather than proof. Clauses: {clause_text}. "
            f"Unmet scope: {scope_text} unchanged in the current revision. Done remains allowed."
        )[:2400]
    return (
        "[COMMON-SENSE GOVERNOR] done refused: open evidence remains. "
        f"Clauses: {clause_text}. Unmet scope: {scope_text} unchanged in the current revision. "
        f"Next legal actions: {action_text}. "
        + ("Host must clear the unavailable route; no model action is admitted." if host_blocked
           else "These actions do not certify or waive any requirement.")
    )[:2400]


def _release_line(decision: GovernorDecision) -> str:
    scope = ", ".join(decision.unmet_scope) or "the named requirement surface"
    return (
        "[COMMON-SENSE GOVERNOR] scope advisory: post-refusal read/no-op evidence "
        f"makes unchanged scope ({scope}) advisory; done remains allowed."
    )


def _premature_frame(runtime: object, tools: object = ()) -> Mapping[str, object] | None:
    """Read the shared FLP1-B frame without rendering a second notice."""
    try:
        from kernel_premature_rule import build_frame
        value = build_frame(runtime, tools)
        return value if isinstance(value, Mapping) else None
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


def decide(runtime: object, tools: object = ()) -> GovernorDecision | None:
    """Return the action decision for the current dependency episode, if any."""
    if not enabled(runtime):
        return None
    facts = _facts(runtime)
    premature_frame = _premature_frame(runtime, tools)
    if not facts.terminal_requested or (not facts.clauses and premature_frame is None):
        return None
    dependency_payload = facts.dependency_payload()
    if premature_frame is not None:
        dependency_payload["flp1_frame"] = {
            "clause_ids": list(premature_frame.get("clause_ids", ())),
            "clause_texts": list(premature_frame.get("clause_texts", ())),
            "surface_roles": list(premature_frame.get("surface_roles", ())),
        }
    dependency_key = _digest(dependency_payload)
    cache = getattr(runtime, "_common_sense_governor_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(runtime, "_common_sense_governor_cache", cache)
    episode = getattr(runtime, "_common_sense_governor_episode", None)
    if not isinstance(episode, dict) or episode.get("dependency_key") != dependency_key:
        episode = {
            "dependency_key": dependency_key,
            "action_signatures": _action_signatures(runtime),
            "released": False,
        }
        setattr(runtime, "_common_sense_governor_episode", episode)
        setattr(runtime, "_common_sense_governor_release", False)
    cached = cache.get(dependency_key)
    if isinstance(cached, GovernorDecision):
        current_signatures = set(_action_signatures(runtime))
        baseline_signatures = set(episode.get("action_signatures", ()))
        if not episode.get("released") and current_signatures - baseline_signatures:
            episode["released"] = True
            setattr(runtime, "_common_sense_governor_release", True)
            setattr(runtime, "_common_sense_governor_advisory", _release_line(cached))
            emit = getattr(getattr(runtime, "journal", None), "emit", None)
            if callable(emit):
                emit(
                    "common_sense_governor_released",
                    producer="kernel_common_sense_governor",
                    event_schema=SCHEMA,
                    dependency_key=dependency_key,
                    reason="post-refusal model action is evidence that scope fact is advisory",
                    dispatch=False,
                )
            return None
        if episode.get("released"):
            return None
        return GovernorDecision(
            dependency_key=cached.dependency_key,
            rule_ids=cached.rule_ids,
            clauses=cached.clauses,
            unmet_scope=cached.unmet_scope,
            actions=_filter_actions(cached.actions, tools) if tools else cached.actions,
            control_line=cached.control_line,
            release_uncertified=cached.release_uncertified,
            host_blocked=cached.host_blocked,
            advisory=cached.advisory,
            frame_kind=cached.frame_kind,
        )

    active_rules: list[str] = []
    clauses = facts.clauses
    unmet_scope = facts.unmet_scope
    blocking_scope = facts.scope_blockers
    advisory = bool(unmet_scope) and not bool(blocking_scope)
    frame_kind = ""
    if premature_frame is not None:
        frame_kind = "FLP1-B"
        clauses = tuple({
            "id": str(identifier),
            "clause": str(clause),
            "verdict": "UNKNOWN",
            "reason": "CLAUSE_CLOSURE_NOT_RECEIPTED",
        } for identifier, clause in zip(
            premature_frame.get("clause_ids", ()), premature_frame.get("clause_texts", ())))
        unmet_scope = tuple(str(item) for item in premature_frame.get("surface_roles", ()) if str(item))
        blocking_scope = ()
        active_rules.extend(("CSG-T01", "CSG-P01"))
        attempts_complete = all(
            facts.attempts.get(str(clause.get("id", "")), 0) > 0
            or str(clause.get("id", "")) in facts.impossibilities
            for clause in clauses
        )
    else:
        missing_evidence = any(str(clause.get("reason", "")).upper() in _UNKNOWN for clause in facts.clauses)
        if facts.public_green and missing_evidence:
            active_rules.extend(("CSG-E01", "CSG-E02", "CSG-E03", "CSG-E04"))
        if blocking_scope:
            active_rules.extend(("CSG-S01", "CSG-S02", "CSG-S03", "CSG-S04"))
        attempts_complete = all(
            facts.attempts.get(str(clause.get("id", "")), 0) > 0
            or str(clause.get("id", "")) in facts.impossibilities
            for clause in facts.clauses
        )
        if not attempts_complete and not advisory:
            active_rules.extend(("CSG-T01", "CSG-T02", "CSG-P01"))
        if blocking_scope:
            active_rules.extend(("CSG-T03", "CSG-P02", "CSG-P03"))
        if facts.impossibilities and attempts_complete:
            active_rules.extend(("CSG-T04", "CSG-H01"))
        if facts.clauses and not advisory:
            active_rules.extend(("CSG-A01", "CSG-A02", "CSG-C01", "CSG-C02", "CSG-D01"))
    if not active_rules and not advisory:
        return None
    rule_ids = tuple(dict.fromkeys(active_rules))
    release_uncertified = (
        attempts_complete and (not blocking_scope or set(
            str(clause.get("id", "")) for clause in clauses
        ).issubset(set(facts.impossibilities)))
    ) if premature_frame is not None else advisory or (
        attempts_complete and (not blocking_scope or set(
            str(clause.get("id", "")) for clause in facts.clauses
        ).issubset(set(facts.impossibilities)))
    )
    host_blocked = not release_uncertified and not blocking_scope and not attempts_complete
    if premature_frame is not None:
        templates = tuple(
            ControlAction(
                str(tool),
                "expose the unclosed public requirement without treating green as closure",
                "a clause-level closing receipt or the bounded terminal submission",
                rule_ids=rule_ids,
            )
            for tool in premature_frame.get("menu", ()) if str(tool)
        )
    else:
        templates = _action_templates(rule_ids, facts.clauses, unmet_scope)
    actions = _filter_actions(templates, tools) if tools else templates
    line = str(premature_frame.get("notice", "")) if premature_frame is not None else _line(
        facts.clauses, unmet_scope, actions,
        host_blocked=host_blocked, advisory=advisory and not active_rules)
    decision = GovernorDecision(
        dependency_key=dependency_key,
        rule_ids=rule_ids,
        clauses=clauses,
        unmet_scope=unmet_scope,
        actions=actions,
        control_line=line,
        release_uncertified=release_uncertified,
        host_blocked=host_blocked,
        advisory=advisory and not active_rules,
        frame_kind=frame_kind,
    )
    cache[dependency_key] = decision
    fired = getattr(runtime, "_common_sense_governor_fired", None)
    if not isinstance(fired, set):
        fired = set()
        setattr(runtime, "_common_sense_governor_fired", fired)
    if dependency_key not in fired:
        fired.add(dependency_key)
        emit = getattr(getattr(runtime, "journal", None), "emit", None)
        if callable(emit):
            emit(
                "common_sense_governor_fired",
                producer="kernel_common_sense_governor",
                event_schema=SCHEMA,
                rule_ids=list(rule_ids),
                dependency_key=dependency_key,
                unmet_scope=list(unmet_scope),
                open_ids=[str(clause.get("id", "")) for clause in clauses],
                changed_paths=list(facts.changed_paths),
                attempts=dict(facts.attempts),
                release_uncertified=release_uncertified,
                dispatch=False,
            )
    setattr(runtime, "_common_sense_governor_episode", {
        "dependency_key": dependency_key,
        "action_signatures": _action_signatures(runtime),
        "released": False,
    })
    return decision


def rule_specs() -> tuple[RuleSpec, ...]:
    return RULE_SPECS


__all__ = [
    "ControlAction", "GovernorDecision", "MODE_ENV", "RULE_SPECS", "RuleSpec", "SCHEMA",
    "decide", "enabled", "mode", "rule_specs",
]

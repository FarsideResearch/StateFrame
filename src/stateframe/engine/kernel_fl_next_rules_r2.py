"""Model-free qualification contracts for FL-NEXT-RULES-R2.

This module is deliberately an evidence boundary, not a second request runner.
It does not call a provider, dispatch a tool, mutate a workspace, or infer a
question from prompt text.  The records returned here are suitable for focused
host tests and for deciding whether a later live screen is admissible.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


SCHEMA = "fl-next-rules-r2/1"
MAX_SCREEN_OUTPUT_CHARS = 600
Q2_SEEDS = (113, 227, 349)
Q2_MAX_CALLS = 6
PREDICATES = ("import", "coordination", "behavioral", "completion")
_LEAK_WORDS = frozenset({
    "[control]", "omitted", "withheld", "evaluator", "reference fix",
    "gold fix", "treatment-only", "correct site",
})


class QualificationError(ValueError):
    """Raised when a proposed qualification artifact is not admissible."""


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    return value


def _digest(value: object) -> str:
    encoded = json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


# ---------------------------------------------------------------------------
# QWIRE: explicit question construction and exact-once wire validation


@dataclass(frozen=True)
class QuestionWireReceipt:
    policy: str
    append_question: bool
    intended_question: str
    serialized_messages: str
    question_occurrences: int
    semantic_valid: bool
    diagnostic: Mapping[str, object] = field(default_factory=dict)


_QUESTION_POLICIES = frozenset({
    "registered", "paraphrase", "lowercase", "messages_file", "embedded_question", "splice",
})
_PRESERVE_QUESTION_POLICIES = frozenset({"messages_file", "embedded_question", "splice"})


def build_question_wire(
        messages: Sequence[Mapping[str, object]], *, intended_question: str,
        policy: str, append_question: bool,
) -> QuestionWireReceipt:
    """Build/validate final messages with an explicit question policy.

    ``splice`` is the deliberate ``append_question=false`` probe.  It is valid
    only when the caller already supplied the question exactly once.
    """
    if policy not in _QUESTION_POLICIES:
        raise QualificationError(f"unknown question policy: {policy}")
    question = intended_question.strip()
    if not question:
        raise QualificationError("intended question must be non-empty")
    if not isinstance(append_question, bool):
        raise QualificationError("append_question must be explicit")
    if policy in _PRESERVE_QUESTION_POLICIES and append_question:
        raise QualificationError(f"{policy} requires append_question=false")
    if policy not in _PRESERVE_QUESTION_POLICIES and not append_question:
        raise QualificationError(f"{policy} requires append_question=true")

    result = [dict(message) for message in messages]
    contents = [str(message.get("content", "")) for message in result]
    before = sum(content.count(question) for content in contents)
    if append_question:
        user_indexes = [index for index, message in enumerate(result)
                        if str(message.get("role", "")).lower() == "user"]
        if not user_indexes:
            raise QualificationError("question append requires a user message")
        if before:
            raise QualificationError("question is already present; refusing duplicate append")
        index = user_indexes[-1]
        existing = str(result[index].get("content", ""))
        result[index]["content"] = f"{existing.rstrip()}\n\n{question}" if existing.strip() else question

    final_contents = [str(message.get("content", "")) for message in result]
    occurrences = sum(content.count(question) for content in final_contents)
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if occurrences != 1:
        raise QualificationError(f"question must occur exactly once, found {occurrences}")
    return QuestionWireReceipt(
        policy=policy,
        append_question=append_question,
        intended_question=question,
        serialized_messages=serialized,
        question_occurrences=occurrences,
        semantic_valid=True,
    )


def diagnostic_wire_checks(
        serialized_messages: str, *, regex: str | None = None,
        marker: str | None = None, json_key: str | None = None,
) -> Mapping[str, object]:
    """Return diagnostics only; none of these fields establish semantic success."""
    regex_match = bool(re.search(regex, serialized_messages)) if regex else None
    marker_match = marker in serialized_messages if marker else None
    key_present = None
    if json_key:
        try:
            value = json.loads(serialized_messages)
        except (TypeError, ValueError, json.JSONDecodeError):
            value = None
        key_present = isinstance(value, list) and any(
            isinstance(item, Mapping) and json_key in item for item in value
        )
    return {
        "regex_match": regex_match,
        "marker_match": marker_match,
        "json_key_present": key_present,
        "semantic_authority": "question_wire_receipt",
    }


# ---------------------------------------------------------------------------
# SOURCE: one protected semantic unit and deterministic final allocation


@dataclass(frozen=True)
class ToolSchema:
    name: str
    revision: str
    schema: Mapping[str, object]
    required: bool = False

    @property
    def identity(self) -> str:
        return _digest({"name": self.name, "revision": self.revision, "schema": self.schema})

    @property
    def wire_bytes(self) -> int:
        return _json_bytes(self.schema)


@dataclass(frozen=True)
class SourceUnit:
    unit_id: str
    path: str
    language: str
    body: str
    type_facts: str
    binding: str
    revision: str
    required: bool = True
    sidecar: bool = False

    @property
    def semantic_text(self) -> str:
        parts = (self.body.strip(), self.type_facts.strip(), self.binding.strip())
        return "\n".join(part for part in parts if part)

    @property
    def semantic_bytes(self) -> int:
        return len(self.semantic_text.encode("utf-8"))

    @property
    def usable(self) -> bool:
        return bool(self.body.strip() and self.type_facts.strip() and self.binding.strip())


@dataclass(frozen=True)
class SourceAllocation:
    status: str
    retained_units: tuple[SourceUnit, ...]
    omitted_sidecars: tuple[SourceUnit, ...]
    acquisition_handle: str | None
    available_actions: tuple[str, ...]
    base_bytes: int
    schema_bytes: int
    metadata_bytes: int
    final_bytes: int
    required_bytes: int


def allocate_source_units(
        units: Sequence[SourceUnit], *, base_bytes: int,
        schemas: Sequence[ToolSchema], metadata_bytes: int,
        max_request_bytes: int,
) -> SourceAllocation:
    """Allocate once, protecting whole semantic units before optional sidecars."""
    if max_request_bytes < 1 or base_bytes < 0 or metadata_bytes < 0:
        raise QualificationError("request budget values must be non-negative")
    ordered = tuple(sorted(units, key=lambda unit: (unit.path, unit.unit_id, unit.revision)))
    if len({unit.unit_id for unit in ordered}) != len(ordered):
        raise QualificationError("source unit IDs must be unique")
    if any(not unit.semantic_text or not unit.usable for unit in ordered if unit.required):
        raise QualificationError("blank or incomplete required semantic unit is not edit-ready")
    unique_schemas = {schema.identity: schema for schema in schemas}
    schema_bytes = sum(schema.wire_bytes for schema in unique_schemas.values())
    available = max_request_bytes - base_bytes - schema_bytes - metadata_bytes
    required = tuple(unit for unit in ordered if unit.required and not unit.sidecar)
    optional = tuple(unit for unit in ordered if not unit.required or unit.sidecar)
    required_bytes = sum(unit.semantic_bytes for unit in required)
    actions = ("run_tests", "recover_source", "read_source")
    if available < required_bytes or required_bytes < 1:
        handle = "acquire:" + _digest({
            "units": [(unit.unit_id, unit.path, unit.revision) for unit in required],
            "budget": max_request_bytes,
        })
        return SourceAllocation(
            status="ACQUIRE_REQUIRED_UNIT",
            retained_units=(),
            omitted_sidecars=ordered,
            acquisition_handle=handle,
            available_actions=actions,
            base_bytes=base_bytes,
            schema_bytes=schema_bytes,
            metadata_bytes=metadata_bytes,
            final_bytes=base_bytes + schema_bytes + metadata_bytes,
            required_bytes=required_bytes,
        )

    retained: list[SourceUnit] = list(required)
    remaining = available - required_bytes
    omitted: list[SourceUnit] = []
    for unit in optional:
        if unit in retained:
            continue
        if unit.semantic_bytes <= remaining and unit.semantic_text:
            retained.append(unit)
            remaining -= unit.semantic_bytes
        else:
            omitted.append(unit)
    final_bytes = base_bytes + schema_bytes + metadata_bytes + sum(unit.semantic_bytes for unit in retained)
    if final_bytes > max_request_bytes:
        raise QualificationError("allocator produced an over-budget final request")
    return SourceAllocation(
        status="READY",
        retained_units=tuple(retained),
        omitted_sidecars=tuple(omitted),
        acquisition_handle=None,
        available_actions=actions,
        base_bytes=base_bytes,
        schema_bytes=schema_bytes,
        metadata_bytes=metadata_bytes,
        final_bytes=final_bytes,
        required_bytes=required_bytes,
    )


def reallocate_after_growth(
        units: Sequence[SourceUnit], *, base_bytes: int,
        schemas: Sequence[ToolSchema], metadata_bytes: int,
        ceilings: Sequence[int] = (32 * 1024, 48 * 1024),
) -> tuple[SourceAllocation, ...]:
    """Recompute from all units after growth; never reuse a small prior window."""
    if not ceilings:
        raise QualificationError("at least one request ceiling is required")
    return tuple(allocate_source_units(
        units, base_bytes=base_bytes, schemas=schemas,
        metadata_bytes=metadata_bytes, max_request_bytes=ceiling,
    ) for ceiling in ceilings)


# ---------------------------------------------------------------------------
# CLOSURE and EXEC: dependency settlement and attempt-bound execution truth


@dataclass(frozen=True)
class SourceReadReceipt:
    receipt_id: str
    request_id: str
    path: str
    revision: int
    content_identity: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class EditReceipt:
    receipt_id: str
    request_id: str
    action_id: str
    attempt_id: str
    path: str
    previous_revision: int
    new_revision: int
    result: str


@dataclass(frozen=True)
class AttemptIdentity:
    request_id: str
    action_id: str
    dispatch_id: str
    attempt_id: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.request_id, self.action_id, self.dispatch_id, self.attempt_id)


@dataclass(frozen=True)
class RefusalReceipt:
    identity: AttemptIdentity
    reason: str
    receipt_id: str


@dataclass(frozen=True)
class TestExecutionReceipt:
    identity: AttemptIdentity
    path: str
    revision: int
    exit_code: int | None
    collected: int
    dots: int
    behavioral_pass: bool
    collection_only: bool = False
    validated_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionTruth:
    status: str
    passed: bool
    reason: str


def classify_execution(receipt: TestExecutionReceipt) -> ExecutionTruth:
    if receipt.exit_code == 137:
        return ExecutionTruth("KILLED", False, "exit 137 is not a successful execution")
    if receipt.collection_only:
        return ExecutionTruth("COLLECTION_ONLY", False, "collection without behavioral observation")
    if receipt.exit_code != 0:
        if receipt.collected and receipt.dots >= receipt.collected:
            return ExecutionTruth("ALL_DOTS_WITHOUT_EXIT_ZERO", False, "progress dots do not replace exit zero")
        return ExecutionTruth("EXECUTION_ERROR", False, "checker did not exit zero")
    if not receipt.behavioral_pass:
        return ExecutionTruth("BEHAVIORAL_FAIL", False, "exit zero lacks a passing behavioral observation")
    return ExecutionTruth("PASS", True, "exit zero with behavioral observation")


def refusal_for_attempt(refusal: RefusalReceipt, identity: AttemptIdentity) -> Mapping[str, object] | None:
    """Return a refusal card only for the exact request/action/dispatch/attempt."""
    if refusal.identity.as_tuple() != identity.as_tuple():
        return None
    return {
        "kind": "RUNTESTS_REFUSAL",
        "request_id": identity.request_id,
        "action_id": identity.action_id,
        "dispatch_id": identity.dispatch_id,
        "attempt_id": identity.attempt_id,
        "reason": refusal.reason,
        "receipt_id": refusal.receipt_id,
        "semantic_authority": "attempt_bound_refusal",
    }


class ClosureLedger:
    """Small source-dependency ledger used by the batch-840 replay contract."""

    def __init__(self, required_paths: Sequence[str]) -> None:
        normalized = tuple(sorted({str(path) for path in required_paths if str(path)}))
        if not normalized:
            raise QualificationError("closure requires at least one active product path")
        self.required_paths = normalized
        self.current_revisions: dict[str, int] = {}
        self.dirty_paths: set[str] = set()
        self.reads: list[SourceReadReceipt] = []
        self.checks: list[TestExecutionReceipt] = []
        self.action_accepted = False
        self.terminal_accepted = False

    def observe_read(self, receipt: SourceReadReceipt) -> None:
        self.reads.append(receipt)
        current = self.current_revisions.get(receipt.path)
        if current is not None and receipt.revision < current:
            return
        self.current_revisions[receipt.path] = receipt.revision
        if receipt.path in self.required_paths:
            self.dirty_paths.discard(receipt.path)

    def observe_edit(self, receipt: EditReceipt) -> None:
        if receipt.result.upper() != "APPLIED":
            return
        self.current_revisions[receipt.path] = receipt.new_revision
        if receipt.path in self.required_paths:
            self.dirty_paths.add(receipt.path)

    def observe_check(self, receipt: TestExecutionReceipt) -> ExecutionTruth:
        self.checks.append(receipt)
        truth = classify_execution(receipt)
        if truth.passed:
            for path in receipt.validated_paths:
                if path in self.required_paths and self.current_revisions.get(path) == receipt.revision:
                    self.dirty_paths.discard(path)
        return truth

    def accept_action(self) -> None:
        self.action_accepted = True

    def accept_terminal(self) -> None:
        if not self.action_accepted:
            raise QualificationError("terminal acceptance requires an accepted action")
        if self.dirty_paths:
            raise QualificationError("terminal acceptance requires settled source dependencies")
        self.terminal_accepted = True

    def completion(self) -> Mapping[str, object]:
        if self.dirty_paths:
            return {
                "source_settled": False,
                "terminal_accepted": False,
                "reason": "DEPENDENCY_VETO/STALE_EVIDENCE",
                "dirty_paths": tuple(sorted(self.dirty_paths)),
                "next_action": "read current source and rerun the affected check",
                "action_accepted": self.action_accepted,
            }
        return {
            "source_settled": True,
            "terminal_accepted": self.terminal_accepted,
            "reason": "CLEAN_EVIDENCE" if self.terminal_accepted else "ACTION_ACCEPTED_NOT_TERMINAL",
            "dirty_paths": (),
            "next_action": None if self.terminal_accepted else "perform terminal acceptance",
            "action_accepted": self.action_accepted,
        }


# ---------------------------------------------------------------------------
# PACKET: dependency-aware identity, including disk-backed non-Python files


@dataclass(frozen=True)
class DependencyIdentity:
    workspace_identity: str
    path: str
    revision: str
    content_identity: str
    parser_capability: str
    read_source: str


@dataclass(frozen=True)
class DependencyPacket:
    cache_key: str
    workspace_identity: str
    target_path: str
    dependencies: tuple[DependencyIdentity, ...]


def _relative_path(root: Path, raw: str) -> str:
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise QualificationError(f"unsafe dependency path: {raw}")
    path = (root / candidate).resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise QualificationError(f"dependency escapes workspace: {raw}") from error


def build_dependency_packet(
        root: Path, *, target_path: str, reached_paths: Sequence[str],
        parser_capability: Mapping[str, object] | str,
) -> DependencyPacket:
    """Hash only reached/imported disk files plus parser capability and workspace."""
    root = root.resolve()
    workspace_identity = _digest({"workspace": root.as_posix()})
    parser_identity = _digest(parser_capability)
    identities: list[DependencyIdentity] = []
    for raw_path in sorted(set(reached_paths)):
        path = _relative_path(root, raw_path)
        file_path = root / path
        try:
            if not file_path.is_file() or file_path.is_symlink():
                raise OSError("not a regular file")
            content = file_path.read_bytes()
        except (OSError, UnicodeError) as error:
            raise QualificationError(f"dependency could not be read: {path}") from error
        content_identity = "sha256:" + hashlib.sha256(content).hexdigest()
        identities.append(DependencyIdentity(
            workspace_identity=workspace_identity,
            path=path,
            revision=content_identity,
            content_identity=content_identity,
            parser_capability=parser_identity,
            read_source="disk",
        ))
    packet_key = _digest({
        "workspace": workspace_identity,
        "target": _relative_path(root, target_path),
        "dependencies": [identity.__dict__ for identity in identities],
    })
    return DependencyPacket(
        cache_key=packet_key,
        workspace_identity=workspace_identity,
        target_path=_relative_path(root, target_path),
        dependencies=tuple(identities),
    )


class PacketCache:
    def __init__(self) -> None:
        self._packets: dict[str, DependencyPacket] = {}
        self.builds = 0

    def get_or_build(
            self, root: Path, *, target_path: str, reached_paths: Sequence[str],
            parser_capability: Mapping[str, object] | str,
    ) -> tuple[DependencyPacket, bool]:
        candidate = build_dependency_packet(
            root, target_path=target_path, reached_paths=reached_paths,
            parser_capability=parser_capability,
        )
        cached = self._packets.get(candidate.cache_key)
        if cached is not None:
            return cached, True
        self._packets[candidate.cache_key] = candidate
        self.builds += 1
        return candidate, False


# ---------------------------------------------------------------------------
# SCHEMA and SCOPE: full contract identity and predicate-scoped presentation


def full_schema_identity(schemas: Sequence[Mapping[str, object]]) -> str:
    normalized = tuple(sorted((_canonical(schema) for schema in schemas), key=lambda item: json.dumps(item, sort_keys=True)))
    return _digest(normalized)


@dataclass(frozen=True)
class RequestContract:
    request_id: str
    schema_identity: str
    provider_identity: str
    dispatcher_identity: str
    reused: bool


class SchemaContractCache:
    def __init__(self) -> None:
        self._contracts: dict[str, RequestContract] = {}

    def resolve(
            self, request_id: str, *, provider: Sequence[Mapping[str, object]],
            dispatcher: Sequence[Mapping[str, object]],
    ) -> RequestContract:
        provider_identity = full_schema_identity(provider)
        dispatcher_identity = full_schema_identity(dispatcher)
        if provider_identity != dispatcher_identity:
            raise QualificationError("provider and dispatcher full schemas differ")
        existing = self._contracts.get(request_id)
        if existing is not None and existing.schema_identity != provider_identity:
            raise QualificationError("same-request schema mutation rejected fail-closed")
        if existing is not None:
            return RequestContract(
                request_id=request_id,
                schema_identity=provider_identity,
                provider_identity=provider_identity,
                dispatcher_identity=dispatcher_identity,
                reused=True,
            )
        contract = RequestContract(
            request_id=request_id,
            schema_identity=provider_identity,
            provider_identity=provider_identity,
            dispatcher_identity=dispatcher_identity,
            reused=False,
        )
        self._contracts[request_id] = contract
        return contract


@dataclass(frozen=True)
class Predicate:
    name: str
    status: str
    evidence_refs: tuple[str, ...] = ()
    next_action: str | None = None
    executable: bool = False


@dataclass(frozen=True)
class ScopeFrame:
    predicates: tuple[Predicate, ...]
    sidecars: tuple[Mapping[str, object], ...]
    truthful_final: bool
    next_actions: Mapping[str, str]

    def serialized(self) -> str:
        return json.dumps({
            "schema": "predicate-scoped-frame/1",
            "predicates": [predicate.__dict__ for predicate in self.predicates],
            "sidecars": list(self.sidecars),
            "truthful_final": self.truthful_final,
            "next_actions": dict(self.next_actions),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def qualify_scope(predicates: Sequence[Predicate], *, sidecars: Sequence[Mapping[str, object]] = ()) -> ScopeFrame:
    by_name = {predicate.name: predicate for predicate in predicates}
    if tuple(sorted(by_name)) != tuple(sorted(PREDICATES)):
        raise QualificationError("scope frame must preserve all import/coordination/behavioral/completion predicates")
    actions: dict[str, str] = {}
    for name in PREDICATES:
        predicate = by_name[name]
        if predicate.status not in {"VERIFIED", "UNRESOLVED", "UNKNOWN"}:
            raise QualificationError(f"invalid predicate state: {predicate.status}")
        if predicate.status != "VERIFIED":
            if not predicate.next_action or not predicate.executable:
                raise QualificationError(f"unresolved predicate lacks executable route: {name}")
            actions[name] = predicate.next_action
    ordered = tuple(by_name[name] for name in PREDICATES)
    return ScopeFrame(
        predicates=ordered,
        sidecars=tuple(dict(sidecar) for sidecar in sidecars),
        truthful_final=all(predicate.status == "VERIFIED" for predicate in ordered),
        next_actions=actions,
    )


# ---------------------------------------------------------------------------
# K11-K20: unrun, answer-keyed, opaque-label screen specifications


@dataclass(frozen=True)
class FixtureSymbol:
    case: str
    language: str
    path: str
    symbol: str
    enclosing: str
    span: tuple[int, int]


@dataclass(frozen=True)
class DecisionScreenSpec:
    screen_id: str
    languages: tuple[str, ...]
    state_keys: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    question: str
    contrast: str
    fixtures: tuple[FixtureSymbol, ...]
    answer_labels: tuple[str, ...] = ("C1", "C2", "C3", "C4")
    append_question: bool = False
    calls: int = 1
    thinking: bool = False
    temperature: float = 0.7
    max_tokens: int = 120
    seeds: tuple[int, ...] = Q2_SEEDS
    max_calls: int = Q2_MAX_CALLS


def _fixture(case: str, language: str, path: str, symbol: str, enclosing: str, span: tuple[int, int]) -> FixtureSymbol:
    return FixtureSymbol(case, language, path, symbol, enclosing, span)


_GO = _fixture("go-1", "go", "render/reader.go", "writeHeaders", "Reader", (15, 46))
_TS1 = _fixture("typescript-1", "typescript", "packages/babel-generator/src/source-map.ts", "SourceMap", "constructor", (48, 65))
_TS2 = _fixture("typescript-2", "typescript", "packages/babel-helpers/src/index.ts", "getDependencies", "loadHelper", (332, 421))
_JAVA = _fixture("java-2", "java", "gson/src/main/java/com/google/gson/internal/bind/JsonTreeReader.java", "hasNext", "JsonTreeReader", (102, 110))
_RUST = _fixture("rust-1", "rust", "tokio/src/runtime/scheduler/multi_thread/handle/metrics.rs", "num_blocking_threads", "Handle", (10, 12))
_PHP3 = _fixture("php-3", "php", "src/PhpSpreadsheet/Calculation/LookupRef/HLookup.php", "lookup", "HLookup", (72, 118))
_PHP6 = _fixture("php-6", "php", "lib/PhpParser/BuilderHelpers.php", "normalizeValue", "BuilderHelpers", (768, 806))


SCREEN_SPECS: tuple[DecisionScreenSpec, ...] = (
    DecisionScreenSpec("K11", ("go", "typescript", "java"), ("fault", "safe"), ("input", "expression", "invariant"), "Choose the first expression violating the supplied invariant, or NO_EDIT when the invariant holds.", "contiguous decision source versus interleaved sidecars", (_GO, _TS1, _JAVA)),
    DecisionScreenSpec("K12", ("go", "typescript", "java"), ("P", "R"), ("edit_receipt", "source_revision", "admission"), "Choose the next action using the linked edit effect and current check admission.", "effect-linked applied/refused edit versus chronological auto-check presentation", (_GO, _TS1, _JAVA)),
    DecisionScreenSpec("K13", ("go", "typescript", "java"), ("U", "V"), ("behavioral_receipt", "completion_scope", "test_admission"), "Choose the next action for the stated scope; do not substitute import or coordination status for behavioral evidence.", "scope-indexed unresolved versus verified predicate next action", (_GO, _TS1, _JAVA)),
    DecisionScreenSpec("K14", ("rust", "java", "php"), ("N", "S"), ("request_id", "path", "span", "revision"), "Choose the offered edit whose complete argument contract is current.", "same-name action with changed versus unchanged full argument contract", (_RUST, _JAVA, _PHP3)),
    DecisionScreenSpec("K15", ("go", "typescript", "php"), ("changed_dependency", "unchanged_packet"), ("dependency_identity", "packet_basis", "current_declaration"), "Choose READ of the current dependency or the exact supported call from the unchanged packet.", "dependency revision requires lookup versus unchanged packet permits exact call", (_GO, _TS2, _PHP6)),
    DecisionScreenSpec("K16", ("go", "typescript", "java"), ("R", "U"), ("product_edge", "input", "assertion_origin"), "Choose the test reaching the real product, or ACQUIRE the bounded fixture when no such test is available.", "real product path versus replica, spy, always-true test, or bounded acquisition", (_GO, _TS1, _JAVA)),
    DecisionScreenSpec("K17", ("rust", "java", "php"), ("E", "C"), ("request_revision", "plan_basis", "scope"), "Choose work for the current user request, not an obsolete plan.", "current expanded scope versus current limited scope", (_RUST, _JAVA, _PHP3)),
    DecisionScreenSpec("K18", ("rust", "java", "php"), ("D", "I"), ("precondition", "owner_span", "sequence"), "Choose the largest offered batch whose actions are useful now and independent of unknown results.", "dependent read/edit sequence versus two independent reads", (_RUST, _JAVA, _PHP3)),
    DecisionScreenSpec("K19", ("go", "typescript", "php"), ("P", "C"), ("page_range", "search_scope", "declaration"), "Choose LOOKUP when the page is complete but search scope is incomplete, otherwise CALL the source-bound helper.", "page-complete but search-incomplete lookup versus source-bound call", (_GO, _TS2, _PHP6)),
    DecisionScreenSpec("K20", ("go", "typescript", "java"), ("B", "F"), ("preservation_invariant", "blocker", "current_patch"), "Choose verification-only after a solved repair, or the minimal correction for a real preservation regression.", "verification-only versus minimal correction for a real preservation regression", (_GO, _TS1, _JAVA)),
)


def _fixture_content(fixture_root: Path, case: str) -> str:
    frame_path = fixture_root / case / "frame.json"
    try:
        value = json.loads(frame_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise QualificationError(f"fixture cannot be read: {case}") from error
    if not isinstance(value, list) or not value or not isinstance(value[0], Mapping):
        raise QualificationError(f"fixture messages missing: {case}")
    content = value[0].get("content")
    if not isinstance(content, str):
        raise QualificationError(f"fixture content missing: {case}")
    return content


def verify_fixture_symbols(fixture_root: Path, spec: DecisionScreenSpec) -> tuple[Mapping[str, object], ...]:
    """Verify each named symbol and source section before screen construction."""
    receipts: list[Mapping[str, object]] = []
    for fixture in spec.fixtures:
        content = _fixture_content(fixture_root, fixture.case)
        marker = f"=== {fixture.path} ==="
        start = content.find(marker)
        if start < 0:
            raise QualificationError(f"missing source section for {fixture.case}:{fixture.path}")
        section = content[start:]
        next_section = section.find("\n=== ", len(marker))
        if next_section >= 0:
            section = section[:next_section]
        if fixture.symbol not in section or fixture.enclosing not in section:
            raise QualificationError(f"symbol/enclosing span not verified: {fixture.case}:{fixture.symbol}")
        receipts.append({
            "case": fixture.case,
            "language": fixture.language,
            "path": fixture.path,
            "symbol": fixture.symbol,
            "enclosing": fixture.enclosing,
            "span": fixture.span,
            "revision": _digest(section),
            "status": "VERIFIED",
        })
    return tuple(receipts)


def build_screen_payload(
        spec: DecisionScreenSpec, *, fixture_root: Path,
        source_blocks: Mapping[str, str], state_key: str,
) -> Mapping[str, object]:
    """Build a public-only screen payload after fixture and wire validation."""
    validate_screen_spec(spec)
    if state_key not in spec.state_keys:
        raise QualificationError(f"state is not part of {spec.screen_id}: {state_key}")
    receipts = verify_fixture_symbols(fixture_root, spec)
    if set(source_blocks) != set(spec.languages):
        raise QualificationError("screen source blocks must cover exactly the named languages")
    public_source = "\n".join(str(source_blocks[language]) for language in spec.languages)
    lowered = public_source.casefold()
    leaked = sorted(word for word in _LEAK_WORDS if word in lowered)
    if leaked:
        raise QualificationError(f"forbidden evaluator/reference leakage in public frame: {leaked}")
    question_wire = build_question_wire(
        [{"role": "user", "content": spec.question}],
        intended_question=spec.question,
        policy="splice",
        append_question=False,
    )
    return {
        "screen_id": spec.screen_id,
        "state": state_key,
        "question": spec.question,
        "messages": json.loads(question_wire.serialized_messages),
        "question_wire": {
            "policy": question_wire.policy,
            "append_question": question_wire.append_question,
            "question_occurrences": question_wire.question_occurrences,
            "serialized_messages": question_wire.serialized_messages,
            "semantic_valid": question_wire.semantic_valid,
        },
        "source": dict(source_blocks),
        "evidence_refs": list(spec.evidence_refs),
        "candidate_labels": list(spec.answer_labels),
        "fixture_receipts": list(receipts),
        "wire": {
            "append_question": False,
            "calls": 1,
            "thinking": False,
            "temperature": 0.7,
            "max_tokens": 120,
            "seeds": list(spec.seeds),
            "max_calls": spec.max_calls,
        },
    }


def validate_screen_spec(spec: DecisionScreenSpec) -> None:
    if len(spec.languages) < 3 or "python" in {language.casefold() for language in spec.languages}:
        raise QualificationError(f"{spec.screen_id} needs at least three non-Python languages")
    if not spec.question.strip() or spec.append_question or spec.calls != 1 or spec.thinking:
        raise QualificationError(f"{spec.screen_id} violates explicit Q2 wire contract")
    if spec.temperature != 0.7 or spec.max_tokens != 120 or spec.max_calls > Q2_MAX_CALLS:
        raise QualificationError(f"{spec.screen_id} violates bounded screen budget")
    if tuple(spec.seeds) != Q2_SEEDS or any(not label.startswith("C") for label in spec.answer_labels):
        raise QualificationError(f"{spec.screen_id} lacks fixed paired opaque labels/seeds")
    if len(set(spec.state_keys)) != len(spec.state_keys) or not spec.evidence_refs:
        raise QualificationError(f"{spec.screen_id} lacks exact state/evidence keys")
    if any(word in (spec.question + " " + spec.contrast).casefold() for word in _LEAK_WORDS):
        raise QualificationError(f"{spec.screen_id} contains evaluator/control leakage")


@dataclass(frozen=True)
class ScreenOutcome:
    transport: str
    truncation: str
    parse: str
    semantic: str


def score_screen_response(
        raw: object, *, expected_choice: str,
        expected_refs: Sequence[str], expected_arguments: object = None,
        expected_sequence: object = None,
) -> ScreenOutcome:
    """Score semantic JSON independently from transport/length/parse outcomes."""
    if not isinstance(raw, str):
        return ScreenOutcome("NO_RESPONSE", "UNKNOWN", "NOT_PARSED", "NOT_SCORED")
    if len(raw) > MAX_SCREEN_OUTPUT_CHARS:
        return ScreenOutcome("RECEIVED", "TRUNCATED", "NOT_PARSED", "NOT_SCORED")
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ScreenOutcome("RECEIVED", "NOT_TRUNCATED", "MALFORMED", "NOT_SCORED")
    if not isinstance(value, Mapping):
        return ScreenOutcome("RECEIVED", "NOT_TRUNCATED", "PARSED", "SEMANTIC_FAIL")
    if value.get("choice") != expected_choice or tuple(value.get("refs", ())) != tuple(expected_refs):
        return ScreenOutcome("RECEIVED", "NOT_TRUNCATED", "PARSED", "SEMANTIC_FAIL")
    if expected_arguments is not None and _canonical(value.get("arguments")) != _canonical(expected_arguments):
        return ScreenOutcome("RECEIVED", "NOT_TRUNCATED", "PARSED", "SEMANTIC_FAIL")
    if expected_sequence is not None and _canonical(value.get("sequence")) != _canonical(expected_sequence):
        return ScreenOutcome("RECEIVED", "NOT_TRUNCATED", "PARSED", "SEMANTIC_FAIL")
    return ScreenOutcome("RECEIVED", "NOT_TRUNCATED", "PARSED", "SEMANTIC_PASS")


def screen_catalog() -> tuple[DecisionScreenSpec, ...]:
    """Return a stable copy-like tuple; no queue or hypothesis file is touched."""
    return SCREEN_SPECS

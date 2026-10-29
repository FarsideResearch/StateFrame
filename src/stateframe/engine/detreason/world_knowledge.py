"""Per-sandbox deterministic world-knowledge projection.

The mission event stream remains authoritative.  ``WorldKnowledgeStore`` is a
rebuildable projection: source observations and invalidations can be emitted to
an existing receipt/event adapter, replayed after process death, and rendered as
small, provenance-bearing answers.  No model call is made in this slice.

SNITCH law: every answer carries its requested revision, fact provenance, and a
deterministic freshness gate.  A stale or contradicted fact can be returned only
as explicitly labelled history; it is never represented as a fresh answer.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TypeAlias

from kernel_launch_identity_cost import PhaseMeter

from .world_knowledge_extractors import (
    ExtractedFact,
    bounded_line_count,
    extract_manifest_facts,
    extract_source_facts,
    extract_test_selector,
    normalize_dependency,
    normalize_path,
    source_digest,
    source_extractor_version,
)

try:
    from kernel_liveness import Evidence, OrganCensusSpec, TriState
except ImportError:  # pragma: no cover - the repository ships kernel_liveness.py
    Evidence = object  # type: ignore[assignment,misc]
    OrganCensusSpec = object  # type: ignore[assignment,misc]

    class TriState:
        YES = "YES"
        NO = "NO"
        UNKNOWN = "UNKNOWN"


SCHEMA = "world.fact/1"
EVENT_SCHEMA = "poc.world_knowledge.event/1"
DETERMINISTIC = "DETERMINISTIC"
LLM_PROPOSAL = "LLM_PROPOSAL"
D0_EXACT = "D0_EXACT"
L0_UNVERIFIED = "L0_UNVERIFIED"
FRESH = "FRESH"
STALE = "STALE"
UNKNOWN = "UNKNOWN"
CONTRADICTED = "CONTRADICTED"

LLM_PROSE_PREDICATES = frozenset(
    {
        "function_intent_summary",
        "test_intent_summary",
        "guard_rationale_summary",
        "file_role_summary",
    }
)
_DETERMINISTIC_PREDICATES = frozenset(
    {
        "file_digest",
        "manifest_digest",
        "parse_error",
        "parser_unsupported",
        "test_defined",
        "symbol_defined",
        "import_edge",
        "guard_predicate",
        "oracle_selector",
        "static_references_symbol",
        "runtime_exercises_symbol",
    }
)
_SUPPORTED_QUERIES = frozenset(
    {
        "tests_cover",
        "tests_covered",
        "what_tests_cover",
        "tests",
        "signature",
        "signatures",
        "imports",
        "import",
        "guards",
        "guard",
        "oracles",
        "oracle",
        "selectors",
        "file_facts",
        "facts",
        "all",
    }
)
_WORLD_EVENTS = frozenset(
    {
        "world_knowledge_observed",
        "world_knowledge_reobserved",
        "world_knowledge_invalidated",
        "world_knowledge_query_hit",
        "world_knowledge_query_miss",
        "world_knowledge_query_unknown",
        "world_knowledge_deduped",
        "world_knowledge_snitch",
    }
)


JSONValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | Mapping[str, "JSONValue"]
    | Sequence["JSONValue"]
)
EventSink: TypeAlias = Callable[[Mapping[str, object]], object]


@dataclass(frozen=True, slots=True)
class _SourceCacheEntry:
    content_digest: str
    extractor_version: str
    fact_set_id: str
    drafts: tuple[ExtractedFact, ...]


@dataclass(frozen=True, slots=True)
class _SourceFactSetMember:
    fact_id: str
    path: str
    workspace_revision: int
    content_digest: str
    spans: tuple[tuple[int, int], ...]
    extractor: str


_SOURCE_CACHE_LIMIT = 64


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _scope(value: Mapping[str, object] | None, **overrides: str) -> dict[str, str]:
    result = {str(key): str(item) for key, item in dict(value or {}).items() if item is not None}
    for key, item in overrides.items():
        if item:
            result[key] = str(item)
    if not result.get("sandbox_id"):
        raise ValueError("sandbox_id is required for world-knowledge isolation")
    for key in ("workspace_id", "mission_id", "sandbox_id"):
        result.setdefault(key, "")
    return {key: result[key] for key in sorted(result)}


def _scope_key(value: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(item)) for key, item in value.items()))


def _span_dicts(spans: Sequence[tuple[int, int]]) -> tuple[dict[str, int], ...]:
    return tuple({"start": int(start), "end": int(end)} for start, end in spans)


def _spans(value: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    result: list[tuple[int, int]] = []
    for item in value:
        if isinstance(item, Mapping):
            start, end = item.get("start"), item.get("end")
        elif isinstance(item, Sequence) and len(item) == 2:
            start, end = item
        else:
            continue
        try:
            result.append((int(start), int(end)))
        except (TypeError, ValueError):
            continue
    return tuple(result)


@dataclass(frozen=True)
class FactProvenance:
    """The exact source witness attached to one fact observation."""

    evidence_ref: str
    path: str
    workspace_revision: int
    content_digest: str
    spans: tuple[tuple[int, int], ...] = ()
    extractor: str = "unknown"

    def __post_init__(self) -> None:
        if not str(self.evidence_ref).strip():
            raise ValueError("every fact provenance entry needs evidence_ref")
        object.__setattr__(self, "evidence_ref", str(self.evidence_ref).strip())
        object.__setattr__(self, "path", normalize_path(self.path))
        object.__setattr__(self, "workspace_revision", int(self.workspace_revision))
        object.__setattr__(self, "spans", _spans(self.spans))
        digest = str(self.content_digest).strip()
        if self.content_digest is None or not digest:
            raise ValueError("every fact provenance entry needs content_digest")
        object.__setattr__(self, "content_digest", digest)

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_ref": self.evidence_ref,
            "path": self.path,
            "workspace_revision": self.workspace_revision,
            "content_digest": self.content_digest,
            "spans": list(_span_dicts(self.spans)),
            "extractor": self.extractor,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FactProvenance":
        return cls(
            evidence_ref=str(value.get("evidence_ref") or value.get("ref") or ""),
            path=str(value.get("path") or "<unknown>"),
            workspace_revision=int(value.get("workspace_revision", value.get("source_revision", 0))),
            content_digest=str(value.get("content_digest") or value.get("digest") or ""),
            spans=_spans(value.get("spans")),
            extractor=str(value.get("extractor") or "unknown"),
        )


@dataclass(frozen=True)
class WorldFact:
    """One versioned observation, retaining history instead of deleting it."""

    scope: Mapping[str, str]
    path: str
    workspace_revision: int
    subject_ref: str
    predicate: str
    value: Mapping[str, object]
    content_digest: str
    spans: tuple[tuple[int, int], ...]
    evidence_ref: str
    dependencies: tuple[str, ...] = ()
    dependencies_complete: bool = True
    verification: str = DETERMINISTIC
    confidence_tier: str = D0_EXACT
    freshness: str = FRESH
    extractor: str = "unknown"
    provenance: tuple[FactProvenance, ...] = ()
    invalidated_by: Mapping[str, object] | None = None
    observation_count: int = 1
    fact_id: str = field(init=False)

    def __post_init__(self) -> None:
        fact_scope = {str(key): str(item) for key, item in self.scope.items()}
        if not fact_scope.get("sandbox_id"):
            raise ValueError("world facts need sandbox_id")
        object.__setattr__(self, "scope", {key: fact_scope[key] for key in sorted(fact_scope)})
        object.__setattr__(self, "path", normalize_path(self.path))
        revision = int(self.workspace_revision)
        if revision < 0:
            raise ValueError("workspace_revision must be non-negative")
        object.__setattr__(self, "workspace_revision", revision)
        subject = str(self.subject_ref).strip()
        predicate = str(self.predicate).strip()
        if not subject or not predicate:
            raise ValueError("world facts need subject_ref and predicate")
        object.__setattr__(self, "subject_ref", subject)
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "value", dict(self.value))
        object.__setattr__(self, "spans", _spans(self.spans))
        object.__setattr__(self, "evidence_ref", str(self.evidence_ref).strip())
        if not self.evidence_ref:
            raise ValueError("world facts need evidence_ref")
        digest = str(self.content_digest).strip()
        if self.content_digest is None or not digest:
            raise ValueError("world facts need content_digest")
        object.__setattr__(self, "content_digest", digest)
        deps = {normalize_dependency(dep) for dep in self.dependencies if str(dep).strip()}
        deps.add(self.path)
        object.__setattr__(self, "dependencies", tuple(sorted(deps)))
        count = int(self.observation_count)
        if count < 1:
            raise ValueError("observation_count must be positive")
        object.__setattr__(self, "observation_count", count)
        provenance = tuple(self.provenance)
        if not provenance:
            provenance = (
                FactProvenance(
                    evidence_ref=self.evidence_ref,
                    path=self.path,
                    workspace_revision=revision,
                    content_digest=self.content_digest,
                    spans=self.spans,
                    extractor=self.extractor,
                ),
            )
        object.__setattr__(self, "provenance", provenance)
        identity = {
            "scope": self.scope,
            "path": self.path,
            "workspace_revision": revision,
            "subject_ref": subject,
            "predicate": predicate,
            "predicate_version": 1,
            "value_digest": self.value_digest,
        }
        object.__setattr__(self, "fact_id", "fact:" + _digest(identity).removeprefix("sha256:"))

    @property
    def value_digest(self) -> str:
        return _digest(self.value)

    @property
    def stale(self) -> bool:
        return self.freshness != FRESH or self.verification != DETERMINISTIC

    @property
    def logical_key(self) -> tuple[object, ...]:
        return (
            _scope_key(self.scope),
            self.path,
            self.workspace_revision,
            self.subject_ref,
            self.predicate,
        )

    @property
    def key(self) -> dict[str, object]:
        return {
            "path": self.path,
            "workspace_revision": self.workspace_revision,
            "subject_ref": self.subject_ref,
            "predicate": self.predicate,
            "predicate_version": 1,
        }

    @property
    def source(self) -> dict[str, object]:
        return {
            "content_digest": self.content_digest,
            "spans": list(_span_dicts(self.spans)),
            "evidence_ref": self.evidence_ref,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "fact_id": self.fact_id,
            "scope": dict(self.scope),
            "key": self.key,
            "value": dict(self.value),
            "value_digest": self.value_digest,
            "source": self.source,
            "dependencies": list(self.dependencies),
            "dependencies_complete": self.dependencies_complete,
            "verification": self.verification,
            "confidence_tier": self.confidence_tier,
            "freshness": self.freshness,
            "extractor": {"name": self.extractor.split("/", 1)[0],
                          "version": self.extractor.split("/", 1)[1]
                          if "/" in self.extractor else ""},
            "provenance": [item.to_dict() for item in self.provenance],
            "invalidated_by": dict(self.invalidated_by) if self.invalidated_by else None,
            "observation_count": self.observation_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "WorldFact":
        key = value.get("key") if isinstance(value.get("key"), Mapping) else value
        source = value.get("source") if isinstance(value.get("source"), Mapping) else value
        extractor_value = value.get("extractor")
        if isinstance(extractor_value, Mapping):
            name = str(extractor_value.get("name") or "unknown")
            version = str(extractor_value.get("version") or "")
            extractor = f"{name}/{version}".rstrip("/")
        else:
            extractor = str(extractor_value or "unknown")
        provenance_values = value.get("provenance")
        provenance = tuple(
            FactProvenance.from_dict(item)
            for item in provenance_values
            if isinstance(item, Mapping)
        ) if isinstance(provenance_values, Sequence) and not isinstance(provenance_values, (str, bytes)) else ()
        return cls(
            scope=value.get("scope") if isinstance(value.get("scope"), Mapping) else {},
            path=str(key.get("path") or ""),
            workspace_revision=int(key.get("workspace_revision", 0)),
            subject_ref=str(key.get("subject_ref") or ""),
            predicate=str(key.get("predicate") or ""),
            value=value.get("value") if isinstance(value.get("value"), Mapping) else {},
            content_digest=str(source.get("content_digest") or ""),
            spans=_spans(source.get("spans")),
            evidence_ref=str(source.get("evidence_ref") or ""),
            dependencies=tuple(str(item) for item in value.get("dependencies", ())
                               if str(item).strip())
            if isinstance(value.get("dependencies", ()), Sequence)
            and not isinstance(value.get("dependencies", ()), (str, bytes)) else (),
            dependencies_complete=bool(value.get("dependencies_complete", True)),
            verification=str(value.get("verification") or DETERMINISTIC),
            confidence_tier=str(value.get("confidence_tier") or D0_EXACT),
            freshness=str(value.get("freshness") or FRESH),
            extractor=extractor,
            provenance=provenance,
            invalidated_by=value.get("invalidated_by")
            if isinstance(value.get("invalidated_by"), Mapping) else None,
            observation_count=int(value.get("observation_count", 1)),
        )


@dataclass(frozen=True)
class KnowledgeAnswer:
    """A fail-closed query result with fresh facts separated from history."""

    query: str
    status: str
    source_revision: int
    facts: tuple[WorldFact, ...] = ()
    stale_facts: tuple[WorldFact, ...] = ()
    stale: bool = False
    changed_paths: tuple[str, ...] = ()
    recommendation: str | None = None
    provenance: tuple[FactProvenance, ...] = ()
    knowledge_digest: str = ""

    def __post_init__(self) -> None:
        fresh = tuple(self.facts)
        stale = tuple(self.stale_facts)
        object.__setattr__(self, "facts", fresh)
        object.__setattr__(self, "stale_facts", stale)
        refs: list[FactProvenance] = list(self.provenance)
        seen = {(item.evidence_ref, item.path, item.workspace_revision) for item in refs}
        for fact in (*fresh, *stale):
            for item in fact.provenance:
                key = (item.evidence_ref, item.path, item.workspace_revision)
                if key not in seen:
                    seen.add(key)
                    refs.append(item)
        object.__setattr__(self, "provenance", tuple(refs))
        if not self.knowledge_digest:
            object.__setattr__(
                self,
                "knowledge_digest",
                _digest(
                    {
                        "query": self.query,
                        "revision": self.source_revision,
                        "facts": [fact.fact_id for fact in (*fresh, *stale)],
                        "stale": self.stale,
                    }
                ),
            )

    @property
    def usable(self) -> bool:
        return self.status == "HIT" and bool(self.facts) and not self.stale

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "query": self.query,
            "source_revision": self.source_revision,
            "facts": [fact.to_dict() for fact in self.facts],
            "stale_facts": [fact.to_dict() for fact in self.stale_facts],
            "stale": self.stale,
            "changed_paths": list(self.changed_paths),
            "recommendation": self.recommendation,
            "provenance": [item.to_dict() for item in self.provenance],
            "knowledge_digest": self.knowledge_digest,
            "declaration": {
                "requested_revision": self.source_revision,
                "freshness_gate": "deterministic_exact_revision_and_digest",
                "provenance_required": True,
            },
        }

    def render(self, max_tokens: int = 220) -> str:
        return render_knowledge_answer(self, max_tokens=max_tokens)


class WorldKnowledgeError(RuntimeError):
    """Base error for malformed or non-durable world-knowledge operations."""


class LLMProposalRejected(WorldKnowledgeError):
    """An LLM proposal referenced a subject or span not proven by source facts."""


class WorldKnowledgeStore:
    """A per-sandbox, replayable projection of deterministic source facts."""

    def __init__(
        self,
        scope: Mapping[str, object] | str | None = None,
        *,
        workspace_id: str = "",
        mission_id: str = "",
        sandbox_id: str = "",
        event_sink: EventSink | object | None = None,
    ) -> None:
        if isinstance(scope, str):
            scope = {"sandbox_id": scope}
        self.scope = _scope(
            scope,
            workspace_id=workspace_id,
            mission_id=mission_id,
            sandbox_id=sandbox_id,
        )
        self.event_sink = event_sink
        self._facts: dict[str, WorldFact] = {}
        self._logical: dict[tuple[object, ...], dict[str, str]] = {}
        self._dependency_fact_ids: dict[str, set[str]] = {}
        self._incomplete_dependency_fact_ids: set[str] = set()
        self._events: list[dict[str, object]] = []
        self._seen_event_ids: set[str] = set()
        self._observation_sequence = 0
        self._invalidation_records: list[dict[str, object]] = []
        self._invalidation_event_ids: set[str] = set()
        self._path_revisions: dict[tuple[str, int], str] = {}
        self._source_cache: OrderedDict[
            tuple[str, int, str, str], _SourceCacheEntry
        ] = OrderedDict()
        self._source_fact_set_members: dict[
            str, dict[int, _SourceFactSetMember]
        ] = {}
        self._source_fact_set_lru: OrderedDict[str, None] = OrderedDict()
        self._durable_source_fact_sets: dict[str, int] = {}
        self.cost_meter = PhaseMeter()
        self._proposals: list[WorldFact] = []
        self._trace: list[dict[str, object]] = []
        self.current_revision = 0
        self.source_archive = None

    def enable_source_archive(self, run_scope: str) -> None:
        """Attach retained bytes to this store; the caller supplies a unique run ID."""
        from detreason.source_archive import SourceArchive
        if self.source_archive is not None:
            if self.source_archive.scope != run_scope:
                raise ValueError('cannot change the scope of retained sources')
            return
        self.source_archive = SourceArchive(run_scope, sink=self._emit)

    @property
    def facts(self) -> tuple[WorldFact, ...]:
        return tuple(sorted(self._facts.values(), key=self._fact_sort_key))

    @property
    def events(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(event) for event in self._events)

    @property
    def llm_proposals(self) -> tuple[WorldFact, ...]:
        return tuple(self._proposals)

    @property
    def trace(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(event) for event in self._trace)

    @property
    def cost_metrics(self) -> dict[str, object]:
        """Return bounded phase diagnostics, never a product verdict."""
        return self.cost_meter.snapshot()

    @staticmethod
    def _fact_sort_key(fact: WorldFact) -> tuple[object, ...]:
        return (fact.path, fact.workspace_revision, fact.predicate, fact.subject_ref, fact.fact_id)

    def _emit(self, event: Mapping[str, object]) -> None:
        payload = dict(event)
        event_id = str(payload.get("event_id") or _digest(payload))
        payload["event_id"] = event_id
        if self.event_sink is None:
            self._events.append(payload)
            self._seen_event_ids.add(event_id)
            return
        try:
            if callable(self.event_sink):
                self.event_sink(payload)
            elif hasattr(self.event_sink, "append"):
                self.event_sink.append(payload)  # type: ignore[union-attr]
            else:
                raise TypeError("event_sink must be callable or expose append")
        except Exception as exc:  # noqa: BLE001 - durable adapter failure is explicit
            raise WorldKnowledgeError(
                f"world-knowledge event sink failed: {type(exc).__name__}"
            ) from exc
        self._events.append(payload)
        self._seen_event_ids.add(event_id)

    def _trace_event(self, event: str, detail: str, **fields: object) -> None:
        self._trace.append(
            {
                "cylinder": "world.knowledge",
                "kind": "world_knowledge",
                "event": event,
                "detail": detail[:180],
                **fields,
            }
        )

    def _source_fact(
        self,
        draft: ExtractedFact,
        revision: int,
        evidence_ref: str,
        content_digest: str | None = None,
    ) -> WorldFact:
        freshness = self._freshness_for_new_fact(
            draft.path,
            revision,
            draft.dependencies,
            draft.dependencies_complete,
        )
        invalidated_by = (
            self._invalidation_for(
                draft.path,
                draft.dependencies,
                draft.dependencies_complete,
                revision,
            )
            if freshness != FRESH
            else None
        )
        return WorldFact(
            scope=self.scope,
            path=draft.path,
            workspace_revision=revision,
            subject_ref=draft.subject_ref,
            predicate=draft.predicate,
            value=draft.value,
            content_digest=content_digest or self._content_digest_for_path(draft.path, draft),
            spans=draft.spans,
            evidence_ref=evidence_ref,
            dependencies=draft.dependencies,
            dependencies_complete=draft.dependencies_complete,
            verification=DETERMINISTIC,
            confidence_tier=D0_EXACT,
            freshness=freshness,
            extractor=draft.extractor,
            invalidated_by=invalidated_by,
        )

    @staticmethod
    def _content_digest_for_path(path: str, draft: ExtractedFact) -> str:
        value_digest = draft.value.get("digest") if draft.predicate in {
            "file_digest", "manifest_digest"
        } else None
        return str(value_digest or draft.value_digest)

    def _freshness_for_new_fact(
        self,
        path: str,
        revision: int,
        dependencies: Sequence[str],
        dependencies_complete: bool,
    ) -> str:
        if revision >= self.current_revision:
            return FRESH
        if not dependencies_complete and self._has_edit_after(revision):
            return UNKNOWN
        if self._dependencies_changed(path, dependencies, revision):
            return STALE
        return FRESH

    def _has_edit_after(self, revision: int) -> bool:
        return any(int(item["after_revision"]) > revision for item in self._invalidation_records)

    @staticmethod
    def _path_matches(dependency: str, changed_path: str) -> bool:
        left = normalize_dependency(dependency)
        return left in WorldKnowledgeStore._dependency_match_tokens(changed_path)

    @staticmethod
    def _dependency_match_tokens(changed_path: str) -> frozenset[str]:
        right = normalize_dependency(changed_path)
        return frozenset({
            right,
            right.removesuffix(".py"),
            right.removesuffix("/__init__.py"),
            right.replace("/", ".").removesuffix(".py"),
        })

    def _dependencies_changed(
        self,
        path: str,
        dependencies: Sequence[str],
        since_revision: int,
    ) -> bool:
        for item in self._invalidation_records:
            if int(item["after_revision"]) <= since_revision:
                continue
            changed = str(item["path"])
            if self._path_matches(path, changed):
                return True
            if any(self._path_matches(dep, changed) for dep in dependencies):
                return True
        return False

    def _invalidation_for(
        self,
        path: str,
        dependencies: Sequence[str],
        dependencies_complete: bool,
        since_revision: int,
    ) -> dict[str, object] | None:
        for record in reversed(self._invalidation_records):
            if int(record["after_revision"]) <= since_revision:
                continue
            changed = str(record["path"])
            if (
                self._path_matches(path, changed)
                or any(self._path_matches(dep, changed) for dep in dependencies)
                or not dependencies_complete
            ):
                return dict(record)
        return None

    def _refresh_status(self, fact: WorldFact, revision: int) -> str:
        if fact.verification != DETERMINISTIC or fact.confidence_tier != D0_EXACT:
            return UNKNOWN
        if fact.workspace_revision > revision:
            return UNKNOWN
        if fact.freshness == CONTRADICTED or fact.verification == CONTRADICTED:
            return UNKNOWN
        if fact.freshness in {STALE, UNKNOWN}:
            return fact.freshness
        if self._dependencies_changed(fact.path, fact.dependencies, fact.workspace_revision):
            return STALE
        if not fact.dependencies_complete and self._has_edit_after(fact.workspace_revision):
            return UNKNOWN
        return FRESH

    def _replace_status(self, fact: WorldFact, status: str, invalidated_by: Mapping[str, object] | None = None) -> None:
        updated = replace(
            fact,
            freshness=status,
            invalidated_by=dict(invalidated_by) if invalidated_by else fact.invalidated_by,
        )
        self._facts[updated.fact_id] = updated

    def _merge_provenance(
        self,
        old: tuple[FactProvenance, ...],
        new: tuple[FactProvenance, ...],
    ) -> tuple[FactProvenance, ...]:
        combined: list[FactProvenance] = []
        seen: set[tuple[str, str, int, str]] = set()
        for item in (*old, *new):
            key = (item.evidence_ref, item.path, item.workspace_revision, item.content_digest)
            if key not in seen:
                seen.add(key)
                combined.append(item)
        if len(combined) <= 4:
            return tuple(combined)
        return tuple((*combined[:2], *combined[-2:]))

    def _ingest_fact(self, fact: WorldFact) -> WorldFact:
        logical = self._logical.setdefault(fact.logical_key, {})
        same_id = self._facts.get(fact.fact_id)
        if same_id is not None:
            updated = replace(
                same_id,
                observation_count=same_id.observation_count + 1,
                provenance=self._merge_provenance(same_id.provenance, fact.provenance),
                freshness=fact.freshness,
                invalidated_by=fact.invalidated_by,
            )
            self._facts[updated.fact_id] = updated
            self._trace_event(
                "world_knowledge_deduped",
                f"fact={updated.fact_id} count={updated.observation_count}",
            )
            return updated
        if fact.verification == DETERMINISTIC and logical:
            conflict_ref = _digest({"logical_key": logical, "new": fact.fact_id})
            for old_id in tuple(logical.values()):
                old = self._facts[old_id]
                self._replace_status(
                    replace(old, verification=CONTRADICTED, freshness=UNKNOWN),
                    UNKNOWN,
                    {"reason": "same_revision_conflict", "conflict_ref": conflict_ref},
                )
            fact = replace(
                fact,
                verification=CONTRADICTED,
                freshness=UNKNOWN,
                invalidated_by={"reason": "same_revision_conflict", "conflict_ref": conflict_ref},
            )
        self._facts[fact.fact_id] = fact
        for dependency in fact.dependencies:
            self._dependency_fact_ids.setdefault(dependency, set()).add(fact.fact_id)
        if not fact.dependencies_complete:
            self._incomplete_dependency_fact_ids.add(fact.fact_id)
        logical[fact.value_digest] = fact.fact_id
        return fact

    def _settled_fact_preview(self, fact: WorldFact) -> WorldFact:
        """Predict the deterministic ingest result before publishing its event."""
        existing = self._facts.get(fact.fact_id)
        if existing is not None:
            return replace(
                existing,
                observation_count=existing.observation_count + 1,
                provenance=self._merge_provenance(existing.provenance, fact.provenance),
                freshness=fact.freshness,
                invalidated_by=fact.invalidated_by,
            )
        logical = self._logical.get(fact.logical_key, {})
        if fact.verification == DETERMINISTIC and logical:
            conflict_ref = _digest({"logical_key": logical, "new": fact.fact_id})
            return replace(
                fact,
                verification=CONTRADICTED,
                freshness=UNKNOWN,
                invalidated_by={"reason": "same_revision_conflict", "conflict_ref": conflict_ref},
            )
        return fact

    def _emit_observation(self, fact: WorldFact, ordinal: int, *, fact_set_id: str | None = None,
                          settled_fact: WorldFact | None = None,
                          previous_freshness: str | None = None,
                          source_fact_set: bool = False) -> None:
        from kernel_observation_delta import ObservationDelta
        observed = settled_fact or fact
        existing = self._facts.get(observed.fact_id)
        before_count = (
            max(0, settled_fact.observation_count - 1)
            if settled_fact is not None else existing.observation_count if existing is not None else 0
        )
        merged_provenance = self._merge_provenance(
            existing.provenance if existing is not None else (), observed.provenance)
        if fact_set_id is None:
            fact_set_id = ObservationDelta.fact_set_identity(
                self.scope, path=observed.path, workspace_revision=observed.workspace_revision,
                content_digest=observed.content_digest, extractor=observed.extractor)
        delta = ObservationDelta(
            fact_set_id=fact_set_id, fact_id=observed.fact_id,
            new_evidence_ref=observed.evidence_ref, ordinal=ordinal,
            observation_count_before=before_count,
            observation_count_after=before_count + 1,
            provenance_refs=tuple(item.evidence_ref for item in merged_provenance),
            freshness_before=(previous_freshness if previous_freshness is not None
                              else existing.freshness if existing is not None else None),
            
            
            freshness_after=observed.freshness,
        )
        self._observation_sequence += 1
        sequence = self._observation_sequence
        event = {
            "schema": EVENT_SCHEMA,
            "event": "world_knowledge_observed",
            "kind": "observation_fact",
            "scope": dict(self.scope),
            "observation_ordinal": sequence,
            "fact": observed.to_dict(),
            "observation_delta": delta.to_dict(),
            "declaration": {
                "predicate": observed.predicate,
                "workspace_revision": observed.workspace_revision,
                "provenance_required": True,
            },
        }
        if source_fact_set:
            event["source_fact_set"] = True
        event["event_id"] = _digest(
            {
                "scope": self.scope,
                "fact_id": observed.fact_id,
                "evidence_ref": observed.evidence_ref,
                "ordinal": sequence,
            }
        )
        self._emit(event)
        if source_fact_set:
            if fact_set_id is None:
                raise WorldKnowledgeError("source observation has no fact-set identity")
            self._record_source_fact_set_member(
                fact_set_id, ordinal, self._source_fact_set_member(fact)
            )

    @staticmethod
    def _source_fact_set_member(fact: WorldFact) -> _SourceFactSetMember:
        return _SourceFactSetMember(
            fact_id=fact.fact_id,
            path=fact.path,
            workspace_revision=fact.workspace_revision,
            content_digest=fact.content_digest,
            spans=fact.spans,
            extractor=fact.extractor,
        )

    def _record_source_fact_set_member(
        self,
        fact_set_id: str,
        ordinal: int,
        member: _SourceFactSetMember,
    ) -> None:
        if not fact_set_id or type(ordinal) is not int or ordinal < 1:
            raise WorldKnowledgeError("source fact-set member identity is invalid")
        members = self._source_fact_set_members.setdefault(fact_set_id, {})
        previous = members.get(ordinal)
        if previous is not None and previous != member:
            raise WorldKnowledgeError("source fact-set member changed under one identity")
        members[ordinal] = member
        if ordinal == 1:
            self._touch_source_fact_set(fact_set_id)

    def _touch_source_fact_set(self, fact_set_id: str) -> None:
        self._source_fact_set_lru[fact_set_id] = None
        self._source_fact_set_lru.move_to_end(fact_set_id)
        while len(self._source_fact_set_lru) > _SOURCE_CACHE_LIMIT:
            expired, _ = self._source_fact_set_lru.popitem(last=False)
            self._source_fact_set_members.pop(expired, None)
            self._durable_source_fact_sets.pop(expired, None)

    def _forget_source_fact_set(self, fact_set_id: str) -> None:
        self._source_fact_set_lru.pop(fact_set_id, None)
        self._source_fact_set_members.pop(fact_set_id, None)
        self._durable_source_fact_sets.pop(fact_set_id, None)

    def _record_replayed_source_fact_set_member(
        self,
        event: Mapping[str, object],
        fact: WorldFact,
    ) -> None:
        delta = event.get("observation_delta")
        if not isinstance(delta, Mapping):
            raise WorldKnowledgeError("source observation has no observation delta")
        fact_set_id = str(delta.get("fact_set_id") or "")
        ordinal = delta.get("ordinal")
        evidence_ref = str(delta.get("new_evidence_ref") or "")
        if (
            not fact_set_id
            or type(ordinal) is not int
            or str(delta.get("fact_id") or "") != fact.fact_id
            or not evidence_ref
        ):
            raise WorldKnowledgeError("source observation delta identity is invalid")
        source = next(
            (item for item in reversed(fact.provenance) if item.evidence_ref == evidence_ref),
            None,
        )
        if source is None:
            raise WorldKnowledgeError("source observation lacks its new provenance binding")
        self._record_source_fact_set_member(
            fact_set_id,
            ordinal,
            _SourceFactSetMember(
                fact_id=fact.fact_id,
                path=source.path,
                workspace_revision=source.workspace_revision,
                content_digest=source.content_digest,
                spans=source.spans,
                extractor=source.extractor,
            ),
        )

    def _complete_source_fact_set(
        self,
        fact_set_id: str,
        fact_count: int,
    ) -> tuple[_SourceFactSetMember, ...] | None:
        if type(fact_count) is not int or fact_count < 1:
            return None
        members = self._source_fact_set_members.get(fact_set_id)
        if members is None or len(members) != fact_count:
            return None
        try:
            return tuple(members[index] for index in range(1, fact_count + 1))
        except KeyError:
            return None

    def _can_reobserve_source_compactly(self, fact_set_id: str, fact_count: int) -> bool:
        return (
            self._durable_source_fact_sets.get(fact_set_id) == fact_count
            and self._complete_source_fact_set(fact_set_id, fact_count) is not None
        )

    def _mark_source_fact_set_durable(self, fact_set_id: str, fact_count: int) -> None:
        if self._complete_source_fact_set(fact_set_id, fact_count) is None:
            raise WorldKnowledgeError("cannot mark an incomplete source fact set durable")
        previous = self._durable_source_fact_sets.get(fact_set_id)
        if previous is not None and previous != fact_count:
            raise WorldKnowledgeError("source fact-set count changed under one identity")
        self._durable_source_fact_sets[fact_set_id] = fact_count

    def _make_source_reobserved_event(
        self,
        *,
        fact_set_id: str,
        path: str,
        revision: int,
        content_digest: str,
        extractor_version: str,
        fact_count: int,
        evidence_ref: str,
    ) -> dict[str, object]:
        self._observation_sequence += 1
        event = {
            "schema": EVENT_SCHEMA,
            "event": "world_knowledge_reobserved",
            "kind": "source_fact_set_reobserved",
            "scope": dict(self.scope),
            "observation_ordinal": self._observation_sequence,
            "fact_set_id": fact_set_id,
            "path": path,
            "workspace_revision": revision,
            "content_digest": content_digest,
            "extractor_version": extractor_version,
            "fact_count": fact_count,
            "evidence_ref": evidence_ref,
            "declaration": {
                "fact_set_complete": True,
                "provenance_required": True,
            },
        }
        event["event_id"] = _digest(
            {
                "scope": self.scope,
                "event": event["event"],
                "fact_set_id": fact_set_id,
                "evidence_ref": evidence_ref,
                "ordinal": self._observation_sequence,
            }
        )
        return event

    def _apply_source_reobserved(self, event: Mapping[str, object]) -> tuple[WorldFact, ...]:
        from kernel_observation_delta import ObservationDelta

        if event.get("schema") != EVENT_SCHEMA or event.get("event") != "world_knowledge_reobserved":
            raise WorldKnowledgeError("invalid compact source reobservation event")
        event_scope = event.get("scope")
        declaration = event.get("declaration")
        if (
            not isinstance(event_scope, Mapping)
            or _scope_key(_scope(event_scope)) != _scope_key(self.scope)
            or not isinstance(declaration, Mapping)
            or declaration.get("fact_set_complete") is not True
            or declaration.get("provenance_required") is not True
        ):
            raise WorldKnowledgeError("compact source reobservation declaration is invalid")
        path = normalize_path(str(event.get("path") or ""))
        revision = event.get("workspace_revision")
        content_digest = str(event.get("content_digest") or "").strip()
        extractor_version = str(event.get("extractor_version") or "").strip()
        fact_count = event.get("fact_count")
        evidence_ref = str(event.get("evidence_ref") or "").strip()
        fact_set_id = str(event.get("fact_set_id") or "")
        ordinal = event.get("observation_ordinal")
        if (
            type(revision) is not int
            or revision < 0
            or not content_digest
            or not extractor_version
            or type(fact_count) is not int
            or fact_count < 1
            or not evidence_ref
            or type(ordinal) is not int
            or ordinal < 1
        ):
            raise WorldKnowledgeError("compact source reobservation metadata is invalid")
        expected_fact_set_id = ObservationDelta.fact_set_identity(
            self.scope,
            path=path,
            workspace_revision=revision,
            content_digest=content_digest,
            extractor=extractor_version,
        )
        if fact_set_id != expected_fact_set_id:
            raise WorldKnowledgeError("compact source reobservation identity does not match its binding")
        members = self._complete_source_fact_set(fact_set_id, fact_count)
        if members is None:
            raise WorldKnowledgeError("compact source reobservation has no complete durable fact set")
        extractor_parts = extractor_version.split(",")
        extractor_versions = set(extractor_parts)
        if (
            not extractor_versions
            or any(not version for version in extractor_versions)
            or ",".join(sorted(extractor_versions)) != extractor_version
        ):
            raise WorldKnowledgeError("compact source reobservation extractor identity is invalid")
        for member in members:
            if (
                member.path != path
                or member.workspace_revision != revision
                or member.content_digest != content_digest
                or member.extractor not in extractor_versions
            ):
                raise WorldKnowledgeError("compact source reobservation crosses fact-set bindings")
            existing = self._facts.get(member.fact_id)
            if (
                existing is None
                or existing.path != path
                or existing.workspace_revision != revision
                or _scope_key(existing.scope) != _scope_key(self.scope)
            ):
                raise WorldKnowledgeError("compact source reobservation references an absent fact")

        self.current_revision = max(self.current_revision, revision)
        self._path_revisions[(path, revision)] = content_digest
        result: list[WorldFact] = []
        for member in members:
            existing = self._facts[member.fact_id]
            freshness = self._freshness_for_new_fact(
                existing.path,
                existing.workspace_revision,
                existing.dependencies,
                existing.dependencies_complete,
            )
            invalidated_by = (
                self._invalidation_for(
                    existing.path,
                    existing.dependencies,
                    existing.dependencies_complete,
                    existing.workspace_revision,
                )
                if freshness != FRESH
                else None
            )
            provenance = FactProvenance(
                evidence_ref=evidence_ref,
                path=member.path,
                workspace_revision=member.workspace_revision,
                content_digest=member.content_digest,
                spans=member.spans,
                extractor=member.extractor,
            )
            updated = replace(
                existing,
                observation_count=existing.observation_count + 1,
                provenance=self._merge_provenance(existing.provenance, (provenance,)),
                freshness=freshness,
                invalidated_by=invalidated_by,
            )
            self._facts[updated.fact_id] = updated
            result.append(updated)
        self._durable_source_fact_sets[fact_set_id] = fact_count
        self._touch_source_fact_set(fact_set_id)
        self._observation_sequence = max(self._observation_sequence, ordinal)
        self._trace_event(
            "world_knowledge_reobserved",
            f"path={path} revision={revision} facts={fact_count}",
            fact_set_id=fact_set_id,
            path=path,
            workspace_revision=revision,
            content_digest=content_digest,
            evidence_ref=evidence_ref,
            fact_count=fact_count,
        )
        return tuple(sorted(result, key=self._fact_sort_key))

    def observe_source(
        self,
        *,
        path: str,
        content: str,
        revision: int,
        evidence_ref: str,
    ) -> tuple[WorldFact, ...]:
        """Extract and record one complete deterministic source observation."""

        normalized_path = normalize_path(path)
        if not isinstance(content, str):
            raise TypeError("content must be text")
        if not str(evidence_ref).strip():
            raise ValueError("evidence_ref is required")
        revision = int(revision)
        if revision < 0:
            raise ValueError("revision must be non-negative")
        content_digest = source_digest(content)
        extractor_version = source_extractor_version(normalized_path)
        cache_key = (normalized_path, revision, content_digest, extractor_version)
        cached = self._source_cache.get(cache_key)
        cache_hit = cached is not None
        from kernel_observation_delta import ObservationDelta

        with self.cost_meter.measure(
            "extraction", count=1, cache_hit=cache_hit
        ):
            if cached is None:
                drafts = extract_source_facts(
                    normalized_path, content, content_digest=content_digest
                )
                extracted_version = ",".join(
                    sorted({draft.extractor for draft in drafts})
                ) or "none"
                if extracted_version != extractor_version:
                    raise WorldKnowledgeError(
                        "source extractor version identity does not match extracted facts"
                    )
                fact_set_id = ObservationDelta.fact_set_identity(
                    self.scope,
                    path=normalized_path,
                    workspace_revision=revision,
                    content_digest=content_digest,
                    extractor=extractor_version,
                )
                cached = _SourceCacheEntry(
                    content_digest=content_digest,
                    extractor_version=extractor_version,
                    fact_set_id=fact_set_id,
                    drafts=tuple(drafts),
                )
                self._source_cache[cache_key] = cached
                while len(self._source_cache) > _SOURCE_CACHE_LIMIT:
                    _, expired = self._source_cache.popitem(last=False)
                    self._forget_source_fact_set(expired.fact_set_id)
            else:
                self._source_cache.move_to_end(cache_key)
                if (
                    cached.content_digest != content_digest
                    or cached.extractor_version != extractor_version
                ):
                    raise WorldKnowledgeError("source cache identity does not match its key")
                drafts = cached.drafts
        if cached is None:
            raise WorldKnowledgeError("source extraction cache entry was not created")
        fact_set_id = cached.fact_set_id
        self.current_revision = max(self.current_revision, revision)
        self._path_revisions[(normalized_path, revision)] = content_digest
        result: list[WorldFact] = []
        compact_reobservation = cache_hit and self._can_reobserve_source_compactly(
            fact_set_id, len(drafts)
        )
        with self.cost_meter.measure(
            "binding_indexing", count=len(drafts), items=len(drafts)
        ):
            if compact_reobservation:
                event = self._make_source_reobserved_event(
                    fact_set_id=fact_set_id,
                    path=normalized_path,
                    revision=revision,
                    content_digest=content_digest,
                    extractor_version=extractor_version,
                    fact_count=len(drafts),
                    evidence_ref=evidence_ref,
                )
                with self.cost_meter.measure("journaling", count=1):
                    self._emit(event)
                result.extend(self._apply_source_reobserved(event))
            else:
                for ordinal, draft in enumerate(drafts, start=1):
                    fact = self._source_fact(draft, revision, evidence_ref, content_digest)
                    previous = self._facts.get(fact.fact_id)
                    settled = self._settled_fact_preview(fact)
                    with self.cost_meter.measure("journaling", count=1):
                        self._emit_observation(
                            fact,
                            ordinal,
                            fact_set_id=fact_set_id,
                            settled_fact=settled,
                            previous_freshness=previous.freshness if previous else None,
                            source_fact_set=True,
                        )
                    result.append(self._ingest_fact(fact))
                self._mark_source_fact_set_durable(fact_set_id, len(drafts))
        parser_status = (
            UNKNOWN if any(fact.predicate in {"parse_error", "parser_unsupported"} for fact in drafts)
            else "OBSERVED"
        )
        line_count = bounded_line_count(content)
        self._trace_event(
            "world_knowledge_observed",
            f"path={normalized_path} revision={revision} facts={len(result)}",
            path=normalized_path,
            workspace_revision=revision,
            content_digest=content_digest,
            cache_hit=cache_hit,
            compact_reobservation=compact_reobservation,
            parser_status=parser_status,
            redundant_work={
                "cache_hit": cache_hit,
                "digest_recomputed": True,
                "extraction_recomputed": not cache_hit,
                "fact_events_emitted": 0 if compact_reobservation else len(drafts),
            },
            whole_file_observation={
                "bounded": True,
                "line_count": min(10_000_000, line_count),
                "line_count_capped": line_count > 10_000_000,
                "content_characters": min(10_000_000, len(content)),
                "content_characters_capped": len(content) > 10_000_000,
                "attribution": "instrumented_observation_only",
            },
            cost_metrics=self.cost_meter.snapshot(),
        )
        return tuple(sorted(result, key=self._fact_sort_key))

    def observe_fact(
        self,
        *,
        path: str,
        predicate: str,
        subject_ref: str,
        value: Mapping[str, object],
        revision: int,
        evidence_ref: str,
        spans: Sequence[tuple[int, int]] = (),
        dependencies: Sequence[str] = (),
        dependencies_complete: bool = True,
        content_digest: str = "",
    ) -> WorldFact:
        """Ingest one adapter-provided deterministic fact with the same guardrails."""

        if predicate not in _DETERMINISTIC_PREDICATES:
            raise WorldKnowledgeError(f"unsupported deterministic predicate: {predicate}")
        normalized_path = normalize_path(path)
        draft = ExtractedFact(
            path=normalized_path,
            predicate=predicate,
            subject_ref=subject_ref,
            value=value,
            spans=tuple(spans),
            dependencies=tuple(dependencies),
            dependencies_complete=dependencies_complete,
            extractor="adapter/1",
        )
        self.current_revision = max(self.current_revision, int(revision))
        fact = self._source_fact(draft, int(revision), evidence_ref, content_digest or draft.value_digest)
        previous = self._facts.get(fact.fact_id)
        settled = self._settled_fact_preview(fact)
        self._emit_observation(fact, 1, settled_fact=settled,
                               previous_freshness=previous.freshness if previous else None)
        return self._ingest_fact(fact)

    def observe_test_selector(
        self,
        *,
        selector: str,
        result_class: str,
        revision: int,
        evidence_ref: str,
        path: str | None = None,
    ) -> tuple[WorldFact, ...]:
        """Record an exact test selector/result receipt as deterministic evidence."""

        if not str(evidence_ref).strip():
            raise ValueError("evidence_ref is required")
        revision = int(revision)
        draft = extract_test_selector(selector, result_class, path=path)
        self.current_revision = max(self.current_revision, revision)
        fact = self._source_fact(draft, revision, evidence_ref)
        previous = self._facts.get(fact.fact_id)
        settled = self._settled_fact_preview(fact)
        self._emit_observation(fact, 1, settled_fact=settled,
                               previous_freshness=previous.freshness if previous else None)
        result = self._ingest_fact(fact)
        self._trace_event(
            "world_knowledge_observed",
            f"selector={str(selector)[:120]} revision={revision}",
            path=draft.path,
            workspace_revision=revision,
        )
        return (result,)

    def observe_manifest(
        self,
        *,
        files: Mapping[str, str],
        revision: int,
        evidence_ref: str,
    ) -> tuple[WorldFact, ...]:
        """Record a deterministic manifest snapshot without parsing files again."""

        if not str(evidence_ref).strip():
            raise ValueError("evidence_ref is required")
        revision = int(revision)
        self.current_revision = max(self.current_revision, revision)
        result: list[WorldFact] = []
        for ordinal, draft in enumerate(extract_manifest_facts(files), start=1):
            fact = self._source_fact(draft, revision, evidence_ref)
            previous = self._facts.get(fact.fact_id)
            settled = self._settled_fact_preview(fact)
            self._emit_observation(fact, ordinal, settled_fact=settled,
                                   previous_freshness=previous.freshness if previous else None)
            result.append(self._ingest_fact(fact))
        return tuple(sorted(result, key=self._fact_sort_key))

    def invalidate_edit(
        self,
        *,
        path: str,
        before_revision: int,
        after_revision: int,
        before_digest: str,
        after_digest: str,
        evidence_ref: str,
    ) -> None:
        """Stale only direct/dependent facts; unknown dependencies fail closed."""

        normalized_path = normalize_path(path)
        before_revision = int(before_revision)
        after_revision = int(after_revision)
        if before_revision < 0 or after_revision < before_revision:
            raise ValueError("edit revisions must be non-negative and monotone")
        if not str(evidence_ref).strip():
            raise ValueError("evidence_ref is required")
        payload = {
            "schema": EVENT_SCHEMA,
            "event": "world_knowledge_invalidated",
            "kind": "invalidation",
            "scope": dict(self.scope),
            "path": normalized_path,
            "before_revision": before_revision,
            "after_revision": after_revision,
            "before_digest": str(before_digest),
            "after_digest": str(after_digest),
            "evidence_ref": str(evidence_ref).strip(),
            "declaration": {
                "changed_path": normalized_path,
                "old_revision": before_revision,
                "new_revision": after_revision,
                "freshness_gate": "dependent_facts_must_not_remain_fresh",
            },
        }
        payload["event_id"] = _digest(payload)
        self._emit(payload)
        self._apply_invalidation(payload)

    def _apply_invalidation(self, payload: Mapping[str, object]) -> None:
        path = normalize_path(str(payload.get("path") or ""))
        before_revision = int(payload.get("before_revision", 0))
        after_revision = int(payload.get("after_revision", before_revision))
        self.current_revision = max(self.current_revision, after_revision)
        record = {
            "event_id": str(payload.get("event_id") or _digest(payload)),
            "path": path,
            "before_revision": before_revision,
            "after_revision": after_revision,
            "before_digest": str(payload.get("before_digest") or ""),
            "after_digest": str(payload.get("after_digest") or ""),
            "evidence_ref": str(payload.get("evidence_ref") or ""),
        }
        if record["event_id"] not in self._invalidation_event_ids:
            self._invalidation_records.append(record)
            self._invalidation_event_ids.add(str(record["event_id"]))
        stale_fact_ids: set[str] = set()
        for token in self._dependency_match_tokens(path):
            stale_fact_ids.update(self._dependency_fact_ids.get(token, ()))
        candidate_fact_ids = stale_fact_ids | self._incomplete_dependency_fact_ids
        for fact_id in candidate_fact_ids:
            fact = self._facts.get(fact_id)
            if fact is None or fact.workspace_revision > before_revision:
                continue
            if fact_id in stale_fact_ids:
                self._replace_status(fact, STALE, record)
            else:
                self._replace_status(fact, UNKNOWN, record)
        self._trace_event(
            "world_knowledge_invalidated",
            f"path={path} before={before_revision} after={after_revision}",
            path=path,
            workspace_revision=after_revision,
        )

    def _iter_events(self, event_stream: object) -> Iterable[Mapping[str, object]]:
        if isinstance(event_stream, Mapping):
            return (event_stream,)
        if hasattr(event_stream, "iter_events"):
            source = event_stream.iter_events()  # type: ignore[union-attr]
        elif hasattr(event_stream, "read_all"):
            source = event_stream.read_all()  # type: ignore[union-attr]
        else:
            source = event_stream
        if isinstance(source, Iterable):
            result: list[Mapping[str, object]] = []
            for item in source:
                if isinstance(item, Mapping):
                    result.append(item)
                elif hasattr(item, "to_dict"):
                    value = item.to_dict()  # type: ignore[union-attr]
                    if isinstance(value, Mapping):
                        result.append(value)
            return tuple(result)
        raise TypeError("event_stream must be iterable, a mapping, or expose read_all/iter_events")

    def replay(self, event_stream: object) -> None:
        """Fold observation/invalidation events; duplicate event IDs are idempotent."""

        for raw in self._iter_events(event_stream):
            event = dict(raw)
            if event.get('kind') == 'world_knowledge_event':
                inner = event.get('event')
                if not isinstance(inner, Mapping):
                    raise WorldKnowledgeError('invalid wrapped world knowledge event')
                if inner.get('schema') == 'retained-source/1' and (
                        self.source_archive is None or event.get('run_id') != self.source_archive.scope):
                    continue
                event = dict(inner)
            if event.get('schema') == 'retained-source/1':
                if self.source_archive is not None:
                    self.source_archive.replay([event])
                continue
            event_scope = event.get("scope")
            if isinstance(event_scope, Mapping) and _scope_key(_scope(event_scope)) != _scope_key(self.scope):
                continue
            event_id = str(event.get("event_id") or _digest(event))
            if event_id in self._seen_event_ids:
                continue
            name = str(event.get("event") or event.get("event_name") or "")
            if name == "world_knowledge_invalidated":
                self._seen_event_ids.add(event_id)
                self._apply_invalidation({**event, "event_id": event_id})
                continue
            if name == "world_knowledge_reobserved":
                self._apply_source_reobserved(event)
                self._seen_event_ids.add(event_id)
                continue
            if name != "world_knowledge_observed":
                continue
            raw_fact = event.get("fact")
            if not isinstance(raw_fact, Mapping):
                raise WorldKnowledgeError("observation_fact event has no fact payload")
            fact = WorldFact.from_dict(raw_fact)
            if _scope_key(fact.scope) != _scope_key(self.scope):
                continue
            self._seen_event_ids.add(event_id)
            self._observation_sequence = max(
                self._observation_sequence,
                int(event.get("observation_ordinal", 0) or 0),
            )
            self.current_revision = max(self.current_revision, fact.workspace_revision)
            status = self._refresh_status(fact, self.current_revision)
            if status != FRESH:
                fact = replace(
                    fact,
                    freshness=status,
                    invalidated_by=fact.invalidated_by or self._invalidation_for(
                        fact.path,
                        fact.dependencies,
                        fact.dependencies_complete,
                        fact.workspace_revision,
                    ),
                )
            self._ingest_fact(fact)
            if event.get("source_fact_set") is True:
                self._record_replayed_source_fact_set_member(event, fact)
            self._seen_event_ids.add(event_id)

    def ingest_llm_proposal(
        self,
        *,
        path: str,
        predicate: str,
        subject_ref: str,
        value: Mapping[str, object],
        revision: int,
        evidence_ref: str,
        spans: Sequence[tuple[int, int]] = (),
        strict: bool = False,
    ) -> WorldFact | None:
        """Validate a prose-only proposal, but keep it outside fresh knowledge."""

        normalized_path = normalize_path(path)
        reason: str | None = None
        if predicate not in LLM_PROSE_PREDICATES:
            reason = f"predicate {predicate!r} is not prose-only"
        elif not subject_ref.startswith(f"env://{normalized_path}#"):
            reason = "subject_ref is outside the supplied source path"
        else:
            subject = subject_ref.split("#", 1)[1]
            structural = [
                fact
                for fact in self._facts.values()
                if fact.path == normalized_path
                and fact.workspace_revision == int(revision)
                and (
                    fact.subject_ref == subject_ref
                    or subject == "file"
                    or fact.value.get("qualname") == subject
                )
            ]
            if subject != "file" and not structural:
                reason = "subject_ref was not proven by deterministic extraction"
            elif spans and not any(set(_spans(spans)).issubset(set(fact.spans)) for fact in structural):
                reason = "spans were not proven by deterministic extraction"
        if reason:
            if strict:
                raise LLMProposalRejected(reason)
            self._trace_event("world_knowledge_snitch", reason)
            return None
        proposal = WorldFact(
            scope=self.scope,
            path=normalized_path,
            workspace_revision=int(revision),
            subject_ref=subject_ref,
            predicate=predicate,
            value=value,
            content_digest=source_digest(_canonical(value)),
            spans=tuple(spans),
            evidence_ref=evidence_ref,
            dependencies=(normalized_path,),
            verification=LLM_PROPOSAL,
            confidence_tier=L0_UNVERIFIED,
            freshness=UNKNOWN,
            extractor="llm_proposal/disabled",
        )
        self._proposals.append(proposal)
        return proposal

    def _query_candidates(
        self,
        query: str,
        path: str | None,
        symbol: str | None,
    ) -> tuple[WorldFact, ...]:
        name = query.strip().lower().replace("-", "_")
        if name in {"tests_cover", "tests_covered", "what_tests_cover", "tests"}:
            if not symbol:
                return ()
            references = [
                fact
                for fact in self._facts.values()
                if fact.predicate == "static_references_symbol"
                and str(fact.value.get("symbol") or "") == symbol
                and (path is None or self._path_matches_filter(fact, path))
            ]
            selected: dict[str, WorldFact] = {}
            for reference in references:
                enclosing = reference.value.get("enclosing_symbol")
                if not enclosing:
                    continue
                tests = [
                    fact
                    for fact in self._facts.values()
                    if fact.predicate == "test_defined"
                    and fact.path == reference.path
                    and fact.value.get("qualname") == enclosing
                ]
                for test in tests:
                    selected[test.fact_id] = test
                    selected[reference.fact_id] = reference
            return tuple(sorted(selected.values(), key=self._fact_sort_key))
        result: list[WorldFact] = []
        for fact in self._facts.values():
            if path is not None and not self._path_matches_filter(fact, path):
                continue
            if name in {"signature", "signatures"} and fact.predicate != "symbol_defined":
                continue
            if name in {"imports", "import"} and fact.predicate != "import_edge":
                continue
            if name in {"guards", "guard"} and fact.predicate != "guard_predicate":
                continue
            if name in {"oracles", "oracle", "selectors"} and fact.predicate != "oracle_selector":
                continue
            if name in {"file_facts", "facts", "all"} and fact.predicate not in _DETERMINISTIC_PREDICATES:
                continue
            if symbol and not self._symbol_matches(fact, symbol):
                continue
            result.append(fact)
        return tuple(sorted(result, key=self._fact_sort_key))

    @staticmethod
    def _symbol_matches(fact: WorldFact, symbol: str) -> bool:
        candidates = {
            str(fact.value.get("symbol") or ""),
            str(fact.value.get("name") or ""),
            str(fact.value.get("qualname") or ""),
            fact.subject_ref.rsplit("#", 1)[-1],
        }
        return symbol in candidates or any(item.endswith(f".{symbol}") for item in candidates)

    def _path_matches_filter(self, fact: WorldFact, path: str) -> bool:
        normalized = normalize_path(path)
        return fact.path == normalized or any(self._path_matches(dep, normalized) for dep in fact.dependencies)

    def query(
        self,
        *,
        query: str,
        path: str | None = None,
        symbol: str | None = None,
        revision: int,
        include_stale: bool = False,
    ) -> KnowledgeAnswer:
        """Answer only from deterministic fresh facts unless history is explicit."""

        revision = int(revision)
        normalized_query = query.strip().lower().replace("-", "_")
        if normalized_query not in _SUPPORTED_QUERIES:
            answer = KnowledgeAnswer(
                query=query,
                status="UNKNOWN",
                source_revision=revision,
                recommendation=f"unsupported knowledge query: {query}",
            )
            self._trace_event(
                "world_knowledge_query_unknown",
                f"unsupported query={query} revision={revision}",
                declaration={"query": query, "revision": revision},
                falsification={"reason": "unsupported_query"},
            )
            return answer
        candidates = self._query_candidates(query, path, symbol)
        fresh: list[WorldFact] = []
        nonfresh: list[WorldFact] = []
        changed_paths: set[str] = set()
        for fact in candidates:
            status = self._refresh_status(fact, revision)
            if status == FRESH:
                fresh.append(fact)
            else:
                invalidated_by = fact.invalidated_by or self._invalidation_for(
                    fact.path,
                    fact.dependencies,
                    fact.dependencies_complete,
                    fact.workspace_revision,
                )
                stale_fact = replace(
                    fact,
                    freshness=status,
                    invalidated_by=invalidated_by,
                )
                nonfresh.append(stale_fact)
                if invalidated_by and invalidated_by.get("path"):
                    changed_paths.add(str(invalidated_by["path"]))
        fresh_tuple = tuple(sorted(fresh, key=self._fact_sort_key))
        stale_tuple = tuple(sorted(nonfresh, key=self._fact_sort_key))
        # Coverage is a relationship between test and product facts.  One stale
        # edge invalidates the whole answer so a partial cache cannot look fresh.
        if stale_tuple and query.strip().lower().replace("-", "_") in {
            "tests_cover",
            "tests_covered",
            "what_tests_cover",
            "tests",
        }:
            fresh_tuple = ()
        status = "HIT" if fresh_tuple else "UNKNOWN" if stale_tuple else "MISS"
        selected = fresh_tuple + (stale_tuple if include_stale else ())
        answer = KnowledgeAnswer(
            query=query,
            status=status,
            source_revision=revision,
            facts=selected,
            stale_facts=stale_tuple,
            stale=bool(stale_tuple),
            changed_paths=tuple(sorted(changed_paths)),
            recommendation=(
                f"re-observe the relevant source at revision {revision}"
                if stale_tuple or not fresh_tuple
                else None
            ),
        )
        self._trace_event(
            {
                "HIT": "world_knowledge_query_hit",
                "MISS": "world_knowledge_query_miss",
                "UNKNOWN": "world_knowledge_query_unknown",
            }[status],
            f"query={query} revision={revision} fresh={len(fresh_tuple)} stale={len(stale_tuple)}",
            declaration={"query": query, "revision": revision},
            falsification={
                "fresh_facts": len(fresh_tuple),
                "stale_facts": len(stale_tuple),
                "provenance_complete": all(bool(fact.provenance) for fact in selected),
            },
        )
        return answer

    def knowledge(self, **kwargs: object) -> KnowledgeAnswer:
        """Read-only tool-shaped alias for callers that name the operation knowledge."""

        return self.query(**kwargs)  # type: ignore[arg-type]

    def _rehydration_facts(
        self,
        unit_scope: Mapping[str, object],
        revision: int,
    ) -> tuple[WorldFact, ...]:
        target_path_raw = str(unit_scope.get("target_path") or "").strip()
        target_path = normalize_path(target_path_raw) if target_path_raw else None
        target_symbol = str(unit_scope.get("target_symbol") or "").strip() or None
        selector = str(unit_scope.get("last_selector") or "").strip()
        failure = str(unit_scope.get("active_failure") or "").strip()
        selector_path = selector.split("::", 1)[0] if selector else ""
        all_facts = tuple(self._facts.values())
        linked_tests: set[tuple[str, str]] = set()
        if target_symbol:
            for fact in all_facts:
                if (
                    fact.predicate == "static_references_symbol"
                    and str(fact.value.get("symbol") or "") == target_symbol
                    and fact.value.get("enclosing_symbol")
                    and (
                        target_path is None
                        or any(
                            self._path_matches(dep, target_path)
                            for dep in fact.dependencies
                        )
                    )
                ):
                    linked_tests.add((fact.path, str(fact.value["enclosing_symbol"])))
        ranked: list[tuple[tuple[object, ...], WorldFact]] = []
        for fact in all_facts:
            status = self._refresh_status(fact, revision)
            symbol_hit = bool(
                target_symbol
                and self._symbol_matches(fact, target_symbol)
                and (
                    target_path is None
                    or fact.path == target_path
                    or any(
                        self._path_matches(dep, target_path)
                        for dep in fact.dependencies
                    )
                )
            )
            linked_test = fact.predicate == "test_defined" and (
                fact.path,
                str(fact.value.get("qualname") or ""),
            ) in linked_tests
            selector_hit = bool(selector and (
                fact.path == selector_path or str(fact.value.get("selector") or "") == selector
            ))
            failure_hit = bool(failure and (failure in fact.path or failure in fact.subject_ref))
            direct_path = bool(target_path and fact.path == target_path)
            dependency_hit = bool(target_path and any(
                self._path_matches(dep, target_path) for dep in fact.dependencies
            ))
            relevant = symbol_hit or linked_test or selector_hit or failure_hit or direct_path or dependency_hit
            if not relevant:
                continue
            priority = (
                0 if symbol_hit else
                1 if linked_test else
                2 if selector_hit or failure_hit else
                3 if direct_path else
                4 if dependency_hit else 5
            )
            ranked.append(
                (
                    (priority, 0 if status == FRESH else 1, -fact.observation_count,
                     fact.path, fact.predicate, fact.subject_ref, fact.fact_id),
                    replace(fact, freshness=status),
                )
            )
        ranked.sort(key=lambda item: item[0])
        # E394/E398: a rehydration brief is a small fresh-evidence digest, not a
        # historical cache dump. Stale/unknown facts remain queryable as history
        # but never enter the model-facing brief.
        fresh_ranked = [item for item in ranked if item[1].freshness == FRESH]
        return tuple(item[1] for item in fresh_ranked[:5])

    def render(
        self,
        *,
        unit_scope: Mapping[str, object],
        revision: int,
        max_tokens: int = 220,
    ) -> str:
        """Render only facts relevant to the current unit, with a hard word cap."""

        facts = self._rehydration_facts(unit_scope, int(revision))
        return _render_fact_digest(
            facts,
            query="rehydration",
            revision=int(revision),
            max_tokens=max_tokens,
        )

    def render_rehydration_digest(
        self,
        *,
        unit_scope: Mapping[str, object],
        revision: int,
        max_tokens: int = 220,
    ) -> str:
        return self.render(unit_scope=unit_scope, revision=revision, max_tokens=max_tokens)


def _status_label(fact: WorldFact) -> str:
    return "fresh" if fact.freshness == FRESH and fact.verification == DETERMINISTIC else fact.freshness.lower()


def _span_label(fact: WorldFact) -> str:
    if not fact.spans:
        return ""
    if len(fact.spans) == 1:
        start, end = fact.spans[0]
        return f"[L{start}-L{end}]"
    return "[" + ",".join(f"L{start}-L{end}" for start, end in fact.spans) + "]"


def _fact_line(fact: WorldFact) -> str:
    value = fact.value
    if fact.predicate == "test_defined":
        body = f"{fact.path}::{value.get('qualname') or value.get('name')}"
    elif fact.predicate == "symbol_defined":
        body = f"{fact.path}: {value.get('signature') or value.get('qualname')}"
    elif fact.predicate == "import_edge":
        alias = value.get("alias")
        body = f"{fact.path} imports {value.get('module')}:{value.get('name')}"
        if alias:
            body += f" as {alias}"
    elif fact.predicate == "guard_predicate":
        body = f"{fact.path} guard `{value.get('predicate')}`"
    elif fact.predicate == "oracle_selector":
        body = f"oracle `{value.get('selector')}` => {value.get('result_class')}"
    elif fact.predicate == "static_references_symbol":
        body = (
            f"{fact.path}::{value.get('enclosing_symbol') or '<module>'} may exercise "
            f"`{value.get('symbol')}`"
        )
    elif fact.predicate == "file_digest":
        body = f"{fact.path} digest={value.get('digest')}"
    elif fact.predicate == "manifest_digest":
        body = f"manifest digest={value.get('digest')}"
    elif fact.predicate == "parse_error":
        body = f"{fact.path} parse error={value.get('message')}"
    else:
        body = f"{fact.path} {fact.predicate}={value}"
    span = _span_label(fact)
    if span:
        body += f" {span}"
    provenance = fact.provenance[0].evidence_ref if fact.provenance else fact.evidence_ref
    invalidation = ""
    if fact.invalidated_by and fact.invalidated_by.get("evidence_ref"):
        invalidation = f" | invalidated_by={fact.invalidated_by['evidence_ref']}"
    return f"- {body} | {fact.confidence_tier} | {_status_label(fact)} | {provenance}{invalidation}"


def _render_lines(lines: Sequence[str], *, footer: str, max_tokens: int) -> str:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    selected: list[str] = []
    used = 0
    for line in lines:
        cost = len(line.split())
        if used + cost > max_tokens:
            break
        selected.append(line)
        used += cost
    remaining = max_tokens - used
    if remaining > 0:
        footer_words = footer.split()
        if len(footer_words) > remaining:
            footer_words = footer_words[:remaining]
        if footer_words:
            selected.append(" ".join(footer_words))
    return "\n".join(selected)


def _render_fact_digest(
    facts: Sequence[WorldFact],
    *,
    query: str,
    revision: int,
    max_tokens: int,
) -> str:
    digest = _digest(
        {
            "query": query,
            "revision": revision,
            "facts": [(fact.fact_id, fact.freshness, fact.verification) for fact in facts],
        }
    )
    header = f"WORLD KNOWLEDGE query={query} revision={revision}"
    fact_lines = [_fact_line(fact) for fact in facts]
    selected: list[str] = []
    used = len(header.split())
    for index, line in enumerate(fact_lines):
        omitted = len(fact_lines) - index - 1
        footer_probe = f"+{omitted} omitted | knowledge_digest={digest}"
        if used + len(line.split()) + len(footer_probe.split()) > max_tokens:
            break
        selected.append(line)
        used += len(line.split())
    lines = [header, *selected]
    footer = f"+{len(fact_lines) - len(selected)} omitted | knowledge_digest={digest}"
    if not facts:
        footer = f"MISS | knowledge_digest={digest}"
    return _render_lines(lines, footer=footer, max_tokens=max_tokens)


def render_knowledge_answer(answer: KnowledgeAnswer, *, max_tokens: int = 220) -> str:
    """Render an on-demand answer; stale history is visibly non-fresh."""

    lines = [
        f"KNOWLEDGE query={answer.query} status={answer.status} "
        f"revision={answer.source_revision} stale={'true' if answer.stale else 'false'}"
    ]
    lines.extend(_fact_line(fact) for fact in answer.facts)
    if not answer.facts and answer.stale_facts:
        lines.extend(_fact_line(fact) for fact in answer.stale_facts[:5])
    if answer.changed_paths:
        lines.append("changed_paths=" + ",".join(answer.changed_paths))
    if answer.recommendation:
        lines.append("recommendation=" + answer.recommendation)
    provenance = ",".join(item.evidence_ref for item in answer.provenance[:4])
    if provenance:
        lines.append("provenance=" + provenance)
    footer = f"knowledge_digest={answer.knowledge_digest}"
    return _render_lines(lines, footer=footer, max_tokens=max_tokens)


def render_rehydration_digest(
    store: WorldKnowledgeStore,
    *,
    unit_scope: Mapping[str, object],
    revision: int,
    max_tokens: int = 220,
) -> str:
    """Standalone renderer for trim/process-death rehydration callers."""

    return store.render(unit_scope=unit_scope, revision=revision, max_tokens=max_tokens)


def _world_knowledge_trigger(evidence: Evidence) -> TriState:
    """Owned deterministic trigger for this organ's census declaration."""

    record = getattr(evidence, "event", {})
    event = str(record.get("event") or record.get("event_name") or "")
    if event in _WORLD_EVENTS:
        return TriState.YES
    if not getattr(evidence, "complete", True) or getattr(evidence, "gaps", ()):
        return TriState.UNKNOWN
    return TriState.NO


_world_knowledge_trigger.__name__ = "world_knowledge_trigger"

# Census declaration lives beside its trigger, matching kernel_liveness.py's organ-owned
# pattern. The current registry loader predates this new organ; the future hot-file wire
# can include this tuple without moving or duplicating the predicate.
CENSUS_SPEC = OrganCensusSpec(
    organ_id="world.knowledge",
    trigger=_world_knowledge_trigger,
    fire_events=frozenset(
        {
            "world_knowledge_observed",
            "world_knowledge_reobserved",
            "world_knowledge_invalidated",
            "world_knowledge_query_hit",
        }
    ),
    suppression_events=frozenset(
        {"world_knowledge_query_miss", "world_knowledge_query_unknown"}
    ),
    dedup_events=frozenset({"world_knowledge_deduped"}),
    challenge_once_events=frozenset({"world_knowledge_snitch"}),
)
CENSUS_SPECS = (CENSUS_SPEC,)


__all__ = [
    "CENSUS_SPEC",
    "CENSUS_SPECS",
    "CONTRADICTED",
    "D0_EXACT",
    "DETERMINISTIC",
    "FactProvenance",
    "FRESH",
    "KnowledgeAnswer",
    "L0_UNVERIFIED",
    "LLM_PROPOSAL",
    "LLMProposalRejected",
    "STALE",
    "UNKNOWN",
    "WorldFact",
    "WorldKnowledgeError",
    "WorldKnowledgeStore",
    "render_knowledge_answer",
    "render_rehydration_digest",
]

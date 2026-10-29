"""Deterministic, post-task liveness census for kernel organs.

The census consumes recorded evidence only.  Organ modules own their trigger
predicates and event vocabularies; this module only invokes predicates, folds
stable events, and emits an append-only census receipt.  It is intentionally
safe to run after a task or against old JSONL/JSON tapes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
try:
    from typing import TypeAlias
except ImportError:  # Python 3.9 compatibility for postponed aliases.
    TypeAlias = object
from typing import Union


JSONScalar: TypeAlias = Union[str, int, float, bool, type(None)]
JSONValue: TypeAlias = Union[JSONScalar, Mapping[str, "JSONValue"], Sequence["JSONValue"]]
Record: TypeAlias = Mapping[str, object]
ReceiptSink: TypeAlias = Callable[[Mapping[str, object]], object]

SCHEMA = "poc.kernel_liveness/1"
_ORGAN_FIELD_NAMES = ("organ_id", "organ", "name", "cylinder")
_EVENT_FIELD_NAMES = ("event", "event_name", "lifecycle_event")
_DETAIL_ORGAN = re.compile(r"(?:^|\s)organ=(?P<organ>[^\s,;]+)", re.IGNORECASE)


class TriState(str, Enum):
    """Three-valued trigger result; UNKNOWN is never treated as NO."""

    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def coerce(cls, value: object) -> "TriState":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.YES if value else cls.NO
        normalized = str(value or "").strip().upper()
        try:
            return cls(normalized)
        except ValueError:
            return cls.UNKNOWN


class OrganVerdict(str, Enum):
    """Per-organ liveness verdict."""

    FIRED = "FIRED"
    SUPPRESSED = "SUPPRESSED"
    DARK = "DARK"
    INCOMPLETE = "INCOMPLETE"
    NOT_TRIGGERED = "NOT_TRIGGERED"


class RegistryInvariantError(ValueError):
    """Raised when the status registry and code declarations drift."""


def _text(value: object) -> str:
    return str(value or "").strip()


def _normal(value: object) -> str:
    return _text(value).lower().replace(" ", "_")


def _clip(value: object, limit: int = 180) -> str:
    return _text(value)[:limit]


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _record_tuple(values: object) -> tuple[Record, ...]:
    if isinstance(values, Mapping):
        return (values,)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    return tuple(item for item in values if isinstance(item, Mapping))


@dataclass(frozen=True)
class Evidence:
    """Run evidence plus one candidate record being tested by a predicate.

    ``record`` and ``source`` are populated by the engine for each candidate;
    callers may also populate them directly for focused predicate tests.
    """

    receipts: tuple[Record, ...] = ()
    traces: tuple[Record, ...] = ()
    organ_outputs: tuple[Record, ...] = ()
    run_id: str = ""
    complete: bool = True
    gaps: tuple[str, ...] = ()
    source: str = "run"
    record: Record = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipts", _record_tuple(self.receipts))
        object.__setattr__(self, "traces", _record_tuple(self.traces))
        object.__setattr__(self, "organ_outputs", _record_tuple(self.organ_outputs))
        object.__setattr__(self, "run_id", _text(self.run_id))
        object.__setattr__(self, "gaps", tuple(_clip(gap, 160) for gap in self.gaps if _text(gap)))
        if not isinstance(self.record, Mapping):
            object.__setattr__(self, "record", {})

    @classmethod
    def from_records(
        cls,
        *,
        receipts: object = (),
        traces: object = (),
        organ_outputs: object = (),
        run_id: str = "",
        complete: bool = True,
        gaps: Iterable[object] = (),
    ) -> "Evidence":
        return cls(
            receipts=_record_tuple(receipts),
            traces=_record_tuple(traces),
            organ_outputs=_record_tuple(organ_outputs),
            run_id=run_id,
            complete=complete,
            gaps=tuple(_text(gap) for gap in gaps if _text(gap)),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Evidence":
        known = {"receipts", "traces", "organ_outputs", "outputs", "run_id", "complete", "gaps"}
        if known.intersection(value):
            return cls.from_records(
                receipts=value.get("receipts", ()),
                traces=value.get("traces", ()),
                organ_outputs=value.get("organ_outputs", value.get("outputs", ())),
                run_id=_text(value.get("run_id")),
                complete=bool(value.get("complete", True)),
                gaps=value.get("gaps", ()) if isinstance(value.get("gaps", ()), Sequence) else (),
            )
        return cls(record=value, source="record")

    @property
    def event(self) -> Record:
        """Alias for the candidate record used by owned trigger predicates."""

        return self.record

    @property
    def organ_out(self) -> tuple[Record, ...]:
        """Compatibility alias for callers that name the output seam directly."""

        return self.organ_outputs


Trigger: TypeAlias = Callable[[Evidence], TriState]


def catalog_unknown_opportunity(unknowns: Iterable[object], *, scope: str = "task") -> TriState:
    """Return the liveness predicate for open host-owned unknowns in scope."""
    requested_scope = _text(scope) or "task"
    for item in unknowns:
        value = item.get("blocking_scope") if isinstance(item, Mapping) else getattr(
            item, "blocking_scope", None)
        if _text(value) == requested_scope:
            return TriState.YES
    return TriState.NO


@dataclass(frozen=True)
class OrganCensusSpec:
    """Owned trigger and event vocabulary for one behavioral organ."""

    organ_id: str
    trigger: Trigger
    fire_events: frozenset[str] = frozenset()
    suppression_events: frozenset[str] = frozenset()
    dedup_events: frozenset[str] = frozenset()
    challenge_once_events: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        organ_id = _text(self.organ_id)
        if not organ_id:
            raise ValueError("organ_id must not be empty")
        if not callable(self.trigger):
            raise TypeError("trigger must be callable")
        object.__setattr__(self, "organ_id", organ_id)
        for name in (
            "fire_events",
            "suppression_events",
            "dedup_events",
            "challenge_once_events",
        ):
            values = getattr(self, name)
            object.__setattr__(self, name, frozenset(_text(item) for item in values if _text(item)))

    def to_dict(self) -> dict[str, object]:
        return {
            "organ_id": self.organ_id,
            "trigger": getattr(self.trigger, "__name__", type(self.trigger).__name__),
            "fire_events": sorted(self.fire_events),
            "suppression_events": sorted(self.suppression_events),
            "dedup_events": sorted(self.dedup_events),
            "challenge_once_events": sorted(self.challenge_once_events),
        }


@dataclass(frozen=True)
class OrganCensusResult:
    """Folded counts and verdict for one organ."""

    organ_id: str
    n: int = 0
    m: int = 0
    s: int = 0
    d: int = 0
    deduped: int = 0
    challenge_once: int = 0
    shadowed: int = 0
    verdict: OrganVerdict = OrganVerdict.NOT_TRIGGERED
    evidence_gaps: tuple[str, ...] = ()
    trigger_unknown: int = 0

    @property
    def N(self) -> int:
        return self.n

    @property
    def M(self) -> int:
        return self.m

    @property
    def S(self) -> int:
        return self.s

    @property
    def D(self) -> int:
        return self.d

    def to_dict(self) -> dict[str, object]:
        return {
            "organ_id": self.organ_id,
            "N": self.n,
            "M": self.m,
            "S": self.s,
            "D": self.d,
            "deduped": self.deduped,
            "challenge_once": self.challenge_once,
            "shadowed": self.shadowed,
            "verdict": self.verdict.value,
            "evidence_gaps": list(self.evidence_gaps),
            "trigger_unknown": self.trigger_unknown,
        }


@dataclass(frozen=True)
class CensusReport:
    """Complete post-task census, including its own receipt."""

    run_id: str
    status: str
    organs: tuple[OrganCensusResult, ...]
    digest: str
    receipt: Mapping[str, object]
    evidence_gaps: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def result_for(self, organ_id: str) -> OrganCensusResult | None:
        return next((item for item in self.organs if item.organ_id == organ_id), None)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "status": self.status,
            "digest": self.digest,
            "evidence_gaps": list(self.evidence_gaps),
            "errors": list(self.errors),
            "organs": [item.to_dict() for item in self.organs],
            "receipt": dict(self.receipt),
        }

    as_dict = to_dict

    def render(self) -> str:
        return render_report(self)


@dataclass(frozen=True)
class RegistryEntry:
    organ_id: str
    owner: str
    status: str
    trigger: str
    fire_events: tuple[str, ...]
    suppression_events: tuple[str, ...]
    dedup_events: tuple[str, ...]
    challenge_once_events: tuple[str, ...]


def _split_registry_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def parse_registry(value: str | Path) -> tuple[RegistryEntry, ...]:
    """Parse the machine-readable CENSUS REGISTRY markdown table."""

    text = value.read_text(encoding="utf-8") if isinstance(value, Path) else value
    header: list[str] | None = None
    entries: list[RegistryEntry] = []
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = _split_registry_row(line)
        if not cells or all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        lowered = [_normal(cell) for cell in cells]
        if "organ_id" in lowered and "owner" in lowered:
            header = lowered
            continue
        if header is None or len(cells) < len(header):
            continue
        row = dict(zip(header, cells))
        required = {
            "organ_id",
            "owner",
            "status",
            "trigger",
            "fire_events",
            "suppression_events",
            "dedup_events",
            "challenge_once_events",
        }
        if not required.issubset(row):
            continue

        def event_list(name: str) -> tuple[str, ...]:
            raw = row.get(name, "")
            return tuple(item.strip() for item in raw.split(";") if item.strip())

        entries.append(
            RegistryEntry(
                organ_id=row["organ_id"].strip(),
                owner=row["owner"].strip(),
                status=row["status"].strip(),
                trigger=row["trigger"].strip(),
                fire_events=event_list("fire_events"),
                suppression_events=event_list("suppression_events"),
                dedup_events=event_list("dedup_events"),
                challenge_once_events=event_list("challenge_once_events"),
            )
        )
    return tuple(entries)


def registry_errors(
    status: str | Path,
    specs: Iterable[OrganCensusSpec],
) -> tuple[str, ...]:
    """Return invariant violations without importing or executing source text."""

    entries = parse_registry(status)
    declared = tuple(specs)
    active_entries = tuple(
        entry for entry in entries
        if not entry.status.upper().startswith("TODO")
        and not entry.owner.upper().startswith("TODO")
    )
    errors: list[str] = []
    entry_ids = [entry.organ_id for entry in active_entries]
    spec_ids = [spec.organ_id for spec in declared]
    for organ_id in sorted({item for item in entry_ids if entry_ids.count(item) > 1}):
        errors.append(f"duplicate registry organ: {organ_id}")
    for organ_id in sorted(set(entry_ids) - set(spec_ids)):
        errors.append(f"registry organ has no declaration: {organ_id}")
    for organ_id in sorted(set(spec_ids) - set(entry_ids)):
        errors.append(f"declaration has no registry organ: {organ_id}")
    for spec in declared:
        if not callable(spec.trigger):
            errors.append(f"predicate is not callable: {spec.organ_id}")
        if not (spec.fire_events or spec.suppression_events or spec.dedup_events or spec.challenge_once_events):
            errors.append(f"event vocabulary is empty: {spec.organ_id}")
    for entry in active_entries:
        if not entry.trigger or not entry.fire_events or not entry.suppression_events:
            errors.append(f"registry fields incomplete: {entry.organ_id}")
    return tuple(errors)


def assert_registry_invariant(
    status: str | Path,
    specs: Iterable[OrganCensusSpec],
) -> None:
    errors = registry_errors(status, specs)
    if errors:
        raise RegistryInvariantError("; ".join(errors))


validate_registry = assert_registry_invariant


def _candidate_evidence(evidence: Evidence) -> tuple[Evidence, ...]:
    candidates: list[Evidence] = []
    if evidence.record:
        candidates.append(evidence)
    else:
        candidates.append(
            Evidence(
                receipts=evidence.receipts,
                traces=evidence.traces,
                organ_outputs=evidence.organ_outputs,
                run_id=evidence.run_id,
                complete=evidence.complete,
                gaps=evidence.gaps,
                source="run",
                record={"source": "run", "event": "RUN"},
            )
        )
    for source, records in (
        ("receipt", evidence.receipts),
        ("trace", evidence.traces),
        ("organ_out", evidence.organ_outputs),
    ):
        candidates.extend(
            Evidence(
                receipts=evidence.receipts,
                traces=evidence.traces,
                organ_outputs=evidence.organ_outputs,
                run_id=evidence.run_id,
                complete=evidence.complete,
                gaps=evidence.gaps,
                source=source,
                record=record,
            )
            for record in records
        )
    return tuple(candidates)


def _field(record: Record, *names: str) -> object:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def _event_name(record: Record) -> str:
    return _text(_field(record, *_EVENT_FIELD_NAMES))


def _event_tokens(record: Record) -> frozenset[str]:
    values: set[str] = set()
    for name in _EVENT_FIELD_NAMES:
        raw = _text(record.get(name))
        if raw:
            values.add(_normal(raw))
    kind = _normal(record.get("kind"))
    result = _normal(record.get("result_class") or record.get("result"))
    intent = _normal(record.get("intent"))
    if kind:
        values.add(kind)
    if result:
        values.add(result)
    if kind and result:
        values.add(f"{kind}:{result}")
    if kind and intent:
        values.add(f"{kind}:{intent}")
    if record.get("emitted") is True:
        values.add("delivered")
    if record.get("shadowed") is True or record.get("would_fire") is True:
        values.add("shadowed")
    return frozenset(values)


def _organ_values(record: Record) -> tuple[str, ...]:
    values: list[str] = []
    for name in _ORGAN_FIELD_NAMES:
        raw = _normal(record.get(name))
        if raw:
            values.append(raw)
    kind = _normal(record.get("kind"))
    if kind:
        values.append(kind)
    detail = _text(record.get("detail"))
    match = _DETAIL_ORGAN.search(detail)
    if match:
        values.append(_normal(match.group("organ")))
    event = _normal(_event_name(record))
    if event.startswith("would_fire:"):
        values.append(event.removeprefix("would_fire:"))
    return tuple(values)


def _belongs(record: Record, organ_id: str) -> bool:
    target = _normal(organ_id)
    base = target.split(".", 1)[0]
    leaf = target.rsplit(".", 1)[-1]
    for value in _organ_values(record):
        if value in {target, base, leaf} or value.startswith(f"{target}.") or target.startswith(f"{value}."):
            return True
    event = _normal(_event_name(record))
    return base in event or target in event


def _matches(record: Record, labels: Iterable[str], *, organ_id: str = "") -> bool:
    tokens = _event_tokens(record)
    normalized = {_normal(label) for label in labels if _text(label)}
    if tokens.intersection(normalized):
        return True
    event = _normal(_event_name(record))
    if organ_id and event.startswith("would_fire:"):
        target = _normal(organ_id)
        shadow_organ = event.removeprefix("would_fire:")
        if shadow_organ in {target, target.split(".", 1)[0]}:
            return True
    return False


def _is_shadow(record: Record) -> bool:
    event = _normal(_event_name(record))
    return bool(
        record.get("shadowed") is True
        or record.get("would_fire") is True
        or event == "organ_shadowed"
        or event.startswith("would_fire:")
    )


def _is_delivered(record: Record) -> bool:
    event = _normal(_event_name(record))
    return bool(record.get("emitted") is True or event in {"organ_delivered", "delivered"})


def _all_records(evidence: Evidence) -> tuple[Record, ...]:
    return evidence.receipts + evidence.traces + evidence.organ_outputs


def _result_verdict(
    *,
    n: int,
    m: int,
    s: int,
    deduped: int,
    challenge_once: int,
    incomplete: bool,
) -> OrganVerdict:
    if incomplete:
        return OrganVerdict.INCOMPLETE
    if m > 0:
        return OrganVerdict.FIRED
    if s > 0 or deduped > 0 or challenge_once > 0:
        return OrganVerdict.SUPPRESSED
    if n > 0:
        return OrganVerdict.DARK
    return OrganVerdict.NOT_TRIGGERED


def _digest_payload(run_id: str, status: str, organs: Sequence[OrganCensusResult]) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "status": status,
        "organs": {item.organ_id: item.to_dict() for item in organs},
    }


def _short_digest(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def _emit_receipt(
    receipt: Mapping[str, object],
    *,
    receipt_path: str | Path | None,
    receipt_sink: ReceiptSink | None,
) -> tuple[str, ...]:
    errors: list[str] = []
    if receipt_path is not None:
        try:
            path = Path(receipt_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(receipt), sort_keys=True) + "\n")
        except OSError as exc:
            errors.append(f"receipt_persist_failed:{type(exc).__name__}")
            print(f"[kernel-wiring] coverage check receipt persist failed: {type(exc).__name__}")
    if receipt_sink is not None:
        try:
            receipt_sink(receipt)
        except Exception as exc:  # noqa: BLE001 - census must not break task finalization
            errors.append(f"receipt_sink_failed:{type(exc).__name__}")
            print(f"[kernel-wiring] coverage check receipt sink failed: {type(exc).__name__}")
    return tuple(errors)


def _coerce_evidence(
    evidence: Evidence | Mapping[str, object] | None,
    *,
    receipts: object,
    traces: object,
    organ_outputs: object,
    run_id: str,
    complete: bool,
    gaps: Iterable[object],
) -> Evidence:
    if isinstance(evidence, Evidence):
        return evidence
    if isinstance(evidence, Mapping):
        return Evidence.from_mapping(evidence)
    return Evidence.from_records(
        receipts=receipts,
        traces=traces,
        organ_outputs=organ_outputs,
        run_id=run_id,
        complete=complete,
        gaps=gaps,
    )


def run_census(
    evidence: Evidence | Mapping[str, object] | None = None,
    specs: Iterable[OrganCensusSpec] | None = None,
    *,
    receipts: object = (),
    traces: object = (),
    organ_outputs: object = (),
    run_id: str = "",
    complete: bool = True,
    gaps: Iterable[object] = (),
    receipt_path: str | Path | None = None,
    receipt_sink: ReceiptSink | None = None,
) -> CensusReport:
    """Run one deterministic census and finalize with exactly one census receipt."""

    base = _coerce_evidence(
        evidence,
        receipts=receipts,
        traces=traces,
        organ_outputs=organ_outputs,
        run_id=run_id,
        complete=complete,
        gaps=gaps,
    )
    effective_run_id = base.run_id or _text(run_id) or "run"
    errors: list[str] = []
    try:
        declarations = tuple(specs) if specs is not None else registered_specs()
        results: list[OrganCensusResult] = []
        candidates = _candidate_evidence(base)
        all_records = _all_records(base)
        for spec in declarations:
            n = 0
            unknown = 0
            local_gaps = list(base.gaps)
            try:
                for candidate in candidates:
                    state = TriState.coerce(spec.trigger(candidate))
                    if state is TriState.YES:
                        n += 1
                    elif state is TriState.UNKNOWN:
                        unknown += 1
                m = s = d = deduped = challenge_once = shadowed = 0
                for record in all_records:
                    if not _belongs(record, spec.organ_id):
                        continue
                    if _matches(record, spec.fire_events, organ_id=spec.organ_id):
                        m += 1
                        if _is_shadow(record):
                            shadowed += 1
                    if _matches(record, spec.suppression_events, organ_id=spec.organ_id):
                        s += 1
                    if _matches(record, spec.dedup_events, organ_id=spec.organ_id):
                        deduped += 1
                    if _matches(record, spec.challenge_once_events, organ_id=spec.organ_id):
                        challenge_once += 1
                    if _is_delivered(record) and _belongs(record, spec.organ_id):
                        d += 1
                if unknown:
                    local_gaps.append(f"trigger_unknown={unknown}")
                incomplete = not base.complete or bool(local_gaps) or unknown > 0
                verdict = _result_verdict(
                    n=n,
                    m=m,
                    s=s,
                    deduped=deduped,
                    challenge_once=challenge_once,
                    incomplete=incomplete,
                )
                results.append(
                    OrganCensusResult(
                        organ_id=spec.organ_id,
                        n=n,
                        m=m,
                        s=s,
                        d=d,
                        deduped=deduped,
                        challenge_once=challenge_once,
                        shadowed=shadowed,
                        verdict=verdict,
                        evidence_gaps=tuple(dict.fromkeys(local_gaps)),
                        trigger_unknown=unknown,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad organ cannot hide census failure
                errors.append(f"{spec.organ_id}:{type(exc).__name__}")
                results.append(
                    OrganCensusResult(
                        organ_id=spec.organ_id,
                        verdict=OrganVerdict.INCOMPLETE,
                        evidence_gaps=(*base.gaps, f"predicate_error:{type(exc).__name__}"),
                    )
                )
        results_tuple = tuple(sorted(results, key=lambda item: item.organ_id))
        status = "ERROR" if errors else "INCOMPLETE" if any(
            item.verdict is OrganVerdict.INCOMPLETE for item in results_tuple
        ) else "COMPLETE"
        payload = _digest_payload(effective_run_id, status, results_tuple)
        digest = _short_digest(payload)
        receipt: dict[str, object] = {
            "kind": "census",
            "intent": "liveness-census",
            "schema": SCHEMA,
            "run_id": effective_run_id,
            "status": status,
            "result_class": status,
            "digest": digest,
            "counts": {item.organ_id: item.to_dict() for item in results_tuple},
        }
        receipt_errors = _emit_receipt(
            receipt,
            receipt_path=receipt_path,
            receipt_sink=receipt_sink,
        )
        errors.extend(receipt_errors)
        return CensusReport(
            run_id=effective_run_id,
            status=status,
            organs=results_tuple,
            digest=digest,
            receipt=receipt,
            evidence_gaps=base.gaps,
            errors=tuple(errors),
        )
    except Exception as exc:  # noqa: BLE001 - finalization must always attempt a receipt
        errors.append(f"census_error:{type(exc).__name__}")
        status = "ERROR"
        digest = _short_digest({"run_id": effective_run_id, "status": status, "errors": errors})
        receipt = {
            "kind": "census",
            "intent": "liveness-census",
            "schema": SCHEMA,
            "run_id": effective_run_id,
            "status": status,
            "result_class": status,
            "digest": digest,
            "counts": {},
            "errors": errors,
        }
        errors.extend(
            _emit_receipt(receipt, receipt_path=receipt_path, receipt_sink=receipt_sink)
        )
        return CensusReport(
            run_id=effective_run_id,
            status=status,
            organs=(),
            digest=digest,
            receipt=receipt,
            evidence_gaps=base.gaps,
            errors=tuple(errors),
        )


def census(*args: object, **kwargs: object) -> CensusReport:
    """Short compatibility entry point for callers naming the operation census."""

    return run_census(*args, **kwargs)  # type: ignore[arg-type]


run_liveness_census = run_census


def registered_specs() -> tuple[OrganCensusSpec, ...]:
    """Load declarations from organ owners without embedding their predicates here."""

    module_names = (
        "kernel_bolt",
        "kernel_semantics",
        "kernel_nitpicker",
        "kernel_vitals",
        "expectation_ledger",
    )
    declarations: list[OrganCensusSpec] = []
    for module_name in module_names:
        module = __import__(module_name, fromlist=["CENSUS_SPECS"])
        values = getattr(module, "CENSUS_SPECS", ())
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"{module_name}.CENSUS_SPECS must be a sequence")
        declarations.extend(item for item in values if isinstance(item, OrganCensusSpec))
    by_id: dict[str, OrganCensusSpec] = {}
    for spec in declarations:
        if spec.organ_id in by_id:
            raise RegistryInvariantError(f"duplicate code declaration: {spec.organ_id}")
        by_id[spec.organ_id] = spec
    return tuple(by_id[key] for key in sorted(by_id))


def _load_json_lines(path: Path) -> tuple[Record, ...]:
    records: list[Record] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                records.append(value)
    except OSError:
        return ()
    return tuple(records)


def _load_json(path: Path) -> Mapping[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _run_suffix(receipts_dir: Path) -> str:
    return receipts_dir.name.removeprefix("receipts")


def retro_census(results_root: str | Path = "results") -> tuple[dict[str, object], ...]:
    """Census old r9 tapes, preserving CENSUS_NOT_RUN as INCOMPLETE."""

    root = Path(results_root)
    rows: list[dict[str, object]] = []
    declarations = registered_specs()
    for receipts_dir in sorted(root.glob("receipts_r9*")):
        if not receipts_dir.is_dir():
            continue
        suffix = _run_suffix(receipts_dir)
        transcript_dir = root / f"react_transcripts{suffix}"
        receipt_files = tuple(sorted(receipts_dir.rglob("*.jsonl")))
        receipt_records = tuple(
            record
            for path in receipt_files
            for record in _load_json_lines(path)
        )
        transcript_files = tuple(sorted(transcript_dir.rglob("*.json"))) if transcript_dir.is_dir() else ()
        traces: list[Record] = []
        transcript_has_liveness = False
        for path in transcript_files:
            transcript = _load_json(path)
            kernel = transcript.get("kernel") if transcript else None
            kernel_map = _mapping(kernel)
            trace = kernel_map.get("trace", ()) if kernel_map else ()
            traces.extend(record for record in _record_tuple(trace))
            if kernel_map and isinstance(kernel_map.get("liveness"), Mapping):
                transcript_has_liveness = True
        raw_census = any(_normal(record.get("kind")) == "census" for record in receipt_records)
        census_trace = transcript_has_liveness or any(
            "census" in _normal(record.get("event")) for record in traces
        )
        gaps: list[str] = []
        if not raw_census:
            gaps.append("CENSUS_NOT_RUN: no kind=census receipt in source tape")
        if not census_trace:
            gaps.append("CENSUS_TRACE_NOT_RUN: no census trace event in source transcript")
        if not transcript_files:
            gaps.append("matching_transcript_missing")
        evidence = Evidence.from_records(
            receipts=receipt_records,
            traces=traces,
            run_id=suffix.removeprefix("_") or receipts_dir.name,
            complete=bool(transcript_files),
            gaps=gaps,
        )
        computed = run_census(evidence, specs=declarations)
        for item in computed.organs:
            verdict = item.verdict
            if gaps:
                verdict = OrganVerdict.INCOMPLETE
            rows.append(
                {
                    "run_id": evidence.run_id,
                    "organ_id": item.organ_id,
                    "N": item.n,
                    "M": item.m,
                    "S": item.s,
                    "D": item.d,
                    "deduped": item.deduped,
                    "challenge_once": item.challenge_once,
                    "shadowed": item.shadowed,
                    "verdict": verdict.value,
                    "evidence_gaps": tuple(dict.fromkeys((*gaps, *item.evidence_gaps))),
                }
            )
    return tuple(rows)


def render_retro(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| run | organ | N | M | S | D | deduped | challenge_once | shadowed | verdict | evidence gaps |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        gaps = "; ".join(_text(item) for item in row.get("evidence_gaps", ()) if _text(item))
        lines.append(
            f"| {_text(row.get('run_id'))} | {_text(row.get('organ_id'))} | "
            f"{int(row.get('N') or 0)} | {int(row.get('M') or 0)} | "
            f"{int(row.get('S') or 0)} | {int(row.get('D') or 0)} | "
            f"{int(row.get('deduped') or 0)} | {int(row.get('challenge_once') or 0)} | "
            f"{int(row.get('shadowed') or 0)} | {_text(row.get('verdict'))} | {gaps} |"
        )
    return "\n".join(lines)


def render_report(report: CensusReport) -> str:
    lines = [
        f"KERNEL LIVENESS CENSUS — run {report.run_id} status={report.status} digest={report.digest}",
        "| organ | N | M | S | D | deduped | challenge_once | shadowed | verdict | gaps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in report.organs:
        gaps = "; ".join(item.evidence_gaps)
        lines.append(
            f"| {item.organ_id} | {item.n} | {item.m} | {item.s} | {item.d} | "
            f"{item.deduped} | {item.challenge_once} | {item.shadowed} | "
            f"{item.verdict.value} | {gaps} |"
        )
        if item.verdict is OrganVerdict.DARK:
            lines.append(
                f"[FINDINGS] DARK ORGAN organ={item.organ_id} N={item.n} M={item.m} S={item.s}"
            )
    if report.errors:
        lines.append("[kernel-wiring] census finalization errors: " + ", ".join(report.errors))
    return "\n".join(lines)


def _main_retro(results_root: str) -> int:
    rows = retro_census(results_root)
    print(render_retro(rows))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an offline liveness census over r9 tapes.")
    parser.add_argument("--retro", action="store_true", help="coverage check results/receipts_r9* and matching transcripts")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args(argv)
    if args.retro:
        return _main_retro(args.results_root)
    parser.error("select --retro")
    return 2


__all__ = [
    "CensusReport",
    "Evidence",
    "OrganCensusResult",
    "OrganCensusSpec",
    "OrganVerdict",
    "RegistryEntry",
    "RegistryInvariantError",
    "SCHEMA",
    "TriState",
    "assert_registry_invariant",
    "catalog_unknown_opportunity",
    "census",
    "parse_registry",
    "registered_specs",
    "registry_errors",
    "render_report",
    "render_retro",
    "retro_census",
    "run_census",
    "run_liveness_census",
    "validate_registry",
]


if __name__ == "__main__":
    raise SystemExit(main())

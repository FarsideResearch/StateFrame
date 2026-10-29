"""UNKNOWNS LEDGER: host producers -> Catalog A unknowns -> ``[OPEN UNKNOWNS]``.

``kernel_catalog`` owns the data model (``UnknownRecord``/``Resolution``) and the
budgeted ``unknowns()`` query; ``StateFrameRuntime`` already replays its host
receipt stream into it and gates completion on blocking-scope rows.  This
module is the missing producer + renderer seam:

* producers read HOST measurements already on the runtime (entry-symbol
  verification, ASSUMECHECK structural facts, the state-owner inventory,
  the requirements ledger) and turn them into explicit catalog receipts;
* an unknown, once opened, stays in the stream (sticky) until a later HOST
  measurement of the same fact produces a resolving receipt
  (``resolves_unknown_ids``).  Model text is never read here, so a model
  claim has no path to resolve anything;
* ``render`` produces the bounded ``[OPEN UNKNOWNS]`` sidecar, each row with
  its cheapest resolving action.

Critical unknowns carry ``blocking_scope`` == the task completion scope and
are gated by the existing ``catalog_completion_blocker``; advisory unknowns use
``blocking_scope="advisory"`` and are only rendered.

Switch: ``STATEFRAME_UNKNOWNS_LEDGER`` (default ``on``).  Off => no receipts,
no sidecar, no events (byte-identical to the pre-ledger runtime).
Pure host code: no provider calls, never imports ``tools.*``.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence

_OFF = frozenset({"0", "off", "false", "no", "disabled"})
ADVISORY_SCOPE = "advisory"
SIDECAR_HEADER = "[OPEN UNKNOWNS]"
RENDER_TOKEN_BUDGET = 1024
_MAX_ROWS = 6
_SEQ_BASE = 9_000_000  # after every runtime-native stream sequence

_ACTION = {
    "READ": "read", "SEARCH": "grep", "CHECK": "run_tests",
    "REFRESH": "read", "USER_INPUT": "question_gate", "UNAVAILABLE": "question_gate",
}


def mode(environ: Mapping[str, str] | None = None) -> str:
    value = (environ if environ is not None else os.environ).get("STATEFRAME_UNKNOWNS_LEDGER", "on")
    return "off" if str(value).strip().lower() in _OFF else "on"


def _emit(runtime: object, event: str, **fields: object) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        try:
            emit(event, producer="kernel_unknowns_ledger", dispatch=False, **fields)
        except (TypeError, ValueError):
            pass


def _clip(text: object, limit: int = 140) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


class _Fact:
    """One host measurement of a ledger key: open (with draft) or resolved."""

    __slots__ = ("key", "open", "predicate", "subject", "reason", "resolution",
                 "critical", "action", "revision")

    def __init__(self, key: str, *, open: bool, predicate: str, subject: str,
                 reason: str = "", resolution: str = "READ", critical: bool = False,
                 action: str = "", revision: str = "") -> None:
        self.key = key
        self.open = open
        self.predicate = predicate
        self.subject = subject
        self.reason = reason
        self.resolution = resolution
        self.critical = critical
        self.action = action
        self.revision = revision or "host"


# ------------------------------------------------------------------ producers

def _entry_facts(runtime: object) -> list[_Fact]:
    """Entry-symbol verification (kernel_request_source_window): NOT_FOUND is
    critical, REFERENCED_ONLY advisory; FOUND* is the resolving measurement."""
    entry = getattr(runtime, "_site_facts_entry", None)
    if not isinstance(entry, Mapping) or not entry.get("symbol"):
        return []
    symbol = str(entry["symbol"])
    status = str(entry.get("status") or "")
    revision = str(entry.get("revision") or "")
    key = f"entry:{symbol}@{revision or 'unknown'}"
    if status.startswith("FOUND"):
        return [_Fact(key, open=False, predicate="entry.symbol", subject=symbol, revision=revision)]
    if status not in {"NOT_FOUND", "REFERENCED_ONLY"} or entry.get("new_symbol"):
        return []
    if status == "NOT_FOUND":
        return [_Fact(key, open=True, predicate="entry.symbol", subject=symbol,
                      reason="no definition in declared files or repo index",
                      resolution="SEARCH", critical=True, revision=revision,
                      action=f"grep for the definition of `{symbol}` (or define it where the ticket says)")]
    return [_Fact(key, open=True, predicate="entry.symbol", subject=symbol,
                  reason="referenced by public tests, defined nowhere",
                  resolution="READ", critical=False, revision=revision,
                  action=f"read the public test that references `{symbol}` to learn its expected shape")]


def _assume_facts(runtime: object) -> list[_Fact]:
    """ASSUMECHECK structural facts (kernel_catalog_assume): MISSING symbols /
    imports the task row assumes are critical; CONFIRMED resolves them.
    (UNKNOWN rows already flow through ``_catalog_assumption_receipts``.)"""
    row = getattr(getattr(runtime, "owner", None), "task_row", {})
    raw = row.get("assumptions") if isinstance(row, Mapping) else None
    if raw in (None, (), []):
        return []
    cache = getattr(runtime, "_unknowns_ledger_assume_cache", None)
    root = getattr(getattr(runtime, "owner", None), "root", None)
    if root is None:
        return []
    from pathlib import Path

    from kernel_catalog_assume import verify_indexed_assumptions
    source_revision = getattr(runtime, "source_revision", getattr(runtime, "product_revision", "unknown"))
    index_revision = getattr(runtime, "index_revision", getattr(runtime, "repository_index_revision", "unknown"))
    fingerprint_payload = json.dumps({
        "assumptions": raw,
        "source_revision": str(source_revision),
        "index_revision": str(index_revision),
        "root": str(Path(root).resolve()),
    }, sort_keys=True, ensure_ascii=False, default=str,
       separators=(",", ":")).encode("utf-8")
    fingerprint = "sha256:" + hashlib.sha256(fingerprint_payload).hexdigest()
    if cache is None or cache[0] != fingerprint:
        report = verify_indexed_assumptions(Path(root), raw)
        cache = (fingerprint, report)
        try:
            runtime._unknowns_ledger_assume_cache = cache
        except (AttributeError, TypeError):
            pass
    facts: list[_Fact] = []
    for fact in cache[1].facts:
        revision = str(getattr(getattr(fact, "coverage", None), "coverage_id", "") or "")
        key = f"assume:{fact.kind}:{fact.target}@{revision or 'unknown'}"
        coverage = getattr(fact, "coverage", None)
        coverage_complete = (
            coverage.get("complete") if isinstance(coverage, Mapping)
            else getattr(coverage, "complete", None)
        )
        if coverage_complete is False:
            facts.append(_Fact(key, open=True, predicate=f"assumption.{fact.kind}",
                               subject=fact.target, reason="incomplete indexed search remains unknown",
                               resolution="SEARCH", critical=True, revision=revision,
                               action=f"expand search for `{fact.target}`; absence is not proven"))
        elif fact.status == "CONFIRMED":
            facts.append(_Fact(key, open=False, predicate=f"assumption.{fact.kind}",
                               subject=fact.target, revision=revision))
        elif fact.status in {"MISSING", "MISMATCH"}:
            facts.append(_Fact(key, open=True, predicate=f"assumption.{fact.kind}",
                               subject=fact.target, reason=_clip(fact.message) or fact.status.lower(),
                               resolution="SEARCH" if fact.kind in {"symbol_exists", "import_path"} else "READ",
                               critical=True, revision=revision,
                               action=f"grep for `{fact.target}`; the task assumes it exists"))
    return facts


def _owner_facts(runtime: object) -> list[_Fact]:
    """State-owner inventory with writers=0 (advisory: nothing sets the field yet)."""
    inventory = getattr(runtime, "_state_owner_inventory", None)
    fields = getattr(inventory, "fields", None)
    if not isinstance(fields, Sequence) or not fields:
        return []
    facts: list[_Fact] = []
    revision = str(getattr(inventory, "revision", "") or "")
    for item in fields[:_MAX_ROWS]:
        name = str(getattr(item, "name", "") or "")
        if not name:
            continue
        key = f"owner:{getattr(item, 'path', '')}:{name}@{revision or 'unknown'}"
        if getattr(item, "writers", None):
            facts.append(_Fact(key, open=False, predicate="state.writer", subject=name, revision=revision))
        else:
            facts.append(_Fact(key, open=True, predicate="state.writer", subject=name,
                               reason=f"0 writers in product code ({getattr(item, 'path', '?')}:{getattr(item, 'line', '?')})",
                               resolution="READ", critical=False, revision=revision,
                               action=f"read `{getattr(item, 'path', '?')}` and decide which transition must set `{name}`"))
    return facts


def _requirement_facts(runtime: object) -> list[_Fact]:
    """Requirement clauses the host could not interpret (kind=unresolved)."""
    ledger = getattr(runtime, "requirements_ledger", None)
    rows = ledger.get("requirements") if isinstance(ledger, Mapping) else None
    if not isinstance(rows, Sequence):
        return []
    binding = getattr(runtime, "binding", None)
    try:
        bound = binding() if callable(binding) else binding
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        bound = None
    product_revision = (bound.get("product_revision") if isinstance(bound, Mapping) else None)
    source_revision = (bound.get("source_revision") if isinstance(bound, Mapping) else None)
    index_revision = (bound.get("index_revision", bound.get("repository_index_revision"))
                      if isinstance(bound, Mapping) else None)
    revision = json.dumps({
        "product": product_revision or ledger.get("current_revision") or "",
        "source": source_revision or getattr(runtime, "source_revision", ""),
        "index": index_revision or getattr(runtime, "index_revision", ""),
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    facts: list[_Fact] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("kind") != "unresolved":
            continue
        req_id = str(row.get("id") or "")
        assessment = row.get("assessment") if isinstance(row.get("assessment"), Mapping) else {}
        key = f"requirement:{req_id}@{revision or 'unknown'}"
        status = str(assessment.get("status") or "open").lower()
        if status != "open" and assessment.get("evidence_ids"):
            facts.append(_Fact(key, open=False, predicate="requirement.clause", subject=req_id, revision=revision))
            continue
        if status == "open":
            contract = row.get("contract") if isinstance(row.get("contract"), Mapping) else {}
            if contract.get("entrypoint") and "input" in contract and "expected" in contract:
                action = f"probe `{contract['entrypoint']}` for clause `{req_id}` with the declared public case"
                resolution = "CHECK"
            else:
                # A clause without a host-checkable contract cannot be marked verified.
                # rendered every turn as an OPEN row "CHECK: run the declared public check for R1" under
                # "CRITICAL rows block done". Nothing can check it, so the model obeyed it into a run_tests loop.
                # A no-contract clause is not an open unknown; it is not rendered.
                continue
            facts.append(_Fact(key, open=True, predicate="requirement.clause", subject=req_id,
                               reason=f"no host-checkable contract: {_clip(row.get('exact_clause'), 80)}",
                               resolution=resolution, critical=False, revision=revision,
                               action=action))
    return facts


_PRODUCERS = (_entry_facts, _assume_facts, _owner_facts, _requirement_facts)


# ------------------------------------------------------------------ ledger

def _state(runtime: object) -> dict[str, object]:
    state = getattr(runtime, "_unknowns_ledger_state", None)
    if not isinstance(state, dict):
        state = {"open": {}, "resolved": set(), "receipts": [], "actions": {}, "rendered": None}
        try:
            runtime._unknowns_ledger_state = state
        except (AttributeError, TypeError):
            pass
    return state


def _stream_marker(value: object) -> tuple[int, int, int]:
    from kernel_history_spool import HistorySpool
    if isinstance(value, HistorySpool):
        # Append-only row identity stays stable when its payload is rehydrated.
        return (id(value), len(value), len(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return (id(value), len(value), id(value[-1]) if value else 0)
    return (id(value), -1, 0)


def _render_input_revision(runtime: object, *, task_id: str, scope: str) -> tuple[object, ...] | None:
    """Return cheap revision axes for skipping an unchanged catalog collection pass.

    The journal cursor advances on receipt and host-state events. Stream shape and
    ledger revisions cover the runtime inputs that are prepared outside those rows.
    """
    # Keep the explicit uncached-replay path and retry transient projection errors.
    if (getattr(runtime, "_catalog_projection_error", None) is not None
            or os.environ.get("STATEFRAME_HOST_STEP_BUDGET2_CACHE", "on").strip().lower() in _OFF):
        return None
    journal = getattr(runtime, "journal", None)
    cursor_reader = getattr(journal, "receipt_cursor", None)
    if not callable(cursor_reader):
        return None
    try:
        cursor = cursor_reader()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
    sequence = getattr(cursor, "sequence", None)
    generation = getattr(cursor, "receipt_generation", None)
    if type(sequence) is not int or type(generation) is not int:
        return None

    owner = getattr(runtime, "owner", None)
    raw_task_row = getattr(owner, "task_row", None)
    task_row = raw_task_row if isinstance(raw_task_row, Mapping) else {}
    raw_ledger = getattr(runtime, "requirements_ledger", None)
    ledger = raw_ledger if isinstance(raw_ledger, Mapping) else {}
    requirement_rows = ledger.get("requirements", ())
    requirement_marker: list[tuple[object, ...]] = []
    if isinstance(requirement_rows, Sequence) and not isinstance(
            requirement_rows, (str, bytes, bytearray)):
        for row in requirement_rows:
            if not isinstance(row, Mapping) or row.get("kind") != "unresolved":
                continue
            assessment = row.get("assessment")
            assessment = assessment if isinstance(assessment, Mapping) else {}
            contract = row.get("contract")
            contract = contract if isinstance(contract, Mapping) else {}
            evidence_ids = assessment.get("evidence_ids", ())
            if not isinstance(evidence_ids, Sequence) or isinstance(
                    evidence_ids, (str, bytes, bytearray)):
                evidence_ids = ()
            requirement_marker.append((
                str(row.get("id") or ""), str(row.get("exact_clause") or ""),
                str(assessment.get("status") or "open").lower(),
                tuple(str(item) for item in evidence_ids),
                str(contract.get("entrypoint") or ""),
                "input" in contract, "expected" in contract,
            ))

    inventory = getattr(runtime, "_state_owner_inventory", None)
    inventory_fields = getattr(inventory, "fields", ())
    owner_field_marker: list[tuple[str, str, str, str, bool]] = []
    if isinstance(inventory_fields, Sequence) and not isinstance(
            inventory_fields, (str, bytes, bytearray)):
        for item in inventory_fields[:_MAX_ROWS]:
            owner_field_marker.append((
                str(getattr(item, "name", "") or ""),
                str(getattr(item, "path", "") or ""),
                str(getattr(item, "line", "") or ""),
                str(getattr(inventory, "revision", "") or ""),
                bool(getattr(item, "writers", None)),
            ))

    entry = getattr(runtime, "_site_facts_entry", None)
    if isinstance(entry, Mapping):
        entry_marker: object = tuple(
            str(entry.get(key) or "") for key in ("symbol", "status", "revision", "new_symbol")
        )
    else:
        entry_marker = id(entry)

    assumptions = task_row.get("assumptions")
    try:
        assumptions_marker: object = json.dumps(
            assumptions, sort_keys=True, ensure_ascii=False, default=str,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError):
        assumptions_marker = _stream_marker(assumptions)

    task_catalog = task_row.get("catalog_receipts", ())
    streams = (
        _stream_marker(getattr(owner, "_receipts", ())),
        _stream_marker(getattr(runtime, "test_receipts", ())),
        _stream_marker(getattr(runtime, "host_check_receipts", ())),
        _stream_marker(getattr(runtime, "host_repair_check_receipts", ())),
        _stream_marker(task_catalog),
    )
    return (
        sequence, generation, getattr(runtime, "_binding_epoch", None),
        task_id, scope, id(owner), id(raw_task_row), streams,
        assumptions_marker, str(getattr(owner, "root", "")),
        str(getattr(runtime, "source_revision", "")),
        str(getattr(runtime, "index_revision", "")),
        str(getattr(runtime, "repository_index_revision", "")),
        id(raw_ledger), str(ledger.get("current_revision") or ""),
        _stream_marker(requirement_rows), tuple(requirement_marker),
        entry_marker, id(inventory), str(getattr(inventory, "revision", "") or ""),
        tuple(owner_field_marker),
    )


def _unknown_id(receipt: Mapping[str, object]) -> str:
    from kernel_catalog import replay
    projection = replay((receipt,))
    rows = projection.blocking_unknowns()
    raw = receipt.get("unknowns", receipt.get("unknown", ()))
    if isinstance(raw, Mapping):
        raw = (raw,)
    requested = next((item for item in raw if isinstance(item, Mapping)), {}) \
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)) else {}
    predicate = str(requested.get("predicate", ""))
    subject = requested.get("subject", requested.get("entity_id", ""))
    subject_id = (str(subject.get("entity_id", subject.get("id", subject.get("path", ""))))
                  if isinstance(subject, Mapping) else str(subject))
    for row in rows:
        if predicate and row.predicate != predicate:
            continue
        if subject_id and row.subject is not None and row.subject.entity_id != subject_id:
            continue
        return row.unknown_id
    # A producer bug must be diagnosable, not an IndexError from a budgeted
    # presentation page. This fallback is never a resolved catalog row.
    payload = json.dumps(dict(receipt), sort_keys=True, default=str,
                         separators=(",", ":")).encode("utf-8")
    return "unknown:unmaterialized:" + hashlib.sha256(payload).hexdigest()[:24]


def observe(runtime: object, *, task_id: str, scope: str) -> tuple[dict[str, object], ...]:
    """Run every producer once and return the sticky ledger receipt stream."""
    if mode() != "on":
        return ()
    state = _state(runtime)
    facts: list[_Fact] = []
    for producer in _PRODUCERS:
        try:
            facts.extend(producer(runtime))
        except (AttributeError, ImportError, KeyError, OSError, RecursionError, SyntaxError,
                TypeError, ValueError) as error:
            _emit(runtime, "unknowns_producer_failed", source=producer.__name__,
                  error_type=type(error).__name__)
    opened: dict[str, str] = state["open"]  # type: ignore[assignment]
    resolved: set[str] = state["resolved"]  # type: ignore[assignment]
    receipts: list[dict[str, object]] = state["receipts"]  # type: ignore[assignment]
    open_revisions: dict[str, str] = state.setdefault("open_revisions", {})  # type: ignore[assignment]
    for fact in facts:
        # W4 / P0.1 stable scoped key: identity is the producer key WITHOUT its
        # revision suffix.  The revision is a binding: a FOUND at r2 resolves
        # (supersedes) the NOT_FOUND opened at r1 instead of being a new key.
        stable_key = fact.key.rsplit("@", 1)[0]
        _apply_fact(runtime, fact, stable_key, opened, resolved, receipts, open_revisions,
                    state, task_id=task_id, scope=scope)
    return tuple(dict(item) for item in receipts)


def _apply_fact(runtime: object, fact: _Fact, key: str, opened: dict[str, str],
                resolved: set[str], receipts: list[dict[str, object]],
                open_revisions: dict[str, str], state: dict[str, object], *,
                task_id: str, scope: str) -> None:
    """Open or resolve one fact under its stable key (one debt per key)."""
    subject_id = f"ul:{key}"
    if fact.open and (key not in opened or key in resolved):
        blocking = scope if fact.critical else ADVISORY_SCOPE
        ordinal = len(receipts)
        receipt = {
            "receipt_id": f"ul:{task_id}:{ordinal}",
            "seq": _SEQ_BASE + ordinal, "kind": "check", "check_id": subject_id,
            "status": "UNKNOWN", "result_class": "UNKNOWN", "complete": True,
            "revision": fact.revision, "required_by": [task_id], "scope": blocking,
            "producer": "kernel_unknowns_ledger", "host_admitted": True,
            "unknowns": [{
                "subject": {"entity_id": subject_id, "kind": fact.predicate.split(".")[0],
                            "locator": fact.subject},
                "predicate": fact.predicate, "reason": fact.reason,
                "resolution": fact.resolution, "required_by": [task_id],
                "blocking_scope": blocking,
            }],
        }
        unknown_id = _unknown_id(receipt)
        receipts.append(receipt)
        opened[key] = unknown_id
        open_revisions[key] = fact.revision
        resolved.discard(key)
        state["actions"][unknown_id] = fact.action  # type: ignore[index]
        _emit(runtime, "unknown_opened", unknown_id=unknown_id, key=key,
              predicate=fact.predicate, resolution=fact.resolution,
              critical=fact.critical, blocking_scope=blocking)
    elif not fact.open and key in opened and key not in resolved:
        ordinal = len(receipts)
        unknown_id = opened[key]
        scope_fields: dict[str, object] = {
            "predicate": fact.predicate,
            "subject": subject_id,
            "check_identity": f"check:{subject_id}",
            "revision": fact.revision,
        }
        if open_revisions.get(key) not in (None, fact.revision):
            # The host measured the same key at a newer revision; this is an
            # explicit, predicate-scoped supersession of the older debt.
            scope_fields["supersedes_obsolete"] = True
        receipts.append({
            "receipt_id": f"ul:{task_id}:{ordinal}",
            "seq": _SEQ_BASE + ordinal, "kind": "check", "check_id": subject_id,
            "status": "PASS", "result_class": "PASS", "complete": True,
            "revision": fact.revision, "required_by": [task_id],
            "producer": "kernel_unknowns_ledger", "host_admitted": True,
            "resolves_unknown_ids": [unknown_id],
            "resolution_scope": scope_fields,
        })
        resolved.add(key)
        _emit(runtime, "unknown_resolved", unknown_id=unknown_id, key=key,
              predicate=fact.predicate, revision=fact.revision)


# ------------------------------------------------------------------ render

def _reuse_rendered(runtime: object, state: dict[str, object], cached: dict[str, object],
                   *, task_id: str, scope: str) -> str | None:
    text = cached["text"]
    if state.get("rendered") != text:
        state["rendered"] = text
        if text is None:
            _emit(runtime, "unknowns_rendered", count=0, critical=0, omitted=0)
        else:
            _emit(runtime, "unknowns_rendered", count=cached["count"],
                  critical=cached["critical"], omitted=cached["omitted"])
    cached["input_revision"] = _render_input_revision(
        runtime, task_id=task_id, scope=scope)
    return text if isinstance(text, str) else None


def _store_render(runtime: object, *, projection: object, task_id: str, scope: str,
                  action_snapshot: tuple[tuple[str, str], ...], text: str | None,
                  count: int, critical: int, omitted: int) -> None:
    try:
        runtime._unknowns_ledger_render_cache = {
            "projection": projection,
            "task_id": task_id,
            "scope": scope,
            "actions": action_snapshot,
            "input_revision": _render_input_revision(
                runtime, task_id=task_id, scope=scope),
            "text": text,
            "count": count,
            "critical": critical,
            "omitted": omitted,
        }
    except (AttributeError, TypeError):
        pass


def render(runtime: object) -> str | None:
    """Bounded ``[OPEN UNKNOWNS]`` sidecar from the catalog projection."""
    if mode() != "on":
        return None
    sync = getattr(runtime, "_catalog_sync", None)
    task_scope = getattr(runtime, "_catalog_task_scope", None)
    if not callable(sync) or not callable(task_scope):
        return None
    task_id, scope = task_scope()
    state = _state(runtime)
    actions: Mapping[str, str] = state["actions"]  # type: ignore[assignment]
    action_snapshot = tuple(sorted((str(key), str(value)) for key, value in actions.items()))
    cached = getattr(runtime, "_unknowns_ledger_render_cache", None)
    input_revision = _render_input_revision(runtime, task_id=task_id, scope=scope)
    current_projection = getattr(runtime, "catalog_projection", None)
    if (input_revision is not None
            and isinstance(cached, dict)
            and cached.get("input_revision") == input_revision
            and cached.get("projection") is current_projection
            and cached.get("task_id") == task_id
            and cached.get("scope") == scope
            and cached.get("actions") == action_snapshot
            and (cached.get("text") is None or isinstance(cached.get("text"), str))):
        return _reuse_rendered(runtime, state, cached, task_id=task_id, scope=scope)

    # The immutable projection and input revision together bound reuse: an
    # unchanged journal/binding/source key bypasses receipt collection entirely.
    projection = sync()
    actions = state["actions"]  # type: ignore[assignment]
    action_snapshot = tuple(sorted((str(key), str(value)) for key, value in actions.items()))
    cached = getattr(runtime, "_unknowns_ledger_render_cache", None)
    if (isinstance(cached, dict)
            and cached.get("projection") is projection
            and cached.get("task_id") == task_id
            and cached.get("scope") == scope
            and cached.get("actions") == action_snapshot
            and (cached.get("text") is None or isinstance(cached.get("text"), str))):
        return _reuse_rendered(runtime, state, cached, task_id=task_id, scope=scope)

    # The catalog budget bounds the JSON rows; _MAX_ROWS + clipping bound the text.
    query = projection.unknowns(required_by=task_id, token_budget=RENDER_TOKEN_BUDGET)
    ordered = sorted(query.unknowns, key=lambda item: item.blocking_scope != scope)
    rows = [{key: value for key, value in item.as_dict().items() if key != "receipt_refs"}
            for item in ordered]
    omissions = list(query.omissions)
    if not rows and not omissions:
        if state.get("rendered") is not None:
            state["rendered"] = None
            _emit(runtime, "unknowns_rendered", count=0, critical=0, omitted=0)
        _store_render(runtime, projection=projection, task_id=task_id, scope=scope,
                      action_snapshot=action_snapshot, text=None, count=0,
                      critical=0, omitted=0)
        return None
    lines = [SIDECAR_HEADER + " host-owned; only a host receipt (read/grep/test result) closes a row, "
             "never a claim. CRITICAL rows block done."]
    critical = 0
    actions: Mapping[str, str] = state["actions"]  # type: ignore[assignment]
    for row in rows[:_MAX_ROWS]:
        is_critical = str(row.get("blocking_scope")) == scope
        critical += is_critical
        subject = row.get("subject") if isinstance(row.get("subject"), Mapping) else {}
        locator = subject.get("locator") or subject.get("entity_id") or "-"
        resolution = str(row.get("resolution") or "READ")
        action = actions.get(str(row.get("unknown_id"))) or f"{_ACTION.get(resolution, 'read')} {locator}"
        lines.append(f"- {'CRITICAL' if is_critical else 'advisory'} {row.get('predicate')} "
                     f"`{_clip(locator, 60)}`: {_clip(row.get('reason'))} -> {resolution}: {_clip(action, 120)}")
    omitted = len(omissions) + max(0, len(rows) - _MAX_ROWS)
    if omitted:
        lines.append(f"- (+{omitted} more open unknowns omitted by budget)")
    text = "\n".join(lines)
    if text != state.get("rendered"):
        state["rendered"] = text
        _emit(runtime, "unknowns_rendered", count=len(rows), critical=critical, omitted=omitted)
    _store_render(runtime, projection=projection, task_id=task_id, scope=scope,
                  action_snapshot=action_snapshot, text=text, count=len(rows),
                  critical=critical, omitted=omitted)
    return text


__all__ = ["ADVISORY_SCOPE", "SIDECAR_HEADER", "mode", "observe", "render"]

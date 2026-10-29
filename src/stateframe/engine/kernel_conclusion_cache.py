"""r44 conclusion-cache extension of the r43 settled-decision cache.

Own switch ``STATEFRAME_CONCLUSION_CACHE`` (default off).  It only has an
effect when the r43 ``STATEFRAME_DECISION_CACHE`` arm is also on; with this
switch off the r43 schema, state and render path are unchanged.

This module owns three host-side mechanisms:

* receipt normalization: real producer shapes (runtime test subprocess
  receipts with ``binding``, bolt receipts with ``workspace_revision``, and
  host-observed verification reuse) become answer-shaped rows whose
  *recorded* identity is never substituted by the *current* identity;
* premise facts: test-definition and environment identities the host knows,
  used as extra invalidation facets for interpretations;
* bounded receipt rows: every rendered field is capped so a receipt row has a
  fixed maximum size and can be budgeted with the conclusion records.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

MODE_ENV = "STATEFRAME_CONCLUSION_CACHE"
SCHEMA = "stateframe-conclusion-cache/1"
MAX_SCOPE_CHARS = 72
MAX_STATUS_CHARS = 32
MAX_IDENTITY_CHARS = 40
MAX_RECEIPT_ID_CHARS = 40
MAX_REUSE_NOTES = 16
PASS_STATUSES = frozenset({
    "PASS", "PASSED", "GREEN", "SUCCESS", "OK", "CERTIFIED",
    "REUSED", "REUSED_VERIFICATION",
})
_IDENTITY_KEYS = (
    "artifact", "product_revision", "revision", "digest",
    "test_harness_revision", "harness_revision", "environment", "environment_revision",
    "policy", "policy_identity", "policy_revision", "authorization_scope", "scope_identity",
)
_DISPLAY_IDENTITY_KEYS = ("artifact", "environment")


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").lower()
    if value in {"0", "off"}:
        return "off"
    if value in {"1", "on"}:
        return "on"
    raise ValueError(f"{MODE_ENV} must be 0/1 or off/on")


def enabled() -> bool:
    """The extension is live only on top of the r43 decision cache."""
    if mode() != "on":
        return False
    from kernel_decision_cache import mode as decision_cache_mode
    return decision_cache_mode() == "on"


def _digest(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def clip(value: object, limit: int) -> str:
    """One-line, length-capped rendering of a host field."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:max(0, limit - 3)] + "..."


# --------------------------------------------------------------------------- identities

def _binding_identity(value: object) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    result = {key: str(value[key]) for key in _IDENTITY_KEYS
              if value.get(key) not in (None, "")}
    return result or None


def identity_label(identity: object) -> str:
    """Short, deterministic text for a recorded/current identity."""
    if identity is None:
        return "UNKNOWN"
    if isinstance(identity, Mapping):
        kind = identity.get("kind")
        value = identity.get("value")
        if kind == "binding" and isinstance(value, Mapping):
            parts = []
            for key in _DISPLAY_IDENTITY_KEYS:
                if key in value:
                    raw = value[key].split(":", 1)[-1]
                    parts.append(f"{key[:3]}:{raw[:8]}")
            return clip(",".join(parts) or "UNKNOWN", MAX_IDENTITY_CHARS)
        if kind == "workspace_revision":
            return clip(f"ws-r{value}", MAX_IDENTITY_CHARS)
        return clip(value, MAX_IDENTITY_CHARS)
    return clip(identity, MAX_IDENTITY_CHARS)


def compare_identity(recorded: object, current: object) -> str:
    """CURRENT/STALE only for comparable identities; everything else UNKNOWN."""
    if not isinstance(recorded, Mapping) or not isinstance(current, Mapping):
        return "UNKNOWN"
    if recorded.get("kind") != current.get("kind"):
        return "UNKNOWN"
    old, new = recorded.get("value"), current.get("value")
    if old is None or new is None:
        return "UNKNOWN"
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        shared = [key for key in _IDENTITY_KEYS if key in old and key in new]
        if not shared:
            return "UNKNOWN"
        return "CURRENT" if all(old[key] == new[key] for key in shared) else "STALE"
    if isinstance(old, Mapping) or isinstance(new, Mapping):
        return "UNKNOWN"
    return "CURRENT" if str(old) == str(new) else "STALE"


def current_identities(*, binding: object = None, workspace_revision: object = None,
                       revision: object = None) -> dict[str, dict[str, object]]:
    """Host-supplied current identities, keyed by identity kind."""
    result: dict[str, dict[str, object]] = {}
    ident = _binding_identity(binding)
    if ident is not None:
        result["binding"] = {"kind": "binding", "value": ident}
    if type(workspace_revision) is int:
        result["workspace_revision"] = {"kind": "workspace_revision",
                                        "value": workspace_revision}
    if revision is not None and str(revision).strip():
        result["revision"] = {"kind": "revision", "value": str(revision)}
    return result


# --------------------------------------------------------------------------- receipts

def normalize_receipt(row: Mapping[str, object],
                      current: Mapping[str, object]) -> dict[str, object]:
    """Map one real host receipt shape to an answer-shaped row.

    The recorded identity comes only from the receipt itself; when it has none
    the row says UNKNOWN rather than borrowing the current identity.
    """
    passed = total = None
    executions: int | None = None
    reused = False
    if row.get("kind") == "verification_reuse":
        selectors = row.get("selectors")
        scope = (", ".join(str(item) for item in selectors)
                 if isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes))
                 and selectors else "declared checks")
        status = "PASSED"
        recorded_value = _binding_identity(row.get("recorded_binding"))
        recorded = {"kind": "binding", "value": recorded_value} if recorded_value else None
        kind = "binding"
        executions = 0
        reused = True
    elif "exit_code" in row and "selectors" in row:
        selectors = row.get("selectors")
        scope = (", ".join(str(item) for item in selectors)
                 if isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes))
                 and selectors else "declared checks")
        code = row.get("exit_code")
        status = "PASSED" if code == 0 else f"FAILED(exit={code})"
        recorded_value = _binding_identity(row.get("binding"))
        recorded = {"kind": "binding", "value": recorded_value} if recorded_value else None
        kind = "binding"
        executions = 1
    elif "workspace_revision" in row:
        scope = str(row.get("intent") or row.get("kind") or "receipt")
        status = str(row.get("result_class") or "UNKNOWN").upper()
        value = row.get("workspace_revision")
        recorded = ({"kind": "workspace_revision", "value": value}
                    if type(value) is int else None)
        kind = "workspace_revision"
        reused = status == "REUSED_VERIFICATION"
        if reused:
            executions = 0
    else:
        scope = next((str(row[key]) for key in ("suite", "selector", "target", "scope", "test")
                      if isinstance(row.get(key), str) and row[key].strip()), "UNKNOWN_SCOPE")
        status = next((str(row[key]).upper() for key in (
            "result_class", "result", "status", "outcome", "verdict")
            if row.get(key) is not None and str(row[key]).strip()), "UNKNOWN")
        value = row.get("revision")
        recorded = ({"kind": "revision", "value": str(value)}
                    if value is not None and str(value).strip() else None)
        kind = "revision"
        reused = bool(row.get("reused")) or status in {"REUSED", "REUSED_VERIFICATION"}
        if type(row.get("executions")) is int and row["executions"] >= 0:
            executions = row["executions"]
        elif reused:
            executions = 0
        if type(row.get("passed")) is int and type(row.get("total")) is int:
            passed, total = row["passed"], row["total"]
    current_identity = current.get(kind) if isinstance(current, Mapping) else None
    receipt_id = row.get("receipt_id") or row.get("prior_receipt_ref") or row.get("event_id")
    return {
        "scope": clip(scope, MAX_SCOPE_CHARS),
        "status": clip(status, MAX_STATUS_CHARS),
        "passed": passed, "total": total,
        "recorded": recorded, "current": current_identity,
        "binding": compare_identity(recorded, current_identity),
        "executions": executions, "reused": reused,
        "receipt_id": (clip(receipt_id, MAX_RECEIPT_ID_CHARS)
                       if receipt_id is not None and str(receipt_id).strip() else None),
    }


def receipt_line(row: Mapping[str, object]) -> str:
    """A single rendered receipt row; every variable field is capped."""
    status = str(row.get("status"))
    if (row.get("passed") is not None and row.get("total") is not None
            and status in PASS_STATUSES):
        answer = f"{row['passed']}/{row['total']} {row['scope']} passed"
    else:
        answer = f"{row['scope']} {status}"
    line = (f"- {answer}; recorded {identity_label(row.get('recorded'))}"
            f"; current {identity_label(row.get('current'))}"
            f"; binding {row.get('binding', 'UNKNOWN')}")
    if row.get("executions") is not None:
        line += f"; executions: {row['executions']}"
    if row.get("receipt_id"):
        line += f"; receipt: {row['receipt_id']}"
    if row.get("reused"):
        line += "; REUSED = no new sample"
    return line


def receipt_lines(receipts: object, current: object) -> list[str]:
    """Normalize host receipts (newest last) into capped rendered rows."""
    raw = receipts.values() if isinstance(receipts, Mapping) else (receipts or ())
    if isinstance(current, Mapping) and not any(
            isinstance(value, Mapping) and "kind" in value for value in current.values()):
        current = current_identities(binding=current)
    elif not isinstance(current, Mapping):
        current = current_identities(revision=current)
    return [receipt_line(normalize_receipt(row, current))
            for row in raw if isinstance(row, Mapping)]


# --------------------------------------------------------------------------- runtime hooks

def note_reuse(runtime: object, selectors: object, current_binding: object,
               prior_receipt_ref: object = None) -> None:
    """Record one host-observed reuse of a prior verification (no subprocess ran)."""
    if getattr(runtime, "conclusion_cache_mode", "off") != "on":
        return
    requested = [str(item) for item in selectors] if isinstance(
        selectors, Sequence) and not isinstance(selectors, (str, bytes)) else []
    prior = None
    for receipt in reversed(list(getattr(runtime, "test_receipts", ()) or ())):
        if not isinstance(receipt, Mapping) or receipt.get("exit_code") != 0:
            continue
        if prior_receipt_ref is not None and receipt.get("receipt_id") == prior_receipt_ref:
            prior = receipt
            break
        if prior_receipt_ref is None and set(requested) <= set(
                str(item) for item in receipt.get("selectors") or ()):
            prior = receipt
            break
    notes = getattr(runtime, "_conclusion_reuse_notes", None)
    if not isinstance(notes, list):
        notes = []
        runtime._conclusion_reuse_notes = notes
    note = {
        "kind": "verification_reuse",
        "selectors": requested,
        "recorded_binding": copy.deepcopy(prior.get("binding")) if prior else None,
        "reuse_binding": copy.deepcopy(current_binding),
        "prior_receipt_ref": (prior.get("receipt_id") if prior else prior_receipt_ref),
        "step": getattr(getattr(runtime, "owner", None), "current_step", None),
    }
    notes.append(note)
    del notes[:-MAX_REUSE_NOTES]
    # r44 F4 R4: a recorded reuse is a mechanism action -> journal it.
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit("conclusion_cache_reuse", producer="kernel_conclusion_cache",
             note=copy.deepcopy(note), prior_found=prior is not None, dispatch=False)


@dataclass(frozen=True, slots=True)
class ReceiptView:
    """Immutable render input with a semantic producer generation."""

    rows: tuple[Mapping[str, object], ...]
    current_revision: Mapping[str, object]
    generation: int
    consumed_cursor: int

    def as_kwargs(self) -> dict[str, object]:
        return {
            "receipts": self.rows,
            "current_revision": self.current_revision,
            "receipt_generation": self.generation,
            "receipt_cursor": self.consumed_cursor,
        }


class _ReceiptRows(tuple):
    """Deeply frozen runtime receipt rows with a stable snapshot identity."""

    def __new__(cls, rows: Iterable[Mapping[str, object]], snapshot_token: object):
        value = super().__new__(cls, rows)
        object.__setattr__(value, "snapshot_token", snapshot_token)
        return value

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("runtime receipt snapshots are immutable")


def _freeze_receipt_value(value: object) -> object:
    """Freeze JSON-shaped receipt containers so cached rows cannot drift."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_receipt_value(item)
                                 for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_receipt_value(item) for item in value)
    return copy.deepcopy(value)


def _receipt_source_edge(value: object) -> tuple[object, int, object, object] | None:
    """Describe an append-only runtime list without walking its receipt rows."""
    # Runtime reuse notes are initialized lazily. Before the first reuse, the
    # absent source is represented by this immutable empty tuple.
    if type(value) is tuple and not value:
        return value, 0, None, None
    if type(value) is not list:
        return None
    before = len(value)
    first = value[0] if before else None
    last = value[-1] if before else None
    after = len(value)
    if before != after:
        return None
    return value, after, first, last


def _same_receipt_sources(previous: object, current: object) -> bool:
    """Compare source list identity, length, and append boundaries in O(1)."""
    if not isinstance(previous, tuple) or not isinstance(current, tuple) \
            or len(previous) != len(current):
        return False
    return all(
        isinstance(old, tuple) and isinstance(new, tuple) and len(old) == len(new) == 4
        and old[0] is new[0] and old[1] == new[1]
        and old[2] is new[2] and old[3] is new[3]
        for old, new in zip(previous, current)
    )


def runtime_receipt_view(runtime: object) -> dict[str, object]:
    """Receipts + current identities for the settled-state render (switch on)."""
    owner = getattr(runtime, "owner", None)
    binding = None
    try:
        binding = runtime.binding()
    except Exception:  # noqa: BLE001 - an unreadable binding is an UNKNOWN identity
        binding = None
    workspace_revision = getattr(owner, "workspace_revision", None)
    current_revision = current_identities(
        binding=binding,
        workspace_revision=workspace_revision if type(workspace_revision) is int else 0)
    journal = getattr(runtime, "journal", None)
    generation = getattr(journal, "receipt_generation", None)
    has_journal_generation = type(generation) is int
    if not has_journal_generation:
        generation = int(getattr(runtime, "_conclusion_receipt_generation", 0) or 0) + 1
    receipt_cursor = None
    cursor_fn = getattr(journal, "receipt_cursor", None)
    if callable(cursor_fn):
        try:
            cursor = cursor_fn()
            receipt_cursor = int(getattr(cursor, "sequence"))
        except (AttributeError, TypeError, ValueError):
            receipt_cursor = None
    view_identity: object = None
    current_view = getattr(runtime, "current_view", None)
    if callable(current_view):
        try:
            view = current_view()
            view_identity = (getattr(view, "mutation_epoch", None),
                             getattr(view, "root_revision", None))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            view_identity = None

    revision_digest = _digest(current_revision)
    test_source = getattr(runtime, "test_receipts", ())
    owner_source = getattr(owner, "_receipts", ())
    reuse_source = getattr(runtime, "_conclusion_reuse_notes", ())
    source_state = tuple(_receipt_source_edge(value)
                         for value in (test_source, owner_source, reuse_source))
    append_only_snapshot = (
        has_journal_generation and type(receipt_cursor) is int
        and callable(getattr(journal, "cached_records", None))
        and all(edge is not None for edge in source_state)
    )
    cached = getattr(runtime, "_conclusion_receipt_view_cache", None)
    if append_only_snapshot and isinstance(cached, Mapping) \
            and cached.get("mode") == "append_only" \
            and _same_receipt_sources(cached.get("sources"), source_state) \
            and cached.get("revision") == revision_digest \
            and cached.get("view_identity") == view_identity:
        view = cached.get("view")
        if isinstance(view, ReceiptView):
            updated_view = ReceiptView(view.rows, view.current_revision,
                                       generation, receipt_cursor)
            setattr(runtime, "_conclusion_receipt_generation", generation)
            setattr(runtime, "_conclusion_receipt_view_cache", {
                "mode": "append_only", "sources": source_state,
                "revision": revision_digest, "view_identity": view_identity,
                "view": updated_view,
            })
            return updated_view.as_kwargs()

    tests = [row for row in (test_source or ()) if isinstance(row, Mapping)]
    rows: list[tuple[object, int, Mapping[str, object]]] = []
    for index, row in enumerate(tests):
        rows.append((row.get("sequence") or 0, index, row))
    # Bolt test receipts duplicate subprocess receipts when both exist; they are
    # the only verification record on hosts without the runtime subprocess path.
    if not tests:
        for index, row in enumerate(owner_source or ()):
            if isinstance(row, Mapping) and row.get("kind") == "test":
                rows.append((0, index, row))
    ordered = [row for _seq, _index, row in rows]
    ordered.extend(copy.deepcopy(list(reuse_source or ())))
    if receipt_cursor is None:
        receipt_cursor = len(ordered)
    view_key = (generation, receipt_cursor, revision_digest, _digest(ordered), view_identity)
    if isinstance(cached, Mapping) and cached.get("mode") == "digest" \
            and cached.get("key") == view_key:
        view = cached.get("view")
        if isinstance(view, ReceiptView):
            return view.as_kwargs()

    if append_only_snapshot:
        frozen_rows = _ReceiptRows(
            (MappingProxyType({key: _freeze_receipt_value(value)
                               for key, value in dict(row).items()})
             for row in ordered if isinstance(row, Mapping)),
            object(),
        )
    else:
        frozen_rows = tuple(MappingProxyType(copy.deepcopy(dict(row)))
                            for row in ordered if isinstance(row, Mapping))
    rows_view = frozen_rows if isinstance(frozen_rows, _ReceiptRows) else tuple(frozen_rows)
    view = ReceiptView(rows_view, MappingProxyType(copy.deepcopy(current_revision)),
                       generation, receipt_cursor)
    setattr(runtime, "_conclusion_receipt_generation", generation)
    if append_only_snapshot:
        setattr(runtime, "_conclusion_receipt_view_cache", {
            "mode": "append_only", "sources": source_state,
            "revision": revision_digest, "view_identity": view_identity,
            "view": view,
        })
    else:
        setattr(runtime, "_conclusion_receipt_view_cache", {
            "mode": "digest", "key": view_key, "view": view,
        })
    return view.as_kwargs()


def premise_context(runtime: object) -> dict[str, object]:
    """Host-known premise identities interpretations may silently rely on."""
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", {})
    row = row if isinstance(row, Mapping) else {}
    tests = {key: row[key] for key in (
        "public_test_source", "public_test_src", "test_source", "test_src", "f2p_ids")
        if row.get(key) is not None}
    f2p = getattr(owner, "f2p_ids", None)
    if isinstance(f2p, Sequence) and not isinstance(f2p, (str, bytes)):
        tests["declared_checks"] = [str(item) for item in f2p]
    journal = getattr(runtime, "journal", None)
    manifest = getattr(journal, "manifest", None)
    environment = (manifest.get("environment_revision")
                   if isinstance(manifest, Mapping) else None)
    return {
        "test_definition": _digest(tests),
        "environment": (str(environment) if environment is not None else None),
    }


__all__ = [
    "MODE_ENV", "clip", "compare_identity", "current_identities", "enabled",
    "ReceiptView", "SCHEMA", "identity_label", "mode", "normalize_receipt", "note_reuse", "premise_context",
    "receipt_line", "receipt_lines", "runtime_receipt_view",
]

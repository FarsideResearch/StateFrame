"""Opt-in deduplication for the r42 UNVERIFIED/checklist presentation.

The open rows themselves remain owned by the live-counterexample and coverage
reducers.  This module stores only model-facing presentation bookkeeping:

* each open item is identified by ``(kind, id)`` AND fingerprinted by its
  semantic row content, so an item whose reason/target/suggestion changes under
  the same ID is a CHANGED item and is rendered in full again;
* an item that leaves the open set forgets its description/checked state, so a
  closed-then-reopened item is rendered in full again;
* an item is marked ``checked by you at turn N`` only after a successful read whose
  executor-resolved path equals one of its named files.  Probes, prose (intent,
  question, detail, file contents) and ID mentions never count, and a row without a
  bound file can never be marked checked.

``STATEFRAME_NOTICE_DEDUP=off`` is deliberately a no-op at the runtime hooks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath


MODE_ENV = "STATEFRAME_NOTICE_DEDUP"
SCHEMA = "stateframe-notice-dedup/2"

_MISSING = object()
# Only the real read tool resolves a path through the executor (receipt_target_path).
_READ_KINDS = frozenset({"read"})
_READ_OK = frozenset({"OK"})
# Keys that never describe an item (bookkeeping that may change without the item changing).
_VOLATILE_KEYS = frozenset({
    "turn", "step", "ts", "timestamp", "generated_at", "receipt_id", "event_id",
    "request_id", "run_id", "captured_at",
})
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_notice_dedup_mode", None)
    return (str(configured).strip().lower() == "on"
            if configured is not None else mode() == "on")


def _text(value: object) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _normal_path(value: object, runtime: object | None = None) -> str:
    """Root-relative POSIX path; exact identity only (no suffix/basename matching)."""
    text = _text(value).replace("\\", "/")
    if not text:
        return ""
    root = getattr(getattr(runtime, "owner", None), "root", None)
    path = Path(text)
    if path.is_absolute() and root is not None:
        try:
            text = path.resolve().relative_to(Path(root).resolve()).as_posix()
        except (OSError, ValueError, RuntimeError):
            return path.as_posix()
    parts = [part for part in PurePosixPath(text).parts if part not in ("", ".")]
    return "/".join(parts)


def _digest(value: object) -> str:
    try:
        body = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str).encode("utf-8")
    except (TypeError, ValueError):
        body = repr(value).encode("utf-8", "replace")
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _revision(runtime: object) -> str:
    """The artifact revision shared with the live window and done-challenge reducers."""
    for name in ("product_revision", "current_revision", "revision"):
        value = getattr(runtime, name, None)
        if value not in (None, ""):
            return str(value)
    try:
        from kernel_challenge_budget import artifact_revision
        return str(artifact_revision(runtime))
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        return "unknown"


def _turn(runtime: object, explicit: object = None) -> int:
    value = explicit
    if value in (None, ""):
        value = getattr(getattr(runtime, "owner", None), "current_step", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _state(runtime: object) -> dict[str, object]:
    state = getattr(runtime, "_notice_dedup_state", None)
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        state = {"schema": SCHEMA, "revision": None, "scopes": {}, "described": {},
                 "checked": {}, "items": {}, "pending": []}
        setattr(runtime, "_notice_dedup_state", state)
    return state


def _begin_revision(state: dict[str, object], revision: str) -> None:
    if state.get("revision") == revision:
        return
    state.update({"revision": revision, "scopes": {}, "described": {}, "checked": {},
                  "items": {}})


def _as_values(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        values = value.keys()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        values = (value,)
    return tuple(_text(item) for item in values if _text(item))


def _row_targets(row: Mapping[str, object], runtime: object | None = None) -> tuple[str, ...]:
    paths: list[str] = []
    for key in ("target_files", "files", "target_paths", "paths", "file", "target_file",
                "target_path", "path", "resolved_path"):
        for value in _as_values(row.get(key)):
            path = _normal_path(value, runtime)
            if path and path.upper() != "UNKNOWN" and path not in paths:
                paths.append(path)
    return tuple(paths)


def _row_symbols(row: Mapping[str, object]) -> tuple[str, ...]:
    """Bound code symbols only (a layer/clause name is prose, not a symbol)."""
    values: list[str] = []
    for key in ("target_symbol", "symbol", "entry_point", "function"):
        for value in _as_values(row.get(key)):
            if _IDENT.fullmatch(value) and len(value) >= 3 and value not in values:
                values.append(value)
    return tuple(values)


def _row_id(row: Mapping[str, object], *, prefix: str, index: int) -> str:
    for name in ("check_id", "obligation_id", "clause_id", "id", "layer"):
        if row.get(name) not in (None, "", [], {}):
            return _text(row[name])
    clause = _text(row.get("clause") or row.get("quoted_clause") or row.get("text"))
    return (f"{prefix}{hashlib.sha256(clause.encode('utf-8')).hexdigest()[:12]}"
            if clause else f"{prefix}{index + 1}")


def _fingerprint(row: Mapping[str, object]) -> str:
    return _digest({str(key): value for key, value in row.items()
                    if str(key) not in _VOLATILE_KEYS})


def _unverified_rows(runtime: object) -> list[dict[str, object]]:
    """The live reducer's rows (frame=None, exactly as ``window_notice``), enriched with the
    disclosed witness row of the same ID so bound files/symbols are known."""
    try:
        from kernel_live_counterexample_challenge import unverified_rows
        values = unverified_rows(runtime)
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        return []
    raw_by_id: dict[str, Mapping[str, object]] = {}
    disclosed = getattr(runtime, "_fgd_disclosed_open", None)
    if isinstance(disclosed, Mapping):
        for raw in disclosed.get("rows", ()) or ():
            if isinstance(raw, Mapping):
                for name in ("check_id", "obligation_id", "id"):
                    ident = _text(raw.get(name))
                    if ident:
                        raw_by_id.setdefault(ident, raw)
    result: list[dict[str, object]] = []
    for value in values:
        if isinstance(value, Mapping):
            row = dict(raw_by_id.get(_text(value.get("check_id")), {}))
            row.update(dict(value))
            result.append(row)
    return result


def _coverage_payload(runtime: object, frame: Mapping[str, object] | None) -> Mapping[str, object] | None:
    """Same source as the baseline checklist renderer (coverage-reminders mode only)."""
    if getattr(runtime, "stateframe_coverage_reminders_mode", "off") != "on":
        return None
    payload = frame.get("coverage_reminders") if isinstance(frame, Mapping) else None
    if not isinstance(payload, Mapping):
        payload = getattr(runtime, "_coverage_reminders_state", None)
    return payload if isinstance(payload, Mapping) else None


def _items(runtime: object, unverified: Sequence[Mapping[str, object]],
           coverage: Mapping[str, object] | None) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []

    def add(kind: str, key_id: str, ident: str, row: Mapping[str, object]) -> None:
        result.append({"key": (kind, key_id), "kind": kind, "id": ident,
                       "row": copy.deepcopy(dict(row)), "fingerprint": _fingerprint(row),
                       "targets": _row_targets(row, runtime), "symbols": _row_symbols(row)})

    for index, row in enumerate(unverified):
        ident = _row_id(row, prefix="U", index=index)
        add("unverified", ident, ident, row)
    if isinstance(coverage, Mapping):
        for index, raw in enumerate(coverage.get("open_rows", ()) or ()):
            if isinstance(raw, Mapping):
                ident = _row_id(raw, prefix="row:", index=index)
                add("checklist", "row:" + ident, ident, raw)
        for index, raw in enumerate(coverage.get("open_review_reminders", ()) or ()):
            if isinstance(raw, Mapping):
                ident = _row_id(raw, prefix="reminder:", index=index)
                add("checklist", "reminder:" + ident, ident, raw)
    return result


def _signature(items: Sequence[Mapping[str, object]]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((*item["key"], item["fingerprint"]) for item in items))


def _inspection(runtime: object, receipt: Mapping[str, object], material: object
                ) -> tuple[str, set[str], str] | None:
    """(kind, host-resolved paths, unused) for a successful named read, else None."""
    kind = _text(receipt.get("kind") or receipt.get("tool")).lower()
    status = _text(receipt.get("result_class") or receipt.get("status")).upper()
    if kind in _READ_KINDS and status in _READ_OK:
        paths: set[str] = set()
        evidence = receipt.get("read_evidence")
        sources = [evidence] if isinstance(evidence, Mapping) else []
        sources.append(receipt)
        for source in sources:
            # Executor-resolved fields only (kernel_tool_paths.receipt_target_path); never
            # the model's raw argument, intent or detail.
            for name in ("resolved_path", "resolved_target_path", "target_path", "path"):
                if source is receipt and name == "path":
                    continue
                path = _normal_path(source.get(name), runtime)
                if path:
                    paths.add(path)
        return ("read", paths, "") if paths else None
    # Probes never mark an item checked (review2 finding 1): probe source text does not show
    # what was executed or which file/symbol it bound to, and no host evidence of that exists
    # yet.  Re-enable only on a host-produced inspected-target record.
    return None


def _matches(item: Mapping[str, object], kind: str, paths: set[str], code: str) -> bool:
    if kind == "read":
        return bool(set(item.get("targets", ()) or ()) & paths)
    return False


def observe_action(runtime: object, receipt: object, material: object = "") -> bool:
    """Mark current open items checked after a successful read of their named file."""
    if not enabled(runtime) or not isinstance(receipt, Mapping):
        return False
    inspection = _inspection(runtime, receipt, material)
    if inspection is None:
        return False
    kind, paths, code = inspection
    state = _state(runtime)
    _begin_revision(state, _revision(runtime))
    checked = state["checked"]
    turn = _turn(runtime, receipt.get("turn"))
    changed = False
    for key, item in tuple(state.get("items", {}).items()):
        current = checked.get(key)
        if current is not None and current[1] == item["fingerprint"]:
            continue
        if _matches(item, kind, paths, code):
            checked[key] = (turn, item["fingerprint"])
            changed = True
    return changed


def _compact(items: Sequence[Mapping[str, object]], state: Mapping[str, object], *, phrase: str) -> str:
    if not items:
        return ""
    by_kind: dict[str, list[Mapping[str, object]]] = {"unverified": [], "checklist": []}
    for item in items:
        by_kind.setdefault(str(item.get("kind")), []).append(item)
    checked = state.get("checked", {})
    parts: list[str] = []
    for kind in ("unverified", "checklist"):
        values = by_kind.get(kind, [])
        if not values:
            continue
        ids = []
        for item in values:
            ident = _text(item.get("id"))
            mark = checked.get(item.get("key"))
            ids.append(f"{ident} (checked by you at turn {mark[0]})"
                       if mark and mark[1] == item["fingerprint"] else ident)
        parts.append(f"{kind}: {len(values)} open ({phrase}); ids: {', '.join(ids)}")
    return "; ".join(parts)


def _filtered_coverage(payload: Mapping[str, object], keys: set[tuple[str, str]]) -> dict[str, object]:
    result = copy.deepcopy(dict(payload))
    for field, prefix in (("open_rows", "row:"), ("open_review_reminders", "reminder:")):
        kept = []
        for index, raw in enumerate(payload.get(field, ()) or ()):
            if isinstance(raw, Mapping):
                ident = _row_id(raw, prefix=prefix, index=index)
                if ("checklist", prefix + ident) in keys:
                    kept.append(copy.deepcopy(dict(raw)))
        result[field] = kept
    return result


def _render_full(items: Sequence[Mapping[str, object]], full_keys: set[tuple[str, str]],
                 coverage: Mapping[str, object] | None, runtime: object | None = None) -> list[str]:
    parts: list[str] = []
    unverified = [item["row"] for item in items
                  if item["kind"] == "unverified" and item["key"] in full_keys]
    if unverified:
        from kernel_live_counterexample_challenge import WINDOW_NOTICE_SUFFIX, budget_applies, unverified_detail
        _suffix = f" {WINDOW_NOTICE_SUFFIX}" if budget_applies(unverified, runtime) else ""  # M1
        parts.append(f"{unverified_detail(unverified, runtime)}{_suffix}")
    checklist_keys = {item["key"] for item in items
                      if item["kind"] == "checklist" and item["key"] in full_keys}
    if checklist_keys and isinstance(coverage, Mapping):
        from kernel_coverage_reminders import render_checklist
        parts.append(render_checklist(_filtered_coverage(coverage, checklist_keys)))
    return [part for part in parts if part]


def render_notice_bundle(runtime: object, frame: Mapping[str, object] | None = None, *,
                         unverified_rows: object = _MISSING,
                         coverage_payload: object = _MISSING,
                         revision: object = None, turn: object = None,
                         scope: str = "bundle",
                         kinds: Sequence[str] = ("unverified", "checklist")) -> str:
    """Render NEW/CHANGED/undescribed-unchecked items in full, everything else as one
    compact ID line.  Never touches the reducers' rows.

    The text is a pure function of the committed state, so every render inside one
    ``prepare`` (previews, hybrid crossing, epoch roll) produces the same text.  The
    "described once" budget is committed only when that text actually reached the
    provider request (``prepare_with_commit``); a standalone call commits at once."""
    if not enabled(runtime):
        return ""
    if unverified_rows is _MISSING:
        unverified = _unverified_rows(runtime)
    elif isinstance(unverified_rows, Sequence):
        unverified = [dict(item) for item in unverified_rows if isinstance(item, Mapping)]
    else:
        unverified = []
    if coverage_payload is _MISSING:
        coverage = _coverage_payload(runtime, frame)
    else:
        coverage = coverage_payload if isinstance(coverage_payload, Mapping) else None
    owned = frozenset(kinds)
    current_turn = _turn(runtime, turn)
    state = _state(runtime)
    _begin_revision(state, str(revision) if revision not in (None, "") else _revision(runtime))
    items = [item for item in _items(runtime, unverified, coverage) if item["kind"] in owned]
    current = {item["key"]: item for item in items}
    described: dict = state["described"]
    checked: dict = state["checked"]
    # Host truths, applied immediately: a closed item forgets its budget/marker (a reopened
    # item is new again) and a CHANGED row (same ID, other content) loses both.
    for store in (described, checked):
        for key in [key for key in store if key[0] in owned and key not in current]:
            del store[key]
    for key, item in current.items():
        if key in described and described[key] != item["fingerprint"]:
            del described[key]
        if key in checked and checked[key][1] != item["fingerprint"]:
            del checked[key]
    state["items"] = {**{key: value for key, value in state.get("items", {}).items()
                         if key[0] not in owned}, **current}
    scopes: dict = state.setdefault("scopes", {})
    view = scopes.get(scope) or {}
    signature = _signature(items)
    changed = view.get("signature") is None or tuple(view["signature"]) != signature
    change_turn = current_turn if changed else view.get("change_turn")
    # Full text is owed until THIS fingerprint was delivered; a "checked" marker never
    # substitutes for delivery (review2 finding 2: preview -> read -> compact-only).
    full_keys = {key for key in current if key not in described}
    parts = _render_full(items, full_keys, coverage, runtime)
    remainder = [item for item in items if item["key"] not in full_keys]
    phrase = (f"changed at turn {current_turn}" if changed
              else f"unchanged since turn {change_turn}")
    compact = _compact(remainder, state, phrase=phrase)
    text = "\n".join(part for part in (*parts, compact) if part)
    pending = {"text": text, "scope": scope, "signature": signature, "change_turn": change_turn,
               "described": {key: current[key]["fingerprint"] for key in full_keys}}
    if getattr(runtime, "_notice_dedup_in_request", False):
        state.setdefault("pending", []).append(pending)
    else:
        _commit(state, pending)
    return text


def _commit(state: dict[str, object], pending: Mapping[str, object]) -> None:
    state.setdefault("scopes", {})[pending["scope"]] = {
        "signature": pending["signature"], "change_turn": pending["change_turn"]}
    state["described"].update(pending["described"])


def _delivered_text(messages: object) -> str:
    parts: list[str] = []
    for message in messages or ():
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, str):
            parts.append(content)
        elif content is not None:
            parts.append(json.dumps(content, ensure_ascii=False, default=str))
    return "\n\x00\n".join(parts)


def prepare_with_commit(runtime: object, prepare, messages, *, uncapped, tools,
                        provider_request: bool = True,
                        request_host_text_preparer: Callable[..., object] | None = None):
    """Run one real ``prepare``; commit a render's description budget only if its text is
    in the provider request that ``prepare`` returned (a dropped or preview render is not
    a delivery)."""
    state = _state(runtime)
    state["pending"] = []
    setattr(runtime, "_notice_dedup_in_request", True)
    try:
        if request_host_text_preparer is None:
            prepared = prepare(runtime, messages, uncapped=uncapped, tools=tools,
                               provider_request=provider_request)
        else:
            prepared = prepare(
                runtime, messages, uncapped=uncapped, tools=tools,
                provider_request=provider_request,
                request_host_text_preparer=request_host_text_preparer,
            )
    finally:
        setattr(runtime, "_notice_dedup_in_request", False)
    pending = list(state.get("pending") or ())
    state["pending"] = []
    if provider_request and pending:
        delivered = _delivered_text(getattr(prepared, "messages", ()) or ())
        committed_scopes: set[str] = set()
        # the last delivered render of each scope wins (all renders of one request agree)
        for entry in reversed(pending):
            if entry["scope"] in committed_scopes:
                continue
            # an empty render (nothing open) has nothing to deliver
            if not entry["text"] or entry["text"] in delivered:
                _commit(state, entry)
                committed_scopes.add(entry["scope"])
    return prepared


def render_f4_notices(runtime: object, frame: Mapping[str, object] | None = None) -> str:
    """Deduped F4 window + checklist.  The window predicate and rows are read exactly as the
    baseline ``window_notice(runtime)`` reads them (frame=None)."""
    rows: object = ()
    if getattr(runtime, "stateframe_live_ce_challenge_mode", "off") == "on":
        try:
            from kernel_live_counterexample_challenge import unverified_window_open
            if unverified_window_open(runtime):
                rows = _unverified_rows(runtime)
        except (ImportError, AttributeError, TypeError, ValueError, OSError):
            rows = ()
    kinds = (("unverified", "checklist")
             if getattr(runtime, "stateframe_coverage_reminders_mode", "off") == "on"
             else ("unverified",))
    return render_notice_bundle(runtime, frame, unverified_rows=rows, scope="f4", kinds=kinds)


def render_coverage_notice(runtime: object, frame: Mapping[str, object] | None = None) -> str:
    """Deduped checklist for non-F4 frame/history presentations."""
    return render_notice_bundle(runtime, frame, unverified_rows=(), scope="coverage",
                                kinds=("checklist",))


def state_snapshot(runtime: object) -> dict[str, object]:
    """JSON-friendly diagnostic view without exposing mutable state."""
    state = _state(runtime)
    return {
        "schema": SCHEMA,
        "revision": state.get("revision"),
        "scopes": {name: {"signature": [list(item) for item in view.get("signature") or ()],
                          "change_turn": view.get("change_turn")}
                   for name, view in (state.get("scopes") or {}).items()},
        "described": sorted("/".join(key) for key in state.get("described", {})),
        "checked": {"/".join(key): value[0] for key, value in state.get("checked", {}).items()},
    }


__all__ = [
    "MODE_ENV", "SCHEMA", "enabled", "mode", "observe_action", "prepare_with_commit",
    "render_coverage_notice",
    "render_f4_notices", "render_notice_bundle", "state_snapshot",
]

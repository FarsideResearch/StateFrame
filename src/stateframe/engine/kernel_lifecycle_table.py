"""Q4 (E5/E7): host lifecycle state table for a stateful ticket.

Given the Q2 owner fields and the ticket clauses, render the states (the
null / non-null lattice of the owner fields, capped), the events (one per
ticket clause with its text as provenance, a sibling task row's clause when it
names the same owner, and the implicit ``success`` of each producer helper),
and the parent-behaviour column (``no transition implemented`` when Q2 counted
0 writers, else the writer sites).  The E7 row ``(S_bound, unchanged) ->
UNDECIDED`` is always appended.  The next-state column is never filled: that
is the model's decision.

Trigger: Q2 produced >= 1 owner field AND the ticket has >= 2 clauses or a
lifecycle word (stale / idempotent / no-op / unchanged / regenerate / failure /
retry / renew).

``STATEFRAME_LIFECYCLE_TABLE`` (default ``on``).
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence

ENV = "STATEFRAME_LIFECYCLE_TABLE"
HEADER = "[LIFECYCLE STATE TABLE"
LIFECYCLE_WORDS = re.compile(
    r"\b(?:stale|idempotent|no-?op|unchanged|regenerat\w*|failure|fails?|failed|retr(?:y|ies)"
    r"|renew\w*)\b", re.IGNORECASE)
_MAX_STATE_FIELDS = 2
_MAX_EVENTS = 8
_MIN_CLAUSE_WORDS = 3
_INVENTORY_UNSET = object()


def mode(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    value = str(values.get(ENV, "on") or "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{ENV} must be on or off")
    return value


def clauses(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;!?])\s+|\n+", str(text or ""))
    result = []
    for part in parts:
        clause = part.strip().rstrip(".;!?").strip()
        if len(re.findall(r"[A-Za-z0-9_]+", clause)) >= _MIN_CLAUSE_WORDS:
            result.append(clause)
    return list(dict.fromkeys(result))


def triggered(ticket: str, fields: Sequence[object]) -> bool:
    return bool(fields) and (len(clauses(ticket)) >= 2 or bool(LIFECYCLE_WORDS.search(ticket)))


def _names_owner(text: str, names: Sequence[str]) -> bool:
    folded = text.casefold().replace(" ", "_")
    return any(name.casefold() in folded for name in names)


def build_table(inventory: object, ticket: str, *,
                sibling_clauses: Sequence[tuple[str, str]] = ()) -> dict[str, object] | None:
    """Return the table rows, or ``None`` when the trigger does not hold."""
    fields = list(getattr(inventory, "fields", ()) or ())
    if not triggered(ticket, fields):
        return None
    state_fields = fields[:_MAX_STATE_FIELDS]
    names = [str(item.name) for item in state_fields]
    states = [
        {"id": "S0", "label": "unbound", "values": {name: "None" for name in names}},
        {"id": "S1", "label": "bound", "values": {name: "set" for name in names}},
    ]
    events: list[dict[str, str]] = []
    for clause in clauses(ticket):
        events.append({"text": clause, "source": "ticket clause"})
    candidates = list(getattr(inventory, "candidates", ()) or ()) + names
    for source, text in sibling_clauses:
        for clause in clauses(text):
            if _names_owner(clause, candidates) or _names_owner(clause, [c.replace("_", " ") for c in candidates]):
                events.append({"text": clause, "source": f"sibling row {source}"})
    for helper in getattr(inventory, "helpers", ()) or ():
        events.append({"text": f"{helper.name} succeeds", "source":
                       f"implicit success of {helper.path}:{helper.line}"})
    events = events[:_MAX_EVENTS]
    for index, event in enumerate(events, 1):
        event["id"] = f"E{index}"
    writers = [(item.name, site) for item in fields for site in getattr(item, "writers", ())]
    complete = bool(getattr(inventory, "complete", False))
    parent = ("no transition implemented (0 product writers of the owner fields)" if complete and not writers
              else "writers: " + "; ".join(f"{site.path}:{site.line}" for _, site in writers[:6])
              if complete else "UNKNOWN: owner scan incomplete; zero writers is not global absence")
    return {"fields": [f"{item.path}:{item.line}" for item in state_fields], "names": names,
            "states": states, "events": events, "parent": parent,
            "completeness": "KNOWN" if complete else "UNKNOWN",
            "dependency_revision": str(getattr(inventory, "dependency_revision", "UNKNOWN")),
            "parser_capabilities": dict(getattr(inventory, "parser_capabilities", {}) or {}),
            "identities": {key: dict(value) for key, value in
                           (getattr(inventory, "identities", {}) or {}).items()},
            "noop_cell": "(S1, unchanged) -> UNDECIDED: what must a no-op still refresh?"}


def render(table: Mapping[str, object]) -> str:
    names = list(table.get("names", ()))
    lines = [f"{HEADER} | host-derived from {', '.join(names)} "
             f"({', '.join(table.get('fields', ()))}) and the ticket clauses]"]
    for state in table.get("states", ()):
        values = ", ".join(f"{key}={value}" for key, value in state["values"].items())
        lines.append(f"state {state['id']}  {state['label']}: {values}")
    for event in table.get("events", ()):
        lines.append(f"event {event['id']}  {event['text']}   <- {event['source']}")
    lines.append(f"parent behaviour: {table.get('parent')}")
    lines.append(f"cell {table.get('noop_cell')}")
    lines.append("Each (state, event) cell needs a decided next state; the host does not fill "
                 "the next-state column.")
    return "\n".join(lines)


def runtime_table(runtime: object, *, inventory: object = _INVENTORY_UNSET) -> str | None:
    """Rendered table for the runtime (cached), built on the Q2 inventory."""
    block = None
    cache_key: tuple[object, ...] = ("off",)
    if mode() == "on":
        from kernel_request_source_window import _ticket_text
        if inventory is _INVENTORY_UNSET:
            from kernel_state_owner_inventory import runtime_inventory
            inventory = runtime_inventory(runtime)
        cache_key = (
            "on",
            _ticket_text(runtime),
            inventory.cache_key if inventory is not None else "UNKNOWN",
        )
        cached = getattr(runtime, "_lifecycle_table_block", False)
        if cached is not False and getattr(runtime, "_lifecycle_table_cache_key", None) == cache_key:
            return cached
        if inventory is not None:
            row = getattr(getattr(runtime, "owner", None), "task_row", {})
            siblings: list[tuple[str, str]] = []
            raw = row.get("sibling_tickets", ()) if isinstance(row, Mapping) else ()
            for index, item in enumerate(raw if isinstance(raw, Sequence) and not isinstance(raw, str) else ()):
                if isinstance(item, Mapping):
                    siblings.append((str(item.get("id") or index + 1), str(item.get("text") or "")))
                elif isinstance(item, str):
                    siblings.append((str(index + 1), item))
            table = build_table(inventory, _ticket_text(runtime), sibling_clauses=siblings)
            block = render(table) if table else None
    elif getattr(runtime, "_lifecycle_table_cache_key", None) == cache_key:
        return getattr(runtime, "_lifecycle_table_block", None)
    try:
        setattr(runtime, "_lifecycle_table_block", block)
        setattr(runtime, "_lifecycle_table_cache_key", cache_key)
    except (AttributeError, TypeError):
        pass
    return block


__all__ = ["ENV", "HEADER", "LIFECYCLE_WORDS", "build_table", "clauses", "mode", "render",
           "runtime_table", "triggered"]

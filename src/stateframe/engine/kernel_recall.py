"""Lossless run recall with a bounded, relevance-ranked frame view.

The run store is intentionally boring: observations are appended once and are
never removed to make room for a prompt.  Rendering is the lossy boundary.  A
compact index lets the model name an old item, while ``recall(id)`` requests
that item for the next frame.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field

from frame_selector import OMISSION_SCHEMA


SCHEMA = "stateframe-recall/1"
MAX_GIST_CHARS = 180
MAX_INDEX_PAGE_SIZE = 16
DEFAULT_INDEX_PAGE_SIZE = 6
DEFAULT_VIEW_ITEMS = 8
DEFAULT_VIEW_TOKENS = 720
MAX_VIEW_TOKENS = 1800
DEFAULT_MAX_OVERFLOW_STREAK = 3
MAX_SELECTOR_EVENTS = 256
LINK_FIELDS = ("requirements", "obligations", "files", "symbols", "hypotheses", "stages")
_PATH_KEYS = {"path", "file", "filepath", "file_path", "source_path", "target_path", "location"}
_SYMBOL_KEYS = {"symbol", "symbols", "fn", "function", "function_name", "entry_point"}
_REQUIREMENT_KEYS = {
    "requirement", "requirements", "requirement_id", "requirement_ids", "requirement_ref",
    "requirement_refs", "ticket", "ticket_refs", "clause_id", "clause_ids",
}
_OBLIGATION_KEYS = {
    "obligation", "obligations", "obligation_id", "obligation_ids", "focus_obligation",
    "open_requirement_ids", "not_done_because", "blocking_requirement", "blocker_ids",
}
_HYPOTHESIS_KEYS = {
    "hypothesis", "hypotheses", "hypothesis_id", "hypothesis_ids", "open_hypothesis",
    "falsified_hypothesis", "falsified_hypotheses", "root_cause_hypothesis",
}
_STAGE_KEYS = {"stage", "idealpath_stage", "idealpath", "phase", "current_stage"}
_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]{0,127}$")


class RecallError(ValueError):
    """A malformed recall operation or an unhealthy render run."""


class RecallHealthError(RecallError):
    """RenderOverflow persisted beyond the configured consecutive-step limit."""


def _text(value: object, limit: int = MAX_GIST_CHARS) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        result: list[str] = []
        for nested in value.values():
            result.extend(_values(nested))
        return tuple(result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for nested in value:
            result.extend(_values(nested))
        return tuple(result)
    text = str(value).strip()
    return (text,) if text else ()


def _normalise(values: Sequence[str], *, path: bool = False) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        item = value.replace("\\", "/").strip() if path else " ".join(value.split())
        if item and item not in result:
            result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class RecallLinks:
    """Typed overlap keys used by relevance selection."""

    requirements: tuple[str, ...] = ()
    obligations: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    stages: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value: object) -> "RecallLinks":
        if isinstance(value, RecallLinks):
            return value
        if not isinstance(value, Mapping):
            return cls()
        return cls(
            requirements=_normalise(_values(value.get("requirements"))),
            obligations=_normalise(_values(value.get("obligations"))),
            files=_normalise(_values(value.get("files")), path=True),
            symbols=_normalise(_values(value.get("symbols"))),
            hypotheses=_normalise(_values(value.get("hypotheses"))),
            stages=_normalise(_values(value.get("stages"))),
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {field: list(getattr(self, field)) for field in LINK_FIELDS}

    def is_empty(self) -> bool:
        return not any(getattr(self, field) for field in LINK_FIELDS)


def links_from(value: object, *, depth: int = 0) -> RecallLinks:
    """Extract links from host-shaped records without trusting model prose."""
    if depth > 3 or not isinstance(value, Mapping):
        return RecallLinks()
    values = {field: [] for field in LINK_FIELDS}
    for key, raw in value.items():
        name = str(key).lower()
        if name in _PATH_KEYS:
            values["files"].extend(_values(raw))
        elif name in _SYMBOL_KEYS:
            values["symbols"].extend(_values(raw))
        elif name in _REQUIREMENT_KEYS:
            values["requirements"].extend(_values(raw))
        elif name in _OBLIGATION_KEYS:
            values["obligations"].extend(_values(raw))
        elif name in _HYPOTHESIS_KEYS:
            values["hypotheses"].extend(_values(raw))
        elif name in _STAGE_KEYS:
            values["stages"].extend(_values(raw))
        elif isinstance(raw, Mapping):
            nested = links_from(raw, depth=depth + 1)
            for field in LINK_FIELDS:
                values[field].extend(getattr(nested, field))
    return RecallLinks(
        requirements=_normalise(values["requirements"]),
        obligations=_normalise(values["obligations"]),
        files=_normalise(values["files"], path=True),
        symbols=_normalise(values["symbols"]),
        hypotheses=_normalise(values["hypotheses"]),
        stages=_normalise(values["stages"]),
    )


def _merge_links(left: RecallLinks, right: RecallLinks) -> RecallLinks:
    return RecallLinks(**{
        field: _normalise([*getattr(left, field), *getattr(right, field)], path=field == "files")
        for field in LINK_FIELDS
    })


def _gist(payload: object) -> str:
    if isinstance(payload, Mapping):
        for key in ("gist", "summary", "detail", "text", "claim", "reason", "observation"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return _text(value)
        parts = []
        for key in ("outcome", "status", "result", "lifecycle", "target", "path"):
            value = payload.get(key)
            if value not in (None, "", (), []):
                parts.append(f"{key}={_text(value, 48)}")
        if parts:
            return _text(" ".join(parts))
    try:
        return _text(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return _text(repr(payload))


def _digest(value: object) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        encoded = repr(value)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _status_resolved(status: object, payload: object) -> bool:
    if isinstance(payload, Mapping) and payload.get("resolved") is True:
        return True
    return str(status or "").upper() in {"RESOLVED", "SATISFIED", "CLOSED", "FIXED", "RETIRED"}


def _dependency_versions(payload: object) -> dict[str, int]:
    if not isinstance(payload, Mapping):
        return {}
    result: dict[str, int] = {}
    for key in ("dependency_versions", "binding_versions"):
        raw = payload.get(key)
        if isinstance(raw, Mapping):
            for path, revision in raw.items():
                if isinstance(revision, int) and revision >= 0:
                    result[str(path).replace("\\", "/")] = revision
    path = next((payload.get(key) for key in (
        "path", "file", "filepath", "file_path", "source_path", "target_path",
    ) if payload.get(key)), "")
    revision = payload.get("workspace_revision", payload.get("revision"))
    if path and isinstance(revision, int) and revision >= 0:
        result[str(path).replace("\\", "/")] = revision
    return result


def _is_pinned(kind: str, payload: object, links: RecallLinks, status: str, explicit: bool) -> bool:
    if explicit or _status_resolved(status, payload):
        return explicit and not _status_resolved(status, payload)
    if not isinstance(payload, Mapping):
        return False
    if payload.get("frozen_repro") is True or payload.get("frozen_reproduction") is True:
        return True
    if payload.get("admitted_root_cause") is True or payload.get("root_cause_admitted") is True:
        return True
    if kind == "hypothesis" and str(payload.get("status", "")).upper() in {"FALSIFIED", "ELIMINATED"}:
        return True
    return bool(links.obligations and payload.get("blocking") is True)


@dataclass(frozen=True)
class RecallView:
    items: tuple[dict[str, object], ...]
    omitted_count: int
    index: dict[str, object]
    used_tokens: int
    budget_tokens: int
    requested_ids: tuple[str, ...] = ()
    selector_metrics: dict[str, object] = dataclass_field(default_factory=dict)
    omissions: tuple[dict[str, object], ...] = ()


class RecallStore:
    """Append-only item tape plus a bounded relevance view."""

    def __init__(self, *, state: Mapping[str, object] | None = None) -> None:
        self._items: list[dict[str, object]] = []
        self._states: dict[str, dict[str, object]] = {}
        self._source_keys: set[str] = set()
        self._requested: list[str] = []
        self._next_seq = 0
        self.render_overflow_count = 0
        self.render_overflow_consecutive = 0
        self.render_count = 0
        self.selector_events: list[dict[str, object]] = []
        self.recall_candidate_misses: list[dict[str, object]] = []
        if state is not None:
            self._restore(state)

    def _restore(self, state: Mapping[str, object]) -> None:
        if state.get("schema") != SCHEMA:
            raise RecallError("unsupported recall store state")
        for raw in state.get("items", ()):
            if isinstance(raw, Mapping):
                item = copy.deepcopy(dict(raw))
                ident = str(item.get("id", ""))
                if ident and ident not in self._states:
                    self._items.append(item)
                    self._states[ident] = {
                        "resolved": bool(item.get("resolved", False)),
                        "pinned": bool(item.get("pinned", False)),
                        "status": str(item.get("status", "OPEN")),
                        "stale": bool(item.get("stale", False)),
                        "recheck_required": bool(item.get("recheck_required", False)),
                    }
                    self._next_seq = max(self._next_seq, int(item.get("seq", 0)) + 1)
        raw_states = state.get("states")
        if isinstance(raw_states, Mapping):
            for ident, raw_state in raw_states.items():
                if str(ident) in self._states and isinstance(raw_state, Mapping):
                    self._states[str(ident)].update(copy.deepcopy(dict(raw_state)))
        self._source_keys = {str(item) for item in state.get("source_keys", ())}
        self._requested = [str(item) for item in state.get("requested", ()) if str(item)]
        metrics = state.get("metrics")
        if isinstance(metrics, Mapping):
            self.render_overflow_count = max(0, int(metrics.get("render_overflow_count", 0) or 0))
            self.render_overflow_consecutive = max(0, int(metrics.get("render_overflow_consecutive", 0) or 0))
            self.render_count = max(0, int(metrics.get("render_count", 0) or 0))
            raw_misses = metrics.get("recall_candidate_misses", ())
            if isinstance(raw_misses, Sequence):
                self.recall_candidate_misses = [dict(item) for item in raw_misses if isinstance(item, Mapping)][-64:]
            raw_events = metrics.get("selector_events", ())
            if isinstance(raw_events, Sequence):
                self.selector_events = [dict(item) for item in raw_events
                                        if isinstance(item, Mapping)][-MAX_SELECTOR_EVENTS:]

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "items": copy.deepcopy(self._items),
            "states": copy.deepcopy(self._states),
            "source_keys": sorted(self._source_keys),
            "requested": list(self._requested),
            "metrics": {
                "render_overflow_count": self.render_overflow_count,
                "render_overflow_consecutive": self.render_overflow_consecutive,
                "render_count": self.render_count,
                "recall_candidate_misses": copy.deepcopy(self.recall_candidate_misses[-64:]),
                "selector_events": copy.deepcopy(self.selector_events[-MAX_SELECTOR_EVENTS:]),
            },
        }

    def __len__(self) -> int:
        return len(self._items)

    def _stateful(self, item: Mapping[str, object]) -> dict[str, object]:
        result = copy.deepcopy(dict(item))
        result.update(self._states.get(str(item.get("id")), {}))
        result["pinned"] = bool(result.get("pinned")) and not bool(result.get("resolved"))
        return result

    def items(self) -> tuple[dict[str, object], ...]:
        return tuple(self._stateful(item) for item in self._items)

    def get(self, item_id: object) -> dict[str, object] | None:
        ident = str(item_id or "")
        for item in self._items:
            if item.get("id") == ident:
                return self._stateful(item)
        return None

    def append(
        self,
        kind: str,
        payload: object,
        *,
        links: RecallLinks | Mapping[str, object] | None = None,
        gist: str = "",
        item_id: str | None = None,
        source_key: str | None = None,
        step: int | None = None,
        status: str = "OPEN",
        pinned: bool = False,
        required_group: str = "",
    ) -> str:
        if source_key and source_key in self._source_keys:
            existing = next((item for item in self._items if item.get("source_key") == source_key), None)
            if existing is not None:
                return str(existing["id"])
        normalized_kind = _text(kind, 48).lower().replace(" ", "_") or "item"
        payload_copy = copy.deepcopy(payload)
        inferred = links_from(payload_copy)
        merged = _merge_links(inferred, RecallLinks.from_value(links))
        ident = item_id or f"{normalized_kind}-{self._next_seq}"
        if not _ID_RE.fullmatch(ident):
            ident = f"{normalized_kind}-{_digest(ident)}"
        if ident in self._states:
            return ident
        status_text = _text(status, 32).upper() or "OPEN"
        item = {
            "id": ident,
            "kind": normalized_kind,
            "seq": self._next_seq,
            "step": int(step) if isinstance(step, int) else None,
            "gist": _text(gist) or _gist(payload_copy),
            "links": merged.to_dict(),
            "dependency_versions": _dependency_versions(payload_copy),
            "assertion_atoms": copy.deepcopy(payload_copy.get("assertion_atoms", payload_copy.get("decision_atoms", ()))
                                if isinstance(payload_copy, Mapping) else ()),
            "payload": payload_copy,
            "status": status_text,
            "pinned": _is_pinned(normalized_kind, payload_copy, merged, status_text, pinned),
            "resolved": _status_resolved(status_text, payload_copy),
            "stale": status_text == "STALE",
            "recheck_required": status_text == "STALE",
            "required_group": _text(required_group or (payload_copy.get("required_group", "")
                                    if isinstance(payload_copy, Mapping) else ""), 80),
        }
        if source_key:
            item["source_key"] = source_key
            self._source_keys.add(source_key)
        self._items.append(item)
        self._states[ident] = {
            "resolved": bool(item["resolved"]),
            "pinned": bool(item["pinned"]),
            "status": status_text,
            "stale": bool(item["stale"]),
            "recheck_required": bool(item["recheck_required"]),
        }
        self._next_seq += 1
        return ident

    def add_unique(self, kind: str, payload: object, **kwargs: object) -> str:
        source_key = kwargs.get("source_key")
        if source_key is None:
            kwargs["source_key"] = f"{kind}:{_digest(payload)}"
        return self.append(kind, payload, **kwargs)  # type: ignore[arg-type]

    def resolve(self, item_id: object, *, status: str = "RESOLVED") -> bool:
        ident = str(item_id or "")
        if ident not in self._states:
            return False
        self._states[ident] = {"resolved": True, "pinned": False, "status": _text(status, 32).upper()}
        return True

    def mark_stale(self, item_id: object, *, reason: str = "EVIDENCE_STALE") -> bool:
        """Mark a recalled item for re-check; falsified pins remain pinned."""
        ident = str(item_id or "")
        if ident not in self._states:
            return False
        current = dict(self._states[ident])
        current.update({"stale": True, "recheck_required": True, "stale_reason": _text(reason, 80)})
        if str(current.get("status", "")).upper() not in {"FALSIFIED", "ELIMINATED"}:
            current["status"] = "STALE"
        self._states[ident] = current
        return True

    def request(self, item_id: object) -> dict[str, object]:
        item = self.get(item_id)
        if item is None:
            return {"status": "RECALL_NOT_FOUND", "id": str(item_id or ""), "next_frame": False}
        ident = str(item["id"])
        if ident not in self._requested:
            self._requested.append(ident)
        return {
            "status": "RECALL_QUEUED",
            "id": ident,
            "kind": item["kind"],
            "gist": item["gist"],
            "next_frame": True,
        }

    def _focus_links(self, focus: object) -> RecallLinks:
        return links_from(focus)

    @staticmethod
    def _overlap(item: Mapping[str, object], focus: RecallLinks) -> int:
        links = RecallLinks.from_value(item.get("links"))
        score = 0
        weights = {"requirements": 12, "obligations": 12, "files": 8,
                   "symbols": 8, "hypotheses": 12, "stages": 5}
        for field, weight in weights.items():
            left = set(getattr(links, field))
            right = set(getattr(focus, field))
            if field == "files":
                score += weight * sum(
                    1 for value in left for target in right
                    if value == target or value.endswith("/" + target) or target.endswith("/" + value)
                )
            else:
                score += weight * len(left & right)
        return score

    @staticmethod
    def _line(item: Mapping[str, object], *, full: bool = False) -> str:
        links = RecallLinks.from_value(item.get("links"))
        link_bits = ",".join(
            f"{field[:2]}={','.join(getattr(links, field)[:2])}"
            for field in LINK_FIELDS if getattr(links, field)
        )
        line = (f"id={item['id']} kind={item['kind']} step={item.get('step', '?')} "
                f"status={item.get('status', 'OPEN')} links=[{link_bits}] gist={item['gist']}")
        validity = str(item.get("selector_validity", "CURRENT"))
        if validity != "CURRENT":
            line += f" validity={validity} invalidation_capsule=RECHECK_REQUIRED"
        if full:
            try:
                detail = json.dumps(item.get("payload"), sort_keys=True, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                detail = repr(item.get("payload"))
            line += " payload=" + _text(detail, 520)
        return _text(line, 900)

    def index(self, *, page: int = 0, page_size: int = DEFAULT_INDEX_PAGE_SIZE) -> dict[str, object]:
        if type(page) is not int or page < 0:
            raise RecallError("recall index page must be a non-negative integer")
        if type(page_size) is not int or not 1 <= page_size <= MAX_INDEX_PAGE_SIZE:
            raise RecallError(f"recall index page_size must be within 1..{MAX_INDEX_PAGE_SIZE}")
        start = page * page_size
        rows = self._items[start:start + page_size]
        return {
            "schema": SCHEMA,
            "page": page,
            "page_size": page_size,
            "total": len(self._items),
            "pages": (len(self._items) + page_size - 1) // page_size,
            "items": [{"id": item["id"], "kind": item["kind"], "step": item.get("step"),
                       "status": self._stateful(item).get("status"), "gist": item["gist"],
                       "links": copy.deepcopy(item["links"])} for item in rows],
        }

    def select(self, focus: object = None, *, budget_tokens: int = DEFAULT_VIEW_TOKENS,
               max_items: int = DEFAULT_VIEW_ITEMS, index_page: int = 0,
               kinds: Sequence[str] | None = None, ledger: object | None = None,
               decision_predicates: Sequence[object] = (), arm: str = "hybrid",
               consume_requested: bool = True, track_shown: bool = True,
               historical_ids: Sequence[str] = ()) -> RecallView:
        if type(budget_tokens) is not int or budget_tokens < 32:
            raise RecallError("recall view budget must be at least 32 tokens")
        budget_tokens = min(budget_tokens, MAX_VIEW_TOKENS)
        if type(max_items) is not int or max_items < 1:
            raise RecallError("recall view max_items must be positive")
        requested = set(self._requested)
        candidates = []
        for raw in self._items:
            item = self._stateful(raw)
            item["token_cost"] = (len(self._line(item, full=str(item["id"]) in requested)) + 3) // 4
            candidates.append(item)
        from frame_selector import select_items
        result = select_items(
            candidates,
            focus,
            budget_tokens=budget_tokens,
            max_items=max_items,
            ledger=ledger,
            requested_ids=tuple(requested),
            decision_predicates=decision_predicates,
            arm=arm,
            historical_ids=historical_ids,
            kinds=kinds,
        )
        omissions = tuple(copy.deepcopy(item) for item in result.omissions)
        for omission in omissions:
            self.selector_events.append({
                "event": "selector_omission",
                "schema": OMISSION_SCHEMA,
                "render": self.render_count + 1,
                **omission,
            })
        self.selector_events = self.selector_events[-MAX_SELECTOR_EVENTS:]
        selected_ids = {str(item["id"]) for item in result.items}
        requested_ids = tuple(str(item["id"]) for item in result.items if str(item["id"]) in requested)
        if consume_requested:
            for ident in selected_ids:
                if ident in self._requested:
                    self._requested.remove(ident)
        if requested and track_shown:
            for ident in sorted(requested - selected_ids):
                miss = {
                    "event": "recall_of_unshown",
                    "id": ident,
                    "candidate_miss": True,
                    "reason": "invalidated" if any(
                        str(item.get("id")) == ident and item.get("selector_validity") == "INVALIDATED"
                        for item in candidates
                    ) else "cap_or_budget",
                    "omission_reason": next(
                        (row.get("reason") for row in omissions
                         if str(row.get("item_id")) == ident),
                        "cap_or_budget",
                    ),
                    "render": self.render_count + 1,
                }
                self.recall_candidate_misses.append(miss)
                self.selector_events.append(dict(miss))
        self.selector_events = self.selector_events[-MAX_SELECTOR_EVENTS:]
        metrics = dict(result.metrics)
        metrics["recall_of_unshown"] = sum(1 for event in self.selector_events if event.get("event") == "recall_of_unshown")
        metrics["selector_omission_count"] = len(omissions)
        metrics["selector_omissions"] = omissions
        return RecallView(tuple(result.items), result.omitted_count, self.index(page=index_page),
                          result.used_tokens, budget_tokens, requested_ids, metrics, omissions)

    def render(self, focus: object = None, *, budget_tokens: int = DEFAULT_VIEW_TOKENS,
               max_items: int = DEFAULT_VIEW_ITEMS, index_page: int = 0,
               ledger: object | None = None, decision_predicates: Sequence[object] = (),
               arm: str = "hybrid") -> str:
        view = self.select(focus, budget_tokens=budget_tokens, max_items=max_items,
                           index_page=index_page, ledger=ledger,
                           decision_predicates=decision_predicates, arm=arm)
        self.render_count += 1
        lines = [f"[RECALL schema={SCHEMA} total={view.index['total']} page={view.index['page']}/{view.index['pages']}]"]
        if view.items:
            lines.append("VIEW")
            lines.extend(self._line(item, full=str(item["id"]) in view.requested_ids) for item in view.items)
        lines.append("INDEX")
        lines.extend(
            f"id={row['id']} kind={row['kind']} step={row.get('step', '?')} gist={row['gist']}"
            for row in view.index["items"]
        )
        if view.index["page"] + 1 < view.index["pages"]:
            lines.append(f"INDEX_NEXT page={view.index['page'] + 1}")
        lines.append(
            f"SELECTOR arm={view.selector_metrics.get('arm', arm)} "
            f"pins={view.selector_metrics.get('pin_count', 0)} "
            f"graph_edges={view.selector_metrics.get('graph_edge_count', 0)} "
            f"recall_of_unshown={view.selector_metrics.get('recall_of_unshown', 0)}"
        )
        reason_counts = view.selector_metrics.get("omission_reasons", {})
        if isinstance(reason_counts, Mapping):
            compact_counts = ",".join(
                f"{reason}={int(reason_counts.get(reason, 0) or 0)}"
                for reason in sorted(reason_counts)
            )
            lines.append(
                f"OMISSIONS count={view.selector_metrics.get('omission_count', 0)} "
                f"reasons={compact_counts or 'none'}"
            )
        lines.append(f"BOUNDED tokens<={view.budget_tokens} used={view.used_tokens} omitted={view.omitted_count}")
        lines.append("Use recall(id) for any indexed item; recall is read-only.")
        return "\n".join(lines)

    def record_render_overflow(self, reason: str = "RenderOverflow") -> None:
        self.render_overflow_count += 1
        self.render_overflow_consecutive += 1

    def record_render_success(self) -> None:
        self.render_overflow_consecutive = 0

    def health(self, *, max_consecutive: int = DEFAULT_MAX_OVERFLOW_STREAK) -> dict[str, object]:
        return {
            "render_overflow_count": self.render_overflow_count,
            "render_overflow_consecutive": self.render_overflow_consecutive,
            "render_count": self.render_count,
            "max_consecutive": max_consecutive,
            "healthy": self.render_overflow_consecutive <= max_consecutive,
        }

    def verify_health(self, *, max_consecutive: int = DEFAULT_MAX_OVERFLOW_STREAK) -> dict[str, object]:
        result = self.health(max_consecutive=max_consecutive)
        if not result["healthy"]:
            raise RecallHealthError(
                f"RenderOverflow persisted for {result['render_overflow_consecutive']} consecutive steps"
            )
        return result


def ensure_runtime_store(runtime: object) -> RecallStore:
    store = getattr(runtime, "recall_store", None)
    if isinstance(store, RecallStore):
        return store
    store = RecallStore()
    setattr(runtime, "recall_store", store)
    return store


def record_runtime_item(runtime: object, kind: str, payload: object, **kwargs: object) -> str:
    try:
        return ensure_runtime_store(runtime).add_unique(kind, payload, **kwargs)
    except (TypeError, ValueError, RecallError):
        # Recall is a presentation aid; malformed optional metadata cannot break
        # the receipt path that owns the actual evidence.
        return ""


def sync_runtime(runtime: object) -> RecallStore:
    """Index newly observed host material without making the model its authority."""
    store = ensure_runtime_store(runtime)
    owner = getattr(runtime, "owner", None)
    sources = (runtime, owner) if owner is not None else (runtime,)
    for source_obj in sources:
        for attr, kind in (("host_check_receipts", "receipt"), ("host_repair_check_receipts", "receipt"),
                           ("test_receipts", "receipt"), ("probe_receipts", "receipt"),
                           ("observation_probe_receipts", "receipt"),
                           ("localize_receipts", "receipt"),
                           ("_receipts", "receipt")):
            values = getattr(source_obj, attr, ()) or ()
            for index, raw in enumerate(values):
                payload = raw.to_dict() if callable(getattr(raw, "to_dict", None)) else raw
                if isinstance(payload, Mapping):
                    source = str(payload.get("receipt_id") or payload.get("check_id") or
                                 f"{attr}:{index}:{_digest(payload)}")
                    store.add_unique(kind, payload, source_key=f"{kind}:{source}", step=payload.get("step"))
    decision_cache = getattr(runtime, "decision_cache", None)
    if decision_cache is not None and callable(getattr(decision_cache, "all_records", None)):
        for row in decision_cache.all_records():
            store.add_unique("decision", row, source_key=f"decision:{row.get('id')}:{row.get('version')}")
    ledger = getattr(runtime, "_f4_reasoning_ledger", None)
    if ledger is not None:
        for receipt in ledger.receipts() if callable(getattr(ledger, "receipts", None)) else ():
            payload = receipt.as_dict() if callable(getattr(receipt, "as_dict", None)) else receipt
            store.add_unique("receipt", payload, source_key=f"ledger-receipt:{_digest(payload)}",
                             step=payload.get("turn") if isinstance(payload, Mapping) else None)
        for claim in ledger.claims() if callable(getattr(ledger, "claims", None)) else ():
            payload = claim.as_dict() if callable(getattr(claim, "as_dict", None)) else claim
            if isinstance(payload, Mapping):
                store.add_unique("hypothesis", payload, source_key=f"ledger-claim:{payload.get('id')}",
                                 step=payload.get("turn"), status=str(payload.get("status", "OPEN")))
    return store


def runtime_focus(runtime: object, frame: Mapping[str, object] | None = None,
                   path: str = "") -> dict[str, object]:
    focus: dict[str, object] = dict(frame or {})
    focus["path"] = path or focus.get("path", "")
    owner = getattr(runtime, "owner", None)
    if owner is not None:
        focus["symbol"] = getattr(owner, "fn", focus.get("symbol", ""))
        row = getattr(owner, "task_row", None)
        if isinstance(row, Mapping):
            focus.update({key: copy.deepcopy(row[key]) for key in (
                "requirement_id", "requirement_ids", "obligation_id", "obligation_ids",
                "target", "target_id", "target_ids", "active_target", "active_target_id",
                "stage", "idealpath_stage") if key in row})
    return focus


def render_runtime(runtime: object, frame: Mapping[str, object] | None = None,
                   path: str = "") -> str:
    store = sync_runtime(runtime)
    try:
        from kernel_invalidation import sync_runtime_invalidation
        ledger = sync_runtime_invalidation(runtime)
    except (ImportError, TypeError, ValueError):
        ledger = None
    return store.render(runtime_focus(runtime, frame, path), ledger=ledger,
                        decision_predicates=runtime_decision_predicates(runtime, frame)) if len(store) else ""


def runtime_decision_predicates(runtime: object, frame: Mapping[str, object] | None = None) -> tuple[object, ...]:
    """Read only explicit host predicate records for the optional bonus."""
    values: list[object] = []
    sources = (frame, runtime, getattr(runtime, "owner", None))
    for source in sources:
        if not isinstance(source, Mapping) and source is not None:
            raw = getattr(source, "decision_predicates", None)
            if raw is None:
                raw = getattr(source, "_decision_predicates", None)
        elif isinstance(source, Mapping):
            raw = source.get("decision_predicates", source.get("host_decision_predicates"))
        else:
            raw = None
        if isinstance(raw, Mapping):
            values.append(raw)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            values.extend(item for item in raw if isinstance(item, Mapping))
    return tuple(values)


def selected_runtime_receipts(runtime: object) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Return selector-chosen test/probe receipts for the unified frame."""
    store = sync_runtime(runtime)
    if not len(store):
        return (), ()
    try:
        from kernel_invalidation import sync_runtime_invalidation
        ledger = sync_runtime_invalidation(runtime)
    except (ImportError, TypeError, ValueError):
        ledger = None
    view = store.select(runtime_focus(runtime), budget_tokens=720, max_items=16,
                        ledger=ledger, kinds=("receipt",), consume_requested=False,
                        track_shown=False,
                           decision_predicates=runtime_decision_predicates(runtime))
    tests: list[object] = []
    probes: list[object] = []
    for item in view.items:
        payload = item.get("payload")
        kind = str(payload.get("kind", "")) if isinstance(payload, Mapping) else ""
        if kind in {"probe", "observation_probe"} or "probe" in kind:
            probes.append(payload)
        elif kind in {"test", "check"} or "test" in kind or "check" in kind:
            tests.append(payload)
        else:
            tests.append(payload)
    return tuple(tests), tuple(probes)


def recall_runtime(runtime: object, item_id: object) -> dict[str, object]:
    return ensure_runtime_store(runtime).request(item_id)


__all__ = [
    "DEFAULT_INDEX_PAGE_SIZE", "DEFAULT_MAX_OVERFLOW_STREAK", "DEFAULT_VIEW_ITEMS",
    "DEFAULT_VIEW_TOKENS", "LINK_FIELDS", "RecallError", "RecallHealthError", "RecallLinks",
    "RecallStore", "RecallView", "SCHEMA", "ensure_runtime_store", "links_from",
    "recall_runtime", "record_runtime_item", "render_runtime", "runtime_decision_predicates", "runtime_focus", "selected_runtime_receipts", "sync_runtime",
]

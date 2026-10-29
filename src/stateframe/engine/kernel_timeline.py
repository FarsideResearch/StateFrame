"""Journal-backed session timeline for bounded StateFrame context.

The journal and RECALL remain the durable authorities.  ``TimelineIndex`` is a
rebuildable projection: it stores no second copy of the session and only keeps
indexes plus the small event metadata needed to answer typed queries.  Message
content is retained in journal timeline events so a later frame can show the
latest user message verbatim without replaying the conversation transcript.
"""

from __future__ import annotations

import copy
import heapq
import json
import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from kernel_history_spool import EventLookup, HistorySpool, history


SCHEMA = "stateframe-timeline/1"
FRAME_SCHEMA = "stateframe-timeline-frame/1"
DEFAULT_RECENT_TURNS = 6
DEFAULT_OPEN_ASKS = 12
DEFAULT_RELEVANT_ITEMS = 8
DEFAULT_FRAME_TOKENS = 900
MAX_FRAME_TOKENS = 2400
MAX_RECENT_TURN_CHARS = 220
MAX_ASK_CHARS = 320
MAX_RELEVANT_CHARS = 360
_RELEVANCE_HEAD_SIZE = 64
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]{2,}")
_PATH_KEYS = frozenset({
    "file", "files", "filepath", "file_path", "path", "paths", "target",
    "target_path", "source_path", "changed_file", "changed_files",
})
_ENTITY_KEYS = frozenset({
    "entity", "entities", "entity_id", "entity_ids", "requirement",
    "requirement_id", "requirements", "obligation", "obligation_id",
    "decision", "decision_id", "decision_ids", "symbol", "symbols",
    "stage", "stage_id", "hypothesis", "hypothesis_id",
})
_EVENT_FIELDS = frozenset({
    "actor", "content", "message", "message_id", "turn", "timestamp_ns",
    "entities", "entity_ids", "prompted_by", "satisfies", "supersedes",
    "status", "ask_id", "changed_entities", "decision_id", "stage",
})


class TimelineError(ValueError):
    """Malformed timeline input or an unsupported query."""


def _copy(value: object) -> object:
    return copy.deepcopy(value)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _compact(value: object, limit: int) -> str:
    text = " ".join(_text(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


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
    item = _text(value).strip()
    return (item,) if item else ()


def _normalise_entity(value: str) -> str:
    return " ".join(value.replace("\\", "/").split()).strip().lower()


def _extract_entities(value: object, *, depth: int = 0) -> tuple[str, ...]:
    """Extract declared identifiers; never turn arbitrary prose into entities."""
    if depth > 3 or not isinstance(value, Mapping):
        return ()
    result: list[str] = []
    for key, raw in value.items():
        name = str(key).lower()
        if name in _PATH_KEYS or name in _ENTITY_KEYS:
            result.extend(_values(raw))
        elif isinstance(raw, Mapping):
            result.extend(_extract_entities(raw, depth=depth + 1))
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            for nested in raw:
                if isinstance(nested, Mapping):
                    result.extend(_extract_entities(nested, depth=depth + 1))
    normalized: list[str] = []
    for item in result:
        entity = _normalise_entity(item)
        if entity and entity not in normalized:
            normalized.append(entity)
    return tuple(normalized)


def _tokens(value: object) -> tuple[str, ...]:
    result: list[str] = []
    for item in _TOKEN_RE.findall(_text(value).lower()):
        if item not in result:
            result.append(item)
    return tuple(result)


def _as_ids(value: object) -> tuple[str, ...]:
    result: list[str] = []
    for item in _values(value):
        ident = item.strip()
        if ident and ident not in result:
            result.append(ident)
    return tuple(result)


def _timestamp(value: object) -> int:
    if value is None:
        return time.time_ns()
    if type(value) is not int or value < 0:
        raise TimelineError("timestamp_ns must be a non-negative integer")
    return value


def _turn(value: object, fallback: int) -> int:
    if value is None:
        return fallback
    if type(value) is not int or value < 0:
        raise TimelineError("turn must be a non-negative integer")
    return value


def _event_type(kind: str) -> str:
    lowered = kind.lower().replace("-", "_")
    if lowered in {"user_message", "user", "user_ask"}:
        return "user_message"
    if lowered in {"assistant_message", "assistant", "assistant_reply"}:
        return "assistant_message"
    if "stage" in lowered or "transition" in lowered:
        return "stage_change"
    if "decision" in lowered or lowered in {"why", "revisit"}:
        return "decision"
    if ("receipt" in lowered or "check" in lowered or "outcome" in lowered
            or lowered in {"test", "probe", "observation"}):
        return "receipt"
    if ("action" in lowered or "dispatch" in lowered or lowered in {"edit", "read", "write"}):
        return "action"
    return "event"


def _line_payload(event: Mapping[str, object]) -> str:
    content = event.get("content", event.get("message", event.get("gist", "")))
    return _text(content)


@dataclass(frozen=True)
class TimelineEvent:
    """A normalized projection row with durable source and message identities."""

    event_id: str
    session_id: str
    kind: str
    actor: str
    turn: int
    timestamp_ns: int
    message_id: str | None = None
    ask_id: str | None = None
    content: str = ""
    entities: tuple[str, ...] = ()
    prompted_by: tuple[str, ...] = ()
    satisfies: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    changed_entities: tuple[str, ...] = ()
    status: str = ""
    source_event_id: str | None = None
    payload: object = None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "kind": self.kind,
            "actor": self.actor,
            "turn": self.turn,
            "timestamp_ns": self.timestamp_ns,
            "message_id": self.message_id,
            "ask_id": self.ask_id,
            "content": self.content,
            "entities": list(self.entities),
            "prompted_by": list(self.prompted_by),
            "satisfies": list(self.satisfies),
            "supersedes": list(self.supersedes),
            "changed_entities": list(self.changed_entities),
            "status": self.status,
            "source_event_id": self.source_event_id,
            "payload": _copy(self.payload),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], *, source_event_id: str | None = None) -> "TimelineEvent":
        event_id = _text(value.get("event_id") or value.get("id")).strip()
        session_id = _text(value.get("session_id") or "session").strip() or "session"
        if not event_id:
            raise TimelineError("timeline event requires event_id")
        return cls(
            event_id=event_id,
            session_id=session_id,
            kind=_event_type(_text(value.get("kind") or "event")),
            actor=_text(value.get("actor") or "system").strip() or "system",
            turn=_turn(value.get("turn"), 0),
            timestamp_ns=_timestamp(value.get("timestamp_ns")),
            message_id=(_text(value["message_id"]) if value.get("message_id") else None),
            ask_id=(_text(value["ask_id"]) if value.get("ask_id") else None),
            content=_text(value.get("content", value.get("message", ""))),
            entities=tuple(_extract_entities({"entities": value.get("entities", value.get("entity_ids", ())),
                                              "payload": value.get("payload")})),
            prompted_by=_as_ids(value.get("prompted_by")),
            satisfies=_as_ids(value.get("satisfies")),
            supersedes=_as_ids(value.get("supersedes")),
            changed_entities=tuple(_extract_entities({"entities": value.get("changed_entities", ())})),
            status=_text(value.get("status", "")).upper(),
            source_event_id=source_event_id or (_text(value.get("source_event_id")) or None),
            payload=_copy(value.get("payload", dict(value))),
        )


@dataclass(frozen=True)
class TimelineFrame:
    """The bounded StateFrame slice; all omitted history remains queryable."""

    latest_user_message: str | None
    latest_user_message_id: str | None
    open_asks: tuple[dict[str, object], ...]
    recent_turns: tuple[str, ...]
    relevant: tuple[dict[str, object], ...]
    rendered: str
    used_tokens: int
    event_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": FRAME_SCHEMA,
            "latest_user_message": self.latest_user_message,
            "latest_user_message_id": self.latest_user_message_id,
            "open_asks": _copy(self.open_asks),
            "recent_turns": list(self.recent_turns),
            "relevant": _copy(self.relevant),
            "rendered": self.rendered,
            "used_tokens": self.used_tokens,
            "event_count": self.event_count,
        }


class TimelineIndex:
    """An append-only, rebuildable projection over journal/RECALL/selector data."""

    def __init__(self, session_id: str = "session", *, journal: object | None = None,
                 recall: object | None = None, selector: Callable[..., object] | None = None,
                 decision_graph: object | None = None) -> None:
        self.session_id = _text(session_id).strip() or "session"
        self.journal = journal
        self.recall = recall
        self.selector = selector
        self.decision_graph = decision_graph
        self._events = history(directory=self._journal_path().parent if self._journal_path() else None)
        self._by_id: Mapping[str, TimelineEvent] = {}
        self._by_source_id: set[str] = set()
        self._by_turn: dict[int, list[str]] = defaultdict(list)
        self._by_time_bucket: dict[int, list[str]] = defaultdict(list)
        self._position_by_id: dict[str, int] = {}
        if isinstance(self._events, HistorySpool):
            self._by_id = EventLookup(self._events, self._position_by_id)
            self._relevance_head = []
        self._by_entity: dict[str, list[str]] = defaultdict(list)
        self._by_message_id: dict[str, list[str]] = defaultdict(list)
        self._by_message_token: dict[str, list[str]] = defaultdict(list)
        self._user_asks: list[str] = []
        self._ask_status: dict[str, str] = {}
        self._current_turn = 0
        self._next_event_number = 1
        self._journal_offset = 0
        self._journal_record_ids: set[str] = set()
        self.refresh()

    @classmethod
    def from_journal(cls, journal: object, *, session_id: str | None = None,
                     recall: object | None = None,
                     selector: Callable[..., object] | None = None,
                     decision_graph: object | None = None) -> "TimelineIndex":
        ident = session_id or _text(getattr(journal, "run_id", "session")) or "session"
        return cls(ident, journal=journal, recall=recall, selector=selector,
                   decision_graph=decision_graph)

    def __len__(self) -> int:
        return len(self._events)

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def latest_turn(self) -> int:
        return self._current_turn

    @property
    def has_session_messages(self) -> bool:
        return any(event.kind in {"user_message", "assistant_message"} for event in self._events)

    def _new_id(self, prefix: str) -> str:
        ident = f"{self.session_id}:{prefix}:{self._next_event_number}"
        self._next_event_number += 1
        return ident

    def _make_event(self, kind: str, *, actor: str, turn: object, timestamp_ns: object,
                    event_id: str | None, message_id: str | None, ask_id: str | None,
                    content: object, entities: object, prompted_by: object,
                    satisfies: object, supersedes: object, changed_entities: object,
                    status: object, payload: object, source_event_id: str | None) -> TimelineEvent:
        event_kind = _event_type(kind)
        fallback_turn = self._current_turn
        event_turn = _turn(turn, fallback_turn)
        event_ident = event_id or self._new_id(event_kind)
        payload_copy = _copy(payload)
        all_entities = list(_extract_entities({"entities": entities, "payload": payload_copy}))
        for entity in _extract_entities(payload_copy):
            if entity not in all_entities:
                all_entities.append(entity)
        changed = list(_extract_entities({"entities": changed_entities}))
        for entity in changed:
            if entity not in all_entities:
                all_entities.append(entity)
        return TimelineEvent(
            event_id=event_ident,
            session_id=self.session_id,
            kind=event_kind,
            actor=_text(actor).strip() or "system",
            turn=event_turn,
            timestamp_ns=_timestamp(timestamp_ns),
            message_id=message_id,
            ask_id=ask_id,
            content=_text(content),
            entities=tuple(all_entities),
            prompted_by=_as_ids(prompted_by),
            satisfies=_as_ids(satisfies),
            supersedes=_as_ids(supersedes),
            changed_entities=tuple(changed),
            status=_text(status).upper(),
            source_event_id=source_event_id,
            payload=payload_copy,
        )

    def _append_local(self, event: TimelineEvent) -> dict[str, object]:
        if event.event_id in self._by_id:
            return self._by_id[event.event_id].as_dict()
        if event.source_event_id and event.source_event_id in self._by_source_id:
            return event.as_dict()
        self._events.append(event)
        if isinstance(self._by_id, dict):
            self._by_id[event.event_id] = event
        self._position_by_id[event.event_id] = len(self._events) - 1
        if hasattr(self, '_relevance_head') and event.kind in {"action", "receipt", "decision", "stage_change"}:
            key = (event.turn, event.timestamp_ns, len(self._events) - 1)
            heapq.heappush(self._relevance_head, key)
            if len(self._relevance_head) > _RELEVANCE_HEAD_SIZE:
                heapq.heappop(self._relevance_head)
        if event.source_event_id:
            self._by_source_id.add(event.source_event_id)
        self._next_event_number = max(self._next_event_number, len(self._events) + 1)
        self._current_turn = max(self._current_turn, event.turn)
        self._by_turn[event.turn].append(event.event_id)
        self._by_time_bucket[event.timestamp_ns // 1_000_000_000].append(event.event_id)
        for entity in event.entities:
            self._by_entity[entity].append(event.event_id)
            if "/" in entity:
                self._by_entity[entity.rsplit("/", 1)[-1]].append(event.event_id)
        for message_id in (event.message_id, event.ask_id):
            if message_id:
                self._by_message_id[message_id].append(event.event_id)
        for token in _tokens(event.content):
            self._by_message_token[token].append(event.event_id)
        if event.kind == "user_message" and event.ask_id:
            self._user_asks.append(event.event_id)
            self._ask_status[event.ask_id] = event.status or "OPEN"
        for ask_id in event.supersedes:
            if ask_id in self._ask_status:
                self._ask_status[ask_id] = "SUPERSEDED"
        for ask_id in event.satisfies:
            if ask_id in self._ask_status:
                self._ask_status[ask_id] = "SATISFIED"
        if event.ask_id and event.status in {"SATISFIED", "SUPERSEDED", "CLOSED"}:
            self._ask_status[event.ask_id] = event.status
        return event.as_dict()

    def append(self, kind: str, payload: object = None, *, actor: str = "system",
               turn: int | None = None, timestamp_ns: int | None = None,
               event_id: str | None = None, message_id: str | None = None,
               ask_id: str | None = None, content: object = "",
               entities: object = (), prompted_by: object = (),
               satisfies: object = (), supersedes: object = (),
               changed_entities: object = (), status: str = "",
               persist: bool = True) -> dict[str, object]:
        """Append one projection row and, by default, one durable journal event."""
        event = self._make_event(
            kind, actor=actor, turn=turn, timestamp_ns=timestamp_ns,
            event_id=event_id, message_id=message_id, ask_id=ask_id,
            content=content, entities=entities, prompted_by=prompted_by,
            satisfies=satisfies, supersedes=supersedes,
            changed_entities=changed_entities, status=status, payload=payload,
            source_event_id=None,
        )
        if persist and self.journal is not None:
            record = getattr(self.journal, "emit")("timeline_event", timeline_event=event.as_dict())
            source_id = _text(record.get("event_id")) if isinstance(record, Mapping) else ""
            event = dataclass_replace(event, source_event_id=source_id or None)
        return self._append_local(event)

    def append_message(self, role: str, content: str, *, turn: int | None = None,
                       message_id: str | None = None, ask_id: str | None = None,
                       entities: object = (), prompted_by: object = (),
                       satisfies: object = (), supersedes: object = (),
                       status: str = "", persist: bool = True) -> dict[str, object]:
        """Record a session message without accepting transcript history."""
        normalized_role = _text(role).strip().lower()
        if normalized_role not in {"user", "assistant"}:
            raise TimelineError("session messages must have role user or assistant")
        if normalized_role == "user" and turn is None:
            turn = self._current_turn + 1
        elif turn is None:
            turn = self._current_turn
        if normalized_role == "user" and supersedes in (None, (), [], "") and "instead" in content.lower():
            previous = self.last_user_ask(open_only=True)
            if previous is not None:
                supersedes = (str(previous.get("ask_id") or previous.get("message_id") or ""),)
        ident = message_id or self._new_id("message")
        message_ask_id = ask_id
        if normalized_role == "user" and not message_ask_id:
            message_ask_id = ident
        return self.append(
            "user_message" if normalized_role == "user" else "assistant_message",
            payload={"role": normalized_role, "content": content}, actor=normalized_role,
            turn=turn, message_id=ident, ask_id=message_ask_id, content=content,
            entities=entities, prompted_by=prompted_by, satisfies=satisfies,
            supersedes=supersedes, status=status or ("OPEN" if message_ask_id else ""),
            persist=persist,
        )

    def record_decision(self, decision_id: str, content: object, *, turn: int | None = None,
                        entities: object = (), prompted_by: object = (),
                        payload: object = None, persist: bool = True) -> dict[str, object]:
        if not prompted_by:
            latest = self.last_user_ask(open_only=False)
            prompted_by = ((latest.get("message_id") or latest.get("event_id")),) if latest else ()
        body = payload if payload is not None else {"decision_id": decision_id, "content": content}
        return self.append(
            "decision", body, actor="agent", turn=turn, event_id=decision_id,
            content=content, entities=entities, prompted_by=prompted_by,
            status="ADOPTED", persist=persist,
        )

    def record_stage_change(self, stage: str, *, turn: int | None = None,
                            entities: object = (), payload: object = None,
                            persist: bool = True) -> dict[str, object]:
        return self.append("stage_change", payload or {"stage": stage}, actor="host",
                           turn=turn, content=stage, entities=entities, status="CHANGED",
                           persist=persist)

    def _journal_path(self) -> Path | None:
        path = getattr(self.journal, "path", None)
        return Path(path) if path else None

    def _ingest_journal_record(self, record: Mapping[str, object]) -> None:
        source_id = _text(record.get("event_id"))
        if source_id and source_id in self._journal_record_ids:
            return
        if source_id:
            self._journal_record_ids.add(source_id)
        kind = _text(record.get("kind") or "event")
        if kind == "timeline_event":
            raw = record.get("timeline_event")
            if not isinstance(raw, Mapping):
                return
            event = TimelineEvent.from_mapping(raw, source_event_id=source_id or None)
            self._append_local(event)
            return
        projected_kind = _event_type(kind)
        if projected_kind == "event" and kind in {"run_started", "attempt_terminated"}:
            projected_kind = "stage_change"
        # Keep journal history queryable, while frame relevance excludes generic
        # audit noise unless the selector explicitly chooses it.
        event = self._make_event(
            projected_kind, actor=_text(record.get("actor") or record.get("producer") or "system"),
            turn=record.get("turn", record.get("step")), timestamp_ns=record.get("timestamp_ns"),
            event_id=f"{self.session_id}:journal:{source_id or self._new_id('journal')}",
            message_id=(_text(record.get("message_id")) or None), ask_id=None,
            content=_line_payload(record), entities=record.get("entities", ()),
            prompted_by=record.get("prompted_by", record.get("prompt_message_id", ())),
            satisfies=record.get("satisfies", ()), supersedes=(),
            changed_entities=record.get("changed_entities", ()),
            status=_text(record.get("status", record.get("result", ""))),
            payload=record, source_event_id=source_id or None,
        )
        self._append_local(event)

    def refresh(self) -> int:
        """Project journal suffix; returns newly indexed source records."""
        path = self._journal_path()
        if path is not None and path.is_file():
            size = path.stat().st_size
            if size < self._journal_offset:
                raise TimelineError("journal was truncated while rebuilding timeline")
            added = 0
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(self._journal_offset)
                while True:
                    line = handle.readline()
                    if not line:
                        break
                    self._journal_offset = handle.tell()
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except (TypeError, ValueError, UnicodeError) as error:
                        raise TimelineError("journal contains invalid JSON") from error
                    if isinstance(record, Mapping):
                        if (record.get("kind") == "timeline_event"
                                and isinstance(record.get("timeline_event"), Mapping)):
                            # AuditJournal externalizes oversized fields; hydrate this
                            # projection payload before treating it as a timeline event.
                            from kernel_measurement import resolve_journal_blob_refs

                            timeline_event = resolve_journal_blob_refs(
                                record["timeline_event"], path)
                            if not isinstance(timeline_event, Mapping):
                                raise TimelineError("journal timeline event is not an object")
                            record = {**record, "timeline_event": timeline_event}
                        before = len(self._journal_record_ids)
                        self._ingest_journal_record(record)
                        added += int(len(self._journal_record_ids) > before)
            return added
        records = getattr(self.journal, "records", None) if self.journal is not None else None
        if callable(records):
            added = 0
            for record in records():
                if isinstance(record, Mapping):
                    before = len(self._journal_record_ids)
                    self._ingest_journal_record(record)
                    added += int(len(self._journal_record_ids) > before)
            return added
        return 0

    def get(self, event_id: str) -> dict[str, object] | None:
        self.refresh()
        event = self._by_id.get(_text(event_id))
        return event.as_dict() if event is not None else None

    def by_turn(self, turn: int) -> tuple[dict[str, object], ...]:
        self.refresh()
        return tuple(self._by_id[ident].as_dict() for ident in self._by_turn.get(turn, ()))

    def by_time(self, start_ns: int, end_ns: int | None = None) -> tuple[dict[str, object], ...]:
        self.refresh()
        end = start_ns if end_ns is None else end_ns
        if type(start_ns) is not int or type(end) is not int or start_ns < 0 or end < start_ns:
            raise TimelineError("time range must be non-negative and ordered")
        first_bucket = start_ns // 1_000_000_000
        last_bucket = end // 1_000_000_000
        event_ids = [event_id for bucket, bucket_ids in self._by_time_bucket.items()
                     if first_bucket <= bucket <= last_bucket for event_id in bucket_ids]
        result = [self._by_id[event_id] for event_id in event_ids
                  if start_ns <= self._by_id[event_id].timestamp_ns <= end]
        result.sort(key=lambda event: (event.timestamp_ns, self._position_by_id[event.event_id]))
        return tuple(event.as_dict() for event in result)

    def by_entity(self, entity: str) -> tuple[dict[str, object], ...]:
        self.refresh()
        ident = _normalise_entity(entity)
        return tuple(self._by_id[event_id].as_dict() for event_id in self._by_entity.get(ident, ()))

    def by_message(self, message: str) -> tuple[dict[str, object], ...]:
        self.refresh()
        ident = _text(message)
        if ident in self._by_message_id:
            ids = self._by_message_id[ident]
        else:
            token_lists = [self._by_message_token[token] for token in _tokens(message)
                           if token in self._by_message_token]
            if token_lists:
                common = set(token_lists[0])
                for token_ids in token_lists[1:]:
                    common.intersection_update(token_ids)
                ids = [event_id for event_id in token_lists[0] if event_id in common]
            else:
                ids = []
        return tuple(self._by_id[event_id].as_dict() for event_id in ids)

    def timeline(self, range: object = None, *, since: object = None,
                 about: object = None, message: object = None,
                 limit: int | None = None) -> tuple[dict[str, object], ...]:
        """Typed query: ``range``, ``since``, ``about`` entity, or message."""
        self.refresh()
        if limit is not None and (type(limit) is not int or limit < 1):
            raise TimelineError("timeline limit must be a positive integer")
        selected: list[TimelineEvent]
        if about is not None:
            ids = self._by_entity.get(_normalise_entity(_text(about)), ())
            selected = [self._by_id[event_id] for event_id in ids]
        elif message is not None:
            selected = [TimelineEvent.from_mapping(row) for row in self.by_message(_text(message))]
        else:
            selected = list(self._events[-limit:] if limit is not None and since is None
                            and range is None and isinstance(self._events, HistorySpool)
                            else self._events)
        if since is not None:
            if type(since) is not int or since < 0:
                raise TimelineError("since must be a non-negative timestamp or turn")
            # Small integers are useful turn cursors; nanosecond values are time cursors.
            if since < 1_000_000_000_000:
                selected = [event for event in selected if event.turn > since]
            else:
                selected = [event for event in selected if event.timestamp_ns >= since]
        if range is not None:
            start: int
            end: int
            if isinstance(range, Mapping):
                start = int(range.get("start", range.get("start_ns", range.get("start_turn", 0))))
                end = int(range.get("end", range.get("end_ns", range.get("end_turn", start))))
                time_range = "start_ns" in range or "end_ns" in range
            elif isinstance(range, Sequence) and not isinstance(range, (str, bytes, bytearray)) and len(range) == 2:
                start, end = int(range[0]), int(range[1])
                time_range = False
            else:
                raise TimelineError("range must be a two-item sequence or mapping")
            if start < 0 or end < start:
                raise TimelineError("timeline range must be ordered")
            if time_range:
                selected = [event for event in selected if start <= event.timestamp_ns <= end]
            else:
                selected = [event for event in selected if start <= event.turn <= end]
        if limit is not None:
            selected = selected[-limit:]
        return tuple(event.as_dict() for event in selected)

    def last_user_ask(self, *, open_only: bool = False) -> dict[str, object] | None:
        self.refresh()
        for event_id in reversed(self._user_asks):
            event = self._by_id[event_id]
            ask_id = event.ask_id or event.message_id or event.event_id
            status = self._ask_status.get(ask_id, event.status or "OPEN")
            if open_only and status != "OPEN":
                continue
            row = event.as_dict()
            row["status"] = status
            return row
        return None

    def open_asks(self, *, limit: int = DEFAULT_OPEN_ASKS) -> tuple[dict[str, object], ...]:
        self.refresh()
        if type(limit) is not int or limit < 1:
            raise TimelineError("open ask limit must be positive")
        result: list[dict[str, object]] = []
        for event_id in reversed(self._user_asks):
            event = self._by_id[event_id]
            ask_id = event.ask_id or event.message_id or event.event_id
            if self._ask_status.get(ask_id, event.status or "OPEN") != "OPEN":
                continue
            row = event.as_dict()
            row["status"] = "OPEN"
            result.append(row)
            if len(result) >= limit:
                break
        result.reverse()
        return tuple(result)

    def what_changed_since(self, turn: int) -> tuple[dict[str, object], ...]:
        if type(turn) is not int or turn < 0:
            raise TimelineError("turn must be a non-negative integer")
        self.refresh()
        changed = {"action", "receipt", "decision", "stage_change"}
        rows: list[TimelineEvent] = []
        for event_turn in sorted(key for key in self._by_turn if key > turn):
            for event_id in self._by_turn[event_turn]:
                event = self._by_id[event_id]
                if event.kind in changed or event.changed_entities:
                    rows.append(event)
        return tuple(event.as_dict() for event in rows)

    def _candidate_rows(self, focus: object, *, limit: int | None = None) -> list[dict[str, object]]:
        focus_entities = set(_extract_entities(focus))
        if focus_entities:
            ids: list[str] = []
            seen: set[str] = set()
            for entity in focus_entities:
                for event_id in self._by_entity.get(entity, ()):
                    if event_id not in seen:
                        seen.add(event_id)
                        ids.append(event_id)
            events = (self._by_id[event_id] for event_id in ids)
        elif (limit is not None and 0 < limit <= _RELEVANCE_HEAD_SIZE
              and hasattr(self, '_relevance_head')):
            # A bounded metadata heap avoids hydrating journal-only payloads.
            # Global source position preserves the legacy stable tie order.
            events = (self._events[position] for _, _, position in sorted(self._relevance_head)[-limit:])
        else:
            events = (event for event in self._events
                      if event.kind in {"action", "receipt", "decision", "stage_change"})
        if limit is not None:
            # The default reader uses a stable sorted suffix. Include source
            # position in ties so selecting before payload copies is identical.
            selected = heapq.nlargest(limit, enumerate(events), key=lambda item: (
                item[1].turn, item[1].timestamp_ns, item[0]))
            events = (event for _, event in reversed(selected))
        rows: list[dict[str, object]] = []
        for event in events:
            row = event.as_dict()
            row["id"] = event.event_id
            row["gist"] = _compact(event.content or event.payload, MAX_RELEVANT_CHARS)
            row["links"] = {"files": list(event.entities), "requirements": [], "obligations": [],
                             "symbols": [], "hypotheses": [], "stages": []}
            row["token_cost"] = max(1, (len(row["gist"]) + 3) // 4)
            rows.append(row)
        return rows

    def _select_relevant(self, focus: object, *, max_items: int,
                         budget_tokens: int) -> tuple[dict[str, object], ...]:
        limit = (max_items if isinstance(self._events, HistorySpool) and max_items > 0
                 and self.selector is None and not callable(getattr(self.recall, "select", None))
                 else None)
        candidates = self._candidate_rows(focus, limit=limit)
        if self.selector is not None and candidates:
            selected = self.selector(candidates, focus=focus, max_items=max_items,
                                     budget_tokens=budget_tokens, ledger=self.decision_graph)
            if hasattr(selected, "items"):
                selected = getattr(selected, "items")
            if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes, bytearray)):
                return tuple(dict(item) for item in selected if isinstance(item, Mapping))[:max_items]
        if self.recall is not None and callable(getattr(self.recall, "select", None)):
            view = self.recall.select(focus, budget_tokens=max(32, budget_tokens), max_items=max_items,
                                      ledger=self.decision_graph, consume_requested=False,
                                      track_shown=False)
            rows = []
            for item in getattr(view, "items", ()):
                row = dict(item)
                row["source"] = "recall"
                rows.append(row)
            return tuple(rows[:max_items])
        return tuple(sorted(candidates, key=lambda row: (int(row.get("turn", 0)),
                                                          int(row.get("timestamp_ns", 0))))[-max_items:])

    def frame_slice(self, focus: object = None, *, recent_turns: int = DEFAULT_RECENT_TURNS,
                    max_open_asks: int = DEFAULT_OPEN_ASKS,
                    max_relevant: int = DEFAULT_RELEVANT_ITEMS,
                    budget_tokens: int = DEFAULT_FRAME_TOKENS) -> TimelineFrame:
        self.refresh()
        if type(recent_turns) is not int or recent_turns < 1:
            raise TimelineError("recent_turns must be positive")
        if type(max_open_asks) is not int or max_open_asks < 1:
            raise TimelineError("max_open_asks must be positive")
        if type(max_relevant) is not int or max_relevant < 0:
            raise TimelineError("max_relevant must be non-negative")
        if type(budget_tokens) is not int or budget_tokens < 32:
            raise TimelineError("budget_tokens must be at least 32")
        budget_tokens = min(budget_tokens, MAX_FRAME_TOKENS)
        latest = self.last_user_ask(open_only=False)
        latest_content = _text(latest.get("content")) if latest else None
        latest_id = (_text(latest.get("message_id")) if latest and latest.get("message_id") else None)
        asks = self.open_asks(limit=max_open_asks)
        turns = sorted(self._by_turn if isinstance(self._events, HistorySpool)
                       else {event.turn for event in self._events}, reverse=True)[:recent_turns]
        recent_lines: list[str] = []
        for turn in reversed(turns):
            rows = (self._by_id[event_id] for event_id in self._by_turn.get(turn, ()))
            bits = []
            for event in rows:
                detail = event.content or _compact(event.payload, 100)
                bits.append(f"{event.actor}/{event.kind}: {_compact(detail, 100)}")
                if (isinstance(self._events, HistorySpool)
                        and len(" ".join((f"turn {turn}: " + " | ".join(bits)).split())) > MAX_RECENT_TURN_CHARS):
                    # The normalized prefix fixes visible bytes and ellipsis.
                    break
            recent_lines.append(_compact(f"turn {turn}: " + " | ".join(bits), MAX_RECENT_TURN_CHARS))
        relevant = self._select_relevant(focus, max_items=max_relevant,
                                         budget_tokens=max(32, budget_tokens // 2)) if max_relevant else ()
        lines = [f"[TIMELINE schema={SCHEMA} events={len(self._events)}]", "LATEST USER MESSAGE"]
        if latest_content is not None:
            # Deliberately do not compact this line: this is the one verbatim user slice.
            lines.append(latest_content)
        else:
            lines.append("<none>")
        lines.append("OPEN ASKS")
        lines.extend(f"{row.get('ask_id') or row.get('message_id')}: {_compact(row.get('content'), MAX_ASK_CHARS)}"
                     for row in asks)
        if not asks:
            lines.append("<none>")
        lines.append("RECENT TURNS")
        lines.extend(recent_lines or ["<none>"])
        lines.append("SELECTOR RELEVANT")
        for row in relevant:
            lines.append(f"{row.get('id', row.get('event_id', '?'))}: "
                         f"{_compact(row.get('gist', row.get('content', row.get('payload', ''))), MAX_RELEVANT_CHARS)}")
        if not relevant:
            lines.append("<none>")
        lines.append("Use timeline(...) or recall(id) for omitted history.")
        rendered = "\n".join(lines)
        return TimelineFrame(
            latest_user_message=latest_content, latest_user_message_id=latest_id,
            open_asks=tuple(asks), recent_turns=tuple(recent_lines),
            relevant=tuple(relevant), rendered=rendered,
            used_tokens=max(1, (len(rendered) + 3) // 4), event_count=len(self._events),
        )

    def render_frame(self, focus: object = None, **kwargs: object) -> str:
        return self.frame_slice(focus, **kwargs).rendered

    def why(self, decision_id: str, *, limit: int = 8) -> dict[str, object]:
        self.refresh()
        graph_why = getattr(self.decision_graph, "why", None)
        if callable(graph_why):
            try:
                result = graph_why(decision_id)
                if isinstance(result, Mapping):
                    return dict(result)
            except (KeyError, TypeError, ValueError):
                # Timeline events remain a compatible fallback for legacy
                # decisions that predate the typed decision graph.
                pass
        decision = self.get(decision_id)
        if decision is None:
            raise TimelineError(f"unknown decision: {decision_id}")
        prompt_ids = [str(item) for item in decision.get("prompted_by", ()) if item]
        prompted: list[dict[str, object] | None] = []
        for reference in prompt_ids:
            direct = self.get(reference)
            if direct is not None:
                prompted.append(direct)
                continue
            message_events = self._by_message_id.get(reference, ())
            prompted.append(self._by_id[message_events[-1]].as_dict() if message_events else None)
        if not prompted:
            same_turn = [event for event in self._events
                         if event.turn == int(decision.get("turn", 0))
                         and event.kind == "user_message"]
            prompted = [event.as_dict() for event in same_turn[-1:]]
        entity = next(iter(decision.get("entities", ())), None)
        related = self.timeline(about=entity, limit=limit) if entity else self.timeline(
            range=(max(0, int(decision.get("turn", 0)) - 1), int(decision.get("turn", 0))), limit=limit)
        return {"decision": decision, "prompted_by": tuple(item for item in prompted if item),
                "related": related, "relevant": self._select_relevant(decision, max_items=limit, budget_tokens=240)}

    def revisit(self, decision_id: str, *, limit: int = 12) -> dict[str, object]:
        graph_revisit = getattr(self.decision_graph, "revisit", None)
        if callable(graph_revisit):
            try:
                result = graph_revisit(decision_id)
                if isinstance(result, Mapping):
                    return dict(result)
            except (KeyError, TypeError, ValueError):
                pass
        explanation = self.why(decision_id, limit=limit)
        decision = explanation["decision"]
        if not isinstance(decision, Mapping):
            raise TimelineError("decision projection is malformed")
        return {
            "decision": decision,
            "why": explanation,
            "changed_since_turn": self.what_changed_since(int(decision.get("turn", 0))),
            "open_asks": self.open_asks(limit=limit),
        }

    def snapshot(self) -> dict[str, object]:
        """Return projection health only; journal remains the durable event store."""
        self.refresh()
        return {"schema": SCHEMA, "session_id": self.session_id, "event_count": len(self._events),
                "latest_turn": self._current_turn,
                "latest_timestamp_ns": self._events[-1].timestamp_ns if self._events else None,
                "indexed_entities": len(self._by_entity), "open_asks": len(self.open_asks())}


def dataclass_replace(event: TimelineEvent, **changes: object) -> TimelineEvent:
    values = event.as_dict()
    values.update(changes)
    return TimelineEvent.from_mapping(values)


__all__ = [
    "DEFAULT_FRAME_TOKENS", "DEFAULT_OPEN_ASKS", "DEFAULT_RECENT_TURNS",
    "FRAME_SCHEMA", "SCHEMA", "TimelineError", "TimelineEvent", "TimelineFrame",
    "TimelineIndex",
]

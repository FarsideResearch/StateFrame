"""Deterministic ledger for model-visible organ advice.

The ledger is deliberately a shadow organ: it records delivery and attention
without deciding whether the advised code is correct or blocking completion.
Callers feed it the same action receipts that already pass through the kernel
receipt chokepoint, plus the exact line returned by ``organ_out``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TypeAlias


SCHEMA = "poc.kernel_advice_ledger/1"
DEFAULT_SETTLEMENT_TURNS = 2
MAX_REPEAT_ESCALATIONS = 1
Record: TypeAlias = Mapping[str, object]


class AdviceStatus(str, Enum):
    """Terminal classification for a delivered advisory row."""

    OPEN = "OPEN"
    FOLLOWED = "FOLLOWED"
    IGNORED = "IGNORED"
    OVERTAKEN = "OVERTAKEN"


@dataclass(frozen=True)
class AdviceTarget:
    """A deterministic file, test, or named atom that an action can touch."""

    kind: str
    value: str
    start_line: int | None = None
    end_line: int | None = None

    def __post_init__(self) -> None:
        kind = self.kind.strip().lower()
        value = self.value.strip()
        if kind not in {"file", "test", "atom"}:
            raise ValueError("advice target kind must be file, test, or atom")
        if not value:
            raise ValueError("advice target value must not be empty")
        start_line = _int(self.start_line) if self.start_line is not None else None
        end_line = _int(self.end_line) if self.end_line is not None else start_line
        if start_line is not None and start_line < 1:
            raise ValueError("advice target start_line must be positive")
        if end_line is not None and end_line < (start_line or 1):
            raise ValueError("advice target end_line must not precede start_line")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "value", _normal_path(value) if kind != "atom" else value)
        object.__setattr__(self, "start_line", start_line)
        object.__setattr__(self, "end_line", end_line)

    @classmethod
    def from_value(cls, value: object) -> "AdviceTarget":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            kind = _text(value.get("kind") or value.get("type"))
            raw = value.get("value") or value.get("ref") or value.get("target")
            if not kind:
                if value.get("test") or value.get("nodeid"):
                    kind = "test"
                elif value.get("file") or value.get("path"):
                    kind = "file"
                else:
                    kind = "atom"
            if not raw and kind == "file":
                raw = value.get("file") or value.get("path")
            if not raw and kind == "test":
                raw = value.get("test") or value.get("nodeid")
            if not raw and kind == "atom":
                raw = value.get("atom")
            return cls(
                kind,
                _text(raw),
                _int(value.get("start_line"), 0) or None,
                _int(value.get("end_line"), 0) or None,
            )
        raw = _text(value)
        lowered = raw.lower()
        for prefix in ("file:", "test:", "atom:"):
            if lowered.startswith(prefix):
                return cls(prefix[:-1], raw[len(prefix) :])
        if "::" in raw or lowered.startswith(("test_", "tests/", "test/")):
            return cls("test", raw)
        if ".py" in lowered or "/" in raw:
            return cls("file", raw)
        return cls("atom", raw)


@dataclass
class AdviceRecord:
    """One model-visible delivery and its append-only classification metadata."""

    organ: str
    turn: int
    fingerprint: str
    advisory_hash: str
    advisory: str
    targets: tuple[AdviceTarget, ...]
    world_revision: int = 0
    status: AdviceStatus = AdviceStatus.OPEN
    classified_turn: int | None = None
    ignored_count: int = 0
    late_match: bool = False
    _elsewhere_seen: bool = field(default=False, repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "organ": self.organ,
            "turn": self.turn,
            "fingerprint": self.fingerprint,
            "advisory_hash": self.advisory_hash,
            "advisory": self.advisory,
            "targets": [
                {
                    "kind": target.kind,
                    "value": target.value,
                    "start_line": target.start_line,
                    "end_line": target.end_line,
                }
                for target in self.targets
            ],
            "world_revision": self.world_revision,
            "status": self.status.value,
            "classified_turn": self.classified_turn,
            "ignored_count": self.ignored_count,
            "late_match": self.late_match,
        }


def _text(value: object) -> str:
    return str(value or "").strip()


def _normal_path(value: str) -> str:
    return value.replace("\\", "/").strip().lstrip("./")


def _hash(value: object, *, length: int = 16) -> str:
    if isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:length]


def advisory_fingerprint(
    organ: str,
    targets: Iterable[AdviceTarget | Mapping[str, object] | str],
    *,
    kind: str = "advisory",
) -> str:
    """Return stable identity for targetable advice, excluding prose and turn."""

    normalized = tuple(
        sorted(
            (
                target.kind,
                target.value.split("@", 1)[0] if target.kind != "atom" else target.value,
            )
            for target in (AdviceTarget.from_value(item) for item in targets)
        )
    )
    return _hash({"organ": _text(organ), "kind": _text(kind), "targets": normalized})


def compute_advisory_hash(advisory: str) -> str:
    """Hash rendered advisory content separately from its stable target fingerprint."""

    return _hash(" ".join(advisory.split()))


advisory_hash = compute_advisory_hash


def _int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _action_text(action: Record) -> str:
    parts: list[str] = []
    for key in ("kind", "intent", "path", "file", "target", "test", "nodeid", "atom", "refs", "why", "detail"):
        value = action.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            parts.extend(_text(item) for item in value)
        elif value is not None:
            parts.append(_text(value))
    return " ".join(parts)


def _action_path(action: Record) -> str:
    for key in ("path", "file", "target_file"):
        value = _text(action.get(key))
        if value:
            return _normal_path(value.split("@", 1)[0])
    intent = _text(action.get("intent"))
    match = re.search(r"([\w./-]+\.py)(?:@|\b)", intent)
    return _normal_path(match.group(1)) if match else ""


def _line_range(action: Record) -> tuple[int | None, int | None]:
    start = action.get("start_line", action.get("line"))
    end = action.get("end_line", start)
    return (_int(start, -1) if start is not None else None, _int(end, -1) if end is not None else None)


def _target_matches(target: AdviceTarget, action: Record) -> bool:
    if target.kind == "atom":
        needle = target.value.casefold()
        return needle in _action_text(action).casefold()
    if target.kind == "test":
        action_text = _action_text(action).casefold()
        return target.value.casefold() in action_text
    path = _action_path(action)
    if not path:
        return False
    expected = target.value.split("@", 1)[0].casefold()
    actual = path.casefold()
    if not (actual == expected or actual.endswith("/" + expected) or expected.endswith("/" + actual)):
        return False
    start, end = _line_range(action)
    target_match = re.search(r"@(?P<start>\d+)(?:-(?P<end>\d+))?$", target.value)
    target_start = target.start_line
    target_end = target.end_line
    if target_match is not None:
        target_start = int(target_match.group("start"))
        target_end = int(target_match.group("end") or target_start)
    if target_start is None or target_end is None or start is None or end is None:
        return True
    return start <= target_end and end >= target_start


def _action_matches(record: AdviceRecord, action: Record) -> bool:
    return any(_target_matches(target, action) for target in record.targets)


def _world_changed(action: Record, record: AdviceRecord) -> bool:
    if bool(action.get("world_changed")) or bool(action.get("overtakes_advice")):
        return True
    revision = action.get("world_revision", action.get("workspace_revision"))
    return revision is not None and _int(revision, record.world_revision) > record.world_revision


def render_repeat_advisory(advisory: str, previous_turn: int) -> str:
    """Prepend the single bounded repeat marker required by the attention law."""

    return f"SECOND NOTICE (previously advised turn {previous_turn}): {advisory}".strip()


class AdviceLedger:
    """In-memory ledger; callers persist ``rows`` and ``trace`` with receipts."""

    def __init__(self, *, settlement_turns: int = DEFAULT_SETTLEMENT_TURNS) -> None:
        if settlement_turns < 2:
            raise ValueError("settlement_turns must be at least 2")
        self.settlement_turns = settlement_turns
        self._rows: list[AdviceRecord] = []
        self._ignored_by_fingerprint: dict[str, list[AdviceRecord]] = {}
        self._escalated: set[str] = set()
        self._escalation_messages: dict[str, str] = {}
        self._trace: list[dict[str, object]] = []

    @property
    def rows(self) -> tuple[AdviceRecord, ...]:
        return tuple(self._rows)

    @property
    def trace(self) -> tuple[dict[str, object], ...]:
        return tuple(self._trace)

    @property
    def escalations(self) -> int:
        return len(self._escalated)

    @property
    def escalation_messages(self) -> Mapping[str, str]:
        """Rendered repeat notices keyed by fingerprint, each emitted once."""

        return dict(self._escalation_messages)

    def deliver(
        self,
        organ: str,
        turn: int,
        advisory: str,
        targets: Iterable[AdviceTarget | Mapping[str, object] | str],
        *,
        fingerprint: str | None = None,
        advisory_hash: str | None = None,
        kind: str = "advisory",
        world_revision: int = 0,
        visible: bool = True,
    ) -> AdviceRecord | None:
        """Record exactly one row for each delivered advisory.

        An advisory without named targets is retained for delivery accounting,
        but cannot be classified from action footprints.
        """

        if not visible or not _text(advisory):
            return None
        normalized_targets = tuple(AdviceTarget.from_value(item) for item in targets)
        stable_fingerprint = fingerprint or advisory_fingerprint(organ, normalized_targets, kind=kind)
        row = AdviceRecord(
            organ=_text(organ),
            turn=_int(turn),
            fingerprint=stable_fingerprint,
            advisory_hash=advisory_hash or compute_advisory_hash(advisory),
            advisory=advisory,
            targets=normalized_targets,
            world_revision=max(0, _int(world_revision)),
        )
        self._rows.append(row)
        self._event("ADVICE_DELIVERED", row, turn=row.turn)
        return row

    record_delivery = deliver
    record_advisory = deliver
    issue = deliver

    def observe_action(self, turn: int, action: Record) -> tuple[str, ...]:
        """Classify open rows against one later action in deterministic order."""

        current_turn = _int(turn)
        events: list[str] = []
        for row in self._rows:
            if row.status is not AdviceStatus.OPEN or current_turn <= row.turn:
                if row.status is not AdviceStatus.OPEN and current_turn > row.turn and _action_matches(row, action):
                    row.late_match = True
                continue
            if _action_matches(row, action):
                self._classify(row, AdviceStatus.FOLLOWED, current_turn)
                events.append("ADVICE_FOLLOWED")
            elif _world_changed(action, row):
                self._classify(row, AdviceStatus.OVERTAKEN, current_turn)
                events.append("ADVICE_OVERTAKEN")
            else:
                row._elsewhere_seen = True
                if current_turn - row.turn >= self.settlement_turns:
                    self._classify(row, AdviceStatus.IGNORED, current_turn)
                    events.append("ADVICE_IGNORED")
        return tuple(events)

    def tick(self, turn: int, *, world_revision: int | None = None) -> tuple[str, ...]:
        """Settle rows only after an unrelated action has been observed."""

        current_turn = _int(turn)
        events: list[str] = []
        for row in self._rows:
            if row.status is not AdviceStatus.OPEN or not row._elsewhere_seen:
                continue
            if current_turn - row.turn >= self.settlement_turns:
                if world_revision is not None and world_revision > row.world_revision:
                    self._classify(row, AdviceStatus.OVERTAKEN, current_turn)
                    events.append("ADVICE_OVERTAKEN")
                else:
                    self._classify(row, AdviceStatus.IGNORED, current_turn)
                    events.append("ADVICE_IGNORED")
        return tuple(events)

    def render_repeat_advisory(self, fingerprint: str) -> str | None:
        """Render once after the same stable advisory identity is ignored twice."""

        ignored = self._ignored_by_fingerprint.get(fingerprint, [])
        if fingerprint in self._escalated:
            return self._escalation_messages[fingerprint]
        if len(ignored) < 2:
            return None
        self._escalated.add(fingerprint)
        current = ignored[-1]
        previous = ignored[-2]
        text = render_repeat_advisory(current.advisory, previous.turn)
        self._escalation_messages[fingerprint] = text
        self._event("ADVICE_ESCALATED", current, turn=current.classified_turn or current.turn)
        return text

    def census(self) -> dict[str, int]:
        """Return the compact advice-ledger census contract."""

        return {
            "rows": len(self._rows),
            "followed": sum(row.status is AdviceStatus.FOLLOWED for row in self._rows),
            "ignored": sum(row.status is AdviceStatus.IGNORED for row in self._rows),
            "escalated": len(self._escalated),
        }

    def summary(self) -> dict[str, object]:
        return {"schema": SCHEMA, "settlement_turns": self.settlement_turns, **self.census()}

    def _classify(self, row: AdviceRecord, status: AdviceStatus, turn: int) -> None:
        row.status = status
        row.classified_turn = turn
        if status is AdviceStatus.IGNORED:
            ignored = self._ignored_by_fingerprint.setdefault(row.fingerprint, [])
            ignored.append(row)
            row.ignored_count = len(ignored)
            self._event("ADVICE_IGNORED", row, turn=turn)
            if len(ignored) == MAX_REPEAT_ESCALATIONS + 1:
                self.render_repeat_advisory(row.fingerprint)
        else:
            self._event(f"ADVICE_{status.value}", row, turn=turn)

    def _event(self, event: str, row: AdviceRecord, *, turn: int) -> None:
        self._trace.append(
            {
                "organ": row.organ,
                "event": event,
                "turn": turn,
                "fingerprint": row.fingerprint,
                "advisory_hash": row.advisory_hash,
            }
        )


IgnoredAdviceTracker = AdviceLedger


def load_receipts(path: str | Path) -> tuple[Record, ...]:
    """Load valid JSON object rows using the liveness receipt idiom."""

    records: list[Record] = []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ()
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            records.append(value)
    return tuple(records)


def _ledger_trigger(evidence: object) -> object:
    event = getattr(evidence, "event", {})
    if not isinstance(event, Mapping):
        return "UNKNOWN"
    return "YES" if _text(event.get("event")).startswith("ADVICE_") else "NO"


try:
    from kernel_liveness import OrganCensusSpec

    CENSUS_SPEC = OrganCensusSpec(
        organ_id="advice_ledger",
        trigger=_ledger_trigger,
        fire_events=frozenset({"ADVICE_DELIVERED", "ADVICE_FOLLOWED", "ADVICE_IGNORED", "ADVICE_OVERTAKEN"}),
        suppression_events=frozenset(),
        dedup_events=frozenset(),
        challenge_once_events=frozenset({"ADVICE_ESCALATED"}),
    )
    CENSUS_SPECS = (CENSUS_SPEC,)
except ImportError:  # pragma: no cover - standalone import outside repository
    CENSUS_SPEC = None
    CENSUS_SPECS: tuple[object, ...] = ()


__all__ = [
    "AdviceLedger",
    "AdviceRecord",
    "AdviceStatus",
    "AdviceTarget",
    "CENSUS_SPEC",
    "CENSUS_SPECS",
    "DEFAULT_SETTLEMENT_TURNS",
    "IgnoredAdviceTracker",
    "MAX_REPEAT_ESCALATIONS",
    "SCHEMA",
    "advisory_fingerprint",
    "advisory_hash",
    "load_receipts",
    "render_repeat_advisory",
]

"""Artifact keys, view ids, per-artifact state and the measured rung choice.

Pure data: no runtime, no I/O. The diet/finalizer/dispatch wiring (B5) owns a
`ViewLadderState` on the runtime, stages changes while a request is being built and commits
them at `commit_after_response` (rollback on a failed send). Rendered bytes are recorded once
per `(rung, vid)` and never change afterwards (stability rule: render once, re-emit verbatim).

The model may promote or demote an artifact's view at any time (`read(view=)`, `keep`), so
this module records transitions but does not enforce a one-way descent; the age-based ladder
is the engine's fallback, not a constraint on the model.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import NamedTuple

from kernel_view_ladder.render import RUNGS, rung_level, sha256_text, view_id

STATE_FORMAT_VERSION = 1
SMALL_ARTIFACT_BYTES = 4096


class ArtifactKey(NamedTuple):
    """`A = (path_display, revision, held_start, held_end)`; equality uses `path_cmp`."""

    path_display: str
    revision: str
    held_start: int
    held_end: int

    @property
    def path_cmp(self) -> str:
        return os.path.normcase(os.path.normpath(self.path_display))

    def same_artifact(self, other: "ArtifactKey") -> bool:
        return (self.path_cmp, self.revision, self.held_start, self.held_end) == \
            (other.path_cmp, other.revision, other.held_start, other.held_end)

    def to_json(self) -> list[object]:
        return [self.path_display, self.revision, self.held_start, self.held_end]

    @classmethod
    def from_json(cls, value: Sequence[object]) -> "ArtifactKey":
        return cls(str(value[0]), str(value[1]), int(value[2]), int(value[3]))


@dataclass
class Keep:
    level: str
    range: tuple[int, int] | None
    turn: int


@dataclass
class ArtifactState:
    rung: str = "full"
    reason: str = "context_limit"
    keep: Keep | None = None
    frozen_selection: tuple[tuple[object, ...], ...] | None = None
    rendered: dict[str, tuple[str, str]] = field(default_factory=dict)  # rung -> (sha256, text)
    held_text: str | None = None
    note: dict[str, object] | None = None
    stale: bool = False
    disk_revision_seen: str | None = None
    fresh_turn: int = 0

    def record_render(self, rung: str, text: str) -> str:
        """Store rendered bytes once; a second render of the same rung must be byte-identical."""
        digest = sha256_text(text)
        existing = self.rendered.get(rung)
        if existing is not None and existing[0] != digest:
            raise ValueError(f"rung {rung!r} already rendered with different bytes")
        self.rendered[rung] = (digest, text)
        return digest

    def set_rung(self, rung: str, reason: str) -> None:
        rung_level(rung)
        self.rung, self.reason = rung, reason
        if rung == "stub":
            self.held_text = None  # R10: drop held text once nothing verbatim is on the wire

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        data["keep"] = None if self.keep is None else {"level": self.keep.level,
                                                        "range": list(self.keep.range) if self.keep.range else None,
                                                        "turn": self.keep.turn}
        data["frozen_selection"] = None if self.frozen_selection is None else [list(r) for r in self.frozen_selection]
        data["rendered"] = {rung: [sha, text] for rung, (sha, text) in sorted(self.rendered.items())}
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "ArtifactState":
        keep_raw = data.get("keep")
        keep = None
        if isinstance(keep_raw, Mapping):
            rng = keep_raw.get("range")
            keep = Keep(str(keep_raw["level"]), (int(rng[0]), int(rng[1])) if rng else None, int(keep_raw["turn"]))
        frozen_raw = data.get("frozen_selection")
        frozen = None if frozen_raw is None else tuple(
            tuple(int(p) if i < 2 else str(p) for i, p in enumerate(row)) for row in frozen_raw)  # type: ignore[union-attr]
        rendered_raw = data.get("rendered") or {}
        rendered = {str(rung): (str(pair[0]), str(pair[1])) for rung, pair in rendered_raw.items()}  # type: ignore[union-attr]
        return cls(rung=str(data.get("rung", "full")), reason=str(data.get("reason", "context_limit")), keep=keep,
                   frozen_selection=frozen, rendered=rendered,
                   held_text=data.get("held_text") if isinstance(data.get("held_text"), str) else None,
                   note=dict(data["note"]) if isinstance(data.get("note"), Mapping) else None,
                   stale=bool(data.get("stale", False)),
                   disk_revision_seen=str(data["disk_revision_seen"]) if data.get("disk_revision_seen") else None,
                   fresh_turn=int(data.get("fresh_turn", 0)))

    def vid(self, key: ArtifactKey, rung: str | None = None, reason: str | None = None) -> str:
        return view_id(display=key.path_display, revision=key.revision, held_start=key.held_start,
                       held_end=key.held_end, rung=rung or self.rung, reason=reason or self.reason,
                       frozen_selection=self.frozen_selection)


class ViewLadderState:
    """Committed artifact states plus a pending overlay applied at commit time."""

    def __init__(self) -> None:
        self.entries: dict[ArtifactKey, ArtifactState] = {}
        self.pending: dict[ArtifactKey, ArtifactState] = {}

    def get(self, key: ArtifactKey) -> ArtifactState | None:
        return self.pending.get(key, self.entries.get(key))

    def stage(self, key: ArtifactKey, state: ArtifactState) -> None:
        self.pending[key] = state

    def commit(self) -> None:
        self.entries.update(self.pending)
        self.pending.clear()

    def rollback(self) -> None:
        self.pending.clear()

    def to_json(self) -> str:
        rows = [{"key": key.to_json(), "state": state.to_json()}
                for key, state in sorted(self.entries.items(), key=lambda item: item[0].to_json())]
        return json.dumps({"format": STATE_FORMAT_VERSION, "entries": rows}, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "ViewLadderState":
        data = json.loads(text)
        if int(data.get("format", 0)) != STATE_FORMAT_VERSION:
            raise ValueError(f"unsupported view ladder state format {data.get('format')!r}")
        out = cls()
        for row in data["entries"]:
            out.entries[ArtifactKey.from_json(row["key"])] = ArtifactState.from_json(row["state"])
        return out


# --- measured rung choice (P2 / P5 / P11) -----------------------------------------------------

Measure = tuple[int, int]  # (wire bytes, tokens)
Meter = Callable[[str], Measure]


def default_meter(text: str) -> Measure:
    """Bytes and a rough token count; the diet injects its own `RequestTokenMeter` instead."""
    data = text.encode("utf-8", errors="surrogatepass")
    return len(data), max(1, (len(data) + 3) // 4)


def candidate_rungs(renders: Mapping[str, str], *, current: str, meter: Meter = default_meter,
                    keep_floor: str | None = None) -> list[str]:
    """Rungs below `current` that are applicable (present in `renders`) AND cheaper.

    A candidate must save both bytes and tokens against the current view (P11) and against
    the previous surviving candidate (P2: nothing larger than the rung above it survives), so
    the result is strictly decreasing in both measures, best content first. When every lower
    rung costs more than the current view the list is empty and the artifact stays as it is
    (P5: `full` is the terminal fallback for a one-line file).
    """
    if current not in renders:
        raise ValueError("renders must include the current rung")
    floor = rung_level(keep_floor) if keep_floor else rung_level("stub")
    bar = meter(renders[current])
    out: list[str] = []
    for rung in RUNGS:
        if rung_level(rung) <= rung_level(current) or rung_level(rung) > floor or rung not in renders:
            continue
        cost = meter(renders[rung])
        if cost[0] < bar[0] and cost[1] < bar[1]:
            out.append(rung)
            bar = cost
    return out


def pick_rung(renders: Mapping[str, str], *, current: str, deficit_bytes: int, deficit_tokens: int,
              meter: Meter = default_meter, keep_floor: str | None = None) -> str | None:
    """Highest candidate whose saving covers the deficit, else the smallest candidate, else None."""
    candidates = candidate_rungs(renders, current=current, meter=meter, keep_floor=keep_floor)
    if not candidates:
        return None
    base = meter(renders[current])
    for rung in candidates:
        cost = meter(renders[rung])
        if base[0] - cost[0] >= deficit_bytes and base[1] - cost[1] >= deficit_tokens:
            return rung
    return candidates[-1]

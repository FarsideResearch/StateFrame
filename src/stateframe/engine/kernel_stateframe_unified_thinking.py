"""Deterministic unified task state arbitration for the existing F4 notices.

This is deliberately a presentation seam, not a second source of truth. The
existing notice/card producers still decide what text is available; when
``STATEFRAME_UNIFIED_THINKING`` is enabled this module chooses at most one of
those already-rendered candidates for the current turn.

The disabled arm is never called by the runtime hook. Its state is local to a
runtime, keyed by product revision, and records only presentation fingerprints
and exposure counts.
"""
from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


MODE_ENV = "STATEFRAME_UNIFIED_THINKING"
BUDGET_ENV = "STATEFRAME_UNIFIED_THINKING_BUDGET"
SCHEMA = "stateframe-unified-thinking/1"
DEFAULT_NOTICE_BUDGET = 600
ADVISORY_EXPOSURE_LIMIT = 2

# Higher values win after change/newness has been considered.  Counterexamples
# are the strongest existing F4 notice; first-green and coverage are advisory
# cards, with unknown existing notices retained at a conservative priority.
_PRIORITY = {
    "counterexample": 100,
    "first_green": 90,
    "coverage": 70,
    "method_card": 40,
    "notice": 20,
}
_ADVISORY_KINDS = frozenset({"first_green", "coverage", "method_card", "notice"})


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_unified_thinking_mode", None)
    return (str(configured).strip().lower() == "on"
            if configured is not None else mode() == "on")


def notice_budget(value: object = None) -> int:
    raw = os.environ.get(BUDGET_ENV, str(DEFAULT_NOTICE_BUDGET)) if value is None else value
    try:
        budget = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{BUDGET_ENV} must be a positive integer") from exc
    if budget <= 0:
        raise ValueError(f"{BUDGET_ENV} must be a positive integer")
    return budget


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _turn(runtime: object, explicit: object = None) -> int:
    value = explicit
    if value in (None, ""):
        value = getattr(getattr(runtime, "owner", None), "current_step", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _revision(runtime: object, frame: Mapping[str, object] | None) -> str:
    # Fix (post-review r44, finding 1): the ordinary live path never sets any
    # of these top-level fields on `frame` or `runtime` -- `binding()` nests
    # the real value under `product_revision` inside its OWN dict, and the
    # live frame only carries `artifact_revisions` (a dict of hashes, not a
    # single field under any of these names). Reset-for-revision therefore
    # always resolved "unknown" on the live path, retaining suppression
    # across real edits whenever the notice text stayed identical.
    for source in (frame, runtime):
        if not isinstance(source, Mapping) and source is None:
            continue
        for name in ("product_revision", "current_revision", "revision", "workspace_revision"):
            value = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
            if value not in (None, ""):
                return str(value)
    # Fall back to the SAME revision authority the notice-dedup organ already
    # uses (live window / done-challenge reducers): it hashes real in-scope
    # artifact bytes (never receipts/conversation text), so it actually
    # changes when a real edit lands.
    try:
        from kernel_stateframe_notice_dedup import _revision as _dedup_revision
        return _dedup_revision(runtime)
    except ImportError:
        return "unknown"


def _tokens(text: str) -> int:
    try:
        from kernel_context import token_estimate
        return int(token_estimate({"role": "user", "content": text}))
    except (ImportError, TypeError, ValueError):
        return max(1, (len(text.encode("utf-8")) + 3) // 4)


def _fingerprint(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _kind(text: str) -> str:
    lowered = text.lower()
    if text.startswith("GREEN LOCK / open counterexamples disclosure:"):
        return "first_green"
    if text.startswith("[unverified checks]") or lowered.startswith("unverified:"):
        return "counterexample"
    if (text.startswith("[Named-layer coverage")
            or text.startswith("[Coverage reminders")
            or lowered.startswith("checklist:")):
        return "coverage"
    if "advisory method" in lowered or text.startswith("[StateFrame advisory sidecar]"):
        return "method_card"
    return "notice"


@dataclass(frozen=True)
class NoticeCandidate:
    key: str
    kind: str
    text: str
    priority: int
    tokens: int
    fingerprint: str
    changed: bool
    exposures: int
    index: int

    @property
    def advisory(self) -> bool:
        return self.kind in _ADVISORY_KINDS


def _new_state() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "revision": None,
        "history": {},
        "selection": None,
        "selection_signature": (),
        "selection_turn": None,
        "selection_budget": None,
    }


def _state(runtime: object) -> dict[str, object]:
    state = getattr(runtime, "_unified_thinking_state", None)
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        state = _new_state()
        setattr(runtime, "_unified_thinking_state", state)
    return state


def _reset_for_revision(state: dict[str, object], revision: str) -> None:
    if state.get("revision") == revision:
        return
    state.update({
        "revision": revision,
        "history": {},
        "selection": None,
        "selection_signature": (),
        "selection_turn": None,
        "selection_budget": None,
    })


def _runtime_budget(runtime: object, explicit: object = None) -> int:
    if explicit is not None:
        return notice_budget(explicit)
    configured = getattr(runtime, "stateframe_unified_thinking_budget", None)
    return notice_budget(configured) if configured is not None else notice_budget()


def _signature(candidates: Sequence[NoticeCandidate]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((candidate.key, candidate.fingerprint) for candidate in candidates))


def _candidate_list(runtime: object, notices: Sequence[object],
                    state: Mapping[str, object]) -> list[NoticeCandidate]:
    history = state.get("history") if isinstance(state.get("history"), Mapping) else {}
    counts: dict[str, int] = {}
    result: list[NoticeCandidate] = []
    for index, raw in enumerate(notices):
        text = _text(raw)
        if not text:
            continue
        kind = _kind(text)
        ordinal = counts.get(kind, 0)
        counts[kind] = ordinal + 1
        key = f"{kind}:{ordinal}"
        fingerprint = _fingerprint(text)
        previous = history.get(key) if isinstance(history, Mapping) else None
        changed = not isinstance(previous, Mapping) or previous.get("fingerprint") != fingerprint
        exposures = 0 if changed else max(0, int(previous.get("exposures", 0) or 0))
        result.append(NoticeCandidate(
            key=key, kind=kind, text=text, priority=_PRIORITY[kind],
            tokens=_tokens(text), fingerprint=fingerprint, changed=changed,
            exposures=exposures, index=index,
        ))
    return result


def _suppressed(candidate: NoticeCandidate) -> bool:
    return (candidate.advisory and not candidate.changed
            and candidate.exposures >= ADVISORY_EXPOSURE_LIMIT)


def _remember_selection(state: dict[str, object], candidate: NoticeCandidate,
                        signature: tuple[tuple[str, str], ...], turn: int,
                        budget: int) -> None:
    history = state.setdefault("history", {})
    if not isinstance(history, dict):
        history = {}
        state["history"] = history
    history[candidate.key] = {
        "fingerprint": candidate.fingerprint,
        "kind": candidate.kind,
        "exposures": candidate.exposures + 1,
        "last_turn": turn,
        "tokens": candidate.tokens,
    }
    state.update({
        "selection": {
            "key": candidate.key,
            "kind": candidate.kind,
            "tokens": candidate.tokens,
            "priority": candidate.priority,
            "changed": candidate.changed,
            "turn": turn,
        },
        "selection_signature": signature,
        "selection_turn": turn,
        "selection_budget": budget,
    })


def select_notices(runtime: object, notices: Sequence[object], *,
                   frame: Mapping[str, object] | None = None,
                   turn: object = None, token_budget: object = None,
                   provider_request: bool = True) -> list[str]:
    """Choose at most one existing notice/card for this turn.

    New or changed candidates are considered before unchanged candidates; within
    each class the fixed kind priority wins, followed by stable input order.
    An over-budget candidate is skipped so a lower-priority fitting candidate
    can still be delivered.  Advisory candidates stop repeating after two
    unchanged exposures in one product revision.

    Fix (post-review r44, finding 2): selection is PROVISIONAL for
    ``provider_request=False`` calls (queued native actions rendered for
    logging, never actually dispatched to the model). Only a
    ``provider_request=True`` call commits exposure/history -- two
    successive non-provider turns must never suppress an advisory the model
    was never shown.
    """
    if not enabled(runtime):
        return list(notices)
    state = _state(runtime)
    revision = _revision(runtime, frame)
    _reset_for_revision(state, revision)
    budget = _runtime_budget(runtime, token_budget)
    current_turn = _turn(runtime, turn)
    candidates = _candidate_list(runtime, notices, state)
    current_keys = {candidate.key for candidate in candidates}
    history = state.get("history")
    if isinstance(history, dict):
        for key in tuple(history):
            if key not in current_keys:
                del history[key]
    signature = _signature(candidates)

    cached = state.get("selection")
    if (provider_request
            and isinstance(cached, Mapping)
            and state.get("selection_turn") == current_turn
            and state.get("selection_budget") == budget
            and tuple(state.get("selection_signature") or ()) == signature):
        selected_key = str(cached.get("key", ""))
        return [candidate.text for candidate in candidates if candidate.key == selected_key]

    eligible = [candidate for candidate in candidates
                if not _suppressed(candidate) and candidate.tokens <= budget]
    eligible.sort(key=lambda candidate: (
        0 if candidate.changed else 1,
        -candidate.priority,
        candidate.exposures,
        candidate.index,
    ))
    selected = eligible[0] if eligible else None
    if not provider_request:
        # Provisional: compute and return for display, but never touch the
        # persisted selection/history -- a later real provider_request call
        # (same or a later turn) must see this candidate as never delivered.
        return [] if selected is None else [selected.text]
    state.update({
        "selection": None,
        "selection_signature": signature,
        "selection_turn": current_turn,
        "selection_budget": budget,
    })
    if selected is None:
        return []
    _remember_selection(state, selected, signature, current_turn, budget)
    return [selected.text]


# Names used by small callers and probes; the implementation remains one path.
choose_notices = select_notices
metacontrol = select_notices


def state_snapshot(runtime: object) -> dict[str, object]:
    """Return an immutable, JSON-friendly diagnostic projection."""
    state = _state(runtime)
    return copy.deepcopy({
        "schema": SCHEMA,
        "revision": state.get("revision"),
        "selection_turn": state.get("selection_turn"),
        "selection_budget": state.get("selection_budget"),
        "selected": state.get("selection"),
        "history": state.get("history", {}),
    })


__all__ = [
    "ADVISORY_EXPOSURE_LIMIT", "BUDGET_ENV", "DEFAULT_NOTICE_BUDGET", "MODE_ENV",
    "NoticeCandidate", "SCHEMA", "choose_notices", "enabled", "metacontrol", "mode",
    "notice_budget", "select_notices", "state_snapshot",
]

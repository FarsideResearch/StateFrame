"""Pure presentation helpers for TOKENS continuity and concise action guidance."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from kernel_tokens_row6_state import RepeatEvent, RepeatState

CONCISE_ACTION_GUIDANCE = (
    "Use current host facts. Carry settled facts while dependencies are current. For a read-only step, "
    "state only the unresolved question and request the needed observation. For a small edit, state changed "
    "behavior and affected invariants; omit unchanged requirements. Batch independent receipt-bound reads or "
    "edits only. Re-plan on new evidence; no separate planning turn."
)
CONCISE_FINAL_GUIDANCE = (
    "Summarize changed behavior, actual checks, and remaining blockers once; cite existing receipts "
    "instead of retelling the plan. Preserve uncertainty and requested deliverables."
)
CONTINUITY_START = "[TOKENS row6 continuity]"
CONTINUITY_END = "[/TOKENS row6 continuity]"
REPEAT_START = "[TOKENS row6 repeat advice]"
REPEAT_END = "[/TOKENS row6 repeat advice]"
GUIDANCE_START = "[TOKENS row6 concise guidance]"
GUIDANCE_END = "[/TOKENS row6 concise guidance]"
MAX_EXTENSION_BYTES = 1200


def _text(value: object) -> str:
    return str(value).strip() if isinstance(value, (str, int, float, bool)) else ""


def _clip(value: object, limit: int) -> str:
    text = " ".join(_text(value).split())
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _ids(value: object, limit: int = 8) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        value = value.keys()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(sorted(dict.fromkeys(_text(item) for item in value if _text(item))))[:limit]


def _within_budget(text: str, budget: int) -> bool:
    return len(text.encode("utf-8")) <= max(128, min(budget, MAX_EXTENSION_BYTES))


def guidance_enabled(policy: object) -> bool:
    """Only policy-owned guidance can enable this optional text."""
    guidance_for = getattr(policy, "guidance_for", None)
    if not callable(guidance_for):
        return False
    try:
        return bool(guidance_for("concise_action"))
    except (AttributeError, TypeError, ValueError):
        return False


def render_guidance(policy: object, *, final: bool = False, budget: int = 600) -> str:
    if not guidance_enabled(policy):
        return ""
    text = CONCISE_FINAL_GUIDANCE if final else CONCISE_ACTION_GUIDANCE
    return text if _within_budget(text, budget) else ""


def render_guidance_block(policy: object, *, final: bool = False, budget: int = 700) -> str:
    text = render_guidance(policy, final=final, budget=budget)
    if not text:
        return ""
    return f"{GUIDANCE_START}\n{text}\n{GUIDANCE_END}"


def render_continuity(snapshot: Mapping[str, object], *, budget: int = MAX_EXTENSION_BYTES) -> str:
    """Render only host-receipted facts plus one explicitly unverified intention."""
    closed = snapshot.get("last_closed")
    closed_row = closed if isinstance(closed, Mapping) else {}
    receipt = _text(closed_row.get("receipt_id"))
    action = _text(closed_row.get("tool"))
    result = _text(closed_row.get("result"))
    effect = _text(closed_row.get("effect")) or "not recorded"
    if receipt and action and result:
        last = f"Last closed action: {action} -> {result}; effect: {_clip(effect, 140)}; receipt {receipt}."
    else:
        last = "Last closed action: no receipt-backed action is available."
    disposition = _text(snapshot.get("current_disposition")) or "not recorded"
    check = _text(snapshot.get("current_check")) or "not recorded"
    open_ids = _ids(snapshot.get("open_clause_ids"))
    clauses = ", ".join(open_ids) if open_ids else "none recorded"
    intention = _clip(snapshot.get("pending_intention"), 180) or "not supplied"
    dependency = _clip(snapshot.get("dependency_revision"), 100) or "UNKNOWN"
    text = (
        f"{CONTINUITY_START}\n"
        f"{last}\n"
        f"Current check/refusal: {_clip(check, 180)}; disposition: {_clip(disposition, 120)}.\n"
        f"Open requirement IDs: {clauses}.\n"
        f"Pending model intention (UNVERIFIED): {intention}.\n"
        f"Dependencies current at: {dependency}.\n"
        f"{CONTINUITY_END}"
    )
    return text if _within_budget(text, budget) else ""


def render_repeat_advice(event: RepeatEvent, *, open_clause_ids: Sequence[str] = (),
                         repair: str = "existing reachable repair", budget: int = 600) -> str:
    if not event.detected or not event.guidance_enabled:
        return ""
    ids = ", ".join(_ids(open_clause_ids)) or "none recorded"
    text = (
        f"{REPEAT_START}\n"
        f"Repeated refused {event.kind.replace('_', ' ')} detected from settled evidence; "
        f"open requirement IDs: {ids}.\n"
        f"Reachable repair/witness: {_clip(repair, 180)}. No acceptance is implied.\n"
        f"{REPEAT_END}"
    )
    return text if _within_budget(text, budget) else ""


def _replace_owned_block(content: str, start: str, end: str, replacement: str) -> str:
    begin = content.find(start)
    if begin < 0:
        return content
    finish = content.find(end, begin + len(start))
    if finish < 0:
        return content
    finish += len(end)
    return content[:begin] + replacement + content[finish:]


def replace_handoff_extension(handoff: Mapping[str, object], extension: str, *, budget: int = 1800,
                              marker: tuple[str, str] = (CONTINUITY_START, CONTINUITY_END)
                              ) -> dict[str, object]:
    """Replace/append only our handoff block; overflow returns the original mapping."""
    if not extension or not isinstance(handoff, Mapping):
        return dict(handoff)
    content = _text(handoff.get("content"))
    if not content.startswith("[Decision handoff]"):
        return dict(handoff)
    start, end = marker
    if start in content:
        updated = _replace_owned_block(content, start, end, extension)
    else:
        insertion = content.find("\nDefeasible:")
        updated = (content[:insertion] + "\n" + extension + content[insertion:]
                   if insertion >= 0 else content.rstrip() + "\n" + extension + "\n")
    if (not _within_budget(extension, budget)
            or len(updated.encode("utf-8")) - len(content.encode("utf-8"))
            > max(128, min(budget, MAX_EXTENSION_BYTES))):
        return dict(handoff)
    result = dict(handoff)
    result["content"] = updated
    return result


def audit_owned_blocks(messages: Sequence[Mapping[str, object]]) -> dict[str, int | bool]:
    """Count only our blocks, allowing offline checks to prove other panels survive."""
    continuity = 0
    repeat = 0
    guidance = 0
    for message in messages:
        content = _text(message.get("content")) if isinstance(message, Mapping) else ""
        continuity += int(CONTINUITY_START in content and CONTINUITY_END in content)
        repeat += int(REPEAT_START in content and REPEAT_END in content)
        guidance += int(GUIDANCE_START in content and GUIDANCE_END in content)
    return {"continuity_blocks": continuity, "repeat_blocks": repeat,
            "guidance_blocks": guidance,
            "at_most_one": continuity <= 1 and repeat <= 1 and guidance <= 1}


def continuity_snapshot(state: RepeatState, *, current_check: str = "",
                        current_disposition: str = "", open_clause_ids: Sequence[str] = (),
                        pending_intention: str = "", dependency_revision: str = "") -> dict[str, object]:
    closed = state.last_closed.to_dict() if state.last_closed else None
    return {
        "schema": "tokens-row6-continuity/1",
        "last_closed": closed,
        "current_check": _clip(current_check, 180),
        "current_disposition": _clip(current_disposition, 120),
        "open_clause_ids": list(_ids(open_clause_ids)),
        "pending_intention": _clip(pending_intention, 180),
        "dependency_revision": _clip(dependency_revision, 100),
        "run_id": state.run_id,
        "request_id": state.request_id,
    }


__all__ = [
    "CONCISE_ACTION_GUIDANCE", "CONCISE_FINAL_GUIDANCE", "CONTINUITY_END", "CONTINUITY_START",
    "GUIDANCE_END", "GUIDANCE_START", "MAX_EXTENSION_BYTES", "REPEAT_END", "REPEAT_START",
    "audit_owned_blocks",
    "continuity_snapshot", "guidance_enabled", "render_continuity", "render_guidance",
    "render_guidance_block", "render_repeat_advice", "replace_handoff_extension",
]

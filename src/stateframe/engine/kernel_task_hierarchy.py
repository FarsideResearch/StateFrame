"""Render the single declared user action as design guidance."""

_ACTION_LABEL = "immediate user action"
_MAX_ACTION_CHARS = 160


def design_action_cue(brief: str) -> str | None:
    action = None
    for line in brief.splitlines():
        label, separator, value = line.partition(":")
        if not separator or label.strip().casefold() != _ACTION_LABEL:
            continue
        value = value.strip()
        if action is not None or not value or len(value) > _MAX_ACTION_CHARS or not value.isprintable():
            return None
        action = value

    if action is None:
        return None
    return (
        f"[Design focus]\n{action}\n"
        "Give the existing entry point for this action the strongest visual emphasis "
        "and preserve its destination."
    )

"""One-factor intervention screen support (tooling only; no product-path hook).

The board item asks to re-enable ``KERNEL_INTERVENTIONS`` with ONE isolated
mechanism per screen.  This module owns the two things that make such a screen
honest:

* ``arm_environment`` builds the COMPLETE environment of an arm from one
  explicit shared base: the master switch on, all seven gated consumers OFF in
  the control, exactly one ``BOTH`` in a candidate, presentation settings
  (``history_cap`` / ``render_selection``) copied unchanged from the ambient
  configuration into every arm, and the master-gated extra
  ``KERNEL_ACTION_MENU_GUARD`` pinned off.  Nothing else differs between arms.
* ``CONSUMER_EVENTS`` names the journal event each consumer emits when it
  actually runs, so "fired" is read from the consumer's own decision, never
  from a marker the screen wrote itself.  A consumer with no such event in a
  run is ``NOT_EXERCISED`` for that run.

The runtime and request bytes are untouched by this module, so the product path
is byte-identical whether or not the screen exists.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence


SCHEMA = "intervention-hygiene-screen/2"
ACTIVE_MODE = "BOTH"
MECHANISMS = (
    "self_oracle",
    "navigation",
    "retry_predispatch",
    "sidekick",
    "pilot",
    "tool_applicability",
    "public_review",
)
PRESENTATION_FACTORS = ("history_cap", "render_selection")
# Master-gated behaviour that is NOT one of the seven factors: pinned identical
# (off) in every arm so it can never masquerade as a factor effect.
PINNED_EXTRAS = {"KERNEL_ACTION_MENU_GUARD": "0"}
_MODES = frozenset({"OFF", "SHADOW", "ADVISE", "ENFORCE", "BOTH"})


def normalize_mechanism(value: object) -> str:
    if not isinstance(value, str) or value.strip().lower() not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {', '.join(MECHANISMS)}")
    return value.strip().lower()


def presentation_entries(ambient_kernel_mode: str) -> tuple[str, ...]:
    """The ambient presentation settings, preserved verbatim for every arm."""
    kept = []
    for pair in str(ambient_kernel_mode or "").split(","):
        name, separator, value = pair.strip().partition(":")
        if not separator:
            continue
        name, value = name.strip(), value.strip().upper()
        if name in PRESENTATION_FACTORS:
            if value not in _MODES:
                raise ValueError(f"invalid ambient KERNEL_MODE entry: {pair}")
            kept.append(f"{name}:{value}")
    return tuple(kept)


def arm_environment(selected: object | None, ambient: Mapping[str, str]) -> dict[str, str]:
    """Complete child environment for the control (``None``) or one candidate.

    Every arm is derived from the SAME ambient mapping; the ambient master
    switch, legacy ``KERNEL_SHADOW`` and any ambient factor modes are replaced,
    so a master-on or master-off caller produces identical arms.
    """
    chosen = None if selected is None else normalize_mechanism(selected)
    env = dict(ambient)
    factors = [f"{name}:{ACTIVE_MODE if name == chosen else 'OFF'}" for name in MECHANISMS]
    env.update({
        "KERNEL_INTERVENTIONS": "1",
        "KERNEL_SHADOW": "",
        "KERNEL_MODE": ",".join([*presentation_entries(ambient.get("KERNEL_MODE", "")),
                                 *factors]),
        **PINNED_EXTRAS,
    })
    return env


def arm_factor_modes(manifest_modes: Mapping[str, object]) -> dict[str, str]:
    """The seven factor modes as the RUNTIME resolved them (effective config)."""
    return {name: str(manifest_modes.get(name, "MISSING")) for name in MECHANISMS}


def audit_arm(selected: str | None, manifest_modes: Mapping[str, object],
              presentation: Mapping[str, object]) -> list[str]:
    """Problems with an arm's EFFECTIVE configuration ([] when isolated)."""
    problems = []
    modes = arm_factor_modes(manifest_modes)
    active = [name for name, mode in modes.items() if mode != "OFF"]
    expected = [] if selected is None else [selected]
    if active != expected:
        problems.append(f"effective active factors {active} != expected {expected}")
    for name in PRESENTATION_FACTORS:
        if str(manifest_modes.get(name)) != str(presentation.get(name)):
            problems.append(f"presentation factor {name} differs from the control")
    return problems


def _live_mode(event: Mapping[str, object]) -> bool:
    # Several consumers journal an evaluation in EVERY arm (the evaluate axis
    # runs even when OFF).  Measured: effect_decision_started and
    # predispatch_decision both appear in the all-OFF control.
    return str(event.get("mode", "")).upper() not in {"", "OFF"}


def _effect_boundary(component: str):
    """self_oracle / navigation: the B2 effect-boundary decision DELIVERED advice
    or APPLIED a state effect (an empty decision is not the consumer acting)."""
    def predicate(event: Mapping[str, object]) -> bool:
        decision = event.get("decision")
        return (event.get("kind") == "effect_decision" and isinstance(decision, Mapping)
                and decision.get("component") == component and _live_mode(decision)
                and bool(decision.get("returned_advice")
                         or decision.get("confirmed_applied_effect")
                         or decision.get("completed_effect_count")))
    return predicate


def _predispatch(event: Mapping[str, object]) -> bool:
    outcome = event.get("outcome")
    return (event.get("kind") == "predispatch_decision" and _live_mode(event)
            and isinstance(outcome, Mapping)
            and bool(outcome.get("blocks") or outcome.get("advice")))


def _sidekick(event: Mapping[str, object]) -> bool:
    finding = event.get("finding")
    return (event.get("kind") == "sidekick_finding" and _live_mode(event)
            and isinstance(finding, Mapping)
            and finding.get("finding") in {"CHALLENGE_ONCE", "INVALIDATE_ATTEMPT",
                                           "QUARANTINE_HARNESS"})


def _pilot_blocked(event: Mapping[str, object]) -> bool:
    # ParentMeter.charge journals every charge in EVERY arm; the pilot factor
    # only changes whether an exhausted budget is ENFORCED.
    return event.get("kind") == "parent_budget" and event.get("blocked") is True


def _tool_applicability(event: Mapping[str, object]) -> bool:
    return (event.get("kind") == "tool_applicability_decision" and _live_mode(event)
            and bool(event.get("removed_tools") or event.get("advice_selected")))


def _public_review(event: Mapping[str, object]) -> bool:
    # 'public_review_unexposed' is the consumer declining; only a review child
    # or selected advice is the consumer acting.
    return event.get("kind") in {"public_review_child", "public_review_advice_selected"}


def _evaluated(kind: str, component: str | None = None):
    def predicate(event: Mapping[str, object]) -> bool:
        if kind == "effect_decision_started":
            return (event.get("kind") == kind and event.get("component") == component
                    and _live_mode(event))
        if kind == "public_review_":
            return str(event.get("kind", "")).startswith(kind)
        if kind == "parent_budget":
            return event.get("kind") == kind
        return event.get("kind") == kind and _live_mode(event)
    return predicate


# The consumer ACTED (delivered advice, applied an effect, blocked, removed a tool).
CONSUMER_EVENTS = {
    "self_oracle": _effect_boundary("self_oracle"),
    "navigation": _effect_boundary("navigation"),
    "retry_predispatch": _predispatch,
    "sidekick": _sidekick,
    "pilot": _pilot_blocked,
    "tool_applicability": _tool_applicability,
    "public_review": _public_review,
}
# The consumer RAN under its live mode (reached its trigger), whether or not it acted.
CONSUMER_EVALUATIONS = {
    "self_oracle": _evaluated("effect_decision_started", "self_oracle"),
    "navigation": _evaluated("effect_decision_started", "navigation"),
    "retry_predispatch": _evaluated("predispatch_decision"),
    "sidekick": _evaluated("sidekick_finding"),
    "pilot": _evaluated("parent_budget"),
    "tool_applicability": _evaluated("tool_applicability_decision"),
    "public_review": _evaluated("public_review_"),
}


def consumer_evaluations(mechanism: str, events: Sequence[Mapping[str, object]]) -> list[dict]:
    predicate = CONSUMER_EVALUATIONS[normalize_mechanism(mechanism)]
    return [dict(event) for event in events if isinstance(event, Mapping) and predicate(event)]


def consumer_events(mechanism: str, events: Sequence[Mapping[str, object]]) -> list[dict]:
    predicate = CONSUMER_EVENTS[normalize_mechanism(mechanism)]
    return [dict(event) for event in events if isinstance(event, Mapping) and predicate(event)]


__all__ = [
    "ACTIVE_MODE", "CONSUMER_EVENTS", "MECHANISMS", "PINNED_EXTRAS", "PRESENTATION_FACTORS",
    "SCHEMA", "arm_environment", "arm_factor_modes", "audit_arm", "consumer_events",
    "normalize_mechanism", "presentation_entries", "CONSUMER_EVALUATIONS",
    "consumer_evaluations",
]

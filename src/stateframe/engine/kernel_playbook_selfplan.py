"""Optional, task-agnostic playbook cards for the D2 self-plan arm.

The card selector only sees the public task brief.  It is deliberately a small
host-side compiler: it can offer one compact method card, but it cannot make a
card a requirement or infer anything from task answers or private material.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kernel_boot_cards import ADVISORY_LABEL


MODE_ENV = "STATEFRAME_PLAYBOOK_SELFPLAN"
CARD_VERSION = "stateframe-playbook/1"
PLAYBOOK_HEADER = "[StateFrame playbook sidecar]"
PLAYBOOK_SCHEMA = "stateframe-playbook/1"


@dataclass(frozen=True)
class PlaybookCard:
    card_id: str
    title: str
    use_when: str
    steps: tuple[str, ...]
    predicate: str


_CARDS: tuple[PlaybookCard, ...] = (
    PlaybookCard(
        "P1", "Is it plugged in",
        "a setting, flag, mode, or no-op may be preventing the path",
        (
            "List the effective settings and switches that can select this path.",
            "Name the event that proves the path was reached.",
            "Only then inspect the path's logic.",
        ),
        r"\b(?:config|configuration|environment|env|flag|switch|setting|mode|effective)\b",
    ),
    PlaybookCard(
        "P2", "Find all the other ends",
        "a value crosses fields, messages, layers, readers, or writers",
        (
            "Write down the value's identity and every writer and reader.",
            "Check the producer, transport, consumer, and display spelling.",
            "Fix the first missing or mismatched end, then trace the value again.",
        ),
        r"\b(?:field|key|route|column|message|writer|reader|consumer|producer|layer|across)\b",
    ),
    PlaybookCard(
        "P3", "Optional field with a default, across layers",
        "an input may be omitted while old callers must keep working",
        (
            "Choose one default and define the field at the boundary.",
            "Parse, validate, store, and return that same field through each layer.",
            "Keep an old-caller test and prove the default is applied once.",
        ),
        r"\b(?:optional|default|omitted|backward|compatibility)\b",
    ),
    PlaybookCard(
        "P4", "Rename with a compatibility alias",
        "a public name must change without breaking an existing caller",
        (
            "Enumerate registries, imports, serialized names, and callers.",
            "Add the new name and a forwarding alias to one implementation.",
            "Test both names and record the alias's removal condition.",
        ),
        r"\b(?:rename|renamed|alias|backward\s+compatibility)\b",
    ),
    PlaybookCard(
        "P5", "Reproduce before you touch it",
        "a failure, regression, error, or surprising symptom is reported",
        (
            "Make the smallest reproducible case and record the observed result.",
            "Confirm the cause with one discriminating change.",
            "Apply the fix, rerun the reproduction, and state what remains uncertain.",
        ),
        r"\b(?:fix|bug|failure|failing|broken|regression|error|symptom)\b",
    ),
    PlaybookCard(
        "P6", "The counterexample you cannot explain",
        "an observation contradicts the current explanation",
        (
            "Stop and write the observation beside the theory it contradicts.",
            "Separate what was measured from what was inferred.",
            "Revise or narrow the theory; do not label a contradiction flaky without evidence.",
        ),
        r"\b(?:contradict|counterexample|wrong|unexpected|edge\s+case|disagree)\b",
    ),
    PlaybookCard(
        "P7", "Graveyard: find all the bodies",
        "a missing behavior may have sibling copies or a second owner",
        (
            "Search the whole tree for the semantic pattern, not one filename.",
            "Classify every hit as writer, reader, test, or dead copy.",
            "Fix all live owners or explain why each remaining hit is safe.",
        ),
        r"\b(?:everywhere|whole\s+tree|all\s+occurrences|sibling|other\s+end|duplicate)\b",
    ),
    PlaybookCard(
        "P8", "Should-be-non-null proof",
        "an expected effect is empty, absent, silent, or swallowed",
        (
            "Prove the producing call ran and record its returned value.",
            "Check the control path separately from the effect path.",
            "Treat an empty result as unknown until the boundary is observed.",
        ),
        r"\b(?:non[-\s]?null|nonnull|empty|silent|swallow|called|must\s+emit|must\s+write)\b",
    ),
    PlaybookCard(
        "P9", "Measured beats claimed",
        "a result, verification, test, or measurement is being discussed",
        (
            "Record the exact observation, command, and revision.",
            "Let the measurement decide whether the claim holds.",
            "If the test is green but the effect is absent, measure the effect directly.",
        ),
        r"\b(?:measure|measured|verify|verification|result|exit\s+code|test)\b",
    ),
    PlaybookCard(
        "P10", "Same code, same run",
        "a current result could be stale or tied to a different build",
        (
            "Record the revision, build identity, dependencies, and run identity.",
            "Check that the observed artifact came from the current code.",
            "Rerun after any identity change before comparing outcomes.",
        ),
        r"\b(?:current|revision|build|stale|same\s+run|re[-\s]?run|rerun)\b",
    ),
    PlaybookCard(
        "P11", "A red test is a question",
        "a failing assertion may expose a contract disagreement",
        (
            "State the intended contract before changing the assertion.",
            "Keep the test failing while the contract is clarified.",
            "Change the product or the contract loudly; never weaken a useful check silently.",
        ),
        r"\b(?:red\s+test|assertion|contract|test)\b",
    ),
    PlaybookCard(
        "P12", "Environment or work",
        "setup, import, dependency, timeout, or harness behavior is suspect",
        (
            "Classify the failure as environment, harness, or product behavior.",
            "Reproduce the setup failure with the same command a clean run uses.",
            "Repair the environment or CI path instead of changing product logic for it.",
        ),
        r"\b(?:setup|import|timeout|dependency|environment\s+failure|harness)\b",
    ),
    PlaybookCard(
        "P13", "Leave the read loop",
        "inspection is expanding without producing a discriminating fact",
        (
            "Set a bounded reading budget and name the edit or decision site.",
            "Ask one question whose resolution would change the next action.",
            "Stop rereading once it is resolved and act on the evidence.",
        ),
        r"\b(?:read|inspect|search|reread|loop)\b",
    ),
    PlaybookCard(
        "P14", "Minimal, explained diff",
        "a change, edit, implementation, or update needs a tight scope",
        (
            "Trace each changed file to the request and the observed defect.",
            "Make the smallest coherent edit that proves the intended behavior.",
            "Explain unusual deletions or compatibility choices and leave unrelated work intact.",
        ),
        r"\b(?:edit|change|diff|smallest|scope|delete|make|build|implement|create|update)\b",
    ),
    PlaybookCard(
        "P15", "Prior art first",
        "a helper, convention, or earlier fix may already exist",
        (
            "Search for an existing helper, contract, and nearby prior fix.",
            "Compare its assumptions with the current request.",
            "Extend the shared authority when it fits; record why a new path is needed otherwise.",
        ),
        r"\b(?:existing|previous|prior|reuse|helper|past\s+fix)\b",
    ),
)

_COMPILED_PREDICATES = {
    card.card_id: re.compile(card.predicate, re.IGNORECASE) for card in _CARDS
}
_ADOPTION_MARKER = re.compile(
    r"\bPLAYBOOK\s+(P\d{1,2})\s*[:=-]\s*(ADOPT|IGNORE)\b", re.IGNORECASE)


def playbook_mode() -> str:
    """Return the strict opt-in mode; all unset/invalid values are OFF."""

    return "on" if os.environ.get(MODE_ENV, "").strip() == "on" else "off"


def all_playbooks() -> tuple[PlaybookCard, ...]:
    return _CARDS


def _brief_text(brief: object) -> str:
    return " ".join(str(brief or "").split())


def predicate_states(brief: object) -> dict[str, str]:
    """Evaluate only allowlisted, public-brief predicates as true/false/unknown."""

    text = _brief_text(brief)
    if not text:
        return {card.card_id: "unknown" for card in _CARDS}
    return {
        card.card_id: ("true" if _COMPILED_PREDICATES[card.card_id].search(text)
                       else "false")
        for card in _CARDS
    }


def select_playbook(brief: object, states: Mapping[str, str] | None = None) -> PlaybookCard | None:
    """Select at most one card in stable library order."""

    resolved = dict(states) if states is not None else predicate_states(brief)
    for card in _CARDS:
        if resolved.get(card.card_id) == "true":
            return card
    return None


def card_text(card: PlaybookCard) -> str:
    steps = "\n".join(f"{index}. {step}" for index, step in enumerate(card.steps, 1))
    return (
        f"{ADVISORY_LABEL}\n"
        f"Name: {card.title}\n"
        f"Use when: {card.use_when}.\n"
        f"Method:\n{steps}\n"
        "This is optional guidance, not task evidence or a completion gate. "
        "It may be adopted, adapted, or ignored. "
        f"Optional choice marker: PLAYBOOK {card.card_id}: ADOPT or IGNORE."
    )


def build_packet(brief: object) -> dict[str, object]:
    mode = playbook_mode()
    states = predicate_states(brief)
    card = select_playbook(brief, states) if mode == "on" else None
    return {
        "schema": PLAYBOOK_SCHEMA,
        "mode": mode,
        "card_version": CARD_VERSION,
        "predicate_states": states,
        "selected_card": card.card_id if card is not None else None,
        "selected_title": card.title if card is not None else None,
        "selection_rule": (f"public_predicate:{card.card_id}" if card is not None else None),
        "card_text": card_text(card) if card is not None else "",
    }


def render_playbook_card(packet: Mapping[str, object], *, first_request: bool = True) -> str:
    if not first_request or packet.get("mode") != "on":
        return ""
    card_id = packet.get("selected_card")
    text = packet.get("card_text")
    if not isinstance(card_id, str) or not isinstance(text, str) or not text:
        return ""
    return f"{PLAYBOOK_HEADER}\n[PLAYBOOK {card_id} | {CARD_VERSION}]\n{text}"


def append_playbook_sidecar(messages: Sequence[Mapping[str, object]],
                            packet: Mapping[str, object], *,
                            first_request: bool = True) -> list[dict]:
    rendered = render_playbook_card(packet, first_request=first_request)
    result = [copy.deepcopy(dict(message)) for message in messages]
    if rendered:
        result.append({"role": "user", "content": rendered})
    return result


def _response_text(response: object) -> str:
    if isinstance(response, str):
        return response
    if not isinstance(response, Mapping):
        return ""
    pieces: list[str] = []
    for key in ("content", "reasoning_content", "text", "arguments"):
        value = response.get(key)
        if isinstance(value, str):
            pieces.append(value)
    message = response.get("message")
    if isinstance(message, Mapping):
        pieces.append(_response_text(message))
    tool_calls = response.get("tool_calls")
    if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes, bytearray)):
        for call in tool_calls:
            if isinstance(call, Mapping):
                pieces.append(_response_text(call))
    function = response.get("function")
    if isinstance(function, Mapping):
        pieces.append(_response_text(function))
    return "\n".join(piece for piece in pieces if piece)


def adoption_decision(card_id: str, response: object) -> tuple[str, str | None]:
    """Classify one explicit marker without treating silence as adoption."""

    markers = [
        (match.group(1).upper(), match.group(2).upper())
        for match in _ADOPTION_MARKER.finditer(_response_text(response))
    ]
    expected = str(card_id).upper()
    matching = [(found, choice) for found, choice in markers if found == expected]
    if matching:
        choices = {choice for _, choice in matching}
        if len(choices) != 1:
            return "ambiguous", f"{expected}: AMBIGUOUS"
        choice = next(iter(choices))
        return ("adopted" if choice == "ADOPT" else "ignored"), f"{expected}: {choice}"
    if markers:
        found, choice = markers[0]
        return "mismatched", f"{found}: {choice}"
    return "unmarked", None


def journal_offer(capture: object, packet: Mapping[str, object]) -> Mapping[str, object] | None:
    journal = getattr(capture, "journal", None)
    card_id = packet.get("selected_card")
    if journal is None or not isinstance(card_id, str):
        return None
    event = journal.emit(
        "playbook_card_offered",
        producer="kernel_playbook_selfplan",
        playbook_schema=PLAYBOOK_SCHEMA,
        playbook_id=card_id,
        card_version=packet.get("card_version", CARD_VERSION),
        selection_rule=packet.get("selection_rule"),
        predicate_states=copy.deepcopy(packet.get("predicate_states", {})),
        request_id=getattr(capture, "request_id", None),
        step=getattr(capture, "step", None),
        model_may_adopt_or_ignore=True,
        dispatch=False,
    )
    capture.playbook_offer_event_id = event.get("event_id") if isinstance(event, Mapping) else None
    return event


def journal_adoption(capture: object) -> Mapping[str, object] | None:
    if getattr(capture, "playbook_adoption_journaled", False):
        return None
    card_id = getattr(capture, "playbook_card_id", None)
    journal = getattr(capture, "journal", None)
    if journal is None or not isinstance(card_id, str):
        return None
    response = {
        "content": getattr(capture, "response_content", None),
        "reasoning_content": getattr(capture, "response_reasoning_content", None),
        "message": getattr(capture, "provider_response_message", None),
    }
    adoption, marker = adoption_decision(card_id, response)
    event = journal.emit(
        "playbook_adoption",
        producer="kernel_playbook_selfplan",
        playbook_schema=PLAYBOOK_SCHEMA,
        playbook_id=card_id,
        card_version=getattr(capture, "playbook_card_version", CARD_VERSION),
        adoption=adoption,
        adoption_marker=marker,
        adoption_source="explicit_marker" if marker else "no_marker",
        offer_event_id=getattr(capture, "playbook_offer_event_id", None),
        request_id=getattr(capture, "request_id", None),
        response_event_id=getattr(capture, "response_event_id", None),
        step=getattr(capture, "step", None),
        dispatch=False,
    )
    capture.playbook_adoption_journaled = True
    return event


__all__ = [
    "ADVISORY_LABEL",
    "CARD_VERSION",
    "MODE_ENV",
    "PLAYBOOK_HEADER",
    "PLAYBOOK_SCHEMA",
    "PlaybookCard",
    "adoption_decision",
    "all_playbooks",
    "append_playbook_sidecar",
    "build_packet",
    "card_text",
    "journal_adoption",
    "journal_offer",
    "playbook_mode",
    "predicate_states",
    "render_playbook_card",
    "select_playbook",
]

"""Host-compiled StateFrame boot packets and optional advisory method cards.

This module only uses public task-row fields and host-local observations.  It
does not inspect hidden tests, task identifiers, gold artifacts, or model
output.  The packet is deliberately plain text because the same rendering is
delivered to both the transcript and StateFrame request paths.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


MODE_ENV = "STATEFRAME_BOOT"
VALID_MODES = frozenset({"off", "boot", "cards"})
ADVISORY_LABEL = "OPTIONAL METHOD — not task evidence or a completion requirement"
CAPSULE_MAX_TOKENS = 400
ORIENTATION_CUE = (
    "[StateFrame orientation — first request only] This is compact host context, "
    "not task evidence. Start from the brief, inspect the available target files, "
    "and verify conclusions against public observations. It may become stale after "
    "a change; source and public execution remain authoritative. It proposes no "
    "action and adds no completion condition."
)

_CLAUSE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|(?<=[;:])\s+")
_NUMERIC_WORDS = re.compile(
    r"\b(?:units?|dimensions?|quantity|quantities|rate|rates|per\s+second|"
    r"per\s+minute|per\s+hour|seconds?|secs?|milliseconds?|milliseconds?|"
    r"minutes?|hours?|days?|duration|durations|elapsed|interval|intervals|"
    r"timeout|timeouts|distance|distances|speed|velocity|meters?|metres?|"
    r"kilometers?|kilometres?|miles?|bytes?|kilobytes?|megabytes?|gigabytes?|"
    r"tokens?|requests?|percent(?:age)?|milliseconds?|ms|sec|min|hr)\b|%|"
    r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|min|h|hr|hrs|d)\b",
    re.IGNORECASE,
)
_NUMERIC_SEMANTIC_WORDS = re.compile(
    r"\b(?:calculate|compute|convert|add|subtract|multiply|divide|refill|"
    r"consume|expire|expires|wait|waiting|remaining|deficit|capacity|"
    r"elapsed|duration|interval|timeout|distance|speed|velocity|unit|units|"
    r"per\s+(?:second|minute|hour)|rate|quantity|quantities)\b",
    re.IGNORECASE,
)
_FORMAT_ONLY_WORDS = re.compile(
    r"\b(?:format|formatting|formatted|display|label|labels|render|rendering|"
    r"pretty[- ]?print|serialize|serialization|spacing|whitespace|color|colou?r)\b",
    re.IGNORECASE,
)
_COMPONENT_ROLE_WORDS = re.compile(
    r"\b(?:class(?:es)?|interface(?:s)?|component(?:s)?|module(?:s)?|"
    r"service(?:s)?|parser(?:s)?|reporter(?:s)?|renderer(?:s)?|adapter(?:s)?|"
    r"queue(?:s)?|worker(?:s)?|client(?:s)?|repository(?:ies)?|"
    r"controller(?:s)?|model(?:s)?|view(?:s)?)\b",
    re.IGNORECASE,
)
_NAMED_COMPONENT = re.compile(
    r"\b(?:class|interface|component|module|service|parser|reporter|renderer|"
    r"adapter|queue|worker|client|repository|controller|model|view)\s+"
    r"[`\"']?([A-Za-z_][A-Za-z0-9_.-]*)",
    re.IGNORECASE,
)
_BACKTICK_NAME = re.compile(r"`([A-Z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)`")
_BARE_COMPONENT_NAME = re.compile(r"\b[A-Z][A-Za-z0-9_]{2,}\b")
_SECRET = re.compile(
    r"(?i)(\b(?:api[_-]?key|authorization|password|passwd|secret|token)\s*[=:]\s*)"
    r"[^\s,;]+"
)


def boot_mode() -> str:
    """Return the validated boot/card switch; the default is the old path."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in VALID_MODES:
        raise ValueError("STATEFRAME_BOOT must be off, boot or cards")
    return value


def _safe_text(value: object, *, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    text = _SECRET.sub(r"\1[redacted]", text)
    if len(text) > limit:
        text = text[: max(0, limit - 3)] + "..."
    return text


def _clauses(brief: str) -> tuple[dict[str, str], ...]:
    spans = [part.strip() for part in _CLAUSE_SPLIT.split(brief) if part.strip()]
    if not spans and brief.strip():
        spans = [brief.strip()]
    # The clause index is model-facing.  Keep its quotes host-derived while
    # applying the same public-text redaction used by environment facts.
    return tuple({"id": f"C{index}", "quote": _safe_text(span, limit=320)}
                 for index, span in enumerate(spans, 1))


def _numeric_semantics(brief: str) -> bool:
    if not _NUMERIC_WORDS.search(brief):
        return False
    # A RateLimiter/UI formatting request is a deliberate near-miss: a word
    # such as "rate" is not enough without a quantity operation or unit cue.
    strong_semantics = re.search(
        r"\b(?:calculate|compute|convert|add|subtract|multiply|divide|refill|"
        r"consume|expire|expires|wait|waiting|remaining|deficit|capacity|"
        r"elapsed|duration|interval|timeout|distance|speed|velocity|"
        r"per\s+(?:second|minute|hour))\b|"
        r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|min|h|hr|hrs|d)\b|%",
        brief, re.IGNORECASE)
    if _FORMAT_ONLY_WORDS.search(brief) and not strong_semantics:
        return False
    return bool(_NUMERIC_SEMANTIC_WORDS.search(brief) or
                re.search(r"\b(?:per\s+(?:second|minute|hour)|\d+\s*(?:ms|s|sec|min|h|hr))\b|%",
                          brief, re.IGNORECASE))


def _component_names(brief: str) -> tuple[str, ...]:
    names = []
    matched_roles = []
    stopwords = {"a", "an", "and", "as", "for", "from", "in", "of", "the",
                 "then", "to", "that", "this", "with"}
    for match in _NAMED_COMPONENT.finditer(brief):
        value = match.group(1).strip(".,:;()[]")
        role = match.group(0).split()[0].lower().rstrip("s")
        matched_roles.append(role)
        component = role if not value or value.lower() in stopwords else value
        if component.lower() not in {item.lower() for item in names}:
            names.append(component)
    for match in _BACKTICK_NAME.finditer(brief):
        value = match.group(1)
        if value.lower() not in {item.lower() for item in names}:
            names.append(value)
    roles = {match.group(0).lower().rstrip("s")
             for match in _COMPONENT_ROLE_WORDS.finditer(brief)}
    # "parser and reporter" is a public two-component signal even when the
    # brief omits class names.  A role already paired with a named class is
    # not counted twice ("class Foo" remains one component).
    for role in sorted(roles - set(matched_roles)):
        names.append(role)
    return tuple(names)


def _multi_component(brief: str) -> bool:
    named = _component_names(brief)
    excluded = {"Add", "Build", "Change", "Create", "Fix", "Implement", "Introduce",
                "Make", "Support", "The", "This", "Update"}
    bare = {match.group(0) for match in _BARE_COMPONENT_NAME.finditer(brief)
            if match.group(0) not in excluded}
    build_context = re.search(
        r"\b(?:add|build|compose|connect|create|implement|introduce|support|wire)\b",
        brief, re.IGNORECASE)
    return len(named) >= 2 or bool(build_context and len(bare) >= 2)


def routing_hints(brief: str) -> dict[str, bool]:
    """Apply only the public, conservative routing predicates."""
    text = brief if isinstance(brief, str) else ""
    return {
        "repair": bool(re.search(r"\b(?:fix|repair|bug|regression|broken|correct)\b",
                                 text, re.IGNORECASE)),
        "build": bool(re.search(r"\b(?:build|add|create|implement|introduce|support)\b",
                                text, re.IGNORECASE)),
        "numeric_semantics": _numeric_semantics(text),
        "multi_component": _multi_component(text),
    }


def select_method_card(brief_or_hints: str | Mapping[str, object],
                       *, mode: str | None = None) -> str | None:
    """Select at most one card, with dimensional reasoning taking precedence."""
    selected_mode = boot_mode() if mode is None else mode
    if selected_mode != "cards":
        return None
    hints = (routing_hints(brief_or_hints) if isinstance(brief_or_hints, str)
             else {key: (value is True or value == "true")
                   for key, value in brief_or_hints.items()})
    if hints.get("numeric_semantics"):
        return "dimensional_reasoning.md"
    if hints.get("multi_component"):
        return "component_composition.md"
    return None


def _card_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "method_cards" / filename


def load_method_card(filename: str) -> str:
    """Read a checked-in card; an absent asset fails closed to no card."""
    if filename not in {"dimensional_reasoning.md", "component_composition.md"}:
        raise ValueError("unknown StateFrame method card")
    path = _card_path(filename)
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    # Card files may carry their contract label for standalone inspection,
    # while the sidecar renderer owns the single wire label.  Normalize here
    # so a real prepared request never presents the advisory contract twice.
    lines = text.splitlines()
    if lines and lines[0].strip() == ADVISORY_LABEL:
        text = "\n".join(lines[1:]).strip()
    return text


def _target_file_list(target_files: object) -> tuple[str, ...]:
    values: list[str] = []
    if isinstance(target_files, Mapping):
        source = target_files.keys()
    elif isinstance(target_files, Sequence) and not isinstance(target_files, (str, bytes)):
        source = target_files
    else:
        source = (target_files,) if target_files else ()
    for item in source:
        if isinstance(item, Mapping):
            item = item.get("path")
        if isinstance(item, (str, Path)):
            value = _safe_text(item, limit=160)
            if value and value not in values:
                values.append(value)
    return tuple(values)


def _public_test_command(test_command: object, interpreter: str) -> tuple[str, str]:
    if isinstance(test_command, str) and test_command.strip():
        return _safe_text(test_command, limit=240), "task_row.test_command"
    return (f"{interpreter} -m pytest -x -q --no-header -p no:cacheprovider "
            "[host-selected public targets]", "worker.run_tests command shape")


def build_boot_packet(*, brief: str, test_command: object = None,
                      interpreter: object = None, target_files: object = (),
                      mode_override: str | None = None) -> dict:
    """Compile a public-observation packet before any provider request."""
    mode = boot_mode() if mode_override is None else mode_override
    if mode not in VALID_MODES:
        raise ValueError("invalid StateFrame boot mode")
    text = brief if isinstance(brief, str) else _safe_text(brief)
    interpreter_text = _safe_text(interpreter or sys.executable, limit=180)
    command, command_provenance = _public_test_command(test_command, interpreter_text)
    files = _target_file_list(target_files)
    if not files:
        files = ("not recorded",)
    clauses = _clauses(text)
    hints = routing_hints(text)
    predicate_states = {
        key: ("unknown" if not text.strip() else ("true" if value else "false"))
        for key, value in hints.items()
    }
    packet = {
        "schema": "stateframe-boot/1",
        "mode": mode,
        "contract_index_complete": False,
        "facts": {
            "test_command": command,
            "test_command_provenance": command_provenance,
            "interpreter": interpreter_text,
            "interpreter_provenance": "task_row.python or sys.executable",
            "target_files": list(files),
            "target_files_provenance": "runtime.target and public task-row file list",
        },
        "clauses": list(clauses),
        "routing_hints": hints,
        "predicate_states": predicate_states,
        "selected_card": select_method_card(hints, mode=mode),
    }
    packet["capsule"] = render_capsule(packet)
    packet["card_text"] = (load_method_card(packet["selected_card"])
                            if packet["selected_card"] else "")
    return packet


def render_capsule(packet: Mapping[str, object]) -> str:
    """Render the persisted capsule without exposing private task metadata."""
    facts = packet.get("facts", {})
    facts = facts if isinstance(facts, Mapping) else {}
    raw_target_files = _target_file_list(facts.get("target_files", ()))
    target_files = tuple(_safe_text(item, limit=56) for item in raw_target_files[:3])
    if not target_files:
        target_files = ("not recorded",)
    if len(raw_target_files) > 3:
        target_files = (*target_files, "... additional target files omitted")
    test_command = _safe_text(facts.get("test_command", "not recorded"), limit=100)
    test_command_provenance = _safe_text(
        facts.get("test_command_provenance", "unknown"), limit=100)
    interpreter = _safe_text(facts.get("interpreter", "not recorded"), limit=64)
    interpreter_provenance = _safe_text(
        facts.get("interpreter_provenance", "unknown"), limit=100)
    target_files_provenance = _safe_text(
        facts.get("target_files_provenance", "unknown"), limit=100)
    clauses = packet.get("clauses", ())
    hints = packet.get("routing_hints", {})
    hints = hints if isinstance(hints, Mapping) else {}
    prefix = [
        "[BOOT CAPSULE | host-compiled | public observations only]",
        "This is a bounded orientation index. The original brief remains authoritative; "
        "this index is not a checklist, gate, test result, or completion requirement.",
        "ENVIRONMENT FACTS (each item includes provenance):",
        f"- test command: {test_command} "
        f"[provenance: {test_command_provenance}].",
        f"- interpreter: {interpreter} "
        f"[provenance: {interpreter_provenance}].",
        f"- target files: {', '.join(target_files)} "
        f"[provenance: {target_files_provenance}].",
        "BRIEF CLAUSE INDEX (exact quoted spans; advisory index; "
        "contract_index_complete=false):",
    ]
    clause_lines = []
    if isinstance(clauses, Sequence) and not isinstance(clauses, (str, bytes)):
        clause_lines.extend(
            f"- {item.get('id')}: "
            f"{json.dumps(_safe_text(item.get('quote', ''), limit=320), ensure_ascii=False)}"
            for item in clauses if isinstance(item, Mapping)
        )
    has_clause = (isinstance(clauses, Sequence) and not isinstance(clauses, (str, bytes))
                  and any(isinstance(item, Mapping) for item in clauses))
    if not has_clause:
        clause_lines.append("- no non-empty public brief span was recorded")
    suffix = [
        "OPTIONAL ROUTING HINTS (allowlisted public rules; not obligations): "
        + ", ".join(f"{key}={str(value).lower()}" for key, value in hints.items()),
        "Rules: numeric_semantics requires quantity/unit/rate/duration language with "
        "semantic context; multi_component requires at least two named component roles "
        "or class/interface-like names. A false hint means only that this conservative "
        "public rule did not fire.",
        "Scope limits: no private tests, task identifiers, gold artifacts, or model-derived "
        "claims are included. Re-check source and public tests as appropriate.",
    ]
    lines = [*prefix, *clause_lines, *suffix]
    rendered = "\n".join(lines)
    if (len(rendered) + 3) // 4 <= CAPSULE_MAX_TOKENS:
        return rendered

    # Keep the index useful under a long public brief: retain complete,
    # redacted clause quotes until the fixed routing/scope footer still fits,
    # then state explicitly that the advisory index is incomplete.  The
    # original brief is never replaced by this bounded projection.
    omitted = "- additional public brief spans omitted from bounded index."
    selected = []
    for line in clause_lines:
        trial = "\n".join([*prefix, *selected, line, omitted, *suffix])
        if (len(trial) + 3) // 4 > CAPSULE_MAX_TOKENS:
            break
        selected.append(line)
    rendered = "\n".join([*prefix, *selected, omitted, *suffix])
    if (len(rendered) + 3) // 4 > CAPSULE_MAX_TOKENS:
        # The fixed public header/footer should fit, but fail closed if future
        # edits make them larger rather than emitting an over-budget capsule.
        rendered = "\n".join([*prefix, omitted, *suffix])
    return rendered


def frame_boot_fields(packet: Mapping[str, object]) -> dict:
    """Return the packet fields persisted inside a live frame."""
    return {
        "boot_packet": {
            "schema": packet.get("schema"),
            "capsule": packet.get("capsule", ""),
            "contract_index_complete": False,
            "routing_hints": dict(packet.get("routing_hints", {})),
            "selected_card": packet.get("selected_card"),
            "card_text": packet.get("card_text", ""),
        }
    }


def render_advisory_sidecar(packet: Mapping[str, object], *, first_request: bool) -> str:
    """Render the exact sidecar used by both provider arms."""
    if packet.get("mode") == "off":
        return ""
    parts = ["[StateFrame advisory sidecar]", str(packet.get("capsule", ""))]
    if first_request:
        parts.append(ORIENTATION_CUE)
    if packet.get("mode") == "cards" and packet.get("selected_card") and packet.get("card_text"):
        parts.extend([ADVISORY_LABEL, str(packet["card_text"])])
    return "\n\n".join(parts)


def append_advisory_sidecar(messages: Sequence[dict], packet: Mapping[str, object], *,
                            first_request: bool) -> list[dict]:
    """Append the sidecar without changing the input message sequence."""
    sidecar = render_advisory_sidecar(packet, first_request=first_request)
    if not sidecar:
        return [dict(message) for message in messages]
    result = [dict(message) for message in messages]
    result.append({"role": "user", "content": sidecar})
    return result


def capsule_token_estimate(packet: Mapping[str, object]) -> int:
    """Use the same transparent character proxy as the local token probes."""
    return (len(str(packet.get("capsule", ""))) + 3) // 4

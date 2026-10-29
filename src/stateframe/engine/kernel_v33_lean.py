"""V33 lean policy: compact duplicate presentation and cache stable context.

This module is deliberately pure at its public boundary.  It does not decide
whether an action is authorized, whether a check passed, or whether completion
is admissible.  It only removes exact duplicate advisory presentation, keeps
stable protocol/ticket/fact blocks at the front of a request, and classifies
when extra diagnostic detail is justified.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


MODE_ENV = "STATEFRAME_TOKENLAB_V33_LEAN"
SCHEMA = "stateframe-v33-lean/1"
QUIET = "quiet"
FOCUSED = "focused"
RECOVERY = "recovery"
_LEVELS = frozenset({QUIET, FOCUSED, RECOVERY})

_VOLATILE_FIELD = re.compile(
    r"(?i)(?:[\"']?(?:step(?:_number)?|timestamp|revision(?:_hash)?|"
    r"(?:git_)?(?:sha|hash)|binding|receipt|request_id|turn)[\"']?\s*[:=])"
)
_VOLATILE_HEADERS = (
    "[StateFrame]", "[Decision handoff]", "[Current source:",
    "[Working source:", "[VERIFICATION", "[Receipts]", "[Receipt]",
    "[ENVIRONMENT STATE]", "[Get-sure state]", "<settled_state>",
    "[COORDINATED CHANGE LEDGER]", "[Passive conclusions",
    "[Unified stop authority]", "[DELIVERY_MANIFEST",
)
_STABLE_FACT_HEADERS = (
    "[Repository layout;", "[Immutable repository facts]", "[REPOSITORY FACTS]",
)
_DEDUP_HEADERS = (
    "[Get-sure state]", "[COORDINATED CHANGE LEDGER]", "[Passive conclusions",
    "[ENVIRONMENT STATE]", "[DELIVERY_MANIFEST",
)
_DIFFICULTY_FAILURES = frozenset({
    "FAIL", "FAILED", "ERROR", "RED", "REFUSED", "REJECTED", "NOT_RUN",
    "STALE", "CONFLICT", "BLOCKED", "NEEDS_WORK",
})
_SAFETY_WORDS = re.compile(
    r"(?i)\b(?:unsafe|pre[-_ ]?write|protected(?:\s+path)?|authorization|"
    r"integrity(?:\s+constraint)?|observed\s+violation|witnessed\s+obligation)\b"
)


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_v33_lean_mode", None)
    if configured is not None:
        return str(configured).strip().lower() == "on"
    return mode() == "on"


@dataclass(frozen=True)
class Difficulty:
    level: str
    reasons: tuple[str, ...] = ()


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _rows(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(item for item in values if isinstance(item, Mapping))


def difficulty(frame: Mapping[str, object] | None = None,
               receipts: Sequence[Mapping[str, object]] = ()) -> Difficulty:
    """Choose quiet by default; expose diagnostics only for concrete trouble."""
    current = frame if isinstance(frame, Mapping) else {}
    reasons: list[str] = []
    statuses: list[str] = []
    def collect_statuses(value: object, *, depth: int = 0) -> None:
        if depth > 3:
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"status", "result", "result_class", "outcome", "check_status", "state"}:
                    statuses.append(str(item).upper())
                elif isinstance(item, (Mapping, list, tuple)):
                    collect_statuses(item, depth=depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect_statuses(item, depth=depth + 1)
    collect_statuses(current)
    for row in (*_rows(current.get("test_receipts")), *_rows(current.get("receipts")),
                *tuple(receipts)):
        for key in ("status", "result", "result_class", "outcome"):
            if row.get(key) is not None:
                statuses.append(str(row[key]).upper())
    if any(status in _DIFFICULTY_FAILURES for status in statuses):
        reasons.append("concrete failure or unresolved check")
    unresolved: list[str] = []
    for key in ("unresolved_requirements", "unresolved_gating_items", "open_requirements"):
        value = current.get(key)
        if isinstance(value, Mapping):
            values = value.keys()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            values = value
        elif value not in (None, ""):
            values = (value,)
        else:
            values = ()
        unresolved.extend(str(item) for item in values if str(item).strip())
    if unresolved:
        reasons.append("unresolved requirement")
    if any(bool(current.get(key)) for key in ("surprise", "binding_changed", "binding_conflict",
                                               "recovery_required", "stale_receipt")):
        reasons.append("binding or recovery surprise")
    no_information_cycles = current.get("no_information_cycles", 0)
    if current.get("repeated_failure") or (
            isinstance(no_information_cycles, (int, float)) and no_information_cycles >= 2):
        reasons.append("repeated no-progress failure")
    if any("conflict" in str(value).lower() or "recovery" in str(value).lower()
           for value in current.values() if isinstance(value, str)):
        reasons.append("recovery signal")
    level = RECOVERY if len(reasons) >= 2 and any(
        "repeated" in reason or "recovery" in reason for reason in reasons) else (
            FOCUSED if reasons else QUIET)
    return Difficulty(level, tuple(dict.fromkeys(reasons)))


def _header(content: str) -> str:
    return content.lstrip().splitlines()[0] if content.strip() else ""


def _stable_message(message: Mapping[str, object]) -> bool:
    role = message.get("role")
    content = _text(message.get("content"))
    if role not in {"system", "developer", "user"} or not content:
        return False
    if _VOLATILE_FIELD.search(content) or any(content.startswith(item) for item in _VOLATILE_HEADERS):
        return False
    if role in {"system", "developer"} or content.startswith("Task:"):
        return True
    return content.startswith(_STABLE_FACT_HEADERS)


def _tool_name(tool: Mapping[str, object]) -> str:
    function = tool.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        return str(function["name"])
    return str(tool.get("name", ""))


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str).encode("utf-8")


def ordered_tools(tools: object) -> object:
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return tools
    values = list(tools)
    return sorted(values, key=lambda item: (
        _tool_name(item) if isinstance(item, Mapping) else "", _canonical(item)))


def _open_prefill(messages: Sequence[Mapping[str, object]]) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    if not messages:
        return [], []
    final = messages[-1]
    if (final.get("role") == "assistant" and final.get("content") is None
            and isinstance(final.get("reasoning_content"), str)):
        return list(messages[:-1]), [final]
    return list(messages), []


def _compact_stateframe(content: str) -> str:
    body = content.split("\n", 1)[1] if "\n" in content else ""
    try:
        packet = json.loads(body)
    except (TypeError, json.JSONDecodeError):
        return content
    state = packet.get("state") if isinstance(packet, Mapping) else None
    if not isinstance(state, Mapping):
        return content

    def status_row(value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        result: dict[str, object] = {}
        for key in ("state", "status", "result", "result_class", "selectors"):
            if key in value:
                result[key] = copy.deepcopy(value[key])
        evidence = value.get("evidence")
        if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray)):
            result["evidence_count"] = len(evidence)
        return result

    def item_count(value: object) -> int:
        if isinstance(value, Mapping):
            return len(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return len(value)
        return 0

    public_checks = status_row(state.get("public_checks"))
    try:
        from kernel_stateframe_f4 import public_checks_were_not_run
        if public_checks_were_not_run(state) and isinstance(public_checks, dict):
            if "state" in public_checks:
                public_checks["state"] = "NOT_RUN"
    except (ImportError, AttributeError, TypeError, ValueError):
        pass

    compact: dict[str, object] = {
        "rev": packet.get("rev"),
        "schema": packet.get("schema"),
        "step": state.get("step"),
        "completion": status_row(state.get("completion")),
        "done_permission": status_row(state.get("done_permission")),
        "public_checks": public_checks,
        "submission": status_row(state.get("submission")),
        "blockers": item_count(state.get("blockers")),
        "advisories": item_count(state.get("advisories")),
    }
    authority = state.get("edit_authority")
    if isinstance(authority, Mapping):
        compact["edit_authority"] = {
            "mode": authority.get("mode"),
            "editable_paths": copy.deepcopy(authority.get("editable_paths", ())),
            "tools": sorted(str(key) for key in (authority.get("tool_matrix", {}) or {})
                             if str(key)),
        }
    history = state.get("history_findings")
    if isinstance(history, Mapping):
        findings = history.get("findings", ())
        latest = findings[-1] if isinstance(findings, Sequence) and findings else None
        compact["history_findings"] = {
            "count": len(findings) if isinstance(findings, Sequence) else 0,
            "latest": {key: latest.get(key) for key in ("id", "verdict", "status")}
            if isinstance(latest, Mapping) else None,
        }
    return "[StateFrame]\n" + json.dumps(compact, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":"))


def _compact_get_sure(content: str) -> str:
    lines = content.splitlines()
    kept = ["[Get-sure state]", "truth: current clause verdicts; ticket text remains authoritative"]
    for line in lines[1:]:
        if line.startswith("EXACT-TEXT CERTAINTY:") or line.startswith("C") and ":" not in line:
            kept.append(line)
        elif re.match(r"^C\d+(?:,C\d+)*:\s+observed:", line):
            kept.append(line)
    clause_rows = [line for line in lines if re.match(r"^C\d+\s+", line)]
    if clause_rows:
        verdicts = []
        for line in clause_rows:
            match = re.match(r"^(C\d+)\s+([^:]+):", line)
            if match:
                verdicts.append(f"{match.group(1)}={match.group(2).strip()}")
        kept.append("verdicts: " + ", ".join(verdicts))
    return "\n".join(dict.fromkeys(kept))


def _compact_content(content: str) -> str:
    if content.startswith("[StateFrame]\n"):
        return _compact_stateframe(content)
    if content.startswith("[Get-sure state]"):
        return _compact_get_sure(content)
    return content


def compact_messages(messages: Sequence[Mapping[str, object]], *, level: str = QUIET) -> list[dict[str, object]]:
    """Compact truth and suppress only non-actionable advice in quiet mode."""
    if level not in _LEVELS:
        raise ValueError(f"unknown presentation level: {level}")
    result: list[dict[str, object]] = []
    seen: set[bytes] = set()
    for original in messages:
        message = copy.deepcopy(dict(original))
        content = _text(message.get("content"))
        if content:
            compacted = _compact_content(content)
            if compacted != content:
                message["content"] = compacted
                content = compacted
        if (level == QUIET and message.get("role") == "user"
                and content.startswith("[ripple-preview]")
                and "STAGED" not in content.upper()
                and "NOT APPLIED" not in content.upper()):
            # Inline ripple advice is already journaled and has no host action
            # when no difficulty is present; staged truth is never suppressed.
            continue
        compactable = (message.get("role") == "user"
                       and any(content.startswith(prefix) for prefix in _DEDUP_HEADERS))
        identity = _canonical(message) if compactable else b""
        if compactable and identity in seen:
            continue
        if compactable:
            seen.add(identity)
        result.append(message)
    return result


@dataclass(frozen=True)
class LayoutResult:
    messages: tuple[dict[str, object], ...]
    tools: object
    stable_messages: tuple[dict[str, object], ...]
    prefix_bytes: bytes
    changed: bool
    removed_duplicates: int
    level: str

    @property
    def prefix_tokens(self) -> int:
        return token_proxy(self.prefix_bytes)


def layout_request(messages: Sequence[Mapping[str, object]], tools: object, *,
                   frame: Mapping[str, object] | None = None,
                   receipts: Sequence[Mapping[str, object]] = ()) -> LayoutResult:
    """Put immutable request context first and volatile state after it.

    Assistant/tool ordering is preserved as one dynamic sequence.  Only system,
    ticket, and explicitly immutable fact messages are moved to the front.
    """
    decision = difficulty(frame, receipts)
    if decision.level == QUIET:
        for message in messages:
            content = _text(message.get("content"))
            if content.startswith("[StateFrame]\n"):
                try:
                    packet = json.loads(content.split("\n", 1)[1])
                except (TypeError, json.JSONDecodeError):
                    packet = None
                if isinstance(packet, Mapping):
                    packet_decision = difficulty(packet)
                    if packet_decision.level != QUIET:
                        decision = packet_decision
                        break
            if content.startswith("[RAISING-H FAILURE PACKET]"):
                decision = Difficulty(FOCUSED, ("concrete failure packet",))
                break
    compacted = compact_messages(messages, level=decision.level)
    ordered, prefill = _open_prefill(compacted)
    stable = [message for message in ordered if _stable_message(message)]
    dynamic = [message for message in ordered if not _stable_message(message)]
    arranged = [*stable, *dynamic, *prefill]
    arranged_tools = ordered_tools(tools)
    prefix_bytes = _canonical({"tools": arranged_tools, "messages": stable})
    removed = len(messages) - len(compacted)
    changed = arranged != list(messages) or arranged_tools != tools
    return LayoutResult(tuple(arranged), arranged_tools, tuple(stable), prefix_bytes,
                        changed, removed, decision.level)


def request_bytes(messages: Sequence[Mapping[str, object]], tools: object) -> bytes:
    return _canonical({"tools": tools, "messages": list(messages)})


def token_proxy(value: object) -> int:
    raw = value if isinstance(value, bytes) else _canonical(value)
    return (len(raw) + 3) // 4


def common_prefix_length(values: Sequence[bytes]) -> int:
    if not values:
        return 0
    shortest = min(len(value) for value in values)
    for index in range(shortest):
        if any(value[index] != values[0][index] for value in values[1:]):
            return index
    return shortest


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def holds_advisory(candidates: Sequence[object]) -> bool:
    """Hold only a witnessed safety/authorization constraint, never review evidence."""
    for candidate in candidates:
        authority = str(getattr(candidate, "authority", "")).upper()
        if authority == "WITNESSED_OBLIGATION" or bool(getattr(candidate, "is_obligation", False)):
            return True
        witness = getattr(candidate, "witness", {})
        if isinstance(witness, Mapping) and any(bool(witness.get(key)) for key in (
                "unsafe_effect", "pre_write_required", "protected_path",
                "authorization_required", "integrity_constraint", "observed_violation")):
            return True
        for evidence in getattr(candidate, "evidence", ()) or ():
            reason = _text(getattr(evidence, "reason", ""))
            if _SAFETY_WORDS.search(reason):
                return True
    return False


__all__ = [
    "FOCUSED", "MODE_ENV", "QUIET", "RECOVERY", "SCHEMA", "Difficulty", "LayoutResult",
    "common_prefix_length", "compact_messages", "difficulty", "digest", "enabled", "holds_advisory",
    "layout_request", "mode", "ordered_tools", "request_bytes", "token_proxy",
]

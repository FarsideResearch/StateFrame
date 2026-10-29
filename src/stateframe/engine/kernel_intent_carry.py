"""Opt-in carry-over for a model's committed, not-yet-applied edit decision.

The ordinary StateFrame is deliberately built from host-observed state.  This
module adds one narrow exception behind ``STATEFRAME_INTENT_CARRY=on``: a
bounded, typed copy of the model's own final edit decision is journaled and
shown on the next frame.  It is never treated as proof or as edit authority.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


MODE_ENV = "STATEFRAME_INTENT_CARRY"
SCHEMA = "stateframe-intent-carry/1"
# r46 (tt_output_gap_v2 table items 1-2): deliver the pending intent on the F4 wire, suppress
# path-disagreeing extractions, drop the stale static BOOT phase sentence.  Strict off|on.
F4_MODE_ENV = "STATEFRAME_INTENT_CARRY_F4"
PENDING_INTENT_HEADER = "[Pending intent — host reminder, not edit authority]"
MODEL_STATED = "MODEL_STATED"
MODEL_REJECTED_EDIT = "MODEL_REJECTED_EDIT"
_EDIT_TOOLS = frozenset({
    "edit", "str_replace", "file_create", "replace_module", "write", "replace",
})
_MAX_SUMMARY = 200
_MAX_AGE = 3

_PATH_RE = re.compile(
    r"(?<![\w./-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9_]+"
    r"|(?<![\w./-])[A-Za-z0-9_.-]+\.py\b"
)
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?\b")
_QUOTED_IDENTIFIER_RE = re.compile(r"[`'\"]([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)['`\"]")
_COMMITMENT_RE = re.compile(
    r"\b(?:let\s+me|let['’]s|i['’]ll|i\s+will|i\s+am\s+going\s+to|"
    r"i['’]m\s+going\s+to)\s+"
    r"(?P<body>(?:(?:make|write|edit|change|update|fix|add|replace|apply|implement)"
    r"\b[^.\n]{0,260}))",
    re.IGNORECASE,
)
_PLAN_RE = re.compile(
    r"^\s*(?:(?:[-*])|(?:\d+[.)]))?\s*"
    r"(?:edit|write|change|update|fix|add|replace)\b(?P<body>.*)$",
    re.IGNORECASE,
)
_PATH_PLAN_RE = re.compile(
    r"^\s*(?:(?:[-*])|(?:\d+[.)]))\s*[*_`]*"
    r"(?P<path>(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9_]+"
    r"|[A-Za-z0-9_.-]+\.py)\b[*_`]*\s*[:\-–—]\s*(?P<body>.*)$",
    re.IGNORECASE,
)
_HEDGING_RE = re.compile(
    r"\b(?:could|might|may|maybe|consider|possibly|if\s+needed|should\s+consider)\b",
    re.IGNORECASE,
)
_GENERIC_SUMMARIES = {
    "make this change", "make the change", "make that change", "make the edit",
    "write the edit", "do the edit", "apply the edit", "fix this", "update this",
}


def mode() -> str:
    """Return the validated intent-carry switch."""
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def f4_mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the validated STATEFRAME_INTENT_CARRY_F4 switch (strict off|on, default off)."""
    source = os.environ if environ is None else environ
    value = str(source.get(F4_MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{F4_MODE_ENV} must be off or on")
    return value


def f4_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return f4_mode(environ) == "on"


def _compact(value: object, limit: int = _MAX_SUMMARY) -> str:
    text = " ".join(str(value or "").split())
    text = text.strip(" \t\r\n`*_-")
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)].rstrip() + "..."


def normalize_path(value: object, root: Path | None = None) -> str:
    """Normalize a workspace-relative path without accepting an outside path."""
    if not isinstance(value, str):
        return ""
    raw = value.strip().strip("`'\" ").rstrip(".,;:)]}")
    if not raw:
        return ""
    raw = raw.replace("\\", "/")
    if root is not None:
        try:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = root / candidate
            return candidate.resolve().relative_to(root.resolve()).as_posix()
        except (OSError, RuntimeError, ValueError):
            return ""
    return raw.lstrip("./")


def _normalize_symbol(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().strip("`'\" ").strip(".,;:()[]")
    if not text or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", text):
        return None
    return text


@dataclass(frozen=True)
class PendingIntent:
    """A typed model decision that has not yet been observed as applied."""

    target_path: str
    target_symbol: str | None
    summary: str
    step: int
    source: str = MODEL_STATED

    def __post_init__(self) -> None:
        path = normalize_path(self.target_path)
        symbol = _normalize_symbol(self.target_symbol)
        summary = _compact(self.summary)
        if not path:
            raise ValueError("PendingIntent.target_path must be non-empty")
        if not summary:
            raise ValueError("PendingIntent.summary must be non-empty")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("PendingIntent.step must be a nonnegative integer")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("PendingIntent.source must be non-empty")
        object.__setattr__(self, "target_path", path)
        object.__setattr__(self, "target_symbol", symbol)
        object.__setattr__(self, "summary", summary)

    @property
    def intent_id(self) -> str:
        raw = f"{self.target_path}\x00{self.target_symbol or ''}"
        return "intent:" + hashlib.sha256(raw.encode()).hexdigest()[:24]

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "PendingIntent",
            "schema": SCHEMA,
            "intent_id": self.intent_id,
            "target_path": self.target_path,
            "target_symbol": self.target_symbol,
            "summary": self.summary,
            "step": self.step,
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PendingIntent":
        return cls(
            target_path=str(value.get("target_path") or ""),
            target_symbol=(str(value["target_symbol"]) if value.get("target_symbol") else None),
            summary=str(value.get("summary") or ""),
            step=int(value.get("step", 0)),
            source=str(value.get("source") or MODEL_STATED),
        )


def _action_info(action: object) -> tuple[str, str, str | None]:
    """Return tool, path, and symbol from either a typed call or legacy command."""
    if isinstance(action, Mapping):
        tool = str(action.get("tool") or action.get("kind") or action.get("name") or "").lower()
        arguments = action.get("arguments")
        if not isinstance(arguments, Mapping):
            arguments = action
        path = normalize_path(arguments.get("path"))
        symbol = (_normalize_symbol(arguments.get("target_symbol"))
                  or _normalize_symbol(arguments.get("symbol"))
                  or _normalize_symbol(arguments.get("function")))
        return tool, path, symbol
    if isinstance(action, str):
        parts = action.strip().split()
        tool = parts[0].lower() if parts else ""
        path = normalize_path(parts[1]) if len(parts) > 1 else ""
        return tool, path, None
    return "", "", None


def _path_from(text: str, context: str, default_path: str) -> str:
    matches = _PATH_RE.findall(context) or _PATH_RE.findall(text[-1400:])
    if matches:
        return normalize_path(matches[-1])
    return normalize_path(default_path)


def _symbol_from(text: str, context: str, default_symbol: str | None) -> str | None:
    quoted = [_normalize_symbol(item) for item in _QUOTED_IDENTIFIER_RE.findall(context)]
    quoted = [item for item in quoted if item]
    # A returned field such as ``'late_fee_cents'`` is part of the change
    # summary, not normally the function being edited.  Backtick-qualified
    # names and explicit function/method labels remain symbol evidence.
    if quoted and not re.search(r"\breturn\s+[`'\"]" + re.escape(quoted[-1])
                                + r"[`'\"]", context, re.IGNORECASE):
        if "`" in context or re.search(r"\b(?:function|method|symbol)\b", context,
                                        re.IGNORECASE):
            return quoted[-1]
    if default_symbol:
        normalized = _normalize_symbol(default_symbol)
        if normalized and (normalized in context or normalized.split(".")[-1] in context):
            return normalized
    # Prefer identifiers explicitly introduced as a function/method/symbol or
    # immediately following "to/for/in" in the decision context.
    labelled = re.findall(
        r"\b(?:function|method|symbol|from|for|in|to|change\s+to)\s+"
        r"[`'\"]?([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)[`'\"]?",
        context,
        re.IGNORECASE,
    )
    for candidate in reversed(labelled):
        normalized = _normalize_symbol(candidate)
        if normalized and normalized.lower() not in {
                "the", "a", "an", "this", "return", "service", "change", "edit"}:
            return normalized
    if default_symbol:
        normalized = _normalize_symbol(default_symbol)
        if normalized:
            return normalized
    if quoted:
        return quoted[-1]
    identifiers = [_normalize_symbol(item) for item in _IDENTIFIER_RE.findall(context)]
    ignored = {
        "let", "me", "make", "the", "change", "edit", "write", "fix", "update",
        "add", "replace", "apply", "implement", "now", "this", "that", "service",
    }
    for candidate in reversed(identifiers):
        if candidate and candidate.lower() not in ignored:
            return candidate
    return None


def _summary_for(text: str, start: int, end: int, target_path: str) -> str:
    body = text[start:end].strip()
    before = text[max(0, start - 520):start].strip()
    prior = re.split(r"(?<=[.!?])\s+|\n+", before)
    prior = next((item.strip() for item in reversed(prior) if item.strip()), "")
    cleaned_body = re.sub(
        r"^(?:let\s+me|let['’]s|i['’]ll|i\s+will|i\s+am\s+going\s+to|"
        r"i['’]m\s+going\s+to)\s+",
        "",
        body,
        flags=re.IGNORECASE,
    )
    cleaned_body = re.sub(r"\b(?:now|please)\s*[.!]?$", "", cleaned_body,
                          flags=re.IGNORECASE).strip()
    # A generic final "write the edit" gets its meaning from the sentence that
    # committed the change immediately before it (the crawl R24 shape).
    candidate = cleaned_body
    if candidate.lower().strip(" .") in _GENERIC_SUMMARIES or len(candidate) < 8:
        candidate = prior or candidate
    if target_path and target_path in candidate:
        candidate = candidate.replace(target_path, "").strip(" :,-")
    return _compact(candidate or "apply the stated edit")


def _committed_candidates(text: str, default_path: str,
                          default_symbol: str | None) -> list[PendingIntent]:
    if not isinstance(text, str) or not text.strip():
        return []
    stripped = text.strip()
    matches = list(_COMMITMENT_RE.finditer(stripped))
    if not matches:
        return []
    match = matches[-1]
    suffix = stripped[match.end():].strip(" \t\r\n.,;:!?`*_-")
    # The decision must be at the end of the model's statement.  This rejects
    # "Let me edit ..." when it is merely followed by a reconsideration.
    if len(suffix) > 40:
        return []
    context = stripped[max(0, match.start() - 900):match.end()]
    if _HEDGING_RE.search(context):
        return []
    target_path = _path_from(stripped, context, default_path)
    if not target_path:
        return []
    target_symbol = _symbol_from(stripped, context, default_symbol)
    try:
        return [PendingIntent(target_path, target_symbol,
                              _summary_for(stripped, match.start(), match.end(), target_path),
                              0, MODEL_STATED)]
    except ValueError:
        return []


def _plan_candidates(text: str, default_path: str,
                     default_symbol: str | None) -> list[PendingIntent]:
    if not isinstance(text, str):
        return []
    result: list[PendingIntent] = []
    for line in text.splitlines()[-16:]:
        match = _PLAN_RE.match(line)
        path_match = _PATH_PLAN_RE.match(line)
        if match is None and path_match is None:
            continue
        if path_match is not None:
            body = path_match.group("body").strip(" :\t-")
            target_path = normalize_path(path_match.group("path"))
        else:
            body = match.group("body").strip(" :\t-")
            target_path = _path_from(body, body, default_path)
        context = body
        if not target_path:
            continue
        # A plan item must name an actual change, not merely restate a
        # requirement.  An explicit workspace path is enough because the line
        # itself already began with an edit verb; otherwise require a second
        # concrete change verb before falling back to the default path.
        if not _PATH_RE.search(body) and not re.search(
                r"\b(?:replace|change|update|fix|add|write|edit|forward|catch|map|validate|implement)\b", body,
                re.IGNORECASE):
            continue
        target_symbol = _symbol_from(body, context, default_symbol)
        summary = _compact(body)
        if target_path in summary:
            summary = _compact(summary.replace(target_path, "").strip(" :,-"))
        try:
            result.append(PendingIntent(target_path, target_symbol, summary, 0, MODEL_STATED))
        except ValueError:
            continue
    return result


def _dedupe(intents: Sequence[PendingIntent]) -> list[PendingIntent]:
    result: list[PendingIntent] = []
    seen = set()
    for intent in intents:
        key = (intent.target_path, intent.target_symbol)
        if key in seen:
            continue
        seen.add(key)
        result.append(intent)
    return result


def extract_pending_intents(
    *, reasoning: object = None, content: object = None,
    default_path: str = "", default_symbol: str | None = None,
    step: int = 0, action: object = None,
) -> tuple[PendingIntent, ...]:
    """Extract only a committed final edit decision from model text.

    The parser intentionally prefers false negatives.  It recognizes a final
    ``Let me ... edit/change/...`` shape and explicit edit-plan rows; tentative
    language such as ``could`` and ``consider`` is rejected.
    """
    sources = [value for value in (reasoning, content) if isinstance(value, str) and value.strip()]
    extracted: list[PendingIntent] = []
    for source_text in sources:
        extracted.extend(_committed_candidates(source_text, default_path, default_symbol))
        extracted.extend(_plan_candidates(source_text, default_path, default_symbol))
    if not extracted:
        return ()
    tool, action_path, action_symbol = _action_info(action)
    stamped: list[PendingIntent] = []
    for intent in _dedupe(extracted):
        try:
            stamped.append(PendingIntent(
                target_path=normalize_path(intent.target_path) or action_path or default_path,
                target_symbol=intent.target_symbol,
                summary=intent.summary,
                step=step,
                source=intent.source,
            ))
        except ValueError:
            continue
    # A model that actually emitted an edit for the stated target has already
    # carried the decision into the action envelope; do not create a duplicate
    # pending reminder.  A read/test/done keeps the decision open.
    if tool in _EDIT_TOOLS:
        stamped = [item for item in stamped
                   if not _same_target(item, action_path, action_symbol)]
    return tuple(_dedupe(stamped))


def split_by_reference_paths(
        intents: Sequence[PendingIntent], reference_paths: Sequence[object],
        root: Path | None = None) -> tuple[tuple[PendingIntent, ...], tuple[PendingIntent, ...]]:
    """(kept, suppressed): an intent survives only when its target agrees with at least one
    reference path (the response's own tool-call path, the task target, or a source path the
    host actually delivered).  With no usable reference every intent is kept -- this guard
    can only drop a demonstrably mis-targeted extraction (J3:386 models->filters.py), never
    invent one."""
    refs = []
    for raw in reference_paths:
        path = normalize_path(raw, root) if root is not None else normalize_path(raw)
        if path and path not in refs:
            refs.append(path)
    if not refs:
        return tuple(intents), ()
    kept: list[PendingIntent] = []
    dropped: list[PendingIntent] = []
    for intent in intents:
        if any(_same_target(intent, ref, None) for ref in refs):
            kept.append(intent)
        else:
            dropped.append(intent)
    return tuple(kept), tuple(dropped)


def _same_target(intent: PendingIntent, path: str, symbol: str | None) -> bool:
    if not path:
        return False
    same_path = intent.target_path == path or intent.target_path.endswith("/" + path) \
        or path.endswith("/" + intent.target_path)
    if not same_path:
        return False
    if intent.target_symbol and symbol:
        return intent.target_symbol == symbol or intent.target_symbol.endswith("." + symbol)
    return True


def _definition_nodes(source: str) -> tuple[ast.Module | None, tuple[ast.AST, ...]]:
    try:
        tree = ast.parse(textwrap.dedent(source))
    except (SyntaxError, TypeError):
        return None, ()
    definitions = tuple(node for node in tree.body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    return tree, definitions


def _definition_map(source: str) -> dict[str, ast.AST]:
    _tree, definitions = _definition_nodes(source)
    return {str(node.name): node for node in definitions}


def unapplied_rejected_definitions(current_source: str, candidate_source: str) -> tuple[str, ...]:
    """Return changed top-level definitions from a rejected multi-def payload."""
    tree, definitions = _definition_nodes(candidate_source)
    if tree is None or not definitions:
        return ()
    # The host rejects both multiple definitions and imports plus a definition.
    if len(definitions) <= 1 and len(tree.body) <= 1:
        return ()
    current = _definition_map(current_source)
    result = []
    for node in definitions:
        old = current.get(str(node.name))
        if old is None or ast.dump(old, include_attributes=False) != ast.dump(
                node, include_attributes=False):
            result.append(str(node.name))
    return tuple(result)


def _safe_symbol_span(source: str, symbol: str | None) -> tuple[int, int] | None:
    if not symbol:
        return None
    wanted = symbol.split(".")
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError):
        return None

    found: tuple[int, int] | None = None

    def visit(body: Sequence[ast.AST], prefix: tuple[str, ...] = ()) -> None:
        nonlocal found
        for node in body:
            name = getattr(node, "name", None)
            if isinstance(name, str):
                qualified = (*prefix, name)
                if (tuple(qualified) == tuple(wanted)
                        or name == wanted[-1]
                        or ".".join(qualified).endswith(symbol)):
                    found = (int(getattr(node, "lineno", 1)),
                             int(getattr(node, "end_lineno", getattr(node, "lineno", 1))))
                nested = getattr(node, "body", None)
                if isinstance(nested, list):
                    visit(nested, qualified)

    visit(tree.body)
    return found


class IntentCarryState:
    """Journal-backed bounded state for one worker attempt."""

    def __init__(self, journal: object, *, root: Path | None = None):
        self.journal = journal
        self.root = root.resolve() if isinstance(root, Path) else None
        self._pending: dict[str, PendingIntent] = {}
        self._restore()

    @property
    def pending(self) -> tuple[PendingIntent, ...]:
        return tuple(self._pending.values())

    def _emit(self, kind: str, **fields: object) -> None:
        emit = getattr(self.journal, "emit", None)
        if callable(emit):
            emit(kind, attribution="intent_carry", **fields)

    def _restore(self) -> None:
        records = getattr(self.journal, "records", None)
        if not callable(records):
            return
        for event in records():
            if not isinstance(event, Mapping):
                continue
            kind = event.get("kind")
            if kind == "intent_extracted":
                raw = event.get("intent")
                if isinstance(raw, Mapping):
                    try:
                        intent = PendingIntent.from_mapping(raw)
                    except (TypeError, ValueError):
                        continue
                    self._pending[intent.intent_id] = intent
            elif kind == "intent_cleared":
                intent_id = event.get("intent_id")
                if isinstance(intent_id, str):
                    self._pending.pop(intent_id, None)
                else:
                    path = normalize_path(event.get("target_path"))
                    for key, intent in list(self._pending.items()):
                        if not path or intent.target_path == path:
                            self._pending.pop(key, None)

    def _clear(self, intent: PendingIntent, *, reason: str, step: int) -> None:
        self._pending.pop(intent.intent_id, None)
        self._emit("intent_cleared", intent_id=intent.intent_id,
                   intent=intent.to_dict(), target_path=intent.target_path,
                   target_symbol=intent.target_symbol, reason=reason, step=step)

    def advance(self, step: int) -> None:
        if type(step) is not int:
            return
        for intent in tuple(self.pending):
            if step - intent.step >= _MAX_AGE:
                self._clear(intent, reason="expired_after_three_steps", step=step)

    def observe_model(self, *, reasoning: object, content: object,
                      default_path: str, default_symbol: str | None,
                      step: int, action: object = None,
                      reference_paths: Sequence[object] | None = None) -> tuple[PendingIntent, ...]:
        self.advance(step)
        intents = extract_pending_intents(
            reasoning=reasoning, content=content, default_path=default_path,
            default_symbol=default_symbol, step=step, action=action)
        if reference_paths is not None and intents:
            # r46 STATEFRAME_INTENT_CARRY_F4: the caller passes references only when on.
            _tool, action_path, _symbol = _action_info(action)
            intents, suppressed = split_by_reference_paths(
                intents, [action_path, *reference_paths], self.root)
            for intent in suppressed:
                self._emit("intent_suppressed", intent=intent.to_dict(),
                           intent_id=intent.intent_id, target_path=intent.target_path,
                           step=step, reason="target_path_disagrees_with_action_or_sources")
        if not intents:
            tool, _path, _symbol = _action_info(action)
            if tool == "done" and self.pending:
                for intent in tuple(self.pending):
                    self._clear(intent, reason="different_decision", step=step)
            return ()
        keys = {intent.intent_id for intent in intents}
        for old in tuple(self.pending):
            if old.intent_id not in keys:
                self._clear(old, reason="different_decision", step=step)
        for intent in intents:
            self._pending[intent.intent_id] = intent
            self._emit("intent_extracted", intent=intent.to_dict(),
                       intent_id=intent.intent_id, target_path=intent.target_path,
                       target_symbol=intent.target_symbol, step=step,
                       source=intent.source)
        return intents

    def clear_for_edit(self, path: object, *, step: int, symbol: str | None = None) -> None:
        normalized = normalize_path(path, self.root)
        if not normalized:
            return
        for intent in tuple(self.pending):
            if _same_target(intent, normalized, _normalize_symbol(symbol)) \
                    or (symbol is None and _same_target(intent, normalized, None)):
                self._clear(intent, reason="edit_applied", step=step)

    def record_rejected_edit(self, *, path: str, current_source: str,
                             candidate_source: str, step: int) -> tuple[PendingIntent, ...]:
        normalized = normalize_path(path, self.root)
        if not normalized:
            return ()
        names = unapplied_rejected_definitions(current_source, candidate_source)
        result = []
        for name in names:
            intent = PendingIntent(
                normalized, name,
                f"apply the proposed {name} definition; split the rejected edit into one definition",
                step, MODEL_REJECTED_EDIT,
            )
            self._pending[intent.intent_id] = intent
            self._emit("intent_extracted", intent=intent.to_dict(),
                       intent_id=intent.intent_id, target_path=normalized,
                       target_symbol=name, step=step, source=MODEL_REJECTED_EDIT,
                       reason="rejected_multi_definition_edit")
            result.append(intent)
        return tuple(result)

    def frame_payload(self, *, step: int) -> dict[str, object] | None:
        self.advance(step)
        if not self.pending:
            return None
        payload = {
            "schema": SCHEMA,
            "status": "open",
            "items": [intent.to_dict() for intent in self.pending],
        }
        self._emit("intent_rendered", step=step, intents=payload["items"],
                   delivery="next_frame")
        return payload

    def redirect_already_present(self, base: Mapping[str, object], path: str,
                                 start_line: int, *, step: int) -> dict[str, object]:
        """Add current target lines and a typed action note to a dedup result."""
        self.advance(step)
        normalized = normalize_path(path, self.root)
        intent = next((item for item in self.pending
                       if _same_target(item, normalized, None)), None)
        if intent is None or self.root is None:
            return dict(base)
        candidate = (self.root / normalized).resolve()
        try:
            candidate.relative_to(self.root)
            source = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError):
            return dict(base)
        span = _safe_symbol_span(source, intent.target_symbol)
        if span is None:
            first = max(1, int(start_line) + 1)
            last = min(len(source.splitlines()), first + 119)
        else:
            first, last = span
            last = min(len(source.splitlines()), first + 119)
        lines = source.splitlines()
        current_lines = "\n".join(lines[first - 1:last])
        pending = intent.to_dict()
        result = dict(base)
        result.update({
            "schema": SCHEMA,
            "path": normalized,
            "current_lines": current_lines,
            "current_span": {"start_line": first, "end_line": last},
            "pending_intent": pending,
            "note": {"pending_intent": pending, "message": "apply the edit now"},
            "no_progress": True,
            "message": (str(base.get("message") or "already present in frame")
                         + "; pending intent unchanged — apply the edit now"),
        })
        self._emit("intent_no_progress", action="read", path=normalized,
                   start_line=start_line, result="ALREADY_PRESENT",
                   pending_intent=pending, message="apply the edit now",
                   current_span=result["current_span"], no_progress=True, step=step)
        return result


def delivered_source_observation(runtime: object, path: str, start_line: int) -> dict | None:
    """Build an ALREADY_PRESENT observation from final request-delivery evidence."""
    owner = getattr(runtime, "owner", None)
    raw_root = getattr(owner, "root", None)
    root = Path(raw_root).resolve() if raw_root is not None else None
    relative = normalize_path(path, root) if root is not None else normalize_path(path)
    serialized = getattr(runtime, "_empirical_serialized_sources", {})
    row = serialized.get(relative) if isinstance(serialized, Mapping) else None
    revisions = getattr(runtime, "_empirical_current_revisions", {})
    current = revisions.get(relative) if isinstance(revisions, Mapping) else None
    if current is None:
        artifacts = getattr(runtime, "artifact_hashes", {})
        current = artifacts.get(relative) if isinstance(artifacts, Mapping) else None
    if (int(start_line) == 0 and isinstance(row, Mapping)
            and row.get("full") is True and row.get("request_hash")
            and current is not None and str(row.get("revision")) == str(current)):
        reference = row.get("reference", f"frame:{relative}@{current}")
        return {
            "schema": "stateframe-read-dedup/1",
            "status": "ALREADY_PRESENT",
            "path": relative,
            "revision": row.get("revision", "UNKNOWN"),
            "reference": reference,
            "message": f"already present in frame (ref) {reference}",
        }
    return None


def render_pending_intent(payload: Mapping[str, object] | PendingIntent | None) -> str:
    """Render the exact model-facing reminder, bounded by typed summaries."""
    if payload is None:
        return ""
    if isinstance(payload, PendingIntent):
        items = [payload.to_dict()]
    elif isinstance(payload, Mapping):
        raw_items = payload.get("items")
        if isinstance(raw_items, Sequence) and not isinstance(raw_items, (str, bytes)):
            items = [item for item in raw_items if isinstance(item, Mapping)]
        elif isinstance(payload.get("pending_intent"), Mapping):
            items = [payload["pending_intent"]]
        else:
            items = []
    else:
        items = []
    if not items:
        return ""
    lines = [PENDING_INTENT_HEADER]
    for item in items:
        path = normalize_path(item.get("target_path"))
        summary = _compact(item.get("summary"))
        if not path or not summary:
            continue
        lines.append(f"YOUR LAST DECISION (not yet applied): edit {path}: {summary}")
        symbol = _normalize_symbol(item.get("target_symbol"))
        if symbol:
            lines.append(f"target_symbol: {symbol}")
    return "\n".join(lines) if len(lines) > 1 else ""


__all__ = [
    "F4_MODE_ENV", "MODE_ENV", "MODEL_REJECTED_EDIT", "PENDING_INTENT_HEADER",
    "f4_enabled", "f4_mode", "split_by_reference_paths", "MODEL_STATED", "PendingIntent", "SCHEMA",
    "IntentCarryState", "enabled", "extract_pending_intents", "mode",
    "delivered_source_observation", "normalize_path", "render_pending_intent",
    "unapplied_rejected_definitions",
]

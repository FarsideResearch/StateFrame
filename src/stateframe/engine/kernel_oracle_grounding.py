"""Opt-in grounding for host-generated oracle expectations.

The ordinary StateFrame path deliberately does not call this module.  When
``STATEFRAME_ORACLE_GROUNDING=on`` is enabled, a host assertion is allowed to
be a violation only when every product-facing literal in its expectation is
present in the public ticket or in explicitly supplied public API/contract
material.  This is a small boundary policy, not a prose interpreter: an
unbound expectation is evidence of an incomplete oracle and is reduced to
``UNKNOWN`` with ``UNBOUND_OBSERVABLE``.

The clause extractor is source-span based.  Newlines inside prose are soft
wraps; Markdown list items and fenced code blocks stay atomic, while blank
lines and Markdown hard breaks remain boundaries.  No positional line-to-
clause alignment is used.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence


MODE_ENV = "STATEFRAME_ORACLE_GROUNDING"
SCHEMA = "stateframe-oracle-grounding/1"
UNKNOWN = "UNKNOWN"
VIOLATED = "VIOLATED"
UNBOUND_OBSERVABLE = "UNBOUND_OBSERVABLE"
GROUNDED = "GROUNDED"

_BULLET = re.compile(r"^\s*(?:[-+*]|•|\d+[.)])\s+")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")

# These names describe the wire shape of a check.  They are not product
# vocabulary and therefore are not themselves normative expected literals.
_STRUCTURAL_KEYS = frozenset({
    "args", "kwargs", "calls", "cases", "base", "variant", "input_index",
    "path", "paths", "anchor", "total_path", "component_paths", "selector",
    "selection", "command", "entry_point", "entry", "target", "target_file",
    "target_path", "symbol", "function", "module", "fixture", "fixtures",
    "input", "inputs", "expected", "expectation", "oracle", "assertion",
    "predicate", "operator", "template", "provenance", "metadata", "source",
    "source_id", "source_span_ids", "support_span_ids", "evidence", "evidence_ids",
    "public_test", "public_tests", "public_source", "public_sources", "contract",
    "api", "public_api", "public_contract", "binding", "binding_ids", "name",
    "check_id", "obligation_id", "id", "version", "revision", "status", "result",
    "result_class", "receipt_id", "execution", "authority", "scope", "kind", "type",
    "data", "body", "response", "output", "value", "detail", "error", "errors",
    "message", "code", "ok", "status_code", "http_status", "http_code", "contains",
    "error_contains", "preserved", "equivalent", "outputs", "mode", "percent",
    "total", "tested_domain", "obligation", "lifecycle", "admission", "tier", "assertions",
    "unsupported_assertions", "keys", "enum_values", "allowed_values",
})

# These fields are invocation/source metadata.  Do not make their contents an
# oracle requirement; e.g. a fixture path must not become a required product
# field.  Expected status codes/messages remain eligible because their parent
# key is structural but the expected value is normative.
_SKIP_SUBTREES = frozenset({
    "args", "kwargs", "calls", "cases", "input_index", "path", "paths", "anchor",
    "total_path", "component_paths", "selector", "selection", "command", "entry_point",
    "entry", "target", "target_file", "target_path", "symbol", "function", "module",
    "fixture", "fixtures", "provenance", "metadata", "source_id", "source_span_ids",
    "support_span_ids", "evidence", "evidence_ids", "public_test", "public_tests",
    "public_source", "public_sources", "binding", "binding_ids", "check_id",
    "obligation_id", "id", "version", "revision", "receipt_id", "execution",
    "authority", "lifecycle", "admission", "tier",
})

_TEXT_KEYS = (
    "text", "source", "src", "content", "public_source", "public_test_source",
    "public_test_src", "test_source", "test_src", "ticket", "ticket_text",
    "public_ticket", "brief", "brief_text", "description", "clause",
)

_PAYLOAD_TEMPLATES = frozenset({
    "OPTIONAL_INPUT", "OUTPUT_EQUALS", "REJECT_WITH_STATUS",
    "PRESERVE_INPUT", "PRESERVE_EXISTING_VALIDATION", "ERROR_TRANSLATION",
})

_RELATION_MARKERS = {
    "PRESERVE_INPUT": frozenset({"preserved"}),
    "PRESERVE_EXISTING_VALIDATION": frozenset({"preserved"}),
    "REPLAY_EQUIVALENT": frozenset({"equivalent"}),
}

_RELATION_MISSING = {
    "PRESERVE_INPUT": "relation:caller_input_preservation",
    "PRESERVE_EXISTING_VALIDATION": "relation:existing_validation_preservation",
    "REPLAY_EQUIVALENT": "relation:replay_equivalence",
}

_REPAIR_PRESENTATION_KEYS = frozenset({
    "alignment_evidence", "alignment_requirement_id", "literal_grounded",
    "repairable", "structural_diff", "target_file", "target_symbol", "witness",
})


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = source.get(MODE_ENV, "off").strip().lower() or "off"
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _line_records(text: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, body)`` records without using line positions."""
    records: list[tuple[int, int, str]] = []
    start = 0
    length = len(text)
    while start < length:
        index = start
        while index < length and text[index] not in "\r\n":
            index += 1
        end = index
        records.append((start, end, text[start:end]))
        if index >= length:
            break
        if text[index] == "\r" and index + 1 < length and text[index + 1] == "\n":
            start = index + 2
        else:
            start = index + 1
    if not records and text == "":
        return []
    return records


def _normal_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _prose_text(records: Sequence[tuple[int, int, str]]) -> tuple[str, list[int]]:
    chars: list[str] = []
    locations: list[int] = []
    for start, _end, raw in records:
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right <= left:
            continue
        if chars:
            chars.append(" ")
            locations.append(start + left)
        for index in range(left, right):
            char = raw[index]
            if char.isspace():
                char = " "
            chars.append(char)
            locations.append(start + index)
    # Collapse whitespace created by unusual Unicode spacing while retaining a
    # source location for every normalized character.
    normalized: list[str] = []
    normalized_locations: list[int] = []
    for char, location in zip(chars, locations):
        if char == " ":
            if not normalized or normalized[-1] == " ":
                continue
        normalized.append(char)
        normalized_locations.append(location)
    while normalized and normalized[0] == " ":
        normalized.pop(0)
        normalized_locations.pop(0)
    while normalized and normalized[-1] == " ":
        normalized.pop()
        normalized_locations.pop()
    return "".join(normalized), normalized_locations


def _opens_quote(text: str, index: int, char: str) -> bool:
    if char != "'" or index == 0 or index + 1 >= len(text):
        return True
    return not (text[index - 1].isalnum() and text[index + 1].isalnum())


def _sentence_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """Split normalized prose at sentence terminals, not soft wraps."""
    ranges: list[tuple[int, int]] = []
    start = 0
    in_backticks = False
    quote = ""
    index = 0
    while index < len(text):
        char = text[index]
        if char == "`":
            in_backticks = not in_backticks
        elif not in_backticks:
            if quote:
                if char == quote and (index == 0 or text[index - 1] != "\\"):
                    quote = ""
            elif char in "'\"" and _opens_quote(text, index, char):
                quote = char
            elif char in ".!?":
                if (char == "." and index > 0 and index + 1 < len(text)
                        and text[index - 1].isdigit() and text[index + 1].isdigit()):
                    index += 1
                    continue
                end = index + 1
                while end < len(text) and text[end] in ".!?":
                    end += 1
                if text[start:end].strip():
                    ranges.append((start, end))
                start = end
                index = end
                continue
        index += 1
    if text[start:].strip():
        ranges.append((start, len(text)))
    return tuple(ranges)


def _mapped_piece(text: str, locations: Sequence[int], start: int, end: int) -> tuple[int, int, str] | None:
    left = start
    right = end
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if left >= right or not locations:
        return None
    left = min(left, len(locations) - 1)
    right = min(max(right, left + 1), len(locations))
    return locations[left], locations[right - 1] + 1, text[left:right]


def _append_prose(result: list[tuple[int, int, str]], records: Sequence[tuple[int, int, str]]) -> None:
    text, locations = _prose_text(records)
    for start, end in _sentence_ranges(text):
        piece = _mapped_piece(text, locations, start, end)
        if piece is not None:
            result.append(piece)


def _append_atomic(result: list[tuple[int, int, str]], records: Sequence[tuple[int, int, str]], *, code: bool = False) -> None:
    if not records:
        return
    start = records[0][0]
    end = records[-1][1]
    raw = "\n".join(item[2] for item in records)
    if code:
        clause = raw.strip()
    else:
        clause = _normal_text(raw)
    if clause:
        result.append((start, end, clause))


_ORDERED_START = re.compile(r"^\s*(\d+)[.)]\s+")


def _wrapped_number_line(raw: str) -> bool:
    """STATEFRAME_POST_GREEN_WORK_MENU rule 4 (CommonMark): inside a paragraph, a soft-wrapped
    line that starts with a number other than 1 (``404.  The CLI ...``) continues the
    sentence; only ``1.``/``1)`` may interrupt prose as an ordered list.  Off: never."""
    if os.environ.get("STATEFRAME_POST_GREEN_WORK_MENU", "off").strip().lower() != "on":
        return False
    match = _ORDERED_START.match(raw)
    return bool(match) and int(match.group(1)) != 1


def extract_ticket_spans(ticket: str) -> tuple[tuple[int, int, str], ...]:
    """Return source character spans for logical ticket clauses.

    Prose lines are joined before sentence splitting.  A bullet starts an
    atomic list item, and a fenced code block is one atomic clause.  Blank
    lines and explicit Markdown hard breaks terminate prose groups.
    """
    if not isinstance(ticket, str):
        raise TypeError("ticket must be text")
    records = _line_records(ticket)
    result: list[tuple[int, int, str]] = []
    prose: list[tuple[int, int, str]] = []
    index = 0

    def flush_prose() -> None:
        nonlocal prose
        _append_prose(result, prose)
        prose = []

    while index < len(records):
        record = records[index]
        raw = record[2]
        if not raw.strip():
            flush_prose()
            index += 1
            continue
        fence = _FENCE.match(raw)
        if fence:
            flush_prose()
            block = [record]
            marker = fence.group(1)[0]
            index += 1
            while index < len(records):
                block.append(records[index])
                if re.match(r"^\s*" + re.escape(marker) + r"{3,}\s*$", records[index][2]):
                    index += 1
                    break
                index += 1
            _append_atomic(result, block, code=True)
            continue
        if _BULLET.match(raw) and not (prose and _wrapped_number_line(raw)):
            flush_prose()
            item = [record]
            index += 1
            while index < len(records):
                candidate = records[index]
                if not candidate[2].strip() or _BULLET.match(candidate[2]) or _FENCE.match(candidate[2]):
                    break
                item.append(candidate)
                index += 1
            _append_atomic(result, item)
            continue
        prose.append(record)
        hard_break = bool(re.search(r" {2,}$", raw) or raw.rstrip().endswith("\\"))
        index += 1
        if hard_break:
            flush_prose()
    flush_prose()
    return tuple(result)


def extract_ticket_clauses(ticket: str | None) -> tuple[str, ...]:
    """Return normalized logical clauses; soft wrapping cannot change them."""
    if not isinstance(ticket, str) or not ticket.strip():
        return ()
    return tuple(item[2] for item in extract_ticket_spans(ticket))


def extract_ticket_sentences(ticket: str | None) -> tuple[str, ...]:
    """Compatibility name used by the requirement-alignment rail."""
    return extract_ticket_clauses(ticket)


def extract_clauses(ticket: str | None) -> tuple[str, ...]:
    return extract_ticket_clauses(ticket)


def _value(value: object, *names: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _unique_literal(result: list[object], value: object) -> None:
    marker = (type(value).__name__, repr(value))
    if not any((type(item).__name__, repr(item)) == marker for item in result):
        result.append(copy.deepcopy(value))


def _literal(value: object, result: list[object], path: tuple[str, ...] = (), *,
             skip: bool = False, preserve_payload: bool = False,
             template: str = "") -> None:
    if skip:
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            lowered = key_text.casefold()
            relation_marker = lowered in _RELATION_MARKERS.get(template, ())
            assertion_operator = template == "ERROR_TRANSLATION" and lowered == "error_contains"
            if relation_marker:
                # ``True`` is the harness' fixed marker, not product evidence.
                # The relation is checked separately against explicit prose.
                continue
            if preserve_payload:
                if not assertion_operator:
                    _unique_literal(result, key_text)
            elif lowered not in _STRUCTURAL_KEYS and lowered not in _SKIP_SUBTREES:
                _unique_literal(result, key_text)
            _literal(
                child, result, (*path, lowered),
                skip=(not preserve_payload and lowered in _SKIP_SUBTREES),
                preserve_payload=preserve_payload,
                template=template,
            )
        return
    if _is_sequence(value):
        for child in value:
            _literal(child, result, path, preserve_payload=preserve_payload, template=template)
        return
    if isinstance(value, (str, int, float)) or value is None:
        if isinstance(value, str) and not value.strip():
            return
        _unique_literal(result, value)


def expected_literals(expectation: object) -> tuple[object, ...]:
    """Extract only product-facing literals from an expectation/check."""
    raw = expectation
    sentinel = object()
    template = str(_value(expectation, "template", "template_id", default="") or "").upper()
    candidate = _value(expectation, "expected", default=sentinel)
    if candidate is not sentinel:
        raw = candidate
    else:
        candidate = _value(expectation, "expectation", default=sentinel)
        if candidate is not sentinel:
            raw = candidate
        else:
            candidate = _value(expectation, "oracle", default=sentinel)
            if candidate is not sentinel:
                raw = candidate
    # ``CheckDefinition.expected=None`` is the API's no-expectation value.  It
    # is not an explicit product literal (an explicit null can still be
    # represented inside a mapping such as {"value": None}).
    if raw is None:
        return ()
    result: list[object] = []
    # Envelope fields (template, fixture, entry point, selectors, ...) are
    # metadata.  Once ``expected`` has been unwrapped, however, its mapping is
    # the product payload: keys such as ``status`` and ``path`` can be real
    # output fields and must remain normative.  A small set of selector-based
    # templates retains the old structural treatment for their path metadata.
    preserve_payload = not template or template in _PAYLOAD_TEMPLATES
    _literal(raw, result, preserve_payload=preserve_payload, template=template)
    return tuple(result)


normative_literals = expected_literals


def _text_values(value: object, *, include_contract: bool = True, _seen: set[int] | None = None) -> list[str]:
    """Collect explicitly public text without reading the filesystem."""
    if _seen is None:
        _seen = set()
    if isinstance(value, str):
        return [value] if value.strip() else []
    if value is None or isinstance(value, (bytes, bytearray)):
        return []
    identity = id(value)
    if identity in _seen:
        return []
    _seen.add(identity)
    result: list[str] = []
    if isinstance(value, Mapping):
        for key in _TEXT_KEYS:
            child = value.get(key)
            if isinstance(child, str) and child.strip():
                result.append(child)
            elif isinstance(child, (Mapping, list, tuple)):
                result.extend(_text_values(child, include_contract=include_contract, _seen=_seen))
        if include_contract:
            for key in ("sources", "public_material", "public_api", "repo_public_api",
                        "public_contract", "repo_contract", "contract", "api",
                        "requirements", "contract_cells"):
                if key in value:
                    result.extend(_text_values(value[key], include_contract=include_contract, _seen=_seen))
        return result
    if _is_sequence(value) or isinstance(value, set):
        for child in value:
            result.extend(_text_values(child, include_contract=include_contract, _seen=_seen))
        return result
    return []


def _explicit_public_repr(value: object) -> str:
    """Render an explicitly supplied API/contract object as searchable text."""
    if isinstance(value, (Mapping, list, tuple, set)):
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)
    return ""


def public_material(*, ticket: object = None, ledger: object = None,
                    public_api: object = None, contract: object = None,
                    sources: object = None) -> tuple[str, str]:
    """Return ``(ticket_text, public_api_or_contract_text)``."""
    ticket_parts = _text_values(ticket, include_contract=False)
    code_parts = _text_values(public_api, include_contract=True)
    api_repr = _explicit_public_repr(public_api)
    if api_repr:
        code_parts.append(api_repr)
    code_parts.extend(_text_values(contract, include_contract=True))
    contract_repr = _explicit_public_repr(contract)
    if contract_repr:
        code_parts.append(contract_repr)
    code_parts.extend(_text_values(sources, include_contract=True))
    source_repr = _explicit_public_repr(sources)
    if source_repr:
        code_parts.append(source_repr)
    if isinstance(ledger, Mapping):
        ticket_id = str(ledger.get("ticket_source_id", ""))
        rows = ledger.get("sources", ())
        if isinstance(rows, Mapping):
            rows = tuple(rows.values())
        if _is_sequence(rows):
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                text = row.get("text", row.get("source", row.get("content")))
                if not isinstance(text, str) or not text.strip():
                    continue
                if str(row.get("id", "")) == ticket_id or str(row.get("kind", "")).lower() == "ticket":
                    ticket_parts.append(text)
                else:
                    code_parts.append(text)
        for key in ("ticket", "ticket_text", "public_ticket", "brief"):
            if key in ledger:
                ticket_parts.extend(_text_values(ledger[key], include_contract=False))
        for key in ("public_sources", "public_tests", "public_api", "public_contract", "contract"):
            if key in ledger:
                code_parts.extend(_text_values(ledger[key], include_contract=True))
                rendered = _explicit_public_repr(ledger[key])
                if rendered:
                    code_parts.append(rendered)
    ticket_text = "\n".join(dict.fromkeys(item for item in ticket_parts if item.strip()))
    public_text = "\n".join(dict.fromkeys(item for item in code_parts if item.strip()))
    return ticket_text, public_text


def _forms(value: object) -> tuple[str, ...]:
    if value is None:
        return ("null", "none")
    if isinstance(value, bool):
        return ("true" if value else "false", str(value))
    text = str(value).strip()
    if not text:
        return ()
    forms = [text]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
            if number.is_integer():
                forms.append(str(int(number)))
            forms.append(str(number))
        except (TypeError, ValueError):
            pass
    return tuple(dict.fromkeys(forms))


def _present(value: object, text: str) -> bool:
    normalized_text = re.sub(r"\s+", " ", text).casefold()
    for form in _forms(value):
        if _NUMBER.fullmatch(form):
            pattern = (rf"(?<![A-Za-z0-9_.]){re.escape(form)}"
                       rf"(?![A-Za-z0-9_]|\.(?=\d))")
            if re.search(pattern, text):
                return True
        elif _IDENTIFIER.fullmatch(form):
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(form)}(?![A-Za-z0-9_])"
            if re.search(pattern, text, re.IGNORECASE):
                return True
        elif (form.casefold() in text.casefold()
              or re.sub(r"\s+", " ", form).strip().casefold() in normalized_text):
            return True
    return False


def _relation_is_supported(template: str, material: str) -> bool:
    """Require explicit semantic evidence for relation-only harness markers.

    A fixed ``preserved: true`` or ``equivalent: true`` value says nothing
    about the product.  Relation checks may be grounded only by an explicit
    caller/input ownership, validation-preservation, or replay-equality
    statement.  In particular, negated and optional prose must not become
    affirmative evidence through a keyword match.
    """
    clauses = extract_ticket_clauses(material) or ((material,) if material.strip() else ())
    for clause in clauses:
        text = _normal_text(clause).casefold()
        if not text:
            continue
        if template in {"PRESERVE_INPUT", "PRESERVE_EXISTING_VALIDATION"}:
            if re.search(r"\b(?:must|should|shall)\s+not\s+(?:remain|stay)\s+unchanged\b", text):
                continue
            if re.search(r"\b(?:must|should|shall|do)\s+not\s+(?:be\s+)?(?:preserv\w*|cop\w*)\b", text):
                continue
            if re.search(r"\boptional(?:ly)?\b", text):
                continue
            if template == "PRESERVE_EXISTING_VALIDATION":
                if re.search(r"\b(?:must|should|shall|do)\s+not\s+(?:retain|keep|preserv\w*)\b.{0,40}\bvalidation\b", text):
                    continue
                if re.search(r"\b(?:existing\s+validation|validation\s+behavior|retain\w*\s+validation)\b", text):
                    return True
                continue
            ownership = r"(?:caller|calling|input|payload|argument|request|object)s?"
            preservation = r"(?:unchanged|unmodified|not\s+(?:mutate|modified|changed)|preserv\w*|cop(?:y|ied|ies))"
            if re.search(rf"\b{ownership}\b.{{0,100}}\b{preservation}\b", text):
                return True
            if re.search(rf"\b{preservation}\b.{{0,100}}\b{ownership}\b", text):
                return True
        elif template == "REPLAY_EQUIVALENT":
            if re.search(r"\b(?:must|should|shall|do)\s+not\s+(?:retry|replay|repeat)\b", text):
                continue
            if re.search(r"\bnever\s+(?:retry|replay|repeat)\b", text):
                continue
            if re.search(r"\boptional(?:ly)?\b", text):
                continue
            replay = r"(?:retry|replay\w*|repeat\w*|idempotent)"
            equality = r"(?:same|identical|equivalent|unchanged)"
            output = r"(?:output|result|response|payload|value)"
            if (re.search(rf"\b{replay}\b.{{0,100}}\b{equality}\b.{{0,40}}\b{output}\b", text)
                    or re.search(rf"\b{equality}\b.{{0,40}}\b{output}\b.{{0,100}}\b{replay}\b", text)
                    or re.search(rf"\b{replay}\b.{{0,100}}\b{equality}\b", text)):
                return True
    return False


def _missing_relations(expectation: object, material: str) -> tuple[str, ...]:
    template = str(_value(expectation, "template", "template_id", default="") or "").upper()
    missing = _RELATION_MISSING.get(template)
    if missing is None or _relation_is_supported(template, material):
        return ()
    return (missing,)


@dataclasses.dataclass(frozen=True, slots=True)
class GroundingResult:
    grounded: bool
    literals: tuple[object, ...] = ()
    missing: tuple[object, ...] = ()
    ticket_text_present: bool = False
    public_material_present: bool = False

    @property
    def unknown(self) -> bool:
        return not self.grounded

    @property
    def reason(self) -> str:
        return GROUNDED if self.grounded else UNBOUND_OBSERVABLE

    @property
    def status(self) -> str:
        return "GROUNDED" if self.grounded else UNKNOWN

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "grounded": self.grounded,
            "unknown": self.unknown,
            "status": self.status,
            "reason": self.reason,
            "literals": copy.deepcopy(list(self.literals)),
            "missing": copy.deepcopy(list(self.missing)),
            "ticket_text_present": self.ticket_text_present,
            "public_material_present": self.public_material_present,
        }

    as_dict = to_dict

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]


def ground_expectation(expectation: object, ticket: object = None, *, ticket_text: object = None,
                       ledger: object = None, public_api: object = None,
                       public_contract: object = None, contract: object = None,
                       sources: object = None) -> GroundingResult:
    """Ground an expectation against ticket and explicit public material."""
    if ticket is None:
        ticket = ticket_text
    ticket_value, public_value = public_material(
        ticket=ticket, ledger=ledger, public_api=public_api,
        contract=public_contract if public_contract is not None else contract,
        sources=sources,
    )
    literals = expected_literals(expectation)
    material = "\n".join(item for item in (ticket_value, public_value) if item)
    relation_missing = _missing_relations(expectation, material)
    if not literals and not relation_missing:
        return GroundingResult(True, (), (), bool(ticket_value), bool(public_value))
    missing = tuple(value for value in literals if not _present(value, material)) + relation_missing
    return GroundingResult(
        grounded=not missing,
        literals=literals,
        missing=missing,
        ticket_text_present=bool(ticket_value),
        public_material_present=bool(public_value),
    )


ground_check = ground_expectation
check_grounding = ground_expectation


def ground_verdict(row: Mapping[str, object], *, ticket: object = None,
                   ledger: object = None, public_api: object = None,
                   public_contract: object = None, contract: object = None,
                   sources: object = None) -> dict[str, object]:
    """Return a model-facing row with unbound violations made non-actionable."""
    result = copy.deepcopy(dict(row))
    if not enabled():
        return result
    sentinel = object()
    expected_value = sentinel
    for key in ("expected", "expectation", "want", "oracle"):
        if key in result:
            expected_value = result[key]
            break
    # Verdict metadata (id, clause, witness, observed, ...) is not itself an
    # expectation.  Only ground a row when its producer supplied one.
    if expected_value is sentinel:
        return result
    grounding_input = {"expected": expected_value}
    for key in ("template", "template_id"):
        if key in result:
            grounding_input[key] = result[key]
    grounding = ground_expectation(
        grounding_input, ticket=ticket, ledger=ledger, public_api=public_api,
        public_contract=public_contract, contract=contract, sources=sources,
    )
    result["oracle_grounding"] = grounding.to_dict()
    if not grounding.unknown:
        return result
    result["reason"] = UNBOUND_OBSERVABLE
    return cap_unbound_presentation(result)


def cap_unbound_presentation(row: Mapping[str, object]) -> dict[str, object]:
    """Apply the final non-actionable presentation cap to an unbound row."""
    result = copy.deepcopy(dict(row))
    if not enabled() or str(result.get("reason", "")).upper() != UNBOUND_OBSERVABLE:
        return result
    result["verdict"] = UNKNOWN
    for key in ("status", "result_class", "outcome", "underlying_status"):
        if key in result and str(result[key]).upper() in {"VIOLATED", "COUNTEREXAMPLE"}:
            result[key] = UNKNOWN
    result["reason"] = UNBOUND_OBSERVABLE
    result["required_edit"] = False
    result["actionable"] = False
    result["blocking"] = False
    for key in _REPAIR_PRESENTATION_KEYS:
        result.pop(key, None)
    return result


def ground_verdicts(rows: Iterable[Mapping[str, object]], **kwargs: object) -> list[dict[str, object]]:
    return [ground_verdict(row, **kwargs) for row in rows if isinstance(row, Mapping)]


_PROBE_ORIGINAL_FIELD = "_oracle_grounding_original_fields"
_PROBE_ADDED_FIELD = "_oracle_grounding_added_fields"


def _probe_original_value(result: Mapping[str, object], key: str) -> tuple[object, bool]:
    stored_key = f"oracle_original_{key}"
    if stored_key in result:
        return copy.deepcopy(result[stored_key]), True
    return result.get(key), False


def _capture_probe_originals(result: dict[str, object]) -> None:
    """Keep display-demoted expectations and fields available for rehydration."""
    for key in ("expected", "expectation", "want"):
        if key not in result:
            continue
        stored_key = f"oracle_original_{key}"
        if stored_key not in result and result[key] != UNKNOWN:
            result[stored_key] = copy.deepcopy(result[key])
    original = result.get(_PROBE_ORIGINAL_FIELD)
    original_fields = dict(original) if isinstance(original, Mapping) else {}
    for key in (
            "status", "verdict", "reason", "required_edit", "actionable", "blocking",
            "coverage", "result_class", "outcome", "underlying_status",
            *_REPAIR_PRESENTATION_KEYS):
        if key in result and key not in original_fields:
            original_fields[key] = copy.deepcopy(result[key])
    if original_fields:
        result[_PROBE_ORIGINAL_FIELD] = original_fields


def _restore_probe_originals(result: dict[str, object]) -> None:
    original = result.get(_PROBE_ORIGINAL_FIELD)
    if isinstance(original, Mapping):
        for key, value in original.items():
            result[str(key)] = copy.deepcopy(value)
    added = result.get(_PROBE_ADDED_FIELD)
    if isinstance(added, Sequence) and not isinstance(added, (str, bytes, bytearray)):
        for key in added:
            result.pop(str(key), None)
    result.pop(_PROBE_ADDED_FIELD, None)


def ground_probe_expectation(row: Mapping[str, object], *, ticket: object = None,
                             ledger: object = None, public_api: object = None,
                             public_contract: object = None, contract: object = None,
                             sources: object = None) -> dict[str, object]:
    """Make an unbound probe expectation an explicit, non-actionable unknown."""
    result = copy.deepcopy(dict(row))
    if not enabled():
        return result
    _capture_probe_originals(result)
    candidates: list[tuple[str, GroundingResult]] = []
    original_values: dict[str, object] = {}
    for key in ("expected", "expectation", "want"):
        if key not in result and f"oracle_original_{key}" not in result:
            continue
        value, stored = _probe_original_value(result, key)
        if value in (None, "", UNKNOWN):
            continue
        original_values[key] = copy.deepcopy(value)
        grounding_input = {"expected": value}
        for template_key in ("template", "template_id"):
            if template_key in result:
                grounding_input[template_key] = result[template_key]
        candidates.append((key, ground_expectation(
            grounding_input, ticket=ticket, ledger=ledger,
            public_api=public_api, public_contract=public_contract,
            contract=contract, sources=sources)))
    if not candidates:
        return result
    literals: list[object] = []
    missing: list[object] = []
    ticket_present = False
    public_present = False
    for _key, item in candidates:
        for value in item.literals:
            _unique_literal(literals, value)
        for value in item.missing:
            _unique_literal(missing, value)
        ticket_present = ticket_present or item.ticket_text_present
        public_present = public_present or item.public_material_present
    grounding = GroundingResult(
        grounded=not missing, literals=tuple(literals), missing=tuple(missing),
        ticket_text_present=ticket_present, public_material_present=public_present,
    )
    result["oracle_grounding"] = grounding.to_dict()
    if grounding.unknown:
        result.update({
            "status": UNKNOWN, "verdict": UNKNOWN, "reason": UNBOUND_OBSERVABLE,
            "required_edit": False, "actionable": False, "coverage": UNKNOWN,
        })
        for status_key in ("result_class", "outcome", "underlying_status"):
            if (status_key in result
                    and str(result[status_key]).upper() in {"VIOLATED", "COUNTEREXAMPLE"}):
                result[status_key] = UNKNOWN
        for key, item in candidates:
            if item.unknown and key in result:
                result[key] = UNKNOWN
        added = result.get(_PROBE_ADDED_FIELD)
        added_keys = set(added) if isinstance(added, Sequence) and not isinstance(
            added, (str, bytes, bytearray)) else set()
        for key in ("status", "verdict", "reason", "required_edit", "actionable", "blocking",
                    "coverage", "result_class", "outcome", "underlying_status",
                    *_REPAIR_PRESENTATION_KEYS):
            if key not in result and key not in result.get(_PROBE_ORIGINAL_FIELD, {}):
                continue
            if key not in result.get(_PROBE_ORIGINAL_FIELD, {}):
                added_keys.add(key)
        result[_PROBE_ADDED_FIELD] = sorted(added_keys)
        return cap_unbound_presentation(result)
    for key, value in original_values.items():
        result[key] = copy.deepcopy(value)
    _restore_probe_originals(result)
    return result


def ground_probe_obligations(rows: Iterable[Mapping[str, object]], *, ticket: object = None,
                             ledger: object = None, public_api: object = None,
                             public_contract: object = None, contract: object = None,
                             sources: object = None) -> list[dict[str, object]]:
    return [ground_probe_expectation(
        row, ticket=ticket, ledger=ledger, public_api=public_api,
        public_contract=public_contract, contract=contract, sources=sources,
    ) for row in rows if isinstance(row, Mapping)]


def admission_reason(expectation: object, *, ticket: object = None, ledger: object = None,
                     public_api: object = None, public_contract: object = None,
                     contract: object = None, sources: object = None) -> str | None:
    if not enabled():
        return None
    result = ground_expectation(
        expectation, ticket=ticket, ledger=ledger, public_api=public_api,
        public_contract=public_contract, contract=contract, sources=sources,
    )
    return result.reason if result.unknown else None


def required_edit_allowed(row: Mapping[str, object]) -> bool:
    return not (enabled() and str(row.get("reason", "")).upper() == UNBOUND_OBSERVABLE)


__all__ = [
    "GROUNDED", "MODE_ENV", "SCHEMA", "UNKNOWN", "UNBOUND_OBSERVABLE", "VIOLATED",
    "GroundingResult", "admission_reason", "check_grounding", "enabled", "expected_literals",
    "extract_clauses", "extract_ticket_clauses", "extract_ticket_sentences", "extract_ticket_spans",
    "ground_check", "ground_expectation", "ground_probe_expectation", "ground_probe_obligations",
    "ground_verdict", "ground_verdicts", "cap_unbound_presentation", "mode",
    "normative_literals", "public_material", "required_edit_allowed",
]

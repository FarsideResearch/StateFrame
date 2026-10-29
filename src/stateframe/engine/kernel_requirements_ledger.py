"""Host-owned, exact-span requirements ledger for requirements ledger.

This module is deliberately small and data-oriented. It accepts only the
public ticket text (and, optionally, explicitly supplied public material),
creates immutable UTF-8 byte spans, and records a conservative interpretation
of the small set of ticket shapes used by the public task corpus. Text that
does not match one of those shapes is retained as unresolved; this compiler
never turns a plausible sentence into a stronger requirement by guessing.

The runtime imports this module lazily behind the requirements-ledger or
requirement-alignment switches. The off path therefore does not add fields,
journal records, or reads.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


ENV = "STATEFRAME_REQUIREMENTS_LEDGER"
SCHEMA_VERSION = "requirements-ledger/1"
SCHEMA_PATH = "research/design-55/requirements-ledger.schema.json"
POLICY_ID = "design-55-precision-first/1"

_FORBIDDEN_PARTS = frozenset({"private", "hidden_tests", "near_miss", "gold", "oracle"})
# Ledger schema quantifier enum. The clause discriminator may emit natural-language universal forms ("each <noun>" crashed
# validate_ledger before request 1 on 10-22); they map to the schema value using the discriminator's own universal set
# (kernel_clause_discriminator.py universal checks). Anything else outside the enum is not a semantic answer.
_LEDGER_QUANTIFIERS = frozenset({"universal_in_stated_scope", "existential", "finite_enumerated", "unresolved"})
_UNIVERSAL_QUANTIFIER_FORMS = frozenset({"universal", "universal_in_stated_scope", "all", "every", "each"})


def _schema_quantifier(value: object) -> str | None:
    """Map a semantic quantifier onto the ledger enum; None means "no usable semantic quantifier"."""
    if not isinstance(value, str):
        return None
    folded = value.strip().casefold()
    if not folded or folded == "unspecified":
        return None
    if folded in _UNIVERSAL_QUANTIFIER_FORMS:
        return "universal_in_stated_scope"
    return folded if folded in _LEDGER_QUANTIFIERS else None


def _is_unmapped_quantifier(value: object) -> bool:
    """A non-empty semantic quantifier that is neither 'unspecified' nor mappable onto the ledger enum."""
    if value is None:
        return False
    if isinstance(value, str) and value.strip().casefold() in {"", "unspecified"}:
        return False
    return _schema_quantifier(value) is None
_NORMATIVE = re.compile(
    r"\b(?:must|shall|should|need(?:s)?|required|requires|valid|accepted|accepts|"
    r"default(?:s|ed)?|return(?:s|ed)?|raise(?:s|d)?|preserve|preserving|"
    r"remain(?:s|ed)?|optional|thread|pass(?:-through)?|without|compute|"
    r"calculate|reject|error|status|compatible|compatibility|alias|"
    r"non[- ]?mutat(?:e|ion)|round(?:ed|ing)?|exact|thread(?:s)?|event\s+loops?)\b",
    re.IGNORECASE,
)
_CONTEXT_ONLY = re.compile(
    r"\b(?:implementation\s+notes?|not\s+(?:a\s+)?public\s+(?:field|interface)|"
    r"prose\s+only|for\s+context|example\s+only)\b",
    re.IGNORECASE,
)
_IMPERATIVE_INSTRUCTION = re.compile(
    r"(?:^|\b)(?:as\s+soon\s+as|before|after|when|once|until|while)\b|"
    r"\b(?:add|avoid|call|compute|continue|create|delete|ensure|extend|keep|"
    r"make|preserve|read|reject|renew|retain|run|set|stop|use|wait|write)\b",
    re.IGNORECASE,
)
_BACKTICK = re.compile(r"`([^`\n]+)`")
_QUOTED = re.compile(r"(?:'([^'\n]+)'|\"([^\"\n]+)\")")
_HTTP = re.compile(r"\b(?:HTTP\s+|status\s+)(\d{3})\b", re.IGNORECASE)
_FIELD_NEAR_NAME = re.compile(
    r"\b(?:optional|field|key|argument|parameter|input|setting)\s+`([A-Za-z_][\w.-]*)`",
    re.IGNORECASE,
)
# A token followed by prose parentheses is not a call.  Keep the syntax
# conservative: a genuine call has no whitespace between its name and `(`;
# source binding/observable evidence remains the authority for broader forms.
_EXPLICIT_CALL = re.compile(r"\b([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)?)\([^\n)]*\)")
_LAYER_WORDS = frozenset({
    "model", "models", "parser", "merge", "validation", "service", "api", "cli",
    "endpoint", "handler", "controller", "repository", "worker",
})
_VALUE_WORDS = frozenset({
    "private", "public", "standard", "express", "active", "inactive", "enabled",
    "disabled", "none", "null", "true", "false", "default",
})


class LedgerError(ValueError):
    """Raised when a ledger violates a host-owned invariant."""


_TICKET_CACHE: dict[tuple[str, str, str, str], tuple[dict[str, Any], ...]] = {}
_PARSE_COUNTS: dict[tuple[str, str], int] = {}


def mode() -> str:
    value = os.environ.get(ENV, "off").lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _revision(value: object, fallback: str) -> str:
    """Return a compact, stable revision without serializing host bindings inline."""
    if value in (None, ""):
        return fallback
    if isinstance(value, Mapping) or (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))):
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), default=str).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()
    text = str(value).strip()
    if not text:
        return fallback
    return text if len(text) <= 512 else sha256_text(text)


def _safe_public_path(value: object, *, default: str) -> str:
    raw = str(value or default).replace("\\", "/")
    path = PurePosixPath(raw)
    if (path.is_absolute() or ".." in path.parts or path.as_posix() != raw
            or any(part.startswith(".") for part in path.parts)
            or any(part.lower() in _FORBIDDEN_PARTS or "backup" in part.lower()
                   for part in path.parts)):
        raise LedgerError("unsafe public ticket/source path")
    return path.as_posix()


def _source_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return None
    for key in ("text", "source", "src", "content", "public_source"):
        if isinstance(value.get(key), str):
            return str(value[key])
    return None


def _source_rows(value: object, *, kind: str, prefix: str) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        if _source_text(value) is not None and value.get("path") is not None:
            items = [(value.get("path"), value)]
        else:
            items = list(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [
            (item.get("path"), item)
            for item in value
            if isinstance(item, Mapping) and item.get("path") is not None
        ]
    else:
        items = []
    result: list[dict[str, Any]] = []
    for index, (raw_path, payload) in enumerate(items, 1):
        text = _source_text(payload)
        if not isinstance(text, str):
            continue
        path = _safe_public_path(raw_path, default=f"{kind}/{index}.txt")
        digest = sha256_text(text)
        revision = str(payload.get("revision", "")) if isinstance(payload, Mapping) else ""
        result.append({
            "id": f"{prefix}{index}",
            "kind": kind,
            "path": path,
            "revision": _revision(revision, f"sha256:{digest}"),
            "sha256": digest,
            "encoding": "utf-8",
            "text": text,
        })
    return result


def _byte_span(text: str, start_char: int, end_char: int) -> tuple[int, int, str]:
    if not (0 <= start_char <= end_char <= len(text)):
        raise LedgerError("invalid character span")
    encoded = text.encode("utf-8")
    start = len(text[:start_char].encode("utf-8"))
    end = len(text[:end_char].encode("utf-8"))
    quote = encoded[start:end].decode("utf-8")
    if not quote:
        raise LedgerError("ledger spans must contain text")
    return start, end, quote


def exact_span(source: str, start_byte: int, end_byte: int) -> str:
    """Host-slice and verify a half-open UTF-8 byte span."""
    raw = source.encode("utf-8")
    if type(start_byte) is not int or type(end_byte) is not int:
        raise LedgerError("span offsets must be integers")
    if start_byte < 0 or end_byte < start_byte or end_byte > len(raw):
        raise LedgerError("span offsets are outside the source")
    try:
        return raw[start_byte:end_byte].decode("utf-8")
    except UnicodeDecodeError as error:
        raise LedgerError("span offsets must land on UTF-8 boundaries") from error


def _ticket_segments(text: str) -> tuple[tuple[int, int, str], ...]:
    """Split only at sentence/newline boundaries; never at conjunctions."""
    if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
        from kernel_oracle_grounding import extract_ticket_spans
        segments = []
        for start_char, end_char, _clause in extract_ticket_spans(text):
            start_byte, end_byte, quote = _byte_span(text, start_char, end_char)
            segments.append((start_byte, end_byte, quote))
        return tuple(segments)
    if os.environ.get("STATEFRAME_REQUIREMENT_ALIGNMENT", "off").strip().lower() == "on":
        from kernel_requirement_alignment import extract_ticket_spans
        segments = []
        for start_char, end_char, _sentence in extract_ticket_spans(text):
            start_byte, end_byte, quote = _byte_span(text, start_char, end_char)
            segments.append((start_byte, end_byte, quote))
        return tuple(segments)
    segments: list[tuple[int, int, str]] = []
    # A line is the safest unit for Markdown briefs: a semicolon or ``and``
    # remains in the same source span and therefore cannot become a stronger
    # pair of requirements.  Sentence punctuation inside a line is split only
    # when it is followed by whitespace; this keeps HTTP/status values intact.
    pattern = re.compile(r"[^\n]+(?:\n|$)")
    matches = tuple(pattern.finditer(text)) or ((re.match(r".*", text),) if text else ())
    for match in matches:
        if match is None:
            continue
        raw_start, raw_end = match.span()
        line = text[raw_start:raw_end].rstrip("\r\n")
        if not line.strip():
            continue
        line_offset = len(line) - len(line.lstrip())
        start = raw_start + line_offset
        end = raw_start + len(line.rstrip())
        # Keep a wrapped paragraph together where a line clearly continues
        # without terminal punctuation.  The public briefs use one normative
        # sentence per line, so this conservative choice is deterministic.
        start_byte, end_byte, quote = _byte_span(text, start, end)
        segments.append((start_byte, end_byte, quote))
    return tuple(segments)


_TASK_METADATA_LABEL = re.compile(r"(?<![A-Za-z0-9_])(?:#{1,6}\s*)?(Title|Symptom|Focus)\s*:", re.IGNORECASE)


def _task_metadata_label(text: str) -> str | None:
    """Return labels that describe a task instead of stating a verifiable requirement."""
    match = _TASK_METADATA_LABEL.search(text)
    return match.group(1).lower() if match else None


def _logical_clause(value: str) -> str:
    if (os.environ.get("STATEFRAME_REQUIREMENT_ALIGNMENT", "off").strip().lower() != "on"
            and os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() != "on"):
        return value
    if (os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on"
            and value.lstrip().startswith(("```", "~~~"))):
        return value.strip()
    return re.sub(r"\s+", " ", value).strip()


def _is_normative(text: str) -> bool:
    return bool(_NORMATIVE.search(text) or _IMPERATIVE_INSTRUCTION.search(text)) \
        and not bool(_CONTEXT_ONLY.search(text) and not _IMPERATIVE_INSTRUCTION.search(text))


def _is_explicit_context_only(text: str) -> bool:
    """Classify context only with an explicit, non-imperative justification."""
    return bool(_CONTEXT_ONLY.search(text)) and not bool(_IMPERATIVE_INSTRUCTION.search(text))


def _span_row(source: Mapping[str, Any], index: int, start: int, end: int,
              quote: str, contexts: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "id": f"S{index}",
        "source_id": source["id"],
        "source_sha256": source["sha256"],
        "start_byte": start,
        "end_byte": end,
        # ``quote`` is supplied only after _byte_span sliced the source.  The
        # validation pass below slices again and rejects any tampering.
        "quote": quote,
        "context_span_ids": list(contexts),
    }


def _literal_value(value: str) -> object:
    value = value.strip().rstrip(".,;:")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", value):
            return float(value)
    except ValueError:
        pass
    return value


def _quoted_values(text: str) -> list[object]:
    values: list[object] = []
    for match in _BACKTICK.finditer(text):
        values.append(_literal_value(match.group(1)))
    for match in _QUOTED.finditer(text):
        values.append(_literal_value(match.group(1) or match.group(2) or ""))
    return values


def _field_name(text: str, full_text: str) -> str | None:
    match = _FIELD_NEAR_NAME.search(text) or _FIELD_NEAR_NAME.search(full_text)
    if match:
        return match.group(1)
    # A sentence such as ``When the key is omitted, private is the default``
    # may name the field in the preceding sentence.  Only use a name adjacent
    # to an explicit field/setting marker; never use an arbitrary class/helper.
    return None


def _field_candidates(text: str) -> tuple[str, ...]:
    names = [match.group(1) for match in _FIELD_NEAR_NAME.finditer(text)]
    return tuple(dict.fromkeys(names))


def _binding_name_candidates(text: str) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for token in _BACKTICK.findall(text):
        token = token.strip()
        if not token or " " in token:
            continue
        if re.fullmatch(r"[A-Za-z_][\w.-]*(?:\([^)]*\))?", token):
            bare = token.split("(", 1)[0]
            # A backticked identifier is often prose (``build (the client)``).
            # Only an adjacent call, a qualified symbol, or an explicitly
            # declared input field is source-shaped enough to bind here.
            if "(" in token or "." in bare:
                result.append((bare, "entrypoint"))
    for match in _EXPLICIT_CALL.finditer(text):
        result.append((match.group(1), "entrypoint"))
    for field in _field_candidates(text):
        result.append((field, "input_field"))
    return tuple(dict.fromkeys(result))


def _values_after_valid(text: str) -> list[object]:
    values = _quoted_values(text)
    fields = set(_field_candidates(text))
    result = [value for value in values if str(value) not in fields]
    if result:
        return list(dict.fromkeys(result))
    # Support the compact public shape ``private and public are valid`` only
    # when each value is a simple word adjacent to the validity phrase.
    match = re.search(r"(?:only|values?\s+(?:are|include))\s+(.+?)\s+(?:are|is)\s+valid", text, re.I)
    if not match:
        match = re.search(r"(.+?)\s+(?:are|is)\s+valid", text, re.I)
    if not match:
        return []
    words = re.findall(r"[A-Za-z_][\w.-]*", match.group(1))
    return list(dict.fromkeys(word for word in words if word.lower() not in {"only", "and", "values"}))


def _layer_names(text: str) -> list[str]:
    marker = re.search(r"\b(?:through|across|between)\s+(.+?)(?:[.;]|$)", text, re.I)
    haystack = marker.group(1) if marker else text
    names: list[str] = []
    for token in re.findall(r"[A-Za-z_][\w-]*", haystack):
        if token.lower() in _LAYER_WORDS:
            names.append(token.lower())
    return list(dict.fromkeys(names))


def _error_text(text: str) -> str | None:
    match = re.search(r"\b(?:error\s+text|message|text)\s+(?:is\s+)?(?:`([^`\n]+)`|'([^'\n]+)'|\"([^\"\n]+)\")", text, re.I)
    if match:
        return next((item for item in match.groups() if item is not None), None)
    return None


def _default_value(text: str) -> object | None:
    match = re.search(r"\bdefaults?\s+(?:to|as)\s+(`[^`\n]+`|'[^'\n]+'|\"[^\"\n]+\"|[A-Za-z_][\w.-]*)", text, re.I)
    if not match:
        match = re.search(r"\b([A-Za-z_][\w.-]*)\s+is\s+the\s+default\b", text, re.I)
    if not match:
        return None
    raw = match.group(1)
    if raw[:1] in "`'\"" and raw[-1:] == raw[:1]:
        raw = raw[1:-1]
    return _literal_value(raw)


def _arithmetic_value(text: str) -> object | None:
    match = re.search(
        r"\b(?:adds?|increase(?:s|d)?|decrease(?:s|d)?|delta(?:\s+of)?)\s+"
        r"(?:[$€£]\s*)?(-?(?:\d+(?:\.\d*)?|\.\d+))\b", text, re.I)
    if not match:
        return None
    return _literal_value(match.group(1))


def _predicate(observable: str, operator: str, expected: object, span_id: str) -> dict[str, Any]:
    return {
        "observable": observable,
        "operator": operator,
        "expected": copy.deepcopy(expected),
        "support_span_ids": [span_id],
    }


def _recognized_contract(quote: str, span_id: str, full_text: str,
                         bindings: Sequence[Mapping[str, Any]]) -> tuple[str, str, dict[str, Any]] | None:
    """Return ``(kind, template, contract)`` for one complete source span."""
    field = _field_name(quote, full_text)
    default = _default_value(quote)
    http = _HTTP.search(quote)
    error_text = _error_text(quote)
    layers = _layer_names(quote)
    allowed = _values_after_valid(quote)
    expectations: list[dict[str, Any]] = []
    preconditions: list[dict[str, Any]] = []
    templates: list[str] = []
    kind = "behavior"

    if http and error_text is not None:
        kind = "error_behavior"
        templates.append("specified-error-mapping/1")
        expectations.extend((
            _predicate("response.status", "equals", int(http.group(1)), span_id),
            _predicate("response.error_text", "equals", error_text, span_id),
        ))
    if re.search(r"\b(?:must|should|need(?:s)?|keep|preserve).{0,80}\b(?:not\s+)?mutat", quote, re.I) \
            or re.search(r"\bwithout\s+(?:changing|mutating)", quote, re.I):
        templates.append("non-mutation/1")
        expectations.append(_predicate("input", "unchanged", True, span_id))
    if re.search(r"\b(?:compute|calculate|set|total|rounded|rounding|percent|cents?|fee|adds?|subtract(?:s|ed)?|delta|increase(?:s|d)?|decrease(?:s|d)?|price)\b", quote, re.I) \
            and re.search(r"[%$]|\b(?:plus|minus|sum|half\s+up|exact)\b", quote, re.I):
        templates.append("exact-arithmetic/1")
        arithmetic_value = _arithmetic_value(quote)
        expectations.append(_predicate(
            "arithmetic.delta" if arithmetic_value is not None else "arithmetic.result",
            "equals" if arithmetic_value is not None else "relation",
            arithmetic_value if arithmetic_value is not None else quote, span_id))
    # A noun such as ``threads`` or ``event loops`` describes an execution
    # mechanism, not a data-flow contract.  Require an explicit flow verb or
    # directional preposition so prose about implementation machinery remains
    # unresolved until an authorized source observable binds it.
    if layers and re.search(
            r"\b(?:through|across|pass(?:es|ed)?\s+through|carry|forward(?:s|ed)?)\b",
            quote, re.I):
        templates.append("pass-through/1")
        expectations.append(_predicate("data_flow", "relation", layers, span_id))
    if re.search(r"\b(?:compatib|alias|existing\s+.*call|call\s+shape|remain\s+valid|backward)", quote, re.I):
        templates.append("compatibility-alias/1")
        expectations.append(_predicate("call_shape", "relation", quote, span_id))
        kind = "compatibility"
    if default is not None or re.search(r"\bis\s+the\s+default\b", quote, re.I):
        templates.append("explicit-default/1")
        if field:
            preconditions.append(_predicate(f"arguments.{field}", "absent", True, span_id))
            expectations.append(_predicate(f"output.{field}", "equals", default, span_id))
        else:
            expectations.append(_predicate("default_value", "equals", default, span_id))
    if allowed and re.search(r"\b(?:valid|accepted|allowed|permitted)\b", quote, re.I):
        templates.append("enumerated-values/1")
        if field:
            expectations.append(_predicate(f"arguments.{field}", "member_of", allowed, span_id))
        else:
            expectations.append(_predicate("accepted_values", "member_of", allowed, span_id))
    if re.search(r"\b(?:optional|existing\s+\w+[- ]argument|argument\s+alias)", quote, re.I) \
            and not templates:
        templates.append("optional-argument-alias/1")
        expectations.append(_predicate("interface", "present", True, span_id))

    if not templates:
        return None
    action = "exercise the named public interface"
    named = [str(item.get("name")) for item in bindings if item.get("role") == "entrypoint"]
    if named:
        action = f"call {named[0]} with an otherwise valid input"
    contract = {
        "modality": "required",
        "subject_binding_ids": [str(item["id"]) for item in bindings] or ["B-UNRESOLVED"],
        "preconditions": preconditions,
        "action": action,
        "expectations": expectations or [_predicate("contract", "relation", quote, span_id)],
        "quantifier": "universal_in_stated_scope",
        "scope": quote,
        "exception_span_ids": [],
        "unresolved_interpretations": [],
        "derivation_template": templates[0],
    }
    return kind, templates[0], contract


def _binding_rows(text: str, span_id: str, source: Mapping[str, Any],
                  start_line: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (name, role) in enumerate(_binding_name_candidates(text), 1):
        if role == "input_field":
            # The schema models field bindings by their role; the binding kind
            # remains one of the public surface categories.
            kind = "public_symbol"
            binding_role = "input_field"
        else:
            kind = "public_symbol"
            binding_role = "entrypoint"
        rows.append({
            "id": f"B-{span_id}-{index}",
            "kind": kind,
            "name": name,
            "role": binding_role,
            "status": "unresolved",
            "locator": None,
            "support_span_ids": [span_id],
            "public_usage_evidence_ids": [],
        })
    return rows


def _unresolved_contract(span_id: str, quote: str) -> dict[str, Any]:
    return {
        "modality": "unresolved",
        "subject_binding_ids": ["B-UNRESOLVED"],
        "preconditions": [],
        "action": "host review of the unresolved public clause",
        "expectations": [_predicate("contract", "relation", quote, span_id)],
        "quantifier": "unresolved",
        "scope": quote,
        "exception_span_ids": [],
            "unresolved_interpretations": ["No supported Design-55 ticket shape matched exactly."],
        "derivation_template": None,
    }


def _scenario_oracle_provenance(
        source: Mapping[str, Any],
        span_id: str,
        quote: str,
        contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind both scenario and oracle claims to the same public ticket span.

    The ledger may preserve a parsed contract, but it must never imply that a
    nearby test or model explanation supplied the expectation.  Unresolved
    clauses therefore retain an explicit unresolved oracle provenance.
    """
    expectations = contract.get("expectations", ())
    oracle_kind = (
        "ticket_span_expectation"
        if isinstance(expectations, Sequence) and expectations
        else "ticket_span_unresolved"
    )
    source_hash = str(source.get("sha256", ""))
    return (
        {
            "kind": "ticket_span",
            "refs": [span_id],
            "source_id": source.get("id"),
            "source_sha256": source_hash,
            "quote": quote,
        },
        {
            "kind": oracle_kind,
            "refs": [span_id],
            "source_id": source.get("id"),
            "source_sha256": source_hash,
            "expectation_ids": [
                str(item.get("observable"))
                for item in expectations
                if isinstance(item, Mapping) and item.get("observable")
            ],
        },
    )


def _cache_key(ticket_source_id: str, text: str, path: str, revision: str) -> tuple[str, str, str, str]:
    # Revision changes describe the workspace, not a changed ticket.  Reusing
    # this parsed source across revisions is the explicit one-parse invariant.
    del revision
    alignment_mode = os.environ.get("STATEFRAME_REQUIREMENT_ALIGNMENT", "off").strip().lower()
    grounding_mode = os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower()
    if grounding_mode == "on":
        alignment_mode += ":oracle=on"
    return (str(ticket_source_id), sha256_text(text), path, alignment_mode)


def parse_ticket_once(ticket_source_id: str, ticket_text: str, *, path: str = "ticket.md",
                      revision: str = "ticket-v1") -> tuple[dict[str, Any], ...]:
    """Parse one ticket source once per immutable source identity."""
    if not isinstance(ticket_text, str) or not ticket_text.strip():
        raise LedgerError("ticket text must be non-empty UTF-8 text")
    path = _safe_public_path(path, default="ticket.md")
    key = _cache_key(str(ticket_source_id), ticket_text, path, str(revision))
    cached = _TICKET_CACHE.get(key)
    if cached is not None:
        return copy.deepcopy(cached)
    spans = _ticket_segments(ticket_text)
    rows: list[dict[str, Any]] = []
    for index, (start, end, quote) in enumerate(spans, 1):
        row = {
            "span_index": index,
            "start_byte": start,
            "end_byte": end,
            "quote": quote,
            "task_metadata_label": _task_metadata_label(quote),
        }
        rows.append(row)
    _TICKET_CACHE[key] = tuple(copy.deepcopy(rows))
    count_key = (str(ticket_source_id), sha256_text(ticket_text))
    _PARSE_COUNTS[count_key] = _PARSE_COUNTS.get(count_key, 0) + 1
    return tuple(copy.deepcopy(rows))


def clear_cache() -> None:
    _TICKET_CACHE.clear()
    _PARSE_COUNTS.clear()


def ticket_parse_count(ticket_source_id: str, ticket_text: str) -> int:
    return _PARSE_COUNTS.get((str(ticket_source_id), sha256_text(ticket_text)), 0)


def _line_number(text: str, byte_offset: int) -> int:
    prefix = exact_span(text, 0, byte_offset)
    return prefix.count("\n") + 1


def _source_entry(source_id: str, kind: str, path: str, revision: str, text: str) -> dict[str, Any]:
    digest = sha256_text(text)
    return {
        "id": source_id,
        "kind": kind,
        "path": path,
        "revision": _revision(revision, f"sha256:{digest}"),
        "sha256": digest,
        "encoding": "utf-8",
        "text": text,
    }


def _ticket_from_row(row: Mapping[str, Any], root: str | Path | None = None) -> tuple[str, str, str]:
    ticket_id = str(row.get("ticket_source_id") or row.get("task_id") or row.get("id") or "ticket")
    for key in ("_bridge_ordinary_brief", "public_brief", "ticket_text", "brief_text", "msg", "symptom", "clause"):
        value = row.get(key)
        if isinstance(value, str) and value.strip() and not value.endswith((".md", ".txt")):
            return ticket_id, value, str(row.get("ticket_path", "ticket.md"))
    value = row.get("brief")
    if isinstance(value, str) and value.strip():
        if root is not None:
            path = Path(root) / value
            try:
                relative = path.resolve().relative_to(Path(root).resolve())
                safe = _safe_public_path(relative.as_posix(), default="ticket.md")
                if safe.endswith((".md", ".txt")) and path.is_file():
                    return ticket_id, path.read_text(encoding="utf-8"), safe
            except (OSError, UnicodeError, ValueError, LedgerError):
                pass
        return ticket_id, value, str(row.get("ticket_path", "ticket.md"))
    raise LedgerError("public ticket text is unavailable")


def build_ledger(ticket_source_id: str | Mapping[str, Any], ticket_text: str | None = None,
                 *, path: str = "ticket.md", revision: str = "ticket-v1",
                 current_revision: object = None, public_sources: object = None,
                 public_tests: object = None, root: str | Path | None = None) -> dict[str, Any]:
    """Compile one immutable ticket into the requirements ledger storage shape."""
    row: Mapping[str, Any] | None = ticket_source_id if isinstance(ticket_source_id, Mapping) else None
    if row is not None:
        ticket_source_id, ticket_text, path = _ticket_from_row(row, root=root)
        public_sources = public_sources if public_sources is not None else row.get("files", row.get("public_sources", ()))
        public_tests = public_tests if public_tests is not None else row.get("public_tests", row.get("public_test_files", ()))
    source_id = str(ticket_source_id or "ticket")
    if not isinstance(ticket_text, str):
        raise LedgerError("ticket text must be supplied")
    path = _safe_public_path(path, default="ticket.md")
    ticket_digest = sha256_text(ticket_text)
    ticket_source = _source_entry(source_id, "ticket", path, str(revision), ticket_text)
    source_rows = [ticket_source]
    source_rows.extend(_source_rows(public_sources, kind="public_source", prefix="SRC"))
    source_rows.extend(_source_rows(public_tests, kind="public_test", prefix="TEST"))
    parsed = parse_ticket_once(source_id, ticket_text, path=path, revision=str(revision))
    spans: list[dict[str, Any]] = []
    binding_rows: list[dict[str, Any]] = []
    requirements: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    # This binding is intentionally unresolved.  It lets an unresolved
    # normative span satisfy the JSON shape without licensing a guessed symbol.
    binding_rows.append({
        "id": "B-UNRESOLVED", "kind": "artifact", "name": "unresolved-ticket-subject",
        "role": "other", "status": "unresolved", "locator": None,
        "support_span_ids": ["S1"], "public_usage_evidence_ids": [],
    })
    for item in parsed:
        span_id = f"S{len(spans) + 1}"
        span = _span_row(ticket_source, len(spans) + 1, item["start_byte"], item["end_byte"], item["quote"])
        span["id"] = span_id
        spans.append(span)
        clause = _logical_clause(item["quote"])
        metadata_label = item.get("task_metadata_label")
        if isinstance(metadata_label, str):
            coverage.append({
                "span_id": span_id,
                "disposition": "context_only",
                "requirement_refs": [],
                "reason": (
                    f"advisory_unmappable_task_metadata:{metadata_label}; "
                    "this descriptive field cannot create a host-blocking requirement."
                ),
            })
            continue
        local_bindings = _binding_rows(item["quote"], span_id, ticket_source,
                                       _line_number(ticket_text, item["start_byte"]))
        for binding in local_bindings:
            # Stable IDs need not repeat if the same public symbol is named in
            # two clauses; one binding with multiple support spans is safer.
            existing = next((row for row in binding_rows
                             if row.get("name") == binding.get("name")
                             and row.get("role") == binding.get("role")), None)
            if existing is not None:
                existing["support_span_ids"] = list(dict.fromkeys(
                    [*existing.get("support_span_ids", ()), span_id]))
            else:
                binding["id"] = f"B{len(binding_rows)}"
                binding_rows.append(binding)
        associated = [row for row in binding_rows
                      if span_id in row.get("support_span_ids", ())]
        from kernel_aha10_honest_loop import resolve_requirement_clause
        binding_context = row if isinstance(row, Mapping) else {}
        semantic = resolve_requirement_clause(
            clause,
            authorized_bindings=binding_context.get("authorized_bindings", {}),
            observable_bindings=binding_context.get("observable_bindings", {}),
        )
        recognized = _recognized_contract(clause, span_id, ticket_text, associated)
        if recognized is None and semantic.get("status") == "RESOLVED":
            observable = next(iter(semantic.get("observables", ())), "observable")
            recognized = (
                "behavior", "source-bound-call/1", {
                    "modality": "required",
                    "subject_binding_ids": [str(item["id"]) for item in associated]
                    or ["B-UNRESOLVED"],
                    "preconditions": [],
                    "action": f"call {semantic['candidates'][0]} with an authorized input",
                    "expectations": [_predicate(observable, "relation", True, span_id)],
                    "quantifier": "universal_in_stated_scope",
                    "scope": clause,
                    "exception_span_ids": [],
                    "unresolved_interpretations": [],
                    "derivation_template": "source-bound-call/1",
                },
            )
        if recognized is not None:
            kind, _template, contract = recognized
            binding_required = _template in {"pass-through/1", "compatibility-alias/1"} \
                or bool(semantic.get("candidates"))
            if binding_required and semantic.get("status") != "RESOLVED":
                contract["modality"] = "unresolved"
                contract["derivation_template"] = None
                contract["unresolved_interpretations"] = [
                    "Named prose remains unresolved until authorized source and observable evidence bind it.",
                ]
                kind = "unresolved"
            raw_quantifier = semantic.get("quantifier")
            semantic_quantifier = _schema_quantifier(raw_quantifier)
            if semantic_quantifier is None and _is_unmapped_quantifier(raw_quantifier):
                # Unknown task wording never crashes the ledger: record it as unresolved with a truthful note.
                semantic_quantifier = "unresolved"
                contract["unresolved_interpretations"] = list(contract.get("unresolved_interpretations") or []) + [
                    f"Quantifier wording {str(raw_quantifier).strip()[:40]!r} has no ledger mapping; treated as unresolved.",
                ]
            contract["quantifier"] = (
                semantic_quantifier if semantic_quantifier is not None
                else contract.get("quantifier") if contract.get("quantifier") in _LEDGER_QUANTIFIERS
                else "unresolved"
            )
            req_id = f"R{len(requirements) + 1}"
            scenario_provenance, oracle_provenance = _scenario_oracle_provenance(
                source_rows[0], span_id, clause, contract
            )
            requirement = {
                "id": req_id, "version": 1, "kind": kind,
                "source_span_ids": [span_id], "exact_clause": clause,
                "contract": contract, "check_refs": [],
                "scenario_provenance": scenario_provenance,
                "oracle_provenance": oracle_provenance,
                "assessment": {
                    "status": "open", "at_revision": _revision(current_revision, f"sha256:{ticket_digest}"),
                    "reason": "not yet assessed by host", "basis": "none", "policy_ref": None,
                    "evidence_ids": [], "scope": clause,
                    "limitations": ["Scope is the exact public ticket span; no inverse is generated."],
                },
                "supersedes": None,
            }
            requirements.append(requirement)
            coverage.append({"span_id": span_id, "disposition": "mapped" if semantic.get("status") == "RESOLVED" else "unresolved",
                              "requirement_refs": [{"id": req_id, "version": 1}],
                              "reason": f"Matched deterministic template {_template}; semantic status={semantic.get('status')}."})
        elif _is_normative(clause):
            req_id = f"R{len(requirements) + 1}"
            unresolved = _unresolved_contract(span_id, clause)
            scenario_provenance, oracle_provenance = _scenario_oracle_provenance(
                source_rows[0], span_id, clause, unresolved
            )
            requirements.append({
                "id": req_id, "version": 1, "kind": "unresolved",
                "source_span_ids": [span_id], "exact_clause": clause,
                "contract": unresolved, "check_refs": [],
                "scenario_provenance": scenario_provenance,
                "oracle_provenance": oracle_provenance,
                "assessment": {
                    "status": "open", "at_revision": _revision(current_revision, f"sha256:{ticket_digest}"),
                    "reason": "unresolved public ticket interpretation", "basis": "none", "policy_ref": None,
                    "evidence_ids": [], "scope": clause,
                    "limitations": ["No supported template matched; host must not guess."],
                },
                "supersedes": None,
            })
            coverage.append({"span_id": span_id, "disposition": "unresolved",
                              "requirement_refs": [{"id": req_id, "version": 1}],
                              "reason": "Normative language was found but no supported template matched."})
        elif _is_explicit_context_only(clause):
            coverage.append({"span_id": span_id, "disposition": "non_normative",
                              "requirement_refs": [], "reason": "Explicitly marked as implementation/context prose."})
        else:
            coverage.append({"span_id": span_id, "disposition": "context_only",
                              "requirement_refs": [], "reason": "No normative trigger was found."})

    # The unresolved binding's support span is repaired after the first actual
    # span exists; an empty ticket was rejected above.
    binding_rows[0]["support_span_ids"] = [spans[0]["id"]]
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "ticket_source_id": source_id,
        "current_revision": _revision(current_revision, f"sha256:{ticket_digest}"),
        "sources": source_rows,
        "spans": spans,
        "bindings": binding_rows,
        "requirements": requirements,
        "checks": [], "certificates": [], "evidence": [], "receipts": [],
        "disputes": [], "coverage_dispositions": coverage,
        "policy": {
            "id": POLICY_ID, "unknown_blocks_completion": False,
            "revalidate_after_source_edits": True, "max_adjudication_steps": 2,
            "solver_may_edit_status": False, "solver_may_edit_checks": False,
        },
    }
    # v22 stores coordination records in this existing requirements ledger.
    # Keep the extension absent unless the v22 capability is enabled so a
    # requirements-only v21 run retains its exact serialized shape.
    if os.environ.get("STATEFRAME_TOKENLAB_COORDINATED_LEDGER", "off").strip().lower() == "on":
        ledger["coordination"] = {
            "schema_version": "requirements-ledger-coordination/1",
            "work_children": [], "change_sets": [],
            "decision_basis": {
                "published_revision": ledger["current_revision"],
                "stage_ref": None, "evidence_watermark": 0,
                "policy_digest": sha256_text(json.dumps(ledger["policy"], sort_keys=True)),
                "holds": [], "next_action": None,
            },
        }
    validate_ledger(ledger)
    return ledger


compile_ledger = build_ledger
compile_requirements_ledger = build_ledger
build_run_ledger = build_ledger


def _default_schema_path() -> Path:
    # Resolve the schema independently of the checkout layout, including src/ and
    # trials/<t>/ (different depths), so walk up to the first ancestor that has it. A fixed
    # parents[N] silently skipped JSON-schema validation from src/ (validate_ledger is
    # skip-if-absent).
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / SCHEMA_PATH
        if candidate.is_file():
            return candidate
    return here.parent / SCHEMA_PATH


def validate_ledger(ledger: Mapping[str, Any], schema_path: str | Path | None = None) -> bool:
    """Validate JSON shape plus host-only exact-span/reference invariants."""
    if not isinstance(ledger, Mapping):
        raise LedgerError("ledger must be a mapping")
    source_map = {str(row.get("id")): row for row in ledger.get("sources", ())
                  if isinstance(row, Mapping)}
    if not source_map:
        raise LedgerError("ledger must contain sources")
    for row in ledger.get("sources", ()):
        if not isinstance(row, Mapping):
            raise LedgerError("source rows must be mappings")
        text = row.get("text")
        if not isinstance(text, str) or sha256_text(text) != row.get("sha256"):
            raise LedgerError(f"source hash mismatch: {row.get('id')}")
    span_ids: set[str] = set()
    for row in ledger.get("spans", ()):
        if not isinstance(row, Mapping):
            raise LedgerError("span rows must be mappings")
        ident = str(row.get("id", ""))
        if not ident or ident in span_ids:
            raise LedgerError("span ids must be unique and non-empty")
        span_ids.add(ident)
        source = source_map.get(str(row.get("source_id")))
        if source is None:
            raise LedgerError(f"unknown span source: {row.get('source_id')}")
        if row.get("source_sha256") != source.get("sha256"):
            raise LedgerError(f"span source hash mismatch: {ident}")
        quote = exact_span(str(source["text"]), row.get("start_byte"), row.get("end_byte"))
        if quote != row.get("quote"):
            raise LedgerError(f"span quote mismatch: {ident}")
        if any(context not in span_ids for context in row.get("context_span_ids", ())):
            # Context spans may be later in the array; validate after the loop.
            pass
    for row in ledger.get("spans", ()):
        if any(str(context) not in span_ids for context in row.get("context_span_ids", ())):
            raise LedgerError(f"unknown context span in {row.get('id')}")
    requirement_ids = {str(row.get("id")): row for row in ledger.get("requirements", ())
                       if isinstance(row, Mapping)}
    binding_ids = {str(row.get("id")) for row in ledger.get("bindings", ())
                   if isinstance(row, Mapping)}
    for req in requirement_ids.values():
        if not set(req.get("source_span_ids", ())).issubset(span_ids):
            raise LedgerError(f"requirement references unknown span: {req.get('id')}")
        for provenance_key in ("scenario_provenance", "oracle_provenance"):
            provenance = req.get(provenance_key)
            if not isinstance(provenance, Mapping):
                raise LedgerError(f"requirement is missing {provenance_key}: {req.get('id')}")
            refs = provenance.get("refs")
            if (
                not isinstance(provenance.get("kind"), str)
                or not provenance["kind"].strip()
                or not isinstance(refs, Sequence)
                or isinstance(refs, (str, bytes, bytearray))
                or not refs
                or not set(str(ref) for ref in refs).issubset(span_ids)
            ):
                raise LedgerError(f"invalid {provenance_key}: {req.get('id')}")
        contract = req.get("contract", {})
        if not set(contract.get("subject_binding_ids", ())).issubset(binding_ids):
            raise LedgerError(f"requirement references unknown binding: {req.get('id')}")
        if not set(req.get("check_refs", ()) and [ref.get("id") for ref in req.get("check_refs", ())]
                   or ()).issubset({str(row.get("id")) for row in ledger.get("checks", ()) if isinstance(row, Mapping)}):
            raise LedgerError(f"requirement references unknown check: {req.get('id')}")
    covered = [str(row.get("span_id")) for row in ledger.get("coverage_dispositions", ())
               if isinstance(row, Mapping)]
    if set(covered) != span_ids or len(covered) != len(set(covered)):
        raise LedgerError("coverage dispositions must cover each span exactly once")
    if any(str(row.get("span_id")) not in span_ids for row in ledger.get("coverage_dispositions", ())):
        raise LedgerError("coverage references an unknown span")
    coordination = ledger.get("coordination")
    if coordination is not None:
        if not isinstance(coordination, Mapping):
            raise LedgerError("coordination extension must be a mapping")
        if coordination.get("schema_version") != "requirements-ledger-coordination/1":
            raise LedgerError("unsupported coordination extension schema")
        for key in ("work_children", "change_sets"):
            rows = coordination.get(key)
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
                raise LedgerError(f"coordination {key} must be a sequence")
            ids: set[str] = set()
            for row in rows:
                if not isinstance(row, Mapping) or not str(row.get("id", "")):
                    raise LedgerError(f"coordination {key} rows need stable ids")
                ident = str(row["id"])
                if ident in ids:
                    raise LedgerError(f"duplicate coordination id: {ident}")
                ids.add(ident)
        basis = coordination.get("decision_basis")
        if not isinstance(basis, Mapping) or not str(basis.get("published_revision", "")):
            raise LedgerError("coordination decision basis needs published_revision")
    schema = Path(schema_path) if schema_path is not None else _default_schema_path()
    if schema.is_file():
        shape = json.loads(schema.read_text(encoding="utf-8"))
    elif schema_path is None:
        # Releases ship .py only: fall back to the embedded copy instead of silently skipping validation.
        from kernel_ledger_schema import SCHEMA_JSON
        shape = json.loads(SCHEMA_JSON)
    else:
        shape = None
    if shape is not None:
        try:
            import jsonschema  # type: ignore
        except ImportError:
            jsonschema = None
        if jsonschema is not None:
            # The shipped requirements ledger schema predates the v22 extension and is
            # intentionally unchanged outside src_v22. Validate its base
            # contract, then validate owned extensions with their host rules.
            base_shape = dict(ledger)
            base_shape.pop("coordination", None)
            base_shape.pop("spine", None)
            base_shape["requirements"] = [
                {
                    key: value for key, value in requirement.items()
                    if key not in {"scenario_provenance", "oracle_provenance", "spine"}
                }
                for requirement in base_shape.get("requirements", ())
                if isinstance(requirement, Mapping)
            ]
            jsonschema.Draft202012Validator(shape).validate(base_shape)
    if "spine" in ledger:
        # The typed S1 contract is an additive projection of these rows.  Its
        # validator imports lazily so the base ledger remains dependency-free
        # and the existing OFF path keeps its exact shape.
        from kernel_s1_spine import validate_spine_ledger

        validate_spine_ledger(ledger)
    return True


def ledger_json(ledger: Mapping[str, Any], *, indent: int | None = None) -> str:
    validate_ledger(ledger)
    return json.dumps(dict(ledger), sort_keys=True, ensure_ascii=False,
                      indent=indent, separators=None if indent else (",", ":"))


def write_ledger(ledger: Mapping[str, Any], path: str | Path) -> Path:
    validate_ledger(ledger)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(ledger_json(ledger, indent=2) + "\n", encoding="utf-8")
    return destination


def read_ticket(ledger: Mapping[str, Any], *, span_ids: Sequence[str] = (),
                source_id: str | None = None) -> dict[str, Any]:
    """Return host-sliced ticket spans for the model-facing ``read_ticket`` action."""
    validate_ledger(ledger)
    wanted = {str(item) for item in span_ids if str(item)}
    ticket_id = str(source_id or ledger.get("ticket_source_id", ""))
    spans = [row for row in ledger.get("spans", ())
             if isinstance(row, Mapping)
             and str(row.get("source_id")) == ticket_id
             and (not wanted or str(row.get("id")) in wanted)]
    return {
        "schema": "requirements-ledger-ticket-read/1",
        "ticket_source_id": ticket_id,
        "spans": [copy.deepcopy(dict(row)) for row in spans],
        "text": "\n".join(str(row.get("quote", "")) for row in spans),
    }


def host_update_assessments(ledger: Mapping[str, Any], assessments: Sequence[object], *,
                            revision: object = None) -> dict[str, Any]:
    """Apply host assessment results; solver-shaped inputs are never accepted here."""
    result = copy.deepcopy(dict(ledger))
    current = str(revision or result.get("current_revision") or "")
    by_id: dict[str, object] = {}
    for item in assessments:
        if isinstance(item, Mapping):
            ident = item.get("obligation_id", item.get("id", ""))
        elif hasattr(item, "obligation_id"):
            ident = getattr(item, "obligation_id", "")
        else:
            continue
        by_id[str(ident)] = item
    for requirement in result.get("requirements", ()):
        ident = str(requirement.get("id", ""))
        assessment = by_id.get(ident)
        if assessment is None:
            requirement_spans = {
                str(value) for value in requirement.get("source_span_ids", ())
                if str(value)
            }
            requirement_source = str(requirement.get("source_id", ""))
            for candidate in assessments:
                candidate_map = candidate if isinstance(candidate, Mapping) else None
                if candidate_map is None:
                    continue
                explicit_owner = candidate_map.get(
                    "requirement_id",
                    candidate_map.get("clause_id", candidate_map.get("owner_requirement_id", candidate_map.get("owner_id"))),
                )
                if str(explicit_owner or "") == ident:
                    assessment = candidate
                    break
                candidate_spans = {
                    str(value) for value in candidate_map.get("source_span_ids", candidate_map.get("span_ids", ()))
                    if str(value)
                }
                candidate_source = str(candidate_map.get("source_id", ""))
                if (requirement_spans and candidate_spans == requirement_spans
                        and (not requirement_source or not candidate_source or candidate_source == requirement_source)):
                    assessment = candidate
                    break
        if assessment is None:
            continue
        get = assessment.get if isinstance(assessment, Mapping) else lambda key, default=None: getattr(assessment, key, default)
        verdict = str(get("verdict", "UNKNOWN")).upper()
        reason = str(get("reason", "host assessment remains open"))
        status = "satisfied" if verdict == "SATISFIED" else "counterexample" if verdict == "UNSATISFIED" and reason == "COUNTEREXAMPLE" else "open"
        limits = ["Status was reduced by the host from current receipts and admission policy."]
        requirement["assessment"] = {
            **requirement.get("assessment", {}), "status": status,
            "at_revision": current or requirement.get("assessment", {}).get("at_revision", "unknown"),
            "reason": reason, "basis": "acceptance_policy" if status != "open" else "none",
            "policy_ref": POLICY_ID if status != "open" else None,
            "evidence_ids": list(get("evidence_refs", get("evidence", ())) or ()),
            "scope": requirement.get("exact_clause", "host assessment"), "limitations": limits,
        }
    result["current_revision"] = current or result.get("current_revision")
    validate_ledger(result)
    return result


def host_record_receipts(ledger: Mapping[str, Any], receipts: Sequence[object], *,
                         revision: object = None, phase: str = "candidate") -> dict[str, Any]:
    """Project executor receipts into the schema's host-owned receipt rows."""
    if phase not in {"base", "candidate", "control"}:
        raise LedgerError("invalid receipt phase")
    result = copy.deepcopy(dict(ledger))
    rows = list(result.get("receipts", ()))
    known = {str(row.get("id")) for row in rows if isinstance(row, Mapping)}
    checks = {str(row.get("id")): row for row in result.get("checks", ())
              if isinstance(row, Mapping)}

    def value(item: object, key: str, default: object = None) -> object:
        if isinstance(item, Mapping):
            return item.get(key, default)
        return getattr(item, key, default)

    def object_value(item: object) -> object:
        if item is None:
            return {}
        return copy.deepcopy(dict(item)) if isinstance(item, Mapping) else {"value": copy.deepcopy(item)}

    for receipt in receipts:
        check_id = str(value(receipt, "check_id", "")).strip()
        if not check_id:
            continue
        receipt_id = str(value(receipt, "receipt_id", "")).strip()
        if not receipt_id:
            receipt_id = "host-receipt:" + sha256_text(json.dumps(
                {"check": check_id, "observed": value(receipt, "observed"),
                 "expected": value(receipt, "expected"),
                 "revision": value(receipt, "snapshot_digest")},
                sort_keys=True, default=str))
        if receipt_id in known:
            continue
        status = ""
        for status_key in ("underlying_status", "underlying_result", "result",
                           "status", "result_class"):
            candidate = value(receipt, status_key, None)
            if candidate is not None and str(candidate).upper() != "REUSED":
                status = str(candidate).upper()
                break
        outcome = {
            "PASS": "pass", "COUNTEREXAMPLE": "assertion_failure",
            "HARNESS_ERROR": "error", "NOT_EXECUTED": "skip",
        }.get(status, "not_applicable")
        artifact = checks.get(check_id, {}).get("artifact_sha256") if check_id in checks else None
        if not isinstance(artifact, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact):
            raw_check = str(value(receipt, "check_digest", check_id))
            artifact = raw_check.removeprefix("sha256:")
            if not re.fullmatch(r"[0-9a-f]{64}", artifact):
                artifact = sha256_text(raw_check)
        execution = str(value(receipt, "execution", "EXECUTED")).upper()
        observed_status = status in {"PASS", "COUNTEREXAMPLE"}
        attribution = str(value(receipt, "attribution", "target_assertion"))
        if attribution not in {"target_assertion", "other_requirement", "harness", "unknown"}:
            attribution = "unknown"
        row = {
            "id": receipt_id,
            "producer": "host",
            "check_ref": {"id": check_id, "version": 1},
            "phase": phase,
            "repository_revision": str(value(receipt, "snapshot_digest", revision or result.get("current_revision", "unknown"))),
            "artifact_sha256": artifact,
            "environment_ref": str(value(receipt, "environment_digest", "host-runner")),
            "command": ["host-check", check_id],
            "outcome": outcome,
            "assertion_reached": bool(observed_status and execution == "EXECUTED"),
            "preconditions_validated": execution not in {"SETUP_ERROR", "SKIPPED"},
            "attribution": attribution,
            "observed": object_value(value(receipt, "observed")),
            "expected": object_value(value(receipt, "expected")),
            "trace_ref": str(value(receipt, "evidence_ref", "")) or None,
            "determinism": "reproduced" if bool(value(receipt, "reused", False)) else "single_run",
        }
        rows.append(row)
        known.add(receipt_id)
        check = checks.get(check_id)
        if isinstance(check, dict):
            receipt_key = "base_receipt_ids" if phase == "base" else "current_receipt_ids"
            receipt_refs = list(check.get(receipt_key, ()))
            if receipt_id not in receipt_refs:
                receipt_refs.append(receipt_id)
            check[receipt_key] = receipt_refs
    result["receipts"] = rows
    validate_ledger(result)
    return result


def reopen_dirty_requirements(
    ledger: Mapping[str, Any], invalidation: object,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Reopen satisfied requirements whose cited evidence is dirty.

    Requirement rows remain append-only for audit, but their assessment is not
    allowed to remain satisfied after a source edit.  The invalidation object
    is deliberately protocol-shaped so this helper can be used by the live
    graph without importing its implementation.
    """
    result = copy.deepcopy(dict(ledger))
    dirty_refs = set(getattr(invalidation, "dirty_evidence_refs", ()) or ())
    if isinstance(invalidation, Mapping):
        dirty_refs.update(str(item) for item in invalidation.get("evidence_refs", ()) or ())
    notices: list[str] = []
    events = list(result.get("invalidation_events", ()))
    for requirement in result.get("requirements", ()):
        if not isinstance(requirement, dict):
            continue
        assessment = requirement.get("assessment")
        if not isinstance(assessment, dict) or assessment.get("status") != "satisfied":
            continue
        evidence_ids = {
            str(item) for item in assessment.get("evidence_ids", requirement.get("evidence_ids", ())) or ()
        }
        hit = sorted(evidence_ids & dirty_refs)
        if not hit:
            continue
        ident = str(requirement.get("id", "requirement"))
        observed = str(assessment.get("at_revision", "unknown"))
        current = str(getattr(invalidation, "current_revision", "unknown"))
        path = str(getattr(invalidation, "path", "source"))
        notice = (
            f"{ident} reopened: its green was observed on {path}@{observed}, "
            f"now {current}; evidence is dirty ({hit[0]})"
        )[:300]
        assessment.update({
            "status": "open",
            "reason": "EVIDENCE_STALE",
            "basis": "none",
            "policy_ref": None,
            "reopened_from": "satisfied",
            "stale_evidence_ids": hit,
        })
        requirement["assessment"] = assessment
        requirement["invalidation_reason"] = "EVIDENCE_STALE"
        notices.append(notice)
        events.append({"event": "REQUIREMENT_REOPENED", "requirement_id": ident,
                       "reason": "EVIDENCE_STALE", "evidence_ids": hit, "notice": notice})
    result["invalidation_events"] = events
    return result, tuple(notices)


def coverage_complete(ledger: Mapping[str, Any]) -> bool:
    try:
        validate_ledger(ledger)
    except Exception:  # schema validators use their own ValidationError type
        return False
    return True


__all__ = [
    "ENV", "SCHEMA_VERSION", "POLICY_ID", "LedgerError", "build_ledger", "build_run_ledger",
    "clear_cache", "compile_ledger", "compile_requirements_ledger", "coverage_complete",
    "enabled", "exact_span", "host_update_assessments", "ledger_json", "mode", "parse_ticket_once",
    "read_ticket", "sha256_text", "ticket_parse_count", "validate_ledger", "write_ledger",
    "host_record_receipts", "reopen_dirty_requirements",
]

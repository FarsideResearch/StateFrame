"""Opt-in alignment between public ticket sentences and host checks.

The normal StateFrame path deliberately does not import this module.  When the
alignment switch is enabled, this module provides three host-owned facts:
sentence-shaped ticket units, literal evidence for a check-to-sentence match,
and a small structural rendering for a current counterexample.
"""
from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping, Sequence


ENV = "STATEFRAME_REQUIREMENT_ALIGNMENT"
SCHEMA = "stateframe-requirement-alignment/1"

_BULLET = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_BACKTICK = re.compile(r"`([^`\n]+)`")
_QUOTED = re.compile(r"(?:'([^'\n]+)'|\"([^\"\n]+)\")")
_NUMBER = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?(?![A-Za-z_])")
_STATUS = re.compile(r"\b(?:HTTP\s+|status\s+)(\d{3})\b", re.IGNORECASE)
_FIELD_MARKER = re.compile(
    r"\b(?:field|key|argument|parameter|setting|option|priority|mode|value)\s+"
    r"[`'\"]?([A-Za-z_][\w.-]*)[`'\"]?",
    re.IGNORECASE,
)
_COMMON_KEYS = frozenset({
    "body", "error", "error_text", "error_contains", "message", "response",
    "result", "status", "value", "data", "input", "output", "ok",
})
_MISSING = "<missing>"


def mode() -> str:
    value = os.environ.get(ENV, "off").strip().lower() or "off"
    if value not in {"off", "on"}:
        raise ValueError(f"{ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _normal_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _join_lines(lines: Sequence[str]) -> str:
    return _normal_text(" ".join(line.strip() for line in lines if line.strip()))


def _opens_quote(text: str, index: int, char: str) -> bool:
    if char != "'" or index == 0 or index + 1 >= len(text):
        return True
    return not (text[index - 1].isalnum() and text[index + 1].isalnum())


def _split_sentences(text: str) -> list[str]:
    """Split prose at terminal punctuation, respecting quoted code and decimals."""
    result: list[str] = []
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
                piece = _normal_text(text[start:end])
                if piece:
                    result.append(piece)
                start = end
                index = end
                continue
        index += 1
    tail = _normal_text(text[start:])
    if tail:
        result.append(tail)
    return result


def _sentence_boundaries(text: str) -> list[tuple[int, int]]:
    """Return normalized-text sentence boundaries using the same scanner."""
    result: list[tuple[int, int]] = []
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
                if _normal_text(text[start:end]):
                    result.append((start, end))
                start = end
                index = end
                continue
        index += 1
    if _normal_text(text[start:]):
        result.append((start, len(text)))
    return result


def _ticket_groups(ticket: str) -> list[tuple[bool, list[tuple[int, str]]]]:
    lines = ticket.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    groups: list[tuple[bool, list[tuple[int, str]]]] = []
    current_kind: bool | None = None
    current: list[tuple[int, str]] = []

    def flush() -> None:
        nonlocal current_kind, current
        if current:
            groups.append((bool(current_kind), current))
        current_kind = None
        current = []

    for line_number, line in enumerate(lines):
        if not line.strip():
            flush()
            continue
        bullet = bool(_BULLET.match(line))
        if current and bullet:
            flush()
        if current_kind is None:
            current_kind = bullet
        current.append((line_number, line))
    flush()
    return groups


def _normalized_ticket(ticket: str) -> tuple[str, list[int]]:
    """Normalize line endings and map each normalized character to its source."""
    chars: list[str] = []
    locations: list[int] = []
    index = 0
    while index < len(ticket):
        char = ticket[index]
        if char == "\r":
            chars.append("\n")
            locations.append(index)
            if index + 1 < len(ticket) and ticket[index + 1] == "\n":
                index += 1
        else:
            chars.append(char)
            locations.append(index)
        index += 1
    return "".join(chars), locations


def _group_normalized(lines: Sequence[tuple[int, str]], ticket: str) -> tuple[str, list[int]]:
    """Normalize a line group while mapping normalized chars to source chars."""
    source_lines = ticket.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    line_offsets: list[int] = []
    source_offset = 0
    for source_line in source_lines:
        line_offsets.append(source_offset)
        source_offset += len(source_line) + 1
    chars: list[str] = []
    locations: list[int] = []
    for line_number, raw in lines:
        offset = line_offsets[line_number]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right <= left:
            continue
        if chars:
            chars.append(" ")
            locations.append(offset + left)
        chars.extend(raw[left:right])
        locations.extend(offset + index for index in range(left, right))
    normalized_chars: list[str] = []
    normalized_locations: list[int] = []
    for char, location in zip(chars, locations):
        if char.isspace():
            if not normalized_chars or normalized_chars[-1] == " ":
                continue
            normalized_chars.append(" ")
            normalized_locations.append(location)
        else:
            normalized_chars.append(char)
            normalized_locations.append(location)
    while normalized_chars and normalized_chars[0] == " ":
        normalized_chars.pop(0)
        normalized_locations.pop(0)
    while normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        normalized_locations.pop()
    return "".join(normalized_chars), normalized_locations


def extract_ticket_spans(ticket: str) -> tuple[tuple[int, int, str], ...]:
    """Return source character spans for logical sentences/bullet items."""
    if not isinstance(ticket, str):
        raise TypeError("ticket must be text")
    if os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on":
        from kernel_oracle_grounding import extract_ticket_spans as grounding_spans
        return grounding_spans(ticket)
    result: list[tuple[int, int, str]] = []
    normalized_ticket, source_locations = _normalized_ticket(ticket)
    for bullet, lines in _ticket_groups(normalized_ticket):
        normalized, locations = _group_normalized(lines, normalized_ticket)
        if not normalized or not locations:
            continue
        if bullet:
            start = source_locations[locations[0]]
            end = source_locations[locations[-1]] + 1
            result.append((start, end, normalized))
            continue
        for start, end in _sentence_boundaries(normalized):
            meaningful = normalized[start:end].strip()
            if not meaningful:
                continue
            left = start + len(normalized[start:end]) - len(normalized[start:end].lstrip())
            right = end - len(normalized[start:end]) + len(normalized[start:end].rstrip())
            # ``locations`` is indexed by normalized characters.  The mapping
            # contains the same length for ordinary public prose; clamp for the
            # harmless whitespace collapse at a line join.
            left = min(max(left, 0), len(locations) - 1)
            right = min(max(right, left + 1), len(locations))
            start = source_locations[locations[left]]
            end = source_locations[locations[right - 1]] + 1
            result.append((start, end, meaningful))
    return tuple(result)


def extract_ticket_sentences(ticket: str) -> tuple[str, ...]:
    """Join wrapped prose lines, split sentences, and keep bullet items atomic."""
    return tuple(item[2] for item in extract_ticket_spans(ticket))


def _value(value: object, *names: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _as_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            converted = converter()
        except (AttributeError, TypeError, ValueError):
            converted = None
        if isinstance(converted, Mapping):
            return dict(converted)
    return {
        name: getattr(value, name)
        for name in ("check_id", "obligation_id", "entry_point", "entry", "symbol",
                     "target_symbol", "target_file", "file", "expected", "fixture",
                     "provenance", "clause", "contract_cell", "binding",
                     "target_binding", "current_binding")
        if hasattr(value, name)
    }


def _ticket_text(ledger: Mapping[str, object] | None) -> str:
    if not isinstance(ledger, Mapping):
        return ""
    source_id = str(ledger.get("ticket_source_id", ""))
    for source in ledger.get("sources", ()):
        if isinstance(source, Mapping) and (not source_id or str(source.get("id")) == source_id):
            value = source.get("text")
            if isinstance(value, str):
                return value
    value = ledger.get("ticket", ledger.get("text", ""))
    return value if isinstance(value, str) else ""


def _requirements(ledger: Mapping[str, object] | None) -> tuple[Mapping[str, object], ...]:
    if not isinstance(ledger, Mapping):
        return ()
    return tuple(row for row in ledger.get("requirements", ()) if isinstance(row, Mapping))


def _expected_atoms(value: object, path: str = "") -> list[tuple[object, str, bool]]:
    result: list[tuple[object, str, bool]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if key_text and key_text.lower() not in _COMMON_KEYS:
                result.append((key_text, child_path, True))
            result.extend(_expected_atoms(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            result.extend(_expected_atoms(child, f"{path}[{index}]"))
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        result.append((value, path, False))
    elif isinstance(value, bool):
        result.append((value, path, False))
    return result


def _coerce_literal(value: str) -> object:
    text = value.strip().rstrip(".,;:")
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", text):
        return float(text)
    return text


def _ticket_literals(sentence: str) -> tuple[object, ...]:
    values: list[object] = []
    for match in _BACKTICK.finditer(sentence):
        values.append(_coerce_literal(match.group(1)))
    for match in _QUOTED.finditer(sentence):
        values.append(_coerce_literal(match.group(1) or match.group(2) or ""))
    values.extend(int(match.group(1)) for match in _STATUS.finditer(sentence))
    values.extend(_coerce_literal(match.group(0)) for match in _NUMBER.finditer(sentence))
    for match in _FIELD_MARKER.finditer(sentence):
        values.append(match.group(1))
    unique: list[object] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return tuple(unique)


def _same_literal(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if isinstance(right, (int, float)) and not isinstance(right, bool):
            return left == right
        return str(left) == str(right).strip()
    if isinstance(right, (int, float)) and not isinstance(right, bool):
        return str(right) == str(left).strip()
    return _normal_text(left).casefold() == _normal_text(right).casefold()


def _literal_matches(expected: object, sentence: str,
                    ticket_literals: Sequence[object]) -> list[object]:
    matches: list[object] = []
    del sentence
    for value, _path, _is_key in _expected_atoms(expected):
        matched = any(_same_literal(value, candidate) for candidate in ticket_literals)
        if matched and value not in matches:
            matches.append(copy.deepcopy(value))
    return matches


def _entry_file(symbol: object) -> str:
    text = str(symbol or "").strip()
    module = text.split(":", 1)[0].strip()
    if not module:
        return ""
    if module.endswith(".py"):
        return module.replace("\\", "/")
    return module.replace(".", "/").lstrip("./") + ".py"


def target_file_from_binding(check: object, *, fallback: object = None) -> str:
    """Derive the product file from the bound public symbol or binding record."""
    row = _as_mapping(check)
    for candidate in (
            row.get("binding"), row.get("target_binding"),
            row.get("current_binding"), row.get("provenance")):
        if isinstance(candidate, Mapping):
            for key in ("target_file", "file", "path"):
                value = candidate.get(key)
                if isinstance(value, str) and value.strip():
                    return value.replace("\\", "/")
            for key in ("target_symbol", "entry_point", "symbol", "module"):
                value = candidate.get(key)
                path = _entry_file(value)
                if path:
                    return path
    for key in ("target_symbol", "entry_point", "entry", "symbol"):
        path = _entry_file(row.get(key))
        if path:
            return path
    cell = row.get("contract_cell")
    if isinstance(cell, Mapping):
        surface = str(cell.get("surface", "")).split(".", 1)[0].lower()
        if surface:
            return f"app/{surface}.py"
    value = row.get("target_file", row.get("file", fallback))
    return str(value or "").replace("\\", "/")


def _check_clause(check: object) -> str:
    row = _as_mapping(check)
    for candidate in (row.get("clause"), row.get("exact_clause")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    provenance = row.get("provenance")
    if isinstance(provenance, Mapping):
        span = provenance.get("clause_span")
        if isinstance(span, Mapping) and isinstance(span.get("text"), str):
            return str(span["text"]).strip()
    return ""


def align_check_to_requirement(check: object,
                               ledger: Mapping[str, object] | None) -> dict[str, object] | None:
    """Return the strongest sentence/literal alignment for a host check."""
    if not enabled() or not isinstance(ledger, Mapping):
        return None
    expected = _value(check, "expected", default=None)
    clause = _check_clause(check)
    ticket_text = _ticket_text(ledger)
    ticket_literals_by_sentence = {
        sentence: _ticket_literals(sentence)
        for sentence in extract_ticket_sentences(ticket_text)
    }
    candidates: list[tuple[int, int, Mapping[str, object], str, list[object], tuple[object, ...]]] = []
    for requirement in _requirements(ledger):
        exact = str(requirement.get("exact_clause", "")).strip()
        sentences = [exact] if exact else []
        # A ledger made before this switch may still contain one fragment per
        # wrapped line.  Permit the logical sentence only when that fragment
        # is actually contained in it; never let one requirement borrow an
        # unrelated neighbouring sentence's literals.
        sentences.extend(
            sentence for sentence in ticket_literals_by_sentence
            if sentence not in sentences
            and (not exact or _normal_text(exact) in _normal_text(sentence))
        )
        for sentence in sentences:
            literals = ticket_literals_by_sentence.get(sentence, _ticket_literals(sentence))
            matched = _literal_matches(expected, sentence, literals)
            exact_bonus = 1 if clause and (_normal_text(clause) == _normal_text(exact)
                                          or _normal_text(clause) in _normal_text(exact)) else 0
            if not matched and not exact_bonus:
                continue
            score = len(matched) * 10 + exact_bonus
            candidates.append((score, len(matched), requirement, sentence, matched, literals))
    if not candidates:
        return None
    _score, _count, requirement, sentence, matched, ticket_literals = max(
        candidates, key=lambda item: (item[0], item[1], -len(str(item[2].get("exact_clause", "")))))
    target_check = _as_mapping(check)
    target_file = target_file_from_binding(target_check)
    target_symbol = str(_value(check, "target_symbol", "entry_point", "entry", default="") or "")
    contract = requirement.get("contract")
    binding_ids = contract.get("subject_binding_ids", ()) if isinstance(contract, Mapping) else ()
    binding_rows = ledger.get("bindings", ()) if isinstance(ledger, Mapping) else ()
    if not isinstance(binding_rows, Sequence) or isinstance(binding_rows, (str, bytes)):
        binding_rows = ()
    if not target_file and isinstance(binding_ids, Sequence) and not isinstance(binding_ids, (str, bytes)):
        for binding in binding_rows:
            if not isinstance(binding, Mapping) or str(binding.get("id")) not in {
                    str(item) for item in binding_ids}:
                continue
            binding_check = {"binding": dict(binding), "target_symbol": binding.get("name", "")}
            target_file = target_file_from_binding(binding_check)
            target_symbol = target_symbol or str(binding.get("name", "") or "")
            if target_file:
                break
    evidence = {
        "schema": SCHEMA,
        "method": "ticket_literals" if matched else "exact_ticket_clause",
        "requirement_id": str(requirement.get("id", "")),
        "matched_literals": copy.deepcopy(matched),
        "ticket_literals": copy.deepcopy(list(ticket_literals)),
        "expected_literals": copy.deepcopy([item[0] for item in _expected_atoms(expected)]),
        "sentence": sentence,
        "target_file": target_file,
        "target_symbol": target_symbol,
    }
    aligned_clause = sentence if matched else str(requirement.get("exact_clause", sentence))
    return {
        "requirement": copy.deepcopy(dict(requirement)),
        "clause": aligned_clause,
        "evidence": evidence,
        "target_file": target_file,
        "target_symbol": target_symbol,
    }


def admission_adjustment(check: object, ledger: Mapping[str, object] | None,
                         tier: str, reason: str) -> dict[str, object]:
    """Apply the literal-alignment safety policy to an existing admission."""
    aligned = align_check_to_requirement(check, ledger)
    if aligned is not None:
        if str(tier).upper() == "REJECTED" or str(reason).lower() in {"not_entailed", "wrong_binding"}:
            return {
                "tier": "ADVISORY", "reason": "ticket_literal_alignment",
                "repairable": True, "alignment": aligned,
            }
        return {"tier": tier, "reason": reason, "repairable": True, "alignment": aligned}
    if str(tier).upper() == "REJECTED":
        # Rejection is an admission label, not a claim that an executed check
        # is irrelevant.  The opt-in surface presents an invented expectation
        # as non-repairable advisory evidence instead.
        return {"tier": "ADVISORY", "reason": reason or "not_entailed",
                "repairable": False, "alignment": None}
    return {"tier": tier, "reason": reason, "repairable": False, "alignment": None}


def _flatten(value: object, path: str = "") -> list[tuple[str, object]]:
    if isinstance(value, Mapping):
        result: list[tuple[str, object]] = []
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            result.extend(_flatten(child, child_path))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for index, child in enumerate(value):
            result.extend(_flatten(child, f"{path}[{index}]"))
        return result
    return [(path or "$", copy.deepcopy(value))]


def structural_diff(expected: object, observed: object) -> dict[str, object] | None:
    """Find compact leaf-path mismatches while tolerating value containment."""
    expected_items = _flatten(expected)
    observed_items = _flatten(observed)
    mismatches: list[dict[str, object]] = []
    used: set[int] = set()
    for expected_path, expected_value in expected_items:
        match_index = next((index for index, (path, observed_value) in enumerate(observed_items)
                            if index not in used and _same_literal(expected_value, observed_value)), None)
        if match_index is None:
            mismatches.append({"expected_path": expected_path, "observed_path": _MISSING,
                               "expected": expected_value})
            continue
        used.add(match_index)
        observed_path, observed_value = observed_items[match_index]
        if observed_path != expected_path:
            mismatches.append({"expected_path": expected_path, "observed_path": observed_path,
                               "expected": expected_value, "observed": observed_value})
    if not mismatches:
        return None
    text = "; ".join(
        f"expected {item['expected_path']}; observed {item['observed_path']}"
        for item in mismatches[:3]
    )
    return {"schema": "stateframe-structural-diff/1", "text": text,
            "mismatches": mismatches[:3]}


def enrich_counterexample(row: Mapping[str, object], *, ledger: Mapping[str, object] | None = None,
                          check: Mapping[str, object] | None = None) -> dict[str, object]:
    """Attach alignment, binding target, and structural mismatch facts to a row."""
    result = copy.deepcopy(dict(row))
    check_row = _as_mapping(check) if check is not None else dict(row)
    nested = result.get("check")
    if isinstance(nested, Mapping):
        check_row = {**dict(nested), **check_row}
    if ledger is not None:
        ident = str(result.get("check_id", check_row.get("check_id", "")))
        for candidate in ledger.get("checks", ()):
            if isinstance(candidate, Mapping) and str(candidate.get("id", candidate.get("check_id", ""))) == ident:
                check_row = {**dict(candidate), **check_row}
                break
    if "expected" not in check_row and "expected" in result:
        check_row["expected"] = copy.deepcopy(result["expected"])
    aligned = align_check_to_requirement(check_row, ledger)
    if aligned is not None:
        result["clause"] = aligned["clause"]
        result["alignment_evidence"] = copy.deepcopy(aligned["evidence"])
        result["alignment_requirement_id"] = str(aligned["requirement"].get("id", ""))
        result["target_file"] = aligned["target_file"] or result.get("target_file", "")
        result["target_symbol"] = result.get("target_symbol") or aligned.get("target_symbol", "")
        result["literal_grounded"] = bool(aligned["evidence"].get("matched_literals"))
        result["repairable"] = bool(result["literal_grounded"])
    expected = result.get("expected", check_row.get("expected"))
    observed = result.get("observed", result.get("observed_values"))
    diff = structural_diff(expected, observed)
    if diff is not None:
        result["structural_diff"] = diff
    return result


def is_repairable_literal_counterexample(row: object) -> bool:
    return (isinstance(row, Mapping) and bool(row.get("repairable"))
            and bool(row.get("literal_grounded"))
            and str(row.get("status", row.get("underlying_status", "COUNTEREXAMPLE"))).upper()
            not in {"PASS", "PASSED", "SATISFIED", "RESOLVED"})


def suppress_done_prompt(prompt: str, rows: Sequence[Mapping[str, object]]) -> str:
    """Replace completion nudges while an aligned literal counterexample is open."""
    if not enabled() or not any(is_repairable_literal_counterexample(row) for row in rows):
        return prompt
    ids = [str(row.get("check_id", row.get("obligation_id", "")))
           for row in rows if is_repairable_literal_counterexample(row)]
    suffix = ", ".join(item for item in dict.fromkeys(ids) if item)
    literals = []
    for row in rows:
        if not is_repairable_literal_counterexample(row):
            continue
        evidence = row.get("alignment_evidence")
        if isinstance(evidence, Mapping):
            literals.extend(str(item) for item in evidence.get("matched_literals", ())
                            if str(item).strip())
    unique_literals = list(dict.fromkeys(literals))
    literal_text = next((item for item in unique_literals
                         if not re.fullmatch(r"-?\d+(?:\.\d+)?", item)),
                        unique_literals[0] if unique_literals else "")
    return ("Repairable ticket counterexample remains open; fix or dispute before completion"
            + (f" ({suffix}" if suffix else "")
            + (f": {literal_text}" if literal_text else "")
            + ").")


def alignment_fields(check: object, ledger: Mapping[str, object] | None = None) -> dict[str, object]:
    """Return serializable alignment fields for journal/decision consumers."""
    aligned = align_check_to_requirement(check, ledger)
    if aligned is None:
        return {"alignment_evidence": {}, "repairable": False}
    return {"alignment_evidence": copy.deepcopy(aligned["evidence"]),
            "aligned_clause": aligned["clause"], "target_file": aligned["target_file"],
            "repairable": bool(aligned["evidence"].get("matched_literals"))}


def receipt_alignment_fields(receipt: Mapping[str, object], *,
                             check: object,
                             ledger: Mapping[str, object] | None = None) -> dict[str, object]:
    """Return the producer-owned alignment fields that belong on a check receipt."""
    enriched = enrich_counterexample(receipt, ledger=ledger, check=check)
    return {
        key: copy.deepcopy(enriched[key])
        for key in ("literal_grounded", "alignment_evidence", "alignment_requirement_id")
        if key in enriched
    }


__all__ = [
    "ENV", "SCHEMA", "admission_adjustment", "align_check_to_requirement",
    "alignment_fields", "enabled", "enrich_counterexample", "extract_ticket_sentences",
    "extract_ticket_spans",
    "is_repairable_literal_counterexample", "mode", "structural_diff",
    "receipt_alignment_fields", "suppress_done_prompt", "target_file_from_binding",
]

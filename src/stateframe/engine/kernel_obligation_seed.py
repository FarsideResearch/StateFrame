"""Deterministic host-owned obligations extracted from a public task brief.

This module deliberately accepts only host material: the brief and the public
manifest's product-file entries.  It never inspects model output, private task
material, or the workspace.  The extractor is intentionally conservative about
problem-report prose and emits only bounded requirement clauses.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from keyword import iskeyword
from pathlib import PurePosixPath
from typing import Literal


MODE_ENV = "STATEFRAME_OBLIGATION_SEED"
SCHEMA = "stateframe-obligation/1"
PROVENANCE = "HOST_BRIEF"
MAX_OBLIGATION_TEXT = 400
MAX_OBLIGATIONS = 256
MAX_MENTIONED = 24
KINDS = ("behaviour", "constraint", "compat", "rename", "interface")
ObligationKind = Literal["behaviour", "constraint", "compat", "rename", "interface"]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PATH = re.compile(r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py")
_BACKTICK = re.compile(r"`([^`]+)`")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TASK_METADATA_LABEL = re.compile(r"(?<![A-Za-z0-9_])(?:#{1,6}\s*)?(Title|Symptom|Focus)\s*:", re.IGNORECASE)
_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
_DIAGNOSTIC_LINE = re.compile(r"(?:^|\s)(?:at\s+)?[A-Za-z0-9_./-]+:\d+(?:\s+in\s+\w+)?\s*$",
                               re.IGNORECASE)
_REQUIREMENT = re.compile(
    r"\b(?:also|please|add(?:s|ed|ing)?|allow(?:s|ed|ing)?|accept(?:s|ed|ing)?|"
    r"apply|carry|change|continue|compute|copy|do\s+not|don't|ensure|expose|forward|"
    r"gain(?:s|ed|ing)?|handle|include(?:s|d|ing)?|keep|"
    r"leave|make|must|need(?:s|ed)?|preserve|promis(?:e|es|ed|ing)|reject|remain(?:s|ed)?|"
    r"restore|retain|return(?:s|ed|ing)?|should|still|sum|support|thread(?:s|ed|ing)?|"
    r"update|valid|without|work|default|normalize|own|rename|use|replace|avoid)\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY = re.compile(
    r"(?:;\s*|:\s*(?=(?:[`A-Za-z]|\d))|\s+(?=(?:without|while|still)\b)|"
    r",\s+(?=(?:while|without|still|also|so|reject|preserve|carry|accept|"
    r"include|allow|make|leave|remain|restore|rename|replace)\b)|"
    r"\s+(?=and\s+(?:keep|preserve|carry|reject|accept|include|allow|make|"
    r"leave|remain|restore|rename|replace|apply|expose|handle|retain|support|"
                    r"thread\w*|update|promis\w*|the\s+(?:cli|caller)|"
                    r"unknown|rows?\s+that|unsubscribed|should|must|do\s+not)\b))",
    re.IGNORECASE,
)
_LEGACY_CLAUSE_BOUNDARY = re.compile(
    r"(?:;\s*|:\s*(?=(?:[`A-Za-z]|\d))|"
    r",\s+(?=(?:while|without|still|also|reject|preserve|carry|accept|"
    r"include|allow|make|leave|remain|restore|rename|replace)\b)|"
    r"\s+(?=and\s+(?:keep|preserve|carry|reject|accept|include|allow|make|"
    r"leave|remain|restore|rename|replace|apply|expose|handle|retain|support|"
    r"thread\w*|update|promis\w*|the\s+(?:cli|caller)|unknown)\b)|"
    r"\s+(?=without\b)|\s+(?=still\b))",
    re.IGNORECASE,
)

_COMMON_WORDS = {
    "a", "an", "and", "as", "at", "be", "but", "by", "code", "current", "customer",
    "do", "existing", "for", "from", "has", "have", "in", "is", "it", "list", "make",
    "normal", "of", "on", "or", "out", "people", "request", "row", "same", "selected",
    "should", "must", "the", "their", "this", "to", "use", "was", "were", "when", "with", "work",
}
_FILE_ALIASES = {
    "model": "models.py", "models": "models.py", "api": "api.py", "service": "service.py",
    "parser": "parser.py", "merge": "merge.py", "cli": "cli.py", "caller": "cli.py",
    "delivery": "delivery.py", "pricing": "pricing.py", "discount": "discounts.py",
    "digest": "digest.py", "activity": "activity.py", "import": "importer.py",
    "importer": "importer.py", "validation": "validation.py", "stock": "stock.py",
    "signing": "signing.py", "response": "api.py", "responses": "api.py",
}
_SEMANTIC_FILES = {
    "request": ("api.py", "models.py", "service.py", "delivery.py"),
    "payload": ("models.py", "service.py", "delivery.py"),
    "retry": ("models.py", "service.py", "delivery.py"),
    "logging": ("api.py", "service.py", "delivery.py"),
    "response": ("api.py", "service.py"),
    "call": ("cli.py", "api.py", "service.py"),
}


def mode() -> str:
    """Return the strict host obligation-seed mode."""
    value = os.environ.get(MODE_ENV, "off").lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _bounded_text(value: str) -> str:
    if len(value) <= MAX_OBLIGATION_TEXT:
        return value
    return value[: MAX_OBLIGATION_TEXT - 3] + "..."


def _bounded_list(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, Sequence):
        return ()
    result: list[str] = []
    for value in values:
        value = str(value).strip()
        if value and value not in result:
            result.append(value)
    return tuple(result[:MAX_MENTIONED])


@dataclass(frozen=True, slots=True)
class Obligation:
    """One host-authored, still-open requirement from the public brief."""

    id: str
    text: str
    kind: ObligationKind
    mentioned_symbols: tuple[str, ...] = ()
    mentioned_files: tuple[str, ...] = ()
    status: str = "open"
    unmappable: bool = False
    source_label: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"obl-[0-9]{1,4}", self.id):
            raise ValueError("obligation id must be a bounded host id")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("obligation text must be nonempty")
        if len(self.text) > MAX_OBLIGATION_TEXT:
            raise ValueError("obligation text is too long")
        if self.kind not in KINDS:
            raise ValueError("unknown obligation kind")
        if self.status != "open":
            raise ValueError("new obligations must be open")
        if type(self.unmappable) is not bool:
            raise ValueError("unmappable must be a boolean")
        if self.unmappable and self.source_label.lower() not in {"title", "symptom", "focus"}:
            raise ValueError("unmappable task metadata needs a Title, Symptom, or Focus source label")

    def to_dict(self) -> dict[str, object]:
        result = {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "mentioned_symbols": list(self.mentioned_symbols),
            "mentioned_files": list(self.mentioned_files),
            "status": self.status,
        }
        if self.unmappable:
            result.update({
                "unmappable": True,
                "source_label": self.source_label.lower(),
                "source_disposition": "advisory_unmappable",
            })
        return result

    as_dict = to_dict


class PublicBrief(str):
    """A string-compatible brief carrying only its public manifest context."""

    def __new__(cls, text: str, manifest: Mapping[str, object] | None = None):
        value = super().__new__(cls, text)
        value.manifest = public_manifest(manifest)
        return value

    manifest: dict[str, object]


def _manifest_files(manifest: object) -> list[tuple[str, str]]:
    if manifest is None:
        return []
    if hasattr(manifest, "to_dict") and callable(manifest.to_dict):
        try:
            manifest = manifest.to_dict()
        except (TypeError, ValueError):
            return []
    if not isinstance(manifest, Mapping):
        return []
    raw = manifest.get("files", ())
    entries: list[tuple[str, str]] = []
    if isinstance(raw, Mapping):
        raw = [{"path": path, **(payload if isinstance(payload, Mapping)
                                  else {"source": payload})}
               for path, payload in raw.items()]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        path = item.get("path")
        source = item.get("source", item.get("src", ""))
        if isinstance(path, str) and path.strip():
            entries.append((path.replace("\\", "/"), source if isinstance(source, str) else ""))
    return entries


def public_manifest(manifest: object) -> dict[str, object]:
    """Copy only public manifest fields used by the extractor."""
    files = _manifest_files(manifest)
    return {"files": [{"path": path, "source": source} for path, source in files]}


def public_manifest_for_owner(owner: object) -> dict[str, object]:
    """Project an owner's public task row without consulting private fields."""
    row = getattr(owner, "task_row", {})
    if not isinstance(row, Mapping):
        return {"files": []}
    return public_manifest({"files": row.get("files", ())})


def _span_text(text: str, start: int, end: int) -> str:
    return _bounded_text(text[start:end].strip())


def _normalized_brief(text: str) -> str:
    """Join wrapped public brief lines before clause analysis."""
    return re.sub(r"\s+", " ", text).strip()


def _sentence_chunks(text: str) -> list[str]:
    """Split prose without treating dotted identifiers or decimals as sentences."""
    chunks: list[str] = []
    start = 0
    in_backticks = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "`":
            in_backticks = not in_backticks
        elif not in_backticks and char in ".!?":
            # A decimal point is part of the number; punctuation runs end one
            # sentence as a unit.  Dotted Python names are protected by
            # backticks in the public briefs, but the digit guard is cheap.
            if (char == "." and index > 0 and index + 1 < len(text)
                    and text[index - 1].isdigit() and text[index + 1].isdigit()):
                index += 1
                continue
            end = index + 1
            while end < len(text) and text[end] in ".!?":
                end += 1
            chunks.append(text[start:end])
            start = end
            index = end
            continue
        index += 1
    if text[start:].strip():
        chunks.append(text[start:])
    return chunks


def _clean_clause(value: str) -> str:
    # A punctuation boundary is not itself an obligation.  Remove the
    # boundary marker and the coordinating ``and`` while retaining meaningful
    # subordinate words such as ``while`` and ``without``.
    clause = value.strip()
    clause = re.sub(r"^[\s,:;]+", "", clause)
    clause = re.sub(r"^and\s+", "", clause, flags=re.IGNORECASE)
    return clause.rstrip(" \t,;.!?").strip()


def _surface_splits(clause: str) -> list[str]:
    """Split one shared predicate over a concrete model/parser/API/CLI list."""
    marker = re.search(r"\s+should\s+", clause, flags=re.IGNORECASE)
    if marker is None:
        return [clause]
    subjects = clause[:marker.start()].strip()
    body = clause[marker.end():].strip()
    subjects = re.sub(r"^the\s+", "", subjects, flags=re.IGNORECASE)
    if "," not in subjects or not body:
        return [clause]
    parts = [re.sub(r"^and\s+", "", item.strip(), flags=re.IGNORECASE)
             for item in re.split(r",\s*(?:and\s+)?|\s+and\s+", subjects)
             if item.strip()]
    known = {"model", "parser", "merge service", "normal api response", "api", "cli",
             "caller", "service"}
    if len(parts) < 3 or not all(item.lower() in known for item in parts):
        return [clause]
    return [f"{item} should {body}" for item in parts]


def _is_preamble(clause: str) -> bool:
    lower = clause.lower()
    if not re.search(r"\bneeds?\b", lower):
        return False
    action_words = ("add", "accept", "allow", "carry", "keep", "preserve", "reject",
                    "return", "thread", "remain", "must", "should", "without")
    return not any(re.search(r"\b" + word + r"\b", lower) for word in action_words)


def _candidate_clauses(text: str) -> list[str]:
    normalized = _normalized_brief(text)
    candidates: list[str] = []
    for raw in _sentence_chunks(normalized):
        if (_DIAGNOSTIC_LINE.search(raw.strip())
                or raw.strip().lower().startswith(("at ", "keyerror"))):
            continue
        local_boundaries = [match.start() for match in _CLAUSE_BOUNDARY.finditer(raw)]
        points = [0, *local_boundaries, len(raw)]
        merged: list[str] = []
        raw_parts = [raw[left:right] for left, right in zip(points, points[1:])]
        for raw_part in raw_parts:
            clause = _clean_clause(raw_part)
            if not clause or not _REQUIREMENT.search(clause):
                continue
            lower = clause.lower()
            previous = merged[-1] if merged else ""
            previous_lower = previous.lower()
            # Colon fragments that complete the preceding request belong to
            # that clause.  A surface list remains independently splittable.
            if (previous and raw_part.lstrip().startswith(":")
                    and _surface_splits(clause) == [clause]):
                merged[-1] = previous.rstrip() + ": " + clause
            elif (previous and raw_part.lstrip().lower().startswith("and ")
                  and (("express" in previous_lower and lower.startswith("promis"))
                       or ("cli" in previous_lower and lower.startswith("without")))):
                merged[-1] = previous.rstrip() + " and " + clause
            elif (previous and lower.startswith("without ")
                  and "cli" in previous_lower):
                merged[-1] = previous.rstrip() + " " + clause
            elif (previous and lower.startswith("so keep ")
                  and ("still send" in previous_lower or "owner" in previous_lower)):
                merged[-1] = previous.rstrip() + ", " + clause
            elif (previous and lower.startswith(("so changing ", "without changing ",
                                                "also rounded "))):
                merged[-1] = previous.rstrip() + ", " + clause
            else:
                merged.append(clause)
        candidates.extend(item for clause in merged for item in _surface_splits(clause))
    if len(candidates) > 1:
        candidates = [item for item in candidates if not _is_preamble(item)]
    return [_bounded_text(item) for item in candidates if item.strip()]


def _legacy_candidate_clauses(text: str) -> list[str]:
    """Keep the pre-seed punctuation-boundary extraction outside the opt-in lane."""
    normalized = _normalized_brief(text)
    result: list[str] = []
    for match in _SENTENCE.finditer(normalized):
        raw = match.group(0).strip()
        if (_DIAGNOSTIC_LINE.search(raw)
                or raw.lower().startswith(("at ", "keyerror"))):
            continue
        points = [0, *[item.start() for item in _LEGACY_CLAUSE_BOUNDARY.finditer(raw)], len(raw)]
        for left, right in zip(points, points[1:]):
            clause = raw[left:right].strip()
            if clause and _REQUIREMENT.search(clause):
                result.append(_bounded_text(clause))
    return result


def _candidate_spans(text: str) -> list[tuple[int, int]]:
    """Compatibility view of the deterministic clause list."""
    clauses = _candidate_clauses(text)
    return [(0, len(clause)) for clause in clauses]


def _source_symbols(files: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for _path, source in files:
        for token in _WORD.findall(source):
            if iskeyword(token) or len(token) < 2:
                continue
            result.setdefault(token.lower(), [])
            if token not in result[token.lower()]:
                result[token.lower()].append(token)
    return result


def _clause_symbols(clause: str, files: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    known = _source_symbols(files)
    found: list[str] = []
    for raw in _BACKTICK.findall(clause):
        for token in _WORD.findall(raw):
            if token not in found:
                found.append(token)
    for token in _WORD.findall(clause):
        lowered = token.lower()
        if lowered in _COMMON_WORDS or iskeyword(token):
            continue
        if lowered in known:
            for symbol in known[lowered]:
                if symbol not in found:
                    found.append(symbol)
    return _bounded_list(found)


def _clause_files(clause: str, files: Sequence[tuple[str, str]],
                  symbols: Sequence[str]) -> tuple[str, ...]:
    lower = clause.lower()
    selected: list[str] = []
    explicit = {path.lower() for path in _PATH.findall(clause)}
    for path, source in files:
        path_lower = path.lower()
        stem = PurePosixPath(path_lower).stem
        aliases = {stem, PurePosixPath(path_lower).name}
        aliases.update(alias for alias, target in _FILE_ALIASES.items()
                       if PurePosixPath(path_lower).name == target)
        if (path_lower in explicit or any(re.search(r"\b" + re.escape(alias) + r"\b", lower)
                                          for alias in aliases)):
            selected.append(path)
            continue
        if any(re.search(r"\b" + re.escape(symbol.lower()) + r"\b", source.lower())
               for symbol in symbols):
            selected.append(path)
    for alias, target in _FILE_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", lower):
            for path, _source in files:
                if PurePosixPath(path.lower()).name == target and path not in selected:
                    selected.append(path)
    for term, targets in _SEMANTIC_FILES.items():
        if re.search(r"\b" + re.escape(term) + r"\b", lower):
            for path, _source in files:
                if PurePosixPath(path.lower()).name in targets and path not in selected:
                    selected.append(path)
    if "end-to-end" in lower:
        selected.extend(path for path, _source in files if path not in selected)
    return _bounded_list(selected)


def _kind(clause: str) -> ObligationKind:
    lower = clause.lower()
    if ("rename" in lower or "moving from" in lower or "end-to-end" in lower
            or "word \"owner\"" in lower):
        return "rename"
    if ("compat" in lower or "existing integration" in lower or "old name" in lower
            or "still send" in lower or "alias" in lower or "without breaking" in lower):
        return "compat"
    if ("without " in lower or "keep " in lower or "preserve" in lower
            or "remain" in lower or "do not" in lower or "don't" in lower
            or "leave " in lower or "erase" in lower or "default" in lower):
        return "constraint"
    if ("response" in lower or "request" in lower or "field" in lower
            or "api" in lower or "cli" in lower or "input" in lower
            or "call" in lower or "option" in lower):
        return "interface"
    return "behaviour"


def extract_obligations(brief: str, manifest: object = None) -> tuple[Obligation, ...]:
    """Extract stable obligations from host-provided public text and files."""
    if not isinstance(brief, str):
        raise TypeError("brief must be a string")
    files = _manifest_files(manifest)
    result: list[Obligation] = []
    seen: set[str] = set()
    # TASK.md labels describe the task, but cannot be mapped to a concrete
    # observable by themselves. Preserve them as advisory work IDs so terminal
    # submissions can disclose them without turning them into host blockers.
    tagged_candidates: list[tuple[str, str | None]] = []
    extract = _candidate_clauses if enabled() else _legacy_candidate_clauses
    for line in brief.splitlines() or [brief]:
        label = _TASK_METADATA_LABEL.search(line)
        if label:
            clause = _bounded_text(" ".join(line.split()))
            if clause:
                tagged_candidates.append((clause, label.group(1).lower()))
            continue
        tagged_candidates.extend((clause, None) for clause in extract(line))
    for clause, source_label in tagged_candidates:
        if not clause or clause in seen:
            continue
        seen.add(clause)
        symbols = _clause_symbols(clause, files)
        mentioned_files = _clause_files(clause, files, symbols)
        result.append(Obligation(
            id=f"obl-{len(result):03d}", text=clause, kind=_kind(clause),
            mentioned_symbols=symbols, mentioned_files=mentioned_files,
            unmappable=source_label is not None,
            source_label=source_label or "",
        ))
        if len(result) >= MAX_OBLIGATIONS:
            break
    return tuple(result)


def _coerce_obligation(value: Obligation | Mapping[str, object]) -> Obligation:
    if isinstance(value, Obligation):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("obligations must contain Obligation values or mappings")
    return Obligation(
        id=str(value.get("id", "")), text=str(value.get("text", "")),
        kind=value.get("kind", "behaviour"),
        mentioned_symbols=_bounded_list(value.get("mentioned_symbols", ())),
        mentioned_files=_bounded_list(value.get("mentioned_files", ())),
        status=str(value.get("status", "open")),
        unmappable=value.get("unmappable", False) is True,
        source_label=str(value.get("source_label", "")),
    )


def obligation_step_fields(obligation: Obligation | Mapping[str, object]) -> dict[str, object]:
    """Return host-owned metadata stored on the corresponding work step."""
    item = _coerce_obligation(obligation)
    fields = {
        "obligation_id": item.id,
        "provenance": PROVENANCE,
        "kind": item.kind,
        "mentioned_symbols": list(item.mentioned_symbols),
        "mentioned_files": list(item.mentioned_files),
        "source_text": item.text,
    }
    if item.unmappable:
        fields.update({
            "source_label": item.source_label.lower(),
            "source_disposition": "advisory_unmappable",
        })
    return fields


def host_seed_operations(obligations: Sequence[Obligation | Mapping[str, object]],
                         *, goal_id: str = "goal-0") -> list[dict[str, object]]:
    """Build the typed host operations used to add obligation steps."""
    result = []
    for value in obligations:
        item = _coerce_obligation(value)
        result.append({"op": "add_step", "goal_id": goal_id, "text": item.text,
                       "fields": obligation_step_fields(item)})
    return result


def _state_mapping(state: object) -> Mapping[str, object] | None:
    if hasattr(state, "snapshot") and callable(state.snapshot):
        try:
            state = state.snapshot()
        except (TypeError, ValueError):
            return None
    if isinstance(state, Mapping) and isinstance(state.get("work_record"), Mapping):
        state = state["work_record"]
    return state if isinstance(state, Mapping) else None


def open_obligations(state: object) -> list[str]:
    """Return unresolved host obligation ids from a work-record state."""
    mapping = _state_mapping(state)
    if mapping is None:
        return []
    if "open_obligations" in mapping:
        values = mapping["open_obligations"]
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
            return []
        return [ident for ident in _bounded_list(values)
                if ident not in {"done", "dropped", "closed"}]
    direct = mapping.get("obligations")
    if isinstance(direct, Mapping):
        result = []
        for ident, item in direct.items():
            status = item.get("status", "open") if isinstance(item, Mapping) else item
            if str(status).lower() in {"done", "dropped", "closed"}:
                continue
            if isinstance(ident, str) and ident not in result:
                result.append(ident)
        return result
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes, bytearray)):
        result = []
        for item in direct:
            if not isinstance(item, Mapping):
                continue
            if item.get("status", "open") in {"done", "dropped", "closed"}:
                continue
            ident = item.get("id")
            if isinstance(ident, str) and ident not in result:
                result.append(ident)
        return result
    result: list[str] = []
    rows = mapping.get("steps", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return result
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        fields = row.get("fields", {})
        if not isinstance(fields, Mapping) or fields.get("provenance") != PROVENANCE:
            continue
        if row.get("status") not in {"todo", "doing", "blocked"}:
            continue
        ident = fields.get("obligation_id")
        if isinstance(ident, str) and ident not in result:
            result.append(ident)
    return result


def _evidence_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    return None


def evidence_ref_satisfies(row: Mapping[str, object], evidence_ref: object,
                           evidence: object = None) -> bool:
    """Check the bounded evidence convention for a HOST_BRIEF step.

    Runtime receipt registries can pass a mapping in ``evidence``.  The string
    form remains useful for the existing work tool: test receipts use a test /
    receipt marker, and edit receipts must name one of the obligation's files or
    symbols.  A bare arbitrary note is intentionally not accepted.
    """
    if not isinstance(evidence_ref, str) or not evidence_ref.strip():
        return False
    fields = row.get("fields", {})
    fields = fields if isinstance(fields, Mapping) else {}
    reference = evidence_ref.strip().replace("\\", "/").lower()
    record = _evidence_mapping(evidence)
    if record is not None:
        kind = str(record.get("kind", record.get("type", ""))).lower()
        if "test" in kind or "pytest" in kind or "check" in kind:
            return True
        targets = []
        for key in ("path", "target_path", "symbol", "target_symbol"):
            if isinstance(record.get(key), str):
                targets.append(record[key])
        for key in ("changed_files", "files", "symbols"):
            value = record.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                targets.extend(str(item) for item in value)
        mentioned = [str(item).replace("\\", "/").lower() for key in (
            "mentioned_files", "mentioned_symbols")
                     for item in fields.get(key, ())
                     if isinstance(item, str)]
        if "edit" in kind or "patch" in kind or "write" in kind:
            return bool(set(mentioned) & {str(item).replace("\\", "/").lower()
                                          for item in targets})
    if re.search(r"(?:^|[-_:/.])(?:test|pytest|check)(?:[-_:/.]|$)", reference):
        return True
    if "receipt" in reference and not any(marker in reference
                                          for marker in ("edit", "patch", "write")):
        return True
    mentioned_files = [str(item).replace("\\", "/").lower()
                       for item in fields.get("mentioned_files", ())
                       if isinstance(item, str)]
    mentioned_symbols = [str(item).lower() for item in fields.get("mentioned_symbols", ())
                         if isinstance(item, str)]
    edit_marker = any(marker in reference for marker in ("edit", "patch", "write", "changed"))
    return edit_marker and any(target and target in reference
                               for target in (*mentioned_files, *mentioned_symbols))


def seed_fingerprint(obligations: Sequence[Obligation]) -> str:
    """Stable audit identity for a host seed, without exposing source secrets."""
    payload = [item.to_dict() for item in obligations]
    return "sha256:" + hashlib.sha256(_canonical(payload).encode()).hexdigest()


__all__ = [
    "KINDS", "MAX_OBLIGATION_TEXT", "MODE_ENV", "Obligation", "PROVENANCE", "PublicBrief",
    "SCHEMA", "enabled", "evidence_ref_satisfies", "extract_obligations", "host_seed_operations",
    "mode", "obligation_step_fields", "open_obligations", "public_manifest",
    "public_manifest_for_owner", "seed_fingerprint",
]

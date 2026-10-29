"""Request-bound ticket and source-closure contracts.

This module is a projection seam, not a second repository index.  It binds
the authorized ticket bytes, plans bounded semantic source spans, and asks the
existing ``mc-outline/1`` and EC01 contracts to describe what is current.  It
never executes product code and never turns source discovery into behavioral
proof.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from functools import lru_cache
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable

from kernel_evidence_contracts import (
    ApplicabilityState,
    Availability,
    EvidenceCoverage,
    EvidenceDependency,
    EvidenceSelector,
    EvidenceView,
    EvidenceVisibility,
    SelectorMode,
    SourceSnapshot,
    CoverageStatus,
)


SCHEMA = "stateframe-request-evidence-closure/1"
TICKET_SCHEMA = "stateframe-authorized-ticket/1"
TICKET_HEADER = "[AUTHORIZED TICKET"
CLOSURE_HEADER = "[SOURCE CLOSURE RECEIPT"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_QUOTED = re.compile(r"(['\"`])([^'\"`\n]{3,96})\1")
_SLASH_DIGIT = re.compile(
    r"(?<![\w/.:-])([A-Za-z][A-Za-z0-9.]*(?:[/-][A-Za-z0-9.]+)+|"
    r"[A-Za-z][A-Za-z_]*\d[A-Za-z0-9_]*)(?![\w/-])"
)
_CODE_IDENT = re.compile(
    r"(?<![\w.])((?:[A-Za-z_$][A-Za-z0-9_$]*\.)*"
    r"[A-Za-z_$][A-Za-z0-9_$]*)(?![\w.])"
)
_IMPORT_FROM = re.compile(r"\bfrom\s+['\"]([^'\"]+)['\"]")
_REQUIRE = re.compile(r"\b(?:require|import)\s*\(\s*['\"]([^'\"]+)['\"]")
_DENIED_PARTS = frozenset({".git", "backups", "evaluator", "grader", "graders", "gold", "hidden"})
_SOURCE_EXTENSIONS = (".py", ".go", ".rs", ".ts", ".tsx", ".js", ".jsx", ".java", ".php", ".vue")
_SOURCE_PATH_MENTION = re.compile(
    r"(?<![A-Za-z0-9_.-])((?:\./)?(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\."
    + "(?:" + "|".join(re.escape(ext[1:]) for ext in _SOURCE_EXTENSIONS) + ")"
    + r")(?![A-Za-z0-9_-]|\.[A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_DISCLAIMED_SOURCE_PATH = re.compile(
    r"^[\t `\"'():,]*(?:is\s+)?(?:(?:an?|the)\s+)?"
    r"(?:unrelated|read[- ]only|context[- ]only|not\s+(?:a\s+)?product)\b",
    re.IGNORECASE,
)


def _code_identifier(value: str) -> bool:
    """Keep qualified names and code-shaped terminals, not ordinary prose."""

    terminal = value.rsplit(".", 1)[-1]
    return "." in value or "_" in terminal or bool(re.search(r"[a-z][A-Z]", terminal))


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def source_revision(source: str | bytes) -> str:
    """Return the exact-byte revision used by request/evidence identities."""

    raw = source if isinstance(source, bytes) else source.encode("utf-8")
    return _sha256(raw)


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{label} contains NUL")
    return value


@dataclass(frozen=True)
class TicketIdentity:
    """Immutable public ticket text; task identity is deliberately separate."""

    task_id: str
    text: str
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _nonempty(self.task_id, "task_id")
        _nonempty(self.text, "ticket text")
        object.__setattr__(self, "sha256", _sha256(self.text.encode("utf-8")))

    @property
    def ticket_id(self) -> str:
        return "ticket:" + self.sha256.removeprefix("sha256:")

    @property
    def ticket_hash(self) -> str:
        return self.sha256

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": TICKET_SCHEMA,
            "task_id": self.task_id,
            "ticket_id": self.ticket_id,
            "ticket_sha256": self.sha256,
            "text": self.text,
        }


class EvidenceStatus(str, Enum):
    """Source-resolution states kept distinct from behavioral outcomes."""

    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    ABSENT = "ABSENT"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"
    PARTIAL = "PARTIAL"
    UNRUNNABLE = "UNRUNNABLE"
    UNKNOWN = "UNKNOWN"


class EvidenceOutcome(str, Enum):
    """Behavioral outcomes are not inferred from lexical/source coverage."""

    UNCOVERED = "UNCOVERED"
    ENVIRONMENT = "ENVIRONMENT"
    RESPONSE_BACKED = "RESPONSE_BACKED"
    UNRUNNABLE = "UNRUNNABLE"


def _row_ticket_text(row: Mapping[str, object]) -> str:
    # ``msg`` is the live_adapter's already-assembled symptom + Focus ticket.
    # Do not reconstruct it from symptom/focus here: doing so would create a
    # second producer and could change whitespace or newline bytes.
    for key in ("_authorized_ticket", "ticket", "msg", "_bridge_ordinary_brief",
                "public_brief", "brief", "symptom"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def ticket_identity_from_row(row: Mapping[str, object]) -> TicketIdentity | None:
    if not isinstance(row, Mapping):
        return None
    text = _row_ticket_text(row)
    if not text:
        return None
    task_id = row.get("task_id") or row.get("sha") or "task:anonymous"
    return TicketIdentity(str(task_id), text)


def ticket_identity_for_runtime(runtime: object) -> TicketIdentity | None:
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", {})
    return ticket_identity_from_row(row) if isinstance(row, Mapping) else None


def ticket_identity_for_row(row: Mapping[str, object]) -> TicketIdentity | None:
    """Compatibility name used by host fact and IDEALPATH owners."""

    return ticket_identity_from_row(row)


def ticket_digest(text: str) -> str:
    """Digest exact UTF-8 ticket bytes without trimming or reconstruction."""

    if not isinstance(text, str) or not text:
        raise ValueError("ticket text must be a non-empty string")
    return _sha256(text.encode("utf-8"))


def bind_ticket_identity(runtime: object) -> TicketIdentity | None:
    """Attach the immutable ticket identity to a live runtime for this request."""

    identity = ticket_identity_for_runtime(runtime)
    if identity is not None:
        setattr(runtime, "_ticket_identity", identity)
    return identity


def _contains_exact(value: object, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, Mapping):
        return any(_contains_exact(item, needle) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_exact(item, needle) for item in value)
    return False


def retrieval_ticket_receipt(identity: TicketIdentity, payload: object) -> dict[str, object]:
    """Record exact-ticket admission at the retrieval/source boundary."""

    matched = _contains_exact(payload, identity.text)
    return {
        "schema": TICKET_SCHEMA,
        "boundary": "retrieval",
        "task_id": identity.task_id,
        "ticket_sha256": identity.sha256,
        "delivered_sha256": identity.sha256 if matched else None,
        "exact_match": matched,
        "status": "DELIVERED" if matched else "NOT_DELIVERED",
    }


def model_brief_receipt(identity: TicketIdentity, messages: object) -> dict[str, object]:
    """Record exact-ticket admission at the model-brief boundary."""

    matched = _contains_exact(messages, identity.text)
    return {
        "schema": TICKET_SCHEMA,
        "boundary": "model_brief",
        "task_id": identity.task_id,
        "ticket_sha256": identity.sha256,
        "delivered_sha256": identity.sha256 if matched else None,
        "retrieval_ticket_sha256": identity.sha256 if matched else None,
        "model_brief_ticket_sha256": identity.sha256 if matched else None,
        "ticket_bytes": len(identity.text.encode("utf-8")),
        "exact_match": matched,
        "status": "DELIVERED" if matched else "NOT_DELIVERED",
    }


def ensure_ticket_messages(
        identity: TicketIdentity,
        messages: Sequence[Mapping[str, object]],
        uncapped: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Carry live_adapter's exact ticket bytes to both request projections."""

    block = {
        "role": "user",
        "content": render_ticket_brief(identity),
        "_request_budget_section": "authorized_ticket",
    }

    def carry(source: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        result = [dict(item) for item in source if isinstance(item, Mapping)]
        if not _contains_exact(result, identity.text):
            result.append(dict(block))
        return result

    return carry(messages), carry(uncapped)


def render_ticket_brief(identity: TicketIdentity) -> str:
    """Render the exact ticket text plus a non-authoritative identity receipt."""

    return (f"{TICKET_HEADER} | schema={TICKET_SCHEMA}; task_id={identity.task_id}; "
            f"ticket_sha256={identity.sha256}]\n{identity.text}\n[END AUTHORIZED TICKET]")


def serialized_request_bytes(messages: Sequence[Mapping[str, object]], tools: object = ()) -> bytes:
    return json.dumps({"messages": list(messages), "tools": tools}, ensure_ascii=False,
                      sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def final_wire_ticket_receipt(
        identity: TicketIdentity,
        messages: Sequence[Mapping[str, object]],
        tools: object = (),
        *,
        retrieval_ticket_sha256: str | None = None,
        model_brief_ticket_sha256: str | None = None,
        wire_bytes: bytes | None = None,
) -> dict[str, object]:
    """Prove that the immutable ticket value is present in the final request."""

    contents = [str(item.get("content", "")) for item in messages if isinstance(item, Mapping)]
    occurrences = sum(content.count(identity.text) for content in contents)
    wire = wire_bytes if isinstance(wire_bytes, bytes) else serialized_request_bytes(messages, tools)
    retrieval_match = retrieval_ticket_sha256 == identity.sha256
    model_match = model_brief_ticket_sha256 == identity.sha256
    wire_match = False
    try:
        decoded = json.loads(wire.decode("utf-8"))
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
        decoded = None
    if decoded is not None:
        wire_match = _contains_exact(decoded, identity.text)
    delivered = occurrences > 0 and wire_match and retrieval_match and model_match
    return {
        "schema": "stateframe-final-wire-ticket-receipt/1",
        "task_id": identity.task_id,
        "ticket_id": identity.ticket_id,
        "ticket_sha256": identity.sha256,
        "ticket_bytes_sha256": _sha256(identity.text.encode("utf-8")),
        "retrieval_ticket_sha256": retrieval_ticket_sha256,
        "model_brief_ticket_sha256": model_brief_ticket_sha256,
        "wire_sha256": _sha256(wire),
        "ticket_content_occurrences": occurrences,
        "wire_contains_ticket": wire_match,
        "retrieval_matches": retrieval_match,
        "model_brief_matches": model_match,
        "delivered": delivered,
        "status": "DELIVERED" if delivered else "OMITTED",
    }


def final_wire_receipt(identity: TicketIdentity, body: object, wire: bytes) -> dict[str, object]:
    """Legacy-shaped final-wire receipt backed by the canonical ticket contract."""

    messages = body.get("messages", ()) if isinstance(body, Mapping) else ()
    tools = body.get("tools", ()) if isinstance(body, Mapping) else ()
    receipt = final_wire_ticket_receipt(
        identity,
        messages if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes, bytearray)) else (),
        tools,
        retrieval_ticket_sha256=identity.sha256,
        model_brief_ticket_sha256=identity.sha256,
        wire_bytes=wire,
    )
    return {
        **receipt,
        "exact_match": receipt["delivered"],
        "delivered_sha256": identity.sha256 if receipt["delivered"] else None,
    }


@dataclass(frozen=True)
class LanguageAdapter:
    """Initialization/construction semantics, not execution evidence."""

    language: str
    parser: str
    module_initialization: str
    construction_ownership: str
    supported: bool = True

    @property
    def evidence_status(self) -> str:
        return "UNCOVERED" if self.supported else "UNRUNNABLE"


_ADAPTERS = {
    ".py": LanguageAdapter("python", "ast+mc-outline/1", "imports and module statements execute during import", "module or callable owns construction"),
    ".go": LanguageAdapter("go", "mc-outline/1", "package initialization precedes explicit entrypoint", "package var/init or callable owns construction"),
    ".rs": LanguageAdapter("rust", "mc-outline/1", "explicit or lazy initialization; no import execution claim", "static/lazy initializer or callable owns construction"),
    ".ts": LanguageAdapter("typescript", "mc-outline/1", "module evaluation precedes exported entrypoint", "module statement or callable owns construction"),
    ".tsx": LanguageAdapter("typescript", "mc-outline/1", "module evaluation precedes exported entrypoint", "module statement or callable owns construction"),
    ".js": LanguageAdapter("javascript", "mc-outline/1", "module evaluation precedes exported entrypoint", "module statement or callable owns construction"),
    ".jsx": LanguageAdapter("javascript", "mc-outline/1", "module evaluation precedes exported entrypoint", "module statement or callable owns construction"),
    ".java": LanguageAdapter("java", "mc-outline/1", "class loading/static initialization precedes instance entry", "static initializer or callable owns construction"),
    ".php": LanguageAdapter("php", "mc-outline/1", "include/bootstrap order is explicit; no import execution claim", "file-scope statement or callable owns construction"),
    ".vue": LanguageAdapter("vue", "mc-outline/1+SFC blocks", "script/module context is evaluated per SFC runtime", "script setup/options block owns construction"),
}


def language_adapter_for(path: str | Path) -> LanguageAdapter:
    return _ADAPTERS.get(Path(str(path)).suffix.lower(), LanguageAdapter(
        "unknown", "unavailable", "runtime semantics unavailable", "construction owner unavailable", False
    ))


@dataclass(frozen=True)
class SpanNeed:
    path: str
    start_line: int
    end_line: int
    role: str
    reason: str
    priority: int
    token: str = ""

    def key(self) -> tuple[str, int, int, str]:
        return self.path, self.start_line, self.end_line, self.role

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "role": self.role,
            "reason": self.reason,
            "priority": self.priority,
            "token": self.token,
        }


def _ticket_literals(ticket: str) -> tuple[str, ...]:
    values: list[tuple[int, str]] = []
    for match in _QUOTED.finditer(ticket):
        value = match.group(2).strip()
        if re.search(r"[A-Za-z0-9]", value):
            values.append((match.start(), value))
    for match in _SLASH_DIGIT.finditer(ticket):
        raw = match.group(1).strip(".")
        normalized = raw.replace("/", "").replace("-", "").lower()
        if len(normalized) >= 4:
            values.append((match.start(), normalized))
    for match in _CODE_IDENT.finditer(ticket):
        value = match.group(1)
        if len(value.rsplit(".", 1)[-1].strip("_")) >= 4 and _code_identifier(value):
            values.append((match.start(), value))
    seen: set[str] = set()
    result: list[str] = []
    for _, value in sorted(values):
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result[:16])


def ticket_identifiers(ticket: str) -> tuple[str, ...]:
    """Return bounded identifier-shaped responsibility hints from the ticket."""

    values: list[str] = []
    for match in _CODE_IDENT.finditer(ticket):
        value = match.group(1)
        if _code_identifier(value):
            values.extend((value, value.rsplit(".", 1)[-1]))
    values.extend(re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*", ticket))
    stop = {"the", "and", "for", "with", "from", "this", "that"}
    return tuple(dict.fromkeys(value for value in values
                               if len(value) >= 3 and value.casefold() not in stop))[:48]


def _ticket_source_paths(ticket: str) -> tuple[str, ...]:
    """Return positively referenced relative source paths from the full ticket."""

    paths: list[str] = []
    seen: set[str] = set()
    for match in _SOURCE_PATH_MENTION.finditer(ticket):
        path = match.group(1)
        if path.startswith("./"):
            path = path[2:]
        parts = path.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            continue
        before = ticket[max(0, match.start() - 48):match.start()]
        after = ticket[match.end():match.end() + 64]
        # A disclaimer must describe this path immediately. Looking farther
        # ahead can demote earlier requested files because a later one is
        # labeled unrelated, including across sentence boundaries.
        if (_DISCLAIMED_SOURCE_PATH.match(after)
                or re.search(r"\b(?:do\s+not|don't|never|exclude)\s+"
                             r"(?:edit|modify|change|include)\s+$", before, re.IGNORECASE)):
            continue
        key = path.casefold()
        if key not in seen:
            seen.add(key)
            paths.append(path)
    return tuple(paths)


def _literal_pattern(token: str) -> re.Pattern[str]:
    variants = [re.escape(token)]
    if "/" not in token and "-" not in token and len(token) > 3:
        variants.append(re.escape(token.replace("_", "")))
    terminal = token.rsplit(".", 1)[-1]
    if "." in token and terminal != token:
        variants.append(re.escape(terminal))
    return re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(variants) + r")(?![A-Za-z0-9_])", re.IGNORECASE)


def lexical_sites(source: str, path: str, tokens: Sequence[str]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    lines = source.splitlines()
    for line_number, line in enumerate(lines, 1):
        if line.lstrip().startswith(("#", "//", "/*", "*", "<!--", "--")):
            continue
        for token in tokens:
            if _literal_pattern(str(token)).search(line):
                rows.append({"path": path, "line": line_number, "token": str(token), "text": line})
                break
    return tuple(rows)


def _outline(source: str, path: str) -> Mapping[str, object] | None:
    try:
        from mc_repo_intel_vendor.ts_dispatch import outline_source
        value = outline_source(source, path)
    except (ImportError, OSError, TypeError, ValueError, SyntaxError):
        return None
    return value if isinstance(value, Mapping) else None


def _outline_capability(
        outline: Mapping[str, object] | None,
        adapter: LanguageAdapter,
) -> dict[str, str]:
    """Normalize parser capability into evidence, never a silent absence."""

    if not isinstance(outline, Mapping):
        return {
            "parser": adapter.parser,
            "capability": "UNAVAILABLE",
            "completeness": "UNKNOWN",
            "status": "UNKNOWN",
            "reason": "outline unavailable",
        }
    degraded = bool(outline.get("degraded"))
    return {
        "parser": str(outline.get("parser") or adapter.parser),
        "capability": "DEGRADED" if degraded else "AVAILABLE",
        "completeness": "UNKNOWN" if degraded else "KNOWN",
        "status": "UNKNOWN" if degraded else adapter.evidence_status,
        "reason": str(outline.get("degraded_reason") or "") if degraded else "",
    }


def parser_capability(path: str, source: str) -> dict[str, str]:
    """Return the language/parser receipt used by caches and inventory owners."""

    adapter = language_adapter_for(path)
    return _outline_capability(_outline(source, path), adapter)


def _row_span(row: Mapping[str, object], fallback: int) -> tuple[int, int]:
    start = int(row.get("line", row.get("start_line", fallback)) or fallback)
    end = int(row.get("end_line", row.get("end", start)) or start)
    return max(1, start), max(start, end)


def _python_span_nodes(source: str, sites: Sequence[Mapping[str, object]]) -> tuple[SpanNeed, ...]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return tuple()
    nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    result: list[SpanNeed] = []
    for site in sites:
        line = int(site["line"])
        token = str(site["token"])
        containing = [node for node in nodes
                      if int(getattr(node, "lineno", 0) or 0) <= line <= int(getattr(node, "end_lineno", node.lineno) or node.lineno)]
        if containing:
            node = min(containing, key=lambda item: int(getattr(item, "end_lineno", item.lineno) or item.lineno) - int(item.lineno))
            result.append(SpanNeed(str(site["path"]), int(node.lineno), int(getattr(node, "end_lineno", node.lineno) or node.lineno), "body", "lexical-site enclosing body", 0, token))
            continue
        result.append(SpanNeed(str(site["path"]), line, line, "statement", "lexical-site statement; AST enclosure unavailable", 1, token))
    return tuple(result)


def _outline_span_needs(source: str, path: str, sites: Sequence[Mapping[str, object]],
                        adapter: LanguageAdapter) -> tuple[SpanNeed, ...]:
    outline = _outline(source, path)
    if adapter.language == "python":
        result = list(_python_span_nodes(source, sites))
    else:
        result = []
        rows = [row for key in ("functions", "types", "module_statements", "imports")
                for row in (outline.get(key, ()) if outline else ()) if isinstance(row, Mapping)]
        for site in sites:
            line = int(site["line"])
            containing = [row for row in rows if _row_span(row, line)[0] <= line <= _row_span(row, line)[1]]
            if containing:
                row = min(containing, key=lambda item: _row_span(item, line)[1] - _row_span(item, line)[0])
                start, end = _row_span(row, line)
                role = "body" if row in (outline or {}).get("functions", ()) else "statement"
                result.append(SpanNeed(path, start, end, role, "lexical-site outline enclosure", 0, str(site["token"])))
            else:
                result.append(SpanNeed(path, line, line, "statement", "lexical-site; outline enclosure unavailable", 1, str(site["token"])))
    return tuple(result)


def source_closure_ranges(
        path: str, source: str, anchor_lines: Iterable[tuple[int, str]],
) -> tuple[tuple[int, int, str], ...]:
    """Expand lexical anchors to complete outline-owned statements/bodies."""

    anchors = tuple(
        (int(line), str(reason))
        for line, reason in anchor_lines
        if int(line) > 0
    )
    rows = tuple({"path": path, "line": line, "token": reason}
                 for line, reason in anchors)
    if not rows:
        return ()
    needs = _outline_span_needs(source, path, rows, language_adapter_for(path))
    if not needs:
        return tuple((line, line, reason) for line, reason in anchors)
    return tuple(dict.fromkeys((need.start_line, need.end_line, need.reason) for need in needs))


def _module_needs(source: str, path: str, adapter: LanguageAdapter, *, limit: int = 24) -> tuple[SpanNeed, ...]:
    if adapter.language == "python":
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return tuple()
        # Definitions are decisive closure units, not module initialization.
        # Including their headers here spends the initialization floor before
        # the selected body/type members can be delivered.
        nodes = [node for node in tree.body
                 if isinstance(node, ast.stmt)
                 and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        return tuple(SpanNeed(path, int(node.lineno), int(getattr(node, "end_lineno", node.lineno) or node.lineno),
                              "module", "module initialization/import boundary", 1)
                     for node in nodes[:limit])
    outline = _outline(source, path)
    if outline is None:
        return tuple()
    result: list[SpanNeed] = []
    for key in ("imports", "module_statements"):
        for row in outline.get(key, ()) or ():
            if not isinstance(row, Mapping):
                continue
            start, end = _row_span(row, 1)
            result.append(SpanNeed(path, start, end, "module", "module initialization/import boundary", 1))
            if len(result) >= limit:
                return tuple(result)
    return tuple(result)


def _safe_path(root: Path, raw: str, *, root_resolved: Path | None = None) -> str | None:
    raw_text = str(raw).replace("\\", "/")
    if raw_text.startswith(("/", "~")) or any(part == ".." for part in raw_text.split("/")):
        return None
    relative = raw_text.removeprefix("./")
    if not relative or any(part.casefold() in _DENIED_PARTS for part in relative.split("/")):
        return None
    resolved_root = root_resolved if root_resolved is not None else root.resolve()
    candidate = (resolved_root / relative).resolve()
    try:
        return candidate.relative_to(resolved_root).as_posix()
    except (OSError, ValueError):
        return None


def _read_source(root: Path, relative: str, *, root_resolved: Path | None = None) -> tuple[str, str | None]:
    safe = _safe_path(root, relative, root_resolved=root_resolved)
    if safe is None:
        return "OUT_OF_SCOPE", None
    path = (root_resolved if root_resolved is not None else root.resolve()) / safe
    try:
        if not path.is_file():
            return "ABSENT", None
        return "PRESENT", path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return "UNAVAILABLE", None


@lru_cache(maxsize=256)
def _source_imports(source: str, language: str) -> tuple[str, ...]:
    if language == "python":
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return ()
        imports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(("." * node.level) + node.module)
        return tuple(imports)
    return tuple(
        [match.group(1) for match in _IMPORT_FROM.finditer(source)]
        + [match.group(1) for match in _REQUIRE.finditer(source)]
    )


def _dependency_candidates(root: Path, path: str, source: str, adapter: LanguageAdapter,
                           *, root_resolved: Path | None = None,
                           safe_path: Callable[[str], str | None] | None = None) -> tuple[str, ...]:
    imports = _source_imports(source, adapter.language)
    base = Path(path).parent
    candidates: list[str] = []
    for imported in imports[:32]:
        if imported.startswith("."):
            stem = (base / imported.lstrip("."))
        else:
            stem = Path(imported.replace(".", "/")) if adapter.language == "python" else base / imported
        # Keep candidates relative to ``root``.  Passing ``root / stem`` to
        # _safe_path would make the normalizer join root twice and silently
        # drop the dependency from the revision vector.
        roots = (Path("."), Path("backend"), Path("src"))
        for prefix in roots:
            raw = (prefix / stem).as_posix() if not stem.is_absolute() else stem.as_posix()
            variants = [raw]
            if Path(raw).suffix == "":
                variants.extend(raw + ext for ext in _SOURCE_EXTENSIONS)
                variants.extend((Path(raw) / ("__init__" + ext)).as_posix() for ext in (".py",))
            for variant in variants:
                safe = safe_path(variant) if safe_path is not None else _safe_path(
                    root, variant, root_resolved=root_resolved)
                source_root = root_resolved if root_resolved is not None else root
                if safe and (source_root / safe).is_file() and not _is_test_path(safe):
                    candidates.append(safe)
    return tuple(dict.fromkeys(candidates))


def connected_source_paths(
        root: Path, declared_paths: Sequence[str], target_path: str, ticket: str,
        *, lexical_paths: Iterable[str] = (), construction_names: Iterable[str] = (),
) -> tuple[str, ...]:
    """Select connected product source from lexical/import responsibility."""

    normalized = tuple(sorted({
        safe for raw in declared_paths if (safe := _safe_path(root, str(raw))) is not None
    }, key=lambda path: (path != target_path, path)))
    selected: set[str] = {str(path) for path in lexical_paths if str(path) in normalized}
    if target_path in normalized:
        selected.add(target_path)
    names = tuple(dict.fromkeys((*ticket_identifiers(ticket), *construction_names)))
    pending = [path for path in normalized if path in selected]
    seen: set[str] = set()
    while pending:
        path = pending.pop(0)
        if path in seen:
            continue
        seen.add(path)
        state, source = _read_source(root, path)
        if state != "PRESENT" or source is None:
            continue
        adapter = language_adapter_for(path)
        dependencies = _dependency_candidates(root, path, source, adapter)
        for dependency in dependencies:
            if dependency in normalized and dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
        for candidate in normalized:
            if candidate in selected:
                continue
            candidate_state, candidate_source = _read_source(root, candidate)
            if candidate_state != "PRESENT" or candidate_source is None:
                continue
            stem = Path(candidate).stem
            if any(Path(dependency).stem == stem for dependency in dependencies):
                selected.add(candidate)
                pending.append(candidate)
                continue
            if any(re.search(rf"\b{re.escape(name)}\b", candidate_source)
                   for name in names):
                selected.add(candidate)
    return tuple(path for path in normalized if path in selected)


def _is_test_path(path: str) -> bool:
    parts = {part.casefold() for part in path.split("/")}
    name = Path(path).name.casefold()
    return bool(parts & {"test", "tests", "__tests__", "spec", "specs"}) or name.startswith("test_") or ".test." in name or ".spec." in name


def dependency_revision(root: Path, paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(dict.fromkeys(str(item) for item in paths)):
        digest.update(path.encode("utf-8") + b"\0")
        try:
            digest.update((root / path).read_bytes())
        except (OSError, UnicodeError):
            digest.update(b"<unavailable>")
    return "sha256:" + digest.hexdigest()


class RevisionAwareFactCache:
    """Small per-run cache keyed by source *and dependency* revisions."""

    def __init__(self) -> None:
        self._values: dict[tuple[object, ...], Mapping[str, object] | None] = {}

    def get_or_compute(self, key: tuple[object, ...], compute: Callable[[], Mapping[str, object] | None]) -> Mapping[str, object] | None:
        if key not in self._values:
            self._values[key] = compute()
        return self._values[key]

    def invalidate_path(self, path: str) -> None:
        self._values = {key: value for key, value in self._values.items() if key[0] != path}


def initialization_fact(root: Path, path: str, names: Sequence[str], *, ticket: str = "",
                        cache: RevisionAwareFactCache | None = None) -> Mapping[str, object] | None:
    """Resolve initialization/construction ownership without universalizing Python."""

    adapter = language_adapter_for(path)
    state, source = _read_source(root, path)
    if not adapter.supported:
        return {"status": "UNRUNNABLE", "language": adapter.language, "path": path,
                "reason": "unsupported language/runtime adapter", "parser": adapter.parser,
                "parser_capability": "UNKNOWN", "completeness": "UNKNOWN"}
    if state != "PRESENT" or source is None:
        return {"status": "UNKNOWN", "language": adapter.language, "path": path,
                "reason": state, "parser": adapter.parser,
                "parser_capability": "UNAVAILABLE", "completeness": "UNKNOWN"}
    dependencies = _dependency_candidates(root, path, source, adapter)
    source_rev = source_revision(source)
    deps_rev = dependency_revision(root, (path, *dependencies))
    wanted = tuple(dict.fromkeys(str(name) for name in names if str(name)))
    capability = parser_capability(path, source)
    key = (path, source_rev, deps_rev, wanted, adapter.language,
           capability["parser"], capability["capability"])

    def compute() -> Mapping[str, object] | None:
        if capability["completeness"] != "KNOWN":
            return {"status": "UNKNOWN", "language": adapter.language, "path": path,
                    "reason": ("parser capability unavailable: " + str(capability["reason"])
                               if capability["reason"] else "parser capability degraded"),
                    "parser": capability["parser"],
                    "parser_capability": capability["capability"],
                    "completeness": capability["completeness"],
                    "source_revision": source_rev, "dependency_revision": deps_rev,
                    "adapter": adapter.module_initialization}
        if adapter.language == "python":
            try:
                from kernel_repo_intel import import_construction_fact
                fact = import_construction_fact(root, path, wanted)
            except (ImportError, OSError, TypeError, ValueError):
                fact = None
            if fact is not None:
                return {**dict(fact), "status": "UNCOVERED", "language": adapter.language,
                        "source_revision": source_rev, "dependency_revision": deps_rev,
                        "adapter": adapter.module_initialization,
                        "parser": capability["parser"],
                        "parser_capability": capability["capability"],
                        "completeness": capability["completeness"]}
        outline = _outline(source, path)
        outline_state = _outline_capability(outline, adapter)
        if outline_state["completeness"] != "KNOWN":
            return {"status": "UNKNOWN", "language": adapter.language, "path": path,
                    "reason": outline_state["reason"] or "parser capability degraded",
                    "parser": outline_state["parser"],
                    "parser_capability": outline_state["capability"],
                    "completeness": outline_state["completeness"],
                    "source_revision": source_rev, "dependency_revision": deps_rev,
                    "adapter": adapter.module_initialization}
        rows = [row for group in ("module_statements", "imports")
                for row in (outline.get(group, ()) if outline else ()) if isinstance(row, Mapping)]
        wanted_lower = tuple(item.casefold() for item in wanted)
        lines = source.splitlines()
        for row in rows:
            start, end = _row_span(row, 1)
            text = " ".join(lines[start - 1:end]) if lines else ""
            if any(item in text.casefold() for item in wanted_lower):
                name = next((item for item in wanted if item.casefold() in text.casefold()), "construction")
                return {"status": "UNCOVERED", "language": adapter.language, "path": path,
                        "line": start, "statement": text.strip(), "name": name,
                        "constructor": {"path": path, "line": start, "text": text.strip()},
                        "chain": [], "source_revision": source_rev,
                        "dependency_revision": deps_rev,
                        "adapter": adapter.module_initialization,
                        "parser": outline_state["parser"],
                        "parser_capability": outline_state["capability"],
                        "completeness": outline_state["completeness"]}
        return {"status": "UNCOVERED", "language": adapter.language, "path": path,
                "reason": "no current construction statement resolved",
                "source_revision": source_rev, "dependency_revision": deps_rev,
                "adapter": adapter.module_initialization,
                "parser": outline_state["parser"],
                "parser_capability": outline_state["capability"],
                "completeness": outline_state["completeness"]}

    return (cache or RevisionAwareFactCache()).get_or_compute(key, compute)


def snapshot_for_source(path: str, source: str) -> SourceSnapshot:
    """Capture immutable source bytes through the existing EC01 contract."""

    revision = source_revision(source)
    return SourceSnapshot.capture(
        source.encode("utf-8"), store_id="store:request", namespace_id="namespace:public",
        repo_id="repo:workspace", worktree_id="worktree:current", source_ref=path,
        source_revision=revision, origin_receipt_id="receipt:request-source",
        visibility=EvidenceVisibility.PUBLIC,
    )


@dataclass(frozen=True)
class ConstructionFact:
    """Compatibility DTO for static ownership facts; never behavioral proof."""

    language: str
    status: EvidenceStatus
    source_ref: str
    source_revision: str
    dependency_revisions: tuple[tuple[str, str], ...]
    requested_names: tuple[str, ...]
    owner_path: str | None = None
    owner_line: int | None = None
    initialization_kind: str | None = None
    reason: str = ""
    parser: str = "unknown"
    parser_capability: str = "UNKNOWN"
    completeness: str = "UNKNOWN"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-construction-fact/1",
            "language": self.language,
            "status": self.status.value,
            "source_ref": self.source_ref,
            "source_revision": self.source_revision,
            "dependency_revisions": [list(item) for item in self.dependency_revisions],
            "requested_names": list(self.requested_names),
            "owner_path": self.owner_path,
            "owner_line": self.owner_line,
            "initialization_kind": self.initialization_kind,
            "reason": self.reason,
            "parser": self.parser,
            "parser_capability": self.parser_capability,
            "completeness": self.completeness,
        }


def construction_fact_for_source(
        path: str, source: str, requested_names: Iterable[str] = (),
        dependencies: Sequence[tuple[str, str]] = (), *,
        runtime_available: bool = True,
        cache: "RevisionBoundFactCache | None" = None,
) -> ConstructionFact:
    """Resolve source ownership with a language adapter, not Python import logic."""

    snapshot = snapshot_for_source(path, source)
    names = tuple(dict.fromkeys(str(name) for name in requested_names if str(name)))
    dependency_revisions = tuple((str(dep_path), source_revision(dep_source))
                                 for dep_path, dep_source in dependencies)
    adapter = language_adapter_for(path)
    if not runtime_available:
        return ConstructionFact(adapter.language, EvidenceStatus.UNRUNNABLE, path,
                                snapshot.source_revision, dependency_revisions, names,
                                reason="runtime-unavailable", parser=adapter.parser)
    if not adapter.supported:
        return ConstructionFact(adapter.language, EvidenceStatus.UNKNOWN, path,
                                snapshot.source_revision, dependency_revisions, names,
                                reason="language-unsupported", parser=adapter.parser)
    if not names:
        return ConstructionFact(adapter.language, EvidenceStatus.UNKNOWN, path,
                                snapshot.source_revision, dependency_revisions, names,
                                reason="no-construction-name", parser=adapter.parser)
    outline = _outline(source, path)
    outline_state = _outline_capability(outline, adapter)
    parser = outline_state["parser"]
    if outline is None:
        return ConstructionFact(adapter.language, EvidenceStatus.UNAVAILABLE, path,
                                snapshot.source_revision, dependency_revisions, names,
                                reason="outline-unavailable", parser=parser,
                                parser_capability=outline_state["capability"],
                                completeness=outline_state["completeness"])
    if outline_state["completeness"] != "KNOWN":
        return ConstructionFact(adapter.language, EvidenceStatus.UNKNOWN, path,
                                snapshot.source_revision, dependency_revisions, names,
                                reason=outline_state["reason"] or "parser capability degraded",
                                parser=parser,
                                parser_capability=outline_state["capability"],
                                completeness=outline_state["completeness"])
    for line_number, line in enumerate(source.splitlines(), 1):
        if not any(re.search(rf"\b{re.escape(name)}\b", line) for name in names):
            continue
        status = EvidenceStatus.PARTIAL if line.rstrip().endswith(("=", ":", "->")) else EvidenceStatus.FOUND
        return ConstructionFact(
            adapter.language, status, path, snapshot.source_revision,
            dependency_revisions, names, path, line,
            "module" if not line.startswith((" ", "\t")) else "runtime",
            "outline-backed static construction site", parser,
            outline_state["capability"], outline_state["completeness"],
        )
    for dependency_path, dependency_source in dependencies:
        nested = construction_fact_for_source(dependency_path, dependency_source, names,
                                               runtime_available=runtime_available)
        if nested.status in {EvidenceStatus.FOUND, EvidenceStatus.PARTIAL}:
            return ConstructionFact(
                adapter.language, nested.status, path, snapshot.source_revision,
                dependency_revisions, names, nested.owner_path, nested.owner_line,
                nested.initialization_kind, "dependency-backed static construction site",
                nested.parser, nested.parser_capability, nested.completeness,
            )
    return ConstructionFact(adapter.language, EvidenceStatus.NOT_FOUND, path,
                            snapshot.source_revision, dependency_revisions, names,
                            reason="no current construction site", parser=parser,
                            parser_capability=outline_state["capability"],
                            completeness=outline_state["completeness"])


class RevisionBoundFactCache:
    """Compatibility cache whose key includes source and dependency revisions."""

    def __init__(self) -> None:
        self._values: dict[tuple[object, ...], ConstructionFact] = {}
        self._last: dict[tuple[str, tuple[str, ...]], tuple[object, ...]] = {}
        self.invalidated_count = 0

    def resolve(self, snapshot: SourceSnapshot, requested_names: Iterable[str] = (),
                dependencies: Sequence[SourceSnapshot] = (), *,
                runtime_available: bool = True) -> ConstructionFact:
        names = tuple(dict.fromkeys(str(name) for name in requested_names if str(name)))
        dependency_key = tuple((item.source_ref, item.source_revision, item.source_sha256)
                               for item in dependencies)
        source = snapshot.body.decode(snapshot.encoding, errors="replace")
        capability = parser_capability(snapshot.source_ref, source)
        identity = (snapshot.source_ref, names)
        key = (snapshot.source_ref, snapshot.source_revision, snapshot.source_sha256,
               dependency_key, runtime_available, capability["parser"],
               capability["capability"])
        previous = self._last.get(identity)
        if previous is not None and previous != key:
            self.invalidated_count += 1
        self._last[identity] = key
        if key not in self._values:
            deps = tuple((item.source_ref, item.body.decode(item.encoding, errors="replace"))
                         for item in dependencies)
            self._values[key] = construction_fact_for_source(
                snapshot.source_ref, source, names, deps,
                runtime_available=runtime_available,
            )
        return self._values[key]


def render_construction_fact(fact: ConstructionFact | Mapping[str, object]) -> str:
    values = fact.to_dict() if isinstance(fact, ConstructionFact) else dict(fact)
    owner_path = values.get("owner_path") or values.get("path")
    owner_line = values.get("owner_line", values.get("line"))
    owner = f"{owner_path}:{owner_line}" if owner_path and owner_line else "none"
    return (
        "[CONSTRUCTION OWNERSHIP]\n"
        f"language={values.get('language', 'unknown')} status={values.get('status', 'UNKNOWN')} "
        f"parser={values.get('parser', 'unknown')} "
        f"parser_capability={values.get('parser_capability', 'UNKNOWN')} "
        f"completeness={values.get('completeness', 'UNKNOWN')} "
        f"source={values.get('source_ref', values.get('path', 'unknown'))}@"
        f"{values.get('source_revision', 'UNKNOWN')} owner={owner} "
        f"initialization={values.get('initialization_kind', 'unknown')}\n"
        f"reason={values.get('reason', 'static source ownership only; execution remains unobserved')}"
    )


@dataclass
class SourceClosure:
    ticket: TicketIdentity
    paths: tuple[str, ...]
    lexical_sites: tuple[dict[str, object], ...]
    units: tuple[SpanNeed, ...]
    omissions: list[dict[str, object]]
    revisions: dict[str, str]
    adapters: dict[str, dict[str, object]]
    coverage: dict[str, dict[str, object]]
    delivery: dict[str, dict[str, object]] = field(default_factory=dict)
    final_delivery: dict[str, dict[str, object]] = field(default_factory=dict)
    skipped_files: list[dict[str, object]] = field(default_factory=list)
    dependency_revisions: dict[str, str] = field(default_factory=dict)
    identities: dict[str, dict[str, object]] = field(default_factory=dict)
    target_path: str = ""
    source_contents: dict[str, str] = field(default_factory=dict)
    workspace_authorization: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "ticket_sha256": self.ticket.sha256,
            "target_path": self.target_path,
            "paths": list(self.paths),
            "lexical_sites": [dict(item) for item in self.lexical_sites],
            "units": [item.as_dict() for item in self.units],
            "omissions": [dict(item) for item in self.omissions],
            "skipped_files": [dict(item) for item in self.skipped_files],
            "revisions": dict(self.revisions),
            "dependency_revisions": dict(self.dependency_revisions),
            "identities": {key: dict(value) for key, value in self.identities.items()},
            "adapters": {key: dict(value) for key, value in self.adapters.items()},
            "coverage": {key: dict(value) for key, value in self.coverage.items()},
            "delivery": {key: dict(value) for key, value in self.delivery.items()},
            "final_delivery": {key: dict(value) for key, value in self.final_delivery.items()},
            # Host-only current public bytes let the final-wire verifier prove
            # bounded windows rather than trusting their range metadata.  The
            # renderer never includes this field in model-visible receipts.
            "source_contents": dict(self.source_contents),
            "workspace_authorization": dict(self.workspace_authorization),
        }


def build_source_closure(root: Path, paths: Sequence[str], ticket: TicketIdentity | str,
                         *, target_path: str = "", max_files: int = 12,
                         max_units: int = 96,
                         parent_sources: Mapping[str, str] | None = None) -> SourceClosure:
    identity = ticket if isinstance(ticket, TicketIdentity) else TicketIdentity("task:anonymous", ticket)
    root_resolved = root.resolve()
    safe_path_cache: dict[str, str | None] = {}
    source_cache: dict[str, tuple[str, str | None]] = {}
    dependency_bytes: dict[str, bytes | None] = {}

    def safe_path(raw: str) -> str | None:
        raw_text = str(raw)
        if raw_text not in safe_path_cache:
            safe_path_cache[raw_text] = _safe_path(
                root, raw_text, root_resolved=root_resolved)
        return safe_path_cache[raw_text]

    def read_source(raw: str) -> tuple[str, str | None]:
        safe = safe_path(raw)
        if safe is None:
            return "OUT_OF_SCOPE", None
        if safe not in source_cache:
            source_cache[safe] = _read_source(root, safe, root_resolved=root_resolved)
        return source_cache[safe]

    def revision_for_paths(revision_paths: Sequence[str]) -> str:
        digest = hashlib.sha256()
        for relative in sorted(dict.fromkeys(str(item) for item in revision_paths)):
            digest.update(relative.encode("utf-8") + b"\0")
            if relative not in dependency_bytes:
                try:
                    dependency_bytes[relative] = (root_resolved / relative).read_bytes()
                except (OSError, UnicodeError):
                    dependency_bytes[relative] = None
            contents = dependency_bytes[relative]
            digest.update(contents if contents is not None else b"<unavailable>")
        return "sha256:" + digest.hexdigest()

    declared: list[str] = []
    omissions: list[dict[str, object]] = []
    skipped_files: list[dict[str, object]] = []
    parent_sources = parent_sources or {}
    for raw in paths:
        raw_text = str(raw)
        safe = safe_path(raw_text)
        if safe is None:
            omission = {"path": raw_text, "status": "OUT_OF_SCOPE",
                        "reason": "source path is outside authorized workspace scope"}
            omissions.append(omission)
            skipped_files.append(dict(omission))
            continue
        declared.append(safe)
    ordered = list(dict.fromkeys(([target_path] if target_path else []) + declared))
    tokens = _ticket_literals(identity.text)
    mentioned_source_paths = {
        safe.casefold()
        for raw in _ticket_source_paths(identity.text)
        if (safe := safe_path(raw)) is not None
    }
    # Tickets commonly name one full path then list sibling basenames. Match
    # those explicit filenames against declared paths before bounded selection.
    explicit_source_paths = {
        path.casefold() for path in ordered
        if path.casefold() in mentioned_source_paths
        or path.rsplit("/", 1)[-1].casefold() in mentioned_source_paths
    }
    for path in ordered:
        if (not path.lower().endswith(_SOURCE_EXTENSIONS)
                and re.search(r"(?<![A-Za-z0-9_./-])" + re.escape(path)
                              + r"(?![A-Za-z0-9_/-]|\.[A-Za-z0-9_.-])", identity.text)):
            explicit_source_paths.add(path.casefold())
    explicit_names = {
        value.casefold()
        for token in tokens
        for value in (token, token.rsplit(".", 1)[-1])
    }
    explicit_qualified = {
        token.casefold() for token in tokens if "." in token
    }

    def source_priority(path: str) -> tuple[int, str]:
        # Declaration order is not relevance. A normalized path uniquely
        # identifies the revision read from this closure's source snapshot.
        if path == target_path:
            return (0, path)
        if path.casefold() in explicit_source_paths:
            return (1, path)
        state, source = read_source(path)
        if state != "PRESENT" or source is None:
            return (4, path)
        outline = _outline(source, path)
        named_definition = False
        qualified_definition = False
        if isinstance(outline, Mapping):
            for group in ("types", "functions"):
                for row in outline.get(group, ()) or ():
                    if not isinstance(row, Mapping):
                        continue
                    name = str(row.get("name") or "").casefold()
                    parent = str(row.get("parent") or "").casefold()
                    qualified = ".".join(part for part in (parent, name) if part)
                    if parent and qualified and any(
                            value == qualified or value.endswith("." + qualified)
                            for value in explicit_qualified):
                        qualified_definition = True
                        named_definition = True
                        break
                    if name in explicit_names or qualified in explicit_names:
                        named_definition = True
                        break
                if named_definition:
                    break
        if qualified_definition:
            return (2, path)
        if named_definition:
            return (3, path)
        if lexical_sites(source, path, tokens):
            return (4, path)
        return (5, path)

    priorities = {path: source_priority(path) for path in ordered}
    ordered.sort(key=priorities.__getitem__)
    priority_by_path = {path: priority[0] for path, priority in priorities.items()}
    lexical: list[dict[str, object]] = []
    units: list[SpanNeed] = []
    revisions: dict[str, str] = {}
    dependency_revisions: dict[str, str] = {}
    identities: dict[str, dict[str, object]] = {}
    adapters: dict[str, dict[str, object]] = {}
    source_contents: dict[str, str] = {}
    selected: list[str] = []
    required_dependencies: set[str] = set()
    pending = list(ordered)
    while pending and len(selected) < max_files:
        path = pending.pop(0)
        if path in selected:
            continue
        state, source = read_source(path)
        if state != "PRESENT" or source is None:
            omission = {"path": path, "status": state, "reason": "source unavailable for closure"}
            omissions.append(omission)
            skipped_files.append(dict(omission))
            continue
        selected.append(path)
        source_contents[path] = source
        revisions[path] = source_revision(source)
        parent = parent_sources.get(path)
        parent_revision = source_revision(parent) if isinstance(parent, str) else None
        identities[path] = {
            "current_revision": revisions[path],
            "parent_revision": parent_revision,
            "status": ("UNKNOWN" if parent is None else
                       "UNCHANGED" if parent_revision == revisions[path] else "CHANGED"),
        }
        adapter = language_adapter_for(path)
        outline_state = _outline_capability(_outline(source, path), adapter)
        adapters[path] = {"language": adapter.language, "parser": adapter.parser,
                          "status": outline_state["status"],
                          "parser_capability": outline_state["capability"],
                          "completeness": outline_state["completeness"],
                          "degraded_reason": outline_state["reason"],
                          "module_initialization": adapter.module_initialization,
                          "construction_ownership": adapter.construction_ownership}
        sites = lexical_sites(source, path, tokens)
        lexical.extend(sites)
        units.extend(_outline_span_needs(source, path, sites, adapter))
        if path == target_path or sites:
            units.extend(_module_needs(source, path, adapter))
        dependencies = _dependency_candidates(
            root, path, source, adapter, root_resolved=root_resolved,
            safe_path=safe_path)
        required_dependencies.update(dependencies)
        for dependency in dependencies:
            if dependency not in selected and dependency not in pending and len(selected) + len(pending) < max_files:
                pending.append(dependency)
        dependency_revisions[path] = revision_for_paths((path, *dependencies))
    for path in ordered:
        if path not in selected and not any(item.get("path") == path for item in omissions):
            # A lexical hit is a selection hint, not an unbounded delivery debt.
            # Named paths/definitions, dependencies and unavailable sources keep
            # blocking omissions; omitted incidental matches remain visible as
            # unselected candidates in the closure's partial-scan diagnostics.
            not_selected = (priority_by_path.get(path, 5) >= 4
                            and path not in required_dependencies
                            and read_source(path)[0] == "PRESENT")
            omission = {
                "path": path,
                "status": "NOT_SELECTED" if not_selected else "PARTIAL",
                "reason": ("no ticket-relevant source selected" if not_selected
                           else "closure file limit reached"),
            }
            if not not_selected:
                omission["continuation"] = "expand:file:" + hashlib.sha256(path.encode()).hexdigest()
            omissions.append(omission)
            skipped_files.append(dict(omission))
    deduped: list[SpanNeed] = []
    seen: set[tuple[str, int, int, str]] = set()
    for unit in sorted(units, key=lambda item: (item.priority, item.path, item.start_line, item.end_line, item.role)):
        if unit.key() not in seen:
            seen.add(unit.key())
            deduped.append(unit)
    if len(deduped) > max_units:
        for unit in deduped[max_units:]:
            omissions.append({"path": unit.path, "range": f"{unit.start_line}-{unit.end_line}",
                              "status": "PARTIAL", "reason": "semantic unit exceeds bounded closure",
                              "continuation": "expand:" + hashlib.sha256(repr(unit.as_dict()).encode()).hexdigest()})
        deduped = deduped[:max_units]
    complete = not omissions and not skipped_files and all(
        item.get("status") not in {"UNKNOWN", "UNRUNNABLE", "PARTIAL"}
        and item.get("completeness") == "KNOWN"
        for item in adapters.values()
    )
    completeness_status = "COMPLETE" if complete else "PARTIAL" if selected else "UNKNOWN"
    coverage = {
        "completeness": {"status": completeness_status, "selected_files": len(selected),
                         "declared_files": len(ordered), "skipped_files": len(skipped_files)},
        "skipped_file": {"status": "COMPLETE" if not skipped_files else "PARTIAL",
                         "count": len(skipped_files)},
        "omission": {"status": "COMPLETE" if not omissions else "PARTIAL",
                      "count": len(omissions)},
        "lexical_site": {"status": "COMPLETE" if complete else "PARTIAL", "selected": len(lexical), "expected": len(lexical)},
        "acquired_span": {"status": "NOT_DELIVERED", "selected": len(deduped), "acquired": 0},
        "delivered_span": {"status": "UNCOVERED", "selected": len(deduped), "delivered": 0},
        "dependency_revision": {"status": "CURRENT" if dependency_revisions else "UNKNOWN",
                                 "values": dict(dependency_revisions)},
        "parser_capability": {
            "status": "KNOWN" if complete else "UNKNOWN",
            "values": {path: dict(value) for path, value in adapters.items()},
        },
        "current_parent_identity": {"status": "CURRENT" if identities else "UNKNOWN",
                                     "values": {key: dict(value) for key, value in identities.items()}},
        "textual_edit": {"status": "NOT_EVALUATED", "changed_sites": 0},
        "resolved_use_definition": {"status": "UNCOVERED", "reason": "navigation closure is source-bound, not a graph claim"},
        "observed_behavior": {"status": "UNCOVERED", "reason": "no execution evidence in request assembly"},
    }
    workspace_authorization = {
        "scope": "workspace:" + hashlib.sha256(str(root_resolved).encode("utf-8")).hexdigest()[:32],
        "generation": "generation:" + hashlib.sha256(json.dumps(
            {"revisions": revisions, "dependencies": dependency_revisions},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()[:32],
    }
    return SourceClosure(identity, tuple(selected), tuple(lexical), tuple(deduped), omissions,
                         revisions, adapters, coverage, skipped_files=skipped_files,
                         dependency_revisions=dependency_revisions, identities=identities,
                         target_path=target_path, source_contents=source_contents,
                         workspace_authorization=workspace_authorization)


def _empty_delivery_status(plan: Mapping[str, object], path: str) -> str:
    """Distinguish an intentionally unselected path from unavailable source."""

    declared = plan.get("paths", ())
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes, bytearray)):
        return "UNKNOWN"
    if path not in {str(item) for item in declared}:
        return "UNKNOWN"
    for key in ("omissions", "skipped_files"):
        rows = plan.get(key, ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            continue
        if any(isinstance(row, Mapping) and str(row.get("path", "")) == path for row in rows):
            return "UNKNOWN"
    adapters = plan.get("adapters", {})
    adapter = adapters.get(path) if isinstance(adapters, Mapping) else None
    if isinstance(adapter, Mapping):
        status = str(adapter.get("status", "")).upper()
        completeness = str(adapter.get("completeness", "")).upper()
        if status in {"UNKNOWN", "UNRUNNABLE", "UNAVAILABLE", "FAILED", "PARTIAL", "MISSING", "NOT_FOUND"}:
            return "UNKNOWN"
        if completeness in {"UNKNOWN", "UNAVAILABLE", "FAILED", "PARTIAL", "MISSING", "NOT_FOUND"}:
            return "UNKNOWN"
    if path == str(plan.get("target_path", "")):
        return "PARTIAL"
    return "NOT_SELECTED"


def _canonical_line_ranges(ranges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Sort and merge overlapping or adjacent positive source line ranges."""
    normalized = sorted(
        (int(start), int(end)) for start, end in ranges
        if type(start) is int and type(end) is int and start >= 1 and end >= start
    )
    merged: list[list[int]] = []
    for start, end in normalized:
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def unit_carried(start: int, end: int, carriers: Sequence[tuple[int, int]]) -> bool:
    """The shared per-unit delivery predicate: ONE delivered carrier contains the whole unit.

    The FRAME_ONLY allocator places each unit in exactly one carrier (kernel_frame_request_allocation rejects a
    unit delivered by several), so the final verifier must not call a unit delivered when only the union of
    several windows covers it (bug class two-views-1: a recorded long run PRE request 9)."""
    return any(low <= start and end <= high for low, high in carriers)


def record_source_delivery(plan: Mapping[str, object], path: str,
                           ranges: Sequence[tuple[int, int]], *,
                           acquired_spans: int | None = None,
                           single_carrier: bool = False) -> dict[str, object]:
    """Record delivered semantic units without turning delivery into behavior.

    ``single_carrier`` (final-wire receipts): ``ranges`` lists one span per delivered carrier window, and a unit
    counts as delivered only when one of them contains it (``unit_carried``). Merged spans still drive span counts.
    """

    carriers = tuple(
        (int(start), int(end)) for start, end in ranges
        if type(start) is int and type(end) is int and start >= 1 and end >= start
    )
    ranges = _canonical_line_ranges(ranges)
    units = [item for item in plan.get("units", ()) if isinstance(item, Mapping) and item.get("path") == path]
    covered = []
    omitted = []
    for index, unit in enumerate(units):
        start, end = int(unit.get("start_line", 0)), int(unit.get("end_line", 0))
        if unit_carried(start, end, carriers if single_carrier else ranges):
            covered.append(index)
        else:
            omitted.append(index)
    unit_handles = [str(unit.get("handle", unit.get("unit_id", index)))
                    for index, unit in enumerate(units)]
    covered_ranges = tuple(
        (low, high) for low, high in ranges
        if any(low <= int(unit.get("end_line", 0))
               and int(unit.get("start_line", 0)) <= high for unit in units)
    )
    delivery = plan.setdefault("delivery", {})
    if isinstance(delivery, dict):
        # Create an explicit uncovered row for every selected path before
        # recording the current one; a missing row must never look like an
        # untracked complete delivery after a reducer drops that source.
        for declared_path in (
                str(item) for item in plan.get("paths", ()) or () if str(item).strip()
        ):
            if declared_path in delivery:
                continue
            selected = [item for item in plan.get("units", ())
                        if isinstance(item, Mapping) and item.get("path") == declared_path]
            handles = [str(item.get("handle", item.get("unit_id", index)))
                       for index, item in enumerate(selected)]
            delivery[declared_path] = {
                "selected_units": len(selected), "delivered_units": 0,
                "omitted_units": len(selected), "omitted_indexes": list(range(len(selected))),
                "selected_unit_handles": handles, "delivered_unit_handles": [],
                "omitted_unit_handles": handles, "acquired_spans": 0,
                "delivered_spans": 0, "ranges": [],
                "status": _empty_delivery_status(plan, declared_path),
            }
        acquired = len(ranges) if acquired_spans is None else max(0, int(acquired_spans))
        revision_map = plan.get("revisions", {})
        revision = (
            str(revision_map.get(path, "UNKNOWN"))
            if isinstance(revision_map, Mapping) else "UNKNOWN"
        )
        authorization = plan.get("workspace_authorization", {})
        authorized = (
            isinstance(authorization, Mapping)
            and bool(str(authorization.get("scope", "")).strip())
            and bool(str(authorization.get("generation", "")).strip())
        )
        retrieval_route = None
        if omitted and units:
            omitted_ranges = _canonical_line_ranges(tuple(
                (int(units[index].get("start_line", 0)),
                 int(units[index].get("end_line", 0)))
                for index in omitted
            ))
            retrieval_route = {
                "status": (
                    "EXECUTABLE_BOUNDED_READ"
                    if authorized and revision not in {"", "UNKNOWN"} else "INFEASIBLE_NO_SEND"
                ),
                "tool": "read",
                "path": path,
                "revision": revision,
                "unit_handles": [unit_handles[index] for index in omitted],
                "ranges": [list(item) for item in omitted_ranges],
            }
        delivery[path] = {"selected_units": len(units), "delivered_units": len(covered),
                           "omitted_units": len(omitted), "omitted_indexes": omitted,
                           "selected_unit_handles": [unit_handles[index] for index in range(len(units))],
                           "delivered_unit_handles": [unit_handles[index] for index in covered],
                           "omitted_unit_handles": [unit_handles[index] for index in omitted],
                           "acquired_spans": acquired, "delivered_spans": len(covered_ranges),
                           "ranges": [list(item) for item in ranges],
                           "retrieval_route": retrieval_route,
                           "status": (_empty_delivery_status(plan, path) if not units else
                                      "COMPLETE" if not omitted else "PARTIAL")}
    coverage = plan.get("coverage")
    if isinstance(coverage, dict) and isinstance(coverage.get("delivered_span"), dict):
        delivered = sum(int(row.get("delivered_units", 0)) for row in delivery.values() if isinstance(row, Mapping))
        declared_paths = {
            str(item) for item in plan.get("paths", ()) or () if str(item).strip()
        }
        if not declared_paths:
            declared_paths = set(delivery)
        all_complete = bool(declared_paths) and all(
            isinstance(delivery.get(path), Mapping)
            and delivery[path].get("status") == "COMPLETE"
            for path in declared_paths
        )
        acquired = sum(int(row.get("acquired_spans", 0)) for row in delivery.values()
                       if isinstance(row, Mapping))
        coverage["delivered_span"].update({
            "status": "COMPLETE" if all_complete else "PARTIAL" if (delivery or declared_paths) else "UNKNOWN",
            "delivered": delivered,
        })
        if isinstance(coverage.get("acquired_span"), dict):
            coverage["acquired_span"].update({
                "status": "COMPLETE" if acquired else "UNKNOWN", "acquired": acquired,
            })
    acquired = len(ranges) if acquired_spans is None else max(0, int(acquired_spans))
    return {"path": path, "selected_units": len(units), "delivered_units": len(covered),
            "omitted_units": len(omitted), "acquired_spans": acquired,
            "delivered_spans": len(covered_ranges),
            "selected_unit_handles": [unit_handles[index] for index in range(len(units))],
            "delivered_unit_handles": [unit_handles[index] for index in covered],
            "omitted_unit_handles": [unit_handles[index] for index in omitted],
            "retrieval_route": (
                delivery.get(path, {}).get("retrieval_route")
                if isinstance(delivery.get(path), Mapping) else None
            ),
            "status": (_empty_delivery_status(plan, path) if not units else
                       "COMPLETE" if not omitted else "PARTIAL")}


def record_final_source_delivery(
        plan: Mapping[str, object],
        delivered_ranges: Mapping[str, Sequence[tuple[int, int]]],
        *,
        payload_revision: str = "",
        context_digest: str = "",
        submitted_http_bytes_digest: str = "",
        wire_proof_verified: bool | None = None,
        wire_failure_reason: str = "",
) -> dict[str, object]:
    """Reconcile the source floor against ranges surviving final serialization."""

    previous = plan.get("delivery", {})
    previous = previous if isinstance(previous, Mapping) else {}
    delivery: dict[str, object] = {}
    if isinstance(plan, dict):
        plan["delivery"] = delivery
        plan["final_delivery"] = {}
    paths = [str(item) for item in plan.get("paths", ()) or ()]
    results: dict[str, dict[str, object]] = {}
    for path in paths:
        ranges = delivered_ranges.get(path, ())
        old = previous.get(path, {}) if isinstance(previous, Mapping) else {}
        acquired = int(old.get("acquired_spans", 0)) if isinstance(old, Mapping) else 0
        result = record_source_delivery(plan, path, tuple(ranges), acquired_spans=acquired,
                                        single_carrier=True)
        results[path] = result
    selected_results = {
        path: item for path, item in results.items()
        if int(item.get("selected_units", 0)) > 0
        or item.get("status") not in {"NOT_SELECTED", "UNKNOWN"}
    }
    total_units = sum(int(item.get("selected_units", 0)) for item in selected_results.values())
    complete = bool(selected_results) and total_units > 0 and all(
        item.get("status") == "COMPLETE" for item in selected_results.values()
    )
    selected_paths = set(selected_results)
    def relevant_omission(row: object) -> bool:
        if not isinstance(row, Mapping):
            return True
        path = str(row.get("path", "")).strip()
        # A declared dependency with no selected source unit is an explicit
        # NOT_SELECTED fact, not a failed delivery of the required source.
        if not path:
            return True
        if path in selected_paths:
            return True
        return str(row.get("status", "")).upper() != "NOT_SELECTED"

    plan_has_omissions = any(
        relevant_omission(row)
        for key in ("omissions", "skipped_files")
        for row in (plan.get(key, ()) if isinstance(plan.get(key, ()), Sequence)
                    and not isinstance(plan.get(key, ()), (str, bytes, bytearray)) else ())
    )
    status = ("COMPLETE" if complete and not plan_has_omissions
              else "PARTIAL" if total_units or plan_has_omissions else "UNKNOWN")
    if wire_proof_verified is False:
        status = "PARTIAL" if total_units or plan_has_omissions else "UNKNOWN"
        for path, item in results.items():
            if item.get("status") == "COMPLETE":
                item["status"] = "PARTIAL"
            item["proof_status"] = "FAILED"
            if wire_failure_reason:
                item["wire_failure_reason"] = wire_failure_reason
            path_delivery = delivery.get(path)
            if isinstance(path_delivery, dict):
                if path_delivery.get("status") == "COMPLETE":
                    path_delivery["status"] = "PARTIAL"
                path_delivery["proof_status"] = "FAILED"
                if wire_failure_reason:
                    path_delivery["wire_failure_reason"] = wire_failure_reason
                if path_delivery.get("retrieval_route") is None and int(
                        item.get("selected_units", 0)) > 0:
                    authorization = plan.get("workspace_authorization", {})
                    authorized = (
                        isinstance(authorization, Mapping)
                        and bool(str(authorization.get("scope", "")).strip())
                        and bool(str(authorization.get("generation", "")).strip())
                    )
                    revisions = plan.get("revisions", {})
                    revision = (
                        str(revisions.get(path, "UNKNOWN"))
                        if isinstance(revisions, Mapping) else "UNKNOWN"
                    )
                    path_delivery["retrieval_route"] = {
                        "status": (
                            "EXECUTABLE_BOUNDED_READ"
                            if authorized and revision not in {"", "UNKNOWN"}
                            else "INFEASIBLE_NO_SEND"
                        ),
                        "tool": "read",
                        "path": path,
                        "revision": revision,
                        "unit_handles": item.get("selected_unit_handles", []),
                    }
                    item["retrieval_route"] = path_delivery["retrieval_route"]
    elif wire_proof_verified is True:
        for path, item in results.items():
            item["proof_status"] = "VERIFIED"
            path_delivery = delivery.get(path)
            if isinstance(path_delivery, dict):
                path_delivery["proof_status"] = "VERIFIED"
        if isinstance(plan, dict):
            coverage = plan.get("coverage")
            if isinstance(coverage, dict):
                delivered_span = coverage.get("delivered_span")
                if isinstance(delivered_span, dict):
                    delivered_span["status"] = status
                    delivered_span["proof_status"] = "VERIFIED"
                    if wire_failure_reason:
                        delivered_span["wire_failure_reason"] = wire_failure_reason
    if wire_proof_verified is None and status == "COMPLETE":
        # Range metadata is a locator, not delivery: without a byte proof over
        # the final serialized payload nothing may be reported COMPLETE.
        status = "PARTIAL"
        for path, item in results.items():
            if item.get("status") == "COMPLETE":
                item["status"] = "PARTIAL"
            item["proof_status"] = "UNVERIFIED"
            path_delivery = delivery.get(path)
            if isinstance(path_delivery, dict) and path_delivery.get("status") == "COMPLETE":
                path_delivery["status"] = "PARTIAL"
                path_delivery["proof_status"] = "UNVERIFIED"
    receipt = {
        "phase": "final_provider_payload",
        "payload_revision": payload_revision or "UNKNOWN",
        "context_digest": context_digest or payload_revision or "UNKNOWN",
        "submitted_http_bytes_digest": submitted_http_bytes_digest or "UNVERIFIED",
        "transport_identity": "VERIFIED" if submitted_http_bytes_digest else "UNVERIFIED",
        "proof_status": (
            "FAILED" if wire_proof_verified is False
            else "VERIFIED" if wire_proof_verified is True else "UNVERIFIED"
        ),
        "status": status,
        "paths": results,
    }
    if wire_failure_reason:
        receipt["wire_failure_reason"] = wire_failure_reason
    if isinstance(plan, dict):
        plan["final_delivery"] = results
        plan["delivery_phase"] = "final_provider_payload"
        plan["final_delivery_status"] = status
        plan["proof_status"] = (
            "FAILED" if wire_proof_verified is False
            else "VERIFIED" if wire_proof_verified is True else "UNVERIFIED"
        )
        coverage = plan.get("coverage")
        if isinstance(coverage, dict) and isinstance(coverage.get("delivered_span"), dict):
            coverage["delivered_span"].update({
                "status": status,
                "delivered": sum(int(item.get("delivered_units", 0)) for item in results.values()),
                "final_payload_revision": payload_revision or "UNKNOWN",
                "proof_status": plan["proof_status"],
            })
    return receipt


def closure_plan_for_runtime(runtime: object, ticket: TicketIdentity | None = None) -> dict[str, object] | None:
    plan = getattr(runtime, "_request_source_closure", None)
    if isinstance(plan, Mapping):
        return plan if isinstance(plan, dict) else dict(plan)
    return None


def units_for_path(plan: Mapping[str, object] | None, path: str) -> tuple[SpanNeed, ...]:
    if not isinstance(plan, Mapping):
        return tuple()
    return tuple(SpanNeed(str(row["path"]), int(row["start_line"]), int(row["end_line"]),
                         str(row.get("role", "statement")), str(row.get("reason", "closure")),
                         int(row.get("priority", 1)), str(row.get("token", "")))
                 for row in plan.get("units", ())
                 if isinstance(row, Mapping) and str(row.get("path")) == path)


def render_closure_receipt(plan: Mapping[str, object]) -> str:
    lines = [f"{CLOSURE_HEADER} | schema={SCHEMA}; ticket_sha256={plan.get('ticket_sha256')}]" ]
    revisions = plan.get("revisions", {})
    dependencies = plan.get("dependency_revisions", {})
    identities = plan.get("identities", {})
    adapters = plan.get("adapters", {})
    delivery = plan.get("delivery", {})
    final_delivery = plan.get("final_delivery", {})
    paths = plan.get("paths", ())
    for path in paths if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)) else ():
        path_text = str(path)
        adapter = adapters.get(path_text, {}) if isinstance(adapters, Mapping) else {}
        delivered = delivery.get(path_text, {}) if isinstance(delivery, Mapping) else {}
        identity = identities.get(path_text, {}) if isinstance(identities, Mapping) else {}
        lines.append(f"- {path_text} revision={revisions.get(path_text, 'UNKNOWN')} "
                     f"dependency_revision={dependencies.get(path_text, 'UNKNOWN')} "
                     f"identity={identity.get('status', 'UNKNOWN')} "
                     f"adapter={adapter.get('language', 'unknown')}:{adapter.get('status', 'UNKNOWN')} "
                     f"parser={adapter.get('parser_capability', 'UNKNOWN')} "
                     f"acquired={delivered.get('acquired_spans', 0)} "
                     f"delivered={delivered.get('delivered_units', 0)}/{delivered.get('selected_units', 0)} "
                     f"final={final_delivery.get(path_text, {}).get('status', 'UNVERIFIED') if isinstance(final_delivery, Mapping) else 'UNVERIFIED'}")
    skipped = plan.get("skipped_files", ())
    if isinstance(skipped, Sequence):
        for item in skipped:
            if isinstance(item, Mapping):
                lines.append(f"skipped: {item.get('path')} status={item.get('status')} "
                             f"reason={item.get('reason')} continuation={item.get('continuation', 'none')}")
    omissions = plan.get("omissions", ())
    for omission in omissions if isinstance(omissions, Sequence) else ():
        if isinstance(omission, Mapping):
            lines.append(f"omission: {omission.get('path')} status={omission.get('status')} "
                         f"reason={omission.get('reason')} continuation={omission.get('continuation', 'none')}")
    coverage = plan.get("coverage", {})
    if isinstance(coverage, Mapping):
        lines.append("coverage: " + "; ".join(
            f"{key}={value.get('status', 'UNKNOWN')}" for key, value in coverage.items()
            if isinstance(value, Mapping)))
    return "\n".join(lines)


def evidence_view_for_span(path: str, source: str, start_line: int, end_line: int,
                           *, dependency_paths: Sequence[str] = ()) -> tuple[SourceSnapshot, EvidenceView]:
    """Project one exact closure span through EC01's immutable source contracts."""

    revision = source_revision(source)
    snapshot = SourceSnapshot.capture(
        source.encode("utf-8"), store_id="store:request", namespace_id="namespace:public",
        repo_id="repo:workspace", worktree_id="worktree:current", source_ref=path,
        source_revision=revision, origin_receipt_id="receipt:request-closure",
    )
    span = snapshot.span(start_line, end_line, anchor_id="anchor:" + hashlib.sha256(
        f"{path}:{start_line}:{end_line}".encode("utf-8")).hexdigest()[:24])
    coverage = EvidenceCoverage("coverage:request-closure", "source_read", CoverageStatus.COMPLETE,
                                f"request:{path}", inspected=1, expected=1)
    dependencies = tuple(EvidenceDependency("dependency:" + hashlib.sha256(item.encode()).hexdigest()[:24],
                                             source_revision(item)) for item in dependency_paths)
    view = EvidenceView.create(
        snapshot, transform_version="request-closure/1", snapshot_seq=0,
        selector=EvidenceSelector(SelectorMode.SPAN, start_line=start_line, end_line=end_line),
        coverage=coverage, segments=(span,), dependency_vector=dependencies,
        availability=Availability.PRESENT, applicability=ApplicabilityState.CURRENT,
        visibility=EvidenceVisibility.PUBLIC, protected_anchor_ids=(span.span_id,),
    )
    return snapshot, view


__all__ = [
    "CLOSURE_HEADER", "SCHEMA", "TICKET_HEADER", "TICKET_SCHEMA", "ConstructionFact",
    "EvidenceOutcome", "EvidenceStatus", "LanguageAdapter",
    "RevisionAwareFactCache", "RevisionBoundFactCache", "SourceClosure", "SpanNeed", "TicketIdentity",
    "bind_ticket_identity", "connected_source_paths", "ensure_ticket_messages",
    "build_source_closure", "closure_plan_for_runtime", "dependency_revision",
    "construction_fact_for_source", "evidence_view_for_span", "final_wire_receipt",
    "final_wire_ticket_receipt", "initialization_fact", "language_adapter_for", "lexical_sites",
    "model_brief_receipt", "record_source_delivery", "render_closure_receipt",
    "render_construction_fact", "render_ticket_brief", "retrieval_ticket_receipt",
    "parser_capability", "record_final_source_delivery", "serialized_request_bytes",
    "snapshot_for_source", "source_closure_ranges", "source_revision",
    "ticket_digest", "ticket_identifiers", "ticket_identity_for_row", "ticket_identity_for_runtime",
    "ticket_identity_from_row", "units_for_path",
]

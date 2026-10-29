"""Opt-in coverage checks for layers explicitly named by a public ticket.

The organ is intentionally conservative. It reads the public ticket text
supplied by the task row, matches named files/symbols against product files,
and compares those files with a host-captured start snapshot. It never turns
an ambiguous name into a guessed target, and it never scans test trees when it
builds a mapping.

Switch: STATEFRAME_LAYER_COVERAGE=off|on (default off).
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import kernel_parse_cache as _parse_cache


MODE_ENV = "STATEFRAME_LAYER_COVERAGE"
SCHEMA = "stateframe-named-layer-coverage/1"
UNKNOWN = "UNKNOWN"
CODE = "NAMED_LAYER_COVERAGE_UNVERIFIED"

_FORBIDDEN_PARTS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".hypothesis", "private", "hidden_tests", "near_miss", "gold", "oracle",
})
_TEST_DIRS = frozenset({"test", "tests", "__tests__", "testdata"})
_FILE_EXTENSIONS = (
    "py", "js", "jsx", "ts", "tsx", "go", "rs", "java", "rb", "php",
    "c", "cc", "cpp", "h", "html", "css", "md", "rst", "toml", "yaml",
    "yml", "json", "ini", "cfg", "sh",
)
_CODE_EXTENSIONS = frozenset({
    "py", "js", "jsx", "ts", "tsx", "go", "rs", "java", "rb", "php",
    "c", "cc", "cpp", "h",
})

# The order is also the public checklist order when names occur at the same
# position (for example, CLI/API).
_LAYER_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CLI", (r"\bCLI\b", r"\bcommand[- ]line(?:\s+interface|\s+entry[- ]point)?\b",
             r"\bconsole\s+interface\b", r"\bterminal\s+interface\b")),
    ("API", (r"\bAPI\b", r"\bHTTP\s+(?:API|endpoint)\b", r"\bREST\s+(?:API|endpoint)\b",
             r"\bweb\s+API\b", r"\bweb\s+service\b", r"\bendpoint\b")),
    ("storage", (r"\bstorage\b", r"\bpersistence\b", r"\bdatabase\b",
                  r"\bdata[- ]access\b", r"\brepository\b")),
    ("config", (r"\bconfig(?:uration)?\b", r"\bsettings?\b", r"\benvironment\s+config")),
    ("docs", (r"\bdocs?\b", r"\bdocumentation\b", r"\bREADME\b", r"\bmanual\b")),
    ("model", (r"\bmodels?\b", r"\bdomain\s+(?:object|model)\b", r"\bentity\b")),
    ("validation", (r"\bvalidation\b", r"\bvalidate\b", r"\binput\s+checking\b")),
    ("service", (r"\bservice\b", r"\bbusiness\s+logic\b")),
    ("parser", (r"\bparser\b", r"\bparsing\b")),
    ("merge", (r"\bmerge\b", r"\bmerging\b")),
    ("worker", (r"\bworker\b", r"\bbackground\s+job\b")),
    # Only the literal layer name.  A bare "HTTP 422" states an expected
    # response, not a named error-code layer (review4 #3: no inferred layers).
    ("error codes", (r"\berror\s+codes?\b",)),
)

_LAYER_ALIASES: dict[str, frozenset[str]] = {
    "CLI": frozenset({"cli", "command", "commands", "console", "terminal"}),
    "API": frozenset({"api", "endpoint", "endpoints", "route", "routes", "router",
                      "http", "handler", "handlers", "view", "views"}),
    "storage": frozenset({"storage", "store", "persistence", "database", "db",
                          "repository", "repo", "dao"}),
    "config": frozenset({"config", "configuration", "settings", "setting", "env",
                         "environment"}),
    "docs": frozenset({"docs", "doc", "documentation", "readme", "manual"}),
    "model": frozenset({"model", "models", "domain", "entity", "entities"}),
    "validation": frozenset({"validation", "validator", "validators", "validate"}),
    "service": frozenset({"service", "services", "business"}),
    "parser": frozenset({"parser", "parsing"}),
    "merge": frozenset({"merge", "merging"}),
    "worker": frozenset({"worker", "workers", "job", "jobs"}),
    "error codes": frozenset({"error", "errors", "status", "codes", "code"}),
}
_EXTENSION_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_.-]+\."
    + "(?:" + "|".join(_FILE_EXTENSIONS) + r")(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"\x60([^\x60\n]+)\x60|'([^'\n]+)'|\"([^\"\n]+)\"")
_CALL_RE = re.compile(r"\b([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)\s*\(")
_STATUS_RE = re.compile(
    r"\bHTTP\s+(?P<http>[45]\d{2})\b|"
    r"\bstatus(?:\s+code)?\s*(?:(?:must|should|shall)\s+)?"
    r"(?:be|return|returns|is|of|=|:)?\s*(?P<status>[45]\d{2})\b|"
    r"\b(?P<response>[45]\d{2})\s+(?:HTTP\s+)?(?:status|response|code)\b",
    re.IGNORECASE,
)
_INTERFACE_RE = re.compile(
    r"\b(?:CLI|API|HTTP\s+endpoint|REST\s+API|web\s+API|endpoint|"
    r"command[- ]line(?:\s+(?:interface|entry[- ]point))?|"
    r"console\s+interface|terminal\s+interface)\b",
    re.IGNORECASE,
)
_MISSING = object()


@dataclasses.dataclass(frozen=True)
class LayerRow:
    layer: str
    files: tuple[str, ...]
    confidence: str
    diff_from_start: bool | None
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        diff_label = (
            "CHECK_INCONCLUSIVE" if self.confidence == "CHECK_INCONCLUSIVE"
            else _diff_label(self.diff_from_start)
        )
        return {
            "layer": self.layer,
            "files": list(self.files),
            "confidence": self.confidence,
            "diff_from_start": self.diff_from_start,
            "diff_label": diff_label,
            "evidence": list(self.evidence),
        }


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _diff_label(value: bool | None) -> str:
    return "yes" if value is True else "no" if value is False else UNKNOWN


def _text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("text", "ticket", "brief", "content", "source"):
            if isinstance(value.get(key), str) and value[key].strip():
                return value[key].strip()
    return ""


def public_ticket(value: object) -> str:
    """Return only direct public ticket text; never dereference a path."""
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, Mapping):
        return ""
    for key in (
        "public_ticket", "ticket_text", "ticket", "public_brief", "brief",
        "_bridge_ordinary_brief", "msg", "symptom", "description", "request",
    ):
        text = _text(value.get(key))
        if text:
            return text
    return ""


def public_test_text(value: object) -> str:
    """Flatten supplied public-test material without reading a test path."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return "\n".join(public_test_text(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "\n".join(public_test_text(item) for item in value)
    return ""


def _safe_relative(value: object) -> str | None:
    raw = str(value or "").replace("\\", "/").strip().strip("\x60'\"")
    if not raw or raw.startswith("/"):
        return None
    path = PurePosixPath(raw)
    if ".." in path.parts or any(part.startswith(".") for part in path.parts):
        return None
    return path.as_posix()


def _is_test_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    suffix = path.suffix.lower().lstrip(".")
    stem = path.stem.lower()
    conventional_test_name = (
        suffix in _CODE_EXTENSIONS
        and (stem.startswith("test_") or stem.endswith(("_test", ".test", ".spec")))
    )
    return (
        any(part in _TEST_DIRS or part.startswith("test_") for part in path.parts[:-1])
        or conventional_test_name
        or path.name == "conftest.py"
    )


def _is_product_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    return not _is_test_path(relative) and not any(part in _FORBIDDEN_PARTS for part in path.parts)


def _file_stem_alias(relative: str) -> str:
    stem = PurePosixPath(relative).stem.lower()
    for marker in (".test", ".spec", "_test"):
        if stem.endswith(marker):
            return stem[:-len(marker)]
    if stem.startswith("test_"):
        return stem[5:]
    return stem


def _relative(root: Path, path: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return None
    return relative if _is_product_path(relative) else None


def product_files(root: Path) -> tuple[Path, ...]:
    root = Path(root).resolve()
    result: list[Path] = []
    if not root.is_dir():
        return ()
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(
            name for name in dirs
            if name not in _FORBIDDEN_PARTS and not name.startswith(".")
            and name not in _TEST_DIRS
        )
        for name in sorted(names):
            path = Path(directory) / name
            relative = _relative(root, path)
            if relative is not None and path.is_file():
                result.append(path)
    return tuple(result)


def snapshot_workspace(root: Path) -> dict[str, str | None]:
    """Capture product-file bytes for start/current comparisons.

    A present key with ``None`` is a read failure.  It is deliberately not the
    same as an absent key, and neither state can be treated as an unchanged
    product file.
    """
    result: dict[str, str | None] = {}
    root = Path(root).resolve()
    for path in product_files(root):
        relative = _relative(root, path)
        if relative is None:
            continue
        try:
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, UnicodeError):
            result[relative] = None
    return result


def _explicit_files(ticket: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _EXTENSION_RE.finditer(ticket):
        value = _safe_relative(match.group(0))
        if value and value not in values:
            values.append(value)
    for match in _QUOTED_RE.finditer(ticket):
        value = _safe_relative(match.group(1) or match.group(2) or match.group(3))
        if value and "." in PurePosixPath(value).name and value not in values:
            if PurePosixPath(value).suffix.lower().lstrip(".") in _FILE_EXTENSIONS:
                values.append(value)
    return tuple(values)


def _explicit_symbols(ticket: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _QUOTED_RE.finditer(ticket):
        token = (match.group(1) or match.group(2) or match.group(3)).strip()
        token = token.rstrip("()")
        if (re.fullmatch(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*", token)
                and PurePosixPath(token).suffix.lower().lstrip(".") not in _FILE_EXTENSIONS):
            values.append(token)
    for match in _CALL_RE.finditer(ticket):
        values.append(match.group(1))
    return tuple(dict.fromkeys(values))


def _layer_hits(ticket: str) -> dict[str, tuple[int, tuple[str, ...]]]:
    found: dict[str, tuple[int, tuple[str, ...]]] = {}
    for layer, patterns in _LAYER_RULES:
        positions: list[int] = []
        for pattern in patterns:
            positions.extend(match.start() for match in re.finditer(pattern, ticket, re.IGNORECASE))
        if positions:
            found[layer] = (min(positions), tuple(patterns))
    for file_name in _explicit_files(ticket):
        stem = _file_stem_alias(file_name)
        for layer, aliases in _LAYER_ALIASES.items():
            if stem in aliases and layer not in found:
                found[layer] = (ticket.lower().find(file_name.lower()), ("filename:" + file_name,))
    # A qualified symbol such as cli.run() explicitly names the interface
    # even when the prose says only "entry point".
    for symbol in _explicit_symbols(ticket):
        parts = tuple(part.lower() for part in symbol.split("."))
        for layer, aliases in _LAYER_ALIASES.items():
            prefix = next((part for part in parts if part in aliases), None)
            if prefix is not None and layer not in found:
                found[layer] = (ticket.lower().find(prefix), ("symbol:" + symbol,))
    return found


def extract_named_layers(ticket: str) -> tuple[str, ...]:
    """Return explicitly named public layers in ticket order."""
    ticket = str(ticket or "")
    found = _layer_hits(ticket)
    labels = dict(found)
    for file_name in _explicit_files(ticket):
        stem = PurePosixPath(file_name).stem.lower()
        if not any(stem in aliases for aliases in _LAYER_ALIASES.values()):
            labels[f"file:{file_name}"] = (ticket.lower().find(file_name.lower()),
                                             ("explicit-file:" + file_name,))
    for symbol in _explicit_symbols(ticket):
        if not any(any(part in aliases for part in symbol.lower().split("."))
                   for aliases in _LAYER_ALIASES.values()):
            labels[f"symbol:{symbol}"] = (ticket.lower().find(symbol.lower()),
                                            ("explicit-symbol:" + symbol,))
    return tuple(layer for layer, _ in sorted(labels.items(), key=lambda item: (item[1][0], item[0])))


def _path_lookup(root: Path, files: Sequence[Path]) -> dict[str, Path]:
    return {
        relative: path
        for path in files
        if (relative := _relative(root, path)) is not None
    }


def _explicit_file_matches(root: Path, lookup: Mapping[str, Path], names: Sequence[str]) -> list[str]:
    del root
    matches: list[str] = []
    for raw in names:
        relative = _safe_relative(raw)
        if not relative:
            continue
        if relative in lookup:
            matches.append(relative)
            continue
        # A qualified path is an assertion about its complete location.  A
        # missing ``missing/cli.py`` must never bind to ``other/cli.py``.
        if len(PurePosixPath(relative).parts) > 1:
            continue
        basename = PurePosixPath(relative).name
        matches.extend(path for path in lookup if PurePosixPath(path).name == basename)
    return list(dict.fromkeys(matches))


def _layer_file_matches(layer: str, lookup: Mapping[str, Path]) -> list[str]:
    aliases = _LAYER_ALIASES.get(layer, frozenset())
    result: list[str] = []
    for relative in lookup:
        path = PurePosixPath(relative)
        stem = path.stem.lower()
        parent_parts = {part.lower() for part in path.parts[:-1]}
        if stem in aliases or bool(parent_parts & aliases):
            result.append(relative)
    return result


def _definition_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b|"
        rf"^\s*{re.escape(name)}\s*[:=]", re.MULTILINE)


def _defines(path: Path, name: str) -> bool | None:
    """True/False when the file was read; None when it could not be read."""
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError):
        return None
    # L112: the verdict is a pure function of (name, text); memoized by content hash.
    return _parse_cache.derived_source(
        ("named_layer_coverage.defines", name), text, "",
        lambda source: bool(_definition_pattern(name).search(source)))


def _symbol_binding(lookup: Mapping[str, Path], symbol: str) -> tuple[list[str], bool]:
    """Bind ONE symbol as a complete (module, leaf) pair.

    Returns ``(files, read_failed)``.  A qualified symbol ``a.b.run`` binds only
    to the product module ``a/b.<ext>`` (or ``<pkg>/a/b.<ext>``) and only when
    that exact file defines ``run``.  Module qualifiers and leaf names are
    never combined across different requested symbols (review6 #3).
    """
    parts = symbol.split(".")
    leaf = parts[-1]
    module = "/".join(parts[:-1])
    files: list[str] = []
    read_failed = False
    for relative, path in lookup.items():
        if module:
            module_path = relative.rsplit(".", 1)[0]
            if module_path != module and not module_path.endswith("/" + module):
                continue
        found = _defines(path, leaf)
        if found is None:
            read_failed = True
        elif found:
            files.append(relative)
    return files, read_failed


def _layer_evidence(layer: str, ticket: str) -> tuple[str, ...]:
    hits = _layer_hits(ticket).get(layer)
    if hits is None and layer.startswith("file:"):
        return ("explicit-file:" + layer[5:],)
    if hits is None and layer.startswith("symbol:"):
        return ("explicit-symbol:" + layer[7:],)
    return tuple(hits[1]) if hits else ()


def _target_alias_match(value: str, aliases: frozenset[str]) -> bool:
    return any(part.lower() in aliases for part in value.split("."))


def _hash_diff(path: str, start: Mapping[str, str | None],
               current: Mapping[str, str | None]) -> tuple[bool | None, str | None]:
    start_value = start.get(path, _MISSING)
    current_value = current.get(path, _MISSING)
    if start_value is _MISSING or current_value is _MISSING:
        return None, "hash-missing"
    if start_value is None or current_value is None:
        return None, "hash-read-failed"
    return current_value != start_value, None


def _bound_row(layer: str, target: str | None, files: Sequence[str], evidence: tuple[str, ...],
               start: Mapping[str, str | None], current: Mapping[str, str | None],
               *, read_failed: bool = False) -> LayerRow:
    """One row per binding target: confident only with exactly one file and two hashes."""
    tag = (*evidence, f"target:{target}") if target else evidence
    unique = list(dict.fromkeys(files))
    # review8 #2: an unreadable candidate may also define the target, so uniqueness is
    # unknown even when another candidate matched.  Any read failure is inconclusive.
    if read_failed:
        return LayerRow(layer, (UNKNOWN,), "CHECK_INCONCLUSIVE", None, (*tag, "binding-read-failed"))
    if len(unique) != 1:
        reason = "unbound" if not unique else "ambiguous"
        return LayerRow(layer, (UNKNOWN,), UNKNOWN, None, (*tag, reason))
    diff, issue = _hash_diff(unique[0], start, current)
    if issue:
        return LayerRow(layer, tuple(unique), "CHECK_INCONCLUSIVE", None, (*tag, issue))
    return LayerRow(layer, tuple(unique), "confident", diff, tag)


def _rows_for_layer(layer: str, ticket: str, root: Path,
                    lookup: Mapping[str, Path], start: Mapping[str, str | None],
                    current: Mapping[str, str | None]) -> tuple[LayerRow, ...]:
    """Every explicit target keeps its own binding result, including UNKNOWN."""
    evidence = _layer_evidence(layer, ticket)
    if layer.startswith("file:"):
        requested = layer[5:]
        return (_bound_row(layer, requested, _explicit_file_matches(root, lookup, (requested,)),
                           evidence, start, current),)
    if layer.startswith("symbol:"):
        requested = layer[7:]
        files, failed = _symbol_binding(lookup, requested)
        return (_bound_row(layer, requested, files, evidence, start, current,
                           read_failed=failed),)
    aliases = _LAYER_ALIASES.get(layer, frozenset())
    # review8 #3: explicit files AND explicit symbols each keep their own row; the
    # broad-layer fallback applies only when neither exists.
    named_files = [item for item in _explicit_files(ticket) if _file_stem_alias(item) in aliases]
    rows = [_bound_row(layer, name, _explicit_file_matches(root, lookup, (name,)),
                       evidence, start, current) for name in named_files]
    symbols = [symbol for symbol in _explicit_symbols(ticket)
               if _target_alias_match(symbol, aliases)]
    for symbol in symbols:
        files, failed = _symbol_binding(lookup, symbol)
        rows.append(_bound_row(layer, symbol, files, evidence, start, current,
                               read_failed=failed))
    if rows:
        return tuple(rows)
    # A broad layer name is confident only when it resolves to one product file.
    return (_bound_row(layer, None, _layer_file_matches(layer, lookup), evidence, start, current),)


# --- HTTP-status probe suggestions ------------------------------------------------------

_CONDITION_RE = re.compile(r"\b(?P<operator>when|if|whenever|unless)\s+(?P<condition>[^,;]+)",
                           re.IGNORECASE)


def _ticket_clauses(ticket: str) -> tuple[str, ...]:
    """Clause boundaries: sentence ends, semicolons and line breaks.  Status,
    interface and condition are only ever bound within ONE clause (review6 #2)."""
    return tuple(part.strip() for part in re.split(r"(?<=[.!?])\s+|;|\n+", ticket)
                 if part.strip())


_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|nor|without|cannot)\b|n't\b|\bmust\s+stay\b|\bunchanged\b",
    re.IGNORECASE)
_UNCERTAIN_SCOPE_RE = re.compile(r"\b(?:unless|except|or|either|otherwise)\b", re.IGNORECASE)
_CONDITION_WORD_RE = re.compile(r"\b(?:when|if|whenever|unless)\b", re.IGNORECASE)


def affirmative_clause(clause: str) -> bool:
    """review8 #1: only an explicitly affirmative, single-condition clause without
    alternatives is actionable.  Negation, ``unless``/``except``, alternatives and more
    than one condition keyword are neutral review only (probes AND completion evidence)."""
    text = str(clause or "")
    return (not _NEGATION_RE.search(text) and not _UNCERTAIN_SCOPE_RE.search(text)
            and len(_CONDITION_WORD_RE.findall(text)) <= 1)


def affirmative_ticket(ticket: str) -> str:
    """The ticket restricted to affirmative clauses: layer/target rows (the evidence the
    done challenge can act on) never come from "Do not change the CLI"."""
    return "\n".join(clause for clause in _ticket_clauses(ticket) if affirmative_clause(clause))


def _status_contexts(ticket: str) -> tuple[dict[str, object], ...]:
    contexts: list[dict[str, object]] = []
    for clause in _ticket_clauses(ticket):
        statuses = tuple(dict.fromkeys(
            next(value for value in match.groups() if value)
            for match in _STATUS_RE.finditer(clause)))
        if not statuses:
            continue
        interfaces = tuple(dict.fromkeys(
            " ".join(match.group(0).split()) for match in _INTERFACE_RE.finditer(clause)))
        condition_match = _CONDITION_RE.search(clause)
        condition = (condition_match.group("condition").strip(" .,:;")
                     if condition_match else None)
        contexts.append({
            "clause": clause.rstrip("."),
            "statuses": statuses,
            "interface": interfaces[0] if len(interfaces) == 1 else None,
            "interfaces": interfaces,
            "condition": condition or None,
            # review8 #1: the operator is preserved and affirmation is checked.
            "operator": (condition_match.group("operator").lower() if condition_match else None),
            "affirmative": affirmative_clause(clause),
        })
    return tuple(contexts)


def _public_test_evidence(statuses: Sequence[str], public_tests: object) -> tuple[dict[str, object], ...]:
    test_text = public_test_text(public_tests)
    result = []
    for status in statuses:
        present = bool(re.search(rf"(?<!\d){re.escape(status)}(?!\d)", test_text))
        result.append({
            "status": status,
            "evidence": "literal_presence" if present else "literal_absence",
            "literal_present": present,
            "execution_inferred": False,
        })
    return tuple(result)


def _probe_suggestions(ticket: str, public_tests: object) -> tuple[str, ...]:
    """One suggestion per ticket clause.  A concrete probe quotes that clause's own
    interface, condition and single status; anything less bound is a neutral review
    note.  Nothing is suggested for a status the public tests already mention."""
    suggestions: list[str] = []
    for context in _status_contexts(ticket):
        statuses = tuple(context["statuses"])
        evidence = _public_test_evidence(statuses, public_tests)
        if all(item["literal_present"] for item in evidence):
            continue
        literal = ", ".join(
            f"literal {item['status']} {'present' if item['literal_present'] else 'absent'}"
            for item in evidence)
        interface = context["interface"]
        condition = context["condition"]
        operator = context.get("operator")
        actionable = (bool(context.get("affirmative")) and operator in {"when", "if", "whenever"}
                      and condition is not None
                      and not re.search(r"\band\b", str(condition), re.IGNORECASE))
        if len(statuses) == 1 and interface and condition and actionable:
            suggestions.append(
                f'probe: exercise interface "{interface}" under condition "{operator} {condition}"; '
                f"expect HTTP {statuses[0]} per ticket clause \"{context['clause']}\"; "
                f"public-test evidence: {literal} (literal presence only; execution not inferred).")
            continue
        missing = []
        if condition and not actionable:
            missing.append("affirmative single condition")
        if not condition:
            missing.append("condition")
        if not interface:
            missing.append("interface")
        if len(statuses) != 1:
            missing.append("single status")
        suggestions.append(
            f"review: ticket clause \"{context['clause']}\" names "
            + " and ".join("HTTP " + status for status in statuses)
            + " without one explicit " + ", ".join(missing)
            + f"; inspect manually. Public-test evidence: {literal} (literal presence only).")
    return tuple(suggestions)


# --- report ----------------------------------------------------------------------------

def _challenge_rows(rows: Sequence[LayerRow]) -> tuple[str, ...]:
    result: list[str] = []
    for row in rows:
        if row.confidence != "confident" or row.diff_from_start is not False:
            continue
        for file_name in row.files:
            result.append(
                f"UNVERIFIED: ticket names {row.layer}; {file_name} is unchanged since start; "
                "check whether it needs the change before submitting.")
    return tuple(result)


def normalise_snapshot(snapshot: Mapping[str, object]) -> dict[str, str | None]:
    """A present key with ``None`` stays a read failure (never the string 'None')."""
    return {
        str(key).replace("\\", "/"): (None if value is None else str(value))
        for key, value in snapshot.items()
    }


def build_report(ticket: object, root: Path, *,
                 start_snapshot: Mapping[str, object] | None = None,
                 current_snapshot: Mapping[str, object] | None = None,
                 public_tests: object = None) -> dict[str, object]:
    text = public_ticket(ticket)
    root = Path(root).resolve()
    start = normalise_snapshot(
        snapshot_workspace(root) if start_snapshot is None else start_snapshot)
    current = normalise_snapshot(
        snapshot_workspace(root) if current_snapshot is None else current_snapshot)
    lookup = _path_lookup(root, product_files(root))
    # review8 #1: rows (the only completion evidence) come from affirmative clauses only.
    scoped = affirmative_ticket(text)
    rows = tuple(row for layer in extract_named_layers(scoped)
                 for row in _rows_for_layer(layer, scoped, root, lookup, start, current))
    challenges = _challenge_rows(rows)
    statuses = tuple(dict.fromkeys(
        status for context in _status_contexts(text) for status in context["statuses"]))
    return {
        "schema": SCHEMA,
        "turn": 1,
        "ticket_digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "rows": [row.to_dict() for row in rows],
        "probe_suggestions": list(_probe_suggestions(text, public_tests)),
        "public_test_evidence": list(_public_test_evidence(statuses, public_tests)),
        "done_challenges": list(challenges),
        "challenge_class": "UNVERIFIED" if challenges else None,
    }


def missing_rows(report: Mapping[str, object] | None) -> tuple[Mapping[str, object], ...]:
    if not isinstance(report, Mapping):
        return ()
    rows = report.get("rows", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return ()
    return tuple(row for row in rows if isinstance(row, Mapping)
                 and row.get("confidence") == "confident"
                 and row.get("diff_from_start") is False)


CHECKLIST_HEADER = "[Named-layer coverage | turn 1]"


def render_checklist(report: Mapping[str, object] | None) -> str:
    """Render the turn-one checklist used by every host-state presentation."""
    if not isinstance(report, Mapping):
        return ""
    lines = [CHECKLIST_HEADER,
             "Layers the public ticket names: layer -> files -> diff-from-start"]
    rows = report.get("rows", ())
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)) and rows:
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            files = ", ".join(str(item) for item in row.get("files", ())) or UNKNOWN
            lines.append(f"- {row.get('layer', UNKNOWN)} -> {files} -> "
                         f"diff-from-start: {row.get('diff_label', _diff_label(row.get('diff_from_start')))}")
    else:
        lines.append("- no explicitly named layers were extracted")
    suggestions = report.get("probe_suggestions", ())
    if isinstance(suggestions, Sequence) and not isinstance(suggestions, (str, bytes, bytearray)):
        lines.extend(str(item) for item in suggestions if str(item).strip())
    return "\n".join(lines)


# --- task-row readers (the same public fields the rest of the runtime uses) ------------

def row_ticket(row: Mapping[str, object], *, root: Path | None = None) -> str:
    """Public ticket text via the requirements-ledger reader (``msg``/``brief`` file)."""
    try:
        from kernel_requirements_ledger import _ticket_from_row
        return str(_ticket_from_row(row, root=root)[1] or "")
    except Exception:  # noqa: BLE001 - unavailable ticket => no rows, never a guess
        return public_ticket(row)


def row_public_tests(row: Mapping[str, object]) -> dict[str, str]:
    """Declared public test sources (kernel_public_evidence); never hidden material."""
    try:
        from kernel_public_evidence import public_test_sources
        return dict(public_test_sources(row))
    except Exception:  # noqa: BLE001 - evidence is optional; absence is literal_absence
        return {}


# --- completion evidence ---------------------------------------------------------------

def _arbiter_active(runtime: object) -> bool:
    if getattr(runtime, "stateframe_done_challenge_once_mode", "off") == "on":
        return True
    try:
        from kernel_first_green_disclosure import enabled
        return bool(enabled(runtime))
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _menu(runtime: object) -> tuple[str, ...]:
    value = getattr(runtime, "completion_menu", None)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item).strip().lower() for item in value if str(item).strip())
    return ()


def _emit(runtime: object, kind: str, **fields: object) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit(kind, producer="kernel_named_layer_coverage", dispatch=False, **fields)


def _reopen_for_named_layer(runtime: object, rows: Sequence[Mapping[str, object]]) -> bool:
    """STATEFRAME_NAMED_LAYER_REPAIR rule D (autopsy-ldev01-cross-release #3): a done-only menu with a
    confident untouched layer reopens the post-green menu once per revision (the existing reopen latch)
    instead of waving the sole done through.  False when the switch is off or the reopen is refused."""
    try:
        from kernel_named_layer_repair import enabled as named_layer_repair_enabled
        if not named_layer_repair_enabled():
            return False
        from kernel_post_green_menu import reopen_instead_of_sole_submission
        lock = getattr(runtime, "green_lock", None)
        ids = [f"{row.get('layer')}:{','.join(str(item) for item in row.get('files', ()))}" for row in rows]
        return bool(reopen_instead_of_sole_submission(
            runtime, getattr(lock, "_green_revision", None), ids, named_layer=True))
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def precommit_assessment(runtime: object, *, admissible: bool,
                         detail: str = "",
                         reason_codes: Sequence[str] = ()) -> dict[str, object] | None:
    """Evidence for the existing done-challenge, computed BEFORE the committing reducer.

    Returns ``None`` (no effect on completion) unless ALL hold:
      * the switch is on and the caller itself judged the done admissible;
      * an arbiter owns the challenge (challenge-once or first-green disclosure) --
        coverage never becomes a veto of its own;
      * the offered menu has more than ``done`` and includes ``read`` -- a sole
        ``done`` is always submittable and the refusal carries an executable repair;
      * a row is CONFIDENT and unchanged.  UNKNOWN / CHECK_INCONCLUSIVE never refuse.
    """
    if getattr(runtime, "stateframe_layer_coverage_mode", "off") != "on" or not admissible:
        return None
    provider = getattr(runtime, "layer_coverage_report", None)
    report = provider() if callable(provider) else getattr(runtime, "_layer_coverage_report", None)
    rows = missing_rows(report)
    if not rows:
        return None
    menu = _menu(runtime)
    ids = [f"{row.get('layer')}:{','.join(str(item) for item in row.get('files', ()))}"
           for row in rows]
    reason = None
    if not _arbiter_active(runtime):
        reason = "no_challenge_arbiter"
    elif not menu or set(menu) <= {"done"}:
        reason = "sole_done_menu"
    elif "read" not in menu:
        reason = "no_read_on_menu"
    reopened = False
    if reason in {"sole_done_menu", "no_read_on_menu"}:
        reopened = _reopen_for_named_layer(runtime, rows)
        if reopened:
            reason = None
    if reason is not None:
        _emit(runtime, "named_layer_coverage_evidence", enforced=False, reason=reason,
              rows=ids, menu=list(menu))
        return None
    challenge = " ".join(str(item) for item in report.get("done_challenges", ())
                         if str(item).strip())
    first_file = str(rows[0].get("files", [UNKNOWN])[0])
    _emit(runtime, "named_layer_coverage_evidence", enforced=True, reason=CODE,
          rows=ids, menu=list(menu), **({"reopened": True} if reopened else {}))
    return {
        "reason_codes": tuple(dict.fromkeys((*reason_codes, CODE))),
        "detail": (detail + "\n" if detail else "") + "[named-layer coverage] " + challenge,
        "repair_action": {"tool": "read", "arguments": {"path": first_file},
                          "reason": "inspect the named layer the ticket requires"},
        "rows": rows,
    }


__all__ = [
    "CHECKLIST_HEADER", "CODE", "MODE_ENV", "SCHEMA", "UNKNOWN", "build_report", "enabled",
    "extract_named_layers", "missing_rows", "mode", "normalise_snapshot", "product_files",
    "precommit_assessment", "public_test_text", "public_ticket", "render_checklist",
    "row_public_tests", "row_ticket", "snapshot_workspace",
]

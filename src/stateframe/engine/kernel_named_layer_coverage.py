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
import stat
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import kernel_parse_cache as _parse_cache

if TYPE_CHECKING:
    from kernel_workspace_state import WorkspaceState
    from kernel_workspace_tree import Directory


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
_ROW_POLICY_VERSION = 1


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


def _is_test_path(path: PurePosixPath) -> bool:
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


def _is_product_path(path: PurePosixPath) -> bool:
    return not _is_test_path(path) and not any(part in _FORBIDDEN_PARTS for part in path.parts)


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
    return relative if _is_product_path(PurePosixPath(relative)) else None


def _file_lookup(root: Path) -> _FileLookup:
    root = Path(root).resolve()
    projection = _active_projection(root)
    if projection is not None:
        return projection.files
    scoped = _COVERAGE_FILES.get()
    if scoped is not None and scoped.root == root:
        return scoped.files
    return CoverageFileCache().lookup(root)


def product_files(root: Path) -> Iterable[Path]:
    files = _file_lookup(root)
    return files.values() if files.paths is None else files.paths


def snapshot_workspace(root: Path) -> dict[str, str | None]:
    """Capture product-file bytes for start/current comparisons.

    A present key with ``None`` is a read failure.  It is deliberately not the
    same as an absent key, and neither state can be treated as an unchanged
    product file.
    """
    return _snapshot_files(_file_lookup(root))


def _snapshot_files(files: Mapping[str, Path]) -> dict[str, str | None]:
    observation = _file_observation(files)
    if observation is not None and _reuse_current_rows():
        return {relative: observation.read(path).digest for relative, path in files.items()}
    result: dict[str, str | None] = {}
    for relative, path in files.items():
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
        if isinstance(lookup, _FileLookup):
            matches.extend(lookup.basenames.get(basename, ()))
        else:
            matches.extend(path for path in lookup if PurePosixPath(path).name == basename)
    return list(dict.fromkeys(matches))


def _layer_file_matches(layer: str, lookup: Mapping[str, Path]) -> list[str]:
    aliases = _LAYER_ALIASES.get(layer, frozenset())
    if isinstance(lookup, _FileLookup):
        return list({path for alias in aliases for path in lookup.aliases.get(alias, ())})
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


def _defines(path: Path, name: str, observation: _FileObservation | None = None) -> bool | None:
    """True/False when the file was read; None when it could not be read."""
    if observation is None:
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError):
            return None
    else:
        text = observation.read(path).text
        if text is None:
            return None
        if name not in text:
            return False
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
    if isinstance(lookup, _ProjectedFiles):
        return lookup.projection.symbol_binding(module, leaf)
    observation = _file_observation(lookup)
    indexed = observation is not None and observation.reuse_rows
    candidates = observation.candidates(module) if indexed else lookup.items()
    files: list[str] = []
    read_failed = False
    for relative, path in candidates:
        if not indexed and module and not _in_module(relative, module):
            continue
        found = _defines(path, leaf, observation) if indexed else _defines(path, leaf)
        if found is None:
            read_failed = True
        elif found:
            files.append(relative)
    return files, read_failed


def _in_module(relative: str, module: str) -> bool:
    module_path = relative.rsplit(".", 1)[0]
    return module_path == module or module_path.endswith("/" + module)


def _reuse_current_rows() -> bool:
    try:
        from stateframe import _startup_optimization_enabled as enabled
    except ModuleNotFoundError as error:
        if error.name != "stateframe":
            raise
        return False
    return enabled()


def _row_policy() -> tuple[object, ...]:
    return (
        _ROW_POLICY_VERSION, _LAYER_RULES, tuple(_LAYER_ALIASES.items()), _FILE_EXTENSIONS,
        _EXTENSION_RE, _QUOTED_RE, _CALL_RE,
        _NEGATION_RE, _UNCERTAIN_SCOPE_RE, _CONDITION_WORD_RE,
    )


# A declaration can span lines in the legacy matcher. Look ahead so that a
# second declaration beginning on a consumed line still gets its own match.
_DECLARATION = re.compile(
    r"^(?=[^\S\n]*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)\b)", re.MULTILINE)
_ASSIGNMENT = re.compile(r"^(?=[^\S\n]*([A-Za-z_]\w*)\s*[:=])", re.MULTILINE)


def _blob_definitions(data: bytes) -> frozenset[str] | None:
    """Collect the same textual definitions as the per-name legacy matcher."""
    try:
        source = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError:
        return None
    return frozenset(match.group(1) for pattern in (_DECLARATION, _ASSIGNMENT)
                     for match in pattern.finditer(source))


def _set_posting(postings: dict[str, set[str]], key: str, path: str, present: bool) -> None:
    if present:
        postings.setdefault(key, set()).add(path)
    elif key in postings:
        postings[key].discard(path)
        if not postings[key]:
            del postings[key]


class _DefinitionScope:
    def __init__(self) -> None:
        self.names: dict[str, set[str]] = {}
        self.unreadable: set[str] = set()

    def update(self, path: str, definitions: frozenset[str] | None, present: bool) -> None:
        if definitions is None:
            (self.unreadable.add if present else self.unreadable.discard)(path)
        else:
            for name in definitions:
                _set_posting(self.names, name, path, present)


class _FileLookup(dict[str, Path]):
    def __init__(self) -> None:
        super().__init__()
        self.paths: tuple[Path, ...] | None = None
        self.basenames: dict[str, set[str]] = {}
        self.aliases: dict[str, set[str]] = {}

    def update_postings(self, relative: str, path: PurePosixPath, *, present: bool) -> None:
        aliases = {path.stem.lower(), *(part.lower() for part in path.parts[:-1])}
        _set_posting(self.basenames, path.name, relative, present)
        for alias in aliases.intersection(_ALL_LAYER_ALIASES):
            _set_posting(self.aliases, alias, relative, present)


class _ProjectedFiles(_FileLookup):
    def __init__(self, projection: CoverageProjection) -> None:
        super().__init__()
        self.projection = projection
        self.modules: dict[str, set[str]] = {}


def _directory_stamp(path: Path) -> tuple[int, int, int, int] | None:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError:
        return None
    if stat.S_ISDIR(info.st_mode):
        return info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns
    return None


class CoverageFileCache:
    """Reuse admitted path membership and the last verified set of report rows."""

    _report: tuple[_FileLookup, tuple[object, ...], tuple[LayerRow, ...]] | None = None

    def __init__(self) -> None:
        self._saved: tuple[Path, _FileLookup, dict[Path, tuple[int, int, int, int]]] | None = None

    def __getstate__(self) -> dict[str, object]:
        return {"_saved": None}

    def _membership_current(self, root: Path, files: _FileLookup) -> bool:
        if self._saved is None:
            return False
        saved_root, saved_files, stamps = self._saved
        return (root == saved_root and files is saved_files
                and all(_directory_stamp(path) == stamp for path, stamp in stamps.items()))

    def lookup(self, root: Path) -> _FileLookup:
        if self._saved is not None and self._membership_current(root, self._saved[1]):
            return self._saved[1]
        files, stamps = self._scan(root)
        self._saved = (root, files, stamps) if stamps else None
        if self._report is not None:
            self._report = None
        return files

    def report_rows(self, observation: _FileObservation, ticket: str, scoped: str,
                    layers: tuple[str, ...], start: dict[str, str | None],
                    current: dict[str, str | None]) -> tuple[LayerRow, ...]:
        files = observation.files
        # A supplied snapshot is a diff input. Only bytes read through this
        # observation can prove a symbol lookup, including failed reads.
        modules = dict.fromkeys(
            symbol.rpartition(".")[0].replace(".", "/")
            for symbol in _explicit_symbols(scoped)
        )
        scopes = ("",) if "" in modules else modules
        sources = tuple(
            (relative, observation.read(path).digest)
            for module in scopes
            for relative, path in observation.candidates(module)
        )
        key = (_row_policy(), ticket, scoped, layers, start, current, sources)
        cached = self._report
        rows = (cached[2] if cached is not None and cached[0] is files and cached[1] == key
                else tuple(row for layer in layers for row in
                           _compute_layer_rows(layer, scoped, observation.root, files, start, current)))
        if not self._membership_current(observation.root, files) or not observation.sources_current():
            self._report = None
            fresh, _ = self._scan(observation.root)
            return tuple(row for layer in layers for row in
                         _compute_layer_rows(layer, scoped, observation.root, fresh, start, current))
        self._report = (files, key, rows)
        return rows

    @staticmethod
    def _scan(root: Path) -> tuple[_FileLookup, dict[Path, tuple[int, int, int, int]]]:
        files = _FileLookup()
        paths: list[Path] = []
        stamps = {}
        reusable = True
        pending = [(root, "")]
        while pending:
            directory, prefix = pending.pop()
            stamp = _directory_stamp(directory)
            if stamp is None:
                reusable = False
                continue
            try:
                with os.scandir(directory) as entries:
                    ordered = sorted(entries, key=lambda entry: entry.name)
            except OSError:
                reusable = False
                continue
            stamps[directory] = stamp
            subdirectories = []
            for entry in ordered:
                name = entry.name
                relative = prefix + name
                if entry.is_dir(follow_symlinks=False):
                    if (name not in _FORBIDDEN_PARTS and name not in _TEST_DIRS
                            and not name.startswith(".")):
                        subdirectories.append((directory / name, relative + "/"))
                    continue
                if entry.is_symlink():
                    # A link can change meaning without changing the scanned directories.
                    reusable = False
                    relative = _relative(root, directory / name)
                    if relative is None or not entry.is_file():
                        continue
                elif not entry.is_file(follow_symlinks=False):
                    continue
                relative_path = PurePosixPath(relative)
                if not _is_product_path(relative_path):
                    continue
                path = directory / name
                paths.append(path)
                files[relative] = path
                files.update_postings(relative, relative_path, present=True)
            pending.extend(reversed(subdirectories))
        files.paths = tuple(paths)
        return files, stamps if reusable else {}


class _ObservedFile:
    def __init__(self, path: Path) -> None:
        self.stamp = self.file_stamp(path)
        try:
            self._data = path.read_bytes()
        except (OSError, UnicodeError):
            self._data = None
        self.digest = hashlib.sha256(self._data).hexdigest() if self._data is not None else None

    @staticmethod
    def file_stamp(path: Path) -> tuple[int, ...] | None:
        try:
            info = path.stat()
        except OSError:
            return None
        return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns)

    @cached_property
    def text(self) -> str | None:
        data, self._data = self._data, None
        if data is None:
            return None
        try:
            return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeError:
            return None


class _FileObservation:
    reuse_rows = False
    """Share source reads for the lifetime of one coverage operation."""

    def __init__(self, root: Path, file_cache: CoverageFileCache) -> None:
        self.root = root
        self._file_cache = file_cache
        self._sources: dict[Path, _ObservedFile] = {}

    @cached_property
    def _candidates(self) -> dict[str, tuple[tuple[str, Path], ...]]:
        return {}

    @cached_property
    def files(self) -> _FileLookup:
        return self._file_cache.lookup(self.root)

    def read(self, path: Path) -> _ObservedFile:
        if path not in self._sources:
            self._sources[path] = _ObservedFile(path)
        return self._sources[path]

    def sources_current(self) -> bool:
        return all(source.stamp is not None and source.file_stamp(path) == source.stamp
                   for path, source in self._sources.items())

    def candidates(self, module: str) -> Iterable[tuple[str, Path]]:
        if not module:
            return self.files.items()
        candidates = self._candidates.get(module)
        if candidates is None:
            candidates = tuple((relative, path) for relative, path in self.files.items()
                               if _in_module(relative, module))
            self._candidates[module] = candidates
        return candidates

    def report_rows(self, ticket: str, scoped: str, layers: tuple[str, ...],
                    start: dict[str, str | None],
                    current: dict[str, str | None]) -> tuple[LayerRow, ...]:
        self.reuse_rows = True
        return self._file_cache.report_rows(self, ticket, scoped, layers, start, current)


class CoverageProjection:
    """Coverage facts derived from successive pins of one protected workspace."""

    def __init__(self, root: Path, state: WorkspaceState) -> None:
        self.root = root
        self.state = state
        self._tree: Directory | None = None
        self._links: set[str] = set()
        self._unsupported: set[str] = set()
        self.hashes: dict[str, str | None] = {}
        self.files = _ProjectedFiles(self)
        self.revision = object()
        self._definitions: dict[str, frozenset[str] | None] = {}
        self._blob_users: dict[str, int] = {}
        self._scopes: dict[str, _DefinitionScope] = {}
        self._starts: dict[int, tuple[Mapping[str, object], dict[str, str | None]]] = {}
        self._rows: dict[tuple[int, str, str],
                         tuple[Mapping[str, str | None], tuple[LayerRow, ...]]] = {}

    def _read_definitions(self, relative: str, digest: str) -> frozenset[str] | None:
        from kernel_workspace_state import WorkspaceUnavailable

        if digest not in self._definitions:
            try:
                data = self.files[relative].read_bytes()
            except OSError as error:
                raise WorkspaceUnavailable("coverage source is no longer readable") from error
            if hashlib.sha256(data).hexdigest() != digest:
                raise WorkspaceUnavailable("coverage source changed during observation")
            self._definitions[digest] = _blob_definitions(data)
        return self._definitions[digest]

    def symbol_binding(self, module: str, leaf: str) -> tuple[list[str], bool]:
        if module not in self._scopes:
            scope = _DefinitionScope()
            candidates = self.files.modules.get(module, ()) if module else self.files
            for relative in candidates:
                digest = self.hashes[relative]
                assert digest is not None
                scope.update(relative, self._read_definitions(relative, digest), True)
            self._scopes[module] = scope
        scope = self._scopes[module]
        return list(scope.names.get(leaf, ())), bool(scope.unreadable)

    def _update_file(self, relative: str, digest: str, *, present: bool) -> None:
        module = relative.rsplit(".", 1)[0]
        modules = [module]
        while "/" in module:
            module = module.split("/", 1)[1]
            modules.append(module)
        if present:
            self.files[relative] = self.root / relative
            self.hashes[relative] = digest
            self._blob_users[digest] = self._blob_users.get(digest, 0) + 1
        self.files.update_postings(relative, PurePosixPath(relative), present=present)
        for module in modules:
            _set_posting(self.files.modules, module, relative, present)
        for module in ("", *modules):
            scope = self._scopes.get(module)
            if scope is not None:
                scope.update(relative, self._read_definitions(relative, digest), present)
        if not present:
            del self.files[relative]
            del self.hashes[relative]
            self._blob_users[digest] -= 1
            if not self._blob_users[digest]:
                del self._blob_users[digest]
                self._definitions.pop(digest, None)

    def refresh(self, tree: Directory) -> bool:
        from kernel_workspace_tree import File, Symlink, diff, directory

        if self._tree is not None and self._tree.digest == tree.digest:
            return self._supports_legacy_scope()
        previous = self._tree if self._tree is not None else directory(tree.mode, ())
        changes = []
        for change in diff(previous, tree):
            parts = tuple(map(os.fsdecode, change.path))
            if not parts or any(part.startswith(".") or part in _FORBIDDEN_PARTS
                                or part in _TEST_DIRS for part in parts[:-1]):
                continue
            relative = "/".join(parts)
            if isinstance(change.after, Symlink):
                self._links.add(relative)
            else:
                self._links.discard(relative)
            if isinstance(change.after, File) and "\\" in relative:
                self._unsupported.add(relative)
            else:
                self._unsupported.discard(relative)
            old = self.hashes.get(relative)
            new = (change.after.sha256.hex() if isinstance(change.after, File)
                   and _is_product_path(PurePosixPath(relative)) else None)
            if old != new:
                changes.append((relative, old, new))
        # Remove every old posting first. A blob moved between paths can then
        # retain its definitions without reading through any old path.
        if changes:
            moved = {new: self._definitions[new] for _, _, new in changes
                     if new is not None and new in self._definitions}
            for relative, old, _ in changes:
                if old is not None:
                    self._update_file(relative, old, present=False)
            self._definitions.update(moved)
            for relative, _, new in changes:
                if new is not None:
                    self._update_file(relative, new, present=True)
            self.revision = object()
            self._rows.clear()
        self._tree = tree
        return self._supports_legacy_scope()

    def _supports_legacy_scope(self) -> bool:
        if self._unsupported:
            return False
        # Legacy ignores escaped links and does not descend through directory
        # aliases. An in-tree file alias can expose a path outside our scope.
        for relative in self._links:
            try:
                target = (self.root / relative).resolve()
                if target.is_relative_to(self.root) and target.is_file():
                    return False
            except (OSError, RuntimeError):
                return False
        return True

    def initial_snapshot(self) -> dict[str, str | None]:
        # Match the legacy sorted os.walk order for the two journal payloads.
        ordered = sorted(self.hashes, key=lambda path: (path.split("/")[:-1], path))
        return {path: self.hashes[path] for path in ordered}

    def normalized_start(self, snapshot: Mapping[str, object]) -> dict[str, str | None]:
        key = id(snapshot)
        if key not in self._starts:
            if len(self._starts) >= 2:
                self._starts.clear()
            self._starts[key] = snapshot, normalise_snapshot(snapshot)
        return self._starts[key][1]

    def rows(self, layer: str, ticket: str,
             start: Mapping[str, str | None]) -> tuple[LayerRow, ...]:
        key = (id(start), layer, ticket)
        cached = self._rows.get(key)
        if cached is None:
            rows = _compute_layer_rows(layer, ticket, self.root, self.files, start, self.hashes)
            if len(self._rows) >= 256:
                self._rows.clear()
            self._rows[key] = start, rows
            return rows
        return cached[1]


_ALL_LAYER_ALIASES = frozenset().union(*_LAYER_ALIASES.values())
_COVERAGE_PROJECTION: ContextVar[CoverageProjection | None] = ContextVar(
    "coverage_workspace_projection", default=None)
_COVERAGE_FILES: ContextVar[_FileObservation | None] = ContextVar(
    "coverage_file_lookup", default=None)


def _file_observation(files: Mapping[str, Path]) -> _FileObservation | None:
    observation = _COVERAGE_FILES.get()
    return observation if observation is not None and observation.files is files else None


@contextmanager
def use_file_cache(root: Path, file_cache: CoverageFileCache | None = None) -> Iterator[None]:
    """Share path membership and source reads within a coverage operation."""
    observation = _COVERAGE_FILES.get()
    if file_cache is None and observation is not None and observation.root == root:
        yield
        return
    token = _COVERAGE_FILES.set(_FileObservation(root, file_cache or CoverageFileCache()))
    try:
        yield
    finally:
        _COVERAGE_FILES.reset(token)


def _active_projection(root: Path) -> CoverageProjection | None:
    projection = _COVERAGE_PROJECTION.get()
    return projection if projection is not None and root == projection.root else None


def projection_for_snapshot(root: Path, snapshot: object) -> CoverageProjection | None:
    projection = _active_projection(root)
    return projection if projection is not None and snapshot is projection.hashes else None


@contextmanager
def use_projection(projection: CoverageProjection) -> Iterator[None]:
    """Keep named-layer and reminder lookups on the same pinned workspace."""
    token = _COVERAGE_PROJECTION.set(projection)
    try:
        yield
    finally:
        _COVERAGE_PROJECTION.reset(token)


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
    if isinstance(lookup, _ProjectedFiles) and current is lookup.projection.hashes:
        return lookup.projection.rows(layer, ticket, start)
    return _compute_layer_rows(layer, ticket, root, lookup, start, current)


def _compute_layer_rows(layer: str, ticket: str, root: Path,
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
    projection = projection_for_snapshot(root, current_snapshot)
    reuse_rows = (projection is None and start_snapshot is not None
                  and current_snapshot is not None and _reuse_current_rows())
    with use_file_cache(root) if reuse_rows else nullcontext():
        if projection is not None and start_snapshot is not None:
            start = projection.normalized_start(start_snapshot)
            current = projection.hashes
            lookup = projection.files
        else:
            lookup = _file_lookup(root)
            start = normalise_snapshot(
                _snapshot_files(lookup) if start_snapshot is None else start_snapshot)
            current = normalise_snapshot(
                _snapshot_files(lookup) if current_snapshot is None else current_snapshot)
        # Rows (the only completion evidence) come from affirmative clauses only.
        scoped = affirmative_ticket(text)
        layers = extract_named_layers(scoped)
        observation = _file_observation(lookup) if reuse_rows else None
        if observation is not None:
            rows = observation.report_rows(text, scoped, layers, start, current)
        else:
            rows = tuple(row for layer in layers
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

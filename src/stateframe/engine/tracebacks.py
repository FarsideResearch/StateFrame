"""Parse traceback evidence into source locations and failure observations.

Tracebacks identify where execution failed; they do not by themselves prove
which change caused the failure.
"""

from __future__ import annotations

import ast
import hashlib
import posixpath
import re
from dataclasses import dataclass, replace
from typing import Optional

SCHEMA_VERSION = "traceback-evidence/1"

# --------------------------------------------------------------------------
# input contract
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoIndex:
    """Read-only view of the repository the traceback is being resolved against.

    ``repo_files`` holds repo-relative POSIX paths (``app/users.py``).
    ``repo_roots`` holds absolute prefixes that map onto the repo root, so an
    absolute frame path ``/workspace/app/users.py`` with root ``/workspace``
    resolves to ``app/users.py``.
    """

    repo_roots: tuple = ()
    repo_files: frozenset = frozenset()
    generated_files: frozenset = frozenset()
    vendored_files: frozenset = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "repo_roots", tuple(
            _norm_path(r).rstrip("/") for r in self.repo_roots))
        object.__setattr__(self, "repo_files", frozenset(
            _norm_path(p).lstrip("/") for p in self.repo_files))
        object.__setattr__(self, "generated_files", frozenset(
            _norm_path(p).lstrip("/") for p in self.generated_files))
        object.__setattr__(self, "vendored_files", frozenset(
            _norm_path(p).lstrip("/") for p in self.vendored_files))


EMPTY_INDEX = RepoIndex()


@dataclass(frozen=True)
class TracebackInput:
    text: str
    repo_index: RepoIndex = EMPTY_INDEX
    container_prefixes: tuple = ()      # ("/app/", "/srv/service/") -> repo root
    max_input_bytes: int = 5_000_000
    max_tracebacks: int = 256
    max_frames_per_traceback: int = 2_048


# --------------------------------------------------------------------------
# typed evidence
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnostic:
    code: str
    detail: str
    input_line: int = 0


@dataclass(frozen=True)
class MessageLiteral:
    raw_text: str
    decoded_text: Optional[str]
    start_offset: int
    end_offset: int
    is_secret_candidate: bool = False


@dataclass(frozen=True)
class SymptomToken:
    value: str
    normalized_value: str
    source: str            # exception_class|message_literal|message_word|
    #                        message_identifier|route_subtype
    grounded: bool
    retrieval_allowed: bool


@dataclass(frozen=True)
class MessageEvidence:
    raw_text: str
    normalized_text: str
    literals: tuple = ()
    tokens: tuple = ()


@dataclass(frozen=True)
class ExceptionEvidence:
    raw_class: str
    qualified_class: str
    class_leaf: str
    class_match: str       # qualified_exact|leaf_exact|unknown
    message: MessageEvidence
    terminal_input_line: int
    terminal_raw_text: str


@dataclass(frozen=True)
class FrameEvidence:
    ordinal: int
    input_line: int
    file_raw: str
    file_normalized: str
    repo_path: Optional[str]
    line: int
    function: Optional[str]
    source_text: Optional[str]
    in_repo: bool
    noise: str             # see NOISE_* below
    noise_reason: Optional[str]
    anchor: Optional[str]
    resolution: str = "unresolved"      # root|exact|container|suffix|ambiguous|unresolved
    candidates: tuple = ()


@dataclass(frozen=True)
class TracebackRecord:
    traceback_id: str
    occurrence_index: int
    input_start_line: int
    input_end_line: int
    frames: tuple
    exception: Optional[ExceptionEvidence]
    parse_status: str
    fingerprint: str
    preceding_log_context: tuple = ()
    is_chain_leaf: bool = True
    async_hints: tuple = ()
    reraised_hints: tuple = ()
    group_path: Optional[str] = None
    severity_markers: tuple = ()
    route: Optional["RouteMatch"] = None

    # -- derived frame views -------------------------------------------------
    @property
    def deepest_frame(self):
        return self.frames[-1] if self.frames else None

    @property
    def deepest_in_repo_frame(self):
        for fr in reversed(self.frames):
            if fr.in_repo:
                return fr
        return None

    @property
    def deepest_actionable_in_repo_frame(self):
        for fr in reversed(self.frames):
            if fr.in_repo and fr.noise == NOISE_NONE:
                return fr
        return None


@dataclass(frozen=True)
class ChainLink:
    cause_traceback_id: str
    effect_traceback_id: str
    kind: str              # direct_cause|implicit_context|exception_group_member|
    #                        repeated_emission|unknown
    explicit: bool


@dataclass(frozen=True)
class ExceptionChain:
    chain_id: str
    traceback_ids: tuple
    root_traceback_id: str
    terminal_traceback_id: str
    links: tuple = ()


@dataclass(frozen=True)
class RouteMatch:
    route_id: str
    family: str
    exception_class: str
    confidence: str        # exact|class_only|message_subtype|unknown
    priority: int
    message_subtype: Optional[str] = None
    parent_route_id: Optional[str] = None
    required_evidence: tuple = ()
    prohibited_assumptions: tuple = ()


@dataclass(frozen=True)
class OracleAnchor:
    anchor_type: str
    authority: str         # oracle|structural|lexical
    repo_path: Optional[str]
    line: Optional[int]
    function: Optional[str]
    anchor: str
    rank_tier: int


@dataclass(frozen=True)
class LocateEvidence:
    mode: str              # oracle_exact|oracle_partial|lexical_fallback|unresolved
    primary_anchor: Optional[str]
    anchors: tuple
    resolved_repo_paths: tuple
    bm25_allowed: bool
    bm25_reason: str


@dataclass(frozen=True)
class RepairSignal:
    traceback_id: str
    exception_class: str
    route_id: Optional[str]
    family: str
    anchor: Optional[str]
    grounded_tokens: tuple
    automatic_patch_allowed: bool = False
    refusal_reason: Optional[str] = None


@dataclass(frozen=True)
class TracebackEvidence:
    schema_version: str
    input_sha256: str
    parse_status: str      # complete|partial|no_traceback|over_limit
    tracebacks: tuple
    chains: tuple
    links: tuple
    primary_chain_id: Optional[str]
    primary_traceback_id: Optional[str]
    selection_confidence: str
    alternatives: tuple
    repair_signal: Optional[RepairSignal]
    locate: Optional[LocateEvidence]
    diagnostics: tuple = ()

    def by_id(self, tb_id):
        for rec in self.tracebacks:
            if rec.traceback_id == tb_id:
                return rec
        return None


# --------------------------------------------------------------------------
# lexical normalization
# --------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# Recognized log prefixes. Applied ONLY when what remains still parses as a
# traceback structure (see _classify): conservative by construction.
# A prefix must contain at least one STRONG token (bracket group, timestamp,
# level word, or key=value pair). After those, up to two bare lower-case
# component/logger names may follow (`api`, `worker`, `app.tasks`). A run of
# bare words alone is never treated as a prefix - that would let ordinary prose
# be rewritten into traceback structure.
_LOG_PREFIX_RE = re.compile(
    r"""^\s*(?:
          \[[^\]\r\n]+\]\s*
        | \d{4}-\d{2}-\d{2}[T ][0-9:.,+Z-]+\s*
        | (?:DEBUG|INFO|NOTICE|WARNING|WARN|ERROR|CRITICAL|FATAL)[ \t]+
        | [A-Za-z0-9_.:/-]+=(?:"[^"\r\n]*"|\S+)\s*
        | [a-z0-9_.-]+:[ \t]+(?=(?:File\s"|Traceback\s\())
    )+
    (?:(?!File\s")(?!Traceback\s\()[a-z][a-z0-9_.:-]*[ \t]){0,2}""",
    re.VERBOSE,
)

_GUTTER_RE = re.compile(r"^(?P<pad>\s*)(?P<bars>(?:[+|]\s?)+)")
# CPython emits `+-+---- 1 ----` for the first branch and `+---- 2 ----` for
# every later one; the closing rule carries no digit and must not match.
_GROUP_SEP_RE = re.compile(r"^\s*\+(?:-\+)?-+\s*(\d+)\s*-+")

_START_RE = re.compile(
    r"^(?:\+\s*)?(?P<group>Exception Group )?Traceback \(most recent call last\):\s*$")
_FRAME_RE = re.compile(
    r'^\s*File "(?P<path>[^"\r\n]+)", line (?P<line>[1-9][0-9]*)'
    r'(?:, in (?P<func>.*?))?\s*$')
_EXC_RE = re.compile(
    r"^(?P<class>(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*)"
    r"(?::[ \t]?(?P<message>.*))?\s*$")
_CAUSE_DELIM_RE = re.compile(
    r"^The above exception was the direct cause of the following exception:\s*$")
_CONTEXT_DELIM_RE = re.compile(
    r"^During handling of the above exception, another exception occurred:\s*$")
_REPEAT_RE = re.compile(r"^\s*\[Previous line repeated \d+ more times?\]\s*$")
_CARET_RE = re.compile(r"^\s*[\^~ ]+\s*$")
_ELLIPSIS_RE = re.compile(r"^\s*\.\.\.\s*$")

_SEVERITY_HARD_RE = re.compile(r"(?i)\b(?:uncaught|unhandled|fatal|panic)\b")
_SEVERITY_SOFT_RE = re.compile(r"(?i)\b(?:critical|fatal|error)\b")
_ASGI_MARKER_RE = re.compile(r"(?i)exception in asgi application")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _strip_gutter(line: str):
    """Remove exception-group gutter bars. Returns (line, gutter_depth)."""
    m = _GUTTER_RE.match(line)
    if not m:
        return line, 0
    bars = m.group("bars")
    depth = bars.count("|") + bars.count("+")
    return line[m.end():], depth


def _strip_log_prefix(line: str) -> str:
    m = _LOG_PREFIX_RE.match(line)
    if not m or m.end() == 0:
        return line
    return line[m.end():]


LINE_START = "start"
LINE_FRAME = "frame"
LINE_EXC = "exception"
LINE_CAUSE = "cause_delim"
LINE_CONTEXT = "context_delim"
LINE_GROUP_SEP = "group_sep"
LINE_SOURCE = "source"
LINE_CARET = "caret"
LINE_REPEAT = "repeat_marker"
LINE_ELLIPSIS = "ellipsis"
LINE_BLANK = "blank"
LINE_OTHER = "other"


@dataclass(frozen=True)
class ScanLine:
    index: int             # 1-based input line number
    raw: str
    body: str              # ANSI + gutter + log-prefix stripped
    kind: str
    indent: int
    gutter: int
    group_branch: Optional[int] = None


def _classify_body(body: str) -> str:
    if _START_RE.match(body):
        return LINE_START
    if _FRAME_RE.match(body):
        return LINE_FRAME
    if _CAUSE_DELIM_RE.match(body.strip()):
        return LINE_CAUSE
    if _CONTEXT_DELIM_RE.match(body.strip()):
        return LINE_CONTEXT
    if _REPEAT_RE.match(body):
        return LINE_REPEAT
    if _ELLIPSIS_RE.match(body):
        return LINE_ELLIPSIS
    if not body.strip():
        return LINE_BLANK
    if _CARET_RE.match(body):
        return LINE_CARET
    indent = len(body) - len(body.lstrip(" "))
    if indent == 0 and _EXC_RE.match(body):
        return LINE_EXC
    return LINE_SOURCE if indent else LINE_OTHER


_EXC_NAME_SHAPE_RE = re.compile(
    r"^[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning|Exit|Interrupt|Group|Failure)$")


def _plausible_exception_line(body: str) -> bool:
    m = _EXC_RE.match(body)
    if not m:
        return False
    cls = m.group("class")
    leaf = cls.rsplit(".", 1)[-1]
    return ("." in cls or leaf in KNOWN_BUILTIN_EXCEPTIONS
            or bool(_EXC_NAME_SHAPE_RE.match(leaf)))


def scan(text: str):
    """Line scanner: raw text -> classified ``ScanLine`` tuple.

    Prefix stripping is conservative: a prefix is removed only when doing so
    turns an otherwise-unstructured line into recognized traceback structure.
    """
    out = []
    for i, raw in enumerate(text.splitlines(), start=1):
        plain = strip_ansi(raw)
        sep = _GROUP_SEP_RE.match(plain)
        if sep:
            out.append(ScanLine(i, raw, plain, LINE_GROUP_SEP, 0, 0,
                                int(sep.group(1))))
            continue
        body, gutter = _strip_gutter(plain)
        kind = _classify_body(body)
        # Log prefixes start at column 0. Indented text is traceback body
        # (source lines), and stripping a leading word off it would turn
        # `    return user.name` into a phantom `user.name` exception line.
        if kind in (LINE_OTHER, LINE_SOURCE, LINE_BLANK) and not body[:1].isspace():
            stripped = _strip_log_prefix(body)
            if stripped != body:
                alt = _classify_body(stripped)
                # An exception line recovered by prefix stripping must name a
                # PLAUSIBLE exception, or every `INFO some chatter` line in the
                # log becomes a phantom terminal exception.
                if alt == LINE_EXC and not _plausible_exception_line(stripped):
                    alt = None
                # SOURCE/BLANK are accepted too: a prefix strong enough to match
                # carries a timestamp/level/kv token, so the remainder is log
                # payload - and a prefixed source line left as OTHER would
                # truncate the traceback block before its exception line.
                if alt in (LINE_START, LINE_FRAME, LINE_EXC, LINE_CAUSE,
                           LINE_CONTEXT, LINE_SOURCE, LINE_BLANK):
                    body, kind = stripped, alt
        indent = len(body) - len(body.lstrip(" "))
        out.append(ScanLine(i, raw, body, kind, indent, gutter))
    return tuple(out)


# --------------------------------------------------------------------------
# path handling / noise classification
# --------------------------------------------------------------------------

NOISE_NONE = "none"
NOISE_FRAMEWORK = "framework"
NOISE_STDLIB = "stdlib"
NOISE_THIRD_PARTY = "third_party"
NOISE_DB_DRIVER = "database_driver"
NOISE_GENERATED = "generated"
NOISE_VENDORED = "vendored"
NOISE_UNKNOWN_EXTERNAL = "unknown_external"

# Matched on PATH SEGMENTS, never substrings: src/my_sqlalchemy_adapter.py is
# application code, not a SQLAlchemy internal frame.
FRAMEWORK_SEGMENTS = frozenset({
    "fastapi", "starlette", "uvicorn", "sqlalchemy", "django", "flask",
    "werkzeug", "gunicorn", "celery", "asgiref", "alembic",
})
STDLIB_SEGMENTS = frozenset({
    "asyncio", "concurrent", "contextlib", "unittest", "importlib", "runpy",
    "threading", "multiprocessing", "json", "socketserver", "http",
})
THIRD_PARTY_SEGMENTS = frozenset({
    "anyio", "pydantic", "pydantic_core", "httpx", "httpcore", "requests",
    "urllib3", "boto3", "botocore", "redis", "pytest", "_pytest", "click",
})
DB_DRIVER_SEGMENTS = frozenset({
    "psycopg", "psycopg2", "asyncpg", "aiosqlite", "sqlite3", "pymysql",
    "MySQLdb", "cx_Oracle", "pyodbc",
})
ASYNC_SEGMENTS = frozenset({
    "asyncio", "anyio", "uvicorn", "starlette", "fastapi", "asgiref",
    "trio", "aiohttp",
})
EXTERNAL_DIR_SEGMENTS = frozenset({
    "site-packages", "dist-packages", ".venv", "venv", "node_modules",
})
_STDLIB_DIR_RE = re.compile(r"^(?:python|pypy)\d+\.\d+$")

_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/]")


def _norm_path(raw: str) -> str:
    """Normalize a path for comparison. Purely lexical - never touches disk."""
    if not raw:
        return ""
    p = raw.replace("\\", "/")
    m = _WIN_DRIVE_RE.match(raw)
    if m:
        p = "/" + m.group(1).lower() + "/" + p[len(m.group(0)):]
    while "//" in p:
        p = p.replace("//", "/")
    absolute = p.startswith("/")
    p = posixpath.normpath(p)
    if absolute and not p.startswith("/"):
        p = "/" + p
    return p


def _segments(path: str):
    return tuple(s for s in path.split("/") if s and s != ".")


def classify_noise(normalized: str, repo_path, repo_index: RepoIndex):
    """-> (noise_class, reason). Segment-based, deterministic."""
    segs = _segments(normalized)
    if repo_path is not None:
        if repo_path in repo_index.generated_files:
            return NOISE_GENERATED, "configured generated file"
        if repo_path in repo_index.vendored_files:
            return NOISE_VENDORED, "configured vendored file"
    segset = set(segs)
    stem = frozenset(s.split(".py")[0] for s in segs)
    if not segset & EXTERNAL_DIR_SEGMENTS:
        for i, s in enumerate(segs):
            if _STDLIB_DIR_RE.match(s) and i and segs[i - 1] in ("lib", "lib64"):
                return NOISE_STDLIB, "stdlib install directory"
    if segset & EXTERNAL_DIR_SEGMENTS:
        # inside site-packages: refine by distribution package segment
        for s in segs:
            if s in DB_DRIVER_SEGMENTS:
                return NOISE_DB_DRIVER, "database driver package %s" % s
            if s in FRAMEWORK_SEGMENTS:
                return NOISE_FRAMEWORK, "framework package %s" % s
            if s in THIRD_PARTY_SEGMENTS:
                return NOISE_THIRD_PARTY, "third-party package %s" % s
        return NOISE_THIRD_PARTY, "installed package directory"
    if repo_path is None:
        for s in segs:
            if s in DB_DRIVER_SEGMENTS:
                return NOISE_DB_DRIVER, "database driver package %s" % s
            if s in FRAMEWORK_SEGMENTS:
                return NOISE_FRAMEWORK, "framework package %s" % s
            if s in THIRD_PARTY_SEGMENTS:
                return NOISE_THIRD_PARTY, "third-party package %s" % s
            if s in STDLIB_SEGMENTS or (stem & STDLIB_SEGMENTS and s.endswith(".py")):
                return NOISE_STDLIB, "stdlib module %s" % s
        return NOISE_UNKNOWN_EXTERNAL, "path not resolvable to the repository"
    return NOISE_NONE, None


def _async_hints(frames):
    hints = []
    for fr in frames:
        for s in _segments(fr.file_normalized):
            base = s[:-3] if s.endswith(".py") else s
            if base in ASYNC_SEGMENTS:
                tag = ("asyncio_frame" if base in ("asyncio", "anyio", "trio")
                       else "asgi_framework_frame")
                if tag not in hints:
                    hints.append(tag)
    return tuple(hints)


def resolve_repo_path(normalized: str, repo_index: RepoIndex,
                      container_prefixes=()):
    """-> (repo_path|None, resolution, candidates).

    Order: repo root -> exact relative -> container prefix -> unique suffix.
    An ambiguous suffix never guesses.
    """
    if not normalized:
        return None, "unresolved", ()
    rel = normalized.lstrip("/")
    for root in repo_index.repo_roots:
        r = root.rstrip("/")
        if r and normalized.startswith(r + "/"):
            cand = normalized[len(r) + 1:]
            if cand in repo_index.repo_files:
                return cand, "root", ()
    if not normalized.startswith("/") and normalized in repo_index.repo_files:
        return normalized, "exact", ()
    for pref in container_prefixes:
        p = _norm_path(pref).rstrip("/")
        if p and normalized.startswith(p + "/"):
            cand = normalized[len(p) + 1:]
            if cand in repo_index.repo_files:
                return cand, "container", ()
    if ".." in _segments(normalized):
        return None, "unresolved", ()
    parts = _segments(rel)
    ambiguous = ()
    for take in range(1, len(parts) + 1):
        suffix = "/".join(parts[len(parts) - take:])
        hits = sorted(p for p in repo_index.repo_files
                      if p == suffix or p.endswith("/" + suffix))
        if len(hits) == 1:
            return hits[0], "suffix", ()
        if len(hits) > 1:
            # keep lengthening the suffix; a longer suffix may disambiguate
            ambiguous = tuple(hits)
            continue
        break
    if ambiguous:
        return None, "ambiguous", ambiguous
    return None, "unresolved", ()


# --------------------------------------------------------------------------
# message parsing / grounded symptom tokens
# --------------------------------------------------------------------------

_QUOTED_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_DOTTED_ID_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}")
_AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_BEARER_RE = re.compile(r"(?i)\b(?:bearer|authorization|password|passwd|secret|token|api[_-]?key)\b")
_OPAQUE_RE = re.compile(r"^[A-Za-z0-9+/=_-]{32,}$")

# Phrases we are willing to lift verbatim as a grounded symptom phrase.
KNOWN_MESSAGE_PHRASES = (
    "UNIQUE constraint failed",
    "NOT NULL constraint failed",
    "FOREIGN KEY constraint failed",
    "CHECK constraint failed",
    "duplicate key value violates unique constraint",
    "violates not-null constraint",
    "violates foreign key constraint",
    "no such table",
    "no such column",
    "connection refused",
    "database is locked",
    "has no attribute",
    "object is not subscriptable",
    "object is not callable",
    "not enough values to unpack",
    "positional argument",
)

_STOPWORDS = frozenset({"the", "a", "an", "is", "was", "of", "for", "and", "or",
                        "to", "in", "on", "at", "by", "with", "not", "no"})


def is_secret_candidate(text: str) -> bool:
    if not text:
        return False
    if _EMAIL_RE.search(text) or _JWT_RE.search(text) or _AWS_KEY_RE.search(text):
        return True
    if _BEARER_RE.search(text):
        return True
    return bool(_OPAQUE_RE.match(text.strip()))


def _safe_literal(raw: str):
    """``ast.literal_eval`` on ONE extracted literal. Never ``eval``."""
    try:
        val = ast.literal_eval(raw)
    except Exception:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, (int, float, bool)) or val is None:
        return str(val)
    if isinstance(val, (tuple, list, set, frozenset)):
        return None      # container: individual members are extracted separately
    return None


def parse_message(raw_message: str, qualified_class: str) -> MessageEvidence:
    raw = raw_message or ""
    normalized = raw.strip()
    literals = []
    tokens = []
    seen = set()

    def add_token(value, source, retrieval_allowed=True):
        value = value.strip()
        if not value:
            return
        norm = value.lower()
        key = (norm, source)
        if key in seen:
            return
        seen.add(key)
        tokens.append(SymptomToken(value=value, normalized_value=norm,
                                   source=source, grounded=True,
                                   retrieval_allowed=retrieval_allowed))

    if qualified_class:
        add_token(qualified_class.rsplit(".", 1)[-1], "exception_class")

    for m in _QUOTED_RE.finditer(normalized):
        rawlit = m.group(0)
        decoded = _safe_literal(rawlit)
        secret = is_secret_candidate(decoded or rawlit)
        literals.append(MessageLiteral(raw_text=rawlit, decoded_text=decoded,
                                       start_offset=m.start(), end_offset=m.end(),
                                       is_secret_candidate=secret))
        if decoded:
            add_token(decoded, "message_literal", retrieval_allowed=not secret)

    for m in _DOTTED_ID_RE.finditer(normalized):
        val = m.group(0)
        if val.endswith("."):
            continue
        add_token(val, "message_identifier",
                  retrieval_allowed=not is_secret_candidate(val))

    low = normalized.lower()
    for phrase in KNOWN_MESSAGE_PHRASES:
        if phrase.lower() in low:
            add_token(phrase, "message_word")

    if not tokens or all(t.source == "exception_class" for t in tokens):
        # last resort: single meaningful bare identifier words, still verbatim
        for w in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", normalized)[:6]:
            if w.lower() in _STOPWORDS:
                continue
            add_token(w, "message_word",
                      retrieval_allowed=not is_secret_candidate(w))

    return MessageEvidence(raw_text=raw, normalized_text=normalized,
                           literals=tuple(literals), tokens=tuple(tokens))


# --------------------------------------------------------------------------
# exception routing table
# --------------------------------------------------------------------------

KNOWN_BUILTIN_EXCEPTIONS = frozenset({
    "BaseException", "Exception", "ArithmeticError", "AssertionError",
    "AttributeError", "BlockingIOError", "BrokenPipeError", "BufferError",
    "ConnectionError", "ConnectionRefusedError", "ConnectionResetError",
    "EOFError", "FileExistsError", "FileNotFoundError", "FloatingPointError",
    "GeneratorExit", "ImportError", "IndentationError", "IndexError",
    "InterruptedError", "IsADirectoryError", "KeyError", "KeyboardInterrupt",
    "LookupError", "MemoryError", "ModuleNotFoundError", "NameError",
    "NotADirectoryError", "NotImplementedError", "OSError", "OverflowError",
    "PermissionError", "ProcessLookupError", "RecursionError",
    "ReferenceError", "RuntimeError", "StopAsyncIteration", "StopIteration",
    "SyntaxError", "SystemError", "SystemExit", "TabError", "TimeoutError",
    "TypeError", "UnboundLocalError", "UnicodeDecodeError",
    "UnicodeEncodeError", "UnicodeError", "ValueError", "ZeroDivisionError",
    "ExceptionGroup", "BaseExceptionGroup",
})

# Generic wrappers must never hide a specific cause.
GENERIC_WRAPPER_CLASSES = frozenset({
    "Exception", "BaseException", "RuntimeError", "ApplicationError",
    "RequestError", "ServiceError", "ExceptionGroup", "BaseExceptionGroup",
})

_NONE_ATTR_RE = re.compile(
    r"(?i)^'NoneType' object has no attribute '([A-Za-z_][A-Za-z0-9_]*)'")
_MODULE_ATTR_RE = re.compile(
    r"(?i)^module '([^']+)' has no attribute '([A-Za-z_][A-Za-z0-9_]*)'")
_ANY_ATTR_RE = re.compile(
    r"(?i)has no attribute '([A-Za-z_][A-Za-z0-9_]*)'")

# (class_regex, message_regex|None, route_id, family, priority, subtype)
ROUTING_TABLE = (
    (r"^(?:builtins\.)?KeyError$", None,
     "key.missing_guard", "missing_key_guard", 110, None),
    (r"^(?:builtins\.)?AttributeError$", _NONE_ATTR_RE.pattern,
     "attribute.none_guard", "none_guard", 130, "none_attribute"),
    (r"^(?:builtins\.)?AttributeError$", _MODULE_ATTR_RE.pattern,
     "attribute.swap", "attribute_swap", 120, "module_attribute"),
    (r"^(?:builtins\.)?AttributeError$", None,
     "attribute.swap", "attribute_swap", 110, None),
    (r"^sqlalchemy\.exc\.IntegrityError$", None,
     "integrity.constraint_mismatch", "constraint_mismatch", 120, None),
    (r"^(?:\w+\.)*IntegrityError$", None,
     "integrity.constraint_mismatch", "constraint_mismatch", 110, None),
    (r"^sqlalchemy\.exc\.OperationalError$", None,
     "database.operational", "database_operational", 120, None),
    (r"^(?:\w+\.)*OperationalError$", None,
     "database.operational", "database_operational", 105, None),
    (r"^sqlalchemy\.exc\.ProgrammingError$", None,
     "database.query_contract", "query_contract", 120, None),
    (r"^(?:\w+\.)*ProgrammingError$", None,
     "database.query_contract", "query_contract", 105, None),
    (r"^(?:builtins\.)?ModuleNotFoundError$", None,
     "import.dependency", "import_dependency", 120, None),
    (r"^(?:builtins\.)?ImportError$", None,
     "import.dependency", "import_dependency", 100, None),
    (r"^(?:builtins\.)?UnboundLocalError$", None,
     "name.initialization", "initialization_order", 110, None),
    (r"^(?:builtins\.)?NameError$", None,
     "name.binding", "missing_binding", 100, None),
    (r"^(?:builtins\.)?TypeError$", None,
     "type.contract_mismatch", "contract_mismatch", 100, None),
    (r"^(?:builtins\.)?IndexError$", None,
     "index.bounds_guard", "bounds_guard", 100, None),
    (r"^(?:builtins\.)?ValueError$", None,
     "value.validation", "validation_parse", 100, None),
    (r"^(?:builtins\.)?AssertionError$", None,
     "assertion.invariant", "invariant_violation", 100, None),
    (r"^(?:builtins\.|asyncio\.)?TimeoutError$", None,
     "timeout.retry_or_latency", "timeout_retry", 100, None),
    (r"^(?:builtins\.)?ZeroDivisionError$", None,
     "value.validation", "validation_parse", 100, None),
    (r"^(?:builtins\.)?FileNotFoundError$", None,
     "io.missing_path", "missing_path", 100, None),
    (r"^(?:builtins\.)?RecursionError$", None,
     "recursion.termination", "recursion_termination", 100, None),
)

INTEGRITY_SUBTYPES = (
    (r"(?i)\b23505\b|unique constraint|duplicate key|unique violation",
     "integrity.unique", "unique", "uniqueness_duplicate"),
    (r"(?i)\b23502\b|not[- ]null constraint|violates not-null|NOT NULL",
     "integrity.not_null", "not_null", "required_field_default"),
    (r"(?i)\b23503\b|foreign key constraint|FOREIGN KEY",
     "integrity.foreign_key", "foreign_key", "referential_order"),
    (r"(?i)\b23514\b|check constraint",
     "integrity.check", "check", "validation_invariant"),
)

REQUIRED_EVIDENCE = {
    "missing_key_guard": ("grounded key literal", "actionable in-repo frame"),
    "none_guard": ("attribute name", "actionable in-repo frame"),
    "attribute_swap": ("attribute name", "actionable in-repo frame"),
    "constraint_mismatch": ("constraint subtype", "actionable in-repo frame"),
    "import_dependency": ("module name",),
}

PROHIBITED_ASSUMPTIONS = {
    "missing_key_guard": (
        "a missing REQUIRED field is not fixed by returning a default",),
    "constraint_mismatch": (
        "a uniqueness violation is not fixed by adding a retry",),
    "none_guard": (
        "silencing the attribute access hides the real None producer",),
    "timeout_retry": (
        "a timeout is not automatically fixed by raising the timeout",),
}

_COMPILED_ROUTES = tuple(
    (re.compile(cls_pat), re.compile(msg_pat) if msg_pat else None,
     rid, fam, prio, sub)
    for cls_pat, msg_pat, rid, fam, prio, sub in ROUTING_TABLE)
_COMPILED_INTEGRITY = tuple(
    (re.compile(p), rid, sub, fam) for p, rid, sub, fam in INTEGRITY_SUBTYPES)

UNKNOWN_ROUTE_FAMILY = "unknown_exception"


def route_exception(qualified_class: str, class_leaf: str, message: str):
    """Table-driven routing. -> RouteMatch|None (None = unknown exception)."""
    best = None
    for cls_re, msg_re, rid, fam, prio, sub in _COMPILED_ROUTES:
        qualified_hit = cls_re.match(qualified_class or "")
        leaf_hit = cls_re.match(class_leaf or "")
        if not (qualified_hit or leaf_hit):
            continue
        if msg_re is not None and not msg_re.search(message or ""):
            continue
        confidence = ("message_subtype" if msg_re is not None
                      else ("exact" if qualified_hit and "." in (qualified_class or "")
                            else "class_only"))
        score = (prio, 1 if qualified_hit else 0, 1 if msg_re is not None else 0)
        cand = (score, RouteMatch(
            route_id=rid, family=fam, exception_class=qualified_class or class_leaf,
            confidence=confidence, priority=prio, message_subtype=sub,
            required_evidence=REQUIRED_EVIDENCE.get(fam, ()),
            prohibited_assumptions=PROHIBITED_ASSUMPTIONS.get(fam, ())))
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        return None
    route = best[1]
    if route.family == "constraint_mismatch":
        for pat, rid, sub, subfam in _COMPILED_INTEGRITY:
            if pat.search(message or ""):
                route = replace(route, route_id=rid, message_subtype=sub,
                                parent_route_id="integrity.constraint_mismatch",
                                confidence="message_subtype")
                break
    return route


def route_subtype_tokens(route: Optional[RouteMatch]):
    """Route-derived vocabulary, explicitly NOT grounded in the log text."""
    if route is None:
        return ()
    vals = [route.family]
    if route.message_subtype:
        vals.append(route.message_subtype)
    return tuple(SymptomToken(value=v, normalized_value=v.lower(),
                              source="route_subtype", grounded=False,
                              retrieval_allowed=False) for v in vals)


# --------------------------------------------------------------------------
# block parsing
# --------------------------------------------------------------------------


def _find_blocks(lines):
    """-> list of (start_idx, end_idx_exclusive) over the scan tuple."""
    starts = [i for i, ln in enumerate(lines) if ln.kind == LINE_START]
    blocks = []
    for n, s in enumerate(starts):
        limit = starts[n + 1] if n + 1 < len(starts) else len(lines)
        end = limit
        seen_frame = False
        for j in range(s + 1, limit):
            k = lines[j].kind
            if k == LINE_FRAME:
                seen_frame = True
                continue
            if k in (LINE_SOURCE, LINE_CARET, LINE_BLANK, LINE_REPEAT,
                     LINE_ELLIPSIS, LINE_GROUP_SEP):
                continue
            if k == LINE_EXC:
                # absorb continuation lines of a multiline exception message
                end = j + 1
                while end < limit and lines[end].kind == LINE_SOURCE:
                    end += 1
                break
            if k in (LINE_CAUSE, LINE_CONTEXT):
                end = j
                break
            if k == LINE_OTHER and seen_frame:
                end = j
                break
        blocks.append((s, end))
    return blocks


def _parse_frames(lines, start, end, repo_index, container_prefixes, max_frames):
    frames = []
    truncated = False
    for j in range(start, end):
        ln = lines[j]
        if ln.kind == LINE_ELLIPSIS or ln.kind == LINE_REPEAT:
            truncated = True
            continue
        if ln.kind != LINE_FRAME:
            continue
        if len(frames) >= max_frames:
            truncated = True
            break
        m = _FRAME_RE.match(ln.body)
        raw_path = m.group("path")
        normalized = _norm_path(raw_path)
        repo_path, resolution, candidates = resolve_repo_path(
            normalized, repo_index, container_prefixes)
        noise, reason = classify_noise(normalized, repo_path, repo_index)
        func = m.group("func")
        func = func.strip() if func else None
        source = None
        for k in range(j + 1, min(j + 3, end)):
            if lines[k].kind == LINE_SOURCE:
                source = lines[k].body.strip()
                break
            if lines[k].kind != LINE_CARET:
                break
        anchor = None
        if repo_path:
            anchor = ("%s:%d:%s" % (repo_path, int(m.group("line")), func)
                      if func else "%s:%d" % (repo_path, int(m.group("line"))))
        frames.append(FrameEvidence(
            ordinal=len(frames), input_line=ln.index, file_raw=raw_path,
            file_normalized=normalized, repo_path=repo_path,
            line=int(m.group("line")), function=func, source_text=source,
            in_repo=repo_path is not None, noise=noise, noise_reason=reason,
            anchor=anchor, resolution=resolution, candidates=candidates))
    return tuple(frames), truncated


def _reraise_hints(frames, exc):
    hints = []
    paths = [fr.anchor or fr.file_normalized for fr in frames]
    if len(paths) != len(set(paths)):
        hints.append("overlapping_frame_prefix")
    if exc is not None and frames:
        deepest = frames[-1]
        if deepest.source_text and re.match(r"^\s*raise\b", deepest.source_text):
            if not re.search(r"\bfrom\b", deepest.source_text):
                hints.append("bare_reraise_source")
    return tuple(hints)


def _fingerprint(exc, frames):
    h = hashlib.sha256()
    h.update((exc.qualified_class if exc else "<none>").encode())
    h.update(b"\x00")
    h.update((exc.message.normalized_text if exc else "").encode())
    for fr in frames:
        h.update(b"\x00")
        h.update(("%s:%d:%s" % (fr.repo_path or fr.file_normalized, fr.line,
                                fr.function or "")).encode())
    return h.hexdigest()[:16]


def _preceding_context(lines, start, span=3):
    out = []
    for j in range(max(0, start - span), start):
        if lines[j].raw.strip():
            out.append(lines[j].raw.rstrip())
    return tuple(out)


def _severity(context_lines, own_lines):
    markers = []
    blob = "\n".join(list(context_lines) + list(own_lines))
    if _SEVERITY_HARD_RE.search(blob):
        markers.append("unhandled")
    if _ASGI_MARKER_RE.search(blob):
        markers.append("asgi_application")
    if _SEVERITY_SOFT_RE.search(blob):
        markers.append("error_level")
    return tuple(markers)


def _parse_block(lines, span, idx, repo_index, container_prefixes, max_frames,
                 diagnostics):
    start, end = span
    frames, truncated = _parse_frames(lines, start + 1, end, repo_index,
                                      container_prefixes, max_frames)
    exc = None
    for j in range(end - 1, start, -1):
        if lines[j].kind == LINE_EXC:
            m = _EXC_RE.match(lines[j].body)
            raw_class = m.group("class")
            message_raw = m.group("message") or ""
            # multiline messages: absorb following deeper-indented text that is
            # not itself structure
            k = j + 1
            extra = []
            while k < end and lines[k].kind in (LINE_SOURCE,):
                extra.append(lines[k].body.strip())
                k += 1
            if extra:
                message_raw = (message_raw + " " + " ".join(extra)).strip()
            leaf = raw_class.rsplit(".", 1)[-1]
            if "." in raw_class:
                match_kind = "qualified_exact"
            elif leaf in KNOWN_BUILTIN_EXCEPTIONS:
                match_kind = "leaf_exact"
            else:
                match_kind = "unknown"
            exc = ExceptionEvidence(
                raw_class=raw_class, qualified_class=raw_class, class_leaf=leaf,
                class_match=match_kind,
                message=parse_message(message_raw, raw_class),
                terminal_input_line=lines[j].index,
                terminal_raw_text=lines[j].raw.rstrip())
            break
    status = "complete"
    if exc is None:
        status = "partial"
        diagnostics.append(Diagnostic(
            "missing_terminal_exception",
            "traceback block has no terminal exception line",
            lines[start].index))
    elif truncated:
        status = "partial"
        diagnostics.append(Diagnostic(
            "truncated_frames", "frame block was truncated in the excerpt",
            lines[start].index))
    if not frames and exc is not None:
        status = "partial"
        diagnostics.append(Diagnostic(
            "no_frames", "traceback block carries no frame lines",
            lines[start].index))
    own = [lines[j].raw for j in range(start, end)]
    ctx = _preceding_context(lines, start)
    group_branch = None
    for j in range(start, -1, -1):
        if lines[j].kind == LINE_GROUP_SEP:
            group_branch = "group[%d]" % lines[j].group_branch
            break
        if lines[j].kind == LINE_START and j != start:
            break
    route = None
    if exc is not None:
        route = route_exception(exc.qualified_class, exc.class_leaf,
                                exc.message.normalized_text)
    return TracebackRecord(
        traceback_id="tb-%03d" % (idx + 1), occurrence_index=idx,
        input_start_line=lines[start].index, input_end_line=lines[end - 1].index,
        frames=frames, exception=exc, parse_status=status,
        fingerprint=_fingerprint(exc, frames), preceding_log_context=ctx,
        async_hints=_async_hints(frames),
        reraised_hints=_reraise_hints(frames, exc),
        group_path=group_branch, severity_markers=_severity(ctx, own),
        route=route)


# --------------------------------------------------------------------------
# chains
# --------------------------------------------------------------------------


def _resolve_links(records, lines, blocks):
    links = []
    for n in range(len(records) - 1):
        _, end_a = blocks[n]
        start_b, _ = blocks[n + 1]
        kind = None
        for j in range(end_a, start_b):
            if lines[j].kind == LINE_CAUSE:
                kind = "direct_cause"
            elif lines[j].kind == LINE_CONTEXT:
                kind = "implicit_context"
        if kind:
            links.append(ChainLink(records[n].traceback_id,
                                   records[n + 1].traceback_id, kind, True))
    # exception-group membership: a group parent owns the branches after it
    parent = None
    for rec in records:
        if rec.exception and rec.exception.class_leaf in (
                "ExceptionGroup", "BaseExceptionGroup"):
            parent = rec
            continue
        if parent is not None and rec.group_path:
            links.append(ChainLink(rec.traceback_id, parent.traceback_id,
                                   "exception_group_member", True))
    # repeated emissions (metadata relationship, never a causal chain)
    seen = {}
    for rec in records:
        prev = seen.get(rec.fingerprint)
        if prev is not None:
            links.append(ChainLink(prev, rec.traceback_id,
                                   "repeated_emission", False))
        else:
            seen[rec.fingerprint] = rec.traceback_id
    return tuple(links)


def build_chains(records, links):
    causal = [l for l in links if l.kind in ("direct_cause", "implicit_context",
                                             "exception_group_member")]
    parent = {r.traceback_id: r.traceback_id for r in records}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for l in causal:
        a, b = find(l.cause_traceback_id), find(l.effect_traceback_id)
        if a != b:
            parent[b] = a
    groups = {}
    for r in records:
        groups.setdefault(find(r.traceback_id), []).append(r.traceback_id)
    chains = []
    for n, key in enumerate(sorted(groups, key=lambda k: groups[k][0])):
        ids = tuple(groups[key])
        member_links = tuple(l for l in causal
                             if l.cause_traceback_id in ids
                             and l.effect_traceback_id in ids)
        effects = {l.effect_traceback_id for l in member_links}
        causes = {l.cause_traceback_id for l in member_links}
        roots = [i for i in ids if i not in effects] or [ids[0]]
        terminals = [i for i in ids if i not in causes] or [ids[-1]]
        chains.append(ExceptionChain(
            chain_id="chain-%03d" % (n + 1), traceback_ids=ids,
            root_traceback_id=roots[0], terminal_traceback_id=terminals[-1],
            links=member_links))
    return tuple(chains)


def _has_actionable(rec):
    return rec.deepest_actionable_in_repo_frame is not None


def _selection_key(chain, records_by_id, dup_ids):
    """Lexicographic selection key - deliberately NOT a float score."""
    recs = [records_by_id[i] for i in chain.traceback_ids]
    actionable = any(_has_actionable(r) for r in recs)
    routed = any(r.route is not None for r in recs)
    fatal = any("unhandled" in r.severity_markers or
                "asgi_application" in r.severity_markers for r in recs)
    in_repo = any(r.deepest_in_repo_frame is not None for r in recs)
    depth = max((len([f for f in r.frames if f.in_repo and f.noise == NOISE_NONE])
                 for r in recs), default=0)
    not_dup = 0 if all(i in dup_ids for i in chain.traceback_ids) else 1
    latest = max(r.occurrence_index for r in recs)
    return (int(actionable), int(routed), int(fatal), int(in_repo), depth,
            not_dup, latest)


def select_primary_chain(chains, records, links):
    if not chains:
        return None, "unknown", ()
    by_id = {r.traceback_id: r for r in records}
    dup_ids = {l.effect_traceback_id for l in links
               if l.kind == "repeated_emission"}
    scored = sorted(((_selection_key(c, by_id, dup_ids), c) for c in chains),
                    key=lambda t: t[0], reverse=True)
    best_key, best = scored[0]
    ties = [c for k, c in scored if k[:-1] == best_key[:-1]]
    if len(chains) == 1:
        confidence = "high" if best_key[0] and best_key[1] else "medium"
    elif len(ties) > 1:
        confidence = "low"
    elif best_key[:2] > scored[1][0][:2]:
        confidence = "high"
    else:
        confidence = "medium"
    alternatives = tuple(
        {"traceback_id": c.terminal_traceback_id,
         "chain_id": c.chain_id,
         "reason": "independent actionable traceback with comparable evidence"}
        for _, c in scored[1:4])
    return best, confidence, alternatives


# --------------------------------------------------------------------------
# repair signal + locate
# --------------------------------------------------------------------------


def select_repair_signal(chain, records):
    """The incident-facing exception is not always the best repair signal."""
    if chain is None:
        return None
    by_id = {r.traceback_id: r for r in records}
    members = [by_id[i] for i in chain.traceback_ids]
    terminal = by_id.get(chain.terminal_traceback_id) or members[-1]

    def specific(rec):
        return (rec.exception is not None
                and rec.route is not None
                and rec.exception.class_leaf not in GENERIC_WRAPPER_CLASSES)

    chosen = None
    if specific(terminal) and _has_actionable(terminal):
        chosen = terminal
    if chosen is None:
        cands = [r for r in members if specific(r)]
        # prefer actionable frame, then message-subtype route, then priority,
        # then the deepest cause (earliest block in a chain is the root cause)
        cands.sort(key=lambda r: (
            int(_has_actionable(r)),
            int(r.route.message_subtype is not None),
            r.route.priority,
            -r.occurrence_index))
        chosen = cands[-1] if cands else None
    if chosen is None:
        chosen = terminal
    frame = chosen.deepest_actionable_in_repo_frame or chosen.deepest_in_repo_frame
    tokens = tuple(t.value for t in (chosen.exception.message.tokens
                                     if chosen.exception else ())
                   if t.retrieval_allowed and t.source in
                   ("message_literal", "message_identifier", "message_word"))
    family = chosen.route.family if chosen.route else UNKNOWN_ROUTE_FAMILY
    return RepairSignal(
        traceback_id=chosen.traceback_id,
        exception_class=(chosen.exception.qualified_class
                         if chosen.exception else "<unknown>"),
        route_id=chosen.route.route_id if chosen.route else None,
        family=family, anchor=frame.anchor if frame else None,
        grounded_tokens=tokens)


def build_locate(signal, records, chain):
    """Frames outrank lexical retrieval by hard precedence."""
    if signal is None:
        return LocateEvidence("unresolved", None, (), (), True,
                              "no traceback parsed")
    by_id = {r.traceback_id: r for r in records}
    rec = by_id.get(signal.traceback_id)
    order = [rec] + [by_id[i] for i in (chain.traceback_ids if chain else ())
                     if by_id[i] is not rec]
    anchors, paths, seen = [], [], set()

    def push(a):
        if a.anchor in seen:
            return
        seen.add(a.anchor)
        anchors.append(a)

    for r in order:
        if r is None:
            continue
        actionable = [f for f in r.frames if f.in_repo and f.noise == NOISE_NONE]
        deepest = actionable[-1] if actionable else None
        for fr in reversed(actionable):
            tier = 0 if fr is deepest and r is rec else 1
            push(OracleAnchor(
                "file_line_function" if fr.function else "file_line", "oracle",
                fr.repo_path, fr.line, fr.function, fr.anchor, tier))
            if fr.repo_path not in paths:
                paths.append(fr.repo_path)
    for r in order:
        if r is None:
            continue
        for fr in r.frames:
            if fr.in_repo and fr.noise != NOISE_NONE:
                push(OracleAnchor("file_line", "oracle", fr.repo_path, fr.line,
                                  fr.function, "%s:%d" % (fr.repo_path, fr.line),
                                  2))
                if fr.function:
                    push(OracleAnchor("file_function", "oracle", fr.repo_path,
                                      None, fr.function,
                                      "%s::%s" % (fr.repo_path, fr.function), 3))
                if fr.repo_path not in paths:
                    paths.append(fr.repo_path)
    anchors.sort(key=lambda a: (a.rank_tier, a.anchor))
    if anchors and anchors[0].rank_tier == 0:
        mode, bm25, reason = "oracle_exact", False, "resolved oracle frame exists"
    elif anchors:
        mode, bm25, reason = ("oracle_partial", False,
                              "repository frame resolved without actionable tier")
    else:
        has_frames = any(r and r.frames for r in order)
        mode = "lexical_fallback" if has_frames else "unresolved"
        bm25, reason = True, ("no repository frame resolved"
                              if has_frames else "no frames in excerpt")
    return LocateEvidence(mode, anchors[0].anchor if anchors else None,
                          tuple(anchors), tuple(paths), bm25, reason)


def _automatic_patch_gate(evidence_status, confidence, signal, locate, records):
    if evidence_status != "complete":
        return False, "parse_status=%s" % evidence_status
    if confidence == "low":
        return False, "ambiguous primary traceback selection"
    if signal is None or signal.route_id is None:
        return False, "no registered exception route"
    if locate is None or locate.mode != "oracle_exact":
        return False, "no resolved oracle anchor"
    rec = next((r for r in records if r.traceback_id == signal.traceback_id), None)
    if rec and rec.exception:
        for lit in rec.exception.message.literals:
            if lit.is_secret_candidate:
                return False, "security-sensitive message literal present"
    required = REQUIRED_EVIDENCE.get(signal.family, ())
    if required and not signal.grounded_tokens and "grounded" in " ".join(required):
        return False, "route requires grounded evidence that is absent"
    return True, None


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


def ingest(input_data) -> TracebackEvidence:
    """Parse a raw excerpt into ``TracebackEvidence``. Pure and deterministic:
    same text + same RepoIndex + same version -> identical evidence."""
    if isinstance(input_data, str):
        input_data = TracebackInput(text=input_data)
    text = input_data.text or ""
    sha = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    diagnostics = []

    if len(text.encode("utf-8", "replace")) > input_data.max_input_bytes:
        return TracebackEvidence(
            SCHEMA_VERSION, sha, "over_limit", (), (), (), None, None,
            "unknown", (), None,
            LocateEvidence("unresolved", None, (), (), False, "input over limit"),
            (Diagnostic("over_limit", "input exceeds max_input_bytes"),))

    lines = scan(text)
    blocks = _find_blocks(lines)
    if len(blocks) > input_data.max_tracebacks:
        diagnostics.append(Diagnostic(
            "too_many_tracebacks",
            "excerpt holds %d tracebacks; truncated to %d"
            % (len(blocks), input_data.max_tracebacks)))
        blocks = blocks[:input_data.max_tracebacks]
    if not blocks:
        return TracebackEvidence(
            SCHEMA_VERSION, sha, "no_traceback", (), (), (), None, None,
            "unknown", (), None,
            LocateEvidence("unresolved", None, (), (), True,
                           "no traceback in excerpt"),
            tuple(diagnostics))

    records = tuple(
        _parse_block(lines, span, i, input_data.repo_index,
                     input_data.container_prefixes,
                     input_data.max_frames_per_traceback, diagnostics)
        for i, span in enumerate(blocks))
    links = _resolve_links(records, lines, blocks)
    causal_effects = {l.effect_traceback_id for l in links
                      if l.kind in ("direct_cause", "implicit_context")}
    records = tuple(replace(r, is_chain_leaf=r.traceback_id not in causal_effects)
                    for r in records)
    chains = build_chains(records, links)
    primary, confidence, alternatives = select_primary_chain(chains, records, links)
    signal = select_repair_signal(primary, records)
    locate = build_locate(signal, records, primary)
    status = ("complete" if all(r.parse_status == "complete" for r in records)
              else "partial")
    allowed, refusal = _automatic_patch_gate(status, confidence, signal, locate,
                                             records)
    if signal is not None:
        signal = replace(signal, automatic_patch_allowed=allowed,
                         refusal_reason=refusal)
    return TracebackEvidence(
        SCHEMA_VERSION, sha, status, records, chains, links,
        primary.chain_id if primary else None,
        primary.terminal_traceback_id if primary else None,
        confidence, alternatives, signal, locate, tuple(diagnostics))


# --------------------------------------------------------------------------
# adapters + the consumable seam
# --------------------------------------------------------------------------


def repo_index_from_func_index(fidx, repo_roots=(), generated=(), vendored=()):
    """Build a ``RepoIndex`` from a ``func_index.FuncIndex``.

    ``FuncIndex.by_path`` is keyed by repo-relative path, which is exactly the
    ``repo_files`` vocabulary this module resolves against. Kept as an adapter
    (not an import) so ``tracebacks.py`` has zero dependency on the gym.
    """
    paths = frozenset(getattr(fidx, "by_path", {}) or ())
    return RepoIndex(repo_roots=tuple(repo_roots), repo_files=paths,
                     generated_files=frozenset(generated),
                     vendored_files=frozenset(vendored))


def repo_index_from_paths(paths, repo_roots=(), generated=(), vendored=()):
    return RepoIndex(repo_roots=tuple(repo_roots), repo_files=frozenset(paths),
                     generated_files=frozenset(generated),
                     vendored_files=frozenset(vendored))


def traceback_signal(text, repo_index=None, container_prefixes=()):
    """THE seam. Raw log text -> locate anchors + family hints + symptom tokens.

    Returns a plain dict so a downstream consumer (the fixer, a locate ranker,
    a SWE-bench harness) needs no import of this module's dataclasses:

        {
          "status":            complete|partial|no_traceback|over_limit,
          "confidence":        high|medium|low|unknown,
          "primary_anchor":    "app/users.py:88:load_user" | None,
          "anchors":           [{"anchor","repo_path","line","function","tier",
                                 "authority"}, ...]   # tier 0 = deepest
                                                      # actionable in-repo frame
          "repo_paths":        ["app/users.py", ...],
          "family":            routing family for the repair signal,
          "families":          every routed family in the excerpt, best first,
          "route_id":          e.g. "integrity.unique" | None,
          "exception_class":   qualified class of the repair signal,
          "tokens":            grounded, retrieval-safe symptom strings,
          "route_tokens":      route-derived vocabulary (NOT grounded),
          "bm25_allowed":      False whenever any frame anchor resolved,
          "automatic_patch_allowed": bool,
          "refusal_reason":    str | None,
          "prohibited_assumptions": tuple,
          "evidence":          the full TracebackEvidence,
        }

    ``bm25_allowed=False`` is the load-bearing field: a resolved repository
    frame is authoritative, and lexical retrieval must not override it.
    """
    idx = repo_index if repo_index is not None else EMPTY_INDEX
    if not isinstance(idx, RepoIndex):
        idx = repo_index_from_paths(idx)
    ev = ingest(TracebackInput(text=text or "", repo_index=idx,
                               container_prefixes=tuple(container_prefixes)))
    sig, loc = ev.repair_signal, ev.locate
    rec = ev.by_id(sig.traceback_id) if sig else None
    families, seen = [], set()
    if sig and sig.family not in seen:
        families.append(sig.family)
        seen.add(sig.family)
    for r in sorted((r for r in ev.tracebacks if r.route),
                    key=lambda r: -r.route.priority):
        if r.route.family not in seen:
            families.append(r.route.family)
            seen.add(r.route.family)
    return {
        "status": ev.parse_status,
        "confidence": ev.selection_confidence,
        "primary_anchor": loc.primary_anchor if loc else None,
        "anchors": [{"anchor": a.anchor, "repo_path": a.repo_path,
                     "line": a.line, "function": a.function,
                     "tier": a.rank_tier, "authority": a.authority}
                    for a in (loc.anchors if loc else ())],
        "repo_paths": list(loc.resolved_repo_paths) if loc else [],
        "family": sig.family if sig else UNKNOWN_ROUTE_FAMILY,
        "families": families,
        "route_id": sig.route_id if sig else None,
        "exception_class": sig.exception_class if sig else None,
        "tokens": list(sig.grounded_tokens) if sig else [],
        "route_tokens": [t.value for t in route_subtype_tokens(
            rec.route if rec else None)],
        "bm25_allowed": loc.bm25_allowed if loc else True,
        "automatic_patch_allowed": bool(sig and sig.automatic_patch_allowed),
        "refusal_reason": sig.refusal_reason if sig else "no traceback",
        "prohibited_assumptions": (rec.route.prohibited_assumptions
                                   if rec and rec.route else ()),
        "evidence": ev,
    }


#: Published seam, mirroring `block_holes.BLOCK_HOLE_SEAM`.
#:
#: One function, one dict, no gym coupling. The caller supplies raw log text
#: and (optionally) the repo's file list; it gets back ranked locate anchors,
#: candidate repair families, and grounded symptom tokens. Anything the excerpt
#: does not prove is absent from the dict rather than guessed: `primary_anchor`
#: is None, `route_id` is None, `bm25_allowed` flips True, and
#: `automatic_patch_allowed` is False with a `refusal_reason`.
TRACEBACK_SEAM = (
    "tracebacks.traceback_signal(text, repo_index=None, container_prefixes=()) "
    "-> dict(status, confidence, primary_anchor, anchors, repo_paths, family, "
    "families, route_id, exception_class, tokens, route_tokens, bm25_allowed, "
    "automatic_patch_allowed, refusal_reason, prohibited_assumptions, evidence)")

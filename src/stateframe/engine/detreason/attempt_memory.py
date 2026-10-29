"""Deterministic cross-run failure memory for the repair kernel.

This module reads only agent-world artifacts: transcripts, event tapes, and tool
receipts.  It never receives a corpus row or a source-of-truth overlay.  The
result is an append-only, provenance-labelled ``AttemptMemoryV1`` record that a
caller may render into the next attempt's advisory brief.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeAlias, TypeVar

from .attempt_capsule import capsule_digest, render_capsule, render_prior_attempts


SCHEMA = "poc.attempt_memory/1"
AST_NORMALIZER = "python-3.x/ast-v2"
AGENT_WORLD_PROVENANCE = "AGENT_WORLD_ONLY"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | Sequence["JsonValue"]
Record: TypeAlias = Mapping[str, object]
GradeResult = TypeVar("GradeResult")

_MAX_EDITS = 512
_MAX_THEORIES = 64
_MAX_FAILED_LEGS = 64
_MAX_ATOMS = 64
_MAX_FRONTIER = 32

_FORBIDDEN_FIELDS = frozenset(
    {
        "answer",
        "answer_key",
        "answer_keys",
        "diff",
        "diff_text",
        "expected_patch",
        "fixed_source",
        "fixed_src",
        "gold",
        "gold_source",
        "gold_src",
        "golden",
        "hidden_overlay",
        "hidden_source",
        "hidden_tests",
        "patch",
        "patch_text",
        "reference_patch",
        "solution_patch",
    }
)

_ALLOWED_TASK_KEYS = (
    "task_id",
    "id",
    "fn",
    "target_symbol",
    "target_file",
    "path",
    "rel",
    "repo",
    "sha",
    "task_sha",
    "campaign",
    "attempt_no",
    "attempt",
    "run_id",
    "tag",
)

_DEFAULT_ATTEMPT_ORDER = {
    "r9advance2": 1,
    "r9advT": 2,
    "r9advN": 3,
    "r9advS": 4,
    "r9advS2": 5,
    "r9advS3": 6,
}

_EDIT_TOOLS = frozenset(
    {
        "apply_patch",
        "edit",
        "file_create",
        "file_write",
        "patch",
        "replace",
        "str_replace",
        "write_file",
    }
)
_TEST_TOOLS = frozenset({"run_tests", "test", "pytest", "run_test", "check"})
_READ_TOOLS = frozenset({"read", "cat", "open", "view"})
_GREP_TOOLS = frozenset({"grep", "rg", "search", "find"})

_VOLATILE_FIELDS = frozenset(
    {"created_at", "finished_at", "run_started_at", "timestamp", "ts", "updated_at"}
)

_THEORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("state-row", ("state-row", "state row", "job-row", "job row")),
    ("error_code", ("error_code", "error code")),
    ("attempts", ("attempts=", "attempts ", "attempt count")),
    ("cost_spent", ("cost_spent", "cost spent")),
    ("return-label", ("return label", "return value", "label-only", "label only")),
    ("re-arm", ("re-arm", "rearm", "re-armed", "rearmed", "awaiting_dispatch_recovery")),
    ("sr-none", ("sr is none", "sr=none", "sr=None", "no active sr")),
    ("no-active-stage", ("no-active-stage", "no active stage")),
    ("never-dispatched", ("never-dispatched", "never_dispatched", "never launched")),
    ("dispatch-recovery", ("dispatch recovery", "dispatch_recovery")),
    ("logging", ("logging",)),
)

_UNTRUSTED_PROVENANCE_MARKERS = frozenset(
    {"gold", "golden", "hidden", "fixed", "grader", "answer", "solution", "overlay"}
)


class ForbiddenAgentWorldFieldError(ValueError):
    """Raised when an artifact tries to carry answer-key material."""


def _normal_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _forbidden_paths(value: object, prefix: str = "") -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if _normal_key(key_text) in _FORBIDDEN_FIELDS:
                found.append(path)
            found.extend(_forbidden_paths(child, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return tuple(found)


def validate_agent_world_payload(value: object) -> object:
    """Validate a decoded agent-world payload and return it unchanged."""

    paths = _forbidden_paths(value)
    if paths:
        raise ForbiddenAgentWorldFieldError(
            "forbidden answer-key field in agent-world payload: " + ", ".join(paths[:8])
        )
    return value


def _agent_world_record(value: Mapping[str, object]) -> bool:
    """Accept only rows whose oracle metadata is not gold/grader-derived."""

    if _bool(value.get("agent_world")) is False:
        return False
    for key in ("provenance", "oracle_provenance", "source", "oracle", "basis"):
        rendered = _normal_key(value.get(key, ""))
        if any(marker in rendered.split("_") for marker in _UNTRUSTED_PROVENANCE_MARKERS):
            return False
    return True


def agent_world_identity(task: Mapping[str, object]) -> dict[str, object]:
    """Copy only scalar task identity fields, never source or grading fields.

    The allowlist is intentionally read by name; an object exposing a hostile
    ``fixed_src`` accessor is therefore never asked for that value.
    """

    if not isinstance(task, Mapping):
        raise TypeError("task identity must be a mapping")
    identity: dict[str, object] = {}
    aliases = (
        ("task_id", ("task_id", "id")),
        ("target_symbol", ("target_symbol", "fn")),
        ("target_file", ("target_file", "path", "rel")),
        ("repo", ("repo",)),
        ("task_sha", ("task_sha", "sha")),
        ("campaign", ("campaign",)),
        ("attempt_no", ("attempt_no", "attempt")),
        ("run_id", ("run_id",)),
        ("tag", ("tag",)),
    )
    for output_key, keys in aliases:
        for key in keys:
            value = task.get(key)
            if isinstance(value, (str, int, float, bool)) and str(value).strip():
                identity[output_key] = value
                break
    return identity


build_agent_world_identity = agent_world_identity


def _jsonable(value: object, *, drop_volatile: bool = False) -> object:
    if isinstance(value, Mapping):
        pairs = []
        for key, child in value.items():
            key_text = str(key)
            if drop_volatile and _normal_key(key_text) in _VOLATILE_FIELDS:
                continue
            pairs.append((key_text, _jsonable(child, drop_volatile=drop_volatile)))
        return {key: child for key, child in sorted(pairs)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(child, drop_volatile=drop_volatile) for child in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _canonical(value: object, *, drop_volatile: bool = False) -> str:
    return json.dumps(
        _jsonable(value, drop_volatile=drop_volatile),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _digest(value: object, *, drop_volatile: bool = False) -> str:
    return hashlib.sha256(
        _canonical(value, drop_volatile=drop_volatile).encode("utf-8")
    ).hexdigest()


def _text(value: object, limit: int = 400) -> str:
    rendered = str(value or "").replace("\r", " ").replace("\n", " ")
    return " ".join(rendered.split())[:limit]


def _mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        candidate = to_dict()
        if isinstance(candidate, Mapping):
            return candidate
    return None


def _records(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _values(value: object) -> tuple[object, ...]:
    """Treat scalar text as one value instead of iterating its characters."""

    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _record_copy(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): child for key, child in value.items()}


def _record_tuple(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_record_copy(item) for item in _records(value))


def _first_text(value: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        item = value.get(key)
        if item is not None and str(item).strip():
            return str(item).strip()
    return ""


def _payload_sources(value: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    """Expose identity envelopes without widening the source-data boundary."""

    sources: list[Mapping[str, object]] = [value]
    for key in ("key", "world", "task", "identity", "metadata", "context"):
        nested = _mapping(value.get(key))
        if nested is not None and nested not in sources:
            sources.append(nested)
    return tuple(sources)


def _payload_text(value: Mapping[str, object], *keys: str) -> str:
    for source in _payload_sources(value):
        found = _first_text(source, *keys)
        if found:
            return found
    return ""


def _int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "green", "proven"}:
            return True
        if lowered in {"false", "no", "0", "red", "unknown"}:
            return False
    return None


def _normal_path(value: object) -> str:
    rendered = _text(value, 300).replace("\\", "/")
    if rendered.startswith("/"):
        rendered = Path(rendered).name
    return rendered


def _results_dir(artifact_root: str | Path) -> Path:
    root = Path(artifact_root)
    return root if root.name == "results" else root / "results"


def _tag_value(tag: str | None) -> str:
    value = str(tag or "").strip()
    value = value[1:] if value.startswith("_") else value
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("tag must be a single artifact directory name")
    return value


def _artifact_name(prefix: str, tag: str, task_fn: str, suffix: str) -> Path:
    dirname = prefix if not tag else f"{prefix}_{tag}"
    return Path(dirname) / f"{task_fn}{suffix}"


def discover_tags(artifact_root: str | Path, task_fn: str) -> tuple[str, ...]:
    """Return tags with at least one matching transcript or receipt file."""

    results = _results_dir(artifact_root)
    tags: set[str] = set()
    for prefix, suffix in (("react_transcripts", ".json"), ("receipts", ".jsonl")):
        if not results.is_dir():
            continue
        for directory in results.iterdir():
            if not directory.is_dir() or not directory.name.startswith(prefix):
                continue
            tag = directory.name[len(prefix) :].lstrip("_")
            if (directory / f"{task_fn}{suffix}").is_file():
                tags.add(tag)
    return tuple(sorted(tags))


def _read_json(path: Path) -> tuple[Mapping[str, object] | None, str, list[str]]:
    if not path.is_file():
        return None, "", [f"missing:{path.parent.name}/{path.name}"]
    try:
        raw = path.read_bytes()
        decoded = json.loads(raw)
        validate_agent_world_payload(decoded)
    except ForbiddenAgentWorldFieldError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, "", [f"invalid:{path.parent.name}/{path.name}:{type(exc).__name__}"]
    if not isinstance(decoded, Mapping):
        return None, "", [f"invalid:{path.parent.name}/{path.name}:not_mapping"]
    return decoded, _digest(decoded, drop_volatile=True), []


def _read_jsonl(path: Path) -> tuple[tuple[Mapping[str, object], ...], str, list[str]]:
    if not path.is_file():
        return (), "", [f"missing:{path.parent.name}/{path.name}"]
    records: list[Mapping[str, object]] = []
    gaps: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return (), "", [f"invalid:{path.parent.name}/{path.name}:{type(exc).__name__}"]
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
            validate_agent_world_payload(decoded)
        except ForbiddenAgentWorldFieldError:
            raise
        except (json.JSONDecodeError, TypeError) as exc:
            gaps.append(f"invalid:{path.parent.name}/{path.name}:{line_no}:{type(exc).__name__}")
            continue
        if not isinstance(decoded, Mapping):
            gaps.append(f"invalid:{path.parent.name}/{path.name}:{line_no}:not_mapping")
            continue
        records.append(decoded)
    digest_input = {"path": f"{path.parent.name}/{path.name}", "records": records}
    return tuple(records), _digest(digest_input, drop_volatile=True), gaps


def _event_path(transcript_path: Path) -> Path:
    event_dir = transcript_path.parent.parent / transcript_path.parent.name.replace(
        "react_transcripts", "react_events"
    )
    return event_dir / f"{transcript_path.stem}.jsonl"


def _message_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if isinstance(value.get("text"), str):
            return str(value["text"])
        if isinstance(value.get("content"), str):
            return str(value["content"])
        return _text(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "\n".join(_message_text(item) for item in value)
    return _text(value)


def _trim_marker(message: Mapping[str, object]) -> bool:
    content = _message_text(message.get("content"))
    return (
        _text(message.get("role"), 20).lower() == "user"
        and content.startswith("[kernel] (")
        and "trimmed to fit context" in content
    )


def _reconstruct_messages(
    transcript_path: Path,
    messages: Sequence[Mapping[str, object]],
    events: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    trimmed = [
        message
        for event in events
        if _text(event.get("kind"), 40).lower() == "trimmed"
        and isinstance((message := event.get("msg")), Mapping)
        and not _trim_marker(message)
    ]
    if not trimmed:
        return list(messages)
    head = list(messages[:2])
    tail = [message for message in messages[2:] if not _trim_marker(message)]
    return [*head, *trimmed, *tail]


@dataclass(frozen=True)
class AttemptContext:
    """Immutable context a future retry orchestrator can pass to the weld."""

    campaign: str
    task_id: str
    attempt_no: int
    run_id: str
    temperature: float | None
    artifact_root: str

    def __post_init__(self) -> None:
        if not self.campaign.strip() or not self.task_id.strip() or not self.run_id.strip():
            raise ValueError("campaign, task_id, and run_id are required")
        if self.attempt_no < 1:
            raise ValueError("attempt_no must be positive")
        object.__setattr__(self, "artifact_root", str(Path(self.artifact_root)))

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign": self.campaign,
            "task_id": self.task_id,
            "attempt_no": self.attempt_no,
            "run_id": self.run_id,
            "temperature": self.temperature,
            "artifact_root": self.artifact_root,
        }


@dataclass(frozen=True)
class AttemptMemoryV1:
    """One deterministic, provenance-labelled attempt record."""

    key: Mapping[str, object]
    world: Mapping[str, object]
    edits: tuple[Mapping[str, object], ...] = ()
    theories: tuple[Mapping[str, object], ...] = ()
    failed_legs: tuple[Mapping[str, object], ...] = ()
    atoms: tuple[Mapping[str, object], ...] = ()
    terminal: Mapping[str, object] = field(default_factory=dict)
    frontier: tuple[Mapping[str, object], ...] = ()
    audit: Mapping[str, object] = field(default_factory=dict)
    verdict: str = "COMPLETE"
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"unsupported attempt memory schema: {self.schema}")
        if self.verdict not in {"COMPLETE", "PARTIAL"}:
            raise ValueError("attempt memory verdict must be COMPLETE or PARTIAL")
        for section in (
            self.key,
            self.world,
            self.edits,
            self.theories,
            self.failed_legs,
            self.atoms,
            self.terminal,
            self.frontier,
            self.audit,
        ):
            validate_agent_world_payload(section)
        object.__setattr__(self, "key", _record_copy(self.key))
        object.__setattr__(self, "world", _record_copy(self.world))
        object.__setattr__(self, "edits", _record_tuple(self.edits))
        object.__setattr__(self, "theories", _record_tuple(self.theories))
        object.__setattr__(self, "failed_legs", _record_tuple(self.failed_legs))
        object.__setattr__(self, "atoms", _record_tuple(self.atoms))
        object.__setattr__(self, "terminal", _record_copy(self.terminal))
        object.__setattr__(self, "frontier", _record_tuple(self.frontier))
        object.__setattr__(self, "audit", _record_copy(self.audit))

    @property
    def attempt_no(self) -> int:
        value = _int(self.key.get("attempt_no"))
        return value or 0

    @property
    def tag(self) -> str:
        return _text(self.key.get("tag"), 120)

    @property
    def seal(self) -> str:
        return _text(self.terminal.get("seal"), 40).upper()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "verdict": self.verdict,
            "key": _jsonable(self.key),
            "world": _jsonable(self.world),
            "edits": _jsonable(self.edits),
            "theories": _jsonable(self.theories),
            "failed_legs": _jsonable(self.failed_legs),
            "atoms": _jsonable(self.atoms),
            "terminal": _jsonable(self.terminal),
            "frontier": _jsonable(self.frontier),
            "audit": _jsonable(self.audit),
        }

    def serialize(self) -> str:
        return _canonical(self.to_dict())

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]


@dataclass(frozen=True)
class _Action:
    turn: int
    name: str
    kind: str
    args: Mapping[str, object]
    raw_args: str
    assistant_text: str
    observation: str


@dataclass(frozen=True)
class _Receipt:
    line: int
    record: Mapping[str, object]

    @property
    def kind(self) -> str:
        return _text(self.record.get("kind"), 40).lower()

    @property
    def result_class(self) -> str:
        return _text(self.record.get("result_class"), 50).upper()

    @property
    def turn(self) -> int | None:
        return _int(self.record.get("turn"))


def _decode_arguments(value: object) -> tuple[Mapping[str, object], str]:
    if isinstance(value, Mapping):
        return value, _canonical(value)
    raw = str(value or "")
    try:
        decoded = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"_unparsed": True}, raw
    if not isinstance(decoded, Mapping):
        return {"_unparsed": True}, raw
    return decoded, raw


def _tool_call_parts(call: Mapping[str, object]) -> tuple[str, Mapping[str, object], str]:
    fn = call.get("function")
    function = fn if isinstance(fn, Mapping) else call
    name = _text(function.get("name"), 80).lower()
    args, raw = _decode_arguments(function.get("arguments"))
    return name, args, raw


def _action_kind(name: str) -> str:
    if name in _EDIT_TOOLS:
        return "edit"
    if name in _TEST_TOOLS:
        return "test"
    if name in _READ_TOOLS:
        return "read"
    if name in _GREP_TOOLS:
        return "grep"
    if name == "done":
        return "finish"
    return "other"


def _actions(messages: Sequence[Mapping[str, object]]) -> tuple[tuple[_Action, ...], tuple[str, ...]]:
    actions: list[_Action] = []
    prose: list[str] = []
    current: Mapping[str, object] | None = None
    observations: list[Mapping[str, object]] = []
    turn = -1

    def flush() -> None:
        nonlocal current, observations, turn
        if current is None:
            return
        calls = current.get("tool_calls")
        call_list = calls if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)) else ()
        assistant_text = _message_text(current.get("content"))
        if assistant_text.strip():
            prose.append(assistant_text)
        observation = "\n".join(_message_text(item.get("content")) for item in observations)
        for call in call_list:
            if not isinstance(call, Mapping):
                continue
            name, args, raw_args = _tool_call_parts(call)
            actions.append(
                _Action(
                    turn=turn,
                    name=name,
                    kind=_action_kind(name),
                    args=args,
                    raw_args=raw_args,
                    assistant_text=assistant_text,
                    observation=observation,
                )
            )
        current = None
        observations = []

    for message in messages:
        role = _text(message.get("role"), 20).lower()
        if role == "assistant":
            flush()
            current = message
            turn += 1
        elif role in {"tool", "user"} and current is not None:
            observations.append(message)
    flush()
    return tuple(actions), tuple(prose)


def _receipt_objects(records: Sequence[Mapping[str, object]]) -> tuple[_Receipt, ...]:
    return tuple(_Receipt(line=index, record=record) for index, record in enumerate(records, start=1))


def _receipt_kind(record: Mapping[str, object]) -> str:
    kind = _text(record.get("kind"), 40).lower()
    if kind:
        return kind
    intent = _text(record.get("intent"), 120).lower()
    if any(tool in intent for tool in _EDIT_TOOLS):
        return "edit"
    if any(tool in intent for tool in _TEST_TOOLS):
        return "test"
    return "other"


def _receipts_by_kind(receipts: Sequence[_Receipt], kind: str) -> tuple[_Receipt, ...]:
    return tuple(receipt for receipt in receipts if _receipt_kind(receipt.record) == kind)


def _arg_text(args: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = args.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _source_pair(args: Mapping[str, object], receipt: Mapping[str, object] | None) -> tuple[str, str]:
    values = [args, receipt or {}]
    old_keys = ("before_source", "old_str", "old", "before")
    new_keys = ("after_source", "new_str", "new", "replacement", "after")
    old = next((_arg_text(value, *old_keys) for value in values if _arg_text(value, *old_keys)), "")
    new = next((_arg_text(value, *new_keys) for value in values if _arg_text(value, *new_keys)), "")
    return old, new


def _parse_ast(source: str) -> ast.AST | None:
    if not source.strip():
        return None
    candidates = (source, textwrap.dedent(source))
    for candidate in candidates:
        try:
            return ast.parse(candidate)
        except (SyntaxError, ValueError, TypeError):
            continue
    return None


def _ast_dump(tree: ast.AST | None) -> str:
    if tree is None:
        return ""
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _node_dump(value: object) -> str:
    return _ast_dump(value if isinstance(value, ast.AST) else None)


def _changed_ast_paths(before: ast.AST, after: ast.AST) -> tuple[str, ...]:
    changed: list[str] = []

    def visit(left: object, right: object, path: str) -> None:
        if isinstance(left, ast.AST) and isinstance(right, ast.AST):
            if _node_dump(left) == _node_dump(right):
                return
        elif left == right:
            return
        if isinstance(left, ast.AST) and isinstance(right, ast.AST) and type(left) is type(right):
            fields = tuple(dict.fromkeys((*left._fields, *right._fields)))
            for field_name in fields:
                left_value = getattr(left, field_name, None)
                right_value = getattr(right, field_name, None)
                if isinstance(left_value, list) or isinstance(right_value, list):
                    left_list = left_value if isinstance(left_value, list) else []
                    right_list = right_value if isinstance(right_value, list) else []
                    for index in range(max(len(left_list), len(right_list))):
                        visit(
                            left_list[index] if index < len(left_list) else None,
                            right_list[index] if index < len(right_list) else None,
                            f"{path}.{field_name}[{index}]",
                        )
                else:
                    visit(left_value, right_value, f"{path}.{field_name}")
            return
        if path not in changed:
            changed.append(path)

    visit(before, after, "module")
    return tuple(changed[:32]) or ("module",)


def ast_delta_fingerprint(
    *,
    path: str,
    scope: str,
    before_source: str,
    after_source: str,
) -> tuple[str | None, str, str, str]:
    """Return AST delta fp, before AST fp, after AST fp, and capture quality."""

    before_tree = _parse_ast(before_source)
    after_tree = _parse_ast(after_source)
    if before_tree is None or after_tree is None:
        return None, "", "", "LEGACY_CLIPPED"
    before_dump = _ast_dump(before_tree)
    after_dump = _ast_dump(after_tree)
    material = {
        "normalizer": AST_NORMALIZER,
        "path": _normal_path(path),
        "scope": _text(scope, 160),
        "changed_ast_paths": _changed_ast_paths(before_tree, after_tree),
        "before": before_dump,
        "after": after_dump,
    }
    return (
        _digest(material),
        _digest(before_dump),
        _digest(after_dump),
        "REPLAYED_FULL",
    )


def _edit_status(action: _Action | None, receipt: _Receipt | None) -> str:
    if receipt is not None:
        result = receipt.result_class
        if result in {"APPLIED", "GREEN", "PASS", "SUCCESS"}:
            return "APPLIED"
        if result in {"REJECTED", "NO_OP", "FAILED", "ERROR"}:
            return result
    text = "\n".join(
        part
        for part in (
            action.observation if action else "",
            _text(receipt.record.get("detail"), 500) if receipt else "",
        )
        if part
    ).lower()
    if any(marker in text for marker in ("rejected", "no-op", "no op", "not applied")):
        return "REJECTED"
    if any(marker in text for marker in ("applied", "success", "created", "written")):
        return "APPLIED"
    return "UNKNOWN"


def _edit_records(
    actions: Sequence[_Action],
    receipts: Sequence[_Receipt],
    tag: str,
    task_fn: str,
) -> tuple[dict[str, object], ...]:
    edit_actions = [action for action in actions if action.kind == "edit"]
    edit_receipts = list(_receipts_by_kind(receipts, "edit"))
    count = max(len(edit_actions), len(edit_receipts))
    output: list[dict[str, object]] = []
    for index in range(min(count, _MAX_EDITS)):
        action = edit_actions[index] if index < len(edit_actions) else None
        receipt = edit_receipts[index] if index < len(edit_receipts) else None
        args = action.args if action else {}
        receipt_map = receipt.record if receipt else {}
        path = _normal_path(_arg_text(args, "path", "file", "target") or _arg_text(
            receipt_map, "path", "file", "target"
        ))
        scope = _arg_text(args, "scope", "symbol", "function") or task_fn
        operation = action.name if action else _text(receipt_map.get("intent"), 100) or "edit"
        old_source, new_source = _source_pair(args, receipt_map)
        ast_fp, before_ast_fp, after_ast_fp, capture = ast_delta_fingerprint(
            path=path,
            scope=scope,
            before_source=old_source,
            after_source=new_source,
        )
        if ast_fp is not None and action is not None:
            capture = "LIVE_FULL" if "before_source" in args or "after_source" in args else capture
        raw_material = {
            "tool": operation,
            "path": path,
            "scope": scope,
            "old": old_source,
            "new": new_source,
            "receipt_fingerprint": receipt_map.get("fingerprint"),
        }
        operation_fp = _digest(raw_material)
        evidence_refs = [f"transcript:{tag}:{index + 1}"] if action else []
        if receipt is not None:
            evidence_refs.append(f"receipt:{tag}:{receipt.line}")
        output.append(
            {
                "seq": index + 1,
                "turn": receipt.turn if receipt and receipt.turn is not None else (action.turn if action else None),
                "path": path,
                "scope": scope,
                "operation": operation,
                "status": _edit_status(action, receipt),
                "before_file_sha256": _text(
                    args.get("before_file_sha256") or receipt_map.get("before_file_sha256"), 100
                ) or None,
                "after_file_sha256": _text(
                    args.get("after_file_sha256") or receipt_map.get("after_file_sha256"), 100
                ) or None,
                "before_ast_sha256": before_ast_fp or None,
                "after_ast_sha256": after_ast_fp or None,
                "ast_delta_fp": ast_fp,
                "operation_fp": operation_fp,
                "capture": capture,
                "evidence_refs": tuple(evidence_refs),
            }
        )
    return tuple(output)


def _theory_label_set(texts: Iterable[str]) -> tuple[str, ...]:
    combined = "\n".join(texts)
    lowered = combined.lower()
    labels: list[str] = []
    for label, markers in _THEORY_RULES:
        if any(marker.lower() in lowered for marker in markers):
            labels.append(label)
    if {"status", "attempts", "cost_spent"}.issubset(set(re.findall(r"[a-z_]+", lowered))):
        if "state-row" not in labels:
            labels.insert(0, "state-row")
    return tuple(labels)


def _explicit_theories(payload: Mapping[str, object]) -> list[dict[str, object]]:
    values: list[Mapping[str, object]] = []
    values.extend(_records(payload.get("theories")))
    kernel = _mapping(payload.get("kernel"))
    if kernel:
        values.extend(_records(kernel.get("theories")))
    output: list[dict[str, object]] = []
    for item in values:
        if not _agent_world_record(item):
            continue
        claim = _text(item.get("claim") or item.get("theory") or item.get("render"), 240)
        if not claim:
            continue
        status = _text(item.get("status"), 40).upper() or "UNVERIFIED"
        if status not in {"UNVERIFIED", "SUPPORTED", "CONTRADICTED"}:
            status = "UNVERIFIED"
        output.append(
            {
                "fp": _text(item.get("fp"), 100) or _digest(claim),
                "claim": claim,
                "source": _text(item.get("source"), 60).upper() or "MODEL_DECLARATION",
                "status": status,
                "evidence_refs": tuple(_text(ref, 160) for ref in _values(item.get("evidence_refs"))
                                        if _text(ref, 160)),
            }
        )
    return output


def _theories(
    payload: Mapping[str, object],
    actions: Sequence[_Action],
    receipts: Sequence[_Receipt],
    edits: Sequence[Mapping[str, object]],
    prose: Sequence[str],
) -> tuple[dict[str, object], ...]:
    output = _explicit_theories(payload)
    edit_actions = [action for action in actions if action.kind == "edit"]
    edit_receipts = list(_receipts_by_kind(receipts, "edit"))
    texts: list[str] = []
    for index in range(max(len(edit_actions), len(edit_receipts))):
        if index < len(edit_actions):
            action = edit_actions[index]
            texts.extend(
                [
                    _arg_text(action.args, "why", "reason", "theory", "claim"),
                    _arg_text(action.args, "old_str", "new_str", "old", "new"),
                ]
            )
        if index < len(edit_receipts):
            texts.append(_text(edit_receipts[index].record.get("why"), 500))
    if not edit_actions and not edit_receipts:
        texts.extend(prose)
    labels = _theory_label_set(text for text in texts if text)
    for label in labels:
        output.append(
            {
                "fp": _digest({"claim": label, "source": "MODEL_DECLARATION"}),
                "claim": label,
                "source": "MODEL_DECLARATION",
                "status": "UNVERIFIED",
                "evidence_refs": (),
            }
        )
    if not output and edits:
        paths = sorted({_text(edit.get("path"), 120) for edit in edits if _text(edit.get("path"), 120)})
        if paths:
            claim = "edit target: " + ", ".join(paths[:3])
            output.append(
                {
                    "fp": _digest(claim),
                    "claim": claim,
                    "source": "KERNEL_INFERRED",
                    "status": "UNVERIFIED",
                    "evidence_refs": (),
                }
            )
    deduped: dict[str, dict[str, object]] = {}
    for item in output:
        claim = _text(item.get("claim"), 240)
        if claim:
            deduped.setdefault(claim.lower(), item)
    return tuple(
        deduped[key]
        for key in sorted(deduped)
    )[:_MAX_THEORIES]


def _exit_code(text: str) -> int | None:
    match = re.search(r"pytest\s+exit\s+(-?\d+)", text, re.IGNORECASE)
    return _int(match.group(1)) if match else None


def _result_class(text: str, receipt: _Receipt | None) -> str:
    if receipt is not None and receipt.result_class:
        return receipt.result_class
    if re.search(r"pytest\s+exit\s+0", text, re.IGNORECASE):
        return "GREEN"
    if re.search(r"pytest\s+exit\s+-?[1-9]", text, re.IGNORECASE):
        return "RED"
    if any(marker in text.upper() for marker in ("NO_VERDICT", "NO TEST ORACLE", "TIMED OUT")):
        return "NO_VERDICT"
    return "UNKNOWN"


def _failure_anchor(text: str, fallback: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    for line in text.splitlines():
        stripped = " ".join(line.split())
        if not stripped:
            continue
        if (
            stripped.startswith("E ")
            or "AssertionError" in stripped
            or stripped.startswith("assert ")
        ):
            return stripped[:220]
    for stripped in lines:
        if stripped.startswith("FAILED "):
            return stripped[:220]
    return _text(fallback, 220)


def _failed_legs(
    actions: Sequence[_Action],
    receipts: Sequence[_Receipt],
    tag: str,
) -> tuple[dict[str, object], ...]:
    test_actions = [action for action in actions if action.kind == "test"]
    test_receipts = list(_receipts_by_kind(receipts, "test"))
    count = max(len(test_actions), len(test_receipts))
    output: list[dict[str, object]] = []
    for index in range(count):
        action = test_actions[index] if index < len(test_actions) else None
        receipt = test_receipts[index] if index < len(test_receipts) else None
        observation = action.observation if action else ""
        receipt_map = receipt.record if receipt else {}
        detail = _text(receipt_map.get("detail"), 600)
        why = _text(receipt_map.get("why"), 300)
        combined = "\n".join(part for part in (observation, detail) if part)
        result_class = _result_class(combined, receipt)
        if result_class != "RED":
            continue
        if any(marker in combined.upper() for marker in ("TIMED OUT", "NO TEST ORACLE", "HARNESS")):
            continue
        args = action.args if action else {}
        selector = _arg_text(args, "selector", "test", "tests", "path", "target")
        selector = selector or _arg_text(receipt_map, "selector", "test", "tests", "intent") or "test"
        exit_code = _exit_code(combined) or _int(receipt_map.get("exit_code")) or 1
        anchor = _failure_anchor(combined, why or selector)
        fp = _digest({"selector": selector, "exit": exit_code, "anchor": anchor})
        refs = []
        if action:
            refs.append(f"transcript:{tag}:{index + 1}")
        if receipt:
            refs.append(f"receipt:{tag}:{receipt.line}")
        output.append(
            {
                "fp": fp,
                "test_selector": _text(selector, 220),
                "exit": exit_code,
                "anchor": anchor,
                "turn": receipt.turn if receipt and receipt.turn is not None else (action.turn if action else None),
                "genuine_red": True,
                "evidence_refs": tuple(refs),
            }
        )
    deduped: dict[str, dict[str, object]] = {}
    for item in sorted(output, key=lambda row: (_int(row.get("turn")) or 0, _text(row.get("fp"), 100))):
        deduped.setdefault(_text(item.get("fp"), 100), item)
    return tuple(deduped.values())[:_MAX_FAILED_LEGS]


def _atom_record(item: Mapping[str, object], evidence_ref: str) -> dict[str, object] | None:
    if not _agent_world_record(item):
        return None
    kind = _text(item.get("kind") or item.get("type"), 80)
    if not kind:
        return None
    state = _text(item.get("state"), 40).upper() or "UNKNOWN"
    if state not in {"SUPPORTED", "CONTRADICTED", "UNKNOWN"}:
        state = "UNKNOWN"
    basis = _text(item.get("basis"), 80).upper() or "RECEIPT"
    oracle = _text(item.get("oracle"), 40).upper() or "NONE"
    expected = _jsonable(item.get("expected"))
    observed = _jsonable(item.get("observed"))
    fp = _text(item.get("fp"), 100) or _digest(
        {"kind": kind, "expected": expected, "observed": observed, "state": state}
    )
    refs = [_text(ref, 180) for ref in _values(item.get("evidence_refs")) if _text(ref, 180)]
    if evidence_ref not in refs:
        refs.append(evidence_ref)
    return {
        "fp": fp,
        "kind": kind,
        "expected": expected,
        "observed": observed,
        "state": state,
        "basis": basis,
        "oracle": oracle,
        "evidence_refs": tuple(refs),
    }


def _atoms(
    payload: Mapping[str, object],
    receipts: Sequence[_Receipt],
    tag: str,
) -> tuple[dict[str, object], ...]:
    candidates: list[tuple[Mapping[str, object], str]] = []
    candidates.extend((item, f"transcript:{tag}:atom") for item in _records(payload.get("atoms")))
    kernel = _mapping(payload.get("kernel"))
    if kernel:
        candidates.extend((item, f"transcript:{tag}:kernel") for item in _records(kernel.get("atoms")))
        for trace in _records(kernel.get("trace")):
            atom = trace.get("atom")
            if isinstance(atom, Mapping):
                candidates.append((atom, f"transcript:{tag}:trace"))
            candidates.extend((item, f"transcript:{tag}:trace") for item in _records(trace.get("atoms")))
    for receipt in receipts:
        atom = receipt.record.get("atom")
        if isinstance(atom, Mapping):
            candidates.append((atom, f"receipt:{tag}:{receipt.line}"))
    output: dict[str, dict[str, object]] = {}
    for item, ref in candidates:
        normalized = _atom_record(item, ref)
        if normalized:
            output.setdefault(_text(normalized.get("fp"), 100), normalized)
    return tuple(output[key] for key in sorted(output))[:_MAX_ATOMS]


def _result_value(item: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, Mapping):
            nested = _first_text(value, "result_class", "verdict", "result", "class")
            if nested:
                return nested.upper()
        elif value is not None and str(value).strip():
            return str(value).strip().upper()
    return ""


def _frontier_candidate(item: Mapping[str, object]) -> tuple[str, str]:
    evidence = _records(item.get("evidence"))
    candidate = _result_value(item, "candidate_result", "candidate_verdict", "candidate")
    parent = _result_value(item, "parent_result", "parent_verdict", "parent")
    if not candidate and _bool(item.get("candidate_green")) is True:
        candidate = "GREEN"
    if not parent and _bool(item.get("parent_red")) is True:
        parent = "RED"
    for record in evidence:
        result = _result_value(record, "result_class", "verdict", "result", "class")
        side = _text(record.get("side"), 40).lower()
        if side == "candidate" and not candidate:
            candidate = result
        if side == "parent" and not parent:
            parent = result
    return candidate, parent


def _frontier(payload: Mapping[str, object], receipts: Sequence[_Receipt], tag: str) -> tuple[dict[str, object], ...]:
    values: list[Mapping[str, object]] = []
    for key in ("frontier", "proven_frontier", "frontier_rows", "state_rows"):
        values.extend(_records(payload.get(key)))
    kernel = _mapping(payload.get("kernel"))
    if kernel:
        for key in ("frontier", "proven_frontier", "frontier_rows", "state_rows"):
            values.extend(_records(kernel.get(key)))
    for receipt in receipts:
        values.extend(_records(receipt.record.get("frontier")))

    output: dict[str, dict[str, object]] = {}
    for item in values:
        if not _agent_world_record(item):
            continue
        candidate, parent = _frontier_candidate(item)
        proven = _bool(item.get("proven")) is True or _text(item.get("status"), 60).upper() == "PROVEN_UNEXPLORED"
        observed = _bool(item.get("observed")) is True
        if not proven or observed or candidate != "GREEN" or parent != "RED":
            continue
        key = _text(item.get("key") or item.get("id"), 180)
        rendered = _text(item.get("render") or item.get("question") or key, 240)
        if not key or not rendered:
            continue
        basis = tuple(_text(value, 180) for value in _values(item.get("basis")) if _text(value, 180))
        if not basis:
            basis = ("candidate GREEN", "pristine parent RED")
        row = {
            "fp": _text(item.get("fp"), 100) or _digest({"key": key, "render": rendered}),
            "key": key,
            "render": rendered,
            "basis": basis,
            "status": "PROVEN_UNEXPLORED",
            "candidate_result": "GREEN",
            "parent_result": "RED",
            "evidence_refs": (f"transcript:{tag}:frontier",),
        }
        output.setdefault(key, row)
    return tuple(output[key] for key in sorted(output))[:_MAX_FRONTIER]


def _count_kind(actions: Sequence[_Action], receipts: Sequence[_Receipt], kind: str) -> int:
    receipt_count = sum(1 for receipt in receipts if _receipt_kind(receipt.record) == kind)
    action_count = sum(1 for action in actions if action.kind == kind)
    return max(receipt_count, action_count)


def _summary_int(payload: Mapping[str, object], key: str) -> int | None:
    return _int(payload.get(key))


def _terminal(
    payload: Mapping[str, object],
    actions: Sequence[_Action],
    receipts: Sequence[_Receipt],
    transcript_present: bool,
    gaps: Sequence[str],
    transcript_status: str = "COMPLETE",
) -> dict[str, object]:
    edits = _summary_int(payload, "edits")
    tests = _summary_int(payload, "tests")
    reads = _count_kind(actions, receipts, "read")
    greps = _count_kind(actions, receipts, "grep")
    if edits is None:
        edits = sum(1 for action in actions if action.kind == "edit")
        if not edits:
            edits = sum(
                1
                for receipt in receipts
                if _receipt_kind(receipt.record) == "edit"
                and receipt.result_class in {"APPLIED", "GREEN", "PASS", "SUCCESS"}
            )
    if tests is None:
        tests = _count_kind(actions, receipts, "test")
    steps = _summary_int(payload, "steps")
    action_last = max((action.turn for action in actions), default=-1)
    receipt_last = max((receipt.turn or -1 for receipt in receipts), default=-1)
    last_turn = max(action_last, receipt_last)
    if steps is None and transcript_present and last_turn >= 0:
        steps = last_turn + 1
    result = _text(payload.get("result"), 60).upper()
    has_finish = any(action.kind == "finish" for action in actions)
    if not transcript_present:
        shape = "PARTIAL"
    elif result.startswith("ERR") or any("exception" in gap.lower() for gap in gaps):
        shape = "EXCEPTION"
    elif result == "HARNESS_SUSPECT" or "HARNESS_SUSPECT" in result:
        shape = "HARNESS_SUSPECT"
    elif has_finish or result == "PASS":
        shape = "EXPLICIT_DONE"
    elif edits == 0 and tests == 0 and (reads or greps):
        shape = "BUDGET_EXHAUSTED" if (steps or 0) >= 200 else "FAIL_FAST_NO_ACTION"
    elif (steps or 0) >= 200:
        shape = "BUDGET_EXHAUSTED"
    else:
        shape = "EXCEPTION" if not actions and not receipts else "FAIL_FAST_NO_ACTION"
    return {
        "shape": shape,
        "underlying_shape": shape if shape != "PARTIAL" else None,
        "steps": steps,
        "edits": edits,
        "tests": tests,
        "reads": reads,
        "greps": greps,
        "last_turn": last_turn if last_turn >= 0 else None,
        "result": result or None,
        "transcript": transcript_status if transcript_present else "MISSING",
        "seal": "COMPLETE" if transcript_present and not gaps else "PARTIAL",
    }


def _attempt_no_for_tag(
    tag: str,
    explicit: int | None,
    attempt_order: Mapping[str, int] | Sequence[str] | None,
) -> int:
    if explicit is not None and explicit > 0:
        return explicit
    normalized = _tag_value(tag)
    if isinstance(attempt_order, Mapping):
        value = _int(attempt_order.get(normalized))
        if value and value > 0:
            return value
    elif attempt_order is not None:
        ordered = {_tag_value(item): index for index, item in enumerate(attempt_order, start=1)}
        if normalized in ordered:
            return ordered[normalized]
    if normalized in _DEFAULT_ATTEMPT_ORDER:
        return _DEFAULT_ATTEMPT_ORDER[normalized]
    for pattern in (
        r"(?:attempt|try|run)[_-]?(\d+)$",
        r"^r\d+.*?([0-9]+)$",
        r"^[a-z_-]*?(\d+)$",
    ):
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def _artifact_metadata(
    transcript: Mapping[str, object] | None,
    receipts: Sequence[_Receipt],
    task_fn: str,
    task_sha: str,
    tag: str,
) -> tuple[str, str, str, int | None, str]:
    top = transcript or {}
    receipt_first = receipts[0].record if receipts else {}
    task_id = _payload_text(top, "task_id", "id") or _payload_text(
        receipt_first, "task_id", "id"
    )
    run_id = _payload_text(top, "run_id", "attempt_id", "correlation_id") or _payload_text(
        receipt_first, "run_id", "attempt_id", "correlation_id"
    )
    repo = _payload_text(top, "repo") or _payload_text(receipt_first, "repo")
    rel_path = _payload_text(top, "rel_path", "target_file", "path", "file")
    explicit = _int(_payload_text(top, "attempt_no", "attempt"))
    if explicit is None and receipt_first:
        explicit = _int(_payload_text(receipt_first, "attempt_no", "attempt"))
    artifact_sha = _payload_text(top, "task_sha", "sha", "parent_sha256")
    if not artifact_sha:
        artifact_sha = _payload_text(receipt_first, "task_sha", "sha", "parent_sha256")
    if task_sha and artifact_sha and artifact_sha != task_sha:
        task_id = task_id or f"{task_fn}@{task_sha}"
    task_id = task_id or f"{task_fn}@{task_sha or 'unknown'}"
    run_id = run_id or f"{tag or 'base'}:{task_fn}"
    return task_id, run_id, repo, explicit, rel_path


def distill_attempt(
    artifact_root: str | Path,
    task_fn: str | None = None,
    task_sha: str = "",
    tag: str = "",
    *,
    fn: str | None = None,
    attempt_no: int | None = None,
    campaign: str = "repair",
    task_id: str | None = None,
    run_id: str | None = None,
    attempt_order: Mapping[str, int] | Sequence[str] | None = None,
) -> AttemptMemoryV1:
    """Distil one ``(task function, task sha, tag)`` artifact set."""

    task_fn = task_fn or fn
    if not task_fn or Path(task_fn).name != task_fn or task_fn in {".", ".."}:
        raise ValueError("task_fn must be a single artifact filename stem")
    tag = _tag_value(tag)
    results = _results_dir(artifact_root)
    transcript_path = results / _artifact_name("react_transcripts", tag, task_fn, ".json")
    receipt_path = results / _artifact_name("receipts", tag, task_fn, ".jsonl")
    transcript, transcript_digest, transcript_gaps = _read_json(transcript_path)
    receipt_records, receipt_digest, receipt_gaps = _read_jsonl(receipt_path)
    receipts = _receipt_objects(receipt_records)
    gaps = [*transcript_gaps, *receipt_gaps]
    artifact_sha = _payload_text(transcript or {}, "task_sha", "sha", "parent_sha256")
    if not artifact_sha and receipts:
        artifact_sha = _payload_text(receipts[0].record, "task_sha", "sha", "parent_sha256")
    if task_sha and artifact_sha and artifact_sha != task_sha:
        gaps.append("task_sha_mismatch")
        transcript = None
        transcript_digest = ""
        receipts = ()
        receipt_records = ()
    artifact_campaign = _payload_text(transcript or {}, "campaign") or (
        _payload_text(receipts[0].record, "campaign") if receipts else ""
    )
    if campaign and artifact_campaign and artifact_campaign != campaign:
        gaps.append("campaign_mismatch")
        transcript = None
        transcript_digest = ""
        receipts = ()
        receipt_records = ()
    transcript_present = transcript is not None and isinstance(transcript.get("messages"), list)
    if transcript is not None and _payload_text(transcript, "fn", "target_symbol") and _payload_text(
        transcript, "fn", "target_symbol"
    ) != task_fn:
        gaps.append("transcript_fn_mismatch")
        transcript_present = False
    if receipts and _payload_text(receipts[0].record, "fn", "target_symbol") and _payload_text(
        receipts[0].record, "fn", "target_symbol"
    ) != task_fn:
        gaps.append("receipt_fn_mismatch")
        receipts = ()
    events: tuple[Mapping[str, object], ...] = ()
    event_digest = ""
    if transcript_present:
        event_path = _event_path(transcript_path)
        if event_path.is_file():
            event_records, event_digest, event_gaps = _read_jsonl(event_path)
            events = event_records
            gaps.extend(gap for gap in event_gaps if gap.startswith("invalid:"))
    messages = _records(transcript.get("messages")) if transcript_present and transcript else ()
    messages = tuple(_record_copy(message) for message in messages)
    messages = tuple(_reconstruct_messages(transcript_path, messages, events))
    actions, prose = _actions(messages)
    edits = _edit_records(actions, receipts, tag, task_fn)
    theories = _theories(transcript or {}, actions, receipts, edits, prose)
    failed_legs = _failed_legs(actions, receipts, tag)
    atoms = _atoms(transcript or {}, receipts, tag)
    frontier = _frontier(transcript or {}, receipts, tag)
    derived_task_id, derived_run_id, repo, explicit_attempt, rel_path = _artifact_metadata(
        transcript, receipts, task_fn, task_sha, tag
    )
    selected_attempt = _attempt_no_for_tag(tag, attempt_no or explicit_attempt, attempt_order)
    manifest = {
        "transcript": {"ref": f"{transcript_path.parent.name}/{transcript_path.name}", "digest": transcript_digest},
        "events": {"ref": f"{_event_path(transcript_path).parent.name}/{_event_path(transcript_path).name}", "digest": event_digest},
        "receipts": {"ref": f"{receipt_path.parent.name}/{receipt_path.name}", "digest": receipt_digest},
    }
    input_digest = _digest(manifest, drop_volatile=True)
    transcript_status = (
        "RECONSTRUCTED"
        if any(
            _text(event.get("kind"), 40).lower() == "trimmed"
            and isinstance(event.get("msg"), Mapping)
            for event in events
        )
        else "COMPLETE"
    )
    terminal = _terminal(
        transcript or {},
        actions,
        receipts,
        transcript_present,
        gaps,
        transcript_status,
    )
    final_campaign = _payload_text(transcript or {}, "campaign") or campaign or "repair"
    key = {
        "campaign": final_campaign,
        "task_id": task_id or derived_task_id,
        "task_sha": task_sha or artifact_sha or _payload_text(transcript or {}, "task_sha", "sha"),
        "attempt_no": selected_attempt,
        "run_id": run_id or derived_run_id,
        "tag": tag,
    }
    world = {
        "repo": repo,
        "rel_path": rel_path,
        "symbol": task_fn,
        "parent_sha256": task_sha or artifact_sha or None,
        "ast_normalizer": AST_NORMALIZER,
        "provenance": AGENT_WORLD_PROVENANCE,
    }
    audit = {
        "input_digest": input_digest,
        "fixed_source_seen": False,
        "grader_imported": False,
        "forbidden_fields_seen": (),
        "gaps": tuple(sorted(set(gaps))),
        "source_refs": tuple(
            ref["ref"] for ref in manifest.values() if isinstance(ref, Mapping) and ref.get("digest")
        ),
    }
    # A transcript without optional receipt/event companions is still a complete
    # agent-world tape; malformed data or a sha mismatch makes the seal partial.
    effective_gaps = [gap for gap in gaps if not gap.startswith("missing:")]
    verdict = "COMPLETE" if transcript_present and not effective_gaps else "PARTIAL"
    if transcript_present and not effective_gaps:
        terminal["seal"] = "COMPLETE"
        audit["gaps"] = ()
    if not transcript_present and receipts:
        terminal["shape"] = "PARTIAL"
        terminal["underlying_shape"] = None
    return AttemptMemoryV1(
        key=key,
        world=world,
        edits=edits,
        theories=theories,
        failed_legs=failed_legs,
        atoms=atoms,
        terminal=terminal,
        frontier=frontier,
        audit=audit,
        verdict=verdict,
    )


def distill_attempts(
    artifact_root: str | Path,
    task_fn: str | None = None,
    task_sha: str = "",
    *,
    fn: str | None = None,
    current_attempt: int | None = None,
    campaign: str = "repair",
    task_id: str | None = None,
    attempt_order: Mapping[str, int] | Sequence[str] | None = None,
) -> tuple[AttemptMemoryV1, ...]:
    """Distil all matching tagged attempts before ``current_attempt``."""

    task_fn = task_fn or fn
    if not task_fn:
        raise ValueError("task_fn is required")
    records = [
        distill_attempt(
            artifact_root,
            task_fn,
            task_sha,
            tag,
            campaign=campaign,
            task_id=task_id,
            attempt_order=attempt_order,
        )
        for tag in discover_tags(artifact_root, task_fn)
    ]
    records = [
        record
        for record in records
        if not {
            "task_sha_mismatch",
            "campaign_mismatch",
            "transcript_fn_mismatch",
            "receipt_fn_mismatch",
        }.intersection(
            str(gap) for gap in record.audit.get("gaps", ())
        )
    ]
    records.sort(key=lambda record: (record.attempt_no, record.tag, _text(record.key.get("run_id"), 100)))
    records = [record for record in records if record.attempt_no > 0]
    if current_attempt is not None:
        records = [record for record in records if record.attempt_no < current_attempt]
    return tuple(records)


def distill_prior_attempts(
    artifact_root: str | Path,
    task_fn: str | None = None,
    task_sha: str = "",
    current_attempt: int | None = None,
    *,
    fn: str | None = None,
    campaign: str = "repair",
    task_id: str | None = None,
    attempt_order: Mapping[str, int] | Sequence[str] | None = None,
) -> tuple[AttemptMemoryV1, ...]:
    """Named entry point for the next-attempt memory load."""

    return distill_attempts(
        artifact_root,
        task_fn,
        task_sha,
        fn=fn,
        current_attempt=current_attempt,
        campaign=campaign,
        task_id=task_id,
        attempt_order=attempt_order,
    )


def distill(
    artifact_root: str | Path,
    task_fn: str | None = None,
    task_sha: str = "",
    tag: str | None = None,
    *,
    fn: str | None = None,
    current_attempt: int | None = None,
    campaign: str = "repair",
    task_id: str | None = None,
    attempt_order: Mapping[str, int] | Sequence[str] | None = None,
) -> AttemptMemoryV1 | tuple[AttemptMemoryV1, ...]:
    """Compatibility dispatcher: one tag returns one record, otherwise prior records."""

    if tag is not None:
        return distill_attempt(
            artifact_root,
            task_fn,
            task_sha,
            tag,
            fn=fn,
            campaign=campaign,
            task_id=task_id,
            attempt_order=attempt_order,
        )
    return distill_prior_attempts(
        artifact_root,
        task_fn,
        task_sha,
        current_attempt,
        fn=fn,
        campaign=campaign,
        task_id=task_id,
        attempt_order=attempt_order,
    )


def render_prior_capsule(
    artifact_root: str | Path,
    task_fn: str,
    task_sha: str,
    current_attempt: int,
    *,
    max_tokens: int = 800,
    max_chars: int | None = None,
    attempt_order: Mapping[str, int] | Sequence[str] | None = None,
) -> str:
    """Load prior records and render the exact next-attempt prompt capsule."""

    records = distill_prior_attempts(
        artifact_root,
        task_fn,
        task_sha,
        current_attempt,
        attempt_order=attempt_order,
    )
    return render_capsule(records, max_tokens=max_tokens, max_chars=max_chars)


def seal_attempt(memory: AttemptMemoryV1, path: str | Path) -> Path:
    """Write one canonical memory record without allowing an existing record overwrite."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as sink:
        sink.write(memory.serialize() + "\n")
    return destination


def seal_before_grade(
    memory: AttemptMemoryV1,
    path: str | Path,
    grader: Callable[[], GradeResult],
) -> GradeResult:
    """Provide the pre-grade weld: persist agent-world memory, then call grader."""

    seal_attempt(memory, path)
    return grader()


def exact_repeat_challenge(
    attempts: Iterable[object],
    candidate_ast_delta_fp: str | None,
    current_attempt: int,
    *,
    challenged_fingerprints: Iterable[str] = (),
    applied: bool = False,
    reconciliation: str = "",
) -> dict[str, object] | None:
    """Return one advisory repeat challenge for an exact, fully replayed AST delta."""

    candidate = _text(candidate_ast_delta_fp, 100)
    if not candidate or candidate in {_text(value, 100) for value in _values(challenged_fingerprints)}:
        return None
    prior_refs: list[str] = []
    for item in attempts:
        record = item if isinstance(item, Mapping) else _mapping(item)
        if record is None:
            continue
        for edit in _records(record.get("edits")):
            capture = _text(edit.get("capture"), 40).upper()
            fp = _text(edit.get("ast_delta_fp"), 100)
            if fp == candidate and capture != "LEGACY_CLIPPED":
                prior_refs.extend(
                    _text(ref, 180)
                    for ref in _values(edit.get("evidence_refs"))
                    if _text(ref, 180)
                )
    if not prior_refs:
        return None
    event: dict[str, object] = {
        "event": "PRIOR_REPEAT_CHALLENGE",
        "current_attempt": current_attempt,
        "candidate_ast_delta_fp": candidate,
        "decision": "ADVISORY",
        "veto": False,
        "evidence_refs": tuple(sorted(set(prior_refs))),
    }
    if applied:
        event["snitch"] = {
            "event": "SNITCH_PRIOR_REPEAT_OVERRIDE",
            "candidate_ast_delta_fp": candidate,
            "reconciliation": _text(reconciliation, 300) or "model chose to retry unchanged AST delta",
        }
    return event


repeat_challenge = exact_repeat_challenge
check_exact_repeat = exact_repeat_challenge


__all__ = [
    "AGENT_WORLD_PROVENANCE",
    "AST_NORMALIZER",
    "AttemptContext",
    "AttemptMemoryV1",
    "ForbiddenAgentWorldFieldError",
    "SCHEMA",
    "agent_world_identity",
    "ast_delta_fingerprint",
    "build_agent_world_identity",
    "capsule_digest",
    "check_exact_repeat",
    "discover_tags",
    "distill",
    "distill_attempt",
    "distill_attempts",
    "distill_prior_attempts",
    "exact_repeat_challenge",
    "render_capsule",
    "render_prior_attempts",
    "render_prior_capsule",
    "repeat_challenge",
    "seal_attempt",
    "seal_before_grade",
    "validate_agent_world_payload",
]

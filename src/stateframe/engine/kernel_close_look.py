"""Pure close-look organ for deliberate, bounded repair focus.

The live ReAct loop owns execution and mutation.  This module owns only the
deterministic state that tells that loop when it must stop skimming and study
one source region, what bounded evidence to render, and when the study has
been discharged.  It is intentionally dependency-light so it can be tested
against archived receipt tapes before wiring it into ``KernelBolt``.

The organ follows the kernel laws:

* receipt turns, not trace turns, are the canonical replay clock;
* ``UNKNOWN`` is retained as a count and digest, never as an unbounded prompt;
* every claim in an answer must cite a dossier reference (SNITCH);
* challenges are one-shot and every active focus has an explicit escape.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import PurePosixPath
from typing import TypeAlias

from kernel_liveness import Evidence, OrganCensusSpec, TriState


MODULE_SCHEMA = "poc.close_look/1"
SNITCH_LAW_ID = "SNITCH_ARCHITECTURE"
E394_LAW_ID = "E394_ATTENTION_BUDGET"
E398_LAW_ID = "E398_UNKNOWN_NO_FLOOD"
CHALLENGE_ONCE_LAW_ID = "CHALLENGE_ONCE_NEVER_A_CAGE"

SAME_ANCHOR_RED_THRESHOLD = 2
IN_REGION_EDIT_THRESHOLD = 5
MAX_OPEN_ACTIONS = 8
MAX_DOSSIER_TOKENS = 1800
MAX_SOURCE_LINES = 32
MAX_CALLERS = 2
MAX_CALLEES = 2
MAX_WITNESS_VECTORS = 4
WALKTHROUGH_ITEM_COUNT = 5

_RESULTS_THAT_ARE_NOT_PRODUCT_EVIDENCE = frozenset(
    {
        "INFRA",
        "INFRA_FAILURE",
        "ENVIRONMENT",
        "SELECTOR_ERROR",
        "NO_TESTS",
        "NO_TEST",
        "NO_VERDICT",
        "UNKNOWN",
        "UNAVAILABLE",
        "TEST_SIDE",
        "TEST_FAILURE_SIDE",
        "NO_OP",
        "REJECTED",
        "FAILED",
    }
)
_VALID_TEST_RESULTS = frozenset({"RED", "PRODUCT_RED", "GREEN", "PRODUCT_GREEN"})
_VALID_RED_RESULTS = frozenset({"RED", "PRODUCT_RED"})
_VALID_GREEN_RESULTS = frozenset({"GREEN", "PRODUCT_GREEN"})
_IGNORED_KINDS = frozenset({"infra", "selector", "selector_error", "no_test", "test_side"})
_FOCUS_PATH_RE = re.compile(
    r"^(?P<path>.+?):(?P<start>[1-9][0-9]*)(?:[-:](?P<end>[1-9][0-9]*))?$"
)
_ANCHOR_RE = re.compile(
    r"^(?P<path>.+?):(?P<start>[1-9][0-9]*)(?:-(?P<end>[1-9][0-9]*))?(?::(?P<symbol>[^:]+))?$"
)

JSONMapping: TypeAlias = Mapping[str, object]


class CloseLookError(ValueError):
    """Base error for invalid focus input or an impossible local contract."""


class FocusCommandError(CloseLookError):
    """Raised for malformed or unsafe ``focus`` commands."""


class FocusState(str, Enum):
    """Lifecycle of one deliberate-study transaction."""

    IDLE = "IDLE"
    OPEN = "OPEN"
    ANSWERED = "ANSWERED"
    EDITED = "EDITED"
    CLOSED = "CLOSED"
    ABORTED = "ABORTED"


class AnswerState(str, Enum):
    """Per-item comparison state in the model's numbered walkthrough."""

    MATCH = "MATCH"
    DIVERGES = "DIVERGES"
    UNKNOWN = "UNKNOWN"


def _clean(value: object, *, limit: int = 512) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)] + "…"


def _display(value: object, *, limit: int = 512) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, str):
        return _clean(value, limit=limit)
    try:
        return _clean(json.dumps(value, sort_keys=True, ensure_ascii=True, default=str), limit=limit)
    except (TypeError, ValueError):
        return _clean(value, limit=limit)


def _canonical(value: object) -> object:
    """Make supported values deterministic and JSON-safe for fingerprints."""

    if isinstance(value, Enum):
        return value.value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict())
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        values = [_canonical(item) for item in value]
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return _display(value)


def canonical_json(value: object) -> str:
    """Serialize values with stable keys and collection ordering."""

    return json.dumps(
        _canonical(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _positive_int(value: object, field_name: str, *, default: int | None = None) -> int:
    if value in (None, "") and default is not None:
        return default
    if isinstance(value, bool):
        raise CloseLookError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CloseLookError(f"{field_name} must be a positive integer") from exc
    if parsed < 1:
        raise CloseLookError(f"{field_name} must be a positive integer")
    return parsed


def _optional_int(value: object) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def normalize_relative_path(value: object) -> str:
    """Normalize a repo-relative POSIX path and reject traversal/absolute paths."""

    raw = str(value or "").strip().replace("\\", "/")
    if not raw or raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
        raise FocusCommandError("focus path must be a non-empty repository-relative path")
    parts = PurePosixPath(raw).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise FocusCommandError("focus path cannot contain traversal or empty components")
    return PurePosixPath(*parts).as_posix()


def _paths_match(left: str, right: str) -> bool:
    """Match exact paths plus safe suffixes from basename-only archived receipts."""

    try:
        a = normalize_relative_path(left)
        b = normalize_relative_path(right)
    except FocusCommandError:
        return False
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


@dataclass(frozen=True, slots=True, order=True)
class Region:
    """A normalized inclusive source region."""

    path: str
    start_line: int
    end_line: int
    symbol: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_relative_path(self.path))
        start = _positive_int(self.start_line, "start_line")
        end = _positive_int(self.end_line, "end_line")
        if end < start:
            raise CloseLookError("end_line must not precede start_line")
        object.__setattr__(self, "start_line", start)
        object.__setattr__(self, "end_line", end)
        object.__setattr__(self, "symbol", _clean(self.symbol, limit=160))

    @property
    def label(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def contains(self, line: int) -> bool:
        return self.start_line <= line <= self.end_line

    def intersects(self, other: "Region") -> bool:
        return _paths_match(self.path, other.path) and not (
            self.end_line < other.start_line or other.end_line < self.start_line
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol": self.symbol,
            "label": self.label,
        }

    @classmethod
    def from_value(cls, value: object, *, symbol: str = "") -> "Region":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            start = value.get("start_line", value.get("start", value.get("line")))
            end = value.get("end_line", value.get("end", start))
            return cls(
                path=str(value.get("path", value.get("file", ""))),
                start_line=_positive_int(start, "start_line"),
                end_line=_positive_int(end, "end_line"),
                symbol=str(value.get("symbol", symbol) or symbol),
            )
        match = _FOCUS_PATH_RE.fullmatch(str(value or "").strip())
        if not match:
            raise FocusCommandError("region must use path:start-end or path:line")
        return cls(
            path=match.group("path"),
            start_line=int(match.group("start")),
            end_line=int(match.group("end") or match.group("start")),
            symbol=symbol,
        )


@dataclass(frozen=True, slots=True)
class Anchor:
    """E312 anchor plus the contract fingerprints that make it stable."""

    workspace: str
    path: str
    symbol: str
    start_line: int
    end_line: int
    e312_anchor: str = ""
    contract_fingerprints: tuple[str, ...] = ()
    fallback_used: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", _clean(self.workspace, limit=160) or "workspace")
        object.__setattr__(self, "path", normalize_relative_path(self.path))
        object.__setattr__(self, "symbol", _clean(self.symbol, limit=160) or "unknown")
        start = _positive_int(self.start_line, "anchor start_line")
        end = _positive_int(self.end_line, "anchor end_line")
        if end < start:
            raise CloseLookError("anchor end_line must not precede start_line")
        object.__setattr__(self, "start_line", start)
        object.__setattr__(self, "end_line", end)
        canonical = f"{self.path}:{self.start_line}:{self.symbol}"
        object.__setattr__(self, "e312_anchor", _clean(self.e312_anchor, limit=320) or canonical)
        raw_fingerprints = (
            (self.contract_fingerprints,)
            if isinstance(self.contract_fingerprints, str)
            else self.contract_fingerprints
        )
        fps = tuple(sorted({
            _clean(item, limit=160)
            for item in raw_fingerprints
            if _clean(item, limit=160)
        }))
        object.__setattr__(self, "contract_fingerprints", fps)

    @property
    def region(self) -> Region:
        return Region(self.path, self.start_line, self.end_line, self.symbol)

    @property
    def stable_key(self) -> str:
        return _digest(
            {
                "workspace": self.workspace,
                "path": self.path,
                "symbol": self.symbol,
                "e312_anchor": self.e312_anchor,
                "contract_fingerprints": self.contract_fingerprints,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "workspace": self.workspace,
            "path": self.path,
            "symbol": self.symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "e312_anchor": self.e312_anchor,
            "contract_fingerprints": list(self.contract_fingerprints),
            "fallback_used": self.fallback_used,
            "stable_key": self.stable_key,
        }


def normalize_anchor(
    value: object,
    *,
    workspace: str = "workspace",
    region: Region | Mapping[str, object] | None = None,
    symbol: str = "",
    contract_fingerprints: Sequence[object] = (),
    fallback_used: bool = False,
) -> Anchor:
    """Normalize E312's string/mapping shapes without executing repository code."""

    default_region = Region.from_value(region, symbol=symbol) if region is not None else None
    if isinstance(value, Anchor):
        if fallback_used and not value.fallback_used:
            return replace(value, fallback_used=True)
        return value
    raw = value
    if isinstance(value, Mapping):
        # An Anchor.to_dict() contains both the structured path/line fields and
        # the E312 marker.  Prefer an explicitly nested anchor; otherwise keep
        # the mapping so its structured fields survive round-tripping.
        nested = value.get("primary_anchor", value.get("anchor"))
        raw = nested if nested not in (None, "") else value
    path = ""
    start: int | None = None
    end: int | None = None
    parsed_symbol = symbol
    e312 = ""
    if isinstance(raw, Mapping):
        path = str(raw.get("path", raw.get("file", "")) or "")
        start = _optional_int(raw.get("start_line", raw.get("line")))
        end = _optional_int(raw.get("end_line", raw.get("end")))
        parsed_symbol = str(raw.get("symbol", raw.get("function", parsed_symbol)) or parsed_symbol)
        e312 = str(raw.get("e312_anchor", raw.get("primary_anchor", "")) or "")
        source_fps = raw.get("contract_fingerprints", raw.get("atom_fingerprints", ()))
        if isinstance(source_fps, Sequence) and not isinstance(source_fps, (str, bytes)):
            contract_fingerprints = tuple(source_fps)
    elif raw not in (None, ""):
        match = _ANCHOR_RE.fullmatch(str(raw).strip())
        if not match:
            raise CloseLookError("anchor must use path:line[:symbol]")
        path = match.group("path")
        start = int(match.group("start"))
        end = int(match.group("end") or match.group("start"))
        parsed_symbol = match.group("symbol") or parsed_symbol
        e312 = str(raw).strip()
    if default_region is not None:
        path = path or default_region.path
        start = start or default_region.start_line
        end = end or default_region.end_line
        parsed_symbol = parsed_symbol or default_region.symbol
    if not path or start is None:
        raise CloseLookError("anchor needs a path and line")
    return Anchor(
        workspace=workspace,
        path=path,
        symbol=parsed_symbol,
        start_line=start,
        end_line=end or start,
        e312_anchor=e312,
        contract_fingerprints=tuple(str(item) for item in contract_fingerprints),
        fallback_used=fallback_used,
    )


@dataclass(frozen=True, slots=True)
class FocusCommand:
    """Parsed explicit focus command."""

    action: str
    region: Region | None = None

    def to_dict(self) -> dict[str, object]:
        return {"action": self.action, "region": self.region.to_dict() if self.region else None}


def parse_focus_command(command: str, *, symbol: str = "") -> FocusCommand:
    """Parse ``focus path:start-end`` or the explicit ``focus abort`` escape."""

    text = str(command or "").strip()
    if re.fullmatch(r"focus\s+abort", text, flags=re.IGNORECASE):
        return FocusCommand("abort")
    match = re.fullmatch(r"focus\s+(.+)", text, flags=re.IGNORECASE)
    if not match:
        raise FocusCommandError("use `focus path:start-end` or `focus abort`")
    target = match.group(1).strip()
    try:
        region = Region.from_value(target, symbol=symbol)
    except CloseLookError as exc:
        raise FocusCommandError(str(exc)) from exc
    return FocusCommand("open", region)


@dataclass(frozen=True, slots=True)
class CloseLookPolicy:
    """Hard kernel bounds; models cannot tune these per turn."""

    same_anchor_red_threshold: int = SAME_ANCHOR_RED_THRESHOLD
    in_region_edit_threshold: int = IN_REGION_EDIT_THRESHOLD
    max_open_actions: int = MAX_OPEN_ACTIONS
    max_dossier_tokens: int = MAX_DOSSIER_TOKENS
    max_source_lines: int = MAX_SOURCE_LINES
    max_callers: int = MAX_CALLERS
    max_callees: int = MAX_CALLEES
    max_witness_vectors: int = MAX_WITNESS_VECTORS

    def __post_init__(self) -> None:
        for name in (
            "same_anchor_red_threshold",
            "in_region_edit_threshold",
            "max_open_actions",
            "max_dossier_tokens",
            "max_source_lines",
            "max_callers",
            "max_callees",
            "max_witness_vectors",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CloseLookError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "same_anchor_red_threshold": self.same_anchor_red_threshold,
            "in_region_edit_threshold": self.in_region_edit_threshold,
            "max_open_actions": self.max_open_actions,
            "max_dossier_tokens": self.max_dossier_tokens,
            "max_source_lines": self.max_source_lines,
            "max_callers": self.max_callers,
            "max_callees": self.max_callees,
            "max_witness_vectors": self.max_witness_vectors,
        }


DEFAULT_POLICY = CloseLookPolicy()


def _source_line(value: object, *, fallback_number: int) -> "SourceLine":
    if isinstance(value, SourceLine):
        return value
    if isinstance(value, Mapping):
        number = _positive_int(value.get("number", value.get("line", fallback_number)), "source line")
        text = str(value.get("text", value.get("source", "")) or "")
        return SourceLine(number, text)
    return SourceLine(fallback_number, str(value or ""))


@dataclass(frozen=True, slots=True, order=True)
class SourceLine:
    number: int
    text: str

    def __post_init__(self) -> None:
        if isinstance(self.number, bool) or not isinstance(self.number, int) or self.number < 1:
            raise CloseLookError("source line number must be positive")
        object.__setattr__(self, "text", str(self.text))

    def to_dict(self) -> dict[str, object]:
        return {"line": self.number, "text": self.text}


@dataclass(frozen=True, slots=True, order=True)
class ContractAtom:
    """A contract demand/observation reference used by branches and witnesses."""

    ref: str
    kind: str
    demanded: str
    produced: str = ""
    line: int | None = None
    polarity: str = ""
    path: str = ""
    fingerprint: str = ""

    def __post_init__(self) -> None:
        ref = _clean(self.ref, limit=80)
        if not ref:
            raise CloseLookError("contract atom ref is required")
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "kind", _clean(self.kind, limit=80) or "contract")
        object.__setattr__(self, "demanded", _display(self.demanded))
        object.__setattr__(self, "produced", _display(self.produced) if self.produced != "" else "UNKNOWN")
        object.__setattr__(self, "line", _optional_int(self.line))
        object.__setattr__(self, "polarity", _clean(self.polarity, limit=160))
        if self.path:
            object.__setattr__(self, "path", normalize_relative_path(self.path))
        expected = _digest(
            {
                "ref": self.ref,
                "kind": self.kind,
                "demanded": self.demanded,
                "produced": self.produced,
                "line": self.line,
                "polarity": self.polarity,
                "path": self.path,
            }
        )
        supplied = _clean(self.fingerprint, limit=160)
        object.__setattr__(self, "fingerprint", supplied or expected)

    @property
    def atom_id(self) -> str:
        return self.ref

    @property
    def expected(self) -> str:
        return self.demanded

    @property
    def observed(self) -> str:
        return self.produced

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self.ref,
            "kind": self.kind,
            "demanded": self.demanded,
            "produced": self.produced,
            "line": self.line,
            "polarity": self.polarity,
            "path": self.path,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_value(cls, value: object, *, default_ref: str) -> "ContractAtom":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return cls(default_ref, "contract", _display(value), "UNKNOWN")
        return cls(
            ref=str(value.get("ref", value.get("id", value.get("atom_id", default_ref))) or default_ref),
            kind=str(value.get("kind", "contract") or "contract"),
            demanded=_display(value.get("demanded", value.get("expected", value.get("demand", "UNKNOWN")))),
            produced=_display(value.get("produced", value.get("observed", value.get("actual", "UNKNOWN")))),
            line=_optional_int(value.get("line", value.get("start_line"))),
            polarity=str(value.get("polarity", "") or ""),
            path=str(value.get("path", "") or ""),
            fingerprint=str(value.get("fingerprint", "") or ""),
        )


@dataclass(frozen=True, slots=True, order=True)
class BranchCondition:
    """A branch plus true/false polarity and atom references."""

    ref: str
    line: int
    end_line: int
    condition: str
    true_polarity: str
    false_polarity: str
    contract_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _clean(self.ref, limit=80))
        object.__setattr__(self, "line", _positive_int(self.line, "branch line"))
        object.__setattr__(self, "end_line", _positive_int(self.end_line, "branch end_line"))
        if self.end_line < self.line:
            raise CloseLookError("branch end_line must not precede line")
        object.__setattr__(self, "condition", _clean(self.condition, limit=300))
        object.__setattr__(self, "true_polarity", _clean(self.true_polarity, limit=300))
        object.__setattr__(self, "false_polarity", _clean(self.false_polarity, limit=300))
        object.__setattr__(
            self,
            "contract_refs",
            tuple(dict.fromkeys(_clean(item, limit=80) for item in self.contract_refs if _clean(item, limit=80))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self.ref,
            "line": self.line,
            "end_line": self.end_line,
            "condition": self.condition,
            "true_polarity": self.true_polarity,
            "false_polarity": self.false_polarity,
            "contract_refs": list(self.contract_refs),
        }


@dataclass(frozen=True, slots=True, order=True)
class EffectStep:
    """One scoped fact-compiler EFFECT-lens value-flow step."""

    ref: str
    line: int
    end_line: int
    input_value: str
    operation: str
    produced: str
    effect: str
    contract_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _clean(self.ref, limit=80))
        object.__setattr__(self, "line", _positive_int(self.line, "effect line"))
        object.__setattr__(self, "end_line", _positive_int(self.end_line, "effect end_line"))
        if self.end_line < self.line:
            raise CloseLookError("effect end_line must not precede line")
        for field_name in ("input_value", "operation", "produced", "effect"):
            object.__setattr__(self, field_name, _clean(getattr(self, field_name), limit=600) or "UNKNOWN")
        object.__setattr__(
            self,
            "contract_refs",
            tuple(dict.fromkeys(_clean(item, limit=80) for item in self.contract_refs if _clean(item, limit=80))),
        )

    @property
    def input(self) -> str:
        return self.input_value

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self.ref,
            "line": self.line,
            "end_line": self.end_line,
            "input": self.input_value,
            "operation": self.operation,
            "produced": self.produced,
            "effect": self.effect,
            "contract_refs": list(self.contract_refs),
        }


ValueFlowStep = EffectStep


@dataclass(frozen=True, slots=True, order=True)
class GraphEdge:
    """One caller/callee edge, capped to one hop by dossier assembly."""

    ref: str
    relation: str
    symbol: str
    path: str = ""
    line: int | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _clean(self.ref, limit=80))
        relation = _clean(self.relation, limit=30).lower()
        if relation not in {"caller", "callee"}:
            raise CloseLookError("graph relation must be caller or callee")
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "symbol", _clean(self.symbol, limit=180) or "UNKNOWN")
        if self.path:
            object.__setattr__(self, "path", normalize_relative_path(self.path))
        object.__setattr__(self, "line", _optional_int(self.line))
        object.__setattr__(self, "reason", _clean(self.reason, limit=300))

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self.ref,
            "relation": self.relation,
            "symbol": self.symbol,
            "path": self.path,
            "line": self.line,
            "reason": self.reason,
        }

    @classmethod
    def from_value(cls, value: object, *, relation: str, default_ref: str) -> "GraphEdge":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                ref=str(value.get("ref", value.get("id", default_ref)) or default_ref),
                relation=str(value.get("relation", relation) or relation),
                symbol=str(value.get("symbol", value.get("name", "UNKNOWN")) or "UNKNOWN"),
                path=str(value.get("path", value.get("file", "")) or ""),
                line=_optional_int(value.get("line", value.get("start_line"))),
                reason=str(value.get("reason", "") or ""),
            )
        return cls(default_ref, relation, _display(value))


@dataclass(frozen=True, slots=True, order=True)
class WitnessVector:
    """Oracle demanded value beside the value produced by the current code."""

    ref: str
    demanded: str
    produced: str
    line: int | None = None
    contract_refs: tuple[str, ...] = ()
    state: AnswerState = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _clean(self.ref, limit=80))
        object.__setattr__(self, "demanded", _display(self.demanded))
        object.__setattr__(self, "produced", _display(self.produced))
        object.__setattr__(self, "line", _optional_int(self.line))
        object.__setattr__(
            self,
            "contract_refs",
            tuple(dict.fromkeys(_clean(item, limit=80) for item in self.contract_refs if _clean(item, limit=80))),
        )
        if self.demanded.upper() == "UNKNOWN" or self.produced.upper() == "UNKNOWN":
            state = AnswerState.UNKNOWN
        else:
            state = AnswerState.MATCH if self.demanded == self.produced else AnswerState.DIVERGES
        object.__setattr__(self, "state", state)

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self.ref,
            "demanded": self.demanded,
            "produced": self.produced,
            "line": self.line,
            "contract_refs": list(self.contract_refs),
            "state": self.state.value,
        }

    @classmethod
    def from_value(cls, value: object, *, default_ref: str) -> "WitnessVector":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return cls(default_ref, "UNKNOWN", _display(value))
        return cls(
            ref=str(value.get("ref", value.get("id", value.get("witness", default_ref))) or default_ref),
            demanded=_display(value.get("demanded", value.get("expected", value.get("oracle", "UNKNOWN")))),
            produced=_display(value.get("produced", value.get("actual", value.get("observed", "UNKNOWN")))),
            line=_optional_int(value.get("line", value.get("divergence_line"))),
            contract_refs=tuple(str(item) for item in value.get("contract_refs", value.get("refs", ())) or ()),
        )


def _sequence(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def _line_overlap(line: int, end_line: int, region: Region) -> bool:
    return not (end_line < region.start_line or line > region.end_line)


def _atom_refs_for_span(atoms: Sequence[ContractAtom], line: int, end_line: int, region: Region) -> tuple[str, ...]:
    refs = [
        atom.ref
        for atom in atoms
        if atom.line is not None
        and _line_overlap(atom.line, atom.line, Region(region.path, line, end_line))
        and (not atom.path or _paths_match(atom.path, region.path))
    ]
    return tuple(sorted(set(refs)))


def _normalize_atoms(values: object) -> tuple[ContractAtom, ...]:
    atoms = [ContractAtom.from_value(item, default_ref=f"A{index}") for index, item in enumerate(_sequence(values), 1)]
    by_ref: dict[str, ContractAtom] = {}
    for atom in atoms:
        by_ref.setdefault(atom.ref, atom)
    return tuple(sorted(by_ref.values(), key=lambda item: (item.line or 0, item.ref)))


def _normalize_branches(values: object, atoms: Sequence[ContractAtom], region: Region) -> tuple[BranchCondition, ...]:
    branches: list[BranchCondition] = []
    for index, item in enumerate(_sequence(values), 1):
        if isinstance(item, BranchCondition):
            branches.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        line = _positive_int(item.get("line", region.start_line), "branch line")
        end = _positive_int(item.get("end_line", item.get("end", line)), "branch end_line")
        refs = item.get("contract_refs", item.get("refs", ()))
        references = tuple(str(ref) for ref in refs) if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)) else ()
        if not references:
            references = _atom_refs_for_span(atoms, line, end, region)
        branches.append(
            BranchCondition(
                ref=str(item.get("ref", item.get("id", f"B{index}"))) or f"B{index}",
                line=line,
                end_line=end,
                condition=str(item.get("condition", item.get("test", "UNKNOWN")) or "UNKNOWN"),
                true_polarity=str(item.get("true_polarity", item.get("true", "condition is true")) or "condition is true"),
                false_polarity=str(item.get("false_polarity", item.get("false", "condition is false")) or "condition is false"),
                contract_refs=references,
            )
        )
    return tuple(sorted(branches, key=lambda item: (item.line, item.ref)))


def _normalize_effects(values: object, atoms: Sequence[ContractAtom], region: Region) -> tuple[EffectStep, ...]:
    effects: list[EffectStep] = []
    for index, item in enumerate(_sequence(values), 1):
        if isinstance(item, EffectStep):
            effects.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        line = _positive_int(item.get("line", region.start_line), "effect line")
        end = _positive_int(item.get("end_line", item.get("end", line)), "effect end_line")
        refs = item.get("contract_refs", item.get("refs", ()))
        references = tuple(str(ref) for ref in refs) if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)) else ()
        if not references:
            references = _atom_refs_for_span(atoms, line, end, region)
        effects.append(
            EffectStep(
                ref=str(item.get("ref", item.get("id", f"E{index}"))) or f"E{index}",
                line=line,
                end_line=end,
                input_value=str(item.get("input", item.get("input_value", "UNKNOWN")) or "UNKNOWN"),
                operation=str(item.get("operation", item.get("statement", "UNKNOWN")) or "UNKNOWN"),
                produced=str(item.get("produced", item.get("output", "UNKNOWN")) or "UNKNOWN"),
                effect=str(item.get("effect", item.get("effect_kind", "UNKNOWN")) or "UNKNOWN"),
                contract_refs=references,
            )
        )
    return tuple(sorted(effects, key=lambda item: (item.line, item.ref)))


def _normalize_graph(values: object, *, relation: str, limit: int) -> tuple[GraphEdge, ...]:
    edges = [
        GraphEdge.from_value(item, relation=relation, default_ref=f"G{relation[0].upper()}{index}")
        for index, item in enumerate(_sequence(values), 1)
    ]
    unique: dict[tuple[str, str, str, int | None], GraphEdge] = {}
    for edge in edges:
        unique.setdefault((edge.relation, edge.symbol, edge.path, edge.line), edge)
    return tuple(sorted(unique.values(), key=lambda item: (item.path, item.line or 0, item.symbol))[:limit])


def _normalize_witnesses(values: object, *, limit: int) -> tuple[WitnessVector, ...]:
    witnesses = [
        WitnessVector.from_value(item, default_ref=f"W{index}")
        for index, item in enumerate(_sequence(values), 1)
    ]
    unique: dict[str, WitnessVector] = {}
    for witness in witnesses:
        unique.setdefault(witness.ref, witness)
    return tuple(sorted(unique.values(), key=lambda item: (item.line or 0, item.ref))[:limit])


def _parse_tree(source: str) -> ast.AST | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return None


def _node_end_line(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))


def _segment(source: str, node: ast.AST, *, limit: int) -> str:
    try:
        text = ast.get_source_segment(source, node) or ast.unparse(node)
    except (AttributeError, TypeError, ValueError):
        text = "UNKNOWN"
    return _clean(text, limit=limit)


def _derive_branches(source: str, region: Region, atoms: Sequence[ContractAtom]) -> tuple[BranchCondition, ...]:
    tree = _parse_tree(source)
    if tree is None:
        return ()
    branches: list[BranchCondition] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        line = int(getattr(node, "lineno", 1))
        end = _node_end_line(node)
        if not _line_overlap(line, end, region):
            continue
        refs = _atom_refs_for_span(atoms, line, end, region)
        has_else = bool(node.orelse)
        true_effect = "execute true branch"
        false_effect = "execute else branch" if has_else else "skip true branch"
        branches.append(
            BranchCondition(
                ref=f"B{len(branches) + 1}",
                line=line,
                end_line=end,
                condition=_segment(source, node.test, limit=300),
                true_polarity=f"TRUE -> {true_effect}",
                false_polarity=f"FALSE -> {false_effect}",
                contract_refs=refs,
            )
        )
    return tuple(sorted(branches, key=lambda item: (item.line, item.ref)))


def _assigned_names(node: ast.AST) -> tuple[str, ...]:
    targets: list[str] = []
    if isinstance(node, ast.Assign):
        values = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        values = (node.target,)
    else:
        values = ()
    for target in values:
        if isinstance(target, ast.Name):
            targets.append(target.id)
        else:
            targets.append(_segment("", target, limit=120))
    return tuple(targets)


def _loaded_names(node: ast.AST) -> tuple[str, ...]:
    return tuple(sorted({child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}))


def _derive_effects(source: str, region: Region, atoms: Sequence[ContractAtom]) -> tuple[EffectStep, ...]:
    tree = _parse_tree(source)
    if tree is None:
        return ()
    statements: list[ast.stmt] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Raise, ast.Expr)):
            line = int(getattr(node, "lineno", 1))
            end = _node_end_line(node)
            if _line_overlap(line, end, region):
                statements.append(node)
    statements.sort(key=lambda item: (int(getattr(item, "lineno", 1)), int(getattr(item, "col_offset", 0))))
    effects: list[EffectStep] = []
    seen: set[tuple[int, int, str]] = set()
    for node in statements:
        line = int(getattr(node, "lineno", 1))
        end = _node_end_line(node)
        operation = _segment(source, node, limit=600)
        key = (line, end, operation)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(node, ast.Return):
            produced = "return value"
            effect = "returns"
        elif isinstance(node, ast.Raise):
            produced = "exception"
            effect = "raises"
        elif isinstance(node, ast.Expr):
            produced = "side-effect"
            effect = "calls" if any(isinstance(child, ast.Call) for child in ast.walk(node)) else "reads"
        else:
            assigned = _assigned_names(node)
            produced = ", ".join(assigned) if assigned else "updated value"
            effect = "writes"
        loaded = ", ".join(_loaded_names(node)) or "UNKNOWN"
        effects.append(
            EffectStep(
                ref=f"E{len(effects) + 1}",
                line=line,
                end_line=end,
                input_value=loaded,
                operation=operation,
                produced=produced,
                effect=effect,
                contract_refs=_atom_refs_for_span(atoms, line, end, region),
            )
        )
    return tuple(effects[: max(MAX_SOURCE_LINES, 1)])


@dataclass(frozen=True, slots=True)
class Dossier:
    """The one bounded microscope artifact delivered to the model."""

    focus_id: str
    region: Region
    source_revision: str
    source_digest: str
    source_lines: tuple[SourceLine, ...]
    contract_atoms: tuple[ContractAtom, ...]
    branches: tuple[BranchCondition, ...]
    value_flow: tuple[EffectStep, ...]
    callers: tuple[GraphEdge, ...]
    callees: tuple[GraphEdge, ...]
    witnesses: tuple[WitnessVector, ...]
    unknown_count: int = 0
    unknown_digest: str | None = None
    first_divergence_line: int | None = None
    first_divergence_reason: str = ""
    policy: CloseLookPolicy = DEFAULT_POLICY
    laws: tuple[str, ...] = (
        SNITCH_LAW_ID,
        E394_LAW_ID,
        E398_LAW_ID,
        CHALLENGE_ONCE_LAW_ID,
    )

    def __post_init__(self) -> None:
        if len(self.source_lines) > self.policy.max_source_lines:
            object.__setattr__(self, "source_lines", self.source_lines[: self.policy.max_source_lines])
        object.__setattr__(self, "source_revision", _clean(self.source_revision, limit=180) or "unknown")
        object.__setattr__(self, "source_digest", _clean(self.source_digest, limit=180) or _digest(()))
        object.__setattr__(self, "unknown_count", max(0, int(self.unknown_count)))
        if self.unknown_count and not self.unknown_digest:
            object.__setattr__(self, "unknown_digest", _digest({"unknown_count": self.unknown_count}))
        if not self.unknown_count:
            object.__setattr__(self, "unknown_digest", None)
        object.__setattr__(self, "first_divergence_reason", _clean(self.first_divergence_reason, limit=500))

    @property
    def reference_ids(self) -> frozenset[str]:
        refs = {atom.ref for atom in self.contract_atoms}
        refs.update(branch.ref for branch in self.branches)
        refs.update(step.ref for step in self.value_flow)
        refs.update(edge.ref for edge in (*self.callers, *self.callees))
        refs.update(witness.ref for witness in self.witnesses)
        return frozenset(refs)

    @property
    def digest(self) -> str:
        """Content digest excludes ``focus_id`` so identical dossiers deduplicate."""

        return _digest(self._payload())

    @property
    def dossier_digest(self) -> str:
        return self.digest

    @property
    def estimated_tokens(self) -> int:
        return (len(self.render()) + 3) // 4

    @property
    def divergence_question(self) -> str:
        return (
            "Walk items 1-5 line by line. For each item, state the produced value, "
            "demanded value, and whether they match. Identify the first causal "
            "divergence. Then propose exactly one edit and one targeted test."
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": MODULE_SCHEMA,
            "region": self.region.to_dict(),
            "source_revision": self.source_revision,
            "source_digest": self.source_digest,
            "source_lines": [line.to_dict() for line in self.source_lines],
            "contract_atoms": [atom.to_dict() for atom in self.contract_atoms],
            "branches": [branch.to_dict() for branch in self.branches],
            "value_flow": [step.to_dict() for step in self.value_flow],
            "callers": [edge.to_dict() for edge in self.callers],
            "callees": [edge.to_dict() for edge in self.callees],
            "witnesses": [witness.to_dict() for witness in self.witnesses],
            "unknown_count": self.unknown_count,
            "unknown_digest": self.unknown_digest,
            "first_divergence_line": self.first_divergence_line,
            "first_divergence_reason": self.first_divergence_reason,
            "policy": self.policy.to_dict(),
            "laws": list(self.laws),
        }

    def to_dict(self) -> dict[str, object]:
        return {"schema": MODULE_SCHEMA, "focus_id": self.focus_id, "dossier_digest": self.digest, **self._payload()}

    def _render_source(self) -> list[str]:
        lines = ["1. EXACT REGION LINES (source-backed, inclusive):"]
        if not self.source_lines:
            lines.append("  UNKNOWN: source slice unavailable")
            return lines
        for item in self.source_lines:
            lines.append(f"  L{item.number}: {item.text}")
        return lines

    def _render_branches(self) -> list[str]:
        lines = ["2. BRANCH POLARITY vs CONTRACT ATOMS:"]
        if not self.branches:
            lines.append("  none observed in region")
            return lines
        for branch in self.branches:
            refs = ",".join(branch.contract_refs) or "none"
            lines.append(
                f"  [{branch.ref}] L{branch.line}-{branch.end_line} if {branch.condition}; "
                f"TRUE={branch.true_polarity}; FALSE={branch.false_polarity}; atoms={refs}"
            )
        return lines

    def _render_flow(self) -> list[str]:
        lines = ["3. VALUE FLOW (fact-compiler EFFECT lens, scoped):"]
        if not self.value_flow:
            lines.append("  UNKNOWN: no effect steps available")
            return lines
        for step in self.value_flow:
            refs = ",".join(step.contract_refs) or "none"
            lines.append(
                f"  [{step.ref}] L{step.line}-{step.end_line}: "
                f"{step.input_value} -> {step.operation} -> {step.produced}; "
                f"effect={step.effect}; atoms={refs}"
            )
        return lines

    def _render_graph(self) -> list[str]:
        lines = ["4. ONE-HOP CALL GRAPH (capped):"]
        edges = (*self.callers, *self.callees)
        if not edges:
            lines.append("  UNKNOWN: caller/callee evidence unavailable")
            return lines
        for edge in edges:
            location = f"{edge.path}:{edge.line}" if edge.path and edge.line else edge.path or "location unknown"
            reason = f"; {edge.reason}" if edge.reason else ""
            lines.append(f"  [{edge.ref}] {edge.relation} {edge.symbol} ({location}){reason}")
        return lines

    def _render_witnesses(self) -> list[str]:
        lines = ["5. WITNESS VECTORS (demanded beside produced, capped at four):"]
        if not self.witnesses:
            lines.append("  UNKNOWN: no witness vectors available")
            return lines
        for witness in self.witnesses:
            line = f"; line={witness.line}" if witness.line else ""
            refs = ",".join(witness.contract_refs) or witness.ref
            lines.append(
                f"  [{witness.ref}] demanded={witness.demanded}; produced={witness.produced}; "
                f"state={witness.state.value}; refs={refs}{line}"
            )
        return lines

    def render(self) -> str:
        """Render one numbered, attention-bounded artifact."""

        lines = [
            f"CLOSE-LOOK DOSSIER | focus={self.focus_id} | region={self.region.label}",
            f"source_revision={self.source_revision} | dossier_digest={self.digest}",
            f"policy=N{self.policy.same_anchor_red_threshold}/M{self.policy.in_region_edit_threshold} "
            f"actions={self.policy.max_open_actions} tokens<={self.policy.max_dossier_tokens}",
            *self._render_source(),
            *self._render_branches(),
            *self._render_flow(),
            *self._render_graph(),
            *self._render_witnesses(),
        ]
        if self.first_divergence_line is not None:
            lines.append(
                f"FIRST CAUSAL DIVERGENCE CANDIDATE: L{self.first_divergence_line} "
                f"{self.first_divergence_reason or 'witness mismatch'}"
            )
        if self.unknown_count:
            lines.append(f"UNKNOWN SUMMARY: count={self.unknown_count} digest={self.unknown_digest}")
        rendered = "\n".join(lines)
        max_chars = self.policy.max_dossier_tokens * 4
        if len(rendered) <= max_chars:
            return rendered + f"\nQUESTION: {self.divergence_question}"
        tail = "\n[attention cap reached; omitted content retained only by dossier digest]\nQUESTION: " + self.divergence_question
        return rendered[: max(1, max_chars - len(tail))] + tail

    serialize = render

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Dossier":
        """Rehydrate a dossier for answer validation without source access."""

        region = Region.from_value(value.get("region", {}))
        policy_value = value.get("policy", {})
        policy = CloseLookPolicy(**policy_value) if isinstance(policy_value, Mapping) else DEFAULT_POLICY
        source_lines = tuple(
            _source_line(item, fallback_number=region.start_line + index)
            for index, item in enumerate(_sequence(value.get("source_lines", ())))
        )
        atoms = _normalize_atoms(value.get("contract_atoms", ()))
        branches = _normalize_branches(value.get("branches", ()), atoms, region)
        effects = _normalize_effects(value.get("value_flow", ()), atoms, region)
        callers = _normalize_graph(value.get("callers", ()), relation="caller", limit=policy.max_callers)
        callees = _normalize_graph(value.get("callees", ()), relation="callee", limit=policy.max_callees)
        witnesses = _normalize_witnesses(value.get("witnesses", ()), limit=policy.max_witness_vectors)
        dossier = cls(
            focus_id=str(value.get("focus_id", "focus:rehydrated") or "focus:rehydrated"),
            region=region,
            source_revision=str(value.get("source_revision", "unknown") or "unknown"),
            source_digest=str(value.get("source_digest", "unknown") or "unknown"),
            source_lines=source_lines,
            contract_atoms=atoms,
            branches=branches,
            value_flow=effects,
            callers=callers,
            callees=callees,
            witnesses=witnesses,
            unknown_count=int(value.get("unknown_count", 0) or 0),
            unknown_digest=str(value.get("unknown_digest", "") or "") or None,
            first_divergence_line=_optional_int(value.get("first_divergence_line")),
            first_divergence_reason=str(value.get("first_divergence_reason", "") or ""),
            policy=policy,
            laws=tuple(str(item) for item in value.get("laws", ()) or ()),
        )
        supplied = value.get("dossier_digest", value.get("digest"))
        if supplied not in (None, "") and str(supplied) != dossier.digest:
            raise CloseLookError("dossier digest does not match its content")
        return dossier


def build_dossier(
    source: str,
    path: str,
    start_line: int,
    end_line: int,
    *,
    symbol: str = "",
    source_revision: str = "",
    focus_id: str = "focus:pending",
    contract_atoms: object = (),
    branches: object = (),
    value_flow: object = (),
    effects: object = (),
    callers: object = (),
    callees: object = (),
    witnesses: object = (),
    unknowns: object = (),
    first_divergence_line: int | None = None,
    first_divergence_reason: str = "",
    policy: CloseLookPolicy = DEFAULT_POLICY,
    effect_lens: Callable[[str, Region], object] | object | None = None,
) -> Dossier:
    """Assemble a deterministic, capped microscope dossier.

    ``effect_lens`` is an optional adapter seam for the existing fact compiler;
    callers can pass its already-computed EFFECT rows or a pure callable.  The
    fallback is a local AST summary with the same input/operation/produced/effect
    shape, keeping this slice executable before integration.
    """

    region = Region(path, start_line, end_line, symbol)
    source_text = str(source or "")
    source_digest = _digest(source_text)
    revision = str(source_revision or source_digest)
    raw_lines = source_text.splitlines()
    source_lines = tuple(
        SourceLine(number, raw_lines[number - 1])
        for number in range(region.start_line, min(region.end_line, len(raw_lines)) + 1)
    )
    atoms = _normalize_atoms(contract_atoms)
    branch_values = _normalize_branches(branches, atoms, region) if _sequence(branches) else _derive_branches(source_text, region, atoms)
    provided_effects = value_flow if _sequence(value_flow) else effects
    if effect_lens is not None:
        try:
            provided_effects = effect_lens(source_text, region) if callable(effect_lens) else effect_lens
        except (CloseLookError, OSError, RuntimeError, TypeError, ValueError):
            provided_effects = ()
    flow = _normalize_effects(provided_effects, atoms, region) if _sequence(provided_effects) else _derive_effects(source_text, region, atoms)
    caller_edges = _normalize_graph(callers, relation="caller", limit=policy.max_callers)
    callee_edges = _normalize_graph(callees, relation="callee", limit=policy.max_callees)
    witness_values = _normalize_witnesses(witnesses, limit=policy.max_witness_vectors)
    unknown_values = _sequence(unknowns)
    unknown_digest = _digest(unknown_values) if unknown_values else None
    unknown_count = len(unknown_values)
    if first_divergence_line is None:
        divergent = [item for item in witness_values if item.state is AnswerState.DIVERGES and item.line]
        if divergent:
            first_divergence_line = min(int(item.line) for item in divergent if item.line is not None)
    if not first_divergence_reason and first_divergence_line is not None:
        divergent = next((item for item in witness_values if item.line == first_divergence_line), None)
        if divergent:
            first_divergence_reason = f"produced {divergent.produced!r} diverges from demanded {divergent.demanded!r}"
    return Dossier(
        focus_id=focus_id,
        region=region,
        source_revision=revision,
        source_digest=source_digest,
        source_lines=source_lines[: policy.max_source_lines],
        contract_atoms=atoms,
        branches=branch_values,
        value_flow=flow,
        callers=caller_edges,
        callees=callee_edges,
        witnesses=witness_values,
        unknown_count=unknown_count,
        unknown_digest=unknown_digest,
        first_divergence_line=first_divergence_line,
        first_divergence_reason=first_divergence_reason,
        policy=policy,
    )


@dataclass(frozen=True, slots=True)
class WalkthroughItem:
    item: int
    produced: str
    demanded: str
    state: AnswerState
    refs: tuple[str, ...]
    line: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "item": self.item,
            "produced": self.produced,
            "demanded": self.demanded,
            "state": self.state.value,
            "refs": list(self.refs),
            "line": self.line,
        }


@dataclass(frozen=True, slots=True)
class FirstDivergence:
    line: int
    reason: str
    refs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"line": self.line, "reason": self.reason, "refs": list(self.refs)}


@dataclass(frozen=True, slots=True)
class ProposedEdit:
    path: str
    start_line: int
    end_line: int
    intent: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_relative_path(self.path))
        object.__setattr__(self, "start_line", _positive_int(self.start_line, "edit start_line"))
        object.__setattr__(self, "end_line", _positive_int(self.end_line, "edit end_line"))
        if self.end_line < self.start_line:
            raise CloseLookError("edit end_line must not precede start_line")
        object.__setattr__(self, "intent", _clean(self.intent, limit=500))
        if not self.intent:
            raise CloseLookError("one_edit.intent is required")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "intent": self.intent,
        }


@dataclass(frozen=True, slots=True)
class FocusAnswer:
    """The structured declaration the SNITCH check can later falsify."""

    focus_id: str
    dossier_digest: str
    walkthrough: tuple[WalkthroughItem, ...]
    first_divergence: FirstDivergence
    one_edit: ProposedEdit
    one_test: str

    def to_dict(self) -> dict[str, object]:
        return {
            "focus_id": self.focus_id,
            "dossier_digest": self.dossier_digest,
            "walkthrough": [item.to_dict() for item in self.walkthrough],
            "first_divergence": self.first_divergence.to_dict(),
            "one_edit": self.one_edit.to_dict(),
            "one_test": self.one_test,
        }


@dataclass(frozen=True, slots=True)
class AnswerValidation:
    valid: bool
    reasons: tuple[str, ...] = ()
    answer: FocusAnswer | None = None

    def to_dict(self) -> dict[str, object]:
        return {"valid": self.valid, "reasons": list(self.reasons), "answer": self.answer.to_dict() if self.answer else None}


def _parse_answer_item(value: object, *, valid_refs: frozenset[str]) -> tuple[WalkthroughItem | None, list[str]]:
    reasons: list[str] = []
    if not isinstance(value, Mapping):
        return None, ["walkthrough item must be an object"]
    number = _optional_int(value.get("item", value.get("number")))
    if number is None:
        reasons.append("walkthrough item number is missing")
        return None, reasons
    raw_state = str(value.get("state", "") or "").upper()
    try:
        state = AnswerState(raw_state)
    except ValueError:
        reasons.append(f"walkthrough item {number} has invalid state")
        state = AnswerState.UNKNOWN
    refs_value = value.get("refs", value.get("references", ()))
    refs = tuple(str(item) for item in _sequence(refs_value) if str(item).strip())
    if not refs:
        reasons.append(f"walkthrough item {number} has no evidence refs")
    unknown_refs = sorted(set(refs) - valid_refs)
    if unknown_refs:
        reasons.append(f"walkthrough item {number} cites unknown refs: {', '.join(unknown_refs)}")
    produced = _display(value.get("produced", "UNKNOWN"))
    demanded = _display(value.get("demanded", "UNKNOWN"))
    if state is AnswerState.MATCH and (
        produced.upper() == "UNKNOWN" or demanded.upper() == "UNKNOWN" or produced != demanded
    ):
        reasons.append(f"walkthrough item {number} MATCH state contradicts its values")
    elif state is AnswerState.DIVERGES and (
        produced.upper() == "UNKNOWN" or demanded.upper() == "UNKNOWN" or produced == demanded
    ):
        reasons.append(f"walkthrough item {number} DIVERGES state contradicts its values")
    elif state is AnswerState.UNKNOWN and produced.upper() != "UNKNOWN" and demanded.upper() != "UNKNOWN":
        reasons.append(f"walkthrough item {number} UNKNOWN state has two known values")
    line = _optional_int(value.get("line"))
    return WalkthroughItem(number, produced, demanded, state, refs, line), reasons


def validate_answer(
    answer: object,
    dossier: Dossier,
    *,
    expected_focus_id: str | None = None,
    expected_region: Region | None = None,
) -> AnswerValidation:
    """Validate the complete answer declaration without accepting model prose."""

    reasons: list[str] = []
    if isinstance(answer, FocusAnswer):
        return validate_answer(
            answer.to_dict(),
            dossier,
            expected_focus_id=expected_focus_id,
            expected_region=expected_region,
        )
    if not isinstance(answer, Mapping):
        return AnswerValidation(False, ("answer must be an object",))
    focus_id = str(answer.get("focus_id", "") or "")
    digest = str(answer.get("dossier_digest", "") or "")
    if not focus_id:
        reasons.append("focus_id is required")
    if expected_focus_id and focus_id != expected_focus_id:
        reasons.append("focus_id does not match the active focus")
    if not digest:
        reasons.append("dossier_digest is required")
    elif digest != dossier.digest:
        reasons.append("dossier_digest is stale")
    raw_walkthrough = answer.get("walkthrough")
    if isinstance(raw_walkthrough, Mapping) or isinstance(raw_walkthrough, (str, bytes)):
        reasons.append("walkthrough must be a numbered sequence")
        raw_items: tuple[object, ...] = ()
    else:
        raw_items = _sequence(raw_walkthrough)
    items: list[WalkthroughItem] = []
    for raw_item in raw_items:
        item, item_reasons = _parse_answer_item(raw_item, valid_refs=dossier.reference_ids)
        reasons.extend(item_reasons)
        if item is not None:
            items.append(item)
    for item in items:
        if item.line is not None and not dossier.region.contains(item.line):
            reasons.append(f"walkthrough item {item.item} line must be inside the focused region")
    numbers = [item.item for item in items]
    expected_numbers = list(range(1, WALKTHROUGH_ITEM_COUNT + 1))
    if sorted(numbers) != expected_numbers:
        missing = sorted(set(expected_numbers) - set(numbers))
        duplicates = sorted({number for number in numbers if numbers.count(number) > 1})
        if missing:
            reasons.append("walkthrough is missing numbered items: " + ", ".join(map(str, missing)))
        if duplicates:
            reasons.append("walkthrough repeats numbered items: " + ", ".join(map(str, duplicates)))
        if not missing and not duplicates and sorted(numbers) != expected_numbers:
            reasons.append("walkthrough numbering must be exactly 1-5")
    raw_first = answer.get("first_divergence")
    if not isinstance(raw_first, Mapping):
        reasons.append("first_divergence is required")
        first = None
    else:
        first_line = _optional_int(raw_first.get("line"))
        first_reason = _clean(raw_first.get("reason"), limit=500)
        first_refs = tuple(str(item) for item in _sequence(raw_first.get("refs", ())) if str(item).strip())
        if first_line is None or not (expected_region or dossier.region).contains(first_line):
            reasons.append("first_divergence.line must be inside the focused region")
            first_line = first_line or dossier.region.start_line
        if not first_reason:
            reasons.append("first_divergence.reason is required")
        if not first_refs:
            reasons.append("first_divergence must cite evidence refs")
        unknown_refs = sorted(set(first_refs) - dossier.reference_ids)
        if unknown_refs:
            reasons.append("first_divergence cites unknown refs: " + ", ".join(unknown_refs))
        first = FirstDivergence(first_line, first_reason or "UNKNOWN", first_refs)
    raw_edits = answer.get("edits")
    if raw_edits is not None:
        edit_values = _sequence(raw_edits)
        if len(edit_values) != 1:
            reasons.append("answer must propose exactly one edit")
    raw_edit = answer.get("one_edit")
    if isinstance(raw_edit, Sequence) and not isinstance(raw_edit, (str, bytes, Mapping)):
        reasons.append("one_edit must contain exactly one edit object, not a list")
        edit = None
    elif not isinstance(raw_edit, Mapping):
        reasons.append("one_edit is required")
        edit = None
    else:
        try:
            edit = ProposedEdit(
                path=str(raw_edit.get("path", "") or ""),
                start_line=_positive_int(raw_edit.get("start_line"), "one_edit.start_line"),
                end_line=_positive_int(raw_edit.get("end_line", raw_edit.get("start_line")), "one_edit.end_line"),
                intent=str(raw_edit.get("intent", "") or ""),
            )
            active_region = expected_region or dossier.region
            proposed_region = Region(edit.path, edit.start_line, edit.end_line)
            if not active_region.intersects(proposed_region):
                reasons.append("one_edit must intersect the focused region")
        except CloseLookError as exc:
            reasons.append(str(exc))
            edit = None
    one_test = _clean(answer.get("one_test"), limit=300)
    if not one_test:
        reasons.append("one_test is required")
    if first is None or edit is None:
        return AnswerValidation(False, tuple(dict.fromkeys(reasons)))
    candidate = FocusAnswer(
        focus_id=focus_id,
        dossier_digest=digest,
        walkthrough=tuple(sorted(items, key=lambda item: item.item)),
        first_divergence=first,
        one_edit=edit,
        one_test=one_test,
    )
    return AnswerValidation(not reasons, tuple(dict.fromkeys(reasons)), candidate if not reasons else candidate)


@dataclass(frozen=True, slots=True)
class FocusReceipt:
    """Open/close state projection carried into the situation card."""

    focus_id: str
    state: FocusState
    trigger: str
    region: Region
    opened_turn: int | None
    dossier_digest: str
    source_revision: str
    remaining_actions: int
    outside_challenge_used: bool
    shadow: bool = False
    would_fire: bool = False
    closed_turn: int | None = None
    close_result: str = ""
    dossier: Dossier | None = field(default=None, repr=False, compare=False)

    def to_dict(self, *, include_dossier: bool = False) -> dict[str, object]:
        value = {
            "schema": MODULE_SCHEMA,
            "focus_id": self.focus_id,
            "state": self.state.value,
            "trigger": self.trigger,
            "region": self.region.label,
            "opened_turn": self.opened_turn,
            "dossier_digest": self.dossier_digest,
            "source_revision": self.source_revision,
            "remaining_actions": self.remaining_actions,
            "outside_challenge_used": self.outside_challenge_used,
            "shadow": self.shadow,
            "would_fire": self.would_fire,
            "closed_turn": self.closed_turn,
            "close_result": self.close_result,
        }
        if include_dossier and self.dossier is not None:
            value["dossier"] = self.dossier.to_dict()
        return value

    def render_dossier(self) -> str:
        return self.dossier.render() if self.dossier is not None else ""


@dataclass(frozen=True, slots=True)
class FocusRow:
    """Optional ``focus`` row for the existing SituationCard projection."""

    state: FocusState
    region: str
    trigger: str
    opened_turn: int | None
    required_response: str
    dossier_digest: str
    remaining_actions: int
    outside_challenge_used: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "region": self.region,
            "trigger": self.trigger,
            "opened_turn": self.opened_turn,
            "required_response": self.required_response,
            "dossier_digest": self.dossier_digest,
            "remaining_actions": self.remaining_actions,
            "outside_challenge_used": self.outside_challenge_used,
        }


@dataclass(frozen=True, slots=True)
class CloseReceipt:
    focus_id: str
    dossier_digest: str
    turn: int
    result_class: str
    source_revision: str
    anchor_key: str

    def to_dict(self) -> dict[str, object]:
        return {
            "focus_id": self.focus_id,
            "dossier_digest": self.dossier_digest,
            "turn": self.turn,
            "result_class": self.result_class,
            "source_revision": self.source_revision,
            "anchor_key": self.anchor_key,
        }


@dataclass(frozen=True, slots=True)
class EditDecision:
    allowed: bool
    state: FocusState
    in_region: bool
    challenge: bool = False
    message: str = ""
    reason: str = ""
    triggered: bool = False
    would_fire: bool = False
    focus: FocusReceipt | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "state": self.state.value,
            "in_region": self.in_region,
            "challenge": self.challenge,
            "message": self.message,
            "reason": self.reason,
            "triggered": self.triggered,
            "would_fire": self.would_fire,
            "focus": self.focus.to_dict() if self.focus else None,
        }


@dataclass(frozen=True, slots=True)
class PivotDecision:
    allowed: bool
    requested_tier: int
    state: FocusState
    event: str
    message: str = ""
    focus: FocusReceipt | None = None
    would_fire: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "requested_tier": self.requested_tier,
            "state": self.state.value,
            "event": self.event,
            "message": self.message,
            "focus": self.focus.to_dict() if self.focus else None,
            "would_fire": self.would_fire,
        }


@dataclass(frozen=True, slots=True)
class KernelDecision:
    action: str
    state: FocusState
    allowed: bool = True
    opened: bool = False
    closed: bool = False
    would_fire: bool = False
    delivered: bool = False
    trigger: str = ""
    turn: int | None = None
    message: str = ""
    focus: FocusReceipt | None = None
    edit: EditDecision | None = None
    pivot: PivotDecision | None = None
    validation: AnswerValidation | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "state": self.state.value,
            "allowed": self.allowed,
            "opened": self.opened,
            "closed": self.closed,
            "would_fire": self.would_fire,
            "delivered": self.delivered,
            "trigger": self.trigger,
            "turn": self.turn,
            "message": self.message,
            "focus": self.focus.to_dict() if self.focus else None,
            "edit": self.edit.to_dict() if self.edit else None,
            "pivot": self.pivot.to_dict() if self.pivot else None,
            "validation": self.validation.to_dict() if self.validation else None,
        }


FocusDecision = KernelDecision


@dataclass(frozen=True, slots=True)
class CloseLookEvent:
    event: str
    turn: int | None
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"cylinder": "closelook", "event": self.event, "turn": self.turn, "detail": self.detail}


class CloseLookKernel:
    """Per-task close-look state machine; no filesystem or model mutation."""

    def __init__(
        self,
        *,
        task_id: str = "task",
        workspace: str = "workspace",
        region: Region | Mapping[str, object] | None = None,
        anchor: Anchor | Mapping[str, object] | str | None = None,
        source: str = "",
        symbol: str = "",
        source_revision: str = "",
        contract_atoms: object = (),
        branches: object = (),
        value_flow: object = (),
        effects: object = (),
        callers: object = (),
        callees: object = (),
        witnesses: object = (),
        unknowns: object = (),
        policy: CloseLookPolicy = DEFAULT_POLICY,
        shadow: bool = False,
        archived_replay: bool = False,
        effect_lens: Callable[[str, Region], object] | object | None = None,
        first_divergence_line: int | None = None,
        first_divergence_reason: str = "",
    ) -> None:
        self.task_id = _clean(task_id, limit=160) or "task"
        self.workspace = _clean(workspace, limit=160) or "workspace"
        self.policy = policy
        self.shadow = bool(shadow)
        self.archived_replay = bool(archived_replay)
        self.source = str(source or "")
        self.source_revision = str(source_revision or _digest(self.source))
        self._target_region = Region.from_value(region, symbol=symbol) if region is not None else None
        self._target_anchor = (
            normalize_anchor(
                anchor,
                workspace=self.workspace,
                region=self._target_region,
                symbol=symbol,
            )
            if anchor is not None
            else (
                Anchor(
                    self.workspace,
                    self._target_region.path,
                    self._target_region.symbol or symbol or "unknown",
                    self._target_region.start_line,
                    self._target_region.end_line,
                )
                if self._target_region is not None
                else None
            )
        )
        if self._target_region is None and self._target_anchor is not None:
            self._target_region = self._target_anchor.region
        self._dossier_inputs = {
            "contract_atoms": contract_atoms,
            "branches": branches,
            "value_flow": value_flow,
            "effects": effects,
            "callers": callers,
            "callees": callees,
            "witnesses": witnesses,
            "unknowns": unknowns,
            "effect_lens": effect_lens,
            "first_divergence_line": first_divergence_line,
            "first_divergence_reason": first_divergence_reason,
        }
        self.state = FocusState.IDLE
        self._focus: FocusReceipt | None = None
        self._answer: FocusAnswer | None = None
        self._close_receipt: CloseReceipt | None = None
        self._last_shadow_dossier: Dossier | None = None
        self._red_counts: dict[str, int] = {}
        self._edit_counts: dict[str, int] = {}
        self._pending_edit: dict[str, int] = {}
        self._known_receipt_ids: set[str] = set()
        self._last_receipt_turn: int | None = None
        self._action_count = 0
        self._outside_challenge_used = False
        self._outside_exception_used = False
        self._outside_rejection_seen: set[str] = set()
        self._pivot_challenge_emitted = False
        self._events: list[CloseLookEvent] = []
        self._record(
            "CLOSELOOK_ARMED",
            f"task={self.task_id} policy={canonical_json(self.policy.to_dict())} shadow={self.shadow}",
        )

    @property
    def trace(self) -> tuple[dict[str, object], ...]:
        return tuple(event.to_dict() for event in self._events)

    @property
    def events(self) -> tuple[CloseLookEvent, ...]:
        return tuple(self._events)

    @property
    def focus(self) -> FocusReceipt | None:
        return self._focus

    @property
    def dossier(self) -> Dossier | None:
        return self._focus.dossier if self._focus else None

    @property
    def answer(self) -> FocusAnswer | None:
        return self._answer

    @property
    def close_receipt(self) -> CloseReceipt | None:
        return self._close_receipt

    @property
    def action_count(self) -> int:
        return self._action_count

    @property
    def remaining_actions(self) -> int:
        return max(0, self.policy.max_open_actions - self._action_count)

    @property
    def target_anchor(self) -> Anchor | None:
        return self._target_anchor

    @property
    def target_region(self) -> Region | None:
        return self._target_region

    def _record(self, event: str, detail: object, turn: int | None = None) -> None:
        self._events.append(CloseLookEvent(event, turn, _clean(detail, limit=600)))

    def _receipt_turn(self, record: Mapping[str, object]) -> int | None:
        value = record.get("turn", record.get("receipt_turn"))
        return _optional_int(value)

    def _record_id(self, record: Mapping[str, object]) -> str:
        # Fingerprints identify the operation, not an individual receipt.  A
        # repeated RED test can legitimately reuse one fingerprint across
        # receipt turns, so only explicit receipt identities are deduplicated.
        raw = record.get("receipt_id", record.get("id", ""))
        return _clean(raw, limit=180)

    def _result_class(self, record: Mapping[str, object]) -> str:
        return str(record.get("result_class", record.get("result", "")) or "").strip().upper()

    def _is_test_side(self, record: Mapping[str, object]) -> bool:
        if record.get("test_side") is True or record.get("infrastructure") is True:
            return True
        kind = str(record.get("kind", "") or "").strip().lower()
        source = str(record.get("source", "") or "").strip().lower()
        return kind in _IGNORED_KINDS or source in _IGNORED_KINDS or self._result_class(record) == "TEST_SIDE"

    def _is_valid_product_test(self, record: Mapping[str, object]) -> bool:
        kind = str(record.get("kind", "") or "").strip().lower()
        result = self._result_class(record)
        if kind != "test" or result not in _VALID_TEST_RESULTS or self._is_test_side(record):
            return False
        if record.get("targeted") is False or record.get("valid_targeted") is False:
            return False
        return result not in _RESULTS_THAT_ARE_NOT_PRODUCT_EVIDENCE

    def _is_valid_product_red(self, record: Mapping[str, object]) -> bool:
        return self._is_valid_product_test(record) and self._result_class(record) in _VALID_RED_RESULTS

    def _is_valid_product_green(self, record: Mapping[str, object]) -> bool:
        return self._is_valid_product_test(record) and self._result_class(record) in _VALID_GREEN_RESULTS

    def _record_anchor(self, record: Mapping[str, object], *, fallback: bool = False) -> Anchor | None:
        raw = record.get("anchor", record.get("primary_anchor"))
        if raw in (None, ""):
            path = record.get("path", record.get("target_path", ""))
            line = record.get("line", record.get("start_line"))
            if path not in (None, "") and line not in (None, ""):
                raw = {
                    "path": path,
                    "line": line,
                    "end_line": record.get("end_line", line),
                    "symbol": record.get("symbol", record.get("target_symbol", "")),
                    "contract_fingerprints": record.get("contract_fingerprints", ()),
                }
        if raw not in (None, ""):
            try:
                return normalize_anchor(
                    raw,
                    workspace=str(record.get("workspace", self.workspace) or self.workspace),
                    region=self._target_region,
                    symbol=str(record.get("symbol", record.get("target_symbol", "")) or ""),
                )
            except CloseLookError:
                return None
        if fallback and self.archived_replay and self._target_anchor is not None:
            if self._target_anchor.fallback_used:
                return self._target_anchor
            return replace(self._target_anchor, fallback_used=True)
        return None

    def _anchor_key(self, anchor: Anchor | None) -> str:
        return anchor.stable_key if anchor is not None else ""

    def _same_scope(self, left: Anchor | None, right: Anchor | None) -> bool:
        if left is None or right is None:
            return False
        return left.stable_key == right.stable_key

    def _edit_path(self, record: Mapping[str, object]) -> str:
        if "target_path" in record:
            target = record["target_path"]
            return target if isinstance(target, str) else ""
        raw = record.get("path", record.get("file", record.get("target_path", "")))
        if raw not in (None, ""):
            return str(raw)
        intent = str(record.get("intent", "") or "").strip()
        if not intent:
            return ""
        pieces = intent.split()
        return pieces[-1] if len(pieces) > 1 else ""

    def _edit_region(self, record: Mapping[str, object]) -> Region | None:
        path = self._edit_path(record)
        if not path:
            return None
        start = _optional_int(record.get("start_line", record.get("line")))
        end = _optional_int(record.get("end_line", record.get("end")))
        spans = record.get("changed_spans", record.get("spans", ()))
        if (start is None or end is None) and spans:
            first = _sequence(spans)[0] if _sequence(spans) else None
            if isinstance(first, Mapping):
                start = start or _optional_int(first.get("start_line", first.get("line")))
                end = end or _optional_int(first.get("end_line", first.get("end", start)))
        lines = record.get("changed_lines", record.get("lines", ()))
        if (start is None or end is None) and isinstance(lines, Sequence) and not isinstance(lines, (str, bytes)):
            parsed = [_optional_int(item) for item in lines]
            parsed = [item for item in parsed if item is not None]
            if parsed:
                start, end = min(parsed), max(parsed)
        if start is None or end is None:
            return None
        try:
            return Region(path, start, end)
        except CloseLookError:
            return None

    def _scope_region(self, anchor: Anchor | None) -> Region | None:
        return self._target_region or (anchor.region if anchor is not None else None)

    def _build_dossier(self, *, focus_id: str) -> Dossier:
        if self._target_region is None:
            raise CloseLookError("cannot render close-look dossier without a target region")
        return build_dossier(
            self.source,
            self._target_region.path,
            self._target_region.start_line,
            self._target_region.end_line,
            symbol=self._target_region.symbol,
            source_revision=self.source_revision,
            focus_id=focus_id,
            policy=self.policy,
            **self._dossier_inputs,
        )

    def _make_focus_receipt(
        self,
        *,
        trigger: str,
        turn: int | None,
        anchor: Anchor | None,
        dossier: Dossier,
        shadow: bool,
        would_fire: bool,
        state: FocusState,
    ) -> FocusReceipt:
        focus_id = dossier.focus_id
        return FocusReceipt(
            focus_id=focus_id,
            state=state,
            trigger=trigger,
            region=dossier.region,
            opened_turn=turn,
            dossier_digest=dossier.digest,
            source_revision=dossier.source_revision,
            remaining_actions=self.remaining_actions,
            outside_challenge_used=self._outside_challenge_used,
            shadow=shadow,
            would_fire=would_fire,
            dossier=dossier,
        )

    def _open(self, *, trigger: str, turn: int | None, anchor: Anchor | None) -> KernelDecision:
        if self.state not in {FocusState.IDLE, FocusState.ABORTED}:
            self._record("CLOSELOOK_DEDUPED", f"trigger={trigger} state={self.state.value}", turn)
            return KernelDecision(
                action="focus_deduped",
                state=self.state,
                allowed=False,
                trigger=trigger,
                turn=turn,
                message="close-look is already active; answer it, close it, or abort it",
                focus=self._focus,
            )
        if self.state is FocusState.ABORTED:
            self._record("CLOSELOOK_REARMED", "new focus after explicit escape", turn)
            self.state = FocusState.IDLE
            self._action_count = 0
            self._outside_challenge_used = False
            self._outside_exception_used = False
            self._answer = None
            self._close_receipt = None
        if self._target_region is None and anchor is not None:
            self._target_region = anchor.region
        if self._target_region is None:
            self._record("CLOSELOOK_SUPPRESSED", "no target region", turn)
            return KernelDecision(
                action="focus_suppressed",
                state=self.state,
                allowed=True,
                trigger=trigger,
                turn=turn,
                message="close-look needs an anchored repository region",
            )
        key = self._anchor_key(anchor or self._target_anchor)
        focus_id = f"focus:{key[:20] or _digest(self._target_region.label)[:20]}:{turn if turn is not None else 'manual'}"
        try:
            dossier = self._build_dossier(focus_id=focus_id)
        except CloseLookError as exc:
            self._record("CLOSELOOK_SUPPRESSED", f"dossier error={type(exc).__name__}", turn)
            return KernelDecision(
                action="focus_suppressed",
                state=self.state,
                allowed=True,
                trigger=trigger,
                turn=turn,
                message=f"close-look dossier unavailable: {exc}",
            )
        if self.shadow:
            self._last_shadow_dossier = dossier
            receipt = self._make_focus_receipt(
                trigger=trigger,
                turn=turn,
                anchor=anchor,
                dossier=dossier,
                shadow=True,
                would_fire=True,
                state=FocusState.OPEN,
            )
            self._record("CLOSELOOK_WOULD_FIRE", f"trigger={trigger} digest={dossier.digest}", turn)
            self._record("CLOSELOOK_SHADOWED", "dossier suppressed from model context", turn)
            return KernelDecision(
                action="focus_would_fire",
                state=FocusState.IDLE,
                allowed=True,
                opened=True,
                would_fire=True,
                delivered=False,
                trigger=trigger,
                turn=turn,
                focus=receipt,
            )
        self.state = FocusState.OPEN
        self._focus = self._make_focus_receipt(
            trigger=trigger,
            turn=turn,
            anchor=anchor,
            dossier=dossier,
            shadow=False,
            would_fire=False,
            state=FocusState.OPEN,
        )
        self._action_count = 0
        self._answer = None
        self._close_receipt = None
        self._outside_challenge_used = False
        self._outside_exception_used = False
        self._record("CLOSELOOK_OPENED", f"trigger={trigger} digest={dossier.digest}", turn)
        return KernelDecision(
            action="focus_opened",
            state=self.state,
            allowed=False,
            opened=True,
            delivered=True,
            trigger=trigger,
            turn=turn,
            message=dossier.divergence_question,
            focus=self._focus,
        )

    def open_focus(
        self,
        *,
        trigger: str = "explicit_focus",
        turn: int | None = None,
        region: Region | Mapping[str, object] | str | None = None,
        anchor: Anchor | Mapping[str, object] | str | None = None,
    ) -> KernelDecision:
        """Open explicit focus, or return the active focus without duplicating it."""

        if region is not None:
            self._target_region = Region.from_value(region, symbol=self._target_region.symbol if self._target_region else "")
        if anchor is not None:
            self._target_anchor = normalize_anchor(
                anchor,
                workspace=self.workspace,
                region=self._target_region,
                symbol=self._target_region.symbol if self._target_region else "",
            )
            if self._target_region is None:
                self._target_region = self._target_anchor.region
        return self._open(trigger=trigger, turn=turn, anchor=self._target_anchor)

    def handle_focus_command(self, command: str, *, turn: int | None = None) -> KernelDecision:
        parsed = parse_focus_command(command, symbol=self._target_region.symbol if self._target_region else "")
        if parsed.action == "abort":
            return self.abort_focus(turn=turn, reason="explicit focus abort")
        return self.open_focus(trigger="explicit_focus", turn=turn, region=parsed.region)

    def _consume_action(self, *, turn: int | None, label: str) -> bool:
        if self.state not in {FocusState.OPEN, FocusState.ANSWERED, FocusState.EDITED}:
            return True
        if self._action_count >= self.policy.max_open_actions:
            self._abort_internal(turn=turn, reason="open action budget exhausted")
            return False
        self._action_count += 1
        self._record("CLOSELOOK_ACTION", f"{label} remaining={self.remaining_actions}", turn)
        if self._action_count >= self.policy.max_open_actions:
            self._abort_internal(turn=turn, reason="open action budget exhausted")
            return False
        return True

    def _update_focus_receipt(self, **changes: object) -> None:
        if self._focus is not None:
            self._focus = replace(
                self._focus,
                remaining_actions=self.remaining_actions,
                outside_challenge_used=self._outside_challenge_used,
                **changes,
            )

    def _abort_internal(self, *, turn: int | None, reason: str) -> None:
        self.state = FocusState.ABORTED
        self._record("CLOSELOOK_ABORTED", reason, turn)
        self._update_focus_receipt(state=FocusState.ABORTED)

    def abort_focus(self, *, turn: int | None = None, reason: str = "explicit focus abort") -> KernelDecision:
        if self.state in {FocusState.IDLE, FocusState.ABORTED}:
            self._record("CLOSELOOK_ABORT_DEDUPED", reason, turn)
            return KernelDecision(action="focus_abort", state=self.state, allowed=True, turn=turn, message="")
        self._abort_internal(turn=turn, reason=reason)
        return KernelDecision(action="focus_abort", state=self.state, allowed=True, turn=turn, message="focus escaped")

    def _justification_refs(self, justification: object) -> tuple[str, ...]:
        if not isinstance(justification, Mapping):
            return ()
        raw = justification.get("refs", justification.get("references", ()))
        return tuple(str(item) for item in _sequence(raw) if str(item).strip())

    def _valid_justification(self, justification: object) -> bool:
        if not isinstance(justification, Mapping):
            return False
        reason = _clean(justification.get("reason"), limit=500)
        refs = self._justification_refs(justification)
        valid_refs = self.dossier.reference_ids if self.dossier else frozenset()
        return bool(reason and refs and set(refs).issubset(valid_refs))

    def check_edit(
        self,
        edit: Mapping[str, object],
        *,
        justification: object = None,
        turn: int | None = None,
    ) -> EditDecision:
        """Apply region discipline to one proposed/applied edit without mutating code."""

        path = self._edit_path(edit)
        edit_region = self._edit_region(edit)
        active_region = self._target_region or (self._focus.region if self._focus else None)
        in_region = bool(active_region and edit_region and active_region.intersects(edit_region))
        if "target_path" in edit and active_region and edit_region:
            in_region = in_region and active_region.path == edit_region.path
        if self.state in {FocusState.IDLE, FocusState.CLOSED, FocusState.ABORTED}:
            return EditDecision(True, self.state, in_region, reason="no_active_focus")
        if in_region:
            if self.state is FocusState.OPEN:
                allowed = self._consume_action(turn=turn, label="in-region edit before answer")
                return EditDecision(
                    False if allowed else self.state is not FocusState.ABORTED,
                    self.state,
                    True,
                    message="[closelook] answer the numbered dossier before editing the focused region",
                    reason="answer_required",
                )
            if self.state is FocusState.ANSWERED:
                return EditDecision(True, self.state, True, reason="focused_edit_allowed")
            return EditDecision(
                False,
                self.state,
                True,
                message="[closelook] exactly one in-scope edit is allowed; run the named test or abort focus",
                reason="one_edit_already_used",
            )
        if self._valid_justification(justification) and not self._outside_exception_used:
            if not self._consume_action(turn=turn, label="justified outside edit"):
                return EditDecision(
                    False,
                    self.state,
                    False,
                    message="[closelook] open action budget exhausted; abort or rearm focus",
                    reason="action_budget_exhausted",
                )
            self._outside_exception_used = True
            self._record("CLOSELOOK_OUTSIDE_JUSTIFIED", f"path={path}", turn)
            return EditDecision(True, self.state, False, reason="justified_exception")
        key = path or "unknown-edit"
        if not self._outside_challenge_used:
            self._outside_challenge_used = True
            self._record("CLOSELOOK_CHALLENGE_ONCE", f"outside edit path={key}", turn)
            self._update_focus_receipt()
            return EditDecision(
                False,
                self.state,
                False,
                challenge=True,
                message=(
                    f"[closelook] You declared focus on {active_region.label if active_region else 'the active region'}; "
                    "this edit is elsewhere. Close the look or provide a source/evidence-backed justification."
                ),
                reason="outside_region_challenge_once",
            )
        if key not in self._outside_rejection_seen:
            self._outside_rejection_seen.add(key)
            self._record("CLOSELOOK_OUTSIDE_REJECTED", f"path={key} (challenge already used)", turn)
        return EditDecision(False, self.state, False, reason="outside_region_rejected")

    def _observe_edit(self, record: Mapping[str, object], *, turn: int | None) -> KernelDecision:
        result = self._result_class(record)
        explicit_empty = any(
            record.get(name) is False for name in ("applied", "changed", "non_empty")
        ) or record.get("no_op") is True
        applied = result == "APPLIED" and not explicit_empty and not self._is_test_side(record)
        if not applied:
            self._record("CLOSELOOK_EDIT_IGNORED", f"result={result or 'UNKNOWN'}", turn)
            return KernelDecision(action="edit_ignored", state=self.state, allowed=True, turn=turn, message="edit is not counted")
        decision = self.check_edit(record, turn=turn)
        if not decision.allowed:
            return KernelDecision(
                action="edit_blocked",
                state=self.state,
                allowed=False,
                turn=turn,
                message=decision.message,
                edit=decision,
            )
        edit_region = self._edit_region(record)
        explicit_anchor = record.get("anchor", record.get("primary_anchor"))
        anchor = (
            self._record_anchor(record, fallback=False)
            if explicit_anchor not in (None, "")
            else self._target_anchor
        )
        if anchor is None:
            anchor = self._record_anchor(record, fallback=False)
        active_region = self._target_region or self._scope_region(anchor)
        if self.state is FocusState.ANSWERED and decision.in_region:
            if not self._consume_action(turn=turn, label="one in-region edit"):
                return KernelDecision(action="edit_blocked", state=self.state, allowed=False, turn=turn, edit=decision)
            self.state = FocusState.EDITED
            revision = str(record.get("source_revision", self.source_revision) or self.source_revision)
            self.source_revision = revision
            self._update_focus_receipt(state=FocusState.EDITED)
            self._record("CLOSELOOK_EDIT_ACCEPTED", f"path={self._edit_path(record)}", turn)
            return KernelDecision(action="edit_accepted", state=self.state, allowed=True, turn=turn, edit=decision)
        if self.state in {FocusState.OPEN, FocusState.ANSWERED, FocusState.EDITED}:
            return KernelDecision(action="edit_accepted", state=self.state, allowed=True, turn=turn, edit=decision)
        if active_region is None or edit_region is None or not active_region.intersects(edit_region):
            return KernelDecision(action="edit_observed", state=self.state, allowed=True, turn=turn, edit=decision)
        if anchor is None:
            self._record("CLOSELOOK_EDIT_UNKNOWN", "cannot form stable anchor", turn)
            return KernelDecision(action="edit_observed", state=self.state, allowed=True, turn=turn, edit=decision)
        key = anchor.stable_key
        self._edit_counts[key] = self._edit_counts.get(key, 0) + 1
        self._pending_edit[key] = turn if turn is not None else (self._last_receipt_turn or 0)
        self._record("CLOSELOOK_EDIT_COUNTED", f"anchor={key[:16]} count={self._edit_counts[key]}", turn)
        if self._edit_counts[key] >= self.policy.in_region_edit_threshold:
            trigger = f"in_region_edit_{self.policy.in_region_edit_threshold}"
            opened = self._open(trigger=trigger, turn=turn, anchor=anchor)
            return replace(
                opened,
                action="edit_triggered" if opened.opened else opened.action,
                allowed=True,
                edit=replace(decision, triggered=opened.opened, would_fire=opened.would_fire, focus=opened.focus),
            )
        return KernelDecision(action="edit_observed", state=self.state, allowed=True, turn=turn, edit=decision)

    def observe_edit(self, record: Mapping[str, object]) -> KernelDecision:
        """Observe an edit receipt; applied edits remain the caller's responsibility."""

        turn = self._receipt_turn(record)
        self._last_receipt_turn = turn if turn is not None else self._last_receipt_turn
        return self._observe_edit(record, turn=turn)

    def _green_matches_pending_edit(self, anchor: Anchor | None, turn: int | None) -> bool:
        if anchor is None or turn is None:
            return False
        edit_turn = self._pending_edit.get(anchor.stable_key)
        return edit_turn is not None and edit_turn < turn

    def _observe_test(self, record: Mapping[str, object], *, turn: int | None) -> KernelDecision:
        result = self._result_class(record)
        anchor = self._record_anchor(record, fallback=True)
        if self.state is FocusState.EDITED and self._is_valid_product_test(record):
            focus_anchor = self._target_anchor or anchor
            if anchor is None or focus_anchor is None or self._same_scope(anchor, focus_anchor):
                if self._consume_action(turn=turn, label=f"close test {result}"):
                    self.state = FocusState.CLOSED
                    self._close_receipt = CloseReceipt(
                        focus_id=self._focus.focus_id if self._focus else "",
                        dossier_digest=self.dossier.digest if self.dossier else "",
                        turn=turn if turn is not None else self._last_receipt_turn or 0,
                        result_class=result,
                        source_revision=self.source_revision,
                        anchor_key=(focus_anchor or anchor).stable_key if (focus_anchor or anchor) else "",
                    )
                    self._update_focus_receipt(
                        state=FocusState.CLOSED,
                        closed_turn=self._close_receipt.turn,
                        close_result=result,
                    )
                    self._record("CLOSELOOK_CLOSED", f"test result={result}", turn)
                    return KernelDecision(action="focus_closed", state=self.state, allowed=True, closed=True, turn=turn, message="focus closed on one targeted test")
        if self.state is FocusState.OPEN:
            allowed = self._consume_action(turn=turn, label="test before answer")
            return KernelDecision(
                action="test_blocked",
                state=self.state,
                allowed=False if allowed else self.state is FocusState.ABORTED,
                turn=turn,
                message="[closelook] answer the numbered dossier before running the closing test",
            )
        if self.state is FocusState.ANSWERED:
            allowed = self._consume_action(turn=turn, label="test before edit")
            return KernelDecision(
                action="test_blocked",
                state=self.state,
                allowed=False if allowed else self.state is FocusState.ABORTED,
                turn=turn,
                message="[closelook] apply exactly one focused edit before the closing test",
            )
        if self._is_valid_product_green(record) and self._green_matches_pending_edit(anchor, turn):
            key = anchor.stable_key if anchor else ""
            self._red_counts[key] = 0
            self._edit_counts[key] = 0
            self._pending_edit.pop(key, None)
            self._record("CLOSELOOK_GREEN_RESET", f"same_scope anchor={key[:16]}", turn)
            return KernelDecision(action="green_reset", state=self.state, allowed=True, turn=turn)
        if self._is_valid_product_red(record) and anchor is not None:
            key = anchor.stable_key
            self._red_counts[key] = self._red_counts.get(key, 0) + 1
            self._target_anchor = self._target_anchor or anchor
            self._target_region = self._target_region or anchor.region
            self._record("CLOSELOOK_RED_COUNTED", f"anchor={key[:16]} count={self._red_counts[key]}", turn)
            if self._red_counts[key] >= self.policy.same_anchor_red_threshold:
                trigger = f"same_anchor_red_{self.policy.same_anchor_red_threshold}"
                opened = self._open(trigger=trigger, turn=turn, anchor=anchor)
                return replace(opened, action="red_triggered" if opened.opened else opened.action)
        if result in _RESULTS_THAT_ARE_NOT_PRODUCT_EVIDENCE or not self._is_valid_product_test(record):
            self._record("CLOSELOOK_TEST_IGNORED", f"result={result or 'UNKNOWN'}", turn)
            return KernelDecision(action="test_ignored", state=self.state, allowed=True, turn=turn)
        return KernelDecision(action="test_observed", state=self.state, allowed=True, turn=turn)

    def observe_test(self, record: Mapping[str, object]) -> KernelDecision:
        """Observe one test receipt; RED/GREEN eligibility is structural."""

        turn = self._receipt_turn(record)
        receipt_id = self._record_id(record)
        if receipt_id:
            if receipt_id in self._known_receipt_ids:
                self._record("CLOSELOOK_RECEIPT_DEDUPED", receipt_id, turn)
                return KernelDecision(action="receipt_deduped", state=self.state, allowed=True, turn=turn)
            self._known_receipt_ids.add(receipt_id)
        self._last_receipt_turn = turn if turn is not None else self._last_receipt_turn
        return self._observe_test(record, turn=turn)

    def process_receipt(self, record: Mapping[str, object]) -> KernelDecision:
        """Route a receipt by kind; this is the future ``KernelBolt.receipt`` adapter."""

        kind = str(record.get("kind", "") or "").strip().lower()
        if self.state is FocusState.ABORTED and kind in {"test", "edit"}:
            self._record("CLOSELOOK_REARMED", "receipt after escape", self._receipt_turn(record))
            self.state = FocusState.IDLE
            self._focus = None
            self._answer = None
            self._close_receipt = None
            self._action_count = 0
        if kind == "test":
            return self.observe_test(record)
        if kind == "edit":
            return self.observe_edit(record)
        self._record("CLOSELOOK_RECEIPT_IGNORED", f"kind={kind or 'UNKNOWN'}", self._receipt_turn(record))
        return KernelDecision(action="receipt_ignored", state=self.state, allowed=True, turn=self._receipt_turn(record))

    observe_receipt = process_receipt
    on_receipt = process_receipt

    def accept_answer(self, answer: object, *, turn: int | None = None) -> KernelDecision:
        """Accept the numbered declaration or leave focus open with bounded feedback."""

        if self.state is not FocusState.OPEN or self.dossier is None or self._focus is None:
            validation = AnswerValidation(False, ("no open close-look is awaiting an answer",))
            return KernelDecision(action="answer_rejected", state=self.state, allowed=False, turn=turn, validation=validation)
        if not self._consume_action(turn=turn, label="answer dossier"):
            validation = AnswerValidation(False, ("close-look action budget exhausted",))
            return KernelDecision(action="answer_rejected", state=self.state, allowed=False, turn=turn, validation=validation)
        validation = validate_answer(
            answer,
            self.dossier,
            expected_focus_id=self._focus.focus_id,
            expected_region=self._focus.region,
        )
        if not validation.valid:
            self._record("CLOSELOOK_ANSWER_REJECTED", "; ".join(validation.reasons), turn)
            return KernelDecision(
                action="answer_rejected",
                state=self.state,
                allowed=False,
                turn=turn,
                message="; ".join(validation.reasons),
                focus=self._focus,
                validation=validation,
            )
        self._answer = validation.answer
        self.state = FocusState.ANSWERED
        self._update_focus_receipt(state=FocusState.ANSWERED)
        self._record("CLOSELOOK_ANSWERED", f"digest={self.dossier.digest}", turn)
        return KernelDecision(
            action="answer_accepted",
            state=self.state,
            allowed=True,
            turn=turn,
            message="answer accepted; apply exactly one in-region edit",
            focus=self._focus,
            validation=validation,
        )

    submit_answer = accept_answer

    def update_source_revision(self, revision: str, *, source: str | None = None) -> None:
        """Invalidate active/closed focus when the source snapshot changes."""

        new_revision = _clean(revision, limit=180)
        if not new_revision:
            raise CloseLookError("source revision is required")
        if source is not None:
            self.source = str(source)
        if new_revision == self.source_revision:
            return
        old_state = self.state
        self.source_revision = new_revision
        if old_state is not FocusState.IDLE:
            self._record("CLOSELOOK_SOURCE_INVALIDATED", f"{old_state.value}->{new_revision}")
            self.state = FocusState.IDLE
            self._focus = None
            self._answer = None
            self._close_receipt = None
            self._action_count = 0

    def rebuild_dossier(self, *, revision: str | None = None, source: str | None = None) -> Dossier:
        """Build a fresh dossier after an admissible source/evidence revision."""

        if revision is not None:
            self.update_source_revision(revision, source=source)
        elif source is not None:
            self.source = str(source)
        if self._target_region is None:
            raise CloseLookError("cannot rebuild dossier without a target region")
        return self._build_dossier(focus_id="focus:rebuild")

    def valid_close_receipt(
        self,
        *,
        source_revision: str | None = None,
        anchor: Anchor | Mapping[str, object] | str | None = None,
    ) -> bool:
        """Return whether the close receipt can license a T2+ pivot."""

        receipt = self._close_receipt
        if self.state is not FocusState.CLOSED or receipt is None:
            return False
        if source_revision not in (None, "") and str(source_revision) != receipt.source_revision:
            return False
        if receipt.source_revision != self.source_revision:
            return False
        if anchor is not None:
            try:
                candidate = normalize_anchor(anchor, workspace=self.workspace, region=self._target_region)
            except CloseLookError:
                return False
            if candidate.stable_key != receipt.anchor_key:
                return False
        return bool(receipt.dossier_digest == (self.dossier.digest if self.dossier else ""))

    def pivot_gate(
        self,
        requested_tier: int,
        *,
        turn: int | None = None,
        source_revision: str | None = None,
        anchor: Anchor | Mapping[str, object] | str | None = None,
    ) -> PivotDecision:
        """Gate T2/T3 materialization behind a current close receipt."""

        if isinstance(requested_tier, bool) or requested_tier not in {0, 1, 2, 3}:
            raise CloseLookError("requested_tier must be 0, 1, 2, or 3")
        if requested_tier <= 1:
            if self.state in {FocusState.OPEN, FocusState.ANSWERED, FocusState.EDITED}:
                candidate = self._target_anchor
                if anchor is not None:
                    try:
                        candidate = normalize_anchor(
                            anchor,
                            workspace=self.workspace,
                            region=self._target_region,
                        )
                    except CloseLookError:
                        candidate = None
                inside_active_focus = bool(
                    candidate
                    and self._target_region
                    and candidate.region.intersects(self._target_region)
                    and (
                        self.state in {FocusState.ANSWERED, FocusState.EDITED}
                        and self._same_scope(candidate, self._target_anchor)
                    )
                )
                if inside_active_focus:
                    self._record("CLOSELOOK_PIVOT_ALLOWED", f"tier={requested_tier} inside_focus", turn)
                    return PivotDecision(True, requested_tier, self.state, "pivot_t0_t1_inside_focus")
                message = "[closelook] answer and complete the active focus before a T0/T1 pivot"
                if not self._pivot_challenge_emitted:
                    self._pivot_challenge_emitted = True
                    self._record("CLOSELOOK_CHALLENGE_ONCE", "pivot requested while focus open", turn)
                return PivotDecision(False, requested_tier, self.state, "pivot_held_for_focus", message)
            self._record("CLOSELOOK_PIVOT_ALLOWED", f"tier={requested_tier}", turn)
            return PivotDecision(True, requested_tier, self.state, "pivot_t0_t1_allowed")
        if self.valid_close_receipt(source_revision=source_revision, anchor=anchor):
            self._record("CLOSELOOK_PIVOT_ALLOWED", f"tier={requested_tier} close_receipt=current", turn)
            return PivotDecision(True, requested_tier, self.state, "pivot_close_receipt_valid")
        if self.state is FocusState.ABORTED:
            self.state = FocusState.IDLE
            self._focus = None
            self._answer = None
            self._close_receipt = None
            self._action_count = 0
        focus = None
        if self.state is FocusState.IDLE:
            opened = self._open(trigger="before_t2_pivot", turn=turn, anchor=self._target_anchor)
            focus = opened.focus
            message = "[closelook] FOCUS_REQUIRED_FOR_PIVOT; T2+ pivot is held until this focus closes"
            if opened.opened:
                self._pivot_challenge_emitted = True
                self._record("FOCUS_REQUIRED_FOR_PIVOT", f"tier={requested_tier}", turn)
                return PivotDecision(False, requested_tier, self.state, "FOCUS_REQUIRED_FOR_PIVOT", message, focus, opened.would_fire)
        if not self._pivot_challenge_emitted:
            self._pivot_challenge_emitted = True
            self._record("FOCUS_REQUIRED_FOR_PIVOT", f"tier={requested_tier}", turn)
            message = "[closelook] FOCUS_REQUIRED_FOR_PIVOT; close-look receipt is missing or stale"
        else:
            message = ""
        return PivotDecision(False, requested_tier, self.state, "FOCUS_REQUIRED_FOR_PIVOT", message, focus)

    def focus_row(self) -> FocusRow | None:
        if self._focus is None:
            return None
        return FocusRow(
            state=self._focus.state,
            region=self._focus.region.label,
            trigger=self._focus.trigger,
            opened_turn=self._focus.opened_turn,
            required_response="walkthrough -> one edit -> one test",
            dossier_digest=self._focus.dossier_digest,
            remaining_actions=self.remaining_actions,
            outside_challenge_used=self._outside_challenge_used,
        )

    def situation_card_hook(self, card: Mapping[str, object]) -> dict[str, object]:
        """Return a copy of an existing SituationCard payload with optional focus."""

        projected = {str(key): value for key, value in card.items()}
        row = self.focus_row()
        if row is None:
            projected.pop("focus", None)
        else:
            projected["focus"] = row.to_dict()
        return projected


def _close_look_trigger(evidence: Evidence) -> TriState:
    """Census predicate for admissible close-look candidate receipts."""

    record = evidence.event
    if evidence.source == "trace":
        event = str(record.get("event", "") or "").strip().lower()
        if not event:
            return TriState.UNKNOWN
        return TriState.YES if event in {"closelook_opened", "closelook_would_fire", "focus_required_for_pivot"} else TriState.NO
    if evidence.source != "receipt":
        return TriState.NO
    kind = str(record.get("kind", "") or "").strip().lower()
    result = str(record.get("result_class", record.get("result", "")) or "").strip().upper()
    if kind == "test":
        if result in _VALID_RED_RESULTS and record.get("anchor", record.get("primary_anchor")) not in (None, ""):
            return TriState.YES
        if result in _VALID_TEST_RESULTS:
            return TriState.UNKNOWN
        return TriState.NO if result else TriState.UNKNOWN
    if kind == "edit":
        if result == "APPLIED":
            return TriState.YES
        return TriState.NO if result else TriState.UNKNOWN
    if kind == "pivot":
        tier = _optional_int(record.get("requested_tier", record.get("tier")))
        if tier is None:
            return TriState.UNKNOWN
        return TriState.YES if tier >= 2 else TriState.NO
    return TriState.NO


CENSUS_SPEC = OrganCensusSpec(
    organ_id="closelook",
    trigger=_close_look_trigger,
    fire_events=frozenset(
        {
            "CLOSELOOK_OPENED",
            "CLOSELOOK_WOULD_FIRE",
            "close_look_opened",
            "close_look_would_fire",
        }
    ),
    suppression_events=frozenset({"CLOSELOOK_SUPPRESSED", "close_look_suppressed"}),
    dedup_events=frozenset({"CLOSELOOK_DEDUPED", "CLOSELOOK_RECEIPT_DEDUPED", "close_look_deduped"}),
    challenge_once_events=frozenset(
        {
            "CLOSELOOK_CHALLENGE_ONCE",
            "CLOSELOOK_OUTSIDE_JUSTIFIED",
            "FOCUS_REQUIRED_FOR_PIVOT",
            "close_look_challenge_once",
        }
    ),
)
CENSUS_SPECS = (CENSUS_SPEC,)
close_look_trigger = _close_look_trigger


def emit_via_organ_out(bolt: object, line: str | None) -> str | None:
    """Route a rendered dossier through KernelBolt's shadow chokepoint."""

    if not line:
        return None
    organ_out = getattr(bolt, "organ_out", None)
    if not callable(organ_out):
        raise TypeError("bolt must expose callable organ_out(name, line)")
    return organ_out("closelook", line)


__all__ = [
    "Anchor",
    "AnswerState",
    "AnswerValidation",
    "BranchCondition",
    "CENSUS_SPEC",
    "CENSUS_SPECS",
    "CHALLENGE_ONCE_LAW_ID",
    "CloseLookError",
    "CloseLookEvent",
    "CloseLookKernel",
    "CloseLookPolicy",
    "CloseReceipt",
    "ContractAtom",
    "DEFAULT_POLICY",
    "Dossier",
    "E394_LAW_ID",
    "E398_LAW_ID",
    "EffectStep",
    "EditDecision",
    "FirstDivergence",
    "FocusAnswer",
    "FocusCommand",
    "FocusCommandError",
    "FocusDecision",
    "FocusReceipt",
    "FocusRow",
    "FocusState",
    "GraphEdge",
    "IN_REGION_EDIT_THRESHOLD",
    "KernelDecision",
    "MAX_CALLERS",
    "MAX_CALLEES",
    "MAX_DOSSIER_TOKENS",
    "MAX_OPEN_ACTIONS",
    "MAX_SOURCE_LINES",
    "MAX_WITNESS_VECTORS",
    "ProposedEdit",
    "PivotDecision",
    "Region",
    "SNITCH_LAW_ID",
    "SAME_ANCHOR_RED_THRESHOLD",
    "SourceLine",
    "ValueFlowStep",
    "WalkthroughItem",
    "build_dossier",
    "canonical_json",
    "close_look_trigger",
    "emit_via_organ_out",
    "normalize_anchor",
    "normalize_relative_path",
    "parse_focus_command",
    "validate_answer",
]

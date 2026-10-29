"""Opt-in settled-decision memory for StateFrame.

The cache is deliberately a small host-owned ledger.  A model may propose a
typed conclusion on an ordinary action, but it cannot assign ids, mark work
complete, or manufacture evidence.  The ledger keeps semantic decision
versions separate from the existing work reducer and derives dependency
fingerprints from host source/receipt state.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from kernel_measurement import profiled_function_span


MODE_ENV = "STATEFRAME_DECISION_CACHE"
SCHEMA = "stateframe-decision-cache/1"
MEMORY_OPS = ("ADOPT", "CHALLENGE", "SUPERSEDE")
ANSWER_TYPES = (
    "CALL_REQUIREMENT",
    "RESPONSIBILITY",
    "COMPAT_ALIAS",
    "CLAUSE_INTERPRETATION",
)
DECISION_QUESTIONS = (
    "REQUIREMENT_MEANING",
    "APPARENT_CONFLICT",
    "BEHAVIOR_OWNER",
    "COMPATIBILITY_STRATEGY",
)
DECISION_LIFECYCLES = ("PROPOSED", "ADOPTED", "DISPUTED", "SUPERSEDED", "RETIRED")
BASIS_BINDINGS = ("CURRENT", "STALE", "UNKNOWN")
RESOLUTION_KINDS = (
    "JOINTLY_SATISFIABLE",
    "DIFFERENT_SCOPES",
    "EXPLICIT_SUPERSESSION",
)
CHALLENGE_REASONS = ("COUNTEREVIDENCE", "LOGIC_ERROR", "MISSING_ASSUMPTION")
# The decision tape is lossless.  Prompt pressure is handled by the relevance
# view below, never by rejecting or evicting an older decision.
MAX_RECORDS: int | None = None
MAX_SIDECAR_BYTES = 16_000
MAX_STRING = 800
MAX_REFS = 16
MAX_DEPENDENCIES = 32
MAX_RENDER_TOKENS = 300
MAX_RENDER_RECORDS = 6
MAX_RENDER_REJECTIONS = 2
MAX_RENDER_RECEIPTS = 4
_ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")


class DecisionCacheError(ValueError):
    """A malformed proposal or an unsafe host-side cache mutation."""


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").lower()
    if value in {"0", "off"}:
        return "off"
    if value in {"1", "on"}:
        return "on"
    raise ValueError(f"{MODE_ENV} must be 0/1 or off/on")


def enabled() -> bool:
    return mode() == "on"


def _ext() -> bool:
    """r44 conclusion-cache extension (own default-off switch on top of r43)."""
    from kernel_conclusion_cache import enabled as conclusion_cache_enabled
    return conclusion_cache_enabled()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _plain_receipt_value(value: object) -> object:
    """Convert immutable receipt mappings into canonical JSON containers."""
    if isinstance(value, Mapping):
        return {str(key): _plain_receipt_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_receipt_value(item) for item in value]
    return value


def _basis_revision(context: Mapping[str, object]) -> str:
    """Fingerprint the host view used when a model interpretation was adopted."""
    source_files = context.get("source_files")
    if not isinstance(source_files, Mapping):
        source_files = {}
    receipts = context.get("receipts")
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes, bytearray)):
        receipts = []
    basis = {
        "brief": context.get("brief", ""),
        "source_files": {str(path): source_files[path] for path in sorted(source_files, key=str)},
        "receipts": list(receipts),
    }
    for key in ("test_definition", "environment"):
        if key in context:
            basis[key] = context.get(key)
    return _digest(basis)


def _text(value: object, name: str, limit: int = MAX_STRING, *, required: bool = True) -> str:
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        suffix = " and nonempty" if required else ""
        raise DecisionCacheError(f"{name} must be a string of at most {limit} characters{suffix}")
    return value


def _id(value: object, name: str) -> str:
    result = _text(value, name, 64)
    if not _ID_RE.fullmatch(result):
        raise DecisionCacheError(f"{name} has an invalid id")
    return result


def _json_value(value: object, *, depth: int = 0) -> object:
    if depth > 4:
        raise DecisionCacheError("sidecar value nesting exceeds four levels")
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
            raise DecisionCacheError("sidecar numeric values must be finite")
        if isinstance(value, str) and len(value) > MAX_STRING:
            raise DecisionCacheError(f"sidecar strings must be at most {MAX_STRING} characters")
        return copy.deepcopy(value)
    if isinstance(value, list):
        if len(value) > 32:
            raise DecisionCacheError("sidecar arrays may contain at most 32 values")
        return [_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, Mapping):
        if len(value) > 32:
            raise DecisionCacheError("sidecar objects may contain at most 32 keys")
        return {_text(key, "sidecar key", 64): _json_value(item, depth=depth + 1)
                for key, item in value.items()}
    raise DecisionCacheError("sidecar values must be JSON-compatible")


def _refs(value: object, name: str = "basis_refs") -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_REFS:
        raise DecisionCacheError(f"{name} must be a list of at most {MAX_REFS} references")
    result = []
    for item in value:
        ref = _text(item, name + " item", 180)
        if ref in result:
            raise DecisionCacheError(f"{name} contains duplicate references")
        result.append(ref)
    return result


def _string_list(value: object, name: str, *, limit: int = MAX_REFS) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > limit:
        raise DecisionCacheError(f"{name} must be a list of at most {limit} strings")
    return [_text(item, name + " item", 400) for item in value]


def _key(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise DecisionCacheError("draft.key must be an object")
    if set(value) - {"subject", "question"}:
        raise DecisionCacheError("draft.key contains an unknown field")
    subject = _text(value.get("subject"), "draft.key.subject", 240)
    question = _text(value.get("question"), "draft.key.question", 64).upper()
    if question not in DECISION_QUESTIONS:
        raise DecisionCacheError("draft.key.question is not a supported decision question")
    return {"subject": subject, "question": question}


def _answer(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DecisionCacheError("draft.selected must be an object")
    if "completed" in value:
        raise DecisionCacheError("the model cannot write completed decision metadata")
    answer_type = _text(value.get("type", value.get("answer_type")),
                        "draft.selected.type", 64).upper()
    if answer_type not in ANSWER_TYPES:
        raise DecisionCacheError("draft.selected.type is not a supported answer type")
    result: dict[str, object] = {"type": answer_type}
    for key, item in value.items():
        if key == "type":
            continue
        result[_text(key, "answer field", 64)] = _json_value(item)

    if answer_type == "CALL_REQUIREMENT":
        shapes = (value.get("call_shapes") if "call_shapes" in value else
                  value.get("must_accept") if "must_accept" in value else
                  value.get("must_accept_call_shapes"))
        if shapes is not None:
            result["call_shapes"] = _string_list(shapes, "call_shapes", limit=16)
        defaults = value.get("defaults")
        if defaults is not None:
            if not isinstance(defaults, Mapping):
                raise DecisionCacheError("CALL_REQUIREMENT defaults must be an object")
            result["defaults"] = _json_value(defaults)
    elif answer_type == "CLAUSE_INTERPRETATION":
        proposition = value.get("proposition", value.get("bounded_proposition"))
        if proposition is not None:
            result["proposition"] = _text(proposition, "proposition", 600)
    return result


def _resolution(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DecisionCacheError("draft.resolution must be an object")
    result: dict[str, object] = {}
    if "kind" in value:
        kind = _text(value["kind"], "resolution.kind", 64).upper()
        if kind not in RESOLUTION_KINDS:
            raise DecisionCacheError("resolution.kind is not supported")
        result["kind"] = kind
    if "inputs" in value:
        result["inputs"] = _string_list(value["inputs"], "resolution.inputs")
    if "witness" in value:
        result["witness"] = _text(value["witness"], "resolution.witness", 400)
    for key in set(value) - {"kind", "inputs", "witness"}:
        if key == "completed":
            raise DecisionCacheError("resolution cannot contain completed metadata")
        result[_text(key, "resolution field", 64)] = _json_value(value[key])
    return result


def _rejected(value: object) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 16:
        raise DecisionCacheError("draft.rejected must contain at most 16 alternatives")
    return [_json_value(item) for item in value]


@dataclass(frozen=True)
class DecisionDraft:
    """The model-writable semantic portion of a decision record."""

    key: dict[str, str]
    scope: object
    selected: dict[str, object]
    basis_refs: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    reopen_on: tuple[str, ...] = ()
    resolution: dict[str, object] | None = None
    rejected: tuple[object, ...] = ()

    @classmethod
    def from_mapping(cls, value: object) -> "DecisionDraft":
        if not isinstance(value, Mapping):
            raise DecisionCacheError("draft must be an object")
        if "completed" in value:
            raise DecisionCacheError("the model cannot write completed decision metadata")
        allowed = {"key", "scope", "selected", "basis_refs", "assumptions",
                   "resolution", "rejected"}
        if _ext():
            allowed.add("reopen_on")
        unknown = set(value) - allowed
        if unknown:
            raise DecisionCacheError("unknown draft field: " + sorted(map(str, unknown))[0])
        scope_value = value.get("scope", "")
        # ``scope`` is useful context, but it is not needed for the smallest
        # useful ADOPT proposal.  Treat an omitted/empty scope as the host
        # default instead of rejecting an otherwise complete decision.
        if scope_value is None:
            scope = ""
        elif isinstance(scope_value, str):
            scope = _text(scope_value, "draft.scope", 400, required=False)
        else:
            scope = _json_value(scope_value)
        return cls(
            key=_key(value.get("key")),
            scope=scope,
            selected=_answer(value.get("selected")),
            basis_refs=tuple(_refs(value.get("basis_refs"))),
            assumptions=tuple(_string_list(value.get("assumptions"), "assumptions")),
            reopen_on=tuple(_string_list(value.get("reopen_on"), "reopen_on", limit=8)),
            resolution=_resolution(value.get("resolution")),
            rejected=tuple(_rejected(value.get("rejected"))),
        )

    def to_dict(self) -> dict[str, object]:
        result = {
            "key": copy.deepcopy(self.key),
            "scope": copy.deepcopy(self.scope),
            "selected": copy.deepcopy(self.selected),
            "basis_refs": list(self.basis_refs),
            "assumptions": list(self.assumptions),
            "resolution": copy.deepcopy(self.resolution or {}),
            "rejected": copy.deepcopy(list(self.rejected)),
        }
        if _ext():
            result = {key: result[key] for key in ("key", "scope", "selected",
                                                  "basis_refs", "assumptions")}
            result.update({
                "reopen_on": list(self.reopen_on),
                "resolution": copy.deepcopy(self.resolution or {}),
                "rejected": copy.deepcopy(list(self.rejected)),
            })
        return result


def _normalize_sidecar(value: object) -> tuple[str, DecisionDraft | None, dict[str, object]]:
    if not isinstance(value, Mapping):
        raise DecisionCacheError("memory sidecar must be an object")
    if "memory" in value and len(value) == 1:
        value = value["memory"]
        if not isinstance(value, Mapping):
            raise DecisionCacheError("memory sidecar must be an object")
    if "completed" in value:
        raise DecisionCacheError("memory sidecar cannot write completed metadata")
    raw_op = value.get("op", value.get("operation"))
    op = _text(raw_op, "memory.op", 32).upper()
    if op not in MEMORY_OPS:
        raise DecisionCacheError("memory.op must be ADOPT, CHALLENGE, or SUPERSEDE")
    if op == "CHALLENGE":
        allowed = {"op", "operation", "decision_id", "target_id", "target", "reason",
                   "basis_refs", "counterevidence_refs"}
        if set(value) - allowed:
            raise DecisionCacheError("unknown CHALLENGE sidecar field")
        target = value.get("decision_id", value.get("target_id", value.get("target")))
        target = _id(target, "challenge decision_id")
        reason = _text(value.get("reason"), "challenge reason", 64).upper()
        if reason not in CHALLENGE_REASONS:
            raise DecisionCacheError("challenge reason is not supported")
        refs = _refs(value.get("basis_refs", value.get("counterevidence_refs")))
        return op, None, {"decision_id": target, "reason": reason, "basis_refs": refs}

    allowed = {"op", "operation", "draft", "supersedes", "decision_id", "target"}
    if set(value) - allowed:
        raise DecisionCacheError("unknown memory sidecar field")
    draft = DecisionDraft.from_mapping(value.get("draft"))
    target = value.get("supersedes", value.get("decision_id", value.get("target")))
    if op == "SUPERSEDE":
        target = _id(target, "supersedes")
        return op, draft, {"supersedes": target}
    if target is not None:
        raise DecisionCacheError("ADOPT cannot set supersedes")
    return op, draft, {}


def _latest(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        ident = raw.get("id")
        if not isinstance(ident, str):
            continue
        version = raw.get("version", 0)
        if type(version) is not int:
            continue
        previous = result.get(ident)
        if previous is None or version >= int(previous.get("version", -1)):
            result[ident] = copy.deepcopy(dict(raw))
    return result


def _symbol_name(subject: str, context: Mapping[str, object]) -> str:
    symbol = context.get("symbol")
    if isinstance(symbol, str) and symbol.strip():
        return symbol.split("::")[-1].split(".")[-1]
    return subject.split("::")[-1].split(".")[-1]


def _target_path(subject: str, context: Mapping[str, object]) -> str:
    path = context.get("target_path")
    if isinstance(path, str) and path.strip():
        return path.replace("\\", "/").lstrip("./")
    if "::" in subject:
        return subject.split("::", 1)[0].replace("\\", "/").lstrip("./")
    return ""


def _find_definition(source: str, symbol: str) -> ast.AST | None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return None
    candidates = [node for node in ast.walk(tree)
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  and node.name == symbol]
    return max(candidates, key=lambda node: getattr(node, "end_lineno", node.lineno), default=None)


def _source_slice(source: str, start: int, end: int) -> str:
    lines = source.splitlines(keepends=True)
    if not lines:
        return ""
    return "".join(lines[max(0, start - 1):min(len(lines), end)])


def _dependency_rows(draft: DecisionDraft, context: Mapping[str, object]) -> list[dict[str, object]]:
    selected_type = str(draft.selected.get("type", ""))
    question = draft.key["question"]
    path = _target_path(draft.key["subject"], context)
    symbol = _symbol_name(draft.key["subject"], context)
    target = f"{path}::{symbol}" if path else symbol
    rows: list[dict[str, object]] = []

    def add(facet: str, target_name: str, value: object, *, unknown: bool = False) -> None:
        rows.append({"target": target_name, "facet": facet,
                     "digest": "UNKNOWN" if unknown else _digest(value)})

    source_files = context.get("source_files")
    if not isinstance(source_files, Mapping):
        source_files = {}
    source = source_files.get(path)
    definition = _find_definition(source, symbol) if isinstance(source, str) else None
    structural = selected_type in {"CALL_REQUIREMENT", "COMPAT_ALIAS", "RESPONSIBILITY"}
    needs_body = selected_type in {"CLAUSE_INTERPRETATION", "RESPONSIBILITY"} or not structural
    extension = _ext()
    interpretation = extension and (selected_type == "CLAUSE_INTERPRETATION" or question in {
        "REQUIREMENT_MEANING", "APPARENT_CONFLICT"})
    if needs_body:
        if definition is None or not isinstance(source, str):
            add("body", target, None, unknown=True)
        else:
            body_start = min((getattr(item, "lineno", definition.lineno)
                              for item in getattr(definition, "body", ())),
                             default=definition.lineno)
            body_end = max((getattr(item, "end_lineno", body_start)
                            for item in getattr(definition, "body", ())), default=body_start)
            body = _source_slice(source, body_start, body_end)
            add("body", target, {"text": body,
                                 "ast": [ast.dump(item, include_attributes=False)
                                         for item in getattr(definition, "body", ())]})
    if interpretation and not structural:
        # r44: an interpretation of a clause also rests on the definition's
        # signature (a changed parameter list changes what "optional" means).
        if definition is None or not isinstance(source, str):
            add("signature", target, None, unknown=True)
        else:
            add("signature", target, ast.dump(getattr(definition, "args", definition),
                                               include_attributes=False))
    if interpretation:
        # Host-known premises; absent from the context means unresolved, which
        # conservatively binds the record UNKNOWN instead of CURRENT.
        for facet in ("test_definition", "environment"):
            value = context.get(facet)
            add(facet, "task", value, unknown=value is None)
    if structural:
        if definition is None or not isinstance(source, str):
            add("signature", target, None, unknown=True)
        else:
            add("signature", target, ast.dump(getattr(definition, "args", definition),
                                               include_attributes=False))

        caller_rows: list[dict[str, object]] = []
        caller_unknown = False
        for raw_path, raw_source in source_files.items():
            if not isinstance(raw_path, str) or not isinstance(raw_source, str):
                caller_unknown = True
                continue
            try:
                tree = ast.parse(raw_source)
            except (SyntaxError, ValueError, TypeError):
                caller_unknown = True
                continue
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                called = (call.func.id if isinstance(call.func, ast.Name)
                          else call.func.attr if isinstance(call.func, ast.Attribute) else None)
                if called != symbol:
                    continue
                caller_rows.append({
                    "path": raw_path.replace("\\", "/"),
                    "line": getattr(call, "lineno", 0),
                    "text": ast.get_source_segment(raw_source, call) or "",
                })
        add("callers", target, sorted(caller_rows, key=lambda row: (
            row["path"], row["line"], row["text"])), unknown=caller_unknown)

    # Requirement wording is an independent basis facet.  It is intentionally
    # host-derived from the brief, not a model-supplied dependency claim.
    if question in {"REQUIREMENT_MEANING", "APPARENT_CONFLICT", "COMPATIBILITY_STRATEGY"}:
        brief = context.get("brief")
        add("ticket", "task", brief if isinstance(brief, str) and brief else None,
            unknown=not isinstance(brief, str) or not brief)

    receipts = context.get("receipts")
    receipt_map = {}
    if isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes, bytearray)):
        for receipt in receipts:
            if isinstance(receipt, Mapping) and isinstance(receipt.get("receipt_id"), str):
                receipt_map[receipt["receipt_id"]] = receipt
    for ref in draft.basis_refs:
        add("evidence", ref, receipt_map.get(ref, {"receipt_id": ref}))
    return rows[:MAX_DEPENDENCIES]


def _binding(old: Sequence[Mapping[str, object]], new: Sequence[Mapping[str, object]]) -> str:
    old_map = {(row.get("target"), row.get("facet")): row.get("digest") for row in old}
    new_map = {(row.get("target"), row.get("facet")): row.get("digest") for row in new}
    if not old_map or not new_map:
        return "UNKNOWN"
    if any(value == "UNKNOWN" for value in (*old_map.values(), *new_map.values())):
        return "UNKNOWN"
    if set(old_map) != set(new_map):
        return "UNKNOWN"
    return "CURRENT" if all(old_map[key] == new_map[key] for key in old_map) else "STALE"


def _changed_facets(old: Sequence[Mapping[str, object]], new: Sequence[Mapping[str, object]]) -> list[str]:
    old_map = {(row.get("target"), row.get("facet")): row.get("digest") for row in old}
    new_map = {(row.get("target"), row.get("facet")): row.get("digest") for row in new}
    return sorted({str(key[1]) for key in set(old_map) | set(new_map)
                   if old_map.get(key) != new_map.get(key)})


def _contradiction_ids(receipt: Mapping[str, object] | None) -> set[str]:
    """Read only explicit host contradiction links; a failing test is not enough."""
    if not isinstance(receipt, Mapping):
        return set()
    values: list[object] = []
    for key in ("contradicts", "contradiction_for", "counterexample_for"):
        value = receipt.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            values.extend(value)
    return {str(value) for value in values if isinstance(value, str) and value.strip()}


def _record_id_number(ident: str) -> int:
    match = re.fullmatch(r"decision-(\d+)", ident)
    return int(match.group(1)) if match else -1


def _draft_from_record(row: Mapping[str, object]) -> DecisionDraft:
    """Strip host metadata before revalidating a stored semantic draft."""
    keys = ["key", "scope", "selected", "basis_refs", "assumptions", "resolution", "rejected"]
    if _ext():
        keys.insert(5, "reopen_on")
    return DecisionDraft.from_mapping({key: copy.deepcopy(row.get(key)) for key in keys})


class DecisionCacheStore:
    """Append-only semantic decisions with host-owned lifecycle metadata."""

    def __init__(self, *, state: Mapping[str, object] | None = None,
                 emit: Callable[..., object] | None = None,
                 source_files: Mapping[str, object] | None = None,
                 target_path: str = "", symbol: str = "", brief: str = "",
                 receipts: Sequence[object] = (),
                 **premises: object):
        unknown = set(premises) - {"test_definition", "environment"}
        if unknown:
            raise TypeError("unexpected decision-cache context field: " + sorted(unknown)[0])
        self._emit_callback = emit
        self._records: list[dict[str, object]] = []
        self._next_id = 0
        self._context: dict[str, object] = {
            "source_files": copy.deepcopy(dict(source_files or {})),
            "target_path": target_path,
            "symbol": symbol,
            "brief": brief,
            "receipts": copy.deepcopy(list(receipts)),
        }
        # r44 host premises (only supplied when the conclusion-cache switch is on).
        self._context.update(copy.deepcopy(premises))
        self._input_receipt_refs: set[str] = set()
        self._last_delta: list[dict[str, object]] = []
        self._record_generation = 0
        self._active_records_cache: list[dict[str, object]] | None = None
        self._selected_rows_cache: dict[tuple[object, ...], list[dict[str, object]]] = {}
        self._record_text_cache: dict[tuple[object, ...], str] = {}
        self._receipt_lines_cache: dict[tuple[object, ...], list[str]] = {}
        self.metrics = {"accepted": 0, "rejected": 0, "rendered": 0,
                        "rendered_tokens": 0, "dropped_records": 0}
        if state is not None:
            self._restore(state)
        else:
            self._advance_counter()

    @classmethod
    def from_state(cls, state: Mapping[str, object], *, emit=None,
                   context: Mapping[str, object] | None = None,
                   **context_fields) -> "DecisionCacheStore":
        if context is not None:
            context_fields = {**dict(context), **context_fields}
        return cls(state=state, emit=emit, **context_fields)

    def _restore(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != SCHEMA:
            raise DecisionCacheError("unsupported decision-cache state")
        rows = state.get("records")
        if not isinstance(rows, list):
            raise DecisionCacheError("decision-cache records must be a list")
        restored = []
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise DecisionCacheError("decision-cache records must be objects")
            row = copy.deepcopy(dict(raw))
            ident = _id(row.get("id"), "decision id")
            if type(row.get("version")) is not int or row["version"] < 1:
                raise DecisionCacheError("decision version must be a positive integer")
            if row.get("lifecycle") not in DECISION_LIFECYCLES:
                raise DecisionCacheError("unknown decision lifecycle")
            if row.get("basis_binding") not in BASIS_BINDINGS:
                raise DecisionCacheError("unknown decision basis binding")
            if not isinstance(row.get("basis_dependencies"), list):
                raise DecisionCacheError("basis_dependencies must be a list")
            row["id"] = ident
            row.setdefault("redecision_required", False)
            restored.append(row)
        self._records = restored
        self._record_changed()
        metrics = state.get("metrics")
        if isinstance(metrics, Mapping):
            for key in self.metrics:
                if type(metrics.get(key)) is int and metrics[key] >= 0:
                    self.metrics[key] = metrics[key]
        self._advance_counter()

    def _advance_counter(self) -> None:
        self._next_id = max((_record_id_number(str(row.get("id")))
                             for row in self._records), default=-1) + 1

    def _emit(self, kind: str, **fields: object) -> None:
        if self._emit_callback is not None:
            self._emit_callback(kind, producer="stateframe_decision_cache", **fields)

    def snapshot(self) -> dict[str, object]:
        return {"schema": SCHEMA, "records": copy.deepcopy(self._records),
                "metrics": copy.deepcopy(self.metrics)}

    def all_records(self) -> list[dict[str, object]]:
        return copy.deepcopy(self._records)

    def active_records(self) -> list[dict[str, object]]:
        return copy.deepcopy(self._active_records())

    def _active_records(self) -> list[dict[str, object]]:
        if self._active_records_cache is None:
            latest = _latest(self._records)
            self._active_records_cache = [copy.deepcopy(row) for row in self._records
                                          if latest.get(row.get("id"), {}).get("version")
                                          == row.get("version")
                                          and row.get("lifecycle") not in {"SUPERSEDED", "RETIRED"}]
        return self._active_records_cache

    def _record_changed(self) -> None:
        self._record_generation += 1
        self._active_records_cache = None
        self._selected_rows_cache.clear()
        self._record_text_cache.clear()

    def begin_frame(self, frame: Mapping[str, object], *, context: Mapping[str, object] | None = None) -> None:
        """Record exactly what receipt references were visible in the input frame."""
        self._input_receipt_refs = collect_receipt_refs(frame)
        self._last_delta = []
        if context is not None:
            self.refresh(context)

    def _new_record(self, draft: DecisionDraft, context: Mapping[str, object], ident: str,
                    *, version: int = 1, lifecycle: str = "ADOPTED") -> dict[str, object]:
        dependencies = _dependency_rows(draft, context)
        if not _ext():
            return {
                "id": ident,
                "version": version,
                "author": "host_ingestion",
                "lifecycle": lifecycle,
                "basis_binding": _binding(dependencies, dependencies),
                "basis_dependencies": dependencies,
                "redecision_required": False,
                **draft.to_dict(),
            }
        basis_revision = _basis_revision(context)
        return {
            "id": ident,
            "version": version,
            "author": "host_ingestion",
            "lifecycle": lifecycle,
            "basis_binding": _binding(dependencies, dependencies),
            "basis_dependencies": dependencies,
            "basis_changed_facets": [],
            "redecision_required": False,
            "provenance": {
                "kind": "MODEL_INTERPRETATION",
                "author": "model",
                "ingested_by": "kernel",
                "basis_revision": basis_revision,
            },
            **draft.to_dict(),
        }

    @staticmethod
    def _input_refs(value: object) -> set[str]:
        if isinstance(value, Mapping):
            return collect_receipt_refs(value)
        if isinstance(value, (set, frozenset)):
            return {str(item) for item in value if isinstance(item, str)}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return {str(item) for item in value if isinstance(item, str)}
        return set()

    def ingest(self, sidecar: object, *, input_receipt_refs: object = None,
               context: Mapping[str, object] | None = None, step: int | None = None) -> dict[str, object]:
        """Validate and install one optional model proposal without touching its action."""
        started = time.perf_counter_ns()
        try:
            if isinstance(sidecar, (dict, list)) and len(_canonical(sidecar).encode("utf-8")) > MAX_SIDECAR_BYTES:
                raise DecisionCacheError("memory sidecar exceeds its byte budget")
            op, draft, extra = _normalize_sidecar(sidecar)
            refs = (self._input_refs(input_receipt_refs)
                    if input_receipt_refs is not None else set(self._input_receipt_refs))
            proposed_refs = (draft.basis_refs if draft is not None else tuple(extra.get("basis_refs", ())))
            missing = [ref for ref in proposed_refs if ref not in refs]
            if missing:
                raise DecisionCacheError("basis reference was not present in the input frame: " + missing[0])
            active = _latest(self._records)
            use_context = dict(self._context)
            if context is not None:
                use_context.update(copy.deepcopy(dict(context)))
            self._context = use_context
            if op == "CHALLENGE":
                target_id = str(extra["decision_id"])
                target = active.get(target_id)
                if target is None or target.get("lifecycle") in {"SUPERSEDED", "RETIRED"}:
                    raise DecisionCacheError("challenge target is not active")
                old_version = int(target.get("version", 1))
                challenged = copy.deepcopy(target)
                challenged.update({
                    "version": old_version + 1,
                    "author": "host_ingestion",
                    "lifecycle": "DISPUTED",
                    "challenge_reason": extra["reason"],
                })
                if extra.get("basis_refs"):
                    challenged["basis_refs"] = list(extra["basis_refs"])
                challenged["basis_dependencies"] = _dependency_rows(
                    _draft_from_record(challenged), use_context)
                challenged["basis_binding"] = _binding(
                    target.get("basis_dependencies", ()), challenged["basis_dependencies"])
                self._records.append(challenged)
                record = challenged
                self._last_delta.append({"id": target_id, "change": "DISPUTED",
                                         "from": target.get("lifecycle"), "reason": extra["reason"]})
            else:
                assert draft is not None
                if op == "SUPERSEDE":
                    target_id = str(extra["supersedes"])
                    target = active.get(target_id)
                    if target is None or target.get("lifecycle") in {"SUPERSEDED", "RETIRED"}:
                        raise DecisionCacheError("supersession target is not active")
                    superseded = copy.deepcopy(target)
                    superseded.update({
                        "version": int(target.get("version", 1)) + 1,
                        "author": "host_ingestion",
                        "lifecycle": "SUPERSEDED",
                    })
                    new_id = f"decision-{self._next_id}"
                    self._next_id += 1
                    record = self._new_record(draft, use_context, new_id)
                    record["supersedes"] = target_id
                    self._records.extend((superseded, record))
                    self._last_delta.extend((
                        {"id": target_id, "change": "SUPERSEDED", "from": target.get("lifecycle")},
                        {"id": new_id, "change": "ADOPTED", "supersedes": target_id},
                    ))
                else:
                    ident = f"decision-{self._next_id}"
                    self._next_id += 1
                    record = self._new_record(draft, use_context, ident)
                    self._records.append(record)
                    self._last_delta.append({"id": ident, "change": "ADOPTED"})
            self.metrics["accepted"] += 1
            self._record_changed()
            self._emit("decision_cache_op_accepted", op=op, decision_id=record["id"],
                       version=record["version"], step=step,
                       basis_refs=list(proposed_refs), elapsed_ns=time.perf_counter_ns() - started)
            return {"accepted": True, "op": op, "record": copy.deepcopy(record),
                    "state": self.snapshot()}
        except (DecisionCacheError, TypeError, ValueError, KeyError) as error:
            self.metrics["rejected"] += 1
            self._emit("decision_cache_op_rejected", op=(sidecar.get("op")
                       if isinstance(sidecar, Mapping) else None), reason=str(error), step=step,
                       elapsed_ns=time.perf_counter_ns() - started)
            return {"accepted": False, "reason": str(error), "state": self.snapshot()}

    def refresh(self, context: Mapping[str, object] | None = None,
                *, receipt: Mapping[str, object] | None = None) -> list[dict[str, object]]:
        started = time.perf_counter_ns()
        if context is not None:
            self._context.update(copy.deepcopy(dict(context)))
        changes: list[dict[str, object]] = []
        latest = _latest(self._records)
        extension = _ext()
        contradicted = _contradiction_ids(receipt) if extension else set()
        for row in self._records:
            if latest.get(row.get("id"), {}).get("version") != row.get("version"):
                continue
            if row.get("lifecycle") in {"SUPERSEDED", "RETIRED"}:
                continue
            try:
                draft = _draft_from_record(row)
                dependencies = _dependency_rows(draft, self._context)
            except (DecisionCacheError, TypeError, ValueError):
                dependencies = [{"target": "decision", "facet": "broad", "digest": "UNKNOWN"}]
            previous_binding = row.get("basis_binding")
            previous_dependencies = copy.deepcopy(row.get("basis_dependencies", []))
            new_binding = _binding(previous_dependencies, dependencies)
            row["basis_binding"] = new_binding
            facets = _changed_facets(previous_dependencies, dependencies)
            if extension:
                row["basis_changed_facets"] = list(facets)
            if previous_binding != new_binding:
                row["redecision_required"] = True
                change = {"id": row["id"], "change": "BASIS_REBOUND",
                          "from": previous_binding, "to": new_binding,
                          "facets": facets,
                          "enforcement": "REDECISION_REQUIRED"}
                changes.append(change)
            if row["id"] in contradicted and row.get("lifecycle") != "DISPUTED":
                previous_lifecycle = row.get("lifecycle")
                row["lifecycle"] = "DISPUTED"
                row["challenge_reason"] = "COUNTEREVIDENCE"
                changes.append({"id": row["id"], "change": "CONTRADICTED",
                                "from": previous_lifecycle, "reason": "COUNTEREVIDENCE"})
        if changes:
            self._last_delta.extend(copy.deepcopy(changes))
        if changes:
            self._record_changed()
        if context is not None or receipt is not None:
            self._emit("decision_cache_maintenance", receipt_id=(
                receipt.get("receipt_id") if isinstance(receipt, Mapping) else None),
                records_checked=len(latest), changed=len(changes),
                elapsed_ns=time.perf_counter_ns() - started,
                basis_bindings={row["id"]: row.get("basis_binding") for row in self.active_records()})
        return copy.deepcopy(changes)

    def action_gate(self, action: str = "done") -> dict[str, object] | None:
        """Reject gated actions until a rebound basis has a replacement decision."""
        gated_actions = {"done", "complete", "stage", "stage_transition", "finish"}
        if str(action).strip().lower() not in gated_actions:
            return None
        rebound = [
            row for row in self.active_records()
            if row.get("redecision_required") is True
        ]
        if not rebound:
            return None
        ids = [str(row.get("id", "decision")) for row in rebound]
        return {
            "admitted": False,
            "reason_code": "DECISION_REDECISION_REQUIRED",
            "finding": "decision basis rebound; a fresh decision is required before completion",
            "next_action": "re-decide against the current evidence basis",
            "decision_ids": ids,
        }

    def delta(self) -> list[dict[str, object]]:
        return copy.deepcopy(self._last_delta)

    def render(self, *, work_frontier: Sequence[object] = (), active_symbols: Sequence[object] = (),
               max_tokens: int = MAX_RENDER_TOKENS,
               receipts: Sequence[object] | None = None,
               current_revision: object = None,
               receipt_generation: object = None,
               receipt_cursor: object = None,
               emit: Callable[..., object] | None = None) -> str:
        return render_settled_state(self, work_frontier=work_frontier,
                                    active_symbols=active_symbols, max_tokens=max_tokens,
                                     receipts=receipts, current_revision=current_revision,
                                     receipt_generation=receipt_generation,
                                     receipt_cursor=receipt_cursor,
                                     emit=emit or self._emit)

    def _render_record(self, row: Mapping[str, object]) -> str:
        key = (self._record_generation, row.get("id"), row.get("version"))
        cached = self._record_text_cache.get(key)
        if cached is None:
            cached = _record_text(row)
            self._record_text_cache[key] = cached
        return cached

    def _render_record_r43(self, row: Mapping[str, object]) -> str:
        key = (self._record_generation, "r43", row.get("id"), row.get("version"))
        cached = self._record_text_cache.get(key)
        if cached is None:
            cached = _record_text_r43(row)
            self._record_text_cache[key] = cached
        return cached

    def _receipt_lines(self, receipts: Sequence[object] | None,
                       current_revision: object, receipt_generation: object = None,
                       receipt_cursor: object = None) -> list[str]:
        from kernel_conclusion_cache import _ReceiptRows, receipt_lines
        plain_revision = _plain_receipt_value(current_revision)
        try:
            revision_key = json.dumps(plain_revision, sort_keys=True,
                                      separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            revision_key = repr(plain_revision)
        cursor_key = receipt_cursor if type(receipt_cursor) is int else None

        if isinstance(receipts, _ReceiptRows):
            # Frozen runtime snapshots keep identity across unrelated journal
            # writes, so reuse rendered lines without rescanning old receipts.
            snapshot_token = receipts.snapshot_token
            sequence = receipts
            cache_key = ("runtime_snapshot", revision_key, snapshot_token)
            cached = self._receipt_lines_cache.get(cache_key)
            if isinstance(cached, dict):
                return list(cached.get("rows", ()))
            input_token: object = snapshot_token
        else:
            values = receipts or ()
            raw_sequence = values.values() if isinstance(values, Mapping) else values
            sequence = tuple(_plain_receipt_value(value) for value in raw_sequence)
            semantic_generation = receipt_generation
            if type(semantic_generation) is not int:
                semantic_generation = _digest(sequence)
            input_token = _digest(sequence)
            cache_key = ("stream", revision_key, semantic_generation, cursor_key, input_token)

        cached = self._receipt_lines_cache.get(cache_key)
        if isinstance(cached, dict):
            return list(cached.get("rows", ()))
        rendered = list(receipt_lines(sequence, plain_revision))
        if len(self._receipt_lines_cache) >= 8:
            self._receipt_lines_cache.clear()
        self._receipt_lines_cache[cache_key] = {
            "rows": rendered, "cursor": cursor_key, "input_token": input_token,
        }
        return rendered


def _token_counter(text: str) -> tuple[int, str]:
    """Use an installed tokenizer when explicitly usable; otherwise report chars/4."""
    model = os.environ.get("REACT_TOKENIZER_MODEL") or os.environ.get("REACT_MODEL")
    if model:
        try:
            tiktoken = importlib.import_module("tiktoken")
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text)), "tiktoken:" + model
        except (ImportError, KeyError, ValueError, RuntimeError):
            pass
    return (len(text) + 3) // 4, "proxy:chars/4"


def _compact(value: object) -> str:
    return _canonical(value)


def render_reusable_receipts(receipts: Sequence[object] | None, *,
                             current_revision: object = None,
                             max_rows: int = MAX_RENDER_RECEIPTS) -> str:
    """Render bounded host receipts as answers the model can reuse (r44 switch only)."""
    if not enabled() or not _ext() or not isinstance(max_rows, int) or max_rows < 1:
        return ""
    from kernel_conclusion_cache import receipt_lines
    lines = receipt_lines(receipts, current_revision)
    if not lines:
        return ""
    return "\n".join(["REUSABLE RECEIPTS", *lines[-max_rows:]])


def _record_text_r43(row: Mapping[str, object]) -> str:
    selected = row.get("selected") if isinstance(row.get("selected"), Mapping) else {}
    selected_type = selected.get("type", "")
    selected_fields: dict[str, object] = {"type": selected_type}
    if selected_type == "CALL_REQUIREMENT":
        if "call_shapes" in selected:
            selected_fields["call_shapes"] = selected["call_shapes"]
        elif "must_accept" in selected:
            selected_fields["call_shapes"] = selected["must_accept"]
        if "defaults" in selected:
            selected_fields["defaults"] = selected["defaults"]
    elif selected_type == "CLAUSE_INTERPRETATION" and "proposition" in selected:
        selected_fields["proposition"] = selected["proposition"]
    else:
        for key in ("behavior", "owner", "adapters", "alias", "target", "mapping"):
            if key in selected:
                selected_fields[key] = selected[key]
    resolution = row.get("resolution") if isinstance(row.get("resolution"), Mapping) else {}
    dependencies = [
        {"facet": item.get("facet"), "digest": item.get("digest")}
        for item in row.get("basis_dependencies", ())
        if isinstance(item, Mapping)
    ]
    lines = [
        "RECORD id=" + _compact(row.get("id"))
        + " version=" + _compact(row.get("version"))
        + " lifecycle=" + _compact(row.get("lifecycle"))
        + " basis_binding=" + _compact(row.get("basis_binding")),
        "KEY subject=" + _compact(row.get("key", {}).get("subject")
                                  if isinstance(row.get("key"), Mapping) else "")
        + " question=" + _compact(row.get("key", {}).get("question")
                                   if isinstance(row.get("key"), Mapping) else ""),
        "SCOPE " + _compact(row.get("scope", "")),
        "SELECTED " + _compact(selected_fields),
        "BASIS refs=" + _compact(row.get("basis_refs", []))
        + " dependencies=" + _compact(dependencies),
    ]
    if resolution:
        lines.append("RESOLUTION " + _compact(resolution))
    if row.get("challenge_reason"):
        lines.append("CHALLENGE reason=" + _compact(row["challenge_reason"]))
    return "\n".join(lines)


def _short_revision(row: Mapping[str, object]) -> str:
    """Rendered (not stored) basis revision: a 16-hex prefix of the kernel stamp."""
    provenance = row.get("provenance")
    value = provenance.get("basis_revision") if isinstance(provenance, Mapping) else None
    if not isinstance(value, str) or not value:
        return "UNKNOWN"
    algorithm, _sep, hexdigest = value.partition(":")
    return f"{algorithm}:{hexdigest[:16]}" if hexdigest else value[:24]


def _record_text(row: Mapping[str, object]) -> str:
    binding = str(row.get("basis_binding", "UNKNOWN"))
    lifecycle = str(row.get("lifecycle", "UNKNOWN"))
    key = row.get("key") if isinstance(row.get("key"), Mapping) else {}
    record_head = (
        "RECORD id=" + _compact(row.get("id"))
        + " version=" + _compact(row.get("version"))
        + " lifecycle=" + _compact(row.get("lifecycle"))
        + " basis_binding=" + _compact(row.get("basis_binding"))
    )
    key_line = ("KEY subject=" + _compact(key.get("subject", ""))
                + " question=" + _compact(key.get("question", "")))
    if lifecycle == "DISPUTED" or binding != "CURRENT":
        changed = row.get("basis_changed_facets")
        if not isinstance(changed, list) or not changed:
            changed = ["unknown"] if binding == "UNKNOWN" else []
        return "\n".join((
            record_head,
            key_line,
            "SCOPE " + _compact(row.get("scope", "")),
            "REOPEN binding=" + _compact(binding)
            + " lifecycle=" + _compact(lifecycle)
            + " facets=" + _compact(changed)
            + " reopen_on=" + _compact(row.get("reopen_on", [])),
            "PROVENANCE basis_revision=" + _compact(_short_revision(row)),
        ))
    selected = row.get("selected") if isinstance(row.get("selected"), Mapping) else {}
    selected_type = selected.get("type", "")
    selected_fields: dict[str, object] = {"type": selected_type}
    if selected_type == "CALL_REQUIREMENT":
        if "call_shapes" in selected:
            selected_fields["call_shapes"] = selected["call_shapes"]
        elif "must_accept" in selected:
            selected_fields["call_shapes"] = selected["must_accept"]
        if "defaults" in selected:
            selected_fields["defaults"] = selected["defaults"]
    elif selected_type == "CLAUSE_INTERPRETATION" and "proposition" in selected:
        selected_fields["proposition"] = selected["proposition"]
    else:
        for key in ("behavior", "owner", "adapters", "alias", "target", "mapping"):
            if key in selected:
                selected_fields[key] = selected[key]
    resolution = row.get("resolution") if isinstance(row.get("resolution"), Mapping) else {}
    # r44: short digests keep a CURRENT record small enough to share the budget.
    dependencies = [
        str(item.get("facet")) + ":" + str(item.get("digest", "")).split(":", 1)[-1][:10]
        for item in row.get("basis_dependencies", ())
        if isinstance(item, Mapping)
    ]
    lines = [
        record_head,
        key_line,
        "SCOPE " + _compact(row.get("scope", "")),
        "SELECTED " + _compact(selected_fields),
        "BASIS refs=" + _compact(row.get("basis_refs", []))
        + " dependencies=" + _compact(dependencies),
    ]
    rejected = row.get("rejected")
    if isinstance(rejected, Sequence) and not isinstance(rejected, (str, bytes, bytearray)):
        rejected = list(rejected)[:MAX_RENDER_REJECTIONS]
        if rejected:
            lines.append("REJECTED " + _compact(rejected))
    if resolution:
        lines.append("RESOLUTION " + _compact(resolution))
    if isinstance(row.get("provenance"), Mapping):
        lines.append("PROVENANCE basis_revision=" + _compact(_short_revision(row)))
    if row.get("challenge_reason"):
        lines.append("CHALLENGE reason=" + _compact(row["challenge_reason"]))
    return "\n".join(lines)


def _selected_rows(store: DecisionCacheStore, active_symbols: Sequence[object]) -> list[dict[str, object]]:
    focus = [str(item).replace("\\", "/") for item in active_symbols if str(item)]
    key = (store._record_generation, tuple(focus))
    cached = store._selected_rows_cache.get(key)
    if cached is not None:
        return cached
    rows = store._active_records()

    def score(pair: tuple[int, Mapping[str, object]]) -> tuple[object, ...]:
        index, row = pair
        key = row.get("key") if isinstance(row.get("key"), Mapping) else {}
        searchable = json.dumps({
            "key": key,
            "scope": row.get("scope", ""),
            "basis_refs": row.get("basis_refs", ()),
            "dependencies": row.get("basis_dependencies", ()),
        }, sort_keys=True, ensure_ascii=False, default=str).lower()
        overlap = sum(1 for value in focus if value.lower() in searchable)
        pinned = int(row.get("lifecycle") == "DISPUTED" or row.get("basis_binding") in {"STALE", "UNKNOWN"})
        return (-overlap, -pinned, -int(row.get("version", 0)), -index)

    selected = [row for _index, row in sorted(enumerate(rows), key=score)]
    store._selected_rows_cache[key] = selected
    return selected


@profiled_function_span("decision_cache_render")
def render_settled_state(store: DecisionCacheStore | Mapping[str, object], *,
                         work_frontier: Sequence[object] = (),
                         active_symbols: Sequence[object] = (),
                         max_tokens: int = MAX_RENDER_TOKENS,
                         receipts: Sequence[object] | None = None,
                         current_revision: object = None,
                         receipt_generation: object = None,
                         receipt_cursor: object = None,
                         emit: Callable[..., object] | None = None) -> str:
    """Render a bounded, deterministic projection with whole-record overflow."""
    if type(max_tokens) is not int or max_tokens < 16:
        raise ValueError("max_tokens must be an integer of at least 16")
    cache = store if isinstance(store, DecisionCacheStore) else DecisionCacheStore(state=store)
    if not _ext():
        return _render_settled_state_r43(cache, work_frontier=work_frontier,
                                         active_symbols=active_symbols,
                                         max_tokens=max_tokens, emit=emit)
    return _render_settled_state_r44(cache, work_frontier=work_frontier,
                                     active_symbols=active_symbols, max_tokens=max_tokens,
                                      receipts=receipts, current_revision=current_revision,
                                      receipt_generation=receipt_generation,
                                      receipt_cursor=receipt_cursor,
                                      emit=emit)


def _render_settled_state_r43(cache: DecisionCacheStore, *,
                              work_frontier: Sequence[object],
                              active_symbols: Sequence[object],
                              max_tokens: int,
                              emit: Callable[..., object] | None) -> str:
    """The r43 projection, unchanged, used whenever the r44 switch is off."""
    rows = _selected_rows(cache, active_symbols)
    frontier = [str(item) for item in work_frontier if str(item)]
    if enabled() and not rows and not frontier and not cache.delta():
        return ""
    active_ids = {str(row.get("id")) for row in rows}
    delta = [{key: copy.deepcopy(value) for key, value in change.items()
              if key in {"id", "change", "from", "to", "facets", "reason"}}
             for change in cache.delta()
             if str(change.get("id")) in active_ids]
    open_items = [
        row["id"] for row in rows
        if row.get("lifecycle") == "DISPUTED" or row.get("basis_binding") in {"STALE", "UNKNOWN"}
    ]
    static = [
        "WORK frontier=" + _compact(list(dict.fromkeys(frontier))),
        "DELTA changes=" + _compact(delta),
        "OPEN items=" + _compact(list(dict.fromkeys(open_items))),
    ]
    chosen: list[str] = []
    dropped: list[str] = []

    def compose(record_blocks: Sequence[str], dropped_ids: Sequence[str]) -> tuple[str, int, str]:
        manifest = ("MANIFEST selected_records=" + str(len(record_blocks))
                    + " dropped_records=" + _compact(list(dropped_ids))
                    + " estimator=")
        provisional = "<settled_state>\n" + "\n".join(
            [*static, *record_blocks, manifest + "proxy:chars/4"]
        ) + "\n</settled_state>"
        count, estimator = _token_counter(provisional)
        rendered = "<settled_state>\n" + "\n".join(
            [*static, *record_blocks,
             manifest + estimator + " rendered_tokens=" + str(count)]
        ) + "\n</settled_state>"
        final_count, _ = _token_counter(rendered)
        return rendered, final_count, estimator

    for row in rows:
        block = cache._render_record_r43(row)
        candidate, count, _ = compose([*chosen, block], dropped)
        if count <= max_tokens:
            chosen.append(block)
        else:
            ident = str(row.get("id"))
            dropped.append(ident)
    rendered, count, estimator = compose(chosen, dropped)
    while count > max_tokens and chosen:
        # Remove complete records from the stable tail; no partial record is sent.
        last = chosen.pop()
        match = re.search(r"RECORD id=(\"[^\"]+\"|[^ ]+)", last)
        dropped.append(json.loads(match.group(1)) if match and match.group(1).startswith('"')
                       else match.group(1) if match else "unknown")
        rendered, count, estimator = compose(chosen, dropped)
    cache.metrics["rendered"] += 1
    cache.metrics["rendered_tokens"] = count
    cache.metrics["dropped_records"] += len(dropped)
    if emit is not None:
        emit("decision_cache_render", selected_records=len(chosen),
             dropped_records=list(dropped), rendered_tokens=count,
             estimator=estimator, max_tokens=max_tokens)
    return rendered


MAX_FRONTIER_ITEM_CHARS = 160
MAX_MANIFEST_IDS = 8


def _render_settled_state_r44(cache: DecisionCacheStore, *,
                              work_frontier: Sequence[object],
                              active_symbols: Sequence[object],
                              max_tokens: int,
                              receipts: Sequence[object] | None,
                              current_revision: object,
                              receipt_generation: object,
                              receipt_cursor: object,
                              emit: Callable[..., object] | None) -> str:
    """Whole-projection budget: records, receipt rows, frontier and manifest.

    Every variable-size part is either capped per item or removed whole, and the
    final text is re-measured; if even the empty skeleton cannot fit, nothing is
    rendered (fail closed) instead of exceeding ``max_tokens``.
    """
    from kernel_conclusion_cache import clip

    all_rows = _selected_rows(cache, [*active_symbols, *work_frontier])
    rows = all_rows[:MAX_RENDER_RECORDS]
    overflow_ids = [str(row.get("id")) for row in all_rows[MAX_RENDER_RECORDS:]]
    frontier = list(dict.fromkeys(clip(item, MAX_FRONTIER_ITEM_CHARS)
                                  for item in work_frontier if str(item)))
    if receipts is None:
        receipts = cache._context.get("receipts", ())
    all_receipt_lines = cache._receipt_lines(
        receipts, current_revision, receipt_generation, receipt_cursor)
    receipt_rows = all_receipt_lines[-MAX_RENDER_RECEIPTS:]
    receipts_over_cap = len(all_receipt_lines) - len(receipt_rows)
    if not rows and not frontier and not cache.delta() and not receipt_rows:
        return ""
    blocks = [(str(row.get("id")), row, cache._render_record(row)) for row in rows]
    full_delta = cache.delta()

    def compose(chosen: Sequence[int], chosen_receipts: Sequence[int],
                frontier_items: Sequence[str]) -> tuple[str, int, str]:
        chosen_rows = [blocks[index] for index in chosen]
        ids = {ident for ident, _row, _text in chosen_rows}
        delta = [{key: copy.deepcopy(value) for key, value in change.items()
                  if key in {"id", "change", "from", "to", "facets", "reason"}}
                 for change in full_delta if str(change.get("id")) in ids]
        open_items = [ident for ident, row, _text in chosen_rows
                      if row.get("lifecycle") == "DISPUTED"
                      or row.get("basis_binding") in {"STALE", "UNKNOWN"}]
        dropped_ids = [ident for index, (ident, _row, _text) in enumerate(blocks)
                       if index not in chosen] + overflow_ids
        lines = [
            "WORK frontier=" + _compact(list(frontier_items)),
            "DELTA changes=" + _compact(delta),
            "OPEN items=" + _compact(list(dict.fromkeys(open_items))),
        ]
        if chosen_receipts:
            lines.append("REUSABLE RECEIPTS")
            lines.extend(receipt_rows[index] for index in sorted(chosen_receipts))
        lines.extend(text for _ident, _row, text in chosen_rows)
        manifest = ("MANIFEST selected_records=" + str(len(chosen_rows))
                    + " dropped_records=" + str(len(dropped_ids))
                    + " dropped_ids=" + _compact(dropped_ids[:MAX_MANIFEST_IDS])
                    + " selected_receipts=" + str(len(chosen_receipts))
                    + " dropped_receipts=" + str(
                        receipts_over_cap + len(receipt_rows) - len(chosen_receipts))
                    + " frontier_omitted=" + str(len(frontier) - len(frontier_items))
                    + " estimator=")
        provisional = "<settled_state>\n" + "\n".join(
            [*lines, manifest + "proxy:chars/4"]) + "\n</settled_state>"
        count, estimator = _token_counter(provisional)
        rendered = "<settled_state>\n" + "\n".join(
            [*lines, manifest + estimator + " rendered_tokens=" + str(count)]
        ) + "\n</settled_state>"
        final_count, _ = _token_counter(rendered)
        return rendered, final_count, estimator

    chosen: list[int] = []
    chosen_receipts: list[int] = []
    shown_frontier = list(frontier)
    for index in range(len(blocks)):
        if compose([*chosen, index], chosen_receipts, shown_frontier)[1] <= max_tokens:
            chosen.append(index)
    for index in reversed(range(len(receipt_rows))):  # newest receipts first
        if compose(chosen, [*chosen_receipts, index], shown_frontier)[1] <= max_tokens:
            chosen_receipts.append(index)
    rendered, count, estimator = compose(chosen, chosen_receipts, shown_frontier)
    while count > max_tokens and (chosen_receipts or chosen or shown_frontier):
        if chosen_receipts:
            chosen_receipts.remove(min(chosen_receipts))
        elif chosen:
            chosen.pop()
        else:
            shown_frontier.pop()
        rendered, count, estimator = compose(chosen, chosen_receipts, shown_frontier)
    dropped_count = len(blocks) - len(chosen) + len(overflow_ids)
    cache.metrics["rendered"] += 1
    cache.metrics["dropped_records"] += dropped_count
    over_budget = count > max_tokens
    cache.metrics["rendered_tokens"] = 0 if over_budget else count
    if emit is not None:
        emit("decision_cache_render", selected_records=len(chosen),
             dropped_records=dropped_count, rendered_tokens=0 if over_budget else count,
             estimator=estimator, max_tokens=max_tokens,
             receipt_rows=len(chosen_receipts),
             dropped_receipts=receipts_over_cap + len(receipt_rows) - len(chosen_receipts),
             reused_receipts=sum("REUSED = no new sample" in receipt_rows[index]
                                 for index in chosen_receipts),
             over_budget=over_budget)
    return "" if over_budget else rendered


def projection_message(store: DecisionCacheStore, *, work_frontier: Sequence[object] = (),
                       active_symbols: Sequence[object] = (), emit=None) -> dict[str, str]:
    return {"role": "user", "content": render_settled_state(
        store, work_frontier=work_frontier, active_symbols=active_symbols, emit=emit)}


def memory_schema() -> dict[str, object]:
    """Describe the compact model-written arm of the host-owned cache."""
    text = {"type": "string"}
    refs = {"type": "array", "items": {"type": "string"}}
    selected = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": list(ANSWER_TYPES)},
            "answer_type": {"type": "string", "enum": list(ANSWER_TYPES)},
            "owner": text, "behavior": text, "alias": text, "target": text,
            "mapping": {}, "adapters": {},
            "call_shapes": refs, "must_accept": refs,
            "must_accept_call_shapes": refs, "defaults": {"type": "object"},
            "proposition": text, "bounded_proposition": text,
        },
        "required": ["type"],
        "additionalProperties": False,
    }
    draft = {
        "type": "object",
        "properties": {
            "key": {"type": "object", "properties": {
                "subject": text,
                "question": {"type": "string", "enum": list(DECISION_QUESTIONS)},
            }, "required": ["subject", "question"], "additionalProperties": False},
            "scope": text, "selected": selected, "basis_refs": refs,
            "assumptions": refs, "resolution": {"type": "object"},
            "rejected": {"type": "array", "items": {}},
        },
        "required": ["key", "selected"],
        "additionalProperties": False,
    }
    description = (
        "Optional decision memory. Minimal ADOPT inside ordinary tool args: "
        '{"op":"ADOPT","draft":{"key":{"subject":"path::symbol",'
        '"question":"BEHAVIOR_OWNER"},"selected":{"type":"RESPONSIBILITY",'
        '"owner":"symbol"}}}. '
        "The host assigns id, version, author, lifecycle, basis_binding, and "
        "basis_dependencies."
    )
    if _ext():
        properties = dict(draft["properties"])
        draft["properties"] = {
            **{key: properties[key] for key in ("key", "scope", "selected",
                                                "basis_refs", "assumptions")},
            "reopen_on": {"type": "array", "maxItems": 8, "items": text},
            **{key: properties[key] for key in ("resolution", "rejected")},
        }
        description = description[:-len("basis_dependencies.")] + (
            "basis_dependencies and provenance; stale premises require reopening.")
    return {
        "type": "object",
        "description": description,
        "properties": {
            "op": {"type": "string", "enum": list(MEMORY_OPS)},
            "operation": {"type": "string", "enum": list(MEMORY_OPS)},
            "draft": draft,
            "supersedes": text, "decision_id": text, "target": text,
            "reason": {"type": "string", "enum": list(CHALLENGE_REASONS)},
            "basis_refs": refs, "counterevidence_refs": refs,
        },
        "required": ["op"],
        "additionalProperties": False,
    }


def tool_surface(base: object) -> object:
    """Advertise an optional args-side memory field for native tool transports."""
    if not enabled() or base is None:
        return base
    result = copy.deepcopy(list(base))
    for tool in result:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        parameters = function.get("parameters") if isinstance(function, Mapping) else None
        if not isinstance(function, dict) or not isinstance(parameters, dict):
            continue
        properties = parameters.setdefault("properties", {})
        if isinstance(properties, dict):
            # Replace an old/empty arm too: a caller may have reused a tool
            # list that was serialized before the cache switch was enabled.
            properties["memory"] = memory_schema()
        parameters.setdefault("additionalProperties", False)
    from kernel_tool_schema_diet import filter_tools as filter_single_carrier_tools
    return filter_single_carrier_tools(result)


def split_action_arguments(arguments: object, *, call: Mapping[str, object] | None = None,
                           message: Mapping[str, object] | None = None) -> tuple[dict[str, object], object]:
    """Remove only the optional sidecar field before ordinary tool validation/dispatch."""
    if not isinstance(arguments, Mapping):
        return {}, None
    result = copy.deepcopy(dict(arguments))
    sidecar = None
    if enabled() and "memory" in result:
        sidecar = result.pop("memory")
    if enabled() and isinstance(call, Mapping) and "memory" in call:
        sidecar = call.get("memory")
    if enabled() and isinstance(message, Mapping) and "memory" in message:
        sidecar = message.get("memory")
    return result, sidecar


def collect_receipt_refs(frame: object) -> set[str]:
    """Collect only receipt/evidence ids actually serialized in an input frame."""
    found: set[str] = set()

    def visit(value: object, key: str = "") -> None:
        if isinstance(value, Mapping):
            for name, item in value.items():
                name = str(name)
                if name == "receipt_id" and isinstance(item, str):
                    found.add(item)
                elif name in {"receipt_ids", "evidence_refs", "receipt_refs"} and isinstance(item, list):
                    found.update(str(ref) for ref in item if isinstance(ref, str))
                else:
                    visit(item, name)
        elif isinstance(value, list):
            for item in value:
                visit(item, key)

    visit(frame)
    return found


def sidecar_from_call(call: Mapping[str, object] | None, *, message: Mapping[str, object] | None = None,
                      arguments: Mapping[str, object] | None = None) -> object:
    if not enabled():
        return None
    if isinstance(call, Mapping) and "memory" in call:
        return call.get("memory")
    if isinstance(message, Mapping) and "memory" in message:
        return message.get("memory")
    if isinstance(arguments, Mapping) and "memory" in arguments:
        return arguments.get("memory")
    return None


__all__ = [
    "ANSWER_TYPES", "BASIS_BINDINGS", "CHALLENGE_REASONS", "DecisionCacheError",
    "DecisionCacheStore", "DecisionDraft", "DECISION_LIFECYCLES", "DECISION_QUESTIONS",
    "MAX_RENDER_RECORDS", "MAX_RENDER_RECEIPTS", "MAX_RENDER_TOKENS", "MEMORY_OPS",
    "MODE_ENV", "RESOLUTION_KINDS", "SCHEMA",
    "collect_receipt_refs", "decision_cache_enabled", "enabled", "memory_schema", "mode",
    "projection_message", "render_reusable_receipts", "render_settled_state",
    "sidecar_from_call", "split_action_arguments",
    "tool_surface",
]


decision_cache_enabled = enabled

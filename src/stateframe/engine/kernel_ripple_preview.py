"""RIPPLE PREVIEW v1: bounded, revision-bound change-impact advice.

The organ is intentionally independent of the StateFrame completion reducer.  It
answers three different questions without turning a relationship into a demand:

* what deserves a review;
* what check would make the review cheaper; and
* which already-admitted witness creates a real obligation.

``STATEFRAME_RIPPLE_PREVIEW`` is the only activation switch.  The module is
usable as a pure package for offline evaluation and has a small adapter surface
for the live runtime.  The adapter accepts graph-shaped mappings so concurrent
graph lanes can be merged later without making this lane import their modules at
startup.
"""
from __future__ import annotations

import ast
import copy
import functools
import hashlib
import json
import os
import re
import textwrap
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field


MODE_ENV = "STATEFRAME_RIPPLE_PREVIEW"
SCHEMA = "stateframe-ripple-preview/1"
MODES = ("off", "shadow", "advisory")

# State axes from change ledger. They are strings on purpose: journals and receipts
# must survive a Python-version or enum refactor at the merge point.
ADVISORY = "ADVISORY"
CHECK_SUGGESTED = "CHECK_SUGGESTED"
WITNESSED_OBLIGATION = "WITNESSED_OBLIGATION"
CURRENT = "CURRENT"
STALE = "STALE"
SUPERSEDED = "SUPERSEDED"

_AUTHORITIES = {ADVISORY, CHECK_SUGGESTED, WITNESSED_OBLIGATION}
_VALIDITIES = {CURRENT, STALE, SUPERSEDED}
_DISPOSITIONS = {
    "unresolved", "inspecting", "changed-awaiting-check", "changed-verified",
    "verified-unaffected", "reviewed-no-change", "out-of-scope", "deferred",
}

MIN_CANDIDATE_SCORE = 2.0
MAX_RENDER_TOKENS = 300
MAX_RENDER_GROUPS = 4
MAX_RENDER_ITEMS = 8
MAX_REASON_CHARS = 180
_MAX_WHERE_USED_SCAN_FILES = 256
_MAX_WHERE_USED_SCAN_DIRECTORIES = 256
_MAX_WHERE_USED_SCAN_BYTES = 16 * 1024 * 1024

_GRAPH_MODULES = {
    "where_used_closure": "kernel_where_used",
    "repo_signals": "kernel_repo_signals",
    "requirements_ledger": "kernel_requirements_ledger",
    "project_registries": "kernel_project_registries",
    "registry_parity": "kernel_registry_parity",
    "other_end_contract": "kernel_other_end_contract",
    "same_shape_sweep": "kernel_same_shape_sweep",
}

_DIMENSION_ALIASES = {
    "wire": "wire_field",
    "wire_key": "wire_field",
    "json_key": "wire_field",
    "schema_field": "wire_field",
    "field": "wire_field",
    "wire_field": "wire_field",
    "local": "local_var",
    "variable": "local_var",
    "local_variable": "local_var",
    "local_var": "local_var",
    "attribute": "attribute",
    "attr": "attribute",
    "event": "event_field",
    "event_key": "event_field",
    "event_field": "event_field",
    "column": "column",
    "database_column": "column",
    "cli": "cli_option",
    "option": "cli_option",
    "cli_option": "cli_option",
    "signature": "signature",
    "parameter": "signature",
    "param": "signature",
    "optional_parameter": "optional_parameter",
    "registry": "registry_member",
    "registration": "registry_member",
    "registry_member": "registry_member",
    "config": "config_key",
    "config_key": "config_key",
    "contract": "contract",
    "same_shape": "same_shape",
    "call_site": "call_site",
    "behaviour": "behaviour",
    "behavior": "behaviour",
    "unknown": "unknown",
}

_FAMILY_ALIASES = {
    "where_used": "structural",
    "where_used_closure": "structural",
    "closure": "structural",
    "call_graph": "structural",
    "repo_signals": "history",
    "co_change": "history",
    "history": "history",
    "text": "textual",
    "textual": "textual",
    "runtime": "runtime",
    "dynamic": "runtime",
    "structural": "structural",
    "same_shape": "same_shape",
    "same_shape_sweep": "same_shape",
    "parity": "parity",
    "registry_parity": "parity",
    "project_registries": "parity",
    "contract": "contract",
    "other_end_contract": "contract",
    "requirements": "contract",
    "requirements_ledger": "contract",
    "registry": "parity",
}

_FAMILY_WEIGHT = {
    "witness": 100.0,
    "contract": 4.0,
    "parity": 4.0,
    "structural": 3.0,
    "runtime": 3.0,
    "same_shape": 2.0,
    "history": 1.0,
    "textual": 1.0,
    "unknown": 1.0,
}

_FAMILY_PRIORITY = {
    "witness": 0,
    "contract": 1,
    "parity": 1,
    "structural": 2,
    "runtime": 2,
    "same_shape": 3,
    "history": 4,
    "textual": 4,
    "unknown": 5,
}

_EDIT_KINDS = frozenset({
    "edit", "str_replace", "file_create", "replace_module", "write", "replace",
})


def mode() -> str:
    """Read and validate the strict three-way activation switch."""
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in MODES:
        raise ValueError(f"{MODE_ENV} must be off, shadow or advisory")
    return value


def enabled() -> bool:
    return mode() != "off"


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _text(value: object, default: str = "") -> str:
    return value.strip() if isinstance(value, str) else default


def _clip(value: object, limit: int = MAX_REASON_CHARS) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _relative_target(value: object) -> str:
    target = _text(value)
    if not target:
        return "unindexed target"
    return target.replace("\\", "/")


def _normal_dimension(value: object) -> str:
    raw = _text(value, "unknown").lower().replace("-", "_").replace(" ", "_")
    return _DIMENSION_ALIASES.get(raw, raw or "unknown")


def _dimensions(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values = re.split(r"[,|]", value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = [value]
    out = []
    for item in values:
        dimension = _normal_dimension(item)
        if dimension not in out and dimension != "unknown":
            out.append(dimension)
    return tuple(out or ["unknown"])


def _normal_family(value: object) -> str:
    raw = _text(value, "unknown").lower().replace("-", "_").replace(" ", "_")
    return _FAMILY_ALIASES.get(raw, raw or "unknown")


def _source_digest(source: object) -> str:
    return "sha256:" + hashlib.sha256(str(source or "").encode("utf-8")).hexdigest()


def _safe_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class Evidence:
    """One typed vote from one graph family."""

    family: str
    reason: str = ""
    strength: float = 1.0
    dimension: str = "unknown"
    correlation_id: str = ""
    source: str = ""
    ast_id: str = ""

    @classmethod
    def from_value(cls, value: object, *, default_family: str = "unknown",
                   default_dimension: str = "unknown") -> "Evidence":
        if isinstance(value, Mapping):
            family = _normal_family(value.get("family", value.get("source", default_family)))
            raw_strength = value.get("strength", value.get("confidence", 1.0))
            try:
                strength = max(0.0, float(raw_strength))
            except (TypeError, ValueError):
                strength = 1.0
            correlation = value.get("correlation_id", value.get("correlated_by",
                                      value.get("node_id", value.get("ast_id", ""))))
            return cls(
                family=family,
                reason=_clip(value.get("reason", value.get("detail", ""))),
                strength=strength,
                dimension=_normal_dimension(value.get("dimension", default_dimension)),
                correlation_id=_text(correlation),
                source=_text(value.get("source", "")),
                ast_id=_text(value.get("ast_id", value.get("node_id", ""))),
            )
        return cls(family=_normal_family(default_family),
                   dimension=_normal_dimension(default_dimension),
                   reason=_clip(value))

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family, "reason": self.reason,
            "strength": self.strength, "dimension": self.dimension,
            "correlation_id": self.correlation_id, "source": self.source,
            "ast_id": self.ast_id,
        }


@dataclass(frozen=True)
class Candidate:
    """A review/check candidate after graph evidence has been merged."""

    candidate_id: str
    target: str
    dimension: str
    decision: str
    reason: str
    evidence: tuple[Evidence, ...] = ()
    authority: str = ADVISORY
    score: float = 0.0
    priority: int = 5
    required_role: str | None = None
    requirement: str = ""
    witness: Mapping[str, object] = field(default_factory=dict)
    optional_change: bool = False
    correlated_evidence: int = 0

    @property
    def is_obligation(self) -> bool:
        return self.authority == WITNESSED_OBLIGATION

    @property
    def actions(self) -> tuple[str, ...]:
        # Only host tools that exist on the live menu (read / run_tests).  A
        # re-preview happens by re-issuing the edit, not by a phantom tool.
        target = _clip(self.target.split(":", 1)[0], 96)
        if self.decision == "check":
            return (f"inspect=read({target})", f"check=run_tests({target})")
        return (f"inspect=read({target})",)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "target": self.target,
            "dimension": self.dimension,
            "decision": self.decision,
            "reason": self.reason,
            "evidence": [item.to_dict() for item in self.evidence],
            "authority": self.authority,
            "score": round(self.score, 3),
            "priority": self.priority,
            "required_role": self.required_role,
            "requirement": self.requirement,
            "witness": dict(self.witness),
            "optional_change": self.optional_change,
            "correlated_evidence": self.correlated_evidence,
            "actions": list(self.actions),
        }


@dataclass(frozen=True)
class Disposition:
    """Durable state for one candidate, including the revision binding."""

    candidate_id: str
    authority: str
    validity: str
    disposition: str
    base_revision: str
    patch_hash: str
    target: str = ""
    trigger: str = "candidate-patch"
    requirement: str = ""
    contract_endpoints: tuple[str, ...] = ()
    checker_version: str = ""
    input_digest: str = ""
    observed_violation: str = ""
    control: str = ""
    deferred_trigger: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id, "authority": self.authority,
            "validity": self.validity, "disposition": self.disposition,
            "base_revision": self.base_revision, "patch_hash": self.patch_hash,
            "target": self.target, "trigger": self.trigger,
            "requirement": self.requirement,
            "contract_endpoints": list(self.contract_endpoints),
            "checker_version": self.checker_version,
            "input_digest": self.input_digest,
            "observed_violation": self.observed_violation,
            "control": self.control, "deferred_trigger": self.deferred_trigger,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Disposition":
        endpoints = value.get("contract_endpoints", ())
        if isinstance(endpoints, str):
            endpoints = (endpoints,)
        elif not isinstance(endpoints, Sequence):
            endpoints = ()
        authority = _text(value.get("authority"), ADVISORY)
        validity = _text(value.get("validity"), CURRENT)
        disposition = _text(value.get("disposition"), "unresolved")
        return cls(
            candidate_id=_text(value.get("candidate_id")), authority=authority,
            validity=validity, disposition=disposition,
            base_revision=_text(value.get("base_revision")),
            patch_hash=_text(value.get("patch_hash")), target=_text(value.get("target")),
            trigger=_text(value.get("trigger"), "candidate-patch"),
            requirement=_text(value.get("requirement")),
            contract_endpoints=tuple(str(item) for item in endpoints),
            checker_version=_text(value.get("checker_version")),
            input_digest=_text(value.get("input_digest")),
            observed_violation=_text(value.get("observed_violation")),
            control=_text(value.get("control")),
            deferred_trigger=_text(value.get("deferred_trigger")),
        )


class DispositionStore:
    """Small journal-replayable store for the six change ledger state axes."""

    def __init__(self, records: Iterable[Disposition] = ()):
        self._records: dict[str, Disposition] = {
            record.candidate_id: record for record in records
            if isinstance(record, Disposition) and record.candidate_id
        }

    def get(self, candidate_id: str) -> Disposition | None:
        return self._records.get(str(candidate_id))

    def values(self) -> tuple[Disposition, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def observe(self, candidate_id: str, *, authority: str = ADVISORY,
                base_revision: str, patch_hash: str, target: str = "",
                requirement: str = "", witness: Mapping[str, object] | None = None,
                trigger: str = "candidate-patch") -> Disposition:
        candidate_id = str(candidate_id)
        authority = authority if authority in _AUTHORITIES else ADVISORY
        old = self._records.get(candidate_id)
        same_binding = (old is not None and old.base_revision == base_revision
                        and old.patch_hash == patch_hash)
        witness = witness if isinstance(witness, Mapping) else {}
        endpoints = witness.get("contract_endpoints", ())
        if not isinstance(endpoints, Sequence) or isinstance(endpoints, (str, bytes)):
            endpoints = ()
        record = Disposition(
            candidate_id=candidate_id, authority=authority,
            validity=CURRENT, disposition=(old.disposition if same_binding else "unresolved"),
            base_revision=str(base_revision), patch_hash=str(patch_hash),
            target=_relative_target(target), trigger=trigger, requirement=requirement,
            contract_endpoints=tuple(str(item) for item in endpoints),
            checker_version=_text(witness.get("checker_version")),
            input_digest=_text(witness.get("input_digest")),
            observed_violation=_text(witness.get("observed_violation")),
            control=_text(witness.get("control")),
            deferred_trigger=(old.deferred_trigger if same_binding else ""),
        )
        self._records[candidate_id] = record
        return record

    def set_disposition(self, candidate_id: str, disposition: str, *,
                        base_revision: str | None = None,
                        patch_hash: str | None = None,
                        deferred_trigger: str = "") -> Disposition:
        current = self._records[str(candidate_id)]
        if disposition not in _DISPOSITIONS:
            raise ValueError(f"unknown Ripple Preview disposition: {disposition!r}")
        if base_revision is not None and str(base_revision) != current.base_revision:
            updated = self._replace(current, validity=STALE, disposition=disposition,
                                    deferred_trigger=deferred_trigger)
            self._records[current.candidate_id] = updated
            return updated
        if patch_hash is not None and str(patch_hash) != current.patch_hash:
            updated = self._replace(current, validity=STALE, disposition=disposition,
                                    deferred_trigger=deferred_trigger)
            self._records[current.candidate_id] = updated
            return updated
        updated = self._replace(current, disposition=disposition,
                                deferred_trigger=deferred_trigger)
        self._records[current.candidate_id] = updated
        return updated

    def mark_stale(self, *, base_revision: str, patch_hash: str) -> tuple[Disposition, ...]:
        changed = []
        for key, current in list(self._records.items()):
            if current.base_revision == str(base_revision) and current.patch_hash == str(patch_hash):
                updated = self._replace(current, validity=CURRENT)
            else:
                updated = self._replace(current, validity=STALE)
            self._records[key] = updated
            changed.append(updated)
        return tuple(changed)

    def _replace(self, current: Disposition, **changes: object) -> Disposition:
        return Disposition(
            candidate_id=current.candidate_id,
            authority=str(changes.get("authority", current.authority)),
            validity=str(changes.get("validity", current.validity)),
            disposition=str(changes.get("disposition", current.disposition)),
            base_revision=current.base_revision, patch_hash=current.patch_hash,
            target=current.target, trigger=current.trigger, requirement=current.requirement,
            contract_endpoints=current.contract_endpoints,
            checker_version=current.checker_version, input_digest=current.input_digest,
            observed_violation=current.observed_violation, control=current.control,
            deferred_trigger=str(changes.get("deferred_trigger", current.deferred_trigger)),
        )

    def snapshot(self) -> dict[str, object]:
        return {"payload_schema": "stateframe-ripple-dispositions/1",
                "records": [item.to_dict() for item in self.values()]}

    @classmethod
    def from_snapshot(cls, value: object) -> "DispositionStore":
        if not isinstance(value, Mapping):
            return cls()
        rows = value.get("records", ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            return cls()
        return cls(Disposition.from_mapping(row) for row in rows
                   if isinstance(row, Mapping))

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]]) -> "DispositionStore":
        latest: dict[str, Disposition] = {}
        for event in records:
            if not isinstance(event, Mapping):
                continue
            if event.get("kind") not in {"ripple_preview_disposition", "ripple_preview_state"}:
                continue
            rows = event.get("dispositions", event.get("records", ()))
            if isinstance(rows, Mapping):
                rows = (rows,)
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                for row in rows:
                    if isinstance(row, Mapping) and row.get("candidate_id"):
                        record = Disposition.from_mapping(row)
                        latest[record.candidate_id] = record
        return cls(latest.values())


@dataclass(frozen=True)
class PatchOverlay:
    path: str
    base_source: str
    after_source: str
    base_revision: str
    patch_hash: str
    binding: str
    valid: bool
    error: str = ""
    changed_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path, "base_revision": self.base_revision,
            "patch_hash": self.patch_hash, "binding": self.binding,
            "valid": self.valid, "error": self.error,
            "changed_paths": list(self.changed_paths),
        }


@dataclass(frozen=True)
class Reconciliation:
    status: str
    base_revision: str
    patch_hash: str
    changed_targets: tuple[str, ...] = ()
    recall: Mapping[str, object] = field(default_factory=dict)
    dispositions: tuple[Disposition, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status, "base_revision": self.base_revision,
            "patch_hash": self.patch_hash,
            "changed_targets": list(self.changed_targets),
            "recall": dict(self.recall),
            "dispositions": [item.to_dict() for item in self.dispositions],
            "completion_blockers": [],
        }


@dataclass(frozen=True)
class Preview:
    mode: str
    overlay: PatchOverlay | None
    dimensions: tuple[str, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    obligations: tuple[Candidate, ...] = ()
    shared_facts: tuple[str, ...] = ()
    source_graphs: tuple[str, ...] = ()
    unindexed_scope: tuple[str, ...] = ()
    max_groups: int = MAX_RENDER_GROUPS
    fired: bool = False
    engine: "RipplePreviewEngine | None" = field(default=None, repr=False, compare=False)

    @property
    def base_revision(self) -> str:
        return self.overlay.base_revision if self.overlay is not None else ""

    @property
    def patch_hash(self) -> str:
        return self.overlay.patch_hash if self.overlay is not None else ""

    @property
    def binding(self) -> str:
        return self.overlay.binding if self.overlay is not None else ""

    @property
    def total_candidates(self) -> int:
        return len(self.candidates)

    @property
    def completion_blockers(self) -> tuple[object, ...]:
        # Ripple advice never owns completion.  Existing witness admission is
        # represented separately in ``obligations`` and can be consumed by an
        # existing authority at its own boundary.
        return ()

    def render(self) -> str:
        return render_preview(self)

    def to_dict(self) -> dict[str, object]:
        if self.mode == "off":
            return {"schema": SCHEMA, "mode": "off", "fired": False}
        return {
            "schema": SCHEMA, "mode": self.mode, "fired": self.fired,
            "base_revision": self.base_revision, "patch_hash": self.patch_hash,
            "binding": self.binding, "dimensions": list(self.dimensions),
            "candidates": [item.to_dict() for item in self.candidates],
            "obligations": [item.to_dict() for item in self.obligations],
            "shared_facts": list(self.shared_facts),
            "source_graphs": list(self.source_graphs),
            "unindexed_scope": list(self.unindexed_scope),
            "render": self.render(),
        }


def _literal_strings(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError, ValueError):
        return set()
    return {str(node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def _identifier_tokens(source: str) -> set[str]:
    try:
        tree = ast.parse(source or "")
    except (SyntaxError, TypeError, ValueError):
        return set(re.findall(r"\b[A-Za-z_]\w*\b", source or ""))
    tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.arg):
            tokens.add(node.arg)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            tokens.add(node.arg)
    return tokens


def _function_signatures(source: str) -> dict[str, tuple[str, ...]]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError, ValueError):
        return {}
    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        positional = [*node.args.posonlyargs, *node.args.args]
        positional_required = len(positional) - len(node.args.defaults)
        positional_flags = (["required"] * positional_required
                            + ["optional"] * len(node.args.defaults))
        keyword_only = list(node.args.kwonlyargs)
        keyword_flags = ["optional" if item is not None else "required"
                         for item in node.args.kw_defaults]
        args = [*zip(positional, positional_flags), *zip(keyword_only, keyword_flags)]
        # The exact default expression is deliberately not part of the identity;
        # an optional append is a compatibility fact, not a caller edit.
        result[node.name] = tuple(f"{arg.arg}:{flag}" for arg, flag in args)
    return result


def _signature_delta(before: Mapping[str, tuple[str, ...]],
                     after: Mapping[str, tuple[str, ...]]) -> tuple[bool, bool]:
    """Return ``(changed, compatible_append)`` over every function signature.

    A change is a verified compatible append only when EVERY changed function
    keeps its old parameter list as an exact, identical prefix and only adds
    optional parameters.  A removal, rename, reorder, required addition or a
    deleted function makes the whole change incompatible (callers transfer).
    """
    changed_names = [name for name in set(before) | set(after)
                     if before.get(name) != after.get(name)]
    if not changed_names:
        return False, False
    for name in changed_names:
        old = before.get(name)
        new = after.get(name)
        if old is None or new is None:
            return True, False
        if len(new) <= len(old) or tuple(new[:len(old)]) != tuple(old):
            return True, False
        if not all(item.endswith(":optional") for item in new[len(old):]):
            return True, False
    return True, True


def _function_bodies(source: str) -> dict[str, str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError, ValueError):
        return {}
    return {node.name: ast.dump(ast.Module(body=node.body, type_ignores=[]))
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def changed_function_names(before: str, after: str) -> tuple[str, ...]:
    """Functions whose signature or body differ between two source versions."""
    old_sigs, new_sigs = _function_signatures(before), _function_signatures(after)
    old_bodies, new_bodies = _function_bodies(before), _function_bodies(after)
    names = set(old_sigs) | set(new_sigs)
    return tuple(sorted(name for name in names
                        if old_sigs.get(name) != new_sigs.get(name)
                        or old_bodies.get(name) != new_bodies.get(name)))


def infer_change_dimensions(before: str, after: str, *, intent: object = None,
                            patch: Mapping[str, object] | None = None) -> tuple[str, ...]:
    """Infer the decisive field used to transfer graph evidence.

    This is intentionally conservative and can return more than one dimension
    for a patch that genuinely changes two surfaces.  Candidate admission still
    requires a candidate dimension to match one of the returned fields.
    """
    explicit: list[str] = []
    for value in (
        patch.get("dimension") if isinstance(patch, Mapping) else None,
        patch.get("dimensions") if isinstance(patch, Mapping) else None,
        intent.get("dimension") if isinstance(intent, Mapping) else None,
    ):
        if value is not None:
            explicit.extend(_dimensions(value))
    found: list[str] = [item for item in explicit if item != "unknown"]
    before_signatures = _function_signatures(before)
    after_signatures = _function_signatures(after)
    signature_changed, compatible_append = _signature_delta(
        before_signatures, after_signatures)
    if found:
        # A caller-supplied intent/patch dimension is the decisive field.  Do
        # not let the same spelling in a JSON string and a local identifier
        # widen a wire-field rename into a local-variable rename.  Signature
        # analysis is retained only for the compatibility fact below.
        if compatible_append:
            found.append("optional_parameter")
        result: list[str] = []
        for dimension in found:
            dimension = _normal_dimension(dimension)
            if dimension not in result and dimension != "unknown":
                result.append(dimension)
        return tuple(result or ["unknown"])
    if signature_changed:
        found.append("signature")
        if compatible_append:
            found.append("optional_parameter")
    old_literals = _literal_strings(before)
    new_literals = _literal_strings(after)
    literal_delta = old_literals ^ new_literals
    patch_text = _json(patch or {})
    text_hint = (before + "\n" + after + "\n" + patch_text).lower()
    if literal_delta or any(token in text_hint for token in (
            "json", "schema", "wire", "payload", "event field", "serialized", "field=")):
        found.append("wire_field")
    if any(token in text_hint for token in ("required_role", "registration", "registry", "register(")):
        found.append("registry_member")
    if any(token in text_hint for token in ("config", "environment variable", "env key")):
        found.append("config_key")
    if any(token in text_hint for token in ("--", "cli", "command line", "option")):
        found.append("cli_option")
    old_names = _identifier_tokens(before)
    new_names = _identifier_tokens(after)
    if old_names ^ new_names:
        found.append("local_var")
    if (not {"signature", "wire_field"} & set(found)
            and _function_bodies(before) != _function_bodies(after)):
        # A body change with no more specific surface: callers keep their
        # call shape but observe new behaviour (review, not edit).
        found.append("behaviour")
    result: list[str] = []
    for dimension in found:
        dimension = _normal_dimension(dimension)
        if dimension not in result and dimension != "unknown":
            result.append(dimension)
    return tuple(result or ["unknown"])


def _patch_mapping(patch: object) -> dict[str, object]:
    if isinstance(patch, Mapping):
        return dict(patch)
    if isinstance(patch, Patch):
        return patch.to_dict()
    return {}


@dataclass(frozen=True)
class Patch:
    path: str
    old: str = ""
    new: str = ""
    kind: str = "str_replace"
    dimension: str | None = None
    function_name: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "old_str": self.old, "new_str": self.new,
                "kind": self.kind, "dimension": self.dimension,
                "function_name": self.function_name}


def _patch_parts(patch: object) -> tuple[str, str, str, str, str]:
    value = _patch_mapping(patch)
    path = _relative_target(value.get("path", value.get("target_path", "app.py")))
    kind = _text(value.get("kind", value.get("tool", "str_replace")), "str_replace").lower()
    old = next((value.get(key) for key in ("old_str", "old", "old_text", "before")
                if isinstance(value.get(key), str)), "")
    new = next((value.get(key) for key in (
        "new_str", "new", "new_text", "after", "content", "source", "function_source")
                if isinstance(value.get(key), str)), "")
    function_name = _text(value.get("function_name", value.get("name", "")))
    return path, kind, old, new, function_name


def _replace_definition(source: str, function_name: str, replacement: str) -> tuple[str, str]:
    """Mirror the host ``edit`` tool: replace ONE top-level def/class by name.

    The host dedents ``function_source`` and swaps the complete top-level
    definition (decorators included); nested definitions are never targets.
    """
    if not function_name or not replacement:
        return source, ""
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError, ValueError):
        return source, "source is not parseable"
    matches = [node for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               and node.name == function_name]
    if len(matches) != 1:
        return source, "function anchor is not unique"
    node = matches[0]
    start_line = min([node.lineno, *[item.lineno for item in node.decorator_list]])
    end_line = int(getattr(node, "end_lineno", node.lineno) or node.lineno)
    lines = source.splitlines(keepends=True)
    body = textwrap.dedent(replacement)
    if not body.endswith("\n"):
        body += "\n"
    return "".join([*lines[:start_line - 1], body, *lines[end_line:]]), ""


def definition_name(function_source: object) -> str:
    """The single top-level definition an ``edit`` payload names ("" if none)."""
    if not isinstance(function_source, str) or not function_source.strip():
        return ""
    try:
        tree = ast.parse(textwrap.dedent(function_source))
    except (SyntaxError, TypeError, ValueError):
        return ""
    if not tree.body or not isinstance(
            tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return ""
    return tree.body[0].name


def build_overlay(base_source: str, patch: object, *, base_revision: str | None = None,
                  base_sources: Mapping[str, str] | None = None) -> PatchOverlay:
    """Apply a candidate patch to an in-memory overlay only.

    Exact anchors are required.  No file is opened or written here; the live
    adapter supplies the source it captured immediately before dispatch.
    """
    path, kind, old, new, function_name = _patch_parts(patch)
    value = _patch_mapping(patch)
    source = str(base_source or "")
    error = ""
    if kind in {"file_create", "create"} and not old:
        after = new
    elif kind in {"replace_module", "write", "replace"} and not old:
        after = new
    elif function_name and not old:
        after, error = _replace_definition(source, function_name, new)
    elif old:
        count = source.count(old)
        if count != 1:
            after = source
            error = "candidate anchor is not unique"
        else:
            after = source.replace(old, new, 1)
    else:
        after = source
        error = "candidate patch has no source replacement"
    files = value.get("files")
    changed_paths = [path] if after != source else []
    if isinstance(files, Mapping):
        # Multi-file callers may provide complete before/after text.  The target
        # source remains the primary overlay while the changed path inventory is
        # retained for recall and unindexed-scope reporting.
        changed_paths = sorted({path, *[str(item).replace("\\", "/") for item in files]})
    patch_payload = {key: val for key, val in value.items() if key not in {"files"}}
    patch_hash = _digest(patch_payload)
    revision = str(base_revision) if base_revision else _source_digest(source)
    binding = _digest({"base_revision": revision, "patch_hash": patch_hash})
    return PatchOverlay(
        path=path, base_source=source, after_source=after,
        base_revision=revision, patch_hash=patch_hash, binding=binding,
        valid=not bool(error), error=error,
        changed_paths=tuple(changed_paths),
    )


def _candidate_rows(value: object, *, graph_name: str) -> tuple[Mapping[str, object], ...]:
    """Extract rows without depending on any one concurrent graph lane."""
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(item for item in value if isinstance(item, Mapping))
    if not isinstance(value, Mapping):
        for method_name in ("candidate_sources", "candidates", "candidate_set", "report", "snapshot"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    return _candidate_rows(method(), graph_name=graph_name)
                except (TypeError, ValueError, OSError):
                    return ()
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return _candidate_rows(to_dict(), graph_name=graph_name)
            except (TypeError, ValueError):
                return ()
        return ()
    if any(key in value for key in ("target", "path", "candidate_id", "id", "required_role")):
        return (value,)
    rows: list[Mapping[str, object]] = []
    keys = (
        "candidates", "candidate_set", "findings", "obligations", "links", "neighbors",
        "test_links", "co_change", "co_changes", "sites", "items", "assessments",
        "required_roles", "requirements", "registrations", "parity", "contracts",
    )
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, Mapping):
            raw = list(raw.values())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            continue
        for row in raw:
            if isinstance(row, Mapping):
                rows.append(row)
    return tuple(rows)


_WITNESS_MARK = "__ripple_witness_admitted__"


def _row_requirement(row: Mapping[str, object]) -> str:
    raw_requirement = row.get("requirement", row.get("requirement_id",
                                                       row.get("requirement_ref", "")))
    return (_text(raw_requirement) if not isinstance(raw_requirement, Mapping)
            else _text(raw_requirement.get("id", raw_requirement.get("ref", ""))))


def _mark_admitted_witnesses(rows: Iterable[tuple[str, Mapping[str, object]]],
                             overlay: PatchOverlay) -> list[tuple[str, Mapping[str, object]]]:
    """Strip any caller-supplied mark, then re-mark only validated witnesses."""
    marked: list[tuple[str, Mapping[str, object]]] = []
    for graph_name, row in rows:
        clean = {key: value for key, value in row.items() if key != _WITNESS_MARK}
        witness = clean.get("witness", clean.get("witness_admission"))
        if admit_witness(witness, requirement=_row_requirement(clean), overlay=overlay):
            clean[_WITNESS_MARK] = True
        marked.append((graph_name, clean))
    return marked


def _candidate_from_row(row: Mapping[str, object], *, graph_name: str) -> Candidate | None:
    required_role = _text(row.get("required_role", row.get("role", ""))) or None
    candidate_id = _text(row.get("candidate_id", row.get("id", "")))
    requirement = _row_requirement(row)
    registered = row.get("registered", row.get("present", row.get("exists", True)))
    missing_role = required_role is not None and registered is False
    target = _relative_target(row.get("target", row.get("path", row.get("dst",
                                         row.get("entity", row.get("entity_id", ""))))))
    if target == "unindexed target" and requirement:
        target = f"requirement:{requirement}"
    elif target == "unindexed target" and candidate_id:
        target = f"candidate:{candidate_id}"
    dimension = _normal_dimension(row.get("dimension", row.get("change_dimension",
                                           "registry_member" if required_role else "unknown")))
    witness_value = row.get("witness", row.get("witness_admission"))
    # A row can never self-promote: flags, an authority string or a
    # witness-shaped mapping stay advisory until ``admit_witness`` validates the
    # record against the preview's revision/patch binding (see preview_patch).
    witness_admitted = row.get(_WITNESS_MARK) is True
    authority = WITNESSED_OBLIGATION if witness_admitted else (
        CHECK_SUGGESTED if _text(row.get("decision", row.get("kind", ""))).lower()
        in {"check", "check_suggested", "suggested_check"} else ADVISORY)
    raw_decision = _text(row.get("decision", row.get("kind", ""))).lower()
    if missing_role and not witness_admitted:
        raw_decision = raw_decision or "contract"
    if authority == WITNESSED_OBLIGATION:
        decision = "obligation"
    elif raw_decision in {"contract", "shared_fact", "shared", "compatibility", "parity"}:
        decision = "contract"
    elif raw_decision in {"check", "check_suggested", "suggested_check", "test"}:
        decision = "check"
    elif raw_decision in {"unaffected", "no_change", "reviewed-no-change"}:
        decision = "unaffected"
    elif raw_decision in {"change", "edit"}:
        decision = "change"
    else:
        decision = "review"
    if not candidate_id:
        candidate_id = _digest({"graph": graph_name, "target": target,
                                "dimension": dimension, "decision": decision})[7:23]
    default_family = _normal_family(row.get("family", row.get("evidence_family", graph_name)))
    raw_evidence = row.get("evidence", row.get("evidence_rows", ()))
    if isinstance(raw_evidence, Mapping):
        raw_evidence = (raw_evidence,)
    if (isinstance(raw_evidence, Sequence)
            and not isinstance(raw_evidence, (str, bytes)) and raw_evidence):
        evidence = tuple(Evidence.from_value(item, default_family=default_family,
                                             default_dimension=dimension)
                         for item in raw_evidence)
    else:
        evidence = (Evidence.from_value(row, default_family=default_family,
                                        default_dimension=dimension),)
    reason = _clip(row.get("reason", row.get("detail", row.get("message", ""))))
    if not reason:
        reason = (f"required role {required_role} has no registered counterpart"
                  if missing_role else f"{default_family} evidence points at {target}")
    witness = dict(witness_value) if isinstance(witness_value, Mapping) else {}
    optional_change = bool(row.get("optional_change", row.get("optional", False)))
    return Candidate(
        candidate_id=candidate_id, target=target, dimension=dimension,
        decision=decision, reason=reason, evidence=evidence, authority=authority,
        required_role=required_role, requirement=requirement, witness=witness,
        optional_change=optional_change,
    )


def _dimension_matches(candidate: Candidate, dimensions: Sequence[str]) -> bool:
    if candidate.dimension == "call_site":
        # Signature -> call-site transfer: every signature change except a
        # verified compatible (identical-prefix, optional-only) append.
        return "signature" in dimensions and "optional_parameter" not in dimensions
    if candidate.dimension in {"unknown", "contract"} or "unknown" in dimensions:
        return True
    if candidate.dimension in dimensions:
        return True
    if candidate.dimension == "optional_parameter" and "signature" in dimensions:
        return True
    # An explicit contract with no decisive dimension is transferable; an
    # unrelated local/attribute candidate is not.
    return candidate.decision == "contract" and candidate.dimension == "contract"


def _correlation_key(evidence: Evidence) -> str:
    return evidence.correlation_id or evidence.ast_id or f"family:{evidence.family}"


def _rank_candidate(candidate: Candidate, dimensions: Sequence[str]) -> Candidate:
    effective: list[Evidence] = []
    seen_correlations: set[str] = set()
    score = 0.0
    best_priority = 5
    for evidence in candidate.evidence:
        key = _correlation_key(evidence)
        if key in seen_correlations:
            continue
        seen_correlations.add(key)
        effective.append(evidence)
        score += _FAMILY_WEIGHT.get(evidence.family, 1.0) * max(0.25, evidence.strength)
        best_priority = min(best_priority, _FAMILY_PRIORITY.get(evidence.family, 5))
    if candidate.authority == WITNESSED_OBLIGATION:
        best_priority = 0
    elif candidate.decision == "contract" and candidate.dimension in dimensions:
        best_priority = min(best_priority, 1)
    correlated = max(0, len(candidate.evidence) - len(effective))
    return Candidate(**{**candidate.__dict__, "evidence": tuple(candidate.evidence),
                        "score": score, "priority": best_priority,
                        "correlated_evidence": correlated})


def _merge_candidates(rows: Iterable[tuple[str, Mapping[str, object]]],
                      dimensions: Sequence[str], minimum_score: float) -> tuple[Candidate, ...]:
    merged: dict[tuple[str, str, str], Candidate] = {}
    for graph_name, row in rows:
        candidate = _candidate_from_row(row, graph_name=graph_name)
        if candidate is None or not _dimension_matches(candidate, dimensions):
            continue
        if candidate.optional_change and "optional_parameter" in dimensions:
            continue
        key = (candidate.target, candidate.dimension, candidate.required_role or "")
        old = merged.get(key)
        if old is None:
            merged[key] = candidate
        else:
            decision = old.decision
            if candidate.authority == WITNESSED_OBLIGATION:
                decision = "obligation"
            elif old.decision == "review" and candidate.decision != "review":
                decision = candidate.decision
            authority = (WITNESSED_OBLIGATION if
                         WITNESSED_OBLIGATION in {old.authority, candidate.authority}
                         else CHECK_SUGGESTED if CHECK_SUGGESTED in {old.authority, candidate.authority}
                         else ADVISORY)
            merged[key] = Candidate(
                candidate_id=old.candidate_id,
                target=old.target,
                dimension=old.dimension,
                decision=decision,
                reason=old.reason or candidate.reason,
                evidence=(*old.evidence, *candidate.evidence),
                authority=authority,
                required_role=old.required_role or candidate.required_role,
                requirement=old.requirement or candidate.requirement,
                witness={**dict(old.witness), **dict(candidate.witness)},
                optional_change=old.optional_change or candidate.optional_change,
            )
    ranked = [_rank_candidate(item, dimensions) for item in merged.values()]
    admitted = [item for item in ranked
                if item.authority == WITNESSED_OBLIGATION or item.score >= minimum_score]
    admitted.sort(key=lambda item: (item.priority, -item.score, item.target,
                                    item.candidate_id))
    return tuple(admitted)


def _within_where_used_scan_budget(root: object) -> bool:
    """Bound optional repository scans before the where-used pack reads files."""
    from pathlib import Path

    skipped_directories = {
        ".git", ".hg", ".svn", "__pycache__", ".pytest_cache",
        ".mypy_cache", ".ruff_cache", ".tox", ".venv", "venv", "node_modules",
        "dist", "build", "backup", "backups",
    }
    try:
        root_path = Path(root).resolve()
        folded_root_parts = tuple(part.casefold() for part in root_path.parts)
        if any(part in {"backup", "backups"} for part in folded_root_parts):
            return False
        for index, part in enumerate(folded_root_parts[:-1]):
            next_part = folded_root_parts[index + 1]
            if part == "tools" and next_part == "bug_replay":
                return False
            if part == "bin" and next_part.startswith(("operator2_", "belt_")):
                return False
        pending = [root_path]
        file_count = 0
        directory_count = 0
        byte_count = 0
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name.casefold() in skipped_directories:
                            continue
                        relative_parts = Path(entry.path).relative_to(root_path).parts
                        folded_parts = tuple(part.casefold() for part in relative_parts)
                        if (len(folded_parts) >= 2
                                and folded_parts[0] == "tools"
                                and folded_parts[1] == "bug_replay"):
                            continue
                        if (len(folded_parts) >= 2
                                and folded_parts[0] == "bin"
                                and folded_parts[1].startswith(("operator2_", "belt_"))):
                            continue
                        directory_count += 1
                        if directory_count > _MAX_WHERE_USED_SCAN_DIRECTORIES:
                            return False
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        file_count += 1
                        if file_count > _MAX_WHERE_USED_SCAN_FILES:
                            return False
                        byte_count += entry.stat(follow_symlinks=False).st_size
                        if byte_count > _MAX_WHERE_USED_SCAN_BYTES:
                            return False
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _available_graphs(explicit: Mapping[str, object] | None,
                      configured: Mapping[str, object] | None) -> tuple[dict[str, object], tuple[str, ...]]:
    sources: dict[str, object] = {}
    for value in (configured or {}, explicit or {}):
        if isinstance(value, Mapping):
            sources.update({str(key): item for key, item in value.items()})
    # No speculative provider discovery: a graph either arrives as data (from
    # the runtime adapters below or an explicit caller) or it is reported as
    # unindexed scope.  Nothing is imported for its side effects.
    missing = [name for name in _GRAPH_MODULES if name not in sources]
    return sources, tuple(missing)


def _is_test_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.startswith("tests/") or "/tests/" in path or name.startswith("test_") \
        or name.endswith("_test.py")


def _where_used_rows_uncached(root: object, path: str, before: str, after: str,
                              dimensions: Sequence[str], *, reference_index: object = None,
                              root_revision: object = None) -> dict[str, object]:
    """Adapter over the EXISTING where-used code pack (kernel_where_used).

    Runs the host's own AST pack over the candidate overlay (the patched text
    is an in-memory overlay; nothing is written) and translates reference
    sites of each changed function into Ripple rows.  Signature changes make
    callers ``call_site`` candidates; behaviour-only changes make them review
    candidates; test files become check candidates.
    """
    from pathlib import Path as _Path
    from kernel_where_used import build_code_pack

    relative = str(path).replace("\\", "/")
    names = changed_function_names(before, after)
    if not names:
        return {"candidates": [], "unindexed_scope": []}
    overlays = {relative: after}
    if reference_index is not None and callable(getattr(reference_index, "code_pack", None)):
        pack = reference_index.code_pack(_Path(str(root)), overlays=overlays,
                                         root_revision=root_revision)
        coverage = reference_index.coverage()
    else:
        pack = build_code_pack(_Path(str(root)), include_literals=False, overlays=overlays)
        coverage = {"status": "UNKNOWN", "counts": {}, "unsupported_paths": []}
    signature = "signature" in dimensions and "optional_parameter" not in dimensions
    rows: list[dict[str, object]] = []
    for name in names:
        for site in pack.find(name):
            if site.kind not in {"reference", "import"}:
                continue
            if site.path == relative:
                continue  # same-file sites are covered by the edit itself
            is_test = _is_test_path(site.path)
            dimension = ("call_site" if signature
                         else "behaviour" if "behaviour" in dimensions else "unknown")
            rows.append({
                "path": site.path, "dimension": dimension,
                "decision": "check" if is_test else "review",
                "family": "structural", "correlation_id": site.site_id,
                "reason": f"{site.kind} of {name} at line {site.start_line}: "
                          f"{_clip(site.text, 90)}",
            })
    unsupported = coverage.get("unsupported_paths", ()) if isinstance(coverage, Mapping) else ()
    unsupported = tuple(str(item) for item in unsupported) if isinstance(unsupported, Sequence) else ()
    return {"candidates": rows, "unindexed_scope": list(unsupported), "coverage": coverage}


@functools.lru_cache(maxsize=64)
def _where_used_rows_cached(root: str, path: str, before: str, after: str,
                            dimensions: tuple[str, ...], root_revision: str) -> dict[str, object]:
    return _where_used_rows_uncached(root, path, before, after, dimensions)


def where_used_rows(root: object, path: str, before: str, after: str,
                    dimensions: Sequence[str], *, root_revision: object = None,
                    reference_index: object = None) -> dict[str, object]:
    """Return where-used rows, reusing only an explicitly revision-bound repo result."""
    if root_revision in (None, "") or reference_index is not None:
        return _where_used_rows_uncached(root, path, before, after, dimensions,
                                         reference_index=reference_index,
                                         root_revision=root_revision)
    result = _where_used_rows_cached(
        str(root), str(path), str(before), str(after), tuple(map(str, dimensions)),
        str(root_revision),
    )
    return copy.deepcopy(result)


def repo_signal_rows(signals: object, dimensions: Sequence[str]) -> dict[str, object]:
    """Translate the ACTUAL kernel_repo_signals schema into Ripple rows.

    ``structural_neighbors.neighbors[]`` (static references) and
    ``history.co_change.companions[]`` are nested under their own keys; both
    are suggestion-only and therefore enter as low-weight textual/history
    evidence that can corroborate, never create, a structural candidate.
    """
    if not isinstance(signals, Mapping):
        return {"candidates": []}
    rows: list[dict[str, object]] = []
    neighbors = _safe_mapping(signals.get("structural_neighbors")).get("neighbors", ())
    signature = "signature" in dimensions and "optional_parameter" not in dimensions
    for row in neighbors if isinstance(neighbors, Sequence) else ():
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            continue
        caller = row.get("kind") == "caller"
        rows.append({
            "path": row["path"],
            "dimension": ("call_site" if signature and caller
                          else "behaviour" if "behaviour" in dimensions else "unknown"),
            "decision": "check" if _is_test_path(row["path"]) else "review",
            "family": "textual",
            "reason": f"repo signal {row.get('kind', 'reference')} of "
                      f"{', '.join(map(str, row.get('symbols', ())[:3]))}",
        })
    history = _safe_mapping(signals.get("history"))
    companions = _safe_mapping(history.get("co_change")).get("companions", ())
    for row in companions if isinstance(companions, Sequence) else ():
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            continue
        rows.append({
            "path": row["path"], "dimension": "unknown", "decision": "review",
            "family": "history",
            "reason": f"co-change support={row.get('support')} "
                      f"confidence={row.get('confidence')}",
        })
    return {"candidates": rows}


def admit_witness(witness: object, *, requirement: str, overlay: PatchOverlay) -> bool:
    """Validate an EXISTING admission record before it may create an obligation.

    The record must be a ``requirements-admission/1`` decision at GATING tier
    (kernel_check_admission.AdmissionDecision.to_dict), reference this
    candidate's requirement, name its checker and contract endpoints, and be
    bound to the exact base revision and patch hash of the current overlay.
    Anything else (empty, rejected, stale, unbound) stays advisory.
    """
    if not isinstance(witness, Mapping) or not witness or not requirement:
        return False
    if witness.get("schema") != "requirements-admission/1":
        return False
    if str(witness.get("tier", "")).lower() != "gating":
        return False
    refs = witness.get("requirement_refs", ())
    ref_ids = {str(item.get("id", item.get("ref", ""))) for item in refs
               if isinstance(item, Mapping)} if isinstance(refs, Sequence) else set()
    if requirement not in ref_ids:
        return False
    if not _text(witness.get("check_id")):
        return False
    endpoints = witness.get("contract_endpoints", ())
    if not isinstance(endpoints, Sequence) or isinstance(endpoints, (str, bytes)) \
            or not [item for item in endpoints if str(item).strip()]:
        return False
    return (witness.get("base_revision") == overlay.base_revision
            and witness.get("patch_hash") == overlay.patch_hash)


def _unindexed_from_graphs(sources: Mapping[str, object], missing: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for name, source in sources.items():
        if isinstance(source, Mapping):
            raw = source.get("unindexed_scope", source.get("unindexed", ()))
            if isinstance(raw, str):
                raw = (raw,)
            if isinstance(raw, Sequence):
                values.extend(str(item) for item in raw)
    values.extend(f"{name} graph unavailable" for name in missing)
    return tuple(dict.fromkeys(item for item in values if item))


def _shared_facts(dimensions: Sequence[str], overlay: PatchOverlay) -> tuple[str, ...]:
    facts: list[str] = []
    if "optional_parameter" in dimensions:
        facts.append("Optional parameter append preserves existing callers; caller edits are not demanded.")
    if "wire_field" in dimensions:
        facts.append("Wire-field evidence is dimension-bound; a local variable with the same spelling is not a counterpart.")
    if "registry_member" in dimensions:
        facts.append("Registration evidence is a required-role relationship, not a general edit list.")
    if not overlay.valid:
        facts.append("Candidate overlay was not admitted because its exact anchor was unavailable.")
    return tuple(facts)


def _emit(journal: object, kind: str, **fields: object) -> None:
    if journal is None:
        return
    emitter = getattr(journal, "emit", None)
    if callable(emitter):
        emitter(kind, **fields)


class RipplePreviewEngine:
    """Pure candidate merger plus optional journal/state adapter."""

    def __init__(self, selected_mode: str | None = None, *, journal: object = None,
                 graph_sources: Mapping[str, object] | None = None,
                 minimum_score: float = MIN_CANDIDATE_SCORE,
                 dispositions: DispositionStore | None = None):
        selected = mode() if selected_mode is None else str(selected_mode).strip().lower()
        if selected not in MODES:
            raise ValueError("STATEFRAME_RIPPLE_PREVIEW must be off, shadow or advisory")
        self.mode = selected
        self.journal = journal
        self.graph_sources = dict(graph_sources or {})
        self.minimum_score = float(minimum_score)
        records = getattr(journal, "records", None)
        replay = records() if callable(records) else ()
        self.dispositions = dispositions or DispositionStore.from_records(replay)
        self.current_preview: Preview | None = None
        from kernel_reference_index import ReferenceIndex
        self.reference_index = ReferenceIndex()

    def preview_patch(self, base_source: str, patch: object, *,
                      graphs: Mapping[str, object] | None = None,
                      intent: object = None, base_revision: str | None = None,
                      unindexed_scope: Sequence[str] = (),
                      max_groups: int = MAX_RENDER_GROUPS,
                      root: object = None,
                      root_revision: object = None) -> Preview:
        if self.mode == "off":
            preview = Preview(mode="off", overlay=None, engine=self, fired=False)
            self.current_preview = preview
            return preview
        overlay = build_overlay(base_source, patch, base_revision=base_revision)
        patch_map = _patch_mapping(patch)
        dimensions = infer_change_dimensions(base_source, overlay.after_source,
                                             intent=intent, patch=patch_map)
        sources, missing = _available_graphs(graphs, self.graph_sources)
        signals = sources.get("repo_signals")
        if isinstance(signals, Mapping) and (
                "structural_neighbors" in signals or "history" in signals):
            sources["repo_signals"] = repo_signal_rows(signals, dimensions)
        rows = []
        for graph_name, source in sources.items():
            rows.extend((graph_name, row) for row in _candidate_rows(source, graph_name=graph_name))
        rows = _mark_admitted_witnesses(rows, overlay)
        candidates = _merge_candidates(rows, dimensions, self.minimum_score)
        deferred_scope: list[str] = []
        if root is not None and overlay.valid and "where_used_closure" not in sources:
            # Repo-derived rows are review/check suggestions; validated gating
            # obligations must already be present in the supplied graph rows.
            should_scan = (self.mode != "advisory"
                           or any(item.is_obligation for item in candidates))
            if not should_scan:
                should_scan = _within_where_used_scan_budget(root)
                if not should_scan:
                    missing = tuple(item for item in missing if item != "where_used_closure")
                    deferred_scope.append(
                        "where_used_closure scan deferred by advisory scan budget")
            if should_scan:
                try:
                    generated = where_used_rows(
                        root, overlay.path, base_source, overlay.after_source, dimensions,
                        root_revision=root_revision or sources.get("workspace_revision"),
                        reference_index=self.reference_index)
                except (OSError, SyntaxError, ValueError, TypeError, ImportError):
                    pass  # stays reported as unindexed scope
                else:
                    sources["where_used_closure"] = generated
                    missing = tuple(item for item in missing if item != "where_used_closure")
                    generated_rows = [
                        ("where_used_closure", row)
                        for row in _candidate_rows(generated, graph_name="where_used_closure")
                    ]
                    rows.extend(_mark_admitted_witnesses(generated_rows, overlay))
                    candidates = _merge_candidates(rows, dimensions, self.minimum_score)
        obligations = tuple(item for item in candidates if item.is_obligation)
        unindexed = tuple(dict.fromkeys([*map(str, unindexed_scope),
                                          *deferred_scope,
                                          *_unindexed_from_graphs(sources, missing)]))
        self.dispositions.mark_stale(base_revision=overlay.base_revision,
                                    patch_hash=overlay.patch_hash)
        for candidate in candidates:
            self.dispositions.observe(
                candidate.candidate_id, authority=candidate.authority,
                base_revision=overlay.base_revision, patch_hash=overlay.patch_hash,
                target=candidate.target,
                requirement=candidate.requirement or candidate.required_role or "",
                witness=candidate.witness)
        preview = Preview(
            mode=self.mode, overlay=overlay, dimensions=dimensions,
            candidates=candidates, obligations=obligations,
            shared_facts=_shared_facts(dimensions, overlay),
            source_graphs=tuple(sorted(sources)), unindexed_scope=unindexed,
            max_groups=max(1, min(MAX_RENDER_GROUPS, int(max_groups))), fired=True,
            engine=self,
        )
        self.current_preview = preview
        _emit(self.journal, "ripple_preview_fired", payload_schema=SCHEMA,
              preview_version="ripple-preview/1", mode=self.mode,
              trigger="candidate-patch", base_revision=overlay.base_revision,
              patch_hash=overlay.patch_hash, binding=overlay.binding,
              dimensions=list(dimensions), candidate_count=len(candidates),
              obligation_count=len(obligations), source_graphs=list(preview.source_graphs),
              unindexed_scope=list(unindexed), overlay_valid=overlay.valid)
        self._persist_dispositions()
        return preview

    def reconcile(self, preview: Preview, after_source: str, *,
                  applied_targets: Sequence[str] = (), checks: object = None,
                  actual_patch_hash: str | None = None) -> Reconciliation:
        if self.mode == "off" or preview.overlay is None:
            return Reconciliation("disabled", "", "")
        target_set = tuple(dict.fromkeys(str(item).replace("\\", "/") for item in applied_targets))
        expected = preview.overlay.after_source
        patch_matches = actual_patch_hash is None or actual_patch_hash == preview.patch_hash
        status = "reconciled" if after_source == expected and patch_matches else "stale"
        check_passed = False
        if isinstance(checks, Mapping):
            check_passed = bool(checks.get("passed", checks.get("status") in {
                "PASS", "PASSED", "GREEN", "SUCCESS", "OK"}))
        updated: list[Disposition] = []
        actual_paths = set(target_set)
        if preview.overlay.path:
            actual_paths.add(preview.overlay.path)
        for candidate in preview.candidates:
            target_path = candidate.target.split(":", 1)[0].replace("\\", "/")
            touched = not target_path or target_path in actual_paths or any(
                target_path.endswith(path) or path.endswith(target_path)
                for path in actual_paths if path)
            record = self.dispositions.get(candidate.candidate_id)
            if record is None:
                continue
            if status == "stale":
                updated.append(self.dispositions.set_disposition(
                    candidate.candidate_id, "deferred", base_revision="never",
                    deferred_trigger="base or patch hash changed"))
                continue
            if not touched:
                # An untouched candidate was NOT reviewed by virtue of the
                # primary edit landing.  It keeps its prior disposition
                # (normally ``unresolved``) until explicit review evidence
                # arrives through ``record_disposition``.
                updated.append(record)
                continue
            disposition = "changed-verified" if check_passed else "changed-awaiting-check"
            updated.append(self.dispositions.set_disposition(
                candidate.candidate_id, disposition,
                base_revision=preview.base_revision, patch_hash=preview.patch_hash))
        recall = {}
        if self.mode == "shadow":
            predicted = {item.target.split(":", 1)[0] for item in preview.candidates
                         if item.target and item.target != "unindexed target"}
            actual = set(target_set)
            matched = predicted & actual
            recall = {
                "predicted_targets": sorted(predicted), "actual_targets": sorted(actual),
                "matched_targets": sorted(matched),
                "recall": (len(matched) / len(actual)) if actual else None,
                "unpredicted_targets": sorted(actual - predicted),
                "missed_candidates": sorted(predicted - actual),
            }
            _emit(self.journal, "ripple_preview_shadow_recall", payload_schema=SCHEMA,
                  binding=preview.binding, base_revision=preview.base_revision,
                  patch_hash=preview.patch_hash, **recall)
        reconciliation = Reconciliation(
            status=status, base_revision=preview.base_revision,
            patch_hash=preview.patch_hash, changed_targets=target_set,
            recall=recall, dispositions=tuple(updated),
        )
        _emit(self.journal, "ripple_preview_reconciled", payload_schema=SCHEMA,
              binding=preview.binding, base_revision=preview.base_revision,
              patch_hash=preview.patch_hash, status=status,
              changed_targets=list(target_set), disposition_count=len(updated),
              advisory_only=True)
        self._persist_dispositions()
        return reconciliation

    def record_disposition(self, candidate_id: str, disposition: str, *,
                           evidence: Mapping[str, object] | None = None,
                           deferred_trigger: str = "") -> Disposition:
        """Explicit disposition update; review verdicts require evidence.

        ``reviewed-no-change`` / ``verified-unaffected`` must cite what was
        inspected or checked (a receipt id and the target it covered) — the
        store never infers a completed review.
        """
        if disposition in {"reviewed-no-change", "verified-unaffected"}:
            if not isinstance(evidence, Mapping) or not _text(evidence.get("receipt_id")) \
                    or not _text(evidence.get("target")):
                raise ValueError(f"{disposition} requires review evidence "
                                 "(receipt_id and target)")
        if disposition == "deferred" and not deferred_trigger:
            raise ValueError("deferred requires a deferred_trigger")
        if self.dispositions.get(candidate_id) is None:
            raise KeyError(candidate_id)
        updated = self.dispositions.set_disposition(
            candidate_id, disposition, deferred_trigger=deferred_trigger)
        _emit(self.journal, "ripple_preview_disposition_update", payload_schema=SCHEMA,
              candidate_id=str(candidate_id), disposition=disposition,
              evidence=dict(evidence or {}), advisory_only=True)
        self._persist_dispositions()
        return updated

    def _persist_dispositions(self) -> None:
        if self.mode != "off":
            _emit(self.journal, "ripple_preview_disposition", payload_schema=SCHEMA,
                  dispositions=[item.to_dict() for item in self.dispositions.values()],
                  advisory_only=True)


def render_preview(preview: Preview) -> str:
    """Render at most four decision groups and 300 whitespace tokens."""
    if preview.mode == "off" or not preview.fired:
        return ""
    candidates = list(preview.candidates)
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        group = "obligation" if candidate.is_obligation else candidate.decision
        grouped[group].append(candidate)
    order = ("obligation", "contract", "check", "change", "review", "unaffected")
    groups = [(name, grouped[name]) for name in order if grouped.get(name)]
    groups = groups[: max(1, min(MAX_RENDER_GROUPS, preview.max_groups))]
    shown: list[Candidate] = []
    for _, values in groups:
        shown.extend(values[: max(1, MAX_RENDER_ITEMS // max(1, len(groups)))])
    # Keep the witness class visible even when the adaptive group budget would
    # otherwise consume all slots with advisory reviews.
    for candidate in candidates:
        if candidate.is_obligation and candidate not in shown:
            shown.insert(0, candidate)
    shown = shown[:MAX_RENDER_ITEMS]
    lines = ["RIPPLE PREVIEW (advisory; no completion veto)",
             f"{len(shown)} shown / {len(candidates)} candidates"]
    if preview.shared_facts:
        lines.extend(f"FACT: {_clip(fact, 150)}" for fact in preview.shared_facts[:2])
    by_group = defaultdict(list)
    for candidate in shown:
        by_group["obligation" if candidate.is_obligation else candidate.decision].append(candidate)
    for name in order:
        values = by_group.get(name)
        if not values:
            continue
        lines.append(f"GROUP {name.upper()}")
        for candidate in values:
            lines.append(
                f"- {candidate.target} [{candidate.dimension}] {_clip(candidate.reason)}; "
                + "; ".join(candidate.actions))
    scope = ", ".join(_clip(item, 80) for item in preview.unindexed_scope[:5]) \
        if preview.unindexed_scope else "none reported"
    lines.append(f"unindexed scope: {scope}")
    # Bound by the product's deliberately simple token estimator.  Preserve the
    # count and scope lines even if a very verbose graph reason filled the budget.
    while _render_tokens(lines) > MAX_RENDER_TOKENS and len(lines) > 3:
        removable = next((index for index in range(len(lines) - 1, -1, -1)
                          if lines[index].startswith("- ")), None)
        if removable is None:
            fact = next((index for index, line in enumerate(lines)
                         if line.startswith("FACT: ")), None)
            if fact is not None:
                lines.pop(fact)
                continue
            scope_index = next((index for index, line in enumerate(lines)
                                if line.startswith("unindexed scope: ")), None)
            if scope_index is not None and lines[scope_index] != "unindexed scope: truncated":
                lines[scope_index] = "unindexed scope: truncated"
                continue
            break
        lines.pop(removable)
    # Drop group headers emptied by trimming and report the TRUE shown count.
    lines = [line for index, line in enumerate(lines)
             if not line.startswith("GROUP ")
             or (index + 1 < len(lines) and lines[index + 1].startswith("- "))]
    lines[1] = f"{sum(1 for line in lines if line.startswith('- '))} shown / {len(candidates)} candidates"
    return "\n".join(lines)


def _render_tokens(lines: Sequence[str]) -> int:
    """Conservative estimate: max(whitespace words, chars/4) like the runtime cap."""
    text = "\n".join(lines)
    return max(len(text.split()), (len(text) + 3) // 4)


def project_frame(frame: Mapping[str, object], preview: Preview | None,
                  *, mode: str | None = None) -> dict[str, object]:
    """Append only the advisory presentation; OFF and SHADOW return the input."""
    selected = mode if mode is not None else (preview.mode if preview is not None else "off")
    if selected != "advisory" or preview is None or not preview.fired:
        return frame  # preserve the OFF/SHADOW object and byte shape exactly
    projected = dict(frame)
    projected["ripple_preview"] = preview.render()
    return projected


def preview_patch(base_source: str, patch: object, *, selected_mode: str | None = None,
                  graphs: Mapping[str, object] | None = None, journal: object = None,
                  **kwargs: object) -> Preview:
    engine = RipplePreviewEngine(selected_mode, journal=journal)
    return engine.preview_patch(base_source, patch, graphs=graphs, **kwargs)


def patch_from_action(kind: str, arguments: Mapping[str, object], *, path: str = "") -> Patch:
    """Translate the live edit vocabulary into the pure overlay vocabulary."""
    values = dict(arguments or {})
    target = _relative_target(values.get("path", path))
    kind = _text(kind, "str_replace")
    old = next((values.get(key) for key in ("old_str", "old", "old_text", "before")
                if isinstance(values.get(key), str)), "")
    new = next((values.get(key) for key in (
        "new_str", "new", "new_text", "after", "content", "source", "function_source")
                if isinstance(values.get(key), str)), "")
    function_name = _text(values.get("function_name", values.get("name", "")))
    if not function_name and kind == "edit" and not old:
        # The live edit tool carries only path + function_source; the host
        # derives the definition identity from the payload itself.
        function_name = definition_name(values.get("function_source"))
    return Patch(path=target, old=old, new=new, kind=kind,
                 dimension=_text(values.get("dimension"), "") or None,
                 function_name=function_name)


__all__ = [
    "ADVISORY", "CHECK_SUGGESTED", "CURRENT", "Disposition", "DispositionStore",
    "Evidence", "MAX_RENDER_GROUPS", "MIN_CANDIDATE_SCORE", "MODES", "Patch",
    "PatchOverlay", "Preview", "Reconciliation", "RipplePreviewEngine", "SCHEMA",
    "STALE", "SUPERSEDED", "WITNESSED_OBLIGATION", "build_overlay", "enabled",
    "infer_change_dimensions", "mode", "patch_from_action", "preview_patch",
    "project_frame", "render_preview",
]

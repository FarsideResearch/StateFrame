"""Revision-bound edit plans, candidate generations, and validation receipts.

This module is deliberately pure at its public boundary.  A host creates site
references from the source revision it owns; callers submit only those opaque
references and replacement text.  Candidate composition is byte-offset based,
and publication changes one generation pointer rather than a sequence of
materialized files.

The module does not decide whether an application fix is semantically correct.
It records the source/check/observation identities needed for an existing
behavior witness or language adapter to make that decision.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import os
import re
import threading
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
import kernel_workspace_truth as _workspace_truth
try:  # Python 3.9 does not expose TypeAlias in typing.
    from typing import TypeAlias, Union
except ImportError:  # pragma: no cover - exercised by the repository's 3.9 runner
    TypeAlias = object  # type: ignore[misc,assignment]
    from typing import Union


SCHEMA = "revision-bound-edit/1"
PLAN_SCHEMA = "revision-bound-edit-plan/1"
GENERATION_SCHEMA = "revision-bound-edit-generation/1"
CHECK_SCHEMA = "revision-bound-edit-check/1"
SUPPORTED_OPERATIONS = frozenset({"replace", "insert_before", "move"})
SUPPORTED_DISPOSITIONS = frozenset({"repair", "preserve"})
EDIT_SUFFIXES = frozenset({
    ".py", ".json", ".go", ".rs", ".ts", ".tsx", ".js", ".jsx", ".java",
    ".php", ".vue", ".html", ".htm",
})
BytesLike: TypeAlias = Union[bytes, bytearray, memoryview, str]


class EditPlanError(ValueError):
    """A typed admission or publication refusal."""

    def __init__(self, code: str, message: str, *, diagnostics: Sequence[str] = ()):
        self.code = code
        self.diagnostics = tuple(diagnostics) or (message,)
        super().__init__(message)


class ReceiptStatus(str, Enum):
    CREATED = "CREATED"
    STAGED = "STAGED"
    REFUSED = "REFUSED"
    PUBLISHED = "PUBLISHED"
    UNKNOWN_PUBLISHED = "UNKNOWN_PUBLISHED"
    OBSERVED_PASS = "OBSERVED_PASS"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    ENVIRONMENT = "ENVIRONMENT"
    UNRUNNABLE = "UNRUNNABLE"
    UNCOVERED = "UNCOVERED"


class _AdapterState(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    UNRUNNABLE = "UNRUNNABLE"


def _as_bytes(value: BytesLike) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError("source must be bytes or text")


def _digest(value: BytesLike) -> str:
    return "sha256:" + hashlib.sha256(_as_bytes(value)).hexdigest()


def _json_digest(value: object) -> str:
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _path(value: object) -> str:
    raw = str(value).replace("\\", "/").strip()
    if not raw or "\x00" in raw:
        raise EditPlanError("PATH", "path must be a non-empty relative path")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise EditPlanError("PATH_ESCAPE", f"path is not a canonical relative path: {raw}")
    return candidate.as_posix()


def _revision(value: object, label: str = "revision") -> str:
    text = str(value or "").strip()
    if not text:
        raise EditPlanError("REVISION", f"{label} is required")
    return text


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise EditPlanError("SCHEMA", f"{label} must be a string")
    return value


def _required_text(value: object, label: str) -> str:
    text = _text(value, label)
    if not text.strip():
        raise EditPlanError("SCHEMA", f"{label} must not be empty")
    return text


def _unique_strings(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise EditPlanError("SCHEMA", f"{label} must be a list of strings")
    result = tuple(dict.fromkeys(str(item) for item in value if str(item)))
    return result


def _relative_to_root(root: Path, relative: str) -> Path:
    target = (root / relative).resolve(strict=False)
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise EditPlanError("PATH_ESCAPE", f"path escapes authority root: {relative}") from error
    return target


@dataclass(frozen=True)
class SiteReference:
    """Host-issued address for one byte span or insertion boundary."""

    site_id: str
    path: str
    file_revision: str
    source_digest: str
    start: int
    end: int
    preimage_digest: str
    node_kind: str
    source_receipt: str
    permitted_operations: tuple[str, ...]
    authority_ref: str
    boundary: str | None = None
    display_line: int | None = None
    attached_digest: str | None = None
    dependency_digest: str | None = None

    @property
    def is_boundary(self) -> bool:
        return self.boundary is not None and self.start == self.end

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "site_id": self.site_id,
            "path": self.path,
            "file_revision": self.file_revision,
            "source_digest": self.source_digest,
            "span": {"start": self.start, "end": self.end},
            "boundary": self.boundary,
            "preimage_digest": self.preimage_digest,
            "node_kind": self.node_kind,
            "source_receipt": self.source_receipt,
            "permitted_operations": list(self.permitted_operations),
            "authority_ref": self.authority_ref,
            "display_line": self.display_line,
            "attached_digest": self.attached_digest,
            "dependency_digest": self.dependency_digest,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SiteReference":
        if value.get("schema") != SCHEMA:
            raise EditPlanError("SCHEMA", "site reference has the wrong schema")
        span = value.get("span")
        if not isinstance(span, Mapping):
            raise EditPlanError("SCHEMA", "site reference requires a span object")
        try:
            start, end = int(span["start"]), int(span["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise EditPlanError("SCHEMA", "site span requires integer start/end") from error
        boundary = value.get("boundary")
        if boundary is not None and str(boundary) not in {"before", "after"}:
            raise EditPlanError("BOUNDARY", "boundary must be before or after")
        if start < 0 or end < start:
            raise EditPlanError("SPAN", "site span must be non-negative and ordered")
        if boundary is not None and end != start:
            raise EditPlanError("BOUNDARY", "boundary references must have zero width")
        if boundary is None and end == start:
            raise EditPlanError("SPAN", "non-boundary references must have non-zero width")
        permitted = _unique_strings(value.get("permitted_operations", ()), "permitted_operations")
        if not permitted or not set(permitted) <= SUPPORTED_OPERATIONS:
            raise EditPlanError("OPERATIONS", "site has no valid permitted operations")
        return cls(
            site_id=_required_text(value.get("site_id"), "site_id"),
            path=_path(value.get("path")),
            file_revision=_revision(value.get("file_revision"), "file_revision"),
            source_digest=_revision(value.get("source_digest"), "source_digest"),
            start=start,
            end=end,
            preimage_digest=_revision(value.get("preimage_digest"), "preimage_digest"),
            node_kind=_required_text(value.get("node_kind"), "node_kind"),
            source_receipt=_required_text(value.get("source_receipt"), "source_receipt"),
            permitted_operations=permitted,
            authority_ref=_required_text(value.get("authority_ref"), "authority_ref"),
            boundary=str(boundary) if boundary is not None else None,
            display_line=(int(value["display_line"]) if value.get("display_line") is not None else None),
            attached_digest=(str(value["attached_digest"]) if value.get("attached_digest") else None),
            dependency_digest=(str(value["dependency_digest"]) if value.get("dependency_digest") else None),
        )


def bind_site(
    *,
    path: str,
    file_revision: object,
    source: BytesLike,
    start: int,
    end: int | None = None,
    boundary: str | None = None,
    node_kind: str,
    source_receipt: str,
    permitted_operations: Sequence[str],
    authority_ref: str,
    display_line: int | None = None,
    attached_digest: str | None = None,
    dependency_digest: str | None = None,
) -> SiteReference:
    """Create a host-owned span/boundary reference without storing old bytes."""
    relative = _path(path)
    raw = _as_bytes(source)
    finish = start if end is None and boundary is not None else end
    if not isinstance(start, int) or not isinstance(finish, int) or start < 0 or finish < start:
        raise EditPlanError("SPAN", "site span must be non-negative and ordered")
    if finish > len(raw) or (boundary is None and finish == start):
        raise EditPlanError("SPAN", "span is outside source or is not a boundary")
    if boundary is not None and finish != start:
        raise EditPlanError("BOUNDARY", "insertion boundary must have zero width")
    if boundary is not None and boundary not in {"before", "after"}:
        raise EditPlanError("BOUNDARY", "boundary must be before or after")
    operations = tuple(dict.fromkeys(str(item) for item in permitted_operations))
    if not operations or not set(operations) <= SUPPORTED_OPERATIONS:
        raise EditPlanError("OPERATIONS", "permitted_operations must contain known operations")
    revision = _revision(file_revision, "file_revision")
    receipt = _required_text(source_receipt, "source_receipt")
    kind = _required_text(node_kind, "node_kind")
    authority = _required_text(authority_ref, "authority_ref")
    if boundary is None:
        preimage = raw[start:finish]
    else:
        left, right = max(0, start - 32), min(len(raw), start + 32)
        preimage = raw[left:right]
    identity = {
        "path": relative, "file_revision": revision, "source_digest": _digest(raw),
        "start": start, "end": finish, "preimage_digest": _digest(preimage),
        "node_kind": kind, "source_receipt": receipt, "permitted_operations": operations,
        "authority_ref": authority, "boundary": boundary,
    }
    site_id = "site:" + _json_digest(identity)[7:31]
    return SiteReference(
        site_id=site_id,
        path=relative,
        file_revision=revision,
        source_digest=_digest(raw),
        start=start,
        end=finish,
        preimage_digest=_digest(preimage),
        node_kind=kind,
        source_receipt=receipt,
        permitted_operations=operations,
        authority_ref=authority,
        boundary=boundary,
        display_line=display_line,
        attached_digest=attached_digest,
        dependency_digest=dependency_digest,
    )


def _ref(value: object) -> SiteReference:
    if isinstance(value, SiteReference):
        return value
    if not isinstance(value, Mapping):
        raise EditPlanError("TARGET_REF", "typed edit target_ref must be an object")
    return SiteReference.from_mapping(value)


@dataclass(frozen=True)
class EditOperation:
    operation: str
    site_ref: SiteReference | None = None
    new_text: str = ""
    destination_ref: SiteReference | None = None
    legacy_old: str | None = None
    legacy_path: str | None = None
    order: int | None = None
    dependency_digest: str | None = None

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {"op": self.operation, "new": self.new_text}
        if self.site_ref is not None:
            value["site_ref"] = self.site_ref.to_dict()
        if self.destination_ref is not None:
            value["destination_ref"] = self.destination_ref.to_dict()
        if self.legacy_old is not None:
            value["old_str"] = self.legacy_old
        if self.legacy_path is not None:
            value["path"] = self.legacy_path
        if self.order is not None:
            value["order"] = self.order
        if self.dependency_digest is not None:
            value["dependency_digest"] = self.dependency_digest
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EditOperation":
        operation = str(value.get("op", value.get("operation", "replace"))).strip()
        if operation not in SUPPORTED_OPERATIONS:
            raise EditPlanError("OPERATION", f"unsupported edit operation: {operation}")
        target_value = value.get("site_ref", value.get("target_ref"))
        legacy_old = value.get("old_str", value.get("old"))
        if target_value is None:
            if not isinstance(legacy_old, str):
                raise EditPlanError("TARGET_REF", "typed operation requires site_ref/target_ref")
            path_value = value.get("path")
            if not isinstance(path_value, str) or not path_value:
                raise EditPlanError("TARGET_REF", "legacy operation requires path")
            target = None
            legacy_path = _path(path_value)
        else:
            target = _ref(target_value)
            legacy_old = None
            legacy_path = None
        destination = value.get("destination_ref", value.get("destination"))
        destination_ref = _ref(destination) if destination is not None else None
        if operation == "move" and destination_ref is None:
            raise EditPlanError("MOVE", "move requires destination_ref")
        order = value.get("order")
        if order is not None:
            try:
                order = int(order)
            except (TypeError, ValueError) as error:
                raise EditPlanError("ORDER", "order must be an integer") from error
        new_text = value.get("new", value.get("new_str", ""))
        if not isinstance(new_text, str):
            raise EditPlanError("SCHEMA", "new text must be a string")
        dependency = value.get("dependency_digest")
        return cls(operation, target, new_text, destination_ref,
                   legacy_old if isinstance(legacy_old, str) else None, legacy_path, order,
                   str(dependency) if dependency else None)


@dataclass(frozen=True)
class EditPlan:
    base_revision: str
    authority_ref: str
    inventory_ref: str
    edits: tuple[EditOperation, ...]
    dispositions: tuple[Mapping[str, object], ...] = ()
    check_refs: tuple[str, ...] = ()
    publish: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EditPlan":
        if value.get("schema", PLAN_SCHEMA) != PLAN_SCHEMA:
            raise EditPlanError("SCHEMA", "edit plan has the wrong schema")
        raw_edits = value.get("edits")
        if isinstance(raw_edits, (str, bytes)) or not isinstance(raw_edits, Sequence) or not raw_edits:
            raise EditPlanError("SCHEMA", "edit plan requires a non-empty edits list")
        edits = tuple(EditOperation.from_mapping(item) for item in raw_edits if isinstance(item, Mapping))
        if len(edits) != len(raw_edits):
            raise EditPlanError("SCHEMA", "every edit plan item must be an object")
        raw_dispositions = value.get("dispositions", ())
        if raw_dispositions is None:
            raw_dispositions = ()
        if isinstance(raw_dispositions, (str, bytes)) or not isinstance(raw_dispositions, Sequence):
            raise EditPlanError("SCHEMA", "dispositions must contain objects")
        dispositions = tuple(item for item in raw_dispositions if isinstance(item, Mapping))
        if len(dispositions) != len(raw_dispositions):
            raise EditPlanError("SCHEMA", "dispositions must contain objects")
        return cls(
            base_revision=_revision(value.get("base_revision"), "base_revision"),
            authority_ref=_required_text(value.get("authority_ref"), "authority_ref"),
            inventory_ref=_required_text(value.get("inventory_ref"), "inventory_ref"),
            edits=edits,
            dispositions=dispositions,
            check_refs=_unique_strings(value.get("check_refs", ()), "check_refs"),
            publish=bool(value.get("publish", False)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PLAN_SCHEMA,
            "base_revision": self.base_revision,
            "authority_ref": self.authority_ref,
            "inventory_ref": self.inventory_ref,
            "edits": [edit.to_dict() for edit in self.edits],
            "dispositions": [dict(item) for item in self.dispositions],
            "check_refs": list(self.check_refs),
            "publish": self.publish,
        }


@dataclass(frozen=True)
class AdmissionResult:
    accepted: bool
    diagnostics: tuple[str, ...] = ()
    operation_indexes: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"accepted": self.accepted, "diagnostics": list(self.diagnostics),
                "operation_indexes": list(self.operation_indexes)}


class EditAuthority:
    """Host-owned source/revision/authority registry for one base generation."""

    def __init__(
        self,
        *,
        root: str | Path,
        sources: Mapping[str, BytesLike],
        base_revision: object,
        authority_ref: str,
        allowed_paths: Iterable[str] | None = None,
        journal: object | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.base_revision = _revision(base_revision, "base_revision")
        self.authority_ref = _required_text(authority_ref, "authority_ref")
        self.sources = MappingProxyType({_path(key): _as_bytes(value) for key, value in sources.items()})
        self.source_revision = _tree_revision(self.sources)
        self.allowed_paths = frozenset(_path(item) for item in (allowed_paths or self.sources))
        self.journal = journal
        self.sites: dict[str, SiteReference] = {}

    def _snapshot_absence(self, path: str, *, source_kind: str = "source") -> str:
        fact = _workspace_truth.exists(self.root, path, journal=self.journal)
        return (
            f"{source_kind} bytes are absent from the supplied snapshot for {path}; "
            f"{_workspace_truth.render_fact(fact)}"
        )

    def _check_path(self, relative: str) -> None:
        relative = _path(relative)
        if relative not in self.allowed_paths:
            raise EditPlanError("OUT_OF_AUTHORITY", f"path is outside edit authority: {relative}")
        raw_target = self.root / relative
        if raw_target.is_symlink():
            raise EditPlanError("SYMLINK", f"symlink target is not editable: {relative}")
        _relative_to_root(self.root, relative)

    def bind_site(self, **kwargs: object) -> SiteReference:
        path = _path(kwargs.get("path"))
        self._check_path(path)
        source = self.sources.get(path)
        if source is None:
            raise EditPlanError("SOURCE", f"source is not present in authority: {path}")
        if str(kwargs.get("authority_ref", self.authority_ref)) != self.authority_ref:
            raise EditPlanError("AUTHORITY", "site authority does not match plan authority")
        kwargs.pop("path", None)
        kwargs.pop("authority_ref", None)
        raw_source = kwargs.pop("source", source)
        if _as_bytes(raw_source) != source:
            raise EditPlanError("STALE_SOURCE", f"source bytes do not match authority for {path}")
        ref = bind_site(file_revision=kwargs.pop("file_revision", _digest(raw_source)),
                        source=raw_source, authority_ref=self.authority_ref, path=path, **kwargs)
        self.sites[ref.site_id] = ref
        return ref

    def _validate_ref(self, ref: SiteReference, sources: Mapping[str, bytes]) -> list[str]:
        diagnostics: list[str] = []
        try:
            self._check_path(ref.path)
        except EditPlanError as error:
            diagnostics.append(str(error))
        if ref.authority_ref != self.authority_ref:
            diagnostics.append(f"AUTHORITY: site {ref.site_id} belongs to another authority")
        current = sources.get(ref.path)
        if current is None:
            if _workspace_truth.enabled():
                diagnostics.append(f"SOURCE: {self._snapshot_absence(ref.path)}")
            else:
                diagnostics.append(f"SOURCE: current source is missing for {ref.path}")
            return diagnostics
        if _digest(current) != ref.source_digest:
            diagnostics.append(f"STALE_SOURCE: {ref.path} no longer matches {ref.site_id}")
        if ref.file_revision.startswith("sha256:") and ref.file_revision != _digest(current):
            diagnostics.append(f"STALE_REVISION: file revision changed for {ref.path}")
        if ref.start < 0 or ref.end < ref.start or ref.end > len(current):
            diagnostics.append(f"STALE_SPAN: {ref.site_id} is outside current source")
        if ref.boundary is None:
            preimage = current[ref.start:ref.end]
        else:
            left, right = max(0, ref.start - 32), min(len(current), ref.start + 32)
            preimage = current[left:right]
        if _digest(preimage) != ref.preimage_digest:
            diagnostics.append(f"STALE_SPAN: preimage changed for {ref.site_id}")
        return diagnostics

    def admit(self, plan: EditPlan, sources: Mapping[str, BytesLike] | None = None) -> AdmissionResult:
        current = {_path(key): _as_bytes(value) for key, value in (sources or self.sources).items()}
        diagnostics: list[str] = []
        indexes: list[int] = []
        if plan.authority_ref != self.authority_ref:
            diagnostics.append("AUTHORITY: plan authority does not match host authority")
        if plan.base_revision != self.base_revision:
            diagnostics.append("STALE_REVISION: plan base revision is not current")
        ranges: dict[str, list[tuple[int, int, int, str]]] = defaultdict(list)
        boundaries: dict[tuple[str, int], list[tuple[int | None, int]]] = defaultdict(list)
        seen_sites: set[str] = set()
        for index, edit in enumerate(plan.edits):
            ref = edit.site_ref
            if ref is None:
                if edit.operation != "replace":
                    diagnostics.append(f"TARGET_REF[{index}]: legacy text edits only support replace")
                    indexes.append(index)
                elif edit.legacy_path is None or edit.legacy_path not in self.allowed_paths:
                    diagnostics.append(f"OUT_OF_AUTHORITY[{index}]: legacy path is outside authority")
                    indexes.append(index)
                elif edit.legacy_path not in current:
                    if _workspace_truth.enabled():
                        diagnostics.append(
                            f"SOURCE[{index}]: {self._snapshot_absence(edit.legacy_path)}"
                        )
                    else:
                        diagnostics.append(f"SOURCE[{index}]: legacy source is missing")
                    indexes.append(index)
                elif edit.legacy_old is None or current[edit.legacy_path].count(edit.legacy_old.encode()) != 1:
                    diagnostics.append(f"MATCH[{index}]: legacy old_str must match exactly once")
                    indexes.append(index)
                continue
            if ref.site_id in seen_sites:
                diagnostics.append(f"AMBIGUOUS[{index}]: site {ref.site_id} was selected twice")
                indexes.append(index)
            seen_sites.add(ref.site_id)
            if edit.operation not in ref.permitted_operations:
                diagnostics.append(f"OPERATION[{index}]: {edit.operation} is not permitted at {ref.site_id}")
                indexes.append(index)
            for message in self._validate_ref(ref, current):
                diagnostics.append(f"{message}[{index}]")
                indexes.append(index)
            if edit.operation == "move":
                destination = edit.destination_ref
                if destination is None:
                    diagnostics.append(f"MOVE[{index}]: destination is missing")
                    indexes.append(index)
                else:
                    for message in self._validate_ref(destination, current):
                        diagnostics.append(f"{message}[{index}]")
                        indexes.append(index)
                    if destination.boundary is None or not destination.is_boundary:
                        diagnostics.append(f"MOVE[{index}]: destination must be an insertion boundary")
                        indexes.append(index)
                    if edit.dependency_digest and destination.dependency_digest != edit.dependency_digest:
                        diagnostics.append(f"DEPENDENCY[{index}]: move dependency closure is stale")
                        indexes.append(index)
                    if ref.attached_digest and destination.attached_digest != ref.attached_digest:
                        diagnostics.append(f"ATTACHED[{index}]: move would detach comments/decorators")
                        indexes.append(index)
                    boundaries[(destination.path, destination.start)].append((edit.order, index))
                ranges[ref.path].append((ref.start, ref.end, index, "move"))
            elif ref.is_boundary:
                boundaries[(ref.path, ref.start)].append((edit.order, index))
            else:
                ranges[ref.path].append((ref.start, ref.end, index, edit.operation))
        for path, path_ranges in ranges.items():
            for left_index, left in enumerate(sorted(path_ranges)):
                for right in sorted(path_ranges)[left_index + 1:]:
                    if max(left[0], right[0]) < min(left[1], right[1]):
                        diagnostics.append(f"OVERLAP: operations {left[2]} and {right[2]} overlap in {path}")
                        indexes.extend((left[2], right[2]))
        for (path, offset), entries in boundaries.items():
            if len(entries) > 1:
                orders = [order for order, _ in entries]
                if any(order is None for order in orders) or len(set(orders)) != len(orders):
                    diagnostics.append(f"AMBIGUOUS_ORDER: insertion boundary {path}:{offset} needs unique order")
                    indexes.extend(index for _, index in entries)
            for start, end, range_index, _operation in ranges.get(path, ()):
                if start < offset < end:
                    diagnostics.append(f"OVERLAP: insertion boundary {path}:{offset} is inside operation {range_index}")
                    indexes.extend(index for _, index in entries)
                    indexes.append(range_index)
        return AdmissionResult(not diagnostics, tuple(dict.fromkeys(diagnostics)), tuple(sorted(set(indexes))))


def _legacy_apply(source: bytes, old: str, new: str) -> bytes:
    old_bytes = old.encode("utf-8")
    count = source.count(old_bytes)
    if count != 1:
        raise EditPlanError("MATCH", f"legacy old_str matches {count} places; expected exactly one")
    return source.replace(old_bytes, new.encode("utf-8"), 1)


def compose_candidate(authority: EditAuthority, plan: EditPlan) -> Mapping[str, bytes]:
    """Compose every operation against base offsets, then return final bytes."""
    admission = authority.admit(plan)
    if not admission.accepted:
        raise EditPlanError("ADMISSION", "edit plan refused before mutation", diagnostics=admission.diagnostics)
    replacements: dict[str, list[tuple[int, int, int, bytes]]] = defaultdict(list)
    candidates = dict(authority.sources)
    for index, edit in enumerate(plan.edits):
        ref = edit.site_ref
        if ref is None:
            path_value = edit.legacy_path or ""
            if not path_value:
                raise EditPlanError("TARGET_REF", f"legacy edit {index} lost its path")
            current = authority.sources.get(path_value)
            if current is None:
                message = (authority._snapshot_absence(path_value)
                           if _workspace_truth.enabled()
                           else f"legacy source is missing: {path_value}")
                raise EditPlanError("SOURCE", message)
            if edit.legacy_old is None:
                message = ("legacy old_str is absent" if _workspace_truth.enabled()
                           else f"legacy source is missing: {path_value}")
                raise EditPlanError("SOURCE", message)
            candidates[path_value] = _legacy_apply(current, edit.legacy_old, edit.new_text)
            continue
        source = authority.sources[ref.path]
        if edit.operation == "replace":
            replacements[ref.path].append((ref.start, ref.end, edit.order or index, edit.new_text.encode()))
        elif edit.operation == "insert_before":
            replacements[ref.path].append((ref.start, ref.start, edit.order or index, edit.new_text.encode()))
        elif edit.operation == "move":
            destination = edit.destination_ref
            if destination is None:
                raise EditPlanError("MOVE", "move destination is missing")
            moved = source[ref.start:ref.end]
            replacements[ref.path].append((ref.start, ref.end, edit.order or index, b""))
            replacements[destination.path].append((destination.start, destination.start,
                                                    edit.order or index, moved))
    for path, edits in replacements.items():
        current = authority.sources.get(path)
        if current is None:
            message = (authority._snapshot_absence(path, source_kind="candidate source")
                       if _workspace_truth.enabled()
                       else f"candidate source is missing: {path}")
            raise EditPlanError("SOURCE", message)
        result = current
        for start, end, order, replacement in sorted(edits, key=lambda item: (item[0], item[1], item[2]), reverse=True):
            result = result[:start] + replacement + result[end:]
        candidates[path] = result
    return MappingProxyType(candidates)


@dataclass(frozen=True)
class CandidateGeneration:
    generation_id: str
    base_revision: str
    candidate_revision: str
    sources: Mapping[str, bytes]
    diagnostics: tuple[str, ...] = ()
    status: ReceiptStatus = ReceiptStatus.STAGED
    plan: EditPlan | None = None

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema": GENERATION_SCHEMA,
            "generation_id": self.generation_id,
            "base_revision": self.base_revision,
            "candidate_revision": self.candidate_revision,
            "status": self.status.value,
            "sources": {path: _digest(source) for path, source in sorted(self.sources.items())},
            "diagnostics": list(self.diagnostics),
            "plan": self.plan.to_dict() if self.plan else None,
        }


def _tree_revision(sources: Mapping[str, bytes]) -> str:
    return _json_digest([(path, _digest(source)) for path, source in sorted(sources.items())])


class GenerationStore:
    """Immutable generations plus one writer-locked active pointer."""

    def __init__(self, storage_root: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._generations: dict[str, CandidateGeneration] = {}
        self.storage_root = Path(storage_root).resolve() if storage_root is not None else None
        if self.storage_root is not None:
            (self.storage_root / "generations").mkdir(parents=True, exist_ok=True)
        self._active: str | None = self._read_active()

    def _read_active(self) -> str | None:
        if self.storage_root is None:
            return None
        try:
            value = (self.storage_root / "active.ptr").read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None
        return value or None

    def _generation_root(self, generation_id: str) -> Path:
        if self.storage_root is None:
            raise EditPlanError("STORAGE", "generation store has no storage root")
        safe_id = re.sub(r"[^A-Za-z0-9_.:-]", "_", generation_id)
        return self.storage_root / "generations" / safe_id

    def _persist(self, generation: CandidateGeneration) -> None:
        if self.storage_root is None:
            return
        generation_root = self._generation_root(generation.generation_id)
        if (generation_root / "manifest.json").exists():
            return
        candidate_root = generation_root / "candidate"
        candidate_root.mkdir(parents=True, exist_ok=True)
        for relative, source in generation.sources.items():
            relative_path = _path(relative)
            target = candidate_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.candidate")
            with temporary.open("wb") as handle:
                handle.write(source)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        manifest = generation.to_manifest()
        temporary_manifest = generation_root / "manifest.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        with temporary_manifest.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_manifest, generation_root / "manifest.json")

    def _load(self, generation_id: str) -> CandidateGeneration | None:
        if self.storage_root is None:
            return None
        root = self._generation_root(generation_id)
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            sources = {
                _path(relative): (root / "candidate" / _path(relative)).read_bytes()
                for relative in manifest["sources"]
            }
            status = ReceiptStatus(str(manifest["status"]))
        except (KeyError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return None
        return CandidateGeneration(
            generation_id=str(manifest["generation_id"]),
            base_revision=str(manifest["base_revision"]),
            candidate_revision=str(manifest["candidate_revision"]),
            sources=MappingProxyType(sources),
            diagnostics=tuple(str(item) for item in manifest.get("diagnostics", ())),
            status=status,
        )

    @property
    def active_generation(self) -> str | None:
        with self._lock:
            if self._active is None:
                self._active = self._read_active()
            return self._active

    def get(self, generation_id: str) -> CandidateGeneration | None:
        with self._lock:
            generation = self._generations.get(generation_id)
            if generation is not None:
                return generation
            generation = self._load(generation_id)
            if generation is not None:
                self._generations[generation_id] = generation
            return generation

    def retain(self, generation: CandidateGeneration) -> None:
        with self._lock:
            if generation.generation_id in self._generations:
                return
            self._persist(generation)
            self._generations[generation.generation_id] = generation

    def publish(self, generation: CandidateGeneration, *, expected_active: str | None = None) -> None:
        with self._lock:
            if expected_active is not None and self._active != expected_active:
                raise EditPlanError("CONCURRENT_PUBLICATION", "active generation changed")
            self.retain(generation)
            if self.storage_root is not None:
                pointer = self.storage_root / "active.ptr"
                temporary = self.storage_root / "active.ptr.tmp"
                temporary.write_text(generation.generation_id, encoding="utf-8")
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, pointer)
            self._active = generation.generation_id


@dataclass(frozen=True)
class PublicationReceipt:
    status: ReceiptStatus
    generation_id: str
    base_revision: str
    published: bool
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"schema": GENERATION_SCHEMA, "status": self.status.value,
                "generation_id": self.generation_id, "base_revision": self.base_revision,
                "published": self.published, "diagnostics": list(self.diagnostics)}


class CandidateTransaction:
    """Prepare, retain, and publish one complete candidate overlay."""

    def __init__(self, authority: EditAuthority, *, store: GenerationStore | None = None) -> None:
        self.authority = authority
        self.store = store or GenerationStore()
        self.staged_id: str | None = None
        self.expected_active: str | None = self.store.active_generation

    def prepare(
        self,
        plan: EditPlan,
        *,
        validator: Callable[[Mapping[str, bytes]], Iterable[str]] | None = None,
    ) -> CandidateGeneration:
        diagnostics: list[str] = []
        candidate = dict(self.authority.sources)
        admission = self.authority.admit(plan)
        diagnostics.extend(admission.diagnostics)
        if admission.accepted:
            try:
                candidate = dict(compose_candidate(self.authority, plan))
            except EditPlanError as error:
                diagnostics.extend(error.diagnostics)
        if not diagnostics and validator is not None:
            diagnostics.extend(str(item) for item in validator(MappingProxyType(candidate)) if str(item))
        candidate_revision = _tree_revision(candidate)
        identity = {"base": self.authority.base_revision, "candidate": candidate_revision,
                    "plan": plan.to_dict()}
        generation_id = "generation:" + _json_digest(identity)[7:31]
        generation = CandidateGeneration(
            generation_id=generation_id,
            base_revision=self.authority.base_revision,
            candidate_revision=candidate_revision,
            sources=MappingProxyType(candidate),
            diagnostics=tuple(dict.fromkeys(diagnostics)),
            status=ReceiptStatus.STAGED if not diagnostics else ReceiptStatus.REFUSED,
            plan=plan,
        )
        self.store.retain(generation)
        self.staged_id = generation_id
        return generation

    def manifest(self) -> dict[str, object]:
        if self.staged_id is None:
            raise EditPlanError("STAGE", "no candidate is staged")
        generation = self.store.get(self.staged_id)
        if generation is None:
            raise EditPlanError("STAGE", "staged candidate is unavailable")
        return generation.to_manifest()

    def resume(self, manifest: Mapping[str, object]) -> CandidateGeneration:
        if manifest.get("schema") != GENERATION_SCHEMA:
            raise EditPlanError("SCHEMA", "staged generation has the wrong schema")
        if manifest.get("plan") is None or not isinstance(manifest["plan"], Mapping):
            raise EditPlanError("STAGE", "staged generation did not retain its edit plan")
        plan = EditPlan.from_mapping(manifest["plan"])
        generation = self.prepare(plan)
        expected = str(manifest.get("generation_id", ""))
        if expected and expected != generation.generation_id:
            raise EditPlanError("STALE_STAGE", "resumed plan identity differs from staged generation")
        return generation

    def publish(
        self,
        *,
        current_sources: Mapping[str, BytesLike],
        expected_active: str | None = None,
        crash_at: str | None = None,
    ) -> PublicationReceipt:
        if self.staged_id is None:
            raise EditPlanError("STAGE", "no candidate is staged")
        generation = self.store.get(self.staged_id)
        if generation is None:
            raise EditPlanError("STAGE", "staged candidate is unavailable")
        if generation.status != ReceiptStatus.STAGED:
            return PublicationReceipt(ReceiptStatus.REFUSED, generation.generation_id,
                                      generation.base_revision, False, generation.diagnostics)
        current = {_path(path): _as_bytes(source) for path, source in current_sources.items()}
        if _tree_revision(current) != self.authority.source_revision:
            return PublicationReceipt(ReceiptStatus.REFUSED, generation.generation_id,
                                      generation.base_revision, False,
                                      ("CONCURRENT_CHANGE: base generation changed",))
        if crash_at == "before_publish":
            return PublicationReceipt(ReceiptStatus.REFUSED, generation.generation_id,
                                      generation.base_revision, False, ("CRASH_BEFORE_PUBLISH",))
        try:
            self.store.publish(generation, expected_active=(
                self.expected_active if expected_active is None else expected_active))
        except EditPlanError as error:
            return PublicationReceipt(ReceiptStatus.REFUSED, generation.generation_id,
                                      generation.base_revision, False, error.diagnostics)
        if crash_at == "after_publish":
            return PublicationReceipt(ReceiptStatus.UNKNOWN_PUBLISHED, generation.generation_id,
                                      generation.base_revision, True, ("CRASH_AFTER_PUBLISH",))
        return PublicationReceipt(ReceiptStatus.PUBLISHED, generation.generation_id,
                                  generation.base_revision, True)


@dataclass(frozen=True)
class CheckReceipt:
    check_id: str
    candidate_id: str
    creation: ReceiptStatus
    source_reached: bool
    parent_revision: str
    candidate_revision: str
    observation_receipt: str
    parent_observation: Mapping[str, object] | None = None
    candidate_observation: Mapping[str, object] | None = None
    execution_failure: str | None = None
    disposition: str = ""
    outcome: ReceiptStatus = ReceiptStatus.UNCOVERED

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": CHECK_SCHEMA, "check_id": self.check_id, "candidate_id": self.candidate_id,
            "creation": self.creation.value, "source_reached": self.source_reached,
            "parent_revision": self.parent_revision, "candidate_revision": self.candidate_revision,
            "observation_receipt": self.observation_receipt,
            "parent_observation": dict(self.parent_observation) if self.parent_observation else None,
            "candidate_observation": dict(self.candidate_observation) if self.candidate_observation else None,
            "execution_failure": self.execution_failure, "disposition": self.disposition,
            "outcome": self.outcome.value,
        }


def create_check_receipt(*, check_id: str, candidate: CandidateGeneration,
                         parent_revision: object, observation_receipt: str) -> CheckReceipt:
    return CheckReceipt(
        check_id=_text(check_id, "check_id"), candidate_id=candidate.generation_id,
        creation=ReceiptStatus.CREATED, source_reached=False,
        parent_revision=_revision(parent_revision, "parent_revision"),
        candidate_revision=candidate.candidate_revision,
        observation_receipt=_text(observation_receipt, "observation_receipt"),
    )


def _resolve_runner_value(value: object) -> object:
    """Resolve one adapter awaitable without silently treating it as evidence."""
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)

    result: list[object] = []
    failure: list[BaseException] = []

    def resolve_in_thread() -> None:
        try:
            result.append(asyncio.run(value))
        except BaseException as error:  # noqa: BLE001 - preserve adapter failure in receipt
            failure.append(error)

    worker = threading.Thread(target=resolve_in_thread, daemon=True)
    worker.start()
    worker.join()
    if failure:
        raise failure[0]
    if not result:
        raise RuntimeError("awaitable returned no result")
    return result[0]


def run_behavior_check(
    receipt: CheckReceipt,
    *,
    candidate: CandidateGeneration,
    entrypoint: str,
    parent_runner: Callable[[str], Mapping[str, object] | Awaitable[Mapping[str, object]]],
    candidate_runner: Callable[[str], Mapping[str, object] | Awaitable[Mapping[str, object]]],
    source_reachable: bool,
    disposition: str,
) -> CheckReceipt:
    """Run parent and candidate through the actual caller supplied by the host."""
    if receipt.candidate_id != candidate.generation_id:
        return CheckReceipt(**{**receipt.__dict__, "outcome": ReceiptStatus.UNCOVERED,
                               "execution_failure": "candidate identity mismatch"})
    if disposition not in SUPPORTED_DISPOSITIONS:
        return CheckReceipt(**{**receipt.__dict__, "outcome": ReceiptStatus.UNCOVERED,
                               "execution_failure": "disposition must be repair or preserve"})
    if not source_reachable or not entrypoint.strip():
        return CheckReceipt(**{**receipt.__dict__, "disposition": disposition,
                               "outcome": ReceiptStatus.UNCOVERED})
    try:
        parent_value = _resolve_runner_value(parent_runner(entrypoint))
        observed_value = _resolve_runner_value(candidate_runner(entrypoint))
        if not isinstance(parent_value, Mapping) or not isinstance(observed_value, Mapping):
            raise TypeError("behavior adapter must return a mapping")
        parent = dict(parent_value)
        observed = dict(observed_value)
    except Exception as error:  # noqa: BLE001 - receipt must preserve execution failure
        return CheckReceipt(**{**receipt.__dict__, "source_reached": True,
                               "disposition": disposition,
                               "execution_failure": type(error).__name__,
                               "outcome": ReceiptStatus.ENVIRONMENT})
    if not bool(parent.get("reached", True)) or not bool(observed.get("reached", True)):
        return CheckReceipt(**{**receipt.__dict__, "source_reached": True,
                               "parent_observation": parent, "candidate_observation": observed,
                               "disposition": disposition, "outcome": ReceiptStatus.UNCOVERED})
    if "expected" not in observed or "value" not in observed or observed.get("value") is None:
        return CheckReceipt(**{**receipt.__dict__, "source_reached": True,
                               "parent_observation": parent, "candidate_observation": observed,
                               "disposition": disposition,
                               "execution_failure": "NO_STATE_OBSERVATION",
                               "outcome": ReceiptStatus.UNCOVERED})
    expected = observed["expected"]
    actual = observed["value"]
    outcome = ReceiptStatus.OBSERVED_PASS if actual == expected else ReceiptStatus.COUNTEREXAMPLE
    return CheckReceipt(**{**receipt.__dict__, "source_reached": True,
                           "parent_observation": parent, "candidate_observation": observed,
                           "disposition": disposition, "outcome": outcome})


@dataclass(frozen=True)
class ValidationResult:
    language: str
    parser: str
    syntax_status: str
    semantic_status: str
    diagnostics: tuple[str, ...] = ()
    outline: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"language": self.language, "parser": self.parser,
                "syntax_status": self.syntax_status, "semantic_status": self.semantic_status,
                "diagnostics": list(self.diagnostics), "outline": dict(self.outline)}


def _language(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python", ".go": "go", ".rs": "rust", ".ts": "typescript",
        ".tsx": "tsx", ".js": "javascript", ".jsx": "jsx", ".java": "java",
        ".php": "php", ".vue": "vue", ".json": "json",
    }.get(suffix, "unknown")


def _outline(source: str, path: str) -> dict[str, object]:
    try:
        from mc_repo_intel_vendor.ts_dispatch import outline_source
        value = outline_source(source, path)
    except Exception as error:  # noqa: BLE001 - unavailable parser is typed below
        return {"parser": "unavailable", "degraded": True, "error": type(error).__name__}
    return value if isinstance(value, Mapping) else {"parser": "unavailable", "degraded": True}


def validate_source(path: str, source: BytesLike, *, baseline: BytesLike | None = None) -> ValidationResult:
    """Return syntax/parser facts; semantic behavior remains UNKNOWN here."""
    relative = _path(path)
    text = _as_bytes(source).decode("utf-8", errors="replace")
    language = _language(relative)
    if language == "unknown":
        return ValidationResult(language, "unavailable", _AdapterState.UNRUNNABLE.value,
                                _AdapterState.UNRUNNABLE.value, ("no language adapter",))
    if language == "python":
        try:
            ast.parse(text, filename=relative)
        except SyntaxError as error:
            return ValidationResult(language, "python-ast", _AdapterState.FAIL.value,
                                    _AdapterState.UNKNOWN.value, (str(error),))
        return ValidationResult(language, "python-ast", _AdapterState.PASS.value,
                                _AdapterState.UNKNOWN.value)
    if language == "json":
        try:
            json.loads(text)
        except json.JSONDecodeError as error:
            return ValidationResult(language, "json", _AdapterState.FAIL.value,
                                    _AdapterState.UNKNOWN.value, (str(error),))
        return ValidationResult(language, "json", _AdapterState.PASS.value,
                                _AdapterState.UNKNOWN.value)
    outline = _outline(text, relative)
    parser = str(outline.get("parser", "unavailable"))
    if parser == "unavailable":
        return ValidationResult(language, parser, _AdapterState.UNRUNNABLE.value,
                                _AdapterState.UNRUNNABLE.value, ("parser unavailable",), outline)
    if language == "vue" and baseline is not None:
        old_tags = tuple(re.findall(r"<(template|script|style)(?:\s[^>]*)?>", _as_bytes(baseline).decode("utf-8", errors="replace")))
        new_tags = tuple(re.findall(r"<(template|script|style)(?:\s[^>]*)?>", text))
        if old_tags != new_tags:
            return ValidationResult(language, parser, _AdapterState.FAIL.value,
                                    _AdapterState.UNKNOWN.value, ("SFC block boundaries changed",), outline)
    if bool(outline.get("degraded")) or parser != "tree-sitter":
        return ValidationResult(language, parser, _AdapterState.UNKNOWN.value,
                                _AdapterState.UNKNOWN.value, (str(outline.get("degraded_reason", "degraded outline")),), outline)
    return ValidationResult(language, parser, _AdapterState.PASS.value,
                            _AdapterState.UNKNOWN.value, (), outline)


def discover_editable_files(root: str | Path) -> tuple[str, ...]:
    base = Path(root).resolve()
    paths: list[str] = []
    for candidate in base.rglob("*"):
        if not candidate.is_file() or candidate.is_symlink() or candidate.suffix.lower() not in EDIT_SUFFIXES:
            continue
        paths.append(candidate.relative_to(base).as_posix())
    return tuple(sorted(paths))


def entrypoint_reachable(source: BytesLike, path: str, entrypoint: str) -> bool:
    """Cheap source reach precondition; execution still owns actual reach."""
    result = validate_source(path, source)
    outline = result.outline
    wanted = entrypoint.rsplit(":", 1)[-1].rsplit(".", 1)[-1]
    functions = outline.get("functions", ()) if isinstance(outline, Mapping) else ()
    if isinstance(functions, Sequence) and any(isinstance(item, Mapping) and item.get("name") == wanted for item in functions):
        return True
    if result.language == "python":
        return re.search(rf"\b(?:def|class)\s+{re.escape(wanted)}\b", _as_bytes(source).decode("utf-8", errors="replace")) is not None
    return False


def insertion_affordance(
    *,
    path: str,
    source: BytesLike,
    symbol: str,
    source_receipt: str,
    authority_ref: str,
    allowed_paths: Iterable[str],
    protected_paths: Iterable[str] = (),
    file_revision: object | None = None,
) -> dict[str, object]:
    """Offer one AST-verified module-end insertion, never a fuzzy replacement."""
    relative = _path(path)
    allowed = frozenset(_path(item) for item in allowed_paths)
    protected = frozenset(_path(item) for item in protected_paths)
    authority = _text(authority_ref, "authority_ref")
    source_receipt_value = _text(source_receipt, "source_receipt")
    if not authority or not source_receipt_value:
        raise EditPlanError("AUTHORITY", "insertion requires source and write authority receipts")
    if relative not in allowed:
        raise EditPlanError("OUT_OF_AUTHORITY", f"path is outside insertion authority: {relative}")
    if relative in protected:
        raise EditPlanError("PROTECTED", f"protected path is not editable: {relative}")
    if not re.fullmatch(r"[A-Za-z_]\w*", symbol):
        raise EditPlanError("AMBIGUOUS_TARGET", "insertion target must be one unqualified symbol")
    raw = _as_bytes(source)
    if Path(relative).suffix.lower() != ".py":
        return {
            "schema": SCHEMA,
            "status": "UNSUPPORTED",
            "code": "UNSUPPORTED_LANGUAGE",
            "path": relative,
            "symbol": symbol,
            "write_authority": {"authority_ref": authority, "allowed_paths": [relative]},
        }
    try:
        tree = ast.parse(raw.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError) as error:
        raise EditPlanError("SOURCE_INVALID", f"cannot verify insertion boundary: {type(error).__name__}") from error
    definitions = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if symbol in definitions:
        raise EditPlanError("DUPLICATE_DEFINITION", f"definition already exists: {symbol}")
    revision = _digest(raw) if file_revision is None else _revision(file_revision, "file_revision")
    site = bind_site(
        path=relative,
        file_revision=revision,
        source=raw,
        start=len(raw),
        end=len(raw),
        boundary="after",
        node_kind="module-end",
        source_receipt=source_receipt_value,
        permitted_operations=("insert_before",),
        authority_ref=authority,
        display_line=raw.count(b"\n") + 1,
    )
    return {
        "schema": SCHEMA,
        "status": "AVAILABLE",
        "operation": "insert_before",
        "path": relative,
        "symbol": symbol,
        "target_ref": site.to_dict(),
        "write_authority": {"authority_ref": authority, "allowed_paths": [relative]},
    }


def site_ref_property() -> dict[str, object]:
    """JSON-schema property for a host-issued typed reference."""
    return {
        "type": "object",
        "properties": {
            "schema": {"const": SCHEMA},
            "site_id": {"type": "string", "minLength": 1},
            "path": {"type": "string", "minLength": 1},
            "file_revision": {"type": "string", "minLength": 1},
            "source_digest": {"type": "string", "minLength": 1},
            "span": {"type": "object", "properties": {
                "start": {"type": "integer", "minimum": 0},
                "end": {"type": "integer", "minimum": 0},
            }, "required": ["start", "end"], "additionalProperties": False},
            "boundary": {"enum": ["before", "after"]},
            "preimage_digest": {"type": "string", "minLength": 1},
            "node_kind": {"type": "string", "minLength": 1},
            "source_receipt": {"type": "string", "minLength": 1},
            "permitted_operations": {"type": "array", "items": {"enum": sorted(SUPPORTED_OPERATIONS)}, "minItems": 1},
            "authority_ref": {"type": "string", "minLength": 1},
            "display_line": {"type": "integer", "minimum": 1},
            "attached_digest": {"type": "string"},
            "dependency_digest": {"type": "string"},
        },
        "required": ["schema", "site_id", "path", "file_revision", "source_digest",
                      "span", "preimage_digest", "node_kind", "source_receipt",
                      "permitted_operations", "authority_ref"],
        "additionalProperties": False,
    }


__all__ = [
    "CHECK_SCHEMA", "CandidateGeneration", "CandidateTransaction", "CheckReceipt",
    "EditAuthority", "EditOperation", "EditPlan", "EditPlanError", "GenerationStore",
    "PLAN_SCHEMA", "PublicationReceipt", "ReceiptStatus", "SiteReference", "ValidationResult",
    "bind_site", "compose_candidate", "create_check_receipt", "discover_editable_files",
    "entrypoint_reachable", "insertion_affordance", "run_behavior_check", "site_ref_property",
    "validate_source",
]

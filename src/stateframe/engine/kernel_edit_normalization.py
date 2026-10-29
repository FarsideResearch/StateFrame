"""Mechanical, revision-bound normalization for faithful source edits.

Normalization has one job: represent exactly what an authorized payload changes.
It does not repair semantics, choose a better hunk, format source, or infer a
missing definition.  Publication remains the caller's responsibility and must
continue through its existing authority, syntax, scope, destructive, and effect
guards.
"""
from __future__ import annotations

import ast
import base64
import difflib
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


NORMALIZATION_SCHEMA = "edit-normalization/1"
_DEFINITION_KINDS = frozenset({"class", "function", "method", "type", "namespace", "sfc-block"})
_MISSING = object()


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _as_bytes(value: object, *, label: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError(f"{label} must be bytes or UTF-8 text")


def _path(value: object) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    candidate = Path(raw)
    if not raw or candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("path must be a canonical relative path")
    return candidate.as_posix()


def _mapping_value(value: object, *names: str, default: object = _MISSING) -> object:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if not isinstance(value, Mapping) and hasattr(value, name):
            return getattr(value, name)
    if default is not _MISSING:
        return default
    raise KeyError(names[0])


def _normal_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


@dataclass(frozen=True)
class EditHunk:
    """One exact base-byte replacement, including intentional deletions."""

    start: int
    end: int
    old_bytes: bytes
    replacement: bytes

    @property
    def old_digest(self) -> str:
        return _digest(self.old_bytes)

    @property
    def replacement_digest(self) -> str:
        return _digest(self.replacement)

    def to_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "old_digest": self.old_digest,
            "replacement_digest": self.replacement_digest,
            "old_base64": base64.b64encode(self.old_bytes).decode("ascii"),
            "replacement_base64": base64.b64encode(self.replacement).decode("ascii"),
        }


@dataclass(frozen=True)
class ChangedSpan:
    start: int
    end: int

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class CurrentSourceSegment:
    start: int
    end: int
    source: bytes
    site: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "start": self.start,
            "end": self.end,
            "digest": _digest(self.source),
            "source_base64": base64.b64encode(self.source).decode("ascii"),
        }
        if self.site is not None:
            value["site"] = dict(self.site)
        return value


@dataclass(frozen=True)
class NormalizedEdit:
    mode: str
    path: str
    expected_base_digest: str
    candidate_digest: str
    hunks: tuple[EditHunk, ...]
    changed_spans: tuple[ChangedSpan, ...]
    preservation_proof: Mapping[str, object]
    current_source_segments: tuple[CurrentSourceSegment, ...]
    candidate_bytes: bytes
    base_bytes: bytes
    target_site: Mapping[str, object] | None = None

    @property
    def candidate(self) -> bytes:
        return self.candidate_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": NORMALIZATION_SCHEMA,
            "status": "NORMALIZED",
            "mode": self.mode,
            "path": self.path,
            "expected_base_digest": self.expected_base_digest,
            "candidate_digest": self.candidate_digest,
            "hunks": [hunk.to_dict() for hunk in self.hunks],
            "changed_spans": [span.to_dict() for span in self.changed_spans],
            "preservation_proof": dict(self.preservation_proof),
            "current_source_segments": [segment.to_dict() for segment in self.current_source_segments],
            "candidate_base64": base64.b64encode(self.candidate_bytes).decode("ascii"),
            "target_site": dict(self.target_site) if self.target_site is not None else None,
        }


@dataclass(frozen=True)
class AmbiguousScope:
    reason: str
    candidate_sites: tuple[Mapping[str, object], ...]
    current_source_segments: tuple[CurrentSourceSegment, ...]
    path: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": NORMALIZATION_SCHEMA,
            "status": "AMBIGUOUS_SCOPE",
            "reason": self.reason,
            "path": self.path,
            "candidate_sites": [dict(site) for site in self.candidate_sites],
            "current_source_segments": [segment.to_dict() for segment in self.current_source_segments],
        }


@dataclass(frozen=True)
class Rejected:
    reason: str
    current_source_segments: tuple[CurrentSourceSegment, ...] = ()
    path: str = ""
    expected_base_digest: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": NORMALIZATION_SCHEMA,
            "status": "REJECTED",
            "reason": self.reason,
            "path": self.path,
            "expected_base_digest": self.expected_base_digest,
            "current_source_segments": [segment.to_dict() for segment in self.current_source_segments],
        }


NormalizationResult = NormalizedEdit | AmbiguousScope | Rejected


class _ParserUnavailable(ValueError):
    pass


def _hunks(base: bytes, candidate: bytes) -> tuple[EditHunk, ...]:
    matcher = difflib.SequenceMatcher(a=base, b=candidate, autojunk=False)
    return tuple(
        EditHunk(i1, i2, base[i1:i2], candidate[j1:j2])
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )


def apply_hunks(base: bytes, hunks: Sequence[EditHunk]) -> bytes:
    """Apply hunks against their original offsets, rejecting stale/overlap data."""
    result = bytearray()
    cursor = 0
    for hunk in sorted(hunks, key=lambda item: (item.start, item.end)):
        if hunk.start < cursor or hunk.end < hunk.start or hunk.end > len(base):
            raise ValueError("hunks overlap or are outside the authoritative base")
        if base[hunk.start:hunk.end] != hunk.old_bytes:
            raise ValueError("hunk preimage does not match the authoritative base")
        result.extend(base[cursor:hunk.start])
        result.extend(hunk.replacement)
        cursor = hunk.end
    result.extend(base[cursor:])
    return bytes(result)


def _segments(source: bytes, sites: Sequence[Mapping[str, object]] = ()) -> tuple[CurrentSourceSegment, ...]:
    if not sites:
        return (CurrentSourceSegment(0, len(source), source),)
    return tuple(
        CurrentSourceSegment(int(site["start"]), int(site["end"]), source[int(site["start"]):int(site["end"])], site)
        for site in sites
        if isinstance(site.get("start"), int) and isinstance(site.get("end"), int)
    )


def _authority_source(authority: object, path: str) -> bytes | None:
    sources = _mapping_value(authority, "sources", default=None)
    if isinstance(sources, Mapping) and path in sources:
        return _as_bytes(sources[path], label="authority source")
    return None


def _check_authority_path(authority: object, path: str) -> None:
    checker = getattr(authority, "_check_path", None)
    if callable(checker):
        checker(path)
        return
    allowed = _mapping_value(authority, "allowed_paths", default=None)
    if allowed is not None and path not in {str(item).replace("\\", "/") for item in allowed}:
        raise ValueError(f"path is outside edit authority: {path}")


def _base_content(path: str, authority: object, handle: object) -> bytes:
    authoritative = _authority_source(authority, path)
    handle_source: object = _MISSING
    if isinstance(handle, Mapping):
        handle_source = _mapping_value(handle, "bytes", "content_bytes", "source", "content", "body", default=_MISSING)
    elif isinstance(handle, (bytes, bytearray, memoryview, str)):
        handle_source = handle
    if authoritative is not None:
        if handle_source is not _MISSING and _as_bytes(handle_source, label="base content") != authoritative:
            raise ValueError("base content handle does not match edit authority")
        return authoritative
    if handle_source is _MISSING:
        raise ValueError("authoritative base content is unavailable")
    return _as_bytes(handle_source, label="base content")


def _payload_bytes(payload: Mapping[str, object], *, mode: str) -> bytes:
    names = ("candidate", "full_file", "replacement", "new_source", "source", "content", "new")
    value: object = _MISSING
    for name in names:
        if name in payload:
            value = payload[name]
            break
    if isinstance(value, Mapping):
        value = _mapping_value(value, "bytes", "source", "content", "body", default=_MISSING)
    if value is _MISSING:
        raise ValueError(f"{mode} payload is missing its complete candidate")
    return _as_bytes(value, label="edit candidate")


def _payload_mode(payload: Mapping[str, object]) -> str:
    raw = _normal_text(payload.get("mode", payload.get("scope", "complete_file"))).casefold()
    if raw in {"complete_file", "complete-file", "whole_file", "whole-file", "file", "complete"}:
        return "complete-file"
    if raw in {"complete_definition", "complete-definition", "definition", "function", "method", "structural"}:
        return "complete-definition"
    raise ValueError(f"unsupported normalization mode: {raw}")


def _target_mapping(payload: Mapping[str, object], target_site: object) -> Mapping[str, object]:
    value: object = target_site
    if value is None:
        value = payload.get("target_site", payload.get("site_ref", payload.get("target", {})))
    if isinstance(value, Mapping):
        result = dict(value)
        span = result.get("span")
        if isinstance(span, Mapping):
            result.update({"start": span.get("start"), "end": span.get("end")})
        return result
    if value is None:
        return {}
    result = {}
    for name in ("site_id", "path", "start", "end", "node_kind", "boundary", "owner", "name", "signature",
                 "authority_ref", "source_digest", "file_revision", "permitted_operations"):
        if hasattr(value, name):
            result[name] = getattr(value, name)
    return result


def _authority_allows_definition(authority: object, payload: Mapping[str, object], target: Mapping[str, object]) -> bool:
    """Require an explicit definition scope or a registered bound definition site.

    A span is evidence about a target, not permission to change the meaning of a
    whole-file contract.  The old implementation treated any ``start/end`` pair
    as definition authority, which allowed a caller to silently narrow a
    whole-file affordance.
    """
    if bool(payload.get("supports_definition")) or bool(payload.get("definition_authority")):
        return True
    if bool(_mapping_value(authority, "supports_definition", "definition_authority", default=False)):
        return True
    scope = _mapping_value(authority, "edit_scope", "scope", default=None)
    if isinstance(scope, Mapping):
        mode = _normal_text(scope.get("mode", scope.get("accepted_scope", ""))).casefold()
        if mode in {
            "complete-definition", "complete_definition", "top_level_function_or_class",
            "qualified_function", "definition",
        }:
            return True
    site_id = str(target.get("site_id", "") or "")
    permitted = target.get("permitted_operations")
    can_replace = isinstance(permitted, Sequence) and not isinstance(
        permitted, (str, bytes, bytearray)
    ) and "replace" in {str(item) for item in permitted}
    registered = _mapping_value(authority, "sites", "bound_sites", default=None)
    if site_id and can_replace:
        if isinstance(registered, Mapping) and site_id in registered:
            return True
        if isinstance(registered, Sequence) and not isinstance(registered, (str, bytes, bytearray)):
            return any(
                (item is not None and str(_mapping_value(item, "site_id", default=item)) == site_id)
                for item in registered
            )
    return False


def _authority_scope_mode(authority: object) -> str:
    scope = _mapping_value(authority, "edit_scope", "scope", "normalization", default=None)
    if not isinstance(scope, Mapping):
        return ""
    return _normal_text(scope.get("mode", scope.get("accepted_scope", ""))).casefold()


def _python_definition_spans(source: bytes) -> tuple[dict[str, object], ...]:
    try:
        text = source.decode("utf-8")
        tree = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise _ParserUnavailable(f"python parser rejected source: {type(error).__name__}") from error
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    def byte_offset(char_offset: int) -> int:
        return len(text[:char_offset].encode("utf-8"))

    records: list[dict[str, object]] = []

    def visit(node: ast.AST, owner: str | None = None) -> None:
        is_definition = isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        next_owner = owner
        if is_definition:
            decorators = list(getattr(node, "decorator_list", ()))
            start_line = min([int(getattr(node, "lineno", 1)), *[int(item.lineno) for item in decorators]])
            end_line = int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))
            start_char = offsets[max(0, start_line - 1)] + min(
                [int(getattr(node, "col_offset", 0)), *[int(item.col_offset) for item in decorators]]
            )
            end_char = offsets[max(0, end_line - 1)] + int(getattr(node, "end_col_offset", 0))
            name = str(getattr(node, "name", ""))
            signature_line = text[offsets[int(node.lineno) - 1]:offsets[int(node.lineno) - 1] + 400].splitlines()[0].strip()
            kind = "class" if isinstance(node, ast.ClassDef) else ("method" if owner else "function")
            record: dict[str, object] = {
                "kind": kind, "name": name, "owner": owner, "signature": signature_line,
                "start": byte_offset(start_char), "end": byte_offset(end_char),
                "line": start_line, "end_line": end_line, "node_type": type(node).__name__,
                "parser": "python-ast",
            }
            records.append(record)
            next_owner = name if owner is None else f"{owner}.{name}"
        for child in ast.iter_child_nodes(node):
            visit(child, next_owner)

    visit(tree)
    return tuple(records)


def _definition_spans(path: str, source: bytes) -> tuple[dict[str, object], ...]:
    if Path(path).suffix.casefold() == ".py":
        return _python_definition_spans(source)
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _ParserUnavailable("source is not valid UTF-8 for a structural parser") from error
    adapter_path = path
    if Path(path).suffix.casefold() == ".vue" and not re.search(r"<\s*(?:template|script|style)\b", text, re.IGNORECASE):
        # A Vue script member may be supplied as the complete member only; validate
        # it with the verified TS adapter rather than pretending a bare fragment is
        # an SFC.  The enclosing SFC span remains authority-bound on the current side.
        adapter_path = path + ".ts"
    try:
        from mc_repo_intel_vendor.ts_dispatch import structural_spans
        result = structural_spans(text, adapter_path)
    except Exception as error:  # noqa: BLE001 - unavailable adapter is typed below
        raise _ParserUnavailable(f"structural parser unavailable: {type(error).__name__}") from error
    if str(result.get("parser")) == "unavailable":
        raise _ParserUnavailable(str(result.get("reason", "structural parser unavailable")))
    return tuple(item for item in result.get("spans", ()) if isinstance(item, Mapping) and item.get("kind") in _DEFINITION_KINDS)


def _member_context(path: str, replacement: bytes, current: Mapping[str, object]) -> tuple[str, str] | None:
    """Build parser input for a member payload that is not a standalone unit.

    Java/TypeScript/PHP methods and Rust impl/trait members are valid source
    definitions but their grammars require an enclosing owner.  The enclosing
    wrapper is parser input only; the returned candidate bytes are never changed.
    """
    suffix = Path(path).suffix.casefold()
    if suffix not in {".ts", ".tsx", ".js", ".jsx", ".java", ".php", ".rs", ".vue"}:
        return None
    try:
        text = replacement.decode("utf-8")
    except UnicodeDecodeError:
        return None
    owner = str(current.get("owner") or "NormalizationOwner")
    owner = re.sub(r"[^A-Za-z0-9_$]", "", owner.split(".")[-1]) or "NormalizationOwner"
    if suffix == ".rs":
        keyword = "trait" if text.rstrip().endswith(";") else "impl"
        return f"{keyword} {owner} {{\n{text}\n}}\n", path
    if suffix == ".php":
        return f"<?php\nclass {owner} {{\n{text}\n}}\n", path
    if suffix == ".vue":
        return f"<script>\nclass {owner} {{\n{text}\n}}\n</script>\n", path
    return f"class {owner} {{\n{text}\n}}\n", path


def _candidate_definition_spans(
    path: str,
    replacement: bytes,
    current: Mapping[str, object],
    requested_name: str,
    requested_owner: str,
    requested_block_tag: str,
) -> tuple[tuple[dict[str, object], ...], bool]:
    """Resolve replacement structure, returning ``(spans, wrapped)``."""
    direct_error: _ParserUnavailable | None = None
    try:
        direct = _definition_spans(path, replacement)
    except _ParserUnavailable as error:
        direct = ()
        direct_error = error

    def matches(rows: Sequence[Mapping[str, object]]) -> bool:
        for row in rows:
            if requested_name and str(row.get("name")) != requested_name:
                continue
            if requested_owner and _normal_text(row.get("owner")) != _normal_text(requested_owner):
                continue
            if requested_block_tag and _normal_text(row.get("block_tag")) != requested_block_tag:
                continue
            return True
        return False

    if direct and matches(direct):
        return direct, False
    context = _member_context(path, replacement, current)
    if context is not None and str(current.get("kind")) == "method":
        wrapped_source, wrapped_path = context
        try:
            wrapped = _definition_spans(wrapped_path, wrapped_source.encode("utf-8"))
        except _ParserUnavailable:
            wrapped = ()
        if matches(wrapped):
            return wrapped, True
    if direct_error is not None:
        raise direct_error
    return direct, False


def _target_site_for_authority(authority: object, target: Mapping[str, object], path: str, base: bytes) -> Mapping[str, object] | None:
    if target.get("start") is None or target.get("end") is None:
        return None
    try:
        start, end = int(target["start"]), int(target["end"])
    except (TypeError, ValueError):
        return None
    if start < 0 or end <= start or end > len(base):
        return None
    return {
        "path": path, "start": start, "end": end,
        "node_kind": str(target.get("node_kind", target.get("kind", "definition"))),
        "site_id": str(target.get("site_id", "")),
    }


def _definition_result(
    *,
    path: str,
    base: bytes,
    replacement: bytes,
    payload: Mapping[str, object],
    authority: object,
    target_site: object,
) -> NormalizationResult:
    target = _target_mapping(payload, target_site)
    if not _authority_allows_definition(authority, payload, target):
        return Rejected("DEFINITION_AUTHORITY_UNAVAILABLE", _segments(base), path)
    try:
        current_sites = _definition_spans(path, base)
    except _ParserUnavailable as error:
        return Rejected(f"PARSER_UNAVAILABLE: {error}", _segments(base), path)
    requested_name = str(target.get("name", target.get("symbol", target.get("member", payload.get("name", "")))) or "")
    requested_owner = str(target.get("owner", target.get("parent", payload.get("owner", ""))) or "")
    requested_signature = _normal_text(target.get("signature", payload.get("signature", "")))
    requested_block_tag = _normal_text(target.get("block_tag", payload.get("block_tag", "")))
    target_authority = target.get("authority_ref")
    authority_ref = _mapping_value(authority, "authority_ref", default=None)
    if target_authority is not None and authority_ref is not None and str(target_authority) != str(authority_ref):
        return Rejected("TARGET_AUTHORITY_MISMATCH", _segments(base), path)
    target_path = target.get("path")
    if target_path is not None and str(target_path).replace("\\", "/") != path:
        return Rejected("TARGET_PATH_MISMATCH", _segments(base), path)
    target_digest = target.get("source_digest")
    if target_digest is not None and str(target_digest) != _digest(base):
        return Rejected("STALE_TARGET_SOURCE", _segments(base), path, _digest(base))
    target_revision = target.get("file_revision")
    if target_revision is not None and str(target_revision).startswith("sha256:") \
            and str(target_revision) != _digest(base):
        return Rejected("STALE_TARGET_REVISION", _segments(base), path, _digest(base))
    permitted = target.get("permitted_operations")
    if isinstance(permitted, Sequence) and not isinstance(permitted, (str, bytes, bytearray)):
        if "replace" not in {str(item) for item in permitted}:
            return Rejected("TARGET_OPERATION_UNAUTHORIZED", _segments(base), path)
    if target.get("boundary") is not None:
        return Rejected("TARGET_IS_INSERTION_BOUNDARY", _segments(base), path)
    site = _target_site_for_authority(authority, target, path, base)
    candidates: list[Mapping[str, object]] = []
    for row in current_sites:
        if site is not None and (row.get("start") != site["start"] or row.get("end") != site["end"]):
            continue
        if requested_name and str(row.get("name")) != requested_name:
            continue
        if requested_owner and _normal_text(row.get("owner")) != _normal_text(requested_owner):
            continue
        if requested_signature and _normal_text(row.get("signature")) != requested_signature:
            continue
        wanted_kind = str(target.get("kind", target.get("node_kind", "")))
        if wanted_kind and wanted_kind in _DEFINITION_KINDS and str(row.get("kind")) != wanted_kind:
            continue
        if requested_block_tag and _normal_text(row.get("block_tag")) != requested_block_tag:
            continue
        candidates.append(row)
    if not candidates:
        return Rejected("NO_UNIQUE_CURRENT_OWNER", _segments(base, current_sites), path)
    if len(candidates) != 1:
        return AmbiguousScope("multiple current owners match the supplied definition target", tuple(candidates), _segments(base, candidates), path)
    current = candidates[0]
    try:
        candidate_sites, wrapped = _candidate_definition_spans(
            path, replacement, current, requested_name, requested_owner, requested_block_tag,
        )
    except _ParserUnavailable as error:
        return Rejected(f"REPLACEMENT_PARSER_UNAVAILABLE: {error}", _segments(base, (current,)), path)
    candidate_sites = tuple(item for item in candidate_sites if item.get("kind") in _DEFINITION_KINDS)
    if requested_name:
        candidate_sites = tuple(item for item in candidate_sites if str(item.get("name")) == requested_name)
    if requested_owner:
        candidate_sites = tuple(item for item in candidate_sites if _normal_text(item.get("owner")) == _normal_text(requested_owner))
    if requested_block_tag:
        candidate_sites = tuple(item for item in candidate_sites if _normal_text(item.get("block_tag")) == requested_block_tag)
    if len(candidate_sites) != 1:
        return Rejected("MALFORMED_OR_PARTIAL_DEFINITION_PAYLOAD", _segments(base, (current,)), path)
    replacement_site = candidate_sites[0]
    if str(replacement_site.get("kind")) != str(current.get("kind")):
        return Rejected("DEFINITION_KIND_MISMATCH", _segments(base, (current,)), path)
    if str(current.get("kind")) == "method" and not wrapped:
        prefix = replacement[:int(replacement_site.get("start", 0))]
        suffix = replacement[int(replacement_site.get("end", len(replacement))):]
        if prefix.strip() or suffix.strip():
            return Rejected(
                "MALFORMED_OR_PARTIAL_DEFINITION_PAYLOAD",
                _segments(base, (current,)), path,
            )
    start, end = int(current["start"]), int(current["end"])
    candidate = base[:start] + replacement + base[end:]
    hunks = _hunks(base, candidate)
    if apply_hunks(base, hunks) != candidate:
        return Rejected("DIFF_ROUND_TRIP_FAILED", _segments(base, (current,)), path)
    exterior_before = base[:start] + base[end:]
    exterior_after = candidate[:start] + candidate[start + len(replacement):]
    proof = {
        "exact_bytes": True,
        "round_trip": True,
        "exterior_unchanged": exterior_before == exterior_after,
        "exterior_digest": _digest(exterior_before),
        "target_span": {"start": start, "end": end},
        "parser": current.get("parser"),
        "owner": current.get("owner"),
        "signature": current.get("signature"),
    }
    if not bool(proof["exterior_unchanged"]):
        return Rejected("EXTERIOR_BYTES_CHANGED", _segments(base, (current,)), path)
    return NormalizedEdit(
        "complete-definition", path, _digest(base), _digest(candidate), hunks,
        tuple(ChangedSpan(hunk.start, hunk.end) for hunk in hunks), proof,
        _segments(base, (current,)), candidate, base, site or dict(current),
    )


def normalize_edit(
    public_authority: object,
    base_content_handle: object,
    payload: Mapping[str, object],
    target_site: object = None,
) -> NormalizationResult:
    """Normalize one complete-file or complete-definition payload mechanically."""
    if not isinstance(payload, Mapping):
        return Rejected("MALFORMED_PAYLOAD")
    try:
        mode = _payload_mode(payload)
        target = _target_mapping(payload, target_site)
        raw_path = payload.get("path", target.get("path", _mapping_value(base_content_handle, "path", default="")))
        path = _path(raw_path)
        _check_authority_path(public_authority, path)
        base = _base_content(path, public_authority, base_content_handle)
    except (KeyError, TypeError, ValueError) as error:
        return Rejected(f"MALFORMED_PAYLOAD: {error}")
    expected = str(payload.get("expected_base_digest", payload.get("base_digest", "")) or "").strip()
    if not expected and isinstance(base_content_handle, Mapping):
        expected = str(base_content_handle.get("content_digest", base_content_handle.get("digest", "")) or "").strip()
    actual = _digest(base)
    if expected and expected != actual:
        return Rejected(f"STALE_BASE: expected {expected}, current {actual}", _segments(base), path, expected)
    authority_revision = _mapping_value(public_authority, "base_revision", default=None)
    requested_revision = payload.get("base_revision")
    if requested_revision is not None and authority_revision is not None and str(requested_revision) != str(authority_revision):
        return Rejected("STALE_REVISION: normalization payload is not bound to the current authority", _segments(base), path, expected or actual)
    handle_revision = None
    if isinstance(base_content_handle, Mapping):
        handle_revision = base_content_handle.get(
            "base_revision", base_content_handle.get("file_revision", base_content_handle.get("revision"))
        )
    if handle_revision is not None and authority_revision is not None \
            and str(handle_revision) != str(authority_revision):
        return Rejected("STALE_REVISION: base content handle is not current", _segments(base), path, expected or actual)
    authority_ref = _mapping_value(public_authority, "authority_ref", default=None)
    if payload.get("authority_ref") is not None and authority_ref is not None and str(payload["authority_ref"]) != str(authority_ref):
        return Rejected("AUTHORITY_MISMATCH", _segments(base), path, expected or actual)
    scope_mode = _authority_scope_mode(public_authority)
    if scope_mode in {"whole_file", "whole-file", "complete-file", "complete_file"} \
            and mode == "complete-definition":
        return Rejected("DEFINITION_AUTHORITY_UNAVAILABLE", _segments(base), path, expected or actual)
    if scope_mode in {
        "complete-definition", "complete_definition", "definition", "top_level_function_or_class",
        "qualified_function",
    } and mode == "complete-file":
        return Rejected("WHOLE_FILE_AUTHORITY_UNAVAILABLE", _segments(base), path, expected or actual)
    try:
        candidate = _payload_bytes(payload, mode=mode)
    except (TypeError, ValueError) as error:
        return Rejected(f"MALFORMED_PAYLOAD: {error}", _segments(base), path, expected or actual)
    if candidate == base:
        return Rejected("NO_CHANGE", _segments(base), path, expected or actual)
    if mode == "complete-definition":
        return _definition_result(
            path=path, base=base, replacement=candidate, payload=payload,
            authority=public_authority, target_site=target_site,
        )
    hunks = _hunks(base, candidate)
    try:
        round_trip = apply_hunks(base, hunks)
    except ValueError as error:
        return Rejected(f"DIFF_ROUND_TRIP_FAILED: {error}", _segments(base), path, expected or actual)
    if round_trip != candidate:
        return Rejected("DIFF_ROUND_TRIP_FAILED", _segments(base), path, expected or actual)
    equal_bytes = sum(i2 - i1 for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(a=base, b=candidate, autojunk=False).get_opcodes() if tag == "equal")
    proof = {
        "exact_bytes": True,
        "round_trip": True,
        "all_hunks_retained": True,
        "broad_change": len(hunks) > 1 or any(len(hunk.old_bytes) > 256 or len(hunk.replacement) > 256 for hunk in hunks),
        "unchanged_byte_count": equal_bytes,
    }
    return NormalizedEdit(
        "complete-file", path, expected or actual, _digest(candidate), hunks,
        tuple(ChangedSpan(hunk.start, hunk.end) for hunk in hunks), proof,
        _segments(base), candidate, base,
    )


__all__ = [
    "AmbiguousScope", "ChangedSpan", "CurrentSourceSegment", "EditHunk",
    "NormalizedEdit", "NormalizationResult", "Rejected", "apply_hunks",
    "normalize_edit",
]

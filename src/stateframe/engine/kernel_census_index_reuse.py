"""Run-local, externally validated reuse for the call-site census index."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kernel_repo_catalog import (
    MAX_FILES,
    MAX_OVERLAY_FILES,
    MAX_SCAN_ENTRIES,
    StructuralIndex,
    StructuralIndexError,
    _relative as _catalog_relative,
    combine_structural_indexes,
    merge_structural_index,
    scan_entry_set,
    subset_structural_index,
)


MODE_ENV = "STATEFRAME_CENSUS_INDEX_REUSE"
VALIDATION_CAP_MIB_ENV = "STATEFRAME_CENSUS_VALIDATION_CAP_MIB"
CACHE_ATTR = "_stateframe_census_structural_cache"
MIB = 1024 * 1024
DEFAULT_VALIDATION_SOURCE_CAP_MIB = 32
MAX_VALIDATION_SOURCE_CAP_MIB = 256
MAX_VALIDATION_SOURCE_BYTES = DEFAULT_VALIDATION_SOURCE_CAP_MIB * MIB
MAX_VALIDATION_SOURCE_BYTES_HARD_LIMIT = MAX_VALIDATION_SOURCE_CAP_MIB * MIB
MAX_VALIDATION_DIRECTORIES = 16_384

_Signature = tuple[int, ...]


@dataclass(slots=True)
class RuntimeIndexCache:
    """One validated base generation plus its active focus overlay."""

    root: str
    scope_key: str
    parser_version: str
    priority_paths: tuple[str, ...]
    index: StructuralIndex
    file_signatures: dict[str, _Signature]
    structural_revision: str
    analysis: object | None = None
    overlay_paths: tuple[str, ...] = ()
    overlay_index: StructuralIndex | None = None
    overlay_revision: str | None = None
    last_priority_paths: tuple[str, ...] = ()
    legacy_priority: bool = False


@dataclass(frozen=True, slots=True)
class IndexLease:
    index: StructuralIndex
    cache: RuntimeIndexCache | None
    changed_paths: tuple[str, ...] = ()
    incremental: bool = False
    reuse_status: Literal["off", "cold", "warm", "warm_overlay", "incremental", "uncached"] = "cold"
    reuse_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _Validation:
    kind: Literal["warm", "incremental", "miss"]
    changed_paths: tuple[str, ...] = ()
    file_signatures: Mapping[str, _Signature] | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _OverlayResolution:
    paths: tuple[str, ...]
    index: StructuralIndex | None
    revision: str | None
    changed_paths: tuple[str, ...] = ()
    rebuilt: bool = False
    error: str | None = None
    focus_changed: bool = False


def reuse_enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    value = source.get(MODE_ENV, "on")
    if value not in {"on", "off"}:
        raise ValueError(f"{MODE_ENV} must be on or off")
    return value == "on"


def validation_source_cap_mib(environ: Mapping[str, str] | None = None) -> int:
    """Return the configured MiB cap, defaulting safely and enforcing the hard ceiling."""
    source = os.environ if environ is None else environ
    try:
        configured = int(source.get(VALIDATION_CAP_MIB_ENV, str(DEFAULT_VALIDATION_SOURCE_CAP_MIB)))
    except (TypeError, ValueError):
        configured = DEFAULT_VALIDATION_SOURCE_CAP_MIB
    return min(max(configured, 0), MAX_VALIDATION_SOURCE_CAP_MIB)


def validation_source_byte_limit(environ: Mapping[str, str] | None = None) -> int:
    return validation_source_cap_mib(environ) * MIB


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str, allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _signature(path: Path) -> _Signature | None:
    try:
        current = path.lstat()
    except OSError:
        return None
    return (
        int(current.st_dev), int(current.st_ino), int(current.st_mode),
        int(current.st_size), int(current.st_mtime_ns), int(current.st_ctime_ns),
    )


def _dependency_revision(index: StructuralIndex) -> str:
    modules = [
        (module.path, tuple(module.imports))
        for module in index.modules
    ]
    edges = [
        (row.source_id, row.target_id, row.relation, row.distance)
        for row in index.dependencies
    ]
    return _digest({"modules": modules, "edges": edges})


def _priority_identity(priority_paths: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({
        str(path).replace("\\", "/").strip()
        for path in priority_paths if str(path).strip()
    }))


def _catalog_complete(index: StructuralIndex) -> bool:
    """Return true only when the stable catalog proves every candidate was indexed."""
    stats = getattr(index, "stats", None)
    candidate_files = getattr(stats, "candidate_files", None)
    indexed_files = getattr(stats, "indexed_files", None)
    omitted_files = getattr(stats, "omitted_files", None)
    priority_files_omitted = getattr(stats, "priority_files_omitted", None)
    return (
        bool(getattr(index, "scan_stable", False))
        and getattr(index, "scan_entries", None) is not None
        and type(candidate_files) is int
        and type(indexed_files) is int
        and candidate_files == indexed_files
        and type(omitted_files) is int
        and omitted_files == 0
        and type(priority_files_omitted) is int
        and priority_files_omitted == 0
        and getattr(stats, "scan_scope", None) in {"full", "prioritized"}
    )


def structural_revision(
    index: StructuralIndex,
    *,
    scope_key: str,
    parser_version: str,
    priority_paths: Sequence[str] = (),
) -> str:
    """Bind structure to scope, parser and source bytes; legacy scans also bind focus."""
    priority_scope = (
        _priority_identity(priority_paths)
        if not _catalog_complete(index) and priority_paths else None
    )
    return _digest({
        "namespace": index.namespace.to_dict(),
        "scope": scope_key,
        "parser": parser_version,
        "catalog": "catalog-structural-index/1",
        "priority_paths": priority_scope,
        "files": [(module.path, module.content.content_digest) for module in index.modules],
        "dependencies": _dependency_revision(index),
        "scan_entries": index.scan_entries,
    })


def _source_bytes(index: StructuralIndex) -> int:
    return sum(module.content.byte_length for module in index.modules)


def index_source_bytes(index: StructuralIndex | None) -> int:
    """Expose bounded index accounting for outcome journaling."""
    return _source_bytes(index) if index is not None else 0


def _fresh_signatures(index: StructuralIndex) -> dict[str, _Signature] | None:
    if len(index.modules) > MAX_FILES:
        return None
    signatures: dict[str, _Signature] = {}
    for module in index.modules:
        signature = _signature(index.root / module.path)
        if signature is None or not stat.S_ISREG(signature[2]):
            return None
        signatures[module.path] = signature
    return signatures


def _state_for(
    index: StructuralIndex,
    *,
    root: Path,
    scope_key: str,
    parser_version: str,
    priority_paths: Sequence[str],
    max_source_bytes: int,
    legacy_priority: bool = False,
) -> tuple[RuntimeIndexCache | None, str | None]:
    entries = getattr(index, "scan_entries", None)
    if len(index.modules) > MAX_FILES or _source_bytes(index) > max_source_bytes:
        return None, "limits"
    if entries is not None and (
            len(entries) > MAX_SCAN_ENTRIES
            or sum(kind == "directory" for _, kind in entries) > MAX_VALIDATION_DIRECTORIES):
        return None, "limits"
    if not index.scan_stable or entries is None:
        return None, "scan_unstable"
    signatures = _fresh_signatures(index)
    if signatures is None:
        return None, "scan_unstable"
    request_priorities = _priority_identity(priority_paths)
    key_priorities = request_priorities if legacy_priority else ()
    state = RuntimeIndexCache(
        root=root.as_posix(),
        scope_key=scope_key,
        parser_version=parser_version,
        priority_paths=key_priorities,
        index=index,
        file_signatures=signatures,
        structural_revision=structural_revision(
            index, scope_key=scope_key, parser_version=parser_version,
            priority_paths=key_priorities,
        ),
        last_priority_paths=request_priorities,
        legacy_priority=legacy_priority,
    )
    return state, None


def _entry_state_matches(index: StructuralIndex) -> tuple[bool, str | None]:
    expected = getattr(index, "scan_entries", None)
    if expected is None:
        return False, "scan_unstable"
    if (len(expected) > MAX_SCAN_ENTRIES
            or sum(kind == "directory" for _, kind in expected) > MAX_VALIDATION_DIRECTORIES):
        return False, "limits"
    current = scan_entry_set(index.root)
    if current is None:
        return False, "scan_unstable"
    if current != expected:
        return False, "directory_entries_changed"
    return True, None


def _file_digest(path: Path, *, max_bytes: int) -> tuple[_Signature, str] | None:
    before = _signature(path)
    if before is None or not stat.S_ISREG(before[2]) or before[3] > max_bytes:
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    digest = hashlib.sha256()
    byte_count = 0
    try:
        opened = os.fstat(descriptor)
        opened_signature = (
            int(opened.st_dev), int(opened.st_ino), int(opened.st_mode),
            int(opened.st_size), int(opened.st_mtime_ns), int(opened.st_ctime_ns),
        )
        if opened_signature != before or not stat.S_ISREG(opened.st_mode):
            return None
        while True:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes - byte_count + 1))
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > max_bytes:
                return None
            digest.update(chunk)
        after = os.fstat(descriptor)
        after_signature = (
            int(after.st_dev), int(after.st_ino), int(after.st_mode),
            int(after.st_size), int(after.st_mtime_ns), int(after.st_ctime_ns),
        )
        if after_signature != before or _signature(path) != before:
            return None
        if byte_count != before[3]:
            return None
        return before, "sha256:" + digest.hexdigest()
    except OSError:
        return None
    finally:
        os.close(descriptor)


def _digest_paths_within_source_budget(
    root: Path,
    paths: Sequence[str],
    *,
    max_source_bytes: int,
    already_counted: int = 0,
) -> dict[str, tuple[_Signature, str]] | None:
    """Hash each path once without allowing aggregate bytes to exceed the cap."""
    if already_counted < 0 or already_counted > max_source_bytes:
        return None
    validated: dict[str, tuple[_Signature, str]] = {}
    total_bytes = already_counted
    for relative in paths:
        if relative in validated:
            return None
        result = _file_digest(root / relative, max_bytes=max_source_bytes - total_bytes)
        if result is None:
            return None
        signature, digest = result
        total_bytes += signature[3]
        if total_bytes > max_source_bytes:
            return None
        validated[relative] = (signature, digest)
    return validated


def _validate_entry(cache: RuntimeIndexCache, max_source_bytes: int) -> _Validation:
    """Compare the bounded scan inventory, then hash every indexed source file."""
    entries = getattr(cache.index, "scan_entries", None)
    if (len(cache.index.modules) > MAX_FILES
            or _source_bytes(cache.index) > max_source_bytes):
        return _Validation("miss", reason="limits")
    if entries is not None and (
            len(entries) > MAX_SCAN_ENTRIES
            or sum(kind == "directory" for _, kind in entries) > MAX_VALIDATION_DIRECTORIES):
        return _Validation("miss", reason="limits")
    if not cache.index.scan_stable or entries is None:
        return _Validation("miss", reason="scan_unstable")
    if structural_revision(
            cache.index, scope_key=cache.scope_key, parser_version=cache.parser_version,
            priority_paths=cache.priority_paths,
    ) != cache.structural_revision:
        return _Validation("miss", reason="cache_integrity_changed")
    matches, reason = _entry_state_matches(cache.index)
    if not matches:
        return _Validation("miss", reason=reason)

    validated = _digest_paths_within_source_budget(
        cache.index.root,
        tuple(module.path for module in cache.index.modules),
        max_source_bytes=max_source_bytes,
    )
    if validated is None:
        return _Validation("miss", reason="scan_unstable")

    changed: list[str] = []
    signatures: dict[str, _Signature] = {}
    for module in cache.index.modules:
        signature, digest = validated[module.path]
        signatures[module.path] = signature
        if digest != module.content.content_digest:
            changed.append(module.path)

    matches, reason = _entry_state_matches(cache.index)
    if not matches:
        return _Validation("miss", reason=reason)
    changed_paths = tuple(changed)
    return _Validation(
        "incremental" if changed_paths else "warm",
        changed_paths, signatures,
        "file_digest_changed" if changed_paths else None,
    )


def _install(runtime: object, cache: RuntimeIndexCache | None) -> RuntimeIndexCache | None:
    try:
        setattr(runtime, CACHE_ATTR, cache)
    except (AttributeError, TypeError):
        return None
    return cache


def _scope_compatibility(
    previous: object,
    *,
    root: Path,
    scope_key: str,
    parser_version: str,
    priority_paths: Sequence[str],
) -> tuple[bool, bool, str | None]:
    if not isinstance(previous, RuntimeIndexCache):
        reason = "cache_missing" if previous is None else "cache_unavailable"
        return False, False, reason
    same_scope = (
        previous.root == root.resolve().as_posix()
        and previous.scope_key == scope_key
        and previous.parser_version == parser_version
    )
    if not same_scope:
        return False, False, "scope_changed"

    requested = _priority_identity(priority_paths)
    priority_mismatch = previous.last_priority_paths != requested
    if (previous.legacy_priority and priority_mismatch
            and not _catalog_complete(previous.index)):
        return False, False, "priority_scope_changed_truncated_scan"
    return True, priority_mismatch, None


def _candidate_priorities(
    root: Path,
    priority_paths: Sequence[str],
    entries: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    candidates = {path for path, kind in entries if kind == "source"}
    selected: set[str] = set()
    for path in priority_paths:
        try:
            normalized = _catalog_relative(root, path)
        except StructuralIndexError:
            continue
        if normalized in candidates:
            selected.add(normalized)
    return tuple(sorted(selected))


def _split_cached_index(
    index: StructuralIndex,
    *,
    priority_paths: Sequence[str],
) -> tuple[StructuralIndex, StructuralIndex | None, tuple[str, ...]] | None:
    entries = index.scan_entries
    candidate_files = index.stats.candidate_files
    if entries is None or type(candidate_files) is not int or not index.scan_stable:
        return None
    candidates = tuple(sorted(path for path, kind in entries if kind == "source"))
    requested = set(_candidate_priorities(index.root, priority_paths, entries))
    selected_count = max(0, candidate_files - index.stats.omitted_files)
    base_count = min(MAX_FILES, len(candidates))
    for count in range(base_count, -1, -1):
        extras = requested.difference(candidates[:count])
        if count + len(extras) == selected_count and len(extras) <= MAX_OVERLAY_FILES:
            base_count = count
            break
    else:
        return None
    base_paths = candidates[:base_count]
    overlay_paths = tuple(sorted(requested.difference(base_paths)))
    expected_paths = set(base_paths).union(overlay_paths)
    if {module.path for module in index.modules} != expected_paths:
        return None

    omitted = max(0, candidate_files - base_count)
    base_scope = "full" if omitted == 0 else "truncated"
    base = subset_structural_index(
        index,
        base_paths,
        candidate_files=candidate_files,
        omitted_files=omitted,
        priority_files_omitted=len(requested.difference(base_paths)),
        scan_scope=base_scope,
        keep_scan_inventory=True,
    )
    overlay = None
    if overlay_paths:
        overlay = subset_structural_index(
            index,
            overlay_paths,
            candidate_files=None,
            omitted_files=0,
            scan_scope="focused",
        )
    return base, overlay, overlay_paths


def _validate_overlay(cache: RuntimeIndexCache, max_source_bytes: int) -> _Validation:
    overlay = cache.overlay_index
    if not cache.overlay_paths:
        return _Validation("warm") if overlay is None else _Validation(
            "miss", reason="cache_integrity_changed",
        )
    if overlay is None or {module.path for module in overlay.modules} != set(cache.overlay_paths):
        return _Validation("miss", reason="scan_unstable")
    if len(cache.overlay_paths) > MAX_OVERLAY_FILES:
        return _Validation("miss", reason="limits")
    if _source_bytes(cache.index) + _source_bytes(overlay) > max_source_bytes:
        return _Validation("miss", reason="limits")
    expected_revision = structural_revision(
        overlay,
        scope_key=cache.scope_key,
        parser_version=cache.parser_version,
        priority_paths=cache.overlay_paths,
    )
    if expected_revision != cache.overlay_revision:
        return _Validation("miss", reason="cache_integrity_changed")

    validated = _digest_paths_within_source_budget(
        overlay.root,
        cache.overlay_paths,
        max_source_bytes=max_source_bytes,
        already_counted=_source_bytes(cache.index),
    )
    if validated is None:
        return _Validation("miss", reason="scan_unstable")

    changed: list[str] = []
    for module in overlay.modules:
        _signature_value, digest = validated[module.path]
        if digest != module.content.content_digest:
            changed.append(module.path)
    return _Validation(
        "incremental" if changed else "warm",
        tuple(changed),
        reason="file_digest_changed" if changed else None,
    )


def acquire_index(
    runtime: object,
    root: Path,
    *,
    scope_key: str,
    parser_version: str,
    priority_paths: Sequence[str],
    enabled: bool,
    build_index_fn: Callable[..., StructuralIndex],
    environ: Mapping[str, str] | None = None,
) -> IndexLease:
    """Return a validated run-local base and a focus-specific overlay."""
    max_source_bytes = min(
        validation_source_byte_limit(environ), MAX_VALIDATION_SOURCE_BYTES_HARD_LIMIT,
    )
    if not enabled:
        index = build_index_fn(
            root,
            priority_paths=priority_paths,
            include_call_graph=False,
        )
        return IndexLease(index, None, reuse_status="off", reuse_reason="disabled")

    resolved = root.resolve()
    requested_priorities = _priority_identity(priority_paths)
    previous = getattr(runtime, CACHE_ATTR, None)
    compatible, priority_mismatch, cold_reason = _scope_compatibility(
        previous,
        root=resolved,
        scope_key=scope_key,
        parser_version=parser_version,
        priority_paths=requested_priorities,
    )
    if compatible and isinstance(previous, RuntimeIndexCache):
        validation = _validate_entry(previous, max_source_bytes)
        if validation.kind in {"warm", "incremental"}:
            lease, refresh_reason = _reuse_cached(
                previous,
                validation,
                priority_mismatch=priority_mismatch,
                priority_paths=priority_paths,
                requested_priorities=requested_priorities,
                scope_key=scope_key,
                parser_version=parser_version,
                max_source_bytes=max_source_bytes,
                build_index_fn=build_index_fn,
            )
            if lease is not None:
                return lease
            cold_reason = refresh_reason or validation.reason or "scan_unstable"
        else:
            cold_reason = validation.reason or "cache_unavailable"

    return _build_cold_lease(
        runtime,
        resolved,
        scope_key=scope_key,
        parser_version=parser_version,
        priority_paths=priority_paths,
        requested_priorities=requested_priorities,
        cold_reason=cold_reason or "cache_unavailable",
        max_source_bytes=max_source_bytes,
        build_index_fn=build_index_fn,
    )


def _build_cold_lease(
    runtime: object,
    root: Path,
    *,
    scope_key: str,
    parser_version: str,
    priority_paths: Sequence[str],
    requested_priorities: tuple[str, ...],
    cold_reason: str | None,
    max_source_bytes: int,
    build_index_fn: Callable[..., StructuralIndex],
) -> IndexLease:
    index = build_index_fn(
        root,
        priority_paths=priority_paths,
        include_call_graph=False,
        capture_scan_entries=True,
    )
    if getattr(index.stats, "scan_scope", None) == "overlay_limit":
        state, state_reason = _state_for(
            index,
            root=root,
            scope_key=scope_key,
            parser_version=parser_version,
            priority_paths=requested_priorities,
            max_source_bytes=max_source_bytes,
            legacy_priority=True,
        )
        installed = _install(runtime, state)
        return IndexLease(
            index,
            installed,
            reuse_status="cold" if installed is not None else "uncached",
            reuse_reason="overlay_limit" if installed is not None else state_reason or "overlay_limit",
        )

    split = _split_cached_index(index, priority_paths=priority_paths)
    if split is None:
        return IndexLease(index, None, reuse_status="uncached", reuse_reason="scan_unstable")
    base, overlay, overlay_paths = split
    if _source_bytes(base) + (_source_bytes(overlay) if overlay is not None else 0) > max_source_bytes:
        return IndexLease(index, None, reuse_status="uncached", reuse_reason="limits")

    state, state_reason = _state_for(
        base,
        root=root,
        scope_key=scope_key,
        parser_version=parser_version,
        priority_paths=(),
        max_source_bytes=max_source_bytes,
    )
    if state is None:
        return IndexLease(
            index,
            None,
            reuse_status="uncached",
            reuse_reason=state_reason or "cache_unavailable",
        )
    state.last_priority_paths = requested_priorities
    if overlay is not None:
        state.overlay_paths = overlay_paths
        state.overlay_index = overlay
        state.overlay_revision = structural_revision(
            overlay,
            scope_key=scope_key,
            parser_version=parser_version,
            priority_paths=overlay_paths,
        )
    installed = _install(runtime, state)
    return IndexLease(
        index,
        installed,
        reuse_status="cold" if installed is not None else "uncached",
        reuse_reason=(cold_reason if installed is not None else state_reason or "cache_unavailable"),
    )


def _refresh_base(
    cache: RuntimeIndexCache,
    changed_paths: Sequence[str],
    *,
    max_source_bytes: int,
    build_index_fn: Callable[..., StructuralIndex],
) -> tuple[StructuralIndex | None, dict[str, _Signature] | None, str | None]:
    reason = "scan_unstable"
    try:
        delta = build_index_fn(
            Path(cache.root),
            namespace=cache.index.namespace,
            focus_paths=changed_paths,
            include_call_graph=False,
        )
        current = merge_structural_index(cache.index, delta)
        matches, entry_reason = _entry_state_matches(cache.index)
        if not matches:
            return None, None, entry_reason or reason
        changed_set = set(changed_paths)
        unchanged_source_bytes = sum(
            module.content.byte_length
            for module in current.modules
            if module.path not in changed_set
        )
        validated = _digest_paths_within_source_budget(
            current.root,
            changed_paths,
            max_source_bytes=max_source_bytes,
            already_counted=unchanged_source_bytes,
        )
        if validated is None:
            return None, None, reason
        for path in changed_paths:
            module = current.module(path)
            verified = validated.get(path)
            if module is None or verified is None or verified[1] != module.content.content_digest:
                return None, None, reason
        signatures = _fresh_signatures(current)
        if signatures is None or _source_bytes(current) > max_source_bytes:
            return None, None, "limits"
        return current, signatures, None
    except (OSError, StructuralIndexError, ValueError):
        return None, None, reason


def _build_overlay_index(
    cache: RuntimeIndexCache,
    overlay_paths: tuple[str, ...],
    *,
    max_source_bytes: int,
    build_index_fn: Callable[..., StructuralIndex],
    previous_overlay: StructuralIndex | None = None,
    changed_paths: Sequence[str] = (),
) -> tuple[StructuralIndex | None, str | None, str | None]:
    if not overlay_paths:
        return None, None, None
    try:
        if previous_overlay is not None and changed_paths:
            delta = build_index_fn(
                Path(cache.root),
                namespace=cache.index.namespace,
                focus_paths=changed_paths,
                include_call_graph=False,
            )
            overlay = merge_structural_index(previous_overlay, delta)
        else:
            overlay = build_index_fn(
                Path(cache.root),
                namespace=cache.index.namespace,
                focus_paths=overlay_paths,
                include_call_graph=False,
            )
    except (OSError, StructuralIndexError, ValueError):
        return None, None, "scan_unstable"

    if {module.path for module in overlay.modules} != set(overlay_paths):
        return None, None, "scan_unstable"
    if len(overlay_paths) > MAX_OVERLAY_FILES:
        return None, None, "overlay_limit"
    if _source_bytes(cache.index) + _source_bytes(overlay) > max_source_bytes:
        return None, None, "limits"
    matches, reason = _entry_state_matches(cache.index)
    if not matches:
        return None, None, reason or "scan_unstable"
    validated = _digest_paths_within_source_budget(
        overlay.root,
        overlay_paths,
        max_source_bytes=max_source_bytes,
        already_counted=_source_bytes(cache.index),
    )
    if validated is None:
        return None, None, "scan_unstable"
    for module in overlay.modules:
        verified = validated.get(module.path)
        if verified is None or verified[1] != module.content.content_digest:
            return None, None, "scan_unstable"
    revision = structural_revision(
        overlay,
        scope_key=cache.scope_key,
        parser_version=cache.parser_version,
        priority_paths=overlay_paths,
    )
    return overlay, revision, None


def _reuse_legacy_priority(
    cache: RuntimeIndexCache,
    validation: _Validation,
    *,
    priority_mismatch: bool,
    requested_priorities: tuple[str, ...],
    scope_key: str,
    parser_version: str,
    max_source_bytes: int,
    build_index_fn: Callable[..., StructuralIndex],
) -> tuple[IndexLease | None, str | None]:
    if validation.kind == "incremental":
        current, signatures, reason = _refresh_base(
            cache, validation.changed_paths, max_source_bytes=max_source_bytes,
            build_index_fn=build_index_fn,
        )
        if current is None or signatures is None:
            return None, reason or "scan_unstable"
        cache.index = current
        cache.file_signatures = signatures
        cache.structural_revision = structural_revision(
            current,
            scope_key=scope_key,
            parser_version=parser_version,
            priority_paths=cache.priority_paths,
        )
        return IndexLease(
            current,
            cache,
            validation.changed_paths,
            incremental=True,
            reuse_status="incremental",
            reuse_reason="file_digest_changed",
        ), None

    cache.file_signatures = dict(validation.file_signatures or {})
    cache.last_priority_paths = requested_priorities
    return IndexLease(
        cache.index,
        cache,
        reuse_status="warm",
        reuse_reason="scope_priority_ignored_complete_scan" if priority_mismatch else None,
    ), None


def _refresh_cached_base(
    cache: RuntimeIndexCache,
    validation: _Validation,
    *,
    max_source_bytes: int,
    build_index_fn: Callable[..., StructuralIndex],
) -> tuple[tuple[str, ...], str | None]:
    if validation.kind != "incremental":
        cache.file_signatures = dict(validation.file_signatures or {})
        return (), None

    current, signatures, reason = _refresh_base(
        cache, validation.changed_paths, max_source_bytes=max_source_bytes,
        build_index_fn=build_index_fn,
    )
    if current is None or signatures is None:
        return (), reason or "scan_unstable"
    cache.index = current
    cache.file_signatures = signatures
    return validation.changed_paths, None


def _resolve_overlay(
    cache: RuntimeIndexCache,
    priority_paths: Sequence[str],
    *,
    max_source_bytes: int,
    build_index_fn: Callable[..., StructuralIndex],
) -> _OverlayResolution:
    entries = cache.index.scan_entries
    if entries is None:
        return _OverlayResolution((), None, None, error="scan_unstable")

    base_paths = {module.path for module in cache.index.modules}
    focus_paths = _candidate_priorities(cache.index.root, priority_paths, entries)
    overlay_paths = tuple(path for path in focus_paths if path not in base_paths)
    focus_changed = overlay_paths != cache.overlay_paths
    if (
        len(overlay_paths) > MAX_OVERLAY_FILES
        or len(base_paths) + len(overlay_paths) > MAX_FILES + MAX_OVERLAY_FILES
    ):
        return _OverlayResolution(
            overlay_paths, None, None, error="overlay_limit", focus_changed=focus_changed,
        )

    overlay = cache.overlay_index
    revision = cache.overlay_revision
    changed_paths: tuple[str, ...] = ()
    rebuilt = False
    if focus_changed:
        overlay, revision, error = _build_overlay_index(
            cache, overlay_paths, max_source_bytes=max_source_bytes,
            build_index_fn=build_index_fn,
        )
        if error is not None:
            return _OverlayResolution(
                overlay_paths, None, None, error=error, focus_changed=True,
            )
        rebuilt = bool(overlay_paths)
    elif overlay_paths:
        validation = _validate_overlay(cache, max_source_bytes)
        if validation.kind == "incremental":
            overlay, revision, error = _build_overlay_index(
                cache,
                overlay_paths,
                max_source_bytes=max_source_bytes,
                build_index_fn=build_index_fn,
                previous_overlay=overlay,
                changed_paths=validation.changed_paths,
            )
            if error is not None:
                return _OverlayResolution(
                    overlay_paths, None, None, error=error, focus_changed=False,
                )
            changed_paths = validation.changed_paths
            rebuilt = True
        elif validation.kind == "miss":
            overlay, revision, error = _build_overlay_index(
                cache, overlay_paths, max_source_bytes=max_source_bytes,
                build_index_fn=build_index_fn,
            )
            if error is not None:
                return _OverlayResolution(
                    overlay_paths, None, None, error=error, focus_changed=False,
                )
            rebuilt = True

    if _source_bytes(cache.index) + (_source_bytes(overlay) if overlay is not None else 0) > max_source_bytes:
        return _OverlayResolution(
            overlay_paths, None, None, error="limits", focus_changed=focus_changed,
        )
    return _OverlayResolution(
        overlay_paths, overlay, revision, changed_paths, rebuilt,
        focus_changed=focus_changed,
    )


def _overlay_reuse_outcome(
    *,
    focus_changed: bool,
    overlay_changed_paths: tuple[str, ...],
    overlay_rebuilt: bool,
) -> tuple[str, str]:
    if focus_changed:
        return "warm_overlay", "overlay_focus_changed"
    if overlay_changed_paths:
        return "warm_overlay", "overlay_file_digest_changed"
    if overlay_rebuilt:
        return "warm_overlay", "overlay_rebuilt"
    return "warm_overlay", "overlay_reused"


def _base_reuse_outcome(
    cache: RuntimeIndexCache,
    *,
    base_changed_paths: tuple[str, ...],
    priority_mismatch: bool,
) -> tuple[str, str | None]:
    if base_changed_paths:
        reason = (
            "scope_priority_ignored_complete_scan_file_digest_changed"
            if priority_mismatch and _catalog_complete(cache.index)
            else "file_digest_changed"
        )
        return "incremental", reason
    if not priority_mismatch:
        return "warm", None
    if _catalog_complete(cache.index):
        return "warm", "scope_priority_ignored_complete_scan"
    return "warm", "scope_priority_ignored_truncated_base"


def _reuse_outcome(
    cache: RuntimeIndexCache,
    *,
    focus_changed: bool,
    overlay_paths: tuple[str, ...],
    overlay_changed_paths: tuple[str, ...],
    overlay_rebuilt: bool,
    base_changed_paths: tuple[str, ...],
    priority_mismatch: bool,
) -> tuple[str, str | None]:
    if overlay_paths:
        return _overlay_reuse_outcome(
            focus_changed=focus_changed,
            overlay_changed_paths=overlay_changed_paths,
            overlay_rebuilt=overlay_rebuilt,
        )
    if focus_changed:
        return ("incremental" if base_changed_paths else "warm"), "overlay_cleared"
    return _base_reuse_outcome(
        cache,
        base_changed_paths=base_changed_paths,
        priority_mismatch=priority_mismatch,
    )


def _reuse_cached(
    cache: RuntimeIndexCache,
    validation: _Validation,
    *,
    priority_mismatch: bool,
    priority_paths: Sequence[str],
    requested_priorities: tuple[str, ...],
    scope_key: str,
    parser_version: str,
    max_source_bytes: int,
    build_index_fn: Callable[..., StructuralIndex],
) -> tuple[IndexLease | None, str | None]:
    if cache.legacy_priority:
        return _reuse_legacy_priority(
            cache,
            validation,
            priority_mismatch=priority_mismatch,
            requested_priorities=requested_priorities,
            scope_key=scope_key,
            parser_version=parser_version,
            max_source_bytes=max_source_bytes,
            build_index_fn=build_index_fn,
        )

    base_changed_paths, error = _refresh_cached_base(
        cache, validation, max_source_bytes=max_source_bytes,
        build_index_fn=build_index_fn,
    )
    if error is not None:
        return None, error

    overlay_state = _resolve_overlay(
        cache, priority_paths, max_source_bytes=max_source_bytes,
        build_index_fn=build_index_fn,
    )
    if overlay_state.error is not None:
        return None, overlay_state.error

    combined = combine_structural_indexes(
        cache.index,
        overlay_state.index,
        overlay_paths=overlay_state.paths,
        priority_paths=priority_paths,
    )
    if overlay_state.focus_changed:
        cache.analysis = None
    cache.overlay_paths = overlay_state.paths
    cache.overlay_index = overlay_state.index
    cache.overlay_revision = overlay_state.revision
    cache.priority_paths = ()
    cache.last_priority_paths = requested_priorities
    cache.structural_revision = structural_revision(
        cache.index,
        scope_key=scope_key,
        parser_version=parser_version,
        priority_paths=(),
    )

    status, reason = _reuse_outcome(
        cache,
        focus_changed=overlay_state.focus_changed,
        overlay_paths=overlay_state.paths,
        overlay_changed_paths=overlay_state.changed_paths,
        overlay_rebuilt=overlay_state.rebuilt,
        base_changed_paths=base_changed_paths,
        priority_mismatch=priority_mismatch,
    )
    changed_paths = tuple(sorted(
        set(base_changed_paths).union(overlay_state.changed_paths)
    ))
    return IndexLease(
        combined,
        cache,
        changed_paths,
        incremental=bool(changed_paths) and not overlay_state.focus_changed,
        reuse_status=status,
        reuse_reason=reason,
    ), None


__all__ = [
    "CACHE_ATTR", "IndexLease", "MODE_ENV", "RuntimeIndexCache",
    "VALIDATION_CAP_MIB_ENV", "acquire_index", "index_source_bytes",
    "reuse_enabled", "structural_revision", "validation_source_byte_limit",
    "validation_source_cap_mib",
]

"""Shared read execution, range-aware reuse, and live receipt closure.

React owns path authorization and annotations. This module normalizes the request, delivers the
effective source range, records the contract fields on the receipt before it is journaled, and
keeps the native live dispatch and its replay tests on one executor.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kernel_nav_basics import missing_path_message
from kernel_navigation_pages import directory_page
from kernel_source_delivery import (
    READ_CONTRACT_FIELDS, NormalizedReadRequest, ReadArgumentError,
    deliver_source, normalize_read_request,
)
from kernel_launch_identity_cost import ENVIRONMENT, UNKNOWN, PhaseMeter


def read_limit(args: object) -> int:
    """Return the actual line budget after the canonical range is resolved."""
    return normalize_read_request(args).max_lines


@dataclass
class _ReadSourceCacheEntry:
    content: bytes | None
    revision: str
    signature: tuple[int, int, int, int, int]
    workspace_generation: str | None
    pages: OrderedDict[str, dict[str, object]] = field(default_factory=OrderedDict)


_READ_CACHE_MAX_FILES = 16
_READ_CACHE_MAX_FILE_BYTES = 1_000_000
_READ_CACHE_MAX_TOTAL_BYTES = 4_000_000
_READ_CACHE_MAX_PAGES_PER_FILE = 8


def _source_identity(root: Path, path: Path) -> str:
    # Identity is cache bookkeeping only: it must never raise before the read's own failure is recorded as an
    # UNKNOWN outcome (10-08 regression: test_read_delivery_retains_explicit_environment_and_unknown_outcomes).
    try:
        base = root.resolve()
    except (OSError, AttributeError, TypeError):
        base = root
    try:
        relative = path.resolve().relative_to(base).as_posix()
    except (OSError, ValueError, AttributeError, TypeError):
        try:
            relative = str(path.resolve())
        except (OSError, AttributeError, TypeError):
            relative = str(path)
    return f"{base}::{relative}"


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size),
            int(stat.st_mtime_ns), int(stat.st_ctime_ns))


def _workspace_generation(measurement: object) -> str | None:
    for name in ("workspace_revision", "current_revision"):
        try:
            value = getattr(measurement, name, None)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            continue
        if value is not None:
            return str(value)
    return None


def _read_cache(measurement: object) -> OrderedDict[str, _ReadSourceCacheEntry] | None:
    try:
        cache = getattr(measurement, "_read_contract_source_cache", None)
        if isinstance(cache, OrderedDict):
            return cache
        cache = OrderedDict()
        setattr(measurement, "_read_contract_source_cache", cache)
        return cache
    except (AttributeError, TypeError):
        return None


def _cached_source(measurement: object, identity: str, signature: tuple[int, int, int, int, int],
                   generation: str | None) -> _ReadSourceCacheEntry | None:
    cache = _read_cache(measurement)
    if cache is None:
        return None
    entry = cache.get(identity)
    if (entry is None or entry.signature != signature
            or entry.workspace_generation != generation):
        if entry is not None:
            del cache[identity]
        return None
    cache.move_to_end(identity)
    return entry


def _store_source(measurement: object, identity: str, content: bytes,
                  signature: tuple[int, int, int, int, int],
                  generation: str | None) -> _ReadSourceCacheEntry | None:
    cache = _read_cache(measurement)
    if cache is None:
        return None
    previous = cache.get(identity)
    pages = (previous.pages if previous is not None
             and previous.signature == signature
             and previous.workspace_generation == generation else OrderedDict())
    entry = _ReadSourceCacheEntry(
        content=content if len(content) <= _READ_CACHE_MAX_FILE_BYTES else None,
        revision="sha256:" + hashlib.sha256(content).hexdigest(),
        signature=signature,
        workspace_generation=generation,
        pages=pages,
    )
    cache[identity] = entry
    cache.move_to_end(identity)
    while (len(cache) > _READ_CACHE_MAX_FILES
           or sum(len(item.content or b"") for item in cache.values())
           > _READ_CACHE_MAX_TOTAL_BYTES):
        cache.popitem(last=False)
    return entry if cache.get(identity) is entry else None


def _page_request_fingerprint(request: NormalizedReadRequest, path: str,
                              revision: str) -> str:
    payload = {
        "arguments": dict(request.identity_arguments),
        "path": path,
        "revision": revision,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cached_page_for_request(
        entry: _ReadSourceCacheEntry, request: NormalizedReadRequest,
        path: str) -> tuple[str, dict[str, object]] | None:
    fingerprint = _page_request_fingerprint(request, path, entry.revision)
    for key, page in reversed(entry.pages.items()):
        if page.get("request_fingerprint") == fingerprint:
            return key, page
    return None


def _page_cache_key(request: NormalizedReadRequest, path: str, revision: str,
                    metadata: Mapping[str, object], content: str) -> str:
    effective = metadata.get("effective_range")
    effective = dict(effective) if isinstance(effective, Mapping) else {
        "start_line": metadata.get("start_line"),
        "end_line": metadata.get("selected_end_line"),
        "byte_start": metadata.get("byte_start"),
        "byte_end": metadata.get("byte_end"),
    }
    payload = {
        "arguments": dict(request.identity_arguments),
        "path": path,
        "revision": revision,
        "effective_interval": effective,
        "content_sha256": "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _visible_in_messages(messages: Sequence[Mapping[str, object]] | None,
                         prior_text: object) -> bool:
    if not isinstance(prior_text, str) or not prior_text:
        return False
    for message in messages or ():
        content = message.get("content")
        if isinstance(content, str) and prior_text in content:
            return True
    return False


def _dropped_range_withheld(
        measurement: object, relative_path: str, effective_range: object,
) -> bool:
    notice = getattr(measurement, "_request_dropped_ranges_notice", None)
    if not isinstance(notice, str) or "[Request diet: dropped source ranges]" not in notice:
        return False
    bounds = effective_range if isinstance(effective_range, Mapping) else None
    start = bounds.get("start_line") if bounds is not None else None
    end = bounds.get("end_line") if bounds is not None else None
    has_bounds = (
        type(start) is int and type(end) is int and start >= 1 and end >= start
    )
    for line in notice.splitlines():
        match = re.match(r"^- (.+):(\d+)-(\d+) revision=\S+ reason=\S+$", line.strip())
        if match is None:
            continue
        path, dropped_start, dropped_end = match.groups()
        if path.replace("\\", "/") != relative_path:
            continue
        if not has_bounds:
            return True
        # Read metadata uses an exclusive end line; diet notices use inclusive ends.
        if start < int(dropped_end) + 1 and int(dropped_start) < end:
            return True
    return False


def _withheld_from_wire(
        measurement: object, root: Path, p: Path, effective_range: object = None,
) -> bool:
    """True when the exact requested source span is absent from this request wire."""
    raw_paths = getattr(measurement, "_request_deferred_source_paths", ())
    try:
        relative = p.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return False
    deferred_paths = (
        {str(path).replace("\\", "/").strip() for path in raw_paths}
        if isinstance(raw_paths, Sequence)
        and not isinstance(raw_paths, (str, bytes, bytearray))
        else set()
    )
    return relative in deferred_paths or _dropped_range_withheld(
        measurement, relative, effective_range,
    )


def _reuse_text(path: str, revision: str, requested: object,
                effective: object) -> str:
    path_label = path[:120]
    requested_label = json.dumps(requested, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"))[:120]
    effective_label = json.dumps(effective, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"))[:120]
    text = (f"[read reuse] path={path_label} revision={revision[:80]} "
            f"requested={requested_label} effective={effective_label}; "
            "the source bytes may be absent from the current request until final visibility qualification; this result omits duplicate bytes.")
    return text if len(text) <= 512 else text[:509] + "..."


def _retention_revision(measurement: object, step: object) -> int:
    for name in ("workspace_revision", "current_revision"):
        value = getattr(measurement, name, None)
        try:
            revision = int(value)
        except (TypeError, ValueError):
            continue
        if revision >= 0:
            return revision
    try:
        return max(0, int(step))
    except (TypeError, ValueError):
        return 0


def _observe_retained_source(
        retained: object, path: str, content: bytes, *, measurement: object,
        step: object) -> None:
    """Adapt legacy byte retention and world-fact text retention at one seam."""
    observer = getattr(retained, "observe_source", None)
    if not callable(observer):
        raise TypeError("retained source store has no observe_source method")
    evidence_ref = f"read@{step}"
    try:
        parameters = inspect.signature(observer).parameters
    except (TypeError, ValueError):
        parameters = {}
    if {"path", "content", "revision", "evidence_ref"} <= set(parameters):
        observer(
            path=path,
            content=content.decode("utf-8", errors="replace"),
            revision=_retention_revision(measurement, step),
            evidence_ref=evidence_ref,
        )
        return
    observer(path, content, evidence_ref)


def _ordinary_read_receipt(step: object, raw: str, start: int, reason: str) -> dict[str, object]:
    """Keep a chargeable read fact even when no source bytes exist to reuse."""
    from kernel_read_reuse import ORDINARY_READ, ReadReuseReceipt

    ordinal = int(step) if str(step).isdigit() else 0
    return ReadReuseReceipt(
        f"read:{step}:{raw}:{start}", ordinal, ORDINARY_READ, None, b"",
        reason=reason,
    ).as_dict()


def _record_requested_byte_delivery(
        delivery: MutableMapping[str, object] | None, *, status: str,
        content: str = "", metadata: Mapping[str, object] | None = None,
) -> None:
    """Record bytes returned by this request, excluding proof-backed reuse stubs."""
    from kernel_gate_switches import deliver_requested_bytes_enabled

    if delivery is None or not deliver_requested_bytes_enabled():
        return
    delivered = content.encode("utf-8", errors="replace") if status == "DELIVERED" else b""
    delivery.update({
        "delivery_status": status,
        "requested_bytes_delivered": len(delivered),
    })
    page_digest = (metadata or {}).get("page_bytes_sha256")
    if isinstance(page_digest, str) and page_digest:
        delivery["requested_bytes_sha256"] = page_digest
    elif status == "DELIVERED":
        delivery["requested_bytes_sha256"] = (
            "sha256:" + hashlib.sha256(delivered).hexdigest())


def _stale_coordinate_result(expected: object, current: str, *, display: str,
                             delivery: MutableMapping[str, object] | None,
                             step: object, raw: str, start: int
                             ) -> tuple[str, str, None] | None:
    if not isinstance(expected, str) or not expected or expected == current:
        return None
    if delivery is not None:
        delivery.update({
            "host_outcome": "STALE", "result": "STALE_REVISION",
            "expected_revision": expected, "revision": current,
            "clip_reason": "stale_coordinate", "effective_range": None,
            "next_cursor": None, "byte_length": 0,
            "read_reuse": _ordinary_read_receipt(step, raw, start, "stale_coordinate"),
        })
    path = display[:160]
    observation = (f"[read stale] path={path} expected_revision={expected} "
                   f"current_revision={current}; coordinates expired, refresh the source outline.")
    return "STALE_REVISION", observation, None


def execute_read(root: Path, p: Path, raw: str, display: str, start: int, *,
                 args: object, measurement: object, step: object,
                 policy: object | None = None,
                 delivery: MutableMapping[str, object] | None = None,
                 visible_messages: Sequence[Mapping[str, object]] | None = None,
                 page_line_limit: int | None = None,
                 page_byte_limit: int | None = None,
                 ) -> tuple[str, str, bytes | None]:
    """Read ``p`` exactly as the runner does.  Returns (result_class, observation, bytes)."""
    read_bytes: bytes | None = None
    from kernel_reread_result_truth import enabled as result_truth_enabled, result_note
    result_truth = result_truth_enabled()
    if delivery is not None:
        # A requested read is chargeable even when authorization or the host
        # reports a typed failure; callers must not silently erase its receipt.
        delivery["budget_charge"] = 1
    meter = getattr(measurement, "host_cost_meter", None)
    if not isinstance(meter, PhaseMeter):
        meter = PhaseMeter()
        try:
            setattr(measurement, "host_cost_meter", meter)
        except (AttributeError, TypeError):
            pass
    started = time.perf_counter_ns()
    delivery_elapsed_ns = 0
    delivery_segment_started: int | None = started
    try:
        request = normalize_read_request(
            args, default_start_line=start + 1,
            page_line_limit=page_line_limit,
            page_byte_limit=page_byte_limit,
        )
        if not p.exists():
            if delivery is not None:
                delivery.update({"host_outcome": UNKNOWN, "failure_class": "missing"})
                delivery["read_reuse"] = _ordinary_read_receipt(step, raw, start, "source_missing")
            return "MISSING", missing_path_message(root, display), None
        if p.is_dir():
            page = directory_page(
                p, display,
                cursor=args.get("cursor") if isinstance(args, Mapping) else None,
                requested_page_size=(args.get("page_size") if isinstance(args, Mapping) else None),
                policy=policy,
            )
            if delivery is not None:
                delivery.update(page.metadata)
                delivery["host_outcome"] = "OBSERVED"
                delivery["read_reuse"] = _ordinary_read_receipt(step, raw, start, "directory_listing")
            return "DIRECTORY", page.text, None
        identity = _source_identity(root, p)
        generation = _workspace_generation(measurement)
        try:
            signature = _file_signature(p)
        except (OSError, AttributeError, TypeError):  # cache key only; the read records its own failure
            signature = None
        expected_revision = request.arguments.get("expected_revision")
        fresh_revision_bytes: bytes | None = None
        if expected_revision:
            # Revision-pinned reads bypass stat-keyed content so coordinates
            # are checked against current bytes, even when a cache has a page.
            fresh_revision_bytes = p.read_bytes()
            try:
                signature = _file_signature(p)
            except OSError:
                signature = None
            cache_entry = (_store_source(
                measurement, identity, fresh_revision_bytes, signature, generation,
            ) if signature is not None else None)
        else:
            cache_entry = (_cached_source(measurement, identity, signature, generation)
                           if signature is not None else None)
        cached_page = (_cached_page_for_request(cache_entry, request, identity)
                       if cache_entry is not None else None)
        if cache_entry is not None:
            stale = _stale_coordinate_result(
                expected_revision, cache_entry.revision, display=display,
                delivery=delivery, step=step, raw=raw, start=start,
            )
            if stale is not None:
                return stale
        if cache_entry is not None and cache_entry.content is None and cached_page is not None:
            cached_key, page = cached_page
            cached_metadata = dict(page.get("metadata", {}))
            cached_content = str(page.get("content", ""))
            cached_observation = str(page.get("observation", ""))
            effective = cached_metadata.get("effective_range")
            dedup_receipt: dict[str, object] = {
                "schema": "stateframe-read-dedup/1",
                "status": "CACHED_BYTES",
                "key": cached_key,
                "revision": cache_entry.revision,
                "effective_range": effective,
            }
            observation = cached_observation
            from kernel_gate_switches import deliver_requested_bytes_enabled
            deliver_requested = deliver_requested_bytes_enabled() or result_truth
            if (not deliver_requested and not _withheld_from_wire(
                    measurement, root, p, cached_metadata.get("effective_range"))
                    and _visible_in_messages(visible_messages, cached_observation)):
                dedup_receipt["status"] = "UNCHANGED_VISIBLE"
                observation = _reuse_text(
                    display, cache_entry.revision,
                    cached_metadata.get("requested_range"), effective,
                )
            cached_metadata["read_dedup"] = dedup_receipt
            if delivery is not None:
                delivery.update(cached_metadata)
                delivery["host_outcome"] = "OBSERVED"
                delivery["read_reuse"] = _ordinary_read_receipt(
                    step, raw, start, "cached_exact_page")
                _record_requested_byte_delivery(
                    delivery,
                    status="DELIVERED" if observation == cached_observation else "REUSED_PROVEN",
                    content=cached_content, metadata=cached_metadata,
                )
            if measurement.step_ledger is not None:
                try:
                    evidence_path = p.resolve().relative_to(root.resolve()).as_posix()
                except ValueError:
                    evidence_path = str(p.resolve())
                measurement.step_ledger.read_evidence(
                    path=evidence_path, requested_path=raw,
                    revision=cache_entry.revision, content=cached_content,
                    start_line=int(cached_metadata.get("start_line", start + 1)),
                    returned_lines=(int(cached_metadata["selected_end_line"])
                                    - int(cached_metadata.get("start_line", start + 1)) + 1),
                    total_lines=int(cached_metadata.get("total_lines", 0)),
                    delivery_status=(str(delivery.get("delivery_status"))
                                     if delivery is not None and deliver_requested else None),
                    delivered_bytes=(int(delivery.get("requested_bytes_delivered", 0))
                                     if delivery is not None and deliver_requested else None))
            if str(cached_metadata.get("result_class", "OK")) == "OK":
                try:
                    from kernel_read_reuse_visibility import remember_read_span
                    remember_read_span(
                        measurement, root, p, cache_entry.revision,
                        cached_metadata, cached_content,
                    )
                except (AttributeError, OSError, TypeError, ValueError):
                    pass
            if str(cached_metadata.get("result_class", "OK")) == "OK" and result_truth:
                note = result_note(measurement, identity, generation, cache_entry.revision, step)
                if note:
                    observation += "\n" + note
            return str(cached_metadata.get("result_class", "OK")), observation, None

        file_cache_hit = False
        if cache_entry is not None and cache_entry.content is not None:
            file_cache_hit = True
            read_bytes = cache_entry.content
        else:
            if fresh_revision_bytes is not None:
                read_bytes = fresh_revision_bytes
            else:
                read_bytes = p.read_bytes()
                try:
                    after_signature = _file_signature(p)
                except OSError:
                    after_signature = None
                if signature is not None and after_signature == signature:
                    cache_entry = _store_source(
                        measurement, identity, read_bytes, signature, generation)
                else:
                    signature = after_signature
        current_revision = cache_entry.revision if cache_entry is not None else (
            "sha256:" + hashlib.sha256(read_bytes).hexdigest())
        stale = _stale_coordinate_result(
            expected_revision, current_revision, display=display,
            delivery=delivery, step=step, raw=raw, start=start,
        )
        if stale is not None:
            return stale
        now = time.perf_counter_ns()
        delivery_elapsed_ns += now - (delivery_segment_started or now)
        delivery_segment_started = now
        if measurement.retained is not None:
            # Retention has its own extraction/binding/journaling meters; do not
            # attribute its work to the request-delivery phase.
            delivery_segment_started = None
            retained_meter = getattr(measurement.retained, "cost_meter", None)
            if isinstance(retained_meter, PhaseMeter) and retained_meter is not meter:
                # Source phases and read phases share one bounded receipt; this
                # avoids merging cumulative snapshots on every reread.
                try:
                    measurement.retained.cost_meter = meter
                except (AttributeError, TypeError):
                    pass
            try:
                retained_path = p.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                measurement.journal.emit('retained_external_read_unbound', requested_path=str(p))
            else:
                _observe_retained_source(
                    measurement.retained,
                    retained_path,
                    read_bytes,
                    measurement=measurement,
                    step=step,
                )
                trace = getattr(measurement.retained, "trace", ())
                if delivery is not None and trace:
                    latest = trace[-1]
                    if isinstance(latest, Mapping):
                        parser_status = latest.get("parser_status")
                        if parser_status is not None:
                            delivery["parser_status"] = parser_status
                            delivery["source_observation_outcome"] = (
                                "UNKNOWN" if parser_status == UNKNOWN else "OBSERVED"
                            )
        delivery_segment_started = time.perf_counter_ns()
        reuse_dispatch = None
        try:
            from kernel_read_reuse import runtime_dispatch
            options = request.arguments
            range_requested = start != 0 or any(
                options.get(key) not in (None, "", 0)
                for key in ("line", "start_line", "end_line", "symbol", "cursor",
                            "column", "max_lines", "max_chars", "page_size")
            )
            relative = p.resolve().relative_to(root.resolve()).as_posix()
            reuse_dispatch = runtime_dispatch(
                measurement, relative, start,
                range_requested=range_requested,
                current_content=read_bytes,
            )
        except (OSError, TypeError, ValueError, RuntimeError):
            reuse_dispatch = None
        shaped: dict[str, object] = {}
        source = deliver_source(read_bytes, display, start, args=request, metadata_out=shaped)
        if source.result_class == "OK":
            try:
                from kernel_read_reuse_visibility import remember_read_span
                remember_read_span(
                    measurement, root, p, source.metadata.get("revision"),
                    source.metadata, source.content,
                )
            except (AttributeError, OSError, TypeError, ValueError):
                pass
        observation = source.text
        dedup_key = _page_cache_key(
            request, identity, str(source.metadata.get("revision") or ""),
            source.metadata, source.content)
        duplicate = (cache_entry.pages.get(dedup_key)
                     if cache_entry is not None and source.result_class == "OK" else None)
        dedup_receipt: dict[str, object] = {
            "schema": "stateframe-read-dedup/1",
            "status": "CACHE_DELIVERED" if file_cache_hit else "EXECUTED",
            "key": dedup_key,
            "revision": source.metadata.get("revision"),
            "effective_range": source.metadata.get("effective_range"),
        }
        from kernel_gate_switches import deliver_requested_bytes_enabled
        deliver_requested = deliver_requested_bytes_enabled() or result_truth
        if duplicate is not None:
            prior_text = duplicate.get("observation")
            if (not deliver_requested and not _withheld_from_wire(
                    measurement, root, p, source.metadata.get("effective_range"))
                    and _visible_in_messages(visible_messages, prior_text)):
                dedup_receipt["status"] = "UNCHANGED_VISIBLE"
                observation = _reuse_text(
                    display, str(source.metadata.get("revision") or ""),
                    source.metadata.get("requested_range"),
                    source.metadata.get("effective_range"),
                )
            elif file_cache_hit:
                dedup_receipt["status"] = "CACHED_BYTES"
            else:
                dedup_receipt["status"] = "REEXECUTED"
        if cache_entry is not None and source.result_class == "OK":
            cache_entry.pages[dedup_key] = {
                "observation": source.text,
                "content": source.content,
                "revision": source.metadata.get("revision"),
                "effective_range": source.metadata.get("effective_range"),
                "metadata": dict(source.metadata),
                "request_fingerprint": _page_request_fingerprint(
                    request, identity, str(source.metadata.get("revision") or "")),
            }
            cache_entry.pages.move_to_end(dedup_key)
            while len(cache_entry.pages) > _READ_CACHE_MAX_PAGES_PER_FILE:
                cache_entry.pages.popitem(last=False)
        shaped["read_dedup"] = dedup_receipt
        if reuse_dispatch is not None and reuse_dispatch.exact_bytes_delivered:
            shaped["read_cache_dispatch"] = reuse_dispatch.as_dict()
        if delivery is not None:
            delivery.update(shaped)
            delivery["host_outcome"] = (
                "OBSERVED" if source.result_class in {"OK", "OUT_OF_RANGE"}
                else "UNKNOWN"
            )
            if source.result_class != "OK":
                delivery["read_reuse"] = _ordinary_read_receipt(
                    step, raw, start, f"host_result:{source.result_class}")
            if source.result_class == "OK":
                exact_reuse = False
                try:
                    from kernel_read_reuse import ReadRequest, content_handle, decide_read
                    relative = p.resolve().relative_to(root.resolve()).as_posix()
                    generation = getattr(measurement, "workspace_revision",
                                         getattr(measurement, "current_revision", step))
                    handle = content_handle(
                        relative, read_bytes or b"",
                        workspace_scope=str(getattr(measurement, "workspace_scope", root.resolve())),
                        workspace_generation=str(generation),
                        decoding=str(source.metadata.get("decoding", "utf-8")),
                        presentation_version=str(source.metadata.get("presentation_version", "read-page/1")),
                    )
                    request = ReadRequest(
                        f"read:{step}:{relative}:{start}", relative,
                        int(source.metadata.get("byte_start", 0)),
                        int(source.metadata.get("byte_end", len(read_bytes or b""))),
                        workspace_scope=handle.workspace_scope,
                        workspace_generation=handle.workspace_generation,
                        decoding=handle.decoding,
                        presentation_version=handle.presentation_version,
                    )
                    receipt = decide_read(
                        handle, request,
                        getattr(measurement, "read_delivery_proofs", ()),
                        ordinal=int(step) if str(step).isdigit() else 0,
                    )
                    delivery["read_reuse"] = receipt.as_dict()
                    exact_reuse = receipt.exact_bytes_delivered
                except (OSError, TypeError, ValueError, RuntimeError):
                    # Reuse is advisory; inability to construct its proof never
                    # suppresses the ordinary exact read already in progress.
                    delivery["read_reuse"] = {
                        "schema": "stateframe-read-reuse/1", "status": "ORDINARY_READ",
                        "reason": "reuse_planner_unavailable", "budget_charge": 1,
                    }
                if deliver_requested:
                    if exact_reuse and not result_truth:
                        reuse = delivery.get("read_reuse")
                        reuse_status = (reuse.get("status")
                                        if isinstance(reuse, Mapping) else None)
                        dedup_receipt["status"] = (
                            "UNCHANGED_VISIBLE" if reuse_status == "UNCHANGED_VISIBLE"
                            else "CACHED_BYTES")
                        observation = _reuse_text(
                            display, str(source.metadata.get("revision") or ""),
                            source.metadata.get("requested_range"),
                            source.metadata.get("effective_range"),
                        )
                        _record_requested_byte_delivery(
                            delivery, status="REUSED_PROVEN",
                            content=source.content, metadata=source.metadata,
                        )
                    else:
                        _record_requested_byte_delivery(
                            delivery, status="DELIVERED",
                            content=source.content, metadata=source.metadata,
                        )
            elif deliver_requested:
                _record_requested_byte_delivery(delivery, status="WITHHELD")
        if measurement.step_ledger is not None:
            # Legacy information policy can permit external reads. Keep their
            # absolute identity without claiming workspace freshness.
            try:
                evidence_path = p.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                evidence_path = str(p.resolve())
            measurement.step_ledger.read_evidence(
                path=evidence_path, requested_path=raw,
                revision=str(source.metadata.get("revision") or "UNKNOWN"),
                content=source.content,
                start_line=int(source.metadata.get("start_line", start + 1)),
                returned_lines=(int(source.metadata["selected_end_line"])
                                - int(source.metadata.get("start_line", start + 1)) + 1
                                if source.result_class == "OK" else 0),
                total_lines=int(source.metadata.get("total_lines", 0)),
                delivery_status=(str(delivery.get("delivery_status"))
                                 if delivery is not None and deliver_requested else None),
                delivered_bytes=(int(delivery.get("requested_bytes_delivered", 0))
                                 if delivery is not None and deliver_requested else None))
        if delivery_segment_started is not None:
            delivery_elapsed_ns += time.perf_counter_ns() - delivery_segment_started
            delivery_segment_started = None
        if source.result_class == "OK" and result_truth:
            note = result_note(measurement, identity, _workspace_generation(measurement),
                               source.metadata.get("revision"), step)
            if note:
                observation += "\n" + note
        return source.result_class, observation, read_bytes
    except ReadArgumentError as error:
        repair = error.as_dict(path=raw)
        if delivery is not None:
            coordinate_fields = (
                "line", "start_line", "end_line", "cursor", "max_lines",
                "max_chars", "symbol", "column",
            )
            requested_range = ({key: args[key] for key in coordinate_fields if key in args}
                               if isinstance(args, Mapping) else {})
            delivery.update({
                "schema": repair["schema"], "argument_error": repair,
                "revision": None, "requested_range": requested_range,
                "effective_range": None, "next_cursor": None,
                "clip_reason": "invalid_request", "host_outcome": "OBSERVED",
                "failure_class": "read_contract",
                "byte_length": 0,
                "page_bytes_sha256": "sha256:" + hashlib.sha256(b"").hexdigest(),
                "read_reuse": _ordinary_read_receipt(step, raw, start, "invalid_read_arguments"),
            })
        return "INVALID_ARGUMENT", json.dumps(
            repair, sort_keys=True, ensure_ascii=False, separators=(",", ":")), None
    except OSError as error:
        if delivery is not None:
            delivery.update({
                "host_outcome": "ENVIRONMENT",
                "failure_class": "host",
                "error_type": type(error).__name__,
            })
            delivery.setdefault("read_reuse", _ordinary_read_receipt(step, raw, start, "host_error"))
        return "ERROR", f"Read failed for {raw}: {type(error).__name__}. Retry or choose another path.", read_bytes
    except TypeError as error:
        if delivery is not None:
            delivery.update({
                "host_outcome": ENVIRONMENT,
                "failure_class": "retention_contract",
                "error_type": type(error).__name__,
            })
            delivery.setdefault("read_reuse", _ordinary_read_receipt(step, raw, start, "retention_contract"))
        return "ERROR", f"Read failed for {raw}: {type(error).__name__}. Retry or choose another path.", read_bytes
    except RuntimeError as error:
        if delivery is not None:
            delivery.update({
                "host_outcome": ENVIRONMENT,
                "failure_class": "retention_or_delivery",
                "error_type": type(error).__name__,
            })
            delivery.setdefault("read_reuse", _ordinary_read_receipt(step, raw, start, "retention_or_delivery"))
        return "ERROR", f"Read failed for {raw}: {type(error).__name__}. Retry or choose another path.", read_bytes
    except ValueError as error:
        # Directory pagination is a model-visible argument boundary.  Letting a
        # stale/foreign cursor escape as an unknown effect strands the action
        # continuation; return a typed repair that can be retried without losing
        # the normal read receipt.
        reason = " ".join(str(error).split())[:240] or "the continuation is invalid"
        code = "INVALID_READ_CURSOR" if "cursor" in reason.lower() or "continuation" in reason.lower() else "INVALID_READ_ARGUMENT"
        repair = {
            "schema": "stateframe-read-argument-repair/1",
            "status": "REJECTED",
            "code": code,
            "tool": "read",
            "field": "cursor" if code == "INVALID_READ_CURSOR" else "arguments",
            "path": raw,
            "reason": reason,
            "repair": {
                "action": "read",
                "arguments": {
                    "path": raw,
                    "page_size": (args.get("page_size")
                                   if isinstance(args, Mapping) else None),
                },
                "instruction": "retry without cursor or use the latest continuation returned for this path",
            },
        }
        if delivery is not None:
            delivery.update({
                "host_outcome": "UNKNOWN",
                "failure_class": "parser_or_argument",
            })
            delivery["read_reuse"] = _ordinary_read_receipt(step, raw, start, "parser_or_argument")
        return "INVALID_ARGUMENT", json.dumps(repair, sort_keys=True, ensure_ascii=False), read_bytes
    finally:
        if delivery_segment_started is not None:
            delivery_elapsed_ns += time.perf_counter_ns() - delivery_segment_started
        if delivery_elapsed_ns <= 0:
            delivery_elapsed_ns = time.perf_counter_ns() - started
        meter.record(
            "request_delivery",
            delivery_elapsed_ns,
            count=1,
            items=len(read_bytes or b""),
        )


def emit_read_receipt(bolt: object, root: Path, p: Path, raw: str, start: int,
                      result: str, delivery: Mapping[str, object] | None = None) -> None:
    """E346: read receipts complete the action tape (re-read-rate vital)."""
    from kernel_tool_paths import receipt_target_path
    result_status = str(result).upper()
    if result_status in {
            "INVALID_ARGUMENT", "READ_CURSOR_INVALID", "READ_CURSOR_STALE",
            "SYMBOL_NOT_FOUND", "OUT_OF_RANGE", "MISSING"}:
        result_status = "REJECTED"
    elif result_status == "ERROR":
        result_status = "FAILED"
    elif result_status == "DIRECTORY":
        result_status = "OK"
    bolt.receipt("read", f"{raw}@{start + 1}", f"{raw}|{start}",
                 result_status,
                 detail=json.dumps(dict(delivery or {}), sort_keys=True,
                                     ensure_ascii=False, separators=(",", ":"))[:150],
                 target_path=receipt_target_path(root, p),
                 read_contract={field: (delivery or {}).get(field)
                                for field in READ_CONTRACT_FIELDS})
    receipts = getattr(bolt, "_receipts", None)
    if receipts is None:
        receipts = getattr(bolt, "receipts", None)
    if receipts and delivery:
        receipts[-1]["delivery"] = dict(delivery)


def dispatch_live_read(root: Path, p: Path, raw: str, display: str, start: int, *,
                       args: object, measurement: object, step: object,
                       bolt: object, policy: object | None, resolved_path: str,
                       visible_messages: Sequence[Mapping[str, object]] | None = None,
                       page_line_limit: int | None = None,
                       page_byte_limit: int | None = None,
                       ) -> tuple[str, str, bytes | None, dict[str, object]]:
    """Run the shared live read executor and close its receipt with the same delivery."""
    delivery: dict[str, object] = {}
    result, observation, source_bytes = execute_read(
        root, p, raw, display, start, args=args, measurement=measurement,
        step=step, policy=policy, delivery=delivery,
        visible_messages=visible_messages, page_line_limit=page_line_limit,
        page_byte_limit=page_byte_limit,
    )
    from kernel_read_once import note_for as read_once_note_for
    read_once_note = read_once_note_for(
        # Bind repeat tracking to the workspace file itself. ``resolved_path``
        # is the tool-facing relative name in the live repair loop.
        measurement, delivery, source_path=p,
    )
    from kernel_reread_result_truth import enabled as result_truth_enabled
    if read_once_note is not None and not result_truth_enabled():
        observation = f"{observation}\n{read_once_note}"
    from kernel_serve_reads import enabled as serve_reads_enabled, note_read_outcome
    if serve_reads_enabled():
        from kernel_gate_switches import deliver_requested_bytes_enabled
        bytes_read = (int(delivery.get("requested_bytes_delivered", 0) or 0)
                      if deliver_requested_bytes_enabled()
                      else len(source_bytes or b""))
        note_read_outcome(
            measurement, resolved_path, result,
            bytes_read=bytes_read,
        )
    emit_read_receipt(bolt, root, p, raw, start, result, delivery)
    return result, observation, source_bytes, delivery

"""Fail-open, request-local memoization for expensive StateFrame host facts."""
from __future__ import annotations

import hashlib
import os
import pickle
import copy
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import TypeVar


CACHE_FORMAT = 1
_T = TypeVar("_T")
_TRUE = frozenset({"1", "on", "true", "yes"})
# Request-local marker shared with StateFrameRuntime's host-check bundle cache.
_NO_BUNDLE = object()
_PER_CACHE_SWITCHES = {
    "JOURNAL_RECORDS": "STATEFRAME_HOST_CACHE_JOURNAL_RECORDS",
    "TURN_FACTS": "STATEFRAME_HOST_CACHE_TURN_FACTS",
}
_TURN_FACTS_V2_SWITCH = "STATEFRAME_HOST_CACHE_TURN_FACTS_V2"


@dataclass(frozen=True)
class _CacheEntry:
    format: int
    digest: str | None
    value: object


def _switch(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE


def cache_enabled(name: str) -> bool:
    """Respect the global kill switch and an independently controllable cache."""
    normalized = "".join(char if char.isalnum() else "_" for char in name.upper()).strip("_")
    try:
        cache_switch = _PER_CACHE_SWITCHES.get(
            normalized, f"STATEFRAME_HOST_CACHE_{normalized}")
        return (_switch("STATEFRAME_HOST_CACHE", "off")
                and _switch(cache_switch, "on"))  # global default OFF (BUILD r2EY cut rule); per-cache switches only narrow
    except Exception:
        return False


def optimized_turn_facts_enabled() -> bool:
    """Return whether the opt-in TurnFacts reuse path is active."""
    try:
        return (cache_enabled("TURN_FACTS")
                and _switch(_TURN_FACTS_V2_SWITCH, "off"))
    except Exception:
        return False


def _deeply_immutable(value: object) -> bool:
    """Recognize only exact builtin containers with immutable contents."""
    if type(value) in (type(None), bool, int, str, bytes):
        return True
    if type(value) is tuple or type(value) is frozenset:
        return all(_deeply_immutable(item) for item in value)
    return False


def _copy_for_cache(value: _T, *, fast_immutable: bool) -> _T:
    if fast_immutable and _deeply_immutable(value):
        return value
    return copy.deepcopy(value)


def _shadow_values_match(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if _deeply_immutable(left) and _deeply_immutable(right):
        return bool(left == right)
    try:
        return _entry_digest(left) == _entry_digest(right)
    except Exception:
        try:
            result = left == right
            return type(result) is bool and result
        except Exception:
            return False


def _notify_outcome(
        callback: Callable[[str], None] | None, outcome: str,
) -> None:
    if callback is None:
        return
    try:
        callback(outcome)
    except Exception:
        pass


def _entry_digest(value: object) -> str:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest()


def _emit_cache_event(journal: object | None, event: str, *, name: str,
                      reason: str | None = None) -> None:
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    try:
        fields: dict[str, object] = {
            "producer": "kernel_host_cache",
            "name": name[:80],
            "dispatch": False,
        }
        if reason is not None:
            fields["reason"] = reason[:80]
        emit(event, **fields)
    except Exception:
        # Telemetry is advisory; a journal failure must never affect the caller.
        pass


def _recompute(store: MutableMapping[object, object], key: object,
               compute: Callable[[], _T], *, name: str,
               journal: object | None, write_back: bool,
               fast_immutable: bool) -> _T:
    value = compute()
    if write_back:
        try:
            stored = _copy_for_cache(value, fast_immutable=fast_immutable)
            entry_digest = (
                None if fast_immutable and _deeply_immutable(value)
                else _entry_digest(stored)
            )
            store[key] = _CacheEntry(CACHE_FORMAT, entry_digest, stored)
        except Exception as error:
            _emit_cache_event(
                journal, "cache_bypass", name=name,
                reason=f"store_error:{type(error).__name__}")
    return value


def memo_by(
        store: MutableMapping[object, object], key: object,
        compute: Callable[[], _T], *, name: str,
        journal: object | None = None,
        outcome_callback: Callable[[str], None] | None = None,
) -> _T:
    """Read or populate one entry; every cache operation fails open to compute()."""
    fast_immutable = name == "TURN_FACTS" and optimized_turn_facts_enabled()
    try:
        present = key in store
        raw_entry = store[key] if present else None
    except Exception as error:
        _notify_outcome(outcome_callback, "miss")
        _emit_cache_event(
            journal, "cache_bypass", name=name,
            reason=f"key_error:{type(error).__name__}")
        # In particular, an unhashable key never reaches the cache store again.
        return compute()

    if not present:
        # Keep misses cheap: no journal emit or pickle digest; only mutable results
        # need a private copy for the cache.
        _notify_outcome(outcome_callback, "miss")
        value = compute()
        try:
            store[key] = _CacheEntry(
                CACHE_FORMAT, None,
                _copy_for_cache(value, fast_immutable=fast_immutable),
            )
        except Exception:  # noqa: BLE001 - cache store failure is a silent pass-through
            pass
        return value

    try:
        if not isinstance(raw_entry, _CacheEntry):
            raise ValueError("corrupt_entry")
        if raw_entry.format != CACHE_FORMAT:
            raise ValueError("format_mismatch")
        if raw_entry.digest is not None and raw_entry.digest != _entry_digest(raw_entry.value):
            raise ValueError("digest_mismatch")
        cached = _copy_for_cache(raw_entry.value, fast_immutable=fast_immutable)
    except Exception as error:
        _notify_outcome(outcome_callback, "miss")
        reason = str(error) if isinstance(error, ValueError) else type(error).__name__
        _emit_cache_event(journal, "cache_bypass", name=name, reason=reason)
        return _recompute(
            store, key, compute, name=name, journal=journal,
            write_back=True, fast_immutable=fast_immutable,
        )

    if _switch("STATEFRAME_HOST_CACHE_SHADOW", "off"):
        try:
            fresh = compute()
            if not _shadow_values_match(fresh, raw_entry.value):
                _notify_outcome(outcome_callback, "miss")
                _emit_cache_event(journal, "cache_mismatch", name=name, reason="shadow_value")
                return fresh
        except Exception as error:
            # A shadow failure is not a cache failure for the primary result.
            _emit_cache_event(
                journal, "cache_bypass", name=name,
                reason=f"shadow_error:{type(error).__name__}")
    _notify_outcome(outcome_callback, "hit")
    return cached  # type: ignore[return-value]


def clear_request_cache(runtime: object) -> None:
    """Drop request-local memoized facts and their invalidation snapshots."""
    try:
        setattr(runtime, "_host_audited_body_cache", {})
    except Exception:
        pass
    try:
        if getattr(runtime, "_host_check_bundle_cache", None) is _NO_BUNDLE:
            setattr(runtime, "_host_check_bundle_cache", None)
    except Exception:
        # A cache invalidation failure must not interrupt dispatch or journal work.
        pass
    if not cache_enabled("TURN_FACTS"):
        return
    try:
        setattr(runtime, "_host_fact_cache", {})
        setattr(runtime, "_host_cache_request_identity", None)
        setattr(runtime, "_host_cache_receipt_generation", None)
        setattr(runtime, "_host_cache_product_revision", None)
    except Exception:
        pass


def _request_identity(runtime: object) -> object:
    capture = getattr(runtime, "_active_capture", None)
    request_id = getattr(capture, "request_id", None) or getattr(runtime, "request_id", None)
    return request_id if request_id is not None else id(runtime)


def _request_outcome_callback(
        journal: object | None, runtime: object, *, enabled: bool,
) -> Callable[[str], None] | None:
    if not enabled:
        return None
    try:
        request_id = _request_identity(runtime)
    except Exception:
        return None
    if not isinstance(request_id, str) or not request_id:
        return None

    def record_outcome(outcome: str) -> None:
        _record_request_cache_outcome(journal, request_id, outcome)

    return record_outcome


def _product_revision(runtime: object, binding: object = _NO_BUNDLE) -> object:
    if binding is _NO_BUNDLE:
        binding_fn = getattr(runtime, "binding", None)
        binding = binding_fn() if callable(binding_fn) else getattr(runtime, "binding", None)
    return (
        binding.get("product_revision", binding.get("product", binding.get("artifact")))
        if isinstance(binding, Mapping)
        else getattr(runtime, "product_revision", binding)
    )


def _tool_names(runtime: object, tools: object | None) -> tuple[str, ...]:
    candidates = tools
    if (candidates is None or not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes, bytearray))):
        candidates = next((getattr(runtime, name, None) for name in (
            "_completion_owner_dispatch_tools", "_request_tools", "tools",
        ) if isinstance(getattr(runtime, name, None), Sequence)), ())
    names: list[str] = []
    for item in candidates if isinstance(candidates, Sequence) else ():
        if not isinstance(item, Mapping):
            continue
        function = item.get("function")
        name = function.get("name") if isinstance(function, Mapping) else item.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return tuple(names)


def _receipt_counts(runtime: object) -> tuple[tuple[str, int | None], ...]:
    owner = getattr(runtime, "owner", None)
    names = ("test_receipts", "host_check_receipts", "host_repair_check_receipts",
             "_receipts", "receipts")
    result: list[tuple[str, int | None]] = []
    for holder_name, holder in (("runtime", runtime), ("owner", owner)):
        if holder is None:
            continue
        for name in names:
            value = getattr(holder, name, None)
            if value is None:
                continue
            try:
                count = len(value)
            except (TypeError, AttributeError):
                count = None
            result.append((f"{holder_name}.{name}", count))
    return tuple(result)


def turn_facts_key(
        runtime: object, tools: object | None = None, *, reads_tools: bool = True,
        binding: object = _NO_BUNDLE, reads_menu: bool = True,
) -> tuple[object, ...]:
    """Build the explicit invalidation key for facts that live within one request."""
    request_identity = _request_identity(runtime)
    product_revision = _product_revision(runtime, binding)
    journal = getattr(runtime, "journal", None)
    cached_identity = getattr(runtime, "_host_cache_request_identity", None)
    generation_snapshot = getattr(runtime, "_host_cache_receipt_generation", None)
    identity_changed = cached_identity != request_identity
    if identity_changed or generation_snapshot is None:
        if (identity_changed and cached_identity is not None
                and optimized_turn_facts_enabled()):
            clear_request_cache(runtime)
        generation_snapshot = getattr(journal, "receipt_generation", None)
        setattr(runtime, "_host_cache_request_identity", request_identity)
        setattr(runtime, "_host_cache_receipt_generation", generation_snapshot)
        setattr(runtime, "_host_cache_product_revision", product_revision)
    live_generation = getattr(journal, "receipt_generation", generation_snapshot)
    key_generation = (
        live_generation if optimized_turn_facts_enabled() else generation_snapshot
    )
    stopped = get_sure = False
    if reads_menu:
        lock = getattr(runtime, "green_lock", None)
        try:
            from kernel_greenlock_menu_authority import _get_sure, _stopped
            stopped = _stopped(lock) if lock is not None else False
            get_sure = _get_sure(runtime)
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            stopped = bool(getattr(runtime, "_stopped", False))
            get_sure = getattr(runtime, "stateframe_get_sure_mode", "off") == "on"
    return (
        CACHE_FORMAT,
        request_identity,
        product_revision,
        key_generation,
        _receipt_counts(runtime),
        _tool_names(runtime, tools) if reads_tools else (),
        bool(stopped),
        bool(get_sure),
    )


def memo_runtime_fact(
        runtime: object, name: str, compute: Callable[[], _T], *,
        tools: object | None = None, reads_tools: bool = True,
        fact_key: object = "default",
) -> _T:
    """Memoize a state fact for this request; key errors recompute without raising."""
    if not cache_enabled(name):
        return compute()
    optimized = name == "TURN_FACTS" and optimized_turn_facts_enabled()
    journal = getattr(runtime, "journal", None)
    outcome_callback = _request_outcome_callback(
        journal, runtime, enabled=optimized)
    try:
        key_reads_tools = reads_tools if optimized else True
        request_key = turn_facts_key(
            runtime, tools, reads_tools=key_reads_tools)
        key = (name, fact_key, *request_key)
        caches = getattr(runtime, "_host_fact_cache", None)
        if not isinstance(caches, dict):
            caches = {}
            setattr(runtime, "_host_fact_cache", caches)
        store = caches.setdefault(name, {})
        if not isinstance(store, MutableMapping):
            raise TypeError("invalid_cache_store")
    except Exception as error:
        _notify_outcome(outcome_callback, "miss")
        _emit_cache_event(
            journal, "cache_bypass", name=name,
            reason=f"key_error:{type(error).__name__}")
        return compute()

    return memo_by(
        store, key, compute, name=name, journal=journal,
        outcome_callback=outcome_callback,
    )


def memo_runtime_body(
        runtime: object, fact_key: str, compute: Callable[[], _T], *,
        inputs: Callable[[], object], cacheable: Callable[[_T], bool] | None = None,
        binding: object = _NO_BUNDLE,
) -> _T:
    """Flag-free memo for pure bodies whose caller retains every audit effect.

    Keep one entry per fact, keyed by the existing request axes plus the full
    inputs: receipt counts alone cannot detect same-length receipt amendments.
    Cache faults silently recompute; emitting cache telemetry here would change
    the journal sequence this optimization is required to preserve.
    """
    try:
        # Verdict bodies do not read menus. The menu-key readers may themselves
        # derive readiness from a verdict, so invoking them would add effects or
        # re-enter this call. Actual body inputs are fingerprinted below.
        key = (*turn_facts_key(runtime, reads_tools=False, binding=binding, reads_menu=False),
               _entry_digest(inputs()))
        hash(key)
        caches = getattr(runtime, "_host_audited_body_cache", None)
        if not isinstance(caches, dict):
            caches = {}
            setattr(runtime, "_host_audited_body_cache", caches)
        store = caches.setdefault(fact_key, {})
        if not isinstance(store, dict):
            raise TypeError("invalid_body_cache_store")
        if key not in store:
            store.clear()
    except Exception:
        return compute()
    value = memo_by(store, key, compute, name="AUDITED_BODY")
    if cacheable is not None:
        try:
            if not cacheable(value):
                store.clear()
        except Exception:
            store.clear()
    return value


def request_cache_invalidation_changed(runtime: object) -> bool:
    """Clear cached facts at close_step only when a keyed revision axis changed."""
    if not optimized_turn_facts_enabled():
        return True
    caches = getattr(runtime, "_host_fact_cache", None)
    if not isinstance(caches, dict) or not any(
            isinstance(store, Mapping) and bool(store) for store in caches.values()):
        return False
    try:
        if (not hasattr(runtime, "_host_cache_receipt_generation")
                or not hasattr(runtime, "_host_cache_product_revision")):
            return True
        journal = getattr(runtime, "journal", None)
        generation = getattr(journal, "receipt_generation", None)
        product_revision = _product_revision(runtime)
        return (
            generation != getattr(runtime, "_host_cache_receipt_generation")
            or product_revision != getattr(runtime, "_host_cache_product_revision")
        )
    except Exception:
        # If an invalidation axis cannot be read, retain the old safe clear behavior.
        return True


def _record_request_cache_outcome(
        journal: object | None, request_id: str, outcome: str,
) -> None:
    if journal is None or outcome not in {"hit", "miss"} or len(request_id) > 120:
        return
    try:
        emitted = getattr(journal, "_host_cache_summary_emitted", None)
        if not isinstance(emitted, set):
            emitted = set()
            setattr(journal, "_host_cache_summary_emitted", emitted)
        if request_id in emitted:
            return
        pending = getattr(journal, "_host_cache_request_counts", None)
        if not isinstance(pending, dict):
            pending = {}
            setattr(journal, "_host_cache_request_counts", pending)
        counts = pending.get(request_id)
        if not isinstance(counts, dict):
            counts = {"hits": 0, "misses": 0}
            pending[request_id] = counts
        counter = "hits" if outcome == "hit" else "misses"
        counts[counter] = int(counts.get(counter, 0)) + 1
    except Exception:
        pass


def flush_journal_cache_summaries(journal: object) -> None:
    """Append one aggregate cache row per request at journal close."""
    emit = getattr(journal, "emit", None)
    pending = getattr(journal, "_host_cache_request_counts", None)
    if not callable(emit) or not isinstance(pending, dict):
        return
    emitted = getattr(journal, "_host_cache_summary_emitted", None)
    if not isinstance(emitted, set):
        emitted = set()
        try:
            setattr(journal, "_host_cache_summary_emitted", emitted)
        except Exception:
            return
    for request_id, counts in tuple(pending.items()):
        if not isinstance(request_id, str) or not isinstance(counts, Mapping):
            pending.pop(request_id, None)
            continue
        hits = counts.get("hits")
        misses = counts.get("misses")
        if (type(hits) is not int or hits < 0
                or type(misses) is not int or misses < 0):
            pending.pop(request_id, None)
            continue
        if request_id in emitted:
            pending.pop(request_id, None)
            continue
        emitted.add(request_id)
        try:
            emit(
                "cache_request_summary",
                producer="kernel_host_cache",
                event_schema="stateframe-cache-request-summary/1",
                request_id=request_id,
                cache_hit_count=hits,
                cache_miss_count=misses,
                dispatch=False,
            )
        except Exception:
            emitted.discard(request_id)
            continue
        pending.pop(request_id, None)


def sidecar_cache_enabled() -> bool:
    """Return whether opt-in sidecar text reuse is enabled; default is OFF."""
    try:
        return _switch("STATEFRAME_SIDECAR_KEYED", "off")
    except Exception:
        return False


def sidecar_render_key(runtime: object) -> tuple[str, int] | None:
    """Build the prescribed cross-request key without guessing missing axes."""
    try:
        binding_fn = getattr(runtime, "binding", None)
        binding = binding_fn() if callable(binding_fn) else getattr(runtime, "binding", None)
        product_revision = (
            binding.get("product_revision") if isinstance(binding, Mapping)
            else getattr(runtime, "product_revision", None)
        )
        journal = getattr(runtime, "journal", None)
        receipt_generation = getattr(journal, "receipt_generation", None)
        if (not isinstance(product_revision, str) or not product_revision.strip()
                or type(receipt_generation) is not int or receipt_generation < 0):
            return None
        return product_revision, receipt_generation
    except Exception:
        # An incomplete key is a cache miss, never authority to reuse a sidecar.
        return None


def memo_runtime_sidecar_text(
        runtime: object, name: str, compute: Callable[[], str | None]) -> str | None:
    """Memoize immutable sidecar text on (product_revision, receipt_generation).

    The runtime owns this single-entry-per-name cache across its requests. Shadow
    mode renders the current value and compares UTF-8 bytes before any reuse. A
    mismatch disables that sidecar for the lifetime of the runtime. Cache errors
    always pass through to the existing renderer.
    """
    if not sidecar_cache_enabled():
        return compute()

    key = sidecar_render_key(runtime)
    if key is None:
        return compute()

    try:
        journal = getattr(runtime, "journal", None)
        uncacheable = getattr(runtime, "_sidecar_render_uncacheable", None)
        if uncacheable is None:
            uncacheable = set()
            setattr(runtime, "_sidecar_render_uncacheable", uncacheable)
        if not isinstance(uncacheable, set):
            raise TypeError("invalid_sidecar_uncacheable")
        if name in uncacheable:
            return compute()

        cache = getattr(runtime, "_sidecar_render_cache", None)
        if cache is None:
            cache = {}
            setattr(runtime, "_sidecar_render_cache", cache)
        if not isinstance(cache, dict):
            raise TypeError("invalid_sidecar_cache")
        entry = cache.get(name)
        if entry is not None and (
                not isinstance(entry, tuple) or len(entry) != 2
                or not isinstance(entry[0], tuple) or len(entry[0]) != 2
                or not isinstance(entry[0][0], str) or type(entry[0][1]) is not int
                or not isinstance(entry[1], str)):
            raise TypeError("invalid_sidecar_entry")
    except Exception as error:
        _emit_cache_event(
            journal, "cache_bypass", name=f"SIDECAR_{name}",
            reason=f"sidecar_state_error:{type(error).__name__}")
        return compute()

    if isinstance(entry, tuple) and entry[0] == key:
        cached_text = entry[1]
        try:
            shadow = _switch("STATEFRAME_HOST_CACHE_SHADOW", "off")
        except Exception as error:
            _emit_cache_event(
                journal, "cache_bypass", name=f"SIDECAR_{name}",
                reason=f"shadow_switch_error:{type(error).__name__}")
            return compute()
        if not shadow:
            return cached_text

        fresh_text = compute()
        try:
            matches = (isinstance(fresh_text, str)
                       and fresh_text.encode("utf-8", "surrogatepass")
                       == cached_text.encode("utf-8", "surrogatepass"))
            if not matches:
                try:
                    cache.pop(name, None)
                except Exception:
                    pass
                try:
                    uncacheable.add(name)
                except Exception:
                    pass
                _emit_cache_event(
                    journal, "cache_mismatch", name=f"SIDECAR_{name}",
                    reason="shadow_render_changed")
        except Exception as error:
            try:
                cache.pop(name, None)
            except Exception:
                pass
            try:
                uncacheable.add(name)
            except Exception:
                pass
            _emit_cache_event(
                journal, "cache_bypass", name=f"SIDECAR_{name}",
                reason=f"shadow_compare_error:{type(error).__name__}")
        return fresh_text

    fresh_text = compute()
    if not isinstance(fresh_text, str):
        try:
            cache.pop(name, None)
        except Exception:
            pass
        return fresh_text
    try:
        cache[name] = (key, fresh_text)
    except Exception as error:
        _emit_cache_event(
            journal, "cache_bypass", name=f"SIDECAR_{name}",
            reason=f"sidecar_store_error:{type(error).__name__}")
    return fresh_text


def journal_records(journal: object, *, name: str = "JOURNAL_RECORDS") -> tuple[object, ...]:
    """Use the journal's protected materialized cache, falling back on its reader."""
    cached = getattr(journal, "cached_records", None)
    records = getattr(journal, "records", None)
    if not callable(cached):
        _emit_cache_event(journal, "cache_bypass", name=name, reason="accessor_missing")
        return tuple(records()) if callable(records) else ()
    try:
        return tuple(cached())
    except Exception as error:
        _emit_cache_event(
            journal, "cache_bypass", name=name,
            reason=f"journal_error:{type(error).__name__}")
        return tuple(records()) if callable(records) else ()


__all__ = [
    "CACHE_FORMAT", "cache_enabled", "clear_request_cache", "flush_journal_cache_summaries",
    "journal_records", "memo_by", "memo_runtime_body", "memo_runtime_fact", "optimized_turn_facts_enabled",
    "request_cache_invalidation_changed", "turn_facts_key", "memo_runtime_sidecar_text",
    "sidecar_cache_enabled", "sidecar_render_key",
]

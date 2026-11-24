"""True save/load resume: a generic snapshot of the live engine at the step boundary.

At the top of every step-loop iteration in ``react_repair.work`` the engine's whole
live state is pickled in ONE pass (one memo, so shared objects stay shared):

* the ``work()`` frame's locals and cell variables (PEP 667 ``frame.f_locals``),
* every engine module's data globals and engine classes' data attributes,
* monkeypatched callables installed on any module (product wrappers, the seeded uuid4),
* every ``contextvars.ContextVar`` value, and the ``random`` module state.

Run-dir files (journal, wire, workspace, ...) are APFS-cloned beside the state.

A restore re-runs ``work()`` setup in a new process, then, just before the step loop,
puts the run-dir files back and overwrites the live state from the snapshot.  The
caller proves fidelity by comparing the first rebuilt request with the recorded one;
any refusal or mismatch falls back to replay resume.  Nothing here is trusted
silently: every unsupported object fails the save, which then records why.

Switch: STATEFRAME_CHECKPOINT = on (default) | off | probe.
Restore: STATEFRAME_CHECKPOINT_RESTORE=<checkpoint dir> (set by the product resume).
Stdlib only.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import contextvars
import fcntl
import hashlib
import io
import json
import marshal
import errno
import os
import pickle
import random
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
import types
import weakref
import zlib
from pathlib import Path

if __package__:
    from .kernel_secret_scan import SecretScanner
else:
    from kernel_secret_scan import SecretScanner

SCHEMA = "stateframe-engine-checkpoint/1"
MODE_ENV = "STATEFRAME_CHECKPOINT"
RESTORE_ENV = "STATEFRAME_CHECKPOINT_RESTORE"
EVERY_ENV = "STATEFRAME_CHECKPOINT_EVERY"
KEEP_ENV = "STATEFRAME_CHECKPOINT_KEEP"
_PUBLICATION_SWITCH = "STATEFRAME_TURN_ATOMIC_PUBLICATION"
_PUBLICATION_INDEX = ".turn-checkpoint-index.json"
MIN_FREE_ENV = "STATEFRAME_CHECKPOINT_MIN_FREE_BYTES"
STATE_FILE = "state.pkl"
META_FILE = "checkpoint.json"
STATE_CODEC = "zlib1"
# The largest previously measured raw state was about 460 MiB; this 2 GiB ceiling is over 4x
# that size while bounding output from any checkpoint before it can be handed to pickle.
MAX_STATE_BYTES = 2 << 30
# Run-dir entries never copied into a checkpoint (inputs, history, or the checkpoints themselves).
# Hook bundles are pinned read-only inputs. Hook records, delivery history and scratch are live
# per-attempt data; the hook tape keeps the logical-run history across attempts. None may be
# replaced by an older snapshot. The tool tape and engine log are live for the same reason.
# Subframe decisions and supervisor channels are live host authority too: moving
# their directory races the supervisor and can roll back durable transactions.
# The mutation journal and evidence are durable launch and edit records that resume reconciles
# before the engine starts; an older copy would erase that reconciliation.
SKIP_RUN_ENTRIES = frozenset({"checkpoints", "attempts", "parent", ".engine.lock", ".run.lock",
                              "spec.json", "child-traceback.txt", "engine-result.json", "summary.json",
                              "engine.log", "checkpoint-refused.json", "tool-tape.jsonl",
                              "hook-tape.jsonl", "hook-bundle", "hooks", "hook-scratch", "parallel",
                              "mutations", ".workspace-objects.json", ".turn-publication"})

# Keep writer diagnostics independent of the mutable errno.errorcode table and exception text.
_OS_ERROR_NAMES = types.MappingProxyType({
    1: "EPERM",
    2: "ENOENT",
    5: "EIO",
    13: "EACCES",
    28: "ENOSPC",
    30: "EROFS",
})
_WRITER_FAILURE_REASONS = frozenset({
    "checkpoint refused: registered secret in checkpoint",
    "checkpoint refused: registered secret in state",
    "STATEFRAME_HOME_IDENTITY_CHANGED",
    "checkpoint refused",
    "checkpoint write failed: os error",
    "checkpoint write failed: EPERM",
    "checkpoint write failed: ENOENT",
    "checkpoint write failed: EIO",
    "checkpoint write failed: EACCES",
    "checkpoint write failed: ENOSPC",
    "checkpoint write failed: EROFS",
    "checkpoint write failed: memory error",
    "checkpoint serialization failed",
    "unexpected writer error",
})
_WRITER_ERROR_NAMES = frozenset({
    "TypeError", "ValueError", "AttributeError", "RuntimeError", "OverflowError", "RecursionError",
})


class CheckpointError(RuntimeError):
    """A snapshot could not be taken or restored faithfully."""


class _StateFileSecretError(CheckpointError):
    """Marker for secret bytes rejected while scanning captured files."""


_OBJECT_CAPTURE_STAGES = frozenset({
    "admission", "workspace", "run_tail", "semantic_state", "objects", "publication",
})


class ObjectCaptureError(CheckpointError):
    """A failed object checkpoint with a fixed stage and a redacted reason."""

    def __init__(self, stage: str, reason: str):
        super().__init__(reason)
        self.stage = stage
        self.reason = reason


class RestoreRefused(BaseException):
    """The engine refuses to continue from a checkpoint (raised before any model request).

    A BaseException so no engine handler turns it into an ordinary run outcome; the host sees
    REFUSAL_FILE and falls back to replay resume.
    """


REFUSAL_FILE = "checkpoint-refused.json"


def mode() -> str:
    value = os.environ.get(MODE_ENV, "").strip().lower() or "on"  # empty = unset = on
    value = {"on": "save", "1": "save", "0": "off"}.get(value, value)  # on = save
    if value not in {"off", "save", "probe"}:
        raise ValueError(f"{MODE_ENV} must be off, on/save or probe")
    return value


# GCFREEZE: once the long-lived engine structures exist, freeze them out of
# generation-2 GC scans so every later collection stops re-walking the whole warm heap. Changes no object, value or byte.
# Experiment switch SF_LAB_GC_FREEZE: off (PC3G behaviour) | once (default: one freeze per process; a resumed run re-freezes in its new pid)
# | periodic (collect + freeze again every GC_FREEZE_PERIOD steps so objects that became long-lived later also leave gen-2 scans).
# Every freeze collects generation 2 FIRST, so only live objects are frozen (freezing uncollected cycles would leak them).
GC_FREEZE_ENV = "SF_LAB_GC_FREEZE"
GC_FREEZE_AFTER_STEP = 5
GC_FREEZE_PERIOD = 50
_GC_FROZEN_PID: int | None = None
_GC_LAST_FREEZE_STEP: int | None = None


def gc_freeze_mode() -> str:
    value = os.environ.get(GC_FREEZE_ENV, "once").strip().lower()
    if value in {"off", "0", "false", "no"}:
        return "off"
    return "periodic" if value == "periodic" else "once"


def freeze_warm_heap(step: int) -> bool:
    """Freeze the warm heap after warm-up (once, or every GC_FREEZE_PERIOD steps); returns True when it froze this call."""
    global _GC_FROZEN_PID, _GC_LAST_FREEZE_STEP
    mode = gc_freeze_mode()
    if mode == "off" or step < GC_FREEZE_AFTER_STEP:
        return False
    same_process = _GC_FROZEN_PID == os.getpid()
    if same_process and (mode == "once" or (
            _GC_LAST_FREEZE_STEP is not None and step - _GC_LAST_FREEZE_STEP < GC_FREEZE_PERIOD)):
        return False
    import gc
    gc.collect(2)
    gc.freeze()
    _GC_FROZEN_PID = os.getpid()
    _GC_LAST_FREEZE_STEP = step
    return True


# --------------------------------------------------------------------------- scope

_ENGINE_ROOT: str | None = None


def _engine_root() -> str:
    global _ENGINE_ROOT
    if _ENGINE_ROOT is None:
        _ENGINE_ROOT = os.path.dirname(os.path.abspath(__file__)) + os.sep
    return _ENGINE_ROOT


def _is_engine_module(module: object) -> bool:
    path = getattr(module, "__file__", None)
    return isinstance(path, str) and os.path.abspath(path).startswith(_engine_root())


def _is_state_module(name: str, module: object) -> bool:
    short_name = name.rsplit(".", 1)[-1]
    if (_workspace_objects_on() and short_name.startswith("kernel_workspace_")
            and short_name != "kernel_workspace_truth"):
        return False
    # __main__ (the product child script) is the HOST: it re-installs its own wrappers in the
    # new process and hands their counters over as participants (register_participant).
    return name != "__main__" and _is_engine_module(module)


HOST_MODULE = "__main__"
_PARTICIPANTS: dict[str, dict] = {}


def register_participant(name: str, state: dict) -> None:
    """A host-owned state dict (e.g. the wire send counter) saved by value and restored IN PLACE.

    The host re-installs its wrappers in the new process before the engine restores; their
    live state dicts are then refilled from the checkpoint, and every engine reference to a
    participant dict resolves to the live one.
    """
    if not isinstance(state, dict):
        raise TypeError("participant state must be a dict")
    _PARTICIPANTS[name] = state


def _host_functions() -> dict[int, tuple]:
    """Host wrappers by location: (module, attr) or (module, class qualname, attr)."""
    found: dict[int, tuple] = {}
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        try:
            items = list(vars(module).items())
        except TypeError:
            continue
        for key, value in items:
            if isinstance(value, types.FunctionType) and value.__module__ == HOST_MODULE:
                found.setdefault(id(value), (name, "", key))
            elif isinstance(value, type) and value.__module__ == name and _is_state_module(name, module):
                for attr, member in list(vars(value).items()):
                    if isinstance(member, types.FunctionType) and member.__module__ == HOST_MODULE:
                        found.setdefault(id(member), (name, value.__qualname__, attr))
    return found


def _resolve_host(location: tuple):
    module_name, qualname, attr = location
    target: object = sys.modules[module_name]
    for part in qualname.split(".") if qualname else ():
        target = getattr(target, part)
    return vars(target)[attr] if qualname else getattr(target, attr)


def _resolves_to(obj: object) -> bool:
    """True when ``obj`` is reachable by its own module + qualname (pickle-by-reference works)."""
    module = sys.modules.get(getattr(obj, "__module__", None) or "")
    if module is None:
        return False
    target: object = module
    for part in str(getattr(obj, "__qualname__", "")).split("."):
        if part == "<locals>":
            return False
        target = getattr(target, part, None)
        if target is None:
            return False
    return target is obj


_STATIC_TYPES = (types.ModuleType, type, types.BuiltinFunctionType)


def _is_static(value: object) -> bool:
    """Values a fresh process recreates identically by import (never saved)."""
    if isinstance(value, _STATIC_TYPES):
        return True
    if isinstance(value, (types.FunctionType, types.MethodType)):
        return _resolves_to(value)
    return False


# Reviewed exclusions: module globals a restore keeps from the NEW process (never saved).
# Each is a pure cache, a lazily rebuilt holder, or per-thread runtime state (threading.local), which
# cannot be pickled at all. Dropping a cache changes only speed; the thread-local entries keep their
# import-time (empty) state in the new process, and their readers fall back to defaults.
DROP_GLOBALS = {
    ("kernel_file_digest_memo", "_MEMO"):
        "process-local file observations; never valid across checkpoint restore",
    ("kernel_file_digest_memo", "_LAST_CLOCK_NS"):
        "process-local file observations; never valid across checkpoint restore",
    ("engine.kernel_file_digest_memo", "_MEMO"):
        "process-local file observations; never valid across checkpoint restore",
    ("engine.kernel_file_digest_memo", "_LAST_CLOCK_NS"):
        "process-local file observations; never valid across checkpoint restore",
    ("stateframe.engine.kernel_file_digest_memo", "_MEMO"):
        "process-local file observations; never valid across checkpoint restore",
    ("stateframe.engine.kernel_file_digest_memo", "_LAST_CLOCK_NS"):
        "process-local file observations; never valid across checkpoint restore",
    ("kernel_green_menu_open_obligation", "_FINGERPRINTS"):
        "rebuildable text of large receipts; never persisted",
    ("engine.kernel_green_menu_open_obligation", "_FINGERPRINTS"):
        "rebuildable text of large receipts; never persisted",
    ("stateframe.engine.kernel_green_menu_open_obligation", "_FINGERPRINTS"):
        "rebuildable text of large receipts; never persisted",
    ("kernel_fl_checkpoint_capture", "_API"):
        "lazy import holder of checkpoint helper classes; re-created on first use",
    ("repo_intel_core.lsp_resolve", "_PROJECTS"):
        "jedi project cache holding a live jedi subprocess; rebuilt on demand from the files",
    ("kernel_request_auth_identity", "_AUTH_BINDING_SECRET_CACHE"):
        "installation keys are reloaded from StateFrame home after restore",
    ("kernel_request_auth_identity", "_AUTH_BINDING_FALLBACK_SECRET"):
        "fallback binding salt is process-local and regenerated after restore",
    ("kernel_request_auth_identity", "_AUTH_BINDING_SECRET_LOCK"):
        "process-local lock protecting auth binding secret cache",
    ("engine.kernel_request_auth_identity", "_AUTH_BINDING_SECRET_CACHE"):
        "installation keys are reloaded from StateFrame home after restore",
    ("engine.kernel_request_auth_identity", "_AUTH_BINDING_FALLBACK_SECRET"):
        "fallback binding salt is process-local and regenerated after restore",
    ("engine.kernel_request_auth_identity", "_AUTH_BINDING_SECRET_LOCK"):
        "process-local lock protecting auth binding secret cache",
    ("stateframe.engine.kernel_request_auth_identity", "_AUTH_BINDING_SECRET_CACHE"):
        "installation keys are reloaded from StateFrame home after restore",
    ("stateframe.engine.kernel_request_auth_identity", "_AUTH_BINDING_FALLBACK_SECRET"):
        "fallback binding salt is process-local and regenerated after restore",
    ("stateframe.engine.kernel_request_auth_identity", "_AUTH_BINDING_SECRET_LOCK"):
        "process-local lock protecting auth binding secret cache",
    # threading.local objects cannot be pickled. Each is per-thread runtime state that the new
    # process rebuilds from its own import; the readers below tolerate it being unset.
    ("kernel_bounded_http", "_PROBE_STATE"):
        "per-thread /props probe outcome; each probe resets it before use",
    ("engine.kernel_bounded_http", "_PROBE_STATE"):
        "per-thread /props probe outcome; each probe resets it before use",
    ("stateframe.engine.kernel_bounded_http", "_PROBE_STATE"):
        "per-thread /props probe outcome; each probe resets it before use",
    ("kernel_navigation_pages", "_LINE_SCOPE"):
        "per-thread regex line scope; unset outside a search and read through getattr defaults",
    ("engine.kernel_navigation_pages", "_LINE_SCOPE"):
        "per-thread regex line scope; unset outside a search and read through getattr defaults",
    ("stateframe.engine.kernel_navigation_pages", "_LINE_SCOPE"):
        "per-thread regex line scope; unset outside a search and read through getattr defaults",
    ("fixer", "_TRACE"):
        "per-thread last trace ref; each captured call resets it before use",
    ("engine.fixer", "_TRACE"):
        "per-thread last trace ref; each captured call resets it before use",
    ("stateframe.engine.fixer", "_TRACE"):
        "per-thread last trace ref; each captured call resets it before use",
    ("fixer", "_CALL_META"):
        "per-thread metadata of the current model call; read through getattr defaults",
    ("engine.fixer", "_CALL_META"):
        "per-thread metadata of the current model call; read through getattr defaults",
    ("stateframe.engine.fixer", "_CALL_META"):
        "per-thread metadata of the current model call; read through getattr defaults",
    ("fixer", "_USAGE_SINK"):
        "per-thread usage accumulator installed by set_usage_sink; read through getattr defaults",
    ("engine.fixer", "_USAGE_SINK"):
        "per-thread usage accumulator installed by set_usage_sink; read through getattr defaults",
    ("stateframe.engine.fixer", "_USAGE_SINK"):
        "per-thread usage accumulator installed by set_usage_sink; read through getattr defaults",
    ("fixer", "_TRUNCATION_SINK"):
        "per-thread truncation telemetry installed by set_truncation_sink; read through getattr defaults",
    ("engine.fixer", "_TRUNCATION_SINK"):
        "per-thread truncation telemetry installed by set_truncation_sink; read through getattr defaults",
    ("stateframe.engine.fixer", "_TRUNCATION_SINK"):
        "per-thread truncation telemetry installed by set_truncation_sink; read through getattr defaults",
    ("prompt_ledger", "_CONTEXT"):
        "per-thread run/task labels for prompt capture; set by fixer.run via set_context",
    ("engine.prompt_ledger", "_CONTEXT"):
        "per-thread run/task labels for prompt capture; set by fixer.run via set_context",
    ("stateframe.engine.prompt_ledger", "_CONTEXT"):
        "per-thread run/task labels for prompt capture; set by fixer.run via set_context",
}
# Host configuration the product child sets on the engine before work() runs. The new process's
# host sets them again (possibly to new values: a raised --max-turns), and KEY is a credential that
# must never be written to disk. Replay control also belongs to the new process, whose tape can
# include recorded operations after the saved boundary.
HOST_GLOBALS = frozenset(
    (module, name)
    for module in ("probes.react_repair", "react_repair")
    for name in ("KEY", "API", "MODEL", "MAX_STEPS", "POC", "OUT")
) | frozenset({
    # The host creates these for the resumed attempt before restore. The new tape includes the
    # recorded tail; the saved tape stops at this boundary and replay was off in the old process.
    ("kernel_tool_tape", "_active_tape"),
    ("kernel_tool_tape", "_replay_active"),
    # These providers read the resumed child's recorded tail or pending request.
    # Restoring old None values (or old provider closures) erases that authority.
    ("kernel_cache_layout_candidate", "_REPLAY_LAYOUT_DECISION_PROVIDER"),
    ("kernel_cache_layout_candidate", "_REPLAY_STABLE_WIRE_DECISION_PROVIDER"),
    ("kernel_cache_layout_candidate", "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER"),
    ("kernel_cache_layout_candidate", "_REPLAY_STABLE_WIRE_RESERVES_PROVIDER"),
    # The mutation runtime the resuming host installed; kernel_mutate_runtime.rebind_after_checkpoint
    # copies its host settings onto the restored _active runtime.
    ("kernel_mutate_runtime", "_host"),
})
# Derived or runtime-only instance attributes do not belong in restored task state.
DROP_ATTRS = {
    ("kernel_continuation_journal", "ContinuationJournal"): frozenset({
        "_warm_host_span_pause_counts", "_warm_host_span_resume_counts",
    }),
    ("kernel_stateframe_runtime", "StateFrameRuntime"): frozenset({
        "_model_router_projection_cache", "_parent_product_comparison_cache", "_working_set_test_injection",
    }),
    # Tokenizer auth is rebound from the new child process on every meter lookup;
    # generation credentials and derived SSL state do not belong in checkpoints.
    ("kernel_request_diet", "RequestTokenMeter"): frozenset({
        "_tokenizer_auth_key", "_generation_endpoint", "_http_opener",
        "_http_ssl_context", "_http_ssl_context_lock",
        "_count_key_last",
    }),
    # Provider recovery reads the current host API/KEY when this transient destination is
    # absent, so a restored capture never needs the physical request's credential.
    ("kernel_measurement", "RequestCapture"): frozenset({"_provider_probe_destination"}),
}
_OWN_MODULE = __name__
_SECRETS: list[bytes] = []
_SECRET_CHECKPOINT_ERROR = "a registered secret would be written into the checkpoint"
_SECRET_GENERATION = 0
_VERIFIED_POOL_IDENTITIES: set[tuple[int, int, int, int, int, int, str]] = set()
LEGACY_DIRECT_ENGINE_ENV = "STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE"
_HOME_IDENTITY_UNSET = object()
_RUNNER_HOME_IDENTITY_CONFIGURED = False
_RUNNER_HOME_IDENTITY_PIN: dict[str, int | str] | None = None
_HOME_IDENTITY_FIELDS = frozenset({
    "realpath", "home_st_dev", "home_st_ino", "key_st_dev", "key_st_ino", "key_st_ctime_ns",
})


_SIGNING_KEY: list[bytes] = []


def register_signing_key(key: bytes) -> None:
    """The host's per-install checkpoint key. Loading a pickle executes code, so a checkpoint is
    written only when it can be signed and loaded only when its signature verifies. The key never
    goes to the run directory, the environment, or a log; it is also registered as a secret."""
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("checkpoint signing key must be at least 32 bytes")
    _SIGNING_KEY[:] = [key]
    register_secret(key)


def _signature(meta: dict, state_sha256: str) -> str:
    """HMAC-SHA256 over the canonical metadata (minus the signature) and the state digest; the
    metadata pins every journal chunk's sha256, so chunks are covered too."""
    import hmac
    if not _SIGNING_KEY:
        raise CheckpointError("no checkpoint signing key registered by the host")
    body = {key: value for key, value in meta.items() if key != "signature"}
    message = json.dumps(body, sort_keys=True, separators=(",", ":")).encode() + b"\0" + state_sha256.encode()
    return hmac.new(_SIGNING_KEY[0], message, hashlib.sha256).hexdigest()


def _verify_stored_state(blob: bytes, meta: dict):
    """Verify encoded bytes before any compressed state is passed to zlib."""
    codec = meta.get("state_codec")  # absent means the original raw state.pkl format
    if codec == STATE_CODEC:
        size = meta.get("state_stored_bytes")
        digest = meta.get("state_stored_sha256")
        if type(size) is not int or size < 0 or not isinstance(digest, str):
            raise CheckpointError("checkpoint stored state metadata is invalid")
        if len(blob) != size or hashlib.sha256(blob).hexdigest() != digest:
            raise CheckpointError("checkpoint state checksum mismatch")
    return codec


def _decode_verified_state(blob: bytes, meta: dict, codec) -> bytes:
    """Decode one state representation with bounded output, then verify the raw pickle digest."""
    size = meta.get("state_bytes")
    digest = meta.get("state_sha256")
    if type(size) is not int or size < 0 or not isinstance(digest, str):
        raise CheckpointError("checkpoint state metadata is invalid")

    if codec is None:
        raw = blob
    elif codec == STATE_CODEC:
        cap = min(size, MAX_STATE_BYTES)
        if size > MAX_STATE_BYTES:
            raise CheckpointError(f"checkpoint state exceeds the {MAX_STATE_BYTES}-byte limit")
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(blob, cap + 1)
        except zlib.error:
            raise CheckpointError("checkpoint state decompression failed") from None
        if len(raw) > cap or decoder.unconsumed_tail:
            raise CheckpointError("checkpoint state exceeds its declared or absolute size limit")
        if not decoder.eof:
            raise CheckpointError("checkpoint state compressed stream is truncated")
        if decoder.unused_data:
            raise CheckpointError("checkpoint state compressed stream has trailing data")
    else:
        raise CheckpointError(f"unknown checkpoint state codec {codec!r}")

    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise CheckpointError("checkpoint state checksum mismatch")
    return raw


def decode_state(blob: bytes, meta: dict) -> bytes:
    """Validate stored state bytes and return the verified raw pickle for resume preflight."""
    codec = _verify_stored_state(blob, meta)
    return _decode_verified_state(blob, meta, codec)


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _owned_private(path: Path) -> bool:
    """Owned by this user and not writable by group/others (a file anyone else could edit is not ours)."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    import stat as stat_module
    return (info.st_uid == os.getuid() and not stat_module.S_ISLNK(info.st_mode)
            and not info.st_mode & (stat_module.S_IWGRP | stat_module.S_IWOTH))


def register_secret(value: str | bytes) -> None:
    """Register text credentials or raw installation bytes with the checkpoint byte guard."""
    global _SECRET_GENERATION
    if isinstance(value, str) and len(value) >= 8:
        candidates = (value.encode("utf-8"),)
    elif isinstance(value, bytes) and len(value) >= 8:
        candidates = (value, value.hex().encode("ascii"))
    else:
        return
    added = False
    for candidate in candidates:
        if candidate not in _SECRETS:
            _SECRETS.append(candidate)
            added = True
    if added:
        _SECRET_GENERATION += 1
        _VERIFIED_POOL_IDENTITIES.clear()


# Structural bundle traversal keeps its own parse budget. The lexical byte
# scanner has no encoded-string size limit and carries state across every read.
_SECRET_JSON_MAX_DEPTH = 8
_SECRET_JSON_MAX_BYTES = 1 << 20


def _check_registered_secrets(
        data: bytes, tail: SecretScanner | None = None, *, final: bool = True,
        failure_type: type[CheckpointError] = CheckpointError) -> SecretScanner:
    """Use one shared scanner for objects, files, state writes and export bytes.

    Streaming callers pass final=False and carry the returned scanner until EOF.
    One-shot callers also flush incomplete escapes, including lone surrogates.
    """
    scanner = tail if isinstance(tail, SecretScanner) else SecretScanner(
        _SECRETS, stages=_SECRET_JSON_MAX_DEPTH)
    if scanner.feed(data) or (final and scanner.finish()):
        raise failure_type(_SECRET_CHECKPOINT_ERROR)
    return scanner


def _valid_home_identity(value: object) -> bool:
    if type(value) is not dict or set(value) != _HOME_IDENTITY_FIELDS:
        return False
    if not isinstance(value.get("realpath"), str) or not Path(value["realpath"]).is_absolute():
        return False
    return all(type(value.get(key)) is int and value[key] >= 0
               for key in _HOME_IDENTITY_FIELDS - {"realpath"})


def configure_runner_home_identity_pin(serialized_identity: str | None) -> None:
    """Bind this child to the runner's startup pin, independently of the live run spec."""
    global _RUNNER_HOME_IDENTITY_CONFIGURED, _RUNNER_HOME_IDENTITY_PIN
    _RUNNER_HOME_IDENTITY_CONFIGURED = True
    _RUNNER_HOME_IDENTITY_PIN = None
    if not isinstance(serialized_identity, str):
        return
    try:
        identity = json.loads(serialized_identity)
    except (ValueError, TypeError):
        return
    if _valid_home_identity(identity):
        _RUNNER_HOME_IDENTITY_PIN = dict(identity)


def _register_legacy_home_secret() -> None:
    """Legacy pin opt-in never opts an existing installation key out of byte protection."""
    if __package__:
        from .kernel_request_auth_identity import _installation_home, protected_home_identity
    else:
        from kernel_request_auth_identity import _installation_home, protected_home_identity
    try:
        home = _installation_home()
    except (OSError, ValueError, RuntimeError):
        return  # A direct-engine fixture need not have an installation home.
    try:
        os.lstat(home / "identity.key")
    except FileNotFoundError:
        return
    except OSError as error:
        raise CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED") from error
    try:
        protected_home_identity(home)
    except (OSError, ValueError, RuntimeError) as error:
        raise CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED") from error


def _snapshot_file_cache_key(info: os.stat_result, generation: int, digest: str
                             ) -> tuple[int, int, int, int, int, int, str]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            generation, digest)


def _snapshot_home_identity(
        run_dir: Path, *, expected_identity: object = _HOME_IDENTITY_UNSET,
        allow_legacy: bool = False,
) -> dict[str, int | str] | None:
    """Verify live metadata against the runner's immutable startup pin."""
    runner_bound = expected_identity is not _HOME_IDENTITY_UNSET
    if runner_bound:
        startup_identity = expected_identity
    elif _RUNNER_HOME_IDENTITY_CONFIGURED:
        runner_bound = True
        startup_identity = _RUNNER_HOME_IDENTITY_PIN
    else:
        startup_identity = None
    if runner_bound and not _valid_home_identity(startup_identity):
        raise CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED")

    spec_path = run_dir / "spec.json"
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        if (not runner_bound and (allow_legacy
                or os.environ.get(LEGACY_DIRECT_ENGINE_ENV, "").strip() == "1")):
            _register_legacy_home_secret()
            return None
        raise CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED") from error
    except (OSError, ValueError, TypeError) as error:
        raise CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED") from error
    if not isinstance(spec, dict):
        raise CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED")
    try:
        recorded_identity = recorded_home_identity(spec)
    except ValueError as error:
        raise CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED") from error
    environment = spec.get("environment")
    selected_home = environment.get("STATEFRAME_HOME") if isinstance(environment, dict) else None
    if (allow_legacy and not runner_bound and "stateframe_home_identity" not in spec
            and (not isinstance(environment, dict)
                 or "STATEFRAME_RUNNER_HOME_IDENTITY" not in environment)):
        _register_legacy_home_secret()
        return None
    if (not _valid_home_identity(recorded_identity)
            or not isinstance(selected_home, str) or not selected_home):
        raise CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED")
    try:
        if __package__:
            from .kernel_request_auth_identity import _installation_home, protected_home_identity
        else:
            from kernel_request_auth_identity import _installation_home, protected_home_identity

        expected = startup_identity if runner_bound else recorded_identity
        # Only the child binds process authority. The parent CLI may validate a
        # profile/extra-env home different from its own ambient installation home.
        if _RUNNER_HOME_IDENTITY_CONFIGURED and (
                not _valid_home_identity(_RUNNER_HOME_IDENTITY_PIN)
                or os.path.realpath(_installation_home()) != _RUNNER_HOME_IDENTITY_PIN["realpath"]):
            raise ValueError("effective process HOME differs from startup")
        if runner_bound and (recorded_identity != expected
                             or os.path.realpath(selected_home) != expected["realpath"]):
            raise ValueError("live HOME pin differs from startup")
        # This also registers identity.key with the byte guard when there is no API key.
        current = protected_home_identity(Path(expected["realpath"]))
        repo = Path(spec["repo"])
        home = Path(str(current["realpath"]))
        run_root = _real_path(run_dir)
        repo_root = _real_path(repo)
        if (_path_within(home, run_root) or _path_within(home, repo_root)
                or current != expected):
            raise ValueError("protected home changed")
        return expected
    except Exception as error:
        raise CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED") from error


def recorded_home_identity(spec: dict[str, object]) -> object:
    """Read the persisted pin without adding to the public run-spec key surface.

    Accept the original top-level representation for interrupted HOME-era runs.
    Conflicting representations are corruption, never an opportunity to choose a pin.
    """
    identity = spec.get("stateframe_home_identity")
    environment = spec.get("environment")
    if isinstance(environment, dict) and "STATEFRAME_RUNNER_HOME_IDENTITY" in environment:
        serialized = environment["STATEFRAME_RUNNER_HOME_IDENTITY"]
        if not isinstance(serialized, str):
            raise ValueError("invalid recorded HOME identity")
        persisted = json.loads(serialized)
        if "stateframe_home_identity" in spec and identity != persisted:
            raise ValueError("conflicting recorded HOME identities")
        identity = persisted
        if not _valid_home_identity(identity):
            raise ValueError("invalid recorded HOME identity")
    elif "stateframe_home_identity" in spec and not _valid_home_identity(identity):
        raise ValueError("invalid recorded HOME identity")
    return identity


def _check_snapshot_file(path: Path, *, digest_hint: str | None = None,
                         cache_verified: bool = True,
                         force_scan: bool = False) -> tuple[int, str]:
    """Reject secret-bearing captured files, streaming with overlap across read chunks."""
    try:
        before = os.lstat(path)
    except OSError as error:
        raise CheckpointError("run file changed during checkpoint capture") from error
    if not stat.S_ISREG(before.st_mode):
        return before.st_size, ""

    identity = _STATE.get("protected_home_identity")
    if isinstance(identity, dict) and (
            before.st_dev, before.st_ino) == (
                identity.get("key_st_dev"), identity.get("key_st_ino")):
        raise _StateFileSecretError(_SECRET_CHECKPOINT_ERROR)

    generation = _SECRET_GENERATION
    if digest_hint and cache_verified and not force_scan:
        cached = _snapshot_file_cache_key(before, generation, digest_hint)
        if cached in _VERIFIED_POOL_IDENTITIES:
            return before.st_size, digest_hint

    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                             | getattr(os, "O_CLOEXEC", 0))
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns,
                    opened.st_ctime_ns) != (
                        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                        before.st_ctime_ns)):
            raise CheckpointError("run file changed during checkpoint capture")
        digest = hashlib.sha256()
        tail = None
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            tail = _check_registered_secrets(
                block, tail, final=False, failure_type=_StateFileSecretError)
            digest.update(block)
        _check_registered_secrets(b"", tail, failure_type=_StateFileSecretError)
        after = os.fstat(descriptor)
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns):
            raise CheckpointError("run file changed during checkpoint capture")
        value = digest.hexdigest()
        if digest_hint is not None and value != digest_hint:
            raise CheckpointError("pooled file changed during checkpoint capture")
        if _snapshot_file_cache_key(os.lstat(path), generation, value) != (
                _snapshot_file_cache_key(opened, generation, value)):
            raise CheckpointError("run file changed during checkpoint capture")
        if cache_verified:
            _VERIFIED_POOL_IDENTITIES.add(_snapshot_file_cache_key(opened, generation, value))
        return opened.st_size, value
    except OSError as error:
        raise CheckpointError("run file could not be scanned for registered secrets") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def check_registered_secret_file(path: str | os.PathLike[str]) -> str:
    """Stream-scan a file at an export boundary and return its SHA-256 digest."""
    return _check_snapshot_file(Path(path))[1]


def verify_stateframe_home_identity(
        run_dir: str | os.PathLike[str], *, expected_identity: object = _HOME_IDENTITY_UNSET,
        allow_legacy: bool = False,
) -> None:
    """Fail closed if a run's protected home identity moved or entered a protected tree."""
    _verify_stateframe_home_identity(
        Path(run_dir), expected_identity=expected_identity, allow_legacy=allow_legacy)


def _verify_stateframe_home_identity(
        run_dir: Path, *, expected_identity: object = _HOME_IDENTITY_UNSET,
        allow_legacy: bool = False,
) -> dict[str, int | str] | None:
    return _snapshot_home_identity(
        run_dir, expected_identity=expected_identity, allow_legacy=allow_legacy)


def _module_state(name: str, module: types.ModuleType) -> dict[str, object]:
    out = {}
    if name in {_OWN_MODULE, "kernel_checkpoint", "engine.kernel_checkpoint",
                "stateframe.engine.kernel_checkpoint"}:
        return out
    for key, value in list(vars(module).items()):
        if key.startswith("__") and key.endswith("__"):
            continue
        if name.rsplit(".", 1)[-1] == "kernel_callsite_census" and key == "_CENSUS_REVISION_CACHE":
            value = type(value)()
        if (_workspace_objects_on() and name == "kernel_bridge_evidence"
                and key == "capture_candidate"):
            continue
        if (_workspace_objects_on() and (name, key) in {
                ("kernel_shell_tool", "_active"),
                ("kernel_agent_tools", "_active"),
                ("kernel_workspace_consumers", "_bindings"),
        }):
            # These live bindings are rebuilt before restore_point. The feed
            # contains an inotify callback backed by a native pointer.
            continue
        if (name, key) in DROP_GLOBALS or (name, key) in HOST_GLOBALS:
            continue
        if _is_static(value):
            continue
        if isinstance(value, contextvars.ContextVar):
            continue
        out[key] = value
    return out


def _class_state(cls: type) -> dict[str, object]:
    import enum
    out = {}
    if issubclass(cls, enum.Enum):
        return out  # members are pickled by reference; an Enum class cannot be reassigned
    native_layout = issubclass(cls, (ctypes.Structure, ctypes.Union))
    native_fields = {field[0] for field in vars(cls).get("_fields_", ())} if native_layout else set()
    for key, value in list(vars(cls).items()):
        if key.startswith("__") and key.endswith("__") or key == "_abc_impl":
            continue
        if native_layout and (key == "_fields_" or key in native_fields):
            # ctypes fixes its layout at import; generated field types and CField descriptors
            # cannot be pickled or reassigned during restore. Other class data remains state.
            continue
        if isinstance(value, (staticmethod, classmethod, property, types.MemberDescriptorType,
                              types.GetSetDescriptorType)):
            continue
        if _is_static(value):
            continue
        out[key] = value
    return out


def _patched_callables() -> dict[str, dict[str, object]]:
    """Monkeypatches on NON-engine modules (e.g. uuid.uuid4 = seeded closure)."""
    out: dict[str, dict[str, object]] = {}
    for name, module in list(sys.modules.items()):
        if module is None or _is_state_module(name, module):
            continue
        try:
            items = list(vars(module).items())
        except TypeError:
            continue
        for key, value in items:
            if isinstance(value, types.FunctionType) and not _resolves_to(value):
                if getattr(value, "__module__", "") in sys.modules and _is_state_module(
                        value.__module__, sys.modules[value.__module__]):
                    out.setdefault(name, {})[key] = value
    return out


def _context_items() -> list[tuple[contextvars.ContextVar, object]]:
    return list(contextvars.copy_context().items())


def _cvar_names() -> dict[int, tuple[str, str]]:
    names: dict[int, tuple[str, str]] = {}
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        try:
            items = list(vars(module).items())
        except TypeError:
            continue
        for key, value in items:
            if isinstance(value, contextvars.ContextVar):
                names.setdefault(id(value), (name, key))
    return names


# --------------------------------------------------------------------------- reducers

def _globals_module(fn) -> str:
    """The module whose namespace IS ``fn.__globals__`` (functools.wraps rewrites __module__)."""
    globs = fn.__globals__
    name = globs.get("__name__")
    module = sys.modules.get(name) if isinstance(name, str) else None
    if module is not None and vars(module) is globs:
        return name
    for other, module in list(sys.modules.items()):
        if module is not None and getattr(module, "__dict__", None) is globs:
            return other
    raise pickle.PicklingError(f"function {fn.__qualname__} has globals of no imported module")


def _rebuild_function(code_bytes, module_name, name, qualname, defaults, closure, kwdefaults, attrs,
                      declared_module=None):
    module = sys.modules.get(module_name)
    if module is None:
        __import__(module_name)
        module = sys.modules[module_name]
    fn = types.FunctionType(marshal.loads(code_bytes), vars(module), name, defaults, closure)
    fn.__qualname__ = qualname
    if declared_module is not None:
        fn.__module__ = declared_module
    fn.__kwdefaults__ = kwdefaults
    if attrs:
        fn.__dict__.update(attrs)
    return fn


def _set_cell(cell, state):
    # The state is a 1-tuple: pickle skips the setter for a bare None state, which would
    # leave a cell holding None EMPTY after the load.
    cell.cell_contents = state[0]


def _empty_cell():
    return types.CellType()


class _TokenMissing:
    """Pickle stand-in for contextvars.Token.MISSING."""


def _rebuild_token(var, old_value):
    """A Token of the CURRENT context whose reset() restores ``old_value``.

    ``var`` is left set (the final context restore overwrites it). A MISSING old value
    (the variable was unset when the token was made) is rebuilt as the variable's own
    default, which every get() reads identically; a variable without a default cannot
    be restored that way and fails the restore instead of guessing.
    """
    if old_value is _TokenMissing:
        try:
            old_value = contextvars.Context().run(var.get)
        except LookupError:
            raise CheckpointError(f"token for ContextVar {var.name!r} without a default") from None
    marker = object()
    current = var.get(marker)
    var.set(old_value)
    return var.set(old_value if current is marker else current)


def _token_old(token):
    old = token.old_value
    return _TokenMissing if old is contextvars.Token.MISSING else old


def _builtin_type(name):
    return getattr(types, name)


_TYPES_BY_ID = {id(v): k for k, v in vars(types).items() if isinstance(v, type)}


def _new_container(cls, base):
    """Allocate without running the subclass's own __new__/__init__ (state is filled after)."""
    return base.__new__(cls)


def _fill_container(obj, state):
    """Refill a dict/list/set subclass through the BASE type's methods (frozen subclasses reject
    their own mutators), then its instance attributes."""
    items, attrs = state
    if isinstance(obj, dict):
        dict.update(obj, items)
    elif isinstance(obj, list):
        list.extend(obj, items)
    else:
        set.update(obj, items)
    if attrs:
        obj.__dict__.update(attrs)


def _new_frozen(cls, base, items):
    """Immutable tuple/frozenset subclass with its own __new__ signature: allocate via the base."""
    return base.__new__(cls, items)


def _set_attrs(obj, attrs):
    for key, value in attrs.items():
        object.__setattr__(obj, key, value)


def _frozen_sequence(obj) -> type | None:
    """tuple/frozenset subclasses whose own __new__ pickle's default path cannot call."""
    cls = type(obj)
    if cls in (tuple, frozenset) or hasattr(cls, "_fields") or cls.__module__ == "builtins":
        return None
    for base in (tuple, frozenset):
        if isinstance(obj, base) and any("__new__" in vars(k) for k in cls.__mro__[:cls.__mro__.index(base)]):
            return base
    return None


def _new_weak(cls):
    return cls()


def _fill_weak(obj, state):
    """Weak containers keep only entries whose objects are alive: the same live objects of the graph."""
    items = state[0]
    if isinstance(obj, weakref.WeakSet):
        for item in items:
            obj.add(item)
    else:
        for key, value in items:
            obj[key] = value


class _Dead:
    """Referent for a weakref that was already dead when saved."""


def _rebuild_ref(referent):
    if referent is None:
        return weakref.ref(_Dead())  # dies at once: a dead ref, as it was
    return weakref.ref(referent)


_WEAK_CONTAINERS = (weakref.WeakValueDictionary, weakref.WeakKeyDictionary, weakref.WeakSet)


def _rebuild_match(pattern, string, start, endpos, regs):
    """A regex match object: the same pattern re-matched at the same start must give the same groups."""
    match = pattern.match(string, start, endpos)
    if match is None or match.regs != regs:
        raise CheckpointError("a saved regex match does not re-match identically")
    return match


_MATCH_TYPE = type(__import__("re").match("", ""))


_CONTAINER_MUTATORS = {dict: "__setitem__", list: "append", set: "add"}


def _guarded_container(obj) -> type | None:
    """The base type of a Python-level dict/list/set subclass that overrides its mutator."""
    cls = type(obj)
    if cls in _CONTAINER_MUTATORS or cls.__module__ in {"collections", "builtins"}:
        return None
    for base, mutator in _CONTAINER_MUTATORS.items():
        if isinstance(obj, base) and getattr(cls, mutator) is not getattr(base, mutator):
            return base
    return None


def _new_lock():
    return threading.Lock()


def _new_rlock():
    return threading.RLock()


def _new_event(flag):
    event = threading.Event()
    if flag:
        event.set()
    return event


def _new_condition():
    return threading.Condition()


def _reopen(path, mode_, position):
    stream = open(path, mode_ if "a" in mode_ or "r" in mode_ else mode_.replace("w", "a"))
    try:
        if "a" not in mode_:
            stream.seek(position)
    except OSError:
        pass
    return stream


_LOCK_TYPE = type(threading.Lock())
_RLOCK_TYPE = type(threading.RLock())


def _lease_state(obj) -> dict:
    state = dict(vars(obj))
    state["_fd"] = None if state.get("_fd") is None else "OPEN"
    state.pop("_pid", None)
    return state


def _workspace_state(obj) -> dict:
    state = _lease_state(obj)
    root = _SAVE.get("root")
    if root is not None:
        try:
            state["_checkpoint_directory"] = obj.directory.relative_to(root.parent).as_posix()
        except ValueError:
            pass  # A lease outside this run keeps its absolute location.
    return state


# ---- append-only element store ----------------------------------------------------------------
# Most of a late checkpoint is a few append-only lists that mirror the journal, one item per journal
# record (step 85 of a T2 run: journal record cache, TimelineIndex._events, the capability-manifest
# journal projection = ~120 of ~150 MB). Their items never change once appended. Each such list is
# saved as immutable chunk files holding only the items added since the previous checkpoint, and
# every reference to one of its items anywhere in the main pickle becomes a tiny persistent id
# (key, index), so identity between the list and its indexes (_by_id, menu, ...) is kept. Chunk
# names and sha256s are in the signed metadata. Item graphs must be plain values (no references
# back into the mutable engine graph): records are JSON dicts, timeline events frozen dataclasses.
CACHE_DIR = "journal-cache"
# Chunks are zlib level 1 (written by the background writer, so no engine pause): the records are
# JSON-like dicts and compress ~4.6x (T2 live @55 sends: 338 -> 74 MB). sha256 covers the stored bytes.
CHUNK_SUFFIX = ".zpkl"
ARTIFACT_STAGE_DIR = "artifacts"
ARTIFACT_TRANSACTION_FILE = ".artifact-transaction.json"
_SAVE: dict = {}  # one dump: roots, prior/output indexes, persistent IDs, and restore tables
_PERSISTENT_ID_ATOMS = (str, bytes, int, float, bool, type(None))


class _ListRef:
    """Pickled stand-in for an append-only list stored as chunks."""

    def __init__(self, key, length):
        self.key, self.length = key, length


def _stable_key(*parts) -> str:
    return hashlib.sha256("\0".join(str(part) for part in parts).encode()).hexdigest()[:16]


class _DictRef(_ListRef):
    """Pickled stand-in for an append-only dict stored as (key, value) chunks."""


def _chunk_dict(key: str, mapping: dict):
    """Store an insertion-ordered, append-only dict incrementally; values get persistent ids."""
    if type(mapping) is not dict:
        return None
    pairs = list(mapping.items())
    ref = _chunk_list(key, pairs, guard=lambda pair: pair[0], values=True)
    return None if ref is None else _DictRef(ref.key, ref.length)


def _shape(value):
    """Key order of every nested dict (journal lines are written key-sorted, so loading one sorts
    them); None where the loaded line already has it (the usual case: nothing to apply on restore)."""
    if type(value) is dict:
        keys = list(value)
        shapes = tuple((key, _shape(item)) for key, item in value.items())
        if keys == sorted(keys) and all(shape is None for _, shape in shapes):
            return None
        return ("d", shapes)
    if type(value) is list:
        shapes = tuple(_shape(item) for item in value)
        return ("l", shapes) if any(shape is not None for shape in shapes) else None
    return None


def _apply_shape(shape, value):
    if shape is None:
        return value
    if shape[0] == "d":
        if type(value) is not dict or len(value) != len(shape[1]):
            raise CheckpointError("journal record does not match its saved shape")
        return {key: _apply_shape(sub, value[key]) for key, sub in shape[1]}
    if type(value) is not list or len(value) != len(shape[1]):
        raise CheckpointError("journal record does not match its saved shape")
    return [_apply_shape(sub, item) for sub, item in zip(shape[1], value)]


def _same(a, b) -> bool:
    """Equal with identical types and dict key order all the way down (1 == 1.0 == True is not)."""
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return list(a) == list(b) and all(_same(a[key], b[key]) for key in a)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _journal_line(line: bytes, journal: Path, blobs: dict, shape, known: dict | None = None):
    """One cached record from its journal line: parsed, blob references hydrated, key order applied.

    ``blobs`` caches blob bytes by reference hash. ``known`` (restore) maps every blob the save used to
    the sha256 of its stored bytes; a blob outside it, or with other bytes, is refused. Without
    ``known`` (save) each blob's stored-bytes sha256 is recorded in ``blobs["#sha"]``.
    """
    try:
        stored = json.loads(line)
    except ValueError:
        raise CheckpointError(f"journal {journal} line is not JSON") from None
    if type(stored) is not dict:
        raise CheckpointError(f"journal {journal} line is not an object")
    if b'"$blob"' in line:  # a line without that key holds no blob reference to hydrate
        stored = _hydrate(stored, journal, blobs, known)
    return _apply_shape(shape, stored)


def _hydrate(value, journal: Path, blobs: dict, known: dict | None):
    """Replace journal blob references ({"$blob", "bytes"}) by their decoded content, recursively.

    Same result as kernel_measurement.resolve_journal_blob_refs (each use a fresh object), without its
    re-encoding checks: at save the rebuilt record is compared with the live one, and at restore every
    blob's stored bytes must match the sha256 the save recorded."""
    if __package__:
        from .kernel_measurement import _blob_reference, _read_journal_blob
    else:
        from kernel_measurement import _blob_reference, _read_journal_blob
    try:
        reference = _blob_reference(value) if type(value) is dict and len(value) == 2 else None
    except ValueError:
        raise CheckpointError(f"journal {journal} blob reference is malformed") from None
    if reference is not None:
        blob_hash = reference[0]
        encoded = blobs.get(blob_hash)
        if encoded is None:
            try:
                encoded = _read_journal_blob(journal, blob_hash)
            except (OSError, ValueError):
                raise CheckpointError(f"journal {journal} blob {blob_hash[:12]} unreadable") from None
            sha = hashlib.sha256(encoded).hexdigest()
            if known is None:
                blobs.setdefault("#sha", {})[blob_hash] = sha
            elif known.get(blob_hash) != sha:
                raise CheckpointError(f"journal {journal} blob {blob_hash[:12]} hash mismatch")
            blobs[blob_hash] = encoded
        decoded = json.loads(encoded)
        return _hydrate(decoded, journal, blobs, known) if b'"$blob"' in encoded else decoded
    if type(value) is dict:
        if "$blob" in value:
            raise CheckpointError(f"journal {journal} blob reference is malformed")
        return {key: _hydrate(item, journal, blobs, known) for key, item in value.items()}
    if type(value) is list:
        return [_hydrate(item, journal, blobs, known) for item in value]
    return value


def _journal_entries(journal: Path, offset: int, records: list):
    """Entries for records[i] = journal line i after ``offset``: the record's shape when the line
    rebuilds it exactly, else the record itself. Returns (entries, end offset, sha256 of the line
    bytes, {blob hash: sha256 of its stored bytes}); None when the journal cannot cover them."""
    entries, shapes, blobs, lines = [], {}, {}, hashlib.sha256()
    try:
        if __package__:
            from .kernel_measurement import open_journal
        else:
            from kernel_measurement import open_journal
        with open_journal(journal, binary=True) as stream:
            stream.seek(offset)
            for record in records:
                line = stream.readline()
                if not line.endswith(b"\n"):
                    return None
                lines.update(line)
                shape = _shape(record)
                shape = shapes.setdefault(shape, shape)  # one pickled copy per distinct shape
                try:
                    exact = _same(_journal_line(line, journal, blobs, shape), record)
                except (CheckpointError, OSError, ValueError, TypeError, KeyError):
                    exact = False
                entries.append(("J", shape) if exact else ("R", record))
            return entries, stream.tell(), lines.hexdigest(), blobs.get("#sha", {})
    except OSError:
        return None


def _journal_offset(journal: Path, lines: int):
    """Byte offset after the first ``lines`` lines (for a lineage started before journal backing)."""
    try:
        if __package__:
            from .kernel_measurement import open_journal
        else:
            from kernel_measurement import open_journal
        with open_journal(journal, binary=True) as stream:
            for _ in range(lines):
                if not stream.readline().endswith(b"\n"):
                    return None
            return stream.tell()
    except OSError:
        return None


def _load_journal_chunk(part: dict) -> list:
    """Rebuild a journal-backed chunk: its line bytes and every blob must hash as at the save."""
    journal = Path(part["journal"])
    try:
        if __package__:
            from .kernel_measurement import open_journal
        else:
            from kernel_measurement import open_journal
        with open_journal(journal, binary=True) as stream:
            stream.seek(part["offset"])
            data = stream.read(part["end"] - part["offset"])
    except OSError as error:
        raise CheckpointError(f"journal {journal} unreadable: {type(error).__name__}") from None
    if len(data) != part["end"] - part["offset"]:
        raise CheckpointError(f"journal {journal} shorter than its checkpoint chunk")
    if hashlib.sha256(data).hexdigest() != part["sha"]:
        raise CheckpointError(f"journal {journal} lines hash mismatch")
    lines = data.splitlines(keepends=True)
    if len(lines) != len(part["entries"]):
        raise CheckpointError(f"journal {journal} chunk has the wrong line count")
    blobs, known = {}, part["blobs"]
    return [_journal_line(line, journal, blobs, value, known) if kind == "J" else value
            for line, (kind, value) in zip(lines, part["entries"])]


def _chunk_list(key: str, items: list, *, guard=None, values: bool = False, journal: Path | None = None):
    """Store ``items`` incrementally under ``key``; returns a _ListRef, or None to pickle inline.

    ``guard`` returns a stable value for the last list item and is recorded per chunk; when the
    item at the end of the covered prefix no longer matches, the lineage restarts from scratch.
    Dict mode guards the last key. A length check catches truncation before any chunks are reused.

    ``journal``: the items mirror that append-only JSONL file line for line (the continuation
    journal's record cache). A chunk then stores, per record, only the dict key order needed to
    rebuild it from its own verified line, keeping the record itself only where that rebuild is not
    exact; each row also records the byte offset its chunk ends at.
    """
    root = _SAVE.get("root")
    if root is None or type(items) is not list:
        return None  # probe dumps and unusual containers stay inline
    chunks = [tuple(row) for row in _SAVE.get("index", {}).get(key, ())]
    covered = chunks[-1][2] if chunks else 0
    if len(items) < covered:  # the owner rebuilt the list: start the lineage again
        chunks, covered = [], 0
    check_guard = (lambda pair: pair[0]) if values else guard
    if check_guard is not None and covered:
        if len(chunks[-1]) < 5:
            chunks, covered = [], 0
        else:
            try:
                matches = _same(check_guard(items[covered - 1]), chunks[-1][4])
            except Exception:  # noqa: BLE001 - an unverifiable index stays inline
                return None
            if not matches:
                chunks, covered = [], 0  # the covered prefix changed: rewrite it
    if len(items) > covered:
        try:
            tail_guard = check_guard(items[-1]) if check_guard is not None else None
        except Exception:  # noqa: BLE001 - an unverifiable index stays inline
            return None
        payload, end_offset = items[covered:], None
        if journal is not None:
            start = (chunks[-1][4] if chunks and len(chunks[-1]) > 4
                     else 0 if not covered else _journal_offset(journal, covered))
            backed = None if start is None else _journal_entries(journal, start, items[covered:])
            if backed is not None:
                entries, end_offset, sha, blob_shas = backed
                payload = {"journal": str(journal), "offset": start, "end": end_offset, "sha": sha,
                           "blobs": blob_shas, "entries": entries}
        try:
            raw = pickle.dumps(payload, protocol=5)
        except Exception:  # noqa: BLE001 - an item that is not a plain value: keep the list inline
            return None
        _check_registered_secrets(raw)
        try:
            blob = zlib.compress(raw, 1)
        except Exception:  # noqa: BLE001 - an item that is not a plain value: keep the list inline
            return None
        name = f"{key}-{covered:07d}-{len(items):07d}-{os.getpid()}-{os.urandom(8).hex()}{CHUNK_SUFFIX}"
        artifact_root = _SAVE.get("artifact_root", root)
        folder = Path(artifact_root) / CACHE_DIR
        _private_dir(folder)
        temporary = folder / f".{name}.tmp"
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, folder / name)
        row = (name, covered, len(items), hashlib.sha256(blob).hexdigest())
        stored_guard = tail_guard if check_guard is not None else end_offset
        chunks.append(row + ((stored_guard,) if stored_guard is not None else ()))
    _SAVE.setdefault("index_out", {})[key] = [list(row) for row in chunks]
    ids = _SAVE.setdefault("ids", {})
    for position, item in enumerate(items):
        if values:  # a dict: its VALUES are referenced elsewhere (receipts -> same event objects)
            value = item[1]
            if type(value) not in _PERSISTENT_ID_ATOMS:
                ids.setdefault(id(value), (key, position, 1))
        elif type(item) not in _PERSISTENT_ID_ATOMS:
            ids.setdefault(id(item), (key, position))
    return _ListRef(key, len(items))


def _table(key: str) -> list:
    """The items of ``key`` for this restore, loaded once from its verified chunks."""
    tables = _SAVE.setdefault("tables", {})
    if key in tables:
        return tables[key]
    folder, index = _SAVE.get("load_root"), _SAVE.get("load_index", {})
    if folder is None or key not in index:
        raise CheckpointError(f"append-only list {key} referenced outside a restore or not indexed")
    items: list = []
    for name, start, end, sha, *_guard in (tuple(row) for row in index[key]):
        try:
            blob = (folder / CACHE_DIR / name).read_bytes()
        except OSError:
            raise CheckpointError(f"chunk {name} missing") from None
        if hashlib.sha256(blob).hexdigest() != sha or start != len(items):
            raise CheckpointError(f"chunk {name} damaged or out of order")
        try:
            part = pickle.loads(zlib.decompress(blob) if name.endswith(CHUNK_SUFFIX) else blob)
        except zlib.error:
            raise CheckpointError(f"chunk {name} damaged") from None
        if type(part) is dict:
            part = _load_journal_chunk(part)
        if len(part) != end - start:
            raise CheckpointError(f"chunk {name} has the wrong length")
        items.extend(part)
    tables[key] = items
    return items


def _list_from_ref(ref) -> list:
    items = _table(ref.key)
    if len(items) < ref.length:
        raise CheckpointError(f"append-only list {ref.key} shorter than the checkpoint")
    return list(items[:ref.length])


def _journal_key(obj) -> str:
    return _stable_key("journal", getattr(obj, "path", ""))


def _journal_state(obj) -> dict:
    state = _lease_state(obj)
    for name in DROP_ATTRS.get((type(obj).__module__, type(obj).__qualname__), ()):
        state.pop(name, None)
    # Repeated-cursor snapshots are derived cache data; do not duplicate them
    # beside the authoritative journal records in a checkpoint.
    state.pop("_records_since_cache", None)
    state.pop("_records_since_cache_bytes", None)
    path = getattr(obj, "path", None)
    if state.get("_record_cache_start", 0):
        # This is a bounded suffix, not the line-zero append-only list expected
        # by _chunk_list. Restore validates the whole journal before use.
        state["_checkpoint_validate_journal"] = True
        return state
    ref = _chunk_list(_journal_key(obj), state.get("_record_cache"),
                      journal=Path(path) if path else None)
    if ref is not None:
        state["_record_cache"] = ref
    for name, kind in (("_record_hashes", "journal-hashes"), ("_record_offsets", "journal-offsets")):
        values = state.get(name)
        ref = _chunk_list(_stable_key(kind, _journal_key(obj)), values, guard=lambda item: item)
        if ref is not None:
            state[name] = ref
    return state


def _timeline_state(obj):
    """TimelineIndex: append-only events and event positions keyed by its session id."""
    reduced = obj.__reduce_ex__(5)
    state = dict(reduced[2]) if len(reduced) > 2 and isinstance(reduced[2], dict) else None
    if state is None:
        return NotImplemented
    key = _stable_key("timeline", state.get("session_id"))
    changed = False
    events = state.get("_events")
    ref = _chunk_list(key, events, guard=_chunk_item_guard) if type(events) is list else None
    if ref is not None:
        state["_events"] = ref
        changed = True
    positions = state.get("_position_by_id")
    ref = _chunk_dict(_stable_key("timeline-position", key), positions) if type(positions) is dict else None
    if ref is not None:
        state["_position_by_id"] = ref
        lookup = state.get("_by_id")
        # kernel_timeline builds its EventLookup from the flat child module, so a
        # package-relative import here would name a different class object and
        # isinstance would miss it. Match the alias by shape and rebuild with the
        # instance's own class so restore keeps the original class identity.
        if (type(lookup).__name__ == "EventLookup"
                and getattr(lookup, "positions", None) is positions):
            # EventLookup's reducer otherwise serializes a second full copy of
            # the same position map. Share this stand-in so restore also keeps
            # the original alias between both timeline fields.
            state["_by_id"] = type(lookup)(lookup.events, ref)
        changed = True
    if not changed:
        return NotImplemented
    return (reduced[0], reduced[1], state) + tuple(reduced[3:])


def _chunk_item_guard(item) -> str:
    """Small stable guard for the last immutable record in an append-only list."""
    return hashlib.sha256(pickle.dumps(item, protocol=5)).hexdigest()


def _generated_artifact_state(obj):
    reduced = obj.__reduce_ex__(5)
    state = dict(reduced[2]) if len(reduced) > 2 and isinstance(reduced[2], dict) else None
    if state is None:
        return NotImplemented
    ref = _chunk_dict(_stable_key("generated-artifacts", state.get("run_id")), state.get("_events"))
    if ref is None:
        return NotImplemented
    state["_events"] = ref
    return (reduced[0], reduced[1], state) + tuple(reduced[3:])


def _world_knowledge_state(obj):
    reduced = obj.__reduce_ex__(5)
    state = dict(reduced[2]) if len(reduced) > 2 and isinstance(reduced[2], dict) else None
    if state is None:
        return NotImplemented
    scope = json.dumps(state.get("scope", {}), sort_keys=True, separators=(",", ":"))
    key = _stable_key("world-knowledge-events", _SAVE.get("root"), scope)
    events = state.get("_events")
    ref = _chunk_list(key, events, guard=_chunk_item_guard) if type(events) is list else None
    if ref is None:
        return NotImplemented
    state["_events"] = ref
    return (reduced[0], reduced[1], state) + tuple(reduced[3:])


def _recall_state(obj):
    reduced = obj.__reduce_ex__(5)
    state = dict(reduced[2]) if len(reduced) > 2 and isinstance(reduced[2], dict) else None
    if state is None:
        return NotImplemented
    items = state.get("_items")
    if type(items) is not list or not items:
        return NotImplemented
    first = items[0].get("id", "") if type(items[0]) is dict else ""
    key = _stable_key("recall-items", _SAVE.get("root"), first)
    ref = _chunk_list(key, items, guard=_chunk_item_guard)
    if ref is None:
        return NotImplemented
    state["_items"] = ref
    return (reduced[0], reduced[1], state) + tuple(reduced[3:])


def _runtime_state(obj, dropped):
    """StateFrameRuntime: drop id()-keyed caches; the capability-manifest journal projection's
    records as an append-only list keyed by the journal path it projects."""
    reduced = obj.__reduce_ex__(5)
    if not (isinstance(reduced, tuple) and len(reduced) >= 3 and isinstance(reduced[2], dict)):
        raise pickle.PicklingError(f"cannot reduce {type(obj).__qualname__}")
    state = {k: v for k, v in reduced[2].items() if k not in dropped}
    projection = state.get("_capability_manifest_journal_projection")
    if isinstance(projection, dict) and type(projection.get("records")) is list:
        ref = _chunk_list(_stable_key("capability-manifest", projection.get("path")), projection["records"])
        if ref is not None:
            projection = dict(projection)
            projection["records"] = ref
            state["_capability_manifest_journal_projection"] = projection
    return (reduced[0], reduced[1], state) + tuple(reduced[3:])


def _load_list(key, length):
    return _list_from_ref(_ListRef(key, length))


def _load_dict(key, length):
    return dict(_list_from_ref(_ListRef(key, length)))


def _evidence_state(obj):
    """HostEvidence (continuation executive): ``events`` mirrors evidence journal records, one per
    unique event_id, never replaced; ``receipts`` points at the same event objects."""
    reduced = obj.__reduce_ex__(5)
    state = dict(reduced[2]) if len(reduced) > 2 and isinstance(reduced[2], dict) else None
    if state is None:
        return NotImplemented
    if __package__:
        from .kernel_memfix_history import SpoolEventMap
    else:
        from kernel_memfix_history import SpoolEventMap
    if isinstance(state.get("events"), SpoolEventMap):
        # Every other HostEvidence collection is a rebuildable index or a
        # bounded decoded view. The spool and receipt IDs are the authority.
        state = {key: state[key] for key in (
            "run_id", "source_refresh", "native_prefix", "native_consumed",
            "events", "receipts") if key in state}
        return (reduced[0], reduced[1], state, None, None, _restore_evidence_state)
    ref = _chunk_dict(_stable_key("evidence", state.get("run_id")), state.get("events"))
    if ref is None:
        return NotImplemented
    state["events"] = ref
    return (reduced[0], reduced[1], state, None, None, _restore_evidence_state)


def _restore_evidence_state(obj, state):
    obj.__setstate__(state)
    rebuild = getattr(obj, "rebuild_indexes", None)
    if rebuild is not None:
        rebuild()


# Owners whose append-only lists go to the element store: (module, qualname) -> state function.
CHUNKED_TYPES = {
    ("kernel_timeline", "TimelineIndex"): _timeline_state,
    ("kernel_continuation_evidence", "HostEvidence"): _evidence_state,
    ("kernel_generated_artifacts", "GeneratedArtifactIndex"): _generated_artifact_state,
    ("detreason.world_knowledge", "WorldKnowledgeStore"): _world_knowledge_state,
    ("kernel_recall", "RecallStore"): _recall_state,
}


def _new_instance(cls):
    return object.__new__(cls)


def _rebuild_journal(obj, state):
    """ContinuationJournal: re-open the restored events.jsonl under a fresh flock lease."""
    was_open = state.pop("_fd") == "OPEN"
    validate_journal = state.pop("_checkpoint_validate_journal", False)
    rebuild_indexes = any(name not in state for name in (
        "_record_offsets", "_record_hashes", "_request_sequences",
        "_tool_reconciliation_records", "_journal_applied_edit_rows",
        "_vacuity_latest_check", "_vacuity_last_edit"))
    rebuild_indexes = rebuild_indexes or not isinstance(
        state.get("_tool_reconciliation_records"), dict)
    saved_size = state.get("_size")
    obj.__dict__.update(state)
    obj._fd = None
    obj._pid = os.getpid()
    obj._warm_host_span_pause_counts = {}
    obj._warm_host_span_resume_counts = {}
    from collections import OrderedDict
    if __package__:
        from .kernel_continuation_journal import MEMBOUND_ENV, RECORD_CACHE_LIMIT
    else:
        from kernel_continuation_journal import MEMBOUND_ENV, RECORD_CACHE_LIMIT
    obj._records_since_cache = OrderedDict()
    obj._records_since_cache_bytes = 0
    obj._record_cache_limit = (None if os.environ.get(MEMBOUND_ENV, "bounded").lower() == "legacy"
                               else RECORD_CACHE_LIMIT)
    obj._record_cache_start = getattr(obj, "_record_cache_start", 0)
    obj._trim_record_cache()
    if was_open and not _STATE.get("probe_load"):
        obj._open(create=False)
        try:
            if obj._size != saved_size:
                raise CheckpointError(f"restored journal {obj.path} size {obj._size} != {saved_size}")
            if rebuild_indexes:
                # A PC3G checkpoint predates these projections. Reconstruct
                # them once from authenticated rows before the first append.
                for _record in obj.records():
                    pass
                if (len(obj._record_hashes) != obj._sequence
                        or (obj._record_hashes[-1] if obj._record_hashes else None) != obj._head):
                    raise CheckpointError("checkpoint journal head does not match saved state")
            elif validate_journal:
                obj.validate_checkpoint_journal()
                if obj._record_cache_limit is None:
                    for _record in obj.records():
                        pass
        except BaseException as error:
            os.close(obj._fd)
            obj._fd = None
            if (validate_journal or rebuild_indexes) and not isinstance(error, CheckpointError):
                raise CheckpointError(f"restored journal validation failed: {error}") from error
            raise


def _rebuild_workspace(obj, state):
    """ContinuationWorkspace: re-acquire owner.lock and re-bind the restored directory identity."""
    import fcntl
    import stat as stat_module
    was_open = state.pop("_fd") == "OPEN"
    relative = state.pop("_checkpoint_directory", None)
    obj.__dict__.update(state)
    obj._fd = None
    obj._pid = os.getpid()
    if relative is not None and not _STATE.get("probe_load"):
        if not isinstance(relative, str):
            raise CheckpointError("invalid restored workspace directory")
        path = Path(relative)
        run_dir = _SAVE.get("load_run_dir")
        if not path.parts or path.is_absolute() or ".." in path.parts or run_dir is None:
            raise CheckpointError("invalid restored workspace directory")
        obj.directory = run_dir / path
    if was_open and not _STATE.get("probe_load"):
        fd = os.open(obj.directory / "owner.lock", os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            opened = os.fstat(fd)
            if not stat_module.S_ISREG(opened.st_mode):
                raise CheckpointError("restored workspace lease is not a regular file")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            value = obj.directory.lstat()
            lock = (obj.directory / "owner.lock").lstat()
            if (not stat_module.S_ISDIR(value.st_mode)
                    or (lock.st_dev, lock.st_ino) != (opened.st_dev, opened.st_ino)):
                raise CheckpointError("restored workspace ownership changed while acquiring lease")
        except BaseException:
            os.close(fd)
            raise
        obj._fd = fd
        obj._identity = (value.st_dev, value.st_ino)


def _restored_job_handle(ident, lost_bytes):
    if __package__:
        from .kernel_exec_session import RestoredJobHandle
    else:
        from kernel_exec_session import RestoredJobHandle
    return RestoredJobHandle(ident, lost_bytes)


def _exec_session_state(obj) -> dict:
    """ExecSession: keep only what a fresh session needs. The supervisor process, its control
    socket, log file and lock belong to the saving process; live jobs already make a step
    checkpoint-ineligible, so no job handle outlives the save."""
    return {"run_dir": str(obj.run_dir), "backend": obj.backend,
            "capability": obj._initial_capability, "closed": obj._closed}


def _rebuild_exec_session(obj, state):
    """Restore an unstarted session in a fresh exec directory; it launches its supervisor lazily."""
    fresh = type(obj)(Path(state["run_dir"]), platform_backend=state["backend"])
    obj.__dict__.update(fresh.__dict__)
    obj._initial_capability = state["capability"]
    obj._closed = state["closed"]


# Process-bound engine classes: (module, qualname) -> (state getter, state setter that re-acquires).
LEASE_TYPES = {
    ("kernel_continuation_journal", "ContinuationJournal"): (_journal_state, _rebuild_journal),
    ("kernel_continuation_workspace", "ContinuationWorkspace"): (_workspace_state, _rebuild_workspace),
    ("kernel_exec_session", "ExecSession"): (_exec_session_state, _rebuild_exec_session),
}


class _Unsupported(Exception):
    pass


class _Pickler(pickle.Pickler):
    def __init__(self, stream, ctx, protocol=5):
        super().__init__(stream, protocol=protocol)
        self.ctx = ctx
        self.workspace_objects_on = _workspace_objects_on()

    def persistent_id(self, obj):
        ctx = self.ctx
        if ((type(obj).__module__, type(obj).__qualname__)
                == ("stateframe.startup_cache_writes", "StartupCacheWriter")):
            # This service is installed before restore. Save only its binding,
            # never its queue, context, reader handles, or writer thread.
            from stateframe.startup_cache_writes import active_writer

            if active_writer() is not obj:
                raise pickle.PicklingError("startup cache checkpoint has no current host binding")
            return ("host-startup-cache-writer",)
        if isinstance(obj, types.ModuleType):
            return ("mod", obj.__name__)
        if (self.workspace_objects_on and (type(obj).__module__, type(obj).__qualname__)
                == ("kernel_shell_tool", "ShellTool")):
            from kernel_shell_tool import service

            root = getattr(obj, "root", None)
            if not isinstance(root, Path) or service() is not obj:
                raise pickle.PicklingError("shell checkpoint has no current host binding")
            return ("host-shell-tool", os.fspath(root))
        if (self.workspace_objects_on and (type(obj).__module__, type(obj).__qualname__)
                == ("kernel_agent_tools", "AgentTools")):
            from kernel_agent_tools import service

            root = getattr(getattr(obj, "shell", None), "root", None)
            if not isinstance(root, Path) or service() is not obj:
                raise pickle.PicklingError("agent tools checkpoint has no current host binding")
            return ("host-agent-tools", os.fspath(root))
        if (self.workspace_objects_on and (type(obj).__module__, type(obj).__qualname__)
                in {("kernel_workspace_consumers", "WorkspaceConsumers"),
                    ("kernel_workspace_state", "WorkspaceState")}):
            from kernel_workspace_consumers import consumers_for

            state = obj.state if type(obj).__qualname__ == "WorkspaceConsumers" else obj
            root = getattr(getattr(state, "fence", None), "root", None)
            consumers = consumers_for(root) if isinstance(root, Path) else None
            if consumers is None or consumers.state is not state or (
                    type(obj).__qualname__ == "WorkspaceConsumers" and consumers is not obj):
                raise pickle.PicklingError("workspace checkpoint has no current host binding")
            return ("host-workspace-consumers" if consumers is obj else "host-workspace-state",
                    os.fspath(root))
        ident = id(obj)
        element = _SAVE.get("ids", {}).get(ident)
        if element is not None:
            return ("elem",) + element
        if ident in ctx["hostfns"]:
            return ("hostfn",) + ctx["hostfns"][ident]
        if ident in ctx["participants"]:
            return ("participant", ctx["participants"][ident])
        if ident in ctx["framecells"]:
            return ("framecell", ctx["framecells"][ident])
        if ident in ctx["framefuncs"]:
            return ("framefunc", ctx["framefuncs"][ident])
        if isinstance(obj, contextvars.ContextVar):
            where = ctx["cvars"].get(ident)
            if where is None:
                raise pickle.PicklingError(f"unregistered ContextVar {obj.name}")
            return ("cvar",) + where
        return None

    def reducer_override(self, obj):
        clock = _clock_name(obj) if _REAL_CLOCKS else None
        if clock is not None:
            return (_current_clock, (clock,))
        if isinstance(obj, types.FunctionType):
            if _resolves_to(obj):
                return NotImplemented
            return (_rebuild_function, (
                marshal.dumps(obj.__code__), _globals_module(obj), obj.__name__, obj.__qualname__,
                obj.__defaults__, obj.__closure__, obj.__kwdefaults__, dict(obj.__dict__) or None,
                obj.__module__))
        if isinstance(obj, _DictRef):
            return (_load_dict, (obj.key, obj.length))
        if isinstance(obj, _ListRef):
            return (_load_list, (obj.key, obj.length))
        owner_key = (type(obj).__module__, type(obj).__qualname__)
        if owner_key == ("kernel_exec_session", "JobHandle"):
            # Live jobs make a step checkpoint-ineligible, so this job finished in the saving
            # process; restore it detached instead of bound to a supervisor that no longer exists.
            return (_restored_job_handle, (obj.id, obj.lost_bytes))
        if owner_key == ("kernel_capability_projection", "CapabilityProjection"):
            return obj.__reduce_ex__(5)
        if owner_key[0] == "kernel_history_receipts" and owner_key[1] in {"ReceiptDict", "ReceiptList"}:
            # These mutable views have an explicit owner/index reduction. The
            # generic guarded-container path would serialize their weak state.
            return obj.__reduce_ex__(5)
        dropped = DROP_ATTRS.get(owner_key)
        if dropped is not None and owner_key not in LEASE_TYPES:
            return _runtime_state(obj, dropped)
        chunked = CHUNKED_TYPES.get(owner_key)
        if chunked is not None:
            result = chunked(obj)
            if result is not NotImplemented:
                return result
        lease = LEASE_TYPES.get((type(obj).__module__, type(obj).__qualname__))
        if lease is not None:
            get_state, rebuild = lease
            return (_new_instance, (type(obj),), get_state(obj), None, None, rebuild)
        frozen = _frozen_sequence(obj)
        if frozen is not None:
            return (_new_frozen, (type(obj), frozen, frozen(obj)), dict(getattr(obj, "__dict__", {}) or {}) or None,
                    None, None, _set_attrs)
        base = _guarded_container(obj)
        if base is not None:
            items = list(obj.items()) if base is dict else list(obj)
            if base is dict:
                items = dict(items)
            return (_new_container, (type(obj), base), (items, dict(getattr(obj, "__dict__", {}) or {})),
                    None, None, _fill_container)
        if isinstance(obj, contextvars.Token):
            return (_rebuild_token, (obj.var, _token_old(obj)))
        if isinstance(obj, types.MappingProxyType):
            import gc
            referents = gc.get_referents(obj)
            if len(referents) != 1:
                raise pickle.PicklingError("mappingproxy without a single referent")
            return (types.MappingProxyType, (referents[0],))
        if isinstance(obj, type) and not _resolves_to(obj):
            name = _TYPES_BY_ID.get(id(obj))
            if name is not None:
                return (_builtin_type, (name,))
            return NotImplemented
        if isinstance(obj, types.CellType):
            try:
                contents = obj.cell_contents
            except ValueError:
                return (_empty_cell, ())
            return (_empty_cell, (), (contents,), None, None, _set_cell)
        if isinstance(obj, _LOCK_TYPE):
            return (_new_lock, ())
        if isinstance(obj, _RLOCK_TYPE):
            return (_new_rlock, ())
        if isinstance(obj, threading.Event):
            return (_new_event, (obj.is_set(),))
        if isinstance(obj, threading.Condition):
            return (_new_condition, ())
        if isinstance(obj, io.IOBase) and not isinstance(obj, (io.BytesIO, io.StringIO)):
            name = getattr(obj, "name", None)
            if obj.closed:
                return (io.StringIO, ())
            if isinstance(name, str) and os.path.isabs(name):
                try:
                    position = obj.tell()
                except OSError:
                    position = 0
                return (_reopen, (name, getattr(obj, "mode", "a"), position))
            raise pickle.PicklingError(f"unsupported stream {type(obj).__name__} name={name!r}")
        if isinstance(obj, _MATCH_TYPE):
            return (_rebuild_match, (obj.re, obj.string, obj.start(), obj.endpos, obj.regs))
        if isinstance(obj, _WEAK_CONTAINERS):
            items = list(obj) if isinstance(obj, weakref.WeakSet) else list(obj.items())
            return (_new_weak, (type(obj),), (items,), None, None, _fill_weak)
        if type(obj) is weakref.ref:
            return (_rebuild_ref, (obj(),))
        if isinstance(obj, (types.GeneratorType, types.CoroutineType, types.AsyncGeneratorType,
                            types.FrameType, threading.Thread, weakref.ref)):
            raise pickle.PicklingError(f"unsupported {type(obj).__name__}")
        return NotImplemented


class _Unpickler(pickle.Unpickler):
    def __init__(self, stream, ctx):
        super().__init__(stream)
        self.ctx = ctx

    def persistent_load(self, pid):
        kind = pid[0]
        if kind == "host-startup-cache-writer":
            if len(pid) != 1:
                raise CheckpointError("invalid startup cache checkpoint binding")
            from stateframe.startup_cache_writes import active_writer

            writer = active_writer()
            if ((type(writer).__module__, type(writer).__qualname__)
                    != ("stateframe.startup_cache_writes", "StartupCacheWriter")):
                raise CheckpointError("restored startup cache binding is unavailable")
            return writer
        if kind == "host-shell-tool":
            if len(pid) != 2 or not isinstance(pid[1], str):
                raise CheckpointError("invalid shell checkpoint binding")
            from kernel_shell_tool import service

            shell = service()
            if shell is None or getattr(shell, "root", None) != Path(pid[1]):
                raise CheckpointError("restored shell workspace binding is unavailable")
            return shell
        if kind == "host-agent-tools":
            if len(pid) != 2 or not isinstance(pid[1], str):
                raise CheckpointError("invalid agent tools checkpoint binding")
            from kernel_agent_tools import service

            agent_tools = service()
            if (agent_tools is None
                    or getattr(getattr(agent_tools, "shell", None), "root", None) != Path(pid[1])):
                raise CheckpointError("restored agent tools workspace binding is unavailable")
            return agent_tools
        if kind in {"host-workspace-consumers", "host-workspace-state"}:
            if len(pid) != 2 or not isinstance(pid[1], str):
                raise CheckpointError("invalid workspace checkpoint binding")
            from kernel_workspace_consumers import consumers_for

            consumers = consumers_for(Path(pid[1]))
            if consumers is None or consumers.state.fence.root != Path(pid[1]):
                raise CheckpointError("restored workspace binding is unavailable")
            return consumers if kind == "host-workspace-consumers" else consumers.state
        if kind == "elem":
            item = _table(pid[1])[pid[2]]
            return item[pid[3]] if len(pid) > 3 else item
        if kind == "hostfn":
            return _resolve_host(pid[1:])
        if kind == "participant":
            if pid[1] not in _PARTICIPANTS:
                raise CheckpointError(f"host did not re-register participant {pid[1]!r}")
            return _PARTICIPANTS[pid[1]]
        if kind == "mod":
            __import__(pid[1])
            return sys.modules[pid[1]]
        if kind == "framecell":
            cell = self.ctx["framecells"].get(pid[1])
            if cell is None:
                raise CheckpointError(f"restored frame has no cell {pid[1]!r}")
            return cell
        if kind == "framefunc":
            fn = self.ctx["framefuncs"].get(pid[1])
            if fn is None:
                raise CheckpointError(f"restored frame has no function {pid[1]!r}")
            return fn
        if kind == "cvar":
            __import__(pid[1])
            return getattr(sys.modules[pid[1]], pid[2])
        raise CheckpointError(f"unknown persistent id {pid!r}")


# --------------------------------------------------------------------------- frame helpers

def _frame_maps(frame, setup: dict | None = None) -> tuple[dict[str, object], dict[str, object]]:
    """Cells and nested functions of ``frame``'s code, found through its live closures.

    With ``setup`` (what existed at the restore hook of this run), only functions a restoring
    process recreates before that hook qualify: defined above it and present there. A function
    defined later (inside the step loop) is saved by value instead.
    """
    code = frame.f_code
    nested = {c for c in code.co_consts if isinstance(c, types.CodeType)}
    cells: dict[str, object] = {}
    funcs: dict[str, object] = {}
    own_cells = set(code.co_cellvars) | set(code.co_freevars)
    for name, value in list(frame.f_locals.items()):
        if not (isinstance(value, types.FunctionType) and value.__code__ in nested):
            continue
        if setup is not None and (name not in setup["funcs"]
                                  or value.__code__.co_firstlineno >= setup["line"]):
            continue
        funcs[name] = value
        for var, cell in zip(value.__code__.co_freevars, value.__closure__ or ()):
            if var in own_cells:
                cells.setdefault(var, cell)
    return cells, funcs


def _frame_state(frame) -> dict[str, object]:
    out = {}
    for name, value in list(frame.f_locals.items()):
        if _workspace_objects_on() and name == "capture_candidate":
            # The bridge callable is re-imported when used. Saving its bound
            # ProposalCapture would serialize the workspace secret policy.
            continue
        if _is_static(value):
            continue
        out[name] = value
    return out


# --------------------------------------------------------------------------- capture

def _engine_classes() -> list[type]:
    seen: list[type] = []
    for name, module in list(sys.modules.items()):
        if module is None or not _is_state_module(name, module):
            continue
        for value in list(vars(module).values()):
            if isinstance(value, type) and value.__module__ == name:
                seen.append(value)
    return seen


_SCANS: dict = {}


def _scanned(name: str, function):
    """Module-wide scans (ContextVar names, host-function locations, engine classes) change only when
    modules are imported; reuse them while sys.modules keeps its size. A stale entry can only make a
    save fail closed (an unregistered object), and save() then rescans once."""
    key = len(sys.modules)
    cached = _SCANS.get(name)
    if cached is not None and cached[0] == key:
        return cached[1]
    value = function()
    _SCANS[name] = (key, value)
    return value


def collect(frame) -> tuple[dict, dict]:
    setup = _STATE.get("setup")
    if setup is None or setup["code"] is not frame.f_code:
        raise CheckpointError("the restore hook did not run before the step loop of this frame")
    cells, funcs = _frame_maps(frame, setup)
    payload = {
        "frame": _frame_state(frame),
        "modules": {name: _module_state(name, module) for name, module in list(sys.modules.items())
                    if module is not None and _is_state_module(name, module)},
        "classes": {},
        "patched": _patched_callables(),
        "context": _context_items(),
        "random": random.getstate(),
        "participants": {name: dict(state) for name, state in _PARTICIPANTS.items()},
    }
    for cls in _scanned("classes", _engine_classes):
        state = _class_state(cls)
        if state:
            payload["classes"][f"{cls.__module__}:{cls.__qualname__}"] = state
    ctx = {"framecells": {id(c): n for n, c in cells.items()},
           "framefuncs": {id(f): n for n, f in funcs.items()},
           "cvars": _scanned("cvars", _cvar_names), "hostfns": _scanned("hostfns", _host_functions),
           "participants": {id(state): name for name, state in _PARTICIPANTS.items()}}
    return payload, ctx


def dumps(payload, ctx, protocol=5) -> bytes:
    stream = io.BytesIO()
    _Pickler(stream, ctx, protocol).dump(payload)
    return stream.getvalue()


def dump_to(stream, payload, ctx, protocol=5) -> None:
    """Serialize with the same pickler as dumps(), directly into a writable stream."""
    _Pickler(stream, ctx, protocol).dump(payload)


_PATH_NODE_LIMIT = 500_000


def _serialization_field_path(root) -> str:
    """Find a known non-serializable field without including mapping keys or values."""
    from collections import deque

    unsupported = (contextvars.Context, types.GeneratorType, types.CoroutineType,
                   types.AsyncGeneratorType, types.FrameType, threading.Thread)
    seen: set[int] = set()
    queue = deque([(root, "payload")])
    while queue and len(seen) < _PATH_NODE_LIMIT:
        obj, where = queue.popleft()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, unsupported):
            return where[:500]
        if isinstance(obj, (str, bytes, int, float, type, types.ModuleType,
                            types.FunctionType, contextvars.ContextVar)):
            continue
        if isinstance(obj, dict):
            queue.extend((value, f"{where}.mapping_value[{index}]")
                         for index, value in enumerate(obj.values()))
        elif isinstance(obj, (list, tuple, set, frozenset)):
            queue.extend((value, f"{where}.sequence_item[{index}]")
                         for index, value in enumerate(obj))
        else:
            fields = getattr(obj, "__dict__", None)
            if isinstance(fields, dict):
                queue.extend((value, f"{where}.{name}") for name, value in fields.items())
    return "unresolved"


def _mark_serialization_failure(error: BaseException, payload: object) -> None:
    """Attach only a fixed exception kind and a value-free field path to the exception."""
    if isinstance(error, pickle.PicklingError):
        error_type = "PicklingError"
    elif type(error) in {TypeError, AttributeError, RecursionError}:
        error_type = type(error).__name__
    else:
        return
    try:
        BaseException.__setattr__(error, "_stateframe_serialization_error_type", error_type)
        BaseException.__setattr__(error, "_stateframe_serialization_field_path",
                                  _serialization_field_path(payload))
    except BaseException:
        pass


def _serialization_failure_fields(error: BaseException) -> dict[str, str]:
    try:
        error_type = BaseException.__getattribute__(error, "_stateframe_serialization_error_type")
        field_path = BaseException.__getattribute__(error, "_stateframe_serialization_field_path")
    except BaseException:
        return {}
    if (type(error_type) is not str or not error_type.isidentifier() or len(error_type) > 80
            or type(field_path) is not str or not field_path or len(field_path) > 500):
        return {}
    return {"serialization_error_type": error_type,
            "serialization_field_path": field_path}


def _safe_error_type(error: BaseException) -> str:
    if isinstance(error, pickle.PicklingError):
        return "PicklingError"
    import builtins

    error_class = type(error)
    if (error_class.__module__ == "builtins" and error_class.__name__.isidentifier()
            and len(error_class.__name__) <= 80
            and vars(builtins).get(error_class.__name__) is error_class):
        return error_class.__name__
    for known in (TypeError, ValueError, AttributeError, KeyError, IndexError, RuntimeError,
                  OSError, MemoryError):
        if type(error) is known:
            return known.__name__
    return "OtherError"


def _log_serialization_failure(fields: dict[str, str]) -> None:
    if fields:
        print("checkpoint: serialization failed "
              f"({fields['serialization_error_type']} at {fields['serialization_field_path']})",
              file=sys.stderr, flush=True)


def _process_bound_path(root) -> str:
    """Breadth-first path to the first process-bound object (socket, lock, open file, child
    process) in a payload that failed to pickle; bounded so a failure report stays cheap."""
    import socket
    import subprocess
    import _thread
    bound = (socket.socket, io.IOBase, subprocess.Popen, _thread.LockType, _thread.RLock)
    seen: set[int] = set()
    queue = [(root, "payload")]
    while queue and len(seen) < _PATH_NODE_LIMIT:
        obj, where = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, bound):
            return f"{where} ({type(obj).__name__})"
        if isinstance(obj, (str, bytes, int, float, type, types.ModuleType)):
            continue
        if isinstance(obj, dict):
            queue.extend((value, f"{where}[{key!r}]"[:300]) for key, value in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            queue.extend((value, f"{where}[{index}]") for index, value in enumerate(obj))
        else:
            fields = getattr(obj, "__dict__", None)
            if isinstance(fields, dict):
                queue.extend((value, f"{where}<{type(obj).__qualname__}>.{name}")
                             for name, value in fields.items())
    return "not found"


def _failure_reason(error: BaseException) -> str:
    """Return an allowlisted writer code without exposing exception-controlled text."""
    try:
        if (type(error) is ObjectCaptureError and type(error.reason) is str
                and error.reason in _WRITER_FAILURE_REASONS):
            return error.reason
        if type(error) is _StateFileSecretError:
            return "checkpoint refused: registered secret in state"
        if type(error) is CheckpointError:
            args = BaseException.args.__get__(error, BaseException)
            if type(args) is tuple and len(args) == 1 and type(args[0]) is str:
                if args[0] == _SECRET_CHECKPOINT_ERROR:
                    return "checkpoint refused: registered secret in checkpoint"
                if args[0] == "STATEFRAME_HOME_IDENTITY_CHANGED":
                    return "STATEFRAME_HOME_IDENTITY_CHANGED"
            return "checkpoint refused"
        if isinstance(error, OSError):
            try:
                number = object.__getattribute__(error, "errno")
            except BaseException:
                return "checkpoint write failed: os error"
            name = _OS_ERROR_NAMES.get(number) if type(number) is int else None
            return f"checkpoint write failed: {name or 'os error'}"
        if isinstance(error, MemoryError):
            return "checkpoint write failed: memory error"
        if type(error) is pickle.PicklingError:
            return "checkpoint serialization failed"
    except BaseException:
        pass
    return "unexpected writer error"


def _writer_reason(reason: str, writer_error_type: object) -> str:
    """Enrich a generic warning without changing durable reason codes."""
    if (reason == "unexpected writer error" and type(writer_error_type) is str
            and writer_error_type in _WRITER_ERROR_NAMES):
        return f"{reason}: {writer_error_type}"
    return reason


def _capture_stage(error: BaseException) -> dict[str, str]:
    if type(error) is ObjectCaptureError:
        stage = getattr(error, "stage", None)
        if type(stage) is str and stage in _OBJECT_CAPTURE_STAGES:
            return {"failure_stage": stage}
    return {}


class _StateWriter:
    """Compress pickle writes while retaining digests and incremental secret state."""

    __slots__ = ("stream", "digest", "count", "tail", "compressor",
                 "stored_digest", "stored_count")

    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.count = 0
        self.tail = None
        self.compressor = zlib.compressobj(level=1)
        self.stored_digest = hashlib.sha256()
        self.stored_count = 0

    def _write_stored(self, chunk: bytes) -> None:
        if not chunk:
            return
        written = self.stream.write(chunk)
        if written != len(chunk):
            raise OSError("short checkpoint state write")
        self.stored_digest.update(chunk)
        self.stored_count += written

    def write(self, chunk: bytes) -> int:
        if len(chunk) > MAX_STATE_BYTES - self.count:
            raise CheckpointError(f"checkpoint raw state exceeds the {MAX_STATE_BYTES}-byte limit")
        self.tail = _check_registered_secrets(chunk, self.tail, final=False)
        self.digest.update(chunk)
        self.count += len(chunk)
        self._write_stored(self.compressor.compress(chunk))
        return len(chunk)

    def finish(self) -> None:
        _check_registered_secrets(b"", self.tail)
        self._write_stored(self.compressor.flush())


def _locate(value, ctx, path, depth, out, seen):
    """Depth-first search for the smallest unpicklable pieces under ``value``."""
    if len(out) >= 60 or depth > 14 or id(value) in seen:
        return
    seen.add(id(value))
    try:
        dumps(value, ctx)
        return
    except Exception as error:  # noqa: BLE001 - diagnostic search
        reason = f"{type(error).__name__}: {str(error)[:160]}"
    children: list[tuple[str, object]] = []
    if isinstance(value, dict):
        children = [(f"[{k!r}]"[:80], v) for k, v in list(value.items())]
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = [(f"[{i}]", v) for i, v in enumerate(list(value))]
    else:
        state = getattr(value, "__dict__", None)
        if isinstance(state, dict):
            children = [(f".{k}", v) for k, v in list(state.items())]
        for slot in getattr(type(value), "__slots__", ()) or ():
            if isinstance(slot, str) and hasattr(value, slot):
                children.append((f".{slot}", getattr(value, slot)))
        if isinstance(value, types.FunctionType) and value.__closure__:
            children += [(f".<cell {n}>", c.cell_contents) for n, c in
                         zip(value.__code__.co_freevars, value.__closure__) if _cell_full(c)]
    before = len(out)
    for label, child in children:
        _locate(child, ctx, path + label, depth + 1, out, seen)
    if len(out) == before:
        out.append({"path": path, "type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "reason": reason})


def _cell_full(cell) -> bool:
    try:
        cell.cell_contents
    except ValueError:
        return False
    return True


def probe(frame, step: int, out_dir: Path) -> dict:
    """Try to pickle every root; record sizes, time and the unpicklable leaves."""
    if __package__:
        from .kernel_agent_tools import service as agent_tools_service
    else:
        from kernel_agent_tools import service as agent_tools_service
    host_tools = agent_tools_service()
    if host_tools is not None and any(job.state == "running" for job in host_tools.jobs.jobs.values()):
        raise CheckpointError("live background jobs are checkpoint-ineligible; resume uses observation replay")
    started = time.perf_counter()
    payload, ctx = collect(frame)
    collected = time.perf_counter() - started
    report = {"schema": "stateframe-checkpoint-probe/1", "step": step, "collect_s": round(collected, 3),
              "roots": {}, "failures": []}
    groups = {"frame": payload["frame"], "patched": payload["patched"], "context": payload["context"],
              "classes": payload["classes"]}
    groups.update({f"module:{k}": v for k, v in payload["modules"].items()})
    for group, value in groups.items():
        t0 = time.perf_counter()
        try:
            size = len(dumps(value, ctx))
            report["roots"][group] = {"bytes": size, "s": round(time.perf_counter() - t0, 3)}
        except Exception as error:  # noqa: BLE001 - diagnostic
            report["roots"][group] = {"error": f"{type(error).__name__}: {str(error)[:200]}"}
            leaves: list[dict] = []
            if isinstance(value, dict):
                for key, child in list(value.items()):
                    _locate(child, ctx, f"{group}.{key}", 0, leaves, set())
            else:
                _locate(value, ctx, group, 0, leaves, set())
            report["failures"].extend(leaves)
    t0 = time.perf_counter()
    try:
        whole = dumps(payload, ctx)
        report["whole"] = {"bytes": len(whole), "s": round(time.perf_counter() - t0, 3)}
        # Load it back in this process (objects are discarded): catches load-side failures such
        # as constructors that the default pickle path cannot call, without a restart.
        t1 = time.perf_counter()
        cells, funcs = _frame_maps(frame, _STATE.get("setup"))
        _STATE["probe_load"] = True
        try:
            loaded = _Unpickler(io.BytesIO(whole), {"framecells": cells, "framefuncs": funcs}).load()
            report["load"] = {"s": round(time.perf_counter() - t1, 3)}
            # Protocol 3 has no length-prefixed frames, so the first differing byte is the first
            # differing object, not a frame header.
            whole = dumps(payload, ctx, 3)
            again = dumps(loaded, ctx, 3)
            report["redump_equal"] = again == whole
            if again != whole:
                offset = next((i for i, (x, y) in enumerate(zip(whole, again)) if x != y),
                              min(len(whole), len(again)))
                report["redump_diff"] = {"offset": offset, "bytes": [len(whole), len(again)],
                                         "saved": whole[max(0, offset - 160):offset + 80].decode("latin-1"),
                                         "again": again[max(0, offset - 160):offset + 80].decode("latin-1")}
        except Exception as error:  # noqa: BLE001
            import traceback
            report["load"] = {"error": f"{type(error).__name__}: {str(error)[:300]}",
                              "where": traceback.format_exc()[-1500:]}
        finally:
            _STATE["probe_load"] = False
    except Exception as error:  # noqa: BLE001
        report["whole"] = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"probe_{step:04d}.json").write_text(json.dumps(report, indent=1, default=str))
    return report


# --------------------------------------------------------------------------- save

_STATE = {"saved": 0, "failed": 0, "restored_step": None, "disabled": None, "last_sends": None,
          "child": None, "child_tmp": None, "child_journal_offset": None,
          "child_journal_file_id": None, "child_journal_missing": False,
          "consecutive_failed": 0, "low_disk": False}
# Warn once after repeated failures; save attempts are gated by new model sends.
FAILURE_WARNING_THRESHOLD = 3


def _every() -> int:
    raw = os.environ.get(EVERY_ENV, "").strip()
    if not raw:
        return 1  # empty = unset
    try:
        return max(1, int(raw))
    except ValueError:
        _warn_once("every", f"{EVERY_ENV}={raw[:40]!r} is not a whole number; saving at every step")
        return 1


def _warn_once(key: str, text: str) -> None:
    """A setting problem the user must see (the run continues): one stderr line per run."""
    warned = _STATE.setdefault("warned", set())
    if key not in warned:
        warned.add(key)
        print(f"checkpoint: WARNING {text}", file=sys.stderr, flush=True)


def _keep() -> int:
    try:
        return max(1, int(os.environ.get(KEEP_ENV, "2")))
    except ValueError:
        return 2


_CLONEFILE = None


def _clonefile(source: Path, target: Path) -> bool:
    """macOS clonefile(2): a whole directory tree cloned copy-on-write in one call (~40 ms for
    2.2k files, where a per-file copy or link costs ~1 s on a busy machine)."""
    global _CLONEFILE
    if sys.platform != "darwin":
        return False
    if _CLONEFILE is None:
        import ctypes
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            function = libc.clonefile
            function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
            function.restype = ctypes.c_int
            _CLONEFILE = function
        except (OSError, AttributeError):
            _CLONEFILE = False
    if not _CLONEFILE:
        return False
    CLONE_NOFOLLOW = 0x0001
    return _CLONEFILE(os.fsencode(str(source)), os.fsencode(str(target)), CLONE_NOFOLLOW) == 0


def _clone(source: Path, target: Path) -> None:
    """APFS clone (copy-on-write) when available, plain copy otherwise."""
    if _clonefile(source, target):
        return
    if source.is_dir() and not source.is_symlink():
        shutil.copytree(source, target, symlinks=True)
    else:
        shutil.copy2(source, target, follow_symlinks=False)


_BIG_FILE = 4 << 20

# Journal blob stores (worker/measurement-runs/<run>/blobs) are write-once and content-addressed
# (kernel_measurement._write_journal_blob: temp file + os.link, never rewritten) and grow by
# thousands of files per run. Re-cloning them every save made the file phase grow with run length
# (0.41 s at 101 sends). They go to one pool per run instead: each save copies only the NEW
# blobs into the private checkpoints/blob-pool and records the name list; a restore copies them back.
POOL_DIR = "blob-pool"
MANIFEST_FILE = "blobs.json"
# Pool store -> {name: (size, digest)}. Pooled files are write-once; their identity is
# computed once per process and signed in each checkpoint manifest.
_POOL_KNOWN: dict[str, dict[str, tuple[int, str]]] = {}


def _pool_identity(path: Path) -> tuple[int, str]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise CheckpointError(f"pooled file {path.name} is not regular")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return info.st_size, digest.hexdigest()


def _capture_snapshot_file(
        source: Path, target: Path, *, known: tuple[int, str] | None = None,
        new_pool_files: list[Path] | None = None, prefer_clone: bool = False,
) -> tuple[tuple[int, str], bool]:
    """Capture and scan a private file before atomically publishing it to a snapshot."""
    _check_registered_secrets(os.fsencode(source), failure_type=_StateFileSecretError)
    try:
        source_info = os.lstat(source)
    except OSError as error:
        raise CheckpointError("run file changed during checkpoint capture") from error
    if not stat.S_ISREG(source_info.st_mode):
        raise CheckpointError("snapshot source is not a regular run file")

    protected_identity = _STATE.get("protected_home_identity")
    if isinstance(protected_identity, dict) and (
            source_info.st_dev, source_info.st_ino) == (
                protected_identity.get("key_st_dev"), protected_identity.get("key_st_ino")):
        raise _StateFileSecretError(_SECRET_CHECKPOINT_ERROR)

    generation = _SECRET_GENERATION
    target_info = None
    try:
        target_info = os.lstat(target)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise CheckpointError("snapshot file could not be inspected") from error
    if new_pool_files is not None and known is not None and target_info is not None:
        source_cached = (_snapshot_file_cache_key(source_info, generation, known[1])
                         in _VERIFIED_POOL_IDENTITIES)
        target_cached = (stat.S_ISREG(target_info.st_mode)
                         and _snapshot_file_cache_key(target_info, generation, known[1])
                         in _VERIFIED_POOL_IDENTITIES)
        if (known[0] == source_info.st_size and source_cached and target_cached
                and target_info.st_size == known[0]
                and (target_info.st_dev, target_info.st_ino)
                != (source_info.st_dev, source_info.st_ino)):
            return known, False

    staged_target = None
    artifact_root = _SAVE.get("artifact_root")
    pool_root = _SAVE.get("pool_root")
    if (new_pool_files is not None and isinstance(artifact_root, Path)
            and isinstance(pool_root, Path)):
        try:
            staged_target = artifact_root / POOL_DIR / target.relative_to(pool_root)
        except ValueError as error:
            raise CheckpointError("pooled file is outside its checkpoint store") from error
        _private_dir(staged_target.parent)
    else:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    capture_target = staged_target or target
    source_fd = None
    temporary_fd = None
    temporary_path = None

    def stable_signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    try:
        source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                            | getattr(os, "O_CLOEXEC", 0))
        opened = os.fstat(source_fd)
        if (not stat.S_ISREG(opened.st_mode)
                or stable_signature(opened) != stable_signature(source_info)):
            raise CheckpointError("run file changed during checkpoint capture")
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=".capture-", dir=capture_target.parent)
        temporary_path = Path(temporary_name)
        os.close(temporary_fd)
        temporary_fd = None
        temporary_path.unlink()

        cloned = (sys.platform == "darwin" and (prefer_clone or opened.st_size >= _BIG_FILE)
                  and _clonefile(source, temporary_path))
        copied = 0
        if not cloned:
            if temporary_path.exists():
                temporary_path.unlink()
            temporary_fd = os.open(
                temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(temporary_fd, "wb") as output:
                temporary_fd = None
                while True:
                    block = os.read(source_fd, 1 << 20)
                    if not block:
                        break
                    output.write(block)
                    copied += len(block)
                os.fchmod(output.fileno(), stat.S_IMODE(opened.st_mode))
                output.flush()
                os.fsync(output.fileno())
        else:
            copied = opened.st_size
            os.chmod(temporary_path, stat.S_IMODE(opened.st_mode), follow_symlinks=False)
        after_copy = os.fstat(source_fd)
        path_after_copy = os.lstat(source)
        if (copied != opened.st_size
                or stable_signature(after_copy) != stable_signature(opened)
                or stable_signature(path_after_copy) != stable_signature(opened)):
            raise CheckpointError("run file changed during checkpoint capture")
        os.utime(temporary_path, ns=(opened.st_atime_ns, opened.st_mtime_ns),
                 follow_symlinks=False)
        captured_before = os.lstat(temporary_path)

        while True:
            scan_generation = _SECRET_GENERATION
            identity = _check_snapshot_file(
                temporary_path, digest_hint=known[1] if known is not None else None,
                cache_verified=False)
            if scan_generation == _SECRET_GENERATION:
                break
        captured_after = os.lstat(temporary_path)
        if (captured_before.st_dev, captured_before.st_ino, captured_before.st_size,
                captured_before.st_mtime_ns, captured_before.st_ctime_ns) != (
                captured_after.st_dev, captured_after.st_ino, captured_after.st_size,
                captured_after.st_mtime_ns, captured_after.st_ctime_ns):
            raise CheckpointError("captured file changed before checkpoint publication")
        if known is not None and known != identity:
            raise CheckpointError("pooled file identity changed during checkpoint capture")

        current_target = None
        try:
            current_target = os.lstat(target)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise CheckpointError("snapshot file could not be inspected") from error
        if new_pool_files is not None and current_target is not None and stat.S_ISREG(
                current_target.st_mode):
            expected_digest = known[1] if known is not None else identity[1]
            if (_snapshot_file_cache_key(current_target, _SECRET_GENERATION, expected_digest)
                    not in _VERIFIED_POOL_IDENTITIES):
                target_identity = _check_snapshot_file(target, digest_hint=expected_digest)
                if known is not None and target_identity != known:
                    raise CheckpointError("pooled file identity changed during checkpoint capture")
        target_shares_source = (
            current_target is not None
            and (current_target.st_dev, current_target.st_ino)
            == (source_info.st_dev, source_info.st_ino)
        )
        if staged_target is not None and target_shares_source:
            # A transaction cannot replace a shared artifact used by older checkpoints.
            # Refuse the unsafe alias rather than let publication reuse its matching bytes.
            raise CheckpointError("checkpoint artifact identity conflict")
        replace_target = (new_pool_files is None or current_target is None
                          or not stat.S_ISREG(current_target.st_mode) or target_shares_source)
        if replace_target:
            os.replace(temporary_path, capture_target)
            temporary_path = None
            if new_pool_files is not None:
                # Pooled files are write-once: read-only from the moment they are published.
                published = os.lstat(capture_target)
                os.chmod(capture_target, stat.S_IMODE(published.st_mode) & ~0o222)
                new_pool_files.append(capture_target)
        else:
            temporary_path.unlink()
            temporary_path = None

        if new_pool_files is not None and scan_generation == _SECRET_GENERATION:
            _VERIFIED_POOL_IDENTITIES.add(
                _snapshot_file_cache_key(source_info, scan_generation, identity[1]))
            verified_target = capture_target if replace_target else target
            target_after_publish = os.lstat(verified_target)
            _VERIFIED_POOL_IDENTITIES.add(
                _snapshot_file_cache_key(target_after_publish, scan_generation, identity[1]))
        return identity, cloned
    except OSError as error:
        raise CheckpointError("run file could not be captured for checkpoint") from error
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _real_path(path: Path) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _snapshot_home_exclusions(run_dir: Path) -> tuple[Path | None, Path | None, set[Path]]:
    """Find StateFrame-home paths to omit from a run-file snapshot, resolving symlinks first."""
    try:
        if __package__:
            from .kernel_request_auth_identity import _installation_home
        else:
            from kernel_request_auth_identity import _installation_home

        identity = _STATE.get("protected_home_identity")
        home = (_real_path(Path(str(identity["realpath"]))) if _valid_home_identity(identity)
                else _real_path(_installation_home()))
    except (OSError, ValueError):
        return None, None, set()

    run_root = _real_path(run_dir)
    home_subtree = home if _path_within(home, run_root) else None
    excluded: set[Path] = set()
    if home_subtree is not None:
        excluded.add(home_subtree)

    for name in sorted(os.listdir(run_dir)):
        if name in SKIP_RUN_ENTRIES:
            continue
        source = run_dir / name
        if source.is_symlink():
            if _path_within(_real_path(source), home):
                excluded.add(Path(os.path.abspath(source)))
            continue
        if not source.is_dir():
            continue
        pending = [source]
        while pending:
            current = pending.pop()
            if home_subtree is not None and _path_within(_real_path(current), home_subtree):
                excluded.add(Path(os.path.abspath(current)))
                continue
            try:
                with os.scandir(current) as scan:
                    children = list(scan)
            except OSError as error:
                raise CheckpointError(
                    "could not inspect run files for StateFrame home exclusions"
                ) from error
            for entry in children:
                path = Path(entry.path)
                try:
                    is_symlink = entry.is_symlink()
                    is_directory = not is_symlink and entry.is_dir(follow_symlinks=False)
                except OSError as error:
                    raise CheckpointError(
                        "could not inspect run files for StateFrame home exclusions"
                    ) from error
                if is_symlink:
                    if _path_within(_real_path(path), home):
                        excluded.add(Path(os.path.abspath(path)))
                elif is_directory:
                    pending.append(path)
    return home, home_subtree, excluded


def _snapshot_path_excluded(path: Path, home: Path | None, home_subtree: Path | None,
                            excluded: set[Path], is_symlink: bool | None = None) -> bool:
    """True when ``path`` is inside the home boundary; ``is_symlink`` lets a scandir caller skip the lstat."""
    _check_registered_secrets(os.fsencode(path), failure_type=_StateFileSecretError)
    linked = path.is_symlink() if is_symlink is None else is_symlink
    if linked:
        _check_registered_secrets(os.fsencode(os.readlink(path)), failure_type=_StateFileSecretError)
    absolute = Path(os.path.abspath(path))
    if absolute in excluded:
        return True
    if home_subtree is not None and _path_within(_real_path(path), home_subtree):
        return True
    return linked and home is not None and _path_within(_real_path(path), home)


def _capture_snapshot_symlink(source: Path, target: Path, home: Path | None) -> None:
    """Validate the captured link text, then publish those same bytes without rereading the source."""
    link = os.readlink(source)
    _check_registered_secrets(os.fsencode(source), failure_type=_StateFileSecretError)
    _check_registered_secrets(os.fsencode(link), failure_type=_StateFileSecretError)
    if home is not None and _path_within(_real_path(source.parent / link), home):
        raise CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED")
    os.symlink(link, target)


def _tree_has_home_exclusions(source: Path, home_subtree: Path | None,
                              excluded: set[Path]) -> bool:
    source_absolute = Path(os.path.abspath(source))
    source_real = _real_path(source)
    if home_subtree is not None and (
            _path_within(home_subtree, source_real) or _path_within(source_real, home_subtree)):
        return True
    return any(_path_within(path, source_absolute) or _path_within(source_absolute, path)
               for path in excluded)


# Other write-once directories, one or more new files per model send: the product's wire/, requests/
# and checks/ records (written once each) and the engine's candidate captures (created with mode "x").
POOLED_DIRS = frozenset({"wire", "requests", "checks", "worker/candidates"})


def _is_blob_store(relative: str) -> bool:
    parts = relative.split("/")
    return len(parts) == 4 and parts[0] == "worker" and parts[1] == "measurement-runs" and parts[3] == "blobs"


def _is_pooled(relative: str) -> bool:
    return relative in POOLED_DIRS or _is_blob_store(relative)


def _snapshot_appended_file(source: Path, target: Path, relative: str, root: Path,
                            previous: list, file_chunks: dict, seq: int) -> bool:
    """Store an audit journal in the existing chunk cache, writing only its new suffix."""
    if (not relative.startswith("worker/measurement-runs/")
            or relative.count("/") != 3 or not relative.endswith("/events.jsonl")
            or not source.is_file() or source.is_symlink()):
        return False
    size = source.stat().st_size
    rows = [list(row) for row in previous]
    covered = rows[-1][2] if rows else 0
    if covered > size:
        rows, covered = [], 0
    if rows:
        last = rows[-1]
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            stream.seek(last[1])
            for block in iter(lambda: stream.read(min(1 << 20, last[2] - stream.tell())), b""):
                digest.update(block)
        if digest.hexdigest() != last[3]:
            rows, covered = [], 0
    if size > covered:
        artifact_root = _SAVE.get("artifact_root", root)
        folder = Path(artifact_root) / CACHE_DIR
        _private_dir(folder)
        name = f"file-{_stable_key(relative)}-{seq:08d}-{os.getpid()}{CHUNK_SUFFIX}"
        temporary = folder / f".{name}.tmp"
        raw_digest, stored_digest = hashlib.sha256(), hashlib.sha256()
        compressor = zlib.compressobj(1)
        secret_tail = None
        temporary_created = False
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            temporary_created = True
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as source_stream:
                if _SECRETS:
                    # Rebuild decoder state from the complete committed prefix.
                    # No finite raw overlap covers arbitrarily expanded escapes.
                    prefix_remaining = covered
                    while prefix_remaining:
                        prefix = source_stream.read(min(1 << 20, prefix_remaining))
                        if not prefix:
                            raise CheckpointError(f"audit journal {relative} shrank during snapshot")
                        prefix_remaining -= len(prefix)
                        secret_tail = _check_registered_secrets(prefix, secret_tail, final=False)
                else:
                    source_stream.seek(covered)
                remaining = size - covered
                while remaining:
                    block = source_stream.read(min(1 << 20, remaining))
                    if not block:
                        raise CheckpointError(f"audit journal {relative} shrank during snapshot")
                    remaining -= len(block)
                    secret_tail = _check_registered_secrets(
                        block, secret_tail, final=False)
                    raw_digest.update(block)
                    encoded = compressor.compress(block)
                    output.write(encoded)
                    stored_digest.update(encoded)
                _check_registered_secrets(b"", secret_tail)
                encoded = compressor.flush()
                output.write(encoded)
                stored_digest.update(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, folder / name)
            temporary_created = False
        except BaseException:
            if temporary_created:
                try:
                    temporary.unlink()
                except OSError:
                    pass
            raise
        rows.append([name, covered, size, raw_digest.hexdigest(), stored_digest.hexdigest()])
    target.touch()
    shutil.copystat(source, target)
    file_chunks[relative] = rows
    return True


def _snapshot_tree(source: Path, target: Path, relative: str, pool: Path, manifests: dict,
                   new_pool_files: list[Path], file_chunks: dict | None = None,
                   previous_files: dict | None = None, seq: int = 0, *,
                   home: Path | None = None, home_subtree: Path | None = None,
                   excluded: set[Path] | None = None) -> None:
    """Clone ``source`` into ``target`` with as few clonefile calls as possible, pooling write-once dirs."""
    excluded = excluded or set()
    if _snapshot_path_excluded(source, home, home_subtree, excluded):
        return
    if source.is_dir() and not source.is_symlink() and _tree_has_home_exclusions(
            source, home_subtree, excluded):
        target.mkdir()
        with os.scandir(source) as scan:
            entries = sorted(scan, key=lambda entry: entry.name)
        for entry in entries:
            child = Path(entry.path)
            if not _snapshot_path_excluded(child, home, home_subtree, excluded):
                _snapshot_tree(child, target / entry.name, f"{relative}/{entry.name}", pool,
                               manifests, new_pool_files, file_chunks, previous_files, seq,
                               home=home, home_subtree=home_subtree, excluded=excluded)
        shutil.copystat(source, target, follow_symlinks=False)
        return
    entries = list(os.scandir(source)) if _is_pooled(relative) and source.is_dir() else None
    if entries is not None and all(entry.is_file(follow_symlinks=False) for entry in entries):
        names = sorted(entry.name for entry in entries)
        target.mkdir()  # present (empty) in the copy; restore copies the pooled files back in
        store = pool / relative.replace("/", "__")
        known = _POOL_KNOWN.get(str(store))
        if known is None:  # first save of this process: adopt what an earlier process pooled
            store.mkdir(mode=0o700, parents=True, exist_ok=True)
            known = _POOL_KNOWN.setdefault(str(store), {
                name: _pool_identity(store / name) for name in os.listdir(store)})
        live_infos = {entry.name: entry.stat(follow_symlinks=False) for entry in entries}
        for name in names:
            cached = known.get(name)
            identity, _cloned = _capture_snapshot_file(
                source / name, store / name, known=cached, new_pool_files=new_pool_files)
            if cached is not None and cached != identity:
                raise CheckpointError("pooled file identity changed during checkpoint capture")
            known[name] = identity
            # A pool file is shared by every checkpoint holding these bytes, so its own mtime is the
            # first capture's; the live mtime of this checkpoint is recorded for restore.
            _note_mtime(f"{relative}/{name}", live_infos[name])
        manifests[relative] = [[name, known[name][0]] for name in names]
        return
    if source.is_dir() and not source.is_symlink():
        target.mkdir()
        with os.scandir(source) as scan:
            children = sorted(scan, key=lambda entry: entry.name)
        for entry in children:
            child = Path(entry.path)
            if _snapshot_path_excluded(child, home, home_subtree, excluded):
                continue
            _snapshot_tree(child, target / entry.name, f"{relative}/{entry.name}", pool,
                           manifests, new_pool_files, file_chunks, previous_files, seq,
                           home=home, home_subtree=home_subtree, excluded=excluded)
        shutil.copystat(source, target, follow_symlinks=False)
        return
    if file_chunks is not None and _snapshot_appended_file(
            source, target, relative, pool.parent,
            (previous_files or {}).get(relative, ()), file_chunks, seq):
        return
    if source.is_file() and not source.is_symlink():
        _capture_run_file(source, target, relative)
        return
    if source.is_symlink():
        _capture_snapshot_symlink(source, target, home)
        return
    if not _clonefile(source, target):
        _clone(source, target)


# Per-run content-addressed store of run files. A file is stored once per (bytes, mode): the object
# is named by its sha256 and mode, made read-only, and every checkpoint holding those bytes
# hard-links it. The live run dir never links into the store; restore always makes private copies.
# mtime is not part of the key (a touched file with the same bytes reuses the object), so a checkpoint
# copy keeps the mtime of the first capture of its bytes. Objects no snapshot links any more
# (nlink == 1) are collected when a checkpoint is published.
OBJECT_DIR = "env-objects"
# os.link errnos meaning "this filesystem cannot link here": the checkpoint takes a private copy.
LINK_FALLBACK_ERRNOS = frozenset({errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP})
# Live stat tuple (dev, ino, size, mtime_ns, ctime_ns, secret generation) -> sha256 scanned under it.
# ctime moves on every write, so a changed file misses. Cleared when the run changes.
_DIGEST_MEMO: dict[tuple[int, int, int, int, int, int], str] = {}
# Pool objects are immutable; verify an existing inode once in each process before reusing it.
_VERIFIED_OBJECTS: set[tuple[int, str, int, int, int, int, str]] = set()
# Timestamps come from a coarse clock, so a rewrite shortly after a scan can keep the same ctime. As
# git does for its index, a digest is remembered only when the file's mtime and ctime both predate
# the scan by this window; a recently touched file is always hashed again.
_RACY_WINDOW_NS = 2_000_000_000


def _note_new_bytes(count: int) -> None:
    """Add bytes this save newly wrote to its ledger (meta "bytes_new"); a no-op outside a save."""
    ledger = _SAVE.get("bytes_new")
    if isinstance(ledger, list) and count:
        ledger[0] += count


def _note_mode(key: str, info: os.stat_result) -> None:
    """Record the live mode and mtime_ns of a captured file so restore can give the copy both back.

    The mtime matters because stored objects are shared by content: a stored copy carries the mtime
    of the first capture of those bytes, not the live file's mtime at this checkpoint.
    """
    modes = _SAVE.get("copy_modes")
    if isinstance(modes, dict):
        modes[key] = stat.S_IMODE(info.st_mode)
    _note_mtime(key, info)


def _note_mtime(key: str, info: os.stat_result) -> None:
    """Record a live file's mtime_ns under its checkpoint key (meta "file_mtime_ns"); no-op outside a save."""
    mtimes = _SAVE.get("copy_mtimes")
    if isinstance(mtimes, dict):
        mtimes[key] = info.st_mtime_ns


def _live_digest(path: Path, info: os.stat_result) -> tuple[int, str]:
    """(size, sha256) of a live regular file after the registered-secret scan.

    A digest is reused only for the exact stat tuple it was computed from, under the current secret
    generation, so an unchanged file costs no read and no lstat.
    """
    _check_registered_secrets(os.fsencode(path), failure_type=_StateFileSecretError)
    generation = _SECRET_GENERATION
    key = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, generation)
    cached = _DIGEST_MEMO.get(key)
    if cached is not None:
        return info.st_size, cached
    started = time.time_ns()
    size, digest = _check_snapshot_file(path)
    newest_change = max(info.st_mtime_ns, info.st_ctime_ns)
    if generation == _SECRET_GENERATION and newest_change < started - _RACY_WINDOW_NS:
        _DIGEST_MEMO[key] = digest
    return size, digest


def _store_object(path: str, info: os.stat_result, digest: str, objects: Path) -> tuple[Path, int]:
    """Make the object for (bytes ``digest``, ``info``'s mode) exist; return it and the bytes written.

    An existing object is reused without reading the source. A new one is copied to a temporary while
    its bytes are hashed: a source that changed since it was scanned raises instead of storing bytes
    the scan never saw. The copy is fsynced, made read-only, then renamed into place.
    """
    objects.mkdir(mode=0o700, exist_ok=True)
    mode = stat.S_IMODE(info.st_mode)
    final = objects / f"{digest}-{mode:o}"
    folder = os.open(objects, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            existing = os.stat(final.name, dir_fd=folder, follow_symlinks=False)
            if stat.S_ISREG(existing.st_mode) and existing.st_size == info.st_size:
                identity = (os.getpid(), os.fspath(final), existing.st_dev, existing.st_ino,
                            existing.st_size, existing.st_mtime_ns, digest)
                if (stat.S_IMODE(existing.st_mode) == (mode & ~0o222)
                        and (identity in _VERIFIED_OBJECTS
                             or _file_sha256(final.name, dir_fd=folder) == digest)):
                    _VERIFIED_OBJECTS.add(identity)
                    return final, 0
                aside_name = f"quarantine-{time.time_ns()}-{final.name}"
                os.rename(final.name, aside_name, src_dir_fd=folder, dst_dir_fd=folder)
        except FileNotFoundError:
            pass
    finally:
        os.close(folder)
    temporary_name: str | None = None
    try:
        with open(path, "rb") as reader:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{os.getpid()}-", suffix=".tmp", dir=objects)
            hasher = hashlib.sha256()
            written = 0
            with os.fdopen(descriptor, "wb") as writer:
                for block in iter(lambda: reader.read(1 << 20), b""):
                    hasher.update(block)
                    writer.write(block)
                    written += len(block)
                writer.flush()
                os.fsync(writer.fileno())
        if hasher.hexdigest() != digest or written != info.st_size:
            raise CheckpointError("run file changed during checkpoint capture")
        shutil.copystat(path, temporary_name, follow_symlinks=False)
        os.chmod(temporary_name, mode & ~0o222)
        os.replace(temporary_name, final)
    except BaseException:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
        raise
    return final, written


def _store_object_batch(batch, objects: Path, minimum: int = 8) -> dict[str, tuple[Path, int]]:
    if len(batch) < minimum:
        return {item[3]: _store_object(*item[:3], objects) for item in batch}
    stopped = threading.Event()

    def store(item):
        if stopped.is_set():
            return None
        try:
            return _store_object(*item[:3], objects)
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt) or getattr(error, "boundary_recovery_stop", False):
                stopped.set()
            raise

    with ThreadPoolExecutor(max_workers=16) as pool:
        try:
            futures = [pool.submit(store, item) for item in batch]
            return {item[3]: future.result() for item, future in zip(batch, futures)}
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt) or getattr(error, "boundary_recovery_stop", False):
                stopped.set()
            raise


def _link_or_copy(stored: Path, destination: str | Path, live: Path, relpath: str,
                  digests: dict[str, str] | None, counts: dict[str, int]) -> str:
    """Hard-link ``stored`` to ``destination``; a filesystem refusing the link gets a private copy.

    Returns "linked", "copied" or "cloned". A fallback is counted in ``counts["link_fallback"]`` and
    never fails the checkpoint.
    """
    try:
        os.link(stored, destination)
        return "linked"
    except OSError as error:
        if error.errno not in LINK_FALLBACK_ERRNOS:
            raise
    counts["link_fallback"] = counts.get("link_fallback", 0) + 1
    cloned = _capture_recorded(live, Path(destination), relpath, digests)
    if not cloned:
        _note_new_bytes(os.lstat(live).st_size)
    return "cloned" if cloned else "copied"


def _file_sha256(path: str | os.PathLike[str], *, dir_fd: int | None = None) -> str | None:
    """SHA-256 of a regular file, streamed in 1 MiB blocks; None when it is not readable as one."""
    descriptor = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = (os.open(path, flags) if dir_fd is None
                      else os.open(path, flags, dir_fd=dir_fd))
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)
    except OSError:
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _capture_recorded(source: Path, destination: Path, relpath: str,
                      digests: dict[str, str] | None, **kwargs: bool) -> bool:
    """``_capture_snapshot_file`` that also records the sha256 of the bytes it published."""
    identity, cloned = _capture_snapshot_file(source, destination, **kwargs)
    if digests is not None and identity[1]:
        digests[relpath] = identity[1]
    return cloned


def _copy_tree(source: Path, target: Path, previous: Path | None,
               durable_inodes: set[tuple[int, int]], *, seed: Path | None = None,
               objects: Path | None = None,
               home: Path | None = None, home_subtree: Path | None = None,
               excluded: set[Path] | None = None) -> dict[str, int]:
    """Copy ``source`` to ``target``, sharing every regular file whose bytes already exist in the set.

    Each regular file is hashed once per stat tuple (``_live_digest``). Resolution, in order: (1) the
    previous snapshot's copy, hard-linked, when its recorded sha256 equals the live digest; (2) the
    content-addressed ``objects`` store when the save has one (stored once, ever); (3) a private copy.
    ``seed`` is accepted for the call contract only: content matching already shares seed-identical
    bytes, so no seed-specific copy path remains.
    """
    if objects is None:
        objects = _SAVE.get("snapshot_objects")
    if home is None and home_subtree is None and excluded is None:
        # Keep the original copy-hook call contract while carrying the save's
        # boundary filters through wrappers that only forward the seed option.
        home, home_subtree, excluded = _SAVE.get("snapshot_exclusions", (None, None, None))
    excluded = excluded or set()
    # (digests written for this copy, digests the prior snapshot recorded); both optional.
    digests, prior_digests = _SAVE.get("copy_digests", (None, None))
    prefix = _SAVE.get("copy_prefix", "")
    darwin = sys.platform == "darwin"
    seed_root = None if seed is None else os.fspath(seed)
    counts = {"linked": 0, "copied": 0, "cloned": 0}
    src_root, dst_root = os.fspath(source), os.fspath(target)
    prev_root = None if previous is None else os.fspath(previous)
    pending = [""]
    batch, names = [], {}

    def account_object(stored, new_bytes, digest, destination, path, relpath):
        _note_new_bytes(new_bytes)
        status = _link_or_copy(stored, destination, Path(path), relpath, digests, counts)
        if status == "linked":
            stored_info = os.stat(stored)
            durable_inodes.add((stored_info.st_dev, stored_info.st_ino))
        if digests is not None:
            digests[relpath] = digest
        counts["copied" if new_bytes or status != "linked" else "linked"] += 1

    def flush_objects() -> None:
        if not batch:
            return
        stored = _store_object_batch(list(names.values()), objects)
        for path, info, digest, destination, relpath in batch:
            name = f"{digest}-{stat.S_IMODE(info.st_mode):o}"
            object_path, new_bytes = stored[name]
            stored[name] = (object_path, 0)
            account_object(object_path, new_bytes, digest, destination, path, relpath)
        batch.clear()
        names.clear()

    if home is not None or home_subtree is not None or excluded:
        if _snapshot_path_excluded(source, home, home_subtree, excluded):
            return counts
    else:
        _check_registered_secrets(os.fsencode(source), failure_type=_StateFileSecretError)
    while pending:
        rel = pending.pop()
        directory = os.path.join(src_root, rel) if rel else src_root
        out_dir = os.path.join(dst_root, rel) if rel else dst_root
        os.makedirs(out_dir, exist_ok=True)
        try:
            with os.scandir(directory) as scan:
                entries = list(scan)
        except OSError:
            continue
        old_dir = None if prev_root is None else (os.path.join(prev_root, rel) if rel else prev_root)
        for entry in entries:
            name = entry.name
            path = entry.path
            destination = os.path.join(out_dir, name)
            is_link = entry.is_symlink()  # scandir caches the type: no extra lstat per entry
            if _snapshot_path_excluded(Path(path), home, home_subtree, excluded, is_link):
                continue
            if is_link:
                _capture_snapshot_symlink(Path(path), Path(destination), home)
                continue
            if entry.is_dir(follow_symlinks=False):
                pending.append(os.path.join(rel, name) if rel else name)
                continue
            info = entry.stat(follow_symlinks=False)
            relpath = os.path.join(rel, name) if rel else name
            if not stat.S_ISREG(info.st_mode):
                _capture_recorded(Path(path), Path(destination), relpath, digests)  # refuses it
                counts["copied"] += 1
                continue
            _note_mode(os.path.join(prefix, relpath) if prefix else relpath, info)
            # A digest is needed only to compare with a prior copy or to name an object. A plain
            # private copy is scanned as it is written, which keeps the bytes it publishes scanned.
            size, digest = info.st_size, None
            if old_dir is not None or objects is not None:
                size, digest = _live_digest(Path(path), info)
            if old_dir is not None and digest is not None:
                prior_path = os.path.join(old_dir, name)
                try:
                    prior = os.lstat(prior_path)
                except OSError:
                    prior = None
                if (prior is not None and stat.S_ISREG(prior.st_mode)
                        and (prior.st_dev, prior.st_ino) != (info.st_dev, info.st_ino)
                        and prior.st_size == size):
                    recorded = prior_digests.get(relpath) if prior_digests is not None else None
                    prior_digest = recorded if type(recorded) is str else _file_sha256(prior_path)
                    if prior_digest == digest:
                        status = _link_or_copy(Path(prior_path), destination, Path(path),
                                               relpath, digests, counts)
                        if status == "linked":
                            durable_inodes.add((prior.st_dev, prior.st_ino))
                        if digests is not None:
                            digests[relpath] = digest
                        counts[status] += 1
                        continue
            if seed_root is not None and darwin:
                # Clone preference only: the live bytes are captured, never the seed's copy.
                try:
                    seed_info = os.stat(os.path.join(seed_root, relpath))
                except OSError:
                    seed_info = None
                if (seed_info is not None and stat.S_ISREG(seed_info.st_mode)
                        and seed_info.st_size == info.st_size
                        and seed_info.st_mtime_ns == info.st_mtime_ns):
                    cloned = _capture_recorded(Path(path), Path(destination), relpath, digests,
                                               prefer_clone=True)
                    _note_new_bytes(0 if cloned else info.st_size)
                    counts["cloned" if cloned else "copied"] += 1
                    continue
            if objects is not None:
                object_name = f"{digest}-{stat.S_IMODE(info.st_mode):o}"
                if os.path.lexists(objects / object_name):
                    stored, new_bytes = _store_object(path, info, digest, objects)
                    account_object(stored, new_bytes, digest, destination, path, relpath)
                else:
                    batch.append((path, info, digest, destination, relpath))
                    names.setdefault(object_name, (path, info, digest, object_name))
                    if len(batch) == 256:
                        flush_objects()
                continue
            cloned = _capture_private(Path(path), info, destination, relpath, digests)
            _note_new_bytes(0 if cloned else info.st_size)
            counts["cloned" if cloned else "copied"] += 1
    flush_objects()
    return counts


def _capture_run_file(source: Path, target: Path, relative: str) -> None:
    """Store one top-level or worker run file through the content-addressed store on the save path."""
    objects = _SAVE.get("snapshot_objects")
    if not isinstance(objects, Path):
        try:
            info = os.lstat(source)
        except OSError:
            info = None  # the capture below reports the missing or unreadable source
        if info is not None and stat.S_ISREG(info.st_mode):
            _note_mode(relative, info)
        _capture_snapshot_file(source, target)
        return
    try:
        info = os.lstat(source)
    except OSError as error:
        raise CheckpointError("run file changed during checkpoint capture") from error
    if not stat.S_ISREG(info.st_mode):
        raise CheckpointError("snapshot source is not a regular run file")
    _size, digest = _live_digest(source, info)
    _note_mode(relative, info)
    stored, new_bytes = _store_object(str(source), info, digest, objects)
    _note_new_bytes(new_bytes)
    counts = _SAVE.get("counts")
    if not isinstance(counts, dict):
        counts = {}
    status = _link_or_copy(stored, target, source, relative, None, counts)
    if status == "linked":
        durable = _SAVE.get("durable_inodes")
        stored_info = os.stat(stored)
        if isinstance(durable, set):
            durable.add((stored_info.st_dev, stored_info.st_ino))
    key = "copied" if new_bytes or status != "linked" else "linked"
    counts[key] = counts.get(key, 0) + 1


def _capture_private(path: Path, info: os.stat_result, destination: str, relpath: str,
                     digests: dict[str, str] | None) -> bool:
    """Private copy of a live file, scanned as written; remembers its digest for the stat tuple it was read under.

    The digest comes from the bytes that were captured (the same bytes that were scanned), so a later
    save can skip re-reading the file while its stat tuple is unchanged and clearly older than the racy window.
    """
    generation = _SECRET_GENERATION
    started = time.time_ns()
    identity, cloned = _capture_snapshot_file(path, Path(destination))
    digest = identity[1]
    if digests is not None and digest:
        digests[relpath] = digest
    newest_change = max(info.st_mtime_ns, info.st_ctime_ns)
    if digest and generation == _SECRET_GENERATION and newest_change < started - _RACY_WINDOW_NS:
        _DIGEST_MEMO[(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
                      generation)] = digest
    return cloned


def _copy_pooled_fallback(source: Path, target: Path) -> None:
    """Private copy of a staged pool file for a filesystem that refuses the hard link."""
    shutil.copyfile(source, target)
    os.chmod(target, 0o444)
    if _pool_identity(target) != _pool_identity(source):
        raise CheckpointError("checkpoint artifact identity conflict")


_OBJECT_NAME = re.compile(
    r"(?P<object>[0-9a-f]{64}-[0-7]+)"
    r"|(?P<temporary>\.[0-9]+-[^/]*\.tmp)"
    r"|quarantine-(?P<quarantine_ns>0|[1-9][0-9]*)-"
    r"(?P<quarantine_object>[0-9a-f]{64}-[0-7]+)"
)


def _prune_objects(root: Path) -> None:
    """Drop objects no checkpoint links any more, and temporaries a killed writer left behind.

    Only names this store writes, in the store's own folder: a replaced folder is never followed.
    """
    try:
        folder = os.open(root / OBJECT_DIR, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return
    objects = root / OBJECT_DIR
    try:
        names = os.listdir(folder)
        quarantines: dict[str, tuple[int, str]] = {}
        quarantine_names: dict[str, tuple[str, int]] = {}
        for name in names:
            match = _OBJECT_NAME.fullmatch(name)
            if match is None or match.group("quarantine_object") is None:
                continue
            try:
                timestamp = int(match.group("quarantine_ns"))
            except ValueError:
                continue
            object_name = match.group("quarantine_object")
            quarantine_names[name] = (object_name, timestamp)
            candidate = (timestamp, name)
            if candidate > quarantines.get(object_name, (-1, "")):
                quarantines[object_name] = candidate

        live: dict[str, tuple[int, int, int, int]] = {}
        for name in names:
            match = _OBJECT_NAME.fullmatch(name)
            if match is None:
                continue
            try:
                info = os.stat(name, dir_fd=folder, follow_symlinks=False)
                if match.group("quarantine_object") is not None:
                    recognized = quarantine_names.get(name)
                    if (recognized is not None
                            and quarantines.get(recognized[0]) != (recognized[1], name)
                            and stat.S_ISREG(info.st_mode)):
                        os.unlink(name, dir_fd=folder)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    continue
                if match.group("temporary") is not None or info.st_nlink <= 1:
                    os.unlink(name, dir_fd=folder)
                    continue
                live[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            except OSError:
                pass

        _VERIFIED_OBJECTS.intersection_update(
            item for item in _VERIFIED_OBJECTS
            if Path(item[1]).parent != objects
            or live.get(Path(item[1]).name) == item[2:6]
        )
    except (OSError, NotImplementedError):
        pass
    finally:
        os.close(folder)


def _sends(run_dir: Path) -> int:
    """Physical model sends so far: the host's wire counter when it registered one."""
    wire_state = _PARTICIPANTS.get("wire")
    if wire_state is not None and type(wire_state.get("n")) is int:
        return wire_state["n"]
    wire = run_dir / "wire"
    return len(list(wire.glob("*.request.json"))) if wire.is_dir() else 0


def _fsync_dir(path: Path) -> None:
    if _atomic_checkpoints():
        if __package__:
            from .kernel_turn_publication import sync_directory
        else:
            from kernel_turn_publication import sync_directory
        sync_directory(path)
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_snapshot(root: Path, new_pool_files: list[Path],
                    durable_inodes: set[tuple[int, int]] | None = None) -> None:
    """Flush snapshot data in the writer before LATEST can name it or old data is pruned."""
    seen: set[tuple[int, int]] = set(durable_inodes or ())
    for directory, _, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = Path(directory) / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                continue
            identity = (info.st_dev, info.st_ino)
            if identity not in seen:
                _fsync_file(path)
                seen.add(identity)
    for path in new_pool_files:
        info = path.stat()
        identity = (info.st_dev, info.st_ino)
        if identity not in seen:
            _fsync_file(path)
            seen.add(identity)
    for directory, _, _ in os.walk(root, topdown=False, followlinks=False):
        _fsync_dir(Path(directory))
    for directory in {path.parent for path in new_pool_files}:
        _fsync_dir(directory)
    if new_pool_files:
        _fsync_dir(new_pool_files[0].parent.parent)


def _unpublished_pool_files(previous: Path | None, root: Path,
                            manifests: dict[str, list],
                            staged_files: list[Path] | None = None) -> list[Path]:
    """Return pre-existing pool files that need an fsync before a snapshot references them."""
    try:
        old = json.loads((previous / MANIFEST_FILE).read_text()) if previous is not None else {}
    except (OSError, ValueError):
        old = {}
    if not isinstance(old, dict):
        old = {}
    staged_targets = set()
    artifact_root = _SAVE.get("artifact_root")
    if isinstance(artifact_root, Path):
        staged_pool_root = artifact_root / POOL_DIR
        for path in staged_files or ():
            try:
                staged_targets.add(root / POOL_DIR / path.relative_to(staged_pool_root))
            except ValueError:
                continue
    files = []
    for relative, rows in manifests.items():
        old_names = {row[0] for row in old.get(relative, ())}
        store = root / POOL_DIR / relative.replace("/", "__")
        files.extend(store / row[0] for row in rows
                     if row[0] not in old_names and store / row[0] not in staged_targets)
    return files


def _artifact_path(root: Path, relative: str) -> Path:
    parts = Path(relative).parts
    if (Path(relative).is_absolute() or not parts or parts[0] not in {POOL_DIR, CACHE_DIR}
            or any(part in {"", ".", ".."} for part in parts)):
        raise CheckpointError("invalid checkpoint artifact transaction")
    return root.joinpath(*parts)


def _write_artifact_transaction(tmp: Path, artifacts: list[str], placed: frozenset[str] | set[str] = frozenset()) -> None:
    """Record the files this save owns; ``placed`` ones already left staging for their shared place."""
    marker = tmp / ARTIFACT_TRANSACTION_FILE
    temporary = tmp / f"{ARTIFACT_TRANSACTION_FILE}.tmp"
    staged_root = tmp / ARTIFACT_STAGE_DIR
    entries = []
    for relative in sorted(set(artifacts)):
        info = _artifact_path(tmp.parent if relative in placed else staged_root, relative).lstat()
        if not stat.S_ISREG(info.st_mode):
            raise CheckpointError("checkpoint artifact staging contains a non-file")
        entries.append({"path": relative, "device": info.st_dev, "inode": info.st_ino})
    data = json.dumps({"artifacts": entries}, separators=(",", ":")).encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, marker)
    _fsync_dir(tmp)


def _publish_staged_artifacts(root: Path, tmp: Path) -> None:
    """Publish staged pool/chunk files only after the captured HOME identity was rechecked."""
    staged_root = tmp / ARTIFACT_STAGE_DIR
    if not staged_root.is_dir():
        return
    if staged_root.is_symlink():
        raise CheckpointError("checkpoint artifact staging contains a symlink")

    staged_files: list[tuple[str, Path]] = []
    for directory, dirnames, filenames in os.walk(staged_root, followlinks=False):
        current = Path(directory)
        for name in dirnames:
            if (current / name).is_symlink():
                raise CheckpointError("checkpoint artifact staging contains a symlink")
        for name in filenames:
            source = current / name
            if not stat.S_ISREG(source.lstat().st_mode):
                raise CheckpointError("checkpoint artifact staging contains a non-file")
            relative = source.relative_to(staged_root).as_posix()
            target = _artifact_path(root, relative)
            staged_files.append((relative, source))
    staged_files.sort()
    owned: list[str] = []
    real_root = _real_path(root)
    for relative, source in staged_files:
        target = _artifact_path(root, relative)
        if not _path_within(_real_path(target.parent), real_root):
            raise CheckpointError("checkpoint artifact target is outside its store")
        try:
            target_info = target.lstat()
        except FileNotFoundError:
            owned.append(relative)
            continue
        if (not stat.S_ISREG(target_info.st_mode)
                or _pool_identity(source) != _pool_identity(target)):
            raise CheckpointError("checkpoint artifact identity conflict")

    if owned:
        _write_artifact_transaction(tmp, owned)
    owned_set = set(owned)
    placed: set[str] = set()
    published_dirs: set[Path] = set()
    published_files: list[tuple[Path, tuple[int, str]]] = []
    for relative, source in staged_files:
        target = _artifact_path(root, relative)
        known = _POOL_KNOWN.get(str(target.parent), {}).get(target.name)
        expected = known if Path(relative).parts[0] == POOL_DIR and known is not None else (
            _pool_identity(source))
        published_files.append((target, expected))
        if relative not in owned_set:
            continue
        _private_dir(target.parent)
        if not _path_within(_real_path(target.parent), real_root):
            raise CheckpointError("checkpoint artifact target is outside its store")
        try:
            os.link(source, target)
        except FileExistsError:
            target_info = target.lstat()
            if (not stat.S_ISREG(target_info.st_mode)
                    or _pool_identity(source) != _pool_identity(target)):
                raise CheckpointError("checkpoint artifact identity conflict")
            # A target created after the preflight is not owned by this transaction.
            owned_set.remove(relative)
            _write_artifact_transaction(tmp, sorted(owned_set), placed)
        except OSError as error:
            if error.errno not in LINK_FALLBACK_ERRNOS:
                raise
            if os.path.lexists(target):
                raise CheckpointError("checkpoint artifact identity conflict") from error
            os.chmod(source, 0o444)
            try:
                os.rename(source, target)  # keeps the recorded inode, so a refused save can remove it
                placed.add(relative)
            except OSError as moved:
                if moved.errno not in LINK_FALLBACK_ERRNOS:
                    raise
                try:
                    _copy_pooled_fallback(source, target)
                    placed.add(relative)
                    _write_artifact_transaction(tmp, sorted(owned_set), placed)
                except BaseException:
                    target.unlink(missing_ok=True)
                    raise
            _record(root.parent, event="link_fallback", target=relative)
        published_dirs.add(target.parent)

    for directory in published_dirs:
        _fsync_dir(directory)
    if published_dirs:
        for directory in {path.parent for path in published_dirs}:
            _fsync_dir(directory)
        _fsync_dir(root)
    shutil.rmtree(staged_root)
    # Link changes invalidate staging identities. Hash and scan the published bytes
    # from one fd; only that scan may cache the published inode under its digest.
    for target, expected in published_files:
        generation = _SECRET_GENERATION
        identity = _check_snapshot_file(target, digest_hint=expected[1], force_scan=True)
        if identity != expected:
            raise CheckpointError("checkpoint artifact identity conflict")
        if generation != _SECRET_GENERATION:
            raise CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED")
    if owned and not owned_set:
        (tmp / ARTIFACT_TRANSACTION_FILE).unlink(missing_ok=True)
        _fsync_dir(tmp)


def _checkpoint_artifact_references(root: Path) -> set[str] | None:
    """Return every shared artifact referenced by a complete checkpoint, or None if uncertain."""
    referenced: set[str] = set()
    # _checkpoints is a best-effort reader that skips unreadable metadata. Cleanup
    # needs a complete reference set, so enumerate candidates without that filter.
    checkpoints = []
    try:
        for path in root.iterdir():
            if path.name.startswith(".") or path.name in {POOL_DIR, CACHE_DIR, OBJECT_DIR}:
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                return None
            if stat.S_ISDIR(info.st_mode):
                checkpoints.append(path)
    except OSError:
        return None
    for checkpoint in checkpoints:
        try:
            meta = json.loads((checkpoint / META_FILE).read_text())
        except (OSError, ValueError, RecursionError):
            return None
        if not isinstance(meta, dict):
            return None
        try:
            manifest = json.loads((checkpoint / MANIFEST_FILE).read_text())
        except FileNotFoundError:
            if meta.get("blob_manifest_sha256") is not None:
                return None
            manifest = {}  # A legacy checkpoint had no pooled files to keep.
        except (OSError, ValueError, RecursionError):
            return None
        if not isinstance(manifest, dict):
            return None
        chunks = meta.get("chunks", ())
        if not isinstance(chunks, (list, tuple)):
            return None
        for name in chunks:
            if type(name) is not str or Path(name).name != name or name in {"", ".", ".."}:
                return None
            referenced.add(f"{CACHE_DIR}/{name}")
        for field in ("chunk_index", "file_chunks"):
            indexes = meta.get(field, {})
            if not isinstance(indexes, dict):
                return None
            for rows in indexes.values():
                if not isinstance(rows, (list, tuple)):
                    return None
                for row in rows:
                    if (not isinstance(row, (list, tuple)) or not row
                            or type(row[0]) is not str or Path(row[0]).name != row[0]
                            or row[0] in {"", ".", ".."}):
                        return None
                    referenced.add(f"{CACHE_DIR}/{row[0]}")
        for relative, rows in manifest.items():
            if (type(relative) is not str or Path(relative).is_absolute()
                    or ".." in Path(relative).parts or not isinstance(rows, list)):
                return None
            store = relative.replace("/", "__")
            for row in rows:
                if (not isinstance(row, (list, tuple)) or not row or type(row[0]) is not str
                        or Path(row[0]).name != row[0] or row[0] in {"", ".", ".."}):
                    return None
                referenced.add(f"{POOL_DIR}/{store}/{row[0]}")
    return referenced


def _cleanup_checkpoint_transaction(root: Path, tmp: Path) -> None:
    """Remove only transaction-owned artifacts that no complete checkpoint references."""
    marker = tmp / ARTIFACT_TRANSACTION_FILE
    try:
        document = json.loads(marker.read_text())
        artifacts = document.get("artifacts") if isinstance(document, dict) else None
        references = _checkpoint_artifact_references(root)
        if isinstance(artifacts, list) and references is not None:
            real_root = _real_path(root)
            for entry in artifacts:
                if not isinstance(entry, dict):
                    continue
                relative = entry.get("path")
                device, inode = entry.get("device"), entry.get("inode")
                if (type(relative) is not str or type(device) is not int
                        or type(inode) is not int):
                    continue
                target = _artifact_path(root, relative)
                if relative in references or not _path_within(_real_path(target.parent), real_root):
                    continue
                try:
                    info = target.lstat()
                except FileNotFoundError:
                    continue
                if ((info.st_dev, info.st_ino) == (device, inode)
                        and (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode))):
                    target.unlink()
    except (OSError, ValueError, TypeError, CheckpointError):
        # A damaged ownership record is not authority to delete shared data.
        pass
    finally:
        _POOL_KNOWN.clear()
        _VERIFIED_POOL_IDENTITIES.clear()
        shutil.rmtree(tmp, ignore_errors=True)


def _record(run_dir: Path, **fields) -> None:
    try:
        target = run_dir / "checkpoints" / "events.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as stream:
            stream.write(json.dumps({"ts": round(time.time(), 3), **fields}, default=str) + "\n")
    except OSError:
        pass


ASYNC_ENV = "STATEFRAME_CHECKPOINT_ASYNC"
CHILD_TIMEOUT_S = 600


def _async_enabled() -> bool:
    value = os.environ.get(ASYNC_ENV, "on").strip().lower() or "on"
    if value not in {"on", "off"}:
        raise ValueError(f"{ASYNC_ENV} must be on or off")
    return value == "on" and hasattr(os, "fork")


_OBJECT_CHECKPOINT_SCHEMA = "stateframe-engine-checkpoint/2"


def _workspace_objects_on() -> bool:
    return os.environ.get("STATEFRAME_WORKSPACE_OBJECTS") == "on"


def _workspace_checkpoint_codec():
    if __package__:
        from . import kernel_workspace_checkpoint
    else:
        import kernel_workspace_checkpoint
    return kernel_workspace_checkpoint


def _object_checkpoint(directory: Path) -> bool:
    try:
        with (directory / META_FILE).open("rb") as stream:
            prefix = stream.read(16 << 10)
        value = json.loads(prefix)
    except (OSError, ValueError, RecursionError):
        return False
    return isinstance(value, dict) and value.get("schema") == _OBJECT_CHECKPOINT_SCHEMA


def save(frame, step: int, run_dir: Path, *, background: bool = False) -> Path:
    """Write checkpoint ``<run>/checkpoints/<step>-<pid>`` atomically; returns its directory.

    The in-memory capture and run-file snapshot happen here, synchronously, at the step
    boundary (pooled files are copied into private storage and scanned before publication).
    Pickling and writing can then run in a forked
    child (``background``): it pickles its own copy-on-write image of the memory captured at this
    instant while the engine continues, and makes the checkpoint visible (LATEST) only when complete.
    """
    if _workspace_objects_on():
        codec = _workspace_checkpoint_codec()
        try:
            return codec.capture(frame, step, run_dir)
        except (OSError, ValueError, pickle.PickleError) as error:
            raise CheckpointError("workspace checkpoint could not be saved") from error
        finally:
            _SAVE.clear()
            _STATE.pop("protected_home_identity", None)
    _STATE["protected_home_identity"] = _verify_stateframe_home_identity(run_dir)
    cache_run = os.path.realpath(run_dir)
    if _STATE.get("verified_pool_cache_run") != cache_run:
        _VERIFIED_POOL_IDENTITIES.clear()
        _DIGEST_MEMO.clear()
        _POOL_KNOWN.clear()
        _STATE["verified_pool_cache_run"] = cache_run
    started = time.perf_counter()
    payload, ctx = collect(frame)
    collected = time.perf_counter()
    # The checkpoint is taken at the TOP of iteration ``step`` (nothing of it has run yet);
    # restore resumes the loop from ``step`` again, so the loop variable is stored one back.
    payload["frame"]["step"] = step - 1
    clocks = {"monotonic": time.monotonic(), "perf_counter": time.perf_counter()}
    if not _SIGNING_KEY:
        raise CheckpointError("no checkpoint signing key registered by the host: not saving")
    root = run_dir / "checkpoints"
    _private_dir(root)
    with (root / ".checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # The writer inherits this lock across fork; its publication and GC finish before
        # another process can inspect or change the pool or LATEST.
        final = root / f"{step:04d}-{os.getpid()}"
        tmp = root / f".{step:04d}-{os.getpid()}.tmp"
        if tmp.exists():
            raise CheckpointError(f"checkpoint temporary {tmp.name} already exists")
        _private_dir(tmp)
        try:
            return _save_into(payload, ctx, step, run_dir, root, final, tmp, clocks, started,
                              collected, background)
        except BaseException:
            if _STATE.get("child_tmp") != tmp:
                _cleanup_checkpoint_transaction(root, tmp)
                _prune_objects(root)  # a refused save leaves no run-file bytes behind
            raise
        finally:
            _SAVE.clear()
            _STATE.pop("protected_home_identity", None)


def _seed_for_run(run_dir: Path) -> Path | None:
    try:
        seed = Path(json.loads((run_dir / "env-seed.json").read_text())["path"])
        document = json.loads((seed / "seed.json").read_text())
        if (seed.parent.name == "env-seeds" and re.fullmatch(r"[0-9a-f]{64}", seed.name)
                and document.get("key") == seed.name and seed.is_dir()):
            return seed
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _atomic_checkpoints() -> bool:
    return os.environ.get(_PUBLICATION_SWITCH, "").strip().lower() == "on"


def _read_publication_index(root: Path, *, names: bool = True) -> tuple[int, tuple[str, ...]] | None:
    try:
        with os.fdopen(os.open(root / _PUBLICATION_INDEX, os.O_RDONLY | os.O_NOFOLLOW
                               | os.O_NONBLOCK), "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("invalid index file type")
            header = stream.readline(512)
            if not header.endswith(b"\n"):
                raise ValueError("invalid index header")
            document = json.loads(header)
            payload = stream.read() if names else None
        if not isinstance(document, dict):
            raise ValueError("invalid checkpoint publication index")
        seq, latest = document.get("seq"), document.get("latest")
        if (document.get("format") != 1 or type(seq) is not int or seq < -1
                or (latest is not None and not _valid_checkpoint_name(latest))):
            raise ValueError("invalid checkpoint publication index header")
        if payload is None:
            return seq, (latest,) if latest is not None else ()
        if hashlib.sha256(payload).hexdigest() != document.get("sha256"):
            raise ValueError("checkpoint publication index checksum mismatch")
        listed = json.loads(payload)
        if (not isinstance(listed, list) or any(not _valid_checkpoint_name(item) for item in listed)
                or len(set(listed)) != len(listed) or (listed[-1] if listed else None) != latest):
            raise ValueError("invalid checkpoint publication index")
        return seq, tuple(listed)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, RecursionError) as error:
        raise CheckpointError("checkpoint publication index cannot be read") from error


def _valid_checkpoint_name(name: object) -> bool:
    return (isinstance(name, str) and bool(name) and not name.startswith(".")
            and Path(name).name == name)


def _write_publication_index(root: Path, seq: int, names: tuple[str, ...]) -> None:
    if __package__:
        from .kernel_turn_publication import atomic_file
    else:
        from kernel_turn_publication import atomic_file

    payload = json.dumps(names, separators=(",", ":")).encode()
    data = json.dumps({"format": 1, "seq": seq, "latest": names[-1] if names else None,
                       "sha256": hashlib.sha256(payload).hexdigest()},
                      sort_keys=True, separators=(",", ":")).encode()
    atomic_file(root / _PUBLICATION_INDEX, data + b"\n" + payload)


def publication_candidates(root: Path) -> tuple[Path, ...] | None:
    """Expose committed checkpoint candidates to the legacy selector on resume."""
    index = _read_publication_index(root)
    return None if index is None else tuple(root / name for name in index[1])


def recover_publication_latest(root: Path) -> None:
    """Recover the legacy pointer from the durably indexed checkpoint list."""
    candidates = publication_candidates(root)
    if not candidates:
        return
    name = candidates[-1].name
    try:
        mirrored = (root / "LATEST").read_text().strip()
    except FileNotFoundError:
        mirrored = None
    if mirrored != name:
        _mirror_checkpoint(root, name)


def _mirror_checkpoint(root: Path, name: str) -> None:
    if __package__:
        from .kernel_turn_publication import atomic_file
    else:
        from kernel_turn_publication import atomic_file
    atomic_file(root / "LATEST", (name + "\n").encode())


def _publish_indexed_checkpoint(root: Path, tmp: Path, final: Path, run_dir: Path) -> None:
    if __package__:
        from .kernel_turn_publication import sync_directory
    else:
        from kernel_turn_publication import sync_directory
    index = _read_publication_index(root)
    if index is None:
        raise CheckpointError("checkpoint publication index was not initialized")
    seq, previous = index
    new_seq = json.loads((tmp / META_FILE).read_bytes())["seq"]
    if type(new_seq) is not int or new_seq <= seq:
        raise CheckpointError("checkpoint publication sequence did not advance")
    # The directory and every referenced file are durable before the index may name it.
    os.rename(tmp, final)
    sync_directory(root)
    names = (*previous, final.name)[-_keep():]
    _write_publication_index(root, new_seq, names)
    _mirror_checkpoint(root, final.name)
    retained = tuple(root / name for name in names)
    keep = set(names)
    for name in previous:
        if name not in keep:
            shutil.rmtree(root / name, ignore_errors=True)
    _prune_chunks(root, retained)
    _prune_pool(root, run_dir, retained)
    _prune_objects(root)
    marker = final / ARTIFACT_TRANSACTION_FILE
    if marker.exists():
        marker.unlink()
        sync_directory(final)


def _publish_checkpoint(root: Path, tmp: Path, final: Path, run_dir: Path) -> None:
    """Publish only complete snapshots; keep the pointer's target throughout pruning."""
    if final.exists():
        raise CheckpointError(f"checkpoint {final.name} already exists")
    if _atomic_checkpoints():
        _publish_indexed_checkpoint(root, tmp, final, run_dir)
        return
    os.rename(tmp, final)
    _fsync_dir(root)
    checkpoints = _checkpoints(root)
    newest = checkpoints[-1]
    latest_tmp = root / f".LATEST.{os.getpid()}"
    latest_tmp.write_text(newest.name + "\n")
    _fsync_file(latest_tmp)
    os.replace(latest_tmp, root / "LATEST")
    _fsync_dir(root)
    keep = set(checkpoints[-_keep():])
    keep.add(newest)
    for old in checkpoints:
        if old not in keep:
            shutil.rmtree(old, ignore_errors=True)
    _prune_chunks(root)
    _prune_pool(root, run_dir)
    _prune_objects(root)
    marker = final / ARTIFACT_TRANSACTION_FILE
    try:
        marker.unlink()
    except FileNotFoundError:
        pass
    else:
        _fsync_dir(final)


def _save_into(payload, ctx, step, run_dir, root, final, tmp, clocks, started, collected, background):
    """The rest of save(): snapshot the run files into ``tmp`` and write (here or in the writer)."""
    _STATE["protected_home_identity"] = _verify_stateframe_home_identity(run_dir)
    captured_secret_generation = _SECRET_GENERATION
    _SAVE.clear()
    _SAVE.update(artifact_root=tmp / ARTIFACT_STAGE_DIR, pool_root=root / POOL_DIR)
    (tmp / "run").mkdir()
    seq = _next_seq(root)  # ordering for pruning/LATEST (see _checkpoints), fixed before any fork
    _STATE["next_seq"] = seq + 1
    home, home_subtree, excluded = _snapshot_home_exclusions(run_dir)
    _SAVE["snapshot_exclusions"] = (home, home_subtree, excluded)
    if (root / OBJECT_DIR).is_symlink():
        raise CheckpointError("checkpoint object store is a symlink")
    _SAVE["snapshot_objects"] = root / OBJECT_DIR
    entries = sorted(
        path.name for path in run_dir.iterdir()
        if path.name not in SKIP_RUN_ENTRIES
        and not _snapshot_path_excluded(path, home, home_subtree, excluded)
    )
    previous = latest(run_dir)
    prior_meta = {}
    if previous is not None:
        try:
            prior_meta = json.loads((previous / META_FILE).read_text())
        except (OSError, ValueError):
            pass
    recorded_digests = prior_meta.get("file_sha256")
    recorded_digests = recorded_digests if isinstance(recorded_digests, dict) else {}
    file_digests: dict[str, dict[str, str]] = {}
    copied = {"linked": 0, "copied": 0, "cloned": 0}
    manifests: dict[str, list] = {}
    file_chunks: dict[str, list] = {}
    new_pool_files: list[Path] = []
    durable_inodes: set[tuple[int, int]] = set()
    file_modes: dict[str, int] = {}
    file_mtimes: dict[str, int] = {}
    bytes_new = [0]
    _SAVE.update(copy_modes=file_modes, copy_mtimes=file_mtimes, bytes_new=bytes_new, counts=copied,
                 durable_inodes=durable_inodes)
    for name in entries:
        source, destination = run_dir / name, tmp / "run" / name
        if (name == "worker" or name in POOLED_DIRS) and source.is_dir() and not source.is_symlink():
            _snapshot_tree(source, destination, name, root / POOL_DIR, manifests, new_pool_files,
                           file_chunks, prior_meta.get("file_chunks", {}), seq,
                           home=home, home_subtree=home_subtree, excluded=excluded)
            copied["cloned"] += 1
            continue
        if not source.is_dir() or source.is_symlink():
            if source.is_symlink():
                _capture_snapshot_symlink(source, destination, home)
                continue
            if source.is_file() and not source.is_symlink():
                _capture_run_file(source, destination, name)
                continue
            if _clonefile(source, destination):
                copied["cloned"] += 1
                continue
            _clone(source, destination)
            continue
        prior = previous / "run" / name if previous is not None else None
        seed = _seed_for_run(run_dir) if name == "env" else None
        # Boundary filters and the object store ride in _SAVE, so wrappers that only forward
        # ``seed`` keep the original copy-hook call contract.
        prior_map = recorded_digests.get(name)
        file_digests[name] = {}
        _SAVE["copy_digests"] = (file_digests[name],
                                 prior_map if isinstance(prior_map, dict) else {})
        _SAVE["copy_prefix"] = name
        try:
            for key, value in _copy_tree(source, destination, prior, durable_inodes,
                                         seed=seed).items():
                copied[key] = copied.get(key, 0) + value
        finally:
            _SAVE.pop("copy_digests", None)
            _SAVE.pop("copy_prefix", None)
    manifest_bytes = json.dumps(manifests, sort_keys=True, separators=(",", ":")).encode()
    pool_sha256 = {
        f"{relative}/{name}": _POOL_KNOWN[str(root / POOL_DIR / relative.replace('/', '__'))][name][1]
        for relative, rows in manifests.items() for name, _size in rows
    }
    new_pool_files.extend(_unpublished_pool_files(previous, root, manifests, new_pool_files))
    with os.fdopen(os.open(tmp / MANIFEST_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(manifest_bytes)
    files_done = time.perf_counter()
    meta = {
        "schema": SCHEMA, "step": step, "sends": _sends(run_dir), "entries": entries,
        "python": list(sys.version_info[:3]), "engine_root": _engine_root(),
        "engine_files_sha256": engine_fingerprint(bind_host=True), "pid": os.getpid(),
        "created": time.time(), "seq": seq, "clocks": clocks, "files": copied,
        "blob_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "pool_sha256": pool_sha256,
        "file_chunks": file_chunks,
        "file_sha256": file_digests,
        "file_modes": file_modes,
        "file_mtime_ns": file_mtimes,
        "link_fallback": copied.get("link_fallback", 0),
        "bytes_new": bytes_new[0] + len(manifest_bytes),
        "blob_count": sum(len(names) for names in manifests.values()),
        "seconds": {"collect": round(collected - started, 3), "files": round(files_done - collected, 3),
                    "capture": round(time.perf_counter() - started, 3)},
    }
    _STATE["phase_seconds"] = dict(meta["seconds"])
    prior_index = prior_meta.get("chunk_index", {}) or {}

    def write() -> None:
        t0 = time.perf_counter()
        def dump(writer):
            _SAVE.clear()
            _SAVE.update(root=root, index=prior_index, artifact_root=tmp / ARTIFACT_STAGE_DIR)
            try:
                dump_to(writer, payload, ctx)
                return dict(_SAVE.get("index_out", {}))
            except Exception as error:  # noqa: BLE001 - identify failures at the serializer boundary
                _mark_serialization_failure(error, payload)
                raise
            finally:
                _SAVE.clear()
        with os.fdopen(os.open(tmp / STATE_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            writer = _StateWriter(stream)
            try:
                index = dump(writer)
            except pickle.PicklingError:
                # The first attempt may have written a prefix; discard it and all writer state.
                _SCANS.clear()
                ctx.update(cvars=_cvar_names(), hostfns=_host_functions())
                stream.seek(0)
                stream.truncate(0)
                writer = _StateWriter(stream)
                index = dump(writer)
            writer.finish()
            stream.flush()
        meta.update(state_bytes=writer.count, state_sha256=writer.digest.hexdigest(),
                    state_codec=STATE_CODEC, state_stored_sha256=writer.stored_digest.hexdigest(),
                    state_stored_bytes=writer.stored_count,
                    chunk_index=index,
                     chunks=sorted({row[0] for rows in index.values() for row in rows}
                                   | {row[0] for rows in file_chunks.values() for row in rows}))
        meta["bytes_new"] += writer.stored_count
        meta["seconds"]["write"] = round(time.perf_counter() - t0, 3)
        meta["signature"] = _signature(meta, meta["state_sha256"])
        with os.fdopen(os.open(tmp / META_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            stream.write(json.dumps(meta, indent=1, sort_keys=True))
        _fsync_snapshot(tmp, new_pool_files, durable_inodes)
        if (root / CACHE_DIR).is_dir():
            _fsync_dir(root / CACHE_DIR)
        if (root / OBJECT_DIR).is_dir():
            _fsync_dir(root / OBJECT_DIR)
        _STATE["protected_home_identity"] = _verify_stateframe_home_identity(run_dir)
        if _SECRET_GENERATION != captured_secret_generation:
            # A newly discovered key was absent from earlier scans, even on coarse-timestamp filesystems.
            raise CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED")
        _publish_staged_artifacts(root, tmp)
        _STATE["protected_home_identity"] = _verify_stateframe_home_identity(run_dir)
        if _SECRET_GENERATION != captured_secret_generation:
            raise CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED")
        _publish_checkpoint(root, tmp, final, run_dir)
        return meta

    try:
        if background:
            _fork_writer(run_dir, step, final, write, tmp, diagnostic_payload=payload)
        else:
            write()
    except BaseException:
        _cleanup_checkpoint_transaction(root, tmp)
        raise
    return final


def _rss_mb() -> float | None:
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
        return round(peak / (1 << 20) if sys.platform == "darwin" else peak / 1024, 1)
    except (ImportError, OSError, ValueError):
        return None


def _minimum_free_bytes(run_dir: Path) -> int:
    """Reserve space for another stored state image; process peak is the first-save estimate."""
    raw = os.environ.get(MIN_FREE_ENV, "").strip()
    if raw:
        try:
            value = int(raw)
            if value < 0:
                raise ValueError("negative free-space floor")
            return value
        except ValueError:
            _warn_once("min_free", f"{MIN_FREE_ENV} must be a non-negative integer; using the automatic floor")
    previous = latest(run_dir)
    if previous is not None:
        try:
            stored_state_bytes = (previous / STATE_FILE).stat().st_size
        except OSError:
            stored_state_bytes = 0
    else:
        stored_state_bytes = 0
    if not stored_state_bytes:
        stored_state_bytes = int((_rss_mb() or 0) * (1 << 20))
    return max(1 << 30, 3 * stored_state_bytes)


def _low_disk_space(run_dir: Path) -> tuple[int, int] | None:
    """Return (free, floor) only below the floor; unknown free space permits saving."""
    floor = _minimum_free_bytes(run_dir)
    if floor == 0:
        return None
    root = run_dir / "checkpoints"
    try:
        filesystem = os.statvfs(root if root.is_dir() else run_dir)
    except OSError:
        return None
    free = filesystem.f_bavail * filesystem.f_frsize
    return (free, floor) if free < floor else None


def _reap() -> bool:
    """True when no background writer is running (reaps a finished one)."""
    pid = _STATE.get("child")
    if not pid:
        return True
    try:
        done, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        done, status = pid, 0
    if done == 0:
        return False
    _STATE["child"] = None
    tmp, _STATE["child_tmp"] = _STATE.get("child_tmp"), None
    run_dir, step = _STATE.get("child_run_dir"), _STATE.get("child_step")
    _STATE["child_run_dir"], _STATE["child_step"] = None, None
    writer_generation = _STATE.get("child_writer_generation")
    _STATE["child_writer_generation"] = None
    journal_offset, _STATE["child_journal_offset"] = _STATE.get("child_journal_offset"), None
    journal_file_id, _STATE["child_journal_file_id"] = _STATE.get("child_journal_file_id"), None
    journal_missing, _STATE["child_journal_missing"] = _STATE.get("child_journal_missing"), False
    if os.waitstatus_to_exitcode(status) != 0:
        if tmp is not None:
            with (tmp.parent / ".checkpoint.lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                _cleanup_checkpoint_transaction(tmp.parent, tmp)
                _prune_objects(tmp.parent)  # a refused save leaves no run-file bytes behind
            _sweep_stale(tmp.parent)
        _SCANS.clear()  # the writer may have failed on a stale scan: rescan at the next boundary
        reason = _writer_failure_reason(
            run_dir, step, writer_generation, pid, journal_offset,
            journal_file_id, journal_missing=journal_missing,
        )
        _note_failure(run_dir, reason)
    else:
        _STATE["consecutive_failed"] = 0
    return True


def _writer_journal_start(run_dir: Path) -> tuple[int, tuple[int, int] | None, bool]:
    """Return the readable journal's byte size and identity; a missing journal starts at zero."""
    path = Path(run_dir) / "checkpoints" / "events.jsonl"
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            return 0, None, False
        return info.st_size, (info.st_dev, info.st_ino), False
    except FileNotFoundError:
        return 0, None, True
    except OSError:
        return 0, None, False
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _writer_failure_reason(
        run_dir, step: object, writer_generation: object, writer_pid: object,
        journal_offset: object, journal_file_id: object,
        *, journal_missing: object = False) -> str:
    """Read only a post-fork failure from the matching writer process and journal."""
    if (run_dir is None or type(step) is not int or type(writer_generation) is not str
            or type(writer_pid) is not int or type(journal_offset) is not int or journal_offset < 0
            or type(journal_missing) is not bool):
        return "unexpected writer error"
    if journal_file_id is None:
        if not journal_missing:
            return "unexpected writer error"
    elif (type(journal_file_id) is not tuple or len(journal_file_id) != 2
          or any(type(value) is not int for value in journal_file_id) or journal_missing):
        return "unexpected writer error"
    matched_reason = None
    try:
        path = Path(run_dir) / "checkpoints" / "events.jsonl"
        with path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                return "unexpected writer error"
            if journal_file_id is not None and (info.st_dev, info.st_ino) != journal_file_id:
                return "unexpected writer error"
            stream.seek(journal_offset)
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if (not isinstance(event, dict) or event.get("event") != "checkpoint_failed"
                        or type(event.get("step")) is not int or event["step"] != step
                        or event.get("background") is not True
                        or event.get("writer_generation") != writer_generation
                        or type(event.get("writer_pid")) is not int
                        or event["writer_pid"] != writer_pid):
                    continue
                reason = event.get("failure_reason")
                if type(reason) is str and reason in _WRITER_FAILURE_REASONS:
                    matched_reason = _writer_reason(reason, event.get("writer_error_type"))
                else:
                    matched_reason = "unexpected writer error"
    except OSError:  # noqa: BLE001 - a missing or unreadable event is a generic writer failure
        return "unexpected writer error"
    return matched_reason or "unexpected writer error"


def failure_summary() -> dict[str, object]:
    """Failed saves so far in this process and whether saving was turned off (for run vitals)."""
    return {"checkpoint_failures": int(_STATE.get("failed", 0) or 0),
            "checkpoint_saving_disabled": bool(_STATE.get("disabled"))}


def _note_failure(run_dir, reason: str) -> None:
    """Count every failed save and warn once; a later model send makes another attempt eligible."""
    _STATE["failed"] += 1
    _STATE["consecutive_failed"] = _STATE.get("consecutive_failed", 0) + 1
    if _STATE["consecutive_failed"] >= FAILURE_WARNING_THRESHOLD:
        warning = f"{FAILURE_WARNING_THRESHOLD} consecutive checkpoint failures: {reason}"
        if "repeated_save_failures" not in _STATE.setdefault("warned", set()):
            _record(run_dir, event="checkpoint_warning", reason=warning)
        _warn_once("repeated_save_failures",
                   f"{warning}; saving continues after a new model send")


_STALE = re.compile(
    r"\.(?:\d{4}-|LATEST\.)(\d+)(?:\.tmp)?$|\..*-(\d+)(?:-[0-9a-f]{16})?\.(?:z)?pkl\.tmp$"
)


def _sweep_stale(root: Path) -> int:
    """Remove temporaries left by checkpoint writers that no longer run; returns how many."""
    try:
        lock_stream = (root / ".checkpoint.lock").open("a+b")
    except OSError:
        return 0
    with lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        return _sweep_stale_locked(root)


def _sweep_stale_locked(root: Path) -> int:
    removed = 0
    for folder in (root, root / CACHE_DIR):
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            match = _STALE.match(entry.name)
            if match is None:
                continue
            pid = int(match.group(1) or match.group(2))
            if pid == _STATE.get("child") or _pid_alive(pid):
                continue
            if entry.is_dir(follow_symlinks=False):
                _cleanup_checkpoint_transaction(root, Path(entry.path))
                shutil.rmtree(entry.path, ignore_errors=True)
            else:
                try:
                    os.unlink(entry.path)
                except OSError:
                    continue
            removed += 1
    return removed


def _pid_alive(pid: int) -> bool:
    # A Linux zombie owner counts as dead, so its temp file is swept (kernel_pid_liveness).
    if __package__:
        from .kernel_pid_liveness import pid_alive
    else:
        from kernel_pid_liveness import pid_alive
    return pid_alive(pid)


def _fork_writer(run_dir: Path, step: int, final: Path, write, tmp: Path,
                 *, diagnostic_payload=None) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    journal_offset, journal_file_id, journal_missing = _writer_journal_start(run_dir)
    writer_generation = f"{step}:{journal_offset}"
    forked = time.perf_counter()
    pid = os.fork()
    if pid == 0:  # the writer: pickle this frozen image, write, and leave without any cleanup
        if __package__:
            from .kernel_exec_session import close_owner_pipe_writers_after_fork
        else:
            from kernel_exec_session import close_owner_pipe_writers_after_fork
        close_owner_pipe_writers_after_fork()
        code = 1
        try:
            import signal
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.alarm(CHILD_TIMEOUT_S)
            started = time.perf_counter()
            meta = write()
            _record(run_dir, event="checkpoint_saved", step=step, dir=final.name, background=True,
                    writer_generation=writer_generation, writer_pid=os.getpid(),
                    seconds=round(time.perf_counter() - started, 3), sends=meta["sends"],
                    state_bytes=meta["state_bytes"], chunks=len(meta["chunks"]))
            code = 0
        except BaseException as error:  # noqa: BLE001 - recorded, then exit
            _cleanup_checkpoint_transaction(tmp.parent, tmp)
            _prune_objects(tmp.parent)  # a refused save leaves no run-file bytes behind
            reason = _failure_reason(error)
            if (diagnostic_payload is not None and not _serialization_failure_fields(error)
                    and (isinstance(error, pickle.PicklingError)
                         or type(error) in {TypeError, AttributeError, RecursionError})):
                _mark_serialization_failure(error, diagnostic_payload)
            diagnostics = _serialization_failure_fields(error)
            _record(run_dir, event="checkpoint_failed", step=step, background=True,
                    error=reason, failure_reason=reason, writer_error_type=_safe_error_type(error),
                    writer_generation=writer_generation, writer_pid=os.getpid(), **diagnostics)
            _log_serialization_failure(diagnostics)
        finally:
            os._exit(code)
    _STATE["child"], _STATE["child_tmp"], _STATE["child_run_dir"] = pid, tmp, run_dir
    _STATE["child_step"] = step
    _STATE["child_writer_generation"] = writer_generation
    _STATE["child_journal_offset"] = journal_offset
    _STATE["child_journal_file_id"] = journal_file_id
    _STATE["child_journal_missing"] = journal_missing
    _STATE["fork_seconds"] = round(time.perf_counter() - forked, 3)


def _prune_chunks(root: Path, checkpoints: tuple[Path, ...] | None = None) -> None:
    """Remove journal cache chunks no remaining checkpoint references (only those of this run)."""
    folder = root / CACHE_DIR
    if not folder.is_dir():
        return
    referenced: set[str] = set()
    for checkpoint in _checkpoints(root) if checkpoints is None else checkpoints:
        try:
            referenced.update(json.loads((checkpoint / META_FILE).read_text()).get("chunks", ()))
        except (OSError, ValueError):
            return  # cannot prove a chunk unused: keep everything
    for chunk in folder.iterdir():
        if chunk.suffix in (".pkl", CHUNK_SUFFIX) and chunk.name not in referenced:
            chunk.unlink()


def _checkpoints(root: Path) -> list[Path]:
    """Complete checkpoints, oldest first: by the per-run save sequence number (``seq``), which only
    grows, never by wall-clock time (a clock stepped back would sort the newest checkpoint first and
    pruning would delete the one LATEST names). Checkpoints written before ``seq`` existed sort first,
    by creation time."""
    found = []
    for path in root.iterdir() if root.is_dir() else ():
        if path.is_dir() and not path.name.startswith(".") and (path / META_FILE).is_file():
            try:
                meta = json.loads((path / META_FILE).read_text())
            except (OSError, ValueError):
                continue
            seq = meta.get("seq")
            found.append((seq if type(seq) is int else -1, meta.get("created", 0), path.name, path))
    return [row[-1] for row in sorted(found)]


def _next_seq(root: Path) -> int:
    """One more than the highest save sequence number among this run's checkpoints."""
    if _atomic_checkpoints() and root.is_dir():
        index = _read_publication_index(root, names=False)
        if index is None:
            paths = _checkpoints(root)
            seq = json.loads((paths[-1] / META_FILE).read_bytes()).get("seq", -1) if paths else -1
            seq = seq if type(seq) is int else -1
            _write_publication_index(root, seq, tuple(path.name for path in paths))
            # The index names this directory across a power loss, including on a
            # run whose checkpoint root was created for this first transaction.
            _fsync_dir(root.parent)
        else:
            seq = index[0]
        return max(seq + 1, _STATE.get("next_seq", 0))
    highest = -1
    for path in root.iterdir() if root.is_dir() else ():
        if path.is_dir() and not path.name.startswith(".") and (path / META_FILE).is_file():
            try:
                seq = json.loads((path / META_FILE).read_text()).get("seq")
            except (OSError, ValueError):
                continue
            if type(seq) is int:
                highest = max(highest, seq)
    return max(highest + 1, _STATE.get("next_seq", 0))


_FINGERPRINT: str | None = None
_PRODUCT_HOST = False
_HOST_IDENTITY_BOUND = False


def configure_product_host() -> None:
    """Declare product startup: CLI validators and children share the package host.

    Embeddings leave this unset so HOST_MODULE still identifies their own code,
    even when the engine is imported from an installed StateFrame package.
    Once a checkpoint uses that identity, a late declaration cannot change it.
    This process authority is excluded from checkpoint state with this module.
    """
    global _PRODUCT_HOST, _FINGERPRINT
    if not _PRODUCT_HOST and not _HOST_IDENTITY_BOUND:
        _PRODUCT_HOST = True
        _FINGERPRINT = None


def engine_fingerprint(*, bind_host: bool = False) -> str:
    """Hash engine/host code; checkpoint use binds host ownership for this process."""
    global _FINGERPRINT, _HOST_IDENTITY_BOUND
    if _FINGERPRINT is None:
        digest = hashlib.sha256()
        root = Path(_engine_root())
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        # Startup declares who owns the wrappers; package layout cannot distinguish
        # a product child from an embedding using the installed engine.
        if _PRODUCT_HOST:
            host_root: Path | None = root.parent
        else:
            host = getattr(sys.modules.get(HOST_MODULE), "__file__", None)
            host_root = Path(host).resolve().parent if isinstance(host, str) else None
        if host_root is not None:
            for path in sorted(host_root.glob("*.py")):
                digest.update(b"host:" + path.name.encode() + b"\0")
                digest.update(hashlib.sha256(path.read_bytes()).digest())
        _FINGERPRINT = digest.hexdigest()
    if bind_host:
        # Bind before the background writer forks, and independently of restorable state
        # or the digest cache. Parent-side selection must honor an in-process writer.
        _HOST_IDENTITY_BOUND = True
    return _FINGERPRINT


def save_point(frame, step: int, poc) -> None:
    """Step-loop hook. Save failures are recorded; engine stop exceptions propagate after cleanup.

    ``blocked_seconds`` in each save's event is the wall time from entering this hook to leaving
    it (mode checks, send count, writer reap, capture, fork and the event write), plus the time spent
    in hook calls since the previous save that did not save. It is the request-path cost.
    """
    entered = time.perf_counter()
    try:
        _save_point(frame, step, poc, entered)
    finally:
        if not _STATE.pop("just_saved", False):  # a saving call already reported its own time
            _STATE["unsaved_hook_seconds"] = _STATE.get("unsaved_hook_seconds", 0.0) + (time.perf_counter() - entered)


def _save_point(frame, step: int, poc, entered: float) -> None:
    try:
        current = mode()
    except ValueError as error:
        if not _STATE["disabled"]:  # say so once instead of silently never saving
            _STATE["disabled"] = str(error)
            _record(Path(poc).parent, event="checkpoint_disabled", step=step, reason=str(error))
            _warn_once("mode", f"{error}; checkpoints are off for this run (resume will replay)")
        return
    if current == "off" or _STATE["disabled"]:
        return
    if _STATE["restored_step"] == step:
        return  # the state on disk IS this boundary
    if step % _every():
        return
    out = Path(poc)
    run_dir = out.parent
    if current == "save":
        # Consume the send before attempting: a failed capture must not retry at this boundary and
        # publish over the only checkpoint with a replayable tail.
        sends = _sends(run_dir)
        previous_sends = _STATE["last_sends"]
        if sends == 0:
            return  # nothing to resume past yet: a resume before the first send just replays (none)
        if previous_sends is not None and sends <= previous_sends:
            return
        _STATE["last_sends"] = sends
    try:
        if current == "probe":
            probe(frame, step, out / "checkpoint-probe")
            return
        background = _async_enabled() and not _workspace_objects_on()
        if background and not _reap():
            _STATE["last_sends"] = previous_sends  # still writing the last one: try at the next boundary
            _record(run_dir, event="checkpoint_skipped", step=step, reason="previous writer still running")
            return
        low_disk = _low_disk_space(run_dir)
        if low_disk is not None:
            _STATE["last_sends"] = previous_sends
            if not _STATE.get("low_disk", False):
                _record(run_dir, event="checkpoint_skipped_low_disk", step=step,
                        free_bytes=low_disk[0], floor_bytes=low_disk[1])
            _STATE["low_disk"] = True
            return
        _STATE["low_disk"] = False
        started = time.perf_counter()
        # No cyclic-GC pass inside the pause: capture allocates many small objects, and on a 2 GB heap
        # one triggered gen-2 collection took seconds (pauses up to 2.8 s at 100 sends with fork <=25 ms).
        import gc
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            try:
                target = save(frame, step, run_dir, background=background)
            except BaseException as error:  # noqa: BLE001 - record failures, but pass engine stops through
                if (isinstance(error, KeyboardInterrupt)
                        or getattr(error, "boundary_recovery_stop", False)):
                    raise
                reason = _failure_reason(error)
                writer_error_type = _safe_error_type(error)
                _record(run_dir, event="checkpoint_failed", step=step,
                        error=reason, failure_reason=reason, **_capture_stage(error),
                        writer_error_type=writer_error_type,
                        **_serialization_failure_fields(error))
                _log_serialization_failure(_serialization_failure_fields(error))
                _note_failure(run_dir, _writer_reason(reason, writer_error_type))
                return
        finally:
            if gc_was_enabled:
                gc.enable()
        _STATE["saved"] += 1
        if not background:
            _STATE["consecutive_failed"] = 0
            _record(run_dir, event="checkpoint_saved", step=step, dir=target.name,
                    seconds=round(time.perf_counter() - started, 3))
        else:
            pause = round(time.perf_counter() - started, 3)
            fields = dict(event="checkpoint_started", step=step, dir=target.name, pause_seconds=pause,
                          fork_seconds=_STATE.get("fork_seconds"), rss_peak_mb=_rss_mb(),
                          phases=_STATE.get("phase_seconds"))
            # everything up to the event write, plus the non-saving hook calls since the last save
            blocked = time.perf_counter() - entered + _STATE.get("unsaved_hook_seconds", 0.0)
            _record(run_dir, **fields, blocked_seconds=round(blocked, 3))
            _STATE["unsaved_hook_seconds"], _STATE["just_saved"] = 0.0, True
    except Exception as error:  # noqa: BLE001 - checkpoint errors fail open unless the engine is stopping
        if (isinstance(error, KeyboardInterrupt)
                or getattr(error, "boundary_recovery_stop", False)):
            raise
        reason = _failure_reason(error)
        writer_error_type = _safe_error_type(error)
        _record(run_dir, event="checkpoint_failed", step=step,
                error=reason, failure_reason=reason, **_capture_stage(error),
                writer_error_type=writer_error_type,
                **_serialization_failure_fields(error))
        _log_serialization_failure(_serialization_failure_fields(error))
        _note_failure(run_dir, _writer_reason(reason, writer_error_type))


# --------------------------------------------------------------------------- restore

def latest(run_dir: Path) -> Path | None:
    if _atomic_checkpoints():
        root = run_dir / "checkpoints"
        index = _read_publication_index(root, names=False)
        if index is not None:
            if not index[1]:
                return None
            target = root / index[1][-1]
            if not (target / META_FILE).is_file():
                raise CheckpointError("committed checkpoint is missing")
            return target
    pointer = run_dir / "checkpoints" / "LATEST"
    try:
        name = pointer.read_text().strip()
    except OSError:
        return None
    target = run_dir / "checkpoints" / name
    return target if (target / META_FILE).is_file() else None


def _authenticate_state(checkpoint: Path) -> tuple[dict, bytes]:
    """Verify ownership, encoded bytes, signature and raw state before returning pickle bytes."""
    for path in (checkpoint.parent, checkpoint, checkpoint / META_FILE, checkpoint / STATE_FILE):
        if not _owned_private(path):
            raise CheckpointError(f"checkpoint path {path.name} is not this user's private file")
    try:
        meta = json.loads((checkpoint / META_FILE).read_text())
    except (OSError, ValueError) as error:
        raise CheckpointError(f"checkpoint metadata unreadable: {type(error).__name__}") from None
    if meta.get("schema") != SCHEMA:
        raise CheckpointError(f"checkpoint schema {meta.get('schema')!r} is not {SCHEMA}")
    try:
        blob = (checkpoint / STATE_FILE).read_bytes()
    except OSError:
        raise CheckpointError("checkpoint state missing") from None
    codec = _verify_stored_state(blob, meta)
    # The legacy format has no separate stored digest, so its raw checksum is checked here as before.
    raw = _decode_verified_state(blob, meta, codec) if codec is None else None
    import hmac
    state_sha256 = meta.get("state_sha256")
    signature = meta.get("signature")
    if not isinstance(state_sha256, str) or not isinstance(signature, str) or not hmac.compare_digest(
            signature, _signature(meta, state_sha256)):
        raise CheckpointError("checkpoint signature does not verify")
    if raw is None:
        raw = _decode_verified_state(blob, meta, codec)
    del blob
    try:
        manifest_bytes = (checkpoint / MANIFEST_FILE).read_bytes()
    except OSError:
        manifest_bytes = None
    if manifest_bytes is not None or meta.get("blob_manifest_sha256") is not None:
        if manifest_bytes is None or hashlib.sha256(manifest_bytes).hexdigest() != meta.get("blob_manifest_sha256"):
            raise CheckpointError("blob manifest missing or does not match the signed metadata")
        for relative, files in json.loads(manifest_bytes).items():
            store = checkpoint.parent / POOL_DIR / relative.replace("/", "__")
            for name, size in files:
                try:
                    info = os.lstat(store / name)
                except OSError:
                    raise CheckpointError(f"missing pooled blob {relative}/{name}") from None
                if not stat.S_ISREG(info.st_mode):
                    raise CheckpointError(f"pooled file {relative}/{name} is not regular")
                if info.st_size != size:
                    raise CheckpointError(f"pooled file {relative}/{name} changed size since the checkpoint")
                expected = meta.get("pool_sha256", {}).get(f"{relative}/{name}")
                if expected is not None and _pool_identity(store / name)[1] != expected:
                    raise CheckpointError(f"pooled file {relative}/{name} checksum mismatch")
    return meta, raw


def authenticate(checkpoint: Path) -> dict:
    """Is this checkpoint provably ours? Verify both state representations before unpickling."""
    if _object_checkpoint(checkpoint):
        return _workspace_checkpoint_codec().validate_directory(checkpoint, compatible=False)
    meta, _raw = _authenticate_state(checkpoint)
    return meta


def _prune_pool(root: Path, run_dir: Path, checkpoints: tuple[Path, ...] | None = None) -> None:
    """Drop pooled files that no kept checkpoint lists and that the live run no longer has.

    A pooled file still present in the live directory is listed again by the next save, so it is
    kept. A file removed from the live run is freed
    once no kept checkpoint names it, so the pool is bounded by the kept checkpoints plus the live run.
    """
    pool = root / POOL_DIR
    if not pool.is_dir():
        return
    referenced: dict[str, set[str]] = {}
    for checkpoint in root.iterdir() if checkpoints is None else checkpoints:
        if (not checkpoint.is_dir() or checkpoint.is_symlink()
                or checkpoint.name in {POOL_DIR, CACHE_DIR, OBJECT_DIR}):
            continue
        try:
            manifest = json.loads((checkpoint / MANIFEST_FILE).read_text())
        except FileNotFoundError:
            try:
                meta = json.loads((checkpoint / META_FILE).read_text())
            except (OSError, ValueError):
                return
            if meta.get("blob_manifest_sha256") is not None:
                return
            continue
        except (OSError, ValueError):
            return  # cannot prove a pooled file unused: keep everything
        if not isinstance(manifest, dict):
            return
        for relative, files in manifest.items():
            if (not isinstance(relative, str) or not isinstance(files, list)
                    or any(not isinstance(row, list) or not row or not isinstance(row[0], str)
                           for row in files)):
                return
            referenced.setdefault(relative.replace("/", "__"), set()).update(row[0] for row in files)
    for store in pool.iterdir():
        if not store.is_dir():
            continue
        live = run_dir / store.name.replace("__", "/")
        keep = referenced.get(store.name, set())
        for entry in os.scandir(store):
            if entry.name not in keep and not os.path.lexists(live / entry.name):
                os.unlink(entry.path)


def _validate(checkpoint: Path) -> tuple[dict, bytes]:
    """Authenticate the state and check compatibility before any run files are restored."""
    meta, raw = _authenticate_state(checkpoint)
    recorded_files = meta.get("file_sha256")
    if recorded_files is not None:
        if not isinstance(recorded_files, dict):
            raise CheckpointError("checkpoint file digest metadata is invalid")
        for directory, files in recorded_files.items():
            if (not isinstance(directory, str) or Path(directory).name != directory
                    or not isinstance(files, dict)):
                raise CheckpointError("checkpoint file digest metadata is invalid")
            for name, expected in files.items():
                relative = Path(name) if isinstance(name, str) else Path(".")
                if (not isinstance(name, str) or not name or relative.is_absolute()
                        or ".." in relative.parts):
                    raise CheckpointError("checkpoint file digest metadata is invalid")
                key = Path(directory) / relative
                if _file_sha256(checkpoint / "run" / key) != expected:
                    raise CheckpointError(f"checkpoint run file {key.as_posix()} checksum mismatch")
    if list(sys.version_info[:3]) != meta.get("python"):
        raise CheckpointError("checkpoint was written by another Python version")
    if meta.get("engine_files_sha256") != engine_fingerprint(bind_host=True):
        raise CheckpointError("checkpoint was written by another engine build")
    for name in meta.get("entries", ()):
        if not (checkpoint / "run" / name).exists():
            raise CheckpointError(f"checkpoint run copy lacks {name}")
    chunk_digests: dict[str, str] = {}

    def add_chunk(name: object, digest: object) -> None:
        if (not isinstance(name, str) or Path(name).name != name or not name.endswith(CHUNK_SUFFIX)
                or not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise CheckpointError("checkpoint chunk metadata is invalid")
        previous = chunk_digests.setdefault(name, digest)
        if previous != digest:
            raise CheckpointError(f"checkpoint chunk {name} has conflicting signed digests")

    index = meta.get("chunk_index", {})
    if not isinstance(index, dict):
        raise CheckpointError("checkpoint chunk index metadata is invalid")
    for rows in index.values():
        if not isinstance(rows, list):
            raise CheckpointError("checkpoint chunk index metadata is invalid")
        for row in rows:
            if not isinstance(row, list) or len(row) < 4:
                raise CheckpointError("checkpoint chunk index metadata is invalid")
            add_chunk(row[0], row[3])

    file_chunks = meta.get("file_chunks", {})
    if not isinstance(file_chunks, dict):
        raise CheckpointError("checkpoint file chunk metadata is invalid")
    for rows in file_chunks.values():
        if not isinstance(rows, list):
            raise CheckpointError("checkpoint file chunk metadata is invalid")
        for row in rows:
            if not isinstance(row, list) or len(row) != 5:
                raise CheckpointError("checkpoint file chunk metadata is invalid")
            add_chunk(row[0], row[4])

    names = meta.get("chunks", [])
    if (not isinstance(names, list) or any(not isinstance(name, str) for name in names)
            or len(names) != len(set(names)) or set(names) != set(chunk_digests)):
        raise CheckpointError("checkpoint chunk references do not match signed metadata")
    for name in names:
        chunk = checkpoint.parent / CACHE_DIR / name
        if not chunk.is_file() or not _owned_private(chunk):
            raise CheckpointError(f"journal cache chunk {name} missing or not private")
        digest = hashlib.sha256()
        try:
            with chunk.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
        except OSError:
            raise CheckpointError(f"journal cache chunk {name} missing or unreadable") from None
        if digest.hexdigest() != chunk_digests[name]:
            raise CheckpointError(f"journal cache chunk {name} checksum mismatch")
    return meta, raw


def validate(checkpoint: Path) -> dict:
    """Authenticate plus compatibility with this process (Python, engine and host files)."""
    if _object_checkpoint(checkpoint):
        return _workspace_checkpoint_codec().validate_directory(checkpoint, compatible=True)
    meta, _raw = _validate(checkpoint)
    return meta


def _restore_modes(run_dir: Path, meta: dict) -> None:
    """Give each restored file its recorded mode; a file with none (a pooled blob) gains owner-write.

    Checkpoint copies are read-only object links, so a plain copy would restore read-only files that the
    engine must keep writing. Every restored file is writable unless it was recorded read-only.
    """
    recorded = meta.get("file_modes", {})
    if not isinstance(recorded, dict):
        raise CheckpointError("checkpoint file modes are invalid")
    for name in meta["entries"]:
        top = run_dir / name
        if not os.path.lexists(top) or top.is_symlink():
            continue
        if top.is_dir():
            paths = [Path(directory, file) for directory, _dirs, files in os.walk(top, followlinks=False)
                     for file in files]
        else:
            paths = [top]
        for path in paths:
            current_info = os.lstat(path)
            if not stat.S_ISREG(current_info.st_mode):
                continue
            current = stat.S_IMODE(current_info.st_mode)
            mode = recorded.get(path.relative_to(run_dir).as_posix())
            if mode is None:
                target_mode = current | stat.S_IWUSR
            elif type(mode) is int and 0 <= mode <= 0o7777:
                target_mode = mode
            else:
                raise CheckpointError("checkpoint file mode is invalid")
            if target_mode != current:
                os.chmod(path, target_mode)


def _restore_files(checkpoint: Path, run_dir: Path, meta: dict) -> None:
    # Refuse old or damaged snapshots before moving any new-attempt files aside. In particular,
    # a snapshot must never replace the pinned bundle or the live hook delivery records.
    for name in meta["entries"]:
        if name in SKIP_RUN_ENTRIES:
            raise CheckpointError(f"checkpoint includes live run entry {name!r}")
    discard = run_dir / "checkpoints" / f"setup-discard.{os.getpid()}"
    for path in list(run_dir.iterdir()):
        if path.name in SKIP_RUN_ENTRIES:
            continue
        discard.mkdir(parents=True, exist_ok=True)
        os.rename(path, discard / path.name)  # the new attempt's setup output; kept, never deleted
    for name in meta["entries"]:
        _clone(checkpoint / "run" / name, run_dir / name)
    manifest = checkpoint / MANIFEST_FILE
    pool_sources: dict[str, Path] = {}
    if manifest.is_file():
        for relative, files in json.loads(manifest.read_bytes()).items():
            store = checkpoint.parent / POOL_DIR / relative.replace("/", "__")
            live = run_dir / relative
            live.mkdir(mode=0o700, parents=True, exist_ok=True)
            for row in files:
                name = row[0]
                _clone(store / name, live / name)
                pool_sources[f"{relative}/{name}"] = store / name
    _restore_modes(run_dir, meta)
    for relative, rows in meta.get("file_chunks", {}).items():
        if (not relative.startswith("worker/measurement-runs/")
                or relative.count("/") != 3 or not relative.endswith("/events.jsonl")):
            raise CheckpointError(f"invalid audit journal path {relative!r}")
        destination = run_dir / relative
        if not destination.is_file() or destination.is_symlink():
            raise CheckpointError(f"audit journal placeholder missing: {relative}")
        with destination.open("wb") as output:
            for name, start, end, raw_sha, stored_sha in rows:
                if output.tell() != start:
                    raise CheckpointError(f"audit journal chunks are discontinuous: {relative}")
                chunk = checkpoint.parent / CACHE_DIR / name
                stored = hashlib.sha256()
                raw = hashlib.sha256()
                decoder = zlib.decompressobj()
                with chunk.open("rb") as stream:
                    for block in iter(lambda: stream.read(1 << 20), b""):
                        stored.update(block)
                        try:
                            decoded = decoder.decompress(block)
                        except zlib.error:
                            raise CheckpointError(
                                f"audit journal chunk checksum mismatch: {relative}") from None
                        output.write(decoded)
                        raw.update(decoded)
                decoded = decoder.flush()
                output.write(decoded)
                raw.update(decoded)
                if (not decoder.eof or decoder.unused_data or stored.hexdigest() != stored_sha
                        or raw.hexdigest() != raw_sha or output.tell() != end):
                    raise CheckpointError(f"audit journal chunk checksum mismatch: {relative}")
        shutil.copystat(checkpoint / "run" / relative, destination)
    # Last: the journal placeholders above are rewritten, and their copystat would reset the mtime.
    _restore_mtimes(checkpoint, run_dir, meta, pool_sources)


def _restore_mtimes(checkpoint: Path, run_dir: Path, meta: dict,
                    pool_sources: dict[str, Path] | None = None) -> None:
    """Give each restored file the mtime_ns it had when checkpointed (meta "file_mtime_ns").

    Only the private copy in ``run_dir`` is touched. A file that is not a single-link inode, or that
    shares an inode with its stored source (the checkpoint's run copy, or the blob pool file for pooled
    blobs), is replaced by a private copy before utime, so os.utime never reaches a shared object. A
    missing or malformed record is ignored and restore continues (old checkpoints keep their behavior).
    """
    recorded = meta.get("file_mtime_ns")
    if not isinstance(recorded, dict):
        return
    entries = set(meta.get("entries", ()))
    for key, mtime_ns in recorded.items():
        parts = key.split("/") if isinstance(key, str) else []
        if (type(mtime_ns) is not int or not -(1 << 63) < mtime_ns < (1 << 63)
                or not parts or parts[0] not in entries or any(part in ("", ".", "..") for part in parts)):
            _warn_once("file_mtime_ns", "checkpoint mtime record has an invalid entry; mtimes not restored")
            continue
        target = run_dir.joinpath(*parts)
        try:
            info = os.lstat(target)
            source = (pool_sources or {}).get(key, checkpoint / "run" / key)
            stored = os.lstat(source)
            if not stat.S_ISREG(info.st_mode):
                continue
            if info.st_nlink != 1 or os.path.samestat(info, stored):
                _replace_with_private_copy(target)
            os.utime(target, ns=(mtime_ns, mtime_ns), follow_symlinks=False)
        except OSError:
            _warn_once("file_mtime_restore", "could not restore a checkpointed file mtime; continuing")


def _replace_with_private_copy(path: Path) -> None:
    """Swap ``path`` for a fresh single-link copy with the same bytes, mode and times."""
    descriptor, temporary = tempfile.mkstemp(prefix=".mtime-", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        shutil.copy2(path, temporary, follow_symlinks=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


_ABSENT = object()


def restore(frame, checkpoint: Path, run_dir: Path) -> dict:
    if _object_checkpoint(checkpoint):
        return _workspace_checkpoint_codec().restore_checkpoint(run_dir, checkpoint, frame)
    t0 = time.perf_counter()
    meta, whole = _validate(checkpoint)
    t1 = time.perf_counter()
    _restore_files(checkpoint, run_dir, meta)
    t2 = time.perf_counter()
    cells, funcs = _frame_maps(frame)
    ctx = {"framecells": cells, "framefuncs": funcs}
    _SAVE.clear()
    _SAVE.update(load_root=checkpoint.parent, load_run_dir=run_dir,
                 load_index=meta.get("chunk_index", {}) or {})
    try:
        payload = _Unpickler(io.BytesIO(whole), ctx).load()
    finally:
        _SAVE.clear()
    _apply_restored_state(frame, payload)
    t3 = time.perf_counter()
    install_continuous_clock(meta["clocks"])
    meta["restore_phases"] = {"validate": round(t1 - t0, 3), "files": round(t2 - t1, 3),
                              "load_and_apply": round(t3 - t2, 3)}
    return meta


def _apply_restored_state(frame, payload: dict) -> None:
    for name, state in payload["modules"].items():
        module = sys.modules.get(name)
        if module is None:
            __import__(name)
            module = sys.modules[name]
        for key, value in state.items():
            setattr(module, key, value)
    for name, state in payload["classes"].items():
        module_name, qualname = name.split(":", 1)
        target: object = sys.modules[module_name]
        for part in qualname.split("."):
            target = getattr(target, part)
        for key, value in state.items():
            if vars(target).get(key, _ABSENT) is not value:
                setattr(target, key, value)
    for name, state in payload["patched"].items():
        module = sys.modules.get(name)
        if module is not None:
            for key, value in state.items():
                setattr(module, key, value)
    # Older payloads may contain these globals (including patched globals).
    # Also discard observations made by this process before loading the snapshot.
    for name in ("kernel_file_digest_memo", "engine.kernel_file_digest_memo",
                 "stateframe.engine.kernel_file_digest_memo"):
        module = sys.modules.get(name)
        if module is not None:
            module._MEMO = {}
            module._LAST_CLOCK_NS = None
    for name, saved in payload["participants"].items():
        live = _PARTICIPANTS.get(name)
        if live is None:
            raise CheckpointError(f"host did not re-register participant {name!r}")
        live.clear()
        live.update(saved)
    saved_vars = {id(var) for var, _ in payload["context"]}
    for var, _ in list(contextvars.copy_context().items()):
        if id(var) in saved_vars:
            continue
        # Set by this process's setup but unset at the checkpoint: back to "unset" as far as any
        # reader can tell (its default); a variable without a default cannot be unset.
        try:
            default = contextvars.Context().run(var.get)
        except LookupError:
            raise CheckpointError(f"ContextVar {var.name!r} set by setup has no default to reset to") from None
        var.set(default)
    for var, value in payload["context"]:
        var.set(value)
    random.setstate(payload["random"])
    proxy = frame.f_locals
    for name, value in payload["frame"].items():
        proxy[name] = value
    if isinstance(frame, types.FrameType) and isinstance(proxy, dict):
        # Before PEP 667, f_locals is a snapshot. Push the restored values into
        # the suspended frame's fast locals and closure cells before it resumes.
        import ctypes
        try:
            sync_locals = ctypes.pythonapi.PyFrame_LocalsToFast
        except AttributeError as error:
            raise CheckpointError("runtime cannot restore frame locals") from error
        sync_locals.argtypes = (ctypes.py_object, ctypes.c_int)
        sync_locals.restype = None
        sync_locals(frame, 0)

_REAL_CLOCKS: dict[str, object] = {}
_SHIFTED_CLOCKS: dict[str, object] = {}
_CLOCK_NAMES = ("monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns")


def _clock_name(obj) -> str | None:
    """A time clock function held by state (real or shifted): pickled by name, never by value."""
    for name in _CLOCK_NAMES:
        if obj is _REAL_CLOCKS.get(name) or obj is _SHIFTED_CLOCKS.get(name):
            return name
    return None


def _current_clock(name: str):
    return getattr(time, name)


def install_continuous_clock(saved: dict) -> None:
    """Make monotonic time continue from the checkpoint instant in this process.

    Restored state holds raw time.monotonic()/perf_counter() values (deadlines, elapsed budgets,
    host spans). Shifting this process's clocks by (saved - now) keeps every stored value
    meaningful: the time the run was stopped is invisible, as it is to a replayed run, and it
    works after a reboot (where raw monotonic values are meaningless). Wall time is untouched.
    Every module attribute bound to the real function (``from time import monotonic``) is
    rebound too, so the whole engine sees one clock.
    """
    if _REAL_CLOCKS:
        raise CheckpointError("continuous clock already installed in this process")
    real = {name: getattr(time, name) for name in ("monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns")}
    _REAL_CLOCKS.update(real)
    offset_s = float(saved["monotonic"]) - real["monotonic"]()
    offset_p = float(saved["perf_counter"]) - real["perf_counter"]()
    offset_ns, offset_pns = int(offset_s * 1e9), int(offset_p * 1e9)

    def monotonic():
        return real["monotonic"]() + offset_s

    def monotonic_ns():
        return real["monotonic_ns"]() + offset_ns

    def perf_counter():
        return real["perf_counter"]() + offset_p

    def perf_counter_ns():
        return real["perf_counter_ns"]() + offset_pns

    shifted = {"monotonic": monotonic, "monotonic_ns": monotonic_ns,
               "perf_counter": perf_counter, "perf_counter_ns": perf_counter_ns}
    _SHIFTED_CLOCKS.update(shifted)
    by_id = {id(fn): shifted[name] for name, fn in real.items()}
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            items = list(vars(module).items())
        except TypeError:
            continue
        for key, value in items:
            replacement = by_id.get(id(value))
            if replacement is not None:
                try:
                    setattr(module, key, replacement)
                except (AttributeError, TypeError):
                    pass


def refuse_restored_setup(run_dir: Path, error: Exception, *, source: str = "") -> None:
    """Refuse a checkpoint if process-bound setup cannot be rebound before the next request."""
    detail = f"{type(error).__name__}: {str(error)[:300]}"
    try:
        (run_dir / REFUSAL_FILE).write_text(json.dumps(
            {"error": "RestoreRefused", "detail": detail, "checkpoint": source}, indent=1))
    except OSError:
        pass
    _record(run_dir, event="checkpoint_restore_refused", detail=detail, source=source)
    raise RestoreRefused(detail) from error


def restore_point(frame, poc) -> bool:
    """Loop-entry hook: load the checkpoint named by STATEFRAME_CHECKPOINT_RESTORE, if any.

    Raises CheckpointError (before any model request) when it cannot restore faithfully;
    the product then falls back to replay resume.
    """
    code = frame.f_code
    nested = {c for c in code.co_consts if isinstance(c, types.CodeType)}
    _STATE["setup"] = {"code": code, "line": frame.f_lineno, "funcs": frozenset(
        name for name, value in list(frame.f_locals.items())
        if isinstance(value, types.FunctionType) and value.__code__ in nested)}
    run_dir = Path(poc).parent
    if (run_dir / "checkpoints").is_dir():
        swept = _sweep_stale(run_dir / "checkpoints")
        if swept:
            _record(run_dir, event="checkpoint_stale_temporaries_removed", count=swept)
    source = os.environ.pop(RESTORE_ENV, "")
    if not source:
        return False
    started = time.perf_counter()
    try:
        meta = restore(frame, Path(source), run_dir)
    except Exception as error:  # noqa: BLE001 - every failure is a refusal, never a partial state
        refuse_restored_setup(run_dir, error, source=source)
    from kernel_shell_tool import invalidate_active_environment_identity
    invalidate_active_environment_identity()
    _STATE["restored_step"] = meta["step"]
    _STATE["last_sends"] = meta["sends"]
    _record(run_dir, event="checkpoint_restored", step=meta["step"], sends=meta["sends"],
            phases=meta.get("restore_phases"),
            source=str(source),
            # measured on the real clock: restore installed the shifted one mid-way
            seconds=round(_REAL_CLOCKS.get("perf_counter", time.perf_counter)() - started, 3))
    return True


__all__ = ["CheckpointError", "RestoreRefused", "SCHEMA", "register_participant", "register_secret",
           "register_signing_key", "latest", "mode", "probe", "restore", "restore_point",
           "save", "save_point", "validate"]

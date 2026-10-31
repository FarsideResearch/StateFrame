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

Switch: STATEFRAME_CHECKPOINT = off (default) | save | probe.
Restore: STATEFRAME_CHECKPOINT_RESTORE=<checkpoint dir> (set by the product resume).
Stdlib only.
"""
from __future__ import annotations

import contextvars
import hashlib
import io
import json
import marshal
import os
import pickle
import random
import re
import shutil
import sys
import threading
import time
import types
import weakref
import zlib
from pathlib import Path

SCHEMA = "stateframe-engine-checkpoint/1"
MODE_ENV = "STATEFRAME_CHECKPOINT"
RESTORE_ENV = "STATEFRAME_CHECKPOINT_RESTORE"
EVERY_ENV = "STATEFRAME_CHECKPOINT_EVERY"
KEEP_ENV = "STATEFRAME_CHECKPOINT_KEEP"
STATE_FILE = "state.pkl"
META_FILE = "checkpoint.json"
# Run-dir entries never copied into a checkpoint (inputs, history, or the checkpoints themselves).
# engine.log is the runner's live log of this attempt: it is never snapshotted or replaced.
SKIP_RUN_ENTRIES = frozenset({"checkpoints", "attempts", "parent", ".engine.lock", ".run.lock",
                              "spec.json", "child-traceback.txt", "engine-result.json", "summary.json",
                              "engine.log", "checkpoint-refused.json"})


class CheckpointError(RuntimeError):
    """A snapshot could not be taken or restored faithfully."""


class RestoreRefused(BaseException):
    """The engine refuses to continue from a checkpoint (raised before any model request).

    A BaseException so no engine handler turns it into an ordinary run outcome; the host sees
    REFUSAL_FILE and falls back to replay resume.
    """


REFUSAL_FILE = "checkpoint-refused.json"


def mode() -> str:
    value = os.environ.get(MODE_ENV, "").strip().lower() or "off"  # empty = unset = off
    value = {"on": "save", "1": "save", "0": "off"}.get(value, value)  # on = save
    if value not in {"off", "save", "probe"}:
        raise ValueError(f"{MODE_ENV} must be off, on/save or probe")
    return value


# GCFREEZE: once the long-lived engine structures exist, freeze them out of
# generation-2 GC scans so every later collection stops re-walking the whole warm heap. Changes no object, value or byte.
# Lab switch SF_LAB_GC_FREEZE: off (PC3G behaviour) | once (default: one freeze per process; a resumed run re-freezes in its new pid)
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
# Each is a pure cache or a lazily rebuilt holder; dropping it can change only speed.
DROP_GLOBALS = {
    ("kernel_fl_checkpoint_capture", "_API"):
        "lazy import holder of checkpoint helper classes; re-created on first use",
    ("mc_repo_intel_vendor.lsp_resolve", "_PROJECTS"):
        "jedi project cache holding a live jedi subprocess; rebuilt on demand from the files",
}
# Host configuration the product child sets on the engine before work() runs. The new process's
# host sets them again (possibly to new values: a raised --max-turns), and KEY is a credential that
# must never be written to disk.
HOST_GLOBALS = frozenset(
    (module, name)
    for module in ("probes.react_repair", "react_repair")
    for name in ("KEY", "API", "MODEL", "MAX_STEPS", "POC", "OUT")
)
# Instance attributes that are caches keyed by id() of live objects: after a restore those ids
# name nothing (or, by reuse, the wrong object), so they are dropped and rebuilt.
DROP_ATTRS = {
    ("kernel_stateframe_runtime", "StateFrameRuntime"): frozenset({"_model_router_projection_cache"}),
}
_OWN_MODULE = __name__
_SECRETS: list[bytes] = []


_SIGNING_KEY: list[bytes] = []


def register_signing_key(key: bytes) -> None:
    """The host's per-install checkpoint key. Loading a pickle executes code, so a checkpoint is
    written only when it can be signed and loaded only when its signature verifies. The key never
    goes to the run directory, the environment, or a log; it is also registered as a secret."""
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("checkpoint signing key must be at least 32 bytes")
    _SIGNING_KEY[:] = [key]
    _SECRETS.append(key)
    _SECRETS.append(key.hex().encode())


def _signature(meta: dict, state_sha256: str) -> str:
    """HMAC-SHA256 over the canonical metadata (minus the signature) and the state digest; the
    metadata pins every journal chunk's sha256, so chunks are covered too."""
    import hmac
    if not _SIGNING_KEY:
        raise CheckpointError("no checkpoint signing key registered by the host")
    body = {key: value for key, value in meta.items() if key != "signature"}
    message = json.dumps(body, sort_keys=True, separators=(",", ":")).encode() + b"\0" + state_sha256.encode()
    return hmac.new(_SIGNING_KEY[0], message, hashlib.sha256).hexdigest()


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


def register_secret(value: str) -> None:
    """A value (an API key) that must never appear in a checkpoint; a save that contains it fails."""
    if isinstance(value, str) and len(value) >= 8:
        _SECRETS.append(value.encode("utf-8"))


def _module_state(name: str, module: types.ModuleType) -> dict[str, object]:
    out = {}
    if name == _OWN_MODULE:
        return out
    for key, value in list(vars(module).items()):
        if key.startswith("__") and key.endswith("__"):
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
    for key, value in list(vars(cls).items()):
        if key.startswith("__") and key.endswith("__") or key == "_abc_impl":
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
_SAVE: dict = {}  # one dump: {"root", "index" (prior chunk index), "index_out", "ids", "load_root", "tables"}


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
    ref = _chunk_list(key, pairs, guard=pairs[-1][0] if pairs else None, values=True)
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
        with open(journal, "rb") as stream:
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
        with open(journal, "rb") as stream:
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
        with open(journal, "rb") as stream:
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

    ``guard`` (dict mode: the last key) is recorded per chunk; when the item at the end of the
    covered prefix is no longer what was stored there, the lineage restarts from scratch.

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
    if values and covered and (len(chunks[-1]) < 5 or items[covered - 1][0] != chunks[-1][4]):
        chunks, covered = [], 0  # the dict's covered prefix changed: rewrite it
    if len(items) > covered:
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
            blob = zlib.compress(pickle.dumps(payload, protocol=5), 1)
        except Exception:  # noqa: BLE001 - an item that is not a plain value: keep the list inline
            return None
        name = f"{key}-{covered:07d}-{len(items):07d}-{os.getpid()}{CHUNK_SUFFIX}"
        folder = root / CACHE_DIR
        _private_dir(folder)
        temporary = folder / f".{name}.tmp"
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, folder / name)
        row = (name, covered, len(items), hashlib.sha256(blob).hexdigest())
        chunks.append(row + ((items[-1][0],) if values else (end_offset,) if end_offset is not None else ()))
    _SAVE.setdefault("index_out", {})[key] = [list(row) for row in chunks]
    ids = _SAVE.setdefault("ids", {})
    for position, item in enumerate(items):
        if values:  # a dict: its VALUES are referenced elsewhere (receipts -> same event objects)
            ids.setdefault(id(item[1]), (key, position, 1))
        else:
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
    return state


def _timeline_state(obj):
    """TimelineIndex: _events as an append-only list keyed by its session id."""
    reduced = obj.__reduce_ex__(5)
    state = dict(reduced[2]) if len(reduced) > 2 and isinstance(reduced[2], dict) else None
    if state is None:
        return NotImplemented
    ref = _chunk_list(_stable_key("timeline", state.get("session_id")), state.get("_events"))
    if ref is None:
        return NotImplemented
    state["_events"] = ref
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
    from kernel_continuation_journal import MEMBOUND_ENV, RECORD_CACHE_LIMIT
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
    obj.__dict__.update(state)
    obj._fd = None
    obj._pid = os.getpid()
    if was_open and not _STATE.get("probe_load"):
        value = obj.directory.lstat()
        fd = os.open(obj.directory / "owner.lock", os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not stat_module.S_ISREG(os.fstat(fd).st_mode):
                raise CheckpointError("restored workspace lease is not a regular file")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise
        obj._fd = fd
        obj._identity = (value.st_dev, value.st_ino)


# Process-bound engine classes: (module, qualname) -> state setter (obj, state) that re-acquires.
LEASE_TYPES = {
    ("kernel_continuation_journal", "ContinuationJournal"): _rebuild_journal,
    ("kernel_continuation_workspace", "ContinuationWorkspace"): _rebuild_workspace,
}


class _Unsupported(Exception):
    pass


class _Pickler(pickle.Pickler):
    def __init__(self, stream, ctx, protocol=5):
        super().__init__(stream, protocol=protocol)
        self.ctx = ctx

    def persistent_id(self, obj):
        ctx = self.ctx
        if isinstance(obj, types.ModuleType):
            return ("mod", obj.__name__)
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
        if owner_key == ("kernel_capability_projection", "CapabilityProjection"):
            return obj.__reduce_ex__(5)
        if owner_key[0] == "kernel_history_receipts" and owner_key[1] in {"ReceiptDict", "ReceiptList"}:
            # These mutable views have an explicit owner/index reduction. The
            # generic guarded-container path would serialize their weak state.
            return obj.__reduce_ex__(5)
        dropped = DROP_ATTRS.get(owner_key)
        if dropped is not None:
            return _runtime_state(obj, dropped)
        chunked = CHUNKED_TYPES.get(owner_key)
        if chunked is not None:
            result = chunked(obj)
            if result is not NotImplemented:
                return result
        rebuild = LEASE_TYPES.get((type(obj).__module__, type(obj).__qualname__))
        if rebuild is not None:
            state = _journal_state(obj) if rebuild is _rebuild_journal else _lease_state(obj)
            return (_new_instance, (type(obj),), state, None, None, rebuild)
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


class _StateWriter:
    """Forward pickle writes while retaining only the secret-boundary overlap."""

    __slots__ = ("stream", "digest", "count", "tail", "overlap")

    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.count = 0
        self.tail = b""
        self.overlap = max((len(secret) for secret in _SECRETS), default=0) - 1

    def write(self, chunk: bytes) -> int:
        window = self.tail + chunk
        if any(secret in window for secret in _SECRETS):
            raise CheckpointError("a registered secret would be written into the checkpoint")
        written = self.stream.write(chunk)
        if written != len(chunk):
            raise OSError("short checkpoint state write")
        self.digest.update(chunk)
        self.count += written
        self.tail = window[-self.overlap:] if self.overlap > 0 else b""
        return written


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
          "child": None, "child_tmp": None, "consecutive_failed": 0}
# Saving stops after this many failures in a row (a full disk would otherwise retry, and leave a
# partial copy, at every boundary for the rest of the run).
MAX_CONSECUTIVE_FAILURES = 3


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
# (0.41 s at 101 sends). They go to one pool per run instead: each save hard-links only the NEW
# blobs into checkpoints/blob-pool and records the name list; a restore links them back.
POOL_DIR = "blob-pool"
MANIFEST_FILE = "blobs.json"
# Pool store -> {name: size when first pooled}. Pooled files are write-once, so a save stats only the
# names it has not seen (one stat per file for the 8.6k-blob stores cost up to 0.6 s per save under
# load); a later size change is still caught by authenticate()'s size check against the pool.
_POOL_KNOWN: dict[str, dict[str, int]] = {}


def _size(path) -> int:
    return os.lstat(path).st_size


# Other write-once directories, one or more new files per model send: the product's wire/, requests/
# and checks/ records (written once each) and the engine's candidate captures (created with mode "x").
POOLED_DIRS = frozenset({"wire", "requests", "checks", "worker/candidates"})


def _is_blob_store(relative: str) -> bool:
    parts = relative.split("/")
    return len(parts) == 4 and parts[0] == "worker" and parts[1] == "measurement-runs" and parts[3] == "blobs"


def _is_pooled(relative: str) -> bool:
    return relative in POOLED_DIRS or _is_blob_store(relative)


def _snapshot_tree(source: Path, target: Path, relative: str, pool: Path, manifests: dict) -> None:
    """Clone ``source`` into ``target`` with as few clonefile calls as possible, pooling write-once dirs."""
    entries = list(os.scandir(source)) if _is_pooled(relative) and source.is_dir() else None
    if entries is not None and all(entry.is_file(follow_symlinks=False) for entry in entries):
        names = sorted(entry.name for entry in entries)
        target.mkdir()  # present (empty) in the copy; the restore links the pooled files back in
        store = pool / relative.replace("/", "__")
        known = _POOL_KNOWN.get(str(store))
        if known is None:  # first save of this process: adopt what an earlier process pooled
            store.mkdir(mode=0o700, parents=True, exist_ok=True)
            known = _POOL_KNOWN.setdefault(str(store), {name: _size(store / name) for name in os.listdir(store)})
        for name in names:
            if name not in known:
                size = _size(source / name)
                try:
                    os.link(source / name, store / name)
                except FileExistsError:
                    pass
                known[name] = size
        manifests[relative] = [[name, known[name]] for name in names]
        return
    if relative in ("worker", "worker/measurement-runs") or (
            relative.startswith("worker/measurement-runs/") and relative.count("/") == 2):
        target.mkdir()
        for entry in os.scandir(source):
            _snapshot_tree(Path(entry.path), target / entry.name, f"{relative}/{entry.name}", pool, manifests)
        return
    if not _clonefile(source, target):
        _clone(source, target)


def _copy_tree(source: Path, target: Path, previous: Path | None) -> dict[str, int]:
    """Copy ``source`` to ``target``; a file unchanged since the previous checkpoint is hard-linked.

    Checkpoint copies are never modified after they are written, so linking to one is safe; a file
    is unchanged when size and mtime_ns equal the previous copy (copies keep the source mtime).
    Big changed files are APFS-cloned (copy-on-write), small ones copied.
    """
    counts = {"linked": 0, "copied": 0, "cloned": 0}
    for directory, dirnames, filenames in os.walk(source):
        rel = os.path.relpath(directory, source)
        out_dir = target if rel == "." else target / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in list(dirnames):
            path = os.path.join(directory, name)
            if os.path.islink(path):
                os.symlink(os.readlink(path), out_dir / name)
                dirnames.remove(name)
        for name in filenames:
            path = os.path.join(directory, name)
            destination = out_dir / name
            info = os.lstat(path)
            if os.path.islink(path):
                os.symlink(os.readlink(path), destination)
                continue
            old = None if previous is None else (previous if rel == "." else previous / rel) / name
            if old is not None:
                try:
                    prior = os.lstat(old)
                except OSError:
                    prior = None
                if prior is not None and prior.st_size == info.st_size and prior.st_mtime_ns == info.st_mtime_ns:
                    os.link(old, destination)
                    counts["linked"] += 1
                    continue
            if info.st_size >= _BIG_FILE and sys.platform == "darwin":
                if _clonefile(Path(path), destination):
                    counts["cloned"] += 1
                    continue
            shutil.copy2(path, destination, follow_symlinks=False)
            counts["copied"] += 1
    return counts


def _sends(run_dir: Path) -> int:
    """Physical model sends so far: the host's wire counter when it registered one."""
    wire_state = _PARTICIPANTS.get("wire")
    if wire_state is not None and type(wire_state.get("n")) is int:
        return wire_state["n"]
    wire = run_dir / "wire"
    return len(list(wire.glob("*.request.json"))) if wire.is_dir() else 0


def _fsync_dir(path: Path) -> None:
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


def save(frame, step: int, run_dir: Path, *, background: bool = False) -> Path:
    """Write checkpoint ``<run>/checkpoints/<step>-<pid>`` atomically; returns its directory.

    The in-memory capture and the run-file snapshot happen here, synchronously, at the step
    boundary (the files are cloned copy-on-write). Pickling and writing can then run in a forked
    child (``background``): it pickles its own copy-on-write image of the memory captured at this
    instant while the engine continues, and makes the checkpoint visible (LATEST) only when complete.
    """
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
    # Immutable names: a checkpoint is never replaced, only pruned once newer ones are durable.
    final = root / f"{step:04d}-{os.getpid()}"
    tmp = root / f".{step:04d}-{os.getpid()}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    _private_dir(tmp)
    try:
        return _save_into(payload, ctx, step, run_dir, root, final, tmp, clocks, started, collected,
                          background)
    except BaseException:
        if _STATE.get("child_tmp") != tmp:  # not yet handed to a writer: nobody else will remove it
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def _save_into(payload, ctx, step, run_dir, root, final, tmp, clocks, started, collected, background):
    """The rest of save(): snapshot the run files into ``tmp`` and write (here or in the writer)."""
    (tmp / "run").mkdir()
    seq = _next_seq(root)  # ordering for pruning/LATEST (see _checkpoints), fixed before any fork
    _STATE["next_seq"] = seq + 1
    entries = sorted(p.name for p in run_dir.iterdir() if p.name not in SKIP_RUN_ENTRIES)
    previous = latest(run_dir)
    copied = {"linked": 0, "copied": 0, "cloned": 0}
    manifests: dict[str, list] = {}
    for name in entries:
        source, destination = run_dir / name, tmp / "run" / name
        if (name == "worker" or name in POOLED_DIRS) and source.is_dir() and not source.is_symlink() \
                and sys.platform == "darwin":
            _snapshot_tree(source, destination, name, root / POOL_DIR, manifests)
            copied["cloned"] += 1
            continue
        if _clonefile(source, destination):
            copied["cloned"] += 1
            continue
        prior = previous / "run" / name if previous is not None else None
        if source.is_dir() and not source.is_symlink():
            for key, value in _copy_tree(source, destination, prior).items():
                copied[key] += value
        else:
            _clone(source, destination)
    manifest_bytes = json.dumps(manifests, sort_keys=True, separators=(",", ":")).encode()
    with os.fdopen(os.open(tmp / MANIFEST_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(manifest_bytes)
    files_done = time.perf_counter()
    meta = {
        "schema": SCHEMA, "step": step, "sends": _sends(run_dir), "entries": entries,
        "python": list(sys.version_info[:3]), "engine_root": _engine_root(),
        "engine_files_sha256": engine_fingerprint(), "pid": os.getpid(),
        "created": time.time(), "seq": seq, "clocks": clocks, "files": copied,
        "blob_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "blob_count": sum(len(names) for names in manifests.values()),
        "seconds": {"collect": round(collected - started, 3), "files": round(files_done - collected, 3),
                    "capture": round(time.perf_counter() - started, 3)},
    }
    _STATE["phase_seconds"] = dict(meta["seconds"])
    prior_index = {}
    if previous is not None:
        try:
            prior_index = json.loads((previous / META_FILE).read_text()).get("chunk_index", {}) or {}
        except (OSError, ValueError):
            prior_index = {}

    def write() -> None:
        t0 = time.perf_counter()
        def dump(writer):
            _SAVE.clear()
            _SAVE.update(root=root, index=prior_index)
            try:
                dump_to(writer, payload, ctx)
                return dict(_SAVE.get("index_out", {}))
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
            stream.flush()
            os.fsync(stream.fileno())
        meta.update(state_bytes=writer.count, state_sha256=writer.digest.hexdigest(),
                    chunk_index=index,
                    chunks=sorted(row[0] for rows in index.values() for row in rows))
        meta["seconds"]["write"] = round(time.perf_counter() - t0, 3)
        meta["signature"] = _signature(meta, meta["state_sha256"])
        with os.fdopen(os.open(tmp / META_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            stream.write(json.dumps(meta, indent=1, sort_keys=True))
        _fsync_dir(tmp)
        if final.exists():
            raise CheckpointError(f"checkpoint {final.name} already exists")
        os.rename(tmp, final)
        latest_tmp = root / f".LATEST.{os.getpid()}"
        latest_tmp.write_text(final.name + "\n")
        os.replace(latest_tmp, root / "LATEST")
        _fsync_dir(root)
        for old in _checkpoints(root)[:-_keep()]:
            shutil.rmtree(old, ignore_errors=True)
        _prune_chunks(root)
        _prune_pool(root, run_dir)
        return meta

    try:
        if background:
            _fork_writer(run_dir, step, final, write, tmp)
        else:
            write()
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def _rss_mb() -> float | None:
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
        return round(peak / (1 << 20) if sys.platform == "darwin" else peak / 1024, 1)
    except (ImportError, OSError, ValueError):
        return None


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
    if os.waitstatus_to_exitcode(status) != 0:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
            _sweep_stale(tmp.parent)
        _SCANS.clear()  # the writer may have failed on a stale scan: rescan at the next boundary
        _note_failure(_STATE.get("child_run_dir"), "background writer failed")
    else:
        _STATE["consecutive_failed"] = 0
    return True


def _note_failure(run_dir, reason: str) -> None:
    """Count a failed save; after MAX_CONSECUTIVE_FAILURES in a row, stop saving (said once)."""
    _STATE["failed"] += 1
    _STATE["consecutive_failed"] = _STATE.get("consecutive_failed", 0) + 1
    if _STATE["consecutive_failed"] >= MAX_CONSECUTIVE_FAILURES and not _STATE["disabled"]:
        _STATE["disabled"] = f"{MAX_CONSECUTIVE_FAILURES} consecutive checkpoint failures ({reason})"
        if run_dir is not None:
            _record(Path(run_dir), event="checkpoint_disabled", reason=_STATE["disabled"])
        print(f"checkpoint: WARNING saving turned off for this run after {MAX_CONSECUTIVE_FAILURES} "
              f"failed saves in a row ({reason}); resume will replay", file=sys.stderr, flush=True)


_STALE = re.compile(r"\.(?:\d{4}-|LATEST\.)(\d+)(?:\.tmp)?$|\..*-(\d+)\.(?:z)?pkl\.tmp$")


def _sweep_stale(root: Path) -> int:
    """Remove temporaries left by checkpoint writers that no longer run; returns how many."""
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
                shutil.rmtree(entry.path, ignore_errors=True)
            else:
                try:
                    os.unlink(entry.path)
                except OSError:
                    continue
            removed += 1
    return removed


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _fork_writer(run_dir: Path, step: int, final: Path, write, tmp: Path) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    forked = time.perf_counter()
    pid = os.fork()
    if pid == 0:  # the writer: pickle this frozen image, write, and leave without any cleanup
        code = 1
        try:
            import signal
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.alarm(CHILD_TIMEOUT_S)
            started = time.perf_counter()
            meta = write()
            _record(run_dir, event="checkpoint_saved", step=step, dir=final.name, background=True,
                    seconds=round(time.perf_counter() - started, 3), sends=meta["sends"],
                    state_bytes=meta["state_bytes"], chunks=len(meta["chunks"]))
            code = 0
        except BaseException as error:  # noqa: BLE001 - recorded, then exit
            shutil.rmtree(tmp, ignore_errors=True)
            _record(run_dir, event="checkpoint_failed", step=step, background=True,
                    error=f"{type(error).__name__}: {str(error)[:300]}")
        finally:
            os._exit(code)
    _STATE["child"], _STATE["child_tmp"], _STATE["child_run_dir"] = pid, tmp, run_dir
    _STATE["fork_seconds"] = round(time.perf_counter() - forked, 3)


def _prune_chunks(root: Path) -> None:
    """Remove journal cache chunks no remaining checkpoint references (only those of this run)."""
    folder = root / CACHE_DIR
    if not folder.is_dir():
        return
    referenced: set[str] = set()
    for checkpoint in _checkpoints(root):
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


def engine_fingerprint() -> str:
    """Hash of every engine .py file (path + bytes): a checkpoint only loads into the same code."""
    global _FINGERPRINT
    if _FINGERPRINT is None:
        digest = hashlib.sha256()
        root = Path(_engine_root())
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        host = getattr(sys.modules.get(HOST_MODULE), "__file__", None)
        if isinstance(host, str):  # the product wrapper and its helpers sit beside the child script
            for path in sorted(Path(host).resolve().parent.glob("*.py")):
                digest.update(b"host:" + path.name.encode() + b"\0")
                digest.update(hashlib.sha256(path.read_bytes()).digest())
        _FINGERPRINT = digest.hexdigest()
    return _FINGERPRINT


def save_point(frame, step: int, poc) -> None:
    """Step-loop hook. Never raises: a failed save is recorded and the run continues.

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
        # One checkpoint per model send: a boundary with no new send since the last checkpoint is
        # recovered from that checkpoint plus the (empty) replayed tail, so saving it buys nothing.
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
        background = _async_enabled()
        if background and not _reap():
            _STATE["last_sends"] = previous_sends  # still writing the last one: try at the next boundary
            _record(run_dir, event="checkpoint_skipped", step=step, reason="previous writer still running")
            return
        started = time.perf_counter()
        # No cyclic-GC pass inside the pause: capture allocates many small objects, and on a 2 GB heap
        # one triggered gen-2 collection took seconds (pauses up to 2.8 s at 100 sends with fork <=25 ms).
        import gc
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            target = save(frame, step, run_dir, background=background)
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
    except Exception as error:  # noqa: BLE001 - fail open, never break the run
        if current == "save":
            _STATE["last_sends"] = previous_sends  # nothing was saved: the next boundary may retry
        _record(run_dir, event="checkpoint_failed", step=step,
                error=f"{type(error).__name__}: {str(error)[:300]}")
        _note_failure(run_dir, type(error).__name__)


# --------------------------------------------------------------------------- restore

def latest(run_dir: Path) -> Path | None:
    pointer = run_dir / "checkpoints" / "LATEST"
    try:
        name = pointer.read_text().strip()
    except OSError:
        return None
    target = run_dir / "checkpoints" / name
    return target if (target / META_FILE).is_file() else None


def authenticate(checkpoint: Path) -> dict:
    """Is this checkpoint provably ours? Ownership/permissions, checksum and HMAC signature,
    checked before any byte of the pickle is loaded."""
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
    if len(blob) != meta.get("state_bytes") or hashlib.sha256(blob).hexdigest() != meta.get("state_sha256"):
        raise CheckpointError("checkpoint state checksum mismatch")
    import hmac
    if not isinstance(meta.get("signature"), str) or not hmac.compare_digest(
            meta["signature"], _signature(meta, hashlib.sha256(blob).hexdigest())):
        raise CheckpointError("checkpoint signature does not verify")
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
                    actual = os.lstat(store / name).st_size
                except OSError:
                    raise CheckpointError(f"pool lacks {relative}/{name}") from None
                if actual != size:  # a pooled file must be write-once; this one changed after the save
                    raise CheckpointError(f"pooled file {relative}/{name} changed size since the checkpoint")
    return meta


def _prune_pool(root: Path, run_dir: Path) -> None:
    """Drop pooled files that no kept checkpoint lists and that the live run no longer has.

    A pooled file still present in the live directory is listed again by the next save, so it is
    kept (it is a hard link: it costs no extra space). One the run deleted is freed once no kept
    checkpoint names it, so the pool is bounded by the kept checkpoints plus the live run.
    """
    pool = root / POOL_DIR
    if not pool.is_dir():
        return
    referenced: dict[str, set[str]] = {}
    for checkpoint in _checkpoints(root):
        try:
            manifest = json.loads((checkpoint / MANIFEST_FILE).read_text())
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            return  # cannot prove a pooled file unused: keep everything
        for relative, files in manifest.items():
            referenced.setdefault(relative.replace("/", "__"), set()).update(name for name, _ in files)
    for store in pool.iterdir():
        if not store.is_dir():
            continue
        live = run_dir / store.name.replace("__", "/")
        keep = referenced.get(store.name, set())
        for entry in os.scandir(store):
            if entry.name not in keep and not os.path.lexists(live / entry.name):
                os.unlink(entry.path)


def validate(checkpoint: Path) -> dict:
    """authenticate() plus compatibility with this process (Python, engine and host files)."""
    meta = authenticate(checkpoint)
    if list(sys.version_info[:3]) != meta.get("python"):
        raise CheckpointError("checkpoint was written by another Python version")
    if meta.get("engine_files_sha256") != engine_fingerprint():
        raise CheckpointError("checkpoint was written by another engine build")
    for name in meta.get("entries", ()):
        if not (checkpoint / "run" / name).exists():
            raise CheckpointError(f"checkpoint run copy lacks {name}")
    for name in meta.get("chunks", ()):
        chunk = checkpoint.parent / CACHE_DIR / name
        if not chunk.is_file() or not _owned_private(chunk):
            raise CheckpointError(f"journal cache chunk {name} missing or not private")
    return meta


def _restore_files(checkpoint: Path, run_dir: Path, meta: dict) -> None:
    discard = run_dir / "checkpoints" / f"setup-discard.{os.getpid()}"
    for path in list(run_dir.iterdir()):
        if path.name in SKIP_RUN_ENTRIES:
            continue
        discard.mkdir(parents=True, exist_ok=True)
        os.rename(path, discard / path.name)  # the new attempt's setup output; kept, never deleted
    for name in meta["entries"]:
        _clone(checkpoint / "run" / name, run_dir / name)
    manifest = checkpoint / MANIFEST_FILE
    if manifest.is_file():
        for relative, files in json.loads(manifest.read_bytes()).items():
            store = checkpoint.parent / POOL_DIR / relative.replace("/", "__")
            live = run_dir / relative
            live.mkdir(mode=0o700, parents=True, exist_ok=True)
            for name, _size in files:  # write-once files: a hard link IS the content
                os.link(store / name, live / name)


_ABSENT = object()


def restore(frame, checkpoint: Path, run_dir: Path) -> dict:
    t0 = time.perf_counter()
    meta = validate(checkpoint)
    t1 = time.perf_counter()
    _restore_files(checkpoint, run_dir, meta)
    t2 = time.perf_counter()
    cells, funcs = _frame_maps(frame)
    ctx = {"framecells": cells, "framefuncs": funcs}
    _SAVE.clear()
    _SAVE.update(load_root=checkpoint.parent, load_index=meta.get("chunk_index", {}) or {})
    try:
        payload = _Unpickler(io.BytesIO((checkpoint / STATE_FILE).read_bytes()), ctx).load()
    finally:
        _SAVE.clear()
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
    t3 = time.perf_counter()
    install_continuous_clock(meta["clocks"])
    meta["restore_phases"] = {"validate": round(t1 - t0, 3), "files": round(t2 - t1, 3),
                              "load_and_apply": round(t3 - t2, 3)}
    return meta


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


def restore_point(frame, poc) -> None:
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
        return
    started = time.perf_counter()
    try:
        meta = restore(frame, Path(source), run_dir)
    except Exception as error:  # noqa: BLE001 - every failure is a refusal, never a partial state
        detail = f"{type(error).__name__}: {str(error)[:300]}"
        try:
            (run_dir / REFUSAL_FILE).write_text(json.dumps(
                {"error": "RestoreRefused", "detail": detail, "checkpoint": str(source)}, indent=1))
        except OSError:
            pass
        _record(run_dir, event="checkpoint_restore_refused", detail=detail, source=str(source))
        raise RestoreRefused(detail) from error
    _STATE["restored_step"] = meta["step"]
    _STATE["last_sends"] = meta["sends"]
    _record(run_dir, event="checkpoint_restored", step=meta["step"], sends=meta["sends"],
            phases=meta.get("restore_phases"),
            source=str(source),
            # measured on the real clock: restore installed the shifted one mid-way
            seconds=round(_REAL_CLOCKS.get("perf_counter", time.perf_counter)() - started, 3))


__all__ = ["CheckpointError", "RestoreRefused", "SCHEMA", "register_participant", "register_secret",
           "register_signing_key", "latest", "mode", "probe", "restore", "restore_point",
           "save", "save_point", "validate"]

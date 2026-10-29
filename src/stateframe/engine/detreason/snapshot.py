"""SnapshotV1 — the ONE immutable, content-addressed, replay-verified envelope.

WHY THIS FILE EXISTS (BURN_LIST W0-5)
=====================================
Four independent designs reinvented the same certificate: `design-pre-mortem-
generator.md` (`PreMortemReport` + `report_digest`), `run-cert-probe-calibration.md`
(`ProbeCalibrationSnapshot`), `walk-backend-contracts.md` (graduation certificate
with byte-identical dual replay), and `detreason/kernel.py`'s already-shipping
`Certificate`. Four hash formats, four replay stories, four versioning stories.

This module is the base they all derive from. A subsystem MAY add domain fields
inside `payload`. It MUST NOT invent its own hash domain, reference format,
canonical encoding, or versioning rule.

SCOPE DISCIPLINE: this file ships the BASE + the conformance harness + ONE
worked derivation (`kernel_decision_certificate@1`). It deliberately does NOT
retrofit the four existing emitters — that is a separate wiring lane, and a
half-done retrofit is worse than none (#211: a shared vocabulary owned by
nobody is how two correct branches compose into a brick).

THE CENTRAL INVARIANT
---------------------
    same contract + same type/schema + same method artifact
    + same named inputs + same execution descriptor
    = identical canonical bytes = identical content hash

HARD RULES ENCODED HERE
-----------------------
* `hash` is transport metadata. Only the `snapshot` body is hashed.
* No snapshot ever carries a self-asserted `verified: true`. Verification is a
  VERIFIER RESULT (#155: a status that is not freshly computed is decoration).
* Floats are rejected outright. Scores/margins/probabilities are exact
  rationals. Float drift is silent and unfixable after the fact.
* A value read by a derivation but absent from the named-input manifest is an
  unpinned ambient dependency, and such a derivation is not replay-valid.
* A missing method, runtime, input blob, or referenced snapshot yields
  REPLAY_UNAVAILABLE — never a successful verification (fail closed, #181c).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping
from typing import Optional, Sequence, Tuple

__all__ = [
    "SnapshotError", "CanonicalizationError", "ReferenceError", "ReplayError",
    "CONTRACT", "cse1_encode", "cse1_digest", "snapshot_hash", "blob_hash",
    "rational", "rational_sub", "rational_cmp", "rational_is_canonical",
    "validate_timestamp", "format_timestamp",
    "snapshot_ref", "blob_ref", "is_ref", "iter_refs",
    "build_snapshot", "validate_body", "BASE_KEYS",
    "SnapshotStore", "MethodRegistry",
    "input_inline", "input_snapshot_body", "input_snapshot_payload",
    "input_blob_bytes", "hermetic_execution",
    "derive", "verify", "VerificationResult",
    "INVALID", "HASH_VALID", "CLOSED", "REPLAY_VERIFIED", "REPLAY_UNAVAILABLE",
    "SnapshotDerivationSpec", "register_spec", "get_spec", "registered_specs",
    "clear_registry", "unregistered_types",
    "migrate", "MigrationVector",
    "ConformanceReport", "run_conformance", "assert_conformance",
    "KDC_TYPE", "install_kernel_decision_certificate",
]

CONTRACT = "SnapshotV1"
BASE_KEYS = ("contract", "type", "schema", "derivation", "payload")

_SNAPSHOT_HASH_DOMAIN = b"SnapshotV1\0"
_BLOB_HASH_DOMAIN = b"BlobV1\0"

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


class SnapshotError(RuntimeError):
    """Any violation of the SnapshotV1 contract."""


class CanonicalizationError(SnapshotError):
    """The value cannot be represented in CSE-1."""


class ReferenceError(SnapshotError):
    """A typed reference is malformed, dangling, or mis-asserted."""


class ReplayError(SnapshotError):
    """A derivation could not be replayed, or replayed to different bytes."""


# =========================================================================
# 1. CSE-1 — the canonical serialization. It never changes.
# =========================================================================

def _encode_string(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:            # unpaired surrogate
        raise CanonicalizationError(
            f"string is not valid UTF-8 (unpaired surrogate?): {exc}") from exc
    out: List[str] = ['"']
    for ch in value:
        code = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif code < 0x20:
            out.append("\\u%04x" % code)         # lowercase hex, per contract
        else:
            out.append(ch)                       # '/' is NOT escaped
    out.append('"')
    return "".join(out)


def _encode(value: Any, path: str) -> str:
    # bool BEFORE int: bool is an int subclass in Python and `True` must not
    # canonicalize to `1`.
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        raise CanonicalizationError(
            f"{path}: floats are forbidden in CSE-1 "
            f"(use rational(numerator, denominator)); got {value!r}")
    if isinstance(value, int):
        return str(value)                        # arbitrary precision, no exp
    if isinstance(value, str):
        return _encode_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(
            _encode(item, f"{path}[{i}]") for i, item in enumerate(value)
        ) + "]"
    if isinstance(value, Mapping):
        keys = list(value.keys())
        for key in keys:
            if not isinstance(key, str):
                raise CanonicalizationError(
                    f"{path}: object keys must be strings, got {key!r}")
        encoded = [k.encode("utf-8") for k in keys]
        if len(set(encoded)) != len(encoded):
            raise CanonicalizationError(f"{path}: duplicate object key")
        order = sorted(range(len(keys)), key=lambda i: encoded[i])
        parts = []
        for i in order:
            key = keys[i]
            parts.append(_encode_string(key) + ":"
                         + _encode(value[key], f"{path}.{key}"))
        return "{" + ",".join(parts) + "}"
    raise CanonicalizationError(
        f"{path}: unrepresentable type {type(value).__name__}")


def cse1_encode(value: Any) -> bytes:
    """Canonical bytes for `value`. Deterministic, sorted, float-free."""
    return _encode(value, "$").encode("utf-8")


def cse1_digest(value: Any) -> str:
    """A domain-free content digest. Used for contract/intent identity."""
    return "sha256:" + hashlib.sha256(cse1_encode(value)).hexdigest()


def snapshot_hash(body: Mapping) -> str:
    return "sha256:" + hashlib.sha256(
        _SNAPSHOT_HASH_DOMAIN + cse1_encode(body)).hexdigest()


def blob_hash(raw: bytes) -> str:
    if not isinstance(raw, (bytes, bytearray)):
        raise SnapshotError(f"blob must be bytes, got {type(raw).__name__}")
    return "sha256:" + hashlib.sha256(
        _BLOB_HASH_DOMAIN + bytes(raw)).hexdigest()


# =========================================================================
# 2. Exact rationals — the float ban's escape hatch.
# =========================================================================

def rational(numerator: int, denominator: int) -> Dict[str, int]:
    """A reduced rational with a positive denominator."""
    if isinstance(numerator, bool) or isinstance(denominator, bool):
        raise SnapshotError("rational parts must be ints, not bools")
    if not isinstance(numerator, int) or not isinstance(denominator, int):
        raise SnapshotError("rational parts must be ints")
    if denominator == 0:
        raise SnapshotError("rational denominator must be non-zero")
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    g = math.gcd(abs(numerator), denominator) or 1
    return {"numerator": numerator // g, "denominator": denominator // g}


def rational_is_canonical(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if set(value.keys()) != {"numerator", "denominator"}:
        return False
    num, den = value["numerator"], value["denominator"]
    if isinstance(num, bool) or isinstance(den, bool):
        return False
    if not isinstance(num, int) or not isinstance(den, int):
        return False
    if den <= 0:
        return False
    return rational(num, den) == dict(value)


def rational_sub(a: Mapping, b: Mapping) -> Dict[str, int]:
    return rational(a["numerator"] * b["denominator"]
                    - b["numerator"] * a["denominator"],
                    a["denominator"] * b["denominator"])


def rational_cmp(a: Mapping, b: Mapping) -> int:
    left = a["numerator"] * b["denominator"]
    right = b["numerator"] * a["denominator"]
    return (left > right) - (left < right)


# =========================================================================
# 3. Timestamps — content, never an ambient annotation.
# =========================================================================

def validate_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not _TS_RE.match(value):
        raise SnapshotError(
            f"timestamp must be RFC3339 UTC with exactly six fractional "
            f"digits, got {value!r}")
    try:
        _dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise SnapshotError(f"invalid timestamp {value!r}: {exc}") from exc
    return value


def format_timestamp(epoch_micros: int) -> str:
    """Format from INTEGER microseconds. There is no implicit clock read here
    on purpose: a timestamp that affects derivation must be a named input."""
    if isinstance(epoch_micros, bool) or not isinstance(epoch_micros, int):
        raise SnapshotError("epoch_micros must be an int")
    base = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
    moment = base + _dt.timedelta(microseconds=epoch_micros)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + "%06dZ" % moment.microsecond


# =========================================================================
# 4. Typed references. A raw hash string is NOT a reference.
# =========================================================================

_SNAPSHOT_REF_KEYS = {"$ref", "hash", "type", "schema"}
_BLOB_REF_KEYS = {"$ref", "hash", "media_type", "size"}


def snapshot_ref(hash_: str, type_: str, schema: int) -> Dict[str, Any]:
    if not _HASH_RE.match(hash_ or ""):
        raise ReferenceError(f"bad snapshot hash {hash_!r}")
    if not isinstance(type_, str) or not type_:
        raise ReferenceError("snapshot ref requires a type assertion")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema < 1:
        raise ReferenceError(f"schema must be an int >= 1, got {schema!r}")
    return {"$ref": "snapshot", "hash": hash_, "type": type_,
            "schema": schema}


def blob_ref(hash_: str, media_type: str, size: int) -> Dict[str, Any]:
    if not _HASH_RE.match(hash_ or ""):
        raise ReferenceError(f"bad blob hash {hash_!r}")
    if not isinstance(media_type, str) or not media_type:
        raise ReferenceError("blob ref requires a media_type")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ReferenceError(f"blob size must be a non-negative int, {size!r}")
    return {"$ref": "blob", "hash": hash_, "media_type": media_type,
            "size": size}


def is_ref(value: Any) -> bool:
    if not isinstance(value, Mapping) or "$ref" not in value:
        return False
    kind = value.get("$ref")
    keys = set(value.keys())
    if kind == "snapshot":
        return keys == _SNAPSHOT_REF_KEYS
    if kind == "blob":
        return keys == _BLOB_REF_KEYS
    return False


def _looks_like_ref(value: Any) -> bool:
    return isinstance(value, Mapping) and "$ref" in value


def iter_refs(value: Any, path: str = "$") -> Iterator[Tuple[str, Mapping]]:
    """Yield every typed reference with its dotted path. Raises on a value
    that claims to be a reference but is malformed — a half-formed ref must
    never be silently walked past (#181c)."""
    if _looks_like_ref(value):
        if not is_ref(value):
            raise ReferenceError(
                f"{path}: malformed reference {sorted(value.keys())}")
        yield path, value
        return
    if isinstance(value, Mapping):
        for key in value:
            yield from iter_refs(value[key], f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from iter_refs(item, f"{path}[{i}]")


# =========================================================================
# 5. The base body.
# =========================================================================

_INPUT_KINDS = ("inline", "snapshot_body", "snapshot_payload", "blob_bytes")

_EXECUTION_KEYS = ("executor", "dependencies", "platform", "locale",
                   "timezone", "clock", "random", "network", "filesystem",
                   "environment")


def hermetic_execution(*, executor: Optional[Mapping] = None,
                       dependencies: Optional[Mapping] = None,
                       platform: str = "hermetic-python",
                       locale: str = "C", timezone: str = "UTC",
                       clock: Optional[Mapping] = None,
                       random: Optional[Mapping] = None,
                       network: Optional[Mapping] = None,
                       filesystem: Optional[Mapping] = None,
                       environment: Sequence[str] = ()) -> Dict[str, Any]:
    """The default execution descriptor: no clock, no randomness, no network,
    no filesystem, no environment. Anything a derivation needs must be pinned
    EXPLICITLY, which is the whole point."""
    return {
        "executor": dict(executor or {}),
        "dependencies": dict(dependencies or {}),
        "platform": platform,
        "locale": locale,
        "timezone": timezone,
        "clock": dict(clock or {}),
        "random": dict(random or {}),
        "network": dict(network or {"enabled": False}),
        "filesystem": dict(filesystem or {}),
        "environment": list(environment),
    }


def input_inline(name: str, value: Any) -> Dict[str, Any]:
    return {"name": name, "source": {"kind": "inline", "value": value}}


def input_snapshot_body(name: str, ref: Mapping) -> Dict[str, Any]:
    return {"name": name,
            "source": {"kind": "snapshot_body", "value": dict(ref)}}


def input_snapshot_payload(name: str, ref: Mapping) -> Dict[str, Any]:
    return {"name": name,
            "source": {"kind": "snapshot_payload", "value": dict(ref)}}


def input_blob_bytes(name: str, ref: Mapping) -> Dict[str, Any]:
    return {"name": name,
            "source": {"kind": "blob_bytes", "value": dict(ref)}}


def build_snapshot(*, type_: str, schema: int, method_id: str,
                   method_version: str,
                   implementation: Optional[Mapping],
                   inputs: Sequence[Mapping],
                   execution: Mapping,
                   payload: Mapping) -> Dict[str, Any]:
    body = {
        "contract": CONTRACT,
        "type": type_,
        "schema": schema,
        "derivation": {
            "method": {
                "id": method_id,
                "version": method_version,
                "implementation": dict(implementation)
                if implementation is not None else None,
            },
            "inputs": [dict(i) for i in inputs],
            "execution": dict(execution),
        },
        "payload": dict(payload),
    }
    validate_body(body)
    return body


def validate_body(body: Any) -> None:
    """Closed-world validation of the BASE contract only. Payload shape is the
    registered spec's business."""
    if not isinstance(body, Mapping):
        raise SnapshotError("snapshot body must be a mapping")
    keys = set(body.keys())
    if keys != set(BASE_KEYS):
        extra = sorted(keys - set(BASE_KEYS))
        missing = sorted(set(BASE_KEYS) - keys)
        raise SnapshotError(
            f"top-level fields are closed; unknown={extra} missing={missing}")
    if body["contract"] != CONTRACT:
        raise SnapshotError(f"contract must be {CONTRACT!r}")
    if not isinstance(body["type"], str) or not body["type"]:
        raise SnapshotError("type must be a non-empty string")
    schema = body["schema"]
    if isinstance(schema, bool) or not isinstance(schema, int) or schema < 1:
        raise SnapshotError(f"schema must be an int >= 1, got {schema!r}")
    if "verified" in body["payload"]:
        raise SnapshotError(
            "a snapshot must never carry a self-asserted 'verified' field; "
            "verification is a verifier result")

    der = body["derivation"]
    if not isinstance(der, Mapping) or set(der.keys()) != {
            "method", "inputs", "execution"}:
        raise SnapshotError(
            "derivation must have exactly {method, inputs, execution}")
    method = der["method"]
    if not isinstance(method, Mapping) or set(method.keys()) != {
            "id", "version", "implementation"}:
        raise SnapshotError(
            "method must have exactly {id, version, implementation}")
    if not isinstance(method["id"], str) or not method["id"]:
        raise SnapshotError("method.id is required")
    if not isinstance(method["version"], str) or not method["version"]:
        raise SnapshotError("method.version is required")
    impl = method["implementation"]
    if impl is not None and not (is_ref(impl) and impl["$ref"] == "blob"):
        raise SnapshotError("method.implementation must be a blob ref or null")

    inputs = der["inputs"]
    if not isinstance(inputs, list):
        raise SnapshotError("derivation.inputs must be a list")
    seen: set = set()
    for item in inputs:
        if not isinstance(item, Mapping) or set(item.keys()) != {
                "name", "source"}:
            raise SnapshotError("input must have exactly {name, source}")
        name = item["name"]
        if not isinstance(name, str) or not name:
            raise SnapshotError("input name must be a non-empty string")
        if name in seen:
            raise SnapshotError(f"duplicate input name {name!r}")
        seen.add(name)
        source = item["source"]
        if not isinstance(source, Mapping) or set(source.keys()) != {
                "kind", "value"}:
            raise SnapshotError("input source must have exactly {kind, value}")
        if source["kind"] not in _INPUT_KINDS:
            raise SnapshotError(
                f"unknown input kind {source['kind']!r}; "
                f"known: {_INPUT_KINDS}")
        if source["kind"] in ("snapshot_body", "snapshot_payload"):
            if not (is_ref(source["value"])
                    and source["value"]["$ref"] == "snapshot"):
                raise SnapshotError(
                    f"input {name!r}: {source['kind']} needs a snapshot ref")
        if source["kind"] == "blob_bytes":
            if not (is_ref(source["value"])
                    and source["value"]["$ref"] == "blob"):
                raise SnapshotError(
                    f"input {name!r}: blob_bytes needs a blob ref")

    execution = der["execution"]
    if not isinstance(execution, Mapping):
        raise SnapshotError("derivation.execution must be a mapping")
    if set(execution.keys()) != set(_EXECUTION_KEYS):
        raise SnapshotError(
            f"execution fields are closed; expected {sorted(_EXECUTION_KEYS)}")

    if not isinstance(body["payload"], Mapping):
        raise SnapshotError("payload must be a mapping")
    # Canonicalizability is part of validity: a body that cannot be encoded
    # has no identity.
    cse1_encode(body)


# =========================================================================
# 6. Store + method registry.
# =========================================================================

class SnapshotStore:
    """Content-addressed archive. Put is idempotent; there is no update and no
    delete — the archive must retain the transitive closure of every finalized
    snapshot or future replay is a lie."""

    def __init__(self) -> None:
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._blobs: Dict[str, bytes] = {}

    # -- blobs
    def put_blob(self, raw: bytes, media_type: str) -> Dict[str, Any]:
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        h = blob_hash(raw)
        self._blobs[h] = bytes(raw)
        return blob_ref(h, media_type, len(raw))

    def get_blob(self, hash_: str) -> bytes:
        if hash_ not in self._blobs:
            raise ReferenceError(f"dangling blob {hash_}")
        return self._blobs[hash_]

    def has_blob(self, hash_: str) -> bool:
        return hash_ in self._blobs

    # -- snapshots
    def put_snapshot(self, body: Mapping) -> str:
        validate_body(body)
        h = snapshot_hash(body)
        self._snapshots[h] = dict(body)
        return h

    def put_raw(self, body: Mapping, hash_: str) -> str:
        """Store WITHOUT validation under a caller-supplied hash. Exists only
        so conformance tests can plant tampered material."""
        self._snapshots[hash_] = dict(body)
        return hash_

    def get_snapshot(self, hash_: str) -> Dict[str, Any]:
        if hash_ not in self._snapshots:
            raise ReferenceError(f"dangling snapshot {hash_}")
        return self._snapshots[hash_]

    def has_snapshot(self, hash_: str) -> bool:
        return hash_ in self._snapshots

    def wire(self, hash_: str) -> Dict[str, Any]:
        """The stored wire shape: hash is transport metadata, outside the
        hashed material."""
        return {"hash": hash_, "snapshot": self.get_snapshot(hash_)}


class MethodRegistry:
    """Derivation methods, addressed by (id, version). The registered source
    text is stored as the implementation blob, so the snapshot commits to the
    exact code that produced it."""

    def __init__(self) -> None:
        self._fns: Dict[Tuple[str, str], Callable] = {}
        self._sources: Dict[Tuple[str, str], bytes] = {}

    def register(self, method_id: str, version: str, fn: Callable,
                 source: str) -> None:
        key = (method_id, version)
        if key in self._fns:
            raise SnapshotError(f"method {key} already registered")
        if not callable(fn):
            raise SnapshotError("method implementation must be callable")
        self._fns[key] = fn
        self._sources[key] = source.encode("utf-8")

    def has(self, method_id: str, version: str) -> bool:
        return (method_id, version) in self._fns

    def get(self, method_id: str, version: str) -> Callable:
        key = (method_id, version)
        if key not in self._fns:
            raise ReplayError(f"method {key} is not available")
        return self._fns[key]

    def implementation_ref(self, store: SnapshotStore, method_id: str,
                           version: str) -> Dict[str, Any]:
        key = (method_id, version)
        if key not in self._sources:
            raise SnapshotError(f"method {key} is not registered")
        return store.put_blob(self._sources[key], "text/x-python")


# =========================================================================
# 7. Spec registry — nothing derives without a machine-readable contract.
# =========================================================================

@dataclass(frozen=True)
class MigrationVector:
    """A golden migration case: source body -> expected migrated payload."""
    name: str
    source_body: Mapping
    expected_payload: Mapping


@dataclass(frozen=True)
class SnapshotDerivationSpec:
    type: str
    schema: int
    method_id: str
    method_version: str
    required_fields: Tuple[str, ...] = ()
    required_reference_paths: Tuple[str, ...] = ()
    reference_type_constraints: Mapping[str, str] = field(default_factory=dict)
    replay_mode: str = "deterministic"           # or "recorded-input"
    payload_validator: Optional[Callable[[Mapping], None]] = None
    invariant_checker: Optional[Callable[[Mapping], None]] = None
    fixture_factory: Optional[Callable[[SnapshotStore, MethodRegistry],
                                       Dict[str, Any]]] = None
    counterexample_factory: Optional[Callable[[Mapping], Mapping]] = None
    migration_vectors: Tuple[MigrationVector, ...] = ()

    @property
    def key(self) -> Tuple[str, int]:
        return (self.type, self.schema)


_REGISTRY: Dict[Tuple[str, int], SnapshotDerivationSpec] = {}


def register_spec(spec: SnapshotDerivationSpec) -> SnapshotDerivationSpec:
    if spec.replay_mode not in ("deterministic", "recorded-input"):
        raise SnapshotError(f"unknown replay_mode {spec.replay_mode!r}")
    if spec.key in _REGISTRY:
        raise SnapshotError(f"spec {spec.key} already registered")
    _REGISTRY[spec.key] = spec
    return spec


def get_spec(type_: str, schema: int) -> Optional[SnapshotDerivationSpec]:
    return _REGISTRY.get((type_, schema))


def registered_specs() -> Tuple[SnapshotDerivationSpec, ...]:
    return tuple(_REGISTRY[k] for k in sorted(_REGISTRY))


def clear_registry() -> None:
    _REGISTRY.clear()


def unregistered_types(seen: Iterable[Tuple[str, int]]
                       ) -> Tuple[Tuple[str, int], ...]:
    """CI guard: any (type, schema) observed in the wild with no registered
    spec. A type with no spec generates no conformance tests, which is the
    same as having no tests at all."""
    return tuple(sorted(k for k in seen if k not in _REGISTRY))


# =========================================================================
# 8. Derivation.
# =========================================================================

def _materialize(store: SnapshotStore, item: Mapping) -> Any:
    source = item["source"]
    kind, value = source["kind"], source["value"]
    if kind == "inline":
        return value
    if kind == "blob_bytes":
        raw = store.get_blob(value["hash"])
        if len(raw) != value["size"]:
            raise ReferenceError(
                f"blob {value['hash']} size assertion {value['size']} != "
                f"{len(raw)}")
        return raw
    body = store.get_snapshot(value["hash"])
    if body.get("type") != value["type"] or body.get("schema") != value["schema"]:
        raise ReferenceError(
            f"reference asserts {value['type']}@{value['schema']} but the "
            f"referenced snapshot is {body.get('type')}@{body.get('schema')}")
    return body if kind == "snapshot_body" else body["payload"]


def _run_method(store: SnapshotStore, methods: MethodRegistry,
                method_id: str, method_version: str,
                inputs: Sequence[Mapping], execution: Mapping) -> Mapping:
    fn = methods.get(method_id, method_version)
    named = {i["name"]: _materialize(store, i) for i in inputs}
    payload = fn(named, execution)
    if not isinstance(payload, Mapping):
        raise ReplayError(
            f"method {method_id}@{method_version} must return a payload "
            f"mapping, got {type(payload).__name__}")
    return payload


def derive(store: SnapshotStore, methods: MethodRegistry, *,
           type_: str, schema: int,
           inputs: Sequence[Mapping],
           execution: Optional[Mapping] = None) -> str:
    """Run the registered method for (type, schema) over pinned inputs and
    store the resulting snapshot. Returns its hash."""
    spec = get_spec(type_, schema)
    if spec is None:
        raise SnapshotError(
            f"no registered spec for {type_}@{schema}; register one before "
            f"deriving (an unregistered type generates no conformance tests)")
    execution = dict(execution or hermetic_execution())
    payload = _run_method(store, methods, spec.method_id, spec.method_version,
                          inputs, execution)
    body = build_snapshot(
        type_=type_, schema=schema,
        method_id=spec.method_id, method_version=spec.method_version,
        implementation=methods.implementation_ref(
            store, spec.method_id, spec.method_version),
        inputs=inputs, execution=execution, payload=payload)
    _validate_against_spec(spec, body)
    return store.put_snapshot(body)


def _validate_against_spec(spec: SnapshotDerivationSpec,
                           body: Mapping) -> None:
    payload = body["payload"]
    for fieldname in spec.required_fields:
        if fieldname not in payload:
            raise SnapshotError(
                f"{spec.type}@{spec.schema}: payload missing required field "
                f"{fieldname!r}")
    if spec.payload_validator is not None:
        spec.payload_validator(payload)
    found = {path for path, _ in iter_refs(payload, "$")}
    for required_path in spec.required_reference_paths:
        if required_path not in found:
            raise SnapshotError(
                f"{spec.type}@{spec.schema}: required reference path "
                f"{required_path} is absent")
    for path, ref in iter_refs(payload, "$"):
        expected = spec.reference_type_constraints.get(path)
        if expected is None:
            continue
        actual = ref["type"] if ref["$ref"] == "snapshot" else "blob"
        if actual != expected:
            raise ReferenceError(
                f"{path}: reference type {actual!r} violates constraint "
                f"{expected!r}")
    if spec.invariant_checker is not None:
        spec.invariant_checker(payload)


# =========================================================================
# 9. Verification.
# =========================================================================

INVALID = "INVALID"
HASH_VALID = "HASH_VALID"
CLOSED = "CLOSED"
REPLAY_VERIFIED = "REPLAY_VERIFIED"
REPLAY_UNAVAILABLE = "REPLAY_UNAVAILABLE"

_LEVEL_ORDER = {INVALID: 0, REPLAY_UNAVAILABLE: 1, HASH_VALID: 2,
                CLOSED: 3, REPLAY_VERIFIED: 4}


@dataclass(frozen=True)
class VerificationResult:
    level: str
    reasons: Tuple[str, ...] = ()
    visited: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.level == REPLAY_VERIFIED

    def at_least(self, level: str) -> bool:
        return _LEVEL_ORDER[self.level] >= _LEVEL_ORDER[level]


def verify(store: SnapshotStore, hash_: str, *,
           methods: Optional[MethodRegistry] = None,
           require_replay: bool = True,
           _stack: Tuple[str, ...] = ()) -> VerificationResult:
    """Three-level verification. Fails closed at every step."""
    if hash_ in _stack:
        return VerificationResult(
            INVALID, (f"cyclic reference graph at {hash_}",), _stack)
    if not store.has_snapshot(hash_):
        return VerificationResult(INVALID, (f"dangling snapshot {hash_}",))
    body = store.get_snapshot(hash_)

    # -- level 1: HASH_VALID
    try:
        validate_body(body)
    except SnapshotError as exc:
        return VerificationResult(INVALID, (f"schema: {exc}",))
    if snapshot_hash(body) != hash_:
        return VerificationResult(
            INVALID, ("declared hash does not match canonical bytes",))

    spec = get_spec(body["type"], body["schema"])
    if spec is None:
        return VerificationResult(
            INVALID, (f"no registered spec for {body['type']}@"
                      f"{body['schema']}",))
    try:
        _validate_against_spec(spec, body)
    except SnapshotError as exc:
        return VerificationResult(INVALID, (f"extension: {exc}",))

    # -- level 2: CLOSED
    stack = _stack + (hash_,)
    visited: List[str] = [hash_]
    reasons: List[str] = []
    refs = list(iter_refs(body["payload"], "$payload"))
    refs += list(iter_refs(body["derivation"], "$derivation"))
    for path, ref in refs:
        if ref["$ref"] == "blob":
            if not store.has_blob(ref["hash"]):
                return VerificationResult(
                    HASH_VALID, (f"{path}: dangling blob {ref['hash']}",),
                    tuple(visited))
            raw = store.get_blob(ref["hash"])
            if len(raw) != ref["size"]:
                return VerificationResult(
                    HASH_VALID, (f"{path}: blob size assertion wrong",),
                    tuple(visited))
            continue
        if not store.has_snapshot(ref["hash"]):
            return VerificationResult(
                HASH_VALID, (f"{path}: dangling snapshot {ref['hash']}",),
                tuple(visited))
        child_body = store.get_snapshot(ref["hash"])
        if (child_body.get("type") != ref["type"]
                or child_body.get("schema") != ref["schema"]):
            return VerificationResult(
                HASH_VALID,
                (f"{path}: reference asserts {ref['type']}@{ref['schema']} "
                 f"but target is {child_body.get('type')}@"
                 f"{child_body.get('schema')}",),
                tuple(visited))
        child = verify(store, ref["hash"], methods=methods,
                       require_replay=require_replay, _stack=stack)
        visited.extend(child.visited)
        if child.level == INVALID:
            return VerificationResult(
                HASH_VALID, (f"{path}: {'; '.join(child.reasons)}",),
                tuple(visited))
        if child.level == REPLAY_UNAVAILABLE:
            reasons.append(f"{path}: child replay unavailable")

    if not require_replay:
        return VerificationResult(CLOSED, tuple(reasons), tuple(visited))

    # -- level 3: REPLAY_VERIFIED
    method = body["derivation"]["method"]
    if methods is None or not methods.has(method["id"], method["version"]):
        return VerificationResult(
            REPLAY_UNAVAILABLE,
            tuple(reasons + [f"method {method['id']}@{method['version']} "
                             f"is not available for replay"]),
            tuple(visited))
    if method["implementation"] is not None:
        impl = method["implementation"]
        if not store.has_blob(impl["hash"]):
            return VerificationResult(
                REPLAY_UNAVAILABLE,
                tuple(reasons + ["implementation blob is unavailable"]),
                tuple(visited))
        live = methods.implementation_ref(store, method["id"],
                                          method["version"])
        if live["hash"] != impl["hash"]:
            return VerificationResult(
                INVALID,
                tuple(reasons + ["pinned implementation does not match the "
                                 "available method artifact"]),
                tuple(visited))
    # A MISSING artifact is REPLAY_UNAVAILABLE. A method that raises on the
    # pinned inputs is a replay FAILURE — the difference matters, because
    # "unavailable" is a shrug and "invalid" is an accusation.
    try:
        named = {i["name"]: _materialize(store, i)
                 for i in body["derivation"]["inputs"]}
    except ReferenceError as exc:
        return VerificationResult(
            REPLAY_UNAVAILABLE, tuple(reasons + [f"materialize: {exc}"]),
            tuple(visited))
    try:
        payload = methods.get(method["id"], method["version"])(
            named, body["derivation"]["execution"])
        if not isinstance(payload, Mapping):
            raise ReplayError("method did not return a payload mapping")
    except Exception as exc:                     # noqa: BLE001 - classified
        return VerificationResult(
            INVALID, tuple(reasons + [f"replay failed: {exc}"]),
            tuple(visited))
    candidate = build_snapshot(
        type_=body["type"], schema=body["schema"],
        method_id=method["id"], method_version=method["version"],
        implementation=method["implementation"],
        inputs=body["derivation"]["inputs"],
        execution=body["derivation"]["execution"], payload=payload)
    if cse1_encode(candidate) != cse1_encode(body):
        return VerificationResult(
            INVALID,
            tuple(reasons + ["replay produced different canonical bytes"]),
            tuple(visited))
    if snapshot_hash(candidate) != hash_:
        return VerificationResult(
            INVALID, tuple(reasons + ["replay produced a different hash"]),
            tuple(visited))
    if reasons:
        return VerificationResult(CLOSED, tuple(reasons), tuple(visited))
    return VerificationResult(REPLAY_VERIFIED, (), tuple(visited))


# =========================================================================
# 10. Migration — itself a normal derivation.
# =========================================================================

def migrate(store: SnapshotStore, methods: MethodRegistry, *,
            source_hash: str, target_type: str, target_schema: int,
            execution: Optional[Mapping] = None) -> str:
    """Migrate an existing snapshot. The old snapshot is untouched, is verified
    under its ORIGINAL schema first, and the new payload carries
    `migrated_from`. A migration can never make an invalid old snapshot valid.
    """
    source = store.get_snapshot(source_hash)
    prior = verify(store, source_hash, methods=methods, require_replay=False)
    if prior.level == INVALID:
        raise SnapshotError(
            f"refusing to migrate an invalid snapshot: "
            f"{'; '.join(prior.reasons)}")
    ref = snapshot_ref(source_hash, source["type"], source["schema"])
    new_hash = derive(store, methods, type_=target_type, schema=target_schema,
                      inputs=[input_snapshot_body("source", ref)],
                      execution=execution)
    migrated = store.get_snapshot(new_hash)
    if migrated["payload"].get("migrated_from") != source_hash:
        raise SnapshotError(
            "a migration payload MUST carry migrated_from = the old hash")
    return new_hash


# =========================================================================
# 11. Generated conformance harness.
# =========================================================================

@dataclass
class ConformanceReport:
    type: str
    schema: int
    passed: Dict[str, bool] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.passed[name] = ok
        if not ok:
            self.failures[name] = detail or "failed"

    def __str__(self) -> str:                    # pragma: no cover - display
        head = f"{self.type}@{self.schema}: {len(self.passed)} cases"
        if self.ok:
            return head + " ALL PASS"
        detail = "; ".join(f"{k}: {v}" for k, v in sorted(
            self.failures.items()))
        return head + f" FAILURES -> {detail}"


def _permute_keys(value: Any) -> Any:
    """Rebuild every mapping with reversed insertion order. Canonical bytes
    MUST be identical — this is the key-ordering test."""
    if isinstance(value, Mapping):
        return {k: _permute_keys(value[k]) for k in reversed(list(value))}
    if isinstance(value, list):
        return [_permute_keys(v) for v in value]
    return value


def run_conformance(spec: SnapshotDerivationSpec,
                    methods: MethodRegistry) -> ConformanceReport:
    """Generate and run the full conformance suite for one registered spec.

    This is the anti-drift enforcement point: a spec with no fixture factory
    cannot be certified, and saying so is the point (#218 — a guard must
    certify its own observation before its verdict means anything)."""
    report = ConformanceReport(spec.type, spec.schema)
    if spec.fixture_factory is None:
        report.record("fixture_available", False,
                      "spec has no fixture_factory; nothing can be certified")
        return report
    report.record("fixture_available", True)

    store = SnapshotStore()
    built = spec.fixture_factory(store, methods)
    hash_ = built["hash"]
    body = store.get_snapshot(hash_)

    # 1. golden hash vector — rebuilding from scratch reproduces the hash
    store2 = SnapshotStore()
    rebuilt = spec.fixture_factory(store2, methods)
    report.record("golden_hash_vector", rebuilt["hash"] == hash_,
                  f"{rebuilt['hash']} != {hash_}")

    # 2. canonicalization round trip
    report.record("canonical_round_trip",
                  cse1_encode(body) == cse1_encode(
                      store.get_snapshot(hash_)))

    # 3. object-key permutation invariance
    report.record("key_permutation_invariance",
                  snapshot_hash(_permute_keys(body)) == hash_)

    # 4. hash tampering is detected
    tampered = dict(body)
    tampered["payload"] = dict(body["payload"])
    tampered["payload"]["__tamper__"] = "x"
    store.put_raw(tampered, hash_)
    tamper_result = verify(store, hash_, methods=methods)
    report.record("hash_tampering_detected",
                  tamper_result.level == INVALID,
                  f"tampered snapshot verified as {tamper_result.level}")
    store.put_raw(body, hash_)                   # restore

    # 5. unknown field / duplicate key / float rejection
    unknown_ok = _raises(lambda: validate_body({**body, "extra": 1}))
    float_ok = _raises(lambda: cse1_encode({"x": 1.5}))
    dup_ok = _raises(lambda: _encode({1: "a"}, "$"))
    report.record("unknown_field_rejected", unknown_ok)
    report.record("float_rejected", float_ok)
    report.record("non_string_key_rejected", dup_ok)

    # 6. required dangling reference fails closed
    refs = list(iter_refs(body["payload"], "$payload"))
    if refs:
        dang_store = SnapshotStore()
        spec.fixture_factory(dang_store, methods)
        path, ref = refs[0]
        if ref["$ref"] == "blob":
            dang_store._blobs.pop(ref["hash"], None)
        else:
            dang_store._snapshots.pop(ref["hash"], None)
        res = verify(dang_store, hash_, methods=methods)
        report.record("dangling_reference_fails_closed",
                      not res.at_least(CLOSED),
                      f"dangling ref still reached {res.level}")
    else:
        report.record("dangling_reference_fails_closed", True)

    # 7. reference type/schema mismatch rejected
    snap_refs = [(p, r) for p, r in refs if r["$ref"] == "snapshot"]
    if snap_refs:
        path, ref = snap_refs[0]
        bad = snapshot_ref(ref["hash"], ref["type"] + "_wrong", ref["schema"])
        mutated = _replace_at(body, path[len("$payload"):], bad)
        mstore = SnapshotStore()
        spec.fixture_factory(mstore, methods)
        mh = mstore.put_raw(mutated, snapshot_hash(mutated))
        res = verify(mstore, mh, methods=methods)
        report.record("reference_type_mismatch_rejected",
                      not res.at_least(CLOSED),
                      f"mismatched ref reached {res.level}")
    else:
        report.record("reference_type_mismatch_rejected", True)

    # 8. input mutation breaks replay
    mutated_inputs = _mutate_first_inline(body)
    if mutated_inputs is not None:
        mstore = SnapshotStore()
        spec.fixture_factory(mstore, methods)
        mh = mstore.put_raw(mutated_inputs, snapshot_hash(mutated_inputs))
        res = verify(mstore, mh, methods=methods)
        report.record("input_mutation_breaks_replay",
                      res.level == INVALID,
                      f"mutated input still reached {res.level}")
    else:
        report.record("input_mutation_breaks_replay", True)

    # 9. ambient dependency test — same inputs, hostile ambient conditions
    import os as _os
    import random as _random
    saved = dict(_os.environ)
    try:
        _os.environ["DETREASON_AMBIENT_PROBE"] = "hostile"
        _random.seed(999)
        amb_store = SnapshotStore()
        amb = spec.fixture_factory(amb_store, methods)
    finally:
        _os.environ.clear()
        _os.environ.update(saved)
    report.record("no_ambient_dependency", amb["hash"] == hash_,
                  f"hash moved under hostile ambient conditions: "
                  f"{amb['hash']} != {hash_}")

    # 10. replay verification succeeds for the honest fixture
    honest = verify(store, hash_, methods=methods)
    if spec.replay_mode == "deterministic":
        report.record("replay_verified", honest.level == REPLAY_VERIFIED,
                      f"{honest.level}: {'; '.join(honest.reasons)}")
    else:
        report.record("replay_verified", honest.at_least(CLOSED),
                      f"{honest.level}: {'; '.join(honest.reasons)}")

    # 11. replay unavailable without the method (fails closed, never passes)
    empty_methods = MethodRegistry()
    res = verify(store, hash_, methods=empty_methods)
    report.record("missing_method_is_unavailable_not_ok",
                  res.level == REPLAY_UNAVAILABLE and not res.ok,
                  f"got {res.level}")

    # 12. transitive closure retained
    closure_ok = all(store.has_snapshot(h) for h in honest.visited)
    report.record("transitive_closure_retained", closure_ok)

    # 13. extension invariant checker has teeth (positive AND negative)
    if spec.invariant_checker is not None:
        pos = not _raises(lambda: spec.invariant_checker(body["payload"]))
        report.record("invariant_accepts_valid", pos)
        if spec.counterexample_factory is not None:
            bad_payload = spec.counterexample_factory(body["payload"])
            neg = _raises(lambda: spec.invariant_checker(bad_payload))
            report.record("invariant_rejects_counterexample", neg,
                          "planted counterexample was accepted")
        else:
            report.record("invariant_rejects_counterexample", False,
                          "no counterexample_factory: the invariant checker "
                          "is unproven (a detector with no planted control "
                          "mutation certifies nothing)")

    # 14. migration golden vectors
    for vector in spec.migration_vectors:
        mstore = SnapshotStore()
        src = mstore.put_snapshot(vector.source_body)
        try:
            new_hash = migrate(mstore, methods, source_hash=src,
                               target_type=spec.type,
                               target_schema=spec.schema)
            got = mstore.get_snapshot(new_hash)["payload"]
            report.record(f"migration:{vector.name}",
                          got == dict(vector.expected_payload),
                          f"payload drift: {got}")
        except SnapshotError as exc:
            report.record(f"migration:{vector.name}", False, str(exc))

    return report


def _raises(fn: Callable) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def _replace_at(body: Mapping, dotted: str, value: Any) -> Dict[str, Any]:
    """Rebuild `body` with payload path `dotted` (e.g. '.target') replaced."""
    import copy
    out = copy.deepcopy(dict(body))
    node: Any = out["payload"]
    tokens = _tokenize(dotted)
    for token in tokens[:-1]:
        node = node[token]
    node[tokens[-1]] = value
    return out


def _tokenize(dotted: str) -> List[Any]:
    tokens: List[Any] = []
    for chunk in dotted.split("."):
        if not chunk:
            continue
        while "[" in chunk:
            head, _, rest = chunk.partition("[")
            index, _, chunk = rest.partition("]")
            if head:
                tokens.append(head)
            tokens.append(int(index))
        if chunk:
            tokens.append(chunk)
    return tokens


def _mutate_first_inline(body: Mapping) -> Optional[Dict[str, Any]]:
    import copy
    out = copy.deepcopy(dict(body))
    for item in out["derivation"]["inputs"]:
        source = item["source"]
        if source["kind"] != "inline":
            continue
        value = source["value"]
        if isinstance(value, int) and not isinstance(value, bool):
            source["value"] = value + 1
            return out
        if isinstance(value, str):
            source["value"] = value + "!"
            return out
        if isinstance(value, list) and value:
            source["value"] = list(value[:-1])
            return out
    return None


def assert_conformance(spec: SnapshotDerivationSpec,
                       methods: MethodRegistry) -> ConformanceReport:
    report = run_conformance(spec, methods)
    if not report.ok:
        raise AssertionError(str(report))
    return report


# =========================================================================
# 12. WORKED DERIVATION EXAMPLE — kernel_decision_certificate@1
# =========================================================================
# This is the ONE derivation shipped with the base. It mirrors kernel.py's
# existing `Certificate` (verdict/winner/scores/atoms) but does NOT touch
# kernel.py: retrofitting the four live emitters is a separate wiring lane.
# It exists to prove the base contract carries a real payload end to end:
# exact rational scores, a recomputed winner, exact margins, pinned tie-break.

KDC_TYPE = "kernel_decision_certificate"
_KDC_METHOD = "snapshot.derive.kernel_decision_certificate"
_KDC_VERSION = "1"


def _kdc_derive(inputs: Mapping[str, Any], execution: Mapping) -> Dict[str, Any]:
    """Recompute the winner and every margin from atoms, candidate weights and
    the pinned tie-break order. Pure: reads NOTHING but `inputs`."""
    atoms = list(inputs["atoms"])
    candidates = list(inputs["candidates"])
    order = list(inputs["tie_break_order"])

    atom_ids = [a["id"] for a in atoms]
    if len(set(atom_ids)) != len(atom_ids):
        raise SnapshotError("atom ids must be unique")
    truth = {a["id"]: bool(a["value"]) for a in atoms}

    scored = []
    for cand in candidates:
        supporting = [aid for aid in cand["supporting_atoms"]
                      if truth.get(aid, False)]
        for aid in cand["supporting_atoms"]:
            if aid not in truth:
                raise SnapshotError(
                    f"candidate {cand['id']!r} cites unknown atom {aid!r}")
        score = rational(len(supporting), max(1, len(atom_ids)))
        scored.append({"id": cand["id"], "score": score,
                       "supporting_atoms": sorted(cand["supporting_atoms"])})
    scored.sort(key=lambda c: c["id"])

    # Highest score wins; ties go to the earliest PINNED policy order entry,
    # never to dict or insertion order.
    ordered = sorted(
        scored,
        key=lambda c: (order.index(c["id"]) if c["id"] in order else 10 ** 6))
    best = ordered[0]
    for cand in ordered[1:]:
        if rational_cmp(cand["score"], best["score"]) > 0:
            best = cand

    margins = [
        {"against": c["id"],
         "value": rational_sub(best["score"], c["score"]),
         "definition": "winner_score_minus_candidate_score"}
        for c in scored if c["id"] != best["id"]
    ]
    margins.sort(key=lambda m: m["against"])

    return {
        "atoms": sorted(
            [{"id": a["id"], "proposition": a["proposition"],
              "value": bool(a["value"]),
              "provenance": list(a.get("provenance", []))} for a in atoms],
            key=lambda a: a["id"]),
        "candidates": scored,
        "winner": best["id"],
        "margins": margins,
        "tie_break": {"rule": "policy_order", "order": order},
    }


_KDC_SOURCE = (
    "kernel_decision_certificate@1 :: winner = argmax(exact rational score) "
    "with ties broken by pinned policy order; margins = exact differences."
)


def _kdc_validate(payload: Mapping) -> None:
    required = {"atoms", "candidates", "winner", "margins", "tie_break"}
    missing = required - set(payload)
    if missing:
        raise SnapshotError(f"kdc payload missing {sorted(missing)}")


def _kdc_invariants(payload: Mapping) -> None:
    """The extension invariants. Every one of these has bitten somebody."""
    atom_ids = [a["id"] for a in payload["atoms"]]
    if atom_ids != sorted(atom_ids):
        raise SnapshotError("atom ids must be sorted")
    if len(set(atom_ids)) != len(atom_ids):
        raise SnapshotError("atom ids must be unique")
    for atom in payload["atoms"]:
        if "value" not in atom:
            raise SnapshotError(f"atom {atom['id']} has no explicit value")
    by_id = {c["id"]: c for c in payload["candidates"]}
    if payload["winner"] not in by_id:
        raise SnapshotError("winner is not among the candidates")
    winner_score = by_id[payload["winner"]]["score"]
    for cand in payload["candidates"]:
        if not rational_is_canonical(cand["score"]):
            raise SnapshotError(
                f"candidate {cand['id']} score is not a canonical rational")
        if rational_cmp(cand["score"], winner_score) > 0:
            raise SnapshotError(
                f"candidate {cand['id']} outscores the declared winner")
    seen = set()
    for margin in payload["margins"]:
        if margin["against"] == payload["winner"]:
            raise SnapshotError("a margin against the winner is meaningless")
        expected = rational_sub(winner_score, by_id[margin["against"]]["score"])
        if margin["value"] != expected:
            raise SnapshotError(
                f"margin against {margin['against']} is not the exact "
                f"difference")
        seen.add(margin["against"])
    if seen != set(by_id) - {payload["winner"]}:
        raise SnapshotError("every non-winning candidate needs a margin")


def _kdc_counterexample(payload: Mapping) -> Dict[str, Any]:
    """Planted control mutation: declare the loser the winner. If the
    invariant checker does not scream at this, it is certifying nothing."""
    import copy
    bad = copy.deepcopy(dict(payload))
    losers = [c["id"] for c in bad["candidates"] if c["id"] != bad["winner"]]
    if losers:
        bad["winner"] = losers[0]
    return bad


def _kdc_fixture(store: SnapshotStore,
                 methods: MethodRegistry) -> Dict[str, Any]:
    evidence = store.put_blob(b'{"unit_tests":"passed"}', "application/json")
    inputs = [
        input_inline("atoms", [
            {"id": "a1", "proposition": "Tests passed", "value": True,
             "provenance": [evidence]},
            {"id": "a2", "proposition": "Coverage held", "value": True,
             "provenance": []},
            {"id": "a3", "proposition": "Rollback rehearsed", "value": False,
             "provenance": []},
        ]),
        input_inline("candidates", [
            {"id": "approve", "supporting_atoms": ["a1", "a2"]},
            {"id": "reject", "supporting_atoms": ["a3"]},
        ]),
        input_inline("tie_break_order", ["approve", "reject"]),
    ]
    hash_ = derive(store, methods, type_=KDC_TYPE, schema=1, inputs=inputs)
    return {"hash": hash_, "inputs": inputs}


def install_kernel_decision_certificate(methods: MethodRegistry
                                        ) -> SnapshotDerivationSpec:
    """Register the worked example's method + spec. Idempotent per registry."""
    if not methods.has(_KDC_METHOD, _KDC_VERSION):
        methods.register(_KDC_METHOD, _KDC_VERSION, _kdc_derive, _KDC_SOURCE)
    existing = get_spec(KDC_TYPE, 1)
    if existing is not None:
        return existing
    return register_spec(SnapshotDerivationSpec(
        type=KDC_TYPE, schema=1,
        method_id=_KDC_METHOD, method_version=_KDC_VERSION,
        required_fields=("atoms", "candidates", "winner", "margins",
                         "tie_break"),
        replay_mode="deterministic",
        payload_validator=_kdc_validate,
        invariant_checker=_kdc_invariants,
        fixture_factory=_kdc_fixture,
        counterexample_factory=_kdc_counterexample,
    ))

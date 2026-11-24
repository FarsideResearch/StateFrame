"""Keyed stable identifiers for local telemetry occurrences."""
from __future__ import annotations

import hashlib
import hmac

from .types import Namespace, RunRef


_DOMAIN = b"sf-otel/1"
_MAX_PART_BYTES = 256
_U64_MAX = (1 << 64) - 1


def derive(key: bytes, tag: str, parts: tuple[bytes, ...], size: int) -> bytes:
    """Derive a bounded identifier with unambiguous length-prefixed HMAC input."""
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("invalid identity key")
    if type(tag) is not str or not tag:
        raise ValueError("invalid identity tag")
    try:
        encoded_tag = tag.encode("ascii", "strict")
    except UnicodeError as error:
        raise ValueError("invalid identity tag") from error
    if len(encoded_tag) > _MAX_PART_BYTES:
        raise ValueError("invalid identity tag")
    if type(parts) is not tuple or any(type(part) is not bytes for part in parts):
        raise ValueError("invalid identity parts")
    if any(len(part) > _MAX_PART_BYTES for part in parts):
        raise ValueError("invalid identity parts")
    if type(size) is not int or not 1 <= size <= hashlib.sha256().digest_size:
        raise ValueError("invalid identity size")

    base = (_DOMAIN, encoded_tag, *parts)
    counter = 0
    while counter <= _U64_MAX:
        encoded_parts = base if counter == 0 else (*base, counter.to_bytes(8, "big"))
        mac = hmac.new(key, digestmod=hashlib.sha256)
        for part in encoded_parts:
            mac.update(len(part).to_bytes(4, "big"))
            mac.update(part)
        result = mac.digest()[:size]
        if any(result):
            return result
        counter += 1
    raise ValueError("identity derivation exhausted")


def namespace_bytes(space: Namespace) -> bytes:
    unit = space.unit or b""
    return (space.run.value + space.attempt.to_bytes(8, "big") + len(unit).to_bytes(4, "big")
            + unit + (space.incarnation or 0).to_bytes(8, "big"))


def from_seed(key: bytes, seed: bytes) -> tuple[RunRef, bytes]:
    """Return the stable run reference and logical trace ID for a run seed."""
    if type(seed) is not bytes or len(seed) != 16:
        raise ValueError("invalid run seed")
    return RunRef(derive(key, "run", (seed,), 16)), derive(key, "trace", (seed,), 16)


__all__ = ["derive", "from_seed", "namespace_bytes"]

from __future__ import annotations

import hashlib
import hmac

import pytest

from stateframe.otel.identity import derive, from_seed
from stateframe.otel.types import RunRef


def test_hmac_derivation_matches_pinned_wire_independent_vector():
    key = bytes(range(32))
    seed = bytes.fromhex("00112233445566778899aabbccddeeff")
    assert derive(key, "trace", (seed,), 16) == bytes.fromhex(
        "8801677ff6e161a9bad32553c3541c0c"
    )


def test_identity_domain_and_tag_separate_equal_seed_material():
    key = bytes(range(32))
    seed = bytes.fromhex("00112233445566778899aabbccddeeff")
    run_ref, trace_id = from_seed(key, seed)
    assert isinstance(run_ref, RunRef)
    assert len(run_ref.value) == 16 and any(run_ref.value)
    assert len(trace_id) == 16 and any(trace_id)
    assert run_ref.value != trace_id
    assert derive(key, "run", (seed,), 16) == run_ref.value
    assert derive(key, "other", (seed,), 16) not in (run_ref.value, trace_id)


def test_identity_is_stable_for_same_domain_and_changes_with_key_or_seed():
    key = bytes(range(32))
    other_key = bytes(reversed(range(32)))
    seed = bytes.fromhex("00112233445566778899aabbccddeeff")
    other_seed = bytes.fromhex("10112233445566778899aabbccddeeff")

    original = from_seed(key, seed)
    assert from_seed(key, seed) == original
    assert from_seed(other_key, seed) != original
    assert from_seed(key, other_seed) != original


@pytest.mark.parametrize(
    ("key", "tag", "parts", "size"),
    [
        (b"short", "trace", (b"seed",), 16),
        (bytes(32), "", (b"seed",), 16),
        (bytes(32), "trace", ("seed",), 16),
        (bytes(32), "trace", (b"seed",), 33),
        (bytes(32), "trace", (b"x" * 257,), 16),
    ],
)
def test_invalid_identity_inputs_fail_closed(key, tag, parts, size):
    with pytest.raises((TypeError, ValueError)):
        derive(key, tag, parts, size)


def test_reference_hmac_encoding_is_length_prefixed_and_unambiguous():
    key = bytes(range(32))

    def reference(tag: str, parts: tuple[bytes, ...]) -> bytes:
        framed = (b"sf-otel/1", tag.encode("ascii"), *parts)
        payload = b"".join(len(part).to_bytes(4, "big") + part for part in framed)
        return hmac.new(key, payload, hashlib.sha256).digest()[:16]

    assert derive(key, "x", (b"ab", b"c"), 16) == reference("x", (b"ab", b"c"))
    assert derive(key, "x", (b"a", b"bc"), 16) == reference("x", (b"a", b"bc"))
    assert derive(key, "x", (b"ab", b"c"), 16) != derive(key, "x", (b"a", b"bc"), 16)

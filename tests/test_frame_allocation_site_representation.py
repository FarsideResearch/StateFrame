"""Unit site identity ignores only the inline-vs-reference byte representation (FU-070)."""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"))

from kernel_frame_allocation_validation import site_authority_identity

BYTES = "VALUE_1 = 1\nVALUE_2 = 2\n"
DIGEST = hashlib.sha256(BYTES.encode()).hexdigest()
BASE = {"path": "a.py", "line": 1, "lines": {"start": 1, "end": 2, "total": 9},
        "revision": "sha256:r", "content_sha256": DIGEST, "read_this_request": True}
INLINE = {**BASE, "current_bytes": BYTES}
REF = {**BASE, "current_bytes": None,
       "source_ref": {"path": "a.py", "lines": {"start": 1, "end": 2}, "content_sha256": DIGEST}}


def test_inline_and_verified_reference_forms_share_one_identity() -> None:
    assert site_authority_identity(INLINE) == site_authority_identity(REF)


def test_identity_stays_strict_on_every_other_field() -> None:
    for key, value in (("path", "b.py"), ("line", 2), ("revision", "sha256:x"),
                       ("read_this_request", False), ("lines", {"start": 1, "end": 3, "total": 9})):
        other = copy.deepcopy(REF)
        other[key] = value
        assert site_authority_identity(INLINE) != site_authority_identity(other), key


def test_unverified_bytes_or_reference_are_not_normalized() -> None:
    forged = {**INLINE, "current_bytes": "tampered\n"}
    assert site_authority_identity(forged) != site_authority_identity(REF)
    bad_ref = copy.deepcopy(REF)
    bad_ref["source_ref"]["content_sha256"] = "0" * 64
    assert site_authority_identity(INLINE) != site_authority_identity(bad_ref)


def test_site_without_a_current_bytes_key_does_not_raise() -> None:
    bare = {k: v for k, v in REF.items() if k != "current_bytes"}
    assert site_authority_identity(bare) == site_authority_identity(REF)

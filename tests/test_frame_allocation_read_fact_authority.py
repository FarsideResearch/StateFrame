"""Unit authority comparison: read_facts delivery metadata is wire-owned (FU-070)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"))

from kernel_frame_allocation_validation import READ_FACT_DELIVERY_KEYS, _authority_view

# Shape captured from the failing subframe child: the wire row carried delivery
# metadata, the host-derived unit projection did not.
WIRE = {"read_facts": [{
    "path": "owned_source.py", "current_revision": "sha256:a", "read_revision": None,
    "status": "UNVERIFIABLE", "unchanged_since_read": False,
    "body_delivery": "not_shown", "delivery_notice": "[not shown] owned_source.py",
    "journal_workspace_revisions": ["0"], "receipt_count": 1,
    "receipts": [{"receipt_id": "r", "workspace_revision": "0"}],
    "revision_scope": "unavailable",
}], "open_sites": []}
HOST = {"read_facts": [{
    "path": "owned_source.py", "current_revision": "sha256:a", "read_revision": None,
    "status": "UNVERIFIABLE", "unchanged_since_read": False, "body_delivery": "present",
}], "open_sites": []}


def test_delivery_metadata_alone_does_not_break_authority() -> None:
    assert _authority_view(WIRE) == _authority_view(HOST)


def test_authority_fields_stay_strict() -> None:
    for key, value in (("status", "UNCHANGED"), ("current_revision", "sha256:b"),
                       ("path", "other.py"), ("unchanged_since_read", True)):
        host = copy.deepcopy(HOST)
        host["read_facts"][0][key] = value
        assert _authority_view(WIRE) != _authority_view(host), key


def test_row_count_and_other_frame_keys_stay_strict() -> None:
    host = copy.deepcopy(HOST)
    host["read_facts"].append(dict(host["read_facts"][0], path="b.py"))
    assert _authority_view(WIRE) != _authority_view(host)
    other = copy.deepcopy(HOST)
    other["open_sites"] = [{"path": "x"}]
    assert _authority_view(WIRE) != _authority_view(other)


def test_view_does_not_mutate_input() -> None:
    before = copy.deepcopy(WIRE)
    _authority_view(WIRE)
    assert WIRE == before
    assert "receipts" in READ_FACT_DELIVERY_KEYS


def test_whole_file_prune_keys_are_ignored_but_revision_identity_is_not() -> None:
    wire = copy.deepcopy(WIRE)
    wire["read_facts"][0].update({"content_sha256": "d", "lines": {"start": 1, "end": 9, "total": 9}})
    assert _authority_view(wire) == _authority_view(HOST)
    host = copy.deepcopy(HOST)
    host["read_facts"][0]["read_revision"] = "sha256:other"
    assert _authority_view(wire) != _authority_view(host)


def test_open_site_byte_representation_is_normalized_in_the_frame_view() -> None:
    import hashlib

    data = "x = 1\n"
    digest = hashlib.sha256(data.encode()).hexdigest()
    base = {"path": "a.py", "line": 1, "content_sha256": digest}
    wire = {"open_sites": [{**base, "current_bytes": data}], "next_site": {**base, "current_bytes": data}}
    host = {"open_sites": [{**base, "current_bytes": None, "source_ref": {"content_sha256": digest}}],
            "next_site": {**base, "current_bytes": None, "source_ref": {"content_sha256": digest}}}
    assert _authority_view(wire) == _authority_view(host)
    host["next_site"]["path"] = "b.py"
    assert _authority_view(wire) != _authority_view(host)

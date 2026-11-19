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


LINES_SHORT = {"start": 1, "end": 10, "total": 500}
LINES_WHOLE = {"start": 1, "end": 500, "total": 500}


def _views(wire, host, whole_file_paths=frozenset()):
    from kernel_frame_allocation_validation import _whole_file_pruned_keys

    pruned = _whole_file_pruned_keys(wire, host, frozenset(whole_file_paths))
    return _authority_view(wire, pruned), _authority_view(host, pruned)


def test_read_fact_lines_mismatch_without_whole_file_window_still_replans() -> None:
    wire, host = copy.deepcopy(WIRE), copy.deepcopy(HOST)
    wire["read_facts"][0]["lines"] = LINES_WHOLE
    host["read_facts"][0]["lines"] = LINES_SHORT
    assert _authority_view(wire) != _authority_view(host)
    wire["read_facts"][0]["lines"] = LINES_SHORT
    wire["read_facts"][0]["content_sha256"] = "one"
    host["read_facts"][0]["content_sha256"] = "two"
    assert _authority_view(wire) != _authority_view(host)
    # One side missing the keys is a difference too when nothing pruned them.
    del host["read_facts"][0]["content_sha256"], host["read_facts"][0]["lines"]
    assert _authority_view(wire) != _authority_view(host)
    frame_view, host_view = _views(wire, host)
    assert frame_view != host_view


def test_whole_file_window_prune_is_ignored_only_for_that_path() -> None:
    wire, host = copy.deepcopy(WIRE), copy.deepcopy(HOST)
    wire["read_facts"][0].update({"content_sha256": "d", "lines": LINES_WHOLE})
    frame_view, host_view = _views(wire, host, {"owned_source.py"})
    assert frame_view == host_view
    frame_view, host_view = _views(wire, host, {"other.py"})
    assert frame_view != host_view
    host["read_facts"][0]["read_revision"] = "sha256:other"
    frame_view, host_view = _views(wire, host, {"owned_source.py"})
    assert frame_view != host_view


def test_whole_file_window_does_not_hide_a_two_sided_difference() -> None:
    wire, host = copy.deepcopy(WIRE), copy.deepcopy(HOST)
    wire["read_facts"][0].update({"content_sha256": "d", "lines": LINES_WHOLE})
    host["read_facts"][0].update({"content_sha256": "d", "lines": LINES_SHORT})
    frame_view, host_view = _views(wire, host, {"owned_source.py"})
    assert frame_view != host_view
    # A key only one side lost is pruned; the key both still carry is compared.
    del host["read_facts"][0]["lines"]
    host["read_facts"][0]["content_sha256"] = "other"
    frame_view, host_view = _views(wire, host, {"owned_source.py"})
    assert frame_view != host_view
    host["read_facts"][0]["content_sha256"] = "d"
    frame_view, host_view = _views(wire, host, {"owned_source.py"})
    assert frame_view == host_view


def test_whole_file_window_paths_match_the_source_metadata_prune(tmp_path) -> None:
    import hashlib
    import json

    from kernel_frame_allocation_validation import whole_file_window_paths
    from kernel_frame_source_snapshot import _current_window, _prune_redundant_source_metadata
    from kernel_stateframe_framefit5 import _state_payload

    facts = []
    windows = []
    for name, stale in (("pkg/fresh.py", False), ("pkg/stale.py", True)):
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True)
        target.write_text(f"VALUE = {len(name)}\n")
        revision = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
        window = _current_window(tmp_path, name, 1, 1, revision)
        assert window is not None
        windows.append({"role": "user", "content": window[0]})
        current = "sha256:changed" if stale else revision
        facts.append({"path": name, "current_revision": current, "read_revision": current,
                      "content_sha256": "d", "lines": {"start": 1, "end": 1, "total": 1}})
    facts.append({"path": "pkg/unshown.py", "current_revision": "sha256:u",
                  "content_sha256": "d", "lines": {"start": 1, "end": 1, "total": 9}})
    messages = [{"role": "user", "content": "[StateFrame]\n" + json.dumps({
        "state": {"framefit5": {"mode": "FRAME_ONLY", "read_facts": facts}},
    })}, *windows]
    paths = whole_file_window_paths(messages, copy.deepcopy(facts))
    assert paths == frozenset({"pkg/fresh.py"})
    _prune_redundant_source_metadata(messages)
    pruned = {fact["path"] for fact in _state_payload(messages[0])[1]["framefit5"]["read_facts"]
              if "lines" not in fact and "content_sha256" not in fact}
    assert pruned == paths


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

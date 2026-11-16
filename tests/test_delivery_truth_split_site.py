"""A next site whose range is shown across several FULL captures.

Delivery truth marks the range present when the union of shown captures covers
it, but keeps current_bytes only when one capture holds those exact bytes. A
site split across two adjacent captures used to leave read_this_request=True
with current_bytes=None, which final validation rejects as an incomplete
receipt and which crashed the default path before any edit.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from kernel_working_set_facts import _update_frame
from test_working_set_render import snapshot, source

import kernel_stateframe_framefit5 as framefit5

TEXT = "one\ntwo\nthree\nfour\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _setup(tmp_path: Path):
    revision = "sha256:" + _sha(TEXT)
    first = replace(source("first", "one\ntwo\n", first=1, revision=revision,
                           state="RESIDENT"), capture_ids=("c0",))
    second = replace(source("second", "three\nfour\n", first=3, ordinal=2,
                            revision=revision, state="RESIDENT"), capture_ids=("c1",))
    site = {
        "schema": "stateframe-framefit4-open-edit-site/1",
        "source": "last_read_or_edit_region", "literal": "", "path": "a.py", "line": 1,
        "lines": {"start": 1, "end": 4, "total": 4}, "revision": revision,
        "content_sha256": _sha(TEXT), "current_bytes": TEXT, "current_line": "one",
    }
    (tmp_path / "a.py").write_text(TEXT)
    snap = snapshot(first, second, current_revisions=(("a.py", revision),))
    return site, snap, [first, second]


def test_split_site_passes_final_verification(tmp_path: Path):
    site, snap, full = _setup(tmp_path)
    frame = {"next_site": site, "open_sites": [site]}
    _update_frame(frame, snap, full)
    assert frame["next_site"]["current_bytes"] is None
    assert frame["next_site"]["read_this_request"] is False
    framefit5._verify_next_site_bytes(frame["next_site"], tmp_path)


def test_single_capture_site_keeps_its_bytes(tmp_path: Path):
    site, snap, full = _setup(tmp_path)
    whole = replace(source("whole", TEXT, first=1, revision=site["revision"],
                           state="RESIDENT"), capture_ids=("c0",))
    frame = {"next_site": site}
    _update_frame(frame, snapshot(whole, current_revisions=snap.current_revisions), [whole])
    assert frame["next_site"]["current_bytes"] == TEXT
    assert frame["next_site"]["read_this_request"] is True
    framefit5._verify_next_site_bytes(frame["next_site"], tmp_path)

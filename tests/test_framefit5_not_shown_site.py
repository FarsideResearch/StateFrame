"""Next-site pointers whose bytes were not delivered by this request.

The working-set delivery pass clears current_bytes (value None) on a site that
was not shown in the request. Such a site is a location pointer: the model is
told to re-read it. Final validation must accept that shape, while any other
site lacking both current_bytes and source_ref still fails loud.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import kernel_stateframe_framefit5 as framefit5

SOURCE = "".join(f"line {number}\n" for number in range(1, 21))


def _pointer(path: str, source: str, **overrides: object) -> dict[str, object]:
    lines = source.splitlines(keepends=True)
    excerpt = "".join(lines[0:20])
    site: dict[str, object] = {
        "schema": "stateframe-framefit4-open-edit-site/1",
        "source": "last_read_or_edit_region",
        "literal": "", "path": path, "line": 1,
        "lines": {"start": 1, "end": 20, "total": len(lines)},
        "revision": "sha256:" + hashlib.sha256(source.encode()).hexdigest(),
        "content_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
        "current_bytes": excerpt, "current_line": "line 1",
        "read_this_request": True,
    }
    site.update(overrides)
    return site


def _workspace(tmp_path: Path) -> Path:
    target = tmp_path / "tests" / "validators" / "test_urlport.py"
    target.parent.mkdir(parents=True)
    target.write_text(SOURCE)
    return tmp_path


def test_not_shown_pointer_passes_final_verification(tmp_path: Path):
    root = _workspace(tmp_path)
    pointer = _pointer("tests/validators/test_urlport.py", SOURCE,
                       read_this_request=False, current_bytes=None)
    framefit5._verify_next_site_bytes(pointer, root)


def test_site_missing_both_fields_and_not_marked_not_shown_still_raises(tmp_path: Path):
    root = _workspace(tmp_path)
    site = _pointer("tests/validators/test_urlport.py", SOURCE, current_bytes=None)
    with pytest.raises(framefit5.FrameFit5Error, match="current_bytes"):
        framefit5._verify_next_site_bytes(site, root)


def test_not_shown_marker_without_the_null_key_still_raises(tmp_path: Path):
    root = _workspace(tmp_path)
    site = _pointer("tests/validators/test_urlport.py", SOURCE, read_this_request=False)
    site.pop("current_bytes")
    with pytest.raises(framefit5.FrameFit5Error, match="current_bytes"):
        framefit5._verify_next_site_bytes(site, root)


@pytest.mark.parametrize("case", ["covered", "uncovered", "stale_revision"])
def test_delivery_truth_shapes_pass_final_verification(tmp_path: Path, case: str):
    root = _workspace(tmp_path)
    if case == "covered":
        site = _pointer("tests/validators/test_urlport.py", SOURCE)
    elif case == "uncovered":
        site = _pointer("tests/validators/test_urlport.py", SOURCE,
                        read_this_request=False, current_bytes=None)
    else:
        site = _pointer("tests/validators/test_urlport.py", SOURCE,
                        revision="sha256:" + "0" * 64, read_this_request=False,
                        current_bytes=None)
    if case == "stale_revision":
        with pytest.raises(framefit5._CandidateSourceChanged):
            framefit5._verify_next_site_bytes(site, root)
    else:
        framefit5._verify_next_site_bytes(site, root)

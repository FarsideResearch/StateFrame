"""Route-only next-site receipts: undelivered bytes stay nulled but the source revision still binds."""

import hashlib

import pytest
from kernel_stateframe_framefit5 import (
    FrameFit5Error,
    _CandidateSourceChanged,
    _verify_next_site_bytes,
)

BODY = b"alpha = 1\nbeta = 2\ngamma = 3\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_bytes(BODY)
    return tmp_path


def _route_only_site(**overrides):
    excerpt = b"beta = 2\n"
    site = {
        "path": "pkg/mod.py",
        "revision": "sha256:" + _sha(BODY),
        "lines": {"start": 2, "end": 2},
        "line": 2,
        "current_line": "beta = 2",
        "current_bytes": None,
        "read_this_request": False,
        "content_sha256": _sha(excerpt),
    }
    site.update(overrides)
    return site


def test_route_only_site_passes_and_returns_file_bytes(workspace):
    raw = _verify_next_site_bytes(_route_only_site(), workspace)
    assert raw == BODY


def test_null_bytes_without_read_flag_still_incomplete(workspace):
    site = _route_only_site()
    del site["read_this_request"]
    with pytest.raises(FrameFit5Error, match="incomplete: current_bytes"):
        _verify_next_site_bytes(site, workspace)


def test_null_bytes_with_read_flag_true_still_incomplete(workspace):
    with pytest.raises(FrameFit5Error, match="incomplete: current_bytes"):
        _verify_next_site_bytes(_route_only_site(read_this_request=True), workspace)


def test_route_only_wrong_revision_raises_source_changed(workspace):
    site = _route_only_site(revision="sha256:" + "0" * 64)
    with pytest.raises(_CandidateSourceChanged):
        _verify_next_site_bytes(site, workspace)


def test_route_only_range_past_eof_is_invalid(workspace):
    site = _route_only_site(lines={"start": 2, "end": 99})
    with pytest.raises(FrameFit5Error, match="range is invalid"):
        _verify_next_site_bytes(site, workspace)


def test_route_only_line_outside_range_raises(workspace):
    site = _route_only_site(line=3, current_line="gamma = 3")
    with pytest.raises(FrameFit5Error, match="outside its source window"):
        _verify_next_site_bytes(site, workspace)


def test_delivered_str_bytes_mismatching_disk_still_raise_source_changed(workspace):
    site = _route_only_site(current_bytes="beta = 9\n", read_this_request=True)
    with pytest.raises(_CandidateSourceChanged):
        _verify_next_site_bytes(site, workspace)

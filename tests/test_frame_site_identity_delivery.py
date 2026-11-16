"""Open-site inventory identity ignores per-request delivery annotations.

The working-set delivery pass rewrites each open site with read_this_request
and current_bytes for the request that renders it, and the carrier may swap
between current_bytes and source_ref per request. The allocation inventory
must still match the fresh host sites on re-prepare, while the source pins
(path, revision, lines, content_sha256) keep detecting real changes.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from kernel_frame_optional_allocation import identity, site_identity, snapshot_optional_inputs

SOURCE = "".join(f"line {number}\n" for number in range(1, 21))
PATH = "tests/validators/test_urlport.py"


def _site(**overrides: object) -> dict[str, object]:
    excerpt = "".join(SOURCE.splitlines(keepends=True)[0:20])
    site: dict[str, object] = {
        "schema": "stateframe-framefit4-open-edit-site/1",
        "source": "last_read_or_edit_region",
        "literal": "",
        "path": PATH,
        "line": 1,
        "lines": {"start": 1, "end": 20, "total": 20},
        "revision": "sha256:" + hashlib.sha256(SOURCE.encode()).hexdigest(),
        "content_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
        "current_bytes": excerpt,
        "current_line": "line 1",
        "read_this_request": True,
    }
    site.update(overrides)
    return site


def _annotated(site: dict[str, object]) -> dict[str, object]:
    """The shape rebuild_delivery_truth leaves when the request did not show the site."""
    copied = copy.deepcopy(site)
    copied["read_this_request"] = False
    copied["current_bytes"] = None
    return copied


def _workspace(tmp_path: Path) -> Path:
    target = tmp_path / PATH
    target.parent.mkdir(parents=True)
    target.write_text(SOURCE)
    return tmp_path


def _frame(site: dict[str, object]) -> dict[str, object]:
    return {"open_sites": [site], "next_site": site, "applied_edits": []}


def test_delivery_annotations_do_not_change_site_identity():
    fresh = _site()
    assert site_identity(_annotated(fresh)) == site_identity(fresh)


def test_source_ref_carrier_swap_does_not_change_site_identity():
    inline = _site()
    by_ref = copy.deepcopy(inline)
    by_ref.pop("current_bytes")
    by_ref["source_ref"] = {
        "path": PATH, "lines": {"start": 1, "end": 20},
        "content_sha256": inline["content_sha256"],
    }
    assert site_identity(by_ref) == site_identity(inline)


def test_raw_identity_still_sees_the_annotations():
    """Documents the defect site_identity exists to avoid: identity() is annotation-sensitive."""
    fresh = _site()
    assert identity(_annotated(fresh)) != identity(fresh)


@pytest.mark.parametrize(
    "change",
    [
        {"content_sha256": "sha256:" + "0" * 64},
        {"revision": "sha256:" + "1" * 64},
        {"lines": {"start": 1, "end": 19, "total": 20}},
        {"path": "tests/validators/other_file.py"},
    ],
    ids=["content_sha256", "revision", "lines", "path"],
)
def test_source_pins_still_change_site_identity(change: dict[str, object]):
    base = _site()
    assert site_identity({**base, **change}) != site_identity(base)


def test_open_site_inventory_matches_after_delivery_annotation(tmp_path: Path):
    """Real inventory builder: the allocated (annotated) frame must yield the same inventory
    as the fresh-site frame that validation recomputes."""
    root = _workspace(tmp_path)
    fresh = _site()
    _, allocated_inventory = snapshot_optional_inputs(_frame(_annotated(fresh)), root)
    _, fresh_inventory = snapshot_optional_inputs(_frame(fresh), root)
    assert allocated_inventory["inventory_identity"] == fresh_inventory["inventory_identity"]
    assert allocated_inventory["next_site"] == fresh_inventory["next_site"]
    assert allocated_inventory["inventory_identity"] == [site_identity(fresh)]

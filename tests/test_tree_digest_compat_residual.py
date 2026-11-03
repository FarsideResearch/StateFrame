"""Pin the known residual of the request-compatible digest encoding.

`_revision12`, `dependency_revision` and the coordinated-change tree revision keep the 0.3.18
bytes (path + NUL + content, no terminator) for regular files so that request bytes stay
identical. Two different path lists can therefore still collide when one file's content ends
with the start of the next path. Susan's option (a) keeps this residual for now; option (b),
the fully typed encoding, changes request bytes. Whoever lands (b) must flip these assertions
to `!=` on purpose and update KNOWN_LIMITATIONS.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kernel_coordinated_change_ledger import _compute_tree_revision
from kernel_request_evidence_closure import dependency_revision
from kernel_request_source_window import _revision12


def _make(root: Path, files: dict[str, bytes]) -> Path:
    root.mkdir()
    for name, content in files.items():
        (root / name).write_bytes(content)
    return root


@pytest.mark.parametrize("site", [_revision12, dependency_revision, _compute_tree_revision],
                         ids=["_revision12", "dependency_revision", "_compute_tree_revision"])
@pytest.mark.parametrize("left,right", [
    ({"a": b"bc", "d": b"e"}, {"a": b"b", "cd": b"e"}),
    ({"a": b"bc", "d": b""}, {"a": b"b", "cd": b""}),
], ids=["sequence", "empty-tail"])
def test_option_a_residual_sequence_boundary_still_collides(tmp_path, site, left, right):
    first = _make(tmp_path / "first", left)
    second = _make(tmp_path / "second", right)
    # Known residual under option (a): identical revisions for different path lists.
    assert site(first, sorted(left)) == site(second, sorted(right))

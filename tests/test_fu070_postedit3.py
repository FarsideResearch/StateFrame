"""FU-070-ALLOC post-edit pass 3: an edit with no prior read has no open site."""
from __future__ import annotations

import hashlib

from stateframe.engine.kernel_stateframe_action_ledger import (
    _fallback_edit_open_site_locator,
)


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


_REVISION = _sha(b"post-edit source")
_ARGS = {"path": "owned_source.py", "old_str": "VALUE = 1\n", "new_str": "VALUE = 2\n"}


def test_fallback_edit_without_prior_read_returns_no_site():
    # No read in this request: the fallback must not anchor at line 1 with an empty revision.
    locator = _fallback_edit_open_site_locator(
        "str_replace", _ARGS, "str_replace applied.", {}, _REVISION
    )
    assert locator is None


def test_fallback_edit_with_prior_read_still_anchors_on_that_read():
    read_fact = {"revision": _sha(b"pre-edit source"), "lines": {"start": 4, "end": 9}}
    locator = _fallback_edit_open_site_locator(
        "str_replace",
        _ARGS,
        "str_replace applied.",
        {"owned_source.py": (read_fact, "VALUE = 1\n")},
        _REVISION,
    )
    assert locator is not None
    assert locator["path"] == "owned_source.py"
    assert locator["lines"] == {"start": 4, "end": 4}
    assert locator["last_read_revision"] == read_fact["revision"]
    assert locator["last_action"] == "edit"


def test_fallback_edit_without_prior_read_is_none_for_every_edit_tool_name():
    for name in ("str_replace", "edit", "edit_file"):
        assert (
            _fallback_edit_open_site_locator(
                name, _ARGS, f"{name} applied.", {}, _REVISION
            )
            is None
        )

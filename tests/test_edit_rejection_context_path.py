"""Path-resolution behavior for rejected edit context."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from kernel_edit_rejection_fresh_text import annotate_rejection  # noqa: E402
from kernel_tool_reply import tool_reply_for_call  # noqa: E402
from kernel_v33_lean import request_bytes  # noqa: E402

_REJECTION = "REFUSED OLD_TEXT_MISMATCH: old_str not found"


@pytest.mark.parametrize("error_class", [OSError, ValueError, RuntimeError])
def test_rejection_path_resolution_failure_returns_unavailable_context(
        tmp_path, monkeypatch, error_class):
    monkeypatch.setenv("STATEFRAME_EDIT_REJECTION_FRESH_TEXT", "on")
    candidate = tmp_path / "candidate.py"
    candidate.write_text("value = 2\n", encoding="utf-8")
    expected = _REJECTION + (
        "\n\n[Fresh current-file context unavailable: "
        f"{error_class.__name__}; source was not changed.]"
    )
    original_resolve = Path.resolve
    error_detail = "synthetic resolver failure"

    def fail_candidate_resolution(path, *args, **kwargs):
        if path == candidate:
            raise error_class(error_detail)
        return original_resolve(path, *args, **kwargs)

    with patch.object(Path, "resolve", autospec=True,
                      side_effect=fail_candidate_resolution), \
            patch.object(Path, "read_bytes", autospec=True) as read_bytes:
        observed = annotate_rejection(
            _REJECTION,
            root=tmp_path,
            path="candidate.py",
            old_text="value = 1\n",
        )

    assert observed == expected
    assert str(tmp_path) not in observed
    assert "candidate.py" not in observed
    assert error_detail not in observed
    read_bytes.assert_not_called()


def test_disabled_context_returns_original_rejection_without_path_resolution(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_EDIT_REJECTION_FRESH_TEXT", "off")
    monkeypatch.setenv("STATEFRAME_EDIT_ANCHOR_REBASE", "off")
    with patch.object(Path, "resolve", autospec=True) as resolve, \
            patch.object(Path, "read_bytes", autospec=True) as read_bytes:
        observed = annotate_rejection(
            _REJECTION, root=tmp_path, path="candidate.py", old_text="value = 1\n")

    assert observed == _REJECTION
    messages = [
        {"role": "system", "content": "engine rules"},
        {"role": "user", "content": "edit the setting"},
    ]
    base_messages = messages.copy()
    messages.append(tool_reply_for_call(messages, "synthetic", observed, tool="str_replace"))
    base_messages.append(tool_reply_for_call(
        base_messages, "synthetic", _REJECTION, tool="str_replace",
    ))
    assert request_bytes(messages, []) == request_bytes(base_messages, [])
    resolve.assert_not_called()
    read_bytes.assert_not_called()


def test_relative_path_returns_fresh_context_excerpt(tmp_path):
    candidate = tmp_path / "nested" / "candidate.py"
    candidate.parent.mkdir()
    candidate.write_bytes(b"fresh = 2\n")

    observed = annotate_rejection(
        _REJECTION,
        root=tmp_path,
        path="nested/candidate.py",
        old_text="fresh = 1\n",
    )

    assert observed.startswith(
        _REJECTION + "\n\n[Fresh current-file context: path=nested/candidate.py;")
    assert "fresh = 2" in observed
    assert "source was not changed" not in observed


def test_lookup_failure_returns_typed_unavailable_context(tmp_path):
    observed = annotate_rejection(
        _REJECTION, root=tmp_path, path="missing.py", old_text="value = 1\n")

    assert observed == _REJECTION + (
        "\n\n[Fresh current-file context unavailable: "
        "FileNotFoundError; source was not changed.]"
    )


def test_outside_workspace_path_returns_unavailable_context_without_reading(
        tmp_path):
    with patch.object(Path, "read_bytes", autospec=True) as read_bytes:
        observed = annotate_rejection(
            _REJECTION, root=tmp_path, path="../outside.py", old_text="value = 1\n")

    assert observed == _REJECTION + (
        "\n\n[Fresh current-file context unavailable: "
        "ValueError; source was not changed.]"
    )
    read_bytes.assert_not_called()


def test_malformed_observation_returns_unchanged_without_path_resolution(tmp_path):
    with patch.object(Path, "resolve", autospec=True) as resolve:
        observed = annotate_rejection(
            None, root=tmp_path, path="candidate.py", old_text="value = 1\n")

    assert observed is None
    resolve.assert_not_called()

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

import kernel_navigation_pages as navigation
from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
from kernel_workspace_state import WorkspaceFence, WorkspaceState


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "off")
    root = tmp_path / "workspace"
    root.mkdir()
    consumers = WorkspaceConsumers(WorkspaceState(
        WorkspaceFence.create(root, tmp_path / "fence"),
    ))
    bind_consumers(root, consumers)
    return root, consumers


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _release_page(digest, total, offset, lines):
    shown = len(lines)
    remaining = max(0, total - offset - shown)
    cursor = f"nav1:{digest}:{offset + shown}" if remaining else None
    metadata = {
        "schema": "navigation-page/1", "result_set_digest": digest,
        "shown": shown, "total": total, "offset": offset,
        "remaining": remaining, "complete": cursor is None,
        "continuation": cursor, "mode": "literal", "pattern": "HIT",
        "scope": ".", "file_glob": None, "page_size": 2,
        "scan_complete": True, "unreadable_files": 0, "non_text_files": 0,
    }
    header = (f"[search page] shown={shown} total={total} offset={offset} "
              f"complete={str(cursor is None).lower()} scan_complete=true")
    next_line = (f"{remaining} more, page with cursor={cursor}"
                 if cursor else "0 more; complete=true")
    return navigation.NavigationPage("\n".join([header, next_line, *lines]), metadata)


def _assert_release_reply(root, digest, total, first, second):
    args = {"mode": "literal", "requested_page_size": 2}
    expected = _release_page(digest, total, 0, first)
    actual = navigation.search_repository(root, "HIT", **args)
    assert dataclasses.replace(actual, evidence_digest="") == expected
    assert actual.evidence_digest
    assert navigation.search_repository(root, "HIT", **args) == actual
    if second:
        expected_next = _release_page(digest, total, 2, second)
        actual_next = navigation.search_repository(
            root, "HIT", cursor=expected.metadata["continuation"], **args,
        )
        assert dataclasses.replace(actual_next, evidence_digest="") == expected_next
        assert actual_next.evidence_digest not in ("", actual.evidence_digest)
    return actual


def test_cached_replies_match_release_across_tree_and_ignore_changes(workspace):
    root, _consumers = workspace
    same = root / "same.txt"
    keep = root / "keep.txt"
    moved = root / "tree" / "nested.txt"
    ignored = root / "ignored" / "match.txt"
    _write(same, "HIT alpha\n")
    _write(keep, "HIT keep\n")
    _write(moved, "HIT moved\n")
    _write(ignored, "HIT ignored\n")

    initial_digest = "9f763d833041c2ef61371f3e"
    initial = _assert_release_reply(root, initial_digest, 4,
                          ["keep.txt:1:HIT keep", "same.txt:1:HIT alpha"],
                          ["ignored/match.txt:1:HIT ignored", "tree/nested.txt:1:HIT moved"])

    old = same.stat()
    _write(same, "ALT alpha\n")
    os.utime(same, ns=(old.st_atime_ns, old.st_mtime_ns))
    _assert_release_reply(root, initial_digest, 4,
                          ["keep.txt:1:HIT keep", "same.txt:1:HIT alpha"],
                          ["ignored/match.txt:1:HIT ignored", "tree/nested.txt:1:HIT moved"])

    _write(root / "new.txt", "HIT new\n")
    changed = _assert_release_reply(root, "d0822d94e6b9af911891f48d", 4,
                          ["keep.txt:1:HIT keep", "new.txt:1:HIT new"],
                          ["ignored/match.txt:1:HIT ignored", "tree/nested.txt:1:HIT moved"])

    assert changed.evidence_digest != initial.evidence_digest
    keep.unlink()
    _assert_release_reply(root, "041be3d9aad20991e711ab96", 3,
                          ["new.txt:1:HIT new", "ignored/match.txt:1:HIT ignored"],
                          ["tree/nested.txt:1:HIT moved"])

    (root / "tree").rename(root / "renamed")
    renamed_digest = "348359cd3d3769d1e763ac45"
    renamed_first = ["new.txt:1:HIT new", "ignored/match.txt:1:HIT ignored"]
    renamed_second = ["renamed/nested.txt:1:HIT moved"]
    _assert_release_reply(root, renamed_digest, 3, renamed_first, renamed_second)

    _write(root / ".gitignore", "ignored/\n")
    first_ignore_reply = _assert_release_reply(
        root, renamed_digest, 3, renamed_first, renamed_second,
    )
    assert "ignored/match.txt:1:HIT ignored" in first_ignore_reply.text
    _write(root / ".gitignore", "!ignored/\n")
    assert _assert_release_reply(
        root, renamed_digest, 3, renamed_first, renamed_second,
    ) == first_ignore_reply


def test_pinned_cache_uses_release_keys_and_result_limit(workspace):
    root, consumers = workspace
    _write(root / "a.txt", "HIT\n")
    navigation.search_repository(root, "HIT", mode="literal")
    cache = vars(consumers)["_navigation_search_cache"]

    assert cache.results.limit == 24
    result_keys = tuple(cache.results.entries)
    assert len(result_keys) == 1
    assert isinstance(result_keys[0][1], bytes)
    assert result_keys[0] == (
        root.resolve(), result_keys[0][1], ("literal", "HIT", None), None,
    )
    assert len(tuple(cache.files.entries)) == 1
    file_key = next(iter(cache.files.entries))
    assert file_key[:3] == ("literal", "HIT", "a.txt")
    assert isinstance(file_key[3], bytes)


def test_ascii_literal_miss_skips_decode_and_opens_file_once(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "large.txt"
    path.write_bytes(b"ordinary text\n" * 100_000)
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "off")
    opened = []
    original_open = os.open

    def tracked_open(name, flags, *args, **kwargs):
        if kwargs.get("dir_fd") is not None:
            opened.append(name)
        return original_open(name, flags, *args, **kwargs)

    def unexpected_decode(_raw):
        pytest.fail("ASCII literal misses should skip UTF-8 decoding")

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {tracked_open})
    monkeypatch.setattr(navigation, "_decode_search_text", unexpected_decode)
    monkeypatch.setattr(Path, "read_bytes", lambda _path: pytest.fail("path reader used"))
    result = navigation.search_repository(root, "MISSING", mode="literal")

    assert result.metadata["total"] == 0
    assert result.metadata["scan_complete"] is True
    assert result.metadata["non_text_files"] == 0
    assert opened == ["large.txt"]


@pytest.mark.parametrize("raw", [b"ordinary \xff\n", b"ordinary \x00\n", b"ordinary " * 2048 + b"\xff"])
def test_literal_miss_keeps_non_text_evidence(tmp_path, monkeypatch, raw):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "not-text.txt").write_bytes(raw)
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")

    page = navigation.search_repository(root, "MISSING", mode="literal")

    assert page.metadata["total"] == 0
    assert page.metadata["non_text_files"] == 1
    assert page.metadata["scan_complete"] is True


def test_unicode_literal_prefilter_keeps_full_matching_line(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    line = "café " + "x" * ((1 << 20) + 31)
    (root / "large.txt").write_text(line, encoding="utf-8")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "off")
    opened = []
    original_open = os.open

    def tracked_open(name, flags, mode=0o777, *, dir_fd=None):
        if not flags & getattr(os, "O_DIRECTORY", 0):
            opened.append(name)
        return original_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {tracked_open})

    page = navigation.search_repository(root, "café", mode="literal")

    row = {"path": "large.txt", "line": 1, "text": line}
    assert page.metadata["total"] == 1
    assert page.metadata["result_set_digest"] == navigation._fingerprint([row])
    assert page.text.endswith(f"large.txt:1:{line}")
    assert "truncated_lines" not in page.metadata
    assert opened == ["large.txt"]


def test_literal_prefilter_does_not_match_across_line_boundaries(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "lines.txt").write_text("part one\npart two\n", encoding="utf-8")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")

    page = navigation.search_repository(root, "one\npart", mode="literal")

    assert page.metadata["total"] == 0
    assert page.metadata["scan_complete"] is True


def test_literal_prefilter_is_not_used_for_regex(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "large.txt").write_bytes(b"A-B\n" + b"." * (16 << 10))
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")

    page = navigation.search_repository(root, "A.B", mode="regex")

    assert page.metadata["total"] == 1
    assert page.text.startswith("[search page] shown=1 total=1")
    assert "large.txt:1:A-B" in page.text

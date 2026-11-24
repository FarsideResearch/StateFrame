"""Surrogate-escaped search names are safe display labels, never input paths."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

import kernel_navigation_pages as navigation
from kernel_file_ops import FileOperationRefused, admit_file_delete
from kernel_tool_paths import ToolPathEscape, resolve_tool_path
from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
from kernel_workspace_state import WorkspaceFence, WorkspaceState, WorkspaceUnavailable


def _backend(root: Path, tmp_path: Path, monkeypatch, mode: str):
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "off")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off" if mode == "unpinned" else "on")
    if mode == "unpinned":
        return
    consumers = WorkspaceConsumers(WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence")))
    bind_consumers(root, consumers)
    if mode == "fallback":
        @contextmanager
        def busy(_boundary):
            raise WorkspaceUnavailable("writer busy")
            yield
        monkeypatch.setattr(consumers.state, "pin", busy)


def _write_bytes(root: Path, relative: bytes, content: bytes = b"MATCH\n") -> None:
    path = os.fsencode(root) + b"/" + relative
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as stream:
        stream.write(content)


def _expected_rows() -> list[dict[str, object]]:
    paths = [
        "a.txt", "b.txt", "//undecodable-name/c-\\xff.txt",
        "//undecodable-name/d-\\xfe.txt",
        "//undecodable-name/e-\\xfd/sub-\\xfe.txt",
        "undecodable-name/c-\\xff.txt",
    ]
    return [{"path": path, "line": 1, "text": "MATCH"} for path in paths]


@pytest.mark.skipif(os.name != "posix", reason="surrogate filenames require POSIX filesystem encoding")
@pytest.mark.parametrize("mode", ["pinned", "fallback", "unpinned"])
def test_odd_names_keep_all_hits_and_page_digest(tmp_path, monkeypatch, mode):
    root = tmp_path / "repo"
    root.mkdir()
    _write_bytes(root, b"a.txt")
    _write_bytes(root, b"b.txt")
    _write_bytes(root, b"c-\xff.txt")
    _write_bytes(root, b"d-\xfe.txt")
    _write_bytes(root, b"e-\xfd/sub-\xfe.txt")
    _write_bytes(root, b"undecodable-name/c-\\xff.txt")
    _backend(root, tmp_path, monkeypatch, mode)

    rows = _expected_rows()
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
    first = navigation.search_repository(root, "MATCH", mode="literal", requested_page_size=3)
    second = navigation.search_repository(root, "MATCH", mode="literal",
                                          cursor=first.metadata["continuation"], requested_page_size=3)
    assert first.metadata["total"] == second.metadata["total"] == len(rows)
    assert first.metadata["result_set_digest"] == second.metadata["result_set_digest"] == digest
    assert first.metadata["continuation"] == f"nav1:{digest}:3"
    assert second.metadata["continuation"] is None
    assert first.text.splitlines()[2:] == [f"{row['path']}:1:MATCH" for row in rows[:3]]
    assert second.text.splitlines()[2:] == [f"{row['path']}:1:MATCH" for row in rows[3:]]


@pytest.mark.skipif(os.name != "posix", reason="surrogate filenames require POSIX filesystem encoding")
@pytest.mark.parametrize("mode", ["pinned", "fallback", "unpinned"])
@pytest.mark.parametrize("search_mode,query", [("literal", "MATCH"), ("path", ".txt")])
def test_unpaged_search_escapes_surrogate_names(tmp_path, monkeypatch, mode, search_mode, query):
    root = tmp_path / "repo"
    root.mkdir()
    _write_bytes(root, b"bad-\xff.txt")
    _write_bytes(root, b"good.txt")
    _backend(root, tmp_path, monkeypatch, mode)

    page = navigation.search_repository(root, query, mode=search_mode)
    assert page.metadata["total"] == 2
    assert "//undecodable-name/bad-\\xff.txt" in page.text
    page.text.encode("utf-8")


@pytest.mark.skipif(os.name != "posix", reason="surrogate filenames require POSIX filesystem encoding")
def test_directory_page_escapes_surrogate_file_and_directory_names_with_stable_pages(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _write_bytes(root, b"z.txt")
    _write_bytes(root, b"bad-\xff.txt")
    _write_bytes(root, b"folder-\xfe/inside.txt")
    _write_bytes(root, b"a.txt")

    rows = [
        {"name": "a.txt"},
        {"name": "//undecodable-name/bad-\\xff.txt"},
        {"name": "//undecodable-name/folder-\\xfe/"},
        {"name": "z.txt"},
    ]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
    first = navigation.directory_page(root, ".", requested_page_size=2)
    second = navigation.directory_page(root, ".", cursor=first.metadata["continuation"],
                                       requested_page_size=2)

    assert first.metadata["result_set_digest"] == second.metadata["result_set_digest"] == digest
    assert first.metadata["continuation"] == f"nav1:{digest}:2"
    assert second.metadata["continuation"] is None
    assert first.text.splitlines()[3:] == [row["name"] for row in rows[:2]]
    assert second.text.splitlines()[3:] == [row["name"] for row in rows[2:]]
    first.text.encode("utf-8")
    second.text.encode("utf-8")

    from kernel_tool_paths import ToolPathEscape, resolve_tool_path

    for label in (rows[1]["name"], rows[2]["name"]):
        with pytest.raises(ToolPathEscape):
            resolve_tool_path(root, label)


@pytest.mark.skipif(os.name != "posix", reason="surrogate filenames require POSIX filesystem encoding")
def test_workspace_layout_escapes_surrogate_names_and_keeps_cursor_digest(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(80):
        _write_bytes(root, f"f{index:03d}.txt".encode())
    _write_bytes(root, b"bad-\xff.txt")

    paths = ["bad-\udcff.txt", *(f"f{index:03d}.txt" for index in range(80))]
    rows = [{"path": ("//undecodable-name/bad-\\xff.txt" if path.startswith("bad-") else path)}
            for path in paths]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
    text = navigation.workspace_layout(root)

    assert "//undecodable-name/bad-\\xff.txt" in text
    assert "- f000.txt" in text and "- f078.txt" in text and "- f079.txt" not in text
    assert f"cursor=nav1:{digest}:80" in text
    text.encode("utf-8")


@pytest.mark.skipif(os.name != "posix", reason="surrogate filenames require POSIX filesystem encoding")
@pytest.mark.parametrize("mode", ["pinned", "fallback", "unpinned"])
def test_path_glob_pages_escape_names_from_tree_walk(tmp_path, monkeypatch, mode):
    root = tmp_path / "repo"
    root.mkdir()
    _write_bytes(root, b"bad-\xff.txt")
    _write_bytes(root, b"good.txt")
    _write_bytes(root, b"folder-\xfe/inside.txt")
    _backend(root, tmp_path, monkeypatch, mode)

    rows = [
        {"path": "//undecodable-name/bad-\\xff.txt"},
        {"path": "good.txt"},
        {"path": "//undecodable-name/folder-\\xfe/inside.txt"},
    ]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
    first = navigation.search_repository(root, ".txt", mode="path", file_glob="*.txt",
                                         requested_page_size=2)
    second = navigation.search_repository(root, ".txt", mode="path", file_glob="*.txt",
                                          cursor=first.metadata["continuation"],
                                          requested_page_size=2)

    assert first.metadata["result_set_digest"] == second.metadata["result_set_digest"] == digest
    assert first.metadata["total"] == 3
    assert first.text.splitlines()[2:] == [row["path"] for row in rows[:2]]
    assert second.text.splitlines()[2:] == [row["path"] for row in rows[2:]]
    first.text.encode("utf-8")
    second.text.encode("utf-8")


@pytest.mark.skipif(os.name != "posix", reason="surrogate filenames require POSIX filesystem encoding")
def test_display_name_is_rejected_as_read_scope_and_edit_input(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _write_bytes(root, b"bad-\xff.txt")
    _write_bytes(root, b"undecodable-name/bad-\\xff.txt")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    page = navigation.search_repository(root, "MATCH", mode="literal", requested_page_size=10)
    display = "//undecodable-name/bad-\\xff.txt"
    assert display in page.text

    opened = []
    original_open = os.open

    def trace_open(path, *args, **kwargs):
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", trace_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {trace_open})
    with pytest.raises(ToolPathEscape):
        resolve_tool_path(root, display)
    with pytest.raises(ValueError, match="inside the repository root"):
        navigation.search_repository(root, "MATCH", scope=display, mode="literal")
    with pytest.raises(FileOperationRefused):
        admit_file_delete(root, display, protected=lambda _path: False,
                          editable=lambda _relative: True)
    assert opened == []


def _display_label_target(root: Path) -> str:
    target = root / "reachable.txt"
    return "//undecodable-name/../../" + target.as_posix().lstrip("/")


@pytest.mark.skipif(os.name != "posix", reason="double-slash paths are POSIX-specific")
def test_display_label_continuation_is_rejected_by_shared_path_resolver(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "reachable.txt").write_text("MATCH\n", encoding="utf-8")
    raw = _display_label_target(root)
    assert raw.startswith("//undecodable-name/")

    with pytest.raises(ToolPathEscape):
        resolve_tool_path(root, raw)


@pytest.mark.skipif(os.name != "posix", reason="double-slash paths are POSIX-specific")
def test_other_double_slash_path_keeps_its_containment_refusal(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    requested = "//host/path"

    with pytest.raises(ToolPathEscape) as refused:
        resolve_tool_path(root, requested)
    assert str(refused.value) == (
        f"Path {requested!r} is outside the workspace. Use a repository-relative path.")


@pytest.mark.skipif(os.name != "posix", reason="double-slash paths are POSIX-specific")
def test_display_label_continuation_is_rejected_before_scoped_search_opens(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "reachable.txt").write_text("MATCH\n", encoding="utf-8")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "off")
    opened = []
    original_open = os.open

    def trace_open(path, *args, **kwargs):
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", trace_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {trace_open})
    with pytest.raises(ValueError, match="navigation scope must be inside the repository root"):
        navigation.search_repository(root, "MATCH", scope=_display_label_target(root), mode="literal")
    assert opened == []


@pytest.mark.skipif(os.name != "posix", reason="double-slash paths are POSIX-specific")
def test_display_label_continuation_is_rejected_before_scoped_listing_resolves(tmp_path, monkeypatch):
    from kernel_nav_basics import list_workspace_files

    root = tmp_path / "repo"
    root.mkdir()
    (root / "reachable.txt").write_text("MATCH\n", encoding="utf-8")
    raw = _display_label_target(root)
    resolved_scopes = []
    original_resolve = Path.resolve

    def trace_resolve(path, *args, **kwargs):
        if str(path) == raw:
            resolved_scopes.append(path)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", trace_resolve)
    with pytest.raises(ValueError, match="workspace navigation scope must be inside the repository root"):
        list_workspace_files(root, scope=raw)
    assert resolved_scopes == []


@pytest.mark.skipif(os.name != "posix", reason="double-slash paths are POSIX-specific")
def test_native_delete_refuses_display_label_continuation(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "reachable.txt").write_text("MATCH\n", encoding="utf-8")
    opened = []
    original_open = os.open

    def trace_open(path, *args, **kwargs):
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", trace_open)
    with pytest.raises(FileOperationRefused) as refused:
        admit_file_delete(root, _display_label_target(root),
                          protected=lambda _path: False, editable=lambda _relative: True)
    assert refused.value.code == "OUTSIDE_WORKSPACE"
    assert opened == []


@pytest.mark.parametrize("count", [40, 875, 3500])
def test_valid_odd_looking_names_keep_search_bytes(tmp_path, monkeypatch, count):
    from test_grep_retained_rows import _plain_search

    root = tmp_path / "repo"
    root.mkdir()
    for index in range(count):
        name = f'{index:05d}-100%-"quoted"-\u202e-\n.txt'
        (root / name).write_text("MATCH\n", encoding="utf-8")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "off")
    size = 500
    expected = _plain_search(root, "MATCH", mode="literal", page_size=size)
    actual = navigation.search_repository(root, "MATCH", mode="literal", requested_page_size=size)
    assert actual.text.encode("utf-8") == expected["text"].encode("utf-8")
    assert dict(actual.metadata) == expected["metadata"]

"""Search fallback must keep reads inside the workspace during tree changes."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

import kernel_navigation_pages as navigation
from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
from kernel_workspace_state import WorkspaceFence, WorkspaceState, WorkspaceUnavailable


def _mode(root, tmp_path, monkeypatch, mode):
    if mode == "unpinned":
        monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
        return None
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    consumers = WorkspaceConsumers(WorkspaceState(
        WorkspaceFence.create(root, tmp_path / "fence")))
    bind_consumers(root, consumers)
    if mode == "fallback":
        @contextmanager
        def busy(_boundary):
            raise WorkspaceUnavailable("writer busy")
            yield
        monkeypatch.setattr(consumers.state, "pin", busy)
    return consumers


def _expected(rows, *, pattern="needle", scope=".", size=1, offset=0):
    digest = hashlib.sha256(json.dumps(list(rows), sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()[:24]
    shown = rows[offset:offset + size]
    remaining = max(0, len(rows) - offset - len(shown))
    cursor = f"nav1:{digest}:{offset + len(shown)}" if remaining else None
    metadata = {
        "schema": "navigation-page/1", "result_set_digest": digest,
        "shown": len(shown), "total": len(rows), "offset": offset,
        "remaining": remaining, "complete": cursor is None,
        "continuation": cursor, "mode": "literal", "pattern": pattern,
        "scope": scope, "file_glob": None, "page_size": size,
        "scan_complete": True, "unreadable_files": 0, "non_text_files": 0,
    }
    next_line = (f"{remaining} more, page with cursor={cursor}" if cursor
                 else "0 more; complete=true")
    rendered = [f"{row['path']}:{row['line']}:{row['text']}" for row in shown]
    text = (f"[search page] shown={len(shown)} total={len(rows)} offset={offset} "
            f"complete={str(cursor is None).lower()} scan_complete=true\n"
            + next_line + "\n" + "\n".join(rendered))
    return navigation.NavigationPage(text, metadata)


@pytest.mark.parametrize("mode", ["pinned", "fallback", "unpinned"])
@pytest.mark.parametrize("route", ["root_link", "parent_link"])
def test_root_reached_through_link_keeps_027_pages(tmp_path, monkeypatch, mode, route):
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "off")
    parent = tmp_path / "parent"
    root = parent / "workspace"
    root.mkdir(parents=True)
    (root / "a.py").write_text("needle\n", encoding="utf-8")
    (root / "b.py").write_text("needle\n", encoding="utf-8")
    _mode(root, tmp_path, monkeypatch, mode)
    if route == "root_link":
        alias = tmp_path / "workspace-link"
        alias.symlink_to(root, target_is_directory=True)
    else:
        alias_parent = tmp_path / "parent-link"
        alias_parent.symlink_to(parent, target_is_directory=True)
        alias = alias_parent / "workspace"

    first = navigation.search_repository(alias, "needle", mode="literal",
                                         requested_page_size=1)
    assert first == _expected([
        {"path": "a.py", "line": 1, "text": "needle"},
        {"path": "b.py", "line": 1, "text": "needle"},
    ])
    assert navigation.search_repository(alias, "needle", mode="literal",
                                        cursor=first.metadata["continuation"],
                                        requested_page_size=1) == _expected([
        {"path": "a.py", "line": 1, "text": "needle"},
        {"path": "b.py", "line": 1, "text": "needle"},
    ], offset=1)


@pytest.mark.parametrize("mode", ["pinned", "fallback", "unpinned"])
@pytest.mark.parametrize("case", [
    "file_inside", "dir_inside", "file_outside", "dir_outside",
    "file_dangling", "dir_dangling", "file_loop", "dir_loop",
])
def test_in_workspace_links_keep_027_replies(tmp_path, monkeypatch, mode, case):
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "off")
    root = tmp_path / "workspace"
    inside = root / "inside"
    inside.mkdir(parents=True)
    (inside / "a.py").write_text("needle\n", encoding="utf-8")
    (inside / "b.py").write_text("needle\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("needle outside_token\n", encoding="utf-8")
    (root / "file_inside.py").symlink_to(inside / "a.py")
    (root / "dir_inside").symlink_to(inside, target_is_directory=True)
    (root / "file_outside.py").symlink_to(outside / "secret.py")
    (root / "dir_outside").symlink_to(outside, target_is_directory=True)
    (root / "file_dangling.py").symlink_to(root / "missing.py")
    (root / "dir_dangling").symlink_to(root / "missing-dir", target_is_directory=True)
    (root / "file_loop_a").symlink_to(root / "file_loop_b")
    (root / "file_loop_b").symlink_to(root / "file_loop_a")
    (root / "dir_loop_a").symlink_to(root / "dir_loop_b", target_is_directory=True)
    (root / "dir_loop_b").symlink_to(root / "dir_loop_a", target_is_directory=True)
    _mode(root, tmp_path, monkeypatch, mode)

    rows = [
        {"path": "inside/a.py", "line": 1, "text": "needle"},
        {"path": "inside/b.py", "line": 1, "text": "needle"},
    ]
    # Whole-workspace scans skip every link, including links to inside targets.
    assert navigation.search_repository(root, "needle", mode="literal",
                                        requested_page_size=1) == _expected(rows)
    scope = {
        "file_inside": "file_inside.py", "dir_inside": "dir_inside",
        "file_outside": "file_outside.py", "dir_outside": "dir_outside",
        "file_dangling": "file_dangling.py", "dir_dangling": "dir_dangling",
        "file_loop": "file_loop_a", "dir_loop": "dir_loop_a",
    }[case]
    if case == "file_inside":
        expected = _expected([rows[0]], scope=scope, size=12)
        assert navigation.search_repository(root, "needle", scope, mode="literal") == expected
    elif case == "dir_inside":
        assert navigation.search_repository(root, "needle", scope, mode="literal",
                                            requested_page_size=1) == _expected(rows, scope=scope)
    elif case.endswith("outside"):
        with pytest.raises(ValueError, match="scope must be inside"):
            navigation.search_repository(root, "needle", scope, mode="literal")
    elif case.endswith("dangling"):
        with pytest.raises(FileNotFoundError):
            navigation.search_repository(root, "needle", scope, mode="literal")
    else:
        with pytest.raises(FileNotFoundError):
            navigation.search_repository(root, "needle", scope, mode="literal")


def test_pinned_read_rejects_file_replaced_by_outside_link(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    root = tmp_path / "workspace"
    nested = root / "nested"
    nested.mkdir(parents=True)
    victim = nested / "victim.py"
    victim.write_text("inside\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("outside_token\n", encoding="utf-8")
    consumers = WorkspaceConsumers(WorkspaceState(
        WorkspaceFence.create(root, tmp_path / "fence")))
    bind_consumers(root, consumers)
    read_text = navigation._read_text
    changed = False

    def replace_after_listing(path, *args, **kwargs):
        nonlocal changed
        if root / path == victim and not changed:
            changed = True
            victim.unlink()
            victim.symlink_to(outside)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(navigation, "_read_text", replace_after_listing)
    page = navigation.search_repository(root, "outside_token", mode="literal")
    assert changed
    assert page.metadata["total"] == 0
    assert page.metadata["unreadable_files"] == 1
    assert page.metadata["scan_complete"] is False
    assert "nested/victim.py:1:outside_token" not in page.text


def test_pinned_read_rejects_parent_replaced_by_outside_link(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    root = tmp_path / "workspace"
    nested = root / "outer" / "nested"
    nested.mkdir(parents=True)
    victim = nested / "victim.py"
    victim.write_text("inside\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.py").write_text("outside_token\n", encoding="utf-8")
    consumers = WorkspaceConsumers(WorkspaceState(
        WorkspaceFence.create(root, tmp_path / "fence")))
    bind_consumers(root, consumers)
    read_text = navigation._read_text
    changed = False

    def replace_after_listing(path, *args, **kwargs):
        nonlocal changed
        if root / path == victim and not changed:
            changed = True
            nested.rename(nested.with_name("saved"))
            nested.symlink_to(outside, target_is_directory=True)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(navigation, "_read_text", replace_after_listing)
    page = navigation.search_repository(root, "outside_token", mode="literal")
    assert changed
    assert page.metadata["total"] == 0
    assert page.metadata["unreadable_files"] == 1
    assert page.metadata["scan_complete"] is False
    assert "outer/nested/victim.py:1:outside_token" not in page.text


def test_fallback_rejects_root_replaced_by_outside_link(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "inside.py").write_text("inside\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("outside_token\n", encoding="utf-8")
    _mode(root, tmp_path, monkeypatch, "fallback")
    walk = navigation._fd_paths
    changed = False

    def replace_before_root_open(path, start, errors):
        nonlocal changed
        if not changed:
            changed = True
            path.rename(path.with_name("saved-workspace"))
            path.symlink_to(outside, target_is_directory=True)
        return walk(path, start, errors)

    monkeypatch.setattr(navigation, "_fd_paths", replace_before_root_open)
    page = navigation.search_repository(root, "outside_token", mode="literal")
    assert changed
    assert page.metadata["total"] == 0
    assert page.metadata["unreadable_files"] == 1
    assert page.metadata["scan_complete"] is False
    assert "secret.py:1:outside_token" not in page.text


def test_pinned_search_does_not_grow_fds_over_500_searches(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "a.py").write_text("needle\n", encoding="utf-8")
    consumers = WorkspaceConsumers(WorkspaceState(
        WorkspaceFence.create(root, tmp_path / "fence")))
    bind_consumers(root, consumers)
    with consumers.state.pin("fd-check"):
        pass
    descriptors = Path("/proc/self/fd")
    if not descriptors.is_dir():
        descriptors = Path("/dev/fd")
    if not descriptors.is_dir():
        pytest.skip("descriptor listing is unavailable")
    before = len(os.listdir(descriptors))
    for index in range(500):
        navigation.search_repository(root, f"needle-{index}", mode="literal")
    assert len(os.listdir(descriptors)) <= before


@pytest.fixture
def busy_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    root = tmp_path / "workspace"
    root.mkdir()
    consumers = WorkspaceConsumers(WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence")))
    bind_consumers(root, consumers)

    @contextmanager
    def busy(_boundary):
        raise WorkspaceUnavailable("writer busy")
        yield

    monkeypatch.setattr(consumers.state, "pin", busy)
    return root


@pytest.mark.parametrize("replacement", ["outside_link", "deleted"])
def test_busy_fallback_does_not_read_a_file_changed_after_listing(
        busy_workspace, monkeypatch, replacement):
    root = busy_workspace
    victim = root / "victim.py"
    victim.write_text("inside\n", encoding="utf-8")
    outside = root.parent / "outside.py"
    outside.write_text("secret_token\n", encoding="utf-8")
    listed = changed = False
    walk, stat = os.walk, os.stat

    def mark_walk(*args, **kwargs):
        nonlocal listed
        for row in walk(*args, **kwargs):
            listed = True
            yield row

    def change_after_listing(path, *args, **kwargs):
        nonlocal changed
        result = stat(path, *args, **kwargs)
        directory = kwargs.get("dir_fd")
        candidate = ((path == "victim.py" and directory is not None)
                     or (listed and isinstance(path, (str, bytes, os.PathLike)) and Path(path) == victim))
        if candidate and not changed:
            changed = True
            victim.unlink()
            if replacement == "outside_link":
                victim.symlink_to(outside)
        return result

    monkeypatch.setattr(os, "walk", mark_walk)
    monkeypatch.setattr(os, "stat", change_after_listing)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {change_after_listing})
    page = navigation.search_repository(root, "secret_token", mode="literal")

    assert changed
    assert page.metadata["total"] == 0
    assert page.metadata["unreadable_files"] == 1
    assert page.metadata["scan_complete"] is False
    assert "victim.py:1:secret_token" not in page.text


def test_busy_fallback_keeps_link_hardlink_unreadable_and_deep_tree_behavior(
        busy_workspace, monkeypatch):
    root = busy_workspace
    (root / "nested").mkdir()
    source = root / "inside.py"
    source.write_text("needle\n", encoding="utf-8")
    (root / "hard.py").hardlink_to(source)
    (root / "file_link.py").symlink_to(source)
    (root / "dir_link").symlink_to("nested", target_is_directory=True)
    deep = root
    for index in range(40):
        deep = deep / f"d{index:02d}"
        deep.mkdir()
    (deep / "found.py").write_text("needle\n", encoding="utf-8")
    blocked = root / "blocked.py"
    blocked.write_text("needle\n", encoding="utf-8")
    open_file = os.open

    def deny_blocked(path, *args, **kwargs):
        if path == blocked.name and kwargs.get("dir_fd") is not None:
            raise PermissionError("unreadable control")
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny_blocked)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {deny_blocked})
    page = navigation.search_repository(root, "needle", mode="literal", requested_page_size=1)
    assert page.metadata["total"] == 3
    assert page.metadata["unreadable_files"] == 1
    assert page.metadata["scan_complete"] is False
    assert "file_link.py" not in page.text
    assert "dir_link" not in page.text
    assert "d39/found.py:1:needle" in navigation.search_repository(
        root, "needle", mode="literal", cursor=page.metadata["continuation"],
        requested_page_size=500,
    ).text
    scoped = navigation.search_repository(root, "needle", "file_link.py", mode="literal")
    assert scoped.metadata["total"] == 1
    assert "inside.py:1:needle" in scoped.text


def test_busy_fallback_reports_directory_replaced_after_listing(busy_workspace, monkeypatch):
    root = busy_workspace
    (root / "swap").mkdir()
    outside = root.parent / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("outside_token\n", encoding="utf-8")
    stat = os.stat
    changed = False

    def replace_after_stat(path, *args, **kwargs):
        nonlocal changed
        result = stat(path, *args, **kwargs)
        if path == "swap" and kwargs.get("dir_fd") is not None and not changed:
            changed = True
            (root / "swap").rmdir()
            (root / "swap").symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(os, "stat", replace_after_stat)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {replace_after_stat})
    page = navigation.search_repository(root, "outside_token", mode="literal")
    assert changed
    assert page.metadata["total"] == 0
    assert page.metadata["unreadable_files"] == 1
    assert page.metadata["scan_complete"] is False
    assert "swap/secret.py" not in page.text


@pytest.mark.parametrize("mode", ["pinned", "fallback", "unpinned"])
@pytest.mark.parametrize("search_mode,query", [("literal", "needle"), ("path", ".py")])
def test_missing_dir_fd_support_reads_nothing(tmp_path, monkeypatch, mode, search_mode, query):
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "on")
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "hit.py").write_text("needle\n")
    consumers = _mode(root, tmp_path, monkeypatch, mode)
    opened = []
    read_bytes = Path.read_bytes

    def observe_read(path):
        if path.is_relative_to(root):
            opened.append(path)
        return read_bytes(path)

    if mode == "pinned":
        pin = consumers.state.pin

        def observe_pin(*args, **kwargs):
            opened.append("pin")
            return pin(*args, **kwargs)

        monkeypatch.setattr(consumers.state, "pin", observe_pin)
    monkeypatch.setattr(os, "supports_dir_fd", set())
    monkeypatch.setattr(Path, "read_bytes", observe_read)
    page = navigation.search_repository(root, query, mode=search_mode)
    assert opened == []
    assert page.metadata["total"] == 0
    assert page.metadata["unreadable_files"] == 1
    assert page.metadata["scan_complete"] is False
    assert page.metadata["result"] == "PARTIAL"


@pytest.mark.parametrize("failure", ["fstat", "dup"])
def test_descriptor_closed_on_failed_fstat_or_dup(busy_workspace, monkeypatch, failure):
    root = busy_workspace
    (root / "hit.py").write_text("needle\n")
    original = getattr(os, failure)
    fstat, close = os.fstat, os.close
    target = os.stat(root / "hit.py" if failure == "fstat" else root)
    failed = []

    def fail(fd):
        if os.path.samestat(fstat(fd), target):
            failed.append(fd)
            raise OSError(f"injected {failure} failure")
        return original(fd)

    try:
        with monkeypatch.context() as fault:
            fault.setattr(os, failure, fail)
            page = navigation.search_repository(root, "needle", mode="literal")
        assert len(failed) == 1
        with pytest.raises(OSError):
            fstat(failed[0])
        assert page.metadata["total"] == 0
        assert page.metadata["unreadable_files"] == 1
        assert page.metadata["scan_complete"] is False
    finally:
        for fd in failed:
            try:
                close(fd)
            except OSError:
                pass


@pytest.mark.parametrize("low_limit", [False, True])
def test_search_walks_1000_levels_with_bounded_descriptors(busy_workspace, low_limit):
    import resource

    root = busy_workspace
    directories = [root]
    for _ in range(1000):
        directory = directories[-1] / "d"
        directory.mkdir()
        directories.append(directory)
    hit = directories[-1] / "hit.py"
    hit.write_text("needle\n")
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        if low_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, limits[1]))
        page = navigation.search_repository(root, "needle", mode="literal")
        assert page == _expected([
            {"path": "d/" * 1000 + "hit.py", "line": 1, "text": "needle"},
        ], size=12)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, limits)
        hit.unlink()
        for directory in reversed(directories[1:]):
            directory.rmdir()


@pytest.mark.parametrize("mode", ["pinned", "fallback", "unpinned"])
def test_search_passes_plain_relative_path_to_reader(tmp_path, monkeypatch, mode):
    root = tmp_path / "workspace"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "hit.py").write_text("needle\n")
    _mode(root, tmp_path, monkeypatch, mode)
    read_text = navigation._read_text

    def check_path(path, *args, **kwargs):
        assert type(path) is str
        assert (root / path).relative_to(root) == Path("nested/hit.py")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(navigation, "_read_text", check_path)
    assert navigation.search_repository(root, "needle", mode="literal").metadata["total"] == 1


@pytest.mark.parametrize("mode", ["pinned", "fallback", "unpinned"])
@pytest.mark.parametrize("search_mode,query", [("literal", "needle"), ("path", ".py")])
def test_search_escapes_non_utf8_names_and_keeps_other_hits(tmp_path, monkeypatch, mode,
                                                        search_mode, query):
    root = tmp_path / "workspace"
    root.mkdir()
    with open(os.fsencode(root) + b"/bad\xff.py", "wb") as stream:
        stream.write(b"needle\n")
    (root / "good.py").write_text("needle\n")
    _mode(root, tmp_path, monkeypatch, mode)

    page = navigation.search_repository(root, query, mode=search_mode, requested_page_size=1)
    assert page.metadata["total"] == 2
    assert page.metadata.get("scan_complete", True)
    assert r"bad\udcff.py" in page.text
    page.text.encode("utf-8")
    second = navigation.search_repository(root, query, mode=search_mode,
                                          cursor=page.metadata["continuation"])
    assert "good.py" in second.text


@pytest.mark.parametrize("mode", ["pinned", "fallback", "unpinned"])
def test_search_discards_file_if_parent_moves_out_during_read(tmp_path, monkeypatch, mode):
    root = tmp_path / "workspace"
    parent = root / "outer" / "nested"
    parent.mkdir(parents=True)
    victim = parent / "victim.py"
    victim.write_text("needle moved\n")
    (root / "kept.py").write_text("needle kept\n")
    consumers = _mode(root, tmp_path, monkeypatch, mode)
    if mode == "pinned":
        with consumers.state.pin("fixture"):
            pass
    info = victim.stat()
    changed = False
    read = os.read

    def move_after_read(fd, size):
        nonlocal changed
        data = read(fd, size)
        if data and not changed and os.path.samestat(os.fstat(fd), info):
            changed = True
            parent.rename(tmp_path / "moved")
            parent.mkdir()
            victim.write_text("replacement\n")
        return data

    monkeypatch.setattr(os, "read", move_after_read)
    page = navigation.search_repository(root, "needle", mode="literal")
    assert changed
    assert page.metadata["total"] == 1
    assert page.metadata["unreadable_files"] == 1
    assert page.metadata["scan_complete"] is False
    assert "kept.py:1:needle kept" in page.text
    assert "needle moved" not in page.text

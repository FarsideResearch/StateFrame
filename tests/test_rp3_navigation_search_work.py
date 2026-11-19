"""Work and output contracts for legacy scoped grep."""
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import pytest

import kernel_navigation_pages as navigation
import kernel_workspace_tree as workspace_tree
from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
from kernel_workspace_state import WorkspaceFence, WorkspaceState


@pytest.fixture
def pinned_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    root = tmp_path / "workspace"
    root.mkdir()
    state = WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence"))
    consumers = WorkspaceConsumers(state)
    bind_consumers(root, consumers)
    return root, consumers


def test_repeated_scoped_grep_work(pinned_workspace, monkeypatch):
    root, consumers = pinned_workspace
    for scope in ("src", "other"):
        for directory in range(15):
            package = root / scope / f"package_{directory:02d}"
            package.mkdir(parents=True)
            for index in range(150):
                (package / f"module_{index:03d}.py").write_text("VALUE = 1\n")
    changed_path = root / "src/package_00/module_000.py"

    # Charge the work the search actually does after other workspace consumers
    # have admitted the tree, including any reconciliation from later writes.
    started = perf_counter()
    with consumers.state.pin("fixture"):
        pass
    work = consumers.state.work
    print(f"shared initial pin: {perf_counter() - started:.6f}s, "
          f"index walks/reads={(work.directories_scanned, work.content_reads)}")

    walks: list[Path] = []
    reads: list[Path] = []
    index_walks = index_reads = projected_dirs = 0
    children = workspace_tree.children
    refresh = consumers.state._index.refresh

    def count_refresh(*args, **kwargs):
        nonlocal index_walks, index_reads
        result = refresh(*args, **kwargs)
        index_work = consumers.state.work
        index_walks += index_work.directories_scanned
        index_reads += index_work.content_reads
        return result

    def count_children(directory):
        nonlocal projected_dirs
        projected_dirs += 1
        return children(directory)

    monkeypatch.setattr(consumers.state._index, "refresh", count_refresh)
    monkeypatch.setattr(workspace_tree, "children", count_children)
    walk = os.walk
    read = Path.read_bytes

    def count_walk(path, *args, **kwargs):
        if Path(path).is_relative_to(root):
            walks.append(Path(path))
        return walk(path, *args, **kwargs)

    def count_read(path):
        if path.is_relative_to(root):
            reads.append(path)
        return read(path)

    monkeypatch.setattr(os, "walk", count_walk)
    monkeypatch.setattr(Path, "read_bytes", count_read)

    def grep(scope, label):
        nonlocal index_walks, index_reads, projected_dirs
        index_walks = index_reads = projected_dirs = 0
        walks.clear()
        reads.clear()
        start = perf_counter()
        result = navigation.search_repository(root, "VALUE", scope, mode="literal")
        seconds = perf_counter() - start
        counts = (len(walks), len(reads), index_walks, index_reads, projected_dirs)
        print(f"{label}: {seconds:.6f}s, "
              f"live walks/grep reads/index walks/index reads/projected directories={counts}")
        return result, counts, seconds

    first, _, _ = grep(".", "first")
    second, steady_work, steady_seconds = grep(".", "repeat")
    before, _, _ = grep("src", "initial src")
    other, _, _ = grep("other", "initial other")
    assert first == second
    assert first.metadata["total"] == 4500
    assert before.metadata["total"] == other.metadata["total"] == 2250

    with consumers.state.begin_write(("src/package_00/module_000.py",)):
        changed_path.write_text("VALUE = 2\n")

    still_other, other_work, _ = grep("other", "repeat other after edit")
    after, edited_work, _ = grep("src", "repeat src after edit")
    edited_reads = reads.copy()
    assert still_other == other
    assert before.metadata["total"] == after.metadata["total"]
    assert before.metadata["result_set_digest"] != after.metadata["result_set_digest"]
    assert "src/package_00/module_000.py:1:VALUE = 2" in after.text

    assert steady_work == (0, 0, 0, 0, 0)
    assert steady_seconds < 2
    assert other_work == (0, 0, 0, 1, 0)
    assert edited_work[:1] == (0,)
    assert edited_reads in ([], [changed_path])
    assert edited_work[2:] == (0, 0, 16)


def _legacy_search(root, monkeypatch, *args, **kwargs):
    with monkeypatch.context() as settings:
        settings.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
        return navigation.search_repository(root, *args, **kwargs)


@pytest.mark.parametrize("coordinates", ["off", "on"])
def test_scoped_paging_matches_legacy_through_topology_changes(pinned_workspace, monkeypatch, coordinates):
    root, consumers = pinned_workspace
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", coordinates)
    (root / "src/.git").mkdir(parents=True)
    (root / "src/nested").mkdir()
    (root / "src/.git/skipped.py").write_text("hit café\n")
    (root / "src/nested/c.py").write_text("hit café\n")
    (root / "src/a.py").write_text("skip\r\nhit café\u2028hit café\n")
    (root / "src/data.py").write_bytes(b"\x00not text")
    (root / "src/bad.py").write_bytes(b"\xff")
    (root / "src/z.txt").write_text("hit café\n")
    (root / "src/inside.py").symlink_to("a.py")
    (root / "inside").symlink_to("src", target_is_directory=True)
    outside = root.parent / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("hit outside\n")
    (root / "src/outside.py").symlink_to(outside / "secret.py")
    (root / "src/outside").symlink_to(outside, target_is_directory=True)

    def compare(pattern, scope, **kwargs):
        actual = navigation.search_repository(root, pattern, scope, **kwargs)
        assert actual == _legacy_search(root, monkeypatch, pattern, scope, **kwargs)
        assert "outside" not in actual.text
        return actual

    args = {"mode": "regex", "file_glob": "*.py", "requested_page_size": 1}
    first = compare(r"hit caf.", "src", **args)
    cursor = first.metadata["continuation"]
    assert first.metadata["total"] == 3
    assert first.metadata["non_text_files"] == 2
    assert first.text.endswith("src/a.py:2:hit café")
    assert compare(r"hit caf.", "src", cursor=cursor, **args).text.endswith("src/a.py:3:hit café")
    assert compare("hit café", "inside", mode="text", file_glob="*.py").metadata["total"] == 3
    assert compare("hit", "src/inside.py", mode="literal").metadata["total"] == 2
    assert compare(".py", "src", mode="filename").metadata["total"] == 4

    with consumers.state.begin_write(("src/z.txt",)):
        (root / "src/z.txt").write_text("a nonmatching edit outside the glob\n")
    assert compare(r"hit caf.", "src", **args) == first
    compare(r"hit caf.", "src", cursor=cursor, **args)

    with consumers.state.begin_write(("src/nested/c.py", "src/nested/d.py")):
        (root / "src/nested/c.py").rename(root / "src/nested/d.py")
    assert compare(r"hit caf.", "src", **args).metadata["result_set_digest"] != first.metadata["result_set_digest"]
    for setting in ("on", "off"):
        with monkeypatch.context() as switches:
            switches.setenv("STATEFRAME_WORKSPACE_OBJECTS", setting)
            with pytest.raises(ValueError, match="continuation is stale"):
                navigation.search_repository(root, r"hit caf.", "src", cursor=cursor, **args)
            with pytest.raises(ValueError, match="scope must be inside"):
                navigation.search_repository(root, "[", "src/outside")
            with pytest.raises(ValueError, match="scope must be inside"):
                navigation.search_repository(root, "hit", "../outside")
            with pytest.raises(FileNotFoundError, match="scope does not exist"):
                navigation.search_repository(root, "hit", "missing")
            with pytest.raises(ValueError, match="invalid regex"):
                navigation.search_repository(root, "[", "src")


@pytest.mark.parametrize("available", [True, False])
def test_unreadable_file_keeps_partial_evidence_and_recovers(pinned_workspace, monkeypatch, available):
    from contextlib import contextmanager

    from kernel_workspace_state import WorkspaceUnavailable

    root, consumers = pinned_workspace
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "on")
    (root / "a.py").write_text("hit\n")
    (root / "b.py").write_text("hit\n")
    blocked = root / "c.py"
    blocked.write_text("miss\n")
    read = Path.read_bytes
    attempts = 0

    def read_denied(path):
        nonlocal attempts
        if path == blocked:
            attempts += 1
            raise PermissionError("cannot read file")
        return read(path)

    @contextmanager
    def unavailable(_boundary):
        raise WorkspaceUnavailable("cannot admit workspace")
        yield

    with monkeypatch.context() as denying:
        denying.setattr(Path, "read_bytes", read_denied)
        if not available:
            denying.setattr(consumers.state, "pin", unavailable)
        args = {"mode": "literal", "requested_page_size": 1}
        first = navigation.search_repository(root, "hit", **args)
        assert first == _legacy_search(root, monkeypatch, "hit", **args)
        assert first.metadata["total"] == 2
        assert first.metadata["scan_complete"] is False
        assert first.metadata["unreadable_files"] == 1
        cursor = first.metadata["continuation"]
        assert navigation.search_repository(root, "hit", cursor=cursor, **args) == _legacy_search(
            root, monkeypatch, "hit", cursor=cursor, **args,
        )
        empty = navigation.search_repository(root, "miss", **args)
        assert empty.metadata["result"] == "PARTIAL"
        assert empty.metadata["total"] == 0
        assert attempts == 5

    # Clearing the read failure without a content change must retry missing evidence.
    assert navigation.search_repository(root, "miss").metadata["total"] == 1
    restored = navigation.search_repository(root, "hit", **args)
    assert restored.metadata["scan_complete"] is True
    assert restored.metadata["continuation"] == cursor


@pytest.mark.parametrize("cap,limit", [
    ("_SEARCH_MAX_ENTRIES", 2), ("_SEARCH_MAX_BYTES", 8), ("_SEARCH_MAX_FILE_BYTES", 4),
])
def test_coverage_cap_reports_partial_and_keeps_canonical_cursor(pinned_workspace, monkeypatch, cap, limit):
    root, _ = pinned_workspace
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "on")
    (root / "a.py").write_text("hit\n")
    (root / "b.py").write_text("hit\n")
    (root / "c.py").write_text("hit and more\n")
    monkeypatch.setattr(navigation, cap, limit, raising=False)

    args = {"mode": "literal", "requested_page_size": 1}
    page = navigation.search_repository(root, "hit", **args)
    assert page.metadata["scan_complete"] is False
    assert page.metadata["coverage_limited"] is True
    assert page.metadata["unreadable_files"] == 0
    assert page.metadata["result"] == "HITS"
    assert page.metadata["total"] == 2
    assert "coverage limit reached" in page.text
    assert page.metadata["result_set_digest"] == navigation._fingerprint([
        {"path": "a.py", "line": 1, "text": "hit"},
        {"path": "b.py", "line": 1, "text": "hit"},
    ])
    cursor = page.metadata["continuation"]
    assert navigation.search_repository(root, "hit", **args) == page
    assert navigation.search_repository(root, "hit", cursor=cursor, **args).text.endswith("b.py:1:hit")
    with pytest.raises(ValueError, match="continuation is stale"):
        _legacy_search(root, monkeypatch, "hit", cursor=cursor, **args)
    empty = navigation.search_repository(root, "absent")
    assert empty.metadata["result"] == "PARTIAL"
    assert empty.metadata["scan_complete"] is False


def test_large_result_retains_exact_pages_and_digest(pinned_workspace, monkeypatch):
    root, _ = pinned_workspace
    (root / "a.py").write_text("hit\n" * 100)
    monkeypatch.setattr(navigation, "_SEARCH_RESULT_BYTES", 96, raising=False)
    args = {"mode": "literal", "requested_page_size": 2}
    first = navigation.search_repository(root, "hit", **args)
    assert first == _legacy_search(root, monkeypatch, "hit", **args)
    cursor = first.metadata["continuation"]
    expected_next = _legacy_search(root, monkeypatch, "hit", cursor=cursor, **args)

    def unexpected_read(_path):
        pytest.fail("retained query evidence should avoid rereading unchanged contents")

    monkeypatch.setattr(navigation, "_read_text", unexpected_read)
    assert navigation.search_repository(root, "hit", **args) == first
    assert navigation.search_repository(root, "hit", cursor=cursor, **args) == expected_next
    assert navigation.search_repository(root, "hit", cursor=cursor, **args) == expected_next

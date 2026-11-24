"""Fallback grep keeps only the requested page while preserving base semantics."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
from pathlib import Path

import pytest

import kernel_navigation_pages as navigation


_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".tox", ".venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".test-tmp", "node_modules", "dist", "build",
})


def _plain_files(root: Path, scope: object) -> tuple[tuple[str, Path], ...]:
    root = root.resolve()
    requested = root if scope in (None, "", ".") else Path(str(scope))
    candidate = requested if requested.is_absolute() else root / requested
    start = candidate.resolve()
    if not start.is_relative_to(root):
        raise ValueError("navigation scope must be inside the repository root")
    if not start.exists():
        raise FileNotFoundError(f"scope does not exist: {scope or root}")
    if start.is_file():
        return ((start.relative_to(root).as_posix(), start),)

    files: list[tuple[str, Path]] = []
    for current, directories, names in os.walk(start, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name for name in directories
            if name not in _SKIP_DIRS and not (current_path / name).is_symlink()
        )
        for name in sorted(names):
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                continue
            files.append((path.resolve().relative_to(root).as_posix(), path))
    return tuple(files)


def _plain_offset(cursor: object, digest: str) -> int:
    if cursor in (None, ""):
        return 0
    parts = str(cursor).split(":")
    if len(parts) != 3 or parts[0] != "nav1" or parts[1] != digest:
        raise ValueError("navigation continuation is stale; repeat the search")
    try:
        offset = int(parts[2])
    except ValueError as error:
        raise ValueError("navigation continuation is invalid; repeat the search") from error
    if offset < 0:
        raise ValueError("navigation continuation is invalid; repeat the search")
    return offset


def _plain_search(root: Path, pattern: str, scope: object = None, *, mode: str = "regex",
                  file_glob: str | None = None, cursor: object = None,
                  page_size: int = 12) -> dict[str, object]:
    """Straightforward all-rows reference, deliberately independent of production helpers."""
    root = root.resolve()
    normalized_mode = str(mode or "regex").strip().casefold()
    if normalized_mode in {"literal", "text"}:
        normalized_mode = "literal"
    elif normalized_mode in {"path", "filename", "file"}:
        normalized_mode = "path"
    elif normalized_mode != "regex":
        raise ValueError("grep mode must be regex, literal, or path")

    files = _plain_files(root, scope)
    if normalized_mode == "path":
        matcher = None
    else:
        try:
            matcher = re.compile(re.escape(pattern) if normalized_mode == "literal" else pattern)
        except re.error as error:
            raise ValueError(f"invalid regex: {error}") from error

    rows: list[dict[str, object]] = []
    unreadable = non_text = 0
    for relative, path in files:
        if file_glob and not (fnmatch.fnmatch(relative, file_glob)
                              or fnmatch.fnmatch(relative.rsplit("/", 1)[-1], file_glob)):
            continue
        if matcher is None:
            if pattern in relative:
                rows.append({"path": relative})
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            unreadable += 1
            continue
        if b"\x00" in raw:
            non_text += 1
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            non_text += 1
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if matcher.search(line):
                rows.append({"path": relative, "line": line_number, "text": line})

    encoded = json.dumps(rows, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:24]
    offset = _plain_offset(cursor, digest)
    shown = rows[offset:offset + page_size]
    remaining = max(0, len(rows) - (offset + len(shown)))
    continuation = f"nav1:{digest}:{offset + len(shown)}" if remaining else None
    metadata: dict[str, object] = {
        "schema": "navigation-page/1", "result_set_digest": digest,
        "shown": len(shown), "total": len(rows), "offset": offset,
        "remaining": remaining, "complete": continuation is None,
        "continuation": continuation,
        "mode": normalized_mode, "pattern": pattern, "scope": str(scope or "."),
        "file_glob": file_glob, "page_size": page_size,
    }
    if normalized_mode != "path":
        metadata.update({"scan_complete": unreadable == 0,
                         "unreadable_files": unreadable, "non_text_files": non_text})

    if normalized_mode == "path":
        rendered = [str(row["path"]) for row in shown]
    else:
        rendered = [f"{row['path']}:{row['line']}:{row['text']}" for row in shown]
    if not metadata["total"]:
        summary = (f"No hits for {normalized_mode} search: pattern={pattern!r} "
                   f"scope={metadata['scope']!r}")
        if file_glob:
            summary += f" file_glob={file_glob!r}"
        summary += "."
        if normalized_mode == "literal" and "|" in pattern:
            summary += " The '|' character was matched literally."
        rendered.append(summary)

    scan_status = f" scan_complete={str(metadata['scan_complete']).lower()}" \
        if "scan_complete" in metadata else ""
    header = (f"[search page] shown={metadata['shown']} total={metadata['total']} "
              f"offset={metadata['offset']} complete={str(metadata['complete']).lower()}"
              f"{scan_status}")
    if continuation:
        next_line = f"{remaining} more, page with cursor={continuation}"
    else:
        next_line = "0 more; complete=true"
    if metadata.get("scan_complete") is False:
        next_line += (f"; scan incomplete ({metadata.get('unreadable_files', 0)} unreadable files); "
                      "do not treat this page as repository-wide absence")
    return {"text": "\n".join([header, next_line, *rendered]), "metadata": metadata}


@pytest.fixture
def search_tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "scope/subdir").mkdir(parents=True)
    (root / "a.txt").write_text(
        "MATCH alpha\nMATCH beta\nplain literal with spaces\n", encoding="utf-8")
    (root / "b.py").write_text(
        "MATCH source\nregex left\nline with Ω\n", encoding="utf-8")
    (root / "scope/subdir/target.py").write_text(
        "MATCH target\nfoo bar\nMATCH Ω\n", encoding="utf-8")
    (root / "unicode-Ω.txt").write_text("MATCH Café\n", encoding="utf-8")
    (root / "line-endings.txt").write_text(
        "before\rMATCH cr\nMATCH lf\r\nMATCH crlf\vMATCH vt\fMATCH ff"
        "\x1cMATCH fs\x1dMATCH gs\x1eMATCH rs\x85MATCH nel"
        "\u2028MATCH ls\u2029MATCH ps\n",
        encoding="utf-8")
    (root / "unreadable.txt").write_text("MATCH secret\n", encoding="utf-8")
    (root / "empty.txt").write_bytes(b"")
    (root / "binary.bin").write_bytes(b"binary\x00MATCH\n")
    (root / "invalid.bin").write_bytes(b"\xffMATCH\n")
    return root


def _open_trace(monkeypatch, root: Path, *, fail_names: frozenset[str] = frozenset()):
    root = root.resolve()
    slots = {"reference": [], "actual": []}
    active = {"slot": "actual"}
    original_read_bytes = Path.read_bytes
    original_os_open = os.open
    directory_paths: dict[tuple[int, int], str] = {}

    def relative_path(path, dir_fd=None) -> str | None:
        candidate = Path(os.fsdecode(path))
        if candidate.is_absolute():
            try:
                return candidate.relative_to(root).as_posix()
            except ValueError:
                return None
        if dir_fd is None:
            return None
        try:
            info = os.fstat(dir_fd)
        except OSError:
            return None
        parent = directory_paths.get((info.st_dev, info.st_ino))
        return (Path(parent) / candidate).as_posix() if parent is not None else None

    def observe(relative: str | None) -> None:
        if relative is None or relative == ".":
            return
        slots[active["slot"]].append(relative)
        if relative in fail_names:
            raise PermissionError("fixture denies this source")

    def traced_read(path: Path) -> bytes:
        observe(relative_path(path))
        return original_read_bytes(path)

    def traced_os_open(path, flags, mode=0o777, *, dir_fd=None):
        relative = relative_path(path, dir_fd)
        directory = bool(flags & getattr(os, "O_DIRECTORY", 0))
        if not directory:
            observe(relative)
        descriptor = original_os_open(path, flags, mode, dir_fd=dir_fd)
        if directory and relative is not None:
            info = os.fstat(descriptor)
            directory_paths[(info.st_dev, info.st_ino)] = relative
        return descriptor

    monkeypatch.setattr(Path, "read_bytes", traced_read)
    monkeypatch.setattr(os, "open", traced_os_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {traced_os_open})
    return active, slots


@pytest.mark.parametrize("page_size", [1, 2, 3, 7])
@pytest.mark.parametrize(("pattern", "mode", "scope", "file_glob"), [
    pytest.param("MATCH", "literal", None, None, id="literal-all-pages"),
    pytest.param("MATCH|foo", "regex", None, None, id="regex-all-pages"),
    pytest.param("target", "path", None, None, id="path-all-pages"),
    pytest.param("space literal", "literal", None, None, id="literal-with-space"),
    pytest.param("Ω|Café", "regex", None, None, id="unicode-regex"),
    pytest.param("MATCH", "literal", "scope", "*.py", id="scoped-glob"),
    pytest.param("MATCH", "regex", "scope/subdir/target.py", None, id="scoped-file"),
    pytest.param("MATCH", "literal", "line-endings.txt", None, id="splitlines-boundaries"),
    pytest.param("missing|absent", "literal", "scope", "*.txt", id="empty-page"),
    pytest.param("scope", "path", None, "*.py", id="path-glob"),
])
def test_pages_match_independent_plain_reference(
        search_tree: Path, monkeypatch, page_size: int, pattern: str, mode: str,
        scope: str | None, file_glob: str | None):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "off")
    active, opened = _open_trace(monkeypatch, search_tree)
    cursor = None
    visited = 0
    while True:
        active["slot"] = "reference"
        opened["reference"].clear()
        expected = _plain_search(search_tree, pattern, scope, mode=mode,
                                 file_glob=file_glob, cursor=cursor, page_size=page_size)
        reference_opens = tuple(opened["reference"])

        active["slot"] = "actual"
        opened["actual"].clear()
        actual = navigation.search_repository(
            search_tree, pattern, scope, mode=mode, file_glob=file_glob,
            cursor=cursor, requested_page_size=page_size)
        actual_opens = tuple(opened["actual"])

        assert actual.text.encode("utf-8") == expected["text"].encode("utf-8")
        assert dict(actual.metadata) == expected["metadata"]
        assert actual_opens == reference_opens
        visited += 1
        assert visited <= 32
        cursor = actual.metadata["continuation"]
        if cursor is None:
            break


def test_bad_and_stale_cursors_fail_after_the_same_full_scan(search_tree: Path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "off")
    active, opened = _open_trace(monkeypatch, search_tree)
    offsets_at_validation: list[tuple[str, ...]] = []
    original_offset = navigation._offset

    def traced_offset(cursor: object, digest: str) -> int:
        offsets_at_validation.append(tuple(opened["actual"]))
        return original_offset(cursor, digest)

    monkeypatch.setattr(navigation, "_offset", traced_offset)
    active["slot"] = "reference"
    baseline = _plain_search(search_tree, "MATCH", mode="literal", page_size=2)
    digest = baseline["metadata"]["result_set_digest"]
    bad_cursors = (
        "not-a-navigation-cursor",
        f"nav1:{digest}:not-an-integer",
        f"nav1:{digest}:-1",
        f"nav1:{'f' * 24}:1",
    )
    assert digest != "f" * 24

    for cursor in bad_cursors:
        active["slot"] = "reference"
        opened["reference"].clear()
        with pytest.raises(ValueError) as expected_error:
            _plain_search(search_tree, "MATCH", mode="literal", cursor=cursor, page_size=2)
        expected_opens = tuple(opened["reference"])

        active["slot"] = "actual"
        opened["actual"].clear()
        offsets_at_validation.clear()
        with pytest.raises(ValueError) as actual_error:
            navigation.search_repository(search_tree, "MATCH", mode="literal",
                                         cursor=cursor, requested_page_size=2)
        actual_opens = tuple(opened["actual"])

        assert (type(actual_error.value), str(actual_error.value)) == (
            type(expected_error.value), str(expected_error.value))
        assert actual_opens == expected_opens
        assert offsets_at_validation == [actual_opens]
        assert actual_opens

    valid_beyond_end = f"nav1:{digest}:99999"
    expected = _plain_search(search_tree, "MATCH", mode="literal",
                             cursor=valid_beyond_end, page_size=2)
    actual = navigation.search_repository(search_tree, "MATCH", mode="literal",
                                          cursor=valid_beyond_end, requested_page_size=2)
    assert actual.text.encode("utf-8") == expected["text"].encode("utf-8")
    assert dict(actual.metadata) == expected["metadata"]


def test_only_the_requested_rows_are_retained(search_tree: Path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "off")
    first = navigation.search_repository(search_tree, "MATCH", mode="literal",
                                         requested_page_size=2)
    observed_sizes: list[int] = []
    original_add = navigation._SearchPageAccumulator.add

    def observe_add(accumulator, row):
        original_add(accumulator, row)
        observed_sizes.append(len(accumulator.rows))

    monkeypatch.setattr(navigation._SearchPageAccumulator, "add", observe_add)
    page = navigation.search_repository(search_tree, "MATCH", mode="literal",
                                        cursor=first.metadata["continuation"],
                                        requested_page_size=2)
    assert page.metadata["shown"] == 2
    assert observed_sizes
    assert max(observed_sizes) <= 2
    assert max(observed_sizes) == 2


def test_stringifiable_cursor_uses_the_same_offset_for_row_window(
        tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "matches.txt").write_text(
        "MATCH zero\nMATCH one\nMATCH two\nMATCH three\n", encoding="utf-8")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "off")
    first = _plain_search(root, "MATCH", mode="literal", page_size=2)
    cursor_text = first["metadata"]["continuation"]

    class CursorLike:
        def __init__(self):
            self.stringify_calls = 0

        def __str__(self) -> str:
            self.stringify_calls += 1
            return str(cursor_text)

    expected_cursor = CursorLike()
    expected = _plain_search(root, "MATCH", mode="literal",
                             cursor=expected_cursor, page_size=2)
    actual_cursor = CursorLike()
    actual = navigation.search_repository(root, "MATCH", mode="literal",
                                          cursor=actual_cursor,
                                          requested_page_size=2)

    assert expected_cursor.stringify_calls == 1
    assert actual_cursor.stringify_calls == 1
    assert actual.text.encode("utf-8") == expected["text"].encode("utf-8")
    assert dict(actual.metadata) == expected["metadata"]


@pytest.mark.skipif(os.name != "posix", reason="surrogate filenames require POSIX filesystem encoding")
def test_serialization_error_is_raised_after_later_files_are_opened(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("MATCH first\n", encoding="utf-8")
    invalid_name = os.fsdecode(b"z-\xff.txt")
    (root / invalid_name).write_text("MATCH invalid path\n", encoding="utf-8")
    (root / "zz-after.txt").write_text("MATCH later\n", encoding="utf-8")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "off")
    active, opened = _open_trace(monkeypatch, root)

    active["slot"] = "reference"
    with pytest.raises(UnicodeEncodeError) as expected_error:
        _plain_search(root, "MATCH", mode="literal")
    reference_opens = tuple(opened["reference"])

    active["slot"] = "actual"
    with pytest.raises(UnicodeEncodeError) as actual_error:
        navigation.search_repository(root, "MATCH", mode="literal")
    actual_opens = tuple(opened["actual"])
    assert (type(actual_error.value), str(actual_error.value),
            actual_error.value.object, actual_error.value.start,
            actual_error.value.end) == (
        type(expected_error.value), str(expected_error.value),
        expected_error.value.object, expected_error.value.start,
        expected_error.value.end)
    assert actual_opens == reference_opens
    assert actual_opens[-1] == "zz-after.txt"


_MEMORY_CHILD = r"""
import json, resource, sys, time
from pathlib import Path
kind, root = sys.argv[1], Path(sys.argv[2])
if kind == "reference":
    from test_grep_retained_rows import _plain_search
    start = time.perf_counter()
    result = _plain_search(root, "MATCH", mode="literal", page_size=12)
    metadata, text = result["metadata"], result["text"]
else:
    from kernel_navigation_pages import search_repository
    start = time.perf_counter()
    result = search_repository(root, "MATCH", mode="literal", requested_page_size=12)
    metadata, text = result.metadata, result.text
elapsed = time.perf_counter() - start
peak = None
if sys.platform.startswith("linux"):
    # ru_maxrss survives exec, so a large parent would set the floor; VmHWM belongs to this process image.
    for line in open("/proc/self/status", encoding="ascii", errors="replace"):
        if line.startswith("VmHWM:"):
            peak = int(line.split()[1])
if peak is None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform == "darwin":
    peak //= 1024
print(json.dumps({"elapsed_s": elapsed, "peak_rss_kib": peak,
                  "total": metadata["total"], "shown": metadata["shown"],
                  "digest": metadata["result_set_digest"],
                  "text_bytes": len(text.encode("utf-8"))}, sort_keys=True))
"""


def _measure_memory(kind: str, root: Path) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("STATEFRAME_WORKSPACE_OBJECTS", None)
    environment["STATEFRAME_OWNER_COORDS"] = "off"
    environment["PYTHONPATH"] = os.pathsep.join((
        str(repository / "src"), str(repository / "src/stateframe/engine"),
        str(repository / "tests"),
    ))
    completed = subprocess.run(
        [sys.executable, "-c", _MEMORY_CHILD, kind, str(root)],
        cwd=repository, env=environment, text=True, capture_output=True,
        check=True, timeout=120,
    )
    return json.loads(completed.stdout)


def test_peak_memory_is_lower_than_all_rows_reference(tmp_path: Path):
    root = tmp_path / "large"
    root.mkdir()
    line = "MATCH " + "x" * (8192 - len("MATCH "))
    block = (line + "\n") * 128
    with (root / "rows.txt").open("w", encoding="utf-8", newline="") as output:
        for _ in range(250):
            output.write(block)
    assert (root / "rows.txt").stat().st_size == 32_000 * 8193

    reference = _measure_memory("reference", root)
    candidate = _measure_memory("candidate", root)
    for key in ("total", "shown", "digest", "text_bytes"):
        assert candidate[key] == reference[key]
    assert candidate["total"] == 32_000
    assert candidate["peak_rss_kib"] < reference["peak_rss_kib"] * 0.75

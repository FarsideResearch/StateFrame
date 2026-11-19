"""Work-count regression coverage for repository signal inventories."""
from __future__ import annotations

import os
from pathlib import Path
import time

import kernel_repo_signals as signals


_SOURCE_LIMIT = 200_000
_EXCLUDED_DIR_NAMES = ("node_modules", ".generated")


class _Counts:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.excluded_roots = tuple(root / name for name in _EXCLUDED_DIR_NAMES)
        self.scandir_paths: list[Path] = []
        self.excluded_scandir_paths: list[Path] = []
        self.entries_seen = 0
        self.kind_checks = 0
        self.read_sizes: list[int] = []
        self.read_characters = 0

    def summary(self) -> str:
        return (
            f"scandir_calls={len(self.scandir_paths)}, "
            f"excluded_scandir_calls={len(self.excluded_scandir_paths)}, "
            f"entries_seen={self.entries_seen}, kind_checks={self.kind_checks}, "
            f"read_sizes={self.read_sizes}, read_characters={self.read_characters}"
        )


class _TrackedEntry:
    def __init__(self, entry: os.DirEntry[str], counts: _Counts) -> None:
        self._entry = entry
        self._counts = counts

    def __getattr__(self, name: str) -> object:
        return getattr(self._entry, name)

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        self._counts.kind_checks += 1
        return self._entry.is_dir(follow_symlinks=follow_symlinks)

    def is_file(self, *, follow_symlinks: bool = True) -> bool:
        self._counts.kind_checks += 1
        return self._entry.is_file(follow_symlinks=follow_symlinks)

    def is_symlink(self) -> bool:
        self._counts.kind_checks += 1
        return self._entry.is_symlink()


class _TrackedScandir:
    def __init__(self, iterator: os.ScandirIterator[str], counts: _Counts) -> None:
        self._iterator = iterator
        self._counts = counts

    def __enter__(self) -> _TrackedScandir:
        return self

    def __exit__(self, *_args: object) -> None:
        self._iterator.close()

    def __iter__(self) -> _TrackedScandir:
        return self

    def __next__(self) -> _TrackedEntry:
        entry = next(self._iterator)
        self._counts.entries_seen += 1
        return _TrackedEntry(entry, self._counts)


class _TrackedTextStream:
    def __init__(self, stream, counts: _Counts) -> None:
        self._stream = stream
        self._counts = counts

    def __enter__(self) -> _TrackedTextStream:
        self._stream.__enter__()
        return self

    def __exit__(self, *args: object) -> object:
        return self._stream.__exit__(*args)

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)

    def read(self, size: int = -1) -> str:
        self._counts.read_sizes.append(size)
        text = self._stream.read(size)
        self._counts.read_characters += len(text)
        return text


def _install_work_counters(monkeypatch, source: Path, counts: _Counts) -> None:
    original_scandir = os.scandir

    def tracked_scandir(path: str | os.PathLike[str] = ".") -> _TrackedScandir:
        scanned = Path(path).resolve()
        counts.scandir_paths.append(scanned)
        if any(scanned == excluded or scanned.is_relative_to(excluded)
               for excluded in counts.excluded_roots):
            counts.excluded_scandir_paths.append(scanned)
        return _TrackedScandir(original_scandir(path), counts)

    original_is_file = Path.is_file
    original_is_symlink = Path.is_symlink

    def tracked_is_file(path: Path, *args: object, **kwargs: object) -> bool:
        counts.kind_checks += 1
        return original_is_file(path, *args, **kwargs)

    def tracked_is_symlink(path: Path) -> bool:
        counts.kind_checks += 1
        return original_is_symlink(path)

    original_open = Path.open

    def tracked_open(path: Path, *args: object, **kwargs: object):
        stream = original_open(path, *args, **kwargs)
        if path == source:
            return _TrackedTextStream(stream, counts)
        return stream

    monkeypatch.setattr(os, "scandir", tracked_scandir)
    monkeypatch.setattr(Path, "is_file", tracked_is_file)
    monkeypatch.setattr(Path, "is_symlink", tracked_is_symlink)
    monkeypatch.setattr(Path, "open", tracked_open)


def _populate_repo(root: Path) -> tuple[Path, str]:
    source = root / "src" / "main.py"
    source.parent.mkdir(parents=True)
    content = "π = 1\n" * 2_000_000
    source.write_text(content, encoding="utf-8")
    try:
        (source.parent / "alias.py").symlink_to(source.name)
    except (NotImplementedError, OSError):
        pass

    (root / "docs").mkdir()
    (root / "docs" / "README.md").write_text("notes\n", encoding="utf-8")
    (root / ".env").write_text("MODE=local\n", encoding="utf-8")

    for directory in _EXCLUDED_DIR_NAMES:
        excluded = root / directory
        excluded.mkdir()
        for index in range(2_500):
            (excluded / f"generated_{index:04}.py").touch()
    return source, content


def test_repo_signal_inventory_work_is_bounded(tmp_path: Path, monkeypatch) -> None:
    source, content = _populate_repo(tmp_path)
    counts = _Counts(tmp_path.resolve())
    _install_work_counters(monkeypatch, source, counts)

    started = time.perf_counter()
    source_paths = signals._source_paths(tmp_path)
    workspace_paths = signals._workspace_paths(tmp_path)
    inventory = signals._source_inventory(tmp_path)
    elapsed = time.perf_counter() - started

    assert source_paths == [source]
    assert workspace_paths == [
        tmp_path / ".env",
        tmp_path / "docs" / "README.md",
        source,
    ]
    assert inventory == {"src/main.py": content[:_SOURCE_LIMIT]}

    violations = []
    if counts.excluded_scandir_paths:
        violations.append(
            f"descended into excluded directories: {counts.excluded_scandir_paths[:4]}"
        )
    if counts.entries_seen > 100:
        violations.append(f"visited {counts.entries_seen} directory entries; budget is 100")
    if counts.kind_checks > 100:
        violations.append(f"made {counts.kind_checks} file-kind checks; budget is 100")
    if counts.read_sizes != [_SOURCE_LIMIT]:
        violations.append(f"source reads were {counts.read_sizes}; expected [{_SOURCE_LIMIT}]")
    if counts.read_characters != _SOURCE_LIMIT:
        violations.append(
            f"read {counts.read_characters} source characters; budget is {_SOURCE_LIMIT}"
        )
    if elapsed >= 10.0:
        violations.append(f"inventory took {elapsed:.3f}s; budget is <10.0s")
    assert not violations, (
        f"repo signal work budget exceeded: {violations}; elapsed={elapsed:.3f}s; "
        f"{counts.summary()}"
    )


def test_source_reader_limits_decode_and_handles_errors(
        tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "large.py"
    body = "π = 1\n" * 50_000
    source.write_bytes(body.encode("utf-8") + b"x" * 128_000 + b"\xff")
    assert signals._read_source(source) == (body + "x" * 128_000)[:_SOURCE_LIMIT]

    source.write_bytes(b"\xff" + body.encode("utf-8"))
    assert signals._read_source(source) == ""

    def fail_open(_path: Path, *_args: object, **_kwargs: object):
        raise OSError("source unavailable")

    monkeypatch.setattr(Path, "open", fail_open)
    assert signals._read_source(source) == ""

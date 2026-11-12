"""Test helpers for checking mutation process and artifact cleanup."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import threading
import uuid

import pytest

from kernel_exec_tracker import ProcessTable


_seen_lock = threading.Lock()
_seen_launches: dict[str, set[tuple[int, tuple[int, int] | None, str | None]]] = {}


@dataclass(frozen=True)
class LeftoverSnapshot:
    runtime_root: Path
    runtime_entries: frozenset[tuple[str, int, int, int, int]]
    launch_identities: frozenset[tuple[int, tuple[int, int] | None, str | None]]
    processes: frozenset[tuple[int, tuple[int, int], bool]]
    runtime_processes: frozenset[tuple[int, tuple[int, int], str]]
    tmp_entries: frozenset[tuple[str, int, int, int]]
    file_descriptors: frozenset[tuple[int, int, int, int, str]]
    sockets: frozenset[tuple[int, int, int, str]]


def _start_identity(value: object) -> tuple[int, int] | None:
    if type(value) is int:
        return value, 0
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        first = value.get("sec", value.get("seconds"))
        second = value.get("usec", value.get("microseconds", 0))
        try:
            return int(first), int(second)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        parts = value.replace(":", ",").split(",")
        try:
            numbers = tuple(int(part) for part in parts)
        except ValueError:
            return None
        if len(numbers) == 1:
            return numbers[0], 0
        if len(numbers) == 2:
            return numbers[0], numbers[1]
    return None


def _ledger_identities(runtime_root: Path) -> set[tuple[int, tuple[int, int] | None, str | None]]:
    ledger = runtime_root / "launches.jsonl"
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()

    found = set()
    for line in lines:
        try:
            row = json.loads(line)
        except (TypeError, ValueError):
            continue
        identity = row.get("identity") if isinstance(row, dict) else None
        if not isinstance(identity, dict):
            continue
        try:
            pid = int(identity["pid"])
        except (KeyError, TypeError, ValueError):
            continue
        found.add((pid, _start_identity(identity.get("start", identity.get("birth"))),
                   str(identity["boot_id"]) if identity.get("boot_id") is not None else None))
    return found


def _runtime_entries(root: Path) -> frozenset[tuple[str, int, int, int, int]]:
    if not root.exists():
        return frozenset()
    entries: set[tuple[str, int, int, int, int]] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = list(os.scandir(directory))
        except OSError:
            continue
        for child in children:
            path = Path(child.path)
            try:
                info = path.lstat()
            except OSError:
                continue
            relative = path.relative_to(root).as_posix()
            entries.add((relative, info.st_dev, info.st_ino, info.st_mode, info.st_size))
            if path.is_dir() and not path.is_symlink():
                pending.append(path)
    return frozenset(entries)


def _tmp_entries() -> frozenset[tuple[str, int, int, int]]:
    root = Path("/tmp")
    try:
        children = list(os.scandir(root))
    except OSError:
        return frozenset()
    found = set()
    for child in children:
        name = child.name.casefold()
        if not (name.startswith("stateframe-mutate") or name.startswith("mutate")):
            continue
        try:
            info = Path(child.path).lstat()
        except OSError:
            continue
        found.add((child.path, info.st_dev, info.st_ino, info.st_mode))
    return frozenset(found)


def _fd_entries() -> tuple[frozenset[tuple[int, int, int, int, str]],
                            frozenset[tuple[int, int, int, str]]]:
    directory = Path("/proc/self/fd") if Path("/proc/self/fd").is_dir() else Path("/dev/fd")
    try:
        descriptors = list(os.scandir(directory))
    except OSError:
        return frozenset(), frozenset()
    files = set()
    sockets = set()
    for item in descriptors:
        try:
            fd = int(item.name)
            info = os.fstat(fd)
            target = os.readlink(item.path)
        except (OSError, ValueError):
            continue
        fingerprint = (fd, info.st_dev, info.st_ino, info.st_mode, target)
        files.add(fingerprint)
        if target.startswith("socket:["):
            sockets.add((fd, info.st_dev, info.st_ino, target))
    return frozenset(files), frozenset(sockets)


def _inside(path: str, root: Path) -> bool:
    cleaned = path.removesuffix(" (deleted)")
    try:
        Path(cleaned).resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def snapshot_leftovers(runtime_root: Path) -> LeftoverSnapshot:
    """Capture launch identities, runtime files, temp entries, fds and live processes."""
    root = Path(runtime_root).absolute()
    recorded = _ledger_identities(root)
    key = str(root)
    with _seen_lock:
        remembered = _seen_launches.setdefault(key, set())
        remembered.update(recorded)
        all_recorded = frozenset(remembered)

    table = ProcessTable()
    processes = set()
    for pid, started, _boot_id in all_recorded:
        try:
            row = table.identity(pid)
        except OSError:
            continue
        if row is not None and (started is None or row.start == started):
            processes.add((pid, row.start, row.zombie))

    runtime_processes: set[tuple[int, tuple[int, int], str]] = set()
    if sys.platform.startswith("linux"):
        try:
            proc_entries = list(os.scandir("/proc"))
        except OSError:
            proc_entries = []
        for entry in proc_entries:
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            for field in ("cwd", "exe"):
                try:
                    target = os.readlink(f"/proc/{pid}/{field}")
                except OSError:
                    continue
                if _inside(target, root):
                    try:
                        row = table.identity(pid)
                    except OSError:
                        row = None
                    if row is not None:
                        runtime_processes.add((pid, row.start, field))
                    break

    descriptors, sockets = _fd_entries()
    return LeftoverSnapshot(root, _runtime_entries(root), all_recorded, frozenset(processes),
                            frozenset(runtime_processes), _tmp_entries(), descriptors, sockets)


def assert_no_leftovers(before: LeftoverSnapshot, after: LeftoverSnapshot) -> None:
    """Assert that a test left no new process, runtime artifact, temp entry or open fd."""
    problems = []
    added_runtime = after.runtime_entries - before.runtime_entries
    if added_runtime:
        problems.append(f"runtime-root entries remain: {sorted(added_runtime)!r}")
    added_tmp = after.tmp_entries - before.tmp_entries
    if added_tmp:
        problems.append(f"new /tmp stateframe/mutate entries remain: {sorted(added_tmp)!r}")
    added_fds = after.file_descriptors - before.file_descriptors
    if added_fds:
        problems.append(f"test process fds remain open: {sorted(added_fds)!r}")
    added_sockets = after.sockets - before.sockets
    if added_sockets:
        problems.append(f"test process sockets remain open: {sorted(added_sockets)!r}")

    current = {(pid, start, zombie) for pid, start, zombie in after.processes}
    survivors = []
    for pid, started, _boot_id in after.launch_identities | before.launch_identities:
        # A zombie has exited and holds no runtime authority; only its original
        # parent can reap it, and that parent may have been SIGKILLed by this test.
        matches = [row for row in current if row[0] == pid and not row[2]
                   and (started is None or row[1] == started)]
        survivors.extend(matches)
    if survivors:
        problems.append(f"recorded mutation processes remain: {sorted(set(survivors))!r}")
    if after.runtime_processes:
        problems.append(f"processes still reference runtime root: {sorted(after.runtime_processes)!r}")
    assert not problems, "mutation leftovers found:\n" + "\n".join(problems)


@pytest.fixture
def mutation_root(tmp_path: Path) -> Path:
    parent = tmp_path / "runtime"
    parent.mkdir()
    root = parent / "run-acceptance"
    root.mkdir()
    before = snapshot_leftovers(root)
    yield root
    after = snapshot_leftovers(root)
    assert_no_leftovers(before, after)


def sweep_and_assert(runtime_root: Path):
    """Run the public sweeper and verify it removed the root and all recorded owners."""
    from kernel_mutate_sweep import sweep

    before = snapshot_leftovers(runtime_root)
    if runtime_root.is_dir() and not any(runtime_root.iterdir()):
        runtime_root.rmdir()
    report = sweep(runtime_root.parent, run_locks=lambda _run: False)
    assert not report.unresolved, report.unresolved
    after = snapshot_leftovers(runtime_root)
    assert_no_leftovers(before, after)
    assert not runtime_root.exists(), "mutation sweeper left the runtime root behind"
    return report


def stage_test_unit(build, runtime_root: Path, source: str, *, name: str = "test_reader",
                    input_schema: dict | None = None, output_schema: dict | None = None,
                    network: bool = False, writes: bool = False, smoke: list | None = None,
                    limits: dict | None = None):
    manifest = {
        "schema": "stateframe-mutation-unit/1",
        "name": name,
        "revision": 1,
        "parent": None,
        "entry": "main.py",
        "summary": "Acceptance test unit",
        "kind": "reader",
        "input_schema": input_schema or {"type": "object"},
        "output_schema": output_schema or {"type": "object"},
        "requires": {"network": network, "writes": writes},
        "smoke": smoke if smoke is not None else [{"input": {}, "expect": {}}],
        "limits": limits or {"wall_s": 3, "cpu_s": 2, "mem_mb": 256},
    }
    # The executor requires a fresh runtime root; source bundles are separate inputs.
    return build.stage_unit({"main.py": source.encode("utf-8")}, manifest,
                            runtime_root.parent.parent / "source-bundles")


def make_launch_spec(executor_api, wire, staged, *, role: str = "payload", call: str = "case-1",
                     phase: int = 0, input_value: dict | None = None,
                     input_files: dict[str, Path] | None = None, response_docs: list | None = None,
                     response_files: dict[str, Path] | None = None, network: bool = False,
                     writes: bool = False, limits=None):
    launch = uuid.uuid4().hex
    input_files = input_files or {}
    response_docs = response_docs or []
    response_files = response_files or {}
    input_rows = []
    for ident, path in input_files.items():
        input_rows.append({"id": ident, "path": f"/in/{ident}", "bytes": path.stat().st_size,
                           "name": path.name})
    stdin_doc = wire.build_stdin_doc(
        wire=1,
        call=call,
        launch=launch,
        phase=phase,
        role=role,
        input=input_value or {},
        inputs=input_rows,
        responses=response_docs,
        grants={"network": network, "writes": writes},
        max_fetch=4,
    )
    return executor_api.LaunchSpec(
        role=role,
        call=call,
        launch=launch,
        unit_dir=staged.dir if staged is not None else None,
        inputs=input_files,
        responses=response_files,
        stdin_doc=stdin_doc,
        limits=limits or executor_api.MutationLimits(),
    )

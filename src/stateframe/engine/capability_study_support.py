"""Isolated helpers for the bounded capability study, never enabled by import."""
from __future__ import annotations

import ast
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import time


def split_tests(source: str, public_names: set[str]) -> tuple[str, str]:
    """Preserve setup/helpers exactly; partition complete top-level test functions."""
    tree = ast.parse(source)
    tests = {node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name.startswith("test_")}
    if not public_names or not public_names < tests:
        raise ValueError("public names must be a nonempty proper subset of tests")
    lines = source.splitlines(keepends=True)
    outputs = []
    for names in (public_names, tests - public_names):
        excluded = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_") and node.name not in names:
                    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                    excluded.update(range(start, node.end_lineno + 1))
        rendered = "".join(line for index, line in enumerate(lines, 1)
                           if index not in excluded)
        ast.parse(rendered)
        outputs.append(rendered)
    return outputs[0], outputs[1]


@contextlib.contextmanager
def test_process_lock(path: Path, audit: Path, *, deadline: float | None = None):
    """Serialize every study pytest across independent inference workers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with path.open("a") as lock:
        if deadline is None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        else:
            if not math.isfinite(deadline):
                raise ValueError("test lock deadline must be finite")
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    with audit.open("a") as stream:
                        stream.write(json.dumps({"pid": os.getpid(), "status": "admission_timeout",
                                                 "wait_s": time.monotonic() - started}) + "\n")
                    raise TimeoutError("test lock admission deadline exceeded")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(min(0.05, remaining))
        acquired = time.monotonic()
        try:
            yield
        finally:
            finished = time.monotonic()
            with audit.open("a") as stream:
                stream.write(json.dumps({"pid": os.getpid(), "wait_s": acquired - started,
                                         "held_s": finished - acquired,
                                         "acquired_monotonic": acquired,
                                         "released_monotonic": finished}) + "\n")
            fcntl.flock(lock, fcntl.LOCK_UN)


def install_test_serialization(rr, lock_path: Path, audit_path: Path) -> None:
    # Indirect public probes launch tests outside rr.run_tests. Both entry paths
    # must inherit the same absolute admission lock, including after cwd changes.
    lock_path = lock_path.resolve()
    os.environ["CAPABILITY_TEST_LOCK"] = str(lock_path)
    original = rr.run_tests

    def serialized(*args, **kwargs):
        if os.environ.get("STATEFRAME_TEST_RUN_SPANS", "off").strip().lower() not in {"shadow", "on"}:
            with test_process_lock(lock_path, audit_path):
                return original(*args, **kwargs)
        from kernel_test_run_spans import profile_scope
        with profile_scope() as profile:
            if profile is None:
                with test_process_lock(lock_path, audit_path):
                    return original(*args, **kwargs)
            profile.mark("test_lock_requested")
            wait_span = profile.start_span("test_lock_wait")
            held_span = None
            try:
                with test_process_lock(lock_path, audit_path):
                    profile.end_span(wait_span)
                    profile.mark("test_lock_acquired")
                    held_span = profile.start_span("test_lock_hold")
                    return original(*args, **kwargs)
            finally:
                if held_span is not None:
                    profile.end_span(held_span)
                    profile.mark("test_lock_released")

    rr.run_tests = serialized


def public_parent_row(public_row: dict) -> dict:
    """Use public parents as diagnostic references, including every declared file.

    Legacy AST comparisons require fixed_src. They are never correctness scores
    in this study, and must never receive the private reference implementation.
    """
    import copy
    from build_tasks.multifile_row import normalize_repo_path

    row = copy.deepcopy(public_row)
    files = row.get('files')
    if files is not None:
        if not isinstance(files, dict) or not files:
            raise ValueError('public multi-file row requires a nonempty files mapping')
        normalized = {}
        for raw_path, payload in files.items():
            path = normalize_repo_path(raw_path, field='public files path')
            if path in normalized:
                raise ValueError('duplicate public source path')
            if not isinstance(payload, dict) or not isinstance(payload.get('parent_src'), str):
                raise ValueError('each public file requires parent_src')
            normalized[path] = {'parent_src': payload['parent_src'], 'fixed_src': payload['parent_src']}
        entry = row.get('entry', {})
        if not isinstance(entry, dict):
            raise ValueError('public entry must be a mapping')
        primary = entry.get('path', sorted(normalized)[0])
        primary = normalize_repo_path(primary, field='public entry path')
        if primary not in normalized:
            raise ValueError('public entry is not a declared file')
        row['files'] = normalized
        row['parent_src'] = normalized[primary]['parent_src']
    if not isinstance(row.get('parent_src'), str):
        raise ValueError('public parent source required')
    row['fixed_src'] = row['parent_src']
    return row

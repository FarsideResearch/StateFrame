"""Build the public engine row from an admitted subframe specification."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath

if __package__:
    from .build_tasks.multifile_row import MultiFileRowError, normalize_repo_path
    from .kernel_parallel_projection import render_unit_task
    from .kernel_parallel_records import UnitSpec
else:
    from build_tasks.multifile_row import MultiFileRowError, normalize_repo_path
    from kernel_parallel_projection import render_unit_task
    from kernel_parallel_records import UnitSpec


def _text(start_files: Mapping[str, bytes], path: str) -> str:
    try:
        value = start_files[path]
    except KeyError as error:
        raise ValueError(f"subframe start snapshot is missing {path}") from error
    if not isinstance(value, bytes):
        raise TypeError(f"subframe start snapshot {path} must contain bytes")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"subframe source {path} is not UTF-8") from error


def _validated_check_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        try:
            path = normalize_repo_path(raw_path, field="subframe public-test path")
        except MultiFileRowError as error:
            raise ValueError(f"INVALID_PUBLIC_TEST_PATH: {error}") from error
        if path != raw_path:
            raise ValueError("INVALID_PUBLIC_TEST_PATH: check source path is not normalized")
        if path in seen:
            raise ValueError("INVALID_PUBLIC_TEST_PATH: duplicate check source path")
        filename = PurePosixPath(path).name
        if not (filename.startswith("test_") and filename.endswith(".py")) \
                and not filename.endswith("_test.py"):
            raise ValueError(f"INVALID_PUBLIC_TEST_PATH: check source is not a pytest test file: {path}")
        seen.add(path)
        normalized.append(path)
    return tuple(normalized)


def build_public_row(spec: UnitSpec, start_files: Mapping[str, bytes]) -> dict:
    if not isinstance(spec, UnitSpec):
        raise TypeError("subframe task requires UnitSpec")
    owns = tuple(spec.contract.owns)
    creates = tuple(spec.contract.creates)
    if not owns and not creates:
        raise ValueError("subframe has no owned or created paths")
    for path in creates:
        if path in start_files:
            raise ValueError(f"declared create already exists in start snapshot: {path}")
    files = {path: {"parent_src": _text(start_files, path)} for path in owns}
    check_paths = _validated_check_paths(spec.contract.check.source_paths)
    public_tests = {path: _text(start_files, path) for path in check_paths}
    entry = ({"path": owns[0], "fn": ""} if owns else
             {"create_path": creates[0], "absent_paths": list(creates)})
    row = {
        "task_id": spec.execution_id,
        "sha": spec.execution_id,
        "repo": "build",
        "shape": "multi_file",
        "task_kind": "bug",
        "tier": "USER-REPO",
        "msg": render_unit_task(spec),
        "files": files,
        "scoped_edits": True,
        "allowed_product_paths": list(owns),
        "new_file_roots": list(creates),
        "create_paths": list(creates),
        "public_tests": public_tests,
        "check_selectors": list(spec.contract.check.selectors),
        "entry": entry,
        "path": owns[0] if owns else "",
        "rel": owns[0] if owns else "",
        "fn": "",
        "_skip_db": True,
    }
    if check_paths:
        row["test_path"] = check_paths[0]
        row["test_src"] = public_tests[check_paths[0]]
        row["public_check_paths"] = list(check_paths)
    return row

"""Regression guards for authoritative run snapshots and in-place promotion evidence."""
from __future__ import annotations

import ast
import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from stateframe import request_records, runner, storage_gc, workcopy


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "src/stateframe/runner.py",
    "src/stateframe/_child.py",
    "src/stateframe/resume.py",
    "src/stateframe/shell_host.py",
    "src/stateframe/storage_gc.py",
    "src/stateframe/rehydrate.py",
    "src/stateframe/request_records.py",
)

# These files are part of the durable run record. Journals opened in append mode are intentionally
# excluded; the per-attempt engine.log is included because it currently opens in truncating mode.
DESTINATION_MARKERS = (
    "spec.json",
    "engine-result.json",
    "summary.json",
    "apply-receipt.json",
    "tool-settings.json",
    "skill-settings.json",
    "resume-pending-verified.json",
    "resume-divergence.json",
    "checkpoint-refused.json",
    "child-traceback.txt",
    "engine.log",
    "final.diff",
    "tests-after.txt",
    ".request.json",
    ".route.json",
    ".json.z",
    "env-manifest.json",
    "env-seed.json",
    "requests.pack.xz",
    "requests",
)
EXPECTED_DESTINATIONS = {
    "spec.json",
    "engine-result.json",
    "summary.json",
    "tool-settings.json",
    "skill-settings.json",
    "resume-pending-verified.json",
    "resume-divergence.json",
    "checkpoint-refused.json",
    "child-traceback.txt",
    "engine.log",
    "final.diff",
    "tests-after.txt",
    "env-manifest.json",
    "env-seed.json",
    "requests.pack.xz",
}


class SnapshotWriteSite(NamedTuple):
    module: str
    line: int
    function: str
    destination: str
    operation: str

    def render(self) -> str:
        return f"{self.module}:{self.line} {self.function}: {self.destination} ({self.operation})"


def _assigned_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_assigned_names(item) for item in target.elts))
    return set()


def _destination(expression: ast.AST, aliases: dict[str, str]) -> str | None:
    rendered = ast.unparse(expression)
    for marker in DESTINATION_MARKERS:
        if marker in rendered:
            if marker == "requests":
                return "requests/*.json"
            if marker == ".request.json":
                return "wire/*.request.json"
            if marker == ".route.json":
                return "wire/*.route.json"
            if marker == ".json.z":
                return "checks/*.json.z"
            return marker
    for node in ast.walk(expression):
        if isinstance(node, ast.Name) and node.id in aliases:
            return aliases[node.id]
    return None


def _function_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    names = []
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(parent.name)
        parent = parents.get(parent)
    return ".".join(reversed(names)) or "<module>"


def _find_snapshot_writes(source: str, module: str) -> list[SnapshotWriteSite]:
    """Find writes to named run snapshots, resolving local path aliases in each function."""
    tree = ast.parse(source, filename=module)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    constants = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant):
            if isinstance(statement.value.value, str):
                for target in statement.targets:
                    for name in _assigned_names(target):
                        constants[name] = statement.value.value
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    sites: dict[tuple[int, str, str], SnapshotWriteSite] = {}

    for function in functions:
        aliases: dict[str, str] = dict(constants)
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                destination = _destination(node.value, aliases)
                if destination:
                    for target in node.targets:
                        for name in _assigned_names(target):
                            aliases[name] = destination
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                destination = _destination(node.value, aliases)
                if destination:
                    for name in _assigned_names(node.target):
                        aliases[name] = destination

        for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
            function_expr = call.func
            operation = None
            destination_expr = None
            if isinstance(function_expr, ast.Attribute):
                method = function_expr.attr
                if method in {"write_text", "write_bytes"}:
                    operation = method
                    destination_expr = function_expr.value
                elif method == "_write_json_atomic":
                    operation = "_write_json_atomic"
                    destination_expr = call.args[0] if call.args else None
                elif method == "open":
                    if ast.unparse(function_expr.value) == "os":
                        flags = ast.unparse(call.args[1]) if len(call.args) > 1 else ""
                        if "O_TRUNC" in flags:
                            operation = "os.open(O_TRUNC)"
                            destination_expr = call.args[0] if call.args else None
                    else:
                        mode = call.args[0] if call.args else None
                        if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
                            if "w" in mode.value or "x" in mode.value:
                                operation = f"open({mode.value!r})"
                                destination_expr = function_expr.value
                elif method == "replace":
                    if ast.unparse(function_expr.value) == "os":
                        operation = "os.replace"
                        destination_expr = call.args[1] if len(call.args) > 1 else None
                    else:
                        operation = "path.replace"
                        destination_expr = call.args[0] if call.args else function_expr.value
            elif isinstance(function_expr, ast.Name) and function_expr.id == "open":
                mode = call.args[1] if len(call.args) > 1 else None
                if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
                    if "w" in mode.value or "x" in mode.value:
                        operation = f"open({mode.value!r})"
                        destination_expr = call.args[0] if call.args else None
            elif isinstance(function_expr, ast.Name) and function_expr.id in {
                    "atomic_bytes", "_write_json_atomic", "_write_hook_spec"}:
                operation = function_expr.id
                destination_expr = call.args[0] if call.args else None

            if operation is None or destination_expr is None:
                continue
            destination = _destination(destination_expr, aliases)
            if destination is None:
                continue
            key = (call.lineno, destination, operation)
            sites[key] = SnapshotWriteSite(
                module=module,
                line=call.lineno,
                function=_function_name(call, parents),
                destination=destination,
                operation=operation,
            )

    return sorted(sites.values(), key=lambda item: (item.module, item.line, item.destination))


def _repository_snapshot_writes(root: Path = ROOT) -> list[SnapshotWriteSite]:
    sites = []
    for relative in SOURCE_FILES:
        path = root / relative
        sites.extend(_find_snapshot_writes(path.read_text(encoding="utf-8"), relative))
    return sites


def _is_direct_truncating_write(site: SnapshotWriteSite) -> bool:
    return (site.operation in {"write_text", "write_bytes", "os.open(O_TRUNC)"}
            or site.operation.startswith("open('w"))


def test_run_state_write_inventory_covers_authoritative_destinations():
    sites = _repository_snapshot_writes()
    destinations = {site.destination for site in sites}
    missing = EXPECTED_DESTINATIONS - destinations
    assert not missing, f"run-state snapshot inventory missed: {sorted(missing)}"


def test_authoritative_run_snapshots_are_not_written_by_truncating_apis():
    sites = _repository_snapshot_writes()
    direct = [site for site in sites if _is_direct_truncating_write(site)]
    assert not direct, "authoritative run snapshots require atomic publication:\n" + "\n".join(
        site.render() for site in direct)


def test_child_wire_publisher_fsyncs_file_and_parent_directory():
    source = (ROOT / "src/stateframe/_child.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    atomic_bytes = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "atomic_bytes")
    calls = [node for node in ast.walk(atomic_bytes) if isinstance(node, ast.Call)]
    attributes = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
    source_text = ast.unparse(atomic_bytes)

    assert "replace" in attributes, "wire records must publish by rename"
    assert "fsync" in attributes, "wire temp file must be fsynced before rename"
    assert "O_DIRECTORY" in source_text or "_fsync_directory" in source_text, (
        "wire record directory must be fsynced after rename")


@pytest.mark.parametrize(
    ("destination", "fault"),
    [
        (destination, fault)
        for destination in (
            "spec.json",
            "engine-result.json",
            "summary.json",
            "apply-receipt.json",
            "tool-settings.json",
            "skill-settings.json",
            "resume-pending-verified.json",
            "resume-divergence.json",
            "checkpoint-refused.json",
            "env-manifest.json",
            "env-seed.json",
        )
        for fault in ("partial_write", "fsync")
    ],
)
def test_atomic_snapshot_fault_keeps_predecessor_parseable(tmp_path, monkeypatch, destination, fault):
    """The shared JSON publisher must leave the previous authoritative record intact on ENOSPC."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / destination
    predecessor = {"generation": "before", "destination": destination}
    path.write_text(json.dumps(predecessor), encoding="utf-8")

    if fault == "partial_write":
        def fail_after_partial_json(value, stream, **kwargs):
            stream.write('{"generation":"partial')
            raise OSError(errno.ENOSPC, "injected disk full during snapshot write")

        monkeypatch.setattr(storage_gc.json, "dump", fail_after_partial_json)
    else:
        def fail_fsync(_descriptor):
            raise OSError(errno.ENOSPC, "injected disk full at snapshot fsync")

        monkeypatch.setattr(storage_gc.os, "fsync", fail_fsync)

    with pytest.raises(OSError) as raised:
        storage_gc._write_json_atomic(path, {"generation": "after", "destination": destination})

    assert raised.value.errno == errno.ENOSPC
    assert json.loads(path.read_text(encoding="utf-8")) == predecessor


@pytest.mark.parametrize("fault", ("partial_write", "fsync"))
def test_request_pack_fault_keeps_previous_pack_and_records_readable(tmp_path, monkeypatch, fault):
    run_dir = tmp_path / "run"
    request_dir = run_dir / "requests"
    request_dir.mkdir(parents=True)
    first_record = b'{"request":1}'
    (request_dir / "0001.json").write_bytes(first_record)
    request_records.seal_request_records(run_dir)
    pack = run_dir / request_records.PACK_NAME
    previous_pack = pack.read_bytes()

    second_record = b'{"request":2}'
    (request_dir / "0002.json").write_bytes(second_record)
    if fault == "partial_write":
        def fail_after_partial_archive(stream, records):
            stream.write(b"partial archive")
            raise OSError(errno.ENOSPC, "injected disk full during request pack write")

        monkeypatch.setattr(request_records, "_write_archive", fail_after_partial_archive)
    else:
        def fail_fsync(_descriptor):
            raise OSError(errno.ENOSPC, "injected disk full at request pack fsync")

        monkeypatch.setattr(request_records.os, "fsync", fail_fsync)

    with pytest.raises(OSError) as raised:
        request_records.seal_request_records(run_dir)

    assert raised.value.errno == errno.ENOSPC
    assert pack.read_bytes() == previous_pack
    assert request_records.read_request_record(run_dir, "0001.json") == first_record
    assert request_records.read_request_record(run_dir, "0002.json") == second_record


def test_guard_detects_an_injected_direct_snapshot_writer():
    injected = '''\
def injected(run_dir, payload):
    (run_dir / "summary.json").write_text(json.dumps(payload))
'''
    sites = _find_snapshot_writes(injected, "injected.py")
    direct = [site for site in sites if _is_direct_truncating_write(site)]
    assert [site.destination for site in direct] == ["summary.json"]
    assert direct[0].line == 2


def test_in_place_apply_receipt_is_durable_before_summary_publication(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary_path = run_dir / "summary.json"
    receipt_path = run_dir / "apply-receipt.json"
    original_replace = os.replace
    original_write_text = Path.write_text
    summary_publications = []

    def assert_receipt_precedes_summary():
        assert receipt_path.is_file(), "summary published before durable apply receipt"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["applied_files"] == ["src.py"]

    def watch_write_text(path, data, *args, **kwargs):
        if path == summary_path:
            assert_receipt_precedes_summary()
            summary_publications.append("write_text")
        return original_write_text(path, data, *args, **kwargs)

    def watch_replace(source, destination, *args, **kwargs):
        if Path(destination) == summary_path:
            assert_receipt_precedes_summary()
            summary_publications.append("os.replace")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", watch_write_text)
    monkeypatch.setattr(os, "replace", watch_replace)

    options = runner.RunOptions(
        repo=tmp_path,
        task="exercise in-place receipt ordering",
        endpoint=SimpleNamespace(public=lambda: {"model": "test"}),
        runner_kind="generic",
        in_place=True,
        profile=runner.profiles.DEFAULT,
    )
    work = workcopy.WorkCopy(original=tmp_path / "repo", copy=tmp_path / "copy",
                             baseline={"src.py": "before"})
    spec = {
        "public_row": {"entry": {"path": "src.py"}, "files": ["src.py"]},
        "engine_source": {},
        "runner_kind": "generic",
        "protect_tests": "all",
    }

    monkeypatch.setattr(runner, "_remaining", lambda _options: None)
    monkeypatch.setattr(runner, "drive", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(runner.outcomes, "terminal_receipt", lambda _run: {"result": "SUBMITTED"})
    monkeypatch.setattr(runner, "final_candidate", lambda _run: {"src.py": b"after"})
    monkeypatch.setattr(runner, "_artifact_digest", lambda _candidate: "sha256:artifact")
    monkeypatch.setattr(runner, "edit_scope_widened", lambda _spec: False)
    monkeypatch.setattr(runner, "_dropped_existing_candidate_paths", lambda *_args: [])
    monkeypatch.setattr(runner.workcopy, "apply_files", lambda *_args, **_kwargs: ["src.py"])
    monkeypatch.setattr(runner, "candidate_digest", lambda _root: "sha256:candidate")
    monkeypatch.setattr(runner.workcopy, "unified_diff", lambda _work: "diff --git a/src.py b/src.py\n")
    monkeypatch.setattr(runner, "summarize", lambda *_args: {
        "cached_share_display": "cache summary",
        "engine_terminal": {"result": "SUBMITTED"},
    })
    monkeypatch.setattr(runner, "_queue_wait_warning", lambda _run: None)
    monkeypatch.setattr(runner.workcopy, "changed_files", lambda _work: {"src.py": "modified"})
    monkeypatch.setattr(runner, "_provider_preflight_records", lambda _options: {})
    monkeypatch.setattr(runner.outcomes, "project", lambda *_args, **_kwargs: {
        "runtime_outcome": "completed", "check_outcome": "pass", "artifact": "saved",
    })
    monkeypatch.setattr(runner, "succeeded", lambda _summary: True)
    monkeypatch.setattr(runner.workcopy, "apply_in_place", lambda _work: ["src.py"])
    monkeypatch.setattr(runner, "finish_run_storage", lambda *_args, **_kwargs: {
        "checkpoints": None, "env": None,
    })

    runner._execute(options, 0.0, lambda _message: None, run_dir, work, spec)

    assert summary_publications, "summary was not durably published"
    assert_receipt_precedes_summary()

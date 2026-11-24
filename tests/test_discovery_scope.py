"""Discovery scope stays read-only until an existing path is admitted."""
from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from _journal import measurement_events
from _timeouts import scaled
from acceptance._support import Endpoint, PYTHON, environment, read_requests, run_case
from engine_run import start_monitored
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified


DISCOVERY_TASK = (
    "Inspect the repository and ask me which existing source file to change. "
    "Do not edit until I select a path."
)
DISCOVERY_REASON = "discovery scope is empty"


def _repo(root: Path, product_files: int) -> Path:
    repo = root / "repo"
    package = repo / "pkg"
    package.mkdir(parents=True)
    for index in range(product_files):
        (package / f"source_{index:05d}.py").write_text(
            f"VALUE = {index}\n", encoding="utf-8")
    checks = repo / "tests"
    checks.mkdir()
    (checks / "test_contract.py").write_text(
        "def test_repository_contract():\n    assert True\n", encoding="utf-8")
    return repo


def _script(path: Path, steps: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"},
                                "steps": steps}), encoding="utf-8")
    return path


def _repository_snapshot(repo: Path) -> dict[str, tuple[str, bytes | str | None]]:
    result: dict[str, tuple[str, bytes | str | None]] = {}
    for path in sorted(repo.rglob("*")):
        relative = path.relative_to(repo).as_posix()
        if path.is_symlink():
            result[relative] = ("symlink", os.readlink(path))
        elif path.is_dir():
            result[relative] = ("directory", None)
        elif path.is_file():
            result[relative] = ("file", path.read_bytes())
    return result


def _tool_results(case: object) -> list[str]:
    requests = getattr(case, "requests", ())
    return [
        str(message.get("content", ""))
        for request in requests if isinstance(request, dict)
        for message in request.get("messages", ())
        if isinstance(message, dict) and message.get("role") == "tool"
    ]


def _unaffected_case(root: Path, product_files: int, task: str,
                     flags: tuple[str, ...] = ()):
    root.mkdir(parents=True, exist_ok=True)
    repo = _repo(root, product_files)
    script = _script(root / "parity.script.json", [
        {"tool": "read", "arguments": {"path": "pkg/source_00000.py"}},
        {"tool": "done", "arguments": {"summary": "I inspected the selected path."}},
    ])
    return run_case(root, str(script), repo=repo, task=task, flags=flags,
                    default_test_cmd=False, timeout=300)


@pytest.mark.parametrize(
    "product_files", [201, 4999], ids=["201-product-files", "5000-total-files"])
def test_real_cli_large_repo_reads_candidates_and_finishes_without_writes(
    tmp_path: Path, product_files: int,
) -> None:
    repo = _repo(tmp_path, product_files)
    before = _repository_snapshot(repo)
    first = "pkg/source_00000.py"
    last = f"pkg/source_{product_files - 1:05d}.py"
    script = _script(tmp_path / "discovery.script.json", [
        {"tool": "read", "arguments": {"path": first}},
        {"tool": "read", "arguments": {"path": last}},
        {"tool": "done", "arguments": {
            "summary": f"I read {first} and {last}. Which one should I edit?"}},
    ])

    case = run_case(tmp_path, str(script), repo=repo, task=DISCOVERY_TASK,
                    default_test_cmd=False, timeout=360)

    assert case.process.returncode != 2, case.process.stdout + case.process.stderr
    assert case.requests, case.process.stdout + case.process.stderr
    spec = json.loads((case.run_dir / "spec.json").read_text(encoding="utf-8"))
    row = spec["public_row"]
    assert row["files"] == {}
    assert row["scoped_edits"] is True
    assert row["scope_source"] == "discovery"
    assert row["entry"]["path"] == ""
    assert row["path"] == ""
    assert row["new_file_roots"] == []
    transcript = case.transcript
    assert first in transcript and last in transcript
    assert f"VALUE = {product_files - 1}" in transcript
    assert _repository_snapshot(repo) == before
    assert not (repo / "pkg" / "discovery.json").exists()


def test_real_cli_refuses_every_legacy_writer_while_scope_is_empty(tmp_path: Path) -> None:
    repo = _repo(tmp_path, 201)
    before = _repository_snapshot(repo)
    target = "pkg/source_00000.py"
    script = _script(tmp_path / "writers.script.json", [
        {"tool": "read", "arguments": {"path": target}},
        {"tool": "edit", "arguments": {"path": target,
                                            "function_source": "VALUE = -1\n"}},
        {"tool": "str_replace", "arguments": {"path": target,
                                                   "old_str": "VALUE = 0",
                                                   "new_str": "VALUE = -1"}},
        {"tool": "insert_before", "arguments": {"path": target, "target_ref": {},
                                                    "new_str": "VALUE = -1\n"}},
        {"tool": "file_create", "arguments": {"path": "pkg/unauthorized.py",
                                                   "content": "VALUE = -1\n"}},
        {"tool": "file_move", "arguments": {"source": target,
                                                  "destination": "pkg/moved.py"}},
        {"tool": "file_delete", "arguments": {"path": target}},
        {"tool": "str_replace", "arguments": {
            "path": target, "old_str": "VALUE = 0", "new_str": "VALUE = -1",
            "edits": [
                {"path": target, "old": "VALUE = 0", "new": "VALUE = -1"},
                {"path": "pkg/source_00001.py", "old": "VALUE = 1", "new": "VALUE = -1"},
            ],
        }},
        {"tool": "shell", "arguments": {"command": "printf 'VALUE = -1\\n' > pkg/shell_output.py"}},
        {"tool": "done", "arguments": {"summary": "No source path was admitted."}},
    ])

    case = run_case(
        tmp_path, str(script), repo=repo, task=DISCOVERY_TASK,
        default_test_cmd=False, timeout=300,
    )

    assert case.process.returncode != 2, case.process.stdout + case.process.stderr
    assert _repository_snapshot(repo) == before
    assert "pkg/unauthorized.py" not in case.diff
    assert sum(DISCOVERY_REASON in result for result in _tool_results(case)) >= 8
    assert not (repo / "pkg" / "shell_output.py").exists()


def test_discovery_refuses_outside_symlink_and_protected_test_paths(tmp_path: Path) -> None:
    repo = _repo(tmp_path, 201)
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 7\n", encoding="utf-8")
    link = repo / "pkg" / "source_link.py"
    link.symlink_to(repo / "pkg" / "source_00000.py")
    before = _repository_snapshot(repo)
    outside_before = outside.read_bytes()
    script = _script(tmp_path / "hostile-paths.script.json", [
        {"tool": "file_create", "arguments": {"path": "../outside.py",
                                                     "content": "VALUE = -1\\n"}},
        {"tool": "file_delete", "arguments": {"path": "pkg/source_link.py"}},
        {"tool": "file_delete", "arguments": {"path": "tests/test_contract.py"}},
        {"tool": "done", "arguments": {"summary": "No path was admitted."}},
    ])

    case = run_case(tmp_path, str(script), repo=repo, task=DISCOVERY_TASK,
                    default_test_cmd=False, timeout=300)

    assert case.process.returncode != 2, case.process.stdout + case.process.stderr
    assert _repository_snapshot(repo) == before
    assert outside.read_bytes() == outside_before
    assert not (repo / "outside.py").exists()
    results = _tool_results(case)
    assert any("outside the allowed workspace" in result.lower() for result in results), results
    assert any("pkg/source_link.py" in result and "REFUSED" in result for result in results), results
    assert any("tests/test_contract.py" in result and DISCOVERY_REASON in result
               for result in results), results


def test_discovery_shell_and_hook_effects_cannot_create_product_files(tmp_path: Path) -> None:
    from kernel_effects import DISCOVERY_SCOPE_REFUSAL, diff, effect_policy_for_run, scan

    repo = tmp_path / "repo"
    package = repo / "pkg"
    package.mkdir(parents=True)
    (package / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "hook.py").write_text("# trusted hook\n", encoding="utf-8")
    row = {
        "tier": "USER-REPO", "runner_kind": "generic", "files": {},
        "scoped_edits": True, "scope_source": "discovery",
        "allowed_product_paths": ["pkg"], "new_file_roots": ["pkg"],
    }
    policy = effect_policy_for_run(repo, row, set(), hook_paths=frozenset({"pkg/hook.py"}))
    before = scan(repo, exclude=lambda _: False)
    (package / "shell_output.py").write_text("VALUE = 2\n", encoding="utf-8")
    shell_effect = next(item for item in diff(before, scan(repo, exclude=lambda _: False))
                        if item.path == "pkg/shell_output.py")
    assert policy.authorize(shell_effect) == (False, DISCOVERY_SCOPE_REFUSAL)

    hook_before = scan(repo, exclude=lambda _: False)
    (package / "hook_output.py").write_text("VALUE = 3\n", encoding="utf-8")
    hook_effect = next(item for item in diff(hook_before, scan(repo, exclude=lambda _: False))
                       if item.path == "pkg/hook_output.py")
    assert policy.authorize(hook_effect) == (False, DISCOVERY_SCOPE_REFUSAL)

    directory_before = scan(repo, exclude=lambda _: False)
    (package / "generated").mkdir()
    directory_effect = next(item for item in diff(
        directory_before, scan(repo, exclude=lambda _: False)
    ) if item.path == "pkg/generated")
    assert policy.authorize(directory_effect) == (False, DISCOVERY_SCOPE_REFUSAL)


def test_discovery_widening_receipt_records_current_existing_product_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from probes.react_repair import FileSet, _widen_edit_scope
    from kernel_scope_widen_guard import forget, read as guarded_read

    root = tmp_path / "repo"
    package = root / "pkg"
    package.mkdir(parents=True)
    path = package / "source.py"
    current = b"VALUE = 4\n"
    path.write_bytes(current)
    row = {
        "tier": "USER-REPO", "runner_kind": "generic", "files": {},
        "scoped_edits": True, "scope_source": "discovery",
        "allowed_product_paths": ["pkg"], "new_file_roots": ["pkg"],
        "entry": {"fn": "", "path": "pkg/discovery.json"},
    }
    fileset = FileSet.from_row(row)
    events: list[tuple[str, dict[str, object]]] = []

    class Journal:
        def emit(self, kind: str, **fields: object) -> None:
            events.append((kind, fields))

    monkeypatch.setenv("STATEFRAME_EDIT_SCOPE_WIDEN", "on")
    try:
        note = _widen_edit_scope(root, "pkg/source.py", row, fileset, set(), journal=Journal())
        assert note and "pkg/source.py" in note
        assert events == [("scope_widen", {
            "producer": "react_repair", "dispatch": False, "path": "pkg/source.py",
            "pre_edit_sha256": hashlib.sha256(current).hexdigest(),
            "reason": "edit outside the editable set", "widened_count": 1, "cap": 24,
        })]
        assert guarded_read(path) == current
        assert row["files"]["pkg/source.py"]["parent_src"] == current.decode()
    finally:
        forget(path)


def test_discovery_opt_in_widening_admits_only_the_read_existing_file(tmp_path: Path) -> None:
    repo = _repo(tmp_path, 201)
    before = _repository_snapshot(repo)
    target = "pkg/source_00000.py"
    script = _script(tmp_path / "widen.script.json", [
        {"tool": "read", "arguments": {"path": target}},
        {"tool": "str_replace", "arguments": {
            "path": target, "old_str": "VALUE = 0", "new_str": "VALUE = 88"}},
        {"tool": "done", "arguments": {"summary": "Edited only the admitted source."}},
    ])

    case = run_case(
        tmp_path, str(script), repo=repo, task=DISCOVERY_TASK,
        default_test_cmd=False, timeout=300,
        env_overrides={"STATEFRAME_EDIT_SCOPE_WIDEN": "on"},
    )

    assert case.process.returncode != 2, case.process.stdout + case.process.stderr
    assert "+VALUE = 88" in case.diff
    changed = {
        line[6:] for line in case.diff.splitlines()
        if line.startswith(("--- a/", "+++ b/"))
    }
    assert changed == {target}
    assert _repository_snapshot(repo) == before
    admissions = [event for event in measurement_events(case.run_dir)
                  if event.get("kind") == "scope_widen"]
    assert len(admissions) == 1
    assert admissions[0]["path"] == target
    assert admissions[0]["pre_edit_sha256"] == hashlib.sha256(b"VALUE = 0\n").hexdigest()


def test_discovery_widening_rejects_paths_without_current_product_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from probes.react_repair import FileSet, _edit_scope_widen_refusal

    root = tmp_path / "repo"
    package = root / "pkg"
    package.mkdir(parents=True)
    source = package / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    link = package / "alias.py"
    link.symlink_to(source)
    test_path = root / "tests" / "test_contract.py"
    test_path.parent.mkdir()
    test_path.write_text("def test_contract(): pass\n", encoding="utf-8")
    row = {
        "tier": "USER-REPO", "files": {}, "scoped_edits": True,
        "scope_source": "discovery", "allowed_product_paths": ["pkg"],
        "new_file_roots": ["pkg"], "public_check_paths": ["tests/test_contract.py"],
        "public_tests": {"tests/test_contract.py": "def test_contract(): pass\n"},
    }
    fileset = FileSet.from_row(row)
    monkeypatch.setenv("STATEFRAME_EDIT_SCOPE_WIDEN", "on")

    assert "existing file" in (_edit_scope_widen_refusal(
        root, "pkg/not_created.py", row, fileset, set()) or "")
    assert "symlink" in (_edit_scope_widen_refusal(
        root, "pkg/alias.py", row, fileset, set()) or "").lower()
    test_path_refusal = _edit_scope_widen_refusal(
        root, "tests/test_contract.py", row, fileset, set())
    assert test_path_refusal and "product source" in test_path_refusal.lower()


def test_discovery_retained_workspace_can_resume_without_an_edit_target(tmp_path: Path) -> None:
    from kernel_continuation_restart import execution_paths

    session = tmp_path / "session"
    session.mkdir()
    root = session / "repo"
    root.mkdir()

    class Workspace:
        directory = session

        def read(self, _name: str) -> dict[str, object]:
            return {"root": "repo", "target": None, "scope_source": "discovery"}

    assert execution_paths(Workspace()) == (root, None)


def test_discovery_request_gets_empty_source_authority_without_an_edit_target(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from kernel_request_source_window import bound_runtime_sources

    root = tmp_path / "repo"
    root.mkdir()
    runtime = SimpleNamespace(
        target=None,
        owner=SimpleNamespace(root=root, task_row={
            "msg": DISCOVERY_TASK,
            "files": {},
            "scoped_edits": True,
            "scope_source": "discovery",
        }),
    )
    result = bound_runtime_sources([], runtime, max_request_bytes=8192)

    assert result["paths"] == ()
    assert result["closure"]["paths"] == []
    assert result["closure"]["source_contents"] == {}
    from kernel_request_source_window import verify_final_source_delivery

    receipt = verify_final_source_delivery(
        result["closure"], [], payload={"messages": []}, sources={})
    assert receipt["status"] == "COMPLETE"
    assert receipt["proof_status"] == "VERIFIED"


@pytest.mark.parametrize("receipt_state", ["current", "stale", "forged"])
def test_discovery_carries_only_current_full_read_receipts(
    tmp_path: Path, receipt_state: str,
) -> None:
    from types import SimpleNamespace

    from kernel_request_source_window import bound_runtime_sources

    root = tmp_path / "repo"
    package = root / "pkg"
    package.mkdir(parents=True)
    path = package / "source.py"
    observed = "VALUE = 1\n"
    path.write_text(observed if receipt_state != "stale" else "VALUE = 2\n", encoding="utf-8")
    record = {
        "full": True,
        "content": observed,
        "revision": "sha256:" + hashlib.sha256(observed.encode()).hexdigest(),
        "reference": "read@0",
    }
    if receipt_state == "forged":
        record["reference"] = ""
    runtime = SimpleNamespace(
        target=None,
        _empirical_frame_sources={"pkg/source.py": record},
        owner=SimpleNamespace(root=root, task_row={
            "msg": DISCOVERY_TASK, "files": {}, "scoped_edits": True,
            "scope_source": "discovery",
        }),
    )

    result = bound_runtime_sources([], runtime, max_request_bytes=8192)
    receipts = result["closure"]["read_source_receipts"]
    from kernel_frame_source_snapshot import frame_source_snapshot

    if receipt_state == "current":
        assert receipts == {"pkg/source.py": {
            "revision": record["revision"], "reference": "read@0",
        }}
        assert frame_source_snapshot(runtime, result["closure"]) == {"pkg/source.py": observed}
    else:
        assert receipts == {}
        assert frame_source_snapshot(runtime, result["closure"]) == {}


def test_discovery_workspace_binding_requires_empty_scoped_authority(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from kernel_continuation_workspace import ContinuationWorkspace, WorkspaceError

    workspace = ContinuationWorkspace.create(tmp_path / "sessions", task_digest="sha256:" + "a" * 64)
    root = workspace.directory / "repo"
    root.mkdir()
    owner = SimpleNamespace(
        root=root,
        task_row={"scope_source": "discovery", "scoped_edits": True, "files": {}},
    )

    class Journal:
        run_id = "run"
        task_id = "task"
        config_hash = "config"
        path = workspace.directory / "events.jsonl"

        def emit(self, *_args: object, **_kwargs: object) -> None:
            pass

    runtime = SimpleNamespace(target=None, owner=owner, journal=Journal(),
                              protected_tests={}, binding=lambda: {})
    try:
        workspace.bind(runtime)
        assert workspace.read("initialized.json")["target"] is None
        assert workspace.read("initialized.json")["scope_source"] == "discovery"
        owner.task_row = {"scope_source": "inferred", "scoped_edits": True, "files": {}}
        with pytest.raises(WorkspaceError, match="resolved target"):
            workspace.bind(runtime)
    finally:
        workspace.close()


def test_discovery_mutation_proposal_cannot_create_a_path(tmp_path: Path) -> None:
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    repo = tmp_path / "repo"
    repo.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    row = {"files": {}, "scoped_edits": True, "scope_source": "discovery"}
    runtime = MutationRuntime(
        level="authored", journal=Journal(run), repo=repo,
        policy=MutationPolicy(level="authored", write_globs=("pkg/*.py",)),
        edit_scope=set(), effect_row=row,
    )

    with pytest.raises(ValueError, match=DISCOVERY_REASON):
        runtime.apply_edit_proposals("empty-discovery", [{
            "op": "create", "path": "pkg/new.py", "base_sha256": None,
            "text": "VALUE = 1\n",
        }], cleanup_ok=True)
    assert not (repo / "pkg" / "new.py").exists()


@pytest.mark.parametrize("tool", ["edit_source", "rewrite_source"])
def test_discovery_protocol_writers_refuse_before_read_or_write(tmp_path: Path, tool: str) -> None:
    from kernel_edit_handles import DeliveredSpan, SpanKind, mint_handles
    from kernel_edit_source_tool import (
        EditProtocolRunState, execute_edit_source, execute_rewrite_source,
    )

    repo = tmp_path / "repo"
    package = repo / "pkg"
    package.mkdir(parents=True)
    source = package / "source.py"
    current = "def answer():\n    return 1\n"
    source.write_text(current, encoding="utf-8")
    span_kind = SpanKind.REWRITE if tool == "rewrite_source" else SpanKind.SPAN
    span = DeliveredSpan.from_source("pkg/source.py", current, kind=span_kind)
    state = EditProtocolRunState()
    state.set_request(mint_handles(3, [span]))
    handle = state.handle_map.records[0].handle
    reads: list[Path] = []
    writes: list[Path] = []

    def read_bytes(path: Path) -> bytes:
        reads.append(path)
        return path.read_bytes()

    def write_bytes(path: Path, data: bytes) -> tuple[bool, str | None]:
        writes.append(path)
        path.write_bytes(data)
        return True, None

    deny = lambda _path: DISCOVERY_REASON
    if tool == "edit_source":
        outcome = execute_edit_source(
            repo, {"edits": [{"handle": handle, "old_text": "return 1",
                               "new_text": "return 2"}]},
            state=state, read_bytes=read_bytes, write_bytes=write_bytes, not_editable=deny)
    else:
        outcome = execute_rewrite_source(
            repo, {"handle": handle, "new_text": "def answer():\n    return 2\n"},
            state=state, read_bytes=read_bytes, write_bytes=write_bytes, not_editable=deny)

    assert outcome.status == "REJECTED"
    assert DISCOVERY_REASON in outcome.detail
    assert not reads
    assert not writes
    assert source.read_text(encoding="utf-8") == current


def test_discovery_edit_contract_does_not_admit_the_entry_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kernel_edit_affordance import contract, path_editable, resolve_editable_paths

    monkeypatch.setenv("KERNEL_EDIT_AFFORDANCE", "1")
    monkeypatch.setenv("STATEFRAME_VIEW", "operational")
    scope = contract(
        multi_file=True, entry_path="pkg/discovery.json", entry_fn=None,
        product_paths=("pkg/discovery.json",), scope_source="discovery",
    )
    assert scope is not None
    row = {"files": {}, "scope_source": "discovery"}
    authority = scope["edit_authority"]
    assert scope["editable_paths"] == []
    assert resolve_editable_paths(authority, row) == []
    assert not path_editable(authority, "pkg/discovery.json", row=row)
    assert "discovery scope is empty" in scope["instruction"]


def test_discovery_file_set_accepts_an_empty_entry_hint() -> None:
    from probes.react_repair import FileSet

    fileset = FileSet.from_row({
        "tier": "USER-REPO", "runner_kind": "generic", "files": {
            "pkg/source.py": {"parent_src": "VALUE = 1\n"},
        },
        "entry": {"path": "", "fn": ""},
        "path": "", "rel": "", "scope_source": "discovery",
    })

    assert fileset.present
    assert fileset.entry_path is None
    assert fileset.entry_fn is None
    assert fileset.primary_path is None
    assert fileset.primary_fn == ""


def test_discovery_compacts_large_workspace_revision_without_changing_other_frames() -> None:
    from kernel_stateframe_f4 import _revision, project_f4_state

    revisions = {
        f"pkg/source_{index:05d}.py": f"sha256:{index:064x}"
        for index in range(5000)
    }
    frame = {"artifact_revisions": revisions}
    legacy = "|".join(sorted(set(revisions.values())))

    assert _revision(frame) == legacy
    compact = _revision(frame, compact_many=True)
    assert compact == "sha256:" + hashlib.sha256(legacy.encode("utf-8")).hexdigest()
    assert len(compact) == len("sha256:") + 64
    assert project_f4_state(frame)["revision"] == legacy
    assert project_f4_state(frame, compact_many_revisions=True)["revision"] == compact


def test_discovery_scope_survives_information_policy_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    from kernel_information_policy import InformationPolicy

    monkeypatch.setenv("INITIAL_ORACLE_SYMPTOM", "OFF")
    row = {
        "scope_source": "discovery", "scoped_edits": True, "files": {},
        "entry": {"path": "pkg/discovery.json", "fn": ""},
    }

    worker_row = InformationPolicy(row).worker_row(row)

    assert worker_row.get("scope_source") == "discovery"
    assert worker_row["files"] == {}


def test_discovery_readers_do_not_re_admit_the_entry_hint(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from kernel_edit_affordance import (
        WIDENED_PATHS_KEY, bundle_editable_targets, path_editable, whole_set_editable,
        resolve_editable_paths,
    )
    from kernel_editready_fact import _editable_paths
    from kernel_green_lock import _editable_product_paths
    from kernel_green_lock_repair import RepairController
    from kernel_request_source_window import editable_product_paths
    from kernel_frame_source_snapshot import recent_editable_read_windows

    package = tmp_path / "pkg"
    package.mkdir()
    for name in ("hint.py", "listed.py", "admitted.py"):
        (package / name).write_text("VALUE = 1\n", encoding="utf-8")
    row = {
        "scope_source": "discovery", "files": {"pkg/listed.py": "sha256:source"},
        "editable_paths": ["pkg/listed.py"], "editable_all": True,
    }
    owner = SimpleNamespace(root=tmp_path, task_row=row)
    runtime = SimpleNamespace(owner=owner, target=package / "hint.py",
                              product_paths=("pkg/listed.py",))
    holder = {"scope_source": "discovery", "editable_all": True,
              "editable_paths": ["pkg/listed.py"]}

    assert _editable_product_paths(runtime) == ()
    assert _editable_paths(runtime) == frozenset()
    assert editable_product_paths(runtime) == ()
    assert resolve_editable_paths(holder, row) == []
    assert whole_set_editable(holder) is False
    assert bundle_editable_targets(row, holder) == []
    assert not path_editable(holder, "pkg/hint.py", product_paths=("pkg/hint.py",), row=row)
    assert not path_editable(holder, "pkg/listed.py", product_paths=("pkg/listed.py",), row=row)
    assert RepairController(runtime)._product_paths(package / "hint.py") == ()
    source = "VALUE = 1\n"
    revision = "sha256:" + hashlib.sha256(source.encode()).hexdigest()
    runtime._read_reuse_visibility_candidates = [{
        "path": "pkg/listed.py", "revision": revision,
        "start_line": 1, "end_line": 1, "content": source,
        "content_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }]
    assert recent_editable_read_windows(runtime, {}) == []

    row[WIDENED_PATHS_KEY] = ["pkg/admitted.py"]
    holder[WIDENED_PATHS_KEY] = ["pkg/admitted.py"]
    assert _editable_product_paths(runtime) == ("pkg/admitted.py",)
    assert _editable_paths(runtime) == frozenset({"pkg/admitted.py"})
    assert editable_product_paths(runtime) == ("pkg/admitted.py",)
    assert resolve_editable_paths(holder, row) == ["pkg/admitted.py"]
    assert bundle_editable_targets(row, holder) == ["pkg/admitted.py"]
    assert path_editable(holder, "pkg/admitted.py", product_paths=(), row=row)
    assert RepairController(runtime)._product_paths(package / "admitted.py") == (
        "pkg/admitted.py",
    )


def test_discovery_edit_contract_adds_no_writer_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    from kernel_edit_affordance import DISCOVERY_SCOPE_INSTRUCTION, contract, tool_surface

    monkeypatch.setenv("KERNEL_EDIT_AFFORDANCE", "1")
    monkeypatch.setenv("STATEFRAME_VIEW", "operational")
    scope = contract(multi_file=True, entry_path=None, entry_fn=None,
                     product_paths=(), scope_source="discovery")
    writers = ["edit", "str_replace", "insert_before", "file_create", "file_move",
               "file_delete", "replace_module", "edit_source", "rewrite_source", "shell"]
    tools = []
    for name in writers:
        function = {"name": name, "description": name}
        if name == "edit":
            function["parameters"] = {"properties": {"function_source": {"description": ""}}}
        tools.append({"type": "function", "function": function})
    surfaced = tool_surface(tools, scope)

    assert [tool["function"]["name"] for tool in surfaced] == writers
    assert all("discovery scope is empty" in tool["function"]["description"]
               for tool in surfaced)
    assert "headless run" in DISCOVERY_SCOPE_INSTRUCTION
    assert "list every candidate path read" in DISCOVERY_SCOPE_INSTRUCTION
    assert "resumable needs-input outcome" not in DISCOVERY_SCOPE_INSTRUCTION


def test_discovery_final_candidate_drops_unadmitted_new_paths(tmp_path: Path) -> None:
    from stateframe import runner, workcopy

    repo = tmp_path / "repo"
    package = repo / "pkg"
    package.mkdir(parents=True)
    (package / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    work = workcopy.create(repo, tmp_path / "work")
    row = {"files": {}, "scoped_edits": True, "scope_source": "discovery"}
    candidate = {"pkg/unadmitted.py": b"VALUE = 2\n"}

    dropped = runner._dropped_discovery_candidate_paths(work, row, candidate, set())
    written, dropped_paths, error = runner._apply_sealed_candidate(
        work, candidate, set(), lambda _message: None, dropped_paths=dropped,
        discovery_scope=True)

    assert dropped == ["pkg/unadmitted.py"]
    assert written == []
    assert dropped_paths == dropped
    assert error
    assert not (work.copy / "pkg" / "unadmitted.py").exists()


def test_non_discovery_sealed_candidate_error_stays_unchanged(tmp_path: Path) -> None:
    from stateframe import runner, workcopy

    repo = tmp_path / "repo"
    package = repo / "pkg"
    package.mkdir(parents=True)
    (package / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    work = workcopy.create(repo, tmp_path / "work")
    candidate = {"pkg/unadmitted.py": b"VALUE = 2\n"}
    written, dropped, error = runner._apply_sealed_candidate(
        work, candidate, set(), lambda _message: None,
        dropped_paths=["pkg/unadmitted.py"])

    assert written == []
    assert dropped == ["pkg/unadmitted.py"]
    assert error == "sealed candidate changed existing files outside edit scope"


@pytest.mark.parametrize("pause_at", [1, 2, 3])
def test_discovery_resume_replays_each_pending_request_without_writes(
    tmp_path: Path, pause_at: int,
) -> None:
    from acceptance.test_resume_completion import _engine_child

    repo = _repo(tmp_path, 201)
    before = _repository_snapshot(repo)
    first = "pkg/source_00000.py"
    last = "pkg/source_00200.py"
    steps = [
        {"tool": "read", "arguments": {"path": first}},
        {"tool": "read", "arguments": {"path": last}},
        *({"tool": "done", "arguments": {
            "summary": f"I read {first} and {last}. Which source should I edit?"}}
          for _ in range(8)),
    ]
    script = _script(tmp_path / "resume.script.json", steps)
    run_dir, marker = tmp_path / "run", tmp_path / "paused"
    endpoint = subprocess.Popen(
        [PYTHON, str(Path(__file__).parent / "acceptance" / "fixtures" / "pause_endpoint.py"),
         "--script", str(script), "--marker", str(marker), "--pause-at", str(pause_at),
         "--pause-seconds", "600"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment(),
    )
    runner = None
    try:
        assert endpoint.stdout is not None
        url = endpoint.stdout.readline().strip()
        assert url.startswith("http://127.0.0.1:")
        launched_after = time.time()
        runner = start_monitored(
            [PYTHON, "-m", "stateframe", "run", "--repo", str(repo), "--task", DISCOVERY_TASK,
             "--endpoint", url, "--model", "scripted", "--run-dir", str(run_dir),
             "--test-python", PYTHON, "--max-turns", "32", "--quiet", "--shell", "off"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment(),
            start_new_session=True, expected_seconds=scaled(600), step="discovery interruption",
        )
        deadline = time.monotonic() + scaled(600)
        while not marker.exists():
            assert runner.poll() is None, runner.communicate()
            remaining = deadline - time.monotonic()
            assert remaining > 0, "discovery request did not reach the loopback pause"
            time.sleep(min(0.1, remaining))

        spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
        assert spec["public_row"]["files"] == {}
        assert spec["public_row"]["scope_source"] == "discovery"
        expected = (run_dir / "wire" / f"{pause_at:04d}.request.json").read_bytes()
        child = _engine_child(runner.pid)
        assert child is not None
        identity = record_identity(child, launched_after=launched_after,
                                   command_fragment="_child.py")
        signal_run_verified(run_dir, child, identity, signal.SIGKILL, deadline=deadline,
                            what="discovery pending request",
                            require_checkpoint_saved=pause_at > 1)
        runner.communicate(timeout=scaled(30))
        runner = None

        resume_steps = steps[pause_at - 1:]
        resume_script = _script(tmp_path / "resume-remaining.script.json", resume_steps)
        with Endpoint(resume_script) as resumed_endpoint:
            resumed = subprocess.run(
                [PYTHON, "-m", "stateframe", "resume", str(run_dir),
                 "--endpoint", resumed_endpoint.url, "--model", "scripted", "--quiet"],
                capture_output=True, text=True, env=environment(), timeout=scaled(300), check=False,
            )
        output = resumed.stdout + resumed.stderr
        assert resumed.returncode != 2, output
        assert (run_dir / "wire" / f"{pause_at:04d}.request.json").read_bytes() == expected
        assert _repository_snapshot(repo) == before
        assert not (repo / "pkg" / "discovery.json").exists()
        assert len(read_requests(run_dir)) >= pause_at
    finally:
        if runner is not None and runner.poll() is None:
            signal_run_owned(run_dir, runner, signal.SIGKILL, deadline=time.monotonic(),
                             what="discovery resume test cleanup", require_checkpoint_saved=False)
            runner.wait(timeout=scaled(5))
        if endpoint.poll() is None:
            endpoint.terminate()
            try:
                endpoint.wait(timeout=scaled(5))
            except subprocess.TimeoutExpired:
                endpoint.kill()
                endpoint.wait(timeout=scaled(5))


@pytest.mark.parametrize("receipt_digest", ["0" * 64, hashlib.sha256(b"stale").hexdigest()],
                         ids=["forged", "stale"])
def test_host_rejects_forged_or_stale_discovery_widen_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, receipt_digest: str,
) -> None:
    from stateframe import runner
    from stateframe.engine import kernel_journal_store

    baseline = {"pkg/source.py": hashlib.sha256(b"current\n").hexdigest()}
    record = {"kind": "scope_widen", "path": "pkg/source.py",
              "pre_edit_sha256": receipt_digest}

    monkeypatch.setattr(kernel_journal_store, "journal_paths", lambda _root: [tmp_path / "engine.jsonl"])

    @contextmanager
    def open_records(_path):
        yield io.StringIO(json.dumps(record) + "\n")

    monkeypatch.setattr(kernel_journal_store, "open_journal", open_records)
    with pytest.raises(runner.RunError, match="invalid edit scope admission"):
        runner._widen_admissions(tmp_path, baseline)


@pytest.mark.parametrize("case_name", ["explicit", "inferred", "task_text", "small"])
def test_unaffected_scope_keeps_its_source_and_no_discovery_text(
    tmp_path: Path, case_name: str,
) -> None:
    product_count, task, flags, source = {
        "explicit": (201, "Inspect this repository.", ("--edit", "pkg/source_00000.py"), "explicit"),
        "inferred": (201, "Fix pkg/source_00000.py.", (), "inferred"),
        "task_text": (201, "Only edit pkg/source_00000.py. Fix its behavior.", (), "task_text"),
        "small": (5, "Inspect this repository.", (), "whole_repo"),
    }[case_name]
    case = _unaffected_case(tmp_path / case_name, product_count, task, flags)

    assert case.requests, case.process.stdout + case.process.stderr
    row = json.loads((case.run_dir / "spec.json").read_text(encoding="utf-8"))["public_row"]
    assert row["scope_source"] == source
    assert DISCOVERY_REASON not in (case.run_dir / "wire" / "0001.request.json").read_text(encoding="utf-8")

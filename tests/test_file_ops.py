import json
from types import SimpleNamespace

import pytest

from stateframe import runner, workcopy
from stateframe.chat import Session, TaskRecord
from stateframe.inspect_run import overview
from kernel_done_is_submit_authority import artifact_digest, store_candidate
from kernel_file_ops import FileOperationRefused, file_delete, file_move
from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
from kernel_frame_source_snapshot import frame_source_snapshot
from kernel_post_edit_refresh import rebind_after_file_operation, refresh_after_applied_edit
from kernel_request_evidence_closure import build_source_closure
from kernel_request_source_window import editable_product_paths, source_revision


def _allow(_path):
    return True


def _unprotected(_path):
    return False


@pytest.mark.parametrize("operation", ["move", "delete"])
def test_file_operation_rebinds_source_proof_for_next_request(tmp_path, operation):
    root = tmp_path / "repo"
    source = root / "pkg" / "old.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 42\n")
    owner = SimpleNamespace(root=root, task_row={"files": {"pkg/old.py": {}}})
    runtime = SimpleNamespace(owner=owner, target=source,
                              _request_source_closure_cache={})
    old_plan = build_source_closure(
        root, editable_product_paths(runtime), "Rename pkg/old.py",
        target_path="pkg/old.py").as_dict()
    runtime._request_source_closure = old_plan
    assert frame_source_snapshot(runtime, old_plan)["pkg/old.py"] == "VALUE = 42\n"

    if operation == "move":
        effect = file_move(root, "pkg/old.py", "pkg/new.py",
                           protected=_unprotected, editable=_allow,
                           new_file_allowed=_allow)
    else:
        effect = file_delete(root, "pkg/old.py", protected=_unprotected,
                             editable=_allow)
    with pytest.raises(FrameAllocationReplanNeeded,
                       match="source proof workspace revision unavailable"):
        frame_source_snapshot(runtime, old_plan)

    target = rebind_after_file_operation(runtime, effect.source, effect.destination)
    assert target == (root / "pkg/new.py" if operation == "move" else None)
    assert runtime._request_source_closure is None
    current_paths = editable_product_paths(runtime)
    assert current_paths == (("pkg/new.py",) if operation == "move" else ())
    fresh_plan = build_source_closure(
        root, current_paths, "Rename pkg/old.py",
        target_path=target.relative_to(root).as_posix() if target else None).as_dict()
    sources = frame_source_snapshot(runtime, fresh_plan)
    assert sources == ({"pkg/new.py": "VALUE = 42\n"} if target else {})

    revision = source_revision("VALUE = 42\n")
    binding = {"product_files": {"pkg/new.py": revision} if target else {},
               "product_revision": "current"}
    receipt = {"kind": "edit", "result_class": "APPLIED",
               "target_path": effect.destination or effect.source,
               "changed_files": [{"path": path} for path in effect.changed_paths]}
    refreshed = refresh_after_applied_edit(runtime, receipt, binding)
    assert refreshed == ({"pkg/new.py": revision} if target else {})
    assert runtime._post_edit_refresh["paths"]["pkg/old.py"] == "UNAVAILABLE"


def test_file_operation_retires_closure_when_source_is_not_target(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "active.py"
    target.write_text("ACTIVE = True\n")
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=root), target=target,
        _request_source_closure={"units": [{"path": "stale.py"}]},
        _request_source_closure_cache={"old": object()},
    )
    assert rebind_after_file_operation(runtime, "stale.py", None) == target
    assert runtime._request_source_closure is None
    assert runtime._request_source_closure_cache == {}


def test_delete_and_move_guard_paths_and_collisions(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "old.py").write_text("value = 1\n")
    (root / "other.py").write_text("value = 2\n")
    with pytest.raises(FileOperationRefused, match="OUTSIDE_WORKSPACE"):
        file_delete(root, "../old.py", protected=_unprotected, editable=_allow)
    with pytest.raises(FileOperationRefused, match="DESTINATION_EXISTS"):
        file_move(root, "old.py", "other.py", protected=_unprotected,
                  editable=_allow, new_file_allowed=_allow)
    with pytest.raises(FileOperationRefused, match="NOT_EDITABLE"):
        file_delete(root, "old.py", protected=_unprotected, editable=lambda _: False)
    with pytest.raises(FileOperationRefused, match="NEW_FILE_FORBIDDEN"):
        file_move(root, "old.py", "new.py", protected=_unprotected,
                  editable=_allow, new_file_allowed=lambda _: False)
    with pytest.raises(FileOperationRefused, match="PATH_NOT_FOUND"):
        file_move(root, "old.py", "other.py/child.py", protected=_unprotected,
                  editable=_allow, new_file_allowed=_allow)
    effect = file_move(root, "old.py", "sub/new.py", protected=_unprotected,
                       editable=_allow, new_file_allowed=_allow)
    assert effect.changed_paths == ("old.py", "sub/new.py")
    assert not (root / "old.py").exists()
    assert (root / "sub/new.py").read_text() == "value = 1\n"
    assert file_delete(root, "sub/new.py", protected=_unprotected,
                       editable=_allow).changed_paths == ("sub/new.py",)


def test_protected_inode_symlinks_and_directories_refused(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    protected = root / "tests" / "test_source.py"
    protected.parent.mkdir()
    protected.write_text("assert True\n")
    alias = root / "alias.py"
    alias.hardlink_to(protected)
    def is_protected(path):
        return path.exists() and path.samefile(protected)
    with pytest.raises(FileOperationRefused, match="PROTECTED_TEST"):
        file_delete(root, "alias.py", protected=is_protected, editable=_allow)
    (root / "link.py").symlink_to(alias)
    with pytest.raises(FileOperationRefused, match="SYMLINK"):
        file_delete(root, "link.py", protected=_unprotected, editable=_allow)
    (root / "dir").mkdir()
    with pytest.raises(FileOperationRefused, match="IS_DIRECTORY"):
        file_delete(root, "dir", protected=_unprotected, editable=_allow)
    (root / "outer").symlink_to(tmp_path)
    with pytest.raises(FileOperationRefused, match="SYMLINK"):
        file_move(root, "alias.py", "outer/new.py", protected=_unprotected,
                  editable=_allow, new_file_allowed=_allow)
    assert protected.exists() and alias.exists()


def test_case_only_move_preserves_bytes(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    original = root / "Camel.py"
    original.write_bytes(b"hello\n")
    effect = file_move(root, "Camel.py", "camel.py", protected=_unprotected,
                       editable=_allow, new_file_allowed=_allow)
    assert effect.changed_paths == ("Camel.py", "camel.py")
    assert (root / "camel.py").read_bytes() == b"hello\n"
    assert [path.name for path in root.iterdir()] == ["camel.py"]


@pytest.mark.parametrize("sources", [
    {"old.py": None},
    {"old.py": None, "new.py": "value = 1\n"},
])
def test_sealed_candidate_transports_deletions(tmp_path, sources):
    original = tmp_path / "original"
    original.mkdir()
    (original / "old.py").write_text("value = 1\n")
    work = workcopy.create(original, tmp_path / "work")
    run = tmp_path / "run"
    directory = run / "worker" / "results" / "react_done_submissions"
    stored = store_candidate(directory, "task", sources)
    digest = stored["artifact_digest"]
    assert digest != artifact_digest({"old.py": "value = 1\n"})
    record = json.loads((directory / "task" / f"{digest[7:]}.json").read_text())
    assert record["schema"] == "stateframe-done-artifact/2"
    assert "old.py" in record["deletes"]
    (run / "engine-result.json").write_text(json.dumps({
        "result": {"result": "SUBMITTED", "submission": {
            "stored": True, "seal_id": "sha256:" + "0" * 64,
            "artifact_digest": digest}}}))
    candidate = runner.final_candidate(run)
    assert candidate is not None and candidate["old.py"] is None
    workcopy.apply_files(work, candidate, declared={"old.py"})
    assert not (work.copy / "old.py").exists()
    if "new.py" in sources:
        assert (work.copy / "new.py").read_text() == sources["new.py"]
    assert "old.py" in workcopy.changed_files(work)
    (run / "wire").mkdir()
    (run / "summary.json").write_text(json.dumps({
        "engine_result": "SUBMITTED", "files_changed": sorted(workcopy.changed_files(work)),
        "tests_after": {"passed": True}}))
    assert "old.py" in overview(run)


def test_tampered_sealed_deletion_refused(tmp_path):
    run = tmp_path / "run"
    directory = run / "worker" / "results" / "react_done_submissions"
    stored = store_candidate(directory, "task", {"old.py": None})
    digest = stored["artifact_digest"]
    path = directory / "task" / f"{digest[7:]}.json"
    record = json.loads(path.read_text())
    record["deletes"] = ["other.py"]
    path.write_text(json.dumps(record))
    (run / "engine-result.json").write_text(json.dumps({
        "result": {"result": "SUBMITTED", "submission": {
            "stored": True, "seal_id": "sha256:" + "0" * 64,
            "artifact_digest": digest}}}))
    with pytest.raises(runner.RunError, match="digest does not match"):
        runner.final_candidate(run)
    record["deletes"] = ["old.py"]
    record["unsealed_effect"] = "other.py"
    path.write_text(json.dumps(record))
    with pytest.raises(runner.RunError, match="malformed sealed"):
        runner.final_candidate(run)


def test_chat_promote_diff_undo_and_apply_deletion(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "old.py").write_text("obsolete = True\n")
    session = Session(repo, tmp_path / "session", None)
    session.start()
    workspace = session.home / "task-01" / "workspace"
    workspace.mkdir(parents=True)
    record = TaskRecord("remove obsolete module", "SUBMITTED", True, ["old.py"], "",
                        str(workspace.parent), False, attempt=1)
    session.history.append(record)
    session._promote(record)
    assert not (session.copy / "old.py").exists()
    assert "old.py" in session.session_diff()
    assert session.apply() == ["old.py"]
    assert not (repo / "old.py").exists()
    assert session.undo().startswith("undid task")
    assert (session.copy / "old.py").exists()


def test_rename_apply_installs_destination_before_deleting_source(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "a_old.py").write_text("value = 1\n")
    work = workcopy.create(original, tmp_path / "work")
    applied = workcopy.apply_files(
        work, {"a_old.py": None, "z_new.py": b"value = 1\n"},
        declared={"a_old.py"})
    assert applied == ["z_new.py", "a_old.py"]
    assert not (work.copy / "a_old.py").exists()
    assert (work.copy / "z_new.py").read_bytes() == b"value = 1\n"


@pytest.mark.parametrize('raw,code', [('', 'OUTSIDE_WORKSPACE'), ('../x', 'OUTSIDE_WORKSPACE'),
                                    ('missing', 'PATH_NOT_FOUND'), ('dir', 'IS_DIRECTORY')])
def test_file_refusal_contract(tmp_path, raw, code):
    (tmp_path / 'dir').mkdir()
    with pytest.raises(FileOperationRefused) as caught:
        file_delete(tmp_path, raw, protected=_unprotected, editable=_allow)
    assert caught.value.code == code
    assert caught.value.reply().startswith(f'REFUSED {code}: ')

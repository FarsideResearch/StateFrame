import os
import stat
import time
from pathlib import Path

import pytest

from kernel_effects import (ContentStore, EffectRestoreError, WorkspaceEffectPolicy,
                            diff, effect_policy_for_run, manifest_digest, reconcile, scan)


def snapshot(root: Path, store: ContentStore):
    manifest = scan(root, exclude=lambda _: False)
    store.capture(root, manifest)
    return manifest


def policy(protected=lambda _: False, scope=None, new=lambda _: True, baseline=None):
    return WorkspaceEffectPolicy(protected, scope, new, baseline)


def test_same_size_rewrite_with_restored_mtime(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "code.py"
    path.write_text("old")
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    timestamp = path.stat().st_mtime_ns
    path.write_text("new")
    os.utime(path, ns=(timestamp, timestamp))
    after = scan(root, exclude=lambda _: False)
    assert [(effect.path, effect.change) for effect in diff(before, after)] == [("code.py", "modified")]
    result = reconcile(root, before, after, policy(), store, None)
    assert result.applied[0].path == "code.py"
    assert result.manifest["code.py"].digest != before["code.py"].digest
    assert manifest_digest(result.manifest) != manifest_digest(before)


def test_chmod_only_restored(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "code.py"
    path.write_text("code")
    path.chmod(0o644)
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    path.chmod(0o755)
    after = scan(root, exclude=lambda _: False)
    assert diff(before, after)[0].change == "mode"
    result = reconcile(root, before, after, policy(scope=lambda _: False), store, None)
    assert result.restored[0][0].change == "mode"
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_created_hardlink_and_symlink_removed(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    protected = root / "test_code.py"
    protected.write_text("original")
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    os.link(protected, root / "alias")
    os.symlink("test_code.py", root / "shortcut")
    after = scan(root, exclude=lambda _: False)
    assert after["alias"].kind == "other"
    assert after["shortcut"].kind == "symlink"
    result = reconcile(root, before, after,
                       policy(protected=lambda p: p == "test_code.py"), store, None)
    assert {effect.path for effect, _ in result.restored} == {"alias", "shortcut", "test_code.py"}
    assert not (root / "alias").exists()
    assert not (root / "shortcut").is_symlink()
    assert protected.read_text() == "original"


def test_symlink_target_change_same_size_is_detected(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    os.symlink("aaa", root / "link")
    before = scan(root, exclude=lambda _: False)
    (root / "link").unlink()
    os.symlink("bbb", root / "link")
    after = scan(root, exclude=lambda _: False)
    assert [(effect.path, effect.change) for effect in diff(before, after)] == [("link", "modified")]


def test_protected_file_replaced_by_directory_restored_from_baseline(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "test_code.py"
    path.write_text("baseline")
    baseline_store = ContentStore(tmp_path / "baseline")
    baseline = snapshot(root, baseline_store)
    store = ContentStore(tmp_path / "run")
    path.write_text("precall!")
    before = snapshot(root, store)
    path.unlink()
    path.mkdir()
    (path / "child").write_text("unsafe")
    after = scan(root, exclude=lambda _: False)
    result = reconcile(root, before, after,
                       policy(protected=lambda p: p == "test_code.py", baseline=baseline),
                       store, baseline_store)
    assert path.read_text() == "baseline"
    assert result.manifest["test_code.py"] == baseline["test_code.py"]


def test_out_of_scope_edit_restores_precall_not_baseline(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "code.py"
    path.write_text("baseline")
    store = ContentStore(tmp_path / "run")
    path.write_text("precall!")
    before = snapshot(root, store)
    path.write_text("changed!")
    after = scan(root, exclude=lambda _: False)
    reconcile(root, before, after, policy(scope=lambda _: False), store, None)
    assert path.read_text() == "precall!"


def test_run_adapter_uses_declared_edit_scope_and_product_roots(tmp_path):
    root = tmp_path / "workspace"
    source = root / "src"
    source.mkdir(parents=True)
    (source / "allowed.py").write_text("old")
    (source / "other.py").write_text("precall")
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    row = {"files": {"src/allowed.py": {"parent_src": "old"}},
           "allowed_product_paths": ["src"]}
    adapted = effect_policy_for_run(root, row, set(), baseline_manifest=before)
    (source / "allowed.py").write_text("new")
    (source / "other.py").write_text("changed")
    (source / "created.py").write_text("created")
    after = scan(root, exclude=lambda _: False)
    result = reconcile(root, before, after, adapted, store, None)
    assert {effect.path for effect in result.applied} == {"src/allowed.py", "src/created.py"}
    assert {effect.path for effect, _ in result.restored} == {"src/other.py"}
    assert (source / "other.py").read_text() == "precall"


def test_run_adapter_removes_protected_inode_alias_after_original_deleted(tmp_path):
    root = tmp_path / "workspace"
    source = root / "src"
    source.mkdir(parents=True)
    (source / "code.py").write_text("code")
    protected = root / "test_code.py"
    protected.write_text("secret test")
    baseline_store = ContentStore(tmp_path / "baseline")
    baseline = snapshot(root, baseline_store)
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    row = {"files": {"src/code.py": {"parent_src": "code"}},
           "allowed_product_paths": ["src"]}
    adapted = effect_policy_for_run(root, row, {str(protected)}, baseline_manifest=baseline)
    os.link(protected, source / "alias.py")
    protected.unlink()
    after = scan(root, exclude=lambda _: False)
    assert after["src/alias.py"].kind == "file"
    result = reconcile(root, before, after, adapted, store, baseline_store)
    assert not (source / "alias.py").exists()
    assert protected.read_text() == "secret test"
    assert {effect.path for effect, _ in result.restored} == {"src/alias.py", "test_code.py"}


def test_file_directory_replacement_restored(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "folder"
    path.mkdir()
    (path / "original").write_text("saved")
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    (path / "original").unlink()
    path.rmdir()
    path.write_text("replacement")
    after = scan(root, exclude=lambda _: False)
    reconcile(root, before, after, policy(scope=lambda _: False), store, None)
    assert (path / "original").read_text() == "saved"


def test_deleted_parent_of_protected_file_restores_whole_subtree(tmp_path):
    root = tmp_path / "workspace"
    directory = root / "tests"
    directory.mkdir(parents=True)
    protected = directory / "test_code.py"
    protected.write_text("baseline")
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    protected.unlink()
    directory.rmdir()
    after = scan(root, exclude=lambda _: False)
    result = reconcile(root, before, after,
                       policy(protected=lambda path: path == "tests/test_code.py"), store, None)
    assert protected.read_text() == "baseline"
    assert result.manifest == before
    assert {effect.path for effect, _ in result.restored} == {"tests", "tests/test_code.py"}


def test_excluded_tree_is_not_added_to_reconciled_manifest(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "code.py").write_text("old")
    cache = root / "cache"
    cache.mkdir()
    (cache / "ignore").write_text("before")
    store = ContentStore(tmp_path / "run")
    before = scan(root, exclude=lambda path: path == "cache")
    store.capture(root, before)
    (root / "code.py").write_text("new")
    (cache / "ignore").write_text("after")
    after = scan(root, exclude=lambda path: path == "cache")
    result = reconcile(root, before, after, policy(), store, None)
    assert set(result.manifest) == {"code.py"}
    assert (cache / "ignore").read_text() == "after"


def test_restore_failure_is_typed_and_names_path(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "code.py"
    path.write_text("old")
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    path.write_text("new")
    after = scan(root, exclude=lambda _: False)
    original = os.replace

    def denied(source, destination):
        if Path(destination) == path:
            raise PermissionError("read-only parent directory")
        return original(source, destination)

    monkeypatch.setattr(os, "replace", denied)
    with pytest.raises(EffectRestoreError, match="code.py") as raised:
        reconcile(root, before, after, policy(scope=lambda _: False), store, None)
    assert raised.value.paths == ("code.py",)


def test_large_tree_scan_timing(tmp_path, record_property):
    root = tmp_path / "workspace"
    root.mkdir()
    block = b"x" * 25_000
    for index in range(2_000):
        (root / f"f{index:04}.bin").write_bytes(block)
    started = time.perf_counter()
    manifest = scan(root, exclude=lambda _: False)
    elapsed_ms = (time.perf_counter() - started) * 1_000
    record_property("scan_2000_files_50mb_ms", round(elapsed_ms, 2))
    assert len(manifest) == 2_000
    assert sum(entry.size for entry in manifest.values()) == 50_000_000


def test_new_directory_keeps_authorized_child_when_sibling_denied(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    (root / "src/new").mkdir(parents=True)
    (root / "src/new/code.py").write_text("authorized")
    (root / "src/new/secret").write_text("denied")
    result = reconcile(root, before, scan(root, exclude=lambda _: False),
        policy(new=lambda path: path.endswith(".py")), store, None)
    assert (root / "src/new/code.py").read_text() == "authorized"
    assert not (root / "src/new/secret").exists()
    assert {e.path for e in result.applied} == {"src", "src/new", "src/new/code.py"}
    assert {e.path for e, _ in result.restored} == {"src/new/secret"}


@pytest.mark.parametrize("scoped", [False, True])
def test_explicit_scope_flag_controls_existing_non_row_files(tmp_path, scoped):
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src/code.py").write_text("old")
    (root / "src/data.bin").write_bytes(b"old")
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    row = {"files": {"src/code.py": {}}, "allowed_product_paths": ["src"],
           "scoped_edits": scoped}
    adapted = effect_policy_for_run(root, row, set(), baseline_manifest=before)
    assert (adapted.edit_scope is None) is (not scoped)
    (root / "src/data.bin").write_bytes(b"new")
    reconcile(root, before, scan(root, exclude=lambda _: False), adapted, store, None)
    assert (root / "src/data.bin").read_bytes() == (b"old" if scoped else b"new")


def test_empty_product_directory_is_structurally_allowed(tmp_path):
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    row = {"files": {}, "allowed_product_paths": ["src"], "scoped_edits": False}
    adapted = effect_policy_for_run(root, row, set(), baseline_manifest=before)
    (root / "src/empty").mkdir()
    result = reconcile(root, before, scan(root, exclude=lambda _: False), adapted, store, None)
    assert not result.restored
    assert (root / "src/empty").is_dir()


@pytest.mark.parametrize("count,edits,target,scoped", [
    (2, None, None, False), (2, None, "code0.py", False),
    (2, ["code0.py"], None, True), (201, None, "code0.py", True),
])
def test_task_row_records_actual_scope_decision(tmp_path, count, edits, target, scoped):
    from stateframe.task_row import build_row
    from stateframe.workcopy import create

    original = tmp_path / "repo"
    original.mkdir()
    for index in range(count):
        (original / f"code{index}.py").write_text("VALUE = 1\n")
    (original / "test_code.py").write_text("from code0 import VALUE\ndef test_value(): assert VALUE == 1\n")
    work = create(original, tmp_path / "copy")
    row = build_row(work, task="Check the code", edit_paths=edits, target=target)
    assert row["scoped_edits"] is scoped
    assert len(row["files"]) == (1 if scoped else count)

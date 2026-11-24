import errno
import hashlib
import os
import stat
import time
import tracemalloc
from pathlib import Path

import pytest

from kernel_effects import (BaselineStore, ContentStore, EffectError, EffectRestoreError, Entry, WorkspaceEffectPolicy,
                            diff, effect_policy_for_run, manifest_digest, reconcile,
                            reconcile_effects, scan)


def snapshot(root: Path, store: ContentStore):
    manifest = scan(root, exclude=lambda _: False)
    store.capture(root, manifest)
    return manifest


def policy(protected=lambda _: False, scope=None, new=lambda _: True, baseline=None):
    return WorkspaceEffectPolicy(protected, scope, new, baseline)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


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


def test_workspace_under_symlinked_parent_keeps_effect_restore_behavior(tmp_path):
    real_parent = tmp_path / "real-parent"
    root = real_parent / "workspace"
    root.mkdir(parents=True)
    path = root / "code.py"
    path.write_text("before")
    alias = tmp_path / "workspace-link"
    alias.symlink_to(real_parent, target_is_directory=True)
    workspace = alias / "workspace"
    store = ContentStore(tmp_path / "run")
    before = snapshot(workspace, store)

    path.write_text("changed")
    after = scan(workspace, exclude=lambda _: False)
    reconcile(workspace, before, after, policy(scope=lambda _: False), store, None)

    assert path.read_text() == "before"


def test_scan_allows_more_than_20000_workspace_entries(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    for index in range(20_001):
        (root / f"empty-{index:05d}.bin").touch()

    manifest = scan(root, exclude=lambda _: False)

    assert len(manifest) == 20_001


def test_scan_allows_workspace_total_over_256_mib_with_per_file_limit(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    file_size = 16 << 20
    for index in range(17):
        path = root / f"sparse-{index:02d}.bin"
        with path.open("wb") as stream:
            stream.truncate(file_size)

    manifest = scan(root, exclude=lambda _: False)

    assert len(manifest) == 17
    assert sum(entry.size for entry in manifest.values()) == 272 << 20


def test_scan_streams_a_20_mib_repository_file(tmp_path):
    path = tmp_path / "archive.bin"
    size = 20 << 20
    with path.open("wb") as output:
        output.truncate(size)
    expected = hashlib.sha256()
    block = bytes(1 << 20)
    for _ in range(size >> 20):
        expected.update(block)

    manifest = scan(tmp_path, exclude=lambda _: False)

    assert manifest["archive.bin"].size == size
    assert manifest["archive.bin"].digest == expected.hexdigest()


def test_scan_hashes_a_200_mib_file_with_flat_memory(tmp_path):
    path = tmp_path / "archive.bin"
    size = 200 << 20
    with path.open("wb") as output:
        output.truncate(size)
    expected = hashlib.sha256()
    block = bytes(1 << 20)
    for _ in range(size >> 20):
        expected.update(block)

    tracemalloc.start()
    try:
        manifest = scan(tmp_path, exclude=lambda _: False)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert manifest["archive.bin"].size == size
    assert manifest["archive.bin"].digest == expected.hexdigest()
    assert peak < 8 << 20


def test_scan_refuses_a_file_that_changes_during_hashing(tmp_path, monkeypatch):
    import kernel_effects as effects

    path = tmp_path / "archive.bin"
    with path.open("wb") as output:
        output.truncate(20 << 20)
    original_read = os.read
    original = path.stat()
    changed = False

    def changing_read(fd, size):
        nonlocal changed
        block = original_read(fd, size)
        info = os.fstat(fd)
        if not changed and (info.st_dev, info.st_ino) == (original.st_dev, original.st_ino):
            changed = True
            with path.open("r+b") as output:
                output.seek(1 << 20)
                output.write(b"x")
                output.flush()
        return block

    monkeypatch.setattr(effects.os, "read", changing_read)

    with pytest.raises(EffectError, match="changed while scanning"):
        scan(tmp_path, exclude=lambda _: False)


def test_scan_preserves_real_open_errors_instead_of_returning_partial_manifest(
        tmp_path, monkeypatch):
    import kernel_effects as effects

    (tmp_path / "first.bin").write_bytes(b"first")
    failed = tmp_path / "second.bin"
    failed.write_bytes(b"second")
    original_open = os.open

    def fail_open(path, flags, *args, **kwargs):
        if Path(path) == failed:
            raise OSError(errno.EMFILE, "too many open files")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(effects.os, "open", fail_open)

    with pytest.raises(OSError) as raised:
        scan(tmp_path, exclude=lambda _: False)

    assert raised.value.errno == errno.EMFILE
    assert "too many open files" in str(raised.value)


def test_scan_preserves_real_read_errors_instead_of_returning_partial_manifest(
        tmp_path, monkeypatch):
    import kernel_effects as effects

    path = tmp_path / "large.bin"
    with path.open("wb") as output:
        output.truncate(20 << 20)
    original_read = os.read
    identity = path.stat()

    def fail_read(fd, size):
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) == (identity.st_dev, identity.st_ino):
            raise OSError(errno.ENOSPC, "no space left on device")
        return original_read(fd, size)

    monkeypatch.setattr(effects.os, "read", fail_read)

    with pytest.raises(OSError) as raised:
        scan(tmp_path, exclude=lambda _: False)

    assert raised.value.errno == errno.ENOSPC
    assert "no space left on device" in str(raised.value)


def test_renaming_a_large_file_is_a_complete_delete_and_create(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    original = root / "before.bin"
    with original.open("wb") as output:
        output.truncate(20 << 20)
    before = scan(root, exclude=lambda _: False)

    original.rename(root / "after.bin")
    after = scan(root, exclude=lambda _: False)

    assert [(item.path, item.change) for item in diff(before, after)] == [
        ("after.bin", "created"), ("before.bin", "deleted"),
    ]
    assert after["after.bin"].size == 20 << 20


def test_denied_large_file_effect_is_restored_by_streaming_copy(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "payload.bin"
    size = 20 << 20
    with target.open("wb") as output:
        output.truncate(size)
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    expected_digest = before["payload.bin"].digest
    with target.open("r+b") as output:
        output.write(b"changed")

    monkeypatch.setattr(store, "get", lambda _digest: pytest.fail("restore buffered a whole file"))
    after = scan(root, exclude=lambda _: False)
    result = reconcile(root, before, after, policy(scope=lambda _: False), store, None)

    assert result.restored[0][0].path == "payload.bin"
    assert target.stat().st_size == size
    assert _file_digest(target) == expected_digest


def test_denied_large_baseline_effect_is_restored_by_streaming_copy(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "payload.bin"
    with target.open("wb") as output:
        output.truncate(20 << 20)
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    baseline = BaselineStore(tmp_path / "baseline", before)
    baseline.publish(root)
    expected_digest = before["payload.bin"].digest
    with target.open("r+b") as output:
        output.write(b"changed")
    monkeypatch.setattr(baseline, "get", lambda _digest: pytest.fail("restore buffered a whole file"))

    result = reconcile(root, before, scan(root, exclude=lambda _: False),
                       policy(protected=lambda path: path == "payload.bin", baseline=before),
                       store, baseline)

    assert result.restored[0][0].path == "payload.bin"
    assert target.stat().st_size == 20 << 20
    assert _file_digest(target) == expected_digest


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


def test_reconcile_60000_denied_siblings_without_quadratic_root_search(tmp_path):
    count = 60_000
    after = {"generated": Entry("dir", 0o755, None, 0)}
    after.update({f"generated/{index:05d}.txt": Entry("file", 0o644, "0" * 64, 0)
                  for index in range(count)})
    denied_policy = WorkspaceEffectPolicy(
        lambda _path: False, lambda _path: False, lambda _path: True,
        directory_allowed=lambda _path: True,
    )

    result = reconcile_effects(tmp_path, {}, after, denied_policy, object(), None,
                               changes=diff({}, after), observe_final=lambda: {})

    assert len(result.restored) == count + 1
    assert result.applied == []


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


@pytest.mark.parametrize("replacement,expected", [("file", False), ("symlink", True)])
def test_protection_probe_keeps_symlink_alias_protected(
        tmp_path, replacement, expected):
    root = tmp_path / "workspace"
    package = root / "pkg"
    package.mkdir(parents=True)
    hook = package / "hook.py"
    hook.write_text("protected hook", encoding="ascii")
    outside = tmp_path / "outside"
    outside.mkdir()
    before = scan(root, exclude=lambda _: False)
    adapted = effect_policy_for_run(
        root, {"files": {"pkg/hook.py": {}}, "protect_tests": "off"}, set(),
        baseline_manifest=before, hook_paths=frozenset({"pkg/hook.py"}),
    )

    os.link(hook, outside / "alias.py")
    hook.unlink()
    package.rmdir()
    if replacement == "file":
        package.write_text("now a file", encoding="ascii")
    else:
        package.symlink_to(outside, target_is_directory=True)

    assert adapted.protected("pkg/alias.py") is expected


def test_baseline_test_inode_probe_treats_non_directory_ancestor_as_absent(tmp_path):
    root = tmp_path / "workspace"
    package = root / "pkg"
    package.mkdir(parents=True)
    protected = package / "test_api.py"
    protected.write_text("def test_api(): pass\n", encoding="ascii")
    before = scan(root, exclude=lambda _: False)
    protected.unlink()
    package.rmdir()
    package.write_text("now a file", encoding="ascii")

    adapted = effect_policy_for_run(
        root, {"files": {}, "protect_tests": "off"}, {str(protected)},
        baseline_manifest=before,
    )

    assert adapted.protected("pkg/other.py") is False
    assert adapted.protected("pkg/test_api.py") is True


def test_symlink_alias_to_baseline_test_remains_protected(tmp_path):
    root = tmp_path / "workspace"
    protected = root / "pkg" / "test_api.py"
    protected.parent.mkdir(parents=True)
    protected.write_text("def test_api(): pass\n", encoding="ascii")
    (protected.parent / "test_alias.py").symlink_to("test_api.py")
    before = scan(root, exclude=lambda _: False)

    adapted = effect_policy_for_run(
        root, {"files": {}, "protect_tests": "off"}, {str(protected)},
        baseline_manifest=before,
    )

    assert adapted.protected("pkg/test_alias.py") is True


def test_directory_replaced_by_symlink_is_restored_without_touching_target(tmp_path):
    root = tmp_path / "workspace"
    slot = root / "pkg" / "nested" / "slot"
    slot.mkdir(parents=True)
    (slot / "marker.py").write_text("saved", encoding="ascii")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "marker.py").write_text("external", encoding="ascii")
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)

    (slot / "marker.py").unlink()
    slot.rmdir()
    slot.symlink_to(outside, target_is_directory=True)
    after = scan(root, exclude=lambda _: False)
    result = reconcile(root, before, after, policy(scope=lambda _: False), store, None)

    assert (slot / "marker.py").read_text(encoding="ascii") == "saved"
    assert not slot.is_symlink()
    assert (outside / "marker.py").read_text(encoding="ascii") == "external"
    assert result.manifest == before


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


def test_many_new_directories_are_authorized_without_a_pairwise_scan(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    import kernel_effects as effects

    count = 2_000

    class CountedPath(str):
        checks = 0

        def startswith(self, prefix, *args):
            CountedPath.checks += 1
            return super().startswith(prefix, *args)

    directory = effects.Entry("dir", 0o755, None, 0)
    file = effects.Entry("file", 0o644, hashlib.sha256(b"").hexdigest(), 0)
    after = {f"pkg_{index:05d}": directory for index in range(count)}
    after.update({CountedPath(f"pkg_{index:05d}/data.txt"): file for index in range(count)})
    policy = effects.effect_policy_for_run(tmp_path, {
        "files": {"app.py": {}}, "tier": "USER-REPO", "runner_kind": "generic", "protect_tests": "off",
        "scoped_edits": False, "allowed_product_paths": ["app.py"]}, set())
    monkeypatch.setattr(effects, "scan", lambda *_args, **_kwargs: after)
    result = effects.reconcile(tmp_path, {}, after, policy, SimpleNamespace(capture=lambda *_: None), None)
    assert len(result.applied) == len(after)
    assert CountedPath.checks <= 20 * len(after)


@pytest.mark.parametrize("size", [10, 20 << 20])
def test_missing_saved_bytes_leave_the_refused_target_in_place(tmp_path, size):
    import kernel_effects as effects

    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "payload.bin"
    with target.open("wb") as output:
        output.truncate(size)
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    target.unlink()
    target.symlink_to("elsewhere")

    class Missing:
        def get(self, _digest):
            raise effects.EffectError("saved bytes unavailable")

        def copy_to(self, _digest, _output):
            raise effects.EffectError("saved bytes unavailable")

    with pytest.raises(effects.EffectError):
        effects._restore(root, "payload.bin", before, Missing(), None, policy(scope=lambda _: False))
    assert target.is_symlink()
    assert [path.name for path in root.iterdir()] == ["payload.bin"]


def test_store_without_streaming_still_restores(tmp_path):
    import kernel_effects as effects

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "a.txt").write_bytes(b"before")
    store = ContentStore(tmp_path / "run")
    before = snapshot(root, store)
    (root / "a.txt").write_bytes(b"after")

    class Plain:
        def get(self, digest):
            return store.get(digest)

    effects._restore(root, "a.txt", before, Plain(), None, policy(scope=lambda _: False))
    assert (root / "a.txt").read_bytes() == b"before"


def test_object_linked_while_it_is_read_is_not_corrupt(tmp_path, monkeypatch):
    import hashlib
    import os

    import kernel_effects as effects

    path = tmp_path / "object"
    path.write_bytes(b"x" * (3 << 20))
    read, linked = os.read, []

    def link_once(fd, size):
        if not linked:
            linked.append(1)
            os.link(path, tmp_path / "other")
        return read(fd, size)

    monkeypatch.setattr(effects.os, "read", link_once)
    fd = os.open(path, os.O_RDONLY)
    try:
        effects._verify_fd(fd, hashlib.sha256(b"x" * (3 << 20)).hexdigest())
    finally:
        os.close(fd)

import multiprocessing
import json

import pytest

from kernel_mutate_store import MutationStore, project_id
import kernel_mutate_store
from kernel_mutate_build import source_digest


def bundle():
    manifest = {"schema": "stateframe-mutation-unit/1", "name": "x", "revision": 1, "parent": None, "entry": "main.py", "summary": "x", "kind": "reader", "input_schema": {"type": "object"}, "output_schema": {"type": "object"}, "requires": {"network": False, "writes": False}, "smoke": [{"input": {}, "expect": {}}], "limits": {"wall_s": 10, "cpu_s": 5, "mem_mb": 256}}
    files = {"main.py": b"def handle(input, ctx): return {}\n"}
    return source_digest(files, manifest), {"unit.json": json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(), **files}


def _race(home, digest, bundle):
    store = MutationStore(home)
    store.publish(digest, bundle)
    store.gc()


def test_publish_verify_index_and_gc(tmp_path):
    store = MutationStore(tmp_path)
    digest, files = bundle()
    store.publish(digest, files)
    assert store.get(digest) == files
    store.index_add("user", digest)
    assert digest.removeprefix("sha256:") not in store.gc()
    store.index_remove("user", digest)
    assert digest.removeprefix("sha256:") in store.gc()


def test_two_process_publish_gc_race(tmp_path):
    digest, files = bundle()
    processes = [multiprocessing.Process(target=_race, args=(tmp_path, digest, files)) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0


def test_publish_rejects_conflicting_claim(tmp_path):
    digest, files = bundle()
    files["main.py"] = b"different"
    with pytest.raises(ValueError):
        MutationStore(tmp_path).publish(digest, files)


def test_live_lease_roots_then_gc(tmp_path):
    digest, files = bundle()
    store = MutationStore(tmp_path)
    store.publish(digest, files)
    with store.lease("run-1", [digest]):
        assert store.gc() == []
        assert store.get(digest) == files
    assert store.gc() == [digest.removeprefix("sha256:")]


def test_stale_lease_does_not_root_object(tmp_path):
    digest, files = bundle()
    store = MutationStore(tmp_path)
    store.publish(digest, files)
    (store.leases / "dead-run").write_text(json.dumps([digest.removeprefix("sha256:")]))
    assert store.gc() == [digest.removeprefix("sha256:")]
    assert not (store.leases / "dead-run").exists()


def test_index_symlink_is_refused(tmp_path):
    digest, files = bundle()
    store = MutationStore(tmp_path)
    store.publish(digest, files)
    target = tmp_path / "other.json"
    target.write_text("[]")
    (store.index / "user.json").symlink_to(target)
    with pytest.raises(ValueError):
        store.index_add("user", digest)
    assert target.read_text() == "[]"


def test_store_root_symlink_is_refused_before_creating_children(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "mutations").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        MutationStore(tmp_path)
    assert list(outside.iterdir()) == []


def test_get_handles_short_regular_file_reads(tmp_path, monkeypatch):
    digest, files = bundle()
    store = MutationStore(tmp_path)
    store.publish(digest, files)
    original = kernel_mutate_store.os.read
    with monkeypatch.context() as patch:
        patch.setattr(kernel_mutate_store.os, "read", lambda fd, size: original(fd, min(size, 5)))
        assert store.get(digest) == files


def test_project_index_and_powerup_reference_are_gc_roots(tmp_path):
    digest, files = bundle()
    store = MutationStore(tmp_path)
    store.publish(digest, files)
    project = project_id(tmp_path)
    assert project == project_id(tmp_path)
    store.index_add(project, digest)
    assert store.gc() == []
    store.index_remove(project, digest)
    assert store.gc(powerup_refs=[digest]) == []
    assert store.gc() == [digest.removeprefix("sha256:")]


def test_publish_can_atomically_create_retention_reference(tmp_path):
    digest, files = bundle()
    store = MutationStore(tmp_path)
    store.publish(digest, files, scope="user")
    assert store.gc() == []
    assert store.get(digest) == files


def test_get_refuses_object_with_too_many_modules(tmp_path):
    store = MutationStore(tmp_path)
    digest = "a" * 64
    target = store.objects / digest
    target.mkdir()
    (target / "unit.json").write_bytes(b"{}")
    for number in range(9):
        (target / f"m{number}.py").write_bytes(b"")
    with pytest.raises(ValueError):
        store.get(digest, verify=False)


def test_publish_refuses_duplicate_manifest_keys(tmp_path):
    digest, files = bundle()
    duplicate = files["unit.json"].decode().replace('"name":"x"', '"name":"x","name":"x"')
    assert duplicate != files["unit.json"].decode()
    files["unit.json"] = duplicate.encode()
    with pytest.raises(ValueError):
        MutationStore(tmp_path).publish(digest, files)


def test_store_gc_directory_swap_preserves_replacement(tmp_path, monkeypatch):
    store = MutationStore(tmp_path)
    digest, files = bundle()
    store.publish(digest, files)
    name = digest.removeprefix("sha256:")
    target = store.objects / name
    displaced = tmp_path / "displaced"
    original = store._get_unlocked
    swapped = False
    def swap_after_verify(value):
        nonlocal swapped
        result = original(value)
        if not swapped:
            swapped = True
            target.chmod(0o700)
            target.rename(displaced)
            target.mkdir()
            (target / "sentinel").write_text("replacement")
        return result
    monkeypatch.setattr(store, "_get_unlocked", swap_after_verify)
    with pytest.raises((ValueError, OSError)) as error:
        store.gc()
    assert swapped, str(error.value)
    assert (target / "sentinel").exists(), str(error.value)
    assert (target / "sentinel").read_text() == "replacement"

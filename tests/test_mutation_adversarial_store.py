"""Adversarial tests for retained mutation objects and project indexes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
import json

import pytest

from mutate_leftovers import mutation_root, stage_test_unit, sweep_and_assert


def _publish_store_unit(tmp_path, runtime_root):
    import kernel_mutate_build as build
    from kernel_mutate_store import MutationStore

    staged = stage_test_unit(build, runtime_root,
                             "def handle(input, ctx):\n    return {'rows': 1}\n")
    home = tmp_path / "home"
    home.mkdir()
    return MutationStore(home), staged, home


def _bundle(staged):
    return {"unit.json": json.dumps(staged.manifest, sort_keys=True,
                                     separators=(",", ":")).encode(),
            **{path.name: path.read_bytes() for path in staged.dir.glob("*.py")}}


def test_mutation_persisted_store_two_run_race_and_digest_tamper(mutation_root, tmp_path):
    store, staged, home = _publish_store_unit(tmp_path, mutation_root)
    second_store = type(store)(home)
    barrier = threading.Barrier(2)

    def publish(target):
        barrier.wait(timeout=5)
        return target.publish(staged.source_digest, _bundle(staged), scope="user")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(publish, store)
        second = pool.submit(publish, second_store)
        assert first.result(timeout=10) == second.result(timeout=10)

    stored = store.get(staged.source_digest, verify=True)
    object_root = store.objects / staged.source_digest.removeprefix("sha256:")
    source = object_root / "main.py"
    assert source.read_bytes() == b"def handle(input, ctx):\n    return {'rows': 1}\n"
    source.chmod(0o600)
    source.write_bytes(b"def handle(input, ctx): return {'tampered': True}\n")
    with pytest.raises((ValueError, RuntimeError)):
        store.get(staged.source_digest, verify=True)
    sweep_and_assert(mutation_root)


def test_mutation_retained_project_identity_gc_and_tamper(mutation_root, tmp_path):
    store, staged, _home = _publish_store_unit(tmp_path, mutation_root)
    first_project = "a" * 64
    moved_clone = "b" * 64
    store.publish(staged.source_digest, _bundle(staged), scope=first_project)
    store.index_add(first_project, staged.source_digest)
    store.gc()
    assert store.get(staged.source_digest, verify=True)
    assert store._read_index(store._index_path(moved_clone)) == []

    store.index_remove(first_project, staged.source_digest)
    store.gc()
    with pytest.raises((FileNotFoundError, KeyError, LookupError, ValueError)):
        store.get(staged.source_digest, verify=True)
    sweep_and_assert(mutation_root)


def test_mutation_store_active_run_lease_prevents_collection(mutation_root, tmp_path):
    store, staged, _home = _publish_store_unit(tmp_path, mutation_root)
    store.publish(staged.source_digest, _bundle(staged), scope="user")
    with store.lease("run-123", [staged.source_digest]):
        store.index_remove("user", staged.source_digest)
        store.gc()
        assert store.get(staged.source_digest, verify=True)
    store.gc()
    with pytest.raises((FileNotFoundError, KeyError, LookupError, ValueError)):
        store.get(staged.source_digest, verify=True)
    sweep_and_assert(mutation_root)

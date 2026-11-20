"""Bounded multi-process contention coverage for workspace object locks."""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import time
from pathlib import Path


_WORKER = r"""
import hashlib
import io
import sys
from pathlib import Path
from kernel_workspace_objects import ObjectRef, open_workspace_objects

store = open_workspace_objects(Path(sys.argv[1]), "contended", enabled=True)
assert store is not None
worker = int(sys.argv[2])
iterations = int(sys.argv[3])
shared = ObjectRef(sys.argv[4], len(b"shared-reader-payload"))
with store.lease() as lease:
    for iteration in range(iterations):
        payload = f"writer-{worker}-{iteration}".encode()
        ref = ObjectRef(hashlib.sha256(payload).hexdigest(), len(payload))
        lease.publish_many([(ref, lambda payload=payload: io.BytesIO(payload))])
        assert lease.read(ref) == payload
        assert lease.read(shared) == b"shared-reader-payload"
print("completed", flush=True)
"""


def test_workspace_object_read_write_contention_is_bounded(tmp_path, monkeypatch):
    """Concurrent publishers and readers finish promptly and retain valid bytes."""
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    store_parent = tmp_path / "store"
    store_parent.mkdir(mode=0o700)
    from kernel_workspace_objects import ObjectRef, open_workspace_objects

    store = open_workspace_objects(store_parent, "contended", enabled=True)
    assert store is not None
    shared_payload = b"shared-reader-payload"
    shared_ref = ObjectRef(hashlib.sha256(shared_payload).hexdigest(), len(shared_payload))
    with store.lease() as lease:
        lease.publish_many([(shared_ref, lambda: io.BytesIO(shared_payload))])

    process_count = 8
    iterations = 32
    worker_env = os.environ.copy()
    worker_env["STATEFRAME_WORKSPACE_OBJECTS"] = "on"
    engine_path = str(Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine")
    package_path = str(Path(__file__).resolve().parents[1] / "src")
    worker_env["PYTHONPATH"] = os.pathsep.join(
        [engine_path, package_path, worker_env.get("PYTHONPATH", "")]
    )
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(store_parent), str(worker), str(iterations),
             shared_ref.digest],
            env=worker_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for worker in range(process_count)
    ]
    deadline = time.monotonic() + 20
    results = []
    try:
        for worker in workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(worker.args, 20)
            stdout, stderr = worker.communicate(timeout=remaining)
            results.append((worker.returncode, stdout, stderr))
    except subprocess.TimeoutExpired as error:
        stalled = [worker.pid for worker in workers if worker.poll() is None]
        for worker in workers:
            if worker.poll() is None:
                try:
                    worker.kill()
                except ProcessLookupError:
                    pass
        for worker in workers:
            worker.communicate()
        raise AssertionError(
            f"workspace object contention exceeded the 20 second bound; "
            f"workers still running at deadline: {stalled}"
        ) from error

    assert all(code == 0 and output.strip() == "completed" for code, output, _ in results), results
    assert all(not error for _, _, error in results), results

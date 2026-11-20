"""Bound control-state work when a large source history is unchanged."""
from __future__ import annotations

import io
import sys
import zlib
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import kernel_workspace_checkpoint as checkpoint


class _MemoryLease:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.publications = 0
        self.batches = 0
        self.batch_sizes: list[int] = []
        self.reads = 0

    def publish_many(self, entries):
        self.batches += 1
        batch_size = 0
        for ref, source in entries:
            batch_size += ref.size
            if ref.digest in self.objects:
                continue
            with source() as stream:
                data = stream.read()
            assert sha256(data).hexdigest() == ref.digest
            self.objects[ref.digest] = data
            self.publications += 1
        self.batch_sizes.append(batch_size)

    def resolve(self, oid, *, max_bytes):
        assert len(self.objects[oid]) <= max_bytes
        return oid

    def copy_verified(self, oid, output, *, max_bytes):
        self.reads += 1
        data = self.objects[oid]
        assert len(data) <= max_bytes
        output.write(data)


def test_unchanged_large_text_history_uses_object_refs(tmp_path, monkeypatch):
    # The control history holds source text from a repository-sized file set.
    repo = tmp_path / "repo"
    repo.mkdir()
    history = []
    for index in range(4_500):
        source = ("source-\ud800:" if index == 100 else f"path-{index:05d}:") + "x" * 8192
        path = repo / f"path-{index:05d}.py"
        path.write_bytes(source.encode("utf-8", "surrogatepass"))
        history.append(path.read_bytes().decode("utf-8", "surrogatepass"))
    source_reads = 0
    original_read = Path.read_bytes

    def counted_read(path):
        nonlocal source_reads
        if path.is_relative_to(repo):
            source_reads += 1
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read)
    payload = {"history": history, "alias": history[2250], "binary": b"z" * 8192}
    lease = _MemoryLease()
    monkeypatch.setattr(
        checkpoint, "_blob_identity",
        lambda active, ref: (len(active.objects[ref.oid]),)
        if ref.oid in active.objects else None,
    )

    def serialize(previous):
        stream = io.BytesIO()
        writer = checkpoint.engine._StateWriter(stream)
        capture = checkpoint._StateCapture(writer, {
            "hostfns": {}, "participants": {}, "framecells": {}, "framefuncs": {}, "cvars": {},
        }, lease, previous)
        capture.dump(payload)
        if hasattr(capture, "publish_pending"):
            capture.publish_pending()
        writer.finish()
        return capture, writer.count, stream.getvalue()

    started = perf_counter()
    first, first_bytes, _ = serialize({})
    first_elapsed = perf_counter() - started
    first_publications = lease.publications
    first_batches = lease.batches
    prior = {
        identity: (obj, ref, checkpoint._blob_identity(lease, ref))
        for identity, (obj, ref, _, _) in first.leaves.items()
        if sys.getsizeof(obj) >= 4096
    }
    assert len(prior) == len(first.leaves)
    lease.publications = 0
    lease.batches = 0
    started = perf_counter()
    second, second_bytes, control = serialize(prior)
    elapsed = perf_counter() - started
    print(f"first={first_elapsed:.3f}s raw={first_bytes} publishes={first_publications}; "
          f"second={elapsed:.3f}s raw={second_bytes} publishes={lease.publications} "
          f"batches={first_batches}/{lease.batches}")
    assert first_bytes < 1_000_000, (f"control pickle copied {first_bytes} source bytes; "
                                     f"first={first_elapsed:.3f}s second={elapsed:.3f}s "
                                     f"publishes={first_publications}/{lease.publications}")
    assert second_bytes < 1_000_000
    assert lease.publications == 0, "unchanged content was re-encoded or republished"
    assert first_batches <= 2 and lease.batches == 0, "payloads were not batched"
    assert max(lease.batch_sizes) <= 32 << 20, "publication batch exceeded its byte budget"
    assert lease.reads == 0, "serialization fetched stored payload bytes"
    assert source_reads == 0, "serialization reread source files"
    assert elapsed < 2.0, f"unchanged history serialization took {elapsed:.3f}s"
    assert len(second.leaves) == len(first.leaves)

    restored = checkpoint._StateRestore(io.BytesIO(zlib.decompress(control)), {
        "framecells": {}, "framefuncs": {},
    }, lease, set(lease.objects)).load()
    assert restored == payload
    assert restored["history"][2250] is restored["alias"]
    assert isinstance(restored["history"][2250], str)
    assert isinstance(restored["binary"], bytes)

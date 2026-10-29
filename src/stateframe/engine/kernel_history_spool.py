"""Private immutable history, authenticated by a bounded in-memory Merkle root.

Records/index nodes are append-only. Forked checkpoint writers keep their root
while the parent appends. Files live in the run directory and are unlinked and
owned by the spool. Checkpoints store verified incremental blocks, never paths
to temporary files as history authority.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import struct
import tempfile
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from operator import index as integer_index
from pathlib import Path

_HEADER = '<Q'
_HEADER_SIZE = 8
_ENTRY = '<Q32sQ'
_ENTRY_SIZE = 48
_EMPTY = bytes(64 * _ENTRY_SIZE)
_ZERO = (0, bytes(32), 0)
_NODE_CACHE_SIZE = 256
# Process-wide count of HistorySpool pickles that fell back to a full image
# instead of reduce_spool's chunked checkpoint form (MEMGROW2 fold review).
FULL_IMAGE_REDUCTIONS = 0


def _spool_span(name):
    # Import lazily: measurement owns the recorder and also imports history
    # readers during runtime assembly. No recorder means a no-op context.
    from kernel_measurement import profile_active_host_span
    return profile_active_host_span(name)


def legacy() -> bool:
    return os.environ.get('SF_LAB_MEMGROW2', '').strip().lower() == 'legacy'


def journal_directory(journal):
    path = getattr(journal, 'path', None)
    return Path(path).resolve().parent if path is not None else None


def history(directory=None):
    """Set the lab arm once, before constructing/restoring a run."""
    return [] if legacy() else HistorySpool(directory=directory)


def bind_histories(owner, journal):
    directory = journal_directory(journal)
    if directory is not None:
        for name in ('trace', '_receipts'):
            rows = getattr(owner, name, None)
            if isinstance(rows, HistorySpool):
                rows.relocate(directory)


def _restore_history(records):
    # Compatibility with the first MEMGROW2 inline checkpoint representation.
    if legacy():
        return [pickle.loads(record) for record in records]
    result = HistorySpool()
    for record in records:
        result._append_encoded(record)
    return result


def restore_image(cls, blocks, snapshot):
    directory, root, height, length, end, _lineage = snapshot
    result = cls(directory=directory)
    try:
        position = 0
        for block in blocks:
            result._write_all(block, position)
            position += len(block)
        if position != end:
            raise OSError('history checkpoint image length mismatch')
        result._root, result._height = root, height
        result._length, result._end = length, end
        # Independent restores/deepcopies may diverge in the same process.
        # Each gets a new chunk lineage; subsequent live saves remain incremental.
        if length:
            result._node(root, leaf=height == 0)  # authenticate before publication
        if legacy() and not getattr(cls, '_memfix_forced', False):
            values = [pickle.loads(result._encoded(i)) for i in range(length)]
            result.close()
            return values
        return result
    except BaseException:
        result.close()
        raise


class HistorySpool(Sequence):
    """Detached values, at most 256 branch nodes and one scan leaf resident."""

    def __init__(self, directory=None):
        self._owned_directory = None
        self._directory = str(Path(directory).resolve()) if directory is not None else None
        if directory is None:
            # Standalone/pre-runtime holders get an owned run directory. Ignore
            # TMPDIR and tempfile's cwd fallback entirely.
            base = '/private/tmp' if Path('/private/tmp').is_dir() else '/tmp'
            self._owned_directory = tempfile.TemporaryDirectory(prefix='sf-history-run-', dir=base)
            directory = self._owned_directory.name
        try:
            self._file = tempfile.TemporaryFile(mode='w+b', buffering=0, dir=directory)
        except BaseException:
            if self._owned_directory is not None:
                self._owned_directory.cleanup()
            raise
        self._file.write(b'\0')
        self._end = 1
        self._root = _ZERO
        self._height = 0
        self._length = 0
        self._nodes = OrderedDict()
        self._leaf_reference, self._leaf_node = _ZERO, _EMPTY
        self._writer_pid = os.getpid()
        # Do not consume the replay harness's seeded uuid4 wire-ID stream.
        self._lineage = os.urandom(16).hex()

    def __len__(self):
        return self._length

    @property
    def spool_bytes(self):
        return self._end

    def close(self):
        self._file.close()
        self._nodes.clear()
        self._leaf_reference, self._leaf_node = _ZERO, _EMPTY
        if self._owned_directory is not None and os.getpid() == self._writer_pid:
            self._owned_directory.cleanup()

    def __del__(self):
        file = getattr(self, '_file', None)
        if file is not None:
            file.close()
        directory = getattr(self, '_owned_directory', None)
        if directory is not None and os.getpid() == getattr(self, '_writer_pid', None):
            directory.cleanup()

    def relocate(self, directory):
        """Bind a pre-runtime holder to its journal directory without changing aliases."""
        directory = str(Path(directory).resolve())
        if directory == self._directory:
            return
        if os.getpid() != self._writer_pid:
            raise RuntimeError('forked history snapshots are read-only')
        file = tempfile.TemporaryFile(mode='w+b', buffering=0, dir=directory)
        try:
            for start in range(0, self._end, 1024 * 1024):
                block = self._read_exact(start, min(1024 * 1024, self._end - start))
                view = memoryview(block)
                while view:
                    written = file.write(view)
                    if not written:
                        raise OSError('history relocation made no progress')
                    view = view[written:]
        except BaseException:
            file.close()
            raise
        self._file.close()
        self._file = file
        if self._owned_directory is not None:
            self._owned_directory.cleanup()
            self._owned_directory = None
        self._directory = directory

    def _write_all(self, data, position):
        if os.getpid() != self._writer_pid:
            raise RuntimeError('forked history snapshots are read-only')
        with _spool_span('history_spool_write'):
            written = 0
            while written < len(data):
                count = os.pwrite(self._file.fileno(), data[written:], position + written)
                if count <= 0:
                    raise OSError('history spool write made no progress')
                written += count

    def _write(self, payload: bytes):
        position = self._end
        data = struct.pack(_HEADER, len(payload)) + payload
        self._write_all(data, position)
        self._end += len(data)
        return position, hashlib.sha256(data).digest(), len(payload)

    def _read_exact(self, position: int, size: int) -> bytes:
        if position < 0 or size < 0 or position + size > self._end:
            raise OSError('history spool read exceeds snapshot')
        with _spool_span('history_spool_read'):
            parts = []
            remaining = size
            while remaining:
                part = os.pread(self._file.fileno(), remaining, position)
                if not part:
                    raise OSError('history spool is truncated')
                parts.append(part)
                remaining -= len(part)
                position += len(part)
            return b''.join(parts)

    def _read(self, reference) -> bytes:
        position, digest, expected_size = reference
        header = self._read_exact(position, _HEADER_SIZE)
        size = struct.unpack(_HEADER, header)[0]
        if size != expected_size:
            raise OSError('history spool record length mismatch')
        payload = self._read_exact(position + _HEADER_SIZE, size)
        with _spool_span('history_spool_hash_verify'):
            if hashlib.sha256(header + payload).digest() != digest:
                raise OSError('history spool digest mismatch')
        return payload

    def _node(self, reference, *, cache=True, leaf=False):
        if reference == _ZERO:
            return _EMPTY
        if reference == self._leaf_reference:
            return self._leaf_node
        cached = self._nodes.get(reference)
        if cached is not None:
            self._nodes.move_to_end(reference)
            return cached
        node = self._read(reference)
        if len(node) != len(_EMPTY):
            raise OSError('history spool index size mismatch')
        if cache:
            # Sequential readers reuse a leaf for 64 rows. Keep that locality
            # without letting a scan evict the shared branch working set.
            if leaf:
                self._leaf_reference, self._leaf_node = reference, node
            else:
                self._nodes[reference] = node
                if len(self._nodes) > _NODE_CACHE_SIZE:
                    self._nodes.popitem(last=False)
        return node

    def _replace(self, reference, height: int, index: int, value):
        node = bytearray(self._node(reference, cache=False))
        slot = ((index >> (height * 6)) & 63) * _ENTRY_SIZE
        previous = struct.unpack_from(_ENTRY, node, slot)
        child = self._replace(previous, height - 1, index, value) if height else value
        struct.pack_into(_ENTRY, node, slot, *child)
        self._nodes.pop(reference, None)  # do not retain obsolete path versions
        if reference == self._leaf_reference:
            self._leaf_reference, self._leaf_node = _ZERO, _EMPTY
        return self._write(node)

    def _append_encoded(self, encoded: bytes):
        root, height = self._root, self._height
        if self._length == 1 << ((height + 1) * 6):
            node = bytearray(_EMPTY)
            struct.pack_into(_ENTRY, node, 0, *root)
            root, height = self._write(node), height + 1
        record = self._write(encoded)
        root = self._replace(root, height, self._length, record)
        self._root, self._height = root, height
        self._length += 1

    def append(self, value):
        self._append_encoded(pickle.dumps(value, protocol=5))

    def _replace_encoded(self, index: int, encoded: bytes):
        index = self._normalize(index)
        record = self._write(encoded)
        self._root = self._replace(self._root, self._height, index, record)

    def extend(self, values):
        for value in values:
            self.append(value)

    def _position(self, index: int, root, height: int):
        reference = root
        for level in range(height, -1, -1):
            reference = struct.unpack_from(_ENTRY, self._node(reference, leaf=level == 0),
                                           ((index >> (level * 6)) & 63) * _ENTRY_SIZE)
        if not reference[0]:
            raise OSError('history spool index is incomplete')
        return reference

    def _encoded(self, index: int) -> bytes:
        return self._read(self._position(index, self._root, self._height))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        encoded = self._encoded(self._normalize(index))
        with _spool_span('history_spool_decode'):
            return pickle.loads(encoded)

    def _normalize(self, index):
        index = integer_index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError('history index out of range')
        return index

    def __iter__(self) -> Iterator:
        root, height, length = self._root, self._height, len(self)
        for index in range(length):
            encoded = self._read(self._position(index, root, height))
            with _spool_span('history_spool_decode'):
                value = pickle.loads(encoded)
            yield value

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    def _snapshot(self):
        return self._directory, self._root, self._height, len(self), self._end, self._lineage

    def __reduce_ex__(self, protocol):
        from kernel_history_checkpoint import reduce_spool
        reduced = reduce_spool(self)
        if reduced is not None:
            return reduced
        # Visible, never silent: a real checkpoint save must take the chunked path
        # (asserted by test_full_image_fallback_is_counted with a planted control).
        global FULL_IMAGE_REDUCTIONS
        FULL_IMAGE_REDUCTIONS += 1
        blocks =tuple(self._read_exact(start, min(1024 * 1024, self._end - start))
                       for start in range(0, self._end, 1024 * 1024))
        return restore_image, (type(self), blocks, self._snapshot())


def _restore_lookup(events, positions):
    if isinstance(events, list):
        return {key: events[position] for key, position in positions.items()}
    return EventLookup(events, positions)


class EventLookup(Mapping):
    """Timeline's existing id/position metadata, without another event owner."""

    def __init__(self, events, positions):
        self.events, self.positions = events, positions

    def __getitem__(self, key):
        return self.events[self.positions[key]]

    def __iter__(self):
        return iter(self.positions)

    def __len__(self):
        return len(self.positions)

    def __reduce_ex__(self, protocol):
        return _restore_lookup, (self.events, self.positions)

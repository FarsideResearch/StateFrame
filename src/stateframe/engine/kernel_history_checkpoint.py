"""Incremental checkpoint blocks for immutable MEMGROW2 spool files.

Physical append-only blocks include receipt amendments and index versions. The
existing checkpoint chunk index and _ListRef authenticate and reclaim them.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import zlib

_BLOCK_BYTES = 1024 * 1024


def reduce_spool(spool):
    import kernel_checkpoint as checkpoint
    from kernel_history_spool import restore_image

    root = checkpoint._SAVE.get('root')
    if root is None:
        return None
    key = checkpoint._stable_key('memgrow2-spool', spool._lineage)
    chunks = [tuple(row) for row in checkpoint._SAVE.get('index', {}).get(key, ())]
    covered = chunks[-1][4] if chunks else 0
    if covered > spool._end:
        raise checkpoint.CheckpointError('history spool checkpoint prefix exceeds snapshot')
    folder = root / checkpoint.CACHE_DIR
    checkpoint._private_dir(folder)
    while covered < spool._end:
        end = min(covered + _BLOCK_BYTES, spool._end)
        payload = spool._read_exact(covered, end - covered)
        blob = zlib.compress(pickle.dumps([payload], protocol=5), 1)
        ordinal = len(chunks)
        digest = hashlib.sha256(blob).hexdigest()
        name = f'{key}-{ordinal:07d}-{ordinal + 1:07d}-{digest}{checkpoint.CHUNK_SUFFIX}'
        temporary = folder / f'.{name}.tmp'
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'wb') as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, folder / name)
        chunks.append((name, ordinal, ordinal + 1, digest, end))
        covered = end
    checkpoint._SAVE.setdefault('index_out', {})[key] = [list(row) for row in chunks]
    ref = checkpoint._ListRef(key, len(chunks))
    return restore_image, (type(spool), ref, spool._snapshot())

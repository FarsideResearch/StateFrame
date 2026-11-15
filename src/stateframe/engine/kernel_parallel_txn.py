"""Validated redo transitions and durable materialization of parallel commits."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Mapping, Sequence

if __package__:
    from .kernel_parallel_records import (ExecutionState, ProposalEntry, RecoveryDescriptor, TxnDisposition,
                                          TxnRecord, canonical_json)
else:
    from kernel_parallel_records import (ExecutionState, ProposalEntry, RecoveryDescriptor, TxnDisposition,
                                         TxnRecord, canonical_json)


class TxnStoreError(RuntimeError):
    """The transaction log or materialized workspace is corrupt."""


def _manifest_with_overrides(workspace: Path, overrides: Mapping[str, tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    paths = {path.relative_to(workspace).as_posix() for path in Path(workspace).rglob('*')
             if not path.is_dir() or path.is_symlink()}
    paths.update(overrides)
    for relative in sorted(paths):
        path = workspace / relative
        if relative in overrides:
            sha256, mode = overrides[relative]
        else:
            raw_mode = path.lstat().st_mode
            if not stat.S_ISREG(raw_mode):
                raise TxnStoreError(f'unsupported workspace file {relative}')
            sha256, mode = _hash_file(path), stat.S_IMODE(raw_mode)
        digest.update(canonical_json({'path': relative, 'kind': 'file',
                                      'mode': mode, 'sha256': sha256}))
        digest.update(b'\n')
    return digest.hexdigest()


def manifest_digest(workspace: Path) -> str:
    return _manifest_with_overrides(Path(workspace), {})


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pre_manifest_blob(workspace: Path) -> bytes:
    """The authenticated before-state blob supplied beside after-state blobs to decide()."""
    root = Path(workspace)
    result = []
    for path in sorted(root.rglob('*')):
        if path.is_dir() and not path.is_symlink():
            continue
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise TxnStoreError(f'unsupported workspace file {path.relative_to(root)}')
        result.append(canonical_json({'path': path.relative_to(root).as_posix(), 'kind': 'file',
                                      'mode': stat.S_IMODE(mode), 'sha256': _hash_file(path)}) + b'\n')
    return b''.join(result)


def _read_pre_manifest(data: bytes, digest: str) -> dict[str, dict]:
    if hashlib.sha256(data).hexdigest() != digest:
        raise TxnStoreError('pre-manifest digest mismatch')
    rows = {}
    try:
        for line in data.splitlines():
            row = json.loads(line)
            if set(row) != {'path', 'kind', 'mode', 'sha256'} or row['kind'] != 'file':
                raise ValueError('invalid pre-manifest entry')
            path = row['path']
            if (not isinstance(path, str) or path.startswith('/') or '\\' in path or
                    any(part in ('', '.', '..') for part in path.split('/')) or
                    type(row['mode']) is not int or not 0 <= row['mode'] <= 0o7777 or
                    not isinstance(row['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', row['sha256'])):
                raise ValueError('invalid pre-manifest file identity')
            if row['path'] in rows:
                raise ValueError('duplicate pre-manifest path')
            rows[row['path']] = row
    except (ValueError, TypeError, KeyError) as exc:
        raise TxnStoreError(f'invalid pre-manifest: {exc}') from exc
    if b''.join(canonical_json(rows[path]) + b'\n' for path in sorted(rows)) != data:
        raise TxnStoreError('noncanonical pre-manifest')
    return rows


def _expected_post_manifest(before_rows: Mapping[str, dict], entries: Sequence[ProposalEntry]) -> str:
    rows = dict(before_rows)
    for entry in entries:
        rows[entry.path] = {'path': entry.path, 'kind': 'file',
                            'mode': entry.mode, 'sha256': entry.after_sha256}
    digest = hashlib.sha256()
    for path in sorted(rows):
        digest.update(canonical_json(rows[path]))
        digest.update(b'\n')
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_transition(history: list[tuple[TxnRecord, TxnDisposition, str | None]],
                         record: TxnRecord, state: TxnDisposition) -> None:
    if not history:
        if state != 'DECIDED' or record.txn_id != 1 or record.post_generation != record.pre_generation + 1:
            raise TxnStoreError('first transaction must decide id 1 and advance one generation')
        return
    previous, previous_state, _ = history[-1]
    if record.txn_id == previous.txn_id:
        if record.digest() != previous.digest():
            raise TxnStoreError(f'transaction {record.txn_id} changed after decision')
        successor = {'DECIDED': 'MATERIALIZED', 'MATERIALIZED': 'PUBLISHED'}.get(previous_state)
        if state != successor:
            raise TxnStoreError(f'invalid transaction transition {previous_state} -> {state}')
        return
    if previous_state != 'PUBLISHED' or state != 'DECIDED' or record.txn_id != previous.txn_id + 1:
        raise TxnStoreError('transaction ids must be contiguous after publication')
    if record.pre_generation != previous.post_generation or record.post_generation != record.pre_generation + 1:
        raise TxnStoreError('transaction generations are not contiguous')
    if record.pre_manifest_digest != previous.post_manifest_digest:
        raise TxnStoreError('transaction manifest chain does not match')
    if record.pre_binding != previous.post_binding:
        raise TxnStoreError('transaction engine binding chain does not match')


class TxnStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.blobs = self.root / 'blobs'
        self.log = self.root / 'log.jsonl'
        self.torn_last_line = False

    def _repair_tail(self, prefix: bytes, removed: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        repair = canonical_json({'removed_sha256': hashlib.sha256(removed).hexdigest(),
                                 'removed_bytes': len(removed),
                                 'prefix_sha256': hashlib.sha256(prefix).hexdigest()}) + b'\n'
        repairs = self.root / 'repairs.jsonl'
        existing = repairs.read_bytes() if repairs.exists() else b''
        try:
            for line in existing.splitlines(keepends=True):
                if not line.endswith(b'\n'):
                    raise ValueError('unterminated repair record')
                row = json.loads(line)
                if set(row) != {'removed_sha256', 'removed_bytes', 'prefix_sha256'} or canonical_json(row) + b'\n' != line:
                    raise ValueError('invalid repair record')
        except (ValueError, TypeError) as exc:
            raise TxnStoreError(f'corrupt repair log: {exc}') from exc
        if repair not in existing.splitlines(keepends=True):
            repair_descriptor, repair_name = tempfile.mkstemp(dir=self.root, prefix='.repair-')
            try:
                with os.fdopen(repair_descriptor, 'wb') as output:
                    output.write(existing + repair)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(repair_name, repairs)
            finally:
                if os.path.exists(repair_name):
                    os.unlink(repair_name)
            _fsync_dir(self.root)
        descriptor, name = tempfile.mkstemp(dir=self.root, prefix='.log-repair-')
        try:
            with os.fdopen(descriptor, 'wb') as output:
                output.write(prefix)
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, self.log)
            _fsync_dir(self.root)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _read(self, *, repair: bool = True) -> list[tuple[TxnRecord, TxnDisposition, str | None]]:
        if not self.log.exists():
            return []
        data = self.log.read_bytes()
        lines = data.splitlines(keepends=True)
        self.torn_last_line = False
        result = []
        for number, line in enumerate(lines, 1):
            if not line.endswith(b'\n'):
                if number != len(lines):
                    raise TxnStoreError(f'corrupt transaction log line {number}: unterminated line')
                self.torn_last_line = True
                if repair:
                    self._repair_tail(b''.join(lines[:-1]), line)
                break
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise TxnStoreError(f'corrupt transaction log line {number}: {exc}') from exc
            try:
                if set(item) != {'record', 'disposition', 'publication_id'}:
                    raise ValueError('wrong log keys')
                record = TxnRecord.from_json(item['record'])
                state = item['disposition']
                if state not in ('DECIDED', 'MATERIALIZED', 'PUBLISHED'):
                    raise ValueError('unknown disposition')
                publication = item['publication_id']
                if publication is not None and (state != 'PUBLISHED' or not isinstance(publication, str)):
                    raise ValueError('invalid publication id')
                _validate_transition(result, record, state)
                result.append((record, state, publication))
            except (ValueError, TypeError, KeyError) as exc:
                raise TxnStoreError(f'corrupt transaction log line {number}: {exc}') from exc
        return result

    def _append(self, record: TxnRecord, state: TxnDisposition, publication_id: str | None = None) -> None:
        history = self._read()
        _validate_transition(history, record, state)
        self.root.mkdir(parents=True, exist_ok=True)
        line = canonical_json({'record': record.to_json(), 'disposition': state,
                               'publication_id': publication_id}) + b'\n'
        descriptor = os.open(self.log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            written = 0
            while written < len(line):
                count = os.write(descriptor, line[written:])
                if count <= 0:
                    raise TxnStoreError('short write to transaction log')
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_dir(self.root)

    def records(self) -> list[tuple[TxnRecord, TxnDisposition]]:
        history = self._read()
        return [(record, state) for record, state, _ in history]

    def decide(self, rec: TxnRecord, blobs: Mapping[str, bytes]) -> None:
        history = self._read()
        if history and history[-1][0].txn_id == rec.txn_id:
            if history[-1][0].digest() != rec.digest():
                raise TxnStoreError(f'transaction {rec.txn_id} conflicts with existing decision')
            return
        _validate_transition(history, rec, 'DECIDED')
        if len({entry.path for entry in rec.entries}) != len(rec.entries):
            raise TxnStoreError('duplicate transaction entry path')
        for entry in rec.entries:
            if entry.path.startswith('/') or any(part in ('', '.', '..') for part in entry.path.split('/')):
                raise TxnStoreError(f'unsafe transaction path {entry.path}')
            if entry.kind == 'create' and entry.before_sha256 is not None:
                raise TxnStoreError(f'create entry has before digest: {entry.path}')
            if entry.kind == 'modify' and entry.before_sha256 is None:
                raise TxnStoreError(f'modify entry lacks before digest: {entry.path}')
        required = {entry.after_sha256 for entry in rec.entries} | {rec.pre_manifest_digest}
        if set(blobs) != required:
            raise TxnStoreError('blob set must include after-state and pre-manifest blobs')
        before_rows = _read_pre_manifest(blobs[rec.pre_manifest_digest], rec.pre_manifest_digest)
        for entry in rec.entries:
            prior = before_rows.get(entry.path)
            if entry.kind == 'modify' and (prior is None or prior['sha256'] != entry.before_sha256):
                raise TxnStoreError(f'pre-manifest does not match modify entry {entry.path}')
            if entry.kind == 'create' and prior is not None:
                raise TxnStoreError(f'pre-manifest contains create entry {entry.path}')
        if _expected_post_manifest(before_rows, rec.entries) != rec.post_manifest_digest:
            raise TxnStoreError('post-manifest does not match authenticated pre-manifest and entries')
        self.blobs.mkdir(parents=True, exist_ok=True)
        for digest, data in blobs.items():
            if hashlib.sha256(data).hexdigest() != digest:
                raise TxnStoreError(f'blob digest mismatch: {digest}')
            target = self.blobs / digest
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise TxnStoreError(f'corrupt stored blob: {digest}')
                continue
            descriptor, name = tempfile.mkstemp(dir=self.blobs, prefix='.blob-')
            try:
                with os.fdopen(descriptor, 'wb') as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(name, target)
                _fsync_dir(self.blobs)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
        self._append(rec, 'DECIDED')

    def materialize(self, txn_id: int, workspace: Path) -> None:
        history = self._read()
        match = [(record, state) for record, state, _ in history if record.txn_id == txn_id]
        if not match:
            raise TxnStoreError(f'unknown transaction {txn_id}')
        rec, state = match[-1]
        if state not in ('DECIDED', 'MATERIALIZED', 'PUBLISHED'):
            raise TxnStoreError(f'unknown transaction disposition {state}')
        workspace = Path(workspace)
        pre_blob = self.blobs / rec.pre_manifest_digest
        if not pre_blob.is_file():
            raise TxnStoreError(f'missing pre-manifest blob {rec.pre_manifest_digest}')
        before_rows = _read_pre_manifest(pre_blob.read_bytes(), rec.pre_manifest_digest)
        expected_post = _expected_post_manifest(before_rows, rec.entries)
        if expected_post != rec.post_manifest_digest:
            raise TxnStoreError(f'post-manifest does not match authenticated entries for transaction {txn_id}')
        overrides = {entry.path: (entry.after_sha256, entry.mode) for entry in rec.entries}
        if len(overrides) != len(rec.entries):
            raise TxnStoreError(f'duplicate paths in transaction {txn_id}')
        prepared = []
        for entry in rec.entries:
            target = workspace / entry.path
            if entry.path.startswith('/') or any(part in ('', '.', '..') for part in entry.path.split('/')):
                raise TxnStoreError(f'unsafe transaction path {entry.path}')
            parent = workspace
            for part in entry.path.split('/')[:-1]:
                parent = parent / part
                if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                    raise TxnStoreError(f'unsafe transaction parent {parent}')
            if target.is_symlink():
                raise TxnStoreError(f'unsafe transaction path {entry.path}')
            blob = self.blobs / entry.after_sha256
            if not blob.is_file():
                raise TxnStoreError(f'missing blob {entry.after_sha256}')
            data = blob.read_bytes()
            if hashlib.sha256(data).hexdigest() != entry.after_sha256:
                raise TxnStoreError(f'corrupt blob {entry.after_sha256}')
            already = False
            if target.exists():
                raw_mode = target.lstat().st_mode
                if not stat.S_ISREG(raw_mode):
                    raise TxnStoreError(f'non-file target {entry.path}')
                current = _hash_file(target)
                current_mode = stat.S_IMODE(raw_mode)
                if current == entry.after_sha256 and current_mode == entry.mode:
                    already = True
                elif current == entry.before_sha256 and entry.kind == 'modify' and current_mode == before_rows[entry.path]['mode']:
                    pass
                elif current != entry.before_sha256:
                    raise TxnStoreError(f'stale target {entry.path}')
                else:
                    raise TxnStoreError(f'pre-manifest mode mismatch at {entry.path}')
            elif entry.kind != 'create':
                raise TxnStoreError(f'missing modify target {entry.path}')
            prepared.append((entry, target, data, already))
        if _manifest_with_overrides(workspace, overrides) != expected_post:
            raise TxnStoreError(f'post manifest would not match transaction {txn_id}')
        for entry, target, data, already in prepared:
            if already:
                continue
            absent = []
            parent = target.parent
            while not parent.exists():
                absent.append(parent)
                parent = parent.parent
            for directory in reversed(absent):
                directory.mkdir()
                _fsync_dir(directory.parent)
            descriptor, name = tempfile.mkstemp(dir=target.parent, prefix='.parallel-')
            try:
                with os.fdopen(descriptor, 'wb') as output:
                    output.write(data)
                    output.flush()
                    os.fchmod(output.fileno(), entry.mode)
                    os.fsync(output.fileno())
                os.replace(name, target)
                _fsync_dir(target.parent)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
        if manifest_digest(workspace) != rec.post_manifest_digest:
            raise TxnStoreError(f'post manifest mismatch for transaction {txn_id}')
        if state == 'DECIDED':
            self._append(rec, 'MATERIALIZED')

    def mark_published(self, txn_id: int, publication_id: str) -> None:
        history = self._read()
        match = [(record, state, pub) for record, state, pub in history if record.txn_id == txn_id]
        if not match:
            raise TxnStoreError(f'unknown transaction {txn_id}')
        rec, state, prior = match[-1]
        if state == 'PUBLISHED':
            if prior != publication_id:
                raise TxnStoreError(f'publication id mismatch for transaction {txn_id}')
            return
        if state != 'MATERIALIZED':
            raise TxnStoreError(f'transaction {txn_id} is not materialized')
        self._append(rec, 'PUBLISHED', publication_id)


def inspect_recovery(artifact_root: Path, intent: Mapping) -> RecoveryDescriptor:
    root = Path(artifact_root)
    txn_root = root if root.name == 'txn' else root / 'txn'
    if root.name == 'txn':
        root = root.parent
    store = TxnStore(txn_root)
    history = [(record, state) for record, state, _ in store._read(repair=False)]
    latest: dict[int, tuple[TxnRecord, TxnDisposition]] = {}
    for record, state in history:
        latest[record.txn_id] = (record, state)
    call_id = intent.get('call_id')
    if not isinstance(call_id, str):
        raise TxnStoreError('recovery intent lacks call_id')
    if any(record.call_id != call_id for record, _ in latest.values()):
        raise TxnStoreError('recovery intent call_id does not match transaction log')
    proposals = tuple(sorted(path.name for path in (root / 'proposals').iterdir())) if (root / 'proposals').is_dir() else ()
    executions = []
    for value in intent.get('executions', ()):
        try:
            executions.append(ExecutionState.from_json(value))
        except ValueError as exc:
            raise TxnStoreError(f'invalid execution state: {exc}') from exc
    return RecoveryDescriptor(call_id, hashlib.sha256(canonical_json(intent)).hexdigest(),
                              tuple((txn_id, state) for txn_id, (_, state) in sorted(latest.items())),
                              proposals, tuple(executions))

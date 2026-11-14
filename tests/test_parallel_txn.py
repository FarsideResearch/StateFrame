import hashlib
import os
from dataclasses import replace

import pytest

import kernel_parallel_txn as txn
from kernel_parallel_records import EngineBinding, ProposalEntry, TxnRecord, UnitUsage


def sha(data):
    return hashlib.sha256(data).hexdigest()


def record(workspace, entries, blobs, txn_id=1, generation=0, pre_manifest=None, binding=None):
    before = pre_manifest or txn.manifest_digest(workspace)
    blobs[before] = txn.pre_manifest_blob(workspace)
    next_root = workspace.parent / ('expected' + str(txn_id))
    next_root.mkdir()
    for item in workspace.rglob('*'):
        if item.is_file():
            target = next_root / item.relative_to(workspace)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.read_bytes())
    for entry in entries:
        target = next_root / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blobs[entry.after_sha256])
        os.chmod(target, entry.mode)
    identity = binding or EngineBinding('product', 'harness')
    return TxnRecord(txn_id, 'call', 'a', 'execution', generation, generation + 1, tuple(entries),
                     before, txn.manifest_digest(next_root), identity, identity,
                     'contract', None, UnitUsage(1, 1, 10, 2, 30))


def test_decide_materialize_publish_and_idempotent_replay(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'a.py').write_bytes(b'old\n')
    blobs = {sha(b'new\n'): b'new\n', sha(b'created\n'): b'created\n'}
    entries = (ProposalEntry('a.py', 'modify', sha(b'old\n'), sha(b'new\n'), 0o644),
               ProposalEntry('pkg/b.py', 'create', None, sha(b'created\n'), 0o644))
    rec = record(workspace, entries, blobs)
    store = txn.TxnStore(tmp_path / 'txn')
    store.decide(rec, blobs)
    assert store.records() == [(rec, 'DECIDED')]
    store.materialize(1, workspace)
    assert (workspace / 'a.py').read_bytes() == b'new\n'
    assert (workspace / 'pkg/b.py').read_bytes() == b'created\n'
    assert txn.manifest_digest(workspace) == rec.post_manifest_digest
    store.materialize(1, workspace)
    store.mark_published(1, 'receipt')
    store.mark_published(1, 'receipt')
    assert store.records() == [(rec, 'DECIDED'), (rec, 'MATERIALIZED'), (rec, 'PUBLISHED')]
    with pytest.raises(txn.TxnStoreError, match='publication id mismatch'):
        store.mark_published(1, 'other')


def test_decide_rejects_post_manifest_with_unlisted_change(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'a.py').write_bytes(b'a0')
    (workspace / 'other.py').write_bytes(b'o0')
    blobs = {sha(b'a1'): b'a1'}
    entry = ProposalEntry('a.py', 'modify', sha(b'a0'), sha(b'a1'), 0o644)
    valid = record(workspace, (entry,), blobs)
    bogus_root = tmp_path / 'bogus'
    bogus_root.mkdir()
    (bogus_root / 'a.py').write_bytes(b'a1')
    (bogus_root / 'other.py').write_bytes(b'o1')
    bogus = replace(valid, post_manifest_digest=txn.manifest_digest(bogus_root))
    store = txn.TxnStore(tmp_path / 'txn')
    with pytest.raises(txn.TxnStoreError, match='post-manifest.*entries'):
        store.decide(bogus, blobs)
    assert store.records() == []
    assert (workspace / 'other.py').read_bytes() == b'o0'


def test_crash_between_files_replays_identical_manifest(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'a.py').write_bytes(b'a0')
    (workspace / 'b.py').write_bytes(b'b0')
    blobs = {sha(b'a1'): b'a1', sha(b'b1'): b'b1'}
    entries = (ProposalEntry('a.py', 'modify', sha(b'a0'), sha(b'a1'), 0o644),
               ProposalEntry('b.py', 'modify', sha(b'b0'), sha(b'b1'), 0o644))
    rec = record(workspace, entries, blobs)
    store = txn.TxnStore(tmp_path / 'txn')
    store.decide(rec, blobs)
    original = txn.os.replace
    calls = 0
    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('injected crash')
        return original(source, target)
    monkeypatch.setattr(txn.os, 'replace', fail_second)
    with pytest.raises(OSError, match='injected crash'):
        store.materialize(1, workspace)
    assert store.records()[-1] == (rec, 'DECIDED')
    monkeypatch.setattr(txn.os, 'replace', original)
    store.materialize(1, workspace)
    assert txn.manifest_digest(workspace) == rec.post_manifest_digest
    assert (workspace / 'a.py').read_bytes() == b'a1'
    assert (workspace / 'b.py').read_bytes() == b'b1'


def test_corruption_torn_line_and_read_only_inspection(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'a.py').write_bytes(b'a0')
    blobs = {sha(b'a1'): b'a1'}
    entry = ProposalEntry('a.py', 'modify', sha(b'a0'), sha(b'a1'), 0o644)
    rec = record(workspace, (entry,), blobs)
    store = txn.TxnStore(tmp_path / 'txn')
    store.decide(rec, blobs)
    descriptor = txn.inspect_recovery(tmp_path, {'call_id': 'call', 'executions': []})
    assert descriptor.txns == ((1, 'DECIDED'),)
    log_before = store.log.read_bytes()
    store.log.write_bytes(log_before + b'{"record":')
    assert txn.inspect_recovery(tmp_path, {'call_id': 'call', 'executions': []}).txns == ((1, 'DECIDED'),)
    assert store.log.read_bytes() == log_before + b'{"record":'
    assert store.records() == [(rec, 'DECIDED')]
    assert store.torn_last_line is True
    assert store.log.read_bytes() == log_before
    terminated_corruption = b'bad\n' + log_before
    store.log.write_bytes(terminated_corruption)
    with pytest.raises(txn.TxnStoreError, match='corrupt transaction log line 1'):
        store.records()
    assert store.log.read_bytes() == terminated_corruption


def test_valid_json_with_invalid_last_disposition_is_corruption_not_repair(tmp_path):
    store = txn.TxnStore(tmp_path / 'txn')
    store.root.mkdir()
    bad = b'{"record":{},"disposition":"INVALID","publication_id":null}\n'
    store.log.write_bytes(bad)
    with pytest.raises(txn.TxnStoreError, match='corrupt transaction log line 1'):
        store.records()
    assert store.log.read_bytes() == bad
    assert not (store.root / 'repairs.jsonl').exists()


@pytest.mark.parametrize('bad_line', [b'{not json}\n', b'\xff\n'])
def test_invalid_json_final_line_with_newline_is_corruption_not_repaired(tmp_path, bad_line):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'a.py').write_bytes(b'old')
    blobs = {sha(b'new'): b'new'}
    rec = record(workspace, (ProposalEntry('a.py', 'modify', sha(b'old'), sha(b'new'), 0o644),), blobs)
    store = txn.TxnStore(tmp_path / 'txn')
    store.decide(rec, blobs)
    store.log.write_bytes(store.log.read_bytes() + bad_line)
    corrupt_log = store.log.read_bytes()
    with pytest.raises(txn.TxnStoreError, match='corrupt transaction log line 2'):
        store.materialize(1, workspace)
    assert store.log.read_bytes() == corrupt_log
    assert not (store.root / 'repairs.jsonl').exists()
    assert (workspace / 'a.py').read_bytes() == b'old'


def test_bad_blob_and_stale_target_fail_closed(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'a.py').write_bytes(b'a0')
    blobs = {sha(b'a1'): b'a1'}
    entry = ProposalEntry('a.py', 'modify', sha(b'a0'), sha(b'a1'), 0o644)
    rec = record(workspace, (entry,), blobs)
    store = txn.TxnStore(tmp_path / 'txn')
    with pytest.raises(txn.TxnStoreError, match='blob digest mismatch'):
        store.decide(rec, {sha(b'a1'): b'bad', rec.pre_manifest_digest: blobs[rec.pre_manifest_digest]})
    assert store.records() == []
    store.decide(rec, blobs)
    (workspace / 'a.py').write_bytes(b'foreign')
    with pytest.raises(txn.TxnStoreError, match='stale target'):
        store.materialize(1, workspace)
    assert (workspace / 'a.py').read_bytes() == b'foreign'


def test_unrelated_drift_and_symlink_parent_are_refused_before_any_write(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'a.py').write_bytes(b'a0')
    (workspace / 'untouched.py').write_bytes(b'u0')
    blobs = {sha(b'a1'): b'a1'}
    entry = ProposalEntry('a.py', 'modify', sha(b'a0'), sha(b'a1'), 0o644)
    rec = record(workspace, (entry,), blobs)
    store = txn.TxnStore(tmp_path / 'txn')
    store.decide(rec, blobs)
    (workspace / 'untouched.py').write_bytes(b'u1')
    with pytest.raises(txn.TxnStoreError, match='post manifest would not match'):
        store.materialize(1, workspace)
    assert (workspace / 'a.py').read_bytes() == b'a0'

    second = tmp_path / 'second'
    second.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (second / 'link').symlink_to(outside, target_is_directory=True)
    empty_digest = sha(b'')
    post_digest = sha(txn.canonical_json({'path': 'link/payload.py', 'kind': 'file',
                                         'mode': 0o644, 'sha256': sha(b'x')}) + b'\n')
    rec2 = TxnRecord(1, 'call', 'a', 'execution', 0, 1,
                     (ProposalEntry('link/payload.py', 'create', None, sha(b'x'), 0o644),),
                     empty_digest, post_digest, EngineBinding('p', 'h'), EngineBinding('p', 'h'),
                     'contract', None, UnitUsage(0, 0, 0, 0, None))
    store2 = txn.TxnStore(tmp_path / 'txn2')
    store2.decide(rec2, {sha(b'x'): b'x', empty_digest: b''})
    with pytest.raises(txn.TxnStoreError):
        store2.materialize(1, second)
    assert not (outside / 'payload.py').exists()


def test_directory_snapshot_uses_transaction_manifest_algorithm(tmp_path):
    from kernel_parallel_admission import DirectorySnapshot
    (tmp_path / 'a.py').write_bytes(b'alpha')
    assert DirectorySnapshot(tmp_path, 2).manifest_digest == txn.manifest_digest(tmp_path)


def test_published_transaction_materializes_into_fresh_replay_workspace(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'a.py').write_bytes(b'old')
    blobs = {sha(b'new'): b'new'}
    entry = ProposalEntry('a.py', 'modify', sha(b'old'), sha(b'new'), 0o644)
    rec = record(workspace, (entry,), blobs)
    store = txn.TxnStore(tmp_path / 'txn')
    store.decide(rec, blobs)
    store.materialize(1, workspace)
    store.mark_published(1, 'publication')
    replay = tmp_path / 'replay'
    replay.mkdir()
    (replay / 'a.py').write_bytes(b'old')
    store.materialize(1, replay)
    assert (replay / 'a.py').read_bytes() == b'new'
    assert txn.manifest_digest(replay) == rec.post_manifest_digest
    assert store.records() == [(rec, 'DECIDED'), (rec, 'MATERIALIZED'), (rec, 'PUBLISHED')]


@pytest.mark.parametrize('stage', ['DECIDED', 'MATERIALIZED', 'PUBLISHED'])
def test_torn_disposition_append_repairs_and_reaches_one_published_txn(tmp_path, monkeypatch, stage):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'a.py').write_bytes(b'old')
    blobs = {sha(b'new'): b'new'}
    rec = record(workspace, (ProposalEntry('a.py', 'modify', sha(b'old'), sha(b'new'), 0o644),), blobs)
    store = txn.TxnStore(tmp_path / 'txn')
    original = txn.os.write
    triggered = False
    def torn(fd, data):
        nonlocal triggered
        if not triggered and stage.encode() in data:
            triggered = True
            original(fd, data[:12])
            raise OSError('injected append crash')
        return original(fd, data)
    if stage != 'DECIDED':
        store.decide(rec, blobs)
    if stage == 'PUBLISHED':
        store.materialize(1, workspace)
    monkeypatch.setattr(txn.os, 'write', torn)
    with pytest.raises(OSError, match='injected append crash'):
        if stage == 'DECIDED':
            store.decide(rec, blobs)
        elif stage == 'MATERIALIZED':
            store.materialize(1, workspace)
        else:
            store.mark_published(1, 'receipt')
    assert triggered
    monkeypatch.setattr(txn.os, 'write', original)
    removed = store.log.read_bytes().split(b'\n')[-1]
    store = txn.TxnStore(tmp_path / 'txn')
    store.decide(rec, blobs)
    store.materialize(1, workspace)
    store.mark_published(1, 'receipt')
    assert [state for _, state in store.records()] == ['DECIDED', 'MATERIALIZED', 'PUBLISHED']
    repairs = (store.root / 'repairs.jsonl').read_text().splitlines()
    assert len(repairs) == 1
    assert __import__('json').loads(repairs[0])['removed_sha256'] == sha(removed)


def test_mode_drift_before_first_and_partial_apply_refused(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'a.py').write_bytes(b'a0')
    (workspace / 'b.py').write_bytes(b'b0')
    # Freeze the intended 0644 preimage independently of the host's umask/default ACL.
    (workspace / 'a.py').chmod(0o644)
    (workspace / 'b.py').chmod(0o644)
    blobs = {sha(b'a1'): b'a1', sha(b'b1'): b'b1'}
    entries = (ProposalEntry('a.py', 'modify', sha(b'a0'), sha(b'a1'), 0o644),
               ProposalEntry('b.py', 'modify', sha(b'b0'), sha(b'b1'), 0o644))
    rec = record(workspace, entries, blobs)
    store = txn.TxnStore(tmp_path / 'txn')
    store.decide(rec, blobs)
    os.chmod(workspace / 'a.py', 0o600)
    with pytest.raises(txn.TxnStoreError, match='pre.manifest|mode mismatch'):
        store.materialize(1, workspace)
    assert (workspace / 'a.py').read_bytes() == b'a0'
    os.chmod(workspace / 'a.py', 0o644)
    original = txn.os.replace
    calls = 0
    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('injected crash')
        return original(source, target)
    monkeypatch.setattr(txn.os, 'replace', fail_second)
    with pytest.raises(OSError, match='injected crash'):
        store.materialize(1, workspace)
    monkeypatch.setattr(txn.os, 'replace', original)
    os.chmod(workspace / 'b.py', 0o600)
    with pytest.raises(txn.TxnStoreError, match='pre.manifest|mode mismatch'):
        store.materialize(1, workspace)
    assert (workspace / 'b.py').read_bytes() == b'b0'

"""Keep already-bound coverage requests free of journal and registry work."""
from __future__ import annotations

import weakref

from kernel_continuation_journal import ContinuationJournal
import kernel_stateframe_call_coverage as coverage


class _CountingLock:
    def __init__(self, lock):
        self._lock = lock
        self.acquisitions = 0

    def acquire(self, *args, **kwargs):
        self.acquisitions += 1
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        return self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False


class _CountingRegistry:
    def __init__(self):
        self._values = weakref.WeakValueDictionary()
        self.writes = 0

    def get(self, key, default=None):
        return self._values.get(key, default)

    def __setitem__(self, key, value):
        self.writes += 1
        self._values[key] = value


def test_rebinding_and_context_lookup_use_zero_locks_or_registry_writes(tmp_path, monkeypatch):
    journal = ContinuationJournal(tmp_path, "coverage-bind-fast-path", {"origin": "bind-fast-path"})
    journal_lock = _CountingLock(journal._lock)
    monkeypatch.setattr(journal, "_lock", journal_lock)
    owner_calls = 0
    original_assert_owner = journal.assert_owner

    def count_assert_owner():
        nonlocal owner_calls
        owner_calls += 1
        return original_assert_owner()

    monkeypatch.setattr(journal, "assert_owner", count_assert_owner)
    registry = _CountingRegistry()
    registry_lock = _CountingLock(coverage._CONTEXT_INDEX_LOCK)
    monkeypatch.setattr(coverage, "_CONTEXT_INDEXES", registry)
    monkeypatch.setattr(coverage, "_CONTEXT_INDEX_LOCK", registry_lock)

    try:
        context = coverage.bind_index({}, journal)
        index = getattr(journal, "_framefit_call_coverage_index")
        assert context[coverage.CONTEXT_KEY] == index.context_handle
        assert registry.get(index.context_handle) is index
        assert journal_lock.acquisitions > 0
        assert owner_calls > 0
        assert registry_lock.acquisitions == 1
        assert registry.writes == 1

        journal_lock.acquisitions = 0
        owner_calls = 0
        registry_lock.acquisitions = 0
        registry.writes = 0
        for _ in range(100):
            assert coverage.bind_index(context, journal) is context
            assert coverage.context_index(context) is index

        assert owner_calls == 0
        assert journal_lock.acquisitions == 0
        assert registry_lock.acquisitions == 0
        assert registry.writes == 0
    finally:
        journal.close()

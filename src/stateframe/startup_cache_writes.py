"""Keep installation cache publication outside the first-request boundary."""
from __future__ import annotations

from collections import deque
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar, copy_context
import threading


_HOST = ContextVar("startup_cache_writer", default=None)
_CHILD = None
_MAX_PENDING = 512
_DRAIN_SECONDS = 5


class StartupCacheWriter:
    def __init__(self, report):
        self._report = report
        self._ready = threading.Condition()
        self._pending = deque()
        self._thread = None
        self._context = None
        self._closed = False
        self._stores = {}
        self._store_lock = threading.Lock()
        self._readers = ExitStack()

    def reader(self, key, open_store):
        with self._store_lock:
            if self._closed:
                return None
            if key not in self._stores:
                try:
                    self._stores[key] = self._readers.enter_context(open_store())
                except (OSError, ValueError):
                    self._stores[key] = None
            return self._stores[key]

    def submit(self, cache, write):
        if threading.current_thread() is self._thread:
            return self._write(cache, write)
        with self._ready:
            if len(self._pending) < _MAX_PENDING and not self._closed:
                self._pending.append((cache, write))
                self._ready.notify()
                return True
        self._report("cache_write_failed", cache=cache, error="QueueUnavailable")
        return False

    def seal(self):
        with self._ready:
            if self._thread is not None or self._closed:
                return
            self._thread = threading.Thread(target=self._context.run, args=(self._drain,),
                                            name="startup-cache-writer", daemon=True)
            try:
                self._thread.start()
            except RuntimeError:
                self._thread = None
                self._report("cache_write_failed", cache="startup", error="WriterUnavailable")

    def _drain(self):
        while True:
            with self._ready:
                self._ready.wait_for(lambda: self._pending or self._closed)
                if not self._pending:
                    return
                cache, write = self._pending.popleft()
            self._write(cache, write)

    def _write(self, cache, write):
        try:
            return write()
        except Exception as error:
            self._report("cache_write_failed", cache=cache, error=type(error).__name__)
            return False

    def close(self):
        with self._ready:
            self._closed = True
            if self._thread is None:
                self._pending.clear()
            self._ready.notify()
        if self._thread is not None:
            self._thread.join(_DRAIN_SECONDS)
            if self._thread.is_alive():
                self._report("cache_write_failed", cache="startup", error="WriterDrainTimeout")
        with self._store_lock:
            self._readers.close()


def active_writer():
    return _HOST.get() or _CHILD


@contextmanager
def writer_scope(report, *, child=False):
    global _CHILD
    writer = StartupCacheWriter(report)
    previous = _CHILD
    token = _HOST.set(writer)
    writer._context = copy_context()
    if child:
        _CHILD = writer
    try:
        yield writer
    finally:
        try:
            writer.close()
        finally:
            _CHILD = previous
            _HOST.reset(token)


def defer(cache, write):
    writer = active_writer()
    return writer.submit(cache, write) if writer is not None else write()

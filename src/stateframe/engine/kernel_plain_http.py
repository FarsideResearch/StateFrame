"""HTTP transport helpers that reset sockets when a plain provider call is abandoned."""
from __future__ import annotations

import functools
import http.client
import socket
import struct
import threading
import urllib.request


class _TrackedConnectionMixin:
    def __init__(self, *args, tracker: "AbortiveHTTPTransport", tracker_id: int, **kwargs) -> None:
        self._tracker = tracker
        self._tracker_id = tracker_id
        self._abort_armed = True
        self._active_socket: socket.socket | None = None
        super().__init__(*args, **kwargs)
        tracker.register(self)

    def connect(self) -> None:
        if self._tracker.was_aborted(self._tracker_id):
            raise ConnectionError("HTTP request was aborted")
        super().connect()
        active_socket = self.sock
        if active_socket is not None:
            active_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            self._active_socket = active_socket
        if self._tracker.was_aborted(self._tracker_id):
            self.abort()
            raise ConnectionError("HTTP request was aborted")

    def close(self) -> None:
        # urllib calls close on Connection: close before the response body is read.
        if not self._abort_armed:
            super().close()

    def abort(self) -> None:
        if not self._abort_armed:
            return
        self._abort_armed = False
        active_socket = self._active_socket or self.sock
        if active_socket is not None:
            try:
                active_socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            except OSError:
                pass
            try:
                active_socket.shutdown(socket.SHUT_RD)
            except OSError:
                pass
        try:
            super().close()
        except OSError:
            pass

    def finish(self) -> None:
        if not self._abort_armed:
            return
        self._abort_armed = False
        active_socket = self._active_socket or self.sock
        if active_socket is not None:
            try:
                active_socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 0, 0))
            except OSError:
                pass
        try:
            super().close()
        except OSError:
            pass


class _TrackedHTTPConnection(_TrackedConnectionMixin, http.client.HTTPConnection):
    pass


class _TrackedHTTPSConnection(_TrackedConnectionMixin, http.client.HTTPSConnection):
    pass


class _TrackedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, tracker: "AbortiveHTTPTransport", tracker_id: int) -> None:
        self._tracker = tracker
        self._tracker_id = tracker_id
        super().__init__()

    def http_open(self, request):
        factory = functools.partial(
            _TrackedHTTPConnection, tracker=self._tracker, tracker_id=self._tracker_id)
        return self.do_open(factory, request)


class _TrackedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, tracker: "AbortiveHTTPTransport", tracker_id: int) -> None:
        self._tracker = tracker
        self._tracker_id = tracker_id
        super().__init__()

    def https_open(self, request):
        factory = functools.partial(
            _TrackedHTTPSConnection, tracker=self._tracker, tracker_id=self._tracker_id)
        return self.do_open(
            factory, request, context=self._context, check_hostname=self._check_hostname)


class AbortiveHTTPTransport:
    """Track active urllib connections and expose an abort callback to the loop watchdog."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: set[int] = set()
        self._aborted: set[int] = set()
        self._next_id = 0
        self._trackers: dict[int, list[_TrackedConnectionMixin]] = {}

    def was_aborted(self, tracker_id: int) -> bool:
        with self._lock:
            return tracker_id in self._aborted

    def begin(self) -> int:
        with self._lock:
            self._next_id += 1
            tracker_id = self._next_id
            self._active.add(tracker_id)
            self._trackers[tracker_id] = []
            return tracker_id

    def register(self, connection: _TrackedConnectionMixin) -> None:
        tracker_id = connection._tracker_id
        with self._lock:
            self._trackers.setdefault(tracker_id, []).append(connection)
            aborted = tracker_id in self._aborted
        if aborted:
            connection.abort()

    def opener(self, tracker_id: int):
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _TrackedHTTPHandler(self, tracker_id),
            _TrackedHTTPSHandler(self, tracker_id),
        )

    def abort(self, tracker_id: int) -> None:
        with self._lock:
            self._aborted.add(tracker_id)
            connections = tuple(self._trackers.get(tracker_id, ()))
        for connection in connections:
            connection.abort()

    def abort_active(self) -> None:
        with self._lock:
            active = tuple(self._active)
        for tracker_id in active:
            self.abort(tracker_id)

    def finish(self, tracker_id: int) -> None:
        with self._lock:
            connections = tuple(self._trackers.get(tracker_id, ()))
        for connection in connections:
            connection.finish()

    def release(self, tracker_id: int) -> None:
        with self._lock:
            self._active.discard(tracker_id)
            self._aborted.discard(tracker_id)
            self._trackers.pop(tracker_id, None)

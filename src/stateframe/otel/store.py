"""Private SQLite outbox and acknowledgement ledger for historical traces."""
from __future__ import annotations

from contextlib import closing
from dataclasses import replace
import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3
import stat

from .identity import derive
from .levels import ATTRIBUTE_NAMES, Levels, detail_attribute_keys
from .schema import pack_local, unpack_local
from .types import SafeSpan, Signal, SignalBatch

DB_FILENAME = "state.sqlite3"
SCHEMA_VERSION = 2
MAX_DB_BYTES = 28 * 1024 * 1024
_BATCH_BYTES = 96 * 1024
_BATCH_ITEMS = 64


class Store:
    """The caller holds the installation lock for every write and delivery."""

    def __init__(self, root: Path, key: bytes, *, read_only: bool = False):
        self.key = key
        self.resource_id = derive(key, "resource", (), 16)
        self._validated_traces: set[bytes] = set()
        path = root / DB_FILENAME
        if not read_only:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            try:
                os.close(os.open(path, flags, 0o600))
            except FileExistsError:
                pass
        for candidate in (path, path.with_name(path.name + "-journal")):
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                if candidate == path:
                    raise
                continue
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or info.st_mode & 0o077 or info.st_size > MAX_DB_BYTES):
                raise ValueError("state_unavailable")
        mode = "ro" if read_only else "rw"
        self.db = sqlite3.connect(path.as_uri() + "?mode=" + mode, uri=True, timeout=0)
        try:
            self.db.execute("PRAGMA trusted_schema=OFF")
            self.db.execute("PRAGMA temp_store=MEMORY")
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if not read_only:
                self.db.execute("PRAGMA journal_mode=DELETE")
                self.db.execute("PRAGMA synchronous=FULL")
                self.db.execute("PRAGMA secure_delete=ON")
                pages = MAX_DB_BYTES // self.db.execute("PRAGMA page_size").fetchone()[0]
                if self.db.execute(f"PRAGMA max_page_count={pages}").fetchone()[0] != pages:
                    raise ValueError("state_unavailable")
                if version == 0:
                    self._initialize()
                    version = SCHEMA_VERSION
            if version != SCHEMA_VERSION:
                raise ValueError("unsupported_schema")
            expected = self._signature(b"key", b"")
            actual = self.db.execute("SELECT key_check FROM installation").fetchone()
            if actual is None or not hmac.compare_digest(actual[0], expected):
                raise ValueError("state_unavailable")
        except BaseException:
            self.db.close()
            raise

    def close(self) -> None:
        self.db.close()

    def _initialize(self) -> None:
        self.db.executescript(f"""
            BEGIN IMMEDIATE;
            CREATE TABLE installation (key_check BLOB NOT NULL);
            CREATE TABLE runs (
                trace BLOB PRIMARY KEY, runtime TEXT NOT NULL, check_outcome TEXT NOT NULL,
                complete INTEGER NOT NULL, reasons TEXT NOT NULL
            );
            CREATE TABLE spans (
                destination BLOB NOT NULL, trace BLOB NOT NULL, span BLOB NOT NULL,
                payload BLOB, signature BLOB, state TEXT NOT NULL
                    CHECK (state IN ('queued', 'acked', 'partial')),
                UNIQUE(destination, trace, span)
            );
            CREATE INDEX span_delivery ON spans (destination, trace, state);
            PRAGMA user_version={SCHEMA_VERSION};
        """)
        self.db.execute("INSERT INTO installation VALUES (?)", (self._signature(b"key", b""),))
        self.db.commit()

    def record(self, destination: bytes, trace: bytes, spans: tuple[SafeSpan, ...],
               *, runtime: str, check: str, complete: bool, reasons: tuple, sampled: bool,
               levels: Levels = Levels(),
               legacy: frozenset[bytes] = frozenset()) -> None:
        with self.db:
            self._validate_states(trace)
            self.db.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)",
                (trace, runtime, check, int(complete), json.dumps(reasons)),
            )
            if not sampled:
                return
            current: dict[bytes, SafeSpan] = {}
            for span in spans:
                if span.trace_id != trace:
                    raise ValueError("invalid_field")
                current.setdefault(span.span_id, span)
            for prior in (destination, *sorted(legacy)):
                with closing(self.db.execute(
                    "SELECT span, payload, signature FROM spans WHERE destination=? AND trace=? AND state='queued'",
                    (prior, trace),
                )) as rows:
                    for identity, payload, signature in rows:
                        self._authenticate(prior + trace + identity, payload, signature)
                        if identity not in current:
                            current[identity] = self._unpack_span(trace, identity, payload)
                if prior != destination:
                    self.db.execute(
                        "INSERT INTO spans SELECT ?, trace, span, NULL, NULL, state FROM spans "
                        "WHERE destination=? AND trace=? AND state IN ('acked', 'partial') "
                        "ON CONFLICT(destination, trace, span) DO UPDATE SET state=excluded.state, "
                        "payload=NULL, signature=NULL WHERE spans.state='queued' OR excluded.state='acked'",
                        (destination, prior, trace),
                    )
                    self.db.execute("DELETE FROM spans WHERE destination=? AND trace=?", (prior, trace))
            allowed = frozenset(detail_attribute_keys(levels))
            for span in current.values():
                span = _select_detail(span, allowed)
                payload = pack_local(SignalBatch(Signal.TRACES, self.resource_id, (span,)))
                identity = destination + trace + span.span_id
                self.db.execute(
                    "INSERT INTO spans VALUES (?, ?, ?, ?, ?, 'queued') "
                    "ON CONFLICT(destination, trace, span) DO UPDATE SET "
                    "payload=excluded.payload, signature=excluded.signature "
                    "WHERE spans.state='queued' AND spans.signature<>excluded.signature",
                    (destination, trace, span.span_id, payload, self._signature(identity, payload)),
                )

    def destinations(self, trace: bytes) -> frozenset[bytes]:
        self._validate_states(trace)
        rows = self.db.execute("SELECT DISTINCT destination FROM spans WHERE trace=?", (trace,))
        return frozenset(row[0] for row in rows)

    def known(self, trace: bytes) -> bool:
        self._validate_states(trace)
        return self.db.execute("SELECT 1 FROM runs WHERE trace=?", (trace,)).fetchone() is not None

    def partial(self, destination: bytes, trace: bytes) -> bool:
        self._validate_states(trace)
        return self.db.execute(
            "SELECT 1 FROM spans WHERE destination=? AND trace=? AND state='partial' LIMIT 1",
            (destination, trace),
        ).fetchone() is not None

    def next_batch(self, destination: bytes, trace: bytes) -> SignalBatch | None:
        self._validate_states(trace)
        spans = []
        size = 0
        with closing(self.db.execute(
            "SELECT span, payload, signature FROM spans WHERE destination=? AND trace=? AND state='queued' "
            "ORDER BY rowid LIMIT ?",
            (destination, trace, _BATCH_ITEMS),
        )) as rows:
            for identity, payload, authenticator in rows:
                if size + len(payload) > _BATCH_BYTES and spans:
                    break
                self._authenticate(destination + trace + identity, payload, authenticator)
                spans.append(self._unpack_span(trace, identity, payload))
                size += len(payload)
        return SignalBatch(Signal.TRACES, self.resource_id, tuple(spans)) if spans else None

    def settle(self, destination: bytes, trace: bytes, batch: SignalBatch, *, partial: bool) -> None:
        with self.db:
            self._validate_states(trace)
            self.db.executemany(
                "UPDATE spans SET state=?, payload=CASE WHEN ? THEN payload END, "
                "signature=CASE WHEN ? THEN signature END WHERE destination=? AND trace=? AND span=?",
                (("partial" if partial else "acked", partial, partial, destination, trace, span.span_id)
                 for span in batch.spans),
            )

    def status(self, trace: bytes) -> dict[str, object] | None:
        self._validate_states(trace)
        row = self.db.execute(
            "SELECT runtime, check_outcome, complete, reasons FROM runs WHERE trace=?", (trace,),
        ).fetchone()
        if row is None:
            return None
        runtime, check, complete, reasons = row
        counts = {state: count for state, count in self.db.execute(
            "SELECT state, COUNT(*) FROM spans WHERE trace=? GROUP BY state", (trace,),
        )}
        return {
            "runtime": runtime, "check": check, "complete": bool(complete),
            "reasons": json.loads(reasons),
            "signals": {"traces": {"queue_items": counts.get("queued", 0), "acked": counts.get("acked", 0),
                                   "partial_items": counts.get("partial", 0)}},
        }

    def _authenticate(self, identity: bytes, payload: bytes, signature: bytes) -> None:
        if not hmac.compare_digest(signature, self._signature(identity, payload)):
            raise ValueError("state_unavailable")

    def _validate_states(self, trace: bytes) -> None:
        if trace in self._validated_traces:
            return
        damaged = self.db.execute(
            "SELECT 1 FROM spans WHERE trace=? AND state NOT IN ('queued', 'acked', 'partial') LIMIT 1",
            (trace,),
        ).fetchone()
        if damaged is not None:
            raise ValueError("state_unavailable")
        self._validated_traces.add(trace)

    @staticmethod
    def _unpack_span(trace: bytes, identity: bytes, payload: bytes) -> SafeSpan:
        batch = unpack_local(payload)
        if len(batch.spans) != 1 or (batch.spans[0].trace_id, batch.spans[0].span_id) != (trace, identity):
            raise ValueError("state_unavailable")
        return batch.spans[0]

    def _signature(self, identity: bytes, payload: bytes) -> bytes:
        return hmac.new(self.key, b"sf-otel/historical/2" + identity + payload, hashlib.sha256).digest()


def _select_detail(span: SafeSpan, allowed: frozenset[str]) -> SafeSpan:
    return replace(
        span,
        attrs=tuple(attr for attr in span.attrs
                    if attr.key not in ATTRIBUTE_NAMES or attr.key in allowed),
    )

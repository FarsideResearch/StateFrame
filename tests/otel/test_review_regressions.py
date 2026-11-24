"""Regression coverage for exporter privacy, delivery and local recovery."""
from collections import Counter
from contextlib import closing
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
import hashlib
import hmac
import json
import sqlite3
from urllib.parse import urlsplit
import time

import pytest

from stateframe.otel import authority, cli, companion
from stateframe.otel.config import Settings, resolve
from stateframe.otel.identity import from_seed
from stateframe.otel.store import DB_FILENAME, Store
from stateframe.otel.transport import destination_id, send
from stateframe.otel.types import Coverage, Family, Reason, ResultKind
from tests.otel.support import synthetic_batch


@pytest.fixture
def historical(private_root, monkeypatch):
    from tests.otel.test_export_refusals import _prepare_export
    from tests.otlp_receiver import OTLPReceiver

    run = private_root / "run"
    run.mkdir(mode=0o700)
    root = private_root / ".stateframe-otel"
    root.mkdir(mode=0o700)
    _prepare_export(monkeypatch)
    monkeypatch.setattr(companion, "_state_root", lambda **_: root)
    monkeypatch.setattr(companion, "_identity_key", lambda *_, **__: b"k" * 32)
    spec = {"api_key_env": "APP_LOGIN"}
    monkeypatch.setattr(authority, "_read_record",
                        lambda path: spec if path.name == "spec.json" else {"runtime_outcome": "completed"})
    monkeypatch.setattr(companion, "resolve", resolve)
    class Space:
        attempt, unit, incarnation = 1, None, None

    def poll():
        records = (SimpleNamespace(space=Space(), family=Family.JOURNAL), *(
            SimpleNamespace(space=Space(), family=Family.SPEC, row=row) for row in spec.get("_archives", ())
        ))
        return records, Coverage(True, ())

    monkeypatch.setattr(companion, "Reader", lambda *_: SimpleNamespace(poll=poll, close=lambda: None))
    spans = []

    def projector(_key, _run_ref, trace, **_):
        batch = SimpleNamespace(spans=tuple(replace(span, trace_id=trace) for span in spans))
        return SimpleNamespace(
            consume=lambda occurrence: (batch,) if occurrence.family is Family.JOURNAL else (),
            seal=lambda *_: (), coverage=Coverage(True, ()),
        )
    monkeypatch.setattr(companion, "Projector", projector)
    with OTLPReceiver() as receiver:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", receiver.url)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json")
        def export(**flags):
            return companion.export(run, settings=flags.items(), trust_source=True)
        yield root, spec, spans, receiver, export


def _span(**changes):
    return replace(synthetic_batch().spans[0], **changes)


def _legacy_store(root, key):
    with sqlite3.connect(root / DB_FILENAME) as db:
        db.executescript("""
            CREATE TABLE installation (key_check BLOB NOT NULL);
            CREATE TABLE runs (
                trace BLOB PRIMARY KEY, runtime TEXT NOT NULL, check_outcome TEXT NOT NULL,
                complete INTEGER NOT NULL, reasons TEXT NOT NULL
            );
            CREATE TABLE spans (
                destination BLOB NOT NULL, trace BLOB NOT NULL, span BLOB NOT NULL,
                payload BLOB, signature BLOB, state TEXT NOT NULL,
                UNIQUE(destination, trace, span)
            );
            CREATE INDEX span_delivery ON spans (destination, trace, state);
            PRAGMA user_version=2;
        """)
        check = hmac.new(key, b"sf-otel/historical/2key", hashlib.sha256).digest()
        db.execute("INSERT INTO installation VALUES (?)", (check,))
    (root / DB_FILENAME).chmod(0o600)


def test_unknown_legacy_state_stops_export_and_status(historical, monkeypatch, capsys):
    root, _spec, spans, _receiver, export = historical
    _legacy_store(root, b"k" * 32)
    spans.append(_span())
    monkeypatch.setattr(companion, "_send", lambda *_: SimpleNamespace(
        kind=ResultKind.RETRY, accepted=0, reason=Reason.PEER_FAILURE,
    ))
    assert export() == 1
    capsys.readouterr()
    with sqlite3.connect(root / DB_FILENAME) as db:
        assert db.execute("SELECT state FROM spans").fetchone()[0] == "queued"
        db.execute("UPDATE spans SET state='queuet'")
        db.execute("DELETE FROM runs")
    monkeypatch.setattr(companion, "_send", lambda *_: pytest.fail("damaged row reached sender"))

    assert companion.export(root.parent / "run", trust_source=False) == 1
    assert capsys.readouterr().err == "stateframe-otel: export not delivered (local export failure)\n"
    status_code, report = companion.status(root.parent / "run")
    assert status_code == 1
    assert report["reason"] == "state_unavailable"
    assert "reason=state_unavailable" in cli._status_text(report)


def test_partial_delivery_leaves_only_ambiguous_batch_and_drains_later_work(historical, monkeypatch):
    from tests.otlp_receiver import ResponseStep

    root, _, spans, receiver, export = historical
    monkeypatch.setattr("stateframe.otel.store._BATCH_ITEMS", 2)
    receiver._responses = (ResponseStep("partial_success"),)
    spans.extend(_span(span_id=bytes([n]) * 8) for n in (1, 2, 3))
    assert export() == 1
    spans.append(_span(span_id=b"4" * 8))
    assert export() == 1
    received = Counter(
        item["spanId"] for req in receiver.requests for res in req.decoded_json["resourceSpans"]
        for scope in res["scopeSpans"] for item in scope["spans"]
    )
    assert received == Counter(span.span_id.hex() for span in spans)
    with sqlite3.connect(root / DB_FILENAME) as db:
        assert dict(db.execute("SELECT state, count(*) FROM spans GROUP BY state")) == {"acked": 2, "partial": 2}


def test_timeout_change_preserves_earlier_acknowledgements_and_queue(historical):
    root, _, spans, receiver, export = historical
    key = b"k" * 32
    _, trace = from_seed(key, b"s" * 16)
    spans.extend((_span(trace_id=trace, span_id=b"1" * 8),
                  _span(trace_id=trace, span_id=b"2" * 8)))
    parts = (b"sf-otel/destination/1", b"http", b"127.0.0.1",
             str(urlsplit(receiver.url).port).encode(), b"/v1/traces", b"http/json",
             b"none", b"1250", hashlib.sha256().digest(), b"", b"", b"")
    data = b"".join(len(part).to_bytes(4, "big") + part for part in parts)
    old = hmac.digest(key, data, "sha256")[:16]
    with closing(Store(root, key)) as store:
        for span in spans:
            store.record(old, trace, (span,), runtime="completed", check="pass",
                         complete=True, reasons=(), sampled=True)
        batch = store.next_batch(old, trace)
        store.settle(old, trace, replace(batch, spans=(spans[0],)), partial=False)
    assert export(timeout_ms=9000) == 0
    assert len(receiver.requests) == 1
    assert receiver.requests[0].raw_body.count(spans[1].span_id.hex().encode()) == 1
    assert spans[0].span_id.hex().encode() not in receiver.requests[0].raw_body
    with closing(Store(root, key, read_only=True)) as store:
        assert store.status(trace)["signals"]["traces"] == {
            "acked": 2, "queue_items": 0, "partial_items": 0,
        }


def test_invalid_check_record_hash_makes_export_coverage_incomplete(historical, monkeypatch, capsys):
    from stateframe.otel.projection import Projector as SourceProjector
    from stateframe.otel.sources import Reader as SourceReader
    from tests.otel.test_sources import SEED, _event, _journal, _write_json, _write_run

    _root, _spec, _spans, _receiver, export = historical
    run = _root.parent / "run"
    _write_run(run)
    seed = bytes.fromhex(SEED)
    monkeypatch.setattr(companion, "policy_view", lambda _: SimpleNamespace(
        seed=seed, attempt=1, profile="restricted", environment={}, grants=(),
        authority_records=(), source_generation=None,
    ))
    _write_json(run / "summary.json", {
        "runtime_outcome": "completed", "check_outcome": "pass", "artifact": "saved",
    }, root=run)
    receipt = _event(1, "check_receipt", status="PASS", check_kind="unit", execution_id="check-a")
    receipt["record_hash"] = "sha256:" + hashlib.sha256(json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    receipt["status"] = "PAWS"
    _journal(run, [receipt])
    reader = SourceReader(run, b"k" * 32)
    try:
        _occurrences, coverage = reader.poll()
    finally:
        reader.close()
    assert coverage.reasons == ((Reason.SOURCE_CORRUPT, 1),)
    monkeypatch.setattr(companion, "Reader", SourceReader)
    monkeypatch.setattr(companion, "Projector", SourceProjector)

    assert export() == 1
    assert capsys.readouterr().err == "stateframe-otel: export incomplete coverage (source_corrupt)\n"
    status_code, status = companion.status(run)
    assert status_code == 0
    assert status["complete"] is False
    assert status["reasons"] == [["source_corrupt", 1]]


def test_destination_identity_ignores_timeout():
    settings = resolve(Settings(timeout_ms=1250), {}).traces
    assert destination_id(b"k" * 32, settings) == destination_id(b"k" * 32, replace(settings, timeout_ms=9000))


@pytest.mark.parametrize("protocol, status, media, body, accepted", [
    ("http/protobuf", 200, "text/html", b"", False),
    ("http/protobuf", 200, None, b"", False),
    ("http/protobuf", 200, "application/x-protobuf", b"\x0a\x01", False),
    ("http/protobuf", 200, "Application/X-Protobuf", b"", True),
    ("http/json", 201, "application/json", b"{}", False),
    ("http/json", 200, "application/x-protobuf", b"{}", False),
    ("http/json", 200, "application/json", b"{", False),
    ("http/json", 200, "application/json", b'{"accepted":true}', False),
    ("http/json", 200, "application/json; charset=utf-8", b"{}", True),
    ("http/json", 200, "application/json", b"", True),
])
def test_only_a_matching_otlp_response_acknowledges(protocol, status, media, body, accepted):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            if media is not None:
                self.send_header("Content-Type", media)
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            settings = resolve(Settings(protocol=protocol), {
                "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{server.server_port}",
            }).traces
            result = send(synthetic_batch(), settings, time.monotonic_ns() + 3_000_000_000)
        finally:
            server.shutdown()
            thread.join()
    assert (result.kind.value == "accepted") is accepted


@pytest.mark.parametrize("from_file", [False, True])
def test_export_refuses_application_credential_header(historical, monkeypatch, capsys, from_file):
    root, _, _, receiver, export = historical
    marker = "fixture-value"
    monkeypatch.setenv("APP_API_KEY", marker)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://127.0.0.1:4318")
    monkeypatch.setattr(companion, "_send", lambda *_: pytest.fail("credential authorized for send"))
    flags = {}
    if from_file:
        path = root.parent / "otlp.toml"
        path.write_text('[otlp]\nheaders = "x-collector=Bearer%20fixture-value"\n')
        path.chmod(0o600)
        flags["config"] = path
    else:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-collector=Bearer%20fixture-value")
    assert export(**flags) == 2
    assert marker not in repr(capsys.readouterr())
    assert not receiver.requests


@pytest.mark.parametrize("unsafe", ["mode", "dangling", "sidecar"])
def test_purge_refuses_retained_store(private_root, monkeypatch, unsafe):
    monkeypatch.setattr(companion, "safe_home", lambda: private_root)
    root = private_root / ".stateframe-otel"
    assert companion.purge() == 0
    if unsafe == "dangling":
        root.symlink_to(private_root / "missing", target_is_directory=True)
    else:
        root.mkdir(mode=0o700)
        retained = root / (DB_FILENAME if unsafe == "mode" else DB_FILENAME + "-wal")
        retained.touch(mode=0o600)
        if unsafe == "mode":
            root.chmod(0o750)
        else:
            retained.chmod(0o640)
    assert companion.purge() == 2


@pytest.mark.parametrize("size", [0, 7])
def test_interrupted_installation_key_is_recovered_only_without_a_store(private_root, size):
    key = private_root / "identity.key"
    key.write_bytes(bytes(size))
    key.chmod(0o600)
    database = private_root / DB_FILENAME
    database.touch(mode=0o600)
    assert companion._identity_key(private_root, create=True) is None
    assert key.read_bytes() == bytes(size)
    database.unlink()
    assert companion._identity_key(private_root, create=False) is None
    value = companion._identity_key(private_root, create=True)
    assert value is not None and len(value) == 32
    assert companion._identity_key(private_root, create=True) == value


def test_executable_surfaces_pass_shipped_main_gate():
    from tests.test_shipped_main_allowlist import unlisted_mains

    source = Path(__file__).resolve().parents[2] / "src"
    assert unlisted_mains({
        name: (source / name).read_bytes()
        for name in ("stateframe/otel/cli.py", "stateframe/otel/companion.py")
    }) == []


def test_text_status_shows_delivery_counts():
    status = cli._status_text({"signals": {"traces": {"acked": 3, "queue_items": 2, "partial_items": 1}}})
    assert "acked=3 queue_items=2 partial_items=1" in status


def test_present_unreadable_summary_has_a_fixed_refusal(private_root, monkeypatch, capsys):
    from tests.otel.test_export_refusals import _prepare_export

    _prepare_export(monkeypatch, summary=False)
    (private_root / "summary.json").touch()
    assert companion.export(private_root) == 2
    assert capsys.readouterr().err == "stateframe-otel: export summary refused\n"

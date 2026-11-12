"""Smoke contracts and hostile-input bounds for the shipped mutation units."""

import hashlib
import importlib.util
import io
import json
import sqlite3
import subprocess
import sys
import tarfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREBUILT = ROOT / "src" / "stateframe" / "mutations_prebuilt"
BROKER = ROOT / "src" / "stateframe" / "engine" / "sf_mutate_fetch.py"
UNIT_NAMES = (
    "file_facts", "structured_peek", "archive_list", "sqlite_peek",
    "snapshot_diff", "text_grep_struct", "log_watch", "html_extract",
)


class FakeContext:
    def __init__(self, inputs=None, responses=None):
        self.inputs = dict(inputs or {})
        self.responses = dict(responses or {})
        self.logs = []

    def input_bytes(self, input_id, limit=None):
        data = self.inputs[input_id].read_bytes()
        return data if limit is None else data[:limit]

    def input_text(self, input_id, encoding="utf-8", errors="replace", limit=None):
        return self.input_bytes(input_id, limit).decode(encoding, errors=errors)

    def fetch(self, url, accept=None):
        return self.responses[url]

    def fetch_many(self, requests):
        return [self.fetch(url, accept) for url, accept in requests]

    def log(self, message):
        self.logs.append(str(message))

    def propose_edit(self, *_args, **_kwargs):
        raise AssertionError("prebuilt units cannot propose edits")


def _load_unit(name):
    path = PREBUILT / name / "main.py"
    spec = importlib.util.spec_from_file_location(f"mutation_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_digest(unit_dir):
    from kernel_mutate_build import source_digest

    manifest = json.loads((unit_dir / "unit.json").read_text(encoding="utf-8"))
    files = {module.name: module.read_bytes() for module in unit_dir.glob("*.py")}
    return source_digest(files, manifest).removeprefix("sha256:")


def _assert_subset(actual, expected):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        for key, value in expected.items():
            assert key in actual
            _assert_subset(actual[key], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) >= len(expected)
        for got, wanted in zip(actual, expected):
            _assert_subset(got, wanted)
    else:
        assert actual == expected


def _smoke_context(unit_dir, smoke, tmp_path):
    paths = {}
    input_value = smoke["input"]
    input_ids = [input_value[key] for key in
                 ("input_id", "left_id", "right_id") if key in input_value]
    for index, fixture in enumerate(smoke.get("inputs", [])):
        fixture_path = unit_dir / fixture
        assert fixture_path.is_file()
        input_id = input_ids[index] if index < len(input_ids) else f"in{index}"
        paths[input_id] = fixture_path
    responses = {}
    if smoke.get("responses"):
        response_path = unit_dir / smoke["responses"][0]["body"]
        url = smoke["responses"][0]["url"]
        responses[url] = SimpleNamespace(status=200, headers={"content-type": "text/html"},
                                         url=url, body=response_path.read_bytes(), error=None)
    return FakeContext(paths, responses)


@pytest.mark.parametrize("name", UNIT_NAMES)
def test_prebuilt_unit_smoke_and_manifest_digest(name, tmp_path):
    unit_dir = PREBUILT / name
    manifest = json.loads((unit_dir / "unit.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "stateframe-mutation-unit/1"
    assert manifest["name"] == name
    assert manifest["parent"] is None
    assert manifest["requires"]["writes"] is False
    assert manifest["smoke"]
    smoke = manifest["smoke"][0]
    assert smoke.get("expect") is not None
    fixture_paths = [unit_dir / name for name in smoke.get("inputs", [])]
    fixture_paths.extend(unit_dir / row["body"] for row in smoke.get("responses", []))
    fixture_digests = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in fixture_paths}
    module = _load_unit(name)
    ctx = _smoke_context(unit_dir, smoke, tmp_path)
    result = module.handle(smoke["input"], ctx)
    _assert_subset(result, smoke["expect"])
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in fixture_paths} == fixture_digests

    catalog = json.loads((PREBUILT / "manifest.json").read_text(encoding="utf-8"))
    assert catalog["units"][name] == "sha256:" + _source_digest(unit_dir)


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _run(name, value, inputs, responses=None):
    return _load_unit(name).handle(value, FakeContext(inputs, responses))


def test_file_facts_handles_large_single_line_invalid_utf8_and_bom(tmp_path):
    module = _load_unit("file_facts")
    huge = _write(tmp_path, "huge.txt", b"x" * (1 << 20))
    facts = module.handle({"input_id": "in0"}, FakeContext({"in0": huge}))
    assert facts["size_bytes"] == 1 << 20
    assert facts["line_count"] == 1
    assert facts["binary"] is False

    invalid = _write(tmp_path, "invalid.bin", b"\xff\x00\x80")
    facts = module.handle({"input_id": "in0"}, FakeContext({"in0": invalid}))
    assert facts["binary"] is True
    assert facts["encoding"] in {"binary", "unknown"}

    bom = _write(tmp_path, "bom.txt", b"\xef\xbb\xbfhello\r\n")
    facts = module.handle({"input_id": "in0"}, FakeContext({"in0": bom}))
    assert facts["bom"] == "utf-8"
    assert facts["newline"] == "crlf"


def test_structured_peek_bom_json_and_invalid_utf8(tmp_path):
    module = _load_unit("structured_peek")
    good = _write(tmp_path, "data.json", b"\xef\xbb\xbf{\"items\":[1,2]}")
    result = module.handle({"input_id": "in0", "format": "json"},
                           FakeContext({"in0": good}))
    assert result["error"] == ""
    assert result["top_level_type"] == "object"

    bad = _write(tmp_path, "bad.csv", b"name,value\n\xff,2\n")
    result = module.handle({"input_id": "in0", "format": "csv"},
                           FakeContext({"in0": bad}))
    assert result["error"]

    huge = _write(tmp_path, "long.json",
                  b'{"value":"' + b"x" * (1 << 20) + b'"}')
    result = module.handle({"input_id": "in0", "format": "json"},
                           FakeContext({"in0": huge}))
    assert result["error"] == ""
    assert len(result["rows"][0][0]) <= 2048


def test_archive_list_refuses_bombs_bounds_member_count_and_never_extracts(tmp_path):
    module = _load_unit("archive_list")
    traversal = tmp_path / "paths.zip"
    with zipfile.ZipFile(traversal, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("../escape.txt", "escaped")
        archive.writestr("nested/inside.txt", "inside")
    result = module.handle({"input_id": "in0", "member": "../escape.txt"},
                           FakeContext({"in0": traversal}))
    assert any(item["name"] == "../escape.txt" and item["unsafe_name"]
               for item in result["members"])
    assert result["selected_text"] == "escaped"
    assert not (tmp_path.parent / "escape.txt").exists()

    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("large.txt", b"B" * ((64 << 20) + 1))
    result = module.handle({"input_id": "in0"}, FakeContext({"in0": bomb}))
    assert result["error"] == "expanded_size_limit"

    many = tmp_path / "many.zip"
    with zipfile.ZipFile(many, "w") as archive:
        for index in range(5000):
            archive.writestr(f"{index}.txt", b"")
    result = module.handle({"input_id": "in0"}, FakeContext({"in0": many}))
    assert result["error"] == "member_count_limit"


def test_archive_list_reads_supported_tar_compressions(tmp_path):
    module = _load_unit("archive_list")
    for suffix, mode in (("gz", "w:gz"), ("bz2", "w:bz2"), ("xz", "w:xz")):
        buffer = io.BytesIO()
        content = b"tar sample\n"
        with tarfile.open(fileobj=buffer, mode=mode) as archive:
            info = tarfile.TarInfo("docs/readme.txt")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        path = _write(tmp_path, f"sample.tar.{suffix}", buffer.getvalue())
        result = module.handle({"input_id": "in0", "member": "docs/readme.txt"},
                               FakeContext({"in0": path}))
        assert result["error"] == ""
        assert result["selected_text"] == "tar sample\n"


def test_sqlite_peek_malformed_wal_and_uri_special_filename(tmp_path):
    module = _load_unit("sqlite_peek")
    special = tmp_path / "data #?% name.sqlite"
    connection = sqlite3.connect(special)
    try:
        connection.execute("create table items (id integer, label text)")
        connection.execute("insert into items values (1, 'ok')")
        connection.commit()
    finally:
        connection.close()
    Path(str(special) + "-wal").write_bytes(b"wal evidence only")
    result = module.handle({"input_id": "in0"}, FakeContext({"in0": special}))
    assert result["error"] == ""
    assert result["wal_present"] is True
    assert result["tables"][0]["name"] == "items"
    assert result["tables"][0]["rows"] == [["1", "ok"]]

    without_sidecar = _write(tmp_path, "without-sidecar.sqlite", special.read_bytes())
    result = module.handle({"input_id": "in0", "wal_present": True},
                           FakeContext({"in0": without_sidecar}))
    assert result["wal_present"] is True
    result = module.handle({"input_id": "in0", "wal_present": False},
                           FakeContext({"in0": special}))
    assert result["wal_present"] is False

    malformed = _write(tmp_path, "malformed.sqlite", b"not a database")
    result = module.handle({"input_id": "in0"}, FakeContext({"in0": malformed}))
    assert result["error"] == "database_error"


def test_snapshot_diff_is_pure_and_reports_changed_lines(tmp_path):
    left = _write(tmp_path, "left.txt", b"one\ntwo\n")
    right = _write(tmp_path, "right.txt", b"one\nthree\n")
    result = _run("snapshot_diff", {"left_id": "left", "right_id": "right"},
                  {"left": left, "right": right})
    assert result["changed"] is True
    assert result["removed_lines"] == ["two"]
    assert result["added_lines"] == ["three"]

    many = _write(tmp_path, "many.txt",
                  "".join(f"unique-{index}\n" for index in range(12000)).encode())
    bounded = _run("snapshot_diff", {"left_id": "left", "right_id": "right"},
                   {"left": many, "right": many})
    assert bounded["changed"] is False
    assert bounded["truncated"] is True


def test_text_grep_struct_catastrophic_pattern_stops_at_step_budget(tmp_path):
    path = _write(tmp_path, "long.txt", b"a" * 6000 + b"!")
    result = _run("text_grep_struct", {"input_id": "in0", "pattern": "(a+)+$"},
                  {"in0": path})
    assert result["error"] == "step_budget_exceeded"
    assert 0 < result["steps"] <= result["step_limit"]


def test_text_grep_struct_rejects_unsupported_regex_escapes(tmp_path):
    path = _write(tmp_path, "letters.txt", b"dddd\n")
    result = _run("text_grep_struct", {"input_id": "in0", "pattern": r"[\d]+"},
                  {"in0": path})
    assert result["error"] == "character_class_escape_unsupported"


def test_log_watch_reports_rotation_and_truncation_from_snapshot_facts(tmp_path):
    module = _load_unit("log_watch")
    tail = _write(tmp_path, "service.log", b"WARN retry\nERROR failed\n")
    result = module.handle({
        "input_id": "in0", "patterns": ["WARN", "ERROR"],
        "previous": {"identity": "old", "size": 500, "start": 400},
        "current": {"identity": "new", "size": 80, "start": 20},
    }, FakeContext({"in0": tail}))
    assert result["snapshot_change"] == "rotated_truncated"
    assert result["counts"] == [{"pattern": "WARN", "count": 1},
                                {"pattern": "ERROR", "count": 1}]


def test_html_extract_bounds_hostile_html_and_filters_script_content():
    module = _load_unit("html_extract")
    url = "https://example.test/page"
    links = "".join(f"<a href='/link-{index}'>link</a>" for index in range(250))
    tables = "".join("<table>" + "<tr><td>cell</td></tr>" * 205 + "</table>"
                     for _ in range(12))
    hostile = ("<script>secret</script>" + "<div>" * 30000 +
               "<title>Hostile</title><p>visible</p>" + "</div>" * 30000 +
               "<a href='/ok'>link</a>" + links + tables +
               "<p><a href='javascript:bad'>malformed" + "x" * (10 << 20)).encode("utf-8")
    response = SimpleNamespace(status=200, headers={"content-type": "text/html"},
                               url=url, body=hostile, error=None)
    result = module.handle({"url": url}, FakeContext(responses={url: response}))
    assert result["title"] == "Hostile"
    assert "visible" in result["text"]
    assert "secret" not in result["text"]
    assert len(result["text"].encode("utf-8")) <= 64 << 10
    assert len(result["links"]) <= 200
    assert result["links"][0] == "https://example.test/ok"
    assert len(result["tables"]) <= 10
    assert len(result["tables"][0]) == 200


class BrokerHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", f"http://other.test:{self.server.server_port}/data")
            self.end_headers()
            return
        body = b"broker response"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def broker_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), BrokerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _parse_broker_frame(data):
    line, payload = data.split(b"\n", 1)
    tag, json_length, body_length = line.split(b" ")
    assert tag == b"SFMB1"
    json_size = int(json_length)
    raw_size = int(body_length)
    metadata = json.loads(payload[:json_size])
    body = payload[json_size:]
    assert len(body) == raw_size
    return metadata, body


def _parse_broker_frames(data):
    frames = []
    while data:
        line, payload = data.split(b"\n", 1)
        tag, json_length, body_length = line.split(b" ")
        assert tag == b"SFMB1"
        json_size = int(json_length)
        raw_size = int(body_length)
        metadata = json.loads(payload[:json_size])
        body = payload[json_size:json_size + raw_size]
        assert len(body) == raw_size
        frames.append((metadata, body))
        data = payload[json_size + raw_size:]
    return frames


@pytest.mark.host_sandbox
def test_fetch_broker_subprocess_returns_sfmb1_frame_and_raw_body(broker_server):
    engine = ROOT / "src" / "stateframe" / "engine"
    url = f"http://broker.test:{broker_server}/data"
    request = json.dumps({"url": url, "accept": "text/plain", "max_bytes": 1024,
                          "timeout_s": 2, "domains": ["broker.test"]}).encode()
    runner = ("import sys; sys.path.insert(0, " + repr(str(engine)) + "); "
              "import socket, sf_mutate_fetch; "
              "sf_mutate_fetch.run_once(sys.stdin.buffer, sys.stdout.buffer, "
              "allow_private_for_tests=True, resolver=lambda host, port, *, type: "
              "[(socket.AF_INET, socket.SOCK_STREAM, 0, '', ('127.0.0.1', port))])")
    completed = subprocess.run([sys.executable, "-c", runner], input=request,
                               capture_output=True, timeout=15, check=False)
    assert completed.returncode == 0, completed.stderr
    metadata, body = _parse_broker_frame(completed.stdout)
    assert metadata == {"status": 200, "headers": {"content-type": "text/plain",
                                                   "content-length": "15",
                                                   "server": "BaseHTTP/0.6 Python/"
                                                   + sys.version.split()[0],
                                                   "date": metadata["headers"]["date"]},
                        "final_url": url, "redirects": [], "error": None}
    assert body == b"broker response"


@pytest.mark.host_sandbox
def test_fetch_broker_revalidates_domain_allowlist_on_redirect(broker_server):
    engine = ROOT / "src" / "stateframe" / "engine"
    url = f"http://broker.test:{broker_server}/redirect"
    request = json.dumps({"url": url, "accept": None, "max_bytes": 1024,
                          "timeout_s": 2, "domains": ["broker.test"]}).encode()
    runner = ("import socket, sys; sys.path.insert(0, " + repr(str(engine)) + "); "
              "import sf_mutate_fetch; "
              "sf_mutate_fetch.run_once(sys.stdin.buffer, sys.stdout.buffer, "
              "allow_private_for_tests=True, resolver=lambda host, port, *, type: "
              "[(socket.AF_INET, socket.SOCK_STREAM, 0, '', ('127.0.0.1', port))])")
    completed = subprocess.run([sys.executable, "-c", runner], input=request,
                               capture_output=True, timeout=15, check=False)
    assert completed.returncode == 0, completed.stderr
    metadata, body = _parse_broker_frame(completed.stdout)
    assert metadata["error"] == "domain_not_allowed"
    assert metadata["redirects"] == [f"http://other.test:{broker_server}/data"]
    assert body == b""


def test_fetch_broker_enforces_request_size_and_time_caps():
    engine = ROOT / "src" / "stateframe" / "engine"
    base = {"url": "https://example.test/", "accept": None,
            "max_bytes": (8 << 20) + 1, "timeout_s": 2, "domains": None}
    oversized = dict(base)
    slow = dict(base, max_bytes=1024, timeout_s=20.01)
    runner = ("import io, json, sys; sys.path.insert(0, " + repr(str(engine)) + "); "
              "import sf_mutate_fetch; "
              "requests = [" + repr(json.dumps(oversized)) + ", " +
              repr(json.dumps(slow)) + "]; "
              "[sf_mutate_fetch.run_once(io.BytesIO(item.encode()), sys.stdout.buffer) "
              "for item in requests]")
    completed = subprocess.run([sys.executable, "-c", runner], capture_output=True,
                               timeout=15, check=False)
    assert completed.returncode == 0, completed.stderr
    frames = _parse_broker_frames(completed.stdout)
    assert [metadata["error"] for metadata, _body in frames] == [
        "invalid_request", "invalid_request"]
    assert all(body == b"" for _metadata, body in frames)


def test_fetch_broker_cli_frames_bad_request_and_exits_zero():
    completed = subprocess.run([sys.executable, str(BROKER)], input=b"not json",
                               capture_output=True, timeout=15, check=False)
    assert completed.returncode == 0
    metadata, body = _parse_broker_frame(completed.stdout)
    assert metadata["error"] == "invalid_request"
    assert metadata["status"] is None
    assert body == b""


def test_fetch_broker_crash_exits_nonzero_without_error_frame():
    engine = ROOT / "src" / "stateframe" / "engine"
    request = json.dumps({"url": "https://example.test/", "accept": None,
                          "max_bytes": 1024, "timeout_s": 2,
                          "domains": ["example.test"]}).encode()
    runner = ("import sys; sys.path.insert(0, " + repr(str(engine)) + "); "
              "import sf_mutate_fetch; "
              "sf_mutate_fetch.fetch_raw = lambda *args, **kwargs: "
              "(_ for _ in ()).throw(SystemExit(7)); sf_mutate_fetch.main()")
    completed = subprocess.run([sys.executable, "-c", runner], input=request,
                               capture_output=True, timeout=15, check=False)
    assert completed.returncode == 7
    assert completed.stdout == b""

"""Adversarial tests for raw broker responses and bundled parsers."""
from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
import socket
import sqlite3
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import zipfile

import pytest

from mutate_leftovers import make_launch_spec, mutation_root, sweep_and_assert


def _prebuilt_bundle(_build, _runtime_root: Path, name: str):
    package = importlib.resources.files("stateframe.mutations_prebuilt").joinpath(name)
    manifest = json.loads(package.joinpath("unit.json").read_text(encoding="utf-8"))
    files = {entry.name: entry.read_bytes() for entry in package.iterdir()
             if entry.name.endswith(".py") and entry.is_file()}
    return files, manifest


def _prebuilt_input(manifest: dict, *, member: str | None = None) -> dict:
    schema = manifest["input_schema"]
    props = schema.get("properties", {})
    result = {}
    for name, spec in props.items():
        if name in {"path_id", "input_id", "archive_id", "database_id", "file_id"}:
            result[name] = "in0"
        elif name == "member":
            result[name] = member or "bomb.txt"
        elif spec.get("type") == "boolean":
            result[name] = False
        elif "enum" in spec:
            result[name] = spec["enum"][0]
        elif spec.get("type") == "string":
            result[name] = ""
        elif spec.get("type") == "integer":
            result[name] = 0
        elif spec.get("type") == "number":
            result[name] = 0
        elif spec.get("type") == "array":
            result[name] = []
        elif spec.get("type") == "object":
            result[name] = {}
    return result


def _execute(execution, wire, files, runtime_root, *, manifest, input_value, input_files,
             call="prebuilt-case"):
    import kernel_mutate_build as build

    executor = execution.MutationExecutor(runtime_root, network_envelope="none")
    try:
        staged = build.stage_unit(files, manifest, executor.root)
        outcome = executor.run(make_launch_spec(execution, wire, staged, call=call,
                                                input_value=input_value, input_files=input_files))
    finally:
        cleanup = executor.close()
    assert cleanup.ok, cleanup
    sweep_and_assert(runtime_root)
    return outcome


@pytest.mark.host_sandbox
def test_mutation_html_extract_broker_preserves_structure():
    from kernel_web_fetch import WebPolicy, fetch_raw

    html = b"<html><title>Ledger</title><table><tr><td>Q1</td><td>42</td></tr></table></html>"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        policy = WebPolicy(("127.0.0.1",), max_body=4096, allowed_ports=(server.server_port,),
                           allow_private_for_tests=True)
        result = fetch_raw(f"http://127.0.0.1:{server.server_port}/report", policy,
                           max_bytes=4096,
                           resolver=lambda host, port, **_kw: [
                               (socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, port))])
        assert result.status == 200
        assert result.body == html
        assert b"<table>" in result.body and b"<td>42</td>" in result.body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.host_sandbox
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux bubblewrap/seccomp contain prebuilt archive payloads",
)
def test_mutation_archive_zip_bomb_and_member_traversal(mutation_root, tmp_path):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    archive = tmp_path / "hostile.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        bundle.writestr("bomb.txt", b"A" * (65 << 20))
        bundle.writestr("../../escape.txt", b"must never be extracted")
    files, manifest = _prebuilt_bundle(build, mutation_root, "archive_list")
    outcome = _execute(execution, wire, files, mutation_root, manifest=manifest,
                       input_value=_prebuilt_input(manifest, member="bomb.txt"),
                       input_files={"in0": archive})
    assert outcome.cleanup.ok
    assert len(outcome.stdout) <= execution.MutationLimits().output_bytes
    assert outcome.exit_code == 0
    frame = wire.parse_stdout(outcome.stdout, call="prebuilt-case")
    assert frame.kind == "result"
    assert frame.result["error"] == "expanded_size_limit"
    assert frame.result["selected_text"] == ""
    assert len(json.dumps(frame.result or frame.error)) < 1 << 20
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path.parent / "escape.txt").exists()


@pytest.mark.host_sandbox
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux bubblewrap/seccomp isolate prebuilt SQLite payloads",
)
def test_mutation_sqlite_immutable_wal_uri_and_malformed_db(mutation_root, tmp_path):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_runtime import admit_inputs

    database = tmp_path / "ledger?part#one.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE ledger (value TEXT)")
        connection.commit()
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("INSERT INTO ledger VALUES ('only-in-wal')")
        connection.commit()
        assert Path(str(database) + "-wal").exists()

        snapshot = admit_inputs(tmp_path, [database.name], MutationPolicy(level="prebuilt"),
                                snapshot_root=tmp_path / "sqlite-snapshot")
        assert snapshot.facts["in0"]["wal_present"] is True
        files, manifest = _prebuilt_bundle(build, mutation_root, "sqlite_peek")
        input_value = _prebuilt_input(manifest)
        input_value["wal_present"] = snapshot.facts["in0"]["wal_present"]
        outcome = _execute(execution, wire, files, mutation_root, manifest=manifest,
                           input_value=input_value, input_files=snapshot.files)
        assert outcome.exit_code == 0
        frame = wire.parse_stdout(outcome.stdout, call="prebuilt-case")
        assert frame.kind == "result"
        assert frame.result.get("wal_present") is True
        assert "only-in-wal" not in json.dumps(frame.result)
    finally:
        connection.close()

    malformed = tmp_path / "not-a-database.sqlite"
    malformed.write_bytes(b"not a sqlite database")
    mutation_root.mkdir()
    files, manifest = _prebuilt_bundle(build, mutation_root, "sqlite_peek")
    outcome = _execute(execution, wire, files, mutation_root, manifest=manifest,
                       input_value=_prebuilt_input(manifest), input_files={"in0": malformed},
                       call="malformed-db")
    assert outcome.exit_code in {0, 3}
    assert len(outcome.stdout) < 1 << 20


def test_mutation_secret_alias_case_hardlink_and_audit(mutation_root, tmp_path):
    from kernel_mutate_policy import policy_from
    from kernel_mutate_records import Journal

    repo = tmp_path / "repo"
    repo.mkdir()
    alias = repo / "API_KEY.TXT"
    alias.write_text("SECRET_ALIAS_CANARY", encoding="utf-8")
    hardlink = repo / "ordinary.txt"
    hardlink.hardlink_to(alias)
    policy = policy_from({"level": "authored", "network": "none", "domains": (),
                          "write_globs": (), "read_globs": ("**/*",), "retain": "run",
                          "audit": "digest"})
    from kernel_mutate_runtime import admit_inputs
    run_dir = tmp_path / "audit-run"
    run_dir.mkdir()
    journal = Journal(run_dir)

    for relative in ("API_KEY.TXT", "ordinary.txt"):
        with pytest.raises((PermissionError, ValueError, RuntimeError)) as caught:
            admit_inputs(repo, [relative], policy,
                         snapshot_root=tmp_path / ("snapshot-" + relative))
        assert "SECRET_ALIAS_CANARY" not in str(caught.value)
        journal.append("INPUT_REFUSED", code="READ_GRANT", reason=str(caught.value))
    assert "SECRET_ALIAS_CANARY" not in json.dumps(journal.events())


@pytest.mark.host_sandbox
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux tmpfs mounts/cgroups enforce aggregate quotas",
)
def test_mutation_tmpfs_aggregate_quota_and_dev_writes(mutation_root):
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    source = (
        "import os\n"
        "def handle(input, ctx):\n"
        "    results = {'shm_exists': os.path.exists('/dev/shm'), 'tmp_written': {}}\n"
        "    for path in input['paths']:\n"
        "        try:\n"
        "            with open(path, 'wb') as stream: count = stream.write(b'x' * (512 << 10))\n"
        "            results['tmp_written'][path] = count\n"
        "        except OSError:\n"
        "            results['tmp_written'][path] = 0\n"
        "    return results\n"
    )
    manifest = {
        "schema": "stateframe-mutation-unit/1", "name": "scratch_probe", "revision": 1,
        "parent": None, "entry": "main.py", "summary": "Scratch probe", "kind": "reader",
        "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
        "requires": {"network": False, "writes": False},
        "smoke": [{"input": {"paths": []}, "expect": {"shm_exists": False,
                                                    "tmp_written": {}}}],
        "limits": {"wall_s": 2, "cpu_s": 1, "mem_mb": 128},
    }
    token = uuid.uuid4().hex
    paths = [f"/tmp/stateframe-mutate-quota-{token}-{index}" for index in range(3)]
    outcome = _execute(execution, wire, {"main.py": source.encode()}, mutation_root,
                       manifest=manifest,
                       input_value={"paths": paths}, input_files={})
    frame = wire.parse_stdout(outcome.stdout, call="prebuilt-case")
    assert all(count == 0 for count in frame.result["tmp_written"].values())
    assert frame.result["shm_exists"] is False

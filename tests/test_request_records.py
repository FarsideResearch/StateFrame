"""Preflight request records remain byte exact through lossless sealing."""
from __future__ import annotations

import json

import pytest

from stateframe import inspect_run, request_records, runner


@pytest.fixture(autouse=True)
def _legacy_direct_engine_checkpoint(monkeypatch):
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE", "1")


def _run_with_requests(tmp_path):
    run = tmp_path / "run"
    records = run / "requests"
    records.mkdir(parents=True)
    (run / "wire").mkdir()
    originals = {
        "0001.json": b'{"body":{"messages":[{"role":"user"}]},"turn":1}\n',
        "0002.json": b'{ "body": {"tools": []}, "turn": 2 }\n',
    }
    for name, data in originals.items():
        (records / name).write_bytes(data)
    return run, originals


@pytest.fixture
def _isolated_stateframe_home(tmp_path, monkeypatch):
    home = tmp_path / "stateframe-home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))


@pytest.mark.parametrize("legacy_spec", [None, {"schema": "stateframe-run/1"}])
def test_legacy_seal_needs_no_home_pin_or_fixture_opt_in(tmp_path, monkeypatch, legacy_spec):
    from stateframe.engine import kernel_checkpoint as checkpoint

    monkeypatch.delenv("STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE")
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", False)
    run, originals = _run_with_requests(tmp_path)
    if legacy_spec is not None:
        (run / "spec.json").write_text(json.dumps(legacy_spec))

    assert request_records.seal_request_records(run) > 0
    assert (run / request_records.PACK_NAME).is_file()
    for name, original in originals.items():
        assert request_records.read_request_record(run, name) == original


def test_legacy_seal_cannot_bypass_a_runner_binding(tmp_path, monkeypatch):
    from stateframe.engine import kernel_checkpoint as checkpoint

    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", True)
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_PIN", None)
    run, originals = _run_with_requests(tmp_path)

    with pytest.raises(request_records.RequestRecordsError, match="STATEFRAME_HOME_IDENTITY_CHANGED"):
        request_records.seal_request_records(run)
    assert not (run / request_records.PACK_NAME).exists()
    assert {name: (run / "requests" / name).read_bytes() for name in originals} == originals


def test_legacy_seal_still_registers_the_installation_key(tmp_path, monkeypatch):
    from stateframe.engine import kernel_checkpoint as checkpoint

    monkeypatch.delenv("STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE")
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", False)
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    home = tmp_path / "installation"
    home.mkdir(mode=0o700)
    key = bytes(range(32))
    (home / "identity.key").write_bytes(key)
    (home / "identity.key").chmod(0o600)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    run, _originals = _run_with_requests(tmp_path)
    record = run / "requests" / "0001.json"
    original = json.dumps({"payload": key.hex()}).encode()
    record.write_bytes(original)

    with pytest.raises(request_records.RequestRecordsError, match="registered secret"):
        request_records.seal_request_records(run)
    assert not (run / request_records.PACK_NAME).exists()
    assert record.read_bytes() == original


@pytest.mark.parametrize("serialized_pin", ["null", "{}", "not JSON"])
def test_legacy_seal_refuses_corrupt_persisted_home_authority(tmp_path, monkeypatch, serialized_pin):
    from stateframe.engine import kernel_checkpoint as checkpoint

    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", False)
    run, originals = _run_with_requests(tmp_path)
    (run / "spec.json").write_text(json.dumps({
        "environment": {"STATEFRAME_RUNNER_HOME_IDENTITY": serialized_pin},
    }))
    with pytest.raises(request_records.RequestRecordsError, match="STATEFRAME_HOME_IDENTITY_CHANGED"):
        request_records.seal_request_records(run)
    assert not (run / request_records.PACK_NAME).exists()
    assert {name: (run / "requests" / name).read_bytes() for name in originals} == originals


def test_request_record_reader_returns_original_bytes_before_and_after_seal(tmp_path):
    run, originals = _run_with_requests(tmp_path)
    wire_record = run / "wire/0001.request.json"
    wire_record.write_bytes(b'{"wire":"untouched"}\n')
    wire_before = wire_record.read_bytes()

    assert request_records.list_request_records(run) == sorted(originals)
    before = {name: request_records.read_request_record(run, name) for name in originals}
    assert before == originals
    count_before = len(request_records.list_request_records(run))

    request_records.seal_request_records(run)

    assert (run / "requests.pack.xz").is_file()
    assert not list((run / "requests").glob("*.json"))
    assert request_records.list_request_records(run) == sorted(originals)
    after = {name: request_records.read_request_record(run, name) for name in originals}
    assert after == before
    assert len(request_records.list_request_records(run)) == count_before
    assert wire_record.read_bytes() == wire_before


def test_interrupted_request_seal_leaves_loose_records_readable(tmp_path, monkeypatch):
    run, originals = _run_with_requests(tmp_path)

    def interrupt(*_args, **_kwargs):
        raise OSError("injected crash before pack publication")

    monkeypatch.setattr(request_records.os, "replace", interrupt)
    with pytest.raises(OSError, match="injected crash"):
        request_records.seal_request_records(run)

    assert not (run / "requests.pack.xz").exists()
    assert request_records.list_request_records(run) == sorted(originals)
    assert {name: request_records.read_request_record(run, name) for name in originals} == originals


def test_damaged_sealed_pack_raises_request_records_error(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / request_records.PACK_NAME).write_bytes(b"\xfd7zXZ\x00invalid")

    with pytest.raises(request_records.RequestRecordsError, match="damaged"):
        request_records.list_request_records(run)


def test_inspect_and_summary_count_sealed_preflight_records(tmp_path, monkeypatch):
    run, originals = _run_with_requests(tmp_path)
    request_records.seal_request_records(run)
    (run / "summary.json").write_text(json.dumps({"engine_result": "SUBMITTED"}))

    overview = inspect_run.overview(run)
    assert "preflight records: 2" in overview

    monkeypatch.setattr(runner, "_read_send_metrics", lambda *_args: [])
    result = runner.summarize(run, 0.0)
    assert result["requests"] == len(originals)


def test_summary_of_old_spec_reports_no_provider_dialects(tmp_path, monkeypatch):
    run, _ = _run_with_requests(tmp_path)
    old_slot = {"slot": None, "selection": "not_selected", "reason": "slot discovery has not run"}
    (run / "spec.json").write_text(json.dumps({"model": "scripted", "prompt_cache_slot": old_slot}))
    monkeypatch.setattr(runner, "_read_send_metrics", lambda *_args: [])

    result = runner.summarize(run, 0.0)

    assert result["provider_capabilities"] == {}
    assert result["prompt_cache_slot"] == old_slot


def test_reading_a_sealed_run_decompresses_the_pack_once(tmp_path, monkeypatch):
    """Readers list then read every record; each read must not rescan the whole xz archive."""
    import tarfile

    run = tmp_path / "run"
    (run / "requests").mkdir(parents=True)
    originals = {f"{index:04d}.json": f'{{"turn":{index}}}\n'.encode() for index in range(1, 41)}
    for name, data in originals.items():
        (run / "requests" / name).write_bytes(data)
    opened = []
    real_open = tarfile.open

    def counting_open(*args, **kwargs):
        opened.append(kwargs.get("mode") or (args[1] if len(args) > 1 else "r"))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(request_records.tarfile, "open", counting_open)
    request_records.seal_request_records(run)
    assert len(opened) <= 4
    opened.clear()
    names = request_records.list_request_records(run)
    assert {name: request_records.read_request_record(run, name) for name in names} == originals
    assert len(opened) <= 1


def test_sealing_refuses_registered_secret_without_publishing_bundle(tmp_path, monkeypatch):
    from stateframe.engine import kernel_checkpoint

    run = tmp_path / "run"
    requests = run / "requests"
    requests.mkdir(parents=True)
    secret = b"request-bundle-test-key"
    monkeypatch.setattr(kernel_checkpoint, "_SECRETS", [secret, secret.hex().encode()])
    (requests / "0001.json").write_bytes(b'{"auth":"' + secret + b'"}\n')

    with pytest.raises(request_records.RequestRecordsError,
                       match="checkpoint refused: registered secret in state"):
        request_records.seal_request_records(run)

    assert not (run / request_records.PACK_NAME).exists()
    assert (requests / "0001.json").is_file()


def test_sealing_refuses_replacement_home_pin_from_startup(tmp_path):
    from stateframe.engine import kernel_request_auth_identity

    run, _originals = _run_with_requests(tmp_path)
    startup_home = tmp_path / "startup-home"
    replacement_home = tmp_path / "replacement-home"
    repo = tmp_path / "repo"
    startup_home.mkdir(mode=0o700)
    replacement_home.mkdir(mode=0o700)
    repo.mkdir()
    startup_identity = kernel_request_auth_identity.protected_home_identity(
        startup_home, initialize=True)
    replacement_identity = kernel_request_auth_identity.protected_home_identity(
        replacement_home, initialize=True)
    (run / "spec.json").write_text(json.dumps({
        "repo": str(repo),
        "environment": {"STATEFRAME_HOME": str(replacement_home)},
        "stateframe_home_identity": replacement_identity,
    }), encoding="utf-8")

    with pytest.raises(request_records.RequestRecordsError,
                       match="STATEFRAME_HOME_IDENTITY_CHANGED"):
        request_records.seal_request_records(
            run, startup_home_identity=startup_identity)

    assert not (run / request_records.PACK_NAME).exists()
    assert (run / "requests" / "0001.json").is_file()


def test_sealing_refuses_a_secret_discovered_after_the_payload_scan(tmp_path, monkeypatch):
    from stateframe.engine import kernel_checkpoint as checkpoint

    run, _ = _run_with_requests(tmp_path)
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    original = checkpoint.check_registered_secret_file

    def scan_then_register(path):
        result = original(path)
        checkpoint.register_secret(b"late-bundle-identity-key-fixture!!")
        return result

    monkeypatch.setattr(checkpoint, "check_registered_secret_file", scan_then_register)
    with pytest.raises(request_records.RequestRecordsError, match="STATEFRAME_HOME_IDENTITY_CHANGED"):
        request_records.seal_request_records(run)
    assert not (run / request_records.PACK_NAME).exists()
    assert (run / "requests" / "0001.json").exists()


def test_sealing_refuses_secret_reconstructed_from_nested_json_string(
        tmp_path, monkeypatch, _isolated_stateframe_home):
    from stateframe.engine import kernel_checkpoint

    run, _ = _run_with_requests(tmp_path)
    secret = b"A" * 15 + b"\n" + b"B" * 16
    record = json.dumps({
        "body": {"messages": [{"role": "user", "content": secret.decode("ascii")}]},
    }).encode("utf-8") + b"\n"
    assert secret not in record
    assert secret.hex().encode("ascii") not in record
    assert json.loads(record)["body"]["messages"][0]["content"].encode("ascii") == secret
    (run / "requests" / "0001.json").write_bytes(record)
    monkeypatch.setattr(kernel_checkpoint, "_SECRETS", [secret, secret.hex().encode("ascii")])

    with pytest.raises(request_records.RequestRecordsError,
                       match="checkpoint refused: registered secret in state"):
        request_records.seal_request_records(run)

    assert not (run / request_records.PACK_NAME).exists()
    assert (run / "requests" / "0001.json").read_bytes() == record


def test_existing_pack_refuses_secret_in_duplicate_json_member(
        tmp_path, monkeypatch, _isolated_stateframe_home):
    from stateframe.engine import kernel_checkpoint

    run, _ = _run_with_requests(tmp_path)
    secret = b"A" * 15 + b"\n" + b"B" * 16
    key_secret = b"C" * 15 + b"\n" + b"D" * 16
    record = (
        b'{"body":{' + json.dumps(key_secret.decode("ascii")).encode("ascii")
        + b':"safe","content":' + json.dumps(secret.decode("ascii")).encode("ascii")
        + b',"content":"safe"}}\n'
    )
    for candidate in (secret, key_secret):
        assert candidate not in record
        assert candidate.hex().encode("ascii") not in record
    (run / "requests" / "0001.json").write_bytes(record)
    monkeypatch.setattr(kernel_checkpoint, "_SECRETS", [])
    request_records.seal_request_records(run)

    packed_record = request_records.read_request_record(run, "0001.json")
    decoded_members = json.loads(packed_record, object_pairs_hook=lambda pairs: pairs)
    body_members = decoded_members[0][1]
    assert body_members[0][0].encode("ascii") == key_secret
    assert body_members[1][1].encode("ascii") == secret
    monkeypatch.setattr(kernel_checkpoint, "_SECRETS", [
        secret, secret.hex().encode("ascii"), key_secret, key_secret.hex().encode("ascii"),
    ])

    with pytest.raises(request_records.RequestRecordsError,
                       match="checkpoint refused: registered secret in state"):
        request_records.seal_request_records(run)

    assert (run / request_records.PACK_NAME).is_file()
    assert request_records.read_request_record(run, "0001.json") == record


@pytest.mark.parametrize("sealed", [False, True])
def test_bundle_refuses_double_json_key_without_changing_records(
        tmp_path, monkeypatch, _isolated_stateframe_home, sealed):
    from stateframe.engine import kernel_checkpoint as checkpoint

    run, _ = _run_with_requests(tmp_path)
    secret = b"A" * 15 + b"\n" + b"B" * 16
    inner = json.dumps({"key": secret.decode()})
    record = json.dumps({"body": {"messages": [{"content": inner}]}}).encode()
    assert secret not in record
    assert json.loads(json.loads(record)["body"]["messages"][0]["content"])["key"].encode() == secret
    (run / "requests" / "0001.json").write_bytes(record)
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    if sealed:
        request_records.seal_request_records(run)
    checkpoint.register_secret(secret)

    with pytest.raises(request_records.RequestRecordsError, match="registered secret in state"):
        request_records.seal_request_records(run)
    assert request_records.read_request_record(run, "0001.json") == record
    assert (run / request_records.PACK_NAME).exists() == sealed


def test_json_string_traversal_decodes_nested_json_with_a_depth_bound():
    secret = "A" * 15 + "\n" + "B" * 16
    value = secret
    for _ in range(8):
        value = json.dumps({"key": value})
    assert secret in list(request_records._iter_json_strings(value))
    # Excess nesting terminates without walking every encoded layer.
    deeper = json.dumps({"key": value})
    assert secret not in list(request_records._iter_json_strings(deeper))


def test_json_string_traversal_bounds_size_and_handles_unparseable_text(monkeypatch):
    from stateframe.engine import kernel_checkpoint as checkpoint

    oversized = json.dumps({"key": "safe"}) + " " * checkpoint._SECRET_JSON_MAX_BYTES
    original = request_records.json.loads

    def bounded_loads(text, **kwargs):
        assert len(text) <= checkpoint._SECRET_JSON_MAX_BYTES
        return original(text, **kwargs)

    monkeypatch.setattr(request_records.json, "loads", bounded_loads)
    for text in (oversized, '{"invalid":', "[" * 10000 + "0" + "]" * 10000, "ordinary text"):
        assert list(request_records._iter_json_strings(text)) == [text]


def test_safe_nested_json_and_surrogate_strings_remain_byte_exact(
        tmp_path, _isolated_stateframe_home):
    run, _ = _run_with_requests(tmp_path)
    record = json.dumps({"content": json.dumps({"text": "safe \ud800 value"})}).encode()
    (run / "requests" / "0001.json").write_bytes(record)

    request_records.seal_request_records(run)

    assert request_records.read_request_record(run, "0001.json") == record


@pytest.mark.parametrize("loose_records", [False, True])
@pytest.mark.parametrize("first_record", ["safe", "escaped-secret"])
def test_sealing_refuses_duplicate_tar_members_before_deduplication(
        tmp_path, monkeypatch, _isolated_stateframe_home, loose_records, first_record):
    import io
    import tarfile
    from stateframe.engine import kernel_checkpoint as checkpoint

    run = tmp_path / "run"
    run.mkdir()
    secret = b"A" * 15 + b"\n" + b"B" * 16
    safe = b'{"key":"safe"}'
    escaped = json.dumps({"key": secret.decode()}).encode()
    assert secret not in escaped
    monkeypatch.setattr(checkpoint, "_SECRETS", [secret, secret.hex().encode()])
    pack = run / request_records.PACK_NAME
    with tarfile.open(pack, "w:xz") as archive:
        for data in (escaped if first_record == "escaped-secret" else safe, safe):
            member = tarfile.TarInfo("0001.json")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    if loose_records:
        (run / "requests").mkdir()
        (run / "requests/0001.json").write_bytes(safe)
    original = pack.read_bytes()

    with pytest.raises(request_records.RequestRecordsError):
        request_records.seal_request_records(run)

    assert pack.read_bytes() == original
    if loose_records:
        assert (run / "requests/0001.json").read_bytes() == safe


@pytest.mark.parametrize("name,kind", [
    ("unexpected.json", "file"), ("../0001.json", "file"),
    ("nested/0001.json", "file"), ("0001.json", "directory"),
    ("0001.json", "symlink"), ("0001.json", "hardlink"),
    ("0001.json", "fifo"), ("0001.json", "character-device"),
    ("0001.json", "block-device"), ("0001.json", "contiguous"),
    ("0001.json", "sparse"), ("0001.json", "unknown"),
])
def test_sealing_refuses_unexpected_tar_members(
        tmp_path, _isolated_stateframe_home, name, kind):
    import io
    import tarfile

    run = tmp_path / "run"
    run.mkdir()
    pack = run / request_records.PACK_NAME
    with tarfile.open(pack, "w:xz") as archive:
        member = tarfile.TarInfo(name)
        if kind == "file":
            member.size = 2
            archive.addfile(member, io.BytesIO(b"{}"))
        else:
            member.type = {
                "directory": tarfile.DIRTYPE, "symlink": tarfile.SYMTYPE,
                "hardlink": tarfile.LNKTYPE, "fifo": tarfile.FIFOTYPE,
                "character-device": tarfile.CHRTYPE, "block-device": tarfile.BLKTYPE,
                "contiguous": tarfile.CONTTYPE, "sparse": tarfile.GNUTYPE_SPARSE,
                "unknown": b"Z",
            }[kind]
            member.linkname = "0002.json" if kind in {"symlink", "hardlink"} else ""
            archive.addfile(member)

    with pytest.raises(request_records.RequestRecordsError, match="unexpected"):
        request_records.seal_request_records(run)


@pytest.mark.parametrize("kind", ["pax", "global", "solaris", "gnu-name", "gnu-link"])
def test_sealing_refuses_headers_hidden_by_tarfile(tmp_path, _isolated_stateframe_home, kind):
    import io
    import tarfile

    run = tmp_path / "run"
    run.mkdir()
    pack = run / request_records.PACK_NAME
    with tarfile.open(pack, "w:xz") as archive:
        extension = tarfile.TarInfo("0001.json")
        extension.type = {
            "pax": tarfile.XHDTYPE, "global": tarfile.XGLTYPE,
            "solaris": tarfile.SOLARIS_XHDTYPE,
            "gnu-name": tarfile.GNUTYPE_LONGNAME, "gnu-link": tarfile.GNUTYPE_LONGLINK,
        }[kind]
        payload = b"0001.json\0" if kind.startswith("gnu-") else b"16 comment=safe\n"
        extension.size = len(payload)
        archive.addfile(extension, io.BytesIO(payload))
        member = tarfile.TarInfo("0001.json")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"{}"))
    original = pack.read_bytes()
    # Prove that the fixture bypasses validation of yielded members alone.
    with tarfile.open(pack, "r:xz") as archive:
        members = list(archive)
        assert len(members) == 1
        assert members[0].name == "0001.json" and members[0].isfile()

    with pytest.raises(request_records.RequestRecordsError, match="unexpected member"):
        request_records.seal_request_records(run)
    assert pack.read_bytes() == original
    with pytest.raises(request_records.RequestRecordsError, match="unexpected member"):
        request_records.read_request_record(run, "0001.json")


@pytest.mark.parametrize("loose_records", [False, True])
@pytest.mark.parametrize("global_header", [False, True])
def test_sealing_scans_key_in_hidden_pax_comment(
        tmp_path, monkeypatch, _isolated_stateframe_home, loose_records, global_header):
    import io
    import tarfile

    from stateframe.engine import kernel_checkpoint as checkpoint

    secret = b"A" * 15 + b"\n" + b"B" * 16
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    checkpoint.register_secret(secret)
    run = tmp_path / "run"
    run.mkdir()
    pack = run / request_records.PACK_NAME
    headers = {"comment": json.dumps(secret.decode())}
    safe = b"{}"
    with tarfile.open(pack, "w:xz", pax_headers=headers if global_header else None) as archive:
        member = tarfile.TarInfo("0001.json")
        member.pax_headers = {} if global_header else headers
        member.size = len(safe)
        archive.addfile(member, io.BytesIO(safe))
    if loose_records:
        (run / "requests").mkdir()
        (run / "requests/0001.json").write_bytes(safe)
    original = pack.read_bytes()
    with tarfile.open(pack, "r:xz") as archive:
        members = list(archive)
        assert len(members) == 1 and members[0].isfile()
        assert members[0].pax_headers["comment"] == headers["comment"]
        source = archive.extractfile(members[0])
        assert source is not None and source.read() == safe
    assert secret not in original and secret.hex().encode() not in original

    with pytest.raises(request_records.RequestRecordsError, match="registered secret in state"):
        request_records.seal_request_records(run)
    assert pack.read_bytes() == original
    if loose_records:
        assert (run / "requests/0001.json").read_bytes() == safe


@pytest.mark.parametrize("location", ["header", "padding", "trailer", "split-trailer"])
def test_sealing_scans_raw_decompressed_archive_bytes(
        tmp_path, monkeypatch, _isolated_stateframe_home, location):
    import io
    import lzma
    import tarfile

    from stateframe.engine import kernel_checkpoint as checkpoint

    secret = b"A" * 15 + b"\n" + b"B" * 16
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    checkpoint.register_secret(secret)
    escaped = json.dumps(secret.decode()).encode()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        member = tarfile.TarInfo("0001.json")
        member.size = 2
        # linkname is ignored for a regular file, but still exports bytes.
        if location == "header":
            member.linkname = escaped.decode()
        archive.addfile(member, io.BytesIO(b"{}"))
    raw = output.getvalue()
    if location == "padding":
        offset = 512 + 2
        raw = raw[:offset] + escaped + raw[offset + len(escaped):]
    elif location.endswith("trailer"):
        offset = len(raw) if location == "trailer" else 65536 - len(escaped) + 5
        raw += b"\0" * (offset - len(raw)) + escaped
    run = tmp_path / "run"
    run.mkdir()
    pack = run / request_records.PACK_NAME
    original = lzma.compress(raw)
    pack.write_bytes(original)
    # Ordinary readers see just the safe record at every tested location.
    assert request_records.read_request_record(run, "0001.json") == b"{}"

    with pytest.raises(request_records.RequestRecordsError, match="registered secret in state"):
        request_records.seal_request_records(run)
    assert pack.read_bytes() == original


@pytest.mark.parametrize("loose_records", [False, True])
def test_reseal_decodes_current_archive_after_same_size_same_mtime_swap(
        tmp_path, monkeypatch, _isolated_stateframe_home, loose_records):
    import io
    import os
    import tarfile
    from stateframe.engine import kernel_checkpoint as checkpoint

    secret = b"A" * 15 + b"\n" + b"B" * 16
    safe = b'{"key":"safe"}'
    unsafe = json.dumps({"key": secret.decode()}).encode()

    def archive_bytes(record):
        output = io.BytesIO()
        request_records._write_archive(output, {"0001.json": record})
        return output.getvalue()

    safe_pack, unsafe_pack = archive_bytes(safe), archive_bytes(unsafe)
    # XZ permits trailing zero padding in multiples of four bytes.
    size = max(len(safe_pack), len(unsafe_pack))
    safe_pack += b"\0" * (size - len(safe_pack))
    unsafe_pack += b"\0" * (size - len(unsafe_pack))
    assert secret not in unsafe_pack and secret.hex().encode() not in unsafe_pack
    run = tmp_path / "run"
    run.mkdir()
    pack = run / request_records.PACK_NAME
    pack.write_bytes(safe_pack)
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    checkpoint.register_secret(secret)
    assert request_records.read_request_record(run, "0001.json") == safe
    if loose_records:
        (run / "requests").mkdir()
        (run / "requests" / "0002.json").write_bytes(safe)
    before = pack.stat()
    with pack.open("r+b") as output:
        output.write(unsafe_pack)
    os.utime(pack, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = pack.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino, before.st_size, before.st_mtime_ns)
    with tarfile.open(pack, "r:xz") as archive:
        source = archive.extractfile("0001.json")
        assert source is not None
        assert source.read() == unsafe

    with pytest.raises(request_records.RequestRecordsError, match="registered secret in state"):
        request_records.seal_request_records(run)
    assert pack.read_bytes() == unsafe_pack
    if loose_records:
        assert (run / "requests" / "0002.json").read_bytes() == safe


@pytest.mark.parametrize("sealed", [False, True])
def test_bundle_refuses_object_record_with_seven_unicode_layers(
        tmp_path, monkeypatch, _isolated_stateframe_home, sealed):
    from secret_scan_helpers import seven_layer_json

    from stateframe.engine import kernel_checkpoint as checkpoint

    secret = b"A" * 15 + b"\n" + b"B" * 16
    payload = seven_layer_json(secret)
    assert len(payload) > 9_000_000
    # This object adds an eighth JSON string layer. Its content exceeds the
    # structural traversal's parse budget; isolate serialized-byte scanning below.
    record = json.dumps({"body": {"content": payload.decode()}}).encode()
    decoded = json.loads(record)["body"]["content"]
    for _ in range(7):
        decoded = json.loads(decoded)
    assert decoded.encode() == secret
    run, _ = _run_with_requests(tmp_path)
    path = run / "requests" / "0001.json"
    path.write_bytes(record)
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    if sealed:
        request_records.seal_request_records(run)
    checkpoint.register_secret(secret)
    # Prove that this boundary works independently of decoded-string traversal.
    monkeypatch.setattr(request_records, "_iter_json_strings", lambda _value: iter(()))

    with pytest.raises(request_records.RequestRecordsError, match="registered secret in state"):
        request_records.seal_request_records(run)
    assert request_records.read_request_record(run, "0001.json") == record
    assert (run / request_records.PACK_NAME).exists() == sealed

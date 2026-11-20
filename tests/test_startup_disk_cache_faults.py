"""Disk cache faults retain exact results and emit redacted diagnostics."""

import hashlib
import io
import json
import os
from pathlib import Path

import kernel_reason_log as reason_log
import kernel_release_switch_registry as registry
import kernel_request_diet as diet
import pytest
from test_request_token_count_cache import _body, _meter
from test_switch_registry_cache import _toy_root


@pytest.fixture
def diagnostics(tmp_path, monkeypatch):
    problems = tmp_path / "problems.jsonl"
    monkeypatch.setattr(reason_log, "_STATE", reason_log._fresh())
    journal = []

    class Journal:
        def emit(self, kind, **fields):
            journal.append((kind, fields))

    reason_log.configure(level="DEBUG", problems_path=problems, stream=io.StringIO(), journal=Journal())
    monkeypatch.setenv("HOME", str(tmp_path))
    registry._release_registry.cache_clear()
    yield problems
    registry._release_registry.cache_clear()
    assert journal == []


def causes(problems):
    rows = [json.loads(line) for line in problems.read_text().splitlines()] if problems.exists() else []
    assert all(row["category"] == "fallback" for row in rows)
    for row in rows:
        assert str(problems.parent) not in json.dumps(row)
        assert "same prompt" not in json.dumps(row)
    return [row["cause_key"] for row in rows]


@pytest.fixture
def fail_publication(monkeypatch):
    def install(destination, *, cleanup_fails):
        parent = destination.parent
        replace = Path.replace
        unlink = Path.unlink

        def deny_replace(path, target):
            if target == destination:
                raise PermissionError(str(target))
            return replace(path, target)

        def deny_cleanup(path, *args, **kwargs):
            if cleanup_fails and path.parent == parent:
                raise PermissionError(str(path))
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "replace", deny_replace)
        monkeypatch.setattr(Path, "unlink", deny_cleanup)

    return install


@pytest.mark.parametrize(("damage", "cause"), [
    ("digest", "switch_registry_invalid"),
    ("json", "switch_registry_read_failed"),
    ("oversize", "switch_registry_read_failed"),
    ("symlink", "switch_registry_read_failed"),
])
def test_switch_registry_reports_bad_cache_and_rebuilds(tmp_path, diagnostics, damage, cause):
    root = _toy_root(tmp_path)
    expected = registry.build_registry_document(root)
    path = registry._registry_cache_path(root)
    path.parent.mkdir(parents=True)
    if damage == "digest":
        path.write_text(json.dumps({**expected, "digest": "damaged"}))
    elif damage == "json":
        path.write_bytes(b'{"truncated":')
    elif damage == "oversize":
        path.write_text(json.dumps(expected) + " " * registry._MAX_REGISTRY_BYTES)
    else:
        target = tmp_path / "linked-registry"
        original = json.dumps({**expected, "digest": "linked"})
        target.write_text(original)
        path.symlink_to(target)

    switches, digest = registry.release_stateframe_switches(root)
    assert switches == frozenset(expected["switches"])
    assert digest == expected["digest"]
    assert causes(diagnostics) == [cause]
    assert not path.is_symlink()
    assert json.loads(path.read_text()) == expected
    if damage == "symlink":
        assert target.read_text() == original


@pytest.mark.parametrize("cleanup_fails", [False, True], ids=["replace", "cleanup"])
def test_switch_registry_reports_failed_publication(
        tmp_path, diagnostics, fail_publication, cleanup_fails):
    root = _toy_root(tmp_path)
    expected = registry.build_registry_document(root)
    path = registry._registry_cache_path(root)
    fail_publication(path, cleanup_fails=cleanup_fails)

    assert registry.release_stateframe_switches(root) == (
        frozenset(expected["switches"]), expected["digest"])
    expected_causes = ["switch_registry_write_failed"]
    if cleanup_fails:
        expected_causes.append("switch_registry_cleanup_failed")
    assert causes(diagnostics) == expected_causes
    assert not path.exists()
    assert len(list(path.parent.iterdir())) == int(cleanup_fails)


@pytest.mark.parametrize("failed_call", [1, 2], ids=["before-scan", "after-scan"])
def test_switch_registry_reports_unreadable_fingerprint(
        tmp_path, diagnostics, monkeypatch, failed_call):
    root = _toy_root(tmp_path)
    expected = registry.build_registry_document(root)
    fingerprint = registry._sources_fingerprint
    calls = 0

    def unreadable(root, files):
        nonlocal calls
        calls += 1
        if calls == failed_call:
            raise PermissionError(str(root))
        return fingerprint(root, files)

    monkeypatch.setattr(registry, "_sources_fingerprint", unreadable)
    assert registry.release_stateframe_switches(root) == (
        frozenset(expected["switches"]), expected["digest"])
    expected_causes = ["switch_registry_fingerprint_failed"]
    if failed_call == 2:
        expected_causes.append("switch_registry_publish_skipped")
    assert causes(diagnostics) == expected_causes
    assert not registry._registry_cache_path(root).exists()


def test_switch_registry_skips_publication_when_source_changes(tmp_path, diagnostics, monkeypatch):
    root = _toy_root(tmp_path)
    scan = registry._scan

    def change_source(files):
        switches = scan(files)
        source = root / "b.py"
        source.write_text(source.read_text().replace("STATEFRAME_BETA", "STATEFRAME_ZETA"))
        return switches

    with monkeypatch.context() as active:
        active.setattr(registry, "_scan", change_source)
        assert registry.release_stateframe_switches(root)[0] == {"STATEFRAME_ALPHA", "STATEFRAME_BETA"}
    assert causes(diagnostics) == ["switch_registry_publish_skipped"]
    assert not registry._registry_cache_path(root).exists()
    registry._release_registry.cache_clear()
    assert registry.release_stateframe_switches(root)[0] == {"STATEFRAME_ALPHA", "STATEFRAME_ZETA"}


@pytest.fixture
def exact_meter(tmp_path, diagnostics, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "installation"))
    monkeypatch.delenv("STATEFRAME_EXACT_TOKEN_COUNT_CACHE", raising=False)
    monkeypatch.setattr(diet.RequestTokenMeter, "_tokenizer_capability_proven", lambda self: True)
    calls = []

    def tokenize(content, *_args, **_kwargs):
        calls.append(content)
        return [0] * 17

    monkeypatch.setattr(diet, "_http_tokenize", tokenize)

    def meter():
        result = _meter("http://127.0.0.1:9/tokenize")
        result.provider_identity = json.dumps({"tokenizer": "a" * 64})
        return result

    body = _body()
    content = diet._tokenization_content(body)
    candidate = meter()
    slot, key = candidate._persistent_count_slot(
        candidate.tokenizer_identity(body), hashlib.sha256(content.encode()).hexdigest())
    return meter, body, slot, key, calls


def test_persistent_count_slot_frames_identity_and_content_digest(exact_meter):
    meter, _body, _slot, _key, _calls = exact_meter
    cache_meter = meter()

    left = cache_meter._persistent_count_slot("a:b", "c")
    right = cache_meter._persistent_count_slot("a", "b:c")

    assert left is not None and right is not None
    assert left[1] != right[1]


@pytest.mark.parametrize(("damage", "expected"), [
    ("truncated", ["exact_count_cache_invalid"]),
    ("out-of-range", ["exact_count_cache_invalid"]),
    ("linked", ["exact_count_cache_read_failed"]),
    ("directory", ["exact_count_cache_invalid", "exact_count_cache_write_failed"]),
])
def test_token_count_reports_bad_cache_and_measures_exactly(exact_meter, diagnostics, damage, expected):
    meter, body, slot, key, calls = exact_meter
    if damage == "truncated":
        slot.write_bytes(f"{key} 2".encode())
    elif damage == "out-of-range":
        slot.write_bytes(f"{key} 9999999\n".encode())
    elif damage == "linked":
        target = slot.parent / "unrelated"
        target.write_bytes(f"{key} 3\n".encode())
        slot.symlink_to(target)
    else:
        slot.mkdir()

    assert meter().measure(body) == (17, "tokenize")
    assert calls == [diet._tokenization_content(body)]
    assert causes(diagnostics) == expected
    if damage == "linked":
        assert target.read_bytes() == f"{key} 3\n".encode()
        assert not slot.is_symlink()
    if damage != "directory":
        assert slot.read_bytes() == f"{key} 17\n".encode()
        assert meter().measure(body) == (17, "tokenize")
        assert len(calls) == 1


@pytest.mark.parametrize("cleanup_fails", [False, True], ids=["replace", "cleanup"])
def test_token_count_reports_failed_publication(
        exact_meter, diagnostics, fail_publication, cleanup_fails):
    meter, body, slot, _, calls = exact_meter
    fail_publication(slot, cleanup_fails=cleanup_fails)
    assert meter().measure(body) == (17, "tokenize")
    assert calls == [diet._tokenization_content(body)]
    expected = ["exact_count_cache_write_failed"]
    if cleanup_fails:
        expected.append("exact_count_cache_cleanup_failed")
    assert causes(diagnostics) == expected
    assert not slot.exists()
    assert len(list(slot.parent.iterdir())) == int(cleanup_fails)


def test_token_count_reports_unusable_cache_directory(exact_meter, diagnostics):
    meter, body, slot, _, calls = exact_meter
    slot.parent.rmdir()
    slot.parent.write_text("occupied")
    assert meter().measure(body) == (17, "tokenize")
    assert calls == [diet._tokenization_content(body)]
    assert causes(diagnostics) == ["exact_count_cache_directory_unavailable"]
    assert slot.parent.read_text() == "occupied"


@pytest.mark.parametrize("kind", ["registry", "token-count"])
def test_cache_read_closes_descriptor_when_stream_creation_fails(
        tmp_path, diagnostics, monkeypatch, kind):
    path = tmp_path / "cache-entry"
    path.touch()
    descriptors = []

    def fail(fd, *args, **kwargs):
        descriptors.append(fd)
        raise PermissionError(str(path))

    monkeypatch.setattr(os, "fdopen", fail)
    if kind == "registry":
        assert registry._read_registry(path, "sha256:unused") is None
        cause = "switch_registry_read_failed"
    else:
        assert diet._read_exact_count(path, "a" * 64) is None
        cause = "exact_count_cache_read_failed"
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    assert causes(diagnostics) == [cause]

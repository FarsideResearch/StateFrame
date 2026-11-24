from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _run_python(code: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    process_env = dict(os.environ)
    process_env["PYTHONPATH"] = str(ROOT / "src")
    if env:
        process_env.update(env)
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=process_env,
                          text=True, capture_output=True, check=False, timeout=10)


def test_off_imports_are_baseline():
    product_import = _run_python(r'''
import importlib, json, sys
importlib.import_module("stateframe.cli")
loaded = sorted(name for name in sys.modules
                if name == "stateframe.otel" or name.startswith("stateframe.otel."))
print(json.dumps(loaded))
''')
    assert product_import.returncode == 0, product_import.stderr
    assert product_import.stdout.strip() == "[]"

NATIVE_TREE = ("src/stateframe", ":(exclude)src/stateframe/otel")


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


@pytest.fixture(scope="module")
def feature_parent():
    added = _git("log", "-1", "--diff-filter=A", "--format=%H", "--",
                 "src/stateframe/otel/__init__.py").strip().decode("ascii")
    try:
        if added:
            parent = _git("rev-parse", "--verify", f"{added}^{{commit}}^1").strip().decode("ascii")
            _git("cat-file", "-e", f"{parent}^{{tree}}")
            return added, parent
    except subprocess.CalledProcessError:
        if _git("rev-parse", "--is-shallow-repository").strip() != b"true":
            raise
    if _git("rev-parse", "--is-shallow-repository").strip() == b"true":
        pytest.skip("OFF source parity requires the feature-adding commit and its parent. Fetch that history.")
    pytest.fail("Cannot locate the OpenTelemetry feature-adding commit.")


def _assert_native_tree_equal(before, after):
    assert before == after, "native source or file inventory differs"


def test_native_product_tree_matches_feature_parent(feature_parent):
    added, parent = feature_parent
    native_changes = _git("diff", "--name-status", parent, added, "--", *NATIVE_TREE)
    assert not native_changes, native_changes.decode(errors="replace")

    # Later releases can change the native sources. Check worktree independence
    # against that release's own committed product, with the feature parent
    # providing the original before/after comparison above.
    head = _git("ls-tree", "-r", "--name-only", "-z", "HEAD", "--", NATIVE_TREE[0])
    candidate = _git("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *NATIVE_TREE)
    expected_paths = {name for name in head.split(b"\0")
                      if name and not name.startswith(b"src/stateframe/otel/")}
    actual_paths = set(candidate.split(b"\0")) - {b""}
    _assert_native_tree_equal(expected_paths, actual_paths)
    assert _git("diff", "--name-status", "HEAD", "--", *NATIVE_TREE) == b""

    planted = b"src/stateframe/new_native_switch.py"
    with pytest.raises(AssertionError, match="file inventory"):
        _assert_native_tree_equal(expected_paths, actual_paths | {planted})


_RECORD_FIXTURE = r'''
import json
from pathlib import Path
import sys
import time
import uuid

from stateframe import _child, cli
import kernel_bounded_http
from kernel_measurement import AuditJournal

run = Path(sys.argv[1])
run.mkdir()
time.time_ns = lambda: 1_730_000_000_000_000_000
time.monotonic = lambda: 1.0
time.monotonic_ns = lambda: 1_000_000_000
uuid.uuid4 = lambda: uuid.UUID("11111111-1111-4111-8111-111111111111")
journal = AuditJournal(run / "worker", "opaque", {})
kernel_bounded_http.request_bytes = lambda *a, **k: b'{"choices":[{}]}'
_child.emit = lambda *a, **k: None
endpoint = "http://127.0.0.1/chat"
_child.install_wire_recorder(run, endpoint)
body = b'{"model":"fixture","messages":[{"role":"user","content":"opaque"}]}'
kernel_bounded_http.request_bytes(body, endpoint, "")
journal.emit("check_receipt", check_kind="final", status="PASS", execution_id="opaque-check")
imports = sorted(name for name in sys.modules if name.startswith(("stateframe", "kernel_")))
print(json.dumps(imports))
'''


def _run_record_fixture(source, run, *, ambient):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("OTEL_", "STATEFRAME_", "REACT_", "PYTHON", "KERNEL_", "DS_"))}
    env.update(ambient, PYTHONPATH=os.pathsep.join((str(source), str(source / "stateframe/engine"))))
    result = subprocess.run([sys.executable, "-c", _RECORD_FIXTURE, str(run)], cwd=run.parent,
                            env=env, capture_output=True, timeout=30, check=True)
    records = {path.relative_to(run).as_posix(): path.read_bytes()
               for path in run.rglob("*") if path.is_file()}
    return json.loads(result.stdout), records


@pytest.mark.parametrize("otel_disabled", ["true", "false"])
def test_native_raw_requests_journals_receipts(tmp_path, feature_parent, otel_disabled):
    _, parent = feature_parent
    # When later releases change the native package, use their same-release
    # product from Git, with the optional module removed from the archive view.
    base = "HEAD" if _git("diff", "--name-only", parent, "HEAD", "--", *NATIVE_TREE) else parent
    archive_path = tmp_path / "native.tar"
    subprocess.run(["git", "archive", "-o", str(archive_path), base, "src/stateframe"],
                   cwd=ROOT, check=True)
    baseline = tmp_path / "base"
    with tarfile.open(archive_path) as archive:
        archive.extractall(baseline, members=(member for member in archive
                                             if not member.name.startswith("src/stateframe/otel")),
                           filter="data")
    imports, expected = _run_record_fixture(baseline / "src", tmp_path / "before", ambient={})
    candidate_imports, actual = _run_record_fixture(
        ROOT / "src", tmp_path / "after",
        ambient={"OTEL_EXPORTER_OTLP_HEADERS": "x-canary=synthetic", "OTEL_SDK_DISABLED": otel_disabled,
                 "OTEL_TRACES_EXPORTER": "otlp"},
    )
    assert candidate_imports == imports
    assert not any(name.startswith("stateframe.otel") for name in imports)
    request = "wire/0001.request.json"
    journal = next(name for name in expected if name.endswith("/events.jsonl"))
    assert b'"kind":"check_receipt"' in expected[journal]
    assert request in expected and "sends.jsonl" in expected
    assert actual == expected, "native requests, journals and receipts differ"
    for name in (request, "sends.jsonl", journal):
        planted = {**actual, name: actual[name] + b" "}
        with pytest.raises(AssertionError, match="file inventory"):
            _assert_native_tree_equal(expected, planted)


def test_on_uuid_stream_and_checkpoint_fingerprint_unchanged(feature_parent):
    added, parent = feature_parent
    paths = (
        "src/stateframe/_child.py",
        "src/stateframe/runner.py",
        "src/stateframe/engine/kernel_checkpoint.py",
        "src/stateframe/engine/kernel_stateframe_runtime.py",
    )
    # Release-specific file bytes come from Git, never a pinned digest.
    for name in paths:
        original = _git("show", f"{parent}:{name}")
        feature = _git("show", f"{added}:{name}")
        assert hashlib.sha256(feature).digest() == hashlib.sha256(original).digest(), name

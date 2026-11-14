"""Provider normalization preserves llama.cpp bytes and seals every emitted request.

The old 30ce267 comparison also froze unrelated prompts and tool schemas. Capture
the normalization input in the current run instead, without masking any fields.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from _journal import measurement_events
from _timeouts import scaled
from engine_run import run_monitored, sandbox_test_python
from kernel_provider_dialect import LLAMA_CPP, PROVIDER_OVERRIDE_ENV

from stateframe.scripted_endpoint import Script, serve
from tests.test_final_cache_layout_host import _parity_launchers

ROOT = Path(__file__).resolve().parents[1]


def _observe_normalization(wrapper: Path) -> None:
    """Record immutable input bytes while leaving the real normalizer in place."""
    wrapper.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "child, run = sys.argv[1:]\n"
        "sys.path.insert(0, str(Path(child).parent.parent))\n"
        "import stateframe._child as host\n"
        "install_identity = host.install_run_identity\n"
        "def observe_after_identity(seed):\n"
        "    install_identity(seed)\n"
        "    sys.path.insert(0, str(Path(child).parent / 'engine'))\n"
        "    import kernel_final_wire_allocation as allocation\n"
        "    normalize = allocation._finalize_provider_request_body\n"
        "    def observed(capture, body, capabilities):\n"
        "        assert capabilities.dialect == 'llama.cpp', capabilities.dialect\n"
        "        before = json.dumps(body)\n"
        "        result = normalize(capture, body, capabilities)\n"
        "        with (Path(run) / 'normalization-inputs.jsonl').open('a', encoding='utf-8') as output:\n"
        "            output.write(json.dumps(before) + '\\n')\n"
        "        return result\n"
        "    allocation._finalize_provider_request_body = observed\n"
        "host.install_run_identity = observe_after_identity\n"
        "raise SystemExit(host.main([child, run]))\n", encoding="utf-8")


def _assert_wire_parity(inputs: list[bytes], wires: list[bytes], ready: set[str]) -> None:
    assert inputs, "no llama.cpp normalization inputs observed"
    assert wires == inputs, "normalization changed request bytes, count, or order"
    assert all("sha256:" + hashlib.sha256(wire).hexdigest() in ready for wire in wires), \
        "emitted request has no matching READY seal"


def _ready_seal_hashes(events: list[dict[str, object]]) -> set[str]:
    # Preliminary menu-admission READY rows do not carry a wire hash.
    return {str(row["body_sha256"]) for row in events
            if row.get("kind") == "request_budget_final_wire" and row.get("status") == "READY"
            and isinstance(row.get("body_sha256"), str)}


def _scripted_environment(tmp_path: Path) -> dict[str, str]:
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("STATEFRAME_", "REACT_"))}
    environment.update({
        "PYTHONPATH": str(ROOT / "src"), "STATEFRAME_FINAL_CACHE_LAYOUT": "off",
        "STATEFRAME_PAGE_LOG": "off", "STATEFRAME_HOME": str(tmp_path / "state-home"),
        "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
        PROVIDER_OVERRIDE_ENV: LLAMA_CPP,
        # Checks use an explicit Python. Do not freeze other workers' temporary
        # interpreter links from the shared venv into this child's PATH authority.
        "PATH": os.defpath,
    })
    return environment


@pytest.mark.host_sandbox
@pytest.mark.parametrize("example,task", [
    ("discount_bug", "Fix the pricing code so the tests pass."),
    ("tiny_site", "Build the small static shop site described in the README."),
    ("brief_to_report", "Write the report the brief asks for."),
])
def test_bundled_llama_wire_matches_normalization_input_and_its_seal(
        tmp_path: Path, example: str, task: str) -> None:
    launcher, wrapper = _parity_launchers(tmp_path)
    _observe_normalization(wrapper)
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples" / example, repo)
    script = Script(json.loads((ROOT / "examples" / f"{example}.script.json").read_text()))
    server = serve(script)
    run = tmp_path / "run"
    environment = _scripted_environment(tmp_path)
    try:
        with sandbox_test_python(sys.executable) as test_python:
            process = run_monitored(
                [sys.executable, str(launcher), str(wrapper), "run", "--repo", str(repo),
                 "--task", task, "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
                 "--model", "scripted", "--run-dir", str(run), "--test-python", test_python,
                 "--max-turns", "16", "--quiet"],
                env=environment, expected_seconds=scaled(600), run_dir=run,
                step=f"llama normalization parity: {example}")
        assert process.returncode == 0, process.stdout + process.stderr
        script.assert_consumed()
        inputs = [json.loads(line).encode() for line in
                  (run / "normalization-inputs.jsonl").read_text(encoding="utf-8").splitlines()]
        wires = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
        ready = _ready_seal_hashes(measurement_events(run))
        _assert_wire_parity(inputs, wires, ready)
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def llama_wire() -> bytes:
    return json.dumps({"model": "scripted", "messages": [{"role": "user", "content": "source"}],
                       "tools": [{"type": "function", "function": {
                           "name": "read", "parameters": {"type": "object", "properties": {
                               "path": {"type": "string"}}}}}]}).encode()


def test_scripted_path_excludes_ambient_temporary_interpreters(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "shared-venv/bin"))
    environment = _scripted_environment(tmp_path)
    assert environment["PATH"] == os.defpath
    assert environment["PYTHONPATH"] == str(ROOT / "src")
    assert environment[PROVIDER_OVERRIDE_ENV] == LLAMA_CPP


def test_wire_guard_accepts_unchanged_sealed_request(llama_wire: bytes) -> None:
    ready = {"sha256:" + hashlib.sha256(llama_wire).hexdigest()}
    _assert_wire_parity([llama_wire], [llama_wire], ready)


def test_ready_seals_require_a_wire_hash_and_ready_status(llama_wire: bytes) -> None:
    digest = "sha256:" + hashlib.sha256(llama_wire).hexdigest()
    ready: dict[str, object] = {"kind": "request_budget_final_wire", "status": "READY"}
    rows = [ready, {**ready, "body_sha256": None},
            {**ready, "body_sha256": digest},
            {**ready, "body_sha256": "failed", "status": "REFUSED"},
            {**ready, "body_sha256": "unrelated", "kind": "other"}]
    assert _ready_seal_hashes(rows) == {digest}


@pytest.mark.parametrize("change", [
    "content", "role", "tool_schema", "missing_wire", "missing_capture", "missing_seal",
])
def test_wire_guard_rejects_planted_changes(llama_wire: bytes, change: str) -> None:
    replacements = {
        "content": (b'"source"', b'"changed"'),
        "role": (b'"user"', b'"system"'),
        "tool_schema": (b'"string"', b'"integer"'),
    }
    changed = llama_wire
    if change in replacements:
        changed = changed.replace(*replacements[change])
        assert changed != llama_wire
    # A matching seal on a changed body must not hide byte drift.
    ready = {"sha256:" + hashlib.sha256(changed).hexdigest()} if change != "missing_seal" else set()
    inputs = [] if change == "missing_capture" else [llama_wire]
    wires = [] if change == "missing_wire" else [changed]
    with pytest.raises(AssertionError):
        _assert_wire_parity(inputs, wires, ready)

"""Live final-layout seal control for a tokenizer outage within one request."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from _stateframe_home import isolated_stateframe_home
from tests import test_allocation_replay_corpus as corpus
from tests.tape_server import TapeServer


CASE_ID = "factory-a-econ-flat-r1"
ROOT = Path(__file__).resolve().parents[1]


class FlipTapeServer(TapeServer):
    """Answer allocation tokenizations, then refuse calls until the next send."""

    def __init__(self, script):
        super().__init__(script)
        self.armed = False
        self.successful_tokenizations = 0
        self.failed_tokenizations = 0
        self.flip_request_id = None

    def _handler(self):
        base = super()._handler()
        tape = self

        class Handler(base):
            def do_POST(self):
                path = self.path.rstrip("/")
                if path.endswith("/__flip_tokenize"):
                    length = int(self.headers.get("Content-Length") or 0)
                    payload = json.loads(self.rfile.read(length))
                    with tape._lock:
                        if tape.flip_request_id is not None:
                            self._send_json(409, {"error": "tokenizer already flipped"})
                            return
                        tape.flip_request_id = payload["request_id"]
                        tape.armed = True
                    self._send_json(200, {"armed": True})
                    return
                if path.endswith("/tokenize"):
                    with tape._lock:
                        failed = tape.armed
                        if failed:
                            tape.failed_tokenizations += 1
                        else:
                            tape.successful_tokenizations += 1
                    if failed:
                        self._send_json(503, {"error": {"message": "tokenizer unavailable"}})
                        return
                super().do_POST()
                if path.endswith("/chat/completions"):
                    with tape._lock:
                        tape.armed = False

        return Handler


def _engine_python(tmp_path: Path, endpoint_url: str) -> Path:
    """Arm the endpoint after the measured layout, before final proof."""
    bootstrap = "\n".join((
        "import json, math, sys, urllib.request",
        "from pathlib import Path",
        "child = Path(sys.argv[1]).resolve()",
        "run = Path(sys.argv[2]).resolve()",
        "sys.path.insert(0, str(child.parent.parent))",
        "import stateframe._child as host",
        "host.install_environment(json.loads((run / 'spec.json').read_text()))",
        "sys.path.insert(0, str(child.parent / 'engine'))",
        "import kernel_cache_layout_candidate as layout",
        "original = layout.render_final_cache_candidate",
        "def after_layout(capture, runtime, body, *, meter, inputs):",
        "    result = original(capture, runtime, body, meter=meter, inputs=inputs)",
        "    marker = run / 'seal-live-flip.json'",
        "    measured = getattr(capture, '_frame_candidate_measurement', None)",
        "    size = len(json.dumps(result).encode())",
        "    ratio = meter._fallback_ratio()[0]",
        "    fallback_count = math.ceil(size / ratio) + meter.template_overhead_tokens",
        "    if (not marker.exists() and result is not body",
        "            and fallback_count > inputs['token_ceiling']",
        "            and getattr(measured, 'method', None) == 'tokenize'):",
        f"        endpoint = {endpoint_url!r} + '/__flip_tokenize'",
        "        data = json.dumps({'request_id': capture.request_id}).encode()",
        "        with urllib.request.urlopen(urllib.request.Request(endpoint, data=data), timeout=5) as response:",
        "            if response.status != 200: raise RuntimeError('tokenizer fault was not armed')",
        "        # The layout measurement is cached. A fresh probe makes the outage",
        "        # observable before proof, as it is after ordinary cache eviction.",
        "        meter.__dict__.get('_count_cache', {}).clear()",
        "        probe_tokens, probe_method = meter.measure(result)",
        "        marker.write_text(json.dumps({'request_id': capture.request_id,",
        "            'layout_bytes': size, 'allocation_method': measured.method,",
        "            'fallback_count': fallback_count, 'token_ceiling': inputs['token_ceiling'],",
        "            'probe_method': probe_method, 'probe_tokens': probe_tokens}) + '\\n')",
        "    return result",
        "layout.render_final_cache_candidate = after_layout",
        "raise SystemExit(host.main([str(child), str(run)]))",
    ))
    wrapper = tmp_path / "python-with-seal-flip"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os,sys\n"
        f"real_python={sys.executable!r}\n"
        "args=sys.argv[1:]\n"
        "if len(args) >= 3 and args[0] == '-B' and os.path.basename(args[1]) == '_child.py':\n"
        f"    os.execv(real_python, [real_python, '-B', '-c', {bootstrap!r}, *args[1:]])\n"
        "os.execv(real_python, [real_python, *args])\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


def _events(run_dir: Path):
    from stateframe.engine.kernel_journal_store import journal_paths, open_journal

    events = []
    for path in journal_paths(run_dir / "worker" / "measurement-runs"):
        with open_journal(path) as source:
            events.extend(json.loads(line) for line in source if line.strip())
    return events


@pytest.mark.host_sandbox
def test_layout_tokenizer_flip_seals_live_request(tmp_path):
    """A projected exact fit survives a later tokenizer outage at final proof."""
    source_root = ROOT / "src"
    assert (source_root / "stateframe" / "_child.py").is_file()
    case_dir = corpus.CORPUS / CASE_ID
    metadata = corpus.CASE_METADATA[CASE_ID]
    repo = tmp_path / "repo"
    shutil.copytree(corpus._case_repo(CASE_ID), repo)
    run_dir = tmp_path / "run"
    profile = corpus._profile_file(metadata, source_root, tmp_path)
    task = (case_dir / "task.txt").read_text(encoding="utf-8")
    settings = metadata.get("recorded_settings") or {}
    slot = settings.get("prompt_cache_slot", {}).get("slot", 6)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STATEFRAME_")}
    environment.update({
        "STATEFRAME_HOME": isolated_stateframe_home(),
        # The replay helper strips STATEFRAME_*; this explicit test-only allowlist
        # passes the release's opt-in layout switch to the real engine child.
        "STATEFRAME_FINAL_CACHE_LAYOUT": "on",
        "PYTHONPATH": os.pathsep.join((str(source_root), str(source_root / "stateframe" / "engine"))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    })
    script = case_dir / "script.json.gz"
    with FlipTapeServer(script) as endpoint:
        command = [
            sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", task, "--endpoint", endpoint.url,
            "--model", str(settings.get("model") or "allocation-replay-model"),
            "--run-dir", str(run_dir), "--test-python", sys.executable,
            "--shell-read", str(source_root), "--slot", str(slot),
            "--max-turns", "120", "--timeout", "1800", "--quiet",
            "--python", str(_engine_python(tmp_path, endpoint.url)),
        ]
        if profile is not None:
            command.extend(("--profile", str(profile)))
        started = time.monotonic()
        process = subprocess.run(command, capture_output=True, text=True,
                                 env=environment, timeout=1860, check=False)
        endpoint_requests = endpoint.request_count
        successful = endpoint.successful_tokenizations
        failed = endpoint.failed_tokenizations
        flip_request_id = endpoint.flip_request_id
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    engine_result = summary.get("engine_result")
    marker_path = run_dir / "seal-live-flip.json"
    marker = json.loads(marker_path.read_text()) if marker_path.is_file() else None
    events = _events(run_dir)
    budget_rows = [row for row in events if row.get("kind") == "request_budget_final_wire"]
    blocked_proofs = [row for row in budget_rows
                      if row.get("phase") == "frozen_transport_revalidation"
                      and row.get("status") == "HOST_BLOCKED"]
    blocked_summary = [
        (row.get("step"), row.get("request_id"), row.get("request_bytes"),
         row.get("measured_prompt_tokens"), row.get("final_validation_failures"))
        for row in blocked_proofs
    ]
    detail = (f"result={engine_result!r}, exit={process.returncode}, "
              f"endpoint_requests={endpoint_requests}, tokenize_ok={successful}, "
              f"tokenize_failed={failed}, flip={marker!r}, "
              f"flip_request_id={flip_request_id!r}, blocked_proofs={blocked_summary!r}, "
              f"elapsed={time.monotonic() - started:.1f}s\n"
              f"{process.stdout[-1500:]}\n{process.stderr[-2500:]}")
    assert marker is not None, "no exact layout request reached the fault boundary; " + detail
    assert marker["probe_method"] == "calibrated_bytes", detail
    assert marker["probe_tokens"] > marker["token_ceiling"], detail
    assert successful > 0 and failed > 0 and flip_request_id == marker["request_id"], detail
    # src/stateframe/outcomes.py exit_code() returns 1 when this fixture's own
    # task checks fail. That CLI code is not an engine failure; use engine_result.
    assert engine_result == "SUBMITTED", detail
    assert endpoint_requests > 0, detail
    assert any(row.get("request_id") == marker["request_id"] and row.get("status") == "READY"
               for row in budget_rows), detail
    sends_path = run_dir / "sends.jsonl"
    sends = [json.loads(line) for line in sends_path.read_text().splitlines() if line.strip()]
    sent_digests = {"sha256:" + row["request_sha256"] for row in sends if "error" not in row}
    sealed_digests = {row.get("body_sha256") for row in budget_rows
                      if row.get("status") == "READY"}
    assert sent_digests and sent_digests <= sealed_digests, detail
    assert not blocked_proofs, detail

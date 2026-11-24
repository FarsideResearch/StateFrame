"""Live final-layout seal control for a tokenizer outage within one request."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time

import pytest

from _stateframe_home import isolated_stateframe_home
from _journal import measurement_events
from engine_run import run_monitored
from tests import test_allocation_replay_corpus as corpus
from tests.tape_server import TapeServer


CASE_ID = "factory-a-econ-flat-r1"
ROOT = Path(__file__).resolve().parents[1]


class FlipTapeServer(TapeServer):
    """Answer allocation tokenizations, then refuse calls for one or all later sends."""

    def __init__(self, script, *, sticky=False):
        super().__init__(script)
        self.sticky = sticky
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
                if path.endswith("/chat/completions") and not tape.sticky:
                    with tape._lock:
                        tape.armed = False

        return Handler


def _engine_python(tmp_path: Path, endpoint_url: str, run_dir: Path) -> Path:
    """Arm the endpoint after the measured layout in each engine child."""
    wrapper = tmp_path / "python-with-seal-flip"
    bootstrap = "\n".join((
        "import builtins, json, sys, urllib.request",
        "from pathlib import Path",
        "child = Path(sys.argv[1]).resolve()",
        "run = Path(sys.argv[2]).resolve()",
        "boot = {'child': str(child), 'run': str(run), 'argv': sys.argv, 'modules': []}",
        "def write_boot():",
        "    (run / 'seal-live-bootstrap.json').write_text(json.dumps(boot) + '\\n')",
        "write_boot()",
        "sys.path.insert(0, str(child.parent.parent))",
        "import stateframe._child as host",
        "host.install_environment(json.loads((run / 'spec.json').read_text()))",
        "sys.path.insert(0, str(child.parent / 'engine'))",
        "marker = run / 'seal-live-flip.json'",
        "observations = run / 'seal-live-observations.jsonl'",
        "patched_modules = set()",
        "def install_layout_hook(layout):",
        "    if id(layout) in patched_modules: return",
        "    patched_modules.add(id(layout))",
        "    boot['modules'].append({'name': layout.__name__, 'file': getattr(layout, '__file__', None)})",
        "    write_boot()",
        "    original = layout.render_final_cache_candidate",
        "    def after_layout(capture, runtime, body, *, meter, inputs):",
        "        measured = getattr(capture, '_frame_candidate_measurement', None)",
        "        result = original(capture, runtime, body, meter=meter, inputs=inputs)",
        "        record = getattr(capture, '_final_cache_layout_record', {})",
        "        observation = {'module': layout.__name__, 'request_id': capture.request_id,",
        "            'allocation_method': getattr(measured, 'method', None),",
        "            'final_layout': record.get('final_layout') if isinstance(record, dict) else None,",
        "            'projected': result is not body}",
        "        with observations.open('a', encoding='utf-8') as stream:",
        "            stream.write(json.dumps(observation) + '\\n')",
        "        if (marker.exists() or getattr(measured, 'method', None) != 'tokenize'",
        "                or not isinstance(record, dict) or record.get('final_layout') != 'applied'):",
        "            return result",
        f"        endpoint = {endpoint_url!r} + '/__flip_tokenize'",
        "        data = json.dumps({'request_id': capture.request_id}).encode()",
        "        with urllib.request.urlopen(urllib.request.Request(endpoint, data=data), timeout=5) as response:",
        "            if response.status != 200: raise RuntimeError('tokenizer fault was not armed')",
        "        meter.__dict__.get('_count_cache', {}).clear()",
        "        probe_tokens, probe_method = meter.measure(result)",
        "        marker.write_text(json.dumps({'request_id': capture.request_id,",
        "            'allocation_method': getattr(measured, 'method', None),",
        "            'final_layout': record.get('final_layout'),",
        "            'probe_method': probe_method, 'probe_tokens': probe_tokens}) + '\\n')",
        "        return result",
        "    layout.render_final_cache_candidate = after_layout",
        "original_import = builtins.__import__",
        "def import_with_layout_hook(name, globals=None, locals=None, fromlist=(), level=0):",
        "    imported = original_import(name, globals, locals, fromlist, level)",
        "    if name.rsplit('.', 1)[-1] == 'kernel_cache_layout_candidate':",
        "        module = sys.modules.get(name)",
        "        if module is not None: install_layout_hook(module)",
        "    return imported",
        "builtins.__import__ = import_with_layout_hook",
        "for module_name, module in tuple(sys.modules.items()):",
        "    if module_name.rsplit('.', 1)[-1] == 'kernel_cache_layout_candidate':",
        "        install_layout_hook(module)",
        "raise SystemExit(host.main([str(child), str(run)]))",
    ))
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os,sys\n"
        f"real_python={sys.executable!r}\n"
        "args=sys.argv[1:]\n"
        "if len(args) >= 3 and args[0] == '-B' and os.path.basename(args[1]) == '_child.py':\n"
        "    import json\n"
        "    with open(os.path.join(args[2],'seal-live-launch.jsonl'),'a',encoding='utf-8') as calls:\n"
        "        calls.write(json.dumps({'args_prefix':args[:2],'arg_count':len(args),'run':args[2]})+'\\n')\n"
        f"    os.execv(real_python, [real_python, '-B', '-c', {bootstrap!r}, *args[1:]])\n"
        "if len(args) >= 3 and args[:2] == ['-B', '-c'] and 'runpy.run_path' in args[2]:\n"
        "    import ast,json,re\n"
        "    match=re.search(r'sys\\.argv=(\\[.*?\\]);',args[2])\n"
        "    if match:\n"
        "        child,run=ast.literal_eval(match.group(1))\n"
        "        if os.path.basename(child) == '_child.py':\n"
        "            with open(os.path.join(run,'seal-live-launch.jsonl'),'a',encoding='utf-8') as calls:\n"
        "                calls.write(json.dumps({'args_prefix':args[:2],'arg_count':len(args),'trailing':args[3:],'child':child,'run':run})+'\\n')\n"
        "            os.chdir(run)\n"
        f"            os.execv(real_python, [real_python, '-B', '-c', {bootstrap!r}, child, run])\n"
        "os.execv(real_python, [real_python, *args])\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


@pytest.mark.host_sandbox
@pytest.mark.parametrize("sticky", [False, True], ids=["one-request", "sticky-outage"])
def test_layout_tokenizer_flip_seals_live_request(tmp_path, sticky):
    """A projected exact fit survives a later tokenizer outage at final proof."""
    source_root = ROOT / "src"
    assert (source_root / "stateframe" / "_child.py").is_file()
    case_dir = corpus.CORPUS / CASE_ID
    metadata = corpus.CASE_METADATA[CASE_ID]
    repo = tmp_path / "repo"
    shutil.copytree(corpus._case_repo(CASE_ID), repo)
    run_dir = tmp_path / "run"
    profile = corpus._profile_file(metadata, source_root, tmp_path)
    if profile is not None:
        profile_data = json.loads(profile.read_text(encoding="utf-8"))
        switches = profile_data.get("switches")
        assert isinstance(switches, dict)
        # This test targets the final cache renderer, which the working-set
        # allocator intentionally bypasses when enabled.
        switches["STATEFRAME_WORKING_SET_ALLOCATOR"] = "off"
        profile.write_text(json.dumps(profile_data, indent=1) + "\n", encoding="utf-8")
        assert json.loads(profile.read_text(encoding="utf-8"))["switches"][
            "STATEFRAME_WORKING_SET_ALLOCATOR"] == "off"
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
    with FlipTapeServer(script, sticky=sticky) as endpoint:
        command = [
            sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", task, "--endpoint", endpoint.url,
            "--model", str(settings.get("model") or "allocation-replay-model"),
            "--run-dir", str(run_dir), "--test-python", sys.executable,
            "--shell-read", str(source_root), "--slot", str(slot),
            "--max-turns", "120", "--timeout", "1800", "--quiet",
            "--python", str(_engine_python(tmp_path, endpoint.url, run_dir)),
        ]
        if profile is not None:
            command.extend(("--profile", str(profile)))
        started = time.monotonic()
        # A 620-second measured window gives the prior 1860-second hard ceiling
        # at normal load; the shared runner adjusts it for host load and progress.
        process = run_monitored(
            command, run_dir=run_dir, expected_seconds=620,
            env=environment, capture_output=True, text=True, check=False)
        endpoint_requests = endpoint.request_count
        successful = endpoint.successful_tokenizations
        failed = endpoint.failed_tokenizations
        flip_request_id = endpoint.flip_request_id
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    engine_result = summary.get("engine_result")
    marker_path = run_dir / "seal-live-flip.json"
    marker = json.loads(marker_path.read_text()) if marker_path.is_file() else None
    observations_path = run_dir / "seal-live-observations.jsonl"
    observations = (observations_path.read_text().splitlines()[-8:]
                    if observations_path.is_file() else [])
    bootstrap_path = run_dir / "seal-live-bootstrap.json"
    bootstrap = json.loads(bootstrap_path.read_text()) if bootstrap_path.is_file() else None
    launch_path = run_dir / "seal-live-launch.jsonl"
    launch_rows = (launch_path.read_text().splitlines()[-8:]
                   if launch_path.is_file() else [])
    events = measurement_events(run_dir)
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
              f"layout_observations={observations!r}, "
              f"bootstrap={bootstrap!r}, launcher_calls={launch_rows!r}, "
              f"elapsed={time.monotonic() - started:.1f}s\n"
              f"{process.stdout[-1500:]}\n{process.stderr[-2500:]}")
    assert marker is not None, "no exact layout request reached the fault boundary; " + detail
    assert marker["probe_method"] == "calibrated_bytes", detail
    assert marker["final_layout"] == "applied", detail
    assert marker["allocation_method"] == "tokenize" and marker["probe_tokens"] > 0, detail
    assert successful > 0 and failed > 0 and flip_request_id == marker["request_id"], detail
    # The recorded task may terminate after its transcript diverges under this
    # test-only allocator override. The contract here is the sealed provider
    # request, proved below by matching final-wire and send digests.
    assert endpoint_requests > 0, detail
    target_proofs = [
        row for row in budget_rows
        if row.get("request_id") == marker["request_id"]
        and row.get("phase") == "frozen_transport_revalidation"
        and row.get("status") == "READY"
        and row.get("measure_method") == marker["allocation_method"]
        and row.get("tokenize_availability") == "unavailable"
        and row.get("tokenize_failure_reason") == "http_error"
    ]
    assert target_proofs, detail
    sends_path = run_dir / "sends.jsonl"
    sends = [json.loads(line) for line in sends_path.read_text().splitlines() if line.strip()]
    sent_digests = {"sha256:" + row["request_sha256"] for row in sends if "error" not in row}
    sealed_digests = {row.get("body_sha256") for row in budget_rows
                      if row.get("status") == "READY"}
    assert sent_digests and sent_digests <= sealed_digests, detail
    assert {row.get("body_sha256") for row in target_proofs} & sent_digests, detail
    assert not blocked_proofs, detail
    if sticky:
        ready_by_digest = {row.get("body_sha256"): row.get("request_id")
                           for row in budget_rows if row.get("status") == "READY"}
        sent_ids = [ready_by_digest.get("sha256:" + row["request_sha256"])
                    for row in sends if "error" not in row]
        assert flip_request_id in sent_ids and sent_ids.index(flip_request_id) < len(sent_ids) - 1, detail
        deferred = [row for row in budget_rows
                    if row.get("request_id") in sent_ids
                    and row.get("phase") == "pre_fit_tool_surface"
                    and row.get("status") == "PRE_FIT_DEFERRED"]
        later_ids = sent_ids[sent_ids.index(flip_request_id) + 1:]
        assert any(row.get("request_id") in later_ids
                   and row.get("measure_method") == "calibrated_bytes"
                   and row.get("phase") == "frozen_transport_revalidation"
                   for row in budget_rows), detail
        assert all(sum(row.get("request_id") == request_id for row in deferred) >= 2
                   for request_id in later_ids), detail
        assert all(row.get("status") == "READY" for row in budget_rows
                   if row.get("phase") == "frozen_transport_revalidation"), detail

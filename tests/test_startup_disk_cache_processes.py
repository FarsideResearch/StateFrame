"""Independent engine processes may publish the same cold cache slot safely."""

import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import pytest
from _timeouts import scaled
from test_request_token_count_cache import _token_server
from test_switch_registry_cache import _toy_root

SOURCE = Path(__file__).resolve().parents[1] / "src"
WORKER = """
import json
import os
import sys
import time
from pathlib import Path

kind, directory, endpoint, phase = sys.argv[1:]
root = Path(directory)
if phase == "cold":
    replace = Path.replace

    def publish(path, destination):
        marker = root / f"ready-{os.getpid()}"
        marker.write_text(str(destination))
        deadline = time.monotonic() + 30
        while not (root / "publish").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("parent never released publication")
            time.sleep(0.01)
        return replace(path, destination)

    Path.replace = publish

if kind == "registry":
    import kernel_release_switch_registry as registry

    scans = 0
    scan = registry._scan

    def count_scan(files):
        global scans
        scans += 1
        return scan(files)

    registry._scan = count_scan
    switches, digest = registry.release_stateframe_switches(root / "engine")
    result = [sorted(switches), digest, scans]
else:
    from kernel_provider_dialect import capabilities_for
    from kernel_request_diet import RequestTokenMeter

    assert capabilities_for(endpoint, "", require_slots=False).continuation
    meter = RequestTokenMeter(endpoint=endpoint, template_overhead_tokens=0)
    meter.provider_identity = json.dumps({"tokenizer": "a" * 64})
    result = meter.measure({"model": "scripted", "messages": [{"role": "user", "content": "shared"}]})

print(json.dumps(result))
"""


@pytest.mark.parametrize("kind", ["registry", "token-count"])
def test_two_processes_publish_and_reuse_the_same_cache_slot(tmp_path, kind):
    env = {**os.environ, "HOME": str(tmp_path), "STATEFRAME_HOME": str(tmp_path / "installation"),
           "PYTHONPATH": os.pathsep.join((str(SOURCE), str(SOURCE / "stateframe/engine")))}
    env.pop("STATEFRAME_EXACT_TOKEN_COUNT_CACHE", None)
    if kind == "registry":
        _toy_root(tmp_path)
    context = _token_server() if kind == "token-count" else nullcontext()

    with context as server:
        endpoint = server.url + "/tokenize" if server is not None else ""
        command = [sys.executable, "-B", "-c", WORKER, kind, str(tmp_path), endpoint]

        def run_pair(phase):
            processes = []
            try:
                for _ in range(2):
                    processes.append(subprocess.Popen(
                        [*command, phase], env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True))
                if phase == "cold":
                    await_publication(tmp_path, processes)

                results = []
                for process in processes:
                    output, errors = process.communicate(timeout=scaled(30))
                    assert process.returncode == 0, errors
                    results.append(json.loads(output))
                return results
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.communicate()

        cold = run_pair("cold")
        assert cold[0] == cold[1]
        warm = run_pair("warm")

        if kind == "registry":
            from kernel_release_switch_registry import build_registry_document

            expected = build_registry_document(tmp_path / "engine")
            assert cold[0] == [expected["switches"], expected["digest"], 1]
            assert warm == [[expected["switches"], expected["digest"], 0]] * 2
        else:
            assert cold == warm == [[4, "tokenize"]] * 2
            assert server.tokenize_request_count == 2


def await_publication(root, processes):
    markers = [root / f"ready-{process.pid}" for process in processes]
    deadline = time.monotonic() + scaled(20)
    while not all(marker.exists() and marker.stat().st_size for marker in markers):
        exited = [process for process in processes if process.poll() is not None]
        assert not exited, [process.communicate() for process in exited]
        assert time.monotonic() < deadline, "both processes must reach atomic publication"
        time.sleep(0.01)

    left, right = (Path(marker.read_text()) for marker in markers)
    assert left == right
    assert not left.exists(), "a reader could see an unfinished cache entry"
    (root / "publish").touch()

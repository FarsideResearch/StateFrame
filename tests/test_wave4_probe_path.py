"""Cross-launch reuse and live checks for executable authority."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys

from _child_env import child_env
import time
from types import SimpleNamespace

import pytest

import kernel_exec_capability as child
from kernel_agent_jobs import AgentJobs
from kernel_custom_tools import load_tool_files
from kernel_shell_tool import ShellExecutionFailed, ShellTool
from kernel_tool_settings import ToolSettings
from kernel_tool_tape import configure_replay
from stateframe import shell_host
from stateframe.engine import kernel_exec_capability as host
from test_shell_tool import ProcessBoundary


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="requires Linux PATH stamps")


class FileStore:
    """Stand in for authenticated storage when the sibling module is absent."""

    def __init__(self, path):
        self.path = path
        self.published = 0

    def host_fingerprint(self):
        return "test host"

    def launch_scope(self, selected, identity):
        return nullcontext()

    def lookup(self, kind, fingerprint, *, max_age):
        try:
            return json.loads(self.path.read_text()).get(fingerprint)
        except (OSError, ValueError):
            return None

    def publish(self, kind, fingerprint, facts):
        self.path.write_text(json.dumps({fingerprint: facts}))
        self.published += 1


def executable(path, interpreter=None):
    path.write_bytes(f"#!{interpreter}\n".encode() if interpreter else b"executable\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    first, second, targets, run, workspace = [
        tmp_path / name for name in ("first", "second", "targets", "run", "workspace")
    ]
    for path in (first, second, targets, run, workspace):
        path.mkdir()
    old, new = (executable(targets / name) for name in ("old", "new"))
    script = executable(first / "example", old)
    link = first / "linked"
    link.symlink_to(old)
    plain = first / "plain"
    plain.write_bytes(b"plain\n")
    missing = tmp_path / "missing-bin"
    missing_interpreter = targets / "missing"
    executable(second / "missing-target", missing_interpreter)
    (second / "dangling").symlink_to(targets / "missing-link")
    path = os.pathsep.join(map(str, (first, second, missing)))
    store = FileStore(tmp_path / "cache.json")

    # The tests also run on Linux hosts whose checkout is on an overlay mount.
    devices = {path.stat().st_dev for path in (*first.parents, first, Path(sys.executable))}
    observed_devices = {f"{os.major(dev)}:{os.minor(dev)}".encode() for dev in devices}
    for module in (host, child):
        monkeypatch.setattr(module, "_local_path_devices", lambda: observed_devices)
    monkeypatch.setenv("PATH", path)
    monkeypatch.setattr(shell_host, "SYSTEM_SHELL_PATHS", ())
    monkeypatch.setattr(child, "SYSTEM_SHELL_PATHS", ())
    monkeypatch.setattr(shell_host, "_probe_store", lambda: store)
    return SimpleNamespace(first=first, second=second, targets=targets, old=old, new=new,
                           script=script, link=link, plain=plain, missing=missing,
                           missing_interpreter=missing_interpreter, path=path,
                           run=run, workspace=workspace, store=store)


def provision(fixture, *, pilot=True, **kwargs):
    settings = ToolSettings(run_env=False, **kwargs)
    spec = {"repo": str(fixture.workspace)}
    if pilot:
        spec["send1_wave4"] = True
    shell_host.provision(SimpleNamespace(tool_settings=settings), fixture.run, spec)
    return settings, spec


def grants(spec):
    return spec["shell_path_dirs"], spec["shell_read_roots"]


def _subprocess_inventory(root, pilot):
    store = FileStore(root / "cache.json")
    shell_host._probe_store = lambda: store
    shell_host.SYSTEM_SHELL_PATHS = ()
    devices = {path.stat().st_dev for path in (root, *root.parents)}
    host._local_path_devices = lambda: {
        f"{os.major(dev)}:{os.minor(dev)}".encode() for dev in devices
    }
    counts = {"enumeration": 0, "header": 0, "metadata_seconds": 0.0}
    original_scan = Path.iterdir
    original_open = Path.open
    original_verify = shell_host.current_path_authority_fingerprint

    def scan(path):
        counts["enumeration"] += 1
        return original_scan(path)

    def opened(path, mode="r", *args, **kwargs):
        if mode == "rb" and path.is_relative_to(root):
            counts["header"] += 1
        return original_open(path, mode, *args, **kwargs)

    def verify(inputs):
        started = time.perf_counter()
        try:
            return original_verify(inputs)
        finally:
            counts["metadata_seconds"] += time.perf_counter() - started

    Path.iterdir, Path.open = scan, opened
    shell_host.current_path_authority_fingerprint = verify
    fixture = SimpleNamespace(run=root / "run", workspace=root / "workspace")
    _, spec = provision(fixture, pilot=pilot)
    print(json.dumps({"counts": counts, "grants": grants(spec), "published": store.published}))


def fresh_process(fixture, pilot=True):
    source = (
        "from pathlib import Path; from test_wave4_probe_path import _subprocess_inventory; "
        f"_subprocess_inventory(Path({str(fixture.run.parent)!r}), {pilot!r})"
    )
    result = subprocess.run([sys.executable, "-c", source], check=True, capture_output=True,
                            text=True, timeout=30, env=child_env())
    return json.loads(result.stdout)


def test_verified_hit_skips_enumeration_and_headers_across_processes(inventory):
    off, cold, warm = (fresh_process(inventory, pilot) for pilot in (False, True, True))
    assert off["grants"] == cold["grants"] == warm["grants"]
    assert off["published"] == 0 and cold["published"] == 1 and warm["published"] == 0
    assert cold["counts"]["enumeration"] == 2
    assert cold["counts"]["header"] >= 1
    assert warm["counts"]["enumeration"] == warm["counts"]["header"] == 0
    assert warm["counts"]["metadata_seconds"] > 0


@pytest.mark.parametrize("change", [
    "order", "spelling", "new-entry", "chmod", "replacement", "shebang", "link",
    "missing-directory", "missing-interpreter", "dangling-entry", "ancestor-link",
])
def test_warm_inventory_matches_fresh_scan_after_mutation(inventory, monkeypatch, change):
    f = inventory
    assert fresh_process(f)["published"] == 1
    assert fresh_process(f)["published"] == 0
    directory_before = f.first.stat()
    if change == "order":
        monkeypatch.setenv("PATH", os.pathsep.join(map(str, (f.second, f.first, f.missing))))
    elif change == "spelling":
        monkeypatch.setenv("PATH", f.path + os.pathsep + str(f.first) + "/")
    elif change == "new-entry":
        executable(f.first / "added", f.new)
    elif change == "chmod":
        f.plain.chmod(0o755)
    elif change == "replacement":
        executable(f.targets / "replacement", f.new).replace(f.script)
    elif change == "shebang":
        before = f.script.stat()
        f.script.write_bytes(f"#!{f.new}\n".encode())
        os.utime(f.script, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == "link":
        f.link.unlink()
        f.link.symlink_to(f.new)
    elif change == "missing-directory":
        f.missing.mkdir()
        executable(f.missing / "added")
    elif change == "missing-interpreter":
        executable(f.missing_interpreter)
    elif change == "dangling-entry":
        executable(f.targets / "missing-link")
    else:
        relocated = f.targets.with_name("relocated")
        f.targets.rename(relocated)
        f.targets.symlink_to(relocated)
    if change in {"replacement", "link"}:
        os.utime(f.first, ns=(directory_before.st_atime_ns, directory_before.st_mtime_ns))
    if change != "new-entry":
        assert f.first.stat().st_mtime_ns == directory_before.st_mtime_ns
    changed = fresh_process(f)
    assert changed["published"] == 1
    assert changed["counts"]["enumeration"] >= 2
    assert changed["grants"] == fresh_process(f, False)["grants"]
    assert fresh_process(f)["published"] == 0


@pytest.mark.parametrize("storage", ["corrupt", "malformed", "unavailable"])
def test_unusable_storage_takes_live_scan(inventory, monkeypatch, storage):
    f = inventory
    _, off = provision(f, pilot=False)
    _, cold = provision(f)
    if storage == "corrupt":
        f.store.path.write_text("{")
    elif storage == "malformed":
        key, = json.loads(f.store.path.read_text())
        f.store.path.write_text(json.dumps({key: {"inventory": "AAAA"}}))
    else:
        def unavailable(*args, **kwargs):
            raise PermissionError("storage unavailable")
        monkeypatch.setattr(f.store, "lookup", unavailable)
        monkeypatch.setattr(f.store, "publish", unavailable)
    _, after = provision(f)
    assert grants(off) == grants(cold) == grants(after)
    assert after["shell_path_snapshot"] == cold["shell_path_snapshot"]


def test_reprovision_retains_recorded_authority(inventory, monkeypatch):
    f = inventory
    settings, _ = provision(f)
    recorded = json.loads((f.run / "spec.json").read_text())
    snapshot = recorded["shell_path_snapshot"]
    assert child.current_path_authority_fingerprint(snapshot["inputs"]) == snapshot["fingerprint"]

    f.plain.chmod(0o755)
    monkeypatch.delenv("STATEFRAME_SEND1_WAVE4", raising=False)
    monkeypatch.setattr(shell_host, "_probe_store", lambda: pytest.fail("reopened startup store"))
    shell_host.provision(SimpleNamespace(tool_settings=settings), f.run, recorded)
    assert recorded["shell_path_snapshot"] == snapshot
    assert child.current_path_authority_fingerprint(snapshot["inputs"]) != snapshot["fingerprint"]


def test_dispatch_uses_live_additions_and_refuses_stale_grants(inventory):
    f = inventory
    explicit = f.targets.with_name("explicit")
    explicit.mkdir()
    _, off = provision(f, pilot=False, shell_read=(str(explicit),))
    provision(f)
    settings, hit = provision(f, shell_read=(str(explicit),))
    assert f.store.published == 1
    assert grants(off) == grants(hit)

    per_run_bin = f.run / "shell-cache" / "bin"
    per_run_bin.mkdir(parents=True)
    dynamic = executable(per_run_bin / "dynamic", f.new)
    definitions = f.run / "tools.json"
    definitions.write_text(json.dumps({
        "schema": "stateframe-tools/1",
        "tools": [{
            "name": "custom_check", "description": "Run the example",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "argv": [str(f.script)], "network": "none", "writes": "none",
            "replay": "local", "timeout_seconds": 13,
        }],
    }))
    custom, = load_tool_files([definitions], builtin_names=frozenset({"shell"}),
                              shell_network=settings.shell_network).tools

    def collect(spec):
        process = ProcessBoundary()
        tool = ShellTool(f.run, settings, spec, process)
        tool.bind(f.workspace, {"files": {}, "allowed_product_paths": [], "new_file_roots": []}, set())
        configure_replay(f.run, active=False)
        tool.execute({"command": "example", "timeout_seconds": 17}, "shell")
        tool._execute({}, "custom", custom=custom)
        job = AgentJobs(tool, time.monotonic() + 30)
        caps = [cap for _, cap, _ in process.calls] + [job.capability("j1", 23)]
        return tool, process, job, caps

    _, _, _, off_caps = collect(off)
    tape = f.run / "tool-tape.jsonl"
    off_receipts = tape.read_bytes()
    tape.unlink()
    tool, process, job, hit_caps = collect(hit)
    assert [cap.as_dict() for cap in off_caps] == [cap.as_dict() for cap in hit_caps]
    assert tape.read_bytes() == off_receipts
    assert [cap.limits.wall_seconds for cap in hit_caps] == [17, 13, 23]
    assert all(any(dynamic.is_relative_to(root) for root in cap.read_roots) for cap in hit_caps)
    assert all({explicit, f.old, f.new}.issubset(cap.read_roots) for cap in hit_caps)

    f.plain.chmod(0o755)
    with pytest.raises(ShellExecutionFailed, match="shell PATH authority changed"):
        tool.execute({"command": "plain"}, "stale-shell")
    with pytest.raises(ShellExecutionFailed, match="shell PATH authority changed"):
        tool._execute({}, "stale-custom", custom=custom)
    with pytest.raises(ShellExecutionFailed, match="shell PATH authority changed"):
        job.launch({"command": "plain"})
    assert len(process.calls) == 2

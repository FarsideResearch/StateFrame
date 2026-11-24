"""Weak sharing of immutable launch capabilities."""
from dataclasses import FrozenInstanceError, asdict, replace
import gc
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import pytest

import kernel_exec_supervisor as supervisor_module
from kernel_exec_capability import ExecCapability, ExecLimits, shell_capability
from kernel_exec_session import ExecSession
from kernel_exec_supervisor import ManagedProcess, Supervisor


class _Table:
    def identity(self, _pid):
        return None


class _Process:
    def __init__(self, pid):
        self.pid = pid
        self.stdout = SimpleNamespace(close=lambda: None)

    def poll(self):
        return None


def _small_capability(tmp_path, *, env=None, network="none", roots=(), write_roots=(),
                      limits=None, purpose="job", workspace_writable=False,
                      sandboxed=False, workspace_name="workspace"):
    workspace = tmp_path / workspace_name
    workspace.mkdir(exist_ok=True)
    return ExecCapability(
        purpose, workspace, workspace_writable, tuple(roots), tuple(write_roots), network,
        {} if env is None else env, ExecLimits() if limits is None else limits, sandboxed,
    )


def _cache_supervisor():
    supervisor = object.__new__(Supervisor)
    supervisor._capability_cache = weakref.WeakValueDictionary()
    return supervisor


def _launch_supervisor(tmp_path, monkeypatch, popen):
    directory = tmp_path / "supervisor"
    directory.mkdir()
    supervisor = object.__new__(Supervisor)
    supervisor.directory = directory
    supervisor.backend = "host"
    supervisor.namespace_network = "host"
    supervisor.table = _Table()
    supervisor.jobs = {}
    supervisor.next_id = 0
    supervisor._capability_cache = weakref.WeakValueDictionary()
    monkeypatch.setattr(supervisor_module, "_cgroup_memory_limit_configured", lambda *_: False)
    monkeypatch.setattr(supervisor_module, "capability_command",
                        lambda command, cap, **_kwargs: (command, dict(cap.env)))
    monkeypatch.setattr(supervisor_module.subprocess, "Popen", popen)
    monkeypatch.setattr(supervisor_module, "_cgroup_leaf_pattern_status", lambda: None)
    return supervisor


def _request(capability, **values):
    return {"cap": capability.as_dict(), "argv": ["synthetic-command"], **values}


def test_identical_capabilities_share_while_jobs_hold_the_reference(tmp_path, monkeypatch):
    processes = iter(range(1000, 1002))
    supervisor = _launch_supervisor(tmp_path, monkeypatch,
                                    lambda *_args, **_kwargs: _Process(next(processes)))
    source = _small_capability(tmp_path, env={"FIRST": "one", "SECOND": "two"})

    first_result = supervisor.start(_request(source))
    first_cap = supervisor.jobs[first_result["id"]].cap
    reordered = source.as_dict()
    reordered["env"] = {"SECOND": "two", "FIRST": "one"}
    second_result = supervisor.start({**_request(source), "cap": reordered})
    second_cap = supervisor.jobs[second_result["id"]].cap

    assert second_cap is first_cap
    assert len(supervisor._capability_cache) == 1
    reference = weakref.ref(first_cap)
    for job in supervisor.jobs.values():
        job.collector.close()
        job.process.stdout.close()
    del job
    supervisor.jobs.clear()
    del first_cap, second_cap
    gc.collect()
    assert reference() is None
    assert supervisor._capability_cache == {}


@pytest.mark.parametrize("field", [
    "purpose", "workspace", "workspace_writable", "read_roots", "write_roots",
    "network", "env", "limits", "sandboxed",
])
def test_changing_any_capability_field_never_shares(field, tmp_path):
    read_a, read_b = tmp_path / "read-a", tmp_path / "read-b"
    write_a, write_b = tmp_path / "write-a", tmp_path / "write-b"
    other_workspace = tmp_path / "other-workspace"
    for path in (read_a, read_b, write_a, write_b, other_workspace):
        path.mkdir()
    base = _small_capability(tmp_path, env={"MODE": "one"}, roots=(read_a,),
                             write_roots=(write_a,))
    changed = {
        "purpose": replace(base, purpose="shell"),
        "workspace": replace(base, workspace=other_workspace),
        "workspace_writable": replace(base, workspace_writable=True),
        "read_roots": replace(base, read_roots=(read_b,)),
        "write_roots": replace(base, write_roots=(write_b,)),
        "network": replace(base, network="full"),
        "env": replace(base, env={"MODE": "two"}),
        "limits": replace(base, limits=replace(base.limits, nproc=base.limits.nproc - 1)),
        "sandboxed": replace(base, sandboxed=True),
    }[field]
    supervisor = _cache_supervisor()
    first, key = supervisor._capability_for(base.as_dict())
    supervisor._capability_cache[key] = first

    second, second_key = supervisor._capability_for(changed.as_dict())

    assert second is not first
    assert second_key != key


def test_shared_capability_fields_and_nested_values_are_immutable(tmp_path):
    read_root = tmp_path / "read-root"
    read_root.mkdir()
    input_env = {"API_TOKEN": "synthetic-value"}
    source = _small_capability(tmp_path, env=input_env, roots=(read_root,))
    input_env["API_TOKEN"] = "changed"
    supervisor = _cache_supervisor()
    first, key = supervisor._capability_for(source.as_dict())
    supervisor._capability_cache[key] = first
    second, _ = supervisor._capability_for(source.as_dict())

    assert second is first
    with pytest.raises(FrozenInstanceError):
        first.network = "full"
    with pytest.raises(TypeError):
        first.env["API_TOKEN"] = "changed"
    with pytest.raises(FrozenInstanceError):
        first.limits.nproc = 1
    with pytest.raises(TypeError):
        first.read_roots[0] = tmp_path
    with pytest.raises(FrozenInstanceError):
        first.read_roots = ()
    assert first.env["API_TOKEN"] == "synthetic-value"
    assert second.read_roots == (read_root,)


def test_wire_dict_mutation_cannot_change_cached_capability(tmp_path):
    source = _small_capability(tmp_path, env={"API_TOKEN": "synthetic-value"})
    wire = source.as_dict()
    capability, _ = _cache_supervisor()._capability_for(wire)

    wire["env"]["API_TOKEN"] = "changed"
    wire["read_roots"].clear()

    assert capability.env["API_TOKEN"] == "synthetic-value"
    assert capability.read_roots == source.read_roots


def test_environment_order_does_not_split_identical_capabilities(tmp_path):
    source = _small_capability(tmp_path, env={"FIRST": "one", "SECOND": "two"})
    supervisor = _cache_supervisor()
    first, key = supervisor._capability_for(source.as_dict())
    supervisor._capability_cache[key] = first
    wire = source.as_dict()
    wire["env"] = {"SECOND": "two", "FIRST": "one"}

    second, second_key = supervisor._capability_for(wire)

    assert second is first
    assert second_key == key


def test_as_dict_mutation_cannot_change_nested_authority_fields(tmp_path):
    root = tmp_path / "read-root"
    root.mkdir()
    capability = _small_capability(tmp_path, env={"API_TOKEN": "synthetic-value"}, roots=(root,))
    wire = capability.as_dict()
    wire["env"]["API_TOKEN"] = "changed"
    wire["read_roots"].clear()
    wire["write_roots"].append(str(root))
    wire["limits"]["nproc"] = 1

    assert capability.env["API_TOKEN"] == "synthetic-value"
    assert capability.read_roots == (root,)
    assert capability.write_roots == ()
    assert capability.limits.nproc == ExecLimits().nproc


def test_weak_cache_retains_secret_only_while_a_job_holds_capability(tmp_path):
    supervisor = _cache_supervisor()
    secret = "synthetic-cache-value"
    capability, key = supervisor._capability_for(
        _small_capability(tmp_path, env={"API_TOKEN": secret}).as_dict())
    job = SimpleNamespace(cap=capability)
    supervisor._capability_cache[key] = capability
    reference = weakref.ref(capability)
    del capability

    gc.collect()
    assert reference() is job.cap
    assert secret.encode() not in key
    assert supervisor._capability_cache[key] is job.cap

    del job
    gc.collect()
    assert reference() is None
    assert key not in supervisor._capability_cache


def test_failed_popen_does_not_cache_capability(tmp_path, monkeypatch):
    def fail_to_start(*_args, **_kwargs):
        raise OSError("synthetic process start failure")

    supervisor = _launch_supervisor(tmp_path, monkeypatch, fail_to_start)
    capability = _small_capability(tmp_path)

    with pytest.raises(OSError, match="synthetic process start failure"):
        supervisor.start(_request(capability))

    assert supervisor._capability_cache == {}


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="post-launch identity qualification is Linux-specific")
def test_failed_post_launch_identity_does_not_cache_capability(tmp_path, monkeypatch):
    supervisor = _launch_supervisor(tmp_path, monkeypatch, lambda *_args, **_kwargs: _Process(1000))

    def fail_identity(*_args):
        raise RuntimeError("synthetic identity failure")

    monkeypatch.setattr(supervisor, "_wait_for_final_exec", fail_identity)
    monkeypatch.setattr(ManagedProcess, "stop", lambda *_args: None)
    capability = replace(_small_capability(tmp_path), purpose="mutation")

    with pytest.raises(RuntimeError, match="final-exec identity unavailable"):
        supervisor.start(_request(capability, mounts={"boot": str(tmp_path)}))

    assert supervisor._capability_cache == {}
    for job in supervisor.jobs.values():
        job.collector.close()
        job.process.stdout.close()


def test_cache_drops_many_unique_values_with_their_jobs(tmp_path):
    supervisor = _cache_supervisor()
    jobs = []
    for index in range(1000):
        capability = _small_capability(tmp_path, env={"API_TOKEN": f"synthetic-{index}"})
        cached, key = supervisor._capability_for(capability.as_dict())
        jobs.append(SimpleNamespace(cap=cached))
        supervisor._capability_cache[key] = cached

    assert len(supervisor._capability_cache) == 1000
    jobs.clear()
    del capability, cached, key
    gc.collect()
    assert supervisor._capability_cache == {}


def test_digest_collision_does_not_merge_distinct_capabilities(monkeypatch, tmp_path):
    class _FixedDigest:
        def digest(self):
            return b"x" * 32

    monkeypatch.setattr(supervisor_module.hashlib, "sha256", lambda _data: _FixedDigest())
    supervisor = _cache_supervisor()
    first_source = _small_capability(tmp_path, env={"API_TOKEN": "synthetic-a"})
    second_source = _small_capability(tmp_path, env={"API_TOKEN": "synthetic-b"})
    first, key = supervisor._capability_for(first_source.as_dict())
    supervisor._capability_cache[key] = first

    second, second_key = supervisor._capability_for(second_source.as_dict())

    assert second is not first
    assert second_key == key
    assert first.env["API_TOKEN"] == "synthetic-a"
    assert second.env["API_TOKEN"] == "synthetic-b"


def _dict_state_pickle(cls, state):
    value = cls.__new__(cls)
    value.__dict__.update(state)
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def test_dict_state_pickles_of_both_dataclasses_still_load(tmp_path):
    limits = ExecLimits()
    restored_limits = pickle.loads(_dict_state_pickle(ExecLimits, asdict(limits)))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    capability_state = {
        "purpose": "job",
        "workspace": workspace,
        "workspace_writable": False,
        "read_roots": (),
        "write_roots": (),
        "network": "none",
        "env": {"MODE": "legacy"},
        "limits": limits,
        "sandboxed": False,
    }
    restored_capability = pickle.loads(_dict_state_pickle(ExecCapability, capability_state))

    assert restored_limits == limits
    assert restored_limits.__dict__ == asdict(limits)
    assert restored_capability.__dict__ == capability_state
    assert restored_capability.env == {"MODE": "legacy"}


def _rss_anon_bytes(pid):
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        pytest.skip("resident memory sampling requires procfs")
    for line in status.read_text(encoding="ascii").splitlines():
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) * 1024
    pytest.fail("supervisor status omitted RssAnon")


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="RssAnon sampling requires Linux procfs")
def test_320_real_launches_keep_rss_slope_below_one_megabyte_per_hundred(tmp_path):
    workspace, run_env, home, run_dir = (tmp_path / name for name in ("workspace", "env", "home", "run"))
    for directory in (workspace, run_env, home, run_dir):
        directory.mkdir()
    capability = replace(
        shell_capability(workspace=workspace, run_env_dir=run_env, scratch_home=home,
                         network="none", writable=False),
        purpose="job", workspace_writable=False, sandboxed=False,
    )
    samples = []
    with ExecSession(run_dir, platform_backend="bwrap") as session:
        for index in range(320):
            result = session.run(["/bin/true"], capability, cwd=workspace)
            assert result.exit_code == 0
            assert result.output == b""
            if index % 100 == 0:
                samples.append(_rss_anon_bytes(session._process.pid))

    slopes = [right - left for left, right in zip(samples, samples[1:])]
    assert len(slopes) == 3
    assert max(slopes) < 1_000_000, f"RssAnon growth per 100 jobs: {slopes} bytes"


def test_finished_job_output_is_incremental_when_read_twice(tmp_path):
    workspace, run_env, home, run_dir = (tmp_path / name for name in ("workspace", "env", "home", "run"))
    for directory in (workspace, run_env, home, run_dir):
        directory.mkdir()
    capability = replace(
        shell_capability(workspace=workspace, run_env_dir=run_env, scratch_home=home,
                         network="none", writable=False),
        purpose="job", workspace_writable=False, sandboxed=False,
    )

    with ExecSession(run_dir, platform_backend="bwrap") as session:
        job = session.spawn(["/bin/echo", "finished output"], capability, cwd=workspace)
        assert job.wait(30) == 0
        assert job.read_available() == b"finished output\n"
        assert job.read_available() == b""

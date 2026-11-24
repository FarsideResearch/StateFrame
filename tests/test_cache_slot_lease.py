import hashlib
import multiprocessing
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

import stateframe.parallel_resources as parallel_resources
import stateframe.runner as runner
from stateframe.endpoint import Endpoint
from stateframe.engine.kernel_provider_dialect import LLAMA_CPP, ProviderCapabilities


def _fake_slots(monkeypatch, endpoint="http://127.0.0.1:8080/v1", count=12):
    # The sandbox denies sockets; keep Endpoint's /slots result local and deterministic.
    slots = [{"id": slot, "is_processing": False} for slot in range(count)]
    monkeypatch.setattr(Endpoint, "get_slots", lambda _self, **_kwargs: slots)
    return endpoint


def _install_child_slots():
    Endpoint.get_slots = lambda _self, **_kwargs: [
        {"id": slot, "is_processing": False} for slot in range(12)
    ]


def _capabilities(*_args, **_kwargs):
    return ProviderCapabilities(dialect=LLAMA_CPP, slots=True, slot_ids=frozenset(range(12)))


def _options(endpoint, slot=None):
    return runner.RunOptions(
        repo=Path.cwd(), task="slot lease test", endpoint=Endpoint(endpoint, "test-model"), slot=slot,
    )


def _select_worker(endpoint, lease_root, barrier, release, results):
    _install_child_slots()
    parallel_resources.LEASE_ROOT = Path(lease_root)
    runner.safe_home = lambda: Path(lease_root).parent
    os.environ["STATEFRAME_PROMPT_CACHE_UNPINNED"] = "off"
    runner._provider_capabilities_for = _capabilities
    options = _options(endpoint)
    try:
        barrier.wait()
        runner._select_prompt_cache_slot(options, lambda *_: None)
        results.put((options._cache_slot, options._cache_slot_selection, options._cache_slot_reason))
        release.wait()
    except BaseException as error:
        results.put(("error", type(error).__name__, str(error)))
        raise


def _hold_selected_worker(endpoint, lease_root, ready, release):
    _install_child_slots()
    parallel_resources.LEASE_ROOT = Path(lease_root)
    runner.safe_home = lambda: Path(lease_root).parent
    os.environ["STATEFRAME_PROMPT_CACHE_UNPINNED"] = "off"
    runner._provider_capabilities_for = _capabilities
    options = _options(endpoint)
    runner._select_prompt_cache_slot(options, lambda *_: None)
    ready.put((options._cache_slot, options._cache_slot_selection))
    ready.close()
    ready.join_thread()  # os._exit would drop a queued result
    release.wait()
    os._exit(9)  # die without any cleanup; only the kernel can free the lock


def _setup(monkeypatch, tmp_path, endpoint):
    monkeypatch.setattr(parallel_resources, "LEASE_ROOT", tmp_path / "leases")
    monkeypatch.setattr(runner, "safe_home", lambda: tmp_path)
    monkeypatch.setattr(runner, "_provider_capabilities_for", _capabilities)
    monkeypatch.setenv("STATEFRAME_PROMPT_CACHE_UNPINNED", "off")
    return _options(endpoint)


def _release(options):
    descriptor, options._run_slot_fd = options._run_slot_fd, None
    if descriptor is not None:
        os.close(descriptor)


def _stub_run(monkeypatch, execute):
    monkeypatch.setattr(runner, "_validate_stateframe_home_boundary", lambda *_args: None)
    monkeypatch.setattr(runner, "_capture_stateframe_home_identity", lambda *_args: None)
    monkeypatch.setattr(runner, "_startup_probe_seconds", lambda: None)
    monkeypatch.setattr(runner, "_startup_optimization_enabled", lambda: False)
    monkeypatch.setattr(runner, "resolve_context", lambda *_args: (65_536, 33_344, "test", {}))
    monkeypatch.setattr(runner, "_run", execute)


def test_four_processes_select_four_distinct_idle_slots(tmp_path, monkeypatch):
    context = multiprocessing.get_context("spawn")
    endpoint = _fake_slots(monkeypatch)
    barrier = context.Barrier(5)
    release = context.Event()
    results = context.Queue()
    processes = [context.Process(target=_select_worker,
                                 args=(endpoint, str(tmp_path / "leases"), barrier, release, results),
                                 daemon=True)
                 for _ in range(4)]
    try:
        for process in processes:
            process.start()
        barrier.wait(timeout=20)
        selected = [results.get(timeout=20) for _ in processes]
        assert all(isinstance(item[0], int) for item in selected), selected
        assert sorted(item[0] for item in selected) == [0, 1, 2, 3]
        assert {item[1] for item in selected} == {"automatic_idle"}
    finally:
        barrier.abort()  # a worker that never met the others must not wait forever
        release.set()
        for process in processes:
            process.join(timeout=20)
        results.close()
        results.join_thread()


def test_process_death_releases_automatic_pin(tmp_path, monkeypatch):
    context = multiprocessing.get_context("spawn")
    endpoint = _fake_slots(monkeypatch)
    ready = context.Queue()
    release = context.Event()
    process = context.Process(target=_hold_selected_worker,
                              args=(endpoint, str(tmp_path / "leases"), ready, release),
                              daemon=True)
    process.start()
    try:
        assert ready.get(timeout=20) == (0, "automatic_idle")
        second = _setup(monkeypatch, tmp_path, endpoint)
        runner._select_prompt_cache_slot(second, lambda *_: None)
        assert second._cache_slot == 1
        _release(second)
        release.set()
        process.join(timeout=20)
        assert process.exitcode == 9
        after_kill = _setup(monkeypatch, tmp_path, endpoint)
        runner._select_prompt_cache_slot(after_kill, lambda *_: None)
        assert after_kill._cache_slot == 0
        _release(after_kill)
    finally:
        release.set()
        process.join(timeout=20)
        ready.close()
        ready.join_thread()


def test_thirteenth_run_is_not_pinned_when_all_twelve_run_slots_are_locked(tmp_path, monkeypatch):
    endpoint = _fake_slots(monkeypatch)
    held = [_setup(monkeypatch, tmp_path, endpoint) for _ in range(12)]
    try:
        for slot, options in enumerate(held):
            runner._select_prompt_cache_slot(options, lambda *_: None)
            assert options._cache_slot == slot
        options = _setup(monkeypatch, tmp_path, endpoint)
        runner._select_prompt_cache_slot(options, lambda *_: None)
        assert options._cache_slot is None
        assert options._cache_slot_selection == "not_pinned"
    finally:
        for options in held:
            _release(options)


@pytest.mark.parametrize("failure", ["directory", "flock"])
def test_run_lock_failure_keeps_legacy_lowest_idle_pin(tmp_path, monkeypatch, failure):
    endpoint = _fake_slots(monkeypatch)
    options = _setup(monkeypatch, tmp_path, endpoint)

    if failure == "directory":
        def deny_lease_directory(*_args):
            raise PermissionError("lease directory is not writable")

        monkeypatch.setattr(runner, "private_directory", deny_lease_directory)
    else:
        def deny_flock(*_args):
            raise OSError("flock is unavailable")

        monkeypatch.setattr(parallel_resources.fcntl, "flock", deny_flock)
    runner._select_prompt_cache_slot(options, lambda *_: None)
    assert options._cache_slot == 0
    assert options._cache_slot_selection == "automatic_idle"
    assert options._run_slot_fd is None


def test_missing_flock_keeps_legacy_lowest_idle_pin(tmp_path, monkeypatch):
    endpoint = _fake_slots(monkeypatch)
    options = _setup(monkeypatch, tmp_path, endpoint)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace())
    runner._select_prompt_cache_slot(options, lambda *_: None)
    assert options._cache_slot == 0
    assert options._run_slot_fd is None


def test_symlinked_run_slot_directory_is_not_followed(tmp_path, monkeypatch):
    endpoint = _fake_slots(monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    options = _setup(monkeypatch, tmp_path, endpoint)
    state_root = tmp_path / ".stateframe"
    state_root.mkdir()
    (state_root / "run-slots").symlink_to(outside, target_is_directory=True)
    runner._select_prompt_cache_slot(options, lambda *_: None)
    endpoint_hash = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:16]
    assert options._cache_slot == 0
    assert options._run_slot_fd is None
    assert not (outside / endpoint_hash).exists()


def test_symlinked_run_slot_file_is_not_followed(tmp_path, monkeypatch):
    endpoint = _fake_slots(monkeypatch)
    outside = tmp_path / "outside.lock"
    outside.write_text("untouched", encoding="utf-8")
    options = _setup(monkeypatch, tmp_path, endpoint)
    endpoint_hash = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:16]
    run_slot_dir = tmp_path / ".stateframe" / "run-slots" / endpoint_hash
    run_slot_dir.mkdir(parents=True)
    lock_path = run_slot_dir / "0.lock"
    lock_path.symlink_to(outside)
    runner._select_prompt_cache_slot(options, lambda *_: None)
    assert options._cache_slot == 0
    assert options._run_slot_fd is None
    assert lock_path.is_symlink()
    assert outside.read_text(encoding="utf-8") == "untouched"


def test_endpoint_leases_are_independent(tmp_path, monkeypatch):
    first = _fake_slots(monkeypatch, "http://127.0.0.1:8080/v1")
    second = _fake_slots(monkeypatch, "http://127.0.0.1:8081/v1")
    first_options = _setup(monkeypatch, tmp_path, first)
    runner._select_prompt_cache_slot(first_options, lambda *_: None)
    second_options = _setup(monkeypatch, tmp_path, second)
    runner._select_prompt_cache_slot(second_options, lambda *_: None)
    first_again = _setup(monkeypatch, tmp_path, first)
    runner._select_prompt_cache_slot(first_again, lambda *_: None)
    try:
        assert (first_options._cache_slot, second_options._cache_slot,
                first_again._cache_slot) == (0, 0, 1)
    finally:
        _release(first_options)
        _release(second_options)
        _release(first_again)


@pytest.mark.parametrize("count", [1, 2])
def test_subframe_can_take_run_pinned_parent_slot_first(tmp_path, monkeypatch, count):
    endpoint = _fake_slots(monkeypatch, count=count)
    parent = _setup(monkeypatch, tmp_path, endpoint)
    runner._select_prompt_cache_slot(parent, lambda *_: None)
    child_lease = None
    try:
        assert parent._cache_slot == 0
        candidates = parallel_resources.slot_candidates(
            None, list(range(count)), parent._cache_slot, fallback_capacity=count,
        )
        child_lease = parallel_resources.lease_slot(endpoint, candidates)
        assert child_lease is not None and child_lease.slot == 0
    finally:
        _release(parent)
        if child_lease is not None:
            child_lease.close()


def test_parent_run_pin_does_not_block_subframe_handoff(tmp_path, monkeypatch):
    endpoint = _fake_slots(monkeypatch)
    parent = _setup(monkeypatch, tmp_path, endpoint)
    runner._select_prompt_cache_slot(parent, lambda *_: None)
    candidates = parallel_resources.slot_candidates(
        None, [0, 1, 2], parent._cache_slot, fallback_capacity=3,
    )
    child_lease = parallel_resources.lease_slot(endpoint, candidates)
    next_parent = _setup(monkeypatch, tmp_path, endpoint)
    runner._select_prompt_cache_slot(next_parent, lambda *_: None)
    try:
        assert parent._cache_slot == 0
        assert child_lease is not None and child_lease.slot == 0
        assert next_parent._cache_slot == 1
    finally:
        _release(parent)
        if child_lease is not None:
            child_lease.close()
        _release(next_parent)


def test_single_run_keeps_existing_slot_request_and_summary_identity(tmp_path, monkeypatch):
    endpoint = _fake_slots(monkeypatch)
    options = _setup(monkeypatch, tmp_path, endpoint)
    runner._select_prompt_cache_slot(options, lambda *_: None)
    try:
        summary_slot = {
            "slot": options._cache_slot,
            "selection": options._cache_slot_selection,
            "reason": options._cache_slot_reason,
        }
        assert summary_slot == {
            "slot": 0,
            "selection": "automatic_idle",
            "reason": "lowest uniquely reported idle slot from loopback GET /slots",
        }
        assert runner._child_env(options)["REACT_SLOT_ID"] == "0"
        assert runner._route_defaults(options)["slot"] == 0
        run_slot_dir = (tmp_path / ".stateframe" / "run-slots" /
                        hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:16])
        assert stat.S_IMODE(run_slot_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE((run_slot_dir / "0.lock").stat().st_mode) == 0o600
    finally:
        _release(options)


def test_explicit_slot_is_unchanged_and_does_not_take_automatic_lease(tmp_path, monkeypatch):
    endpoint = _fake_slots(monkeypatch)
    options = _setup(monkeypatch, tmp_path, endpoint)
    options.slot = 7
    monkeypatch.setattr(runner, "_acquire_run_slot",
                        lambda *_args, **_kwargs: pytest.fail("explicit slot acquired an automatic lease"))
    runner._select_prompt_cache_slot(options, lambda *_: None)
    assert options._cache_slot == 7
    assert options._cache_slot_selection == "explicit"
    assert options._cache_slot_reason == "requested by --slot"
    assert options._run_slot_fd is None
    assert runner._child_env(options)["REACT_SLOT_ID"] == "7"


def test_run_releases_slot_descriptor_on_normal_return(tmp_path, monkeypatch):
    descriptor = os.open(tmp_path / "slot.lock", os.O_RDWR | os.O_CREAT, 0o600)
    options = _options("http://127.0.0.1:8080/v1")
    options._run_slot_fd = descriptor
    _stub_run(monkeypatch, lambda *_args: {"status": "ok"})
    assert runner.run(options) == {"status": "ok"}
    assert options._run_slot_fd is None
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_run_releases_slot_between_chat_or_resume_turns(tmp_path, monkeypatch):
    endpoint = _fake_slots(monkeypatch, count=1)
    options = _setup(monkeypatch, tmp_path, endpoint)
    selected = []

    def execute(current, *_args):
        runner._select_prompt_cache_slot(current, lambda *_: None)
        selected.append(current._cache_slot)
        return {"status": "ok"}

    _stub_run(monkeypatch, execute)
    assert runner.run(options) == {"status": "ok"}
    assert options._run_slot_fd is None
    assert runner.run(options) == {"status": "ok"}
    assert options._run_slot_fd is None
    assert selected == [0, 0]


def test_run_child_does_not_inherit_slot_lock_descriptor(tmp_path, monkeypatch):
    endpoint = _fake_slots(monkeypatch, count=1)
    options = _setup(monkeypatch, tmp_path, endpoint)

    def execute(current, *_args):
        runner._select_prompt_cache_slot(current, lambda *_: None)
        descriptor = current._run_slot_fd
        assert descriptor is not None and not os.get_inheritable(descriptor)
        child_code = (
            "import os, sys\n"
            "try: os.fstat(int(sys.argv[1]))\n"
            "except OSError: print('closed')\n"
            "else: print('open')\n"
        )
        child = subprocess.run(
            [sys.executable, "-c", child_code, str(descriptor)],
            check=True, capture_output=True, text=True, close_fds=False,
        )
        assert child.stdout.strip() == "closed"
        return {"status": "ok"}

    _stub_run(monkeypatch, execute)
    assert runner.run(options) == {"status": "ok"}
    assert options._run_slot_fd is None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_release_frees_the_slot_while_a_forked_child_lives(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "safe_home", lambda: tmp_path)
    slot, descriptor = runner._acquire_run_slot("http://127.0.0.1:1/v1", [0])
    assert slot == 0 and descriptor is not None
    read_end, write_end = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(write_end)
        os.read(read_end, 1)
        os._exit(0)
    os.close(read_end)
    try:
        assert runner._acquire_run_slot("http://127.0.0.1:1/v1", [0]) == (None, None)
        runner._release_run_slot(descriptor)
        again, second = runner._acquire_run_slot("http://127.0.0.1:1/v1", [0])
        assert again == 0 and second is not None
        runner._release_run_slot(second)
    finally:
        os.close(write_end)
        os.waitpid(child, 0)


def test_reselection_after_restart_with_fewer_slots_stays_in_range(tmp_path, monkeypatch):
    endpoint = _fake_slots(monkeypatch)
    first = _setup(monkeypatch, tmp_path, endpoint)
    runner._select_prompt_cache_slot(first, lambda *_: None)
    second = _setup(monkeypatch, tmp_path, endpoint)
    runner._select_prompt_cache_slot(second, lambda *_: None)
    assert (first._cache_slot, second._cache_slot) == (0, 1)
    # The first run ended, then the provider restarted with one slot.
    _release(first)
    _fake_slots(monkeypatch, count=1)
    lines = []
    runner._select_prompt_cache_slot(second, lines.append)
    assert second._cache_slot == 0
    assert any("slot 0" in line for line in lines)
    # The lease that was given up is free again; the new one is held.
    assert runner._acquire_run_slot(endpoint, [0]) == (None, None)
    again, descriptor = runner._acquire_run_slot(endpoint, [1])
    assert again == 1
    runner._release_run_slot(descriptor)
    _release(second)


def test_reselection_without_a_free_slot_drops_the_old_pin(tmp_path, monkeypatch):
    endpoint = _fake_slots(monkeypatch)
    first = _setup(monkeypatch, tmp_path, endpoint)
    runner._select_prompt_cache_slot(first, lambda *_: None)
    second = _setup(monkeypatch, tmp_path, endpoint)
    runner._select_prompt_cache_slot(second, lambda *_: None)
    _fake_slots(monkeypatch, count=1)
    runner._select_prompt_cache_slot(second, lambda *_: None)
    assert second._cache_slot is None
    assert second._cache_slot_selection == "not_pinned"
    assert second._run_slot_fd is None
    _release(first)

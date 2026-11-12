"""Qualification must demonstrate containment before enabling units."""
import os
import sys

import pytest

from kernel_mutate_qualify import qualify, selftest


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "linux", reason="Linux seccomp, syscall and procfs probes are required")
def test_linux_selftest_positive_and_negative_probes(tmp_path):
    report = selftest(tmp_path)
    assert report["ok"], [check for check in report["checks"] if not check["ok"]]
    names = {check["name"] for check in report["checks"]}
    assert {"positive", "fork", "posix_spawn", "subprocess", "clone3", "execveat",
            "af_inet", "af_inet6", "af_unix", "abstract_unix", "home_ssh",
            "engine_source", "etc_shadow", "write_tmp", "write_dev_shm",
            "write_unit", "write_in", "write_stdin", "thread", "dlopen_outside",
            "signal_supervisor", "write_proc_comm"} <= names
    assert {"prebuilt:" + name for name in (
        "file_facts", "structured_peek", "archive_list", "sqlite_peek",
        "snapshot_diff", "text_grep_struct", "log_watch", "html_extract")} <= names
    assert all(check["ok"] and check["detail"] for check in report["checks"])


def test_linux_probe_catalog_includes_procfs_write():
    import kernel_mutate_qualify as module
    assert "write_proc_comm" in module._negative_operations(123, "/outside/lib.so", "linux")


def test_qualification_binding_tracks_isolation_code(monkeypatch):
    import kernel_mutate_qualify as module
    original = module._digest_file
    first = module._binding()["profile_digest"]

    def changed(path):
        if path.name == "kernel_test_isolation.py":
            return "changed"
        return original(path)

    monkeypatch.setattr(module, "_digest_file", changed)
    assert module._binding()["profile_digest"] != first


def test_qualification_cache_fails_closed_when_binding_changes(tmp_path, monkeypatch):
    import kernel_mutate_qualify as module
    calls = []

    def fake_selftest(home):
        calls.append(home)
        return {"ok": True, "platform": "linux", "checks": [{"name": "positive", "ok": True,
                                                               "detail": "passed"}]}

    monkeypatch.setattr(module, "selftest", fake_selftest)
    first = qualify(tmp_path)
    second = qualify(tmp_path)
    assert first.ok and second.ok and len(calls) == 1
    monkeypatch.setattr(module, "_binding", lambda: {**first.record["binding"], "kernel": "changed"})
    third = qualify(tmp_path)
    assert third.ok and len(calls) == 2


def test_qualification_rejects_changed_stdlib_bytes_with_restored_metadata(tmp_path, monkeypatch):
    import kernel_mutate_qualify as module
    stdlib = tmp_path / "stdlib"
    stdlib.mkdir()
    library = stdlib / "module.py"
    library.write_bytes(b"original")
    monkeypatch.setattr(module, "_runtime_roots", lambda: (stdlib,))
    first = module._binding()["closure_digest"]
    before = library.stat()
    library.write_bytes(b"altered!")
    os.utime(library, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert module._binding()["closure_digest"] != first


def test_qualification_timeout_has_named_reason_after_one_retry():
    from types import SimpleNamespace
    import kernel_mutate_qualify as module

    class Executor:
        def __init__(self):
            self.calls = 0

        def run(self, _spec):
            self.calls += 1
            return SimpleNamespace(timed_out=True, stdout_overflow=False, exit_code=None,
                                   cleanup=SimpleNamespace(ok=True), stdout=b"")

    executor = Executor()
    result = module._run_qualification_check(
        executor, lambda: object(), name="simulated_slow_check",
        accepts=lambda _outcome: False, detail=lambda _outcome: "launch timed out",
    )

    assert result["ok"] is False
    assert result["reason"] == "MUTATION_PLATFORM_UNQUALIFIED_TIMEOUT"
    assert result["attempts"] == 2
    assert executor.calls == 2


def test_qualification_failure_has_named_reason_and_is_not_retried():
    from types import SimpleNamespace
    import kernel_mutate_qualify as module

    class Executor:
        calls = 0

        def run(self, _spec):
            self.calls += 1
            return SimpleNamespace(timed_out=False, stdout_overflow=False, exit_code=137,
                                   cleanup=SimpleNamespace(ok=True), stdout=b"")

    executor = Executor()
    result = module._run_qualification_check(
        executor, lambda: object(), name="simulated_failed_check",
        accepts=lambda outcome: outcome.exit_code == 0, detail=lambda _outcome: "exit=137",
    )

    assert result["ok"] is False
    assert result["reason"] == "MUTATION_PLATFORM_UNQUALIFIED_FAILURE"
    assert result["attempts"] == 1
    assert executor.calls == 1


@pytest.mark.parametrize("reason", [
    "MUTATION_PLATFORM_UNQUALIFIED_TIMEOUT",
    "MUTATION_PLATFORM_UNQUALIFIED_FAILURE",
])
def test_qualify_runtime_surfaces_named_refusal_reason(monkeypatch, tmp_path, reason):
    import sys
    from types import SimpleNamespace
    from stateframe import mutate_host
    from kernel_mutate_exec import MutationPlatformError

    monkeypatch.setattr(mutate_host.sys, "platform", "linux")

    monkeypatch.setitem(sys.modules, "kernel_mutate_qualify", SimpleNamespace(
        qualify=lambda _home, refresh=False: SimpleNamespace(
            ok=False, reasons=["check"], record={"report": {"checks": [{
                "name": "check", "ok": False, "reason": reason, "detail": "probe refused"}]}})))

    with pytest.raises(MutationPlatformError, match=reason):
        mutate_host.qualify_runtime(tmp_path)

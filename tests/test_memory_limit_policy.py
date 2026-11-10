"""Linux child-memory policy and audit metadata regressions."""
from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from _portable import TRUE

import kernel_generic_check as generic_check
import kernel_execution_policy as policy
import kernel_exec_supervisor
import sf_launch
from kernel_check_receipts import record_check_output


LIMIT = 64 << 20


def _darwin_guard(node: ast.AST) -> bool:
    return any(
        isinstance(part, ast.Compare)
        and ((isinstance(part.left, ast.Name) and part.left.id == "system")
             or (isinstance(part.left, ast.Attribute)
                 and part.left.attr in {"system", "platform"}))
        and isinstance(part.ops[0], ast.Eq)
        and isinstance(part.comparators[0], ast.Constant)
        and part.comparators[0].value in {"Darwin", "darwin"}
        for part in ast.walk(node)
        if isinstance(part, ast.Compare) and part.comparators
    )


def _address_limit_calls(tree: ast.AST, darwin_guarded: bool = False):
    if isinstance(tree, list):
        for node in tree:
            yield from _address_limit_calls(node, darwin_guarded)
        return
    if isinstance(tree, ast.If):
        yield from _address_limit_calls(tree.test, darwin_guarded)
        yield from _address_limit_calls(
            tree.body, darwin_guarded or _darwin_guard(tree.test))
        yield from _address_limit_calls(tree.orelse, darwin_guarded)
        return
    if (isinstance(tree, ast.Call) and isinstance(tree.func, ast.Attribute)
            and tree.func.attr == "setrlimit" and tree.args
            and isinstance(tree.args[0], ast.Attribute)
            and tree.args[0].attr == "RLIMIT_AS"):
        yield tree, darwin_guarded
    for child in ast.iter_child_nodes(tree):
        yield from _address_limit_calls(child, darwin_guarded)


def test_engine_address_space_limits_are_darwin_guarded_only():
    engine = Path(__file__).parents[1] / "src/stateframe/engine"
    violations = []
    for path in engine.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "RLIMIT_AS" not in source:
            continue
        module = ast.parse(source, filename=str(path))
        trees = [("module", module)]
        for node in ast.walk(module):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            value = node.value
            if names & {"RUNNER_SOURCE", "CHILD_SOURCE"} and isinstance(value, ast.Constant):
                if isinstance(value.value, str):
                    trees.append((next(iter(names)), ast.parse(value.value)))
        for source_name, tree in trees:
            for call, guarded in _address_limit_calls(tree):
                if not guarded:
                    violations.append(f"{path.relative_to(engine)}:{call.lineno}:{source_name}")
    assert violations == []


def test_slow_full_snapshot_does_not_delay_known_member_memory_sample(monkeypatch):
    row = SimpleNamespace(pid=123, start=(1, 0), zombie=False)

    class Table:
        def snapshot(self):
            time.sleep(0.5)
            return {}

        def refresh_members(self, _members, **_kwargs):
            return {row.pid: row}

        def resident_memory_limit_breach(self, live, _memory_bytes):
            return "aggregate resident-memory limit exceeded" if live else ""

    class Family:
        members = {row.pid: row}

        def __init__(self, *_args):
            pass

        def update(self, rows):
            return list(rows.values())

        def leader_reaped(self):
            pass

    class StopAfterOneSample:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _timeout):
            self.stopped = True

        def set(self):
            self.stopped = True

    process = SimpleNamespace(pid=123, returncode=None)
    process.poll = lambda: pytest.fail("memory monitor reaped the leader")
    monkeypatch.setattr(policy, "ProcessTable", Table)
    monkeypatch.setattr(policy, "Descendants", Family)
    monkeypatch.setattr(policy, "recorded_launch_identity", lambda _process: object())
    supervisor = policy._ResidentMemorySupervisor(process, LIMIT)
    supervisor._stopped = StopAfterOneSample()
    supervisor._stop_launch_unit = lambda: None

    started = time.monotonic()
    supervisor._run()

    assert time.monotonic() - started < 0.1
    assert process.memory_limit_killed is True


def test_namespace_init_walk_counts_double_forked_setsid_memory(monkeypatch):
    from kernel_exec_tracker import Descendants, ProcessIdentity, ProcessTable

    root = ProcessIdentity(100, 1, 100, 100, (10, 0), False)
    namespace_init = ProcessIdentity(101, 100, 100, 100, (11, 0), False)
    escaped = ProcessIdentity(102, 101, 102, 102, (12, 0), False)

    class Table(ProcessTable):
        identities = {row.pid: row for row in (root, namespace_init, escaped)}
        namespaces = {100: 10, 101: 20, 102: 20}
        children_by_pid = {100: (101,), 101: (102,), 102: ()}
        resident = {100: 1 << 20, 101: 1 << 20, 102: 2 * LIMIT}

        def identity(self, pid):
            return self.identities.get(pid)

        def children(self, pid):
            return self.children_by_pid.get(pid, ())

        def pid_namespace_inode(self, pid):
            return self.namespaces.get(pid)

        def resident_bytes(self, row):
            return self.resident.get(row.pid, 0)

        def snapshot(self):
            pytest.fail("private PID namespace should not need a full host scan")

    process = SimpleNamespace(pid=root.pid, returncode=None)
    process.poll = lambda: pytest.fail("memory monitor reaped the leader")
    monkeypatch.setattr(policy, "ProcessTable", Table)
    monkeypatch.setattr(policy, "recorded_launch_identity",
                        lambda _process: SimpleNamespace(pid=root.pid, start=root.start))
    supervisor = policy._ResidentMemorySupervisor(process, LIMIT)
    supervisor._stop_launch_unit = lambda: None

    supervisor._run()

    assert supervisor.family.private_pid_namespace_inode == 20
    assert 102 in supervisor.family.members
    assert process.memory_limit_killed is True
    assert process.memory_limit_reason == "aggregate resident-memory limit exceeded"


def test_memory_supervisor_periodically_scans_when_no_private_pid_namespace(monkeypatch):
    class Table:
        snapshots = 0

        def refresh_members(self, _members, **_kwargs):
            return {}

        def snapshot(self):
            self.snapshots += 1
            return {}

        def resident_memory_limit_breach(self, _live, _memory_bytes):
            return ""

    class Family:
        members = {}
        private_pid_namespace_inode = None

        def __init__(self, *_args):
            pass

        def update(self, _rows):
            return []

        def leader_reaped(self):
            pass

    process = SimpleNamespace(pid=123, returncode=0)
    monkeypatch.setattr(policy, "ProcessTable", Table)
    monkeypatch.setattr(policy, "Descendants", Family)
    monkeypatch.setattr(policy, "recorded_launch_identity", lambda _process: object())
    supervisor = policy._ResidentMemorySupervisor(process, LIMIT)
    supervisor._next_full_process_scan = time.monotonic() - 1

    supervisor._run()

    assert supervisor.table.snapshots == 1


def test_public_probe_launch_forwards_its_memory_budget(monkeypatch, tmp_path):
    import hashlib
    from contextlib import contextmanager

    import kernel_public_probe_execution as probe

    root = tmp_path / "workspace"
    output = tmp_path / "output"
    root.mkdir()
    output.mkdir()
    product = root / "module.py"
    product.write_text("value = 1\n")
    digest = "sha256:" + hashlib.sha256(product.read_bytes()).hexdigest()
    captured = {}

    @contextmanager
    def no_lock(*_args, **_kwargs):
        yield

    class Process:
        returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def fake_popen(*_args, **kwargs):
        captured.update(kwargs)
        return Process(), object()

    monkeypatch.setattr(probe, "locked_test_process", no_lock)
    monkeypatch.setattr(probe, "child_exited_unreaped", lambda _process: True)
    monkeypatch.setattr(probe, "kill_launch_unit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(probe, "recorded_launch_identity", lambda _process: object())
    monkeypatch.setattr(policy, "sandboxed_popen", fake_popen)

    result = probe.execute_probe(root, {"module.py": digest}, "def test_ok(): pass\n", output)

    assert result["exit_code"] == 0
    assert captured["memory_bytes"] == 1024 * 1024 * 1024


def test_witness_policy_receives_the_requested_memory_budget():
    from witness_runner import LocalWitnessRunner

    source = inspect.getsource(LocalWitnessRunner.run)
    assert "memory_bytes=budgets.memory_bytes or 512 * 1024 * 1024" in source


@pytest.mark.parametrize(
    ("system", "expected_calls"),
    [("linux", []), ("darwin", [(17, (LIMIT, LIMIT))])],
)
def test_witness_child_uses_parent_memory_policy_on_linux_and_keeps_darwin_limit(
        system, expected_calls):
    from witness_runner import CHILD_SOURCE

    child = ast.parse(CHILD_SOURCE)
    limits_function = next(
        node for node in child.body
        if isinstance(node, ast.FunctionDef) and node.name == "_limits")
    calls = []
    resource_module = SimpleNamespace(
        RLIMIT_AS=17,
        setrlimit=lambda kind, bounds: calls.append((kind, bounds)),
    )
    namespace = {
        "CONTROL": {"budgets": {"cpu_ms": None, "memory_bytes": LIMIT}},
        "resource": resource_module,
        "sys": SimpleNamespace(platform=system),
    }
    isolated = ast.Module(body=[limits_function], type_ignores=[])
    exec(compile(isolated, "witness-child-limits", "exec"), namespace)

    applied, unenforced = namespace["_limits"]()

    assert calls == expected_calls
    assert "memory_bytes" in applied
    assert "memory_bytes" not in unenforced


def test_generic_check_file_bytes_default_is_64_mib_and_overridable():
    assert policy.DEFAULT_FILE_BYTES == 64 << 20
    assert inspect.signature(policy.sandboxed_popen).parameters["file_bytes"].default == 64 << 20
    assert inspect.signature(policy.run_sandboxed).parameters["file_bytes"].default == 64 << 20


def test_resident_monitor_does_not_reap_process_group_leader(monkeypatch):
    class Table:
        def snapshot(self):
            return {}

        def resident_memory_limit_breach(self, _live, _memory_bytes):
            return ""

    class Family:
        def __init__(self, *_args):
            pass

        def update(self, _rows):
            return []

        def leader_reaped(self):
            pass

    class StopAfterOneSample:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _timeout):
            self.stopped = True

        def set(self):
            self.stopped = True

    process = SimpleNamespace(pid=123, returncode=None)
    process.poll = lambda: pytest.fail("memory monitor reaped the leader")
    monkeypatch.setattr(policy, "ProcessTable", Table)
    monkeypatch.setattr(policy, "Descendants", Family)
    monkeypatch.setattr(policy, "recorded_launch_identity", lambda _process: object())
    supervisor = policy._ResidentMemorySupervisor(process, LIMIT)
    supervisor._stopped = StopAfterOneSample()

    supervisor._run()

    assert supervisor.error is None


def _preexec(monkeypatch, *, cgroup=None, join_fails=False):
    calls = []
    monkeypatch.setattr(policy.platform, "system", lambda: "Linux")
    monkeypatch.setattr(policy.resource, "getrlimit", lambda _kind: (
        policy.resource.RLIM_INFINITY, policy.resource.RLIM_INFINITY))
    monkeypatch.setattr(policy.resource, "setrlimit", lambda kind, bounds: calls.append((kind, bounds)))
    if join_fails:
        write_text = Path.write_text

        def fail_join(path, *args, **kwargs):
            if path.name == "cgroup.procs":
                raise PermissionError("join denied")
            return write_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_join)
    policy._resource_preexec(30, LIMIT, 1 << 20, 128, 64, cgroup)()
    return calls


def _memory_group(path: Path) -> Path:
    path.mkdir()
    (path / "memory.max").write_text(str(LIMIT))
    (path / "memory.swap.max").write_text("0")
    return path


def test_linux_preexec_uses_data_limit_without_address_limit(monkeypatch):
    calls = _preexec(monkeypatch)
    kinds = [kind for kind, _bounds in calls]
    assert policy.resource.RLIMIT_AS not in kinds
    assert (policy.resource.RLIMIT_DATA, (LIMIT, LIMIT)) in calls


@pytest.mark.parametrize("join_fails", [False, True], ids=["joined", "join-fails"])
def test_linux_preexec_uses_cgroup_only_after_successful_join(monkeypatch, tmp_path, join_fails):
    group = _memory_group(tmp_path / "group")
    calls = _preexec(monkeypatch, cgroup=group, join_fails=join_fails)
    kinds = [kind for kind, _bounds in calls]
    assert policy.resource.RLIMIT_AS not in kinds
    if join_fails:
        assert (policy.resource.RLIMIT_DATA, (LIMIT, LIMIT)) in calls
    else:
        assert all(kind != policy.resource.RLIMIT_DATA for kind in kinds)
        assert (group / "cgroup.procs").read_text() == str(policy.os.getpid())


def test_command_cgroup_keeps_pids_and_uses_memory_fallback_without_leaf_pattern(
        monkeypatch, tmp_path):
    monkeypatch.setattr(sf_launch, "_current_cgroup", lambda: tmp_path)
    mkdir = Path.mkdir

    def fake_cgroup_files(path, *args, **kwargs):
        mkdir(path, *args, **kwargs)
        if path.parent == tmp_path:
            for name in ("pids.max", "memory.max", "memory.swap.max",
                         "memory.oom.group", "cgroup.procs"):
                (path / name).touch()

    monkeypatch.setattr(Path, "mkdir", fake_cgroup_files)
    group = sf_launch._command_cgroup("memory", 8, memory_bytes=LIMIT)
    assert group is not None
    assert (group / "pids.max").read_text() == "16"
    assert (group / "memory.max").read_text() == ""
    assert (group / "memory.swap.max").read_text() == ""
    assert (group / "memory.oom.group").read_text() == ""


def test_command_cgroup_keeps_pids_group_when_memory_max_is_unwritable(monkeypatch, tmp_path):
    monkeypatch.setattr(sf_launch, "_current_cgroup", lambda: tmp_path)
    mkdir = Path.mkdir

    def fake_cgroup_files(path, *args, **kwargs):
        mkdir(path, *args, **kwargs)
        if path.parent == tmp_path:
            (path / "pids.max").touch()
            (path / "memory.max").touch()

    write_text = Path.write_text

    def fail_memory(path, value, *args, **kwargs):
        if path.name == "memory.max":
            raise PermissionError("memory controller is not writable")
        return write_text(path, value, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fake_cgroup_files)
    monkeypatch.setattr(Path, "write_text", fail_memory)
    group = sf_launch._command_cgroup("pids-only", 8, memory_bytes=LIMIT)
    assert group is not None
    assert (group / "pids.max").read_text() == "16"
    assert not sf_launch._cgroup_memory_limit_configured(group, LIMIT)


def test_command_cgroup_fails_closed_if_controller_is_unavailable_and_cleanup_fails(
        monkeypatch, tmp_path):
    monkeypatch.setattr(sf_launch, "_current_cgroup", lambda: tmp_path)
    mkdir = Path.mkdir

    def no_pids_controller(path, *args, **kwargs):
        mkdir(path, *args, **kwargs)

    original_rmdir = Path.rmdir

    def fail_cleanup(path):
        if path.parent == tmp_path:
            raise PermissionError("delegated controller cleanup denied")
        return original_rmdir(path)

    monkeypatch.setattr(Path, "mkdir", no_pids_controller)
    monkeypatch.setattr(Path, "rmdir", fail_cleanup)
    assert sf_launch._command_cgroup("no-controller", 8, memory_bytes=LIMIT) is None


def test_darwin_preexec_keeps_tolerated_address_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(policy.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(policy.resource, "getrlimit", lambda _kind: (
        policy.resource.RLIM_INFINITY, policy.resource.RLIM_INFINITY))

    def set_limit(kind, bounds):
        calls.append(kind)
        if kind == policy.resource.RLIMIT_AS:
            raise OSError("best-effort Darwin limit")

    monkeypatch.setattr(policy.resource, "setrlimit", set_limit)
    policy._resource_preexec(30, LIMIT, 1 << 20, 128, 64)()
    assert policy.resource.RLIMIT_AS in calls
    assert policy.resource.RLIMIT_DATA not in calls


def test_sandboxed_result_reports_preexec_mechanism(monkeypatch, tmp_path):
    monkeypatch.setattr(policy.platform, "system", lambda: "Linux")
    monkeypatch.setattr(policy, "_command_cgroup", lambda _ident, _nproc, memory_bytes=None: None)
    monkeypatch.setattr(policy, "require_capabilities", lambda: policy.CapabilityReport(
        "Linux", "bwrap", True, True, True, True, True, True))
    result, _report = policy.sandboxed_popen(
        [sys.executable, "-c", "pass"], tmp_path,
        memory_bytes=LIMIT,
        isolation_builder=lambda command, _root, env, **_kwargs: (list(command), env),
    )
    try:
        assert result.wait(timeout=10) == 0
        assert getattr(result, "memory_limit_mechanism", None) == "rlimit_data+rss_supervisor"
    finally:
        result.stdout.close()
        result.stderr.close()


def test_policy_result_records_parent_confirmed_mechanism(monkeypatch, tmp_path):
    group = tmp_path / "group"
    group.mkdir()
    original_read_text = Path.read_text
    seen = []
    monkeypatch.setattr(policy.platform, "system", lambda: "Linux")
    monkeypatch.setattr(policy, "_command_cgroup",
                        lambda _ident, _nproc, memory_bytes=None: group)
    monkeypatch.setattr(policy, "sandboxed_popen", lambda *_args, **_kwargs: (
        SimpleNamespace(pid=123, returncode=0,
                         memory_limit_mechanism="cgroup_memory_max"), object()))
    monkeypatch.setattr(policy, "capture_launch_identity", lambda *_args, **_kwargs: None)

    def empty_cgroup_procs(path, *args, **kwargs):
        if path == group / "cgroup.procs":
            return ""
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", empty_cgroup_procs)
    monkeypatch.setattr(policy, "_collect",
                        lambda *_args, **_kwargs: ("ok", "", False, False, False))
    result = policy.run_sandboxed([TRUE], tmp_path)
    seen.append(result.memory_limit_mechanism)
    assert seen == ["cgroup_memory_max"]
    assert not group.exists()


def test_legacy_policy_result_mechanism_defaults_to_none():
    helper = getattr(policy, "policy_memory_limit_mechanism", None)
    assert callable(helper)
    assert helper(SimpleNamespace()) is None


def test_memory_mechanism_stays_private_and_out_of_model_receipt_shapes():
    from kernel_measurement import canonical_bytes
    from kernel_stateframe_runtime import _model_facing_check_evidence

    def build(mechanism=None, leaf_pattern=None):
        events = []
        journal = SimpleNamespace(
            emit=lambda kind, **fields: events.append({"kind": kind, **fields}))
        runtime = SimpleNamespace(
            owner=SimpleNamespace(current_step=1), revision="revision-1",
            journal=journal, test_receipts=[])
        receipt = record_check_output(
            runtime, selector="go test ./...", status="PASS", outcome="PASS",
            execution_id="stable-memory-check", exit_code=0,
            memory_limit_mechanism=mechanism,
            cgroup_leaf_pattern=leaf_pattern)
        model_evidence = _model_facing_check_evidence(runtime)
        return (runtime, receipt, model_evidence["public_evidence"],
                model_evidence["dependency_identity"], events)

    base, base_receipt, base_evidence, base_identity, _ = build()
    runtime, receipt, public_evidence, dependency_identity, events = build(
        "rlimit_data+rss_supervisor", "failed:foreign_process")

    assert canonical_bytes(receipt) == canonical_bytes(base_receipt)
    assert canonical_bytes(public_evidence) == canonical_bytes(base_evidence)
    assert dependency_identity == base_identity
    assert "memory_limit_mechanism" not in receipt
    assert "cgroup_leaf_pattern" not in receipt
    assert "memory_limit_mechanism" not in runtime.host_check_receipts[-1]
    assert "cgroup_leaf_pattern" not in runtime.host_check_receipts[-1]
    assert all("memory_limit_mechanism" not in row for row in public_evidence)
    private = next(iter(runtime._check_output_receipts.values()))
    assert private["memory_limit_mechanism"] == "rlimit_data+rss_supervisor"
    assert "cgroup_leaf_pattern" not in private
    event = next(row for row in events if row["kind"] == "check_receipt")
    assert event["memory_limit_mechanism"] == "rlimit_data+rss_supervisor"
    assert event["cgroup_leaf_pattern"] == "failed:foreign_process"


def _build_memory_identity_check(*, mechanism=None, leaf_pattern=None,
                                 memory_limit_killed=False,
                                 memory_limit_reason=None):
    from kernel_check_receipts import record_check_output
    from kernel_wire_truth_facts import _current_public_check

    selector = "tests/test_memory_limit_policy.py"
    binding = {"product_revision": "rev-1"}

    class Journal:
        task_id = "task-round6-memory-limit-golden"
        run_id = "ab" * 16
        check_receipt_count = 0
        check_execution_allocations = 0

        def __init__(self):
            self.events = []

        def emit(self, kind, **fields):
            row = {"kind": kind, **fields}
            self.events.append(row)
            return row

    journal = Journal()
    runtime = SimpleNamespace(
        journal=journal,
        owner=SimpleNamespace(current_step=7, f2p_ids=[selector]),
        test_receipts=[],
    )
    receipt = record_check_output(
        runtime, selector=selector, status="PASS", outcome="PASS",
        exit_code=0, counts={"collected": 1, "passed": 1},
        execution_complete=True, memory_limit_mechanism=mechanism,
        cgroup_leaf_pattern=leaf_pattern,
    )
    event = {
        "kind": "test_subprocess_completed",
        "producer": "test_subprocess",
        "selectors": [selector],
        "selector_scope": [selector],
        "exit_code": 0,
        "binding": binding,
        "after_binding": binding,
        "binding_stable": True,
        "sequence": 1,
        "execution_id": receipt["execution_id"],
    }
    if mechanism is not None:
        event["memory_limit_mechanism"] = mechanism
    if leaf_pattern is not None:
        event["cgroup_leaf_pattern"] = leaf_pattern
    if memory_limit_killed:
        event["memory_limit_killed"] = True
    if memory_limit_reason is not None:
        event["memory_limit_reason"] = memory_limit_reason
    runtime.test_receipts.append(event)
    journal.events.append(event)
    return runtime, receipt, event, _current_public_check(runtime, binding)


def test_memory_audit_fields_do_not_change_public_check_execution_ref():
    from kernel_stateframe_runtime import _model_facing_check_evidence

    base = _build_memory_identity_check()
    measured = _build_memory_identity_check(
        mechanism="rlimit_data+rss_supervisor",
        leaf_pattern="failed:foreign_process",
    )

    assert measured[1]["execution_id"] == base[1]["execution_id"]
    assert measured[1]["receipt_id"] == base[1]["receipt_id"]
    assert measured[3][1] == base[3][1]
    assert (_model_facing_check_evidence(measured[0])["dependency_identity"]
            == _model_facing_check_evidence(base[0])["dependency_identity"])
    assert measured[0]._check_output_receipts[measured[1]["receipt_id"]][
        "memory_limit_mechanism"] == "rlimit_data+rss_supervisor"
    check_event = next(row for row in measured[0].journal.events
                       if row["kind"] == "check_receipt")
    assert check_event["memory_limit_mechanism"] == "rlimit_data+rss_supervisor"
    assert check_event["cgroup_leaf_pattern"] == "failed:foreign_process"


def test_memory_kill_audit_fields_do_not_change_public_check_execution_ref():
    from kernel_stateframe_runtime import _model_facing_check_evidence

    base = _build_memory_identity_check()
    killed = _build_memory_identity_check(
        memory_limit_killed=True,
        memory_limit_reason="aggregate resident-memory limit exceeded",
    )

    assert killed[3][1] == base[3][1]
    assert (_model_facing_check_evidence(killed[0])["dependency_identity"]
            == _model_facing_check_evidence(base[0])["dependency_identity"])
    assert killed[2] in killed[0].journal.events
    assert killed[2]["memory_limit_killed"] is True
    assert killed[2]["memory_limit_reason"] == (
        "aggregate resident-memory limit exceeded")


def test_memory_identity_matches_base_algorithm_goldens():
    runtime, receipt, _event, public_check = _build_memory_identity_check()

    assert receipt["execution_id"] == "check-execution:656cb6232bcee864cf21"
    assert receipt["receipt_id"] == "check-receipt:c7657c678526de31ad5a2d91"
    assert public_check[1] == "c6d844c43ea05c7691bb"
    assert runtime._check_output_receipts[receipt["receipt_id"]]["receipt_id"] == (
        "check-receipt:c7657c678526de31ad5a2d91")


def test_command_cgroup_cleanup_retries_transient_rmdir_failures(
        monkeypatch, tmp_path):
    group = tmp_path / "group"
    group.mkdir()
    original_rmdir = Path.rmdir
    original_read_text = Path.read_text
    calls = 0
    sleeps = []

    def fail_twice(path):
        nonlocal calls
        if path == group and calls < 2:
            calls += 1
            raise OSError(16, "cgroup busy")
        return original_rmdir(path)

    def empty_cgroup_procs(path, *args, **kwargs):
        if path == group / "cgroup.procs":
            return ""
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "rmdir", fail_twice)
    monkeypatch.setattr(Path, "read_text", empty_cgroup_procs)
    monkeypatch.setattr(policy.time, "sleep", lambda delay: sleeps.append(delay))

    assert policy._remove_command_cgroup(group)
    assert calls == 2
    assert len(sleeps) == 2
    assert not group.exists()


def test_command_cgroup_cleanup_logs_unresolved_failure(monkeypatch, tmp_path, caplog):
    group = tmp_path / "group"
    group.mkdir()
    original_rmdir = Path.rmdir
    original_read_text = Path.read_text

    def always_fail(path):
        if path == group:
            raise PermissionError("cgroup cleanup denied")
        return original_rmdir(path)

    def empty_cgroup_procs(path, *args, **kwargs):
        if path == group / "cgroup.procs":
            return ""
        return original_read_text(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "rmdir", always_fail)
        scoped.setattr(Path, "read_text", empty_cgroup_procs)
        scoped.setattr(policy.time, "sleep", lambda _delay: None)
        with caplog.at_level("ERROR", logger=policy.__name__):
            assert not policy._remove_command_cgroup(group)
        assert "unable to remove per-command cgroup" in caplog.text
    original_rmdir(group)


class _FakeCommandCgroup:
    def __init__(self, *, rmdir_fails=False, populated=False):
        self.name = "fake-command"
        self.rmdir_fails = rmdir_fails
        self.populated = populated
        self.removed = False
        self.rmdir_calls = 0

    def __truediv__(self, name):
        assert name == "cgroup.procs"
        return SimpleNamespace(
            read_text=lambda **_kwargs: "812\n" if self.populated else "")

    def exists(self):
        return not self.removed

    def rmdir(self):
        self.rmdir_calls += 1
        if self.rmdir_fails or self.populated:
            raise PermissionError("fake cgroup cannot be removed")
        self.removed = True


@pytest.mark.parametrize("completion_path", ["check", "exec_job"])
@pytest.mark.parametrize("failure", ["rmdir", "populated"])
def test_failed_cgroup_cleanup_is_reported_by_completion_path(
        monkeypatch, tmp_path, completion_path, failure):
    cgroup = _FakeCommandCgroup(
        rmdir_fails=failure == "rmdir", populated=failure == "populated")
    monkeypatch.setattr(policy.time, "sleep", lambda _delay: None)

    if completion_path == "check":
        monkeypatch.setattr(policy.platform, "system", lambda: "Linux")
        monkeypatch.setattr(policy, "_command_cgroup",
                            lambda *_args, **_kwargs: cgroup)
        monkeypatch.setattr(policy, "sandboxed_popen", lambda *_args, **_kwargs: (
            SimpleNamespace(pid=123, returncode=0,
                            memory_limit_mechanism="cgroup_memory_max"), object()))
        monkeypatch.setattr(policy, "capture_launch_identity", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(policy, "_collect",
                            lambda *_args, **_kwargs: ("ok", "", False, False, False))

        detail, code = generic_check.check_result(
            tmp_path, {"test_command": [TRUE]})
        assert code == 127
        assert "ContainmentUnverified" in detail
        assert "cgroup cleanup failed" in detail
    else:
        managed = object.__new__(kernel_exec_supervisor.ManagedProcess)
        managed.process = SimpleNamespace(
            poll=lambda: 0, stdout=SimpleNamespace(close=lambda: None))
        managed.collector = SimpleNamespace(seal=lambda: None, total_bytes=0)
        managed.family = SimpleNamespace(signal_errors={}, signal_paths={})
        managed.cgroup = cgroup
        managed.cleanup = None
        managed.containment_failed = False
        managed.timed_out = False
        managed.limit_detail = ""
        managed.term_sent = set()
        managed.killed = set()
        managed.started = time.monotonic()
        managed.memory_limit_mechanism = "cgroup_memory_max"

        managed.complete(True, ())
        status = managed.status()

        assert status["finished"]
        assert status["cleanup"]["ok"] is False
        assert "cgroup cleanup" in status["cleanup"]["detail"]
        assert managed.cgroup is cgroup

    if failure == "populated":
        assert cgroup.rmdir_calls == 0
    else:
        assert cgroup.rmdir_calls >= 1


def test_exec_supervisor_uses_shared_bounded_cgroup_cleanup(monkeypatch, tmp_path):
    import kernel_exec_supervisor

    group = tmp_path / "group"
    group.mkdir()
    calls = []
    monkeypatch.setattr(
        kernel_exec_supervisor, "_remove_command_cgroup",
        lambda path: calls.append(path) or True, raising=False)
    process = object.__new__(kernel_exec_supervisor.ManagedProcess)
    process.cgroup = group

    process.remove_cgroup()

    assert calls == [group]
    assert process.cgroup is None


def test_exec_supervisor_scans_host_at_one_second_interval_with_private_pid_members(
        monkeypatch):
    import kernel_exec_supervisor

    row = SimpleNamespace(pid=123)

    class Table:
        def __init__(self):
            self.samples = []
            self.snapshots = 0

        def refresh_members(self, members, *, namespace_inode=None):
            self.samples.append((members, namespace_inode))
            return {row.pid: row}

        def snapshot(self):
            self.snapshots += 1
            return {}

    supervisor = object.__new__(kernel_exec_supervisor.Supervisor)
    supervisor.table = Table()
    supervisor._next_full_process_scan = time.monotonic() - 1
    job = SimpleNamespace(
        cleanup=None,
        family=SimpleNamespace(members={row.pid: row}, private_pid_namespace_inode=88),
    )
    supervisor.jobs = {"private": job}

    assert supervisor._sample_rows() == {row.pid: row}
    assert supervisor.table.samples == [({row.pid: row}, 88)]
    assert supervisor.table.snapshots == 1

    job.family.private_pid_namespace_inode = None
    supervisor._next_full_process_scan = time.monotonic() - 1
    supervisor._sample_rows()
    supervisor._sample_rows()
    assert supervisor.table.snapshots == 2


def test_generic_check_forwards_mechanism_to_runtime():
    import kernel_generic_check

    calls = []
    runtime = SimpleNamespace(test_observation=lambda *args, **kwargs: calls.append((args, kwargs)))
    kernel_generic_check._record(
        runtime, {}, ("go", "test", "./..."), 0, "ok", outcome="PASS",
        memory_limit_mechanism="rlimit_data+rss_supervisor",
        memory_limit_killed=True)
    assert calls[0][1]["memory_limit_mechanism"] == "rlimit_data+rss_supervisor"
    assert calls[0][1]["memory_limit_killed"] is True

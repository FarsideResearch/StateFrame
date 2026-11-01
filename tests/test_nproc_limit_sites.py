"""Prevent another uid-wide process limit from bypassing the relative policy."""
import ast
import io
from pathlib import Path
import sys

import pytest

import kernel_execution_policy as policy
import kernel_exec_supervisor as supervisor
import kernel_test_isolation as isolation
import sf_launch


SOURCE = Path(__file__).resolve().parents[1] / "src"
ENGINE = SOURCE / "stateframe" / "engine"
ALLOWED = (ENGINE / "sf_launch.py", "_set_relative_nproc_limit")


def _nproc_sites(path: Path, source: str):
    if "RLIMIT_NPROC" not in source:
        return
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id.endswith("SOURCE")
                        for target in node.targets)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                and "RLIMIT_NPROC" in node.value.value):
            yield from _nproc_sites(path, node.value.value)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr != "RLIMIT_NPROC":
            continue
        owner = node
        while owner in parents and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = parents[owner]
        yield path, getattr(owner, "name", None)


def _assert_single_site(sources):
    sites = [site for path, source in sources for site in _nproc_sites(path, source)]
    assert sites == [ALLOWED, ALLOWED], sites


def test_only_relative_helper_sets_nproc():
    _assert_single_site((path, path.read_text()) for path in sorted(SOURCE.rglob("*.py")))
    tree = ast.parse((ENGINE / "sf_launch.py").read_text())
    setters = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == "setrlimit"
               and node.args and any(isinstance(arg, ast.Attribute) and arg.attr == "RLIMIT_NPROC"
                                     for arg in ast.walk(node.args[0]))]
    assert len(setters) == 1


def test_planted_absolute_limit_is_caught():
    sources = [(path, path.read_text()) for path in sorted(SOURCE.rglob("*.py"))]
    sources.append((ENGINE / "planted_violation.py",
                    "def launch():\n    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))\n"))
    with pytest.raises(AssertionError, match="planted_violation"):
        _assert_single_site(sources)


def test_planted_launcher_list_limit_is_caught():
    sources = [(path, path.read_text()) for path in sorted(SOURCE.rglob("*.py"))]
    sources.append((ENGINE / "planted_violation.py",
                    "def launch():\n    limits.append((resource.RLIMIT_NPROC, 256))\n"))
    with pytest.raises(AssertionError, match="planted_violation"):
        _assert_single_site(sources)


def test_planted_embedded_runner_limit_is_caught():
    sources = [(path, path.read_text()) for path in sorted(SOURCE.rglob("*.py"))]
    sources.append((ENGINE / "planted_violation.py",
                    'RUNNER_SOURCE = """def run():\n'
                    '    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))\n"""\n'))
    with pytest.raises(AssertionError, match="planted_violation"):
        _assert_single_site(sources)


def test_outer_preexec_never_lowers_nproc_before_bwrap(monkeypatch):
    calls = []
    monkeypatch.setattr(policy.platform, "system", lambda: "Linux")
    monkeypatch.setattr(policy, "_user_task_count", lambda: 400)
    monkeypatch.setattr(policy.resource, "getrlimit",
                        lambda kind: (policy.resource.RLIM_INFINITY, policy.resource.RLIM_INFINITY))
    monkeypatch.setattr(policy.resource, "setrlimit", lambda kind, limits: calls.append((kind, limits)))
    policy._resource_preexec(30, 512 * 1024 * 1024, 1024 * 1024, 128, 64)()
    assert all(kind != policy.resource.RLIMIT_NPROC for kind, _ in calls)


def test_outer_policy_passes_relative_ceiling_to_builder(monkeypatch, tmp_path):
    received = {}
    monkeypatch.setattr(policy.platform, "system", lambda: "Linux")
    monkeypatch.setattr(policy, "_user_task_count", lambda: 400)
    monkeypatch.setattr(policy, "require_capabilities", lambda: object())

    def build(command, root, env, **kwargs):
        received.update(kwargs)
        return command, env

    policy._prepared(["/bin/true"], tmp_path, {}, None, False, 64, isolation_builder=build)
    assert received["nproc_ceiling"] == 592


def test_outer_command_installs_nproc_after_bwrap(monkeypatch, tmp_path):
    monkeypatch.setattr(isolation, "backend", lambda: "/usr/bin/bwrap")
    monkeypatch.setattr(isolation.platform, "system", lambda: "Linux")
    command, _ = isolation.isolated_command(["/bin/echo", "ok"], tmp_path, {},
                                             nproc_ceiling=592)
    boundary = command.index("--")
    assert command[boundary + 1:boundary + 5] == [
        sys.executable, "-I", "/tmp/sf_launch.py", "--nproc-ceiling"]
    assert command[boundary + 5:] == ["592", "/bin/echo", "ok"]
    assert ["--ro-bind", str(ENGINE / "sf_launch.py"), "/tmp/sf_launch.py"] == command[
        command.index(str(ENGINE / "sf_launch.py")) - 1:
        command.index(str(ENGINE / "sf_launch.py")) + 2]


def test_task_count_uses_real_uid_and_threads(monkeypatch):
    monkeypatch.setattr(sf_launch.os, "getuid", lambda: 1001)
    monkeypatch.setattr(sf_launch.os, "listdir", lambda path: ["11", "12", "13"])
    statuses = {
        "/proc/11/status": b"Uid:\t1001\t0\t0\t0\nThreads:\t401\n",
        "/proc/12/status": b"Uid:\t1002\t1001\t1001\t1001\nThreads:\t50\n",
        "/proc/13/status": b"Uid:\t1001\t1002\t1002\t1002\nThreads:\t2\n",
    }
    monkeypatch.setattr(sf_launch, "open", lambda path, *args, **kwargs: io.BytesIO(statuses[path]), raising=False)
    assert sf_launch._user_task_count() == 403


def test_task_count_skips_unreadable_foreign_status(monkeypatch):
    monkeypatch.setattr(sf_launch.os, "getuid", lambda: 1001)
    monkeypatch.setattr(sf_launch.os, "listdir", lambda path: ["11", "12"])
    def status(path, *args, **kwargs):
        if path == "/proc/12/status":
            raise PermissionError(path)  # hidepid=1 foreign process
        return io.BytesIO(b"Uid:\t1001\t1001\t1001\t1001\nThreads:\t3\n")
    monkeypatch.setattr(sf_launch, "open", status, raising=False)
    assert sf_launch._user_task_count() == 3


def test_task_count_skips_unavailable_proc_listing(monkeypatch):
    def unavailable(path):
        raise PermissionError(path)

    monkeypatch.setattr(sf_launch.os, "listdir", unavailable)
    assert sf_launch._user_task_count() == 0


def test_task_count_ignores_non_ascii_name(monkeypatch):
    monkeypatch.setattr(sf_launch.os, "getuid", lambda: 1001)
    monkeypatch.setattr(sf_launch.os, "listdir", lambda path: ["11"])
    status = b"Name:\tbad\xffname\nUid:\t1001\t1001\t1001\t1001\nThreads:\t2\n"
    monkeypatch.setattr(sf_launch, "open", lambda path, *args, **kwargs: io.BytesIO(status), raising=False)
    assert sf_launch._user_task_count() == 2


def test_command_cgroup_uses_writable_pids_max_or_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor, "_current_cgroup", lambda: tmp_path)
    assert supervisor._command_cgroup("missing", 8) is None
    assert not list(tmp_path.iterdir())

    mkdir = Path.mkdir
    def with_pids_file(path, *args, **kwargs):
        mkdir(path, *args, **kwargs)
        if path.parent == tmp_path:
            (path / "pids.max").touch()
    monkeypatch.setattr(Path, "mkdir", with_pids_file)
    group = supervisor._command_cgroup("delegated", 8)
    assert group is not None
    assert (group / "pids.max").read_text() == "16"

"""Offline sandboxed commands must see the public CA trust store.

Debian's bundled pip wheel points certifi at /etc/ssl/certs/ca-certificates.crt and loads it when
imported, so `ensurepip` inside an offline sandbox probe failed and no run environment was built.
Sandbox probes then tried a bare `python` that does not exist. Regression for RUNENV-ENSUREPIP.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import pytest

import kernel_exec_capability
import kernel_mutate_exec
import kernel_test_isolation as isolation
from kernel_exec_capability import shell_capability

_CA_BUNDLE = Path("/etc/ssl/certs/ca-certificates.crt")
linux_ca = pytest.mark.skipif(sys.platform == "darwin" or not _CA_BUNDLE.is_file(),
                              reason="needs a Linux system CA bundle at /etc/ssl/certs")


def _cap(tmp_path, network):
    for name in ("workspace", "env", "home"):
        (tmp_path / name).mkdir(exist_ok=True)
    return shell_capability(workspace=tmp_path / "workspace", run_env_dir=tmp_path / "env",
                            scratch_home=tmp_path / "home", network=network, writable=False)


def _ro_bind_pairs(command):
    end = command.index("--")
    return [(Path(command[index + 1]), Path(command[index + 2]))
            for index, value in enumerate(command[:end])
            if value in {"--ro-bind", "--ro-bind-try"}]


def _stub_linux_sandbox(monkeypatch):
    monkeypatch.setattr(isolation, "sandbox_executable", lambda name: name)
    monkeypatch.setattr(isolation, "_host_mount_points", lambda: frozenset({Path("/")}))
    monkeypatch.delenv("STATEFRAME_TASK_NETWORK", raising=False)


def _minimal_roots(roots):
    resolved = {path.resolve(strict=True) for path in roots}
    minimal = set()
    for path in sorted(resolved, key=lambda item: (len(item.parts), str(item))):
        if not any(parent.is_dir() and path.is_relative_to(parent) for parent in minimal):
            minimal.add(path)
    return minimal


def _mutation_cap(tmp_path):
    stage = tmp_path / "stage"
    mounts = {}
    for name in ("boot", "unit", "ctl", "in", "resp"):
        (stage / name).mkdir(parents=True)
        mounts[name] = str(stage / name)
    cap = SimpleNamespace(read_roots=tuple(Path(value) for value in mounts.values())
                          + tuple(isolation.public_ca_read_roots()),
                          workspace=Path(mounts["unit"]), workspace_writable=False,
                          write_roots=(), purpose="mutation", network="none", env={})
    return cap, mounts


@linux_ca
def test_check_sandbox_binds_only_the_public_ca_roots_read_only(tmp_path, monkeypatch):
    _stub_linux_sandbox(monkeypatch)
    command, _ = isolation.isolated_command([sys.executable], tmp_path, {"PATH": "/usr/bin"})

    minimal_roots = _minimal_roots(kernel_exec_capability.public_ca_read_roots())
    etc_binds = {destination for source, destination in _ro_bind_pairs(command)
                 if destination == Path("/etc") or destination.is_relative_to(Path("/etc"))}
    loader_cache = {Path("/etc/ld.so.cache")} if Path("/etc/ld.so.cache").is_file() else set()

    assert etc_binds == minimal_roots | loader_cache
    assert "--unshare-all" in command and "--share-net" not in command
    assert Path("/etc") not in etc_binds
    assert not any(path == Path("/etc/ssl/private") or
                   Path("/etc/ssl/private").is_relative_to(path) or
                   path.is_relative_to(Path("/etc/ssl/private")) for path in etc_binds)


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_check_sandbox_with_no_public_ca_roots_keeps_existing_etc_mounts(tmp_path, monkeypatch):
    _stub_linux_sandbox(monkeypatch)
    monkeypatch.setattr(isolation, "public_ca_read_roots", lambda: (), raising=False)
    command, _ = isolation.isolated_command([sys.executable], tmp_path, {"PATH": "/usr/bin"})

    etc_binds = {destination for source, destination in _ro_bind_pairs(command)
                 if destination == Path("/etc") or destination.is_relative_to(Path("/etc"))}
    loader_cache = {Path("/etc/ld.so.cache")} if Path("/etc/ld.so.cache").is_file() else set()
    assert etc_binds == loader_cache


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_network_enabled_check_keeps_existing_mounts_and_adds_public_ca(tmp_path, monkeypatch):
    _stub_linux_sandbox(monkeypatch)
    monkeypatch.setenv("STATEFRAME_TASK_NETWORK", "container")
    command, _ = isolation.isolated_command([sys.executable], tmp_path, {"PATH": "/usr/bin"})

    ca_roots = _minimal_roots(kernel_exec_capability.public_ca_read_roots())
    etc_binds = {destination for source, destination in _ro_bind_pairs(command)
                 if destination == Path("/etc") or destination.is_relative_to(Path("/etc"))}
    loader_cache = {Path("/etc/ld.so.cache")} if Path("/etc/ld.so.cache").is_file() else set()
    resolver = {Path(path) for path in ("/etc/hosts", "/etc/resolv.conf") if Path(path).exists()}

    assert etc_binds == ca_roots | loader_cache | resolver
    assert "--share-net" in command
    assert Path("/etc") not in etc_binds


@linux_ca
def test_mutation_sandbox_binds_public_ca_roots_read_only(tmp_path):
    cap, mounts = _mutation_cap(tmp_path)

    command, _ = isolation._mutation_command(["true"], cap, "sandbox", mounts, False)

    minimal_roots = _minimal_roots(kernel_exec_capability.public_ca_read_roots())
    authorized_roots = {path.resolve(strict=True) for path in cap.read_roots}
    etc_binds = {destination for source, destination in _ro_bind_pairs(command)
                 if destination == Path("/etc") or destination.is_relative_to(Path("/etc"))}
    loader_cache = {Path("/etc/ld.so.cache")} if Path("/etc/ld.so.cache").is_file() else set()
    assert etc_binds == minimal_roots | loader_cache
    assert minimal_roots <= authorized_roots
    assert "--unshare-all" in command and "--share-net" not in command
    assert Path("/etc") not in etc_binds
    assert not any(path == Path("/etc/ssl/private") or
                   Path("/etc/ssl/private").is_relative_to(path) or
                   path.is_relative_to(Path("/etc/ssl/private")) for path in etc_binds)


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux mutation sandbox")
def test_mutation_sandbox_with_no_public_ca_roots_keeps_existing_etc_mounts(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation, "public_ca_read_roots", lambda: (), raising=False)
    cap, mounts = _mutation_cap(tmp_path)
    command, _ = isolation._mutation_command(["true"], cap, "sandbox", mounts, False)

    etc_binds = {destination for source, destination in _ro_bind_pairs(command)
                 if destination == Path("/etc") or destination.is_relative_to(Path("/etc"))}
    loader_cache = {Path("/etc/ld.so.cache")} if Path("/etc/ld.so.cache").is_file() else set()
    assert etc_binds == loader_cache
    assert "--unshare-all" in command and "--share-net" not in command


@linux_ca
def test_ensurepip_succeeds_inside_offline_check_sandbox(tmp_path):
    try:
        isolation.backend()
    except RuntimeError:
        pytest.skip("Linux sandbox is unavailable")
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages",
                    str(venv)], check=True)
    python = venv / "bin" / "python"
    command, env = isolation.isolated_command(
        [str(python), "-I", "-m", "ensurepip", "--default-pip"], tmp_path,
        {"PATH": str(python.parent)}, generic_check=True)

    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@linux_ca
def test_tls_context_loads_public_roots_in_network_enabled_check(tmp_path, monkeypatch):
    try:
        isolation.backend()
    except RuntimeError:
        pytest.skip("Linux sandbox is unavailable")
    monkeypatch.setenv("STATEFRAME_TASK_NETWORK", "container")
    command, env = isolation.isolated_command(
        [sys.executable, "-I", "-c",
         "import ssl; assert ssl.create_default_context().cert_store_stats()['x509_ca'] > 0"],
        tmp_path, {"PATH": str(Path(sys.executable).parent)}, generic_check=True)

    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@linux_ca
@pytest.mark.parametrize("network", ["none", "full"])
def test_offline_and_online_capabilities_admit_public_ca_store(tmp_path, network):
    roots = set(_cap(tmp_path, network).read_roots)
    # The bundle must be readable: some admitted root is the bundle or one of its ancestors.
    assert any(root == _CA_BUNDLE or root in _CA_BUNDLE.parents for root in roots)


@linux_ca
def test_private_keys_and_rest_of_etc_stay_hidden_offline(tmp_path):
    roots = _cap(tmp_path, "none").read_roots
    assert Path("/etc") not in roots and Path("/etc/ssl") not in roots
    assert not any(root == Path("/etc/ssl/private") or Path("/etc/ssl/private") in root.parents
                   for root in roots)
    helper = getattr(kernel_exec_capability, "public_ca_read_roots", tuple)
    assert all(Path("/etc/ssl/private") not in path.parents for path in helper())


@linux_ca
@pytest.mark.skipif(shutil.which("bwrap") is None, reason="needs the bwrap sandbox")
def test_ensurepip_succeeds_inside_offline_probe():
    from kernel_safe_probe import run_probe

    base = Path(tempfile.mkdtemp(prefix="ensurepip-ca-"))
    try:
        venv = base / "v"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages",
                        str(venv)], check=True)
        with tempfile.TemporaryDirectory(dir=str(base)) as workspace:
            result = run_probe([str(venv / "bin" / "python3"), "-m", "ensurepip", "--default-pip"],
                               Path(workspace), timeout=120, write_roots=(venv,))
        output = result.output.decode(errors="replace")
        assert result.exit_code == 0, output[-2000:]
        assert "FileNotFoundError" not in output
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _synthetic_ca_paths(monkeypatch, *paths, targets=None):
    monkeypatch.setattr(kernel_exec_capability, "_PUBLIC_CA_PATHS", tuple(map(str, paths)))
    if targets is not None:
        monkeypatch.setattr(kernel_exec_capability, "_PUBLIC_CA_TARGETS", tuple(targets))


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_ca_directory_symlink_to_home_is_not_admitted(tmp_path, monkeypatch):
    private = tmp_path / "home" / "private-certs"
    private.mkdir(parents=True)
    alias = tmp_path / "etc" / "ssl" / "certs"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(private, target_is_directory=True)
    _synthetic_ca_paths(monkeypatch, alias)

    assert kernel_exec_capability.public_ca_read_roots() == ()
    assert isolation._public_ca_mount_roots() == ((), ())


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux mutation sandbox")
def test_mutation_sandbox_does_not_mount_a_ca_symlink_target_in_home(tmp_path, monkeypatch):
    private = tmp_path / "home" / "private-certs"
    private.mkdir(parents=True)
    alias = tmp_path / "etc" / "ssl" / "certs"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(private, target_is_directory=True)
    _synthetic_ca_paths(monkeypatch, alias)
    cap, mounts = _mutation_cap(tmp_path)

    command, _ = isolation._mutation_command(["true"], cap, "sandbox", mounts, False)

    assert all(source != private and destination != private
               for source, destination in _ro_bind_pairs(command))
    assert not any(value == str(alias) for index, value in enumerate(command[:-1])
                   if index and command[index - 1] == "--symlink")


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_relative_chained_ca_symlink_to_home_is_not_admitted(tmp_path, monkeypatch):
    private = tmp_path / "home" / "private-certs"
    private.mkdir(parents=True)
    bridge = tmp_path / "links" / "bridge"
    bridge.parent.mkdir()
    bridge.symlink_to(os.path.relpath(private, bridge.parent), target_is_directory=True)
    alias = tmp_path / "etc" / "ssl" / "certs"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(os.path.relpath(bridge, alias.parent), target_is_directory=True)
    _synthetic_ca_paths(monkeypatch, alias)

    assert kernel_exec_capability.public_ca_read_roots() == ()


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_ca_bundle_symlink_to_private_file_is_not_admitted(tmp_path, monkeypatch):
    private = tmp_path / "home" / "secret.pem"
    private.parent.mkdir()
    private.write_text("synthetic private material")
    alias = tmp_path / "etc" / "ssl" / "cert.pem"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(private)
    _synthetic_ca_paths(monkeypatch, alias)

    assert kernel_exec_capability.public_ca_read_roots() == ()


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_dangling_ca_symlink_is_ignored(tmp_path, monkeypatch):
    alias = tmp_path / "etc" / "ssl" / "certs"
    alias.parent.mkdir(parents=True)
    alias.symlink_to("missing-target", target_is_directory=True)
    _synthetic_ca_paths(monkeypatch, alias)

    assert kernel_exec_capability.public_ca_read_roots() == ()
    assert isolation._public_ca_mount_roots() == ((), ())


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_missing_ca_root_keeps_the_previous_check_mounts(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    missing = tmp_path / "missing" / "certs"
    _synthetic_ca_paths(monkeypatch, missing, targets=(missing,))
    command, env = isolation.isolated_command([sys.executable], workspace, {"PATH": "/usr/bin"})
    with monkeypatch.context() as baseline:
        baseline.setattr(isolation, "public_ca_read_roots", lambda: ())
        previous_command, previous_env = isolation.isolated_command(
            [sys.executable], workspace, {"PATH": "/usr/bin"})

    assert (command, env) == (previous_command, previous_env)


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_unreadable_ca_root_is_ignored(tmp_path, monkeypatch):
    certificate = tmp_path / "cert.pem"
    certificate.write_text("synthetic public material")
    _synthetic_ca_paths(monkeypatch, certificate, targets=(certificate,))
    monkeypatch.setattr(kernel_exec_capability, "_public_ca_readable", lambda _path, _status: False)

    assert kernel_exec_capability.public_ca_read_roots() == ()


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_public_ca_file_root_is_supported_as_a_single_read_only_mount(tmp_path, monkeypatch):
    certificate = tmp_path / "ca" / "bundle.pem"
    certificate.parent.mkdir()
    certificate.write_text("synthetic public material")
    _synthetic_ca_paths(monkeypatch, certificate, targets=(certificate,))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    roots, mounts = isolation._public_ca_mount_roots()
    command, _ = isolation.isolated_command([sys.executable], workspace, {"PATH": "/usr/bin"})

    assert roots == (certificate,)
    assert mounts == (certificate,)
    assert ("--ro-bind-try", str(certificate), str(certificate)) in [
        tuple(command[index:index + 3]) for index, value in enumerate(command) if value == "--ro-bind-try"]


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_ca_mount_failure_is_optional_and_does_not_change_the_child_argv(tmp_path, monkeypatch):
    certificate = tmp_path / "ca.pem"
    certificate.write_text("synthetic public material")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _stub_linux_sandbox(monkeypatch)
    monkeypatch.setattr(isolation, "public_ca_read_roots", lambda: (certificate,))
    child = [sys.executable, "-I", "-c", "pass"]

    command, _ = isolation.isolated_command(child, workspace, {"PATH": "/usr/bin"})
    certificate.unlink()
    separator = command.index("--")

    assert command[separator + 1:] == child
    assert any(value == "--ro-bind-try" and command[index + 1] == str(certificate)
               for index, value in enumerate(command[:separator]))


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_overlapping_ca_roots_bind_only_the_parent_before_workspace(tmp_path, monkeypatch):
    ca_dir = tmp_path / "ca"
    ca_dir.mkdir()
    ca_file = ca_dir / "bundle.pem"
    ca_file.write_text("synthetic public material")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _stub_linux_sandbox(monkeypatch)
    monkeypatch.setattr(isolation, "public_ca_read_roots", lambda: (ca_file, ca_dir))

    command, _ = isolation.isolated_command([sys.executable], workspace, {"PATH": "/usr/bin"})
    ca_binds = [(command[index], command[index + 1], command[index + 2])
                for index, value in enumerate(command[:command.index("--")])
                if value in {"--ro-bind", "--ro-bind-try"}
                and Path(command[index + 2]).is_relative_to(ca_dir)]

    assert ca_binds == [("--ro-bind-try", str(ca_dir), str(ca_dir))]
    assert command.index("--ro-bind-try") < command.index("--bind", command.index("--ro-bind-try"))


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_private_ca_keys_and_other_home_paths_are_not_added(tmp_path, monkeypatch):
    _stub_linux_sandbox(monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with monkeypatch.context() as baseline:
        baseline.setattr(isolation, "public_ca_read_roots", lambda: ())
        previous_command, _ = isolation.isolated_command(
            [sys.executable], workspace, {"PATH": "/usr/bin"})
    command, _ = isolation.isolated_command([sys.executable], workspace, {"PATH": "/usr/bin"})
    binds = _ro_bind_pairs(command)
    private_keys = Path("/etc/ssl/private")
    home = Path.home().resolve()
    previous_binds = _ro_bind_pairs(previous_command)

    assert all(destination != private_keys and not destination.is_relative_to(private_keys)
               for _, destination in binds)
    assert {(source, destination) for source, destination in binds
            if source == home or source.is_relative_to(home)
            or destination == home or destination.is_relative_to(home)} == {
                (source, destination) for source, destination in previous_binds
                if source == home or source.is_relative_to(home)
                or destination == home or destination.is_relative_to(home)}


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_user_ca_environment_cannot_mount_a_private_file(tmp_path, monkeypatch):
    _stub_linux_sandbox(monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    private = tmp_path / "home" / "secret.pem"
    private.parent.mkdir()
    private.write_text("synthetic private material")
    command, clean = isolation.isolated_command(
        [sys.executable], workspace,
        {"PATH": "/usr/bin", "SSL_CERT_FILE": str(private)})

    assert "SSL_CERT_FILE" not in clean
    assert all(source != private and destination != private
               for source, destination in _ro_bind_pairs(command))


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_nested_command_keeps_the_outer_check_network_disabled(tmp_path, monkeypatch):
    _stub_linux_sandbox(monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    child = ["bwrap", "--unshare-net", "--", "true"]
    command, _ = isolation.isolated_command(child, workspace, {"PATH": "/usr/bin"})

    assert "--unshare-all" in command
    assert "--share-net" not in command
    assert command[command.index("--") + 1:] == child


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
@pytest.mark.skipif(shutil.which("bwrap") is None, reason="requires an installed Linux sandbox")
def test_nested_sandbox_starts_inside_the_check_namespace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executable = isolation.sandbox_executable("bwrap")
    if executable is None:
        pytest.skip("the Linux sandbox executable is unavailable")
    child = [executable, "--die-with-parent", "--unshare-all", "--new-session",
             "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
             "--", "/bin/true"]
    command, clean = isolation.isolated_command(
        child, workspace, {"PATH": str(Path(sys.executable).parent)}, generic_check=True)

    result = subprocess.run(command, cwd=workspace, env=clean, capture_output=True,
                            text=True, check=False)

    if result.returncode and any(message in result.stderr.lower() for message in
                                 ("operation not permitted", "creating new namespace",
                                  "no permissions to create a new namespace")):
        pytest.skip("the host does not allow nested namespaces")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_missing_bwrap_fails_closed_without_an_unsandboxed_command(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation, "sandbox_executable", lambda _name: None)

    with pytest.raises(RuntimeError, match="no unsandboxed fallback"):
        isolation.isolated_command([sys.executable], tmp_path, {"PATH": "/usr/bin"})


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_available_bwrap_is_selected_as_the_check_backend(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(isolation, "sandbox_executable", lambda _name: "/usr/bin/bwrap")

    command, _ = isolation.isolated_command([sys.executable], workspace, {"PATH": "/usr/bin"})

    assert command[0] == "/usr/bin/bwrap"
    assert "--unshare-all" in command


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
@pytest.mark.skipif(shutil.which("bwrap") is None, reason="requires an installed Linux sandbox")
def test_missing_optional_ca_mount_source_does_not_stop_the_sandbox(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    certificate = tmp_path / "optional-ca.pem"
    certificate.write_text("synthetic public material")
    monkeypatch.setattr(isolation, "public_ca_read_roots", lambda: (certificate,))
    command, clean = isolation.isolated_command(
        [sys.executable, "-I", "-c", "print('sandbox-started')"],
        workspace, {"PATH": str(Path(sys.executable).parent)}, generic_check=True)
    certificate.unlink()

    result = subprocess.run(command, cwd=workspace, env=clean, capture_output=True,
                            text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "sandbox-started"


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_private_key_locations_are_never_public_ca_targets():
    private_paths = (Path("/etc/ssl/private"), Path("/etc/pki/private"),
                     Path("/etc/pki/tls/private"), Path("/usr/lib/ssl/private"))

    assert all(not kernel_exec_capability._public_ca_target(path) for path in private_paths)


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_child_argv_environment_and_output_bytes_are_preserved(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    launcher = tmp_path / "bubblewrap-stub"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "offset = sys.argv.index('--')\n"
        "os.execvpe(sys.argv[offset + 1], sys.argv[offset + 1:], os.environ)\n")
    launcher.chmod(0o755)
    _stub_linux_sandbox(monkeypatch)
    monkeypatch.setattr(isolation, "sandbox_executable", lambda _name: str(launcher))
    expected_stdout = bytes.fromhex("00636865636b2d6f7574707574ff")
    expected_stderr = b"check-error"
    source = (
        "import sys; "
        "sys.stdout.buffer.write(bytes.fromhex('00636865636b2d6f7574707574ff')); "
        "sys.stderr.buffer.write(b'check-error')"
    )
    child = [sys.executable, "-I", "-c", source]
    command, clean = isolation.isolated_command(
        child, workspace, {"PATH": "/usr/bin", "LC_ALL": "C"})

    result = subprocess.run(command, cwd=workspace, env=clean, capture_output=True, check=False)

    assert command[command.index("--") + 1:] == child
    assert clean == {
        "PATH": "/usr/bin", "LC_ALL": "C", "HOME": str(workspace / ".test-tmp"),
        "TMPDIR": str(workspace / ".test-tmp"), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    assert result.returncode == 0
    assert result.stdout == expected_stdout
    assert result.stderr == expected_stderr


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_ca_symlink_retargeted_during_discovery_cannot_admit_private_data(tmp_path, monkeypatch):
    public = tmp_path / "public" / "certs"
    public.mkdir(parents=True)
    private = tmp_path / "home" / "private-certs"
    private.mkdir(parents=True)
    alias = tmp_path / "etc" / "ssl" / "certs"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(public, target_is_directory=True)
    _synthetic_ca_paths(monkeypatch, alias, targets=(public,))
    original_hops = kernel_exec_capability.symlink_hops

    def retarget_after_validation(path):
        alias.unlink()
        alias.symlink_to(private, target_is_directory=True)
        return original_hops(path)

    monkeypatch.setattr(kernel_exec_capability, "symlink_hops", retarget_after_validation)

    roots = kernel_exec_capability.public_ca_read_roots()

    assert private not in roots
    assert isolation._public_ca_mount_roots() == ((), ())


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_ca_symlink_cycle_is_ignored(tmp_path, monkeypatch):
    alias = tmp_path / "etc" / "ssl" / "certs"
    alias.parent.mkdir(parents=True)
    alias.symlink_to("certs", target_is_directory=True)
    _synthetic_ca_paths(monkeypatch, alias)

    assert kernel_exec_capability.public_ca_read_roots() == ()
    assert isolation._public_ca_mount_roots() == ((), ())


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_special_file_ca_root_is_ignored(tmp_path, monkeypatch):
    special = tmp_path / "ca-pipe"
    os.mkfifo(special)
    _synthetic_ca_paths(monkeypatch, special, targets=(special,))

    assert kernel_exec_capability.public_ca_read_roots() == ()


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_ca_target_prefix_collision_is_not_a_public_store():
    assert not kernel_exec_capability._public_ca_target(
        Path("/etc/ssl/certs-private/ca-bundle.crt"))


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_case_folded_private_key_directory_link_is_rejected(tmp_path, monkeypatch):
    private = tmp_path / "etc" / "ssl" / "PRIVATE" / "ca.key"
    private.parent.mkdir(parents=True)
    private.write_text("synthetic private key")
    alias = tmp_path / "etc" / "ssl" / "cert.pem"
    alias.symlink_to(os.path.relpath(private, alias.parent))
    _synthetic_ca_paths(monkeypatch, alias, targets=(alias.parent,))

    assert kernel_exec_capability.public_ca_read_roots() == ()


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_single_file_ca_root_does_not_bind_its_parent(tmp_path, monkeypatch):
    store = tmp_path / "public-store"
    store.mkdir()
    certificate = store / "bundle.pem"
    certificate.write_text("synthetic public material")
    alias = tmp_path / "etc" / "ssl" / "cert.pem"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(os.path.relpath(certificate, alias.parent))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _synthetic_ca_paths(monkeypatch, alias, targets=(store,))
    _stub_linux_sandbox(monkeypatch)

    roots, mounts = isolation._public_ca_mount_roots()
    command, _ = isolation.isolated_command([sys.executable], workspace, {"PATH": "/usr/bin"})
    ca_binds = [(command[index], Path(command[index + 1]), Path(command[index + 2]))
                for index, value in enumerate(command[:command.index("--")])
                if value == "--ro-bind-try"]

    assert set(roots) == {alias, certificate}
    assert mounts == (certificate,)
    assert ca_binds == [("--ro-bind-try", certificate, certificate)]


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_ca_link_scan_error_does_not_stop_check_command_construction(tmp_path, monkeypatch):
    certificate = tmp_path / "ca.pem"
    certificate.write_text("synthetic public material")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    child = [sys.executable, "-I", "-c", "pass"]
    _stub_linux_sandbox(monkeypatch)
    monkeypatch.setattr(isolation, "public_ca_read_roots", lambda: (certificate,))

    def fail_ca_links(paths):
        if certificate in paths:
            raise OSError("synthetic changing CA root")
        return {}

    monkeypatch.setattr(isolation, "executable_symlinks", fail_ca_links)
    command, _ = isolation.isolated_command(child, workspace, {"PATH": "/usr/bin"})

    assert command[command.index("--") + 1:] == child
    assert any(value == "--ro-bind-try" and command[index + 1] == str(certificate)
               for index, value in enumerate(command[:command.index("--")]))


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_ca_environment_is_not_forwarded_and_network_stays_off(tmp_path, monkeypatch):
    _stub_linux_sandbox(monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    private = tmp_path / "home" / "private-ca.pem"
    private.parent.mkdir()
    private.write_text("synthetic private material")
    child = [sys.executable, "-I", "-c", "pass"]

    command, clean = isolation.isolated_command(
        child, workspace,
        {"PATH": "/usr/bin", "SSL_CERT_FILE": str(private),
         "SSL_CERT_DIR": str(private.parent), "CURL_CA_BUNDLE": str(private)})

    assert not {"SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE"} & clean.keys()
    assert all(source != private and destination != private
               for source, destination in _ro_bind_pairs(command))
    assert "--unshare-all" in command and "--share-net" not in command
    assert command[command.index("--") + 1:] == child


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
def test_ca_root_disappearing_before_mount_resolution_is_dropped(tmp_path, monkeypatch):
    certificate = tmp_path / "ca.pem"
    certificate.write_text("synthetic public material")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _stub_linux_sandbox(monkeypatch)

    def disappear():
        certificate.unlink(missing_ok=True)
        return (certificate,)

    monkeypatch.setattr(isolation, "public_ca_read_roots", disappear)
    assert isolation._public_ca_mount_roots() == ((), ())

    command, _ = isolation.isolated_command([sys.executable], workspace, {"PATH": "/usr/bin"})
    with monkeypatch.context() as baseline:
        baseline.setattr(isolation, "public_ca_read_roots", lambda: ())
        previous, _ = isolation.isolated_command(
            [sys.executable], workspace, {"PATH": "/usr/bin"})

    assert command == previous


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
@pytest.mark.parametrize("location", ["home", "repository", "workspace", "private"])
def test_ca_symlink_into_user_or_private_paths_is_not_admitted(tmp_path, monkeypatch, location):
    target = tmp_path / location / "certificates"
    target.mkdir(parents=True)
    alias = tmp_path / "etc" / "ssl" / "certs"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(target, target_is_directory=True)
    _synthetic_ca_paths(monkeypatch, alias)

    assert kernel_exec_capability.public_ca_read_roots() == ()


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
@pytest.mark.parametrize("layout", ["debian", "fedora", "opensuse", "nixos", "guix", "alpine", "arch"])
def test_released_ca_roots_remain_admitted_for_distribution_layouts(tmp_path, monkeypatch, layout):
    """The released shell admitted each existing public root after following its links."""
    tree = tmp_path / layout
    if layout == "debian":
        target = tree / "etc" / "ssl" / "certs"
        source = target
    elif layout == "fedora":
        target = tree / "etc" / "pki" / "ca-trust" / "extracted"
        source = target
    elif layout == "opensuse":
        target = tree / "var" / "lib" / "ca-certificates" / "pem"
        source = tree / "etc" / "ssl" / "certs"
    elif layout == "nixos":
        target = tree / "nix" / "store" / "hash-nss-cacert" / "etc" / "ssl" / "certs"
        source = tree / "etc" / "ssl" / "certs"
    elif layout == "guix":
        target = tree / "gnu" / "store" / "hash-nss-certs" / "etc" / "ssl" / "certs"
        source = tree / "etc" / "ssl" / "certs"
    elif layout == "alpine":
        target = tree / "etc" / "ssl" / "cert.pem"
        source = target
    else:
        target = tree / "etc" / "ssl" / "certs"
        source = target
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".pem":
        target.write_text("synthetic public CA material")
    else:
        target.mkdir(exist_ok=True)
        (target / "ca.pem").write_text("synthetic public CA material")
    source.parent.mkdir(parents=True, exist_ok=True)
    if source != target:
        source.symlink_to(os.path.relpath(target, source.parent), target_is_directory=True)
    _synthetic_ca_paths(monkeypatch, source, targets=(target,))

    released_roots = kernel_exec_capability.symlink_hops(source)
    workspace, run_env, scratch_home = (tree / name for name in ("workspace", "env", "home"))
    for directory in (workspace, run_env, scratch_home):
        directory.mkdir(parents=True, exist_ok=True)
    cap = shell_capability(workspace=workspace, run_env_dir=run_env,
                           scratch_home=scratch_home, network="none", writable=False,
                           include_path_executable_roots=False)

    assert kernel_exec_capability.public_ca_read_roots() == released_roots
    assert set(released_roots) <= set(cap.read_roots)


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
@pytest.mark.parametrize("path", [
    Path("/var/lib/ca-certificates/pem"),
    Path("/nix/store/hash-nss-cacert/etc/ssl/certs"),
    Path("/gnu/store/hash-nss-certs/etc/ssl/certs"),
])
def test_public_ca_policy_keeps_released_external_store_targets(path):
    assert kernel_exec_capability._public_ca_target(path)


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux check sandbox")
@pytest.mark.parametrize("path", [Path("/nix/store"), Path("/gnu/store")])
def test_public_ca_policy_does_not_admit_an_entire_package_store(path):
    assert not kernel_exec_capability._public_ca_target(path)


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux mutation sandbox")
def test_store_ca_mount_is_the_resolved_target_not_the_store(tmp_path, monkeypatch):
    store = tmp_path / "nix" / "store"
    target = store / "hash-nss-cacert" / "etc" / "ssl" / "certs"
    target.mkdir(parents=True)
    (target / "ca.pem").write_text("synthetic public CA material")
    alias = tmp_path / "etc" / "ssl" / "certs"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(os.path.relpath(target, alias.parent), target_is_directory=True)
    _synthetic_ca_paths(monkeypatch, alias)
    monkeypatch.setattr(kernel_exec_capability, "_PUBLIC_CA_STORE_ROOTS", (store,))

    roots, mounts = isolation._public_ca_mount_roots()
    cap, stage_mounts = _mutation_cap(tmp_path)
    command, _ = isolation._mutation_command(
        ["true"], cap, "sandbox", stage_mounts, False)
    store_binds = [(command[index], Path(command[index + 1]), Path(command[index + 2]))
                   for index, value in enumerate(command[:command.index("--")])
                   if value in {"--ro-bind", "--ro-bind-try"}
                   and (Path(command[index + 1]) == store
                        or Path(command[index + 1]).is_relative_to(store)
                        or Path(command[index + 2]) == store
                        or Path(command[index + 2]).is_relative_to(store))]

    assert set(roots) == {alias, target}
    assert mounts == (target,)
    assert store not in mounts
    assert store_binds == [("--ro-bind-try", target, target)]


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux mutation sandbox")
def test_mutation_capability_includes_public_ca_roots(tmp_path, monkeypatch):
    certificate = tmp_path / "public-ca.pem"
    certificate.write_text("synthetic public CA material")
    unit = tmp_path / "unit"
    unit.mkdir()
    monkeypatch.setattr(kernel_mutate_exec, "public_ca_read_roots", lambda: (certificate,),
                        raising=False)

    cap = kernel_mutate_exec._fixed_env_capability(
        "mutation", unit, (), "none", kernel_exec_capability.ExecLimits())

    assert certificate in cap.read_roots


@pytest.mark.skipif(sys.platform == "darwin", reason="requires the Linux mutation sandbox")
def test_mutation_command_does_not_recreate_ca_hop_inside_bound_root(tmp_path, monkeypatch):
    outer = tmp_path / "installed"
    alias = outer / "etc" / "ssl" / "certs"
    alias.parent.mkdir(parents=True)
    public = tmp_path / "public" / "certs"
    public.mkdir(parents=True)
    (public / "ca.pem").write_text("synthetic public CA material")
    alias.symlink_to(os.path.relpath(public, alias.parent), target_is_directory=True)
    _synthetic_ca_paths(monkeypatch, alias, targets=(public,))
    cap, mounts = _mutation_cap(tmp_path)
    cap.read_roots = (*cap.read_roots, outer)

    command, _ = isolation._mutation_command(["true"], cap, "sandbox", mounts, False)

    symlink_destinations = {Path(command[index + 2])
                            for index, value in enumerate(command[:command.index("--")])
                            if value == "--symlink"}
    assert alias not in symlink_destinations
    assert (outer, outer) in _ro_bind_pairs(command)

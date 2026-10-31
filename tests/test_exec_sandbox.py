from dataclasses import replace
import os
import subprocess
import sys
import time
import uuid

import pytest

from kernel_exec_capability import shell_capability
from kernel_exec_session import ExecSession

# Sandboxed targets run the real interpreter: the venv symlink lies outside the declared read roots.
PYTHON = os.path.realpath(sys.executable)

pytestmark = pytest.mark.host_sandbox


@pytest.fixture
def setup(tmp_path):
    for name in ("workspace", "env", "home", "run", "fake-home/.ssh"):
        (tmp_path / name).mkdir(parents=True)
    cap = shell_capability(workspace=tmp_path / "workspace", run_env_dir=tmp_path / "env",
                           scratch_home=tmp_path / "home", network="none", writable=True)
    return tmp_path, cap


def run_python(session, cap, name, text):
    path = cap.workspace / name
    path.write_text(text)
    return session.run([PYTHON, "-I", str(path)], cap)


def test_filesystem_boundaries_and_fake_home(setup):
    root, cap = setup
    secret = root / "fake-home/.ssh/id_test"
    secret.write_text("secret-never-readable")
    outside = root / "outside"
    cap = replace(cap, env={**cap.env, "HOME": str(root / "fake-home")})
    with ExecSession(root / "run") as session:
        result = run_python(session, cap, "fs.py", f'''from pathlib import Path
for path in [{str(secret)!r}]:
    try: Path(path).read_bytes()
    except OSError: print("read-denied")
    else: raise SystemExit("secret exposed")
try: Path({str(outside)!r}).write_text("bad")
except OSError: print("write-denied")
else: raise SystemExit("outside write succeeded")
Path("allowed").write_text("ok")
''')
        assert result.exit_code == 0, result.output
        assert result.output == b"read-denied\nwrite-denied\n"
        assert not outside.exists()
        readonly = replace(cap, workspace_writable=False)
        result = session.run(["/bin/sh", "-c", "echo bad > forbidden"], readonly)
        assert result.exit_code != 0 and not (cap.workspace / "forbidden").exists()


@pytest.mark.parametrize("network", ["none", "loopback"])
def test_network_policy_blocks_outbound(setup, network):
    root, cap = setup
    cap = replace(cap, network=network)
    with ExecSession(root / "run") as session:
        result = run_python(session, cap, "network.py", '''import socket
try:
    s = socket.socket()
    s.settimeout(0.5)
    s.connect(("1.1.1.1", 443))
except OSError: print("denied")
else: raise SystemExit("outbound connected")
''')
    assert result.exit_code == 0 and result.output == b"denied\n", result.output


def test_none_blocks_loopback(setup):
    root, cap = setup
    with ExecSession(root / "run") as session:
        result = run_python(session, cap, "loop.py", '''import socket
try:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    client = socket.socket()
    client.settimeout(0.2)
    client.connect(server.getsockname())
except OSError: print("denied")
else: raise SystemExit("loopback connected")
''')
    assert result.exit_code == 0 and result.output == b"denied\n", result.output


def test_loopback_server_and_later_client_share_session(setup):
    root, cap = setup
    cap = replace(cap, network="loopback", workspace_writable=False)
    server = cap.workspace / "server.py"
    server.write_text('''import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
s.listen()
print(s.getsockname()[1], flush=True)
conn, _ = s.accept()
conn.sendall(b"session-connected")
conn.close()
''')
    with ExecSession(root / "run") as session:
        job = session.spawn([PYTHON, "-I", str(server)], replace(cap, purpose="job"))
        data = b""
        deadline = time.monotonic() + 5
        while b"\n" not in data and time.monotonic() < deadline:
            data += job.read_available()
            time.sleep(0.02)
        port = int(data.strip())
        result = run_python(session, cap, "client.py", f'''import socket
s = socket.create_connection(("127.0.0.1", {port}), timeout=1)
print(s.recv(100).decode())
''')
        assert result.exit_code == 0 and result.output == b"session-connected\n", result.output
        assert job.wait(2) == 0


@pytest.mark.parametrize("purpose", ["shell", "job", "custom_tool"])
@pytest.mark.host_sandbox
def test_venv_pip_through_outside_symlink_chain(setup, purpose):
    import venv

    root, original = setup
    env = root / "env"
    venv.EnvBuilder(with_pip=True, symlinks=True).create(env)
    outside = root / "outside-interpreters"
    outside.mkdir()
    secret = outside / "unrelated-secret"
    secret.write_text("must remain unreadable")
    hop = outside / "python-hop"
    hop.symlink_to(sys.executable)
    directory_alias = root / "interpreter-directory"
    directory_alias.symlink_to(outside, target_is_directory=True)
    python = env / "bin/python"
    python.unlink()  # Replace only this test's newly generated venv link.
    python.symlink_to(directory_alias / "python-hop")
    cap = shell_capability(workspace=original.workspace, run_env_dir=env,
                           scratch_home=root / "home", network="none", writable=True)
    cap = replace(cap, purpose=purpose, workspace_writable=purpose != "job")
    with ExecSession(root / "run") as session:
        result = session.run(["/bin/sh", "-c", "pip --version"], cap)
        denied = session.run(["/bin/cat", str(secret)], cap)
    assert result.cleanup.ok
    assert result.exit_code == 0, result.output
    assert result.output.startswith(b"pip ") and str(env).encode() in result.output
    assert denied.cleanup.ok and denied.exit_code != 0
    assert b"must remain unreadable" not in denied.output


def _installer_pids():
    listing = subprocess.run(["/bin/ps", "-axo", "pid=,comm="], capture_output=True, text=True).stdout
    return {line.split()[0] for line in listing.splitlines() if "Install Command Line Developer Tools" in line}


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS developer-tool shims")
def test_developer_tool_shims_run_without_triggering_installer(setup):
    # Regression: a hidden /var/db/xcode_select_link made /usr/bin/make open the CLT install
    # dialog on the user's desktop from inside the sandbox.
    root, cap = setup
    before = _installer_pids()
    with ExecSession(root / "run") as session:
        result = session.run(["/bin/sh", "-c", "/usr/bin/make --version && /usr/bin/git --version"], cap)
    time.sleep(1.0)
    assert result.exit_code == 0, result.output
    assert b"GNU Make" in result.output and b"git version" in result.output
    assert b"couldn't create cache file" not in result.output
    assert _installer_pids() <= before, "sandboxed shim launched the Command Line Tools installer"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS developer-tool cache")
def test_xcrun_cache_exception_cannot_write_siblings_or_follow_links(setup):
    root, cap = setup
    from kernel_test_isolation import _darwin_user_temp_dir
    directory = _darwin_user_temp_dir()
    suffix = uuid.uuid4().hex
    cache = directory / ('xcrun_db-stateframe-' + suffix)
    sibling = directory / ('stateframe-denied-' + suffix)
    link = directory / ('xcrun_db-stateframe-link-' + suffix)
    secret = root / 'outside-cache'
    secret.write_text('untouched')
    link.symlink_to(secret)
    try:
        code = (f'from pathlib import Path\nPath({str(cache)!r}).write_text("cache")\n'
                f'for path in [{str(sibling)!r}, {str(link)!r}]:\n'
                ' try:\n  Path(path).write_text("ESCAPE")\n'
                ' except PermissionError:\n  pass\n'
                ' else:\n  raise AssertionError("cache exception escaped")\n')
        with ExecSession(root / 'run') as session:
            result = run_python(session, cap, 'cache_probe.py', code)
        assert result.exit_code == 0, result.output
        assert cache.read_text() == 'cache'
        assert not sibling.exists()
        assert secret.read_text() == 'untouched'
    finally:
        cache.unlink(missing_ok=True)
        sibling.unlink(missing_ok=True)
        link.unlink()


def test_selected_python_extensions_load_with_readonly_runtime_libraries(setup):
    root, cap = setup
    with ExecSession(root / 'run') as session:
        result = session.run([PYTHON, '-I', '-c',
                              'import ssl, zlib, ctypes; print("runtime libraries loaded")'], cap)
    assert result.exit_code == 0, result.output
    assert b'runtime libraries loaded' in result.output

from dataclasses import replace
import os
import sys
import time

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

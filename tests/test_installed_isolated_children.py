"""Catch imports that work in pytest but fail in an installed isolated child."""

import ast
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
# Direct script targets of -I launchers in request_bytes, outer_checks.run,
# kernel_exec_session, kernel_test_isolation, and kernel_exec_supervisor.
ISOLATED_SCRIPTS = (
    "stateframe/engine/kernel_bounded_http.py",
    "stateframe/outer_checks.py",
    "stateframe/engine/kernel_exec_supervisor.py",
    "stateframe/engine/sf_launch.py",
)


def test_bounded_http_child_uses_only_standard_library_imports():
    source = ROOT / "src/stateframe/engine/kernel_bounded_http.py"
    imports = []
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    assert set(imports) <= sys.stdlib_module_names | {"__future__"}


def _run(argv: list[str], *, cwd: Path, input_data: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return subprocess.run(argv, cwd=cwd, env=env, input=input_data,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=60, check=False)


@pytest.mark.host_sandbox
def test_installed_wheel_direct_isolated_children(tmp_path):
    """Build the release wheel and exercise child imports without checkout paths."""
    backend_path = ROOT / "_build/sf_backend.py"
    spec = importlib.util.spec_from_file_location("sf_backend", backend_path)
    assert spec is not None and spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    wheel = tmp_path / backend.build_wheel(str(tmp_path))

    venv = tmp_path / "venv"
    created = _run([sys.executable, "-I", "-m", "venv", str(venv)], cwd=tmp_path)
    assert created.returncode == 0, created.stderr.decode(errors="replace")
    python = venv / "bin/python"
    installed = _run([str(python), "-I", "-m", "pip", "install", "--no-index",
                      "--no-deps", "--disable-pip-version-check", str(wheel)], cwd=tmp_path)
    assert installed.returncode == 0, installed.stderr.decode(errors="replace")

    located = _run([str(python), "-I", "-B", "-c",
                    "import importlib.util, pathlib; "
                    "print(pathlib.Path(importlib.util.find_spec("
                    "'stateframe.engine.kernel_bounded_http').origin).parent)"],
                   cwd=tmp_path)
    assert located.returncode == 0, located.stderr.decode(errors="replace")
    engine = Path(located.stdout.decode().strip())
    assert engine.is_relative_to(venv)
    package = engine.parent

    # An import probe executes each script's top level without entering its CLI.
    # -I must not place either the checkout or script directory on sys.path.
    probe = ("import pathlib, runpy, sys; "
             "script = pathlib.Path(sys.argv[1]); "
             "assert str(script.parent) not in sys.path; "
             "assert not any(p.startswith(sys.argv[2]) for p in sys.path); "
             "runpy.run_path(str(script), run_name='__isolated_import_probe__'); "
             "print('started')")
    for relative in ISOLATED_SCRIPTS:
        script = package.parent / relative
        assert script.is_file(), relative
        result = _run([str(python), "-I", "-B", "-c", probe,
                       str(script), str(ROOT)], cwd=tmp_path)
        assert result.returncode == 0, (relative, result.stderr.decode(errors="replace"))
        assert result.stdout == b"started\n", relative

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):  # noqa: N802
            self.server.received = self.rfile.read(int(self.headers["Content-Length"]))
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        wire = b'{"model":"test","messages":[]}'
        packet = json.dumps({"wire": wire.decode(),
                             "endpoint": f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                             "key": "", "deadline": time.monotonic() + 30,
                             "socket_timeout": 5, "gzip": False}).encode()
        result = _run([str(python), "-I", "-B", str(engine / "kernel_bounded_http.py"),
                       "--http-child"], cwd=tmp_path, input_data=packet)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert result.stdout == b'{"ok":true}'
        assert server.received == wire
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

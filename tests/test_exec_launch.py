from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from kernel_exec_capability import ExecLimits

LAUNCH = Path(__file__).resolve().parents[1] / "src/stateframe/engine/sf_launch.py"


def launch(tmp_path, argv, env=None, stdin=None):
    spec = {"argv": argv, "cwd": str(tmp_path), "env": env or {}, "limits": asdict(ExecLimits()),
            "memory_supervised": True}
    return subprocess.run([sys.executable, "-I", str(LAUNCH), json.dumps(spec)],
                          input=stdin, stdin=subprocess.DEVNULL if stdin is None else None,
                          capture_output=True, timeout=10)


def test_launcher_complete_environment(tmp_path):
    result = launch(tmp_path, ["/usr/bin/env"], {"EXACT": "only-this"})
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"EXACT=only-this\n"


def test_launcher_stdin_closed(tmp_path):
    result = launch(tmp_path, ["/bin/cat"])
    assert result.returncode == 0 and result.stdout == b"", result.stderr


def test_launcher_session_umask_limits_and_fds(tmp_path):
    script = tmp_path / "inspect.py"
    script.write_text('''import json, os, resource
print(json.dumps({"session": os.getsid(0) == os.getpid(), "umask": os.umask(0),
"cpu": resource.getrlimit(resource.RLIMIT_CPU)[0], "nproc": resource.getrlimit(resource.RLIMIT_NPROC)[0],
"nofile": resource.getrlimit(resource.RLIMIT_NOFILE)[0]}))
''')
    result = launch(tmp_path, [sys.executable, "-I", str(script)])
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["session"] and data["umask"] == 0o077
    assert data["cpu"] == 300 and data["nproc"] <= 256 and data["nofile"] <= 1024


def test_launcher_missing_executable_reports_failure(tmp_path):
    result = launch(tmp_path, ["/does/not/exist"])
    assert result.returncode == 125
    assert b"execution launcher failed" in result.stderr


def test_launcher_closes_inherited_descriptor(tmp_path):
    import os
    descriptor = os.open(tmp_path / "secret", os.O_CREAT | os.O_RDWR, 0o600)
    script = tmp_path / "fd.py"
    script.write_text(f'''import os
try: os.fstat({descriptor})
except OSError: print("closed")
else: raise SystemExit("descriptor leaked")
''')
    spec = {"argv": [sys.executable, "-I", str(script)], "cwd": str(tmp_path),
            "env": {}, "limits": asdict(ExecLimits()), "memory_supervised": True}
    try:
        result = subprocess.run([sys.executable, "-I", str(LAUNCH), json.dumps(spec)],
                                pass_fds=(descriptor,), capture_output=True, timeout=5)
        assert result.returncode == 0 and result.stdout == b"closed\n", result.stderr
    finally:
        os.close(descriptor)


def test_launcher_file_limit(tmp_path):
    script = tmp_path / "write.py"
    script.write_text('from pathlib import Path\nPath("large").write_bytes(b"x" * 131072)\n')
    limits = asdict(ExecLimits(file_bytes=65536))
    spec = {"argv": [sys.executable, "-I", str(script)], "cwd": str(tmp_path),
            "env": {}, "limits": limits, "memory_supervised": True}
    result = subprocess.run([sys.executable, "-I", str(LAUNCH), json.dumps(spec)],
                            capture_output=True, timeout=5)
    assert result.returncode != 0
    assert (tmp_path / "large").stat().st_size <= 65536


def test_macos_launcher_requires_memory_supervision(tmp_path):
    if sys.platform != "darwin":
        return
    spec = {"argv": ["/bin/true"], "cwd": str(tmp_path), "env": {}, "limits": asdict(ExecLimits())}
    result = subprocess.run([sys.executable, "-I", str(LAUNCH), json.dumps(spec)],
                            capture_output=True, timeout=5)
    assert result.returncode == 125
    assert b"supervised resident-memory accounting" in result.stderr

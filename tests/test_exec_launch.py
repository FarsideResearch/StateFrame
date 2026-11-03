from dataclasses import asdict
import json
from pathlib import Path
import resource
import subprocess
import sys

import pytest

from kernel_exec_capability import ExecLimits
import sf_launch

LAUNCH = Path(__file__).resolve().parents[1] / "src/stateframe/engine/sf_launch.py"
_TASK_DRIFT_SLACK = 64  # Linux uid task counts can change between launcher and child sampling.


def _nproc_bounds(t0, t1, soft, budget, slack):
    lo, hi = sorted((t0, t1))
    if soft == resource.RLIM_INFINITY:
        return lo + budget - slack, hi + budget + slack
    return min(soft, lo + budget - slack), min(soft, hi + budget + slack)


@pytest.mark.parametrize(
    ("t0", "t1", "soft", "expected", "launched_nproc", "legacy_cap"),
    [
        pytest.param(1070, 1070, 1064, (1064, 1064), 1064, None,
                     id="soft-cap-below-live-task-count"),
        pytest.param(100, 180, 4096, (420, 628), 484, None,
                     id="task-count-drift-up"),
        pytest.param(180, 100, 4096, (420, 628), 564, None,
                     id="task-count-drift-down"),
        pytest.param(100, 120, resource.RLIM_INFINITY, (420, 568), 484, None,
                     id="infinite-soft-limit"),
        pytest.param(1000, 1000, 4096, (1320, 1448), 1384, 256,
                     id="absolute-256-cap-is-below-lower-bound"),
    ],
)
def test_nproc_bounds(t0, t1, soft, expected, launched_nproc, legacy_cap):
    budget = ExecLimits().nproc + sf_launch._NPROC_HEADROOM
    lower, upper = _nproc_bounds(t0, t1, soft, budget, _TASK_DRIFT_SLACK)

    assert (lower, upper) == expected
    assert lower <= launched_nproc <= upper
    if legacy_cap is not None:
        assert legacy_cap < lower


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
    script.write_text('''import json, os, resource, sys
sys.path.insert(0, %r)
from sf_launch import _user_task_count
print(json.dumps({"session": os.getsid(0) == os.getpid(), "umask": os.umask(0),
"cpu": resource.getrlimit(resource.RLIMIT_CPU)[0], "nproc": resource.getrlimit(resource.RLIMIT_NPROC)[0],
"nofile": resource.getrlimit(resource.RLIMIT_NOFILE)[0],
"tasks": _user_task_count() if sys.platform.startswith("linux") else None}))
''' % str(LAUNCH.parent))
    t0 = sf_launch._user_task_count() if sys.platform.startswith("linux") else None
    result = launch(tmp_path, [sys.executable, "-I", str(script)])
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["session"] and data["umask"] == 0o077
    assert data["cpu"] == 300 and data["nofile"] <= 1024
    inherited_soft = resource.getrlimit(resource.RLIMIT_NPROC)[0]
    if sys.platform.startswith("linux"):
        budget = ExecLimits().nproc + sf_launch._NPROC_HEADROOM
        lower, upper = _nproc_bounds(t0, data["tasks"], inherited_soft, budget,
                                     _TASK_DRIFT_SLACK)
        assert lower <= data["nproc"] <= upper
        if inherited_soft != resource.RLIM_INFINITY:
            assert data["nproc"] <= inherited_soft
    else:
        assert data["nproc"] == inherited_soft


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

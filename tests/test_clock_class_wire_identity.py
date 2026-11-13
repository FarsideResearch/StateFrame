"""No wall-clock value may reach the model: same seed, different clock, identical request bytes.

Class killer for the whole request, not one renderer. Two scripted runs share one identity seed
and one set of paths, but the second one runs under a different clock: ``time.time`` shifted by
weeks, ``time.monotonic``/``time.perf_counter`` running at twice the rate (so every elapsed-time
measurement differs, not just its offset) and a local time zone 26 hours away. Every provider
request body the two runs produce must be byte-identical, request by request. A difference is a
clock leak in model-facing text: it breaks same-seed determinism and the cached prompt prefix.

The engine child receives an allowlisted environment, so the clock shim travels through the
``--python`` interpreter: a wrapper that puts a ``sitecustomize`` on ``PYTHONPATH`` and executes
the real interpreter. Both captures use the same wrapper path, rewritten between them.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from _timeouts import scaled
from stateframe.scripted_endpoint import Script, serve
from tools.compare_no_mcp_requests import write_launcher


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox

# Model-facing fields that may legitimately differ between two same-seed runs under a different
# clock. Empty on purpose: the wire is clock-independent. Add nothing here without a determinism
# argument that survives review.
TIME_DEPENDENT_MODEL_FIELDS: frozenset[str] = frozenset()

_SHIM = '''\
"""Test-only clock shim: one shared base, scaled rate, shifted wall clock (see the killer test)."""
import os as _os
import time as _time

_BASE = float(_os.environ["CLOCK_SHIM_BASE"])
_SCALE = float(_os.environ["CLOCK_SHIM_SCALE"])
_SHIFT = float(_os.environ["CLOCK_SHIM_SHIFT"])
_monotonic, _perf_counter, _wall = _time.monotonic, _time.perf_counter, _time.time


def monotonic():
    return _BASE + (_monotonic() - _BASE) * _SCALE


def perf_counter():
    return _BASE + (_perf_counter() - _BASE) * _SCALE


def wall():
    return _wall() + _SHIFT


_time.monotonic, _time.perf_counter, _time.time = monotonic, perf_counter, wall
_time.monotonic_ns = lambda: int(monotonic() * 1e9)
_time.perf_counter_ns = lambda: int(perf_counter() * 1e9)
_time.time_ns = lambda: int(wall() * 1e9)
'''

_FIELD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^;\\\n\"]*)")


def _install_clock(python_dir: Path, shim_dir: Path, *, base: float, scale: float,
                   shift: float, tz: str) -> Path:
    """Rewrite the interpreter wrapper in place so both captures see one interpreter path."""
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "sitecustomize.py").write_text(_SHIM, encoding="utf-8")
    python_dir.mkdir(parents=True, exist_ok=True)
    wrapper = python_dir / "python3"
    wrapper.write_text(
        "#!/bin/sh\n"
        f"export PYTHONPATH={shim_dir}${{PYTHONPATH:+:$PYTHONPATH}}\n"
        f"export CLOCK_SHIM_BASE={base!r} CLOCK_SHIM_SCALE={scale!r} CLOCK_SHIM_SHIFT={shift!r}\n"
        f"export TZ={tz}\n"
        f"exec {sys.executable} \"$@\"\n", encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper


def _capture(work: Path, python: Path, *, scale: float, shift: float, tz: str,
             base: float) -> tuple[bytes, ...]:
    repo, run, home = work / "repo", work / "run", work / "home"
    for stale in (repo, run, home):
        if stale.exists():
            shutil.rmtree(stale)
    shutil.copytree(ROOT / "examples" / "discount_bug", repo)
    home.mkdir()
    launcher = work / "fixed-seed-run.py"
    write_launcher(launcher)
    script = Script(json.loads((ROOT / "examples" / "discount_bug.script.json").read_text()))
    server = serve(script)
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("STATEFRAME_", "REACT_", "TZ"))}
    environment.update({
        "PYTHONPATH": os.pathsep.join((str(python.parent.parent / "shim"), str(ROOT / "src"))),
        "CLOCK_SHIM_BASE": repr(base), "CLOCK_SHIM_SCALE": repr(scale),
        "CLOCK_SHIM_SHIFT": repr(shift), "TZ": tz,
        "HOME": str(home), "STATEFRAME_HOME": str(home / ".stateframe"),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
    })
    try:
        result = subprocess.run(
            [sys.executable, str(launcher), "run", "--repo", str(repo),
             "--task", "Complete the example task and run its checks.",
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--model", "scripted", "--run-dir", str(run), "--python", str(python),
             "--test-python", sys.executable,
             "--test-cmd", "python -m unittest discover -s tests -q",
             # A wall limit puts the countdown on the limits line; the shim spends it twice as fast.
             "--timeout", "1800", "--max-turns", "32", "--quiet"],
            env=environment, capture_output=True, text=True, timeout=scaled(900))
        assert result.returncode in (0, 1), result.stdout + result.stderr
        script.assert_consumed()
        requests = sorted((run / "wire").glob("[0-9][0-9][0-9][0-9].request.json"))
        assert requests, result.stdout + result.stderr
        return tuple(path.read_bytes() for path in requests)
    finally:
        server.shutdown()
        server.server_close()


def _differing_fields(first: bytes, second: bytes) -> dict[str, tuple[str, str]]:
    """Every ``key=value`` token whose value differs, keyed by field name (first difference each)."""
    left = {match[1]: match[2] for match in _FIELD.finditer(first.decode("utf-8", "replace"))}
    right = {match[1]: match[2] for match in _FIELD.finditer(second.decode("utf-8", "replace"))}
    return {key: (left.get(key, "<absent>"), right.get(key, "<absent>"))
            for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)}


def _first_difference(first: bytes, second: bytes, context: int = 160) -> str:
    offset = next((i for i, (a, b) in enumerate(zip(first, second)) if a != b),
                  min(len(first), len(second)))
    start = max(0, offset - context // 2)
    return (f"byte {offset}:\n  A: {first[start:offset + context]!r}\n"
            f"  B: {second[start:offset + context]!r}")


def test_same_seed_requests_are_identical_under_a_different_clock(tmp_path):
    base = time.monotonic()
    python_dir, shim_dir = tmp_path / "python" / "bin", tmp_path / "python" / "shim"
    python = _install_clock(python_dir, shim_dir, base=base, scale=1.0, shift=0.0, tz="Etc/GMT+12")
    baseline = _capture(tmp_path, python, scale=1.0, shift=0.0, tz="Etc/GMT+12", base=base)
    shutil.move(tmp_path / "run", tmp_path / "run-baseline")
    _install_clock(python_dir, shim_dir, base=base, scale=2.0, shift=37 * 86400.0, tz="Etc/GMT-14")
    shifted = _capture(tmp_path, python, scale=2.0, shift=37 * 86400.0, tz="Etc/GMT-14", base=base)

    assert len(shifted) == len(baseline), (len(baseline), len(shifted))
    problems = []
    for index, (first, second) in enumerate(zip(baseline, shifted), start=1):
        if first == second:
            continue
        fields = {key: value for key, value in _differing_fields(first, second).items()
                  if key not in TIME_DEPENDENT_MODEL_FIELDS}
        if not fields and first != second:
            fields = {"<no key=value token>": ("", "")}
        problems.append(f"request {index:04d}: clock-dependent fields {json.dumps(fields)}\n"
                        + _first_difference(first, second))
    assert not problems, "\n".join(problems)

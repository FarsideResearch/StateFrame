"""Run artifacts (react_live.log, receipts, ...) must never land under src/ or the package."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ENGINE = SRC / "stateframe" / "engine"

_SCRIPT = textwrap.dedent("""
    import os, pathlib, sys
    engine = pathlib.Path(sys.argv[1])
    sys.path.insert(0, str(engine))
    sys.path.insert(0, str(engine / "probes"))
    from probes import react_repair as rr

    # Unbound POC (no run dir): the same path real writers use for react_live.log / receipts.
    out = rr._results_dir()
    (out / "react_live.log").open("a").write("x\\n")
    (out / ("receipts" + os.environ.get("REACT_TAG", ""))).mkdir(exist_ok=True)
    print(out)

    # Run-bound POC: artifacts go under the run's worker dir.
    worker = pathlib.Path(sys.argv[2])
    rr.POC = worker
    print(rr._results_dir())
""")


def _snapshot() -> set[str]:
    return {str(p.relative_to(SRC)) for p in SRC.rglob("*") if "__pycache__" not in p.parts}


def test_results_dir_never_under_src(tmp_path: Path) -> None:
    before = _snapshot()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp_path / "tmp")}
    env.pop("STATEFRAME_RESULTS_DIR", None)
    (tmp_path / "tmp").mkdir()
    done = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(ENGINE), str(tmp_path / "worker")],
        cwd=ENGINE, env=env, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    unbound, bound = (Path(line) for line in done.stdout.split())
    assert SRC not in unbound.parents and unbound.is_dir()
    assert (unbound / "react_live.log").is_file()
    assert bound == tmp_path / "worker" / "results"
    assert _snapshot() == before
    assert not (ENGINE / "results").exists()


def test_results_dir_env_override(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "STATEFRAME_RESULTS_DIR": str(tmp_path / "r")}
    done = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(ENGINE), str(tmp_path / "worker")],
        cwd=ENGINE, env=env, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split()[0] == str(tmp_path / "r")


def test_no_hardcoded_poc_results_in_engine_sources() -> None:
    offenders = [
        str(p.relative_to(ROOT)) for p in (ENGINE / "probes").glob("react_repair.py")
        if 'POC / "results"' in p.read_text(encoding="utf-8").replace('root = POC / "results"', "")
    ]
    assert offenders == []

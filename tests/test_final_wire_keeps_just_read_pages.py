"""Class killer: every page the model just read is on the next request's final wire bytes.

Recorded from a hosted run (provider without /tokenize, 82,026-byte request cap). The model
read a file, and the very next request it was sent no longer carried those bytes. The menus,
system prompt, StateFrame panels and task filled the protected budget, the diet could not
fit, and later fitting compacted the read pages away while the visibility check still passed.
That run dropped 45 of 141 pages it had just shown the model.

The replay drives the real CLI with the recorded model responses and audits the journal's
hash-checked submitted payloads, not any intermediate view. A page read after the last
request is not checked: no wire carried it, so nothing false reached the model.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import pytest

from _stateframe_home import isolated_stateframe_home
from _timeouts import scaled
from engine_run import run_monitored
from fixtures.final_wire_reads import missing_reads
from tests.tape_server import TapeServer

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="recorded on Linux sandboxing")

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "tests" / "retention_replay" / "hosted-read-squeeze"
# Each recorded turn takes about a minute on a loaded host; leave generous headroom.
SECONDS_PER_TURN = 240


def _replay(tmp_path: Path, turns: int, allocator: str = "off") -> Path:
    case = json.loads((CASE / "case.json").read_text(encoding="utf-8"))
    with gzip.open(CASE / "script.json.gz", "rt", encoding="utf-8") as source:
        script = json.load(source)
    script["turns"] = script["turns"][:turns]
    repo = tmp_path / "repo"
    with tarfile.open(CASE / "repo.tar.gz") as archive:
        archive.extractall(repo, filter="data")
    run_dir = tmp_path / "run"
    environment = {key: value for key, value in os.environ.items() if not key.startswith("STATEFRAME_")}
    environment.update(
        STATEFRAME_HOME=isolated_stateframe_home(),
        PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT / "src" / "stateframe" / "engine"))),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
    )
    budget = scaled(SECONDS_PER_TURN * turns)
    with TapeServer(script) as endpoint:
        command = [
            sys.executable, "-m", "stateframe", "run", "--repo", str(repo), "--task", case["task"],
            "--endpoint", endpoint.url, "--model", script["model"], "--run-dir", str(run_dir),
            "--test-python", "/usr/bin/python3", "--test-cmd", case["test_cmd"],
            "--runner-kind", "generic", "--protect-tests", "off",
            "--max-turns", str(turns), "--timeout", str(budget), "--quiet",
        ]
        if allocator == "off":
            environment["STATEFRAME_WORKING_SET_ALLOCATOR"] = "off"
        else:
            # Exercise the shipped default, with no allocator switch or custom profile.
            assert "STATEFRAME_WORKING_SET_ALLOCATOR" not in environment
            profile = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text())
            assert profile["switches"]["STATEFRAME_WORKING_SET_ALLOCATOR"] == "on"
        run_monitored(command, expected_seconds=budget + 120, run_dir=run_dir,
                      step="retention replay", env=environment,
                      progress_probe=lambda: endpoint.request_count)
        # An aborted or stalled run would audit only a prefix and could pass vacuously.
        assert endpoint.request_count >= turns, (
            f"replay served {endpoint.request_count} of {turns} recorded turns")
    spec = json.loads((run_dir / "spec.json").read_text())
    assert spec["environment"]["STATEFRAME_WORKING_SET_ALLOCATOR"] == allocator
    result = _engine_result(run_dir)
    assert result is not None and not result.startswith("ERR"), (
        f"replay ended with engine result {result!r}, not a completed run")
    if allocator == "on":
        assert result != "BLOCKED", "ordinary read pressure must paginate, never suspend"
    return run_dir


def _engine_result(run_dir: Path) -> str | None:
    """The engine_end result the CLI logged for this run."""
    log = run_dir / "engine.log"
    if not log.is_file():
        return None
    result = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[sf] ") and '"kind":"engine_end"' in line:
            try:
                result = json.loads(line[len("[sf] "):]).get("result")
            except ValueError:
                continue
    return result if isinstance(result, str) else None


@pytest.mark.parametrize("allocator", ["off", "on"])
@pytest.mark.parametrize("turns", [
    pytest.param(8, id="first-reads"),
    pytest.param(38, id="protected-over-cap"),
])
def test_just_read_pages_reach_the_next_final_wire(tmp_path: Path, turns: int, allocator: str) -> None:
    started = time.monotonic()
    run_dir = _replay(tmp_path, turns, allocator)
    violations, checked = missing_reads(run_dir)
    assert checked, "the replay committed no read pages to check"
    assert not violations, (
        f"{len(violations)} of {checked} just-read pages were absent from the next request's "
        f"final wire bytes ({time.monotonic() - started:.0f}s):\n"
        + "\n".join(v.describe() for v in violations[:10]))


def test_auditor_counts_source_lines_and_ignores_read_notices() -> None:
    from fixtures import final_wire_reads as audit
    page = "def load(path):\n    return open(path).read()\n"
    assert audit._missing_chars(page, "x\n" + page) == 0
    assert audit._missing_chars(page, "def load(path):\n") == len("    return open(path).read()")
    assert audit._NOTICE.match("136 more lines; page with cursor=read1:abc:121:0")
    assert audit._NOTICE.match("[read] requested bytes are unchanged since request 1")
    assert audit._NOTICE.match("0 more; complete=true")
    assert not audit._NOTICE.match("    return open(path).read()")
    assert not audit._NOTICE.match("more = 3; complete = True")

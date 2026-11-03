"""Replay recorded allocation refusals through the real StateFrame CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from urllib.request import urlopen

import pytest

from tests.tape_server import TapeServer


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "replay_corpus"
REPO_ROOT = (CORPUS / "_repos").resolve()
ALLOCATION_ERRORS = ("FrameRequestAllocationError", "FrameFit5ImpossibleMinimum")
EXPECTED_CASE_IDS = frozenset({
    "econ-flat-r1-off",
    "econ-flat-r1-p1",
    "econ-flat-r1-p2",
    "econ-flat-r2-p1",
    "econ-flat-r2-p2",
    "econ-flat-sub-1",
    "econ-rel10-full-r1-off",
    "econ-rel10-full-r1-p1",
    "econ-rel10-full-r1-p2",
    "econ-rel10-full-sub-6",
    "econ-relb8-full-r1-off",
    "econ-relb8-full-r1-p1",
    "econ-relb8-full-r1-p2",
    "econ-relb8-full-sub-1",
    "factory-a-econ-feature-r1",
    "factory-a-econ-feature-r3",
    "factory-a-econ-flat-r1",
    "factory-a-econ-flat-r2",
    "factory-a-econ-flat-r3",
    "factory-b-econ-feature-r1",
    "factory-b-econ-feature-r2",
    "factory-b-econ-flat-r1",
    "factory-b-econ-flat-r2",
    "factory-b-econ-flat-r3",
    "legacy-b8-off",
    "legacy-b8-p1",
    "legacy-b8-p2",
    "legacy-guard22",
    "legacy-v037-slot0",
    "presend-full",
    "presend-p3-nosvc",
    "presend-p4-nosvc",
    "presend-p4-svc",
    "presend-p5-nosvc",
})
# Each allowlist entry records the exact honest capacity stop and why its minimum cannot fit.
HONEST_UNDER_CAPACITY_ALLOWLIST: dict[str, dict[str, str]] = {
    "legacy-b8-p2": {
        "error": "FrameFit5ImpossibleMinimum",
        "reason": (
            "At the default 33,344-token prompt budget, the compacted mandatory request "
            "still measures 37,453 tokens (then 40,846); the 112,927-byte minimum also "
            "exceeds the 82,026-byte protected capacity, with no eligible source path left."
        ),
    },
}


@dataclass(frozen=True)
class ReplayOutcome:
    case_id: str
    returncode: int | None
    engine_result: str | None
    summary: dict[str, object] | None
    engine_log: str
    stdout: str
    stderr: str
    elapsed_seconds: float
    endpoint_requests: int

    @property
    def evidence(self) -> str:
        return "\n".join((self.engine_log, self.stdout, self.stderr))


CASE_FILES = tuple(sorted(
    path for path in CORPUS.iterdir()
    if path.is_dir() and (path / "case.json").is_file()
))
CASE_METADATA = {
    path.name: json.loads((path / "case.json").read_text(encoding="utf-8"))
    for path in CASE_FILES
}
CASE_IDS = tuple(path.name for path in CASE_FILES)
_SCANNED = 0
_VIOLATED = 0


@pytest.fixture(scope="session", autouse=True)
def _print_corpus_summary(request):
    yield
    line = (f"allocation replay corpus: expected={len(CASE_IDS)} "
            f"scanned={_SCANNED} violated={_VIOLATED}")
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        print(line, flush=True)
    else:
        reporter.write_line(line)


def _source_root() -> Path:
    configured = os.environ.get("STATEFRAME_REPLAY_SOURCE")
    source = Path(configured) if configured else ROOT / "src"
    source = source.resolve()
    if not (source / "stateframe" / "_child.py").is_file():
        raise AssertionError(f"StateFrame source root is incomplete: {source}")
    return source


def _case_repo(case_id: str) -> Path:
    repo_value = CASE_METADATA[case_id].get("repo")
    if not isinstance(repo_value, str) or not repo_value:
        raise AssertionError(f"case {case_id} has no repository reference")
    repo = (CORPUS / repo_value).resolve()
    try:
        repo.relative_to(REPO_ROOT)
    except ValueError as error:
        raise AssertionError(f"case {case_id} repository must be inside _repos") from error
    if not repo.is_dir():
        raise AssertionError(f"case {case_id} repository is missing: {repo_value}")
    return repo


def _profile_file(case: dict[str, object], source_root: Path, tmp_path: Path) -> Path | None:
    settings = case.get("profile_settings")
    if not isinstance(settings, dict):
        raise AssertionError("case profile settings must be an object")
    overrides = settings.get("switch_overrides")
    if not isinstance(overrides, dict):
        raise AssertionError("case switch overrides must be an object")
    if not overrides:
        return None

    base_path = source_root / "stateframe" / "profiles" / "r2hc3.json"
    profile = json.loads(base_path.read_text(encoding="utf-8"))
    switches = profile.get("switches")
    if not isinstance(switches, dict):
        raise AssertionError(f"invalid shipped profile: {base_path}")
    switches.update({str(key): str(value) for key, value in overrides.items()})
    profile_path = tmp_path / "replay-profile.json"
    profile_path.write_text(json.dumps({
        "schema": "stateframe-product-profile/1",
        "name": f"Allocation replay {settings.get('name', 'profile')}",
        "switches": switches,
    }, indent=1) + "\n", encoding="utf-8")
    return profile_path


def _broken_engine_python(tmp_path: Path) -> Path:
    wrapper = tmp_path / "python-with-broken-deferral"
    bootstrap = "\n".join((
        "import json, sys",
        "from pathlib import Path",
        "child = Path(sys.argv[1]).resolve()",
        "run = Path(sys.argv[2]).resolve()",
        "sys.path.insert(0, str(child.parent.parent))",
        "import stateframe._child as host",
        "host.install_environment(json.loads((run / 'spec.json').read_text()))",
        "sys.path.insert(0, str(child.parent / 'engine'))",
        # finalize_frame_candidate imports the helper at call time, so patch its defining module.
        "import kernel_request_source_window as source_window",
        "marker = run.parent / 'deferral-helper-called'",
        "def broken_deferral(*_args, **_kwargs):",
        "    marker.write_text('called\\n', encoding='utf-8')",
        "    return ()",
        "source_window.partial_source_paths_for_deferred_delivery = broken_deferral",
        "raise SystemExit(host.main([str(child), str(run)]))",
    ))
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os,sys\n"
        f"real_python={sys.executable!r}\n"
        "args=sys.argv[1:]\n"
        "if len(args) >= 3 and args[0] == '-B' and os.path.basename(args[1]) == '_child.py':\n"
        f"    os.execv(real_python, [real_python, '-B', '-c', {bootstrap!r}, *args[1:]])\n"
        "os.execv(real_python, [real_python, *args])\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


def _run_case(tmp_path: Path, case_id: str, *, break_deferral: bool = False) -> ReplayOutcome:
    case_dir = CORPUS / case_id
    metadata = CASE_METADATA[case_id]
    source_root = _source_root()
    repo = tmp_path / "repo"
    shutil.copytree(_case_repo(case_id), repo)
    task = (case_dir / "task.txt").read_text(encoding="utf-8")
    script_path = case_dir / "script.json.gz"
    run_dir = tmp_path / "run"
    profile_path = _profile_file(metadata, source_root, tmp_path)
    recorded_settings = metadata.get("recorded_settings")
    recorded_settings = recorded_settings if isinstance(recorded_settings, dict) else {}
    cache_slot = recorded_settings.get("prompt_cache_slot")
    cache_slot = cache_slot if isinstance(cache_slot, dict) else {}
    slot = cache_slot.get("slot", 6)
    if type(slot) is not int or not 0 <= slot < 8:
        raise AssertionError(f"invalid recorded prompt cache slot for {case_id}: {slot!r}")

    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STATEFRAME_")}
    environment["PYTHONPATH"] = os.pathsep.join((str(source_root), str(source_root / "stateframe" / "engine")))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    started = time.monotonic()
    with TapeServer(script_path) as endpoint:
        command = [
            sys.executable, "-m", "stateframe", "run", "--repo", str(repo), "--task", task,
            "--endpoint", endpoint.url,
            "--model", str(recorded_settings.get("model") or "allocation-replay-model"),
            "--run-dir", str(run_dir), "--test-python", sys.executable,
            "--shell-read", str(source_root),
            "--slot", str(slot), "--max-turns", "120", "--timeout", "1800", "--quiet",
        ]
        if profile_path is not None:
            command.extend(("--profile", str(profile_path)))
        if break_deferral:
            command.extend(("--python", str(_broken_engine_python(tmp_path))))
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, env=environment,
                timeout=1860, check=False,
            )
            returncode = process.returncode
            stdout, stderr = process.stdout, process.stderr
        except subprocess.TimeoutExpired as error:
            returncode = None
            stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
            stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or ""
        request_count = endpoint.request_count

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    engine_log = (run_dir / "engine.log").read_text(errors="replace") if (run_dir / "engine.log").is_file() else ""
    engine_result = summary.get("engine_result") if isinstance(summary, dict) else None
    return ReplayOutcome(case_id, returncode,
                         engine_result if isinstance(engine_result, str) else None,
                         summary, engine_log, stdout, stderr, time.monotonic() - started, request_count)


def test_corpus_keeps_its_case_set_and_uses_shared_repositories():
    assert set(CASE_IDS) == EXPECTED_CASE_IDS
    assert set(HONEST_UNDER_CAPACITY_ALLOWLIST) == {"legacy-b8-p2"}
    assert "legacy-guard22" in CASE_METADATA
    repo_references = set()
    for case_id in CASE_IDS:
        case_dir = CORPUS / case_id
        repo_references.add(_case_repo(case_id))
        assert (case_dir / "script.json.gz").is_file()
    assert len(repo_references) < len(CASE_IDS)


def test_tape_server_reads_a_gzip_script_and_keeps_neutral_model_id():
    tape_path = CASE_FILES[0] / "script.json.gz"
    with TapeServer(tape_path) as endpoint:
        with urlopen(f"{endpoint.url}/models", timeout=2) as response:
            payload = json.load(response)
    assert payload["data"][0]["id"] == "allocation-replay-model"


def _violation(outcome: ReplayOutcome) -> str | None:
    errors = [error for error in ALLOCATION_ERRORS if error in outcome.evidence]
    allowlist = HONEST_UNDER_CAPACITY_ALLOWLIST.get(outcome.case_id)
    if allowlist is not None:
        reason = allowlist.get("reason", "").strip()
        expected_error = allowlist.get("error", "")
        if not reason:
            return "allowlist entry has no documented reason"
        if expected_error not in ALLOCATION_ERRORS:
            return "allowlist entry has no recognized allocation error"
        if outcome.engine_result == "SUBMITTED":
            return None
        if (outcome.engine_result == f"ERR:{expected_error}"
                and expected_error in errors):
            return None
        return (
            f"allowlisted case stopped for {outcome.engine_result!r}; expected "
            f"SUBMITTED or ERR:{expected_error} ({reason})"
        )
    if errors:
        return "allocation refusal observed: " + ", ".join(errors)
    if outcome.engine_result != "SUBMITTED":
        return f"engine result was {outcome.engine_result!r}, expected SUBMITTED"
    # A replay may submit a candidate while its fixture checks remain red; this guard measures
    # allocation refusals and the engine result, not whether the recorded task's tests pass.
    return None


@pytest.mark.host_sandbox
@pytest.mark.parametrize("case_id", CASE_IDS, ids=CASE_IDS)
def test_recorded_allocation_refusal_reaches_submitted(tmp_path, case_id):
    global _SCANNED, _VIOLATED
    outcome = _run_case(tmp_path, case_id)
    _SCANNED += 1
    violation = _violation(outcome)
    if violation is not None:
        _VIOLATED += 1
    assert violation is None, (
        f"{case_id}: {violation}; result={outcome.engine_result!r}; "
        f"exit={outcome.returncode}; endpoint requests={outcome.endpoint_requests}; "
        f"elapsed={outcome.elapsed_seconds:.2f}s\n{outcome.engine_log[-6000:]}\n"
        f"{outcome.stdout[-2000:]}\n{outcome.stderr[-2000:]}"
    )


def test_under_capacity_allowlist_requires_its_documented_error():
    entry = HONEST_UNDER_CAPACITY_ALLOWLIST["legacy-b8-p2"]
    expected = ReplayOutcome(
        case_id="legacy-b8-p2", returncode=2,
        engine_result=f"ERR:{entry['error']}", summary=None,
        engine_log=entry["error"], stdout="", stderr="",
        elapsed_seconds=0.0, endpoint_requests=4,
    )
    unrelated = ReplayOutcome(
        case_id="legacy-b8-p2", returncode=2,
        engine_result="ERR:FrameRequestAllocationError", summary=None,
        engine_log="FrameRequestAllocationError", stdout="", stderr="",
        elapsed_seconds=0.0, endpoint_requests=4,
    )
    assert _violation(expected) is None
    assert _violation(unrelated) is not None


@pytest.mark.host_sandbox
def test_planted_broken_deferral_is_reported_as_a_corpus_violation(tmp_path):
    # guard22 is the host-measured late-fit deferral case (refused before the fix, SUBMITTED after).
    case_id = "legacy-guard22"
    assert case_id in CASE_METADATA
    outcome = _run_case(tmp_path, case_id, break_deferral=True)
    assert (tmp_path / "deferral-helper-called").is_file(), (
        "planted late-fit deferral helper was not exercised by this tape"
    )
    violation = _violation(outcome)
    assert violation is not None, (
        f"broken late-fit helper escaped the corpus guard; result={outcome.engine_result!r}; "
        f"exit={outcome.returncode}; endpoint requests={outcome.endpoint_requests}"
    )
    assert any(error in outcome.evidence for error in ALLOCATION_ERRORS), (
        "planted deferral failure did not reach a guarded allocation refusal: " + violation
    )

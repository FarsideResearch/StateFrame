"""Replay recorded allocation refusals through the real StateFrame CLI."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from dataclasses import dataclass
from urllib.request import Request, urlopen

import pytest

from _journal import measurement_events
from _stateframe_home import isolated_stateframe_home
from engine_run import EngineRunTimeout, run_monitored
from tests.tape_server import TapeServer


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "replay_corpus"
REPO_ROOT = (CORPUS / "_repos").resolve()
ALLOCATION_ERRORS = ("FrameRequestAllocationError", "FrameFit5ImpossibleMinimum")
# econ-flat-r1-p2 alone: 244 s / 23 requests (~10 s per turn). This is a generous
# backstop only; progress-aware deadlines come from the shared family-1 helper.
BACKSTOP_SECONDS_PER_TURN = 10
BACKSTOP_HEADROOM_MULTIPLIER = 6
BACKSTOP_FLOOR_SECONDS = 600
PARENT_TIMEOUT_MARGIN_SECONDS = 30
EXPECTED_CASE_IDS = frozenset({
    "clean-meter-22file-r1",
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
    served_steps: int
    total_steps: int
    terminal_served: bool
    tokenize_request_count: int = 0
    delayed_tokenize_calls: tuple[int, ...] = ()
    tokenize_request_log: tuple[dict[str, object], ...] = ()
    measurement_events: tuple[dict[str, object], ...] = ()
    request_hashes: tuple[str, ...] = ()
    request_bytes: tuple[int, ...] = ()

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


def _backstop_timeout_seconds(total_steps: int) -> int:
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError("tape must contain at least one step")
    return max(
        BACKSTOP_FLOOR_SECONDS,
        BACKSTOP_SECONDS_PER_TURN * total_steps * BACKSTOP_HEADROOM_MULTIPLIER,
    )


def _short_tokenize_timeout_engine_python(
        tmp_path: Path, timeout_s: float, *,
        reprobe_after_requests: int | None = None,
        reprobe_after_s: float | None = None,
) -> Path:
    wrapper = tmp_path / "python-with-short-tokenize-timeout"
    bootstrap = "\n".join((
        "import json, sys",
        "from pathlib import Path",
        "child = Path(sys.argv[1]).resolve()",
        "run = Path(sys.argv[2]).resolve()",
        "sys.path.insert(0, str(child.parent.parent))",
        "import stateframe._child as host",
        "host.install_environment(json.loads((run / 'spec.json').read_text()))",
        "sys.path.insert(0, str(child.parent / 'engine'))",
        "import kernel_request_diet as request_diet",
        f"request_diet.TOKENIZE_REQUEST_TIMEOUT_SECONDS = {timeout_s!r}",
        f"request_diet.TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS = {timeout_s!r}",
        f"request_diet.TOKENIZE_QUEUEING_TIMEOUT_SECONDS = {timeout_s!r}",
        *([f"request_diet.TOKENIZE_REPROBE_AFTER_REQUESTS = {reprobe_after_requests!r}"]
          if reprobe_after_requests is not None else []),
        *([f"request_diet.TOKENIZE_REPROBE_AFTER_SECONDS = {reprobe_after_s!r}"]
          if reprobe_after_s is not None else []),
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


def _run_case(
        tmp_path: Path, case_id: str, *, break_deferral: bool = False,
        tokenize_delay_s: float | None = None,
        tokenize_slow_calls: tuple[int, ...] = (),
        tokenize_timeout_s: float | None = None,
        tokenize_reprobe_after_requests: int | None = None,
        tokenize_reprobe_after_s: float | None = None,
        max_turns: int = 120,
) -> ReplayOutcome:
    case_dir = CORPUS / case_id
    metadata = CASE_METADATA[case_id]
    source_root = _source_root()
    repo = tmp_path / "repo"
    shutil.copytree(_case_repo(case_id), repo)
    task = (case_dir / "task.txt").read_text(encoding="utf-8")
    script_path = case_dir / "script.json.gz"
    tape_script = script_path
    if tokenize_delay_s is not None:
        with gzip.open(script_path, "rt", encoding="utf-8") as source:
            tape_script = json.load(source)
        tape_script["tokenize_delay_s"] = tokenize_delay_s
        tape_script["tokenize_slow_calls"] = list(tokenize_slow_calls)
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
    environment["STATEFRAME_HOME"] = isolated_stateframe_home()
    environment["PYTHONPATH"] = os.pathsep.join((str(source_root), str(source_root / "stateframe" / "engine")))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    started = time.monotonic()
    with TapeServer(tape_script) as endpoint:
        case_timeout = _backstop_timeout_seconds(endpoint.total_steps)
        command = [
            sys.executable, "-m", "stateframe", "run", "--repo", str(repo), "--task", task,
            "--endpoint", endpoint.url,
            "--model", str(recorded_settings.get("model") or "allocation-replay-model"),
            "--run-dir", str(run_dir), "--test-python", sys.executable,
            "--shell-read", str(source_root),
            "--slot", str(slot), "--max-turns", str(max_turns), "--timeout", str(case_timeout), "--quiet",
        ]
        if profile_path is not None:
            command.extend(("--profile", str(profile_path)))
        if break_deferral:
            command.extend(("--python", str(_broken_engine_python(tmp_path))))
        elif tokenize_timeout_s is not None:
            command.extend(("--python", str(
                _short_tokenize_timeout_engine_python(
                    tmp_path, tokenize_timeout_s,
                    reprobe_after_requests=tokenize_reprobe_after_requests,
                    reprobe_after_s=tokenize_reprobe_after_s,
                )
            )))
        try:
            process = run_monitored(
                command,
                expected_seconds=case_timeout + PARENT_TIMEOUT_MARGIN_SECONDS,
                run_dir=run_dir,
                step=f"corpus {case_id}",
                env=environment,
                progress_probe=lambda: endpoint.request_count,
            )
            returncode = process.returncode
            stdout, stderr = process.stdout, process.stderr
        except EngineRunTimeout as error:
            returncode = None
            stdout, stderr = error.stdout_tail, error.stderr_tail
        request_count = endpoint.request_count
        served_steps = endpoint.served_steps
        total_steps = endpoint.total_steps
        terminal_served = endpoint.terminal_served
        tokenize_request_count = endpoint.tokenize_request_count
        tokenize_request_log = endpoint.tokenize_request_log
        delayed_tokenize_calls = tuple(
            int(entry["call"])
            for entry in tokenize_request_log
            if entry.get("delayed") is True
        )

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    engine_log = (run_dir / "engine.log").read_text(errors="replace") if (run_dir / "engine.log").is_file() else ""
    engine_result = summary.get("engine_result") if isinstance(summary, dict) else None
    recorded_measurement_events = (
        tuple(measurement_events(run_dir)) if tokenize_delay_s is not None else ()
    )
    request_hashes: list[str] = []
    request_bytes: list[int] = []
    sends_path = run_dir / "sends.jsonl"
    if sends_path.is_file():
        for line in sends_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            digest = row.get("request_sha256")
            size = row.get("request_bytes")
            if isinstance(digest, str):
                request_hashes.append(digest)
            if type(size) is int:
                request_bytes.append(size)
    return ReplayOutcome(
        case_id=case_id,
        returncode=returncode,
        engine_result=engine_result if isinstance(engine_result, str) else None,
        summary=summary,
        engine_log=engine_log,
        stdout=stdout,
        stderr=stderr,
        elapsed_seconds=time.monotonic() - started,
        endpoint_requests=request_count,
        served_steps=served_steps,
        total_steps=total_steps,
        terminal_served=terminal_served,
        tokenize_request_count=tokenize_request_count,
        delayed_tokenize_calls=delayed_tokenize_calls,
        tokenize_request_log=tokenize_request_log,
        measurement_events=recorded_measurement_events,
        request_hashes=tuple(request_hashes),
        request_bytes=tuple(request_bytes),
    )


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


def test_clean_meter_corpus_case_passes_scrub_and_size_audit():
    case_dir = CORPUS / "clean-meter-22file-r1"
    case = CASE_METADATA[case_dir.name]
    with gzip.open(case_dir / "script.json.gz", "rb") as source:
        tape_bytes = source.read()
    tape_hash = hashlib.sha256(tape_bytes).hexdigest()
    corpus_bytes = sum(path.stat().st_size for path in CORPUS.rglob("*") if path.is_file())
    content = tape_bytes + (case_dir / "task.txt").read_bytes() + (case_dir / "case.json").read_bytes()
    tape = json.loads(tape_bytes)

    assert case.get("tape_sha256") == tape_hash
    assert case.get("repo") == "_repos/0420a86aadd2"
    assert case.get("recorded_settings", {}).get("model") == "allocation-replay-model"
    assert tape.get("model") == "allocation-replay-model"
    assert corpus_bytes < 1_000_000
    assert b"/Users/" not in content
    assert b"C:\\Users\\" not in content
    assert b"/home/mike/" not in content.lower()


def test_every_corpus_file_has_neutral_model_and_no_host_identity():
    private_identity = re.compile(
        r"(?i)(?:Qwen|A3B|\.gguf\b|C:\\+Users\\+|/(?:Users|private|home)/"
        r"|marida|farside|sf-work|host_tok28|item28[bc]|(?:\d{1,3}\.){3}\d{1,3})"
    )

    def model_values(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if "model" in key.lower() and isinstance(item, str):
                    yield item
                yield from model_values(item)
        elif isinstance(value, list):
            for item in value:
                yield from model_values(item)

    for path in sorted(item for item in CORPUS.rglob("*") if item.is_file()):
        raw = path.read_bytes()
        if path.suffix == ".gz":
            raw = gzip.decompress(raw)
        content = raw.decode("utf-8", errors="ignore")
        assert private_identity.search(content) is None, path.relative_to(ROOT)
        if path.suffix == ".json" or path.suffix == ".gz":
            try:
                document = json.loads(content)
            except json.JSONDecodeError:
                continue
            assert set(model_values(document)) <= {"allocation-replay-model"}, path.relative_to(ROOT)


def test_tape_server_reads_a_gzip_script_and_keeps_neutral_model_id():
    tape_path = CASE_FILES[0] / "script.json.gz"
    with TapeServer(tape_path) as endpoint:
        with urlopen(f"{endpoint.url}/models", timeout=2) as response:
            payload = json.load(response)
    assert payload["data"][0]["id"] == "allocation-replay-model"


def _violation(outcome: ReplayOutcome) -> str | None:
    if not outcome.terminal_served:
        return (
            "INCOMPLETE: terminal tape turn was not served "
            f"(served/total={outcome.served_steps}/{outcome.total_steps})"
        )
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
@pytest.mark.serial_lane
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
        f"tape served/total={outcome.served_steps}/{outcome.total_steps}; "
        f"terminal_served={outcome.terminal_served}; "
        f"elapsed={outcome.elapsed_seconds:.2f}s\n{outcome.engine_log[-6000:]}\n"
        f"{outcome.stdout[-2000:]}\n{outcome.stderr[-2000:]}"
    )


def test_under_capacity_allowlist_requires_its_documented_error():
    entry = HONEST_UNDER_CAPACITY_ALLOWLIST["legacy-b8-p2"]
    expected = ReplayOutcome(
        case_id="legacy-b8-p2", returncode=2,
        engine_result=f"ERR:{entry['error']}", summary=None,
        engine_log=entry["error"], stdout="", stderr="",
        elapsed_seconds=0.0, endpoint_requests=4, served_steps=6, total_steps=6,
        terminal_served=True,
    )
    unrelated = ReplayOutcome(
        case_id="legacy-b8-p2", returncode=2,
        engine_result="ERR:FrameRequestAllocationError", summary=None,
        engine_log="FrameRequestAllocationError", stdout="", stderr="",
        elapsed_seconds=0.0, endpoint_requests=4, served_steps=6, total_steps=6,
        terminal_served=True,
    )
    assert _violation(expected) is None
    assert _violation(unrelated) is not None


@pytest.mark.host_sandbox
@pytest.mark.serial_lane
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


def _synthetic_tape(messages: list[dict[str, object]]) -> dict[str, object]:
    return {"turns": [{"message": message} for message in messages]}


def _request_tape_turn(endpoint_url: str) -> dict[str, object]:
    body = json.dumps({
        "model": "allocation-replay-model",
        "messages": [{"role": "user", "content": "continue"}],
    }).encode("utf-8")
    request = Request(
        f"{endpoint_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=2) as response:
        payload = json.load(response)
    return payload["choices"][0]["message"]


def test_backstop_timeout_scales_with_tape_length():
    assert _backstop_timeout_seconds(1) == 600
    assert _backstop_timeout_seconds(10) == 600
    assert _backstop_timeout_seconds(11) == 660
    assert _backstop_timeout_seconds(23) == 1380
    with pytest.raises(ValueError, match="at least one step"):
        _backstop_timeout_seconds(0)


@pytest.mark.host_sandbox
def test_slow_progressing_tape_serves_its_terminal_turn():
    messages = [
        {"role": "assistant", "content": f"turn-{index}"}
        for index in range(3)
    ]
    with TapeServer(_synthetic_tape(messages), turn_delay_seconds=0.01) as endpoint:
        for _ in range(endpoint.total_steps):
            _request_tape_turn(endpoint.url)
        assert endpoint.served_steps == endpoint.total_steps == 3
        assert endpoint.terminal_served is True


@pytest.mark.host_sandbox
def test_repeated_nonterminal_tape_is_reported_incomplete():
    repeat_turn = {
        "role": "assistant",
        "tool_calls": [{
            "id": "repeat",
            "type": "function",
            "function": {"name": "continue", "arguments": "{}"},
        }],
    }
    terminal_turn = {"role": "assistant", "content": "submitted"}
    with TapeServer(_synthetic_tape([repeat_turn, repeat_turn, terminal_turn])) as endpoint:
        first = _request_tape_turn(endpoint.url)
        repeated = _request_tape_turn(endpoint.url)
        assert repeated == first
        outcome = ReplayOutcome(
            case_id="synthetic-incomplete", returncode=0, engine_result="SUBMITTED",
            summary=None, engine_log="", stdout="", stderr="", elapsed_seconds=0.0,
            endpoint_requests=endpoint.request_count, served_steps=endpoint.served_steps,
            total_steps=endpoint.total_steps, terminal_served=endpoint.terminal_served,
        )

    assert outcome.terminal_served is False
    assert outcome.served_steps == 2
    assert _violation(outcome) == (
        "INCOMPLETE: terminal tape turn was not served (served/total=2/3)"
    )

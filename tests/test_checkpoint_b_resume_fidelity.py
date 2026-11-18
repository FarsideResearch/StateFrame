"""Checkpoint B must rebuild the same next provider request as an uninterrupted run."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from _timeouts import scaled
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, serve


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ("discount_bug", "tiny_site", "brief_to_report")
COMPLETED_ACTIONS = 1
NEXT_REQUEST = f"{COMPLETED_ACTIONS + 1:04d}.request.json"
_EDIT_RECEIPT = re.compile(rb'edit_receipt\\*":\\*"([0-9a-f]{20})')
_OUTPUT_BYTE_COUNT = re.compile(
    r"\b(stdout_bytes|stderr_bytes|stdout_omitted_bytes|stderr_omitted_bytes|"
    r"assertion_context_omitted_bytes)=\d+"
)


@dataclass(frozen=True)
class Control:
    repo: Path
    launcher: Path
    profile: Path
    home: Path
    next_request: bytes
    folder: Path


@dataclass(frozen=True)
class InterruptedRun:
    run_dir: Path
    pending_request: bytes
    resumed_request: bytes
    checkpoint_dir: Path
    checkpoint_meta: dict[str, object]


class PauseAfterActions(Script):
    """Hold the next reply before and after resume, without running later example steps."""

    def __init__(self, document: dict[str, object], completed_actions: int) -> None:
        super().__init__(document)
        self.completed_actions = completed_actions
        self.waiting = threading.Event()
        self.release = threading.Event()
        self.interrupted = threading.Event()
        self.resumed_waiting = threading.Event()
        self.resumed_release = threading.Event()
        self.resumed_interrupted = threading.Event()
        self.pause_count = 0

    def next_call(self, offered: list[str]) -> dict:
        if self.position == self.completed_actions:
            self.pause_count += 1
            if self.pause_count == 1:
                waiting, release, interrupted = self.waiting, self.release, self.interrupted
            elif self.pause_count == 2:
                waiting, release, interrupted = (
                    self.resumed_waiting, self.resumed_release, self.resumed_interrupted,
                )
            else:
                raise AssertionError("test endpoint received an unexpected third target request")
            waiting.set()
            if not release.wait(scaled(180)):
                raise TimeoutError("test did not release the paused scripted reply")
            interrupted.set()
            raise ScriptStepUnavailable("request interrupted before a response")
        return super().next_call(offered)


def _rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text().splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def _requests(run_dir: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes()
            for path in sorted((run_dir / "wire").glob("*.request.json"))}


def _blank_receipts(data: bytes) -> bytes:
    """Normalize only per-run edit receipts, matching the checkpoint wire-parity helper."""
    for index, receipt in enumerate(dict.fromkeys(_EDIT_RECEIPT.findall(data))):
        data = data.replace(receipt, f"<receipt-{index}>".encode())
    return data


def _canonical_request(data: bytes) -> bytes:
    document = json.loads(_blank_receipts(data).decode("utf-8"))
    document = _canonical_value(document)
    return json.dumps(document, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_value(value: object) -> object:
    """Ignore measured output byte counts; preserve all request text and data."""
    if isinstance(value, str):
        return _OUTPUT_BYTE_COUNT.sub(r"\1=<byte-count>", value)
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical_value(item) for key, item in value.items()}
    return value


def _assert_canonical_equal(left: bytes, right: bytes) -> None:
    left_canonical = _canonical_request(left)
    right_canonical = _canonical_request(right)
    if left_canonical != right_canonical:
        raise AssertionError(
            "canonical request mismatch: "
            f"{hashlib.sha256(left_canonical).hexdigest()} != "
            f"{hashlib.sha256(right_canonical).hexdigest()}"
        )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree_label() -> str:
    return os.environ.get("STATEFRAME_CHECKPOINT_TEST_TREE", "unspecified")


def _write_launcher(path: Path) -> Path:
    """Pin the first task identity and pass the source path to the engine child."""
    path.write_text(
        "import secrets, sys\n"
        "from stateframe import cli, runner\n"
        "runner.PASS_ENV = tuple(dict.fromkeys((*runner.PASS_ENV, "
        "'PYTHONPATH', 'PYTHONHASHSEED', 'STATEFRAME_HOME')))\n"
        "original = secrets.token_hex\n"
        "used = False\n"
        "def fixed(size=None):\n"
        "    global used\n"
        "    if size == 16 and not used:\n"
        "        used = True\n"
        "        return 'a' * 32\n"
        "    return original(size)\n"
        "secrets.token_hex = fixed\n"
        "raise SystemExit(cli.main())\n"
    )
    return path


def _write_restore_index_hook(directory: Path, marker: Path) -> Path:
    """Inject one post-restore message-history index mutation into the engine child."""
    directory.mkdir(parents=True, exist_ok=True)
    marker_literal = repr(str(marker))
    (directory / "sitecustomize.py").write_text(
        "import hashlib\n"
        "import json\n"
        "from pathlib import Path\n"
        "from stateframe import runner\n"
        "runner.PASS_ENV = tuple(dict.fromkeys((*runner.PASS_ENV, 'PYTHONPATH', "
        "'PYTHONHASHSEED', 'STATEFRAME_HOME')))\n"
        "import kernel_checkpoint\n"
        "_original_restore = kernel_checkpoint.restore\n"
        f"_marker = Path({marker_literal})\n"
        "def _restore_with_mutated_message(frame, checkpoint, run_dir):\n"
        "    meta = _original_restore(frame, checkpoint, run_dir)\n"
        "    messages = frame.f_locals.get('msgs')\n"
        "    if type(messages) is not list:\n"
        "        raise RuntimeError('restore control found no restored message history list')\n"
        "    target = next((index for index in range(len(messages) - 1, -1, -1)\n"
        "                   if type(messages[index]) is dict\n"
        "                   and messages[index].get('role') == 'user'\n"
        "                   and isinstance(messages[index].get('content'), str)), None)\n"
        "    if target is None:\n"
        "        raise RuntimeError('restore control found no restored user message entry')\n"
        "    original = messages[target]['content']\n"
        "    messages[target]['content'] = original + '\\n[planted restore index mutation]'\n"
        "    _marker.write_text(json.dumps({\n"
        "        'entry': 'msgs[index]', 'index': target, 'entries': len(messages),\n"
        "        'old_sha256': hashlib.sha256(original.encode()).hexdigest(),\n"
        "        'new_sha256': hashlib.sha256(messages[target]['content'].encode()).hexdigest(),\n"
        "    }, sort_keys=True))\n"
        "    return meta\n"
        "kernel_checkpoint.restore = _restore_with_mutated_message\n"
    )
    return directory


def _write_profile(path: Path) -> Path:
    profile = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text())
    profile.setdefault("switches", {})["STATEFRAME_CHECKPOINT"] = "on"
    profile["switches"]["STATEFRAME_CHECKPOINT_EVERY"] = "1"
    path.write_text(json.dumps(profile))
    return path


def _environment(home: Path, *, restore_hook: Path | None = None) -> dict[str, str]:
    environment = {key: os.environ[key] for key in ("PATH", "LANG", "TMPDIR")
                   if key in os.environ}
    paths = [str(ROOT / "src"), str(ROOT / "src/stateframe/engine")]
    if restore_hook is not None:
        paths.insert(0, str(restore_hook))
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["PYTHONHASHSEED"] = "0"
    environment["HOME"] = str(home)
    environment["STATEFRAME_HOME"] = str(home / ".stateframe")
    environment.pop("STATEFRAME_STABLE_TOOL_HEAD", None)
    return environment


def _script_document(example: str) -> dict[str, object]:
    return json.loads((ROOT / "examples" / f"{example}.script.json").read_text())


def _command(example: str, repo: Path, run_dir: Path, url: str,
             launcher: Path, profile: Path) -> list[str]:
    return [
        sys.executable, str(launcher), "run", "--repo", str(repo),
        "--task", f"Complete the bundled {example} example and run its checks.",
        "--endpoint", url, "--model", "scripted", "--run-dir", str(run_dir),
        "--test-python", sys.executable, "--max-turns", "32", "--shell", "off",
        "--keep-checkpoints", "--profile", str(profile), "--quiet",
    ]


def _run_control(example: str, repo: Path, run_dir: Path,
                 launcher: Path, profile: Path, home: Path) -> bytes:
    script = PauseAfterActions(_script_document(example), COMPLETED_ACTIONS)
    server = serve(script)
    process: subprocess.Popen[str] | None = None
    engine_pid: int | None = None
    try:
        home.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            _command(example, repo, run_dir, f"http://127.0.0.1:{server.server_port}/v1",
                     launcher, profile),
            env=_environment(home), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        _wait_for_pause(script.waiting, process, 360, "uninterrupted control")
        _wait_for_checkpoint(run_dir, process, COMPLETED_ACTIONS)
        requests = _requests(run_dir)
        assert NEXT_REQUEST in requests, sorted(requests)
        assert b"sandbox_seconds_remaining=unbounded" in requests[NEXT_REQUEST]
        print(
            f"UNINTERRUPTED_CONTROL tree={_tree_label()} example={example} "
            f"request={NEXT_REQUEST} raw_sha256={_digest(requests[NEXT_REQUEST])} "
            f"canonical_sha256={_digest(_canonical_request(requests[NEXT_REQUEST]))}",
            flush=True,
        )
        return requests[NEXT_REQUEST]
    finally:
        if process is not None and process.poll() is None:
            if engine_pid is None:
                try:
                    engine_pid = _engine_child_pid(process.pid)
                except (AssertionError, subprocess.SubprocessError):
                    engine_pid = None
            if engine_pid is not None:
                try:
                    identity = record_identity(
                        engine_pid, launched_after=0, command_fragment="_child.py",
                    )
                    signal_run_verified(
                        run_dir, engine_pid, identity, signal.SIGKILL,
                        deadline=time.monotonic(), what="checkpoint B control cleanup",
                        require_checkpoint_saved=False,
                    )
                except (ProcessLookupError, RuntimeError):
                    pass
            script.release.set()
            signal_run_owned(
                run_dir, process, signal.SIGTERM, deadline=time.monotonic(),
                what="checkpoint B control cleanup", require_checkpoint_saved=False,
            )
            try:
                process.communicate(timeout=scaled(30))
            except subprocess.TimeoutExpired:
                signal_run_owned(
                    run_dir, process, signal.SIGKILL, deadline=time.monotonic(),
                    what="checkpoint B control cleanup", require_checkpoint_saved=False,
                )
                process.communicate(timeout=scaled(10))
        script.release.set()
        server.shutdown()
        server.server_close()


def _saved_checkpoint(run_dir: Path, sends: int) -> tuple[Path, dict[str, object]] | None:
    for row in _rows(run_dir / "checkpoints/events.jsonl"):
        if row.get("event") != "checkpoint_saved" or not isinstance(row.get("dir"), str):
            continue
        folder = run_dir / "checkpoints" / str(row["dir"])
        meta_path = folder / "checkpoint.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("sends") == sends:
            return folder, meta
    return None


def _engine_child_pid(parent_pid: int) -> int:
    result = subprocess.run(
        ["ps", "-o", "pid=", "--ppid", str(parent_pid)],
        capture_output=True, text=True, check=True,
    )
    children = [int(value) for value in result.stdout.split() if value.isdigit()]
    assert len(children) == 1, f"expected one owned engine child for PID {parent_pid}, got {children}"
    return children[0]


def _wait_pid_gone(pid: int, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                                capture_output=True, text=True, check=False)
        state = result.stdout.strip().split(maxsplit=1)[0] if result.stdout.strip() else ""
        if result.returncode != 0 or not state or state.startswith("Z"):
            return
        time.sleep(0.05)
    raise AssertionError(f"engine child PID {pid} remained after the runner exited")


def _wait_for_checkpoint(run_dir: Path, process: subprocess.Popen[str],
                         sends: int) -> tuple[Path, dict[str, object]]:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        saved = _saved_checkpoint(run_dir, sends)
        if saved is not None:
            return saved
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=scaled(10))
            raise AssertionError(
                f"run stopped before saving checkpoint at send {sends}:\n{stdout}{stderr}"
            )
        time.sleep(0.1)
    raise AssertionError(f"no complete checkpoint at send {sends} within 180 seconds")


def _wait_for_pause(event: threading.Event, process: subprocess.Popen[str],
                    timeout: float, label: str) -> None:
    deadline = time.monotonic() + timeout
    while not event.wait(0.2):
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=scaled(10))
            raise AssertionError(f"{label} stopped before its target request:\n{stdout}{stderr}")
        if time.monotonic() >= deadline:
            raise AssertionError(f"{label} did not reach its target request within {timeout} seconds")


def _assert_checkpoint_restored(run_dir: Path, sends: int) -> None:
    events = _rows(run_dir / "checkpoints/events.jsonl")
    assert any(row.get("event") == "checkpoint_restored" for row in events), events
    spec = json.loads((run_dir / "spec.json").read_text())
    selected = spec.get("resume", {}).get("checkpoint", {})
    assert selected.get("sends") == sends, selected
    engine_log = run_dir / "engine.log"
    engine_events = [
        json.loads(line.removeprefix("[sf] "))
        for line in engine_log.read_text().splitlines()
        if line.startswith("[sf] ")
    ]
    assert any(row.get("kind") == "checkpoint_verified" for row in engine_events)
    assert not any(row.get("kind") == "checkpoint_unfaithful" for row in engine_events)
    assert not (run_dir / "checkpoint-refused.json").exists()
    assert not (run_dir / "attempts/2").exists()


def _run_interrupted(example: str, repo: Path, run_dir: Path, launcher: Path,
                     profile: Path, home: Path, *,
                     restore_hook: Path | None = None,
                     expect_unfaithful_restore: bool = False,
                     completed_actions: int = COMPLETED_ACTIONS,
                     checkpoint_sends: int | None = None) -> InterruptedRun:
    if checkpoint_sends is None:
        checkpoint_sends = completed_actions
    next_request = f"{completed_actions + 1:04d}.request.json"
    script = PauseAfterActions(_script_document(example), completed_actions)
    server = serve(script)
    url = f"http://127.0.0.1:{server.server_port}/v1"
    process: subprocess.Popen[str] | None = None
    engine_pid: int | None = None
    resume_process: subprocess.Popen[str] | None = None
    resume_pid: int | None = None
    try:
        home.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            _command(example, repo, run_dir, url, launcher, profile),
            env=_environment(home), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        _wait_for_pause(script.waiting, process, 360, "first attempt")

        checkpoint_dir, checkpoint_meta = _wait_for_checkpoint(
            run_dir, process, checkpoint_sends,
        )
        request_path = run_dir / "wire" / next_request
        assert request_path.is_file(), "paused provider request was not recorded"
        pending = request_path.read_bytes()
        completed = [name for name in _requests(run_dir)
                     if (run_dir / "wire" / name.replace(".request.", ".response.")).is_file()]
        assert len(completed) == completed_actions, completed

        engine_pid = _engine_child_pid(process.pid)
        engine_identity = record_identity(
            engine_pid, launched_after=0, command_fragment="_child.py",
        )
        signal_run_verified(
            run_dir, engine_pid, engine_identity, signal.SIGKILL,
            deadline=time.monotonic() + 180, what="checkpoint B first-attempt interruption",
        )
        script.release.set()
        assert script.interrupted.wait(scaled(10)), "scripted endpoint did not release the interrupted call"
        process.communicate(timeout=scaled(90))
        _wait_pid_gone(engine_pid)

        resume_process = subprocess.Popen(
            [sys.executable, "-m", "stateframe", "resume", str(run_dir),
             "--endpoint", url, "--keep-checkpoints", "--quiet"],
            env=_environment(home, restore_hook=restore_hook),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        _wait_for_pause(script.resumed_waiting, resume_process, 360, "resumed attempt")
        if expect_unfaithful_restore:
            resume_spec = json.loads((run_dir / "spec.json").read_text()).get("resume", {})
            refusal = resume_spec.get("checkpoint_refused")
            assert (isinstance(refusal, str) and "CheckpointUnfaithful" in refusal
                    and "request differs from the request in flight" in refusal), refusal
            candidates = sorted((run_dir / "attempts").glob(f"*/wire/{next_request}"))
            changed_requests = []
            for path in candidates:
                data = path.read_bytes()
                if data != pending:
                    changed_requests.append(data)
            assert changed_requests, (
                "restore mutation did not produce a differing recorded request: "
                f"{[str(path.relative_to(run_dir)) for path in candidates]}"
            )
            return InterruptedRun(run_dir, pending, changed_requests[-1],
                                  checkpoint_dir, checkpoint_meta)
        _assert_checkpoint_restored(run_dir, checkpoint_sends)
        requests = _requests(run_dir)
        assert next_request in requests
        assert b"sandbox_seconds_remaining=unbounded" in requests[next_request]
        resumed_request = requests[next_request]
        resume_pid = _engine_child_pid(resume_process.pid)
        resume_identity = record_identity(
            resume_pid, launched_after=0, command_fragment="_child.py",
        )
        signal_run_verified(
            run_dir, resume_pid, resume_identity, signal.SIGKILL,
            deadline=time.monotonic() + 180, what="checkpoint B resumed-attempt interruption",
        )
        script.resumed_release.set()
        assert script.resumed_interrupted.wait(scaled(10)), "resumed endpoint call did not release"
        resume_process.communicate(timeout=scaled(90))
        _wait_pid_gone(resume_pid)
        return InterruptedRun(run_dir, pending, requests[next_request],
                              checkpoint_dir, checkpoint_meta)
    finally:
        if resume_process is not None and resume_process.poll() is None:
            if resume_pid is None:
                try:
                    resume_pid = _engine_child_pid(resume_process.pid)
                except (AssertionError, subprocess.SubprocessError):
                    resume_pid = None
            if resume_pid is not None:
                try:
                    identity = record_identity(
                        resume_pid, launched_after=0, command_fragment="_child.py",
                    )
                    signal_run_verified(
                        run_dir, resume_pid, identity, signal.SIGKILL,
                        deadline=time.monotonic(), what="checkpoint B resumed-run cleanup",
                        require_checkpoint_saved=False,
                    )
                except (ProcessLookupError, RuntimeError):
                    pass
            script.resumed_release.set()
            signal_run_owned(
                run_dir, resume_process, signal.SIGTERM, deadline=time.monotonic(),
                what="checkpoint B resumed-run cleanup", require_checkpoint_saved=False,
            )
            try:
                resume_process.communicate(timeout=scaled(30))
            except subprocess.TimeoutExpired:
                signal_run_owned(
                    run_dir, resume_process, signal.SIGKILL, deadline=time.monotonic(),
                    what="checkpoint B resumed-run cleanup", require_checkpoint_saved=False,
                )
                resume_process.communicate(timeout=scaled(10))
        if process is not None and process.poll() is None:
            if engine_pid is None:
                try:
                    engine_pid = _engine_child_pid(process.pid)
                except (AssertionError, subprocess.SubprocessError):
                    engine_pid = None
            if engine_pid is not None:
                try:
                    identity = record_identity(
                        engine_pid, launched_after=0, command_fragment="_child.py",
                    )
                    signal_run_verified(
                        run_dir, engine_pid, identity, signal.SIGKILL,
                        deadline=time.monotonic(), what="checkpoint B first-run cleanup",
                        require_checkpoint_saved=False,
                    )
                except (ProcessLookupError, RuntimeError):
                    pass
            script.release.set()
            signal_run_owned(
                run_dir, process, signal.SIGTERM, deadline=time.monotonic(),
                what="checkpoint B first-run cleanup", require_checkpoint_saved=False,
            )
            try:
                process.communicate(timeout=scaled(30))
            except subprocess.TimeoutExpired:
                signal_run_owned(
                    run_dir, process, signal.SIGKILL, deadline=time.monotonic(),
                    what="checkpoint B first-run cleanup", require_checkpoint_saved=False,
                )
                process.communicate(timeout=scaled(10))
        script.release.set()
        script.resumed_release.set()
        server.shutdown()
        server.server_close()


@pytest.fixture(scope="module")
def controls(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Control]:
    root = tmp_path_factory.mktemp("checkpoint-b-resume-controls")
    launcher = _write_launcher(root / "fixed-seed.py")
    profile = _write_profile(root / "checkpoint-profile.json")
    result: dict[str, Control] = {}
    for example in EXAMPLES:
        folder = root / example
        folder.mkdir()
        repo = folder / "repo"
        shutil.copytree(ROOT / "examples" / example, repo)
        home = folder / "home" / "control"
        next_request = _run_control(
            example, repo, folder / "control-run", launcher, profile, home,
        )
        result[example] = Control(repo, launcher, profile, home, next_request, folder)
    return result


@pytest.mark.host_sandbox
@pytest.mark.parametrize("example", EXAMPLES)
def test_checkpoint_b_resume_matches_uninterrupted_control(
    tmp_path: Path, controls: dict[str, Control], example: str,
) -> None:
    control = controls[example]
    outcome = _run_interrupted(
        example, control.repo, control.folder / "interrupted", control.launcher,
        control.profile, control.folder / "home" / "resumed",
    )
    assert outcome.resumed_request == outcome.pending_request, (
        "resume rebuilt different bytes from the in-flight request: "
        f"{_digest(outcome.resumed_request)} != {_digest(outcome.pending_request)}"
    )
    assert outcome.resumed_request == control.next_request, (
        "checkpoint resume differs byte-for-byte from uninterrupted control: "
        f"{_digest(outcome.resumed_request)} != {_digest(control.next_request)}"
    )
    _assert_canonical_equal(outcome.resumed_request, control.next_request)
    print(
        f"CHECKPOINT_B_FIDELITY tree={_tree_label()} example={example} "
        f"request={NEXT_REQUEST} resumed_raw_sha256={_digest(outcome.resumed_request)} "
        f"control_raw_sha256={_digest(control.next_request)} "
        f"canonical_sha256={_digest(_canonical_request(outcome.resumed_request))}"
    )


@pytest.mark.host_sandbox
def test_planted_restored_index_change_is_rejected_by_fidelity_guard(
    tmp_path: Path, controls: dict[str, Control],
) -> None:
    control = controls[EXAMPLES[0]]
    marker = tmp_path / "restored-index-mutation.json"
    hook = _write_restore_index_hook(tmp_path / "restore-hook", marker)
    outcome = _run_interrupted(
        EXAMPLES[0], control.repo, tmp_path / "mutated-resume", control.launcher,
        control.profile, tmp_path / "mutated-home", restore_hook=hook,
        expect_unfaithful_restore=True,
    )

    assert marker.is_file(), "restore hook did not mutate a restored index entry"
    mutation = json.loads(marker.read_text())
    assert mutation["entry"] == "msgs[index]", mutation
    assert 0 <= mutation["index"] < mutation["entries"], mutation
    assert mutation["old_sha256"] != mutation["new_sha256"], mutation
    assert outcome.pending_request == control.next_request, (
        "the uninterrupted and pre-restore requests should match before the planted mutation: "
        f"{_digest(outcome.pending_request)} != {_digest(control.next_request)}"
    )
    assert outcome.resumed_request != control.next_request, (
        "the restored message-history mutation did not change the next provider request"
    )
    with pytest.raises(AssertionError, match="canonical request mismatch") as mismatch:
        _assert_canonical_equal(outcome.resumed_request, control.next_request)
    print(
        f"PLANTED_RESTORE_INDEX tree={_tree_label()} example={EXAMPLES[0]} "
        f"entry=msgs[{mutation['index']}] of {mutation['entries']} "
        f"resumed_raw_sha256={_digest(outcome.resumed_request)} "
        f"control_raw_sha256={_digest(control.next_request)} "
        f"guard={mismatch.value}"
    )


def _progressing_discount_repo(path: Path) -> Path:
    shutil.copytree(ROOT / "examples/discount_bug", path)
    test_file = path / "tests/test_pricing.py"
    with test_file.open("a") as tests:
        tests.write(
            "\n\ndef test_cpu_progress():\n"
            "    import time\n"
            "    finish = time.process_time() + 2.8\n"
            "    while time.process_time() < finish:\n"
            "        pass\n"
        )
    return path


@pytest.mark.host_sandbox
def test_progressing_check_rebuilds_full_pending_request_from_earlier_checkpoint(tmp_path):
    repo = _progressing_discount_repo(tmp_path / "repo")
    launcher = _write_launcher(tmp_path / "launcher.py")
    profile = _write_profile(tmp_path / "profile.json")
    doc = json.loads(profile.read_text())
    doc["switches"]["STATEFRAME_CHECKPOINT_EVERY"] = "2"
    profile.write_text(json.dumps(doc))
    run = _run_interrupted("discount_bug", repo, tmp_path / "run", launcher,
                           profile, tmp_path / "home", completed_actions=3, checkpoint_sends=2)
    assert run.resumed_request == run.pending_request
    body = json.loads(run.resumed_request)
    assert body["tools"] and b"check-receipt:" in run.resumed_request


@pytest.mark.host_sandbox
def test_progressing_check_preserves_entire_request_bodies_and_healthy_receipts(
        tmp_path, tmp_path_factory, monkeypatch):
    from _journal import measurement_events
    from test_hook_wiring_parity import _run_checkout, _write_launchers
    from tools.pinned_tree import pinned_tree

    baseline = pinned_tree("77a95c6a8957ab46661f3620505c43894b5eaa15", tmp_path_factory)
    repo = _progressing_discount_repo(tmp_path / "repo")
    launcher, wrapper = tmp_path / "launcher.py", tmp_path / "child.py"
    _write_launchers(launcher, wrapper)
    monkeypatch.setenv("STATEFRAME_LOG_LEVEL", "debug")
    run = tmp_path / "run"

    def capture(tree, trace):
        requests, *_ = _run_checkout(tree, repo, run, launcher, wrapper, tmp_path / trace,
                                     stable_tool_head="off", final_cache_layout="off")
        audit_keys = {"timestamp_ns", "previous_hash", "record_hash", "run_id",
                      "task_id", "event_id", "sequence"}
        receipts = [{key: value for key, value in event.items() if key not in audit_keys}
                    for event in measurement_events(run) if event["kind"] == "check_receipt"]
        assert receipts and any(event["outcome"] == "PASS" for event in receipts)
        return requests, receipts

    base_requests, base_receipts = capture(baseline, "base.json")
    run.rename(tmp_path / "base")
    requests, receipts = capture(ROOT, "current.json")
    assert requests == base_requests
    assert receipts == base_receipts
    summary = json.loads((run / "summary.json").read_text())
    assert summary["problems"]["by_cause"]["degrade:execution_progress"]["count"] >= 3

"""Offline kill/resume parity probe for short and long scripted runs.

Run outside a nested sandbox with the project interpreter. Every case gets a new run
folder; no run data is removed. The output is JSON so timings and diffs can be kept.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from stateframe.replay_tape import normalized_digest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "acceptance" / "fixtures"
PYTHON = sys.executable
REPLAY_KINDS = frozenset({"checkpoint_verified", "checkpoint_unfaithful", "replay_diverged",
                          "replay_done"})


def child_env(checkpoints: bool, checkpoint_every: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT / "src" / "stateframe" / "engine")))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["STATEFRAME_CHECKPOINT"] = "save" if checkpoints else "off"
    if checkpoint_every is not None:
        env["STATEFRAME_CHECKPOINT_EVERY"] = str(checkpoint_every)
    return env


def wait_until(predicate, process: subprocess.Popen[str], seconds: float, label: str) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        if process.poll() is not None:
            output, error = process.communicate()
            raise RuntimeError(f"{label}: process exited {process.returncode}: {output[-1000:]} {error[-1000:]}")
        time.sleep(0.1)
    raise TimeoutError(label)


def start_endpoint(script: Path, *, marker: Path | None = None, pause_at: int = 2):
    if marker is None:
        command = [PYTHON, "-m", "stateframe.scripted_endpoint", "--script", str(script), "--port", "0"]
    else:
        command = [PYTHON, str(FIXTURES / "pause_endpoint.py"), "--script", str(script),
                   "--marker", str(marker), "--pause-at", str(pause_at), "--pause-seconds", "3600"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=child_env(True))
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], 15)
    if not ready:
        process.terminate()
        raise TimeoutError("endpoint did not start")
    url = process.stdout.readline().strip()
    if not url.startswith("http://127.0.0.1:") or not url.endswith("/v1"):
        process.terminate()
        raise RuntimeError(f"invalid endpoint URL: {url!r}")
    return process, url


def stop(process: subprocess.Popen[str] | None, *, engine: bool = False) -> None:
    if process is not None and process.poll() is None:
        if engine:
            try:
                os.kill(engine_child(process.pid), signal.SIGKILL)
            except (RuntimeError, OSError, subprocess.CalledProcessError):
                pass  # the child already stopped, or process inspection is unavailable
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def engine_child(parent_pid: int) -> int:
    rows = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True,
                          text=True, check=True).stdout.splitlines()
    for row in rows:
        parts = row.strip().split(maxsplit=2)
        if len(parts) == 3 and int(parts[1]) == parent_pid and "_child.py" in parts[2]:
            return int(parts[0])
    raise RuntimeError("engine child was not found")


def script_steps(steps: int, step_kind: str = "edit") -> list[dict]:
    if step_kind == "read":
        actions = [{"tool": "read", "arguments": {"path": "probe_steps.txt",
                                                 "line": index, "end_line": index}}
                   for index in range(1, steps - 1)]
        actions.append({"tool": "file_create", "arguments": {
            "path": "probe_complete.txt", "content": "completed\n"}})
    else:
        actions = [{"tool": "file_create", "arguments": {"path": f"step_{index:03d}.txt",
                                                        "content": f"completed {index}\n"}}
                   for index in range(1, steps)]
    return actions + [{"tool": "done", "arguments": {"summary": "Scripted run finished."}}]


def disk_bytes(folder: Path) -> tuple[int, int]:
    apparent = allocated = 0
    seen: set[tuple[int, int]] = set()
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        apparent += stat.st_size
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            allocated += stat.st_blocks * 512
            seen.add(identity)
    return apparent, allocated


def capture(case: Path, point: int, steps: int, checkpoints: bool,
            step_kind: str = "edit", checkpoint_every: int | None = None
            ) -> tuple[Path, bytes, dict, subprocess.Popen[str], str]:
    repo = case / "repo"
    shutil.copytree(FIXTURES / "simple_repo", repo, symlinks=True)
    if step_kind == "read":
        (repo / "probe_steps.txt").write_text(
            "".join(f"probe step {index}\n" for index in range(1, steps - 1)))
    run = case / "run"
    script = case / "original.script.json"
    script.write_text(json.dumps({"model": "scripted", "steps": script_steps(steps, step_kind),
                                  "tokenize": {"mode": "ok"}}))
    task = ("Read the probe file, create probe_complete.txt, and verify it"
            if step_kind == "read" else "Create the requested files and verify them")
    marker = case / "paused"
    endpoint = runner = None
    captured = False
    try:
        endpoint, url = start_endpoint(script, marker=marker, pause_at=point)
        env = child_env(checkpoints, checkpoint_every)
        runner = subprocess.Popen([PYTHON, "-m", "stateframe", "run", "--repo", str(repo),
                                   "--task", task, "--endpoint", url,
                                   "--model", "scripted", "--run-dir", str(run), "--test-python", PYTHON,
                                   "--test-cmd", "python -m unittest discover -s tests -q",
                                   "--max-turns", str(steps * 4 + 10), "--shell", "off", "--quiet"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, env=env)
        wait_until(marker.exists, runner, 1800, f"request {point} not reached")
        if checkpoints:
            pointer = run / "checkpoints" / "LATEST"
            wait_until(pointer.is_file, runner, 180, "checkpoint not durable")
        request = (run / "wire" / f"{point:04d}.request.json").read_bytes()
        os.kill(engine_child(runner.pid), signal.SIGKILL)
        runner.communicate(timeout=40)
        checkpoint = {}
        if checkpoints:
            pointer = run / "checkpoints" / "LATEST"
            folder = pointer.parent / pointer.read_text().strip()
            apparent, allocated = disk_bytes(pointer.parent)
            checkpoint = {"step": json.loads((folder / "checkpoint.json").read_text())["step"],
                          "state_bytes": (folder / "state.pkl").stat().st_size,
                          "checkpoint_dir_apparent_bytes": disk_bytes(folder)[0],
                          "checkpoint_root_apparent_bytes": apparent,
                          "checkpoint_root_allocated_upper_bytes": allocated}
        captured = True
        return run, request, checkpoint, endpoint, url
    finally:
        stop(runner, engine=True)
        if not captured:
            stop(endpoint)


def first_field(original: object, restored: object, path: str = "$") -> str | None:
    if type(original) is not type(restored):
        return path
    if isinstance(original, dict):
        for key in dict.fromkeys((*original, *restored)):
            if key not in original or key not in restored:
                return f"{path}.{key}"
            difference = first_field(original[key], restored[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(original, list):
        if len(original) != len(restored):
            return f"{path}.length"
        for index, (left, right) in enumerate(zip(original, restored)):
            difference = first_field(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    return path if original != restored else None


def set_aside_request(case: Path, run: Path, point: int, original: bytes) -> Path:
    wire = run / "wire"
    request = wire / f"{point:04d}.request.json"
    if request.read_bytes() != original:
        raise RuntimeError("original request changed before resume")
    saved_dir = case / "original-wire"
    saved_dir.mkdir()
    for artifact in sorted(wire.glob(f"{point:04d}.*")):
        saved = saved_dir / artifact.name
        artifact.replace(saved)
        # Resume needs the pending request in its archived wire tape for checkpoint selection.
        os.link(saved, artifact)
    return saved_dir / request.name


def event_rows(path: Path, field: str) -> list[str]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if field == "kind" and line.startswith("[sf] "):
            line = line[5:]
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and isinstance(record.get(field), str):
            rows.append(record[field])
    return rows


def resume(case: Path, run: Path, point: int, original: bytes, checkpoints: bool,
           endpoint: subprocess.Popen[str], url: str, checkpoint_every: int | None = None) -> dict:
    saved = set_aside_request(case, run, point, original)
    runner = None
    try:
        started = time.monotonic()
        runner = subprocess.Popen([PYTHON, "-m", "stateframe", "resume", str(run),
                                   "--endpoint", url, "--model", "scripted", "--quiet",
                                   "--keep-checkpoints"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env=child_env(checkpoints, checkpoint_every))
        rebuilt = run / "wire" / f"{point:04d}.request.json"
        response = run / "wire" / f"{point:04d}.response.json"
        actual = {}

        def rebuilt_ready() -> bool:
            try:
                if not rebuilt.is_file() or rebuilt.samefile(saved):
                    return False
                wire = rebuilt.read_bytes()
                json.loads(wire)
            except (OSError, ValueError):
                return False
            actual["wire"] = wire
            return True

        wait_until(rebuilt_ready, runner, 1800, f"resumed request {point} not rebuilt")
        duration = time.monotonic() - started
        # The engine writes the response only after closing the request file.
        wait_until(response.is_file, runner, 1800, f"resumed request {point} not complete")
        if checkpoints:
            # The parent flushes child events to engine.log on its reader thread.
            deadline = time.monotonic() + 2
            while runner.poll() is None and time.monotonic() < deadline:
                if any(kind in {"checkpoint_verified", "checkpoint_unfaithful", "replay_diverged"}
                       for kind in event_rows(run / "engine.log", "kind")):
                    break
                time.sleep(0.05)
        stop(runner, engine=True)
        kinds = [kind for kind in event_rows(run / "engine.log", "kind") if kind in REPLAY_KINDS]
        attempt_kinds = {path.parent.name: [kind for kind in event_rows(path, "kind")
                                            if kind in REPLAY_KINDS]
                         for path in (run / "attempts").glob("*/engine.log")}
        checkpoint_events = event_rows(run / "checkpoints" / "events.jsonl", "event")
        attempt_checkpoint_events = {
            path.parent.parent.name: event_rows(path, "event")
            for path in (run / "attempts").glob("*/checkpoints/events.jsonl")}
        all_kinds = kinds + [kind for events in attempt_kinds.values() for kind in events]
        all_checkpoint_events = (checkpoint_events
                                 + [event for events in attempt_checkpoint_events.values() for event in events])
        spec = json.loads((run / "spec.json").read_text())
        fallback = bool(spec.get("resume", {}).get("checkpoint_refused"))
        divergence = run / "resume-divergence.json"
        attempt_divergences = {path.parent.name: json.loads(path.read_text())
                               for path in (run / "attempts").glob("*/resume-divergence.json")}
        return {"request": point, "checkpoint": checkpoints, "resume_exit_code": runner.returncode,
                "seconds_to_request": round(duration, 3),
                "raw_identical": original == actual["wire"],
                "normalized_identical": normalized_digest(original) == normalized_digest(actual["wire"]),
                "first_field": first_field(json.loads(original), json.loads(actual["wire"])),
                "engine_kinds": kinds, "attempt_engine_kinds": attempt_kinds,
                "checkpoint_events": checkpoint_events,
                "attempt_checkpoint_events": attempt_checkpoint_events,
                "checkpoint_restored": "checkpoint_restored" in checkpoint_events,
                "checkpoint_verified": "checkpoint_verified" in kinds,
                "checkpoint_refused": ("checkpoint_restore_refused" in all_checkpoint_events
                                       or "checkpoint_unfaithful" in all_kinds),
                "full_replay_fallback": fallback,
                "divergence": json.loads(divergence.read_text()) if divergence.is_file() else None,
                "attempt_divergences": attempt_divergences}
    finally:
        stop(runner, engine=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--point", action="append", type=int, dest="points")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--step-kind", choices=("edit", "read"), default="edit")
    parser.add_argument("--compare-full", action="store_true")
    parser.add_argument("--checkpoint-every", type=int,
                        help="override checkpoint cadence in engine steps (default: product setting, 1)")
    args = parser.parse_args()
    if args.checkpoint_every is not None and args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    points = args.points or [2, 30, 110]
    steps = args.steps if args.steps is not None else max(points) + 2
    if steps <= max(points):
        parser.error("--steps must be greater than every --point")
    for point in points:
        if point < 2:
            parser.error("each point must be at least 2")
        for checkpoints in ([True, False] if args.compare_full else [True]):
            case = Path(tempfile.mkdtemp(prefix=f"checkpoint-{point}-", dir=args.output))
            run, original, disk, endpoint, url = capture(
                case, point, steps, checkpoints, args.step_kind, args.checkpoint_every)
            try:
                row = {**resume(case, run, point, original, checkpoints, endpoint, url,
                                args.checkpoint_every),
                       **disk, "run_dir": str(run)}
            finally:
                stop(endpoint)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    return 0 if all(row["normalized_identical"] and
                    (not row["checkpoint"] or (row["checkpoint_restored"] and row["checkpoint_verified"]
                                                and not row["full_replay_fallback"]))
                    for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

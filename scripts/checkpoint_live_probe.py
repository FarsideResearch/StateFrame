"""Measure a real interrupted run, checkpoint restore, and full replay.

Each attempt uses an existing coding fixture on a copied working tree. The
interrupted run and both resume attempts are retained under the output folder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from checkpoint_restore_probe import (
    disk_bytes,
    engine_child,
    event_rows,
    first_field,
    set_aside_request,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("checkpoint-live-probe")
POLL_SECONDS = 0.05
MAX_WAIT_SECONDS = 7200


def environment(*, interval: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if (
            name.startswith("STATEFRAME_CHECKPOINT")
            or name == "STATEFRAME_KEEP_CHECKPOINTS"
        ):
            env.pop(name)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(ROOT / "src" / "stateframe" / "engine"))
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if interval is not None:
        env["STATEFRAME_CHECKPOINT_EVERY"] = str(interval)
    return env


def launch(
    command: list[str], log: Path, env: dict[str, str]
) -> subprocess.Popen[bytes]:
    with log.open("wb") as output:
        return subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )


def kill_run(runner: subprocess.Popen[bytes]) -> None:
    if runner.poll() is None:
        try:
            os.killpg(engine_child(runner.pid), signal.SIGKILL)
        except (OSError, RuntimeError, subprocess.CalledProcessError):
            pass
        try:
            os.kill(runner.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    runner.wait(timeout=30)


def checkpoint_state(run: Path) -> dict | None:
    pointer = run / "checkpoints" / "LATEST"
    if not pointer.is_file():
        return None
    name = pointer.read_text().strip()
    if not name or Path(name).name != name:
        raise RuntimeError("invalid checkpoint pointer")
    folder = pointer.parent / name
    meta = json.loads((folder / "checkpoint.json").read_text())
    state = folder / "state.pkl"
    if state.stat().st_size != meta["state_bytes"]:
        raise RuntimeError("checkpoint state size differs from metadata")
    try:
        from stateframe.engine.kernel_measurement import open_journal

        with open_journal(pointer.parent) as stream:
            lines = stream.read().splitlines()
    except FileNotFoundError:
        return None
    saved = []
    for line in lines:
        try:
            saved.append(json.loads(line))
        except ValueError:
            continue  # a concurrent journal append may be partial
    if not any(
        row.get("event") == "checkpoint_saved" and row.get("step") == meta["step"]
        for row in saved
    ):
        return (
            None  # the writer has published LATEST but has not journaled the save yet
        )
    return {
        "folder": str(folder),
        "step": meta["step"],
        "sends": meta["sends"],
        "state_bytes": meta["state_bytes"],
        "disk_bytes": disk_bytes(folder)[0],
        "written": True,
    }


def pending_request(run: Path, minimum: int) -> int | None:
    for path in sorted((run / "wire").glob("*.request.json"), reverse=True):
        try:
            number = int(path.name.split(".", 1)[0])
        except ValueError:
            continue
        if (
            number >= minimum
            and not path.with_name(f"{number:04d}.response.json").exists()
        ):
            try:
                json.loads(path.read_bytes())
            except (OSError, ValueError):
                continue
            return number
    return None


def interrupt(
    case: Path, endpoint: str, model: str, interval: int | None, fixture: Path
) -> tuple[Path, dict, int]:
    run = case / "run"
    repo = case / "repo"
    shutil.copytree(fixture, repo)
    task = (repo / "TASK.md").read_text()
    command = [
        sys.executable,
        "-m",
        "stateframe",
        "run",
        "--repo",
        str(repo),
        "--task",
        task,
        "--endpoint",
        endpoint,
        "--model",
        model,
        "--run-dir",
        str(run),
        "--test-python",
        sys.executable,
        "--max-turns",
        "90",
        "--shell",
        "off",
        "--quiet",
    ]
    runner = launch(command, case / "initial.log", environment(interval=interval))
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    checkpoint = None
    try:
        while time.monotonic() < deadline:
            if checkpoint is None or not Path(checkpoint["folder"]).is_dir():
                checkpoint = checkpoint_state(run)
            if checkpoint:
                # Hold this durable checkpoint while waiting for its first later request.
                point = pending_request(run, checkpoint["sends"] + 1)
                if point is not None:
                    first = run / "wire" / f"{checkpoint['sends'] + 1:04d}.request.json"
                    if not first.is_file():
                        raise RuntimeError(
                            "first request after checkpoint was not recorded"
                        )
                    kill_run(runner)
                    if not Path(checkpoint["folder"]).is_dir():
                        raise RuntimeError("chosen checkpoint was pruned before interruption")
                    if (run / "wire" / f"{point:04d}.response.json").exists():
                        raise RuntimeError("kill point completed before interruption")
                    return run, checkpoint, point
            if runner.poll() is not None:
                raise RuntimeError(
                    "run finished before a checkpoint and later pending request"
                )
            time.sleep(POLL_SECONDS)
        raise TimeoutError("run did not reach a checkpoint and later pending request")
    finally:
        if runner.poll() is None:
            kill_run(runner)


def rebuilt_request(
    run: Path, saved: Path, index: int, runner: subprocess.Popen[bytes], started: float
) -> tuple[bytes, float]:
    target = run / "wire" / f"{index:04d}.request.json"
    deadline = started + MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            if target.is_file() and not target.samefile(saved):
                wire = target.read_bytes()
                json.loads(wire)
                return wire, round(time.monotonic() - started, 3)
        except (OSError, ValueError):
            pass
        if runner.poll() is not None:
            raise RuntimeError(f"resume exited before rebuilding request {index}")
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"resume did not rebuild request {index}")


def resume_command(run: Path, endpoint: str, model: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "stateframe",
        "resume",
        str(run),
        "--endpoint",
        endpoint,
        "--model",
        model,
        "--quiet",
        "--keep-checkpoints",
    ]


def replay_control(
    case: Path,
    interrupted: Path,
    index: int,
    original: bytes,
    endpoint: str,
    model: str,
    *,
    source: Path | None = None,
    label: str = "full-replay-control",
) -> tuple[Path, float, bool]:
    held = case / f"held-{label}"
    control = case / label
    if held.exists() or control.exists():
        raise RuntimeError("control paths already exist; use a fresh case")
    # Run records include their absolute directory. Temporarily put the control
    # copy at that path and retain both trees after the measurement.
    interrupted.rename(held)
    try:
        shutil.copytree(
            source or held, interrupted, ignore=shutil.ignore_patterns("checkpoints")
        )
        spec_path = interrupted / "spec.json"
        spec = json.loads(spec_path.read_text())
        spec["environment"]["STATEFRAME_CHECKPOINT"] = "off"
        spec_path.write_text(json.dumps(spec, indent=1, sort_keys=True))
        saved_root = Path(tempfile.mkdtemp(prefix="control-original-", dir=case))
        saved = set_aside_request(saved_root, interrupted, index, original)
        started = time.monotonic()
        runner = launch(
            resume_command(interrupted, endpoint, model),
            case / f"{label}.log",
            environment(),
        )
        try:
            rebuilt, seconds = rebuilt_request(
                interrupted, saved, index, runner, started
            )
            return control, seconds, rebuilt == original
        finally:
            kill_run(runner)
    finally:
        if interrupted.exists():
            interrupted.rename(control)
        held.rename(interrupted)


def restore_and_finish(
    case: Path, run: Path, index: int, original: bytes, endpoint: str, model: str
) -> dict:
    saved_root = case / "checkpoint-original"
    saved_root.mkdir()
    saved = set_aside_request(saved_root, run, index, original)
    started = time.monotonic()
    runner = launch(
        resume_command(run, endpoint, model),
        case / "checkpoint-resume.log",
        environment(),
    )
    try:
        rebuilt, seconds = rebuilt_request(run, saved, index, runner, started)
        returncode = runner.wait(timeout=MAX_WAIT_SECONDS)
    finally:
        if runner.poll() is None:
            kill_run(runner)
    kinds = event_rows(run / "engine.log", "kind")
    events = event_rows(run / "checkpoints" / "events.jsonl", "event")
    spec = json.loads((run / "spec.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    return {
        "seconds_to_request": seconds,
        "raw_identical": rebuilt == original,
        "first_different_field": first_field(json.loads(original), json.loads(rebuilt)),
        "restored": "checkpoint_restored" in events,
        "verified": "checkpoint_verified" in kinds,
        "unfaithful": "checkpoint_unfaithful" in kinds,
        "full_replay_fallback": bool(spec.get("resume", {}).get("checkpoint_refused")),
        "returncode": returncode,
        "outcome": summary.get("runtime_outcome"),
        "checks": summary.get("check_outcome"),
        "tests_passed": summary.get("tests_after", {}).get("passed"),
        "total_requests": summary.get("requests"),
        "succeeded": summary.get("succeeded"),
    }


def report(row: dict, output: Path) -> None:
    (output / "proof.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    cells = [
        ("Checkpoint step", row["checkpoint_step"]),
        ("Kill request", row["kill_request"]),
        ("Written state bytes", row["written_state_bytes"]),
        ("Restored", row["restored"]),
        ("Verified", row["verified"]),
        ("Raw identical", row["raw_identical"]),
        ("Checkpoint seconds", row["seconds_to_request_checkpoint"]),
        ("Full replay seconds", row["seconds_to_request_full_replay"]),
        ("Speedup", f"{row['speedup']}x"),
        ("Final outcome", row["final_outcome"]),
        ("Tests passed", row["tests_passed"]),
        ("Total requests", row["total_requests"]),
    ]
    table = "| Measure | Result |\n|---|---:|\n" + "".join(
        f"| {key} | {value} |\n" for key, value in cells
    )
    (output / "proof.md").write_text(table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Existing coding fixture containing TASK.md and tests/",
    )
    parser.add_argument(
        "--continue-case",
        type=Path,
        help="Continue an interrupted case without repeating the initial run",
    )
    parser.add_argument(
        "--remeasure-control",
        type=Path,
        help="Remeasure full replay from a preserved interrupted snapshot",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not (args.repo / "TASK.md").is_file() or not (args.repo / "tests").is_dir():
        parser.error("--repo must contain TASK.md and tests/")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.remeasure_control:
        case = args.remeasure_control
        row = json.loads((args.output / "proof.json").read_text())
        source = case / "full-replay"
        index = int(row["first_rebuilt_request"])
        original = (source / "wire" / f"{index:04d}.request.json").read_bytes()
        control, seconds, identical = replay_control(
            case,
            case / "run",
            index,
            original,
            args.endpoint,
            args.model,
            source=source,
            label="full-replay-no-save",
        )
        row.update(
            {
                "full_replay_dir": str(control),
                "full_replay_raw_identical": identical,
                "seconds_to_request_full_replay": seconds,
                "speedup": round(seconds / row["seconds_to_request_checkpoint"], 2),
                "control_checkpoint_saving_off": True,
            }
        )
        report(row, args.output)
        print(
            json.dumps(
                {
                    "full_replay_raw_identical": identical,
                    "seconds_to_request_full_replay": seconds,
                    "speedup": row["speedup"],
                }
            )
        )
        return 0 if identical else 1
    case = args.continue_case or Path(
        tempfile.mkdtemp(prefix="deadline-", dir=args.output)
    )
    interval = None
    try:
        if args.continue_case:
            run = case / "run"
            checkpoint = checkpoint_state(run)
            if checkpoint is None:
                raise RuntimeError("interrupted case has no durable checkpoint")
            point = pending_request(run, checkpoint["sends"] + 1)
            if point is None:
                raise RuntimeError("interrupted case has no later pending request")
            spec = json.loads((run / "spec.json").read_text())
            recorded_interval = int(
                spec["environment"].get("STATEFRAME_CHECKPOINT_EVERY", 1)
            )
            interval = None if recorded_interval == 1 else recorded_interval
        else:
            run, checkpoint, point = interrupt(
                case, args.endpoint, args.model, interval, args.repo
            )
        effective_interval = int(json.loads((run / "spec.json").read_text())["environment"].get(
            "STATEFRAME_CHECKPOINT_EVERY", 1))
        index = checkpoint["sends"] + 1
        original = (run / "wire" / f"{index:04d}.request.json").read_bytes()
        control, replay_seconds, replay_identical = replay_control(
            case, run, index, original, args.endpoint, args.model
        )
        resumed = restore_and_finish(
            case, run, index, original, args.endpoint, args.model
        )
        row = {
            "fixture": str(args.repo),
            "case_dir": str(case),
            "run_dir": str(run),
            "full_replay_dir": str(control),
            "checkpoint_interval_steps": effective_interval,
            "default_interval": interval is None,
            "checkpoint_step": checkpoint["step"],
            "checkpoint_sends": checkpoint["sends"],
            "kill_request": point,
            "first_rebuilt_request": index,
            "written": checkpoint["written"],
            "written_state_bytes": checkpoint["state_bytes"],
            "checkpoint_disk_bytes": checkpoint["disk_bytes"],
            "restored": resumed["restored"],
            "verified": resumed["verified"],
            "raw_identical": resumed["raw_identical"],
            "first_different_field": resumed["first_different_field"],
            "unfaithful": resumed["unfaithful"],
            "full_replay_fallback": resumed["full_replay_fallback"],
            "full_replay_raw_identical": replay_identical,
            "control_checkpoint_saving_off": True,
            "seconds_to_request_checkpoint": resumed["seconds_to_request"],
            "seconds_to_request_full_replay": replay_seconds,
            "speedup": round(replay_seconds / resumed["seconds_to_request"], 2)
            if resumed["seconds_to_request"]
            else None,
            "final_outcome": resumed["outcome"],
            "checks": resumed["checks"],
            "tests_passed": resumed["tests_passed"],
            "total_requests": resumed["total_requests"],
            "succeeded": resumed["succeeded"],
            "resume_returncode": resumed["returncode"],
        }
        report(row, args.output)
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "checkpoint_step",
                        "kill_request",
                        "written_state_bytes",
                        "restored",
                        "verified",
                        "raw_identical",
                        "seconds_to_request_checkpoint",
                        "seconds_to_request_full_replay",
                        "final_outcome",
                        "total_requests",
                    )
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return (
            0
            if (
                row["written"]
                and row["restored"]
                and row["verified"]
                and row["raw_identical"]
                and replay_identical
                and not row["unfaithful"]
                and not row["full_replay_fallback"]
                and row["succeeded"]
            )
            else 1
        )
    except Exception as error:
        print(
            f"proof incomplete: {type(error).__name__}: {error}; artifacts: {case}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

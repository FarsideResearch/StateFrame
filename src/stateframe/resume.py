"""``stateframe resume <run-dir>``: decide whether an interrupted run can continue, and set it up.

A run stopped before it finished (Ctrl-C, ``kill``, ``kill -9``, a laptop that slept or crashed, a model
server that went away) keeps everything resume needs in its run directory: the frozen run spec (engine
settings, task, endpoint), the read-only start snapshot ``parent/``, and every model exchange under
``wire/``. Resume moves that attempt's files to ``attempts/<n>/`` (nothing is deleted), gives the engine a
fresh working copy of the start state, and restarts the SAME engine with the SAME settings. The engine is
answered from the recorded responses until it reaches the point where the attempt stopped
(``replay_tape``), then continues live. Requests stay frame-only throughout: no history is added.

Resume refuses, with the reason, when the run cannot continue truthfully: a different engine release is
installed, the run already finished, its records are damaged, an engine is still running on it, or the
last attempt stopped for a reason that replay would only reproduce.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import outcomes, profiles, replay_tape, workcopy
from .engine.kernel_tool_tape import ToolTape, ToolTapeError

ENGINE_LOCK = ".engine.lock"
RUN_LOCK = ".run.lock"
# Per-attempt files: moved to attempts/<n>/ on resume, so the resumed attempt writes its own.
ATTEMPT_FILES = ("worker", "wire", "requests", "sends.jsonl", "engine.log", "engine-result.json",
                 "child-traceback.txt", "summary.json", "final.diff", "tests-after.txt",
                 "resume-divergence.json", "resume-pending-verified.json", "workspace", "checks",
                 "checkpoint-refused.json",
                 "env", "env-sites.json", "shell-home", "shell-cache",
                 "shell-ask.json", "shell-answer.json")


CHECKPOINT_SCHEMA = "stateframe-engine-checkpoint/1"


class ResumeRefused(RuntimeError):
    """The run directory cannot be resumed; the message says why and what to do."""


def _lock_holder(path: Path) -> str | None:
    """The pid recorded in a lock file that some live process still holds, else None."""
    if not path.is_file():
        return None
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.seek(0)
            return handle.read().strip() or "unknown"
        fcntl.flock(handle, fcntl.LOCK_UN)
    return None


def hold_run_lock(run: Path):
    """The runner's lock for one ``run``/``resume`` on this directory (released by the OS on exit)."""
    handle = (run / RUN_LOCK).open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise ResumeRefused("this run directory is in use by another stateframe process") from None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()) + "\n")
    handle.flush()
    return handle


def _journal_lineage_ok(run: Path) -> str | None:
    """Structural check of the engine journal: contiguous sequence and hash links.

    A torn final record (the process was killed while appending) is expected after a hard kill.
    Returns a problem description, or None.
    """
    journals = sorted((run / "worker" / "measurement-runs").glob("*/events.jsonl"))
    if len(journals) > 1:
        return "more than one engine journal in worker/measurement-runs"
    if not journals:
        return None  # stopped before the engine opened its journal
    lines = journals[0].read_bytes().split(b"\n")
    previous, expected = None, 1
    for number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            if number == len(lines):
                return None  # torn tail
            return f"journal record {number} is not valid JSON"
        if (not isinstance(record, dict) or record.get("sequence") != expected
                or record.get("previous_hash") != previous or not isinstance(record.get("record_hash"), str)):
            return f"journal lineage broken at record {number}"
        previous, expected = record["record_hash"], expected + 1
    return None


def carry_forward_records(spec: dict, own_tape: list[dict[str, object]],
                          own_pending: dict[str, object] | None, wire_dir: Path, *, diverged: bool,
                          restart_live: bool = False) -> tuple[list[dict[str, object]],
                                                                dict[str, object] | None, list[dict[str, object]]]:
    """Select the completed tape, pending check, and route moves for the next attempt.

    A faithful replay keeps the earlier unfinished tail. After replaying all carried responses, the
    earlier pending check and route moves survive until the child durably records a matching passing
    comparison. A divergence or live restart makes the current attempt's records authoritative.
    """
    if restart_live:
        return [], None, []
    prior = spec.get("resume") or {}
    previous = prior.get("tape")
    prior_pending = prior.get("pending") if spec.get("routes") else None
    prior_route_changes = (prior.get("route_changes") or []) if spec.get("routes") else []
    faithful = (not diverged and isinstance(previous, list) and len(own_tape) <= len(previous)
                and all(own.get("request_digest") == old.get("request_digest")
                        for own, old in zip(own_tape, previous)))
    if not faithful:
        return own_tape, own_pending, []
    if len(own_tape) < len(previous):
        return own_tape + previous[len(own_tape):], prior_pending, prior_route_changes
    if isinstance(prior_pending, dict):
        marker = wire_dir.parent / "resume-pending-verified.json"
        verified = None
        if marker.is_file():
            try:
                verified = json.loads(marker.read_bytes())
            except (OSError, ValueError):
                pass
        if (not isinstance(verified, dict) or verified.get("send") != prior_pending.get("send")
                or verified.get("recorded_route") != prior_pending.get("route")
                or not isinstance(verified.get("route"), dict)
                or not replay_tape.Replayer([], route_changes=prior_route_changes)._route_matches(
                    prior_pending["route"], verified["route"])):
            return own_tape, prior_pending, prior_route_changes
        request_path = prior_pending.get("request")
        try:
            request_sha256 = (hashlib.sha256(Path(request_path).read_bytes()).hexdigest()
                              if isinstance(request_path, str) else None)
        except OSError:
            return own_tape, prior_pending, prior_route_changes
        if verified.get("request_sha256") != request_sha256:
            return own_tape, prior_pending, prior_route_changes
    return own_tape, own_pending, []


def model_swap_for(spec: dict, model: str | None) -> dict[str, object] | None:
    """The model-name swap a resume runs under, or None.

    ``model`` is the name the endpoint serves now (``stateframe resume --model``); the run's records were made
    under ``spec["model"]`` and, after an earlier swapped resume, partly under older names. Replay sets the
    request's model field back to the name each recorded request carries, and only to one of these names.
    """
    current = str(spec["model"])
    served = current if model is None else model
    previous = ((spec.get("resume") or {}).get("model_swap") or {}).get("recorded") or []
    recorded = sorted(({str(name) for name in previous} | {current}) - {served})
    return {"served": served, "recorded": recorded} if recorded else None


def finished_for_good(run: Path) -> bool:
    """True when resume would refuse this run whatever the user passes: it submitted its result, or it
    stopped on a seal failure that would only repeat. A run stopped at its step limit is NOT finished (a
    larger --max-turns continues it), nor is a cancelled, killed or crashed one."""
    status = outcomes.terminal_receipt(run).get("result")
    return status == "SUBMITTED" or status in outcomes.SEAL_FAILURES


def _tool_safety(run: Path, spec: dict, *, replay_external: bool) -> list[dict[str, str]]:
    """Check the live tape and prior external effects before any attempt is archived."""
    try:
        tape = ToolTape(run)
    except (OSError, ValueError, ToolTapeError) as error:
        raise ResumeRefused(f"tool tape is damaged: {error}") from error
    pending = tape.pending()
    if pending:
        calls = ", ".join(item.call_id for item in pending)
        raise ResumeRefused(f"operation {calls} may or may not have happened; reconcile and start a new run")
    resume_record = spec.get("resume")
    if resume_record is not None and not isinstance(resume_record, dict):
        raise ResumeRefused("resume metadata is damaged")
    prior = (resume_record or {}).get("archived_external_operations", [])
    if not isinstance(prior, list) or any(
            not isinstance(item, dict) or set(item) != {"call_id", "tool"}
            or not all(isinstance(value, str) and value for value in item.values()) for item in prior):
        raise ResumeRefused("archived external operation metadata is damaged")
    archived = []
    for folder in sorted((p for p in (run / "attempts").glob("[0-9]*") if p.name.isdigit()),
                         key=lambda p: int(p.name)):
        if not (folder / "tool-tape.jsonl").exists():
            continue
        try:
            old = ToolTape(folder)
        except (OSError, ValueError, ToolTapeError) as error:
            raise ResumeRefused(f"archived tool tape is damaged in {folder.name}: {error}") from error
        if old.pending():
            calls = ", ".join(item.call_id for item in old.pending())
            raise ResumeRefused(f"operation {calls} may or may not have happened; reconcile and start a new run")
        archived.extend({"call_id": item.call_id, "tool": item.tool}
                        for item in old.records() if item.replay_class == "external")
    if prior != archived:
        raise ResumeRefused("archived external operation metadata differs from the saved tool tapes")
    external = [*archived, *({"call_id": item.call_id, "tool": item.tool}
                             for item in tape.records() if item.replay_class == "external")]
    if external and not replay_external:
        names = ", ".join(f"{item['call_id']} ({item['tool']})" for item in external)
        raise ResumeRefused(f"external operation {names} requires --replay-external")
    return external


def assess(run: Path, *, max_turns: int | None = None, model: str | None = None,
           replay_external: bool = False) -> dict[str, object]:
    """Everything a resume needs, or ResumeRefused with the reason."""
    run = run.resolve()
    spec_path = run / "spec.json"
    if not spec_path.is_file():
        raise ResumeRefused(f"not a stateframe run directory (no spec.json): {run}")
    try:
        spec = json.loads(spec_path.read_text())
    except (OSError, ValueError) as error:
        raise ResumeRefused(f"spec.json is damaged: {error}") from None
    if not isinstance(spec, dict) or spec.get("schema") != "stateframe-run/1":
        raise ResumeRefused("spec.json is not a stateframe-run/1 run spec")
    for key in ("environment", "public_row", "parent_snapshot", "chat_url", "model", "max_turns", "repo"):
        if key not in spec:
            raise ResumeRefused(f"spec.json lacks {key!r}; this run was made by an older stateframe")
    external = _tool_safety(run, spec, replay_external=True)
    recorded = (spec.get("engine_source") or {}).get("shipped_hash")
    installed = profiles.engine_source_summary().get("shipped_hash")
    if not recorded or not installed or recorded != installed:
        raise ResumeRefused(
            f"this run was made by engine release {str(recorded)[:12] or 'unknown'}, the installed engine is "
            f"{str(installed)[:12] or 'unknown'}; resume replays the run on the same engine, so it needs the "
            "release that started it")
    holder = _lock_holder(run / ENGINE_LOCK)
    if holder is not None:
        raise ResumeRefused(f"an engine is still running on this run directory (pid {holder}); "
                            "stop it before resuming")
    if _lock_holder(run / RUN_LOCK) is not None:
        raise ResumeRefused("this run directory is in use by another stateframe process")
    parent = run / "parent"
    if not parent.is_dir():
        raise ResumeRefused("the run's start snapshot parent/ is missing; this run cannot be resumed")
    if Path(spec["parent_snapshot"]).resolve() != parent.resolve():
        original = Path(spec["parent_snapshot"]).parent
        raise ResumeRefused(f"this run directory was moved: it was created at {original}, and its records "
                            f"use that path. Move it back to {original} (or copy it there) and resume from there")

    terminal = outcomes.terminal_receipt(run)
    status = terminal.get("result")
    if status == "SUBMITTED":
        raise ResumeRefused("this run already finished: the engine submitted its result "
                            "(see summary.json and final.diff); start a new run for more work")
    if status in outcomes.SEAL_FAILURES:
        raise ResumeRefused(f"the last attempt stopped with {status} while preparing a request; "
                            "replaying it on the same engine would stop at the same point")
    previous_turns = int(spec["max_turns"])
    turns = previous_turns if max_turns is None else max_turns
    if status in outcomes.BUDGET_STOPS | outcomes.INCOMPLETE and not (
            turns == 0 or (previous_turns != 0 and turns > previous_turns)):
        raise ResumeRefused(f"the last attempt stopped at its limit ({status}, step limit "
                            f"{previous_turns or 'none'}); pass a larger --max-turns (0 = no limit) to continue; "
                            "the run then restarts live from request 1, because the limit is part of the first request")
    if max_turns is not None and turns != previous_turns and status not in outcomes.BUDGET_STOPS | outcomes.INCOMPLETE:
        # The step limit is part of request 1: a different one cannot replay the recorded run. Only a run that
        # stopped AT its limit may raise it (and then restarts live from request 1, with a warning).
        def name(limit: int) -> str:
            return "no limit (0)" if limit == 0 else str(limit)
        raise ResumeRefused(f"this run was started with --max-turns {name(previous_turns)}; resume keeps the original "
                            f"budget (drop --max-turns, or pass --max-turns {previous_turns}). Only a run that stopped "
                            "at its step limit can continue with a larger one")
    if external and not replay_external:
        names = ", ".join(f"{item['call_id']} ({item['tool']})" for item in external)
        raise ResumeRefused(f"external operation {names} requires --replay-external")
    problem = _journal_lineage_ok(run)
    if problem:
        raise ResumeRefused(f"the engine journal is damaged ({problem}); this run cannot be resumed")
    try:
        own_tape = replay_tape.build(run / "wire", run / "sends.jsonl",
                                     routes_required=bool(spec.get("routes")))
        tape, _, _ = carry_forward_records(spec, own_tape, None, run / "wire",
                                           diverged=(run / "resume-divergence.json").exists())
        if spec.get("routes"):
            replay_tape.pending_request(run / "wire", routes_required=True)
    except (OSError, replay_tape.TapeError) as error:
        raise ResumeRefused(f"the recorded model exchanges are damaged ({error}); "
                            "this run cannot be resumed") from None
    attempts = sorted(int(p.name) for p in (run / "attempts").glob("[0-9]*") if p.name.isdigit()) \
        if (run / "attempts").is_dir() else []
    return {"run": run, "spec": spec, "tape_count": len(tape), "previous_status": status,
            "previous_error": terminal.get("error"), "max_turns": turns,
            "external_operations": external,
            "model_swap": model_swap_for(spec, model),
            "attempt": (attempts[-1] if attempts else 0) + 1}


def select_checkpoint(run: Path, tape: list[dict[str, object]], wire_dir: Path) -> dict[str, object] | None:
    """The newest intact engine checkpoint whose next request was recorded (so it can be verified).

    A checkpoint (``checkpoints/<step>/``, written by the engine about once every 5 requests:
    STATEFRAME_CHECKPOINT_EVERY=15, counted in engine steps) holds the
    engine's whole state at a step boundary plus a copy of the run files. Resuming from it skips the
    replay of everything before it; the recorded exchanges after it are replayed strictly, and the
    first rebuilt request must match the recorded one byte for byte (apart from run ids and
    durations), or the resume falls back to replaying the whole run. A checkpoint with no recorded
    request after it cannot be verified that way and is skipped. Returns None when none qualifies.
    """
    root = run / "checkpoints"
    # Selection runs after the stopped attempt is archived, so an I/O failure listing the checkpoints
    # (permissions, a vanished mount) must mean "no usable checkpoint" (full replay), never a crash.
    try:
        if not root.is_dir():
            return None
        folders = [folder for folder in root.iterdir() if folder.is_dir() and not folder.name.startswith(".")]
    except OSError:
        return None
    pending = replay_tape.pending_request(wire_dir)
    recorded = {int(item["send"]) for item in tape}
    candidates = []
    for folder in folders:
        try:
            meta = json.loads((folder / "checkpoint.json").read_text())
        except (OSError, ValueError):
            continue
        if isinstance(meta, dict):
            # Newest = highest engine "seq" (only grows within a run; a wall clock can step back). Checkpoints
            # without seq sort as oldest, the same rule as the engine's own checkpoint list.
            seq = meta.get("seq") if type(meta.get("seq")) is int else -1
            candidates.append((seq, meta.get("created") or 0, folder.name, folder, meta))
    for *_, folder, meta in sorted(candidates, key=lambda row: row[:3], reverse=True):  # newest first
        try:
            if meta.get("schema") != CHECKPOINT_SCHEMA or type(meta.get("sends")) is not int:
                continue
            state = folder / "state.pkl"
            if state.stat().st_size != meta.get("state_bytes"):
                continue
            digest = hashlib.sha256()
            with state.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != meta.get("state_sha256"):
                continue
            if not _blobs_intact(folder, meta):
                continue  # the engine would refuse it (blob manifest / pool); try the next-older one
        except (OSError, ValueError):
            continue
        sends = meta["sends"]
        tail = sorted(n for n in recorded if n > sends)
        in_flight = pending if pending is not None and int(pending["send"]) > sends else None
        if not tail and in_flight is None:
            continue  # nothing recorded after it: unverifiable
        return {"dir": str(folder), "step": meta.get("step"), "sends": sends, "tail": len(tail),
                "pending": in_flight}
    return None


def _blobs_intact(folder: Path, meta: dict) -> bool:
    """The engine's blob-manifest rule (kernel_checkpoint.authenticate), checked before a checkpoint is offered:
    blobs.json must match the signed blob_manifest_sha256, and every pooled file it lists must exist with its
    recorded size. Read-only: an unusable checkpoint is skipped and kept for diagnosis, never deleted."""
    try:
        manifest = (folder / "blobs.json").read_bytes()
    except OSError:
        manifest = None
    if manifest is None and meta.get("blob_manifest_sha256") is None:
        return True  # a checkpoint without pooled blobs
    if manifest is None or hashlib.sha256(manifest).hexdigest() != meta.get("blob_manifest_sha256"):
        return False
    try:
        listing = json.loads(manifest)
        for relative, files in listing.items():
            store = folder.parent / "blob-pool" / str(relative).replace("/", "__")
            for name, size in files:
                if os.lstat(store / name).st_size != size:
                    return False
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return True


def archive_attempt(state: dict[str, object], engine_dir: Path, *, use_checkpoint: bool = True,
                    tape_from: Path | None = None, replay_external: bool = False) -> Path:
    """Move the stopped attempt's files to attempts/<n>/ and point the spec at its tape."""
    run: Path = state["run"]  # type: ignore[assignment]
    spec: dict = state["spec"]  # type: ignore[assignment]
    restart_live = int(state["max_turns"]) != int(spec["max_turns"])
    external = _tool_safety(run, spec, replay_external=replay_external) if restart_live else []
    number = int(state["attempt"])
    folder = run / "attempts" / str(number)
    folder.mkdir(parents=True)
    shutil.copy2(run / "spec.json", folder / "spec.json")
    for name in ATTEMPT_FILES:
        if (run / name).exists():
            os.replace(run / name, folder / name)
    # Rebuilt from the moved files, so the tape's paths name the archived bytes the child will read.
    # A fallback after a refused checkpoint restore replays the INTERRUPTED attempt (``tape_from``),
    # not the refused one, which holds only the restored prefix.
    source = tape_from or folder
    # The step limit is in request 1. A raised limit changes that request, so
    # the prior model and tool tape cannot be replayed on this fresh attempt.
    # Ordinary replay keeps tool-tape.jsonl at the run root for call-id lookup;
    # a live restart must archive it so fresh calls can reuse their original ids.
    if restart_live and (run / "tool-tape.jsonl").exists():
        os.replace(run / "tool-tape.jsonl", folder / "tool-tape.jsonl")
    own_tape = ([] if restart_live else replay_tape.build(
        source / "wire", source / "sends.jsonl", routes_required=bool(spec.get("routes"))))
    own_pending = (replay_tape.pending_request(source / "wire", routes_required=True)
                   if spec.get("routes") and not restart_live else None)
    tape, pending, prior_route_changes = carry_forward_records(
        spec, own_tape, own_pending, source / "wire",
        diverged=(source / "resume-divergence.json").exists(),
        restart_live=restart_live)
    checkpoint = select_checkpoint(run, tape, source / "wire") if use_checkpoint and not restart_live else None
    spec["max_turns"] = int(state["max_turns"])
    spec["environment"]["REACT_STEPS"] = str(spec["max_turns"])
    spec["engine_dir"] = str(engine_dir)  # same release (checked in assess), current install path
    swap = state.get("model_swap")
    if isinstance(swap, dict):
        # The engine now asks for the name the endpoint serves; replay compares with the recorded name.
        spec["model"] = swap["served"]
        spec["environment"]["DS_MODEL"] = swap["served"]
    spec["resume"] = {"schema": replay_tape.SCHEMA, "attempt": number + 1,
                      "previous_attempt": str(source),
                      "previous_status": state.get("previous_status"),
                      "resumed": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "tape": tape,
                      **({"pending": pending} if pending is not None else {}),
                      **({"route_changes": prior_route_changes} if prior_route_changes else {}),
                      **({"restart_live": True} if restart_live else {}),
                      "archived_external_operations": (external if restart_live else
                          (spec.get("resume") or {}).get("archived_external_operations", [])),
                      **({"model_swap": swap} if isinstance(swap, dict) else {})}
    if checkpoint is not None:
        spec["resume"]["checkpoint"] = checkpoint
    elif (run / "checkpoints").is_dir():
        # A replayed attempt rebuilds its own lineage: earlier checkpoints stay with the attempt that
        # made them (kept, never deleted) instead of being offered to a later resume.
        os.replace(run / "checkpoints", folder / "checkpoints")
    (run / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True))
    return folder


def fresh_workcopy(run: Path, repo: Path) -> workcopy.WorkCopy:
    """A new working copy of the start state, with the start-state hashes as its baseline."""
    parent = run / "parent"
    shutil.copytree(parent, run / "workspace", symlinks=False)
    baseline = {p.relative_to(parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(parent.rglob("*")) if p.is_file() and not p.is_symlink()}
    return workcopy.WorkCopy(original=repo.resolve(), copy=run / "workspace", baseline=baseline,
                             snapshot=parent)

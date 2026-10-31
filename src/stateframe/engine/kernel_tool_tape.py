"""Durable non-model tool calls and their replay contract."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ReplayClass = Literal["observation", "local", "external"]


class ToolTapeError(ValueError):
    """The recorded operation cannot be trusted for replay."""


class ToolReplayRefused(BaseException):
    """Fatal resume refusal, including checkpoint fallback."""


_active_tape: ToolTape | None = None
_replay_active = False


def configure_replay(run_dir: Path, *, active: bool) -> None:
    global _active_tape, _replay_active
    _active_tape = ToolTape(run_dir)
    _replay_active = active


def replay_mode_active() -> bool:
    return _replay_active


def tape_for_run() -> ToolTape:
    if _active_tape is None:
        raise ToolTapeError("tool tape is not configured for this run")
    return _active_tape


def end_replay() -> None:
    global _replay_active
    _replay_active = False


@dataclass(frozen=True)
class TapeRecord:
    call_id: str
    tool: str
    args: object
    replay_class: ReplayClass
    capability_digest: str
    definition_digest: str | None
    env_identity: str
    pre_manifest_digest: str
    exit_code: int | None = None
    observation: str | None = None
    post_manifest_digest: str | None = None
    post_env_identity: str | None = None
    completed: bool = False


@dataclass(frozen=True)
class ReplayDecision:
    serve_taped: bool
    reexecute: bool
    refuse_reason: str | None


class ToolTape:
    """Append-only JSONL. INTENT is durable before execution; COMPLETE is durable after."""

    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "tool-tape.jsonl"
        self._records: dict[str, TapeRecord] = {}
        if self.path.exists():
            for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ToolTapeError(f"tool-tape.jsonl line {number}: {error}") from error
                if not isinstance(record, dict) or record.get("kind") not in ("intent", "complete"):
                    raise ToolTapeError(f"tool-tape.jsonl line {number}: unknown record kind")
                if record["kind"] == "intent":
                    call_id = record.get("call_id")
                    if not isinstance(call_id, str) or not call_id or call_id in self._records:
                        raise ToolTapeError(f"tool-tape.jsonl line {number}: duplicate or missing call id")
                    expected = {"kind", *TapeRecord.__dataclass_fields__}
                    if (set(record) != expected or record.get("completed") is not False
                            or any(record.get(key) is not None for key in
                                   ("exit_code", "observation", "post_manifest_digest", "post_env_identity"))
                            or record.get("replay_class") not in ("observation", "local", "external")
                            or any(not isinstance(record.get(key), str) for key in
                                   ("tool", "capability_digest", "env_identity", "pre_manifest_digest"))):
                        raise ToolTapeError(f"tool-tape.jsonl line {number}: invalid intent")
                    try:
                        self._records[call_id] = TapeRecord(**{k: v for k, v in record.items() if k != "kind"})
                    except TypeError as error:
                        raise ToolTapeError(f"tool-tape.jsonl line {number}: invalid intent: {error}") from error
                else:
                    self._apply_complete(record, number)

    def _append(self, record: dict) -> None:
        encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        created = not self.path.exists()
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            with os.fdopen(fd, "ab", closefd=False) as stream:
                stream.write(encoded)
                stream.flush()
            os.fsync(fd)
        finally:
            os.close(fd)
        if created:
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    def intent(self, *, call_id, tool, args, replay_class, capability_digest,
               definition_digest, env_identity, pre_manifest_digest) -> None:
        if not isinstance(call_id, str) or not call_id or call_id in self._records:
            raise ToolTapeError("tool tape intent: duplicate or missing call id")
        if replay_class not in ("observation", "local", "external"):
            raise ToolTapeError("tool tape intent: invalid replay class")
        if any(not isinstance(value, str) or not value for value in
               (tool, capability_digest, env_identity, pre_manifest_digest)):
            raise ToolTapeError("tool tape intent: tool and identity fields must be nonempty strings")
        if definition_digest is not None and not isinstance(definition_digest, str):
            raise ToolTapeError("tool tape intent: definition digest must be a string or null")
        record = TapeRecord(call_id, tool, args, replay_class, capability_digest, definition_digest,
                            env_identity, pre_manifest_digest)
        self._append({"kind": "intent", **record.__dict__})
        self._records[call_id] = record

    def _apply_complete(self, value: dict, number: int | None = None) -> TapeRecord:
        call_id = value.get("call_id")
        if not isinstance(call_id, str):
            raise ToolTapeError(f"tool-tape.jsonl line {number}: invalid completion call id")
        previous = self._records.get(call_id)
        if previous is None or previous.completed:
            raise ToolTapeError(f"tool-tape.jsonl line {number}: completion without unique intent")
        allowed = {"kind", "call_id", "exit_code", "observation", "post_manifest_digest", "env_identity"}
        if (set(value) != allowed or (value["exit_code"] is not None and type(value["exit_code"]) is not int)
                or not isinstance(value["observation"], str)
                or not isinstance(value["post_manifest_digest"], str)
                or not isinstance(value["env_identity"], str)):
            raise ToolTapeError(f"tool-tape.jsonl line {number}: invalid completion")
        completed = TapeRecord(**{**previous.__dict__, "exit_code": value["exit_code"],
            "observation": value["observation"], "post_manifest_digest": value["post_manifest_digest"],
            "post_env_identity": value["env_identity"], "completed": True})
        if number is not None:
            self._records[call_id] = completed
        return completed

    def complete(self, *, call_id, exit_code, observation: str, post_manifest_digest, env_identity) -> None:
        value = {"kind": "complete", "call_id": call_id, "exit_code": exit_code,
                 "observation": observation, "post_manifest_digest": post_manifest_digest,
                 "env_identity": env_identity}
        if call_id not in self._records or self._records[call_id].completed:
            raise ToolTapeError("tool tape completion: missing or already complete intent")
        completed = self._apply_complete(value)
        self._append(value)
        self._records[call_id] = completed

    def lookup(self, call_id) -> TapeRecord | None:
        return self._records.get(call_id)

    def pending(self) -> tuple[TapeRecord, ...]:
        return tuple(record for record in self._records.values() if not record.completed)


def replay_decision(record: TapeRecord | None, *, replay_external_allowed: bool) -> ReplayDecision:
    if record is None:
        return ReplayDecision(False, False, "tool call is absent from the recorded tape")
    if not record.completed:
        return ReplayDecision(False, False,
            f"operation {record.call_id} may or may not have happened; reconcile and start a new run")
    if record.replay_class == "observation":
        return ReplayDecision(True, False, None)
    if record.replay_class == "local":
        return ReplayDecision(True, True, None)
    if record.replay_class == "external":
        if not replay_external_allowed:
            return ReplayDecision(False, False,
                f"external operation {record.call_id} ({record.tool}) requires --replay-external")
        return ReplayDecision(True, True, None)
    return ReplayDecision(False, False, f"invalid replay class for {record.call_id}")


def verify_intent(record: TapeRecord, *, tool, args, replay_class, capability_digest,
                  definition_digest, env_identity, pre_manifest_digest) -> str | None:
    """Check the current call before serving an observation or repeating an effect."""
    current = (tool, args, replay_class, capability_digest, definition_digest,
               env_identity, pre_manifest_digest)
    recorded = (record.tool, record.args, record.replay_class, record.capability_digest,
                record.definition_digest, record.env_identity, record.pre_manifest_digest)
    names = ("tool", "arguments", "replay class", "capability", "definition", "environment", "pre-state")
    differences = [name for name, a, b in zip(names, current, recorded) if a != b]
    return (f"tool call {record.call_id} diverged before replay: " + ", ".join(differences)) if differences else None


def verify_reexecution(record: TapeRecord, *, exit_code, post_manifest_digest,
                       env_identity) -> str | None:
    if not record.completed:
        return f"operation {record.call_id} has no completion"
    differences = []
    if exit_code != record.exit_code:
        differences.append("exit code")
    if post_manifest_digest != record.post_manifest_digest:
        differences.append("workspace manifest")
    if env_identity != record.post_env_identity:
        differences.append("environment identity")
    return (f"environment diverged at tool call {record.call_id} ({record.tool}): " +
            ", ".join(differences)) if differences else None

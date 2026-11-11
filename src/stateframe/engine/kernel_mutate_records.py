"""Durable mutation evidence and conservative tape reconciliation."""
from __future__ import annotations

import hashlib
import base64
import binascii
import fcntl
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .kernel_mutate_wire import parse_stdout, request_key, strict_loads
else:
    from kernel_mutate_wire import parse_stdout, request_key, strict_loads


class RecordError(ValueError):
    pass


class Journal:
    def __init__(self, run_dir, *, max_bytes=2 << 20):
        if type(max_bytes) is not int or not 0 < max_bytes <= 8 << 20:
            raise RecordError("invalid journal byte budget")
        self.max_bytes = max_bytes
        self.root = Path(run_dir) / "mutations"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not stat.S_ISDIR(self.root.lstat().st_mode):
            raise RecordError("invalid journal directory")
        self.path = self.root / "journal.jsonl"
        self.evidence = self.root / "evidence"
        self.evidence.mkdir(mode=0o700, exist_ok=True)
        if not stat.S_ISDIR(self.evidence.lstat().st_mode):
            raise RecordError("invalid evidence directory")
        self.torn_tail = False

    def events(self):
        if not self.path.exists() and not self.path.is_symlink():
            return []
        fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > self.max_bytes:
                raise RecordError("journal file is invalid or over budget")
            data = bytearray()
            while len(data) < info.st_size:
                block = os.read(fd, info.st_size - len(data))
                if not block:
                    raise RecordError("journal file changed during read")
                data.extend(block)
        finally:
            os.close(fd)
        if len(data) > self.max_bytes:
            raise RecordError("journal file is over budget")
        self.torn_tail = bool(data and not data.endswith(b"\n"))
        complete = data.rsplit(b"\n", 1)[0] if self.torn_tail else data
        result = []
        for index, line in enumerate(complete.splitlines(), 1):
            try:
                value = strict_loads(line)
            except (UnicodeError, ValueError) as error:
                raise RecordError(f"journal line {index} is corrupt") from error
            if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
                raise RecordError(f"journal line {index} is invalid")
            result.append(value)
        return result

    def append(self, kind, **fields):
        if not isinstance(kind, str) or not kind or "kind" in fields:
            raise RecordError("invalid journal event")
        data = (json.dumps({"kind": kind, **fields}, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                raise RecordError("journal has a torn tail; reconcile before append")
            limit = self.max_bytes - (256 << 10) if kind in ("AUDIT", "SOURCE", "DRAFT") else self.max_bytes
            if size + len(data) > limit:
                raise RecordError("journal byte budget exceeded")
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_dir(self.root)

    def put_blob(self, data: bytes) -> str:
        if not isinstance(data, bytes) or len(data) > 32 << 20:
            raise RecordError("invalid evidence blob")
        digest = hashlib.sha256(data).hexdigest()
        target = self.evidence / digest
        if target.exists():
            if self.get_blob(digest) != data:
                raise RecordError("conflicting evidence blob")
            return digest
        fd, name = tempfile.mkstemp(prefix=".blob-", dir=self.evidence)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(name, 0o400)
            try:
                os.link(name, target, follow_symlinks=False)
            except FileExistsError:
                if self.get_blob(digest) != data:
                    raise RecordError("conflicting evidence blob")
            _fsync_dir(self.evidence)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return digest

    def get_blob(self, digest: str) -> bytes:
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RecordError("invalid evidence digest")
        fd = os.open(self.evidence / digest, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 32 << 20:
                raise RecordError("invalid evidence file")
            data = b""
            while block := os.read(fd, 1 << 20):
                data += block
                if len(data) > 32 << 20:
                    raise RecordError("oversize evidence file")
        finally:
            os.close(fd)
        if hashlib.sha256(data).hexdigest() != digest:
            raise RecordError("evidence digest mismatch")
        return data


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class Action:
    kind: str
    call_id: str
    detail: str = ""
    attempt: str = "current"
    evidence: dict | None = None


@dataclass(frozen=True)
class Reconciliation:
    actions: list[Action]
    refuse: str | None = None


_METADATA_OPS = {"enter": ("episode_enter", "EPISODE_STARTED"),
                 "leave": ("episode_leave", "EPISODE_CLOSED"),
                 "watch": ("watch_start", "WATCH_REGISTERED"),
                 "unwatch": ("watch_stop", "WATCH_REMOVED"),
                 "retain": ("retention", "RETENTION")}


def _operation(intent):
    args = intent.get("args")
    if isinstance(args, dict):
        return args.get("op")
    if isinstance(args, str):
        try:
            value = json.loads(args)
        except ValueError:
            return None
        return value.get("op") if isinstance(value, dict) else None
    return None


def _effect_evidence(rows, journal):
    prepared = [row for row in rows if row["kind"] == "EFFECT_PREPARED"]
    if not prepared:
        return None
    if len(prepared) != 1:
        raise RecordError("duplicate effect preparation")
    event = prepared[0]
    files = event.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 16:
        raise RecordError("missing per-file effect evidence")
    paths = set()
    expected = {"path", "pre_sha256", "post_sha256", "pre_blob", "post_blob"}
    for item in files:
        if not isinstance(item, dict) or set(item) != expected:
            raise RecordError("invalid per-file effect evidence")
        path = item["path"]
        if (not isinstance(path, str) or path.startswith("/") or any(
                part in ("", ".", "..") for part in path.split("/")) or path in paths):
            raise RecordError("invalid effect path")
        paths.add(path)
        for side in ("pre", "post"):
            digest = item[side + "_sha256"]
            blob = item[side + "_blob"]
            if digest is None:
                if blob is not None:
                    raise RecordError("unexpected effect blob")
            elif (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                  or blob != digest):
                raise RecordError("effect digest differs from blob")
            else:
                journal.get_blob(blob)
        if item["pre_sha256"] is None and item["post_sha256"] is None:
            raise RecordError("effect has no pre or post image")
    return event


def _broker_evidence(rows, journal):
    intents = {}
    completes = set()
    for row in rows:
        kind = row["kind"]
        if kind not in ("BROKER_INTENT", "BROKER_COMPLETE"):
            continue
        key = row.get("key")
        if not isinstance(key, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", key) is None:
            raise RecordError("broker request key is missing or malformed")
        if kind == "BROKER_INTENT":
            if row.get("method", "GET") != "GET" or key in intents or key != request_key(row.get("url"), row.get("accept")):
                raise RecordError("broker intent key mismatch")
            intents[key] = row
        else:
            if key not in intents or key in completes:
                raise RecordError("broker completion has no unique intent")
            error = row.get("error")
            if error is None:
                digest = row.get("body_digest")
                if (type(row.get("status")) is not int or not 0 <= row["status"] <= 599
                        or not isinstance(row.get("headers"), dict) or not isinstance(row.get("final_url"), str)
                        or not isinstance(digest, str)):
                    raise RecordError("broker response evidence is incomplete")
                body = journal.get_blob(digest.removeprefix("sha256:"))
                if "bytes" in row and row["bytes"] != len(body):
                    raise RecordError("broker response byte count differs")
            elif not isinstance(error, str) or not error:
                raise RecordError("broker error evidence is invalid")
            completes.add(key)
    return len(intents), len(completes)


def reconcile(tape_rows: list[dict], journal: Journal, ledger_rows: list[dict], *, replay_external=False) -> Reconciliation:
    """Plan durable tape completions; callers apply actions before pending-intent checks."""
    try:
        events = journal.events()
    except (OSError, ValueError) as error:
        return Reconciliation([], f"MUTATION_EVIDENCE_TAMPERED: {error}")
    intents = {}
    completions = set()
    for row in tape_rows:
        if not isinstance(row, dict):
            return Reconciliation([], "MUTATION_TAPE_CORRUPT")
        call = row.get("call_id")
        attempt = row.get("attempt", "current")
        if not isinstance(attempt, str) or not attempt:
            return Reconciliation([], "MUTATION_TAPE_CORRUPT")
        key = (attempt, call)
        if row.get("tool") in ("mutate", "mutation_watch", "mutation_hook") and row.get("kind") == "intent":
            if not isinstance(call, str) or key in intents:
                return Reconciliation([], "MUTATION_TAPE_CORRUPT")
            intents[key] = row
        elif row.get("kind") == "complete" and key in intents:
            if key in completions:
                return Reconciliation([], "MUTATION_TAPE_CORRUPT")
            completions.add(key)
    by_call = {key: [] for key in intents}
    ledger_latest = {}
    allowed_states = {"PREPARED", "STARTED", "REGISTERED", "RELEASED", "EXITED", "REAPED", "CLEANUP_FAILED"}
    for row in ledger_rows:
        if (not isinstance(row, dict) or row.get("schema") != "stateframe-mutation-launch/1"
                or not isinstance(row.get("launch"), str) or not row["launch"]
                or row.get("state") not in allowed_states):
            return Reconciliation([], "MUTATION_LEDGER_CORRUPT")
        key = (row.get("attempt", "current"), row.get("call"))
        if key not in intents:
            return Reconciliation([], "MUTATION_LEDGER_WITHOUT_INTENT")
        ledger_latest[(*key, row["launch"])] = row["state"]
    for event in events:
        if event["kind"].startswith("BUDGET_"):
            continue
        call = event.get("call") or event.get("call_id")
        if call is None:
            continue  # definition and budget metadata have no tool call
        key = (event.get("attempt", "current"), call)
        if key not in intents:
            return Reconciliation([], "MUTATION_JOURNAL_WITHOUT_INTENT")
        by_call[key].append(event)
    actions = []
    tick_intents = {row.get("invocation_id"): row for row in events
                    if row["kind"] == "WATCH_TICK_INTENT" and isinstance(row.get("invocation_id"), str)}
    tick_done = {row.get("invocation_id") for row in events
                 if row["kind"] in ("WATCH_TICK_COMPLETE", "WATCH_TICK_INTERRUPTED")}
    for invocation_id, row in tick_intents.items():
        if invocation_id not in tick_done:
            actions.append(Action("interrupt_watch", invocation_id,
                                  detail=str(row.get("parent_turn_id", "")), evidence=row))
    hook_intents = {row.get("invocation_id"): row for row in events
                    if row["kind"] == "HOOK_TRIGGER_INTENT" and isinstance(row.get("invocation_id"), str)}
    hook_done = {row.get("invocation_id") for row in events
                 if row["kind"] in ("HOOK_TRIGGER_COMPLETE", "HOOK_TRIGGER_INTERRUPTED",
                                    "HOOK_FAILURE", "HOOK_REFUSED")}
    for invocation_id, row in hook_intents.items():
        if invocation_id not in hook_done:
            actions.append(Action("interrupt_hook", invocation_id, evidence=row))
    for (attempt, call), intent in intents.items():
        rows = by_call[(attempt, call)]
        kinds = [row["kind"] for row in rows]
        op = _operation(intent)
        launch_states = [state for (row_attempt, row_call, _), state in ledger_latest.items()
                         if (row_attempt, row_call) == (attempt, call)]
        if "CLEANUP_FAILED" in kinds or any(state not in ("PREPARED", "REAPED") for state in launch_states):
            actions.append(Action("cleanup", call, attempt=attempt))
            continue
        if intent.get("tool") in ("mutation_watch", "mutation_hook"):
            if rows:
                return Reconciliation([], "MUTATION_EVIDENCE_TAMPERED: auxiliary invocation has tool-call evidence")
            continue
        for row in rows:
            for field in ("blob", "evidence_digest", "body_digest"):
                digest = row.get(field)
                if digest is not None:
                    try:
                        if not isinstance(digest, str):
                            raise RecordError("invalid evidence digest")
                        journal.get_blob(digest.removeprefix("sha256:"))
                    except (OSError, ValueError) as error:
                        return Reconciliation([], f"MUTATION_EVIDENCE_TAMPERED: {error}")
        terminals = [row for row in rows if row["kind"] == "TERMINAL"]
        if len(terminals) > 1:
            return Reconciliation([], "MUTATION_EVIDENCE_TAMPERED: duplicate terminal")
        terminal = terminals[0] if terminals else None
        terminal_kind = None
        if terminal is not None:
            try:
                encoded = terminal.get("frame_b64")
                if not isinstance(encoded, str):
                    raise ValueError("terminal frame must be base64 text")
                if terminal.get("stdout_overflow") is not False or terminal.get("timed_out") is not False:
                    raise ValueError("terminal launch did not finish cleanly")
                raw = base64.b64decode(encoded, validate=True)
                frame = parse_stdout(raw, call=call, output_schema=terminal.get("output_schema"))
                terminal_kind = frame.kind
                exit_code = terminal.get("exit_code")
                if ((frame.kind == "result" and (type(exit_code) is not int or exit_code != 0))
                        or (frame.kind == "error" and (type(exit_code) is not int or exit_code not in (3, 4)))
                        or frame.kind not in ("result", "error")):
                    raise ValueError("terminal frame and exit code disagree")
            except (ValueError, binascii.Error) as error:
                return Reconciliation([], f"MUTATION_EVIDENCE_TAMPERED: {error}")
        try:
            broker_intents, broker_completes = _broker_evidence(rows, journal)
        except (OSError, ValueError) as error:
            return Reconciliation([], f"MUTATION_BROKER_INCONSISTENT: {error}")
        if broker_completes > broker_intents:
            return Reconciliation([], "MUTATION_BROKER_INCONSISTENT")
        if broker_intents != broker_completes:
            if any(kind.startswith("EFFECT_") for kind in kinds):
                return Reconciliation([], "MUTATION_EFFECT_INDETERMINATE: broker and local effect overlap")
            if replay_external and (attempt, call) not in completions:
                actions.append(Action("interrupt", call, "unknown broker outcome accepted", attempt))
                continue
            return Reconciliation([], "MUTATION_BROKER_UNKNOWN")
        prepared = sum(kind == "EFFECT_PREPARED" for kind in kinds)
        committed = sum(kind == "EFFECT_COMMITTED" for kind in kinds)
        if committed > prepared or prepared > 1 or committed > 1:
            return Reconciliation([], "MUTATION_EFFECT_INDETERMINATE")
        try:
            effect = _effect_evidence(rows, journal)
        except (OSError, ValueError) as error:
            return Reconciliation([], f"MUTATION_EVIDENCE_TAMPERED: {error}")
        if effect is not None and (terminal is None or "CLEANUP_VERIFIED" not in kinds):
            return Reconciliation([], "MUTATION_EVIDENCE_TAMPERED: edit lacks terminal or cleanup evidence")
        if effect is not None and terminal_kind != "result":
            return Reconciliation([], "MUTATION_EFFECT_INDETERMINATE: edit follows failed terminal")
        if "EFFECT_INDETERMINATE" in kinds:
            return Reconciliation([], "MUTATION_EFFECT_INDETERMINATE: edit rollback is unverified")
        if "EFFECT_REFUSED" in kinds:
            if committed or kinds.count("EFFECT_REFUSED") != 1:
                return Reconciliation([], "MUTATION_EFFECT_INDETERMINATE: edit outcome conflicts")
            actions.append(Action("serve" if (attempt, call) in completions else "interrupt",
                                  call, attempt=attempt))
            continue
        interrupted = [row for row in rows if row["kind"] == "INTERRUPTED"]
        if interrupted:
            if (len(interrupted) != 1 or (attempt, call) not in completions or
                    not isinstance(interrupted[0].get("observation"), str) or
                    not interrupted[0]["observation"].startswith("REFUSED MUTATION_INTERRUPTED")):
                return Reconciliation([], "MUTATION_EVIDENCE_TAMPERED: interrupted completion differs")
            actions.append(Action("serve", call, attempt=attempt))
            continue
        refusals = [row for row in rows if row["kind"] == "REFUSAL"]
        if refusals:
            if len(refusals) != 1 or (attempt, call) not in completions:
                return Reconciliation([], "MUTATION_EVIDENCE_TAMPERED: refusal completion differs")
            actions.append(Action("serve", call, attempt=attempt))
            continue
        if op in ("catalog", "inspect"):
            if rows:
                return Reconciliation([], "MUTATION_EVIDENCE_TAMPERED: unexpected catalog evidence")
            actions.append(Action("serve" if (attempt, call) in completions else "interrupt", call, attempt=attempt))
            continue
        if op in _METADATA_OPS:
            expected = _METADATA_OPS[op]
            if any(kind in kinds for kind in expected):
                actions.append(Action("serve" if (attempt, call) in completions else "complete_metadata",
                                      call, op, attempt, next(row for row in rows if row["kind"] in expected)))
            elif (attempt, call) in completions:
                return Reconciliation([], "MUTATION_EVIDENCE_TAMPERED: missing metadata event")
            else:
                actions.append(Action("interrupt", call, attempt=attempt))
            continue
        if op == "submit":
            admissions = [row for row in rows if row["kind"] in ("ADMISSION", "REVISION_FAILED")]
            if len(admissions) > 1:
                return Reconciliation([], "MUTATION_EVIDENCE_TAMPERED: duplicate admission outcome")
            if admissions:
                evidence = admissions[0]
                digest = evidence.get("admission_digest")
                if evidence["kind"] == "ADMISSION" and (not isinstance(digest, str)
                    or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None):
                    return Reconciliation([], "MUTATION_EVIDENCE_TAMPERED: invalid admission digest")
                actions.append(Action("serve" if (attempt, call) in completions else "complete_metadata",
                                      call, op, attempt, evidence))
            elif (attempt, call) in completions:
                return Reconciliation([], "MUTATION_EVIDENCE_TAMPERED: missing admission outcome")
            else:
                actions.append(Action("interrupt", call, attempt=attempt))
            continue
        if (attempt, call) in completions:
            if "CLEANUP_VERIFIED" not in kinds or not ({"TERMINAL", "EFFECT_COMMITTED"} & set(kinds)):
                return Reconciliation([], "MUTATION_EVIDENCE_TAMPERED: missing terminal or cleanup evidence")
            if prepared and not committed:
                return Reconciliation([], "MUTATION_EFFECT_INDETERMINATE")
            actions.append(Action("serve", call, attempt=attempt,
                                  evidence={"effect": effect, "terminal": terminal} if effect is not None else terminal))
            continue
        if prepared and not committed:
            actions.append(Action("recover_effect", call, attempt=attempt,
                                  evidence={"effect": effect, "terminal": terminal}))
            continue
        if committed:
            if "CLEANUP_VERIFIED" not in kinds:
                return Reconciliation([], "MUTATION_CLEANUP_UNKNOWN")
            actions.append(Action("verify_effect", call, "verify committed effect identities", attempt,
                                  {"effect": effect, "terminal": terminal}))
            continue
        if "TERMINAL" in kinds and "CLEANUP_VERIFIED" in kinds:
            if terminal is None:
                return Reconciliation([], "MUTATION_EVIDENCE_TAMPERED: missing terminal frame")
            actions.append(Action("complete", call, "terminal evidence", attempt, terminal))
        else:
            actions.append(Action("interrupt", call, attempt=attempt))
    return Reconciliation(actions)

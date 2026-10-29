"""``stateframe inspect``: read a finished run directory.

Everything here comes from the exact bytes the engine sent (``wire/NNNN.request.json``),
so the frame you see is the frame the model saw, not a re-render.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FRAME_MARKER = "[StateFrame]"


class InspectError(ValueError):
    pass


def _wire_files(run: Path) -> list[Path]:
    wire = run / "wire"
    if not wire.is_dir():
        raise InspectError(f"{run} has no wire/ directory; is it a stateframe run directory?")
    return sorted(wire.glob("[0-9][0-9][0-9][0-9].request.json"))


def load_send(run: Path, number: int) -> tuple[bytes, dict]:
    path = run / "wire" / f"{number:04d}.request.json"
    if not path.is_file():
        available = len(_wire_files(run))
        raise InspectError(f"no send {number}; this run has sends 1..{available}")
    data = path.read_bytes()
    return data, json.loads(data)


def frame_of(body: dict) -> dict | None:
    """The typed StateFrame payload carried in a request body, parsed from its JSON."""
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        # The engine writes "[StateFrame]" then a newline (or a space) then the JSON payload.
        if isinstance(content, str) and content.startswith(FRAME_MARKER):
            payload = content[len(FRAME_MARKER):].lstrip()
            try:
                return json.loads(payload)
            except ValueError:
                return {"_unparsed": payload}
    return None


def _sends(run: Path) -> dict[int, dict]:
    path = run / "sends.jsonl"
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {int(r["send"]): r for r in records}


def overview(run: Path) -> str:
    files = _wire_files(run)
    sends = _sends(run)
    lines = [f"run: {run}"]
    summary_path = run / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        lines.append(f"result: {summary.get('engine_result')}  tests after: "
                     f"{'PASS' if (summary.get('tests_after') or {}).get('passed') else 'FAIL'}  "
                     f"files changed: {', '.join(summary.get('files_changed') or []) or 'none'}")
    lines.append(f"{'send':>4}  {'bytes':>7}  {'frame B':>7}  {'msgs':>4}  {'roles':<10} "
                 f"{'prompt':>7}  {'cached':>7}  action")
    for path in files:
        number = int(path.name[:4])
        body = json.loads(path.read_bytes())
        frame = frame_of(body)
        frame_bytes = len(json.dumps(frame, separators=(",", ":"))) if frame else 0
        roles = "".join((m.get("role") or "?")[0] for m in body.get("messages") or [])
        record = sends.get(number, {})
        lines.append(f"{number:>4}  {path.stat().st_size:>7}  {frame_bytes:>7}  "
                     f"{len(body.get('messages') or []):>4}  {roles:<10} "
                     f"{str(record.get('prompt_tokens', '-')):>7}  "
                     f"{str(record.get('cached_tokens', '-')):>7}  "
                     f"{', '.join(record.get('actions') or []) or '-'}")
    lines.append("inspect one send: stateframe inspect <run> --send N [--frame | --bytes | --messages]")
    return "\n".join(lines)


def show_send(run: Path, number: int, *, what: str = "summary") -> str | bytes:
    data, body = load_send(run, number)
    if what == "bytes":
        return data
    if what == "frame":
        frame = frame_of(body)
        if frame is None:
            raise InspectError(f"send {number} carries no [StateFrame] message")
        return json.dumps(frame, indent=2, ensure_ascii=False, sort_keys=True)
    if what == "messages":
        out = []
        for index, message in enumerate(body.get("messages") or []):
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            calls = message.get("tool_calls")
            out.append(f"--- [{index}] {message.get('role')} ({len(text or '')} chars)"
                       + (f" tool_calls={json.dumps(calls)}" if calls else ""))
            out.append(text or "")
        return "\n".join(out)
    frame = frame_of(body)
    record = _sends(run).get(number, {})
    tools = [t.get("function", {}).get("name") for t in body.get("tools") or []]
    lines = [
        f"send {number}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}",
        f"file: {run / 'wire' / f'{number:04d}.request.json'}",
        f"model: {body.get('model')}  max_tokens: {body.get('max_tokens')}  "
        f"temperature: {body.get('temperature')}",
        f"messages: {' '.join((m.get('role') or '?') for m in body.get('messages') or [])}",
        f"tools offered ({len(tools)}): {', '.join(str(t) for t in tools)}",
        f"frame: {len(json.dumps(frame, separators=(',', ':'))) if frame else 0} bytes, "
        f"top-level fields: {', '.join(sorted(frame)) if isinstance(frame, dict) else '-'}",
        f"usage: prompt {record.get('prompt_tokens')} cached {record.get('cached_tokens')} "
        f"completion {record.get('completion_tokens')}  -> {', '.join(record.get('actions') or []) or '-'}",
    ]
    return "\n".join(lines)

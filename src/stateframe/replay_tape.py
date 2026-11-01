"""The recorded model responses of an interrupted run, replayed to resume it (stdlib only).

``stateframe resume`` restarts the same engine with the same settings on the same run directory. For
the requests the interrupted attempt already completed, the engine's wire send is answered with the
response recorded in ``wire/NNNN.response.json`` instead of asking the model again. The engine therefore
re-executes exactly the actions it took before, on a fresh copy of the start state, and every part of
its state (the typed StateFrame, receipts, journal) is rebuilt by the engine itself. After the last
recorded response the run continues live.

A recorded response is served only while the engine's new request matches the recorded request byte
for byte, apart from run-scoped identifiers (fresh 32-hex run ids and handles) and test durations.
A resume that continues under another model NAME (``stateframe resume --model``: the same weights served
under a different name) also compares with the request's leading top-level ``model`` value set back to
the recorded one; that field and nothing else.
On the first mismatch replay stops and the run continues live from that point: frame-only means the
current frame is the whole state, so a faithful prefix is always a valid place to continue from.

Imported by the runner as ``stateframe.replay_tape`` and by the engine child by file path.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SCHEMA = "stateframe-resume/1"
# A uuid4 hex run id or handle, the only per-process difference between two identical runs.
_RUN_ID = re.compile(rb"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")
# pytest's wall-clock duration in a test summary ("1 failed in 0.05s").
_DURATION = re.compile(rb" in [0-9]+\.[0-9]+s\b")
_ADDRESS = re.compile(rb"\bat 0x[0-9a-fA-F]+")


_MODEL_FIELD = b'{"model": '


class TapeError(ValueError):
    """The recorded sends of a run cannot be trusted as a replay tape."""


REFUSAL_FILE = "checkpoint-refused.json"


class CheckpointUnfaithful(BaseException):
    """A checkpoint-restored engine rebuilt a request that differs from the recorded one.

    Raised before that request is sent, so nothing live happens on a wrong state; the runner then
    falls back to replay resume from the start of the run. A BaseException, like KeyboardInterrupt,
    so no engine transport handler can mistake it for a retryable send failure.
    """


_TEXT_RUN_ID = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")
_TEXT_DURATION = re.compile(r" in [0-9]+(?:\.[0-9]+)?s(?: \(\d+:\d\d:\d\d\))?")
# A Python object's memory address in a repr ("<MonkeyPatch object at 0x106cd7360>"), printed in tracebacks.
_TEXT_ADDRESS = re.compile(r"\bat 0x[0-9a-fA-F]+")


def mask_volatile(text: str) -> str:
    """A check's output without what legitimately differs between two executions: durations, run ids and
    object addresses."""
    return _TEXT_ADDRESS.sub("at 0x<addr>", _TEXT_DURATION.sub(" in <t>s", _TEXT_RUN_ID.sub("<id>", text)))


def _normalized(request: bytes) -> bytes:
    return _ADDRESS.sub(b"at 0x<addr>", _DURATION.sub(b" in <t>s", _RUN_ID.sub(b"<id>", request)))


def normalized_digest(request: bytes) -> str:
    """Request identity for replay matching: exact bytes except run ids and test durations."""
    return "sha256:" + hashlib.sha256(_normalized(request)).hexdigest()


def first_difference(recorded: bytes, replayed: bytes, context: int = 160) -> dict[str, object]:
    """Where two requests first differ after normalization, with a short window of each (for the run record)."""
    a, b = _normalized(recorded), _normalized(replayed)
    offset = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    start = max(0, offset - context // 2)
    return {"byte_offset": offset, "recorded_bytes": len(recorded), "replayed_bytes": len(replayed),
            "recorded_window": a[start:offset + context].decode("utf-8", "replace"),
            "replayed_window": b[start:offset + context].decode("utf-8", "replace")}


def _leading_model(request: bytes) -> tuple[str, int] | None:
    """The leading top-level ``"model"`` value of a chat request and the byte offset just past it, else None."""
    if not request.startswith(_MODEL_FIELD):
        return None
    try:
        text = request.decode("utf-8")
        value, end = json.JSONDecoder().raw_decode(text, len(_MODEL_FIELD))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(value, str):
        return None
    return value, len(text[:end].encode("utf-8"))


def with_recorded_model(request: bytes, recorded: bytes, model_swap: dict | None) -> bytes | None:
    """``request`` with its model value replaced by the recorded request's exact bytes, when ``model_swap``
    allows exactly that swap (``served`` -> one of ``recorded``); None when it does not apply."""
    if not isinstance(model_swap, dict):
        return None
    ours, theirs = _leading_model(request), _leading_model(recorded)
    allowed = model_swap.get("recorded")
    if (ours is None or theirs is None or ours[0] != model_swap.get("served")
            or not isinstance(allowed, list) or theirs[0] not in allowed):
        return None
    return recorded[:theirs[1]] + request[ours[1]:]


def _send_records(sends_path: Path) -> dict[int, dict]:
    """sends.jsonl by send index; a torn final line (process killed mid-append) is ignored."""
    if not sends_path.is_file():
        return {}
    lines = sends_path.read_bytes().split(b"\n")
    records: dict[int, dict] = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            if number == len(lines):  # unterminated last line
                continue
            raise TapeError(f"sends.jsonl line {number} is not valid JSON") from None
        if not isinstance(record, dict) or type(record.get("send")) is not int:
            raise TapeError(f"sends.jsonl line {number} is not a send record")
        records[record["send"]] = record
    return records


def _response_ok(data: bytes) -> bool:
    try:
        parsed = json.loads(data)
    except (ValueError, RecursionError):
        return False
    return isinstance(parsed, dict) and isinstance(parsed.get("choices"), list) and bool(parsed["choices"])


def _route_record(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise TapeError(f"wire/{path.name} is not a valid route record") from error
    if (not isinstance(value, dict) or not isinstance(value.get("role"), str)
            or not isinstance(value.get("endpoint"), str)
            or not isinstance(value.get("model"), str)
            or not {"role", "scope", "unit_class", "endpoint", "model", "escalated_from"}.issubset(value)
            or set(value) - {"role", "scope", "unit_class", "endpoint", "endpoint_host", "model",
                             "escalated_from"}):
        raise TapeError(f"wire/{path.name} is not a complete route record")
    return value


def _route_path(wire_dir: Path, index: int) -> Path:
    return wire_dir / f"{index:04d}.route.json"


def build(wire_dir: Path, sends_path: Path, *, routes_required: bool = False) -> list[dict[str, object]]:
    """Every completed model exchange of an attempt, in send order.

    A send that failed (``NNNN.error.txt``) is skipped: the engine retried it, and the retry is the
    exchange that counted. A torn response written while the process was killed ends the tape. Any
    other damage (a broken response followed by later sends, a request whose bytes differ from its
    record) is corruption and refuses the tape.
    """
    if not wire_dir.is_dir():
        return []
    records = _send_records(sends_path)
    indexes = sorted(int(p.name[:4]) for p in wire_dir.glob("[0-9][0-9][0-9][0-9].request.json"))
    route_indexes = sorted(int(p.name[:4]) for p in wire_dir.glob("[0-9][0-9][0-9][0-9].route.json"))
    if routes_required and indexes and not route_indexes:
        raise TapeError("wire/ route records are required for this run")
    if route_indexes and route_indexes != list(range(1, route_indexes[-1] + 1)):
        raise TapeError("wire/ route files are not a contiguous sequence")
    if route_indexes and (route_indexes[:-1] != indexes[:len(route_indexes) - 1]
                          or route_indexes[-1] not in indexes + [len(indexes) + 1]):
        raise TapeError("wire/ route record is missing or has no matching send index")
    if indexes and indexes != list(range(1, indexes[-1] + 1)):
        raise TapeError("wire/ request files are not a contiguous sequence")
    tape: list[dict[str, object]] = []
    for index in indexes:
        stem = wire_dir / f"{index:04d}"
        request_path, response_path = stem.with_suffix(".request.json"), stem.with_suffix(".response.json")
        request = request_path.read_bytes()
        route_path = _route_path(wire_dir, index)
        route = _route_record(route_path) if route_path.is_file() else None
        if route_indexes and route is None:
            raise TapeError(f"wire/{route_path.name} missing for routed request")
        record = records.get(index)
        if record is not None and record.get("request_sha256") != hashlib.sha256(request).hexdigest():
            raise TapeError(f"wire/{request_path.name} does not match its sends.jsonl record")
        if (wire_dir / f"{index:04d}.error.txt").is_file():
            continue
        last = index == indexes[-1]
        if not response_path.is_file():
            if last:
                break  # the request in flight when the attempt stopped
            raise TapeError(f"wire/{response_path.name} missing although later sends exist")
        response = response_path.read_bytes()
        if not _response_ok(response):
            if last:
                break  # torn write at the kill
            raise TapeError(f"wire/{response_path.name} is not a valid chat response")
        tape.append({"send": index, "request": str(request_path), "response": str(response_path),
                      "request_digest": normalized_digest(request),
                      "response_sha256": hashlib.sha256(response).hexdigest(),
                      **({"route": route} if route is not None else {})})
    return tape


def pending_request(wire_dir: Path, *, routes_required: bool = False) -> dict[str, object] | None:
    """The last request of an attempt that got neither a response nor an error (in flight at the stop)."""
    if not wire_dir.is_dir():
        return None
    indexes = sorted(int(p.name[:4]) for p in wire_dir.glob("[0-9][0-9][0-9][0-9].request.json"))
    routes = sorted(int(p.name[:4]) for p in wire_dir.glob("[0-9][0-9][0-9][0-9].route.json"))
    if routes_required and indexes and not routes:
        raise TapeError("wire/ route records are required for pending send")
    if routes and (not indexes or routes[-1] == indexes[-1] + 1):
        route = _route_record(_route_path(wire_dir, routes[-1]))
        return {"send": routes[-1], "route": route, "orphan_route": True}
    if not indexes:
        return None
    stem = wire_dir / f"{indexes[-1]:04d}"
    response = stem.with_suffix(".response.json")
    if (wire_dir / f"{indexes[-1]:04d}.error.txt").is_file():
        return None
    if response.is_file() and _response_ok(response.read_bytes()):
        return None
    request = stem.with_suffix(".request.json")
    return {"send": indexes[-1], "request": str(request),
            "request_digest": normalized_digest(request.read_bytes()),
            **({"route": _route_record(_route_path(wire_dir, indexes[-1]))}
               if _route_path(wire_dir, indexes[-1]).is_file() else {})}


class Replayer:
    """Engine-child side: answer matching sends from the tape until it ends or diverges."""

    def __init__(self, tape: list[dict[str, object]], model_swap: dict | None = None, *,
                 strict: bool = False, pending: dict[str, object] | None = None,
                 route_changes: list[dict[str, object]] | None = None) -> None:
        self.tape = list(tape)
        self.cursor = 0
        self.active = bool(self.tape)
        self.diverged: dict[str, object] | None = None
        self.route_mismatch = False
        self.model_swap = model_swap if isinstance(model_swap, dict) else None
        self.model_swapped = 0  # replayed requests that matched only with the model name set back
        # strict (checkpoint resume): a divergence is a refusal, never a switch to live.
        self.strict = strict
        # The request that was in flight (sent, no response) when the attempt stopped: the first live
        # request after the tape must rebuild it byte for byte (checked once).
        self.pending = dict(pending) if pending else None
        self.route_changes = list(route_changes or [])

    def _route_matches(self, expected: dict[str, object], actual: dict[str, object] | None) -> bool:
        if actual is None:
            return False
        allowed = dict(expected)
        for change in self.route_changes:
            if change.get("role") == expected.get("role") and change.get("field") in {"endpoint", "model"}:
                field = str(change["field"])
                before = str(change.get("from"))
                if field == "endpoint":
                    before = before.rstrip("/") + "/chat/completions"
                if allowed.get(field) != before:
                    continue
                value = str(change.get("to"))
                allowed[field] = value.rstrip("/") + "/chat/completions" if field == "endpoint" else value
                if field == "endpoint" and "endpoint_host" in allowed:
                    from urllib.parse import urlsplit
                    allowed["endpoint_host"] = urlsplit(value).netloc
        return allowed == actual

    def answer(self, request: bytes, *, route: dict[str, object] | None = None) -> bytes | None:
        """The recorded response for this request, or None to send it live."""
        if not self.active:
            if self.pending is not None and self.diverged is None:
                expected, self.pending = self.pending, None
                if "route" in expected and not self._route_matches(expected["route"], route):
                    self.route_mismatch = True
                    self.diverged = {"reason": "pending routed send destination or model differs",
                                     "recorded_send": expected["send"]}
                    return None
                if expected.get("orphan_route"):
                    return None
                digest = normalized_digest(request)
                if digest != expected["request_digest"]:
                    swap = self._route_model_swap(expected.get("route"))
                    recorded = Path(str(expected["request"])).read_bytes()
                    alternate = with_recorded_model(request, recorded, swap)
                    if alternate is not None and normalized_digest(alternate) == expected["request_digest"]:
                        return None
                    self.diverged = {
                        "tape_position": self.cursor + 1, "tape_total": len(self.tape),
                        "recorded_send": expected["send"], "recorded_digest": expected["request_digest"],
                        "replayed_digest": digest,
                        "reason": "the engine's rebuilt request differs from the request in flight when the "
                                  "attempt stopped",
                        "first_difference": first_difference(Path(str(expected["request"])).read_bytes(),
                                                             request)}
            return None
        item = self.tape[self.cursor]
        if "route" in item and not self._route_matches(item["route"], route):
            self.route_mismatch = True
            self.active = False
            self.diverged = {"reason": "routed send destination or model differs",
                             "recorded_send": item["send"]}
            return None
        digest = normalized_digest(request)
        compared = request
        if digest != item["request_digest"]:
            swapped = with_recorded_model(request, Path(str(item["request"])).read_bytes(),
                                          self._route_model_swap(item.get("route")))
            if swapped is not None:
                compared, digest = swapped, normalized_digest(swapped)
                if digest == item["request_digest"]:
                    self.model_swapped += 1
        if digest != item["request_digest"]:
            self.active = False
            self.diverged = {"tape_position": self.cursor + 1, "tape_total": len(self.tape),
                             "recorded_send": item["send"],
                             "recorded_digest": item["request_digest"], "replayed_digest": digest,
                             "reason": "the engine's rebuilt request differs from the recorded request",
                             # With a model swap, the window shows the difference beyond the model name.
                             "first_difference": first_difference(Path(str(item["request"])).read_bytes(),
                                                                  compared)}
            if self.model_swap is not None:
                self.diverged["model_swap"] = dict(self.model_swap, swapped_matches=self.model_swapped)
            return None
        response = Path(str(item["response"])).read_bytes()
        if hashlib.sha256(response).hexdigest() != item["response_sha256"]:
            raise TapeError(f"recorded response for send {item['send']} changed after resume started")
        self.cursor += 1
        if self.cursor == len(self.tape):
            self.active = False
        return response

    def _route_model_swap(self, route: object) -> dict | None:
        if not isinstance(route, dict):
            return self.model_swap
        for change in self.route_changes:
            if change.get("role") == route.get("role") and change.get("field") == "model":
                return {"served": change.get("to"), "recorded": [change.get("from")]}
        return self.model_swap

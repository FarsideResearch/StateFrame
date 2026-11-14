"""Default-off read capture history and dispatch audit, never prompt authority.

Raw coordinates come only from the source page's top-level byte fields. Lossy
pages retain decoded columns separately; effective_range byte offsets are not
raw-file coordinates. Capture history is append-only, including repeated reads.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Literal


def enabled() -> bool:
    return os.environ.get("STATEFRAME_WORKING_SET_ALLOCATOR", "off") == "on"


READ_STATUSES = frozenset({
    "PAGE", "OK", "DIRECTORY", "BINARY_FILE", "DENIED", "REFUSED", "INVALID_ARGUMENT",
    "STALE_CURSOR", "STALE_REVISION", "OUT_OF_RANGE", "SYMBOL_NOT_FOUND",
    "MISSING", "PERMISSION", "DEADLINE", "NOT_DISPATCHED", "INTERRUPTED", "ERROR",
    "OK_UNCAPTURED", "DEDUP_STUB", "SHELL_READ", "GREP", "CHECK_OUTPUT", "REFUSED_ADMISSION",
})


def typed_status(status: str) -> str:
    aliases = {"READ_CURSOR_INVALID": "INVALID_ARGUMENT", "READ_CURSOR_STALE": "STALE_CURSOR",
               "NOT_FOUND": "MISSING", "INVALID_STREAM": "INVALID_ARGUMENT",
               "INVALID_OFFSET": "INVALID_ARGUMENT", "INVALID_LIMIT": "INVALID_ARGUMENT"}
    status = aliases.get(status, status)
    return status if status in READ_STATUSES else "ERROR"


def uncaptured_path_reason(root: object, path: object) -> str:
    """Distinguish a lexical external path from a workspace symlink escape."""
    candidate = Path(os.path.abspath(str(path)))
    return "symlink_outside" if candidate.is_relative_to(Path(str(root)).resolve()) else "external_path"


def capture_skipped(runtime: object, *, reason: str, status: str = "OK_UNCAPTURED") -> None:
    if not enabled():
        return
    scope = getattr(runtime, "_read_attempt_scope", None)
    if scope is not None:
        scope.skip_reason = reason
    runtime.journal.emit("read_capture_skipped", attempt_id=scope.attempt_id if scope else None,
                         status=typed_status(status), reason=reason)


def capture_failed(runtime: object, error: Exception) -> None:
    if not enabled():
        return
    scope = getattr(runtime, "_read_attempt_scope", None)
    if scope is not None:
        scope.skip_reason = "capture_failed"
    runtime.journal.emit("read_capture_failed", attempt_id=scope.attempt_id if scope else None,
                         error_type=type(error).__name__, message=str(error).split("\n", 1)[0][:160])


def reduce_read_results(events: Iterable[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    """Audit view of a journal prefix; an unpaired attempt was interrupted.

    This reduction neither appends synthetic events nor restores capture authority.
    """
    if not enabled():
        return ()
    attempts: dict[object, Mapping[str, object]] = {}
    results: dict[object, dict[str, object]] = {}
    for event in events:
        attempt_id = event.get("attempt_id")
        if event.get("kind") == "read_attempt":
            attempts[attempt_id] = event
        elif event.get("kind") == "read_result":
            results[attempt_id] = dict(event)
    return tuple(results.get(key, {"kind": "read_result", "attempt_id": key,
                                  "request_index": event.get("request_index"),
                                  "status": "INTERRUPTED"})
                 for key, event in attempts.items())


def line_digest(line: str) -> str:
    """Digest one source line, excluding its line terminator."""
    return hashlib.blake2b(line.rstrip().encode("utf-8"), digest_size=8).hexdigest()


def line_digests_for(payload: str) -> tuple[str, ...]:
    return tuple(line_digest(line) for line in payload.splitlines() if line.strip())


def line_digests_with_numbers(payload: str, first_line: int) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Digests of the non-blank payload lines, with each one's source line number."""
    pairs = [(line_digest(line), first_line + offset)
             for offset, line in enumerate(payload.splitlines()) if line.strip()]
    return tuple(d for d, _ in pairs), tuple(n for _, n in pairs)


def relative_path(root: object, path: object) -> str | None:
    """Workspace-relative POSIX path, or None when the resolved path leaves the root."""
    try:
        return Path(str(path)).resolve().relative_to(Path(str(root)).resolve()).as_posix()
    except (ValueError, OSError):
        return None


@dataclass(frozen=True)
class DecodedSpan:
    first_column: int
    last_column: int
    presentation_version: str

    def __post_init__(self) -> None:
        if (type(self.first_column) is not int or self.first_column < 0
                or type(self.last_column) is not int or self.last_column < 0
                or not self.presentation_version):
            raise ValueError("invalid decoded span")


@dataclass(frozen=True)
class ReceiptIdentity:
    path: str
    revision: str
    first_line: int
    last_line: int
    byte_start: int | None
    byte_end: int | None
    payload_sha256: str
    lossy_span: DecodedSpan | None

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if (not self.path or path.is_absolute() or ".." in path.parts
                or path.as_posix() != self.path or self.path == "."):
            raise ValueError("capture path must be canonical workspace-relative POSIX")
        for digest in (self.revision.removeprefix("sha256:"), self.payload_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid capture digest")
        if not self.revision.startswith("sha256:"):
            raise ValueError("capture revision must name raw SHA-256")
        if (type(self.first_line) is not int or type(self.last_line) is not int
                or self.first_line < 1 or self.last_line < self.first_line):
            raise ValueError("invalid capture line range")
        if self.lossy_span is None:
            if (type(self.byte_start) is not int or type(self.byte_end) is not int
                    or self.byte_start < 0 or self.byte_end < self.byte_start):
                raise ValueError("capture requires a raw byte interval")
        elif self.byte_start is not None or self.byte_end is not None:
            raise ValueError("lossy decoded spans cannot claim raw byte offsets")
        elif not isinstance(self.lossy_span, DecodedSpan):
            raise ValueError("lossy coordinates require a decoded span")
        if (self.lossy_span is not None and self.first_line == self.last_line
                and self.lossy_span.last_column < self.lossy_span.first_column):
            raise ValueError("invalid single-line decoded interval")


@dataclass(frozen=True)
class ReadReceipt:
    """Agreed consumer view; delivery is deliberately absent."""
    path: str
    revision: str
    first_line: int
    last_line: int
    payload_sha256: str
    capture_index: int
    due_request: int
    source: Literal["read", "dedup_stub"]
    line_digests: tuple[str, ...] = field(compare=False, hash=False)
    line_numbers: tuple[int, ...] = field(compare=False, hash=False)
    byte_start: int | None
    byte_end: int | None
    lossy_span: DecodedSpan | None = None


@dataclass(frozen=True)
class ReadCapture:
    capture_id: str
    attempt_id: str
    identity: ReceiptIdentity
    payload_blob_ref: str
    original_page_blob_ref: str
    source: Literal["disk", "file_cache", "page_cache"]
    origin_response_id: str
    capture_ordinal: int
    recovery_arguments: bytes
    capture_index: int
    due_request: int
    line_digests: tuple[str, ...] = field(compare=False, hash=False)
    line_numbers: tuple[int, ...] = field(compare=False, hash=False)

    def receipt(self, *, dedup: bool = False) -> ReadReceipt:
        return ReadReceipt(**vars(self.identity), capture_index=self.capture_index,
                           due_request=self.due_request,
                           source="dedup_stub" if dedup else "read",
                           line_digests=self.line_digests,
                           line_numbers=self.line_numbers)


class CaptureLedger:
    """Creation-ordered immutable captures; identities can have many attempts."""
    def __init__(self) -> None:
        self._captures: list[ReadCapture] = []
        self._identities: dict[ReceiptIdentity, list[ReadCapture]] = {}

    def __len__(self) -> int:
        return len(self._captures)

    def add(self, capture: ReadCapture) -> None:
        self._captures.append(capture)
        self._identities.setdefault(capture.identity, []).append(capture)

    def all(self) -> tuple[ReadCapture, ...]:
        return tuple(self._captures)

    def live(self) -> tuple[ReadCapture, ...]:
        return self.all()

    def by_identity(self, identity: ReceiptIdentity) -> tuple[ReadCapture, ...]:
        return tuple(self._identities.get(identity, ()))


ReceiptLedger = CaptureLedger


def _request_index(runtime: object) -> int:
    return int(getattr(runtime, "_frame_budget_request_count", 0))


def _blob(runtime: object, text: str) -> str:
    if __package__:
        from .kernel_measurement import externalize_journal_fields
    else:
        from kernel_measurement import externalize_journal_fields
    # Force even tiny pages through the established JSON-value blob store.
    ref = externalize_journal_fields({"payload": text}, runtime.journal.path,
                                    threshold_bytes=0)["payload"]
    return ref["$blob"]


def capture_read(runtime: object, *, path: str | None, metadata: Mapping[str, object],
                 payload: str, original_page: str, source: str,
                 arguments: Mapping[str, object], skip_reason: str = "external_path") -> ReadCapture | None:
    """The sole capture producer, called before either display/dedup branch."""
    if not enabled():
        return None
    if metadata.get("result_class") != "OK":
        capture_skipped(runtime, reason="stale" if metadata.get("result_class") == "STALE_REVISION"
                        else "non_page", status=str(metadata.get("result_class") or "ERROR"))
        return None
    if path is None:
        capture_skipped(runtime, reason=skip_reason)
        return None
    effective = metadata["effective_range"]
    lossy = bool(metadata.get("decoded_with_replacement"))
    span = (DecodedSpan(effective["start_column"], effective["end_column"],
                        str(metadata["presentation_version"])) if lossy else None)
    identity = ReceiptIdentity(
        path, str(metadata["revision"]), effective["start_line"], effective["end_line"],
        None if lossy else metadata.get("byte_start"),
        None if lossy else metadata.get("byte_end"),
        hashlib.sha256(payload.encode("utf-8")).hexdigest(), span)
    if not lossy and identity.byte_end - identity.byte_start != len(payload.encode("utf-8")):
        raise ValueError("raw capture interval does not match payload bytes")
    if source not in {"disk", "file_cache", "page_cache"}:
        raise ValueError("unknown capture source")
    ledger = getattr(runtime, "read_capture_ledger", None)
    if ledger is None:
        ledger = runtime.read_capture_ledger = CaptureLedger()
    attempt = getattr(runtime, "_read_attempt_scope", None)
    ordinal = len(ledger) + 1
    index = _request_index(runtime)
    capture = ReadCapture(
        f"{attempt.attempt_id}:capture" if attempt else f"read-capture-{ordinal}",
        attempt.attempt_id if attempt else "",
        identity, _blob(runtime, payload), _blob(runtime, original_page), source,
        str(getattr(getattr(runtime, "_active_capture", None),
                    "selected_response_event_id", "") or ""),
        ordinal, json.dumps(dict(arguments), sort_keys=True, separators=(",", ":")).encode(),
        index, index + 1, *line_digests_with_numbers(payload, identity.first_line))
    ledger.add(capture)
    if __package__:
        from .kernel_working_set_state import record_capture
    else:
        from kernel_working_set_state import record_capture
    record_capture(runtime, capture)
    if attempt is not None:
        attempt.capture = capture
    return capture


class ReadAttemptScope:
    """One selected read, including predispatch refusal and interruption.

    finish is idempotent because native batch cancellation and worker cleanup
    can both observe the same selected action. Failure never advertises a capture.
    """
    def __init__(self, runtime: object, arguments: object, *, call_id: object = None,
                 batch_index: int | None = None) -> None:
        self.runtime = runtime
        self.call_id = call_id
        self.capture: ReadCapture | None = None
        self.skip_reason: str | None = None
        self.finished = False
        ordinal = getattr(runtime, "_read_attempt_ordinal", 0) + 1
        runtime._read_attempt_ordinal = ordinal
        # Audit IDs may use the journal position: unlike wire identities they
        # must remain unique when a continuation reconstructs a fresh runtime.
        position = getattr(runtime.journal, "_sequence", ordinal - 1) + 1
        self.attempt_id = f"read-attempt-{position}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                arguments = {"raw_arguments": arguments}
        args = dict(arguments) if isinstance(arguments, Mapping) else {}
        runtime.journal.emit(
            "read_attempt", attempt_id=self.attempt_id, path=args.get("path"),
            requested_range={k: v for k, v in args.items() if k != "path"},
            request_index=_request_index(runtime), provider_call_id=call_id,
            batch_index=batch_index)

    def finish(self, status: str = "REFUSED", *, error: BaseException | None = None,
               dedup: bool = False) -> None:
        if self.finished:
            return
        if error is not None:
            status = ("INTERRUPTED" if isinstance(error, (KeyboardInterrupt, SystemExit)) else
                      "DEADLINE" if isinstance(error, TimeoutError) else
                      "PERMISSION" if isinstance(error, PermissionError) else
                      "MISSING" if isinstance(error, FileNotFoundError) else "ERROR")
        status = typed_status(status)
        fields: dict[str, object] = {}
        if self.capture is not None and status in {"OK", "PAGE"}:
            status = "DEDUP_STUB" if dedup else status
            fields = {**asdict(self.capture.receipt(dedup=dedup)),
                      "capture_id": self.capture.capture_id,
                      "identity": asdict(self.capture.identity),
                      "payload_blob_ref": self.capture.payload_blob_ref,
                      "original_page_blob_ref": self.capture.original_page_blob_ref,
                      "capture_source": self.capture.source,
                      "origin_response_id": self.capture.origin_response_id,
                      "capture_ordinal": self.capture.capture_ordinal,
                      "recovery_arguments": self.capture.recovery_arguments.decode("utf-8")}
        elif status in {"OK", "PAGE", "OK_UNCAPTURED"}:
            status = "OK_UNCAPTURED"
            fields["reason"] = self.skip_reason or "unavailable"
        elif self.skip_reason is not None:
            fields["reason"] = self.skip_reason
        self.runtime.journal.emit(
            "read_result", attempt_id=self.attempt_id, status=status,
            request_index=_request_index(self.runtime), **fields)
        self.finished = True

    def __enter__(self) -> ReadAttemptScope:
        return self

    def __exit__(self, _type: object, error: BaseException | None, _tb: object) -> None:
        self.finish(error=error)


def finish_pending(runtime: object, *, error: BaseException | None = None,
                   status: str = "REFUSED") -> None:
    if not enabled():
        return
    scope = getattr(runtime, "_read_attempt_scope", None)
    if scope is not None:
        scope.finish(status, error=error)
        runtime._read_attempt_scope = None


def begin_read_attempt(runtime: object, arguments: object, *, call_id: object = None,
                       batch_index: int | None = None) -> ReadAttemptScope | None:
    if not enabled():
        return None
    finish_pending(runtime)
    scope = ReadAttemptScope(runtime, arguments, call_id=call_id, batch_index=batch_index)
    runtime._read_attempt_scope = scope
    runtime._read_last_attempt_call_id = call_id
    return scope


def begin_nested_read_attempt(runtime: object, arguments: object, *, call_id: object = None,
                              ) -> tuple[ReadAttemptScope | None, ReadAttemptScope | None]:
    """Open an inner audit without closing an enclosing read attempt.

    Returns (scope, prior); the caller must hand prior to end_nested_read_attempt
    in a finally so the outer attempt keeps its own lifecycle and status.
    """
    if not enabled():
        return None, None
    prior = getattr(runtime, "_read_attempt_scope", None)
    runtime._read_attempt_scope = None
    return begin_read_attempt(runtime, arguments, call_id=call_id), prior


def end_nested_read_attempt(runtime: object, scope: ReadAttemptScope | None,
                            prior: ReadAttemptScope | None, *,
                            error: BaseException | None = None) -> None:
    if scope is not None:
        scope.finish(error=error)  # idempotent; only an unfinished scope changes
    runtime._read_attempt_scope = prior


def refused_calls(runtime: object, calls: object) -> None:
    """Account identifiable reads in batches rejected before worker dispatch."""
    if not enabled() or not isinstance(calls, list):
        return
    for index, call in enumerate(calls):
        fn = call.get("function") if isinstance(call, Mapping) else None
        if isinstance(fn, Mapping) and fn.get("name") == "read":
            scope = begin_read_attempt(runtime, fn.get("arguments"),
                                       call_id=call.get("id"), batch_index=index)
            scope.finish("NOT_DISPATCHED")

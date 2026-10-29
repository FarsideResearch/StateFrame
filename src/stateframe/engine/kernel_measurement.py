"""Durable experiment identity and wire-level evidence, independent of model advice."""
from __future__ import annotations

import contextvars
import copy
import functools
import hashlib
import http.client
import json
import math
import os
import platform
import stat
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from kernel_effect_boundary import Mode
from kernel_frame_request_allocation import FrameRequestAllocation
from kernel_launch_identity_cost import (
    ENVIRONMENT,
    PhaseMeter,
    ProviderBoundaryContractError,
    ProviderMalformedResponseError,
    SEMANTIC_ERROR,
    UNKNOWN,
    build_run_certificate,
    enforce_provider_body,
    reasoning_profile,
)

REQUEST_FINAL_WIRE_PREPARED_SCHEMA = "stateframe-request-final-wire-prepared/1"
REQUEST_BODY_SCHEMA = "stateframe-request-body/1"


def _journal_safe_fields(payload):
    """Emit boundary (10-05 L109): rename journal-RESERVED keys; see kernel_continuation_journal."""
    from kernel_continuation_journal import journal_safe_fields

    return journal_safe_fields(payload)


def _emit_scope_turns_final_wire_events(
        journal: object, wire: bytes, parsed_messages: object, *,
        runtime: object | None, request_id: str | None, submission_event_id: str,
) -> None:
    """Journal scoped guidance and optional ranking against exact submitted bytes."""
    from kernel_release_wire_claims import scope_turns_on_wire

    scope_turns = scope_turns_on_wire(parsed_messages)
    body_sha256 = "sha256:" + hashlib.sha256(wire).hexdigest()
    for scope_turn in scope_turns:
        journal.emit(
            "edit_nturn_scope_turns_on_wire",
            producer="kernel_step_contract",
            request_id=request_id,
            body_sha256=body_sha256,
            **scope_turn,
            on_wire=True,
            dispatch=False,
        )

    from kernel_capability_floor import (
        scope_turns_notice_receipt,
        scope_turns_ranked_receipt,
    )

    notice = scope_turns_notice_receipt(wire, runtime=runtime)
    if notice is not None:
        notice_event = notice.pop("notice_event", None)
        if notice_event == "autopsy_r2_scope_turns_notice_fired":
            journal.emit(
                str(notice_event),
                producer="kernel_step_contract",
                request_id=request_id,
                submission_event_id=submission_event_id,
                on_wire=True,
                wire_transport=True,
                dispatch=False,
                **notice,
            )

    ranked = scope_turns_ranked_receipt(wire, runtime=runtime)
    if ranked is not None:
        ranked_event = ranked.pop("ranked_event", None)
        if ranked_event == "autopsy_r2_scope_turns_ranked":
            journal.emit(
                str(ranked_event),
                producer="kernel_step_contract",
                request_id=request_id,
                submission_event_id=submission_event_id,
                on_wire=True,
                dispatch=False,
                **ranked,
            )


def _stop_pre_dispatch_contract(
        capture: object, *, event: str, code: str, detail: str,
        failures: Sequence[str] = (),
        error: BaseException | None = None,
) -> None:
    """Journal a typed harness STOP, then raise the ORIGINAL contract error.

    The raised type/message is a contract (bin/release_request_proof.py and callers
    distinguish WireTruthBlocked / RequestPreparationContractError); the STOP event
    adds the stable ``code``. Journal details are bounded allowlisted messages and
    failure kinds, so paths, source text, and secrets never enter this event.
    """
    from kernel_boundary_recovery import (
        safe_error_class, safe_error_detail, safe_failure_codes,
        safe_proof_failure_kinds,
    )
    if error is None:
        from kernel_request_reprepare import RequestPreparationContractError
        error = RequestPreparationContractError(detail)
    error_failures = getattr(error, "failures", ())
    safe_failures = safe_failure_codes(error_failures, fallback=failures)
    proof_failures = getattr(error, "proof_failures", None)
    if not proof_failures:
        proof_failures = getattr(capture, "request_source_final_delivery", None)
    proof_kinds = safe_proof_failure_kinds(proof_failures)
    error_detail = safe_error_detail(error)
    error_class = safe_error_class(error)
    safe_code = safe_failure_codes((code,))[0]
    emit = getattr(getattr(capture, "journal", None), "emit", None)
    if callable(emit):
        emit(
            event,
            producer="kernel_measurement",
            request_id=getattr(capture, "request_id", None),
            step=getattr(capture, "step", None),
            code=safe_code,
            error_class=error_class,
            error_detail=error_detail,
            failures=safe_failures,
            proof_failure_kinds=proof_kinds,
            outcome=HARNESS_CONTRACT,
            failure_origin="harness",
            dispatch=False,
        )
    for name, value in (("harness_contract", True), ("failure_origin", "harness"),
                        ("outcome", HARNESS_CONTRACT), ("stop_code", safe_code),
                        ("failures", tuple(safe_failures)),
                        ("proof_failure_kinds", tuple(proof_kinds))):
        try:
            setattr(error, name, value)
        except (AttributeError, TypeError):
            pass
    raise error


def _assert_framefit5_provider_boundary(
        capture: object, wire: bytes, assembler_messages: bytes | None,
) -> None:
    """Fail before dispatch if serialized FrameFit5 messages escaped final assembly."""
    if assembler_messages is None:
        return
    try:
        submitted = json.loads(wire)
        messages = submitted.get("messages") if isinstance(submitted, Mapping) else None
        if not isinstance(messages, list):
            raise ValueError("provider body has no message list")
        submitted_messages = json.dumps(messages).encode("utf-8")
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        submitted_messages = b""
    expected_sha256 = "sha256:" + hashlib.sha256(assembler_messages).hexdigest()
    submitted_sha256 = "sha256:" + hashlib.sha256(submitted_messages).hexdigest()
    matched = submitted_messages == assembler_messages
    capture.journal.emit(
        "framefit5_provider_boundary_checked",
        producer="kernel_measurement",
        request_id=getattr(capture, "request_id", None),
        step=getattr(capture, "step", None),
        assembler_messages_sha256=expected_sha256,
        provider_messages_sha256=submitted_sha256,
        matched=matched,
        dispatch=False,
    )
    if matched:
        return
    from kernel_request_reprepare import RequestPreparationContractError

    detail = "final provider messages differ from shared FrameFit5 assembly"
    _stop_pre_dispatch_contract(
        capture,
        event="request_final_validation_stopped",
        code="final_request_bytes_changed",
        detail=detail,
        failures=("final_request_bytes_changed",),
        error=RequestPreparationContractError(detail),
    )


def _submitted_tool_names(body: Mapping[str, object]) -> list[str]:
    raw_tools = body.get("tools", ())
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes, bytearray)):
        return []
    names: list[str] = []
    for tool in raw_tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if isinstance(function, Mapping):
            names.append(str(function.get("name", "")))
    return names


MISSING = "MISSING"
HARNESS_TRANSPORT = "HARNESS_TRANSPORT"
HARNESS_CONTRACT = "HARNESS_CONTRACT"
# Harness-owned terminal outcomes: the run ends typed, never as a model failure.
HARNESS_TERMINAL_OUTCOMES = frozenset({HARNESS_TRANSPORT, HARNESS_CONTRACT})
FAST_TRANSPORT_LIMIT_NS = 2_000_000_000
RELAY_ERROR_LABELS = frozenset({
    "upstream_reset", "upstream_timeout", "upstream_refused", "upstream_error",
    "downstream_disconnect", "relay_ttft_timeout",
})
ACTIVE_REQUEST: contextvars.ContextVar[RequestCapture | None] = contextvars.ContextVar(
    "stateframe_request", default=None)


@dataclass(slots=True)
class _BoundaryRecoveryContext:
    original_request_id: str
    phase: str
    runtime: object | None
    authoritative_body: dict[str, object]
    endpoint: str
    key: str
    deadline: float | None
    side_call_contract: object
    required: bool
    transport_attempt_limit: int | None
    journal_timeouts: bool
    preparation: object | None = None
    transport_body: dict[str, object] | None = None
    repair_body: dict[str, object] | None = None
    wire_truth_error: BaseException | None = None
    submitted: bool = False
    recovery_attempted: bool = False
    wire_truth_rebuild_attempted: bool = False
    body_digest: str | None = None


def _request_boundary_contract_error(error: BaseException) -> bool:
    from kernel_request_reprepare import RequestPreparationContractError
    from kernel_completion_owner import RequestActionContractError
    from kernel_wire_truth_gate import WireTruthBlocked

    if isinstance(error, RequestActionContractError):
        return getattr(error, "boundary_recovery_eligible", True) is not False
    if isinstance(error, (RequestPreparationContractError, ProviderBoundaryContractError,
                          WireTruthBlocked)):
        return True
    return False


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


MAX_FRAMEFIT_RECEIPT_EVENTS = 4096


def framefit_receipt_projection(event: Mapping[str, object]) -> dict[str, object] | None:
    """Project only durable action/check receipt facts needed by FRAMEFIT5."""
    kind = event.get("kind")
    if kind not in {"action_receipt", "check_receipt"}:
        return None
    event_id = event.get("source_event_id", event.get("event_id"))
    sequence = event.get("source_sequence", event.get("sequence"))
    projected: dict[str, object] = {"kind": kind}
    if isinstance(event_id, str) and event_id:
        projected["event_id"] = event_id[:160]
    if type(sequence) is int and sequence > 0:
        projected["sequence"] = sequence

    def safe_fields(source: Mapping[str, object], names: Sequence[str]) -> dict[str, object]:
        result: dict[str, object] = {}
        for name in names:
            value = source.get(name)
            if isinstance(value, str) and len(value) <= 512:
                result[name] = value
            elif type(value) is int and value >= 0:
                result[name] = value
        return result

    if kind == "action_receipt":
        receipt = event.get("receipt")
        if not isinstance(receipt, Mapping):
            return None
        projected["receipt"] = safe_fields(receipt, (
            "receipt_id", "kind", "result_class", "target_path",
            "workspace_revision", "request_id", "turn",
        ))
        return projected

    projected.update(safe_fields(event, (
        "receipt_id", "status", "revision", "tested_rev", "producer",
        "request_id", "execution_id", "selector",
    )))
    return projected


JOURNAL_BLOB_THRESHOLD_BYTES = 4096
_JOURNAL_BLOB_MAX_BYTES = 256 * 1024 * 1024


def _blob_reference(value: object) -> tuple[str, int] | None:
    if not isinstance(value, Mapping) or set(value) != {"$blob", "bytes"}:
        return None
    blob_hash, size = value.get("$blob"), value.get("bytes")
    if (not isinstance(blob_hash, str) or len(blob_hash) != 64
            or any(char not in "0123456789abcdef" for char in blob_hash)
            or type(size) is not int or size <= 0):
        return None
    if size > _JOURNAL_BLOB_MAX_BYTES:
        raise ValueError("journal blob reference exceeds the supported size")
    return blob_hash, size


def _journal_blob_directory(journal_path: str | os.PathLike[str]) -> Path:
    run_dir = Path(journal_path).parent
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("journal run directory must be a real directory")
    blob_dir = run_dir / "blobs"
    if blob_dir.is_symlink() or not blob_dir.is_dir():
        raise FileNotFoundError("journal blob store is missing")
    return blob_dir


def _ensure_journal_blob_directory(journal_path: str | os.PathLike[str]) -> Path:
    run_dir = Path(journal_path).parent
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("journal run directory must be a real directory")
    blob_dir = run_dir / "blobs"
    created = False
    try:
        blob_dir.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        pass
    if blob_dir.is_symlink() or not blob_dir.is_dir():
        raise ValueError("journal blob store must be a real directory")
    if created:
        directory_fd = os.open(run_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return blob_dir


def _read_journal_blob(journal_path: str | os.PathLike[str], blob_hash: str) -> bytes:
    journal_location = os.fspath(journal_path)
    if "::" in journal_location:
        import tarfile
        from pathlib import PurePosixPath

        archive_name, member_name = journal_location.split("::", 1)
        relative_member = PurePosixPath(member_name)
        if relative_member.is_absolute() or ".." in relative_member.parts:
            raise ValueError("archived journal member path is unsafe")
        blob_member = str(relative_member.parent / "blobs" / f"{blob_hash}.zlib")
        with tarfile.open(archive_name, "r:*") as archive:
            try:
                member = archive.getmember(blob_member)
            except KeyError as error:
                raise FileNotFoundError("archived journal blob is missing") from error
            if not member.isfile() or not 0 <= member.size <= _JOURNAL_BLOB_MAX_BYTES:
                raise ValueError("archived journal blob is not a bounded regular file")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError("archived journal blob cannot be read")
            compressed_size = member.size
            compressed = extracted.read(compressed_size + 1)
    else:
        blob_path = _journal_blob_directory(journal_path) / f"{blob_hash}.zlib"
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        fd = os.open(blob_path, flags)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > _JOURNAL_BLOB_MAX_BYTES:
                raise ValueError("journal blob is not a bounded regular file")
            # Buffered reads can allocate the requested size up front. Reading
            # the 256 MiB cap for every tiny blob defeats bounded journal reads.
            compressed_size = info.st_size
            compressed = handle.read(compressed_size + 1)
    if len(compressed) != compressed_size:
        raise ValueError("journal blob size changed while reading")
    if len(compressed) > _JOURNAL_BLOB_MAX_BYTES:
        raise ValueError("compressed journal blob exceeds the supported size")
    try:
        decoder = zlib.decompressobj()
        data = decoder.decompress(compressed, _JOURNAL_BLOB_MAX_BYTES + 1)
        if (len(data) > _JOURNAL_BLOB_MAX_BYTES or not decoder.eof
                or decoder.unused_data or decoder.unconsumed_tail):
            raise ValueError("journal blob compression framing is invalid")
    except zlib.error as error:
        raise ValueError("journal blob cannot be decompressed") from error
    # A parent blob may encode child refs; verify its digest only after the
    # resolver hydrates those refs back into the logical canonical value.
    return data


def _write_journal_blob(journal_path: str | os.PathLike[str], blob_hash: str,
                        logical_data: bytes, stored_data: bytes,
                        cache: dict[str, int] | None) -> None:
    size = len(logical_data)
    if size > _JOURNAL_BLOB_MAX_BYTES or len(stored_data) > _JOURNAL_BLOB_MAX_BYTES:
        raise ValueError("journal payload exceeds the supported blob size")
    if cache is not None and blob_hash in cache:
        if cache[blob_hash] != size:
            raise ValueError("journal blob hash was reused with a different size")
        return
    blob_dir = _ensure_journal_blob_directory(journal_path)
    blob_path = blob_dir / f"{blob_hash}.zlib"
    try:
        _read_journal_blob(journal_path, blob_hash)
    except FileNotFoundError:
        fd, temporary_name = tempfile.mkstemp(prefix=".blob-", dir=blob_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                os.fchmod(handle.fileno(), 0o600)
                compressed = zlib.compress(stored_data, level=6)
                if len(compressed) > _JOURNAL_BLOB_MAX_BYTES:
                    raise ValueError("compressed journal blob exceeds the supported size")
                handle.write(compressed)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_name, blob_path)
            except FileExistsError:
                existing = resolve_journal_blob_refs(
                    {"$blob": blob_hash, "bytes": size}, journal_path)
                if canonical_bytes(existing) != logical_data:
                    raise ValueError("journal blob content collision")
            else:
                directory_fd = os.open(blob_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    else:
        existing = resolve_journal_blob_refs(
            {"$blob": blob_hash, "bytes": size}, journal_path)
        if canonical_bytes(existing) != logical_data:
            raise ValueError("journal blob content collision")
    if cache is not None:
        cache[blob_hash] = size


def _externalize_journal_value(value: object, journal_path: str | os.PathLike[str],
                               cache: dict[str, int] | None,
                               threshold_bytes: int) -> object:
    if isinstance(value, Mapping) and "$blob" in value:
        raise ValueError("journal payload uses the reserved blob-reference key")
    original = canonical_bytes(value)
    if len(original) > threshold_bytes and len(original) > _JOURNAL_BLOB_MAX_BYTES:
        raise ValueError("journal payload exceeds the supported blob size")
    if isinstance(value, Mapping):
        normalized: object = {
            key: _externalize_journal_value(item, journal_path, cache, threshold_bytes)
            for key, item in value.items()
        }
    elif isinstance(value, (list, tuple)):
        normalized = [
            _externalize_journal_value(item, journal_path, cache, threshold_bytes)
            for item in value
        ]
    else:
        normalized = value
    stored = canonical_bytes(normalized)
    if len(original) <= threshold_bytes:
        return normalized
    blob_hash = hashlib.sha256(original).hexdigest()
    _write_journal_blob(journal_path, blob_hash, original, stored, cache)
    return {"$blob": blob_hash, "bytes": len(original)}


def externalize_journal_fields(fields: Mapping[str, object],
                               journal_path: str | os.PathLike[str],
                               cache: dict[str, int] | None = None,
                               *, threshold_bytes: int = JOURNAL_BLOB_THRESHOLD_BYTES
                               ) -> dict[str, object]:
    """Store oversized event values once and return compact on-disk fields."""
    if (type(threshold_bytes) is not int or threshold_bytes < 0
            or threshold_bytes >= _JOURNAL_BLOB_MAX_BYTES):
        raise ValueError("journal blob threshold is outside the supported range")
    return {
        key: _externalize_journal_value(value, journal_path, cache, threshold_bytes)
        for key, value in fields.items()
    }


def _resolve_journal_blob_value(value: object, journal_path: str | os.PathLike[str],
                                cache: dict[str, object], active: set[str]) -> object:
    reference = _blob_reference(value)
    if reference is None and isinstance(value, Mapping) and "$blob" in value:
        raise ValueError("malformed journal blob reference")
    if reference is not None:
        blob_hash, expected_size = reference
        if blob_hash in cache:
            cached = cache[blob_hash]
            if (not isinstance(cached, tuple) or len(cached) != 2
                    or cached[0] != expected_size):
                raise ValueError("journal blob reference size conflicts with verified content")
            return copy.deepcopy(cached[1])
        if blob_hash in active:
            raise ValueError("journal blob references contain a cycle")
        active.add(blob_hash)
        try:
            encoded = _read_journal_blob(journal_path, blob_hash)
            decoded = json.loads(encoded)
            if canonical_bytes(decoded) != encoded:
                raise ValueError("journal blob payload is not canonical JSON")
            resolved = _resolve_journal_blob_value(decoded, journal_path, cache, active)
            logical = canonical_bytes(resolved)
            if (len(logical) != expected_size
                    or hashlib.sha256(logical).hexdigest() != blob_hash):
                raise ValueError("journal blob content hash or size mismatch")
            cache[blob_hash] = (expected_size, resolved)
        finally:
            active.remove(blob_hash)
        return copy.deepcopy(resolved)
    if isinstance(value, Mapping):
        return {
            key: _resolve_journal_blob_value(item, journal_path, cache, active)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_journal_blob_value(item, journal_path, cache, active) for item in value
        ]
    return value


def resolve_journal_blob_refs(value: object,
                              journal_path: str | os.PathLike[str],
                              cache: dict[str, object] | None = None) -> object:
    """Hydrate verified blob references relative to their per-run journal."""
    return _resolve_journal_blob_value(
        value, journal_path, cache if cache is not None else {}, set()
    )


def iter_journal_records(journal_path: str | os.PathLike[str]) -> Iterator[dict[str, object]]:
    """Read event JSONL and hydrate verified refs with a per-file blob cache."""
    source = Path(journal_path)
    blob_cache: dict[str, object] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (TypeError, ValueError, UnicodeError) as error:
                raise ValueError(f"invalid journal JSON at {source}:{line_number}") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"journal event is not an object at {source}:{line_number}")
            resolved = resolve_journal_blob_refs(value, source, blob_cache)
            if not isinstance(resolved, Mapping):
                raise ValueError(f"journal event did not resolve to an object at {source}:{line_number}")
            yield dict(resolved)


def skip_capped_length_retry_enabled() -> bool:
    raw = os.environ.get("REACT_SKIP_CAPPED_LENGTH_RETRY", "0")
    if raw not in {"0", "1"}:
        raise ValueError("REACT_SKIP_CAPPED_LENGTH_RETRY must be 0 or 1")
    return raw == "1"


def output_token_limits(initial: int = 4000) -> tuple[int, int]:
    """Allow explicit reasoning headroom without changing legacy request limits."""
    first = os.environ.get('REACT_INITIAL_OUTPUT_TOKENS')
    ceiling = os.environ.get('REACT_MAX_OUTPUT_TOKENS')
    from kernel_endpoint_size import ENDPOINT_SIZE
    chosen_ceiling = min(16000, ENDPOINT_SIZE.output_tokens) if ENDPOINT_SIZE.configured else 16000
    if first is None and ceiling is None:
        return (min(initial, chosen_ceiling) if ENDPOINT_SIZE.configured else initial), chosen_ceiling
    try:
        initial = int(first) if first is not None else initial
        maximum = int(ceiling) if ceiling is not None else chosen_ceiling
    except ValueError as error:
        raise ValueError('output token limits must be positive integers') from error
    if initial <= 0 or (maximum < initial and (ceiling is not None or not ENDPOINT_SIZE.configured)):
        raise ValueError('output token limits require 0 < initial <= maximum')
    if ENDPOINT_SIZE.configured:
        # Existing release profiles pin 24k. A smaller chosen slot must still
        # work when the operator changes only STATEFRAME_CONTEXT_TOKENS.
        maximum = min(maximum, ENDPOINT_SIZE.output_tokens)
        initial = min(initial, maximum)
    return initial, maximum


DEFAULT_PER_REQUEST_TIMEOUT_S = 600.0
TIMEOUT_RETRY_FROZEN_ENV = "STATEFRAME_TIMEOUT_RETRY_FROZEN"
TTFT_RETRY_BUDGET_ENV = "STATEFRAME_TTFT_RETRY_BUDGET"
TTFT_RETRY_BUDGET_SECONDS_ENV = "STATEFRAME_TTFT_RETRY_BUDGET_S"
DEFAULT_TTFT_RETRY_BUDGET_SECONDS = 600.0
TTFT_RETRY_DELAYS_SECONDS = (0.0, 15.0, 30.0, 60.0)


def timeout_retry_frozen_enabled() -> bool:
    """Use transport retries over the already-qualified request bytes by default."""
    value = os.environ.get(TIMEOUT_RETRY_FROZEN_ENV, "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{TIMEOUT_RETRY_FROZEN_ENV} must be 'on' or 'off'")
    return value == "on"


def ttft_retry_budget_enabled() -> bool:
    """Keep relay TTFT retries outside the ordinary transport attempt budget by default."""
    value = os.environ.get(TTFT_RETRY_BUDGET_ENV, "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{TTFT_RETRY_BUDGET_ENV} must be 'on' or 'off'")
    return value == "on"


def ttft_retry_budget_seconds() -> float:
    """Return the positive total elapsed-time budget for relay TTFT retries."""
    raw = os.environ.get(
        TTFT_RETRY_BUDGET_SECONDS_ENV,
        str(DEFAULT_TTFT_RETRY_BUDGET_SECONDS),
    )
    try:
        seconds = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{TTFT_RETRY_BUDGET_SECONDS_ENV} must be finite and positive") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(
            f"{TTFT_RETRY_BUDGET_SECONDS_ENV} must be finite and positive")
    return seconds


def _ttft_retry_delay_seconds(retry_index: int) -> float:
    """Use a short bounded backoff that stops growing after the fourth retry."""
    if type(retry_index) is not int or retry_index < 1:
        raise ValueError("TTFT retry index must be a positive integer")
    return TTFT_RETRY_DELAYS_SECONDS[
        min(retry_index - 1, len(TTFT_RETRY_DELAYS_SECONDS) - 1)
    ]


def _legacy_request_timeouts() -> tuple[float, float]:
    """Return the historical socket and wall limits for the surrounding run."""
    raw = os.environ.get('REACT_REQUEST_TIMEOUT_S')
    if raw is None:
        return 300.0, 420.0
    seconds = float(raw)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('REACT_REQUEST_TIMEOUT_S must be finite and positive')
    return seconds, seconds + 120.0


def per_request_timeout_seconds() -> float | None:
    """Return the independent call cap, preserving the no-knob legacy path."""
    raw = os.environ.get("STATEFRAME_PER_REQUEST_TIMEOUT_S")
    run_timeout = os.environ.get("REACT_REQUEST_TIMEOUT_S")
    if raw is None and run_timeout is None:
        return None
    if raw is None:
        return DEFAULT_PER_REQUEST_TIMEOUT_S
    try:
        seconds = float(raw)
    except ValueError as error:
        raise ValueError(
            "STATEFRAME_PER_REQUEST_TIMEOUT_S must be finite and positive") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("STATEFRAME_PER_REQUEST_TIMEOUT_S must be finite and positive")
    return seconds


def request_timeouts() -> tuple[float, float]:
    """Return per-request limits, retaining exact legacy values when both knobs are absent."""
    per_request = per_request_timeout_seconds()
    legacy = _legacy_request_timeouts()
    if per_request is None:
        return legacy
    return per_request, per_request


def request_timeout_budget_seconds() -> float:
    """Return the unchanged historical wall allowance for a captured send."""
    return _legacy_request_timeouts()[1]


def _safe_timeout_endpoint(endpoint: str) -> str:
    """Keep endpoint identity without credentials, query values, or fragments."""
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        host = parsed.hostname or ""
        try:
            port = parsed.port
        except ValueError:
            port = None
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if port is not None:
            host = f"{host}:{port}"
        return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))[:512]
    except (TypeError, ValueError):
        safe = str(endpoint).rsplit("@", 1)[-1]
        return safe.split("?", 1)[0].split("#", 1)[0][:512]


def journal_provider_request_timeout(
        journal: object | None, *, request_id: str | None, step: int | None,
        timeout_seconds: float, attempt: int, endpoint: str,
        retry_scheduled: bool) -> object | None:
    """Persist a provider timeout as a host fact, never as a model verdict."""
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        return None
    return emitter(
        "provider_request_timeout",
        producer="kernel_measurement",
        fact_owner="HOST",
        request_id=request_id,
        step=step,
        seconds=float(timeout_seconds),
        attempt=attempt,
        endpoint=_safe_timeout_endpoint(endpoint),
        retry_scheduled=bool(retry_scheduled),
        failure_origin="harness",
        outcome=HARNESS_TRANSPORT,
        dispatch=False,
    )


def is_provider_liveness_timeout_error(error: BaseException) -> bool:
    """Distinguish lost endpoint liveness from the independent total deadline."""
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if type(current).__name__ == "ProviderLivenessTimeout":
            return True
        for name in ("__cause__", "__context__", "reason"):
            nested = getattr(current, name, None)
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def is_provider_request_timeout_error(error: BaseException) -> bool:
    """Recognize socket, future, wrapped-URL, and typed transport timeouts."""
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, TimeoutError):
            return True
        error_type = getattr(current, "error_type", None)
        if isinstance(error_type, str) and error_type in {
                "TimeoutError", "socket.timeout", "TimeoutExpired"}:
            return True
        if getattr(current, "relay_error", None) == "upstream_timeout":
            return True
        for name in ("__cause__", "__context__", "reason"):
            nested = getattr(current, name, None)
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def _configured_endpoint_profile() -> object | None:
    """Load the measured profile only when an operator explicitly configured it."""

    if not os.environ.get("FRAME_LAB_ENDPOINT_PROFILE_FILE"):
        return None
    from kernel_optional_engine_module import optional_module
    extension = optional_module("STATEFRAME_ENDPOINT_PROFILE_MODULE")
    if extension is None:
        return None
    try:
        return extension.endpoint_profile()
    except (AttributeError, OSError, ValueError):
        return None


def _prompt_token_estimate(body: Mapping[str, object]) -> int:
    explicit = body.get("prompt_tokens")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit >= 0:
        return explicit
    messages = body.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        return max(1, (len(canonical_bytes(messages)) + 3) // 4)
    return max(1, (len(canonical_bytes(body)) + 3) // 4)


REBUDGET_ENV = "STATEFRAME_FINAL_WIRE_REBUDGET"


def _rebudget_enabled() -> bool:
    """STATEFRAME_FINAL_WIRE_REBUDGET on (default) | off; anything else fails closed."""
    mode = str(os.environ.get(REBUDGET_ENV, "on")).strip().lower()
    if mode not in {"on", "off"}:
        raise ValueError(f"{REBUDGET_ENV} must be 'on' or 'off'")
    return mode == "on"


def refit_guarded_body(candidate, *, limit, budget, finalize, guard, emit=None):
    """Re-fit a request the final-wire guard grew past the budgeted byte cap.

    10-15 live batch 1198 (full engine): the final budget pass dropped tools from the menu, then the
    final-wire guard rewrote prose that still named those tools (sanitize_tool_references:
    "use grep" -> "use only a tool listed in this request"), adding 58 bytes AFTER budgeting; the
    81,997-of-82,026-byte request became 82,055 and final validation raised
    request_bytes_exceed_max_request_bytes, killing the run. Budget the GUARDED body (so the guard's
    text is counted and other content is trimmed instead), re-finalize, and guard again (the guard
    is idempotent on its own output). One pass; final validation stays strict.
    """
    if not _rebudget_enabled():
        return candidate
    if type(limit) is not int or limit <= 0:
        return candidate
    before = len(json.dumps(candidate).encode())
    if before <= limit:
        return candidate
    refit = guard(finalize(budget(candidate)))
    if emit is not None:
        emit(limit=limit, bytes_before=before, bytes_after=len(json.dumps(refit).encode()))
    return refit


@dataclass(frozen=True, slots=True)
class HostSpan:
    """Host-owned monotonic interval; timing never enters a model request."""

    name: str
    started_ns: int
    ended_ns: int
    boundary: str = "recorded_prefix"
    sequence: int | None = None
    action_id: str | None = None
    attributes: tuple[tuple[str, str], ...] = ()
    parent_span: str | None = None
    cpu_ns: int | None = None
    cache_hits: int = 0
    cache_misses: int = 0
    bytes_scanned: int = 0
    records_folded: int = 0
    files_read: int = 0
    fsyncs: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("host span name is required")
        if (type(self.started_ns) is not int or type(self.ended_ns) is not int
                or self.started_ns < 0 or self.ended_ns < self.started_ns):
            raise ValueError("host span clock is not monotonic")
        if self.boundary not in {
                "recorded_prefix", "terminal_drain", "initialization", "finalization", "export",
                "request_preparation"}:
            raise ValueError(f"unsupported host span boundary: {self.boundary}")
        counters = (self.cache_hits, self.cache_misses, self.bytes_scanned,
                    self.records_folded, self.files_read, self.fsyncs)
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("host span counters must be non-negative integers")
        if self.cpu_ns is not None and (type(self.cpu_ns) is not int or self.cpu_ns < 0):
            raise ValueError("host span cpu_ns must be non-negative")

    @property
    def elapsed_ns(self) -> int:
        return self.ended_ns - self.started_ns

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "started_ns": self.started_ns,
            "ended_ns": self.ended_ns, "elapsed_ns": self.elapsed_ns,
            "boundary": self.boundary, "sequence": self.sequence,
            "action_id": self.action_id,
            "attributes": dict(self.attributes), "parent_span": self.parent_span,
            "cpu_ns": self.cpu_ns, "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses, "bytes_scanned": self.bytes_scanned,
            "records_folded": self.records_folded, "files_read": self.files_read,
            "fsyncs": self.fsyncs,
        }


_HOST_PROFILE_SPAN_NAMES = (
    "prepare",
    "structural_index_build",
    "callsite_census_build_packet",
    "callsite_census_prepare_request",
    "source_closure_projection",
    "source_window_projection",
    "binding_computation",
    "request_budget",
    "request_diet",
    "final_wire_truth_validation",
    "journal_emit",
    "journal_blob_io",
    "append_fsync",
    "retained_context_subscribers",
    "subscriber_fanout",
    "provider",
    "runtime_initialization",
    "request_source_target_reservation",
    "request_source_literal_site_inventory",
    "request_source_closure_build",
    "request_source_unit_materialization",
    "request_source_windowing",
    "source_anchor_resolution",
    "source_message_window",
    "outline_resolution",
    "census_structure_build",
    "census_structure_refresh",
    "census_call_projection",
    "frame_series_restrictions_projection",
    "tool_menu_composition",
    "tool_menu_repeat_guard",
    "tool_menu_work_record_restore",
    "tool_menu_terminal_green_lock",
    "tool_menu_closure_reserve",
    "readiness_menu_projection",
    "green_lock_menu_projection",
    "rulebook_mechanism_render",
    "first_turn_to_submit",
    "tokenlab_v22_unified_ledger",
    "runtime_snapshot_to_organ_wiring",
    "environment_observation_to_tokenlab_ledger",
    "outline_parser_to_census_outcome",
    "mechanism_notice_to_source_closure",
    "continuation_projection_to_tool_menu",
    "unified_ledger_to_decision_cache",
    "frame_series_restrictions_to_tool_menu",
    "readiness_menu_to_green_lock_authority",
    "vacuity_verdict_to_tokenlab_ledger",
)
_HOST_SPANS_ENV = "STATEFRAME_HOST_SPANS"


HOST_SPAN_TREE_MAX_ROWS = 512
HOST_SPAN_TREE_AGGREGATE_MIN = 4


def bounded_host_span_tree(rows: Sequence[Mapping[str, object]], *,
                           max_rows: int = HOST_SPAN_TREE_MAX_ROWS,
                           aggregate_min: int = HOST_SPAN_TREE_AGGREGATE_MIN,
                           ) -> tuple[list[dict[str, object]], int]:
    """Return (bounded rows, dropped row count) for one request's span trace.

    Leaf spans (no children) repeated >= aggregate_min times under the same parent with the same
    name become ONE row: count, elapsed_ns = summed time, started_ns = earliest start,
    ended_ns = latest end, max_elapsed_ns, aggregated=True. Structural spans are kept as-is.
    If still over max_rows, the slowest rows are kept (parents of kept rows are always kept)."""
    plain = [dict(row) for row in rows if isinstance(row, Mapping)]
    parents = {str(row.get("parent_span")) for row in plain if row.get("parent_span") is not None}
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    kept: list[dict[str, object]] = []
    for row in plain:
        span_id = str(row.get("span_id"))
        if span_id in parents:
            kept.append(row)
            continue
        groups.setdefault((str(row.get("parent_span")), str(row.get("name"))), []).append(row)
    for (parent, name), members in groups.items():
        if len(members) < aggregate_min:
            kept.extend(members)
            continue
        elapsed = [max(0, int(m.get("elapsed_ns") or 0)) for m in members]
        starts = [int(m["started_ns"]) for m in members if type(m.get("started_ns")) is int]
        ends = [int(m["ended_ns"]) for m in members if type(m.get("ended_ns")) is int]
        kept.append({
            "span_id": f"agg:{parent}:{name}",
            "parent_span": members[0].get("parent_span"),
            "name": name,
            "request_id": members[0].get("request_id"),
            "aggregated": True,
            "count": len(members),
            "elapsed_ns": sum(elapsed),
            "max_elapsed_ns": max(elapsed) if elapsed else 0,
            "started_ns": min(starts) if starts else None,
            "ended_ns": max(ends) if ends else None,
            "sequence": members[0].get("sequence"),
        })
    if len(kept) <= max_rows:
        return kept, 0
    by_id = {str(row.get("span_id")): row for row in kept}
    chosen: dict[str, dict[str, object]] = {}
    for row in sorted(kept, key=lambda r: -int(r.get("elapsed_ns") or 0)):
        chain = []
        current = row
        while current is not None and str(current.get("span_id")) not in chosen:
            chain.append(current)
            parent = current.get("parent_span")
            current = by_id.get(str(parent)) if parent is not None else None
        if len(chosen) + len(chain) > max_rows:
            continue
        for item in chain:
            chosen[str(item.get("span_id"))] = item
    bounded = [row for row in kept if str(row.get("span_id")) in chosen]
    return bounded, len(kept) - len(bounded)


@dataclass(slots=True)
class _HostSpanProfile:
    request_id: str
    step: int | None
    started_ns: int
    samples: list[tuple[str, int]]
    lock: threading.Lock
    recorder: object
    claimed: bool = False

    def add(self, name: str, elapsed_ns: int) -> None:
        with self.lock:
            if len(self.samples) >= 4096:
                self.samples.pop(0)
            self.samples.append((name, max(0, int(elapsed_ns))))

    def snapshot(self) -> tuple[tuple[str, int], ...]:
        with self.lock:
            return tuple(self.samples)


@dataclass(slots=True)
class _HostSpanHandle:
    recorder: object
    span_id: str
    name: str
    started_ns: int
    boundary: str
    sequence: int | None
    parent_span: str | None


_ACTIVE_HOST_SPAN_PROFILE: contextvars.ContextVar[_HostSpanProfile | None] = (
    contextvars.ContextVar("stateframe_host_span_profile", default=None)
)
_ACTIVE_HOST_SPAN_REQUEST_SCOPE: contextvars.ContextVar[bool] = (
    contextvars.ContextVar("stateframe_host_span_request_scope", default=False)
)
_ACTIVE_HOST_SPAN_STACK: contextvars.ContextVar[tuple[tuple[object, str], ...]] = (
    contextvars.ContextVar("stateframe_host_span_stack", default=())
)
_HOST_SPAN_PROFILES: dict[str, _HostSpanProfile] = {}
_HOST_SPAN_PROFILES_LOCK = threading.RLock()
_ACTIVE_TURN_TIMING: contextvars.ContextVar["TurnTimingRecorder | None"] = (
    contextvars.ContextVar("stateframe_turn_timing_recorder", default=None)
)


def turn_timing_enabled() -> bool:
    return os.environ.get("STATEFRAME_TURN_TIMING", "off").strip().lower() in {
        "on", "1", "true", "yes",
    }


def _turn_timing_bucket(name: str) -> str:
    lowered = name.lower()
    if "journal" in lowered or "fsync" in lowered or "blob_io" in lowered:
        return "journal_io"
    if "menu" in lowered or "readiness" in lowered:
        return "menu_composition"
    if "contract" in lowered or "gate" in lowered:
        return "contracts"
    if "budget" in lowered or "diet" in lowered:
        return "budget_diet"
    if "prepare" in lowered or "binding" in lowered or "initialization" in lowered:
        return "prepare_subphases"
    return "tool_execution"


class TurnTimingRecorder:
    """Collect non-overlapping host phases between a response and next submission."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._open: dict[str, object] | None = None

    def note_response(self, event: object) -> None:
        if not turn_timing_enabled() or not isinstance(event, Mapping):
            return
        try:
            provider_ns = max(0, int(event.get("elapsed_ns", 0) or 0))
        except (TypeError, ValueError):
            provider_ns = 0
        with self._lock:
            self._open = {
                "request_id": event.get("request_id"),
                "step": event.get("step"),
                "provider_ns": provider_ns,
                "started_ns": time.perf_counter_ns(),
                "spans": [],
                "manual": [],
            }

    @contextmanager
    def measure(self, name: str):
        with self._lock:
            interval = self._open
        if interval is None or not turn_timing_enabled():
            yield
            return
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            ended_ns = time.perf_counter_ns()
            with self._lock:
                if self._open is interval:
                    spans = interval.get("spans")
                    if isinstance(spans, list):
                        spans.append((name, started_ns, ended_ns))

    def start_step(self, name: str) -> None:
        if not turn_timing_enabled():
            return
        with self._lock:
            if self._open is None:
                return
            manual = self._open.get("manual")
            if isinstance(manual, list):
                manual.append((name, time.perf_counter_ns()))

    def finish_submission(self, event: object) -> dict[str, object] | None:
        with self._lock:
            interval = self._open
            self._open = None
        if interval is None:
            return None
        if not turn_timing_enabled():
            return None
        ended_ns = time.perf_counter_ns()
        started_ns = int(interval.get("started_ns", ended_ns))
        wall_gap_ns = max(0, ended_ns - started_ns)
        spans = list(interval.get("spans", ()))
        for name, step_started in interval.get("manual", ()):
            spans.append((name, step_started, ended_ns))
        clipped: list[tuple[str, int, int]] = []
        for name, start, end in spans:
            left = max(started_ns, int(start))
            right = min(ended_ns, int(end))
            if right > left:
                clipped.append((str(name), left, right))
        boundaries = {started_ns, ended_ns}
        for _name, start, end in clipped:
            boundaries.add(start)
            boundaries.add(end)
        ordered = sorted(boundaries)
        buckets = {
            "prepare_subphases": 0, "menu_composition": 0, "contracts": 0,
            "budget_diet": 0, "journal_io": 0, "tool_execution": 0,
        }
        breakdown: dict[str, int] = {}
        priority = {
            "tool_execution": 0, "prepare_subphases": 1, "menu_composition": 2,
            "contracts": 3, "budget_diet": 4, "journal_io": 5,
        }
        for left, right in zip(ordered, ordered[1:]):
            if right <= left:
                continue
            active = [row for row in clipped if row[1] <= left and row[2] >= right]
            if active:
                name, _start, _end = min(
                    active,
                    key=lambda row: (-priority[_turn_timing_bucket(row[0])],
                                     row[2] - row[1], row[0]),
                )
                bucket = _turn_timing_bucket(name)
            else:
                name, bucket = "tool_execution", "tool_execution"
            duration = right - left
            buckets[bucket] += duration
            breakdown[name] = breakdown.get(name, 0) + duration
        steps_sum_ns = sum(buckets.values())
        next_request_id = event.get("request_id") if isinstance(event, Mapping) else None
        return {
            "producer": "kernel_measurement",
            "request_id": interval.get("request_id"),
            "next_request_id": next_request_id,
            "step": interval.get("step"),
            "provider_ns": int(interval.get("provider_ns", 0)),
            "wall_gap_ns": wall_gap_ns,
            "steps_ns": buckets,
            "step_breakdown_ns": breakdown,
            "steps_sum_ns": steps_sum_ns,
            "reconciliation_error_ns": wall_gap_ns - steps_sum_ns,
            "reconciles_within_5_percent": (
                abs(wall_gap_ns - steps_sum_ns) <= max(1, wall_gap_ns * 0.05)),
            "dispatch": False,
        }


@contextmanager
def turn_timing_step(name: str):
    recorder = _ACTIVE_TURN_TIMING.get()
    if recorder is None:
        yield
        return
    with recorder.measure(name):
        yield


def activate_turn_timing(journal: object | None) -> None:
    recorder = _turn_timing_recorder(journal)
    if recorder is not None:
        _ACTIVE_TURN_TIMING.set(recorder)


def start_turn_timing_step(name: str) -> None:
    recorder = _ACTIVE_TURN_TIMING.get()
    if recorder is not None:
        recorder.start_step(name)


def _turn_timing_recorder(journal: object | None) -> TurnTimingRecorder | None:
    if journal is None or not turn_timing_enabled():
        return None
    host_spans = getattr(journal, "host_spans", None)
    if host_spans is None:
        return None
    recorder = getattr(host_spans, "_turn_timing_recorder", None)
    if isinstance(recorder, TurnTimingRecorder):
        return recorder
    try:
        recorder = TurnTimingRecorder()
        setattr(host_spans, "_turn_timing_recorder", recorder)
        return recorder
    except Exception:
        return None


def host_span_profiles_enabled() -> bool:
    value = os.environ.get(_HOST_SPANS_ENV, "on").strip().lower()
    return value in {"on", "1", "true", "yes"}


class HostSpanRecorder:
    """Bounded in-process spans and request-scoped monotonic profiling."""

    def __init__(self, *, limit: int = 4096) -> None:
        self.limit = max(1, int(limit))
        self._spans: list[HostSpan] = []
        self._trace_spans: list[dict[str, object]] = []
        self._trace_parent_ids: dict[str, str | None] = {}
        self._trace_names: dict[str, str] = {}
        self._trace_request_ids: dict[str, str | None] = {}
        self._span_sequence = 0
        self._lock = threading.RLock()
        self._turn_timing_recorder = TurnTimingRecorder()

    def begin_request_profile(self, request_id: object, step: object = None) -> _HostSpanProfile | None:
        if not host_span_profiles_enabled():
            return None
        normalized_id = str(request_id or "").strip()
        if not normalized_id:
            return None
        with _HOST_SPAN_PROFILES_LOCK:
            existing = _HOST_SPAN_PROFILES.get(normalized_id)
            if existing is not None:
                return existing
            if len(_HOST_SPAN_PROFILES) >= 256:
                _HOST_SPAN_PROFILES.pop(next(iter(_HOST_SPAN_PROFILES)))
            normalized_step = step if type(step) is int else None
            profile = _HostSpanProfile(
                normalized_id, normalized_step, time.perf_counter_ns(), [],
                threading.Lock(), self)
            _HOST_SPAN_PROFILES[normalized_id] = profile
            # The cold-start root and startup spans begin before RequestCapture exists.
            # Bind those pending trace rows to the first request that consumes them.
            with self._lock:
                for span_id, trace_request_id in tuple(self._trace_request_ids.items()):
                    if trace_request_id is None:
                        self._trace_request_ids[span_id] = normalized_id
                for row in self._trace_spans:
                    if row.get("request_id") is None:
                        row["request_id"] = normalized_id
            return profile

    @contextmanager
    def request_preparation_scope(self, step: object = None):
        """Start one request profile around preparation, reusing nested scopes."""
        active = _ACTIVE_HOST_SPAN_PROFILE.get()
        if active is not None:
            yield active, False
            return
        if _ACTIVE_HOST_SPAN_REQUEST_SCOPE.get():
            yield None, False
            return
        profile = self.begin_request_profile(uuid.uuid4().hex, step)
        if profile is None:
            yield None, False
            return
        token = _ACTIVE_HOST_SPAN_PROFILE.set(profile)
        try:
            yield profile, True
        finally:
            _ACTIVE_HOST_SPAN_PROFILE.reset(token)

    @contextmanager
    def request_scope(self, request_id: object):
        normalized_id = str(request_id or "").strip()
        with _HOST_SPAN_PROFILES_LOCK:
            profile = _HOST_SPAN_PROFILES.get(normalized_id)
        profile_token = _ACTIVE_HOST_SPAN_PROFILE.set(profile)
        scope_token = _ACTIVE_HOST_SPAN_REQUEST_SCOPE.set(True)
        try:
            yield
        finally:
            _ACTIVE_HOST_SPAN_REQUEST_SCOPE.reset(scope_token)
            _ACTIVE_HOST_SPAN_PROFILE.reset(profile_token)

    def claim_request_profile(self, profile: _HostSpanProfile | None, request_id: object) -> bool:
        """Bind an unclaimed preparation profile to the capture's own request id (first claim wins).

        Identity flows capture -> profile, never profile -> capture: a replacement capture built in the
        same scope must keep a distinct id (adoption refuses a reused one)."""
        normalized_id = str(request_id or "").strip()
        if profile is None or profile.recorder is not self or not normalized_id:
            return False
        with _HOST_SPAN_PROFILES_LOCK:
            if profile.claimed:
                return profile.request_id == normalized_id
            previous = profile.request_id
            if _HOST_SPAN_PROFILES.get(previous) is profile:
                _HOST_SPAN_PROFILES.pop(previous, None)
            profile.request_id = normalized_id
            profile.claimed = True
            _HOST_SPAN_PROFILES[normalized_id] = profile
            with self._lock:
                for span_id, trace_request_id in tuple(self._trace_request_ids.items()):
                    if trace_request_id == previous:
                        self._trace_request_ids[span_id] = normalized_id
                for row in self._trace_spans:
                    if row.get("request_id") == previous:
                        row["request_id"] = normalized_id
            return True

    def finish_request_profile(self, request_id: object) -> _HostSpanProfile | None:
        normalized_id = str(request_id or "").strip()
        with _HOST_SPAN_PROFILES_LOCK:
            return _HOST_SPAN_PROFILES.pop(normalized_id, None)

    def discard_request_profile(self, request_id: object) -> None:
        normalized_id = str(request_id or "").strip()
        with _HOST_SPAN_PROFILES_LOCK:
            _HOST_SPAN_PROFILES.pop(normalized_id, None)

    def start_span(self, name: str, *, boundary: str = "recorded_prefix",
                   sequence: int | None = None, parent_span: str | None = None,
                   force: bool = False) -> _HostSpanHandle | None:
        if not host_span_profiles_enabled():
            return None
        active_stack = _ACTIVE_HOST_SPAN_STACK.get()
        active_profile = _ACTIVE_HOST_SPAN_PROFILE.get()
        active_id = next(
            (span_id for recorder, span_id in reversed(active_stack) if recorder is self),
            None,
        )
        if not force and active_id is None and not (
                active_profile is not None and active_profile.recorder is self):
            return None
        with self._lock:
            self._span_sequence += 1
            span_id = str(self._span_sequence)
            self._trace_parent_ids[span_id] = parent_span or active_id
            self._trace_names[span_id] = name
            self._trace_request_ids[span_id] = (
                active_profile.request_id
                if active_profile is not None and active_profile.recorder is self
                else None
            )
        _ACTIVE_HOST_SPAN_STACK.set((*active_stack, (self, span_id)))
        return _HostSpanHandle(
            self, span_id, name, time.perf_counter_ns(), boundary, sequence,
            parent_span or active_id,
        )

    def finish_span(self, handle: _HostSpanHandle | None, *,
                    ended_ns: int | None = None, action_id: str | None = None,
                    attributes: Mapping[str, str] | None = None,
                    cpu_ns: int | None = None, cache_hits: int = 0,
                    cache_misses: int = 0, bytes_scanned: int = 0,
                    records_folded: int = 0, files_read: int = 0,
                    fsyncs: int = 0) -> None:
        if handle is None or handle.recorder is not self:
            return
        ended = max(handle.started_ns, int(ended_ns or time.perf_counter_ns()))
        try:
            self.record(
                handle.name, handle.started_ns, ended, boundary=handle.boundary,
                sequence=handle.sequence, action_id=action_id, attributes=attributes,
                parent_span=handle.parent_span, cpu_ns=cpu_ns,
                cache_hits=cache_hits, cache_misses=cache_misses,
                bytes_scanned=bytes_scanned, records_folded=records_folded,
                files_read=files_read, fsyncs=fsyncs,
            )
            row = {
                "span_id": handle.span_id,
                "parent_span": handle.parent_span,
                "name": handle.name,
                "request_id": self._trace_request_ids.get(handle.span_id),
                "started_ns": handle.started_ns,
                "ended_ns": ended,
                "elapsed_ns": ended - handle.started_ns,
                "sequence": handle.sequence,
            }
            with self._lock:
                if len(self._trace_spans) >= self.limit:
                    self._trace_spans.pop(0)
                self._trace_spans.append(row)
                self._trace_request_ids.pop(handle.span_id, None)
                # Ancestor lookups walk only ACTIVE spans (whose ancestors are still open),
                # so a finished span's name/parent entries can go; keeping them grew without
                # bound (~1 entry per journal emit, 8k+ per benchmark run).
                self._trace_names.pop(handle.span_id, None)
                self._trace_parent_ids.pop(handle.span_id, None)
        except Exception:
            # Profiling is advisory; it cannot change the host operation.
            pass
        finally:
            active_stack = _ACTIVE_HOST_SPAN_STACK.get()
            _ACTIVE_HOST_SPAN_STACK.set(tuple(
                item for item in active_stack
                if not (item[0] is self and item[1] == handle.span_id)
            ))

    def find_active_ancestor(self, name: str) -> str | None:
        active_stack = _ACTIVE_HOST_SPAN_STACK.get()
        with self._lock:
            for recorder, span_id in reversed(active_stack):
                if recorder is not self:
                    continue
                current = span_id
                while current is not None:
                    if self._trace_names.get(current) == name:
                        return current
                    current = self._trace_parent_ids.get(current)
        return None

    @contextmanager
    def measure(self, name: str, *, boundary: str = "recorded_prefix",
                sequence: int | None = None, action_id: str | None = None,
                attributes: Mapping[str, str] | None = None,
                parent_span: str | None = None, cpu_ns: int | None = None,
                cache_hits: int = 0, cache_misses: int = 0,
                bytes_scanned: int = 0, records_folded: int = 0,
                files_read: int = 0, fsyncs: int = 0):
        handle = self.start_span(
            name, boundary=boundary, sequence=sequence, parent_span=parent_span)
        started = handle.started_ns if handle is not None else time.perf_counter_ns()
        try:
            yield
        finally:
            ended = time.perf_counter_ns()
            if handle is not None:
                self.finish_span(
                    handle, ended_ns=ended, action_id=action_id, attributes=attributes,
                    cpu_ns=cpu_ns, cache_hits=cache_hits, cache_misses=cache_misses,
                    bytes_scanned=bytes_scanned, records_folded=records_folded,
                    files_read=files_read, fsyncs=fsyncs,
                )
            else:
                self.record(
                    name, started, ended, boundary=boundary, sequence=sequence,
                    action_id=action_id, attributes=attributes, parent_span=parent_span,
                    cpu_ns=cpu_ns, cache_hits=cache_hits, cache_misses=cache_misses,
                    bytes_scanned=bytes_scanned, records_folded=records_folded,
                    files_read=files_read, fsyncs=fsyncs,
                )

    def snapshot(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(span.to_dict() for span in self._spans)

    def trace_snapshot(
            self, names: Sequence[str] | None = None, *,
            request_id: str | None = None) -> tuple[dict[str, object], ...]:
        allowed = frozenset(names) if names is not None else None
        with self._lock:
            rows = tuple(dict(row) for row in self._trace_spans)
        return tuple(
            row for row in rows
            if (allowed is None or row.get("name") in allowed)
            and (request_id is None or row.get("request_id") == request_id)
        )

    def record(self, name: str, started_ns: int, ended_ns: int, *,
               boundary: str = "recorded_prefix", sequence: int | None = None,
               action_id: str | None = None,
               attributes: Mapping[str, str] | None = None,
               parent_span: str | None = None, cpu_ns: int | None = None,
               cache_hits: int = 0, cache_misses: int = 0,
               bytes_scanned: int = 0, records_folded: int = 0,
               files_read: int = 0, fsyncs: int = 0) -> None:
        span = HostSpan(
            name, int(started_ns), int(ended_ns), boundary, sequence, action_id,
            tuple(sorted((attributes or {}).items())), parent_span, cpu_ns,
            cache_hits, cache_misses, bytes_scanned, records_folded, files_read, fsyncs,
        )
        with self._lock:
            if len(self._spans) >= self.limit:
                self._spans.pop(0)
            self._spans.append(span)
        profile = _ACTIVE_HOST_SPAN_PROFILE.get()
        if profile is not None and profile.recorder is self:
            try:
                profile.add(name, max(0, int(ended_ns) - int(started_ns)))
            except Exception:
                pass


def _runtime_stateframe_mode_snapshot() -> dict[str, object]:
    """Capture only registered mode values and safe scalar STATEFRAME flags."""
    from kernel_lean import enabled as lean_enabled, runtime_lean_snapshot

    lean_on = lean_enabled()
    try:
        from kernel_protocol_spec import _SWITCH_REFS
    except ImportError:
        _SWITCH_REFS = ()
    defaults = {
        str(row.name): str(row.default)
        for row in _SWITCH_REFS
        if str(getattr(row, "name", "")).startswith("STATEFRAME_")
    }
    # Gate switches (r2Z) default to on; record them even when unset.
    defaults.setdefault("STATEFRAME_IDLE_GOVERNOR", "on")
    defaults.setdefault("STATEFRAME_REPEAT_GATE", "on")
    # Requested-byte delivery remains enabled under LEAN: LEAN controls what
    # context is collected, not whether an explicit authorized read is truthful.
    defaults.setdefault("STATEFRAME_DELIVER_REQUESTED_BYTES", "on")
    defaults.setdefault("STATEFRAME_M3_WORKING_CONTEXT", "on")
    defaults.setdefault(TIMEOUT_RETRY_FROZEN_ENV, "on")
    names = set(defaults)
    names.update(name for name in os.environ if name.startswith("STATEFRAME_"))
    sensitive = ("KEY", "TOKEN", "SECRET", "PASSWORD", "URL")
    non_mode_text = ("PROMPT", "TEXT", "CONTENT")
    modes: dict[str, str] = {}
    for name in sorted(names):
        if name == _HOST_SPANS_ENV:
            continue
        upper_name = name.upper()
        if (any(part in upper_name for part in sensitive)
                or (name not in defaults and any(part in upper_name for part in non_mode_text))):
            continue
        value = os.environ.get(name, defaults.get(name, "unset"))
        text = str(value).strip()
        safe = bool(text) and len(text) <= 48 and all(
            character.isalnum() or character in "_-.:"
            for character in text
        )
        modes[name] = text if safe else ("configured" if value else "unset")
    if lean_on:
        # These are effective modes. The configured values remain explicit in
        # the LEAN detail so an operator can distinguish forced-off from unset.
        modes["STATEFRAME_IDLE_GOVERNOR"] = "off"
        modes["STATEFRAME_REPEAT_GATE"] = "off"
    return {
        "schema": "stateframe-runtime-modes/1",
        "modes": modes,
        "lean": runtime_lean_snapshot(),
    }


def _host_span_context_active(recorder: HostSpanRecorder) -> bool:
    profile = _ACTIVE_HOST_SPAN_PROFILE.get()
    if profile is not None and profile.recorder is recorder:
        return True
    if _ACTIVE_HOST_SPAN_REQUEST_SCOPE.get():
        return False
    return any(
        active_recorder is recorder
        for active_recorder, _span_id in _ACTIVE_HOST_SPAN_STACK.get()
    )


@contextmanager
def profile_host_span(owner: object, name: str, *, request_id: object = None):
    recorder = getattr(owner, "host_spans", None)
    if not isinstance(recorder, HostSpanRecorder):
        recorder = getattr(getattr(owner, "journal", None), "host_spans", None)
    if not isinstance(recorder, HostSpanRecorder):
        yield
        return
    if request_id is not None:
        with recorder.request_scope(request_id):
            if not _host_span_context_active(recorder):
                yield
                return
            with recorder.measure(name, boundary="request_preparation"):
                yield
        return
    if not _host_span_context_active(recorder):
        yield
        return
    with recorder.measure(name, boundary="request_preparation"):
        yield


@contextmanager
def profile_active_host_span(name: str):
    """Measure a named request span through the existing journal recorder."""
    profile = _ACTIVE_HOST_SPAN_PROFILE.get()
    recorder = getattr(profile, "recorder", None)
    if not isinstance(recorder, HostSpanRecorder):
        if _ACTIVE_HOST_SPAN_REQUEST_SCOPE.get():
            yield
            return
        recorder = next(
            (item for item, _span_id in reversed(_ACTIVE_HOST_SPAN_STACK.get())
             if isinstance(item, HostSpanRecorder)),
            None,
        )
    if not isinstance(recorder, HostSpanRecorder):
        yield
        return
    with recorder.measure(name, boundary="request_preparation"):
        yield


def profiled_function_span(name: str):
    """Decorate a function with an optional active-request host span."""
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with turn_timing_step(name):
                with profile_active_host_span(name):
                    return function(*args, **kwargs)
        return wrapped
    return decorate


def _apply_request_capture_budget(
        capture: object, body: dict[str, object]) -> dict[str, object]:
    """Keep every final budget pass attached to the capture that emits its wire."""
    from kernel_request_budget import apply_capture_budget

    recorder = getattr(getattr(capture, "journal", None), "host_spans", None)
    if not isinstance(recorder, HostSpanRecorder):
        return apply_capture_budget(capture, body)
    with recorder.request_scope(getattr(capture, "request_id", None)):
        return apply_capture_budget(capture, body)


def profiled_journal_span(name: str):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(journal, *args, **kwargs):
            kind = args[0] if args else kwargs.get("kind")
            boundary = getattr(journal, "_host_span_event_boundary", None)
            if callable(boundary) and isinstance(kind, str):
                boundary("before", kind)
            with turn_timing_step(name):
                with profile_host_span(journal, name, request_id=kwargs.get("request_id")):
                    result = function(journal, *args, **kwargs)
            if callable(boundary) and isinstance(kind, str):
                boundary("after", kind)
            timing = _turn_timing_recorder(journal)
            if kind == "response_received" and timing is not None:
                timing.note_response(result)
            elif kind == "request_submitted" and timing is not None:
                timing_event = timing.finish_submission(result)
                if timing_event is not None:
                    journal.emit("turn_timing", **timing_event)
            return result
        return wrapped
    return decorate


def cleanup_request_host_span_profile(function):
    """Release unfinished capture profiles on every send exit path."""
    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        request_id = getattr(self, "request_id", None)
        recorder = getattr(getattr(self, "journal", None), "host_spans", None)
        try:
            if isinstance(recorder, HostSpanRecorder):
                recorder.begin_request_profile(request_id, getattr(self, "step", None))
                with recorder.request_scope(request_id):
                    return function(self, *args, **kwargs)
            return function(self, *args, **kwargs)
        finally:
            if isinstance(recorder, HostSpanRecorder):
                for candidate in (request_id, getattr(self, "request_id", None)):
                    try:
                        recorder.discard_request_profile(candidate)
                    except Exception:
                        pass
    return wrapped


def profile_request_capture(function):
    """Attach profiling to direct and recovery provider attempts using capture identity."""
    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        recorder = getattr(getattr(self, "journal", None), "host_spans", None)
        if not isinstance(recorder, HostSpanRecorder):
            return function(self, *args, **kwargs)
        request_id = getattr(self, "request_id", None)
        try:
            recorder.begin_request_profile(request_id, getattr(self, "step", None))
        except Exception:
            # Profiling remains advisory at the provider boundary.
            return function(self, *args, **kwargs)
        with recorder.request_scope(request_id):
            return function(self, *args, **kwargs)
    return wrapped


def _host_span_profile_spans(
        profile: _HostSpanProfile) -> dict[str, dict[str, int | float]]:
    totals: dict[str, list[int]] = {
        name: [0, 0, 0] for name in _HOST_PROFILE_SPAN_NAMES
    }
    for name, elapsed_ns in profile.snapshot():
        values = totals.get(name)
        if values is None:
            continue
        duration = max(0, int(elapsed_ns))
        values[0] += 1
        values[1] += duration
        values[2] = max(values[2], duration)
    return {
        name: {
            "count": values[0],
            "total_ms": round(values[1] / 1_000_000, 6),
            "max_ms": round(values[2] / 1_000_000, 6),
        }
        for name, values in totals.items()
    }


def _normalized_wire_digest(value: object) -> str:
    text = str(value or "").strip()
    return text.removeprefix("sha256:")


def _verified_source_ranges(path_row: object) -> tuple[str, list[list[int]]] | None:
    if not isinstance(path_row, Mapping) or path_row.get("proof_status") != "VERIFIED":
        return None
    revision = str(path_row.get("revision") or "").strip()
    raw_ranges = path_row.get("ranges", ())
    if not isinstance(raw_ranges, Sequence) or isinstance(raw_ranges, (str, bytes, bytearray)):
        return None
    ranges = [
        [item[0], item[1]]
        for item in raw_ranges
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray))
        and len(item) == 2 and type(item[0]) is int and type(item[1]) is int
        and 1 <= item[0] <= item[1]
    ]
    return (revision, ranges) if revision and ranges else None


def _verified_source_bytes(
        raw_proofs: object, path: str, wire_sha256: str) -> int | None:
    if not isinstance(raw_proofs, Sequence) or isinstance(raw_proofs, (str, bytes, bytearray)):
        return None
    delivered = 0
    verified = False
    for proof in raw_proofs:
        if not isinstance(proof, Mapping):
            continue
        proof_key = proof.get("key")
        amount = proof.get("delivered_bytes")
        valid_proof = (
            _normalized_wire_digest(proof.get("submitted_http_bytes_digest")) == wire_sha256
            and proof.get("transport_identity") == "VERIFIED"
            and proof.get("wire_contains_exact") is True
            and isinstance(proof_key, Mapping)
            and proof_key.get("canonical_path") == path
            and type(amount) is int and amount >= 0
        )
        if valid_proof:
            delivered += amount
            verified = True
    return delivered if verified else None


def _deferred_source_paths(receipt: Mapping[str, object]) -> list[str]:
    raw_paths = receipt.get("deferred_paths", ())
    if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes, bytearray)):
        return []
    return sorted({path.strip() for path in raw_paths if isinstance(path, str) and path.strip()})


def _verified_source_blocks(
        receipt: object, wire_sha256: str) -> tuple[list[dict[str, object]], list[str]]:
    if not isinstance(receipt, Mapping) or receipt.get("proof_status") != "VERIFIED":
        return [], []
    if _normalized_wire_digest(receipt.get("submitted_http_bytes_digest")) != wire_sha256:
        return [], []
    path_rows = receipt.get("paths")
    proofs_by_path = receipt.get("delivery_proofs")
    if not isinstance(path_rows, Mapping) or not isinstance(proofs_by_path, Mapping):
        return [], []
    blocks = []
    for path, path_row in sorted(path_rows.items(), key=lambda item: str(item[0])):
        if not isinstance(path, str) or not path.strip():
            continue
        source_ranges = _verified_source_ranges(path_row)
        if source_ranges is None:
            continue
        revision, ranges = source_ranges
        delivered = _verified_source_bytes(proofs_by_path.get(path), path, wire_sha256)
        if delivered is not None:
            blocks.append({
                "path": path,
                "revision": revision,
                "ranges": ranges,
                "bytes": delivered,
            })
    return blocks, _deferred_source_paths(receipt)


def build_host_span_final_wire_manifest(
        capture: object, body: Mapping[str, object], wire: bytes) -> dict[str, object]:
    body_sha256 = hashlib.sha256(wire).hexdigest()
    raw_messages = body.get("messages", ())
    messages = (raw_messages if isinstance(raw_messages, Sequence)
                and not isinstance(raw_messages, (str, bytes, bytearray)) else ())
    role_counts: dict[str, int] = {}
    panel_counts: dict[str, int] = {}
    try:
        from kernel_request_diet import section_name
    except ImportError:
        section_name = None
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        if callable(section_name):
            try:
                section = str(section_name(message, index) or "unknown")
            except Exception:
                section = "unknown"
            panel_counts[section] = panel_counts.get(section, 0) + 1
    receipt = getattr(capture, "request_source_final_delivery", None)
    source_blocks, deferred_paths = _verified_source_blocks(receipt, body_sha256)
    return {
        "body_sha256": body_sha256,
        "body_bytes": len(wire),
        "message_count_by_role": dict(sorted(role_counts.items())),
        "advertised_tool_names": _submitted_tool_names(body),
        "source_blocks": source_blocks,
        "deferred_paths": deferred_paths,
        "panel_counts_by_section": dict(sorted(panel_counts.items())),
    }


def _journal_greenfact_error(journal: object | None, step: object, error: Exception) -> None:
    try:
        from kernel_boundary_recovery import safe_error_class
        emit = getattr(journal, "emit", None)
        if callable(emit):
            emit(
                "greenfact_error",
                producer="kernel_greenfact",
                error_class=safe_error_class(error),
                step=step,
                dispatch=False,
            )
    except Exception:
        # A journal failure must not turn an optional GREENFACT failure into a failed turn.
        return


def _greenfact_context_fits(
        sender: object, body: dict[str, object], endpoint: str, key: str) -> bool:
    from kernel_context_capacity import ContextCapacityError, enforce_context_capacity
    try:
        enforce_context_capacity(sender, body, endpoint=endpoint, key=key)
    except ContextCapacityError:
        return False
    return True


def _apply_greenfact_at_send(
        sender: object, messages: object, *,
        validate: Callable[[], bool] | None = None) -> bool:
    """Apply GREENFACT at the shared send boundary without changing turn authority."""
    from kernel_request_kind import is_auxiliary
    if is_auxiliary(sender):
        return False
    journal = getattr(sender, "journal", None)
    step = getattr(sender, "step", None)
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
        if runtime is None:
            runtime = getattr(sender, "completion_owner_runtime", None)
        from kernel_greenfact import maybe_append_greenfact
        return maybe_append_greenfact(
            messages, runtime, journal=journal, turn=step, validate=validate,
            request_id=getattr(sender, "request_id", None))
    except Exception as error:
        _journal_greenfact_error(journal, step, error)
        return False


class AuditJournal:
    """One exclusive run directory; fsync each complete append before proceeding."""
    def __init__(self, parent: Path, task_id: str, manifest: Mapping[str, object]):
        self.run_id = uuid.uuid4().hex
        self.task_id = task_id
        self.path = parent / self.run_id / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=False)
        self._sequence = 0
        self._lock = threading.RLock()
        self._blob_cache: dict[str, int] = {}
        self._closed_action_records: dict[str, dict[str, object]] = {}
        self._frame_journal_index = None
        self._framefit_receipt_records: list[dict[str, object]] = []
        self._framefit_receipt_overflow = False
        self.manifest = dict(manifest)
        self.host_cost_meter = PhaseMeter()
        self.host_spans = HostSpanRecorder()
        self.config_hash = digest(self.manifest)
        if os.environ.get("STATEFRAME_RUN_VITALS", "off") == "on":
            try:
                from kernel_run_vitals import RunVitalsAccumulator
                self._run_vitals_accumulator = RunVitalsAccumulator()
                self._run_vitals_emitted = False
                self._run_vitals_observer_failed = False
            except Exception:
                # Optional telemetry must not prevent journal creation or a send.
                self._run_vitals_observer_failed = True
        self.emit("run_started", manifest=self.manifest, config_hash=self.config_hash)
        self.emit(
            "runtime_mode_snapshot", producer="kernel_measurement",
            modes=_runtime_stateframe_mode_snapshot(),
        )

    @profiled_journal_span("journal_emit")
    def emit(self, kind: str, **fields: object) -> dict[str, object]:
        identity_fire = False
        try:
            from kernel_verdict_identity import prepare_journal_event
            fields, identity_fire = prepare_journal_event(self, kind, fields)
        except ImportError:
            # The measurement journal is also used as a small standalone
            # utility outside the kernel release.  Its historical path has no
            # dependency on the optional identity organ.
            pass
        with self._lock:
            self._sequence += 1
            record = {**fields, "schema": "stateframe-audit/1", "kind": kind,
                      "run_id": self.run_id, "task_id": self.task_id,
                      "event_id": f"{self.run_id}:{self._sequence}",
                      "sequence": self._sequence, "timestamp_ns": time.time_ns()}
            accumulator = getattr(self, "_run_vitals_accumulator", None)
            if accumulator is not None:
                try:
                    accumulator.observe(record)
                except Exception:
                    # The observation is advisory; keep the original journal/send path.
                    self._run_vitals_observer_failed = True
            with profile_host_span(self, "journal_blob_io"):
                externalized = externalize_journal_fields(
                    fields, self.path, self._blob_cache)
            stored_record = {
                **externalized,
                "schema": record["schema"], "kind": record["kind"],
                "run_id": record["run_id"], "task_id": record["task_id"],
                "event_id": record["event_id"], "sequence": record["sequence"],
                "timestamp_ns": record["timestamp_ns"],
            }
            line = canonical_bytes(stored_record) + b"\n"
            with self.host_cost_meter.measure("journaling", count=1):
                with self.host_spans.measure("append_fsync"):
                    with self.path.open("ab") as handle:
                        handle.write(line)
                        handle.flush()
                        os.fsync(handle.fileno())
            if kind in {"action_receipt", "check_receipt", "completion_authority_state"}:
                self._observe_frame_journal_event(record)
            self._index_framefit_receipt(record)
            action_record = fields.get("record") if kind == "action_outcome_closed" else None
            if isinstance(action_record, Mapping):
                tool_call_id = action_record.get("tool_call_id")
                if isinstance(tool_call_id, str) and tool_call_id:
                    self._closed_action_records[tool_call_id] = {
                        "kind": kind, "record": copy.deepcopy(dict(action_record)),
                    }
            generated = getattr(self, "generated_artifacts", None)
            if generated is not None:
                generated.observe(record)
            if identity_fire:
                from kernel_verdict_identity import journal_identity_fired
                journal_identity_fired(self, kind)
            return record

    def _observe_frame_journal_event(self, record: Mapping[str, object]) -> None:
        try:
            from kernel_frame_journal_gate import observe_journal_event
            observe_journal_event(self, record)
        except Exception:
            pass

    def _index_framefit_receipt(self, event: Mapping[str, object]) -> None:
        projection = framefit_receipt_projection(event)
        if projection is None:
            return
        if len(self._framefit_receipt_records) >= MAX_FRAMEFIT_RECEIPT_EVENTS:
            self._framefit_receipt_overflow = True
            return
        self._framefit_receipt_records.append(projection)

    def framefit_receipt_records(self) -> tuple[dict[str, object], ...]:
        """Return a bounded, sanitized projection of durably appended receipts."""
        with self._lock:
            if self._framefit_receipt_overflow:
                raise RuntimeError("FRAMEFIT5 journal receipt projection exceeded its bound")
            return tuple(copy.deepcopy(self._framefit_receipt_records))

    def settled_action_records(self) -> tuple[dict[str, object], ...]:
        """Return exact closed action observations keyed by their provider call IDs."""
        with self._lock:
            return tuple(copy.deepcopy(tuple(self._closed_action_records.values())))

    def frame_journal_snapshot(self) -> object:
        """Return the receipt projection used by the optional send-boundary gate."""
        with self._lock:
            try:
                from kernel_frame_journal_gate import snapshot_for_journal
                return snapshot_for_journal(self)
            except Exception:
                return None

    def host_span_snapshot(self) -> tuple[dict[str, object], ...]:
        return self.host_spans.snapshot()


def resolved_modes(consumers: Mapping[str, object]) -> dict[str, str]:
    """Reject misspellings, invalid modes and activated factors with dead consumers."""
    modes = {"self_oracle": "BOTH", "navigation": "BOTH", "history_cap": "OFF",
             "retry_predispatch": "OFF", "render_selection": "OFF",
             "sidekick": "OFF", "pilot": "OFF", "tool_applicability": "OFF", "public_review": "OFF"}
    for name in os.environ.get("KERNEL_SHADOW", "").split(","):
        name = name.strip()
        if name == "all":
            modes.update(self_oracle="SHADOW", navigation="SHADOW")
        elif name in {"self_oracle", "navigation"}:
            modes[name] = "SHADOW"
    for pair in os.environ.get("KERNEL_MODE", "").split(","):
        if not pair.strip():
            continue
        name, sep, value = pair.partition(":")
        name, value = name.strip(), value.strip().upper()
        if not sep or value not in Mode._value2member_map_:
            raise ValueError(f"invalid experiment factor configuration: {name}")
        if name not in modes:
            if value != "OFF":
                raise ValueError(f"enabled factor has no live consumer: {name}")
        modes[name] = value
    from kernel_intervention_policy import interventions_enabled
    if not interventions_enabled():
        modes.update(self_oracle="OFF", navigation="OFF", retry_predispatch="OFF",
                     sidekick="OFF", pilot="OFF", tool_applicability="OFF", public_review="OFF")
    for name, mode in modes.items():
        if mode != "OFF" and not callable(consumers.get(name)):
            raise ValueError(f"enabled factor has no live consumer: {name}")
    return modes


def run_manifest(*, model: str, tools: object, consumers: Mapping[str, object],
                 history_cap: int, frame_target: int, frame_ceiling: int,
                 temperature: float, modules: tuple[Path, ...]) -> dict[str, object]:
    from sampler_config import effective_identity
    from kernel_context import SELECTION_POLICY_VERSION
    from kernel_intervention_policy import interventions_enabled
    from kernel_step_contract import working_view_mode
    from kernel_test_python import test_python
    selected_reasoning = reasoning_profile()
    try:
        from kernel_idealpath import enabled as idealpath_enabled
        idealpath_mode = "on" if idealpath_enabled() else "off"
    except ImportError:
        idealpath_mode = UNKNOWN
    except ValueError:
        # A malformed host declaration is environment evidence, never an
        # inferred product mode or a complete launch certificate.
        idealpath_mode = ENVIRONMENT
    modes = resolved_modes(consumers)
    source_digests: dict[str, str] = {}
    for raw_path in modules:
        path = Path(raw_path)
        try:
            source_digests[path.name] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError):
            source_digests[path.name] = MISSING
    contract_path = Path(__file__).with_name("kernel_launch_identity_cost.py")
    try:
        source_digests[contract_path.name] = (
            "sha256:" + hashlib.sha256(contract_path.read_bytes()).hexdigest()
        )
    except OSError:
        source_digests[contract_path.name] = MISSING
    return {
        "modes": modes,
        "interventions_enabled": interventions_enabled(),
        "kernel_shadow": os.environ.get("KERNEL_SHADOW", ""),
        "procedure_guidance": os.environ.get("STATEFRAME_PROCEDURE_GUIDANCE", "0"),
        "working_view": working_view_mode(),
        "liveness": {name: callable(consumers.get(name)) for name in modes},
        "history_cap_tokens": history_cap, "frame_target_tokens": frame_target,
        "budget_estimator": "serialized characters / 4, ceiling rounded up; not measured prefill",
        "frame_ceiling_tokens": frame_ceiling,
        "trim_policy": "D2 enforcement replaces REACT_TRIM; SHADOW/OFF keep legacy trim",
        "react_trim_messages": int(os.environ.get("REACT_TRIM", "90")),
        "selector_version": SELECTION_POLICY_VERSION, "renderer_version": "stateframe-render/1",
        "model": model, "quantization": os.environ.get("REACT_QUANTIZATION", MISSING),
        "tokenizer": os.environ.get("REACT_TOKENIZER", MISSING),
        "chat_template": os.environ.get("REACT_CHAT_TEMPLATE", MISSING),
        "server_build": os.environ.get("REACT_SERVER_BUILD", MISSING),
        "cache_policy": os.environ.get("REACT_CACHE_POLICY", MISSING),
        "environment_revision": os.environ.get("REACT_ENVIRONMENT_REVISION", MISSING),
        "tool_schema_hash": digest(tools), "temperature": temperature,
        "sampler": effective_identity(temperature),
        "tool_output_token_limits": dict(zip(('initial', 'maximum'), output_token_limits())),
        "seed": int(os.environ["REACT_SEED"]) if "REACT_SEED" in os.environ else MISSING,
        "python": platform.python_version(), "platform": platform.platform(),
        "project_test_python": test_python(),
        "source_digests": source_digests,
        "deployed_source_digest": digest(source_digests),
        "reasoning_profile": selected_reasoning.to_dict(),
        "reasoning_mode": selected_reasoning.mode,
        # The runtime writes the effective owner state after construction.
        # UNKNOWN here is intentional and is never inferred from an env label.
        "idealpath_mode": idealpath_mode,
    }


def _measured(value: object, *, integer: bool = False) -> int | float | str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return MISSING
    if not math.isfinite(value) or value < 0 or (integer and value != int(value)):
        return MISSING
    return int(value) if integer else float(value)


class HarnessTransportError(ConnectionError):
    """A terminal provider transport failure, never a model/product failure."""

    outcome = HARNESS_TRANSPORT
    failure_origin = "harness"
    transport_retryable = False
    harness_contract = False

    def __init__(self, *, attempts: int, error_type: str,
                 http_status: int | None, relay_error: str | None,
                 transport_class: str | None = None,
                 server_error: str | None = None,
                 attempt_reasons: Sequence[Mapping[str, object]] = ()) -> None:
        status = f" HTTP {http_status}" if http_status is not None else ""
        super().__init__(f"transport failed after {attempts} attempt(s): {error_type}{status}")
        self.attempts = attempts
        self.error_type = error_type
        self.http_status = http_status
        self.relay_error = relay_error
        self.transport_class = transport_class
        self.server_error = server_error
        self.attempt_reasons = tuple(dict(reason) for reason in attempt_reasons)
        if transport_class == "contract":
            # The provider rejected the REQUEST (schema/grammar/4xx): a harness
            # contract defect, terminal, never retried, never charged to the model.
            self.outcome = HARNESS_CONTRACT
            self.harness_contract = True


class ProviderRequestTimeoutError(HarnessTransportError):
    """A provider call exhausted its host timeout and its one retry."""


class TransportTimeoutError(HarnessTransportError):
    """A relay-labeled timeout that exhausted its bounded retry policy."""


def _transport_error_class(relay_error: str | None) -> type[HarnessTransportError]:
    return (TransportTimeoutError
            if relay_error in {"upstream_timeout", "relay_ttft_timeout"}
            else HarnessTransportError)


def _http_error_body(exc: BaseException) -> bytes | None:
    """Read an HTTPError body ONCE and cache it (the stream is single-use)."""
    if not isinstance(exc, urllib.error.HTTPError):
        return None
    cached = getattr(exc, "_kernel_error_body", None)
    if isinstance(cached, bytes):
        return cached
    try:
        body = exc.read(8 * 1024 + 1) or b""
    except (OSError, ValueError, AttributeError):
        body = b""
    finally:
        try:
            exc.close()
        except (OSError, ValueError, AttributeError):
            pass
    try:
        exc._kernel_error_body = body
    except AttributeError:
        pass
    return body


def _relay_label_from_error(exc: BaseException) -> str | None:
    label = getattr(exc, "relay_error", None)
    if isinstance(label, str) and label in RELAY_ERROR_LABELS:
        return label
    error_type = getattr(exc, "error_type", type(exc).__name__)
    if (isinstance(exc, http.client.IncompleteRead)
            or (isinstance(error_type, str)
                and error_type in {"IncompleteRead", "RemoteDisconnected"})):
        return "downstream_disconnect"
    if isinstance(exc, urllib.error.HTTPError):
        body = _http_error_body(exc) or b""
        http_status = getattr(exc, "code", None)
    else:
        server_error = getattr(exc, "server_error", None)
        if isinstance(server_error, str):
            body = server_error.encode("utf-8", "replace")
        elif isinstance(server_error, bytes):
            body = server_error
        else:
            return None
        http_status = getattr(exc, "http_status", None)
    if not body:
        return None
    try:
        from kernel_bounded_http import MAX_ERROR_BYTES, _relay_error_label
        if len(body) > MAX_ERROR_BYTES:
            return None
        label = _relay_error_label(body)
    except (ImportError, OSError, ValueError):
        return None
    if label is not None:
        return label
    if http_status != 502:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        message = body.decode("utf-8", "replace")
    else:
        error = payload.get("error") if isinstance(payload, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str):
        return None
    prefix = "relay upstream error:"
    normalized = message.strip().casefold()
    if not normalized.startswith(prefix):
        return None
    timeout_name = normalized[len(prefix):].strip()
    if timeout_name in {"timeout", "timeouterror", "socket.timeout", "timeoutexpired"}:
        return "upstream_timeout"
    return None


def _server_error_from(exc: BaseException, key: str = "") -> str | None:
    """Redacted <=2 KB provider error text (audit #4: the body was thrown away)."""
    from kernel_bounded_http import safe_server_error
    carried = getattr(exc, "server_error", None)
    if isinstance(carried, str):
        return safe_server_error(carried.encode("utf-8"), key)
    return safe_server_error(_http_error_body(exc), key)


def _parsed_server_error(text: str | None) -> dict[str, object] | None:
    """``{"message", "type"}`` from a JSON provider error body, when present."""
    if not text:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return {"message": text[:2048], "type": None}
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return {"message": str(error.get("message", ""))[:2048],
                "type": error.get("type") if isinstance(error.get("type"), str) else None}
    return {"message": text[:2048], "type": None}


def _transport_metadata(exc: BaseException) -> tuple[str, int | None, str | None]:
    error_type = getattr(exc, "error_type", type(exc).__name__)
    if not isinstance(error_type, str) or not error_type.isidentifier():
        error_type = type(exc).__name__
    status = getattr(exc, "http_status", getattr(exc, "code", None))
    if type(status) is not int or not 100 <= status <= 599:
        status = None
    return error_type, status, _relay_label_from_error(exc)


LLAMA_FORMAT_FAILURE_REASON = "llama_cpp_model_format_failure"
_LLAMA_FORMAT_FAILURE_MARKERS = (
    "the model produced output that does not match the expected peg-native format",
    "failed to parse a tool call",
    "failed to parse tool call",
    "failed to parse input at pos",
    "tool call parse failed",
    "failed to parse grammar",
    "grammar parse failed",
    "failed to parse chat template",
    "failed to apply chat template",
    "failed to apply template",
    "failed to parse tool use chat template",
    "failed to parse template",
    "template parse failed",
)


def _llama_format_failure_reason(http_status: int | None,
                                server_error_body: str | None) -> str | None:
    """Match llama.cpp parse/format failures without treating generic 500s as model errors."""
    if http_status != 500 or not isinstance(server_error_body, str):
        return None
    normalized = " ".join(server_error_body.casefold().split())
    return (LLAMA_FORMAT_FAILURE_REASON
            if any(marker in normalized for marker in _LLAMA_FORMAT_FAILURE_MARKERS)
            else None)


def _transport_retryable(exc: BaseException, *, elapsed_ns: int,
                         http_status: int | None, relay_error: str | None,
                         error_type: str, transport_class: str | None = None,
                         matched_reason: str | None = None) -> bool:
    """Retry bounded transient transport failures, never provider format outcomes."""
    # 10-08: a port had added `if transport_class == "capacity": return False`, which stopped retrying a fast 503
    # (capacity) blip - a regression vs min2 caught by test_503_is_capacity_and_retried. Capacity blips retry as before.
    if matched_reason == LLAMA_FORMAT_FAILURE_REASON:
        return False
    if error_type == "ProviderLivenessTimeout":
        return True
    if transport_class == "contract":
        return False
    if relay_error == "relay_ttft_timeout":
        # The relay has reset the upstream before any SSE data arrived, so this is
        # safe to retry even though the first-token deadline itself was long.
        return http_status == 504
    if (relay_error == "downstream_disconnect"
            or error_type in {"IncompleteRead", "RemoteDisconnected"}):
        return True
    if http_status == 502 and relay_error == "upstream_timeout":
        return True
    if http_status in {500, 502, 503, 504}:
        # Only a FAST 5xx is a provable transient blip.  A slow one may have done
        # the work upstream, so resending is ambiguous; it stays terminal.
        return (relay_error != "downstream_disconnect"
                and elapsed_ns < FAST_TRANSPORT_LIMIT_NS)
    if (elapsed_ns >= FAST_TRANSPORT_LIMIT_NS
            or relay_error in {"downstream_disconnect", "upstream_timeout"}
            or (http_status is not None and 400 <= http_status < 500)):
        return False
    return (
        relay_error == "upstream_reset"
        or error_type in {"ConnectionResetError", "ConnectionAbortedError", "BrokenPipeError",
                          "RemoteDisconnected"}
        or isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError))
    )


def backend_accounting(response: Mapping[str, object], *,
                       include_prompt_cache: bool = False) -> dict[str, object]:
    response = response if isinstance(response, Mapping) else {}
    timings = response.get("timings")
    timings = timings if isinstance(timings, Mapping) else {}
    usage = response.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    from kernel_prompt_accounting import normalize_prompt_tokens, compatibility_fields
    counts = normalize_prompt_tokens(response)
    accounting = {
        "prompt_tokens": counts,
        "prompt_n": counts["uncached"],
        "prompt_ms": _measured(timings.get("prompt_ms")),
        "cache_n": counts["cached"],
        "usage_prompt_tokens": _measured(usage.get("prompt_tokens"), integer=True),
        "source": "backend_response", "raw_timings": dict(timings), "raw_usage": dict(usage)}
    accounting.update(compatibility_fields(counts))
    if include_prompt_cache:
        from kernel_prompt_cache import prompt_cache_response_fields
        accounting.update(prompt_cache_response_fields(accounting))
    return accounting


def _cache_prefill_response_row(
    capture: object,
    accounting: Mapping[str, object],
    *,
    response: Mapping[str, object],
    request: Mapping[str, object],
) -> dict[str, object] | None:
    modes = (
        os.environ.get("STATEFRAME_CACHE_PREFILL_ECONOMICS", "off"),
        os.environ.get("STATEFRAME_TRAILING_PANELS", "off"),
    )
    if all(value == "off" for value in modes):
        return None
    from cache_prefill_economics import response_accounting_enabled, response_row

    if not response_accounting_enabled():
        return None
    return response_row(capture, accounting, response=response, request=request)


def _reconcile_frame_journal_gate(capture: object, body: object) -> None:
    """Run the optional frame correction without widening the request failure path."""
    try:
        from kernel_frame_journal_gate import reconcile_request_body
        reconcile_request_body(
            getattr(capture, "journal", None),
            body,
            request_id=getattr(capture, "request_id", None),
            step=getattr(capture, "step", None),
        )
    except Exception:
        pass


class RequestCapture:
    def __init__(self, journal: AuditJournal, step: int, snapshot_id: str, *,
                 call_purpose: str | None = None,
                 endpoint_profile: object | None = None,
                 parent_meter: object | None = None):
        self.journal = journal
        self.step = step
        self.snapshot_id = snapshot_id
        self.endpoint_profile = endpoint_profile if endpoint_profile is not None else _configured_endpoint_profile()
        if parent_meter is None:
            try:
                from kernel_pilot import ACTIVE_PARENT
                parent_meter = ACTIVE_PARENT.get()
            except (ImportError, LookupError):
                parent_meter = None
        if parent_meter is None:
            parent_meter = getattr(journal, "parent_meter", None)
        self.parent_meter = parent_meter
        if parent_meter is not None:
            try:
                setattr(journal, "parent_meter", parent_meter)
            except (AttributeError, TypeError):
                pass
        # Provider-visible identity is distinct from the random host journal directory.
        # The profiler adopts this identity; it never supplies it.
        from kernel_wire_identity import next_wire_id
        # DETFIX3: never feed snapshot_id here. It digests a frame that
        # carries live environment readings (timestamps, disk bytes, host epoch), so it
        # differs between any two runs and made this model-visible ID per-run random.
        # Journal sequence + step + the per-position ordinal are unique and replay-stable.
        if os.environ.get("SF_LAB_DETFIX3") == "legacy":
            self.request_id = next_wire_id(journal, "request", step, snapshot_id)
        else:
            self.request_id = next_wire_id(journal, "request", step)
        active_profile = _ACTIVE_HOST_SPAN_PROFILE.get()
        if active_profile is not None:
            recorder = getattr(active_profile, "recorder", None)
            claim = getattr(recorder, "claim_request_profile", None)
            try:
                claimed = bool(claim(active_profile, self.request_id)) if callable(claim) else False
            except Exception:  # profiling is advisory; it cannot change the request
                claimed = False
            if claimed and active_profile.step is None and type(step) is int:
                active_profile.step = step
        self._send_generation = 0
        self.model_router_call_purpose = (
            str(call_purpose) if call_purpose not in (None, "") else None)
        self.model_router_child_purpose = self.model_router_call_purpose
        self.selected_response_event_id: str | None = None
        self.response_event_id: str | None = None
        self.transport_attempt_id: str | None = None
        # The response is also persisted in response_received. Keep these exact
        # message fields live so the next StateFrame can carry the prior model
        # statement without changing the history-loop transcript.
        self.response_reasoning_content: str | None = None
        self.response_content: str | None = None
        # Provider-returned tool calls are provenance, not model-visible state.
        # F4 uses these IDs to distinguish a real exchange from host-injected
        # synthetic verification traffic without changing the legacy capture.
        self.provider_tool_calls: list[dict] = []
        self.provider_response_message: dict | None = None
        # Bound by StateFrameRuntime after the producer's exact ticket has been
        # selected.  The final wire receipt is emitted only when this is set.
        self.ticket_identity: dict[str, object] | None = None
        self.request_action_contract: dict[str, object] | None = None
        self.frame_request_allocation: FrameRequestAllocation | None = None
        self.qualified_request = None
        self._qualified_request_gate_passed = False
        self._frame_request_allocation_candidate = None
        # f4e (release blocker on 72beeb96: AttributeError _request_source_closure_override at
        # kernel_request_candidate.py:109 on live benchmark/benchmark): every attribute other modules set lazily on a capture is
        # initialized here with the default its readers already assume. Guard: probes/test_request_capture_lazy_attrs_unittest.py.
        self._request_source_closure_override = None
        self._frame_append_pending = None
        self._frame_append_active = False
        self._frame_append_mode = None
        self._lean_required_source_paths = ()
        self._one_snapshot_degraded_validation = None
        self._one_snapshot_wire_invariant_violated = False
        self._boundary_recovery_attempted = False
        self._boundary_recovery_original_request_id = None
        self._frame_candidate_inputs = None
        self._frame_candidate_measurement = None
        self._framefit5_newest_user_suffix = ()
        self._framefit5_owner_presentation_contract = None
        self._framefit5_task_message = None
        self._qualified_adapter = None
        self._release_prefix_boundary = None
        self._replacement_allocation = None
        self._request_measurement = None
        self._request_protocol = None
        self._stateframe_page_log_pending = None
        self._stateframe_tool_guidance_pending = ()
        self._stateframe_tool_menu_memo_frame_digest = None
        self._stateframe_tool_menu_memo_key = None
        self._stateframe_tool_menu_memo_shadow_match = None
        self._stateframe_tool_menu_memo_surface_bytes = None
        self._stateframe_tool_surface_pending = None
        self._tokenize_deadline = None
        self._tokenize_endpoint = None
        self._tokenize_logical_request_id = None
        self._visibility_required_paths = ()
        self._wire_truth_context = None
        self._wire_truth_rebuild_used = False
        self._wire_truth_runtime = None
        self._wire_truth_side_call = False
        # Deliberately NOT initialized (readers distinguish "unset"): _framefit5_semantic_messages (hasattr + body fallback),
        # _stateframe_tool_menu_memo_frame_snapshot (_MISSING sentinel), _replacement_lineage (falls back to request_id).
        self._qualified_request_gate = None
        self._qualified_host_text_preparer = None
        self.owner_presentation = None
        self._framefit5_pending_qualification = False
        self.final_tool_surface: dict[str, object] | None = None
        self.lean_prepared_history: list[dict[str, object]] | None = None
        self.lean_prepared_history_sha256: str | None = None
        self.lean_append_only_receipt: dict[str, object] | None = None
        self.lean_capacity_outcome: dict[str, object] | None = None
        self.stateframe_f4_mode = "off"
        self.stateframe_think_prefill_mode = "off"
        self.stateframe_think_prefill_text = ""
        self.stateframe_think_prefill_tokens = 0
        self.stateframe_think_prefill_echoed = False
        self.self_plan_wants_plan = False
        self.self_plan_frame = None
        self.self_plan_trigger = "first"
        self.self_plan_plan_text = ""
        self.self_plan_plan_tokens = 0
        self.self_plan_plan_request_id = None
        self.self_plan_plan_accounting: dict[str, object] = {}
        self.self_plan_pair_cache = False
        self.prompt_cache_arm: str | None = None
        # The long-horizon telemetry hook is deliberately opt-in.  Keeping
        # the role on the capture lets the history summarizer label its own
        # provider response without changing ordinary request records.
        self.long_horizon_request_role = "solver"
        self.long_horizon_rendered_prompt_tokens: int | None = None
        self.long_horizon_rendered_prompt_tokens_source: str | None = None
        self.long_horizon_logical_depth_tokens: int | None = None
        self.long_horizon_cumulative_logical_depth_tokens: int | None = None
        self.long_horizon_cumulative_logical_depth_source: str | None = None
        self.long_horizon_depth_tracker = None
        self.history_summary_ordinal = 0
        self.frame_epoch_ordinal = 0
        self.hybrid_crossed = False
        self.hybrid_crossing_request_id: str | None = None
        self.hybrid_crossing = False
        self.phase = 1
        self.presentation = "history"
        self.cumulative_logical_depth_tokens: int | None = None
        self.cumulative_logical_depth_source: str | None = None
        self.long_horizon_history_summary_ordinal = 0
        self.long_horizon_frame_epoch_ordinal = 0
        self.long_horizon_hybrid_crossed = False
        self.long_horizon_hybrid_crossing_request_id: str | None = None
        self.long_horizon_hybrid_crossing = False
        self.long_horizon_phase = 1
        self.long_horizon_presentation = "history"
        # Kept as a compatibility alias.  Runtime binding sets it to the sum
        # of history_summary_ordinal and frame_epoch_ordinal.
        self.long_horizon_compaction_ordinal = 0
        # A length finish is retained even when the transport layer retries so
        # the next StateFrame request can suppress automatic escalation.
        self.finish_reason: str | None = None
        self.output_cap_hit = False
        # V43G keeps logical P+C across a discarded degenerate response and
        # its one retry; the retry is a new wire attempt, never free work.
        self.v43g_degradation_retry_used = False
        self.v43g_logical_prompt_tokens = 0
        self.v43g_logical_completion_tokens = 0
        self.accounting: dict[str, object] = {}
        self.advice_candidates: list[dict] = []
        self.transport_errors: list[dict] = []
        self.provider_errors: list[dict] = []
        self.semantic_errors: list[dict] = []
        # D2 playbook metadata is populated only when the optional sidecar was
        # actually retained in the prepared request.
        self.playbook_card_id: str | None = None
        self.playbook_card_version: str | None = None
        self.playbook_offer_event_id: str | None = None
        self.playbook_adoption_journaled = False

    def _ensure_contract_bound_evidence_pin(
            self, body: dict[str, object], *, runtime: object | None = None,
            side_call_contract: object = None, preparation: object | None = None) -> bool:
        """Restore the typed owner pin after final body transforms, before serialization."""
        if side_call_contract is not None or not isinstance(body.get("messages"), list):
            return False
        if runtime is None:
            try:
                from kernel_pilot import ACTIVE_RUNTIME
                runtime = ACTIVE_RUNTIME.get()
            except (ImportError, LookupError):
                runtime = None
        if runtime is None:
            runtime = getattr(self, "completion_owner_runtime", None)
        if runtime is None:
            return False

        contract = getattr(self, "request_action_contract", None)
        runtime_contract = getattr(runtime, "_completion_owner_request_contract", None)
        if isinstance(runtime_contract, Mapping):
            runtime_request_id = runtime_contract.get("request_id")
            if runtime_request_id in (None, self.request_id):
                contract = runtime_contract
        if (not isinstance(contract, Mapping)
                or contract.get("evidence_pin_wire_digest") is None):
            return False

        from kernel_completion_owner import _digest, _wire_evidence_pin
        from kernel_refusal_compact import ensure_final_evidence_pin
        before_pin, before_count = _wire_evidence_pin(body["messages"])
        messages, changed = ensure_final_evidence_pin(runtime, body["messages"])
        if not changed:
            return False
        from kernel_stateframe_framefit5 import (
            assemble_framefit5_request, has_framefit5_state,
        )
        if has_framefit5_state(messages):
            from kernel_request_reprepare import (
                _source_delivery_inputs, _source_messages_from_preparation,
            )
            source_plan, source_contents = _source_delivery_inputs(
                self, runtime, preparation,
            )
            source_messages = _source_messages_from_preparation(preparation, messages)
            messages = assemble_framefit5_request(
                messages,
                source_messages,
                task_message=getattr(self, "_framefit5_task_message", None),
                source_plan=source_plan,
                sources=source_contents,
                runtime=runtime,
                owner_contract=contract,
                owner_source_messages=source_messages,
                frame_request_allocation=self.frame_request_allocation,
            )
        body["messages"] = messages
        # Forensics: record what the late transforms left, so the producer that
        # dropped/altered the pin can be identified from the journal alone.
        self.journal.emit(
            "request_action_pin_repaired",
            producer="kernel_measurement",
            request_id=self.request_id,
            boundary="pre_wire",
            wire_pin_count_before=before_count,
            wire_pin_digest_before=_digest([before_pin]) if before_pin else None,
            contract_pin_digest=contract.get("evidence_pin_wire_digest"),
            dispatch=False,
        )
        return True

    def _request_capture_pre_wire_extensions(
            self, body: dict[str, object], endpoint: str, key: str, *,
            deadline: float | None = None,
            side_call_contract: object = None,
            preparation: object | None = None,
            runtime: object | None = None,
            read_only: bool = False) -> dict[str, object]:
        """Return provider-owned request fields for the shared capture seam."""
        del endpoint, key
        from kernel_request_kind import is_auxiliary
        request_action_contract = None
        if runtime is None:
            try:
                from kernel_pilot import ACTIVE_RUNTIME
                runtime = ACTIVE_RUNTIME.get()
            except (ImportError, LookupError):
                runtime = None
        if runtime is None:
            # ThreadPoolExecutor does not copy ContextVars. The capture's owner
            # binding keeps owner-on wire assertions identical on both paths.
            runtime = getattr(self, "completion_owner_runtime", None)
        if not read_only and not is_auxiliary(self):
            self._ensure_contract_bound_evidence_pin(
                body, runtime=runtime, side_call_contract=side_call_contract,
                preparation=preparation,
            )
        frozen_contract = self.request_action_contract
        side_extension: dict[str, object] | None = None
        if side_call_contract is not None:
            side_extension = {
                "side_call_contract": {
                    "decision_id": str(getattr(side_call_contract, "decision_id", "")),
                    "permitted_effect": str(getattr(side_call_contract, "permitted_effect", "")),
                    "tool_name": str(getattr(side_call_contract, "tool_name", "")),
                }
            }
        elif (runtime is not None
                and not is_auxiliary(self)
                and (getattr(runtime, "stateframe_completion_owner_mode", "off") == "on"
                     or (isinstance(self.request_action_contract, Mapping)
                         and self.request_action_contract.get("contract_kind")
                         == "lean_stable_surface"))):
            from kernel_completion_owner import assert_request_action_contract
            request_action_contract = assert_request_action_contract(
                runtime,
                body.get("messages"),
                body.get("tools"),
                request_id=self.request_id,
                boundary="wire",
                read_only=read_only,
            )
            self.request_action_contract = copy.deepcopy(request_action_contract)
        elif isinstance(frozen_contract, Mapping):
            from kernel_completion_owner import (
                RequestActionContractError,
                assert_request_action_contract,
            )
            try:
                request_action_contract = assert_request_action_contract(
                    runtime,
                    body.get("messages"),
                    body.get("tools"),
                    request_id=self.request_id,
                    boundary="wire",
                    frozen_contract=frozen_contract,
                    read_only=read_only,
                )
            except RequestActionContractError as error:
                self.journal.emit(
                    "provider_boundary_contract_error",
                    producer="kernel_completion_owner",
                    event_schema="stateframe-request-action-contract/1",
                    request_id=self.request_id,
                    boundary="wire",
                    error=type(error).__name__,
                    dispatch=False,
                )
                raise
            self.request_action_contract = copy.deepcopy(request_action_contract)
            self.journal.emit(
                "provider_boundary_contract_asserted",
                producer="kernel_completion_owner",
                event_schema="stateframe-request-action-contract/1",
                request_id=self.request_id,
                boundary="wire",
                route=request_action_contract.get("route"),
                advertised_actions=request_action_contract.get("advertised_actions", []),
                serialized_actions=[
                    str(tool.get("function", {}).get("name", ""))
                    for tool in body.get("tools", [])
                    if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
                ],
                admitted=True,
                decision_digest=request_action_contract.get("decision_digest"),
            )
        contract = None
        if self.journal.manifest.get("stateframe_provider_adapter") == "on":
            from kernel_provider_adapter import validate_wire_request
            contract = validate_wire_request(
                body,
                manifest=self.journal.manifest,
                deadline=deadline,
                cold_resume=bool(getattr(self, "provider_adapter_cold_resume", False)),
                cancel_requested=bool(getattr(self, "cancel_requested", False)),
            )
            self.provider_adapter_contract = copy.deepcopy(contract)
        fields: dict[str, object] = {}
        if contract is not None:
            fields["provider_adapter"] = copy.deepcopy(contract)
        if request_action_contract is not None:
            fields["request_action_contract"] = copy.deepcopy(request_action_contract)
        if side_extension is not None:
            fields.update(side_extension)
        return fields

    def _request_capture_response_extensions(
            self, response: dict[str, object], *, tools: object = None) -> dict[str, object]:
        """Return provider-owned response fields for the shared capture seam."""
        contract = None
        if self.journal.manifest.get("stateframe_provider_adapter") == "on":
            from kernel_provider_adapter import validate_response
            contract = validate_response(response, tools=tools)
            self.provider_adapter_response_contract = copy.deepcopy(contract)
        return ({"provider_adapter_response": copy.deepcopy(contract)}
                if contract is not None else {})

    def _ensure_final_wire_ticket(self, body: dict[str, object]) -> dict[str, object]:
        """Re-carry the authorized ticket at the last mutable request seam.

        ``prepare`` carries the ticket through ``ensure_ticket_messages``, but a
        later reshaping (post-green terminal/reserve composition, capture
        budget) can rebuild the message list without it.  b923: after GREEN the
        5th request lost the ticket and the final-wire guard killed the run
        before authored_check was offered.  This re-applies the SAME ticket
        renderer (one owner, no second ticket text) and journals the repair;
        the final-wire guard below still verifies the actual bytes.
        """
        if self.ticket_identity is None or not isinstance(body.get("messages"), list):
            return body
        from kernel_request_evidence_closure import TicketIdentity, render_ticket_brief
        identity = TicketIdentity(str(self.ticket_identity["task_id"]),
                                  str(self.ticket_identity["text"]))
        messages = body["messages"]
        if any(isinstance(item, Mapping) and identity.text in str(item.get("content") or "")
               for item in messages):
            return body
        repaired = dict(body)
        repaired["messages"] = [*messages, {"role": "user", "content": render_ticket_brief(identity)}]
        self.journal.emit("ticket_final_wire_repaired", producer="kernel_measurement",
                          request_id=self.request_id, step=self.step,
                          message_count_before=len(messages), dispatch=False)
        return repaired

    def _v43g_prepare_body(self, body: dict[str, object]) -> None:
        """Apply V43G's terminal-only function choice at the final wire seam."""
        from kernel_request_kind import is_auxiliary
        if is_auxiliary(self):
            return
        from kernel_v43g_guard import forced_done_tool_choice, terminal_choice_mode
        if terminal_choice_mode() != "on":
            return
        try:
            from kernel_pilot import ACTIVE_RUNTIME
            runtime = ACTIVE_RUNTIME.get()
            from kernel_completion_owner import is_reserved_request
            terminal_offer = is_reserved_request(
                runtime, body.get("messages"), body.get("tools"))
        except (ImportError, LookupError, TypeError, ValueError):
            terminal_offer = False
        if not terminal_offer:
            return
        tool_choice = forced_done_tool_choice(body.get("tools"))
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

    def _v43g_note_usage(self, response: Mapping[str, object]) -> dict[str, int | None]:
        from kernel_v43g_guard import usage_counts
        counts = usage_counts(response)
        prompt = counts["prompt_tokens"]
        completion = counts["completion_tokens"]
        if prompt is not None:
            self.v43g_logical_prompt_tokens += prompt
        if completion is not None:
            self.v43g_logical_completion_tokens += completion
        return counts

    def _v43g_retry_degenerate(self, response: dict[str, object], body: dict[str, object],
                               endpoint: str, key: str, *, deadline: float | None,
                               side_call_contract: object = None) -> dict | None:
        """Discard one provably repetitive response and retry the same request once."""
        from kernel_v43g_guard import (
            detect_response, degeneration_mode, response_journal_fields,
        )
        counts = self._v43g_note_usage(response)
        if degeneration_mode() != "on" or self.v43g_degradation_retry_used:
            return None
        finding = detect_response(response)
        if finding is None:
            return None
        if getattr(self, "request_kind", None) == "framefit5_primary" and self.qualified_request is None:
            from kernel_request_kind import reject_changed_body_retry
            reject_changed_body_retry(self)
        # Resubmit the same draft input; primary qualification reuses its
        # sealed output only after checking the input digest and live authority.
        retried_body = copy.deepcopy(dict(body))
        if retried_body != body:
            from kernel_request_kind import reject_changed_body_retry
            reject_changed_body_retry(self)
        # Journal the sealed provider temperature, not a draft default.
        submitted_body = (self.qualified_request.body
                          if self.qualified_request is not None else retried_body)
        retry_temperature = submitted_body.get("temperature", 1.0)
        self.journal.emit(
            "degeneration_detected",
            request_id=self.request_id,
            step=self.step,
            retry_scheduled=True,
            discarded_response=True,
            usage=counts,
            **_journal_safe_fields(response_journal_fields(
                finding, response,
                retry_temperature=float(retry_temperature),
                logical_prompt_tokens=self.v43g_logical_prompt_tokens,
                logical_completion_tokens=self.v43g_logical_completion_tokens,
            )),
        )
        self.v43g_degradation_retry_used = True
        return self.send(retried_body, endpoint, key, deadline=deadline, side_call_contract=side_call_contract)

    def _raise_malformed_provider_response(
            self, raw_response: object, detail: str, *, request_id: str | None = None,
            step: int | None = None, transport_attempt_id: str | None = None) -> None:
        """Record provider-shape failure without classifying it as transport."""
        raw_bytes = raw_response if isinstance(raw_response, bytes) else str(raw_response).encode()
        error = ProviderMalformedResponseError(detail)
        setattr(error, "journaled", True)
        record = self.journal.emit(
            "provider_malformed_response",
            request_id=request_id or self.request_id,
            transport_attempt_id=transport_attempt_id,
            step=self.step if step is None else step,
            error_type=type(error).__name__,
            detail=error.detail,
            response_bytes=len(raw_bytes),
            response_digest="sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
            outcome=error.outcome,
            failure_origin=error.failure_origin,
            dispatch=True,
        )
        self.provider_errors.append(record)
        raise error

    def _durable_response_received(self) -> bool:
        """Consult the journal's response index before a retry can resubmit a request."""
        if self.response_event_id is not None:
            return True
        checker = getattr(self.journal, "has_response_received", None)
        return bool(checker(self.request_id)) if callable(checker) else False

    def _guard_retry_submission(
            self, attempt: int,
            attempt_reasons: Sequence[Mapping[str, object]]) -> None:
        """Stop a scheduled retry if its durable request already has a response."""
        if attempt <= 0 or not self._durable_response_received():
            return
        self.journal.emit(
            "provider_transport_retry_aborted",
            producer="kernel_measurement",
            fact_owner="HOST",
            request_id=self.request_id,
            step=self.step,
            attempt=attempt,
            reason="response_already_received",
            attempt_reasons=list(attempt_reasons),
            outcome=HARNESS_TRANSPORT,
            failure_origin="harness",
            dispatch=False,
        )
        raise HarnessTransportError(
            attempts=attempt,
            error_type="ResponseAlreadyReceived",
            http_status=None,
            relay_error=None,
            transport_class="upstream",
            attempt_reasons=attempt_reasons,
        )

    def _assert_wire_string_hygiene(self, wire: bytes) -> None:
        from kernel_wire_string_hygiene import first_wire_string_violation

        violation = first_wire_string_violation(wire)
        if violation is None:
            return
        self.journal.emit(
            "wire_string_hygiene_violation",
            producer="kernel_measurement",
            request_id=self.request_id,
            step=self.step,
            path=violation.path[:192],
            violation_kind=violation.kind,
            outcome=HARNESS_CONTRACT,
            failure_origin="harness",
            dispatch=False,
        )
        from kernel_request_reprepare import RequestPreparationContractError

        error = RequestPreparationContractError(
            "serialized request failed wire string hygiene")
        error.stop_code = "wire_string_hygiene_violation"
        raise error

    def _send_transport_with_retries(
            self, body: dict[str, object], wire: bytes, endpoint: str, key: str,
            *, admission: object | None, socket_timeout: float, deadline: float | None,
            wall_timeout: float, backend_identity: Mapping[str, object],
            request_extensions: Mapping[str, object],
            kv_chunk_reuse_transport: object | None = None,
            required: bool = False, permit_prefix: str | None = None,
            transport_attempt_limit: int | None = None,
            journal_timeouts: bool = True,
            boundary_state: _BoundaryRecoveryContext | None = None,
            frozen_retry_only: bool = False,
    ) -> tuple[dict, dict[str, object], int]:
        """Retry classified transport failures under the shared bounded policy."""
        from kernel_pilot import POLICY, ParentBudgetExhausted
        from kernel_boundary_recovery import enabled as boundary_recovery_enabled
        import kernel_one_snapshot_preparation as _one_snapshot
        one_snapshot_retry_enabled = _one_snapshot.enabled()

        def _record_retry_disagreement(
                validator: str, second_verdict: str,
                candidate_wire: bytes,
        ) -> bool:
            if not one_snapshot_retry_enabled:
                return False
            first_wire = getattr(self, "_one_snapshot_first_wire", None)
            if not isinstance(first_wire, bytes):
                return False
            preparation = getattr(self, "_request_preparation_input", None)
            first_id = getattr(self, "_one_snapshot_first_snapshot_id", None)
            if not isinstance(first_id, str):
                first_id = _one_snapshot.snapshot_id(preparation, first_wire)
            return _one_snapshot.record_disagreement(
                self,
                validator=validator,
                first_verdict=str(getattr(self, "_one_snapshot_first_verdict", "UNKNOWN")),
                second_verdict=second_verdict,
                first_snapshot_id=first_id,
                second_snapshot_id=_one_snapshot.snapshot_id(preparation, candidate_wire),
                flag_enabled=one_snapshot_retry_enabled,
            )

        if self._durable_response_received():
            raise HarnessTransportError(
                attempts=0, error_type="ResponseAlreadyReceived", http_status=None,
                relay_error=None, transport_class="upstream")
        self._assert_wire_string_hygiene(wire)

        if transport_attempt_limit is not None and (
                type(transport_attempt_limit) is not int or transport_attempt_limit < 1):
            raise ValueError("transport_attempt_limit must be a positive integer")
        ordinary_attempt_limit = min(
            POLICY.transport_attempts,
            transport_attempt_limit if transport_attempt_limit is not None
            else POLICY.transport_attempts,
        )
        per_request_timeout_active = per_request_timeout_seconds() is not None
        ttft_retry_budget_on = ttft_retry_budget_enabled()
        ttft_retry_budget_limit = (
            ttft_retry_budget_seconds() if ttft_retry_budget_on else None)
        timeout_retry_used = False
        timeout_retry_attempt: int | None = None
        attempt_reasons: list[dict[str, object]] = []
        provider_permit: object | None = None
        ordinary_attempt_count = 0
        ttft_retry_count = 0
        ttft_retry_started_at: float | None = None
        ttft_retry_budget_deadline: float | None = None
        ttft_retry_deadline: float | None = None
        ttft_retry_deadline_reason: str | None = None

        first_submission_event_id: str | None = None
        certificate = build_run_certificate(
            run_id=str(getattr(self.journal, "run_id", UNKNOWN)),
            body=body,
            wire=wire,
            manifest=getattr(self.journal, "manifest", None),
            backend_identity=backend_identity,
            profile=reasoning_profile(manifest=getattr(self.journal, "manifest", None)),
        )
        frozen_request_id = self.request_id
        frozen_wire_digest = "sha256:" + hashlib.sha256(wire).hexdigest()
        frozen_request_digest = certificate.request_body_digest
        frozen_contract_digest = (
            digest(self.request_action_contract)
            if isinstance(getattr(self, "request_action_contract", None), Mapping)
            else None
        )
        attempt = 0
        while True:
            if boundary_state is not None:
                boundary_state.phase = (
                    "transport_retry_validation" if attempt else "transport_attempt_validation")
            self._guard_retry_submission(attempt, attempt_reasons)
            from kernel_request_kind import guard_physical_send
            guard_physical_send(self, getattr(self, "_wire_truth_runtime", None), body, wire)
            sealed = self.qualified_request
            if sealed is not None:
                from kernel_request_seal import RequestSealError
                sealed.validate(self, self._wire_truth_runtime, body)
                from kernel_request_kind import gate_required
                if wire != sealed.wire or (gate_required(self) and not self._qualified_request_gate_passed):
                    raise RequestSealError("transport retry lacks the exact qualified wire")
            if json.dumps(body).encode() != wire:
                first_wire = getattr(self, "_one_snapshot_first_wire", None)
                if _record_retry_disagreement(
                        "transport_body_wire", "BYTES_CHANGED", wire):
                    body = json.loads(first_wire)
                    wire = bytes(first_wire)
                else:
                    if not boundary_recovery_enabled():
                        raise RuntimeError("transport body differs from qualified serialized request")
                    from kernel_request_reprepare import RequestPreparationContractError
                    error = RequestPreparationContractError(
                        "transport body differs from qualified serialized request")
                    error.stop_code = "transport_body_wire_mismatch"
                    raise error
            if sealed is not None:
                # Recheck freshness on every physical attempt. The immutable
                # record already binds body, source, menu and owner authority;
                # legacy repair/refresh paths must never see sealed bytes.
                pass
            elif attempt > 0 and frozen_retry_only:
                from kernel_boundary_recovery import BoundaryRecoveryStop
                current_contract = getattr(self, "request_action_contract", None)
                current_contract_digest = (
                    digest(current_contract) if isinstance(current_contract, Mapping) else None
                )
                if (self.request_id != frozen_request_id
                        or "sha256:" + hashlib.sha256(wire).hexdigest() != frozen_wire_digest
                        or certificate.request_body_digest != frozen_request_digest
                        or current_contract_digest != frozen_contract_digest):
                    raise BoundaryRecoveryStop(
                        "frozen transport submission identity is unavailable")
            elif attempt > 0:
                from kernel_wire_truth_gate import capture_context, check_wire
                from kernel_request_reprepare import RequestPreparationContractError
                try:
                    from kernel_pilot import ACTIVE_RUNTIME
                    retry_runtime = ACTIVE_RUNTIME.get()
                except (ImportError, LookupError):
                    retry_runtime = getattr(self, "_wire_truth_runtime", None)
                if retry_runtime is None:
                    retry_runtime = getattr(self, "_wire_truth_runtime", None)
                retry_body = json.loads(wire)
                retry_check_failed = False
                try:
                    from kernel_request_reprepare import qualify_source
                    wire = qualify_source(self, retry_runtime,
                        getattr(self, "_request_preparation_input", None), retry_body, wire)
                    self._request_capture_pre_wire_extensions(retry_body, endpoint, key,
                        deadline=deadline, side_call_contract=getattr(
                            self, "_wire_truth_side_call_contract", None))
                    if json.dumps(retry_body).encode() != wire:
                        if not boundary_recovery_enabled():
                            raise RuntimeError("retry authority changed qualified serialized request")
                        error = RequestPreparationContractError(
                            "retry authority changed qualified serialized request")
                        error.stop_code = "retry_authority_wire_mismatch"
                        raise error
                    retry_context = capture_context(
                        self, retry_runtime, retry_body,
                        side_call=bool(getattr(self, "_wire_truth_side_call", False)))
                    retry_check = check_wire(
                        wire, retry_context, emit=self.journal.emit,
                        attempt=attempt + 1, allow_done_repair=False,
                    )
                    if retry_check.wire != wire or retry_check.verdict not in {
                            "PASS", "DISABLED"}:
                        raise RequestPreparationContractError(
                            "wire truth retry did not pass strict qualification")
                except Exception as error:
                    if _record_retry_disagreement(
                            "transport_retry_validation", type(error).__name__, wire):
                        first_wire = getattr(self, "_one_snapshot_first_wire", None)
                        if isinstance(first_wire, bytes):
                            body = json.loads(first_wire)
                            wire = first_wire
                        self._wire_truth_check = getattr(
                            self, "_one_snapshot_first_check", None,
                        )
                        retry_check_failed = True
                    else:
                        raise
                if not retry_check_failed:
                    self._wire_truth_check = retry_check
            parent = self.parent_meter
            pending_id = (
                parent.take_pending_provider_permit()
                if parent is not None and attempt == 0 else None
            )
            active_deadline = deadline
            if ttft_retry_deadline is not None:
                active_deadline = (
                    ttft_retry_deadline if active_deadline is None
                    else min(active_deadline, ttft_retry_deadline))
            remaining = (
                None if active_deadline is None
                else active_deadline - time.monotonic())
            if remaining is not None and remaining <= 0:
                if attempt > 0:
                    self.journal.emit(
                        "provider_transport_retry_aborted",
                        producer="kernel_measurement",
                        fact_owner="HOST",
                        request_id=self.request_id,
                        step=self.step,
                        attempt=attempt + 1,
                        reason=(ttft_retry_deadline_reason or "total_deadline_expired"),
                        ordinary_attempt_count=ordinary_attempt_count,
                        ordinary_attempt_limit=ordinary_attempt_limit,
                        ttft_retry_count=ttft_retry_count,
                        cumulative_ttft_wait_seconds=(
                            max(0.0, time.monotonic() - ttft_retry_started_at)
                            if ttft_retry_started_at is not None else None),
                        attempt_reasons=list(attempt_reasons),
                        outcome=HARNESS_TRANSPORT,
                        failure_origin="harness",
                        dispatch=False,
                    )
                    raise HarnessTransportError(
                        attempts=attempt,
                        error_type="TimeoutError",
                        http_status=None,
                        relay_error=None,
                        attempt_reasons=attempt_reasons,
                    ) from TimeoutError("request deadline expired before retry")
                if parent is not None:
                    if pending_id is not None:
                        pending = parent.provider_permit_for_id(pending_id)
                        if pending is None:
                            raise ParentBudgetExhausted(
                                "model_calls", "unknown_legacy_permit")
                        # Re-enter the durable boundary so an expired legacy
                        # admission gets an explicit non-charged denial event.
                        parent.admit_provider(
                            permit_id=pending.permit_id, reason=pending.reason,
                            journal=self.journal, optional=pending.optional,
                            deadline=deadline)
                    provider_permit_id = f"{permit_prefix or self.request_id}:provider:1"
                    parent.admit_provider(
                        permit_id=provider_permit_id,
                        reason=(f"{self.model_router_call_purpose or 'provider'}:transport:1"),
                        journal=self.journal, optional=not required,
                        deadline=deadline)
                    raise ParentBudgetExhausted("model_calls", "deadline_expired")
                raise HarnessTransportError(
                    attempts=attempt, error_type="TimeoutError", http_status=None,
                    relay_error=None, attempt_reasons=attempt_reasons,
                ) from TimeoutError("request deadline expired before submission")
            attempt_started = time.monotonic()
            attempt_wall_timeout = (
                wall_timeout if remaining is None else min(wall_timeout, remaining))
            attempt_deadline = deadline
            if per_request_timeout_active:
                per_request_deadline = attempt_started + attempt_wall_timeout
                attempt_deadline = (
                    per_request_deadline if deadline is None
                    else min(deadline, per_request_deadline))
            if attempt_deadline is None:
                attempt_deadline = attempt_started + attempt_wall_timeout
            if ttft_retry_deadline is not None:
                attempt_deadline = min(attempt_deadline, ttft_retry_deadline)
            attempt_timeout = min(socket_timeout, attempt_wall_timeout)
            permit = None
            if parent is not None:
                if attempt > 0:
                    permit = provider_permit
                    if permit is None:
                        raise ParentBudgetExhausted(
                            "model_calls", "missing_logical_request_permit")
                elif pending_id is not None:
                    permit = parent.provider_permit_for_id(pending_id)
                    if permit is None:
                        raise ParentBudgetExhausted("model_calls", "unknown_legacy_permit")
                else:
                    provider_permit_id = f"{permit_prefix or self.request_id}:provider:1"
                    permit = parent.admit_provider(
                        permit_id=provider_permit_id,
                        reason=(f"{self.model_router_call_purpose or 'provider'}:transport:1"),
                        journal=self.journal,
                        optional=not required,
                        deadline=deadline,
                    )
                if attempt == 0:
                    provider_permit = permit
            submission_fields: dict[str, object] = {
                "snapshot_id": self.snapshot_id,
                "context_hash": digest({"messages": body.get("messages"), "tools": body.get("tools")}),
                "advice_delivery":[
                    {**item, "present_in_submitted_context": item["text"] in "\n".join(
                        str(message.get("content") or "")
                        for message in body.get("messages", []))}
                    for item in self.advice_candidates
                ],
                "backend_identity": backend_identity,
                "transport_attempt": attempt + 1,
                "retry_index": attempt,
                "payload_hash": "sha256:" + hashlib.sha256(wire).hexdigest(),
                "request_body_digest": certificate.request_body_digest,
                "run_certificate": certificate.to_dict(),
            }
            degraded_validation = getattr(
                self, "_one_snapshot_degraded_validation", None,
            )
            if isinstance(degraded_validation, str) and degraded_validation:
                submission_fields["degraded_validation"] = degraded_validation
                submission_fields["wire_invariant_violated"] = bool(
                    getattr(self, "_one_snapshot_wire_invariant_violated", False)
                )
            if permit is not None:
                submission_fields["provider_permit"] = permit.mapping()
                submission_fields["provider_permit_id"] = permit.permit_id
            if attempt == 0:
                submission_fields.update({"payload": body, "payload_utf8": wire.decode()})
            else:
                submission_fields.update({
                    "payload_ref": first_submission_event_id,
                    "payload_delta": {
                        "base_attempt": 1,
                        "changed_keys": [],
                        "identical_wire": True,
                    },
                })
            preedit_delivery = None
            if attempt == 0 and getattr(self, "preedit_coverage_claim", None) is not None:
                from kernel_preedit_coverage_runtime import submission as preedit_submission
                preedit_delivery = preedit_submission(self, wire)
                if preedit_delivery is not None:
                    submission_fields["preedit_coverage_delivery"] = preedit_delivery
            restate_delivery = None
            if attempt == 0 and getattr(self, "restate_requirement_claim", None) is not None:
                from kernel_restate_requirement import submission as restate_submission
                restate_delivery = restate_submission(self, wire)
            self._guard_retry_submission(attempt, attempt_reasons)
            final_wire_body = json.loads(wire)
            self._evictfix_final_wire_messages = final_wire_body.get("messages", ())
            self.journal.emit(
                "request_final_wire_prepared",
                producer="kernel_measurement",
                event_schema=REQUEST_FINAL_WIRE_PREPARED_SCHEMA,
                body_artifact="final_wire_pre_dispatch",
                request_id=self.request_id,
                step=self.step,
                provider_request=True,
                transport_attempt=attempt + 1,
                payload_sha256="sha256:" + hashlib.sha256(wire).hexdigest(),
                messages=copy.deepcopy(final_wire_body.get("messages")),
                tools=copy.deepcopy(final_wire_body.get("tools")),
                dispatch=False,
            )
            if boundary_state is not None:
                boundary_state.phase = "provider_submission"
                boundary_state.submitted = True
            self._transport_submission_started = True
            sent = self.journal.emit(
                "request_submitted", request_id=self.request_id, step=self.step,
                socket_timeout_s=attempt_timeout, wall_timeout_s=attempt_wall_timeout,
                absolute_deadline_monotonic=attempt_deadline,
                hard_deadline_transport=attempt_deadline is not None,
                prompt_tokens=admission.prompt_tokens if admission is not None else None,
                measured_prefill_tokens_per_second=(
                    admission.prefill_tokens_per_second if admission is not None else None
                ),
                derived_deadline_seconds=(
                    admission.deadline_seconds if admission is not None else None
                ),
                admission=admission.mapping() if admission is not None else None,
                **_journal_safe_fields(submission_fields),
                **({"kv_chunk_reuse": kv_chunk_reuse_transport}
                   if kv_chunk_reuse_transport is not None else {}),
                **_journal_safe_fields({
                    key: value for key, value in request_extensions.items()
                    if key != "kv_chunk_reuse"
                }),
            )
            if attempt == 0:
                self.journal.emit(
                    "request_body",
                    producer="kernel_measurement",
                    event_schema=REQUEST_BODY_SCHEMA,
                    body_artifact="submitted_payload",
                    request_id=self.request_id,
                    step=self.step,
                    transport_attempt=attempt + 1,
                    payload=copy.deepcopy(body),
                    payload_utf8=wire.decode(),
                    payload_sha256="sha256:" + hashlib.sha256(wire).hexdigest(),
                    submitted_event_id=(
                        sent.get("event_id") if isinstance(sent, Mapping) else None
                    ),
                    dispatch=False,
                )
            from kernel_exposure_truth import emit_exposure_composition
            emit_exposure_composition(
                self.journal,
                wire=wire,
                request_id=str(self.request_id),
                step=self.step,
                submitted_event_id=str(sent["event_id"]),
                transport_attempt=attempt + 1,
            )
            if preedit_delivery is not None:
                from kernel_preedit_coverage_runtime import committed as preedit_committed
                preedit_committed(self, preedit_delivery)
                self.journal.emit(
                    "preedit_coverage_fired", request_id=self.request_id,
                    producer="kernel_preedit_coverage_runtime", submission_event_id=sent["event_id"],
                    on_wire=True, dispatch=False, **preedit_delivery,
                )
            if attempt == 0:
                from kernel_anchor_recovery import wire_receipt as anchor_wire_receipt
                anchor_delivery = anchor_wire_receipt(wire)
                if anchor_delivery is not None:
                    self.journal.emit(
                        "anchor_recovery_fired", request_id=self.request_id,
                        producer="kernel_anchor_recovery", submission_event_id=sent["event_id"],
                        on_wire=True, dispatch=False, **anchor_delivery,
                    )
                from kernel_read_once import wire_receipt as read_once_wire_receipt
                from kernel_read_once import final_wire_receipt as read_once_final_receipt
                read_once_final = read_once_final_receipt(wire)
                if read_once_final is not None:
                    self.journal.emit(
                        "read_once_final_fired", request_id=self.request_id,
                        producer="kernel_read_once", submission_event_id=sent["event_id"],
                        on_wire=True, dispatch=False, **read_once_final,
                    )
                read_once_delivery = read_once_wire_receipt(wire)
                if read_once_delivery is not None:
                    self.journal.emit(
                        "read_once_fired", request_id=self.request_id,
                        producer="kernel_read_once", submission_event_id=sent["event_id"],
                        on_wire=True, dispatch=False, **read_once_delivery,
                    )
                    from kernel_read_once import confirm_wire_delivery
                    confirm_wire_delivery(
                        getattr(self, "_wire_truth_runtime", None), read_once_delivery,
                    )
                from kernel_read_pointer import (
                    inline_wire_receipt as read_pointer_inline_wire_receipt,
                    wire_receipt as read_pointer_wire_receipt,
                )
                read_pointer_delivery = read_pointer_wire_receipt(wire)
                if read_pointer_delivery is not None:
                    self.journal.emit(
                        "read_pointer_fired", request_id=self.request_id,
                        producer="kernel_read_pointer", submission_event_id=sent["event_id"],
                        on_wire=True, dispatch=False, **read_pointer_delivery,
                    )
                read_pointer_inline_delivery = read_pointer_inline_wire_receipt(wire)
                if read_pointer_inline_delivery is not None:
                    self.journal.emit(
                        "read_pointer_inline_fired", request_id=self.request_id,
                        producer="kernel_read_pointer",
                        submission_event_id=sent["event_id"],
                        send_time_ns=sent["timestamp_ns"],
                        on_wire=True, dispatch=False,
                        **read_pointer_inline_delivery,
                    )
                from kernel_rejection_template import (
                    wire_receipt as rejection_template_wire_receipt,
                )
                rejection_template_delivery = rejection_template_wire_receipt(wire)
                if rejection_template_delivery is not None:
                    self.journal.emit(
                        "rejection_template_fired", request_id=self.request_id,
                        producer="kernel_rejection_template",
                        submission_event_id=sent["event_id"],
                        on_wire=True, dispatch=False, **rejection_template_delivery,
                    )
                try:
                    from kernel_pilot import ACTIVE_RUNTIME
                    restriction_runtime = ACTIVE_RUNTIME.get()
                except (ImportError, LookupError):
                    restriction_runtime = getattr(self, "completion_owner_runtime", None)
                if restriction_runtime is None:
                    restriction_runtime = getattr(self, "completion_owner_runtime", None)
                from kernel_rejection_template import (
                    floor_notice_receipt as rejection_template_floor_notice_receipt,
                )
                rejection_template_notice_delivery = rejection_template_floor_notice_receipt(
                    wire, runtime=restriction_runtime)
                if rejection_template_notice_delivery is not None:
                    notice_event = rejection_template_notice_delivery.pop("notice_event", None)
                    event_kind = (
                        str(notice_event)
                        if notice_event == "rejection_template_notice_fired"
                        else "rejection_template_notice_fired"
                    )
                    self.journal.emit(
                        event_kind,
                        request_id=self.request_id,
                        producer="kernel_rejection_template",
                        submission_event_id=sent["event_id"],
                        on_wire=True,
                        dispatch=False,
                        **rejection_template_notice_delivery,
                    )
                from kernel_rejection_template_restrict import (
                    wire_receipt as rejection_template_restrict_wire_receipt,
                )
                rejection_template_restrict_delivery = rejection_template_restrict_wire_receipt(
                    wire, runtime=restriction_runtime, request_id=self.request_id)
                if rejection_template_restrict_delivery is not None:
                    ranked_event = rejection_template_restrict_delivery.pop(
                        "ranked_event", None)
                    event_kind = (
                        str(ranked_event) if ranked_event in {
                            "rejection_template_restrict_ranked",
                            "rejection_restrict_only_ranked",
                        } else "rejection_template_restrict_fired"
                    )
                    self.journal.emit(
                        event_kind,
                        request_id=self.request_id,
                        producer="kernel_rejection_template_restrict",
                        submission_event_id=sent["event_id"],
                        on_wire=True,
                        dispatch=False,
                        **rejection_template_restrict_delivery,
                    )
                from kernel_rejection_template_restrict import (
                    read_pointer_menu_wire_receipt,
                )
                read_pointer_menu_delivery = read_pointer_menu_wire_receipt(
                    wire, runtime=restriction_runtime, request_id=self.request_id)
                if read_pointer_menu_delivery is not None:
                    ranked_event = read_pointer_menu_delivery.pop("ranked_event", None)
                    event_kind = (
                        str(ranked_event)
                        if ranked_event == "read_pointer_edit_menu_ranked"
                        else "read_pointer_edit_menu_fired"
                    )
                    self.journal.emit(
                        event_kind,
                        request_id=self.request_id,
                        producer="kernel_read_pointer",
                        submission_event_id=sent["event_id"],
                        send_time_ns=sent["timestamp_ns"],
                        on_wire=True,
                        dispatch=False,
                        **read_pointer_menu_delivery,
                    )
                from kernel_noread import wire_receipt as noread_wire_receipt
                noread_delivery = noread_wire_receipt(
                    wire, runtime=restriction_runtime, request_id=self.request_id)
                if noread_delivery is not None:
                    ranked_event = noread_delivery.pop("ranked_event", None)
                    event_kind = (
                        str(ranked_event) if ranked_event == "noread_ranked"
                        else "noread_fired"
                    )
                    self.journal.emit(
                        event_kind,
                        request_id=self.request_id,
                        producer="kernel_noread",
                        submission_event_id=sent["event_id"],
                        send_time_ns=sent["timestamp_ns"],
                        on_wire=True,
                        dispatch=False,
                        **noread_delivery,
                    )
            if restate_delivery is not None:
                self.journal.emit(
                    "restate_requirement_fired", request_id=self.request_id,
                    producer="kernel_restate_requirement", submission_event_id=sent["event_id"],
                    on_wire=True, dispatch=False, **restate_delivery,
                )
            lean_shape = getattr(self, "lean_request_shape", None)
            if isinstance(lean_shape, Mapping):
                body_sha256 = "sha256:" + hashlib.sha256(wire).hexdigest()
                shape_event_id = getattr(self, "lean_shape_event_id", None)
                self.journal.emit(
                    "lean_fired", producer="kernel_lean", request_id=self.request_id,
                    shape_event_id=shape_event_id, submission_event_id=sent["event_id"],
                    body_sha256=body_sha256, wire_transport=True, on_wire=True,
                    prefix_ratio=lean_shape.get("prefix_ratio"),
                    bytes_before=lean_shape.get("bytes_before"),
                    bytes_after=lean_shape.get("bytes_after"),
                    transport_attempt=attempt + 1, dispatch=False,
                )
                self.journal.emit(
                    "FIRED_ON_WIRE", producer="kernel_lean", mechanism="lean",
                    request_id=self.request_id, shape_event_id=shape_event_id,
                    submission_event_id=sent["event_id"], body_sha256=body_sha256,
                    wire_transport=True, on_wire=True,
                    transport_attempt=attempt + 1, dispatch=False,
                )
            stall_observation = getattr(self, "stall_observation", None)
            if isinstance(stall_observation, Mapping):
                observation = str(stall_observation.get("observation") or "")
                parsed_wire_messages = json.loads(wire).get("messages", ())
                visible = any(
                    isinstance(message, Mapping) and message.get("role") == "tool"
                    and observation in str(message.get("content") or "")
                    for message in parsed_wire_messages
                )
                if visible:
                    body_sha256 = "sha256:" + hashlib.sha256(wire).hexdigest()
                    if getattr(self, "_stall_observation_applied_body_sha256", None) != body_sha256:
                        self.journal.emit(
                            "stall_observation_applied", producer="kernel_stall_observation",
                            request_id=self.request_id, submission_event_id=sent["event_id"],
                            repeat_count=stall_observation.get("repeat_count"),
                            observation_sha256=stall_observation.get("observation_sha256"),
                            body_sha256=body_sha256, wire_transport=True, on_wire=True,
                            dispatch=False,
                        )
                        self._stall_observation_applied_body_sha256 = body_sha256
                    self.journal.emit(
                        "stall_observation_fired", producer="kernel_stall_observation",
                        request_id=self.request_id, submission_event_id=sent["event_id"],
                        repeat_count=stall_observation.get("repeat_count"),
                        observation_sha256=stall_observation.get("observation_sha256"),
                        body_sha256=body_sha256,
                        on_wire=True, wire_transport=True,
                        transport_attempt=attempt + 1, dispatch=False,
                    )
                    self.journal.emit(
                        "FIRED_ON_WIRE", producer="kernel_stall_observation",
                        mechanism="stall_observation", request_id=self.request_id,
                        submission_event_id=sent["event_id"], body_sha256=body_sha256,
                        repeat_count=stall_observation.get("repeat_count"),
                        wire_transport=True, on_wire=True,
                        transport_attempt=attempt + 1, dispatch=False,
                    )
            truth = getattr(self, "_wire_truth_check", None)
            if truth is not None and truth.verdict == "PASS":
                self.journal.emit(
                    "FIRED_ON_WIRE", producer="kernel_wire_truth_gate",
                    mechanism="wire_truth_gate", request_id=self.request_id,
                    body_sha256=truth.body_sha256, schema_digest=truth.schema_digest,
                    transport_attempt=attempt + 1, dispatch=False,
                )
            parsed_messages = json.loads(wire).get("messages", ())
            capsule = getattr(self, "read_contract_wire_capsule", None)
            if isinstance(capsule, str) and any(
                    isinstance(message, Mapping) and message.get("content") == capsule
                    for message in parsed_messages):
                self.journal.emit(
                    "read_contract_fired", producer="kernel_read_exec",
                    request_id=self.request_id, body_sha256="sha256:" + hashlib.sha256(wire).hexdigest(),
                    on_wire=True, read_contract=self.read_contract_wire_fields,
                    dispatch=False,
                )
            _emit_scope_turns_final_wire_events(
                self.journal,
                wire,
                parsed_messages,
                runtime=getattr(self, "completion_owner_runtime", None),
                request_id=self.request_id,
                submission_event_id=sent["event_id"],
            )
            census_claim = getattr(self, "callsite_census_wire_claim", None)
            if isinstance(census_claim, Mapping):
                parsed_messages = json.loads(wire).get("messages", ())
                census_present = any(
                    isinstance(message, Mapping)
                    and str(census_claim["payload"]) in str(message.get("content", ""))
                    for message in parsed_messages)
                self.journal.emit(
                    "callsite_census_wire_outcome", producer="kernel_callsite_census",
                    request_id=self.request_id,
                    status="fired" if census_present else "omitted_after_prepare",
                    body_sha256="sha256:" + hashlib.sha256(wire).hexdigest(),
                    on_wire=census_present, dispatch=False,
                )
                if census_present:
                    self.journal.emit(
                        "callsite_census_fired", producer="kernel_callsite_census",
                        request_id=self.request_id,
                        body_sha256="sha256:" + hashlib.sha256(wire).hexdigest(),
                        header_path=census_claim["header_path"],
                        placement=census_claim["placement"], on_wire=True,
                        dispatch=False,
                    )
            if first_submission_event_id is None:
                first_submission_event_id = sent["event_id"]
            truth_check = getattr(self, "_wire_truth_check", None)
            truth_context = getattr(self, "_wire_truth_context", None)
            if (getattr(truth_check, "verdict", None) == "PASS"
                    and isinstance(truth_context, Mapping)
                    and truth_context.get("require_done") is True
                    and any(
                        isinstance(tool, Mapping)
                        and isinstance(tool.get("function"), Mapping)
                        and tool["function"].get("name") == "done"
                        for tool in body.get("tools", [])
                    )):
                self.journal.emit(
                    "FIRED_ON_WIRE", producer="kernel_wire_truth_gate",
                    mechanism="done_offered_after_green",
                    request_id=self.request_id,
                    body_sha256=truth_check.body_sha256,
                    transport_attempt=attempt + 1,
                    dispatch=False,
                )
            if attempt == 0:
                # A reserved completion response is charged after the first
                # durable submission, while retries remain one logical call.
                try:
                    from kernel_pilot import ACTIVE_RUNTIME
                except ImportError:
                    ACTIVE_RUNTIME = None
                runtime = ACTIVE_RUNTIME.get() if ACTIVE_RUNTIME is not None else None
                if runtime is None:
                    runtime = getattr(self, "completion_owner_runtime", None)
                from kernel_request_kind import is_auxiliary
                if runtime is not None and not is_auxiliary(self):
                    from kernel_completion_owner import note_request_submitted
                    submitted_tool_names = _submitted_tool_names(body)
                    note_request_submitted(
                        runtime, request_id=self.request_id, step=self.step,
                        body_sha256=str(submission_fields.get("payload_hash") or ""),
                        tool_names=submitted_tool_names,
                        preserve_owner_blocks=self.qualified_request is not None,
                    )
            started = time.perf_counter_ns()
            self._assert_wire_string_hygiene(wire)
            guard_physical_send(self, getattr(self, "_wire_truth_runtime", None), body, wire)
            try:
                from kernel_bounded_http import request_bytes
                raw_response = request_bytes(
                    wire, endpoint, key, deadline=attempt_deadline,
                    socket_timeout=attempt_timeout)
            except Exception as exc:
                elapsed_ns = time.perf_counter_ns() - started
                error_type, http_status, relay_error = _transport_metadata(exc)
                from kernel_bounded_http import transport_class as _classify_transport
                server_error_body = _server_error_from(exc, key)
                transport_class = _classify_transport(http_status, server_error_body, error_type)
                matched_reason = _llama_format_failure_reason(
                    http_status, server_error_body)
                model_format_failure = matched_reason == LLAMA_FORMAT_FAILURE_REASON
                if model_format_failure:
                    transport_class = "model_format"
                liveness_lost = is_provider_liveness_timeout_error(exc)
                submission_state = getattr(exc, "submission_state", "unknown")
                if not isinstance(submission_state, str) or not submission_state:
                    submission_state = "unknown"
                usage_state = getattr(exc, "usage_state", "unknown")
                if submission_state in {"not_sent", "not_submitted"}:
                    usage_state = "not_incurred"
                transient = (not model_format_failure and transport_class != "contract"
                             and _transport_retryable(
                    exc, elapsed_ns=elapsed_ns, http_status=http_status,
                    relay_error=relay_error, error_type=error_type,
                    transport_class=transport_class, matched_reason=matched_reason))
                response_already_received = self._durable_response_received()
                request_timed_out = (
                    per_request_timeout_active
                    and not liveness_lost
                    and relay_error not in {"downstream_disconnect", "upstream_timeout"}
                    and is_provider_request_timeout_error(exc))
                if request_timed_out and not model_format_failure:
                    transient = True
                terminal_outcome = (
                    None if model_format_failure else
                    HARNESS_CONTRACT if transport_class == "contract" else HARNESS_TRANSPORT)
                if liveness_lost:
                    transport_reason = "liveness_timeout"
                elif isinstance(relay_error, str):
                    transport_reason = relay_error
                else:
                    transport_reason = error_type
                ttft_retry = bool(
                    ttft_retry_budget_on
                    and transient
                    and http_status == 504
                    and relay_error == "relay_ttft_timeout"
                )
                if ttft_retry:
                    if ttft_retry_started_at is None:
                        # Include the first failed TTFT request, but exclude earlier
                        # ordinary transport errors from the TTFT-specific window.
                        ttft_retry_started_at = attempt_started
                        assert ttft_retry_budget_limit is not None
                        ttft_retry_budget_deadline = (
                            ttft_retry_started_at + ttft_retry_budget_limit)
                        if (deadline is not None
                                and deadline <= ttft_retry_budget_deadline):
                            ttft_retry_deadline = deadline
                            ttft_retry_deadline_reason = "request_deadline"
                        else:
                            ttft_retry_deadline = ttft_retry_budget_deadline
                            ttft_retry_deadline_reason = "ttft_retry_budget_exhausted"
                else:
                    ordinary_attempt_count += 1
                ttft_retry_index = ttft_retry_count + 1 if ttft_retry else None
                cumulative_ttft_wait = (
                    max(0.0, time.monotonic() - ttft_retry_started_at)
                    if ttft_retry_started_at is not None else None
                )
                attempt_reasons.append({
                    "attempt": attempt + 1,
                    "attempt_kind": "ttft_retry" if ttft_retry else "ordinary_transport",
                    "ordinary_attempt_count": ordinary_attempt_count,
                    "reason": transport_reason,
                    "error_type": error_type,
                    "http_status": http_status,
                    "relay_error": relay_error,
                    **({"ttft_retry_index": ttft_retry_index}
                       if ttft_retry_index is not None else {}),
                })
                has_attempt = (
                    True if ttft_retry
                    else ordinary_attempt_count < ordinary_attempt_limit
                )
                retry_attempt_failed = timeout_retry_attempt == attempt
                if request_timed_out and not ttft_retry:
                    has_attempt = has_attempt and not timeout_retry_used
                elif retry_attempt_failed and not ttft_retry:
                    has_attempt = False
                delay = (
                    _ttft_retry_delay_seconds(int(ttft_retry_index))
                    if ttft_retry and ttft_retry_index is not None
                    else 0.0 if request_timed_out or relay_error == "relay_ttft_timeout"
                    else POLICY.transport_delay(ordinary_attempt_count)
                )
                retry_deadline = deadline
                if ttft_retry_deadline is not None:
                    retry_deadline = (
                        ttft_retry_deadline if retry_deadline is None
                        else min(retry_deadline, ttft_retry_deadline)
                    )
                remaining_after = (
                    None if retry_deadline is None
                    else retry_deadline - time.monotonic()
                )
                retry_scheduled = bool(
                    transient and has_attempt
                    and not response_already_received
                    and (remaining_after is None or remaining_after > delay)
                )
                if request_timed_out and retry_scheduled:
                    timeout_retry_used = True
                    timeout_retry_attempt = attempt + 1
                if request_timed_out and journal_timeouts:
                    journal_provider_request_timeout(
                        self.journal, request_id=self.request_id, step=self.step,
                        timeout_seconds=attempt_wall_timeout, attempt=attempt + 1,
                        endpoint=endpoint, retry_scheduled=retry_scheduled)
                if retry_scheduled:
                    retry_blocked_reason = None
                elif request_timed_out and timeout_retry_used:
                    retry_blocked_reason = "timeout_retry_exhausted"
                elif retry_attempt_failed and not ttft_retry:
                    retry_blocked_reason = "timeout_retry_failed"
                elif ttft_retry:
                    retry_blocked_reason = (
                        ttft_retry_deadline_reason or "ttft_retry_budget_exhausted")
                elif model_format_failure:
                    retry_blocked_reason = "model_format_outcome"
                elif response_already_received:
                    retry_blocked_reason = "response_already_received"
                elif not transient:
                    retry_blocked_reason = (
                        "ambiguous_submission" if submission_state == "unknown"
                        else "not_retryable")
                elif not has_attempt:
                    retry_blocked_reason = "attempt_limit"
                else:
                    retry_blocked_reason = "deadline"
                host_fact_fields: dict[str, object] = {}
                if http_status == 502 and relay_error == "upstream_timeout":
                    host_fact_fields.update({
                        "producer": "kernel_measurement",
                        "fact_owner": "HOST",
                        "dispatch": False,
                    })
                if matched_reason is not None:
                    host_fact_fields.update({
                        "producer": "kernel_measurement",
                        "fact_owner": "HOST",
                        "dispatch": False,
                    })
                error = self.journal.emit(
                    "transport_error", request_id=self.request_id,
                    transport_attempt_id=sent["event_id"], step=self.step,
                    error_type=error_type, http_status=http_status,
                    relay_error=relay_error, failure_origin="harness",
                    matched_reason=matched_reason,
                    transport_class=transport_class,
                    submission_state=submission_state,
                    usage_state=usage_state,
                    server_error=_parsed_server_error(server_error_body),
                    server_error_body=server_error_body,
                    request_bytes=len(wire),
                    outcome=terminal_outcome if not retry_scheduled else None,
                    retryable=transient, retry_scheduled=retry_scheduled,
                    retry_blocked_reason=retry_blocked_reason,
                    transport_attempt=attempt + 1, retry_index=attempt,
                    retry_budget=("ttft" if ttft_retry else "ordinary_transport"),
                    ordinary_attempt_count=ordinary_attempt_count,
                    ordinary_attempt_limit=ordinary_attempt_limit,
                    ttft_retry_index=ttft_retry_index,
                    ttft_retry_budget_seconds=ttft_retry_budget_limit,
                    cumulative_ttft_wait_seconds=cumulative_ttft_wait,
                    prompt_tokens=admission.prompt_tokens if admission is not None else None,
                    measured_prefill_tokens_per_second=(
                        admission.prefill_tokens_per_second if admission is not None else None),
                    derived_deadline_seconds=(
                        admission.deadline_seconds if admission is not None else None),
                    admission=admission.mapping() if admission is not None else None,
                    elapsed_ns=elapsed_ns,
                    transport_reason=transport_reason,
                    attempt_reasons=list(attempt_reasons),
                    **host_fact_fields)
                self.transport_errors.append(error)
                if retry_scheduled:
                    if ttft_retry:
                        ttft_retry_count += 1
                    self.journal.emit(
                        "provider_transport_retry",
                        producer="kernel_measurement",
                        fact_owner="HOST",
                        request_id=self.request_id,
                        step=self.step,
                        transport_attempt=attempt + 1,
                        next_transport_attempt=attempt + 2,
                        reason=transport_reason,
                        error_type=error_type,
                        http_status=http_status,
                        relay_error=relay_error,
                        retry_delay_seconds=delay,
                        request_bytes=len(wire),
                        payload_hash="sha256:" + hashlib.sha256(wire).hexdigest(),
                        retry_budget=("ttft" if ttft_retry else "ordinary_transport"),
                        ordinary_attempt_count=ordinary_attempt_count,
                        ordinary_attempt_limit=ordinary_attempt_limit,
                        ttft_retry_index=ttft_retry_index,
                        ttft_retry_count=ttft_retry_count,
                        ttft_retry_budget_seconds=ttft_retry_budget_limit,
                        cumulative_ttft_wait_seconds=cumulative_ttft_wait,
                        attempt_reasons=list(attempt_reasons),
                        failure_origin="harness",
                        retry_scheduled=True,
                        dispatch=False,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                if model_format_failure:
                    self._raise_malformed_provider_response(
                        server_error_body or b"",
                        "llama.cpp returned a model-format failure",
                        transport_attempt_id=sent["event_id"])
                raise _transport_error_class(relay_error)(
                    attempts=attempt + 1, error_type=error_type,
                    http_status=http_status, relay_error=relay_error,
                    transport_class=transport_class,
                    server_error=server_error_body,
                    attempt_reasons=attempt_reasons) from exc
            try:
                response = json.loads(raw_response)
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
                self._raise_malformed_provider_response(
                    raw_response,
                    f"provider response is not valid JSON: {type(exc).__name__}",
                    transport_attempt_id=sent["event_id"],
                )
            if not isinstance(response, Mapping):
                self._raise_malformed_provider_response(
                    raw_response,
                    "provider response JSON must be an object",
                    transport_attempt_id=sent["event_id"],
                )
            choices = response.get("choices")
            if (not isinstance(choices, Sequence)
                    or isinstance(choices, (str, bytes, bytearray))
                    or not choices
                    or not isinstance(choices[0], Mapping)):
                self._raise_malformed_provider_response(
                    raw_response,
                    "provider response must contain a non-empty choices object list",
                    transport_attempt_id=sent["event_id"],
                )
            return response, sent, started

        raise AssertionError("transport retry loop completed without a result")

    def _adopt_reprepared(self, replacement: object, runtime: object, *,
                          deadline: float | None,
                          expected_owner_request_id: str | None = None,
                          boundary_recovery: bool = False,
                          source_replan: bool = False,
                          publish_event: bool = True) -> None:
        """Replace an unsent transaction as a whole, preserving caller aliases."""
        if (not isinstance(replacement, RequestCapture) or replacement is self
                or (getattr(self, "_wire_truth_rebuild_used", False)
                    and (not boundary_recovery
                         or getattr(self, "_boundary_recovery_attempted", False)))
                or getattr(self, "_transport_submission_started", False)
                or self.transport_attempt_id is not None or self.response_event_id is not None):
            raise RuntimeError("wire recovery requires an unsent original capture")
        snapshot_matches = replacement.snapshot_id == self.snapshot_id or (
            source_replan and replacement.snapshot_id == getattr(runtime, "snapshot_id", None)
        )
        if (replacement.journal is not self.journal or replacement.step != self.step
                or not snapshot_matches
                or replacement.parent_meter is not self.parent_meter
                or replacement.endpoint_profile is not self.endpoint_profile
                or replacement.request_id == self.request_id
                or replacement._send_generation != 0
                or getattr(replacement, "_transport_submission_started", False)
                or replacement.transport_attempt_id is not None
                or replacement.response_event_id is not None):
            raise RuntimeError("wire recovery requires a fresh capture in the same transaction scope")
        contract = replacement.request_action_contract
        if isinstance(contract, Mapping) and contract.get("request_id") != replacement.request_id:
            raise RuntimeError("wire recovery owner contract has a different request identity")
        for owner in (runtime, getattr(runtime, "owner", None)):
            current_id = getattr(owner, "current_request_id", None)
            expected_id = expected_owner_request_id or replacement.request_id
            if current_id is not None and current_id != expected_id:
                raise RuntimeError("wire recovery runtime has a different request identity")
        if (getattr(self, "completion_owner_runtime", None) is not None
                and getattr(replacement, "completion_owner_runtime", None) is not runtime):
            raise RuntimeError("wire recovery capture belongs to a different runtime")
        if deadline is not None and (not math.isfinite(deadline) or deadline <= time.monotonic()):
            raise TimeoutError("request deadline expired before wire recovery")
        old_id = self.request_id
        event_kind = (
            "boundary_contract_recovery_reprepared"
            if boundary_recovery else "request_reprepared"
        )
        if publish_event:
            self.journal.emit(
                event_kind, producer="kernel_measurement", request_id=old_id,
                replacement_request_id=replacement.request_id, step=self.step,
                reason=("boundary_contract_recovery" if boundary_recovery
                        else "wire_truth_bounded_recovery"),
                submitted=False, dispatch=False)
        replacement._wire_truth_rebuild_used = True
        # One dictionary assignment adopts every prepared owner/evidence/budget
        # field together. Existing caller/runtime aliases observe the real new
        # request and response, including when its final guard or transport fails.
        from kernel_cache_layout_candidate import pending_publication
        replacement.__dict__.update(pending_publication(replacement, runtime))
        self.__dict__ = replacement.__dict__
        runtime.current_request_id = self.request_id
        runtime_owner = getattr(runtime, "owner", None)
        if runtime_owner is not None:
            runtime_owner.current_request_id = self.request_id
        if getattr(runtime, "_completion_owner_pending_request_id", None) == old_id:
            runtime._completion_owner_pending_request_id = self.request_id
        for name in ("hybrid_crossing_request_id", "long_horizon_hybrid_crossing_request_id"):
            if getattr(self, name, None) == old_id:
                setattr(self, name, self.request_id)
            if getattr(runtime, name, None) == old_id:
                setattr(runtime, name, self.request_id)
        if getattr(runtime, "_active_capture", None) in (self, replacement):
            runtime._active_capture = self

    def send_with_frozen_timeout_retry(
            self, body: dict[str, object], endpoint: str, key: str, *,
            deadline: float | None = None,
            side_call_contract: object = None, required: bool = False,
            preview: bool = False) -> dict:
        """Prepare once, then let the bounded transport layer retry its frozen wire."""
        from kernel_boundary_recovery import BoundaryRecoveryStop

        if not timeout_retry_frozen_enabled():
            return self.send(
                body, endpoint, key, deadline=deadline,
                side_call_contract=side_call_contract, required=required,
                preview=preview, transport_attempt_limit=1, journal_timeouts=False,
            )
        if (getattr(self, "_transport_submission_started", False)
                and getattr(self, "response_event_id", None) is None):
            raise BoundaryRecoveryStop(
                "frozen transport retry requires an unsubmitted capture entrypoint")
        return self.send(
            body, endpoint, key, deadline=deadline,
            side_call_contract=side_call_contract, required=required,
            preview=preview, journal_timeouts=True, _frozen_retry_only=True,
        )

    def _emit_host_span_profile(self, body: Mapping[str, object], wire: bytes) -> None:
        recorder = getattr(self.journal, "host_spans", None)
        if not isinstance(recorder, HostSpanRecorder):
            return
        profile = None
        try:
            profile = recorder.finish_request_profile(self.request_id)
            if profile is None:
                return
            manifest = build_host_span_final_wire_manifest(self, body, wire)
            span_tree, span_tree_dropped = bounded_host_span_tree(recorder.trace_snapshot(
                _HOST_PROFILE_SPAN_NAMES, request_id=profile.request_id))
            self.journal.emit(
                "host_span_profile",
                event_schema="stateframe-host-span-profile/1",
                request_id=profile.request_id,
                step=profile.step if profile.step is not None else self.step,
                spans=_host_span_profile_spans(profile),
                span_tree=span_tree,
                span_tree_dropped=span_tree_dropped,
                wall_ms=round(max(0, time.perf_counter_ns() - profile.started_ns) / 1_000_000, 6),
                final_wire_manifest=manifest,
                dispatch=False,
            )
        except Exception as error:
            # Profiling is advisory and must not block or mutate a provider request, but a
            # failure must be visible: live benchmark showed 0 profiles while offline replays had one
            # per request. Journal a typed, sanitized failure and carry on.
            try:
                from kernel_boundary_recovery import safe_error_class
                self.journal.emit(
                    "host_span_profile_failed",
                    request_id=self.request_id,
                    step=self.step,
                    error_class=safe_error_class(error),
                    error_type=type(error).__name__[:80],
                    dispatch=False,
                )
            except Exception:
                pass
            return
        finally:
            try:
                recorder.discard_request_profile(self.request_id)
            except Exception:
                pass

    @cleanup_request_host_span_profile
    def send(self, body: dict[str, object], endpoint: str, key: str, *,
             deadline: float | None = None,
             side_call_contract: object = None, required: bool = False,
             preview: bool = False,
             transport_attempt_limit: int | None = None,
             journal_timeouts: bool = True,
             _frozen_retry_only: bool = False) -> dict:
        """Run one shared prepare/validate/send boundary with bounded recovery."""
        from kernel_pilot import ACTIVE_RUNTIME
        from kernel_request_kind import classify
        request_runtime = ACTIVE_RUNTIME.get() or getattr(self, "completion_owner_runtime", None)
        if not preview:
            classify(self, request_runtime, body, side_call_contract=side_call_contract)
        if not preview and getattr(self, "_release_prefix_boundary", None) is not None:
            from kernel_pilot import ACTIVE_RUNTIME
            from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
            from kernel_release_host_prefix import qualify_release_host_prefix
            prefix_runtime = ACTIVE_RUNTIME.get() or getattr(
                self, "completion_owner_runtime", None,
            )
            try:
                body = qualify_release_host_prefix(
                    self, prefix_runtime, body, deadline=deadline,
                )
            except FrameAllocationReplanNeeded as error:
                from kernel_request_reprepare import SourceRevisionReplanRefused
                # The enclosing wrapper changed a prepared prefix. Re-preparing
                # it would grant new authority to an unverified host mutation.
                raise SourceRevisionReplanRefused(
                    "release_prefix_replan_refused") from error
        # Frozen timeout mode still enters this request once; only retries inside
        # _send_transport_with_retries reuse the finalized bytes without hooks.
        if isinstance(body, dict):
            try:
                from kernel_tool_reply import enabled as tool_reply_constructor_enabled
                n3_tool_replies_on = tool_reply_constructor_enabled()
            except Exception:
                n3_tool_replies_on = False
            if not n3_tool_replies_on:
                from kernel_done_reply import fill_empty_done_replies
                filled = fill_empty_done_replies(body.get("messages"))
                if filled:
                    self.journal.emit("done_reply_filled", producer="kernel_done_reply", count=filled,
                                      request_id=self.request_id, step=self.step, dispatch=False)
            # REMAINING_SITES (default OFF): refresh literal-site facts at the shared send boundary.
            try:
                from kernel_remaining_sites import maybe_append_remaining_sites
                from kernel_pilot import ACTIVE_RUNTIME
                maybe_append_remaining_sites(body.get("messages"), ACTIVE_RUNTIME.get())
            except Exception:
                pass
        _reconcile_frame_journal_gate(self, body)
        host_spans = getattr(self.journal, "host_spans", None)
        if isinstance(host_spans, HostSpanRecorder):
            try:
                host_spans.begin_request_profile(self.request_id, self.step)
            except Exception:
                pass
        from kernel_boundary_recovery import enabled as recovery_enabled

        recovery_on = recovery_enabled()
        try:
            from kernel_pilot import ACTIVE_RUNTIME
            runtime = ACTIVE_RUNTIME.get()
        except (ImportError, LookupError):
            runtime = getattr(self, "completion_owner_runtime", None)
        state = _BoundaryRecoveryContext(
            str(self.request_id), "send_entry", runtime, copy.deepcopy(body),
            endpoint, key, deadline, side_call_contract, required,
            transport_attempt_limit, journal_timeouts,
        )
        try:
            if getattr(self, "_wire_truth_rebuild_used", False):
                # An adopted capture cannot start a new send transaction.
                _stop_pre_dispatch_contract(
                    self, event="request_preparation_stopped",
                    code="replacement_capture_reentry",
                    detail="adopted replacement capture cannot re-enter send",
                )
            from kernel_request_kind import isolated_auxiliary_runtime
            with isolated_auxiliary_runtime(self):
                result = self._send_once(
                    body, endpoint, key, deadline=deadline,
                    side_call_contract=side_call_contract, required=required,
                    preview=preview, transport_attempt_limit=transport_attempt_limit,
                    journal_timeouts=journal_timeouts, _boundary_state=state,
                    _frozen_retry_only=_frozen_retry_only,
                )
            from kernel_request_kind import validate_auxiliary_response
            validate_auxiliary_response(self, result)
            return result
        except Exception as error:
            return self._handle_boundary_contract_error(error, state, recovery_on)
        finally:
            # Caller inputs are needed only until the pre-dispatch recovery
            # decision. Do not retain a second full prompt after this send.
            self._source_replan_inputs = None

    def _handle_boundary_contract_error(
            self, error: Exception, state: _BoundaryRecoveryContext,
            recovery_on: bool) -> dict:
        """Keep the shared send seam fail-closed after typed boundary failures."""
        from kernel_boundary_recovery import BoundaryRecoveryStop
        from kernel_frame_optional_allocation import FrameAllocationReplanNeeded

        if isinstance(error, FrameAllocationReplanNeeded):
            return self._recover_source_revision_replan(error, state)

        # Re-entry is the receipt guard itself, not a new final-wire failure.
        # Preserve its exact typed refusal so the request-receipt contract remains
        # observable and no replacement work can run again.
        if getattr(error, "stop_code", None) == "replacement_capture_reentry":
            raise error
        # A failed owner replacement has already consumed the one preparation
        # attempt. Do not route its producer failure through a second, parallel
        # replacement path; boundary recovery remains available after a
        # replacement was successfully adopted and then failed strict validation.
        if (state.wire_truth_rebuild_attempted
                and not getattr(self, "_wire_truth_rebuild_used", False)):
            stop = BoundaryRecoveryStop(
                "owner replacement preparation failed before adoption")
            for name, value in (
                ("boundary_recovery_stop", True),
                ("boundary_recovery_stop_code", stop.stop_code),
                ("boundary_recovery_stop_detail", str(stop)),
                ("boundary_recovery_stop_error", stop),
                ("harness_contract", True),
                ("failure_origin", "harness"),
                ("outcome", HARNESS_CONTRACT),
            ):
                try:
                    setattr(error, name, value)
                except (AttributeError, TypeError):
                    pass
            raise error
        from kernel_request_kind import PRIMARY, is_auxiliary
        if (getattr(self, "request_kind", None) == PRIMARY
                or is_auxiliary(self)
                or self._framefit5_pending_qualification or self.qualified_request is not None):
            # Complete-envelope qualification is terminal on host faults. A
            # replacement cannot turn a rejected corpus request into success.
            raise error
        if not recovery_on or not _request_boundary_contract_error(error):
            raise error
        if (getattr(self, "_wire_truth_rebuild_used", False)
                and not state.wire_truth_rebuild_attempted):
            raise BoundaryRecoveryStop(
                "request boundary failed after prior adoption") from error
        if state.recovery_attempted:
            raise BoundaryRecoveryStop(
                "request boundary recovery was already attempted") from error
        return self._recover_boundary_contract(error, state)

    def _recover_source_revision_replan(
            self, error: Exception, state: _BoundaryRecoveryContext) -> dict:
        """Re-run the normal preparation once while this request is unsent."""
        from kernel_v46_mechanism_wiring import sanitized_error_message
        from kernel_request_reprepare import (
            SourceRevisionReplanRefused, source_replan_preparation,
        )
        from kernel_request_kind import require_adapter
        from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
        from kernel_release_host_prefix import (
            bind_release_prefix_boundary, qualify_release_host_prefix,
        )

        runtime = state.runtime
        inputs = getattr(self, "_source_replan_inputs", None)
        if getattr(self, "_source_replan_attempted", False):
            raise SourceRevisionReplanRefused(
                "source_revision_replan_exhausted") from error
        if (runtime is None or state.side_call_contract is not None
                or not isinstance(inputs, Mapping)
                or state.submitted or getattr(self, "_transport_submission_started", False)
                or self.transport_attempt_id is not None or self.response_event_id is not None):
            raise SourceRevisionReplanRefused("source_revision_replan_unavailable") from error
        self._source_replan_attempted = True
        original_id = self.request_id
        self.journal.emit(
            "source_revision_replan_started", producer="kernel_measurement",
            request_id=original_id, step=self.step, submitted=False,
            error_detail=sanitized_error_message(error), dispatch=False,
        )
        try:
            # Refresh the same workspace view used by replacement validation.
            # Normal prepare builds a new closure, optional inputs, and owner
            # contract; none of the old source snapshots are promoted.
            messages = copy.deepcopy(inputs["messages"])
            uncapped = (messages if inputs["messages"] is inputs["uncapped"]
                        else copy.deepcopy(inputs["uncapped"]))
            with source_replan_preparation(runtime, self):
                prepared = runtime.prepare(
                    messages,
                    uncapped=uncapped,
                    tools=copy.deepcopy(inputs["tools"]),
                    provider_request=inputs["provider_request"],
                    request_host_text_preparer=inputs["request_host_text_preparer"],
                )
            replacement = prepared.capture
            from kernel_read_reuse_visibility import retire_changed_revision_candidates
            retired_reads = retire_changed_revision_candidates(replacement, runtime)
            if retired_reads:
                self.journal.emit(
                    "source_revision_read_pins_retired", producer="kernel_measurement",
                    request_id=original_id, step=self.step, count=retired_reads,
                    submitted=False, dispatch=False,
                )
            body = copy.deepcopy(state.authoritative_body)
            body["messages"] = list(prepared.messages)
            body["tools"] = list(prepared.tools or ())
            adapter = getattr(self, "_qualified_adapter", None)
            if adapter is not None:
                require_adapter(
                    replacement, runtime, body["messages"],
                    gate=self._qualified_request_gate,
                    host_preparer=self._qualified_host_text_preparer,
                    adapter=adapter,
                )
            if getattr(self, "_release_prefix_boundary", None) is not None:
                bind_release_prefix_boundary(replacement, body["messages"], body["tools"])
                qualify_release_host_prefix(replacement, runtime, body, deadline=state.deadline)
            self._adopt_reprepared(
                replacement, runtime, deadline=state.deadline,
                source_replan=True, publish_event=False,
            )
            self._source_replan_attempted = True
        except Exception as preparation_error:
            self.journal.emit(
                "source_revision_replan_refused", producer="kernel_measurement",
                request_id=original_id, step=self.step, reason="preparation_failed",
                submitted=False, dispatch=False,
            )
            raise SourceRevisionReplanRefused(
                "source_revision_replan_exhausted"
                if isinstance(preparation_error, FrameAllocationReplanNeeded)
                else "source_revision_replan_failed") from preparation_error
        self.journal.emit(
            "source_revision_reprepared", producer="kernel_measurement",
            request_id=original_id, replacement_request_id=self.request_id,
            step=self.step, submitted=False, dispatch=False,
        )
        state.phase = "source_revision_reprepared"
        state.preparation = None
        try:
            return self._send_once(
                body, state.endpoint, state.key, deadline=state.deadline,
                side_call_contract=None, required=state.required, preview=False,
                transport_attempt_limit=state.transport_attempt_limit,
                journal_timeouts=state.journal_timeouts,
                _boundary_state=state, _recovery_prepared=True,
            )
        except FrameAllocationReplanNeeded as repeated:
            self.journal.emit(
                "source_revision_replan_refused", producer="kernel_measurement",
                request_id=original_id, step=self.step, reason="revision_changed_again",
                error_detail=sanitized_error_message(repeated),
                submitted=False, dispatch=False,
            )
            raise SourceRevisionReplanRefused(
                "source_revision_replan_exhausted") from repeated

    def _recover_boundary_contract(
            self, error: BaseException, state: _BoundaryRecoveryContext) -> dict:
        from kernel_boundary_recovery import (
            BoundaryRecoveryStop, digest_bytes, error_fields, reserve,
            safe_error_class, safe_error_detail, safe_failure_codes,
            safe_proof_failure_kinds,
        )

        attempted_body = (
            state.repair_body or state.transport_body or state.authoritative_body
        )
        try:
            attempted_digest = digest_bytes(attempted_body)
        except (TypeError, ValueError):
            attempted_digest = None
        fields = error_fields(error, phase=state.phase)
        original_failure = (
            error_fields(state.wire_truth_error, phase="wire_truth_validation")
            if state.wire_truth_error is not None else None
        )
        allowed, cap_reason = reserve(self.journal, state.original_request_id)
        self.journal.emit(
            "boundary_contract_recovery_started",
            producer="kernel_measurement", request_id=state.original_request_id,
            step=self.step, invariant=fields["invariant"],
            error_class=fields["error_class"], code=fields["code"],
            failures=fields["failures"], phase=fields["phase"],
            original_failure=original_failure, repair_failure=fields,
            wire_truth_rebuild_attempted=state.wire_truth_rebuild_attempted,
            body_digest=attempted_digest, recovery_reserved=allowed,
            recovery_limit_reason=None if allowed else cap_reason, dispatch=False,
        )
        state.recovery_attempted = True

        def finish(
                outcome: str, body_digest: str | None, reason: str | None = None,
                error: BaseException | None = None,
        ) -> None:
            fields: dict[str, object] = {}
            if outcome == "failed":
                failure_error = error or BoundaryRecoveryStop(
                    reason or "request boundary recovery failed",
                )
                error_failures = getattr(failure_error, "failures", ())
                proof_failures = getattr(failure_error, "proof_failures", None)
                if not proof_failures:
                    proof_failures = getattr(
                        self, "request_source_final_delivery", None,
                    )
                fields = {
                    "error_class": safe_error_class(failure_error),
                    "error_detail": safe_error_detail(failure_error),
                    "code": safe_failure_codes((
                        getattr(
                            failure_error, "stop_code",
                            getattr(failure_error, "code", "unclassified_failure"),
                        ),
                    ))[0],
                    "failures": safe_failure_codes(
                        error_failures, fallback=(type(failure_error).__name__,),
                    ),
                    "proof_failure_kinds": safe_proof_failure_kinds(proof_failures),
                }
            self.journal.emit(
                "boundary_contract_recovery_finished",
                producer="kernel_measurement", request_id=state.original_request_id,
                replacement_request_id=getattr(self, "request_id", None),
                step=self.step, outcome=outcome, body_digest=body_digest,
                failure_reason=(
                    safe_failure_codes((reason,))[0] if reason is not None else None
                ),
                dispatch=False, **fields,
            )

        if not allowed:
            finish("failed", attempted_digest, cap_reason, error=error)
            raise BoundaryRecoveryStop(cap_reason) from error
        if state.submitted or getattr(self, "_transport_submission_started", False):
            finish("failed", attempted_digest, "request_already_submitted_or_adopted", error=error)
            # A strict retry check may fail after the first physical submission.
            # Preserve its original typed stop and frozen-byte retry semantics;
            # preparing another body here would violate transport authority.
            raise BoundaryRecoveryStop(
                f"request boundary failed after submission: {error}") from error
        if (getattr(self, "_wire_truth_rebuild_used", False)
                and not state.wire_truth_rebuild_attempted):
            finish("failed", attempted_digest, "request_already_submitted_or_adopted", error=error)
            raise BoundaryRecoveryStop(
                "request boundary failed after prior adoption") from error
        if state.runtime is None or state.side_call_contract is not None:
            finish("failed", attempted_digest, "owner_recovery_context_unavailable", error=error)
            # With no owner context there is no authoritative state from which
            # to rebuild. Preserve its detail while returning a normal boundary stop.
            raise BoundaryRecoveryStop(
                f"owner-authorized request recovery context is unavailable: {error}") from error

        try:
            preparation = state.preparation
            if preparation is None or state.wire_truth_rebuild_attempted:
                # The in-engine repair adopted a new request identity and owner
                # preparation. Snapshot that current authority instead of relabeling
                # the stale original snapshot with the replacement request ID.
                from kernel_request_reprepare import PreparationInput
                preparation = PreparationInput.capture(
                    self, state.runtime, attempted_body)
            from kernel_request_reprepare import prepare_replacement
            prepared = prepare_replacement(
                state.runtime, self, preparation, attempted_body,
                deadline=state.deadline, boundary_recovery=True,
                boundary_recovery_body=attempted_body,
            )
            rebuilt_body = copy.deepcopy(attempted_body)
            rebuilt_body["messages"] = list(prepared.messages)
            rebuilt_body["tools"] = copy.deepcopy(list(prepared.tools or ()))
            # prepare_replacement adopts a new request ID and settles its
            # request-local source override. Snapshot the runtime source plan for
            # preparation validation; qualification below separately honors the
            # active capture's own request-local override.
            from kernel_request_reprepare import PreparationInput
            preparation = PreparationInput.capture(
                self, state.runtime, rebuilt_body,
            )
            state.preparation = preparation
            state.phase = "recovery_same_validation"
            state.body_digest = None
            result = self._send_once(
                rebuilt_body, state.endpoint, state.key, deadline=state.deadline,
                side_call_contract=None, required=state.required, preview=False,
                transport_attempt_limit=state.transport_attempt_limit,
                journal_timeouts=state.journal_timeouts,
                _boundary_state=state, _recovery_prepared=True,
            )
        except Exception as recovery_error:
            body_digest = state.body_digest or attempted_digest
            if _request_boundary_contract_error(recovery_error) or not state.submitted:
                finish("failed", body_digest, type(recovery_error).__name__, error=recovery_error)
                stop = BoundaryRecoveryStop(
                    "rebuilt request failed strict boundary validation")
                stop.__cause__ = recovery_error
                # Keep the original typed repair failure (and its producer cause)
                # as the outward exception while directing the worker through its
                # BoundaryRecoveryStop candidate-export terminal.
                for name, value in (
                    ("boundary_recovery_stop", True),
                    ("boundary_recovery_stop_code", stop.stop_code),
                    ("boundary_recovery_stop_detail", str(stop)),
                    ("boundary_recovery_stop_error", stop),
                    ("harness_contract", True),
                    ("failure_origin", "harness"),
                    ("outcome", HARNESS_CONTRACT),
                ):
                    try:
                        setattr(error, name, value)
                    except (AttributeError, TypeError):
                        pass
                raise error
            finish("recovered", body_digest, type(recovery_error).__name__)
            raise
        finish("recovered", state.body_digest or attempted_digest)
        return result

    @profile_request_capture
    def _send_once(self, body: dict[str, object], endpoint: str, key: str, *,
             deadline: float | None = None,
             side_call_contract: object = None, required: bool = False,
             preview: bool = False,
             transport_attempt_limit: int | None = None,
              journal_timeouts: bool = True,
              _boundary_state: _BoundaryRecoveryContext | None = None,
              _recovery_prepared: bool = False,
              _frozen_retry_only: bool = False) -> dict:
        """Capture the exact bytes submitted, every retry and the full response."""
        if preview:
            if self.parent_meter is not None:
                self.parent_meter.admit_provider(
                    reason="provider preview", journal=self.journal, preview=True)
            return {"preview": True, "charged": False, "sent": False}
        from kernel_request_reprepare import RequestPreparationContractError
        if _boundary_state is not None:
            _boundary_state.phase = "request_sanitization"
            _boundary_state.endpoint = endpoint
            _boundary_state.key = key
            _boundary_state.deadline = deadline
        if transport_attempt_limit is not None and (
                type(transport_attempt_limit) is not int or transport_attempt_limit < 1):
            raise ValueError("transport_attempt_limit must be a positive integer")
        per_request_timeout = per_request_timeout_seconds()
        from sampler_config import apply_to_request
        self._send_generation += 1
        permit_prefix = f"{self.request_id}:send:{self._send_generation}"
        entry_body = copy.deepcopy(body)
        framefit5_assembler_messages: bytes | None = None
        framefit5_request = False
        from kernel_wire_string_hygiene import record_replacements, sanitize_request_body

        entry_body, replacements = sanitize_request_body(entry_body)
        record_replacements(
            self.journal, replacements, request_id=self.request_id, step=self.step)
        body = copy.deepcopy(entry_body)
        from kernel_lean import (
            enabled as _lean_enabled,
            finalize_wire_body as _finalize_lean_wire_body,
            reconcile_tool_call_results as _reconcile_tool_call_results,
            sanitize_tool_references as _sanitize_tool_references,
            validate_deferred_reads as _validate_deferred_reads,
        )
        lean_on = _lean_enabled()
        from kernel_stateframe_m3 import m3_working_context_enabled
        m3_on = m3_working_context_enabled()
        from kernel_tool_reply import (
            compose_tool_replies as _compose_tool_replies,
            enabled as _tool_reply_enabled,
            record_and_strip_receipts as _record_tool_reply_receipts,
        )
        tool_reply_on = _tool_reply_enabled()

        def _apply_wire_stability(candidate: dict[str, object]) -> dict[str, object]:
            from kernel_wire_stability import project as _project_wire_stability
            stability = _project_wire_stability(candidate)
            if not stability.changed:
                return candidate
            self.journal.emit(
                "wire_stability_applied",
                producer="kernel_wire_stability",
                request_id=self.request_id,
                step=self.step,
                volatile_marker_added=stability.volatile_marker_added,
                disk_capacity_lines_removed=stability.disk_capacity_lines_removed,
                dispatch=False,
            )
            return stability.body

        def _qualify_final_allocation(candidate: dict[str, object]) -> dict[str, object]:
            nonlocal preparation
            from kernel_final_wire_allocation import qualify_final_wire_allocation
            previous = self.frame_request_allocation
            candidate = qualify_final_wire_allocation(
                self, wire_truth_runtime, candidate, preparation, deadline=deadline,
            )
            if self.frame_request_allocation is not previous:
                from kernel_request_reprepare import PreparationInput
                preparation = PreparationInput.capture(self, wire_truth_runtime, candidate)
                self._request_preparation_input = preparation
                if _boundary_state is not None:
                    _boundary_state.preparation = preparation
            return candidate

        def _guard_final_wire(candidate: dict[str, object]) -> dict[str, object]:
            raw_messages = candidate.get("messages")
            raw_tools = candidate.get("tools")
            if not isinstance(raw_messages, list):
                return candidate
            tools_list = raw_tools if isinstance(raw_tools, list) else []
            sanitized, rewrite_count = _sanitize_tool_references(raw_messages, tools_list)
            guarded = copy.deepcopy(candidate)
            guarded["messages"] = sanitized
            try:
                _validate_deferred_reads(sanitized, tools_list)
            except ValueError as error:
                raise RequestPreparationContractError(str(error)) from error
            guarded, dropped = _finalize_lean_wire_body(guarded, lean=lean_on)
            if isinstance(guarded.get("messages"), list):
                from kernel_read_once import ensure_wire_note
                guarded["messages"] = ensure_wire_note(
                    wire_truth_runtime, guarded["messages"],
                )
                if lean_on or m3_on:
                    reconcile_records = getattr(self.journal, "tool_reconciliation_records", None)
                    cached_records = getattr(self.journal, "cached_records", None)
                    settled_records = getattr(self.journal, "settled_action_records", None)
                    record_reader = (
                        reconcile_records if callable(reconcile_records)
                        else cached_records if callable(cached_records)
                        else settled_records if callable(settled_records)
                        else None
                    )
                    journal_records = record_reader() if record_reader is not None else ()
                    prepared_results = getattr(self, "lean_prepared_tool_results", ())
                    if not isinstance(prepared_results, (list, tuple)):
                        prepared_results = ()
                    try:
                        reconciled_messages, recovered_ids = _reconcile_tool_call_results(
                            guarded["messages"], journal_records,
                            settled_results=prepared_results,
                        )
                    except ValueError as error:
                        if m3_on:
                            call_count = sum(
                                len(message.get("tool_calls", ()))
                                for message in guarded["messages"]
                                if isinstance(message, Mapping)
                                and message.get("role") == "assistant"
                                and isinstance(message.get("tool_calls"), list)
                            )
                            self.journal.emit(
                                "stateframe_m3_tool_pairing_rejected",
                                producer="kernel_stateframe_m3",
                                request_id=self.request_id, step=self.step,
                                tool_call_count=call_count,
                                error_type=type(error).__name__, dispatch=False,
                            )
                        raise RequestPreparationContractError(str(error)) from error
                    guarded["messages"] = reconciled_messages
                    if recovered_ids:
                        if lean_on:
                            self.journal.emit(
                                "lean_tool_results_reconciled", producer="kernel_lean",
                                boundary="final_wire", request_id=self.request_id,
                                step=self.step, tool_call_ids=list(recovered_ids),
                                dispatch=False,
                            )
                        if m3_on:
                            self.journal.emit(
                                "stateframe_m3_tool_results_reconciled",
                                producer="kernel_stateframe_m3",
                                boundary="final_wire", request_id=self.request_id,
                                step=self.step, recovered_count=len(recovered_ids),
                                dispatch=False,
                            )
                if tool_reply_on:
                    try:
                        guarded["messages"] = _compose_tool_replies(
                            guarded["messages"], self.journal,
                            request_id=self.request_id, step=self.step,
                            defer_receipts=True,
                        )
                    except Exception as error:
                        try:
                            self.journal.emit(
                                "tool_reply_composition_error",
                                producer="kernel_tool_reply",
                                request_id=self.request_id, step=self.step,
                                error_type=type(error).__name__, dispatch=False,
                            )
                        except Exception:
                            pass
            if rewrite_count or dropped:
                self.journal.emit(
                    "wire_host_guard_applied", producer="kernel_lean",
                    request_id=self.request_id, step=self.step, lean=lean_on,
                    tool_reference_rewrites=rewrite_count,
                    dropped_block_kinds=list(dropped), dispatch=False,
                )
            guarded = _apply_wire_stability(guarded)
            guarded_messages = guarded.get("messages")
            from kernel_request_budget import _live_framefit5_product_request
            if (isinstance(guarded_messages, list)
                    and _live_framefit5_product_request(wire_truth_runtime, guarded_messages)):
                from kernel_stateframe_framefit5 import (
                    assemble_framefit5_request,
                    has_framefit5_state,
                )
                if has_framefit5_state(guarded_messages):
                    guarded = _qualify_final_allocation(guarded)
                    guarded_messages = guarded["messages"]
                    from kernel_request_reprepare import (
                        _source_delivery_inputs,
                        _source_messages_from_preparation,
                    )
                    source_plan, source_contents = _source_delivery_inputs(
                        self, wire_truth_runtime, preparation,
                    )
                    source_messages = _source_messages_from_preparation(
                        preparation, guarded_messages,
                    )
                    guarded["messages"] = assemble_framefit5_request(
                        guarded_messages,
                        source_messages,
                        task_message=getattr(self, "_framefit5_task_message", None),
                        source_plan=source_plan,
                        allocation_plan={},
                        sources=source_contents,
                        runtime=wire_truth_runtime,
                        owner_contract=getattr(self, "request_action_contract", None),
                        owner_source_messages=source_messages,
                        frame_request_allocation=self.frame_request_allocation,
                    )
            return guarded

        def _apply_final_byte_fit(candidate: dict[str, object]) -> dict[str, object]:
            nonlocal preparation
            if self.frame_request_allocation is not None:
                candidate = _qualify_final_allocation(candidate)
                from kernel_context import token_estimate
                from kernel_frame_allocation_capture import finalize_frame_candidate
                from kernel_frame_allocation_validation import validate_allocation_authority
                from kernel_frame_optional_allocation import identity
                from kernel_request_budget import _request_token_meter, _wire_bytes

                allocation = self.frame_request_allocation
                validate_allocation_authority(
                    allocation, wire_truth_runtime, candidate["messages"],
                    check_layout=False,
                )
                capacity = allocation.capacity
                meter = _request_token_meter(wire_truth_runtime, self)
                if (_wire_bytes(candidate) <= capacity["byte_ceiling"]
                        and meter.measure(candidate)[0] <= capacity["token_ceiling"]
                        and token_estimate(candidate) <= capacity["frame_ceiling"]
                        and identity(candidate.get("tools", [])) == capacity["tool_schema_digest"]):
                    return candidate
                fitted = finalize_frame_candidate(
                    wire_truth_runtime, self, candidate, frame_counter=token_estimate,
                )
                # This is a new measured generation. Recovery must retain its
                # qualified layout instead of the superseded pre-fit selection.
                from kernel_request_reprepare import PreparationInput
                preparation = PreparationInput.capture(self, wire_truth_runtime, fitted)
                self._request_preparation_input = preparation
                if _boundary_state is not None:
                    _boundary_state.preparation = preparation
                return fitted
            from kernel_stateframe_f4_fit import (
                RETAIN_BYTE_HEADROOM, _state_message_parts,
                byte_fit_enabled, fit_request_to_byte_cap,
            )
            try:
                if not byte_fit_enabled():
                    return candidate
            except ValueError:
                return candidate
            candidate_messages = candidate.get("messages")
            if (not isinstance(candidate_messages, list)
                    or not any(
                        _state_message_parts(message) is not None
                        for message in candidate_messages if isinstance(message, Mapping)
                    )):
                return candidate
            from kernel_request_budget import HARD_MAX_REQUEST_BYTES
            max_bytes = int(HARD_MAX_REQUEST_BYTES)
            before = len(json.dumps(candidate).encode())
            if before <= max_bytes - RETAIN_BYTE_HEADROOM:
                return candidate

            from kernel_request_reprepare import _source_delivery_inputs
            from kernel_request_source_window import record_deferred_source_paths
            source_plan, _sources = _source_delivery_inputs(
                self, wire_truth_runtime, preparation,
            )
            working = copy.deepcopy(candidate)
            protected_paths: set[str] = set()
            if isinstance(visibility_check, Mapping):
                raw_candidates = visibility_check.get("candidates", ())
                if isinstance(raw_candidates, Sequence):
                    protected_paths.update(
                        str(row["path"]).replace("\\", "/")
                        for row in raw_candidates if isinstance(row, Mapping)
                        and isinstance(row.get("path"), str)
                    )
            if isinstance(source_plan, Mapping):
                units = source_plan.get("units", ())
                if isinstance(units, Sequence):
                    for unit in units:
                        if not isinstance(unit, Mapping):
                            continue
                        role = str(unit.get("role", "")).strip().lower()
                        if role in {
                            "target", "target_reservation", "task_target",
                            "open_edit", "open_edit_bytes", "protected",
                        }:
                            raw_path = unit.get("path")
                            if isinstance(raw_path, str):
                                protected_paths.add(raw_path.replace("\\", "/"))
            owner = getattr(wire_truth_runtime, "owner", None)
            root_value = getattr(owner, "root", None)
            target_value = getattr(wire_truth_runtime, "target", None)
            if root_value is not None and target_value is not None:
                try:
                    protected_paths.add(
                        Path(target_value).resolve().relative_to(
                            Path(root_value).resolve(),
                        ).as_posix()
                    )
                except (OSError, RuntimeError, TypeError, ValueError):
                    pass

            working_plan = copy.deepcopy(dict(source_plan)) \
                if isinstance(source_plan, Mapping) else None
            all_dropped: list[dict[str, object]] = []
            initial_bytes: int | None = before
            changed = False
            for _attempt in range(4):
                result = fit_request_to_byte_cap(
                    working, max_request_bytes=max_bytes,
                    protected_source_paths=tuple(sorted(protected_paths)),
                )
                after = result.get("after")
                all_dropped.extend(
                    copy.deepcopy(row) for row in result.get("dropped", ())
                    if isinstance(row, Mapping)
                )
                if not result.get("fits"):
                    self.journal.emit(
                        "fit_false", producer="kernel_stateframe_f4_fit",
                        request_id=self.request_id, step=self.step,
                        reason="minimum_frame_byte_cap", before=initial_bytes,
                        after=after, vital=True, dispatch=False,
                    )
                    return candidate
                working = result["body"]
                changed = changed or bool(result.get("changed"))
                omitted = result.get("omitted_sources", ())
                if omitted:
                    if working_plan is None or not isinstance(working.get("messages"), list):
                        self.journal.emit(
                            "fit_false", producer="kernel_stateframe_f4_fit",
                            request_id=self.request_id, step=self.step,
                            reason="source_omission_without_request_plan",
                            before=initial_bytes, after=after,
                            vital=True, dispatch=False,
                        )
                        return candidate
                    omitted_paths = tuple(dict.fromkeys(
                        str(row.get("path", "")).replace("\\", "/")
                        for row in omitted if isinstance(row, Mapping)
                        and isinstance(row.get("path"), str)
                    ))
                    deferred = record_deferred_source_paths(
                        working_plan, working["messages"], omitted_paths,
                    )
                    if set(omitted_paths) != set(deferred):
                        self.journal.emit(
                            "fit_false", producer="kernel_stateframe_f4_fit",
                            request_id=self.request_id, step=self.step,
                            reason="source_omission_route_unverified",
                            before=initial_bytes, after=after,
                            vital=True, dispatch=False,
                        )
                        return candidate
                current = len(json.dumps(working).encode())
                if current <= max_bytes - RETAIN_BYTE_HEADROOM:
                    break
            else:
                current = len(json.dumps(working).encode())

            current = len(json.dumps(working).encode())
            if current > max_bytes - RETAIN_BYTE_HEADROOM or not changed:
                self.journal.emit(
                    "fit_false", producer="kernel_stateframe_f4_fit",
                    request_id=self.request_id, step=self.step,
                    reason="minimum_frame_byte_cap",
                    before=initial_bytes, after=current,
                    vital=True, dispatch=False,
                )
                return candidate
            if working_plan is not None and working_plan.get("deferred_paths"):
                setattr(self, "_request_source_closure_override", working_plan)
            self.journal.emit(
                "f4_byte_fit", producer="kernel_stateframe_f4_fit",
                request_id=self.request_id, step=self.step,
                before=initial_bytes, after=current, dropped=all_dropped,
                dispatch=False,
            )
            return working

        def _rebudget_if_guard_grew(candidate: dict[str, object], *, ticket: bool,
                                    provider_profile: object = None) -> dict[str, object]:
            """Re-fit a body the final-wire guard grew past the budgeted cap (see refit_guarded_body)."""
            receipt = getattr(self, "request_budget_final_wire", None)
            limit = receipt.get("max_request_bytes") if isinstance(receipt, Mapping) else None

            def finalize(value: dict[str, object]) -> dict[str, object]:
                if ticket:
                    value = self._ensure_final_wire_ticket(value)
                if provider_profile is not None:
                    value = enforce_provider_body(value, profile=provider_profile)
                return value

            def emit(**fields: object) -> None:
                self.journal.emit("final_wire_rebudgeted", producer="kernel_measurement",
                                  request_id=self.request_id, step=self.step, dispatch=False,
                                  **fields)

            return refit_guarded_body(
                candidate, limit=limit, budget=lambda value: _apply_request_capture_budget(self, value),
                finalize=finalize, guard=_guard_final_wire, emit=emit,
            )

        if not _recovery_prepared:
            apply_to_request(body)
        admission = None
        if self.endpoint_profile is not None:
            from kernel_optional_engine_module import optional_module
            endpoint_extension = optional_module("STATEFRAME_ENDPOINT_PROFILE_MODULE")
            if endpoint_extension is None:
                raise RuntimeError("endpoint admission module is not configured")
            AdmissionStatus = endpoint_extension.AdmissionStatus
            EndpointAdmissionError = endpoint_extension.EndpointAdmissionError
            admission_for_request = endpoint_extension.admission_for_request
            profile = self.endpoint_profile
            observation = getattr(profile, "observation", None)
            slots_processing = getattr(observation, "slots_processing", None)
            queue_time = getattr(profile, "queue_time", lambda: None)()
            # RequestCapture is called after the live runtime has selected its
            # provider path; there is no local lease wait to add. A missing
            # queue observation remains absent from the profile and is not
            # mistaken for a measured queue duration.
            admission = admission_for_request(
                profile,
                _prompt_token_estimate(body),
                slots_processing=slots_processing,
                queue_time_seconds=queue_time,
            )
            if admission.status is not AdmissionStatus.ADMITTED or admission.deadline_seconds is None:
                self.journal.emit(
                    "request_not_admitted",
                    request_id=self.request_id,
                    step=self.step,
                    admission=admission.mapping(),
                    dispatch=False,
                )
                raise EndpointAdmissionError(admission)
            derived_deadline = time.monotonic() + admission.deadline_seconds
            deadline = derived_deadline if deadline is None else min(deadline, derived_deadline)
        if admission is None:
            socket_timeout, wall_timeout = request_timeouts()
            if deadline is None and per_request_timeout is not None:
                total_wall_budget = max(
                    request_timeout_budget_seconds(), wall_timeout * 2)
                deadline = time.monotonic() + total_wall_budget
        else:
            # A measured profile is the sole budget authority for this request;
            # legacy REACT_REQUEST_TIMEOUT_S must not widen its deadline.
            assert admission.deadline_seconds is not None
            socket_timeout = min(float(getattr(self.endpoint_profile, "connect_timeout_seconds")),
                                 admission.deadline_seconds)
            wall_timeout = admission.deadline_seconds
            if per_request_timeout is not None:
                socket_timeout = min(socket_timeout, per_request_timeout)
                wall_timeout = min(wall_timeout, per_request_timeout)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if not math.isfinite(deadline):
                raise TimeoutError("request deadline expired before submission")
            if remaining <= 0:
                if self.parent_meter is None:
                    raise TimeoutError("request deadline expired before submission")
            else:
                socket_timeout = min(socket_timeout, remaining)
                wall_timeout = min(wall_timeout, remaining)
        router = getattr(self, "thinking_router", None)
        classification = getattr(self, "thinking_router_classification", None)
        if not _recovery_prepared and router is not None and classification is not None:
            body = router.prepare_body(body, classification)
        model_router = getattr(self, "model_router", None)
        if model_router is None:
            model_router = getattr(self.journal, "model_router", None)
        if not _recovery_prepared and model_router is not None:
            prepared_call = model_router.prepare_call(self, body, endpoint, key)
            body, endpoint, key = (prepared_call.body, prepared_call.endpoint,
                                   prepared_call.key)
            model_router.validate_final_request(self, body)
        if not _recovery_prepared:
            self._v43g_prepare_body(body)
        # Preserve the once-adapted transport input before budget/fact edits.
        # Recovery must not acquire a new route or rerun router adaptation.
        transport_body = copy.deepcopy(body)
        if _boundary_state is not None:
            _boundary_state.phase = "pre_budget_request_snapshot"
            _boundary_state.transport_body = copy.deepcopy(transport_body)
            _boundary_state.endpoint = endpoint
            _boundary_state.key = key
            _boundary_state.deadline = deadline
        try:
            from kernel_pilot import ACTIVE_RUNTIME
            wire_truth_runtime = ACTIVE_RUNTIME.get()
        except (ImportError, LookupError):
            wire_truth_runtime = None
        if wire_truth_runtime is None:
            # Deadline transport may run in a worker thread, where ContextVars
            # are empty. prepare() explicitly binds this capture to its owner.
            wire_truth_runtime = getattr(self, "completion_owner_runtime", None)
        from kernel_request_kind import is_auxiliary
        if is_auxiliary(self):
            # Auxiliary prompts may quote a frame, but cannot acquire the
            # solver's source/menu/dispatch authority through ambient context.
            wire_truth_runtime = None
        if _boundary_state is not None:
            _boundary_state.runtime = wire_truth_runtime
        from kernel_request_reprepare import PreparationInput, qualify_source
        preparation = _boundary_state.preparation if _recovery_prepared and _boundary_state else None
        from kernel_wire_truth_gate import (
            add_fact_messages as _add_wire_truth_fact_messages,
            capture_context as _capture_wire_truth_context,
            enabled as _wire_truth_gate_enabled,
        )
        import kernel_one_snapshot_preparation as _one_snapshot
        from kernel_final_wire_budget import (
            budget_verdict_signature, snapshot_revalidation_inputs,
        )
        one_snapshot_enabled = _one_snapshot.enabled()
        one_snapshot_first_wire: bytes | None = None
        one_snapshot_first_verdict = "UNKNOWN"
        one_snapshot_first_id = ""
        one_snapshot_first_check: object | None = None
        one_snapshot_first_context: dict[str, object] | None = None
        one_snapshot_first_request_extensions: dict[str, object] | None = None
        one_snapshot_first_action_contract: object | None = None
        one_snapshot_first_action_contract_captured = False
        one_snapshot_skip_recovery = False

        def _one_snapshot_record(
                validator: str, second_verdict: str,
                candidate_wire: bytes | None = None,
        ) -> bool:
            if not one_snapshot_enabled or one_snapshot_first_wire is None:
                return False
            checked_wire = (
                candidate_wire if isinstance(candidate_wire, bytes)
                else one_snapshot_first_wire
            )
            return _one_snapshot.record_disagreement(
                self,
                validator=validator,
                first_verdict=one_snapshot_first_verdict,
                second_verdict=second_verdict,
                first_snapshot_id=one_snapshot_first_id,
                second_snapshot_id=_one_snapshot.snapshot_id(preparation, checked_wire),
                flag_enabled=one_snapshot_enabled,
            )

        def _one_snapshot_fallback_body(
                validator: str, error: BaseException,
                candidate: Mapping[str, object],
        ) -> dict[str, object] | None:
            candidate_wire: bytes | None
            try:
                candidate_wire = json.dumps(candidate).encode()
            except Exception:
                candidate_wire = None
            if not _one_snapshot_record(
                    validator, type(error).__name__, candidate_wire):
                return None
            first_wire = getattr(self, "_one_snapshot_first_wire", None)
            if not _one_snapshot.has_sendable_body(first_wire):
                return None
            try:
                restored = json.loads(first_wire)
            except (TypeError, ValueError, UnicodeDecodeError):
                return None
            return restored if isinstance(restored, dict) else None

        # (digest, verdict signature, frozen non-body inputs) at the wire-truth PASS.
        wire_truth_revalidation_baseline: tuple[
            str, tuple[object, ...], object,
        ] | None = None

        def _capture_final_truth_context(candidate: Mapping[str, object]) -> dict[str, object]:
            context = _capture_wire_truth_context(
                self, wire_truth_runtime, candidate,
                side_call=side_call_contract is not None,
            )
            if lean_on:
                required = context.get("required_facts")
                known: dict[str, object] = {}
                if isinstance(required, Mapping):
                    for name, value in required.items():
                        fact_value = value.get("value") if isinstance(value, Mapping) else value
                        if str(fact_value).strip().upper() not in {"UNKNOWN", "OMITTED"}:
                            known[str(name)] = value
                context["required_facts"] = known
            return context

        from kernel_stateframe_framefit5 import has_framefit5_state
        from kernel_request_kind import classify
        # Classification also runs for direct recovery entry and refuses a
        # missing frame/runtime before the legacy writers below can execute.
        classify(self, wire_truth_runtime, body, side_call_contract=side_call_contract)
        if (wire_truth_runtime is not None and side_call_contract is None
                and getattr(self, "request_kind", None) in (None, "framefit5_primary")
                and has_framefit5_state(body.get("messages", []))):
            from kernel_final_wire_allocation import build_qualified_request
            from kernel_wire_truth_gate import WireTruthBlocked
            # Draft bookkeeping cannot opt a main FrameFit5 send out of sealing.
            # Mark this lifecycle before any fallible proof so a rejected draft
            # remains terminal instead of entering legacy boundary recovery.
            if self.qualified_request is None:
                self._framefit5_pending_qualification = True
            # Lazily set by the budget pass (kernel_request_budget); a capture that skipped it has no attribute (census b1604 benchmark-v2
            # req 17 AttributeError). Read it the way every other site does.
            proposal = getattr(self, "_frame_request_allocation_candidate", None)
            if proposal is not None:
                from kernel_frame_optional_allocation import validate_optional_dependencies
                from kernel_stateframe_framefit5 import _workspace_root
                validate_optional_dependencies(proposal.optional_heads, root=_workspace_root(wire_truth_runtime))
            preparation = PreparationInput.capture(self, wire_truth_runtime, body)
            try:
                record = build_qualified_request(
                    self, wire_truth_runtime, body, endpoint=endpoint, key=key, deadline=deadline,
                )
            except WireTruthBlocked as blocked:
                # Preserve the existing single bounded wire-truth replacement.
                # Source/allocation/owner failures are never repair triggers.
                # Acceptance still counts any recovered block as a failure.
                if self.qualified_request is not None or getattr(self, "_wire_truth_rebuild_used", False):
                    raise
                if _boundary_state is not None:
                    _boundary_state.wire_truth_rebuild_attempted = True
                if blocked.source_layout_proposal is not None:
                    from dataclasses import replace
                    preparation = replace(
                        preparation, source_layout_proposal=blocked.source_layout_proposal,
                    )
                wire_truth_runtime.prepare_request_replacement(
                    self, preparation, body, deadline=deadline, endpoint=endpoint, key=key)
                record = self.qualified_request
                if record is None:
                    raise RequestPreparationContractError("replacement did not publish a qualified request")
            wire, body = record.wire, record.body
            sealed_metadata = json.loads(record.metadata)
            request_extensions = sealed_metadata["request_extensions"]
            visibility_receipt = sealed_metadata["visibility_receipt"]
            wire_truth_check = self._wire_truth_check
            framefit5_request = True
            framefit5_assembler_messages = json.dumps(body["messages"]).encode("utf-8")
            kv_chunk_reuse_transport = None
            if _boundary_state is not None:
                _boundary_state.preparation = self._request_preparation_input
                _boundary_state.phase = "qualified_request_sealed"
                _boundary_state.body_digest = "sha256:" + record.body_digest
        else:
            wire_truth_context = _capture_final_truth_context(body)
            self._wire_truth_runtime = wire_truth_runtime
            self._wire_truth_side_call = side_call_contract is not None
            if _wire_truth_gate_enabled():
                body = _add_wire_truth_fact_messages(
                    body, wire_truth_context, append_only=lean_on)
                from kernel_release_wire_claims import READ_PREFIX, read_contract_capsule
                capsule, capsule_fields = read_contract_capsule(wire_truth_runtime)
                self.read_contract_wire_capsule = capsule
                self.read_contract_wire_fields = capsule_fields
                if capsule is not None and not lean_on:
                    body["messages"] = [message for message in body.get("messages", ())
                        if not (isinstance(message, Mapping)
                                and str(message.get("content", "")).startswith(READ_PREFIX))]
                    body["messages"].append({"role": "user", "content": capsule})
            # Soak read_once stage 2 (opt-in final_message_at_3): bare sentence as the LAST
            # message, after every host wire note.
            if not lean_on:
                from kernel_read_once import append_final_message as _read_once_final
                body["messages"] = _read_once_final(wire_truth_runtime, body.get("messages"))
            from kernel_read_reuse_visibility import prepare_final_visibility, verify_final_visibility
            if _boundary_state is not None:
                _boundary_state.phase = "final_source_visibility"
            body, initial_visibility_check = prepare_final_visibility(
                self, wire_truth_runtime, body, preparation=preparation,
            )
            # LEAN removes advisory material, while source delivery and final-wire
            # contract proofs remain required at this transport boundary.
            if _boundary_state is not None:
                _boundary_state.phase = "final_request_budget"
            body = _apply_request_capture_budget(self, body)
            if (getattr(self, "_boundary_recovery_attempted", False)
                    and wire_truth_runtime is not None and preparation is not None
                    and side_call_contract is None):
                # Budgeting resets request-local deferred-path state and may trim its
                # notice. Re-derive the exact eligible gap after that transform so
                # the same final validators see a notice and dispatchable read route.
                if getattr(wire_truth_runtime, "_boundary_recovery_read_admitted", False):
                    from kernel_read_reuse_visibility import defer_unproven_source_claims
                    defer_unproven_source_claims(
                        self, wire_truth_runtime, preparation, body, boundary_recovery=True,
                    )
                    source_plan = getattr(self, "_request_source_closure_override", None)
                    if not isinstance(source_plan, Mapping):
                        source_plan = getattr(wire_truth_runtime, "_request_source_closure", None)
                    raw_deferred = (
                        source_plan.get("deferred_paths", ())
                        if isinstance(source_plan, Mapping) else ()
                    )
                    deferred_paths = tuple(
                        str(path) for path in raw_deferred
                        if isinstance(path, str) and path.strip()
                    ) if isinstance(raw_deferred, (list, tuple)) else ()
                    if deferred_paths and not any(
                            isinstance(tool, Mapping)
                            and isinstance(tool.get("function"), Mapping)
                            and tool["function"].get("name") == "read"
                            for tool in body.get("tools", ())):
                        raise RequestPreparationContractError(
                            "boundary source deferral lost its dispatchable read route")
                    wire_truth_runtime._request_deferred_source_paths = deferred_paths
                else:
                    # The frozen recovery menu admitted no read resolver; later budget
                    # transforms must not revive a source claim from its prior plan.
                    wire_truth_runtime._request_deferred_source_paths = ()
            body = self._ensure_final_wire_ticket(body)
            if (wire_truth_runtime is not None and side_call_contract is None
                    and isinstance(self.request_action_contract, Mapping)):
                # Source deferrals are request-local finalization state. Snapshot
                # them after budget/source settlement so recovery validates the
                # exact prepared source plan it will re-verify on the replacement.
                if not _recovery_prepared:
                    preparation = PreparationInput.capture(self, wire_truth_runtime, entry_body)
                if _boundary_state is not None:
                    _boundary_state.preparation = preparation
            final_profile = reasoning_profile(manifest=self.journal.manifest)
            if _boundary_state is not None:
                _boundary_state.phase = "provider_body_validation"
            try:
                # This is the last mutable request seam.  It intentionally runs
                # after router, sampler, template, and budget transforms.
                body = enforce_provider_body(body, profile=final_profile)
            except ProviderBoundaryContractError as error:
                self.journal.emit(
                    "provider_boundary_contract_error",
                    producer="kernel_launch_identity_cost",
                    request_id=self.request_id,
                    step=self.step,
                    code=error.code,
                    error=str(error),
                    outcome=error.outcome,
                    failure_origin=error.failure_origin,
                    dispatch=False,
                )
                raise
            try:
                if _boundary_state is not None:
                    _boundary_state.phase = "owner_pre_wire_validation"
                request_extensions = self._request_capture_pre_wire_extensions(
                    body, endpoint, key, deadline=deadline,
                    side_call_contract=side_call_contract, preparation=preparation)
            except RequestPreparationContractError as error:
                try:
                    _stop_pre_dispatch_contract(
                        self, event="request_pre_wire_validation_stopped",
                        code="request_pre_wire_preparation_failed",
                        detail="request pre-wire preparation failed before provider dispatch",
                        failures=(type(error).__name__,), error=error,
                    )
                except ProviderBoundaryContractError as stop:
                    raise stop from error
            # Audit 2026-10-05 #1: a schema llama.cpp cannot compile (e.g. enum: [])
            # is a harness contract defect.  Refuse BEFORE the wire, journal once.
            from kernel_tool_schema_lint import ToolSchemaContractError, lint_tools
            if _boundary_state is not None:
                _boundary_state.phase = "tool_schema_validation"
            problems = lint_tools(body.get("tools"))
            if problems:
                self.journal.emit(
                    "harness_contract_error", request_id=self.request_id, step=self.step,
                    producer="kernel_tool_schema_lint", problems=problems,
                    outcome=HARNESS_CONTRACT, failure_origin="harness", dispatch=False)
                raise ToolSchemaContractError(problems)
            # Audit #8: never send a prompt + max_output that overflows n_ctx.
            from kernel_context_capacity import ContextCapacityError, enforce_context_capacity
            if _boundary_state is not None:
                _boundary_state.phase = "context_capacity_validation"
            wire_truth_context = _capture_final_truth_context(body)
            self._wire_truth_context = wire_truth_context
            enforce_context_capacity(self, body, endpoint=endpoint, key=key)
            if _boundary_state is not None:
                _boundary_state.phase = "final_wire_guard_and_rebudget"
            body = _guard_final_wire(body)
            body = _rebudget_if_guard_grew(body, ticket=False)
            enforce_context_capacity(self, body, endpoint=endpoint, key=key)
            if self._ensure_contract_bound_evidence_pin(
                    body, runtime=wire_truth_runtime, side_call_contract=side_call_contract,
                    preparation=preparation):
                # A restored pin adds bytes after budgeting; re-check capacity (fail closed).
                enforce_context_capacity(self, body, endpoint=endpoint, key=key)
            if (wire_truth_runtime is not None
                    and getattr(wire_truth_runtime, "stateframe_trailing_panels_mode", "off") == "on"):
                from kernel_cache_layout import (
                    history_messages, history_prefix_audit, layout_messages,
                    trailing_panels_audit,
                )
                expected_history = getattr(self, "trailing_panels_history_at_prepare", None)
                current_history = history_messages(body.get("messages", ()))
                history_check = history_prefix_audit(
                    expected_history if isinstance(expected_history, list) else None,
                    current_history,
                    previous_request_id=getattr(
                        self, "trailing_panels_history_request_id", self.request_id,
                    ),
                )
                fold_pointer = history_check.get("fold_pointer")
                if isinstance(fold_pointer, str):
                    body["messages"].append({"role": "user", "content": fold_pointer})
                    self.journal.emit(
                        "trailing_panels_history_fold_pointer",
                        producer="kernel_cache_layout",
                        request_id=self.request_id,
                        previous_request_id=getattr(
                            self, "trailing_panels_history_request_id", None,
                        ),
                        first_changed_index=history_check.get("first_changed_index"),
                        previous_history_sha256=history_check.get("previous_history_sha256"),
                        current_history_sha256=history_check.get("current_history_sha256"),
                        dispatch=False,
                    )
                before_trailing_layout = list(body.get("messages", ()))
                body["messages"] = [
                    dict(message) for message in layout_messages(
                        before_trailing_layout, mode="off", trailing_panels=True,
                    )
                ]
                trailing_audit = trailing_panels_audit(
                    before_trailing_layout, body["messages"],
                )
                trailing_audit.update({
                    "history_prefix_unchanged": history_check.get("prefix_unchanged"),
                    "history_prefix_changed_index": history_check.get("first_changed_index"),
                    "fold_pointer_appended": isinstance(fold_pointer, str),
                })
                self.journal.emit(
                    "trailing_panels_wire_layout",
                    producer="kernel_cache_layout",
                    request_id=self.request_id,
                    dispatch=False,
                    **trailing_audit,
                )
                enforce_context_capacity(self, body, endpoint=endpoint, key=key)
            from kernel_request_tool_surface import author_final_tool_surface
            final_tool_surface = author_final_tool_surface(
                self, wire_truth_runtime, body, request_id=self.request_id,
                side_call=side_call_contract is not None,
            )
            if (isinstance(final_tool_surface, Mapping)
                    and final_tool_surface.get("source_tool_schema_digest")
                    != final_tool_surface.get("tool_schema_digest")):
                enforce_context_capacity(self, body, endpoint=endpoint, key=key)
            wire = json.dumps(body).encode()  # matches the legacy serializer exactly
            if _boundary_state is not None:
                _boundary_state.phase = "final_serialization_validation"
                _boundary_state.body_digest = "sha256:" + hashlib.sha256(wire).hexdigest()
            verify_final_visibility(wire, initial_visibility_check)
            self._assert_wire_string_hygiene(wire)
            try:
                if _boundary_state is not None:
                    _boundary_state.phase = "source_proof_validation"
                wire = qualify_source(
                    self, wire_truth_runtime, preparation, body, wire,
                    allow_requalification=False,
                )
            except RequestPreparationContractError:
                # Defer source repair until the final byte-fit pass; final proof remains a hard gate.
                pass
            wire_truth_context = _capture_final_truth_context(body)
            self._wire_truth_context = wire_truth_context
            from kernel_wire_truth_gate import WireTruthBlocked, check_wire
            try:
                if _boundary_state is not None:
                    _boundary_state.phase = "wire_truth_validation"
                first_wire_truth_check = check_wire(
                    wire, wire_truth_context, emit=self.journal.emit,
                    attempt=1, allow_done_repair=False,
                )
                if one_snapshot_enabled:
                    one_snapshot_first_wire = bytes(wire)
                    one_snapshot_first_verdict = str(first_wire_truth_check.verdict)
                    one_snapshot_first_id = _one_snapshot.snapshot_id(preparation, wire)
                    one_snapshot_first_check = first_wire_truth_check
                    one_snapshot_first_context = copy.deepcopy(wire_truth_context)
                    one_snapshot_first_request_extensions = dict(request_extensions)
                    try:
                        one_snapshot_first_action_contract = copy.deepcopy(
                            getattr(self, "request_action_contract", None),
                        )
                        one_snapshot_first_action_contract_captured = True
                    except Exception:
                        one_snapshot_first_action_contract = getattr(
                            self, "request_action_contract", None,
                        )
                        one_snapshot_first_action_contract_captured = True
                    self._one_snapshot_first_wire = one_snapshot_first_wire
                    self._one_snapshot_first_check = one_snapshot_first_check
                    self._one_snapshot_first_verdict = one_snapshot_first_verdict
                    self._one_snapshot_first_snapshot_id = one_snapshot_first_id
                if first_wire_truth_check.verdict == "PASS":
                    try:
                        first_signature = budget_verdict_signature(
                            getattr(self, "request_budget_final_wire", None),
                        )
                        if first_signature is not None:
                            wire_truth_revalidation_baseline = (
                                "sha256:" + hashlib.sha256(wire).hexdigest(),
                                first_signature,
                                snapshot_revalidation_inputs(self, wire_truth_runtime, body),
                            )
                    except Exception as error:
                        if not _one_snapshot_record(
                                "wire_truth_revalidation_snapshot", type(error).__name__, wire):
                            raise
            except WireTruthBlocked as error:
                # The existing owner-driven repair runs first; retain its cause if
                # the replacement also fails the final request contract.
                if one_snapshot_enabled:
                    from kernel_wire_truth_gate import WireTruthResult, schema_digest
                    one_snapshot_first_wire = bytes(wire)
                    one_snapshot_first_verdict = "BLOCKED"
                    one_snapshot_first_id = _one_snapshot.snapshot_id(preparation, wire)
                    one_snapshot_first_context = copy.deepcopy(wire_truth_context)
                    one_snapshot_first_check = WireTruthResult(
                        request_id=self.request_id,
                        revision=str(wire_truth_context.get("revision", "UNKNOWN")),
                        body_sha256=(error.body_sha256 or
                            "sha256:" + hashlib.sha256(wire).hexdigest()),
                        schema_digest=schema_digest(body.get("tools")),
                        verdict="BLOCKED",
                        failed=tuple(getattr(error, "failed", ())),
                        body=copy.deepcopy(body),
                        wire=bytes(wire),
                        attempt=1,
                    )
                    one_snapshot_first_request_extensions = dict(request_extensions)
                    try:
                        one_snapshot_first_action_contract = copy.deepcopy(
                            getattr(self, "request_action_contract", None),
                        )
                        one_snapshot_first_action_contract_captured = True
                    except Exception:
                        one_snapshot_first_action_contract = getattr(
                            self, "request_action_contract", None,
                        )
                        one_snapshot_first_action_contract_captured = True
                    self._one_snapshot_first_wire = one_snapshot_first_wire
                    self._one_snapshot_first_check = one_snapshot_first_check
                    self._one_snapshot_first_verdict = one_snapshot_first_verdict
                    self._one_snapshot_first_snapshot_id = one_snapshot_first_id
                    self._wire_truth_check = one_snapshot_first_check
                if _boundary_state is not None:
                    _boundary_state.wire_truth_error = error
                one_snapshot_skip_recovery = False
                if (preparation is None or side_call_contract is not None
                        or getattr(self, "_wire_truth_rebuild_used", False)):
                    if _one_snapshot_record(
                            "wire_truth_recovery_precondition", "RECOVERY_UNAVAILABLE", wire):
                        one_snapshot_skip_recovery = True
                    else:
                        _stop_pre_dispatch_contract(
                            self, event="request_wire_truth_recovery_stopped",
                            code="wire_truth_recovery_unavailable",
                            detail="wire-truth validation remained blocked without an eligible replacement",
                            failures=tuple(str(value) for value in getattr(error, "failed", ())), error=error,
                        )
                # Once the established repair path is entered, preserve its typed
                # terminal failure instead of replaying the same request through fallback.
                if not one_snapshot_skip_recovery:
                    if _boundary_state is not None:
                        _boundary_state.wire_truth_rebuild_attempted = True
                    try:
                        prepared = wire_truth_runtime.prepare_request_replacement(
                            self, preparation, transport_body, deadline=deadline)
                    except Exception as error:
                        if getattr(error, "outcome", None) == HARNESS_TRANSPORT:
                            raise
                        original = error
                        if not (getattr(error, "harness_contract", False)
                                or getattr(error, "outcome", None) == HARNESS_CONTRACT):
                            original = RequestPreparationContractError(
                                "request replacement preparation failed: " + type(error).__name__)
                            original.__cause__ = error
                        if _one_snapshot_record(
                                "wire_truth_recovery", type(original).__name__, wire):
                            one_snapshot_skip_recovery = True
                        else:
                            _stop_pre_dispatch_contract(
                                self, event="request_wire_truth_recovery_stopped",
                                code="wire_truth_recovery_failed",
                                detail="bounded wire-truth recovery could not prepare a qualified replacement",
                                failures=(type(error).__name__,), error=original,
                            )
                if not one_snapshot_skip_recovery:
                    if prepared.capture is not self or not getattr(self, "_wire_truth_rebuild_used", False):
                        if _one_snapshot_record(
                                "wire_truth_recovery_commit", "UNCOMMITTED", wire):
                            one_snapshot_skip_recovery = True
                        else:
                            _stop_pre_dispatch_contract(
                                self, event="request_wire_truth_recovery_stopped",
                                code="wire_truth_recovery_uncommitted",
                                detail="replacement preparation did not commit the active capture",
                            )
                if not one_snapshot_skip_recovery:
                    self._send_generation = 1
                    permit_prefix = f"{self.request_id}:send:{self._send_generation}"
                    body = copy.deepcopy(transport_body)
                    body["messages"] = list(prepared.messages)
                    from kernel_request_tool_surface import author_final_tool_surface
                    author_final_tool_surface(
                        self, wire_truth_runtime, body, request_id=self.request_id,
                        tools_override=list(prepared.tools or ()),
                        side_call=side_call_contract is not None,
                    )
                    if _boundary_state is not None:
                        _boundary_state.repair_body = copy.deepcopy(body)
                    context = _capture_final_truth_context(body)
                    if _wire_truth_gate_enabled():
                        body = _add_wire_truth_fact_messages(
                            body, context, append_only=lean_on)
                        capsule, capsule_fields = read_contract_capsule(wire_truth_runtime)
                        self.read_contract_wire_capsule = capsule
                        self.read_contract_wire_fields = capsule_fields
                        if capsule is not None and not lean_on:
                            body["messages"] = [message for message in body.get("messages", ())
                                if not (isinstance(message, Mapping)
                                        and str(message.get("content", "")).startswith(READ_PREFIX))]
                            body["messages"].append({"role": "user", "content": capsule})
            # No lifecycle, routing or auxiliary transport hooks have run twice.
            # Pin current source visibility before budget validation of the replacement.
            from kernel_read_reuse_visibility import (
                defer_unproven_source_claims, prepare_final_visibility,
                verify_final_visibility,
            )
            if preparation is not None:
                try:
                    defer_unproven_source_claims(
                        self, wire_truth_runtime, preparation, body)
                except RequestPreparationContractError as error:
                    fallback_body = _one_snapshot_fallback_body(
                        "final_source_closure", error, body,
                    )
                    if fallback_body is None:
                        raise
                    body = fallback_body
            body_before_pins = copy.deepcopy(body)
            try:
                body, visibility_check = prepare_final_visibility(
                    self, wire_truth_runtime, body, preparation=preparation)
            except RequestPreparationContractError as error:
                fallback_body = _one_snapshot_fallback_body(
                    "final_source_visibility", error, body,
                )
                if fallback_body is None:
                    raise
                body = fallback_body
                visibility_check = initial_visibility_check
            # Complete every mutable transform before the final serialized proof.
            try:
                body = _apply_request_capture_budget(self, body)
                body = self._ensure_final_wire_ticket(body)
                body = enforce_provider_body(body, profile=final_profile)
                body = _guard_final_wire(body)
                body = _rebudget_if_guard_grew(
                    body, ticket=True, provider_profile=final_profile,
                )
            except Exception as error:
                fallback_body = _one_snapshot_fallback_body(
                    "final_request_transforms", error, body,
                )
                if fallback_body is None:
                    raise
                body = fallback_body
                visibility_check = initial_visibility_check
            final_problems = lint_tools(body.get("tools"))
            if final_problems:
                if _one_snapshot_record(
                        "final_tool_schema", "INVALID", json.dumps(body).encode()):
                    body = json.loads(one_snapshot_first_wire)
                else:
                    raise ToolSchemaContractError(final_problems)
            try:
                enforce_context_capacity(self, body, endpoint=endpoint, key=key)
            except ContextCapacityError:
                # Keep truthful missing-range language when a current source span cannot fit.
                try:
                    body = body_before_pins
                    body, visibility_check = prepare_final_visibility(
                        self, wire_truth_runtime, body, allow_pins=False,
                        preparation=preparation)
                    body = _apply_request_capture_budget(self, body)
                    body = self._ensure_final_wire_ticket(body)
                    body = enforce_provider_body(body, profile=final_profile)
                    body = _guard_final_wire(body)
                    body = _rebudget_if_guard_grew(
                        body, ticket=True, provider_profile=final_profile,
                    )
                    final_problems = lint_tools(body.get("tools"))
                    if final_problems:
                        if _one_snapshot_record(
                                "final_tool_schema", "INVALID", json.dumps(body).encode()):
                            body = json.loads(one_snapshot_first_wire)
                            visibility_check = initial_visibility_check
                        else:
                            raise ToolSchemaContractError(final_problems)
                    enforce_context_capacity(self, body, endpoint=endpoint, key=key)
                except Exception as error:
                    if _one_snapshot_record(
                            "final_context_capacity_fallback", type(error).__name__,
                            json.dumps(body).encode()):
                        body = json.loads(one_snapshot_first_wire)
                        visibility_check = initial_visibility_check
                    else:
                        raise
            if (wire_truth_runtime is not None and preparation is not None
                    and (_boundary_state is not None
                         or getattr(self, "_boundary_recovery_attempted", False))):
                final_tools = body.get("tools", ())
                final_tools = final_tools if isinstance(final_tools, Sequence) \
                    and not isinstance(final_tools, (str, bytes, bytearray)) else ()
                final_messages = body.get("messages", ())
                final_messages = final_messages if isinstance(final_messages, Sequence) \
                    and not isinstance(final_messages, (str, bytes, bytearray)) else ()
                read_available = any(
                    isinstance(tool, Mapping)
                    and isinstance(tool.get("function"), Mapping)
                    and tool["function"].get("name") == "read"
                    for tool in final_tools
                )
                deferred_paths = getattr(
                    wire_truth_runtime, "_request_deferred_source_paths", (),
                )
                deferred_notice = any(
                    isinstance(message, Mapping)
                    and str(message.get("content", "")).startswith(
                        "[Deferred required source paths]",
                    )
                    for message in final_messages
                )
                self.journal.emit(
                    "boundary_recovery_final_source_menu_checked",
                    producer="kernel_measurement",
                    request_id=self.request_id,
                    step=self.step,
                    read_available=read_available,
                    deferred_path_count=(
                        len(deferred_paths) if isinstance(deferred_paths, Sequence)
                        and not isinstance(deferred_paths, (str, bytes, bytearray)) else 0
                    ),
                    deferred_notice_present=deferred_notice,
                    dispatch=False,
                )
                if not read_available and (deferred_paths or deferred_notice):
                    # The final budget pass can remove read after an earlier
                    # source-deferral decision. Reconcile against the exact menu
                    # that will be serialized, then budget the qualified body again.
                    try:
                        defer_unproven_source_claims(
                            self, wire_truth_runtime, preparation, body,
                            boundary_recovery=True,
                        )
                    except RequestPreparationContractError as error:
                        fallback_body = _one_snapshot_fallback_body(
                            "final_boundary_source_menu", error, body,
                        )
                        if fallback_body is None:
                            raise
                        body = fallback_body
                    source_plan = getattr(self, "_request_source_closure_override", None)
                    if not isinstance(source_plan, Mapping):
                        source_plan = getattr(wire_truth_runtime, "_request_source_closure", None)
                    remaining_deferred = (
                        source_plan.get("deferred_paths", ())
                        if isinstance(source_plan, Mapping) else ()
                    )
                    if remaining_deferred:
                        if not _one_snapshot_record(
                                "final_source_menu", "UNROUTED_DEFERRED_PATHS",
                                json.dumps(body).encode()):
                            raise RequestPreparationContractError(
                                "final boundary menu cannot dispatch deferred source reads")
                    wire_truth_runtime._request_deferred_source_paths = ()
                    try:
                        body, visibility_check = prepare_final_visibility(
                            self, wire_truth_runtime, body, preparation=preparation,
                        )
                        body = _apply_request_capture_budget(self, body)
                        body = self._ensure_final_wire_ticket(body)
                        body = enforce_provider_body(body, profile=final_profile)
                        body = _guard_final_wire(body)
                        body = _rebudget_if_guard_grew(
                            body, ticket=True, provider_profile=final_profile,
                        )
                    except Exception as error:
                        fallback_body = _one_snapshot_fallback_body(
                            "final_boundary_source_transforms", error, body,
                        )
                        if fallback_body is None:
                            raise
                        body = fallback_body
                        visibility_check = initial_visibility_check
                    final_problems = lint_tools(body.get("tools"))
                    if final_problems:
                        if _one_snapshot_record(
                                "final_tool_schema", "INVALID", json.dumps(body).encode()):
                            body = json.loads(one_snapshot_first_wire)
                        else:
                            raise ToolSchemaContractError(final_problems)
                    try:
                        enforce_context_capacity(self, body, endpoint=endpoint, key=key)
                    except ContextCapacityError as error:
                        if _one_snapshot_record(
                                "final_context_capacity", type(error).__name__,
                                json.dumps(body).encode()):
                            body = json.loads(one_snapshot_first_wire)
                            visibility_check = initial_visibility_check
                        else:
                            raise
                    final_messages = body.get("messages", ())
                    final_messages = final_messages if isinstance(final_messages, Sequence) \
                        and not isinstance(final_messages, (str, bytes, bytearray)) else ()
                    if any(
                            isinstance(message, Mapping)
                            and str(message.get("content", "")).startswith(
                                "[Deferred required source paths]",
                            )
                            for message in final_messages):
                        if not _one_snapshot_record(
                                "final_source_menu", "UNROUTED_DEFERRED_NOTICE",
                                json.dumps(body).encode()):
                            raise RequestPreparationContractError(
                                "final boundary body retained an unrouted deferred-source notice")
            # read_once notes are confirmed on the exact final body (lean writer, r2AB);
            # the r2AT merge dropped this call, so read_once queued but never fired.
            owner_runtime = getattr(self, "completion_owner_runtime", None)
            if not lean_on:
                from kernel_read_once import ensure_wire_note
                body["messages"] = ensure_wire_note(
                    wire_truth_runtime, body.get("messages"))
                if owner_runtime is not wire_truth_runtime:
                    body["messages"] = ensure_wire_note(
                        owner_runtime, body.get("messages"))
            kv_chunk_reuse_transport = None
            try:
                from kernel_kv_chunk_reuse import enabled as kv_chunk_reuse_enabled
                if kv_chunk_reuse_enabled():
                    from kernel_kv_chunk_reuse import configure_transport
                    kv_chunk_reuse_transport = configure_transport(body, endpoint, key)
            except ImportError:
                pass
            template_mode = getattr(self, "template_conformance_mode", "off")
            if template_mode != "off":
                from kernel_template_conformance import capture_submitted
                capture_submitted(self, body, endpoint, key, mode_value=template_mode,
                    runtime=getattr(self, "template_conformance_runtime", None), deadline=deadline)
            self._wire_truth_runtime = wire_truth_runtime
            self._wire_truth_side_call = side_call_contract is not None
            self._wire_truth_side_call_contract = side_call_contract
            self._request_preparation_input = preparation
            self._ensure_contract_bound_evidence_pin(
                body, runtime=wire_truth_runtime, side_call_contract=side_call_contract,
                preparation=preparation,
            )
            # Late notes, evidence pins, and optional panel layout can follow the
            # guard; rebuild the stable prefix before the trailing volatile region.
            # Normalize again so the only volatile tail is last. The guard may run
            # during refit; keep typed receipts through its final pass.
            if tool_reply_on:
                _record_tool_reply_receipts(
                    body.get("messages"), self.journal,
                    request_id=self.request_id, step=self.step,
                )
            body = _apply_wire_stability(body)
            try:
                enforce_context_capacity(self, body, endpoint=endpoint, key=key)
            except ContextCapacityError as error:
                if _one_snapshot_record(
                        "final_context_capacity", type(error).__name__,
                        json.dumps(body).encode()):
                    body = json.loads(one_snapshot_first_wire)
                else:
                    raise
            _apply_greenfact_at_send(
                self,
                body.get("messages"),
                validate=lambda: _greenfact_context_fits(self, body, endpoint, key),
            )
            body = _apply_final_byte_fit(body)
            wire_messages = body.get("messages")
            if isinstance(wire_messages, list):
                from kernel_stateframe_framefit5 import (
                    assemble_framefit5_request,
                    has_framefit5_state,
                    refresh_final_product_inputs,
                    validate_final_product_boundary,
                )
                is_stateframe_runtime = any(
                    base.__module__ == "kernel_stateframe_runtime"
                    and base.__name__ == "StateFrameRuntime"
                    for base in type(wire_truth_runtime).__mro__
                ) if wire_truth_runtime is not None else False
                if is_stateframe_runtime and has_framefit5_state(wire_messages):
                    framefit5_request = True
                    refresh_final_product_inputs(
                        wire_truth_runtime, wire_messages,
                        frame_request_allocation=self.frame_request_allocation,
                    )
                    from kernel_request_reprepare import _source_delivery_inputs
                    source_plan, source_contents = _source_delivery_inputs(
                        self, wire_truth_runtime, preparation,
                    )
                    body["messages"] = assemble_framefit5_request(
                        wire_messages,
                        wire_messages,
                        task_message=getattr(self, "_framefit5_task_message", None),
                        source_plan=source_plan,
                        sources=source_contents,
                        runtime=wire_truth_runtime,
                        owner_contract=getattr(self, "request_action_contract", None),
                        frame_request_allocation=self.frame_request_allocation,
                    )
                    framefit5_assembler_messages = json.dumps(
                        body["messages"],
                    ).encode("utf-8")
                    validate_final_product_boundary(
                        wire_truth_runtime,
                        body["messages"],
                        body.get("tools"),
                        refresh_sources=False,
                        refresh_headcarry=False,
                        frame_request_allocation=self.frame_request_allocation,
                    )
            wire = json.dumps(body).encode()
            if _boundary_state is not None:
                _boundary_state.phase = "final_serialization_validation"
                _boundary_state.body_digest = "sha256:" + hashlib.sha256(wire).hexdigest()
            try:
                self._assert_wire_string_hygiene(wire)
            except RequestPreparationContractError as error:
                if _one_snapshot_record(
                        "final_wire_string_hygiene", type(error).__name__, wire):
                    wire = bytes(one_snapshot_first_wire)
                    body = json.loads(wire)
                else:
                    raise
            body = json.loads(wire)
            try:
                if _boundary_state is not None:
                    _boundary_state.phase = "final_owner_validation"
                request_extensions = self._request_capture_pre_wire_extensions(
                    body, endpoint, key, deadline=deadline, side_call_contract=side_call_contract)
            except RequestPreparationContractError as error:
                if not _one_snapshot_record(
                        "final_request_preparation", type(error).__name__, wire):
                    try:
                        _stop_pre_dispatch_contract(
                            self, event="request_final_validation_stopped",
                            code="final_request_preparation_failed",
                            detail="final request preparation failed after bounded recovery",
                            failures=(type(error).__name__,), error=error,
                        )
                    except ProviderBoundaryContractError as stop:
                        raise stop from error
            if json.dumps(body).encode() != wire:
                if _one_snapshot_record("final_request_bytes", "BYTES_CHANGED", wire):
                    first_wire = getattr(self, "_one_snapshot_first_wire", None)
                    if _one_snapshot.has_sendable_body(first_wire):
                        wire = bytes(first_wire)
                        body = json.loads(wire)
                else:
                    _stop_pre_dispatch_contract(
                        self, event="request_final_validation_stopped",
                        code="final_request_bytes_changed",
                        detail="final authority changed qualified serialized request",
                    )
            try:
                visibility_receipt = verify_final_visibility(wire, visibility_check)
            except Exception as error:
                if _one_snapshot_record(
                        "final_visibility_validation", type(error).__name__, wire):
                    visibility_receipt = {
                        "result": "DEGRADED",
                        "missing_ranges": [],
                        "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest(),
                    }
                else:
                    raise
            if not framefit5_request:
                self.journal.emit(
                    "read_reuse_visibility_checked", producer="kernel_read_reuse_visibility",
                    request_id=self.request_id, step=self.step,
                    result=visibility_receipt["result"],
                    missing_ranges=visibility_receipt["missing_ranges"],
                    pins_added=visibility_check.get("pins_added", 0),
                    candidate_statuses=visibility_check.get("candidate_statuses", ()),
                    body_sha256=visibility_receipt["body_sha256"], dispatch=False,
                )
            if (not framefit5_request
                    and isinstance(self.request_action_contract, Mapping)
                    and wire_truth_runtime is not None):
                from kernel_final_wire_budget import reconcile_frozen_budget
                current_digest = "sha256:" + hashlib.sha256(wire).hexdigest()
                frozen_inputs = (
                    wire_truth_revalidation_baseline[2]
                    if wire_truth_revalidation_baseline is not None
                    and wire_truth_revalidation_baseline[0] == current_digest else None
                )
                # B1403/B1421 class: identical bytes revalidate as a pure function of the
                # wire-truth snapshot, never of state the extensions moved afterwards.
                try:
                    final_budget_receipt = reconcile_frozen_budget(
                        self, wire_truth_runtime, body, frozen_inputs=frozen_inputs,
                    )
                    if wire_truth_revalidation_baseline is not None:
                        prior_digest, prior_signature, _ = wire_truth_revalidation_baseline
                        if current_digest == prior_digest:
                            from kernel_final_wire_budget import (
                                record_wire_truth_revalidation_consistency,
                            )
                            final_budget_receipt = record_wire_truth_revalidation_consistency(
                                self,
                                previous_digest=prior_digest,
                                previous_signature=prior_signature,
                                current_digest=current_digest,
                                receipt=final_budget_receipt,
                            )
                except Exception as error:
                    if not _one_snapshot_record(
                            "final_budget_revalidation", type(error).__name__, wire):
                        raise
            self._wire_truth_context = _capture_final_truth_context(body)
            try:
                if _boundary_state is not None:
                    _boundary_state.phase = "final_source_proof_validation"
                wire = qualify_source(
                    self, wire_truth_runtime, preparation, body, wire,
                    allow_requalification=True,
                    framefit5_final_assembly_pending=(
                        framefit5_request
                    ),
                )
            except RequestPreparationContractError as error:
                if not _one_snapshot_record(
                        "final_source_proof", type(error).__name__, wire):
                    try:
                        _stop_pre_dispatch_contract(
                            self, event="request_final_validation_stopped",
                            code="final_request_source_unqualified",
                            detail="final request source delivery remained unqualified after bounded recovery",
                            failures=(type(error).__name__,), error=error,
                        )
                    except ProviderBoundaryContractError as stop:
                        raise stop from error
            final_budget = getattr(self, "request_budget_final_wire", None)
            if (not framefit5_request
                    and isinstance(final_budget, Mapping)
                    and final_budget.get("final_validation_failures")):
                failures = tuple(str(item) for item in final_budget["final_validation_failures"])
                self.journal.emit(
                    "request_final_validation_blocked", producer="kernel_measurement",
                    request_id=self.request_id, step=self.step, dispatch=False,
                    failures=list(failures), outcome=HARNESS_CONTRACT,
                    failure_origin="harness",
                )
                if not _one_snapshot_record(
                        "final_budget_revalidation", ",".join(failures), wire):
                    _stop_pre_dispatch_contract(
                        self, event="request_final_validation_stopped",
                        code="final_request_surface_unqualified",
                        detail="final request surface failed authoritative validation",
                        failures=failures,
                    )
            one_snapshot_final_check_blocked = False
            if not framefit5_request:
                try:
                    if _boundary_state is not None:
                        _boundary_state.phase = "final_wire_truth_validation"
                    with profile_host_span(
                            self, "final_wire_truth_validation", request_id=self.request_id):
                        wire_truth_check = check_wire(wire, self._wire_truth_context,
                            emit=self.journal.emit, attempt=1, allow_done_repair=False)
                except WireTruthBlocked as error:
                    if _one_snapshot_record(
                            "final_wire_truth_validation", "BLOCKED", wire):
                        from kernel_wire_truth_gate import WireTruthResult, schema_digest
                        wire_truth_check = WireTruthResult(
                            request_id=self.request_id,
                            revision=str(self._wire_truth_context.get("revision", "UNKNOWN")),
                            body_sha256=(error.body_sha256 or
                                "sha256:" + hashlib.sha256(wire).hexdigest()),
                            schema_digest=schema_digest(body.get("tools")),
                            verdict="BLOCKED",
                            failed=tuple(getattr(error, "failed", ())),
                            body=copy.deepcopy(body),
                            wire=bytes(wire),
                            attempt=1,
                        )
                        one_snapshot_final_check_blocked = True
                    else:
                        _stop_pre_dispatch_contract(
                            self, event="request_final_wire_truth_stopped",
                            code="final_wire_truth_unqualified",
                            detail="final request wire remained blocked after bounded recovery",
                            failures=tuple(str(value) for value in getattr(error, "failed", ())), error=error,
                        )
                first_check_digest = getattr(one_snapshot_first_check, "body_sha256", None)
                if (not one_snapshot_final_check_blocked
                        and one_snapshot_enabled and one_snapshot_first_check is not None
                        and (wire_truth_check.verdict != one_snapshot_first_verdict
                             or wire_truth_check.body_sha256 != first_check_digest)):
                    _one_snapshot_record(
                        "final_wire_truth_validation", str(wire_truth_check.verdict), wire,
                    )
                if wire_truth_check.wire != wire or wire_truth_check.verdict not in {"PASS", "DISABLED"}:
                    if (not one_snapshot_final_check_blocked
                            and not _one_snapshot_record(
                                "final_wire_truth_validation", str(wire_truth_check.verdict), wire)):
                        _stop_pre_dispatch_contract(
                            self, event="request_final_wire_truth_stopped",
                            code="final_wire_truth_unqualified",
                            detail="final request wire did not pass strict qualification",
                            failures=(str(wire_truth_check.verdict),),
                        )
            else:
                # FrameFit5 recovery is checked after the final shared assembly below.
                wire_truth_check = one_snapshot_first_check
            if one_snapshot_enabled and getattr(self, "_one_snapshot_degraded_validation", None):
                first_wire = getattr(self, "_one_snapshot_first_wire", None)
                if _one_snapshot.has_sendable_body(first_wire):
                    wire = bytes(first_wire)
                    body = json.loads(wire)
                    if one_snapshot_first_context is not None:
                        self._wire_truth_context = one_snapshot_first_context
                    if one_snapshot_first_check is not None:
                        wire_truth_check = one_snapshot_first_check
                    visibility_check = initial_visibility_check
                    try:
                        visibility_receipt = verify_final_visibility(
                            wire, initial_visibility_check,
                        )
                    except Exception as error:
                        _one_snapshot_record(
                            "first_snapshot_visibility", type(error).__name__, wire,
                        )
                        visibility_receipt = _one_snapshot.degraded_visibility_receipt(wire)
                    if one_snapshot_first_request_extensions is not None:
                        request_extensions = dict(one_snapshot_first_request_extensions)
                    if one_snapshot_first_action_contract_captured:
                        from kernel_request_tool_surface import restore_authored_action_contract
                        restore_authored_action_contract(self, one_snapshot_first_action_contract)
                    # Chunk reuse is a body-specific optimization configured before
                    # final wire checks; drop it if the first qualified bytes are restored.
                    kv_chunk_reuse_transport = None
            final_messages = body.get("messages")
            from kernel_stateframe_framefit5 import (
                assemble_framefit5_request,
                has_framefit5_state,
                refresh_final_product_inputs,
                validate_final_product_boundary,
            )
            is_stateframe_runtime = any(
                base.__module__ == "kernel_stateframe_runtime"
                and base.__name__ == "StateFrameRuntime"
                for base in type(wire_truth_runtime).__mro__
            ) if wire_truth_runtime is not None else False
            if (is_stateframe_runtime and isinstance(final_messages, list)
                    and has_framefit5_state(final_messages)):
                framefit5_request = True
                refresh_final_product_inputs(
                    wire_truth_runtime, final_messages,
                    frame_request_allocation=self.frame_request_allocation,
                )
                from kernel_request_reprepare import _source_delivery_inputs
                source_plan, source_contents = _source_delivery_inputs(
                    self, wire_truth_runtime, preparation,
                )
                body["messages"] = assemble_framefit5_request(
                    final_messages,
                    final_messages,
                    task_message=getattr(self, "_framefit5_task_message", None),
                    source_plan=source_plan,
                    sources=source_contents,
                    runtime=wire_truth_runtime,
                    owner_contract=getattr(self, "request_action_contract", None),
                    frame_request_allocation=self.frame_request_allocation,
                )
                framefit5_assembler_messages = json.dumps(
                    body["messages"],
                ).encode("utf-8")
                validate_final_product_boundary(
                    wire_truth_runtime, body["messages"], body.get("tools"),
                    refresh_sources=False,
                    refresh_headcarry=False,
                        frame_request_allocation=self.frame_request_allocation,
                )
            elif framefit5_request:
                raise RequestPreparationContractError(
                    "final FrameFit5 recovery lost its shared frame state")

            if framefit5_request:
                wire = json.dumps(body).encode()
                try:
                    self._assert_wire_string_hygiene(wire)
                except RequestPreparationContractError as error:
                    _stop_pre_dispatch_contract(
                        self, event="request_final_validation_stopped",
                        code="final_wire_string_hygiene_failed",
                        detail="final assembled request failed wire string hygiene",
                        failures=(type(error).__name__,), error=error,
                    )
                body = json.loads(wire)
                try:
                    visibility_receipt = verify_final_visibility(wire, visibility_check)
                except Exception as error:
                    _stop_pre_dispatch_contract(
                        self, event="request_final_validation_stopped",
                        code="final_visibility_unqualified",
                        detail="final assembled request failed source visibility validation",
                        failures=(type(error).__name__,), error=error,
                    )
                self.journal.emit(
                    "read_reuse_visibility_checked",
                    producer="kernel_read_reuse_visibility",
                    request_id=self.request_id,
                    step=self.step,
                    result=visibility_receipt["result"],
                    missing_ranges=visibility_receipt["missing_ranges"],
                    pins_added=visibility_check.get("pins_added", 0),
                    candidate_statuses=visibility_check.get("candidate_statuses", ()),
                    body_sha256=visibility_receipt["body_sha256"],
                    dispatch=False,
                )
                self._wire_truth_context = _capture_final_truth_context(body)
                try:
                    wire = qualify_source(
                        self, wire_truth_runtime, preparation, body, wire,
                        allow_requalification=False,
                        framefit5_preassembled=True,
                    )
                except RequestPreparationContractError as error:
                    _stop_pre_dispatch_contract(
                        self, event="request_final_validation_stopped",
                        code="final_request_source_unqualified",
                        detail="final assembled source delivery remained unqualified",
                        failures=(type(error).__name__,), error=error,
                    )
                if json.dumps(body).encode() != wire:
                    _stop_pre_dispatch_contract(
                        self, event="request_final_validation_stopped",
                        code="final_request_bytes_changed",
                        detail="final source proof changed the shared assembled request",
                        failures=("final_request_bytes_changed",),
                    )
                if (isinstance(self.request_action_contract, Mapping)
                        and wire_truth_runtime is not None):
                    from kernel_final_wire_budget import (
                        reconcile_frozen_budget,
                        snapshot_revalidation_inputs,
                    )
                    current_digest = "sha256:" + hashlib.sha256(wire).hexdigest()
                    frozen_inputs = (
                        wire_truth_revalidation_baseline[2]
                        if wire_truth_revalidation_baseline is not None
                        and wire_truth_revalidation_baseline[0] == current_digest
                        else snapshot_revalidation_inputs(self, wire_truth_runtime, body)
                    )
                    try:
                        final_budget_receipt = reconcile_frozen_budget(
                            self, wire_truth_runtime, body, frozen_inputs=frozen_inputs,
                        )
                        if (wire_truth_revalidation_baseline is not None
                                and wire_truth_revalidation_baseline[0] == current_digest):
                            from kernel_final_wire_budget import (
                                record_wire_truth_revalidation_consistency,
                            )
                            final_budget_receipt = record_wire_truth_revalidation_consistency(
                                self,
                                previous_digest=wire_truth_revalidation_baseline[0],
                                previous_signature=wire_truth_revalidation_baseline[1],
                                current_digest=current_digest,
                                receipt=final_budget_receipt,
                            )
                    except Exception as error:
                        _stop_pre_dispatch_contract(
                            self, event="request_final_validation_stopped",
                            code="final_budget_revalidation_failed",
                            detail="final assembled request budget could not be revalidated",
                            failures=(type(error).__name__,), error=error,
                        )
                    failures = tuple(
                        str(item) for item in final_budget_receipt.get(
                            "final_validation_failures", (),
                        )
                    )
                    if failures:
                        self.journal.emit(
                            "request_final_validation_blocked",
                            producer="kernel_measurement",
                            request_id=self.request_id,
                            step=self.step,
                            dispatch=False,
                            failures=list(failures),
                            outcome=HARNESS_CONTRACT,
                            failure_origin="harness",
                        )
                        _stop_pre_dispatch_contract(
                            self, event="request_final_validation_stopped",
                            code="final_request_surface_unqualified",
                            detail="final assembled request failed authoritative validation",
                            failures=failures,
                        )
                self._wire_truth_context = _capture_final_truth_context(body)
                same_qualified_wire = (
                    wire_truth_check is not None
                    and getattr(wire_truth_check, "wire", None) == wire
                    and getattr(wire_truth_check, "verdict", None) in {"PASS", "DISABLED"}
                )
                if not same_qualified_wire:
                    try:
                        if _boundary_state is not None:
                            _boundary_state.phase = "final_framefit5_wire_truth_validation"
                        with profile_host_span(
                                self, "final_wire_truth_validation",
                                request_id=self.request_id):
                            wire_truth_check = check_wire(
                                wire,
                                self._wire_truth_context,
                                emit=self.journal.emit,
                                attempt=1,
                                allow_done_repair=False,
                            )
                    except WireTruthBlocked as error:
                        _stop_pre_dispatch_contract(
                            self, event="request_final_wire_truth_stopped",
                            code="final_wire_truth_unqualified",
                            detail="final assembled request remained wire-truth blocked",
                            failures=tuple(
                                str(value) for value in getattr(error, "failed", ())
                            ),
                            error=error,
                        )
                if (wire_truth_check.wire != wire
                        or wire_truth_check.verdict not in {"PASS", "DISABLED"}):
                    _stop_pre_dispatch_contract(
                        self, event="request_final_wire_truth_stopped",
                        code="final_wire_truth_unqualified",
                        detail="final assembled request did not pass strict wire-truth validation",
                        failures=(str(wire_truth_check.verdict),),
                    )
        self._wire_truth_check = wire_truth_check
        if self.endpoint_profile is not None:
            # Admission is a pure measured-profile calculation, not a lease.
            # Bind its receipt to the selected final body without extending the
            # original absolute deadline or acquiring another reservation.
            final_admission = admission_for_request(profile, _prompt_token_estimate(body),
                slots_processing=slots_processing, queue_time_seconds=queue_time)
            if (final_admission.status is not AdmissionStatus.ADMITTED
                    or final_admission.deadline_seconds is None):
                self.journal.emit("request_not_admitted", request_id=self.request_id,
                    step=self.step, admission=final_admission.mapping(), dispatch=False)
                raise EndpointAdmissionError(final_admission)
            admission = final_admission
            deadline = min(deadline, time.monotonic() + admission.deadline_seconds)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("request deadline expired before submission")
            socket_timeout = min(socket_timeout, remaining)
            wall_timeout = min(wall_timeout, remaining)
        if self.ticket_identity is not None:
            from kernel_request_evidence_closure import TicketIdentity, final_wire_ticket_receipt
            identity = TicketIdentity(
                str(self.ticket_identity["task_id"]),
                str(self.ticket_identity["text"]),
            )
            if self.ticket_identity.get("ticket_sha256") != identity.sha256:
                raise RuntimeError("authorized ticket identity changed before final wire")
            ticket_receipt = final_wire_ticket_receipt(
                identity,
                body.get("messages", []) if isinstance(body.get("messages"), list) else [],
                body.get("tools", ()),
                retrieval_ticket_sha256=self.ticket_identity.get("retrieval_ticket_sha256"),
                model_brief_ticket_sha256=self.ticket_identity.get("model_brief_ticket_sha256"),
                wire_bytes=wire,
            )
            self.journal.emit("ticket_final_wire_receipt", producer="kernel_measurement",
                              request_id=self.request_id, step=self.step,
                              **_journal_safe_fields(ticket_receipt), dispatch=False)
            if ticket_receipt["delivered"] is not True:
                raise RuntimeError("authorized ticket was not delivered to the final model wire")
        content = "\n".join(str(message.get("content") or "")
                            for message in body.get("messages", []))
        manifest = self.journal.manifest
        configured_identity = {name: manifest.get(name, UNKNOWN) for name in (
            "model", "server_build", "tokenizer", "chat_template", "cache_policy")}
        backend_identity = configured_identity
        if model_router is not None:
            # The run manifest describes the configured host.  A routed call
            # must separately record the effective transport/model.  Unknown
            # backend metadata is explicit rather than stale manifest data.
            decision = getattr(self, "model_router_decision", None)
            adapter = getattr(getattr(decision, "endpoint", None), "adapter", None)
            backend_identity = {
                "configured_identity": configured_identity,
                "model": getattr(self, "model_router_effective_model_id",
                                  body.get("model", manifest.get("model"))),
                "server_build": "UNKNOWN",
                "tokenizer": "UNKNOWN",
                "chat_template": getattr(adapter, "chat_template", "UNKNOWN"),
                "cache_policy": "UNKNOWN",
                "effective_endpoint": endpoint,
                "router_endpoint": getattr(self, "model_router_selected_endpoint", None),
                "router_applied": bool(getattr(self, "model_router_applied", False)),
            }
        information = getattr(self.journal, "information_policy", None)
        decode_depth_per_arm_on = os.environ.get(
            "LONG_HORIZON_DECODE_DEPTH_PER_ARM", "off"
        ).strip().lower() in {"on", "1", "true", "yes"}
        if decode_depth_per_arm_on and not hasattr(self, "decode_depth_per_arm_slots_busy"):
            from decode_depth_per_arm import observe_slots
            observe_slots(self, endpoint, deadline=deadline)
        if os.environ.get("LONG_HORIZON_TELEMETRY", "off").strip().lower() in {
                "on", "1", "true", "yes"} and not hasattr(self, "decode_depth_slots_busy"):
            from long_horizon_decode_logging import observe_slots
            observe_slots(self, endpoint, deadline=deadline)
        cost_meter = getattr(self.journal, "host_cost_meter", None)
        host_spans = getattr(self.journal, "host_spans", None)

        def provider_call():
            if self.qualified_request is not None:
                self.qualified_request.validate(self, wire_truth_runtime, body)
                if wire != self.qualified_request.wire:
                    from kernel_request_seal import RequestSealError
                    raise RequestSealError("transport bytes differ from the qualified request")
            _assert_framefit5_provider_boundary(
                self, wire, framefit5_assembler_messages,
            )
            if _boundary_state is not None:
                _boundary_state.phase = "provider_transport_validation"
                _boundary_state.body_digest = "sha256:" + hashlib.sha256(wire).hexdigest()
            try:
                return self._send_transport_with_retries(
                    body, wire, endpoint, key, admission=admission,
                    socket_timeout=socket_timeout, deadline=deadline,
                    wall_timeout=wall_timeout, backend_identity=backend_identity,
                    request_extensions=request_extensions,
                    kv_chunk_reuse_transport=kv_chunk_reuse_transport,
                    required=required, permit_prefix=permit_prefix,
                    transport_attempt_limit=transport_attempt_limit,
                    journal_timeouts=journal_timeouts,
                    boundary_state=_boundary_state,
                    frozen_retry_only=_frozen_retry_only)
            finally:
                from kernel_completion_owner import finish_request_submission
                finish_request_submission(wire_truth_runtime, request_id=self.request_id)

        try:
            if isinstance(cost_meter, PhaseMeter):
                with cost_meter.measure(
                    "request_delivery", count=1, items=min(len(wire), 10_000_000)
                ):
                    if isinstance(host_spans, HostSpanRecorder):
                        with host_spans.measure("provider"):
                            response, sent, started = provider_call()
                    else:
                        response, sent, started = provider_call()
            else:
                if isinstance(host_spans, HostSpanRecorder):
                    with host_spans.measure("provider"):
                        response, sent, started = provider_call()
                else:
                    response, sent, started = provider_call()
        finally:
            self._emit_host_span_profile(body, wire)
        delivered_sources = visibility_receipt.get("delivered_sources", ())
        if (wire_truth_runtime is not None
                and isinstance(delivered_sources, Sequence)
                and not isinstance(delivered_sources, (str, bytes, bytearray))
                and delivered_sources):
            from kernel_tokenlab_stall_breaker import note_provider_source_delivery
            note_provider_source_delivery(
                wire_truth_runtime, delivered_sources, request_id=self.request_id,
            )
        source_runtime = getattr(self, "completion_owner_runtime", None)
        final_budget = getattr(self, "request_budget_final_wire", None)
        source_delivery = (
            final_budget.get("source_delivery")
            if isinstance(final_budget, Mapping) else None
        )
        submitted_messages = body.get("messages")
        if (source_runtime is not None and isinstance(submitted_messages, list)
                and isinstance(source_delivery, Mapping)):
            from kernel_request_source_window import record_verified_source_messages
            record_verified_source_messages(source_runtime, submitted_messages, source_delivery)
        if information is not None and information.payload and not information.off:
            delivered = information.payload[:1200]
            if delivered in content:
                event = self.journal.emit("information_exposure", channel="initial_oracle_symptom",
                                          request_id=self.request_id, submitted_event_id=sent["event_id"],
                                          payload_hash=information.manifest()["symptom_payload_hash"],
                                          delivered_text=delivered, delivered_characters=len(delivered))
                information.exposures.append(event["event_id"])
                information.initial_exposures.append(event["event_id"])
        if information is not None:
            for delivered in information.live_payloads:
                if delivered in content:
                    event = self.journal.emit("information_exposure", channel="live_grader_feedback",
                                              feedback_policy=information.feedback,
                                              request_id=self.request_id, submitted_event_id=sent["event_id"],
                                              delivered_text=delivered)
                    information.exposures.append(event["event_id"])
                    information.live_exposures.append(event["event_id"])
        try:
            # The builder seals the *draft* input and publishes a different final
            # wire. A degeneration retry re-enters the builder with that same
            # draft so its input seal can validate and return the frozen wire.
            retry_input = (
                _boundary_state.authoritative_body
                if self.qualified_request is not None and _boundary_state is not None
                else body
            )
            discarded_retry = self._v43g_retry_degenerate(
                response, retry_input, endpoint, key, deadline=deadline,
                side_call_contract=side_call_contract)
            if discarded_retry is not None:
                return discarded_retry
            include_prompt_cache = False
            try:
                from kernel_prompt_cache import prompt_cache_enabled
                include_prompt_cache = prompt_cache_enabled()
            except ImportError:
                # Standalone historical utilities may use this capture without
                # the optional prompt-cache module in their import path.
                pass
            accounting = backend_accounting(
                response, include_prompt_cache=include_prompt_cache)
            # Keep the shared response-extension seam after accounting so the
            # provider owner composes with r120/r121's generic capture path.
            response_extensions = self._request_capture_response_extensions(
                response, tools=body.get("tools"))
            if os.environ.get("STATEFRAME_DEGENERATION_GUARD", "off").strip().lower() == "on":
                response_extensions["logical_token_accounting"] = {
                    "prompt_tokens": self.v43g_logical_prompt_tokens,
                    "completion_tokens": self.v43g_logical_completion_tokens,
                    "total_tokens": (self.v43g_logical_prompt_tokens
                                      + self.v43g_logical_completion_tokens),
                    "attempts_including_discarded": 2 if self.v43g_degradation_retry_used else 1,
                }
            decode_depth_per_arm_row = None
            if decode_depth_per_arm_on:
                from decode_depth_per_arm import response_record
                _decode_elapsed_ns = time.perf_counter_ns() - started
                _decode_event = {
                    "kind": "response_received", "request_id": self.request_id,
                    "run_id": self.journal.run_id,
                    "transport_attempt_id": sent["event_id"], "step": self.step,
                    "elapsed_ns": _decode_elapsed_ns, "response": response,
                    "accounting": accounting,
                    "task_id": os.environ.get("LONG_HORIZON_TASK_ID", self.journal.task_id),
                    "arm": os.environ.get("LONG_HORIZON_ARM"),
                    "seed": int(os.environ["REACT_SEED"])
                    if os.environ.get("REACT_SEED", "").isdigit() else None,
                    "request_role": self.long_horizon_request_role,
                    "decode_depth_per_arm": getattr(
                        self, "decode_depth_per_arm_request_metadata", None),
                }
                decode_depth_per_arm_row = response_record(
                    _decode_event,
                    capture=self,
                )
            cache_prefill_row = _cache_prefill_response_row(
                self, accounting, response=response, request=body)
            # One canonical response_received per response: every lane-owned
            # payload is a nested field of that single event (rule #211).
            response_extensions = {
                **response_extensions,
                **({"decode_depth_per_arm": decode_depth_per_arm_row}
                   if decode_depth_per_arm_row is not None else {}),
                **({"cache_prefill_economics": cache_prefill_row}
                   if cache_prefill_row is not None else {}),
            }
            if os.environ.get("LONG_HORIZON_TELEMETRY", "off").strip().lower() in {
                    "on", "1", "true", "yes"}:
                from long_horizon_metrics import response_record
                _elapsed_ns = time.perf_counter_ns() - started
                _telemetry_event = {
                    "kind": "response_received", "request_id": self.request_id,
                    "transport_attempt_id": sent["event_id"], "step": self.step,
                    "elapsed_ns": _elapsed_ns, "response": response,
                    "accounting": accounting,
                    "task_id": os.environ.get("LONG_HORIZON_TASK_ID", self.journal.task_id),
                    "arm": os.environ.get("LONG_HORIZON_ARM"),
                    "seed": int(os.environ["REACT_SEED"])
                    if os.environ.get("REACT_SEED", "").isdigit() else None,
                    "request_role": self.long_horizon_request_role,
                    "rendered_prompt_tokens": self.long_horizon_rendered_prompt_tokens,
                    "rendered_prompt_tokens_source": (
                        self.long_horizon_rendered_prompt_tokens_source
                    ),
                    "depth_source": self.long_horizon_rendered_prompt_tokens_source,
                    "logical_depth_tokens": self.long_horizon_logical_depth_tokens,
                    "compaction_ordinal": self.long_horizon_compaction_ordinal,
                    "usable_input_tokens": int(
                        os.environ["LONG_HORIZON_USABLE_INPUT_TOKENS"]
                    )
                    if os.environ.get("LONG_HORIZON_USABLE_INPUT_TOKENS", "").isdigit()
                    else None,
                    "context_window_tokens": int(
                        os.environ["LONG_HORIZON_CONTEXT_LIMIT_TOKENS"]
                    )
                    if os.environ.get("LONG_HORIZON_CONTEXT_LIMIT_TOKENS", "").isdigit()
                    else None,
                    "decode_depth": {
                        "rendered_context_depth": getattr(
                            self, "decode_depth_rendered_context_depth", None),
                        "rendered_context_depth_source": getattr(
                            self, "decode_depth_rendered_context_depth_source", None),
                        "slots_busy": getattr(self, "decode_depth_slots_busy", None),
                        "slots_busy_source": getattr(self, "decode_depth_slots_busy_source", None),
                    },
                }
                _telemetry = response_record(
                    _telemetry_event,
                    metadata={
                        "task_id": _telemetry_event["task_id"],
                        "arm": _telemetry_event["arm"],
                        "seed": _telemetry_event["seed"],
                        "usable_input_tokens": int(os.environ["LONG_HORIZON_USABLE_INPUT_TOKENS"])
                        if os.environ.get("LONG_HORIZON_USABLE_INPUT_TOKENS", "").isdigit()
                        else None,
                    },
                )
                # long_horizon_metrics.py is the shared offline contract and
                # intentionally remains unaware of lane-local run counters.
                # Annotate its compact row here, after response_record returns,
                # so every canonical response_received row carries the live
                # mechanism/depth provenance.
                _history_ordinal = max(
                    int(getattr(self, "history_summary_ordinal", 0) or 0),
                    int(getattr(self, "long_horizon_history_summary_ordinal", 0) or 0),
                )
                _frame_ordinal = max(
                    int(getattr(self, "frame_epoch_ordinal", 0) or 0),
                    int(getattr(self, "long_horizon_frame_epoch_ordinal", 0) or 0),
                )
                _hybrid_crossing_request_id = (
                    getattr(self, "hybrid_crossing_request_id", None)
                    or getattr(self, "long_horizon_hybrid_crossing_request_id", None))
                _hybrid_crossing = bool(
                    getattr(self, "hybrid_crossing", False)
                    or getattr(self, "long_horizon_hybrid_crossing", False))
                _phase = max(
                    int(getattr(self, "phase", 1) or 1),
                    int(getattr(self, "long_horizon_phase", 1) or 1),
                )
                _presentation = getattr(self, "presentation", None)
                _long_presentation = getattr(
                    self, "long_horizon_presentation", "history")
                if (not isinstance(_presentation, str)
                        or (_presentation == "history"
                            and _long_presentation not in {None, "history"})):
                    _presentation = _long_presentation or "history"
                _cumulative = getattr(self, "cumulative_logical_depth_tokens", None)
                if _cumulative is None:
                    _cumulative = getattr(
                        self, "long_horizon_cumulative_logical_depth_tokens", None)
                _cumulative_source = getattr(
                    self, "cumulative_logical_depth_source", None)
                if _cumulative_source is None:
                    _cumulative_source = getattr(
                        self, "long_horizon_cumulative_logical_depth_source", None)
                _telemetry.update({
                    "history_summary_ordinal": _history_ordinal,
                    "frame_epoch_ordinal": _frame_ordinal,
                    # Legacy field: the combined mechanism count.
                    "compaction_ordinal": _history_ordinal + _frame_ordinal,
                    "hybrid_crossed": bool(
                        getattr(self, "hybrid_crossed", False)
                        or getattr(self, "long_horizon_hybrid_crossed", False)),
                    "hybrid_crossing_request_id": _hybrid_crossing_request_id,
                    "hybrid_crossing": _hybrid_crossing,
                    "phase": _phase,
                    "presentation": _presentation,
                    "cumulative_logical_depth_tokens": _cumulative,
                    "cumulative_logical_depth_source": _cumulative_source,
                })
                received = self.journal.emit(
                    "response_received", request_id=self.request_id,
                    transport_attempt_id=sent["event_id"], step=self.step,
                    elapsed_ns=_elapsed_ns, response=response,
                    accounting=accounting, long_horizon=_telemetry,
                    # decode_depth_logging owns this canonical response_received
                    # envelope; provider, per-arm and cache-economics payloads
                    # are additive nested fields.
                    **_journal_safe_fields(response_extensions))
                from long_horizon_decode_logging import record_response
                record_response(self.journal, {
                    **_telemetry,
                    "event_id": received["event_id"],
                    "timestamp_ns": received.get("timestamp_ns"),
                    "sequence": received.get("sequence"),
                })
            else:
                # Keep the default event construction byte-for-byte identical
                # to the pre-protocol path.
                received = self.journal.emit(
                    "response_received", request_id=self.request_id,
                    transport_attempt_id=sent["event_id"], step=self.step,
                    elapsed_ns=time.perf_counter_ns() - started, response=response,
                    accounting=accounting, **_journal_safe_fields(response_extensions))
            delivery_runtime = (getattr(self, "_wire_truth_runtime", None)
                                or getattr(self, "completion_owner_runtime", None))
            if delivery_runtime is not None:
                from kernel_step_contract import commit_environment_notice_delivery
                commit_environment_notice_delivery(
                    delivery_runtime, getattr(self, "_evictfix_final_wire_messages", ()))
            self._evictfix_final_wire_messages = None
            from kernel_cache_layout_candidate import commit_after_response
            commit_after_response(self)
            if decode_depth_per_arm_row is not None:
                from decode_depth_per_arm import record_response
                record_response(self.journal, decode_depth_per_arm_row)
            if self.parent_meter is not None:
                self.parent_meter.record_response(
                    permit_id=sent.get("provider_permit_id"), journal=self.journal)
            self.accounting = accounting
            if cache_prefill_row is not None:
                from cache_prefill_economics import record_response
                record_response(self.journal, {
                    **cache_prefill_row,
                    "response_event_id": received["event_id"],
                    "sequence": received.get("sequence"),
                    "timestamp_ns": received.get("timestamp_ns"),
                })
            if include_prompt_cache:
                from kernel_prompt_cache import record_response_accounting, request_arm
                record_response_accounting(
                    self.journal, accounting,
                    self.prompt_cache_arm or request_arm(body.get("messages", ())))
            self.response_event_id = received["event_id"]
            self.transport_attempt_id = sent["event_id"]
            choice = response["choices"][0]
            if isinstance(choice, Mapping):
                self.finish_reason = choice.get("finish_reason")
                self.output_cap_hit = self.output_cap_hit or self.finish_reason == "length"
            message = choice.get("message") if isinstance(choice, Mapping) else None
            if isinstance(message, Mapping):
                self.provider_response_message = copy.deepcopy(dict(message))
                raw_tool_calls = message.get("tool_calls")
                if isinstance(raw_tool_calls, Sequence) and not isinstance(
                        raw_tool_calls, (str, bytes, bytearray)):
                    self.provider_tool_calls = [
                        copy.deepcopy(dict(call)) for call in raw_tool_calls
                        if isinstance(call, Mapping)
                    ]
                self.response_reasoning_content = message.get("reasoning_content")
                self.response_content = message.get("content")
                prefill = self.stateframe_think_prefill_text
                if prefill and isinstance(self.response_reasoning_content, str):
                    from kernel_frame_continuation import strip_prefill
                    raw_reasoning = self.response_reasoning_content
                    self.response_reasoning_content = strip_prefill(raw_reasoning, prefill)
                    self.stateframe_think_prefill_echoed = raw_reasoning != self.response_reasoning_content
                    self.journal.emit(
                        "think_prefill_metrics", request_id=self.request_id,
                        prefill_text=prefill,
                        prefill_token_count=self.stateframe_think_prefill_tokens,
                        generated_token_exclusion="prompt_prefill",
                        prefill_region_not_model_reasoning=True,
                        plan_tokens=self.self_plan_plan_tokens,
                        echoed_by_backend=self.stateframe_think_prefill_echoed,
                        generated_reasoning_tokens=len(
                            self.response_reasoning_content.split()),
                        generated_action_tokens=len(
                            str(self.response_content or "").split()),
                        total_generated_including_plan=(
                            self.self_plan_plan_tokens
                            + len(self.response_reasoning_content.split())
                            + len(str(self.response_content or "").split())),
                    )
                if self.self_plan_plan_text:
                    plan_accounting = self.self_plan_plan_accounting
                    from kernel_prompt_accounting import accounting_tokens
                    plan_counts = accounting_tokens(plan_accounting)
                    thinking_counts = accounting_tokens(self.accounting)
                    plan_cache_n = plan_counts["cached"]
                    plan_prompt_n = plan_counts["uncached"]
                    plan_total = plan_counts["total"]
                    thinking_cache_n = thinking_counts["cached"]
                    thinking_prompt_n = thinking_counts["uncached"]
                    total_generated = (
                        self.self_plan_plan_tokens
                        + len(str(self.response_reasoning_content or "").split())
                        + len(str(self.response_content or "").split()))
                    self.accounting.update({
                        "plan_tokens": self.self_plan_plan_tokens,
                        "generated_cost_tokens": total_generated,
                        "total_generated_including_plan": total_generated,
                        "prefill_region_not_model_reasoning": True,
                    })
                    self.journal.emit(
                        "self_plan_pair_metrics", request_id=self.request_id,
                        plan_request_id=self.self_plan_plan_request_id,
                        plan_text=self.self_plan_plan_text,
                        plan_tokens=self.self_plan_plan_tokens,
                        total_generated_including_plan=total_generated,
                        plan_cache_n=plan_cache_n, plan_prompt_n=plan_prompt_n,
                        plan_prompt_tokens=plan_counts, thinking_prompt_tokens=thinking_counts,
                        thinking_cache_n=thinking_cache_n,
                        thinking_prompt_n=thinking_prompt_n,
                        cache_n=thinking_cache_n,
                        prompt_n=thinking_prompt_n,
                        cache_prompt=self.self_plan_pair_cache,
                        cache_reuse_proven=(
                            self.self_plan_pair_cache
                            and isinstance(plan_cache_n, int)
                            and isinstance(thinking_cache_n, int)
                            and isinstance(plan_total, int)
                            and thinking_cache_n >= plan_total),
                        prefill_region_not_model_reasoning=True,
                    )
            if getattr(self, "playbook_card_id", None):
                from kernel_playbook_selfplan import journal_adoption
                journal_adoption(self)
            if (getattr(self.journal, "generated_artifacts", None) is not None
                    or os.environ.get("REACT_NATIVE_TOOL_BATCHES", "0") == "1"
                    or self.journal.manifest.get("continuation_source_refresh") is True):
                choice = dict(choice, _kernel_response_event_id=received["event_id"])
            return choice
        except ProviderMalformedResponseError:
            # Shape failures are journaled at the provider boundary; never
            # relabel them as transport or semantic outcomes here.
            raise
        except Exception as exc:
            error = self.journal.emit("semantic_error", request_id=self.request_id,
                              transport_attempt_id=sent["event_id"], step=self.step,
                              error_type=type(exc).__name__,
                              outcome=SEMANTIC_ERROR,
                              failure_origin="provider",
                              prompt_tokens=admission.prompt_tokens if admission is not None else None,
                              measured_prefill_tokens_per_second=(
                                  admission.prefill_tokens_per_second if admission is not None else None),
                              derived_deadline_seconds=(
                                  admission.deadline_seconds if admission is not None else None),
                              admission=admission.mapping() if admission is not None else None,
                              elapsed_ns=time.perf_counter_ns() - started)
            self.semantic_errors.append(error)
            raise

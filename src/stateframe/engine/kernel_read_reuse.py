"""Pure, revision-bound planning for source-read reuse.

The planner never opens a path and never decides that a path is authorized by
its spelling.  Callers first provide an authorized content handle, then this
module compares immutable identity (scope, digest, exact byte range, decoding,
presentation, and workspace generation) before allowing cached bytes to stand
in for a fresh read.
"""
from __future__ import annotations

import hashlib
import json
import posixpath
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace


SCHEMA = "stateframe-read-reuse/1"
ORDINARY_READ = "ORDINARY_READ"
CACHED_BYTES = "CACHED_BYTES"
UNCHANGED_VISIBLE = "UNCHANGED_VISIBLE"
DENIED = "DENIED"
STALE = "STALE"
MAX_PATH_LENGTH = 1024
MAX_PRESENTATION_LENGTH = 128
MAX_WORKSPACE_SCOPE_LENGTH = 4096


def _digest(value: bytes | str) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical_path(value: object) -> str:
    path = str(value or "").replace("\\", "/")
    if (not path or len(path) > MAX_PATH_LENGTH or path.startswith("/")
            or "\x00" in path):
        raise ValueError("source path must be a bounded workspace-relative path")
    raw_parts = path.split("/")
    # Reject traversal before normalisation.  Normalising first would turn
    # ``../public.py`` into a plausible in-workspace identity.
    if any(part == ".." for part in raw_parts):
        raise ValueError("source path escapes the workspace")
    while raw_parts and raw_parts[0] == ".":
        raw_parts.pop(0)
    if not raw_parts or any(part in {"", "."} for part in raw_parts):
        raise ValueError("source path is not canonical")
    normalized = posixpath.normpath("/".join(raw_parts))
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise ValueError("source path escapes the workspace")
    return normalized


def _bounded_text(value: object, label: str) -> str:
    result = str(value or "").strip()
    # Absolute workspace roots legitimately exceed compact presentation IDs,
    # particularly macOS temporary continuation workspaces.
    limit = MAX_WORKSPACE_SCOPE_LENGTH if label == "workspace scope" else MAX_PRESENTATION_LENGTH
    if not result or len(result) > limit:
        raise ValueError(f"{label} must be a bounded non-empty value")
    return result


@dataclass(frozen=True, slots=True)
class ReadKey:
    """The complete identity of one exact authorized byte span."""

    canonical_path: str
    workspace_scope: str
    content_digest: str
    byte_start: int
    byte_end: int
    decoding: str = "utf-8"
    presentation_version: str = "read-page/1"
    workspace_generation: str = "0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_path", _canonical_path(self.canonical_path))
        object.__setattr__(self, "workspace_scope", _bounded_text(self.workspace_scope, "workspace scope"))
        object.__setattr__(self, "content_digest", _bounded_text(self.content_digest, "content digest"))
        if type(self.byte_start) is not int or type(self.byte_end) is not int:
            raise TypeError("read byte bounds must be integers")
        if self.byte_start < 0 or self.byte_end < self.byte_start:
            raise ValueError("read byte range is invalid")
        if len(self.content_digest) < 8:
            raise ValueError("content digest is malformed")
        object.__setattr__(self, "decoding", _bounded_text(self.decoding, "decoding"))
        object.__setattr__(self, "presentation_version",
                           _bounded_text(self.presentation_version, "presentation version"))
        object.__setattr__(self, "workspace_generation",
                           _bounded_text(self.workspace_generation, "workspace generation"))

    @property
    def byte_range(self) -> tuple[int, int]:
        return self.byte_start, self.byte_end

    def as_dict(self) -> dict[str, object]:
        return {
            "canonical_path": self.canonical_path,
            "workspace_scope": self.workspace_scope,
            "content_digest": self.content_digest,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "decoding": self.decoding,
            "presentation_version": self.presentation_version,
            "workspace_generation": self.workspace_generation,
        }


@dataclass(frozen=True, slots=True)
class AuthorizedContentHandle:
    """Immutable bytes admitted by a caller's workspace/path policy."""

    canonical_path: str
    workspace_scope: str
    content: bytes
    workspace_generation: str
    decoding: str = "utf-8"
    presentation_version: str = "read-page/1"
    authorized: bool = True
    revoked: bool = False
    symlink: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_path", _canonical_path(self.canonical_path))
        object.__setattr__(self, "workspace_scope", _bounded_text(self.workspace_scope, "workspace scope"))
        if not isinstance(self.content, bytes):
            raise TypeError("authorized content must be bytes")
        object.__setattr__(self, "workspace_generation",
                           _bounded_text(self.workspace_generation, "workspace generation"))
        object.__setattr__(self, "decoding", _bounded_text(self.decoding, "decoding"))
        object.__setattr__(self, "presentation_version",
                           _bounded_text(self.presentation_version, "presentation version"))

    @property
    def content_digest(self) -> str:
        return _digest(self.content)

    @property
    def key(self) -> ReadKey:
        return ReadKey(
            self.canonical_path, self.workspace_scope, self.content_digest,
            0, len(self.content), self.decoding, self.presentation_version,
            self.workspace_generation,
        )

    def refreshed(self, content: bytes, *, workspace_generation: str | None = None) -> "AuthorizedContentHandle":
        """Rehash bytes after external writes; mtime is deliberately irrelevant."""
        return replace(self, content=bytes(content),
                       workspace_generation=(workspace_generation or self.workspace_generation))


@dataclass(frozen=True, slots=True)
class DeliveryProof:
    """Host proof that exact bytes crossed a particular serialized boundary."""

    key: ReadKey
    request_id: str
    serialized_request_digest: str
    delivered_bytes: bytes
    same_next_request: bool = False
    wire_contains_exact: bool = True
    proof_kind: str = "serialized_request"
    final_message_location: str = ""
    submitted_http_bytes_digest: str | None = None

    def __post_init__(self) -> None:
        if not str(self.request_id).strip():
            raise ValueError("delivery proof requires a request identity")
        if not str(self.serialized_request_digest).strip():
            raise ValueError("delivery proof requires a serialized request digest")
        if not isinstance(self.delivered_bytes, bytes):
            raise TypeError("delivery proof bytes must be bytes")
        if self.delivered_bytes != self.delivered_bytes[:]:
            raise ValueError("delivery proof bytes are not stable")
        if len(self.delivered_bytes) != self.key.byte_end - self.key.byte_start:
            raise ValueError("delivery proof byte range does not match delivered bytes")
        if self.wire_contains_exact and not str(self.final_message_location).strip():
            raise ValueError("delivery proof requires a final-message location")
        if self.submitted_http_bytes_digest is not None \
                and not str(self.submitted_http_bytes_digest).strip():
            raise ValueError("submitted HTTP-byte digest must be non-empty when present")

    def proves(self, key: ReadKey) -> bool:
        return (self.key == key and self.wire_contains_exact
                and bool(self.final_message_location)
                and self.delivered_bytes == self._expected_bytes(key))

    def _expected_bytes(self, key: ReadKey) -> bytes:
        # The proof stores the exact span.  This helper exists so malformed
        # subclass-like objects cannot accidentally broaden a comparison.
        if self.key != key:
            return b""
        return self.delivered_bytes

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-delivery-proof/1",
            "key": self.key.as_dict(),
            "request_id": self.request_id,
            "serialized_request_digest": self.serialized_request_digest,
            "delivered_digest": _digest(self.delivered_bytes),
            "delivered_bytes": len(self.delivered_bytes),
            "same_next_request": self.same_next_request,
            "wire_contains_exact": self.wire_contains_exact,
            "proof_kind": self.proof_kind,
            "final_message_location": self.final_message_location,
            "context_digest": self.serialized_request_digest,
            "submitted_http_bytes_digest": self.submitted_http_bytes_digest or "UNVERIFIED",
            "transport_identity": (
                "VERIFIED" if self.submitted_http_bytes_digest else "UNVERIFIED"
            ),
            "submitted_payload_hash": self.serialized_request_digest,
        }


@dataclass(frozen=True, slots=True)
class ReadRequest:
    action_id: str
    canonical_path: str
    byte_start: int = 0
    byte_end: int | None = None
    workspace_scope: str | None = None
    workspace_generation: str | None = None
    decoding: str = "utf-8"
    presentation_version: str = "read-page/1"

    def __post_init__(self) -> None:
        if not str(self.action_id).strip():
            raise ValueError("read action requires a unique action identity")
        object.__setattr__(self, "canonical_path", _canonical_path(self.canonical_path))
        if type(self.byte_start) is not int or self.byte_start < 0:
            raise ValueError("read byte start is invalid")
        if self.byte_end is not None and (type(self.byte_end) is not int
                                          or self.byte_end < self.byte_start):
            raise ValueError("read byte end is invalid")


@dataclass(frozen=True, slots=True)
class ReadReuseReceipt:
    """Ordered, chargeable outcome for one requested read."""

    action_id: str
    ordinal: int
    status: str
    key: ReadKey | None
    content: bytes
    budget_charge: int = 1
    reason: str = ""
    proof: DeliveryProof | None = None

    def __post_init__(self) -> None:
        if self.budget_charge != 1:
            raise ValueError("every requested read has exactly one normal budget charge")
        if self.status not in {ORDINARY_READ, CACHED_BYTES, UNCHANGED_VISIBLE, DENIED, STALE}:
            raise ValueError(f"unknown read reuse status: {self.status}")
        if not isinstance(self.content, bytes):
            raise TypeError("read receipt content must be bytes")

    @property
    def exact_bytes_delivered(self) -> bool:
        return (
            self.status in {CACHED_BYTES, UNCHANGED_VISIBLE}
            and self.key is not None
            and self.proof is not None
            and self.proof.proves(self.key)
            and self.proof.delivered_bytes == self.content
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "action_id": self.action_id,
            "ordinal": self.ordinal,
            "status": self.status,
            "key": self.key.as_dict() if self.key is not None else None,
            "content_digest": _digest(self.content) if self.content else None,
            "content_bytes": len(self.content),
            "budget_charge": self.budget_charge,
            "reason": self.reason,
            "proof": self.proof.as_dict() if self.proof is not None else None,
        }


def content_handle(
        canonical_path: str,
        content: bytes | bytearray,
        *,
        workspace_scope: str,
        workspace_generation: str | int,
        decoding: str = "utf-8",
        presentation_version: str = "read-page/1",
        authorized: bool = True,
        revoked: bool = False,
        symlink: bool = False,
) -> AuthorizedContentHandle:
    return AuthorizedContentHandle(
        canonical_path, workspace_scope, bytes(content), str(workspace_generation),
        decoding, presentation_version, authorized, revoked, symlink,
    )


def refresh_content_handle(
        handle: AuthorizedContentHandle,
        content: bytes | bytearray,
        *,
        workspace_generation: str | int | None = None,
) -> AuthorizedContentHandle:
    """Rehash current bytes after an external write; never trust mtime alone."""

    if not isinstance(handle, AuthorizedContentHandle):
        raise TypeError("refresh requires an authorized content handle")
    return handle.refreshed(
        bytes(content),
        workspace_generation=(
            None if workspace_generation is None else str(workspace_generation)
        ),
    )


def _requested_key(handle: AuthorizedContentHandle, request: ReadRequest) -> ReadKey:
    if request.canonical_path != handle.canonical_path:
        raise PermissionError("read path does not match the authorized content handle")
    end = len(handle.content) if request.byte_end is None else request.byte_end
    if end > len(handle.content):
        raise ValueError("requested read range is outside the authorized content handle")
    if request.workspace_scope not in (None, handle.workspace_scope):
        raise PermissionError("read workspace authorization scope changed")
    if request.workspace_generation not in (None, handle.workspace_generation):
        raise RuntimeError("read handle belongs to a stale workspace generation")
    if request.decoding != handle.decoding or request.presentation_version != handle.presentation_version:
        raise ValueError("read presentation identity changed")
    return ReadKey(
        handle.canonical_path, handle.workspace_scope, handle.content_digest,
        request.byte_start, end, handle.decoding, handle.presentation_version,
        handle.workspace_generation,
    )


def decide_read(
        handle: AuthorizedContentHandle | None,
        request: ReadRequest,
        proofs: Iterable[DeliveryProof] = (),
        *,
        ordinal: int = 0,
) -> ReadReuseReceipt:
    """Choose reuse only when one proof matches every identity field exactly."""
    if handle is None or not handle.authorized or handle.revoked or handle.symlink:
        return ReadReuseReceipt(request.action_id, ordinal, DENIED, None, b"", reason="authorization_denied")
    try:
        key = _requested_key(handle, request)
    except PermissionError as error:
        return ReadReuseReceipt(request.action_id, ordinal, DENIED, None, b"", reason=str(error))
    except RuntimeError as error:
        return ReadReuseReceipt(request.action_id, ordinal, STALE, None, b"", reason=str(error))
    except ValueError as error:
        return ReadReuseReceipt(request.action_id, ordinal, ORDINARY_READ, None, b"", reason=str(error))
    expected = handle.content[key.byte_start:key.byte_end]
    for proof in proofs:
        if not proof.proves(key) or proof.delivered_bytes != expected:
            continue
        status = UNCHANGED_VISIBLE if proof.same_next_request else CACHED_BYTES
        return ReadReuseReceipt(request.action_id, ordinal, status, key, expected,
                                reason="exact_revision_scope_range_match", proof=proof)
    return ReadReuseReceipt(request.action_id, ordinal, ORDINARY_READ, key, b"",
                            reason="no_exact_serialized_delivery_proof")


def plan_read_batch(
        handles: Mapping[str, AuthorizedContentHandle],
        operations: Sequence[ReadRequest | Mapping[str, object]],
        proofs: Iterable[DeliveryProof] = (),
) -> tuple[ReadReuseReceipt, ...]:
    """Evaluate reads in order; edits must be represented by a new handle.

    A mapping operation with ``kind=edit`` is accepted for convenience and
    updates the local handle only after its preceding reads have been charged.
    This is deliberately an in-memory projection, not a filesystem mutation.
    """
    current = dict(handles)
    available = tuple(proofs)
    receipts: list[ReadReuseReceipt] = []
    seen_actions: set[str] = set()
    for ordinal, operation in enumerate(operations):
        if isinstance(operation, ReadRequest):
            request = operation
        elif isinstance(operation, Mapping) and str(operation.get("kind", "read")) == "read":
            request = ReadRequest(
                str(operation.get("action_id") or f"read:{ordinal}"),
                str(operation.get("path") or operation.get("canonical_path") or ""),
                int(operation.get("byte_start", operation.get("start", 0)) or 0),
                (None if operation.get("byte_end", operation.get("end")) in (None, "")
                 else int(operation.get("byte_end", operation.get("end")))),
                (None if operation.get("workspace_scope") in (None, "")
                 else str(operation.get("workspace_scope"))),
                (None if operation.get("workspace_generation") in (None, "")
                 else str(operation.get("workspace_generation"))),
                str(operation.get("decoding", "utf-8")),
                str(operation.get("presentation_version", "read-page/1")),
            )
        elif isinstance(operation, Mapping) and str(operation.get("kind")) == "edit":
            path = _canonical_path(operation.get("path") or operation.get("canonical_path"))
            previous = current.get(path)
            if previous is None or not previous.authorized or previous.revoked or previous.symlink:
                continue
            body = operation.get("content")
            if not isinstance(body, (bytes, bytearray, str)):
                raise TypeError("batch edit content must be bytes or text")
            edited = body.encode(previous.decoding) if isinstance(body, str) else bytes(body)
            current[path] = previous.refreshed(
                edited, workspace_generation=str(
                    operation.get("workspace_generation", previous.workspace_generation)))
            continue
        else:
            raise TypeError("batch operation must be a read request or kind=edit mapping")
        if request.action_id in seen_actions:
            raise ValueError(f"read action identity reused: {request.action_id}")
        seen_actions.add(request.action_id)
        receipts.append(decide_read(current.get(request.canonical_path), request, available,
                                    ordinal=ordinal))
    return tuple(receipts)


def proof_for_serialized_request(
        key: ReadKey,
        exact_bytes: bytes,
        wire: bytes | str,
        *,
        request_id: str,
        same_next_request: bool = False,
        final_message_context: bytes | str | None = None,
        expected_message_location: str | None = None,
        context_digest: str | None = None,
        submitted_http_bytes: bytes | None = None,
) -> DeliveryProof:
    """Create a proof only at the exact final message location.

    ``wire`` is the bytes being inspected.  It may be a canonical host context
    projection, so transport identity is recorded only when the caller supplies
    the actual submitted HTTP bytes explicitly.
    """
    if len(exact_bytes) != key.byte_end - key.byte_start:
        raise ValueError("exact proof bytes do not match the requested byte range")
    wire_bytes = wire.encode("utf-8") if isinstance(wire, str) else bytes(wire)
    if submitted_http_bytes is not None and not isinstance(submitted_http_bytes, bytes):
        raise TypeError("submitted HTTP bytes must be bytes")
    try:
        decoded: object = json.loads(wire_bytes.decode("utf-8"))
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
        decoded = None

    context_text = None
    if final_message_context is not None:
        context_text = (final_message_context.decode(key.decoding, errors="strict")
                        if isinstance(final_message_context, bytes)
                        else str(final_message_context))

    def locate(value: object, location: tuple[object, ...] = ()) -> str | None:
        if isinstance(value, str):
            exact_text = exact_bytes.decode(key.decoding, errors="strict")
            if exact_text not in value or (context_text is not None and context_text not in value):
                return None
            if not location:
                return "$"
            rendered = "$"
            for part in location:
                rendered += f"[{part}]" if isinstance(part, int) else f".{part}"
            return rendered
        if isinstance(value, Mapping):
            for item_key, item in value.items():
                found = locate(item, (*location, str(item_key)))
                if found is not None:
                    return found
            return None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                found = locate(item, (*location, index))
                if found is not None:
                    return found
        return None

    def value_at_location(value: object, rendered: str) -> object | None:
        if not rendered.startswith("$"):
            return None
        remainder = rendered[1:]
        cursor = 0
        current = value
        token_pattern = re.compile(r"(?:\.([A-Za-z_][A-Za-z0-9_-]*)|\[(\d+)\])")
        for match in token_pattern.finditer(remainder):
            if match.start() != cursor:
                return None
            key, index = match.groups()
            try:
                if key is not None:
                    if not isinstance(current, Mapping):
                        return None
                    current = current[key]
                else:
                    if (not isinstance(current, Sequence)
                            or isinstance(current, (str, bytes, bytearray))):
                        return None
                    current = current[int(index)]
            except (KeyError, IndexError, TypeError, ValueError):
                return None
            cursor = match.end()
        return current if cursor == len(remainder) else None

    exact_text = exact_bytes.decode(key.decoding, errors="strict")
    if expected_message_location is not None:
        selected = value_at_location(decoded, expected_message_location) if decoded is not None else None
        location = (
            expected_message_location
            if isinstance(selected, str)
            and exact_text in selected
            and (context_text is None or selected == context_text)
            else None
        )
    else:
        location = locate(decoded) if decoded is not None else None
    if location is None:
        raise ValueError("serialized request does not contain exact source bytes at its final message location")
    return DeliveryProof(
        key, request_id, context_digest or _digest(wire_bytes), bytes(exact_bytes),
        same_next_request=same_next_request,
        final_message_location=location,
        submitted_http_bytes_digest=(
            _digest(submitted_http_bytes) if submitted_http_bytes is not None else None
        ),
    )


def runtime_dispatch(
        runtime: object,
        path: str,
        start: int,
        *,
        range_requested: bool,
        current_content: bytes | None = None,
) -> ReadReuseReceipt | None:
    """Return one explicit dispatch decision from the final-wire ledger.

    The legacy ledger intentionally lacks exact range proofs for most reads;
    those requests return ``None`` and proceed through the ordinary executor.
    No serve/reread flag may turn that absence into a model-visible denial.
    """
    if range_requested:
        return None
    relative = _canonical_path(path)
    sources = getattr(runtime, "_empirical_frame_sources", {})
    serialized = getattr(runtime, "_empirical_serialized_sources", {})
    row = serialized.get(relative) if isinstance(serialized, Mapping) else None
    source = sources.get(relative) if isinstance(sources, Mapping) else None
    if not isinstance(row, Mapping) or not isinstance(source, Mapping):
        return None
    content = source.get("content")
    if not isinstance(content, str) or row.get("full") is not True:
        return None
    ledger_bytes = content.encode("utf-8")
    if current_content is not None and bytes(current_content) != ledger_bytes:
        # The runtime adapter may use a prior serialized ledger only after the
        # authorized current bytes have been rehashed; mtime is not identity.
        return None
    current_revisions = getattr(runtime, "_empirical_current_revisions", {})
    current_revision = (current_revisions.get(relative)
                        if isinstance(current_revisions, Mapping) else None)
    if current_revision is not None and str(current_revision) != str(row.get("revision")):
        return None
    generation = str(getattr(runtime, "workspace_revision", row.get("revision", "0")))
    handle = content_handle(
        relative, ledger_bytes,
        workspace_scope=str(getattr(runtime, "workspace_scope", "runtime")),
        workspace_generation=generation,
        presentation_version="read-page/1",
    )
    if str(row.get("source_digest", row.get("revision"))) != handle.content_digest:
        return None
    sequence = int(getattr(runtime, "_read_dispatch_sequence", 0) or 0) + 1
    try:
        setattr(runtime, "_read_dispatch_sequence", sequence)
    except (AttributeError, TypeError):
        sequence = 0
    request = ReadRequest(
        f"read-dispatch:{sequence}:{relative}:{start}", relative, 0, len(handle.content),
        workspace_scope=str(getattr(runtime, "workspace_scope", "runtime")),
        workspace_generation=generation,
    )
    # A prior request is a cache candidate, never UNCHANGED_VISIBLE.  The
    # branch may use this only if it can render the exact cached bytes; callers
    # without a proof fall through to execute_read.
    proofs = getattr(runtime, "read_delivery_proofs", ())
    if not isinstance(proofs, Iterable):
        return None
    receipt = decide_read(handle, request, proofs, ordinal=0)
    return receipt if receipt.exact_bytes_delivered else None


__all__ = [
    "AuthorizedContentHandle", "CACHED_BYTES", "DENIED", "DeliveryProof",
    "ORDINARY_READ", "ReadKey", "ReadRequest", "ReadReuseReceipt", "STALE",
    "UNCHANGED_VISIBLE", "content_handle", "decide_read", "plan_read_batch",
    "proof_for_serialized_request", "runtime_dispatch",
    "refresh_content_handle",
]

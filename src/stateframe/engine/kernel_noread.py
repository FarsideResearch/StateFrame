"""Request-scoped read-menu withdrawal after an unchanged repeat read."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

ENV = "STATEFRAME_NOREAD"
STATE_ATTR = "_noread_menu_state"
MAX_TURNS = 8


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Parse the opt-in switch strictly so misspellings cannot silently change behavior."""
    values = os.environ if environ is None else environ
    value = str(values.get(ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{ENV} must be off or on")
    return value == "on"


def _request_id(value: object) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _emit(runtime: object, kind: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(kind, producer="kernel_noread", dispatch=False, **fields)


def _validated_revision(value: object) -> str | None:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return None
    try:
        return value if len(bytes.fromhex(value[7:])) == 32 else None
    except ValueError:
        return None


def arm_on_repeat(
    runtime: object,
    delivery: object,
    *,
    source_path: object = None,
    repeat_confirmed: bool = False,
) -> bool:
    """Arm only for an exact repeated path/revision/range with bound source bytes."""
    if not enabled():
        return False
    dedup = delivery.get("read_dedup") if isinstance(delivery, Mapping) else None
    if not isinstance(dedup, Mapping):
        return False
    from kernel_read_once import _REPEAT_STATUSES

    status = dedup.get("status")
    if status not in _REPEAT_STATUSES and repeat_confirmed is not True:
        return False
    revision = _validated_revision(dedup.get("revision"))
    if revision is None or source_path is None:
        return False
    try:
        path = str(Path(source_path).resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return False

    if isinstance(getattr(runtime, STATE_ATTR, None), Mapping):
        return False
    capture = getattr(runtime, "_active_capture", None)
    source_request_id = _request_id(getattr(capture, "request_id", None))
    if source_request_id is None:
        return False
    state: dict[str, object] = {
        "source_request_id": source_request_id,
        "source_path": path,
        "source_revision": revision,
        "checked_signature": None,
        "turn_limit": MAX_TURNS,
        "requests_sent": 0,
        "sent_request_ids": [],
        "active_request_id": None,
        "allowed_tools": [],
        "read_removed": False,
    }
    setattr(runtime, STATE_ATTR, state)
    from kernel_rejection_template_restrict import PROJECTION_RELEASE_ATTR

    setattr(runtime, PROJECTION_RELEASE_ATTR, None)
    _emit(
        runtime,
        "noread_applied",
        request_id=source_request_id,
        source_request_id=source_request_id,
        source_revision=revision,
        trigger_status=status,
        trigger_basis=("read_dedup_status" if status in _REPEAT_STATUSES
                       else "same_path_revision_range"),
        turn_limit=MAX_TURNS,
    )
    return True


def pending(runtime: object | None) -> bool:
    return (
        runtime is not None
        and enabled()
        and isinstance(getattr(runtime, STATE_ATTR, None), Mapping)
    )


def _file_signature(path: Path) -> tuple[int, int, int, int, int] | None:
    try:
        stat = path.stat()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(getattr(stat, "st_mtime_ns", stat.st_mtime * 1_000_000_000)),
        int(getattr(stat, "st_ctime_ns", stat.st_ctime * 1_000_000_000)),
    )


def _file_revision(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while True:
                chunk = source.read(128 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return "sha256:" + digest.hexdigest()


def source_content_changed(runtime: object | None) -> bool:
    """Compare the triggering file's current bytes with the repeat-read revision."""
    state = getattr(runtime, STATE_ATTR, None) if runtime is not None else None
    if not isinstance(state, Mapping):
        return False
    path = state.get("source_path")
    revision = state.get("source_revision")
    if not isinstance(path, str) or not path or not isinstance(revision, str) or not revision:
        return True
    source = Path(path)
    signature = _file_signature(source)
    if signature is None:
        return True
    if isinstance(state, dict) and state.get("checked_signature") == signature:
        return False
    current = _file_revision(source)
    if current != revision:
        return True
    if isinstance(state, dict):
        state["checked_signature"] = signature
    return False


def observe_read(
    runtime: object,
    delivery: object,
    *,
    source_path: object = None,
) -> bool:
    """Release a live policy if a later read observes changed bytes for its source file."""
    state = getattr(runtime, STATE_ATTR, None)
    dedup = delivery.get("read_dedup") if isinstance(delivery, Mapping) else None
    if not isinstance(state, Mapping) or not isinstance(dedup, Mapping) or source_path is None:
        return False
    try:
        path = str(Path(source_path).resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    revision = dedup.get("revision")
    if path != state.get("source_path") or not isinstance(revision, str):
        return False
    if revision == state.get("source_revision"):
        return False
    from kernel_rejection_template_restrict import release_noread

    return release_noread(
        runtime,
        request_id=state.get("active_request_id"),
        reason="source_content_changed",
    )


def wire_receipt(
    wire: bytes,
    *,
    runtime: object | None,
    request_id: object,
) -> dict[str, object] | None:
    """Return the shared policy's exact-body proof; measurement owns event emission."""
    from kernel_rejection_template_restrict import noread_wire_receipt

    return noread_wire_receipt(wire, runtime=runtime, request_id=request_id)

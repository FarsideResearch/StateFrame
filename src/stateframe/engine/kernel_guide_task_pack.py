"""Research guide-task treatment cues and the independent export scorer.

The task-pack switch is inert by default.  Its four enrolled values are a
small factorial: the capture cue can be placed early or after first use, and
the handover check can be status/JSON-only or bound to the ticket and exact
requested send bytes.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_GUIDE_TASK_PACK"
_MODE_VALUES = frozenset({"off", "neither", "ordering", "result_check", "both"})
PACK_MARKER = "[GUIDE TASK PACK | FF05/1]"
MAX_EXPORT_BYTES = 16 * 1024 * 1024
MAX_RUNS = 8
MAX_SENDS_PER_RUN = 64
MAX_FIXTURE_BYTES = 64 * 1024 * 1024

_CAPTURE_CUE = (
    "Record each printed run directory alongside the incident reference for the launch "
    "that produced it."
)
_ORDERING_EARLY = (
    "Place this capture cue before the guide's first use of a run directory: "
    + _CAPTURE_CUE
)
_ORDERING_LATE = (
    "Place this same capture cue immediately after the guide's first use of a run directory: "
    + _CAPTURE_CUE
)
_RESULT_CHECK = (
    "Before handover, read the incident reference from the exported request's task text and "
    "compare it with the support ticket. If they differ, reopen the launch receipt for the "
    "ticketed reference and export that run again. Once they agree, compare the exported file's "
    "bytes with the digest for that same run's requested send; if they differ, check the send "
    "number and export view."
)
_STATUS_JSON_CHECK = (
    "Before handover, confirm the commands succeeded and the exported file is readable JSON."
)


@dataclasses.dataclass(frozen=True)
class GuideTaskPackSettings:
    """The two independent treatment bits plus whether the pack was selected."""

    enabled: bool = False
    ordering: bool = False
    result_check: bool = False


def settings(value: object | None = None) -> GuideTaskPackSettings:
    """Read the default-OFF switch or parse one of its four treatment cells."""
    raw = os.environ.get(MODE_ENV, "off") if value is None else value
    mode = str(raw).strip().lower() or "off"
    if mode not in _MODE_VALUES:
        allowed = ", ".join(sorted(_MODE_VALUES))
        raise ValueError(f"{MODE_ENV} must be one of: {allowed}")
    if mode == "off":
        return GuideTaskPackSettings()
    return GuideTaskPackSettings(
        enabled=True,
        ordering=mode in {"ordering", "both"},
        result_check=mode in {"result_check", "both"},
    )


def render_cues(config: GuideTaskPackSettings) -> str:
    """Render the matched authoring slots for one enrolled guide brief."""
    if not config.enabled:
        return ""
    ordering = _ORDERING_EARLY if config.ordering else _ORDERING_LATE
    result_check = _RESULT_CHECK if config.result_check else _STATUS_JSON_CHECK
    return (
        f"{PACK_MARKER}\n"
        f"Ordering slot: {ordering}\n"
        f"Result-check slot: {result_check}"
    )


def _with_cues(messages: Sequence[object], config: GuideTaskPackSettings) -> list[object]:
    copied = [dict(row) if isinstance(row, Mapping) else row for row in messages]
    cues = render_cues(config)
    if not cues:
        return copied
    for index, row in enumerate(copied):
        if (not isinstance(row, Mapping) or row.get("role") != "user"
                or not isinstance(row.get("content"), str)):
            continue
        content = row["content"]
        if PACK_MARKER in content:
            return copied
        # Task text is kept in the existing user message so its meaning and task
        # identity remain visible to all request builders that key off that row.
        updated = dict(row)
        updated["content"] = content + "\n\n" + cues
        copied[index] = updated
        return copied
    return copied


def assemble_request_cues(
        messages: Sequence[object], uncapped: Sequence[object],
        config: GuideTaskPackSettings,
) -> tuple[list[object], list[object]]:
    """Apply selected cues to both request views, preserving repeated-prepare idempotence."""
    if not config.enabled:
        # Preserve the original objects on the ordinary path; callers must not
        # observe a copy or any changed wire structure when the switch is off.
        return messages, uncapped  # type: ignore[return-value]
    prepared = _with_cues(messages, config)
    if messages is uncapped:
        return prepared, prepared
    return prepared, _with_cues(uncapped, config)


@dataclasses.dataclass(frozen=True)
class ExportScore:
    requested_run: str
    requested_send: int
    observed_run: str | None
    observed_send: int | None
    identity_match: bool
    send_match: bool
    wire_format_match: bool
    exact_bytes_match: bool
    selected_summary_digest_match: bool
    accepted: bool
    reason: str


def _semantic_request_layout(messages: object) -> tuple[Sequence[object], object] | None:
    """Materialize request views and validate task/frame positions structurally."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return None
    try:
        if __package__:
            # Flat runtime imports and package-level readers share the engine helpers.
            engine_directory = os.path.dirname(os.path.abspath(__file__))
            if engine_directory not in sys.path:
                sys.path.insert(0, engine_directory)
        from kernel_cache_layout_view import semantic_messages
        from kernel_host_authorship import structural_class
        from kernel_stateframe_message_layout import validate_stateframe_message_order

        semantic = semantic_messages(messages)
        layout = validate_stateframe_message_order(
            semantic, require_task_label=False, allow_native_exchange=True,
            allow_user_context_before_stateframe=True,
        )
        # The layout contract identifies a unique frame. When the stable-wire
        # ownership reader can classify it, require the positional class too.
        frame_class = structural_class(semantic, layout.state_index)
    except (AttributeError, ImportError, IndexError, KeyError, TypeError, ValueError,
            RecursionError):
        return None
    if frame_class not in {None, "frame"}:
        return None
    return semantic, layout


def _request_task_text(document: object) -> str:
    if not isinstance(document, Mapping):
        return ""
    messages = document.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes, bytearray)):
        request = _semantic_request_layout(messages)
        if request is not None:
            semantic, layout = request
            row = semantic[layout.task_index]
            if (isinstance(row, Mapping) and row.get("role") == "user"
                    and isinstance(row.get("content"), str)):
                return row["content"]
        # Minimal request fixtures may omit StateFrame. Preserve their first
        # user task; valid StateFrame requests take the structurally located path.
        for row in messages:
            if (isinstance(row, Mapping) and row.get("role") == "user"
                    and isinstance(row.get("content"), str)):
                return row["content"]
    return ""


def _observed_run(task_text: str, known_runs: Sequence[str]) -> str | None:
    # Ticket references are identifiers, not arbitrary substrings. In
    # particular, INC-PRICE-731 must not claim INC-PRICE-7310 or a single-
    # hyphen extension. ASCII identifier boundaries also allow references in
    # Unicode prose; a doubled hyphen is treated as punctuation.
    matches = [
        reference for reference in known_runs
        if reference and re.search(
            rf"(?<![A-Za-z0-9_-]){re.escape(reference)}"
            rf"(?![A-Za-z0-9_]|-(?=[A-Za-z0-9_]))", task_text)
    ]
    return matches[0] if len(matches) == 1 else None


def _request_frame_payload(document: object) -> object | None:
    if not isinstance(document, Mapping):
        return None
    messages = document.get("messages")
    request = _semantic_request_layout(messages)
    if request is None:
        return None
    semantic, layout = request
    row = semantic[layout.state_index]
    content = row.get("content") if isinstance(row, Mapping) else None
    if not isinstance(content, str):
        return None
    _, separator, payload = content.partition("\n")
    if not separator:
        return None
    payload = payload.lstrip()
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, RecursionError):
        return {"_unparsed": payload}


def _observed_run_from_frame(
        frame: object, exported_send: int | None,
        run_sends: Mapping[str, Mapping[int, bytes]], known_runs: Sequence[str],
) -> str | None:
    if type(exported_send) is not int:
        return None
    matches: list[str] = []
    for run_reference, sends in run_sends.items():
        raw = sends.get(exported_send) if isinstance(sends, Mapping) else None
        if not isinstance(raw, bytes):
            continue
        try:
            request = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            continue
        if _request_frame_payload(request) != frame:
            continue
        task_run = _observed_run(_request_task_text(request), known_runs)
        if task_run == run_reference:
            matches.append(task_run)
    return matches[0] if len(matches) == 1 else None


def score_export(
        *, requested_run: str, requested_send: int, exported_bytes: bytes,
        exported_send: int | None, run_sends: Mapping[str, Mapping[int, bytes]],
        selected_summary_digest: str | None = None,
) -> ExportScore:
    """Score a handover against the ticket's run/send, never the selected run's own digest.

    ``run_sends`` is the fixture's frozen provider-boundary evidence.  The
    selected summary digest is reported as a diagnostic only; a decoy can
    truthfully match its own digest while failing ticket identity.
    """
    try:
        if (len(run_sends) > MAX_RUNS
                or any(len(sends) > MAX_SENDS_PER_RUN for sends in run_sends.values())
                or sum(len(raw) for sends in run_sends.values() for raw in sends.values()
                       if isinstance(raw, bytes)) > MAX_FIXTURE_BYTES):
            raise ValueError("fixture bounds exceeded")
        if any(not isinstance(raw, bytes) for sends in run_sends.values()
               for raw in sends.values()):
            raise ValueError("fixture send is not bytes")
    except (AttributeError, TypeError, ValueError):
        return ExportScore(
            requested_run, requested_send, None, exported_send, False,
            exported_send == requested_send, False, False, False, False,
            "fixture_bounds_exceeded_or_malformed",
        )
    known_runs = tuple(run_sends)
    summary_match = (
        isinstance(exported_bytes, bytes)
        and isinstance(selected_summary_digest, str)
        and len(exported_bytes) <= MAX_EXPORT_BYTES
        and hashlib.sha256(exported_bytes).hexdigest() == selected_summary_digest
    )
    observed_run = None
    wire_format_match = False
    if not isinstance(exported_bytes, bytes) or len(exported_bytes) > MAX_EXPORT_BYTES:
        return ExportScore(
            requested_run, requested_send, None, exported_send, False,
            exported_send == requested_send, False, False, summary_match, False,
            "export_too_large_or_not_bytes",
        )
    try:
        document = json.loads(exported_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        document = None
    if isinstance(document, Mapping):
        wire_format_match = (
            isinstance(document.get("model"), str)
            and isinstance(document.get("messages"), list)
        )
        if wire_format_match:
            observed_run = _observed_run(_request_task_text(document), known_runs)
        else:
            observed_run = _observed_run_from_frame(
                document, exported_send, run_sends, known_runs)
    identity_match = observed_run == requested_run
    send_match = exported_send == requested_send
    target_sends = run_sends.get(requested_run)
    target_bytes = target_sends.get(requested_send) if isinstance(target_sends, Mapping) else None
    exact_bytes_match = isinstance(target_bytes, bytes) and exported_bytes == target_bytes
    accepted = identity_match and send_match and wire_format_match and exact_bytes_match
    if accepted:
        reason = "ticketed_run_send_exact_bytes"
    elif not identity_match:
        reason = "wrong_or_ambiguous_run_identity"
    elif not send_match:
        reason = "wrong_send"
    elif not wire_format_match:
        reason = "not_a_provider_request_export"
    else:
        reason = "requested_send_bytes_mismatch"
    return ExportScore(
        requested_run, requested_send, observed_run, exported_send,
        identity_match, send_match, wire_format_match, exact_bytes_match,
        summary_match, accepted, reason,
    )


__all__ = [
    "ExportScore", "GuideTaskPackSettings", "MAX_EXPORT_BYTES", "MAX_FIXTURE_BYTES",
    "MAX_RUNS", "MAX_SENDS_PER_RUN", "MODE_ENV", "PACK_MARKER", "assemble_request_cues",
    "render_cues", "score_export", "settings",
]

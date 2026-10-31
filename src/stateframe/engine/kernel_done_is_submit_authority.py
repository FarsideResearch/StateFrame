"""Flagged, absorbing done submission authority.

STATEFRAME_DONE_IS_SUBMIT changes done from a readiness request into an artifact
submission. Readiness, certification and remaining obligations are recorded as
labels and never decide whether a captured candidate is submitted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MODE_ENV = "STATEFRAME_DONE_IS_SUBMIT"
ADVISORY_ENV = "STATEFRAME_DONE_ADVISORY"
SCHEMA = "stateframe-done-seal/1"
_ADVISORY_SPECS = {
    "CLAUSE_COVERAGE": (
        "stateframe_check_generator_fixes",
        "normative clause coverage is incomplete",
    ),
    "UNRESOLVED_GATING_ITEMS": (
        "kernel_host_done_facts",
        "unresolved gating items remain",
    ),
}
TERMINAL_RESULT = "SUBMITTED"
MAX_SETTLE_SECONDS = 30.0
_MUTATION_TOOLS = frozenset({"edit", "str_replace", "file_create", "file_delete", "file_move"})
_REQUIRED_CANDIDATE_TOOLS = ("edit", "str_replace", "file_create", "file_delete", "file_move", "done")
_REQUIRED_CANDIDATE_MUTATION_TOOLS = _REQUIRED_CANDIDATE_TOOLS[:-1]
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
OWNERLESS_EARLY_DONE_REASON = "OWNERLESS_DONE_SUBMIT_EARLY"
OWNERLESS_EARLY_DONE_REFUSAL_LIMIT = 2
_OWNERLESS_PROGRESS_TOOLS = frozenset({
    "edit", "str_replace", "file_create", "file_delete", "file_move", "replace_module", "write", "replace",
    "run_tests", "test",
})
_OWNERLESS_GATE_SKIP_REASONS = frozenset({
    "no_menu", "parent_unproven", "green", "idealpath_on", "sealed", "bound",
    "mutation_present", "proof_unavailable",
})


class DoneCaptureError(RuntimeError):
    """The current candidate could not be durably captured for submission."""


@dataclass(frozen=True, slots=True)
class DoneSubmitIntent:
    source: str
    requested_label: str
    accepted: bool = True

    def to_dict(self) -> dict[str, object]:
        return {"outcome": "SUBMIT_PENDING", "accepted": True,
                "source": self.source, "label": self.requested_label}


@dataclass(frozen=True, slots=True)
class DoneSeal:
    schema: str
    seal_id: str
    source: str
    label: str
    artifact_digest: str
    labels_json: str


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()  # default OFF in the flagged release
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_done_is_submit_mode", None) if runtime is not None else None
    if configured is not None:
        value = str(configured).strip().lower()
        if value not in {"off", "on"}:
            raise ValueError(f"{MODE_ENV} must be off or on")
        return value == "on"
    return mode() == "on"


def advisory_mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(ADVISORY_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{ADVISORY_ENV} must be off or on")
    return value


def advisory_enabled(runtime: object | None = None) -> bool:
    """Enable advisory downgrades only inside the existing done-submit path."""
    try:
        if not enabled(runtime):
            return False
        configured = (
            getattr(runtime, "stateframe_done_advisory_mode", None)
            if runtime is not None else None
        )
        value = (
            advisory_mode({ADVISORY_ENV: str(configured)})
            if configured is not None else advisory_mode()
        )
    except Exception:  # advisory configuration must never gate an existing submission
        return False
    return value == "on"


def _requested_advisory_codes(
        reason_codes: Sequence[object], detail: object) -> tuple[str, ...]:
    codes = {
        str(code).strip().upper().split(":", 1)[0]
        for code in reason_codes
        if str(code).strip()
    } if isinstance(reason_codes, Sequence) and not isinstance(
        reason_codes, (str, bytes, bytearray)) else set()
    text = str(detail or "").upper()
    if "NORMATIVE CLAUSE COVERAGE IS INCOMPLETE" in text:
        codes.add("CLAUSE_COVERAGE")
    if "UNRESOLVED_GATING_ITEMS" in text:
        codes.add("UNRESOLVED_GATING_ITEMS")
    return tuple(code for code in _ADVISORY_SPECS if code in codes)


def _record_advisory_downgrade(runtime: object, code: str) -> None:
    recorded = getattr(runtime, "_done_advisory_downgrade_event_codes", None)
    if not isinstance(recorded, set):
        recorded = set()
        setattr(runtime, "_done_advisory_downgrade_event_codes", recorded)
    if code in recorded:
        return
    recorded.add(code)
    authority, _what = _ADVISORY_SPECS[code]
    _emit(runtime, "done_advisory_downgraded", authority=authority, code=code)


def capture_advisory_downgrades(
        runtime: object, *, reason_codes: Sequence[object] = (),
        detail: object = "") -> tuple[str, ...]:
    """Record known soft done blockers for the accepted submission reply."""
    if not advisory_enabled(runtime):
        return ()
    try:
        requested = _requested_advisory_codes(reason_codes, detail)
        setattr(runtime, "_done_advisory_downgrade_reply_codes", set(requested))
        for code in requested:
            _record_advisory_downgrade(runtime, code)
        return tuple(
            f"[host] advisory: {_ADVISORY_SPECS[code][1]}; the submission was accepted"
            for code in requested
        )
    except Exception:  # advisory recording must never gate an existing submission
        return ()


def advisory_lines(runtime: object | None) -> tuple[str, ...]:
    try:
        if not advisory_enabled(runtime):
            return ()
        requested = getattr(runtime, "_done_advisory_downgrade_reply_codes", ())
        if not isinstance(requested, (set, frozenset, tuple, list)):
            return ()
        return tuple(
            f"[host] advisory: {_ADVISORY_SPECS[code][1]}; the submission was accepted"
            for code in _ADVISORY_SPECS
            if code in requested
        )
    except Exception:  # advisory presentation cannot gate an existing submission
        return ()


def reply_detail(runtime: object | None, base: str) -> str:
    lines = advisory_lines(runtime)
    return "\n".join((base, *lines)) if lines else base


def candidate_exists(runtime: object | None) -> bool:
    if runtime is None:
        return False
    binding = getattr(runtime, "binding", None)
    if callable(binding):
        try:
            value = binding()
        except (OSError, RuntimeError, TypeError, ValueError):
            value = None
        if isinstance(value, Mapping) and any(
                value.get(key) not in (None, "", False)
                for key in ("product_revision", "artifact_identity", "target", "candidate_id")):
            return True
    target = getattr(runtime, "target", None)
    owner = getattr(runtime, "owner", None)
    root = getattr(owner, "root", None)
    if target is not None:
        try:
            path = Path(target)
            if root is not None:
                path.resolve().relative_to(Path(root).resolve())
            return path.is_file()
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
    return bool(getattr(runtime, "_candidate_exists", False))


def candidate_changed(runtime: object | None) -> bool:
    """True only when host product identity is positively different from task parent."""
    if runtime is None:
        return False
    try:
        from kernel_green_lock import product_revision_change_from_parent
        changed, revision, _comparison = product_revision_change_from_parent(runtime)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return bool(changed and revision and str(revision).strip().upper() != "UNKNOWN")


def _ownerless_progress_tool_names(available_tools: object) -> tuple[str, ...]:
    names: list[str] = []
    if isinstance(available_tools, Mapping):
        items = available_tools.values()
    elif isinstance(available_tools, Sequence) and not isinstance(
            available_tools, (str, bytes, bytearray)):
        items = available_tools
    else:
        return ()
    for item in items:
        if isinstance(item, str):
            name = item.strip().lower()
        elif isinstance(item, Mapping):
            name = _tool_name(item).lower()
        else:
            continue
        if name in _OWNERLESS_PROGRESS_TOOLS and name not in names:
            names.append(name)
    return tuple(names)


def record_ownerless_gate_skipped(runtime: object, reason: str) -> None:
    """Record a typed reason whenever the ownerless gate cannot refuse DONE."""
    safe_reason = reason if reason in _OWNERLESS_GATE_SKIP_REASONS else "proof_unavailable"
    _emit(runtime, "early_done_gate_skipped", reason=safe_reason)


def _ownerless_gate_preflight(
        runtime: object, available_tools: object,
) -> tuple[str | None, tuple[str, ...]]:
    if not enabled(runtime):
        return "proof_unavailable", ()
    if done_sealed(runtime) is not None:
        return "sealed", ()
    try:
        from kernel_idealpath import enabled as idealpath_enabled
        if idealpath_enabled(runtime):
            return "idealpath_on", ()

        actions = _ownerless_progress_tool_names(available_tools)
        if not actions:
            return "no_menu", ()

        from kernel_green_lock import (
            current_public_verdict_is_green,
            product_parent_proven_unchanged,
        )
        if candidate_changed(runtime):
            return "mutation_present", actions
        if current_public_verdict_is_green(runtime):
            return "green", actions
        if not product_parent_proven_unchanged(runtime):
            return "parent_unproven", actions
    except Exception as error:  # noqa: BLE001 - unavailable proof cannot create a false refusal
        _emit(runtime, "done_submit_ownerless_gate_unavailable",
              error_type=type(error).__name__)
        return "proof_unavailable", ()
    return None, actions


def ownerless_early_done_refusal(
        runtime: object, *, available_tools: object = (),
) -> dict[str, object] | None:
    """Refuse only an unchanged, unchecked ownerless submission with a live recovery menu."""
    skip_reason, actions = _ownerless_gate_preflight(runtime, available_tools)
    if skip_reason is not None:
        record_ownerless_gate_skipped(runtime, skip_reason)
        return None

    prior_count = getattr(runtime, "_done_submit_ownerless_early_refusals", 0)
    count = prior_count if isinstance(prior_count, int) and not isinstance(prior_count, bool) else 0
    count = max(0, count)
    if count >= OWNERLESS_EARLY_DONE_REFUSAL_LIMIT:
        record_ownerless_gate_skipped(runtime, "bound")
        return None
    count += 1
    try:
        setattr(runtime, "_done_submit_ownerless_early_refusals", count)
    except (AttributeError, TypeError):
        _emit(runtime, "done_submit_ownerless_gate_unavailable",
              error_type="RuntimeStateUnavailable")
        record_ownerless_gate_skipped(runtime, "proof_unavailable")
        return None

    detail = (
        "[host] done refused: no product file has been changed yet; "
        "no check has passed on the current revision. Available now: "
        + " or ".join(actions) + "."
    )
    _emit(
        runtime,
        "early_done_refused",
        refusal_index=count,
        refusal_bound=OWNERLESS_EARLY_DONE_REFUSAL_LIMIT,
        reason="no_mutation_and_no_green_check",
    )
    return {
        "reason_code": OWNERLESS_EARLY_DONE_REASON,
        "detail": detail,
        "repair_action": {
            "tool": actions[0],
            "arguments": {},
            "reason": "Make a product edit or pass a current-revision public check before trying done again.",
        },
        "available_tools": actions,
        "refusal_count": count,
    }


def request_submission(runtime: object, *, source: str, label: str = "submitted") -> DoneSubmitIntent:
    if source not in {"model", "host"}:
        raise ValueError("done submission source must be model or host")
    existing = done_sealed(runtime)
    if existing is not None:
        return DoneSubmitIntent(source=existing.source, requested_label=existing.label)
    pending = getattr(runtime, "_done_submit_pending", None)
    if isinstance(pending, DoneSubmitIntent):
        return pending
    intent = DoneSubmitIntent(source=source, requested_label=_safe_label(label))
    setattr(runtime, "_done_submit_pending", intent)
    setattr(runtime, "_done_submit_mutation_wait_completed", False)
    setattr(runtime, "_done_submit_pending_mutations", ())
    setattr(runtime, "_done_submit_pending_mutation_count", 0)
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit("done_submit_requested", producer=__name__, source=source,
             label=intent.requested_label, dispatch=False)
    return intent


def submission_pending(runtime: object | None) -> bool:
    return isinstance(getattr(runtime, "_done_submit_pending", None), DoneSubmitIntent)


def wait_for_pending_mutations(runtime: object, *, timeout_seconds: float = MAX_SETTLE_SECONDS,
                               poll_seconds: float = 0.05) -> tuple[str, ...]:
    """Wait at most 30 seconds for known in-flight mutation effects, then continue."""
    if getattr(runtime, "_done_submit_mutation_wait_completed", False) is True:
        previous = getattr(runtime, "_done_submit_pending_mutations", ())
        return tuple(previous) if isinstance(previous, Sequence) and not isinstance(
            previous, (str, bytes, bytearray)) else ()
    timeout = min(MAX_SETTLE_SECONDS, max(0.0, float(timeout_seconds)))
    deadline = time.monotonic() + timeout
    pending = _pending_mutations(runtime)
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        # Passive bounded polling cannot inherit an unbounded callback wait from a
        # plugin or leave a background waiter running after submission.
        time.sleep(min(max(0.001, float(poll_seconds)), remaining))
        pending = _pending_mutations(runtime)
    setattr(runtime, "_done_submit_pending_mutations", pending)
    setattr(runtime, "_done_submit_pending_mutation_count", len(pending))
    setattr(runtime, "_done_submit_mutation_wait_completed", True)
    return pending


def _pending_mutations(runtime: object) -> tuple[str, ...]:
    direct = getattr(runtime, "_pending_mutation_ids", ())
    values: list[str] = []
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes, bytearray)):
        values.extend(str(item) for item in direct if str(item).strip())
    continuation = getattr(runtime, "continuation", None)
    effects = getattr(continuation, "effects", None)
    if isinstance(effects, Mapping):
        for effect_id, effect in effects.items():
            status = str(getattr(effect, "status", "")).strip().lower()
            tool = str(getattr(effect, "tool", "")).strip().lower()
            if status in {"pending", "unresolved", "submitted", "unknown"} and tool in _MUTATION_TOOLS:
                values.append(str(effect_id))
    engine = getattr(continuation, "engine", None)
    has_unsettled = getattr(engine, "has_unsettled_effects", None)
    if callable(has_unsettled):
        try:
            if has_unsettled():
                values.append("continuation:unsettled_effects")
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            values.append("continuation:unsettled_status_unavailable")
    return tuple(dict.fromkeys(values))


def _normalized_sources(sources: Mapping[str, str | None]) -> dict[str, str | None]:
    from pathlib import PurePosixPath

    normalized = {}
    for path, source in sources.items():
        if (not isinstance(path, str) or not path or "\\" in path or "\x00" in path
                or PurePosixPath(path).is_absolute()
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or not (isinstance(source, str) or source is None)):
            raise DoneCaptureError(f"invalid candidate source entry: {path!r}")
        normalized[path] = source
    return dict(sorted(normalized.items()))


def artifact_digest(sources: Mapping[str, str | None]) -> str:
    """Hash canonical writes and tombstones in the versioned product artifact."""
    normalized = _normalized_sources(sources)
    payload = {"schema": "stateframe-done-artifact/2",
               "writes": {path: source for path, source in normalized.items() if source is not None},
               "deletes": [path for path, source in normalized.items() if source is None]}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_immutable_snapshot(path: Path, encoded: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise DoneCaptureError("existing candidate snapshot does not match its digest")


def store_candidate(destination: str | Path, task_id: object,
                    sources: Mapping[str, str | None]) -> dict[str, object]:
    """Persist one immutable source snapshot, keyed by content digest."""
    try:
        normalized_sources = _normalized_sources(sources)
        if not normalized_sources:
            raise DoneCaptureError("candidate snapshot contains no product effect")
        digest = artifact_digest(normalized_sources)
        artifact = {
            "schema": "stateframe-done-artifact/2",
            "artifact_digest": digest,
            "writes": {path: source for path, source in normalized_sources.items() if source is not None},
            "deletes": [path for path, source in normalized_sources.items() if source is None],
        }
        encoded = (json.dumps(artifact, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")) + "\n").encode("utf-8")
        slug = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", str(task_id or "candidate")
        )[:120].strip("._") or "candidate"
        destination_root = Path(destination).resolve()
        folder = destination_root / slug
        folder.mkdir(parents=True, exist_ok=True)
        if folder.is_symlink() or folder.resolve().parent != destination_root:
            raise DoneCaptureError("candidate snapshot directory escaped its destination")
        path = folder / f"{digest.removeprefix('sha256:')}.json"
        _write_immutable_snapshot(path, encoded)
        return {"stored": True, "artifact_digest": digest}
    except DoneCaptureError:
        raise
    except Exception as error:  # serialization and filesystem failures are capture failures
        raise DoneCaptureError(
            f"candidate snapshot could not be stored: {type(error).__name__}"
        ) from error


def seal_submission(runtime: object, *, source: str, artifact_digest_value: object,
                    label: str, labels: Mapping[str, object] | None = None,
                    artifact_stored: bool = True) -> DoneSeal:
    """Create the one immutable submission seal after durable artifact storage."""
    existing = done_sealed(runtime)
    if existing is not None:
        return existing
    if source not in {"model", "host"}:
        raise ValueError("done submission source must be model or host")
    digest = str(artifact_digest_value or "")
    if not artifact_stored or not _SHA256.fullmatch(digest):
        raise DoneCaptureError("candidate snapshot is not durably stored")
    normalized_labels = _safe_labels(labels or {})
    labels_json = json.dumps(normalized_labels, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"))
    label_value = _safe_label(label)
    payload = {
        "schema": SCHEMA, "source": source, "label": label_value,
        "artifact_digest": digest, "labels": normalized_labels,
    }
    seal_id = "sha256:" + hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    seal = DoneSeal(SCHEMA, seal_id, source, label_value, digest, labels_json)
    setattr(runtime, "_done_is_submit_seal", seal)
    setattr(runtime, "completion_accepted", True)
    setattr(runtime, "v2_accepted_done", True)
    setattr(runtime, "_done_submit_pending", None)
    _emit(runtime, "done_sealed", source=source, label=label_value,
          artifact_digest=digest, seal_id=seal_id, stored=True)
    refusal_count = getattr(runtime, "_done_submit_ownerless_early_refusals", 0)
    if (isinstance(refusal_count, int) and not isinstance(refusal_count, bool)
            and refusal_count == OWNERLESS_EARLY_DONE_REFUSAL_LIMIT):
        _emit(
            runtime,
            "early_done_refusal_bound_reached",
            refusals=OWNERLESS_EARLY_DONE_REFUSAL_LIMIT,
        )
    return seal


def capture_and_seal(runtime: object, *, destination: str | Path, task_id: object,
                     sources: Mapping[str, str | None], parent_sources: Mapping[str, str],
                     out_of_scope: bool = False,
                     certification: object = "not_evaluated") -> DoneSeal:
    """Persist current source bytes and seal their digest as the submitted artifact."""
    intent = getattr(runtime, "_done_submit_pending", None)
    if not isinstance(intent, DoneSubmitIntent):
        intent = request_submission(runtime, source="model")
    stored = store_candidate(destination, task_id, sources)
    label = label_for_candidate(
        str(stored["artifact_digest"]), artifact_digest(parent_sources),
        out_of_scope=out_of_scope,
    )
    try:
        labels = completion_labels(runtime, certification=certification)
    except Exception as error:  # labels never gate a captured candidate
        labels = {"status": "unavailable"}
        _emit(runtime, "done_labels_unavailable", error_type=type(error).__name__)
    seal = seal_submission(
        runtime, source=intent.source, artifact_digest_value=stored["artifact_digest"],
        label=label, labels=labels, artifact_stored=stored.get("stored") is True,
    )
    setattr(runtime, "_done_submit_artifact_digest", seal.artifact_digest)
    _emit(
        runtime, "done_submission_stored", source=seal.source, label=seal.label,
        artifact_digest=seal.artifact_digest, stored=True,
        pending_mutation_count=getattr(runtime, "_done_submit_pending_mutation_count", 0),
    )
    return seal


def submit_done(runtime: object, *, source: str, destination: str | Path,
                task_id: object, sources: Mapping[str, str | None],
                parent_sources: Mapping[str, str], out_of_scope: bool = False,
                certification: object = "not_evaluated",
                timeout_seconds: float = MAX_SETTLE_SECONDS) -> DoneSeal:
    """Run the single model/host submission path: settle, store, then seal."""
    existing = done_sealed(runtime)
    if existing is not None:
        return existing
    intent = request_submission(runtime, source=source)
    wait_for_pending_mutations(runtime, timeout_seconds=timeout_seconds)
    # Preserve the first initiator when model and host triggers race.
    if intent.source != source:
        _emit(runtime, "done_submit_source_reused", requested_source=source,
              source=intent.source)
    return capture_and_seal(
        runtime, destination=destination, task_id=task_id, sources=sources,
        parent_sources=parent_sources, out_of_scope=out_of_scope,
        certification=certification,
    )


def accepted_decision(runtime: object):
    """Return a typed accepted receipt when a seal is replayed by legacy callers."""
    seal = done_sealed(runtime)
    if seal is None:
        return None
    from kernel_verdict import ACCEPTED, CompletionBasis, CompletionDecision

    basis = CompletionBasis.capture(seal.seal_id, seal.artifact_digest,
                                    evidence=seal.artifact_digest)
    decision = CompletionDecision(
        ACCEPTED, basis, accepted=True, reason_codes=("DONE_SEALED",),
        detail=reply_detail(runtime, "done submission is sealed"), artifact_preserved=True, sealed=True,
    )
    setattr(runtime, "_last_completion_decision", decision)
    return decision


def pending_decision(runtime: object, *, source: str = "model"):
    """Stage a typed accepted intent; artifact storage and immutable sealing follow."""
    intent = request_submission(runtime, source=source)
    from kernel_verdict import ACCEPTED, CompletionBasis, CompletionDecision

    binding = getattr(runtime, "binding", None)
    try:
        current_binding = binding() if callable(binding) else {}
    except (OSError, RuntimeError, TypeError, ValueError):
        current_binding = {}
    basis = CompletionBasis.capture("done-submit-pending", current_binding,
                                    evidence=intent.to_dict())
    return CompletionDecision(
        ACCEPTED, basis, accepted=True, reason_codes=("DONE_SUBMIT_PENDING",),
        detail=reply_detail(runtime, "current candidate will be captured and submitted"), artifact_preserved=True,
    )


def done_sealed(runtime: object | None) -> DoneSeal | None:
    value = getattr(runtime, "_done_is_submit_seal", None) if runtime is not None else None
    return value if isinstance(value, DoneSeal) else None


def terminal_from_seal(runtime: object) -> dict[str, object] | None:
    seal = done_sealed(runtime)
    if seal is None:
        return None
    try:
        labels = json.loads(seal.labels_json)
    except (TypeError, ValueError):
        labels = {}
    open_ids = labels.get("open_item_ids", []) if isinstance(labels, Mapping) else []
    if not isinstance(open_ids, list):
        open_ids = []
    certification = str(labels.get("certification", "unknown")) if isinstance(labels, Mapping) else "unknown"
    certified = certification.strip().lower() in {"certified", "accepted", "pass", "passed"}
    terminal = {
        "result": TERMINAL_RESULT,
        "outcome": TERMINAL_RESULT,
        "result_class": TERMINAL_RESULT,
        "accepted": True,
        "certified": certified,
        "uncertified": not certified,
        "artifact": "STORED",
        "delivery": "STORED",
        "artifact_digest": seal.artifact_digest,
        "stop_reason": TERMINAL_RESULT,
        "open_ids": open_ids,
        "unresolved_gating_items": open_ids,
        "behaviour": "SUBMITTED",
        "done_seal": {
            "schema": seal.schema,
            "seal_id": seal.seal_id,
            "source": seal.source,
            "label": seal.label,
            "artifact_digest": seal.artifact_digest,
            "stored": True,
        },
        "labels": labels,
    }
    lines = advisory_lines(runtime)
    if lines:
        terminal["detail"] = "\n".join(lines)
    return terminal


def final_result(runtime: object, fallback: str) -> str:
    """Project the sealed submission as the run result without mutating runtime outcome state."""
    if done_sealed(runtime) is not None:
        return TERMINAL_RESULT
    outcome = getattr(runtime, "system_outcome", None)
    return str(outcome).upper() if outcome is not None else fallback


def label_for_candidate(candidate_digest: str, parent_digest: str | None,
                        *, out_of_scope: bool = False) -> str:
    if out_of_scope:
        return "out_of_scope"
    if parent_digest and candidate_digest == parent_digest:
        return "no_change"
    return "submitted"


def _identifier_values(values: Sequence[object], *, limit: int = 32) -> list[str]:
    identifiers: list[str] = []
    for item in values[:limit]:
        value: object = item
        if isinstance(item, Mapping):
            value = next((item.get(key) for key in (
                "id", "item_id", "obligation_id", "requirement_id", "code")
                if item.get(key) not in (None, "")), "")
        elif not isinstance(item, str):
            value = next((getattr(item, key, None) for key in (
                "id", "item_id", "obligation_id", "requirement_id", "code")
                if getattr(item, key, None) not in (None, "")), "")
        identifier = str(value).strip()
        if identifier and identifier not in identifiers:
            identifiers.append(identifier[:120])
    return identifiers


def completion_labels(runtime: object, *, certification: object = None) -> dict[str, object]:
    """Snapshot bounded status labels; none of these values gates submission."""
    readiness = getattr(runtime, "_readiness_decision", None)
    readiness_value = getattr(readiness, "decision", None)
    verdict = getattr(runtime, "public_test_verdict", None)
    try:
        verdict_value = getattr(verdict(), "state", None) if callable(verdict) else None
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        verdict_value = None
    gate_status = "unavailable"
    gate_codes: list[str] = []
    gate_fn = getattr(runtime, "pass_gates_vector", None)
    if callable(gate_fn):
        try:
            vector = gate_fn()
            gate_status = "ready" if getattr(vector, "ready", False) is True else "open"
            gate_codes = [str(item.code)[:80] for item in
                          tuple(getattr(vector, "blockers", ()) or ())[:16]]
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
    open_items = getattr(runtime, "_host_done_facts_gating_items", ())
    if not isinstance(open_items, Sequence) or isinstance(open_items, (str, bytes, bytearray)):
        open_items = ()
    counterexamples = getattr(runtime, "counterexample_receipts", ())
    if not isinstance(counterexamples, Sequence) or isinstance(counterexamples, (str, bytes, bytearray)):
        counterexamples = ()
    pending_mutations = getattr(runtime, "_done_submit_pending_mutations", ())
    if not isinstance(pending_mutations, Sequence) or isinstance(
            pending_mutations, (str, bytes, bytearray)):
        pending_mutations = ()
    open_item_ids = _identifier_values(open_items)
    open_item_ids.extend(
        code for code in gate_codes
        if code.startswith("CATALOG_UNKNOWN:") and code not in open_item_ids
    )
    mutation_status = "mutation_in_flight" if pending_mutations else "settled"
    return {
        "readiness": str(readiness_value or "unknown")[:80],
        "certification": str(certification or verdict_value or "unknown")[:80],
        "evidence_currency": str(getattr(runtime, "evidence_currency", "unknown"))[:80],
        "open_items": len(open_items),
        "open_item_ids": open_item_ids[:32],
        "counterexamples": len(counterexamples),
        "counterexample_ids": _identifier_values(counterexamples),
        "coverage": str(getattr(runtime, "coverage_status", "unknown"))[:80],
        "catalog": {"status": gate_status, "open_codes": gate_codes},
        "pending_mutations": [str(item)[:120] for item in pending_mutations[:32]],
        "mutation_status": mutation_status,
    }


def _required_candidate_tools(runtime: object, tools: object) -> tuple[str, ...]:
    if not enabled(runtime) or not candidate_exists(runtime):
        return ()
    offered_names = {
        _tool_name(item) for item in tools
        if isinstance(item, Mapping)
    } if isinstance(tools, Sequence) and not isinstance(
        tools, (str, bytes, bytearray)) else set()
    required = list(_REQUIRED_CANDIDATE_MUTATION_TOOLS)
    if "done" in offered_names or candidate_changed(runtime):
        required.append("done")
    return tuple(required)


def _project_candidate_tools(
        runtime: object, tools: object, *, full_tools: object,
        required: Sequence[str]) -> list[dict[str, object]]:
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        offered: list[dict[str, object]] = []
    else:
        offered = [copy.deepcopy(dict(item)) for item in tools
                   if isinstance(item, Mapping)]
    if not required:
        return offered
    # A terminal projection can become the remembered surface after a product
    # edit. Merge every host-owned schema source so one narrowed snapshot cannot
    # permanently remove a candidate action from later requests.
    sources = (
        full_tools,
        getattr(runtime, "_completion_owner_full_tools", None),
        getattr(runtime, "_completion_owner_dispatch_tools", None),
        getattr(runtime, "_readiness_full_tools", None),
        getattr(runtime, "available_tools", None),
    )
    by_name: dict[str, Mapping[str, object]] = {}
    for source in sources:
        if isinstance(source, Mapping):
            items = source.values()
        elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            items = source
        else:
            continue
        for item in items:
            if isinstance(item, Mapping):
                name = _tool_name(item)
                if name:
                    by_name.setdefault(name, item)
    present = {_tool_name(item) for item in offered}
    for name in required:
        if name not in present and isinstance(by_name.get(name), Mapping):
            offered.append(copy.deepcopy(dict(by_name[name])))
            present.add(name)
    return offered


def project_candidate_tools(runtime: object, tools: object,
                            *, full_tools: object = None) -> list[dict[str, object]]:
    """Preserve candidate edit tools; restore done only for a changed candidate."""
    required = _required_candidate_tools(runtime, tools)
    return _project_candidate_tools(
        runtime, tools, full_tools=full_tools, required=required,
    )


def project_final_candidate_tools(
        runtime: object, tools: object, *, request_id: object,
        full_tools: object = None,
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    """Project and journal the required candidate tools on one exact request."""
    required = _required_candidate_tools(runtime, tools)
    projected = _project_candidate_tools(
        runtime, tools, full_tools=full_tools, required=required,
    )
    names = tuple(_tool_name(item) for item in projected)
    missing = tuple(name for name in required if name not in names)
    _emit(runtime, "done_is_submit_final_tools", request_id=request_id,
          required=required,
          present=[name for name in required if name in names])
    return projected, missing



def _tool_name(value: Mapping[str, object]) -> str:
    function = value.get("function")
    return str(function.get("name", "")).strip() if isinstance(function, Mapping) else ""


def _safe_label(value: object) -> str:
    label = re.sub(r"[^a-z0-9_.-]+", "_", str(value or "submitted").strip().lower())
    return label[:64] or "submitted"


def _safe_labels(values: Mapping[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in values.items():
        name = _safe_label(key)
        if isinstance(value, Mapping):
            nested = {}
            for child_key, child_value in tuple(value.items())[:16]:
                if isinstance(child_value, (str, int, float, bool)) or child_value is None:
                    nested[_safe_label(child_key)] = (
                        child_value[:120] if isinstance(child_value, str) else child_value)
                elif isinstance(child_value, Sequence) and not isinstance(
                        child_value, (str, bytes, bytearray)):
                    nested[_safe_label(child_key)] = [
                        str(item)[:80] for item in child_value[:16]
                    ]
            safe[name] = nested
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[name] = value if not isinstance(value, str) else value[:160]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            safe[name] = [str(item)[:120] for item in value[:32]]
    return safe


def _emit(runtime: object, kind: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        try:
            emit(kind, producer=__name__, dispatch=False, **fields)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass


__all__ = [
    "ADVISORY_ENV", "DoneCaptureError", "DoneSeal", "DoneSubmitIntent", "MODE_ENV", "SCHEMA",
    "OWNERLESS_EARLY_DONE_REASON", "OWNERLESS_EARLY_DONE_REFUSAL_LIMIT",
    "accepted_decision", "advisory_enabled", "advisory_lines", "advisory_mode",
    "artifact_digest", "candidate_changed", "candidate_exists", "capture_advisory_downgrades",
    "capture_and_seal", "completion_labels", "done_sealed", "enabled", "label_for_candidate", "mode",
    "ownerless_early_done_refusal", "pending_decision", "record_ownerless_gate_skipped",
    "project_candidate_tools", "project_final_candidate_tools", "request_submission",
    "reply_detail", "seal_submission", "store_candidate",
    "submit_done", "submission_pending", "terminal_from_seal", "wait_for_pending_mutations",
]

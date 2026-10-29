"""Host-owned, opt-in bridge from the public census to pre-edit wire exposure."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import logging
import os

from kernel_preedit_coverage import (
    BoundSource, CoverageSite, PreeditInput, PublicRequirement, ScopeInventory,
    render_preedit_coverage,
)

SCHEMA = "kernel-preedit-coverage-state/1"
ENV = "STATEFRAME_PREEDIT_COVERAGE"
STATE_FIELD = "_preedit_coverage_state"
SUBMISSION_FIELD = "preedit_coverage_delivery"
_EDIT_KINDS = frozenset({"edit", "str_replace", "file_create", "replace_module", "write", "replace"})


def enabled(runtime: object) -> bool:
    return getattr(runtime, "stateframe_preedit_coverage_arm", None) in {"treatment", "placebo"}


def initialize(runtime: object, *, resumed: bool) -> None:
    """Only called by the host constructor; OFF/control creates no state."""
    arm = os.environ.get(ENV, "off").strip().lower()
    if arm not in {"treatment", "placebo"}:
        return
    runtime.stateframe_preedit_coverage_arm = arm
    runtime._preedit_coverage_state = {
        "schema": SCHEMA, "attempt_id": runtime.journal.run_id,
        "arm": arm, "seen": [], "resume_verified": not resumed,
    }
    runtime._preedit_coverage_pending = {}


def _rows(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _key(attempt: str, revision: str) -> str:
    body = json.dumps({"attempt_id": attempt, "candidate_revision": revision},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return _sha(b"kernel-preedit-coverage/1\0" + body)


def _state(runtime: object) -> dict[str, object] | None:
    value = getattr(runtime, STATE_FIELD, None)
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA
            or value.get("attempt_id") != getattr(runtime.journal, "run_id", None)
            or value.get("arm") != getattr(runtime, "stateframe_preedit_coverage_arm", None)
            or value.get("resume_verified") is not True
            or type(value.get("seen")) is not list or len(value["seen"]) > 256):
        return None
    keys: set[str] = set()
    for item in value["seen"]:
        if (not isinstance(item, dict) or not _valid_record(item, value["attempt_id"])
                or item["arm"] != value["arm"] or item["delivery_key"] in keys):
            return None
        keys.add(item["delivery_key"])
    return value


def _valid_record(item: Mapping[str, object], attempt: object) -> bool:
    revision = item.get("candidate_revision")
    return (type(attempt) is str and 0 < len(attempt) <= 512
            and type(revision) is str and 0 < len(revision) <= 512
            and item.get("attempt_id") == attempt
            and item.get("delivery_key") == _key(attempt, revision)
            and type(item.get("arm")) is str and item["arm"] in {"treatment", "placebo"}
            and type(item.get("scope_ref")) is str and 0 < len(item["scope_ref"]) <= 512
            and all(_digest(item.get(name)) for name in
                    ("facts_sha256", "exposure_sha256", "requirements_sha256", "body_sha256")))


def _zero_edits(runtime: object) -> bool:
    """Never use a model phase label or a bounded recent-changes projection."""
    owner = runtime.owner
    receipts = getattr(owner, "_receipts", ())
    if not _rows(receipts):
        return False
    if any(not isinstance(receipt, Mapping) for receipt in receipts):
        return False
    worker = getattr(runtime, "resume_worker_state", None)
    # Coverage is exposed only before any plausible mutation history. Keep this
    # conservative guard separate from progress authority: unknown edit receipts
    # are not proof of an acted product change, but they do block pre-edit framing.
    for receipt in receipts:
        kind = receipt.get("progress_kind", receipt.get("kind", receipt.get("tool")))
        if kind not in _EDIT_KINDS:
            continue
        status = receipt.get("progress_result_class", receipt.get(
            "result_class", receipt.get("status")))
        if type(status) is not str or status.upper() not in {
                "REFUSED", "REFUSED_PREDISPATCH", "REJECTED",
                "NOT_APPLIED", "NO_CHANGE"}:
            return False
    from kernel_progress_facts import (
        derive_host_progress_facts, host_truth_progress_enabled,
    )
    if host_truth_progress_enabled():
        finish_result = getattr(runtime, "_last_finish_result", None)
        if not isinstance(finish_result, Mapping):
            finish_result = {}
        prior = (worker.get("host_progress_facts")
                 if isinstance(worker, Mapping) else None)
        progress = derive_host_progress_facts(
            edit_receipts=receipts,
            effect_receipts=getattr(runtime, "_effect_application_receipts", ()),
            delivery_manifest=finish_result.get("delivery_manifest"),
            done_facts=finish_result.get("done_facts"),
            product_edits=getattr(owner, "product_edits", 0),
            file_create_results=getattr(owner, "created_paths", ()),
            owner_receipts=receipts,
            prior_facts=prior,
            enabled=True,
        )
        if progress.edits > 0 or progress.first_edit_step is not None:
            return False
    else:
        if worker is not None:
            if (not isinstance(worker, Mapping) or type(worker.get("edits")) is not int
                    or worker["edits"] != 0
                    or worker.get("first_edit_step", "missing") is not None):
                return False
    revision = getattr(owner, "workspace_revision", 0)
    if type(revision) is not int or revision != 0:
        return False
    # Full host step records provide a second mutation witness for native paths.
    ledger = getattr(runtime, "step_ledger", None)
    records = getattr(ledger, "records", ())
    if not _rows(records):
        return False
    for record in records:
        if not isinstance(record, Mapping):
            return False
        if record.get("tool") in _EDIT_KINDS and str(record.get("result", "")).upper() == "APPLIED":
            return False
    return True


def _revision(runtime: object, boundary: str) -> str | None:
    if getattr(runtime, "projector_kind", None) != "rich":
        return None
    view = runtime.current_view(boundary)
    value = getattr(view, "product_revision", None)
    return value if type(value) is str and value and "unknown" not in value.lower() else None


def _sources_current(runtime: object, source_facts: object) -> bool:
    if not _rows(source_facts) or not source_facts:
        return False
    view = runtime.current_view("external")
    files = {**view.product_files, **view.harness_files}
    return all(isinstance(row, Mapping) and _digest(row.get("raw_sha256"))
               and files.get(row.get("path")) == "sha256:" + row["raw_sha256"]
               for row in source_facts)


def _requirements(runtime: object) -> tuple[PublicRequirement, ...] | None:
    from kernel_request_evidence_closure import ticket_identity_for_runtime
    ticket = ticket_identity_for_runtime(runtime)
    ledger = getattr(runtime, "requirements_ledger", None)
    if ticket is None or not isinstance(ledger, Mapping):
        return None
    if len(ticket.text) > 1_048_576:
        return None
    raw = ticket.text.encode("utf-8", errors="strict")
    if len(raw) > 1_048_576:
        return None
    sources, spans, requirements = (ledger.get(name) for name in ("sources", "spans", "requirements"))
    if not all(_rows(value) for value in (sources, spans, requirements)) or not requirements or len(requirements) > 64:
        return None
    if len(sources) > 64 or len(spans) > 512:
        return None
    source_map: dict[str, Mapping[str, object]] = {}
    for source in sources:
        if not isinstance(source, Mapping) or type(source.get("id")) is not str or source["id"] in source_map:
            return None
        source_map[source["id"]] = source
    span_map: dict[str, Mapping[str, object]] = {}
    for span in spans:
        if not isinstance(span, Mapping) or type(span.get("id")) is not str or span["id"] in span_map:
            return None
        span_map[span["id"]] = span
    result: list[PublicRequirement] = []
    for req in requirements:
        if not isinstance(req, Mapping) or not _rows(req.get("source_span_ids")) or len(req["source_span_ids"]) != 1:
            return None
        span = span_map.get(req["source_span_ids"][0])
        if span is None:
            return None
        source = source_map.get(span.get("source_id"))
        if (source is None or source.get("kind") != "ticket" or source.get("text") != ticket.text
                or source.get("sha256") != _sha(raw)
                or span.get("source_sha256") != source["sha256"]):
            return None
        start, end = span.get("start_byte"), span.get("end_byte")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw):
            return None
        quote = raw[start:end].decode("utf-8", errors="strict")
        if quote != span.get("quote") or type(req.get("id")) is not str:
            return None
        result.append(PublicRequirement(req["id"], quote,
                      f"ticket:sha256:{source['sha256']}:{start}-{end}"))
    return tuple(result)


def _requirement_hash(requirements: tuple[PublicRequirement, ...]) -> str:
    return _sha(json.dumps([(row.id, row.text, row.source_ref) for row in requirements],
                          ensure_ascii=False, separators=(",", ":")).encode())


def _message_matches(message: str, record: Mapping[str, object]) -> bool:
    if len(message.encode()) > 65_536 or _sha(message.encode()) != record["exposure_sha256"]:
        return False
    _, marker, facts_text = message.partition("PRE_EDIT_COVERAGE_FACTS_JSON (untrusted data):\n")
    footer = "\nEND_PRE_EDIT_COVERAGE_FACTS"
    if not marker or not facts_text.endswith(footer):
        return False
    facts_text = facts_text[:-len(footer)]
    if _sha(facts_text.encode()) != record["facts_sha256"]:
        return False
    facts = json.loads(facts_text)
    requirements = tuple(PublicRequirement(row["id"], row["text"], row["source_ref"])
                         for row in facts["requirements"])
    return (facts["candidate_revision"] == record["candidate_revision"]
            and facts["scope_ref"] == record["scope_ref"]
            and _requirement_hash(requirements) == record["requirements_sha256"])


def prepare(runtime: object, delivery: object, messages: list[dict[str, object]],
            tools: object, capture: object, *, token_counter) -> list[dict[str, object]]:
    """Remove owned prior exposures and prepare one grounded advisory, if eligible."""
    if not enabled(runtime):
        return messages
    capture.preedit_coverage_claim = None
    capture.preedit_coverage_runtime = runtime
    capture.preedit_coverage_status = {"status": "NOOP", "reason": "unproved_host_or_inventory"}
    try:
        state = _state(runtime)
        pending = getattr(runtime, "_preedit_coverage_pending", {})
        if state is not None:
            owned = {item["exposure_sha256"] for item in state["seen"]}
            if isinstance(pending, dict):
                owned.update(pending)
            messages = [row for row in messages if not (
                row.get("role") == "user" and type(row.get("content")) is str
                and _sha(row["content"].encode()) in owned)]
        packet = getattr(delivery, "packet", None)
        if state is None or packet is None or not _zero_edits(runtime):
            return messages
        revision = _revision(runtime, "run")
        requirements = _requirements(runtime)
        if (revision is None or requirements is None or len(state["seen"]) >= 256
                or not _sources_current(runtime, delivery.source_facts)):
            return messages
        sites = tuple(CoverageSite(**{key: site[key] for key in
                      ("source_handle", "path", "owner", "start_byte", "end_byte")}, id=site["site_id"])
                      for group in packet.groups for site in group["sites"])
        groups = tuple(group["group_id"] for group in packet.groups)
        scope = "census:constructor-groups:" + _sha(json.dumps(groups, separators=(",", ":")).encode())
        coverage = packet.coverage
        inventory = ScopeInventory(revision, packet.index_revision, scope,
            "All indexed sites in the eligible shown cross-file constructor groups; excludes other relations and does not establish task or transitive-import completeness.",
            sites, True, coverage["unresolved_count"], 0, coverage["files_partial"],
            coverage["scan_truncated"], "exact" if all(group["total_relation"] == "exact" for group in packet.groups) else "lower_bound")
        sources = tuple(BoundSource(revision, row["handle"], row["path"], row["text"], row["sha256"])
                        for row in delivery.source_facts)
        plan = render_preedit_coverage(PreeditInput(state["attempt_id"], revision, inventory,
            sources, requirements, seen_delivery_keys=tuple(item["delivery_key"] for item in state["seen"])),
            enabled=True, arm=state["arm"])
        capture.preedit_coverage_status = {"status": plan.status, "reason": plan.reason}
        if plan.status != "READY" or _revision(runtime, "external") != revision:
            if plan.status == "READY":
                capture.preedit_coverage_status = {"status": "NOOP", "reason": "candidate_changed"}
            return messages
        candidate = [row for row in messages if not (
            row.get("role") == "user" and row.get("content") == plan.message)]
        candidate.append({"role": "user", "content": plan.message})
        tokens = token_counter({"messages": candidate, "tools": tools})
        if type(tokens) is not int or tokens < 0 or tokens > runtime.frame_ceiling:
            capture.preedit_coverage_status = {"status": "NOOP", "reason": "request_budget_unavailable_or_exceeded"}
            return messages
        capture.preedit_coverage_claim = {
            "attempt_id": state["attempt_id"], "candidate_revision": revision,
            "delivery_key": plan.delivery_key, "arm": state["arm"], "scope_ref": scope,
            "facts_sha256": plan.facts_sha256, "exposure_sha256": plan.exposure_sha256,
            "requirements_sha256": _requirement_hash(requirements),
            "message": plan.message,
        }
        runtime._preedit_coverage_pending = {plan.exposure_sha256: plan.message}
        return candidate
    except (AttributeError, KeyError, LookupError, OSError, RuntimeError, TypeError, ValueError) as error:
        capture.preedit_coverage_status = {"status": "NOOP", "reason": "host_input_unavailable"}
        logging.getLogger(__name__).debug("pre-edit coverage unavailable", extra={"error_type": type(error).__name__})
        return messages


def submission(capture: object, wire: bytes) -> dict[str, object] | None:
    """Qualify the exact admitted serializer bytes before their durable event."""
    runtime = getattr(capture, "preedit_coverage_runtime", None)
    claim = getattr(capture, "preedit_coverage_claim", None)
    if runtime is None or not enabled(runtime) or not isinstance(claim, Mapping):
        return None
    messages = json.loads(wire).get("messages", ())
    message = claim.get("message")
    if not any(isinstance(row, Mapping) and row.get("role") == "user"
               and row.get("content") == message for row in messages):
        return None
    try:
        state = _state(runtime)
        requirements = _requirements(runtime)
        if (state is None or not _zero_edits(runtime)
                or _revision(runtime, "run") != claim["candidate_revision"]
                or requirements is None or _requirement_hash(requirements) != claim["requirements_sha256"]
                or any(item["delivery_key"] == claim["delivery_key"] for item in state["seen"])):
            raise RuntimeError("pre-edit coverage binding changed before submission")
        if (type(message) is not str or not _message_matches(message, claim)
                or sum(isinstance(row, Mapping) and row.get("role") == "user"
                       and row.get("content") == message for row in messages) != 1):
            raise RuntimeError("pre-edit coverage serialized exposure is invalid")
        record = {key: value for key, value in claim.items() if key != "message"}
        record["body_sha256"] = _sha(wire)
        if not _valid_record(record, state["attempt_id"]) or record["arm"] != state["arm"]:
            raise RuntimeError("pre-edit coverage delivery identity is invalid")
        return record
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError("pre-edit coverage submission could not be authenticated") from error


def committed(capture: object, record: object) -> None:
    if not isinstance(record, Mapping):
        return
    runtime = getattr(capture, "preedit_coverage_runtime", None)
    state = _state(runtime) if runtime is not None else None
    if state is not None and not any(item["delivery_key"] == record["delivery_key"] for item in state["seen"]):
        state["seen"].append(dict(record))


def restore(runtime: object, records: Sequence[Mapping[str, object]], saved_state: object) -> None:
    """Join checkpoint state with the canonical journal's durable submissions."""
    if not enabled(runtime):
        return
    runtime._preedit_coverage_pending = {}
    runtime._preedit_coverage_state = saved_state
    if isinstance(saved_state, dict):
        saved_state["resume_verified"] = True
    state = _state(runtime)
    if state is None:
        runtime._preedit_coverage_state = None
        return
    seen: dict[str, dict[str, object]] = {}
    for event in records:
        if not isinstance(event, Mapping):
            runtime._preedit_coverage_state = None
            return
        item = event.get(SUBMISSION_FIELD)
        if item is None:
            continue
        try:
            wire = event.get("payload_utf8")
            if (event.get("kind") != "request_submitted" or event.get("run_id") != state["attempt_id"]
                    or not isinstance(item, Mapping) or not _valid_record(item, state["attempt_id"])
                    or item["arm"] != state["arm"] or type(wire) is not str
                    or _sha(wire.encode()) != item["body_sha256"]
                    or event.get("payload_hash") != "sha256:" + item["body_sha256"]):
                raise ValueError("invalid pre-edit submission")
            messages = json.loads(wire).get("messages", ())
            if sum(isinstance(row, Mapping) and row.get("role") == "user"
                   and type(row.get("content")) is str
                   and _message_matches(row["content"], item) for row in messages) != 1:
                raise ValueError("missing exact pre-edit exposure")
            key = item["delivery_key"]
            if key in seen:
                raise ValueError("conflicting once delivery")
            seen[key] = dict(item)
            if len(seen) > 256:
                raise ValueError("too many durable pre-edit deliveries")
        except (KeyError, TypeError, ValueError):
            runtime._preedit_coverage_state = None
            return
    if len(seen) > 256 or any(seen.get(item["delivery_key"]) != item for item in state["seen"]):
        runtime._preedit_coverage_state = None
        return
    state["seen"] = list(seen.values())

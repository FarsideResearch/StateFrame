"""Small, opt-in presentation arms for state continuation frame continuation.

The module owns only deterministic presentation helpers. It does not inspect
the repository, run checks, or infer a result from model prose. The runtime
passes host-owned frame records in and receives JSON-shaped messages out.
"""
from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping, Sequence


STATE_PLACEMENT_ENV = "STATEFRAME_STATE_PLACEMENT"
THINK_PREFILL_ENV = "STATEFRAME_THINK_PREFILL"
WORKED_EXAMPLE_ENV = "STATEFRAME_WORKED_EXAMPLE"
CHECK_RENDERING_ENV = "STATEFRAME_CHECK_RENDERING"

_PLACEMENTS = frozenset({"off", "end"})
_PREFILLS = frozenset({"off", "on", "user", "self_plan"})
_WORKED = frozenset({"off", "on", "instruction"})
_PASS = frozenset({"PASS", "PASSED", "GREEN", "SATISFIED", "ACCEPTED"})
_COUNTEREXAMPLE = frozenset({"COUNTEREXAMPLE", "FAIL", "FAILED", "RED"})
_STALE = frozenset({"STALE", "CONFLICT", "INVALID", "REJECTED"})
_TERMINAL_OPEN = frozenset({
    "PASS", "PASSED", "GREEN", "SATISFIED", "CLOSED", "DONE", "RESOLVED",
})

# The words are intentionally duplicated from probes/test_prompt_v2.py.  The
# production renderer must not import a test module, but the screen arm must
# remain mechanically protected from evaluation-task vocabulary.
TOPIC_WORD_LEAK_GUARD = frozenset({
    "visibility", "memo", "invoice", "marketplace", "cafe", "receipt", "alert",
    "podcast", "profile", "notify", "priority", "urgent", "library", "expense",
    "calendar", "helpdesk", "delivery", "account_rename", "studio", "museum", "bike",
    "ferry", "recipe", "photo", "vault", "weather", "adoption", "sample_intake",
    "archive", "garden",
})


def _mode(name: str, allowed: frozenset[str], default: str = "off",
          environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = source.get(name, default)
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}")
    return value


def state_placement_mode(environ: Mapping[str, str] | None = None) -> str:
    return _mode(STATE_PLACEMENT_ENV, _PLACEMENTS, environ=environ)


def think_prefill_mode(environ: Mapping[str, str] | None = None) -> str:
    return _mode(THINK_PREFILL_ENV, _PREFILLS, environ=environ)


def worked_example_mode(environ: Mapping[str, str] | None = None) -> str:
    return _mode(WORKED_EXAMPLE_ENV, _WORKED, environ=environ)


def check_rendering_mode(environ: Mapping[str, str] | None = None) -> str:
    """Optional fourth arm for the check-line presentation itself.

    It is separate from placement because the placement experiment must remain
    a pure permutation of the existing state block.
    """
    return _mode(CHECK_RENDERING_ENV, frozenset({"off", "on"}), environ=environ)


def configuration(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    return {
        "state_placement": state_placement_mode(environ),
        "think_prefill": think_prefill_mode(environ),
        "worked_example": worked_example_mode(environ),
    }


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    config = configuration(environ)
    return any(value != "off" for value in config.values()) or (
        check_rendering_mode(environ) == "on")


def _content(message: object) -> str:
    if not isinstance(message, Mapping):
        return ""
    value = message.get("content")
    return value if isinstance(value, str) else ""


def is_state_block(message: object) -> bool:
    return _content(message).startswith("[StateFrame]\n")


def state_block_index(messages: Sequence[Mapping[str, object]]) -> int:
    indexes = [index for index, message in enumerate(messages) if is_state_block(message)]
    if len(indexes) != 1:
        return -1
    return indexes[0]


def state_placement(messages: Sequence[dict], mode: str | None = None) -> list[dict]:
    """Move exactly one authoritative StateFrame block to the end.

    The OFF branch copies the sequence without changing message dictionaries or
    their serialized content.  The ON branch only pops and appends the existing
    block; it never edits, duplicates, or synthesizes state text.
    """
    selected = state_placement_mode() if mode is None else mode
    if selected not in _PLACEMENTS:
        raise ValueError(f"{STATE_PLACEMENT_ENV} must be off or end")
    result = list(messages)
    if selected == "off":
        return result
    indexes = [index for index, message in enumerate(result) if is_state_block(message)]
    if len(indexes) != 1:
        raise ValueError("state placement requires exactly one [StateFrame] block")
    index = indexes[0]
    if index != len(result) - 1:
        block = result.pop(index)
        result.append(block)
    return result


def _rows(value: object, *, depth: int = 0) -> list[dict[str, object]]:
    if depth > 5 or value is None:
        return []
    if isinstance(value, Mapping):
        result = [dict(value)]
        for key in ("value", "fact", "facts", "receipts", "checks", "items",
                    "open_work", "open_items", "assessments"):
            if key in value:
                result.extend(_rows(value[key], depth=depth + 1))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[dict[str, object]] = []
        for item in value:
            result.extend(_rows(item, depth=depth + 1))
        return result
    return []


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping) or (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return " ".join(str(value).split())


def _first(row: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _value_row(row: Mapping[str, object]) -> dict[str, object]:
    value = row.get("value")
    return dict(value) if isinstance(value, Mapping) else {}


def _status(row: Mapping[str, object]) -> str:
    value = _value_row(row)
    candidate = _first(value, "status", "result", "result_class", "verdict")
    if candidate is None:
        candidate = _first(row, "result", "result_class", "verdict", "status")
    return str(candidate or "").upper()


def _scope(row: Mapping[str, object], revision: str) -> str:
    value = _value_row(row)
    scope = _first(row, "scope", "scope_ref", "binding", "revision", "snapshot_digest")
    if scope is None:
        scope = _first(value, "scope", "scope_ref", "revision", "snapshot_digest")
    if isinstance(scope, Mapping):
        scope = _first(scope, "product_revision", "revision", "scope")
    return _text(scope) or revision or "current workspace"


def _revision_forms(value: object) -> set[str]:
    text = _text(value)
    if not text:
        return set()
    values = {text}
    if text.startswith("sha256:"):
        values.add(text[7:])
    else:
        values.add("sha256:" + text)
    return values


def _revision_matches(candidate: object, accepted: set[str]) -> bool:
    forms = _revision_forms(candidate)
    for form in forms:
        if form in accepted:
            return True
        # Host-closure receipts in older journals retain a short digest while
        # the enclosing closure stores the full digest.  A sufficiently long
        # prefix is still a scope reference; arbitrary short labels are not.
        if len(form) >= 8 and any(
                len(target) >= 8
                and (form.startswith(target) or target.startswith(form))
                for target in accepted):
            return True
    return False


def _in_scope(row: Mapping[str, object], revision: str,
              frame: Mapping[str, object] | None = None) -> bool:
    if str(row.get("status", "")).upper() in _STALE:
        return False
    if row.get("check_valid_for_workspace") is False:
        return False
    value = _value_row(row)
    if value.get("check_valid_for_workspace") is False:
        return False
    candidates = []
    for source in (row, value):
        candidates.extend(source.get(key) for key in (
            "revision", "snapshot_digest", "artifact_revision", "product_revision"))
    candidates = [str(item) for item in candidates if item is not None]
    if not candidates or not revision or revision in {"current", "unknown"}:
        return True
    accepted = _revision_forms(revision)
    if isinstance(frame, Mapping):
        closure = frame.get("host_closure")
        if isinstance(closure, Mapping):
            accepted.update(_revision_forms(closure.get("snapshot")))
    return any(_revision_matches(candidate, accepted) for candidate in candidates)


def _check_id(row: Mapping[str, object]) -> str:
    value = _value_row(row)
    return _text(_first(row, "check_id", "subject", "id", "name") or
                 _first(value, "check_id", "subject", "id", "name")) or "CHECK"


def _raw_check_id(row: Mapping[str, object]) -> str:
    value = _value_row(row)
    return _text(_first(row, "check_id", "subject", "id", "name") or
                 _first(value, "check_id", "subject", "id", "name"))


def _assessment_details(frame: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Join host-closure expected/observed values onto receipt-only rows."""
    details: dict[str, dict[str, object]] = {}
    for key in ("host_closure", "host_facts", "obligations", "verification"):
        for row in _rows(frame.get(key)):
            sources: list[Mapping[str, object]] = [row]
            value = _value_row(row)
            if value:
                sources.append(value)
            counterexample = _first(row, "counterexample")
            if counterexample is None:
                counterexample = _first(value, "counterexample")
            if isinstance(counterexample, Mapping):
                sources.append(counterexample)
            identifiers: list[str] = []
            for source in sources:
                for field in ("failing_check_ids", "check_ids", "checks"):
                    candidate = source.get(field)
                    if isinstance(candidate, str):
                        identifiers.append(candidate)
                    elif isinstance(candidate, Sequence):
                        identifiers.extend(str(item) for item in candidate)
            if not identifiers:
                continue
            expected = next((candidate for source in sources
                             for candidate in (_first(source, "expected", "required"),)
                             if candidate is not None), None)
            observed = next((candidate for source in sources
                            for candidate in (_first(source, "observed", "actual", "observation"),)
                            if candidate is not None), None)
            assertion = next((candidate for source in sources
                              for candidate in (_first(source, "assertion", "clause"),)
                              if candidate is not None), None)
            if expected is None and observed is None and assertion is None:
                continue
            detail = {"expected": expected, "observed": observed, "assertion": assertion}
            for identifier in identifiers:
                details.setdefault(identifier, detail)
    return details


def _check_rows(frame: Mapping[str, object]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for key in ("host_facts", "host_check_receipts", "check_receipts", "test_receipts",
                "receipts", "host_closure", "verification", "current_public_checks"):
        candidates.extend(_rows(frame.get(key)))
    # Host facts are authoritative.  Preserve their order, then de-duplicate
    # mirrored receipt rows by (check id, status, revision).
    result = []
    seen = set()
    for row in candidates:
        predicate = str(row.get("predicate", ""))
        status = _status(row)
        if not _raw_check_id(row):
            continue
        if predicate and predicate not in {"host_check_result", "test_result", "check_result"}:
            continue
        if status not in _PASS | _COUNTEREXAMPLE:
            continue
        revision_value = _first(row, "revision", "snapshot_digest")
        closure = frame.get("host_closure")
        closure_snapshot = (closure.get("snapshot")
                            if isinstance(closure, Mapping) else None)
        revision_key = _text(revision_value)
        if (revision_value is not None and closure_snapshot is not None
                and _revision_matches(revision_value, _revision_forms(closure_snapshot))):
            revision_key = "closure:" + _text(closure_snapshot)
        key = (_check_id(row), status, revision_key)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _current_revision(frame: Mapping[str, object]) -> str:
    for key in ("current_revision", "revision", "product_revision", "workspace_revision"):
        value = frame.get(key)
        if value is not None and not isinstance(value, Mapping):
            return _text(value)
    binding = frame.get("binding")
    if isinstance(binding, Mapping):
        return _text(_first(binding, "product_revision", "revision"))
    revisions = frame.get("artifact_revisions")
    source_access = frame.get("source_access")
    source_path = (source_access.get("path") if isinstance(source_access, Mapping)
                   else None)
    if isinstance(revisions, Mapping) and isinstance(source_path, str):
        value = revisions.get(source_path)
        if value is not None and not isinstance(value, Mapping):
            return _text(value)
    return "current"


def _check_parts(row: Mapping[str, object], revision: str,
                 frame: Mapping[str, object] | None = None) -> tuple[str, str, str, str]:
    value = _value_row(row)
    status = _status(row)
    check_id = _check_id(row)
    detail = (_assessment_details(frame).get(check_id, {})
              if isinstance(frame, Mapping) else {})
    assertion = _first(value, "assertion", "clause", "expected")
    if assertion is None:
        assertion = _first(row, "assertion", "clause", "expected")
    if assertion is None:
        assertion = detail.get("assertion") or detail.get("expected")
    observed = _first(value, "observed", "actual", "observation")
    if observed is None:
        observed = _first(row, "observed", "actual", "observation", "detail")
    if observed is None:
        observed = detail.get("observed")
    return (check_id, status, _text(assertion) or "unspecified assertion",
            _text(observed) or status, _scope(row, revision))


def render_check_lines(frame: Mapping[str, object]) -> list[str]:
    """Render one line per current check with assertion, observation, scope."""
    revision = _current_revision(frame)
    lines = []
    for row in _check_rows(frame):
        if not _in_scope(row, revision, frame):
            continue
        check_id, status, assertion, observed, scope = _check_parts(row, revision, frame)
        lines.append(
            f"{check_id} {status} — assertion={assertion}; observation={observed}; scope={scope}")
    return lines


def _open_item_ids(frame: Mapping[str, object]) -> list[str]:
    found: list[str] = []
    for key in ("open_work", "open_items", "pending", "obligations", "host_closure",
                "completion_contract", "completion_verdict"):
        for row in _rows(frame.get(key)):
            status = str(_first(row, "status", "verdict", "state") or "").upper()
            if status in _TERMINAL_OPEN:
                continue
            identifier = _first(row, "id", "check_id", "obligation_id", "work_id", "name")
            if identifier:
                found.append(str(identifier))
    return list(dict.fromkeys(found))


def render_host_delta(*, previous_action: object, host_consequence: object,
                      still_open: object) -> str:
    return (f"Previous action {_text(previous_action) or 'NONE'} → host consequence "
            f"{_text(host_consequence) or 'NONE'} → still open "
            f"{_text(still_open) or 'NONE'}")


def render_frame_delta(frame: Mapping[str, object]) -> str:
    previous = frame.get("last_action") or frame.get("previous_action") or "NONE"
    consequence = "NONE"
    rows = _check_rows(frame)
    if rows:
        check_id, status, _assertion, _observed, _scope_value = _check_parts(
            rows[-1], _current_revision(frame), frame)
        consequence = f"{check_id} {status}"
    return render_host_delta(
        previous_action=_text(previous), host_consequence=consequence,
        still_open=", ".join(_open_item_ids(frame)) or "NONE")


def _prefill_rows(frame: Mapping[str, object]) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    revision = _current_revision(frame)
    details = _assessment_details(frame)
    passed: list[tuple[str, str]] = []
    counterexamples: list[tuple[str, str, str]] = []
    for row in _check_rows(frame):
        if not _in_scope(row, revision, frame):
            continue
        check_id, status, _assertion, observed, _scope_value = _check_parts(row, revision, frame)
        value = _value_row(row)
        expected = _first(value, "expected")
        if expected is None:
            expected = _first(row, "expected")
        if expected is None:
            expected = details.get(check_id, {}).get("expected")
        if status in _PASS:
            passed.append((check_id, status))
        elif status in _COUNTEREXAMPLE:
            counterexamples.append((check_id, _text(expected) or "unknown", observed))
    return passed, counterexamples


def build_prefill_seed(frame: Mapping[str, object]) -> str:
    """Build the exact short host-fact seed used by both prefill controls."""
    revision = _current_revision(frame)
    passed, counterexamples = _prefill_rows(frame)
    parts = [f"At {revision}"]
    if passed:
        parts.append(",".join(item[0] for item in passed) + " PASS within scope.")
    for check_id, expected, observed in counterexamples:
        parts.append(f"{check_id} COUNTEREXAMPLE expected {expected} observed {observed}.")
    parts.append("The unresolved decision is")
    return " ".join(parts)


def _worked_example_body() -> str:
    return (
        "[Worked continuation — synthetic]\n"
        "Frame: A1 asserts a quartz token keeps its count; host observed A1 PASS at r4. "
        "Open item O2: blank input returns an empty sequence.\n"
        "Decision: preserve the confirmed path and address O2 with one bounded change.\n"
        "Next action: run_tests({\"target\":\"tests/test_quartz.py\",\"why\":\"check O2\"})"
    )


def _worked_instruction_body() -> str:
    example = _worked_example_body()
    value = (
        "[Continuation policy — synthetic]\n"
        "Use the frame assertion and host observation as premises; preserve confirmed work "
        "and address the open item with one bounded change.\n"
        "Next action: run_tests({\"target\":\"tests/test_quartz.py\",\"why\":\"check O2\"})"
    )
    if len(value) < len(example):
        value += " " * (len(example) - len(value))
    return value[:len(example)]


def worked_example(mode: str | None = None) -> str:
    selected = worked_example_mode() if mode is None else mode
    if selected not in _WORKED:
        raise ValueError(f"{WORKED_EXAMPLE_ENV} must be off, on or instruction")
    if selected == "off":
        return ""
    return _worked_example_body() if selected == "on" else _worked_instruction_body()


def augment_state_facts(facts: str, frame: Mapping[str, object], *,
                        check_mode: str = "off", worked_mode: str = "off") -> str:
    """Add opt-in clarity blocks without modifying the legacy facts string."""
    if check_mode not in {"off", "on"}:
        raise ValueError("check_mode must be off or on")
    additions = []
    if check_mode == "on":
        additions.extend(render_check_lines(frame))
        additions.append(render_frame_delta(frame))
    example = worked_example(worked_mode)
    if example:
        additions.append(example)
    return facts if not additions else facts + "\n" + "\n".join(additions)


def apply_think_prefill(messages: Sequence[dict], seed: str, mode: str) -> dict[str, object]:
    """Compose the two controls; ``on`` is an open assistant continuation."""
    if mode not in _PREFILLS:
        raise ValueError(f"{THINK_PREFILL_ENV} must be off, on, user or self_plan")
    result = [copy.deepcopy(message) for message in messages]
    metadata: dict[str, object] = {
        "messages": result,
        "prefill_text": seed if mode not in {"off", "self_plan"} else "",
        "generated_token_exclusion": "prompt_prefill"
        if mode not in {"off", "self_plan"} else None,
        "prefill_mode": mode,
    }
    if mode == "off":
        return metadata
    # The self-plan arm owns its seed only after the preceding no-thinking
    # request returns. Keep this frame helper a no-op for that mode so the
    # runtime cannot accidentally prefill the thinking call before a plan is
    # available.
    if mode == "self_plan":
        metadata["self_plan_pending"] = True
        return metadata
    if not isinstance(seed, str) or not seed:
        raise ValueError("think prefill requires non-empty host-derived seed")
    if mode == "user":
        # The control arm is a suffix of the complete user turn.  It must not
        # land inside an earlier StateFrame when source or toy-example user
        # messages follow it.
        target = next((message for message in reversed(result)
                       if message.get("role") == "user"), None)
        if target is None:
            target = {"role": "user", "content": ""}
            result.append(target)
        current = target.get("content")
        target["content"] = (current if isinstance(current, str) else "") + "\n" + seed
        metadata["messages"] = result
        return metadata
    # MiniCPM5's llama.cpp continuation branch consumes this assistant message
    # with continue_final_message=true and renders an open <think> prefix.
    # An empty string keeps the OpenAI-compatible message schema valid; the
    # reasoning sibling carries the continuation text.
    result.append({"role": "assistant", "content": "", "reasoning_content": seed})
    metadata["messages"] = result
    metadata["continue_final_message"] = "reasoning_content"
    return metadata


def render_minicpm5_prefill(request: Mapping[str, object]) -> str:
    """Offline rendering proof for the MiniCPM5 llama.cpp continuation shape."""
    messages = request.get("messages", ())
    rendered: list[str] = []
    last = len(messages) - 1 if isinstance(messages, Sequence) else -1
    for index, message in enumerate(messages if isinstance(messages, Sequence) else ()):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "user"))
        if role == "assistant" and index == last and request.get("continue_final_message"):
            rendered.append("<|im_start|>assistant\n<think>\n"
                            + str(message.get("reasoning_content") or ""))
            continue
        content = message.get("content")
        rendered.append(f"<|im_start|>{role}\n{content or ''}<|im_end|>\n")
    return "".join(rendered)


def strip_prefill(reasoning: object, seed: str) -> str:
    text = reasoning if isinstance(reasoning, str) else ""
    if seed and text.startswith(seed):
        text = text[len(seed):]
        if text.startswith("\n"):
            text = text[1:]
    return text


def prefill_token_count(text: object) -> int:
    return len(str(text or "").split())


# The note specifies the first 512 *new reasoning tokens*.  The pattern is
# deliberately about common re-derivation openings, not a semantic verdict.
RESTART_LANGUAGE_RE = re.compile(
    r"(?ix)\b(?:"
    r"let\s+me\s+(?:understand|restate|reanaly[sz]e)\s+the\s+(?:task|problem)|"
    r"(?:first,?\s+)?(?:i\s+need\s+to\s+)?(?:understand|re-?read|restate)\s+the\s+"
    r"(?:task|problem)|"
    r"let['’]?s\s+start\s+by\s+understanding\s+the\s+(?:task|problem)"
    r")\b"
)


__all__ = [
    "CHECK_RENDERING_ENV", "RESTART_LANGUAGE_RE", "TOPIC_WORD_LEAK_GUARD",
    "apply_think_prefill", "augment_state_facts", "build_prefill_seed", "check_rendering_mode",
    "configuration", "enabled", "is_state_block", "prefill_token_count", "render_check_lines",
    "render_frame_delta", "render_host_delta", "render_minicpm5_prefill", "state_block_index",
    "state_placement", "state_placement_mode", "strip_prefill", "think_prefill_mode",
    "worked_example", "worked_example_mode",
]


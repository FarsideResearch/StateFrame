"""Wire-level cache-prefix checks for model-free engine runs.

A per-turn request is stable when the bytes the engine sent for the protected bands (system, task, repo index, and
retained source only after a write/edit result) still form a prefix of the next request. Stable-prefix end E comes from
kernel_cache_layout bands, never from the observed longest common prefix. Cached tokens are scored as the last
<|im_start|> boundary B at or before the LCP, which is what provider reuse can actually reach.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

from enginecheck import ROOT, HarnessError, Trace
from stateframe.engine import kernel_cache_layout as layout

RELAY = "http://127.0.0.1:18592"
RELAY_RETRIES = 3
RELAY_TIMEOUT_SECONDS = 120
OPEN_ASSISTANT = "<|im_start|>assistant"
STABLE_BANDS = frozenset({"system", "task", "repo_index", "retained_source"})
HARD_STABLE_BANDS = frozenset({"system", "task", "repo_index"})
_RELAY_PATHS = frozenset({"/apply-template", "/tokenize"})
_WRITE_EDIT_TOOL_NAMES = frozenset({
    "apply_patch", "create_file", "edit", "edit_file", "file_edit", "file_write",
    "patch_file", "replace_in_file", "str_replace", "write", "write_file", "write_text",
})
_APPLIED_OK = frozenset({"APPLIED", "SUCCESS", "SUCCEEDED", "COMPLETED"})
SHIPPED_PROFILE = ROOT / "src" / "stateframe" / "profiles" / "r2hc3.json"
# Launcher-only policy, not an engine switch: the shipped runtime excludes it (tests/conftest.py shipped_runtime_profile).
_LAUNCHER_ONLY_SWITCHES = frozenset({"STATEFRAME_PROTECT_TESTS"})


@dataclass(frozen=True)
class Wire:
    turn: int
    raw: bytes
    body: dict[str, object]
    tokens: tuple[int, ...] = ()
    stable_prefix_end: int = 0
    stable_end_message: int = -1
    retained_source_prefix_ends: dict[int, int] = field(default_factory=dict)
    # Set for a refused, unsent or response-less turn: it is no sample and never becomes the baseline.
    no_sample_reason: str | None = None


@dataclass(frozen=True)
class PairMeasurement:
    turn: int
    lcp: int
    stable_prefix_end: int
    cached_tokens: int
    failure: str | None = None


@dataclass(frozen=True)
class WireReport:
    pairs: tuple[PairMeasurement, ...]
    cached_tokens: int
    no_sample: tuple[int, ...]
    failures: tuple[str, ...]


def shipped_switches() -> dict[str, str]:
    """The engine switches of the shipped profile, minus launcher-only policy. Use as Scenario.switches."""
    try:
        profile = json.loads(SHIPPED_PROFILE.read_text(encoding="utf-8"))
        switches = profile["switches"]
    except (OSError, ValueError, KeyError) as error:
        raise HarnessError("shipped_profile_unreadable") from error
    if not isinstance(switches, dict) or not all(isinstance(v, str) for v in switches.values()):
        raise HarnessError("shipped_profile_malformed")
    return {name: value for name, value in switches.items() if name not in _LAUNCHER_ONLY_SWITCHES}


class OfflineTokenizer:
    """Client for the offline relay: /apply-template renders a prompt, /tokenize returns ids. Raises on any relay fault."""

    def __init__(self, endpoint: str | None = None) -> None:
        self.endpoint = RELAY if endpoint is None else endpoint
        self._prompts: dict[bytes, str] = {}
        self._tokens: dict[str, tuple[int, ...]] = {}

    def render(self, messages: Sequence[Mapping[str, object]], tools: object) -> str:
        request_body = {"messages": list(messages), "tools": tools}
        key = _canonical(request_body)
        if key not in self._prompts:
            prompt = _post_json("/apply-template", request_body, endpoint=self.endpoint).get("prompt")
            if not isinstance(prompt, str):
                raise HarnessError("tokenizer_relay_malformed_prompt")
            self._prompts[key] = prompt
        return self._prompts[key]

    def tokenize(self, prompt: str) -> tuple[int, ...]:
        if prompt not in self._tokens:
            values = _post_json("/tokenize", {"content": prompt, "add_special": False},
                                endpoint=self.endpoint).get("tokens")
            if not isinstance(values, list) or not all(type(token) is int for token in values):
                raise HarnessError("tokenizer_relay_malformed_tokens")
            self._tokens[prompt] = tuple(values)
        return self._tokens[prompt]

    def marker(self) -> int:
        marker_tokens = self.tokenize("<|im_start|>")
        if len(marker_tokens) != 1:
            raise HarnessError("tokenizer_relay_bad_im_start")
        return marker_tokens[0]


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _post_json(path: str, body: Mapping[str, object], *, endpoint: str) -> dict[str, object]:
    """POST to an allowlisted relay path. Connection faults retry, then raise tokenizer_relay_unreachable."""
    if path not in _RELAY_PATHS:
        raise HarnessError("offline_endpoint_refused")
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(endpoint + path, payload, {"Content-Type": "application/json"})
    for attempt in range(RELAY_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=RELAY_TIMEOUT_SECONDS) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            # The relay answered: retrying would hide a real fault behind "unreachable".
            raise HarnessError("tokenizer_relay_http_error") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt + 1 == RELAY_RETRIES:
                raise HarnessError("tokenizer_relay_unreachable") from error
            time.sleep(1)
            continue
        try:
            result = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HarnessError("tokenizer_malformed_response") from error
        if not isinstance(result, dict):
            raise HarnessError("tokenizer_malformed_response")
        return result
    raise HarnessError("tokenizer_relay_unreachable")


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (first, second) in enumerate(zip(left, right)):
        if first != second:
            return index
    return min(len(left), len(right))


def scorewire_cached_tokens(tokens: Sequence[int], lcp: int, im_start_token: int) -> int:
    """Index of the last <|im_start|> at or before the LCP. A boundary at index b covers tokens [0, b)."""
    if lcp < 0:
        raise ValueError("LCP cannot be negative")
    boundary = 0
    for index, token in enumerate(tokens[: lcp + 1]):
        if token == im_start_token:
            boundary = index
    return boundary


def _message_band(module: ModuleType, message: Mapping[str, object]) -> str:
    """Map kernel_cache_layout's message kinds onto the bands this check protects."""
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    kind = module.layout_message_kind(message)
    if message.get("role") in {"system", "developer"} or kind == "protocol":
        return "system"
    if kind == "ticket" or text.startswith("Task:"):
        return "task"
    if (kind == "ticket_facts" or text.startswith(module._TICKET_FACT_PREFIXES)
            or text.startswith("[FRAME_CLARITY_BOOTSTRAP]")):
        return "repo_index"
    if kind == "source" or text.startswith(module._SOURCE_PREFIXES):
        return "retained_source"
    return kind


def stable_message_end(body: Mapping[str, object], module: ModuleType) -> int:
    """Index of the last request message in a stable band, or -1."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HarnessError("request_messages_missing")
    indexes = [index for index, message in enumerate(messages)
               if isinstance(message, Mapping) and _message_band(module, message) in STABLE_BANDS]
    return max(indexes, default=-1)


def _is_write_edit_name(name: object) -> bool:
    if not isinstance(name, str):
        return False
    normalized = name.strip().lower().replace("-", "_")
    return (normalized in _WRITE_EDIT_TOOL_NAMES or normalized.startswith(("edit_", "write_"))
            or normalized.endswith(("_edit", "_write")))


def _has_write_edit_tool_result(body: Mapping[str, object]) -> bool:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    names_by_id: dict[str, str] = {}
    for message in messages:
        calls = message.get("tool_calls") if isinstance(message, Mapping) else None
        for call in calls if isinstance(calls, list) else []:
            function = call.get("function") if isinstance(call, Mapping) else None
            if (isinstance(function, Mapping) and isinstance(function.get("name"), str)
                    and isinstance(call.get("id"), str)):
                names_by_id[call["id"]] = function["name"]
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") not in {"tool", "function"}:
            continue
        name = message.get("name")
        if not isinstance(name, str):
            call_id = message.get("tool_call_id")
            name = names_by_id.get(call_id) if isinstance(call_id, str) else None
        if _is_write_edit_name(name):
            return True
    return False


def _applied_edit_records(body: Mapping[str, object]) -> set[bytes]:
    """Applied-edit receipts StateFrame writes into a [StateFrame] message instead of a tool-role result."""
    messages = body.get("messages")
    records: set[bytes] = set()
    for message in messages if isinstance(messages, list) else []:
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or "[StateFrame]" not in content:
            continue
        start = content.find("{", content.find("[StateFrame]") + len("[StateFrame]"))
        try:
            payload, _end = json.JSONDecoder().raw_decode(content[start:]) if start >= 0 else (None, 0)
        except json.JSONDecodeError:
            continue
        state = payload.get("state") if isinstance(payload, Mapping) else None
        framefit = state.get("framefit5") if isinstance(state, Mapping) else None
        ledger = framefit.get("action_ledger") if isinstance(framefit, Mapping) else None
        applied = ledger.get("applied_edits") if isinstance(ledger, Mapping) else None
        for record in applied if isinstance(applied, list) else []:
            if not isinstance(record, Mapping) or not _is_write_edit_name(record.get("action_kind")):
                continue
            result = record.get("result_class")
            if isinstance(result, str) and result.upper() not in _APPLIED_OK:
                continue
            records.add(_canonical(record))
    return records


def _changed_indexes(previous: Wire, current: Wire) -> list[int]:
    left = previous.body.get("messages")
    right = current.body.get("messages")
    if not isinstance(left, list) or not isinstance(right, list):
        return []
    return [index for index in range(max(len(left), len(right)))
            if _canonical(left[index] if index < len(left) else None)
            != _canonical(right[index] if index < len(right) else None)]


def _changed_band(previous: Wire, current: Wire, module: ModuleType) -> str:
    if _canonical(previous.body.get("tools")) != _canonical(current.body.get("tools")):
        return "tools"
    changed = _changed_indexes(previous, current)
    if changed:
        index = changed[0]
        right = current.body.get("messages")
        left = previous.body.get("messages")
        candidate = None
        if isinstance(right, list) and index < len(right):
            candidate = right[index]
        elif isinstance(left, list) and index < len(left):
            candidate = left[index]
        return _message_band(module, candidate) if isinstance(candidate, Mapping) else "messages"
    return "template" if previous.body == current.body else "request"


def _hard_bands_unchanged(previous: Wire, current: Wire, module: ModuleType) -> bool:
    if _canonical(previous.body.get("tools")) != _canonical(current.body.get("tools")):
        return False
    left = previous.body.get("messages")
    right = current.body.get("messages")
    if not isinstance(left, list) or not isinstance(right, list):
        return False
    for band in HARD_STABLE_BANDS:
        left_rows = [_canonical(row) for row in left
                     if isinstance(row, Mapping) and _message_band(module, row) == band]
        right_rows = [_canonical(row) for row in right
                      if isinstance(row, Mapping) and _message_band(module, row) == band]
        if left_rows != right_rows:
            return False
    return True


def _changed_retained_indexes(previous: Wire, current: Wire, module: ModuleType) -> list[int]:
    left = previous.body.get("messages")
    right = current.body.get("messages")
    if not isinstance(left, list) or not isinstance(right, list):
        return []
    retained: list[int] = []
    for index in _changed_indexes(previous, current):
        candidates = [rows[index] for rows in (left, right) if index < len(rows)]
        if any(_is_retained_source(candidate, module) for candidate in candidates):
            retained.append(index)
    return retained


def _allowed_retained_source_end(previous: Wire, current: Wire, module: ModuleType) -> int | None:
    """E may advance over retained source only when a write/edit result appears and hard bands are untouched."""
    changed = _changed_retained_indexes(previous, current, module)
    new_edits = _applied_edit_records(current.body) - _applied_edit_records(previous.body)
    if not changed or not (_has_write_edit_tool_result(previous.body) or new_edits):
        return None
    if not _hard_bands_unchanged(previous, current, module):
        return None
    messages = previous.body.get("messages")
    if not isinstance(messages, list) or any(
            isinstance(message, Mapping) and _message_band(module, message) in HARD_STABLE_BANDS
            for message in messages[changed[0]:]):
        return None
    return previous.retained_source_prefix_ends.get(changed[0])


def _is_retained_source(message: object, module: ModuleType) -> bool:
    return isinstance(message, Mapping) and _message_band(module, message) == "retained_source"


def _tool_list(tools: object) -> list[Mapping[str, object]] | None:
    """The tool entries of one request, or None when the menu is not a list of mappings (malformed)."""
    if tools is None:
        return []
    if not isinstance(tools, list) or not all(isinstance(tool, Mapping) for tool in tools):
        return None
    return list(tools)


def _tool_menu_failure(previous: Wire, current: Wire, module: ModuleType) -> str | None:
    """A tools change is allowed only as an exact-prefix append of GATED_TOOL_NAMES. Anything else names the tools.

    Returns None when the menu is unchanged or the change is an allowed gated append.
    """
    before_raw = previous.body.get("tools")
    after_raw = current.body.get("tools")
    if _canonical(before_raw) == _canonical(after_raw):
        return None
    before = _tool_list(before_raw)
    after = _tool_list(after_raw)
    if before is None or after is None:
        return "tool_menu_malformed"
    before_names = [module._tool_name(tool) for tool in before]
    after_names = [module._tool_name(tool) for tool in after]
    prefix_intact = [_canonical(tool) for tool in after[:len(before)]] == [_canonical(tool) for tool in before]
    appended = after_names[len(before):]
    if prefix_intact and all(name in module.GATED_TOOL_NAMES for name in appended):
        return None
    added = sorted(set(after_names) - set(before_names))
    removed = sorted(set(before_names) - set(after_names))
    if not added and not removed:
        reason = "reordered" if before_names != after_names and sorted(before_names) == sorted(after_names) \
            else "schema_edited"
    elif removed:
        reason = "removed"
    else:
        reason = "added_ungated" if any(name not in module.GATED_TOOL_NAMES for name in added) else "gated_not_appended"
    return (f"tool_menu_{reason}_added_{'+'.join(added) or '-'}_removed_{'+'.join(removed) or '-'}")


def check_prefix_pair(previous: Wire, current: Wire, *, im_start_token: int, layout: ModuleType) -> PairMeasurement:
    lcp = common_prefix_length(previous.tokens, current.tokens)
    end = previous.stable_prefix_end
    allowed = _allowed_retained_source_end(previous, current, layout)
    if allowed is not None:
        end = allowed
    cached = scorewire_cached_tokens(previous.tokens, lcp, im_start_token)
    failure = None
    menu = _tool_menu_failure(previous, current, layout)
    if menu is not None:
        failure = f"prefix_turn_{current.turn}_{menu}_offset_{lcp}"
    elif lcp < end:
        failure = (f"prefix_turn_{current.turn}_band_{_changed_band(previous, current, layout)}"
                   f"_offset_{lcp}")
    elif cached < end:
        failure = f"prefix_turn_{current.turn}_boundary_before_protected_extent_b_{cached}_e_{end}"
    return PairMeasurement(current.turn, lcp, end, cached, failure)


def pair_measurements(wires: Sequence[Wire], *, im_start_token: int, layout: ModuleType) -> list[PairMeasurement]:
    """Compare each sent request with the last SENT one. A refused turn is a failure and never the baseline."""
    measurements: list[PairMeasurement] = []
    baseline: Wire | None = None
    for wire in wires:
        if wire.no_sample_reason is not None:
            measurements.append(PairMeasurement(wire.turn, 0, 0, 0, f"no_sample:{wire.turn}"))
            continue
        if baseline is not None:
            measurements.append(check_prefix_pair(baseline, wire, im_start_token=im_start_token, layout=layout))
        baseline = wire
    return measurements


def analyze_wires(wires: Sequence[Wire], *, im_start_token: int, layout: ModuleType) -> WireReport:
    pairs = tuple(pair_measurements(wires, im_start_token=im_start_token, layout=layout))
    return WireReport(
        pairs=pairs,
        cached_tokens=sum(pair.cached_tokens for pair in pairs),
        no_sample=tuple(wire.turn for wire in wires if wire.no_sample_reason is not None),
        failures=tuple(pair.failure for pair in pairs if pair.failure is not None),
    )


def _strip_open_assistant(prompt: str) -> str:
    index = prompt.rfind(OPEN_ASSISTANT)
    if index < 0:
        raise HarnessError("template_missing_assistant_boundary")
    return prompt[:index]


def tokenize_wire(wire: Wire, tokenizer: OfflineTokenizer, module: ModuleType) -> Wire:
    """Fill tokens and E from the rendered prompt. Every band boundary must be a real token boundary."""
    messages = wire.body.get("messages")
    if not isinstance(messages, list) or not all(isinstance(row, Mapping) for row in messages):
        raise HarnessError("provider_request_invalid_messages")
    tools = wire.body.get("tools")
    full_prompt = tokenizer.render(messages, tools)
    full_tokens = tokenizer.tokenize(full_prompt)

    end_message = stable_message_end(wire.body, module)
    stable_prompt = _strip_open_assistant(
        tokenizer.render(messages[:end_message + 1] if end_message >= 0 else [], tools))
    if not full_prompt.startswith(stable_prompt):
        raise HarnessError("stable_band_not_prompt_prefix")
    stable_tokens = tokenizer.tokenize(stable_prompt)
    if full_tokens[:len(stable_tokens)] != stable_tokens:
        raise HarnessError("stable_band_token_boundary_mismatch")

    retained_ends: dict[int, int] = {}
    for index, message in enumerate(messages):
        if not _is_retained_source(message, module):
            continue
        prefix_prompt = _strip_open_assistant(tokenizer.render(messages[:index], tools))
        if not full_prompt.startswith(prefix_prompt):
            raise HarnessError("retained_source_prefix_not_prompt_prefix")
        prefix_tokens = tokenizer.tokenize(prefix_prompt)
        if full_tokens[:len(prefix_tokens)] != prefix_tokens:
            raise HarnessError("retained_source_token_boundary_mismatch")
        retained_ends[index] = len(prefix_tokens)

    return Wire(turn=wire.turn, raw=wire.raw, body=wire.body, tokens=full_tokens,
                stable_prefix_end=len(stable_tokens), stable_end_message=end_message,
                retained_source_prefix_ends=retained_ends)


def _no_sample_reasons(run_dir: Path, count: int) -> list[str | None]:
    """Per send, from the engine's wire siblings: an error record or a missing response makes it no sample."""
    wire_dir = run_dir / "wire"
    paths = sorted(wire_dir.glob("[0-9][0-9][0-9][0-9].request.json")) if wire_dir.is_dir() else []
    if len(paths) != count:
        raise HarnessError(f"wire_request_count_mismatch: {len(paths)} request files for {count} requests")
    reasons: list[str | None] = []
    for path in paths:
        stem = path.name[:4]
        if (wire_dir / f"{stem}.error.txt").is_file():
            reasons.append("send_error")
        elif not (wire_dir / f"{stem}.response.json").is_file():
            reasons.append("no_response")
        else:
            reasons.append(None)
    return reasons


def analyze(trace: Trace, tokenizer: OfflineTokenizer | None = None) -> WireReport:
    """Tokenize each sent request through the relay and report stable-prefix pairs, cached tokens and failures."""
    client = tokenizer if tokenizer is not None else OfflineTokenizer()
    reasons = _no_sample_reasons(trace.run_dir, len(trace.requests))
    wires = [Wire(turn=index + 1, raw=raw, body=body, no_sample_reason=reason)
             for index, (raw, body, reason) in enumerate(zip(trace.requests, trace.request_json, reasons))]
    # No-sample sends are never tokenized: they cannot be reuse, and they never become the baseline.
    tokenized = [wire if wire.no_sample_reason is not None else tokenize_wire(wire, client, layout)
                 for wire in wires]
    return analyze_wires(tokenized, im_start_token=client.marker(), layout=layout)


def assert_prefix_stable(trace: Trace, tokenizer: OfflineTokenizer | None = None) -> None:
    report = analyze(trace, tokenizer)
    if report.failures:
        raise AssertionError("cache prefix not stable: " + "; ".join(report.failures))


def _first_difference(left: bytes, right: bytes) -> int:
    for index, (first, second) in enumerate(zip(left, right)):
        if first != second:
            return index
    return min(len(left), len(right))


def _assert_same_requests(label: str, left: Trace, right: Trace) -> None:
    if len(left.requests) != len(right.requests):
        raise AssertionError(f"{label}: request count differs ({len(left.requests)} vs {len(right.requests)})")
    for index, (first, second) in enumerate(zip(left.requests, right.requests)):
        if first != second:
            raise AssertionError(f"{label}: turn {index + 1} request bytes differ, "
                                 f"first difference at byte offset {_first_difference(first, second)}")


def assert_off_identical(off: Trace, base: Trace) -> None:
    _assert_same_requests("switch OFF wire vs trusted base", off, base)


def assert_deterministic(a: Trace, b: Trace) -> None:
    _assert_same_requests("repeated run", a, b)


def assert_no_cache_drop(off_report: WireReport, on_report: WireReport, tolerance: float = 0.01) -> None:
    """Fail when the switch-ON run cached more than `tolerance` (fraction) fewer tokens than the OFF run."""
    if off_report.cached_tokens <= 0:
        raise AssertionError("cache drop check is vacuous: the OFF run cached 0 tokens")
    drop = off_report.cached_tokens - on_report.cached_tokens
    if drop > off_report.cached_tokens * tolerance:
        raise AssertionError(f"cached tokens dropped: off {off_report.cached_tokens} vs on "
                             f"{on_report.cached_tokens} (tolerance {tolerance:.0%})")


__all__ = [
    "OfflineTokenizer", "PairMeasurement", "Wire", "WireReport", "analyze", "analyze_wires",
    "assert_deterministic", "assert_no_cache_drop", "assert_off_identical", "assert_prefix_stable",
    "check_prefix_pair", "common_prefix_length", "pair_measurements", "scorewire_cached_tokens",
    "stable_message_end", "tokenize_wire",
]

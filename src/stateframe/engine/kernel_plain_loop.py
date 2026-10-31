"""Plain native-tool loop: a faithful port of bin/baseline_harness.py's agent loop.

Selected only by STATEFRAME_AGENT_LOOP=plain. Same system prompt, the same seven
tools and schemas, append-only history, every call of a response executed in
order, a 12,000-character result cap, and no governors, gates, notices or menu
changes. Stops only on done, wall clock, request cap, or a transport/context end.

Deliberate differences from the baseline (all host-side, none model-visible
unless a hook is switched on):
- provider, executor, journal and clocks are injected so the loop runs offline;
- each provider call runs on a watchdog thread bounded by the request timeout,
  so a stalled socket cannot outlive the wall budget;
- a timed-out or malformed response may be retried within the same bounded
  transport budget, always BEFORE any tool of that response executes;
- the exact serialized body is hashed and journaled before it is sent.

This module must not import the StateFrame runtime (KernelBolt, StateFrameRuntime,
completion owner, F4, react_repair).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import re
import threading
import time

from kernel_plain_contracts import PlainContractError, validate_pairing
from kernel_shell_tool import SHELL_SCHEMA
from kernel_tool_reply import Outcome, record_and_strip_receipts, tool_reply

OUT_CAP = 12000  # chars of any single tool result returned to the model
JOIN_GRACE_S = 15.0  # watchdog grace past the socket timeout before a call is abandoned

SYSTEM = """You are a software engineer fixing a bug in a repository.
The repository is at /workspace. Use the tools to inspect the code, make the fix, and run tests.
Make the smallest durable production-quality change that repairs the described behavior.
When you are finished, call done with a short summary."""

CONTINUE = "Continue. Use the tools; call done when finished."

TOOLS = [
    {"type": "function", "function": {"name": "list_dir", "description": "List a directory (relative to /workspace).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file with line numbers. Optional 1-based start/end lines.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Recursive regex search (grep -rnE) under a path.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "str_replace", "description": "Replace exactly one occurrence of old with new in a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run python -m pytest with the given arguments (cwd /workspace, 300s timeout).",
     "parameters": {"type": "object", "properties": {"args": {"type": "string"}}, "required": ["args"]}}},
    {"type": "function", "function": {"name": "done", "description": "Finish the task.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]


TOOLS.append(SHELL_SCHEMA)

def cap(text: str) -> str:
    text = text.replace("\x00", "").replace("\\u0000", "<NUL>")  # server re-parses tool text as JSON
    return text if len(text) <= OUT_CAP else text[:OUT_CAP] + f"\n...[truncated {len(text) - OUT_CAP} chars]"


_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def scrub(obj):
    """Make every string llama-server-parseable.

    NUL however encoded: llama-server re-parses tool args/content and rejects any decoded U+0000.
    Lone surrogates: files read with errors="surrogateescape" carry U+DC80..U+DCFF; json.dumps
    escapes them as "\\udcXX" and llama-server's parser rejects the whole request with
    500 parse_error.101 "surrogate U+DC00..U+DFFF must follow U+D800..U+DBFF" (10-13: benchmark plain
    1056, every request after a binary-ish file entered the transcript). A Python str never holds
    a valid surrogate PAIR, so every surrogate code unit here is unpaired: map it to U+FFFD.
    """
    if isinstance(obj, str):
        return _LONE_SURROGATE.sub("�", obj.replace("\x00", "")).replace(
            "\\u0000", "<NUL>").replace("\\x00", "<NUL>")
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, dict):
        raw_args = obj.get("arguments") if "name" in obj else None
        # Keys too: a model-authored tool-argument KEY carrying U+0000 survives json.loads ->
        # json.dumps and llama-server's nested argument parse rejects it (10-13 benchmark plain 1076:
        # 500 parse_error.101 "control character U+0000 ... last read: '\"tool_call_<NUL>'").
        out = {(scrub(k) if isinstance(k, str) else k): scrub(v) for k, v in obj.items()
               if k != "arguments" or raw_args is None}
        if isinstance(raw_args, str):  # tool-call args: parse FIRST, scrub values, re-encode (never string-edit JSON)
            try:
                out["arguments"] = json.dumps(scrub(json.loads(raw_args.replace("\x00", ""))))
            except ValueError:
                out["arguments"] = scrub(raw_args)
        return out
    return obj


class ProviderHTTPError(Exception):
    def __init__(self, code: int, body: str):
        super().__init__(f"HTTP {code}")
        self.code, self.body = int(code), str(body)[:500]


class ProviderTransportError(Exception):
    def __init__(self, detail: str, *, timed_out: bool):
        super().__init__(detail)
        self.timed_out = bool(timed_out)


@dataclass(frozen=True)
class Limits:
    model: str
    wall_s: float = 1800.0
    max_requests: int = 150
    request_timeout_s: float = 600.0
    temperature: float = 1.0
    max_tokens: int = 24000
    reasoning_budget: int = 0
    sampler: Mapping[str, object] = field(default_factory=dict)
    # Retry budget is per CONSECUTIVE failure streak (reset by any good response): a relay that
    # drops three connections in a row mid-run must not end a run that is otherwise healthy.
    # ``max_total_transport_retries`` bounds a provider that never recovers; the wall clock bounds both.
    transport_retries: int = 5
    max_total_transport_retries: int = 20
    retry_sleep_s: float = 10.0
    retry_max_sleep_s: float = 60.0
    retry_timeouts: bool = True

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("plain loop requires a model id")
        for name in ("wall_s", "request_timeout_s"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and value > 0 and value != float("inf")):
                raise ValueError(f"{name} must be a positive finite number")
        if self.max_requests < 1 or self.max_tokens < 1:
            raise ValueError("max_requests and max_tokens must be positive")


@dataclass
class LoopResult:
    end: str = "unfinished"
    requests: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    transport_retries: int = 0
    consecutive_transport_failures: int = 0
    wall_s: float = 0.0
    summary: str = ""
    messages: list = field(default_factory=list)


Provider = Callable[[dict, float], Mapping[str, object]]
Executor = Callable[[str, dict], str]


def build_body(messages: list, limits: Limits) -> dict:
    from kernel_shell_tool import service
    shell = service()
    tools = [t for t in TOOLS if t["function"]["name"] != "shell"
             or (shell is not None and shell.settings.shell != "off")]
    import copy
    tools = copy.deepcopy(tools)
    for tool in tools:
        if tool["function"]["name"] == "shell":
            tool["function"]["parameters"]["properties"].pop("background", None)
    body = {"model": limits.model, "messages": messages, "tools": tools, "temperature": limits.temperature,
            "max_tokens": limits.max_tokens, **dict(limits.sampler)}
    if limits.reasoning_budget:
        body["reasoning_budget_tokens"] = limits.reasoning_budget
    return body


def initial_messages(task_text: str) -> list:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task_text}]


def bounded_call(provider: Provider, body: dict, timeout: float) -> Mapping[str, object]:
    """Run one provider call; never wait past ``timeout`` plus a small join grace."""
    box: dict[str, object] = {}

    def target() -> None:
        try:
            box["value"] = provider(body, timeout)
        except BaseException as error:  # delivered to the caller below
            box["error"] = error

    worker = threading.Thread(target=target, name="plain-provider", daemon=True)
    worker.start()
    worker.join(timeout + JOIN_GRACE_S)
    if worker.is_alive():
        abort = getattr(provider, "abort", None)
        if callable(abort):
            try:
                abort()
            except Exception:
                pass
            worker.join(min(0.5, JOIN_GRACE_S))
        raise ProviderTransportError(f"provider call exceeded {timeout:.0f}s (abandoned)", timed_out=True)
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]  # type: ignore[return-value]


def _response_message(response: object) -> tuple[dict, str | None]:
    if not isinstance(response, Mapping):
        raise ValueError("response is not a JSON object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ValueError("response choice has no message")
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list) or any(
            not isinstance(c, Mapping) or not isinstance(c.get("function"), Mapping)
            or not isinstance(c["function"].get("name"), str) for c in calls):
        raise ValueError("response tool_calls are malformed")
    return dict(message), choices[0].get("finish_reason")


def run_loop(task_text: str, *, provider: Provider, execute: Executor, limits: Limits, journal,
             prepare: Callable[[list], list] | None = None,
             before_request: Callable[[dict], None] | None = None,
             clock: Callable[[], float] = time.monotonic,
             sleep: Callable[[float], None] = time.sleep) -> LoopResult:
    """Drive one episode. ``msgs`` is append-only; ``prepare`` may only shape the wire view."""
    msgs = initial_messages(task_text)
    result = LoopResult(messages=msgs)
    t0 = clock()
    journal.write("loop_started", system_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),
                  tools=[t["function"]["name"] for t in TOOLS], limits=limits.__dict__)
    try:
        while True:
            if clock() - t0 > limits.wall_s:
                result.end = "time_limit"
                break
            if result.requests >= limits.max_requests:
                result.end = "request_cap"
                break
            wire_messages = prepare(list(msgs)) if prepare is not None else msgs
            validate_pairing(wire_messages)
            body = scrub(build_body(wire_messages, limits))
            encoded = json.dumps(body).encode()
            if before_request is not None:
                before_request(body)
            remaining = int(limits.wall_s - (clock() - t0)) + 5
            timeout = min(limits.request_timeout_s, max(remaining, 30))
            journal.write("request_submitted", request=result.requests + 1, body_sha256="sha256:"
                          + hashlib.sha256(encoded).hexdigest(), body_bytes=len(encoded),
                          messages=len(wire_messages), timeout_s=timeout,
                          new_messages=msgs[-_tail_since_last(msgs):])
            started = clock()
            try:
                response = bounded_call(provider, body, timeout)
                message, finish = _response_message(response)
            except ProviderHTTPError as error:
                journal.write("transport_error", http_error=error.code, body=error.body)
                # A relay-detected truncated upload is a transport fault on the way out: the
                # same bytes resent intact are a fresh, valid request.
                transient = error.code in (502, 503, 504) or "short_request_body" in error.body
                if transient and _may_retry(result, limits, clock() - t0, journal, "http"):
                    sleep(_retry_delay(result, limits))
                    continue
                lowered = error.body.lower()
                if "context" in lowered:
                    result.end = "context_overflow"
                elif "parse_error" in lowered:
                    # The server could not parse the bytes it received. Resending the same body is
                    # deterministic, so end with a label that names the cause instead of http_500.
                    result.end = "request_body_rejected:parse_error"
                else:
                    result.end = f"http_{error.code}"
                break
            except ProviderTransportError as error:
                journal.write("transport_error", transport_error=str(error)[:200], timed_out=error.timed_out)
                if (not error.timed_out or limits.retry_timeouts) and _may_retry(
                        result, limits, clock() - t0, journal, "timeout" if error.timed_out else "transport"):
                    sleep(_retry_delay(result, limits))
                    continue
                result.end = "request_error:" + ("TimeoutError" if error.timed_out else "TransportError")
                break
            except ValueError as error:  # malformed response: nothing executed yet, safe to retry
                journal.write("transport_error", malformed_response=str(error)[:200])
                if _may_retry(result, limits, clock() - t0, journal, "malformed"):
                    sleep(_retry_delay(result, limits))
                    continue
                result.end = "malformed_response"
                break
            result.requests += 1
            result.consecutive_transport_failures = 0
            usage = response.get("usage") or {}
            usage = usage if isinstance(usage, Mapping) else {}
            result.prompt_tokens += _count(usage.get("prompt_tokens"))
            result.completion_tokens += _count(usage.get("completion_tokens"))
            calls = [dict(c, function=dict(c["function"])) for c in message.get("tool_calls") or []]
            for call in calls:  # llama-server rejects raw NUL when re-rendering history
                call["function"]["arguments"] = (call["function"].get("arguments") or "").replace("\x00", "")
            journal.write("response", request=result.requests, seconds=round(clock() - started, 1),
                          usage=dict(usage), finish=finish, content=message.get("content") or "",
                          reasoning_content=message.get("reasoning_content") or "", calls=calls)
            msgs.append({"role": "assistant", "content": (message.get("content") or "").replace("\x00", ""),
                         **({"tool_calls": calls} if calls else {})})
            if not calls:
                msgs.append({"role": "user", "content": CONTINUE})
                continue
            finished = False
            for call in calls:
                result.tool_calls += 1
                name = call["function"]["name"]
                reply_outcome = Outcome.ACCEPTED
                reply_receipt = None
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                    if name == "done":
                        output = "finished"
                        result.summary = str(args.get("summary", "")) if isinstance(args, dict) else ""
                    else:
                        bind_call = getattr(getattr(execute, "__self__", None), "bind_tool_call", None)
                        if bind_call is not None:
                            bind_call(call.get("id", ""))
                        output = execute(name, args)
                except PlainContractError:  # a broken host boundary is never a model observation
                    raise
                except Exception as error:  # tool errors go back to the model, as in any plain harness
                    output = f"error: {type(error).__name__}: {error}"
                    reply_outcome = Outcome.ERROR
                    reply_receipt = {"code": "tool_error"}
                if name == "done":
                    finished = True
                content = output if name == "shell" else cap(output)
                reply = tool_reply(
                    call.get("id", ""), name, reply_outcome, content,
                    receipt=reply_receipt,
                )
                record_and_strip_receipts(
                    [reply], journal, request_id=result.requests, step=result.tool_calls,
                )
                msgs.append(reply)
                journal.write("completed_action", request=result.requests, action={"action_kind": name},
                              call_id=call.get("id", ""), output=content, raw_chars=len(output))
            if finished:
                result.end = "done"
                break
    finally:
        result.wall_s = round(clock() - t0, 1)
        journal.write("loop_ended", end=result.end, requests=result.requests, tool_calls=result.tool_calls,
                      prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                      transport_retries=result.transport_retries, wall_s=result.wall_s)
    return result


def _may_retry(result: LoopResult, limits: Limits, elapsed: float, journal, reason: str) -> bool:
    allowed = (result.consecutive_transport_failures < limits.transport_retries
               and result.transport_retries < limits.max_total_transport_retries
               and elapsed < limits.wall_s - 60)
    if allowed:
        result.transport_retries += 1
        result.consecutive_transport_failures += 1
    journal.write("retry_decision", reason=reason, retry_scheduled=allowed, retries=result.transport_retries,
                  consecutive=result.consecutive_transport_failures)
    return allowed


def _retry_delay(result: LoopResult, limits: Limits) -> float:
    """Exponential backoff over the current failure streak, capped at ``retry_max_sleep_s``."""
    streak = max(1, result.consecutive_transport_failures)
    return min(limits.retry_max_sleep_s, limits.retry_sleep_s * (2 ** (streak - 1)))


def _count(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _tail_since_last(msgs: list) -> int:
    """Number of trailing messages appended since the previous request (journal delta)."""
    for back in range(1, len(msgs) + 1):
        if msgs[-back].get("role") == "assistant":
            return back
    return len(msgs)


__all__ = ["CONTINUE", "Limits", "LoopResult", "OUT_CAP", "ProviderHTTPError", "ProviderTransportError",
           "SYSTEM", "TOOLS", "bounded_call", "build_body", "cap", "initial_messages", "run_loop", "scrub"]

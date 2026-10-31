"""Allocate opaque execution observations after the authoritative request frame.

The tool tape retains the original observation. Only its delivery is shortened;
tool identity, status and effect summaries remain in the newest exchange.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
import re


def record_delivery(run_dir: Path, head: str, output: str, tail: str) -> str:
    """Persist engine-owned boundaries separately from untrusted process text."""
    content = head + output + tail
    directory = run_dir / "execution-delivery"
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / (hashlib.sha256(content.encode()).hexdigest() + ".json")
    payload = json.dumps([head, output, tail], ensure_ascii=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError("execution delivery identity collision")
    return content


def _parts(content: str, name: str, run_dir: Path | None = None) -> tuple[str, bytes, str]:
    if run_dir is not None:
        path = run_dir / "execution-delivery" / (hashlib.sha256(content.encode()).hexdigest() + ".json")
        if path.is_file():
            parts = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(parts, list) or len(parts) != 3
                    or not all(isinstance(item, str) for item in parts) or "".join(parts) != content):
                raise ValueError("invalid execution delivery metadata")
            return parts[0], parts[1].encode(), parts[2]
    # Legacy/taped replies without structured metadata get no protected suffix.
    # Preserve only status lines, never infer authority from process text.
    lines = content.splitlines(keepends=True)
    header_lines = 2 if name == "shell" and content.startswith("$ ") else 1
    if content.startswith("REFUSED TIMEOUT:"):
        header_lines = 3 if name == "shell" else 2
    elif content.startswith("REFUSED "):
        return content, b"", ""
    return "".join(lines[:header_lines]), "".join(lines[header_lines:]).encode(), ""


def fit_execution_replies(
        body: dict[str, object], *, max_request_bytes: int,
        measure_bytes: Callable[[], int], max_request_tokens: int | None = None,
        measure_tokens: Callable[[], int] | None = None,
) -> tuple[str, ...]:
    """Fit all execution replies together using whole-request measurements.

    No estimate of tokens per output byte is used. If the irreducible request
    still cannot fit, leave its explicit minimal observations for the normal
    fail-closed capacity check. Never remove an exchange or invent check credit.
    """
    if (max_request_tokens is None) != (measure_tokens is None):
        raise ValueError("execution reply token ceiling and meter must be supplied together")
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ()
    from kernel_agent_tools import service
    host = service()
    names = {"shell", "job_wait", "job_output", *(host.custom if host is not None else ())}
    calls = {}
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            for call in message.get("tool_calls", ()):
                if isinstance(call, Mapping) and isinstance(call.get("function"), Mapping):
                    calls[call.get("id")] = call["function"].get("name")

    def fits() -> bool:
        return (measure_bytes() <= max_request_bytes and
                (measure_tokens is None or measure_tokens() <= max_request_tokens))

    if fits():
        return ()
    rows = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        name = calls.get(call_id)
        content = message.get("content")
        if name not in names or not isinstance(content, str):
            continue
        if "[execution output delivery:" in content:
            # A rebuilt candidate can gain mandatory source notices. Refit
            # from durable authority, never truncate an already shortened
            # excerpt and mislabel its digest as the original observation.
            if host is None:
                continue
            from kernel_tool_tape import tape_for_run
            record = tape_for_run().lookup(call_id)
            if record is None or not record.completed or record.tool != name:
                continue
            content = record.observation
        head, output, tail = _parts(content, name, host.shell.run_dir if host is not None else None)
        if len(output) > 512:
            rows.append((message, call_id, content, head, output, tail))
    if not rows:
        return ()

    def render(keep: int) -> None:
        for message, call_id, original, head, output, tail in rows:
            if len(output) <= keep:
                message["content"] = original
                continue
            left = output[:keep // 2].decode("utf-8", "ignore")
            right = output[-(keep // 2):].decode("utf-8", "ignore") if keep // 2 else ""
            omitted = len(output) - len(left.encode()) - len(right.encode())
            prior_omitted = sum(int(value) for value in re.findall(
                r"\[\.\.\. (\d+) bytes omitted \.\.\.\]", original))
            marker = (f"\n[execution output delivery: {omitted} observation bytes omitted; "
                      f"process output bytes already omitted: {prior_omitted}; "
                      f"full observation in tool-tape.jsonl call {call_id}; "
                      f"sha256 {hashlib.sha256(original.encode()).hexdigest()}]\n")
            message["content"] = head + left + marker + right + "\n" + tail

    # Halving bounds tokenizer work even for several flood replies. Retain the
    # largest tested allowance that fits; tokenization need not be monotonic.
    allowance = max(len(row[4]) for row in rows)
    while allowance:
        allowance //= 2
        render(allowance)
        if fits():
            break
    return tuple(str(row[1]) for row in rows if row[0]["content"] != row[2])


def project_execution_delivery(messages, delivered):
    """Carry fitted replies through a detached frame reconstruction."""
    import copy
    replacements = {m["tool_call_id"]: m["content"] for m in delivered
                    if isinstance(m, Mapping) and m.get("role") == "tool"
                    and "[execution output delivery:" in str(m.get("content", ""))}
    result = copy.deepcopy(messages)
    for message in result:
        if message.get("role") == "tool" and message.get("tool_call_id") in replacements:
            message["content"] = replacements[message["tool_call_id"]]
    return result

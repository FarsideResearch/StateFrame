"""Opt-in study capacity gate on exact captured provider bodies."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request


class CapacityRefusal(ValueError):
    """The request cannot reserve its declared output budget."""

    def __init__(self, message: str, *, terminal: bool = False,
                 prompt_tokens: int | None = None, context_limit: int | None = None,
                 capacity: dict | None = None, request_id: str | None = None,
                 reroutable: bool = False):
        super().__init__(message)
        self.long_horizon_terminal = terminal
        # A non-terminal exact refusal is a handoff to the runtime compaction
        # reroute.  Keep the exact receipt on the exception so the ask-site can
        # journal the reroute without re-counting or re-rendering the request.
        self.long_horizon_reroutable = reroutable
        self.prompt_tokens = prompt_tokens
        self.context_limit = context_limit
        self.capacity = dict(capacity or {})
        self.request_id = request_id


def count_request(body, endpoint, *, context_limit, post):
    if type(context_limit) is not int or context_limit < 1:
        raise ValueError("positive context limit required")
    budget = body.get("max_tokens")
    if type(budget) is not int or budget < 1:
        raise ValueError("explicit positive output budget required")
    payload = dict(body, add_generation_prompt=True)
    rendered = post("/apply-template", payload)
    if not isinstance(rendered, dict) or not isinstance(rendered.get("prompt"), str):
        raise TypeError("invalid template response")
    tokenized = post("/tokenize", {
        "content": rendered["prompt"], "add_special": False, "parse_special": True})
    tokens = tokenized.get("tokens") if isinstance(tokenized, dict) else None
    if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("invalid tokenizer response")
    if not tokens:
        raise ValueError("empty tokenized request")
    return {
        "capacity_schema": "study-capacity/1",
        "payload_sha256": hashlib.sha256(json.dumps(body).encode()).hexdigest(),
        "prompt_tokens": len(tokens), "output_tokens_reserved": budget,
        "context_limit": context_limit,
        "fits": len(tokens) + budget <= context_limit,
        "template_sha256": hashlib.sha256(rendered["prompt"].encode()).hexdigest(),
    }


def install(spec, capture_class):
    config = spec.get("capacity_preflight")
    if config is None:
        return
    if not isinstance(config, dict) or set(config) != {"context_limit"}:
        raise ValueError("invalid capacity preflight configuration")
    limit = config["context_limit"]
    if type(limit) is not int or limit < 1:
        raise ValueError("positive context limit required")
    original = capture_class.send
    if getattr(original, "_study_capacity", False):
        raise ValueError("capacity preflight already installed")

    def send(capture, body, endpoint, key, *, deadline=None):
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path != "/v1/chat/completions":
            raise ValueError("capacity preflight requires the local study server")
        base = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        def post(route, payload):
            from kernel_bounded_http import request_bytes
            expires = min(deadline, time.monotonic() + 15) if deadline is not None else time.monotonic() + 15
            return json.loads(request_bytes(json.dumps(payload).encode(), base + route, key,
                                           deadline=expires, socket_timeout=10))
        timeout = min(10, deadline - time.monotonic()) if deadline is not None else 10
        if timeout <= 0:
            raise TimeoutError("capacity preflight deadline expired")
        with urllib.request.urlopen(base + "/slots", timeout=timeout) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("oversized live slot metadata")
        slots = json.loads(raw)
        slot_id = body.get("id_slot")
        if type(slot_id) is not int or not isinstance(slots, list):
            raise ValueError("explicit live slot required")
        selected = [item for item in slots if isinstance(item, dict) and item.get("id") == slot_id]
        if len(selected) != 1 or type(selected[0].get("n_ctx")) is not int or selected[0]["n_ctx"] != limit:
            raise ValueError("live slot differs from declared capacity")
        from long_horizon_decode_logging import enabled as decode_depth_enabled, slots_busy
        decode_depth_per_arm_on = os.environ.get(
            "LONG_HORIZON_DECODE_DEPTH_PER_ARM", "off"
        ).strip().lower() in {"on", "1", "true", "yes"}
        if decode_depth_enabled():
            capture.decode_depth_slots_busy = slots_busy(slots)
            capture.decode_depth_slots_busy_source = "live_slots"
        if decode_depth_per_arm_on:
            from decode_depth_per_arm import slots_busy as per_arm_slots_busy
            busy = per_arm_slots_busy(slots)
            if busy is not None:
                capture.decode_depth_per_arm_slots_busy = busy
                capture.decode_depth_per_arm_slots_busy_source = "live_slots"
        receipt = count_request(body, endpoint, context_limit=limit, post=post)
        if decode_depth_enabled():
            capture.decode_depth_rendered_context_depth = receipt["prompt_tokens"]
            capture.decode_depth_rendered_context_depth_source = "capacity_preflight_tokenize"
        if decode_depth_per_arm_on:
            capture.decode_depth_per_arm_rendered_context_depth = receipt["prompt_tokens"]
            capture.decode_depth_per_arm_rendered_context_depth_source = (
                "capacity_preflight_tokenize"
            )
        capture.journal.emit("capacity_preflight", request_id=capture.request_id, **receipt)
        if not receipt["fits"]:
            no_compaction = (
                os.environ.get("LONG_HORIZON_NO_COMPACTION", "off").strip().lower()
                in {"on", "1", "true", "yes"})
            protocol = (
                os.environ.get("LONG_HORIZON_PROTOCOL", "off").strip().lower()
                in {"on", "1", "true", "yes"})
            # Protocol-on compaction arms get one runtime reroute before the
            # historical terminal event.  The explicit uncompacted control keeps
            # the old terminal behavior exactly.
            terminal_overflow = no_compaction or (protocol and not no_compaction)
            reroutable = protocol and not no_compaction
            if reroutable:
                terminal_overflow = False
            if os.environ.get("LONG_HORIZON_TELEMETRY", "off").strip().lower() in {
                    "on", "1", "true", "yes"}:
                capture.long_horizon_rendered_prompt_tokens = receipt["prompt_tokens"]
                capture.long_horizon_rendered_prompt_tokens_source = (
                    "rendered_prompt_tokens"
                )
                capture.long_horizon_context_limit_tokens = limit
            if terminal_overflow:
                capture.journal.emit(
                    "terminal_result",
                    request_id=capture.request_id,
                    terminal_result="CONTEXT_OVERFLOW",
                    failure_class="context_overflow",
                    assigned=True,
                    capacity=receipt,
                )
            raise CapacityRefusal(
                "request exceeds context with full output reservation",
                terminal=terminal_overflow,
                prompt_tokens=receipt["prompt_tokens"],
                context_limit=receipt["context_limit"],
                capacity=receipt,
                request_id=capture.request_id,
                reroutable=reroutable,
            )
        if os.environ.get("LONG_HORIZON_TELEMETRY", "off").strip().lower() in {
                "on", "1", "true", "yes"}:
            capture.long_horizon_rendered_prompt_tokens = receipt["prompt_tokens"]
            capture.long_horizon_rendered_prompt_tokens_source = "rendered_prompt_tokens"
            capture.long_horizon_context_limit_tokens = limit
        return original(capture, body, endpoint, key, deadline=deadline)

    send._study_capacity = True
    capture_class.send = send

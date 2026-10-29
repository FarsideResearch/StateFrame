"""Bounded typed host operators for the opt-in figure_it_out workbench.

The workbench is deliberately an evidence rail, not a second completion
authority.  Operators accept strict JSON-shaped data, return a bounded host
receipt, and never turn a model claim into a verified repair.  The runtime and
runner are the only callers that expose or dispatch these operators.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable, Mapping, Sequence


MODE_ENV = "STATEFRAME_FIGURE_IT_OUT"
SCHEMA = "figure-it-out-receipt/1"
RESULT_SCHEMA = "figure-it-out-result/1"
OPERATOR_NAMES = ("BIND_QUANTITY", "CONTRACT_CASES", "TRACE_BEHAVIOR")
OPERATOR_ALIASES = {"CONTRACT_CHECK": "CONTRACT_CASES"}
MAX_ITEMS = 8
MAX_TEXT = 512
MAX_DEPTH = 6
MAX_PACKET_BYTES = 16_384


class FigureItOutError(ValueError):
    """A typed operator packet is outside the bounded JSON contract."""


def mode(environment: Mapping[str, object] | None = None) -> str:
    """Return the strictly validated workbench switch."""
    values = os.environ if environment is None else environment
    value = values.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return str(value)


def enabled(environment: Mapping[str, object] | None = None) -> bool:
    return mode(environment) == "on"


def _canonical_operator(name: object) -> str | None:
    if not isinstance(name, str):
        return None
    candidate = name.strip().upper()
    if candidate in OPERATOR_NAMES:
        return candidate
    return OPERATOR_ALIASES.get(candidate)


def is_operator(name: object) -> bool:
    return _canonical_operator(name) is not None


def _bounded(value: object, *, depth: int = 0) -> object:
    """Copy a strict, finite JSON value with small protocol bounds."""
    if depth > MAX_DEPTH:
        raise FigureItOutError("operator packet is too deeply nested")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FigureItOutError("operator packet contains a non-finite number")
        return value
    if isinstance(value, str):
        if len(value) > MAX_TEXT:
            raise FigureItOutError("operator text field exceeds the bounded limit")
        return value
    if isinstance(value, Mapping):
        if len(value) > 32:
            raise FigureItOutError("operator object has too many fields")
        result: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str) or len(key) > 96:
                raise FigureItOutError("operator object keys must be short strings")
            result[key] = _bounded(child, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_ITEMS:
            raise FigureItOutError(f"operator arrays are limited to {MAX_ITEMS} items")
        return [_bounded(child, depth=depth + 1) for child in value]
    raise FigureItOutError("operator packets must contain JSON values only")


def _safe_bounded(value: object, *, depth: int = 0) -> object:
    """Like ``_bounded`` but NEVER raises: it clamps oversized/malformed data
    instead of rejecting it.  ``_finish`` uses this (rather than ``_bounded``)
    for the failure path, where the input being recorded is often the exact
    already-rejected packet that failed ``_bounded`` once already -- raising
    again there escapes the caller's try/except and crashes dispatch instead
    of returning a typed failure receipt (fix round, rv_figure_it_out_REPORT.md
    finding 2)."""
    if depth > MAX_DEPTH:
        return "...[truncated: exceeds bounded nesting depth]"
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:MAX_TEXT]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in list(value.items())[:32]:
            result[str(key)[:96]] = _safe_bounded(child, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_bounded(child, depth=depth + 1) for child in list(value)[:MAX_ITEMS]]
    return str(value)[:MAX_TEXT]


def _json(value: object) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise FigureItOutError("operator result is not strict JSON") from error
    if len(rendered.encode("utf-8")) > MAX_PACKET_BYTES:
        raise FigureItOutError("operator packet exceeds the bounded byte limit")
    return rendered


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _safe_digest(value: object) -> str:
    """Digest for the receipt path: never raises on an oversized aggregate.

    Under the byte limit this is exactly ``_digest``.  Above it (every field
    clamped, but the aggregate > MAX_PACKET_BYTES) hash the same canonical
    JSON without the size cap; ``_finish`` then falls back to its minimal
    bounded REJECTED body."""
    try:
        return _digest(value)
    except FigureItOutError:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False, default=str)
        return "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _operator_schema(name: str, description: str, properties: Mapping[str, object],
                     required: Sequence[str] = ()) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(properties),
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


def _schemas() -> tuple[dict[str, object], ...]:
    bounded_string = {"type": "string", "maxLength": MAX_TEXT}
    return (
        _operator_schema(
            "BIND_QUANTITY",
            "Bind quantity identities and check dimensions using supplied data. "
            "The result is evidence only; it does not certify a repair or completion.",
            {
                "expression": bounded_string,
                "bindings": {"type": "object"},
                "requested_units": {"type": ["string", "object"]},
                "requested_outputs": {"type": ["array", "object"], "maxItems": MAX_ITEMS},
                "witness": {"type": "object"},
                "why": bounded_string,
            },
            ("expression", "bindings", "requested_units"),
        ),
        _operator_schema(
            "CONTRACT_CASES",
            "Construct a small set of public contract cases and compare supplied observations. "
            "Expected behavior alone remains unverified evidence.",
            {
                "cases": {
                    "type": "array", "minItems": 1, "maxItems": MAX_ITEMS,
                    "items": {"type": "object"},
                },
                "why": bounded_string,
            },
            ("cases",),
        ),
        _operator_schema(
            "TRACE_BEHAVIOR",
            "Record bounded host observations for a candidate behavior. "
            "Observations are evidence only and never a completion certificate.",
            {
                "snapshot_id": bounded_string,
                "entry_point": bounded_string,
                "observations": {
                    "type": "array", "maxItems": MAX_ITEMS,
                    "items": {"type": "object"},
                },
                "events": {
                    "type": "array", "maxItems": MAX_ITEMS,
                    "items": {"type": "object"},
                },
                "why": bounded_string,
            },
        ),
    )


_SCHEMAS = _schemas()


def tool_surface(tools: Sequence[Mapping[str, object]], *, runtime: object = None):
    """Add the three native host tools only when the feature is explicitly on."""
    del runtime  # Reserved for a later applicability policy; no implicit policy here.
    if not enabled():
        return tools
    existing = {
        tool.get("function", {}).get("name")
        for tool in tools
        if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
    }
    return [*tools, *[copy.deepcopy(tool) for tool in _SCHEMAS
                     if tool["function"]["name"] not in existing]]


operator_tool_surface = tool_surface


def tool_names() -> tuple[str, ...]:
    return OPERATOR_NAMES


def _context_fields(context: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(context, Mapping):
        return {}
    result = {}
    for key in ("task_id", "snapshot_id", "path", "source_revision"):
        value = context.get(key)
        if value is not None:
            result[key] = str(value)[:MAX_TEXT]
    return result


def _error_result(name: str, args: object, error: object, *, context=None,
                  journal=None):
    return _finish(
        name, "REJECTED", "HOST_ERROR",
        {"errors": [str(error)[:MAX_TEXT]]}, context=context,
        input_payload=args if isinstance(args, Mapping) else {}, journal=journal,
    )


def _finish(name: str, status: str, authority: str, result: Mapping[str, object], *,
            context: Mapping[str, object] | None, input_payload: Mapping[str, object],
            journal: object = None, reason: str | None = None) -> dict[str, object]:
    canonical = _canonical_operator(name) or str(name).strip().upper()[:MAX_TEXT]
    # _safe_bounded (never raises) -- see its docstring: this is the receipt
    # construction path reached even for already-rejected, oversized/malformed
    # input, so it must never itself raise.
    bounded_result = _safe_bounded(dict(result))
    bounded_input = _safe_bounded(dict(input_payload))
    from kernel_wire_identity import next_wire_id
    public_id = (next_wire_id(journal, "figure-it-out", canonical,
                              _safe_digest(bounded_input), _safe_digest(bounded_result))
                 if journal is not None else uuid.uuid4().hex)
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "receipt_id": "figure-it-out:" + public_id,
        "operator": canonical,
        "status": status,
        "result_class": status,
        "authority": authority,
        "input_digest": _safe_digest(bounded_input),
        "output_digest": _safe_digest(bounded_result),
        **_context_fields(context),
    }
    body: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "operator": canonical,
        "status": status,
        "authority": authority,
        "result": bounded_result,
        "receipt": receipt,
    }
    if reason:
        body["reason"] = str(reason)[:MAX_TEXT]
    # The nested receipt keeps reserved journal keys out of the event's top-level
    # field namespace.  A broken audit sink must not convert evidence into a
    # runner crash.
    if journal is not None and callable(getattr(journal, "emit", None)):
        try:
            journal.emit(
                "figure_it_out_receipt",
                producer="kernel_figure_it_out",
                operator=canonical,
                status=status,
                authority=authority,
                receipt=copy.deepcopy(receipt),
                dispatch=False,
            )
        except Exception:  # noqa: BLE001 - optional audit sink is non-fatal
            pass
    try:
        _json(body)
    except FigureItOutError:
        # _safe_bounded clamps individual fields but, in pathological nested
        # cases, the assembled body can still exceed MAX_PACKET_BYTES.  Never
        # let that escape as an uncaught exception (finding 2) -- fall back to
        # a minimal, guaranteed-bounded rejection receipt instead.
        canonical_status = "REJECTED"
        body = {
            "schema": RESULT_SCHEMA,
            "operator": canonical,
            "status": canonical_status,
            "authority": "HOST_ERROR",
            "result": {"errors": ["operator result exceeds the bounded packet size"]},
            "receipt": {
                "schema": SCHEMA, "receipt_id": receipt["receipt_id"],
                "operator": canonical, "status": canonical_status,
                "result_class": canonical_status, "authority": "HOST_ERROR",
            },
        }
    return body


def _binding_mapping(raw: object) -> dict[str, object]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        result = {}
        for item in raw:
            if not isinstance(item, Mapping) or not isinstance(item.get("symbol"), str):
                raise FigureItOutError("each quantity binding needs a symbol")
            item_copy = dict(item)
            symbol = str(item_copy.pop("symbol"))
            result[symbol] = item_copy
        return result
    raise FigureItOutError("bindings must be an object or array")


def _requested_units(args: Mapping[str, object]) -> object:
    value = args.get("requested_units", args.get("output_units", args.get("units")))
    if value is not None:
        return value
    outputs = args.get("requested_outputs", args.get("outputs"))
    if isinstance(outputs, Mapping):
        return outputs.get("units", outputs.get("unit"))
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes, bytearray)):
        if outputs and isinstance(outputs[0], Mapping):
            return outputs[0].get("units", outputs[0].get("unit"))
    raise FigureItOutError("requested output units are required")


def _bind_quantity(args: Mapping[str, object], *, context=None, journal=None):
    expression = args.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise FigureItOutError("expression must be a non-empty string")
    bindings = _binding_mapping(args.get("bindings"))
    requested = _requested_units(args)
    from kernel_bind_quantity import check_dimensional_consistency, validate_witness

    check = check_dimensional_consistency(expression, bindings, requested)
    result: dict[str, object] = {
        "expression": expression,
        "requested_units": check["requested_units"],
        "actual_units": check["units"],
        "bindings": sorted(str(key) for key in bindings),
        "dimensionally_consistent": bool(check["ok"]),
        "verified": False,
    }
    if check["reason"]:
        result["reason"] = check["reason"]
    witness = args.get("witness")
    if witness is not None:
        witness_result = validate_witness(witness, expression, bindings, requested)
        result["witness"] = witness_result
        if not witness_result.get("ok"):
            return _finish("BIND_QUANTITY", "COUNTEREXAMPLE", "DETERMINISTIC_HOST",
                           result, context=context, input_payload=args, journal=journal,
                           reason=str(witness_result.get("reason", "invalid witness")))
    status = "ADMITTED" if check["ok"] else "COUNTEREXAMPLE"
    return _finish("BIND_QUANTITY", status, "DETERMINISTIC_HOST", result,
                   context=context, input_payload=args, journal=journal,
                   reason=check["reason"])


def _contract_cases(args: Mapping[str, object], *, context=None, journal=None):
    raw_cases = args.get("cases")
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes, bytearray)):
        raise FigureItOutError("cases must be an array")
    if not raw_cases or len(raw_cases) > MAX_ITEMS:
        raise FigureItOutError(f"cases must contain 1-{MAX_ITEMS} items")
    cases = []
    counterexamples = []
    unverified = []
    for index, raw in enumerate(raw_cases, 1):
        if not isinstance(raw, Mapping) or "expected" not in raw:
            raise FigureItOutError("each contract case needs expected behavior")
        expected = raw["expected"]
        case: dict[str, object] = {
            "id": str(raw.get("id", raw.get("name", f"case-{index}")))[:MAX_TEXT],
            "input": raw.get("input", raw.get("fixture")),
            "expected": expected,
        }
        if "basis" in raw:
            case["basis"] = raw["basis"]
        has_observation = "observed" in raw or "actual" in raw
        if has_observation:
            observed = raw.get("observed", raw.get("actual"))
            case["observed"] = observed
            if _json(_bounded(observed)) != _json(_bounded(expected)):
                case["comparison"] = "MISMATCH"
                counterexamples.append(case["id"])
            else:
                case["comparison"] = "MATCH"
        else:
            case["comparison"] = "NOT_OBSERVED"
            unverified.append(case["id"])
        cases.append(_bounded(case))
    # Fix round (rv_figure_it_out_REPORT.md finding 1): "observed"/"actual" is
    # caller (model) supplied -- there is no host-only execution channel wired
    # for contract cases.  Marking a caller-supplied comparison HOST_OBSERVED
    # launders an unverified claim into host authority (a fabricated
    # {"expected": 42, "observed": 42} case previously returned PASS +
    # HOST_OBSERVED without anything actually executing).  The deterministic
    # comparison itself remains meaningful evidence; its provenance does not.
    if counterexamples:
        status, authority = "COUNTEREXAMPLE", "MODEL_PROPOSED"
    elif unverified:
        status, authority = "UNVERIFIED", "MODEL_PROPOSED"
    else:
        status, authority = "PASS", "MODEL_PROPOSED"
    result = {"cases": cases, "counterexamples": counterexamples,
              "unverified": unverified, "observed_count": len(cases) - len(unverified)}
    return _finish("CONTRACT_CASES", status, authority, result,
                   context=context, input_payload=args, journal=journal)


def _observation_rows(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise FigureItOutError("trace observations must be an array")
    if not raw or len(raw) > MAX_ITEMS:
        raise FigureItOutError(f"trace observations must contain 1-{MAX_ITEMS} items")
    rows = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise FigureItOutError("each trace observation must be an object")
        if not any(key in item for key in (
                "input", "output", "side_effects", "exception", "exceptions",
                "state", "state_changes", "event")):
            raise FigureItOutError("trace observation needs an input, output, state, or exception")
        rows.append(_bounded(item))
    return rows


def _trace_behavior(args: Mapping[str, object], *, context=None, journal=None,
                    executor: Callable[..., object] | None = None):
    # Fix round (rv_figure_it_out_REPORT.md finding 1): authority must reflect
    # WHERE the observations came from.  The prior code marked every path
    # HOST_OBSERVED, including one where the model supplied "observations"/
    # "events" directly in its arguments -- a fabricated trace was returned as
    # OBSERVED/HOST_OBSERVED without anything actually executing.  Only
    # observations acquired through the real host executor callback earn
    # HOST_OBSERVED; caller-supplied observations are MODEL_PROPOSED/UNVERIFIED.
    direct_raw = args.get("observations", args.get("events"))
    executor_used = False
    if direct_raw is None:
        candidate_executor = executor
        if candidate_executor is None and isinstance(context, Mapping):
            maybe_executor = context.get("executor")
            if callable(maybe_executor):
                candidate_executor = maybe_executor
        if candidate_executor is not None:
            try:
                raw = candidate_executor(copy.deepcopy(dict(args)), copy.deepcopy(dict(context or {})))
            except TypeError:
                raw = candidate_executor(copy.deepcopy(dict(args)))
            except Exception as error:  # noqa: BLE001 - typed host failure receipt
                return _finish("TRACE_BEHAVIOR", "HOST_ERROR", "HOST_ERROR",
                               {"errors": [f"executor failed: {type(error).__name__}: {error}"]},
                               context=context, input_payload=args, journal=journal)
            if isinstance(raw, Mapping) and "observations" in raw:
                raw = raw["observations"]
            elif isinstance(raw, Mapping):
                raw = [raw]
            executor_used = True
        else:
            return _finish("TRACE_BEHAVIOR", "NEEDS_EVIDENCE", "NEEDS_EVIDENCE",
                           {"observations": [], "entry_point": args.get("entry_point")},
                           context=context, input_payload=args, journal=journal,
                           reason="no host observation was supplied")
    else:
        raw = direct_raw
    observations = _observation_rows(raw)
    result = {
        "snapshot_id": args.get("snapshot_id", _context_fields(context).get("snapshot_id")),
        "entry_point": args.get("entry_point"),
        "observations": observations,
        "verified": False,
    }
    if executor_used:
        return _finish("TRACE_BEHAVIOR", "OBSERVED", "HOST_OBSERVED", result,
                       context=context, input_payload=args, journal=journal)
    return _finish("TRACE_BEHAVIOR", "UNVERIFIED", "MODEL_PROPOSED", result,
                   context=context, input_payload=args, journal=journal,
                   reason="observations were supplied directly, not acquired by a host executor")


def execute_operator(name: str, args: Mapping[str, object] | None = None, *,
                     context: Mapping[str, object] | None = None,
                     journal: object = None,
                     executor: Callable[..., object] | None = None) -> dict[str, object]:
    """Execute one bounded operator and return its typed host result."""
    canonical = _canonical_operator(name)
    if canonical is None:
        return _error_result(str(name), args or {}, "unknown figure_it_out operator",
                             context=context, journal=journal)
    try:
        bounded_args = _bounded(dict(args or {}))
        if not isinstance(bounded_args, dict):
            raise FigureItOutError("operator arguments must be an object")
        if canonical == "BIND_QUANTITY":
            return _bind_quantity(bounded_args, context=context, journal=journal)
        if canonical == "CONTRACT_CASES":
            return _contract_cases(bounded_args, context=context, journal=journal)
        return _trace_behavior(bounded_args, context=context, journal=journal,
                               executor=executor)
    except Exception as error:  # noqa: BLE001 - every operator has a typed failure receipt
        return _error_result(canonical, args or {}, error, context=context, journal=journal)


run_operator = execute_operator


def render_operator_block(receipts: Sequence[Mapping[str, object]] | None = None) -> str:
    """Render a small host-authored instruction/receipt block for the request."""
    lines = [
        "[FIGURE_IT_OUT HOST OPERATORS]",
        "Available typed host operators: " + ", ".join(OPERATOR_NAMES) + ".",
        "CONTRACT_CHECK is accepted as a compatibility alias for CONTRACT_CASES.",
        "Host results are evidence only; they do not certify a repair or completion.",
        "Use public inputs and bounded observations; do not infer unseen behavior.",
    ]
    rows = []
    for item in list(receipts or [])[-MAX_ITEMS:]:
        if not isinstance(item, Mapping):
            continue
        row = item.get("receipt") if isinstance(item.get("receipt"), Mapping) else item
        rows.append("receipt={receipt} operator={operator} status={status} authority={authority}".format(
            receipt=str(row.get("receipt_id", ""))[:MAX_TEXT],
            operator=str(row.get("operator", ""))[:MAX_TEXT],
            status=str(row.get("status", row.get("result_class", "")))[:MAX_TEXT],
            authority=str(row.get("authority", ""))[:MAX_TEXT],
        ))
    lines.append("[FIGURE_IT_OUT RECEIPTS]")
    lines.extend(rows or ["No operator receipt has been produced in this snapshot."])
    return "\n".join(lines)[:MAX_PACKET_BYTES]


def append_operator_message(prepared: Sequence[Mapping[str, object]],
                            receipts: Sequence[Mapping[str, object]] | None = None,
                            *, provider_request: bool = True):
    """Append the host block only on the enabled provider-request path."""
    if not enabled() or not provider_request:
        return prepared
    return [*prepared, {"role": "user", "content": render_operator_block(receipts)}]


__all__ = [
    "MODE_ENV", "SCHEMA", "RESULT_SCHEMA", "OPERATOR_NAMES", "OPERATOR_ALIASES",
    "FigureItOutError", "mode", "enabled", "is_operator", "tool_surface",
    "operator_tool_surface", "tool_names", "execute_operator", "run_operator",
    "render_operator_block", "append_operator_message",
]

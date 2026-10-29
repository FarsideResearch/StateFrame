"""Kernel-owned BIND_QUANTITY operator.

The operator is deliberately small and conservative.  The provider supplies a
semantic interpretation, while this module owns trigger eligibility, JSON
parsing, dimensional evaluation, witness admission, provenance, and bounded
rendering.  No provider prose is promoted to a verified fact.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence


MODE_ENV = "STATEFRAME_BIND_QUANTITY"
OPERATOR = "BIND_QUANTITY"
MAX_REASONING_BUDGET = 1024
FINDINGS_TOKEN_CAP = 400
_STATUS_VALUES = {
    "ANSWERED", "UNKNOWN", "NEEDS_EVIDENCE", "NEEDS_JOINT_CONTEXT",
    "INCOMPLETE", "BUDGET_EXHAUSTED",
}
_INACTIVE_FINDING_STATUSES = {
    "EXECUTION_FAILURE", "FAILED", "MALFORMED", "REJECTED", "SUPERSEDED",
    "INVALIDATED", "HISTORICAL", "INACTIVE",
}
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
_UNIT = (r"seconds?|minutes?|hours?|milliseconds?|microseconds?|ms|us|"
         r"tokens?|bytes?|requests?|items?|records?|percent(?:age)?|%|"
         r"kilobytes?|megabytes?|gigabytes?|kb|mb|gb|hertz|hz")
_UNIT_LANGUAGE = re.compile(
    r"(?ix)(?<!\w)(?:"
    rf"{_NUMBER}\s*(?:{_UNIT})|"
    rf"(?:{_UNIT})\s*{_NUMBER}|"
    rf"{_NUMBER}\s*/\s*(?:s|sec|second|ms|millisecond)\b|"
    r"per\s+(?:second|minute|hour|request|token|byte)|"
    r"/\s*(?:s|sec|second|ms|millisecond)\b|"
    r"(?:rate|duration|elapsed|interval|timeout|latency|capacity|quota|refill)"
    r"(?:\s+(?:of|is|was|at|per|within|over|under|above|below))?"
    r"(?:\s+|\s*[:=])?"
    rf"(?:{_NUMBER}|{_UNIT})|"
    r"(?:at\s+least|at\s+most|no\s+more\s+than|no\s+less\s+than|"
    r"less\s+than|more\s+than|under|over|below|above|threshold|cap|limit|"
    r"max(?:imum)?|min(?:imum)?)"
    rf"\s*[:=]?\s*{_NUMBER}"
    r")(?!\w)"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])(?:\s+|$)")


class BindQuantityError(ValueError):
    """A bounded operator packet or result was not admissible."""


class BindQuantityParseError(BindQuantityError):
    """The provider did not return a strict JSON object."""


class BindQuantityUnavailable(BindQuantityError):
    """The optional live measurement could not be bound to an actor slot."""


def bind_quantity_mode() -> str:
    """Return the strictly validated feature mode."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return bind_quantity_mode() == "on"


def live_slot_id() -> int | None:
    """Return the worker's explicit provider slot, or None when unusable.

    BIND_QUANTITY is an optional measurement.  It must not invent a slot or
    borrow the actor's request body, so an absent, malformed, or negative slot
    disables only this measurement.
    """
    raw = os.environ.get("REACT_SLOT_ID")
    if raw is None or not raw.strip():
        return None
    try:
        slot = int(raw)
    except (TypeError, ValueError):
        return None
    return slot if slot >= 0 else None


def lexical_token_count(value: object) -> int:
    return len(_TOKEN_PATTERN.findall(value if isinstance(value, str) else str(value)))


def contains_quantity_language(brief: object) -> bool:
    """Use only public brief text for routing; source and tests are not cues."""
    return isinstance(brief, str) and bool(_UNIT_LANGUAGE.search(brief))


def extract_quantity_sentences(brief: str | None) -> list[str]:
    """Return exact public sentences containing quantity/unit language."""
    if not isinstance(brief, str) or not brief.strip():
        return []
    pieces = _SENTENCE_SPLIT.split(brief)
    if len(pieces) == 1 and "\n" in brief:
        pieces = brief.splitlines()
    return [piece.strip() for piece in pieces
            if piece.strip() and _UNIT_LANGUAGE.search(piece)]


def brief_from_messages(messages: Sequence[Mapping[str, object]] | None) -> str:
    """Recover the public task brief from the worker's protected user message."""
    if not messages:
        return ""
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = re.match(r"(?:Task|Bug report):\s*(.*?)(?:\n\nTarget:|\Z)", content,
                        flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        if content.strip():
            return content.strip()
    return ""


def _finding_kind(value: Mapping[str, object]) -> bool:
    names = {str(value.get(key, "")).upper() for key in ("kind", "operator", "type")}
    return OPERATOR in names or "QUANTITY_BINDING" in names


def is_active_binding_finding(value: object) -> bool:
    """Find an active BIND_QUANTITY finding without treating failures as findings."""
    if isinstance(value, Mapping):
        if _finding_kind(value):
            if value.get("active") is False:
                return False
            status = str(value.get("status", value.get("disposition", ""))).upper()
            if status in _INACTIVE_FINDING_STATUSES:
                return False
            if value.get("disposition") in {"superseded", "rejected", "invalidated"}:
                return False
            return True
        for key, child in value.items():
            if key in {"model_response", "response_content", "reasoning_content"}:
                continue
            if is_active_binding_finding(child):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(is_active_binding_finding(child) for child in value)
    return False


def should_trigger(brief: object, findings: object = None, *, attempted: bool = False) -> bool:
    """Public eligibility: unit language, no active finding, and one task attempt."""
    return (not attempted and contains_quantity_language(brief)
            and not is_active_binding_finding(findings))


def _source_fragment(source: str, symbol: str) -> str:
    """Extract the complete current function/class when the source parses."""
    if not isinstance(source, str):
        return ""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    parts = symbol.rsplit(".", 1)
    if len(parts) == 2:
        owner, member = parts
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == owner:
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and child.name == member:
                        return ast.get_source_segment(source, child) or source
    name = parts[-1]
    matches = [node for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               and node.name == name]
    if matches:
        return ast.get_source_segment(source, matches[0]) or source
    return source


OUTPUT_CONTRACT = {
    "status": "ANSWERED|UNKNOWN|NEEDS_EVIDENCE|NEEDS_JOINT_CONTEXT|INCOMPLETE|BUDGET_EXHAUSTED",
    "bindings": "[{symbol, meaning, semantic_kind, units, scale, clock, basis_refs}]",
    "requested_outputs": "[{name, units}]",
    "expressions": "[{requested_output, expression}]",
    "witness": "{values, expected_result, away_from_caps, saturated}",
    "assumptions": "[string]",
}


def compile_packet(*, brief: str, source: str, symbol: str, path: str = "",
                   task_id: str | None = None, snapshot_id: str | None = None) -> dict:
    """Compile a complete, public-evidence-only operator packet."""
    quotes = extract_quantity_sentences(brief)
    if not quotes:
        raise BindQuantityError("BIND_QUANTITY requires a quantity-bearing public brief")
    source_fragment = {
        "path": path,
        "symbol": symbol,
        "content": _source_fragment(source, symbol),
    }
    contract_constraints = [
        "Use only the supplied public brief and current source.",
        "A dimensionally valid expression is not semantic proof; preserve assumptions.",
        "Do not claim verification or completion.",
    ]
    return {
        "operator": OPERATOR,
        "task_id": task_id,
        "snapshot_id": snapshot_id,
        "root_header": {
            "goal": "derive quantity bindings for the current task basis",
            "current_decision": "operator diagnosis only; no repair or completion decision",
            "contract_constraints": contract_constraints,
            "snapshot_identity": {"task_id": task_id, "snapshot_id": snapshot_id},
        },
        "question": (
            "For every quantity symbol in the supplied brief and current source, state its "
            "meaning, semantic kind, units as an exponent map, scale, and clock convention "
            "when relevant. State the REQUESTED OUTPUT "
            "quantity of each relevant expression, then give the unit-consistent expression "
            "for that output. Supply one NON-DEGENERATE witness whose numeric values are not "
            "1 or 0 and are away from caps/saturation. Allow UNKNOWN or NEEDS_EVIDENCE."
        ),
        "brief_quotes_with_units": quotes,
        "evidence": {"brief_quotes_with_units": quotes, "current_source": source_fragment},
        "current_source": source_fragment,
        "prerequisites": [],
        "findings": [],
        "contradictions": [],
        "resources": {
            "request_count_ceiling": 2,
            "reasoning_tokens_per_request_ceiling": MAX_REASONING_BUDGET,
            "format_repair_ceiling": 1,
        },
        "output_contract": OUTPUT_CONTRACT,
        "constraints": [
            "Return one JSON object in the message content; no Markdown or commentary.",
            *contract_constraints,
        ],
    }


def _packet_messages(packet: Mapping[str, object], malformed: str | None = None) -> list[dict]:
    system = (
        "You are the bounded BIND_QUANTITY thinking-only worker. Do not call tools, edit, "
        "test, browse, or propose actions. Treat the packet as the complete public evidence. "
        "Return only a strict JSON object in the final content. Model interpretations remain "
        "unverified and may be UNKNOWN or NEEDS_EVIDENCE."
    )
    content = json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": content}]
    if malformed is not None:
        messages.append({"role": "user", "content": (
            "FORMAT REPAIR ONLY. The following is untrusted prior output. Return only one "
            "strict JSON object matching the output_contract in the packet; do not explain it.\n"
            "<prior_output>\n" + malformed + "\n</prior_output>"
        )})
    return messages


def configure_operator_request(body: dict) -> dict:
    """Use the existing budget configurator, then enforce this request's hard cap."""
    try:
        from kernel_step_contract import configure_reasoning_budget
    except ImportError:
        configure_reasoning_budget = None
    if configure_reasoning_budget is not None:
        configure_reasoning_budget(body)
    raw = body.get("reasoning_budget_tokens")
    if type(raw) is not int or raw < 0 or raw > MAX_REASONING_BUDGET:
        body["reasoning_budget_tokens"] = MAX_REASONING_BUDGET
    body["reasoning_budget_tokens"] = min(
        MAX_REASONING_BUDGET, int(body["reasoning_budget_tokens"]))
    return body


def build_request_body(packet: Mapping[str, object], *, model: str,
                       temperature: float = 0.0, malformed: str | None = None) -> dict:
    """Build the no-tools JSON-object provider request."""
    body = {
        "model": model,
        "messages": _packet_messages(packet, malformed),
        "max_tokens": MAX_REASONING_BUDGET,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    configure_operator_request(body)
    body.pop("tools", None)
    body.pop("tool_choice", None)
    body.pop("parallel_tool_calls", None)
    return body


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def _estimate_tokens(value: object) -> int:
    return max(1, (len(_canonical_json(value)) + 3) // 4)


def _strict_object_pairs(pairs: list[tuple[str, object]]) -> dict:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BindQuantityParseError("duplicate JSON object key: " + key)
        result[key] = value
    return result


def parse_json_object(content: object) -> dict:
    """Parse only strict JSON object content; fences and NaN are rejected."""
    if not isinstance(content, str):
        raise BindQuantityParseError("provider content is not a string")
    try:
        value = json.loads(content, object_pairs_hook=_strict_object_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               BindQuantityParseError("invalid JSON number: " + value)))
    except BindQuantityParseError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise BindQuantityParseError("malformed JSON: " + str(error)) from error
    if not isinstance(value, dict):
        raise BindQuantityParseError("operator result must be a JSON object")
    return value


_UNIT_ALIASES: dict[str, tuple[str, float]] = {
    "token": ("token", 1.0), "tokens": ("token", 1.0),
    "second": ("second", 1.0), "seconds": ("second", 1.0), "sec": ("second", 1.0),
    "s": ("second", 1.0), "millisecond": ("second", 0.001),
    "milliseconds": ("second", 0.001), "ms": ("second", 0.001),
    "microsecond": ("second", 0.000001), "microseconds": ("second", 0.000001),
    "us": ("second", 0.000001),
    "byte": ("byte", 1.0), "bytes": ("byte", 1.0),
    "request": ("request", 1.0), "requests": ("request", 1.0),
    "item": ("item", 1.0), "items": ("item", 1.0),
    "record": ("record", 1.0), "records": ("record", 1.0),
    "percent": ("percent", 1.0), "percentage": ("percent", 1.0), "%": ("percent", 1.0),
    "job": ("job", 1.0), "jobs": ("job", 1.0),
    "count": ("count", 1.0), "counts": ("count", 1.0),
    "dimensionless": ("__dimensionless__", 1.0), "unitless": ("__dimensionless__", 1.0),
}


def _unit_atom(value: object) -> tuple[str, float]:
    if not isinstance(value, str):
        raise BindQuantityError("unit atom must be a string")
    atom = value.strip().lower().replace("−", "-")
    atom = re.sub(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*", "", atom)
    atom = atom.replace("_", "")
    if atom in _UNIT_ALIASES:
        return _UNIT_ALIASES[atom]
    raise BindQuantityError("unknown unit: " + value)


def parse_units(value: object) -> dict[str, int]:
    """Normalize a unit expression to a base-unit exponent map."""
    if isinstance(value, Mapping):
        if "units" in value and len(value) == 1:
            return parse_units(value["units"])
        result: dict[str, int] = {}
        for raw_key, raw_exponent in value.items():
            if raw_key in {"scale", "factor", "kind", "semantic_kind"}:
                continue
            base, _scale = _unit_atom(str(raw_key))
            if base == "__dimensionless__":
                continue
            if isinstance(raw_exponent, bool) or not isinstance(raw_exponent, (int, float)) \
                    or not math.isfinite(float(raw_exponent)) \
                    or float(raw_exponent) != int(float(raw_exponent)):
                raise BindQuantityError("unit exponents must be finite integers")
            exponent = int(raw_exponent)
            if exponent:
                result[base] = result.get(base, 0) + exponent
        return {key: exponent for key, exponent in sorted(result.items()) if exponent}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: dict[str, int] = {}
        for item in value:
            for key, exponent in parse_units(item).items():
                result[key] = result.get(key, 0) + exponent
        return {key: exponent for key, exponent in sorted(result.items()) if exponent}
    if not isinstance(value, str):
        raise BindQuantityError("units are required")
    text = value.strip().lower().replace("·", "*").replace("−", "-")
    if not text or text in {"1", "", "dimensionless", "unitless"}:
        return {}
    text = re.sub(r"\bper\b", "/", text)
    text = re.sub(r"\s+", "", text)
    numerator, *denominators = text.split("/")
    result: dict[str, int] = {}

    def add_product(product: str, sign: int) -> None:
        for atom in filter(None, product.split("*")):
            match = re.fullmatch(r"(.+?)(?:\^([+-]?\d+))?", atom)
            if not match:
                raise BindQuantityError("invalid unit expression: " + value)
            base, scale = _unit_atom(match.group(1))
            if base == "__dimensionless__":
                continue
            exponent = int(match.group(2) or "1") * sign
            result[base] = result.get(base, 0) + exponent

    add_product(numerator, 1)
    for denominator in denominators:
        add_product(denominator, -1)
    return {key: exponent for key, exponent in sorted(result.items()) if exponent}


def unit_scale(value: object) -> float:
    """Return the SI-like scale carried by a simple unit expression."""
    if isinstance(value, Mapping):
        raw = value.get("scale", value.get("factor", 1.0))
        if isinstance(raw, Mapping):
            raw = raw.get("factor", 1.0)
        try:
            scale = float(raw)
        except (TypeError, ValueError) as error:
            raise BindQuantityError("scale must be numeric") from error
        if not math.isfinite(scale) or scale <= 0:
            raise BindQuantityError("scale must be finite and positive")
        return scale
    if not isinstance(value, str):
        return 1.0
    text = re.sub(r"\bper\b", "/", value.lower().replace("·", "*"))
    text = re.sub(r"\s+", "", text)
    numerator, *denominators = text.split("/")
    scale = 1.0
    for atom in filter(None, numerator.split("*")):
        scale *= _unit_atom(atom)[1]
    for denominator in denominators:
        for atom in filter(None, denominator.split("*")):
            scale /= _unit_atom(atom)[1]
    return scale


def _binding_value(raw: object, symbol: str) -> dict:
    if isinstance(raw, Mapping):
        units_raw = raw.get("units", raw.get("unit"))
        if units_raw is None:
            units_raw = {key: value for key, value in raw.items()
                         if key not in {"meaning", "scale", "factor", "kind", "semantic_kind",
                                        "basis", "basis_refs"}}
        meaning = raw.get("meaning", "UNKNOWN")
        semantic_kind = raw.get("semantic_kind", raw.get("kind", "UNKNOWN"))
        clock = raw.get("clock", raw.get("clock_domain", "UNKNOWN"))
        scale_raw = raw.get("scale", unit_scale(units_raw))
        basis = raw.get("basis_refs", raw.get("basis", []))
    else:
        units_raw, meaning, semantic_kind, clock, scale_raw, basis = (
            raw, "UNKNOWN", "UNKNOWN", "UNKNOWN", unit_scale(raw), [])
    units = parse_units(units_raw)
    try:
        scale = float(scale_raw) if not isinstance(scale_raw, str) else scale_raw
    except (TypeError, ValueError):
        scale = str(scale_raw)
    if isinstance(basis, str):
        basis = [basis]
    elif not isinstance(basis, Sequence):
        basis = []
    return {"symbol": symbol, "meaning": str(meaning),
            "semantic_kind": str(semantic_kind), "units": units,
            "scale": scale, "clock": str(clock),
            "basis_refs": [str(item) for item in basis]}


def _normalise_bindings(raw: object) -> dict[str, dict]:
    if isinstance(raw, Mapping):
        entries = [{"symbol": key, **(value if isinstance(value, Mapping) else {"units": value})}
                   for key, value in raw.items()]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        entries = list(raw)
    else:
        raise BindQuantityError("bindings must be an object or array")
    result: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("symbol"), str):
            raise BindQuantityError("each binding needs a symbol")
        symbol = entry["symbol"]
        if symbol in result:
            raise BindQuantityError("duplicate binding: " + symbol)
        result[symbol] = _binding_value(entry, symbol)
    if not result:
        raise BindQuantityError("at least one quantity binding is required")
    return result


def _normalise_outputs(raw: object) -> list[dict]:
    if raw is None:
        raise BindQuantityError("requested output quantity is required")
    if isinstance(raw, Mapping):
        raw_values = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        raw_values = list(raw)
    else:
        raw_values = [{"name": "requested_output", "units": raw}]
    result = []
    for index, item in enumerate(raw_values):
        if isinstance(item, str):
            item = {"name": "output_" + str(index + 1), "units": item}
        if not isinstance(item, Mapping):
            raise BindQuantityError("requested outputs must be objects")
        name = item.get("name", item.get("symbol", item.get("quantity", item.get("output"))))
        if isinstance(name, Mapping):
            name = name.get("name", name.get("symbol"))
        if not isinstance(name, str) or not name.strip():
            name = "output_" + str(index + 1)
        units_raw = item.get("units", item.get("unit"))
        if units_raw is None:
            raise BindQuantityError("requested output needs units: " + name)
        result.append({"name": name, "units": parse_units(units_raw),
                       "scale": item.get("scale", unit_scale(units_raw)),
                       "basis_refs": item.get("basis_refs", item.get("basis", []))})
    if not result:
        raise BindQuantityError("at least one requested output is required")
    return result


def _normalise_expressions(payload: Mapping[str, object], outputs: list[dict]) -> list[dict]:
    raw = payload.get("expressions")
    if raw is None and "expression" in payload:
        raw = [{"requested_output": payload.get("requested_output"),
                "expression": payload.get("expression")}]
    if isinstance(raw, str):
        raw = [{"requested_output": payload.get("requested_output"), "expression": raw}]
    elif isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise BindQuantityError("expressions are required")
    result = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            item = {"expression": item}
        if not isinstance(item, Mapping) or not isinstance(item.get("expression"), str):
            raise BindQuantityError("each expression needs a string")
        target = item.get("requested_output", item.get("output", item.get("quantity",
                         item.get("name"))))
        if isinstance(target, Mapping):
            target = target.get("name", target.get("symbol"))
        if not isinstance(target, str):
            if len(outputs) == 1:
                target = outputs[0]["name"]
            elif index < len(outputs):
                target = outputs[index]["name"]
            else:
                raise BindQuantityError("expression is not tied to a requested output")
        result.append({"requested_output": target, "expression": item["expression"],
                       "basis_refs": item.get("basis_refs", item.get("basis", []))})
    if len(result) != len(outputs):
        raise BindQuantityError("every requested output needs one expression")
    return result


class _Evaluated:
    def __init__(self, units: dict[str, int], value: float | None = None):
        self.units = units
        self.value = value


def _merge_units(left: Mapping[str, int], right: Mapping[str, int], sign: int) -> dict[str, int]:
    result = dict(left)
    for key, exponent in right.items():
        result[key] = result.get(key, 0) + sign * exponent
        if not result[key]:
            result.pop(key)
    return dict(sorted(result.items()))


def _numeric_constant(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BindQuantityError("only finite numeric constants are allowed")
    value = float(value)
    if not math.isfinite(value):
        raise BindQuantityError("non-finite numeric constant")
    return value


def _evaluate_node(node: ast.AST, bindings: Mapping[str, dict], values: Mapping[str, object] | None) -> _Evaluated:
    if isinstance(node, ast.Name):
        if node.id not in bindings:
            raise BindQuantityError("unbound quantity symbol: " + node.id)
        value = None
        if values is not None and node.id in values:
            value = _numeric_constant(values[node.id])
        return _Evaluated(dict(bindings[node.id]["units"]), value)
    if isinstance(node, ast.Constant):
        return _Evaluated({}, _numeric_constant(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_node(node.operand, bindings, values)
        if value.value is not None and isinstance(node.op, ast.USub):
            value.value = -value.value
        return value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div,
                                                               ast.Add, ast.Sub)):
        left = _evaluate_node(node.left, bindings, values)
        right = _evaluate_node(node.right, bindings, values)
        if isinstance(node.op, ast.Mult):
            units = _merge_units(left.units, right.units, 1)
        elif isinstance(node.op, ast.Div):
            units = _merge_units(left.units, right.units, -1)
        else:
            if left.units != right.units:
                raise BindQuantityError("addition/subtraction requires like units")
            units = dict(left.units)
        numeric = None
        if left.value is not None and right.value is not None:
            if isinstance(node.op, ast.Mult):
                numeric = left.value * right.value
            elif isinstance(node.op, ast.Div):
                if right.value == 0:
                    raise BindQuantityError("witness division by zero")
                numeric = left.value / right.value
            elif isinstance(node.op, ast.Add):
                numeric = left.value + right.value
            else:
                numeric = left.value - right.value
        return _Evaluated(units, numeric)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in {"min", "max"} and not node.keywords and node.args:
        values_evaluated = [_evaluate_node(arg, bindings, values) for arg in node.args]
        if any(item.units != values_evaluated[0].units for item in values_evaluated[1:]):
            raise BindQuantityError("min/max requires like units")
        numeric_values = [item.value for item in values_evaluated]
        numeric = None if any(item is None for item in numeric_values) else (
            min(numeric_values) if node.func.id == "min" else max(numeric_values))
        return _Evaluated(dict(values_evaluated[0].units), numeric)
    raise BindQuantityError("unsupported expression syntax")


def evaluate_expression(expression: str, bindings: Mapping[str, object],
                        values: Mapping[str, object] | None = None) -> dict:
    """Evaluate dimensions, and numeric value when a witness value map is supplied."""
    if not isinstance(expression, str) or not expression.strip():
        raise BindQuantityError("expression must be a non-empty string")
    canonical = {str(symbol): _binding_value(raw, str(symbol))
                 for symbol, raw in bindings.items()}
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise BindQuantityError("invalid expression: " + str(error)) from error
    result = _evaluate_node(tree.body, canonical, values)
    return {"units": result.units, "value": result.value}


def evaluate_units(expression: str, bindings: Mapping[str, object]) -> dict[str, int]:
    return evaluate_expression(expression, bindings)["units"]


def check_dimensional_consistency(expression: str, bindings: Mapping[str, object],
                                  requested_units: object) -> dict:
    expected = parse_units(requested_units)
    try:
        actual = evaluate_units(expression, bindings)
    except BindQuantityError as error:
        return {"ok": False, "units": None, "requested_units": expected,
                "reason": str(error)}
    return {"ok": actual == expected, "units": actual, "requested_units": expected,
            "reason": None if actual == expected else "expression units do not match requested output"}


def dimensional_consistency(expression: str, bindings: Mapping[str, object],
                            requested_units: object) -> bool:
    return bool(check_dimensional_consistency(expression, bindings, requested_units)["ok"])


def _number(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise BindQuantityError(label + " must be numeric") from error
    if not math.isfinite(number):
        raise BindQuantityError(label + " must be finite")
    return number


def validate_witness(witness: object, expression: str, bindings: Mapping[str, dict],
                     requested_units: object) -> dict:
    if not isinstance(witness, Mapping):
        return {"ok": False, "reason": "witness must be an object"}
    if witness.get("saturated") is True or witness.get("at_cap") is True \
            or witness.get("clipped") is True or witness.get("away_from_caps") is False:
        return {"ok": False, "reason": "saturation/cap witness is degenerate"}
    saturation = str(witness.get("saturation", "")).lower()
    if saturation and saturation not in {"none", "false", "no", "unsaturated"}:
        return {"ok": False, "reason": "saturation/cap witness is degenerate"}
    values = witness.get("values")
    if not isinstance(values, Mapping) or not values:
        return {"ok": False, "reason": "witness values are required"}
    numeric_values = {}
    try:
        for name, value in values.items():
            number = _number(value, "witness value " + str(name))
            if number in {0.0, 1.0}:
                return {"ok": False, "reason": "witness values must be non-degenerate (not 0 or 1)"}
            numeric_values[str(name)] = number
    except BindQuantityError as error:
        return {"ok": False, "reason": str(error)}
    expected_raw = witness.get("expected_result", witness.get("expected"))
    if isinstance(expected_raw, Mapping):
        expected_raw = expected_raw.get("value", expected_raw.get("amount"))
    try:
        expected = _number(expected_raw, "witness expected result")
        evaluated = evaluate_expression(expression, bindings, numeric_values)
    except BindQuantityError as error:
        return {"ok": False, "reason": str(error)}
    if evaluated["units"] != parse_units(requested_units):
        return {"ok": False, "reason": "witness expression units do not match requested output"}
    if evaluated["value"] is None or not math.isclose(evaluated["value"], expected,
                                                        rel_tol=1e-9, abs_tol=1e-9):
        return {"ok": False, "reason": "witness expected result does not evaluate from values"}
    caps = []
    for key in ("cap", "capacity", "maximum", "max_value"):
        if key in witness:
            caps.append(witness[key])
        if key in values:
            caps.append(values[key])
    try:
        if any(math.isclose(expected, _number(cap, "cap"), rel_tol=0, abs_tol=1e-12)
               for cap in caps):
            return {"ok": False, "reason": "witness result is at a cap/saturation boundary"}
    except BindQuantityError as error:
        return {"ok": False, "reason": str(error)}
    return {"ok": True, "values": numeric_values, "expected_result": expected,
            "units": evaluated["units"]}


def _basis_refs(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def _dependency_coverage(bindings: Mapping[str, dict], quotes: Sequence[str]) -> str:
    references = [reference for item in bindings.values()
                 for reference in item.get("basis_refs", [])]
    if not references:
        return "unknown"
    quote_set = set(quotes)
    if all(reference in quote_set or re.fullmatch(r"(?:C\d+|brief|contract)", reference,
                                                   flags=re.I) for reference in references):
        return "contract-only"
    return "unknown"


def _finding_id(packet: Mapping[str, object], payload: Mapping[str, object]) -> str:
    return "bindq:" + hashlib.sha256(_canonical_json({
        "task_id": packet.get("task_id"), "snapshot_id": packet.get("snapshot_id"),
        "payload": payload,
    })).hexdigest()[:24]


def validate_response(payload: Mapping[str, object], packet: Mapping[str, object]) -> dict:
    """Admit a parsed provider object or return a factual rejection."""
    if not isinstance(payload, Mapping):
        return {"ok": False, "status": "EXECUTION_FAILURE", "reason": "result is not an object"}
    status = str(payload.get("status", "ANSWERED")).upper()
    if status not in _STATUS_VALUES:
        return {"ok": False, "status": "EXECUTION_FAILURE", "reason": "unknown result status"}
    if status in {"INCOMPLETE", "BUDGET_EXHAUSTED"}:
        return {"ok": False, "status": "EXECUTION_FAILURE",
                "reason": "provider reported an execution budget status"}
    if status != "ANSWERED":
        return {"ok": False, "status": status, "reason": "model routed the uncertainty"}
    try:
        bindings = _normalise_bindings(payload.get("bindings"))
        outputs_raw = payload.get("requested_outputs", payload.get("requested_output"))
        outputs = _normalise_outputs(outputs_raw)
        expressions = _normalise_expressions(payload, outputs)
        for expression in expressions:
            output = next((item for item in outputs
                           if item["name"] == expression["requested_output"]), None)
            if output is None:
                raise BindQuantityError("expression targets unknown requested output")
            check = check_dimensional_consistency(expression["expression"], bindings,
                                                  output["units"])
            if not check["ok"]:
                raise BindQuantityError(
                    f"{expression['expression']} is dimensionally inconsistent: {check['reason']}")
        witness_raw = payload.get("witness")
        witnesses = payload.get("witnesses")
        if witnesses is None:
            witnesses = [witness_raw]
        elif isinstance(witnesses, Mapping):
            witnesses = [witnesses]
        if not isinstance(witnesses, Sequence) or isinstance(witnesses, (str, bytes)):
            raise BindQuantityError("witness is required")
        valid_witness = None
        witness_reasons = []
        first_expression = expressions[0]
        first_output = next(item for item in outputs
                            if item["name"] == first_expression["requested_output"])
        for witness in witnesses:
            checked = validate_witness(witness, first_expression["expression"], bindings,
                                       first_output["units"])
            if checked["ok"]:
                valid_witness = checked
                break
            witness_reasons.append(checked.get("reason", "invalid witness"))
        if valid_witness is None:
            raise BindQuantityError("no non-degenerate witness: " + "; ".join(witness_reasons))
    except BindQuantityError as error:
        return {"ok": False, "status": "EXECUTION_FAILURE", "reason": str(error)}

    quotes = packet.get("brief_quotes_with_units", [])
    quotes = [str(quote) for quote in quotes] if isinstance(quotes, Sequence) else []
    assumptions = payload.get("assumptions", [])
    if isinstance(assumptions, str):
        assumptions = [assumptions]
    if not isinstance(assumptions, Sequence):
        assumptions = []
    basis_quotes = list(quotes)
    finding = {
        "schema": "stateframe-finding/1",
        "id": _finding_id(packet, payload),
        "kind": OPERATOR,
        "operator": OPERATOR,
        "status": "ADOPTED",
        "active": True,
        "origin": "model interpretation",
        "authority": "model interpretation",
        "authority_class": "MODEL_INTERPRETATION",
        "verification_status": "UNVERIFIED",
        "applicability": "current",
        "disposition": "adopted",
        "basis_quotes": basis_quotes,
        "assumptions": [str(item) for item in assumptions],
        "dependency_coverage": _dependency_coverage(bindings, basis_quotes),
        "bindings": list(bindings.values()),
        "requested_outputs": outputs,
        "expressions": expressions,
        "witness": valid_witness,
        "task_id": packet.get("task_id"),
        "snapshot_id": packet.get("snapshot_id"),
    }
    return {"ok": True, "status": "ADOPTED", "finding": finding}


def _choice_content(choice: object) -> str | None:
    if isinstance(choice, str):
        return choice
    if not isinstance(choice, Mapping):
        return None
    choices = choice.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)) and choices:
        return _choice_content(choices[0])
    message = choice.get("message")
    if isinstance(message, Mapping):
        value = message.get("content")
        return value if isinstance(value, str) else None
    value = choice.get("content")
    return value if isinstance(value, str) else None


def _split_send_result(result: object) -> tuple[object, dict]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return result[0], dict(result[1])
    return result, {}


def _usage(result: object, body: Mapping[str, object], meta: Mapping[str, object]) -> dict:
    usage = meta.get("usage") if isinstance(meta, Mapping) else None
    if not isinstance(usage, Mapping) and isinstance(result, Mapping):
        usage = result.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    prompt = usage.get("prompt_tokens", usage.get("prompt_n"))
    completion = usage.get("completion_tokens", usage.get("completion_n"))
    total = usage.get("total_tokens", usage.get("total_n"))
    if type(prompt) is not int or prompt < 0:
        prompt = _estimate_tokens(body.get("messages", []))
        prompt_basis = "estimate"
    else:
        prompt_basis = "backend"
    if type(completion) is not int or completion < 0:
        completion = MAX_REASONING_BUDGET
        completion_basis = "reserved_ceiling"
    else:
        completion_basis = "backend"
    if type(total) is not int or total < 0:
        total = prompt + completion
        total_basis = "derived"
    else:
        total_basis = "backend"
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total, "prompt_basis": prompt_basis,
            "completion_basis": completion_basis, "total_basis": total_basis,
            "counted": total}


def run_operator(packet: Mapping[str, object], *, model: str,
                 send: Callable[[dict], object], temperature: float = 0.0,
                 journal: object = None, charge: Callable[[str], object] | None = None) -> dict:
    """Run at most one request plus one strict-format repair request."""
    malformed: str | None = None
    request_count = 0
    for attempt in range(2):
        body = build_request_body(packet, model=model, temperature=temperature,
                                   malformed=malformed)
        request_count += 1
        request_id = uuid.uuid4().hex
        accounting = _usage(None, body, {})
        if journal is not None:
            journal.emit("bind_quantity_request", operator=OPERATOR,
                          request_id=request_id, attempt=attempt,
                          format_repair=attempt == 1, tools_present=False,
                          reasoning_budget_tokens=body["reasoning_budget_tokens"],
                          requested_output_tokens=body["max_tokens"],
                          prompt_tokens=accounting["prompt_tokens"],
                          counted_token_ceiling=accounting["counted"])
        try:
            if charge is not None:
                charge("BIND_QUANTITY provider request")
            sent = send(body)
            choice, meta = _split_send_result(sent)
            accounting = _usage(choice, body, meta)
        except Exception as error:  # transport/host failure is not a model conclusion
            if isinstance(error, BindQuantityUnavailable) or (
                    isinstance(error, ValueError)
                    and any(term in str(error).lower()
                            for term in ("live slot", "id_slot", "slot required"))):
                if journal is not None:
                    journal.emit("bind_quantity_execution", operator=OPERATOR,
                                 request_id=request_id, attempt=attempt,
                                 status="degraded", failure_class="host",
                                 error_type=type(error).__name__, reason=str(error),
                                 token_accounting=accounting)
                return {"status": "DEGRADED", "reason": str(error),
                        "requests": request_count}
            failure_class = ("transport" if isinstance(error, (OSError, TimeoutError,
                                                                  ConnectionError))
                             else "host")
            if journal is not None:
                journal.emit("bind_quantity_execution", operator=OPERATOR,
                             request_id=request_id, attempt=attempt,
                             status="execution_failure", failure_class=failure_class,
                             error_type=type(error).__name__,
                             reason=str(error),
                             token_accounting=accounting)
            return {"status": "EXECUTION_FAILURE", "reason": str(error),
                    "requests": request_count}
        content = _choice_content(choice)
        try:
            payload = parse_json_object(content)
        except BindQuantityParseError as error:
            if attempt == 0:
                malformed = content if isinstance(content, str) else repr(content)
                if journal is not None:
                    journal.emit("bind_quantity_execution", operator=OPERATOR,
                                 request_id=request_id, attempt=attempt,
                                 status="format_repair_requested", failure_class="malformed_json",
                                 reason=str(error), token_accounting=accounting)
                continue
            if journal is not None:
                journal.emit("bind_quantity_execution", operator=OPERATOR,
                             request_id=request_id, attempt=attempt,
                             status="execution_failure", failure_class="malformed_json",
                             reason=str(error), token_accounting=accounting)
            return {"status": "EXECUTION_FAILURE", "reason": str(error),
                    "requests": request_count}
        admitted = validate_response(payload, packet)
        if journal is not None:
            journal.emit("bind_quantity_execution", operator=OPERATOR,
                         request_id=request_id, attempt=attempt,
                         status=admitted.get("status", "EXECUTION_FAILURE"),
                         failure_class=None if admitted.get("ok") else "admission",
                         reason=admitted.get("reason"),
                         finding=(admitted.get("finding") if admitted.get("ok") else None),
                         token_accounting=accounting)
        admitted["requests"] = request_count
        return admitted
    return {"status": "EXECUTION_FAILURE", "reason": "format repair exhausted",
            "requests": request_count}


def _worker_provider() -> tuple[str, str]:
    modules = list(sys.modules.values())
    for name in ("probes.react_repair", "react_repair", "__main__"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "API", None):
            return str(module.API), str(getattr(module, "KEY", ""))
    for module in modules:
        if getattr(module, "API", None):
            return str(module.API), str(getattr(module, "KEY", ""))
    # Fail closed: the operator must use the worker's own model endpoint. A silent
    # fallback to a hosted provider would change the model under study.
    raise RuntimeError("BIND_QUANTITY: worker provider (module API) not found")


def make_capture_sender(capture_factory: Callable[[], object]) -> Callable[[dict], object]:
    """Adapt the worker's RequestCapture.send path without importing the worker."""
    endpoint, key = _worker_provider()

    def response_usage(capture: object) -> dict:
        """Recover backend usage already persisted by RequestCapture."""
        request_event_id = getattr(capture, "response_event_id", None)
        journal = getattr(capture, "journal", None)
        events = getattr(journal, "events", None)
        candidates = list(events) if isinstance(events, Sequence) else []
        path = getattr(journal, "path", None)
        if path is not None:
            try:
                with open(path, encoding="utf-8") as handle:
                    candidates.extend(json.loads(line) for line in handle if line.strip())
            except (OSError, TypeError, ValueError):
                pass
        for event in reversed(candidates):
            if not isinstance(event, Mapping) or event.get("event_id") != request_event_id:
                continue
            response = event.get("response")
            usage = response.get("usage") if isinstance(response, Mapping) else None
            if isinstance(usage, Mapping):
                return {"usage": dict(usage)}
        return {}

    def send(body: dict) -> object:
        slot = live_slot_id()
        if slot is None:
            raise BindQuantityUnavailable(
                "BIND_QUANTITY live slot unavailable; measurement skipped")
        body = dict(body)
        # The actor's ask_tools() path adds this field, but BIND_QUANTITY uses
        # its own bounded sender and therefore has to carry the explicit slot
        # across this seam too.  Keep the provider/model request otherwise
        # unchanged.
        from kernel_prompt_cache import configure_pinned_slot
        configure_pinned_slot(body, slot)
        capture = capture_factory()
        from kernel_request_kind import bind_auxiliary
        bind_auxiliary(capture, "bind_quantity")
        choice = capture.send(body, endpoint, key)
        return choice, {"request_id": getattr(capture, "request_id", None),
                        "response_event_id": getattr(capture, "response_event_id", None),
                        **response_usage(capture)}

    return send


def render_quantity_findings(frame: Mapping[str, object], cap: int = FINDINGS_TOKEN_CAP) -> list[str]:
    """Render only an active adopted finding, in a bounded non-imperative block."""
    candidates = frame.get("quantity_bindings")
    if candidates is None and isinstance(frame.get("kernel_findings"), Mapping):
        candidates = frame["kernel_findings"].get("quantity_bindings")
    finding = None
    if isinstance(candidates, Mapping):
        finding = candidates if is_active_binding_finding(candidates) else None
    elif isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        finding = next((item for item in candidates if is_active_binding_finding(item)), None)
    if not isinstance(finding, Mapping):
        return []
    heading = ("YOUR DERIVED QUANTITY BINDINGS (previously derived and adopted; applicable "
               "to the current basis; no recorded contradiction)")
    lines = [heading]
    bindings = finding.get("bindings", [])
    if isinstance(bindings, Mapping):
        bindings = list(bindings.values())
    for binding in bindings if isinstance(bindings, Sequence) else []:
        if not isinstance(binding, Mapping):
            continue
        units = _format_units(binding.get("units", {}))
        lines.append("BINDING: " + str(binding.get("symbol", "UNKNOWN"))
                     + "; meaning=" + str(binding.get("meaning", "UNKNOWN"))
                     + "; semantic_kind=" + str(binding.get("semantic_kind", "UNKNOWN"))
                     + "; units=" + units + "; scale=" + str(binding.get("scale", "UNKNOWN"))
                     + "; clock=" + str(binding.get("clock", "UNKNOWN")))
    outputs = finding.get("requested_outputs", [])
    if isinstance(outputs, Mapping):
        outputs = [outputs]
    for output in outputs if isinstance(outputs, Sequence) else []:
        if isinstance(output, Mapping):
            lines.append("REQUESTED OUTPUT: " + str(output.get("name", "UNKNOWN"))
                         + "; units=" + _format_units(output.get("units", {})))
    expressions = finding.get("expressions", [])
    if isinstance(expressions, Mapping):
        expressions = [expressions]
    for expression in expressions if isinstance(expressions, Sequence) else []:
        if isinstance(expression, Mapping):
            target = str(expression.get("requested_output", "UNKNOWN"))
            output = next((item for item in outputs if isinstance(item, Mapping)
                           and item.get("name") == target), {})
            lines.append("EXPRESSION: " + target + " = " + str(expression.get("expression", "UNKNOWN"))
                         + "; units=" + _format_units(output.get("units", {})
                                                         if isinstance(output, Mapping) else {}))
    witness = finding.get("witness")
    if isinstance(witness, Mapping):
        values = witness.get("values", {})
        lines.append("WITNESS: " + json.dumps(values, ensure_ascii=False, sort_keys=True,
                                              separators=(",", ":"))
                     + " -> " + str(witness.get("expected_result", "UNKNOWN"))
                     + "; units=" + _format_units(witness.get("units", {})))
    lines.append("ORIGIN: model interpretation; authority=model interpretation; "
                 "basis quotes=" + json.dumps(finding.get("basis_quotes", []), ensure_ascii=False)
                 + "; assumptions=" + json.dumps(finding.get("assumptions", []), ensure_ascii=False)
                 + "; dependency_coverage=" + str(finding.get("dependency_coverage", "unknown")))
    if cap <= 0:
        return []
    # The normal 400-token findings budget leaves ample room.  For unusually long
    # meanings/quotes, drop optional provenance detail before required quantity rows.
    if lexical_token_count("\n".join(lines)) > cap:
        lines[-1] = "ORIGIN: model interpretation; authority=model interpretation; " \
                   "dependency_coverage=" + str(finding.get("dependency_coverage", "unknown"))
    if lexical_token_count("\n".join(lines)) > cap:
        lines = [line for line in lines if not line.startswith("BINDING:")]
    if lexical_token_count("\n".join(lines)) > cap:
        lines = [line for line in lines if not line.startswith("ORIGIN:")]
    if lexical_token_count("\n".join(lines)) > cap:
        lines = [heading]
    return lines


def _format_units(units: object) -> str:
    try:
        canonical = parse_units(units)
    except BindQuantityError:
        return "UNKNOWN"
    if not canonical:
        return "dimensionless"
    names = {"token": "tokens", "second": "seconds", "byte": "bytes",
             "request": "requests", "item": "items", "record": "records",
             "percent": "percent", "job": "jobs", "count": "counts"}
    numerator, denominator = [], []
    for key, exponent in canonical.items():
        text = names.get(key, key)
        magnitude = abs(exponent)
        if magnitude != 1:
            text += "^" + str(magnitude)
        (numerator if exponent > 0 else denominator).append(text)
    result = "*".join(numerator) or "1"
    if denominator:
        result += "/" + "*".join(denominator)
    return result


__all__ = [
    "BindQuantityError", "BindQuantityParseError", "FINDINGS_TOKEN_CAP",
    "MAX_REASONING_BUDGET", "MODE_ENV", "OPERATOR", "OUTPUT_CONTRACT",
    "bind_quantity_mode", "build_request_body", "check_dimensional_consistency",
    "compile_packet", "contains_quantity_language", "dimensional_consistency",
    "enabled", "evaluate_expression", "evaluate_units", "extract_quantity_sentences",
    "BindQuantityUnavailable", "is_active_binding_finding", "live_slot_id",
    "make_capture_sender", "parse_json_object", "parse_units",
    "render_quantity_findings", "run_operator", "should_trigger", "unit_scale",
    "validate_response", "validate_witness", "brief_from_messages",
]

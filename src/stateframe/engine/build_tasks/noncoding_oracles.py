"""Mechanical truth signals for the non-coding bench.

The public functions deliberately return a receipt-shaped mapping rather than a
bare boolean.  A corpus row is eligible only when its trivial output fails and
its curated output passes the complete oracle specification.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence


OracleResult = dict[str, object]


def _result(passed: bool, reason: str, evidence: Mapping[str, object]) -> OracleResult:
    return {"pass": passed, "reason": reason, "evidence": dict(evidence)}


def _type_matches(value: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _schema_errors(value: object, schema: Mapping[str, object], path: str = "$",) -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _type_matches(value, expected_type):
        return [f"{path}: expected {expected_type}"]

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path}: {value!r} is outside enum")

    if isinstance(value, Mapping):
        required = schema.get("required", [])
        if isinstance(required, list):
            for name in required:
                if isinstance(name, str) and name not in value:
                    errors.append(f"{path}: missing required field {name!r}")
        properties = schema.get("properties", {})
        property_map = properties if isinstance(properties, Mapping) else {}
        if schema.get("additionalProperties") is False:
            unknown = sorted(str(key) for key in value if key not in property_map)
            errors.extend(f"{path}: unknown field {name!r}" for name in unknown)
        for name, child_schema in property_map.items():
            if name not in value or not isinstance(child_schema, Mapping):
                continue
            errors.extend(_schema_errors(value[name], child_schema, f"{path}.{name}"))

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"{path}: expected at least {min_items} items")
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            errors.append(f"{path}: expected at most {max_items} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                errors.extend(_schema_errors(item, item_schema, f"{path}[{index}]"))

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            errors.append(f"{path}: expected at least {min_length} characters")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            errors.append(f"{path}: does not match required pattern")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path}: must be at least {minimum}")
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path}: must be at most {maximum}")

    return errors


def schema_validity(output: object, schema: Mapping[str, object]) -> OracleResult:
    """Validate JSON-like output against the subset of JSON Schema used by the bench."""
    if not isinstance(schema, Mapping):
        return _result(False, "schema_invalid", {"schema_type": type(schema).__name__})
    errors = _schema_errors(output, schema)
    return _result(
        not errors,
        "schema_valid" if not errors else "schema_invalid",
        {"errors": errors[:20], "error_count": len(errors)},
    )


def _path_parts(path: str) -> list[str]:
    normalized = path[2:] if path.startswith("$.") else path
    return [part for part in normalized.split(".") if part]


def _values_at(value: object, path: str) -> list[tuple[str, object]]:
    """Resolve dotted paths with a single ``[*]`` list wildcard."""
    current: list[tuple[str, object]] = [("$", value)]
    for part in _path_parts(path):
        wildcard = part.endswith("[*]")
        name = part[:-3] if wildcard else part
        next_values: list[tuple[str, object]] = []
        for current_path, current_value in current:
            if not isinstance(current_value, Mapping) or name not in current_value:
                continue
            child = current_value[name]
            child_path = f"{current_path}.{name}"
            if wildcard:
                if isinstance(child, list):
                    next_values.extend(
                        (f"{child_path}[{index}]", item)
                        for index, item in enumerate(child)
                    )
            else:
                next_values.append((child_path, child))
        current = next_values
    return current


def closed_set_membership(
    value: object,
    allowed: Iterable[object],
    *,
    field: str = "value",
) -> OracleResult:
    """Require an exact, case-sensitive label match against ``allowed``.

    ``value`` may be a scalar or a mapping.  For a mapping, ``field`` is a dotted
    path and may contain ``[*]``; every resolved value must be allowed.
    """
    allowed_values = list(allowed)
    if isinstance(value, Mapping) and field != "value":
        resolved = _values_at(value, field)
        values = [item for _path, item in resolved]
        paths = [path for path, _item in resolved]
    else:
        values = [value]
        paths = [field]
    invalid = [item for item in values if item not in allowed_values]
    passed = bool(values) and not invalid
    return _result(
        passed,
        "closed_set_match" if passed else "closed_set_mismatch",
        {
            "field": field,
            "paths": paths,
            "values": values,
            "invalid": invalid,
            "allowed": allowed_values,
        },
    )


def _source_text(source: object) -> str:
    """Return raw scalar text plus canonical JSON so exact spans remain checkable."""
    fragments: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, str):
            fragments.append(value)
        elif isinstance(value, Mapping):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(source)
    try:
        fragments.append(json.dumps(source, ensure_ascii=False, sort_keys=True, default=str))
    except (TypeError, ValueError):
        fragments.append(str(source))
    return "\n".join(fragments)


def _citation_map(output: object) -> dict[str, str]:
    if not isinstance(output, Mapping):
        return {}
    raw = output.get("evidence")
    if not isinstance(raw, list):
        raw = output.get("citations")
    if not isinstance(raw, list):
        return {}
    citations: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        field = item.get("field")
        quote = item.get("quote")
        if isinstance(field, str) and isinstance(quote, str):
            citations[field] = quote
    return citations


def _grounding_declarations(
    claim_paths: Sequence[str | Mapping[str, object]],
) -> list[tuple[str, str]]:
    """Normalize legacy string paths and explicit per-path grounding modes."""
    declarations: list[tuple[str, str]] = []
    for declaration in claim_paths:
        if isinstance(declaration, str):
            declarations.append((declaration, "span"))
            continue
        if not isinstance(declaration, Mapping):
            continue
        path = declaration.get("path")
        mode = declaration.get("mode", "span")
        if isinstance(path, str) and mode in {"span", "atoms"}:
            declarations.append((path, str(mode)))
    return declarations


_ISO_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})\b"
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.-])\d+(?:\.\d+)?%?(?![A-Za-z0-9_.-])")
_TOKEN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_./:-]*[A-Za-z0-9])?")
_BOOLEAN_RE = re.compile(r"\b(?:true|false)\b", re.IGNORECASE)
_IDENTIFIER_SHAPE_CHARS = frozenset("_-./:")


def _source_token_fragments(source: object) -> list[str]:
    fragments: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, str):
            fragments.append(value)
        elif isinstance(value, bool):
            fragments.append("true" if value else "false")
        elif isinstance(value, (int, float)):
            fragments.append(str(value))
        elif isinstance(value, Mapping):
            for key, child in value.items():
                if isinstance(key, str):
                    fragments.append(key)
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(source)
    return fragments


def _input_token_vocabulary(source: object) -> set[str]:
    """Return case-sensitive tokens present in input fields and scalar values."""
    vocabulary: set[str] = set()
    for fragment in _source_token_fragments(source):
        vocabulary.update(match.group(0) for match in _TOKEN_RE.finditer(fragment))
        vocabulary.update(match.group(0) for match in _ISO_TIMESTAMP_RE.finditer(fragment))
        vocabulary.update(match.group(0) for match in _NUMBER_RE.finditer(fragment))
        vocabulary.update(match.group(0) for match in _BOOLEAN_RE.finditer(fragment))
    return vocabulary


def _schema_and_closed_set_atoms(spec: Mapping[str, object] | None) -> set[str]:
    """Collect enum/closed-set literals so atom mode sees non-identifier labels."""
    atoms: set[str] = set()

    def collect_schema(value: object) -> None:
        if not isinstance(value, Mapping):
            return
        enum = value.get("enum")
        if isinstance(enum, list):
            atoms.update(item for item in enum if isinstance(item, str))
        for child in value.values():
            collect_schema(child)

    if spec is None:
        return atoms
    schema_spec = spec.get("schema_validity")
    if isinstance(schema_spec, Mapping):
        collect_schema(schema_spec.get("schema"))
    closed_specs = spec.get("closed_set_membership")
    if isinstance(closed_specs, list):
        for closed_spec in closed_specs:
            if not isinstance(closed_spec, Mapping):
                continue
            allowed = closed_spec.get("allowed")
            if isinstance(allowed, list):
                atoms.update(item for item in allowed if isinstance(item, str))
    return atoms


def _factual_atoms(
    value: str,
    known_literals: set[str],
    input_vocabulary: set[str] | None = None,
) -> list[str]:
    """Extract shaped/factual tokens and exact input-vocabulary words."""
    vocabulary = input_vocabulary or set()
    atoms: set[str] = set()
    for match in _ISO_TIMESTAMP_RE.finditer(value):
        atoms.add(match.group(0))
    masked = _ISO_TIMESTAMP_RE.sub(lambda match: " " * len(match.group(0)), value)
    atoms.update(match.group(0) for match in _NUMBER_RE.finditer(masked))
    for match in _TOKEN_RE.finditer(masked):
        token = match.group(0)
        if _NUMBER_RE.fullmatch(token):
            continue
        if (
            any(character in _IDENTIFIER_SHAPE_CHARS for character in token)
            or any(character.isdigit() for character in token)
            or token in vocabulary
        ):
            atoms.add(token)
    atoms.update(match.group(0) for match in _BOOLEAN_RE.finditer(value))
    for literal in known_literals:
        if re.search(rf"(?<!\w){re.escape(literal)}(?!\w)", value):
            atoms.add(literal)
    return sorted(atoms)


def citation_span_grounding(
    source: object,
    output: object,
    *,
    claim_paths: Sequence[str | Mapping[str, object]],
    threshold: float = 1.0,
    grounding_spec: Mapping[str, object] | None = None,
) -> OracleResult:
    """Ground declared output fields using exact spans or factual atoms.

    Legacy string paths default to ``span``.  Atom mode checks every extracted
    identifier, number, timestamp, enum/closed-set literal, and boolean against
    the source, while still rejecting a narration with no factual atoms.
    """
    source_text = _source_text(source)
    input_vocabulary = _input_token_vocabulary(source)
    citations = _citation_map(output)
    declarations = _grounding_declarations(claim_paths)
    known_literals = _schema_and_closed_set_atoms(grounding_spec)
    claims: list[dict[str, object]] = []
    for expression, mode in declarations:
        resolved_values = _values_at(output, expression)
        if mode == "atoms" and not resolved_values:
            claims.append(
                {
                    "field": expression if expression.startswith("$") else f"$.{expression}",
                    "mode": mode,
                    "quote": "",
                    "atoms": [],
                    "ungrounded_atoms": [],
                    "grounded": False,
                }
            )
            continue
        for path, value in resolved_values:
            if mode == "atoms" and isinstance(value, str):
                atoms = _factual_atoms(value, known_literals, input_vocabulary)
                ungrounded_atoms = [atom for atom in atoms if atom not in input_vocabulary]
                claims.append(
                    {
                        "field": path,
                        "mode": mode,
                        "quote": value,
                        "atoms": atoms,
                        "ungrounded_atoms": ungrounded_atoms,
                        "grounded": bool(atoms) and not ungrounded_atoms,
                    }
                )
                continue
            if mode == "atoms":
                serialized = json.dumps(value, ensure_ascii=False, default=str)
                claims.append(
                    {
                        "field": path,
                        "mode": mode,
                        "quote": serialized,
                        "atoms": [],
                        "ungrounded_atoms": [serialized],
                        "grounded": False,
                    }
                )
                continue
            quote = citations.get(path) or citations.get(expression)
            if quote is None:
                if isinstance(value, str):
                    quote = value
                elif value is None:
                    quote = "null"
                else:
                    quote = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
            grounded = bool(quote) and quote in source_text
            claims.append({"field": path, "mode": mode, "quote": quote, "grounded": grounded})

    grounded_count = sum(bool(claim["grounded"]) for claim in claims)
    total = len(claims)
    fraction = grounded_count / total if total else 0.0
    passed = total > 0 and fraction >= threshold
    no_factual_atoms = any(
        claim.get("mode") == "atoms" and not claim.get("atoms")
        for claim in claims
    )
    has_atoms_mode = any(mode == "atoms" for _path, mode in declarations)
    failure_reason = (
        "no_factual_atoms"
        if no_factual_atoms
        else "ungrounded_atoms"
        if has_atoms_mode
        else "ungrounded_claims"
    )
    return _result(
        passed,
        "grounded" if passed else failure_reason,
        {
            "grounded": grounded_count,
            "total_claims": total,
            "grounded_fraction": fraction,
            "threshold": threshold,
            "modes": {path: mode for path, mode in declarations},
            "claims": claims[:50],
        },
    )


def _value_at(value: object, path: str) -> object:
    resolved = _values_at(value, path)
    if not resolved:
        return None
    return resolved[0][1]


def _recompute(input_data: object, spec: Mapping[str, object]) -> object:
    operation = spec.get("operation")
    input_path = spec.get("input_path")
    if not isinstance(input_path, str):
        raise ValueError("input_path is required")
    source = _value_at(input_data, input_path)
    if operation == "count":
        if not isinstance(source, list):
            raise ValueError("count input must be a list")
        return len(source)
    if operation == "identity":
        return source
    if operation == "date_from_parts":
        if not isinstance(source, Mapping):
            raise ValueError("date_from_parts input must be an object")
        year, month, day = (source.get(key) for key in ("year", "month", "day"))
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    if operation == "join_ids":
        if not isinstance(source, list):
            raise ValueError("join_ids input must be a list")
        return ",".join(str(item) for item in source)
    raise ValueError(f"unsupported operation: {operation!r}")


def deterministic_recomputation(
    input_data: object,
    output: object,
    field_specs: Sequence[Mapping[str, object]],
) -> OracleResult:
    """Recompute declared counts/dates/ids from input and compare exact output values."""
    checks: list[dict[str, object]] = []
    for spec in field_specs:
        field = spec.get("field")
        if not isinstance(field, str):
            checks.append({"field": field, "pass": False, "reason": "field_missing"})
            continue
        actual = _value_at(output, field)
        try:
            expected = _recompute(input_data, spec)
            matches = actual == expected
            reason = "recomputed_match" if matches else "recomputed_mismatch"
        except (TypeError, ValueError, KeyError) as exc:
            expected = None
            matches = False
            reason = f"recompute_error:{type(exc).__name__}"
        checks.append(
            {
                "field": field,
                "operation": spec.get("operation"),
                "actual": actual,
                "expected": expected,
                "pass": matches,
                "reason": reason,
            }
        )
    passed = bool(checks) and all(bool(check["pass"]) for check in checks)
    return _result(
        passed,
        "recomputed" if passed else "recomputation_mismatch",
        {"checks": checks},
    )


def evaluate_oracle_spec(
    source: object,
    output: object,
    spec: Mapping[str, object],
) -> dict[str, OracleResult]:
    """Run all configured adapters in a stable order."""
    results: dict[str, OracleResult] = {}
    schema_spec = spec.get("schema_validity")
    if isinstance(schema_spec, Mapping):
        schema = schema_spec.get("schema")
        results["schema_validity"] = schema_validity(output, schema) if isinstance(schema, Mapping) else _result(
            False, "schema_missing", {"schema": schema}
        )

    grounding_spec = spec.get("citation_span_grounding")
    if isinstance(grounding_spec, Mapping):
        claim_paths = grounding_spec.get("claim_paths", [])
        threshold = grounding_spec.get("threshold", 1.0)
        results["citation_span_grounding"] = citation_span_grounding(
            source,
            output,
            claim_paths=[
                path for path in claim_paths if isinstance(path, (str, Mapping))
            ]
            if isinstance(claim_paths, list)
            else [],
            threshold=float(threshold),
            grounding_spec=spec,
        )

    closed_specs = spec.get("closed_set_membership", [])
    if isinstance(closed_specs, list):
        for index, closed_spec in enumerate(closed_specs):
            if not isinstance(closed_spec, Mapping):
                results[f"closed_set_membership[{index}]"] = _result(
                    False, "closed_set_spec_invalid", {"spec": closed_spec}
                )
                continue
            field = closed_spec.get("field", "value")
            allowed = closed_spec.get("allowed", [])
            results[f"closed_set_membership[{index}]"] = closed_set_membership(
                output,
                allowed if isinstance(allowed, list) else [],
                field=str(field),
            )

    recomputation = spec.get("deterministic_recomputation")
    if isinstance(recomputation, Mapping):
        fields = recomputation.get("fields", [])
        results["deterministic_recomputation"] = deterministic_recomputation(
            source,
            output,
            [field for field in fields if isinstance(field, Mapping)] if isinstance(fields, list) else [],
        )
    return results


def all_passed(results: Mapping[str, OracleResult]) -> bool:
    """Return the aggregate verdict for already-computed oracle receipts."""
    return bool(results) and all(result.get("pass") is True for result in results.values())


__all__ = [
    "OracleResult",
    "all_passed",
    "citation_span_grounding",
    "closed_set_membership",
    "deterministic_recomputation",
    "evaluate_oracle_spec",
    "schema_validity",
]

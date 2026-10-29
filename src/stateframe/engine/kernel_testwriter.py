"""Host-owned typed TestWriter adapter for the live IDEALPATH S3 rail.

The public contract mirrors the screened ``typed_test_writer`` method.  It is
an admission/front-end layer only: executable checks are lowered to the
existing REPROSPEC compiler and runner, so parent PASS and environment errors
retain their existing non-reproduction semantics.
"""
from __future__ import annotations

import copy
import ast
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


SCHEMA = "stateframe-testwriter/1"
TOOL_NAME = "typed_test_writer"
FIXTURE_RECIPES = ("DIRECT_CALL", "ASYNC_CALL", "FIXTURE_SETUP", "CALL_SPY", "CLOCK_BOUNDARY")
INVOCATIONS = ("CALL", "AWAIT")
OBSERVATIONS = ("RETURN", "EXCEPTION", "CALL_SEQUENCE", "BEFORE_AFTER", "EFFECT_COUNT")
SETUP_RECIPES = ("NO_SETUP", "PUBLIC_FIXTURE", "INJECTED_DEPENDENCY", "ASYNC_LIFECYCLE")
MAX_ASSERTION_CHARS = 240
_FORBIDDEN = re.compile(
    r"(?:grader|evaluator|hidden\s+fix|reference\s+patch|private\s+test|\b(?:import|def|assert)\s+)",
    re.IGNORECASE,
)


class TestWriterError(ValueError):
    """Raised for host-side TestWriter contract or lowering failures."""


def _items(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _text(value: object, *, limit: int = 2000) -> str:
    if not isinstance(value, str):
        return ""
    result = value.replace("\x00", " ").strip()
    return result if len(result) <= limit else ""


def _json_value(value: object) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _target_menu(packet: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    from kernel_reprospec import symbol_target_menu

    return symbol_target_menu(packet)


def _target_menu_for_owner(
        packet: Mapping[str, object], owner: object | None,
) -> dict[str, Mapping[str, object]]:
    """Keep the live schema aligned with the host's S2 localization target."""
    targets = _target_menu(packet)
    context = getattr(owner, "s2_force_target", {}) if owner is not None else {}
    if not isinstance(context, Mapping):
        return targets
    path = str(context.get("path", "") or "").replace("\\", "/").lstrip("./")
    symbol = str(context.get("symbol", "") or "").strip()
    if not path and not symbol:
        return targets
    narrowed = {
        target_id: target
        for target_id, target in targets.items()
        if isinstance(target, Mapping)
        and (not path or str(target.get("source_path", "")).replace("\\", "/").lstrip("./") in {
            path, path.removeprefix("backend/"), "backend/" + path,
        })
        and (not symbol or target.get("name") == symbol)
    }
    if narrowed:
        return narrowed
    # A path-only localization may not carry the exact symbol; retain all
    # symbols from that host file rather than advertising an impossible menu.
    by_path = {
        target_id: target
        for target_id, target in targets.items()
        if isinstance(target, Mapping)
        and str(target.get("source_path", "")).replace("\\", "/").lstrip("./") in {
            path, path.removeprefix("backend/"), "backend/" + path,
        }
    }
    return by_path or targets


def _signature_required_parameters(signature: object) -> tuple[str, ...] | None:
    if not isinstance(signature, str) or not signature.strip():
        return None
    try:
        function = ast.parse(f"def {signature}: pass").body[0]
    except SyntaxError:
        if signature.strip().startswith("class "):
            return ()
        return None
    if not isinstance(function, ast.FunctionDef):
        return None
    positional = [*function.args.posonlyargs, *function.args.args]
    positional_defaults = [None] * (len(positional) - len(function.args.defaults))
    positional_defaults.extend(function.args.defaults)
    required = [arg.arg for arg, default in zip(positional, positional_defaults) if default is None]
    required.extend(
        arg.arg for arg, default in zip(function.args.kwonlyargs, function.args.kw_defaults)
        if default is None
    )
    return tuple(required)


def _target_required_parameters(target: Mapping[str, object]) -> tuple[str, ...] | None:
    declared = target.get("required_parameters")
    if isinstance(declared, list) and all(isinstance(item, str) and item.strip() for item in declared):
        return tuple(dict.fromkeys(str(item) for item in declared))
    return _signature_required_parameters(target.get("signature"))


def _binding_ref(value: object, *, available: set[str], field: str) -> dict[str, object] | None:
    if not isinstance(value, Mapping) or set(value) != {"source", "path"}:
        return None
    source = value.get("source")
    path = value.get("path")
    if not isinstance(source, str) or source not in available or source == "$last":
        raise TestWriterError(f"{field}.source must reference an earlier host output")
    if not isinstance(path, list) or len(path) > 16 or any(
            isinstance(item, bool) or not isinstance(item, (str, int)) for item in path):
        raise TestWriterError(f"{field}.path must be a bounded string/integer list")
    return {"source": source, "path": list(path)}


def _validate_bindings(
        raw: object, *, target: Mapping[str, object], recipe: str,
        fixture_ids: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if raw is None:
        return None, {"code": "BINDINGS_REQUIRED", "field": "bindings",
                      "detail": "fixture, argument, and observation bindings must be explicit"}
    if not isinstance(raw, Mapping) or set(raw) - {
            "fixture_id", "argument_bindings", "observation", "output_refs"}:
        return None, {"code": "BINDINGS_SCHEMA", "field": "bindings",
                      "detail": "bindings is a closed typed object"}
    normalized: dict[str, object] = {"argument_bindings": {}, "observation": None}
    fixture_id = raw.get("fixture_id")
    if recipe == "FIXTURE_SETUP":
        if not isinstance(fixture_id, str) or fixture_id not in fixture_ids:
            return None, {"code": "FIXTURE_BINDING_REQUIRED", "field": "bindings.fixture_id",
                          "detail": "fixture_id must be selected from the host fixture menu",
                          "allowed": sorted(fixture_ids)}
        normalized["fixture_id"] = fixture_id
    elif fixture_id is not None:
        return None, {"code": "UNSUPPORTED_BINDING", "field": "bindings.fixture_id",
                      "detail": "fixture_id requires FIXTURE_SETUP"}
    argument_bindings = raw.get("argument_bindings", {})
    if not isinstance(argument_bindings, Mapping):
        return None, {"code": "BINDINGS_SCHEMA", "field": "bindings.argument_bindings",
                      "detail": "argument_bindings must be an object"}
    required = _target_required_parameters(target)
    if required is None:
        return None, {"code": "TARGET_SIGNATURE_UNAVAILABLE", "field": "target_id",
                      "detail": "host target signature is required for argument preflight"}
    unknown = sorted(str(key) for key in argument_bindings if str(key) not in set(required))
    if unknown:
        return None, {"code": "ARGUMENT_BINDING_UNKNOWN", "field": "bindings.argument_bindings",
                      "detail": "argument binding names are not target parameters", "allowed": list(required)}
    missing = [name for name in required if name not in argument_bindings]
    if missing:
        return None, {"code": "BINDING_REQUIRED", "field": "bindings.argument_bindings",
                      "detail": f"required target parameters are unresolved: {', '.join(missing)}"}
    available: set[str] = set()
    if recipe == "FIXTURE_SETUP":
        available.add("$1")
    converted: dict[str, object] = {}
    for name, value in argument_bindings.items():
        try:
            ref = _binding_ref(value, available=available,
                               field=f"bindings.argument_bindings.{name}") if isinstance(value, Mapping) else None
        except TestWriterError as error:
            return None, {"code": "ARGUMENT_BINDING", "field": f"bindings.argument_bindings.{name}",
                          "detail": str(error)}
        if isinstance(value, Mapping) and ref is None:
            return None, {"code": "BINDING_SCHEMA", "field": f"bindings.argument_bindings.{name}",
                          "detail": "output references require source and path"}
        if ref is not None:
            converted[str(name)] = {"$ref": ref["source"], "path": ref["path"]}
        elif not _json_value(value):
            return None, {"code": "BINDING_VALUE", "field": f"bindings.argument_bindings.{name}",
                          "detail": "argument values must be JSON data"}
        else:
            converted[str(name)] = copy.deepcopy(value)
    normalized["argument_bindings"] = converted
    observation = raw.get("observation")
    if observation is not None:
        try:
            normalized["observation"] = _binding_ref(
                observation, available=available | {"$%d" % (2 if recipe == "FIXTURE_SETUP" else 1)},
                field="bindings.observation",
            )
        except TestWriterError as error:
            return None, {"code": "OBSERVATION_BINDING", "field": "bindings.observation",
                          "detail": str(error)}
        if normalized["observation"] is None:
            return None, {"code": "OBSERVATION_BINDING", "field": "bindings.observation",
                          "detail": "observation requires source and path"}
    else:
        return None, {"code": "OBSERVATION_BINDING_REQUIRED", "field": "bindings.observation",
                      "detail": "observation source and path must be explicit"}
    output_refs = raw.get("output_refs", {})
    if output_refs:
        if not isinstance(output_refs, Mapping):
            return None, {"code": "BINDINGS_SCHEMA", "field": "bindings.output_refs",
                          "detail": "output_refs must be an object"}
        normalized_outputs: dict[str, object] = {}
        for name, value in output_refs.items():
            try:
                ref = _binding_ref(value, available=available | {"$%d" % (2 if recipe == "FIXTURE_SETUP" else 1)},
                                   field=f"bindings.output_refs.{name}")
            except TestWriterError as error:
                return None, {"code": "OUTPUT_BINDING", "field": f"bindings.output_refs.{name}",
                              "detail": str(error)}
            if ref is None:
                return None, {"code": "OUTPUT_BINDING", "field": f"bindings.output_refs.{name}",
                              "detail": "output reference requires source and path"}
            normalized_outputs[str(name)] = ref
        normalized["output_refs"] = normalized_outputs
    return normalized, None


def ticket_span_menu(owner: object) -> dict[str, str]:
    """Return exact public ticket quotes keyed by their canonical span IDs."""
    draft = getattr(owner, "host_draft", None)
    spans = draft.get("spans", ()) if isinstance(draft, Mapping) else ()
    result: dict[str, str] = {}
    for span in _items(spans):
        if not isinstance(span, Mapping):
            continue
        ident = _text(span.get("id"), limit=120)
        quote = _text(span.get("quote"), limit=1200)
        if ident and quote:
            result[ident] = quote
    if result:
        return result
    for value in getattr(owner, "ticket_spans", ()) or ():
        text = _text(value, limit=1200)
        if text:
            result[text] = text
    return result


def _tool_schema(properties: Mapping[str, object]) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Submit one host-validated repository-native typed check proposal; "
                "provide no Python or private-evidence claims."
            ),
            "parameters": {
                "type": "object",
                "properties": dict(properties),
                "required": [
                    "target_id", "ticket_span", "fixture_recipe", "invocation",
                    "observations", "setup_recipe_ids", "assertion",
                ],
                "additionalProperties": False,
            },
        },
    }


def typed_test_writer_schema(
        packet: Mapping[str, object], *, owner: object | None = None,
        ticket_spans: Mapping[str, str] | Sequence[str] = (),
) -> dict[str, object] | None:
    """Render the exact host menus used by the live TestWriter tool.

    Returns ``None`` (drop the tool) when the target or ticket-span menu is
    empty: ``enum: []`` makes llama.cpp answer HTTP 500 (audit 2026-10-05 #1),
    and a tool with no admissible target cannot be called validly anyway.
    """
    targets = sorted(_target_menu_for_owner(packet, owner))
    if isinstance(ticket_spans, Mapping):
        spans = list(ticket_spans.values())
    else:
        spans = [item for item in ticket_spans if isinstance(item, str) and item.strip()]
    if owner is not None and not spans:
        spans = list(ticket_span_menu(owner).values())
    if not targets or not spans:
        return None
    return _tool_schema({
        "target_id": {"type": "string", "enum": targets},
        "ticket_span": {"type": "string", "enum": list(dict.fromkeys(spans))},
        "fixture_recipe": {"type": "string", "enum": list(FIXTURE_RECIPES)},
        "invocation": {"type": "string", "enum": list(INVOCATIONS)},
        "observations": {
            "type": "array", "items": {"type": "string", "enum": list(OBSERVATIONS)},
            "minItems": 1, "maxItems": len(OBSERVATIONS),
        },
                "setup_recipe_ids": {
                    "type": "array", "items": {"type": "string", "enum": list(SETUP_RECIPES)},
                    "maxItems": 4,
                },
                "bindings": {
                    "type": "object",
                    "properties": {
                        "fixture_id": {"type": "string"},
                        "argument_bindings": {"type": "object"},
                        "observation": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string", "pattern": r"^\$[1-9][0-9]*$"},
                                "path": {"type": "array", "maxItems": 16},
                            },
                            "required": ["source", "path"],
                            "additionalProperties": False,
                        },
                        "output_refs": {"type": "object"},
                    },
                    "additionalProperties": False,
                },
                "assertion": {"type": "string", "minLength": 1, "maxLength": MAX_ASSERTION_CHARS},
    })


def validate_test_writer(
        arguments: object, packet: Mapping[str, object], *, owner: object | None = None,
        ticket_spans: Mapping[str, str] | Sequence[str] = (),
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Validate the method contract without accepting model-authored code."""
    if not isinstance(arguments, Mapping):
        return None, {"type": "typed_test_writer_error", "field": "arguments", "message": "arguments must be an object"}
    required = {
        "target_id", "ticket_span", "fixture_recipe", "invocation",
        "observations", "setup_recipe_ids", "assertion",
    }
    if not required.issubset(set(arguments)) or not set(arguments).issubset(required | {"bindings"}):
        return None, {"type": "typed_test_writer_error", "field": "arguments", "message": "TestWriter fields are host-owned and closed"}
    targets = _target_menu(packet)
    target_id = arguments.get("target_id")
    if not isinstance(target_id, str) or target_id not in targets:
        return None, {"type": "typed_test_writer_error", "field": "target_id", "message": "target_id is not in the host S2 target menu", "allowed": sorted(targets)}
    spans = dict(ticket_spans) if isinstance(ticket_spans, Mapping) else {
        str(item): str(item) for item in ticket_spans if isinstance(item, str) and item.strip()
    }
    if owner is not None and not spans:
        spans = ticket_span_menu(owner)
    ticket_span = arguments.get("ticket_span")
    if not isinstance(ticket_span, str) or ticket_span not in set(spans.values()):
        return None, {"type": "typed_test_writer_error", "field": "ticket_span", "message": "ticket_span is not copied from the public ticket", "allowed": list(spans.values())}
    recipe = arguments.get("fixture_recipe")
    invocation = arguments.get("invocation")
    if recipe not in FIXTURE_RECIPES:
        return None, {"type": "typed_test_writer_error", "field": "fixture_recipe", "message": "fixture_recipe is not registered", "allowed": list(FIXTURE_RECIPES)}
    if invocation not in INVOCATIONS:
        return None, {"type": "typed_test_writer_error", "field": "invocation", "message": "invocation is not registered", "allowed": list(INVOCATIONS)}
    if recipe == "ASYNC_CALL" and invocation != "AWAIT":
        return None, {"type": "typed_test_writer_error", "field": "invocation", "message": "ASYNC_CALL requires AWAIT"}
    observations = arguments.get("observations")
    if not isinstance(observations, list) or not observations or any(item not in OBSERVATIONS for item in observations):
        return None, {"type": "typed_test_writer_error", "field": "observations", "message": "observations must contain registered values"}
    if len(set(observations)) != len(observations):
        return None, {"type": "typed_test_writer_error", "field": "observations", "message": "observations must be unique"}
    setups = arguments.get("setup_recipe_ids")
    if not isinstance(setups, list) or any(item not in SETUP_RECIPES for item in setups):
        return None, {"type": "typed_test_writer_error", "field": "setup_recipe_ids", "message": "setup_recipe_ids contains an unknown recipe"}
    if recipe == "DIRECT_CALL" and setups not in ([], ["NO_SETUP"]):
        return None, {"type": "typed_test_writer_error", "field": "setup_recipe_ids", "message": "DIRECT_CALL may not claim an unrelated setup recipe"}
    assertion = arguments.get("assertion")
    if not isinstance(assertion, str) or not assertion.strip() or len(assertion.strip()) > MAX_ASSERTION_CHARS:
        return None, {"type": "typed_test_writer_error", "field": "assertion", "message": "assertion must be bounded plain-language text"}
    if _FORBIDDEN.search(assertion):
        return None, {"type": "typed_test_writer_error", "field": "assertion", "message": "assertion contains code or private-evidence language"}
    bindings, binding_error = _validate_bindings(
        arguments.get("bindings"),
        target=targets[target_id],
        recipe=str(recipe),
        fixture_ids=(
            __import__("kernel_reprospec", fromlist=["symbol_fixture_menu"]).symbol_fixture_menu(packet)
        ),
    )
    if binding_error is not None or bindings is None:
        return None, {
            "type": "typed_test_writer_error",
            "field": str(binding_error.get("field", "bindings")) if isinstance(binding_error, Mapping) else "bindings",
            "message": str(binding_error.get("detail", "bindings are not admissible")) if isinstance(binding_error, Mapping) else "bindings are not admissible",
            "code": binding_error.get("code") if isinstance(binding_error, Mapping) else "BINDINGS_SCHEMA",
        }
    if not all(_json_value(value) for value in arguments.values()):
        return None, {"type": "typed_test_writer_error", "field": "arguments", "message": "typed check is not serializable"}
    return {
        "target_id": target_id,
        "ticket_span": ticket_span,
        "ticket_span_id": next((ident for ident, quote in spans.items() if quote == ticket_span), ticket_span),
        "fixture_recipe": recipe,
        "invocation": invocation,
        "observations": list(observations),
        "setup_recipe_ids": list(setups),
        "assertion": assertion.strip(),
        "target": copy.deepcopy(dict(targets[target_id])),
        "bindings": bindings,
    }, None


def _row_for_span(owner: object, span_id: str) -> Mapping[str, object] | None:
    rows = getattr(owner, "rows", {})
    matches = [
        row for row in rows.values()
        if isinstance(row, Mapping)
        and span_id in {str(item) for item in _items(row.get("ticket_refs"))}
    ] if isinstance(rows, Mapping) else []
    if len(matches) == 1:
        row = matches[0]
        if isinstance(row.get("spine"), Mapping):
            return row
        row_id = str(row.get("row_id", ""))
        for ledger_name in ("requirement_ledger", "s1_spine_ledger", "spine_ledger"):
            ledger = getattr(owner, ledger_name, None)
            requirements = ledger.get("requirements") if isinstance(ledger, Mapping) else ()
            requirement = next(
                (
                    item for item in _items(requirements)
                    if isinstance(item, Mapping) and str(item.get("id", "")) == row_id
                ),
                None,
            )
            if isinstance(requirement, Mapping) and isinstance(requirement.get("spine"), Mapping):
                enriched = copy.deepcopy(dict(row))
                enriched["spine"] = copy.deepcopy(requirement["spine"])
                return enriched
        return row
    return None


def _predicate_expected(row: Mapping[str, object], observation: str) -> tuple[dict[str, object] | None, str | None]:
    spine = row.get("spine")
    if not isinstance(spine, Mapping) or spine.get("semantic_admission") != "ADMITTED":
        return None, "S1SPINE semantic admission is required"
    observables = [item for item in _items(spine.get("observables")) if isinstance(item, Mapping)]
    if not observables:
        return None, "S1SPINE has no admitted observable"
    predicates = [item for item in _items(observables[0].get("predicates")) if isinstance(item, Mapping)]
    if not predicates:
        return None, "S1SPINE has no executable predicate"
    predicate = predicates[0]
    rhs = predicate.get("rhs")
    relation = str(predicate.get("relation", ""))
    if not isinstance(rhs, Mapping) or rhs.get("kind") != "LITERAL":
        return None, "S1SPINE predicate is not literal-bound"
    if relation == "EQ" and observation != "EXCEPTION":
        return {"relation": "equals", "value": copy.deepcopy(rhs.get("value"))}, None
    if relation == "CONTAINS" and observation != "EXCEPTION":
        return {"relation": "contains", "value": copy.deepcopy(rhs.get("value"))}, None
    if relation == "RAISES" and observation == "EXCEPTION":
        exception_type = rhs.get("value")
        if isinstance(exception_type, str) and exception_type in {
                "AssertionError", "AttributeError", "KeyError", "LookupError",
                "ModuleNotFoundError", "NameError", "RuntimeError", "TimeoutError",
                "TypeError", "ValueError"}:
            return {"relation": "raises", "exception_type": exception_type}, None
        return None, "RAISES oracle must name a supported public exception type"
    if relation == "ABSENT" and observation in {"CALL_SEQUENCE", "EFFECT_COUNT"}:
        return {"relation": "not_called", "value": copy.deepcopy(rhs.get("value"))}, None
    return None, f"S1SPINE relation {relation or '<missing>'} is unsupported for {observation}"


def lower_to_reprospec(
        normalized: Mapping[str, object], owner: object, packet: Mapping[str, object],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Lower one admitted writer proposal to the existing typed REPROSPEC DSL."""
    row = _row_for_span(owner, str(normalized["ticket_span_id"]))
    if row is None:
        return None, {"code": "REQUIREMENT_ROW_REQUIRED", "detail": "ticket span must identify exactly one S1SPINE row"}
    recipe = str(normalized["fixture_recipe"])
    invocation = str(normalized["invocation"])
    observations = [str(item) for item in _items(normalized.get("observations"))]
    observation = next((item for item in (
        "EXCEPTION", "CALL_SEQUENCE", "BEFORE_AFTER", "EFFECT_COUNT", "RETURN",
    ) if item in observations), None)
    if observation is None:
        return None, {"code": "UNSUPPORTED_OBSERVATION", "detail": "the typed observation menu contains no executable observation"}
    target = normalized.get("target")
    if not isinstance(target, Mapping):
        return None, {"code": "TARGET_MENU_INCOMPLETE", "detail": "host target entry is incomplete"}
    kind = str(target.get("kind", ""))
    setup_kind = "construct" if kind == "class" else "call" if kind == "function" else ""
    if not setup_kind:
        return None, {"code": "UNSUPPORTED_TARGET", "detail": "target must be a host function or class"}
    expected, expected_error = _predicate_expected(row, observation)
    if expected_error is not None or expected is None:
        return None, {"code": "MISSING_ORACLE", "detail": expected_error or "no executable oracle"}
    span_id = str(normalized["ticket_span_id"])
    requirement_id = str(row.get("row_id", ""))
    if not requirement_id:
        return None, {"code": "REQUIREMENT_ROW_REQUIRED", "detail": "S1SPINE row has no stable requirement identity"}
    setup: list[dict[str, object]] = []
    setup_ids = {str(item) for item in _items(normalized.get("setup_recipe_ids"))}
    bindings = normalized.get("bindings")
    bindings = bindings if isinstance(bindings, Mapping) else {}
    if recipe == "FIXTURE_SETUP":
        from kernel_reprospec import symbol_fixture_menu

        fixtures = symbol_fixture_menu(packet)
        if "PUBLIC_FIXTURE" not in setup_ids:
            return None, {"code": "FIXTURE_SETUP_RECIPE_REQUIRED", "detail": "FIXTURE_SETUP requires PUBLIC_FIXTURE"}
        fixture_id = bindings.get("fixture_id")
        if not isinstance(fixture_id, str) or fixture_id not in fixtures:
            return None, {"code": "FIXTURE_BINDING_REQUIRED", "detail": "FIXTURE_SETUP requires an explicit host fixture binding"}
        setup.append({"kind": "fixture", "fixture": fixture_id})
    if recipe == "ASYNC_CALL" or (recipe == "FIXTURE_SETUP" and invocation == "AWAIT"):
        call_kind = "async_call"
    elif recipe == "CALL_SPY":
        call_kind = "call_spy"
    elif recipe == "CLOCK_BOUNDARY":
        call_kind = "clock_boundary"
    else:
        call_kind = setup_kind
    argument_bindings = bindings.get("argument_bindings", {})
    if not isinstance(argument_bindings, Mapping):
        return None, {"code": "BINDINGS_SCHEMA", "detail": "argument bindings are not an object"}
    setup.append({
        "kind": call_kind,
        "target": str(normalized["target_id"]),
        "args": copy.deepcopy(dict(argument_bindings)),
    })
    if recipe == "ASYNC_CALL" and invocation != "AWAIT":
        return None, {"code": "ASYNC_INVOCATION_REQUIRED", "detail": "ASYNC_CALL requires AWAIT"}
    if recipe != "ASYNC_CALL" and invocation not in {"CALL", "AWAIT"}:
        return None, {"code": "INVOCATION_REQUIRED", "detail": "invocation is not registered"}
    observation_kind = {
        "CALL_SEQUENCE": "recorded_calls",
        "BEFORE_AFTER": "before_after",
        "EFFECT_COUNT": "effect_count",
    }.get(observation, "result_path")
    scenario = {
        "ticket_span": span_id,
        "inputs": "host-bounded direct invocation",
        "conditions": str(row.get("trigger", "")),
        "expected_observation": str(row.get("expected_observable", "")),
        "wrong_fix_rejected": str(row.get("wrong_behavior", "")),
    }
    observation_binding = bindings.get("observation")
    if not isinstance(observation_binding, Mapping):
        return None, {"code": "OBSERVATION_BINDING_REQUIRED",
                      "detail": "lowering requires the explicitly bound observation"}
    observation_source = str(observation_binding.get("source", ""))
    observation_path = list(_items(observation_binding.get("path")))
    return {
        "requirement_ref": requirement_id,
        "expected_relation": str(normalized["assertion"]),
        "plausible_wrong_impl": str(row.get("wrong_behavior", "")),
        "distinguishing_scenario": scenario,
        "check_spec": {
            "setup": setup,
            "observation": {"kind": observation_kind, "source": observation_source, "path": observation_path},
            "expected": expected,
        },
    }, None


def _runtime_revision(runtime: object) -> str:
    binding = getattr(runtime, "binding", None)
    try:
        value = binding() if callable(binding) else binding
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        value = None
    if value is None:
        for name in ("candidate_revision", "product_revision", "current_revision"):
            value = getattr(runtime, name, None)
            if value is not None:
                break
    return str(value or "").strip()


def _authored_receipt_binding(
        receipt: Mapping[str, object], *, clause_id: str,
        parent_revision: str, candidate_revision: str,
) -> tuple[dict[str, object], bool]:
    """Require the complete clause/check/revision identity before host binding."""
    freshness = receipt.get("freshness")
    freshness = freshness if isinstance(freshness, Mapping) else {}
    check_hash = receipt.get("check_hash", freshness.get("check_hash"))
    check_id = receipt.get("check_id")
    receipt_clause = receipt.get("clause_id", receipt.get("requirement_id"))
    receipt_parent = receipt.get("parent_revision", freshness.get("parent_revision"))
    receipt_candidate = receipt.get("candidate_revision", freshness.get("product_revision"))
    expected_parent = str(parent_revision or "")
    expected_candidate = str(candidate_revision or "")
    binding = {
        "clause_id": clause_id,
        "check_id": str(check_id or ""),
        "check_hash": str(check_hash or ""),
        "receipt_id": str(receipt.get("receipt_id") or ""),
        "parent_revision": str(receipt_parent or ""),
        "candidate_revision": str(receipt_candidate or ""),
        "selector": str(receipt.get("selector") or receipt.get("test_path") or ""),
        "source": str((receipt.get("provenance") or {}).get("kind", "REPROSPEC")
                      if isinstance(receipt.get("provenance"), Mapping) else "REPROSPEC"),
    }
    complete = all(binding[key] for key in (
        "clause_id", "check_id", "check_hash", "receipt_id",
        "parent_revision", "candidate_revision", "selector",
    )) and binding["parent_revision"] != binding["candidate_revision"]
    complete = complete and str(receipt_clause or "") == clause_id
    complete = complete and binding["parent_revision"] == expected_parent
    complete = complete and binding["candidate_revision"] == expected_candidate
    complete = complete and receipt.get("certifies_clause") is True
    complete = complete and receipt.get("observation_only") is not True
    complete = complete and str(receipt.get("result", receipt.get("status", ""))).upper() \
        in {"PASS", "BEHAVIOR_FIXED", "COMPLETED"}
    return binding, complete


def execute_typed_test_writer(
        runtime: object, arguments: object, *, root: Path,
        packet: Mapping[str, object], runner: object = None,
) -> dict[str, object]:
    """Validate, lower, and execute one TestWriter proposal on REPROSPEC."""
    owner_method = getattr(runtime, "_idealpath_owner", None)
    owner = owner_method() if callable(owner_method) else getattr(runtime, "idealpath_owner", None)
    if owner is None or getattr(owner, "stage", None) != "S3":
        return {
            "schema": "stateframe-idealpath-reprospec/1",
            "tool": TOOL_NAME,
            "status": "REJECTED",
            "outcome": "INVALID_CHECK",
            "code": "S3_REQUIRED",
        }
    normalized, validation_error = validate_test_writer(arguments, packet, owner=owner)
    if validation_error is not None or normalized is None:
        return {
            "schema": "stateframe-idealpath-reprospec/1",
            "tool": TOOL_NAME,
            "status": "REJECTED",
            "outcome": "INVALID_CHECK",
            "code": "TESTWRITER_SPEC_ERROR",
            "error": copy.deepcopy(validation_error),
        }
    lowered, lowering_error = lower_to_reprospec(normalized, owner, packet)
    if lowering_error is not None or lowered is None:
        return {
            "schema": "stateframe-idealpath-reprospec/1",
            "tool": TOOL_NAME,
            "status": "REJECTED",
            "outcome": "INVALID_CHECK",
            "code": str((lowering_error or {}).get("code", "TESTWRITER_LOWERING_ERROR")),
            "detail": str((lowering_error or {}).get("detail", "typed proposal could not be lowered")),
            "testwriter": public_result(normalized),
        }
    from kernel_reprospec import execute_typed_check

    result = execute_typed_check(
        runtime, lowered, root=root, packet=packet,
        runner=runner if callable(runner) else None,
    )
    result = dict(result)
    # REPROSPEC exposes its execution class as ``status``; IDEALPATH's
    # reducer consumes the common ``outcome`` field used by stage receipts.
    # Preserve both so typed TestWriter executions cannot silently skip S3
    # admission or host-force transition handling.
    result.setdefault("outcome", result.get("status"))
    result.update({
        "tool": TOOL_NAME,
        "mechanism": "REPROSPEC",
        "testwriter_schema": SCHEMA,
        "testwriter": public_result(normalized),
        "lowered_requirement_ref": lowered.get("requirement_ref"),
    })
    if str(getattr(owner, "row_role", lambda _row: "REPAIR")(
            lowered.get("requirement_ref"))).upper() == "NEW_BEHAVIOR":
        receipt = result.get("receipt")
        parent_revision = str(getattr(owner, "base_revision", "") or "")
        candidate_revision = _runtime_revision(runtime)
        binding, binding_complete = _authored_receipt_binding(
            receipt if isinstance(receipt, Mapping) else {},
            clause_id=str(lowered.get("requirement_ref")),
            parent_revision=parent_revision, candidate_revision=candidate_revision,
        )
        result["authored_check_binding"] = binding
        result["authored_check_binding_status"] = "ADMITTED" if binding_complete else "UNBOUND"
        record_witness = getattr(owner, "record_authored_witness", None)
        if binding_complete and callable(record_witness):
            result["authored_check_provenance"] = bool(record_witness(
                requirement_id=str(lowered.get("requirement_ref")),
                receipt_id=binding["receipt_id"],
            ))
        else:
            result["authored_check_provenance"] = False
    return result


def public_result(normalized: Mapping[str, object]) -> dict[str, object]:
    """Keep model-visible admission metadata bounded and JSON-safe."""
    return {
        "schema": SCHEMA,
        "tool": TOOL_NAME,
        "target_id": normalized.get("target_id"),
        "ticket_span": normalized.get("ticket_span"),
        "fixture_recipe": normalized.get("fixture_recipe"),
        "invocation": normalized.get("invocation"),
        "observations": list(_items(normalized.get("observations"))),
        "setup_recipe_ids": list(_items(normalized.get("setup_recipe_ids"))),
        "bindings": copy.deepcopy(normalized.get("bindings", {})),
        "assertion": normalized.get("assertion"),
        "proposal_digest": _digest({key: normalized.get(key) for key in (
            "target_id", "ticket_span", "fixture_recipe", "invocation",
            "observations", "setup_recipe_ids", "bindings", "assertion",
        )}),
    }


__all__ = [
    "FIXTURE_RECIPES", "INVOCATIONS", "OBSERVATIONS", "SCHEMA", "SETUP_RECIPES",
    "TOOL_NAME", "TestWriterError", "execute_typed_test_writer", "lower_to_reprospec", "public_result",
    "ticket_span_menu", "typed_test_writer_schema", "validate_test_writer",
]

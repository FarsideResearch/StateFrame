"""Small, host-fact-only contracts for the v30 token repairs.

This module is deliberately side-effect free except for the idempotent telemetry
adapter installed by :func:`install_telemetry_patch`.  It gives the v30 overlays
one vocabulary for executable recovery, verification identity, and normalized
provider accounting without changing completion authority or observation policy.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence

MISSING = "MISSING"
VERIFY_ACTIONS = frozenset({"run_tests", "test", "pytest"})
OBSERVATION_ACTIONS = frozenset({"read", "grep", "find", "search", "probe"})


def _canonical(value: object) -> object:
    try:
        return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))
    except (TypeError, ValueError):
        return str(value)


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


def _first(runtime: object, names: Sequence[str]) -> object:
    for name in names:
        value = getattr(runtime, name, None)
        if value not in (None, "", (), [], {}):
            return value
    return None


def _selector_list(value: object) -> list[str]:
    if value in (None, "", (), [], {}):
        return []
    values = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
    return sorted(dict.fromkeys(_text(item) for item in values if _text(item)))


def _selectors(runtime: object, arguments: Mapping[str, object] | None = None) -> list[str]:
    args = arguments if isinstance(arguments, Mapping) else {}
    value = args.get("selector", args.get("target"))
    if value in (None, "", (), [], {}):
        value = _first(runtime, (
            "verification_selector_witness", "next_verification_selector",
            "requested_verification_selector", "candidate_verification_selector",
            "_post_green_test_selectors", "_last_test_selectors", "test_selectors",
            "verification_selectors"))
    return _selector_list(value)


def _binding_dimension(binding: object, names: Sequence[str]) -> object:
    sources = [binding]
    if isinstance(binding, Mapping):
        sources.extend(binding.get(name) for name in ("verification", "harness", "environment", "checks"))
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for name in names:
            value = source.get(name)
            if value not in (None, "", (), [], {}):
                return value
    return None


def verification_identity(runtime: object, arguments: Mapping[str, object] | None = None) -> dict[str, object]:
    """Return the binding identity that makes a verification receipt reusable.

    Product revision alone is insufficient: the selected check, harness, and
    environment are part of the witness.  Missing optional dimensions remain
    absent rather than being invented, so callers can fail closed.
    """
    binding = None
    getter = getattr(runtime, "binding", None)
    if callable(getter):
        try:
            binding = getter()
        except (AttributeError, OSError, TypeError, ValueError):
            binding = None
    if isinstance(binding, Mapping):
        product = binding.get("artifact", binding.get("product_revision",
                                                        binding.get("product", binding)))
    else:
        product = binding
    result: dict[str, object] = {
        "product": _canonical(product),
        "selectors": _selectors(runtime, arguments),
    }
    dimensions = {
        "harness": ("test_harness_revision", "harness_revision", "test_runner_revision",
                     "_test_harness_revision"),
        "environment": ("environment_revision", "environment_digest", "env_revision",
                         "_environment_revision"),
        "checks": ("check_definition_revision", "check_definitions_revision", "check_revision",
                    "verification_check_revision"),
    }
    for key, names in dimensions.items():
        value = _binding_dimension(binding, names)
        if value is None:
            value = _first(runtime, names)
        if value is not None:
            result[key] = _canonical(value)
    return result


def verification_menu_arguments(runtime: object) -> dict[str, object]:
    """Arguments used by menu and dispatch identity checks alike."""
    selectors = _selectors(runtime)
    return {"selector": selectors} if selectors else {}


def _normalize_identity(value: Mapping[str, object]) -> dict[str, object]:
    """Normalize persisted and live identities to one JSON-shaped schema."""
    result: dict[str, object] = {
        "product": _canonical(value.get("product", value.get("product_revision",
                                                  value.get("artifact")))),
        "selectors": _selector_list(value.get("selectors", value.get("selector",
                                                        value.get("target", ())))),
    }
    for key in ("harness", "environment", "checks"):
        if key in value and value[key] not in (None, ""):
            result[key] = _canonical(value[key])
    return result


def _baseline_verification_identity(runtime: object) -> dict[str, object] | None:
    for name in ("_v30_green_verification_identity", "green_verification_identity",
                 "green_test_identity", "green_verification_binding"):
        value = getattr(runtime, name, None)
        if isinstance(value, Mapping):
            return _normalize_identity(value)
    state = getattr(runtime, "_post_green_menu_state", None)
    if not isinstance(state, Mapping):
        return None
    product = state.get("green_revision")
    selectors = state.get("green_selectors")
    if product in (None, "") and selectors in (None, "", (), []):
        return None
    result: dict[str, object] = {"product": _canonical(product),
                                 "selectors": _selector_list(selectors)}
    for key, names in {
        "harness": ("green_test_harness_revision", "green_harness_revision",
                    "green_test_runner_revision"),
        "environment": ("green_environment_revision", "green_environment_digest"),
        "checks": ("green_check_definition_revision", "green_check_revision"),
    }.items():
        value = state.get(f"green_{key}")
        if value in (None, ""):
            value = _first(runtime, names)
        if value is not None:
            result[key] = _canonical(value)
    return result


def admits_new_verification_binding(runtime: object, arguments: Mapping[str, object] | None = None) -> bool:
    """Return true only when a refused rerun has a changed, named witness."""
    current = verification_identity(runtime, arguments)
    baseline = _baseline_verification_identity(runtime)
    if baseline is None:
        return False
    if current == baseline:
        return False
    # A missing witness is not evidence of a changed binding. Compare only
    # dimensions present on both sides, except that newly exposed optional
    # metadata requires an explicit host change marker.
    changed = False
    for key in ("product", "selectors", "harness", "environment", "checks"):
        current_present = key in current and current[key] not in (None, "")
        baseline_present = key in baseline and baseline[key] not in (None, "")
        if not current_present or not baseline_present:
            continue
        # An empty current selector set cannot prove a previously named
        # selector was removed; fail closed on that incomplete observation.
        if key == "selectors" and baseline[key] and not current[key]:
            continue
        if current[key] != baseline[key]:
            changed = True
            break
    if changed:
        return True
    optional = {"harness", "environment", "checks"}
    if optional.intersection(current) - optional.intersection(baseline):
        return any(getattr(runtime, name, False) is True for name in (
            "verification_binding_changed", "test_harness_changed",
            "verification_environment_changed", "check_definition_changed"))
    return False


def _offered(runtime: object) -> tuple[str, ...] | None:
    for name in ("completion_menu", "offered_tools", "offered_actions", "tool_names"):
        value = getattr(runtime, name, None)
        if value is None:
            continue
        if isinstance(value, Mapping):
            value = tuple(value.keys())
        if isinstance(value, str):
            value = (value,)
        if isinstance(value, Iterable):
            return tuple(dict.fromkeys(_text(item).lower() for item in value if _text(item)))
    return None


def _probe_remaining(runtime: object) -> int | None:
    for name in ("probe_remaining", "remaining_probe_allocations", "probe_budget_remaining"):
        value = getattr(runtime, name, None)
        if type(value) is int:
            return max(0, value)
    budget = getattr(runtime, "probe_budget", None)
    if budget is not None:
        remaining = getattr(budget, "remaining", None)
        if type(remaining) is int:
            return max(0, remaining)
        count, limit = getattr(budget, "count", None), getattr(budget, "limit", None)
        if type(count) is int and type(limit) is int:
            return max(0, limit - count)
    return None


def executable_verification_route(runtime: object, missing: object = "") -> str:
    """Describe only actions present in the current host menu and budget."""
    offered = _offered(runtime)
    if offered is None:
        return "No executable verification action is currently offered; preserve the host refusal."
    missing_text = _text(missing) or "the unresolved host evidence"
    routes: list[str] = []
    if "run_tests" in offered:
        routes.append(f"run_tests with a new verification binding for {missing_text}")
    probe_remaining = _probe_remaining(runtime)
    if "probe" in offered and probe_remaining is not None and probe_remaining > 0:
        routes.append("probe using bounded Python observation-only code; no shell commands or pytest")
    if "read" in offered or "grep" in offered:
        names = "/".join(name for name in ("read", "grep") if name in offered)
        routes.append(f"{names} the exact missing witness")
    if not routes:
        return "No executable verification action is currently offered; preserve the host refusal."
    return "; ".join(routes)


def refusal_with_route(reason: object, missing: object, runtime: object) -> str:
    """Preserve one refusal reason and append one capability-accurate route."""
    reason_text = _text(reason) or "completion remains refused by the host"
    missing_text = _text(missing) or "the unresolved host evidence"
    return (f"{reason_text}. Missing: {missing_text}. "
            f"Executable route: {executable_verification_route(runtime, missing_text)}")


def verification_selector_is_executable(runtime: object, selector: object) -> bool:
    """Use an explicit host witness when one exists; never invent authorization."""
    if selector in (None, "", (), [], {}):
        return True
    for name in ("can_run_verification", "can_run_test", "verification_selector_allowed"):
        checker = getattr(runtime, name, None)
        if callable(checker):
            try:
                return bool(checker(selector))
            except (AttributeError, OSError, TypeError, ValueError):
                return False
    for name in ("authorized_verification_selectors", "authorized_check_selectors",
                 "local_check_selectors", "host_check_selectors"):
        values = getattr(runtime, name, None)
        if values is None:
            continue
        allowed = set(_selector_list(values))
        return bool(set(_selector_list(selector)) <= allowed)
    # The fallback preserves ordinary named targets but never turns a shell
    # fragment into a check selector.
    shell_prefixes = ("-", "/bin/", "sh ", "bash ", "rm ", "python ", "python3 ",
                      "pytest ", "pip ", "git ", "curl ", "wget ", "sudo ",
                      "chmod ", "echo ")
    return all(not any(marker in item for marker in ("\n", "\r", ";", "&&", "||", "`", "$("))
               and not item.lstrip().startswith(shell_prefixes)
               for item in _selector_list(selector))


def _tool_names(tools: object) -> tuple[str, ...]:
    if isinstance(tools, Mapping):
        tools = (tools,)
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return ()
    names: list[str] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if _text(name) and _text(name) not in names:
            names.append(_text(name))
    return tuple(names)


def _has_decision_carrier(tools: object) -> bool:
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return False
    for tool in tools:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        parameters = function.get("parameters") if isinstance(function, Mapping) else None
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        if isinstance(properties, Mapping) and any(
                key in properties for key in ("decision_delta", "decision_slot")):
            return True
    return False


def sanitize_final_advice(messages: object, tools: object) -> tuple[list[object], bool]:
    """Remove advice whose operation/carrier disappeared at the final menu seam."""
    offered = set(_tool_names(tools))
    has_decision_carrier = _has_decision_carrier(tools)
    work_advice = (
        "The `work` tool is intent-only in host-closure mode: submit only a "
        "claim_implemented operation with host evidence references. The host owns "
        "status and re-runs the bound checks; do not attempt to set or drop records.",
        "The `work` tool is the durable typed work record. Use it for goals, ordered steps, "
        "and decisions that must survive the next frame. It is host-owned: ids are assigned "
        "by the host, invalid operations are rejected with the current state, drops require "
        "a reason, and there is no replace operation. Keep mutations small; use evidence_ref "
        "when reopening a completed item.",
    )
    stale_slot = "supply the missing decision slot and choose the next discriminating action"
    replacement = ("No optional decision slot was supplied; choose the next offered action "
                    "from the current evidence")
    result: list[object] = []
    changed = False
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return [], False
    for message in messages:
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            result.append(message)
            continue
        content = str(message["content"])
        if "work" not in offered:
            for advice in work_advice:
                content = content.replace(advice, "")
        if not has_decision_carrier:
            content = content.replace(stale_slot, replacement)
        if content != message["content"]:
            updated = dict(message)
            updated["content"] = content
            result.append(updated)
            changed = True
        else:
            result.append(message)
    return result, changed


def _nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def normalized_prompt_accounting(response: Mapping[str, object]) -> dict[str, object]:
    """Compatibility names backed by the same provider-ingress normalizer."""
    from kernel_prompt_accounting import normalize_prompt_tokens, compatibility_fields
    return compatibility_fields(normalize_prompt_tokens(response))


def install_telemetry_patch() -> None:
    """Retain the overlay hook; accounting is now normalized at core ingestion."""
    try:
        import kernel_measurement
    except ImportError:
        return
    kernel_measurement._v30_accounting_patch = True


__all__ = [
    "MISSING", "OBSERVATION_ACTIONS", "VERIFY_ACTIONS", "executable_verification_route",
    "install_telemetry_patch", "normalized_prompt_accounting", "refusal_with_route",
    "admits_new_verification_binding", "sanitize_final_advice", "verification_identity",
    "verification_menu_arguments", "verification_selector_is_executable",
]

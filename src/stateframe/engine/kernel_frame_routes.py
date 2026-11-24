"""Per-request model route authority.

The last closed action is a checkpoint participant owned by the child.  A plan
request is selected only before any action, or after the latest action was a
run_tests call with an aggregate FAIL and *all* canonical check receipts bound
to the current workspace revision.  Other actions always replace the record.
Under routing schema 2 a plan lease also stays open through read-kind actions
(see PLAN_LEASE); schema 1 never consults it.
"""
from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from kernel_edit_tool_names import PROTOCOL_EDIT_TOOL_NAMES
from kernel_routing_schema import routing_v2

SCHEMA = "stateframe-routes/1"
SPEC_READERS = ("STATEFRAME_ROUTE_SCOPE", "STATEFRAME_ROUTE_UNIT_CLASS", "STATEFRAME_ROUTE_SLOT")
FIELDS = frozenset({"endpoint", "model", "api_key_env", "temperature", "top_p", "top_k",
                    "min_p", "max_tokens", "timeout_s", "slot", "context_tokens", "tokenizer",
                    "allow_remote", "cost_input_per_mtok", "cost_cached_input_per_mtok",
                    "cost_output_per_mtok"})
WRITE = frozenset({"edit", "str_replace", "insert_before", "replace_module", "file_create",
                   "file_delete", "file_move", "authored_check", "write", "restore_candidate",
                   *PROTOCOL_EDIT_TOOL_NAMES})
DONE = frozenset({"done"})
READ = frozenset({"grep", "list_files", "file_list", "read", "probe", "recall", "references",
                  "where_defined", "who_calls", "read_check_receipt", "run_tests", "git",
                  "web_fetch", "shell", "job_wait", "job_output", "job_stop", "skill_load",
                  "reproduce", "decide", "predict", "dependency_fact",
                  "dispute_check", "read_ticket", "dispute", "work", "test",
                  "BIND_QUANTITY", "CONTRACT_CASES", "TRACE_BEHAVIOR"})
from kernel_git_commit import registered_names as _commit_names  # noqa: E402 - empty unless STATEFRAME_GIT_COMMIT=on
READ = READ | _commit_names()


def _digest(roles: object, overrides: object) -> str:
    value = {"roles": roles, "unit_overrides": overrides} if overrides else roles
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Route:
    role: str
    scope: str
    unit_class: str | None
    settings: Mapping[str, object]
    escalated_from: str | None = None

    @property
    def endpoint(self) -> str:
        return str(self.settings["endpoint"]).rstrip("/") + "/chat/completions"

    @property
    def model(self) -> str:
        return str(self.settings["model"])

    @property
    def overridden_fields(self) -> list[str]:
        if _baseline is None:
            return []
        baseline = dict(_baseline)
        if self.role == "decide2":
            baseline.update(model="decision-tier", temperature=0.0, max_tokens=384)
        return sorted(field for field, value in self.settings.items()
                      if field not in {"allow_remote", "cost_input_per_mtok",
                                       "cost_cached_input_per_mtok", "cost_output_per_mtok"}
                      and value != baseline.get(field))

    def record(self) -> dict[str, object]:
        host = urlsplit(self.endpoint).netloc
        return {"role": self.role, "scope": self.scope, "unit_class": self.unit_class,
                "endpoint": redacted_url(self.endpoint), "model": self.model,
                "endpoint_host": host, "escalated_from": self.escalated_from}


_roles: dict[str, dict[str, object]] | None = None
ROUTES_ACTIVE = False
_overrides: dict[str, tuple[str, ...]] = {}
_baseline: dict[str, object] | None = None
LAST_CLOSED_ACTION: dict[str, object] = {}
# Routing v2 (MR-5): withheld edit/finish proposals awaiting the next provider request, bound to a revision.
PRESERVED_PROPOSALS: list[dict[str, object]] = []
# Routing v2 (MR-7): target role of the proposal the current request carries. Set when the carry is
# consumed (that happens before the request is prepared), cleared when the next action closes.
CARRY_TARGET: dict[str, str] = {}
ACCOUNTING: dict[str, int] = {"physical": 0, "accepted": 0, "discarded": 0}
# MR-4 (routing schema 2 only). Read-kind frame tools that keep a plan lease open. These are the
# router's _READ_ACTIONS lowercased; tests/test_frame_routes_plan_lease.py pins the match.
LEASE_READ_ACTIONS = frozenset({
    "read", "read_file", "read_requested_file", "refresh_read", "recall_evidence",
    "mechanical_read", "grep",
})
# open: True from start and after a qualifying FAIL run_tests; closed by the first non-read action.
# by_fail/revision: a FAIL-opened lease only holds while the revision it was opened on is current.
PLAN_LEASE: dict[str, object] = {}


def _reset_plan_lease() -> None:
    PLAN_LEASE.clear()
    PLAN_LEASE.update(open=True, by_fail=False, revision=None)


_reset_plan_lease()


def install(spec: object, *, baseline: Mapping[str, object] | None = None) -> None:
    """Verify Lane 1's effective public table before accepting it as authority."""
    global _roles, _overrides, _baseline, ROUTES_ACTIVE
    if spec is None:
        _roles, _overrides, _baseline = None, {}, None
        ROUTES_ACTIVE = False
        LAST_CLOSED_ACTION.clear()
        _reset_plan_lease()
        PRESERVED_PROPOSALS.clear()
        CARRY_TARGET.clear()
        ACCOUNTING.update(physical=0, accepted=0, discarded=0)
        return
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise ValueError("invalid routes spec schema")
    roles, overrides = spec.get("roles"), spec.get("unit_overrides", {})
    if not isinstance(roles, dict) or not isinstance(overrides, dict):
        raise ValueError("invalid routes spec table")
    if _digest(roles, overrides) != spec.get("digest"):
        raise ValueError("routes spec digest mismatch")
    required = {"read", "plan", "edit", "finish", "decide2", "unit"}
    if not required.issubset(roles) or any(not isinstance(value, dict) for value in roles.values()):
        raise ValueError("routes spec is missing an effective role")
    if any((role not in required and not (role.startswith("unit.") and role[5:]
              and role[5].islower() and all(char in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                                            for char in role[5:])))
           or set(settings) - FIELDS or not isinstance(settings.get("endpoint"), str)
           or not isinstance(settings.get("model"), str)
           for role, settings in roles.items()):
        raise ValueError("routes spec contains an unknown role or field")
    for role, fields in overrides.items():
        if role not in roles or not (role == "unit" or role.startswith("unit.")) or not isinstance(fields, list):
            raise ValueError("invalid unit route override mask")
        if any(field not in roles[role] for field in fields):
            raise ValueError("unit route override mask names an absent field")
    _roles = {name: dict(value) for name, value in roles.items()}
    _overrides = {name: tuple(fields) for name, fields in overrides.items()}
    _baseline = dict(baseline) if baseline is not None else None
    ROUTES_ACTIVE = True
    LAST_CLOSED_ACTION.clear()
    _reset_plan_lease()
    PRESERVED_PROPOSALS.clear()
    CARRY_TARGET.clear()
    ACCOUNTING.update(physical=0, accepted=0, discarded=0)


def note_response(*, discarded: bool) -> None:
    ACCOUNTING["physical"] += 1
    ACCOUNTING["discarded" if discarded else "accepted"] += 1


def reject_capture(runtime: object, capture: object) -> None:
    """Keep a rejected leg out of finalization if its replacement is refused."""
    capture.route_response_discarded = True
    runtime._active_capture = runtime._last_capture


def active() -> bool:
    return ROUTES_ACTIVE


def route_model_ids() -> frozenset[str]:
    """Every model name the installed route table names; empty when no table is installed.

    The router's launch policy set adds these under routing v2 (MR-5), so a routed leg may select its model.
    """
    if not ROUTES_ACTIVE or not _roles:
        return frozenset()
    return frozenset(str(settings["model"]) for settings in _roles.values()
                     if isinstance(settings.get("model"), str) and settings["model"])


def unit_scope() -> tuple[str, str | None, int | None]:
    if not active():
        return "run", None, None
    scope = os.environ.get("STATEFRAME_ROUTE_SCOPE", "run")
    if scope not in {"run", "unit"}:
        raise ValueError("STATEFRAME_ROUTE_SCOPE must be run or unit")
    unit_class = os.environ.get("STATEFRAME_ROUTE_UNIT_CLASS", "default") if scope == "unit" else None
    if unit_class is not None and (not unit_class or not unit_class[0].islower() or
                                   any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in unit_class)):
        raise ValueError("invalid STATEFRAME_ROUTE_UNIT_CLASS")
    raw_slot = os.environ.get("STATEFRAME_ROUTE_SLOT") if scope == "unit" else None
    if raw_slot is not None and (not raw_slot.isdigit() or int(raw_slot) < 0):
        raise ValueError("STATEFRAME_ROUTE_SLOT must be a nonnegative integer")
    return scope, unit_class, int(raw_slot) if raw_slot is not None else None


def resolve(role: str, *, forced_by: str | None = None) -> Route:
    if _roles is None or role not in _roles:
        raise ValueError("route role is unavailable")
    scope, unit_class, slot = unit_scope()
    settings = dict(_roles[role])
    if scope == "unit":
        for name in ("unit", f"unit.{unit_class}"):
            for field in _overrides.get(name, ()):
                settings[field] = _roles[name][field]
        if slot is not None:
            settings["slot"] = slot
    return Route(role, scope, unit_class, settings, forced_by)


def role_for_request(revision: object = None, *, request_kind: str | None = None) -> str:
    if routing_v2() and request_kind != "decide2" and PRESERVED_PROPOSALS:
        # MR-7: the request that carries a withheld proposal runs on the proposal's target route.
        # Oldest entry first; this outranks the plan lease and the legacy role logic below.
        return str(PRESERVED_PROPOSALS[0]["target_role"])
    if routing_v2() and request_kind != "decide2" and CARRY_TARGET.get("role"):
        return CARRY_TARGET["role"]
    if os.environ.get("STATEFRAME_MUTATIONS", "off") in {"prebuilt", "authored"}:
        from kernel_mutate_runtime import service as mutation_service
        _mutations = mutation_service()
        if _mutations is not None and _mutations.episode is not None:
            return "edit"
    if request_kind == "decide2":
        return "decide2"
    if routing_v2() and _plan_lease_holds(revision):
        return "plan"
    if not LAST_CLOSED_ACTION:
        return "plan"
    row = LAST_CLOSED_ACTION
    if (row.get("action") == "run_tests" and row.get("status") == "FAIL"
            and row.get("revision") == revision and row.get("all_receipts_current") is True
            and row.get("execution_ids") and row.get("check_receipt_ids")):
        return "plan"
    return "read"


def close_action(record: Mapping[str, object], *, revision: object = None,
                 canonical: Mapping[str, object] | None = None,
                 receipts: list[Mapping[str, object]] | None = None) -> None:
    CARRY_TARGET.clear()
    canonical = canonical or {}
    ids = list(canonical.get("execution_ids") or [])
    receipt_ids = list(canonical.get("check_receipt_ids") or [])
    all_current = bool(ids and receipt_ids and len(ids) == len(receipt_ids) and receipts and
                       len(set(receipt_ids)) == len(receipt_ids) and
                       len(receipts) == len(receipt_ids) and
                       {row.get("receipt_id") for row in receipts} == set(receipt_ids) and
                       all(item.get("revision") == revision for item in receipts))
    qualifying_fail = bool(record.get("tool") == "run_tests" and canonical.get("status") == "FAIL"
                           and all_current and ids and receipt_ids)
    LAST_CLOSED_ACTION.clear()
    LAST_CLOSED_ACTION.update(action=record.get("tool"), call_id=record.get("tool_call_id"),
                              execution_ids=ids, check_receipt_ids=receipt_ids,
                              status=canonical.get("status"), revision=revision,
                              all_receipts_current=all_current)
    if routing_v2():
        _advance_plan_lease(record.get("tool"), qualifying_fail=qualifying_fail, revision=revision)


def _plan_lease_holds(revision: object) -> bool:
    if PLAN_LEASE.get("open") is not True:
        return False
    return PLAN_LEASE.get("by_fail") is False or PLAN_LEASE.get("revision") == revision


def _advance_plan_lease(action: object, *, qualifying_fail: bool, revision: object) -> None:
    """Reads keep the lease as it is; a qualifying FAIL opens it; any other action closes it."""
    if qualifying_fail:
        PLAN_LEASE.clear()
        PLAN_LEASE.update(open=True, by_fail=True, revision=revision)
    elif action not in LEASE_READ_ACTIONS:
        PLAN_LEASE.update(open=False)


def classify_calls(calls: object, *, custom_writes: set[str] | frozenset[str] = frozenset()) -> str:
    names = [(call.get("function") or {}).get("name") for call in calls] if isinstance(calls, list) else []
    if not names or any(not isinstance(name, str) for name in names):
        return "read"
    if any(name in WRITE or name in custom_writes for name in names):
        return "edit"
    if any(name in DONE for name in names):
        return "finish"
    return "read"


def workspace_write_tools(agent_tools: object) -> frozenset[str]:
    """Names of custom command, skill and MCP tools that declare workspace writes; they classify as edit."""
    custom = getattr(agent_tools, "custom", None)
    return frozenset(name for name, tool in (custom.items() if isinstance(custom, Mapping) else ())
                     if getattr(tool, "writes", None) == "workspace")


def forced_leg_options(capture: object) -> dict[str, object]:
    """prepare() options that rebuild an escalated leg with its forced role; {} for any other capture."""
    route = getattr(capture, "frame_route", None)
    frozen = getattr(capture, "_frame_route_frozen_capture", None)
    if route is None or route.escalated_from is None or frozen is None:
        return {}
    return {"forced_route": route.role, "frozen_route_capture": frozen}


def _well_formed(calls: list) -> bool:
    return not any(not isinstance(call, dict) or not isinstance(call.get("function"), dict)
                   or not isinstance(call["function"].get("name"), str)
                   or not isinstance(call["function"].get("arguments"), str)
                   for call in calls)


def escalation_target(answering: Route | None, calls: object,
                      *, custom_writes: set[str] | frozenset[str] = frozenset()) -> str | None:
    if answering is None or not isinstance(calls, list) or not calls:
        return None
    if not _well_formed(calls):
        return None
    role = classify_calls(calls, custom_writes=custom_writes)
    if role not in {"edit", "finish"}:
        return None
    target = resolve(role)
    # MR-7: settings-only differences (temperature, credential, slot, ...) escalate too. The caller
    # re-prepares the leg on the target role, which applies the target settings. A new marker value
    # would need react_repair.py changes (forced_route -> resolve), so the role itself is returned.
    return role if _route_differs(target, answering) else None


def _route_differs(target: Route, answering: Route) -> bool:
    """Routing v2 compares the full route identity (MR-7); v1 keeps the legacy (endpoint, model) pair."""
    if routing_v2():
        return route_identity(target) != route_identity(answering)
    return (target.endpoint, target.model) != (answering.endpoint, answering.model)


class RouteEscalation(Exception):
    def __init__(self, target: str, calls: list[dict]) -> None:
        self.target = target
        self.calls = calls


def route_split(answering: Route | None, calls: object, *,
                custom_writes: set[str] | frozenset[str] = frozenset()
                ) -> tuple[list[dict], list[dict], str | None]:
    """Routing v2 (MR-5): split a batch into calls that run on the answering route and withheld calls.

    Returns (kept, withheld, target). Only edit/finish calls whose role resolves to another route
    (see _route_differs: full identity under v2) are withheld; read-kind calls always stay. target is None when nothing escalates,
    and then kept is the whole batch. Callers use this only under routing_v2().
    """
    if answering is None or not isinstance(calls, list) or not _well_formed(calls):
        return (list(calls) if isinstance(calls, list) else []), [], None
    kept: list[dict] = []
    withheld: list[dict] = []
    targets: set[str] = set()
    for call in calls:
        role = classify_calls([call], custom_writes=custom_writes)
        if role in {"edit", "finish"}:
            target = resolve(role)
            if _route_differs(target, answering):
                withheld.append(call)
                targets.add(role)
                continue
        kept.append(call)
    if not withheld:
        return list(calls), [], None
    return kept, withheld, "edit" if "edit" in targets else "finish"


def _proposal_digest(calls: list[dict]) -> str:
    return hashlib.sha256(json.dumps(calls, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def preserve_proposal(journal: Any, *, request_id: object, target_role: str,
                      calls: list[dict], revision: object) -> None:
    """Routing v2 (MR-5): keep a withheld proposal, bound to the revision it was made against."""
    snapshot = copy.deepcopy(list(calls))
    digest = _proposal_digest(snapshot)
    PRESERVED_PROPOSALS.append({"calls": snapshot, "target_role": target_role, "revision": revision})
    journal.emit("route_proposal_preserved", request_id=request_id, target_role=target_role,
                 revision=revision, call_names=[(call.get("function") or {}).get("name") for call in snapshot],
                 proposal_digest=digest, dispatch=False)


def has_preserved_proposals() -> bool:
    return bool(PRESERVED_PROPOSALS)


def take_preserved_messages(current_revision: object) -> list[dict[str, str]]:
    """Consume withheld proposals as user messages for the next request; stale ones are marked."""
    items = list(PRESERVED_PROPOSALS)
    PRESERVED_PROPOSALS.clear()
    if items and routing_v2():
        CARRY_TARGET["role"] = str(items[0]["target_role"])
    messages: list[dict[str, str]] = []
    for item in items:
        body = json.dumps(item["calls"], sort_keys=True, indent=2)
        proposed = item["revision"]
        if proposed == current_revision:
            head = f"A previous step proposed these changes against revision {proposed}; apply them if still correct, or revise:"
        else:
            head = (f"A previous step proposed these changes against revision {proposed}, which is stale "
                    f"(the workspace is now at revision {current_revision}); re-check them against the "
                    "current state before applying, or revise:")
        messages.append({"role": "user", "content": f"{head}\n{body}"})
    return messages


def unclassified_dispatcher(names: set[str] | frozenset[str]) -> set[str]:
    return set(names) - WRITE - DONE - READ


BODY_FIELDS = ("model", "temperature", "top_p", "top_k", "min_p", "max_tokens")


def applied_settings(route: Route) -> dict[str, object]:
    """The settings a route puts on a leg: body fields apply_body writes, slot, and api_key_env.

    Single source for apply_body and the routing v2 identity, so the comparison and the application
    cannot drift. A body field equal to the baseline (or unset) is omitted, since apply_body never writes it.
    """
    settings = route.settings
    applied: dict[str, object] = {}
    for field in BODY_FIELDS:
        inherited = (_baseline or {}).get(field)
        if route.role == "decide2":
            inherited = {"model": "decision-tier", "temperature": 0.0,
                         "max_tokens": 384}.get(field, inherited)
        value = settings.get(field)
        if value is not None and (_baseline is None or value != inherited):
            applied[field] = value
    slot = settings.get("slot")
    if slot is not None:
        applied["slot"] = slot
    applied["api_key_env"] = settings.get("api_key_env")
    return applied


def route_identity(route: Route) -> dict[str, object]:
    """Routing v2 (MR-7): effective identity, including settings outside the provider body."""
    settings = route.settings
    identity = {"endpoint": route.endpoint, "model": route.model, **applied_settings(route)}
    # Use runtime fallbacks so older saved route tables remain comparable after resume.
    identity.update(
        timeout_s=settings.get("timeout_s") or 1800.0,
        context_tokens=settings.get("context_tokens") or (_baseline or {}).get("context_tokens") or 65536,
        tokenizer=settings.get("tokenizer") or settings.get("endpoint") or route.endpoint,
        allow_remote=bool(settings.get("allow_remote")),
    )
    return identity


def apply_body(body: dict[str, object], route: Route, *, default_endpoint: str) -> None:
    applied = applied_settings(route)
    for field in BODY_FIELDS:
        if field not in applied:
            continue
        if field == "max_tokens" and type(body.get(field)) is int:
            body[field] = min(body[field], applied[field])
        else:
            body[field] = applied[field]
    slot = applied.get("slot")
    if route.endpoint != default_endpoint and slot is None:
        body.pop("id_slot", None)
        body.pop("cache_prompt", None)
    elif type(slot) is int:
        body["id_slot"] = slot
        body["cache_prompt"] = True


def redacted_url(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parsed.port is not None:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def require_local(url: str, *, allow_remote: bool) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("routed endpoint is not a safe HTTP URL")
    try:
        address = ipaddress.ip_address(parsed.hostname)
        local = address.is_loopback
    except ValueError:
        local = parsed.hostname.lower() == "localhost"
    if not local and not allow_remote and os.environ.get("STATEFRAME_ALLOW_REMOTE_ENDPOINT") != "1":
        raise ValueError("routed endpoint is not on this machine; enable allow_remote for this role")

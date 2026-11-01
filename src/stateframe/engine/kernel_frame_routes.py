"""Per-request model route authority.

The last closed action is a checkpoint participant owned by the child.  A plan
request is selected only before any action, or after the latest action was a
run_tests call with an aggregate FAIL and *all* canonical check receipts bound
to the current workspace revision.  Other actions always replace the record.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit

SCHEMA = "stateframe-routes/1"
SPEC_READERS = ("STATEFRAME_ROUTE_SCOPE", "STATEFRAME_ROUTE_UNIT_CLASS", "STATEFRAME_ROUTE_SLOT")
FIELDS = frozenset({"endpoint", "model", "api_key_env", "temperature", "top_p", "top_k",
                    "min_p", "max_tokens", "timeout_s", "slot", "context_tokens", "tokenizer",
                    "allow_remote", "cost_input_per_mtok", "cost_cached_input_per_mtok",
                    "cost_output_per_mtok"})
WRITE = frozenset({"edit", "str_replace", "insert_before", "replace_module", "file_create",
                   "file_delete", "file_move", "authored_check", "write", "restore_candidate"})
DONE = frozenset({"done"})
READ = frozenset({"grep", "list_files", "file_list", "read", "probe", "recall", "references",
                  "where_defined", "who_calls", "read_check_receipt", "run_tests", "git",
                  "web_fetch", "shell", "job_wait", "job_output", "job_stop", "skill_load",
                  "reproduce", "decide", "predict", "dependency_fact",
                  "dispute_check", "read_ticket", "dispute", "work", "test",
                  "BIND_QUANTITY", "CONTRACT_CASES", "TRACE_BEHAVIOR"})


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
ACCOUNTING: dict[str, int] = {"physical": 0, "accepted": 0, "discarded": 0}


def install(spec: object, *, baseline: Mapping[str, object] | None = None) -> None:
    """Verify Lane 1's effective public table before accepting it as authority."""
    global _roles, _overrides, _baseline, ROUTES_ACTIVE
    if spec is None:
        _roles, _overrides, _baseline = None, {}, None
        ROUTES_ACTIVE = False
        LAST_CLOSED_ACTION.clear()
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
    if request_kind == "decide2":
        return "decide2"
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
    canonical = canonical or {}
    ids = list(canonical.get("execution_ids") or [])
    receipt_ids = list(canonical.get("check_receipt_ids") or [])
    all_current = bool(ids and receipt_ids and len(ids) == len(receipt_ids) and receipts and
                       len(set(receipt_ids)) == len(receipt_ids) and
                       len(receipts) == len(receipt_ids) and
                       {row.get("receipt_id") for row in receipts} == set(receipt_ids) and
                       all(item.get("revision") == revision for item in receipts))
    LAST_CLOSED_ACTION.clear()
    LAST_CLOSED_ACTION.update(action=record.get("tool"), call_id=record.get("tool_call_id"),
                              execution_ids=ids, check_receipt_ids=receipt_ids,
                              status=canonical.get("status"), revision=revision,
                              all_receipts_current=all_current)


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


def escalation_target(answering: Route | None, calls: object,
                      *, custom_writes: set[str] | frozenset[str] = frozenset()) -> str | None:
    if answering is None or not isinstance(calls, list) or not calls:
        return None
    if any(not isinstance(call, dict) or not isinstance(call.get("function"), dict)
           or not isinstance(call["function"].get("name"), str)
           or not isinstance(call["function"].get("arguments"), str)
           for call in calls):
        return None
    role = classify_calls(calls, custom_writes=custom_writes)
    if role not in {"edit", "finish"}:
        return None
    target = resolve(role)
    return role if (target.endpoint, target.model) != (answering.endpoint, answering.model) else None


class RouteEscalation(Exception):
    def __init__(self, target: str, calls: list[dict]) -> None:
        self.target = target
        self.calls = calls


def unclassified_dispatcher(names: set[str] | frozenset[str]) -> set[str]:
    return set(names) - WRITE - DONE - READ


def apply_body(body: dict[str, object], route: Route, *, default_endpoint: str) -> None:
    settings = route.settings
    for field in ("model", "temperature", "top_p", "top_k", "min_p", "max_tokens"):
        inherited = (_baseline or {}).get(field)
        if route.role == "decide2":
            inherited = {"model": "decision-tier", "temperature": 0.0,
                         "max_tokens": 384}.get(field, inherited)
        if settings.get(field) is not None and (_baseline is None or settings[field] != inherited):
            if field == "max_tokens" and type(body.get(field)) is int:
                body[field] = min(body[field], settings[field])
            else:
                body[field] = settings[field]
    slot = settings.get("slot")
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

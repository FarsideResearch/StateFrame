"""Resolve public, per-role model settings without storing API key values."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .endpoint import EndpointError, _is_loopback, normalize_base_url

SCHEMA = "stateframe-routes/1"
ROLES = ("read", "plan", "edit", "finish", "decide2", "unit")
FIELDS = ("endpoint", "model", "api_key_env", "temperature", "top_p", "top_k", "min_p",
          "max_tokens", "timeout_s", "slot", "context_tokens", "tokenizer", "allow_remote",
          "cost_input_per_mtok", "cost_cached_input_per_mtok", "cost_output_per_mtok")
_CLASS = re.compile(r"[a-z][a-z0-9_-]*\Z")
_ENV_FIELDS = sorted(FIELDS, key=len, reverse=True)
_SPEC_ONLY = {"STATEFRAME_ROUTE_SCOPE", "STATEFRAME_ROUTE_UNIT_CLASS", "STATEFRAME_ROUTE_SLOT"}


class RouteError(ValueError):
    """A routing source cannot be used safely."""


class RecordedRouteError(RouteError):
    """A route table saved in a run is malformed."""


@dataclass(frozen=True)
class RouteTable:
    roles: dict[str, dict[str, object]]
    sources: dict[str, dict[str, str]]
    active: bool
    digest: str
    first_change: tuple[str, str] | None = None

    def public(self) -> dict[str, object]:
        public: dict[str, object] = {"schema": SCHEMA, "digest": self.digest, "roles": self.roles}
        overrides = self.unit_overrides()
        if overrides:
            public["unit_overrides"] = overrides
        return public

    def unit_overrides(self) -> dict[str, list[str]]:
        """Explicit unit fields needed to overlay a request role from the public spec."""
        return {role: sorted(field for field, source in self.sources[role].items() if source != "default")
                for role in self.roles if role == "unit" or role.startswith("unit.")
                if any(source != "default" for source in self.sources[role].values())}

    def resolve(self, role: str, *, unit_class: str | None = None) -> dict[str, object]:
        """Unit settings overlay the request role field by field."""
        result = dict(self.roles[role])
        if unit_class is not None:
            for name in ("unit", f"unit.{unit_class}"):
                if name in self.roles:
                    result.update({key: value for key, value in self.roles[name].items()
                                   if self.sources[name].get(key) != "default"})
        return result


def valid_role(role: str) -> bool:
    return role in ROLES or (role.startswith("unit.") and bool(_CLASS.fullmatch(role[5:])))


def uses_other_model(role: str, model: object, default_model: str) -> bool:
    """Whether a role's model differs from the run's model; decide2's built-in alias is the default model."""
    return model not in {default_model, "decision-tier" if role == "decide2" else default_model}


def _check_role(role: object) -> str:
    if not isinstance(role, str) or not valid_role(role):
        raise RouteError(f"unknown route role {role!r}; roles are read, plan, edit, finish, decide2, unit, unit.<class>")
    return role


def _value(role: str, field: str, raw: object, *, text_value: bool = False) -> object:
    label = f"route {role}.{field}"
    if field in ("endpoint", "tokenizer"):
        if not isinstance(raw, str):
            raise RouteError(f"{label} must be an http(s) URL; change {field}")
        try:
            return normalize_base_url(raw)
        except EndpointError as error:
            raise RouteError(f"{label} is invalid: {error}; change {field}") from None
    if field == "model":
        if not isinstance(raw, str) or not raw.strip():
            raise RouteError(f"{label} must be a nonempty model name; change model")
        return raw.strip()
    if field == "api_key_env":
        if (not isinstance(raw, str) or not raw or not raw.replace("_", "").isalnum()
                or raw[0].isdigit()):
            raise RouteError(f"{label} must be an environment variable NAME, not a key; change api_key_env")
        return raw
    if field == "allow_remote":
        if type(raw) is not bool:
            if text_value and isinstance(raw, str) and raw.lower() in ("true", "false"):
                return raw.lower() == "true"
            raise RouteError(f"{label} must be true or false; change allow_remote")
        return raw
    if field == "slot" and raw == "auto":
        return raw
    if field in ("slot", "top_k", "max_tokens", "context_tokens"):
        try:
            number = int(raw) if text_value and isinstance(raw, str) and raw.strip().lstrip("+").isdigit() else raw
        except (ValueError, OverflowError):
            number = None
        minimum = 0 if field in ("slot", "top_k") else 1
        if type(number) is not int or number < minimum:
            raise RouteError(f"{label} must be {'a nonnegative' if minimum == 0 else 'a positive'} integer; change {field}")
        return number
    try:
        number = float(raw) if (type(raw) in (int, float) or text_value and isinstance(raw, str)) else math.nan
    except (TypeError, ValueError, OverflowError):
        number = math.nan
    bounds = {"temperature": (0, 2), "top_p": (0, 1), "min_p": (0, 1)}
    if field in bounds:
        low, high = bounds[field]
        good = math.isfinite(number) and low <= number <= high
    else:
        good = math.isfinite(number) and number >= (0 if field.startswith("cost_") else 0.000001)
    if not good:
        raise RouteError(f"{label} must be a finite number in the allowed range; change {field}")
    return number


def _blocks(document: object, source: str) -> dict[str, dict[str, object]]:
    if not isinstance(document, dict):
        raise RouteError(f"{source} must contain a roles table; change the file")
    if document.get("schema", SCHEMA) != SCHEMA or set(document) - {"schema", "roles"}:
        raise RouteError(f"{source} must use schema {SCHEMA} and a roles table; change the file")
    roles = document.get("roles", {})
    if not isinstance(roles, dict):
        raise RouteError(f"{source} roles must be a table; change the file")
    expanded = dict(roles)
    unit = expanded.get("unit")
    if isinstance(unit, dict):
        unit = dict(unit)
        expanded["unit"] = unit
        classes = {key: value for key, value in unit.items() if key not in FIELDS and isinstance(value, dict)}
        for class_name, values in classes.items():
            if f"unit.{class_name}" in expanded:
                raise RouteError(f"duplicate route role unit.{class_name}; use one table in {source}")
            expanded[f"unit.{class_name}"] = values
            del unit[class_name]
    output = {}
    for role, values in expanded.items():
        role = _check_role(role)
        if not isinstance(values, dict):
            raise RouteError(f"route {role} must be a table of fields; change {source}")
        output[role] = {}
        for field, raw in values.items():
            if field not in FIELDS:
                raise RouteError(f"unknown route field {role}.{field}; fields are {', '.join(FIELDS)}; change {source}")
            output[role][field] = _value(role, field, raw)
    return output


def load_file(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    try:
        raw = path.read_bytes()
        document = json.loads(raw) if path.suffix.lower() == ".json" else tomllib.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeError) as error:
        raise RouteError(f"cannot read routes file {path}: {error}; fix it or remove --routes") from None
    if path.suffix.lower() == ".json" and (not isinstance(document, dict) or document.get("schema") != SCHEMA):
        raise RouteError(f"routes JSON {path} needs schema {SCHEMA}; change the file")
    return _blocks(document, str(path))


def routes_path(explicit: Path | None, environ: Mapping[str, str]) -> Path | None:
    if explicit is not None:
        return explicit.expanduser()
    if environ.get("STATEFRAME_ROUTES"):
        return Path(environ["STATEFRAME_ROUTES"]).expanduser()
    default = Path.home() / ".stateframe" / "routes.toml"
    return default if default.is_file() else None


def _env_blocks(environ: Mapping[str, str]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, raw in environ.items():
        if not name.startswith("STATEFRAME_ROUTE_") or name in _SPEC_ONLY:
            continue
        tail = name.removeprefix("STATEFRAME_ROUTE_")
        field = next((candidate for candidate in _ENV_FIELDS if tail.endswith("_" + candidate.upper())), None)
        if field is None:
            raise RouteError(f"unknown route setting {name}; use STATEFRAME_ROUTE_<ROLE>_<FIELD> or unset it")
        role = tail[:-(len(field) + 1)].lower()
        if role.startswith("unit_") and role != "unit":
            role = "unit." + role[5:]
        _check_role(role)
        result.setdefault(role, {})[field] = _value(role, field, raw, text_value=True)
    return result


def _cli_blocks(values: list[str]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for entry in values:
        left, sep, raw = entry.partition("=")
        role, dot, field = left.rpartition(".")
        if not sep or not dot:
            raise RouteError(f"--route {entry!r} must be ROLE.FIELD=VALUE; change --route")
        _check_role(role)
        if field not in FIELDS:
            raise RouteError(f"unknown route field {role}.{field}; fields are {', '.join(FIELDS)}; change --route")
        result.setdefault(role, {})[field] = _value(role, field, raw, text_value=True)
    return result


def _role_base(default: Mapping[str, object], role: str) -> dict[str, object]:
    base = dict(default)
    if role == "decide2":
        base.update(model="decision-tier", temperature=0.0, max_tokens=384)
    return base


def _recorded_blocks(document: object, default: Mapping[str, object]) -> dict[str, dict[str, object]]:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise RecordedRouteError(f"recorded routes must use schema {SCHEMA}")
    saved_roles = document.get("roles")
    saved_masks = document.get("unit_overrides", {})
    if not isinstance(saved_roles, dict) or not isinstance(saved_masks, dict):
        raise RecordedRouteError("recorded routes.roles and routes.unit_overrides must be tables")
    validated: dict[str, dict[str, object]] = {}
    for role, settings in saved_roles.items():
        field_label = next(iter(settings), "*") if isinstance(settings, dict) else "*"
        try:
            _check_role(role)
        except RouteError as error:
            raise RecordedRouteError(f"recorded route {role}.{field_label}: {error}") from None
        if not isinstance(settings, dict):
            raise RecordedRouteError(f"recorded route {role}.* must be a table of fields")
        validated[role] = {}
        for field, raw in settings.items():
            if field not in FIELDS:
                raise RecordedRouteError(f"unknown recorded route field {role}.{field}")
            try:
                validated[role][field] = (None if raw is None and field in ("api_key_env", "slot")
                                           else _value(role, field, raw))
            except RouteError as error:
                raise RecordedRouteError(f"recorded {error}") from None
    blocks: dict[str, dict[str, object]] = {}
    for role, settings in validated.items():
        if role == "unit" or role.startswith("unit."):
            blocks[role] = {}
            continue
        original = _role_base(default, role)
        blocks[role] = {field: value for field, value in settings.items()
                        if value != original.get(field)}
    for role, fields in saved_masks.items():
        if not isinstance(role, str) or (role != "unit" and not role.startswith("unit.")):
            raise RecordedRouteError(f"recorded route {role}.* is not a unit role")
        try:
            _check_role(role)
        except RouteError as error:
            raise RecordedRouteError(f"recorded route {role}.*: {error}") from None
        if not isinstance(fields, list):
            raise RecordedRouteError(f"recorded route {role}.* unit_overrides must be a list of fields")
        blocks[role] = {}
        for field in fields:
            if not isinstance(field, str) or field not in FIELDS:
                raise RecordedRouteError(f"unknown recorded route field {role}.{field}")
            if field not in validated.get(role, {}):
                raise RecordedRouteError(f"recorded route {role}.{field} has no saved value")
            blocks[role][field] = validated[role][field]
    return blocks


def resolve(default: Mapping[str, object], *, cli: list[str] | None = None, file: Path | None = None,
            profile: object = None, environ: Mapping[str, str] | None = None,
            recorded: object = None, recorded_default: Mapping[str, object] | None = None) -> RouteTable:
    environ = os.environ if environ is None else environ
    layers = (("recorded", _recorded_blocks(recorded, recorded_default or default) if recorded is not None else {}),
              ("profile", _blocks({} if profile is None else profile, "profile routes")),
              ("file", load_file(routes_path(file, environ))),
              ("env", _env_blocks(environ)), ("cli", _cli_blocks(cli or [])))
    names = list(ROLES)
    names.extend(sorted({role for _, layer in layers for role in layer if role.startswith("unit.")}))
    roles: dict[str, dict[str, object]] = {}
    sources: dict[str, dict[str, str]] = {}
    for role in names:
        base = _role_base(default, role)
        chosen = {field: "default" for field in base}
        for source, layer in layers:
            for field, value in layer.get(role, {}).items():
                base[field] = value
                chosen[field] = source
        endpoint = str(base["endpoint"])
        if not _is_loopback(endpoint) and not (base.get("allow_remote") or
                                              environ.get("STATEFRAME_ALLOW_REMOTE_ENDPOINT") == "1"):
            raise RouteError(f"route {role} endpoint {endpoint} is not on this machine; add {role}.allow_remote = true "
                             "(your code will be sent there) or use a local endpoint")
        key_name = base.get("api_key_env")
        if key_name and key_name not in environ:
            raise RouteError(f"route {role} uses api_key_env {key_name}, which is not set; export it or remove api_key_env")
        roles[role], sources[role] = base, chosen
    table = RouteTable(roles, sources, False, "")
    unit_classes = {"default", *(role[5:] for role in roles if role.startswith("unit."))}
    for role in ROLES:
        if role == "unit":
            continue
        for unit_class in unit_classes:
            effective = table.resolve(role, unit_class=unit_class)
            endpoint = str(effective["endpoint"])
            if not _is_loopback(endpoint) and not (effective.get("allow_remote") or
                                                  environ.get("STATEFRAME_ALLOW_REMOTE_ENDPOINT") == "1"):
                raise RouteError(f"route {role} (unit {unit_class}) endpoint {endpoint} is not on this machine; "
                                 "set unit.allow_remote = true (your code will be sent there) or use a local endpoint")
    first_change = first_request_setting_change(table, default, environ=environ)
    return with_effective_roles(RouteTable(roles, sources, first_change is not None, "", first_change))


def request_settings_changed(table: RouteTable, default: Mapping[str, object], *,
                             environ: Mapping[str, str] | None = None) -> bool:
    """Compare every request role in main and unit scope with its unrouted request."""
    return first_request_setting_change(table, default, environ=environ) is not None


def first_request_setting_change(table: RouteTable, default: Mapping[str, object], *,
                                 environ: Mapping[str, str] | None = None) -> tuple[str, str] | None:
    """Name the first route setting that changes an unrouted request."""
    ignored = {"allow_remote", "cost_input_per_mtok", "cost_cached_input_per_mtok",
               "cost_output_per_mtok"}
    fallback_tokenizer = (environ or {}).get("LLAMA_CPP_TOKENIZE_URL")
    unit_classes = {"default", *(role[5:] for role in table.roles if role.startswith("unit."))}

    def owner(role: str, unit_class: str | None, field: str) -> str:
        if unit_class is not None:
            for unit_role in (f"unit.{unit_class}", "unit"):
                if table.sources.get(unit_role, {}).get(field, "default") != "default":
                    return unit_role
        return role

    for role in ROLES:
        if role == "unit":
            continue
        expected = _role_base(default, role)
        actual = table.roles[role]
        default_tokenizer = fallback_tokenizer or expected.get("tokenizer") or expected["endpoint"]
        for unit_class in (None, *sorted(unit_classes)):
            settings = actual if unit_class is None else table.resolve(role, unit_class=unit_class)
            route_tokenizer = settings.get("tokenizer") or fallback_tokenizer or settings["endpoint"]
            if route_tokenizer != default_tokenizer:
                field = "tokenizer" if settings.get("tokenizer") else "endpoint"
                return owner(role, unit_class, field), field
            for field in settings:
                if field not in ignored and field != "tokenizer" and settings.get(field) != expected.get(field):
                    return owner(role, unit_class, field), field
    return None


def with_effective_roles(table: RouteTable) -> RouteTable:
    overrides = table.unit_overrides()
    contents: object = {"roles": table.roles, "unit_overrides": overrides} if overrides else table.roles
    canonical = json.dumps(contents, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return RouteTable(table.roles, table.sources, table.active, digest, table.first_change)


def example() -> str:
    descriptions = {"read": "Navigate, read, search, and run checks.",
                    "plan": "Orient at the start and diagnose failed checks.",
                    "edit": "Write or change files.", "finish": "Decide whether the work is done.",
                    "decide2": "Answer the decide2 advisory request.",
                    "unit": "Handle a parallel edit unit.",
                    "unit.default": "Override the default parallel unit class."}
    lines = ["# StateFrame per-role model routes. Uncomment only the roles and fields you need.",
             "# Values not set here inherit the run's default endpoint, model, and profile.", ""]
    for role, description in descriptions.items():
        table_name = f'"{role}"' if role.startswith("unit.") else role
        lines += [f"# {role}: {description}", f"# [roles.{table_name}]"]
        for field, value in (("endpoint", '"http://127.0.0.1:8080/v1"'), ("model", '"model-name"'),
                             ("api_key_env", '"MODEL_API_KEY"'), ("slot", '"auto"'),
                             ("temperature", "1.0"), ("top_p", "1.0"), ("top_k", "40"),
                             ("min_p", "0.0"), ("max_tokens", "24000"), ("timeout_s", "1800"),
                             ("context_tokens", "65536"), ("tokenizer", '"http://127.0.0.1:8080/v1"'),
                             ("allow_remote", "false"), ("cost_input_per_mtok", "0.0"),
                             ("cost_cached_input_per_mtok", "0.0"), ("cost_output_per_mtok", "0.0")):
            lines.append(f"# {field} = {value}")
        lines.append("")
    return "\n".join(lines)

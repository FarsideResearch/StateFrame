"""Machine-readable descriptions of the command and run artifact contracts."""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TypedDict, get_args, get_origin, get_type_hints

from . import __version__


_SCHEMA = "stateframe-contract/1"
_EXIT_MEANINGS = {
    0: "success",
    1: "operation did not succeed",
    2: "usage error or operation failure",
    130: "interrupted",
}


class _Catalog(TypedDict):
    settings: list[dict[str, object]]
    cli: list[dict[str, object]]
    commands: list[dict[str, object]]
    outputs: list[dict[str, object]]
    events: list[dict[str, object]]
    notes: dict[str, str]


def _command_parsers(parser: argparse.ArgumentParser) -> list[tuple[str, argparse.ArgumentParser]]:
    root = parser.prog.replace("\\", "/").rsplit("/", 1)[-1]
    found = [(root, parser)]

    def visit(parent: argparse.ArgumentParser, prefix: tuple[str, ...]) -> None:
        for action in parent._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for name, child in sorted(action.choices.items()):
                path = (*prefix, name)
                found.append((" ".join(path), child))
                visit(child, path)

    visit(parser, ())
    return sorted(found, key=lambda item: item[0])


def _takes_value(action: argparse.Action) -> bool:
    return action.nargs != 0


def _is_plain_json(value: object) -> bool:
    if value is None or type(value) in (str, int, float, bool):
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            return False
        return True
    if isinstance(value, list):
        return all(_is_plain_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_plain_json(item)
                   for key, item in value.items())
    return False


def _default(action: argparse.Action) -> tuple[object, str | None]:
    value = action.default
    if (getattr(action, "contract_default_kind", None) == "runtime"
            or value is argparse.SUPPRESS
            or isinstance(value, Path)
            or (action.type is Path and value is not None and not isinstance(value, list))
            or not _is_plain_json(value)):
        return None, "runtime"
    if isinstance(value, str) and (
            PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()):
        return None, "runtime"
    return value, None


def _type_name(annotation: object) -> str | None:
    if annotation is Path:
        return "path"
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (list, tuple, set) and args:
        members = args[:-1] if args[-1] is Ellipsis else args
        names = {_type_name(member) for member in members}
        names.discard(None)
        if len(names) == 1:
            return f"array<{next(iter(names))}>"
    return None


def _value_type(action: argparse.Action) -> str:
    if not _takes_value(action):
        return "integer" if isinstance(action, argparse._CountAction) else "boolean"
    if action.type is not None:
        direct = _type_name(action.type)
        if direct is not None:
            return direct
        try:
            annotation = get_type_hints(action.type).get("return")
        except (AttributeError, NameError, TypeError, ValueError):
            annotation = None
        inferred = _type_name(annotation)
        if inferred is not None:
            return inferred
    if action.choices:
        choices = action.choices.keys() if isinstance(action.choices, Mapping) else action.choices
        names = {_type_name(type(choice)) for choice in choices}
        names.discard(None)
        if len(names) == 1:
            return next(iter(names))
    return "string"


def _choices(action: argparse.Action) -> list[object] | None:
    choices = action.choices
    if choices is None:
        return None
    values = list(choices.keys()) if isinstance(choices, Mapping) else list(choices)
    if not all(_is_plain_json(value) for value in values):
        return None
    if any(isinstance(value, str) and (
            PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute())
           for value in values):
        return None
    if isinstance(choices, (set, frozenset, Mapping)):
        values.sort(key=lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False))
    return values


def _repeatable(action: argparse.Action) -> bool:
    if isinstance(action, (argparse._AppendAction, argparse._AppendConstAction,
                           argparse._CountAction)):
        return True
    if action.option_strings:
        return False
    return action.nargs in ("*", "+", argparse.REMAINDER) or (
        type(action.nargs) is int and action.nargs > 1)


def _argument(action: argparse.Action) -> dict[str, object]:
    default, default_kind = _default(action)
    required_unless_env = getattr(action, "contract_required_unless_env", None)
    value: dict[str, object] = {
        "required": bool(action.required or required_unless_env),
        "repeatable": _repeatable(action),
        "choices": _choices(action),
        "default": default,
        "type": _value_type(action),
    }
    if required_unless_env is not None:
        value["required_unless_env"] = required_unless_env
    if default_kind is not None:
        value["default_kind"] = default_kind
    return value


def _positionals(parser: argparse.ArgumentParser) -> list[dict[str, object]]:
    return [
        {"name": action.dest, **_argument(action)}
        for action in parser._actions if not action.option_strings
    ]


def _mutually_exclusive_groups(parser: argparse.ArgumentParser) -> list[dict[str, object]]:
    return [
        {"required": bool(group.required),
         "members": sorted(option for action in group._group_actions
                           for option in action.option_strings)}
        for group in parser._mutually_exclusive_groups
    ]


def _exit_codes(name: str) -> list[dict[str, object]]:
    top = name.split(" ", 1)[0]
    codes = [0, 2, 130]
    if top in {"run", "resume", "rehydrate", "mutations"}:
        codes.insert(1, 1)
    return [{"code": code, "meaning": _EXIT_MEANINGS[code]} for code in codes]


def commands(parser: argparse.ArgumentParser) -> list[dict[str, object]]:
    """Describe every parser node and its local option flags, including global flags."""
    return _commands(parser, _catalog_document())


def _commands(parser: argparse.ArgumentParser, catalog: _Catalog) -> list[dict[str, object]]:
    nodes = _command_parsers(parser)
    flag_commands: dict[tuple[str, bool], set[str]] = {}
    local_flags: dict[str, dict[str, argparse.Action]] = {}
    for name, command_parser in nodes:
        local: dict[str, argparse.Action] = {}
        for action in command_parser._actions:
            takes_value = _takes_value(action)
            for flag in action.option_strings:
                local[flag] = action
                flag_commands.setdefault((flag, takes_value), set()).add(name)
        local_flags[name] = local

    promises = {row["name"]: row.get("promised_json") for row in catalog["commands"]}
    result = []
    for name, command_parser in nodes:
        flags = [
            {"name": flag, "commands": sorted(flag_commands[(flag, takes_value)]),
             "takes_value": takes_value, **_argument(action)}
            for flag, action in sorted(local_flags[name].items())
            for takes_value in (_takes_value(action),)
        ]
        result.append({
            "name": name,
            "flags": flags,
            "positionals": _positionals(command_parser),
            "mutually_exclusive_groups": _mutually_exclusive_groups(command_parser),
            "json_output": any(
                action.dest == "json" and "--json" in action.option_strings
                for action in command_parser._actions
            ),
            "promised_json": promises.get(name),
            "exit_codes": _exit_codes(name),
            "promised_lines": [],
        })
    _reject_absolute_paths(result)
    return result


def _catalog_document() -> _Catalog:
    path = Path(__file__).with_name("contract-outputs.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("contract output catalog must be an object")
    _reject_absolute_paths(document)
    result: dict[str, list[dict[str, object]]] = {}
    for section, identity, nested in (
        ("settings", "key", "flags"),
        ("cli", "name", None),
        ("outputs", "file", "fields"),
        ("events", "name", "fields"),
        ("commands", "name", None),
    ):
        rows = document.get(section)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"contract output catalog has invalid {section}")
        result[section] = _sorted_rows(rows, identity, nested=nested)
    notes = document.get("notes")
    if (not isinstance(notes, dict)
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in notes.items())
            or notes.get("settings") != "pending"):
        raise ValueError("contract output catalog has invalid notes")
    return _Catalog(settings=result["settings"], cli=result["cli"],
                    commands=result["commands"], outputs=result["outputs"],
                    events=result["events"], notes=dict(notes))


def schema_version() -> str:
    """Return only the numeric suffix of the public contract schema identifier."""
    return _SCHEMA.rsplit("/", 1)[1]


def document(parser: argparse.ArgumentParser) -> dict[str, object]:
    """Build the complete, path-safe command and artifact contract."""
    catalog = _catalog_document()
    return assemble(settings=catalog["settings"], commands=_commands(parser, catalog),
                    outputs=catalog["outputs"], events=catalog["events"], notes=catalog["notes"])


def cli_snapshot() -> list[dict[str, object]]:
    """Return the checked-in CLI flags and exit-code contract used by regression tests."""
    return _catalog_document()["cli"]


def readable_summary(contract: Mapping[str, object]) -> str:
    """Render a short human-readable summary without exposing paths or runtime details."""
    schema = contract.get("schema")
    commands_value = contract.get("commands")
    outputs_value = contract.get("outputs")
    events_value = contract.get("events")
    if not all(isinstance(value, list) for value in (commands_value, outputs_value, events_value)):
        raise ValueError("contract summary requires command, output and event rows")
    root = "stateframe"
    command_count = sum(
        1 for row in commands_value
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
        and row["name"] != root and " " not in row["name"]
    )
    schema_number = schema.rsplit("/", 1)[-1] if isinstance(schema, str) else "unknown"
    return (f"Contract schema {schema_number}: {command_count} commands, "
            f"{len(outputs_value)} artifacts, {len(events_value)} event types.")


def outputs() -> list[dict[str, object]]:
    """Read the checked-in list of documented run artifacts."""
    return _catalog_document()["outputs"]


def events() -> list[dict[str, object]]:
    """Read the checked-in list of documented event fields."""
    return _catalog_document()["events"]


def _sorted_rows(
    rows: Iterable[Mapping[str, object]], identity: str, *, nested: str | None = None,
) -> list[dict[str, object]]:
    result = [dict(row) for row in rows]
    for row in result:
        if not isinstance(row.get(identity), str):
            raise ValueError(f"contract row is missing string {identity!r}")
        if nested is not None and isinstance(row.get(nested), list):
            children = row[nested]
            if any(not isinstance(child, dict) for child in children):
                raise ValueError(f"contract row has invalid {nested}")
            row[nested] = sorted((dict(child) for child in children), key=lambda child: str(child.get("name", "")))
    return sorted(result, key=lambda row: str(row[identity]))


def _reject_absolute_paths(value: object) -> None:
    if isinstance(value, str):
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError("contract data cannot contain absolute paths")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _reject_absolute_paths(key)
            _reject_absolute_paths(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_absolute_paths(child)


def _sorted_keys(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("contract object keys must be strings")
        return {key: _sorted_keys(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_keys(child) for child in value]
    if isinstance(value, tuple):
        return [_sorted_keys(child) for child in value]
    return value


def assemble(
    settings: Iterable[Mapping[str, object]],
    commands: Iterable[Mapping[str, object]],
    outputs: Iterable[Mapping[str, object]],
    events: Iterable[Mapping[str, object]],
    notes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Assemble and normalize the agreed contract shape from supplied rows."""
    setting_rows = _sorted_rows(settings, "key", nested="flags")
    for row in setting_rows:
        row.setdefault("file_allowed", None)
        row.setdefault("project_safe", None)
        aliases = row.get("aliases")
        if isinstance(aliases, list):
            row["aliases"] = sorted(
                (dict(alias) for alias in aliases if isinstance(alias, Mapping)),
                key=lambda alias: (str(alias.get("kind", "")), str(alias.get("name", ""))),
            )
    result: dict[str, object] = {
        "schema": _SCHEMA,
        "stateframe_version": __version__,
        "settings": setting_rows,
        "commands": _sorted_rows(commands, "name", nested="flags"),
        "outputs": _sorted_rows(outputs, "file", nested="fields"),
        "events": _sorted_rows(events, "name", nested="fields"),
    }
    if notes is not None:
        result["notes"] = dict(notes)
    _reject_absolute_paths(result)
    ordered = _sorted_keys(result)
    if not isinstance(ordered, dict):
        raise TypeError("contract must assemble to an object")
    return ordered


def render(contract: Mapping[str, object]) -> bytes:
    """Render a contract as canonical UTF-8 JSON with one trailing newline."""
    _reject_absolute_paths(contract)
    ordered = _sorted_keys(contract)
    return (json.dumps(ordered, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _records(value: object, identity: str) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, list):
        return {}
    return {
        row[identity]: row for row in value
        if isinstance(row, Mapping) and isinstance(row.get(identity), str)
    }


def _aliases(contract: Mapping[str, object], kind: str) -> set[str]:
    names = set()
    for setting in _records(contract.get("settings"), "key").values():
        aliases = setting.get("aliases")
        if isinstance(aliases, list):
            names.update(
                alias["name"] for alias in aliases
                if isinstance(alias, Mapping) and alias.get("kind") == kind
                and isinstance(alias.get("name"), str)
            )
    return names


def _spelling_exists(contract: Mapping[str, object], kind: str, name: str) -> bool:
    if name in _aliases(contract, kind):
        return True
    for setting in _records(contract.get("settings"), "key").values():
        if kind == "env" and setting.get("env") == name:
            return True
        if kind == "flag" and name in _records(setting.get("flags"), "name"):
            return True
    return False


def _field_severity(stable: object, change: str) -> str:
    return "break" if stable is True or (stable == "type" and change == "type_changed") else "notice"


def compare(previous: Mapping[str, object], current: Mapping[str, object]) -> list[dict[str, str]]:
    """List removed or structurally repurposed names that lack a current alias."""
    _reject_absolute_paths(previous)
    _reject_absolute_paths(current)
    changes: list[dict[str, str]] = []

    def add(kind: str, name: str, change: str, severity: str | None = None) -> None:
        row = {"kind": kind, "name": name, "change": change}
        if severity is not None:
            row["severity"] = severity
        changes.append(row)

    old_settings = _records(previous.get("settings"), "key")
    new_settings = _records(current.get("settings"), "key")
    flag_aliases, env_aliases = _aliases(current, "flag"), _aliases(current, "env")
    for name, old in old_settings.items():
        new = new_settings.get(name)
        if new is None:
            add("setting", name, "removed")
            continue
        if any(old.get(key) != new.get(key) for key in ("type", "section", "class", "meaning")):
            add("setting", name, "repurposed")
        old_env, new_env = old.get("env"), new.get("env")
        if isinstance(old_env, str) and old_env != new_env and old_env not in env_aliases:
            add("setting_env", old_env, "removed")
        old_flags, new_flags = _records(old.get("flags"), "name"), _records(new.get("flags"), "name")
        for flag, old_flag in old_flags.items():
            new_flag = new_flags.get(flag)
            if new_flag is None:
                if flag not in flag_aliases:
                    add("setting_flag", f"{name} {flag}", "removed")
            elif any(old_flag.get(key) != new_flag.get(key) for key in ("takes_value", "commands")):
                add("setting_flag", f"{name} {flag}", "repurposed")
        old_aliases = old.get("aliases")
        if isinstance(old_aliases, list):
            for alias in old_aliases:
                if (isinstance(alias, Mapping) and alias.get("kind") in {"env", "flag"}
                        and isinstance(alias.get("name"), str)
                        and not _spelling_exists(current, str(alias["kind"]), alias["name"])):
                    add("setting_alias", f"{alias['kind']}:{alias['name']}", "removed")

    old_commands = _records(previous.get("commands"), "name")
    new_commands = _records(current.get("commands"), "name")
    for name, old in old_commands.items():
        new = new_commands.get(name)
        if new is None:
            add("command", name, "removed")
            continue
        if old.get("json_output") != new.get("json_output"):
            add("command", name, "repurposed")
        if old.get("exit_codes") != new.get("exit_codes"):
            add("command_exit_codes", name, "changed")
        old_flags, new_flags = _records(old.get("flags"), "name"), _records(new.get("flags"), "name")
        for flag, old_flag in old_flags.items():
            new_flag = new_flags.get(flag)
            if new_flag is None:
                if flag not in flag_aliases:
                    add("command_flag", f"{name} {flag}", "removed")
            elif any(old_flag.get(key) != new_flag.get(key) for key in ("takes_value", "commands")):
                add("command_flag", f"{name} {flag}", "repurposed")
        old_json, new_json = old.get("promised_json"), new.get("promised_json")
        if isinstance(old_json, Mapping):
            old_fields = _records(old_json.get("fields"), "name")
            new_fields = _records(new_json.get("fields"), "name") if isinstance(new_json, Mapping) else {}
            for field, old_field in old_fields.items():
                new_field = new_fields.get(field)
                label = f"{name}#{field}"
                if new_field is None:
                    add("command_json_field", label, "removed",
                        _field_severity(old_field.get("stable"), "removed"))
                elif old_field.get("type") != new_field.get("type"):
                    add("command_json_field", label, "type_changed",
                        _field_severity(old_field.get("stable"), "type_changed"))
                elif old_field.get("stable") != new_field.get("stable"):
                    add("command_json_field", label, "stability_changed", "notice")

    for section, identity, kind in (("outputs", "file", "output"), ("events", "name", "event")):
        old_rows = _records(previous.get(section), identity)
        new_rows = _records(current.get(section), identity)
        for name, old in old_rows.items():
            new = new_rows.get(name)
            if new is None:
                add(kind, name, "removed")
                continue
            if section == "outputs" and old.get("format") != new.get("format"):
                add(kind, name, "repurposed")
            old_fields = _records(old.get("fields"), "name")
            new_fields = _records(new.get("fields"), "name")
            field_kind = f"{kind}_field"
            for field, old_field in old_fields.items():
                new_field = new_fields.get(field)
                if new_field is None:
                    add(field_kind, f"{name}#{field}", "removed",
                        _field_severity(old_field.get("stable"), "removed"))
                elif old_field.get("type") != new_field.get("type"):
                    add(field_kind, f"{name}#{field}", "type_changed",
                        _field_severity(old_field.get("stable"), "type_changed"))
                elif old_field.get("stable") != new_field.get("stable"):
                    add(field_kind, f"{name}#{field}", "stability_changed", "notice")

    return sorted(changes, key=lambda row: (row["kind"], row["name"], row["change"]))

"""Host-authorized tool settings, handed to the engine as a private run file."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = "stateframe-tool-settings/1"
MODES = frozenset(("off", "read-only", "standard", "unrestricted"))
NETWORKS = frozenset(("none", "loopback", "full"))


class ToolSettingsError(ValueError):
    """A host setting is missing, invalid, or has been changed."""


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ToolSettingsError(f"tool settings: duplicate JSON key {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class ToolSettings:
    shell: str = "standard"
    shell_ask: bool = False
    shell_network: str = "loopback"
    shell_allow: tuple[str, ...] = ()
    shell_deny: tuple[str, ...] = ()
    shell_read: tuple[str, ...] = ()
    host_authorization: dict[str, object] | None = None
    run_env: bool = True
    git_tool: bool = True
    web: bool = False
    web_allow: tuple[str, ...] = ()
    tools_files: tuple[dict[str, str], ...] = ()
    mcp_files: tuple[dict[str, str], ...] = ()
    replay_external: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.shell, str) or self.shell not in MODES:
            raise ToolSettingsError("shell: expected off, read-only, standard, or unrestricted")
        if not isinstance(self.shell_network, str) or self.shell_network not in NETWORKS:
            raise ToolSettingsError("shell_network: expected none, loopback, or full")
        for key in ("shell_ask", "run_env", "git_tool", "web", "replay_external"):
            if type(getattr(self, key)) is not bool:
                raise ToolSettingsError(f"{key}: expected boolean")
        for key in ("shell_allow", "shell_deny", "shell_read", "web_allow"):
            value = getattr(self, key)
            if not isinstance(value, tuple) or any(not isinstance(item, str) or not item for item in value):
                raise ToolSettingsError(f"{key}: expected nonempty strings")
        if self.host_authorization is not None:
            auth = self.host_authorization
            if (self.shell != "unrestricted" or not isinstance(auth, dict)
                    or set(auth) != {"granted_at", "flags"}
                    or not isinstance(auth["granted_at"], str)
                    or auth["flags"] != ["--shell unrestricted", "--shell-sandbox off"]):
                raise ToolSettingsError("host_authorization: requires explicit unrestricted host mode")
        if not isinstance(self.tools_files, tuple) or any(
                not isinstance(item, dict) or set(item) != {"path", "digest"}
                or not all(isinstance(v, str) and v for v in item.values())
                for item in self.tools_files):
            raise ToolSettingsError("tools_files: expected path and digest records")
        if not isinstance(self.mcp_files, tuple) or any(
                not isinstance(item, dict) or set(item) != {"path", "digest"}
                or not all(isinstance(v, str) and v for v in item.values())
                for item in self.mcp_files):
            raise ToolSettingsError("mcp_files: expected path and digest records")

    def as_dict(self) -> dict[str, object]:
        # JSON-canonical form (tuples as lists): the child compares this against the spec's JSON copy.
        values = asdict(self)
        if not self.mcp_files:
            values.pop("mcp_files")  # Preserve the 0.2.0 authority digest for runs without MCP.
        return json.loads(json.dumps({"schema": SCHEMA, **values}))

    @classmethod
    def from_dict(cls, data: object) -> "ToolSettings":
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise ToolSettingsError("tool settings: expected stateframe-tool-settings/1")
        names = set(cls.__dataclass_fields__)
        expected = (names | {"schema"}) if "mcp_files" in data else ((names - {"mcp_files"}) | {"schema"})
        if set(data) != expected:
            raise ToolSettingsError("tool settings: missing or unknown fields: " +
                                    ", ".join(sorted(set(data) ^ expected)))
        values = dict(data)
        del values["schema"]
        values.setdefault("mcp_files", [])
        for key in ("shell_allow", "shell_deny", "shell_read", "web_allow", "tools_files", "mcp_files"):
            if not isinstance(values[key], list):
                raise ToolSettingsError(f"{key}: expected list")
            values[key] = tuple(values[key])
        return cls(**values)


def load_tool_settings() -> ToolSettings | None:
    """Read only the host's declared file; malformed data is a startup refusal."""
    name = os.environ.get("STATEFRAME_TOOL_SETTINGS_FILE")
    if name is None:
        return None
    if not name or not Path(name).is_absolute():
        raise ToolSettingsError("STATEFRAME_TOOL_SETTINGS_FILE: expected absolute path")
    try:
        data = json.loads(Path(name).read_text(encoding="utf-8"), object_pairs_hook=_json_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ToolSettingsError(f"STATEFRAME_TOOL_SETTINGS_FILE: {error}") from error
    return ToolSettings.from_dict(data)

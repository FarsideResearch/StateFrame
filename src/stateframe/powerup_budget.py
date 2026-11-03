"""Keep user tool and MCP schemas ahead of optional powerup schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from .engine.kernel_custom_tools import CustomToolError, load_tool_files
from .engine.kernel_hooks import HookConfigError, load_hook_files
from .engine.kernel_request_diet import RequestTokenMeter
from .powerup_manifest import PowerupError
from .powerup_materialize import _read_json
from .workcopy import WorkCopyError


def admit_powerup_units(
    selection,
    *,
    user_tools: Sequence[Path],
    mcp_schemas: Sequence[dict],
    shell_network: str,
    endpoint: str | None,
    user_mcp: Sequence[Path] = (),
    user_hooks: Sequence[Path] = (),
    repo: Path | None = None,
    mcp_omitted: Sequence[str] = (),
    discover_mcp_schemas: Callable[
        [Sequence[Path]], tuple[Sequence[dict], Sequence[str]]
    ]
    | None = None,
    printer=print,
) -> None:
    """Admit unit schemas after already-discovered user tools and MCP schemas."""
    from .powerups import BUILTINS

    meter = RequestTokenMeter(endpoint=endpoint, template_overhead_tokens=0)
    accepted = list(user_tools)
    accepted_mcp = list(user_mcp)
    accepted_hooks = list(user_hooks)
    current_schemas = list(mcp_schemas)
    current_omitted = set(mcp_omitted)
    powerup_count = 0
    kept = []
    for unit in selection.active:
        if any(
            item["executor"] == "mutation" for item in getattr(unit, "attachments", ())
        ):
            kept.append(unit)
            continue
        candidate = [*accepted]
        candidate_mcp = [*accepted_mcp]
        candidate_hooks = [*accepted_hooks]
        schemas = current_schemas
        omitted = current_omitted
        added = 0
        try:
            if "hooks" in unit.parts:
                if repo is None:
                    raise PowerupError("powerup hook admission requires a repository")
                candidate_hooks.append(unit.materialized_dir / "hooks.json")
                load_hook_files(candidate_hooks, repo=repo, shell_network=shell_network)
            if "commands" in unit.parts or "mcp" in unit.parts:
                if "commands" in unit.parts:
                    candidate.append(unit.materialized_dir / "commands.json")
                custom = load_tool_files(
                    candidate, builtin_names=BUILTINS, shell_network=shell_network
                )
                if "mcp" in unit.parts:
                    if discover_mcp_schemas is None:
                        raise PowerupError("powerup MCP discovery is unavailable")
                    candidate_mcp.append(unit.materialized_dir / "mcp.json")
                    schemas, reported = discover_mcp_schemas(candidate_mcp)
                    omitted = set(reported)
                    if omitted - current_omitted:
                        raise PowerupError(
                            "powerup MCP discovery omitted tools: "
                            + ", ".join(sorted(omitted - current_omitted))
                        )
                budget_custom = custom.schemas()
                custom_names = {schema["function"]["name"] for schema in budget_custom}
                mcp_names = {schema["function"]["name"] for schema in schemas}
                collisions = sorted(custom_names & mcp_names)
                if collisions:
                    raise PowerupError(
                        f"powerup schema name collides with MCP tool: {collisions[0]}"
                    )
                added = (
                    len(_read_json(unit.materialized_dir / "commands.json")["tools"])
                    if "commands" in unit.parts
                    else 0
                ) + (len(schemas) - len(current_schemas) if "mcp" in unit.parts else 0)
                if powerup_count + added > 6:
                    raise PowerupError("maximum 6 powerup tools")
                for schema in budget_custom:
                    schema["function"]["parameters"]["properties"].setdefault(
                        "why", {"type": "string"}
                    )
                tokens, _ = meter.measure(
                    {"messages": [], "tools": [*budget_custom, *schemas]}
                )
                if tokens > 1500:
                    raise PowerupError(
                        f"combined user MCP and custom schemas: {tokens} tokens exceeds 1500"
                    )
        except (
            CustomToolError,
            HookConfigError,
            PowerupError,
            ValueError,
            TimeoutError,
            OSError,
            WorkCopyError,
        ) as error:
            reason = str(error)
            selection.skipped.append({"name": unit.name, "reason": reason})
            printer(f"powerup {unit.name} disabled: {reason}")
            continue
        accepted = candidate
        accepted_mcp = candidate_mcp
        accepted_hooks = candidate_hooks
        current_schemas = list(schemas)
        current_omitted = omitted
        powerup_count += added
        kept.append(unit)
    selection.active = kept

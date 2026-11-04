"""Resolve optional powerup inputs before the runner loads ordinary run inputs."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from . import hook_host, powerups, workcopy
from .engine.kernel_custom_tools import load_tool_files
from .engine.kernel_mcp_tools import discover as discover_mcp, load_mcp_files
from .powerup_budget import admit_powerup_units
from .powerup_state import user_home_or_none
from .skills import NAME, resolve_skills

NO_HOME_NOTICE = (
    "stateframe: no home directory found (HOME and STATEFRAME_HOME unset); "
    "user powerups and skills are skipped for this run"
)


def prepare_run_inputs(options, environment: dict, printer=print) -> powerups.Selection:
    """Append admitted powerups; keep the unconfigured path free of optional work."""
    if options.no_powerups and not options.skills:
        return powerups.Selection()
    user_home = user_home_or_none()
    if user_home is None and (not options.no_powerups or any(
            NAME.fullmatch(value) and not Path(value).is_dir() for value in options.skills)):
        print(NO_HOME_NOTICE, file=sys.stderr)
    if options.no_powerups:
        return powerups.Selection()
    project_root = Path(options.repo) / ".stateframe"
    if (not options.powerup_requests and not options.powerup_sets
            and (user_home is None or (not (user_home / "powerups.json").exists()
                                       and not (user_home / "powerups").exists()))
            and not (project_root / "powerups.json").exists()
            and not (project_root / "powerups").exists()):
        return powerups.Selection()

    prior_skill_tools = ()
    if options.trust_skill_tools:
        prior_skill_tools = tuple(
            skill.path / "tools.json" for skill in resolve_skills(options.skills, trusted=True)
            if skill.tools_json_sha256
        )
    agent_loop = str(options.extra_env.get("STATEFRAME_AGENT_LOOP",
                         environment.get("STATEFRAME_AGENT_LOOP", "full"))).lower()
    lean = str(options.extra_env.get("STATEFRAME_LEAN",
               environment.get("STATEFRAME_LEAN", "off"))).lower()
    loop = "plain" if agent_loop == "plain" else "lean" if lean == "on" else "composed"
    user_tool_paths = (*[Path(item["path"]) for item in options.tool_settings.tools_files],
                       *prior_skill_tools)
    user_mcp_paths = [Path(item["path"]) for item in options.tool_settings.mcp_files]
    selection = powerups.resolve_units(
        options.repo, options.task, options.profile, options.powerup_requests,
        options.powerup_sets, False, loop=loop,
        shell_network=options.tool_settings.shell_network,
        user_tools=user_tool_paths, user_mcp=user_mcp_paths,
        user_hooks=options.hooks_files, user_skills=len(options.skills), printer=printer,
    )
    if not selection.active:
        return selection
    # Reject unavailable executors before optional budget discovery can start a server.
    powerups.filter_unavailable_executors(selection, printer=printer)
    if any({"commands", "mcp", "hooks"}.intersection(unit.parts) for unit in selection.active):
        prior_hook_paths = options.hooks_files
        if any("hooks" in unit.parts for unit in selection.active):
            hook_repo = (options.hooks_repo or options.repo).resolve()
            if not options.project_hooks_resolved:
                project = hook_host.resolve_project_hooks(
                    hook_repo, shell_network=options.tool_settings.shell_network,
                    no_project_hooks=options.no_project_hooks,
                )
                options.project_hooks = project.hooks
                options.project_hooks_digest = project.digest
                options.project_hooks_resolved = True
                if project.message:
                    printer(project.message)
            if options.project_hooks is not None:
                prior_hook_paths = (*prior_hook_paths,
                    *(Path(item["path"]) for item in options.project_hooks.files))
        user_custom = load_tool_files(
            user_tool_paths, builtin_names=powerups.BUILTINS,
            shell_network=options.tool_settings.shell_network,
        )

        def discover_powerup_mcp(paths):
            servers = load_mcp_files(paths, shell_network=options.tool_settings.shell_network)
            # Discovery uses the repository snapshot and a scratch run root. The real run
            # has a different workspace and run root, so its discovery result cannot be
            # reused without changing the engine's discovery and execution boundary.
            with tempfile.TemporaryDirectory(prefix="stateframe-powerup-mcp-") as scratch:
                scratch_root = Path(scratch)
                snapshot = workcopy.create(options.repo, scratch_root / "workspace")
                discovered = discover_mcp(
                    servers, snapshot.copy, scratch_root,
                    shell_network=options.tool_settings.shell_network,
                    host_authorization=options.tool_settings.host_authorization is not None,
                    custom_schemas=user_custom.schemas(), endpoint=options.endpoint.chat_url,
                )
            return discovered.schemas(), discovered.omitted

        needs_schema_budget = any({"commands", "mcp"}.intersection(unit.parts)
                                  for unit in selection.active)
        user_schemas, user_omitted = (
            discover_powerup_mcp(user_mcp_paths)
            if user_mcp_paths and needs_schema_budget else ([], []))
        admit_powerup_units(
            selection, user_tools=user_tool_paths, mcp_schemas=user_schemas,
            user_mcp=user_mcp_paths, user_hooks=prior_hook_paths,
            repo=(options.hooks_repo or options.repo).resolve(), mcp_omitted=user_omitted,
            discover_mcp_schemas=discover_powerup_mcp,
            shell_network=options.tool_settings.shell_network,
            endpoint=options.endpoint.chat_url, printer=printer,
        )
    powerups.attach_units(options, selection, printer=printer)
    return selection

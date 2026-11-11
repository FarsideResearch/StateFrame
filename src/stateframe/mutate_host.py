"""Product-side mutation policy resolution and lazy engine lifecycle."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Mapping

if __package__:
    from .engine.kernel_safe_paths import safe_expanduser, safe_home
    from .state_dirs import private_directory
else:
    from engine.kernel_safe_paths import safe_expanduser, safe_home
    from state_dirs import private_directory


LEVEL_ENV = "STATEFRAME_MUTATIONS"
LEVELS = ("off", "prebuilt", "authored")
PLATFORM_UNSUPPORTED = (
    "MUTATION_PLATFORM_UNSUPPORTED: mutations are Linux-only in this release; macOS is planned"
)


def mutation_platform_error(platform_name: str, *, requested: bool) -> str | None:
    """Return the stable refusal for mutation work outside the shipped platform."""
    return PLATFORM_UNSUPPORTED if requested and platform_name != "linux" else None


def spec_requests_mutations(spec: Mapping[str, object]) -> bool:
    """Report whether a saved run carries mutation options or an enabled mutation level."""
    if isinstance(spec.get("mutations"), Mapping):
        return True
    environment = spec.get("environment")
    if not isinstance(environment, Mapping):
        return False
    level = environment.get(LEVEL_ENV, "off")
    return isinstance(level, str) and level.strip().lower() != "off"


def level_from(environment: Mapping[str, str] | None = None) -> str:
    source = os.environ if environment is None else environment
    level = source.get(LEVEL_ENV, "off")
    if level not in LEVELS:
        raise ValueError(f"{LEVEL_ENV} must be off, prebuilt or authored")
    return level


def qualify_runtime(home: Path, *, refresh: bool = False) -> object:
    from kernel_mutate_exec import MutationPlatformError
    from kernel_mutate_qualify import qualify

    result = qualify(Path(home), refresh=refresh)
    if not result.ok:
        checks = (result.record.get("report") or {}).get("checks") or []
        failed = [f"{check.get('name')} [{check.get('reason', 'MUTATION_PLATFORM_UNQUALIFIED_FAILURE')}]: "
                  f"{check.get('detail')}" for check in checks
                  if isinstance(check, dict) and not check.get("ok")]
        raise MutationPlatformError("; ".join(failed or result.reasons) or "selftest failed")
    return result


def validate_ceiling(options: Mapping[str, object], tool_settings: Mapping[str, object]) -> None:
    if options.get("network") == "public-web" and tool_settings.get("shell_network") != "full":
        raise ValueError("mutation network requires --shell-network full")


def start(spec: Mapping[str, object], run: Path) -> object | None:
    options = spec.get("mutations")
    if not isinstance(options, Mapping):
        return None
    refusal = mutation_platform_error(sys.platform, requested=True)
    if refusal is not None:
        raise RuntimeError(refusal)
    from kernel_mutate_policy import policy_from
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime, install
    from kernel_mutate_sweep import sweep

    policy = policy_from(options)
    if not policy.enabled:
        return None
    validate_ceiling(options, spec.get("tool_settings") or {})
    configured_home = os.environ.get("STATEFRAME_HOME")
    if configured_home is not None:
        state_home = safe_expanduser(Path(configured_home))
    else:
        user_home = safe_home()
        if user_home is None:
            raise RuntimeError("cannot determine home directory; set HOME or STATEFRAME_HOME")
        state_home = user_home / ".stateframe"
    runtime_parent = private_directory(state_home, "rt", "mutations")
    run_id = str(spec["run_identity_seed"])
    # Another run may still own its supervisor. Sweep only this run's old root here.
    report = sweep(runtime_parent, run_locks=lambda candidate: Path(candidate).name != run_id)
    if report.unresolved:
        raise RuntimeError("MUTATION_CONTAINMENT: unresolved earlier launches")
    journal = Journal(Path(run))
    row = spec.get("public_row") or {}
    scope = row.get("scope") if isinstance(row, Mapping) else None
    declared_reads = scope.get("declared_paths") if isinstance(scope, Mapping) else None
    if declared_reads is None:
        ordinary_read_globs = None
    elif (isinstance(declared_reads, (tuple, list))
          and all(isinstance(path, str) and path for path in declared_reads)):
        ordinary_read_globs = tuple(declared_reads)
    else:
        raise ValueError("invalid ordinary read scope for mutation inputs")

    def executor_factory():
        from kernel_mutate_exec import MutationExecutor

        qualify_runtime(state_home)
        root = private_directory(state_home, "rt", "mutations", str(spec["run_identity_seed"]))
        return MutationExecutor(root, network_envelope=(
            "full" if policy.network == "public-web" else "none"))

    runtime = MutationRuntime(level=policy.level, policy=policy, journal=journal,
                              run_dir=Path(run), executor_factory=executor_factory,
                              edit_scope=(set((spec.get("public_row") or {}).get("files", ()))
                                          if (spec.get("public_row") or {}).get("scoped_edits") else None),
                              protected_paths=set((spec.get("public_row") or {}).get("protected_test_paths", ())),
                              effect_row=spec.get("public_row"),
                              replay_external_allowed=bool((spec.get("tool_settings") or {}).get(
                                  "replay_external", False)),
                              ordinary_read_globs=ordinary_read_globs)
    hooks = options.get("hooks", ())
    if hooks:
        if policy.level != "authored":
            raise ValueError("mutation hook attachments require the authored level")
        from kernel_mutate_store import MutationStore, project_id
        from kernel_mutate_wire import strict_loads

        store = MutationStore(state_home)
        scopes = ("user", project_id(Path(spec["repo"])))
        retained = {}
        for scope in scopes:
            with store._locked():
                digests = store._read_index(store._index_path(scope))
            for digest in digests:
                bundle = store.get(digest)
                manifest = strict_loads(bundle["unit.json"])
                retained[manifest["name"]] = {"manifest": manifest,
                    "files": {name: data for name, data in bundle.items() if name != "unit.json"},
                    "source_digest": "sha256:" + digest}
        for item in hooks:
            name, separator, event = item.partition(":")
            if not separator or event not in ("post_tool", "session_end") or name not in retained:
                raise ValueError("mutation hook attachment must name a retained observing unit")
            runtime.ready[name] = retained[name]
            runtime.attach_hook(name, event=event)
    runtime.attempt = int((spec.get("resume") or {}).get("attempt", 1))
    install(runtime)
    return runtime

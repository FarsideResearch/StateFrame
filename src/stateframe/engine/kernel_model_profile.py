"""Named, opt-in guardrail profiles for weaker and stronger model lanes.

The small profile is deliberately the default.  This module contains only
configuration and pure helpers so importing it cannot change an existing
request, frame, or tool surface unless a caller explicitly selects ``large``.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


PROFILE_ENV = "STATEFRAME_MODEL_PROFILE"
SMALL_PROFILE = "small"
LARGE_PROFILE = "large"

RELAXATION_ENV = {
    "green_menu": "STATEFRAME_PROFILE_GREEN_MENU",
    "hypothesis_filter": "STATEFRAME_PROFILE_HYPOTHESIS_FILTER",
    "progress_stall": "STATEFRAME_PROFILE_PROGRESS_STALL",
    "edit_scope": "STATEFRAME_PROFILE_EDIT_SCOPE",
    "frame_scale": "STATEFRAME_PROFILE_FRAME_SCALE",
    "gap_list": "STATEFRAME_PROFILE_GAP_LIST",
}
RELAXATIONS = tuple(RELAXATION_ENV)

# A few descriptive aliases make the profile easy to drive from older harness
# launchers while keeping the canonical switch names above stable.
_ALIASES = {
    key: (name, "STATEFRAME_LARGE_" + key.upper(),
          "STATEFRAME_PROFILE_RELAX_" + key.upper())
    for key, name in RELAXATION_ENV.items()
}


def _profile_name() -> str:
    value = os.environ.get(PROFILE_ENV, SMALL_PROFILE).strip().lower()
    if value not in {SMALL_PROFILE, LARGE_PROFILE}:
        raise ValueError(f"{PROFILE_ENV} must be small or large")
    return value


def _switch_value(name: str, *, default: bool) -> bool:
    for env_name in _ALIASES[name]:
        if env_name not in os.environ:
            continue
        value = os.environ[env_name].strip().lower()
        if value in {"1", "on", "true", "yes"}:
            return True
        if value in {"0", "off", "false", "no"}:
            return False
        raise ValueError(f"{env_name} must be on or off")
    return default


@dataclass(frozen=True)
class ModelProfile:
    """Resolved profile and its independently A/B-able relaxations."""

    name: str
    relaxations: Mapping[str, bool]

    def enabled(self, relaxation: str) -> bool:
        try:
            return bool(self.relaxations[relaxation])
        except KeyError as error:
            raise ValueError(f"unknown model-profile relaxation: {relaxation}") from error


def model_profile() -> ModelProfile:
    name = _profile_name()
    # ``large`` opts into every named relaxation.  A switch set to ``off``
    # selects the corresponding small-model guardrail for an A/B run.
    relaxations = MappingProxyType({
        key: _switch_value(key, default=name == LARGE_PROFILE)
        if name == LARGE_PROFILE else False
        for key in RELAXATIONS
    })
    return ModelProfile(name, relaxations)


def profile() -> ModelProfile:
    """Compatibility spelling for callers that prefer a shorter accessor."""
    return model_profile()


def relaxation_enabled(name: str) -> bool:
    return model_profile().enabled(name)


def green_menu_relaxed() -> bool:
    return relaxation_enabled("green_menu")


def hypothesis_filter_relaxed() -> bool:
    return relaxation_enabled("hypothesis_filter")


def progress_stall_relaxed() -> bool:
    return relaxation_enabled("progress_stall")


def edit_scope_relaxed() -> bool:
    return relaxation_enabled("edit_scope")


def frame_scale_enabled() -> bool:
    return relaxation_enabled("frame_scale")


def gap_list_relaxed() -> bool:
    return relaxation_enabled("gap_list")


def _positive_int(value: object, name: str) -> int:
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _ratio(env_name: str, default: float) -> float:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{env_name} must be between 0 and 1") from error
    if not 0 < value <= 1:
        raise ValueError(f"{env_name} must be between 0 and 1")
    return value


def _context_size(context_size: object = None) -> int | None:
    if context_size is not None:
        return _positive_int(context_size, "context_size")
    for name in ("STATEFRAME_CONTEXT_SIZE", "STATEFRAME_MODEL_CONTEXT_SIZE",
                 "STATEFRAME_CONTEXT_TOKENS", "STATEFRAME_MODEL_CONTEXT_TOKENS",
                 "STATEFRAME_CONTEXT_WINDOW", "STATEFRAME_MAX_CONTEXT"):
        if name in os.environ:
            return _positive_int(os.environ[name], name)
    return None


def frame_budgets(*, default_target: int = 4096, default_ceiling: int = 8192,
                  context_size: object = None) -> tuple[int, int]:
    """Return target/ceiling budgets while preserving legacy env precedence.

    Explicit ``STATEFRAME_TARGET`` and ``STATEFRAME_CEILING`` retain their old
    meaning.  The named large-profile frame relaxation only derives absent
    values from a supplied context size; without that parameter it is a no-op.
    """
    target_explicit = "STATEFRAME_TARGET" in os.environ
    ceiling_explicit = "STATEFRAME_CEILING" in os.environ
    target = (_positive_int(os.environ["STATEFRAME_TARGET"], "STATEFRAME_TARGET")
              if target_explicit else _positive_int(default_target, "default_target"))
    ceiling = (_positive_int(os.environ["STATEFRAME_CEILING"], "STATEFRAME_CEILING")
               if ceiling_explicit else _positive_int(default_ceiling, "default_ceiling"))
    if frame_scale_enabled() and not (target_explicit and ceiling_explicit):
        size = _context_size(context_size)
        if size is not None:
            ceiling = (_positive_int(round(size * _ratio(
                "STATEFRAME_FRAME_CEILING_RATIO", 0.30)),
                                     "scaled frame ceiling")
                       if not ceiling_explicit else ceiling)
            if not target_explicit:
                target = _positive_int(round(size * _ratio(
                    "STATEFRAME_FRAME_TARGET_RATIO", 0.15)),
                                       "scaled frame target")
    if not 0 < target <= ceiling:
        raise ValueError("invalid StateFrame budgets")
    return target, ceiling


def ordered_terminal_menu(tools: Sequence[Mapping[str, object]]) -> list[dict]:
    """Return every offered tool, stably ordered for a terminal hand-off."""
    rows = [dict(tool) for tool in tools]
    priority = {"run_tests": 0, "done": 1}
    return sorted(rows, key=lambda tool: priority.get(
        str(tool.get("function", {}).get("name", "")), 2))


__all__ = [
    "LARGE_PROFILE", "ModelProfile", "PROFILE_ENV", "RELAXATION_ENV", "RELAXATIONS",
    "SMALL_PROFILE", "edit_scope_relaxed", "frame_budgets", "frame_scale_enabled",
    "gap_list_relaxed", "green_menu_relaxed", "hypothesis_filter_relaxed",
    "model_profile", "ordered_terminal_menu", "profile", "progress_stall_relaxed",
    "relaxation_enabled",
]

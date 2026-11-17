"""MR-3: model policy follows the SELECTED deployment under STATEFRAME_ROUTING_SCHEMA=2.

Schema 1 (default) must keep the legacy launch-served-wins resolution byte-for-byte.
"""
from types import SimpleNamespace

import pytest

from kernel_model_policy import (
    CONCISE_ACTION_GUIDANCE,
    execution_policy_for,
    policy_for_runtime,
)

LAUNCH_SERVED = "a3b-x"
SELECTED = "frontier-y"
SCHEMA_ENV = "STATEFRAME_ROUTING_SCHEMA"
CONCISE_ENV = "STATEFRAME_TOKENLAB_CONCISE_ACTION_GUIDANCE"
CONCISE_MODELS_ENV = "STATEFRAME_TOKENLAB_CONCISE_ACTION_GUIDANCE_MODELS"
_CLEAR = (
    "DS_SERVED_MODEL", "REACT_SERVED_MODEL", "STATEFRAME_SERVED_MODEL", "SERVED_MODEL",
    "STATEFRAME_MODEL_NAME", "DS_MODEL", "STATEFRAME_PROVIDER",
)


@pytest.fixture
def launch_env(monkeypatch):
    """Launch runs as A3B; only the concise allowlist names the selected frontier model."""
    for key in _CLEAR:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DS_SERVED_MODEL", LAUNCH_SERVED)
    monkeypatch.setenv(CONCISE_ENV, "on")
    monkeypatch.setenv(CONCISE_MODELS_ENV, SELECTED)
    monkeypatch.delenv(SCHEMA_ENV, raising=False)
    return monkeypatch


def _v2(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "2")


def test_v2_selected_deployment_wins_over_launch_served(launch_env):
    _v2(launch_env)
    policy = execution_policy_for(SimpleNamespace(), SELECTED)
    assert policy.model_name == SELECTED
    assert policy.requested_model_name == SELECTED
    assert policy.served_model_name == ""
    assert policy.to_dict()["served_model"] is None
    assert policy.guidance_for(CONCISE_ACTION_GUIDANCE) is True


def test_v1_launch_served_still_wins_exact_legacy_values(launch_env):
    policy = execution_policy_for(SimpleNamespace(), SELECTED)
    assert policy.model_name == LAUNCH_SERVED
    assert policy.requested_model_name == SELECTED
    assert policy.served_model_name == LAUNCH_SERVED
    assert policy.guidance_for(CONCISE_ACTION_GUIDANCE) is False


def test_v2_explicit_served_model_kwarg_is_used_as_served(launch_env):
    _v2(launch_env)
    policy = execution_policy_for(SimpleNamespace(), SELECTED, served_model="frontier-y-serve")
    assert policy.model_name == SELECTED
    assert policy.served_model_name == "frontier-y-serve"
    assert policy.to_dict()["served_model"] == "frontier-y-serve"


def test_v2_sticky_cache_recomputes_for_selected_model(launch_env):
    _v2(launch_env)
    runtime = SimpleNamespace()
    first = execution_policy_for(runtime, "model-a")
    assert policy_for_runtime(runtime) is first
    runtime.selected_model = "model-b"
    policy = policy_for_runtime(runtime)
    assert policy.model_name == "model-b"
    assert runtime.execution_model_policy is policy


def test_v2_cache_kept_when_selected_matches_cached_model(launch_env):
    _v2(launch_env)
    runtime = SimpleNamespace()
    cached = execution_policy_for(runtime, SELECTED)
    runtime.selected_model = SELECTED
    assert policy_for_runtime(runtime) is cached


def test_v2_selected_model_without_cache_resolves_selected(launch_env):
    _v2(launch_env)
    runtime = SimpleNamespace(selected_model=SELECTED)
    policy = policy_for_runtime(runtime)
    assert policy.model_name == SELECTED
    assert policy.guidance_for(CONCISE_ACTION_GUIDANCE) is True


def test_v1_sticky_cache_ignores_selected_model(launch_env):
    runtime = SimpleNamespace()
    cached = execution_policy_for(runtime, "model-a")
    runtime.selected_model = "model-b"
    assert policy_for_runtime(runtime) is cached
    assert cached.model_name == LAUNCH_SERVED


def test_v2_policy_for_runtime_without_selection_keeps_legacy_path(launch_env):
    _v2(launch_env)
    policy = policy_for_runtime(SimpleNamespace(model_name=SELECTED))
    assert policy.model_name == LAUNCH_SERVED
    assert policy.served_model_name == LAUNCH_SERVED


def test_v2_launch_served_env_does_not_leak_into_selected_policy(launch_env):
    _v2(launch_env)
    policy = execution_policy_for(SimpleNamespace(), SELECTED)
    assert LAUNCH_SERVED not in (policy.model_name, policy.served_model_name)

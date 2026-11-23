"""Small contracts for deterministic mutation comparison fixtures."""

import os

from test_mutate_parity import _stable_spec
from test_mutation_adversarial_parity import CURRENT, _parity_environment


def test_stable_spec_canonicalizes_only_provider_origins():
    capability = {"dialect": "llama.cpp", "slots": False}
    baseline = {"provider_capabilities": {"http://127.0.0.1:41001": capability}}
    current = {"provider_capabilities": {"http://127.0.0.1:42002": capability}}
    assert _stable_spec(current) == _stable_spec(baseline)
    assert _stable_spec({"provider_capabilities": {"http://127.0.0.1:42002": {
        **capability, "slots": True,
    }}}) != _stable_spec(baseline)
    assert _stable_spec({"provider_capabilities": {
        "http://127.0.0.1:42002": capability,
        "http://127.0.0.1:42003": capability,
    }}) != _stable_spec(baseline)


def test_run_environment_is_run_scoped_and_ignores_ambient_switches(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_MUTATIONS", "authored")
    monkeypatch.setenv("KERNEL_INTERVENTIONS", "1")
    trace = tmp_path / "current-modules.json"

    environment = _parity_environment(CURRENT, "on", trace)

    assert environment["STATEFRAME_HOME"] == str(tmp_path / "current-modules-stateframe-home")
    assert environment["STATEFRAME_STABLE_TOOL_HEAD"] == "on"
    assert environment["PYTHONPATH"] == os.pathsep.join((
        str(CURRENT / "src"),
        str(CURRENT / "src" / "stateframe" / "engine"),
        str(CURRENT / "tests"),
    ))
    assert environment["PYTHONHASHSEED"] == "0"
    assert "STATEFRAME_MUTATIONS" not in environment
    assert "KERNEL_INTERVENTIONS" not in environment

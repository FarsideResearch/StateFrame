"""The allocator default and opt-out retain the pre-layout wire contracts."""
from pathlib import Path

from test_working_set_off_parity_base import _cached_provider_hashes, _provider_hashes
from tools.pinned_tree import pinned_tree

from stateframe import profiles, runner

SWITCH = "STATEFRAME_WORKING_SET_ALLOCATOR"
OFF_BASE = "56af0dc6d6c7b632808070e137b9d95f66db882e"


def test_shipped_profile_defaults_on_and_shell_can_opt_out(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    environment = profiles.load(profiles.DEFAULT)
    environment.update(runner.resolve_opt_in_switches())
    assert environment[SWITCH] == "on"
    monkeypatch.setenv(SWITCH, "off")
    environment.update(runner.resolve_opt_in_switches())
    assert environment[SWITCH] == "off"


def test_default_wire_equals_explicit_on_and_opt_out_restores_release_bytes(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    default = tuple(_provider_hashes(root, tmp_path_factory.mktemp("default-wire"), "default"))
    explicit_on = tuple(_provider_hashes(root, tmp_path_factory.mktemp("on-wire"), "on"))
    explicit_off = _cached_provider_hashes(root, tmp_path_factory)
    release_off = _cached_provider_hashes(pinned_tree(OFF_BASE, tmp_path_factory), tmp_path_factory)
    assert default == explicit_on
    assert explicit_off == release_off
    assert default != explicit_off, "fixture must distinguish allocator ON from OFF bytes"
    assert explicit_on != release_off, (
        "allocator-on mutant must differ from the feature-absent release")

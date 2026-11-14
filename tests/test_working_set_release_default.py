"""The shipped allocator default and its public opt-out retain exact wire contracts."""
from pathlib import Path

from test_working_set_astra_off_parity import _cached_provider_hashes, _provider_hashes
from tools.pinned_tree import pinned_tree

from stateframe import profiles, runner

SWITCH = "STATEFRAME_WORKING_SET_ALLOCATOR"
BASE = "7a75ed92572613c7832ebfaeafcd91af166386c2"
OFF_BASE = "56af0dc6d6"


def test_shipped_profile_defaults_on_and_shell_can_opt_out(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    environment = profiles.load(profiles.DEFAULT)
    environment.update(runner.resolve_opt_in_switches())
    assert environment[SWITCH] == "on"
    monkeypatch.setenv(SWITCH, "off")
    environment.update(runner.resolve_opt_in_switches())
    assert environment[SWITCH] == "off"


def test_default_wire_equals_proven_on_and_opt_out_restores_release_bytes(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    default = tuple(_provider_hashes(root, tmp_path_factory.mktemp("default-wire"), "default"))
    explicit_on = tuple(_provider_hashes(root, tmp_path_factory.mktemp("on-wire"), "on"))
    proven_on = tuple(_provider_hashes(pinned_tree(BASE, tmp_path_factory),
                                    tmp_path_factory.mktemp("base-on-wire"), "on"))
    explicit_off = _cached_provider_hashes(root, tmp_path_factory)
    release_off = _cached_provider_hashes(pinned_tree(OFF_BASE, tmp_path_factory), tmp_path_factory)
    assert default == explicit_on == proven_on
    assert explicit_off == release_off
    assert default != explicit_off, "fixture must distinguish allocator ON from OFF bytes"

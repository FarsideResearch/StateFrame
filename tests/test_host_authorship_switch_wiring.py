"""STATEFRAME_HOST_ROW_IDENTITY: shipped-profile pair coverage (T1) and config/child wiring (T7).

Prior art: pairs follow test_host_authorship_runtime (prepare-level capture, no subprocess run);
effective-profile check follows test_history_digest_ids.test_effective_profile_and_shell_override;
child inheritance follows test_parallel_worker_prep.test_prepare_worker_pins_working_set_allocator_off_for_subframes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_host_authorship as authorship
import kernel_measurement
import kernel_stateframe_runtime as runtime_module
from kernel_stateframe_runtime import StateFrameRuntime
from stateframe import runner
from stateframe.parallel_worker import prepare_worker
from test_parallel_worker_prep import _unit_fixture

ROOT = Path(__file__).resolve().parents[1]
SWITCH = authorship.MODE_ENV
_SERIAL = iter(range(10**6))


def _runtime(tmp_path, step):
    # Same shape as test_host_authorship_runtime._runtime: one journal per runtime.
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.owner = SimpleNamespace(current_step=step, task_row={})
    runtime.snapshot_id = "snapshot"
    runtime.endpoint_profile = None
    home = tmp_path / f"journal-{next(_SERIAL)}"
    home.mkdir()
    runtime.journal = kernel_measurement.AuditJournal(home, f"host-authorship-switch-{step}", {})
    runtime.system_outcome = None
    return runtime


@pytest.mark.parametrize("pair", ["00", "01", "10", "11"])
def test_switch_pairs_with_shipped_profile(tmp_path, monkeypatch, pair):
    profile_on, identity_on = pair[0] == "1", pair[1] == "1"
    for name in runner.OPT_IN_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    if profile_on:
        # Install the shipped r2hc3 switches the same way the shipped_runtime_profile fixture does.
        profile = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text())["switches"]
        for key, value in profile.items():
            if key != "STATEFRAME_PROTECT_TESTS":
                monkeypatch.setenv(key, value)
    monkeypatch.setenv(SWITCH, "on" if identity_on else "off")

    runtime = _runtime(tmp_path, step=3)
    capture = runtime_module._prepare_request_capture(runtime)
    assert capture.request_id, "prepared request must exist under every pair"

    ledger = authorship.ledger_of(capture)
    if identity_on:
        assert isinstance(ledger, authorship.HostAuthorship)
        assert ledger.step == 3 and ledger.request_id == capture.request_id
        assert authorship.ledger_of(runtime) is ledger
    else:
        assert ledger is None
        assert capture._host_authorship is None
        assert authorship.ledger_of(runtime) is None


@pytest.mark.parametrize("value", [None, "off", "shadow", "on"])
def test_switch_reaches_effective_profile_and_child_inherits(tmp_path, monkeypatch, value):
    for name in runner.OPT_IN_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv(SWITCH, value)
    expected = value or "off"

    # Effective profile: the engine-side environment the run hands to the engine.
    options = runner.RunOptions(repo=ROOT, task="Check host-row identity wiring.", endpoint=None)
    effective = runner._provider_environment(options)
    assert effective.get(SWITCH, "off") == expected

    # Child: a subframe prepared from a parent carrying that environment records the same value,
    snapshot, run_dir, spec_path, spec, parent_spec, _parent_run = _unit_fixture(tmp_path)
    parent_spec["environment"][SWITCH] = effective.get(SWITCH, "off")
    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    prepared = json.loads((run_dir / "spec.json").read_text())
    assert prepared["environment"].get(SWITCH, "off") == expected

    # The child installs its recorded environment before any engine import (_child.install_environment).
    from stateframe import _child
    monkeypatch.setenv(SWITCH, "")
    _child.install_environment(prepared)
    assert os.environ.get(SWITCH, "off") == expected

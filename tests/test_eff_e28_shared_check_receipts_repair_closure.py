from dataclasses import replace

import kernel_obligation_assess as checks
from test_host_eff_09_shared_check_cache import (
    _check, _closure, _repair, _runtime, probe,  # noqa: F401
)


def test_shared_cache_fixture_env_or_ledger_change_executes_again(tmp_path, monkeypatch, probe):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("def value(): return 7\n")
    monkeypatch.setenv("STATEFRAME_NAMED_LAYER_REPAIR", "on")
    monkeypatch.setenv("STATEFRAME_REQUIREMENT_ALIGNMENT", "on")
    cache = {}
    check = _check()

    def execute(definition, env, ledger):
        return checks.run_check(root, definition, cache=cache, snapshot="current",
                                environment=env, requirements_ledger=ledger)

    first = execute(check, "env1", {"revision": 1})
    repeated = execute(check, "env1", {"revision": 1})
    assert repeated.reused and repeated.instance_key == first.instance_key
    assert len(probe.runs) == 1
    for definition, env, ledger in ((replace(check, fixture_id="other"), "env1", {"revision": 1}),
                                    (check, "env2", {"revision": 1}),
                                    (check, "env1", {"revision": 2})):
        fresh = execute(definition, env, ledger)
        assert not fresh.reused and fresh.instance_key != first.instance_key
    assert len(probe.runs) == 4


def test_shared_cache_does_not_substitute_for_independent_check(tmp_path, monkeypatch, probe):
    for mode, calls in (("off", 2), ("on", 1)):
        probe.runs.clear()
        runtime, _ = _runtime(tmp_path / mode, monkeypatch, mode)
        _repair(runtime)
        _closure(runtime)
        assert len(probe.runs) == calls
        receipt = checks.run_check(runtime.owner.root, _check())
        assert receipt.status == checks.PASS and not receipt.reused
        assert len(probe.runs) == calls + 1

"""Subframe feedback switches require run authority, including cached meters."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

import kernel_frame_routes
import kernel_parallel
import kernel_request_diet as diet
from kernel_request_budget import _request_token_meter


@pytest.fixture
def no_parallel_service(monkeypatch):
    # Importing either layout is insufficient to authorize subframe behavior.
    import stateframe.engine.kernel_parallel as packaged_parallel
    import kernel_unit_policy
    import stateframe.engine.kernel_unit_policy as packaged_unit

    monkeypatch.setattr(kernel_parallel, "_service", None)
    monkeypatch.setattr(packaged_parallel, "_service", None)
    monkeypatch.setattr(kernel_unit_policy, "_installed", None)
    monkeypatch.setattr(packaged_unit, "_installed", None)
    monkeypatch.setattr(kernel_frame_routes, "ROUTES_ACTIVE", False)


@pytest.fixture
def main_diet(monkeypatch):
    source = subprocess.check_output(
        ["git", "show", "936b355:src/stateframe/engine/kernel_request_diet.py"],
        cwd=Path(__file__).resolve().parents[1], text=True,
    )
    module = ModuleType("_main_request_diet_feedback")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, "936b355:kernel_request_diet.py", "exec"), module.__dict__)
    return module


def _observe_twice(meter):
    # Exercise both the compatibility API and the live journal observer.
    meter.observe_response(2000, 1000, 0)
    meter.observe_record({"kind": "request_diet_applied", "request_id": "sample",
                          "request_bytes_after": 2000})
    meter.observe_record({"kind": "response_received", "request_id": "sample",
                          "response": {"usage": {"prompt_tokens": 1000}}})


@pytest.mark.parametrize("factory", [False, True])
def test_off_feedback_and_fitted_request_match_main(
        monkeypatch, no_parallel_service, main_diet, factory):
    monkeypatch.setenv("STATEFRAME_PARALLEL_USAGE_FEEDBACK", "off")
    monkeypatch.setattr(diet, "resolve_tokenize_url", lambda *_args, **_kwargs: None)
    meter = (_request_token_meter(SimpleNamespace(), SimpleNamespace(journal=None))
             if factory else diet.RequestTokenMeter(disabled=True))
    reference = main_diet.RequestTokenMeter(disabled=True)
    body = {"messages": [
        {"role": "system", "content": "Follow the task."},
        {"role": "user", "content": "Task: " + "x" * 1000},
        {"role": "user", "content": (
            "USER-PROVIDED SKILL INSTRUCTIONS: feedback-fixture \n" + "a" * 20000),
         "_request_budget_section": "skill_bodies"},
    ], "tools": []}
    fitted = []
    for implementation, candidate in ((diet, meter), (main_diet, reference)):
        _observe_twice(candidate)
        request = copy.deepcopy(body)
        implementation.apply_diet(request, stage="FREE", max_request_bytes=1 << 20,
                                  token_budget=10000, token_meter=candidate)
        fitted.append(json.dumps(request).encode("utf-8"))
    assert reference._calibration_samples == 2
    assert len(fitted[1]) < diet._wire_bytes(body), "fixture must require fitting"
    assert fitted[0] == fitted[1]
    assert meter._minimum_bytes_per_token == reference._minimum_bytes_per_token == 2.0
    assert meter._calibration_samples == reference._calibration_samples
    assert meter._fallback_ratio() == reference._fallback_ratio() == (1.8, 0.1)


@pytest.mark.parametrize("authority", ["off", "parent", "packaged_parent", "unit"])
@pytest.mark.parametrize("feedback", [None, "observe", " OFF "])
@pytest.mark.parametrize("reuse", [False, True])
def test_factory_feedback_requires_run_authority(
        tmp_path, monkeypatch, no_parallel_service, authority, feedback, reuse):
    if feedback is None:
        monkeypatch.delenv("STATEFRAME_PARALLEL_USAGE_FEEDBACK", raising=False)
    else:
        monkeypatch.setenv("STATEFRAME_PARALLEL_USAGE_FEEDBACK", feedback)
    runtime = SimpleNamespace()
    capture = SimpleNamespace(journal=None)
    if authority in {"parent", "packaged_parent"}:
        module = sys.modules["kernel_parallel" if authority == "parent"
                             else "stateframe.engine.kernel_parallel"]
        module.install(tmp_path, {"parallel_mode": "2"})
    elif authority == "unit":
        runtime._unit_policy = SimpleNamespace(spec=SimpleNamespace())
    if reuse:
        runtime._request_token_meter = diet.RequestTokenMeter(disabled=True)
        # A cached/restored meter's old policy must be rebound to this run.
        runtime._request_token_meter.usage_feedback_enabled = False
    meter = _request_token_meter(runtime, capture)
    _observe_twice(meter)
    expected = 0 if authority != "off" and feedback == " OFF " else 2
    assert meter._calibration_samples == expected
    assert meter._minimum_bytes_per_token == (2.0 if expected else None)
    assert _request_token_meter(runtime, capture) is meter


def test_explicit_feedback_policy_is_independent_of_environment(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PARALLEL_USAGE_FEEDBACK", "observe")
    meter = diet.RequestTokenMeter(disabled=True, usage_feedback_enabled=False)
    _observe_twice(meter)
    assert meter._calibration_samples == 0
    assert meter._minimum_bytes_per_token is None


@pytest.mark.parametrize("module_name", ["kernel_unit_policy", "stateframe.engine.kernel_unit_policy"])
@pytest.mark.parametrize("feedback", ["off", "observe"])
def test_installed_worker_feedback_without_solver_runtime(
        monkeypatch, no_parallel_service, module_name, feedback):
    from test_unit_visibility import policy

    module = sys.modules[module_name]
    module.install(module.UnitPolicy(policy().spec))
    monkeypatch.setenv("STATEFRAME_PARALLEL_USAGE_FEEDBACK", feedback)
    # Auxiliary sends clear the solver runtime but retain the run's worker policy.
    meter = _request_token_meter(None, SimpleNamespace(journal=None))
    _observe_twice(meter)
    assert meter._calibration_samples == (0 if feedback == "off" else 2)


def test_legacy_checkpoint_meter_retains_historical_feedback(monkeypatch):
    monkeypatch.setenv("STATEFRAME_PARALLEL_USAGE_FEEDBACK", "off")
    meter = diet.RequestTokenMeter(disabled=True)
    # A meter restored from an older checkpoint has no policy attribute.
    meter.__dict__.pop("usage_feedback_enabled", None)
    _observe_twice(meter)
    assert meter._calibration_samples == 2
    assert meter._fallback_ratio() == (1.8, 0.1)


@pytest.mark.parametrize("feedback", ["observe", "off"])
def test_restored_routed_unit_meter_keeps_calibration_and_feedback_policy(
        monkeypatch, no_parallel_service, feedback):
    parent = diet.RequestTokenMeter(disabled=True)
    _observe_twice(parent)
    runtime = SimpleNamespace(_unit_policy=SimpleNamespace(spec=SimpleNamespace(
        meter_calibration=parent.calibration_snapshot())))
    monkeypatch.setattr(kernel_frame_routes, "ROUTES_ACTIVE", True)
    capture = SimpleNamespace(journal=None, _tokenize_endpoint="http://127.0.0.1:1/v1",
                              frame_route=kernel_frame_routes.Route(
        role="unit", scope="request", unit_class=None,
        settings={"model": "unit-model", "endpoint": "http://127.0.0.1:1/v1",
                  "allow_remote": False}))
    monkeypatch.setenv("STATEFRAME_PARALLEL_USAGE_FEEDBACK", feedback)
    meter = _request_token_meter(runtime, capture)
    assert meter is capture._request_token_meter
    assert meter._calibration_samples == 2
    _observe_twice(meter)
    assert meter._calibration_samples == (2 if feedback == "off" else 4)
    assert meter._minimum_bytes_per_token == 2.0
    assert _request_token_meter(runtime, capture) is meter

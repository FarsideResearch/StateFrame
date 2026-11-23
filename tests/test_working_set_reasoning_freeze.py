"""The allocator certifies provider bytes using request-time reasoning policy."""
import json
from types import SimpleNamespace

import pytest

from kernel_final_wire_allocation import _finalize_provider_request_body
from kernel_launch_identity_cost import ProviderBoundaryContractError
from kernel_provider_dialect import ProviderCapabilities
from kernel_request_seal import RequestSealError
from kernel_working_set_live import assert_certified_body
from kernel_working_set_render import render_assignment, snapshot_digest
from kernel_working_set_snapshot import PreparationSnapshot


EFFORTS = (None, "none", "minimal", "low", "medium", "high", "xhigh", "max")


def _snapshot(*, dialect="openai-strict", reasoning_mode="on", effort=None, body=None):
    body = body or {"model": "hosted", "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: inspect"},
    ]}
    return PreparationSnapshot(
        json.dumps(body).encode(), (), "request", 1, 100_000,
        dialect=dialect, reasoning_mode=reasoning_mode,
        provider_reasoning_effort=effort,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("", None), ("none", "none"), ("minimal", "minimal"),
     ("low", "low"), ("medium", "medium"), ("high", "high"),
     ("xhigh", "xhigh"), ("max", "max"), (" HIGH ", "high")],
)
def test_freeze_shadow_captures_normalized_strict_effort(monkeypatch, raw, expected):
    from kernel_working_set_shadow import freeze_shadow

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_REASONING_PROFILE", "on")
    if raw is None:
        monkeypatch.delenv("REACT_REASONING_EFFORT", raising=False)
    else:
        monkeypatch.setenv("REACT_REASONING_EFFORT", raw)
    body = {"model": "hosted", "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: inspect"},
    ]}

    snapshot = freeze_shadow(body, SimpleNamespace(), request_id="request",
                             dialect="openai-strict", reasoning_mode="on")

    assert snapshot.provider_reasoning_effort == expected


def test_llama_snapshot_keeps_operator_effort_out_of_default_authority(monkeypatch):
    from kernel_working_set_shadow import freeze_shadow

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("REACT_REASONING_EFFORT", "high")
    body = {"model": "local", "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: inspect"},
    ]}

    snapshot = freeze_shadow(body, SimpleNamespace(), request_id="request",
                             dialect="llama.cpp", reasoning_mode="on")

    assert snapshot.provider_reasoning_effort is None


def test_read_admission_snapshot_carries_the_strict_effort(monkeypatch):
    from kernel_working_set_live import read_admission_snapshot

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_REASONING_PROFILE", "on")
    monkeypatch.setenv("REACT_REASONING_EFFORT", " HIGH ")
    body = {"model": "hosted", "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: inspect"},
        {"role": "user", "content": '[StateFrame]\n{"state":{"framefit5":{"mode":"FRAME_ONLY"}}}'},
    ]}
    runtime = SimpleNamespace(
        _read_admission_provider_body=body,
        _read_admission_provider_authority=SimpleNamespace(
            dialect="openai-strict", reasoning_mode="on"),
    )

    snapshot = read_admission_snapshot(runtime, body["messages"], byte_cap=100_000)

    assert snapshot.provider_reasoning_effort == "high"


@pytest.mark.parametrize("effort", EFFORTS)
@pytest.mark.parametrize("reasoning_mode", ("on", "passthrough"))
def test_strict_render_uses_frozen_effort_after_environment_changes(
        monkeypatch, effort, reasoning_mode):
    snapshot = _snapshot(reasoning_mode=reasoning_mode, effort=effort)
    if effort == "high":
        monkeypatch.setenv("REACT_REASONING_EFFORT", "low")
    else:
        monkeypatch.setenv("REACT_REASONING_EFFORT", "high")

    rendered = render_assignment(snapshot, {}, "plain")
    body = json.loads(rendered.wire)

    if effort is None:
        assert "reasoning_effort" not in body
    else:
        assert body["reasoning_effort"] == effort
    assert rendered.wire == json.dumps(body).encode()


@pytest.mark.parametrize("effort", EFFORTS)
def test_strict_off_render_requires_frozen_none(effort):
    snapshot = _snapshot(reasoning_mode="off", effort=effort)

    if effort == "none":
        assert json.loads(render_assignment(snapshot, {}, "plain").wire)[
            "reasoning_effort"] == "none"
    else:
        with pytest.raises(ProviderBoundaryContractError, match="reasoning_off_unproven"):
            render_assignment(snapshot, {}, "plain")


def test_changed_effort_at_final_seal_keeps_certified_body_refusal(monkeypatch):
    from kernel_working_set_shadow import freeze_shadow

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_REASONING_PROFILE", "on")
    monkeypatch.setenv("REACT_REASONING_EFFORT", "high")
    body = {"model": "hosted", "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: inspect"},
    ]}
    snapshot = freeze_shadow(body, SimpleNamespace(), request_id="request",
                             dialect="openai-strict", reasoning_mode="on")
    rendered = render_assignment(snapshot, {}, "plain")
    assert json.loads(rendered.wire)["reasoning_effort"] == "high"
    prepared = SimpleNamespace(
        wire=rendered.wire,
        rendered=SimpleNamespace(certificate=rendered.certificate),
    )

    monkeypatch.setenv("REACT_REASONING_EFFORT", "low")
    sealed = _finalize_provider_request_body(
        SimpleNamespace(journal=SimpleNamespace(manifest={})),
        json.loads(rendered.wire), ProviderCapabilities(dialect="openai-strict"),
    )

    assert sealed["reasoning_effort"] == "low"
    with pytest.raises(RequestSealError, match="validator changed certified working-set body"):
        assert_certified_body(sealed, prepared)


def test_legacy_snapshot_constructor_defaults_effort_to_unset():
    body = {"model": "local", "messages": []}
    legacy_fields = {
        "fixed_body": json.dumps(body).encode(), "sources": (),
        "request_id": "request", "due_ordinal": 1, "byte_cap": 100_000,
    }

    snapshot = PreparationSnapshot(**legacy_fields)

    assert snapshot.provider_reasoning_effort is None
    assert "provider_reasoning_effort" not in repr(snapshot)


def test_unset_effort_keeps_legacy_snapshot_digest():
    body = {"model": "local", "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: inspect"},
    ]}
    snapshot = _snapshot(dialect="llama.cpp", effort=None, body=body)
    rendered = render_assignment(snapshot, {}, "plain")

    assert snapshot_digest(snapshot) == "b6a7a0f5efc0bd2830ca1175730c364b8da9e8b906549e078e9e7ee6522b2971"
    assert "provider_reasoning_effort" not in repr(snapshot)
    assert rendered.wire == (
        b'{"model": "local", "messages": [{"role": "system", "content": "system"}, '
        b'{"role": "user", "content": "Task: inspect"}, {"role": "user", '
        b'"content": "kept 0 selected source windows; about 0 chars of source fit"}]}'
    )
    assert rendered.certificate.snapshot_digest == snapshot_digest(snapshot)
    assert rendered.certificate.body_sha256 == "c2d83bf234de452b2de26d4d39d2938cbb691e4bb06a6c688cf6b2b5257eab5b"
    assert rendered.certificate.assignment_digest == "ae27c29160213129f7361e7d405832f28cc354c83d0b4f1a04edfa81e79447a7"


def test_strict_body_normalization_matches_except_for_frozen_effort(monkeypatch):
    from kernel_final_wire_allocation import _finalize_provider_request_body
    from kernel_working_set_render import _normalize

    body = {
        "model": "hosted", "messages": [{"role": "user", "content": "hi", "private": "drop"}],
        "tools": [{"type": "function", "extra": "drop", "function": {
            "name": "read", "parameters": {"type": "object", "host_extra": "drop",
                "properties": {"x": {"type": "string", "host_extra": "drop"}}}}}],
        "n_probs": 4, "logprobs": False, "reasoning_budget_tokens": 4096,
    }
    snapshot = _snapshot(effort=None, body=body)
    monkeypatch.setenv("STATEFRAME_REASONING_PROFILE", "on")
    monkeypatch.setenv("REACT_REASONING_EFFORT", "high")
    rendered = _normalize(body, snapshot)
    sealed = _finalize_provider_request_body(
        SimpleNamespace(journal=SimpleNamespace(manifest={})), body,
        ProviderCapabilities(dialect="openai-strict"),
    )

    assert "reasoning_effort" not in rendered
    assert sealed["reasoning_effort"] == "high"
    assert {key: value for key, value in rendered.items() if key != "reasoning_effort"} == {
        key: value for key, value in sealed.items() if key != "reasoning_effort"
    }
    assert rendered["logprobs"] is True and rendered["top_logprobs"] == 4
    assert "reasoning_budget_tokens" not in rendered

"""MR-2: advisory decide2/work-frame bodies resolve the configured default model id under routing/2.

Schema 1 (default) must keep the tier aliases byte-identical. Schema 2 replaces the aliases
with the configured id, honouring a per-role override env first.
"""
from __future__ import annotations

import pytest

from kernel_decide2 import DecisionRequest, build_frame, build_micro_request, build_work_request
from kernel_model_ids import TIER_ALIASES, resolve_model_id

SCHEMA_ENV = "STATEFRAME_ROUTING_SCHEMA"
DEFAULT_ENVS = ("STATEFRAME_MODEL_NAME", "DS_MODEL")
OVERRIDE_ENVS = ("STATEFRAME_DECIDE2_MODEL", "STATEFRAME_WORK_FRAME_MODEL")
CONCRETE = "concrete-model-9b"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (SCHEMA_ENV, *DEFAULT_ENVS, *OVERRIDE_ENVS):
        monkeypatch.delenv(name, raising=False)


def _request() -> DecisionRequest:
    return DecisionRequest(
        decision_id="decision-1", question_kind="relevance",
        question="Is the source relevant?", options=("yes", "no", "ABSTAIN"),
    )


def _micro_body() -> dict[str, object]:
    return build_micro_request(
        build_frame(_request()), rung="L1",
        tool_surface=({"type": "function", "function": {"name": "read"}},),
    )


def _work_body() -> dict[str, object]:
    return build_work_request(_request(), reason="low confidence", max_tokens=64)


def test_tier_aliases_are_the_two_advisory_roles():
    assert TIER_ALIASES == frozenset({"decision-tier", "work-frame"})


@pytest.mark.parametrize("schema", [None, "", "1"])
def test_schema_one_keeps_both_alias_bodies_byte_identical(monkeypatch, schema):
    if schema is not None:
        monkeypatch.setenv(SCHEMA_ENV, schema)
    monkeypatch.setenv("STATEFRAME_MODEL_NAME", CONCRETE)
    monkeypatch.setenv("DS_MODEL", CONCRETE)
    monkeypatch.setenv("STATEFRAME_DECIDE2_MODEL", "override-a")
    monkeypatch.setenv("STATEFRAME_WORK_FRAME_MODEL", "override-b")

    assert _micro_body()["model"] == "decision-tier"
    assert _work_body()["model"] == "work-frame"


def test_schema_two_micro_body_carries_configured_default(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "2")
    monkeypatch.setenv("DS_MODEL", CONCRETE)

    assert _micro_body()["model"] == CONCRETE


def test_schema_two_work_body_carries_configured_default(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "2")
    monkeypatch.setenv("DS_MODEL", CONCRETE)

    assert _work_body()["model"] == CONCRETE


def test_schema_two_prefers_statefram_model_name_over_ds_model(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "2")
    monkeypatch.setenv("STATEFRAME_MODEL_NAME", "named-model")
    monkeypatch.setenv("DS_MODEL", CONCRETE)

    assert _micro_body()["model"] == "named-model"
    assert _work_body()["model"] == "named-model"


def test_schema_two_decide2_override_wins_for_micro_body_only(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "2")
    monkeypatch.setenv("DS_MODEL", CONCRETE)
    monkeypatch.setenv("STATEFRAME_DECIDE2_MODEL", "decide2-override")

    assert _micro_body()["model"] == "decide2-override"
    assert _work_body()["model"] == CONCRETE


def test_schema_two_work_frame_override_wins_for_work_body_only(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "2")
    monkeypatch.setenv("DS_MODEL", CONCRETE)
    monkeypatch.setenv("STATEFRAME_WORK_FRAME_MODEL", "work-override")

    assert _work_body()["model"] == "work-override"
    assert _micro_body()["model"] == CONCRETE


def test_schema_two_without_any_configured_model_keeps_alias(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "2")

    assert _micro_body()["model"] == "decision-tier"
    assert _work_body()["model"] == "work-frame"


def test_invalid_schema_fails_loud_at_body_build(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "3")

    with pytest.raises(ValueError, match=SCHEMA_ENV):
        _work_body()


@pytest.mark.parametrize(
    ("requested", "environ", "expected"),
    [
        # schema 1 returns the input unchanged, even for aliases
        ("decision-tier", {"DS_MODEL": CONCRETE}, "decision-tier"),
        ("work-frame", {"STATEFRAME_DECIDE2_MODEL": "x"}, "work-frame"),
        # schema 2 alias resolution
        ("decision-tier", {SCHEMA_ENV: "2", "DS_MODEL": CONCRETE}, CONCRETE),
        ("work-frame", {SCHEMA_ENV: "2", "DS_MODEL": CONCRETE}, CONCRETE),
        ("decision-tier", {SCHEMA_ENV: "2", "STATEFRAME_MODEL_NAME": "named", "DS_MODEL": CONCRETE}, "named"),
        ("decision-tier", {SCHEMA_ENV: "2", "STATEFRAME_DECIDE2_MODEL": "ov", "DS_MODEL": CONCRETE}, "ov"),
        ("work-frame", {SCHEMA_ENV: "2", "STATEFRAME_WORK_FRAME_MODEL": "ov", "DS_MODEL": CONCRETE}, "ov"),
        ("decision-tier", {SCHEMA_ENV: "2", "STATEFRAME_WORK_FRAME_MODEL": "ov", "DS_MODEL": CONCRETE}, CONCRETE),
        ("work-frame", {SCHEMA_ENV: "2", "STATEFRAME_DECIDE2_MODEL": "ov", "DS_MODEL": CONCRETE}, CONCRETE),
        ("decision-tier", {SCHEMA_ENV: "2"}, "decision-tier"),
        ("work-frame", {SCHEMA_ENV: "2"}, "work-frame"),
        # whitespace-only env values count as unset
        ("decision-tier", {SCHEMA_ENV: "2", "DS_MODEL": "   "}, "decision-tier"),
        ("decision-tier", {SCHEMA_ENV: "2", "STATEFRAME_DECIDE2_MODEL": " ", "DS_MODEL": CONCRETE}, CONCRETE),
        # explicit concrete ids pass through under schema 2
        (CONCRETE, {SCHEMA_ENV: "2", "DS_MODEL": "other", "STATEFRAME_DECIDE2_MODEL": "ov"}, CONCRETE),
        ("some-tier-like-id", {SCHEMA_ENV: "2", "DS_MODEL": CONCRETE}, "some-tier-like-id"),
        # empty or missing request resolves to the configured default under schema 2
        ("", {SCHEMA_ENV: "2", "DS_MODEL": CONCRETE}, CONCRETE),
        (None, {SCHEMA_ENV: "2", "DS_MODEL": CONCRETE}, CONCRETE),
        (None, {SCHEMA_ENV: "2"}, None),
    ],
)
def test_resolve_model_id_table(requested, environ, expected):
    assert resolve_model_id(requested, environ=environ) == expected


def test_resolve_model_id_reads_process_env_when_no_mapping_given(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "2")
    monkeypatch.setenv("DS_MODEL", CONCRETE)

    assert resolve_model_id("decision-tier") == CONCRETE

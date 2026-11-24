"""The optional detail layer exposes only hashes, fixed enums and numeric cost."""
from dataclasses import dataclass

import pytest

from stateframe import __version__
from stateframe.otel.levels import (
    Levels,
    cost_attributes,
    describe,
    extra_attributes,
    model_id,
    provider_name,
    valid_attribute,
    valid_version,
)
from stateframe.otel.types import AttrKey


KEY = b"k" * 32
MODEL_CANARY = "private-model-canary-X9"


@dataclass(frozen=True)
class Record:
    request_model: str | None = MODEL_CANARY
    response_model: str | None = "response-model-canary-P4"
    provider: str | None = "local"
    engine_version: str | None = __version__
    reported_cost: int | float | None = 0.0125
    price_table: tuple[int | float, int | float] | None = None
    usage: tuple[int, int] | None = None


def test_model_names_are_replaced_with_keyed_ids():
    values = dict(extra_attributes(Levels(), Record(), KEY))

    assert values[AttrKey.REQUEST_MODEL.value] == model_id(KEY, MODEL_CANARY)
    assert values[AttrKey.RESPONSE_MODEL.value] == model_id(KEY, "response-model-canary-P4")
    assert values[AttrKey.REQUEST_MODEL.value] != model_id(b"x" * 32, MODEL_CANARY)
    assert MODEL_CANARY not in repr(values)
    assert all(valid_attribute(name, values[name]) for name in (
        AttrKey.REQUEST_MODEL.value, AttrKey.RESPONSE_MODEL.value,
    ))
    assert not valid_attribute(AttrKey.REQUEST_MODEL.value, MODEL_CANARY)


def test_fixed_provider_list_and_default_fields():
    values = dict(extra_attributes(Levels(), Record(), KEY))

    assert values[AttrKey.PROVIDER_NAME.value] == "local"
    assert values[AttrKey.ENGINE_VERSION.value] == __version__
    assert describe(Levels()) == tuple(values)
    assert provider_name("private-provider-canary") == "other"
    assert valid_attribute(AttrKey.PROVIDER_NAME.value, "other")
    assert not valid_attribute(AttrKey.PROVIDER_NAME.value, "private-provider-canary")


def test_cost_is_optional_and_numeric():
    assert AttrKey.COST_VALUE.value not in dict(extra_attributes(Levels(), Record(), KEY))
    values = dict(extra_attributes(Levels(include_cost=True), Record(), KEY))
    assert values[AttrKey.COST_VALUE.value] == 12_500
    assert values[AttrKey.COST_KIND.value] == "reported"
    assert all(type(values[name]) is int for name in (AttrKey.COST_VALUE.value,))
    assert cost_attributes(None, None, None) == ()


def test_estimated_cost_uses_token_counts_and_source_rates():
    assert cost_attributes(None, (2, 4), (10, 5)) == (
        (AttrKey.COST_VALUE.value, 40), (AttrKey.COST_KIND.value, "estimated"),
    )


@pytest.mark.parametrize("version", ["0.2.6", "12.4.0-rc.1", "1.2.3+build.7"])
def test_versions_follow_the_strict_version_pattern(version):
    assert valid_version(version)
    assert valid_attribute(AttrKey.ENGINE_VERSION.value, version)


@pytest.mark.parametrize("version", [
    "tenant-private", "/models/alpha.bin", "01.2.3", "1.2",
    "1.2.3-01", "1.2.3-rc..1", "1.2.3+build..7",
])
def test_non_version_text_is_refused(version):
    assert not valid_version(version)
    assert not valid_attribute(AttrKey.ENGINE_VERSION.value, version)

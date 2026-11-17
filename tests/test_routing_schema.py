import pytest

from kernel_routing_schema import ROUTING_SCHEMA_ENV, routing_schema, routing_v2


@pytest.mark.parametrize("raw,expected", [("", 1), ("1", 1), (" 2 ", 2), ("2", 2)])
def test_routing_schema_values(raw, expected):
    assert routing_schema({ROUTING_SCHEMA_ENV: raw}) == expected


def test_unset_is_legacy():
    assert routing_schema({}) == 1 and not routing_v2({})


@pytest.mark.parametrize("raw", ["3", "on", "v2", "routing/2"])
def test_unknown_values_fail_loud(raw):
    with pytest.raises(ValueError):
        routing_schema({ROUTING_SCHEMA_ENV: raw})

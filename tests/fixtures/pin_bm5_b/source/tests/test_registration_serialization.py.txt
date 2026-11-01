import importlib

import pytest


@pytest.mark.feature
def test_batch_capability_is_registered_to_real_runner():
    batch = importlib.import_module("pkg.batch")
    registry = importlib.import_module("pkg.registry")
    assert registry.lookup("batch") is batch.run_batch


@pytest.mark.feature
def test_serialized_results_keep_contract_field_names():
    batch = importlib.import_module("pkg.batch")
    serializers = importlib.import_module("pkg.serializers")
    result = batch.run_batch(["x"], str)[0]
    serialized = serializers.serialize_batch([result])
    assert set(serialized[0]) == {"index", "item", "ok", "value", "error"}
    assert serialized[0] == {"index": 0, "item": "x", "ok": True, "value": "x", "error": None}

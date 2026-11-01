import importlib

import pytest


def _batch_api():
    return importlib.import_module("pkg.batch")


@pytest.mark.feature
def test_module_exports_contract_and_runner():
    batch = _batch_api()
    assert "BatchResult" in batch.__all__
    assert "run_batch" in batch.__all__
    assert batch.BatchResult is not None


@pytest.mark.feature
def test_results_keep_input_order_and_capture_each_error():
    batch = _batch_api()

    def work(item):
        if item == "bad":
            raise ValueError("unavailable")
        return item.upper()

    results = batch.run_batch(["first", "bad", "last"], work)
    assert [result.index for result in results] == [0, 1, 2]
    assert [result.item for result in results] == ["first", "bad", "last"]
    assert [result.ok for result in results] == [True, False, True]
    assert [result.value for result in results] == ["FIRST", None, "LAST"]
    assert results[1].error == "ValueError: unavailable"


@pytest.mark.feature
def test_stop_on_error_records_failure_then_stops_work():
    batch = _batch_api()
    seen = []

    def work(item):
        seen.append(item)
        if item == 2:
            raise RuntimeError("halt")
        return item * 10

    results = batch.run_batch([1, 2, 3], work, stop_on_error=True)
    assert seen == [1, 2]
    assert [result.index for result in results] == [0, 1]
    assert [result.ok for result in results] == [True, False]

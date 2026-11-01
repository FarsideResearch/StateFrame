import pytest

from pkg import run_one, service_run_one
from pkg.registry import lookup
from pkg.worker import Worker


@pytest.mark.preservation
def test_existing_single_item_paths_remain_available():
    plus_one = lambda value: value + 1
    assert service_run_one(4, plus_one) == 5
    assert Worker().process_one(5, plus_one) == 6
    assert run_one(6, plus_one) == 7


@pytest.mark.preservation
def test_existing_registry_entry_remains_available():
    assert lookup("single") == "single-item"

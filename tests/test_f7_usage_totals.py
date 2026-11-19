import json
import time

import pytest

from stateframe.runner import summarize


def _completion_total(tmp_path, records):
    sends = [dict(send=index, **record) for index, record in enumerate(records, start=1)]
    (tmp_path / "sends.jsonl").write_text(
        "".join(json.dumps(send) + "\n" for send in sends), encoding="utf-8"
    )
    return summarize(tmp_path, time.monotonic())["completion_tokens"]


def test_completion_total_sums_complete_usage(tmp_path):
    assert _completion_total(tmp_path, [{"completion_tokens": 5}, {"completion_tokens": 7}]) == 12


@pytest.mark.parametrize(
    "records",
    [
        pytest.param([{"completion_tokens": 5}, {"completion_tokens": None}], id="null-count"),
        pytest.param([{"completion_tokens": 5}, {}], id="missing-count"),
        pytest.param([{"completion_tokens": 5}, {"completion_tokens": -1}], id="negative-count"),
    ],
)
def test_completion_total_is_unknown_when_any_send_lacks_valid_usage(tmp_path, records):
    assert _completion_total(tmp_path, records) is None


def test_completion_total_preserves_a_measured_zero(tmp_path):
    assert _completion_total(tmp_path, [{"completion_tokens": 0}, {"completion_tokens": 0}]) == 0

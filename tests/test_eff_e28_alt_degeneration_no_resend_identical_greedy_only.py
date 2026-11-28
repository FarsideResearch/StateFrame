import pytest

from kernel_response_truncation import MODEL_NOTE
from test_eff_e28_degeneration_replan_single_retry_removed import run


@pytest.mark.parametrize("temperature", [0, 0.7])
def test_only_greedy_at_ceiling_suppresses_second_send(monkeypatch, temperature):
    monkeypatch.setenv("STATEFRAME_DEGENERATION_GREEDY_REPLAN", "off")
    control, baseline = run(monkeypatch, "off", 128, temperature=temperature)
    monkeypatch.setenv("STATEFRAME_DEGENERATION_GREEDY_REPLAN", "on")
    result, posts = run(monkeypatch, "off", 128, temperature=temperature)
    assert baseline == [posts[0], posts[0]]
    if temperature == 0:
        assert result == (MODEL_NOTE, []) and len(posts) == 1
    else:
        assert result == control and posts == baseline

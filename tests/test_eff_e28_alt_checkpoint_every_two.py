import pytest

from test_checkpoint_pool_prune_objects import _synthetic_checkpoint_environment  # noqa: F401
from test_eff_e28_checkpoint_every_four_cadence import check_sparse_resume, publications


def test_every_two_publishes_expected_boundaries(monkeypatch, tmp_path):
    control = publications(monkeypatch, tmp_path / "dense", 1)
    sparse = publications(monkeypatch, tmp_path / "sparse", 2)
    assert list(sparse) == [2, 4, 6, 8]
    assert sparse == {step: control[step] for step in sparse}
    assert sum(n for n, _ in sparse.values()) <= 0.55 * sum(n for n, _ in control.values())


@pytest.mark.host_sandbox
def test_two_step_resume_reproduces_sends_and_effects(tmp_path, monkeypatch):
    check_sparse_resume(tmp_path, monkeypatch, 2)

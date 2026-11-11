"""Security regressions for refused mutation calls."""

from types import SimpleNamespace

import pytest


def _smoke_snapshot_runtime(tmp_path, monkeypatch):
    import kernel_mutate_build
    import kernel_mutate_qualify
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.txt").write_text("smoke input\n")
    root = tmp_path / "runtime"
    root.mkdir()
    journal = Journal(tmp_path / "journal")
    runtime = MutationRuntime(
        level="authored", journal=journal, repo=repo,
        policy=MutationPolicy(level="authored"),
        executor_factory=lambda: SimpleNamespace(root=root),
    )
    manifest = {
        "name": "reader", "revision": 1,
        "requires": {"network": False, "writes": False},
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "smoke": [{"input": {}, "inputs": ["sample.txt"], "expect": {}}],
        "limits": {"wall_s": 1, "cpu_s": 1, "mem_mb": 128},
    }
    runtime.ready["reader"] = {"manifest": manifest, "files": {"main.py": b""}}
    monkeypatch.setattr(kernel_mutate_build, "stage_unit", lambda *_args, **_kwargs:
                        SimpleNamespace(dir=root, source_digest="sha256:source"))
    monkeypatch.setattr(kernel_mutate_qualify, "_binding", lambda: {
        "closure_digest": "sha256:closure", "sdk_digest": "sha256:sdk"})
    return runtime, root, journal


def test_exhausted_mutation_use_does_not_allocate_smoke_snapshots(tmp_path, monkeypatch):
    import kernel_mutate_build
    from kernel_mutate_runtime import MutationBudgetError

    runtime, root, journal = _smoke_snapshot_runtime(tmp_path, monkeypatch)
    journal.append("BUDGET_EXHAUSTED", name="invocations")
    monkeypatch.setattr(kernel_mutate_build, "admit", lambda *_args, **_kwargs:
                        SimpleNamespace(admission_digest="sha256:admitted"))

    for index in range(3):
        with pytest.raises(MutationBudgetError):
            runtime._use("reader", {"input": {}, "inputs": ["sample.txt"]}, f"call-{index}")
        assert not (root / "in").exists()


def test_refused_mutation_use_removes_smoke_snapshots_after_admission(tmp_path, monkeypatch):
    import kernel_mutate_build

    runtime, root, _journal = _smoke_snapshot_runtime(tmp_path, monkeypatch)

    def refuse_admission(*_args, **_kwargs):
        assert list((root / "in").iterdir())
        raise kernel_mutate_build.BuildError("STATIC", "invalid unit")

    monkeypatch.setattr(kernel_mutate_build, "admit", refuse_admission)
    with pytest.raises(kernel_mutate_build.BuildError):
        runtime._use("reader", {"input": {}, "inputs": ["sample.txt"]}, "call")
    assert not list((root / "in").iterdir())


def test_mutation_use_cleans_inputs_if_budget_exhausts_after_admission(tmp_path, monkeypatch):
    import kernel_mutate_build
    from kernel_mutate_runtime import MutationBudgetError

    runtime, root, journal = _smoke_snapshot_runtime(tmp_path, monkeypatch)

    def exhaust_after_admission(*_args, **_kwargs):
        journal.append("BUDGET_EXHAUSTED", name="invocations")
        return SimpleNamespace(admission_digest="sha256:admitted")

    monkeypatch.setattr(kernel_mutate_build, "admit", exhaust_after_admission)
    with pytest.raises(MutationBudgetError):
        runtime._use("reader", {"input": {}, "inputs": ["sample.txt"]}, "call")
    assert not list((root / "in").iterdir())


def test_refused_mutation_submit_removes_smoke_snapshots(tmp_path, monkeypatch):
    import kernel_mutate_build

    runtime, root, _journal = _smoke_snapshot_runtime(tmp_path, monkeypatch)
    manifest = dict(runtime.ready["reader"]["manifest"], parent=None)
    runtime.ready.clear()
    runtime.episode = {"id": "episode", "goal": "read sample", "sends_left": 1,
                       "unit_id": "reader"}

    def refuse_admission(*_args, **_kwargs):
        assert list((root / "in").iterdir())
        raise kernel_mutate_build.BuildError("STATIC", "invalid unit")

    monkeypatch.setattr(kernel_mutate_build, "admit", refuse_admission)
    reply = runtime._submit({"manifest": manifest, "files": {"main.py": ""}}, "call")
    assert reply.startswith("REFUSED MUTATION_ADMISSION")
    assert not list((root / "in").iterdir())

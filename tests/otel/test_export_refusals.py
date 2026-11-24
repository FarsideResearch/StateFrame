from __future__ import annotations

from types import SimpleNamespace

import pytest

from stateframe.otel import cli, companion
from stateframe.otel.types import Coverage


def _prepare_export(monkeypatch, *, seed=b"s" * 16, complete=True, held=False, summary=True,
                    config_allowed=True, transport_allowed=True):
    from stateframe.otel import authority

    monkeypatch.setattr(companion, "protect_process", lambda **_kwargs: None)
    monkeypatch.setattr(authority, "source_mode_problem", lambda _root: None)
    monkeypatch.setattr(
        companion,
        "policy_view",
        lambda _root: SimpleNamespace(seed=seed, attempt=1, profile="restricted", environment={},
                                       grants=(), authority_records=(), source_generation=None,
                                       ),
    )
    monkeypatch.setattr(companion, "_run_lock_held", lambda *_args: held)
    monkeypatch.setattr(companion, "admit", lambda *_args: Coverage(complete, ()))
    monkeypatch.setattr(authority, "_read_record", lambda _root: {"runtime_outcome": "completed"} if summary else None)
    monkeypatch.setattr(companion, "config_path_allowed", lambda *_args: config_allowed)
    monkeypatch.setattr(companion, "transport_paths_allowed", lambda *_args: transport_allowed)

    resolved = SimpleNamespace(traces=object(), sdk_disabled=False, diagnostics=(),
                               levels=None, price_rates=())
    monkeypatch.setattr(companion, "resolve", lambda *_args: resolved)


@pytest.mark.parametrize(
    ("options", "seed", "complete", "held", "summary", "config_allowed", "transport_allowed", "diagnostic"),
    [
        ([], None, True, False, True, True, True, "stateframe-otel: export source unavailable\n"),
        ([], b"s" * 16, True, False, False, True, True, "stateframe-otel: export summary missing\n"),
        ([], b"s" * 16, True, True, True, True, True, "stateframe-otel: export source busy\n"),
        ([], b"s" * 16, False, False, True, True, True,
         "stateframe-otel: export admission incomplete\n"),
        (["--config", "unused.toml"], b"s" * 16, True, False, True, False, True,
         "stateframe-otel: export config refused\n"),
        ([], b"s" * 16, True, False, True, True, False,
         "stateframe-otel: export transport path refused\n"),
    ],
)
def test_export_refusals_print_one_fixed_stderr_line(
    monkeypatch, capsys, tmp_path, options, seed, complete, held, summary, config_allowed,
    transport_allowed, diagnostic,
):
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    _prepare_export(monkeypatch, seed=seed, complete=complete, held=held, summary=summary,
                    config_allowed=config_allowed, transport_allowed=transport_allowed)

    assert cli.main(["export", str(run), *options]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == diagnostic
    assert str(run) not in captured.err

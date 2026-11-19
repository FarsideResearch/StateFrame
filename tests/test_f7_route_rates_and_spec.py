"""Regressions for accounting-only route rates and damaged run metadata."""
import json
import time
from pathlib import Path

from stateframe.endpoint import Endpoint
from stateframe import runner


RATES = {
    "cost_input_per_mtok": 1.0,
    "cost_cached_input_per_mtok": 1.0,
    "cost_output_per_mtok": 2.0,
}


def _route_accounting(rates=RATES):
    return {
        "schema": "stateframe-route-accounting/1",
        "roles": {"read": dict(rates)},
    }


def _route_only_options(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("value = 1\n")
    (repo / "test_app.py").write_text("def test_value(): assert True\n")
    return runner.RunOptions(
        repo=repo,
        task="Inspect app.py",
        endpoint=Endpoint("http://127.0.0.1:8080/v1", "base"),
        run_dir=tmp_path / "run",
        route_cli=[
            "read.cost_input_per_mtok=1",
            "read.cost_cached_input_per_mtok=1",
            "read.cost_output_per_mtok=2",
        ],
    )


def _record_send(run, *, role="read"):
    wire = run / "wire"
    wire.mkdir(exist_ok=True)
    (run / "sends.jsonl").write_text(json.dumps({
        "send": 1,
        "request_id": "request-1",
        "completion_tokens": 10,
    }) + "\n")
    (wire / "0001.response.json").write_text(json.dumps({"usage": {
        "prompt_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 40},
        "completion_tokens": 10,
    }}))
    (wire / "0001.route.json").write_text(json.dumps({
        "role": role,
        "scope": "main",
        "unit_class": None,
        "escalated_from": None,
    }))


def test_rates_only_routes_are_saved_for_accounting_without_activating_wire_routes(tmp_path):
    options = _route_only_options(tmp_path)

    table = runner._resolve_routes(options)
    assert not table.active
    runner._probe_routes(options, lambda *_args: None)
    assert options._routes is None

    run_dir, _, spec = runner.prepare(options)

    assert "routes" not in spec
    accounting = spec["route_accounting"]
    assert accounting["schema"] == "stateframe-route-accounting/1"
    assert accounting["roles"]["read"] == RATES
    assert {"plan", "edit", "finish", "decide2", "unit"} <= accounting["roles"].keys()
    assert json.loads((run_dir / "spec.json").read_text())["route_accounting"] == accounting


def test_accounting_rates_price_attributed_sends(tmp_path):
    (tmp_path / "spec.json").write_text(json.dumps({"route_accounting": _route_accounting()}))
    _record_send(tmp_path)

    summary = runner.summarize(tmp_path, time.monotonic())

    assert summary["route_costs"]["roles"]["read"]["estimated_cost"] == 0.00012
    assert summary["route_costs"]["total"]["estimated_cost"] == 0.00012
    assert summary["route_costs"]["unattributed_sends"] == 0


def test_accounting_rates_report_missing_attribution(tmp_path):
    (tmp_path / "spec.json").write_text(json.dumps({"route_accounting": _route_accounting()}))
    (tmp_path / "sends.jsonl").write_text(json.dumps({
        "send": 1,
        "request_id": "request-1",
        "completion_tokens": 10,
    }) + "\n")
    wire = tmp_path / "wire"
    wire.mkdir()
    (wire / "0001.response.json").write_text(json.dumps({"usage": {
        "prompt_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 40},
        "completion_tokens": 10,
    }}))

    summary = runner.summarize(tmp_path, time.monotonic())

    assert summary["route_costs"]["unattributed_sends"] == 1
    assert summary["route_costs"]["total"]["estimated_cost"] is None


def test_accounting_rates_keep_attributed_unpriced_roles_distinct(tmp_path):
    options = _route_only_options(tmp_path)
    runner._resolve_routes(options)
    assert options._route_accounting is not None
    (tmp_path / "spec.json").write_text(json.dumps({"route_accounting": options._route_accounting}))
    _record_send(tmp_path, role="edit")

    summary = runner.summarize(tmp_path, time.monotonic())

    assert summary["route_costs"]["unattributed_sends"] == 0
    assert summary["route_costs"]["roles"]["edit"]["requests"] == 1
    assert summary["route_costs"]["roles"]["edit"]["estimated_cost"] is None
    assert summary["route_costs"]["total"]["estimated_cost"] is None


def test_accounting_rates_keep_unit_price_overrides(tmp_path):
    options = runner.RunOptions(
        repo=None,
        task="Inspect routes",
        endpoint=Endpoint("http://127.0.0.1:8080/v1", "base"),
        route_cli=[
            "read.cost_input_per_mtok=1",
            "read.cost_cached_input_per_mtok=1",
            "read.cost_output_per_mtok=2",
            "unit.cost_input_per_mtok=10",
            "unit.patch.cost_input_per_mtok=12",
        ],
    )
    runner._resolve_routes(options)
    assert options._route_accounting is not None
    (tmp_path / "spec.json").write_text(json.dumps({"route_accounting": options._route_accounting}))
    _record_send(tmp_path)
    route_path = tmp_path / "wire" / "0001.route.json"
    route_path.write_text(json.dumps({
        "role": "read",
        "scope": "unit",
        "unit_class": "patch",
        "escalated_from": None,
    }))

    summary = runner.summarize(tmp_path, time.monotonic())

    assert options._route_accounting["unit_overrides"] == {
        "unit": ["cost_input_per_mtok"],
        "unit.patch": ["cost_input_per_mtok"],
    }
    assert summary["route_costs"]["roles"]["read"]["estimated_cost"] == 0.00078


def test_legacy_missing_run_spec_keeps_not_recorded_defaults(tmp_path):
    summary = runner.summarize(tmp_path, time.monotonic())

    assert summary["prompt_cache_slot"]["selection"] == "not_recorded"
    assert summary["capabilities"] == {}
    assert summary["provider_capabilities"] == {}
    assert "run_spec_diagnostic" not in summary


def test_valid_run_spec_metadata_is_preserved(tmp_path):
    recorded = {
        "prompt_cache_slot": {"slot": 2, "selection": "explicit"},
        "capabilities": {"display": "pytest"},
        "provider_capabilities": {"local": {"dialect": "openai-strict"}},
    }
    (tmp_path / "spec.json").write_text(json.dumps(recorded))

    summary = runner.summarize(tmp_path, time.monotonic())

    assert summary["prompt_cache_slot"] == recorded["prompt_cache_slot"]
    assert summary["capabilities"] == recorded["capabilities"]
    assert summary["provider_capabilities"] == recorded["provider_capabilities"]
    assert "run_spec_diagnostic" not in summary


def test_malformed_run_spec_has_typed_diagnostic_and_unavailable_fields(tmp_path):
    (tmp_path / "spec.json").write_text("{")

    summary = runner.summarize(tmp_path, time.monotonic())

    assert summary["run_spec_diagnostic"] == {
        "kind": "parse_error",
        "error_type": "JSONDecodeError",
    }
    assert summary["prompt_cache_slot"] is None
    assert summary["capabilities"] is None
    assert summary["provider_capabilities"] is None
    assert summary["route_costs"] is None


def test_unreadable_run_spec_has_typed_diagnostic_and_unavailable_fields(tmp_path, monkeypatch):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{}")
    original_read_text = Path.read_text

    def fail_spec_read(path, *args, **kwargs):
        if path == spec_path:
            raise PermissionError("unavailable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_spec_read)
    summary = runner.summarize(tmp_path, time.monotonic())

    assert summary["run_spec_diagnostic"] == {
        "kind": "read_error",
        "error_type": "PermissionError",
    }
    assert summary["prompt_cache_slot"] is None
    assert summary["capabilities"] is None
    assert summary["provider_capabilities"] is None
    assert summary["route_costs"] is None

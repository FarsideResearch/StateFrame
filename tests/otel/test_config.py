from __future__ import annotations
import os

from pathlib import Path

import pytest

from stateframe.otel import cli, config


@pytest.fixture(autouse=True)
def installed_config_module(monkeypatch, private_root):
    location = private_root / "installation/src/stateframe/otel/config.py"
    monkeypatch.setattr(config, "__file__", str(location))


def _config_file(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "operator.toml"
    path.write_text(body, encoding="utf-8")
    path.chmod(0o600)
    return path


@pytest.mark.parametrize(
    ("winner", "settings_values", "environment", "toml_values", "expected"),
    [
        ("cli", {"protocol": "http/json"},
         {"OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf"},
         '[otlp]\nprotocol = "http/protobuf"\n[traces]\nprotocol = "http/protobuf"\n',
         "http/json"),
        ("signal_env", {},
         {"OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
          "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf"},
         '[otlp]\nprotocol = "http/json"\n[traces]\nprotocol = "http/json"\n',
         "http/protobuf"),
        ("generic_env", {}, {"OTEL_EXPORTER_OTLP_PROTOCOL": "http/json"},
         '[otlp]\nprotocol = "http/protobuf"\n[traces]\nprotocol = "http/protobuf"\n',
         "http/json"),
        ("signal_toml", {}, {},
         '[otlp]\nprotocol = "http/json"\n[traces]\nprotocol = "http/protobuf"\n',
         "http/protobuf"),
        ("generic_toml", {}, {}, '[otlp]\nprotocol = "http/json"\n', "http/json"),
        ("default", {}, {}, "", "http/protobuf"),
    ],
)
def test_cli_env_config_precedence_matrix(
    private_root, winner, settings_values, environment, toml_values, expected
):
    path = _config_file(private_root, toml_values) if toml_values else None
    settings = config.Settings(config=path, **settings_values)

    result = config.resolve(settings, environment)

    assert result.traces.protocol == expected, winner


def test_config_file_rejects_world_writable_ancestor(private_root):
    parent = private_root / "shared"
    parent.mkdir()
    parent.chmod(0o777)
    path = _config_file(parent, '[otlp]\nprotocol = "http/json"\n')

    result = config.resolve(config.Settings(config=path), {})

    assert result.traces is None
    assert "invalid_config" in result.diagnostics


@pytest.mark.parametrize("command", ["status", "export", "purge"])
def test_cli_parser_does_not_echo_private_arguments(command, tmp_path, monkeypatch, capsys):
    run = str(tmp_path / "private-positional-canary")
    prefix = {"status": [run], "export": [run]}.get(command, [])
    args = [command, *prefix, "--entry=private-setting-canary"]

    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert "private-" not in captured.err + captured.out
    assert captured.err == "stateframe-otel: unsupported_setting\n"


@pytest.mark.parametrize("timeout_ms", [0, -1, 99, 30_001, True, "10000"])
def test_timeout_units_limits_and_zero(monkeypatch, timeout_ms):
    result = config.resolve(config.Settings(timeout_ms=timeout_ms), {})

    assert result.traces is None
    assert "invalid_config" in result.diagnostics


def test_otlp_signal_endpoint_joining(monkeypatch):
    result = config.resolve(config.Settings(), {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318/base",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://127.0.0.1:4318/custom",
    })

    assert result.traces.endpoint == "http://127.0.0.1:4318/custom"


@pytest.mark.parametrize("empty_signal_env", [False, True])
def test_endpoint_joining_uses_the_winning_source(private_root, empty_signal_env):
    path = _config_file(private_root, '[traces]\nendpoint = "http://127.0.0.1:4318/lower"\n')
    env = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318/base"}
    if empty_signal_env:
        env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = ""
    result = config.resolve(config.Settings(config=path), env)
    assert result.traces.endpoint == "http://127.0.0.1:4318/base/v1/traces"


def test_both_http_encodings_no_grpc_or_fallback(monkeypatch):
    json_result = config.resolve(config.Settings(protocol="http/json"), {})
    protobuf_result = config.resolve(config.Settings(protocol="http/protobuf"), {})
    unsupported = config.resolve(config.Settings(protocol="grpc"), {})

    assert json_result.traces.protocol == "http/json"
    assert protobuf_result.traces.protocol == "http/protobuf"
    assert unsupported.traces is None
    assert "unsupported_setting" in unsupported.diagnostics


def test_header_percent_escaping_and_reserved_names(monkeypatch):
    result = config.resolve(config.Settings(), {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector.example.test:4318",
        "OTEL_EXPORTER_OTLP_HEADERS": "x-note=one%2Ctwo,x-plus=a+b"
    })

    assert result.traces.headers == (("x-note", "one,two"), ("x-plus", "a+b"))
    secret = "opaque-" + os.urandom(16).hex()
    invalid = config.resolve(config.Settings(), {
        "OTEL_EXPORTER_OTLP_HEADERS": f"Host={secret}"
    })
    assert invalid.traces is None
    assert secret not in repr(invalid)
    assert "invalid_config" in invalid.diagnostics


@pytest.mark.parametrize(
    ("sampler", "ratio", "expected_sampler", "expected_ratio"),
    [
        (None, None, "always_on", 1.0),
        ("always_on", None, "always_on", 1.0),
        ("always_off", None, "always_off", 0.0),
        ("traceidratio", "0.125", "traceidratio", 0.125),
        ("traceidratio", "1", "traceidratio", 1.0),
    ],
)
def test_sampler_matrix_and_unsupported_parents(
    monkeypatch, sampler, ratio, expected_sampler, expected_ratio
):
    result = config.resolve(config.Settings(sample=sampler, sample_ratio=ratio), {})

    assert result.sampler == expected_sampler
    assert result.sample_ratio == expected_ratio


def test_sampler_matrix_rejects_parent_sampler(monkeypatch):
    result = config.resolve(config.Settings(), {"OTEL_TRACES_SAMPLER": "parentbased_always_on"})

    assert result.sampler == "always_off"
    assert "unsupported_setting" in result.diagnostics


def test_no_context_propagation_or_baggage(monkeypatch):
    result = config.resolve(config.Settings(), {
        "OTEL_PROPAGATORS": "tracecontext,baggage",
        "TRACEPARENT": "00-" + "1" * 32 + "-" + "2" * 16 + "-01",
        "OTEL_BAGGAGE": "opaque=value",
    })

    assert not hasattr(result, "propagators")
    assert "unsupported_setting" in result.diagnostics
    assert "opaque=value" not in repr(result)


def test_unsupported_sdk_limits_cannot_override_hard_caps(monkeypatch):
    result = config.resolve(config.Settings(), {
        "OTEL_ATTRIBUTE_COUNT_LIMIT": "999999999",
        "OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT": "999999999",
    })

    assert result.traces is not None
    assert "unsupported_setting" in result.diagnostics
    assert "999999999" not in repr(result)


def test_conflicting_toml_setting_spellings_are_invalid(private_root):
    path = _config_file(private_root, "[otlp]\ntimeout_ms = 250\nmax_export_time_milliseconds = 500\n")
    assert config.resolve(config.Settings(config=path), {}).diagnostics == ("invalid_config",)


def test_cost_detail_uses_only_operator_settings(private_root):
    path = _config_file(private_root, """[detail]
include_cost = true
price_per_million_input_tokens = 2
price_per_million_output_tokens = 4.5
""")
    result = config.resolve(config.Settings(config=path, include_cost=False),
                            {"OTEL_STATEFRAME_INCLUDE_COST": "true"})
    assert result.levels == config.Levels(False)
    assert result.price_rates == (2, 4.5)
    assert config.resolve(config.Settings(), {}).levels == config.Levels()
    invalid = config.resolve(config.Settings(), {"OTEL_STATEFRAME_INCLUDE_COST": "private-value"})
    assert invalid.traces is None
    assert invalid.diagnostics == ("invalid_config",)


@pytest.mark.parametrize("endpoint", [
    "http://127.0.0.1:4318/v1/traces?signature=private-canary",
    "http://127.0.0.1:4318/v1/traces?",
    "http://127.0.0.1:4318/v1/traces#fragment",
    "http://user@127.0.0.1:4318/v1/traces",
    "https://collector.example.test/v1/traces?tenant=private-canary",
    "https://collector.example.test/v1/traces#fragment",
    "https://user@collector.example.test/v1/traces",
    "http://collector.example.test:4318/v1/traces",
    "http://192.0.2.1:4318/v1/traces",
])
def test_endpoints_refuse_queries_fragments_userinfo_and_non_loopback_http(endpoint):
    result = config.resolve(config.Settings(), {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": endpoint})

    assert result.traces is None
    assert result.diagnostics == ("invalid_config",)


def test_https_host_and_http_loopback_literal_remain_valid():
    https = config.resolve(config.Settings(), {
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "https://collector.example.test/v1/traces",
    })
    http = config.resolve(config.Settings(), {
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://[::1]:4318/v1/traces",
    })

    assert https.traces is not None
    assert http.traces is not None


@pytest.mark.parametrize("command", ["status", "export"])
def test_cli_explains_unsafe_parent_permissions_without_exposing_path(
    command, private_root, capsys, monkeypatch,
):
    from stateframe.otel import companion

    shared = private_root / "private-directory-canary"
    shared.mkdir()
    shared.chmod(0o777)
    run = shared / "run"
    run.mkdir(mode=0o700)
    monkeypatch.setattr(companion, "status", lambda *_args, **_kwargs: (0, {}))

    assert cli.main([command, str(run)]) == 2
    diagnostic = capsys.readouterr().err
    assert "permission refused" in diagnostic
    assert "private-directory-canary" not in diagnostic

"""Route source precedence, validation, and public spec contracts."""
import json

import pytest

from stateframe import routes


DEFAULT = {"endpoint": "http://127.0.0.1:8080/v1", "model": "base", "api_key_env": None,
           "temperature": 1.0, "max_tokens": 24000, "timeout_s": 1800.0,
           "slot": None, "context_tokens": 65536, "allow_remote": False}


def resolve(**kwargs):
    return routes.resolve(DEFAULT, environ={}, **kwargs)


def test_sources_merge_per_field_and_digest_ignores_source(tmp_path):
    path = tmp_path / "routes.toml"
    path.write_text('[roles.read]\nmodel = "file"\ntop_p = 0.4\n')
    profile = {"roles": {"read": {"model": "profile", "top_p": 0.2, "top_k": 8}}}
    env = {"STATEFRAME_ROUTE_READ_MODEL": "env"}
    table = routes.resolve(DEFAULT, cli=["read.model=cli"], file=path, profile=profile, environ=env)
    assert table.roles["read"]["model"] == "cli"
    assert table.roles["read"]["top_p"] == 0.4
    assert table.roles["read"]["top_k"] == 8
    assert table.sources["read"] == {**{key: "default" for key in DEFAULT},
                                     "model": "cli", "top_p": "file", "top_k": "profile"}
    same = routes.resolve(DEFAULT, cli=["read.model=cli", "read.top_p=0.4", "read.top_k=8"], environ={})
    assert table.digest == same.digest


def test_no_recorded_layer_preserves_inactive_example_and_plain_table(tmp_path):
    path = tmp_path / "routes.toml"
    path.write_text(routes.example())
    plain = resolve()
    example = resolve(file=path)
    assert not plain.active
    assert example.public() == plain.public()
    assert example.sources == plain.sources
    assert resolve(recorded=None).public() == plain.public()


def test_recorded_unit_class_override_survives_without_source_file():
    started = resolve(cli=["read.model=small", "unit.patch.temperature=0.2"])
    resumed = resolve(recorded=started.public(), recorded_default=DEFAULT)
    assert resumed.public() == started.public()
    assert resumed.sources["unit.patch"]["temperature"] == "recorded"
    assert resumed.resolve("read", unit_class="patch")["temperature"] == 0.2


def test_recorded_routes_survive_removed_file_and_env(tmp_path):
    path = tmp_path / "routes.toml"
    path.write_text('[roles.read]\nendpoint = "http://127.0.0.1:8090/v1"\n')
    started = routes.resolve(DEFAULT, file=path, environ={"STATEFRAME_ROUTE_READ_MODEL": "small"})
    path.unlink()
    resumed = resolve(recorded=started.public(), recorded_default=DEFAULT)
    assert resumed.public() == started.public()
    assert resumed.sources["read"]["endpoint"] == "recorded"
    assert resumed.sources["read"]["model"] == "recorded"


def test_json_schema_and_unit_class_lookup(tmp_path):
    path = tmp_path / "routes.json"
    path.write_text(json.dumps({"schema": routes.SCHEMA, "roles": {
        "unit": {"model": "unit-model", "temperature": 0.2},
        "unit.patch": {"model": "patch-model"}}}))
    table = routes.resolve(DEFAULT, file=path, environ={})
    assert table.resolve("read", unit_class="patch")["model"] == "patch-model"
    assert table.resolve("read", unit_class="patch")["temperature"] == 0.2
    assert table.resolve("read", unit_class="other")["model"] == "unit-model"
    assert table.public()["unit_overrides"] == {"unit": ["model", "temperature"],
                                                "unit.patch": ["model"]}
    env = {"STATEFRAME_ROUTE_UNIT_PATCH_MAX_TOKENS": "512"}
    assert routes.resolve(DEFAULT, file=path, environ=env).roles["unit.patch"]["max_tokens"] == 512


def test_toml_nested_unit_class_table(tmp_path):
    path = tmp_path / "routes.toml"
    path.write_text('[roles.unit]\nmodel = "shared"\n[roles.unit.patch]\nmodel = "patch"\n')
    table = resolve(file=path)
    assert table.resolve("read", unit_class="patch")["model"] == "patch"
    assert table.resolve("read", unit_class="other")["model"] == "shared"


def test_duplicate_nested_and_dotted_unit_class_is_refused():
    profile = {"roles": {"unit": {"patch": {"model": "nested"}},
                         "unit.patch": {"model": "dotted"}}}
    with pytest.raises(routes.RouteError, match="duplicate route role unit.patch"):
        resolve(profile=profile)


def test_unit_override_mask_changes_digest_even_when_value_equals_default():
    plain = resolve(cli=["plan.model=large"])
    unit = resolve(cli=["plan.model=large", "unit.model=base"])
    assert plain.roles == unit.roles
    assert plain.digest != unit.digest
    assert unit.resolve("plan", unit_class="other")["model"] == "base"


def test_unit_default_model_changes_decide2_request_and_activates():
    table = resolve(cli=["unit.model=base"])
    assert table.resolve("decide2", unit_class="default")["model"] == "base"
    assert table.active


@pytest.mark.parametrize("suffix,body", [
    (".toml", '[roles.read]\nallow_remote = "true"\n'),
    (".toml", '[roles.read]\ntemperature = "0.3"\n'),
    (".json", '{"schema":"stateframe-routes/1","roles":{"read":{"allow_remote":"true"}}}'),
    (".json", '{"schema":"stateframe-routes/1","roles":{"read":{"temperature":"0.3"}}}'),
    (".json", '{"schema":"stateframe-routes/1","roles":{"read":{"slot":"2"}}}'),
])
def test_file_values_require_native_types(tmp_path, suffix, body):
    path = tmp_path / ("routes" + suffix)
    path.write_text(body)
    with pytest.raises(routes.RouteError, match="route read"):
        resolve(file=path)


def test_profile_values_require_native_types():
    with pytest.raises(routes.RouteError, match="route read.top_p"):
        resolve(profile={"roles": {"read": {"top_p": "0.3"}}})
    assert resolve(cli=["read.top_p=0.3", "read.allow_remote=true"]).roles["read"]["top_p"] == 0.3


def test_inactive_and_public_view_never_contain_key_value():
    assert not resolve().active
    assert not resolve(cli=["read.model=base"]).active
    assert not resolve(cli=["read.cost_input_per_mtok=1.0", "read.allow_remote=true"]).active
    table = routes.resolve(DEFAULT, cli=["edit.api_key_env=OPENAI_KEY"],
                           environ={"OPENAI_KEY": "very-secret-value"})
    assert table.active
    assert table.public()["roles"]["edit"]["api_key_env"] == "OPENAI_KEY"
    assert "very-secret-value" not in json.dumps(table.public())


def test_default_endpoint_tokenizer_does_not_activate_routes():
    table = resolve(cli=["read.tokenizer=http://127.0.0.1:8080/v1"])
    assert not table.active


def test_unit_overlay_cannot_remove_remote_opt_in():
    with pytest.raises(routes.RouteError, match="route read.*unit.*allow_remote"):
        resolve(cli=["read.endpoint=https://api.example.org/v1", "read.allow_remote=true",
                     "unit.allow_remote=false"])


def test_profile_routes_value_must_be_a_table():
    with pytest.raises(routes.RouteError, match="profile routes must contain a roles table"):
        resolve(profile=[])


def test_oversized_cli_integer_has_route_error():
    with pytest.raises(routes.RouteError, match="route read.max_tokens"):
        resolve(cli=["read.max_tokens=" + "9" * 5000])


def test_oversized_json_rate_has_route_error(tmp_path):
    path = tmp_path / "routes.json"
    path.write_text(json.dumps({"schema": routes.SCHEMA, "roles": {
        "read": {"cost_input_per_mtok": 10 ** 1000}}}))
    with pytest.raises(routes.RouteError, match="route read.cost_input_per_mtok"):
        resolve(file=path)


@pytest.mark.parametrize("cli,fragment", [
    (["reviewer.model=x"], "unknown route role 'reviewer'"),
    (["read.bad=x"], "unknown route field read.bad"),
    (["read.model="], "nonempty model name"),
    (["read.api_key_env=bad-key"], "environment variable NAME"),
    (["read.slot=-1"], "nonnegative integer"),
    (["read.context_tokens=0"], "positive integer"),
    (["read.temperature=inf"], "finite number"),
    (["read.top_p=1.2"], "finite number"),
    (["read.allow_remote=yes"], "true or false"),
    (["read.endpoint=http://example.org/v1"], "not on this machine"),
    (["read.api_key_env=MISSING"], "export it or remove api_key_env"),
])
def test_actionable_refusals(cli, fragment):
    with pytest.raises(routes.RouteError, match=fragment):
        resolve(cli=cli)


def test_file_refusals_and_remote_opt_in(tmp_path):
    path = tmp_path / "routes.json"
    path.write_text('{"roles":{"read":{}}}')
    with pytest.raises(routes.RouteError, match="needs schema"):
        resolve(file=path)
    path.write_text('{"schema":"stateframe-routes/1","roles":{"read":{"bogus":1}}}')
    with pytest.raises(routes.RouteError, match="unknown route field"):
        resolve(file=path)
    table = resolve(cli=["read.endpoint=https://api.example.org/v1", "read.allow_remote=true"])
    assert table.roles["read"]["allow_remote"] is True


def test_commented_example_is_inactive(tmp_path):
    path = tmp_path / "routes.toml"
    path.write_text(routes.example())
    assert not resolve(file=path).active
    from pathlib import Path
    shipped = Path(__file__).resolve().parents[1] / "examples" / "routes.example.toml"
    assert shipped.read_text() == routes.example()
    assert not resolve(file=shipped).active


def test_decide2_alias_is_not_another_model():
    # Regression: decide2's built-in "decision-tier" alias made every routed run with
    # LLAMA_CPP_TOKENIZE_URL refuse to start (runner and engine child both checked it).
    assert not routes.uses_other_model("decide2", "decision-tier", "qwen")
    assert not routes.uses_other_model("read", "qwen", "qwen")
    assert routes.uses_other_model("read", "decision-tier", "qwen")
    assert routes.uses_other_model("decide2", "small", "qwen")

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


def test_unit_override_mask_changes_digest_even_when_value_equals_default():
    plain = resolve(cli=["plan.model=large"])
    unit = resolve(cli=["plan.model=large", "unit.model=base"])
    assert plain.roles == unit.roles
    assert plain.digest != unit.digest
    assert unit.resolve("plan", unit_class="other")["model"] == "base"


def test_inactive_and_public_view_never_contain_key_value():
    assert not resolve().active
    assert not resolve(cli=["read.model=base"]).active
    assert not resolve(cli=["read.cost_input_per_mtok=1.0", "read.allow_remote=true"]).active
    table = routes.resolve(DEFAULT, cli=["edit.api_key_env=OPENAI_KEY"],
                           environ={"OPENAI_KEY": "very-secret-value"})
    assert table.active
    assert table.public()["roles"]["edit"]["api_key_env"] == "OPENAI_KEY"
    assert "very-secret-value" not in json.dumps(table.public())


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

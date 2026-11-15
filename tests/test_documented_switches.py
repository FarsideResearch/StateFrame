"""Every STATEFRAME_* name the public docs tell users to set must actually take effect.

A setting reaches StateFrame in one of three ways: an engine opt-in switch forwarded from the shell through
runner.OPT_IN_SWITCHES, a product setting the CLI reads itself (src/stateframe/*.py outside the engine), or an engine
switch that only a custom profile can set, which the docs must say. A name in none of these is silently ignored when a
user exports it, which is the bug class this file guards (two new switches once shipped docs saying "set it in the
shell" without the allowlist entry).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from stateframe import runner
from stateframe.engine.kernel_parallel_records import PARALLEL_LIMIT_OPTIONS, ParallelLimits

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = ("README.md", "CHANGELOG.md", "KNOWN_LIMITATIONS.md", "AGENTS.md", "docs/guide/*.md")
# Engine switches documented as profile-only: the docs must say "profile" in the same sentence.
PROFILE_ONLY = {
    "STATEFRAME_BOUNDARY_CHECKS",
    "STATEFRAME_HOST_CLOSURE",
    "STATEFRAME_RESULTS_DIR",
    "STATEFRAME_CODE_BOOKMARKS",
    "STATEFRAME_READ_LOCKOUT",
    "STATEFRAME_ROUTEBUDGET_MODE",
    "STATEFRAME_HTTP_GZIP",
    "STATEFRAME_PLAIN_GZIP",
    "STATEFRAME_PAGE_LOG",
    "STATEFRAME_PROVIDER_LIVENESS_S",
    "STATEFRAME_PUBLIC_CHECK_CPU_SECONDS",
    "STATEFRAME_PROVIDER_INDEX_MAX_FILES",
    "STATEFRAME_BOUNDARY_CHECKS",
    "STATEFRAME_RESULTS_DIR",
    "STATEFRAME_LOCALIZE_MAX_FILES",
    "STATEFRAME_LOCALIZE_MAX_SOURCE_BYTES",
}
# Placeholders in the extension guide and prefixes of per-role variables, not settings.
EXAMPLES = {"STATEFRAME_MY_SWITCH"}
# Test-fixture variables named in the CHANGELOG: read only by the test suite, never by the product.
TEST_FIXTURE_ONLY = {"STATEFRAME_RECORDED_WIRE"}
PREFIXES = {"STATEFRAME_ROUTE_"}
NAME = re.compile(r"STATEFRAME_[A-Z0-9_]+")


def _documented(root: Path = ROOT) -> dict[str, list[str]]:
    """Each documented STATEFRAME_* name with the sentences that mention it."""
    names: dict[str, list[str]] = {}
    for pattern in PUBLIC_DOCS:
        for path in sorted(root.glob(pattern)):
            for sentence in re.split(r"(?<=[.!?])\s+|\n", path.read_text(encoding="utf-8")):
                for name in NAME.findall(sentence):
                    names.setdefault(name, []).append(sentence)
    return names


def _product_reads(root: Path = ROOT) -> set[str]:
    """STATEFRAME_* literals the product code (not the engine) contains: settings the CLI reads itself."""
    found: set[str] = set()
    for path in (root / "src/stateframe").glob("*.py"):
        found.update(NAME.findall(path.read_text(encoding="utf-8")))
    return found


def unsupported(documented: dict[str, list[str]], allowlist, product: set[str]) -> list[str]:
    """Documented names a user could export to no effect, with the reason."""
    problems = []
    for name, sentences in sorted(documented.items()):
        if name in EXAMPLES or name in PREFIXES or name in allowlist or name in TEST_FIXTURE_ONLY:
            continue
        if name in PROFILE_ONLY:
            if not any("profile" in sentence.lower() for sentence in sentences):
                problems.append(f"{name}: profile-only switch, but no doc sentence says to set it in a profile")
            continue
        if name not in product:
            problems.append(f"{name}: documented, but not in runner.OPT_IN_SWITCHES, not read by the CLI, "
                            "and not documented as profile-only")
    return problems


def test_every_documented_switch_takes_effect():
    assert unsupported(_documented(), runner.OPT_IN_SWITCHES, _product_reads()) == []


def test_test_fixture_variables_are_never_product_switches():
    product = _product_reads()
    engine = runner.engine_switches()
    for name in TEST_FIXTURE_ONLY:
        assert name not in runner.OPT_IN_SWITCHES
        assert name not in product
        assert name not in engine
        assert any(name in path.read_text(encoding="utf-8") for path in (ROOT / "tests").glob("test_*.py"))


def test_planted_violation_is_caught():
    documented = {"STATEFRAME_NOT_FORWARDED": ["Set `STATEFRAME_NOT_FORWARDED=on` in the shell."],
                  "STATEFRAME_CODE_BOOKMARKS": ["`STATEFRAME_CODE_BOOKMARKS` is on by default."]}
    problems = unsupported(documented, runner.OPT_IN_SWITCHES, set())
    assert any(p.startswith("STATEFRAME_NOT_FORWARDED:") for p in problems)
    assert any(p.startswith("STATEFRAME_CODE_BOOKMARKS:") for p in problems)


def test_terminal_parallel_schema_switch_is_documented_and_registered():
    name = "STATEFRAME_PARALLEL_DROP_TOOL_AFTER_TERMINAL"
    assert runner.OPT_IN_SWITCHES[name] == ("on", "off")
    assert name in runner.engine_switches()
    assert any("on by default" in sentence.lower() and "parallel" in sentence.lower()
               for sentence in _documented()[name])


def test_forced_manifest_switch_is_internal_benchmark_only_and_not_user_documented():
    name = "STATEFRAME_PARALLEL_FORCE_MANIFEST"
    assert runner.OPT_IN_SWITCHES[name] == ()
    assert name in runner.engine_switches()
    assert runner.INTERNAL_BENCHMARK_SWITCHES[name] == {
        "classification": "internal/benchmark-only",
        "stability": "may change",
    }
    assert name not in _documented()


def test_forced_manifest_switch_resolves_tilde_from_current_home(monkeypatch, tmp_path):
    name = "STATEFRAME_PARALLEL_FORCE_MANIFEST"
    monkeypatch.setenv(name, "~/forced-manifest.json")
    for home in (tmp_path / "first-home", tmp_path / "second-home"):
        monkeypatch.setenv("HOME", str(home))
        assert runner.resolve_opt_in_switches()[name] == str(home / "forced-manifest.json")


def test_stable_wire_headroom_is_documented_and_forwarded(monkeypatch):
    name = "STATEFRAME_STABLE_WIRE_HEADROOM"
    assert name in runner.OPT_IN_SWITCHES
    sentences = _documented()[name]
    assert any("16000" in sentence and "retained" in sentence.lower() for sentence in sentences)
    for switch in runner.OPT_IN_SWITCHES:
        monkeypatch.delenv(switch, raising=False)
    monkeypatch.setenv(name, "16000")
    monkeypatch.setattr(runner, "engine_switches", lambda _directory=runner.ENGINE_DIR: frozenset({name}))
    assert runner.resolve_opt_in_switches()[name] == "16000"


@pytest.mark.parametrize(
    "name", sorted(set(runner.OPT_IN_SWITCHES) - runner.HOST_ONLY_SWITCHES)
)
def test_shell_setting_reaches_the_engine_spec(monkeypatch, name):
    assert name in runner.engine_switches()
    for other in runner.OPT_IN_SWITCHES:
        monkeypatch.delenv(other, raising=False)
    # Numeric settings have no choice list; bounded percentages use a valid in-range value.
    choices = runner.OPT_IN_SWITCHES[name]
    value = (choices[0] if choices else
             "0.5" if name == "STATEFRAME_TRANSPORT_BACKOFF_S" else
             "25" if name == "STATEFRAME_PARALLEL_BYTE_TOKEN_HEADROOM_PCT" else "120")
    monkeypatch.setenv(name, value)
    expected = (str(Path(value).absolute())
                if name == "STATEFRAME_PARALLEL_FORCE_MANIFEST" else value)
    assert runner.resolve_opt_in_switches()[name] == expected


def unread_shell_switches(allowlist, engine: frozenset[str], host_only: frozenset[str]) -> list[str]:
    """Shell-settable names that neither the engine nor the host reads: exporting one does nothing."""
    return sorted(name for name in allowlist if name not in engine and name not in host_only)


def test_every_shell_switch_has_a_reader():
    # Without this, test_shell_setting_reaches_the_engine_spec skips an unread name instead of failing.
    assert unread_shell_switches(runner.OPT_IN_SWITCHES, runner.engine_switches(),
                                 runner.HOST_ONLY_SWITCHES) == []


def test_planted_unread_shell_switch_is_caught():
    allowlist = {**runner.OPT_IN_SWITCHES, "STATEFRAME_NO_READER": ("on", "off")}
    assert unread_shell_switches(allowlist, runner.engine_switches(),
                                 runner.HOST_ONLY_SWITCHES) == ["STATEFRAME_NO_READER"]


@pytest.mark.parametrize("setting,value", [
    ("max_units", 8),
    ("min_units", 3),
    ("max_paths_per_unit", 8),
    ("max_paths_total", 40),
    ("unit_request_tokens", 10000),
    ("max_owned_source_bytes", 30000),
    ("unit_max_requests", 16),
    ("parent_reserve_requests", 5),
    ("check_permits", 2),
    ("max_concurrency", 8),
])
def test_parallel_shell_limit_reaches_resolved_setting(monkeypatch, setting, value):
    for name in runner.OPT_IN_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    switch = PARALLEL_LIMIT_OPTIONS[setting]["switch"]
    monkeypatch.setenv(switch, str(value))
    forwarded = runner.resolve_opt_in_switches()
    assert forwarded[switch] == str(value)
    limits = ParallelLimits.from_settings(forwarded, {}, prompt_budget=33344)
    assert getattr(limits, setting) == value


def test_parallel_shell_slots_reach_host_pool(monkeypatch):
    from stateframe.parallel_resources import parse_slot_pool

    monkeypatch.setenv("STATEFRAME_PARALLEL_SLOTS", "2,5")
    forwarded = runner.resolve_opt_in_switches()
    assert parse_slot_pool(forwarded["STATEFRAME_PARALLEL_SLOTS"], source="test") == (2, 5)


def test_parallel_shell_settings_are_recorded_in_run_spec(tmp_path, monkeypatch):
    from stateframe.endpoint import Endpoint

    for name in runner.OPT_IN_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    values = {
        "max_units": 8, "min_units": 3, "max_paths_per_unit": 8,
        "max_paths_total": 40, "unit_request_tokens": 10000,
        "max_owned_source_bytes": 30000, "unit_max_requests": 16,
        "parent_reserve_requests": 5, "check_permits": 2,
        "max_concurrency": 8,
    }
    for setting, value in values.items():
        monkeypatch.setenv(PARALLEL_LIMIT_OPTIONS[setting]["switch"], str(value))
    monkeypatch.setenv("STATEFRAME_PARALLEL_SLOTS", "2,5")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "test_sample.py").write_text("def test_sample(): assert True\n", encoding="utf-8")
    endpoint = Endpoint.from_args("http://127.0.0.1:8080/v1", "local-model", None)
    options = runner.RunOptions(repo, "check", endpoint, run_dir=tmp_path / "run",
                                parallel_mode="2")
    _, _, spec = runner.prepare(options)

    assert {name: spec["parallel_limits"][name] for name in values} == values
    assert spec["parallel_slots"] == [2, 5]
    assert spec["parallel_subframe_max_requests_explicit"] is True
    assert not set(spec["environment"]).intersection(
        option["switch"] for option in PARALLEL_LIMIT_OPTIONS.values())


@pytest.mark.parametrize("name", ["STATEFRAME_BOUNDARY_CHECKS", "STATEFRAME_HOST_CLOSURE", "STATEFRAME_RESULTS_DIR"])
def test_engine_only_settings_require_profile_guidance(name):
    assert unsupported({name: [f"Set {name}=on in the shell."]}, {}, set()) == [
        f"{name}: profile-only switch, but no doc sentence says to set it in a profile"]
    assert unsupported({name: [f"Set {name}=on in a custom profile."]}, {}, set()) == []

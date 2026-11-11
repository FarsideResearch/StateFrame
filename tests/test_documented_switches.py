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

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = ("README.md", "CHANGELOG.md", "KNOWN_LIMITATIONS.md", "AGENTS.md", "docs/guide/*.md")
# Engine switches documented as profile-only: the docs must say "profile" in the same sentence.
PROFILE_ONLY = {
    "STATEFRAME_CODE_BOOKMARKS",
    "STATEFRAME_READ_LOCKOUT",
    "STATEFRAME_HTTP_GZIP",
    "STATEFRAME_PLAIN_GZIP",
    "STATEFRAME_PAGE_LOG",
    "STATEFRAME_PROVIDER_LIVENESS_S",
    "STATEFRAME_PUBLIC_CHECK_CPU_SECONDS",
}
# Placeholders in the extension guide and prefixes of per-role variables, not settings.
EXAMPLES = {"STATEFRAME_MY_SWITCH"}
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
        if name in EXAMPLES or name in PREFIXES or name in allowlist:
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


def test_planted_violation_is_caught():
    documented = {"STATEFRAME_NOT_FORWARDED": ["Set `STATEFRAME_NOT_FORWARDED=on` in the shell."],
                  "STATEFRAME_CODE_BOOKMARKS": ["`STATEFRAME_CODE_BOOKMARKS` is on by default."]}
    problems = unsupported(documented, runner.OPT_IN_SWITCHES, set())
    assert any(p.startswith("STATEFRAME_NOT_FORWARDED:") for p in problems)
    assert any(p.startswith("STATEFRAME_CODE_BOOKMARKS:") for p in problems)


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
    # Numeric settings have no choice list; any in-range number exercises the path.
    choices = runner.OPT_IN_SWITCHES[name]
    value = choices[0] if choices else (
        "0.5" if name == "STATEFRAME_TRANSPORT_BACKOFF_S" else "120")
    monkeypatch.setenv(name, value)
    assert runner.resolve_opt_in_switches()[name] == value


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

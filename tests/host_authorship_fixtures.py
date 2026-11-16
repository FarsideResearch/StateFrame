"""Shared fixtures for host-row identity tests.

A reader test registers the rows its fixture means to be host-authored through
``register_rows``. A row it does not register is an attacker row by construction.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import kernel_host_authorship as authorship

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_TASKS = {
    "discount_bug": "Fix the pricing code so the tests pass.",
    "tiny_site": "Build the small static shop site described in the README.",
    "brief_to_report": "Write the report the brief asks for.",
}
SHADOW_EVENTS = ("host_identity_shadow", "host_identity_missing", "host_authorship_manifest")


@contextlib.contextmanager
def host_authorship_scope(mode: str, *, step: int = 1,
                          request_id: str | None = "request") -> Iterator[authorship.HostAuthorship | None]:
    """Set the switch for the block and yield a ledger (None when off)."""
    previous = os.environ.get(authorship.MODE_ENV)
    os.environ[authorship.MODE_ENV] = mode
    try:
        yield None if authorship.mode() == "off" else authorship.HostAuthorship(step, request_id)
    finally:
        if previous is None:
            os.environ.pop(authorship.MODE_ENV, None)
        else:
            os.environ[authorship.MODE_ENV] = previous


def register_rows(ledger: authorship.HostAuthorship | None, messages: Sequence[Mapping[str, object]],
                  spec: Mapping[int, object]) -> None:
    """Register ``messages[index]`` as host rows.

    ``spec[index]`` is a kind, or ``(kind, data)`` where data is a list of
    ``(start, length)`` spans or of substrings of the content that are data.
    """
    if ledger is None:
        return
    for index, entry in spec.items():
        kind, data = (entry, ()) if isinstance(entry, str) else entry
        row = messages[index]
        content = row["content"]
        spans = []
        cursor = 0
        for item in data:
            if isinstance(item, str):
                start = content.index(item, cursor)
                spans.append((start, len(item)))
                cursor = start + len(item)
            else:
                spans.append(tuple(item))
        if kind in {"panel", "owner_block"} and not spans:
            # The fixture declares the whole row host text.
            built = authorship.builder(ledger).host(content)
            authorship.author_row(ledger, row["role"], built, kind=kind, site=f"fixture.{index}")
        else:
            authorship.author_row(ledger, row["role"], content, kind=kind, site=f"fixture.{index}",
                                  data=spans)


def run_scripted_example(tmp_path: Path, example: str, mode: str | None, *, label: str | None = None,
                         extra_env: Mapping[str, str] | None = None) -> dict[str, object]:
    """Drive one scripted example to its sealed wires with the switch at ``mode`` (None: unset)."""
    from _journal import measurement_events
    from stateframe.scripted_endpoint import Script, serve
    from test_final_cache_layout_host import (
        RUN_TIMEOUT_SECONDS, _parity_launchers, _scripted_environment)

    label = label or (mode or "unset")
    repo, run = tmp_path / "repo", tmp_path / "run"
    shutil.copytree(ROOT / "examples" / example, repo)
    launch = tmp_path / f"launch-{label}"
    launch.mkdir()
    launcher, wrapper = _parity_launchers(launch)
    script = Script(json.loads((ROOT / "examples" / f"{example}.script.json").read_text()))
    server = serve(script)
    environment = _scripted_environment(ROOT)
    environment.pop(authorship.MODE_ENV, None)
    if mode is not None:
        environment[authorship.MODE_ENV] = mode
    environment["STATEFRAME_PAGE_LOG"] = "off"
    environment.update(extra_env or {})
    try:
        result = subprocess.run(
            [sys.executable, str(launcher), str(wrapper), "run", "--repo", str(repo),
             "--task", EXAMPLE_TASKS[example],
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--model", "scripted", "--run-dir", str(run),
             "--test-python", sys.executable, "--max-turns", "16", "--quiet"],
            env=environment, capture_output=True, text=True, timeout=RUN_TIMEOUT_SECONDS)
        assert result.returncode == 0, result.stdout + result.stderr
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()
    events = measurement_events(run)
    wires = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
    # Same repo and run paths for every arm (paths reach the request bytes); renamed afterwards.
    run.rename(tmp_path / f"run-{label}")
    repo.rename(tmp_path / f"repo-{label}")
    named = {name: [row for row in events if (row.get("kind") or row.get("event")) == name]
             for name in SHADOW_EVENTS}
    return {"wires": wires, "events": events, "shadow": named["host_identity_shadow"],
            "missing": named["host_identity_missing"], "manifests": named["host_authorship_manifest"]}


def run_full_path_shadow(tmp_path: Path, example: str = "discount_bug") -> dict[str, object]:
    """The scripted run with the switch at ``shadow``: its shadow and missing events."""
    return run_scripted_example(tmp_path, example, "shadow")

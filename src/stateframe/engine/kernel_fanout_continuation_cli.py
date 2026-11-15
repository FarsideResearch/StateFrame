"""Read captured parallel artifacts and expose the continuation renderer CLI."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from kernel_fanout_continuation import (
    FanoutContinuationError, MAX_CAPTURE_FILE_BYTES, MAX_SECTION_BYTES, _initial_remaining,
    _bytes, _relative_path, _selectors, _unit_rows, disposition_from_records, render,
)
from kernel_fanout_pin import _source_file


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CAPTURE_FILE_BYTES:
        raise FanoutContinuationError("recorded fan-out artifact is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise FanoutContinuationError("recorded fan-out artifact is invalid") from error
    if not isinstance(value, dict):
        raise FanoutContinuationError("recorded fan-out artifact is not an object")
    return value


def _failure_from_text(text: str, workspace: Path, units: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    test_match = re.search(r"_{3,}\s*(test_[A-Za-z0-9_]+)\s*_{3,}", text)
    location_match = re.search(r"(?m)^([A-Za-z0-9_./-]+\.py):(\d+): AssertionError", text)
    observed_match = re.search(
        r"(?m)^E\s+(?:AssertionError:\s*)?assert\s+(.+?)\s*==\s*(.+?)\s*$", text,
    )
    if test_match is None or location_match is None:
        return None
    path_text = _relative_path(location_match.group(1))
    line_number = int(location_match.group(2))
    if path_text is None:
        return None
    unit_id = next((str(unit.get("id")) for unit in units
                    if any(path_text == selector.split("::", 1)[0]
                           for selector in _selectors(unit))), None)
    if unit_id is None:
        unresolved = _initial_remaining({}, units)
        unit_id = unresolved[0] if len(unresolved) == 1 else None
    assertion = ""
    test_source = _source_file(workspace, path_text)
    if test_source is not None:
        try:
            lines = test_source[1].decode("utf-8").splitlines()
            if 1 <= line_number <= len(lines):
                assertion = lines[line_number - 1].strip()
        except UnicodeDecodeError:
            pass
    observed = observed_match.group(1).strip() if observed_match else "unknown"
    expected = observed_match.group(2).strip() if observed_match else "unknown"
    return {
        "unit_id": unit_id or "unknown",
        "nodeid": f"{path_text}::{test_match.group(1)}",
        "location": f"{path_text}:{line_number}",
        "expected": expected,
        "observed": observed,
        "assertion": assertion or "check reported failure",
    }


def load_recorded_run(run_dir: str | Path) -> tuple[dict[str, object], Path]:
    """Load the latest captured parallel result and its parent snapshot, read-only."""
    root = Path(run_dir)
    try:
        root_resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FanoutContinuationError("recorded run directory is unavailable") from error
    if root.is_symlink() or not root_resolved.is_dir():
        raise FanoutContinuationError("recorded run directory is unsafe")
    tape_path = root_resolved / "tool-tape.jsonl"
    if tape_path.is_symlink() or not tape_path.is_file() or tape_path.stat().st_size > MAX_CAPTURE_FILE_BYTES:
        raise FanoutContinuationError("recorded run has no bounded tool tape")
    completed = []
    try:
        for line in tape_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, Mapping) and item.get("kind") == "complete" and isinstance(item.get("call_id"), str):
                completed.append(dict(item))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise FanoutContinuationError("recorded run tool tape is invalid") from error
    parallel_root = root_resolved / "parallel"
    if parallel_root.is_symlink() or not parallel_root.is_dir():
        raise FanoutContinuationError("recorded run has no safe parallel artifact directory")
    selected = None
    for entry in reversed(completed):
        call_id = str(entry["call_id"])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", call_id):
            continue
        call_root = parallel_root / call_id
        if (call_root.is_dir() and not call_root.is_symlink()
                and (call_root / "result.json").is_file()
                and (call_root / "intent.json").is_file()):
            selected = (entry, call_root)
            break
    if selected is None:
        raise FanoutContinuationError("recorded run has no completed parallel result")
    entry, call_root = selected
    result = _read_json(call_root / "result.json")
    intent = _read_json(call_root / "intent.json")
    request = intent.get("request")
    if not isinstance(request, Mapping):
        raise FanoutContinuationError("recorded parallel intent has no request")
    workspace = root_resolved / "parent"
    if not workspace.is_dir() or workspace.is_symlink():
        workspace = root_resolved / "workspace"
    if not workspace.is_dir() or workspace.is_symlink():
        raise FanoutContinuationError("recorded run has no parent workspace snapshot")
    turn = next(index for index, item in enumerate(completed) if item is entry)
    disposition = disposition_from_records(str(entry["call_id"]), request, result, turn_index=turn)
    disposition["lifecycle"] = {"fanout_turn_index": turn, "current_turn_index": turn}
    checks_dir = call_root / "checks" / "final"
    if checks_dir.is_dir() and not checks_dir.is_symlink():
        for path in sorted(checks_dir.glob("*.txt")):
            if path.is_symlink() or path.stat().st_size > MAX_CAPTURE_FILE_BYTES:
                continue
            try:
                failure = _failure_from_text(path.read_text(encoding="utf-8"), workspace,
                                             _unit_rows(disposition))
            except (OSError, UnicodeError):
                failure = None
            if failure is not None:
                disposition["failure"] = failure
                break
    return disposition, workspace


def render_recorded_run(run_dir: str | Path, *, budget: int = MAX_SECTION_BYTES) -> dict[str, object]:
    disposition, workspace = load_recorded_run(run_dir)
    section, pin, events = render(disposition, workspace, budget)
    return {
        "section": section,
        "pin": pin,
        "section_bytes": _bytes(section),
        "pin_bytes": _bytes(pin),
        "journal_events": events,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render typed fan-out continuation examples from a recorded run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="write section.txt and pin.txt into this directory")
    args = parser.parse_args(argv)
    try:
        output = render_recorded_run(args.run_dir)
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "section.txt").write_text(str(output["section"]) + "\n", encoding="utf-8")
            (args.output_dir / "pin.txt").write_text(str(output["pin"]) + ("\n" if output["pin"] else ""), encoding="utf-8")
            (args.output_dir / "sizes.json").write_text(
                json.dumps({key: output[key] for key in ("section_bytes", "pin_bytes", "journal_events")},
                           sort_keys=True, indent=2) + "\n", encoding="utf-8")
        else:
            print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    except (OSError, FanoutContinuationError) as error:
        print(f"fanout continuation render failed: {error}", file=sys.stderr)
        return 2
    return 0

"""One place that says WHY something was killed, refused, dropped, degraded or timed out.

Logging only: nothing here touches request bytes, hashed records or control flow.  Every entry
point is total (never raises) and works with no journal.  Each (category, cause_key) pair is
reported ONCE per run (journal event, stderr line, problems.jsonl row); repeats only count.
Process-global on purpose: kill/probe/sandbox layers have no journal handle.
Request-path callers pass journal=False: that writes stderr + problems.jsonl only and never adds a
journal event (journal positions are identity-bearing there).
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
CATEGORIES = ("kill", "refusal", "fallback", "degrade", "eviction", "timeout", "invariant")
LEVEL_ENV = "STATEFRAME_LOG_LEVEL"
PROBLEMS_ENV = "STATEFRAME_PROBLEMS_FILE"
DEFAULT_LEVEL = "WARN"
MAX_DETAIL = 5          # distinct `dropped` samples kept per cause in the summary
MAX_FIELD_CHARS = 300
MAX_LINE_CHARS = 400

_WS = re.compile(r"\s+")
_LOCK = threading.RLock()
_STATE: dict = {}


def _fresh() -> dict:
    return {"level": None, "invalid_level": None, "journal": None, "path": None, "stream": None,
            "causes": {}, "self_errors": 0, "summarized": False, "configured": False}


_STATE.update(_fresh())


def _one_line(value: object, limit: int = MAX_FIELD_CHARS) -> str:
    text = _WS.sub(" ", str(value)).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _scalar(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _one_line(value)


def _parse_level(raw: object) -> tuple[str, bool]:
    text = str(raw).strip().upper() if raw is not None else ""
    if not text:
        return DEFAULT_LEVEL, False
    if text == "WARNING":
        text = "WARN"
    return (text, False) if text in LEVELS else (DEFAULT_LEVEL, True)


def _threshold() -> int:
    if _STATE["level"] is None:
        _apply_level(os.environ.get(LEVEL_ENV))
    return LEVELS[_STATE["level"]]


def _apply_level(raw: object) -> None:
    level, invalid = _parse_level(raw)
    _STATE["level"] = level
    if invalid:
        _STATE["invalid_level"] = _one_line(raw, 40)


def configure(*, level: str | None = None, journal: object | None = None,
              problems_path: str | os.PathLike | None = None, stream=None) -> None:
    """Idempotent. Unset arguments fall back to STATEFRAME_LOG_LEVEL / STATEFRAME_PROBLEMS_FILE."""
    try:
        with _LOCK:
            _apply_level(level if level is not None else os.environ.get(LEVEL_ENV))
            path = problems_path if problems_path is not None else os.environ.get(PROBLEMS_ENV)
            if path:
                _STATE["path"] = str(path)
            if journal is not None:
                _STATE["journal"] = journal
            if stream is not None:
                _STATE["stream"] = stream
            first = not _STATE["configured"]
            _STATE["configured"] = True
            invalid = _STATE["invalid_level"]
        if first and invalid is not None:
            emit_reason("invariant", "invalid_log_level", "kernel_reason_log",
                        f"invalid {LEVEL_ENV} {invalid!r}", f"using {DEFAULT_LEVEL}")
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        _bump_self_error()


def bind_journal(journal: object | None) -> None:
    """Latest journal wins. The module keeps a plain reference only until the run ends."""
    try:
        with _LOCK:
            _STATE["journal"] = journal
    except BaseException:
        pass


def _bump_self_error() -> None:
    try:
        _STATE["self_errors"] += 1
    except BaseException:
        pass


def _journal_write(journal: object | None, kind: str, payload: dict) -> None:
    """Journal handles expose `emit` (continuation) or `write` (plain loop)."""
    if journal is None:
        return
    writer = getattr(journal, "emit", None) or getattr(journal, "write", None)
    if writer is None:
        _bump_self_error()
        return
    try:
        writer(kind, **payload)
    except Exception:
        # replaying / notifying / closed journals raise by design; logging must not care.
        _bump_self_error()


def _safe_keys(fields: dict) -> dict:
    try:
        from kernel_continuation_journal import journal_safe_fields
        return journal_safe_fields(fields)
    except Exception:
        return {(f"receipt_{k}" if k in {"schema", "task_id", "kind", "seq", "sequence", "ts"} else k): v
                for k, v in fields.items()}


def _line(level: str, component: str, what: str, why: str, fields: dict) -> str:
    extra = []
    if fields.get("measured") is not None or fields.get("limit") is not None:
        extra.append(f"measured {_one_line(fields.get('measured'))} vs limit {_one_line(fields.get('limit'))}")
    if fields.get("dropped") is not None:
        extra.append(f"dropped {_one_line(fields['dropped'])}")
    why_text = "; ".join([_one_line(why) or "unspecified", *extra])
    text = f"[stateframe] {level} {_one_line(component, 60)}: {_one_line(what)} ({why_text})"
    return text if len(text) <= MAX_LINE_CHARS else text[: MAX_LINE_CHARS - 4] + "...)"


def _append_row(path: str | None, row: dict) -> None:
    if not path:
        return
    data = (json.dumps(row, sort_keys=True, default=str) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def emit_reason(category: str, cause_key: str, component: str, what: str, why: str, *,
                level: str = "WARN", journal: object | None = None, **fields: object) -> bool:
    """True only for the first occurrence of (category, cause_key) this run. Never raises."""
    try:
        return _emit(category, cause_key, component, what, why, level, journal, fields)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        _bump_self_error()
        return False


def _emit(category, cause_key, component, what, why, level, journal, fields) -> bool:
    level, _ = _parse_level(level)
    key = (str(category), str(cause_key))
    clean = {str(k): _scalar(v) for k, v in dict(fields).items()}
    with _LOCK:
        threshold = _threshold()
        entry = _STATE["causes"].get(key)
        if entry is not None:
            entry["count"] += 1
            dropped = clean.get("dropped")
            if dropped is not None and dropped not in entry["samples"] and len(entry["samples"]) < MAX_DETAIL:
                entry["samples"].append(dropped)
            return False
        samples = [clean["dropped"]] if clean.get("dropped") is not None else []
        _STATE["causes"][key] = {
            "count": 1, "level": level, "component": _one_line(component, 60),
            "what": _one_line(what), "why": _one_line(why), "first": clean, "samples": samples}
        bound = _STATE["journal"]
        path = _STATE["path"]
        stream = _STATE["stream"]
    if LEVELS[level] < threshold:
        return True
    row = {"ts": round(time.time(), 3), "level": level, "category": key[0], "cause_key": key[1],
           "component": _one_line(component, 60), "what": _one_line(what), "why": _one_line(why),
           "fields": clean}
    try:
        print(_line(level, component, what, why, clean), file=stream or sys.stderr, flush=True)
    except Exception:
        _bump_self_error()
    try:
        _append_row(path, row)
    except Exception:
        _bump_self_error()
    # journal=False is the request-path sentinel: stderr + problems file only, never a journal event,
    # so identity-bearing journal positions cannot shift.
    _journal_write(None if journal is False else (journal if journal is not None else bound), "reason_logged", {
        "producer": "kernel_reason_log", "dispatch": False, "category": key[0], "cause_key": key[1],
        "component": row["component"], "level": level, "what": row["what"], "why": row["why"],
        **_safe_keys(clean)})
    return True


def emit_kill(component: str, *, target: str, why: str, pid: int | None = None, unit: str | None = None,
              measured: object = None, limit: object = None, elapsed_s: float | None = None,
              signal_name: str = "SIGKILL", journal: object | None = None, level: str = "ERROR",
              cause_key: str | None = None) -> bool:
    """cause_key defaults to component:why (never the pid: one line per cause per run)."""
    try:
        what = f"killed {target}" + (f" pid {pid}" if pid is not None else "") + (
            f" unit {unit!r}" if unit else "")
        why_text = _one_line(why)
        if elapsed_s is not None:
            why_text += f", elapsed {elapsed_s:.1f}s"
        return emit_reason("kill", cause_key or f"{component}:{why}", component, what, why_text,
                           level=level, journal=journal, target=target, pid=pid, unit=unit,
                           measured=measured, limit=limit, elapsed_s=elapsed_s, signal=signal_name)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        _bump_self_error()
        return False


def emit_refusal(component: str, *, what: str, why: str, limit: object = None, measured: object = None,
                 dropped: object = None, journal: object | None = None, level: str = "WARN",
                 cause_key: str | None = None) -> bool:
    try:
        return emit_reason("refusal", cause_key or f"{component}:{why}", component, what, why,
                           level=level, journal=journal, limit=limit, measured=measured, dropped=dropped)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        _bump_self_error()
        return False


def emit_timeout(component: str, *, what: str, why: str, deadline_s: float | None = None,
                 elapsed_s: float | None = None, journal: object | None = None, level: str = "WARN",
                 cause_key: str | None = None) -> bool:
    try:
        why_text = _one_line(why)
        if deadline_s is not None:
            why_text += f", deadline {deadline_s:g}s"
        if elapsed_s is not None:
            why_text += f", elapsed {elapsed_s:.1f}s"
        return emit_reason("timeout", cause_key or f"{component}:{why}", component, what, why_text,
                           level=level, journal=journal, deadline_s=deadline_s, elapsed_s=elapsed_s)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        _bump_self_error()
        return False


def run_summary() -> dict:
    """All recorded causes regardless of level (flush_summary filters by level).
    {"total": n, "by_cause": {"<category>:<cause_key>": {"count","level","component","what","why","first"}}}"""
    try:
        with _LOCK:
            by_cause = {
                f"{category}:{cause}": {
                    "count": e["count"], "level": e["level"], "component": e["component"],
                    "what": e["what"], "why": e["why"], "first": dict(e["first"]),
                    **({"dropped_samples": list(e["samples"])} if len(e["samples"]) > 1 else {})}
                for (category, cause), e in _STATE["causes"].items()}
            result = {"total": sum(e["count"] for e in _STATE["causes"].values()), "by_cause": by_cause}
            if _STATE["self_errors"]:
                result["self_errors"] = _STATE["self_errors"]
            return result
    except BaseException:
        return {"total": 0, "by_cause": {}}


def flush_summary(journal: object | None = None) -> dict | None:
    """ONE `reason_log_summary` event, one problems.jsonl summary row, one stderr line. Idempotent."""
    try:
        with _LOCK:
            if _STATE["summarized"]:
                return None
            summary = run_summary()
            # Only causes at/above the configured level were ever printed; keep the summary consistent.
            threshold = _threshold()
            summary["by_cause"] = {k: v for k, v in summary["by_cause"].items()
                                   if LEVELS.get(v["level"], 0) >= threshold}
            summary["total"] = sum(v["count"] for v in summary["by_cause"].values())
            if not summary["by_cause"] and not summary.get("self_errors"):
                return None
            _STATE["summarized"] = True
            bound, path, stream = _STATE["journal"], _STATE["path"], _STATE["stream"]
        errors = sum(1 for c in summary["by_cause"].values() if c["level"] == "ERROR")
        warns = sum(1 for c in summary["by_cause"].values() if c["level"] == "WARN")
        try:
            _append_row(path, {"summary": True, "ts": round(time.time(), 3), "pid": os.getpid(),
                               "by_cause": summary["by_cause"]})
        except Exception:
            _bump_self_error()
        _journal_write(journal if journal is not None else bound, "reason_log_summary", {
            "producer": "kernel_reason_log", "dispatch": False, "total": summary["total"],
            "errors": errors, "warnings": warns,
            "by_cause": {k: {"count": v["count"], "level": v["level"]} for k, v in summary["by_cause"].items()}})
        if errors or warns:
            try:
                print(f"[stateframe] problems: {errors} ERROR, {warns} WARN causes ({summary['total']} events)",
                      file=stream or sys.stderr, flush=True)
            except Exception:
                _bump_self_error()
        return summary
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        _bump_self_error()
        return None


def reset_for_tests() -> None:
    with _LOCK:
        _STATE.clear()
        _STATE.update(_fresh())


def problems_from_file(path: str | os.PathLike, *, log_level: str = "warn") -> dict:
    """Runner-side merge of problems.jsonl (first-occurrence rows + per-process summary rows)."""
    causes: dict[str, dict] = {}
    summed: dict[str, int] = {}
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    for raw in lines:
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("summary"):
            for key, info in (row.get("by_cause") or {}).items():
                if isinstance(info, dict):
                    summed[key] = summed.get(key, 0) + int(info.get("count") or 0)
                    causes.setdefault(key, {"level": info.get("level"), "component": info.get("component"),
                                            "count": 0, "what": info.get("what"), "why": info.get("why"),
                                            "first": info.get("first") or {}})
            continue
        key = f"{row.get('category')}:{row.get('cause_key')}"
        entry = causes.setdefault(key, {"level": row.get("level"), "component": row.get("component"),
                                        "count": 0, "what": row.get("what"), "why": row.get("why"),
                                        "first": row.get("fields") or {}})
        entry["count"] += 1
    for key, entry in causes.items():
        entry["count"] = max(entry["count"], summed.get(key, 0))
    return {"errors": sum(1 for e in causes.values() if e["level"] == "ERROR"),
            "warnings": sum(1 for e in causes.values() if e["level"] == "WARN"),
            "by_cause": causes, "log_level": log_level}

"""Append-only, content-addressed records of every fixer model call.

The ledger deliberately stores prompt bytes separately from call rows.  A prompt
blob is written once under its SHA-256, while each attempt appends one complete
row to a run JSONL file.  Run files are the rotation unit, so prompt capture is
append-only during normal operation and old runs can be evicted as a whole.

The module is dependency-free and fail-soft at its public capture boundary:
ledger I/O is observability, never a reason to fail a model run.

CLI::

    python3 -m prompt_ledger last 20
    python3 -m prompt_ledger show <call-id>
    python3 -m prompt_ledger grep MEASURED:
    python3 -m prompt_ledger stats

Set ``PROMPT_LEDGER_DIR`` to select a store and
``PROMPT_LEDGER_MAX_BYTES`` to bound it.  The fixer config can select both
values for a run through ``gym_config.PromptLedgerCfg``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence


SCHEMA_VERSION = 1
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_PROMPT_SUFFIX = ".bin"
_RUN_SUFFIX = ".jsonl"
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def _default_root() -> Path:
    configured = os.environ.get("PROMPT_LEDGER_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parent / "results" / "prompt_ledger"


def _max_bytes_from_env() -> int:
    raw = os.environ.get("PROMPT_LEDGER_MAX_BYTES")
    if raw is None or not raw.strip():
        return DEFAULT_MAX_BYTES
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_BYTES


def new_run_id() -> str:
    """Return a collision-resistant identifier suitable for a run row."""

    return f"run-{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex[:12]}"


def task_id(task_text: str | None) -> str:
    """Return a stable, non-sensitive identifier for task text."""

    return hashlib.sha256((task_text or "").encode("utf-8")).hexdigest()[:16]


def _safe_json(value: object) -> object:
    """Convert response/request values to JSON without losing normal shapes."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_json(item) for item in value]
    return repr(value)


def _response_parts(response: object) -> tuple[str | None, str | None, str, object, object]:
    """Extract content, reasoning, finish, usage, and logprobs safely."""

    if not isinstance(response, Mapping):
        return None, None, "", None, None
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, Sequence) and choices else None
    if not isinstance(choice, Mapping):
        return None, None, "", response.get("usage"), None
    message = choice.get("message")
    if not isinstance(message, Mapping):
        message = {}
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    return (
        content if isinstance(content, str) else None,
        reasoning if isinstance(reasoning, str) else None,
        str(choice.get("finish_reason") or ""),
        response.get("usage"),
        choice.get("logprobs"),
    )


def _answer_text(content: str | None) -> str:
    if not content:
        return ""
    without_think = _THINK_RE.sub("", content)
    lower = without_think.lower()
    open_tag = lower.rfind("<think>")
    if open_tag >= 0 and "</think>" not in lower[open_tag:]:
        without_think = without_think[:open_tag]
    return without_think.strip()


def parse_answer(content: str | None) -> object:
    """Parse JSON answers while retaining plain-text model answers."""

    text = _answer_text(content)
    if not text:
        return None
    candidate = text
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text


def classify_call(label: str | None, prompt: str) -> tuple[str, str]:
    """Classify a call centrally from its label and rendered prompt.

    The fixer has a few historical unlabeled calls, so prompt wording is a
    fallback.  ``classification``/``unknown`` are honest values for calls that
    are not one of the operational menu kinds.
    """

    label_text = (label or "").lower()
    prompt_text = (prompt or "").lower()
    haystack = f"{label_text}\n{prompt_text}"
    if "confirm" in label_text or "can the observed problem still happen" in haystack:
        return "confirm", "confirm"
    if "block_hole" in label_text or "block hole" in haystack:
        return "block", "block"
    if "hole_fill" in label_text or "fill" in label_text:
        return "fill", "fill"
    if "line" in label_text or "is the bug on this line" in haystack:
        return "line", "line"
    if "function" in label_text or "candidate function" in haystack:
        return "function", "function"
    if "file" in label_text or "candidate file" in haystack:
        return "file", "file"
    if "mutation" in label_text or "correction fixes" in haystack:
        return "mutation", "mutation"
    if "staged_reasoning" in label_text or "visible mechanism in this change" in haystack:
        return "mutation", "mutation"
    if "what kind of bug" in haystack:
        return "classification", "classification"
    return "model_call", "unknown"


def _messages_prompt_bytes(body: Mapping[str, object]) -> tuple[bytes, str]:
    messages = body.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes, bytearray)):
        if len(messages) == 1 and isinstance(messages[0], Mapping):
            content = messages[0].get("content")
            if isinstance(content, str):
                return content.encode("utf-8"), "utf-8-text"
        serialized = json.dumps(
            _safe_json(messages), ensure_ascii=False, separators=(",", ":")
        )
        return serialized.encode("utf-8"), "messages-json"
    return b"", "utf-8-text"


def _model_params(body: Mapping[str, object]) -> dict[str, object]:
    template = body.get("chat_template_kwargs")
    thinking = template.get("enable_thinking") if isinstance(template, Mapping) else None
    return {
        "thinking": thinking,
        "max_tokens": body.get("max_tokens"),
        "max_completion_tokens": body.get("max_completion_tokens"),
        "answer_tokens": body.get("answer_tokens"),
        "temperature": body.get("temperature"),
        "logprobs": body.get("logprobs"),
        "top_logprobs": body.get("top_logprobs"),
    }


def _outcome(response: object, error: BaseException | None) -> dict[str, object]:
    content, _reasoning, finish_reason, _usage, _logprobs = _response_parts(response)
    if error is not None:
        return {
            "parsed_answer": None,
            "accepted": False,
            "reason": f"transport_error:{type(error).__name__}",
        }
    if finish_reason == "length":
        return {
            "parsed_answer": parse_answer(content),
            "accepted": False,
            "reason": "finish_reason:length",
        }
    parsed = parse_answer(content)
    if parsed is None:
        return {"parsed_answer": None, "accepted": False, "reason": "empty_response"}
    reason = "parsed_json" if not isinstance(parsed, str) else "content_received"
    return {"parsed_answer": _safe_json(parsed), "accepted": True, "reason": reason}


class PromptLedger:
    """A file-backed prompt ledger with per-instance synchronization."""

    def __init__(self, root: str | os.PathLike[str] | None = None, max_bytes: int | None = None):
        selected_root = Path(root).expanduser() if root is not None else _default_root()
        if not selected_root.is_absolute():
            selected_root = Path(__file__).resolve().parent / selected_root
        self.root = selected_root
        self.max_bytes = _max_bytes_from_env() if max_bytes is None else max(0, int(max_bytes))
        self._lock = threading.RLock()

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def _run_path(self, run_id: str) -> Path:
        run_hash = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        return self.runs_dir / f"{run_hash}{_RUN_SUFFIX}"

    def _prompt_path(self, prompt_hash: str) -> Path:
        return self.prompts_dir / f"{prompt_hash}{_PROMPT_SUFFIX}"

    def _append_row(self, path: Path, row: Mapping[str, object]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_safe_json(row), ensure_ascii=False, sort_keys=True) + "\n")

    def _write_prompt(self, path: Path, prompt_bytes: bytes) -> None:
        try:
            with path.open("xb") as handle:
                handle.write(prompt_bytes)
        except FileExistsError:
            # A repeated prompt is the normal dedupe path.  Do not rewrite it.
            return

    def _rows_from_path(self, path: Path) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
        except OSError:
            return []
        return rows

    def _run_paths(self) -> list[Path]:
        try:
            return sorted(self.runs_dir.glob(f"*{_RUN_SUFFIX}"))
        except OSError:
            return []

    def _all_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for path in self._run_paths():
            rows.extend(self._rows_from_path(path))
        rows.sort(key=lambda row: (int(row.get("started_at_ns", 0) or 0), str(row.get("call_id", ""))))
        return rows

    def _store_size(self) -> int:
        total = 0
        try:
            for path in self.root.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
        except OSError:
            return total
        return total

    def _referenced_prompts(self) -> set[str]:
        return {
            str(row.get("prompt_hash"))
            for row in self._all_rows()
            if row.get("prompt_hash")
        }

    def _remove_unreferenced_prompts(self) -> None:
        referenced = self._referenced_prompts()
        try:
            for path in self.prompts_dir.glob(f"*{_PROMPT_SUFFIX}"):
                if path.stem not in referenced:
                    path.unlink(missing_ok=True)
        except OSError:
            return

    def _rotate(self, current_run_path: Path | None = None) -> None:
        if self.max_bytes <= 0:
            return
        while self._store_size() > self.max_bytes:
            candidates: list[tuple[int, Path]] = []
            for path in self._run_paths():
                if current_run_path is not None and path == current_run_path:
                    continue
                rows = self._rows_from_path(path)
                first = rows[0] if rows else {}
                started = int(first.get("started_at_ns", 0) or 0)
                candidates.append((started or int(path.stat().st_mtime_ns), path))
            if not candidates:
                # Keep the current run intact even if one run is larger than the cap.
                return
            _started, oldest = min(candidates, key=lambda item: (item[0], str(item[1])))
            try:
                oldest.unlink()
            except OSError:
                return
            self._remove_unreferenced_prompts()

    def record(
        self,
        *,
        body: Mapping[str, object],
        response: object,
        wall_time_seconds: float,
        label: str | None = None,
        run_id: str | None = None,
        task_id_value: str | None = None,
        error: BaseException | None = None,
    ) -> dict[str, object]:
        """Append one model-call attempt and return the row representation."""

        if not isinstance(body, Mapping):
            body = {}
        prompt_bytes, prompt_format = _messages_prompt_bytes(body)
        prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
        prompt_text = prompt_bytes.decode("utf-8", errors="replace")
        stage, menu_kind = classify_call(label, prompt_text)
        resolved_run = str(run_id or "run-unbound")
        resolved_task = str(task_id_value or "task-unbound")
        call_id = f"call-{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        started_at_ns = time.time_ns()
        content, reasoning, finish_reason, usage, logprobs = _response_parts(response)
        request_params = {
            str(key): _safe_json(value)
            for key, value in body.items()
            if key != "messages"
        }
        row: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "call_id": call_id,
            "run_id": resolved_run,
            "task_id": resolved_task,
            "label": label,
            "stage": stage,
            "menu_kind": menu_kind,
            "started_at_ns": started_at_ns,
            "recorded_at": time.time(),
            "wall_time_seconds": max(0.0, float(wall_time_seconds or 0.0)),
            "prompt_hash": prompt_hash,
            "prompt": {
                "sha256": prompt_hash,
                "byte_length": len(prompt_bytes),
                "encoding": "utf-8",
                "format": prompt_format,
                "path": f"prompts/{prompt_hash}{_PROMPT_SUFFIX}",
            },
            "model_params": _model_params(body),
            "request_params": request_params,
            "response": _safe_json(response),
            "response_fields": {
                "content": content,
                "reasoning_content": reasoning,
                "finish_reason": finish_reason,
                "usage": _safe_json(usage),
                "logprobs": _safe_json(logprobs),
            },
            "outcome": _outcome(response, error),
        }
        if error is not None:
            row["transport_error"] = type(error).__name__

        run_path = self._run_path(resolved_run)
        with self._lock:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                self.prompts_dir.mkdir(parents=True, exist_ok=True)
                self.runs_dir.mkdir(parents=True, exist_ok=True)
                self._rotate(current_run_path=run_path)
                self._write_prompt(self._prompt_path(prompt_hash), prompt_bytes)
                self._append_row(run_path, row)
            except Exception as exc:  # noqa: BLE001 - observability must fail soft
                row["ledger_error"] = type(exc).__name__
                try:
                    self.runs_dir.mkdir(parents=True, exist_ok=True)
                    self._append_row(run_path, row)
                except Exception:
                    # No writable sink remains; the caller still receives its model result.
                    pass
            try:
                self._rotate(current_run_path=run_path)
            except Exception:
                pass
        return row

    def rows(self) -> list[dict[str, object]]:
        with self._lock:
            return self._all_rows()

    def prompt_bytes(self, row: Mapping[str, object]) -> bytes:
        prompt = row.get("prompt")
        prompt_hash = row.get("prompt_hash")
        if isinstance(prompt, Mapping):
            prompt_hash = prompt.get("sha256", prompt_hash)
        if not isinstance(prompt_hash, str):
            return b""
        try:
            return self._prompt_path(prompt_hash).read_bytes()
        except OSError:
            return b""

    def grep(self, text: str) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        needle = str(text)
        for row in self.rows():
            prompt = self.prompt_bytes(row).decode("utf-8", errors="replace")
            response = json.dumps(row.get("response"), ensure_ascii=False, sort_keys=True)
            locations: list[str] = []
            if needle in prompt:
                locations.append("prompt")
            if needle in response:
                locations.append("response")
            if locations:
                matches.append({"row": row, "locations": locations})
        return matches

    def stats(self) -> dict[str, object]:
        rows = self.rows()
        stage_counts = Counter(str(row.get("stage", "unknown")) for row in rows)
        finish_counts = Counter(
            str((row.get("response_fields") or {}).get("finish_reason", ""))
            if isinstance(row.get("response_fields"), Mapping)
            else ""
            for row in rows
        )
        unique_prompts = len({str(row.get("prompt_hash")) for row in rows if row.get("prompt_hash")})
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        for row in rows:
            fields = row.get("response_fields")
            usage = fields.get("usage") if isinstance(fields, Mapping) else None
            if not isinstance(usage, Mapping):
                continue
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            total_tokens += int(usage.get("total_tokens", 0) or 0)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        calls = len(rows)
        return {
            "calls": calls,
            "stages": dict(sorted(stage_counts.items())),
            "token_totals": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "finish_reasons": dict(sorted(finish_counts.items())),
            "unique_prompts": unique_prompts,
            "prompt_dedupe_rate": round((calls - unique_prompts) / calls, 6) if calls else 0.0,
            "store_bytes": self._store_size(),
            "max_store_bytes": self.max_bytes,
        }


_ACTIVE_LEDGER: PromptLedger | None = None
_ACTIVE_ENABLED = True
_CONTEXT = threading.local()


def configure(
    *,
    enabled: bool = True,
    root: str | os.PathLike[str] | None = None,
    max_bytes: int | None = None,
) -> PromptLedger:
    """Configure the process-local capture sink and return it."""

    global _ACTIVE_LEDGER, _ACTIVE_ENABLED
    _ACTIVE_ENABLED = bool(enabled)
    _ACTIVE_LEDGER = PromptLedger(root=root, max_bytes=max_bytes)
    return _ACTIVE_LEDGER


def set_context(run_id: str | None, task_id_value: str | None) -> None:
    _CONTEXT.run_id = str(run_id or "run-unbound")
    _CONTEXT.task_id = str(task_id_value or "task-unbound")


def context() -> tuple[str, str]:
    return (
        str(getattr(_CONTEXT, "run_id", "run-unbound")),
        str(getattr(_CONTEXT, "task_id", "task-unbound")),
    )


def active_ledger() -> PromptLedger:
    global _ACTIVE_LEDGER
    if _ACTIVE_LEDGER is None:
        _ACTIVE_LEDGER = PromptLedger()
    return _ACTIVE_LEDGER


def capture_call(
    *,
    body: Mapping[str, object],
    response: object,
    wall_time_seconds: float,
    label: str | None = None,
    error: BaseException | None = None,
) -> dict[str, object] | None:
    """Capture one call; all unexpected ledger failures are swallowed."""

    if not _ACTIVE_ENABLED:
        return None
    try:
        run_id, task_value = context()
        return active_ledger().record(
            body=body,
            response=response,
            wall_time_seconds=wall_time_seconds,
            label=label,
            run_id=run_id,
            task_id_value=task_value,
            error=error,
        )
    except Exception:
        return None


def _row_summary(row: Mapping[str, object]) -> dict[str, object]:
    fields = row.get("response_fields")
    outcome = row.get("outcome")
    return {
        "call_id": row.get("call_id"),
        "run_id": row.get("run_id"),
        "stage": row.get("stage"),
        "menu_kind": row.get("menu_kind"),
        "answer": outcome.get("parsed_answer") if isinstance(outcome, Mapping) else None,
        "accepted": outcome.get("accepted") if isinstance(outcome, Mapping) else False,
        "finish_reason": fields.get("finish_reason") if isinstance(fields, Mapping) else "",
    }


def _ledger_from_args(args: argparse.Namespace) -> PromptLedger:
    if args.directory:
        return PromptLedger(args.directory, max_bytes=args.max_bytes)
    if args.max_bytes is not None:
        return PromptLedger(max_bytes=args.max_bytes)
    return active_ledger()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect fixer prompt ledger")
    parser.add_argument("--dir", dest="directory", help="ledger directory")
    parser.add_argument("--max-bytes", type=int, default=None, help="rotation cap")
    commands = parser.add_subparsers(dest="command", required=True)
    last = commands.add_parser("last", help="show recent calls")
    last.add_argument("count", type=int)
    show = commands.add_parser("show", help="show one full call")
    show.add_argument("call_id")
    grep = commands.add_parser("grep", help="find text in prompts/responses")
    grep.add_argument("text")
    commands.add_parser("stats", help="summarize calls and tokens")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    ledger = _ledger_from_args(args)
    if args.command == "last":
        limit = max(0, args.count)
        rows = [] if limit == 0 else ledger.rows()[-limit:]
        for row in reversed(rows):
            print(json.dumps(_row_summary(row), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "show":
        row = next((item for item in ledger.rows() if item.get("call_id") == args.call_id), None)
        if row is None:
            print(f"call not found: {args.call_id}", file=sys.stderr)
            return 1
        prompt = ledger.prompt_bytes(row).decode("utf-8", errors="replace")
        print(json.dumps({"row": row, "prompt": prompt}, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.command == "grep":
        for item in ledger.grep(args.text):
            row = item["row"]
            print(json.dumps({**_row_summary(row), "locations": item["locations"]},
                             ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "stats":
        print(json.dumps(ledger.stats(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke/tests
    raise SystemExit(main())

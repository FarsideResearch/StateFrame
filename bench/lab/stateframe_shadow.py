"""Maintainer-only archived-transcript shadow projection driver."""
import hashlib, json, pathlib, re, sys
from typing import Iterator
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src/stateframe/engine"))
from probes import stateframe_shadow as _shipped_shadow
from probes.stateframe_shadow import (SCHEMA, Turn, _sha256_bytes, _canon,
    _source_rel_path, _event_id, _kernel_event_turn_index, _refresh_derived,
    initial_frame, fold_turn)
# Preserve the original output-root binding inside the verbatim main() body.
__file__ = _shipped_shadow.__file__
def load_transcript(path: pathlib.Path) -> dict:
    raw = path.read_bytes()
    d = json.loads(raw)
    if not isinstance(d, dict) or not isinstance(d.get("messages"), list):
        raise ValueError(f"not a transcript dict with messages: {path}")
    d["_source_sha256"] = _sha256_bytes(raw)
    d["_source_path"] = str(path)
    rel_path = _source_rel_path(d)
    if rel_path is not None:
        d["_rel_path"] = rel_path
    return d


def _is_trim_marker(m: dict) -> bool:
    c = str(m.get("content", ""))
    return m.get("role") == "user" and c.startswith("[kernel] (") and \
        "trimmed to fit context" in c


def reconstruct_messages(
    transcript_path: pathlib.Path,
    messages: list[dict],
    *,
    max_step: int | None = None,
) -> list[dict]:
    """E316 reader half: splice back exchanges the live loop trimmed (persisted to
    results/react_events<TAG>/<fn>.jsonl by react_repair). Trim markers and
    previously-re-trimmed markers are dropped; order is chronological (trims are
    appended in trim order, each dump preserving message order). ``max_step``
    admits only explicitly step-attributed trim events at or before the cutoff;
    un-attributed events are excluded rather than leaked into an earlier prefix.

    R2 repair: only splice when `messages` itself carries evidence that trimming
    actually happened (a trim marker present). Without that, the events file for
    this filename/tag pairing has nothing to do with THIS `messages` view — the
    previous unconditional splice let evicted content from far later in the run
    (or, in the causal-cutoff regression, from an unrelated synthetic future)
    leak into an earlier/shorter view that never experienced any trim at all.
    """
    if not any(_is_trim_marker(m) for m in messages):
        return messages
    events_dir = transcript_path.parent.parent / \
        transcript_path.parent.name.replace("react_transcripts", "react_events")
    ev_path = events_dir / (transcript_path.stem + ".jsonl")
    if not ev_path.exists():
        return messages
    trimmed: list[dict] = []
    for line in ev_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)  # malformed events fail closed, same as snapshots
        if max_step is not None:
            event_step = rec.get("step")
            if not isinstance(event_step, int) or event_step > max_step:
                continue
        m = rec.get("msg", {})
        if rec.get("kind") == "trimmed" and not _is_trim_marker(m):
            trimmed.append(m)
    head, tail = messages[:2], [m for m in messages[2:] if not _is_trim_marker(m)]
    return head + trimmed + tail


def iter_turns(messages: list[dict]) -> Iterator[Turn]:
    """One turn = one assistant generation + the tool/user observations that follow,
    ending before the next assistant message. System messages and trim markers are
    context, not turns."""
    idx = 0
    cur: Turn | None = None
    for position, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            if cur is not None:
                yield cur
            cur = Turn(index=idx, assistant=m, position=position)
            idx += 1
        elif role in ("tool", "user") and cur is not None:
            cur.observations.append(m)
    if cur is not None:
        yield cur


def attach_kernel_events(turns: list[Turn], trace: object) -> list[Turn]:
    """Attach only causally safe kernel events to reconstructed turns.

    Explicit ``step``/``turn`` coordinates are respected. Final-only events with
    no coordinate may use an ordered assistant-content prefix as evidence; an
    event without that anchor is retained on the terminal turn with an explicit
    unavailable attribution marker and is never visible in an earlier prefix.
    """
    if not turns or not isinstance(trace, list):
        return turns
    attached = [Turn(index=turn.index, assistant=turn.assistant,
                      observations=turn.observations, position=turn.position,
                      kernel_events=list(turn.kernel_events)) for turn in turns]
    terminal = len(attached) - 1
    last_content_match = -1
    for trace_position, raw_event in enumerate(trace):
        if not isinstance(raw_event, dict):
            continue
        event_turn = _kernel_event_turn_index(raw_event)
        event = json.loads(_canon(raw_event))
        event["event_id"] = _event_id("kernel", event, trace_position)
        if event_turn is None:
            # A run-final event may still carry an exact, ordered content anchor.
            # Treat that as explicit evidence, but never guess from event order;
            # an event with no anchor stays terminal and keeps a typed missing
            # step. This preserves the R2 cutoff for genuinely un-attributed
            # future evidence while recovering the R4 anchors the corpus has.
            detail = str(event.get("detail") or "")
            destination = None
            if detail:
                for candidate in range(last_content_match + 1, len(attached)):
                    content = str(attached[candidate].assistant.get("content") or "")
                    if content.startswith(detail):
                        destination = candidate
                        last_content_match = candidate
                        break
            if destination is None:
                destination = terminal
                event["turn"] = None
                event["step"] = None
                event["step_status"] = "unavailable"
            else:
                event["turn"] = destination
                event["step"] = destination + 1
                event["step_status"] = "attributed"
        else:
            destination = min(max(event_turn, 0), terminal)
            event["step_status"] = "attributed"
        attached[destination].kernel_events.append(event)
    return attached


def seal_frame(frame: dict, source_sha: str, turn_index: int) -> dict:
    digest = _sha256_bytes(_canon(frame).encode())
    return {"kind": "snapshot", "key": f"{source_sha}:{turn_index}",
            "turn": turn_index, "state_digest": digest, "state": frame}


def append_idempotent(out_path: pathlib.Path, records: list[dict]) -> int:
    """Append records not already present; identical existing records are skipped,
    conflicting or malformed existing lines FAIL CLOSED."""
    existing: dict[str, str] = {}
    if out_path.exists():
        for i, line in enumerate(out_path.read_text().splitlines()):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError as e:
                raise ValueError(f"malformed existing line {i} in {out_path}") from e
            key = rec.get("key") or f"__{rec.get('kind')}"
            existing[key] = _canon(rec)
    written = 0
    with out_path.open("a") as fh:
        for rec in records:
            key = rec.get("key") or f"__{rec.get('kind')}"
            enc = _canon(rec)
            if key in existing:
                if existing[key] != enc:
                    raise ValueError(f"snapshot conflict for {key} in {out_path}")
                continue
            fh.write(enc + "\n")
            fh.flush()
            import os
            os.fsync(fh.fileno())
            existing[key] = enc
            written += 1
    return written


def project_file(transcript_path: pathlib.Path, out_root: pathlib.Path) -> pathlib.Path:
    src = load_transcript(transcript_path)
    kernel = src.get("kernel") or {}
    trace_sha = _sha256_bytes(_canon(kernel).encode())
    out_dir = out_root / transcript_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (transcript_path.stem + ".jsonl")

    header: dict = {
        "kind": "header", "key": f"__header:{src['_source_sha256']}",
        "schema": SCHEMA, "source_path": src["_source_path"],
        "source_sha256": src["_source_sha256"],
        "kernel_trace_sha256": trace_sha,
        "trace_alignment": "run_final_only",
        "kernel_trace": kernel,
    }
    if src.get("_rel_path") is not None:
        header["rel_path"] = src["_rel_path"]
    records: list[dict] = [header]
    frame = initial_frame(src)
    max_step = src.get("steps")
    max_step = max_step if isinstance(max_step, int) and not isinstance(max_step, bool) else None
    turns = list(iter_turns(reconstruct_messages(
        transcript_path, src["messages"], max_step=max_step)))
    trace = kernel.get("trace") if isinstance(kernel, dict) else None
    turns = attach_kernel_events(turns, trace)
    for position, turn in enumerate(turns):
        frame = fold_turn(frame, turn)
        frame["cursor"]["terminal"] = position == len(turns) - 1
        _refresh_derived(frame)
        records.append(seal_frame(frame, src["_source_sha256"], turn.index))
    append_idempotent(out_path, records)
    return out_path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: stateframe_shadow.py <transcript.json|transcript-dir> [...]")
        return 2
    out_root = pathlib.Path(__file__).resolve().parent.parent / "results" / "stateframes_shadow"
    targets: list[pathlib.Path] = []
    for a in argv[1:]:
        p = pathlib.Path(a)
        targets.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    for t in targets:
        try:
            out = project_file(t, out_root)
            print(f"projected {t.name} -> {out}")
        except ValueError as e:
            print(f"FAILED CLOSED {t.name}: {e}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

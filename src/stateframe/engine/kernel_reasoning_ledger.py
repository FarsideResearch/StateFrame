"""request fitting: typed reasoning ledger (STATEFRAME_REASONING_LEDGER) -- supersedes the r46 reasoning anchor.

Problem (tt_output_gap / f3_r44d review 2e): F4 retains only the last 2 real exchanges and
``_pair_messages`` pops every ``reasoning_content``, so the model's own settled findings ("the fix
is X", "Y is not the issue") and the host's outcomes of older actions leave the context with no
trace.  The model then re-derives them in a fresh reasoning paragraph.  The r46 anchor kept only 3
literal read/verify markers of EVICTED pairs; this organ keeps two typed, bounded streams:

* RECEIPTS -- host facts.  One typed row per closed provider tool exchange, classified from a
  fixed set of literal result markers calibrated on real r46 f4 journals (``classify_receipt``).
* CLAIMS   -- the model's OWN earlier sentences, verbatim (clipped), extracted deterministically
  from ``reasoning_content`` by anchored regexes (``extract_claims``).  They are rendered as
  unverified candidates, never as facts, and they are INVALIDATED by later receipts: an edit that
  touches a claim's footprint stales it, a red test after a claimed fix contradicts it, a refused
  call retires the pending next step, a revision change retires OPEN/NEXT_STEP.  Stale and
  contradicted claims stay in the object for audit and never render.

Hard filter: a sentence saying done/complete/finished/finalize is never a claim (it would carry
a premature-done belief across turns).

Strict ``off|on`` switch, default OFF.  Off: nothing constructs a ledger, ``compose_messages``
gets ``ledger_block=""`` and renders byte-identically.  Deterministic, bounded (render budget
``MAX_RENDER_TOKENS`` via ``kernel_context.token_estimate``), JSON-safe snapshot for checkpoints.
Own journal events: ``reasoning_ledger_evaluated`` / ``reasoning_ledger_delivered`` (emitted by
the runtime wiring, see README).
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
from collections.abc import Mapping

from kernel_context import token_estimate


MODE_ENV = "STATEFRAME_REASONING_LEDGER"
SCHEMA = "stateframe-reasoning-ledger/1"
LEDGER_HEADER = (f"[Reasoning ledger | schema={SCHEMA} | RECEIPTS are host facts; "
                 "CLAIMS are your own earlier sentences, unverified candidates]")
MAX_RENDER_TOKENS = 400
# STATEFRAME_REASONING_LEDGER_MAX_TOKENS: the render budget as an integer env knob (default = the
# constant above; 50..2000).  Read at every render so a lab variant can trim the block without a code fork.
MAX_TOKENS_ENV = "STATEFRAME_REASONING_LEDGER_MAX_TOKENS"


def max_render_tokens(environ: Mapping[str, str] | None = None) -> int:
    source = os.environ if environ is None else environ
    raw = str(source.get(MAX_TOKENS_ENV, "") or "").strip()
    if not raw:
        return MAX_RENDER_TOKENS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{MAX_TOKENS_ENV} must be an integer") from exc
    if not 50 <= value <= 2000:
        raise ValueError(f"{MAX_TOKENS_ENV} must be within 50..2000")
    return value
EVENT_EVALUATED = "reasoning_ledger_evaluated"
EVENT_DELIVERED = "reasoning_ledger_delivered"

OUTCOMES = frozenset({
    "READ_OK", "READ_ALREADY_PRESENT", "READ_REJECTED",
    "EDIT_APPLIED", "EDIT_NO_OP", "EDIT_STAGED", "EDIT_REJECTED",
    "TESTS_RED", "TESTS_GREEN", "TESTS_REUSED",
    "GREP_HITS", "GREP_EMPTY", "GREP_ERROR",
    "PROBE_OK", "PROBE_DENIED",
    "REJECTED_CONTRACT", "REFUSED", "OTHER",
})
CLAIM_KINDS = ("CONCLUSION", "RULED_OUT", "OPEN", "NEXT_STEP")
CLAIM_STATUSES = frozenset({"candidate", "stale", "contradicted"})

# Rendered view caps per kind.  The underlying run tape is lossless; these
# values only bound the legacy ledger block and never delete an observation.
CLAIM_CAPS = {"CONCLUSION": 5, "RULED_OUT": 3, "OPEN": 2, "NEXT_STEP": 1}
# Kept per observed response (last N of each kind in that response).
PER_RESPONSE = {"CONCLUSION": 2, "RULED_OUT": 2, "OPEN": 1, "NEXT_STEP": 1}
MAX_RECEIPTS = 12              # legacy render preference, not a storage limit
MAX_AUDIT_CLAIMS = 48          # legacy render preference, not a storage limit
MAX_SEEN_IDS = 256
MAX_DETAIL = 80
MAX_CLAIM_TEXT = 150
MAX_TARGET = 60
MAX_FOOTPRINT = 8
OLDER_TAG = " (older rev)"

_EDIT_OUTCOMES = frozenset({"EDIT_APPLIED", "EDIT_STAGED", "EDIT_NO_OP", "EDIT_REJECTED"})
_TEST_OUTCOMES = frozenset({"TESTS_RED", "TESTS_GREEN", "TESTS_REUSED"})
_REFUSAL_OUTCOMES = frozenset({"REJECTED_CONTRACT", "REFUSED", "PROBE_DENIED", "READ_REJECTED",
                               "GREP_ERROR"})


# ---------------------------------------------------------------- switch

def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict opt-in switch value."""
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    if value == "on" and environ is None:
        # Single owner of frame reasoning carry (same style as kernel_decision_continuity's
        # F4 + DECISION_CONTINUITY=on rejection): the r47 L3a rationale swap and the L3b F4
        # echo would carry the same reasoning a second time, unbudgeted by this ledger.
        for module_name, env_name in (("kernel_handoff_conclusion", "STATEFRAME_HANDOFF_CONCLUSION"),
                                      ("kernel_f4_reasoning_echo", "STATEFRAME_F4_REASONING_ECHO")):
            try:
                other = __import__(module_name).enabled()
            except ImportError:
                other = False
            if other:
                raise ValueError(f"{MODE_ENV}=on requires {env_name}=off (single owner of reasoning carry)")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


# ---------------------------------------------------------------- typed records

def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"


@dataclasses.dataclass(frozen=True)
class Receipt:
    turn: int
    tool: str
    target: str
    outcome: str
    detail: str
    revision: str
    seq: int = 0            # host-assigned monotonic order across receipts and claims

    def validate(self) -> None:
        if not isinstance(self.turn, int) or self.turn < 0:
            raise ValueError("receipt.turn must be a non-negative int")
        if not isinstance(self.tool, str) or not self.tool or any(ch.isspace() for ch in self.tool):
            raise ValueError("receipt.tool must be a non-empty token")
        if not isinstance(self.target, str) or len(self.target) > MAX_TARGET or "\n" in self.target:
            raise ValueError("receipt.target must be a single-line string <= MAX_TARGET")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"receipt.outcome {self.outcome!r} is not a known outcome")
        if not isinstance(self.detail, str) or len(self.detail) > MAX_DETAIL or "\n" in self.detail:
            raise ValueError("receipt.detail must be a single-line string <= MAX_DETAIL")
        if not isinstance(self.revision, str):
            raise ValueError("receipt.revision must be a string")
        if not isinstance(self.seq, int) or self.seq < 0:
            raise ValueError("receipt.seq must be a non-negative int")

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Claim:
    id: str
    kind: str
    text: str
    turn: int
    revision: str
    footprint: tuple[str, ...]
    status: str = "candidate"
    seq: int = 0

    def validate(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("claim.id must be a non-empty string")
        if self.kind not in CLAIM_KINDS:
            raise ValueError(f"claim.kind {self.kind!r} is not a known kind")
        if (not isinstance(self.text, str) or not self.text or len(self.text) > MAX_CLAIM_TEXT
                or "\n" in self.text):
            raise ValueError("claim.text must be a non-empty single line <= MAX_CLAIM_TEXT")
        if _DONE_RE.search(self.text):
            raise ValueError("claim.text must never carry a done-like sentence")
        if not isinstance(self.turn, int) or self.turn < 0:
            raise ValueError("claim.turn must be a non-negative int")
        if not isinstance(self.revision, str):
            raise ValueError("claim.revision must be a string")
        if (not isinstance(self.footprint, tuple) or len(self.footprint) > MAX_FOOTPRINT
                or not all(isinstance(item, str) and item for item in self.footprint)):
            raise ValueError("claim.footprint must be a tuple of non-empty strings")
        if self.status not in CLAIM_STATUSES:
            raise ValueError(f"claim.status {self.status!r} is not a known status")

    def as_dict(self) -> dict[str, object]:
        row = dataclasses.asdict(self)
        row["footprint"] = list(self.footprint)
        return row


# ---------------------------------------------------------------- claim extraction

_FENCE_RE = re.compile(r"```.*?(?:```|\Z)", re.S)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`])")
_NOISE_RE = re.compile(r"^(?:(?:wait|actually|hmm+|okay|ok|oh)\b[\s,.!:;-]*)+", re.I)
_BULLET_RE = re.compile(r"^(?:[-*]\s+|\d+\.\s+)")
_DONE_RE = re.compile(r"\b(?:done|complete|completed|finished|finalize)\b"
                      # request fitting extension (measured on j460 ldev_12: "So maybe ... no changes are needed."):
                      r"|no (?:further )?changes? (?:is |are )?(?:needed|required)|nothing (?:left|else) to (?:do|change|fix)",
                      re.I)
_RULED_OUT_RE = re.compile(
    r"not the (?:issue|problem|cause)|doesn'?t matter|no need to|is (?:already )?(?:fine|correct)"
    r"|already (?:handled|done|applied|present)|don'?t need to", re.I)
_CONCLUSION_RE = re.compile(
    r"^(?:so\b|therefore\b|thus\b|the (?:fix|issue|problem|bug|root cause)\b|this means\b|that means\b"
    r"|the test (?:expects|asserts|wants|requires)\b|which means\b)", re.I)
_NEXT_STEP_RE = re.compile(r"^(?:i need to|i'll|i will|let me|i should|next i|now i)\b", re.I)
_OPEN_WORDS_RE = re.compile(r"not sure|unclear|\bmaybe\b|i wonder", re.I)
_FIX_CLAIM_RE = re.compile(r"^(?:so )?(?:the )?fix|should (?:be|use)|i(?:'ll| will) (?:change|fix|make)", re.I)

_PATH_RE = re.compile(
    r"(?<![\w/.-])((?:[\w-]+/)*[\w-]+\.(?:py|pyi|js|jsx|ts|tsx|json|toml|cfg|ini|ya?ml|md|txt|go|rs|java|rb))\b")
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_IDENT_STOP = frozenset({"None", "True", "False", "self", "cls", "def", "class", "return", "import",
                         "from", "and", "or", "not", "in", "is", "if", "else", "for", "with", "as",
                         "py"})


def _sentences(reasoning: str) -> list[str]:
    text = _FENCE_RE.sub("\n\n", str(reasoning or ""))
    out: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        for line in paragraph.split("\n"):
            line = line.strip()
            if not line:
                continue
            for sentence in _SENTENCE_SPLIT_RE.split(line):
                sentence = _BULLET_RE.sub("", sentence.strip())
                sentence = _NOISE_RE.sub("", sentence).strip()
                if len(sentence) >= 12:
                    out.append(sentence)
    return out


def classify_sentence(sentence: str) -> str | None:
    """Kind of one cleaned sentence, or None.  Done-like sentences are never claims."""
    if _DONE_RE.search(sentence):
        return None
    if _RULED_OUT_RE.search(sentence):
        return "RULED_OUT"
    if sentence.rstrip().endswith("?"):
        return "OPEN"
    if _CONCLUSION_RE.search(sentence):
        return "CONCLUSION"
    if _NEXT_STEP_RE.search(sentence):
        return "NEXT_STEP"
    if _OPEN_WORDS_RE.search(sentence):
        return "OPEN"
    return None


def footprint_of(sentence: str) -> tuple[str, ...]:
    """Paths and backtick identifiers named in a sentence (sorted, bounded)."""
    found: set[str] = set(_PATH_RE.findall(sentence))
    for span in _BACKTICK_RE.findall(sentence):
        found.update(_PATH_RE.findall(span))
        for ident in _IDENT_RE.findall(span):
            if ident not in _IDENT_STOP and len(ident) >= 2:
                found.add(ident)
    return tuple(sorted(found)[:MAX_FOOTPRINT])


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def _jaccard(left: str, right: str) -> float:
    a, b = set(left.split()), set(right.split())
    return len(a & b) / max(1, len(a | b))


def extract_claims(reasoning: str, *, turn: int, revision: str) -> list[Claim]:
    """Deterministic claims from one response: last 2 CONCLUSION/RULED_OUT, last 1 OPEN/NEXT_STEP.

    ``id``/``seq`` are provisional ("t<turn>.<n>"); ReasoningLedger re-stamps them on admission.
    """
    by_kind: dict[str, list[str]] = {kind: [] for kind in CLAIM_KINDS}
    for sentence in _sentences(reasoning):
        kind = classify_sentence(sentence)
        if kind is not None:
            by_kind[kind].append(sentence)
    picked: list[tuple[int, str, str]] = []
    order = {sentence: index for index, sentence in enumerate(_sentences(reasoning))}
    for kind in CLAIM_KINDS:
        for sentence in by_kind[kind][-PER_RESPONSE[kind]:]:
            picked.append((order.get(sentence, 0), kind, sentence))
    picked.sort()
    claims: list[Claim] = []
    for index, (_pos, kind, sentence) in enumerate(picked):
        claim = Claim(id=f"t{int(turn)}.{index}", kind=kind, text=_clip(sentence, MAX_CLAIM_TEXT),
                      turn=int(turn), revision=str(revision), footprint=footprint_of(sentence))
        claim.validate()
        claims.append(claim)
    return claims


# ---------------------------------------------------------------- receipt classification

_FAILED_ID_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.M)
_ASSERT_HEAD_RE = re.compile(r"^E\s+(\S.*)$", re.M)


def _first_line(text: str) -> str:
    return text.strip().split("\n", 1)[0].strip()


def _after_colon(line: str) -> str:
    return line.split(":", 1)[1].strip() if ":" in line else line


def _red_detail(content: str) -> str:
    match = _FAILED_ID_RE.search(content)
    if match:
        return match.group(1)[:MAX_DETAIL]
    match = _ASSERT_HEAD_RE.search(content)
    if match:
        return match.group(1).strip()[:MAX_DETAIL]
    return ""


def _parse_args(arguments_json: object) -> dict:
    if isinstance(arguments_json, Mapping):
        return dict(arguments_json)
    try:
        parsed = json.loads(arguments_json) if isinstance(arguments_json, str) else {}
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _target_of(tool: str, args: Mapping[str, object]) -> str:
    keys = ("path", "selector", "target", "file")
    if tool == "grep":
        keys = ("pattern", "path")
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _clip(value.replace(" -> ", " - "), MAX_TARGET)
    return ""


def classify_receipt(tool_name: str, arguments_json: str, result_content: str) -> dict[str, str]:
    """Pure, marker-based outcome of one closed tool exchange: {"target","outcome","detail"}.

    Markers are the literal host strings seen in r46 f4 journals (README "markers").  The most
    specific marker wins; tool-specific fallbacks apply only when no marker matched.
    """
    tool = str(tool_name or "").strip() or "?"
    args = _parse_args(arguments_json)
    content = str(result_content or "")
    head = _first_line(content)
    target = _target_of(tool, args)

    def row(outcome: str, detail: str = "") -> dict[str, str]:
        return {"target": target, "outcome": outcome, "detail": _clip(detail, MAX_DETAIL) if detail else ""}

    if head.startswith("[step-contract] REJECTED"):
        return row("REJECTED_CONTRACT", _after_colon(head))
    if "probe POLICY_DENIED" in content:
        return row("PROBE_DENIED", _after_colon(head) if head.startswith("probe POLICY_DENIED:") else "")
    if re.match(r"pytest exit (\d+)", head):
        code = int(re.match(r"pytest exit (\d+)", head).group(1))
        return row("TESTS_GREEN") if code == 0 else row("TESTS_RED", _red_detail(content))
    if "verification_result REUSED_PASS" in content or "REUSED_VERIFICATION" in content:
        return row("TESTS_REUSED")
    if head.startswith("[ripple-preview] STAGED"):
        return row("EDIT_STAGED")
    if head.startswith("edit NO-OP"):
        return row("EDIT_NO_OP")
    if head.startswith("edit applied."):
        return row("EDIT_APPLIED")
    if head.startswith("[stub-sweep]"):
        return row("READ_ALREADY_PRESENT", "stub-sweep")
    if ("already_in_frame_as" in content or "ALREADY_PRESENT" in content
            or "READ_UNCHANGED" in content):
        return row("READ_ALREADY_PRESENT")
    if ("grep:ERROR" in content or "scope does not exist" in content
            or head.startswith("Search failed")):
        return row("GREP_ERROR", re.sub(r"^(?:Search failed:\s*|grep:ERROR\s*)", "", head))
    if head.startswith("[coverage: TRUNCATED"):
        match = re.match(r"\[coverage: TRUNCATED\((\d+),(\d+)\)", head)
        return row("GREP_HITS", f"TRUNCATED {match.group(1)}/{match.group(2)}" if match else "TRUNCATED")
    if head.startswith("probe COMPLETED"):
        return row("PROBE_OK")
    if head.startswith("no such file"):
        return row("READ_REJECTED", head)
    if re.search(r"\bREFUSED\b", head):
        return row("REFUSED", head)
    if tool in {"edit", "str_replace", "write", "insert"} and re.search(r"\bREJECTED\b", head, re.I):
        return row("EDIT_REJECTED", head)
    if tool == "grep":
        lines = [line for line in content.strip().split("\n") if line.strip()]
        if not lines or head.lower().startswith(("no match", "0 match")):
            return row("GREP_EMPTY")
        return row("GREP_HITS", f"{len(lines)} lines")
    if tool == "read" and content.strip():
        return row("READ_OK")
    return row("OTHER", head[:50])


# ---------------------------------------------------------------- the ledger

def _call_parts(call: Mapping[str, object]) -> tuple[str | None, str, str]:
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping):
        return None, "", ""
    name = function.get("name")
    raw = function.get("arguments")
    call_id = call.get("id")
    return (name.strip() if isinstance(name, str) and name.strip() else None,
            raw if isinstance(raw, str) else "",
            call_id if isinstance(call_id, str) else "")


def _edit_text(arguments_json: str) -> str:
    args = _parse_args(arguments_json)
    return "\n".join(str(args.get(key)) for key in ("function_source", "new_str", "old_str", "content",
                                                      "source", "text")
                     if isinstance(args.get(key), str))


def _touches(claim: Claim, path: str, edit_text: str) -> bool:
    if not path and not edit_text:
        return False
    stem = os.path.splitext(os.path.basename(path))[0] if path else ""
    base = os.path.basename(path) if path else ""
    for item in claim.footprint:
        if path and (item == path or item == base or path.endswith("/" + item)):
            return True
        if "/" not in item and not item.endswith(".py"):
            if stem and item.split(".", 1)[0] == stem:
                return True
            if len(item) >= 4 and edit_text and re.search(r"(?<![\w.])" + re.escape(item) + r"\b", edit_text):
                return True
    return False


class ReasoningLedger:
    """Typed receipts + model claims for one run.  Deterministic, bounded, JSON-snapshottable."""

    def __init__(self) -> None:
        self._seq = 0
        self._revision: str | None = None
        self._claims: list[Claim] = []
        self._receipts: list[Receipt] = []
        self._seen_calls: list[str] = []
        self._seen_responses: list[str] = []

    # ---- bookkeeping
    @property
    def current_revision(self) -> str:
        return self._revision if self._revision is not None else "UNKNOWN"

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _candidates(self, kind: str | None = None) -> list[Claim]:
        return [claim for claim in self._claims
                if claim.status == "candidate" and (kind is None or claim.kind == kind)]

    def _retire(self, claim: Claim, status: str) -> None:
        claim.status = status

    def _trim_audit(self) -> None:
        """Keep the append-only tape intact; boundedness belongs to rendering."""
        return

    def counts(self) -> dict[str, int]:
        return {
            "claims": len(self._candidates()),
            "receipts": len(self._receipts),
            "stale": sum(1 for claim in self._claims if claim.status == "stale"),
            "contradicted": sum(1 for claim in self._claims if claim.status == "contradicted"),
        }

    # ---- inputs
    def note_revision(self, revision: str) -> bool:
        """Called on every prepare.  A revision change retires OPEN and NEXT_STEP candidates."""
        revision = str(revision)
        changed = self._revision is not None and revision != self._revision
        if changed:
            for claim in self._candidates():
                if claim.kind in ("OPEN", "NEXT_STEP"):
                    self._retire(claim, "stale")
            self._trim_audit()
        self._revision = revision
        return changed

    def mark_evidence_stale(
        self, *, paths: tuple[str, ...] = (), evidence_refs: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Retire claims whose measured footprint/evidence was invalidated."""
        normalized_paths = tuple(str(item).replace("\\", "/") for item in paths if str(item))
        refs = {str(item) for item in evidence_refs if str(item)}
        changed: list[str] = []
        for claim in self._candidates():
            path_hit = any(_touches(claim, path, "") for path in normalized_paths)
            ref_hit = bool(refs.intersection(claim.footprint))
            if path_hit or ref_hit:
                self._retire(claim, "stale")
                changed.append(claim.id)
        self._trim_audit()
        return tuple(changed)

    def observe_response(self, reasoning: str | None, *, turn: int, revision: str,
                         request_id: str | None = None) -> dict[str, object]:
        """Admit claims from one response's reasoning_content (idempotent per request_id)."""
        added = {kind: 0 for kind in CLAIM_KINDS}
        if request_id:
            if request_id in self._seen_responses:
                return {"added": added, "duplicate": True, **self.counts()}
            self._seen_responses = [*self._seen_responses, request_id][-MAX_SEEN_IDS:]
        if not isinstance(reasoning, str) or not reasoning.strip():
            return {"added": added, "duplicate": False, **self.counts()}
        for claim in extract_claims(reasoning, turn=turn, revision=revision):
            normalized = _norm(claim.text)
            if any(_norm(old.text) == normalized or _jaccard(_norm(old.text), normalized) >= 0.6
                   for old in self._candidates(claim.kind)):
                continue
            if claim.kind == "NEXT_STEP":
                for old in self._candidates("NEXT_STEP"):
                    self._retire(old, "stale")
            seq = self._next_seq()
            claim.id, claim.seq = f"c{seq}", seq
            self._claims.append(claim)
            added[claim.kind] += 1
            live = self._candidates(claim.kind)
            for old in live[:max(0, len(live) - CLAIM_CAPS[claim.kind])]:
                self._retire(old, "stale")
        self._trim_audit()
        return {"added": added, "duplicate": False, **self.counts()}

    def observe_receipt(self, call: Mapping[str, object], result: Mapping[str, object] | str, *,
                        turn: int, revision: str) -> Receipt | None:
        """Admit one closed exchange's host receipt and apply invalidation (idempotent per call id)."""
        tool, raw_args, call_id = _call_parts(call)
        if tool is None:
            return None
        content = result.get("content") if isinstance(result, Mapping) else result
        if not isinstance(content, str):
            return None
        if call_id:
            if call_id in self._seen_calls:
                return None
            self._seen_calls = [*self._seen_calls, call_id][-MAX_SEEN_IDS:]
        info = classify_receipt(tool, raw_args, content)
        receipt = Receipt(turn=int(turn), tool=_clip(tool, 40).replace(" ", "_"), target=info["target"],
                          outcome=info["outcome"], detail=info["detail"], revision=str(revision),
                          seq=self._next_seq())
        receipt.validate()
        self._invalidate(receipt, raw_args)
        self._receipts.append(receipt)
        self._trim_receipts()
        return receipt

    def _invalidate(self, receipt: Receipt, raw_args: str) -> None:
        if receipt.outcome in ("EDIT_APPLIED", "EDIT_STAGED"):
            edit_text = _edit_text(raw_args)
            for claim in self._candidates():
                if _touches(claim, receipt.target, edit_text):
                    self._retire(claim, "stale")
        elif receipt.outcome == "TESTS_RED":
            edits_after = [row.seq for row in self._receipts if row.outcome == "EDIT_APPLIED"]
            for claim in self._candidates("CONCLUSION"):
                if not _FIX_CLAIM_RE.search(claim.text):
                    continue
                if claim.revision != receipt.revision or any(seq > claim.seq for seq in edits_after):
                    self._retire(claim, "contradicted")
        elif receipt.outcome in ("REJECTED_CONTRACT", "REFUSED"):
            for claim in self._candidates("NEXT_STEP"):
                self._retire(claim, "stale")
        self._trim_audit()

    def _protected_receipts(self) -> set[int]:
        protected: set[int] = set()
        for group in ({"TESTS_RED"}, _EDIT_OUTCOMES):
            latest = [row for row in self._receipts if row.outcome in group]
            if latest:
                protected.add(latest[-1].seq)
        return protected

    def _trim_receipts(self) -> None:
        """Compatibility hook retained for callers; it must never evict history."""
        return

    # ---- views
    def claims(self) -> tuple[Claim, ...]:
        return tuple(dataclasses.replace(claim) for claim in self._claims)

    def receipts(self) -> tuple[Receipt, ...]:
        return tuple(self._receipts)

    # ---- render
    def _receipt_rows(self, current_revision: str) -> list[tuple[str, int, str]]:
        """[(drop_class, drop_rank, text)] in render order.  drop_class: 'never'|'read'|'other'."""
        rows = list(self._receipts)
        tests = [row for row in rows if row.outcome in _TEST_OUTCOMES]
        latest_test = tests[-1] if tests else None
        protected = self._protected_receipts()
        newest_first = sorted(rows, key=lambda row: -row.seq)

        def group(row: Receipt) -> int:
            if row is latest_test:
                return 0
            if row.outcome in _EDIT_OUTCOMES:
                return 1
            if row.outcome in _REFUSAL_OUTCOMES:
                return 2
            return 3

        ordered = sorted(newest_first, key=lambda row: (group(row), -row.seq))
        out = []
        for row in ordered:
            text = f"R t{row.turn} {row.tool} {row.target or '-'} -> {row.outcome}"
            if row.detail:
                text += f": {row.detail}"
            if row.revision != current_revision:
                text += OLDER_TAG
            if row.seq in protected:
                cls = "never"
            elif group(row) == 3:
                cls = "read"
            else:
                cls = "other"
            out.append((cls, row.seq, text))
        return out

    def _claim_rows(self, current_revision: str) -> list[tuple[str, int, str]]:
        live = self._candidates()
        kind_rank = {kind: index for index, kind in enumerate(CLAIM_KINDS)}
        live = [claim for kind in CLAIM_KINDS for claim in self._candidates(kind)[-CLAIM_CAPS[kind]:]]
        ordered = sorted(live, key=lambda claim: (claim.revision != current_revision,
                                                  kind_rank[claim.kind], claim.seq))
        out = []
        for claim in ordered:
            text = f"C {claim.kind} t{claim.turn}: {claim.text}"
            if claim.revision != current_revision:
                text += OLDER_TAG
            cls = "never" if claim.kind == "NEXT_STEP" else claim.kind
            out.append((cls, claim.seq, text))
        return out

    def render_block(self, current_revision: str) -> str:
        """One deterministic block within MAX_RENDER_TOKENS, or "" when nothing to render."""
        current_revision = str(current_revision)
        rows = [*self._receipt_rows(current_revision), *self._claim_rows(current_revision)]
        if not rows:
            return ""

        def text_of(kept: list[tuple[str, int, str]]) -> str:
            return "\n".join([LEDGER_HEADER, *(row[2] for row in kept)])

        kept = list(rows)
        # Drop order: reads/greps/probes/older tests -> RULED_OUT -> CONCLUSION -> OPEN ->
        # remaining non-protected receipts (older edits/refusals); each oldest first.
        budget = max_render_tokens()  # 
        for drop_class in ("read", "RULED_OUT", "CONCLUSION", "OPEN", "other"):
            while token_estimate(text_of(kept)) > budget:
                victims = [row for row in kept if row[0] == drop_class]
                if not victims:
                    break
                oldest = min(victims, key=lambda row: row[1])
                kept = [row for row in kept if row is not oldest]
        block = text_of(kept)
        estimate = token_estimate(block)
        if estimate > budget:
            raise ValueError(f"reasoning ledger block is {estimate} tokens; budget is {budget}")
        if len(kept) == 0:
            return ""
        return block

    # ---- checkpoint continuity
    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "seq": self._seq,
            "revision": self._revision,
            "claims": [claim.as_dict() for claim in self._claims],
            "receipts": [receipt.as_dict() for receipt in self._receipts],
            "seen_calls": list(self._seen_calls),
            "seen_responses": list(self._seen_responses),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object] | None) -> "ReasoningLedger":
        ledger = cls()
        if not isinstance(state, Mapping) or state.get("schema") != SCHEMA:
            return ledger
        ledger._seq = int(state.get("seq") or 0)
        revision = state.get("revision")
        ledger._revision = str(revision) if revision is not None else None
        for row in state.get("claims") or ():
            if isinstance(row, Mapping):
                claim = Claim(id=str(row["id"]), kind=str(row["kind"]), text=str(row["text"]),
                              turn=int(row["turn"]), revision=str(row["revision"]),
                              footprint=tuple(str(item) for item in row.get("footprint") or ()),
                              status=str(row.get("status", "candidate")), seq=int(row.get("seq", 0)))
                claim.validate()
                ledger._claims.append(claim)
        for row in state.get("receipts") or ():
            if isinstance(row, Mapping):
                receipt = Receipt(turn=int(row["turn"]), tool=str(row["tool"]), target=str(row["target"]),
                                  outcome=str(row["outcome"]), detail=str(row["detail"]),
                                  revision=str(row["revision"]), seq=int(row.get("seq", 0)))
                receipt.validate()
                ledger._receipts.append(receipt)
        ledger._seen_calls = [str(item) for item in state.get("seen_calls") or ()][-MAX_SEEN_IDS:]
        ledger._seen_responses = [str(item) for item in state.get("seen_responses") or ()][-MAX_SEEN_IDS:]
        return ledger


# ---------------------------------------------------------------- block grammar

_RECEIPT_ROW_RE = re.compile(r"^R t(\d+) (\S+) (.+?) -> ([A-Z_]+)(?:: (.+?))?( \(older rev\))?$")
_CLAIM_ROW_RE = re.compile(r"^C (CONCLUSION|RULED_OUT|OPEN|NEXT_STEP) t(\d+): (.+)$")


def validate_block(text: str) -> bool:
    """Grammar check of a rendered block (header + typed rows + budget).  Raises ValueError."""
    if not isinstance(text, str) or not text:
        raise ValueError("reasoning ledger block must be a non-empty string")
    lines = text.split("\n")
    if lines[0] != LEDGER_HEADER:
        raise ValueError("reasoning ledger block header mismatch")
    if len(lines) < 2:
        raise ValueError("reasoning ledger block has no rows")
    for line in lines[1:]:
        receipt = _RECEIPT_ROW_RE.match(line)
        if receipt:
            if receipt.group(4) not in OUTCOMES:
                raise ValueError(f"unknown receipt outcome in row: {line!r}")
            if receipt.group(5) is not None and len(receipt.group(5)) > MAX_DETAIL:
                raise ValueError("receipt detail over MAX_DETAIL")
            continue
        claim = _CLAIM_ROW_RE.match(line)
        if claim:
            body = claim.group(3)
            if body.endswith(OLDER_TAG):
                body = body[:-len(OLDER_TAG)]
            if len(body) > MAX_CLAIM_TEXT or _DONE_RE.search(body):
                raise ValueError(f"invalid claim row: {line!r}")
            continue
        raise ValueError(f"reasoning ledger row does not match the grammar: {line!r}")
    if token_estimate(text) > max_render_tokens():
        raise ValueError("reasoning ledger block over the render budget")
    return True


__all__ = [
    "CLAIM_CAPS", "CLAIM_KINDS", "CLAIM_STATUSES", "EVENT_DELIVERED", "EVENT_EVALUATED", "LEDGER_HEADER",
    "MAX_RECEIPTS", "MAX_RENDER_TOKENS", "MAX_TOKENS_ENV", "MODE_ENV", "OUTCOMES", "SCHEMA", "max_render_tokens",
    "Claim", "Receipt", "ReasoningLedger",
    "classify_receipt", "classify_sentence", "enabled", "extract_claims", "footprint_of", "mode",
    "validate_block",
]

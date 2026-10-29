"""Two-stage wide-recall pooling + corpus-derived boilerplate stoplist.

Pure, deterministic, stdlib-only. No I/O, no model calls.

Design sources (read in full before implementing):
    /tmp/solq/w50/a_pool_recall.md
    /tmp/solq/w50/a_boilerplate_fix.md

Zero-false-locate guarantee (a_pool_recall.md, "Zero-false-locate guarantee"
section): retrieval alone cannot guarantee zero new false candidates -- a
wrong file may still contain >=2 symptom tokens. The guarantee instead comes
from ROLLOUT behavior: the existing pool prefix (whatever was already being
shown/considered) is preserved byte-identically, and newly recovered
candidates are appended only after that prefix, never spliced in or used to
reorder it. `append_only_union()` implements exactly that append-only
contract; as long as callers keep obeying it (never exposing the widened
tail through a different acceptance path), Delta(false-locates) == 0 while
gold-in-pool recall improves. This module does not and cannot enforce that
callers actually gate on the prefix -- it only guarantees the prefix itself
is never mutated or reordered by the union operation.

Ambiguity notes (choices made where the design docs did not fully specify
a concrete Python API -- called out again in the task report):

1. `gold_set` protocol (a_boilerplate_fix.md Sec 6): the doc's
   `passes_gold_guard` calls a module-level `evaluate_locate(...)` free
   function that isn't defined anywhere in the doc. We treat it as
   injectable: `gold_set` must expose `.tickets` (iterable of held-out
   items), `.exclude(ticket)` (returns the training set minus that ticket),
   and `.evaluate_locate(ticket, stoplist, training_tickets)` (returns a
   float "distinctiveness" score for that configuration). This keeps the
   guard pure/deterministic and lets tests inject a trivial scoring
   function without requiring a real locate-feature pipeline.
2. `DerivedStoplist.stats[word]` needs an attribute-style `.rejection_reason`
   per the doc's test assertions (`result.stats["kafka"].rejection_reason`).
   We use a small frozen dataclass `WordStat` for every word considered
   (accepted or rejected) carrying `rejection_reason` (`None` when accepted)
   plus df/rate/mass_share for audit, matching Sec 2's "recorded for audit"
   instruction.
3. `hits_fn` in `two_stage_pool` is a caller-supplied function rather than
   this module re-deriving tokenization, since Stage B in a_pool_recall.md
   Sec 3 operates over "indexed file/function content" -- a corpus-specific
   concept this module has no access to.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Section: corpus-derived boilerplate stoplist (a_boilerplate_fix.md)
# ---------------------------------------------------------------------------

#: Negations are ALWAYS protected regardless of prevalence (Sec 2.6).
PROTECTED_NEGATIONS = frozenset({"not", "never", "without", "no"})

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
# Issue ids like ABC-123, JIRA-4567, #1234.
_ISSUE_ID_RE = re.compile(r"\b[a-z]{2,}-\d+\b|#\d+\b", re.IGNORECASE)
# Fenced code blocks (```...```) removed before tokenization.
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
# Inline code spans.
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
# Markdown syntax: headers, emphasis markers, list bullets, link brackets.
_MARKDOWN_RE = re.compile(r"[#*_>\[\]()~]")
# Bare numbers (including decimals) -- removed as a token class.
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
# Tokens: unicode word chars plus internal hyphens (hyphenated words kept).
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)


def content_tokens(text: str) -> list[str]:
    """Tokenize per a_boilerplate_fix.md Sec 2.

    Steps: unicode-normalize + lowercase; strip URLs/UUIDs/issue-ids/numbers/
    markdown/code blocks; keep hyphenated words as single tokens; no
    stemming; negations are handled by the caller (this function does not
    drop or alter them -- it only tokenizes).
    """
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = _CODE_BLOCK_RE.sub(" ", normalized)
    normalized = _INLINE_CODE_RE.sub(" ", normalized)
    normalized = _URL_RE.sub(" ", normalized)
    normalized = _UUID_RE.sub(" ", normalized)
    normalized = _ISSUE_ID_RE.sub(" ", normalized)
    normalized = _NUMBER_RE.sub(" ", normalized)
    normalized = _MARKDOWN_RE.sub(" ", normalized)
    return _TOKEN_RE.findall(normalized)


@dataclass(frozen=True)
class Ticket:
    """Minimal ticket shape the derivation algorithm needs."""

    id: str
    title: str
    body: str
    component: Optional[str] = None
    labels: tuple = ()
    fields: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class StoplistConfig:
    prevalence_threshold: float = 0.80
    min_tickets: int = 21
    max_gold_regression: float = 0.01


@dataclass(frozen=True)
class WordStat:
    """Audit record for one candidate word (accepted or rejected)."""

    word: str
    df: int
    rate: float
    mass: int
    mass_share: float
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class DerivedStoplist:
    words: frozenset
    derived_words: frozenset = frozenset()
    status: str = "derived"
    reason: str = ""
    stats: Mapping[str, WordStat] = field(default_factory=dict)


def _quoted_identifiers(text: str) -> set[str]:
    found: set[str] = set()
    for quote in ('"', "'", "`"):
        pattern = re.compile(re.escape(quote) + r"([^" + re.escape(quote) + r"]+)" + re.escape(quote))
        for m in pattern.finditer(text or ""):
            found.update(content_tokens(m.group(1)))
    return found


def _stack_trace_and_path_tokens(text: str) -> set[str]:
    """Symbols, file paths, and stack-trace-looking lines get fail-open protection."""
    found: set[str] = set()
    if not text:
        return found
    # File paths like a/b/c.py or module.submodule.func
    for m in re.finditer(r"\b[\w./\\-]+\.(?:py|js|ts|java|go|rb|cpp|c|h|rs|json|yaml|yml)\b", text, re.IGNORECASE):
        found.update(content_tokens(m.group(0)))
    # Dotted symbol-looking references e.g. foo.bar.baz(
    for m in re.finditer(r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.){1,}[A-Za-z_][A-Za-z0-9_]*\b", text):
        found.update(content_tokens(m.group(0)))
    # Lines that look like stack frames ("at ...", "File "...", line N)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("at ") or stripped.startswith("File ") or "Traceback" in stripped:
            found.update(content_tokens(stripped))
    return found


def build_protected_vocabulary(tickets: Iterable[Ticket], glossary: Iterable[str]) -> frozenset:
    """Protected vocabulary per a_boilerplate_fix.md Sec 5. Fail OPEN: one
    protected occurrence anywhere is enough to protect a word forever."""
    protected: set[str] = set()

    for word in glossary or ():
        protected.update(content_tokens(word))

    protected.update(PROTECTED_NEGATIONS)

    for ticket in tickets:
        protected.update(content_tokens(ticket.title or ""))
        # Component/service/product/subsystem fields.
        if ticket.component:
            protected.update(content_tokens(ticket.component))
        if ticket.fields:
            for key in ("service", "product", "subsystem", "component", "summary"):
                val = ticket.fields.get(key)
                if isinstance(val, str):
                    protected.update(content_tokens(val))
        for label in ticket.labels or ():
            protected.update(content_tokens(label))
        # Stack traces / file paths / symbols / code -- scan the raw body
        # (before code-block stripping) since that's where these surfaces live.
        protected.update(_stack_trace_and_path_tokens(ticket.body or ""))
        # Quoted identifiers.
        protected.update(_quoted_identifiers(ticket.body or ""))
        protected.update(_quoted_identifiers(ticket.title or ""))

    return frozenset(protected)


def leave_one_out_stable(word: str, token_sets: Mapping[str, Sequence[str]], threshold: float) -> bool:
    """Exact port of a_boilerplate_fix.md Sec 4."""
    n = len(token_sets)
    if n <= 1:
        return False
    sets = {tid: set(tokens) for tid, tokens in token_sets.items()}
    total_frequency = sum(word in tokens for tokens in sets.values())

    for tokens in sets.values():
        frequency_without_ticket = total_frequency - int(word in tokens)
        rate_without_ticket = frequency_without_ticket / (n - 1)
        if rate_without_ticket <= threshold:
            return False

    return True


def passes_gold_guard(word: str, hand_stoplist, gold_set, max_regression: float) -> bool:
    """Exact port of a_boilerplate_fix.md Sec 6.

    `gold_set` must provide:
      - `.tickets`: iterable of held-out tickets
      - `.exclude(ticket)`: training set excluding that ticket
      - `.evaluate_locate(ticket, stoplist, training_tickets) -> float`
    (see module docstring, ambiguity note 1, for why this is injected rather
    than calling a free-standing `evaluate_locate`).
    """
    gains = []

    for held_out_ticket in gold_set.tickets:
        training = gold_set.exclude(held_out_ticket)

        baseline = gold_set.evaluate_locate(
            ticket=held_out_ticket,
            stoplist=hand_stoplist,
            training_tickets=training,
        )
        with_word_removed = gold_set.evaluate_locate(
            ticket=held_out_ticket,
            stoplist=frozenset(hand_stoplist) | {word},
            training_tickets=training,
        )
        gains.append(with_word_removed - baseline)

    if not gains:
        return True
    return min(gains) >= -max_regression


def _build_stats(
    *,
    document_frequency: Mapping[str, int],
    mass: Mapping[str, int],
    total_mass: int,
    corpus_size: int,
    rejection_reasons: Mapping[str, Optional[str]],
) -> dict:
    stats: dict[str, WordStat] = {}
    for word, df in document_frequency.items():
        rate = df / corpus_size if corpus_size else 0.0
        m = mass.get(word, 0)
        mass_share = (m / total_mass) if total_mass else 0.0
        stats[word] = WordStat(
            word=word,
            df=df,
            rate=rate,
            mass=m,
            mass_share=mass_share,
            rejection_reason=rejection_reasons.get(word),
        )
    return stats


def derive_stoplist(
    tickets: Sequence[Ticket],
    hand_stoplist,
    glossary,
    gold_set,
    config: StoplistConfig,
) -> DerivedStoplist:
    """Exact port of a_boilerplate_fix.md Sec 3 `derive_stoplist`.

    The derived list REPLACES the previous derived list on every call --
    callers must not accumulate across harvests (Sec 3 preamble; guard test
    `test_harvest_update_removes_no_longer_common_word`).
    """
    hand_stoplist = frozenset(hand_stoplist or ())
    n = len(tickets)

    if n < config.min_tickets:
        return DerivedStoplist(
            words=hand_stoplist,
            derived_words=frozenset(),
            status="hand_only",
            reason="insufficient_corpus",
            stats={},
        )

    token_sets: dict[str, list[str]] = {
        ticket.id: content_tokens(ticket.body) for ticket in tickets
    }

    document_frequency: Counter = Counter()
    mass: Counter = Counter()
    total_mass = 0
    for tokens in token_sets.values():
        for word in set(tokens):
            document_frequency[word] += 1
        for word in tokens:
            mass[word] += 1
            total_mass += 1

    protected = build_protected_vocabulary(tickets=tickets, glossary=glossary)

    candidates: set[str] = set()
    rejection_reasons: dict[str, Optional[str]] = {}

    # Sort for determinism (dict/set iteration order is insertion-order in
    # CPython but we don't rely on that here).
    for word in sorted(document_frequency.keys()):
        frequency = document_frequency[word]

        if word in hand_stoplist:
            continue

        if word in protected:
            rejection_reasons[word] = "protected_domain_surface"
            continue

        if frequency / n <= config.prevalence_threshold:
            continue

        if not leave_one_out_stable(
            word=word,
            token_sets=token_sets,
            threshold=config.prevalence_threshold,
        ):
            rejection_reasons[word] = "leave_one_out_unstable"
            continue

        if gold_set is not None:
            if not passes_gold_guard(
                word=word,
                hand_stoplist=hand_stoplist,
                gold_set=gold_set,
                max_regression=config.max_gold_regression,
            ):
                rejection_reasons[word] = "gold_regression"
                continue

        candidates.add(word)

    stats = _build_stats(
        document_frequency=document_frequency,
        mass=mass,
        total_mass=total_mass,
        corpus_size=n,
        rejection_reasons=rejection_reasons,
    )

    return DerivedStoplist(
        words=frozenset(hand_stoplist | candidates),
        derived_words=frozenset(candidates),
        status="derived",
        reason="",
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Section: two-stage wide recall pooling (a_pool_recall.md)
# ---------------------------------------------------------------------------


def _candidate_identity(candidate) -> str:
    """Deterministic identity string for a candidate, used for tie-breaks and
    digesting. Prefers a `.path`/`.id` attribute, falls back to str()."""
    for attr in ("path", "id", "identity", "key"):
        val = getattr(candidate, attr, None)
        if val is not None:
            return str(val)
    if isinstance(candidate, Mapping):
        for key in ("path", "id", "identity", "key"):
            if key in candidate:
                return str(candidate[key])
    return str(candidate)


def _digest(identities: Iterable[str]) -> str:
    canonical = "\n".join(sorted(identities))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class PoolResult:
    pool: tuple
    baseline: tuple
    added: tuple
    rejected: tuple
    truncated: bool
    stage_a_size: int
    retained_size: int
    min_hits: int = 0
    pool_cap: int = 0
    stage_id: str = "two_stage_pool"

    def record(self) -> dict:
        """Certificate stage record -- JSON-serializable, exact keys per spec."""
        input_identities = [_candidate_identity(c) for c in self.baseline] + [
            _candidate_identity(c) for c in self.added
        ] + [_candidate_identity(c) for c in self.rejected]
        output_identities = [_candidate_identity(c) for c in self.pool]
        return {
            "stage_id": self.stage_id,
            "status": "ok",
            "pool_size": len(self.pool),
            "baseline_size": len(self.baseline),
            "added": [_candidate_identity(c) for c in self.added],
            "rejected_count": len(self.rejected),
            "truncated": self.truncated,
            "stage_a_size": self.stage_a_size,
            "min_hits": self.min_hits,
            "pool_cap": self.pool_cap,
            "input_digest": _digest(input_identities),
            "output_digest": _digest(output_identities),
        }


def two_stage_pool(
    query_tokens: Sequence[str],
    candidates: Sequence,
    *,
    score_fn: Callable[[object], float],
    hits_fn: Callable[[object], Iterable[str]],
    W: int = 100,
    min_hits: int = 2,
    pool_cap: int = 64,
    anchors: Sequence = (),
) -> PoolResult:
    """Two-stage wide recall then re-rank, a_pool_recall.md Sec 3.

    Stage A: rank ALL candidates by score_fn (descending), take top W unique
             files, deterministic tie-break on identity string.
    Stage B: retain candidates (from the Stage-A window) with
             |hits(c)| >= min_hits. Anchors are always retained regardless
             of hit count.
    Stage C: rerank retained candidates by score_fn (descending, identity
             tie-break) and cap at pool_cap. Anchors are pinned -- they can
             never be dropped by the cap and are never counted against it
             (they ride along in addition to the capped ranked set, then the
             combined result is truncated to pool_cap+len(anchors) only if
             anchors would otherwise be evicted).
    """
    anchor_identities = {_candidate_identity(a) for a in anchors}

    # Dedup candidates by identity, keeping first occurrence.
    seen_ident: set[str] = set()
    unique_candidates = []
    for c in candidates:
        ident = _candidate_identity(c)
        if ident in seen_ident:
            continue
        seen_ident.add(ident)
        unique_candidates.append(c)

    def sort_key(c):
        # Descending score, ascending identity for deterministic tie-break.
        return (-score_fn(c), _candidate_identity(c))

    # Stage A: rank all candidates, take top W unique files.
    stage_a_ranked = sorted(unique_candidates, key=sort_key)
    stage_a_pool = stage_a_ranked[:W]
    stage_a_identities = {_candidate_identity(c) for c in stage_a_pool}

    # Baseline = pre-widening top list, i.e. what pool_cap alone (without
    # widening to W) would have produced from the same ranking -- used for
    # digest/added comparison.
    baseline = tuple(stage_a_ranked[:pool_cap])
    baseline_identities = {_candidate_identity(c) for c in baseline}

    # Ensure anchors are present in the working set even if score_fn ranked
    # them outside the Stage-A window -- anchors are ALWAYS protected.
    working = list(stage_a_pool)
    working_identities = set(stage_a_identities)
    for a in anchors:
        ident = _candidate_identity(a)
        if ident not in working_identities:
            working.append(a)
            working_identities.add(ident)

    # Stage B: hit-gate filter, anchors always survive.
    retained = []
    rejected = []
    for c in working:
        ident = _candidate_identity(c)
        if ident in anchor_identities:
            retained.append(c)
            continue
        hits = set(hits_fn(c))
        matched = hits & set(query_tokens)
        if len(matched) >= min_hits:
            retained.append(c)
        else:
            rejected.append(c)

    retained_size = len(retained)

    # Stage C: rerank retained candidates deterministically, cap at pool_cap,
    # anchors always survive the cap.
    retained_sorted = sorted(retained, key=sort_key)

    anchor_items = [c for c in retained_sorted if _candidate_identity(c) in anchor_identities]
    non_anchor_items = [c for c in retained_sorted if _candidate_identity(c) not in anchor_identities]

    capped_non_anchor = non_anchor_items[:pool_cap]
    truncated = len(non_anchor_items) > pool_cap

    # Combine: anchors first-class, preserved regardless of cap; then capped
    # non-anchor items, re-sorted together for a single deterministic order.
    combined = anchor_items + [
        c for c in capped_non_anchor if _candidate_identity(c) not in anchor_identities
    ]
    pool_final = tuple(sorted(combined, key=sort_key))

    pool_identities = {_candidate_identity(c) for c in pool_final}
    added = tuple(
        c for c in stage_a_ranked
        if _candidate_identity(c) in pool_identities
        and _candidate_identity(c) not in baseline_identities
    )

    return PoolResult(
        pool=pool_final,
        baseline=baseline,
        added=added,
        rejected=tuple(rejected),
        truncated=truncated,
        stage_a_size=len(stage_a_pool),
        retained_size=retained_size,
        min_hits=min_hits,
        pool_cap=pool_cap,
    )


# ---------------------------------------------------------------------------
# Section: append-only union (zero-false-locate guarantee helper)
# ---------------------------------------------------------------------------

#: See module docstring "Zero-false-locate guarantee" for why this must be
#: append-only: it is what preserves an unmodified menu/pool prefix while
#: only ever appending newly recovered candidates after it.
APPEND_ONLY = True


def append_only_union(existing: Sequence, added: Sequence) -> tuple:
    """Preserve `existing` byte-identically as a prefix; append only items
    from `added` not already present (by identity), deduped, in their given
    order, canonical order preserved otherwise.
    """
    existing_identities = {_candidate_identity(c) for c in existing}
    result = list(existing)
    seen = set(existing_identities)
    for c in added:
        ident = _candidate_identity(c)
        if ident in seen:
            continue
        seen.add(ident)
        result.append(c)
    return tuple(result)

"""Real type-aware call resolution, layered on top of call_graph's name matching.

WHY THIS EXISTS
---------------
`call_graph.py` resolves a call edge from an import table plus name matching.
That is fast, dependency-free, and correct for the majority of calls — but it
has a hard accuracy ceiling that no amount of name-matching cleverness fixes,
because the missing information is TYPES:

  * ``obj.area()`` where two unrelated classes both define ``area`` — name
    matching over-reports BOTH (``_CONF_BARE_MULTI``).
  * ``child.describe()`` where ``describe`` is inherited from a base class —
    the self-method branch does not apply (the receiver is not ``self``) and
    the short name may not even exist on the receiver's own class.
  * an aliased deep import whose local alias never appears as a def anywhere.

Rather than reimplement type inference (a multi-year project that would rot),
this module shells out to a REAL inference engine where one is available and
fails open to the existing resolver everywhere else.

WHY JEDI, NOT PYRIGHT
---------------------
`pyright` is already in requirements.txt, so "use pyright" is superficially
tempting. It is the wrong tool here:

  * pyright's CLI (`--outputjson`) emits DIAGNOSTICS, not definitions. There is
    no batch "where does this call go?" mode. Getting goto-definition out of it
    means driving `pyright-langserver` over JSON-RPC: a persistent child
    process, an LSP handshake, initialize/didOpen/definition/shutdown
    bookkeeping, and process-lifecycle failure modes (zombie servers in a
    short-lived coding pod) — all for the same answer.
  * the `pyright` PyPI package is a wrapper that downloads the real npm package
    on first run, so its first invocation needs NETWORK. A code-graph build must
    not depend on that.
  * jedi is pure Python, in-process, zero infra, and exposes exactly the call we
    need (`Script.goto(line, col, follow_imports=True)`).

Measured on the regression fixture, jedi resolves all three failure classes
above correctly. It is the right tool; pyright stays what it is today (a
type-CHECK gate), and this module stays independent of it.

DOCTRINE (inherited from ts_extract.py / call_graph.py)
------------------------------------------------------
  * NEVER RAISE. Every public function returns an empty/None result on any
    failure. A degraded LSP layer must never be able to break a graph build.
  * FAIL OPEN. No answer means "keep whatever the name-matching resolver said",
    never "drop the edge". Confidence is an annotation, never a filter
    under the graph resolver contract.
  * BOUNDED. Every file gets a hard wall-clock budget; a timeout is its own
    telemetry line, not a silent skip.
  * BATCHED. One `jedi.Script` per file serves every call site in that file —
    per-call setup is what makes naive LSP integrations unusably slow.

TYPESCRIPT / JAVASCRIPT: DELIBERATELY NOT IMPLEMENTED
-----------------------------------------------------
The equivalent win for TS/JS needs `tsserver` or `typescript-language-server`.
The agent-runner image (see Dockerfile) installs the Node RUNTIME but
deliberately installs **no global npm packages** — repo-local tooling is
resolved from the cloned repo's own `node_modules` at task time, which is a
per-repo coin flip, not something a graph build may assume. Adding a global
TypeScript install to the image to serve one optional accuracy layer is a real
image-size and supply-chain cost for a speculative benefit.

So TS/JS is NOT handled here. `resolve_file()` returns ``{}`` for any
non-Python path. That is honest and kill-switched off by construction rather
than a half-wired path that silently no-ops under this contract. If a future
image ships a language server, the batch shape in this module is the seam to
extend: same `resolve_file(path, sites)` contract, different backend.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard is environment-dependent
    import jedi  # type: ignore
    _JEDI_AVAILABLE = True
except Exception:  # pragma: no cover - any import failure means "not available"
    jedi = None  # type: ignore
    _JEDI_AVAILABLE = False

# ── Configuration ────────────────────────────────────────────────────────────

#: Env kill-switch. Any of these values (case-insensitive) disables the layer
#: entirely, restoring byte-identical pre-LSP behaviour.
_OFF_VALUES = frozenset({"off", "0", "false", "no", "disabled"})

_ENV_KILL_SWITCH = "CODE_GRAPH_LSP"
_ENV_TIMEOUT = "LSP_RESOLVE_TIMEOUT_SECONDS"
_ENV_TOTAL_BUDGET = "LSP_RESOLVE_TOTAL_BUDGET_SECONDS"
_ENV_MAX_SITES = "LSP_RESOLVE_MAX_SITES_PER_FILE"

_DEFAULT_TIMEOUT_SECONDS = 5.0

# Repo-wide ceiling for one graph build's type-resolution pass. WHY THIS EXISTS,
# measured 2026-08-03 on a ~300-file Python repo: a per-file timeout alone
# bounds each file but NOT the build, so a cold pass cost 5s × every file — many
# minutes — for zero benefit. The build is on the critical path of an agent's
# tool call; the layer must be a bounded accuracy bonus, never a stall.
# 10s, not 30: the graph build sits inside an agent's `code_graph` tool call
# (measured ~6.4s average in prod), so the FIRST call on a cold clone is the
# number a human feels. Results are cached per file-content hash and misses are
# cached too, so a truncated pass RESUMES on the next build rather than redoing
# work — measured on a 203-file repo, a 10s budget converges to zero added cost
# within a handful of builds, while a single 30s stall is paid up front. Raise
# it via the env var for a batch/offline index build where nobody is waiting.
_DEFAULT_TOTAL_BUDGET_SECONDS = 10.0

# Per-file site cap. Real files carry thousands of low-confidence call sites
# (3,365 in one measured module); resolving all of them cannot fit any sane
# budget, and the marginal site is worth far less than the first hundred.
# Sites arrive lowest-confidence-first, so the cap keeps the MOST ambiguous
# ones — exactly where type resolution pays.
_DEFAULT_MAX_SITES_PER_FILE = 400

# Grace added to a file's cooperative deadline before the hard thread-level
# backstop fires. The worker checks the deadline BETWEEN gotos, so the backstop
# only matters if a SINGLE goto hangs.
_HARD_TIMEOUT_GRACE_SECONDS = 2.0

#: Cache file, sibling to call_graph's ``.mc_codegraph_cache.json``.
CACHE_FILENAME = ".mc_codegraph_lsp_cache.json"


def _repo_cache_file(repo_root, filename):
    """Out-of-repo cache path, shared with call_graph.repo_cache_file (StateFrame 10-05)."""
    from .call_graph import repo_cache_file

    return repo_cache_file(repo_root, filename)
_CACHE_VERSION = 2

#: jedi ``Name.type`` values that represent a real callable definition. Anything
#: else (``statement``, ``param``, ``instance``, ``module``, ...) means jedi
#: landed somewhere that is not a def — not a usable call target, so we fail
#: open to name matching rather than assert a wrong edge.
_CALLABLE_TYPES = frozenset({"function", "class"})

#: Only these extensions are attempted. See the TS/JS note in the module docstring.
_SUPPORTED_SUFFIXES = frozenset({".py", ".pyi"})


def _timeout_seconds() -> float:
    """Per-file wall-clock budget. Malformed env values fall back to the default."""
    raw = os.getenv(_ENV_TIMEOUT)
    if not raw:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.debug("lsp_resolve: bad %s=%r — using default", _ENV_TIMEOUT, raw)
        return _DEFAULT_TIMEOUT_SECONDS
    return val if val > 0 else _DEFAULT_TIMEOUT_SECONDS


def _positive_env(name: str, default: float) -> float:
    """Read a positive numeric env var. Anything unusable falls back to `default`."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.debug("lsp_resolve: bad %s=%r — using default", name, raw)
        return default
    return val if val > 0 else default


def total_budget_seconds() -> float:
    return _positive_env(_ENV_TOTAL_BUDGET, _DEFAULT_TOTAL_BUDGET_SECONDS)


def max_sites_per_file() -> int:
    return int(_positive_env(_ENV_MAX_SITES, _DEFAULT_MAX_SITES_PER_FILE))


def is_enabled() -> bool:
    """True when the layer may run: not kill-switched AND the backend imported."""
    if (os.getenv(_ENV_KILL_SWITCH) or "").strip().lower() in _OFF_VALUES:
        return False
    return _JEDI_AVAILABLE


def backend_name() -> str:
    """Human-readable backend id for telemetry/report surfaces."""
    if not _JEDI_AVAILABLE:
        return "none"
    return f"jedi-{getattr(jedi, '__version__', 'unknown')}"


def _extractor_version() -> str:
    return f"lsp-resolve/2/{backend_name()}"


# ── Result shape ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ResolvedTarget:
    """A definitive definition site for one call.

    ``file`` is repo-RELATIVE (posix separators) so it joins directly against
    ``SymbolDef.file`` in the call graph. A target outside the repo (stdlib,
    site-packages) is never returned — see ``_to_target``.
    """
    file: str
    lineno: int
    name: str
    kind: str  # "function" | "class"

    def to_dict(self) -> dict:
        return {"file": self.file, "lineno": self.lineno, "name": self.name, "kind": self.kind}

    @classmethod
    def from_dict(cls, d: dict) -> Optional["ResolvedTarget"]:
        try:
            return cls(
                file=str(d["file"]),
                lineno=int(d["lineno"]),
                name=str(d["name"]),
                kind=str(d.get("kind", "function")),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class CallSiteRef:
    """Where to point the type engine, and what we expect to find there.

    ``line``/``col`` are the position of the CALLEE NAME token (1-based line,
    0-based column) — not the start of the call expression. For ``obj.method()``
    that is the ``method`` token, which is what a goto-definition wants.
    """
    line: int
    col: int
    name: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.line, self.col)


# ── Cache ────────────────────────────────────────────────────────────────────

def _cache_key(ref: CallSiteRef) -> str:
    return f"{ref.line}:{ref.col}"


def load_cache(repo_root: Path) -> dict:
    """Load the persisted per-file LSP cache. Never raises; {} on any problem."""
    path = _repo_cache_file(repo_root, CACHE_FILENAME)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, RecursionError, UnicodeError, ValueError):
        return {}
    if (not isinstance(raw, dict) or raw.get("version") != _CACHE_VERSION
            or raw.get("extractor_version") != _extractor_version()):
        return {}
    files = raw.get("files")
    return files if isinstance(files, dict) else {}


def save_cache(repo_root: Path, files: dict) -> None:
    """Persist the per-file LSP cache. Never raises — the cache is an optimization."""
    path = _repo_cache_file(repo_root, CACHE_FILENAME)
    try:
        from .call_graph import _atomic_write_text
        _atomic_write_text(path, json.dumps({
            "version": _CACHE_VERSION,
            "extractor_version": _extractor_version(),
            "files": files,
        }))
    except (ImportError, OSError, RecursionError, TypeError, ValueError) as exc:
        logger.debug("lsp_resolve: could not persist cache to %s: %s", path, exc)


def _decode_cached_sites(entry: object) -> Optional[dict[tuple[int, int], Optional[ResolvedTarget]]]:
    """Decode one cache entry's site map. None when the entry is unusable."""
    if not isinstance(entry, dict):
        return None
    sites = entry.get("sites")
    if not isinstance(sites, dict):
        return None
    out: dict[tuple[int, int], Optional[ResolvedTarget]] = {}
    for key, val in sites.items():
        if type(key) is not str:
            return None
        try:
            line_s, separator, col_s = key.partition(":")
            if not separator or ":" in col_s:
                return None
            pos = (int(line_s), int(col_s))
        except (TypeError, ValueError):
            return None  # corrupt entry — rebuild rather than half-trust it
        if pos[0] < 1 or pos[1] < 0 or pos in out:
            return None
        if val is None:
            out[pos] = None
            continue
        if (not isinstance(val, dict) or set(val) != {"file", "lineno", "name", "kind"}
                or type(val.get("file")) is not str or not val.get("file")
                or type(val.get("name")) is not str or not val.get("name")
                or type(val.get("kind")) is not str or val["kind"] not in _CALLABLE_TYPES
                or type(val.get("lineno")) is not int or val["lineno"] < 1):
            return None
        target = ResolvedTarget.from_dict(val)
        if target is None:
            return None
        out[pos] = target
    return out


def encode_sites(results: dict[tuple[int, int], Optional[ResolvedTarget]]) -> dict:
    """Serialize a resolve_file() result for the cache (misses are cached too).

    A MISS is cached deliberately: re-asking jedi about a call it could not
    resolve costs the same as asking about one it could, and the answer is
    equally stable for a given file content hash.
    """
    return {
        f"{line}:{col}": (target.to_dict() if target is not None else None)
        for (line, col), target in results.items()
    }


# ── Resolution ───────────────────────────────────────────────────────────────

#: One `jedi.Project` per repo root, reused for every file in a build. Project
#: construction resolves an Environment (which can spawn/locate an interpreter),
#: so rebuilding it per file is pure overhead on the hot path.
_PROJECTS: dict[str, object] = {}


def _project_for(repo_root: Path):
    key = str(repo_root)
    proj = _PROJECTS.get(key)
    if proj is None:
        proj = jedi.Project(key)  # type: ignore[union-attr]
        # Bounded: a long-lived runner only ever sees a handful of repo roots,
        # but never let a pathological caller grow this without limit.
        if len(_PROJECTS) > 16:
            _PROJECTS.clear()
        _PROJECTS[key] = proj
    return proj


def _to_target(name_obj, repo_root: Path) -> Optional[ResolvedTarget]:
    """Convert a jedi Name into a repo-relative ResolvedTarget, or None."""
    try:
        module_path = getattr(name_obj, "module_path", None)
        lineno = getattr(name_obj, "line", None)
        kind = getattr(name_obj, "type", None)
        short = getattr(name_obj, "name", None)
    except Exception:
        return None
    if not module_path or not lineno or not short:
        return None
    if kind not in _CALLABLE_TYPES:
        # jedi landed on a variable/param/module — not a call target we can trust.
        return None
    try:
        rel = Path(module_path).resolve().relative_to(Path(repo_root).resolve())
    except (ValueError, OSError):
        # Outside the repo (stdlib / site-packages) — the graph only indexes repo
        # files, so there is nothing to point an edge at. Fail open.
        return None
    return ResolvedTarget(file=rel.as_posix(), lineno=int(lineno), name=str(short), kind=str(kind))


def _resolve_sites_blocking(
    repo_root: Path,
    abs_path: Path,
    source: str,
    sites: list[CallSiteRef],
    deadline: Optional[float] = None,
    results: Optional[dict[tuple[int, int], Optional[ResolvedTarget]]] = None,
) -> dict[tuple[int, int], Optional[ResolvedTarget]]:
    """The jedi work for one file, with a COOPERATIVE deadline.

    `results` is written IN PLACE as sites are resolved, and the loop returns
    early once `deadline` passes. Both matter, and the reason is not
    hypothetical (measured 2026-08-03):

      * a thread killed only by the outer `future.result(timeout)` cannot
        actually be killed — it keeps running, and because jedi is CPU-bound
        pure Python it keeps HOLDING THE GIL. One timed-out file then starved
        every file after it, so a single slow module cascaded into a whole build
        of timeouts with zero resolutions. Checking the deadline in the loop
        means no worker ever outlives its budget, so there is nothing to leak.
      * returning early with what it has turns "5 seconds spent, 0 answers" into
        "5 seconds spent, N answers", and the shared dict means even the hard
        backstop can salvage partial work.
    """
    if results is None:
        results = {}
    project = _project_for(repo_root)
    script = jedi.Script(code=source, path=str(abs_path), project=project)  # type: ignore[union-attr]
    for ref in sites:
        if deadline is not None and time.monotonic() >= deadline:
            break
        target: Optional[ResolvedTarget] = None
        try:
            names = script.goto(ref.line, ref.col, follow_imports=True)
        except Exception as exc:  # jedi raises on odd positions / recursion limits
            logger.debug(
                "lsp_resolve: goto failed at %s:%d:%d (%s): %s",
                abs_path, ref.line, ref.col, ref.name, exc,
            )
            names = []
        # AMBIGUITY IS A MISS. jedi returning several definitions means it could
        # not narrow the type either — asserting one of them at the top
        # confidence tier would be worse than the name-matching over-report,
        # which at least reports ALL candidates honestly.
        if len(names) == 1:
            target = _to_target(names[0], repo_root)
        results[ref.key] = target
    return results


def _snapshot(results: dict) -> dict:
    """Copy a dict a worker thread may still be writing to.

    Only reached on the hard-backstop path, where the worker is by definition
    still alive. A plain `dict(d)` can raise "dictionary changed size during
    iteration" there — losing the partial results we came to salvage — so retry
    briefly, then give up cleanly rather than propagate.
    """
    for _ in range(3):
        try:
            return dict(results)
        except RuntimeError:
            time.sleep(0.01)
    return {}


def resolve_file(
    repo_root: str | Path,
    relpath: str,
    sites: Iterable[CallSiteRef],
    *,
    timeout: Optional[float] = None,
) -> dict[tuple[int, int], ResolvedTarget]:
    """Batch-resolve every call site in one file. NEVER raises.

    Returns only the HITS, keyed by ``(line, col)``. An absent key means "no
    definitive answer — keep the existing resolution".

    One `jedi.Script` is constructed per file and reused for every site, which
    is the whole point of the batch form: per-call setup dominates otherwise.
    """
    full = resolve_file_full(repo_root, relpath, sites, timeout=timeout)
    return {pos: t for pos, t in full.items() if t is not None}


def resolve_file_full(
    repo_root: str | Path,
    relpath: str,
    sites: Iterable[CallSiteRef],
    *,
    timeout: Optional[float] = None,
) -> dict[tuple[int, int], Optional[ResolvedTarget]]:
    """Like :func:`resolve_file` but keeps explicit MISSes (value ``None``).

    Callers that persist results want the misses too — see :func:`encode_sites`.
    A site that was never REACHED (budget exhausted) is simply absent from the
    result: absent means "we did not get to ask", ``None`` means "the type
    engine looked and had no definitive answer". Only the latter is durable and
    therefore cacheable.
    """
    site_list = [s for s in sites if isinstance(s, CallSiteRef)][:max_sites_per_file()]
    if not site_list or not is_enabled():
        return {}
    if Path(relpath).suffix not in _SUPPORTED_SUFFIXES:
        return {}

    root = Path(repo_root)
    abs_path = root / relpath
    try:
        source = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("lsp_resolve: unreadable %s: %s", abs_path, exc)
        return {}

    budget = _timeout_seconds() if timeout is None else timeout
    started = time.monotonic()
    deadline = started + budget
    # Written in place by the worker so a hard-backstop timeout still salvages
    # everything resolved before the hang.
    results: dict[tuple[int, int], Optional[ResolvedTarget]] = {}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lsp-resolve")
    try:
        future = executor.submit(
            _resolve_sites_blocking, root, abs_path, source, site_list, deadline, results)
        try:
            future.result(timeout=budget + _HARD_TIMEOUT_GRACE_SECONDS)
        except _FutureTimeout:
            # The cooperative deadline did NOT stop it, so a single goto hung.
            # Its own telemetry line, at WARNING: a silently-skipped file is how
            # an accuracy layer rots into a no-op nobody notices.
            logger.warning(
                "lsp_resolve: HARD TIMEOUT after %.1fs in %s — a single lookup hung; "
                "salvaged %d/%d site(s) (backend=%s), failing open for the rest",
                budget + _HARD_TIMEOUT_GRACE_SECONDS, relpath,
                len(results), len(site_list), backend_name(),
            )
            future.cancel()
            return _snapshot(results)
        except Exception as exc:
            logger.debug("lsp_resolve: resolution failed for %s: %s", relpath, exc)
            return _snapshot(results)
    except Exception as exc:  # thread pool exhaustion etc. — never fatal
        logger.debug("lsp_resolve: could not schedule resolution for %s: %s", relpath, exc)
        return {}
    finally:
        executor.shutdown(wait=False)

    if len(results) < len(site_list):
        # Cooperative deadline hit — bounded, expected on very large files, and
        # still productive (partial answers are kept). One line, not per site.
        logger.info(
            "lsp_resolve: budget reached in %s — resolved %d of %d site(s) in %.1fs",
            relpath, len(results), len(site_list), time.monotonic() - started,
        )
    logger.debug(
        "lsp_resolve: %s — %d/%d site(s) resolved in %.0fms",
        relpath, sum(1 for v in results.values() if v is not None),
        len(site_list), (time.monotonic() - started) * 1000,
    )
    return results


def resolve_call(
    repo_root: str | Path,
    relpath: str,
    line: int,
    col: int,
    callee_name: str,
    *,
    timeout: Optional[float] = None,
) -> Optional[ResolvedTarget]:
    """Resolve a single call site. Convenience wrapper over :func:`resolve_file`.

    Prefer the batch form inside a loop — this constructs a fresh `jedi.Script`
    per call, which is exactly the per-call cost the batch form exists to avoid.
    """
    ref = CallSiteRef(line=int(line), col=int(col), name=str(callee_name))
    return resolve_file(repo_root, relpath, [ref], timeout=timeout).get(ref.key)


def resolve_repo(
    repo_root: str | Path,
    sites_by_file: dict[str, list[CallSiteRef]],
    *,
    file_hashes: Optional[dict[str, str]] = None,
    cache: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> tuple[dict[str, dict[tuple[int, int], ResolvedTarget]], dict]:
    """Resolve many files, reusing a content-hash cache. NEVER raises.

    Returns ``(hits_by_file, new_cache)``. A file whose content hash matches its
    cached entry is served entirely from cache — a warm build pays nothing.

    ``file_hashes``/``cache`` are optional: without them every file is resolved
    fresh and the returned cache is still well-formed (so a caller can start
    persisting it at any time).
    """
    hits: dict[str, dict[tuple[int, int], ResolvedTarget]] = {}
    new_cache: dict = {}
    if not is_enabled():
        return hits, new_cache

    cache = cache or {}
    file_hashes = file_hashes or {}
    extractor_version = _extractor_version()
    per_file = _timeout_seconds() if timeout is None else timeout
    total_deadline = time.monotonic() + total_budget_seconds()

    skipped_files = 0
    for relpath, sites in sites_by_file.items():
        if not sites:
            continue
        digest = file_hashes.get(relpath)
        entry = cache.get(relpath) if digest else None
        cached = (
            _decode_cached_sites(entry)
            if (isinstance(entry, dict) and entry.get("hash") == digest
                    and entry.get("extractor_version") == extractor_version)
            else None
        ) or {}

        # Serve everything the cache already knows, and ask only about the rest.
        # A cache entry can legitimately UNDER-cover: the low-confidence site set
        # depends on repo-wide state, so an unrelated file's edit can widen it
        # without changing this file's hash — and a budget-truncated pass leaves
        # a partial entry by design. Re-asking only the gap is what makes the
        # truncated case converge across builds instead of redoing the same
        # prefix forever.
        results = {s.key: cached[s.key] for s in sites if s.key in cached}
        missing = [s for s in sites if s.key not in cached]

        if missing:
            remaining = total_deadline - time.monotonic()
            if remaining <= 0:
                # Budget exhausted. Do NOT `continue` — falling through is what
                # keeps this file's EXISTING cached answers in the rewritten
                # cache. Skipping the write would drop them, and the pass would
                # re-resolve the same prefix on every future build instead of
                # converging.
                skipped_files += 1
            else:
                results.update(resolve_file_full(
                    repo_root, relpath, missing, timeout=min(per_file, remaining)))

        # Carry the entry forward whenever the content hash still matches, even
        # if THIS build resolved nothing new: the returned cache REPLACES the
        # old one, so anything not re-emitted here is forgotten. (Caught by
        # test_exhausted_budget_preserves_already_cached_answers — the first
        # version of this line said `if digest and results`, which silently
        # discarded a whole file's cached answers whenever the current build's
        # site set did not overlap them.)
        if digest and (results or cached):
            merged = dict(cached)
            merged.update(results)
            new_cache[relpath] = {
                "hash": digest,
                "extractor_version": extractor_version,
                "sites": encode_sites(merged),
            }
        file_hits = {pos: t for pos, t in results.items() if t is not None}
        if file_hits:
            hits[relpath] = file_hits

    if skipped_files:
        # ONE honest line — never per file. The pass is a bounded bonus; saying
        # so out loud is what keeps a silently-degraded layer from looking
        # healthy (rule #158).
        logger.info(
            "lsp_resolve: repo budget of %.0fs exhausted — %d file(s) left to "
            "name matching this build (they resolve on the next one, cache-forward)",
            total_budget_seconds(), skipped_files,
        )
    return hits, new_cache

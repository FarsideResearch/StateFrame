"""Generation-overlap work counts and scheduler parity at the phase boundary."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
import time
from collections import Counter
from contextlib import contextmanager

import pytest
from kernel_turn_contracts import (
    Artifact,
    Basis,
    Context,
    Dep,
    Head,
    Key,
    Ref,
    Unavailable,
)

SWITCH = "STATEFRAME_TURN_GENERATION_OVERLAP"
SELECTOR = Key("prefetch", b"stable-panel")
SOURCES = Key("query-membership", b"source-inputs")
RECEIPTS = Key("receipt", b"prior-turns")
FOREGROUND_SECONDS = 0.100
WAIT_SECONDS = 5


class _Legacy:
    """The base computes demand on the caller, with no optional worker."""

    def offer(self, at, key, work):
        pass

    def take(self, at, key):
        return None

    def cancel(self):
        pass


def _open(tmp_path, facts, publication, *, attempt="attempt"):
    if importlib.util.find_spec("kernel_turn_overlap") is None:
        return _Legacy()
    from kernel_turn_overlap import open_port

    return open_port(
        Context("principal", "run", attempt, tmp_path, "profile"),
        facts=facts,
        publication=publication,
    )


class _Publication:
    """Run-local immutable staging with independent, thread-affine pins."""

    def __init__(self):
        self.data: dict[Ref, bytes] = {}
        self.pins: Counter[Ref] = Counter()
        self.lock = threading.Lock()
        self.local = threading.local()
        self.stages: Counter[str] = Counter()
        self.fail_load = False

    @contextmanager
    def transaction(self):
        pins: set[Ref] = set()
        assert not hasattr(self.local, "pins")
        self.local.pins = pins
        try:
            yield
        finally:
            with self.lock:
                for ref in pins:
                    self.pins[ref] -= 1
                    if not self.pins[ref]:
                        del self.pins[ref]
                        del self.data[ref]
            del self.local.pins

    def _pin(self, ref):
        pins = self.local.pins
        if ref not in pins:
            pins.add(ref)
            self.pins[ref] += 1

    def stage(self, codec, data):
        ref = Ref(codec, hashlib.sha256(data).hexdigest(), len(data))
        with self.lock:
            self._pin(ref)
            self.data[ref] = data
            self.stages["foreground" if threading.current_thread() is threading.main_thread()
                        else "worker"] += 1
        return ref

    def load(self, ref):
        with self.lock:
            if self.fail_load or ref not in self.data:
                raise Unavailable("object missing")
            self._pin(ref)
            return self.data[ref]


class _Facts:
    def __init__(self):
        self.versions = {SOURCES: "sources", RECEIPTS: "receipts"}
        self.comparisons = 0
        self.unavailable = False

    def current(self, at, deps):
        assert threading.current_thread() is threading.main_thread()
        if self.unavailable:
            raise Unavailable("dependency state missing")
        for dep in deps:
            self.comparisons += 1
            if self.versions.get(dep.key) != dep.revision:
                return False
        return True


def _basis(step):
    return Basis("parent", "live", 7, Head(step, f"head-{step}"), step, "authority")


def _ready(port, key=SELECTOR):
    # Control the scripted generation window with the worker's publication
    # barrier. This wait is outside the timed response-to-demand interval.
    if isinstance(port, _Legacy):
        return False
    worker = port._worker
    with worker._changed:
        return worker._changed.wait_for(
            lambda: ((ticket := worker._active) is not None and ticket.key == key
                     and ticket.result is not None and not ticket.cancelled),
            timeout=WAIT_SECONDS,
        )


def _wire(publication, artifact, exchange, verdict="checks-current"):
    return json.dumps(
        {"latest": exchange, "panel": publication.load(artifact.value).hex(), "verdict": verdict},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()


@pytest.mark.parametrize("turned_on", [False, True], ids=["off", "on"])
def test_stale_and_paused_worker_add_no_join(tmp_path, monkeypatch, turned_on):
    if turned_on:
        monkeypatch.setenv(SWITCH, "on")
    else:
        monkeypatch.delenv(SWITCH, raising=False)
    sources = tmp_path / "sources"
    sources.mkdir()
    paths = []
    for index in range(5_000):
        path = sources / f"module_{index}.py"
        path.write_text(f"value = {index}\n", encoding="utf-8")
        paths.append(path)
    history = tmp_path / "receipts.jsonl"
    history.write_text(
        "".join(json.dumps({"step": step, "observation": "read ✓"}) + "\n"
                for step in range(200)),
        encoding="utf-8",
    )

    facts = _Facts()
    publication = _Publication()
    deps = tuple(Dep(key, revision) for key, revision in facts.versions.items())
    open_started = time.perf_counter()
    port = _open(tmp_path, facts, publication)
    open_elapsed = time.perf_counter() - open_started
    counts: dict[str, Counter[str]] = {"foreground": Counter(), "worker": Counter()}

    def project():
        lane = "foreground" if threading.current_thread() is threading.main_thread() else "worker"
        work = counts[lane]
        work["calls"] += 1
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.read_bytes())
            work["source_reads"] += 1
        with history.open("rb") as receipts:
            for receipt in receipts:
                digest.update(receipt)
                work["receipt_reads"] += 1
        return Artifact(publication.stage("turn/panel/1", digest.digest()), deps)

    with publication.transaction():
        expected = _wire(publication, project(), "real response")
    counts["foreground"].clear()
    publication.stages.clear()

    port.offer(_basis(200), SELECTOR, project)
    if turned_on:
        _ready(port)
    try:
        with publication.transaction():
            started = time.perf_counter()
            actual = port.take(_basis(201), SELECTOR)
            wire = _wire(publication, actual if actual is not None else project(), "real response")
            elapsed = time.perf_counter() - started
        assert wire == expected
        print(
            f"overlap={'on' if turned_on else 'off'} open_port={open_elapsed:.6f}s "
            f"demand={elapsed:.6f}s foreground={dict(counts['foreground'])} "
            f"worker={dict(counts['worker'])} comparisons={facts.comparisons}"
        )
        full = {"calls": 1, "source_reads": 5_000, "receipt_reads": 200}
        assert dict(counts["foreground"]) == ({} if turned_on else full)
        assert dict(counts["worker"]) == (full if turned_on else {})
        assert facts.comparisons == (2 if turned_on else 0)
        assert publication.stages == {("worker" if turned_on else "foreground"): 1}
        if turned_on:
            assert elapsed < FOREGROUND_SECONDS
    finally:
        port.cancel()


@pytest.fixture
def overlap(tmp_path, monkeypatch):
    pytest.importorskip("kernel_turn_overlap")
    monkeypatch.setenv(SWITCH, "on")
    facts, publication = _Facts(), _Publication()
    port = _open(tmp_path, facts, publication)
    try:
        yield port, facts, publication
    finally:
        port.cancel()


def _idle(port):
    worker = port._worker
    with worker._changed:
        return worker._changed.wait_for(lambda: not worker._running, timeout=WAIT_SECONDS)


def _stage(publication, data=b"panel", deps=()):
    return Artifact(publication.stage("turn/panel/1", data), deps)


def test_paused_producer_is_never_joined_and_late_result_is_discarded(overlap, tmp_path):
    port, facts, publication = overlap
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def project():
        started.set()
        assert release.wait(WAIT_SECONDS)
        artifact = _stage(publication, b"obsolete generation")
        finished.set()
        return artifact

    port.offer(_basis(200), SELECTOR, project)
    try:
        assert started.wait(WAIT_SECONDS)
        with publication.transaction():
            before = time.perf_counter()
            assert port.take(_basis(201), SELECTOR) is None
            port.cancel()
            elapsed = time.perf_counter() - before
            demand_wire = _wire(publication, _stage(publication), "real response")
        print(f"paused_poll_and_cancel={elapsed:.6f}s comparisons={facts.comparisons}")
        assert elapsed < FOREGROUND_SECONDS
        assert not finished.is_set()
        assert facts.comparisons == 0

        fresh_publication = _Publication()
        fresh = _open(tmp_path, facts, fresh_publication, attempt="restored")
        with fresh_publication.transaction():
            assert fresh.take(_basis(201), SELECTOR) is None
            assert _wire(fresh_publication, _stage(fresh_publication), "real response") == demand_wire
    finally:
        port.cancel()
        release.set()
    assert _idle(port)
    assert finished.is_set()
    assert publication.data == {}
    with publication.transaction():
        assert port.take(_basis(201), SELECTOR) is None
        assert _wire(publication, _stage(publication), "real response") == demand_wire


@pytest.mark.parametrize("change", ["stale-receipt", "unavailable-deps", "missing-ref", "cancel"])
def test_completed_speculation_requires_current_evidence(overlap, monkeypatch, change):
    port, facts, publication = overlap
    dep = Dep(RECEIPTS, facts.versions[RECEIPTS])
    port.offer(_basis(200), SELECTOR, lambda: _stage(publication, b"old verdict", (dep,)))
    assert _ready(port)
    if change == "stale-receipt":
        facts.versions[RECEIPTS] = "amended"
    elif change == "unavailable-deps":
        facts.unavailable = True
    elif change == "missing-ref":
        publication.fail_load = True
    else:
        def cancel_while_checking(at, deps):
            port.cancel()
            return True
        monkeypatch.setattr(facts, "current", cancel_while_checking)

    with publication.transaction():
        before = time.perf_counter()
        assert port.take(_basis(201), SELECTOR) is None
        elapsed = time.perf_counter() - before
        publication.fail_load = False
        wire = _wire(publication, _stage(publication), "tool reply", verdict="checks-stale")
    print(f"{change} rejection={elapsed:.6f}s comparisons={facts.comparisons}")
    assert elapsed < FOREGROUND_SECONDS
    assert wire == b'{"latest":"tool reply","panel":"70616e656c","verdict":"checks-stale"}'
    assert _idle(port)
    assert publication.data == {}


def test_queue_bounds_and_latest_offer_win(overlap):
    port, _, publication = overlap
    started, release = threading.Event(), threading.Event()
    calls = []

    def project(label):
        calls.append(label)
        if label == "active":
            started.set()
            assert release.wait(WAIT_SECONDS)
        return _stage(publication, label.encode())

    second, third, overflow = (Key("prefetch", name) for name in (b"second", b"third", b"overflow"))
    at = _basis(200)
    port.offer(at, SELECTOR, lambda: project("active"))
    try:
        assert started.wait(WAIT_SECONDS)
        port.offer(at, SELECTOR, lambda: project("duplicate active"))
        port.offer(at, second, lambda: project("old second"))
        port.offer(at, third, lambda: project("third"))
        port.offer(at, overflow, lambda: project("overflow"))
        port.offer(_basis(201), second, lambda: project("new second"))
        assert calls == ["active"]
        with publication.transaction():
            assert port.take(at, second) is None

        release.set()
        for key, expected in ((SELECTOR, b"active"), (second, b"new second"), (third, b"third")):
            assert _ready(port, key)
            with publication.transaction():
                artifact = port.take(_basis(201), key)
                assert artifact is not None
                assert publication.load(artifact.value) == expected
        assert _idle(port)
        assert calls == ["active", "new second", "third"]
        assert publication.data == {}
    finally:
        port.cancel()
        release.set()


def test_worker_error_and_oversize_do_not_publish(overlap):
    port, _, publication = overlap

    def fail():
        raise OSError("optional source unavailable")

    for work in (fail, lambda: _stage(publication, b"x" * (8 * 1024 * 1024 + 1))):
        port.offer(_basis(200), SELECTOR, work)
        assert _idle(port)
        with publication.transaction():
            assert port.take(_basis(201), SELECTOR) is None
        assert publication.data == {}

    port.offer(_basis(202), SELECTOR, lambda: _stage(publication))
    assert _ready(port)
    with publication.transaction():
        artifact = port.take(_basis(203), SELECTOR)
        assert artifact is not None
        assert _idle(port)
        assert publication.load(artifact.value) == b"panel"

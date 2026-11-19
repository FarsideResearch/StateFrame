"""Exact tokenizer-count reuse across identical sends."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from kernel_request_diet import RequestTokenMeter, _tokenization_content
from tape_server import TapeServer


def _long_history() -> dict[str, object]:
    return {
        "model": "tok-cache-long-history-fixture",
        "messages": [{"role": "user", "content": ""} for _ in range(5000)],
        "tools": [],
    }


def _body(*, model: str = "tok-cache-small-fixture", text: str = "same prompt") -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "tools": [],
    }


def _meter(endpoint: str) -> RequestTokenMeter:
    return RequestTokenMeter(
        endpoint=endpoint,
        route_allow_remote=False,
        template_overhead_tokens=0,
        generation_endpoint=endpoint.removesuffix("/tokenize") + "/chat/completions",
    )


def _token_server() -> TapeServer:
    server = TapeServer({"turns": [{"message": {"content": "unused"}}]})
    # Keep the fixture response compact; request-body size and call counts are
    # the measured work, not the fake tokenizer's serialized token-ID volume.
    server._token_count = lambda _text: 4
    return server


def test_identical_long_history_send_reuses_exact_count(monkeypatch):
    monkeypatch.setenv("STATEFRAME_EXACT_TOKEN_COUNT_CACHE", "on")
    body = _long_history()
    with _token_server() as server:
        endpoint = server.url + "/tokenize"
        cache_before = sys.modules.get("stateframe.token_count_cache")
        stats_before = cache_before.cache_info() if cache_before is not None else None
        started = time.perf_counter()
        first_count, first_method = _meter(endpoint).measure(body)
        calls_after_first_send = server.tokenize_request_count
        cache_module = sys.modules.get("stateframe.token_count_cache")
        first_stats = cache_module.cache_info() if cache_module is not None else None
        second_count, second_method = _meter(endpoint).measure(body)
        elapsed = time.perf_counter() - started

        assert first_method == second_method == "tokenize"
        assert first_count == second_count > 0
        assert calls_after_first_send == 1
        assert server.tokenize_request_count == 1, (
            "second identical send made another /tokenize POST; "
            f"calls={server.tokenize_request_count}, expected=1"
        )
        from stateframe import token_count_cache

        assert first_stats is not None, "enabled cache was not initialized on the first exact count"
        after = token_count_cache.cache_info()
        if stats_before is None:
            assert first_stats["lookups"] == 1  # shared key comparisons
            assert first_stats["misses"] == 1
            assert first_stats["computations"] == 1
        else:
            assert first_stats["lookups"] - stats_before["lookups"] == 1
            assert first_stats["misses"] - stats_before["misses"] == 1
            assert first_stats["computations"] - stats_before["computations"] == 1
        assert after["lookups"] - first_stats["lookups"] == 1
        assert after["hits"] - first_stats["hits"] == 1
        assert after["misses"] - first_stats["misses"] == 0
        assert after["computations"] - first_stats["computations"] == 0
        assert len(server.tokenize_request_log) == 1  # one HTTP body read by the endpoint
        assert server.tokenize_request_log[0]["content_bytes"] == len(
            _tokenization_content(body).encode("utf-8"))
        assert elapsed < 10.0, f"two exact counts took {elapsed:.3f}s (budget 10.000s)"


def test_shared_cache_separates_model_and_exact_text(monkeypatch):
    monkeypatch.setenv("STATEFRAME_EXACT_TOKEN_COUNT_CACHE", "on")
    with _token_server() as server:
        endpoint = server.url + "/tokenize"
        body = _body()
        first = _meter(endpoint).measure(body)
        repeated = _meter(endpoint).measure(body)
        other_model = _meter(endpoint).measure(_body(model="another-model"))
        edited_text = _meter(endpoint).measure(_body(text="same prompt plus one word"))

        assert first[1] == repeated[1] == other_model[1] == edited_text[1] == "tokenize"
        assert first[0] == repeated[0]
        assert server.tokenize_request_count == 3


def test_pinned_http_count_survives_process_launches(tmp_path, monkeypatch):
    monkeypatch.delenv("STATEFRAME_EXACT_TOKEN_COUNT_CACHE", raising=False)
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path))
    source = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(source), str(source / "stateframe/engine"))))
    script = """
import json
import sys
from kernel_provider_dialect import capabilities_for
from kernel_request_diet import RequestTokenMeter

endpoint, pin, text = sys.argv[1:]
assert capabilities_for(endpoint, "", require_slots=False).continuation
meter = RequestTokenMeter(endpoint=endpoint, template_overhead_tokens=17)
meter.provider_identity = json.dumps({"tokenizer": pin})
body = {"model": "fixed-model", "messages": [{"role": "user", "content": text}]}
print(json.dumps(meter.measure(body)))
"""
    with _token_server() as server:
        endpoint = server.url + "/tokenize"

        def measure(pin, text="same prompt"):
            result = subprocess.run(
                [sys.executable, "-c", script, endpoint, pin, text],
                check=True, capture_output=True, text=True,
            )
            return json.loads(result.stdout)

        assert measure("a" * 64) == [21, "tokenize"]
        assert server.tokenize_request_count == 1
        assert measure("a" * 64) == [21, "tokenize"]
        assert server.tokenize_request_count == 1
        assert measure("b" * 64) == [21, "tokenize"]
        assert measure("a" * 64, "changed prompt") == [21, "tokenize"]
        assert server.tokenize_request_count == 3
        assert measure("mutable-alias") == measure("mutable-alias") == [21, "tokenize"]
        assert server.tokenize_request_count == 5


def test_builtin_schema_counts_are_exact_and_custom_admission_remains_exact(monkeypatch):
    import kernel_request_diet
    from kernel_agent_schemas import BUILTIN_SCHEMAS, measure_schemas, schema
    from kernel_custom_tools import CustomToolError

    monkeypatch.delenv("STATEFRAME_EXACT_TOKEN_COUNT_CACHE", raising=False)
    names = [item["function"]["name"] for item in BUILTIN_SCHEMAS]
    expected = {name: n for n, name in enumerate(["shell", *names, "custom"], 1)}
    order = []

    def tokenize(content, _url, **_kwargs):
        name = json.loads(content)["tools"][0]["function"]["name"]
        order.append(name)
        return [0] * expected[name]

    monkeypatch.setattr(kernel_request_diet, "_http_tokenize", tokenize)
    monkeypatch.setattr(RequestTokenMeter, "_tokenizer_capability_proven", lambda self: True)
    custom = [schema("custom", "Custom tool", {}, [])]
    endpoint = "http://127.0.0.1:9/tokenize"
    report = measure_schemas(custom, endpoint=endpoint)
    assert report == {
        ("custom_total" if name == "custom" else name): {"tokens": count, "method": "tokenize"}
        for name, count in expected.items()
    }
    assert order[0] == "shell"
    assert sorted(order) == sorted(expected)
    assert order[-1] == "custom"

    expected["custom"] = 1501
    with pytest.raises(CustomToolError, match="1501 tokens exceeds 1500"):
        measure_schemas(custom, endpoint=endpoint)


def test_prompt_count_overlaps_preparation_and_fitting_uses_the_result(monkeypatch):
    import kernel_request_diet as diet
    from kernel_request_budget import FREE_STAGE

    monkeypatch.delenv("STATEFRAME_EXACT_TOKEN_COUNT_CACHE", raising=False)
    started = threading.Event()
    prepared = threading.Event()
    calls = []
    read_spans = diet._read_spans_retained

    def tokenize(content, _url, **_kwargs):
        calls.append(content)
        started.set()
        assert prepared.wait(5)
        return [0] * 73

    def scan(messages):
        assert started.wait(5)
        prepared.set()
        return read_spans(messages)

    monkeypatch.setattr(diet, "_http_tokenize", tokenize)
    monkeypatch.setattr(diet, "_read_spans_retained", scan)
    monkeypatch.setattr(RequestTokenMeter, "_tokenizer_capability_proven", lambda self: True)
    meter = _meter("http://127.0.0.1:9/tokenize")
    body = _body()
    audit = diet.apply_diet(body, stage=FREE_STAGE, max_request_bytes=10_000,
                           token_budget=100, token_meter=meter)
    assert audit["request_tokens_before"] == audit["request_tokens_after"] == 73
    assert audit["measure_method"] == "tokenize"
    assert calls == [_tokenization_content(body)]


def test_subframe_exact_count_uses_shared_cache(monkeypatch):
    monkeypatch.setenv("STATEFRAME_EXACT_TOKEN_COUNT_CACHE", "on")
    body = _body()
    with _token_server() as server:
        endpoint = server.url + "/tokenize"
        first = _meter(endpoint).measure_subframe_cap(body, tokens_per_byte=0.25)
        repeated = _meter(endpoint).measure_subframe_cap(body, tokens_per_byte=0.25)

        assert first == repeated
        assert server.tokenize_request_count == 1
        assert len(server.tokenize_request_log) == 1


def test_cache_block_count_uses_shared_cache_after_full_measure(monkeypatch):
    monkeypatch.setenv("STATEFRAME_EXACT_TOKEN_COUNT_CACHE", "on")
    body = _body(model="tok-cache-block-count-fixture")
    with _token_server() as server:
        endpoint = server.url + "/tokenize"
        first_meter = _meter(endpoint)
        second_meter = _meter(endpoint)

        assert first_meter.measure(body)[1] == "tokenize"
        first_block_count = first_meter.count_text_exact("repeated cache block")
        assert second_meter.measure(body)[1] == "tokenize"
        second_block_count = second_meter.count_text_exact("repeated cache block")

        assert first_block_count == second_block_count == 4
        # One full projected request and one distinct block string; the second
        # meter reuses both exact counts without a new HTTP request.
        assert server.tokenize_request_count == 2
        assert len(server.tokenize_request_log) == 2


def test_failed_exact_count_is_not_cached():
    token_count_cache = pytest.importorskip("stateframe.token_count_cache")

    cache = token_count_cache.ExactTokenCountCache(max_entries=2)
    attempts = 0

    def fail_once() -> int:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary tokenizer failure")
        return 7

    with pytest.raises(RuntimeError, match="temporary tokenizer failure"):
        cache.get_or_compute("model-identity", "exact text", fail_once)
    assert cache.get_or_compute("model-identity", "exact text", fail_once) == 7
    assert attempts == 2
    assert cache.cache_info()["entries"] == 1


def test_concurrent_exact_count_requests_share_one_computation():
    token_count_cache = pytest.importorskip("stateframe.token_count_cache")

    cache = token_count_cache.ExactTokenCountCache(max_entries=4)
    workers = 8
    start = threading.Barrier(workers)
    release = threading.Event()
    compute_started = threading.Event()
    attempts = 0
    attempts_lock = threading.Lock()

    def compute() -> int:
        nonlocal attempts
        with attempts_lock:
            attempts += 1
        compute_started.set()
        assert release.wait(5), "shared computation was not released"
        return 11

    def request() -> int:
        start.wait(timeout=5)
        return cache.get_or_compute("model-identity", "exact text", compute)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [pool.submit(request) for _ in range(workers)]
        assert compute_started.wait(5), "no request started the shared computation"
        deadline = time.monotonic() + 5
        while cache.cache_info()["waits"] < workers - 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert cache.cache_info()["waits"] == workers - 1
        release.set()
        assert [result.result(timeout=5) for result in results] == [11] * workers

    stats = cache.cache_info()
    assert attempts == 1
    assert stats["lookups"] == workers
    assert stats["computations"] == 1
    assert stats["entries"] == 1


def test_exact_count_cache_keeps_a_bounded_lru():
    token_count_cache = pytest.importorskip("stateframe.token_count_cache")

    cache = token_count_cache.ExactTokenCountCache(max_entries=2)
    for text in ("first", "second", "third"):
        assert cache.get_or_compute("model", text, lambda: len(text)) == len(text)
    assert cache.cache_info()["entries"] == 2
    assert cache.get_or_compute("model", "second", lambda: 0) == len("second")
    assert cache.cache_info()["hits"] == 1
    assert cache.get_or_compute("model", "first", lambda: len("first")) == len("first")
    assert cache.cache_info()["entries"] == 2


def test_cache_is_not_imported_when_switch_is_off():
    env = dict(os.environ)
    env.pop("STATEFRAME_EXACT_TOKEN_COUNT_CACHE", None)
    source_root = Path(__file__).resolve().parents[1]
    import_paths = [str(source_root / "src"), str(source_root / "src" / "stateframe" / "engine")]
    if env.get("PYTHONPATH"):
        import_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(import_paths)
    script = """
import sys
import kernel_request_diet
from kernel_request_diet import RequestTokenMeter
kernel_request_diet._http_tokenize = lambda *args, **kwargs: [1]
meter = RequestTokenMeter(endpoint="http://127.0.0.1:9/tokenize", route_allow_remote=False)
body = {"model": "off-switch", "messages": [{"role": "user", "content": "off"}], "tools": []}
assert meter.measure(body)[1] == "tokenize"
assert "stateframe.token_count_cache" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_exact_count_cache_reaches_engine_only_on_request(monkeypatch):
    from stateframe.runner import resolve_opt_in_switches

    switch = "STATEFRAME_EXACT_TOKEN_COUNT_CACHE"
    monkeypatch.delenv(switch, raising=False)
    assert switch not in resolve_opt_in_switches()
    monkeypatch.setenv(switch, "on")
    assert resolve_opt_in_switches().get(switch) == "on"

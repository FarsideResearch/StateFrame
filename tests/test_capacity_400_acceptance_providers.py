# DECLARED RE-FREEZE 11-05 13:21 (release manager): freeze vLLM 16k refusal and 131k recovery.
"""CLI acceptance cases for provider-specific context-window HTTP 400 bodies."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from test_capacity_400_acceptance import Case, _assert_base_terminal, _run_case


pytestmark = pytest.mark.host_sandbox
FIXTURES = Path(__file__).parent / "fixtures" / "capacity_400"
PROVIDERS = ("openai", "llama_cpp", "vllm", "deepinfra", "anthropic_compatible")


def _assert_recovered(case: Case) -> None:
    assert len(case.requests) == 2, (len(case.requests), case.summary, case.output)
    assert len(case.requests[1]) < len(case.requests[0]), (
        len(case.requests[0]),
        len(case.requests[1]),
        case.summary,
        case.output,
    )
    assert case.exit_code == 0, case.output
    assert case.summary["engine_result"] == "SUBMITTED", case.summary
    assert case.summary["succeeded"] is True, case.summary


def _run_provider_case(work: Path, provider: str) -> Case:
    fixture = json.loads((FIXTURES / f"{provider}.json").read_text(encoding="utf-8"))
    assert isinstance(fixture["source"], str) and fixture["source"]
    body = fixture["body"]
    assert isinstance(body, dict)
    error = body["error"]
    assert isinstance(error, dict) and isinstance(error["message"], str)
    task = "Fix app.py so VALUE is 7."
    if provider == "vllm":
        # Add protected task context so refit cannot fit under vLLM's 16k limit.
        preserved_context = (
            "Keep this required release constraint in the active task context verbatim. "
        )
        repetitions = (4095 - len(task)) // len(preserved_context)
        task += "\n" + preserved_context * repetitions
        assert len(task) <= 4096
    return _run_case(work, (body,), task=task)


@pytest.fixture(scope="module")
def capacity_cases(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Future[Case]]]:
    """Independent local CLI runs can overlap without sharing endpoint state."""
    error = {
        "error": {
            "message": "context_length_exceeded",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        }
    }
    work = {
        name: tmp_path_factory.mktemp(f"capacity-400-{name}")
        for name in ("code_only", *PROVIDERS)
    }
    with ThreadPoolExecutor(max_workers=3) as pool:
        cases = {"code_only": pool.submit(_run_case, work["code_only"], (error,))}
        cases.update(
            {
                provider: pool.submit(_run_provider_case, work[provider], provider)
                for provider in PROVIDERS
            }
        )
        yield cases


def test_code_only_context_length_exceeded_400_retries_once_and_completes(
    capacity_cases: dict[str, Future[Case]],
) -> None:
    _assert_recovered(capacity_cases["code_only"].result())


@pytest.mark.parametrize("provider", PROVIDERS)
def test_provider_capacity_400_matches_fixture_expectation(
    capacity_cases: dict[str, Future[Case]], provider: str
) -> None:
    case = capacity_cases[provider].result()
    if provider == "vllm":
        assert any(event.get("kind") == "transport_error"
                   and event.get("transport_class") == "capacity"
                   for event in case.events), case.events
        learned = [event for event in case.events
                   if event.get("kind") == "capacity_context_limit_learned"]
        assert any(event.get("provider_limit_tokens") == 16_384 for event in learned), learned
        assert any(event.get("kind") == "capacity_retry_refit_failed"
                   for event in case.events), case.events
        assert len(case.requests) == 1, (len(case.requests), case.summary, case.output)
        _assert_base_terminal(case, "HARNESS_TRANSPORT", "missing_result")
        visible_result = case.output + json.dumps(case.summary, ensure_ascii=False)
        assert "maximum context length is 16384 tokens" in visible_result, (
            "the CLI result or summary must expose the readable provider capacity reason",
            case.output,
            case.summary,
        )
    else:
        _assert_recovered(case)


def test_vllm_131k_capacity_400_retries_once_and_completes(tmp_path: Path) -> None:
    _assert_recovered(_run_provider_case(tmp_path, "vllm_131k"))

"""Read-only host and local provider diagnostics for the doctor command."""
from __future__ import annotations

import http.client
import platform
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

from . import preflight
from .endpoint import Endpoint, EndpointError, EndpointProbeConfigError, EndpointProbeLimitError, _is_loopback


class Check(TypedDict):
    status: str
    detail: str


class Report(TypedDict):
    checks: dict[str, Check]
    succeeded: bool
    exit_code: int


def _python() -> Check:
    try:
        major, minor = preflight.check_python_version()
    except preflight.PreflightError as error:
        return Check(status="failed", detail=str(error))
    return Check(status="ready", detail=f"Python {major}.{minor} meets the version requirement.")


def _sandbox() -> Check:
    system = platform.system()
    if system not in {"Linux", "Darwin"}:
        return Check(status="failed", detail="Use a Linux or macOS host for sandboxed commands.")
    linux = system == "Linux"
    binary = "bwrap" if linux else "sandbox-exec"
    try:
        from .engine import kernel_safe_probe
        from .engine.kernel_test_isolation import sandbox_executable
    except ImportError:
        return Check(status="untested", detail="Sandbox startup probe unavailable. Check the StateFrame installation.")

    if sandbox_executable(binary) is None:
        action = "install bubblewrap" if linux else "restore sandbox-exec on macOS"
        return Check(status="failed", detail=f"Missing {binary}. To enable isolation, {action}.")

    try:
        with tempfile.TemporaryDirectory(prefix="stateframe-doctor-") as directory:
            result = kernel_safe_probe.run_probe(
                [sys.executable, "-I", "-B", "-c", "pass"], Path(directory))
    except (OSError, ValueError):
        return Check(status="untested", detail=f"Could not run the {binary} probe. Check TMPDIR and host permissions.")
    except kernel_safe_probe.ProbeCleanupError:
        return Check(status="failed", detail="Sandbox cleanup failed. Check host process isolation before running.")
    except kernel_safe_probe.ExecSessionError as error:
        if linux and str(error) == "Linux execution requires a non-root host UID":
            return Check(status="failed", detail="Run StateFrame as a non-root user on Linux.")
    else:
        if result.timed_out:
            return Check(status="untested", detail=f"The {binary} probe timed out. Retry when the host is less busy.")
        if result.exit_code == 0 and not result.output_truncated:
            return Check(status="ready", detail=f"{binary} started an isolated Python interpreter.")
    action = "Check user namespace permissions for bubblewrap." if linux else "Check macOS sandbox permissions."
    return Check(status="failed", detail=f"{binary} is installed but isolated Python startup failed. {action}")


def _slot_context(endpoint: Endpoint) -> Check:
    try:
        from .runner import RunError, resolve_context
    except ModuleNotFoundError as error:
        if error.name != "fcntl":
            raise
        return Check(status="unknown", detail="Check slot context requirements on a Linux or macOS host.")

    try:
        required, *_ = resolve_context()
    except RunError:
        return Check(status="failed", detail="Correct or unset STATEFRAME_CONTEXT_TOKENS and STATEFRAME_PROMPT_BUDGET.")
    try:
        slots = endpoint.get_slots() or []
    except (EndpointError, OSError, ValueError, http.client.HTTPException, RecursionError):
        slots = []
    sizes = [slot["n_ctx"] for slot in slots if isinstance(slot, dict) and type(slot.get("n_ctx")) is int]
    if sizes and min(sizes) < required:
        return Check(status="failed", detail=f"Smallest reported slot holds {min(sizes):,} tokens. "
                     f"Increase the server context to {required:,} tokens per slot or adjust STATEFRAME_CONTEXT_TOKENS.")
    if not sizes or len(sizes) != len(slots):
        return Check(status="unknown", detail="Slot sizes are unavailable or incomplete. "
                     f"Verify {required:,} context tokens per slot in the server configuration.")
    detail = f"Smallest reported slot holds {min(sizes):,} tokens ({required:,} required per slot)."
    if len(sizes) > 1 and len(set(sizes)) == 1 and sizes[0] < required * len(sizes):
        detail += (f" If the server uses --kv-unified, allow {required * len(sizes):,} "
                   "total context tokens for concurrent use of all slots.")
    return Check(status="ready", detail=detail)


def _provider_checks(url: str | None, model: str | None, key_env: str | None) -> dict[str, Check]:
    checks = {name: Check(status="skipped", detail="Provider check skipped.")
              for name in ("endpoint", "model", "slots")}
    if not url and not model and key_env is None:
        checks["endpoint"]["detail"] = "For provider checks, set --endpoint and --model."
        return checks
    if not url or not model or not model.strip():
        checks["endpoint"] = Check(status="failed", detail="Set both --endpoint and --model for provider checks.")
        return checks

    try:
        endpoint = Endpoint.from_args(url, model, key_env)
    except EndpointProbeConfigError:
        checks["endpoint"] = Check(status="failed", detail="Set STATEFRAME_ENDPOINT_MAX_JSON_BYTES to a positive integer.")
        return checks
    except EndpointError as error:
        detail = ("Set --api-key-env to the name of an existing environment variable."
                  if str(error).startswith("--api-key-env") else
                  "Set --endpoint to a loopback HTTP(S) base URL without credentials, query, or fragment.")
        checks["endpoint"] = Check(status="failed", detail=detail)
        return checks

    # Runs can opt into remote providers. Doctor only inspects loopback.
    if not _is_loopback(endpoint.base_url):
        checks["endpoint"] = Check(status="failed", detail="Set --endpoint to a loopback provider URL.")
        return checks

    try:
        observation = endpoint.probe()
    except EndpointProbeLimitError:
        checks["endpoint"] = Check(status="failed", detail="Model list exceeds STATEFRAME_ENDPOINT_MAX_JSON_BYTES. "
                                   "Check the server response size or raise the byte limit.")
        return checks
    except (EndpointError, OSError, ValueError, http.client.HTTPException, RecursionError) as error:
        if str(error).startswith("endpoint refused the credentials"):
            detail = "The server refused authentication. Check the API key supplied through --api-key-env."
        elif str(error).startswith("endpoint unreachable"):
            detail = "Could not reach the server. Start it and check the host and port in --endpoint."
        else:
            detail = "Provider discovery failed. Verify that --endpoint serves OpenAI-compatible JSON at GET /models."
        checks["endpoint"] = Check(status="failed", detail=detail)
        return checks

    auth = " with a configured API key" if endpoint.public()["api_key_present"] else ""
    checks["endpoint"] = Check(status="ready", detail=f"The local server answered GET /models{auth}.")
    found = observation["model_found"]
    if found is None:
        checks["model"] = Check(status="unknown", detail="No model list available. Verify --model in the server configuration.")
    elif found:
        checks["model"] = Check(status="ready", detail="The server lists the configured model.")
    else:
        checks["model"] = Check(status="failed", detail="Load the requested model or set --model to a listed model ID.")
    checks["slots"] = _slot_context(endpoint)
    return checks


def diagnose(url: str | None, model: str | None, key_env: str | None) -> Report:
    checks = {"python": _python(), "sandbox": _sandbox(), **_provider_checks(url, model, key_env)}
    checks["tool_calls"] = Check(status="skipped", detail="Tool-call compatibility requires a generation request.")
    succeeded = all(check["status"] not in {"failed", "untested"} for check in checks.values())
    return Report(checks=checks, succeeded=succeeded, exit_code=0 if succeeded else 2)

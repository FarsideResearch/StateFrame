"""Live V46-WIRE adapters for already-built deterministic mechanisms.

This module is intentionally an adapter, not a second state store or decision
plane.  It compiles bounded typed payloads, journals evidence, and exposes at
most one concise line to the next model request.  Existing completion and edit
authority remain in ``StateFrameRuntime`` and the release runner.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


EVENT_SCHEMA = "stateframe-v46-mechanism-wiring/1"
_ERROR_KIND_SUFFIXES = ("_error", "_failed")
_MAX_ERROR_MESSAGE_CHARS = 512
_ERROR_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|"
    r"secret|password|passwd|authorization|credential|token)\b(\s*[:=]\s*)[^\s,;]+"
)
_ERROR_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_ERROR_URL_CREDENTIALS = re.compile(r"(https?://)[^/@\s:]+:[^/@\s]+@", re.IGNORECASE)
_ERROR_AWS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_ERROR_LONG_TOKEN = re.compile(
    r"\b(?=[A-Za-z0-9_+/=-]{20,}\b)(?=[A-Za-z0-9_+/=-]*[A-Za-z])"
    r"(?=[A-Za-z0-9_+/=-]*\d)[A-Za-z0-9_+/=-]{20,}\b"
)
_ERROR_LOCAL_PATH = re.compile(r"(?i)(?:/Users|/home)/[^\s,;:'\"]+")
EDIT_KINDS = frozenset({
    "edit", "str_replace", "file_create", "replace_module", "write", "replace",
})

FLAGS = {
    "front_door": "STATEFRAME_DETREASON_FRONT_DOOR",
    "sufficiency": "STATEFRAME_DETREASON_SUFFICIENCY",
    "diagnosis": "STATEFRAME_DETREASON_DIAGNOSIS",
    "premortem": "STATEFRAME_DETREASON_PREMORTEM",
    "world_model": "STATEFRAME_DETREASON_WORLD_MODEL",
    "skeptic": "STATEFRAME_DETREASON_SKEPTIC",
    "escalation": "STATEFRAME_KERNEL_ESCALATION",
    "environment_discovery": "STATEFRAME_KERNEL_ENVIRONMENT_DISCOVERY",
    "decide2": "STATEFRAME_DECIDE2",
    "ps_invariants": "STATEFRAME_PS_INVARIANTS",
    "ps_minimize": "STATEFRAME_PS_MINIMIZE",
    "ps_router": "STATEFRAME_PS_ROUTER",
}
QGATES_FLAG = "STATEFRAME_QGATES"


def enabled(name: str, environ: Mapping[str, str] | None = None) -> bool:
    """Read one strict opt-in switch; unknown values are treated as disabled."""
    if name != "qgates" and name not in FLAGS and name not in FLAGS.values():
        raise ValueError(f"unknown V46 mechanism {name!r}")
    flag = QGATES_FLAG if name == "qgates" else FLAGS.get(name, name)
    source = os.environ if environ is None else environ
    return str(source.get(flag, "off")).strip().lower() in {"on", "1", "true", "yes"}


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _clip(value: object, limit: int = 240) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _sanitize_error_text(value: str, limit: int) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", value).strip()
    text = _ERROR_SECRET_ASSIGNMENT.sub(r"\1\2<redacted>", text)
    text = _ERROR_BEARER_TOKEN.sub("Bearer <redacted>", text)
    text = _ERROR_URL_CREDENTIALS.sub(r"\1<redacted>@", text)
    text = _ERROR_AWS_KEY.sub("<redacted>", text)
    text = _ERROR_LONG_TOKEN.sub("<redacted>", text)
    text = _ERROR_LOCAL_PATH.sub("<local-path>", text)
    return _clip(text, limit)


def sanitized_error_message(error: BaseException) -> str:
    """Keep a short diagnostic message while removing common credential forms."""
    try:
        raw = str(error)
    except Exception:  # noqa: BLE001 - formatting must not replace the failure
        raw = "<message unavailable>"
    first_line = raw.splitlines()[0][:2048] if raw else "<no message>"
    message = _sanitize_error_text(first_line, _MAX_ERROR_MESSAGE_CHARS)
    if not message:
        message = "<empty message>"
    return _clip(f"{type(error).__name__}: {message}", _MAX_ERROR_MESSAGE_CHARS)


def bounded_redacted_traceback(error: BaseException) -> str:
    """Render bounded frames without source lines, absolute paths, or credentials."""
    try:
        frames = traceback.extract_tb(error.__traceback__) if error.__traceback__ else ()
    except Exception:  # noqa: BLE001 - diagnostics must not replace the failure
        frames = ()
    if not frames:
        return "<traceback unavailable>"
    lines = ["Traceback (most recent call last):"]
    for frame in frames[-8:]:
        filename = _sanitize_error_text(Path(frame.filename).name or "<unknown>", 120)
        function = _sanitize_error_text(frame.name, 100)
        lines.append(f'  File "{filename}", line {frame.lineno}, in {function}')
    lines.append(sanitized_error_message(error))
    return _clip("\n".join(lines), 1200)


def _exception_journal_details(error: BaseException) -> dict[str, object]:
    """Build size-bounded, path-safe exception fields for an error event."""
    try:
        frames = traceback.extract_tb(error.__traceback__) if error.__traceback__ else ()
    except Exception:  # noqa: BLE001 - diagnostics must not replace the failure
        frames = ()
    if frames:
        frame = frames[-1]
        filename = _sanitize_error_text(Path(frame.filename).name or "<unknown>", 120)
        function = _sanitize_error_text(frame.name, 100)
        last_frame = f"{filename}:{frame.lineno}:{function}"
    else:
        last_frame = "<traceback unavailable>"
    return {
        "error_type": type(error).__name__,
        "message": sanitized_error_message(error),
        "traceback": bounded_redacted_traceback(error),
        "last_traceback_frame": _clip(last_frame, 240),
    }


def _is_error_kind(kind: object) -> bool:
    return isinstance(kind, str) and kind.endswith(_ERROR_KIND_SUFFIXES)


def runtime_smoke_error_kinds(records: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """Return sorted error-shaped journal kinds for the runtime smoke gate."""
    return tuple(sorted({
        kind for record in records
        if isinstance((kind := record.get("kind")), str) and _is_error_kind(kind)
    }))


def assert_runtime_smoke_error_free(records: Sequence[Mapping[str, object]]) -> None:
    """Fail a runtime smoke run and name every error-shaped journal kind."""
    kinds = runtime_smoke_error_kinds(records)
    if kinds:
        raise AssertionError(
            "runtime step smoke journal contains error kinds: " + ", ".join(kinds))


def install_error_journal_details(journal: object) -> bool:
    """Enrich error-shaped events emitted while their exception is active."""
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return False
    wrapped = getattr(emit, "__func__", emit)
    if getattr(wrapped, "_v46_error_details_wrapper", False):
        return True

    def emit_with_error_details(kind: str, **fields: object) -> object:
        if not _is_error_kind(kind):
            return emit(kind, **fields)
        error = sys.exc_info()[1]
        if not isinstance(error, BaseException):
            return emit(kind, **fields)
        payload = fields.get("payload")
        reported_type = fields.get("error_type")
        if not reported_type and isinstance(payload, Mapping):
            reported_type = payload.get("error_type")
        if reported_type and str(reported_type) != type(error).__name__:
            return emit(kind, **fields)
        details = _exception_journal_details(error)
        enriched = {**fields, **details}
        if isinstance(payload, Mapping):
            enriched["payload"] = {**payload, **details}
        return emit(kind, **enriched)

    setattr(emit_with_error_details, "_v46_error_details_wrapper", True)
    try:
        setattr(journal, "emit", emit_with_error_details)
    except (AttributeError, TypeError):
        return False
    return True


def _emit(runtime: object, kind: str, payload: Mapping[str, object]) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        raise RuntimeError("V46 mechanism wiring requires the runtime journal")
    emit(kind, producer="kernel_v46_mechanism_wiring", event_schema=EVENT_SCHEMA,
         payload=dict(payload), dispatch=False)


def _mechanism_error(runtime: object, phase: str, mechanism: str, error: Exception) -> None:
    """Journal a loud, payload-free failure; no corresponding fired event is emitted."""
    details = _exception_journal_details(error)
    message = str(details["message"])
    _emit(runtime, "v46_mechanism_wiring_error", {
        "status": "error",
        "phase": phase,
        "mechanism": mechanism,
        "message": message,
        **details,
    })
    _notice(runtime, f"[v46 error] mechanism={mechanism}; phase={phase}; {message}")


def _notice(runtime: object, line: str) -> None:
    text = _clip(line, 520)
    if text:
        runtime._v46_mechanism_notice = text


def _brief(row: Mapping[str, object]) -> str:
    for key in ("_bridge_ordinary_brief", "msg", "brief", "symptom"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Complete the requested repository work."


def _row_files(row: Mapping[str, object], root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    declared = row.get("files")
    if isinstance(declared, Mapping):
        for raw_path, value in list(declared.items())[:16]:
            if not isinstance(raw_path, str):
                continue
            if isinstance(value, Mapping):
                source = value.get("parent_src", value.get("source"))
            else:
                source = value
            if isinstance(source, str):
                files[raw_path] = source[:120_000]
    rel = row.get("rel", row.get("path"))
    if isinstance(rel, str) and rel not in files:
        path = (root / rel).resolve()
        try:
            if path.is_relative_to(root.resolve()) and path.is_file():
                files[rel] = path.read_text(encoding="utf-8", errors="ignore")[:120_000]
        except (OSError, UnicodeError):
            pass
    return files


def _done_criteria(row: Mapping[str, object]) -> list[str]:
    criteria: list[str] = []
    checks = row.get("public_check_paths")
    if isinstance(checks, Sequence) and not isinstance(checks, (str, bytes, bytearray)):
        criteria.extend(_clip(item, 120) for item in checks if str(item).strip())
    test_path = row.get("test_path")
    if isinstance(test_path, str) and test_path.strip() and test_path not in criteria:
        criteria.append(test_path.strip())
    rel = row.get("rel", row.get("path"))
    if isinstance(rel, str) and rel.strip():
        criteria.append("target exists: " + rel.strip())
    return list(dict.fromkeys(criteria)) or ["completion owner accepts the delivered work"]


def _working_contract(row: Mapping[str, object], root: Path) -> dict[str, object]:
    brief = _brief(row)
    top: dict[str, object] = {}
    question: dict[str, object] | None = None
    from detreason.front_door import interpret
    result = interpret(brief, extra_context={"files": _row_files(row, root), "activity": []})
    result_data = result.to_dict()
    verdict = str(result.verdict)
    if result.top is not None:
        top = result.top.to_dict()
    if result.question is not None:
        question = result.question.to_dict()

    top_reading = _clip(top.get("type", "repository work"), 100)
    assumptions = [
        "fast-loop proceeds with the top typed reading when clarification is unavailable",
        "existing completion authority and public checks remain authoritative",
    ]
    if verdict in {"ASK_ONE", "AMBIGUOUS"}:
        assumptions[0] = f"fast-loop assumption: proceed with top reading {top_reading!r}"
    return {
        "schema": "stateframe-v46-working-contract/1",
        "verdict": verdict,
        "goal": _clip(brief, 300),
        "top_reading": top_reading,
        "contract": top,
        "question": question,
        "assumptions": assumptions,
        "done_criteria": _done_criteria(row),
        "interpretation_digest": _digest(result_data),
    }


def _sufficiency(runtime: object, row: Mapping[str, object], contract: Mapping[str, object]) -> dict[str, object]:
    from detreason.sufficiency import (
        ENTITIES, EVIDENCE, EXTERNAL_SEAMS, INVARIANTS, ORACLE, CONTRACTS,
        RequiredInput, RequiredInputManifest, evaluate_required_inputs,
    )
    target = str(row.get("rel", row.get("path", "")) or "")
    checks = tuple(contract.get("done_criteria") or ())
    inputs = (
        RequiredInput(ENTITIES, target or "task target", "task", bool(target),
                      "target path is present", "resolve the target path"),
        RequiredInput(CONTRACTS, "typed front-door contract", "task", bool(contract.get("contract")),
                      "front door returned a top reading", "state the intended work"),
        RequiredInput(INVARIANTS, "done criteria", "task", bool(checks),
                      "completion criteria are explicit", "declare one check or completion predicate"),
        RequiredInput(EXTERNAL_SEAMS, "host dispatch receipts", "task", True,
                      "live runner owns dispatch receipts"),
        RequiredInput(EVIDENCE, "public check selectors", "task", bool(checks),
                      "criteria are carried by the task row", "identify the public check"),
        RequiredInput(ORACLE, "completion owner", "task", True,
                      "existing completion owner is the authority"),
    )
    manifest = RequiredInputManifest(
        plan_hash=_digest({"goal": contract.get("goal"), "criteria": checks}),
        node_id="task",
        stage="PLAN_PREFLIGHT",
        inputs=inputs,
        derivation_notes=("V46-WIRE adapter uses task row + front-door output; no second ledger",),
    )
    result = evaluate_required_inputs(manifest)
    runtime._v46_sufficiency_manifest = manifest
    runtime._v46_sufficiency_result = result
    return {
        "decision": result.decision,
        "manifest_hash": manifest.manifest_hash,
        "missing": [item.key for item in result.missing],
        "opened_rows": len(result.opened_rows),
    }


def task_start(runtime: object, row: Mapping[str, object], root: Path) -> None:
    """Compile the task contract and sufficiency record at the live task boundary."""
    install_error_journal_details(getattr(runtime, "journal", None))
    contract: dict[str, object] = {}
    front_door_failed = False
    if enabled("front_door"):
        try:
            contract = _working_contract(row, root)
            runtime._v46_working_contract = contract
            question = contract.get("question")
            question_text = (
                str(question.get("text", ""))
                if isinstance(question, Mapping) else ""
            )
            question_line = f"; ask_one={_clip(question_text, 180)}" if question_text else ""
            _emit(runtime, "front_door_fired", {
                "status": "compiled",
                "verdict": contract["verdict"],
                "top_reading": contract["top_reading"],
                "goal": contract["goal"],
                "done_criteria": contract["done_criteria"],
                "assumptions": contract["assumptions"],
                "question": contract["question"],
                "interpretation_digest": contract["interpretation_digest"],
            })
            _notice(
                runtime,
                f"[contract] {contract['verdict']}: goal={_clip(contract['goal'], 130)}; "
                f"done={','.join(str(x) for x in contract['done_criteria'][:3])}; "
                f"assumption={_clip(contract['assumptions'][0], 150)}{question_line}",
            )
        except Exception as error:  # noqa: BLE001 - enabled mechanisms must fail loudly
            front_door_failed = True
            _mechanism_error(runtime, "task_start", "front_door", error)
    if enabled("sufficiency"):
        try:
            if front_door_failed:
                raise RuntimeError("front_door contract unavailable")
            if not contract:
                contract = _working_contract(row, root)
                runtime._v46_working_contract = contract
            sufficiency = _sufficiency(runtime, row, contract)
            contract["sufficiency"] = sufficiency
            _emit(runtime, "sufficiency_fired", sufficiency)
            prior_notice = getattr(runtime, "_v46_mechanism_notice", "")
            missing = ",".join(str(item) for item in sufficiency["missing"]) or "none"
            _notice(
                runtime,
                f"{prior_notice}; sufficiency={sufficiency['decision']}; missing={missing}"
                if prior_notice else
                f"[sufficiency] decision={sufficiency['decision']}; missing={missing}",
            )
        except Exception as error:  # noqa: BLE001 - enabled mechanisms must fail loudly
            _mechanism_error(runtime, "task_start", "sufficiency", error)
    if enabled("qgates"):
        try:
            from kernel_qgates import task_start as qgates_task_start
            qgates_task_start(runtime, row)
        except Exception as error:  # noqa: BLE001 - typed gate wiring must be observable
            _mechanism_error(runtime, "task_start", "qgates", error)


def psgaps_checkpoint(runtime: object, frame: Mapping[str, object]) -> None:
    """Run the guarded PSGAPS survivors at the live request boundary."""
    from kernel_psgaps import checkpoint

    checkpoint(runtime, frame)


def _decide2_tool_surface(
    runtime: object,
) -> tuple[tuple[Mapping[str, object], ...] | None, str | None]:
    """Read the frozen owner surface without rebuilding request-render state."""
    from kernel_decide2 import Decide2Error, normalize_tool_surface

    missing = object()
    contract = getattr(runtime, "_completion_owner_request_contract", missing)
    if contract is missing or contract is None:
        contract = getattr(runtime, "request_action_contract", missing)
    if contract is missing or contract is None:
        return None, None
    if not isinstance(contract, Mapping):
        return None, "runtime request-action contract is not a mapping"
    try:
        surface = normalize_tool_surface(contract.get("admitted_tool_schemas"))
    except (Decide2Error, TypeError, ValueError) as error:
        return None, f"runtime request-action contract surface invalid: {type(error).__name__}"
    advertised = contract.get("advertised_actions")
    if (
        surface is None
        or not isinstance(advertised, Sequence)
        or isinstance(advertised, (str, bytes, bytearray))
    ):
        return None, "runtime request-action contract has no valid admitted tool surface"
    surface_names = tuple(
        str(tool["function"]["name"])
        for tool in surface
        if isinstance(tool.get("function"), Mapping)
    )
    advertised_names = tuple(str(name) for name in advertised)
    if surface_names != advertised_names:
        return None, "runtime request-action contract surface names do not match"
    return tuple(copy.deepcopy(dict(tool)) for tool in surface), None


@dataclass(frozen=True)
class Decide2SideCallContract:
    """Contract for one proposal request, independent of the solver runtime."""

    decision_id: str
    permitted_effect: str = "PROPOSE_ONLY"
    tool_name: str = "decide2_answer"

    def validate(self, body: Mapping[str, object]) -> None:
        kwargs = body.get("chat_template_kwargs")
        if not isinstance(kwargs, Mapping) or kwargs.get("enable_thinking") is not False:
            raise ValueError("DECIDE2 side-call reasoning must be explicitly disabled")
        tools = body.get("tools")
        if not isinstance(tools, list) or len(tools) != 1:
            raise ValueError("DECIDE2 side-call must expose exactly one dedicated tool")
        tool = tools[0]
        function = tool.get("function") if isinstance(tool, Mapping) else None
        if not isinstance(function, Mapping) or function.get("name") != self.tool_name:
            raise ValueError("DECIDE2 side-call tool is not the dedicated answer tool")


def _invoke_decide2_provider(
    provider: Callable[..., object],
    body: Mapping[str, object],
    contract: Decide2SideCallContract,
) -> object:
    """Validate one side-call body and pass its contract out-of-band when supported."""

    contract.validate(body)
    try:
        signature = inspect.signature(provider)
        signature.bind(body, side_call_contract=contract)
    except (TypeError, ValueError):
        return provider(body)
    return provider(body, side_call_contract=contract)


def _fivefix_resource_account(runtime: object) -> object:
    """Return the one resource account shared by episode and provider seams."""
    from kernel_fivefix_control import DecisionEpisodeLedger, ResourceAccount

    ledger = getattr(runtime, "_fivefix_decision_ledger", None)
    account = getattr(ledger, "account", None)
    if not isinstance(account, ResourceAccount):
        account = getattr(runtime, "_fivefix_resource_account", None)
    if not isinstance(account, ResourceAccount):
        account = ResourceAccount()
    if isinstance(ledger, DecisionEpisodeLedger):
        ledger.account = account
    runtime._fivefix_resource_account = account
    return account


def host_trigger_decide2(
    runtime: object,
    request: object,
    provider: Callable[[Mapping[str, object]], object],
    *,
    quality_fallback: bool = False,
) -> dict[str, object] | None:
    """Run DECIDE2 only when a host owner explicitly opens an episode.

    The adapter receives a provider callback from the existing host transport;
    it never exposes a decide tool or lets model output create a new episode.
    The typed receipt is journaled with bounded raw model outputs inside each
    rung receipt so format failures can be diagnosed without persisting the
    request frame or any action authority.
    """
    if not enabled("decide2"):
        return None
    try:
        from kernel_decide2 import (
            Decide2Error,
            DecisionRequest,
            DecisionTier,
            preflight_skip_result,
        )

        if not isinstance(request, DecisionRequest):
            raise TypeError("DECIDE2 host trigger requires DecisionRequest")
        tool_surface, skip_reason = _decide2_tool_surface(runtime)
        if skip_reason is not None:
            result = preflight_skip_result(request, skip_reason)
        else:
            side_call_contract = Decide2SideCallContract(str(request.decision_id))

            def side_call_provider(body: Mapping[str, object]) -> object:
                return _invoke_decide2_provider(provider, body, side_call_contract)

            result = DecisionTier(side_call_provider, tool_surface=tool_surface).run(
                request, quality_fallback=quality_fallback
            )
        receipt = result.receipt.to_dict()
        account = _fivefix_resource_account(runtime)
        rung_receipts = receipt.get("rung_receipts")
        requested_reservations = 0
        if isinstance(rung_receipts, Sequence) and not isinstance(rung_receipts, (str, bytes, bytearray)):
            requested_reservations = sum(
                int(item.get("requested_reservations", 0) or 0)
                for item in rung_receipts if isinstance(item, Mapping)
                and type(item.get("requested_reservations", 0) or 0) is int
            )
        account.charge(
            purpose="sidecall",
            owner="decide2",
            actual_provider_calls=int(receipt.get("provider_calls", 0) or 0),
            requested_reservations=requested_reservations,
            generated_tokens=int(receipt.get("generated_tokens", 0) or 0),
            reasoning_tokens=int(receipt.get("reasoning_tokens", 0) or 0),
            retries=max(0, int(receipt.get("provider_calls", 0) or 0) - 1),
        )
        _emit(runtime, "decide2_completed", receipt)
        if result.work_frame is not None:
            _notice(runtime, "[decide2] micro answer did not clear the host gate; work frame required")
        return receipt
    except Decide2Error as error:
        from kernel_decide2 import DecisionRequest, preflight_skip_result

        if isinstance(request, DecisionRequest):
            receipt = preflight_skip_result(request, str(error)).receipt.to_dict()
            account = _fivefix_resource_account(runtime)
            account.charge(
                purpose="sidecall", owner="decide2",
                actual_provider_calls=int(receipt.get("provider_calls", 0) or 0),
                generated_tokens=int(receipt.get("generated_tokens", 0) or 0),
                reasoning_tokens=int(receipt.get("reasoning_tokens", 0) or 0),
            )
            _emit(runtime, "decide2_completed", receipt)
            _notice(runtime, f"[decide2] skipped preflight: {_clip(error, 240)}")
            return receipt
        _mechanism_error(runtime, "host_trigger", "decide2", error)
        return None
    except Exception as error:  # enabled mechanism failures are observable, not silent
        _mechanism_error(runtime, "host_trigger", "decide2", error)
        return None


def host_trigger_system_one(
    runtime: object,
    facade: object,
    question: object,
    public_snapshot: Mapping[str, object],
    policy_id: str,
    budget: object | None = None,
    *,
    consume_owner: Callable[[object], object] | None = None,
    site_id: str = "",
    site_version: str | None = None,
) -> object | None:
    """Thin opt-in adapter for System One; legacy DECIDE2 is never redirected."""

    profile = getattr(runtime, "decide2_profile", getattr(runtime, "_decide2_profile", "legacy"))
    if str(profile).strip().lower() != "system_one" or not enabled("decide2"):
        return None
    decide = getattr(facade, "decide", None)
    if not callable(decide):
        _mechanism_error(runtime, "host_trigger", "system_one", RuntimeError("facade has no decide method"))
        return None
    try:
        result = decide(
            question,
            public_snapshot,
            policy_id,
            budget,
            site_id=site_id,
            site_version=site_version,
            host_triggered=True,
        )
        if consume_owner is not None and getattr(result, "proposal", None) is not None:
            consume = getattr(facade, "consume", None)
            if not callable(consume):
                raise RuntimeError("facade has no consume method")
            consume(result, public_snapshot, consume_owner)
        journal = getattr(runtime, "journal", None)
        if callable(getattr(journal, "emit", None)):
            receipt = getattr(result, "receipt", None)
            _emit(runtime, "system_one_triggered", {
                "status": str(getattr(getattr(result, "status", None), "value", getattr(result, "status", "unknown"))),
                "receipt_id": getattr(receipt, "receipt_id", None),
                "decision_id": getattr(result, "decision_id", None),
                "dispatch": getattr(receipt, "dispatch_status", None),
            })
        return result
    except Exception as error:  # noqa: BLE001 - adapter must not bypass host owner
        _mechanism_error(runtime, "host_trigger", "system_one", error)
        return None


def decide2_options(routes: Sequence[str]) -> tuple[str, ...]:
    """DECIDE2 options = host-admitted routes (deduped) + the mandatory ABSTAIN."""
    # Keep this ownerless projection import-free.  The full DECIDE2 module is
    # optional on older host Pythons, while its public sentinel is a stable
    # wire value and this reducer never executes a model decision.
    abstain = "ABSTAIN"

    kept: list[str] = []
    for item in routes:
        text = str(item)
        if text.strip().upper() == abstain or text in kept:
            continue
        kept.append(text)
    return (*kept, abstain)


def _ownerless_decide2_checkpoint(runtime: object, frame: Mapping[str, object]) -> dict[str, object] | None:
    """Open one native-evidence episode and optionally run its advisory sidecall."""
    from kernel_aha10_honest_loop import AdvisoryEpisodeLedger

    def skipped(reason: str, routes: Sequence[object] = ()) -> None:
        manifest = {"routes": list(dict.fromkeys(str(item) for item in routes if str(item))),
                    "reason": reason, "model_executed": False}
        setattr(runtime, "_aha10_decide2_effective_manifest", manifest)
        setattr(runtime, "_aha10_decide2_absence_reason", reason)
        return None

    def tool_name(tool: object) -> str:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        return str(function.get("name", "")) if isinstance(function, Mapping) else str(
            tool.get("name", "") if isinstance(tool, Mapping) else ""
        )

    contract = getattr(runtime, "native_action_contract", None)
    if not isinstance(contract, Mapping):
        contract = frame.get("native_action_contract", {})
    if not isinstance(contract, Mapping):
        contract = {}
    contract_routes = contract.get("final_actions", contract.get("admitted_actions",
                               contract.get("actions", contract.get("routes", ()))))
    raw_routes = frame.get("native_admitted_actions", frame.get("advisory_routes",
                         frame.get("routes", contract_routes)))
    if (not isinstance(raw_routes, Sequence)
            or isinstance(raw_routes, (str, bytes, bytearray))
            or not raw_routes):
        raw_routes = tuple(
            tool_name(tool) for tool in frame.get("tools", ())
            if isinstance(tool, Mapping) and tool_name(tool)
        )
    admitted = {
        tool_name(tool) for tool in frame.get("tools", ())
        if isinstance(tool, Mapping) and tool_name(tool)
    }
    routes = tuple(dict.fromkeys(
        str(item) for item in raw_routes
        if str(item).strip() and (not admitted or str(item) in admitted)
    ))
    if len(routes) <= 1:
        # DECIDE2 is a choice mechanism, not a receipt for a deterministic
        # route; ABSTAIN alone is not a real alternative.
        return skipped("deterministic_or_single_route", routes)
    native_evidence = frame.get("public_evidence", frame.get("evidence", frame.get("evidence_refs", ())))
    evidence_present = (
        bool(native_evidence) if isinstance(native_evidence, Mapping)
        else isinstance(native_evidence, Sequence)
        and not isinstance(native_evidence, (str, bytes, bytearray))
        and any(native_evidence)
    )
    if not evidence_present:
        return skipped("public_evidence_unavailable", routes)
    try:
        binding = runtime.binding()
    except (AttributeError, TypeError, ValueError):
        binding = {}
    if not isinstance(binding, Mapping):
        binding = {}
    revision = frame.get("revision", binding.get("product_revision",
                         getattr(runtime, "current_revision", "unknown")))
    dependency_identity = frame.get(
        "dependency_identity",
        frame.get("dependency_revision", binding.get("dependency_identity", "")),
    )
    menu_identity = contract.get("menu_identity", contract.get("menu_digest",
                               frame.get("menu_identity", frame.get("menu_digest", ""))))
    capability = _digest({
        "contract": contract,
        "menu_identity": menu_identity,
        "product_revision": revision,
        "dependency_identity": dependency_identity,
        "routes": routes,
    })
    options = decide2_options(routes)
    ledger = getattr(runtime, "_aha10_decide2_ledger", None)
    if not isinstance(ledger, AdvisoryEpisodeLedger):
        ledger = AdvisoryEpisodeLedger()
        setattr(runtime, "_aha10_decide2_ledger", ledger)
    episode = ledger.open(
        predicate=frame.get("unresolved_predicate", frame.get("point", "native_evidence")),
        evidence=native_evidence,
        capability=capability,
        revision=revision,
        options=options,
    )
    episode["owner"] = "native-evidence"
    episode["model_executed"] = False
    episode["evidence_source"] = "native_public_evidence"
    episode["product_revision"] = revision
    episode["dependency_identity"] = dependency_identity
    episode["menu_identity"] = menu_identity
    episode["capability_digest"] = capability
    episode["options"] = list(options)
    episode["accounting"] = {
        "provider_calls": 0, "generated_tokens": 0, "reasoning_tokens": 0,
        "execution": "not_requested",
    }
    if episode.get("status") != "OPEN":
        _emit(runtime, "decide2_checkpoint_skipped", episode)
        return episode
    provider = getattr(runtime, "decide2_provider", None)
    if callable(provider):
        try:
            from kernel_decide2 import DecisionRequest, EvidenceItem
            evidence_rows: list[EvidenceItem] = []
            evidence_values = (native_evidence.items()
                               if isinstance(native_evidence, Mapping)
                               else enumerate(native_evidence))
            for index, item in evidence_values:
                if isinstance(native_evidence, Mapping):
                    item_id = str(index)
                    description = str(item)
                    kind = "public_evidence"
                elif isinstance(item, Mapping):
                    item_id = str(item.get("id", item.get("item_id", f"public:{index}")))
                    description = str(item.get("description", item.get("text", item_id)))
                    kind = str(item.get("kind", "public_evidence"))
                else:
                    item_id, description, kind = f"public:{index}", str(item), "public_evidence"
                evidence_rows.append(EvidenceItem(item_id, description, kind=kind))
            evidence = tuple(evidence_rows)
            request = DecisionRequest(
                decision_id=_digest({"episode": episode.get("key"), "revision": revision})[:120],
                question_kind="relevance",
                question="Which admitted native route best resolves the current public evidence gap?",
                options=options,
                evidence=evidence,
                pinned_ids=tuple(item.item_id for item in evidence),
                trigger_reason="host native-evidence DECIDE2 checkpoint",
                host_triggered=True,
                risk_class="R1",
                evidence_complete=bool(evidence),
                criteria_count=2,
            )
            receipt = host_trigger_decide2(runtime, request, provider)
            if receipt is not None:
                calls = int(receipt.get("provider_calls", 0) or 0)
                episode["model_executed"] = calls > 0
                episode["provider_receipt"] = receipt
                episode["accounting"] = {
                    "provider_calls": calls,
                    "generated_tokens": int(receipt.get("generated_tokens", 0) or 0),
                    "reasoning_tokens": int(receipt.get("reasoning_tokens", 0) or 0),
                    "execution": "provider_advisory" if calls else "preflight_skipped",
                }
                episode["status"] = "PROVIDER_EXECUTED_ADVISORY" if calls else episode["status"]
                _emit(runtime, "decide2_advisory_completed", episode)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            # A missing native sidecall contract is host evidence, not a model
            # failure and never turns the advisory episode into a pass.
            episode["provider_error"] = type(error).__name__
    _emit(runtime, "decide2_advisory_episode" if episode["status"] in {
        "OPEN", "PROVIDER_EXECUTED_ADVISORY"
    } else "decide2_checkpoint_skipped", episode)
    return episode


def decide2_checkpoint(runtime: object, frame: Mapping[str, object]) -> dict[str, object] | None:
    """Open one host-owned relevance episode at the live request boundary.

    ``StateFrameRuntime.prepare`` is the host decision point immediately before
    a solver request.  The receipt is advisory: it is rendered as a bounded
    notice and never selects tools, edits, or completion.  Keeping this builder
    here makes the host trigger auditable while leaving DECIDE2's typed ladder
    and all authority in their existing modules.
    """
    if not enabled("decide2"):
        return None
    owner_resolver = getattr(runtime, "_idealpath_owner", None)
    idealpath_owner = owner_resolver() if callable(owner_resolver) else None
    if idealpath_owner is None:
        return _ownerless_decide2_checkpoint(runtime, frame)
    provider = getattr(runtime, "decide2_provider", None)
    if not callable(provider):
        return None
    target = _clip(getattr(runtime, "_target_relative", lambda: "")(), 180)
    receipt_count = frame.get("receipt_count", 0)
    snapshot = getattr(idealpath_owner, "_capability_snapshot", None)
    if snapshot is None:
        return None
    admitted = set(str(item) for item in snapshot.final_actions)
    # MENUADMIT: the frozen snapshot predates the menu-admissibility filter, so a
    # route dispatch would refuse must not be offered as a DECIDE2 option either.
    menu_admissible = getattr(idealpath_owner, "menu_admissible_actions", None)
    if callable(menu_admissible):
        admitted &= set(str(item) for item in menu_admissible())
    capability_digest = str(snapshot.capability_digest)
    if not target:
        point = "next_localization_action"
        candidates = ("localize", "localization_frontier")
        question = "Which host-admitted action should localize the unresolved target?"
    elif not isinstance(receipt_count, int) or receipt_count == 0:
        point = "next_evidence_acquisition"
        candidates = ("read", "grep")
        question = "Which host-admitted observation should acquire the next evidence?"
    else:
        point = "safe_relevance"
        candidates = ("question_gate",)
        question = "Which host-admitted route should guide the next review?"
    # DECIDE2 contract: ABSTAIN is always a legal decision option (the "no
    # admitted route fits" answer, not a menu action), so it is never filtered
    # by menu admissibility.  decide2_options is the single owner that adds it.
    routes = tuple(item for item in candidates if item in admitted)
    if len(routes) <= 1:
        return None
    options = decide2_options(routes)
    current_action = getattr(runtime, "current_action", None)
    action_kind = (
        str(current_action.get("kind", ""))
        if isinstance(current_action, Mapping) else ""
    )
    public_evidence = frame.get("public_evidence", frame.get("evidence", frame.get("evidence_refs", ())))
    if not isinstance(public_evidence, Sequence) or isinstance(public_evidence, (str, bytes, bytearray)):
        return None
    public_evidence = tuple(item for item in public_evidence if item)
    if not public_evidence:
        return None
    evidence_digest = _digest({
        "point": point,
        "action_kind": action_kind,
        "evidence": public_evidence,
        "dependency_identity": frame.get("dependency_identity", frame.get("dependency_revision", "")),
    })
    revision = str(snapshot.product_revision)
    dependency_identity = frame.get(
        "dependency_identity",
        frame.get("dependency_revision", getattr(snapshot, "dependency_identity", "")),
    )
    capability_digest = _digest({
        "snapshot_capability": capability_digest,
        "dependency_identity": dependency_identity,
    })
    from kernel_fivefix_control import DecisionEpisodeLedger, ResourceAccount
    ledger = getattr(runtime, "_fivefix_decision_ledger", None)
    if not isinstance(ledger, DecisionEpisodeLedger):
        account = getattr(runtime, "_fivefix_resource_account", None)
        ledger = DecisionEpisodeLedger(
            account=account if isinstance(account, ResourceAccount) else None,
        )
        runtime._fivefix_decision_ledger = ledger
    episode = ledger.open(
        unresolved_predicate=point,
        evidence_digest=evidence_digest,
        capability_digest=capability_digest,
        revision=revision,
        options=options,
    )
    episode["dependency_identity"] = dependency_identity
    if episode.get("status") != "OPEN":
        return None
    checkpoint_key = str(episode["key"])

    try:
        from kernel_decide2 import DecisionRequest, EvidenceItem

        evidence_rows: list[object] = []
        for index, item in enumerate(public_evidence):
            if isinstance(item, Mapping):
                item_id = str(item.get("id", item.get("item_id", f"public:{index}")))
                description = str(item.get("description", item.get("text", item_id)))
                kind = str(item.get("kind", "public_evidence"))
            else:
                item_id = f"public:{index}"
                description = str(item)
                kind = "public_evidence"
            evidence_rows.append(EvidenceItem(item_id, description, kind=kind))
        evidence = tuple(evidence_rows)
        request = DecisionRequest(
            decision_id=_digest({
                "run_id": str(getattr(getattr(runtime, "journal", None), "run_id", "")),
                "checkpoint": checkpoint_key,
            })[:120],
            question_kind=point,
            question=question,
            options=options,
            evidence=evidence,
            pinned_ids=tuple(item.item_id for item in evidence),
            trigger_reason=f"host request-boundary {point} checkpoint",
            host_triggered=True,
            risk_class="R1",
            evidence_complete=bool(evidence),
            criteria_count=2,
        )
        previous = getattr(runtime, "_decide2_last_receipt", None)
        retry_quality = (
            isinstance(previous, Mapping)
            and str(previous.get("status", "")) in {
                "abstain", "invalid", "transport_error", "escalated"
            }
        )
        receipt = host_trigger_decide2(
            runtime,
            request,
            provider,
            quality_fallback=(
                bool(getattr(runtime, "decide2_quality_fallback", False))
                or retry_quality
            ),
        )
        if receipt is None:
            ledger.close_contract_error(
                key=checkpoint_key,
                error_digest=_digest({"status": "sidecall_failed", "key": checkpoint_key}),
            )
            return None
        runtime._decide2_last_receipt = receipt
        ledger.close(
            key=checkpoint_key,
            reason=f"sidecall:{str(receipt.get('status', 'settled'))}",
        )
        _notice(
            runtime,
            f"[decide2] point={point}; "
            f"status={_clip(receipt.get('status'), 40)}; "
            f"route={_clip(receipt.get('route'), 40)}; "
            f"arm={_clip(receipt.get('arm'), 40)}; "
            f"rung={_clip(receipt.get('rung'), 40)}",
        )
        return receipt
    except Exception as error:  # enabled mechanism failures are observable, not fatal
        ledger.close_contract_error(
            key=checkpoint_key,
            error_digest=_digest({
                "error_type": type(error).__name__,
                "message": str(error).splitlines()[0][:240] if str(error) else "<no message>",
            }),
        )
        _mechanism_error(runtime, "request_boundary", "decide2", error)
        return None


def _action_file(runtime: object, arguments: Mapping[str, object]) -> tuple[Path, str, str] | None:
    raw = arguments.get("path", arguments.get("file"))
    if not isinstance(raw, str) or not raw.strip():
        return None
    root = Path(getattr(getattr(runtime, "owner", None), "root", ".")).resolve()
    path = (root / raw).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        return None
    source = ""
    try:
        if path.is_file():
            source = path.read_text(encoding="utf-8", errors="ignore")[:120_000]
    except (OSError, UnicodeError):
        pass
    return path, relative, source


def before_action(runtime: object, kind: str, arguments: Mapping[str, object]) -> None:
    """Compute premortem + impact prediction immediately before first product edit."""
    action_kind = str(kind or "").strip().lower()
    if action_kind not in EDIT_KINDS or getattr(runtime, "_v46_consequential_edit_seen", False):
        return
    action_file = _action_file(runtime, arguments)
    if action_file is None:
        return
    runtime._v46_consequential_edit_seen = True
    _path, relative, source = action_file
    contract = getattr(runtime, "_v46_working_contract", {})
    payload: dict[str, object] = {"file": relative, "kind": action_kind}
    risk_line = "risk=unknown"
    premortem_payload: dict[str, object] = {"status": "disabled"}
    prediction_payload: dict[str, object] = {"status": "disabled"}
    if enabled("premortem"):
        try:
            from detreason import premortem
            feature_contract = dict(contract.get("contract") or {})
            feature_contract.update({
                "objective": contract.get("goal", ""),
                "components": [{"id": relative, "type": "unknown", "name": relative}],
            })
            spec = premortem.front_door_contract_to_feature_spec(
                feature_contract, feature_id=_digest({"file": relative, "goal": contract.get("goal")}),
            )
            executable = tuple(getattr(premortem, "_EXECUTABLE_RULE_SPECS", ()))
            corpus = premortem.build_rule_corpus(tuple(rule.id for rule in executable))
            report = premortem.generate_premortem(spec, corpus)
            risks = tuple(getattr(report, "risks", ()))
            premortem_payload = {
                "status": "predicted",
                "report_digest": getattr(report, "report_digest", ""),
                "risk_count": len(risks),
                "high_risk_ids": [str(item.risk_id) for item in risks[:5]],
                "coverage_complete": bool(getattr(getattr(report, "coverage", None), "complete", False)),
            }
            _emit(runtime, "premortem_fired", premortem_payload)
            coverage_complete = premortem_payload["coverage_complete"] is True
            if risks or coverage_complete:
                risk_line = f"premortem={len(risks)} risks"
            else:
                # Zero-risk is a clearance claim only when the host proved its
                # rule corpus was complete; UNKNOWN must stay UNKNOWN.
                risk_line = ""
        except Exception as error:  # noqa: BLE001 - enabled mechanisms must fail loudly
            _mechanism_error(runtime, "before_action", "premortem", error)
            risk_line = "premortem=error"
    if enabled("world_model"):
        try:
            from detreason.world_model import predict_impact
            files = {relative: source}
            prediction = predict_impact(files, payload, depth=2)
            runtime._v46_world_prediction = prediction
            prediction_payload = {
                "status": "predicted",
                "prediction_id": prediction.prediction_id,
                "risk_tier": prediction.risk_tier,
                "affected_count": len(prediction.affected),
                "affected": [item.target for item in prediction.affected[:6]],
            }
            _emit(runtime, "world_model_fired", prediction_payload)
            affected_count = len(prediction.affected)
            prediction_coverage = getattr(getattr(prediction, "coverage", None), "complete", None)
            if (affected_count or str(prediction.risk_tier).upper() not in {"LOW", "NONE"}
                    or prediction_coverage is True):
                blast_line = f"blast={prediction.risk_tier}/{affected_count} affected"
                risk_line = f"{risk_line}; {blast_line}" if risk_line else blast_line
        except Exception as error:  # noqa: BLE001 - enabled mechanisms must fail loudly
            _mechanism_error(runtime, "before_action", "world_model", error)
            risk_line = f"{risk_line}; blast=error" if risk_line else "blast=error"
    if risk_line:
        _notice(runtime, f"[preflight] {risk_line}")


def _result_class(observation: object) -> str:
    text = str(observation or "").lower()
    if "applied" in text and "reject" not in text:
        return "APPLIED"
    if any(word in text for word in ("red", "failed", "rejected", "blocked", "no-op")):
        return "RED"
    return "OBSERVED"


def _diagnosis_row(runtime: object, action: str, observation: str) -> dict[str, object]:
    return {
        "level": getattr(getattr(runtime, "owner", None), "current_step", 0) or 0,
        "func": action[:120],
        "attempts": [{"n": 0, "mode": "live_action", "raw": observation[:240], "gate": "stall"}],
        "score": {"located": 1, "tests_pass": 0, "no_collateral": 1, "extras": []},
        "result_class": _result_class(observation),
    }


def after_observation(runtime: object, action: str, observation: str) -> None:
    """Score pending impact and attribute repeated no-progress action cycles."""
    # This is the live receipt boundary.  It runs regardless of optional V46
    # feature flags so stale evidence cannot hide behind an off diagnostic.
    try:
        from kernel_invalidation import sync_runtime_invalidation
        sync_runtime_invalidation(runtime)
    except Exception as error:  # noqa: BLE001 - dependency safety is fail-closed at done
        runtime._v46_invalidation_error = type(error).__name__
    if enabled("qgates"):
        try:
            from kernel_qgates import after_observation as qgates_after_observation
            qgates_after_observation(runtime, action, observation)
        except Exception as error:  # noqa: BLE001 - gate telemetry must not break dispatch
            _mechanism_error(runtime, "after_observation", "qgates", error)
    prediction = getattr(runtime, "_v46_world_prediction", None)
    if prediction is not None and enabled("world_model"):
        try:
            from detreason.world_model import observe_outcome
            scored = observe_outcome(prediction, [], [], test_results=[])
            _emit(runtime, "world_model_scored", {
                "prediction_id": prediction.prediction_id,
                "brier": scored.brier,
                "surprises": list(scored.surprises),
                "new_violations": list(scored.new_violations),
                "failed_tests": list(scored.failed_tests),
            })
        except Exception as error:  # noqa: BLE001 - enabled mechanisms must fail loudly
            _mechanism_error(runtime, "after_observation", "world_model", error)
        runtime._v46_world_prediction = None

    if not enabled("diagnosis"):
        return
    current_class = _result_class(observation)
    if current_class == "APPLIED":
        runtime._v46_stall_cycles = {}
        return
    cycles: dict[str, int] = getattr(runtime, "_v46_stall_cycles", {})
    key = _digest({"action": str(action)[:240], "result_class": current_class})
    cycles[key] = cycles.get(key, 0) + 1
    runtime._v46_stall_cycles = cycles
    fired: set[str] = getattr(runtime, "_v46_diagnosis_keys", set())
    if cycles[key] < 2 or key in fired:
        return
    fired.add(key)
    runtime._v46_diagnosis_keys = fired
    try:
        from detreason.diagnosis import CAUSE_METADATA, attribute
        finding = attribute(_diagnosis_row(runtime, action, observation))
    except Exception as error:  # noqa: BLE001 - enabled mechanisms must fail loudly
        fired.discard(key)
        runtime._v46_diagnosis_keys = fired
        _mechanism_error(runtime, "after_observation", "diagnosis", error)
        return
    primary = finding.get("primary")
    next_action = (CAUSE_METADATA.get(primary or "", {}).get("action")
                   or "inspect the latest failed observation, then run the smallest public check")
    payload = {
        "cycle_digest": key,
        "same_cycle_count": cycles[key],
        "primary": primary,
        "confidence": finding.get("confidence"),
        "evidence": finding.get("evidence", {}),
        "next_action": next_action,
    }
    _emit(runtime, "diagnosis_fired", payload)
    _notice(runtime, f"[diagnosis] {primary or 'insufficient evidence'}; next={_clip(next_action, 300)}")


def _test_status(runtime: object) -> tuple[bool, bool]:
    """Prefer current trusted verdicts over historical red receipts."""
    def field(value: object, name: str, default: object = None) -> object:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    verdict_fn = getattr(runtime, "public_test_verdict", None)
    binding_fn = getattr(runtime, "binding", None)
    if callable(verdict_fn) and callable(binding_fn):
        try:
            verdict = verdict_fn()
            current = binding_fn()
            state = str(field(verdict, "state", "")).upper()
            verdict_binding = field(verdict, "current_binding")
            receipts = tuple(field(verdict, "trusted_receipts", ()) or ())
            if state in {"PASS", "PASSED", "SATISFIED"} and verdict_binding == current and receipts and all(
                    field(receipt, "binding") == current
                    and field(receipt, "after_binding") == current
                    and field(receipt, "exit_code") == 0
                    and field(receipt, "binding_stable") is True
                    for receipt in receipts):
                return True, False
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
    receipts = (
        *getattr(runtime, "test_receipts", ()),
        *getattr(runtime, "probe_receipts", ()),
        *getattr(runtime, "fault_receipts", ()),
    )
    green = False
    red = False
    current_binding = None
    if callable(binding_fn):
        try:
            current_binding = binding_fn()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            current_binding = None
    for receipt in receipts:
        if isinstance(receipt, Mapping):
            value = str(receipt.get("result_class", receipt.get("status", ""))).upper()
            exit_code = receipt.get("exit_code")
            receipt_binding = receipt.get("binding", receipt.get("after_binding"))
        else:
            value = str(getattr(receipt, "result_class", getattr(receipt, "status", ""))).upper()
            exit_code = getattr(receipt, "exit_code", None)
            receipt_binding = getattr(receipt, "binding", getattr(receipt, "after_binding", None))
        if current_binding is not None and receipt_binding is not None and receipt_binding != current_binding:
            continue
        green |= value in {"GREEN", "PASS", "PASSED"} or exit_code == 0
        red |= value in {"RED", "FAIL", "FAILED", "REJECTED"} or (
            isinstance(exit_code, int) and exit_code != 0)
    last_exit = getattr(getattr(runtime, "owner", None), "last_test_exit", None)
    red |= isinstance(last_exit, int) and last_exit != 0
    green |= isinstance(last_exit, int) and last_exit == 0
    return green, red


def before_completion(runtime: object, admissible: bool, kwargs: Mapping[str, object]) -> dict[str, object] | None:
    """Audit done against clean evidence; the dependency graph owns this gate."""
    try:
        from kernel_invalidation import runtime_completion_gate
        dependency_finding = runtime_completion_gate(runtime)
        if dependency_finding.get("admitted") is False:
            _notice(runtime, "[depgraph] " + _clip(dependency_finding.get("finding", "stale evidence")))
            return dependency_finding
    except Exception as error:  # noqa: BLE001 - unresolved dependency state must fail closed
        _mechanism_error(runtime, "before_completion", "invalidation", error)
        return {
            "admitted": False,
            "reason_code": "DEPGRAPH_ERROR",
            "finding": "dependency graph could not be evaluated",
            "next_action": "re-run the current public check before completion",
        }
    manifest = getattr(runtime, "_s47_verification_manifest", None)
    if isinstance(manifest, Mapping):
        try:
            from kernel_s47_causal_fix import verify_manifest
            context = getattr(runtime, "_s47_verification_context", {})
            context = context if isinstance(context, Mapping) else {}
            decision = verify_manifest(manifest, context)
            runtime._s47_verification_decision = dict(decision)
            if decision.get("status") != "READY_FOR_S7":
                blockers = ", ".join(str(item) for item in decision.get("blockers", ()) if str(item))
                return {
                    "admitted": False,
                    "reason_code": "S47_VERIFICATION_REQUIRED",
                    "finding": blockers or "S47 verification conjunction is incomplete",
                    "next_action": "host_stop" if not blockers else "run_s47_recovery",
                    "blockers": list(decision.get("blockers", ())),
                }
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            _mechanism_error(runtime, "before_completion", "s47_verification", error)
            return {
                "admitted": False,
                "reason_code": "S47_VERIFICATION_ERROR",
                "finding": "typed verification could not be evaluated",
                "next_action": "host_stop",
            }
    if enabled("qgates") and admissible:
        try:
            from kernel_qgates import before_completion as qgates_before_completion
            finding = qgates_before_completion(runtime, kwargs)
            if finding is not None:
                return finding
        except Exception as error:  # noqa: BLE001 - unresolved gates must fail closed at completion
            _mechanism_error(runtime, "before_completion", "qgates", error)
            return {
                "reason_code": "QGATE_ERROR",
                "finding": "typed state gate could not be evaluated",
                "next_action": "answer the active typed gate",
            }
    if not enabled("skeptic"):
        return None
    if admissible:
        try:
            from kernel_completion_owner import completion_owner_reused_verification

            if completion_owner_reused_verification(runtime):
                _emit(runtime, "skeptic_cleared_by_reused_verification", {
                    "reason": "greenlock identical-rerun refusal equals current-green evidence",
                })
                _notice(runtime, "[skeptic] current-green named evidence admitted; completion may continue")
                return None
        except (ImportError, AttributeError, TypeError, ValueError):
            # The owner evidence is an optional live seam; the ordinary skeptic
            # path remains authoritative when no owner state is installed.
            pass
    green, red = _test_status(runtime)
    contract = getattr(runtime, "_v46_working_contract", {})
    claim = None
    score = 0.0
    breakdown: object = {}
    try:
        from detreason.skeptic import Claim, suspicion_score
        claim = Claim(
            ledger="stateframe-v46",
            line_no=int(getattr(getattr(runtime, "owner", None), "current_step", 0) or 0),
            key=str(contract.get("goal", "completion"))[:180],
            fn=getattr(getattr(runtime, "owner", None), "fn", None),
            kind="completion",
            verdict="ACCEPT" if admissible else "ABSTAIN",
            fixed=1 if green else 0,
            gold_contact=1 if green else 0,
            margins=[],
            stages=[{"contract": bool(contract), "red_evidence": red}],
            raw={"admissible": bool(admissible), "red_evidence": red},
        )
        score, breakdown = suspicion_score(claim)
    except Exception as error:  # noqa: BLE001 - enabled mechanisms must fail loudly
        _mechanism_error(runtime, "before_completion", "skeptic", error)
        return None
    _emit(runtime, "skeptic_fired", {
        "admissible": bool(admissible),
        "green_evidence": green,
        "red_evidence": red,
        "suspicion_score": score,
        "breakdown": breakdown,
        "contract_digest": _digest(contract),
    })
    if not admissible or not red or getattr(runtime, "_v46_skeptic_refused", False):
        return None
    runtime._v46_skeptic_refused = True
    finding = {
        "reason_code": "DETREASON_SKEPTIC",
        "finding": "completion claim conflicts with a recorded red check",
        "next_action": "run the exact failing public check and repair its reported failure",
    }
    _notice(runtime, f"[skeptic] {finding['finding']}; next={finding['next_action']}")
    return finding


def _attempt_record(runtime: object, terminal_result: Mapping[str, object]) -> dict[str, object]:
    result = str(terminal_result.get("result", terminal_result.get("outcome", "RED")))
    return {
        "key": {"attempt_no": 1, "tag": "stateframe-v46-live"},
        "world": {"provenance": "STATEFRAME_RUNTIME"},
        "edits": tuple(),
        "failed_legs": ({
            "test_selector": "public-checks",
            "fp": _digest({"result": result, "task": getattr(getattr(runtime, "owner", None), "task_id", "task")}),
            "anchor": "attempt ceiling reached without accepted completion",
            "evidence_refs": ("receipt:stateframe-v46",),
        },),
        "terminal": {"shape": "BUDGET_EXHAUSTED", "steps": terminal_result.get("steps", 0), "result": result},
        "audit": {"source_refs": ("journal:stateframe-v46",)},
        "autopsy_class": "EXHAUSTED",
    }


def at_attempt_ceiling(runtime: object, terminal_result: Mapping[str, object]) -> None:
    """Discover reproduction commands at the ceiling and escalate only if not converged."""
    from kernel_done_is_submit_authority import done_sealed

    if done_sealed(runtime) is not None:
        return
    if not (enabled("escalation") or enabled("environment_discovery")):
        return
    result = dict(terminal_result)
    root = Path(getattr(getattr(runtime, "owner", None), "root", ".")).resolve()
    if enabled("environment_discovery"):
        try:
            from kernel_environment_discovery import discover_repository
            report = discover_repository(root)
            summary = {
                "status": "discovered",
                "coverage": report.coverage,
                "source_files": list(report.source_files)[:16],
                "checks": [{"id": item.check_id, "argv": list(item.argv)} for item in report.checks[:8]],
                "recipes": [{"id": item.recipe.recipe_id, "command_ref": item.recipe.command_descriptor_ref}
                            for item in report.recipes[:8]],
                "omissions": list(report.omissions)[:8],
            }
        except Exception as error:  # noqa: BLE001 - enabled mechanisms must fail loudly
            summary = {}
            _mechanism_error(runtime, "at_attempt_ceiling", "environment_discovery", error)
        if summary:
            _emit(runtime, "environment_discovery_fired", summary)
        checks = ",".join(item["id"] for item in summary.get("checks", []) if isinstance(item, Mapping)) or "none"
        _notice(runtime, f"[reproduction] checks={checks}; use declared setup recipes before retrying")
    accepted = (
        bool(result.get("accepted"))
        and result.get("certified", True) is not False
    ) or str(result.get("result", "")).upper() in {"PASS", "GREEN", "ACCEPTED"}
    if enabled("escalation") and not accepted:
        try:
            from kernel_escalation import build_escalation_packet
            attempt = _attempt_record(runtime, result)
            packet = build_escalation_packet(
                (attempt,), attempt_ceiling=1,
                task_id=str(getattr(getattr(runtime, "owner", None), "task_id", "task")),
                converged=False,
            )
            if packet is None:
                _emit(runtime, "escalation_fired", {"status": "no_packet", "attempt_ceiling": 1})
                return
            _emit(runtime, "escalation_fired", {
                "status": "built",
                "packet_id": packet.packet_id,
                "attempt_ceiling": packet.attempt_ceiling,
                "decider_question": packet.decider_question,
                "unknown_count": len(packet.unknown),
                "violated_count": len(packet.violated),
            })
            _notice(runtime, f"[escalation] {packet.decider_question}")
        except Exception as error:  # noqa: BLE001 - enabled mechanisms must fail loudly
            _mechanism_error(runtime, "at_attempt_ceiling", "escalation", error)


__all__ = [
    "EDIT_KINDS", "EVENT_SCHEMA", "FLAGS", "QGATES_FLAG", "after_observation", "at_attempt_ceiling",
    "before_action", "before_completion", "decide2_checkpoint", "enabled", "psgaps_checkpoint",
    "host_trigger_decide2", "host_trigger_system_one",
    "Decide2SideCallContract",
    "assert_runtime_smoke_error_free", "bounded_redacted_traceback",
    "install_error_journal_details", "runtime_smoke_error_kinds",
    "sanitized_error_message", "task_start",
]

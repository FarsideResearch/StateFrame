"""Optional host-bound reconciliation of stale nitpicker contradictions."""
import os
from pathlib import PurePosixPath


def enabled() -> bool:
    value = os.environ.get("KERNEL_NIT_BOUND_GREEN", "0")
    if value not in {"0", "1"}:
        raise ValueError("KERNEL_NIT_BOUND_GREEN must be 0 or 1")
    return value == "1"


def selector_covers(green: str, red: str) -> bool:
    def valid(value):
        base = value.split("::", 1)[0]
        path = PurePosixPath(base)
        return (bool(base) and base.endswith(".py") and not path.is_absolute()
                and ".." not in path.parts and str(path) == base
                and "filtered[" not in value and not any(c.isspace() for c in value))
    return (valid(green) and valid(red)
            and (red == green or red.startswith(green + "::")))


def reconcile(runtime, state: dict) -> tuple[str, ...]:
    from kernel_intervention_policy import interventions_enabled
    from kernel_verdict import SATISFIED, evaluate_required_tests
    if not runtime.nit_bound_green or not interventions_enabled():
        return ()
    nit = getattr(runtime.owner, "nit", None)
    world = state.get("world", {})
    if world.get("valid_product_verification") is not True:
        return ()
    required = tuple(world.get("required_selectors", ()))
    binding = runtime.binding()
    authority = (runtime.public_test_verdict() if callable(
        getattr(runtime, "public_test_verdict", None)) else
                 evaluate_required_tests(binding, required, runtime.test_receipts))
    if nit is None or authority.state != SATISFIED:
        return ()
    evidence = set(authority.supporting_evidence)
    receipts = [receipt for receipt in runtime.test_receipts
                if str(receipt.get("receipt_id", receipt.get("event_id", ""))) in evidence]
    checked = tuple(selector for receipt in receipts for selector in receipt.get("selectors", ())
                    if any(selector_covers(scope, selector) for scope in required))
    red_scopes = {atom.contract_key: sorted(nit._red_scopes.get(atom.contract_key, ()))
                  for atom in nit.contradicted}
    resolved = nit.reconcile_public_green(checked)
    if resolved:
        runtime.journal.emit("nitpicker_bound_green", resolved_contracts=list(resolved),
                             checked_selectors=list(checked), binding=binding,
                             required_selectors=list(required), certification=world.get("certification"),
                             valid_product_verification=True,
                             covered_red_scopes={key: red_scopes[key] for key in resolved},
                             receipt_ids=[receipt["receipt_id"] for receipt in receipts])
    return resolved

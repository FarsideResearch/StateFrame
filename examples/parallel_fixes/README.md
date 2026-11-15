# parallel_fixes (StateFrame subframes demo)

This example has five pricing repairs in separate source files, with focused pytest checks for discount, quantity,
rounding, tax, and checkout behavior. The checkout module imports `calculate_tax` from the tax module. If the parent
splits those repairs into subframes and declares that dependency, checkout waits until the tax proposal is committed
and receives its committed interface.

Run it with a configured OpenAI-compatible endpoint:

```bash
stateframe run --repo examples/parallel_fixes \
  --task "Repair the pricing calculations and include tax in checkout." \
  --parallel 2
```

Subframes are opt-in and default to off. `--parallel 2` offers the `parallel` tool and allows up to two subframes to
run concurrently. The model decides whether to use it, so an enabled run can legitimately use zero subframes.

To see whether subframes ran, open `summary.json` in the run directory printed by StateFrame. When present, its
`parallel` section includes:

- `totals.subframes`: the number of recorded subframe outcomes, including refusals and cancellations.
- `totals.requests`: the number of subframe model requests. A positive value confirms subframe model requests were made.
- `measured.peak_active_subframes`: the highest number of active subframes at once. A value above one records overlap.

Individual records are under `parallel/<call>/units/*` in that run directory. If the model never calls `parallel`,
the summary may have no `parallel` section. See the [subframes guide](../../docs/guide/subframes.md) for limits,
refusals, and recovery. Subframes do not promise a general speedup.

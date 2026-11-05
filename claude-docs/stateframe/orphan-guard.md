# StateFrame pytest orphan guard

The test guard only inspects processes explicitly registered by a scope or adopted with complete ownership evidence. A registered launch is eligible for leak detection after its scope ends; a live process is stopped through its retained process handle, and exit must be confirmed separately from signal delivery.

Linux retains a pidfd. macOS captures a PID generation and signals through its audit token. macOS identity probes clear `ctypes` errno before calling libproc, so only an error from that probe can establish absence. `signal.Signals` values are valid positive signal numbers, so validation must accept integer enums while still rejecting booleans, zero, and negatives.

The pytest plugin checks registrations after fixture teardown and records any leak or cleanup gap in the generated teardown report. It must preserve an existing teardown failure and still make the item fail when the guard adds another finding.

Acceptance tests for real pidfd/audit-token behavior are host-sandbox cases. Non-host unit tests cover scope bookkeeping, signal adapter contracts, macOS identity liveness, and pytest report mutation. Acceptance fixtures are frozen; setup defects in them belong in the lane report rather than edits to those tests.

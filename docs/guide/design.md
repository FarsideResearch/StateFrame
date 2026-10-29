# Design rationale

**Typed state replaces history.** A growing transcript makes every turn more expensive and buries the facts that matter. StateFrame keeps those facts as typed state and sends only what the task needs now. When the frame is missing something a model needs, the fix is to make the state carry it, not to bring history back.

**The host owns facts, the model owns decisions.** Whatever the engine can measure (which tests pass, what was read, what changed, what is owed) it records and hands to the model plainly. A model that does not have to remember bookkeeping makes fewer bookkeeping mistakes, which matters most for small local models.

**Done is decided by checks, not by claims.** The model's "done" starts verification; it does not end the task. Protected tests cannot be edited, and the outcome you see comes from checks the host ran.

**Fail closed and say why.** A request that does not fit, a setting the engine cannot read, a resume that would not be faithful: each stops with a named reason instead of degrading silently.

**Every request is reproducible and inspectable.** Requests are recorded byte for byte, ids are not random, and volatile output is normalized, so a run can be replayed and a request can be compared with its record. That is what makes resume safe to trust.

**Mechanisms over instructions.** When behavior matters, the engine enforces it in code (tool menus, guards, budgets) rather than asking the model in prose. Prose can be ignored; a guard cannot.

**Local first and safe by default.** Loopback endpoints only unless you opt in, work on a copy, sandboxed commands without network, and changes to your repository only when you ask.

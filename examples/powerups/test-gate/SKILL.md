---
name: test-gate
description: Require a configured check command before submission.
---
The stop hook runs the configured check command before submission. A failing check vetoes submission and returns the failing output tail. Fix the cause and try to submit again. The command is split into arguments without a shell.

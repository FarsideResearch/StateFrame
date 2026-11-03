---
name: repo-grep
description: Search repository text with a bounded read-only command.
---
Use `grep_repo(pattern, glob)` to find literal text in repository files. Supply a narrow glob when possible. Results are capped; use the built-in read tool to inspect matches in context.

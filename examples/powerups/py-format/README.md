# py-format

This powerup formats Python files changed by a tool call. Its post-tool hook may write only the changed `*.py` paths. Set `line_length` between 40 and 200. Install it with `stateframe powerups install examples/powerups/py-format`, then enable it with `stateframe powerups enable py-format --set line_length=100`.

Ruff is optional and is not installed by StateFrame. It must be available in the hook sandbox PATH or bundled as an executable `ruff` beside `format.py` and declared in each hook's `files` list. If `ruff` is unavailable, session start adds one short notice and later formatter calls exit successfully without changing files. The run continues normally.

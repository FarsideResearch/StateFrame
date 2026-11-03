# test-gate

This powerup runs `check_command` at the stop boundary. A failing check vetoes submission and shows its last 2 KiB of output. The hook stops a check after 120 seconds or 1 MiB of output. The hook explicitly uses `on_error: block`. Install with `stateframe powerups install examples/powerups/test-gate`, then enable it with `stateframe powerups enable test-gate --set 'check_command=python3 -B -m pytest -q -p no:cacheprovider'`.

The configured program is not installed by StateFrame. If it is unavailable, session start adds one short notice and the stop hook refuses submission, so an unchecked result is never submitted. Install the program or disable the powerup. The hook loader refuses a single absolute regular-file operand that is not declared as a bundled file, so use a program name or a command with arguments.

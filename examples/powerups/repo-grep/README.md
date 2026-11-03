# repo-grep

This powerup offers a read-only `grep_repo(pattern, glob)` command that searches literal text. It caps traversal at 20,000 directory entries and output at 50 matches. The executable script uses `#!/usr/bin/env python3` to find Python on `PATH`, requires no network access, and has local replay. Install with `stateframe powerups install examples/powerups/repo-grep`, then enable with `stateframe powerups enable repo-grep`.

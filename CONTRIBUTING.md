# Contributing to StateFrame

Thanks for wanting to help. Bug reports, fixes, tests and documentation improvements are all welcome.

## Before you start

StateFrame is dual-licensed: under the GNU Affero General Public License v3.0 only, or under a separate commercial license from Farside Research LLC. See LICENSE and COMMERCIAL-LICENSE.md.

To keep both options possible, every contribution needs a signed Contributor License Agreement before it can be merged. You keep the copyright in your work; the agreement gives the project the rights it needs to ship it under both licenses. Please read CLA.md before you open a pull request.

## Reporting a problem

Open an issue and include:

- what you ran (the exact command),
- what you expected and what happened instead,
- the run directory's `summary.json` and the last part of `engine.log`, if a run went wrong,
- your Python version and operating system.

Please leave out source code or data you are not allowed to share.

## Making a change

- Keep pull requests small and focused on one thing. A short description of the problem and the fix helps a lot.
- Add or update a test that fails without your change and passes with it.
- Run the test suite before you submit:

  ```bash
  python -m pytest -q tests
  ```

- Match the style of the code around your change: clear names, short functions, comments only where they explain why.
- Do not add a runtime dependency without discussing it in an issue first. Any new dependency must use a permissive license (MIT, BSD, Apache-2.0, PSF or similar).

## Review

A maintainer will read your change, may ask for small adjustments, and will merge it once the tests pass and your signed CLA is on file. Thanks again for helping.

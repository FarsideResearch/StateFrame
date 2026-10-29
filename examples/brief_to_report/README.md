# brief_to_report (StateFrame file-output demo, no code)

The agent reads a short brief and a CSV of support tickets in `input/` and writes two files in `report/`:
`summary.json` (counts by category) and `report.md` (a short written report with the same numbers).
Acceptance is `tests/test_report.py` (stdlib only): it recomputes every number from the CSV and checks the
JSON shape and the report's headings against it. It fails until both files exist and agree with the data.

The same pattern works for any task whose result is files: write the acceptance as a pytest file over the
output files. See "Tasks whose output is files" in the main README.

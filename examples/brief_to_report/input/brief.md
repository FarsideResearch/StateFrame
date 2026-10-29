# Brief: support tickets, week 42

Write a short report on this week's support tickets from `tickets.csv`.

- `report/summary.json`: `{"total": <int>, "open": <int>, "by_category": {"<category>": <int>, ...}}`,
  counting every ticket once. Categories are exactly the ones in the CSV.
- `report/report.md`: a `# Support tickets, week 42` title, a `## Summary` section that states the total and
  how many are still open, and a `## By category` section with one line per category in the form
  `- <category>: <count>`, largest first.

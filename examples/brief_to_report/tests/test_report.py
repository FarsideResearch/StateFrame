"""Acceptance for report/ (stdlib only): every number is recomputed from input/tickets.csv."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"


def tickets():
    with (ROOT / "input" / "tickets.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expected():
    rows = tickets()
    return {"total": len(rows), "open": sum(row["status"] == "open" for row in rows),
            "by_category": dict(Counter(row["category"] for row in rows))}


def test_summary_json_matches_the_data():
    path = REPORT / "summary.json"
    assert path.is_file(), "report/summary.json is missing"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"total", "open", "by_category"}, sorted(data)
    assert all(isinstance(data[key], int) for key in ("total", "open"))
    assert data == expected()


def test_report_has_the_required_sections():
    path = REPORT / "report.md"
    assert path.is_file(), "report/report.md is missing"
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0].strip() == "# Support tickets, week 42"
    assert "## Summary" in text and "## By category" in text
    assert text.index("## Summary") < text.index("## By category")


def test_report_numbers_agree_with_the_data():
    text = (REPORT / "report.md").read_text(encoding="utf-8")
    want = expected()
    summary = text.split("## Summary", 1)[1].split("## By category", 1)[0]
    numbers = [int(n) for n in re.findall(r"\b\d+\b", summary)]
    assert want["total"] in numbers and want["open"] in numbers, summary
    lines = re.findall(r"^- (\w+): (\d+)\s*$", text.split("## By category", 1)[1], flags=re.M)
    assert {name: int(count) for name, count in lines} == want["by_category"]
    counts = [int(count) for _name, count in lines]
    assert counts == sorted(counts, reverse=True), "categories must be listed largest first"

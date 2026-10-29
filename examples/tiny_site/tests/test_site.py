"""Acceptance for the tiny shop site in site/ (stdlib only; no browser)."""
import json
import re
from html.parser import HTMLParser
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.stylesheets = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append(tag)
        if tag == "link" and "stylesheet" in (attrs.get("rel") or ""):
            self.stylesheets.append(attrs.get("href") or "")
        if tag == "script" and attrs.get("src"):
            self.scripts.append(attrs["src"])


def page():
    parser = Page()
    parser.feed((SITE / "index.html").read_text(encoding="utf-8"))
    return parser


def test_files_exist():
    for name in ("index.html", "styles.css", "app.js", "products.json"):
        assert (SITE / name).is_file(), f"site/{name} is missing"


def test_index_links_css_and_js():
    p = page()
    assert any(href.endswith("styles.css") for href in p.stylesheets), p.stylesheets
    assert any(src.endswith("app.js") for src in p.scripts), p.scripts


def test_index_has_header_and_footer():
    p = page()
    assert "header" in p.tags and "footer" in p.tags, sorted(set(p.tags))


def test_products_json_has_three_named_priced_items():
    products = json.loads((SITE / "products.json").read_text(encoding="utf-8"))
    if isinstance(products, dict):
        products = products.get("products")
    assert isinstance(products, list) and len(products) == 3, products
    for item in products:
        assert isinstance(item, dict) and str(item.get("name") or "").strip(), item
        assert isinstance(item.get("price"), (int, float)) or re.fullmatch(
            r"\$?\d+(\.\d{1,2})?", str(item.get("price") or "")), item


def test_app_js_renders_from_json_and_handles_clicks():
    js = (SITE / "app.js").read_text(encoding="utf-8")
    assert "products.json" in js
    assert "addEventListener" in js


def test_styles_have_a_card_rule():
    css = (SITE / "styles.css").read_text(encoding="utf-8")
    assert re.search(r"\.[\w-]*card[\w-]*\s*[,{]", css), "no .card-style rule in styles.css"

"""Company overview from the latest 10-K, without AI.

Finds the latest annual report, converts it to plain text, locates three sections,
and picks excerpts with simple, transparent rules:

  Item 1   Business               -> what the company does, strategy
  Item 1A  Risk Factors           -> risk-factor headlines
  Item 7   Management's Discussion and Analysis (MD&A)
                                  -> overview, revenue streams, challenges

Every excerpt is quoted word for word from the filing. Nothing is paraphrased or
generated, so a keyword match can pick a paragraph that is not the most important
one. The full sections are always available to read.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from . import sec

ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{doc}"
INDEX = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/index.json"


# ---------------------------------------------------------------------------
# HTML -> plain text, one paragraph per line
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}
    SKIP = {"script", "style", "head", "title"}
    VOID = {"br", "img", "hr", "meta", "link", "input", "col"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[tuple[str, bool]] = []   # (text, is bold/italic)
        self._skip = 0
        self._hidden = 0
        self._emph: list[str] = []

    @staticmethod
    def _is_emphasis(tag: str, style: str) -> bool:
        s = style.replace(" ", "").lower()
        return tag in ("b", "strong", "i", "em") or "font-weight:700" in s or "font-weight:bold" in s \
            or "font-weight:600" in s or "font-style:italic" in s

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        style = dict(attrs).get("style") or ""
        # Inline XBRL hides machine-readable data in display:none blocks
        if tag == "div" and "display:none" in style.replace(" ", "").lower():
            self._hidden += 1
        elif tag == "div" and self._hidden:
            self._hidden += 1
        if tag not in self.VOID and self._is_emphasis(tag, style):
            self._emph.append(tag)
        if tag in self.BLOCK:
            self.parts.append(("\n", False))
        elif tag == "td":
            self.parts.append((" | ", False))

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "div" and self._hidden:
            self._hidden -= 1
        if self._emph and self._emph[-1] == tag:
            self._emph.pop()
        if tag in self.BLOCK:
            self.parts.append(("\n", False))

    def handle_data(self, data):
        if not self._skip and not self._hidden:
            self.parts.append((data, bool(self._emph)))


def _clean(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.replace("\xa0", " ").replace("​", "")).strip()


def html_to_text_and_emphasis(html: str) -> tuple[str, set[str]]:
    """Plain text (one paragraph per line) plus the set of lines that are entirely
    bold or italic, which is how 10-Ks typically mark risk-factor headlines."""
    p = _TextExtractor()
    p.feed(html)
    lines: list[str] = []
    emphasized: set[str] = set()
    cur: list[tuple[str, bool]] = []

    def flush():
        line = _clean("".join(t for t, _ in cur))
        if line:
            total = sum(len(t.strip()) for t, _ in cur)
            emph = sum(len(t.strip()) for t, e in cur if e)
            if total and emph / total >= 0.9:
                emphasized.add(line)
        lines.append(line)
        cur.clear()

    for text, emph in p.parts:
        chunks = text.split("\n")
        for i, chunk in enumerate(chunks):
            if i:
                flush()
            if chunk:
                cur.append((chunk, emph))
    flush()

    out, blank = [], False
    for ln in lines:
        if ln:
            out.append(ln)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip(), emphasized


def html_to_text(html: str) -> str:
    return html_to_text_and_emphasis(html)[0]


# ---------------------------------------------------------------------------
# Locating Items 1, 1A and 7
# ---------------------------------------------------------------------------

# A heading line such as "Item 7.", "ITEM 7 -", "Item 7. Management's Discussion..."
_ITEM_RE = re.compile(r"^\s*item\s*(\d{1,2}[a-c]?)\s*[\.\:\-–—]?\s*(.*)$", re.IGNORECASE)

SECTION_ENDS = {
    "1": ("1a", "1b", "1c", "2"),
    "1a": ("1b", "1c", "2"),
    "7": ("7a", "8"),
}


def find_item_headings(text: str) -> list[tuple[int, str]]:
    """Character offset and item number of every line that looks like an Item heading."""
    out = []
    pos = 0
    for line in text.split("\n"):
        if len(line) < 200:
            m = _ITEM_RE.match(line)
            if m:
                out.append((pos, m.group(1).lower()))
        pos += len(line) + 1
    return out


def extract_section(text: str, item: str) -> str:
    """Text of one Item. The table of contents also lists every Item, so among all
    candidate headings we keep the one that starts the longest section."""
    heads = find_item_headings(text)
    ends = SECTION_ENDS[item]
    best = ""
    for i, (start, num) in enumerate(heads):
        if num != item:
            continue
        stop = next((s for s, n in heads[i + 1:] if n in ends), None)
        body = text[start:stop] if stop else text[start:start + 400_000]
        if len(body) > len(best):
            best = body
    return best.strip()


# ---------------------------------------------------------------------------
# Picking excerpts with simple rules
# ---------------------------------------------------------------------------

BOILERPLATE = re.compile(
    r"should be read in conjunction|is intended to help the reader|forward-looking statement|"
    r"in this report, the terms|the terms .{0,40} (mean|refer)|table of contents|was incorporated in|"
    r"this section should be read|this item generally discusses|is organized as follows|"
    r"all references to fiscal|unless otherwise (stated|indicated|noted)",
    re.IGNORECASE)


def _is_boilerplate(p: str) -> bool:
    # Only the opening of a paragraph is checked, so a real description that ends with a
    # note about the fiscal calendar is kept.
    return bool(BOILERPLATE.search(p[:160]))


def paragraphs(section: str, min_len: int = 160) -> list[str]:
    """Prose paragraphs only: skips headings, tables, bullet lists of contents and boilerplate."""
    out = []
    for p in section.split("\n"):
        p = p.strip()
        if len(p) >= min_len and p.count(" | ") < 2 and not _ITEM_RE.match(p) and not _is_boilerplate(p):
            out.append(p)
    return out


def _is_heading(line: str) -> bool:
    return 3 <= len(line) <= 80 and " | " not in line and not line.endswith((".", ",", ";", ":")) \
        and not re.fullmatch(r"[\d\s]+", line)


def paragraphs_after_heading(section: str, heading_pattern: str, limit: int) -> tuple[str | None, list[str]]:
    """Paragraphs that follow the first heading matching the pattern, up to the next heading
    at the same level. Returns the heading found (or None) and the paragraphs."""
    lines = section.split("\n")
    rx = re.compile(heading_pattern, re.IGNORECASE)
    for i, ln in enumerate(lines):
        if _is_heading(ln) and rx.search(ln):
            body = paragraphs("\n".join(lines[i + 1:i + 80]))
            if body:
                return ln, body[:limit]
    return None, []


def ranked_paragraphs(paras: list[str], words: list[str], limit: int,
                      exclude: str | None = None) -> list[str]:
    """Paragraphs with the most keyword matches, kept in the order they appear in the filing."""
    rx = re.compile(r"\b(" + "|".join(words) + r")", re.IGNORECASE)
    ex = re.compile(exclude, re.IGNORECASE) if exclude else None
    scored = [(len(rx.findall(p)), i) for i, p in enumerate(paras) if not (ex and ex.search(p))]
    top = sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])[:limit]
    return [paras[i] for _, i in sorted(top, key=lambda s: s[1])]


STRATEGY_HEADINGS = r"strateg|priorit|our approach|growth plan|how we (create|win)"
STRATEGY_WORDS = ["our strategy", "strategy is", "strategy to", "strategic priorit", "growth strateg",
                  "business strateg", "we are focused on", "we focus on", "our focus", "our priorities",
                  "we continue to invest"]
# Human-capital and accounting paragraphs often use "strategy" in an unrelated sense
STRATEGY_EXCLUDE = r"employee|workforce|talent|career|compensation|hedg|tax strateg|derivative"
OVERVIEW_HEADINGS = r"^(business\s+)?overview$|^executive\s+(summary|overview)$|^our business$|^general$"
STRUGGLE_HEADINGS = r"challenges|risks and uncertainties|factors (impacting|affecting)|headwinds|" \
                    r"macroeconomic|trends and uncertainties|tariffs"
STRUGGLE_WORDS = ["declin", "decreas", "headwind", "challeng", "pressure", "adverse", "lower demand",
                  "unfavorabl", "impairment", "restructur", "weak", "slowdown", "uncertain", "inflation",
                  "tariff", "softness", "competition"]


def risk_headlines(section: str, emphasized: set[str], limit: int = 12) -> list[str]:
    """Risk-factor headlines: bold or italic sentences in Item 1A (category labels such as
    'Macroeconomic Risks', which do not end in a period, are left out)."""
    out = []
    for line in section.split("\n"):
        line = line.strip()
        if line in emphasized and 40 <= len(line) <= 400 and line.endswith(".") \
                and not BOILERPLATE.search(line) and line not in out:
            out.append(line)
    return out[:limit]


# ---------------------------------------------------------------------------
# Revenue table: the MD&A's own breakdown of sales by product, segment or region
# ---------------------------------------------------------------------------

# "Total net sales", "Total revenues", "TOTAL NIKE, INC. REVENUES", "Net operating revenues"
_TOTAL_ROW = re.compile(r"^\W*(total\b.{0,40}\b(sales|revenues?)|net\s+operating\s+revenues?)\W*$", re.IGNORECASE)


def _cells(line: str) -> list[str]:
    """Clean one table row: drop empty and '$' cells, and join '%' or ')' to the number before it."""
    raw = [c.strip() for c in line.split("|")]
    out: list[str] = []
    for c in raw:
        if not c or c == "$":
            continue
        if c in ("%", ")%", "pts", "bps") and out:
            out[-1] += c
        else:
            out.append(c)
    return out


def _first_number(cells: list[str]) -> float | None:
    for c in cells[1:]:
        m = re.fullmatch(r"\(?\$?([\d,]+(\.\d+)?)\)?", c)
        if m:
            return float(m.group(1).replace(",", ""))
    return None


def _is_header(cells: list[str]) -> bool:
    """A column-header row: only years (e.g. 2026, FISCAL 2025) and words like 'Change'."""
    numeric = [c for c in cells if re.fullmatch(r"\(?\$?[\d,]+(\.\d+)?\)?%?", c)]
    return len(cells) >= 2 and all(re.fullmatch(r"(19|20)\d{2}", c) for c in numeric)


def table_total(rows: list[list[str]]) -> float | None:
    return _first_number(rows[-1]) if rows else None


def revenue_table(section: str) -> list[list[str]]:
    """The breakdown table that ends in a 'Total net sales' / 'Total revenues' line.

    A table only counts if the rows directly above the total add up to it (within 1.5%),
    which confirms it is a real revenue breakdown and not, say, a reconciliation table that
    happens to end with revenue. If several qualify (products, regions, segments), the one
    with the largest total is kept, which is usually the whole company."""
    lines = section.split("\n")
    best: list[list[str]] = []
    best_total = -1.0
    for i, ln in enumerate(lines):
        if " | " not in ln:
            continue
        cells = _cells(ln)
        if not cells or not _TOTAL_ROW.match(cells[0]):
            continue
        total = _first_number(cells)
        if not total or total <= best_total:
            continue
        running, rows, matched = 0.0, [], False
        j = i - 1
        while j >= 0 and i - j <= 25:
            row_line = lines[j]
            if " | " not in row_line and row_line.strip() not in ("", "|"):
                break                                   # left the table
            row = _cells(row_line)
            if row:
                n = _first_number(row)
                if n is not None and len(row) > 1:
                    running += n
                rows.insert(0, row)
                if len(rows) >= 2 and abs(running - total) <= 0.015 * total:
                    matched = True
                    break
            j -= 1
        if not matched:
            continue
        # Add the column header row (the nearest row above with no numbers), if any
        k = j - 1
        while k >= 0 and i - k <= 30 and (" | " in lines[k] or lines[k].strip() in ("", "|")):
            hdr = _cells(lines[k])
            if hdr and _is_header(hdr):
                rows.insert(0, [""] + hdr if len(hdr) < len(cells) else hdr)
                break
            if hdr and _first_number(hdr) is not None:
                break
            k -= 1
        best, best_total = rows + [cells], total
    return best


def mdna_overview(section: str, limit: int = 4) -> list[str]:
    """Paragraphs under the MD&A's overview heading (or its first real paragraphs)."""
    _, paras = paragraphs_after_heading(section, OVERVIEW_HEADINGS, limit)
    return paras or paragraphs(section)[:limit]


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

@dataclass
class CompanyOverview:
    form: str = ""
    filed: str = ""
    period: str = ""
    url: str = ""
    what_it_does: list[str] = field(default_factory=list)
    strategy: list[str] = field(default_factory=list)
    revenue: list[str] = field(default_factory=list)
    revenue_table: list[list[str]] = field(default_factory=list)
    strategy_heading: str | None = None
    struggles_heading: str | None = None
    struggles: list[str] = field(default_factory=list)
    risk_headlines: list[str] = field(default_factory=list)
    mdna_overview: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def latest_annual_report(subs: dict) -> dict | None:
    r = subs.get("filings", {}).get("recent", {})
    for i, form in enumerate(r.get("form", [])):
        if form in ("10-K", "10-K/A"):
            return {"form": form, "accn": r["accessionNumber"][i], "doc": r["primaryDocument"][i],
                    "filed": r["filingDate"][i], "period": r["reportDate"][i]}
    return None


def _fetch_text(url: str, user_agent: str) -> tuple[str, set[str]]:
    resp = sec.requests.get(url, headers={"User-Agent": user_agent}, timeout=60)
    resp.raise_for_status()
    return html_to_text_and_emphasis(resp.text)


REVENUE_WORDS = ["net sales", "revenue from", "revenues from", "generat\\w* (net )?(operating )?revenue",
                 "sources of revenue", "operating segments", "reportable segments", "derives? (its )?revenue"]


def _exhibit13(cik: int, accn: str, user_agent: str) -> str | None:
    """Some companies put the MD&A in Exhibit 13 (the annual report to shareholders)."""
    try:
        idx = sec._get(INDEX.format(cik=cik, accn=accn.replace("-", "")), user_agent)
    except Exception:
        return None
    for item in idx.get("directory", {}).get("item", []):
        name = item.get("name", "").lower()
        if re.search(r"ex-?13", name) and name.endswith((".htm", ".html")):
            return ARCHIVE.format(cik=cik, accn=accn.replace("-", ""), doc=item["name"])
    return None


def build_overview(ticker: str, user_agent: str, ticker_map: dict) -> CompanyOverview:
    ticker = ticker.upper().strip()
    cik = ticker_map[ticker]["cik"]
    subs = sec.get_submissions(cik, user_agent)
    filing = latest_annual_report(subs)
    ov = CompanyOverview()
    if not filing:
        ov.notes.append("No 10-K found. Foreign companies file 20-F reports, which this tab does not read yet.")
        return ov

    accn = filing["accn"].replace("-", "")
    ov.form, ov.filed, ov.period = filing["form"], filing["filed"], filing["period"]
    ov.url = ARCHIVE.format(cik=cik, accn=accn, doc=filing["doc"])
    text, emphasized = _fetch_text(ov.url, user_agent)

    business = extract_section(text, "1")
    risks = extract_section(text, "1a")
    mdna = extract_section(text, "7")

    # MD&A "incorporated by reference" to Exhibit 13
    if len(mdna) < 3000 and re.search(r"incorporated\s+(herein\s+)?by\s+reference", mdna, re.IGNORECASE):
        ex_url = _exhibit13(cik, filing["accn"], user_agent)
        if ex_url:
            ex_text, _ = _fetch_text(ex_url, user_agent)
            m = re.search(r"management['’]s\s+discussion\s+and\s+analysis", ex_text, re.IGNORECASE)
            if m:
                mdna = ex_text[m.start():m.start() + 300_000]
                ov.notes.append("MD&A is in Exhibit 13 (annual report to shareholders); excerpts come from there.")

    ov.sections = {"Item 1. Business": business, "Item 1A. Risk Factors": risks,
                   "Item 7. Management's Discussion and Analysis": mdna}
    for name, body in ov.sections.items():
        if len(body) < 1500:
            ov.notes.append(f"Could not reliably locate {name} in this filing.")

    biz_p = paragraphs(business)
    mdna_p = paragraphs(mdna)

    # What the company does: opening paragraphs of Item 1
    ov.what_it_does = biz_p[:3]

    # Strategy: a "Strategy" / "Priorities" heading if there is one, otherwise the
    # paragraphs that talk most explicitly about strategy
    heading, paras = paragraphs_after_heading(business + "\n" + mdna, STRATEGY_HEADINGS, 3)
    ov.strategy_heading = heading
    ov.strategy = paras or ranked_paragraphs(biz_p + mdna_p, STRATEGY_WORDS, 3, exclude=STRATEGY_EXCLUDE)

    # Revenue streams: the MD&A revenue table plus paragraphs describing where revenue comes from
    ov.revenue_table = revenue_table(mdna)
    ov.revenue = ranked_paragraphs(biz_p + mdna_p, REVENUE_WORDS, 3, exclude=r"impairment|goodwill")

    # MD&A overview, without repeating paragraphs already shown above
    shown = set(ov.what_it_does) | set(ov.strategy)
    ov.mdna_overview = [p for p in mdna_overview(mdna, 6) if p not in shown][:4]

    # Risks: a "Challenges" / "Factors impacting" heading in MD&A if
    # there is one, otherwise the MD&A paragraphs with the most problem-related words
    heading, paras = paragraphs_after_heading(mdna, STRUGGLE_HEADINGS, 4)
    ov.struggles_heading = heading
    ov.struggles = paras or ranked_paragraphs(mdna_p, STRUGGLE_WORDS, 4)

    ov.risk_headlines = risk_headlines(risks, emphasized)
    if not ov.risk_headlines:
        ov.notes.append("Risk-factor headlines could not be identified from the filing's formatting.")
    return ov

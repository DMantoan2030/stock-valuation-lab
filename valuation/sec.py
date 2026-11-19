"""Retrieve and interpret financial data from SEC EDGAR.

Data comes from the SEC's free XBRL "company facts" API, which holds every number
a company has tagged in its 10-K and 10-Q filings. The SEC requires a User-Agent
header that identifies the requester (name + email) and allows up to 10 requests
per second.

Every value returned is a DataPoint that records where it came from (XBRL tag,
form, period, filing date, accession number) and whether it is reliable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT"}
QUARTER_FORMS = {"10-Q", "10-Q/A"}
VALID_FORMS = ANNUAL_FORMS | QUARTER_FORMS

_last_call = [0.0]


def _get(url: str, user_agent: str) -> dict:
    """GET a JSON document from the SEC, staying well under its rate limit."""
    wait = 0.15 - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    resp = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_ticker_map(user_agent: str) -> dict[str, dict]:
    """Map ticker -> {"cik": int, "title": str} for every SEC-registered company."""
    raw = _get(TICKERS_URL, user_agent)
    out = {}
    for row in raw.values():
        out[row["ticker"].upper()] = {"cik": int(row["cik_str"]), "title": row["title"]}
    return out


def get_submissions(cik: int, user_agent: str) -> dict:
    return _get(SUBMISSIONS_URL.format(cik=cik), user_agent)


def get_company_facts(cik: int, user_agent: str) -> dict:
    return _get(FACTS_URL.format(cik=cik), user_agent)


# ---------------------------------------------------------------------------
# DataPoint: a number plus its provenance
# ---------------------------------------------------------------------------

@dataclass
class DataPoint:
    name: str
    value: float | None
    status: str = "ok"            # "ok" | "estimated" | "missing"
    tag: str | None = None
    form: str | None = None
    period: str | None = None     # e.g. "TTM ending 2026-06-27" or "As of 2026-06-27"
    filed: str | None = None
    accn: str | None = None
    note: str = ""
    cik: int | None = None

    @property
    def url(self) -> str | None:
        if not (self.accn and self.cik):
            return None
        return f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{self.accn.replace('-', '')}/"

    def as_row(self) -> dict:
        return {
            "Input": self.name,
            "Value": self.value,
            "Status": self.status,
            "Source": f"SEC {self.form}" if self.form else "",
            "Period": self.period or "",
            "Filed": self.filed or "",
            "XBRL tag": self.tag or "",
            "Note": self.note,
            "Filing link": self.url or "",
        }


# ---------------------------------------------------------------------------
# Low-level fact helpers
# ---------------------------------------------------------------------------

def _d(s: str) -> date:
    return date.fromisoformat(s)


def _days(f: dict) -> int:
    return (_d(f["end"]) - _d(f["start"])).days


def _near(a: date, b: date, tol: int = 10) -> bool:
    return abs((a - b).days) <= tol


def _raw(facts: dict, tag: str, unit: str, taxonomy: str = "us-gaap") -> list[dict]:
    try:
        items = facts["facts"][taxonomy][tag]["units"][unit]
    except KeyError:
        return []
    return [f for f in items if f.get("form") in VALID_FORMS and f.get("val") is not None]


def _dedupe(items: list[dict], duration: bool) -> list[dict]:
    """Keep one fact per period: the most recently filed (captures restatements)."""
    best: dict[tuple, dict] = {}
    for f in items:
        key = (f.get("start"), f["end"]) if duration else (f["end"],)
        if key not in best or f["filed"] > best[key]["filed"]:
            best[key] = f
    return sorted(best.values(), key=lambda f: f["end"])


def _durations(facts: dict, tag: str, unit: str) -> list[dict]:
    return _dedupe([f for f in _raw(facts, tag, unit) if "start" in f], duration=True)


def _instants(facts: dict, tag: str, unit: str = "USD", taxonomy: str = "us-gaap") -> list[dict]:
    return _dedupe([f for f in _raw(facts, tag, unit, taxonomy) if "start" not in f], duration=False)


def _annuals(items: list[dict]) -> list[dict]:
    """Full-fiscal-year facts (52/53-week years included)."""
    by_end: dict[str, dict] = {}
    for f in items:
        if 350 <= _days(f) <= 380:
            if f["end"] not in by_end or f["filed"] > by_end[f["end"]]["filed"]:
                by_end[f["end"]] = f
    return sorted(by_end.values(), key=lambda f: f["end"])


# ---------------------------------------------------------------------------
# Trailing-twelve-month (TTM) construction
# ---------------------------------------------------------------------------

@dataclass
class TTMResult:
    value: float
    end: str
    method: str          # "TTM" or "FY"
    latest: dict         # the fact from the most recent filing used
    note: str = ""


def _ttm_for_tag(items: list[dict]) -> TTMResult | None:
    """TTM = last fiscal year + current year-to-date - prior year-to-date.

    10-Q cash-flow statements only report year-to-date figures, and Q4 is never
    reported on its own, so this formula is the standard way to build TTM.
    """
    annuals = _annuals(items)
    if not annuals:
        return None
    fy = annuals[-1]
    fy_start, fy_end = _d(fy["start"]), _d(fy["end"])

    ytd_candidates = [
        f for f in items
        if _near(_d(f["start"]), fy_end + timedelta(days=1)) and _days(f) < 350 and _d(f["end"]) > fy_end
    ]
    if not ytd_candidates:
        return TTMResult(fy["val"], fy["end"], "FY", fy, f"FY{fy.get('fy', '')} 10-K")
    ytd = max(ytd_candidates, key=lambda f: (f["end"], f["filed"]))
    target_prior_end = _d(ytd["end"]) - timedelta(days=364)
    prior = [
        f for f in items
        if _near(_d(f["start"]), fy_start) and _near(_d(f["end"]), target_prior_end) and _days(f) < 350
    ]
    if not prior:
        return TTMResult(fy["val"], fy["end"], "FY", fy,
                         "Could not find prior-year year-to-date figure; used last fiscal year instead")
    p = max(prior, key=lambda f: f["filed"])
    months = round(_days(ytd) / 30.4)
    value = fy["val"] + ytd["val"] - p["val"]
    note = f"FY{fy.get('fy', '')} 10-K + {months}-month YTD ({ytd['form']}) - prior-year {months}-month YTD"
    return TTMResult(value, ytd["end"], "TTM", ytd, note)


# ---------------------------------------------------------------------------
# Tag fallbacks for each financial concept
# ---------------------------------------------------------------------------

CONCEPTS: dict[str, tuple[str, list[str]]] = {
    "revenue": ("USD", [
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
    ]),
    "operating_income": ("USD", ["OperatingIncomeLoss"]),
    "pretax_income": ("USD", [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ]),
    "income_tax": ("USD", ["IncomeTaxExpenseBenefit"]),
    "net_income": ("USD", ["NetIncomeLoss", "ProfitLoss"]),
    "interest_expense": ("USD", [
        "InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense",
    ]),
    "da": ("USD", [
        "DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization", "Depreciation",
    ]),
    "capex": ("USD", [
        "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements", "PaymentsToAcquireOtherPropertyPlantAndEquipment",
    ]),
    "operating_cash_flow": ("USD", [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ]),
    "eps_diluted": ("USD/shares", ["EarningsPerShareDiluted", "IncomeLossFromContinuingOperationsPerDilutedShare"]),
}

LABELS = {
    "revenue": "Revenue",
    "operating_income": "Operating income (EBIT)",
    "pretax_income": "Pretax income",
    "income_tax": "Income tax expense",
    "net_income": "Net income",
    "interest_expense": "Interest expense",
    "da": "Depreciation & amortization",
    "capex": "Capital expenditures",
    "operating_cash_flow": "Operating cash flow",
    "eps_diluted": "Diluted EPS",
}


def ttm_concept(facts: dict, key: str, cik: int) -> DataPoint:
    """Best TTM value across all candidate tags (the tag with the most recent data wins)."""
    unit, tags = CONCEPTS[key]
    best: tuple[TTMResult, str] | None = None
    for tag in tags:
        r = _ttm_for_tag(_durations(facts, tag, unit))
        if r and (best is None or r.end > best[0].end):
            best = (r, tag)
    if best is None:
        return DataPoint(LABELS[key], None, "missing", note="Not reported in SEC XBRL data", cik=cik)
    r, tag = best
    status = "ok"
    note = r.note
    stale_days = (date.today() - _d(r.end)).days
    if stale_days > 200:
        status = "estimated"
        note += f". Latest data is {stale_days} days old"
    return DataPoint(LABELS[key], float(r.value), status, tag, r.latest["form"],
                     f"{r.method} ending {r.end}", r.latest["filed"], r.latest["accn"], note, cik)


def annual_series(facts: dict, key: str) -> dict[str, float]:
    """Fiscal-year values keyed by period end date, merging all candidate tags."""
    unit, tags = CONCEPTS[key]
    merged: dict[str, dict] = {}
    for tag in tags:
        for f in _annuals(_durations(facts, tag, unit)):
            if f["end"] not in merged or f["filed"] > merged[f["end"]]["filed"]:
                merged[f["end"]] = f
    return {end: float(f["val"]) for end, f in sorted(merged.items())}


# ---------------------------------------------------------------------------
# Balance-sheet items
# ---------------------------------------------------------------------------

def _value_at(facts: dict, tag: str, when: date, unit: str = "USD") -> dict | None:
    for f in reversed(_instants(facts, tag, unit)):
        if _near(_d(f["end"]), when, 5):
            return f
    return None


def _first_at(facts: dict, tags: list[str], when: date) -> tuple[float, dict, str] | None:
    for t in tags:
        f = _value_at(facts, t, when)
        if f:
            return float(f["val"]), f, t
    return None


def latest_balance_date(facts: dict) -> date | None:
    dates = [_d(f["end"]) for t in ("Assets", "LiabilitiesAndStockholdersEquity") for f in _instants(facts, t)]
    return max(dates) if dates else None


CASH_TAGS = ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"]
STI_TAGS = ["MarketableSecuritiesCurrent", "ShortTermInvestments",
            "AvailableForSaleSecuritiesDebtSecuritiesCurrent", "OtherShortTermInvestments"]


def cash_at(facts: dict, when: date) -> tuple[float | None, dict | None, list[str], str]:
    c = _first_at(facts, CASH_TAGS, when)
    if not c:
        return None, None, [], "Cash not reported"
    s = _first_at(facts, STI_TAGS, when)
    value = c[0] + (s[0] if s else 0.0)
    tags = [c[2]] + ([s[2]] if s else [])
    note = "Cash & equivalents" + (" + short-term investments" if s else "") + "; long-term investments excluded"
    return value, c[1], tags, note


def debt_at(facts: dict, when: date) -> tuple[float | None, dict | None, list[str], str, str]:
    """Total financial debt (current + long-term). Operating leases are excluded."""
    used: list[str] = []
    src = None
    status = "ok"

    # Current portion
    current = _first_at(facts, ["DebtCurrent"], when)
    if current:
        cur_val, src, t = current
        used.append(t)
    else:
        cur_val = 0.0
        ltc = _first_at(facts, ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"], when)
        stb = _first_at(facts, ["ShortTermBorrowings", "CommercialPaper", "OtherShortTermBorrowings"], when)
        for part in (ltc, stb):
            if part:
                cur_val += part[0]
                src = src or part[1]
                used.append(part[2])

    # Long-term portion
    lt = _first_at(facts, ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
                           "LongTermDebtAndFinanceLeaseObligationsNoncurrent", "LongTermNotesPayable",
                           "SeniorLongTermNotes"], when)
    if lt:
        lt_val, src2, t = lt
        used.append(t)
        src = src or src2
    else:
        total = _first_at(facts, ["LongTermDebt"], when)
        if total:
            # LongTermDebt includes the current portion, which is already counted above.
            ltc = _first_at(facts, ["LongTermDebtCurrent"], when)
            lt_val = total[0] - (ltc[0] if ltc else 0.0)
            used.append(total[2])
            src = src or total[1]
        else:
            lt_val = 0.0

    if not used:
        return None, None, [], "No debt tags found", "missing"
    if not lt:
        status = "estimated"
    note = "Current + long-term debt (incl. commercial paper); operating leases excluded"
    return cur_val + lt_val, src, used, note, status


def nwc_at(facts: dict, when: date) -> float | None:
    """Operating working capital = (current assets - cash - ST investments) - (current liabilities - current debt)."""
    ca = _value_at(facts, "AssetsCurrent", when)
    cl = _value_at(facts, "LiabilitiesCurrent", when)
    if not (ca and cl):
        return None
    cash, _, _, _ = cash_at(facts, when)
    cur_debt = 0.0
    for t in ("DebtCurrent",):
        f = _value_at(facts, t, when)
        if f:
            cur_debt = float(f["val"])
            break
    else:
        for t in ("LongTermDebtCurrent", "CommercialPaper", "ShortTermBorrowings"):
            f = _value_at(facts, t, when)
            if f:
                cur_debt += float(f["val"])
    return (float(ca["val"]) - (cash or 0.0)) - (float(cl["val"]) - cur_debt)


def diluted_shares(facts: dict, cik: int) -> DataPoint:
    """Weighted-average diluted shares from the most recent period (shortest period wins on ties,
    so a 3-month figure beats a year-to-date one). Fallback: cover-page shares."""
    pool = [f for f in _durations(facts, "WeightedAverageNumberOfDilutedSharesOutstanding", "shares")
            if _days(f) <= 380]
    cover = _instants(facts, "EntityCommonStockSharesOutstanding", "shares", "dei")
    if pool:
        f = max(pool, key=lambda x: (x["end"], -_days(x), x["filed"]))
        kind = "quarter" if _days(f) <= 100 else "fiscal year" if _days(f) >= 350 else "period"
        dp = DataPoint("Diluted shares outstanding", float(f["val"]), "ok",
                       "WeightedAverageNumberOfDilutedSharesOutstanding", f["form"],
                       f"{kind} ending {f['end']}", f["filed"], f["accn"],
                       f"Weighted-average diluted shares, latest reported {kind}", cik)
        if cover:
            c = cover[-1]
            if c["end"] > f["end"] and abs(c["val"] / f["val"] - 1) > 0.05:
                dp.status = "estimated"
                dp.note += (f". Cover-page share count ({c['val'] / 1e6:,.0f}M as of {c['end']}) differs by "
                            f"more than 5%: possible split, buyback, or issuance")
        return dp
    if cover:
        c = cover[-1]
        return DataPoint("Diluted shares outstanding", float(c["val"]), "estimated",
                         "dei:EntityCommonStockSharesOutstanding", c["form"], f"As of {c['end']}",
                         c["filed"], c["accn"], "Basic shares from filing cover page (diluted count not found)", cik)
    return DataPoint("Diluted shares outstanding", None, "missing", note="Share count not reported", cik=cik)


# ---------------------------------------------------------------------------
# Company snapshot: everything the model needs
# ---------------------------------------------------------------------------

FINANCIAL_SIC_RANGES = [(6000, 6299, "bank or financial company"), (6300, 6499, "insurance company"),
                        (6798, 6798, "REIT")]


@dataclass
class CompanySnapshot:
    ticker: str
    cik: int
    name: str
    sic: str
    sic_description: str
    fiscal_year_end: str
    points: dict[str, DataPoint] = field(default_factory=dict)
    history: dict[str, dict[str, float]] = field(default_factory=dict)
    defaults: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    dcf_supported: bool = True

    def v(self, key: str) -> float | None:
        p = self.points.get(key)
        return p.value if p else None


def _avg(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_snapshot(ticker: str, user_agent: str, ticker_map: dict) -> CompanySnapshot:
    ticker = ticker.upper().strip()
    if ticker not in ticker_map:
        raise ValueError(f"'{ticker}' is not in the SEC's list of registered companies.")
    cik = ticker_map[ticker]["cik"]
    subs = get_submissions(cik, user_agent)
    facts = get_company_facts(cik, user_agent)

    snap = CompanySnapshot(ticker, cik, subs.get("name") or ticker_map[ticker]["title"],
                           str(subs.get("sic") or ""), subs.get("sicDescription") or "",
                           subs.get("fiscalYearEnd") or "")

    recent_forms = set(subs.get("filings", {}).get("recent", {}).get("form", [])[:40])
    if "us-gaap" not in facts.get("facts", {}):
        snap.flags.append("No U.S. GAAP financial data found. Foreign filers (20-F / IFRS) are not supported.")
        snap.dcf_supported = False
        return snap
    if recent_forms & {"20-F", "40-F"} and not recent_forms & {"10-K", "10-Q"}:
        snap.flags.append("This company files 20-F/40-F reports as a foreign issuer; data may be incomplete.")

    try:
        sic = int(snap.sic)
        for lo, hi, label in FINANCIAL_SIC_RANGES:
            if lo <= sic <= hi:
                snap.dcf_supported = False
                snap.flags.append(f"This looks like a {label} (SIC {sic}). A free-cash-flow DCF does not fit its "
                                  f"business model, so the DCF is skipped. The P/E analysis still applies.")
    except ValueError:
        pass

    # --- Income statement and cash flow (TTM) ---
    for key in CONCEPTS:
        snap.points[key] = ttm_concept(facts, key, cik)

    ebit = snap.points["operating_income"]
    if ebit.value is None and snap.v("pretax_income") is not None:
        est = snap.v("pretax_income") + (snap.v("interest_expense") or 0.0)
        p = snap.points["pretax_income"]
        snap.points["operating_income"] = DataPoint(
            LABELS["operating_income"], est, "estimated", p.tag, p.form, p.period, p.filed, p.accn,
            "Operating income not reported; estimated as pretax income + interest expense", cik)

    # --- Balance sheet (latest filing) ---
    bdate = latest_balance_date(facts)
    if bdate:
        cash, src, tags, note = cash_at(facts, bdate)
        snap.points["cash"] = DataPoint("Cash & short-term investments", cash, "ok" if cash is not None else "missing",
                                        " + ".join(tags), src["form"] if src else None, f"As of {bdate}",
                                        src["filed"] if src else None, src["accn"] if src else None, note, cik)
        debt, src, tags, note, status = debt_at(facts, bdate)
        if debt is None:
            snap.points["debt"] = DataPoint("Total debt", 0.0, "missing", note="No debt reported; assumed 0", cik=cik)
        else:
            snap.points["debt"] = DataPoint("Total debt", debt, status, " + ".join(tags),
                                            src["form"] if src else None, f"As of {bdate}",
                                            src["filed"] if src else None, src["accn"] if src else None, note, cik)
    else:
        snap.points["cash"] = DataPoint("Cash & short-term investments", None, "missing", cik=cik)
        snap.points["debt"] = DataPoint("Total debt", None, "missing", cik=cik)

    snap.points["shares"] = diluted_shares(facts, cik)

    # Free cash flow (reported basis) = operating cash flow - capex
    ocf, capex = snap.points["operating_cash_flow"], snap.points["capex"]
    if ocf.value is not None and capex.value is not None:
        snap.points["fcf"] = DataPoint("Free cash flow (CFO - capex)", ocf.value - capex.value,
                                        "ok" if ocf.status == capex.status == "ok" else "estimated",
                                        f"{ocf.tag} - {capex.tag}", ocf.form, ocf.period, ocf.filed, ocf.accn,
                                        "Operating cash flow minus capital expenditures", cik)
    else:
        snap.points["fcf"] = DataPoint("Free cash flow (CFO - capex)", None, "missing", cik=cik)

    # --- History for default assumptions ---
    for key in ("revenue", "operating_income", "da", "capex", "income_tax", "pretax_income"):
        snap.history[key] = annual_series(facts, key)
    rev_hist = snap.history["revenue"]
    years = list(rev_hist.keys())[-4:]
    nwc_hist = {}
    for end in years:
        n = nwc_at(facts, _d(end))
        if n is not None:
            nwc_hist[end] = n
    snap.history["nwc"] = nwc_hist

    snap.defaults = _default_assumptions(snap)
    _quality_flags(snap)
    return snap


def _default_assumptions(snap: CompanySnapshot) -> dict[str, float]:
    d: dict[str, float] = {}
    rev = snap.history.get("revenue", {})
    rv = list(rev.values())
    last3 = list(rev.keys())[-3:]

    # Revenue growth: blend of last-year growth and 3-year CAGR, kept in a sensible range.
    g_last = rv[-1] / rv[-2] - 1 if len(rv) >= 2 and rv[-2] > 0 else None
    cagr = (rv[-1] / rv[-4]) ** (1 / 3) - 1 if len(rv) >= 4 and rv[-4] > 0 and rv[-1] > 0 else None
    g = _avg([g_last, cagr])
    d["hist_growth"] = g if g is not None else 0.05
    d["growth_y1"] = _clip(d["hist_growth"], -0.10, 0.25)

    rev_ttm = snap.v("revenue")
    ebit = snap.v("operating_income")
    d["op_margin"] = _clip(ebit / rev_ttm, -0.5, 0.8) if rev_ttm and ebit is not None else 0.10

    def pct_of_rev(key: str) -> float | None:
        h = snap.history.get(key, {})
        vals = [h[e] / rev[e] for e in last3 if e in h and rev.get(e)]
        return _avg(vals)

    d["da_pct"] = _clip(pct_of_rev("da") or 0.03, 0.0, 0.3)
    d["capex_pct"] = _clip(pct_of_rev("capex") or 0.04, 0.0, 0.4)

    nwc = snap.history.get("nwc", {})
    nwc_ratios = [nwc[e] / rev[e] for e in last3 if e in nwc and rev.get(e)]
    d["nwc_pct"] = _clip(_avg(nwc_ratios) if nwc_ratios else 0.05, -0.3, 0.5)

    tax, pre = snap.v("income_tax"), snap.v("pretax_income")
    eff = tax / pre if tax is not None and pre and pre > 0 else None
    d["tax_rate"] = eff if eff is not None and 0.0 <= eff <= 0.35 else 0.21
    d["tax_rate_was_estimated"] = 0.0 if eff is not None and 0.0 <= eff <= 0.35 else 1.0

    debt = snap.v("debt")
    interest = snap.v("interest_expense")
    interest_ok = snap.points["interest_expense"].status == "ok"   # stale interest data is not used
    d["cost_of_debt"] = (_clip(interest / debt, 0.02, 0.15)
                         if interest_ok and interest and debt and debt > 0 else None)
    return d


def _quality_flags(snap: CompanySnapshot) -> None:
    consequences = {
        "da": "a default % of revenue is used",
        "capex": "a default % of revenue is used",
        "interest_expense": "cost of debt defaults to risk-free rate + 1.5%",
        "income_tax": "a 21% tax rate is used",
        "pretax_income": "a 21% tax rate is used",
    }
    for key, p in snap.points.items():
        if p.value is None:
            extra = f"; {consequences[key]}" if key in consequences else ""
            snap.flags.append(f"{p.name}: missing from SEC data{extra}.")
        elif p.status == "estimated":
            extra = (". Not used; cost of debt defaults to risk-free rate + 1.5%"
                     if key == "interest_expense" else "")
            snap.flags.append(f"{p.name}: estimated. {p.note}{extra}")
    if snap.defaults.get("tax_rate_was_estimated"):
        snap.flags.append("Effective tax rate was unavailable or outside 0 to 35%; 21% U.S. federal rate used.")
    if len(snap.history.get("revenue", {})) < 4:
        snap.flags.append("Fewer than 4 years of revenue history; growth defaults are less reliable.")
    if snap.v("operating_income") is not None and snap.v("operating_income") < 0:
        snap.flags.append("Operating income is negative; DCF results are highly sensitive to the margin assumption.")

"""Watchlist scan: value every ticker in a list with default assumptions and flag
upside and financial distress.

Upside comes from the same DCF and reverse DCF as the single-company view. Distress
uses standard credit-analysis signals computed from SEC data:

  Altman Z-score   = 1.2 x working capital / assets + 1.4 x retained earnings / assets
                     + 3.3 x EBIT / assets + 0.6 x market cap / liabilities + 1.0 x sales / assets
                     (below 1.81: distress zone; 1.81 to 2.99: grey zone; above 2.99: safe zone)
  Interest coverage = EBIT / interest expense       (below 1.5x is a warning sign)
  Net debt / EBITDA = (debt - cash) / (EBIT + D&A)  (above 4x is a warning sign)
  Free cash flow and operating income below zero

The Z-score was designed for manufacturers and does not apply to banks, insurers or REITs.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

from .defaults import default_inputs, default_wacc
from .model import reverse_dcf, run_dcf, trailing_pe
from .sec import CompanySnapshot

MAX_TICKERS = 1000

# Approximate GICS-style sector from the SEC's 4-digit SIC industry code.
# Checked in order; the first matching range wins, so narrow exceptions come first.
_SECTOR_RANGES: list[tuple[int, int, str]] = [
    (1220, 1229, "Energy"), (1300, 1399, "Energy"), (2900, 2999, "Energy"),
    (2830, 2836, "Health Care"), (3840, 3851, "Health Care"), (5122, 5122, "Health Care"),
    (8000, 8099, "Health Care"),
    (2840, 2844, "Consumer Staples"), (5400, 5499, "Consumer Staples"), (5912, 5912, "Consumer Staples"),
    (5331, 5331, "Consumer Staples"), (5141, 5149, "Consumer Staples"),
    (3570, 3579, "Information Technology"), (3600, 3629, "Information Technology"),
    (3640, 3699, "Information Technology"), (3800, 3839, "Information Technology"),
    (7370, 7379, "Information Technology"),
    (3630, 3639, "Consumer Discretionary"), (3710, 3716, "Consumer Discretionary"),
    (3750, 3751, "Consumer Discretionary"), (3790, 3799, "Consumer Discretionary"),
    (3860, 3999, "Consumer Discretionary"),
    (2700, 2799, "Communication Services"), (4800, 4899, "Communication Services"),
    (7810, 7829, "Communication Services"),
    (4900, 4999, "Utilities"),
    (6798, 6798, "Real Estate"), (6500, 6599, "Real Estate"),
    (6000, 6499, "Financials"), (6700, 6799, "Financials"),
    (100, 999, "Consumer Staples"), (1000, 1499, "Materials"), (1500, 1799, "Industrials"),
    (2000, 2199, "Consumer Staples"), (2200, 2399, "Consumer Discretionary"), (2400, 2499, "Materials"),
    (2500, 2599, "Consumer Discretionary"), (2600, 2699, "Materials"), (2800, 2899, "Materials"),
    (3000, 3199, "Consumer Discretionary"), (3200, 3399, "Materials"), (3400, 3599, "Industrials"),
    (3700, 3799, "Industrials"), (4000, 4799, "Industrials"), (5000, 5199, "Industrials"),
    (5200, 5999, "Consumer Discretionary"), (7000, 7299, "Consumer Discretionary"),
    (7300, 7399, "Industrials"), (7800, 7999, "Consumer Discretionary"),
    (8200, 8299, "Consumer Discretionary"), (8700, 8799, "Industrials"),
]


def sector_from_sic(sic: str | int | None) -> str:
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    for lo, hi, sector in _SECTOR_RANGES:
        if lo <= code <= hi:
            return sector
    return "Other"

_TICKER = re.compile(r"^[A-Z]{1,5}([.\-][A-Z]{1,2})?$")


def parse_tickers(text: str, ticker_map: dict) -> tuple[list[str], list[str]]:
    """Tickers from a thinkorswim watchlist/positions export or any pasted list.

    Uses the 'Symbol' column when there is one, otherwise the first cell of each row,
    and also accepts comma/space separated text. Options (".AAPL..."), futures ("/ES")
    and anything not registered with the SEC are skipped. Returns (tickers, skipped)."""
    rows = list(csv.reader(io.StringIO(text)))
    sym_col = None
    for row in rows[:30]:
        for i, cell in enumerate(row):
            if cell.strip().lower() in ("symbol", "ticker", "instrument"):
                sym_col = i
                break
        if sym_col is not None:
            break

    candidates: list[str] = []
    if sym_col is not None:
        candidates = [r[sym_col] for r in rows if len(r) > sym_col]
    elif len(rows) > 1 or (rows and len(rows[0]) == 1):
        candidates = [r[0] for r in rows if r]
    if len(candidates) <= 1:                       # plain "AAPL, MSFT KO" text
        candidates = re.split(r"[\s,;]+", text)

    tickers, skipped = [], []
    for c in candidates:
        t = c.strip().upper().lstrip("$")
        if not t or t in ("SYMBOL", "TICKER", "INSTRUMENT"):
            continue
        if t.startswith((".", "/")) or not _TICKER.match(t):
            continue                               # options, futures, headers, totals
        t_sec = t.replace(".", "-")                # BRK.B -> BRK-B as the SEC lists it
        key = t if t in ticker_map else t_sec if t_sec in ticker_map else None
        if key is None:
            if t not in skipped:
                skipped.append(t)
        elif key not in tickers:
            tickers.append(key)
    return tickers, skipped


# ---------------------------------------------------------------------------
# Distress metrics
# ---------------------------------------------------------------------------

def altman_z(snap: CompanySnapshot, market_cap: float | None) -> float | None:
    b = snap.balance
    ta, tl = b.get("total_assets"), b.get("total_liabilities")
    ca, cl, re_ = b.get("current_assets"), b.get("current_liabilities"), b.get("retained_earnings")
    ebit, sales = snap.v("operating_income"), snap.v("revenue")
    if not ta or not tl or None in (ca, cl, re_, ebit, sales) or not market_cap:
        return None
    return (1.2 * (ca - cl) / ta + 1.4 * re_ / ta + 3.3 * ebit / ta
            + 0.6 * market_cap / tl + 1.0 * sales / ta)


def z_zone(z: float | None) -> str:
    if z is None:
        return "n/a"
    return "distress" if z < 1.81 else "grey" if z <= 2.99 else "safe"


@dataclass
class ScanRow:
    ticker: str
    name: str = ""
    price: float | None = None
    market_cap: float | None = None
    dcf_value: float | None = None
    upside: float | None = None
    implied_growth: float | None = None
    cagr_3y: float | None = None
    growth_gap: float | None = None          # implied growth minus 3-year CAGR
    pe: float | None = None
    altman_z: float | None = None
    interest_coverage: float | None = None
    net_debt_ebitda: float | None = None
    fcf_margin: float | None = None
    distress_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    sector: str = ""
    industry: str = ""

    @property
    def distress_score(self) -> int:
        return len(self.distress_flags)


def analyze(snap: CompanySnapshot, price: float | None, beta: float | None, rf: float) -> ScanRow:
    row = ScanRow(snap.ticker, snap.name, price)
    row.sector = sector_from_sic(snap.sic)
    row.industry = snap.sic_description.title() if snap.sic_description else ""
    d = snap.defaults
    row.cagr_3y = d.get("cagr_3y")
    shares = snap.v("shares")
    row.market_cap = price * shares if price and shares else None
    row.pe = trailing_pe(price, snap.v("eps_diluted"))

    # --- Upside (same default DCF as the single-company view) ---
    if not price:
        row.notes.append("no live price")
    elif not snap.dcf_supported:
        row.notes.append("DCF not suited to this industry")
    elif snap.v("revenue") and shares:
        try:
            wacc, _ = default_wacc(snap, price, beta, rf)
            inputs = default_inputs(snap, wacc)
            row.dcf_value = run_dcf(inputs).per_share
            row.upside = row.dcf_value / price - 1
            if row.dcf_value < 0:
                row.notes.append("negative DCF value: with default assumptions, current margins and debt "
                                 "leave no value for shareholders (common for loss-makers or companies "
                                 "with large finance arms); adjust assumptions in the single-company view")
            row.implied_growth = reverse_dcf(inputs, price).implied_growth
            if row.implied_growth is not None and row.cagr_3y is not None:
                row.growth_gap = row.implied_growth - row.cagr_3y
        except ValueError as e:
            row.notes.append(str(e))
    if beta is None and price:
        row.notes.append("beta assumed 1.0")

    # --- Distress signals ---
    ebit, revenue = snap.v("operating_income"), snap.v("revenue")
    interest = snap.v("interest_expense") if snap.points.get("interest_expense") and \
        snap.points["interest_expense"].status == "ok" else None
    da = snap.v("da") or 0.0
    fcf = snap.v("fcf")
    debt, cash = snap.v("debt") or 0.0, snap.v("cash") or 0.0

    if snap.dcf_supported:
        row.altman_z = altman_z(snap, row.market_cap)
        if row.altman_z is not None and row.altman_z < 1.81:
            row.distress_flags.append(f"Altman Z {row.altman_z:.1f} (distress zone)")
    if ebit is not None and interest and interest > 0:
        row.interest_coverage = ebit / interest
        if row.interest_coverage < 1.5:
            row.distress_flags.append(f"interest coverage {row.interest_coverage:.1f}x")
    if ebit is not None and ebit + da > 0:
        row.net_debt_ebitda = (debt - cash) / (ebit + da)
        if row.net_debt_ebitda > 4:
            row.distress_flags.append(f"net debt/EBITDA {row.net_debt_ebitda:.1f}x")
    elif ebit is not None and debt > cash:
        row.distress_flags.append("negative EBITDA with net debt")
    if fcf is not None and revenue:
        row.fcf_margin = fcf / revenue
        if fcf < 0:
            row.distress_flags.append("negative free cash flow")
    if ebit is not None and ebit < 0:
        row.distress_flags.append("operating loss")
    return row

"""Valuation math: five-year DCF, reverse DCF, sensitivity table, P/E and WACC.

This module has no network access, so every formula can be tested by hand.

Free cash flow to the firm (FCFF) starts from EBIT:

    FCF = EBIT x (1 - Tax Rate) + D&A - Change in Working Capital - CapEx
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace

FCF_FORMULA = "FCF = EBIT × (1 − Tax Rate) + D&A − Change in Working Capital − CapEx"

MIN_SPREAD = 0.005  # WACC must exceed terminal growth by at least 0.5 percentage points


@dataclass
class DCFInputs:
    revenue: float            # base-year (TTM) revenue
    margin_start: float       # current EBIT margin
    margin_target: float      # EBIT margin reached in the final projection year
    growth_y1: float          # revenue growth in year 1
    growth_y5: float          # revenue growth in the final year (straight line in between)
    tax_rate: float
    da_pct: float             # D&A as % of revenue
    capex_pct: float          # capital expenditures as % of revenue
    nwc_pct: float            # net working capital as % of revenue (change in NWC = nwc_pct x change in revenue)
    wacc: float
    terminal_growth: float
    cash: float
    debt: float
    shares: float
    years: int = 5


@dataclass
class DCFResult:
    rows: list[dict]          # one dict per projection year
    terminal_value: float
    pv_terminal: float
    pv_fcff: float
    enterprise_value: float
    equity_value: float
    per_share: float

    @property
    def terminal_share(self) -> float:
        return self.pv_terminal / self.enterprise_value if self.enterprise_value else float("nan")


def growth_path(g1: float, gn: float, n: int) -> list[float]:
    """Revenue growth for each year, moving in a straight line from g1 (year 1) to gn (year n)."""
    if n == 1:
        return [gn]
    return [g1 + (gn - g1) * i / (n - 1) for i in range(n)]


def margin_path(start: float, target: float, n: int) -> list[float]:
    """EBIT margin for each year, moving in a straight line from today's margin to the target by year n."""
    return [start + (target - start) * t / n for t in range(1, n + 1)]


def run_dcf(inp: DCFInputs, constant_growth: float | None = None) -> DCFResult:
    if inp.wacc - inp.terminal_growth < MIN_SPREAD:
        raise ValueError("WACC must exceed terminal growth by at least 0.5 percentage points.")
    if inp.shares <= 0:
        raise ValueError("Share count must be positive.")

    n = inp.years
    growths = [constant_growth] * n if constant_growth is not None else growth_path(inp.growth_y1, inp.growth_y5, n)
    margins = margin_path(inp.margin_start, inp.margin_target, n)

    rows = []
    prev_rev = inp.revenue
    pv_sum = 0.0
    for t in range(1, n + 1):
        revenue = prev_rev * (1 + growths[t - 1])
        ebit = revenue * margins[t - 1]
        taxes = ebit * inp.tax_rate if ebit > 0 else 0.0   # no tax shield on operating losses
        nopat = ebit - taxes
        da = revenue * inp.da_pct
        d_nwc = inp.nwc_pct * (revenue - prev_rev)
        capex = revenue * inp.capex_pct
        fcff = nopat + da - d_nwc - capex   # nopat = EBIT x (1 - Tax Rate)
        discount = 1 / (1 + inp.wacc) ** t
        pv = fcff * discount
        pv_sum += pv
        rows.append({
            "Year": t, "Revenue growth": growths[t - 1], "Revenue": revenue, "EBIT margin": margins[t - 1],
            "EBIT": ebit, "Taxes on EBIT": -taxes, "EBIT × (1 − Tax Rate)": nopat, "+ D&A": da,
            "− Change in Working Capital": -d_nwc, "− CapEx": -capex, "FCF": fcff, "Discount factor": discount, "PV of FCF": pv,
        })
        prev_rev = revenue

    last_fcff = rows[-1]["FCF"]
    tv = last_fcff * (1 + inp.terminal_growth) / (inp.wacc - inp.terminal_growth)
    pv_tv = tv * rows[-1]["Discount factor"]
    ev = pv_sum + pv_tv
    equity = ev - inp.debt + inp.cash
    return DCFResult(rows, tv, pv_tv, pv_sum, ev, equity, equity / inp.shares)


# ---------------------------------------------------------------------------
# Reverse DCF
# ---------------------------------------------------------------------------

@dataclass
class ReverseDCFResult:
    implied_growth: float | None
    message: str


def reverse_dcf(inp: DCFInputs, price: float, lo: float = -0.5, hi: float = 1.0) -> ReverseDCFResult:
    """Find the constant annual revenue growth (years 1-5) at which DCF value per share = price.

    All other assumptions stay fixed. Value is not guaranteed to rise with growth (for example,
    when margins are negative or working-capital needs are large), so we scan the range first
    and then narrow down the first crossing with bisection.
    """
    def gap(g: float) -> float:
        return run_dcf(inp, constant_growth=g).per_share - price

    steps = 300
    grid = [lo + (hi - lo) * i / steps for i in range(steps + 1)]
    values = [gap(g) for g in grid]
    for i in range(steps):
        a, b = values[i], values[i + 1]
        if a == 0:
            return ReverseDCFResult(grid[i], "ok")
        if a * b < 0:
            x0, x1, f0 = grid[i], grid[i + 1], a
            for _ in range(60):
                mid = (x0 + x1) / 2
                fm = gap(mid)
                if f0 * fm <= 0:
                    x1 = mid
                else:
                    x0, f0 = mid, fm
            return ReverseDCFResult((x0 + x1) / 2, "ok")
    lowest, highest = min(values), max(values)
    if lowest > 0:
        msg = (f"Even with {lo:.0%} annual revenue growth the DCF value exceeds the price, "
               "so the market implies a decline steeper than this model tests.")
    elif highest < 0:
        msg = (f"Even {hi:.0%} annual revenue growth does not justify the price under the current margin "
               "and WACC assumptions. Try a higher target margin or lower WACC.")
    else:
        msg = "No solution found in the tested range."
    return ReverseDCFResult(None, msg)


# ---------------------------------------------------------------------------
# Sensitivity table
# ---------------------------------------------------------------------------

def sensitivity(inp: DCFInputs, waccs: list[float], tgs: list[float]) -> list[list[float | None]]:
    """Value per share for each (WACC, terminal growth) pair. None where WACC is too close to growth."""
    table = []
    for w in waccs:
        row = []
        for g in tgs:
            if w - g < MIN_SPREAD:
                row.append(None)
            else:
                row.append(run_dcf(replace(inp, wacc=w, terminal_growth=g)).per_share)
        table.append(row)
    return table


# ---------------------------------------------------------------------------
# P/E
# ---------------------------------------------------------------------------

PE_MAX = 100.0


def trailing_pe(price: float | None, eps: float | None) -> float | None:
    if not price or eps is None or eps <= 0:
        return None
    return price / eps


@dataclass
class PeerPEResult:
    median_pe: float | None
    implied_value: float | None
    used: dict[str, float]
    excluded: dict[str, str]


def peer_pe_value(eps: float | None, peer_pes: dict[str, float | None]) -> PeerPEResult:
    """Company EPS x median peer P/E. Peers with negative/missing EPS or P/E above 100 are excluded."""
    used, excluded = {}, {}
    for t, pe in peer_pes.items():
        if pe is None or math.isnan(pe):
            excluded[t] = "negative or missing EPS"
        elif pe > PE_MAX:
            excluded[t] = f"P/E above {PE_MAX:.0f} (outlier)"
        else:
            used[t] = pe
    median = statistics.median(used.values()) if used else None
    value = median * eps if median is not None and eps is not None and eps > 0 else None
    return PeerPEResult(median, value, used, excluded)


# ---------------------------------------------------------------------------
# WACC
# ---------------------------------------------------------------------------

@dataclass
class WACCResult:
    wacc: float
    cost_of_equity: float
    after_tax_cost_of_debt: float
    weight_equity: float
    weight_debt: float


def wacc_capm(risk_free: float, beta: float, erp: float, pretax_cost_of_debt: float,
              tax_rate: float, market_cap: float, debt: float) -> WACCResult:
    """WACC = E/(D+E) x (rf + beta x ERP) + D/(D+E) x Kd x (1 - tax)."""
    ke = risk_free + beta * erp
    kd = pretax_cost_of_debt * (1 - tax_rate)
    total = market_cap + max(debt, 0.0)
    we = market_cap / total if total > 0 else 1.0
    wd = 1 - we
    return WACCResult(we * ke + wd * kd, ke, kd, we, wd)

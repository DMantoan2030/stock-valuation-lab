"""Default valuation assumptions, shared by the single-company view, the competitor
table and the watchlist scan, so every part of the app values a company the same way."""
from __future__ import annotations

from .model import DCFInputs, wacc_capm
from .sec import CompanySnapshot

DEFAULT_ERP = 0.05
DEFAULT_TG = 0.025


def default_wacc(snap: CompanySnapshot, price: float | None, beta: float | None,
                 rf: float, erp: float = DEFAULT_ERP) -> tuple[float, dict]:
    tax = snap.defaults.get("tax_rate", 0.21)
    kd = snap.defaults.get("cost_of_debt") or rf + 0.015
    shares = snap.v("shares") or 0
    mcap = (price or 0) * shares
    b = beta if beta else 1.0
    w = wacc_capm(rf, b, erp, kd, tax, mcap, snap.v("debt") or 0.0)
    wacc = min(max(w.wacc, 0.05), 0.15)
    return wacc, {"beta": b, "beta_estimated": beta is None, "kd": kd, "detail": w, "clipped": wacc != w.wacc}


def default_inputs(snap: CompanySnapshot, wacc: float, tg: float = DEFAULT_TG) -> DCFInputs:
    d = snap.defaults
    g1 = d.get("growth_y1", 0.05)
    return DCFInputs(
        revenue=snap.v("revenue") or 0.0, margin_start=d.get("op_margin", 0.1), margin_target=d.get("op_margin", 0.1),
        growth_y1=g1, growth_y5=(g1 + tg) / 2, tax_rate=d.get("tax_rate", 0.21), da_pct=d.get("da_pct", 0.03),
        capex_pct=d.get("capex_pct", 0.04), nwc_pct=d.get("nwc_pct", 0.05), wacc=wacc, terminal_growth=tg,
        cash=snap.v("cash") or 0.0, debt=snap.v("debt") or 0.0, shares=snap.v("shares") or 0.0,
    )

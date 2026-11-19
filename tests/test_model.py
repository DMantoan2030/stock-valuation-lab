"""Checks the valuation math against numbers worked out by hand.

Run with:  python -m pytest
"""
import math

import pytest

from valuation.model import (DCFInputs, peer_pe_value, reverse_dcf, run_dcf, sensitivity,
                             trailing_pe, wacc_capm)


def simple_inputs(**overrides):
    base = dict(revenue=100.0, margin_start=0.20, margin_target=0.20, growth_y1=0.10, growth_y5=0.10,
                tax_rate=0.25, da_pct=0.05, capex_pct=0.05, nwc_pct=0.10, wacc=0.10,
                terminal_growth=0.02, cash=10.0, debt=20.0, shares=10.0)
    base.update(overrides)
    return DCFInputs(**base)


def test_year_one_by_hand():
    # Revenue 110; EBIT 22; taxes 5.5; NOPAT 16.5; D&A 5.5; capex 5.5; change in NWC 1.0 -> FCFF 15.5
    y1 = run_dcf(simple_inputs()).rows[0]
    assert y1["Revenue"] == pytest.approx(110.0)
    assert y1["EBIT"] == pytest.approx(22.0)
    assert y1["EBIT × (1 − Tax Rate)"] == pytest.approx(16.5)
    assert y1["FCF"] == pytest.approx(15.5)
    assert y1["PV of FCF"] == pytest.approx(15.5 / 1.1)


def test_full_valuation_by_hand():
    # With constant growth and margins, FCFF_t = 15.5 x 1.1^(t-1)
    fcff = [15.5 * 1.1 ** (t - 1) for t in range(1, 6)]
    pv = sum(f / 1.1 ** t for t, f in enumerate(fcff, start=1))
    tv = fcff[-1] * 1.02 / (0.10 - 0.02)
    ev = pv + tv / 1.1 ** 5
    expected_per_share = (ev - 20 + 10) / 10
    r = run_dcf(simple_inputs())
    assert r.enterprise_value == pytest.approx(ev)
    assert r.per_share == pytest.approx(expected_per_share)


def test_growth_and_margin_paths():
    r = run_dcf(simple_inputs(growth_y1=0.20, growth_y5=0.04, margin_start=0.10, margin_target=0.30))
    assert [row["Revenue growth"] for row in r.rows] == pytest.approx([0.20, 0.16, 0.12, 0.08, 0.04])
    assert [row["EBIT margin"] for row in r.rows] == pytest.approx([0.14, 0.18, 0.22, 0.26, 0.30])


def test_no_tax_on_losses():
    r = run_dcf(simple_inputs(margin_start=-0.10, margin_target=-0.10))
    assert r.rows[0]["Taxes on EBIT"] == 0


def test_reverse_dcf_round_trip():
    inp = simple_inputs()
    price = run_dcf(inp, constant_growth=0.07).per_share
    res = reverse_dcf(inp, price)
    assert res.implied_growth == pytest.approx(0.07, abs=1e-6)


def test_reverse_dcf_unreachable_price():
    res = reverse_dcf(simple_inputs(), price=1e9)
    assert res.implied_growth is None
    assert "does not justify" in res.message


def test_wacc_must_exceed_terminal_growth():
    with pytest.raises(ValueError):
        run_dcf(simple_inputs(wacc=0.03, terminal_growth=0.03))
    table = sensitivity(simple_inputs(), [0.022, 0.10], [0.02, 0.03])  # 0.2-point gap is too small
    assert table[0] == [None, None]
    assert table[1][0] == pytest.approx(run_dcf(simple_inputs()).per_share)


def test_pe():
    assert trailing_pe(100, 5) == 20
    assert trailing_pe(100, -1) is None
    res = peer_pe_value(5.0, {"A": 10.0, "B": 20.0, "C": 30.0, "D": None, "E": 250.0})
    assert res.median_pe == 20.0
    assert res.implied_value == 100.0
    assert set(res.excluded) == {"D", "E"}


def test_wacc():
    w = wacc_capm(risk_free=0.04, beta=1.2, erp=0.05, pretax_cost_of_debt=0.06,
                  tax_rate=0.25, market_cap=800, debt=200)
    assert w.cost_of_equity == pytest.approx(0.10)
    assert w.after_tax_cost_of_debt == pytest.approx(0.045)
    assert w.wacc == pytest.approx(0.8 * 0.10 + 0.2 * 0.045)
    assert not math.isnan(w.wacc)

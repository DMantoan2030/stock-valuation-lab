"""Checks the historical revenue growth rates (last year, 3-year and 5-year CAGR)."""
import pytest

from valuation import sec


def snapshot_with_revenue(values):
    snap = sec.CompanySnapshot("TEST", 1, "Test Co", "2080", "Beverages", "1231")
    ends = [f"{2020 + i}-12-31" for i in range(len(values))]
    snap.history["revenue"] = dict(zip(ends, values))
    for key in ("revenue", "operating_income", "income_tax", "pretax_income", "interest_expense", "debt"):
        snap.points[key] = sec.DataPoint(key, None, "missing")
    return snap


def test_growth_rates_by_hand():
    # Revenue 100 -> 110 -> 121 -> 133.1 -> 146.41 -> 161.051 grows exactly 10% a year
    d = sec._default_assumptions(snapshot_with_revenue([100, 110, 121, 133.1, 146.41, 161.051]))
    assert d["growth_last_year"] == pytest.approx(0.10)
    assert d["cagr_3y"] == pytest.approx(0.10)
    assert d["cagr_5y"] == pytest.approx(0.10)
    assert d["growth_y1"] == pytest.approx(0.10)   # default uses the 3-year CAGR


def test_three_year_cagr_is_default_not_a_blend():
    # Last year +50%, but 3-year CAGR = (150/100)^(1/3) - 1, about 14.5%
    d = sec._default_assumptions(snapshot_with_revenue([100, 100, 100, 150]))
    assert d["growth_last_year"] == pytest.approx(0.50)
    assert d["cagr_3y"] == pytest.approx(1.5 ** (1 / 3) - 1)
    assert d["cagr_5y"] is None                     # not enough history
    assert d["hist_growth"] == pytest.approx(d["cagr_3y"])

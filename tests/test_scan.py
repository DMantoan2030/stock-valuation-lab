"""Checks the watchlist parsing and the distress metrics."""
import pytest

from valuation import scan, sec

TICKERS = {t: {"cik": i, "title": t} for i, t in enumerate(["AAPL", "MSFT", "KO", "BRK-B", "NKE"])}


def test_parse_thinkorswim_style_export():
    export = (
        '"Watchlist \'Tech\' exported 11/20/2026"\n'
        "Symbol,Last,Net Chng,Volume\n"
        "AAPL,250.10,1.20,1000\n"
        "MSFT,500.00,-2.00,900\n"
        ".AAPL261218C250,12.30,0.40,10\n"      # option: skipped
        "/ES,6000.00,5.00,100\n"               # future: skipped
        "ZZZZ,1.00,0.00,5\n"                   # not SEC-registered: reported as skipped
        "BRK.B,480.00,1.00,50\n"               # converted to the SEC's BRK-B
        "AAPL,250.10,1.20,1000\n"              # duplicate: kept once
    )
    tickers, skipped = scan.parse_tickers(export, TICKERS)
    assert tickers == ["AAPL", "MSFT", "BRK-B"]
    assert skipped == ["ZZZZ"]


def test_parse_pasted_list():
    tickers, _ = scan.parse_tickers("aapl, ko  nke;MSFT", TICKERS)
    assert tickers == ["AAPL", "KO", "NKE", "MSFT"]


def fake_snapshot(**values):
    snap = sec.CompanySnapshot("TEST", 1, "Test Co", "3571", "Computers", "1231")
    for key, val in values.items():
        snap.points[key] = sec.DataPoint(key, val)
    snap.balance = {"total_assets": 1000.0, "total_liabilities": 600.0, "current_assets": 400.0,
                    "current_liabilities": 300.0, "retained_earnings": 200.0}
    snap.defaults = {"cagr_3y": 0.05, "growth_y1": 0.05, "op_margin": 0.10, "tax_rate": 0.21,
                     "da_pct": 0.03, "capex_pct": 0.03, "nwc_pct": 0.05}
    return snap


def test_altman_z_by_hand():
    snap = fake_snapshot(operating_income=100.0, revenue=800.0)
    # 1.2*100/1000 + 1.4*200/1000 + 3.3*100/1000 + 0.6*900/600 + 1.0*800/1000
    expected = 0.12 + 0.28 + 0.33 + 0.9 + 0.8
    assert scan.altman_z(snap, market_cap=900.0) == pytest.approx(expected)
    assert scan.z_zone(expected) == "grey"
    assert scan.z_zone(1.0) == "distress" and scan.z_zone(3.5) == "safe"


def test_distress_flags():
    snap = fake_snapshot(operating_income=10.0, revenue=800.0, interest_expense=20.0, da=5.0,
                         fcf=-30.0, debt=500.0, cash=20.0, shares=10.0, eps_diluted=0.5)
    row = scan.analyze(snap, price=5.0, beta=1.0, rf=0.045)
    assert row.interest_coverage == pytest.approx(0.5)            # 10 / 20
    assert row.net_debt_ebitda == pytest.approx((500 - 20) / 15)  # 32x
    assert row.distress_score >= 3
    assert any("interest coverage" in f for f in row.distress_flags)
    assert "negative free cash flow" in row.distress_flags
    assert row.dcf_value is not None and row.upside is not None


def test_healthy_company_has_no_flags():
    snap = fake_snapshot(operating_income=200.0, revenue=800.0, interest_expense=10.0, da=20.0,
                         fcf=150.0, debt=100.0, cash=50.0, shares=10.0, eps_diluted=15.0)
    row = scan.analyze(snap, price=200.0, beta=1.0, rf=0.045)
    assert row.distress_flags == []

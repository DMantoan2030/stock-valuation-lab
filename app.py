"""Stock Valuation Lab: DCF, reverse DCF and P/E valuation from SEC filings.

Run locally with:  streamlit run app.py
"""
from __future__ import annotations

import os
import time
from dataclasses import replace

import pandas as pd
import streamlit as st

from valuation import filing, market, scan, sec
from valuation.defaults import DEFAULT_ERP, DEFAULT_TG, default_inputs, default_wacc
from valuation.model import (FCF_FORMULA, MIN_SPREAD, peer_pe_value, reverse_dcf, run_dcf,
                             sensitivity, trailing_pe, wacc_capm)
from valuation.peers import suggest_peers

st.set_page_config(page_title="Stock Valuation Lab", page_icon="📈", layout="wide")

# Gray Analyze button
st.markdown("""
<style>
[data-testid="stFormSubmitButton"] button {
    background-color: #6b7280; border: 1px solid #6b7280; color: #ffffff;
}
[data-testid="stFormSubmitButton"] button:hover {
    background-color: #4b5563; border-color: #4b5563; color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

DISCLAIMER = ("**Educational use only. This is not investment advice.** Results depend entirely on the "
              "assumptions entered and on automatically extracted data, which may be incomplete or wrong. "
              "Always verify inputs against the original SEC filings.")


# ---------------------------------------------------------------------------
# Configuration (keys come from environment variables or Streamlit secrets, never from code)
# ---------------------------------------------------------------------------

def secret(name: str, default: str = "") -> str:
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, default) or default   # an empty value falls back to the default
    except Exception:
        return default


SEC_UA = secret("SEC_USER_AGENT", "DPI681 student valuation app contact@example.com")
FINNHUB_KEY = secret("FINNHUB_API_KEY")


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_ticker_map() -> dict:
    return sec.get_ticker_map(SEC_UA)


# Bump this whenever the company snapshot gains new fields, so data saved by an older
# version of the app is not reused after a redeploy.
SNAPSHOT_VERSION = 4


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_snapshot(ticker: str, version: int) -> sec.CompanySnapshot:
    return sec.build_snapshot(ticker, SEC_UA, load_ticker_map())


def load_snapshot(ticker: str) -> sec.CompanySnapshot:
    return _cached_snapshot(ticker, SNAPSHOT_VERSION)


OVERVIEW_VERSION = 2


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _cached_overview(ticker: str, version: int) -> filing.CompanyOverview:
    return filing.build_overview(ticker, SEC_UA, load_ticker_map())


def load_overview(ticker: str) -> filing.CompanyOverview:
    return _cached_overview(ticker, OVERVIEW_VERSION)


# Finnhub results are cached so repeat lookups cost no requests. Failures raise inside
# the cached function, and Streamlit never caches an exception, so the next visitor retries
# instead of seeing a remembered error.

@st.cache_data(ttl=900, show_spinner=False)
def _cached_quote(ticker: str) -> market.Quote:
    return market.get_quote(ticker, FINNHUB_KEY)


def load_quote(ticker: str) -> tuple[market.Quote | None, str | None]:
    try:
        return _cached_quote(ticker), None
    except Exception as e:  # noqa: BLE001 - show any data problem to the user
        return None, str(e)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _cached_beta(ticker: str) -> float | None:
    return market.get_beta(ticker, FINNHUB_KEY)


def load_beta(ticker: str) -> float | None:
    try:
        return _cached_beta(ticker)
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_risk_free() -> market.RiskFree:
    return market.get_treasury_10y()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _cached_peer_suggestions(ticker: str, sic: str, market_cap: float | None):
    return suggest_peers(ticker, sic, market_cap, FINNHUB_KEY, SEC_UA, load_ticker_map())


def load_peer_suggestions(ticker: str, sic: str, market_cap: float | None):
    try:
        return _cached_peer_suggestions(ticker, sic, market_cap)
    except market.MarketDataError as e:
        return [], [f"Competitor suggestions unavailable right now: {e} You can type competitors below."]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def money(x: float | None, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e12:
        return f"{sign}${a / 1e12:,.{digits}f}T"
    if a >= 1e9:
        return f"{sign}${a / 1e9:,.{digits}f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:,.{digits}f}M"
    return f"{sign}${a:,.2f}"


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{digits}f}%"


def per_share(x: float | None) -> str:
    return "n/a" if x is None or pd.isna(x) else f"${x:,.2f}"


def multiple(x: float | None) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.1f}x"


# ---------------------------------------------------------------------------
# Default assumptions shared by the main company and its peers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

VIEWS = ["Single company", "Watchlist scan"]
st.session_state.setdefault("view", VIEWS[0])


def _switch_to_single() -> None:
    st.session_state["view"] = VIEWS[0]


with st.sidebar:
    st.header("📈 Stock Valuation Lab")
    st.radio("View", VIEWS, key="view", horizontal=True)
    with st.form("ticker_form"):
        ticker_in = st.text_input("U.S. stock ticker", value=st.session_state.get("ticker", ""),
                                  placeholder="e.g. AAPL, KO, MSFT").upper().strip()
        go = st.form_submit_button("Analyze", type="primary", width="stretch", on_click=_switch_to_single)
    if go and ticker_in:
        st.session_state["ticker"] = ticker_in
    st.caption("Try: AAPL · MSFT · KO · NKE · COST")
    st.divider()
    st.markdown("**Data sources**")
    st.markdown("- Financials: SEC EDGAR (10-K, 10-Q)\n- Risk-free rate: U.S. Treasury\n"
                f"- Prices, beta, peers: Finnhub {'✅' if FINNHUB_KEY else '⚠️ no API key (enter prices manually)'}")
    st.divider()
    st.warning(DISCLAIMER)


# ---------------------------------------------------------------------------
# Watchlist scan view
# ---------------------------------------------------------------------------

SORTS = {
    "Most upside": lambda r: -(r.upside if r.upside is not None else -9),
    "Most distressed": lambda r: (-r.distress_score, r.altman_z if r.altman_z is not None else 99),
    "Price implies faster growth than history": lambda r: -(r.growth_gap if r.growth_gap is not None else -9),
    "Price implies slower growth than history": lambda r: (r.growth_gap if r.growth_gap is not None else 9),
}


def _open_from_scan() -> None:
    """Clicking a row in the scan table opens that company in the single-company view."""
    sel = st.session_state["scan_table"].selection.rows
    if sel:
        st.session_state["ticker"] = st.session_state["scan_order"][sel[0]]
        st.session_state["view"] = VIEWS[0]


def run_scan(tickers: list[str]) -> list[scan.ScanRow]:
    rf = load_risk_free().rate
    rows: list[scan.ScanRow] = []
    bar = st.progress(0.0, text="Starting scan...")
    window_start, live_calls = time.time(), 0
    for i, t in enumerate(tickers):
        bar.progress(i / len(tickers), text=f"Analyzing {t} ({i + 1} of {len(tickers)})")
        try:
            snap = load_snapshot(t)
        except Exception as e:  # noqa: BLE001
            r = scan.ScanRow(t)
            r.notes.append(f"SEC data unavailable: {e}")
            rows.append(r)
            continue
        price = beta = None
        if FINNHUB_KEY:
            # Finnhub's free tier allows 60 requests per minute. Requests answered from the
            # app's saved data are instant and do not count; for live ones, pause before
            # the minute's allowance runs out.
            if live_calls >= 50 and time.time() - window_start < 60:
                wait = 61 - (time.time() - window_start)
                bar.progress(i / len(tickers), text=f"Pausing {wait:.0f}s for Finnhub's 60-requests-per-minute limit...")
                time.sleep(wait)
            if time.time() - window_start >= 60:
                window_start, live_calls = time.time(), 0
            t0 = time.time()
            quote, _ = load_quote(t)
            beta = load_beta(t)
            if time.time() - t0 > 0.05:
                live_calls += 2
            price = quote.price if quote else None
        rows.append(scan.analyze(snap, price, beta, rf))
    bar.empty()
    return rows


def render_scan() -> None:
    st.title("Watchlist scan")
    st.markdown(
        "Value every company on a watchlist with the app's default assumptions, then rank them by upside, "
        "financial distress, or the gap between the growth the stock price implies and the company's "
        "3-year revenue CAGR. **Click any row to open the full single-company analysis.**")
    with st.expander("How to get your tickers in"):
        st.markdown(
            "- **Paste:** type or paste tickers in any format (commas, spaces or one per line). If your "
            "thinkorswim watchlist lets you select and copy the symbol column, paste it straight in.\n"
            "- **Upload a CSV:** for example, Schwab.com → Accounts → Positions → *Export* downloads your holdings "
            "as a CSV. Any CSV with a *Symbol* column works.\n\n"
            "Options, futures and anything not registered with the SEC are skipped automatically.")
    c1, c2 = st.columns(2)
    up = c1.file_uploader("Watchlist or positions file (CSV)", type=["csv", "txt"])
    pasted = c2.text_area("...or paste tickers", placeholder="AAPL, MSFT, KO, NKE, COST", height=110)
    text = (up.getvalue().decode("utf-8", errors="ignore") if up else "") + "\n" + (pasted or "")
    tickers, skipped = scan.parse_tickers(text, load_ticker_map()) if text.strip() else ([], [])
    if len(tickers) > scan.MAX_TICKERS:
        st.warning(f"{len(tickers)} tickers found; scanning the first {scan.MAX_TICKERS}.")
        tickers = tickers[:scan.MAX_TICKERS]
    if tickers:
        st.caption(f"{len(tickers)} ticker(s) ready: {', '.join(tickers)}"
                   + (f" · skipped (not SEC-registered): {', '.join(skipped)}" if skipped else ""))
        est = max(1, round(len(tickers) * 2 / 50))
        st.caption(f"A first scan takes roughly {est} minute(s) because of Finnhub's free-tier limit; "
                   "rescans of the same tickers are fast.")
    if st.button("Run scan", type="primary", disabled=not tickers):
        st.session_state["scan_rows"] = run_scan(tickers)

    rows: list[scan.ScanRow] = st.session_state.get("scan_rows") or []
    if not rows:
        st.info(DISCLAIMER)
        return

    sort = st.radio("Rank by", list(SORTS), horizontal=True, key="scan_sort")
    sectors = sorted({r.sector for r in rows if r.sector})
    chosen = st.multiselect("Filter by sector", sectors, placeholder="All sectors", key="scan_sectors")
    if chosen:
        rows = [r for r in rows if r.sector in chosen]
    rows = sorted(rows, key=SORTS[sort])
    st.session_state["scan_order"] = [r.ticker for r in rows]
    if not rows:
        st.write("No companies in the selected sectors.")
        return
    df = pd.DataFrame([{
        "Ticker": r.ticker, "Company": r.name, "Sector": r.sector, "Industry": r.industry,
        "Price": r.price, "DCF value": r.dcf_value, "Upside": r.upside,
        "Implied growth": r.implied_growth, "3-yr CAGR": r.cagr_3y, "Growth gap": r.growth_gap,
        "P/E": r.pe, "Altman Z": r.altman_z, "Z zone": scan.z_zone(r.altman_z),
        "Interest cover": r.interest_coverage, "Net debt/EBITDA": r.net_debt_ebitda, "FCF margin": r.fcf_margin,
        "Distress flags": r.distress_score, "Why flagged": "; ".join(r.distress_flags),
        "Notes": "; ".join(r.notes),
    } for r in rows])
    for col in ("Price", "DCF value", "Upside", "Implied growth", "3-yr CAGR", "Growth gap", "P/E", "Altman Z",
                "Interest cover", "Net debt/EBITDA", "FCF margin"):
        df[col] = pd.to_numeric(df[col], errors="coerce")      # missing values show as blank, not "None"
    cc = st.column_config
    st.dataframe(
        df, key="scan_table", on_select=_open_from_scan, selection_mode="single-row", hide_index=True,
        width="stretch",
        column_config={
            "Price": cc.NumberColumn(format="dollar"), "DCF value": cc.NumberColumn(format="dollar"),
            "Upside": cc.NumberColumn(format="percent"), "Implied growth": cc.NumberColumn(format="percent"),
            "3-yr CAGR": cc.NumberColumn(format="percent"), "FCF margin": cc.NumberColumn(format="percent"),
            "Growth gap": cc.NumberColumn(format="percent",
                                          help="Price-implied revenue growth minus the 3-year revenue CAGR"),
            "P/E": cc.NumberColumn(format="%.1fx"), "Altman Z": cc.NumberColumn(format="%.2f"),
            "Interest cover": cc.NumberColumn(format="%.1fx", help="EBIT / interest expense; below 1.5x is a warning"),
            "Net debt/EBITDA": cc.NumberColumn(format="%.1fx", help="Above 4x is a warning"),
            "Distress flags": cc.NumberColumn(help="Number of warning signs (see 'Why flagged')"),
            "Sector": cc.TextColumn(help="Approximate GICS-style sector, derived from the company's SEC industry "
                                         "(SIC) code"),
            "Industry": cc.TextColumn(help="The company's official SEC industry classification (SIC)"),
        })
    st.caption("Click a row to open that company's full analysis. Values use default assumptions "
               "(3-year revenue CAGR, current margin, CAPM WACC, 2.5% terminal growth), so treat the ranking "
               "as a starting point for research. Altman Z zones: below 1.81 distress, 1.81 to 2.99 grey, "
               "above 2.99 safe; it is not meaningful for banks, insurers or REITs.")
    st.info(DISCLAIMER)


if st.session_state["view"] == "Watchlist scan":
    render_scan()
    st.stop()

if st.session_state.get("scan_rows"):
    st.sidebar.button("← Back to watchlist scan", width="stretch",
                      on_click=lambda: st.session_state.update(view=VIEWS[1]))

ticker = st.session_state.get("ticker")

if not ticker:
    st.title("Stock Valuation Lab")
    st.markdown(
        "Value any U.S.-listed company three ways, using data pulled directly from its latest SEC filings:\n\n"
        "1. **Five-year discounted cash flow (DCF)**: what the business is worth based on the cash it can generate.\n"
        "2. **Reverse DCF**: the annual revenue growth that today's stock price implies.\n"
        "3. **P/E comparison**: how the stock's earnings multiple compares with its closest competitors.\n\n"
        "Enter a ticker in the sidebar to begin. Every assumption is adjustable, and every input is linked "
        "to the filing it came from.")
    st.info(DISCLAIMER)
    st.stop()

# ---------------------------------------------------------------------------
# Load company data
# ---------------------------------------------------------------------------

try:
    with st.spinner(f"Reading {ticker}'s SEC filings..."):
        snap = load_snapshot(ticker)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load SEC data for {ticker}: {e}")
    st.stop()

quote, quote_err = load_quote(ticker)
rf = load_risk_free()
beta = load_beta(ticker) if FINNHUB_KEY else None

st.title(f"{snap.name} ({ticker})")
st.caption(f"SIC {snap.sic}: {snap.sic_description} · Fiscal year ends {snap.fiscal_year_end[:2]}/{snap.fiscal_year_end[2:]} "
           f"· CIK {snap.cik}")

if quote:
    price = quote.price
    price_note = f"{quote.source}, {quote.as_of}"
else:
    st.warning(f"Live price unavailable ({quote_err}). Enter the current price manually to continue.")
    price = st.number_input("Current share price ($)", min_value=0.0, value=0.0, step=1.0, key=f"price_{ticker}")
    price_note = "Entered manually by user"
    if price <= 0:
        st.stop()

shares = snap.v("shares")
market_cap = price * shares if shares else None

# ---------------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------------

erp_default = DEFAULT_ERP
wacc0, wacc_info = default_wacc(snap, price, beta, rf.rate, erp_default)
base = default_inputs(snap, wacc0)

K = lambda name: f"{ticker}_{name}"  # noqa: E731 - per-ticker widget keys
defaults_pct = {
    "g1": base.growth_y1 * 100, "g5": base.growth_y5 * 100, "margin": base.margin_target * 100,
    "tax": base.tax_rate * 100, "da": base.da_pct * 100, "capex": base.capex_pct * 100, "nwc": base.nwc_pct * 100,
    "tg": DEFAULT_TG * 100, "rf": rf.rate * 100, "beta": wacc_info["beta"], "erp": erp_default * 100,
    "kd": wacc_info["kd"] * 100, "wacc_manual": wacc0 * 100,
}
defaults_pct["g1"] = float(f"{defaults_pct['g1']:.1f}")   # match the displayed 3-year CAGR
for name, val in defaults_pct.items():
    st.session_state.setdefault(K(name), round(float(val), 2))
st.session_state.setdefault(K("manual"), False)

GROWTH_BASES = {"3-year CAGR": "cagr_3y", "5-year CAGR": "cagr_5y", "Last year's growth": "growth_last_year",
                "Avg of last year & 3-year CAGR": "growth_blend"}
st.session_state.setdefault(K("basis"), "3-year CAGR")


def apply_growth_basis() -> None:
    """When the user picks a growth basis, copy that rate into Year 1 growth.
    Year 5 growth is set halfway between Year 1 and terminal growth."""
    rate = snap.defaults.get(GROWTH_BASES[st.session_state[K("basis")]])
    if rate is not None:
        g1 = float(f"{rate * 100:.1f}")   # same rounding as the displayed CAGR
        st.session_state[K("g1")] = g1
        st.session_state[K("g5")] = round((g1 + st.session_state[K("tg")]) / 2, 2)


with st.expander("⚙️ Assumptions (defaults come from the company's own history; adjust freely)", expanded=True):
    st.markdown("**Historical revenue growth** (from fiscal-year 10-K revenue)")
    h1, h2, h3, h4, h5 = st.columns([1, 1, 1, 1.2, 1.6])
    h1.metric("Last year", pct(snap.defaults.get("growth_last_year")))
    h2.metric("3-year CAGR", pct(snap.defaults.get("cagr_3y")))
    h3.metric("5-year CAGR", pct(snap.defaults.get("cagr_5y")))
    h4.metric("Avg: last yr & 3-yr", pct(snap.defaults.get("growth_blend")))
    h5.selectbox("Base Year 1 growth on", list(GROWTH_BASES), key=K("basis"), on_change=apply_growth_basis,
                 help="Copies the chosen historical rate into Year 1 growth. You can still type any value.")
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Revenue growth (forecast)**")
        st.number_input("Year 1 growth (%)", step=0.5, key=K("g1"), format="%.1f")
        st.number_input("Year 5 growth (%)", step=0.5, key=K("g5"), format="%.1f")
        st.caption("Growth moves in a straight line from Year 1 to Year 5.")
    with c2:
        st.markdown("**Profitability**")
        st.number_input("Target EBIT margin, year 5 (%)", step=0.5, key=K("margin"), format="%.1f")
        st.caption(f"Current EBIT margin: {pct(base.margin_start)}. Margin moves in a straight line to the target.")
        st.number_input("Tax rate (%)", step=0.5, min_value=0.0, max_value=60.0, key=K("tax"), format="%.1f")
    with c3:
        st.markdown("**Reinvestment (% of revenue)**")
        st.number_input("D&A (%)", step=0.25, key=K("da"), format="%.2f")
        st.number_input("Working capital (%)", step=0.5, key=K("nwc"), format="%.1f",
                        help="Net working capital as % of revenue. Change in working capital each year = "
                             "this % × change in revenue. Negative means suppliers/customers fund growth.")
        st.number_input("CapEx (%)", step=0.25, key=K("capex"), format="%.2f")
        st.caption("Defaults: 3-year historical averages")
    with c4:
        st.markdown("**Discount rate**")
        st.number_input("Terminal growth (%)", step=0.25, key=K("tg"), format="%.2f")
        manual = st.toggle("Enter WACC directly", key=K("manual"))
        if manual:
            st.number_input("WACC (%)", step=0.25, key=K("wacc_manual"), format="%.2f")
        else:
            st.number_input("Risk-free rate (%)", step=0.1, key=K("rf"), format="%.2f",
                            help=f"{rf.source}, {rf.as_of}")
            st.number_input("Beta", step=0.05, key=K("beta"), format="%.2f",
                            help="From Finnhub" if not wacc_info["beta_estimated"] else "Not available; 1.0 assumed")
            st.number_input("Equity risk premium (%)", step=0.25, key=K("erp"), format="%.2f")
            st.number_input("Pre-tax cost of debt (%)", step=0.25, key=K("kd"), format="%.2f",
                            help="Default: interest expense ÷ total debt")
    if st.button("Reset to defaults"):
        for name in list(defaults_pct) + ["manual", "basis"]:
            st.session_state.pop(K(name), None)
        st.rerun()

ss = st.session_state
tax_rate = ss[K("tax")] / 100
if ss[K("manual")]:
    wacc = ss[K("wacc_manual")] / 100
    wacc_detail = None
else:
    wacc_detail = wacc_capm(ss[K("rf")] / 100, ss[K("beta")], ss[K("erp")] / 100, ss[K("kd")] / 100,
                            tax_rate, market_cap or 0.0, snap.v("debt") or 0.0)
    wacc = wacc_detail.wacc

inputs = replace(base, growth_y1=ss[K("g1")] / 100, growth_y5=ss[K("g5")] / 100,
                 margin_target=ss[K("margin")] / 100, tax_rate=tax_rate, da_pct=ss[K("da")] / 100,
                 capex_pct=ss[K("capex")] / 100, nwc_pct=ss[K("nwc")] / 100, wacc=wacc,
                 terminal_growth=ss[K("tg")] / 100)

# ---------------------------------------------------------------------------
# Run the three valuations
# ---------------------------------------------------------------------------

dcf_allowed = snap.dcf_supported
if not snap.dcf_supported and snap.v("revenue"):
    dcf_allowed = st.checkbox("Run the DCF anyway (not recommended for this type of company)", key=K("force"))

dcf = rev = None
dcf_error = None
if not snap.v("revenue") or not shares:
    dcf_error = "Revenue or share count is missing, so the DCF cannot run."
elif not dcf_allowed:
    dcf_error = "DCF skipped for this type of company (see Data quality)."
else:
    try:
        dcf = run_dcf(inputs)
        rev = reverse_dcf(inputs, price)
    except ValueError as e:
        dcf_error = str(e)

eps = snap.v("eps_diluted")
pe = trailing_pe(price, eps)

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------

st.info(DISCLAIMER)
m = st.columns(6)
m[0].metric("Current price", per_share(price), help=price_note)
m[1].metric("DCF value / share", per_share(dcf.per_share) if dcf else "n/a")
m[2].metric("Upside / downside", pct(dcf.per_share / price - 1) if dcf else "n/a")
m[3].metric("Implied revenue growth", pct(rev.implied_growth) if rev and rev.implied_growth is not None else "n/a",
            help="Reverse DCF: constant 5-year revenue growth that justifies today's price")
m[4].metric("Trailing P/E", multiple(pe), help=f"Price ÷ TTM diluted EPS ({per_share(eps)})")
peer_metric = m[5].empty()

if dcf_error:
    st.warning(dcf_error)
if snap.flags:
    st.caption(f"⚠️ {len(snap.flags)} data-quality note(s). See the **Sources & data quality** tab.")

tab_dcf, tab_rev, tab_peers, tab_src, tab_method, tab_company = st.tabs(
    ["DCF", "Reverse DCF", "P/E & competitors", "Sources & data quality", "Methodology", "Company overview"])

# ---------------------------------------------------------------------------
# DCF tab
# ---------------------------------------------------------------------------

with tab_dcf:
    if dcf is None:
        st.write(dcf_error or "DCF unavailable.")
    else:
        st.markdown(f"**Free cash flow formula:** `{FCF_FORMULA}`")
        lines = ["Revenue growth", "Revenue", "EBIT margin", "EBIT", "Taxes on EBIT", "EBIT × (1 − Tax Rate)",
                 "+ D&A", "− Change in Working Capital", "− CapEx", "FCF", "Discount factor", "PV of FCF"]
        table = pd.DataFrame(dcf.rows).set_index("Year")[lines].T
        table.columns = [f"Year {c}" for c in table.columns]
        base_col = {"Revenue": inputs.revenue, "EBIT margin": inputs.margin_start,
                    "EBIT": inputs.revenue * inputs.margin_start}
        table.insert(0, "Base (TTM)", [base_col.get(i) for i in table.index])
        fmt = table.copy().astype(object)
        for idx in table.index:
            for col in table.columns:
                v = table.loc[idx, col]
                if v is None or pd.isna(v):
                    fmt.loc[idx, col] = ""
                elif idx in ("Revenue growth", "EBIT margin"):
                    fmt.loc[idx, col] = pct(v)
                elif idx == "Discount factor":
                    fmt.loc[idx, col] = f"{v:.3f}"
                else:
                    fmt.loc[idx, col] = money(v, 2)
        st.dataframe(fmt, width="stretch", height=(len(fmt) + 1) * 35 + 3)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**From cash flows to value per share**")
            bridge = pd.DataFrame([
                ("Sum of PV of FCF (years 1-5)", money(dcf.pv_fcff, 2)),
                (f"Terminal value (year 5, g = {pct(inputs.terminal_growth, 2)})", money(dcf.terminal_value, 2)),
                ("PV of terminal value", money(dcf.pv_terminal, 2)),
                ("Enterprise value", money(dcf.enterprise_value, 2)),
                ("− Total debt", money(-inputs.debt, 2)),
                ("+ Cash & short-term investments", money(inputs.cash, 2)),
                ("Equity value", money(dcf.equity_value, 2)),
                ("÷ Diluted shares", f"{inputs.shares / 1e6:,.1f}M"),
                ("DCF value per share", per_share(dcf.per_share)),
                ("Current price", per_share(price)),
                ("Upside / downside", pct(dcf.per_share / price - 1)),
            ], columns=["Step", "Value"])
            st.dataframe(bridge, hide_index=True, width="stretch")
            share = dcf.terminal_share
            msg = f"Terminal value is {pct(share)} of enterprise value."
            (st.warning if share > 0.75 else st.caption)(
                msg + (" Results depend heavily on the terminal growth and WACC assumptions." if share > 0.75 else ""))
        with c2:
            st.markdown(f"**WACC: {pct(wacc, 2)}**")
            if wacc_detail:
                st.markdown(
                    f"- Cost of equity = {pct(ss[K('rf')] / 100, 2)} + {ss[K('beta')]:.2f} × {pct(ss[K('erp')] / 100, 2)}"
                    f" = **{pct(wacc_detail.cost_of_equity, 2)}**\n"
                    f"- After-tax cost of debt = {pct(ss[K('kd')] / 100, 2)} × (1 − {pct(tax_rate)}) = "
                    f"**{pct(wacc_detail.after_tax_cost_of_debt, 2)}**\n"
                    f"- Weights: equity {pct(wacc_detail.weight_equity)} (market cap {money(market_cap)}), "
                    f"debt {pct(wacc_detail.weight_debt)} ({money(snap.v('debt'))})")
                if wacc_info["beta_estimated"]:
                    st.caption("⚠️ Beta not available; 1.0 assumed.")
            else:
                st.caption("WACC entered manually.")

        st.markdown("**Sensitivity: DCF value per share by WACC and terminal growth**")
        waccs = [round(wacc + d, 4) for d in (-0.01, -0.005, 0, 0.005, 0.01)]
        tgs = [0.015, 0.02, 0.025, 0.03, 0.035]
        if round(inputs.terminal_growth, 4) not in tgs:
            tgs = sorted(set(tgs + [round(inputs.terminal_growth, 4)]))
        sens = sensitivity(inputs, waccs, tgs)
        sens_df = pd.DataFrame([[per_share(v) if v is not None else "n/a" for v in row] for row in sens],
                               index=[f"WACC {pct(w, 2)}" for w in waccs],
                               columns=[f"g = {pct(g, 1)}" for g in tgs])
        st.dataframe(sens_df, width="stretch")
        st.caption(f"Current price: {per_share(price)}. Cells marked n/a have WACC within "
                   f"{MIN_SPREAD:.1%} of terminal growth, where the formula breaks down.")

# ---------------------------------------------------------------------------
# Reverse DCF tab
# ---------------------------------------------------------------------------

with tab_rev:
    if rev is None:
        st.write(dcf_error or "Reverse DCF unavailable.")
    else:
        if rev.implied_growth is not None:
            st.markdown(f"### The market price implies **{pct(rev.implied_growth)}** annual revenue growth for 5 years")
            st.markdown(
                f"Holding every other assumption fixed (target EBIT margin {pct(inputs.margin_target)}, "
                f"WACC {pct(wacc, 2)}, terminal growth {pct(inputs.terminal_growth, 2)}), revenue would need to grow "
                f"**{pct(rev.implied_growth)} per year** to justify {per_share(price)} per share.")
            comp = pd.DataFrame({
                "Benchmark": ["Implied by price", "Your year-1 assumption", "Your year-5 assumption",
                              "Historical: last year", "Historical: 3-year CAGR", "Historical: 5-year CAGR",
                              "Historical: avg of last year & 3-year CAGR"],
                "Annual revenue growth": [pct(rev.implied_growth), pct(inputs.growth_y1), pct(inputs.growth_y5),
                                          pct(snap.defaults.get("growth_last_year")),
                                          pct(snap.defaults.get("cagr_3y")), pct(snap.defaults.get("cagr_5y")),
                                          pct(snap.defaults.get("growth_blend"))],
            })
            st.dataframe(comp, hide_index=True, width="content")
        else:
            st.warning(rev.message)
        st.markdown("**DCF value per share at different constant growth rates**")
        grid = [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        curve = pd.DataFrame({"Revenue growth (constant, yrs 1-5)": [pct(g, 0) for g in grid],
                              "DCF value / share": [per_share(run_dcf(inputs, constant_growth=g).per_share) for g in grid]})
        st.dataframe(curve, hide_index=True)
        st.caption("The implied growth depends heavily on the margin assumption: a higher target margin "
                   "means less growth is needed to justify the same price.")

# ---------------------------------------------------------------------------
# P/E & competitors tab
# ---------------------------------------------------------------------------

peer_rows = []
peer_pes: dict[str, float | None] = {}
peer_result = None

with tab_peers:
    st.markdown("#### Competitors")
    if FINNHUB_KEY:
        with st.spinner("Finding competitors..."):
            suggestions, peer_notes = load_peer_suggestions(ticker, snap.sic, market_cap)
    else:
        suggestions, peer_notes = [], ["Add a Finnhub API key to get suggested competitors and live peer prices."]
    for n in peer_notes:
        st.caption(f"ℹ️ {n}")
    if suggestions:
        st.dataframe(pd.DataFrame([{"Ticker": s.ticker, "Company": s.name, "Market cap": money(s.market_cap),
                                    "SIC": s.sic, "Why suggested": s.reason} for s in suggestions]),
                     hide_index=True, width="stretch")
    st.session_state.setdefault(K("peers"), ", ".join(s.ticker for s in suggestions))
    peer_text = st.text_input("Competitors to compare (comma-separated tickers; edit freely)", key=K("peers"))
    peer_list = [p.strip().upper() for p in peer_text.split(",") if p.strip() and p.strip().upper() != ticker][:6]
    st.caption("Suggestions are based on industry and company size. Check that each one has a similar business model.")

    for p in peer_list:
        row = {"Ticker": p}
        try:
            with st.spinner(f"Loading {p}..."):
                ps = load_snapshot(p)
        except Exception as e:  # noqa: BLE001
            row["Company"] = f"⚠️ {e}"
            peer_rows.append(row)
            continue
        pq, perr = load_quote(p)
        p_price = pq.price if pq else None
        p_eps = ps.v("eps_diluted")
        p_pe = trailing_pe(p_price, p_eps)
        peer_pes[p] = p_pe
        row.update({"Company": ps.name, "Price": p_price, "Market cap": (p_price or 0) * (ps.v("shares") or 0) or None,
                    "Revenue (TTM)": ps.v("revenue"), "EBIT margin": ps.defaults.get("op_margin"),
                    "3-yr CAGR": ps.defaults.get("cagr_3y"), "5-yr CAGR": ps.defaults.get("cagr_5y"),
                    "Trailing P/E": p_pe})
        if p_price and ps.dcf_supported and ps.v("revenue") and ps.v("shares"):
            try:
                p_wacc, _ = default_wacc(ps, p_price, load_beta(p) if FINNHUB_KEY else None, rf.rate, DEFAULT_ERP)
                p_in = default_inputs(ps, p_wacc)
                p_dcf = run_dcf(p_in)
                p_rev = reverse_dcf(p_in, p_price)
                row.update({"WACC": p_wacc, "DCF value / share": p_dcf.per_share,
                            "Upside": p_dcf.per_share / p_price - 1, "Implied growth": p_rev.implied_growth})
            except ValueError:
                pass
        row["Notes"] = ("no live price; " if not p_price else "") + ("DCF n/a for this industry; " if not ps.dcf_supported else "") \
            + (f"{len(ps.flags)} data note(s)" if ps.flags else "")
        peer_rows.append(row)

    peer_result = peer_pe_value(eps, peer_pes) if peer_pes else None

    st.markdown("#### Comparison table")
    me = {"Ticker": f"▶ {ticker}", "Company": snap.name, "Price": price, "Market cap": market_cap,
          "Revenue (TTM)": snap.v("revenue"), "EBIT margin": base.margin_start,
          "3-yr CAGR": snap.defaults.get("cagr_3y"), "5-yr CAGR": snap.defaults.get("cagr_5y"),
          "Trailing P/E": pe, "WACC": wacc,
          "DCF value / share": dcf.per_share if dcf else None,
          "Upside": dcf.per_share / price - 1 if dcf else None,
          "Implied growth": rev.implied_growth if rev else None, "Notes": "your assumptions"}
    comp = pd.DataFrame([me] + peer_rows)
    show = comp.copy().astype(object)
    for col, f in {"Price": per_share, "Market cap": money, "Revenue (TTM)": money, "EBIT margin": pct,
                   "3-yr CAGR": pct, "5-yr CAGR": pct, "Trailing P/E": multiple, "WACC": pct, "DCF value / share": per_share,
                   "Upside": pct, "Implied growth": pct}.items():
        if col in show:
            show[col] = [f(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else "n/a" for v in comp[col]]
    st.dataframe(show.fillna(""), hide_index=True, width="stretch")
    st.caption("Peers are valued with their own default assumptions (3-year revenue CAGR, current margin, "
               "CAPM WACC, 2.5% terminal growth). Only the selected company uses your custom assumptions.")

    st.markdown("#### P/E valuation")
    if eps is None or eps <= 0:
        st.warning(f"{ticker}'s TTM diluted EPS is {per_share(eps)}; a P/E valuation is not meaningful.")
    elif not peer_result or peer_result.median_pe is None:
        st.write("Add competitors with positive earnings and live prices to compute a peer-based value.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{ticker} trailing P/E", multiple(pe))
        c2.metric("Median peer P/E", multiple(peer_result.median_pe))
        c3.metric("Value at peer P/E", per_share(peer_result.implied_value),
                  delta=pct(peer_result.implied_value / price - 1))
        st.markdown(f"Value = {ticker} TTM diluted EPS {per_share(eps)} × median peer P/E "
                    f"{multiple(peer_result.median_pe)} = **{per_share(peer_result.implied_value)}**")
        if peer_result.excluded:
            st.caption("Excluded from the median: " +
                       "; ".join(f"{t} ({why})" for t, why in peer_result.excluded.items()))

if peer_result and peer_result.implied_value:
    peer_metric.metric("Value at peer P/E", per_share(peer_result.implied_value),
                       help=f"TTM EPS × median peer P/E ({multiple(peer_result.median_pe)})")
else:
    peer_metric.metric("Value at peer P/E", "n/a", help="See the P/E & competitors tab")

# ---------------------------------------------------------------------------
# Sources & data quality tab
# ---------------------------------------------------------------------------

with tab_src:
    if snap.flags:
        st.markdown("#### Data-quality flags")
        for f in snap.flags:
            st.warning(f)
    else:
        st.success("No data-quality issues detected for the core inputs.")

    st.markdown("#### SEC inputs")
    rows = []
    for key, p in snap.points.items():
        r = p.as_row()
        v = p.value
        r["Value"] = per_share(v) if key == "eps_diluted" else (f"{v / 1e6:,.1f}M" if key == "shares" and v else money(v, 2))
        rows.append(r)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Filing link": st.column_config.LinkColumn("Filing link", display_text="open filing")})

    st.markdown("#### Market inputs")
    st.dataframe(pd.DataFrame([
        {"Input": "Share price", "Value": per_share(price), "Source": price_note},
        {"Input": "Beta", "Value": f"{wacc_info['beta']:.2f}",
         "Source": "Finnhub /stock/metric" if not wacc_info["beta_estimated"] else "Assumed 1.0 (not available)"},
        {"Input": "10-year Treasury yield", "Value": pct(rf.rate, 2), "Source": f"{rf.source}, {rf.as_of}"},
        {"Input": "Equity risk premium", "Value": pct(DEFAULT_ERP, 2), "Source": "Assumption (long-run U.S. average)"},
    ]), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Methodology tab
# ---------------------------------------------------------------------------

with tab_method:
    st.markdown(f"""
#### Five-year DCF (free cash flow to the firm)
1. **Revenue:** Revenueₜ = Revenueₜ₋₁ × (1 + gₜ). Growth moves in a straight line from your year-1 rate to your year-5 rate.
2. **EBIT** = Revenue × EBIT margin. The margin moves in a straight line from today's margin to your target by year 5.
3. **{FCF_FORMULA}**
   - Tax is applied only when EBIT is positive.
   - D&A and CapEx are % of revenue.
   - Change in Working Capital = working-capital % × change in revenue.
4. **Discounting:** each year's FCF is discounted at WACC (end-of-year convention).
5. **Terminal value** = FCF₅ × (1 + g) ÷ (WACC − g). WACC must exceed g by at least 0.5 percentage points.
6. **Equity value** = enterprise value − total debt + cash and short-term investments. **Value per share** = equity value ÷ diluted shares.

#### WACC
WACC = E/(D+E) × (risk-free + beta × ERP) + D/(D+E) × cost of debt × (1 − tax).
- Equity weight uses market cap.
- Debt weight uses book debt.
- Cost of debt defaults to interest expense ÷ debt.
- The default WACC is kept between 5% and 15%.

#### Reverse DCF
Holds every assumption fixed except revenue growth, then solves for the single constant growth rate (years 1 to 5) at which DCF value per share equals the current price. The app scans −50% to +100% and refines the answer by bisection.

#### P/E
- **Trailing P/E** = price ÷ trailing-twelve-month diluted EPS.
- **Peer-based value** = company EPS × median peer P/E.
- Peers with negative EPS or P/E above 100 are excluded.

#### Data
- **Trailing twelve months (TTM)** = last fiscal year (10-K) + current year-to-date (10-Q) − prior-year year-to-date.
- **Balance-sheet items** come from the latest filing.
- **Restated numbers:** when the same period appears in several filings, the most recently filed value is used.

#### Known limitations
- Operating leases, stock-based compensation, minority interests and long-term investments are not adjusted for.
- Banks, insurers and REITs are not suited to this DCF.
- Foreign filers (20-F) are not supported.
- One-time items can distort EPS.
""")

# ---------------------------------------------------------------------------
# Company overview tab (from the latest 10-K, quoted word for word)
# ---------------------------------------------------------------------------


def quote_block(paras: list[str]) -> None:
    for p in paras:
        st.markdown("> " + p.replace("$", r"\$"))   # stop "$" from being read as math notation


with tab_company:
    try:
        with st.spinner("Reading the latest 10-K..."):
            ov = load_overview(ticker)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the 10-K for {ticker}: {e}")
        ov = None

    if ov is not None:
        if ov.url:
            st.caption(f"Source: [{ov.form} for fiscal year ending {ov.period}]({ov.url}), filed {ov.filed}. "
                       "Every excerpt below is quoted word for word from the filing. Excerpts are picked by "
                       "simple rules (section headings and keywords), not by AI, so read the full sections "
                       "at the bottom for complete context.")
        for n in ov.notes:
            st.info(n)

        if ov.url:
            st.markdown("#### What the company does")
            st.caption("Opening paragraphs of Item 1, Business")
            if ov.what_it_does:
                quote_block(ov.what_it_does)
            else:
                st.write("Not found.")

            st.markdown("#### Strategy")
            if ov.strategy:
                st.caption(f"From the section \"{ov.strategy_heading}\"" if ov.strategy_heading
                           else "Paragraphs that state the company's strategy or priorities")
                quote_block(ov.strategy)
            else:
                st.write("This 10-K does not include an explicit strategy statement.")

            st.markdown("#### Main revenue streams")
            if ov.revenue_table:
                rows = ov.revenue_table
                width = max(len(r) for r in rows)
                rows = [r + [""] * (width - len(r)) for r in rows]
                header = rows[0] if filing._is_header([c for c in rows[0] if c]) else None
                body = rows[1:] if header else rows
                cols = [c or " " * (i + 1) for i, c in enumerate(header)] if header else \
                    [" "] + [f"col {i}" for i in range(1, width)]
                # make duplicate column names (e.g. two "Change" columns) unique
                seen: dict[str, int] = {}
                for i, c in enumerate(cols):
                    seen[c] = seen.get(c, 0) + 1
                    if seen[c] > 1:
                        cols[i] = f"{c} ({seen[c]})"
                st.dataframe(pd.DataFrame(body, columns=cols), hide_index=True, width="stretch")
                total = filing.table_total(ov.revenue_table)
                rev_m = (snap.v("revenue") or 0) / 1e6
                if total and rev_m and total < 0.8 * rev_m and total * 1000 > 1.2 * rev_m:
                    st.warning(f"This table totals {total:,.0f} (millions), but the company's revenue is about "
                               f"{rev_m:,.0f} million, so it likely covers only one segment. "
                               "See the MD&A for the full breakdown.")
                else:
                    st.caption("Revenue breakdown table from the MD&A, as reported (usually in $ millions). "
                               "The rows were checked to add up to the total.")
            else:
                st.caption("No revenue breakdown table was found whose rows add up to the total.")
            quote_block(ov.revenue)

            st.markdown("#### Management's overview (MD&A)")
            if ov.mdna_overview:
                quote_block(ov.mdna_overview)
            else:
                st.write("Not found.")

            st.markdown("#### Risks")
            if ov.struggles:
                st.caption(f"From the MD&A section \"{ov.struggles_heading}\"" if ov.struggles_heading
                           else "MD&A paragraphs that mention declines, pressures or challenges most often")
                quote_block(ov.struggles)
            else:
                st.write("Not found.")

            st.markdown("#### Top risk factors (Item 1A headlines)")
            if ov.risk_headlines:
                st.markdown("\n".join(f"- {h.replace('$', chr(92) + '$')}" for h in ov.risk_headlines))
            else:
                st.write("Not found.")

            st.markdown("#### Read the full sections")
            for name, body in ov.sections.items():
                with st.expander(f"{name} ({len(body.split()):,} words)"):
                    if body:
                        st.text(body[:60_000] + ("\n\n[Truncated. Open the full filing for the rest.]"
                                                 if len(body) > 60_000 else ""))
                    else:
                        st.write("Not found in this filing.")

# Stock Valuation Lab

A web app that values any U.S.-listed company three ways, using data pulled directly from its latest SEC filings:

1. **Five-year DCF**: discounted free cash flow to the firm, starting from EBIT
2. **Reverse DCF**: the annual revenue growth that today's stock price implies
3. **P/E comparison**: trailing P/E against 3 to 5 suggested competitors (which you can replace)

**Version 2 (this branch)** adds a **Company overview** tab built from the latest 10-K: what the company does, its strategy, revenue breakdown, management's overview, risks discussed in the MD&A, and its top risk factors. Every excerpt is quoted word for word and picked by simple, transparent rules (section headings and keywords), not by AI. The full Business, Risk Factors and MD&A sections can be read in the app.

Every assumption can be adjusted. Every input shows the filing it came from, its period and its filing date. Missing or unreliable data is flagged, never invented.

> **Educational use only. This is not investment advice.**

Built for DPI-681 (The Science and Implications of Generative AI), Harvard Kennedy School, Fall 2026.

## How to use it

1. Enter a ticker (try `AAPL`, `MSFT`, `KO`, `NKE` or `COST`) and click **Analyze**.
2. Review the headline numbers: price, DCF value, upside, implied growth, P/E and peer-based value.
3. Adjust the assumptions (growth, margin, tax, reinvestment, WACC, terminal growth) and watch the results update.
4. Use the tabs for the full DCF table, sensitivity grid, reverse DCF, competitor comparison and data sources.

## Method

The free cash flow formula:

```
FCF = EBIT × (1 − Tax Rate) + D&A − Change in Working Capital − CapEx
```

- **Terminal value** = FCF₅ × (1 + g) ÷ (WACC − g)
- **Equity value** = enterprise value − debt + cash
- **WACC** uses CAPM: the 10-year Treasury yield, beta from Finnhub and a 5% equity risk premium.

The **Methodology** tab inside the app has the full details.

## Data sources

| Data | Source |
|---|---|
| Financial statements (10-K, 10-Q) | [SEC EDGAR XBRL API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) |
| Share price, beta, peer list, market cap | [Finnhub](https://finnhub.io) (free tier) |
| Risk-free rate | [U.S. Treasury daily yield curve](https://home.treasury.gov/resource-center/data-chart-center/interest-rates) |

## Run it yourself

```bash
pip install -r requirements.txt
mkdir .streamlit && cp secrets.example.toml .streamlit/secrets.toml   # then add your own keys
streamlit run app.py
```

To run the calculation tests, install pytest and run `python -m pytest`. They check the DCF, reverse DCF, P/E and WACC math against hand-calculated examples.

## Known limitations

- Operating leases, stock-based compensation, minority interests and long-term investments are not adjusted for.
- Banks, insurers and REITs are flagged, and their DCF is skipped by default.
- Foreign filers (20-F / IFRS) are not supported.
- Peers are matched on industry and size. Business-model fit should be checked by the user.
- Without a Finnhub key, prices must be entered manually and peer analysis is limited.

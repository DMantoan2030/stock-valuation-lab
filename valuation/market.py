"""Market data: stock prices, market caps, beta and peer lists (Finnhub), plus the
10-year U.S. Treasury yield (U.S. Treasury, no key needed).

Finnhub's free tier allows about 60 requests per minute. Get a free key at
https://finnhub.io/register and store it as FINNHUB_API_KEY (never in the code).
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime, timezone

import requests

FINNHUB = "https://finnhub.io/api/v1"
TREASURY_CSV = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
                "&field_tdr_date_value={year}&page&_format=csv")


class MarketDataError(Exception):
    pass


def _finnhub(path: str, key: str, **params) -> dict | list:
    if not key:
        raise MarketDataError("No Finnhub API key configured.")
    resp = requests.get(f"{FINNHUB}/{path}", params={**params, "token": key}, timeout=20)
    if resp.status_code == 401:
        raise MarketDataError("Finnhub rejected the API key.")
    if resp.status_code == 429:
        raise MarketDataError("Finnhub rate limit reached. Wait a minute and try again.")
    resp.raise_for_status()
    return resp.json()


@dataclass
class Quote:
    price: float
    as_of: str
    source: str


def get_quote(ticker: str, key: str) -> Quote:
    data = _finnhub("quote", key, symbol=ticker.upper())
    price = data.get("c")
    if not price:
        raise MarketDataError(f"Finnhub has no current price for {ticker}.")
    ts = data.get("t")
    as_of = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts else "unknown"
    return Quote(float(price), as_of, "Finnhub /quote")


def get_profile(ticker: str, key: str) -> dict:
    """Company profile. marketCapitalization is in millions of USD."""
    return _finnhub("stock/profile2", key, symbol=ticker.upper()) or {}


def get_beta(ticker: str, key: str) -> float | None:
    data = _finnhub("stock/metric", key, symbol=ticker.upper(), metric="all") or {}
    beta = (data.get("metric") or {}).get("beta")
    return float(beta) if beta else None


def get_peers(ticker: str, key: str) -> list[str]:
    data = _finnhub("stock/peers", key, symbol=ticker.upper()) or []
    return [p.upper() for p in data if isinstance(p, str) and p.upper() != ticker.upper()]


@dataclass
class RiskFree:
    rate: float
    as_of: str
    source: str


def get_treasury_10y() -> RiskFree:
    """Latest 10-year par yield from the U.S. Treasury's daily yield curve."""
    for year in (date.today().year, date.today().year - 1):
        try:
            resp = requests.get(TREASURY_CSV.format(year=year), timeout=30)
            resp.raise_for_status()
            rows = list(csv.DictReader(io.StringIO(resp.text)))
            if rows and rows[0].get("10 Yr"):
                return RiskFree(float(rows[0]["10 Yr"]) / 100, rows[0]["Date"], "U.S. Treasury daily par yield curve")
        except (requests.RequestException, ValueError, KeyError):
            continue
    return RiskFree(0.045, "n/a", "Default (Treasury data unavailable)")

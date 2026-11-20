"""Competitor suggestions.

1. Start from Finnhub's peer list (same industry classification).
2. Keep only companies that file with the SEC (so their 10-K/10-Q data is available).
3. Prefer peers in the same SEC industry code (SIC), first 4-digit, then 2-digit matches.
4. Rank by how close their market cap is to the company's (0.2x to 5x preferred).

Business-model similarity cannot be judged reliably from free data, so users can edit the list.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import market, sec


@dataclass
class PeerSuggestion:
    ticker: str
    name: str
    market_cap: float | None   # USD
    sic: str
    reason: str
    score: float


def suggest_peers(ticker: str, sic: str, market_cap: float | None, finnhub_key: str,
                  user_agent: str, ticker_map: dict, max_peers: int = 4,
                  max_candidates: int = 8) -> tuple[list[PeerSuggestion], list[str]]:
    """Raises market.MarketDataError if Finnhub is unavailable or busy, so a failed
    attempt is never cached and the next visitor retries."""
    notes: list[str] = []
    candidates = market.get_peers(ticker, finnhub_key)

    candidates = [c for c in candidates if c in ticker_map][:max_candidates]
    if not candidates:
        return [], ["No SEC-registered peers found. Enter competitors manually."]

    suggestions = []
    for c in candidates:
        try:
            subs = sec.get_submissions(ticker_map[c]["cik"], user_agent)
            peer_sic = str(subs.get("sic") or "")
        except Exception:
            peer_sic = ""
        try:
            prof = market.get_profile(c, finnhub_key)
            mcap = float(prof["marketCapitalization"]) * 1e6 if prof.get("marketCapitalization") else None
        except market.MarketDataError:
            raise  # busy or rate-limited: fail the whole lookup rather than cache a partial ranking
        except Exception:
            mcap = None

        reasons = ["same Finnhub industry"]
        score = 0.0
        if peer_sic and sic:
            if peer_sic == sic:
                score += 2
                reasons.append(f"same SIC code {sic}")
            elif peer_sic[:2] == sic[:2]:
                score += 1
                reasons.append(f"same SIC group {sic[:2]}xx")
            else:
                score -= 1
                reasons.append(f"different SIC ({peer_sic})")
        if mcap and market_cap:
            ratio = mcap / market_cap
            score -= abs(math.log(ratio))           # closer in size scores higher
            if 0.2 <= ratio <= 5:
                score += 1
                reasons.append(f"similar size ({ratio:.1f}x market cap)")
            else:
                reasons.append(f"size differs ({ratio:.2f}x market cap)")
        suggestions.append(PeerSuggestion(c, ticker_map[c]["title"], mcap, peer_sic, "; ".join(reasons), score))

    suggestions.sort(key=lambda s: s.score, reverse=True)
    if len(suggestions) < 3:
        notes.append("Fewer than 3 suggested peers found; consider adding competitors manually.")
    return suggestions[:max_peers], notes

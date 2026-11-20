"""Saved watchlist scans, kept in the app's memory for 24 hours.

Scans are grouped under a private key the user types (stored only as a SHA-256 hash),
so people who open the public app cannot see someone else's watchlists. This is a
convenience lock, not bank-grade security: anyone who knows or guesses the key can
see the saved scans. Everything is lost if the app restarts; results can also be
downloaded as CSV.
"""
from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

KEEP_FOR = timedelta(hours=24)
MAX_SCANS_PER_KEY = 20


def _hash(key: str) -> str:
    return hashlib.sha256(key.strip().encode("utf-8")).hexdigest()


@dataclass
class SavedScan:
    name: str
    tickers: list[str]
    rows: list          # list[ScanRow]; the same list object the running scan appends to
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "done"            # "running" | "done" | "stopped" | "error"
    message: str = ""               # what the background scan is doing right now
    stop: threading.Event = field(default_factory=threading.Event)

    @property
    def done(self) -> int:
        return len(self.rows)

    @property
    def running(self) -> bool:
        return self.status == "running"


class ScanStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scans: dict[str, list[SavedScan]] = {}

    def _purge(self, now: datetime) -> None:
        for k in list(self._scans):
            # never drop a scan that is still running
            self._scans[k] = [s for s in self._scans[k] if s.running or now - s.created < KEEP_FOR]
            if not self._scans[k]:
                del self._scans[k]

    def save(self, key: str, scan: SavedScan, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        with self._lock:
            self._purge(now)
            bucket = self._scans.setdefault(_hash(key), [])
            bucket.insert(0, scan)
            del bucket[MAX_SCANS_PER_KEY:]

    def list(self, key: str, now: datetime | None = None) -> list[SavedScan]:
        now = now or datetime.now(timezone.utc)
        with self._lock:
            self._purge(now)
            return list(self._scans.get(_hash(key), []))

    def get(self, key: str, scan_id: str) -> SavedScan | None:
        return next((s for s in self.list(key) if s.scan_id == scan_id), None)

    def delete(self, key: str, scan_id: str) -> None:
        with self._lock:
            k = _hash(key)
            for s in self._scans.get(k, []):
                if s.scan_id == scan_id:
                    s.stop.set()            # deleting a running scan also stops it
            self._scans[k] = [s for s in self._scans.get(k, []) if s.scan_id != scan_id]

    def running(self, key: str) -> SavedScan | None:
        """The scan currently running under this key, if any (only one at a time)."""
        return next((s for s in self.list(key) if s.running), None)

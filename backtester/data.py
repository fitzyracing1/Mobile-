"""
Historical data fetching and caching.

Pulls daily OANDA candles via the v20 REST API and caches them as CSV
files so repeat runs are instant.  The cache is invalidated automatically
when today's date is newer than the most recent cached bar.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

CACHE_DIR = Path(__file__).parent / ".cache"

OANDA_INSTRUMENT: Dict[str, str] = {
    "EUR/USD": "EUR_USD",
    "GBP/USD": "GBP_USD",
    "AUD/USD": "AUD_USD",
    "NZD/USD": "NZD_USD",
    "USD/CHF": "USD_CHF",
    "USD/JPY": "USD_JPY",
    "USD/CAD": "USD_CAD",
}


@dataclass
class DailyBar:
    date: str       # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    volume: float


def _cache_path(pair: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{pair.replace('/', '_')}_D.csv"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False
    last_date = rows[-1]["date"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Fresh if the most recent bar is within the last 3 days
    # (weekends + holidays mean the last bar may not be today)
    from datetime import timedelta
    last_dt = datetime.strptime(last_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last_dt).days <= 3


def _fetch_from_oanda(pair: str, client, count: int = 5000) -> List[DailyBar]:
    import oandapyV20.endpoints.instruments as ep_instruments

    instrument = OANDA_INSTRUMENT[pair]
    params = {"count": count, "granularity": "D", "price": "M"}
    r = ep_instruments.InstrumentsCandles(instrument, params=params)
    client.request(r)

    bars = []
    for c in r.response.get("candles", []):
        if not c.get("complete", True):
            continue
        mid = c["mid"]
        date = c["time"][:10]  # YYYY-MM-DD
        bars.append(DailyBar(
            date=date,
            open=float(mid["o"]),
            high=float(mid["h"]),
            low=float(mid["l"]),
            close=float(mid["c"]),
            volume=float(c.get("volume", 0)),
        ))
    return bars


def _save_cache(pair: str, bars: List[DailyBar]) -> None:
    path = _cache_path(pair)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date","open","high","low","close","volume"])
        writer.writeheader()
        for b in bars:
            writer.writerow({"date":b.date,"open":b.open,"high":b.high,
                             "low":b.low,"close":b.close,"volume":b.volume})


def _load_cache(pair: str) -> List[DailyBar]:
    path = _cache_path(pair)
    bars = []
    with open(path) as f:
        for row in csv.DictReader(f):
            bars.append(DailyBar(
                date=row["date"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))
    return bars


def load_history(
    pair: str,
    client=None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[DailyBar]:
    """
    Return daily bars for a pair, filtered to [start_date, end_date].
    Fetches from OANDA if cache is stale; loads from CSV otherwise.
    """
    path = _cache_path(pair)

    if not _cache_is_fresh(path):
        if client is None:
            raise RuntimeError(f"Cache missing/stale for {pair} and no OANDA client provided.")
        bars = _fetch_from_oanda(pair, client)
        _save_cache(pair, bars)
    else:
        bars = _load_cache(pair)

    if start_date:
        bars = [b for b in bars if b.date >= start_date]
    if end_date:
        bars = [b for b in bars if b.date <= end_date]

    return bars


def load_all(
    pairs: List[str],
    client=None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, List[DailyBar]]:
    return {p: load_history(p, client, start_date, end_date) for p in pairs}

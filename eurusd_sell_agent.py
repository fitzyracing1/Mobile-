#!/usr/bin/env python3
"""Automated EUR/USD sell-signal agent.

This agent treats "the Dutch on the US dollar" as EUR vs USD:
- SELL_EUR_BUY_USD when EUR weakens meaningfully against USD
- HOLD otherwise

Data source:
- European Central Bank EUR/USD reference feed
  https://www.ecb.europa.eu/rss/fxref-usd.html
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ECB_RSS_USD = "https://www.ecb.europa.eu/rss/fxref-usd.html"


@dataclass
class FxSnapshot:
    """One EUR/USD observation from ECB."""

    date: datetime
    eurusd: float


@dataclass
class AgentSignal:
    """Decision payload emitted by the agent."""

    action: str
    confidence: str
    reason: str
    latest_eurusd: float
    previous_eurusd: float
    day_change_bps: float
    three_day_trend: str
    generated_at_utc: str
    source: str = ECB_RSS_USD

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))


class EurUsdSellAgent:
    """Simple threshold + trend EUR/USD sell agent."""

    def __init__(self, sell_drop_bps: float = 10.0) -> None:
        self.sell_drop_bps = sell_drop_bps

    def fetch_snapshots(self) -> List[FxSnapshot]:
        req = urllib.request.Request(
            ECB_RSS_USD,
            headers={"User-Agent": "eurusd-sell-agent/1.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read()

        root = ET.fromstring(raw)
        ns = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "cb": "http://www.cbwiki.net/wiki/index.php/Specification_1.1",
            "dc": "http://purl.org/dc/elements/1.1/",
        }

        items = []
        rss_ns_item = "{http://purl.org/rss/1.0/}item"
        feed_items = root.findall(rss_ns_item)
        if not feed_items:
            # Fallback in case provider changes namespace behavior.
            feed_items = root.findall("item")

        for item in feed_items:
            rate_node = item.find(".//cb:value", ns)
            date_node = item.find(".//dc:date", ns)
            if rate_node is None or date_node is None:
                continue
            try:
                rate = float(rate_node.text or "")
                date = datetime.fromisoformat((date_node.text or "").strip())
            except ValueError:
                continue
            items.append(FxSnapshot(date=date, eurusd=rate))

        items.sort(key=lambda x: x.date, reverse=True)
        if len(items) < 3:
            raise RuntimeError("Not enough ECB points to make decision.")
        return items

    def decide(self, snapshots: List[FxSnapshot]) -> AgentSignal:
        latest = snapshots[0]
        previous = snapshots[1]
        older = snapshots[2]

        day_change_bps = ((latest.eurusd - previous.eurusd) / previous.eurusd) * 10000
        three_day_down = latest.eurusd < previous.eurusd < older.eurusd

        if day_change_bps <= -self.sell_drop_bps and three_day_down:
            action = "SELL_EUR_BUY_USD"
            confidence = "high"
            reason = (
                f"EUR/USD dropped {abs(day_change_bps):.1f} bps day-over-day and "
                "is in a 3-day downward trend."
            )
        elif day_change_bps < 0:
            action = "WAIT_SELL_SETUP"
            confidence = "medium"
            reason = (
                f"EUR/USD is down {abs(day_change_bps):.1f} bps day-over-day, "
                "but trend/threshold confirmation is incomplete."
            )
        else:
            action = "HOLD"
            confidence = "low"
            reason = "EUR/USD is flat/up day-over-day; no EUR sell signal."

        trend = "down" if three_day_down else "mixed_or_up"
        return AgentSignal(
            action=action,
            confidence=confidence,
            reason=reason,
            latest_eurusd=latest.eurusd,
            previous_eurusd=previous.eurusd,
            day_change_bps=round(day_change_bps, 2),
            three_day_trend=trend,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_once(agent: EurUsdSellAgent, output: Path | None) -> AgentSignal:
    signal = agent.decide(agent.fetch_snapshots())
    print(signal.to_json())
    if output:
        append_line(output, signal.to_json())
    return signal


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Agent that sells EUR (Dutch euro) against USD on confirmed weakness."
    )
    parser.add_argument(
        "--sell-drop-bps",
        type=float,
        default=10.0,
        help="Minimum day-over-day EUR/USD drop (in bps) to confirm SELL with trend.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=1800,
        help="Loop interval in seconds when --loop is enabled (default: 1800).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSONL output path for emitted signals.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep running and emit periodic signals.",
    )
    args = parser.parse_args()

    agent = EurUsdSellAgent(sell_drop_bps=args.sell_drop_bps)

    if not args.loop:
        run_once(agent, args.output)
        return 0

    while True:
        try:
            run_once(agent, args.output)
        except Exception as exc:  # pragma: no cover - defensive runtime path
            err = {
                "action": "ERROR",
                "reason": str(exc),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            line = json.dumps(err, separators=(",", ":"))
            print(line)
            if args.output:
                append_line(args.output, line)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

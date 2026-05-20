#!/usr/bin/env python3
"""
Entry point for the Dutch Agent.

Usage
-----
  # Paper-trade for 20 ticks then print summary:
  python main.py --ticks 20

  # Run live (one tick every 5 seconds, Ctrl-C to stop):
  python main.py --live

  # Customise parameters:
  python main.py --ticks 50 --position-size 5000 --stop-loss 0.75 --take-profit 1.5
"""

import argparse
import sys

from dutch_agent import DutchAgent
from dutch_agent.config import AgentConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Dutch Agent — sells the US Dollar across G10 FX pairs"
    )
    p.add_argument(
        "--ticks",
        type=int,
        default=30,
        metavar="N",
        help="Number of ticks to simulate (default: 30). Ignored in --live mode.",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Run indefinitely with real-time sleep between ticks (Ctrl-C to stop).",
    )
    p.add_argument(
        "--position-size",
        type=float,
        default=10_000.0,
        metavar="USD",
        help="Notional size per position in USD (default: 10000).",
    )
    p.add_argument(
        "--max-positions",
        type=int,
        default=4,
        metavar="N",
        help="Maximum concurrent open positions (default: 4).",
    )
    p.add_argument(
        "--stop-loss",
        type=float,
        default=1.0,
        metavar="PCT",
        help="Stop-loss as %% of entry price (default: 1.0).",
    )
    p.add_argument(
        "--take-profit",
        type=float,
        default=2.0,
        metavar="PCT",
        help="Take-profit as %% of entry price (default: 2.0).",
    )
    p.add_argument(
        "--tick-interval",
        type=int,
        default=5,
        metavar="SECS",
        help="Seconds between ticks in live mode (default: 5).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-tick logging; only print final summary.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    config = AgentConfig(
        position_size_usd=args.position_size,
        max_positions=args.max_positions,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        tick_interval_seconds=args.tick_interval,
        verbose=not args.quiet,
    )

    agent = DutchAgent(config=config)

    if args.live:
        agent.run(max_ticks=None)
    else:
        agent.run(max_ticks=args.ticks)

    return 0


if __name__ == "__main__":
    sys.exit(main())

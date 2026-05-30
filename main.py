#!/usr/bin/env python3
"""
Entry point for the Dutch Agent.

Usage
-----
  # Paper-trade for 20 ticks then print summary:
  python3 main.py --ticks 20

  # Run live on OANDA practice account:
  python3 main.py --live --oanda --oanda-token YOUR_TOKEN --oanda-account 101-001-XXXXX-001

  # Customise parameters:
  python3 main.py --ticks 50 --position-size 5000 --stop-loss 0.75 --take-profit 1.5
"""

import argparse
import os
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

    # OANDA integration
    oanda = p.add_argument_group("OANDA (real trading)")
    oanda.add_argument(
        "--oanda",
        action="store_true",
        help="Use OANDA for real prices and order execution.",
    )
    oanda.add_argument(
        "--oanda-token",
        metavar="TOKEN",
        default=os.environ.get("OANDA_API_TOKEN", ""),
        help="OANDA API token (or set OANDA_API_TOKEN env var).",
    )
    oanda.add_argument(
        "--oanda-account",
        metavar="ID",
        default=os.environ.get("OANDA_ACCOUNT_ID", ""),
        help="OANDA account ID (or set OANDA_ACCOUNT_ID env var).",
    )
    oanda.add_argument(
        "--oanda-env",
        metavar="ENV",
        default="practice",
        choices=["practice", "live"],
        help="OANDA environment: practice (default) or live.",
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

    if args.oanda:
        from dutch_agent.oanda import OandaAgent

        token = args.oanda_token
        account_id = args.oanda_account

        if not token:
            print("ERROR: --oanda-token (or OANDA_API_TOKEN env var) is required.")
            return 1
        if not account_id:
            print("ERROR: --oanda-account (or OANDA_ACCOUNT_ID env var) is required.")
            return 1

        agent = OandaAgent(
            token=token,
            account_id=account_id,
            environment=args.oanda_env,
            config=config,
        )
    else:
        agent = DutchAgent(config=config)

    if args.live:
        agent.run(max_ticks=None)
    else:
        agent.run(max_ticks=args.ticks)

    return 0


if __name__ == "__main__":
    sys.exit(main())

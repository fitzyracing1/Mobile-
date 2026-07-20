#!/usr/bin/env python3
"""
Backtesting CLI.

Usage
-----
  # Run all three strategies (requires OANDA token for first-time data fetch):
  python3 run_backtest.py --token YOUR_TOKEN

  # Use cached data (no token needed after first run):
  python3 run_backtest.py

  # Custom date range and parameters:
  python3 run_backtest.py --start 2022-01-01 --end 2026-07-01 \
      --stop-loss 2.0 --take-profit 4.0 --position-size 10000

  # Run only one strategy:
  python3 run_backtest.py --strategy rsi
"""

import argparse
import os
import sys

DEFAULT_PAIRS = [
    "EUR/USD", "GBP/USD", "AUD/USD", "NZD/USD",
    "USD/CHF", "USD/JPY", "USD/CAD",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Dutch Agent Backtester")
    p.add_argument("--token",  default=os.environ.get("OANDA_API_TOKEN", ""),
                   help="OANDA API token (needed only on first run to download data)")
    p.add_argument("--env",    default="practice", choices=["practice", "live"])
    p.add_argument("--start",  default="2020-01-01", metavar="YYYY-MM-DD")
    p.add_argument("--end",    default="",           metavar="YYYY-MM-DD",
                   help="End date (default: today)")
    p.add_argument("--stop-loss",      type=float, default=2.0,    metavar="PCT")
    p.add_argument("--take-profit",    type=float, default=4.0,    metavar="PCT")
    p.add_argument("--position-size",  type=float, default=10_000, metavar="USD")
    p.add_argument("--max-positions",  type=int,   default=3)
    p.add_argument("--strategy",       default="all",
                   choices=["all", "ma", "rsi", "carry"],
                   help="Which strategy to run (default: all)")
    return p.parse_args(argv)


def build_client(token: str, env: str):
    if not token:
        return None
    import oandapyV20
    return oandapyV20.API(access_token=token, environment=env)


def main(argv=None):
    args = parse_args(argv)

    from datetime import date
    end_date = args.end or date.today().isoformat()

    client = build_client(args.token, args.env)

    print(f"\nLoading historical daily data ({args.start} → {end_date}) …")
    from backtester.data import load_all
    try:
        history = load_all(DEFAULT_PAIRS, client=client,
                           start_date=args.start, end_date=end_date)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("Run with --token YOUR_OANDA_TOKEN to download data.")
        return 1

    dates_available = sorted({b.date for bars in history.values() for b in bars})
    print(f"Loaded {len(dates_available)} trading days across {len(DEFAULT_PAIRS)} pairs.\n")

    from backtester.engine  import BacktestEngine
    from backtester.metrics import compute, print_report

    engine_kwargs = dict(
        pairs=DEFAULT_PAIRS,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        position_size_usd=args.position_size,
        max_positions=args.max_positions,
    )

    period_str = f"{args.start} → {end_date}"

    strategies_to_run = []

    if args.strategy in ("all", "ma"):
        from backtester.strategies.ma_cross import MACrossStrategy
        strategies_to_run.append(MACrossStrategy())

    if args.strategy in ("all", "rsi"):
        from backtester.strategies.rsi_extreme import RSIExtremeStrategy
        strategies_to_run.append(RSIExtremeStrategy())

    if args.strategy in ("all", "carry"):
        from backtester.strategies.carry import CarryTradeStrategy
        strategies_to_run.append(CarryTradeStrategy())

    for strategy in strategies_to_run:
        engine = BacktestEngine(strategy=strategy, **engine_kwargs)
        trades = engine.run(history)
        m = compute(trades)
        print_report(strategy.name, period_str, m)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

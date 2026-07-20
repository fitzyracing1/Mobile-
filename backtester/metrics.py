"""Performance metrics for backtest results."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .engine import BacktestTrade


def compute(trades: List[BacktestTrade], starting_balance: float = 100_000.0) -> Dict:
    if not trades:
        return {"error": "No trades to analyse"}

    pnls = [t.pnl for t in trades]
    wins  = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate  = len(wins) / len(pnls) * 100
    avg_win   = float(np.mean(wins))   if wins   else 0.0
    avg_loss  = float(np.mean(losses)) if losses else 0.0
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Equity curve and max drawdown
    equity = starting_balance
    peak   = starting_balance
    equity_curve = [equity]
    max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd
        equity_curve.append(equity)

    total_return_pct = (equity - starting_balance) / starting_balance * 100

    # Annualised return
    if len(trades) >= 2:
        from datetime import datetime
        first = datetime.strptime(trades[0].entry_date, "%Y-%m-%d")
        last  = datetime.strptime(trades[-1].exit_date,  "%Y-%m-%d")
        years = max((last - first).days / 365.25, 0.01)
        ann_return = ((equity / starting_balance) ** (1 / years) - 1) * 100
    else:
        ann_return = 0.0

    # Sharpe ratio (simplified, using per-trade PnL as returns)
    if len(pnls) > 1:
        pnl_arr = np.array(pnls)
        sharpe = (np.mean(pnl_arr) / np.std(pnl_arr, ddof=1)) * math.sqrt(252) if np.std(pnl_arr) > 0 else 0.0
    else:
        sharpe = 0.0

    # Per-pair breakdown
    pair_pnl: Dict[str, float] = {}
    for t in trades:
        pair_pnl[t.pair] = pair_pnl.get(t.pair, 0.0) + t.pnl
    best_pair  = max(pair_pnl, key=pair_pnl.get)
    worst_pair = min(pair_pnl, key=pair_pnl.get)

    return {
        "total_trades":    len(trades),
        "winning_trades":  len(wins),
        "losing_trades":   len(losses),
        "win_rate_pct":    round(win_rate, 1),
        "avg_win":         round(avg_win, 2),
        "avg_loss":        round(avg_loss, 2),
        "profit_factor":   round(profit_factor, 2),
        "total_pnl":       round(total_pnl, 2),
        "total_return_pct":round(total_return_pct, 2),
        "ann_return_pct":  round(ann_return, 2),
        "max_drawdown_pct":round(max_dd, 2),
        "sharpe_ratio":    round(sharpe, 2),
        "final_equity":    round(equity, 2),
        "best_pair":       f"{best_pair} (${pair_pnl[best_pair]:+,.0f})",
        "worst_pair":      f"{worst_pair} (${pair_pnl[worst_pair]:+,.0f})",
        "pair_pnl":        {k: round(v, 2) for k, v in sorted(pair_pnl.items(), key=lambda x: -x[1])},
    }


def print_report(strategy_name: str, period: str, m: Dict) -> None:
    if "error" in m:
        print(f"  {m['error']}")
        return

    w = 56
    print("═" * w)
    print(f"  Strategy : {strategy_name}")
    print(f"  Period   : {period}")
    print("─" * w)
    print(f"  Total return      : {m['total_return_pct']:>+8.2f}%")
    print(f"  Annualised return : {m['ann_return_pct']:>+8.2f}%")
    print(f"  Max drawdown      : {m['max_drawdown_pct']:>8.2f}%")
    print(f"  Sharpe ratio      : {m['sharpe_ratio']:>8.2f}")
    print("─" * w)
    print(f"  Total trades      : {m['total_trades']:>8}")
    print(f"  Win rate          : {m['win_rate_pct']:>8.1f}%")
    print(f"  Avg win           : ${m['avg_win']:>+8.2f}")
    print(f"  Avg loss          : ${m['avg_loss']:>+8.2f}")
    print(f"  Profit factor     : {m['profit_factor']:>8.2f}")
    print(f"  Total P&L         : ${m['total_pnl']:>+10,.2f}")
    print("─" * w)
    print(f"  Best pair  : {m['best_pair']}")
    print(f"  Worst pair : {m['worst_pair']}")
    print("─" * w)
    print("  Per-pair P&L:")
    for pair, pnl in m["pair_pnl"].items():
        bar_len = int(abs(pnl) / 50)
        bar = ("█" * min(bar_len, 20)).ljust(20)
        sign = "+" if pnl >= 0 else "-"
        print(f"    {pair:<10} {sign}{bar} ${pnl:>+,.0f}")
    print("═" * w)

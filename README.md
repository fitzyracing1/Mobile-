# Dutch Agent — sells the US Dollar

A paper-trading agent whose sole thesis is to be **net short the US Dollar** across a basket of G10 FX pairs.

The strategy is named after the *Dutch Book* concept from probability theory: a set of bets constructed so that one side always wins regardless of outcome. Here the agent continuously looks for technical setups where the USD is weakening across multiple pairs simultaneously, building diversified short-USD exposure.

---

## Strategy overview

The agent monitors seven G10 USD pairs and combines three technical indicators into a single composite score:

| Indicator | Weight | Signal logic |
|-----------|--------|--------------|
| MA crossover (10/30 bar) | 40 % | Fast MA above/below slow MA |
| RSI (14 bar) | 35 % | Momentum direction |
| Bollinger Bands (20 bar, 2 σ) | 25 % | %B position for mean reversion |

A **negative composite score** (USD bearish) triggers a new short-USD position:

- On non-inverted pairs (EUR/USD, GBP/USD, AUD/USD, NZD/USD) the agent goes **LONG** the pair (long foreign currency, short USD).
- On inverted pairs (USD/CHF, USD/JPY, USD/CAD) the agent goes **SHORT** the pair (same net effect: short USD).

Each position carries a configurable stop-loss (default 1 %) and take-profit (default 2 %).

---

## Project layout

```
dutch_agent/
    __init__.py     # package entry-point
    config.py       # AgentConfig dataclass
    data.py         # synthetic price feed (GBM)
    signals.py      # MA cross, RSI, Bollinger, composite signal
    position.py     # Position, Trade, Portfolio
    agent.py        # DutchAgent main loop
main.py             # CLI entry point
tests/
    test_signals.py
    test_portfolio.py
    test_agent.py
requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt

# Simulate 30 ticks and print a summary:
python3 main.py

# Run 50 ticks with smaller position sizes:
python3 main.py --ticks 50 --position-size 5000

# Run in live mode (one tick every 5 seconds, Ctrl-C to stop):
python3 main.py --live

# All options:
python3 main.py --help
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--ticks N` | 30 | Number of simulation ticks |
| `--live` | off | Run indefinitely |
| `--position-size USD` | 10000 | Notional size per position |
| `--max-positions N` | 4 | Max concurrent open positions |
| `--stop-loss PCT` | 1.0 | Stop-loss % of entry price |
| `--take-profit PCT` | 2.0 | Take-profit % of entry price |
| `--tick-interval SECS` | 5 | Sleep between ticks (live mode) |
| `--quiet` | off | Suppress per-tick log lines |

---

## Using the agent as a library

```python
from dutch_agent import DutchAgent
from dutch_agent.config import AgentConfig

config = AgentConfig(
    position_size_usd=5_000,
    max_positions=3,
    stop_loss_pct=0.75,
    take_profit_pct=1.5,
    verbose=True,
)

agent = DutchAgent(config=config)
agent.run(max_ticks=100)
```

---

## Connecting a real data feed

Replace `dutch_agent/data.py` with an implementation that reads from your broker (OANDA, Interactive Brokers, a crypto exchange via [ccxt](https://github.com/ccxt/ccxt), etc.).  The `DataFeed` interface only requires two methods:

- `warmup(n_bars)` — pre-populate histories
- `tick()` — return `{pair: Bar}` for the latest bar on each pair

---

## Running tests

```bash
python3 -m pytest tests/ -v
```

---

## Disclaimer

This is a **paper-trading simulation only**. No real orders are ever placed. Past simulated performance is not indicative of future results. Currency trading carries significant risk of loss.

# 3XB Sell Agent (EVM)

Simple Python agent that monitors your token price via GeckoTerminal and sells your
token on an EVM DEX router when rule conditions are met.

> This is provided for automation and learning. Use carefully with small size first.
> You are fully responsible for keys, funds, and trade outcomes.

## What it does

- Watches token USD price from GeckoTerminal
- Checks your wallet token balance
- Triggers a sell based on:
  - `TAKE_PROFIT_USD`
  - `STOP_LOSS_USD`
  - or manual `--sell-now`
- Swaps through a Uniswap V2 style router path (`SELL_PATH`)
- Supports dry-run mode (enabled by default)

## Quick start

1. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment:

```bash
cp .env.example .env
```

3. Edit `.env`:

- Set `RPC_URL` and `PRIVATE_KEY`
- Verify `TOKEN_ADDRESS`, `ROUTER_ADDRESS`, and `SELL_PATH`
- Keep `DRY_RUN=true` for first validation

4. Run one manual sell cycle (still dry-run unless disabled):

```bash
python agent.py --sell-now
```

5. Run rule-based mode:

```bash
python agent.py
```

## Configuration notes

- `SELL_PERCENT=1.0` sells 100% of wallet balance when triggered
- `SLIPPAGE_BPS=300` means 3% slippage tolerance
- `SELL_ONCE=true` runs one cycle then exits
- Set `USE_FEE_ON_TRANSFER=true` for fee-on-transfer tokens

## Safer rollout checklist

1. Dry-run with real config and `--sell-now`
2. Set `SELL_PERCENT` to a small value (e.g. `0.05`)
3. Disable dry-run and test with tiny amount
4. Increase gradually if behavior is correct

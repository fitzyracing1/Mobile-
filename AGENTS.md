# AGENTS.md

## Cursor Cloud specific instructions

### Repository structure

The `main` branch contains only a stub README. All actual project code lives on separate feature branches:

| Branch | Project | Has Tests |
|--------|---------|-----------|
| `cursor/amsterdam-monitor-browser-a7a1` | EUR/USD sell signal agent + HTML dashboard | No |
| `cursor/crypto-sell-agent-ea05` | 3XB EVM crypto sell agent | No |
| `cursor/mars-ai-agent-4892` | Mars autonomous AI agent | Yes (`tests/`) |

### Running each project

**Mars AI Agent** (branch: `cursor/mars-ai-agent-4892`):
```bash
# Install in editable mode (from branch worktree/checkout)
pip install -e .
# Run tests
python3 -m unittest discover -s tests -v
# Run CLI self-directed mode
python3 -m mars_ai --self-directed --cycles 3
# Run interactive mode
python3 -m mars_ai
```

**EUR/USD Sell Agent** (branch: `cursor/amsterdam-monitor-browser-a7a1`):
```bash
# Single run (no dependencies beyond stdlib)
python3 eurusd_sell_agent.py
# Loop mode (runs every 30 min)
python3 eurusd_sell_agent.py --loop --interval-seconds 1800
```

**Crypto Sell Agent** (branch: `cursor/crypto-sell-agent-ea05`):
```bash
# Requires .env file (copy .env.example and fill in RPC_URL + PRIVATE_KEY)
cp .env.example .env
# Edit .env with real credentials, then:
python3 agent.py
# For immediate sell (bypasses price thresholds):
python3 agent.py --sell-now
```

### Key caveats

- Python 3.10+ is required (3.12 works fine).
- The crypto agent needs `web3`, `python-dotenv`, and `requests` (`pip install web3 python-dotenv requests`).
- The crypto agent requires a valid Ethereum RPC URL and private key to function beyond import checks. Use `DRY_RUN=true` in `.env` for safe testing without real transactions.
- The EUR/USD agent and Mars AI agent have zero external pip dependencies (stdlib only).
- There is no unified lint configuration or CI/CD across the repo. Each branch is an independent micro-project.
- Use `git worktree add /tmp/<name> origin/<branch>` to work on multiple branches simultaneously without switching.

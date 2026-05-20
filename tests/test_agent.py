"""Integration tests for the DutchAgent end-to-end loop."""

import pytest

from dutch_agent import DutchAgent
from dutch_agent.config import AgentConfig


def make_config(**kwargs) -> AgentConfig:
    defaults = dict(
        warmup_bars=35,      # enough for slow_ma=30
        max_positions=2,
        position_size_usd=1_000,
        stop_loss_pct=1.0,
        take_profit_pct=2.0,
        verbose=False,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


class TestDutchAgent:
    def test_warmup_populates_histories(self):
        agent = DutchAgent(config=make_config())
        agent.warmup()
        for pair in agent.config.pairs:
            hist = agent.feed.history(pair)
            assert len(hist.bars) == agent.config.warmup_bars

    def test_single_tick_returns_summary(self):
        agent = DutchAgent(config=make_config())
        agent.warmup()
        summary = agent.tick()
        assert "equity" in summary
        assert "total_pnl" in summary
        assert "open_positions" in summary

    def test_run_fixed_ticks(self):
        agent = DutchAgent(config=make_config())
        agent.run(max_ticks=10)
        assert agent._tick_count == 10

    def test_portfolio_equity_never_negative(self):
        """With SL/TP in place equity should stay well above 0 in 50 ticks."""
        agent = DutchAgent(config=make_config(position_size_usd=500))
        agent.warmup()
        prices = {}
        for _ in range(50):
            agent.tick()
        prices = {p: agent.feed.price(p) for p in agent.config.pairs}
        assert agent.portfolio.equity(prices) > 0

    def test_max_positions_respected(self):
        agent = DutchAgent(config=make_config(max_positions=2))
        agent.warmup()
        for _ in range(20):
            agent.tick()
        assert len(agent.portfolio.positions) <= 2

    def test_inverted_pairs_use_short_side(self):
        from dutch_agent.config import INVERTED_PAIRS
        from dutch_agent.position import Side

        agent = DutchAgent(config=make_config(max_positions=7, warmup_bars=40))
        agent.run(max_ticks=30)

        for pair, pos in agent.portfolio.positions.items():
            if pair in INVERTED_PAIRS:
                assert pos.side == Side.SHORT, (
                    f"{pair} is an inverted pair but was opened as LONG"
                )
            else:
                assert pos.side == Side.LONG, (
                    f"{pair} is a normal pair but was opened as SHORT"
                )

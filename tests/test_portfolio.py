"""Tests for the position/portfolio management layer."""

import pytest

from dutch_agent.position import Portfolio, Side, Trade


class TestPortfolio:
    def setup_method(self):
        self.portfolio = Portfolio(starting_balance=100_000.0)

    def test_initial_state(self):
        assert self.portfolio.cash == 100_000.0
        assert len(self.portfolio.positions) == 0
        assert len(self.portfolio.trades) == 0

    def test_open_position_debits_cash(self):
        self.portfolio.open_position(
            pair="EUR/USD", side=Side.LONG, price=1.085,
            size_usd=10_000, stop_loss_pct=1.0, take_profit_pct=2.0,
        )
        assert self.portfolio.cash == 90_000.0
        assert "EUR/USD" in self.portfolio.positions

    def test_cannot_open_duplicate_position(self):
        kw = dict(pair="EUR/USD", side=Side.LONG, price=1.085,
                  size_usd=10_000, stop_loss_pct=1.0, take_profit_pct=2.0)
        pos1 = self.portfolio.open_position(**kw)
        pos2 = self.portfolio.open_position(**kw)
        assert pos1 is not None
        assert pos2 is None
        assert len(self.portfolio.positions) == 1

    def test_close_position_credits_cash(self):
        self.portfolio.open_position(
            pair="EUR/USD", side=Side.LONG, price=1.085,
            size_usd=10_000, stop_loss_pct=1.0, take_profit_pct=2.0,
        )
        trade = self.portfolio.close_position("EUR/USD", current_price=1.096, reason="take_profit")
        assert trade is not None
        assert trade.pnl > 0
        assert "EUR/USD" not in self.portfolio.positions
        assert self.portfolio.cash > 90_000.0  # cash restored + profit

    def test_stop_loss_triggers(self):
        self.portfolio.open_position(
            pair="EUR/USD", side=Side.LONG, price=1.085,
            size_usd=10_000, stop_loss_pct=1.0, take_profit_pct=2.0,
        )
        pos = self.portfolio.positions["EUR/USD"]
        sl_price = pos.stop_loss - 0.001  # below stop
        assert pos.is_stopped(sl_price)
        assert not pos.is_stopped(1.090)  # above stop

    def test_take_profit_triggers(self):
        self.portfolio.open_position(
            pair="EUR/USD", side=Side.LONG, price=1.085,
            size_usd=10_000, stop_loss_pct=1.0, take_profit_pct=2.0,
        )
        pos = self.portfolio.positions["EUR/USD"]
        tp_price = pos.take_profit + 0.001  # above target
        assert pos.is_target_hit(tp_price)
        assert not pos.is_target_hit(1.086)

    def test_win_rate_calculation(self):
        # Simulate two wins and one loss
        for pnl_sign, exit_price in [(1, 1.10), (1, 1.10), (-1, 1.07)]:
            pair = f"TEST/{pnl_sign}/{exit_price}"
            self.portfolio.open_position(
                pair=pair, side=Side.LONG, price=1.085,
                size_usd=1_000, stop_loss_pct=2.0, take_profit_pct=4.0,
            )
            self.portfolio.close_position(pair, exit_price, "test")

        assert self.portfolio.win_rate() == pytest.approx(2 / 3)

    def test_equity_reflects_unrealised_pnl(self):
        self.portfolio.open_position(
            pair="EUR/USD", side=Side.LONG, price=1.0,
            size_usd=10_000, stop_loss_pct=1.0, take_profit_pct=2.0,
        )
        equity_flat = self.portfolio.equity({"EUR/USD": 1.0})
        equity_up = self.portfolio.equity({"EUR/USD": 1.1})
        assert equity_up > equity_flat

    def test_short_position_pnl(self):
        self.portfolio.open_position(
            pair="USD/JPY", side=Side.SHORT, price=150.0,
            size_usd=10_000, stop_loss_pct=1.0, take_profit_pct=2.0,
        )
        trade = self.portfolio.close_position("USD/JPY", current_price=147.0, reason="take_profit")
        assert trade is not None
        assert trade.pnl > 0  # USD fell → short USD/JPY profits

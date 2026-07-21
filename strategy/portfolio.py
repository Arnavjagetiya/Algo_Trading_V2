"""
strategy/portfolio.py
=========================================================
Portfolio Management Engine

Author : Arnav
Python : 3.12+
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ============================================================
# POSITION
# ============================================================


@dataclass(slots=True)
class Position:

    ticker: str

    quantity: float = 0.0

    average_price: float = 0.0

    market_price: float = 0.0

    market_value: float = 0.0

    unrealized_pnl: float = 0.0

    realized_pnl: float = 0.0


# ============================================================
# PORTFOLIO
# ============================================================


class Portfolio:

    """
    Portfolio Engine

    ✓ Cash
    ✓ Holdings
    ✓ Market Value
    ✓ Equity
    ✓ NAV
    ✓ Buying Power
    """

    def __init__(

        self,

        initial_capital: float = 1_000_000,

    ):

        self.initial_capital = float(

            initial_capital

        )

        self.cash = float(

            initial_capital

        )

        self.positions: dict[str, Position] = {}

        self.history = []

    def initialize(self, capital: float | None = None):

        """Reset the portfolio state for a new backtest."""

        self.initial_capital = float(capital if capital is not None else self.initial_capital)
        self.cash = self.initial_capital
        self.positions = {}
        self.history = []

    def set_target_weight(self, ticker: str, weight: float):

        """Store a target weight for later rebalance operations."""

        if not hasattr(self, "target_weights"):

            self.target_weights = {}

        self.target_weights[ticker] = float(weight)

    # ========================================================
    # POSITION
    # ========================================================

    def get_position(

        self,

        ticker: str,

    ) -> Position:

        if ticker not in self.positions:

            self.positions[ticker] = Position(

                ticker=ticker

            )

        return self.positions[ticker]

    # --------------------------------------------------------

    def has_position(

        self,

        ticker,

    ):

        return (

            ticker

            in self.positions

            and

            self.positions[ticker].quantity != 0

        )

    # ========================================================
    # BUY
    # ========================================================

    def buy(

        self,

        ticker,

        quantity,

        price,

        commission=0,

    ):

        cost = (

            quantity

            * price

            + commission

        )

        if cost > self.cash:

            raise ValueError(

                "Insufficient cash."

            )

        position = self.get_position(

            ticker

        )

        total_cost = (

            position.average_price

            * position.quantity

        )

        total_cost += quantity * price

        position.quantity += quantity

        position.average_price = (

            total_cost

            / position.quantity

        )

        self.cash -= cost

    # ========================================================
    # SELL
    # ========================================================

    def sell(

        self,

        ticker,

        quantity,

        price,

        commission=0,

    ):

        if not self.has_position(

            ticker

        ):

            raise ValueError(

                "No position."

            )

        position = self.positions[ticker]

        if quantity > position.quantity:

            raise ValueError(

                "Quantity exceeds holding."

            )

        proceeds = (

            quantity

            * price

            - commission

        )

        pnl = (

            price

            - position.average_price

        ) * quantity

        position.quantity -= quantity

        position.realized_pnl += pnl

        self.cash += proceeds

        if position.quantity == 0:

            position.average_price = 0

            position.market_price = 0

            position.market_value = 0

            position.unrealized_pnl = 0

    # ========================================================
    # MARK TO MARKET
    # ========================================================

    def mark_to_market(

        self,

        prices: dict,

    ):

        for ticker, price in prices.items():

            if ticker not in self.positions:

                continue

            position = self.positions[ticker]

            position.market_price = price

            position.market_value = (

                position.quantity

                * price

            )

            position.unrealized_pnl = (

                price

                - position.average_price

            ) * position.quantity

    # ========================================================
    # VALUES
    # ========================================================

    @property
    def holdings_value(

        self,

    ):

        return sum(

            position.market_value

            for position in self.positions.values()

        )

    # --------------------------------------------------------

    @property
    def equity(

        self,

    ):

        return (

            self.cash

            + self.holdings_value

        )

    # --------------------------------------------------------

    @property
    def nav(

        self,

    ):

        return (

            self.equity

            / self.initial_capital

        )

    # --------------------------------------------------------

    @property
    def buying_power(

        self,

    ):

        return self.cash

    # ========================================================
    # HISTORY
    # ========================================================

    def snapshot(

        self,

        timestamp,

    ):

        self.history.append(

            {

                "Date": timestamp,

                "Cash": self.cash,

                "Holdings": self.holdings_value,

                "Equity": self.equity,

                "NAV": self.nav,

            }

        )

    # ========================================================
    # HOLDINGS
    # ========================================================

    def holdings(

        self,

    ):

        rows = []

        for position in self.positions.values():

            rows.append(

                {

                    "Ticker": position.ticker,

                    "Quantity": position.quantity,

                    "Average Price": position.average_price,

                    "Market Price": position.market_price,

                    "Market Value": position.market_value,

                    "Unrealized PnL": position.unrealized_pnl,

                    "Realized PnL": position.realized_pnl,

                }

            )

        return pd.DataFrame(rows)
    
    # ========================================================
    # REBALANCING
    # ========================================================

    def rebalance(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float],
        commission: float = 0.0,
    ):

        total_equity = self.equity

        for ticker, weight in target_weights.items():

            if ticker not in prices:
                continue

            target_value = (
                total_equity * weight
            )

            current_value = 0.0

            if ticker in self.positions:

                current_value = (
                    self.positions[ticker]
                    .market_value
                )

            difference = (
                target_value
                - current_value
            )

            price = prices[ticker]

            quantity = int(
                difference / price
            )

            if quantity > 0:

                self.buy(
                    ticker,
                    quantity,
                    price,
                    commission,
                )

            elif quantity < 0:

                self.sell(
                    ticker,
                    abs(quantity),
                    price,
                    commission,
                )

    # ========================================================
    # EXPOSURE
    # ========================================================

    @property
    def gross_exposure(
        self,
    ):

        return sum(

            abs(
                p.market_value
            )

            for p in self.positions.values()

        )

    # --------------------------------------------------------

    @property
    def net_exposure(
        self,
    ):

        return sum(

            p.market_value

            for p in self.positions.values()

        )

    # --------------------------------------------------------

    @property
    def leverage(
        self,
    ):

        if self.equity == 0:

            return 0.0

        return (

            self.gross_exposure

            / self.equity

        )

    # ========================================================
    # WEIGHTS
    # ========================================================

    def weights(
        self,
    ) -> pd.Series:

        if self.holdings_value == 0:

            return pd.Series(
                dtype=float
            )

        weights = {}

        for ticker, position in self.positions.items():

            if position.market_value == 0:

                continue

            weights[ticker] = (

                position.market_value

                / self.holdings_value

            )

        return pd.Series(
            weights
        ).sort_values(
            ascending=False
        )

    # ========================================================
    # RETURNS
    # ========================================================

    def history_dataframe(
        self,
    ) -> pd.DataFrame:

        if not self.history:

            return pd.DataFrame()

        df = pd.DataFrame(
            self.history
        )

        df.set_index(
            "Date",
            inplace=True,
        )

        return df

    # --------------------------------------------------------

    def portfolio_returns(
        self,
    ) -> pd.Series:

        history = self.history_dataframe()

        if history.empty:

            return pd.Series(
                dtype=float
            )

        return (

            history["Equity"]

            .pct_change()

            .dropna()

        )

    # --------------------------------------------------------

    def cumulative_returns(
        self,
    ) -> pd.Series:

        returns = self.portfolio_returns()

        if returns.empty:

            return returns

        return (

            1 + returns

        ).cumprod()

    # ========================================================
    # ALLOCATION HISTORY
    # ========================================================

    def allocation_snapshot(
        self,
        timestamp,
    ):

        snapshot = {

            "Date": timestamp

        }

        weights = self.weights()

        for ticker, weight in weights.items():

            snapshot[ticker] = weight

        return snapshot

    # --------------------------------------------------------

    def top_holdings(
        self,
        n: int = 10,
    ) -> pd.DataFrame:

        holdings = self.holdings()

        if holdings.empty:

            return holdings

        return holdings.sort_values(

            "Market Value",

            ascending=False,

        ).head(n)

    # ========================================================
    # PNL
    # ========================================================

    @property
    def realized_pnl(
        self,
    ):

        return sum(

            p.realized_pnl

            for p in self.positions.values()

        )

    # --------------------------------------------------------

    @property
    def unrealized_pnl(
        self,
    ):

        return sum(

            p.unrealized_pnl

            for p in self.positions.values()

        )

    # --------------------------------------------------------

    @property
    def total_pnl(
        self,
    ):

        return (

            self.realized_pnl

            + self.unrealized_pnl

        )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    def drawdown(
        self,
    ) -> pd.Series:

        equity = self.history_dataframe()

        if equity.empty:

            return pd.Series(
                dtype=float
            )

        curve = equity["Equity"]

        running_max = curve.cummax()

        return (
            curve
            - running_max
        ) / running_max

    # --------------------------------------------------------

    def max_drawdown(
        self,
    ) -> float:

        dd = self.drawdown()

        if dd.empty:

            return 0.0

        return float(dd.min())

    # --------------------------------------------------------

    def volatility(
        self,
        periods: int = 252,
    ) -> float:

        returns = self.portfolio_returns()

        if returns.empty:

            return 0.0

        return float(
            returns.std()
            * np.sqrt(periods)
        )

    # --------------------------------------------------------

    def sharpe_ratio(
        self,
        risk_free_rate: float = 0.06,
        periods: int = 252,
    ) -> float:

        returns = self.portfolio_returns()

        if returns.empty:

            return 0.0

        excess = (
            returns.mean() * periods
            - risk_free_rate
        )

        vol = self.volatility(periods)

        if vol == 0:

            return 0.0

        return excess / vol

    # --------------------------------------------------------

    def cagr(
        self,
        periods: int = 252,
    ) -> float:

        history = self.history_dataframe()

        if history.empty:

            return 0.0

        years = len(history) / periods

        if years <= 0:

            return 0.0

        ending = history["Equity"].iloc[-1]

        return (
            ending
            / self.initial_capital
        ) ** (1 / years) - 1

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            {
                "Initial Capital": [
                    self.initial_capital
                ],
                "Cash": [
                    self.cash
                ],
                "Holdings": [
                    self.holdings_value
                ],
                "Equity": [
                    self.equity
                ],
                "NAV": [
                    self.nav
                ],
                "Realized PnL": [
                    self.realized_pnl
                ],
                "Unrealized PnL": [
                    self.unrealized_pnl
                ],
                "Total PnL": [
                    self.total_pnl
                ],
                "Gross Exposure": [
                    self.gross_exposure
                ],
                "Net Exposure": [
                    self.net_exposure
                ],
                "Leverage": [
                    self.leverage
                ],
                "Volatility": [
                    self.volatility()
                ],
                "Sharpe": [
                    self.sharpe_ratio()
                ],
                "CAGR": [
                    self.cagr()
                ],
                "Max Drawdown": [
                    self.max_drawdown()
                ],
            }
        )

    # ========================================================
    # EXPORTS
    # ========================================================

    def export_history(
        self,
        filepath: str,
    ):

        self.history_dataframe().to_csv(
            filepath,
            index=True,
        )

    # --------------------------------------------------------

    def export_holdings(
        self,
        filepath: str,
    ):

        self.holdings().to_csv(
            filepath,
            index=False,
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ):

        self.cash = self.initial_capital

        self.positions.clear()

        self.history.clear()

    # ========================================================
    # INFORMATION
    # ========================================================

    def info(
        self,
    ) -> dict:

        return {

            "Initial Capital": self.initial_capital,

            "Cash": self.cash,

            "Equity": self.equity,

            "NAV": self.nav,

            "Positions": len(
                [
                    p
                    for p in self.positions.values()
                    if p.quantity != 0
                ]
            ),

            "Holdings Value": self.holdings_value,

            "Gross Exposure": self.gross_exposure,

            "Net Exposure": self.net_exposure,

            "Leverage": self.leverage,

        }

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ):

        return (

            f"{self.__class__.__name__}("
            f"equity={self.equity:,.2f}, "
            f"cash={self.cash:,.2f}, "
            f"positions={len(self.positions)})"

        )

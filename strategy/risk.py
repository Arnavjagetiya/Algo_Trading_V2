"""
strategy/risk.py
=========================================================
Risk Management Engine

Author : Arnav
Python : 3.12+
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# RISK RESULT
# ============================================================


@dataclass(slots=True)
class RiskResult:

    approved: bool

    reason: str

    quantity: float

    risk_amount: float


# ============================================================
# RISK MANAGER
# ============================================================


class RiskManager:

    """
    Production Risk Manager

    ✓ Position sizing
    ✓ Fixed % Risk
    ✓ Stop Loss
    ✓ Take Profit
    ✓ Exposure Limits
    ✓ Portfolio Limits
    """

    def __init__(

        self,

        account_size: float = 1_000_000,

        risk_per_trade: float = 0.01,

        max_portfolio_exposure: float = 1.00,

        max_position_weight: float = 0.10,

        stop_loss_pct: float = 0.02,

        take_profit_pct: float = 0.05,

    ):

        self.account_size = float(account_size)

        self.risk_per_trade = float(risk_per_trade)

        self.max_portfolio_exposure = float(
            max_portfolio_exposure
        )

        self.max_position_weight = float(
            max_position_weight
        )

        self.stop_loss_pct = float(stop_loss_pct)

        self.take_profit_pct = float(take_profit_pct)

    # ========================================================
    # ACCOUNT
    # ========================================================

    @property
    def maximum_risk(self):

        return (

            self.account_size

            * self.risk_per_trade

        )

    # ========================================================
    # POSITION SIZING
    # ========================================================

    def position_size(

        self,

        entry_price: float,

        stop_price: float,

    ) -> float:

        risk_per_share = abs(

            entry_price

            - stop_price

        )

        if risk_per_share <= 0:

            return 0

        quantity = (

            self.maximum_risk

            / risk_per_share

        )

        return np.floor(quantity)

    # --------------------------------------------------------

    def fixed_fractional(

        self,

        price: float,

    ) -> float:

        allocation = (

            self.account_size

            * self.max_position_weight

        )

        quantity = allocation / price

        return np.floor(quantity)

    # --------------------------------------------------------

    def fixed_quantity(

        self,

        quantity,

    ):

        return float(quantity)

    # ========================================================
    # STOP LOSS
    # ========================================================

    def stop_loss(

        self,

        entry_price,

    ):

        return (

            entry_price

            * (

                1

                - self.stop_loss_pct

            )

        )

    # --------------------------------------------------------

    def trailing_stop(

        self,

        highest_price,

    ):

        return (

            highest_price

            * (

                1

                - self.stop_loss_pct

            )

        )

    # ========================================================
    # TAKE PROFIT
    # ========================================================

    def take_profit(

        self,

        entry_price,

    ):

        return (

            entry_price

            * (

                1

                + self.take_profit_pct

            )

        )

    # ========================================================
    # RISK AMOUNT
    # ========================================================

    @staticmethod
    def trade_risk(

        entry,

        stop,

        quantity,

    ):

        return (

            abs(entry - stop)

            * quantity

        )

    # --------------------------------------------------------

    @staticmethod
    def reward(

        entry,

        target,

        quantity,

    ):

        return (

            abs(target - entry)

            * quantity

        )

    # --------------------------------------------------------

    @staticmethod
    def reward_risk_ratio(

        entry,

        stop,

        target,

    ):

        risk = abs(

            entry

            - stop

        )

        reward = abs(

            target

            - entry

        )

        if risk == 0:

            return np.inf

        return reward / risk

    # ========================================================
    # POSITION VALUE
    # ========================================================

    @staticmethod
    def position_value(

        quantity,

        price,

    ):

        return quantity * price

    # ========================================================
    # LIMITS
    # ========================================================

    def check_position_limit(

        self,

        quantity,

        price,

    ):

        value = self.position_value(

            quantity,

            price,

        )

        limit = (

            self.account_size

            * self.max_position_weight

        )

        return value <= limit

    # --------------------------------------------------------

    def check_portfolio_limit(

        self,

        current_exposure,

        new_position,

    ):

        limit = (

            self.account_size

            * self.max_portfolio_exposure

        )

        return (

            current_exposure

            + new_position

        ) <= limit

    # ========================================================
    # VALIDATION
    # ========================================================

    def approve_trade(

        self,

        entry_price,

        stop_price,

        current_exposure,

    ):

        quantity = self.position_size(

            entry_price,

            stop_price,

        )

        value = quantity * entry_price

        if quantity <= 0:

            return RiskResult(

                False,

                "Invalid quantity",

                0,

                0,

            )

        if not self.check_position_limit(

            quantity,

            entry_price,

        ):

            return RiskResult(

                False,

                "Position limit exceeded",

                quantity,

                0,

            )

        if not self.check_portfolio_limit(

            current_exposure,

            value,

        ):

            return RiskResult(

                False,

                "Portfolio exposure exceeded",

                quantity,

                0,

            )

        risk = self.trade_risk(

            entry_price,

            stop_price,

            quantity,

        )

        return RiskResult(

            True,

            "Approved",

            quantity,

            risk,

        )

    # ========================================================
    # VALUE AT RISK (VaR)
    # ========================================================

    @staticmethod
    def value_at_risk(
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:

        returns = returns.dropna()

        if returns.empty:
            return 0.0

        percentile = (1.0 - confidence) * 100

        return float(
            np.percentile(
                returns,
                percentile,
            )
        )

    # --------------------------------------------------------

    @staticmethod
    def conditional_value_at_risk(
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:

        returns = returns.dropna()

        if returns.empty:
            return 0.0

        var = RiskManager.value_at_risk(
            returns,
            confidence,
        )

        tail = returns[
            returns <= var
        ]

        if tail.empty:
            return var

        return float(
            tail.mean()
        )

    # ========================================================
    # VOLATILITY TARGETING
    # ========================================================

    @staticmethod
    def realized_volatility(
        returns: pd.Series,
        periods: int = 252,
    ) -> float:

        returns = returns.dropna()

        if returns.empty:
            return 0.0

        return float(
            returns.std()
            * np.sqrt(periods)
        )

    # --------------------------------------------------------

    @staticmethod
    def target_position_size(
        target_volatility: float,
        realized_volatility: float,
        current_quantity: float,
    ) -> float:

        if realized_volatility <= 0:
            return current_quantity

        return (

            current_quantity

            * target_volatility

            / realized_volatility

        )

    # ========================================================
    # CONCENTRATION
    # ========================================================

    @staticmethod
    def concentration_ratio(
        weights: pd.Series,
    ) -> float:

        weights = weights.dropna()

        if weights.empty:
            return 0.0

        return float(
            weights.max()
        )

    # --------------------------------------------------------

    @staticmethod
    def herfindahl_index(
        weights: pd.Series,
    ) -> float:

        weights = weights.fillna(0)

        return float(
            np.square(weights).sum()
        )

    # ========================================================
    # CORRELATION
    # ========================================================

    @staticmethod
    def correlation_matrix(
        returns: pd.DataFrame,
    ) -> pd.DataFrame:

        return returns.corr()

    # --------------------------------------------------------

    @staticmethod
    def average_correlation(
        returns: pd.DataFrame,
    ) -> float:

        corr = returns.corr()

        mask = np.triu(
            np.ones(corr.shape),
            k=1,
        ).astype(bool)

        values = corr.where(mask).stack()

        if values.empty:
            return 0.0

        return float(
            values.mean()
        )

    # --------------------------------------------------------

    @staticmethod
    def highly_correlated_pairs(
        returns: pd.DataFrame,
        threshold: float = 0.80,
    ) -> pd.DataFrame:

        corr = returns.corr()

        pairs = []

        columns = corr.columns

        for i in range(len(columns)):

            for j in range(i + 1, len(columns)):

                value = corr.iloc[i, j]

                if abs(value) >= threshold:

                    pairs.append(

                        {

                            "Asset 1": columns[i],

                            "Asset 2": columns[j],

                            "Correlation": value,

                        }

                    )

        return pd.DataFrame(pairs)

    # ========================================================
    # PORTFOLIO RISK
    # ========================================================

    @staticmethod
    def portfolio_volatility(
        covariance: pd.DataFrame,
        weights: pd.Series,
    ) -> float:

        weights = weights.reindex(
            covariance.columns
        ).fillna(0)

        w = weights.values

        sigma = covariance.values

        return float(
            np.sqrt(
                w.T @ sigma @ w
            )
        )

    # --------------------------------------------------------

    @staticmethod
    def marginal_risk_contribution(
        covariance: pd.DataFrame,
        weights: pd.Series,
    ) -> pd.Series:

        weights = weights.reindex(
            covariance.columns
        ).fillna(0)

        sigma = covariance.values

        w = weights.values

        portfolio_vol = np.sqrt(
            w.T @ sigma @ w
        )

        if portfolio_vol == 0:

            return pd.Series(
                0.0,
                index=weights.index,
            )

        mrc = (

            sigma @ w

        ) / portfolio_vol

        return pd.Series(
            mrc,
            index=weights.index,
        )

    # --------------------------------------------------------

    @staticmethod
    def risk_contribution(
        covariance: pd.DataFrame,
        weights: pd.Series,
    ) -> pd.Series:

        mrc = RiskManager.marginal_risk_contribution(
            covariance,
            weights,
        )

        return weights * mrc

    # ========================================================
    # DRAWDOWN
    # ========================================================

    @staticmethod
    def drawdown(
        equity_curve: pd.Series,
    ) -> pd.Series:

        running_max = equity_curve.cummax()

        return (
            equity_curve
            - running_max
        ) / running_max

    # --------------------------------------------------------

    @staticmethod
    def max_drawdown(
        equity_curve: pd.Series,
    ) -> float:

        dd = RiskManager.drawdown(
            equity_curve
        )

        if dd.empty:
            return 0.0

        return float(dd.min())

    # ========================================================
    # STRESS TESTING
    # ========================================================

    @staticmethod
    def stress_test(
        portfolio_value: float,
        shock: float = -0.10,
    ) -> float:

        return portfolio_value * (1 + shock)

    # --------------------------------------------------------

    @staticmethod
    def scenario_analysis(
        portfolio_value: float,
        scenarios: dict[str, float],
    ) -> pd.DataFrame:

        rows = []

        for name, shock in scenarios.items():

            rows.append(
                {
                    "Scenario": name,
                    "Shock": shock,
                    "Portfolio Value": portfolio_value * (1 + shock),
                    "PnL": portfolio_value * shock,
                }
            )

        return pd.DataFrame(rows)

    # ========================================================
    # RISK REPORT
    # ========================================================

    def report(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
    ) -> pd.DataFrame:

        var95 = self.value_at_risk(
            returns,
            0.95,
        )

        cvar95 = self.conditional_value_at_risk(
            returns,
            0.95,
        )

        volatility = self.realized_volatility(
            returns
        )

        drawdown = self.max_drawdown(
            equity_curve
        )

        return pd.DataFrame(
            {
                "Account Size": [
                    self.account_size
                ],
                "Risk / Trade": [
                    self.risk_per_trade
                ],
                "Maximum Risk": [
                    self.maximum_risk
                ],
                "Stop Loss %": [
                    self.stop_loss_pct
                ],
                "Take Profit %": [
                    self.take_profit_pct
                ],
                "95% VaR": [
                    var95
                ],
                "95% CVaR": [
                    cvar95
                ],
                "Annualized Volatility": [
                    volatility
                ],
                "Maximum Drawdown": [
                    drawdown
                ],
            }
        )

    # ========================================================
    # EXPORTS
    # ========================================================

    @staticmethod
    def export_report(
        report: pd.DataFrame,
        filepath: str,
    ):

        report.to_csv(
            filepath,
            index=False,
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    def info(
        self,
    ) -> dict:

        return {

            "Account Size": self.account_size,

            "Risk Per Trade": self.risk_per_trade,

            "Maximum Risk": self.maximum_risk,

            "Maximum Portfolio Exposure": self.max_portfolio_exposure,

            "Maximum Position Weight": self.max_position_weight,

            "Stop Loss %": self.stop_loss_pct,

            "Take Profit %": self.take_profit_pct,

        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        account_size: float | None = None,
    ):

        if account_size is not None:

            self.account_size = float(account_size)

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ):

        return (

            f"{self.__class__.__name__}("
            f"account_size={self.account_size:,.2f}, "
            f"risk_per_trade={self.risk_per_trade:.2%}, "
            f"max_position_weight={self.max_position_weight:.2%})"

        )

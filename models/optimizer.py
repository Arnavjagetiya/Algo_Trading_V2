"""
models/optimizer.py
=============================================================
Portfolio Optimization Engine

Author : Arnav
Python : 3.12+
=============================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scipy.optimize import minimize

from pypfopt import EfficientFrontier
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt import HRPOpt


# ============================================================
# RESULT
# ============================================================


@dataclass(slots=True)
class OptimizationResult:
    weights: pd.Series
    expected_return: float | None
    volatility: float | None
    sharpe: float | None
    method: str


# ============================================================
# OPTIMIZER
# ============================================================


class PortfolioOptimizer:

    def __init__(
        self,
        risk_free_rate: float = 0.06,
    ) -> None:

        self.risk_free_rate = risk_free_rate

    # =========================================================

    @staticmethod
    def _validate(price_data: pd.DataFrame):

        if not isinstance(price_data, pd.DataFrame):
            raise TypeError("Price data must be DataFrame.")

        if price_data.shape[1] < 2:
            raise ValueError("Need at least two assets.")

    # =========================================================

    @staticmethod
    def returns(price_data: pd.DataFrame) -> pd.DataFrame:

        return (
            price_data
            .pct_change()
            .dropna(how="all")
        )

    # =========================================================

    @staticmethod
    def equal_weight(price_data: pd.DataFrame):

        PortfolioOptimizer._validate(price_data)

        n = price_data.shape[1]

        weights = np.ones(n) / n

        return OptimizationResult(
            weights=pd.Series(
                weights,
                index=price_data.columns,
            ),
            expected_return=None,
            volatility=None,
            sharpe=None,
            method="Equal Weight",
        )

    # =========================================================

    def max_sharpe(
        self,
        price_data: pd.DataFrame,
    ):

        self._validate(price_data)

        mu = expected_returns.mean_historical_return(
            price_data
        )

        cov = risk_models.sample_cov(
            price_data
        )

        try:

            ef = EfficientFrontier(
                mu,
                cov,
            )

            ef.max_sharpe(
                risk_free_rate=self.risk_free_rate
            )

            cleaned = ef.clean_weights()

            performance = ef.portfolio_performance(
                risk_free_rate=self.risk_free_rate
            )

            weights = pd.Series(
                cleaned
            )

            return OptimizationResult(
                weights=weights,
                expected_return=performance[0],
                volatility=performance[1],
                sharpe=performance[2],
                method="Maximum Sharpe",
            )

        except Exception:

            return self.equal_weight(price_data)

    # =========================================================

    def minimum_variance(
        self,
        price_data: pd.DataFrame,
    ):

        self._validate(price_data)

        mu = expected_returns.mean_historical_return(
            price_data
        )

        cov = risk_models.sample_cov(
            price_data
        )

        try:

            ef = EfficientFrontier(
                mu,
                cov,
            )

            ef.min_volatility()

            cleaned = ef.clean_weights()

            performance = ef.portfolio_performance(
                risk_free_rate=self.risk_free_rate
            )

            return OptimizationResult(
                weights=pd.Series(cleaned),
                expected_return=performance[0],
                volatility=performance[1],
                sharpe=performance[2],
                method="Minimum Variance",
            )

        except Exception:

            return self.equal_weight(price_data)

    # =========================================================

    def optimize(
        self,
        price_data: pd.DataFrame,
    ):

        self._validate(price_data)

        try:
            return self.max_sharpe(price_data)
        except Exception:
            try:
                return self.minimum_variance(price_data)
            except Exception:
                try:
                    return self.hrp(price_data)
                except Exception:
                    return self.equal_weight(price_data)

    def hrp(
        self,
        price_data: pd.DataFrame,
    ):

        self._validate(price_data)

        returns = self.returns(price_data)

        try:

            hrp = HRPOpt(returns)

            weights = hrp.optimize()

            return OptimizationResult(
                weights=pd.Series(weights),
                expected_return=None,
                volatility=None,
                sharpe=None,
                method="Hierarchical Risk Parity",
            )

        except Exception:

            return self.equal_weight(price_data)

    # =========================================================

    def risk_parity(
        self,
        price_data: pd.DataFrame,
    ):

        self._validate(price_data)

        returns = self.returns(price_data)

        cov = returns.cov().values

        n = cov.shape[0]

        x0 = np.ones(n) / n

        bounds = tuple(
            (0, 1)
            for _ in range(n)
        )

        constraints = (
            {
                "type": "eq",
                "fun": lambda x: np.sum(x) - 1,
            },
        )

        def portfolio_risk(w):

            return np.sqrt(
                w.T @ cov @ w
            )

        def objective(w):

            sigma = portfolio_risk(w)

            mrc = cov @ w / sigma

            rc = w * mrc

            target = sigma / n

            return np.sum(
                (rc - target) ** 2
            )

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:

            return self.equal_weight(price_data)

        weights = pd.Series(
            result.x,
            index=price_data.columns,
        )

        return OptimizationResult(
            weights=weights,
            expected_return=None,
            volatility=portfolio_risk(result.x),
            sharpe=None,
            method="Risk Parity",
        )

    # =========================================================

    @staticmethod
    def clean_weights(
        weights: pd.Series,
        threshold: float = 1e-4,
    ) -> pd.Series:

        weights = weights.copy()

        weights[weights.abs() < threshold] = 0

        total = weights.sum()

        if total != 0:

            weights /= total

        return weights

    # =========================================================

    @staticmethod
    def top_holdings(
        weights: pd.Series,
        n: int = 10,
    ) -> pd.Series:

        return (
            weights
            .sort_values(
                ascending=False
            )
            .head(n)
        )

    # =========================================================

    @staticmethod
    def portfolio_returns(
        returns: pd.DataFrame,
        weights: pd.Series,
    ) -> pd.Series:

        weights = weights.reindex(
            returns.columns
        ).fillna(0)

        return (
            returns
            @ weights
        )

    # =========================================================

    @staticmethod
    def cumulative_returns(
        returns: pd.Series,
    ) -> pd.Series:

        return (
            1 + returns
        ).cumprod()

    # =========================================================

    @staticmethod
    def rebalance(
        price_data: pd.DataFrame,
        method: str = "max_sharpe",
        risk_free_rate: float = 0.06,
    ) -> OptimizationResult:

        optimizer = PortfolioOptimizer(
            risk_free_rate
        )

        method = method.lower()

        if method == "equal":

            return optimizer.equal_weight(
                price_data
            )

        if method == "max_sharpe":

            return optimizer.max_sharpe(
                price_data
            )

        if method == "min_variance":

            return optimizer.minimum_variance(
                price_data
            )

        if method == "risk_parity":

            return optimizer.risk_parity(
                price_data
            )

        if method == "hrp":

            return optimizer.hrp(
                price_data
            )

        raise ValueError(
            f"Unknown optimization method: {method}"
        )
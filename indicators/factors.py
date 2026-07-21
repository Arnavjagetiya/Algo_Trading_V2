"""
indicators/factors.py
=========================================================
Factor Engine

Professional Cross-Sectional Factor Library

Author : Arnav
Python : 3.12+
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# FACTOR RESULT
# ============================================================


@dataclass(slots=True)
class FactorResult:

    name: str

    values: pd.Series

    score: pd.Series


# ============================================================
# FACTOR ENGINE
# ============================================================


class FactorEngine:

    """
    Production Factor Library

    ✓ Momentum
    ✓ Volatility
    ✓ Value
    ✓ Size
    ✓ Liquidity
    ✓ Quality
    """

    def calculate_all(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        """Compute a compact factor feature set for a price frame."""

        data = dataframe.copy()

        if "Close" not in data.columns:

            return data

        close = data["Close"]

        data["Momentum_1"] = self.returns(close, 1)
        data["Momentum_3"] = self.returns(close, 3)
        data["Momentum_12"] = self.returns(close, 12)
        data["Momentum_126"] = self.momentum(close, 126)
        data["PriceStrength_252"] = self.price_strength(close, 252)
        data["Acceleration"] = self.acceleration(close, 63, 126)

        return data

    def __init__(

        self,

        trading_days: int = 252,

    ):

        self.trading_days = trading_days

    # ========================================================
    # RETURNS
    # ========================================================

    @staticmethod
    def returns(

        close: pd.Series,

        periods: int = 1,

    ) -> pd.Series:

        return close.pct_change(periods)

    # --------------------------------------------------------

    @staticmethod
    def log_returns(

        close: pd.Series,

    ) -> pd.Series:

        return np.log(

            close

            / close.shift(1)

        )

    # ========================================================
    # MOMENTUM
    # ========================================================

    @staticmethod
    def momentum(

        close: pd.Series,

        lookback: int = 126,

    ) -> pd.Series:

        return (

            close

            / close.shift(lookback)

        ) - 1

    # --------------------------------------------------------

    @staticmethod
    def price_strength(

        close: pd.Series,

        window: int = 252,

    ) -> pd.Series:

        highest = close.rolling(

            window

        ).max()

        return close / highest

    # --------------------------------------------------------

    @staticmethod
    def acceleration(

        close: pd.Series,

        short: int = 63,

        long: int = 126,

    ) -> pd.Series:

        return (

            FactorEngine.momentum(

                close,

                short,

            )

            -

            FactorEngine.momentum(

                close,

                long,

            )

        )

    # ========================================================
    # VOLATILITY
    # ========================================================

    def volatility(

        self,

        close: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        returns = self.returns(close)

        return (

            returns

            .rolling(window)

            .std()

            * np.sqrt(self.trading_days)

        )

    # --------------------------------------------------------

    def realized_volatility(

        self,

        returns: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        return (

            returns

            .rolling(window)

            .std()

            * np.sqrt(self.trading_days)

        )

    # ========================================================
    # SIZE
    # ========================================================

    @staticmethod
    def market_cap(

        price: pd.Series,

        shares_outstanding: float,

    ) -> pd.Series:

        return (

            price

            * shares_outstanding

        )

    # --------------------------------------------------------

    @staticmethod
    def log_market_cap(

        price: pd.Series,

        shares_outstanding: float,

    ) -> pd.Series:

        return np.log(

            FactorEngine.market_cap(

                price,

                shares_outstanding,

            )

        )

    # ========================================================
    # VALUE
    # ========================================================

    @staticmethod
    def book_to_market(

        book_value: float,

        market_cap: pd.Series,

    ) -> pd.Series:

        return (

            book_value

            / market_cap

        )

    # --------------------------------------------------------

    @staticmethod
    def earnings_yield(

        eps: float,

        price: pd.Series,

    ) -> pd.Series:

        return (

            eps

            / price

        )

    # --------------------------------------------------------

    @staticmethod
    def sales_to_price(

        sales_per_share: float,

        price: pd.Series,

    ) -> pd.Series:

        return (

            sales_per_share

            / price

        )

    # ========================================================
    # LIQUIDITY
    # ========================================================

    @staticmethod
    def average_volume(

        volume: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        return volume.rolling(

            window

        ).mean()

    # --------------------------------------------------------

    @staticmethod
    def dollar_volume(

        close: pd.Series,

        volume: pd.Series,

    ) -> pd.Series:

        return close * volume

    # --------------------------------------------------------

    @staticmethod
    def average_dollar_volume(

        close: pd.Series,

        volume: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        return (

            close

            * volume

        ).rolling(

            window

        ).mean()

    # ========================================================
    # QUALITY
    # ========================================================

    @staticmethod
    def roe(

        net_income: float,

        equity: float,

    ) -> float:

        if equity == 0:

            return np.nan

        return (

            net_income

            / equity

        )

    # --------------------------------------------------------

    @staticmethod
    def roa(

        net_income: float,

        assets: float,

    ) -> float:

        if assets == 0:

            return np.nan

        return (

            net_income

            / assets

        )

    # --------------------------------------------------------

    @staticmethod
    def gross_margin(

        revenue: float,

        cogs: float,

    ) -> float:

        if revenue == 0:

            return np.nan

        return (

            revenue

            - cogs

        ) / revenue

    # --------------------------------------------------------

    @staticmethod
    def asset_turnover(

        revenue: float,

        assets: float,

    ) -> float:

        if assets == 0:

            return np.nan

        return (

            revenue

            / assets

        )

    # ========================================================
    # BETA
    # ========================================================

    @staticmethod
    def beta(
        asset_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:

        data = pd.concat(
            [asset_returns, benchmark_returns],
            axis=1,
        ).dropna()

        if len(data) < 2:
            return np.nan

        covariance = np.cov(
            data.iloc[:, 0],
            data.iloc[:, 1],
        )[0, 1]

        variance = np.var(
            data.iloc[:, 1]
        )

        if variance == 0:
            return np.nan

        return covariance / variance

    # --------------------------------------------------------

    @staticmethod
    def alpha(
        asset_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.0,
    ) -> float:

        beta = FactorEngine.beta(
            asset_returns,
            benchmark_returns,
        )

        asset_return = asset_returns.mean()

        benchmark_return = benchmark_returns.mean()

        return (

            asset_return

            - (

                risk_free_rate

                + beta
                * (
                    benchmark_return
                    - risk_free_rate
                )

            )

        )

    # ========================================================
    # SHARPE RATIO
    # ========================================================

    def sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
    ) -> float:

        excess = (
            returns
            - risk_free_rate / self.trading_days
        )

        volatility = excess.std()

        if volatility == 0:
            return np.nan

        return (
            excess.mean()
            / volatility
        ) * np.sqrt(self.trading_days)

    # ========================================================
    # SORTINO RATIO
    # ========================================================

    def sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
    ) -> float:

        excess = (
            returns
            - risk_free_rate / self.trading_days
        )

        downside = excess[
            excess < 0
        ]

        downside_std = downside.std()

        if downside_std == 0:
            return np.nan

        return (
            excess.mean()
            / downside_std
        ) * np.sqrt(self.trading_days)

    # ========================================================
    # CALMAR RATIO
    # ========================================================

    def calmar_ratio(
        self,
        returns: pd.Series,
    ) -> float:

        cumulative = (
            1 + returns
        ).cumprod()

        running_max = cumulative.cummax()

        drawdown = (
            cumulative
            - running_max
        ) / running_max

        max_dd = abs(
            drawdown.min()
        )

        if max_dd == 0:
            return np.nan

        annual_return = (
            cumulative.iloc[-1]
            ** (
                self.trading_days
                / len(cumulative)
            )
        ) - 1

        return annual_return / max_dd

    # ========================================================
    # DOWNSIDE RISK
    # ========================================================

    def downside_risk(
        self,
        returns: pd.Series,
    ) -> float:

        downside = returns[
            returns < 0
        ]

        if downside.empty:
            return 0.0

        return (
            downside.std()
            * np.sqrt(self.trading_days)
        )

    # ========================================================
    # ROLLING FACTORS
    # ========================================================

    def rolling_beta(
        self,
        asset_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 126,
    ) -> pd.Series:

        result = pd.Series(
            index=asset_returns.index,
            dtype=float,
        )

        for i in range(
            window,
            len(asset_returns),
        ):

            result.iloc[i] = self.beta(
                asset_returns.iloc[
                    i - window:i
                ],
                benchmark_returns.iloc[
                    i - window:i
                ],
            )

        return result

    # --------------------------------------------------------

    def rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 126,
    ) -> pd.Series:

        return returns.rolling(
            window
        ).apply(

            lambda x: self.sharpe_ratio(
                pd.Series(x)
            ),

            raw=False,

        )

    # --------------------------------------------------------

    def rolling_volatility(
        self,
        returns: pd.Series,
        window: int = 20,
    ) -> pd.Series:

        return (
            returns
            .rolling(window)
            .std()
            * np.sqrt(self.trading_days)
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def z_score(
        series: pd.Series,
    ) -> pd.Series:

        std = series.std()

        if std == 0:
            return pd.Series(
                0.0,
                index=series.index,
            )

        return (
            series
            - series.mean()
        ) / std

    # --------------------------------------------------------

    @staticmethod
    def min_max_scale(
        series: pd.Series,
    ) -> pd.Series:

        minimum = series.min()

        maximum = series.max()

        if minimum == maximum:

            return pd.Series(
                0.0,
                index=series.index,
            )

        return (
            series
            - minimum
        ) / (
            maximum
            - minimum
        )

    # ========================================================
    # COMPOSITE FACTOR SCORE
    # ========================================================

    @staticmethod
    def composite_score(
        factors: pd.DataFrame,
        weights: dict[str, float] | None = None,
    ) -> pd.Series:

        if factors.empty:
            return pd.Series(dtype=float)

        normalized = factors.apply(
            FactorEngine.z_score
        )

        if weights is None:

            return normalized.mean(axis=1)

        score = pd.Series(
            0.0,
            index=normalized.index,
        )

        for factor in normalized.columns:

            score += (
                normalized[factor]
                * weights.get(factor, 0.0)
            )

        return score

    # ========================================================
    # CROSS-SECTIONAL NORMALIZATION
    # ========================================================

    @staticmethod
    def cross_sectional_rank(
        factors: pd.DataFrame,
        ascending: bool = False,
    ) -> pd.DataFrame:

        return factors.rank(
            axis=0,
            pct=True,
            ascending=ascending,
        )

    # --------------------------------------------------------

    @staticmethod
    def percentile_score(
        series: pd.Series,
    ) -> pd.Series:

        return series.rank(
            pct=True
        )

    # ========================================================
    # FACTOR RANKING
    # ========================================================

    @staticmethod
    def rank_assets(
        scores: pd.Series,
        ascending: bool = False,
    ) -> pd.Series:

        return scores.sort_values(
            ascending=ascending
        )

    # --------------------------------------------------------

    @staticmethod
    def top_assets(
        scores: pd.Series,
        n: int = 10,
    ) -> pd.Series:

        return scores.sort_values(
            ascending=False
        ).head(n)

    # --------------------------------------------------------

    @staticmethod
    def bottom_assets(
        scores: pd.Series,
        n: int = 10,
    ) -> pd.Series:

        return scores.sort_values(
            ascending=True
        ).head(n)

    # ========================================================
    # PORTFOLIO FACTOR EXPOSURE
    # ========================================================

    @staticmethod
    def portfolio_exposure(
        factor_values: pd.DataFrame,
        weights: pd.Series,
    ) -> pd.Series:

        weights = weights.reindex(
            factor_values.index
        ).fillna(0)

        return factor_values.mul(
            weights,
            axis=0,
        ).sum()

    # --------------------------------------------------------

    @staticmethod
    def weighted_factor_score(
        factor_scores: pd.DataFrame,
        portfolio_weights: pd.Series,
    ) -> float:

        portfolio_weights = portfolio_weights.reindex(
            factor_scores.index
        ).fillna(0)

        composite = factor_scores.mean(
            axis=1
        )

        return float(
            (composite * portfolio_weights).sum()
        )

    # ========================================================
    # FACTOR REPORT
    # ========================================================

    @staticmethod
    def report(
        factor_scores: pd.DataFrame,
    ) -> pd.DataFrame:

        if factor_scores.empty:

            return pd.DataFrame()

        report = pd.DataFrame({

            "Mean": factor_scores.mean(),

            "Median": factor_scores.median(),

            "Std": factor_scores.std(),

            "Minimum": factor_scores.min(),

            "Maximum": factor_scores.max(),

            "Missing": factor_scores.isna().sum(),

        })

        return report

    # ========================================================
    # EXPORTS
    # ========================================================

    @staticmethod
    def export_scores(
        scores: pd.Series,
        filepath: str,
    ):

        scores.to_csv(
            filepath,
            header=["Score"],
        )

    # --------------------------------------------------------

    @staticmethod
    def export_report(
        report: pd.DataFrame,
        filepath: str,
    ):

        report.to_csv(
            filepath,
            index=True,
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    def info(
        self,
    ) -> dict:

        return {

            "Trading Days": self.trading_days,

            "Available Factors": [

                "Momentum",

                "Volatility",

                "Value",

                "Size",

                "Liquidity",

                "Quality",

                "Beta",

                "Alpha",

                "Sharpe",

                "Sortino",

                "Calmar",

            ],

        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ):

        self.trading_days = 252

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ):

        return (
            f"{self.__class__.__name__}("
            f"trading_days={self.trading_days})"
        )
        

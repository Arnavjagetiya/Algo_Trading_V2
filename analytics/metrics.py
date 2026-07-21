"""
analytics/metrics.py
=========================================================
Portfolio Performance Metrics Engine

Author : Arnav
Python : 3.12+
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# RESULT
# ============================================================


@dataclass(slots=True)
class MetricResult:

    name: str

    value: float


# ============================================================
# PERFORMANCE METRICS
# ============================================================


class PerformanceMetrics:

    """
    Professional Portfolio Analytics

    ✓ Returns
    ✓ CAGR
    ✓ Annual Return
    ✓ Volatility
    ✓ Downside Volatility
    """

    def __init__(

        self,

        trading_days: int = 252,

        risk_free_rate: float = 0.06,

    ):

        self.trading_days = trading_days

        self.risk_free_rate = risk_free_rate

    # ========================================================
    # RETURNS
    # ========================================================

    @staticmethod
    def returns(

        prices: pd.Series,

    ) -> pd.Series:

        return prices.pct_change().dropna()

    # --------------------------------------------------------

    @staticmethod
    def log_returns(

        prices: pd.Series,

    ) -> pd.Series:

        return np.log(

            prices

            / prices.shift(1)

        ).dropna()

    # --------------------------------------------------------

    @staticmethod
    def cumulative_returns(

        returns: pd.Series,

    ) -> pd.Series:

        return (

            1 + returns

        ).cumprod() - 1

    # --------------------------------------------------------

    @staticmethod
    def total_return(

        returns: pd.Series,

    ) -> float:

        if returns.empty:

            return np.nan

        return float(

            (1 + returns).prod()

            - 1

        )

    # ========================================================
    # CAGR
    # ========================================================

    def cagr(

        self,

        returns: pd.Series,

    ) -> float:

        if returns.empty:

            return np.nan

        years = (

            len(returns)

            / self.trading_days

        )

        if years <= 0:

            return np.nan

        cumulative = (

            1 + returns

        ).prod()

        return float(

            cumulative ** (1 / years)

            - 1

        )

    # ========================================================
    # ANNUALIZED RETURN
    # ========================================================

    def annual_return(

        self,

        returns: pd.Series,

    ) -> float:

        if returns.empty:

            return np.nan

        return float(

            returns.mean()

            * self.trading_days

        )

    # ========================================================
    # VOLATILITY
    # ========================================================

    def volatility(

        self,

        returns: pd.Series,

    ) -> float:

        if returns.empty:

            return np.nan

        return float(

            returns.std()

            * np.sqrt(

                self.trading_days

            )

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

            * np.sqrt(

                self.trading_days

            )

        )

    # ========================================================
    # DOWNSIDE VOLATILITY
    # ========================================================

    def downside_volatility(

        self,

        returns: pd.Series,

    ) -> float:

        downside = returns[

            returns < 0

        ]

        if downside.empty:

            return 0.0

        return float(

            downside.std()

            * np.sqrt(

                self.trading_days

            )

        )

    # --------------------------------------------------------

    def rolling_downside_volatility(

        self,

        returns: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        def downside_std(x):

            values = x[x < 0]

            if len(values) == 0:

                return 0

            return values.std()

        return (

            returns

            .rolling(window)

            .apply(

                downside_std,

                raw=False,

            )

            * np.sqrt(

                self.trading_days

            )

        )

    # ========================================================
    # ANNUALIZATION HELPERS
    # ========================================================

    def annualize_return(

        self,

        daily_return: float,

    ) -> float:

        return (

            1 + daily_return

        ) ** self.trading_days - 1

    # --------------------------------------------------------

    def deannualize_return(

        self,

        annual_return: float,

    ) -> float:

        return (

            1 + annual_return

        ) ** (

            1 / self.trading_days

        ) - 1

    # ========================================================
    # SHARPE RATIO
    # ========================================================

    def sharpe_ratio(
        self,
        returns: pd.Series,
    ) -> float:

        if returns.empty:

            return np.nan

        excess = (

            returns

            - self.risk_free_rate

            / self.trading_days

        )

        volatility = excess.std()

        if volatility == 0:

            return np.nan

        return float(

            excess.mean()

            / volatility

            * np.sqrt(self.trading_days)

        )

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

    # ========================================================
    # SORTINO RATIO
    # ========================================================

    def sortino_ratio(
        self,
        returns: pd.Series,
    ) -> float:

        if returns.empty:

            return np.nan

        excess = (

            returns

            - self.risk_free_rate

            / self.trading_days

        )

        downside = excess[

            excess < 0

        ]

        downside_std = downside.std()

        if downside_std == 0:

            return np.nan

        return float(

            excess.mean()

            / downside_std

            * np.sqrt(self.trading_days)

        )

    # ========================================================
    # CALMAR RATIO
    # ========================================================

    def calmar_ratio(
        self,
        returns: pd.Series,
    ) -> float:

        max_dd = abs(

            self.maximum_drawdown(

                returns

            )

        )

        if max_dd == 0:

            return np.nan

        return (

            self.cagr(

                returns

            )

            / max_dd

        )

    # ========================================================
    # INFORMATION RATIO
    # ========================================================

    def information_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:

        active = (

            portfolio_returns

            - benchmark_returns

        ).dropna()

        tracking_error = active.std()

        if tracking_error == 0:

            return np.nan

        return float(

            active.mean()

            / tracking_error

            * np.sqrt(self.trading_days)

        )

    # ========================================================
    # BETA
    # ========================================================

    @staticmethod
    def beta(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:

        data = pd.concat(

            [

                portfolio_returns,

                benchmark_returns,

            ],

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

        return float(

            covariance

            / variance

        )

    # ========================================================
    # ALPHA
    # ========================================================

    def alpha(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:

        beta = self.beta(

            portfolio_returns,

            benchmark_returns,

        )

        rp = (

            portfolio_returns.mean()

            * self.trading_days

        )

        rb = (

            benchmark_returns.mean()

            * self.trading_days

        )

        return float(

            rp

            - (

                self.risk_free_rate

                + beta

                * (

                    rb

                    - self.risk_free_rate

                )

            )

        )

    # ========================================================
    # TREYNOR RATIO
    # ========================================================

    def treynor_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:

        beta = self.beta(

            portfolio_returns,

            benchmark_returns,

        )

        if beta == 0:

            return np.nan

        annual_return = self.annual_return(

            portfolio_returns

        )

        return (

            annual_return

            - self.risk_free_rate

        ) / beta

    # ========================================================
    # TRACKING ERROR
    # ========================================================

    def tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:

        active = (

            portfolio_returns

            - benchmark_returns

        ).dropna()

        return float(

            active.std()

            * np.sqrt(

                self.trading_days

            )

        )

    # ========================================================
    # DRAWDOWN
    # ========================================================

    @staticmethod
    def drawdown(
        returns: pd.Series,
    ) -> pd.Series:

        equity = (
            1 + returns
        ).cumprod()

        peak = equity.cummax()

        return (
            equity
            - peak
        ) / peak

    # --------------------------------------------------------

    def maximum_drawdown(
        self,
        returns: pd.Series,
    ) -> float:

        if returns.empty:

            return np.nan

        return float(
            self.drawdown(
                returns
            ).min()
        )

    # --------------------------------------------------------

    def drawdown_duration(
        self,
        returns: pd.Series,
    ) -> int:

        dd = self.drawdown(
            returns
        )

        underwater = dd < 0

        duration = 0

        maximum = 0

        for value in underwater:

            if value:

                duration += 1

                maximum = max(
                    maximum,
                    duration,
                )

            else:

                duration = 0

        return maximum

    # ========================================================
    # VALUE AT RISK
    # ========================================================

    @staticmethod
    def value_at_risk(
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:

        if returns.empty:

            return np.nan

        percentile = (
            1 - confidence
        ) * 100

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

        var = PerformanceMetrics.value_at_risk(
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
    # ULCER INDEX
    # ========================================================

    def ulcer_index(
        self,
        returns: pd.Series,
    ) -> float:

        dd = self.drawdown(
            returns
        )

        return float(
            np.sqrt(
                np.mean(
                    np.square(dd)
                )
            )
        )

    # ========================================================
    # OMEGA RATIO
    # ========================================================

    @staticmethod
    def omega_ratio(
        returns: pd.Series,
        threshold: float = 0.0,
    ) -> float:

        gains = (
            returns[
                returns > threshold
            ]
            - threshold
        ).sum()

        losses = (
            threshold
            - returns[
                returns < threshold
            ]
        ).sum()

        if losses == 0:

            return np.nan

        return float(
            gains / losses
        )

    # ========================================================
    # WIN RATE
    # ========================================================

    @staticmethod
    def win_rate(
        returns: pd.Series,
    ) -> float:

        if returns.empty:

            return np.nan

        wins = (
            returns > 0
        ).sum()

        return float(
            wins
            / len(returns)
        )

    # ========================================================
    # PROFIT FACTOR
    # ========================================================

    @staticmethod
    def profit_factor(
        returns: pd.Series,
    ) -> float:

        gross_profit = returns[
            returns > 0
        ].sum()

        gross_loss = abs(
            returns[
                returns < 0
            ].sum()
        )

        if gross_loss == 0:

            return np.nan

        return float(
            gross_profit
            / gross_loss
        )

    # ========================================================
    # EXPECTANCY
    # ========================================================

    @staticmethod
    def expectancy(
        returns: pd.Series,
    ) -> float:

        if returns.empty:

            return np.nan

        wins = returns[
            returns > 0
        ]

        losses = returns[
            returns < 0
        ]

        win_rate = (
            len(wins)
            / len(returns)
        )

        loss_rate = (
            len(losses)
            / len(returns)
        )

        average_win = (
            wins.mean()
            if len(wins)
            else 0.0
        )

        average_loss = abs(
            losses.mean()
        ) if len(losses) else 0.0

        return float(

            win_rate
            * average_win

            -

            loss_rate
            * average_loss

        )

    # ========================================================
    # RECOVERY FACTOR
    # ========================================================

    def recovery_factor(
        self,
        returns: pd.Series,
    ) -> float:

        drawdown = abs(
            self.maximum_drawdown(
                returns
            )
        )

        if drawdown == 0:

            return np.nan

        total = self.total_return(
            returns
        )

        return float(
            total
            / drawdown
        )

    # ========================================================
    # PORTFOLIO SUMMARY
    # ========================================================

    def summary(
        self,
        returns: pd.Series,
    ) -> pd.DataFrame:

        if returns.empty:

            return pd.DataFrame()

        summary = {

            "Total Return": self.total_return(returns),

            "Annual Return": self.annual_return(returns),

            "CAGR": self.cagr(returns),

            "Volatility": self.volatility(returns),

            "Downside Volatility": self.downside_volatility(returns),

            "Sharpe Ratio": self.sharpe_ratio(returns),

            "Sortino Ratio": self.sortino_ratio(returns),

            "Calmar Ratio": self.calmar_ratio(returns),

            "Maximum Drawdown": self.maximum_drawdown(returns),

            "Drawdown Duration": self.drawdown_duration(returns),

            "VaR (95%)": self.value_at_risk(returns),

            "CVaR (95%)": self.conditional_value_at_risk(returns),

            "Ulcer Index": self.ulcer_index(returns),

            "Omega Ratio": self.omega_ratio(returns),

            "Win Rate": self.win_rate(returns),

            "Profit Factor": self.profit_factor(returns),

            "Expectancy": self.expectancy(returns),

            "Recovery Factor": self.recovery_factor(returns),

        }

        return pd.DataFrame(
            [summary]
        )

    def calculate(
        self,
        equity_curve: pd.Series | None = None,
        returns: pd.Series | None = None,
        trades: list | None = None,
    ) -> dict:

        if returns is None:
            if equity_curve is None:
                return {}
            returns = self.returns(equity_curve)

        if returns.empty:
            return {
                "total_return": np.nan,
                "annual_return": np.nan,
                "cagr": np.nan,
                "volatility": np.nan,
                "max_drawdown": np.nan,
            }

        summary = self.summary(returns)

        if summary.empty:
            return {}

        result = {
            "total_return": float(summary.iloc[0]["Total Return"]),
            "annual_return": float(summary.iloc[0]["Annual Return"]),
            "cagr": float(summary.iloc[0]["CAGR"]),
            "volatility": float(summary.iloc[0]["Volatility"]),
            "max_drawdown": float(summary.iloc[0]["Maximum Drawdown"]),
        }

        if trades is not None:
            if hasattr(trades, "empty"):
                result["trades"] = len(trades) if not trades.empty else 0
            else:
                result["trades"] = len(trades)

        return result

    # ========================================================
    # BENCHMARK COMPARISON
    # ========================================================

    def benchmark_summary(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            [{

                "Alpha": self.alpha(
                    portfolio_returns,
                    benchmark_returns,
                ),

                "Beta": self.beta(
                    portfolio_returns,
                    benchmark_returns,
                ),

                "Information Ratio": self.information_ratio(
                    portfolio_returns,
                    benchmark_returns,
                ),

                "Treynor Ratio": self.treynor_ratio(
                    portfolio_returns,
                    benchmark_returns,
                ),

                "Tracking Error": self.tracking_error(
                    portfolio_returns,
                    benchmark_returns,
                ),

            }]
        )

    # ========================================================
    # ROLLING METRICS
    # ========================================================

    def rolling_cagr(
        self,
        returns: pd.Series,
        window: int = 252,
    ) -> pd.Series:

        return returns.rolling(
            window
        ).apply(

            lambda x: self.cagr(
                pd.Series(x)
            ),

            raw=False,

        )

    # --------------------------------------------------------

    def rolling_max_drawdown(
        self,
        returns: pd.Series,
        window: int = 252,
    ) -> pd.Series:

        return returns.rolling(
            window
        ).apply(

            lambda x: self.maximum_drawdown(
                pd.Series(x)
            ),

            raw=False,

        )

    # ========================================================
    # EXPORTS
    # ========================================================

    @staticmethod
    def export(
        dataframe: pd.DataFrame,
        filepath: str,
    ):

        dataframe.to_csv(
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

            "Risk Free Rate": self.risk_free_rate,

            "Available Metrics": [

                "Total Return",

                "Annual Return",

                "CAGR",

                "Volatility",

                "Downside Volatility",

                "Sharpe Ratio",

                "Sortino Ratio",

                "Calmar Ratio",

                "Maximum Drawdown",

                "Drawdown Duration",

                "Value at Risk",

                "Conditional VaR",

                "Ulcer Index",

                "Omega Ratio",

                "Win Rate",

                "Profit Factor",

                "Expectancy",

                "Recovery Factor",

                "Alpha",

                "Beta",

                "Information Ratio",

                "Treynor Ratio",

                "Tracking Error",

            ],

        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ):

        self.trading_days = 252

        self.risk_free_rate = 0.06

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ):

        return (

            f"{self.__class__.__name__}("

            f"trading_days={self.trading_days}, "

            f"risk_free_rate={self.risk_free_rate:.2%})"

        )
"""
indicators/volatility.py
=========================================================
Professional Volatility Indicator Library

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
class VolatilityResult:

    name: str

    values: pd.Series


# ============================================================
# VOLATILITY ENGINE
# ============================================================


class VolatilityIndicators:

    """
    Professional Volatility Library

    ✓ Historical Volatility
    ✓ Rolling Volatility
    ✓ EWMA Volatility
    ✓ Parkinson
    ✓ Garman-Klass
    ✓ Rogers-Satchell
    """

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
    # HISTORICAL VOLATILITY
    # ========================================================

    def historical_volatility(

        self,

        close: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        returns = self.log_returns(close)

        return (

            returns

            .rolling(window)

            .std()

            * np.sqrt(self.trading_days)

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
    # EWMA VOLATILITY
    # ========================================================

    def ewma_volatility(

        self,

        close: pd.Series,

        span: int = 30,

    ) -> pd.Series:

        returns = self.log_returns(close)

        variance = (

            returns.pow(2)

            .ewm(

                span=span,

                adjust=False,

            )

            .mean()

        )

        return (

            np.sqrt(variance)

            * np.sqrt(self.trading_days)

        )

    # ========================================================
    # PARKINSON VOLATILITY
    # ========================================================

    def parkinson_volatility(

        self,

        high: pd.Series,

        low: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        rs = np.log(

            high

            / low

        ) ** 2

        estimator = (

            rs

            .rolling(window)

            .mean()

        )

        return np.sqrt(

            estimator

            / (4 * np.log(2))

        ) * np.sqrt(

            self.trading_days

        )

    # ========================================================
    # GARMAN-KLASS
    # ========================================================

    def garman_klass(

        self,

        open_: pd.Series,

        high: pd.Series,

        low: pd.Series,

        close: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        log_hl = np.log(

            high

            / low

        )

        log_co = np.log(

            close

            / open_

        )

        estimator = (

            0.5

            * log_hl ** 2

            -

            (

                2

                * np.log(2)

                - 1

            )

            * log_co ** 2

        )

        return np.sqrt(

            estimator

            .rolling(window)

            .mean()

        ) * np.sqrt(

            self.trading_days

        )

    # ========================================================
    # ROGERS-SATCHELL
    # ========================================================

    def rogers_satchell(

        self,

        open_: pd.Series,

        high: pd.Series,

        low: pd.Series,

        close: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        term1 = np.log(

            high

            / close

        ) * np.log(

            high

            / open_

        )

        term2 = np.log(

            low

            / close

        ) * np.log(

            low

            / open_

        )

        estimator = (

            term1

            + term2

        )

        return np.sqrt(

            estimator

            .rolling(window)

            .mean()

        ) * np.sqrt(

            self.trading_days

        )

    # ========================================================
    # NORMALIZED VOLATILITY
    # ========================================================

    def normalized_volatility(

        self,

        close: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        hv = self.historical_volatility(

            close,

            window,

        )

        return (

            hv

            / close

        ) * 100

    # ========================================================
    # YANG-ZHANG VOLATILITY
    # ========================================================

    def yang_zhang(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
    ) -> pd.Series:

        log_oc = np.log(
            open_ / close.shift(1)
        )

        log_co = np.log(
            close / open_
        )

        rs = (
            np.log(high / close)
            * np.log(high / open_)
            +
            np.log(low / close)
            * np.log(low / open_)
        )

        overnight = (
            log_oc.rolling(window).var()
        )

        open_close = (
            log_co.rolling(window).var()
        )

        rs_mean = rs.rolling(
            window
        ).mean()

        k = (
            0.34
            /
            (
                1.34
                + (window + 1)
                / (window - 1)
            )
        )

        variance = (
            overnight
            +
            k * open_close
            +
            (1 - k) * rs_mean
        )

        return (
            np.sqrt(variance)
            * np.sqrt(self.trading_days)
        )

    # ========================================================
    # REALIZED VOLATILITY
    # ========================================================

    def realized_volatility(
        self,
        returns: pd.Series,
        window: int = 20,
    ) -> pd.Series:

        realized = (
            returns.pow(2)
            .rolling(window)
            .sum()
        )

        return np.sqrt(
            realized
        ) * np.sqrt(
            self.trading_days
        )

    # ========================================================
    # ATR VOLATILITY
    # ========================================================

    @staticmethod
    def atr_volatility(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:

        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(
            window
        ).mean()

        return (
            atr
            / close
        ) * 100

    # ========================================================
    # VOLATILITY REGIME
    # ========================================================

    @staticmethod
    def volatility_regime(
        volatility: pd.Series,
        low_quantile: float = 0.33,
        high_quantile: float = 0.66,
    ) -> pd.Series:

        low = volatility.quantile(
            low_quantile
        )

        high = volatility.quantile(
            high_quantile
        )

        regime = pd.Series(
            "Medium",
            index=volatility.index,
            dtype="object",
        )

        regime[
            volatility <= low
        ] = "Low"

        regime[
            volatility >= high
        ] = "High"

        return regime

    # ========================================================
    # VOLATILITY PERCENTILE
    # ========================================================

    @staticmethod
    def volatility_percentile(
        volatility: pd.Series,
        window: int = 252,
    ) -> pd.Series:

        return volatility.rolling(
            window
        ).rank(
            pct=True
        )

    # ========================================================
    # VOLATILITY OF VOLATILITY
    # ========================================================

    def volatility_of_volatility(
        self,
        volatility: pd.Series,
        window: int = 20,
    ) -> pd.Series:

        return (
            volatility
            .rolling(window)
            .std()
            * np.sqrt(self.trading_days)
        )

    # ========================================================
    # EXPANDING VOLATILITY
    # ========================================================

    def expanding_volatility(
        self,
        returns: pd.Series,
    ) -> pd.Series:

        return (
            returns.expanding()
            .std()
            * np.sqrt(self.trading_days)
        )

    # ========================================================
    # VOLATILITY RATIO
    # ========================================================

    @staticmethod
    def volatility_ratio(
        short_volatility: pd.Series,
        long_volatility: pd.Series,
    ) -> pd.Series:

        return (
            short_volatility
            / long_volatility
        )

    # ========================================================
    # VOLATILITY Z-SCORE
    # ========================================================

    @staticmethod
    def volatility_zscore(
        volatility: pd.Series,
        window: int = 252,
    ) -> pd.Series:

        mean = volatility.rolling(
            window
        ).mean()

        std = volatility.rolling(
            window
        ).std()

        return (
            volatility
            - mean
        ) / std
        
    # ========================================================
    # VOLATILITY FORECAST
    # ========================================================

    def forecast_volatility(
        self,
        volatility: pd.Series,
        horizon: int = 5,
    ) -> pd.Series:

        if volatility.empty:

            return pd.Series(
                dtype=float
            )

        forecast = pd.Series(
            volatility.iloc[-1],
            index=range(1, horizon + 1),
            dtype=float,
        )

        return forecast

    # ========================================================
    # ANNUALIZATION
    # ========================================================

    def annualize(
        self,
        volatility: pd.Series,
        periods: int | None = None,
    ) -> pd.Series:

        if periods is None:

            periods = self.trading_days

        return volatility * np.sqrt(periods)

    # --------------------------------------------------------

    def deannualize(
        self,
        volatility: pd.Series,
        periods: int | None = None,
    ) -> pd.Series:

        if periods is None:

            periods = self.trading_days

        return volatility / np.sqrt(periods)

    # ========================================================
    # ROLLING STATISTICS
    # ========================================================

    @staticmethod
    def rolling_mean(
        volatility: pd.Series,
        window: int = 20,
    ) -> pd.Series:

        return volatility.rolling(window).mean()

    # --------------------------------------------------------

    @staticmethod
    def rolling_std(
        volatility: pd.Series,
        window: int = 20,
    ) -> pd.Series:

        return volatility.rolling(window).std()

    # --------------------------------------------------------

    @staticmethod
    def rolling_max(
        volatility: pd.Series,
        window: int = 20,
    ) -> pd.Series:

        return volatility.rolling(window).max()

    # --------------------------------------------------------

    @staticmethod
    def rolling_min(
        volatility: pd.Series,
        window: int = 20,
    ) -> pd.Series:

        return volatility.rolling(window).min()

    # ========================================================
    # RISK METRICS
    # ========================================================

    @staticmethod
    def coefficient_of_variation(
        volatility: pd.Series,
    ) -> float:

        mean = volatility.mean()

        if mean == 0:

            return np.nan

        return float(

            volatility.std()

            / mean

        )

    # --------------------------------------------------------

    @staticmethod
    def volatility_range(
        volatility: pd.Series,
    ) -> float:

        return float(

            volatility.max()

            - volatility.min()

        )

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
        volatility: pd.Series,
    ) -> pd.DataFrame:

        return pd.DataFrame({

            "Mean": [volatility.mean()],

            "Median": [volatility.median()],

            "Std": [volatility.std()],

            "Minimum": [volatility.min()],

            "Maximum": [volatility.max()],

            "Latest": [volatility.iloc[-1]],

            "Coefficient of Variation": [

                self.coefficient_of_variation(
                    volatility
                )

            ],

            "Range": [

                self.volatility_range(
                    volatility
                )

            ],

        })

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

            "Available Models": [

                "Historical Volatility",

                "Rolling Volatility",

                "EWMA",

                "Parkinson",

                "Garman-Klass",

                "Rogers-Satchell",

                "Yang-Zhang",

                "Realized Volatility",

                "ATR Volatility",

                "Volatility Regime",

                "Volatility Percentile",

                "Volatility of Volatility",

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
        
        
"""
strategy/signal.py
=========================================================
Signal Generation Engine

Author : Arnav
Python : 3.12+
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


# ============================================================
# SIGNAL TYPES
# ============================================================


class SignalType(Enum):

    BUY = 1

    SELL = -1

    HOLD = 0


# ============================================================
# SIGNAL
# ============================================================


@dataclass(slots=True)
class Signal:

    ticker: str

    timestamp: pd.Timestamp

    signal: SignalType

    price: float

    confidence: float

    strategy: str


# ============================================================
# SIGNAL ENGINE
# ============================================================


class SignalEngine:

    """
    Production Signal Generator

    ✓ Moving Average Crossover
    ✓ Momentum
    ✓ RSI
    ✓ Breakout
    ✓ Mean Reversion
    """

    def __init__(

        self,

        fast_ma: int = 20,

        slow_ma: int = 50,

        breakout_period: int = 20,

        momentum_period: int = 20,

        rsi_period: int = 14,

    ):

        self.fast_ma = fast_ma

        self.slow_ma = slow_ma

        self.breakout_period = breakout_period

        self.momentum_period = momentum_period

        self.rsi_period = rsi_period

    # ========================================================
    # MOVING AVERAGES
    # ========================================================

    def moving_average(

        self,

        series: pd.Series,

        window: int,

    ):

        return series.rolling(

            window

        ).mean()

    # --------------------------------------------------------

    def ema(

        self,

        series: pd.Series,

        window: int,

    ):

        return series.ewm(

            span=window,

            adjust=False,

        ).mean()

    # ========================================================
    # MA CROSSOVER
    # ========================================================

    def moving_average_crossover(

        self,

        close: pd.Series,

    ) -> pd.Series:

        fast = self.moving_average(

            close,

            self.fast_ma,

        )

        slow = self.moving_average(

            close,

            self.slow_ma,

        )

        signal = pd.Series(

            SignalType.HOLD.value,

            index=close.index,

        )

        signal[fast > slow] = SignalType.BUY.value

        signal[fast < slow] = SignalType.SELL.value

        return signal

    # ========================================================
    # MOMENTUM
    # ========================================================

    def momentum(

        self,

        close: pd.Series,

    ) -> pd.Series:

        return close.pct_change(

            self.momentum_period

        )

    # --------------------------------------------------------

    def momentum_signal(

        self,

        close: pd.Series,

    ):

        mom = self.momentum(close)

        signal = pd.Series(

            SignalType.HOLD.value,

            index=close.index,

        )

        signal[mom > 0] = SignalType.BUY.value

        signal[mom < 0] = SignalType.SELL.value

        return signal

    # ========================================================
    # BREAKOUT
    # ========================================================

    def breakout_signal(

        self,

        high: pd.Series,

        low: pd.Series,

        close: pd.Series,

    ):

        highest = high.shift(1).rolling(

            self.breakout_period

        ).max()

        lowest = low.shift(1).rolling(

            self.breakout_period

        ).min()

        signal = pd.Series(

            SignalType.HOLD.value,

            index=close.index,

        )

        signal[close > highest] = SignalType.BUY.value

        signal[close < lowest] = SignalType.SELL.value

        return signal

    # ========================================================
    # MEAN REVERSION
    # ========================================================

    def mean_reversion_signal(

        self,

        close: pd.Series,

        window: int = 20,

        std: float = 2.0,

    ):

        mean = close.rolling(window).mean()

        sigma = close.rolling(window).std()

        upper = mean + sigma * std

        lower = mean - sigma * std

        signal = pd.Series(

            SignalType.HOLD.value,

            index=close.index,

        )

        signal[close < lower] = SignalType.BUY.value

        signal[close > upper] = SignalType.SELL.value

        return signal

    # ========================================================
    # RSI
    # ========================================================

    def rsi(

        self,

        close: pd.Series,

    ):

        delta = close.diff()

        gain = delta.clip(

            lower=0

        )

        loss = -delta.clip(

            upper=0

        )

        avg_gain = gain.rolling(

            self.rsi_period

        ).mean()

        avg_loss = loss.rolling(

            self.rsi_period

        ).mean()

        rs = avg_gain / avg_loss

        return 100 - (

            100

            / (1 + rs)

        )

    # --------------------------------------------------------

    def rsi_signal(

        self,

        close: pd.Series,

        oversold: int = 30,

        overbought: int = 70,

    ):

        indicator = self.rsi(close)

        signal = pd.Series(

            SignalType.HOLD.value,

            index=close.index,

        )

        signal[indicator < oversold] = SignalType.BUY.value

        signal[indicator > overbought] = SignalType.SELL.value

        return signal

    # ========================================================
    # MACD
    # ========================================================

    def macd(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ):

        fast_ema = self.ema(
            close,
            fast,
        )

        slow_ema = self.ema(
            close,
            slow,
        )

        macd = fast_ema - slow_ema

        signal_line = macd.ewm(
            span=signal,
            adjust=False,
        ).mean()

        histogram = (
            macd
            - signal_line
        )

        return macd, signal_line, histogram

    # --------------------------------------------------------

    def macd_signal(
        self,
        close: pd.Series,
    ) -> pd.Series:

        macd, signal, _ = self.macd(
            close
        )

        result = pd.Series(
            SignalType.HOLD.value,
            index=close.index,
        )

        result[macd > signal] = SignalType.BUY.value

        result[macd < signal] = SignalType.SELL.value

        return result

    # ========================================================
    # BOLLINGER BANDS
    # ========================================================

    def bollinger_bands(
        self,
        close: pd.Series,
        window: int = 20,
        std: float = 2.0,
    ):

        middle = close.rolling(
            window
        ).mean()

        sigma = close.rolling(
            window
        ).std()

        upper = middle + std * sigma

        lower = middle - std * sigma

        return upper, middle, lower

    # --------------------------------------------------------

    def bollinger_signal(
        self,
        close: pd.Series,
    ) -> pd.Series:

        upper, _, lower = self.bollinger_bands(
            close
        )

        signal = pd.Series(
            SignalType.HOLD.value,
            index=close.index,
        )

        signal[
            close < lower
        ] = SignalType.BUY.value

        signal[
            close > upper
        ] = SignalType.SELL.value

        return signal

    # ========================================================
    # VOLATILITY BREAKOUT
    # ========================================================

    def volatility_breakout(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        multiplier: float = 1.5,
    ) -> pd.Series:

        atr = (
            high
            - low
        ).rolling(window).mean()

        breakout = atr * multiplier

        signal = pd.Series(
            SignalType.HOLD.value,
            index=close.index,
        )

        signal[
            close.diff() > breakout
        ] = SignalType.BUY.value

        signal[
            close.diff() < -breakout
        ] = SignalType.SELL.value

        return signal

    # ========================================================
    # VOLUME FILTER
    # ========================================================

    def volume_filter(
        self,
        volume: pd.Series,
        window: int = 20,
        multiplier: float = 1.20,
    ) -> pd.Series:

        average = volume.rolling(
            window
        ).mean()

        return volume > (
            average
            * multiplier
        )

    # --------------------------------------------------------

    def apply_volume_filter(
        self,
        signal: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:

        valid = self.volume_filter(
            volume
        )

        result = signal.copy()

        result[
            ~valid
        ] = SignalType.HOLD.value

        return result

    # ========================================================
    # TREND FILTER
    # ========================================================

    def trend_filter(
        self,
        close: pd.Series,
        window: int = 200,
    ) -> pd.Series:

        trend = close.rolling(
            window
        ).mean()

        return close > trend

    # --------------------------------------------------------

    def apply_trend_filter(
        self,
        signal: pd.Series,
        close: pd.Series,
    ) -> pd.Series:

        trend = self.trend_filter(
            close
        )

        result = signal.copy()

        result[
            (~trend)
            &
            (signal == SignalType.BUY.value)
        ] = SignalType.HOLD.value

        result[
            (trend)
            &
            (signal == SignalType.SELL.value)
        ] = SignalType.HOLD.value

        return result

    # ========================================================
    # SIGNAL COMBINATION
    # ========================================================

    def combine_signals(
        self,
        *signals: pd.Series,
    ) -> pd.Series:

        if len(signals) == 0:

            return pd.Series(
                dtype=int
            )

        matrix = pd.concat(
            signals,
            axis=1,
        )

        score = matrix.sum(
            axis=1
        )

        result = pd.Series(
            SignalType.HOLD.value,
            index=matrix.index,
        )

        result[
            score > 0
        ] = SignalType.BUY.value

        result[
            score < 0
        ] = SignalType.SELL.value

        return result

    # ========================================================
    # CONFIDENCE SCORE
    # ========================================================

    @staticmethod
    def confidence_score(
        *signals: pd.Series,
    ) -> pd.Series:

        matrix = pd.concat(
            signals,
            axis=1,
        )

        return (
            matrix.abs().sum(axis=1)
            / len(signals)
        )

    # --------------------------------------------------------

    def confirmed_signal(
        self,
        signal: pd.Series,
        confidence: pd.Series,
        threshold: float = 0.70,
    ) -> pd.Series:

        result = signal.copy()

        result[
            confidence < threshold
        ] = SignalType.HOLD.value

        return result

    # ========================================================
    # SINGLE ASSET SIGNAL
    # ========================================================

    def generate_signal(
        self,
        ticker: str,
        data: pd.DataFrame,
    ) -> Signal:

        close = data["Close"]

        signal = self.combine_signals(
            self.moving_average_crossover(close),
            self.momentum_signal(close),
            self.macd_signal(close),
            self.rsi_signal(close),
        )

        confidence = self.confidence_score(
            self.moving_average_crossover(close),
            self.momentum_signal(close),
            self.macd_signal(close),
            self.rsi_signal(close),
        )

        latest_signal = SignalType(
            int(signal.iloc[-1])
        )

        return Signal(
            ticker=ticker,
            timestamp=data.index[-1],
            signal=latest_signal,
            price=float(close.iloc[-1]),
            confidence=float(confidence.iloc[-1]),
            strategy="Combined",
        )

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:

        """Return a compact signal DataFrame for a single asset."""

        if "Close" not in data.columns:

            return pd.DataFrame(index=data.index)

        close = data["Close"]

        signal = self.combine_signals(
            self.moving_average_crossover(close),
            self.momentum_signal(close),
            self.macd_signal(close),
            self.rsi_signal(close),
        )

        result = pd.DataFrame(index=data.index)
        result["Signal"] = signal
        result["SignalValue"] = signal
        result["Confidence"] = self.confidence_score(
            self.moving_average_crossover(close),
            self.momentum_signal(close),
            self.macd_signal(close),
            self.rsi_signal(close),
        )

        return result

    # ========================================================
    # MULTI-ASSET SIGNALS
    # ========================================================

    def generate_portfolio_signals(
        self,
        market_data: dict[str, pd.DataFrame],
    ) -> list[Signal]:

        signals: list[Signal] = []

        for ticker, df in market_data.items():

            try:

                signals.append(
                    self.generate_signal(
                        ticker,
                        df,
                    )
                )

            except Exception:

                continue

        return signals

    # ========================================================
    # RANKING
    # ========================================================

    @staticmethod
    def rank_signals(
        signals: list[Signal],
    ) -> list[Signal]:

        return sorted(

            signals,

            key=lambda x: x.confidence,

            reverse=True,

        )

    # --------------------------------------------------------

    @staticmethod
    def filter_signals(
        signals: list[Signal],
        signal_type: SignalType,
    ) -> list[Signal]:

        return [

            signal

            for signal in signals

            if signal.signal == signal_type

        ]

    # --------------------------------------------------------

    @staticmethod
    def top_signals(
        signals: list[Signal],
        n: int = 10,
    ) -> list[Signal]:

        ranked = SignalEngine.rank_signals(
            signals
        )

        return ranked[:n]

    # ========================================================
    # DATAFRAME EXPORT
    # ========================================================

    @staticmethod
    def to_dataframe(
        signals: list[Signal],
    ) -> pd.DataFrame:

        if len(signals) == 0:

            return pd.DataFrame()

        rows = []

        for signal in signals:

            rows.append(

                {

                    "Ticker": signal.ticker,

                    "Timestamp": signal.timestamp,

                    "Signal": signal.signal.name,

                    "Price": signal.price,

                    "Confidence": signal.confidence,

                    "Strategy": signal.strategy,

                }

            )

        return pd.DataFrame(rows)

    # --------------------------------------------------------

    @staticmethod
    def export_csv(
        signals: list[Signal],
        filepath: str,
    ):

        SignalEngine.to_dataframe(
            signals
        ).to_csv(
            filepath,
            index=False,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    @staticmethod
    def summary(
        signals: list[Signal],
    ) -> pd.DataFrame:

        df = SignalEngine.to_dataframe(
            signals
        )

        if df.empty:

            return pd.DataFrame()

        summary = {

            "Total Signals": len(df),

            "Buy": (
                df["Signal"] == "BUY"
            ).sum(),

            "Sell": (
                df["Signal"] == "SELL"
            ).sum(),

            "Hold": (
                df["Signal"] == "HOLD"
            ).sum(),

            "Average Confidence": df[
                "Confidence"
            ].mean(),

            "Maximum Confidence": df[
                "Confidence"
            ].max(),

        }

        return pd.DataFrame(
            [summary]
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    def info(
        self,
    ) -> dict:

        return {

            "Fast MA": self.fast_ma,

            "Slow MA": self.slow_ma,

            "Momentum Period": self.momentum_period,

            "Breakout Period": self.breakout_period,

            "RSI Period": self.rsi_period,

        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ):

        self.fast_ma = 20

        self.slow_ma = 50

        self.breakout_period = 20

        self.momentum_period = 20

        self.rsi_period = 14

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ):

        return (

            f"{self.__class__.__name__}("
            f"fast_ma={self.fast_ma}, "
            f"slow_ma={self.slow_ma}, "
            f"momentum_period={self.momentum_period}, "
            f"rsi_period={self.rsi_period})"

        )

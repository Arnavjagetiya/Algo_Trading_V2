"""
indicators/technicals.py
=========================================================
Professional Technical Indicator Library

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
class IndicatorResult:

    name: str

    values: pd.Series


# ============================================================
# TECHNICAL INDICATORS
# ============================================================


class TechnicalIndicators:

    """
    Professional Technical Indicator Library

    ✓ SMA
    ✓ EMA
    ✓ WMA
    ✓ HMA
    ✓ VWAP
    ✓ ATR
    ✓ True Range
    """

    def calculate_all(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        """Compute a standard technical-indicator feature set for a price frame."""

        data = dataframe.copy()

        if "Close" not in data.columns:

            return data

        close = data["Close"]

        data["SMA_20"] = self.sma(close, 20)
        data["EMA_20"] = self.ema(close, 20)
        data["EMA_50"] = self.ema(close, 50)
        data["EMA_200"] = self.ema(close, 200)
        data["RSI_14"] = self.rsi(close, 14)
        data["ATR_14"] = self.atr(data["High"], data["Low"], close, 14)
        upper, middle, lower = self.bollinger_bands(close, 20, 2.0)
        data["BB_MIDDLE_20"] = middle
        data["BB_UPPER_20"] = upper
        data["BB_LOWER_20"] = lower
        macd_line, signal_line, _ = self.macd(close, 12, 26, 9)
        data["MACD"] = macd_line
        data["MACD_SIGNAL"] = signal_line
        data["VWAP"] = self.vwap(data["High"], data["Low"], close, data["Volume"])

        return data

    # ========================================================
    # SIMPLE MOVING AVERAGE
    # ========================================================

    @staticmethod
    def sma(

        close: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        return close.rolling(

            window,

            min_periods=window,

        ).mean()

    # ========================================================
    # EXPONENTIAL MOVING AVERAGE
    # ========================================================

    @staticmethod
    def ema(

        close: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        return close.ewm(

            span=window,

            adjust=False,

        ).mean()

    # ========================================================
    # WEIGHTED MOVING AVERAGE
    # ========================================================

    @staticmethod
    def wma(

        close: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        weights = np.arange(

            1,

            window + 1,

        )

        return close.rolling(

            window,

        ).apply(

            lambda x:

            np.dot(x, weights)

            / weights.sum(),

            raw=True,

        )

    # ========================================================
    # HULL MOVING AVERAGE
    # ========================================================

    @staticmethod
    def hma(

        close: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        half = max(

            1,

            window // 2,

        )

        sqrt = max(

            1,

            int(np.sqrt(window)),

        )

        wma_half = TechnicalIndicators.wma(

            close,

            half,

        )

        wma_full = TechnicalIndicators.wma(

            close,

            window,

        )

        raw = (

            2 * wma_half

            - wma_full

        )

        return TechnicalIndicators.wma(

            raw,

            sqrt,

        )

    # ========================================================
    # VOLUME WEIGHTED AVERAGE PRICE
    # ========================================================

    @staticmethod
    def vwap(

        high: pd.Series,

        low: pd.Series,

        close: pd.Series,

        volume: pd.Series,

    ) -> pd.Series:

        typical_price = (

            high

            + low

            + close

        ) / 3

        cumulative_value = (

            typical_price

            * volume

        ).cumsum()

        cumulative_volume = (

            volume.cumsum()

        )

        return (

            cumulative_value

            / cumulative_volume

        )

    # ========================================================
    # TRUE RANGE
    # ========================================================

    @staticmethod
    def true_range(

        high: pd.Series,

        low: pd.Series,

        close: pd.Series,

    ) -> pd.Series:

        previous_close = close.shift(1)

        tr = pd.concat(

            [

                high - low,

                (high - previous_close).abs(),

                (low - previous_close).abs(),

            ],

            axis=1,

        ).max(axis=1)

        return tr

    # ========================================================
    # ATR
    # ========================================================

    @staticmethod
    def atr(

        high: pd.Series,

        low: pd.Series,

        close: pd.Series,

        window: int = 14,

    ) -> pd.Series:

        tr = TechnicalIndicators.true_range(

            high,

            low,

            close,

        )

        return tr.rolling(

            window,

        ).mean()

    # ========================================================
    # NORMALIZED ATR
    # ========================================================

    @staticmethod
    def normalized_atr(

        high: pd.Series,

        low: pd.Series,

        close: pd.Series,

        window: int = 14,

    ) -> pd.Series:

        atr = TechnicalIndicators.atr(

            high,

            low,

            close,

            window,

        )

        return (

            atr

            / close

        ) * 100

    # ========================================================
    # MOVING AVERAGE RIBBON
    # ========================================================

    @staticmethod
    def moving_average_ribbon(

        close: pd.Series,

        windows=(5, 10, 20, 50, 100, 200),

    ) -> pd.DataFrame:

        ribbon = pd.DataFrame(

            index=close.index

        )

        for window in windows:

            ribbon[f"SMA_{window}"] = (

                TechnicalIndicators.sma(

                    close,

                    window,

                )

            )

        return ribbon

    # ========================================================
    # PRICE DISTANCE
    # ========================================================

    @staticmethod
    def price_distance_from_ma(

        close: pd.Series,

        window: int = 20,

    ) -> pd.Series:

        average = TechnicalIndicators.sma(

            close,

            window,

        )

        return (

            close

            - average

        ) / average

    # ========================================================
    # RSI
    # ========================================================

    @staticmethod
    def rsi(
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:

        delta = close.diff()

        gain = delta.clip(lower=0)

        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / window,
            adjust=False,
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / window,
            adjust=False,
        ).mean()

        rs = avg_gain / avg_loss

        return 100 - (
            100 / (1 + rs)
        )

    # ========================================================
    # MACD
    # ========================================================

    @staticmethod
    def macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:

        ema_fast = TechnicalIndicators.ema(
            close,
            fast,
        )

        ema_slow = TechnicalIndicators.ema(
            close,
            slow,
        )

        macd_line = (
            ema_fast
            - ema_slow
        )

        signal_line = macd_line.ewm(
            span=signal,
            adjust=False,
        ).mean()

        histogram = (
            macd_line
            - signal_line
        )

        return (
            macd_line,
            signal_line,
            histogram,
        )

    # ========================================================
    # STOCHASTIC OSCILLATOR
    # ========================================================

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
        smooth: int = 3,
    ) -> tuple[pd.Series, pd.Series]:

        lowest = low.rolling(
            window
        ).min()

        highest = high.rolling(
            window
        ).max()

        k = (
            (
                close
                - lowest
            )
            /
            (
                highest
                - lowest
            )
        ) * 100

        d = k.rolling(
            smooth
        ).mean()

        return k, d

    # ========================================================
    # WILLIAMS %R
    # ========================================================

    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:

        highest = high.rolling(
            window
        ).max()

        lowest = low.rolling(
            window
        ).min()

        return (
            (
                highest
                - close
            )
            /
            (
                highest
                - lowest
            )
        ) * -100

    # ========================================================
    # CCI
    # ========================================================

    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
    ) -> pd.Series:

        tp = (
            high
            + low
            + close
        ) / 3

        sma = tp.rolling(
            window
        ).mean()

        mad = tp.rolling(
            window
        ).apply(
            lambda x: np.mean(
                np.abs(
                    x - np.mean(x)
                )
            ),
            raw=True,
        )

        return (
            tp
            - sma
        ) / (
            0.015
            * mad
        )

    # ========================================================
    # RATE OF CHANGE
    # ========================================================

    @staticmethod
    def roc(
        close: pd.Series,
        window: int = 12,
    ) -> pd.Series:

        return (
            (
                close
                / close.shift(window)
            )
            - 1
        ) * 100

    # ========================================================
    # MOMENTUM
    # ========================================================

    @staticmethod
    def momentum(
        close: pd.Series,
        window: int = 10,
    ) -> pd.Series:

        return (
            close
            - close.shift(window)
        )

    # ========================================================
    # PPO
    # ========================================================

    @staticmethod
    def ppo(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:

        fast_ema = TechnicalIndicators.ema(
            close,
            fast,
        )

        slow_ema = TechnicalIndicators.ema(
            close,
            slow,
        )

        ppo_line = (
            (
                fast_ema
                - slow_ema
            )
            / slow_ema
        ) * 100

        signal_line = ppo_line.ewm(
            span=signal,
            adjust=False,
        ).mean()

        histogram = (
            ppo_line
            - signal_line
        )

        return (
            ppo_line,
            signal_line,
            histogram,
        )

    # ========================================================
    # TRIX
    # ========================================================

    @staticmethod
    def trix(
        close: pd.Series,
        window: int = 15,
    ) -> pd.Series:

        ema1 = TechnicalIndicators.ema(
            close,
            window,
        )

        ema2 = TechnicalIndicators.ema(
            ema1,
            window,
        )

        ema3 = TechnicalIndicators.ema(
            ema2,
            window,
        )

        return ema3.pct_change() * 100

    # ========================================================
    # BOLLINGER BANDS
    # ========================================================

    @staticmethod
    def bollinger_bands(
        close: pd.Series,
        window: int = 20,
        std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:

        middle = TechnicalIndicators.sma(
            close,
            window,
        )

        deviation = close.rolling(
            window
        ).std()

        upper = (
            middle
            + std * deviation
        )

        lower = (
            middle
            - std * deviation
        )

        return (
            upper,
            middle,
            lower,
        )

    # ========================================================
    # KELTNER CHANNEL
    # ========================================================

    @staticmethod
    def keltner_channel(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        ema_window: int = 20,
        atr_window: int = 10,
        multiplier: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:

        middle = TechnicalIndicators.ema(
            close,
            ema_window,
        )

        atr = TechnicalIndicators.atr(
            high,
            low,
            close,
            atr_window,
        )

        upper = middle + multiplier * atr

        lower = middle - multiplier * atr

        return (
            upper,
            middle,
            lower,
        )

    # ========================================================
    # DONCHIAN CHANNEL
    # ========================================================

    @staticmethod
    def donchian_channel(
        high: pd.Series,
        low: pd.Series,
        window: int = 20,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:

        upper = high.rolling(
            window
        ).max()

        lower = low.rolling(
            window
        ).min()

        middle = (
            upper + lower
        ) / 2

        return (
            upper,
            middle,
            lower,
        )

    # ========================================================
    # DIRECTIONAL MOVEMENT
    # ========================================================

    @staticmethod
    def directional_movement(
        high: pd.Series,
        low: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:

        up_move = high.diff()

        down_move = -low.diff()

        plus_dm = pd.Series(
            np.where(
                (up_move > down_move) &
                (up_move > 0),
                up_move,
                0.0,
            ),
            index=high.index,
        )

        minus_dm = pd.Series(
            np.where(
                (down_move > up_move) &
                (down_move > 0),
                down_move,
                0.0,
            ),
            index=high.index,
        )

        return (
            plus_dm,
            minus_dm,
        )

    # ========================================================
    # ADX
    # ========================================================

    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:

        atr = TechnicalIndicators.atr(
            high,
            low,
            close,
            window,
        )

        plus_dm, minus_dm = (
            TechnicalIndicators.directional_movement(
                high,
                low,
            )
        )

        plus_di = (
            100
            * plus_dm.rolling(window).mean()
            / atr
        )

        minus_di = (
            100
            * minus_dm.rolling(window).mean()
            / atr
        )

        dx = (
            (
                plus_di
                - minus_di
            ).abs()
            /
            (
                plus_di
                + minus_di
            )
        ) * 100

        adx = dx.rolling(
            window
        ).mean()

        return (
            adx,
            plus_di,
            minus_di,
        )

    # ========================================================
    # ON BALANCE VOLUME
    # ========================================================

    @staticmethod
    def obv(
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:

        direction = np.sign(
            close.diff()
        ).fillna(0)

        return (
            direction * volume
        ).cumsum()

    # ========================================================
    # MONEY FLOW INDEX
    # ========================================================

    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 14,
    ) -> pd.Series:

        typical = (
            high
            + low
            + close
        ) / 3

        money_flow = (
            typical
            * volume
        )

        positive = money_flow.where(
            typical.diff() > 0,
            0.0,
        )

        negative = money_flow.where(
            typical.diff() < 0,
            0.0,
        )

        positive_sum = positive.rolling(
            window
        ).sum()

        negative_sum = negative.rolling(
            window
        ).sum()

        ratio = (
            positive_sum
            / negative_sum
        )

        return 100 - (
            100
            / (
                1 + ratio
            )
        )

    # ========================================================
    # CHAIKIN MONEY FLOW
    # ========================================================

    @staticmethod
    def cmf(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 20,
    ) -> pd.Series:

        multiplier = (
            (
                (close - low)
                -
                (high - close)
            )
            /
            (high - low)
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0)

        flow_volume = (
            multiplier
            * volume
        )

        return (
            flow_volume.rolling(window).sum()
            /
            volume.rolling(window).sum()
        )

    # ========================================================
    # ICHIMOKU CLOUD
    # ========================================================

    @staticmethod
    def ichimoku(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> tuple[
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
    ]:

        conversion = (
            high.rolling(9).max()
            + low.rolling(9).min()
        ) / 2

        base = (
            high.rolling(26).max()
            + low.rolling(26).min()
        ) / 2

        span_a = (
            (conversion + base) / 2
        ).shift(26)

        span_b = (
            (
                high.rolling(52).max()
                + low.rolling(52).min()
            ) / 2
        ).shift(26)

        lagging = close.shift(-26)

        return (
            conversion,
            base,
            span_a,
            span_b,
            lagging,
        )

    # ========================================================
    # SUPERTREND
    # ========================================================

    @staticmethod
    def supertrend(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> pd.Series:

        atr = TechnicalIndicators.atr(
            high,
            low,
            close,
            period,
        )

        hl2 = (high + low) / 2

        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr

        trend = pd.Series(
            index=close.index,
            dtype=float,
        )

        trend.iloc[0] = lower.iloc[0]

        for i in range(1, len(close)):

            if close.iloc[i] >= trend.iloc[i - 1]:
                trend.iloc[i] = max(
                    lower.iloc[i],
                    trend.iloc[i - 1],
                )
            else:
                trend.iloc[i] = min(
                    upper.iloc[i],
                    trend.iloc[i - 1],
                )

        return trend

    # ========================================================
    # PARABOLIC SAR
    # ========================================================

    @staticmethod
    def parabolic_sar(
        high: pd.Series,
        low: pd.Series,
        step: float = 0.02,
        maximum: float = 0.20,
    ) -> pd.Series:

        sar = pd.Series(
            index=high.index,
            dtype=float,
        )

        sar.iloc[0] = low.iloc[0]

        ep = high.iloc[0]

        af = step

        uptrend = True

        for i in range(1, len(high)):

            previous = sar.iloc[i - 1]

            sar.iloc[i] = previous + af * (
                ep - previous
            )

            if uptrend:

                if high.iloc[i] > ep:

                    ep = high.iloc[i]

                    af = min(
                        af + step,
                        maximum,
                    )

                if low.iloc[i] < sar.iloc[i]:

                    uptrend = False

                    sar.iloc[i] = ep

                    ep = low.iloc[i]

                    af = step

            else:

                if low.iloc[i] < ep:

                    ep = low.iloc[i]

                    af = min(
                        af + step,
                        maximum,
                    )

                if high.iloc[i] > sar.iloc[i]:

                    uptrend = True

                    sar.iloc[i] = ep

                    ep = high.iloc[i]

                    af = step

        return sar

    # ========================================================
    # PIVOT POINTS
    # ========================================================

    @staticmethod
    def pivot_points(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:

        pivot = (
            high + low + close
        ) / 3

        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high

        r2 = pivot + (high - low)
        s2 = pivot - (high - low)

        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)

        return pd.DataFrame(
            {
                "Pivot": pivot,
                "R1": r1,
                "R2": r2,
                "R3": r3,
                "S1": s1,
                "S2": s2,
                "S3": s3,
            }
        )

    # ========================================================
    # SUPPORT / RESISTANCE
    # ========================================================

    @staticmethod
    def rolling_support(
        low: pd.Series,
        window: int = 20,
    ) -> pd.Series:

        return low.rolling(window).min()

    # --------------------------------------------------------

    @staticmethod
    def rolling_resistance(
        high: pd.Series,
        window: int = 20,
    ) -> pd.Series:

        return high.rolling(window).max()

    # ========================================================
    # SUMMARY
    # ========================================================

    @staticmethod
    def summary(
        close: pd.Series,
    ) -> pd.DataFrame:

        macd, signal, hist = TechnicalIndicators.macd(
            close
        )

        upper, middle, lower = (
            TechnicalIndicators.bollinger_bands(
                close
            )
        )

        return pd.DataFrame(
            {
                "Close": close,
                "SMA20": TechnicalIndicators.sma(close, 20),
                "EMA20": TechnicalIndicators.ema(close, 20),
                "RSI": TechnicalIndicators.rsi(close),
                "MACD": macd,
                "MACD Signal": signal,
                "MACD Histogram": hist,
                "BB Upper": upper,
                "BB Middle": middle,
                "BB Lower": lower,
            }
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

    @staticmethod
    def info() -> dict:

        return {

            "Indicators": [

                "SMA",
                "EMA",
                "WMA",
                "HMA",
                "VWAP",
                "ATR",
                "RSI",
                "MACD",
                "Stochastic",
                "Williams %R",
                "CCI",
                "ROC",
                "Momentum",
                "PPO",
                "TRIX",
                "Bollinger Bands",
                "Keltner Channel",
                "Donchian Channel",
                "ADX",
                "OBV",
                "MFI",
                "CMF",
                "Ichimoku",
                "SuperTrend",
                "Parabolic SAR",
                "Pivot Points",

            ],

        }

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ):

        return (
            f"{self.__class__.__name__}("
            "Professional Technical Indicator Library)"
        )

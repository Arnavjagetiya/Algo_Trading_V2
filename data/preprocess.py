"""
data/preprocess.py
=========================================================
Market Data Preprocessing Engine

Author : Arnav
Python : 3.12+
=========================================================
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# PREPROCESSOR
# ============================================================


class Preprocessor:

    """
    Handles

    ✓ Data validation
    ✓ Missing values
    ✓ Duplicate removal
    ✓ Sorting
    ✓ Resampling
    ✓ Returns
    ✓ Log returns
    ✓ Dollar volume
    ✓ OHLC validation
    """

    REQUIRED_COLUMNS = [

        "Open",
        "High",
        "Low",
        "Close",
        "Volume",

    ]

    # =======================================================
    # VALIDATION
    # =======================================================

    @staticmethod
    def validate(data: pd.DataFrame):

        if not isinstance(data, pd.DataFrame):

            raise TypeError(
                "Input must be DataFrame."
            )

        if data.empty:

            raise ValueError(
                "Empty dataframe."
            )

        missing = [

            column

            for column in Preprocessor.REQUIRED_COLUMNS

            if column not in data.columns

        ]

        if missing:

            raise ValueError(

                f"Missing columns : {missing}"

            )

        return True

    # -------------------------------------------------------

    @staticmethod
    def validate_many(

        datasets: dict[str, pd.DataFrame],

    ):

        for ticker, df in datasets.items():

            try:

                Preprocessor.validate(df)

            except Exception as e:

                raise ValueError(

                    f"{ticker} : {e}"

                )

        return True

    # =======================================================
    # CLEANING
    # =======================================================

    @staticmethod
    def sort(

        dataframe: pd.DataFrame,

    ) -> pd.DataFrame:

        df = dataframe.copy()

        df.sort_index(

            inplace=True

        )

        return df

    # -------------------------------------------------------

    @staticmethod
    def remove_duplicates(

        dataframe,

    ):

        df = dataframe.copy()

        df = df.loc[

            ~df.index.duplicated(

                keep="last"

            )

        ]

        return df

    # -------------------------------------------------------

    @staticmethod
    def remove_nan(

        dataframe,

    ):

        df = dataframe.copy()

        df.dropna(

            how="all",

            inplace=True,

        )

        return df

    # -------------------------------------------------------

    @staticmethod
    def replace_inf(

        dataframe,

    ):

        df = dataframe.copy()

        df.replace(

            [

                np.inf,

                -np.inf,

            ],

            np.nan,

            inplace=True,

        )

        return df

    # -------------------------------------------------------

    @staticmethod
    def forward_fill(

        dataframe,

    ):

        return dataframe.ffill()

    # -------------------------------------------------------

    @staticmethod
    def backward_fill(

        dataframe,

    ):

        return dataframe.bfill()

    # -------------------------------------------------------

    @staticmethod
    def clean(

        dataframe,

    ):

        df = dataframe.copy()

        df = Preprocessor.sort(df)

        df = Preprocessor.remove_duplicates(df)

        df = Preprocessor.replace_inf(df)

        df = Preprocessor.forward_fill(df)

        df = Preprocessor.backward_fill(df)

        df = Preprocessor.remove_nan(df)

        return df

    # =======================================================
    # PRICE FEATURES
    # =======================================================

    @staticmethod
    def typical_price(

        dataframe,

    ):

        return (

            dataframe["High"]

            + dataframe["Low"]

            + dataframe["Close"]

        ) / 3

    # -------------------------------------------------------

    @staticmethod
    def median_price(

        dataframe,

    ):

        return (

            dataframe["High"]

            + dataframe["Low"]

        ) / 2

    # -------------------------------------------------------

    @staticmethod
    def weighted_close(

        dataframe,

    ):

        return (

            dataframe["High"]

            + dataframe["Low"]

            + dataframe["Close"] * 2

        ) / 4

    # -------------------------------------------------------

    @staticmethod
    def average_price(

        dataframe,

    ):

        return (

            dataframe["Open"]

            + dataframe["High"]

            + dataframe["Low"]

            + dataframe["Close"]

        ) / 4

    # =======================================================
    # RETURNS
    # =======================================================

    @staticmethod
    def simple_returns(

        dataframe,

    ):

        return (

            dataframe["Close"]

            .pct_change()

        )

    # -------------------------------------------------------

    @staticmethod
    def log_returns(

        dataframe,

    ):

        return np.log(

            dataframe["Close"]

            / dataframe["Close"].shift(1)

        )

    # -------------------------------------------------------

    @staticmethod
    def cumulative_returns(

        returns,

    ):

        return (

            1 + returns

        ).cumprod()

    # -------------------------------------------------------

    @staticmethod
    def annualized_return(

        returns,

        periods=252,

    ):

        mean = returns.mean()

        return (

            (1 + mean)

            ** periods

        ) - 1

    # =======================================================
    # VOLUME
    # =======================================================

    @staticmethod
    def dollar_volume(

        dataframe,

    ):

        return (

            dataframe["Close"]

            * dataframe["Volume"]

        )

    # -------------------------------------------------------

    @staticmethod
    def volume_change(

        dataframe,

    ):

        return (

            dataframe["Volume"]

            .pct_change()

        )

    # =======================================================
    # RESAMPLING
    # =======================================================

    @staticmethod
    def resample(

        dataframe,

        frequency="M",

    ):

        return dataframe.resample(

            frequency

        ).last()

    # -------------------------------------------------------

    @staticmethod
    def weekly(

        dataframe,

    ):

        return Preprocessor.resample(

            dataframe,

            "W",

        )

    # -------------------------------------------------------

    @staticmethod
    def monthly(

        dataframe,

    ):

        return Preprocessor.resample(

            dataframe,

            "M",

        )

    # -------------------------------------------------------

    @staticmethod
    def quarterly(

        dataframe,

    ):

        return Preprocessor.resample(

            dataframe,

            "Q",

        )

    # -------------------------------------------------------

    @staticmethod
    def yearly(

        dataframe,

    ):

        return Preprocessor.resample(

            dataframe,

            "Y",

        )
        
    # =======================================================
    # VOLATILITY
    # =======================================================

    @staticmethod
    def true_range(
        dataframe: pd.DataFrame,
    ) -> pd.Series:

        high = dataframe["High"]
        low = dataframe["Low"]
        close = dataframe["Close"]

        tr1 = high - low

        tr2 = (
            high
            - close.shift(1)
        ).abs()

        tr3 = (
            low
            - close.shift(1)
        ).abs()

        return pd.concat(
            [tr1, tr2, tr3],
            axis=1,
        ).max(axis=1)

    # -------------------------------------------------------

    @staticmethod
    def atr(
        dataframe: pd.DataFrame,
        window: int = 14,
    ) -> pd.Series:

        return (
            Preprocessor.true_range(
                dataframe
            )
            .rolling(window)
            .mean()
        )

    # -------------------------------------------------------

    @staticmethod
    def historical_volatility(
        dataframe: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:

        returns = Preprocessor.log_returns(
            dataframe
        )

        return (
            returns
            .rolling(window)
            .std()
            * np.sqrt(252)
        )

    # -------------------------------------------------------

    @staticmethod
    def garman_klass_volatility(
        dataframe: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:

        log_hl = np.log(
            dataframe["High"]
            / dataframe["Low"]
        )

        log_co = np.log(
            dataframe["Close"]
            / dataframe["Open"]
        )

        rs = (
            0.5
            * log_hl.pow(2)
            - (2 * np.log(2) - 1)
            * log_co.pow(2)
        )

        return (
            rs
            .rolling(window)
            .mean()
            .pow(0.5)
        )

    # =======================================================
    # RSI
    # =======================================================

    @staticmethod
    def rsi(
        dataframe: pd.DataFrame,
        window: int = 14,
    ) -> pd.Series:

        delta = (
            dataframe["Close"]
            .diff()
        )

        gain = delta.clip(
            lower=0
        )

        loss = (
            -delta.clip(
                upper=0
            )
        )

        avg_gain = (
            gain
            .rolling(window)
            .mean()
        )

        avg_loss = (
            loss
            .rolling(window)
            .mean()
        )

        rs = avg_gain / avg_loss

        return (
            100
            - (
                100
                / (1 + rs)
            )
        )

    # =======================================================
    # MACD
    # =======================================================

    @staticmethod
    def macd(
        dataframe: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
    ) -> pd.Series:

        ema_fast = (
            dataframe["Close"]
            .ewm(
                span=fast,
                adjust=False,
            )
            .mean()
        )

        ema_slow = (
            dataframe["Close"]
            .ewm(
                span=slow,
                adjust=False,
            )
            .mean()
        )

        return (
            ema_fast
            - ema_slow
        )

    # -------------------------------------------------------

    @staticmethod
    def macd_signal(
        dataframe: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.Series:

        macd = Preprocessor.macd(
            dataframe,
            fast,
            slow,
        )

        return (
            macd
            .ewm(
                span=signal,
                adjust=False,
            )
            .mean()
        )

    # =======================================================
    # BOLLINGER BANDS
    # =======================================================

    @staticmethod
    def bollinger_bands(
        dataframe: pd.DataFrame,
        window: int = 20,
        std: float = 2.0,
    ) -> pd.DataFrame:

        close = dataframe["Close"]

        middle = (
            close
            .rolling(window)
            .mean()
        )

        sigma = (
            close
            .rolling(window)
            .std()
        )

        upper = (
            middle
            + std * sigma
        )

        lower = (
            middle
            - std * sigma
        )

        return pd.DataFrame(
            {
                "BB_Lower": lower,
                "BB_Middle": middle,
                "BB_Upper": upper,
            },
            index=dataframe.index,
        )

    # =======================================================
    # MOVING AVERAGES
    # =======================================================

    @staticmethod
    def sma(
        dataframe: pd.DataFrame,
        window: int,
    ) -> pd.Series:

        return (
            dataframe["Close"]
            .rolling(window)
            .mean()
        )

    # -------------------------------------------------------

    @staticmethod
    def ema(
        dataframe: pd.DataFrame,
        window: int,
    ) -> pd.Series:

        return (
            dataframe["Close"]
            .ewm(
                span=window,
                adjust=False,
            )
            .mean()
        )

    # =======================================================
    # MOMENTUM
    # =======================================================

    @staticmethod
    def momentum(
        dataframe: pd.DataFrame,
        period: int = 20,
    ) -> pd.Series:

        return (
            dataframe["Close"]
            / dataframe["Close"].shift(period)
            - 1
        )

    # -------------------------------------------------------

    @staticmethod
    def rate_of_change(
        dataframe: pd.DataFrame,
        period: int = 20,
    ) -> pd.Series:

        return (
            dataframe["Close"]
            .pct_change(period)
        )

    # -------------------------------------------------------

    @staticmethod
    def rolling_std(
        dataframe: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:

        return (
            dataframe["Close"]
            .rolling(window)
            .std()
        )

    # -------------------------------------------------------

    @staticmethod
    def rolling_mean(
        dataframe: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:

        return (
            dataframe["Close"]
            .rolling(window)
            .mean()
        )
        
    # =======================================================
    # FEATURE ENGINEERING
    # =======================================================

    @staticmethod
    def add_features(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        df["Return"] = Preprocessor.simple_returns(df)
        df["Log_Return"] = Preprocessor.log_returns(df)

        df["Dollar_Volume"] = Preprocessor.dollar_volume(df)

        df["ATR"] = Preprocessor.atr(df)

        df["RSI"] = Preprocessor.rsi(df)

        df["MACD"] = Preprocessor.macd(df)

        bb = Preprocessor.bollinger_bands(df)

        df = df.join(bb)

        df["GK_Volatility"] = (
            Preprocessor.garman_klass_volatility(df)
        )

        df["Historical_Volatility"] = (
            Preprocessor.historical_volatility(df)
        )

        df["Momentum_1M"] = (
            Preprocessor.momentum(df, 21)
        )

        df["Momentum_3M"] = (
            Preprocessor.momentum(df, 63)
        )

        df["Momentum_6M"] = (
            Preprocessor.momentum(df, 126)
        )

        df["Momentum_12M"] = (
            Preprocessor.momentum(df, 252)
        )

        df["SMA20"] = Preprocessor.sma(df, 20)
        df["SMA50"] = Preprocessor.sma(df, 50)
        df["SMA200"] = Preprocessor.sma(df, 200)

        df["EMA20"] = Preprocessor.ema(df, 20)
        df["EMA50"] = Preprocessor.ema(df, 50)

        return df

    # =======================================================
    # NORMALIZATION
    # =======================================================

    @staticmethod
    def zscore(
        series: pd.Series,
    ) -> pd.Series:

        return (
            series - series.mean()
        ) / series.std()

    # -------------------------------------------------------

    @staticmethod
    def minmax(
        series: pd.Series,
    ) -> pd.Series:

        return (

            series - series.min()

        ) / (

            series.max() - series.min()

        )

    # -------------------------------------------------------

    @staticmethod
    def normalize_dataframe(
        dataframe: pd.DataFrame,
        columns: Iterable[str],
    ) -> pd.DataFrame:

        df = dataframe.copy()

        for column in columns:

            if column in df.columns:

                df[column] = Preprocessor.zscore(
                    df[column]
                )

        return df

    # =======================================================
    # ROLLING FEATURES
    # =======================================================

    @staticmethod
    def rolling_mean_feature(
        dataframe: pd.DataFrame,
        column: str,
        window: int,
    ) -> pd.Series:

        return (
            dataframe[column]
            .rolling(window)
            .mean()
        )

    # -------------------------------------------------------

    @staticmethod
    def rolling_std_feature(
        dataframe: pd.DataFrame,
        column: str,
        window: int,
    ) -> pd.Series:

        return (
            dataframe[column]
            .rolling(window)
            .std()
        )

    # -------------------------------------------------------

    @staticmethod
    def rolling_max_feature(
        dataframe: pd.DataFrame,
        column: str,
        window: int,
    ) -> pd.Series:

        return (
            dataframe[column]
            .rolling(window)
            .max()
        )

    # -------------------------------------------------------

    @staticmethod
    def rolling_min_feature(
        dataframe: pd.DataFrame,
        column: str,
        window: int,
    ) -> pd.Series:

        return (
            dataframe[column]
            .rolling(window)
            .min()
        )

    # =======================================================
    # MONTHLY FEATURES
    # =======================================================

    @staticmethod
    def monthly_returns(
        dataframe: pd.DataFrame,
    ) -> pd.Series:

        monthly = (
            dataframe["Close"]
            .resample("M")
            .last()
        )

        return monthly.pct_change()

    # -------------------------------------------------------

    @staticmethod
    def monthly_dollar_volume(
        dataframe: pd.DataFrame,
    ) -> pd.Series:

        return (
            (
                dataframe["Close"]
                * dataframe["Volume"]
            )
            .resample("M")
            .mean()
        )

    # =======================================================
    # FINAL CLEANING
    # =======================================================

    @staticmethod
    def finalize(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        df.replace(

            [np.inf, -np.inf],

            np.nan,

            inplace=True,

        )

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:

            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.loc[
            ~df["Close"].isna()
            & ~df["Open"].isna()
            & ~df["High"].isna()
            & ~df["Low"].isna()
            & ~df["Volume"].isna()
        ].copy()

        if df.empty:

            return dataframe.copy()

        return df

    # =======================================================
    # PIPELINE
    # =======================================================

    @classmethod
    def process(
        cls,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        cls.validate(dataframe)

        df = cls.clean(dataframe)

        df = cls.add_features(df)

        df = cls.finalize(df)

        return df

    # =======================================================
    # INFORMATION
    # =======================================================

    @staticmethod
    def info(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            {
                "Rows": [len(dataframe)],
                "Columns": [len(dataframe.columns)],
                "Missing": [int(dataframe.isna().sum().sum())],
                "Start": [dataframe.index.min()],
                "End": [dataframe.index.max()],
            }
        )

    # =======================================================
    # REPRESENTATION
    # =======================================================

    def __repr__(self):

        return "Preprocessor()"
    

"""
models/garch.py
=============================================================
Rolling GARCH Volatility Forecasting Engine

Author : Arnav
Python : 3.12+
=============================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arch import arch_model


# ============================================================
# RESULT
# ============================================================


@dataclass(slots=True)
class GARCHResult:
    data: pd.DataFrame
    forecasts: pd.Series
    realized_returns: pd.Series


# ============================================================
# GARCH ENGINE
# ============================================================


class GARCHModel:

    def __init__(
        self,
        window: int = 252,
        p: int = 1,
        q: int = 3,
        mean: str = "Zero",
        vol: str = "GARCH",
        dist: str = "normal",
        scale: float = 100.0,
    ) -> None:

        self.window = window
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist
        self.scale = scale

    # --------------------------------------------------------

    @staticmethod
    def _validate(data: pd.DataFrame):

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be DataFrame.")

        if "Close" not in data.columns:
            raise ValueError("'Close' column not found.")

        if len(data) < 260:
            raise ValueError(
                "Not enough observations for GARCH."
            )

    # --------------------------------------------------------

    def _returns(
        self,
        prices: pd.Series,
    ) -> pd.Series:

        returns = (
            self.scale
            * np.log(prices)
            .diff()
        )

        return returns.dropna()

    def _coerce_to_frame(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame:
        if isinstance(data, pd.Series):
            return pd.DataFrame(data)
        if isinstance(data, pd.DataFrame):
            return data
        raise TypeError("Input must be DataFrame or Series.")

    # --------------------------------------------------------

    def fit(
        self,
        daily_data: pd.DataFrame | pd.Series,
    ) -> GARCHResult:

        df = self._coerce_to_frame(daily_data)
        self._validate(df)

        df = df.copy()

        returns = self._returns(
            df["Close"]
        )

        forecasts = []

        dates = []

        for i in range(
            self.window,
            len(returns),
        ):

            train = returns.iloc[
                i-self.window:i
            ]

            model = arch_model(
                train,
                vol=self.vol,
                p=self.p,
                q=self.q,
                mean=self.mean,
                dist=self.dist,
            )

            fitted = model.fit(
                disp="off",
                show_warning=False,
            )

            variance = (
                fitted
                .forecast(horizon=1)
                .variance
                .iloc[-1, 0]
            )

            forecasts.append(
                np.sqrt(variance)
            )

            dates.append(
                returns.index[i]
            )

        forecast_series = pd.Series(
            forecasts,
            index=dates,
            name="Predicted_Volatility",
        )

        output = df.loc[
            forecast_series.index
        ].copy()

        output["Return"] = returns.loc[
            forecast_series.index
        ]

        output[
            "Predicted_Volatility"
        ] = forecast_series

        return GARCHResult(
            data=output,
            forecasts=forecast_series,
            realized_returns=returns,
        )

    # --------------------------------------------------------

    def merge_intraday(
        self,
        intraday_data: pd.DataFrame,
        forecast_data: pd.DataFrame,
    ) -> pd.DataFrame:

        if "Predicted_Volatility" not in forecast_data.columns:
            raise ValueError(
                "Forecast column missing."
            )

        intraday = intraday_data.copy()

        daily = forecast_data.copy()

        intraday.index = (
            pd.to_datetime(
                intraday.index
            )
            .tz_localize(None)
        )

        daily.index = (
            pd.to_datetime(
                daily.index
            )
            .tz_localize(None)
        )

        intraday["date"] = (
            intraday.index.date
        )

        daily["date"] = (
            daily.index.date
        )

        merged = intraday.merge(
            daily[
                [
                    "date",
                    "Predicted_Volatility",
                ]
            ],
            on="date",
            how="left",
        )

        merged.index = intraday.index

        merged.drop(
            columns="date",
            inplace=True,
        )

        return merged

    # --------------------------------------------------------

    @staticmethod
    def classify_regime(
        dataframe: pd.DataFrame,
        lookback: int = 20,
    ) -> pd.DataFrame:

        if "Predicted_Volatility" not in dataframe.columns:
            raise ValueError(
                "Predicted_Volatility missing."
            )

        df = dataframe.copy()

        threshold = (
            df[
                "Predicted_Volatility"
            ]
            .rolling(lookback)
            .median()
        )

        df["Volatility_Threshold"] = threshold

        df["High_Volatility"] = (
            df["Predicted_Volatility"]
            > threshold
        )

        return df

    # --------------------------------------------------------

    @staticmethod
    def realized_volatility(
        dataframe: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:

        returns = (
            np.log(
                dataframe["Close"]
            )
            .diff()
        )

        realized = (
            returns
            .rolling(window)
            .std()
            * np.sqrt(252)
        )

        realized.name = "Realized_Volatility"

        return realized

    # --------------------------------------------------------

    @staticmethod
    def forecast_error(
        dataframe: pd.DataFrame,
    ) -> pd.Series:

        if (
            "Predicted_Volatility"
            not in dataframe.columns
        ):
            raise ValueError(
                "Prediction column missing."
            )

        realized = GARCHModel.realized_volatility(
            dataframe
        )

        error = (
            realized
            - dataframe[
                "Predicted_Volatility"
            ]
        )

        error.name = "Forecast_Error"

        return error

    # --------------------------------------------------------

    @staticmethod
    def summary(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            {
                "Mean Forecast": [
                    dataframe[
                        "Predicted_Volatility"
                    ].mean()
                ],
                "Max Forecast": [
                    dataframe[
                        "Predicted_Volatility"
                    ].max()
                ],
                "Min Forecast": [
                    dataframe[
                        "Predicted_Volatility"
                    ].min()
                ],
                "Std Forecast": [
                    dataframe[
                        "Predicted_Volatility"
                    ].std()
                ],
            }
        )
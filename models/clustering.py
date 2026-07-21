"""
models/clustering.py
=============================================================
Monthly Cross-Sectional Stock Clustering

Author : Arnav
Python : 3.12+
=============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

DEFAULT_FEATURES = [
    "garman_klass_vol",
    "rsi",
    "bb_low",
    "bb_mid",
    "bb_high",
    "atr",
    "macd",
    "return_1m",
    "return_2m",
    "return_3m",
    "return_6m",
    "return_9m",
    "return_12m",
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA",
]


# ============================================================
# RESULT
# ============================================================

@dataclass(slots=True)
class ClusterResult:
    labels: pd.DataFrame
    model: KMeans
    scaler: StandardScaler
    pca: PCA | None = None


# ============================================================
# CLUSTERER
# ============================================================

class StockClusterer:

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
        n_init: int = 25,
    ) -> None:

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

        self.scaler = StandardScaler()

        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
        )

        self.pca: PCA | None = None

    # --------------------------------------------------------

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:

        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError(
                "Input DataFrame must have MultiIndex "
                "(date, ticker)."
            )

        if "date" not in df.index.names:
            raise ValueError(
                "MultiIndex must contain 'date'."
            )

        if "ticker" not in df.index.names:
            raise ValueError(
                "MultiIndex must contain 'ticker'."
            )

    # --------------------------------------------------------

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:

        df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return df.dropna()

    # --------------------------------------------------------

    def fit_predict(
        self,
        dataframe: pd.DataFrame,
        features: Iterable[str] = DEFAULT_FEATURES,
    ) -> ClusterResult:

        self._validate(dataframe)

        features = list(features)

        missing = [
            feature
            for feature in features
            if feature not in dataframe.columns
        ]

        if missing:
            raise KeyError(
                f"Missing features : {missing}"
            )

        working = dataframe.copy()

        feature_frame = working[features]

        feature_frame = self._clean(feature_frame)

        scaled = self.scaler.fit_transform(
            feature_frame
        )

        labels = self.model.fit_predict(
            scaled
        )

        output = working.loc[
            feature_frame.index
        ].copy()

        output["cluster"] = labels

        return ClusterResult(
            labels=output,
            model=self.model,
            scaler=self.scaler,
        )

    # --------------------------------------------------------

    def monthly_clusters(
        self,
        dataframe: pd.DataFrame,
        features: Iterable[str] = DEFAULT_FEATURES,
    ) -> pd.DataFrame:

        self._validate(dataframe)

        results = []

        for date, frame in dataframe.groupby(
            level="date",
            sort=True,
        ):

            if len(frame) < self.n_clusters:
                continue

            clustered = self.fit_predict(
                frame,
                features,
            ).labels

            results.append(clustered)

        if not results:
            raise ValueError(
                "No monthly clustering results generated."
            )

        return pd.concat(results).sort_index()

    # --------------------------------------------------------

    def cluster_members(
        self,
        clustered_df: pd.DataFrame,
        cluster: int,
    ) -> dict[str, list[str]]:

        if "cluster" not in clustered_df.columns:
            raise ValueError(
                "'cluster' column not found."
            )

        members: dict[str, list[str]] = {}

        grouped = clustered_df.groupby(
            level="date"
        )

        for date, frame in grouped:

            tickers = (
                frame.loc[
                    frame["cluster"] == cluster
                ]
                .index
                .get_level_values("ticker")
                .tolist()
            )

            members[
                str(pd.Timestamp(date).date())
            ] = tickers

        return members

    # --------------------------------------------------------

    def pca_projection(
        self,
        clustered_df: pd.DataFrame,
        features: Iterable[str] = DEFAULT_FEATURES,
        n_components: int = 2,
    ) -> pd.DataFrame:

        features = list(features)

        frame = clustered_df[
            features
        ].copy()

        frame = self._clean(frame)

        scaled = self.scaler.fit_transform(
            frame
        )

        self.pca = PCA(
            n_components=n_components,
            random_state=self.random_state,
        )

        projected = self.pca.fit_transform(
            scaled
        )

        projection = pd.DataFrame(
            projected,
            index=frame.index,
            columns=[
                f"PC{i+1}"
                for i in range(n_components)
            ],
        )

        if "cluster" in clustered_df.columns:

            projection["cluster"] = (
                clustered_df.loc[
                    projection.index,
                    "cluster",
                ]
            )

        return projection

    # --------------------------------------------------------

    @staticmethod
    def cluster_statistics(
        clustered_df: pd.DataFrame,
    ) -> pd.DataFrame:

        if "cluster" not in clustered_df.columns:
            raise ValueError(
                "'cluster' column not found."
            )

        stats = (
            clustered_df
            .groupby("cluster")
            .agg(
                Count=("cluster", "size"),
            )
            .sort_index()
        )

        return stats

    # --------------------------------------------------------

    @staticmethod
    def monthly_cluster_counts(
        clustered_df: pd.DataFrame,
    ) -> pd.DataFrame:

        if "cluster" not in clustered_df.columns:
            raise ValueError(
                "'cluster' column not found."
            )

        return (
            clustered_df
            .groupby(
                [
                    clustered_df.index.get_level_values("date"),
                    "cluster",
                ]
            )
            .size()
            .unstack(fill_value=0)
        )
"""
data/downloader.py
=========================================================
Production Market Data Downloader

Author : Arnav
Python : 3.12+
=========================================================
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from config import CONFIG


logger = logging.getLogger(__name__)


# ============================================================
# DOWNLOADER
# ============================================================


class Downloader:

    """
    Production downloader supporting

    ✓ Daily data
    ✓ Intraday data
    ✓ Batch downloading
    ✓ Parallel execution
    ✓ Automatic retry
    ✓ Cache
    ✓ Incremental updates
    """

    VALID_INTERVALS = {

        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",

        "1h",

        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo"

    }

    VALID_PERIODS = {

        "1d",
        "5d",

        "1mo",
        "3mo",
        "6mo",

        "1y",
        "2y",
        "3y",
        "5y",

        "10y",
        "max",

        "60d"

    }

    # --------------------------------------------------------

    def __init__(

        self,

        cache_dir: str | Path | None = None,

        max_workers: int = 6,

        retries: int = 3,

        timeout: int = 15,

        pause: float = 1.0,

    ):

        self.cache_dir = Path(

            cache_dir
            or CONFIG.cache_directory

        )

        self.cache_dir.mkdir(

            parents=True,
            exist_ok=True

        )

        self.max_workers = max_workers

        self.retries = retries

        self.timeout = timeout

        self.pause = pause

        self._lock = threading.Lock()

    # =======================================================
    # VALIDATION
    # =======================================================

    def validate_interval(

        self,

        interval: str,

    ):

        if interval not in self.VALID_INTERVALS:

            raise ValueError(

                f"Unsupported interval : {interval}"

            )

    # -------------------------------------------------------

    def validate_period(

        self,

        period: str,

    ):

        if period not in self.VALID_PERIODS:

            raise ValueError(

                f"Unsupported period : {period}"

            )

    # =======================================================
    # CACHE
    # =======================================================

    def cache_file(

        self,

        ticker: str,

        interval: str,

        period: str,

    ) -> Path:

        filename = (

            f"{ticker}"

            f"_{interval}"

            f"_{period}.parquet"

        )

        return self.cache_dir / filename

    # -------------------------------------------------------

    def cache_exists(

        self,

        ticker,

        interval,

        period,

    ) -> bool:

        return self.cache_file(

            ticker,

            interval,

            period,

        ).exists()

    # -------------------------------------------------------

    def load_cache(

        self,

        ticker,

        interval,

        period,

    ):

        file = self.cache_file(

            ticker,

            interval,

            period,

        )

        if not file.exists():

            return None

        try:

            return pd.read_parquet(file)

        except Exception:

            return None

    # -------------------------------------------------------

    def save_cache(

        self,

        dataframe,

        ticker,

        interval,

        period,

    ):

        file = self.cache_file(

            ticker,

            interval,

            period,

        )

        dataframe.to_parquet(

            file,

            compression="snappy",

        )

    # =======================================================
    # INTERNAL DOWNLOAD
    # =======================================================

    def _download(

        self,

        ticker: str,

        interval: str = "1d",

        period: str | None = "5y",

        auto_adjust: bool = True,

        repair: bool = True,

        progress: bool = False,

        use_cache: bool = True,

        start: str | None = None,

        end: str | None = None,

    ) -> pd.DataFrame:

        self.validate_interval(interval)

        if period is not None:

            self.validate_period(period)

        use_period = period is not None and start is None and end is None

        if use_cache and use_period:

            cached = self.load_cache(

                ticker,

                interval,

                period,

            )

            if cached is not None:

                return cached

        last_exception = None

        for attempt in range(

            self.retries

        ):

            try:

                download_kwargs = {

                    "interval": interval,

                    "auto_adjust": auto_adjust,

                    "repair": repair,

                    "progress": progress,

                    "threads": False,

                    "timeout": self.timeout,

                }

                if use_period:

                    download_kwargs["period"] = period

                else:

                    if start is not None:

                        download_kwargs["start"] = start

                    if end is not None:

                        download_kwargs["end"] = end

                data = yf.download(

                    ticker,

                    **download_kwargs,

                )

                if data.empty:

                    raise RuntimeError(

                        f"No data returned for {ticker}"

                    )

                if isinstance(data, tuple):

                    data = data[0]

                data.columns = [

                    c[0]

                    if isinstance(c, tuple)

                    else c

                    for c in data.columns

                ]

                data.index = pd.to_datetime(

                    data.index

                )

                data.sort_index(

                    inplace=True

                )

                data = data.loc[

                    ~data.index.duplicated()

                ]

                data.dropna(

                    how="all",

                    inplace=True,

                )

                if use_cache and use_period:

                    self.save_cache(

                        data,

                        ticker,

                        interval,

                        period,

                    )

                return data

            except Exception as e:

                last_exception = e

                if "delisted" in str(e).lower() or "not found" in str(e).lower() or "no data returned" in str(e).lower():

                    break

                time.sleep(

                    self.pause

                )

        if last_exception is not None and (

            "delisted" in str(last_exception).lower()

            or "not found" in str(last_exception).lower()

            or "no data returned" in str(last_exception).lower()

        ):

            return pd.DataFrame()

        raise RuntimeError(

            f"{ticker} download failed.\n"

            f"{last_exception}"

        )

    # =======================================================
    # PUBLIC API
    # =======================================================

    def download(

        self,

        symbols: str | Iterable[str],

        interval: str = "1d",

        period: str | None = "5y",

        start: str | None = None,

        end: str | None = None,

        auto_adjust: bool = True,

        repair: bool = True,

        progress: bool = False,

        use_cache: bool = True,

    ) -> dict[str, pd.DataFrame]:

        if isinstance(symbols, str):

            tickers = [symbols]

        elif isinstance(symbols, Iterable):

            tickers = list(dict.fromkeys(symbols))

        else:

            raise TypeError(

                "symbols must be a ticker string or an iterable of tickers."

            )

        if not tickers:

            return {}

        if len(tickers) == 1:

            df = self._download(

                ticker=tickers[0],

                interval=interval,

                period=period,

                auto_adjust=auto_adjust,

                repair=repair,

                progress=progress,

                use_cache=use_cache,

                start=start,

                end=end,

            )

            return {tickers[0]: df}

        return self.download_many(

            tickers=tickers,

            interval=interval,

            period=period or "5y",

            max_workers=self.max_workers,

            use_cache=use_cache,

            start=start,

            end=end,

        )

    # -------------------------------------------------------

    def download_daily(

        self,

        ticker: str,

        period: str = "5y",

    ) -> pd.DataFrame:

        return self._download(

            ticker=ticker,

            interval="1d",

            period=period,

        )

    # -------------------------------------------------------

    def download_intraday(

        self,

        ticker: str,

        interval: str = "5m",

        period: str = "60d",

    ) -> pd.DataFrame:

        return self._download(

            ticker=ticker,

            interval=interval,

            period=period,

        )
        
    # =======================================================
    # MULTI DOWNLOAD
    # =======================================================

    def download_many(

        self,

        tickers: Iterable[str],

        interval: str = "1d",

        period: str = "5y",

        max_workers: int | None = None,

        use_cache: bool = True,

        start: str | None = None,

        end: str | None = None,

    ) -> dict[str, pd.DataFrame]:

        """
        Parallel downloader.

        Returns
        -------
        dict
            {
                ticker: dataframe
            }
        """

        self.validate_interval(interval)
        self.validate_period(period)

        tickers = list(dict.fromkeys(tickers))

        workers = max_workers or self.max_workers

        results: dict[str, pd.DataFrame] = {}

        def worker(symbol: str):

            try:

                df = self._download(

                    ticker=symbol,

                    interval=interval,

                    period=period,

                    use_cache=use_cache,

                    start=start,

                    end=end,

                )

                return symbol, df

            except Exception as e:

                logger.warning(

                    "%s download failed : %s",

                    symbol,

                    e,

                )

                return symbol, None

        with concurrent.futures.ThreadPoolExecutor(

            max_workers=workers,

        ) as executor:

            futures = {

                executor.submit(

                    worker,

                    ticker,

                ): ticker

                for ticker in tickers

            }

            for future in concurrent.futures.as_completed(

                futures

            ):

                ticker, data = future.result()

                if data is not None:

                    results[ticker] = data

        return results

    # -------------------------------------------------------

    def download_nifty50(

        self,

        tickers: Iterable[str],

        period: str = "5y",

    ) -> dict[str, pd.DataFrame]:

        return self.download_many(

            tickers=tickers,

            interval="1d",

            period=period,

        )

    # -------------------------------------------------------

    def download_intraday_many(

        self,

        tickers: Iterable[str],

        interval: str = "5m",

        period: str = "60d",

    ) -> dict[str, pd.DataFrame]:

        return self.download_many(

            tickers=tickers,

            interval=interval,

            period=period,

        )

    # =======================================================
    # MERGE
    # =======================================================

    @staticmethod
    def merge_close_prices(

        data: dict[str, pd.DataFrame],

    ) -> pd.DataFrame:

        frames = []

        for ticker, df in data.items():

            if "Close" not in df.columns:

                continue

            frames.append(

                df["Close"]

                .rename(ticker)

            )

        if not frames:

            raise ValueError(

                "No Close prices available."

            )

        merged = pd.concat(

            frames,

            axis=1,

        )

        merged.sort_index(

            inplace=True

        )

        return merged

    # -------------------------------------------------------

    @staticmethod
    def merge_volume(

        data: dict[str, pd.DataFrame],

    ) -> pd.DataFrame:

        frames = []

        for ticker, df in data.items():

            if "Volume" not in df.columns:

                continue

            frames.append(

                df["Volume"]

                .rename(ticker)

            )

        if not frames:

            raise ValueError(

                "No Volume data available."

            )

        merged = pd.concat(

            frames,

            axis=1,

        )

        merged.sort_index(

            inplace=True

        )

        return merged

    # =======================================================
    # UPDATE
    # =======================================================

    def update(

        self,

        ticker: str,

        interval: str = "1d",

        period: str = "5y",

    ) -> pd.DataFrame:

        """
        Updates cached data.

        If cache exists,
        downloads only missing observations.
        """

        cache = self.load_cache(

            ticker,

            interval,

            period,

        )

        if cache is None:

            return self._download(

                ticker,

                interval,

                period,

            )

        latest = cache.index.max()

        latest = pd.Timestamp(latest)

        latest -= pd.Timedelta(

            days=5

        )

        fresh = yf.download(

            ticker,

            start=latest,

            interval=interval,

            progress=False,

            auto_adjust=True,

            repair=True,

            threads=False,

        )

        if fresh.empty:

            return cache

        fresh.columns = [

            c[0]

            if isinstance(c, tuple)

            else c

            for c in fresh.columns

        ]

        combined = pd.concat(

            [

                cache,

                fresh,

            ]

        )

        combined = combined.loc[

            ~combined.index.duplicated(

                keep="last"

            )

        ]

        combined.sort_index(

            inplace=True

        )

        self.save_cache(

            combined,

            ticker,

            interval,

            period,

        )

        return combined

    # =======================================================
    # CACHE MANAGEMENT
    # =======================================================

    def clear_cache(self):

        files = list(

            self.cache_dir.glob(

                "*.parquet"

            )

        )

        for file in files:

            try:

                file.unlink()

            except Exception:

                pass

        logger.info(

            "Cache cleared."

        )

    # -------------------------------------------------------

    def cache_size(self):

        size = 0

        for file in self.cache_dir.glob(

            "*.parquet"

        ):

            size += file.stat().st_size

        return size

    # -------------------------------------------------------

    def cache_info(self):

        files = list(

            self.cache_dir.glob(

                "*.parquet"

            )

        )

        return pd.DataFrame(

            {

                "File": [

                    f.name

                    for f in files

                ],

                "Size (MB)": [

                    round(

                        f.stat().st_size

                        / 1024
                        / 1024,

                        3,

                    )

                    for f in files

                ],

            }

        )
        
    # =======================================================
    # UTILITIES
    # =======================================================

    @staticmethod
    def available_columns(
        data: dict[str, pd.DataFrame],
    ) -> dict[str, list[str]]:

        return {
            ticker: list(df.columns)
            for ticker, df in data.items()
        }

    # -------------------------------------------------------

    @staticmethod
    def download_summary(
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:

        summary = []

        for ticker, df in data.items():

            if df.empty:

                continue

            summary.append(
                {
                    "Ticker": ticker,
                    "Rows": len(df),
                    "Columns": len(df.columns),
                    "Start": df.index.min(),
                    "End": df.index.max(),
                    "Missing Values": int(df.isna().sum().sum()),
                }
            )

        return (
            pd.DataFrame(summary)
            .sort_values("Ticker")
            .reset_index(drop=True)
        )

    # -------------------------------------------------------

    @staticmethod
    def validate_download(
        dataframe: pd.DataFrame,
    ) -> bool:

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        if dataframe.empty:
            return False

        for col in required:

            if col not in dataframe.columns:
                return False

        if dataframe["Close"].isna().all():
            return False

        return True

    # -------------------------------------------------------

    @staticmethod
    def remove_invalid(
        data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:

        cleaned = {}

        for ticker, df in data.items():

            if Downloader.validate_download(df):

                cleaned[ticker] = df

        return cleaned

    # =======================================================
    # EXPORT
    # =======================================================

    @staticmethod
    def export_csv(
        dataframe: pd.DataFrame,
        filepath: str | Path,
    ) -> None:

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(filepath)

    # -------------------------------------------------------

    @staticmethod
    def export_parquet(
        dataframe: pd.DataFrame,
        filepath: str | Path,
    ) -> None:

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_parquet(
            filepath,
            compression="snappy",
        )

    # -------------------------------------------------------

    @staticmethod
    def load_parquet(
        filepath: str | Path,
    ) -> pd.DataFrame:

        return pd.read_parquet(filepath)

    # =======================================================
    # BENCHMARK
    # =======================================================

    def benchmark(
        self,
        symbol: str = "^NSEI",
        period: str = "10y",
    ) -> pd.DataFrame:

        return self.download_daily(
            ticker=symbol,
            period=period,
        )

    # =======================================================
    # INDEX CONSTITUENTS
    # =======================================================

    def download_universe(
        self,
        universe,
        period: str = "5y",
    ) -> dict[str, pd.DataFrame]:

        if hasattr(universe, "symbols"):

            tickers = universe.symbols

        elif isinstance(universe, (list, tuple, set)):

            tickers = list(universe)

        else:

            raise TypeError(
                "Universe must be iterable or expose 'symbols'."
            )

        return self.download_many(
            tickers=tickers,
            interval="1d",
            period=period,
        )

    # =======================================================
    # PRICE MATRICES
    # =======================================================

    @staticmethod
    def open_prices(
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:

        frames = []

        for ticker, df in data.items():

            if "Open" in df.columns:

                frames.append(
                    df["Open"].rename(ticker)
                )

        return pd.concat(
            frames,
            axis=1,
        ).sort_index()

    # -------------------------------------------------------

    @staticmethod
    def high_prices(
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:

        frames = []

        for ticker, df in data.items():

            if "High" in df.columns:

                frames.append(
                    df["High"].rename(ticker)
                )

        return pd.concat(
            frames,
            axis=1,
        ).sort_index()

    # -------------------------------------------------------

    @staticmethod
    def low_prices(
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:

        frames = []

        for ticker, df in data.items():

            if "Low" in df.columns:

                frames.append(
                    df["Low"].rename(ticker)
                )

        return pd.concat(
            frames,
            axis=1,
        ).sort_index()

    # =======================================================
    # RETURNS
    # =======================================================

    @staticmethod
    def returns(
        close_prices: pd.DataFrame,
        log_returns: bool = False,
    ) -> pd.DataFrame:

        if log_returns:

            returns = np.log(
                close_prices / close_prices.shift(1)
            )

        else:

            returns = close_prices.pct_change()

        return returns.dropna(how="all")

    # =======================================================
    # INFORMATION
    # =======================================================

    def info(self) -> dict:

        return {

            "Cache Directory": str(self.cache_dir),

            "Cached Files": len(
                list(
                    self.cache_dir.glob("*.parquet")
                )
            ),

            "Cache Size (MB)": round(
                self.cache_size()
                / 1024
                / 1024,
                2,
            ),

            "Max Workers": self.max_workers,

            "Retries": self.retries,

            "Timeout": self.timeout,

            "Pause": self.pause,

        }

    # =======================================================
    # REPRESENTATION
    # =======================================================

    def __repr__(self):

        return (

            f"{self.__class__.__name__}("
            f"workers={self.max_workers}, "
            f"cache='{self.cache_dir}', "
            f"retries={self.retries})"

        )

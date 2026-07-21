"""
data/universe.py
=========================================================
Market Universe Manager

Author : Arnav
Python : 3.12+
=========================================================
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# UNIVERSE
# ============================================================


class Universe:

    """
    Handles

    ✓ Nifty Universes
    ✓ Custom Universes
    ✓ Validation
    ✓ Yahoo Symbol Formatting
    """

    # ========================================================
    # NIFTY 50
    # ========================================================

    NIFTY50 = [

        "ADANIENT.NS",
        "ADANIPORTS.NS",
        "APOLLOHOSP.NS",
        "ASIANPAINT.NS",
        "AXISBANK.NS",
        "BAJAJ-AUTO.NS",
        "BAJFINANCE.NS",
        "BAJAJFINSV.NS",
        "BEL.NS",
        "BHARTIARTL.NS",
        "CIPLA.NS",
        "COALINDIA.NS",
        "DRREDDY.NS",
        "EICHERMOT.NS",
        "ETERNAL.NS",
        "GRASIM.NS",
        "HCLTECH.NS",
        "HDFCBANK.NS",
        "HDFCLIFE.NS",
        "HEROMOTOCO.NS",
        "HINDALCO.NS",
        "HINDUNILVR.NS",
        "ICICIBANK.NS",
        "INDUSINDBK.NS",
        "INFY.NS",
        "ITC.NS",
        "JIOFIN.NS",
        "JSWSTEEL.NS",
        "KOTAKBANK.NS",
        "LT.NS",
        "M&M.NS",
        "MARUTI.NS",
        "NESTLEIND.NS",
        "NTPC.NS",
        "ONGC.NS",
        "POWERGRID.NS",
        "RELIANCE.NS",
        "SBILIFE.NS",
        "SBIN.NS",
        "SHRIRAMFIN.NS",
        "SUNPHARMA.NS",
        "TATACONSUM.NS",
        "TATASTEEL.NS",
        "TCS.NS",
        "TECHM.NS",
        "TITAN.NS",
        "TRENT.NS",
        "ULTRACEMCO.NS",
        "WIPRO.NS",

    ]

    # ========================================================
    # BENCHMARK
    # ========================================================

    BENCHMARK = "^NSEI"

    BANKNIFTY = "^NSEBANK"

    FINNIFTY = "^NSEFIN"

    MIDCAP = "^NSEMDCP50"

    # ========================================================

    def __init__(

        self,

        symbols=None,

    ):

        if symbols is None:

            symbols = self.NIFTY50.copy()

        self.symbols = sorted(

            list(

                dict.fromkeys(symbols)

            )

        )

    # ========================================================
    # BASIC
    # ========================================================

    def __len__(self):

        return len(

            self.symbols

        )

    def __iter__(self):

        return iter(

            self.symbols

        )

    def __contains__(

        self,

        symbol,

    ):

        return symbol in self.symbols

    def __repr__(self):

        return (

            f"Universe({len(self)} symbols)"

        )

    # ========================================================
    # MODIFY
    # ========================================================

    def add(

        self,

        symbol,

    ):

        symbol = self.normalize(symbol)

        if symbol not in self.symbols:

            self.symbols.append(symbol)

            self.symbols.sort()

    # --------------------------------------------------------

    def remove(

        self,

        symbol,

    ):

        symbol = self.normalize(symbol)

        if symbol in self.symbols:

            self.symbols.remove(symbol)

    # --------------------------------------------------------

    def clear(self):

        self.symbols.clear()

    # --------------------------------------------------------

    def copy(self):

        return Universe(

            self.symbols.copy()

        )

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def normalize(

        symbol,

    ):

        symbol = symbol.upper().strip()

        if not symbol.endswith(".NS"):

            symbol += ".NS"

        return symbol

    # --------------------------------------------------------

    @classmethod
    def validate(

        cls,

        symbol,

    ):

        symbol = cls.normalize(symbol)

        return symbol.endswith(".NS")

    # --------------------------------------------------------

    def validate_all(self):

        return all(

            self.validate(symbol)

            for symbol in self.symbols

        )

    # ========================================================
    # GETTERS
    # ========================================================

    def list(self):

        return self.symbols.copy()

    # --------------------------------------------------------

    def dataframe(self):

        return pd.DataFrame(

            {

                "Ticker": self.symbols

            }

        )

    # --------------------------------------------------------

    def benchmark(self):

        return self.BENCHMARK

    # --------------------------------------------------------

    def banknifty(self):

        return self.BANKNIFTY

    # --------------------------------------------------------

    def finnifty(self):

        return self.FINNIFTY

    # --------------------------------------------------------

    def midcap(self):

        return self.MIDCAP

    # ========================================================
    # SEARCH
    # ========================================================

    def search(

        self,

        text,

    ):

        text = text.upper()

        return [

            symbol

            for symbol in self.symbols

            if text in symbol

        ]

    # --------------------------------------------------------

    def startswith(

        self,

        prefix,

    ):

        prefix = prefix.upper()

        return [

            s

            for s in self.symbols

            if s.startswith(prefix)

        ]

    # --------------------------------------------------------

    def endswith(

        self,

        suffix,

    ):

        suffix = suffix.upper()

        return [

            s

            for s in self.symbols

            if s.endswith(suffix)

        ]

    # ========================================================
    # FILTERS
    # ========================================================

    def filter(
        self,
        symbols: list[str],
    ):

        symbols = {
            self.normalize(s)
            for s in symbols
        }

        return Universe(
            [
                s
                for s in self.symbols
                if s in symbols
            ]
        )

    # --------------------------------------------------------

    def exclude(
        self,
        symbols: list[str],
    ):

        symbols = {
            self.normalize(s)
            for s in symbols
        }

        return Universe(
            [
                s
                for s in self.symbols
                if s not in symbols
            ]
        )

    # ========================================================
    # IMPORT / EXPORT
    # ========================================================

    def save_csv(
        self,
        path: str | Path,
    ):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            {
                "Ticker": self.symbols
            }
        ).to_csv(
            path,
            index=False,
        )

    # --------------------------------------------------------

    @classmethod
    def load_csv(
        cls,
        path: str | Path,
    ):

        path = Path(path)

        df = pd.read_csv(path)

        column = df.columns[0]

        symbols = (
            df[column]
            .dropna()
            .astype(str)
            .tolist()
        )

        return cls(symbols)

    # ========================================================
    # CLASS METHODS
    # ========================================================

    @classmethod
    def nifty50(cls):

        return cls(
            cls.NIFTY50.copy()
        )

    # --------------------------------------------------------

    @classmethod
    def from_list(
        cls,
        symbols: list[str],
    ):

        return cls(symbols)

    # --------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        column: str = "Ticker",
    ):

        if column not in dataframe.columns:

            raise ValueError(
                f"{column} not found."
            )

        return cls(
            dataframe[column]
            .astype(str)
            .tolist()
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    def info(self):

        return {

            "Universe": "NSE",

            "Benchmark": self.BENCHMARK,

            "Bank Nifty": self.BANKNIFTY,

            "Fin Nifty": self.FINNIFTY,

            "Midcap": self.MIDCAP,

            "Total Symbols": len(
                self.symbols
            ),

        }

    # --------------------------------------------------------

    def head(
        self,
        n: int = 10,
    ):

        return self.symbols[:n]

    # --------------------------------------------------------

    def tail(
        self,
        n: int = 10,
    ):

        return self.symbols[-n:]

    # --------------------------------------------------------

    def sample(
        self,
        n: int = 5,
        random_state: int = 42,
    ):

        return (
            pd.Series(self.symbols)
            .sample(
                n=min(n, len(self.symbols)),
                random_state=random_state,
            )
            .tolist()
        )

    # ========================================================
    # CONVERSIONS
    # ========================================================

    def to_series(self):

        return pd.Series(
            self.symbols,
            name="Ticker",
        )

    # --------------------------------------------------------

    def to_dataframe(self):

        return pd.DataFrame(
            {
                "Ticker": self.symbols
            }
        )

    # --------------------------------------------------------

    def to_list(self):

        return self.symbols.copy()

    # ========================================================
    # OPERATORS
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        return self.symbols[index]

    # --------------------------------------------------------

    def __add__(
        self,
        other,
    ):

        if not isinstance(
            other,
            Universe,
        ):

            raise TypeError(
                "Can only add Universe objects."
            )

        symbols = sorted(
            set(self.symbols)
            | set(other.symbols)
        )

        return Universe(symbols)

    # --------------------------------------------------------

    def __sub__(
        self,
        other,
    ):

        if not isinstance(
            other,
            Universe,
        ):

            raise TypeError(
                "Can only subtract Universe objects."
            )

        symbols = sorted(
            set(self.symbols)
            - set(other.symbols)
        )

        return Universe(symbols)
